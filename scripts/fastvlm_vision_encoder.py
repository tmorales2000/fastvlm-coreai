"""
fastvlm_vision_encoder.py — Re-authored FastVLM vision encoder for Core AI export.

Strategy: load the existing fastvithd model from timm (inference_mode=True),
then patch two components for ANE compatibility:
  1. LayerNormChannel: replace manual mean/pow/sqrt decomposition with nn.LayerNorm
  2. MHSA: replace raw scaled dot product attention with SDPA composite op

The weights are already in inference mode in the checkpoint — all reparameterization
(MobileOneBlock, RepMixer, ReparamLargeKernelConv, RepCPE) has already happened.
Each of these is just a Conv2d with bias in the checkpoint.

Weight key prefix: model.vision_tower.vision_tower.model.*
Confirmed from discovery output — doubly nested vision_tower.

Output: [B, H*W, C] patch embeddings (reshaped from [B, C, H, W] conv output)
"""

import glob
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch.composite_ops import SDPA
from safetensors import safe_open
from functools import partial

# Add weights directory to path so llava_qwen.py can be imported for fastvithd registration
# (timm needs the @register_model decorator to have run)


# ─── ANE-compatible replacements ─────────────────────────────────────────────


class ANELayerNorm(nn.Module):
    """
    Drop-in replacement for LayerNormChannel that uses F.layer_norm.

    Holds weight and bias as direct parameters (same names as LayerNormChannel)
    so checkpoint key paths match without renaming. Uses F.layer_norm with
    [B,C,H,W] -> permute -> normalize -> permute back pattern.

    F.layer_norm is recognized by the decomp table as the layer_norm composite
    op. Do NOT wrap in nn.LayerNorm — that adds a .norm. prefix to the key
    paths causing a mismatch with the checkpoint.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        # Direct parameters — same key names as LayerNormChannel (weight, bias)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> permute to [B, H, W, C] for layer_norm -> back
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, [x.shape[-1]], self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ANEAttention(nn.Module):
    """
    Drop-in replacement for MHSA that uses SDPA composite op.

    Original MHSA computes attention manually with q@k.T * scale.
    We replace with SDPA composite op so Core AI can recognize and
    optimize it as 'scaled_dot_product_attention'.

    Input: [B, C, H, W] — flattened to [B, N, C] for attention, then reshaped back.
    """

    def __init__(self, dim: int, head_dim: int = 32) -> None:
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = dim // head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.sdpa = SDPA(is_causal=False)  # vision attention is not causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W

        # Flatten spatial dims: [B, C, H, W] -> [B, N, C]
        x_flat = x.flatten(2).transpose(1, 2)

        # QKV projection
        qkv = (
            self.qkv(x_flat)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B, num_heads, N, head_dim]

        # SDPA composite op — recognized by Core AI compiler
        out = self.sdpa(q, k, v)  # [B, num_heads, N, head_dim]

        # Reshape back
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        # Restore spatial dims: [B, N, C] -> [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


# ─── Patch functions ──────────────────────────────────────────────────────────


def _patch_layernorm(model: nn.Module, LayerNormChannel) -> None:
    """Replace all LayerNormChannel instances with ANELayerNorm in-place."""
    for name, module in model.named_modules():
        if isinstance(module, LayerNormChannel):
            # Navigate to parent and replace
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacement = ANELayerNorm(
                num_features=module.weight.shape[0],
                eps=module.eps,
            )
            # Copy weights directly — ANELayerNorm holds weight/bias at top level
            replacement.weight = nn.Parameter(module.weight.clone())
            replacement.bias = nn.Parameter(module.bias.clone())
            setattr(parent, parts[-1], replacement)


def _patch_attention(model: nn.Module, MHSA) -> None:
    """Replace all MHSA instances with ANEAttention in-place."""
    for name, module in model.named_modules():
        if isinstance(module, MHSA):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            dim = module.qkv.in_features
            replacement = ANEAttention(dim=dim, head_dim=module.head_dim)
            # Copy QKV and proj weights
            replacement.qkv.weight = nn.Parameter(module.qkv.weight.clone())
            replacement.proj.weight = nn.Parameter(module.proj.weight.clone())
            replacement.proj.bias = nn.Parameter(module.proj.bias.clone())
            setattr(parent, parts[-1], replacement)


# ─── Vision encoder wrapper ───────────────────────────────────────────────────


class FastVLMVisionEncoder(nn.Module):
    """
    FastVLM vision encoder (FastViTHD) re-authored for Core AI export.

    Loads the fastvithd timm model in inference_mode=True, patches
    LayerNormChannel -> ANELayerNorm and MHSA -> ANEAttention, then
    wraps the output to produce [B, H*W, C] patch embeddings.

    The head (GlobalPool2D linear projection) is bypassed — we use the
    conv_exp output (image_embeddings) directly, reshaped to sequence form.
    This matches MobileCLIPVisionTower.feature_select() in llava_qwen.py.
    """

    def __init__(self, weights_dir: str) -> None:
        super().__init__()

        # Import llava_qwen to get FastViT, LayerNormChannel, MHSA classes
        sys.path.insert(0, weights_dir)
        try:
            import llava_qwen
        finally:
            sys.path.pop(0)

        # Instantiate FastViTHD directly — create_model via timm returns empty model
        # Config matches fastvithd() function in llava_qwen.py lines 1493-1519
        self.model = llava_qwen.FastViT(
            layers=[2, 12, 24, 4, 2],
            embed_dims=[96, 192, 384, 768, 1536],
            mlp_ratios=[4, 4, 4, 4, 4],
            downsamples=[True, True, True, True, True],
            token_mixers=["repmixer", "repmixer", "repmixer", "attention", "attention"],
            norm_layer=llava_qwen.LayerNormChannel,
            pos_embs=[
                None, None, None,
                # RepCPE stages confirmed from checkpoint: network.6 (768ch), network.9 (1536ch)
                llava_qwen.RepCPE,
                llava_qwen.RepCPE,
            ],
            inference_mode=True,
        )

        # Patch for ANE compatibility — pass classes to avoid re-importing llava_qwen
        _patch_layernorm(self.model, llava_qwen.LayerNormChannel)
        _patch_attention(self.model, llava_qwen.MHSA)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W] float16

        Returns:
            image_features: [B, H*W, C] float16 patch embeddings
        """
        # Run backbone + conv_exp, skip head (GlobalPool2D)
        x = self.model.forward_embeddings(pixel_values)    # patch_embed
        x = self.model.forward_tokens(x)                   # network stages
        x = self.model.conv_exp(x)                         # [B, C, H, W]

        # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)        # [B, H*W, C]
        return x

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMVisionEncoder":
        """Load vision encoder weights from SafeTensors checkpoint."""
        model = cls(weights_dir).to(dtype=torch.float16)
        weights = _load_vision_weights(weights_dir)
        missing, unexpected = model.model.load_state_dict(weights, assign=True, strict=False)
        # head.proj is excluded (not used in forward pass) — expected missing
        unexpected_real = [k for k in unexpected if "head" not in k and ".bn." not in k and "num_batches_tracked" not in k]
        missing_real = [k for k in missing if "head" not in k]
        if unexpected_real:
            raise RuntimeError(f"Unexpected keys: {unexpected_real[:5]}")
        if missing_real:
            raise RuntimeError(f"Missing keys: {missing_real[:5]}")
        return model


# ─── Weight loading ───────────────────────────────────────────────────────────

# Vision tower prefix confirmed from discovery output (doubly nested)
_VISION_PREFIX = "model.vision_tower.vision_tower.model."


def _load_vision_weights(
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Load vision tower weights, stripping the doubly-nested prefix.

    Raw key: model.vision_tower.vision_tower.model.network.0.0.convffn.conv.conv.weight
    Stripped: network.0.0.convffn.conv.conv.weight
    """
    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    result = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith(_VISION_PREFIX):
                    continue
                stripped = key.removeprefix(_VISION_PREFIX)
                t = f.get_tensor(key)
                if t.dtype != dtype:
                    t = t.to(dtype)
                result[stripped] = t
    return result
