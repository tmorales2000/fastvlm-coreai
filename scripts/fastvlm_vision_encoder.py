"""
fastvlm_vision_encoder.py — Re-authored FastVLM vision encoder for Core AI export.

WHAT THIS FILE IS
-----------------
FastVLM's vision encoder is FastViTHD (the "fastvithd" model registered with
timm inside llava_qwen.py). This file re-authors it for Core AI export by
loading the architecture directly (not via timm's registry, which returns an
empty model — see note below) and patching two components for ANE compatibility.

WHY RE-AUTHORING IS NEEDED
----------------------------
1. LayerNormChannel (original) decomposes layer norm manually via mean/pow/sqrt
   on a [B,C,H,W] tensor normalizing over C. This manual decomposition prevents
   the ANE compiler from recognizing it as the layer_norm composite op.
   ANELayerNorm replaces it with F.layer_norm after a permute to [B,H,W,C].

2. MHSA (original) computes attention manually as q@k.T * scale, softmax, @v.
   This prevents the ANE compiler from recognizing it as scaled_dot_product_attention.
   ANEAttention replaces it with the SDPA composite op.

The weights are already in INFERENCE MODE in the checkpoint — all
reparameterization (MobileOneBlock, RepMixer, ReparamLargeKernelConv, RepCPE)
has already happened during FastVLM's training pipeline. Each of these blocks
is just a Conv2d with bias in the checkpoint — there is no multi-branch
training-time structure to reconstruct.

WHY FastViT IS INSTANTIATED DIRECTLY, NOT VIA timm.create_model
-------------------------------------------------------------------
llava_qwen.py registers "fastvithd" with timm via @register_model, but calling
timm.create_model("fastvithd") returns an EMPTY model (zero parameters) in this
environment — confirmed empirically. The cause was never fully isolated, but
direct instantiation of llava_qwen.FastViT(...) with the exact config from the
fastvithd() factory function (lines 1493-1519) works correctly and produces a
model with the expected parameter count. This file uses that approach.

CONFIG (confirmed from llava_qwen.py fastvithd() factory + checkpoint inspection)
-------------------------------------------------------------------------------
  layers       = [2, 12, 24, 4, 2]            # blocks per stage
  embed_dims   = [96, 192, 384, 768, 1536]     # channels per stage
  mlp_ratios   = [4, 4, 4, 4, 4]
  downsamples  = [True, True, True, True, True]
  token_mixers = [repmixer, repmixer, repmixer, attention, attention]
  pos_embs     = [None, None, None, RepCPE, RepCPE]   # pass CLASS, not instance —
                   FastViT.__init__ calls pos_embs[i](embed_dims[i], embed_dims[i],
                   inference_mode=...) itself; passing an instance causes a
                   "got multiple values for inference_mode" TypeError.

NETWORK STAGE MAPPING (checkpoint has 11 stages 0-10, model module has 9: 0-8)
-------------------------------------------------------------------------------
  checkpoint stage | content              | model module index
  -----------------|----------------------|--------------------
  network.0        | RepMixerBlock x2     | network.0
  network.1        | PatchEmbed           | network.1
  network.2        | RepMixerBlock x12    | network.2
  network.3        | PatchEmbed           | network.3
  network.4        | RepMixerBlock x24    | network.4
  network.5        | PatchEmbed           | network.5
  network.6        | RepCPE (768ch)       | network.6  (pos_embs[3])
  network.7        | AttentionBlock x4    | network.7
  network.8        | PatchEmbed           | network.8
  network.9        | RepCPE (1536ch)      | network.9  (pos_embs[4])
  network.10       | AttentionBlock x2    | network.10

The RepCPE stages are inserted into the model's network ModuleList by FastViT's
own __init__ when pos_embs[i] is not None — they are not something this file
adds manually. Confirmed by inspecting model.network children after construction.

IMAGE SIZE
----------
1024x1024, NOT 336. Derived from config.mm_vision_tower (e.g. "mobileclip_l_1024")
by parsing the trailing integer — confirmed in MobileCLIPVisionTower.__init__:
  self.input_image_size = int(vision_tower.split("_")[-1])

FORWARD PATH (matches MobileCLIPVisionTower.feature_select() in llava_qwen.py)
-------------------------------------------------------------------------------
  image_forward_outs = vision_tower(images, return_image_embeddings=True)
  image_features = image_forward_outs["image_embeddings"]   # [B, C, H, W]
  B, C, H, W = image_features.shape
  image_features = image_features.reshape(B, C, H*W).transpose(1, 2)  # [B, H*W, C]

This file's forward() reproduces this exactly: forward_embeddings -> forward_tokens
-> conv_exp (equivalent to "image_embeddings") -> reshape -> transpose.
The head (GlobalPool2D classification head) is never called — FastVLM does not
use it, matching feature_select() which reads conv_exp's output directly.

WEIGHT KEY MAPPING
------------------
Checkpoint prefix: model.vision_tower.vision_tower.model.*  (doubly nested —
confirmed from discover_weights.py output). After stripping this prefix, keys
match the model module's own state_dict keys directly (network.N.*, conv_exp.*).

KNOWN fp16 OVERFLOW (see verify_vision_encoder.py for the mitigation)
-------------------------------------------------------------------------
At 1024x1024 input, pure fp16 forward overflows: values reach ~60928 at
network.9 (RepCPE, 1536ch), near the fp16 ceiling of 65504, then NaN at
network.10 (the attention stage that follows). The fp32 forward is clean
(max ~6657). This is a precision issue, not an architecture bug — confirmed
by Stage 1 cross-model verification passing in fp32.
"""

import sys

import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch.composite_ops import SDPA
from safetensors import safe_open


# ─── ANE-compatible replacements ─────────────────────────────────────────────


class ANELayerNorm(nn.Module):
    """
    Drop-in replacement for LayerNormChannel using F.layer_norm.

    Holds weight and bias as direct parameters (same names as LayerNormChannel)
    so checkpoint key paths match without renaming — do NOT wrap in nn.LayerNorm,
    which would add a ".norm." prefix to the key paths and break loading.

    F.layer_norm is recognized by the decomp table as the layer_norm composite
    op, which is what the ANE compiler needs to optimize it.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H, W, C] for layer_norm (normalizes last dim) -> back
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, [x.shape[-1]], self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ANEAttention(nn.Module):
    """
    Drop-in replacement for MHSA using the SDPA composite op.

    Original MHSA computes attention manually (q@k.T * scale, softmax, @v).
    This replaces it with SDPA so the ANE compiler recognizes and optimizes
    it as scaled_dot_product_attention.

    Input/output: [B, C, H, W] — flattened to [B, N, C] for attention internally.
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
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, C]

        qkv = (
            self.qkv(x_flat)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B, num_heads, N, head_dim]

        out = self.sdpa(q, k, v)  # [B, num_heads, N, head_dim]

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


# ─── Patch functions ──────────────────────────────────────────────────────────


def _patch_layernorm(model: nn.Module, LayerNormChannel) -> None:
    """
    Replace all LayerNormChannel instances with ANELayerNorm in-place.

    LayerNormChannel is passed as an argument (rather than imported inside
    this function) so llava_qwen is only imported once, by the caller.
    """
    for name, module in model.named_modules():
        if isinstance(module, LayerNormChannel):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacement = ANELayerNorm(num_features=module.weight.shape[0], eps=module.eps)
            replacement.weight = nn.Parameter(module.weight.clone())
            replacement.bias = nn.Parameter(module.bias.clone())
            setattr(parent, parts[-1], replacement)


def _patch_attention(model: nn.Module, MHSA) -> None:
    """Replace all MHSA instances with ANEAttention in-place. See _patch_layernorm."""
    for name, module in model.named_modules():
        if isinstance(module, MHSA):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            dim = module.qkv.in_features
            replacement = ANEAttention(dim=dim, head_dim=module.head_dim)
            replacement.qkv.weight = nn.Parameter(module.qkv.weight.clone())
            replacement.proj.weight = nn.Parameter(module.proj.weight.clone())
            replacement.proj.bias = nn.Parameter(module.proj.bias.clone())
            setattr(parent, parts[-1], replacement)


# ─── Vision encoder wrapper ───────────────────────────────────────────────────


class FastVLMVisionEncoder(nn.Module):
    """
    FastVLM vision encoder (FastViTHD) re-authored for Core AI export.

    See module docstring for the full config rationale, stage mapping, and
    forward path derivation from MobileCLIPVisionTower.feature_select().
    """

    def __init__(self, weights_dir: str) -> None:
        super().__init__()

        sys.path.insert(0, weights_dir)
        try:
            import llava_qwen
        finally:
            sys.path.pop(0)

        # Direct instantiation — see module docstring "WHY FastViT IS
        # INSTANTIATED DIRECTLY" for why timm.create_model is not used.
        self.model = llava_qwen.FastViT(
            layers=[2, 12, 24, 4, 2],
            embed_dims=[96, 192, 384, 768, 1536],
            mlp_ratios=[4, 4, 4, 4, 4],
            downsamples=[True, True, True, True, True],
            token_mixers=["repmixer", "repmixer", "repmixer", "attention", "attention"],
            norm_layer=llava_qwen.LayerNormChannel,
            pos_embs=[
                None, None, None,
                llava_qwen.RepCPE,  # class, not instance — see module docstring
                llava_qwen.RepCPE,
            ],
            inference_mode=True,
        )

        _patch_layernorm(self.model, llava_qwen.LayerNormChannel)
        _patch_attention(self.model, llava_qwen.MHSA)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]  (H=W=1024 for FastVLM)

        Returns:
            image_features: [B, H'*W', C]  patch embeddings, matching
            MobileCLIPVisionTower.feature_select() exactly.
        """
        x = self.model.forward_embeddings(pixel_values)  # patch_embed
        x = self.model.forward_tokens(x)                 # network stages
        x = self.model.conv_exp(x)                        # [B, C, H', W'] == "image_embeddings"

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)        # [B, H'*W', C]
        return x

    @classmethod
    def from_weights(
        cls,
        config,
        weights_dir: str,
        dtype: torch.dtype = torch.float16,
    ) -> "FastVLMVisionEncoder":
        """Load vision encoder weights from the FastVLM SafeTensors checkpoint."""
        model = cls(weights_dir).to(dtype=dtype)
        weights = _load_vision_weights(weights_dir, dtype=dtype)
        missing, unexpected = model.model.load_state_dict(weights, assign=True, strict=False)
        # head.* (GlobalPool2D classification head) is never called in forward() —
        # expected missing. .bn.* / num_batches_tracked are BatchNorm stats folded
        # into reparam_conv during inference-mode conversion — expected unexpected.
        unexpected_real = [
            k for k in unexpected
            if "head" not in k and ".bn." not in k and "num_batches_tracked" not in k
        ]
        missing_real = [k for k in missing if "head" not in k]
        if unexpected_real:
            raise RuntimeError(f"Unexpected keys: {unexpected_real[:5]}")
        if missing_real:
            raise RuntimeError(f"Missing keys: {missing_real[:5]}")
        return model


# ─── Weight loading ───────────────────────────────────────────────────────────

# Doubly-nested prefix confirmed from discover_weights.py output.
_VISION_PREFIX = "model.vision_tower.vision_tower.model."


def _load_vision_weights(
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """
    Load vision tower weights, stripping the doubly-nested prefix.

    Raw key: model.vision_tower.vision_tower.model.network.0.0.convffn.conv.conv.weight
    Stripped: network.0.0.convffn.conv.conv.weight  (matches self.model state_dict keys)

    Extracted as a standalone function so verify_vision_encoder.py can load
    fp32 weights for the reference model without duplicating this logic.
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
