"""
fastvlm_vision_encoder.py — Re-authored FastVLM vision encoder (FastViTHD) for CoreAI.

This file is a guided stub. Complete the implementation using:
  1. MLXVLM/Models/FastVLM.swift as the authoritative forward pass specification
  2. discovery/fastvlm-1.5b.txt for exact weight keys and shapes
  3. docs/architecture.md for stage dimensions (fill in after running discover_weights.py)
  4. coreai-models/skills/skills/model-authoring/references/neural_engine_rules.md

Architecture map (Swift → PyTorch):
  MobileOneBlock         → nn.Conv2d + GELU                  (reparam_conv weights)
  ReparamLargeKernelConv → nn.Conv2d + GELU                  (lkb_reparam weights)
  RepMixer               → nn.Conv2d depthwise                (groups=dim)
  ConvFFN.ConvWithNorm   → Conv2d + nn.BatchNorm2d            (eval() folds BN automatically)
  LayerNormChannel       → nn.LayerNorm(C)                   (replace manual decomp entirely)
  SEBlock                → AdaptiveAvgPool2d(1) + 2x Conv2d  (MUST fix dynamic pool)
  RepCPE                 → nn.Conv2d depthwise spatial        (reparam_conv)
  MHSA                   → Linear QKV + SDPA composite op     (ExternalizeSpec required)
  RepMixerBlock          → RepMixer + ConvFFN + layer_scale
  AttentionBlock         → LayerNorm + MHSA + ConvFFN + 2x layer_scale
  PatchEmbed             → ReparamLargeKernelConv + MobileOneBlock
  ConvolutionalStem      → 3x MobileOneBlock (strides 2,2,1)
  GlobalPool2D           → AdaptiveAvgPool2d(1) + nn.Linear

ANE rules (all required for full ANE residency):
  - SEBlock: replace dynamic AvgPool2d(h,w) with AdaptiveAvgPool2d(1)
  - LayerNorm: use nn.LayerNorm, not manual mean/pow/sqrt decomposition
  - BatchNorm: call model.eval() before export — BN folds via decomp table
  - No Python float literals as constants — use torch.ones/zeros with explicit dtype
  - MHSA: use SDPA composite op, NOT F.scaled_dot_product_attention directly
  - All Conv2d strides must have prime factors of only 2 and 3
"""

import glob
import os

import torch
import torch.nn as nn
from coreai_torch.composite_ops import SDPA, RoPE
from safetensors import safe_open


# ─── TODO: implement these modules from FastVLM.swift spec ───────────────────
# Reference each class in MLXVLM/Models/FastVLM.swift while implementing.
# Use discovery/fastvlm-1.5b.txt to verify weight key names match.


class MobileOneBlock(nn.Module):
    """
    TODO: Implement from FastVLM.swift MobileOneBlock.
    Uses reparameterized Conv2d weights (reparam_conv key in checkpoint).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        raise NotImplementedError("Implement from FastVLM.swift MobileOneBlock")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LayerNormChannel(nn.Module):
    """
    Replace FastVLM.swift LayerNormChannel manual decomposition with nn.LayerNorm.
    The decomp table handles layer_norm as a composite op automatically.
    Do NOT reimplement the manual mean/pow/sqrt path.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        # Standard LayerNorm over the channel dimension
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, H, W, C] (channels-last, matching MLX convention)
        return self.norm(x)


class SEBlock(nn.Module):
    """
    TODO: Implement SEBlock with FIXED spatial pooling for ANE compatibility.
    
    FastVLM.swift uses dynamic AvgPool2d(h, w) — ANE requires static shapes.
    Replace with AdaptiveAvgPool2d(1) which pools to [B, C, 1, 1] regardless
    of spatial size. Determine fixed h/w for each stage from discovery output.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)   # Fixed — ANE safe
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(mid, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))
        return x * scale


class MHSA(nn.Module):
    """
    TODO: Implement Multi-Head Self-Attention using SDPA composite op.
    
    FastVLM.swift uses MLXFast.scaledDotProductAttention — replace with
    the SDPA composite op so it is preserved as a named composite op
    after externalization (ExternalizeSpec with composite_op_name=
    'scaled_dot_product_attention').
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.sdpa = SDPA(is_causal=False)   # Not causal for vision encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Complete MHSA implementation from FastVLM.swift")


# ─── Top-level encoder ────────────────────────────────────────────────────────


class FastVLMVisionEncoder(nn.Module):
    """
    TODO: Complete FastViTHD implementation.

    Follow the stage structure from FastVLM.swift FastViTHDModel:
      ConvolutionalStem → network stages (RepMixerBlock + AttentionBlock) →
      convExp → GlobalPool2D

    The exact stage configuration (depths, dims, num_heads per stage) comes
    from the vision_config in config.json — check discovery output for values.
    """

    def __init__(self, config):
        super().__init__()
        raise NotImplementedError(
            "Implement FastVLMVisionEncoder from FastVLM.swift FastViTHDModel. "
            "See docs/architecture.md for stage dimensions after running discover_weights.py."
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W] float16

        Returns:
            image_features: [B, h*w, C] float16 patch embeddings
        """
        raise NotImplementedError

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMVisionEncoder":
        """Load from SafeTensors weights directory."""
        model = cls(config).to(dtype=torch.float16)
        # Apply sanitize() renames from FastVLM.swift before loading:
        #   layer_scale_  →  layerScale
        #   vision_model.network.N.M  →  vision_model.network.N.layers.M
        weights = _load_vision_weights(weights_dir)
        _sanitize_vision_keys(weights)
        model.load_state_dict(weights, assign=True, strict=True)
        return model


def _load_vision_weights(
    weights_dir: str, dtype: torch.dtype = torch.float16
) -> dict[str, torch.Tensor]:
    """Load vision_tower weights from SafeTensors, stripping the prefix."""
    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    result = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "vision_tower" not in key:
                    continue
                stripped = key.replace("vision_tower.", "")
                t = f.get_tensor(key)
                if t.dtype != dtype and "zero_point" not in key:
                    t = t.to(dtype)
                result[stripped] = t
    return result


def _sanitize_vision_keys(state_dict: dict[str, torch.Tensor]) -> None:
    """
    Apply the same key renames as FastVLM.swift VisionModel.sanitize():
      layer_scale_  →  layerScale
      vision_model.network.N.M.X  →  vision_model.network.N.layers.M.X

    Modifies state_dict in-place.
    Verify against your discovery output that these transforms are correct.
    """
    import re

    keys_to_rename = {}
    for key in list(state_dict.keys()):
        new_key = key

        # layer_scale_ → layerScale
        new_key = new_key.replace("layer_scale_", "layerScale")

        # vision_model.network.N.M.X → vision_model.network.N.layers.M.X
        match = re.match(
            r"(.+)\.vision_model\.network\.(\d+)\.(\d+)\.(.+)", new_key
        )
        if match:
            new_key = (
                f"{match.group(1)}.vision_model.network.{match.group(2)}"
                f".layers.{match.group(3)}.{match.group(4)}"
            )

        if new_key != key:
            keys_to_rename[key] = new_key

    for old_key, new_key in keys_to_rename.items():
        state_dict[new_key] = state_dict.pop(old_key)
