"""
fastvlm_projector.py — Re-authored FastVLM multimodal projector for Core AI export.

WHAT THIS FILE IS
-----------------
FastVLM's multimodal projector is a small MLP that bridges the vision encoder
and the language decoder. It maps image patch embeddings from the vision
encoder's output space into the language model's embedding space, so the
decoder can attend over image tokens the same way it attends over text tokens.

The projector type is specified in config.mm_projector_type. For FastVLM 1.5B
this is "mlp2x_gelu": two Linear layers with a GELU activation between them.
The "2x" means depth=2 (two linear layers), not that the hidden size doubles.

ARCHITECTURE (mlp2x_gelu, 1.5B)
---------------------------------
  Linear(mm_hidden_size=3072, hidden_size=1536)   # layers.0
  GELU()                                           # layers.1
  Linear(hidden_size=1536, hidden_size=1536)       # layers.2

Input:  [B, N, mm_hidden_size]  — N patch embeddings from the vision encoder
Output: [B, N, hidden_size]     — projected embeddings in the LM's token space

Both dimensions come from config and are not hardcoded. The depth also comes
from config (the integer in "mlp2x_gelu"), so this class handles any mlpNx_gelu
variant without modification.

WEIGHT KEY MAPPING
------------------
The FastVLM checkpoint stores projector weights as:
  model.mm_projector.0.weight   (Linear weight)
  model.mm_projector.0.bias     (Linear bias)
  model.mm_projector.2.weight   (Linear weight — index 2 because GELU is index 1)
  model.mm_projector.2.bias

After stripping "model.mm_projector." the keys are "0.weight", "0.bias", etc.
This file uses self.layers = nn.Sequential(...), whose keys are "layers.0.weight",
"layers.0.bias", etc. The "layers." prefix is prepended during loading to bridge
the two naming schemes.

Note: "layers.N" naming also matches the sanitize() rename in FastVLM.swift
(mm_projector.N → mm_projector.layers.N), which is why this naming was chosen.

VERIFIED RESULTS (1.5B, June 2026)
-----------------------------------
Cross-model PSNR (fp32 port vs original HF mm_projector): see verify_projector.py
Self-consistency PSNR (fp16 vs fp32 port):                 90.9 dB  [PASS]
"""

import glob
import os
import re

import torch
import torch.nn as nn
from safetensors import safe_open


class FastVLMProjector(nn.Module):
    """
    FastVLM multimodal projector (mlp2x_gelu or similar).

    Architecture and depth are derived from config.mm_projector_type at
    construction time — no hardcoded layer counts. For "mlp2x_gelu":
      layers.0 = Linear(mm_hidden_size → hidden_size)
      layers.1 = GELU()
      layers.2 = Linear(hidden_size → hidden_size)

    Uses self.layers = nn.Sequential so that state_dict keys follow the
    "layers.N.*" naming scheme, matching both the checkpoint mapping and
    the FastVLM.swift sanitize() rename.
    """

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size      # LM hidden dim (1536 for 1.5B)
        mm_hidden   = config.mm_hidden_size   # vision encoder output dim (3072 for 1.5B)
        proj_type   = config.mm_projector_type  # e.g. "mlp2x_gelu"

        match = re.match(r"^mlp(\d+)x_gelu$", proj_type)
        depth = int(match.group(1)) if match else 2

        layers: list[nn.Module] = [nn.Linear(mm_hidden, hidden_size)]
        for _ in range(1, depth):
            layers += [nn.GELU(), nn.Linear(hidden_size, hidden_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @classmethod
    def from_weights(
        cls,
        config,
        weights_dir: str,
        dtype: torch.dtype = torch.float16,
    ) -> "FastVLMProjector":
        """
        Load projector weights from the FastVLM SafeTensors checkpoint.

        Key mapping:
          Checkpoint: model.mm_projector.0.weight
          Strip prefix "model.mm_projector.": 0.weight
          Add "layers." prefix: layers.0.weight  ← matches self.layers.N.weight

        All weights are stored as bfloat16 in the checkpoint and cast to
        the target dtype (default float16) on load.
        """
        model = cls(config).to(dtype=dtype)
        weights = _load_projector_weights(weights_dir, dtype=dtype)
        model.load_state_dict(weights, assign=True, strict=True)
        return model


# ─── Weight loading ───────────────────────────────────────────────────────────


def _load_projector_weights(
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """
    Load projector weights from SafeTensors, applying the key mapping:
      model.mm_projector.N.* → layers.N.*

    Extracted as a standalone function so verify_projector.py can load
    fp32 weights for the reference model without duplicating this logic.
    """
    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    weights: dict[str, torch.Tensor] = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mm_projector" not in key:
                    continue
                # "model.mm_projector.0.weight" → "layers.0.weight"
                stripped = "layers." + key.removeprefix("model.mm_projector.")
                t = f.get_tensor(key)
                if t.dtype != dtype:
                    t = t.to(dtype)
                weights[stripped] = t
    return weights
