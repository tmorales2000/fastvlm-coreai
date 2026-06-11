"""
fastvlm_projector.py — Re-authored FastVLM multimodal projector for CoreAI export.

The projector is a small MLP (mlp2x_gelu or similar) that maps image patch
embeddings from the vision encoder into the language model's embedding space.

Weight key note: FastVLM.swift sanitize() renames mm_projector.N → mm_projector.layers.N.
Using self.layers = nn.Sequential(...) ensures the weight keys match the post-sanitize
names without any additional remapping.
"""

import glob
import os
import re

import torch
import torch.nn as nn
from safetensors import safe_open


class FastVLMProjector(nn.Module):
    """
    MLP projector. Depth and activation derived from mm_projector_type config value
    (e.g. 'mlp2x_gelu' → 2 Linear layers with GELU between them).

    self.layers key prefix matches the sanitize() rename in FastVLM.swift
    (mm_projector.N → mm_projector.layers.N), so weights load directly.
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size       # language model hidden dim
        mm_hidden = config.mm_hidden_size      # vision encoder output dim
        proj_type = config.mm_projector_type   # e.g. "mlp2x_gelu"

        match = re.match(r"^mlp(\d+)x_gelu$", proj_type)
        depth = int(match.group(1)) if match else 2

        layers: list[nn.Module] = [nn.Linear(mm_hidden, hidden_size)]
        for _ in range(1, depth):
            layers += [nn.GELU(), nn.Linear(hidden_size, hidden_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMProjector":
        """Load from SafeTensors weights directory."""
        model = cls(config).to(dtype=torch.float16)
        st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
        weights: dict[str, torch.Tensor] = {}
        for path in st_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "mm_projector" not in key:
                        continue
                    # Strip top-level prefix (e.g. "model.mm_projector.layers.0.weight"
                    # → "layers.0.weight") to match self.layers.N.weight
                    stripped = key
                    for remove in ["model.", "mm_projector."]:
                        stripped = stripped.replace(remove, "")
                    t = f.get_tensor(key)
                    if t.dtype != torch.float16:
                        t = t.to(torch.float16)
                    weights[stripped] = t
        model.load_state_dict(weights, assign=True, strict=True)
        return model
