"""
models.py — Model registry helpers for fastvlm-coreai.

Reads models.yaml and provides:
  - resolve_variant(): expand --variant shorthand to registry key
  - load_registry(): load the full models.yaml
  - registry_key_to_paths(): derive standard paths from a registry key

This module is the single source of truth for model naming conventions.
All scripts that accept --variant should call resolve_variant() immediately
after argument parsing.

Variant resolution rules:
  Short FastVLM aliases (maintained for backward compatibility):
    "0.5b" → "fastvlm-0.5b"
    "1.5b" → "fastvlm-1.5b"
    "7b"   → "fastvlm-7b"

  Fully-qualified registry keys pass through unchanged:
    "fastvlm-0.5b" → "fastvlm-0.5b"
    "qwen3-vl-2b"  → "qwen3-vl-2b"

  Unknown strings pass through and let the caller validate against the registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT    = Path(__file__).parent.parent
MODELS_YAML  = REPO_ROOT / "models.yaml"

# FastVLM short aliases — maintained for backward compatibility
_FASTVLM_ALIASES = {"0.5b", "1.5b", "7b"}


def resolve_variant(variant: str) -> str:
    """Expand --variant shorthand to fully-qualified models.yaml registry key.

    Examples:
        resolve_variant("0.5b")         → "fastvlm-0.5b"
        resolve_variant("fastvlm-0.5b") → "fastvlm-0.5b"
        resolve_variant("qwen3-vl-2b")  → "qwen3-vl-2b"
    """
    if variant in _FASTVLM_ALIASES:
        return f"fastvlm-{variant}"
    return variant


def load_registry() -> dict[str, Any]:
    """Load models.yaml and return the models dict.

    Returns:
        Dict mapping registry key → model entry dict.

    Raises:
        FileNotFoundError: if models.yaml is not found.
        ValueError: if models.yaml has no 'models' section.
    """
    import yaml
    if not MODELS_YAML.exists():
        raise FileNotFoundError(
            f"models.yaml not found at {MODELS_YAML}\n"
            f"This file should be in the repository root."
        )
    with open(MODELS_YAML) as f:
        data = yaml.safe_load(f)
    models = data.get("models", {})
    if not models:
        raise ValueError("No models defined in models.yaml.")
    return models


def get_model_entry(variant: str) -> dict[str, Any]:
    """Get the models.yaml entry for a variant (after resolution).

    Args:
        variant: Short alias or fully-qualified registry key.

    Returns:
        Model entry dict with keys: repo, directory, type, variant, description.

    Raises:
        KeyError: if variant is not in the registry.
    """
    key     = resolve_variant(variant)
    models  = load_registry()
    if key not in models:
        available = ", ".join(models.keys())
        raise KeyError(
            f"Model '{key}' not found in models.yaml.\n"
            f"Available: {available}\n"
            f"Short aliases: 0.5b, 1.5b, 7b (FastVLM variants)"
        )
    return models[key]


def weights_dir(variant: str) -> Path:
    """Return the weights directory path for a variant."""
    entry = get_model_entry(variant)
    return REPO_ROOT / entry["directory"]


def registry_keys() -> list[str]:
    """Return all registry keys from models.yaml."""
    return list(load_registry().keys())


def argparse_choices() -> list[str]:
    """Return valid --variant choices: short aliases + fully-qualified keys.

    Use this for argparse choices= to accept both forms:
        ap.add_argument("--variant", choices=models.argparse_choices())
    """
    keys = registry_keys()
    # Add short aliases for FastVLM
    aliases = [v for v in _FASTVLM_ALIASES if f"fastvlm-{v}" in keys]
    return sorted(set(keys) | set(aliases))
