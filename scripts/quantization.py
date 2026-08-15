"""
quantization.py — Compression utilities for FastVLM CoreAI export.

Provides named presets and YAML recipe loading matching Apple's
coreai-models/export/presets.py and pipeline.py patterns.

Requires coreai-opt >= 0.2.2.dev0 (install from local source):
  uv pip install -e ~/git/apple/coreai-optimization/ --no-deps

NAMED PRESETS (--compression PRESET)
--------------------------------------
Mirrors Apple's coreai.llm.export --compression options (macOS):

  4bit   Apple's canonical macOS int4 preset.
         symmetric_with_clipping per_block block_size=32 axis=1.

  8bit   int8 symmetric_with_clipping per_block_32.
         Mirrors Apple's 4bit structure with int8 dtype.
         Apple ships no int8 macOS preset — this is our addition.

  none   No quantization — fp16 only.

YAML RECIPES (--compression-config path.yaml)
----------------------------------------------
Per-model mixed-precision recipes and non-standard compression schemes.
--compression and --compression-config are mutually exclusive.

Pre-built recipes in quantization_recipes/:
  4bit_per_channel.yaml — int4 per_channel symmetric. Fuses with
    batch_matmul on GPU (7× faster than 4bit for large models).
    Recommended for 1.5B and 7B production deployments.
  fastvlm-{variant}-aggressive.yaml — mixed int4/int8/fp16 from
    scan_quantization_sensitivity.py.
  fastvlm-{variant}-conservative.yaml — mixed int8/fp16.

Custom YAML format (QuantizerConfig native):
  Top-level key: quantization_config
  Supports module_name_configs for per-layer targeting.

macOS vs iOS
------------
macOS: linear quantization (coreai-opt QuantizerConfig).
       Presets: 4bit, 8bit, none.
       Applied to decoder weights before torch.export.

iOS:   palettization (k-means codebook, KMeansPalettizer via coreai-opt).
       Presets: 4bit_weight_palettized_group8, 4bit_weight_palettized_group32.
       Different compression API: torch_palettization_config vs torch_quantization_config.
       Excludes nn.Embedding and LoadEmbeddings.
       Note: iOS export requires fastvlm_ios.py (decoder only, no vision).

iOS:   palettization (k-means codebook, KMeansPalettizer via coreai-opt).
       Presets: 4bit_weight_palettized_group8, 4bit_weight_palettized_group32 (default).
       Different compression API: torch_palettization_config vs torch_quantization_config.
       Excludes nn.Embedding and LoadEmbeddings (iOS-specific embedding module).
       Note: iOS palettization uses apply_palettization_from_config(), not
       apply_quantization_from_config(). Full iOS export also requires a
       different model architecture (BC1S layout, readonly KV I/O, 4 entrypoints).

SIMULATION vs EXPORT
---------------------
apply_quantization_from_config() handles both:
  - For verify scripts: finalize=False returns runnable nn.Module.
  - For export: finalize=True returns finalized model for TorchConverter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from coreai_opt.common import ExportBackend
from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig

# ── Composite op exclusions ───────────────────────────────────────────────────
# Mirrors _TORCH_MODULE_EXCLUSIONS from coreai-models/export/presets.py.
# Setting a module type to None in module_type_configs excludes it entirely.
# FastVLMRMSNorm added because global_config traverses all nn.Module subclasses
# with parameters — our RMSNorm has a 1D weight that per_block axis=1 can't handle.
# Apple's models use coreai_models.primitives.macos.RMSNorm (already excluded);
# we need to exclude our own implementation explicitly.
_COMPOSITE_OP_EXCLUSIONS: dict[str, None] = {
    "coreai_torch.composite_ops.SDPA":        None,
    "coreai_torch.composite_ops.RoPE":        None,
    "coreai_torch.composite_ops.RMSNormImpl": None,
    "fastvlm_decoder.FastVLMRMSNorm":         None,
}

# Full dotted name for nn.Linear as coreai-opt expects it
_LINEAR_TYPE = "torch.nn.modules.linear.Linear"

# ── Named macOS presets ───────────────────────────────────────────────────────

# ── iOS palettization exclusions ─────────────────────────────────────────────
# Mirrors _IOS_PALETTIZATION_EMBEDDING_EXCLUSIONS from coreai-models/export/presets.py.
# Embedding modules excluded from iOS palettization — they use specialized ops.
_IOS_EMBEDDING_EXCLUSIONS: dict[str, None] = {
    "torch.nn.modules.sparse.Embedding":                    None,
    "coreai_models.primitives.ios.embedding.LoadEmbeddings": None,
}


MACOS_NAMED_PRESETS: dict[str, dict[str, Any]] = {
    "4bit": {
        # Mirrors Apple's MACOS_PRESETS["4bit"] from coreai-models/export/presets.py
        # exactly, including global_config structure.
        # FastVLMRMSNorm excluded via _COMPOSITE_OP_EXCLUSIONS so global_config
        # doesn't hit its 1D weight with per_block axis=1.
        "description": "int4 symmetric_with_clipping per_block_32 (Apple macOS standard)",
        "quantization_config": {
            "execution_mode": "eager",
            "global_config": {
                "op_state_spec": {
                    "weight": {
                        "dtype": "int4",
                        "qscheme": "symmetric_with_clipping",
                        "granularity": {
                            "type": "per_block",
                            "block_size": 32,
                            "axis": 1,
                        },
                    }
                },
                "op_input_spec": None,
                "op_output_spec": None,
            },
            "module_type_configs": {
                **_COMPOSITE_OP_EXCLUSIONS,
            },
        },
    },
    "8bit": {
        # Mirrors Apple's 4bit preset structure exactly, substituting int8.
        # FastVLMRMSNorm excluded via _COMPOSITE_OP_EXCLUSIONS.
        "description": "int8 symmetric_with_clipping per_block_32 (Apple macOS standard, int8)",
        "quantization_config": {
            "execution_mode": "eager",
            "global_config": {
                "op_state_spec": {
                    "weight": {
                        "dtype": "int8",
                        "qscheme": "symmetric_with_clipping",
                        "granularity": {
                            "type": "per_block",
                            "block_size": 32,
                            "axis": 1,
                        },
                    }
                },
                "op_input_spec": None,
                "op_output_spec": None,
            },
            "module_type_configs": {
                **_COMPOSITE_OP_EXCLUSIONS,
            },
        },
    },
}


# ── Named iOS presets ────────────────────────────────────────────────────────
# Mirrors IOS_PRESETS in coreai-models/export/presets.py.
# iOS uses palettization (k-means codebook), not linear quantization.
# The palettization config is passed to apply_palettization_from_config()
# (not apply_quantization_from_config() — different coreai-opt API).
# DEFAULT: 4bit_weight_palettized_group32

IOS_NAMED_PRESETS: dict[str, dict] = {
    "4bit_weight_palettized_group8": {
        "description": "int4 palettization group_size=8 (iOS, highest quality)",
        "torch_palettization_config": {
            "global_config": {
                "op_state_spec": {
                    "weight": {
                        "n_bits": 4,
                        "granularity": {
                            "type": "per_grouped_channel",
                            "axis": 0,
                            "group_size": 8,
                        },
                    }
                }
            },
            "module_type_configs": _IOS_EMBEDDING_EXCLUSIONS,
        },
    },
    "4bit_weight_palettized_group32": {
        "description": "int4 palettization group_size=32 (iOS, default)",
        "torch_palettization_config": {
            "global_config": {
                "op_state_spec": {
                    "weight": {
                        "n_bits": 4,
                        "granularity": {
                            "type": "per_grouped_channel",
                            "axis": 0,
                            "group_size": 32,
                        },
                    }
                }
            },
            "module_type_configs": _IOS_EMBEDDING_EXCLUSIONS,
        },
    },
}

DEFAULT_IOS_COMPRESSION_PRESET = "4bit_weight_palettized_group32"


def _build_quantizer_config(quantization_config: dict) -> QuantizerConfig:
    """Build a QuantizerConfig from a quantization_config dict.

    Uses QuantizerConfig.from_dict() which supports the full schema
    including module_name_configs for per-layer targeting.
    """
    return QuantizerConfig.from_dict({"quantization_config": quantization_config})


def load_compression_config(
    source: str | Path,
    platform: str = "macOS",
) -> tuple[dict | None, str]:
    """Load a compression config from a named preset or YAML file.

    Args:
        source: Named preset string ("4bit", "4bit_per_channel", "8bit", "none")
                or Path to a YAML file.
        platform: "macOS" or "iOS". iOS raises NotImplementedError.

    Returns:
        (quantization_config_dict, label) where config is None for "none".

    Raises:
        NotImplementedError: If platform is "iOS".
        KeyError: If source is an unknown preset name.
        FileNotFoundError: If source is a Path that does not exist.
        ValueError: If the YAML has an unexpected structure.
    """
    if platform == "iOS":
        raise NotImplementedError(
            "iOS export uses palettization, not linear quantization. "
            "iOS export is not yet implemented."
        )

    # Named preset (string) or YAML file (string or Path)
    if isinstance(source, str):
        if source == "none":
            return None, "none"

        # If it looks like a file path, convert to Path and fall through
        source_path = Path(source)
        if source_path.suffix in ('.yaml', '.yml') or '/' in source or '\\' in source:
            source = source_path
        elif source in MACOS_NAMED_PRESETS:
            if platform == "iOS":
                raise ValueError(
                    f"Preset {source!r} is a macOS linear quantization preset. "
                    f"For iOS use: {', '.join(IOS_NAMED_PRESETS.keys())}."
                )
            entry = MACOS_NAMED_PRESETS[source]
            config = dict(entry["quantization_config"])
            _build_quantizer_config(config)
            return config, source
        elif source in IOS_NAMED_PRESETS:
            if platform == "macOS":
                raise ValueError(
                    f"Preset {source!r} is an iOS palettization preset. "
                    f"For macOS use: {', '.join(MACOS_NAMED_PRESETS.keys())}."
                )
            return dict(IOS_NAMED_PRESETS[source]), source
        else:
            available_macos = ", ".join(MACOS_NAMED_PRESETS.keys())
            available_ios   = ", ".join(IOS_NAMED_PRESETS.keys())
            raise KeyError(
                f"Unknown compression preset: {source!r}. "
                f"macOS presets: {available_macos}, none. "
                f"iOS presets: {available_ios}, none."
            )

    # YAML file — use QuantizerConfig.from_yaml() directly
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Compression config not found: {path}")

    import yaml
    with path.open() as fh:
        yaml_data = yaml.safe_load(fh)

    if not isinstance(yaml_data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level.")

    # Pop optional coreai_models pipeline options
    pipeline_opts = yaml_data.pop("coreai_models", {})

    if "quantization_config" not in yaml_data:
        raise ValueError(
            f"{path}: expected top-level key 'quantization_config', "
            f"got: {sorted(yaml_data)}."
        )

    config = dict(yaml_data["quantization_config"])

    # Re-inline pipeline options
    if "calibrate_activations" in pipeline_opts:
        config["calibrate_activations"] = pipeline_opts["calibrate_activations"]

    # Validate — surfaces schema errors before model loading
    _build_quantizer_config(config)

    return config, path.stem


def apply_quantization_from_config(
    model: nn.Module,
    quantization_config: dict,
    example_inputs: tuple[torch.Tensor, ...],
    finalize: bool = True,
) -> nn.Module:
    """Apply quantization to a model using a resolved config dict.

    Single entry point for both named presets and YAML recipes.
    Supports module_name_configs for per-layer mixed precision
    (requires coreai-opt >= 0.2.2.dev0).

    Args:
        model: fp16 model in eval() mode on CPU.
        quantization_config: Dict from load_compression_config().
        example_inputs: Example inputs for quantizer tracing.
        finalize: True = finalize for CoreAI export (not runnable as PyTorch).
                  False = prepare only, stays runnable (for PSNR comparison).

    Returns:
        Finalized or prepared model depending on finalize flag.
    """
    model.eval()
    assert next(model.parameters()).dtype == torch.float16, (
        "apply_quantization_from_config expects an fp16 model."
    )

    # Pop calibrate_activations — pipeline option, not a coreai-opt field
    config_dict = dict(quantization_config)
    run_calibration = config_dict.pop("calibrate_activations", False)

    config = _build_quantizer_config(config_dict)
    quantizer = Quantizer(model, config)
    prepared = quantizer.prepare(example_inputs)

    if run_calibration:
        print("[WARN] calibrate_activations=True but calibration data not yet wired. "
              "Running without calibration.")

    if not finalize:
        return prepared

    return quantizer.finalize(model=prepared, backend=ExportBackend.CoreAI)


def apply_palettization_from_config(
    model: nn.Module,
    palettization_config: dict,
    example_inputs: tuple[torch.Tensor, ...],
    finalize: bool = True,
) -> nn.Module:
    """Apply iOS palettization to a model using a resolved config dict.

    Uses coreai-opt KMeansPalettizer (different from Quantizer used for macOS).
    Palettization replaces weights with k-means codebook indices — ANE-native
    for iOS deployment.

    Args:
        model: fp16 model in eval() mode on CPU.
        palettization_config: Dict from load_compression_config() for an iOS preset.
            Must contain 'torch_palettization_config' key.
        example_inputs: Example inputs for palettizer tracing.
        finalize: True = finalize for CoreAI export. False = prepare only (runnable).

    Note: iOS export also requires a different model architecture (BC1S layout,
    readonly KV I/O). This function only handles the palettization step.
    """
    from coreai_opt.palettization import Palettizer, PalettizerConfig

    model.eval()
    assert next(model.parameters()).dtype == torch.float16, (
        "apply_palettization_from_config expects an fp16 model."
    )

    pal_config_dict = palettization_config.get("torch_palettization_config", palettization_config)
    config = PalettizerConfig.from_dict({"palettization_config": pal_config_dict})
    palettizer = Palettizer(model, config)
    prepared = palettizer.prepare(example_inputs)

    if not finalize:
        return prepared

    return palettizer.finalize(model=prepared, backend=ExportBackend.CoreAI)


def finalize_for_export(quantized_model: nn.Module, quantizer: Quantizer) -> nn.Module:
    """Legacy entry point. New code should use apply_quantization_from_config(finalize=True)."""
    return quantizer.finalize(model=quantized_model, backend=ExportBackend.CoreAI)


def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """PSNR (dB) between reference and test tensors, computed in float64."""
    ref_f  = ref.detach().float().to(torch.float64)
    test_f = test.detach().float().to(torch.float64)
    mse    = torch.mean((ref_f - test_f) ** 2).item()
    if mse == 0.0:
        return float("inf")
    max_val = ref_f.abs().max().item()
    if max_val == 0.0:
        return float("inf")
    return (
        20.0 * torch.log10(torch.tensor(max_val)).item()
        - 10.0 * torch.log10(torch.tensor(mse)).item()
    )
