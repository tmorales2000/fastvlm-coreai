"""
quantization.py — Compression utilities for FastVLM CoreAI export.

Provides named presets and YAML recipe loading matching Apple's
coreai-models/export/presets.py and pipeline.py patterns.

NAMED PRESETS (--compression PRESET)
--------------------------------------
  4bit  int4 symmetric_with_clipping per_block block_size=32.
        Apple's canonical macOS preset. Clips int4 to (-7,7) for equal
        bins — cleaner dequantization than plain symmetric.
        Excludes: SDPA, RoPE, RMSNorm (composite ops with internal precision).

  8bit  int8 per_channel symmetric.
        One scale per output channel. Most GPU-friendly — dequantization
        fuses with batch_matmul. Apple has no named 8bit macOS preset;
        this is our addition for lower-memory use cases.

  none  No quantization — fp16 only.

YAML RECIPES (--compression-config path.yaml)
----------------------------------------------
Per-model mixed-precision recipes. Top-level key: quantization_config.
Same format as QuantizerConfig.from_dict(). Produced by
scan_quantization_sensitivity.py or written by hand.

Optional coreai_models block for pipeline options:
  coreai_models:
    calibrate_activations: true

macOS vs iOS
------------
macOS: linear quantization (this module).
iOS:   palettization (k-means codebook, different module, not yet implemented).
       load_compression_config() raises NotImplementedError for iOS.

SIMULATION vs EXPORT
---------------------
apply_quantization_from_config() handles both:
  - For verify scripts: returns runnable nn.Module (prepare only, no finalize).
  - For export: returns finalized model ready for TorchConverter.
The caller controls which by passing finalize=True (export) or False (verify).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from coreai_opt.common import ExportBackend
from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig
from coreai_opt.quantization.config import ModuleQuantizerConfig
from coreai_opt.quantization.spec import (
    PerChannelGranularity,
    QuantizationScheme,
    QuantizationSpec,
)

# ── Composite op exclusions ───────────────────────────────────────────────────
# Mirrors _TORCH_MODULE_EXCLUSIONS from coreai-models/export/presets.py.
# These modules use specialized composite ops — quantizing their weights
# produces incorrect results or conflicts with the op's internal precision.
_COMPOSITE_OP_EXCLUSIONS: dict[str, None] = {
    "coreai_torch.composite_ops.SDPA":        None,
    "coreai_torch.composite_ops.RoPE":        None,
    "coreai_torch.composite_ops.RMSNormImpl": None,
}

# ── Named macOS presets ───────────────────────────────────────────────────────
# Mirrors MACOS_PRESETS in coreai-models/export/presets.py.
# Each entry is a quantization_config dict accepted by QuantizerConfig.from_dict().

MACOS_NAMED_PRESETS: dict[str, dict[str, Any]] = {
    "4bit": {
        # Apple's canonical macOS int4 preset.
        # symmetric_with_clipping clips int4 range to (-7, 7) rather than
        # (-8, 7), ensuring equal bins on each side of zero — cleaner
        # dequantization, no zero-point overhead of asymmetric.
        #
        # Uses module_type_configs targeting nn.Linear only (no global_config)
        # to avoid axis resolution errors on custom modules like FastVLMRMSNorm.
        # axis=1 is the reduction axis for nn.Linear weight [out, in].
        "description": "int4 symmetric_with_clipping per_block_32 (Apple macOS standard)",
        "quantization_config": {
            "execution_mode": "eager",
            "global_config": None,
            "module_type_configs": {
                "torch.nn.modules.linear.Linear": {
                    "op_state_spec": {
                        "weight": {
                            "dtype": "int4",
                            "qscheme": "symmetric_with_clipping",
                            "granularity": {
                                "type": "per_block",
                                "block_size": 32,
                                "axis": 1,  # explicit: input/reduction axis for Linear
                            },
                        }
                    },
                    "op_input_spec": None,
                    "op_output_spec": None,
                },
                **_COMPOSITE_OP_EXCLUSIONS,
            },
        },
    },
    "8bit": {
        # Our int8 preset — not in Apple's coreai-models (they don't ship
        # macOS int8). Per-channel symmetric: one scale per output row.
        # Dequantization fuses with batch_matmul on GPU/ANE.
        # Use case: lower memory than fp16 without int4 quality loss.
        #
        # Uses module_type_configs to target nn.Linear only (no global_config)
        # so axis=None can be auto-resolved per module type. global_config with
        # axis=None fails on custom modules (FastVLMRMSNorm etc.) that coreai-opt
        # doesn't know the axis default for.
        "description": "int8 per_channel symmetric",
        "quantization_config": {
            "execution_mode": "eager",
            "global_config": None,
            "module_type_configs": {
                "torch.nn.modules.linear.Linear": {
                    "op_state_spec": {
                        "weight": {
                            "dtype": "int8",
                            "qscheme": "symmetric",
                            "granularity": {
                                "type": "per_channel",
                                "axis": 0,  # explicit: output channel axis for Linear
                            },
                        }
                    },
                    "op_input_spec": None,
                    "op_output_spec": None,
                },
                **_COMPOSITE_OP_EXCLUSIONS,
            },
        },
    },
}


def load_compression_config(
    source: str | Path,
    platform: str = "macOS",
) -> tuple[dict, str]:
    """Load a compression config from a named preset or YAML file.

    Args:
        source: Named preset string ("4bit", "8bit") or Path to a YAML file.
        platform: "macOS" or "iOS". iOS raises NotImplementedError.

    Returns:
        (quantization_config_dict, label) where:
          - quantization_config_dict is passed to apply_quantization_from_config()
          - label is a human-readable string for logging/metadata

    Raises:
        NotImplementedError: If platform is "iOS".
        KeyError: If source is a string not in MACOS_NAMED_PRESETS.
        FileNotFoundError: If source is a Path that does not exist.
        ValueError: If the YAML has an unexpected structure.
    """
    if platform == "iOS":
        raise NotImplementedError(
            "iOS export uses palettization, not linear quantization. "
            "iOS export is not yet implemented."
        )

    # Named preset
    if isinstance(source, str):
        if source == "none":
            return None, "none"
        if source not in MACOS_NAMED_PRESETS:
            available = ", ".join(MACOS_NAMED_PRESETS.keys())
            raise KeyError(
                f"Unknown compression preset: {source!r}. "
                f"Available: {available}, none."
            )
        entry = MACOS_NAMED_PRESETS[source]
        return dict(entry["quantization_config"]), source

    # YAML file
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Compression config not found: {path}")

    with path.open() as fh:
        yaml_data = yaml.safe_load(fh)

    if not isinstance(yaml_data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level.")

    # Pop optional coreai_models pipeline options (e.g. calibrate_activations)
    pipeline_opts = yaml_data.pop("coreai_models", {})

    if len(yaml_data) != 1 or "quantization_config" not in yaml_data:
        raise ValueError(
            f"{path}: expected exactly one top-level key 'quantization_config', "
            f"got: {sorted(yaml_data)}."
        )

    config_dict = dict(yaml_data["quantization_config"])

    # Re-inline pipeline options (pipeline.py pops calibrate_activations
    # from the dict before rebuilding the coreai-opt config)
    if "calibrate_activations" in pipeline_opts:
        config_dict["calibrate_activations"] = pipeline_opts["calibrate_activations"]

    # Validate early — schema errors surface before we start loading the model
    QuantizerConfig.from_dict({"quantization_config": config_dict})

    return config_dict, path.stem


def apply_quantization_from_config(
    model: nn.Module,
    quantization_config: dict,
    example_inputs: tuple[torch.Tensor, ...],
    finalize: bool = True,
) -> nn.Module:
    """Apply quantization to a model using a resolved config dict.

    This is the single entry point for both named presets and YAML recipes.
    The config dict is the same format as QuantizerConfig.from_dict() expects
    under the 'quantization_config' key.

    Args:
        model: fp16 model in eval() mode on CPU.
        quantization_config: Dict from load_compression_config().
        example_inputs: Example inputs for quantizer tracing.
        finalize: If True (default), finalize for CoreAI export.
                  If False, return the prepared model for PSNR comparison.

    Returns:
        If finalize=True: finalized model for TorchConverter (not runnable as PyTorch).
        If finalize=False: prepared model with fake-quantize modules (runnable).
    """
    model.eval()
    assert next(model.parameters()).dtype == torch.float16, (
        "apply_quantization_from_config expects an fp16 model."
    )

    # Pop calibrate_activations before building the coreai-opt config
    # (it's a pipeline-level option, not a coreai-opt field)
    config_dict = dict(quantization_config)
    run_calibration = config_dict.pop("calibrate_activations", False)

    config = QuantizerConfig.from_dict({"quantization_config": config_dict})
    quantizer = Quantizer(model, config)
    prepared = quantizer.prepare(example_inputs)

    if run_calibration:
        # Calibration with real data — currently not wired to a data source.
        # Future: pass calibration_data_fn from export_fastvlm.py.
        print("[WARN] calibrate_activations=True in config but calibration data "
              "not yet wired. Running without calibration.")

    if not finalize:
        return prepared

    return quantizer.finalize(
        model=prepared,
        backend=ExportBackend.CoreAI,
    )


def finalize_for_export(
    quantized_model: nn.Module,
    quantizer: Quantizer,
) -> nn.Module:
    """Finalize a prepared model for CoreAI export.

    Legacy entry point kept for verify_decoder.py compatibility.
    New code should use apply_quantization_from_config(finalize=True).
    """
    return quantizer.finalize(
        model=quantized_model,
        backend=ExportBackend.CoreAI,
    )


def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """PSNR (dB) between reference and test tensors, computed in float64."""
    ref_f = ref.detach().float().to(torch.float64)
    test_f = test.detach().float().to(torch.float64)
    mse = torch.mean((ref_f - test_f) ** 2).item()
    if mse == 0.0:
        return float("inf")
    max_val = ref_f.abs().max().item()
    if max_val == 0.0:
        return float("inf")
    return (
        20.0 * torch.log10(torch.tensor(max_val)).item()
        - 10.0 * torch.log10(torch.tensor(mse)).item()
    )
