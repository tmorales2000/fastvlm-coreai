"""
quantization.py — Shared quantization utility for FastVLM verify and export scripts.

PURPOSE
-------
Provides a single, consistent interface for applying coreai-opt quantization
to the FastVLM decoder, used by:

  - verify_decoder.py  (--quantize flag: simulate quantization, compare PSNR
                         against the Stage 2 fp16 reference)
  - export_fastvlm.py  (--quantize flag: apply quantization before staging
                         into TorchConverter)

NAMING NOTE
-----------
This module is intentionally NOT about "compression" in the general sense.
fp32 -> fp16 is a mandatory precision cast both the decoder and projector
go through for ANE execution — it is not optional and not what this module
calls "quantization." Quantization here means INT8 or INT4 weight-only
quantization, applied only to the decoder, only optionally, on top of the
fp16 cast. See verify_decoder.py's three-stage structure (Correctness ->
Precision -> Quantization) for where this fits in the overall pipeline.

SUPPORTED QUANTIZATION LEVELS
-------------------------------
  int8 : coreai-opt weight-only int8 quantization, grouped asymmetric,
         block_size=64 along input-channel axis. Matches Apple's MLX
         int8 scheme for FastVLM 1.5B.
  int4 : coreai-opt weight-only int4 quantization, same scheme.
         Matches Apple's MLX int4 scheme for FastVLM 7B.

Mutually exclusive — a model is exported as either int8 or int4, never both,
and there is no "all" option. (verify_decoder.py runs each level as a
separate invocation if you want to compare them.)

PRODUCTION QUANTIZATION TARGETS (validated)
----------------------------------------------
  0.5B decoder : no quantization (fp16 only; Apple ships unquantized)
  1.5B decoder : int8  (PSNR vs fp16: 46.2 dB, PASS; matches Apple MLX tier)
  7B   decoder : int8  (int4 fails — 22.4 dB vs fp16, due to fp16 pre-cast
                        sensitivity; see psnr_results.md)
  projector    : no quantization (fp16 only; Apple never quantizes it)
  vision enc.  : no quantization (fp16 only; CoreML export)

SIMULATION vs EXPORT DISTINCTION (critical)
--------------------------------------------
coreai-opt operates in two distinct modes:

  SIMULATION (verify scripts): Call prepare(example_inputs) only.
    The returned model is a standard nn.Module with fake-quantize modules
    inserted around weights. It is directly runnable in PyTorch for forward
    passes / PSNR comparison. Do NOT call finalize() before using it for
    PSNR — finalize() converts the model into a backend-specific
    representation that is no longer runnable as plain PyTorch.

  EXPORT (export_fastvlm.py): Call prepare(example_inputs), then
    finalize(backend=ExportBackend.CoreAI). The finalized model is what
    gets staged into TorchConverter via add_exported_program().

CALIBRATION
-----------
These utilities do NOT perform calibration. Uncalibrated quantization gives
a conservative (pessimistic) estimate of quality sufficient for
characterising whether a quantization level is viable. For production
export, calibration via calibration_mode() using representative
vision-language inputs is a future improvement.

HOW TO USE
----------
For verify scripts (PSNR comparison vs the fp16 Stage 2 output):
    fp16_model = ...            # Stage 2 output
    fp16_out = fp16_model(example_input)
    quantized, _ = apply_quantization(fp32_model, "int8", (example_input,))
    quantized_out = quantized(example_input)
    score = psnr(fp16_out, quantized_out)

For export scripts (before TorchConverter):
    quantized, quantizer = apply_quantization(fp32_model, "int8", (example_input,))
    export_ready = finalize_for_export(quantized, quantizer)
    exported = torch.export.export(export_ready, ...)
    converter.add_exported_program(exported, ...)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from coreai_opt.common import ExportBackend
from coreai_opt.quantization import Quantizer, QuantizerConfig
from coreai_opt.quantization.spec import PerBlockGranularity, QuantizationSpec, QuantizationScheme
from coreai_opt.quantization.config import ModuleQuantizerConfig

# Supported quantization levels. Matches the --quantize CLI flag values
# in verify_decoder.py and export_fastvlm.py. No "fp16" entry here — fp16
# is the mandatory Stage 2 precision cast, handled directly by the verify
# scripts and export_fastvlm.py, not by this module.
QUANTIZATION_LEVELS = [
    "int8",
    "int4",
]

QuantizationLevel = Literal["int8", "int4"]

_DTYPE = {"int8": torch.int8, "int4": torch.int4}


def apply_quantization(
    model: nn.Module,
    level: QuantizationLevel,
    example_inputs: tuple[torch.Tensor, ...],
) -> tuple[nn.Module, Quantizer]:
    """
    Apply the specified quantization level to the FastVLM decoder.

    The input model should be the fp32 Stage 1 port (NOT pre-cast to fp16
    by the caller) — this function performs the fp32 -> fp16 cast itself
    as part of replicating Apple's two-part scheme:
      1. Non-linear tensors (RMSNorm scales, attention biases, etc.) -> fp16.
      2. nn.Linear weight matrices -> grouped asymmetric int8/int4,
         block_size=64 along the input-channel axis, per-group scale +
         zero-point in fp16.

    Returns (quantized_model, quantizer) where:
      - quantized_model is directly runnable for PSNR comparison
        (verify scripts), with fake-quantize modules inserted.
      - quantizer is the coreai-opt object needed for finalize_for_export()
        (export scripts).

    Do NOT call finalize() on quantized_model before using it for PSNR —
    see module docstring.

    Args:
        model: The fp32-verified FastVLM decoder (Stage 1 output). Must
               already be in eval() mode and on CPU.
        level: One of QUANTIZATION_LEVELS ("int8" or "int4").
        example_inputs: Tuple of example input tensors matching the model's
                        forward() signature. Used by coreai-opt to trace the
                        model for fake-quantize module insertion. Content
                        doesn't affect quantization quality for weight-only
                        quantization; random tensors of the right shape are
                        sufficient.

    Returns:
        (quantized_model, quantizer)
    """
    if level not in QUANTIZATION_LEVELS:
        raise ValueError(
            f"Unknown quantization level: {level!r}. "
            f"Must be one of: {QUANTIZATION_LEVELS}"
        )

    model.eval()
    model = model.half()

    weight_spec = QuantizationSpec(
        dtype=_DTYPE[level],
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerBlockGranularity(axis=1, block_size=64),
    )
    linear_config = ModuleQuantizerConfig(
        op_input_spec={},
        op_output_spec={},
        op_state_spec={"weight": weight_spec},
    )
    config = QuantizerConfig(global_config=None).set_module_type(torch.nn.Linear, linear_config)
    quantizer = Quantizer(model, config)
    prepared = quantizer.prepare(example_inputs)
    return prepared, quantizer


def finalize_for_export(
    quantized_model: nn.Module,
    quantizer: Quantizer,
) -> nn.Module:
    """
    Finalize a quantized model for coreai-torch export.

    Call this AFTER apply_quantization() and BEFORE torch.export.export().
    The returned model is in backend-specific form and is no longer suitable
    for plain PyTorch forward passes / PSNR comparison.

    Args:
        quantized_model: The model returned by apply_quantization().
        quantizer: The Quantizer object returned by apply_quantization().

    Returns:
        nn.Module ready to be passed to torch.export.export() and then
        TorchConverter.add_exported_program().
    """
    return quantizer.finalize(
        model=quantized_model,
        backend=ExportBackend.CoreAI,
    )


def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """
    Compute PSNR (dB) between reference and test tensors in float64.

    Both tensors are cast to float64 before comparison to avoid any
    accumulated error from the test tensor's dtype affecting the metric itself.
    Returns float('inf') if MSE is exactly zero (bit-identical).
    """
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
