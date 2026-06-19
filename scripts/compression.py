"""
compression.py — Shared compression utility for FastVLM verify and export scripts.

PURPOSE
-------
Provides a single, consistent interface for applying coreai-opt compression
to any FastVLM component (vision encoder, projector, decoder), used by:

  - verify_decoder.py, verify_projector.py, verify_vision_encoder.py
    (--compression flag: simulate compression, compare PSNR against fp32 reference)
  - export_fastvlm.py
    (--compression flag: apply compression before staging into TorchConverter)

SUPPORTED COMPRESSION LEVELS
------------------------------
  fp16 : Cast model weights to float16. No coreai-opt involved.
         Equivalent to the former verify_*.py Stage 2.
  int8 : coreai-opt weight-only int8 quantization, grouped asymmetric,
         block_size=64 along input-channel axis. Matches Apple's MLX
         int8 scheme for FastVLM 1.5B.
  int4 : coreai-opt weight-only int4 quantization, same scheme.
         Matches Apple's MLX int4 scheme for FastVLM 7B.

PRODUCTION COMPRESSION TARGETS (validated)
-------------------------------------------
  0.5B decoder : fp16  (Apple ships unquantized; no quality benefit to quantize)
  1.5B decoder : int8  (46.2 dB vs fp16, PASS; matches Apple MLX quality tier)
  7B   decoder : int8  (int4 fails at 22.4 dB due to fp16 pre-cast sensitivity)
  projector    : fp16  (Apple never quantizes it; 14M params, negligible size)
  vision enc.  : fp16  (CoreML export; quantization not applied)

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
    gets staged into TorchConverter via add_exported_program(). The
    finalize step converts fake-quantize modules into coreai-backend-specific
    weight representations that TorchConverter knows how to lower to coreai ops.

CALIBRATION
-----------
These utilities do NOT perform calibration. For verify scripts this is
intentional — uncalibrated compression gives a conservative (pessimistic)
estimate of quality sufficient for characterising whether a compression level
is viable. For production export, calibration via calibration_mode() using
representative vision-language inputs is a future improvement.

HOW TO USE
----------
For verify scripts (PSNR comparison):
    model = FastVLMDecoder.from_weights(...)
    ref_out = model(example_input)       # fp32 reference
    compressed, _ = apply_compression(model, "int8", (example_input,))
    compressed_out = compressed(example_input)
    score = psnr(ref_out, compressed_out)

For export scripts (before TorchConverter):
    model = FastVLMDecoder.from_weights(...)
    compressed, compressor = apply_compression(model, "int8", (example_input,))
    export_ready = finalize_for_export(compressed, compressor)
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

# All supported compression levels. Matches the --compression CLI flag values
# in verify_*.py and export_fastvlm.py.
COMPRESSION_LEVELS = [
    "fp16",
    "int8",
    "int4",
]

CompressionLevel = Literal["fp16", "int8", "int4"]


def apply_compression(
    model: nn.Module,
    level: CompressionLevel,
    example_inputs: tuple[torch.Tensor, ...],
) -> tuple[nn.Module, Quantizer | None]:
    """
    Apply the specified compression level to a FastVLM component.

    Returns (compressed_model, compressor) where:
      - compressed_model is directly runnable for PSNR comparison (verify scripts)
      - compressor is the coreai-opt object needed for finalize_for_export()
        (export scripts). None for fp16, which has no coreai-opt object.

    The returned compressed_model should NOT have finalize() called on it before
    using it for PSNR — see module docstring.

    Args:
        model: The fp32-verified FastVLM component (decoder, projector, or
               vision encoder). Must already be in eval() mode and on CPU.
        level: Compression level. One of COMPRESSION_LEVELS.
        example_inputs: Tuple of example input tensors matching the model's
                        forward() signature. Used by coreai-opt to trace the
                        model for fake-quantize module insertion.
                        Content doesn't affect compression quality for
                        weight-only compression; random tensors of the right
                        shape are sufficient.

    Returns:
        (compressed_model, compressor):
          compressed_model: nn.Module with compression applied in simulation mode.
          compressor: Quantizer | None — needed for export.
    """
    model.eval()

    if level == "fp16":
        # Plain cast — no coreai-opt. Returns a copy so the caller's fp32
        # reference model is not modified in place.
        return model.half(), None

    if level == "int8":
        # Matches Apple's MLX int8 scheme (FastVLM 1.5B):
        #   1. Non-linear tensors (RMSNorm scales, attention biases) → fp16.
        #   2. nn.Linear weight matrices: grouped asymmetric int8,
        #      block_size=64 along input-channel axis, per-group scale +
        #      zero-point in fp16.
        # PSNR reflects total quality delta from fp32 (fp16 rounding + int8).
        model = model.half()
        weight_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerBlockGranularity(axis=1, block_size=64),
        )
        linear_config = ModuleQuantizerConfig(
            op_input_spec={},
            op_output_spec={},
            op_state_spec={"weight": weight_spec},
        )
        config = QuantizerConfig(global_config=None).set_module_type(torch.nn.Linear, linear_config)
        compressor = Quantizer(model, config)
        prepared = compressor.prepare(example_inputs)
        return prepared, compressor

    if level == "int4":
        # Same two-part scheme as int8, matching Apple's MLX 7B:
        # fp16 for non-linear tensors, grouped asymmetric int4 for nn.Linear
        # weights (block_size=64, axis=1).
        # PSNR reflects total quality delta from fp32 (fp16 rounding + int4).
        model = model.half()
        weight_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerBlockGranularity(axis=1, block_size=64),
        )
        linear_config = ModuleQuantizerConfig(
            op_input_spec={},
            op_output_spec={},
            op_state_spec={"weight": weight_spec},
        )
        config = QuantizerConfig(global_config=None).set_module_type(torch.nn.Linear, linear_config)
        compressor = Quantizer(model, config)
        prepared = compressor.prepare(example_inputs)
        return prepared, compressor

    raise ValueError(
        f"Unknown compression level: {level!r}. "
        f"Must be one of: {COMPRESSION_LEVELS}"
    )


def finalize_for_export(
    compressed_model: nn.Module,
    compressor: Quantizer | None,
) -> nn.Module:
    """
    Finalize a compressed model for coreai-torch export.

    Call this AFTER apply_compression() and BEFORE torch.export.export().
    The returned model is in backend-specific form and is no longer suitable
    for plain PyTorch forward passes / PSNR comparison.

    For fp16 (compressor=None), returns the model unchanged since .half()
    already produces an export-ready model.

    Args:
        compressed_model: The model returned by apply_compression().
        compressor: The compressor object returned by apply_compression().
                    None for fp16.

    Returns:
        nn.Module ready to be passed to torch.export.export() and then
        TorchConverter.add_exported_program().
    """
    if compressor is None:
        return compressed_model

    return compressor.finalize(
        model=compressed_model,
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
