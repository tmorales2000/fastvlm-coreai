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
  fp16           : Cast model weights to float16. No coreai-opt involved.
                   Equivalent to the former verify_*.py Stage 2.
  int8           : coreai-opt weight-only int8 quantization (linear, per-channel
                   symmetric) via Quantizer.presets.w8().
  int8-palettized: coreai-opt 8-bit k-means palettization via
                   KMeansPalettizerConfig.presets.w8().
  int4           : coreai-opt weight-only int4 quantization (linear, per-channel
                   symmetric) via Quantizer.presets.w4().
  int4-palettized: coreai-opt 4-bit k-means palettization via
                   KMeansPalettizerConfig.presets.w4() (group_size=16 default).

SIMULATION vs EXPORT DISTINCTION (critical)
--------------------------------------------
coreai-opt operates in two distinct modes:

  SIMULATION (verify scripts): Call prepare(example_inputs) only.
    The returned model is a standard nn.Module with fake-quantize or
    fake-palettize modules inserted around weights. It is directly runnable
    in PyTorch for forward passes / PSNR comparison. Do NOT call finalize()
    before using it for PSNR — finalize() converts the model into a
    backend-specific representation that is no longer runnable as plain
    PyTorch. This is documented explicitly in KMeansPalettizer.finalize()'s
    docstring: "For torch-based evaluation, use the model returned by
    prepare() directly rather than calling finalize."

  EXPORT (export_fastvlm.py): Call prepare(example_inputs), then
    finalize(backend=ExportBackend.CoreAI). The finalized model is what
    gets staged into TorchConverter via add_exported_program(). The
    finalize step converts fake-quantize/fake-palettize modules into
    coreai-backend-specific weight representations (lookup tables for
    palettization, quantized weight tensors for quantization) that
    TorchConverter knows how to lower to coreai ops.

CALIBRATION
-----------
These utilities do NOT perform calibration (sensitivity_path, calibration_mode).
For verify scripts this is intentional — uncalibrated compression gives a
conservative (pessimistic) estimate of quality that's sufficient for
characterizing whether a compression level is viable at all. For production
export, consider adding calibration via:
  with compressor.calibration_mode(loss_fn=...):
      for batch in calibration_data:
          ...
after prepare() and before finalize(), using a small representative dataset
of real images. This is left to a future iteration once the basic pipeline
is proven.

HOW TO USE
----------
For verify scripts (PSNR comparison):
    model = FastVLMDecoder.from_weights(...)
    ref_out = model(example_input)  # fp32 reference
    compressed, _ = apply_compression(model, "int4-palettized", (example_input,))
    compressed_out = compressed(example_input)
    psnr = compute_psnr(ref_out, compressed_out)

For export scripts (before TorchConverter):
    model = FastVLMDecoder.from_weights(...)
    compressed, compressor = apply_compression(model, "int4-palettized", (example_input,))
    export_ready = finalize_for_export(compressed, compressor)
    exported = torch.export.export(export_ready, ...)
    converter.add_exported_program(exported, ...)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from coreai_opt.common import ExportBackend
from coreai_opt.palettization.config import KMeansPalettizerConfig
from coreai_opt.palettization.kmeans import KMeansPalettizer
from coreai_opt.quantization import Quantizer, QuantizerConfig

# All supported compression levels. Matches the --compression CLI flag values
# in verify_*.py and export_fastvlm.py.
COMPRESSION_LEVELS = [
    "fp16",
    "int8",
    "int8-palettized",
    "int4",
    "int4-palettized",
]

CompressionLevel = Literal[
    "fp16",
    "int8",
    "int8-palettized",
    "int4",
    "int4-palettized",
]


def apply_compression(
    model: nn.Module,
    level: CompressionLevel,
    example_inputs: tuple[torch.Tensor, ...],
) -> tuple[nn.Module, KMeansPalettizer | Quantizer | None]:
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
                        model for fake-quantize/palettize module insertion.
                        Content doesn't affect compression quality for
                        weight-only compression (int8/int4/palettized);
                        random tensors of the right shape are sufficient.

    Returns:
        (compressed_model, compressor):
          compressed_model: nn.Module with compression applied in simulation mode.
          compressor: KMeansPalettizer | Quantizer | None — needed for export.
    """
    model.eval()

    if level == "fp16":
        # Plain cast — no coreai-opt. Equivalent to the former verify_*.py
        # Stage 2 fp16 health check. Returns a copy so the caller's fp32
        # reference model is not modified in place.
        return model.half(), None

    if level == "int8":
        # axis=0 is specified explicitly rather than using the preset default
        # of axis=None. With axis=None, coreai-opt tries to auto-resolve the
        # quantization axis from the module type — but this fails for our
        # custom composite ops (RoPE, SDPA, RMSNorm) which are not nn.Linear
        # or nn.Conv2d and have no registered axis defaults. axis=0 is the
        # correct output-channel axis for nn.Linear weight matrices (shape
        # [out_features, in_features]) and is applied uniformly across all
        # module types without requiring auto-resolution.
        compressor = Quantizer(model, QuantizerConfig.presets.w8(axis=0))
        prepared = compressor.prepare(example_inputs)
        return prepared, compressor

    if level == "int8-palettized":
        compressor = KMeansPalettizer(model, KMeansPalettizerConfig.presets.w8())
        prepared = compressor.prepare(example_inputs)
        return prepared, compressor

    if level == "int4":
        # Same axis=0 rationale as int8 above.
        compressor = Quantizer(model, QuantizerConfig.presets.w4(axis=0))
        prepared = compressor.prepare(example_inputs)
        return prepared, compressor

    if level == "int4-palettized":
        compressor = KMeansPalettizer(model, KMeansPalettizerConfig.presets.w4())
        prepared = compressor.prepare(example_inputs)
        return prepared, compressor

    raise ValueError(
        f"Unknown compression level: {level!r}. "
        f"Must be one of: {COMPRESSION_LEVELS}"
    )


def finalize_for_export(
    compressed_model: nn.Module,
    compressor: KMeansPalettizer | Quantizer | None,
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
        # fp16 — no finalization needed.
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
    return 20.0 * torch.log10(torch.tensor(max_val)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()
