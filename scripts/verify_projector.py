"""
verify_projector.py — FastVLM projector verification.

STAGES
------
Stage 1 — CORRECTNESS (HF weights -> fp32 port)
    Load the projector weights (mlp2x_gelu: layers.0 Linear -> layers.1 GELU
    -> layers.2 Linear) faithfully into fp32 via FastVLMProjector + the
    shared _load_projector_weights() key-mapping logic. Unlike the decoder,
    there's no independent HF reference implementation to diff against — the
    projector is a direct weight load with no re-architecture — so this
    stage is a structural sanity check (shapes, key coverage, clean forward
    pass) rather than a cross-model PSNR comparison.

Stage 2 — PRECISION (fp32 -> fp16)
    Cast the fp32 port to float16 — the mandatory ANE execution precision.
    NOT optional, NOT "compression": every projector export goes through
    this stage. PSNR is measured against the Stage 1 fp32 port (not
    original HF weights) — isolates the fp16 cast's own error.

Stage 3 — QUANTIZATION (fp16 -> int8 | int4), optional
    Apply coreai-opt weight-only quantization to BOTH projector Linear
    layers (layers.0 and layers.2). PSNR measured against Stage 2's fp16
    output. Mutually exclusive: int8 or int4, no "all".

    CORRECTION (June 2026): an earlier version of this script and its
    docstring claimed "Apple never quantizes the projector" and had no
    Stage 3 at all. This was WRONG — verified false by an exhaustive
    tensor-by-tensor audit (scripts/audit_weight_dtypes.py) of Apple's
    actual shipped MLX weights. Both multi_modal_projector.linear_0.weight
    and linear_2.weight are quantized in Apple's 1.5b (int8) and 7b (int4)
    MLX checkpoints, identically to how the decoder's Linear layers are
    quantized (same group_size=64 asymmetric scheme). Only the Linear
    biases are excluded from quantization, consistent with every other
    component. The assumption was never verified directly against the
    weight files until this audit — a reminder to verify claims about
    Apple's pipeline against actual bytes, not just the public README.

VALIDATED PRODUCTION TARGETS
-----------------------------
  0.5B : Stage 2 only, fp16, no quantization (Apple ships 0.5B fully
         unquantized across every component, decoder included).
  1.5B : Stage 3 int8 (matches decoder's int8 target).
  7B   : Stage 3 int8 (matches decoder's int8 target — int4 not attempted
         here yet given the decoder's int4 instability; revisit together).

USAGE
-----
  python scripts/verify_projector.py --variant 1.5b
  python scripts/verify_projector.py --variant 1.5b --quantize int8
  python scripts/verify_projector.py --variant 1.5b --quantize int4
"""

import argparse
import sys

import torch
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from quantization import QUANTIZATION_LEVELS, apply_quantization, psnr  # noqa: E402
from fastvlm_projector import FastVLMProjector, _load_projector_weights  # noqa: E402

# Pass thresholds (dB). Engineering judgments, not Apple specifications.
PRECISION_PASS = 60.0
QUANTIZATION_PASS = 40.0


def _build_port(config, weights_dir: str, dtype: torch.dtype) -> FastVLMProjector:
    model = FastVLMProjector(config).to(dtype=dtype)
    weights = _load_projector_weights(weights_dir, dtype=dtype)
    model.load_state_dict(weights, assign=True, strict=True)
    model.eval()
    return model


def _example_inputs(config) -> tuple[torch.Tensor, ...]:
    """
    Minimal example input for coreai-opt tracing during Stage 3 preparation.

    dtype is float16, NOT float32 -- apply_quantization() always casts the
    model to fp16 internally before quantizing (matching Apple's bf16->fp16
    pipeline for non-quantized tensors). The traced example input's dtype
    must match the model's post-cast dtype or the first matmul inside
    quantizer.prepare() fails with a Float/Half mismatch (as it did here:
    the decoder's _example_inputs uses int32 input_ids/position_ids, which
    are unaffected by .half(), so this mismatch never surfaced there --
    it's specific to any port whose example input is itself a floating
    point activation tensor, like the projector's).
    """
    torch.manual_seed(0)
    B, seq_len = 1, 256
    x = torch.randn(B, seq_len, config.mm_hidden_size, dtype=torch.float16)
    return (x,)


# ─── Stage 1: correctness (HF weights -> fp32 port) ────────────────────────────


def stage_correctness(config, weights_dir: str) -> tuple[bool, torch.Tensor, torch.Tensor]:
    """
    Stage 1 — CORRECTNESS. Loads projector weights into fp32 via
    FastVLMProjector.from_weights()-equivalent path and confirms the load
    is structurally complete (strict=True in _build_port already enforces
    full key coverage). Returns (passed, fp32_port_out, test_input) — both
    reused by Stage 2.
    """
    print("\n" + "=" * 56)
    print("STAGE 1 — CORRECTNESS (HF weights -> fp32 port)")
    print("=" * 56)

    port = _build_port(config, weights_dir, torch.float32)

    B, seq_len = 1, 256
    torch.manual_seed(0)
    x = torch.randn(B, seq_len, config.mm_hidden_size, dtype=torch.float32)

    with torch.no_grad():
        out = port(x)

    print(f"\nProjector type   : {config.mm_projector_type}")
    print(f"mm_hidden_size   : {config.mm_hidden_size}")
    print(f"hidden_size      : {config.hidden_size}")
    print(f"Output shape     : {list(out.shape)}")
    print("\n[PASS] fp32 port loaded (strict key match) and runs cleanly.")
    return True, out, x


# ─── Stage 2: precision (fp32 -> fp16) ─────────────────────────────────────────


def stage_precision(
    config,
    weights_dir: str,
    fp32_ref: torch.Tensor,
    test_input: torch.Tensor,
) -> bool:
    """
    Stage 2 — PRECISION. Cast to fp16 and compare against the Stage 1 fp32
    port (not original HF weights) to isolate the fp16 cast's own error.
    """
    print("\n" + "=" * 56)
    print("STAGE 2 — PRECISION (fp32 -> fp16)")
    print("=" * 56)

    port_fp16 = _build_port(config, weights_dir, torch.float16)

    with torch.no_grad():
        out_fp16 = port_fp16(test_input.to(torch.float16))

    has_nan = torch.isnan(out_fp16).any().item()
    has_inf = torch.isinf(out_fp16).any().item()
    score = psnr(fp32_ref, out_fp16.float())

    print(f"\nfp16 NaN / Inf : {has_nan} / {has_inf}")
    print(f"PSNR vs fp32 (Stage 1) port: {score:.1f} dB")

    if has_nan or has_inf:
        print("\n[FAIL] NaN/Inf in fp16 output.")
        return False
    if score > PRECISION_PASS:
        print(f"\n[PASS] {score:.1f} dB > {PRECISION_PASS:.0f} dB threshold.")
        return True
    print(f"\n[FAIL] {score:.1f} dB <= {PRECISION_PASS:.0f} dB threshold.")
    return False


# ─── Stage 3: quantization (fp16 -> int8 | int4), optional ───────────────────


def stage_quantization(config, weights_dir: str, level: str) -> bool:
    """
    Stage 3 — QUANTIZATION. Applies int8 or int4 weight-only quantization
    to both projector Linear layers (layers.0, layers.2), matching Apple's
    verified scheme (see module docstring correction note). PSNR is
    measured against a freshly-built Stage 2 fp16 reference, isolating
    quantization-only error.
    """
    print("\n" + "=" * 56)
    print(f"STAGE 3 — QUANTIZATION ({level.upper()}, fp16 -> {level})")
    print("=" * 56)

    port = _build_port(config, weights_dir, torch.float32)
    example_inputs = _example_inputs(config)  # fp16 -- see docstring above
    x = example_inputs[0]

    print(f"Applying {level} quantization...")
    quantized, _ = apply_quantization(port, level, example_inputs)

    with torch.no_grad():
        quantized_out = quantized(x)

    fp16_port = _build_port(config, weights_dir, torch.float16)
    with torch.no_grad():
        fp16_ref_out = fp16_port(x)

    score = psnr(quantized_out.float(), fp16_ref_out.float())
    print(f"\nPSNR vs fp16 (Stage 2) reference : {score:.1f} dB")

    if score > QUANTIZATION_PASS:
        print(f"[PASS] {score:.1f} dB — {level} quantization viable.")
        return True
    print(
        f"[FAIL] {score:.1f} dB — unacceptable quality loss at {level}. "
        f"Consider a less aggressive quantization level."
    )
    return False


# ─── Driver ───────────────────────────────────────────────────────────────────


def verify(variant: str, quantize: str | None) -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying projector: {variant} ({weights_dir})")
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

    passed, fp32_ref, test_input = stage_correctness(config, weights_dir)
    if not passed:
        print("\n>>> Stage 1 FAILED.")
        sys.exit(1)

    passed = stage_precision(config, weights_dir, fp32_ref, test_input)
    if not passed:
        print("\n>>> Stage 2 FAILED. Fix fp16 precision issues before Stage 3.")
        sys.exit(1)

    if not quantize:
        print("\n" + "=" * 56)
        print("STAGES 1-2 PASS — safe to export at fp16.")
        print("=" * 56)
        sys.exit(0)

    q_passed = stage_quantization(config, weights_dir, quantize)

    print("\n" + "=" * 56)
    if q_passed:
        print(f"STAGES 1-3 PASS — safe to export at {quantize}.")
    else:
        print(f"STAGE 3 ({quantize}) FAILED.")
    print("=" * 56)
    sys.exit(0 if q_passed else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Verify FastVLM projector correctness, precision, and quantization quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/verify_projector.py --variant 1.5b
  python scripts/verify_projector.py --variant 1.5b --quantize int8
  python scripts/verify_projector.py --variant 1.5b --quantize int4
""",
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument(
        "--quantize",
        choices=QUANTIZATION_LEVELS,
        default=None,
        help="Run Stage 3 at this quantization level after Stages 1-2 pass. "
             "Default: none (Stages 1-2 only, fp16 export target).",
    )
    args = ap.parse_args()
    verify(args.variant, args.quantize)
