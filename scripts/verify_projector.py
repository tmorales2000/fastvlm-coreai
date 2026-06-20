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
    original HF weights) — isolates the fp16 cast's own error. Apple never
    quantizes the projector (~14M params; negligible size impact) and
    neither do we — there is no Stage 3 for the projector.

VALIDATED PRODUCTION TARGET
-----------------------------
  All variants : Stage 2 only, fp16, no quantization.

USAGE
-----
  python scripts/verify_projector.py --variant 1.5b
"""

import argparse
import sys

import torch
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from quantization import psnr  # noqa: E402
from fastvlm_projector import FastVLMProjector, _load_projector_weights  # noqa: E402

# Pass threshold (dB). Engineering judgment, not an Apple specification.
PRECISION_PASS = 60.0


def _build_port(config, weights_dir: str, dtype: torch.dtype) -> FastVLMProjector:
    model = FastVLMProjector(config).to(dtype=dtype)
    weights = _load_projector_weights(weights_dir, dtype=dtype)
    model.load_state_dict(weights, assign=True, strict=True)
    model.eval()
    return model


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


# ─── Driver ───────────────────────────────────────────────────────────────────


def verify(variant: str) -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying projector: {variant} ({weights_dir})")
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

    passed, fp32_ref, test_input = stage_correctness(config, weights_dir)
    if not passed:
        print("\n>>> Stage 1 FAILED.")
        sys.exit(1)

    passed = stage_precision(config, weights_dir, fp32_ref, test_input)

    print("\n" + "=" * 56)
    if passed:
        print("STAGES 1-2 PASS — safe to export at fp16.")
    else:
        print("STAGE 2 FAILED.")
    print("=" * 56)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Verify FastVLM projector correctness and precision.",
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    args = ap.parse_args()
    verify(args.variant)
