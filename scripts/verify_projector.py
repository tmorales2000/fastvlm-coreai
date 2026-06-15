"""
verify_projector.py — Two-stage verification for the re-authored FastVLM projector.

PURPOSE
-------
This script answers two questions in order:

  1. Is the re-authored projector mathematically equivalent to the original?
  2. Is fp16 precision acceptable?

Unlike the decoder, the projector has no KV cache, no autoregressive loop, and
no stateful components. Verification is a single forward pass comparison.

STAGE 1 — CORRECTNESS (fp32, port vs original HF mm_projector)
---------------------------------------------------------------
Loads the re-authored FastVLMProjector and the original model.mm_projector
from HuggingFace side by side, both in fp32, and compares their outputs on
the same random input.

Why fp32?
  Same reason as the decoder: fp32 eliminates dtype as a variable. Any
  divergence is structural — wrong weight mapping, wrong layer order, wrong
  activation. Since the projector is only 2-3 layers, fp32 errors would be
  obvious even at small magnitude.

Why load the original HF model?
  The 90.9 dB result from the original verify_projector.py compared our fp16
  port against our fp32 port of the same code. That is a self-consistency
  check — it cannot catch a wrong weight mapping or wrong architecture.
  This stage loads the actual original projector from the FastVLM checkpoint
  via AutoModelForCausalLM and compares against it.

PSNR threshold:
  > 80 dB  PASS    — confirmed structural equivalence.
  50-80 dB MARGINAL — weight mapping likely wrong; check key remapping.
  < 50 dB  FAIL    — architecture mismatch.

These thresholds are engineering judgments. In practice a correct port of a
2-3 layer MLP in fp32 should achieve >>100 dB (essentially bit-identical).
Any result below 80 dB indicates something is structurally wrong.

STAGE 2 — fp16 HEALTH
----------------------
Runs the fp16 port and checks for NaN, Inf, and overflow. Compares against
the fp32 port (not the HF original — Stage 1 already confirmed equivalence).

PSNR threshold:
  > 60 dB  PASS   — fp16 precision loss is acceptable for a 2-3 layer MLP.
  < 60 dB  FAIL   — unexpected precision loss; check for unbounded activations.

USAGE
-----
  python scripts/verify_projector.py
  python scripts/verify_projector.py --variant 0.5b
  python scripts/verify_projector.py --stage correctness
  python scripts/verify_projector.py --stage fp16

ARGUMENTS
---------
  --variant   Which FastVLM variant to test. Default: 1.5b.
              Choices: 0.5b, 1.5b, 7b.

  --stage     Which stage to run. Default: all.
              Choices: all, correctness, fp16.

EXIT CODES
----------
  0  All requested stages passed.
  1  At least one stage failed.
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, "scripts")
from fastvlm_projector import FastVLMProjector, _load_projector_weights

# ─── Thresholds ───────────────────────────────────────────────────────────────
# Engineering judgments. A correct 2-3 layer MLP port in fp32 should be >>100 dB.
CORRECTNESS_PASS     = 80.0   # dB — fp32 cross-model; expect >>100 dB
CORRECTNESS_MARGINAL = 50.0   # dB — below here is definitely wrong
FP16_PASS            = 60.0   # dB — fp16 vs fp32 self-consistency

# fp16 ceiling is 65504. Flag values above this as overflow risk.
FP16_OVERFLOW_THRESHOLD = 60000.0


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Peak signal-to-noise ratio between two tensors, computed in fp32."""
    a_f, b_f = a.float(), b.float()
    mse = ((a_f - b_f) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    peak = b_f.abs().max().item() ** 2
    if peak == 0:
        return float("inf")
    return 10 * np.log10(peak / mse)


# ─── Stage 1: cross-model correctness ────────────────────────────────────────


def stage_correctness(config, weights_dir: str) -> bool:
    print("\n" + "=" * 56)
    print("STAGE 1 — CORRECTNESS (fp32, port vs HF mm_projector)")
    print("=" * 56)

    # Load original HF model and extract mm_projector
    # trust_remote_code required — FastVLM uses custom model code in llava_qwen.py
    print("Loading original HF model (this may take a moment)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    hf_proj = hf_model.model.mm_projector.to(torch.float32).eval()
    del hf_model  # free memory — we only need the projector

    # Load re-authored port in fp32
    weights_f32 = _load_projector_weights(weights_dir, dtype=torch.float32)
    port = FastVLMProjector(config).to(torch.float32)
    port.load_state_dict(weights_f32, assign=True, strict=True)
    port.eval()

    # Same random input for both
    torch.manual_seed(0)
    B, N = 1, 256
    x = torch.randn(B, N, config.mm_hidden_size, dtype=torch.float32)

    with torch.no_grad():
        hf_out   = hf_proj(x)
        port_out = port(x)

    score = psnr(port_out, hf_out)
    print(f"\nInput shape  : {x.shape}")
    print(f"Output shape : {port_out.shape}")
    print(f"PSNR fp32 port vs HF original: {score:.1f} dB")

    if score > CORRECTNESS_PASS:
        print(f"\n[PASS] {score:.1f} dB — port matches original HF mm_projector.")
        return True
    if score > CORRECTNESS_MARGINAL:
        print(
            f"\n[MARGINAL] {score:.1f} dB. Weight mapping is likely wrong. "
            "Check: key remapping (model.mm_projector.N.* → layers.N.*), "
            "layer order in nn.Sequential, activation type."
        )
        return False
    print(
        f"\n[FAIL] {score:.1f} dB — architecture mismatch. "
        "Check mm_projector_type parsing and layer construction."
    )
    return False


# ─── Stage 2: fp16 health ─────────────────────────────────────────────────────


def stage_fp16(config, weights_dir: str) -> bool:
    print("\n" + "=" * 56)
    print("STAGE 2 — fp16 HEALTH (fp16 port vs fp32 port)")
    print("=" * 56)

    # Stage 1 confirmed the port matches HF. Stage 2 confirms fp16 precision.
    weights_f32 = _load_projector_weights(weights_dir, dtype=torch.float32)
    port_f32 = FastVLMProjector(config).to(torch.float32)
    port_f32.load_state_dict(weights_f32, assign=True, strict=True)
    port_f32.eval()

    port_fp16 = FastVLMProjector.from_weights(config, weights_dir)
    port_fp16.eval()

    torch.manual_seed(0)
    B, N = 1, 256
    x = torch.randn(B, N, config.mm_hidden_size, dtype=torch.float32)

    with torch.no_grad():
        out_f32  = port_f32(x)
        out_fp16 = port_fp16(x.half())

    has_nan = torch.isnan(out_fp16).any().item()
    has_inf = torch.isinf(out_fp16).any().item()
    max_abs = out_fp16.abs().max().item()
    overflow_risk = max_abs > FP16_OVERFLOW_THRESHOLD

    score = psnr(out_fp16, out_f32)

    print(f"\nfp16 NaN / Inf   : {has_nan} / {has_inf}")
    print(f"fp16 max |output| : {max_abs:.2f}  (fp16 ceiling 65504)")
    print(f"PSNR fp16 vs fp32 : {score:.1f} dB")

    if has_nan or has_inf:
        print("\n[FAIL] NaN/Inf in fp16 output.")
        return False
    if overflow_risk:
        print(
            f"\n[FAIL] fp16 output near saturation ({max_abs:.0f} vs "
            f"threshold {FP16_OVERFLOW_THRESHOLD:.0f})."
        )
        return False
    if score > FP16_PASS:
        print(f"\n[PASS] {score:.1f} dB — fp16 precision acceptable.")
        return True
    print(f"\n[FAIL] {score:.1f} dB < {FP16_PASS} dB threshold.")
    return False


# ─── Driver ───────────────────────────────────────────────────────────────────


def verify(variant: str, stage: str) -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying projector: {variant} ({weights_dir})")
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

    if stage == "correctness":
        sys.exit(0 if stage_correctness(config, weights_dir) else 1)
    if stage == "fp16":
        sys.exit(0 if stage_fp16(config, weights_dir) else 1)

    if not stage_correctness(config, weights_dir):
        print("\n>>> Stopped at Stage 1. Fix correctness before running Stage 2.")
        sys.exit(1)
    if not stage_fp16(config, weights_dir):
        print("\n>>> Stopped at Stage 2. Architecture correct; fp16 precision not.")
        sys.exit(1)

    print("\n" + "=" * 56)
    print("ALL STAGES PASS — projector is ready to export.")
    print("=" * 56)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Two-stage verification for the re-authored FastVLM projector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
stages:
  all          Run Stage 1 then Stage 2. Stop at first failure. (default)
  correctness  Stage 1 only: fp32 port vs original HF mm_projector.
  fp16         Stage 2 only: fp16 vs fp32 self-consistency and health check.

examples:
  python scripts/verify_projector.py
  python scripts/verify_projector.py --variant 0.5b
  python scripts/verify_projector.py --stage correctness
""",
    )
    ap.add_argument(
        "--variant",
        default="1.5b",
        choices=["0.5b", "1.5b", "7b"],
        help="FastVLM variant. Weights must be at weights/fastvlm-{variant}/. (default: 1.5b)",
    )
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "correctness", "fp16"],
        help="Which stage to run. (default: all)",
    )
    args = ap.parse_args()
    verify(args.variant, args.stage)
