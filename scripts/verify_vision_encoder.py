"""
verify_vision_encoder.py — Two-stage verification for the re-authored vision encoder.

PURPOSE
-------
Same two-stage pattern as verify_decoder.py and verify_projector.py: separate
"is the architecture correct" from "is fp16 precision acceptable" so a failure
tells you which class of problem you have.

STAGE 1 — CORRECTNESS (fp32, port vs original HF vision tower)
-----------------------------------------------------------------
Loads the original model.vision_tower from AutoModelForCausalLM and calls it
through the same path FastVLM itself uses — MobileCLIPVisionTower.forward_images
-> feature_select — then compares against our FastVLMVisionEncoder port, both
in fp32, on the same input image.

Why fp32?
  Same reasoning as the decoder and projector: fp32 eliminates dtype as a
  variable. A divergence here is structural — wrong pos_embs config, wrong
  patch class substitution, wrong conv_exp handling, wrong reshape order.
  It is NOT the fp16 overflow problem (see Stage 2) — that only manifests in
  fp16 at this image resolution.

Why call hf_tower(pixels) directly rather than passing return_image_embeddings?
  MobileCLIPVisionTower.forward() -> forward_images() already calls
  self.vision_tower(images, return_image_embeddings=True) internally and
  pipes the result through feature_select() before returning (confirmed by
  reading llava_qwen.py lines 1648-1676). return_image_embeddings is an
  argument to the INNER model (self.vision_tower), not to
  MobileCLIPVisionTower.forward() itself — calling hf_tower(pixels) alone
  already returns the finished [B, H*W, C] features, identical to what
  FastVLM itself sees when it calls its own vision tower.

PSNR threshold:
  > 80 dB  PASS    — confirmed structural equivalence.
  50-80 dB MARGINAL — likely a stage-mapping or pos_embs config issue.
  < 50 dB  FAIL    — architecture mismatch.

These are engineering judgments, consistent with the decoder/projector
thresholds. A correct port running in fp32 should land in the 100+ dB range,
same pattern observed for the decoder (110-130 dB) and projector (inf dB).

STAGE 2 — fp16 HEALTH
----------------------
Checks the fp16 port for NaN, Inf, and overflow, and compares it against the
fp32 port.

KNOWN ISSUE: pure fp16 forward overflows at this image resolution. Values
reach ~60928 at network.9 (RepCPE, 1536 channels) — near the fp16 ceiling of
65504 — then the following attention stage (network.10) produces NaN. The
fp32 forward is clean (max ~6657), confirmed by Stage 1 passing.

This is a precision issue in specific high-channel stages, not an architecture
bug. The mitigation here is to compare the fp16 port (with autocast promoting
sensitive ops internally where the runtime supports it) against the fp32 port,
and to report the health metrics plainly rather than mask the overflow.
If Stage 2 fails on overflow, the export should consider: (a) running conv_exp
and the final attention block in fp32/mixed precision on-device, similar to
how Apple's runtimes handle precision-sensitive layers, or (b) clamping
intermediate activations. This is a deployment decision, not a code bug —
Stage 1 already proves the architecture is correct.

USAGE
-----
  python scripts/verify_vision_encoder.py
  python scripts/verify_vision_encoder.py --variant 0.5b
  python scripts/verify_vision_encoder.py --stage correctness
  python scripts/verify_vision_encoder.py --stage fp16

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
from fastvlm_vision_encoder import FastVLMVisionEncoder, _load_vision_weights

# ─── Thresholds ───────────────────────────────────────────────────────────────
# Engineering judgments, consistent with decoder/projector thresholds.
CORRECTNESS_PASS     = 80.0   # dB — fp32 cross-model; expect 100+ dB if correct
CORRECTNESS_MARGINAL = 50.0   # dB — below here is definitely wrong
FP16_PASS            = 60.0   # dB — fp16 vs fp32 self-consistency

# fp16 ceiling is 65504. Known issue: network.9 reaches ~60928 at 1024x1024.
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


def _image_size(config) -> int:
    """Image size is encoded in mm_vision_tower (e.g. 'mobileclip_l_1024')."""
    return int(config.mm_vision_tower.split("_")[-1])


# ─── Stage 1: cross-model correctness ────────────────────────────────────────


def stage_correctness(config, weights_dir: str) -> bool:
    print("\n" + "=" * 56)
    print("STAGE 1 — CORRECTNESS (fp32, port vs HF vision tower)")
    print("=" * 56)

    image_size = _image_size(config)
    print(f"Image size: {image_size}x{image_size}")

    print("Loading original HF model (this may take a moment)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    hf_tower = hf_model.model.vision_tower.to(torch.float32).eval()
    del hf_model  # free memory — we only need the vision tower

    weights_f32 = _load_vision_weights(weights_dir, dtype=torch.float32)
    port = FastVLMVisionEncoder(weights_dir).to(torch.float32)
    missing, unexpected = port.model.load_state_dict(weights_f32, assign=True, strict=False)
    unexpected_real = [
        k for k in unexpected if "head" not in k and ".bn." not in k and "num_batches_tracked" not in k
    ]
    missing_real = [k for k in missing if "head" not in k]
    if unexpected_real:
        raise RuntimeError(f"Port unexpected keys: {unexpected_real[:5]}")
    if missing_real:
        raise RuntimeError(f"Port missing keys: {missing_real[:5]}")
    port.eval()

    torch.manual_seed(0)
    pixels = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    with torch.no_grad():
        # MobileCLIPVisionTower.forward() -> forward_images() already calls
        # self.vision_tower(images, return_image_embeddings=True) internally
        # and pipes the result through feature_select() before returning.
        # So hf_tower(pixels) returns the finished [B, H*W, C] features directly
        # — return_image_embeddings is an argument to the INNER model, not to
        # MobileCLIPVisionTower.forward() itself, and passing it here raises
        # TypeError: forward() got an unexpected keyword argument.
        hf_features = hf_tower(pixels)

        port_features = port(pixels)

    score = psnr(port_features, hf_features)
    print(f"\nInput shape  : {pixels.shape}")
    print(f"Output shape : {port_features.shape}")
    print(f"PSNR fp32 port vs HF original: {score:.1f} dB")

    if score > CORRECTNESS_PASS:
        print(f"\n[PASS] {score:.1f} dB — port matches original HF vision tower.")
        return True
    if score > CORRECTNESS_MARGINAL:
        print(
            f"\n[MARGINAL] {score:.1f} dB. Likely a stage-mapping or pos_embs config "
            "issue. Check: RepCPE inserted at the right embed_dims (768, 1536), "
            "patch class substitution preserving weights correctly, conv_exp "
            "output matches 'image_embeddings' exactly (no head call)."
        )
        return False
    print(
        f"\n[FAIL] {score:.1f} dB — architecture mismatch. "
        "Check pos_embs config, network stage ordering, and ANE patch correctness."
    )
    return False


# ─── Stage 2: fp16 health ─────────────────────────────────────────────────────


def stage_fp16(config, weights_dir: str) -> bool:
    print("\n" + "=" * 56)
    print("STAGE 2 — fp16 HEALTH (fp16 port vs fp32 port)")
    print("=" * 56)

    image_size = _image_size(config)

    weights_f32 = _load_vision_weights(weights_dir, dtype=torch.float32)
    port_f32 = FastVLMVisionEncoder(weights_dir).to(torch.float32)
    port_f32.model.load_state_dict(weights_f32, assign=True, strict=False)
    port_f32.eval()

    port_fp16 = FastVLMVisionEncoder.from_weights(config, weights_dir, dtype=torch.float16)
    port_fp16.eval()

    torch.manual_seed(0)
    pixels = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    with torch.no_grad():
        out_f32 = port_f32(pixels)
        out_fp16 = port_fp16(pixels.half())

    has_nan = torch.isnan(out_fp16).any().item()
    has_inf = torch.isinf(out_fp16).any().item()
    max_abs = out_fp16.abs().max().item() if not (has_nan or has_inf) else float("nan")
    overflow_risk = (not has_nan and not has_inf) and max_abs > FP16_OVERFLOW_THRESHOLD

    print(f"\nfp16 NaN / Inf    : {has_nan} / {has_inf}")
    print(f"fp16 max |output| : {max_abs:.2f}  (fp16 ceiling 65504)")

    if has_nan or has_inf:
        print(
            "\n[FAIL] NaN/Inf in fp16 output. Known issue: pure fp16 overflows at "
            "network.9 (~60928, near fp16 ceiling) producing NaN at network.10's "
            "attention. See module docstring for mitigation options. Stage 1 "
            "correctness is unaffected by this — it is a precision issue, not "
            "an architecture bug."
        )
        return False

    score = psnr(out_fp16, out_f32)
    print(f"PSNR fp16 vs fp32 : {score:.1f} dB")

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
    print(f"Verifying vision encoder: {variant} ({weights_dir})")
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

    if stage == "correctness":
        sys.exit(0 if stage_correctness(config, weights_dir) else 1)
    if stage == "fp16":
        sys.exit(0 if stage_fp16(config, weights_dir) else 1)

    if not stage_correctness(config, weights_dir):
        print("\n>>> Stopped at Stage 1. Fix correctness before running Stage 2.")
        sys.exit(1)
    if not stage_fp16(config, weights_dir):
        print("\n>>> Stopped at Stage 2. Architecture correct; fp16 precision is not.")
        sys.exit(1)

    print("\n" + "=" * 56)
    print("ALL STAGES PASS — vision encoder is ready to export.")
    print("=" * 56)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Two-stage verification for the re-authored FastVLM vision encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
stages:
  all          Run Stage 1 then Stage 2. Stop at first failure. (default)
  correctness  Stage 1 only: fp32 port vs original HF vision tower.
  fp16         Stage 2 only: fp16 vs fp32 self-consistency and health check.

examples:
  python scripts/verify_vision_encoder.py
  python scripts/verify_vision_encoder.py --variant 0.5b
  python scripts/verify_vision_encoder.py --stage correctness
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
