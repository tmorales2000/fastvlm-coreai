"""
verify_vision_encoder.py — Architecture correctness for the re-authored vision encoder.

Compares the re-authored FastVLMVisionEncoder (fp16) against the HF vision tower
weights loaded at fp32. Equivalent to verify_decoder Phase 1 for the vision encoder
component.

Gate:
  > 70 dB   PASS
  50–70 dB  MARGINAL (exits nonzero — investigate before export)
  < 50 dB   FAIL

Note: Vision encoder uses torch.autocast(cpu, fp16) to promote attention ops to
fp32 internally, matching the ANE compiler's behavior. Pure fp16 overflows at
network.9 (max ~60928, near fp16 ceiling of 65504).

Usage:
    python scripts/verify_vision_encoder.py --variant 0.5b
    python scripts/verify_vision_encoder.py --variant 1.5b
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fastvlm_vision_encoder import FastVLMVisionEncoder, _load_vision_weights  # noqa: E402
from metrics import psnr                                                        # noqa: E402

PASS_THRESHOLD     = 70.0   # dB
MARGINAL_THRESHOLD = 50.0   # dB — exits nonzero


def verify(variant: str = "1.5b") -> None:
    weights_dir = str(REPO_ROOT / "weights" / f"fastvlm-{variant}")
    print(f"Verifying vision encoder: {variant}")
    print(f"{'='*50}")

    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    image_size = int(config.mm_vision_tower.split("_")[-1])
    print(f"Image size: {image_size}×{image_size}")

    # fp16 re-authored model
    print("Loading fp16 model...")
    model_fp16 = FastVLMVisionEncoder.from_weights(config, weights_dir)
    model_fp16.eval()

    # fp32 reference — same weights, cast to float32
    print("Loading fp32 reference model...")
    model_f32 = FastVLMVisionEncoder(weights_dir).to(dtype=torch.float32)
    weights_f32 = _load_vision_weights(weights_dir, dtype=torch.float32)
    missing, unexpected = model_f32.model.load_state_dict(
        weights_f32, assign=True, strict=False
    )
    # head.proj not used in forward — expected missing
    missing_real   = [k for k in missing   if "head" not in k]
    unexpected_real = [k for k in unexpected if "head" not in k]
    if missing_real:
        raise RuntimeError(f"Missing keys in fp32 model: {missing_real[:5]}")
    if unexpected_real:
        raise RuntimeError(f"Unexpected keys in fp32 model: {unexpected_real[:5]}")
    model_f32.eval()

    # Random test input — architecture correctness test (same as verify_decoder Phase 1)
    # Always CPU — IEEE 754 strict fp32 for meaningful PSNR comparison.
    # MPS rounding produces lower scores that don't indicate bugs.
    # torch.autocast promotes fp16 attention ops to fp32 to prevent overflow at network.9.
    model_fp16 = model_fp16.cpu()
    model_f32  = model_f32.cpu()
    pixels = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.float16):
        features_fp16 = model_fp16(pixels.to(torch.float16))
    with torch.no_grad():
        features_f32 = model_f32(pixels.to(torch.float32))

    score = psnr(features_f32, features_fp16)

    print(f"\nOutput shape      : {features_fp16.shape}")
    print(f"Output dtype      : {features_fp16.dtype}")
    print(f"PSNR fp16 vs fp32 : {score:.1f} dB")
    print()

    if score > PASS_THRESHOLD:
        print(f"[PASS] {score:.1f} dB — vision encoder port matches HF vision tower.")
    elif score > MARGINAL_THRESHOLD:
        print(
            f"[MARGINAL] {score:.1f} dB — investigate fp16 overflow risk before export."
            f"\n  Check network.8-10 activation magnitudes with probe_activations.py."
        )
        sys.exit(1)
    else:
        print(f"[FAIL] {score:.1f} dB — vision encoder diverges from HF reference.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    verify(ap.parse_args().variant)
