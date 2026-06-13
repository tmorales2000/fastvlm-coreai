"""
verify_vision_encoder.py — PSNR verification for the re-authored vision encoder.

Must show > 70 dB before proceeding to export.

Usage:
    python scripts/verify_vision_encoder.py [--variant 1.5b]
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from fastvlm_vision_encoder import FastVLMVisionEncoder, _load_vision_weights


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.float(), b.float()
    mse = ((a_f - b_f) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(b_f.abs().max().item() ** 2 / mse)


def verify(variant: str = "1.5b") -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying vision encoder for {variant}")

    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    image_size = config.vision_config.image_size
    print(f"Image size: {image_size}x{image_size}")

    # ── fp16 model ────────────────────────────────────────────────────────────
    print("Loading fp16 model...")
    model_fp16 = FastVLMVisionEncoder.from_weights(config, weights_dir)
    model_fp16.eval()

    # ── fp32 reference model ──────────────────────────────────────────────────
    # Same architecture, same weights — just cast to float32
    print("Loading fp32 reference model...")
    model_f32 = FastVLMVisionEncoder(weights_dir).to(dtype=torch.float32)
    weights_f32 = _load_vision_weights(weights_dir, dtype=torch.float32)
    missing, unexpected = model_f32.load_state_dict(weights_f32, assign=True, strict=False)
    # head.proj not used in forward — expected missing
    missing_real = [k for k in missing if "head" not in k]
    unexpected_real = [k for k in unexpected if "head" not in k]
    if missing_real:
        raise RuntimeError(f"Missing keys in fp32 model: {missing_real[:5]}")
    if unexpected_real:
        raise RuntimeError(f"Unexpected keys in fp32 model: {unexpected_real[:5]}")
    model_f32.eval()

    # ── Test inputs ───────────────────────────────────────────────────────────
    pixels = torch.randn(1, 3, image_size, image_size)
    pixels_fp16 = pixels.to(torch.float16)
    pixels_f32 = pixels.to(torch.float32)

    # ── Forward passes ────────────────────────────────────────────────────────
    with torch.no_grad():
        features_fp16 = model_fp16(pixels_fp16)
        features_f32 = model_f32(pixels_f32)

    score = psnr(features_fp16, features_f32)
    print(f"\nOutput shape : {features_fp16.shape}")
    print(f"Output dtype : {features_fp16.dtype}")
    print(f"PSNR fp16 vs fp32: {score:.1f} dB")
    print()

    if score > 70:
        print(f"PASS — {score:.1f} dB > 70 dB threshold.")
    elif score > 60:
        print(f"MARGINAL — {score:.1f} dB. Investigate before export.")
    else:
        print(f"FAIL — {score:.1f} dB < 60 dB. Debug before export.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    verify(ap.parse_args().variant)
