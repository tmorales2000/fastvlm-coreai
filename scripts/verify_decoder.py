"""
verify_decoder.py — PSNR verification for the re-authored FastVLM decoder.

Runs the re-authored fp16 model and a fp32 reference model on the same
random input and computes PSNR. Must show > 50 dB before proceeding to export.

Usage:
    python scripts/verify_decoder.py [--variant 1.5b]
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from fastvlm_decoder import FastVLMDecoderStateful, MAX_SEQ_LEN, _load_decoder_weights, _mutate_state_dict


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    mse = ((a_f - b_f) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    peak = b_f.abs().max().item() ** 2
    return 10 * np.log10(peak / mse)


def verify(variant: str = "1.5b") -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying decoder for {variant} from {weights_dir}")

    config = AutoConfig.from_pretrained(weights_dir)
    text_cfg = getattr(config, "text_config", config)

    # ── fp16 model ────────────────────────────────────────────────────────────
    print("Loading fp16 model...")
    model_fp16 = FastVLMDecoderStateful.from_weights(text_cfg, weights_dir)
    model_fp16.eval()

    # ── fp32 reference model ──────────────────────────────────────────────────
    print("Loading fp32 reference model...")
    weights_f32 = _load_decoder_weights(weights_dir, dtype=torch.float32)
    _mutate_state_dict(weights_f32)

    model_f32 = FastVLMDecoderStateful(text_cfg).to(dtype=torch.float32)
    model_f32.load_state_dict(weights_f32, assign=True, strict=True)
    model_f32.eval()

    # ── Test inputs ───────────────────────────────────────────────────────────
    B, L = 1, 8
    input_ids = torch.randint(1, text_cfg.vocab_size, (B, L), dtype=torch.int32)
    pos_ids = torch.arange(L, dtype=torch.int32).unsqueeze(0)

    # ── Forward passes ────────────────────────────────────────────────────────
    with torch.no_grad():
        logits_fp16 = model_fp16(input_ids, pos_ids)
        logits_f32 = model_f32(input_ids, pos_ids)

    # ── Results ───────────────────────────────────────────────────────────────
    score = psnr(logits_fp16, logits_f32)
    has_nan = torch.isnan(logits_fp16).any().item()
    has_inf = torch.isinf(logits_fp16).any().item()

    print(f"\nOutput shape : {logits_fp16.shape}")
    print(f"Output dtype : {logits_fp16.dtype}")
    print(f"NaN present  : {has_nan}")
    print(f"Inf present  : {has_inf}")
    print(f"PSNR fp16 vs fp32: {score:.1f} dB")
    print()

    if has_nan or has_inf:
        print("FAIL — NaN or Inf in output. Check weight loading.")
        sys.exit(1)
    elif score > 50:
        print(f"PASS — {score:.1f} dB > 50 dB threshold.")
        print("Safe to proceed to export.")
    elif score > 40:
        print(f"MARGINAL — {score:.1f} dB. Investigate before export.")
        print("Check for fp32 literals or precision issues in the forward pass.")
    else:
        print(f"FAIL — {score:.1f} dB < 40 dB threshold.")
        print("Do NOT proceed to export. Debug weight loading and forward pass.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    verify(ap.parse_args().variant)
