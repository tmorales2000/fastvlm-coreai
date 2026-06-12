"""
verify_projector.py — PSNR verification for the re-authored projector.

Must show > 60 dB before proceeding to export.

Usage:
    python scripts/verify_projector.py [--variant 1.5b]
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from fastvlm_projector import FastVLMProjector


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.float(), b.float()
    mse = ((a_f - b_f) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(b_f.abs().max().item() ** 2 / mse)


def verify(variant: str = "1.5b") -> None:
    weights_dir = f"weights/fastvlm-{variant}"
    print(f"Verifying projector for {variant}")

    config = AutoConfig.from_pretrained(weights_dir)

    model_fp16 = FastVLMProjector.from_weights(config, weights_dir)
    model_fp16.eval()

    # fp32 reference
    model_f32 = FastVLMProjector(config).to(dtype=torch.float32)
    # Load weights with float32 cast
    import glob, os
    from safetensors import safe_open
    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    weights_f32 = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mm_projector" not in key:
                    continue
                stripped = "layers." + key.removeprefix("model.mm_projector.")
                weights_f32[stripped] = f.get_tensor(key).float()
    model_f32.load_state_dict(weights_f32, assign=True, strict=True)
    model_f32.eval()

    # Build test input matching mm_hidden_size
    B, seq_len = 1, 256
    x = torch.randn(B, seq_len, config.mm_hidden_size)

    with torch.no_grad():
        out_fp16 = model_fp16(x.to(torch.float16))
        out_f32 = model_f32(x.to(torch.float32))

    score = psnr(out_fp16, out_f32)
    print(f"\nOutput shape : {out_fp16.shape}")
    print(f"PSNR fp16 vs fp32: {score:.1f} dB")
    print()

    if score > 60:
        print(f"PASS — {score:.1f} dB > 60 dB threshold.")
    else:
        print(f"FAIL — {score:.1f} dB. Debug projector weight loading.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    verify(ap.parse_args().variant)
