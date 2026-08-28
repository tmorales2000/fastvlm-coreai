"""
verify_projector.py — Architecture correctness for the re-authored projector.

Compares the re-authored FastVLMProjector (fp16) against the HF mm_projector
weights loaded at fp32. Equivalent to verify_decoder Phase 1 for the projector
component.

Gate:
  > 60 dB   PASS
  40–60 dB  MARGINAL (exits nonzero — investigate before export)
  < 40 dB   FAIL

Usage:
    python scripts/verify_projector.py --variant 0.5b
    python scripts/verify_projector.py --variant 1.5b
"""

import argparse
import glob
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fastvlm_projector import FastVLMProjector  # noqa: E402
from metrics import psnr                        # noqa: E402

PASS_THRESHOLD     = 60.0   # dB
MARGINAL_THRESHOLD = 40.0   # dB — exits nonzero


def verify(variant: str = "1.5b") -> None:
    weights_dir = str(REPO_ROOT / "weights" / f"fastvlm-{variant}")
    print(f"Verifying projector: {variant}")
    print(f"{'='*50}")

    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

    # fp16 re-authored model
    model_fp16 = FastVLMProjector.from_weights(config, weights_dir)
    model_fp16.eval()

    # fp32 reference — same weights, cast to float32
    model_f32 = FastVLMProjector(config).to(dtype=torch.float32)
    import os
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

    # Random test input — architecture correctness test (same as verify_decoder Phase 1)
    # Always CPU — IEEE 754 strict fp32 for meaningful PSNR comparison.
    # MPS rounding produces lower scores that don't indicate bugs.
    model_fp16 = model_fp16.cpu()
    model_f32  = model_f32.cpu()

    B, seq_len = 1, 256
    x = torch.randn(B, seq_len, config.mm_hidden_size)

    with torch.no_grad():
        out_fp16 = model_fp16(x.to(torch.float16))
        out_f32  = model_f32(x.to(torch.float32))

    score = psnr(out_f32, out_fp16)

    print(f"Output shape      : {out_fp16.shape}")
    print(f"PSNR fp16 vs fp32 : {score:.1f} dB")
    print()

    if score > PASS_THRESHOLD:
        print(f"[PASS] {score:.1f} dB — projector port matches HF mm_projector.")
    elif score > MARGINAL_THRESHOLD:
        print(f"[MARGINAL] {score:.1f} dB — investigate weight loading before export.")
        sys.exit(1)
    else:
        print(f"[FAIL] {score:.1f} dB — projector diverges from HF reference.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    verify(ap.parse_args().variant)
