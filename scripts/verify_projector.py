"""
verify_projector.py — Two-phase verification for the re-authored projector.

Mirrors verify_decoder.py Phases 1 and 2:

  Phase 1 — ARCHITECTURE CORRECTNESS (fp32, port vs HF mm_projector)
      Does our re-authored FastVLMProjector compute the same thing as the
      original HF mm_projector loaded directly from the checkpoint?
      Both loaded from bf16 source weights, cast to fp32.
      Gate: > 60 dB PASS, 40–60 dB MARGINAL (exits nonzero), < 40 dB FAIL.

  Phase 2 — FP16 FIDELITY (port fp32 vs port fp16)
      What does the bf16→fp16 cast cost?
      Re-authored model in fp32 vs re-authored model in fp16, same inputs.
      Informational (MEASURED) — establishes the fp16 deployment baseline.

Both phases run on CPU for IEEE 754 strict fp32 precision.

Usage:
    python scripts/verify_projector.py --variant 0.5b
    python scripts/verify_projector.py --variant 1.5b
    python scripts/verify_projector.py --variant 0.5b --stage correctness
    python scripts/verify_projector.py --variant 0.5b --stage fidelity
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fastvlm_projector import FastVLMProjector, _load_projector_weights  # noqa: E402
from metrics import psnr, full_report, print_report                       # noqa: E402

PASS_THRESHOLD     = 60.0   # dB — Phase 1 gate
MARGINAL_THRESHOLD = 40.0   # dB — Phase 1 MARGINAL exits nonzero


def _build_hf_projector(config, weights_dir: str) -> nn.Sequential:
    """Load HF mm_projector directly — plain nn.Sequential from checkpoint.

    Key mapping: model.mm_projector.N.* → N.*
    Same weights as FastVLMProjector, different key naming.
    """
    from safetensors import safe_open

    hidden_size = config.hidden_size
    mm_hidden   = config.mm_hidden_size
    proj_type   = config.mm_projector_type
    import re
    match = re.match(r"^mlp(\d+)x_gelu$", proj_type)
    depth = int(match.group(1)) if match else 2

    layers: list[nn.Module] = [nn.Linear(mm_hidden, hidden_size)]
    for _ in range(1, depth):
        layers += [nn.GELU(), nn.Linear(hidden_size, hidden_size)]
    model = nn.Sequential(*layers).to(torch.float32)

    # Load weights directly — "model.mm_projector.0.weight" → "0.weight"
    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    weights: dict[str, torch.Tensor] = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mm_projector" not in key:
                    continue
                stripped = key.removeprefix("model.mm_projector.")
                weights[stripped] = f.get_tensor(key).float()

    model.load_state_dict(weights, strict=True)
    return model.eval()


def phase_correctness(config, weights_dir: str) -> bool:
    print("\n" + "=" * 50)
    print("PHASE 1 — ARCHITECTURE CORRECTNESS (fp32, port vs HF)")
    print("=" * 50)
    print("Both models loaded from bf16 checkpoint, cast to fp32.")
    print("Purpose: catch re-authoring bugs, not precision issues.")

    # HF reference — direct mm_projector Sequential
    hf_proj = _build_hf_projector(config, weights_dir)

    # Port — re-authored FastVLMProjector at fp32
    port_fp32 = FastVLMProjector(config).to(torch.float32)
    weights_f32 = _load_projector_weights(weights_dir, dtype=torch.float32)
    port_fp32.load_state_dict(weights_f32, assign=True, strict=True)
    port_fp32.eval()

    B, seq_len = 1, 256
    x = torch.randn(B, seq_len, config.mm_hidden_size)

    with torch.no_grad():
        out_hf   = hf_proj(x.to(torch.float32))
        out_port = port_fp32(x.to(torch.float32))

    score = psnr(out_hf, out_port)

    print(f"\nOutput shape      : {out_port.shape}")
    print(f"PSNR port vs HF   : {score:.1f} dB")
    print()

    if score > PASS_THRESHOLD:
        print(f"[PASS] {score:.1f} dB — port matches HF mm_projector.")
        return True
    if score > MARGINAL_THRESHOLD:
        print(f"[MARGINAL] {score:.1f} dB — investigate weight loading before export.")
        return False
    print(f"[FAIL] {score:.1f} dB — port diverges from HF mm_projector.")
    return False


def phase_fidelity(config, weights_dir: str) -> None:
    print("\n" + "=" * 50)
    print("PHASE 2 — FP16 FIDELITY (port fp32 vs port fp16)")
    print("=" * 50)
    print("Measures what the bf16→fp16 cast costs.")
    print("Establishes the fp16 deployment baseline.")

    port_fp32 = FastVLMProjector(config).to(torch.float32)
    weights_f32 = _load_projector_weights(weights_dir, dtype=torch.float32)
    port_fp32.load_state_dict(weights_f32, assign=True, strict=True)
    port_fp32.eval()

    port_fp16 = FastVLMProjector.from_weights(config, weights_dir, dtype=torch.float16)
    port_fp16.eval()

    B, seq_len = 1, 256
    x = torch.randn(B, seq_len, config.mm_hidden_size)

    with torch.no_grad():
        out_fp32 = port_fp32(x.to(torch.float32))
        out_fp16 = port_fp16(x.to(torch.float16))

    report = full_report(out_fp32, out_fp16)
    print()
    print_report(report, label="FP32 → FP16 (all positions):", indent="  ")
    print(f"\n[MEASURED] FP16 deployment baseline established.")


def verify(variant: str, stage: str) -> None:
    weights_dir = str(REPO_ROOT / "weights" / f"fastvlm-{variant}")
    print(f"Verifying projector: {variant}")
    print(f"Device: cpu (IEEE 754 fp32 — both phases)")

    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

    if stage == "correctness":
        sys.exit(0 if phase_correctness(config, weights_dir) else 1)
    if stage == "fidelity":
        phase_fidelity(config, weights_dir)
        return

    # All phases
    passed = phase_correctness(config, weights_dir)
    if not passed:
        print("\n>>> Stopped at Phase 1.")
        sys.exit(1)
    phase_fidelity(config, weights_dir)

    print("\n" + "=" * 50)
    print("PROJECTOR VERIFICATION COMPLETE")
    print("=" * 50)
    print("  Phase 1 — Architecture correctness : PASS")
    print("  Phase 2 — FP16 fidelity            : MEASURED")
    print("=" * 50)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", default="1.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument(
        "--stage", default="all",
        choices=["all", "correctness", "fidelity"],
        help="Which phase to run (default: all).",
    )
    args = ap.parse_args()
    verify(args.variant, args.stage)
