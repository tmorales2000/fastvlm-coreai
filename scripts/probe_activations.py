"""
probe_activations.py — Inspect intermediate activation magnitudes in the
vision encoder's network stages.

PURPOSE
-------
fp16 has a hard ceiling of 65504. Before deciding whether a component can run
safely in fp16, you need to know what magnitude its activations actually reach
at each stage — guessing or waiting for a NaN in verify_*.py tells you THAT
something overflowed, not WHERE or BY HOW MUCH per variant.

This script runs a forward pass in fp32 (so nothing overflows during probing)
and reports min/max/mean-abs at each requested network stage, across one or
more variants in a single run. Use it whenever you need to compare activation
behavior across variants or localize which stage first produces large values.

ORIGIN
------
Built after discovering that FastVLM's vision tower activation magnitude at
conv_exp scales inversely with language model size:
  0.5B network.9 max abs ~252,866   (3.9x past fp16 ceiling)
  1.5B network.9 max abs  ~55,630   (just under ceiling — real images may exceed it)
  7B   network.9 max abs  ~12,740   (comfortable headroom)
This is NOT random noise — each step down in LM size roughly 5x's the vision
tower's activation scale, likely because there is no normalization between
conv_exp and the projector to constrain it, so each variant's vision tower
settled at whatever scale gradient descent left it during joint training.

USAGE
-----
  # Single variant, default stages (8, 9, 10 — the historically overflow-prone ones)
  python scripts/probe_activations.py --variant 1.5b

  # Compare all three variants at once
  python scripts/probe_activations.py --variant 0.5b 1.5b 7b

  # Probe every stage instead of just 8/9/10
  python scripts/probe_activations.py --variant 1.5b --stages all

  # Specific stages only
  python scripts/probe_activations.py --variant 1.5b --stages 6 7 8 9 10

  # Use a fixed seed for reproducible random input (default: 0)
  python scripts/probe_activations.py --variant 1.5b --seed 42

ARGUMENTS
---------
  --variant   One or more FastVLM variants to probe. Default: 1.5b.
              Choices: 0.5b, 1.5b, 7b.

  --stages    Which network stage indices to report. Default: 8 9 10
              (the stages implicated in the known fp16 overflow). Pass
              "all" to report every stage in the network ModuleList.

  --seed      Random seed for the test input. Default: 0.

OUTPUT
------
A table: variant, stage index, stage type, min, max, mean-abs. Always run
in fp32 regardless of variant, so the probe itself never overflows — the
point is to measure true magnitude, not to reproduce the fp16 failure.
"""

import argparse
import sys

import torch

sys.path.insert(0, "scripts")
from fastvlm_vision_encoder import FastVLMVisionEncoder, _load_vision_weights


def _image_size(config) -> int:
    return int(config.mm_vision_tower.split("_")[-1])


def probe(variant: str, stages: list[int] | str, seed: int) -> list[dict]:
    """Run a single variant and return per-stage activation stats."""
    from transformers import AutoConfig

    weights_dir = f"weights/fastvlm-{variant}"
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
    image_size = _image_size(config)

    model = FastVLMVisionEncoder(weights_dir).to(torch.float32)
    weights = _load_vision_weights(weights_dir, dtype=torch.float32)
    model.model.load_state_dict(weights, assign=True, strict=False)
    model.eval()

    torch.manual_seed(seed)
    pixels = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    n_stages = len(model.model.network)
    target_stages = set(range(n_stages)) if stages == "all" else set(stages)

    rows = []
    with torch.no_grad():
        x = model.model.forward_embeddings(pixels)
        for i, stage in enumerate(model.model.network):
            x = stage(x)
            if i in target_stages:
                rows.append({
                    "variant": variant,
                    "stage": i,
                    "type": type(stage).__name__,
                    "min": x.min().item(),
                    "max": x.max().item(),
                    "mean_abs": x.abs().mean().item(),
                    "max_abs": x.abs().max().item(),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe vision encoder activation magnitudes per network stage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/probe_activations.py --variant 1.5b
  python scripts/probe_activations.py --variant 0.5b 1.5b 7b
  python scripts/probe_activations.py --variant 1.5b --stages all
  python scripts/probe_activations.py --variant 1.5b --stages 6 7 8 9 10
""",
    )
    ap.add_argument(
        "--variant",
        nargs="+",
        default=["1.5b"],
        choices=["0.5b", "1.5b", "7b"],
        help="One or more variants to probe. (default: 1.5b)",
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["8", "9", "10"],
        help='Stage indices to report, or "all". (default: 8 9 10)',
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for test input. (default: 0)",
    )
    args = ap.parse_args()

    stages = "all" if args.stages == ["all"] else [int(s) for s in args.stages]

    all_rows = []
    for variant in args.variant:
        print(f"Probing {variant}...")
        all_rows.extend(probe(variant, stages, args.seed))

    print(f"\n{'variant':>8} | {'stage':>5} | {'type':<16} | {'min':>12} | {'max':>12} | {'mean|x|':>10} | {'max|x|':>12}")
    print("-" * 90)
    for r in all_rows:
        flag = "  <-- fp16 OVERFLOW RISK" if r["max_abs"] > 60000 else ""
        print(
            f"{r['variant']:>8} | {r['stage']:>5} | {r['type']:<16} | "
            f"{r['min']:>12.1f} | {r['max']:>12.1f} | {r['mean_abs']:>10.2f} | "
            f"{r['max_abs']:>12.1f}{flag}"
        )


if __name__ == "__main__":
    main()
