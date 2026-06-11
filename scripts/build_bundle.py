"""
build_bundle.py — Assemble the FastVLM deployment bundle from compiled .aimodel.

Copies the compiled asset and tokenizer files into a bundle directory
suitable for inclusion in Photo Spotter via On-Demand Resources (ODA).

Usage:
    python scripts/build_bundle.py --variant 1.5b --platform ios
    python scripts/build_bundle.py --variant 1.5b --platform macos

Output:
    exports/fastvlm-{variant}-bundle-{platform}/
      ├── metadata.json
      ├── tokenizer/
      │   ├── tokenizer.json
      │   └── tokenizer_config.json
      └── fastvlm.aimodel     ← single compiled asset (3 functions)
"""

import argparse
import json
import os
import shutil
from pathlib import Path


VARIANTS = {
    "0.5b": {"image_size": 336, "vocab_size": 151936, "max_context": 4096},
    "1.5b": {"image_size": 336, "vocab_size": 151936, "max_context": 4096},
    "7b":   {"image_size": 336, "vocab_size": 151936, "max_context": 8192},
}

# Update these from discover_weights.py output if values differ.
# image_size comes from vision_config.image_size in config.json.
# vocab_size comes from text_config.vocab_size.
# max_context comes from text_config.max_position_embeddings.


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble FastVLM deployment bundle")
    ap.add_argument("--variant",  required=True, choices=VARIANTS.keys())
    ap.add_argument("--platform", required=True, choices=["ios", "macos"])
    args = ap.parse_args()

    cfg = VARIANTS[args.variant]
    src_dir = Path(f"exports/fastvlm-{args.variant}")
    bundle_dir = Path(f"exports/fastvlm-{args.variant}-bundle-{args.platform}")
    weights_dir = Path(f"weights/fastvlm-{args.variant}")
    compiled_asset = src_dir / f"fastvlm_{args.platform}.aimodel"

    if not compiled_asset.exists():
        print(f"ERROR: {compiled_asset} not found.")
        print(
            f"Run: xcrun coreai-build compile "
            f"{src_dir}/fastvlm.aimodel "
            f"--platform {args.platform.capitalize()} "
            f"--output {compiled_asset}"
        )
        raise SystemExit(1)

    # Create bundle directory
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "tokenizer").mkdir(exist_ok=True)

    # Copy compiled .aimodel (single asset, three functions)
    dst_asset = bundle_dir / "fastvlm.aimodel"
    if dst_asset.exists():
        shutil.rmtree(dst_asset)
    shutil.copytree(compiled_asset, dst_asset)
    print(f"Copied fastvlm.aimodel")

    # Copy tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src_f = weights_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, bundle_dir / "tokenizer" / fname)
            print(f"Copied tokenizer/{fname}")
        else:
            print(f"Warning: {fname} not found in {weights_dir}")

    # Write metadata.json
    metadata = {
        "name": f"fastvlm-{args.variant}",
        "kind": "vlm",
        "schema": "0.1",
        "tokenizer": "tokenizer",
        "vocab_size": cfg["vocab_size"],
        "max_context_length": cfg["max_context"],
        "image_size": cfg["image_size"],
        "asset": "fastvlm.aimodel",
        "functions": {
            "vision_encode": "vision_encode",
            "project": "project",
            "decode": "decode",
        },
    }
    with open(bundle_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Wrote metadata.json")

    print(f"\nBundle ready: {bundle_dir}/")
    print("Contents:")
    for item in sorted(bundle_dir.rglob("*")):
        if item.is_file():
            size_kb = item.stat().st_size // 1024
            print(f"  {item.relative_to(bundle_dir)}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
