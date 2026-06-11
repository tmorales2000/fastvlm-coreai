"""
palettize.py — 4-bit palettization for iOS deployment (0.5B, 1.5B).
For 7B macOS, use INT4 quantization instead (see comments below).

Run AFTER verify_decoder.py, verify_vision_encoder.py, verify_projector.py
pass their PSNR gates. Re-run verify_runtime.py after palettization.

PSNR target after palettization: > 35 dB vs fp16 baseline.

Usage:
    python scripts/palettize.py --variant 1.5b --platform ios

TODO: Complete this script once coreai-opt API is confirmed from
      apple.github.io/coreai-optimization documentation.
"""

import argparse
import sys

# from coreai_opt import KMeansPalettizer  # import once coreai-opt is installed


def palettize_ios(variant: str = "1.5b") -> None:
    """
    4-bit palettization with per-channel scales.
    Applies to Conv2d and Linear layers only.
    Skip embeddings and norms — they are sensitive to compression.

    Reference: WWDC 2026 "Dive into Core AI model authoring and optimization"
    session shows KMeansPalettizer usage for SAM3:

        from coreai_opt import KMeansPalettizer
        config = {
            'granularity': 'per_grouped_channel',
            'n_bits': 4,
            'group_size': 32,
            'module_name_filters': [
                {'module_type': 'Linear'},
                {'module_type': 'Conv2d'},
            ],
            'skip_module_name_filters': [
                {'module_name': '.*embed.*'},
                {'module_name': '.*norm.*'},
            ],
        }
        palettizer = KMeansPalettizer(model, config)
        palettizer.prepare()
        palettized_model = palettizer.finalize()

    After palettization, re-run the export and compilation pipeline
    using the palettized models in place of the fp16 ones.
    """
    print(f"TODO: Implement palettization for {variant} iOS")
    print("Check: apple.github.io/coreai-optimization for current API")
    sys.exit(0)


def quantize_macos_int4(variant: str = "7b") -> None:
    """
    INT4 weight-only quantization for 7B macOS.
    Uses coreai-opt presets.w4 (4-bit per-channel symmetric).
    """
    print(f"TODO: Implement INT4 quantization for {variant} macOS")
    print("Reference: from coreai_opt import presets; presets.w4")
    sys.exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",  required=True, choices=["0.5b", "1.5b", "7b"])
    ap.add_argument("--platform", required=True, choices=["ios", "macos"])
    args = ap.parse_args()

    if args.platform == "ios":
        palettize_ios(args.variant)
    else:
        if args.variant == "7b":
            quantize_macos_int4(args.variant)
        else:
            # 0.5B and 1.5B macOS can also use palettization if desired
            palettize_ios(args.variant)
