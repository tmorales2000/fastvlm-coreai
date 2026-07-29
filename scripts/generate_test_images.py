#!/usr/bin/env python3
"""
generate_test_images.py — Generate synthetic test images for preprocessing verification.

These images are designed to prove that image preprocessing strategy matters
for VLM accuracy. They use simple, unambiguous geometry that any model should
describe correctly — unless the preprocessing distorts the input.

Generated images:

  tall_narrow_circle.png  (200×800)
    Red circle on white. 1:4 aspect ratio.
    Stretch → circle becomes a flat horizontal oval (4:1 vertical compression)
    Center crop → circle stays round (top/bottom cropped, geometry preserved)

  wide_short_square.png  (800×200)
    Blue square on white. 4:1 aspect ratio.
    Stretch → square becomes a wide horizontal rectangle (4:1 horizontal stretch)
    Center crop → square stays square (left/right cropped, geometry preserved)

Proven results (FastVLM 1.5B fp16 vs Qwen3-VL 2B fp16, M4 Pro, cold export):

  tall_narrow_circle.png:
    FastVLM (center_crop): "The red object is a circle"    ✓
    Qwen3-VL (stretch):    "The red object is an oval"     ✗

  wide_short_square.png:
    FastVLM (center_crop): "The blue object is a square"   ✓  (expected)
    Qwen3-VL (stretch):    "The blue object is a rectangle" ✗  (expected)

Usage:
    python scripts/generate_test_images.py
    python scripts/generate_test_images.py --force   # regenerate even if exists
    python scripts/generate_test_images.py --output-dir /path/to/dir

Requires: Pillow (installed automatically as a transformers dependency)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "test_assets" / "images"


def generate_tall_narrow_circle(dest: Path) -> None:
    """200×800 — red circle at center. Tests vertical stretch distortion.

    Stretch to 1024×1024: vertical axis compressed 4×, circle → flat oval.
    Center crop to 1024×1024: shortest edge (200) scaled to 1024,
    height becomes 4096, center 1024 rows cropped → circle stays round.
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (200, 800), "white")
    draw = ImageDraw.Draw(img)
    # Circle centered at (100, 400), radius 80px
    draw.ellipse([20, 320, 180, 480], fill="red")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    print(f"  ✓ {dest.name}  (200×800, red circle at center)")
    print(f"    Stretch → circle becomes flat oval")
    print(f"    Center crop → circle stays round (top/bottom cropped)")


def generate_wide_short_square(dest: Path) -> None:
    """800×200 — blue square at center. Tests horizontal stretch distortion.

    Stretch to 1024×1024: horizontal axis stretched 4×, square → wide rectangle.
    Center crop to 1024×1024: shortest edge (200) scaled to 1024,
    width becomes 4096, center 1024 columns cropped → square stays square.
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (800, 200), "white")
    draw = ImageDraw.Draw(img)
    # Square centered at (400, 100), 160×160px
    draw.rectangle([320, 20, 480, 180], fill="blue")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    print(f"  ✓ {dest.name}  (800×200, blue square at center)")
    print(f"    Stretch → square becomes wide rectangle")
    print(f"    Center crop → square stays square (left/right cropped)")


IMAGES = [
    ("tall_narrow_circle.png", generate_tall_narrow_circle),
    ("wide_short_square.png",  generate_wide_short_square),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate images even if they already exist",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("ERROR: Pillow is required but not installed.", file=sys.stderr)
        print("Install with: uv pip install pillow", file=sys.stderr)
        print("(Pillow is normally installed automatically with transformers)", file=sys.stderr)
        sys.exit(1)

    print(f"Generating synthetic test images → {args.output_dir}/")
    print()

    for filename, generator in IMAGES:
        dest = args.output_dir / filename
        if dest.exists() and not args.force:
            print(f"  ✓ {filename} (already exists — use --force to regenerate)")
        else:
            generator(dest)
        print()

    print("Done.")
    print()
    print("Use with llm-runner to verify preprocessing strategy:")
    print()
    print("  LLM_RUNNER=~/git/apple/coreai-models/.build/out/Products/Debug/llm-runner")
    print()
    print("  # FastVLM (center_crop) — should say 'circle' and 'square'")
    print("  $LLM_RUNNER --model exports/fastvlm-1.5b \\")
    print(f"    --image {args.output_dir}/tall_narrow_circle.png \\")
    print('    --prompt "What shape is the red object?" --max-tokens 20 --temperature 0')
    print()
    print("  # Qwen3-VL (stretch) — will say 'oval' and 'rectangle'")
    print("  $LLM_RUNNER --model exports/qwen3_vl_2b \\")
    print(f"    --image {args.output_dir}/wide_short_square.png \\")
    print('    --prompt "What shape is the blue object?" --max-tokens 20 --temperature 0')


if __name__ == "__main__":
    main()
