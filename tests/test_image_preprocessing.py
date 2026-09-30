#!/usr/bin/env python3
"""
tests/test_image_preprocessing.py — Verify image preprocessing strategy.

Tests that FastVLM's center_crop preprocessing correctly preserves geometry,
while stretch preprocessing would distort it. This was the root cause of
a real bug filed against apple/coreai-models (issue #100, fixed in PR #108).

The test generates synthetic images with unambiguous geometry, runs them
through an exported FastVLM bundle via llm-runner, and asserts the model
describes the shapes correctly.

Generated test images (owned by this test, created on demand):
  tests/fixtures/tall_narrow_circle.png  (200×800) — red circle
  tests/fixtures/wide_short_square.png   (800×200) — blue square

Expected results with center_crop (correct):
  tall_narrow_circle.png → model says "circle" (not "oval")
  wide_short_square.png  → model says "square" (not "rectangle")

Expected results with stretch (wrong — the original bug):
  tall_narrow_circle.png → model says "oval" or "ellipse"
  wide_short_square.png  → model says "rectangle"

Usage:
    # Run with pytest (requires exported bundle)
    pytest tests/test_image_preprocessing.py --variant 0.5b

    # Run directly
    python tests/test_image_preprocessing.py --variant 0.5b

Requirements:
    - Exported FastVLM bundle at exports/fastvlm-{variant}/
    - llm-runner binary (built from apple/coreai-models)
    - Pillow (pip install pillow)
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT    = Path(__file__).parent.parent
FIXTURE_DIR  = Path(__file__).parent / "fixtures"
LLM_RUNNER   = Path.home() / "git/apple/coreai-models/.build/out/Products/Debug/llm-runner"


# ── Synthetic image generation ─────────────────────────────────────────────────

def _generate_tall_narrow_circle(dest: Path) -> None:
    """200×800 — red circle at center.

    center_crop: shortest edge (200) scaled to 1024, height → 4096,
    center 1024 rows cropped → circle stays round.
    stretch: vertical axis compressed 4×, circle → flat oval.
    """
    from PIL import Image, ImageDraw
    img  = Image.new("RGB", (200, 800), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 320, 180, 480], fill="red")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)


def _generate_wide_short_square(dest: Path) -> None:
    """800×200 — blue square at center.

    center_crop: shortest edge (200) scaled to 1024, width → 4096,
    center 1024 columns cropped → square stays square.
    stretch: horizontal axis stretched 4×, square → wide rectangle.
    """
    from PIL import Image, ImageDraw
    img  = Image.new("RGB", (800, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([320, 20, 480, 180], fill="blue")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)


def ensure_test_images() -> tuple[Path, Path]:
    """Generate synthetic test images if not present. Returns (circle, square) paths."""
    circle = FIXTURE_DIR / "tall_narrow_circle.png"
    square = FIXTURE_DIR / "wide_short_square.png"
    if not circle.exists():
        _generate_tall_narrow_circle(circle)
    if not square.exists():
        _generate_wide_short_square(square)
    return circle, square


# ── llm-runner inference ───────────────────────────────────────────────────────

def run_vlm(model_path: Path, image_path: Path, prompt: str,
            llm_runner: Path = LLM_RUNNER) -> str:
    """Run llm-runner and return the generated text."""
    if not llm_runner.exists():
        raise FileNotFoundError(
            f"llm-runner not found at {llm_runner}\n"
            f"Build with: cd ~/git/apple/coreai-models && swift build --product llm-runner"
        )
    result = subprocess.run(
        [
            str(llm_runner),
            "--model", str(model_path),
            "--image", str(image_path),
            "--prompt", prompt,
            "--max-tokens", "50",
            "--temperature", "0",
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"llm-runner failed:\n{result.stderr}")
    # Extract generated text — lines after the last blank line before stats
    lines = result.stdout.strip().split("\n")
    # Filter out log lines (start with [) and stats
    output_lines = [l for l in lines if not l.startswith("[") and "✅" not in l]
    return " ".join(output_lines).strip().lower()


# ── Test cases ─────────────────────────────────────────────────────────────────

def test_circle_not_oval(variant: str, llm_runner: Path = LLM_RUNNER) -> bool:
    """200×800 red circle should be described as a circle, not an oval."""
    model   = REPO_ROOT / "exports" / f"fastvlm-{variant}"
    circle, _ = ensure_test_images()
    prompt  = "What shape is the red object? Answer in one word."

    print(f"\n[test_circle_not_oval] variant={variant}")
    print(f"  Image: {circle.name} (200×800 — red circle)")
    print(f"  Expected: 'circle' not 'oval'")

    text = run_vlm(model, circle, prompt, llm_runner)
    print(f"  Response: {text!r}")

    passed = "circle" in text and "oval" not in text
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_square_not_rectangle(variant: str, llm_runner: Path = LLM_RUNNER) -> bool:
    """800×200 blue square should be described as a square, not a rectangle."""
    model   = REPO_ROOT / "exports" / f"fastvlm-{variant}"
    _, square = ensure_test_images()
    prompt  = "What shape is the blue object? Answer in one word."

    print(f"\n[test_square_not_rectangle] variant={variant}")
    print(f"  Image: {square.name} (800×200 — blue square)")
    print(f"  Expected: 'square' not 'rectangle'")

    text = run_vlm(model, square, prompt, llm_runner)
    print(f"  Response: {text!r}")

    passed = "square" in text and "rectangle" not in text
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


# ── pytest integration ─────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "preprocessing: image preprocessing strategy tests")


# Default variant for pytest — override with --variant flag in conftest
_DEFAULT_VARIANT = "0.5b"


def test_preprocessing_circle(pytestconfig):
    variant = getattr(pytestconfig, "getoption", lambda x, d=None: d)("--variant") or _DEFAULT_VARIANT
    assert test_circle_not_oval(variant), \
        "Model described circle as oval — image preprocessing may be stretching"


def test_preprocessing_square(pytestconfig):
    variant = getattr(pytestconfig, "getoption", lambda x, d=None: d)("--variant") or _DEFAULT_VARIANT
    assert test_square_not_rectangle(variant), \
        "Model described square as rectangle — image preprocessing may be stretching"


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="0.5b", choices=["0.5b", "1.5b", "7b"])
    ap.add_argument("--llm-runner", type=Path, default=LLM_RUNNER)
    ap.add_argument("--generate-only", action="store_true",
                    help="Generate test images only, do not run inference.")
    args = ap.parse_args()

    circle, square = ensure_test_images()
    print(f"Test images ready:")
    print(f"  {circle}")
    print(f"  {square}")

    if args.generate_only:
        return

    model = REPO_ROOT / "exports" / f"fastvlm-{args.variant}"
    if not model.exists():
        print(f"\nERROR: Bundle not found: {model}")
        print(f"Export first: python scripts/export_fastvlm.py --variant {args.variant}")
        sys.exit(1)

    results = [
        test_circle_not_oval(args.variant, args.llm_runner),
        test_square_not_rectangle(args.variant, args.llm_runner),
    ]

    print(f"\n{'='*50}")
    passed = sum(results)
    print(f"{'PASS' if all(results) else 'FAIL'} — {passed}/{len(results)} tests passed")
    if not all(results):
        print("Image preprocessing may not be applying center_crop correctly.")
        print("Check that metadata.json declares image_strategy: center_crop")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
