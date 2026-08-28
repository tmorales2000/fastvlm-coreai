#!/usr/bin/env python3
"""
build_fixtures.py — Pre-build and cache decoder fixtures for all variants.

Runs the full FastVLM multimodal pipeline (image → vision encoder →
projector → scatter-merge) for each image in the corpus and caches
the resulting decoder inputs_embeds to disk.

Fixtures are used by verify_decoder.py (Phase 2 and Phase 4) and
scan_quantization_sensitivity.py. Building them once up-front means
verification and scanning runs are fast (cache hits, no model loading).

Each variant has its own fixture cache because inputs_embeds dimensions
differ: 0.5B hidden=896, 1.5B hidden=1536, 7B hidden=3584.

Usage:
    # Build fixtures for all variants
    python scripts/build_fixtures.py

    # Build fixtures for a specific variant
    python scripts/build_fixtures.py --variant 0.5b

    # Force rebuild (ignores existing cache)
    python scripts/build_fixtures.py --force

    # Use a custom image directory
    python scripts/build_fixtures.py --image-dir path/to/images

Output:
    test_assets/fixtures/fastvlm-{variant}-{hash}.pt
    One file per (variant, image, prompt) combination.

Notes:
    - Uses MPS by default (780x faster than CPU for vision encoding)
    - Falls back to CPU if MPS unavailable
    - Skips variants whose weights are not downloaded
    - Safe to interrupt and resume — already-cached fixtures are skipped
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fastvlm_fixtures import (
    CORPUS_IMAGES,
    FIXTURE_CACHE_DIR,
    FIXTURE_SCHEMA_VERSION,
    DEFAULT_PROMPT,
    build_corpus_fixtures,
)

ALL_VARIANTS = ["0.5b", "1.5b", "7b"]


def variant_weights_exist(variant: str) -> bool:
    weights_dir = REPO_ROOT / "weights" / f"fastvlm-{variant}"
    return weights_dir.is_dir() and any(weights_dir.glob("*.safetensors"))


def build_variant(
    variant: str,
    images: list[str],
    force: bool = False,
    device: str = "mps",
) -> None:
    print(f"\n{'='*60}")
    print(f"Variant: fastvlm-{variant}")
    print(f"{'='*60}")

    if not variant_weights_exist(variant):
        print(f"[SKIP] Weights not found: weights/fastvlm-{variant}/")
        print(f"       Download: hf download apple/FastVLM-{variant.upper()} "
              f"--local-dir weights/fastvlm-{variant}")
        return

    if force:
        # Delete existing cache for this variant
        cache_dir = REPO_ROOT / FIXTURE_CACHE_DIR
        deleted = list(cache_dir.glob(f"fastvlm-{variant}-*.pt"))
        for f in deleted:
            f.unlink()
        if deleted:
            print(f"[FORCE] Deleted {len(deleted)} cached fixture(s)")

    t0 = time.time()
    fixtures = build_corpus_fixtures(
        variant=variant,
        images=images,
        prompt=DEFAULT_PROMPT,
        device=device,
        use_cache=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nDone: {len(fixtures)}/{len(images)} fixtures in {elapsed:.1f}s")
    print(f"Cache: {FIXTURE_CACHE_DIR}/")
    print(f"Schema version: {FIXTURE_SCHEMA_VERSION}")

    if len(fixtures) < len(images):
        missing = len(images) - len(fixtures)
        print(f"[WARN] {missing} image(s) missing from test_assets/images/")
        print(f"       Run: python scripts/fetch_test_images.py")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--variant",
        choices=ALL_VARIANTS,
        default=None,
        help="Build fixtures for one variant only (default: all available).",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Delete and rebuild all cached fixtures for the selected variant(s).",
    )
    ap.add_argument(
        "--device", default="mps", choices=["mps", "cpu"],
        help="Compute device for vision encoding (default: mps — 780x faster than cpu).",
    )
    ap.add_argument(
        "--image-dir", default=None, metavar="DIR",
        help="Custom image directory (default: test_assets/images/).",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List corpus images and exit.",
    )
    args = ap.parse_args()

    # Resolve image list
    if args.image_dir:
        image_dir = Path(args.image_dir)
        images = [
            str(p) for p in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
            if "circle" not in p.name and "square" not in p.name  # exclude synthetic
        ]
    else:
        images = CORPUS_IMAGES

    if args.list:
        print(f"Corpus ({len(images)} images, schema v{FIXTURE_SCHEMA_VERSION}):")
        for img in images:
            exists = "✓" if Path(img).exists() else "✗ MISSING"
            print(f"  {exists}  {img}")
        return

    variants = [args.variant] if args.variant else ALL_VARIANTS

    print(f"Building fixtures — schema v{FIXTURE_SCHEMA_VERSION}")
    print(f"Corpus: {len(images)} images")
    print(f"Variants: {', '.join(variants)}")
    print(f"Device: {args.device}")
    print(f"Force rebuild: {args.force}")

    total_t0 = time.time()
    for variant in variants:
        build_variant(variant, images, force=args.force, device=args.device)

    print(f"\n{'='*60}")
    print(f"Total time: {time.time()-total_t0:.1f}s")
    print(f"Fixtures stored in: {FIXTURE_CACHE_DIR}/")


if __name__ == "__main__":
    main()
