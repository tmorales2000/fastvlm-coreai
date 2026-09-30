#!/usr/bin/env python3
"""
build_fixtures.py — Download corpus images and build decoder fixtures.

Reads verification/corpus.yaml to determine which images are needed,
downloads any that are missing from verification/images/, then runs
the full FastVLM multimodal pipeline to build cached decoder fixtures.

Fixtures are used by:
  - verify_decoder.py    (Phase 2: FP16 fidelity, Phase 4: compression quality)
  - compression_scanner.py  (KL divergence on real multimodal inputs)

Each variant has its own fixture cache because inputs_embeds dimensions
differ: 0.5B hidden=896, 1.5B hidden=1536, 7B hidden=3584.

Usage:
    # Download images + build fixtures for a specific variant
    python scripts/build_fixtures.py --variant 0.5b

    # Build for all downloaded variants
    python scripts/build_fixtures.py

    # Force rebuild (re-download images, ignore fixture cache)
    python scripts/build_fixtures.py --force

    # Download images only (no fixture build)
    python scripts/build_fixtures.py --images-only

    # List corpus and exit
    python scripts/build_fixtures.py --list

Notes:
    - Uses MPS by default (780x faster than CPU for vision encoding)
    - Falls back to CPU if MPS unavailable
    - Skips variants whose weights are not downloaded
    - Safe to interrupt and resume — cached fixtures are skipped
    - Images are downloaded from Wikimedia Commons API (avoids CDN blocking)
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from models import resolve_variant, argparse_choices, registry_keys

REPO_ROOT    = Path(__file__).parent.parent
CORPUS_YAML  = REPO_ROOT / "verification" / "corpus.yaml"
IMAGE_DIR    = REPO_ROOT / "verification" / "images"
ALL_VARIANTS = registry_keys()

sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Corpus loading ─────────────────────────────────────────────────────────────

def load_corpus() -> dict:
    with open(CORPUS_YAML) as f:
        return yaml.safe_load(f)


def corpus_image_paths(corpus: dict) -> list[str]:
    return [str(IMAGE_DIR / img["local"]) for img in corpus["images"]]


# ── Image download ─────────────────────────────────────────────────────────────

def _get_wikimedia_url(api_base: str, user_agent: str, filename: str) -> str:
    url = (
        f"{api_base}?action=query"
        f"&titles=File:{urllib.request.quote(filename)}"
        f"&prop=imageinfo&iiprop=url&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    for page in data["query"]["pages"].values():
        if "imageinfo" in page:
            return page["imageinfo"][0]["url"]
    raise ValueError(f"No imageinfo found for {filename}")


def _download_file(url: str, dest: Path, user_agent: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := resp.read(65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r    {downloaded/total*100:.0f}% ({downloaded//1024}KB/{total//1024}KB)",
                      end="", flush=True)
        print()


def fetch_images(corpus: dict, force: bool = False) -> int:
    """Download missing corpus images. Returns count of failures."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    api    = corpus["wikimedia_api"]
    ua     = corpus["user_agent"]
    failed = 0

    for entry in corpus["images"]:
        dest = IMAGE_DIR / entry["local"]
        if dest.exists() and not force:
            print(f"  ✓ {entry['local']} (cached)")
            continue
        print(f"  {entry['local']} — {entry['description']}")
        try:
            print(f"    → querying Wikimedia...", end=" ", flush=True)
            url = _get_wikimedia_url(api, ua, entry["wikimedia"])
            print("got URL")
            _download_file(url, dest, ua)
            print(f"    ✓ saved ({dest.stat().st_size//1024}KB)")
        except Exception as e:
            print(f"\n    ✗ failed: {e}")
            print(f"    Manual: https://commons.wikimedia.org/wiki/File:{entry['wikimedia']}")
            if dest.exists():
                dest.unlink()
            failed += 1

    return failed


# ── Fixture build ──────────────────────────────────────────────────────────────

def variant_weights_exist(variant: str) -> bool:
    from models import weights_dir as _weights_dir
    try:
        d = _weights_dir(variant)
        return d.is_dir() and any(d.glob("*.safetensors"))
    except (KeyError, Exception):
        return False


def build_variant(
    variant: str,
    images: list[str],
    force: bool = False,
    device: str = "mps",
) -> None:
    from fastvlm_fixtures import (
        FIXTURE_CACHE_DIR,
        FIXTURE_SCHEMA_VERSION,
        DEFAULT_PROMPT,
        build_corpus_fixtures,
    )

    print(f"\n{'='*60}")
    print(f"Variant: fastvlm-{variant}")
    print(f"{'='*60}")

    if not variant_weights_exist(variant):
        print(f"[SKIP] Weights not found for {variant}")
        print(f"       Download: python scripts/sync_weights.py --variant {variant}")
        return

    if force:
        cache_dir = REPO_ROOT / FIXTURE_CACHE_DIR
        deleted = list(cache_dir.glob(f"fastvlm-{variant}-*.pt"))
        for f in deleted:
            f.unlink()
        if deleted:
            print(f"[FORCE] Deleted {len(deleted)} cached fixture(s)")

    t0       = time.time()
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
        print(f"[WARN] {missing} image(s) missing — run without --images-only flag")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", choices=argparse_choices(), default=None,
                    help="Build fixtures for one variant only (default: all available).")
    ap.add_argument("--force", action="store_true",
                    help="Re-download images and rebuild all cached fixtures.")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"],
                    help="Compute device for vision encoding (default: mps).")
    ap.add_argument("--images-only", action="store_true",
                    help="Download images only — do not build fixtures.")
    ap.add_argument("--list", action="store_true",
                    help="List corpus images and exit.")
    args = ap.parse_args()

    corpus = load_corpus()
    images = corpus_image_paths(corpus)

    if args.list:
        from fastvlm_fixtures import FIXTURE_SCHEMA_VERSION
        print(f"Corpus ({len(images)} images, schema v{FIXTURE_SCHEMA_VERSION}):")
        for entry in corpus["images"]:
            path = IMAGE_DIR / entry["local"]
            mark = "✓" if path.exists() else "✗ MISSING"
            print(f"  {mark}  {entry['local']}  — {entry['purpose']}")
        return

    # Step 1: ensure images are present
    print(f"Corpus images → {IMAGE_DIR}/")
    failed = fetch_images(corpus, force=args.force)
    if failed:
        print(f"\n[WARN] {failed} image(s) failed to download.")
        print("       Fixtures may be incomplete.")

    if args.images_only:
        return

    # Step 2: build fixtures
    variants = [resolve_variant(args.variant)] if args.variant else ALL_VARIANTS
    print(f"\nBuilding fixtures — device: {args.device}")

    total_t0 = time.time()
    for variant in variants:
        build_variant(resolve_variant(variant), images, force=args.force, device=args.device)

    print(f"\n{'='*60}")
    print(f"Total time: {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    main()
