#!/usr/bin/env python3
"""
fetch_test_images.py — Download standard benchmark images for FastVLM testing.

All images are PUBLIC DOMAIN (NASA or pre-1928). Images are NOT committed to
the repo due to file size. Run this script once after cloning.

Uses the Wikimedia Commons API to get direct download URLs, which avoids
CDN bot-blocking that affects direct curl/wget requests.

Usage:
    python scripts/fetch_test_images.py
    python scripts/fetch_test_images.py --force   # re-download even if exists
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
IMG_DIR   = REPO_ROOT / "test_assets" / "images"

# Image catalog: (Wikimedia Commons filename, local filename, description, test purpose)
IMAGES = [
    (
        "NASA-Apollo8-Dec24-Earthrise.jpg",
        "earthrise.jpg",
        "Earth rising over the Moon — Apollo 8, William Anders, 1968 (NASA)",
        "Color / spatial orientation — models should identify both bodies and describe orientation",
    ),
    (
        "Pale_Blue_Dot.png",
        "pale_blue_dot.png",
        "Earth from 3.7 billion miles — Voyager 1, NASA/JPL-Caltech, 1990",
        "Hallucination resistance — minimal content, models should NOT describe rich scene detail",
    ),
    (
        "Lunch_atop_a_Skyscraper.jpg",
        "lunch_skyscraper.jpg",
        "Ironworkers on a steel beam 840ft above Manhattan, 1932 (public domain)",
        "Scene / counting — models should count ~11 workers and describe the skyline below",
    ),
    (
        "HubbleDeepField.800px.jpg",
        "hubble_deep_field.jpg",
        "Hubble Deep Field — ~3000 galaxies, R. Williams / STScI / NASA, 1996",
        "Cosmic scale / galaxy recognition",
    ),
    (
        "The_Earth_seen_from_Apollo_17.jpg",
        "blue_marble.jpg",
        "The Blue Marble — Earth from Apollo 17, NASA, 1972",
        "Whole-Earth recognition — Africa and cloud patterns clearly visible",
    ),
    (
        "Lange-MigrantMother02.jpg",
        "migrant_mother.jpg",
        "Migrant Mother — Dorothea Lange / FSA / US Government, 1936",
        "Portrait detail — woman with children, tests emotion and context reading",
    ),
    (
        "The_Great_Wave_off_Kanagawa.jpg",
        "great_wave.jpg",
        "The Great Wave off Kanagawa — Katsushika Hokusai, c.1831 (public domain)",
        "Art recognition — iconic woodblock print, tests color, composition, and Mt. Fuji identification",
    ),
    (
        "Girl_with_a_Pearl_Earring.jpg",
        "girl_pearl_earring.jpg",
        "Girl with a Pearl Earring — Johannes Vermeer, c.1665 (public domain)",
        "Portrait detail test — tests color accuracy (blue/yellow headscarf, pearl earring)",
    ),
    (
        "Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg",
        "pillars_of_creation.jpg",
        "Pillars of Creation — Hubble Space Telescope, NASA/ESA, 2014",
        "Nebula / cosmic structure — tests ability to describe gas columns and star-forming region",
    ),
]

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"


def get_wikimedia_url(filename: str) -> str:
    """Query Wikimedia Commons API to get the direct download URL for a file."""
    api_url = (
        f"{WIKIMEDIA_API}?action=query"
        f"&titles=File:{urllib.request.quote(filename)}"
        f"&prop=imageinfo&iiprop=url&format=json"
    )
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "fastvlm-coreai/1.0 (https://github.com/tmorales2000/fastvlm-coreai)"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    pages = data["query"]["pages"]
    for page in pages.values():
        if "imageinfo" in page:
            return page["imageinfo"][0]["url"]
    raise ValueError(f"No imageinfo found for {filename}")


def download_image(url: str, dest: Path) -> None:
    """Download an image from a direct URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "fastvlm-coreai/1.0 (https://github.com/tmorales2000/fastvlm-coreai)"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 65536
        while chunk := resp.read(chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {pct:.0f}% ({downloaded // 1024}KB / {total // 1024}KB)", end="", flush=True)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if file already exists")
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching benchmark images → {IMG_DIR}/")
    print("(All images are public domain)")
    print()

    success = 0
    failed  = 0

    for wiki_name, local_name, description, purpose in IMAGES:
        dest = IMG_DIR / local_name
        print(f"  {local_name}")
        print(f"    {description}")

        if dest.exists() and not args.force:
            size = dest.stat().st_size // 1024
            print(f"    ✓ already exists ({size}KB) — use --force to re-download")
            success += 1
            continue

        try:
            print(f"    → querying Wikimedia API...", end=" ", flush=True)
            url = get_wikimedia_url(wiki_name)
            print(f"got URL")
            print(f"    → downloading...", end="")
            download_image(url, dest)
            size = dest.stat().st_size // 1024
            print(f"    ✓ saved ({size}KB)")
            success += 1
        except Exception as e:
            print(f"\n    ✗ failed: {e}")
            print(f"    Manual URL: https://commons.wikimedia.org/wiki/File:{wiki_name}")
            if dest.exists():
                dest.unlink()
            failed += 1
        print()

    print(f"{'='*50}")
    print(f"{success} downloaded, {failed} failed.")
    if failed:
        print("For failed images, open the manual URL in a browser,")
        print(f"click 'Download original file', and save to {IMG_DIR}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
