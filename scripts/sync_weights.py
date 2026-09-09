"""
sync_weights.py — Download and verify model weights from HuggingFace.

Reads models.yaml to determine which models this project supports, then
downloads, verifies, and records provenance for each model's weights.

BEHAVIOR
--------
  If weights directory does not exist:
      → Download from HuggingFace (latest or pinned revision)
      → Record provenance in .provenance.json

  If weights directory exists but .provenance.json is missing:
      → Read revision from existing HF download metadata cache
      → Write .provenance.json (no re-download needed)

  If weights directory exists and .provenance.json is present:
      → Verify recorded revision matches HF metadata cache
      → Report status (no re-download unless --force)

  With --force:
      → Re-download regardless of what's present

PROVENANCE
----------
  After download or verification, writes weights/{model}/.provenance.json:
    {
      "hf_repo": "apple/FastVLM-0.5B",
      "hf_revision": "fc2d1a37...",
      "downloaded_at": "2026-09-08T...",
      "sync_weights_version": "1"
    }

  export_fastvlm.py reads this to stamp metadata.json with the HF revision,
  completing the provenance chain from weights → CoreAI bundle.

MLX WEIGHTS
-----------
  MLX quantized weights (apple/FastVLM-1.5B-INT8 etc.) are NOT managed here.
  They are handled exclusively by compare_weights.py and stored in the
  standard HF cache (~/.cache/huggingface/). They do not appear in --list.

USAGE
-----
  python scripts/sync_weights.py --list
  python scripts/sync_weights.py --variant 0.5b
  python scripts/sync_weights.py --variant qwen3-vl-2b
  python scripts/sync_weights.py --all
  python scripts/sync_weights.py --variant 0.5b --force
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT        = Path(__file__).parent.parent
MODELS_YAML      = REPO_ROOT / "models.yaml"
PROVENANCE_FILE  = ".provenance.json"
SYNC_VERSION     = "1"


# ── Provenance helpers ────────────────────────────────────────────────────────

def read_hf_revision_from_cache(weights_dir: Path) -> str | None:
    """Read the HF commit SHA from the local download metadata cache.

    When huggingface-cli downloads with --local-dir, it writes per-file
    metadata into .cache/huggingface/download/*.metadata. Each file contains:
      line 1: blob SHA (content hash)
      line 2: commit SHA (repo revision)  ← this is what we want
      line 3: download timestamp
    """
    meta = weights_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    try:
        lines = meta.read_text().strip().splitlines()
        return lines[1] if len(lines) >= 2 else None
    except FileNotFoundError:
        return None


def read_provenance(weights_dir: Path) -> dict | None:
    """Read existing .provenance.json, or None if absent."""
    p = weights_dir / PROVENANCE_FILE
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_provenance(weights_dir: Path, repo: str, revision: str) -> None:
    """Write .provenance.json into the weights directory."""
    record = {
        "hf_repo":              repo,
        "hf_revision":          revision,
        "downloaded_at":        datetime.now(timezone.utc).isoformat(),
        "sync_weights_version": SYNC_VERSION,
    }
    p = weights_dir / PROVENANCE_FILE
    p.write_text(json.dumps(record, indent=2) + "\n")


# ── Download ──────────────────────────────────────────────────────────────────

def download_model(entry: dict, force: bool = False) -> bool:
    """Download model weights from HuggingFace.

    Returns True on success, False on failure.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ERROR: huggingface_hub not installed.")
        print("         pip install huggingface-hub")
        return False

    repo      = entry["repo"]
    directory = REPO_ROOT / entry["directory"]
    revision  = entry.get("revision", None)

    print(f"  Downloading {repo}" + (f" @ {revision[:8]}" if revision else " (latest)"))
    print(f"  → {directory}")

    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(directory),
            revision=revision,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"  ERROR: Download failed: {e}")
        return False

    # Read the revision that was actually downloaded
    actual_revision = read_hf_revision_from_cache(directory)
    if actual_revision is None:
        print("  WARN: Could not determine downloaded revision from cache metadata.")
        actual_revision = revision or "unknown"

    write_provenance(directory, repo, actual_revision)
    print(f"  ✓ Downloaded — revision {actual_revision[:12]}")
    return True


# ── Per-model sync ────────────────────────────────────────────────────────────

def sync_model(name: str, entry: dict, force: bool = False) -> bool:
    """Sync one model — download, backfill provenance, or verify."""
    repo      = entry["repo"]
    directory = REPO_ROOT / entry["directory"]

    print(f"\n{'─'*60}")
    print(f"Model: {name}  ({entry.get('description', repo)})")

    # ── Case 1: directory does not exist → download
    if not directory.exists() or force:
        if force and directory.exists():
            print("  --force: re-downloading")
        return download_model(entry, force=force)

    # ── Case 2: directory exists but no .provenance.json → backfill
    provenance = read_provenance(directory)
    if provenance is None:
        revision = read_hf_revision_from_cache(directory)
        if revision:
            write_provenance(directory, repo, revision)
            print(f"  ✓ Weights present — provenance recorded (revision {revision[:12]})")
            print(f"    (run with --force to re-download)")
            return True
        else:
            print("  Weights present but no HF cache metadata found.")
            print("  Unable to determine revision. Re-downloading to record provenance.")
            return download_model(entry, force=True)

    # ── Case 3: directory and .provenance.json both exist → verify
    recorded  = provenance.get("hf_revision", "unknown")
    cache_rev = read_hf_revision_from_cache(directory)

    if cache_rev and cache_rev != recorded:
        print(f"  WARN: Revision mismatch!")
        print(f"    .provenance.json : {recorded[:12]}")
        print(f"    HF cache         : {cache_rev[:12]}")
        print(f"    Run with --force to re-download and reconcile.")
        return False

    downloaded_at = provenance.get("downloaded_at", "unknown")
    print(f"  ✓ Weights present and verified")
    print(f"    Revision    : {recorded[:12]}")
    print(f"    Downloaded  : {downloaded_at[:10]}")
    return True


# ── List ──────────────────────────────────────────────────────────────────────

def list_models(models: dict) -> None:
    """Print status table for all registered models."""
    print(f"\n{'Model':<20} {'Type':<12} {'Status':<14} {'Revision':<14} {'Downloaded'}")
    print("─" * 80)
    for name, entry in models.items():
        directory = REPO_ROOT / entry["directory"]
        model_type = entry.get("type", "unknown")

        if not directory.exists():
            status   = "not downloaded"
            revision = "─"
            date     = "─"
        else:
            provenance = read_provenance(directory)
            if provenance is None:
                # Weights exist but no provenance — try to read from cache
                rev = read_hf_revision_from_cache(directory)
                if rev:
                    status   = "no provenance"
                    revision = rev[:12]
                    date     = "─"
                else:
                    status   = "no provenance"
                    revision = "─"
                    date     = "─"
            else:
                status   = "✓ ready"
                revision = provenance.get("hf_revision", "unknown")[:12]
                date     = provenance.get("downloaded_at", "")[:10]

        print(f"{name:<20} {model_type:<12} {status:<14} {revision:<14} {date}")

    print()
    print("Run 'python scripts/sync_weights.py --variant <name>' to download a model.")
    print("Run 'python scripts/sync_weights.py --all' to download all models.")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_registry() -> dict:
    """Load and validate models.yaml."""
    if not MODELS_YAML.exists():
        print(f"ERROR: {MODELS_YAML} not found.")
        print("       This file should be in the repository root.")
        sys.exit(1)
    with open(MODELS_YAML) as f:
        data = yaml.safe_load(f)
    models = data.get("models", {})
    if not models:
        print("ERROR: No models defined in models.yaml.")
        sys.exit(1)
    return models


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list", action="store_true",
        help="Show download status for all registered models.",
    )
    group.add_argument(
        "--variant", metavar="NAME",
        help="Download/verify one model by registry name (e.g. fastvlm-0.5b).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Download/verify all registered models.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-download even if weights are already present.",
    )

    args = ap.parse_args()
    models = load_registry()

    if args.list:
        list_models(models)
        return

    if args.variant:
        if args.variant not in models:
            # Allow short names like "0.5b" as alias for "fastvlm-0.5b"
            matches = [k for k in models if k.endswith(f"-{args.variant}") or k == args.variant]
            if len(matches) == 1:
                args.variant = matches[0]
            elif len(matches) > 1:
                print(f"ERROR: '{args.variant}' matches multiple models: {matches}")
                print("       Use the full name.")
                sys.exit(1)
            else:
                print(f"ERROR: '{args.variant}' not found in models.yaml.")
                print(f"       Known models: {', '.join(models.keys())}")
                sys.exit(1)

        ok = sync_model(args.variant, models[args.variant], force=args.force)
        sys.exit(0 if ok else 1)

    if args.all:
        results = {}
        for name, entry in models.items():
            results[name] = sync_model(name, entry, force=args.force)

        print(f"\n{'═'*60}")
        print("SYNC COMPLETE")
        print(f"{'═'*60}")
        for name, ok in results.items():
            mark = "✓" if ok else "✗"
            print(f"  {mark} {name}")
        print()
        if not all(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
