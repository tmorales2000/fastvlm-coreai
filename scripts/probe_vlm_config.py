#!/usr/bin/env python3
"""
probe_vlm_config.py — Probe any VLM HF config for native resolution and preprocessing metadata.

Answers:
  1. What does the vision tower config say about input resolution?
  2. Does the model support dynamic spatial dimensions?
  3. What tiling/patch parameters are defined?
  4. What does the image processor config look like?
  5. What safetensors files exist and what are the vision tower weight shapes?

Usage:
    # FastVLM (default) — probes local weights if available, else HF
    python scripts/probe_vlm_config.py
    python scripts/probe_vlm_config.py --variant 1.5b
    python scripts/probe_vlm_config.py --variant 7b

    # Qwen3-VL variants
    python scripts/probe_vlm_config.py --model qwen3-vl
    python scripts/probe_vlm_config.py --model qwen3-vl --variant 4b
    python scripts/probe_vlm_config.py --model qwen3-vl --variant 8b
    python scripts/probe_vlm_config.py --model qwen3-vl --variant 32b
    python scripts/probe_vlm_config.py --model qwen3-vl --variant 72b

    # Add a model to MODEL_REGISTRY for anything not listed above.
    # Use --local-dir to point at weights in a non-standard location.
    python scripts/probe_vlm_config.py --model qwen3-vl --local-dir ~/custom/weights
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Registry mapping (model, variant) → (hf_model_id, weights_subdir)
# weights_subdir is relative to REPO_ROOT/weights/
MODEL_REGISTRY: dict[tuple[str, str], tuple[str, str]] = {
    # FastVLM
    ("fastvlm", "0.5b"): ("apple/FastVLM-0.5B",  "fastvlm-0.5b"),
    ("fastvlm", "1.5b"): ("apple/FastVLM-1.5B",  "fastvlm-1.5b"),
    ("fastvlm", "7b"):   ("apple/FastVLM-7B",     "fastvlm-7b"),
    # Qwen3-VL
    ("qwen3-vl", "2b"):  ("Qwen/Qwen3-VL-2B-Instruct",  "qwen3-vl-2b"),
    ("qwen3-vl", "4b"):  ("Qwen/Qwen3-VL-4B-Instruct",  "qwen3-vl-4b"),
    ("qwen3-vl", "8b"):  ("Qwen/Qwen3-VL-8B-Instruct",  "qwen3-vl-8b"),
    ("qwen3-vl", "32b"): ("Qwen/Qwen3-VL-32B-Instruct", "qwen3-vl-32b"),
    ("qwen3-vl", "72b"): ("Qwen/Qwen3-VL-72B-Instruct", "qwen3-vl-72b"),
}

DEFAULT_VARIANTS = {
    "fastvlm":  "0.5b",
    "qwen3-vl": "2b",
}


def resolve_model(model: str, variant: str) -> tuple[str, Path | None]:
    """Resolve (model, variant) to (hf_model_id, local_weights_path | None)."""
    key = (model.lower(), variant.lower())
    if key not in MODEL_REGISTRY:
        known = [f"--model {m} --variant {v}" for m, v in MODEL_REGISTRY]
        print(f"ERROR: Unknown model/variant combination: {model!r} / {variant!r}", file=sys.stderr)
        print(f"Known combinations:", file=sys.stderr)
        for k in known:
            print(f"  {k}", file=sys.stderr)
        sys.exit(1)

    hf_model_id, weights_subdir = MODEL_REGISTRY[key]
    local_path = REPO_ROOT / "weights" / weights_subdir
    return hf_model_id, local_path if local_path.exists() else None


def probe_config(config: dict, indent: int = 0) -> None:
    """Recursively print config keys relevant to vision/image processing."""
    vision_keys = {
        "image_size", "patch_size", "num_patches", "num_positions",
        "image_token_count", "max_pixels", "min_pixels", "max_image_tokens",
        "temporal_patch_size", "spatial_patch_size", "spatial_merge_size",
        "vision_config", "visual", "image_processor_type",
        "do_resize", "do_center_crop", "do_normalize", "do_rescale",
        "size", "crop_size", "resample", "image_mean", "image_std",
        "rescale_factor", "image_grid_pinpoints", "mm_patch_merge_type",
        "image_aspect_ratio", "mm_vision_tower", "vision_tower",
        "model_type", "architectures", "num_image_tokens",
        "dynamic_image_size", "use_thumbnail", "max_dynamic_patch",
        "min_dynamic_patch",
    }

    prefix = "  " * indent
    for k, v in sorted(config.items()):
        if k.lower() in {vk.lower() for vk in vision_keys} or \
           any(t in k.lower() for t in ["image", "vision", "patch", "pixel"]):
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                probe_config(v, indent + 1)
            else:
                print(f"{prefix}{k}: {v}")


def probe_from_hf(model_id: str) -> None:
    """Download and probe config from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    print(f"Probing {model_id} from HuggingFace Hub...")
    print()

    for filename in ["config.json", "preprocessor_config.json",
                     "generation_config.json", "chat_template.json"]:
        try:
            path = hf_hub_download(model_id, filename)
            cfg = json.loads(Path(path).read_text())
            print(f"{'='*60}")
            print(f"  {filename}")
            print(f"{'='*60}")
            probe_config(cfg)
            print()
        except Exception as e:
            if "404" not in str(e) and "not found" not in str(e).lower():
                print(f"  {filename}: {e}")


def probe_from_local(local_dir: Path) -> None:
    """Probe config from local weights directory."""
    print(f"Probing local weights: {local_dir}")
    print()

    for filename in ["config.json", "preprocessor_config.json",
                     "generation_config.json"]:
        path = local_dir / filename
        if not path.exists():
            print(f"  {filename}: not found")
            continue

        cfg = json.loads(path.read_text())
        print(f"{'='*60}")
        print(f"  {filename}")
        print(f"{'='*60}")
        probe_config(cfg)
        print()


def probe_safetensors(local_dir: Path) -> None:
    """Probe safetensors weight shapes for vision tower weights."""
    import safetensors.torch as st

    print(f"{'='*60}")
    print(f"  safetensors — vision tower weights")
    print(f"{'='*60}")

    st_files = sorted(local_dir.glob("*.safetensors"))
    if not st_files:
        index_path = local_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            vision_shards = set()
            for key, shard in index["weight_map"].items():
                if any(k in key for k in ["visual", "vision", "patch_embed"]):
                    vision_shards.add(shard)
            st_files = [local_dir / s for s in sorted(vision_shards)]
            print(f"  Found {len(vision_shards)} vision shards in index")

    vision_weights = {}
    for st_file in st_files:
        if not st_file.exists():
            continue
        try:
            tensors = st.load_file(str(st_file))
            for name, tensor in tensors.items():
                if any(k in name for k in ["visual", "vision", "patch_embed",
                                            "image_encoder", "vit", "clip"]):
                    vision_weights[name] = list(tensor.shape)
        except Exception as e:
            print(f"  Error loading {st_file.name}: {e}")

    if vision_weights:
        components: dict[str, list] = {}
        for name, shape in sorted(vision_weights.items()):
            parts = name.split(".")
            component = ".".join(parts[:3]) if len(parts) >= 3 else parts[0]
            components.setdefault(component, []).append((name, shape))

        for component, weights in sorted(components.items()):
            print(f"\n  [{component}]")
            for name, shape in weights[:5]:
                print(f"    {name}: {shape}")
            if len(weights) > 5:
                print(f"    ... and {len(weights)-5} more")
    else:
        print("  No vision tower weights found")
    print()


def probe_image_processor_class(model_id: str) -> None:
    """Load the actual HF image processor and inspect its effective parameters."""
    print(f"{'='*60}")
    print(f"  Image processor class (effective parameters)")
    print(f"{'='*60}")

    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        print(f"  Class: {type(proc).__name__}")
        for attr in ["size", "crop_size", "do_resize", "do_center_crop",
                     "do_normalize", "image_mean", "image_std", "resample",
                     "rescale_factor", "max_pixels", "min_pixels",
                     "patch_size", "temporal_patch_size", "spatial_patch_size",
                     "merge_size", "image_grid_pinpoints"]:
            val = getattr(proc, attr, None)
            if val is not None:
                print(f"  {attr}: {val}")
    except Exception as e:
        print(f"  Could not load image processor: {e}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="fastvlm",
        help="Model family (default: fastvlm). Options: fastvlm, qwen3-vl",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Model variant, e.g. 0.5b, 1.5b, 7b for FastVLM; "
             "2b, 4b, 8b, 32b, 72b for Qwen3-VL. "
             "Defaults to 0.5b for fastvlm, 2b for qwen3-vl.",
    )
    parser.add_argument(
        "--local-dir", type=Path, default=None,
        help="Explicit local weights directory (overrides auto-resolution from --model/--variant).",
    )
    args = parser.parse_args()

    # Resolve model ID and local path from registry
    variant = args.variant or DEFAULT_VARIANTS.get(args.model.lower(), "0.5b")
    hf_model_id, resolved_local = resolve_model(args.model, variant)
    local_dir = args.local_dir or resolved_local

    print(f"\n{'#'*60}")
    print(f"  VLM Config Probe")
    print(f"  HF model ID: {hf_model_id}")
    if local_dir:
        print(f"  Local weights: {local_dir}")
    print(f"{'#'*60}\n")

    if local_dir and local_dir.exists():
        probe_from_local(local_dir)
        probe_safetensors(local_dir)
    else:
        if local_dir:
            print(f"[INFO] Local weights not found at {local_dir} — probing HF instead")
        probe_from_hf(hf_model_id)

    probe_image_processor_class(hf_model_id)


if __name__ == "__main__":
    main()
