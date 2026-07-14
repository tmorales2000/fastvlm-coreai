#!/usr/bin/env python3
"""
probe_vlm_config.py — Probe any VLM HF config for native resolution and preprocessing metadata.

Answers:
  1. What does the vision tower config say about input resolution?
  2. Does the model support dynamic spatial dimensions?
  3. What tiling/patch parameters are defined?
  4. What does the image processor config look like?
  5. What safetensors files exist and what are the vision tower weight shapes?

Known Qwen3-VL variants on HuggingFace:
  Qwen/Qwen3-VL-2B-Instruct   (default)
  Qwen/Qwen3-VL-7B-Instruct
  Qwen/Qwen3-VL-32B-Instruct
  Qwen/Qwen3-VL-72B-Instruct

Usage:
    # Default: probe Qwen3-VL-2B-Instruct from HF (downloads configs only, not weights)
    python scripts/probe_qwen3vl.py

    # Probe a specific variant
    python scripts/probe_qwen3vl.py --model-id Qwen/Qwen3-VL-7B-Instruct

    # Probe any VLM by model ID
    python scripts/probe_qwen3vl.py --model-id Qwen/Qwen2-VL-7B-Instruct
    python scripts/probe_qwen3vl.py --model-id apple/FastVLM-0.5B

    # Probe from local weights directory
    python scripts/probe_qwen3vl.py --local-dir ~/path/to/weights

    # Probe from HF cache (faster if already downloaded)
    python scripts/probe_qwen3vl.py \
      --local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/latest
"""

import argparse
import json
import sys
from pathlib import Path


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
        if k.lower() in {vk.lower() for vk in vision_keys} or "image" in k.lower() or "vision" in k.lower() or "patch" in k.lower() or "pixel" in k.lower():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                probe_config(v, indent + 1)
            else:
                print(f"{prefix}{k}: {v}")


def probe_from_hf(model_id: str) -> None:
    """Download and probe config from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    import json

    print(f"Probing {model_id} from HuggingFace Hub...")
    print()

    # Download config files
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
    """Probe config from local directory."""
    print(f"Probing {local_dir}...")
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
    """Probe safetensors weight shapes for vision tower."""
    import safetensors.torch as st

    print(f"{'='*60}")
    print(f"  safetensors — vision tower weights")
    print(f"{'='*60}")

    # Find all safetensors files
    st_files = sorted(local_dir.glob("*.safetensors"))
    if not st_files:
        # Check index
        index_path = local_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            # Find shards that contain vision weights
            vision_shards = set()
            for key, shard in index["weight_map"].items():
                if "visual" in key or "vision" in key or "patch_embed" in key:
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
        # Group by component
        components = {}
        for name, shape in sorted(vision_weights.items()):
            parts = name.split(".")
            component = ".".join(parts[:3]) if len(parts) >= 3 else parts[0]
            if component not in components:
                components[component] = []
            components[component].append((name, shape))

        for component, weights in sorted(components.items()):
            print(f"\n  [{component}]")
            for name, shape in weights[:5]:  # Show first 5 per component
                print(f"    {name}: {shape}")
            if len(weights) > 5:
                print(f"    ... and {len(weights)-5} more")
    else:
        print("  No vision tower weights found in safetensors files")

    print()


def probe_image_processor_class(model_id: str) -> None:
    """Try to load the actual image processor and inspect it."""
    print(f"{'='*60}")
    print(f"  Image processor class inspection")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen3-VL-2B-Instruct). "
                             "Works with any VLM: Qwen3-VL-7B, Qwen2-VL, FastVLM, etc.")
    parser.add_argument("--local-dir", type=Path, default=None,
                        help="Local directory with downloaded weights")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  Qwen3-VL Native Resolution Probe")
    print(f"  Model: {args.model_id}")
    print(f"  (probe any VLM with --model-id)")
    print(f"{'#'*60}\n")

    if args.local_dir and args.local_dir.exists():
        probe_from_local(args.local_dir)
        probe_safetensors(args.local_dir)
    else:
        probe_from_hf(args.model_id)

    probe_image_processor_class(args.model_id)


if __name__ == "__main__":
    main()
