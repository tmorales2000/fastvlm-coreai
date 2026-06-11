"""
discover_weights.py — FastVLM weight architecture discovery.

Run from the repo root with the project venv active:
    python scripts/discover_weights.py

Writes per-variant output to discovery/fastvlm-{variant}.txt
and prints to stdout. Commit the discovery/ directory after running.
"""

import json
import glob
import os

from safetensors import safe_open


def discover(model_dir: str, out_file: str | None = None) -> None:
    lines = [f"\n=== {model_dir} ==="]

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    lines.append("\n--- config.json ---")
    for k in [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "rope_theta",
        "rms_norm_eps",
        "mm_projector_type",
        "mm_hidden_size",
        "vision_config",
    ]:
        if k in cfg:
            lines.append(f"  {k}: {cfg[k]}")

    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    weights: dict[str, tuple[list, str]] = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                weights[key] = (list(t.shape), str(t.dtype))

    sections = [
        ("VISION TOWER", "vision_tower"),
        ("MM PROJECTOR", "mm_projector"),
        ("LANGUAGE MODEL", "language_model"),
        ("OTHER", None),
    ]
    for section, prefix in sections:
        if prefix:
            keys = [k for k in weights if prefix in k]
        else:
            keys = [
                k
                for k in weights
                if not any(
                    p in k for p in ["vision_tower", "mm_projector", "language_model"]
                )
            ]
        lines.append(f"\n--- {section} ({len(keys)} tensors) ---")
        for k in sorted(keys):
            shape, dtype = weights[k]
            lines.append(f"  {k}: {shape}  [{dtype}]")

    output = "\n".join(lines)
    print(output)
    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            f.write(output)
        print(f"\nWrote: {out_file}")


if __name__ == "__main__":
    for variant in ["fastvlm-1.5b", "fastvlm-0.5b", "fastvlm-7b"]:
        path = os.path.join("weights", variant)
        if os.path.exists(path):
            discover(path, out_file=f"discovery/{variant}.txt")
        else:
            print(f"Skipping {variant} — not found at {path}")
