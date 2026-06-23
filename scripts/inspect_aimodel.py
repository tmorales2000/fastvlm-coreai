"""
inspect_aimodel.py — Inspect a FastVLM .aimodel export.

Verifies that the exported model contains the expected entrypoints,
input/output/state signatures, dtypes, and shapes.

USAGE
-----
  python scripts/inspect_aimodel.py exports/fastvlm-0.5b_fp16.aimodel
  python scripts/inspect_aimodel.py exports/fastvlm-1.5b_fp16_int8.aimodel
  python scripts/inspect_aimodel.py exports/fastvlm-7b_fp16_int4.aimodel
"""

import argparse
import asyncio
import sys
from pathlib import Path

from coreai.runtime import AIModel


EXPECTED_ENTRYPOINTS = {"vision_encode", "project", "decode"}
EXPECTED_DECODE_STATES = {"k_cache", "v_cache"}
EXPECTED_IO = {
    "vision_encode": {"inputs": ["pixel_values"], "outputs": ["image_features"]},
    "project":       {"inputs": ["x"],             "outputs": ["projected_features"]},
    "decode":        {"inputs": ["input_ids", "position_ids"], "outputs": ["logits"]},
}


async def inspect(model_path: str) -> bool:
    path = Path(model_path)
    size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9

    print(f"\n{'='*72}")
    print(f"Inspecting : {path.name}")
    print(f"Size       : {size_gb:.2f} GB")
    print("=" * 72)

    model = await AIModel.load(str(path))

    found_names = set(model.function_names)
    missing = EXPECTED_ENTRYPOINTS - found_names
    extra   = found_names - EXPECTED_ENTRYPOINTS

    print(f"\nEntrypoints: {sorted(found_names)}")
    if missing:
        print(f"  [FAIL] Missing: {sorted(missing)}")
    if extra:
        print(f"  [WARN] Unexpected: {sorted(extra)}")
    if not missing and not extra:
        print(f"  [PASS] All 3 expected entrypoints present.")

    all_ok = not missing
    flags = []

    for ep_name in sorted(found_names):
        fn   = model.load_function(ep_name)
        desc = fn.desc

        print(f"\n  {'─'*60}")
        print(f"  {ep_name}")
        print(f"  {'─'*60}")

        # Inputs
        exp_in = EXPECTED_IO.get(ep_name, {}).get("inputs", [])
        for iname in desc.input_names:
            d = desc.input_descriptor(iname)
            marker = "" if iname in exp_in else "  ← UNEXPECTED"
            print(f"    input   {iname:<20} dtype={str(d.dtype):<12} shape={list(d.shape)}{marker}")
        for exp in exp_in:
            if exp not in desc.input_names:
                print(f"    [FAIL] Expected input '{exp}' missing")
                all_ok = False

        # Outputs
        exp_out = EXPECTED_IO.get(ep_name, {}).get("outputs", [])
        for oname in desc.output_names:
            d = desc.output_descriptor(oname)
            marker = "" if oname in exp_out else "  ← UNEXPECTED"
            print(f"    output  {oname:<20} dtype={str(d.dtype):<12} shape={list(d.shape)}{marker}")
        for exp in exp_out:
            if exp not in desc.output_names:
                print(f"    [FAIL] Expected output '{exp}' missing")
                all_ok = False

        # States (decode only)
        if desc.state_names:
            for sname in desc.state_names:
                d = desc.state_descriptor(sname)
                print(f"    state   {sname:<20} dtype={str(d.dtype):<12} shape={list(d.shape)}")
            found_states = set(desc.state_names)
            missing_states = EXPECTED_DECODE_STATES - found_states
            if missing_states:
                print(f"    [FAIL] Missing states: {sorted(missing_states)}")
                all_ok = False
            else:
                print(f"    [PASS] k_cache and v_cache states present.")

        # dtype checks
        for iname in desc.input_names:
            d = desc.input_descriptor(iname)
            dtype = str(d.dtype)
            if iname in ("input_ids", "position_ids") and "int" not in dtype:
                flags.append(f"{ep_name}.{iname}: expected int, got {dtype}")
            elif iname not in ("input_ids", "position_ids") and dtype not in ("float16", "float32"):
                flags.append(f"{ep_name}.{iname}: unexpected dtype {dtype}")
        for oname in desc.output_names:
            d = desc.output_descriptor(oname)
            dtype = str(d.dtype)
            if dtype not in ("float16", "float32"):
                flags.append(f"{ep_name}.{oname}: unexpected dtype {dtype}")

    if flags:
        print(f"\n  [WARN] Dtype notes:")
        for f in flags:
            print(f"    {f}")

    print(f"\n{'='*72}")
    if all_ok:
        print(f"[PASS] {path.name}")
    else:
        print(f"[FAIL] {path.name} — see above.")
    print("=" * 72)
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a FastVLM .aimodel export.")
    ap.add_argument("model_path", help="Path to the .aimodel directory.")
    args = ap.parse_args()
    ok = asyncio.run(inspect(args.model_path))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
