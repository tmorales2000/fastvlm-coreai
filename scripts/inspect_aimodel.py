"""
inspect_aimodel.py — Inspect FastVLM bundle or individual .aimodel files.

Understands the 3-component VLM bundle format:
  vision.aimodel  — encode_image + project entrypoints
  embed.aimodel   — embed_tokens entrypoint
  {variant}.aimodel — main (decoder) entrypoint

USAGE:
  # Inspect full bundle:
  python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset

  # Inspect individual .aimodel:
  python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset/vision.aimodel
  python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset/embed.aimodel
  python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset/fastvlm-0.5b.aimodel
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from coreai.runtime import AIModel


def _fmt_shape(shape: list) -> str:
    return "[" + ", ".join(str(d) if d >= 0 else "-1" for d in shape) + "]"


def _check_dynamic(shape: list, dim: int, name: str) -> str:
    if len(shape) > dim and shape[dim] < 0:
        return f"  ✓ dim {dim} = -1 (dynamic)"
    elif len(shape) > dim:
        return f"  ⚠ dim {dim} = {shape[dim]} (static — expected dynamic for GrowingKVCache)"
    return ""


async def inspect_aimodel(path: Path) -> bool:
    """Inspect a single .aimodel file. Returns True if all checks pass."""
    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"{'='*60}")

    model = await AIModel.load(path)
    all_ok = True

    for fn_name in model.function_names:
        fn   = model.load_function(fn_name)
        desc = fn.desc

        print(f"\n  [{fn_name}]")

        # Inputs
        for iname in desc.input_names:
            d     = desc.input_descriptor(name=iname)
            shape = list(d.shape)
            print(f"    input   {iname:<25} dtype={str(d.dtype):<12} shape={_fmt_shape(shape)}")

        # Outputs
        for oname in desc.output_names:
            d     = desc.output_descriptor(name=oname)
            shape = list(d.shape)
            print(f"    output  {oname:<25} dtype={str(d.dtype):<12} shape={_fmt_shape(shape)}")

        # States
        for sname in desc.state_names:
            d     = desc.state_descriptor(name=sname)
            shape = list(d.shape)
            print(f"    state   {sname:<25} dtype={str(d.dtype):<12} shape={_fmt_shape(shape)}")

            if sname in ("k_cache", "v_cache"):
                # seq dim is axis 3 in Apple's 5D layout [n_layers,1,n_kv,seq,head]
                note = _check_dynamic(shape, 3, sname)
                if note:
                    print(f"    {note}")
                    if "-1" in note:
                        print(f"    ✓ {sname}: GrowingKVCache will be selected")
                    else:
                        print(f"    ⚠ {sname}: StaticKVCache will be used")
                        # all_ok = False  # static KV is acceptable, not a failure

        # Entrypoint-specific checks
        if fn_name == "encode_image":
            ok = "pixel_values" in desc.input_names and "image_features" in desc.output_names
            print(f"    {'✓' if ok else '✗'} encode_image contract: pixel_values → image_features")
            all_ok &= ok

        elif fn_name == "project":
            ok = len(desc.input_names) >= 1 and "projected_features" in desc.output_names
            print(f"    {'✓' if ok else '✗'} project contract: x → projected_features")
            all_ok &= ok

        elif fn_name == "embed_tokens":
            ok = "input_ids" in desc.input_names and "embeddings" in desc.output_names
            print(f"    {'✓' if ok else '✗'} embed_tokens contract: input_ids → embeddings")
            # Check int32 input
            if "input_ids" in desc.input_names:
                d = desc.input_descriptor(name="input_ids")
                if str(d.dtype) == "int32":
                    print(f"    ✓ input_ids dtype = int32")
                else:
                    print(f"    ⚠ input_ids dtype = {d.dtype} (expected int32)")
                    all_ok = False
            all_ok &= ok

        elif fn_name == "main":
            # Decoder: expects inputs_embeds + position_ids, stateful KV
            has_embeds = "inputs_embeds" in desc.input_names or "in_embeddings" in desc.input_names
            has_pos    = "position_ids" in desc.input_names
            has_kv     = set(desc.state_names) >= {"k_cache", "v_cache"}
            has_logits = "logits" in desc.output_names

            print(f"    {'✓' if has_embeds else '✗'} inputs_embeds input present")
            print(f"    {'✓' if has_pos    else '✗'} position_ids input present")
            print(f"    {'✓' if has_kv     else '✗'} k_cache + v_cache states present")
            print(f"    {'✓' if has_logits else '✗'} logits output present")
            if not has_embeds:
                print(f"    ⚠ decoder expects inputs_embeds, not input_ids")
            all_ok &= has_embeds and has_pos and has_kv and has_logits

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  [{status}] {path.name}")
    return all_ok


async def inspect_bundle(bundle_path: Path) -> bool:
    """Inspect a full .vlmasset bundle directory."""
    print(f"\n{'#'*60}")
    print(f"  FastVLM Bundle: {bundle_path.name}")
    print(f"{'#'*60}")

    # Read metadata.json
    meta_path = bundle_path / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"\n  Bundle metadata:")
        print(f"    kind:    {meta.get('kind', '?')}")
        print(f"    name:    {meta.get('name', '?')}")
        print(f"    assets:  {list(meta.get('assets', {}).keys())}")
        vis = meta.get("vision", {})
        if vis:
            print(f"    vision:  image_size={vis.get('image_size')}, "
                  f"patch_size={vis.get('patch_size')}, "
                  f"image_token_count={vis.get('image_token_count')}, "
                  f"image_token_id={vis.get('image_token_id')}")
            # Validate image_token_id
            img_tok_id = vis.get("image_token_id")
            # Check tokenizer has the image token
            tok_dir = bundle_path / "tokenizer"
            if tok_dir.exists():
                try:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(str(tok_dir))
                    actual_id = tok.convert_tokens_to_ids("<image>")
                    if actual_id == img_tok_id:
                        print(f"    ✓ tokenizer <image> token ID matches metadata ({img_tok_id})")
                    else:
                        print(f"    ✗ tokenizer <image> ID ({actual_id}) ≠ metadata ({img_tok_id})")
                except Exception as e:
                    print(f"    ⚠ could not verify tokenizer: {e}")
    else:
        print(f"  ⚠ No metadata.json found")

    # Inspect each .aimodel in the bundle
    all_ok = True
    for aimodel_path in sorted(bundle_path.glob("*.aimodel")):
        ok = await inspect_aimodel(aimodel_path)
        all_ok &= ok

    status = "PASS" if all_ok else "FAIL"
    print(f"\n{'#'*60}")
    print(f"  Bundle [{status}]: {bundle_path.name}")
    print(f"{'#'*60}\n")
    return all_ok


async def main_async(path: Path) -> bool:
    if path.suffix == ".vlmasset" or (path.is_dir() and not path.suffix == ".aimodel"):
        return await inspect_bundle(path)
    else:
        return await inspect_aimodel(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Path to .vlmasset bundle or individual .aimodel file"
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    passed = asyncio.run(main_async(args.path))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
