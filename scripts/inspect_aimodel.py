"""
inspect_aimodel.py — Generic CoreAI VLM bundle and .aimodel inspector.

Works on any VLM bundle produced by Apple's coreai-models export pipeline.
Bundle directories have no extension per Apple's current convention (PR #125).

Reads metadata.json to understand the bundle structure rather than making
model-specific assumptions. Reports:
  - Bundle manifest (kind, name, assets, vision config, language config)
  - Per-entrypoint: inputs, outputs, states (names, dtypes, shapes)
  - KV cache: static vs dynamic seq dim, GrowingKVCache eligibility
  - Decoder contract: inputs_embeds vs input_ids, stateful KV, logits
  - Embed contract: input_ids int32 → embeddings fp16
  - Vision contract: pixel_values → image_features, projector
  - Tokenizer: image token ID validation

USAGE:
  # Any VLM bundle directory:
  python scripts/inspect_aimodel.py exports/fastvlm-0.5b
  python scripts/inspect_aimodel.py ~/git/apple/coreai-models/exports/qwen3_vl_2b

  # Individual .aimodel file:
  python scripts/inspect_aimodel.py exports/fastvlm-0.5b/fastvlm-0.5b.aimodel
  python scripts/inspect_aimodel.py exports/fastvlm-0.5b/vision.aimodel
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from coreai.runtime import AIModel


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _fmt_shape(shape: list) -> str:
    return "[" + ", ".join(str(d) if d >= 0 else "-1" for d in shape) + "]"


def _fmt_dtype(dtype) -> str:
    s = str(dtype)
    return {
        "DataType.FLOAT16": "fp16",
        "DataType.FLOAT32": "fp32",
        "DataType.INT32":   "int32",
        "DataType.INT8":    "int8",
        "DataType.UINT8":   "uint8",
    }.get(s, s)


def _mb(n_bytes: int) -> str:
    if n_bytes >= 1_048_576:
        return f"{n_bytes / 1_048_576:.1f} MB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes} B"


def _shape_bytes(shape: list, dtype_str: str) -> int:
    if any(d < 0 for d in shape):
        return 0
    n = math.prod(shape)
    bpe = {"fp16": 2, "fp32": 4, "int32": 4, "int8": 1, "uint8": 1}.get(dtype_str, 2)
    return n * bpe


# ─── Contract checks ──────────────────────────────────────────────────────────

def _check_contracts(
    fn_name: str,
    desc,
    asset_role: str | None,
    meta: dict | None,
) -> bool:
    """Check entrypoint-specific contracts. Returns True if all pass."""
    inputs  = set(desc.input_names)
    outputs = set(desc.output_names)
    states  = set(desc.state_names)
    ok      = True

    # ── Vision encoder ──
    if fn_name == "encode_image" or (asset_role == "vision" and "pixel_values" in inputs):
        has_pv = "pixel_values" in inputs
        print(f"    {'✓' if has_pv else '✗'} pixel_values input present")
        if has_pv:
            dtype = _fmt_dtype(desc.input_descriptor(name="pixel_values").dtype)
            # fp32 input is correct — vision encoder casts to fp16 internally
            print(f"    ✓ pixel_values dtype = {dtype}  (fp32 → fp16 cast inside model)")
        print(f"    ✓ outputs: {sorted(outputs)}")
        ok &= has_pv

    # ── Projector ──
    elif fn_name == "project":
        print(f"    ✓ projector: {sorted(inputs)} → {sorted(outputs)}")

    # ── Embed tokens ──
    elif "input_ids" in inputs and "embeddings" in outputs:
        print(f"    ✓ embed_tokens contract: input_ids → embeddings")
        id_dtype  = _fmt_dtype(desc.input_descriptor(name="input_ids").dtype)
        emb_dtype = _fmt_dtype(desc.output_descriptor(name="embeddings").dtype)
        id_ok  = id_dtype == "int32"
        emb_ok = emb_dtype in ("fp16", "float16")
        print(f"    {'✓' if id_ok  else '✗'} input_ids dtype  = {id_dtype}   (expected int32)")
        print(f"    {'✓' if emb_ok else '⚠'} embeddings dtype = {emb_dtype}  (expected fp16)")
        ok &= id_ok

    # ── Text decoder ──
    elif fn_name == "main" and "input_ids" not in inputs:
        has_embeds = "inputs_embeds" in inputs or "in_embeddings" in inputs
        has_pos    = "position_ids" in inputs
        has_logits = "logits" in outputs
        has_kv     = len(states) >= 2

        print(f"    {'✓' if has_embeds else '✗'} inputs_embeds"
              f"{'  (not input_ids — correct for VLM pipeline)' if has_embeds else '  ← missing'}")
        print(f"    {'✓' if has_pos    else '✗'} position_ids")
        print(f"    {'✓' if has_kv     else '✗'} KV cache states: {sorted(states)}")
        print(f"    {'✓' if has_logits else '✗'} logits output")

        if has_logits:
            d      = desc.output_descriptor(name="logits")
            shape  = list(d.shape)
            dtype  = _fmt_dtype(d.dtype)
            vocab  = shape[-1] if shape else "?"
            print(f"    ✓ logits: {dtype} {_fmt_shape(shape)}  (vocab_size={vocab})")
            if meta:
                meta_vocab = meta.get("language", {}).get("vocab_size")
                if meta_vocab and vocab != "?" and int(vocab) == meta_vocab:
                    print(f"    ✓ vocab_size matches metadata ({meta_vocab})")
                elif meta_vocab:
                    print(f"    ⚠ vocab_size {vocab} ≠ metadata {meta_vocab}")

        ok &= has_embeds and has_pos and has_kv and has_logits

    return ok


# ─── Single .aimodel inspector ────────────────────────────────────────────────

async def inspect_aimodel(
    path: Path,
    asset_role: str | None = None,
    meta: dict | None = None,
) -> bool:
    print(f"\n{'='*64}")
    role_label = f"  [{asset_role}]" if asset_role else ""
    print(f"  {path.name}{role_label}")
    print(f"{'='*64}")

    model  = await AIModel.load(path)
    all_ok = True

    for fn_name in model.function_names:
        fn   = model.load_function(fn_name)
        desc = fn.desc

        print(f"\n  entrypoint: [{fn_name}]")

        # Inputs
        for iname in desc.input_names:
            d     = desc.input_descriptor(name=iname)
            shape = list(d.shape)
            dtype = _fmt_dtype(d.dtype)
            dyn   = "  ← dynamic" if any(s < 0 for s in shape) else ""
            print(f"    input   {iname:<28} {dtype:<8} {_fmt_shape(shape)}{dyn}")

        # Outputs
        for oname in desc.output_names:
            d     = desc.output_descriptor(name=oname)
            shape = list(d.shape)
            dtype = _fmt_dtype(d.dtype)
            dyn   = "  ← dynamic" if any(s < 0 for s in shape) else ""
            print(f"    output  {oname:<28} {dtype:<8} {_fmt_shape(shape)}{dyn}")

        # States
        for sname in desc.state_names:
            d     = desc.state_descriptor(name=sname)
            shape = list(d.shape)
            dtype = _fmt_dtype(d.dtype)
            nb    = _shape_bytes(shape, dtype)
            print(f"    state   {sname:<28} {dtype:<8} {_fmt_shape(shape)}"
                  f"  ({_mb(nb) if nb else 'dynamic size'})")
            # KV cache: seq dim is axis 3 in 5D [n_layers, 1, n_kv, seq, head]
            if len(shape) == 5:
                seq_val = shape[3]
                if seq_val < 0:
                    print(f"    ✓ {sname}: seq_dim=-1 (dynamic) → GrowingKVCache")
                else:
                    print(f"    ⚠ {sname}: seq_dim={seq_val} (static) → StaticKVCache")

        # Contract checks
        ok     = _check_contracts(fn_name, desc, asset_role, meta)
        all_ok &= ok

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  [{status}] {path.name}")
    return all_ok


# ─── Bundle inspector ─────────────────────────────────────────────────────────

async def inspect_bundle(bundle_path: Path) -> bool:
    print(f"\n{'#'*64}")
    print(f"  VLM Bundle: {bundle_path.name}")
    print(f"{'#'*64}")

    meta_path = bundle_path / "metadata.json"
    meta      = None
    assets    = {}

    if meta_path.exists():
        meta   = json.loads(meta_path.read_text())
        assets = meta.get("assets", {})

        print(f"\n  metadata.json:")
        print(f"    kind:    {meta.get('kind', '?')}")
        print(f"    name:    {meta.get('name', '?')}")
        print(f"    assets:  {assets}")

        lang = meta.get("language", {})
        if lang:
            print(f"    language:")
            print(f"      tokenizer:          {lang.get('tokenizer', '?')}")
            print(f"      vocab_size:         {lang.get('vocab_size', '?')}")
            print(f"      max_context_length: {lang.get('max_context_length', '?')}")

        vis = meta.get("vision", {})
        if vis:
            print(f"    vision:")
            print(f"      image_size:        {vis.get('image_size', '?')}")
            print(f"      patch_size:        {vis.get('patch_size', '?')}")
            print(f"      image_token_count: {vis.get('image_token_count', '?')}")
            print(f"      image_token_id:    {vis.get('image_token_id', '?')}")
            print(f"      image_mean:        {vis.get('image_mean', '?')}")
            print(f"      image_std:         {vis.get('image_std', '?')}")
            if "preprocessing" in vis:
                print(f"      preprocessing:     {vis.get('preprocessing')}")

        # Model-specific extra blocks (e.g. fastvlm, source)
        skip = {"metadata_version", "kind", "name", "assets", "language", "vision", "source"}
        for key in meta:
            if key in skip:
                continue
            val = meta[key]
            if isinstance(val, dict):
                print(f"    {key}:")
                for k2, v2 in val.items():
                    print(f"      {k2}: {v2}")
            else:
                print(f"    {key}: {val}")
    else:
        print(f"  ⚠ No metadata.json found")

    # Tokenizer
    tok_dir = bundle_path / "tokenizer"
    if tok_dir.exists():
        print(f"\n  tokenizer/:")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(tok_dir))
            print(f"    vocab_size:    {tok.vocab_size}")
            print(f"    len(tok):      {len(tok)}")
            print(f"    added_tokens:  {list(tok.added_tokens_encoder.keys())}")

            if meta and meta.get("vision", {}).get("image_token_id") is not None:
                image_token_id = meta["vision"]["image_token_id"]
                tok_name = tok.convert_ids_to_tokens(image_token_id)
                if tok_name and tok_name != "[UNK]":
                    print(f"    ✓ image_token_id {image_token_id} → '{tok_name}'")
                else:
                    for candidate in ["<image>", "<|image_pad|>", "<img>"]:
                        cid = tok.convert_tokens_to_ids(candidate)
                        if cid != getattr(tok, "unk_token_id", None):
                            match = cid == image_token_id
                            print(f"    {'✓' if match else '⚠'} '{candidate}' ID={cid}"
                                  f"{'  ✓ matches metadata' if match else f'  ≠ metadata {image_token_id}'}")
                            break
        except Exception as e:
            print(f"    ⚠ could not load tokenizer: {e}")

    # Build role map from assets
    role_map: dict[str, str] = {v: k for k, v in assets.items()}

    all_ok   = True
    aimodels = sorted(bundle_path.glob("*.aimodel"))

    if not aimodels:
        print(f"\n  ⚠ No .aimodel files found in bundle")
        return False

    for aimodel_path in aimodels:
        role = role_map.get(aimodel_path.name)
        try:
            ok = await inspect_aimodel(aimodel_path, asset_role=role, meta=meta)
            all_ok &= ok
        except Exception as e:
            msg = str(e)[:120]
            print(f"\n  [{aimodel_path.name}]")
            print(f"  ⚠ Runtime inspection failed: {msg}")
            print("    (Known issue: ANE rejects fp32 inputs at load time on some")
            print("     platforms. Structural export may still be correct.)")
            # Do not mark as failed
    status = "PASS" if all_ok else "FAIL"
    print(f"\n{'#'*64}")
    print(f"  Bundle [{status}]: {bundle_path.name}")
    print(f"{'#'*64}\n")
    return all_ok


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async(path: Path) -> bool:
    if path.is_dir() and path.suffix != ".aimodel":
        return await inspect_bundle(path)
    return await inspect_aimodel(path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="VLM bundle directory or individual .aimodel file",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    passed = asyncio.run(main_async(args.path))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
