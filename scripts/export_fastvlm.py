"""
export_fastvlm.py — Export FastVLM to CoreAI VLM bundle format.

Follows Apple's authoritative recipe from:
  coreai-models/python/src/coreai_models/vlm/export.py

Produces a bundle directory (e.g. fastvlm-0.5b.vlmasset/) containing:
  {variant}.aimodel — text decoder (asset role: main)
  embed.aimodel      — token embedding lookup (asset role: embedding)
  vision.aimodel     — vision encoder + projector (asset role: vision)
  tokenizer/         — Qwen2 tokenizer + <image> special token
  metadata.json      — bundle manifest (kind=vlm)

KEY DESIGN DECISIONS (from vlm/export.py)
==========================================
- Uses export_to_coreai() from coreai_models.export.macos
- k_cache/v_cache: static (dynamic_shapes=None, matches Apple) or dynamic (seq_dim symbolic)
- static:  matches Apple vlm/export.py, StaticKVCache, ceiling fixed at export time
- dynamic: GrowingKVCache, Swift app controls ceiling, lower initial memory footprint
- position_ids length = QUERY_LEN + OFFSET at trace time
- seq_pos dim: min=QUERY_LEN=64, max=max_ctx-1
- KV_STATE_NAMES = ("k_cache", "v_cache")

USAGE:
  python scripts/export_fastvlm.py --variant 0.5b
  python scripts/export_fastvlm.py --variant 1.5b --quantize int8
  python scripts/export_fastvlm.py --variant 7b   --quantize int4
  python scripts/export_fastvlm.py --variant 0.5b --components vision embed decode

  # KV cache mode (static = Apple default, dynamic = GrowingKVCache):
  python scripts/export_fastvlm.py --variant 0.5b --kv-cache static   # default
  python scripts/export_fastvlm.py --variant 0.5b --kv-cache dynamic  # GrowingKVCache
  python scripts/export_fastvlm.py --variant 0.5b --kv-cache dynamic --max-context-length 8192
"""

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from transformers import AutoConfig, AutoTokenizer

from fastvlm_decoder import (
    KEY_CACHE_NAME,
    VALUE_CACHE_NAME,
    KV_STATE_NAMES,
    KVCache,
    FastVLMDecoder,
    FastVLMEmbedTokens,
)
from fastvlm_projector import FastVLMProjector
from fastvlm_vision_encoder import FastVLMVisionEncoder
from quantization import QUANTIZATION_LEVELS, apply_quantization

# ---------------------------------------------------------------------------
# Constants (matching vlm/export.py)
# ---------------------------------------------------------------------------
QUERY_LEN = 64   # trace-time query length
OFFSET    = 64   # trace-time position offset (position_ids length = QUERY_LEN + OFFSET)

IMAGE_TOKEN       = "<image>"
IMAGE_SIZE        = 1024
PATCH_SIZE        = 64
NUM_IMAGE_TOKENS  = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256
IMAGE_MEAN        = [0.0, 0.0, 0.0]   # confirmed no-op from llava_qwen.py
IMAGE_STD         = [1.0, 1.0, 1.0]
RESCALE_FACTOR    = 1.0

ALL_COMPONENTS = ["vision", "embed", "decode"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights_dir(variant: str) -> Path:
    return Path(__file__).parent.parent / "weights" / f"fastvlm-{variant}"


def _bundle_path(variant: str, output_dir: Path) -> Path:
    return output_dir / f"fastvlm-{variant}.vlmasset"


def _load_config(weights_dir: Path):
    cfg = AutoConfig.from_pretrained(str(weights_dir), trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None) or cfg
    return cfg, text_cfg


def _setup_tokenizer(weights_dir: Path, bundle_path: Path) -> int:
    tok = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    if IMAGE_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        print(f"[INFO] Added '{IMAGE_TOKEN}' to tokenizer")
    image_token_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"[INFO] '{IMAGE_TOKEN}' token ID: {image_token_id}")
    tok_dir = bundle_path / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(tok_dir))
    return image_token_id


def _write_bundle_metadata(
    bundle_path: Path,
    variant: str,
    text_cfg,
    image_token_id: int,
    assets: dict,
    max_ctx: int,
):
    metadata = {
        "metadata_version": "0.2",
        "kind": "vlm",
        "name": f"fastvlm-{variant}",
        "assets": assets,
        "language": {
            "tokenizer": f"fastvlm-{variant}",
            "vocab_size": text_cfg.vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": {
            "image_size":        IMAGE_SIZE,
            "patch_size":        PATCH_SIZE,
            "image_token_count": NUM_IMAGE_TOKENS,
            "image_token_id":    image_token_id,
            "image_mean":        IMAGE_MEAN,
            "image_std":         IMAGE_STD,
            "rescale_factor":    RESCALE_FACTOR,
        },
        "source": {
            "hf_model_id":      f"apple/FastVLM-{variant.upper()}",
            "model_definition": "torch",
        },
        "fastvlm": {
            "variant":             variant,
            "vocab_size":          text_cfg.vocab_size,
            "hidden_size":         text_cfg.hidden_size,
            "num_hidden_layers":   text_cfg.num_hidden_layers,
            "num_attention_heads": text_cfg.num_attention_heads,
            "num_key_value_heads": text_cfg.num_key_value_heads,
            "mm_hidden_size":      3072,
        },
    }
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote metadata.json")


# ---------------------------------------------------------------------------
# Vision export
# ---------------------------------------------------------------------------

def _export_vision(
    weights_dir: Path,
    config,
    bundle_path: Path,
    overwrite: bool,
):
    image_size = getattr(config, "image_size", IMAGE_SIZE)
    mm_hidden  = getattr(config, "mm_hidden_size", 3072)
    _, text_cfg = _load_config(weights_dir)

    vision_path = bundle_path / "vision.aimodel"
    if vision_path.exists():
        if not overwrite:
            raise FileExistsError(f"{vision_path} exists.")
        shutil.rmtree(vision_path)

    # Vision encoder (encode_image)
    print("[INFO] Tracing encode_image...")
    vision_model = FastVLMVisionEncoder.from_weights(
        config=config, weights_dir=str(weights_dir)
    ).eval()
    program = export_to_coreai(
        vision_model,
        {"pixel_values": torch.randn(1, 3, image_size, image_size)},
        dynamic_shapes=None,
        input_names=("pixel_values",),
        output_names=("image_features",),
    )

    # Projector (project) — add as second entrypoint
    print("[INFO] Tracing project...")
    proj_model = FastVLMProjector.from_weights(
        config=config, weights_dir=str(weights_dir)
    ).eval()

    from coreai_models.export.macos import _EXTERNALIZE_SPECS
    from coreai_models.export.mlir_ops import remove_functionalization, register_custom_torch_lowering
    from coreai_torch import TorchConverter

    converter = TorchConverter()

    def _export_fn_proj(module):
        with torch.no_grad():
            ep = torch.export.export(
                module, args=(),
                kwargs={"x": torch.randn(1, NUM_IMAGE_TOKENS, mm_hidden, dtype=torch.float16)}
            )
        import coreai_torch
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        remove_functionalization(ep)
        return ep

    converter.add_pytorch_module(
        proj_model,
        export_fn=_export_fn_proj,
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=("x",),
        output_names=("projected_features",),
        entrypoint_name="project",
    )
    register_custom_torch_lowering(converter)
    proj_program = converter.to_coreai()

    # Merge both into vision.aimodel — save vision first, add project entrypoints
    # For now save separately and combine via the bundle
    # Simplification: save vision encoder only, project is a separate call
    # Actually re-use export_to_coreai for encode_image then add project
    # The cleanest approach: two separate aimodels or merged via TorchConverter

    # Use TorchConverter to export both entrypoints together
    import coreai_torch as ct

    converter2 = ct.TorchConverter()

    def _export_fn_vision(module):
        with torch.no_grad():
            ep = torch.export.export(
                module, args=(),
                kwargs={"pixel_values": torch.randn(1, 3, image_size, image_size)}
            )
        ep = ep.run_decompositions(ct.get_decomp_table())
        remove_functionalization(ep)
        return ep

    converter2.add_pytorch_module(
        vision_model,
        export_fn=_export_fn_vision,
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=("pixel_values",),
        output_names=("image_features",),
        entrypoint_name="encode_image",
    )
    converter2.add_pytorch_module(
        proj_model,
        export_fn=_export_fn_proj,
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=("x",),
        output_names=("projected_features",),
        entrypoint_name="project",
    )
    register_custom_torch_lowering(converter2)
    program = converter2.to_coreai()
    program.optimize()

    meta = build_aimodel_metadata(f"FastVLM vision encoder + projector")
    program.save_asset(vision_path, meta)
    print(f"[INFO] Saved vision.aimodel")


# ---------------------------------------------------------------------------
# Embed export
# ---------------------------------------------------------------------------

def _export_embed(
    weights_dir: Path,
    text_cfg,
    bundle_path: Path,
    max_ctx: int,
    overwrite: bool,
):
    embed_path = bundle_path / "embed.aimodel"
    if embed_path.exists():
        if not overwrite:
            raise FileExistsError(f"{embed_path} exists.")
        shutil.rmtree(embed_path)

    print("[INFO] Tracing embed_tokens...")
    embed_model = FastVLMEmbedTokens.from_weights(str(weights_dir)).eval()
    program = export_to_coreai(
        embed_model,
        {"input_ids": torch.zeros(1, QUERY_LEN, dtype=torch.int32)},
        dynamic_shapes={"input_ids": {1: torch.export.Dim("embed_seq", max=max_ctx - 1)}},
        input_names=("input_ids",),
        output_names=("embeddings",),
        state_names=None,
    )
    program.optimize()
    meta = build_aimodel_metadata("FastVLM token embedding lookup")
    program.save_asset(embed_path, meta)
    print(f"[INFO] Saved embed.aimodel")


# ---------------------------------------------------------------------------
# Decode export
# ---------------------------------------------------------------------------

def _export_decode(
    weights_dir: Path,
    config,
    text_cfg,
    quantize: str | None,
    variant: str,
    bundle_path: Path,
    max_ctx: int,
    overwrite: bool,
    kv_cache: str = "static",
):
    """
    Export the Qwen2 decoder.

    kv_cache="static"  — matches Apple vlm/export.py exactly.
                         dynamic_shapes=None for caches, seq dim baked at max_ctx.
                         StaticKVCache in Swift. Ceiling fixed at export time.

    kv_cache="dynamic" — seq dim declared symbolic (torch.export.Dim).
                         GrowingKVCache in Swift. Starts small, doubles as needed.
                         Swift app controls the ceiling. Lower initial memory.

    Reference inputs:
      inputs_embeds: [1, QUERY_LEN, hidden]
      position_ids:  [1, QUERY_LEN + OFFSET]
      k_cache:       [n_layers, 1, n_kv_heads, max_ctx, head_dim]
      v_cache:       same
    """
    # Remove stale decoder files
    for stale in bundle_path.glob("fastvlm-*.aimodel"):
        shutil.rmtree(stale)

    model_path = bundle_path / f"fastvlm-{variant}.aimodel"
    if model_path.exists():
        shutil.rmtree(model_path)

    model = FastVLMDecoder.from_weights(text_cfg, str(weights_dir)).eval()

    if quantize:
        print(f"[INFO] Applying {quantize} quantization...")
        model = apply_quantization(model, level=QUANTIZATION_LEVELS[quantize])

    hidden     = text_cfg.hidden_size
    n_layers   = text_cfg.num_hidden_layers
    n_kv_heads = text_cfg.num_key_value_heads
    head_dim   = hidden // text_cfg.num_attention_heads

    # Build reference inputs exactly as vlm/export.py
    k_cache = torch.zeros(n_layers, 1, n_kv_heads, max_ctx, head_dim, dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)

    reference_inputs = {
        "inputs_embeds": torch.randn(1, QUERY_LEN, hidden, dtype=torch.float16),
        "position_ids":  torch.arange(QUERY_LEN + OFFSET, dtype=torch.int32).unsqueeze(0),
        KEY_CACHE_NAME:  k_cache,
        VALUE_CACHE_NAME: v_cache,
    }

    if kv_cache == "dynamic":
        # Dynamic seq dim — GrowingKVCache in Swift. Ceiling set by Swift app at runtime.
        # Cache starts at a small trace-time size; compiler treats seq dim as symbolic.
        TRACE_CACHE_SEQ = 256  # trace-time size — small so compiler sees it as growable
        k_cache = torch.zeros(n_layers, 1, n_kv_heads, TRACE_CACHE_SEQ, head_dim, dtype=torch.float16)
        v_cache = torch.zeros_like(k_cache)
        reference_inputs[KEY_CACHE_NAME]   = k_cache
        reference_inputs[VALUE_CACHE_NAME] = v_cache
        cache_dim = torch.export.Dim("cache_seq", min=0, max=max_ctx)
        cache_shapes = {KEY_CACHE_NAME: {3: cache_dim}, VALUE_CACHE_NAME: {3: cache_dim}}
    else:
        # Static seq dim — StaticKVCache in Swift. Matches Apple vlm/export.py.
        # Ceiling is fixed at max_ctx at export time.
        cache_shapes = {KEY_CACHE_NAME: None, VALUE_CACHE_NAME: None}

    dynamic_shapes = {
        "inputs_embeds": {1: torch.export.Dim("query_len", max=max_ctx - 2)},
        "position_ids":  {1: torch.export.Dim("seq_pos", min=QUERY_LEN, max=max_ctx - 1)},
        **cache_shapes,
    }

    quant_desc  = f" ({quantize})" if quantize else " (fp16)"
    cache_desc  = f" [{kv_cache} KV]"
    print(f"[INFO] Tracing decode{quant_desc}{cache_desc} (inputs_embeds, stateful KV, slice_scatter)...")

    program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids"),
        output_names=("logits",),
        state_names=KV_STATE_NAMES,
    )
    program.optimize()
    meta = build_aimodel_metadata(
        f"FastVLM {variant.upper()} decoder (Qwen2){quant_desc}, inputs_embeds, stateful KV"
    )
    program.save_asset(model_path, meta)
    print(f"[INFO] Saved fastvlm-{variant}.aimodel")
    return f"fastvlm-{variant}.aimodel"


# ---------------------------------------------------------------------------
# Main export orchestration
# ---------------------------------------------------------------------------

async def export_fastvlm(
    variant: str,
    components: list[str],
    quantize: str | None,
    output_dir: Path,
    overwrite: bool,
    max_ctx: int = 4096,
    kv_cache: str = "static",
):
    weights_dir = _weights_dir(variant)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights not found: {weights_dir}")

    config, text_cfg = _load_config(weights_dir)
    bundle_path = _bundle_path(variant, output_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Exporting FastVLM {variant.upper()} → {bundle_path}")
    print(f"[INFO] Components: {components}")

    assets = {}
    image_token_id = 151646  # default

    if "vision" in components:
        _export_vision(weights_dir, config, bundle_path, overwrite)
        assets["vision"] = "vision.aimodel"

    if "embed" in components:
        _export_embed(weights_dir, text_cfg, bundle_path, max_ctx, overwrite)
        assets["embedding"] = "embed.aimodel"
        image_token_id = _setup_tokenizer(weights_dir, bundle_path)

    if "decode" in components:
        model_name = _export_decode(
            weights_dir, config, text_cfg, quantize, variant,
            bundle_path, max_ctx, overwrite, kv_cache=kv_cache
        )
        assets["main"] = model_name

    _write_bundle_metadata(bundle_path, variant, text_cfg, image_token_id, assets, max_ctx)
    print(f"\n[INFO] Bundle complete: {bundle_path}")
    return bundle_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export FastVLM to CoreAI VLM bundle format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--variant", choices=["0.5b", "1.5b", "7b"], required=True)
    parser.add_argument("--quantize", choices=["int8", "int4"], default=None)
    parser.add_argument(
        "--components", nargs="+", choices=ALL_COMPONENTS, default=ALL_COMPONENTS
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "exports",
    )
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument(
        "--kv-cache",
        choices=["static", "dynamic"],
        default="static",
        help=(
            "static: StaticKVCache, ceiling fixed at export time (default, matches Apple). "
            "dynamic: GrowingKVCache, Swift app controls ceiling, lower initial memory."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asyncio.run(export_fastvlm(
        variant=args.variant,
        components=args.components,
        quantize=args.quantize,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_ctx=args.max_context_length,
        kv_cache=args.kv_cache,
    ))


if __name__ == "__main__":
    main()
