"""
export_fastvlm.py — Export FastVLM to CoreAI VLM bundle format.

Follows Apple's authoritative recipe from:
  coreai-models/python/src/coreai_models/vlm/export.py
  coreai-models/python/src/coreai_models/export/pipeline.py
  coreai-models/python/src/coreai_models/export/presets.py

Produces a bundle directory (e.g. exports/fastvlm-0.5b/) containing:
  {variant}.aimodel — text decoder (asset role: main)
  embed.aimodel      — token embedding lookup (asset role: embedding)
  vision.aimodel     — vision encoder + projector (asset role: vision)
  tokenizer/         — Qwen2 tokenizer + <image> special token
  metadata.json      — bundle manifest (kind=vlm)

COMPRESSION
===========
Compression is specified via one of two mutually exclusive options:

  --compression PRESET
      Named preset. Available:
        4bit  — int4 symmetric_with_clipping per_block_32. Apple's canonical
                macOS preset. Default when --quantize int4 is used.
        8bit  — int8 per_channel symmetric. Best for lower memory vs fp16
                with identical throughput on GPU path.
        none  — fp16 only (default when --compression is omitted).

  --compression-config path/to/recipe.yaml
      YAML file with a quantization_config block. Accepts the same format
      as QuantizerConfig.from_dict(). Use for mixed-precision per-model
      recipes produced by scan_quantization_sensitivity.py. Mutually
      exclusive with --compression.

  --platform macOS|iOS
      Target platform (default: macOS). Controls compression defaults:
        macOS: linear quantization (torch pre-export, int4/int8)
        iOS:   palettization (k-means codebook, ANE BC1S layout)
      NOTE: iOS export is not yet implemented. macOS only.

YAML RECIPE FORMAT
==================
Recipes produced by scan_quantization_sensitivity.py or written by hand:

  quantization_config:
    execution_mode: eager
    global_config:
      op_state_spec:
        weight:
          dtype: int4
          qscheme: symmetric_with_clipping
          granularity:
            type: per_block
            block_size: 32
            axis: 1
      op_input_spec: null
      op_output_spec: null
    module_name_configs:
      # Keep sensitive layers at higher precision:
      'model\\.layers\\.(10|11|22)\\.(self_attn|mlp)\\..*':
        op_state_spec:
          weight:
            dtype: int8
            qscheme: symmetric_with_clipping
            granularity: {type: per_channel}
        op_input_spec: null
        op_output_spec: null

  Optional calibration trigger:
  coreai_models:
    calibrate_activations: true

USAGE
=====
  # fp16 (no quantization)
  python scripts/export_fastvlm.py --variant 0.5b

  # Named preset
  python scripts/export_fastvlm.py --variant 1.5b --compression 8bit
  python scripts/export_fastvlm.py --variant 7b   --compression 4bit

  # YAML recipe (mixed precision, from scan_quantization_sensitivity.py)
  python scripts/export_fastvlm.py --variant 7b \\
      --compression-config recipes/fastvlm_7b_mixed.yaml

  # KV cache mode
  python scripts/export_fastvlm.py --variant 0.5b --kv-cache static   # default
  python scripts/export_fastvlm.py --variant 0.5b --kv-cache dynamic  # GrowingKVCache

  # Selective components
  python scripts/export_fastvlm.py --variant 0.5b --components vision embed decode
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml
from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_opt.quantization import QuantizerConfig
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
from quantization import (
    MACOS_NAMED_PRESETS,
    load_compression_config,
    apply_quantization_from_config,
    finalize_for_export,
)

# ---------------------------------------------------------------------------
# Constants (matching vlm/export.py)
# ---------------------------------------------------------------------------
QUERY_LEN = 64   # trace-time query length
OFFSET    = 64   # trace-time position offset

IMAGE_TOKEN       = "<image>"
IMAGE_SIZE        = 1024
PATCH_SIZE        = 64
NUM_IMAGE_TOKENS  = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 256
IMAGE_MEAN        = [0.0, 0.0, 0.0]
IMAGE_STD         = [1.0, 1.0, 1.0]
RESCALE_FACTOR    = 1.0

ALL_COMPONENTS = ["vision", "embed", "decode"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights_dir(variant: str) -> Path:
    return Path(__file__).parent.parent / "weights" / f"fastvlm-{variant}"


def _bundle_path(variant: str, output_dir: Path) -> Path:
    return output_dir / f"fastvlm-{variant}"


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
    compression: str,
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
            "image_strategy":    "center_crop",
        },
        "source": {
            "hf_model_id":      f"apple/FastVLM-{variant.upper()}",
            "model_definition": "torch",
            "compression":      compression,
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
    print(f"[INFO] Wrote metadata.json  (compression: {compression})")


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

    print("[INFO] Tracing encode_image...")
    vision_model = FastVLMVisionEncoder.from_weights(
        config=config, weights_dir=str(weights_dir)
    ).eval()

    print("[INFO] Tracing project...")
    proj_model = FastVLMProjector.from_weights(
        config=config, weights_dir=str(weights_dir)
    ).eval()

    from coreai_models.export.macos import _EXTERNALIZE_SPECS
    from coreai_models.export.mlir_ops import (
        remove_functionalization,
        register_custom_torch_lowering,
    )
    import coreai_torch as ct

    def _export_fn(module, **kwargs):
        with torch.no_grad():
            ep = torch.export.export(module, args=(), kwargs=kwargs)
        ep = ep.run_decompositions(ct.get_decomp_table())
        remove_functionalization(ep)
        return ep

    converter = ct.TorchConverter()
    converter.add_pytorch_module(
        vision_model,
        export_fn=lambda m: _export_fn(
            m, pixel_values=torch.randn(1, 3, image_size, image_size)
        ),
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=("pixel_values",),
        output_names=("image_features",),
        entrypoint_name="encode_image",
    )
    converter.add_pytorch_module(
        proj_model,
        export_fn=lambda m: _export_fn(
            m, x=torch.randn(1, NUM_IMAGE_TOKENS, mm_hidden, dtype=torch.float16)
        ),
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=("x",),
        output_names=("projected_features",),
        entrypoint_name="project",
    )
    register_custom_torch_lowering(converter)
    program = converter.to_coreai()
    program.optimize()
    meta = build_aimodel_metadata("FastVLM vision encoder + projector")
    program.save_asset(vision_path, meta)
    print("[INFO] Saved vision.aimodel")


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
    print("[INFO] Saved embed.aimodel")


# ---------------------------------------------------------------------------
# Decode export
# ---------------------------------------------------------------------------

def _export_decode(
    weights_dir: Path,
    config,
    text_cfg,
    compression_config: dict | None,
    compression_label: str,
    variant: str,
    bundle_path: Path,
    max_ctx: int,
    overwrite: bool,
    kv_cache: str = "static",
):
    """Export the Qwen2 decoder.

    compression_config: resolved QuantizerConfig dict (from named preset or
        YAML), or None for fp16. Passed to apply_quantization_from_config().
    compression_label: human-readable string for logging/metadata ("4bit",
        "8bit", "none", or YAML stem).
    """
    for stale in bundle_path.glob("fastvlm-*.aimodel"):
        shutil.rmtree(stale)

    model_path = bundle_path / f"fastvlm-{variant}.aimodel"
    if model_path.exists():
        shutil.rmtree(model_path)

    model = FastVLMDecoder.from_weights(text_cfg, str(weights_dir)).eval()

    hidden     = text_cfg.hidden_size
    n_layers   = text_cfg.num_hidden_layers
    n_kv_heads = text_cfg.num_key_value_heads
    head_dim   = hidden // text_cfg.num_attention_heads

    if compression_config is not None:
        print(f"[INFO] Applying compression: {compression_label}")
        ex_k = torch.zeros(n_layers, 1, n_kv_heads, max_ctx, head_dim, dtype=torch.float16)
        ex_v = torch.zeros_like(ex_k)
        example_inputs = (
            torch.randn(1, QUERY_LEN, hidden, dtype=torch.float16),
            torch.arange(QUERY_LEN + OFFSET, dtype=torch.int32).unsqueeze(0),
            ex_k,
            ex_v,
        )
        model = apply_quantization_from_config(
            model, compression_config, example_inputs
        )
        print(f"[INFO] Compression finalized for CoreAI export")

    k_cache = torch.zeros(n_layers, 1, n_kv_heads, max_ctx, head_dim, dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)

    reference_inputs = {
        "inputs_embeds": torch.randn(1, QUERY_LEN, hidden, dtype=torch.float16),
        "position_ids":  torch.arange(QUERY_LEN + OFFSET, dtype=torch.int32).unsqueeze(0),
        KEY_CACHE_NAME:  k_cache,
        VALUE_CACHE_NAME: v_cache,
    }

    if kv_cache == "dynamic":
        TRACE_CACHE_SEQ = 256
        k_cache = torch.zeros(
            n_layers, 1, n_kv_heads, TRACE_CACHE_SEQ, head_dim, dtype=torch.float16
        )
        v_cache = torch.zeros_like(k_cache)
        reference_inputs[KEY_CACHE_NAME]   = k_cache
        reference_inputs[VALUE_CACHE_NAME] = v_cache
        cache_dim    = torch.export.Dim("cache_seq", min=0, max=max_ctx)
        cache_shapes = {KEY_CACHE_NAME: {3: cache_dim}, VALUE_CACHE_NAME: {3: cache_dim}}
    else:
        cache_shapes = {KEY_CACHE_NAME: None, VALUE_CACHE_NAME: None}

    dynamic_shapes = {
        "inputs_embeds": {1: torch.export.Dim("query_len", max=max_ctx - 2)},
        "position_ids":  {1: torch.export.Dim("seq_pos", min=QUERY_LEN, max=max_ctx - 1)},
        **cache_shapes,
    }

    comp_desc  = f" ({compression_label})" if compression_label != "none" else " (fp16)"
    cache_desc = f" [{kv_cache} KV]"
    print(f"[INFO] Tracing decoder{comp_desc}{cache_desc}...")

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
        f"FastVLM {variant.upper()} decoder (Qwen2){comp_desc}, inputs_embeds, stateful KV"
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
    compression_config: dict | None,
    compression_label: str,
    output_dir: Path,
    overwrite: bool,
    max_ctx: int = 4096,
    kv_cache: str = "static",
):
    weights_dir = _weights_dir(variant)
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_dir}\n"
            f"Download with: hf download apple/FastVLM-{variant.upper()} "
            f"--local-dir {weights_dir}"
        )

    config, text_cfg = _load_config(weights_dir)
    bundle_path = _bundle_path(variant, output_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    max_pos = getattr(text_cfg, "max_position_embeddings", None)
    if max_pos is not None and max_ctx > max_pos:
        raise ValueError(
            f"--max-context-length {max_ctx} exceeds model's "
            f"max_position_embeddings={max_pos}. Use <= {max_pos}."
        )
    if max_pos is not None:
        head_dim   = text_cfg.hidden_size // text_cfg.num_attention_heads
        kv_mem_mb  = (
            text_cfg.num_hidden_layers * text_cfg.num_key_value_heads
            * head_dim * max_ctx * 2 * 2 / 1_048_576
        )
        alloc_desc = "pre-allocated" if kv_cache == "static" else "ceiling (dynamic)"
        print(f"[INFO] KV cache at max_ctx={max_ctx}: {kv_mem_mb:.1f} MB ({alloc_desc})")

    print(f"\n[INFO] Exporting FastVLM {variant.upper()} → {bundle_path}")
    print(f"[INFO] Components:   {components}")
    print(f"[INFO] Compression:  {compression_label}")
    print(f"[INFO] KV cache:     {kv_cache}")

    assets        = {}
    image_token_id = 151646  # default, overwritten by embed step

    if "vision" in components:
        _export_vision(weights_dir, config, bundle_path, overwrite)
        assets["vision"] = "vision.aimodel"

    if "embed" in components:
        _export_embed(weights_dir, text_cfg, bundle_path, max_ctx, overwrite)
        assets["embedding"]  = "embed.aimodel"
        image_token_id = _setup_tokenizer(weights_dir, bundle_path)

    if "decode" in components:
        model_name = _export_decode(
            weights_dir, config, text_cfg,
            compression_config, compression_label,
            variant, bundle_path, max_ctx, overwrite,
            kv_cache=kv_cache,
        )
        assets["main"] = model_name

    _write_bundle_metadata(
        bundle_path, variant, text_cfg, image_token_id,
        assets, max_ctx, compression_label,
    )
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
    parser.add_argument(
        "--variant", choices=["0.5b", "1.5b", "7b"], required=True,
        help="Model variant to export.",
    )
    parser.add_argument(
        "--platform", choices=["macOS", "iOS"], default="macOS",
        help="Target platform (default: macOS). iOS export not yet implemented.",
    )
    parser.add_argument(
        "--components", nargs="+", choices=ALL_COMPONENTS, default=ALL_COMPONENTS,
        help="Bundle components to export (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "exports",
        help="Output directory (default: exports/).",
    )
    parser.add_argument(
        "--max-context-length", type=int, default=4096,
        help="KV cache ceiling in tokens (default: 4096).",
    )
    parser.add_argument(
        "--kv-cache", choices=["static", "dynamic"], default="static",
        help=(
            "KV cache strategy. static (default): pre-allocates max_ctx tokens. "
            "dynamic: starts small, grows up to max_ctx. StaticKVCache vs "
            "GrowingKVCache in Swift."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    # Compression — mutually exclusive, matching Apple's CLI pattern
    compression_group = parser.add_mutually_exclusive_group()
    compression_group.add_argument(
        "--compression",
        choices=list(MACOS_NAMED_PRESETS.keys()) + ["none"],
        default=None,
        metavar="PRESET",
        help=(
            f"Named compression preset. Available: "
            f"{', '.join(MACOS_NAMED_PRESETS.keys())}, none. "
            f"Default: none (fp16). "
            f"4bit = int4 symmetric_with_clipping per_block_32 (Apple macOS standard). "
            f"8bit = int8 per_channel symmetric."
        ),
    )
    compression_group.add_argument(
        "--compression-config",
        type=Path,
        default=None,
        metavar="YAML",
        help=(
            "Path to a coreai-opt YAML quantization recipe. "
            "Top-level key must be 'quantization_config'. "
            "Use for mixed-precision recipes from scan_quantization_sensitivity.py. "
            "Mutually exclusive with --compression."
        ),
    )

    args = parser.parse_args()

    if args.platform == "iOS":
        parser.error("iOS export is not yet implemented.")

    # Resolve compression config and label
    compression_config: dict | None = None
    compression_label: str = "none"

    if args.compression_config is not None:
        if not args.compression_config.is_file():
            parser.error(f"--compression-config: file not found: {args.compression_config}")
        compression_config, compression_label = load_compression_config(
            args.compression_config, platform=args.platform
        )
        print(f"[INFO] Loaded compression config: {args.compression_config}")

    elif args.compression is not None and args.compression != "none":
        compression_config, compression_label = load_compression_config(
            args.compression, platform=args.platform
        )

    asyncio.run(export_fastvlm(
        variant=args.variant,
        components=args.components,
        compression_config=compression_config,
        compression_label=compression_label,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_ctx=args.max_context_length,
        kv_cache=args.kv_cache,
    ))


if __name__ == "__main__":
    main()
