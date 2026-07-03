"""
export_fastvlm.py — Export FastVLM to CoreAI VLM bundle format.

Follows Apple's vlm/export.py recipe from coreai-models exactly.

Produces a bundle directory (e.g. fastvlm-0.5b.vlmasset/) containing:
  {variant}.aimodel — text decoder (inputs_embeds, stateful KV, keyCache/valueCache)
  embed.aimodel      — token embedding lookup (input_ids → embeddings)
  vision.aimodel     — vision encoder + projector (pixel_values → image_features)
  tokenizer/         — Qwen2 tokenizer + <image> special token (ID 151646)
  metadata.json      — bundle manifest (kind=vlm, assets, language, vision blocks)

KEY DIFFERENCES FROM EARLIER APPROACH
======================================
- Uses add_pytorch_module (not add_exported_program) — matches Apple's recipe
- Uses remove_functionalization from coreai_models.export.mlir_ops — fixes
  auto_functionalized_v2 issue with slice_scatter
- k_cache/v_cache are explicit forward() args with dynamic_shapes, not buffers
- State names: "keyCache" / "valueCache" (camelCase, Swift runner convention)
- TRACE_KV_CACHE_SEQ_LEN = 2048 (trace-time only), max_ctx set separately
- dynamic_shapes: k_cache/v_cache dim 3 is dynamic, inputs/position_ids dynamic

USAGE:
  python scripts/export_fastvlm.py --variant 0.5b
  python scripts/export_fastvlm.py --variant 1.5b --quantize int8
  python scripts/export_fastvlm.py --variant 7b   --quantize int4
  python scripts/export_fastvlm.py --variant 0.5b --components vision embed decode
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
from coreai.authoring.asset import AIModelAssetMetadata
from coreai_torch import TorchConverter
from coreai_torch import get_decomp_table
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
# Try to import Apple's remove_functionalization. Fall back gracefully.
# ---------------------------------------------------------------------------
try:
    from coreai_models.export.mlir_ops import (
        remove_functionalization,
        register_custom_torch_lowering,
    )
    _HAS_COREAI_MODELS = True
except ImportError:
    _HAS_COREAI_MODELS = False
    def remove_functionalization(ep): pass
    def register_custom_torch_lowering(converter): pass

try:
    import coreai_torch
    _EXTERNALIZE_SPECS = [
        coreai_torch.ExternalizeSpec(
            target_class=coreai_torch.composite_ops.RMSNormImpl,
            composite_op_name="rms_norm",
            composite_attrs=["axes", "eps"],
        ),
        coreai_torch.ExternalizeSpec(
            target_class=coreai_torch.composite_ops.RoPE,
            composite_op_name="rope",
            composite_attrs=["scale", "base", "dims", "interleaved"],
        ),
        coreai_torch.ExternalizeSpec(
            target_class=coreai_torch.composite_ops.SDPA,
            composite_op_name="scaled_dot_product_attention",
            composite_attrs=["scale", "is_causal", "window_size"],
        ),
    ]
except Exception:
    _EXTERNALIZE_SPECS = []

# ---------------------------------------------------------------------------
# Constants (matching coreai-models export/_constants.py)
# ---------------------------------------------------------------------------
TRACE_KV_CACHE_SEQ_LEN = 2048   # trace-time cache size (not max context)
QUANT_TRACE_QUERY_LEN  = 64     # trace-time query length
QUANT_TRACE_OFFSET     = 64     # trace-time position offset

IMAGE_TOKEN        = "<image>"
IMAGE_SIZE         = 1024
PATCH_SIZE         = 64
NUM_IMAGE_TOKENS   = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256
IMAGE_MEAN         = [0.0, 0.0, 0.0]   # confirmed no-op from llava_qwen.py
IMAGE_STD          = [1.0, 1.0, 1.0]
RESCALE_FACTOR     = 1.0

ALL_COMPONENTS = ["vision", "embed", "decode"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights_dir(variant: str) -> Path:
    return Path(__file__).parent.parent / "weights" / f"fastvlm-{variant}"


def _bundle_path(variant: str, output_dir: Path) -> Path:
    return output_dir / f"fastvlm-{variant}.vlmasset"


def _build_asset_metadata(description: str) -> AIModelAssetMetadata:
    m = AIModelAssetMetadata()
    m.author = "Apple (FastVLM); re-authored by Agentive Intent LLC"
    m.license = "See FastVLM repository license"
    m.model_description = description
    m.creation_date = int(time.time())
    return m


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
            "hf_model_id":    f"apple/FastVLM-{variant.upper()}",
            "model_definition": "torch",
        },
        "fastvlm": {
            "variant":             variant,
            "hidden_size":         text_cfg.hidden_size,
            "num_hidden_layers":   text_cfg.num_hidden_layers,
            "num_attention_heads": text_cfg.num_attention_heads,
            "num_key_value_heads": text_cfg.num_key_value_heads,
            "vocab_size":          text_cfg.vocab_size,
            "mm_hidden_size":      3072,
        },
    }
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote metadata.json")


# ---------------------------------------------------------------------------
# Core export function (mirrors export_to_coreai from Apple's recipe)
# ---------------------------------------------------------------------------

def _export_module(
    model: torch.nn.Module,
    reference_inputs: dict,
    dynamic_shapes: dict | None,
    input_names: tuple,
    output_names: tuple,
    state_names: tuple | None = None,
    entrypoint_name: str = "main",
    overwrite: bool = False,
    output_path: Path | None = None,
    meta: AIModelAssetMetadata | None = None,
) -> object:
    """Export a module to Core AI AIProgram following Apple's recipe."""

    def export_fn(module: torch.nn.Module):
        with torch.no_grad():
            ep = torch.export.export(
                module,
                args=(),
                kwargs=reference_inputs,
                dynamic_shapes=dynamic_shapes,
            )
        decomp_table = coreai_torch.get_decomp_table()
        ep = ep.run_decompositions(decomp_table)
        remove_functionalization(ep)
        return ep

    model.eval()
    converter = TorchConverter()
    converter.add_pytorch_module(
        model,
        export_fn=export_fn,
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=input_names,
        output_names=output_names,
        state_names=state_names,
        entrypoint_name=entrypoint_name,
    )
    register_custom_torch_lowering(converter)
    program = converter.to_coreai()
    program.optimize()

    if output_path is not None:
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(f"{output_path} exists. Use --overwrite.")
            shutil.rmtree(output_path)
        program.save_asset(output_path, meta or _build_asset_metadata("FastVLM component"))
        print(f"[INFO] Saved {output_path.name}")

    return program


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

    # Vision encoder (encode_image entrypoint)
    vision_model = FastVLMVisionEncoder.from_weights(
        config=config, weights_dir=str(weights_dir)
    ).eval()
    ref_vision = {"pixel_values": torch.randn(1, 3, image_size, image_size)}

    vision_path = bundle_path / "vision.aimodel"
    if vision_path.exists() and not overwrite:
        raise FileExistsError(f"{vision_path} exists.")
    if vision_path.exists():
        shutil.rmtree(vision_path)

    # Export vision encoder first, then add projector as second entrypoint
    print("[INFO] Tracing encode_image...")
    converter = TorchConverter()

    def export_fn_vision(module):
        with torch.no_grad():
            ep = torch.export.export(module, args=(), kwargs=ref_vision)
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        return ep

    converter.add_pytorch_module(
        vision_model,
        export_fn=export_fn_vision,
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=("pixel_values",),
        output_names=("image_features",),
        entrypoint_name="encode_image",
    )

    # Projector (project entrypoint)
    _, text_cfg = _load_config(weights_dir)
    proj_model = FastVLMProjector.from_weights(
        config=config, weights_dir=str(weights_dir)
    ).eval()
    mm_hidden = getattr(config, "mm_hidden_size", 3072)
    ref_proj = {"x": torch.randn(1, NUM_IMAGE_TOKENS, mm_hidden, dtype=torch.float16)}

    print("[INFO] Tracing project...")

    def export_fn_proj(module):
        with torch.no_grad():
            ep = torch.export.export(module, args=(), kwargs=ref_proj)
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        return ep

    converter.add_pytorch_module(
        proj_model,
        export_fn=export_fn_proj,
        externalize_modules=_EXTERNALIZE_SPECS,
        input_names=("x",),
        output_names=("projected_features",),
        entrypoint_name="project",
    )

    program = converter.to_coreai()
    program.optimize()
    meta = _build_asset_metadata(
        "FastVLM vision encoder (FastViTHD) + multimodal projector (mlp2x_gelu)"
    )
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
    embed_model = FastVLMEmbedTokens.from_weights(str(weights_dir)).eval()
    ref = {"input_ids": torch.zeros(1, QUANT_TRACE_QUERY_LEN, dtype=torch.int32)}
    dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("embed_seq", max=max_ctx - 1)}
    }

    print("[INFO] Tracing embed_tokens...")
    embed_path = bundle_path / "embed.aimodel"
    if embed_path.exists() and not overwrite:
        raise FileExistsError(f"{embed_path} exists.")
    if embed_path.exists():
        shutil.rmtree(embed_path)

    _export_module(
        model=embed_model,
        reference_inputs=ref,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids",),
        output_names=("embeddings",),
        state_names=None,
        entrypoint_name="embed_tokens",
        output_path=embed_path,
        meta=_build_asset_metadata("FastVLM token embedding lookup (embed_tokens)"),
    )


# ---------------------------------------------------------------------------
# Decode export
# ---------------------------------------------------------------------------

def _export_decode(
    weights_dir: Path,
    config,
    text_cfg,
    quantize: str | None,
    bundle_path: Path,
    max_ctx: int,
    overwrite: bool,
    variant: str = "decoder",
):
    """
    Export the Qwen2 decoder following Apple's Qwen3-VL recipe exactly:
    - inputs_embeds + position_ids + keyCache + valueCache as forward() args
    - keyCache/valueCache with dynamic seq_len_dim (axis 3)
    - state_names = ("keyCache", "valueCache")
    - remove_functionalization called on exported program
    """
    model = FastVLMDecoder.from_weights(text_cfg, str(weights_dir)).eval()

    if quantize:
        print(f"[INFO] Applying {quantize} quantization...")
        model = apply_quantization(model, level=QUANTIZATION_LEVELS[quantize])

    hidden    = text_cfg.hidden_size
    n_layers  = text_cfg.num_hidden_layers
    n_kv_heads = text_cfg.num_key_value_heads
    head_dim  = hidden // text_cfg.num_attention_heads

    # Build trace-time cache (TRACE_KV_CACHE_SEQ_LEN, not max_ctx)
    k_cache = torch.zeros(n_layers, 1, n_kv_heads, TRACE_KV_CACHE_SEQ_LEN, head_dim, dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)

    reference_inputs = {
        "inputs_embeds": torch.randn(1, QUANT_TRACE_QUERY_LEN, hidden, dtype=torch.float16),
        "position_ids":  torch.arange(
            QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32
        ).unsqueeze(0),
        KEY_CACHE_NAME:   k_cache,
        VALUE_CACHE_NAME: v_cache,
    }

    dynamic_shapes = {
        "inputs_embeds": {1: torch.export.Dim("query_len", max=max_ctx - 2)},
        "position_ids":  {1: torch.export.Dim(
            "seq_pos", min=QUANT_TRACE_QUERY_LEN, max=max_ctx - 1
        )},
        KEY_CACHE_NAME:  {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx
            )
        },
        VALUE_CACHE_NAME: {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx
            )
        },
    }

    quant_desc = f" ({quantize} quantized)" if quantize else " (fp16)"
    model_name = f"fastvlm-{variant}.aimodel"
    model_path = bundle_path / model_name

    print("[INFO] Tracing decode (inputs_embeds, stateful KV, slice_scatter)...")

    # Remove any stale decoder aimodel files from previous exports
    for stale in bundle_path.glob("fastvlm-*.aimodel"):
        shutil.rmtree(stale)

    if model_path.exists():
        shutil.rmtree(model_path)

    _export_module(
        model=model,
        reference_inputs=reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids"),
        output_names=("logits",),
        state_names=KV_STATE_NAMES,
        entrypoint_name="main",
        output_path=model_path,
        meta=_build_asset_metadata(
            f"FastVLM language decoder (Qwen2){quant_desc}, "
            "inputs_embeds, stateful KV (keyCache/valueCache)"
        ),
    )
    return model_name


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
):
    weights_dir = _weights_dir(variant)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights not found: {weights_dir}")

    config, text_cfg = _load_config(weights_dir)
    bundle_path = _bundle_path(variant, output_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Exporting FastVLM {variant.upper()} → {bundle_path}")
    print(f"[INFO] Components: {components}")
    if not _HAS_COREAI_MODELS:
        print("[WARN] coreai_models not found — remove_functionalization skipped. "
              "Install from coreai-models repo for best results.")

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
            weights_dir, config, text_cfg, quantize, bundle_path, max_ctx, overwrite,
            variant=variant
        )
        assets["main"] = f"fastvlm-{variant}.aimodel"

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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asyncio.run(export_fastvlm(
        variant=args.variant,
        components=args.components,
        quantize=args.quantize,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_ctx=args.max_context_length,
    ))


if __name__ == "__main__":
    main()
