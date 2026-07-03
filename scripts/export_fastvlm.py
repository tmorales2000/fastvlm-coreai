"""
export_fastvlm.py — Export FastVLM to CoreAI VLM bundle format.

Produces a bundle directory (e.g. fastvlm-1.5b.vlmasset/) containing:

  {variant}.aimodel   — text decoder (inputs_embeds, stateful KV cache)
  embed.aimodel        — token embedding lookup (input_ids → embeddings)
  vision.aimodel       — vision encoder + projector (pixel_values → image_features)
  tokenizer/           — Qwen2 tokenizer + <image> special token added
  metadata.json        — bundle manifest (kind=vlm, assets, language, vision blocks)

This matches the CoreAISequentialVLMEngine contract from:
  coreai-models/swift/Sources/CoreAILanguageModels/InferenceEngines/CoreAISequentialVLMEngine.swift
  coreai-models/python/src/coreai_models/vlm/export.py

INFERENCE FLOW (engine-side):
  1. encode_image: pixel_values → vision encoder → projector → image_features [1,256,hidden]
  2. embed_tokens (embed.aimodel): all_token_ids → embeddings [1,L+256,hidden]
  3. scatter_merge: replace <image> placeholder positions with image_features
  4. decode ({variant}.aimodel): merged_inputs_embeds → logits
  5. sample → next token → repeat from step 4

KV CACHE:
  Stateful (state_names=["k_cache","v_cache"]), slice_scatter updates (ANE-safe).
  Sequence dim declared dynamic (-1) → CoreAISequentialVLMEngine allocates
  GrowingKVCache starting at 256 tokens, growing 2× on demand.

IMAGE PLACEHOLDER TOKEN:
  <image> added as a special token to the Qwen2 tokenizer.
  Token ID: 151646 (next slot after the 3 existing added tokens).
  256 copies of <image> inserted at image position in the prompt.
  Engine scatter-merges projected image embeddings at those 256 positions.

VISION CONFIG (metadata.json):
  image_size: 1024        — FastVLM input resolution
  patch_size: 64          — effective patch stride (not ViT patch size)
  image_token_count: 256  — (1024/64)^2
  image_token_id: 151646  — <image> token ID
  image_mean: [0,0,0]     — no normalization (confirmed from llava_qwen.py)
  image_std:  [1,1,1]     — no normalization
  rescale_factor: 1.0

USAGE:
  python scripts/export_fastvlm.py --variant 0.5b
  python scripts/export_fastvlm.py --variant 1.5b --quantize int8
  python scripts/export_fastvlm.py --variant 7b   --quantize int4
  python scripts/export_fastvlm.py --variant 0.5b --overwrite

  # Export specific components only:
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
from coreai_torch import TorchConverter, get_decomp_table
from transformers import AutoConfig, AutoTokenizer

from fastvlm_decoder import FastVLMDecoder, FastVLMEmbedTokens
from fastvlm_projector import FastVLMProjector
from fastvlm_vision_encoder import FastVLMVisionEncoder
from quantization import QUANTIZATION_LEVELS, apply_quantization

# ─── Constants ────────────────────────────────────────────────────────────────

IMAGE_TOKEN = "<image>"
IMAGE_SIZE  = 1024
PATCH_SIZE  = 64   # effective stride in FastViTHD producing 16×16 = 256 patches
NUM_IMAGE_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256

# Image normalization (confirmed no-op from llava_qwen.py MobileCLIPVisionTower)
IMAGE_MEAN = [0.0, 0.0, 0.0]
IMAGE_STD  = [1.0, 1.0, 1.0]
RESCALE_FACTOR = 1.0

ALL_COMPONENTS = ["vision", "embed", "decode"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _weights_dir(variant: str) -> Path:
    return Path(__file__).parent.parent / "weights" / f"fastvlm-{variant}"


def _bundle_path(variant: str, output_dir: Path) -> Path:
    return output_dir / f"fastvlm-{variant}.vlmasset"


def _save_aimodel(program, path: Path, overwrite: bool, metadata: AIModelAssetMetadata):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite.")
        shutil.rmtree(path)
    program.save_asset(path, metadata)
    print(f"[INFO] Saved {path.name}")


def _build_asset_metadata(description: str) -> AIModelAssetMetadata:
    m = AIModelAssetMetadata()
    m.author = "Apple (FastVLM); re-authored by Agentive Intent LLC"
    m.license = "See FastVLM repository license"
    m.model_description = description
    m.creation_date = int(time.time())
    return m


def _load_config(weights_dir: Path):
    """Load HF config and return (full_config, text_config)."""
    cfg = AutoConfig.from_pretrained(str(weights_dir), trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None) or cfg
    return cfg, text_cfg


def _setup_tokenizer(weights_dir: Path, bundle_path: Path) -> int:
    """
    Load Qwen2 tokenizer, add <image> special token, save to bundle.
    Returns the image token ID (151646 for all variants).
    """
    tok = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)

    # Add <image> as a special token if not already present
    if IMAGE_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        print(f"[INFO] Added '{IMAGE_TOKEN}' token to tokenizer")

    image_token_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"[INFO] '{IMAGE_TOKEN}' token ID: {image_token_id}")

    tok_dir = bundle_path / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(tok_dir))
    print(f"[INFO] Saved tokenizer to {tok_dir}")

    return image_token_id


def _write_bundle_metadata(
    bundle_path: Path,
    variant: str,
    text_cfg,
    image_token_id: int,
    assets: dict,
):
    """Write metadata.json in VLM bundle format."""
    metadata = {
        "metadata_version": "0.2",
        "kind": "vlm",
        "name": f"fastvlm-{variant}",
        "assets": assets,
        "language": {
            "tokenizer": f"fastvlm-{variant}",
            "vocab_size": text_cfg.vocab_size,
            "max_context_length": 4096,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": {
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "image_token_count": NUM_IMAGE_TOKENS,
            "image_token_id": image_token_id,
            "image_mean": IMAGE_MEAN,
            "image_std": IMAGE_STD,
            "rescale_factor": RESCALE_FACTOR,
        },
        "source": {
            "hf_model_id": f"apple/FastVLM-{variant.upper()}",
            "model_definition": "torch",
        },
        # FastVLM-specific metadata for the Swift app
        "fastvlm": {
            "variant": variant,
            "hidden_size": text_cfg.hidden_size,
            "num_hidden_layers": text_cfg.num_hidden_layers,
            "num_attention_heads": text_cfg.num_attention_heads,
            "num_key_value_heads": text_cfg.num_key_value_heads,
            "mm_hidden_size": 3072,   # FastViTHD output dim, constant across variants
        },
    }
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote metadata.json (kind=vlm, image_token_id={image_token_id})")


# ─── Vision export ────────────────────────────────────────────────────────────

def _export_vision(
    weights_dir: Path,
    config,
    converter: TorchConverter,
):
    """Export vision encoder + projector as two entrypoints in vision.aimodel."""
    image_size = getattr(config, "image_size", IMAGE_SIZE)

    # Vision encoder
    vision_model = FastVLMVisionEncoder.from_weights(
        config=config, weights_dir=str(weights_dir)
    ).eval()
    example_vision = {"pixel_values": torch.randn(1, 3, image_size, image_size)}
    print("[INFO] Tracing encode_image...")
    exported = torch.export.export(vision_model, args=(), kwargs=example_vision)
    exported = exported.run_decompositions(get_decomp_table())
    converter.add_exported_program(
        exported_program=exported,
        input_names=["pixel_values"],
        output_names=["image_features"],
        entrypoint_name="encode_image",
    )

    # Projector
    _, text_cfg = _load_config(weights_dir)
    proj_model = FastVLMProjector.from_weights(
        config=config, weights_dir=str(weights_dir)
    ).eval()
    n_patches = NUM_IMAGE_TOKENS
    mm_hidden = getattr(config, "mm_hidden_size", 3072)
    example_proj = {"x": torch.randn(1, n_patches, mm_hidden, dtype=torch.float16)}
    print("[INFO] Tracing project...")
    exported = torch.export.export(proj_model, args=(), kwargs=example_proj)
    exported = exported.run_decompositions(get_decomp_table())
    converter.add_exported_program(
        exported_program=exported,
        input_names=["x"],
        output_names=["projected_features"],
        entrypoint_name="project",
    )


# ─── Embed export ─────────────────────────────────────────────────────────────

def _export_embed(
    weights_dir: Path,
    text_cfg,
    converter: TorchConverter,
):
    """Export embed_tokens as a standalone entrypoint in embed.aimodel."""
    embed_model = FastVLMEmbedTokens.from_weights(str(weights_dir)).eval()

    seq_len = 64
    example = {"input_ids": torch.zeros(1, seq_len, dtype=torch.int32)}
    seq_dim = torch.export.Dim("embed_seq", min=1, max=text_cfg.vocab_size - 1)
    dynamic_shapes = {"input_ids": {1: seq_dim}}

    print("[INFO] Tracing embed_tokens...")
    exported = torch.export.export(
        embed_model, args=(), kwargs=example, dynamic_shapes=dynamic_shapes
    )
    exported = exported.run_decompositions(get_decomp_table())
    converter.add_exported_program(
        exported_program=exported,
        input_names=["input_ids"],
        output_names=["embeddings"],
        entrypoint_name="embed_tokens",
    )


# ─── Decode export ────────────────────────────────────────────────────────────

def _export_decode(
    weights_dir: Path,
    config,
    text_cfg,
    quantize: str | None,
    converter: TorchConverter,
):
    """
    Export the Qwen2 decoder as main entrypoint in {variant}.aimodel.

    Takes inputs_embeds (pre-merged text + image embeddings) not input_ids.
    Stateful KV cache with slice_scatter, dynamic sequence dimension.
    """
    model = FastVLMDecoder.from_weights(text_cfg, str(weights_dir)).eval()

    if quantize:
        print(f"[INFO] Applying {quantize} quantization...")
        model = apply_quantization(model, level=QUANTIZATION_LEVELS[quantize])

    hidden = text_cfg.hidden_size
    n_layers = text_cfg.num_hidden_layers
    n_kv_heads = text_cfg.num_key_value_heads
    head_dim = hidden // text_cfg.num_attention_heads
    max_ctx = 4096  # KV cache capacity

    # Update model's cache buffers to the actual max_ctx size
    model.k_cache = torch.zeros(
        n_layers, 1, n_kv_heads, max_ctx, head_dim, dtype=torch.float16
    )
    model.v_cache = torch.zeros_like(model.k_cache)

    # Example inputs
    query_len = 64
    offset = 64
    example = {
        "inputs_embeds": torch.randn(1, query_len, hidden, dtype=torch.float16),
        "position_ids":  torch.arange(query_len + offset, dtype=torch.int32).unsqueeze(0),
    }

    # Dynamic shapes:
    #   query_len — varies: 256+N (prefill) vs 1 (decode step)
    #   seq_pos   — grows with each decode step
    #   cache seq dim (axis 3) — declared dynamic → GrowingKVCache
    query_dim = torch.export.Dim("query_len", max=max_ctx - 2)  # tracer requires max=4094 due to slice_scatter guard
    seq_dim   = torch.export.Dim("seq_pos",   min=query_len, max=max_ctx - 1)
    cache_dim = torch.export.Dim("cache_seq", min=0, max=max_ctx)

    # dynamic_shapes only accepts forward() arg names, not buffer names.
    # The KV cache buffers (k_cache, v_cache) are registered buffers —
    # their dynamic seq dim is handled by coreai-torch via state_names.
    # We set the buffer to max_ctx size before export so the compiled
    # state descriptor gets the right shape; coreai-torch then declares
    # the seq dim as dynamic when state_names is specified.
    dynamic_shapes = {
        "inputs_embeds": {1: query_dim},
        "position_ids":  {1: seq_dim},
    }

    print("[INFO] Tracing decode (inputs_embeds, stateful KV, slice_scatter)...")
    exported = torch.export.export(
        model, args=(), kwargs=example, dynamic_shapes=dynamic_shapes
    )
    exported = exported.run_decompositions(get_decomp_table())

    converter.add_exported_program(
        exported_program=exported,
        input_names=["inputs_embeds", "position_ids"],
        output_names=["logits"],
        state_names=["k_cache", "v_cache"],
        entrypoint_name="main",
    )


# ─── Main export orchestration ────────────────────────────────────────────────

async def export_fastvlm(
    variant: str,
    components: list[str],
    quantize: str | None,
    output_dir: Path,
    overwrite: bool,
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
    image_token_id = 151646  # default — will be set by tokenizer if run

    # --- Vision bundle ---
    if "vision" in components:
        converter = TorchConverter()
        _export_vision(weights_dir, config, converter)
        program = converter.to_coreai()
        program.optimize()
        vision_path = bundle_path / "vision.aimodel"
        meta = _build_asset_metadata(
            "FastVLM vision encoder (FastViTHD) + multimodal projector (mlp2x_gelu)"
        )
        _save_aimodel(program, vision_path, overwrite, meta)
        assets["vision"] = "vision.aimodel"

    # --- Embed bundle ---
    if "embed" in components:
        converter = TorchConverter()
        _export_embed(weights_dir, text_cfg, converter)
        program = converter.to_coreai()
        program.optimize()
        embed_path = bundle_path / "embed.aimodel"
        meta = _build_asset_metadata("FastVLM token embedding lookup (embed_tokens)")
        _save_aimodel(program, embed_path, overwrite, meta)
        assets["embedding"] = "embed.aimodel"

        # Also set up tokenizer when exporting embed
        image_token_id = _setup_tokenizer(weights_dir, bundle_path)

    # --- Decode bundle ---
    if "decode" in components:
        converter = TorchConverter()
        _export_decode(weights_dir, config, text_cfg, quantize, converter)
        program = converter.to_coreai()
        program.optimize()
        model_name = f"fastvlm-{variant}.aimodel"
        model_path = bundle_path / model_name
        quant_desc = f" ({quantize} quantized)" if quantize else " (fp16)"
        meta = _build_asset_metadata(
            f"FastVLM {variant.upper()} language decoder (Qwen2){quant_desc}, "
            "inputs_embeds input, stateful KV cache"
        )
        _save_aimodel(program, model_path, overwrite, meta)
        assets["main"] = model_name

    # --- metadata.json ---
    _write_bundle_metadata(bundle_path, variant, text_cfg, image_token_id, assets)

    print(f"\n[INFO] Bundle complete: {bundle_path}")
    print(f"[INFO] Assets: {list(assets.keys())}")
    return bundle_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export FastVLM to CoreAI VLM bundle format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--variant", choices=["0.5b", "1.5b", "7b"], required=True)
    parser.add_argument(
        "--quantize",
        choices=["int8", "int4"],
        default=None,
        help="Quantize the decoder weights (int8 for 1.5b, int4 for 7b)"
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=ALL_COMPONENTS,
        default=ALL_COMPONENTS,
        help=f"Components to export (default: all). Options: {ALL_COMPONENTS}"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "exports",
        help="Output directory for the bundle (default: exports/)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asyncio.run(export_fastvlm(
        variant=args.variant,
        components=args.components,
        quantize=args.quantize,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    ))


if __name__ == "__main__":
    main()
