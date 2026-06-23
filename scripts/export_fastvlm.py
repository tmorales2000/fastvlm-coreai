"""
export_fastvlm.py — Export all three FastVLM components to a single Core AI
.aimodel with three callable entry points: vision_encode, project, decode.

WHY THREE ENTRYPOINTS IN ONE .aimodel
----------------------------------------
FastVLM's three components have different call frequencies per inference:
vision_encode and project each run ONCE per image; decode runs ONCE PER
TOKEN. No VLM example exists in coreai-models to copy this pattern from
directly — every example there is either single-component (Qwen2, Gemma3)
or a single-forward()-call multi-modal model (SAM3 takes image+text in one
forward, Whisper takes audio features + decoder_input_ids in one forward).
FastVLM's call-frequency mismatch doesn't fit a single forward() cleanly.

TorchConverter.add_pytorch_module/add_exported_program's entrypoint_name
parameter is the documented, real mechanism for staging multiple programs
into one converter and producing one .aimodel with multiple callable
functions (confirmed by reading coreai_torch/converter.py directly — it
raises ValueError on a duplicate entrypoint_name, and to_coreai() converts
all staged entries into one MLIR module/AIProgram in a single call). This
is the mechanism this script uses.

PRECISION AND QUANTIZATION
---------------------------
All three components are exported at float16 — the mandatory ANE execution
precision, matching Apple's MLX pipeline (mlx_vlm.convert casts all tensors
bf16->fp16 before quantization, confirmed by the 0.5B unquantized MLX
checkpoint being entirely fp16).

  Stage 2 (all components): bf16->fp16, mandatory, always applied.
  Stage 3 (decoder + projector only, optional): fp16->int8/int4 for
    nn.Linear and nn.Embedding weights; non-linear tensors stay fp16.

Use --quantize int8 or --quantize int4 to apply Stage 3 quantization to the
decoder and projector before export. Vision encoder is never quantized.

fp32 is NOT used as an export dtype — it appears only as a diagnostic dtype
in verify_decoder.py Stage 1 (structural correctness checking).

VALIDATED PRODUCTION TARGETS
------------------------------
  0.5B : --quantize none (fp16 only; Apple ships 0.5B unquantized)
  1.5B : --quantize int8 (49.6 dB vs fp16, PASS)
  7B   : --quantize int8 (int4 fails at 22.7 dB vs fp16)

COMPONENT-SPECIFIC EXPORT NOTES
-----------------------------------
vision_encode:
  Input: pixel_values [1, 3, image_size, image_size], static.
  Output: image_features [1, N, mm_hidden_size].

project:
  Input: image_features [1, N, mm_hidden_size]. N (patch count) is STATIC.
  Output: projected_features [1, N, hidden_size].
  Quantization: if --quantize is set, both Linear layers (layers.0, layers.2)
  are quantized. Biases stay fp16. Matches Apple's MLX scheme exactly.

decode:
  Input: input_ids [1, query_len], position_ids [1, seq_len] (DYNAMIC).
  state_names=["k_cache", "v_cache"] binds registered buffers as mutable
  state. Requires coreai::mutable_slice_update with mutates_args=["x"].
  Quantization: if --quantize is set, all nn.Linear and nn.Embedding weights
  are quantized. Non-linear tensors (norms, biases) stay fp16.

USAGE
-----
  # Export all three components, no quantization (0.5B):
  python scripts/export_fastvlm.py --variant 0.5b

  # Export all three, decoder + projector quantized to int8 (1.5B):
  python scripts/export_fastvlm.py --variant 1.5b --quantize int8

  # Export decoder only (for iteration):
  python scripts/export_fastvlm.py --variant 1.5b --quantize int8 --components decode

ARGUMENTS
---------
  --variant      FastVLM variant. Default: 1.5b. Choices: 0.5b, 1.5b, 7b.
  --quantize     Quantization level for decoder + projector. Default: none.
                 Choices: int8, int4. Vision encoder is never quantized.
  --output-dir   Where to write the .aimodel. Default: exports/ in repo root.
  --overwrite    Overwrite an existing .aimodel at the output path.
  --components   Which entrypoints to stage. Default: all three.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from fastvlm_decoder import FastVLMDecoderStateful, MAX_SEQ_LEN  # noqa: E402
from fastvlm_projector import FastVLMProjector  # noqa: E402
from fastvlm_vision_encoder import FastVLMVisionEncoder  # noqa: E402
from quantization import (  # noqa: E402
    QUANTIZATION_LEVELS, apply_quantization, finalize_for_export
)
from coreai._compiler.dialects import coreai as coreai_dialect  # noqa: E402
from coreai_torch._utils import get_operand as _get_operand  # noqa: E402

ALL_COMPONENTS = ["vision_encode", "project", "decode"]
EXPORT_DTYPE = torch.float16  # mandatory ANE precision; not configurable


def _image_size(config) -> int:
    """Image size is encoded in mm_vision_tower (e.g. 'mobileclip_l_1024')."""
    return int(config.mm_vision_tower.split("_")[-1])


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "exports")


def _asset_path(output_dir: str, variant: str, quantize: Optional[str]) -> Path:
    suffix = f"_fp16_{quantize}" if quantize else "_fp16"
    return Path(output_dir) / f"fastvlm-{variant}{suffix}.aimodel"


def _save_asset(coreai_program, model_path: Path, overwrite: bool) -> None:
    if model_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{model_path} already exists. Pass --overwrite to replace it."
            )
        if model_path.is_dir():
            shutil.rmtree(model_path)
        else:
            model_path.unlink()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    coreai_program.save_asset(model_path, _build_aimodel_metadata())


def _build_aimodel_metadata() -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "Apple (FastVLM); re-authored by Agentive Intent LLC"
    metadata.license = "See FastVLM repository license"
    metadata.model_description = (
        "FastVLM: vision encoder (FastViTHD) + multimodal projector (mlp2x_gelu) "
        "+ language decoder (Qwen2), re-authored for Core AI export. Three "
        "callable entry points: vision_encode, project, decode."
    )
    metadata.creation_date = int(time.time())
    return metadata


def _export_program(model: torch.nn.Module, example_inputs: dict):
    """
    Trace and decompose a model for Core AI export.
    Always traces at EXPORT_DTYPE (float16).
    """
    exported = torch.export.export(model, args=(), kwargs=example_inputs)
    return exported.run_decompositions(get_decomp_table())


def _stage_vision_encoder(
    converter: TorchConverter, config, weights_dir: str, quantize: Optional[str]
) -> None:
    """Vision encoder is always fp16, never quantized."""
    image_size = _image_size(config)
    print(f"[INFO] Sourcing vision encoder (fp16, no quantization)...")
    model = FastVLMVisionEncoder.from_weights(config, weights_dir, dtype=EXPORT_DTYPE)
    model.eval()

    example_inputs = {
        "pixel_values": torch.randn(1, 3, image_size, image_size).to(EXPORT_DTYPE)
    }
    print("[INFO] Tracing vision_encode...")
    exported = _export_program(model, example_inputs)

    converter.add_exported_program(
        exported_program=exported,
        input_names=["pixel_values"],
        output_names=["image_features"],
        entrypoint_name="vision_encode",
    )


def _stage_projector(
    converter: TorchConverter, config, weights_dir: str, quantize: Optional[str]
) -> None:
    """
    Projector exported at fp16. If --quantize is set, both Linear layers are
    quantized (fp16->int8/int4), matching Apple's MLX scheme exactly.
    """
    q_str = f" + {quantize} quantization" if quantize else ", no quantization"
    print(f"[INFO] Sourcing projector (fp16{q_str})...")
    model = FastVLMProjector.from_weights(config, weights_dir, dtype=EXPORT_DTYPE)
    model.eval()

    # N (patch count) is static — probe via vision encoder forward.
    image_size = _image_size(config)
    probe_encoder = FastVLMVisionEncoder.from_weights(
        config, weights_dir, dtype=torch.float32
    )
    probe_encoder.eval()
    with torch.no_grad():
        probe_out = probe_encoder(
            torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
        )
    n_patches = probe_out.shape[1]
    del probe_encoder, probe_out

    example_inputs = {
        "x": torch.randn(1, n_patches, config.mm_hidden_size).to(EXPORT_DTYPE)
    }

    if quantize:
        print(f"[INFO] Applying {quantize} quantization to projector...")
        model, quantizer = apply_quantization(model, quantize, (example_inputs["x"],))
        model = finalize_for_export(model, quantizer)

    print(f"[INFO] Tracing project (N={n_patches} patches)...")
    exported = _export_program(model, example_inputs)

    converter.add_exported_program(
        exported_program=exported,
        input_names=["x"],
        output_names=["projected_features"],
        entrypoint_name="project",
    )


def _stage_decoder(
    converter: TorchConverter, config, weights_dir: str, quantize: Optional[str]
) -> None:
    """
    Decoder exported at fp16. If --quantize is set, all nn.Linear and
    nn.Embedding weights are quantized (fp16->int8/int4); non-linear
    tensors (norms, biases) stay fp16.
    """
    text_cfg = getattr(config, "text_config", config)
    q_str = f" + {quantize} quantization" if quantize else ", no quantization"
    print(f"[INFO] Sourcing decoder (fp16{q_str})...")
    model = FastVLMDecoderStateful.from_weights(text_cfg, weights_dir)
    # from_weights loads at fp16 already; no additional cast needed.
    model.eval()

    if quantize:
        print(f"[INFO] Applying {quantize} quantization to decoder...")
        example_q = (
            torch.randint(1, text_cfg.vocab_size, (1, 8), dtype=torch.int32),
            torch.arange(8, dtype=torch.int32).unsqueeze(0),
        )
        model, quantizer = apply_quantization(model, quantize, example_q)
        model = finalize_for_export(model, quantizer)

    query_len = 8
    example_inputs = {
        "input_ids": torch.randint(
            1, text_cfg.vocab_size, (1, query_len), dtype=torch.int32
        ),
        "position_ids": torch.arange(query_len, dtype=torch.int32).unsqueeze(0),
    }

    # After finalize_for_export, the quantized model may specialize seq_len
    # to the example value (8) when a named Dim is used. Use Dim.AUTO to let
    # torch.export infer dynamism from the model rather than enforcing it.
    # For the unquantized path, the named Dim with max=MAX_SEQ_LEN works fine.
    if quantize:
        dynamic_shapes = {
            "input_ids": {1: torch.export.Dim.AUTO},
            "position_ids": {1: torch.export.Dim.AUTO},
        }
    else:
        seq_len_dim = torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)
        dynamic_shapes = {
            "input_ids": {1: seq_len_dim},
            "position_ids": {1: seq_len_dim},
        }

    print(f"[INFO] Tracing decode (max_seq_len={MAX_SEQ_LEN})...")
    exported = torch.export.export(
        model, args=(), kwargs=example_inputs, dynamic_shapes=dynamic_shapes
    )
    exported = exported.run_decompositions(get_decomp_table())

    converter.add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "position_ids"],
        output_names=["logits"],
        state_names=["k_cache", "v_cache"],
        entrypoint_name="decode",
    )


_STAGE_FNS = {
    "vision_encode": _stage_vision_encoder,
    "project": _stage_projector,
    "decode": _stage_decoder,
}


def export_fastvlm(
    variant: str,
    quantize: Optional[str],
    output_dir: str,
    overwrite: bool,
    components: list[str],
) -> Path:
    weights_dir = f"weights/fastvlm-{variant}"
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

    converter = TorchConverter()

    # Register lowering for fastvlm::mutable_slice_update.
    # This custom op is used in FastVLMAttention to write new K/V vectors
    # into the KV cache buffers at runtime. It must lower to coreai.slice_update
    # so TorchConverter can emit a mutable state write in the compiled graph.
    #
    # Why fastvlm:: not coreai:: namespace: TorchConverter reserves the coreai
    # namespace for ops in _custom_to_core_resolver (quantization/compression ops).
    # register_torch_lowering() rejects reserved namespaces, so the op is
    # registered under fastvlm:: in fastvlm_decoder.py and lowered here.
    #
    # coreai.slice_update signature:
    #   slice_update(input, start_indices, end_indices, strides, update)
    # Our op signature (from fastvlm_decoder.py):
    #   mutable_slice_update(x, update, begin, end)  -- begin/end are 1D int32 tensors
    @converter.register_torch_lowering("fastvlm::mutable_slice_update.default")
    def _lower_mutable_slice_update(values_map, node, loc):
        x      = _get_operand(values_map, node, 0, loc)
        update = _get_operand(values_map, node, 1, loc)
        begin  = _get_operand(values_map, node, 2, loc)
        end    = _get_operand(values_map, node, 3, loc)
        rank = x.type.rank
        strides = [1] * rank
        return coreai_dialect.slice_update(
            x,
            start_indices=begin,
            end_indices=end,
            strides=strides,
            update=update,
            loc=loc,
        )

    for name in components:
        _STAGE_FNS[name](converter, config, weights_dir, quantize)

    print(f"[INFO] Converting {len(components)} entrypoint(s) to Core AI...")
    coreai_program = converter.to_coreai()
    print("[INFO] Model converted.")
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    model_path = _asset_path(output_dir, variant, quantize)
    _save_asset(coreai_program, model_path, overwrite)
    print(f"[INFO] Successfully created and saved Core AI model to {model_path}.")
    return model_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export FastVLM to a Core AI .aimodel (fp16, optionally quantized).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/export_fastvlm.py --variant 0.5b
  python scripts/export_fastvlm.py --variant 1.5b --quantize int8
  python scripts/export_fastvlm.py --variant 1.5b --quantize int8 --components decode
  python scripts/export_fastvlm.py --variant 0.5b --overwrite
""",
    )
    ap.add_argument(
        "--variant",
        default="1.5b",
        choices=["0.5b", "1.5b", "7b"],
        help="FastVLM variant to export. (default: 1.5b)",
    )
    ap.add_argument(
        "--quantize",
        default=None,
        choices=QUANTIZATION_LEVELS,
        help="Quantization level for decoder + projector. Vision encoder is "
             "never quantized. Default: none (fp16 only).",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the .aimodel asset. (default: <repo-root>/exports/)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing .aimodel asset at the output path.",
    )
    ap.add_argument(
        "--components",
        nargs="+",
        default=ALL_COMPONENTS,
        choices=ALL_COMPONENTS,
        help="Which entrypoints to stage. (default: all three)",
    )
    args = ap.parse_args()

    output_dir = args.output_dir or _default_output_dir()
    export_fastvlm(args.variant, args.quantize, output_dir, args.overwrite, args.components)


if __name__ == "__main__":
    main()
