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

WHY fp32 IS THE DEFAULT FOR ALL THREE
------------------------------------------
Every export.py in coreai-models defaults to dtype=float32 (see
export_fastvlm.py's original module docstring for the full survey). This
script follows the same convention for all three components, not just the
vision encoder. The decoder and projector have NO known precision issues
(decoder Stage 2 fp16 health passes cleanly for all variants; projector is
bit-identical to the HF original in fp32 and passes fp16 health at 90+ dB)
— but fp32 is still the safer default per Apple's own convention, and
--dtype float16 remains available per-export if you have reason to override it.

COMPONENT-SPECIFIC EXPORT NOTES
-----------------------------------
vision_encode:
  Input: pixel_values [1, 3, image_size, image_size], static (image_size is
  fixed per variant via config.mm_vision_tower — see fastvlm_vision_encoder.py).
  Output: image_features [1, N, mm_hidden_size]. KNOWN OPEN QUESTION: fp16
  overflow risk at network.8-10 for 0.5b/1.5b — see verify_compiled_vision_encoder.py.

project:
  Input: image_features [1, N, mm_hidden_size]. N (patch count) is STATIC,
  not dynamic — it is fully determined by the fixed input image_size and
  FastViTHD's fixed stride/downsampling stages, not by anything that varies
  at runtime (confirmed: every test run produces exactly [1, 256, 3072] for
  1.5b's 1024x1024 input; there is no FastVLM code path that varies image
  resolution per-inference). Output: projected_features [1, N, hidden_size].

decode:
  Input: input_ids [1, query_len] (query_len varies: full prompt during
  prefill, 1 during single-token decode), position_ids [1, seq_len] (DYNAMIC
  — grows by 1 each decode step, up to model.max_seq_len). state_names=
  ["k_cache", "v_cache"] binds the decoder's registered buffers as mutable
  state in the exported graph — REQUIRES the coreai::mutable_slice_update
  custom op with mutates_args=["x"] to be correctly registered (see
  fastvlm_decoder.py module docstring) or the exporter will not recognize
  the cache writes as state mutations.

USAGE
-----
  python scripts/export_fastvlm.py --variant 1.5b
  python scripts/export_fastvlm.py --variant 1.5b --dtype float16
  python scripts/export_fastvlm.py --variant 0.5b --output-dir ./exports/ --overwrite
  python scripts/export_fastvlm.py --variant 1.5b --components vision_encode project

ARGUMENTS
---------
  --variant      FastVLM variant to export. Default: 1.5b.
                 Choices: 0.5b, 1.5b, 7b.
  --dtype        Torch dtype to trace ALL staged components in. Default: float32.
                 Choices: float32, float16.
  --output-dir   Where to write the .aimodel. Default: exports/ in repo root.
  --overwrite    Overwrite an existing .aimodel at the output path.
  --components   Which entrypoints to stage. Default: all three.
                 Choices (one or more): vision_encode, project, decode.
                 Useful for testing one component's export in isolation.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from fastvlm_decoder import FastVLMDecoderStateful  # noqa: E402
from fastvlm_projector import FastVLMProjector  # noqa: E402
from fastvlm_vision_encoder import FastVLMVisionEncoder  # noqa: E402

ALL_COMPONENTS = ["vision_encode", "project", "decode"]


def _image_size(config) -> int:
    """Image size is encoded in mm_vision_tower (e.g. 'mobileclip_l_1024')."""
    return int(config.mm_vision_tower.split("_")[-1])


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "exports")


def _asset_path(output_dir: str, variant: str, dtype: torch.dtype) -> Path:
    dtype_name = str(dtype).split(".")[-1]
    return Path(output_dir) / f"fastvlm-{variant}_{dtype_name}.aimodel"


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


def _export_program(model: torch.nn.Module, example_inputs: dict, dtype: torch.dtype):
    """
    Shared trace+decompose step for all three components.

    autocast is skipped for float32 (it's a no-op there and emits a spurious
    warning — CPU autocast only supports promoting/demoting to bfloat16 or
    float16, there's nothing to autocast TO when the target is float32).
    """
    if dtype == torch.float32:
        exported = torch.export.export(model, args=(), kwargs=example_inputs)
    else:
        with torch.autocast(device_type="cpu", dtype=dtype):
            exported = torch.export.export(model, args=(), kwargs=example_inputs)
    return exported.run_decompositions(get_decomp_table())


def _stage_vision_encoder(
    converter: TorchConverter, config, weights_dir: str, dtype: torch.dtype
) -> None:
    image_size = _image_size(config)
    print(f"[INFO] Sourcing vision encoder (dtype={dtype})...")
    model = FastVLMVisionEncoder.from_weights(config, weights_dir, dtype=dtype)
    model.eval()

    example_inputs = {"pixel_values": torch.randn(1, 3, image_size, image_size).to(dtype)}
    print("[INFO] Tracing vision_encode...")
    exported = _export_program(model, example_inputs, dtype)

    converter.add_exported_program(
        exported_program=exported,
        input_names=["pixel_values"],
        output_names=["image_features"],
        entrypoint_name="vision_encode",
    )


def _stage_projector(
    converter: TorchConverter, config, weights_dir: str, dtype: torch.dtype
) -> None:
    print(f"[INFO] Sourcing projector (dtype={dtype})...")
    model = FastVLMProjector.from_weights(config, weights_dir, dtype=dtype)
    model.eval()

    # N (patch count) is static — see module docstring "project" section.
    # 256 confirmed empirically for 1.5b at 1024x1024; recompute per-variant
    # rather than hardcoding, in case a future variant's image_size differs.
    image_size = _image_size(config)
    # FastViTHD downsamples by 32x total (5 stride-2 PatchEmbed stages, but
    # conv_exp's stride is 1 — net stride is determined empirically, not
    # assumed). Run a tiny shape probe via the vision encoder's own forward
    # rather than computing this by hand, so it can never silently drift
    # from the real architecture.
    probe_encoder = FastVLMVisionEncoder.from_weights(config, weights_dir, dtype=torch.float32)
    probe_encoder.eval()
    with torch.no_grad():
        probe_out = probe_encoder(torch.zeros(1, 3, image_size, image_size, dtype=torch.float32))
    n_patches = probe_out.shape[1]
    del probe_encoder, probe_out

    example_inputs = {
        "x": torch.randn(1, n_patches, config.mm_hidden_size).to(dtype)
    }
    print(f"[INFO] Tracing project (N={n_patches} patches)...")
    exported = _export_program(model, example_inputs, dtype)

    converter.add_exported_program(
        exported_program=exported,
        input_names=["x"],
        output_names=["projected_features"],
        entrypoint_name="project",
    )


def _stage_decoder(
    converter: TorchConverter, config, weights_dir: str, dtype: torch.dtype
) -> None:
    text_cfg = getattr(config, "text_config", config)
    print(f"[INFO] Sourcing decoder (dtype={dtype})...")
    model = FastVLMDecoderStateful.from_weights(text_cfg, weights_dir)
    model = model.to(dtype=dtype)
    model.eval()

    # Example shapes for tracing: a short prefill (query_len=8, seq_len=8).
    # The dynamic_shapes spec below is what actually allows seq_len to vary
    # at runtime up to model.max_seq_len — these example values only need
    # to be valid, not representative of every call shape.
    query_len = 8
    example_inputs = {
        "input_ids": torch.randint(1, text_cfg.vocab_size, (1, query_len), dtype=torch.int32),
        "position_ids": torch.arange(query_len, dtype=torch.int32).unsqueeze(0),
    }

    seq_len_dim = torch.export.Dim("seq_len", min=1, max=model.max_seq_len)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "position_ids": {1: seq_len_dim},
    }

    print(f"[INFO] Tracing decode (max_seq_len={model.max_seq_len})...")
    if dtype == torch.float32:
        exported = torch.export.export(
            model, args=(), kwargs=example_inputs, dynamic_shapes=dynamic_shapes
        )
    else:
        with torch.autocast(device_type="cpu", dtype=dtype):
            exported = torch.export.export(
                model, args=(), kwargs=example_inputs, dynamic_shapes=dynamic_shapes
            )
    exported = exported.run_decompositions(get_decomp_table())

    converter.add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "position_ids"],
        output_names=["logits"],
        # Order matches fastvlm_decoder.py buffer registration order:
        # k_cache then v_cache (see register_buffer calls in __init__).
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
    dtype: torch.dtype,
    output_dir: str,
    overwrite: bool,
    components: list[str],
) -> Path:
    weights_dir = f"weights/fastvlm-{variant}"
    config = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)

    converter = TorchConverter()
    for name in components:
        _STAGE_FNS[name](converter, config, weights_dir, dtype)

    print(f"[INFO] Converting {len(components)} entrypoint(s) to Core AI...")
    coreai_program = converter.to_coreai()
    print("[INFO] Model converted.")
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    model_path = _asset_path(output_dir, variant, dtype)
    _save_asset(coreai_program, model_path, overwrite)
    print(f"[INFO] Successfully created and saved Core AI model to {model_path}.")
    return model_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export FastVLM (vision encoder + projector + decoder) to a Core AI .aimodel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/export_fastvlm.py --variant 1.5b
  python scripts/export_fastvlm.py --variant 1.5b --dtype float16
  python scripts/export_fastvlm.py --variant 1.5b --components vision_encode project
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
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Torch dtype to trace all staged components in. (default: float32)",
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

    dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]
    output_dir = args.output_dir or _default_output_dir()

    export_fastvlm(args.variant, dtype, output_dir, args.overwrite, args.components)


if __name__ == "__main__":
    main()
