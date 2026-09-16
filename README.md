# fastvlm-coreai

Export pipeline converting [FastVLM](https://github.com/apple/ml-fastvlm) to
Apple's [Core AI](https://developer.apple.com/documentation/coreai) `.aimodel`
format, targeting on-device inference via `CoreAISequentialVLMEngine` on Apple
Silicon.

## What This Produces

A three-component VLM bundle compatible with `CoreAISequentialVLMEngine`:

```
exports/fastvlm-{variant}/
  vision.aimodel              — FastViTHD encoder + mlp2x_gelu projector
  embed.aimodel               — Token embedding lookup (input_ids → embeddings)
  fastvlm-{variant}.aimodel   — Qwen2 decoder with stateful KV cache
  tokenizer/                  — Qwen2 tokenizer + <image> special token (ID 151646)
  metadata.json               — Bundle manifest (kind=vlm) with provenance
```

Supported variants: `0.5b`, `1.5b`, `7b`

## Requirements

### Hardware
- Apple Silicon Mac
- macOS 27 GM or later

### Software
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Xcode 27+ with Metal Toolchain (`xcrun coreai-build`)

---

## Installation

### 1. Clone this repo

```bash
git clone https://github.com/tmorales2000/fastvlm-coreai.git
cd fastvlm-coreai
```

### 2. Clone coreai-models

`coreai-models` is required for the export pipeline. Clone Apple's upstream repo
directly — the PyPI version has an incorrect `Python>=3.14` constraint and cannot
be used (see [Known Issues](#known-issues)).

```bash
git clone https://github.com/apple/coreai-models.git ~/git/apple/coreai-models
```

### 3. Create the Python environment

```bash
uv sync
source .venv/bin/activate
```

Installs pinned versions matching Apple's coreai-models environment:

```
torch==2.9.0
coreai-core==1.0.0b2
coreai-torch==0.4.2
coreai-opt==0.2.1
```

### 4. Install coreai-models from source

```bash
uv pip install -e ~/git/apple/coreai-models/python/ --no-deps
```

> **Note:** `uv sync` will uninstall `coreai-models` because it is not in
> `pyproject.toml`. Always reinstall it after running `uv sync`.

### 5. Download model weights

Use `sync_weights.py` to download weights from HuggingFace. This records
provenance (HF revision, download timestamp) into `.provenance.json` alongside
the weights, which is later stamped into every exported bundle's `metadata.json`.

```bash
# Download a specific variant
python scripts/sync_weights.py --variant fastvlm-0.5b

# Download all registered models
python scripts/sync_weights.py --all

# Show download status
python scripts/sync_weights.py --list
```

See [Provenance](#provenance) for details on what gets recorded and why.

### 6. Download benchmark test images

```bash
python scripts/fetch_test_images.py
```

Downloads 9 public domain images to `test_assets/images/` for use with
verification scripts and `run_hf_fastvlm.py`.

### 7. Build llm-runner

Apple's `coreai-models` includes `llm-runner`, a Swift CLI that uses
`CoreAISequentialVLMEngine` to run any exported VLM bundle end-to-end:

```bash
cd ~/git/apple/coreai-models
swift build --product llm-runner
# Binary: .build/out/Products/Debug/llm-runner
cd -
```

### Verify installation

```bash
python3 -c "
from coreai_models.export.macos import export_to_coreai
from coreai_models.primitives.macos.cache import KVCache
print('coreai-models OK')
import coreai_torch; print(f'coreai-torch {coreai_torch.__version__}')
import torch; print(f'torch {torch.__version__}')
"
```

---

## Typical Workflow

### Export a bundle

```bash
# Full export — vision + embed + decoder (fp16, static KV, max_ctx=4096)
python scripts/export_fastvlm.py --variant 0.5b --overwrite

# With compression preset (recommended)
python scripts/export_fastvlm.py --variant 1.5b --compression 4bit --overwrite
python scripts/export_fastvlm.py --variant 1.5b --compression 8bit --overwrite
python scripts/export_fastvlm.py --variant 7b   --compression 4bit --overwrite

# With mixed-precision YAML recipe
python scripts/export_fastvlm.py --variant 1.5b \
    --compression-config quantization_recipes/fastvlm-1.5b-aggressive.yaml

# Dynamic KV cache (lower initial memory, useful for mobile)
python scripts/export_fastvlm.py --variant 0.5b --kv-cache dynamic --overwrite

# Smaller context for mobile deployment
python scripts/export_fastvlm.py --variant 1.5b --max-context-length 512 --overwrite
```

See [docs/RECIPES.md](docs/RECIPES.md) for compression options, benchmarks, and
how to generate per-model mixed-precision recipes with the scanner.

### Inspect the bundle

```bash
# Works on FastVLM and Qwen3-VL bundles (plain directory)
python scripts/inspect_aimodel.py exports/fastvlm-0.5b/

# Individual component
python scripts/inspect_aimodel.py exports/fastvlm-0.5b/fastvlm-0.5b.aimodel
```

Expected output (0.5B):
```
Bundle [PASS]: fastvlm-0.5b/
  embed.aimodel   [PASS]  input_ids int32 [1,-1] → embeddings fp16 [1,-1,896]
  fastvlm-0.5b    [PASS]  inputs_embeds fp16, k_cache/v_cache states
  vision.aimodel  [PASS]  pixel_values fp32 [1,3,1024,1024] → image_features fp16
```

### Verify correctness

FastVLM CoreAI uses a two-layer verification pipeline. See
[docs/VERIFICATION.md](docs/VERIFICATION.md) for the complete guide.

**Layer 1 — PyTorch verification** (no CoreAI runtime needed):

```bash
# Build fixtures first (one-time per variant, ~5s per image on MPS)
python scripts/build_fixtures.py --variant 0.5b

# Verify decoder — all four phases
python scripts/verify_decoder.py --variant 0.5b

# With compression recipe
python scripts/verify_decoder.py --variant 0.5b --compression 4bit

# Verify projector — two phases
python scripts/verify_projector.py --variant 0.5b

# Verify vision encoder — two phases
python scripts/verify_vision_encoder.py --variant 0.5b
```

**Layer 2 — CoreAI runtime verification** (requires exported bundle):

```bash
python scripts/verify_runtime.py --variant 0.5b \
    --image test_assets/images/great_wave.jpg
```

### Run inference with llm-runner

```bash
LLM_RUNNER=~/git/apple/coreai-models/.build/out/Products/Debug/llm-runner

# Text only
$LLM_RUNNER --model exports/fastvlm-0.5b \
  --prompt "What is the capital of France?" \
  --max-tokens 50

# Image + text (VLM)
$LLM_RUNNER --model exports/fastvlm-0.5b \
  --image test_assets/images/earthrise.jpg \
  --prompt "What do you see in this image?" \
  --max-tokens 300 --temperature 0

# Verbose timing (TTFT, throughput, memory)
$LLM_RUNNER --model exports/fastvlm-0.5b \
  --image test_assets/images/great_wave.jpg \
  --prompt "Describe this image." \
  --max-tokens 300 --temperature 0 --verbose
```

### Compare with HuggingFace reference

```bash
python scripts/run_hf_fastvlm.py \
  --variant 0.5b \
  --image test_assets/images/earthrise.jpg \
  --prompt "What do you see in this image?" \
  --temperature 0 --device mps
```

---

## Provenance

Every exported bundle's `metadata.json` is stamped with a provenance record
tracing the exact inputs that produced it:

```json
"source": {
  "hf_model_id":              "apple/FastVLM-0.5B",
  "model_definition":         "torch",
  "compression":              "4bit",
  "export_timestamp":         "2026-09-15T18:00:00+00:00",
  "fastvlm_coreai_git_sha":   "7f08f2807e2e",
  "hf_revision":              "fc2d1a370002183f44d8a1c1ac62cbf14e0c5c85",
  "hf_weights_downloaded_at": "2026-09-09T04:20:48+00:00"
}
```

This answers the question "where did this bundle come from?" unambiguously:

- **`hf_revision`** — exact HuggingFace commit SHA of the weights used
- **`fastvlm_coreai_git_sha`** — exact commit of this repo that ran the export
- **`export_timestamp`** — when the export ran
- **`hf_weights_downloaded_at`** — when the weights were downloaded from HF

Provenance is written automatically when you use `sync_weights.py` to download
weights. It requires no extra steps at export time.

### Model registry

`models.yaml` in the repo root is the single source of truth for which models
this project supports:

```bash
# See all registered models and their download status
python scripts/sync_weights.py --list

# Download a specific model
python scripts/sync_weights.py --variant fastvlm-1.5b

# Force re-download (picks up any HF updates)
python scripts/sync_weights.py --variant fastvlm-0.5b --force
```

MLX quantized weights (`apple/FastVLM-1.5B-int8`, etc.) are NOT in the registry.
They are managed exclusively by `compare_weights.py` and stored in the standard
HF cache (`~/.cache/huggingface/`).

---

## Verification System

The verification pipeline has two layers and covers three components
(vision encoder, projector, decoder). Each script runs independently.
See [docs/VERIFICATION.md](docs/VERIFICATION.md) for the complete guide.

### verify_decoder.py — Four phases

```bash
python scripts/build_fixtures.py --variant 0.5b  # one-time prerequisite
python scripts/verify_decoder.py --variant 0.5b --compression 4bit
```

| Phase | Device | What it tests | Gate |
|-------|--------|---------------|------|
| 1 — Architecture correctness | CPU | Re-authored decoder vs HF Qwen2 in fp32. Random token sequence using real embedding-table inputs. | >80 dB PASS, 50–80 dB MARGINAL (exits nonzero) |
| 2 — FP16 fidelity | MPS | Decoder fp32 vs fp16 on real fixture inputs. Establishes the fp16 deployment baseline for Phase 4. | Informational (MEASURED) |
| 3 — KV cache correctness | MPS | Incremental cached decode vs full-pass reference. Catches cache offset and head reshape bugs. | >40 dB |
| 4 — Compression quality | CPU prepare → MPS infer | Compressed vs fp16 over 9-image corpus at the final generation position. | Top-5 overlap ≥80% mean, ≥60% worst |

**On PSNR:** PSNR is reported as context throughout but is NOT the gate for any
phase. The 21.7 dB PSNR for 1.5B int4 fails any reasonable PSNR threshold yet
the model produces clean output at 115 tok/sec. Behavioral metrics (top-5 overlap,
KL divergence) are the evidence. See [docs/VERIFICATION.md](docs/VERIFICATION.md).

### verify_projector.py — Two phases

```bash
python scripts/verify_projector.py --variant 0.5b
python scripts/verify_projector.py --variant 0.5b --stage correctness
python scripts/verify_projector.py --variant 0.5b --stage fidelity
```

| Phase | Device | What it tests | Gate |
|-------|--------|---------------|------|
| 1 — Architecture correctness | CPU | Re-authored FastVLMProjector vs HF mm_projector (nn.Sequential) in fp32. | >60 dB PASS, 40–60 dB MARGINAL |
| 2 — FP16 fidelity | CPU | Port fp32 vs port fp16. Measures bf16→fp16 cast cost. | Informational (MEASURED) |

Phase 1 typically produces inf dB (bit-identical) because the projector is two
`nn.Linear` layers — no architectural difference to introduce error.

### verify_vision_encoder.py — Two phases

```bash
python scripts/verify_vision_encoder.py --variant 0.5b
python scripts/verify_vision_encoder.py --variant 0.5b --stage correctness
python scripts/verify_vision_encoder.py --variant 0.5b --stage fidelity
```

| Phase | Device | What it tests | Gate |
|-------|--------|---------------|------|
| 1 — Architecture correctness | CPU | Re-authored encoder vs full HF model integration at fp32. | >70 dB PASS, 50–70 dB MARGINAL |
| 2 — FP16 fidelity | MPS | Port fp32 vs port fp16 on 9 real corpus images via HF image processor. | Informational (MEASURED) |

**Key findings:**
- Phase 2 uses MPS — fp16 overflows on CPU at 1024×1024 (FastViTHD has 186
  conv2d ops, and fp16 saturates at network.9 with random inputs).
- Phase 2 uses real corpus images — random N(0,1) inputs produce completely
  misleading metrics (cosine=0.04) due to conv networks saturating outside their
  training distribution. Real images produce cosine≈1.0, PSNR≈81 dB.
- Phase 2 requires `.eval()` called AFTER `.to(device)` on MPS — calling eval
  on CPU then moving to MPS leaves MPS state incorrectly initialized.

### Device strategy

| Script / Phase | Device | Reason |
|----------------|--------|--------|
| verify_decoder Phase 1 | CPU | IEEE 754 strict fp32 for architecture correctness |
| verify_decoder Phases 2-3 | MPS | Deployment accuracy, fast |
| verify_decoder Phase 4 prepare | CPU | coreai_opt RoPEImpl incompatible with MPS |
| verify_decoder Phase 4 infer | MPS | Fast corpus evaluation |
| verify_projector (both phases) | CPU | Architecture correctness, projector is small |
| verify_vision_encoder Phase 1 | CPU | Architecture correctness |
| verify_vision_encoder Phase 2 | MPS | fp16 overflows on CPU; real images required |

---

## Scripts

### Weight management

| Script | Purpose |
|--------|---------|
| `sync_weights.py` | Download/verify model weights from HuggingFace. Writes `.provenance.json`. `--list`, `--variant`, `--all`, `--force`. |
| `compare_weights.py` | Compare Apple's MLX quantized weights vs HF bf16 source by dequantizing and measuring PSNR. MLX weights auto-downloaded from HF cache. |

### Export pipeline

| Script | Purpose |
|--------|---------|
| `export_fastvlm.py` | Main export script. Produces the full VLM bundle. Supports `--variant`, `--compression`, `--compression-config`, `--kv-cache`, `--max-context-length`. |
| `fastvlm_decoder.py` | Re-authored Qwen2 decoder for CoreAI export. Imported by `export_fastvlm.py`. |
| `fastvlm_vision_encoder.py` | Re-authored FastViTHD vision encoder. Imported by `export_fastvlm.py`. |
| `fastvlm_projector.py` | mlp2x_gelu projector module. Imported by `export_fastvlm.py`. |
| `quantization.py` | Compression presets (`none`, `4bit`, `8bit`). `load_compression_config()`, `apply_quantization_from_config()`. See [docs/RECIPES.md](docs/RECIPES.md). |
| `scan_quantization_sensitivity.py` | Per-layer sensitivity scanner. Generates mixed-precision YAML recipes. See [docs/RECIPES.md](docs/RECIPES.md). |

### Verification

| Script | Purpose |
|--------|---------|
| `verify_decoder.py` | **Layer 1:** Four-phase decoder verification — architecture correctness, FP16 fidelity, KV cache correctness, compression quality. See [docs/VERIFICATION.md](docs/VERIFICATION.md). |
| `verify_projector.py` | **Layer 1:** Two-phase projector verification — port vs HF mm_projector (Phase 1), FP16 fidelity (Phase 2). |
| `verify_vision_encoder.py` | **Layer 1:** Two-phase vision encoder verification — architecture correctness (Phase 1), FP16 fidelity on real corpus images (Phase 2). |
| `verify_runtime.py` | **Layer 2:** CoreAI compiled model vs PyTorch reference PSNR across all pipeline stages. Run on macOS 27 GM. |
| `metrics.py` | Canonical metric module (PSNR, NRMSE, cosine, KL divergence, top-k agreement, margin preservation). Shared by verify_decoder and scanner. |
| `fastvlm_fixtures.py` | Realistic decoder input fixtures from the full HF multimodal pipeline. Shared by verify_decoder and scanner. |
| `build_fixtures.py` | Pre-build and cache decoder fixtures for verify_decoder and scanner. Run once per variant. |

### Inspection and test assets

| Script | Purpose |
|--------|---------|
| `inspect_aimodel.py` | Inspect any CoreAI VLM bundle directory or individual `.aimodel` file. Reports inputs, outputs, state names, KV cache behavior, tokenizer. |
| `fetch_test_images.py` | Download 9 public domain benchmark images to `test_assets/images/`. Run once after cloning. |
| `generate_test_images.py` | Generate synthetic test images (tall_narrow_circle.png, wide_short_square.png) for preprocessing strategy verification. |
| `run_hf_fastvlm.py` | Run FastVLM from original HF weights for ground truth comparison. Supports `--variant`, `--image`, `--prompt`, `--temperature`, `--device`. |
| `probe_vlm_config.py` | Probe any HF VLM config for native resolution and preprocessing metadata. |
| `probe_activations.py` | Profile intermediate activation magnitudes in the vision encoder. Used to identify fp16 overflow risk at network.8-10. |
| `discover_weights.py` | Dump weight shapes and dtypes from HF safetensors to `discovery/`. |
| `inspect_weights.py` | Human-readable PyTorch/MLX weight inspection. |
| `audit_weight_dtypes.py` | Exhaustive dtype/shape audit across HF and Apple MLX checkpoints. |

---

## Export Flags

### `--variant`
Model size. Affects decoder architecture and weight file.
- `0.5b` — 24 layers, hidden=896, 2 KV heads
- `1.5b` — 28 layers, hidden=1536, 2 KV heads
- `7b` — 32 layers, hidden=3584, 8 KV heads

### `--compression`
Named compression preset for the decoder. Vision encoder and embed are always fp16.
See [docs/RECIPES.md](docs/RECIPES.md) for benchmarks and mixed-precision YAML recipes.
- *(none)* — fp16, highest quality
- `8bit` — int8 symmetric_with_clipping per_block_32, ~2.5× smaller, 71 tok/sec (1.5B)
- `4bit` — int4 symmetric_with_clipping per_block_32, ~3.5× smaller, 115 tok/sec (1.5B) — recommended

### `--compression-config`
Path to a mixed-precision YAML recipe generated by `scan_quantization_sensitivity.py`.
Mutually exclusive with `--compression`.

### `--kv-cache`
KV cache allocation strategy.
- `static` *(default)* — pre-allocates `max_ctx` tokens upfront. `StaticKVCache` in Swift.
- `dynamic` — starts at 256 tokens, grows 2× as needed. `GrowingKVCache` in Swift.
  Lower initial memory — useful for mobile deployment with small expected context.

### `--max-context-length`
Maximum context length in tokens. Default: 4096. Hard ceiling in both static and
dynamic modes. For mobile deployment, `512` is a practical default that reduces
KV cache memory significantly.

---

## Architecture

### VLM Inference Flow

```
pixel_values [1, 3, 1024, 1024]
      ↓ vision.aimodel::encode_image
image_features [1, 256, 3072]
      ↓ vision.aimodel::project
projected_features [1, 256, hidden]   ← 256 image tokens in LM space
      ↓
all_token_ids [1, 256+N]              ← 256 <image> placeholders + N text tokens
      ↓ embed.aimodel::main
embeddings [1, 256+N, hidden]
      ↓ scatter-merge (CoreAISequentialVLMEngine)
merged_inputs_embeds [1, 256+N, hidden]  ← image positions replaced
      ↓ fastvlm-{variant}.aimodel::main (+ stateful KV cache)
logits [1, 256+N, 151936]             → sample next token
      ↓ repeat for each decode step
```

### Key Design Decisions

**`mutable_slice_update` for KV cache:** The only pattern supported by
`remove_functionalization` in `coreai-models`. `slice_scatter` does not work for
export. Imported from `coreai_models.primitives.macos.cache.KVCache`.

**`inputs_embeds` not `input_ids`:** The decoder takes pre-computed embeddings
so `CoreAISequentialVLMEngine` can scatter-merge image features before calling
the decoder. `embed_tokens` is a separate `embed.aimodel`.

**State names:** Python export uses `k_cache`/`v_cache`. The `coreai-torch`
compiler renames these to `keyCache`/`valueCache` (camelCase) in the compiled
model for Swift compatibility.

**Image normalization:** FastVLM uses no normalization (mean=0, std=1). Unlike
Qwen3-VL which uses ImageNet stats, FastVLM's vision tower was trained without
normalization.

**Image preprocessing:** FastVLM uses shortest-edge resize + center crop to
1024×1024. The bundle declares `"image_strategy": "center_crop"` in
`metadata.json`, which `CoreAISequentialVLMEngine` reads to select the correct
resize algorithm.

**Performance (M4 Pro, 1.5B int4):** 172ms TTFT warm, 115 tok/sec, 2.0GB peak
memory. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for full benchmarks.

---

## Documentation

| File | Contents |
|------|---------|
| `models.yaml` | Model registry — all supported models, HF repos, local directories |
| `docs/PERFORMANCE.md` | Benchmark results — throughput, TTFT, memory, quality |
| `docs/RECIPES.md` | Compression presets, YAML recipes, scanner pipeline |
| `docs/VERIFICATION.md` | Complete verification guide — both layers, all phases, device strategy |
| `docs/FASTVLM_ARCHITECTURE.md` | Architecture deep-dive — components, quantization, export gotchas |
| `docs/FASTVLM_SWIFT_INTEGRATION.md` | Swift integration guide — llm-runner, CoreAISequentialVLMEngine |
| `docs/STATUS.md` | Current project status, known issues, pending items |

---

## Known Issues

### `coreai-models` PyPI version

The PyPI version of `coreai-models==0.1.0` has an incorrect `Python>=3.14`
constraint and cannot be installed on Python 3.11. Always install from the
GitHub source (see [Installation](#installation)).

### `uv sync` uninstalls coreai-models

`uv sync` enforces exactly what is declared in `pyproject.toml`. Since
`coreai-models` is installed as a local editable install outside of
`pyproject.toml`, it is uninstalled every time you run `uv sync`. Always
reinstall it afterward:

```bash
uv sync
uv pip install -e ~/git/apple/coreai-models/python/ --no-deps
```

### Vision encoder fp32 on ANE

`vision.aimodel` accepts `pixel_values` as `float32`. On some platforms,
`AIModel.load()` triggers ANE compilation which rejects fp32 inputs.
`inspect_aimodel.py` handles this gracefully. The fp32→fp16 cast happens
as the first op inside the model.

---

## Relationship to Apple's coreai-models

This repo follows Apple's authoritative VLM export recipe from
`coreai-models/vlm/export.py` exactly, with FastVLM-specific additions:

- Re-authored `FastVLMVisionEncoder` (FastViTHD via `trust_remote_code`)
- Re-authored `FastVLMDecoder` (Qwen2, matching `Qwen3VLForCausalLMEmbeddings.forward()`)
- `<image>` special token added to Qwen2 tokenizer (ID 151646)
- `--compression`, `--compression-config`, `--kv-cache`, `--max-context-length` export flags
- Compression support (4bit, 8bit) — not available in Apple's VLM exporter
- Two-layer verification pipeline with corpus-based behavioral metrics
- Provenance chain from HF weights to exported bundle

### Issues filed against apple/coreai-models

| Issue | Status | Description |
|-------|--------|-------------|
| [#96](https://github.com/apple/coreai-models/issues/96) | 🔲 Open | PyPI wheel declares incorrect `Python>=3.14` constraint |
| [#100](https://github.com/apple/coreai-models/issues/100) | ✅ Closed — merged as [#108](https://github.com/apple/coreai-models/pull/108) | `CoreAISequentialVLMEngine` image preprocessing strategy support |
