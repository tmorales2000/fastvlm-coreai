# fastvlm-coreai

Export pipeline converting [FastVLM](https://github.com/apple/ml-fastvlm) to
Apple's [Core AI](https://developer.apple.com/documentation/coreai) `.aimodel`
format, targeting on-device inference via `CoreAISequentialVLMEngine` on Apple
Silicon.

## What This Produces

A three-component VLM bundle compatible with `CoreAISequentialVLMEngine`:

```
exports/fastvlm-{variant}.vlmasset/
  vision.aimodel          — FastViTHD encoder + mlp2x_gelu projector
  embed.aimodel           — Token embedding lookup (input_ids → embeddings)
  fastvlm-{variant}.aimodel — Qwen2 decoder with stateful KV cache
  tokenizer/              — Qwen2 tokenizer + <image> special token (ID 151646)
  metadata.json           — Bundle manifest (kind=vlm)
```

Supported variants: `0.5b`, `1.5b`, `7b`

## Requirements

### Hardware
- Apple Silicon Mac
- macOS 26.5+ (macOS 27 beta has an MPSGraph bug affecting Python runtime
  verification on macOS 27 beta — see [Known Issues](#known-issues))

### Software
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Xcode 27+ with Metal Toolchain (`xcrun coreai-build`)

## Installation

### 1. Clone this repo

```bash
git clone https://github.com/tmorales2000/fastvlm-coreai.git
cd fastvlm-coreai
```

### 2. Clone coreai-models

`coreai-models` is required for the export pipeline. Clone Apple's upstream repo
directly — image preprocessing support (`center_crop`, `pad`, `stretch`) was
merged in [apple/coreai-models #108](https://github.com/apple/coreai-models/pull/108),
which closed [#100](https://github.com/apple/coreai-models/issues/100).

```bash
git clone https://github.com/apple/coreai-models.git ~/git/apple/coreai-models
```

### 3. Create the Python environment

```bash
uv sync
source .venv/bin/activate
```

This installs pinned versions matching Apple's own coreai-models environment:

```
torch==2.9.0        — required for asset.executable() with stateful KV cache
coreai-core==1.0.0b2
coreai-torch==0.4.1
coreai-opt==0.2.1
```

### 4. Install coreai-models from source

```bash
uv pip install -e ~/git/apple/coreai-models/python/ --no-deps
```

> **Note:** This step must be repeated after every `uv sync` because `uv sync`
> does not preserve editable installs from local paths outside the project.
> The PyPI version of `coreai-models` has an incorrect `Python>=3.14` constraint
> and cannot be used (see [apple/coreai-models #96](https://github.com/apple/coreai-models/issues/96)).

### 5. Download FastVLM weights

```bash
hf download apple/FastVLM-0.5B --local-dir weights/fastvlm-0.5b
# Repeat for 1.5b and 7b as needed:
# hf download apple/FastVLM-1.5B --local-dir weights/fastvlm-1.5b
# hf download apple/FastVLM-7B   --local-dir weights/fastvlm-7b
```

### 6. Download benchmark test images

```bash
python scripts/fetch_test_images.py
```

This downloads 9 public domain benchmark images to `test_assets/images/`
for use with `llm-runner` and `run_hf_fastvlm.py`. See
`test_assets/images/README.md` for the full catalog and test purposes.

### 7. Build llm-runner (Swift inference tool)

Apple's `coreai-models` includes `llm-runner`, a Swift CLI that uses
`CoreAISequentialVLMEngine` to run any exported VLM bundle end-to-end:

```bash
cd ~/git/apple/coreai-models
swift build --product llm-runner
# Binary: .build/out/Products/Debug/llm-runner
cd -   # return to fastvlm-coreai
```

> **Note:** First build takes ~20 seconds. Subsequent builds are incremental.
> If `swift build --product` doesn't produce a binary, run
> `swift package clean` first then retry.

### Verify installation

```bash
python3 -c "
from coreai_models.export.macos import export_to_coreai
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.export.mlir_ops import remove_functionalization
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

# With quantization
python scripts/export_fastvlm.py --variant 1.5b --quantize int8 --overwrite
python scripts/export_fastvlm.py --variant 7b   --quantize int4 --overwrite

# Dynamic KV cache (GrowingKVCache in Swift, lower initial memory)
python scripts/export_fastvlm.py --variant 0.5b --kv-cache dynamic --overwrite

# Extended context (up to max_position_embeddings=32768)
python scripts/export_fastvlm.py --variant 0.5b --max-context-length 8192 --overwrite

# Export individual components
python scripts/export_fastvlm.py --variant 0.5b --components decode --overwrite
```

### Inspect the bundle

```bash
# Works on FastVLM (.vlmasset) and Qwen3-VL (.llmasset) bundles
python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset

# Individual component
python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset/fastvlm-0.5b.aimodel
```

Expected output (0.5B):
```
Bundle [PASS]: fastvlm-0.5b.vlmasset
  embed.aimodel   [PASS]  input_ids int32 [1,-1] → embeddings fp16 [1,-1,896]
  fastvlm-0.5b    [PASS]  inputs_embeds fp16, keyCache/valueCache states
  vision.aimodel  [PASS]  pixel_values fp32 [1,3,1024,1024] → image_features fp16
```

### Verify correctness

FastVLM CoreAI uses a two-layer verification pipeline. See
[docs/VERIFICATION.md](docs/VERIFICATION.md) for the complete guide.

**Layer 1 — PyTorch verification** (no CoreAI runtime needed):

```bash
# Build fixtures first (one-time per variant)
python scripts/build_fixtures.py --variant 0.5b

# Run all four phases
python scripts/verify_decoder.py --variant 0.5b

# With compression
python scripts/verify_decoder.py --variant 0.5b --compression 4bit
```

**Layer 2 — CoreAI runtime verification** (requires exported bundle):

```bash
python scripts/verify_runtime.py --variant 0.5b     --image test_assets/images/great_wave.jpg
```

Expected Layer 2 output (0.5B fp16, real image):
```
  ✓ vision_encode PSNR:  71.9 dB  (PASS)
  ✓ embed_tokens PSNR:   inf dB   (PASS — bit-identical)
  ✓ decode step 1:       44.4 dB  (PASS > 40 dB)
[PASS] All stages match PyTorch reference.
```

### Run inference with llm-runner

`llm-runner` from Apple's `coreai-models` is the primary way to run the
exported bundle via `CoreAISequentialVLMEngine`:

```bash
LLM_RUNNER=~/git/apple/coreai-models/.build/out/Products/Debug/llm-runner

# Text only
$LLM_RUNNER --model exports/fastvlm-0.5b.vlmasset \
  --prompt "What is the capital of France?" \
  --max-tokens 50

# Image + text (VLM)
$LLM_RUNNER --model exports/fastvlm-0.5b.vlmasset \
  --image test_assets/images/earthrise.jpg \
  --prompt "What do you see in this image? Describe the colors and spatial arrangement." \
  --max-tokens 300 --temperature 0

# Hallucination resistance test
$LLM_RUNNER --model exports/fastvlm-0.5b.vlmasset \
  --image test_assets/images/pale_blue_dot.png \
  --prompt "Describe exactly what you see in this image." \
  --max-tokens 200 --temperature 0

# Verbose timing breakdown (TTFT, throughput, memory)
$LLM_RUNNER --model exports/fastvlm-0.5b.vlmasset \
  --image test_assets/images/great_wave.jpg \
  --prompt "Describe this image." \
  --max-tokens 300 --temperature 0 --verbose
```

> **`--max-tokens`** is the generation limit (default 50), not the KV cache limit.
> Set to 300-500 for typical use. The KV cache ceiling is `--max-context-length`
> set at export time (default 4096).

### Compare with HuggingFace reference

`run_hf_fastvlm.py` runs the original HF model for ground truth comparison:

```bash
# Run HF model on same image/prompt as llm-runner for direct comparison
python scripts/run_hf_fastvlm.py \
  --variant 0.5b \
  --image test_assets/images/earthrise.jpg \
  --prompt "What do you see in this image?" \
  --temperature 0

# MPS acceleration (faster on Apple Silicon)
python scripts/run_hf_fastvlm.py \
  --variant 0.5b \
  --image test_assets/images/great_wave.jpg \
  --prompt "Describe this image." \
  --temperature 0 --device mps
```

### Compile for ANE (ahead-of-time)

```bash
xcrun coreai-build compile \
  --preferred-compute neural-engine \
  exports/fastvlm-0.5b.vlmasset/fastvlm-0.5b.aimodel
```

---

## Scripts

### Export pipeline

| Script | Purpose |
|--------|---------|
| `export_fastvlm.py` | Main export script. Produces the full `.vlmasset` bundle. Supports `--variant`, `--quantize`, `--kv-cache`, `--max-context-length`. |
| `fastvlm_decoder.py` | Re-authored Qwen2 decoder module for CoreAI export. Imported by `export_fastvlm.py`. |
| `fastvlm_vision_encoder.py` | Re-authored FastViTHD vision encoder module. Imported by `export_fastvlm.py`. |
| `fastvlm_projector.py` | mlp2x_gelu projector module. Imported by `export_fastvlm.py`. |
| `quantization.py` | Compression preset system (`MACOS_NAMED_PRESETS`: `4bit`, `8bit`). `load_compression_config()`, `apply_quantization_from_config()`. Supports YAML recipes via `QuantizerConfig.from_dict()`. |

### Inspection and verification

| Script | Purpose |
|--------|---------|
| `inspect_aimodel.py` | Inspect any CoreAI VLM bundle directory or individual `.aimodel` file. Reports inputs, outputs, state names, KV cache behavior, tokenizer. Works on FastVLM and Qwen3-VL bundles. |
| `verify_vision_encoder.py` | **Layer 1:** HF FastVLMVisionEncoder vs re-authored PyTorch encoder PSNR. Verifies the re-authoring is correct. |
| `verify_projector.py` | **Layer 1:** HF projector vs re-authored PyTorch projector PSNR. |
| `verify_decoder.py` | **Layer 1:** Four-phase decoder verification — architecture correctness (fp32 vs HF Qwen2), FP16 fidelity (realistic fixture inputs), KV cache correctness, and compression quality (9-image corpus, behavioral metrics). See [docs/VERIFICATION.md](docs/VERIFICATION.md). |
| `verify_runtime.py` | **Layer 2:** CoreAI compiled model vs PyTorch reference PSNR across all 6 pipeline stages. Use `--image` for meaningful vision PSNR. Run on macOS 26.5 (see Known Issues). |

### Test assets

| Script | Purpose |
|--------|---------|
| `fetch_test_images.py` | Download 9 public domain benchmark images to `test_assets/images/`. Uses Wikimedia Commons API. Run once after cloning. |
| `build_fixtures.py` | Pre-build and cache decoder fixtures for `verify_decoder.py` Phase 2 and Phase 4. Run once per variant after downloading weights and images. |
| `generate_test_images.py` | Generate synthetic test images (tall_narrow_circle.png, wide_short_square.png) for preprocessing strategy verification. No external downloads needed. |
| `run_hf_fastvlm.py` | Run FastVLM from original HF weights for ground truth comparison against CoreAI export. Supports `--variant`, `--image`, `--prompt`, `--temperature`, `--device`. |
| `probe_vlm_config.py` | Probe any HF VLM config for native resolution and preprocessing metadata. Supports Qwen3-VL (2B/7B/32B/72B), FastVLM, and any HF VLM. |

### Diagnostics and discovery

| Script | Purpose |
|--------|---------|
| `discover_weights.py` | Dump weight shapes and dtypes from HF safetensors to `discovery/`. Run when adding a new variant. |
| `inspect_weights.py` | Human-readable PyTorch/MLX weight inspection. Complements `inspect_aimodel.py` (which inspects compiled CoreAI models). |
| `probe_activations.py` | Profile intermediate activation magnitudes in the vision encoder's network stages. Used to identify fp16 overflow risk at `network.8-10`. |
| `audit_weight_dtypes.py` | Exhaustive dtype/shape audit across HF and Apple MLX checkpoints. Used to verify that `quantization.py` matches Apple's exact quantization scope. |
| `compare_weights.py` | Compare Apple's MLX quantized weights against HF bf16 source by dequantizing and measuring PSNR. Requires Apple's MLX checkpoints locally. |

---

## Export Flags

### `--variant`
Model size. Affects decoder architecture and weight file.
- `0.5b` — 24 layers, hidden=896, 2 KV heads
- `1.5b` — 28 layers, hidden=1536, 2 KV heads
- `7b` — 32 layers, hidden=3584, 8 KV heads

### `--quantize`
Post-export quantization of the decoder. Vision encoder and embed are always fp16.
- *(none)* — fp16, highest quality
- `int8` — ~2× smaller, minimal quality loss, recommended for 1.5b
- `int4` — ~4× smaller, some quality loss, recommended for 7b

### `--kv-cache`
KV cache allocation strategy. Both modes use `--max-context-length` as the hard
ceiling — the compiled graph rejects inputs beyond it.
- `static` *(default)* — pre-allocates `max_ctx` tokens of Metal memory upfront.
  `StaticKVCache` in Swift. Matches Apple's `vlm/export.py`.
- `dynamic` — starts at 256 tokens, grows 2× as needed up to `max_ctx`.
  `GrowingKVCache` in Swift. Lower initial memory footprint — useful when
  exporting with a large `max_ctx` but expecting mostly short conversations.

FastVLM uses only 12 KB/token (KV), so static `max_ctx=32768` (393 MB) is
feasible on device. Compare: Qwen3-VL-2B uses 115 KB/token (3.6 GB at 32768,
requiring dynamic).

### `--max-context-length`
Maximum context length in tokens. Hard ceiling in both static and dynamic modes.
Must not exceed `max_position_embeddings` from the model config (32768 for all
FastVLM variants). Default: 4096.

---

## Architecture

### VLM Inference Flow

```
pixel_values [1, 3, 1024, 1024]
      ↓ vision.aimodel::encode_image
image_features [1, 256, 3072]
      ↓ vision.aimodel::project
projected_features [1, 256, hidden]   ← 256 image tokens in LM space (hidden=896/1536/3584 by variant)
      ↓
all_token_ids [1, 256+N]              ← 256 <image> placeholders + N text tokens
      ↓ embed.aimodel::main
embeddings [1, 256+N, 896]
      ↓ scatter-merge (CoreAISequentialVLMEngine)
merged_inputs_embeds [1, 256+N, 896]  ← image positions replaced with projected_features
      ↓ fastvlm-{variant}.aimodel::main (+ stateful KV cache)
logits [1, 256+N, 151936]             → sample next token
      ↓ repeat for each decode step
```

### Key Design Decisions

**`mutable_slice_update` for KV cache:** The only pattern supported by
`remove_functionalization` in `coreai-models`. `slice_scatter` does not work for
export (it doesn't create `AutoFunctionalized` nodes). Imported directly from
`coreai_models.primitives.macos.cache.KVCache`.

**`inputs_embeds` not `input_ids`:** The decoder takes pre-computed embeddings.
`embed_tokens` is a separate model (`embed.aimodel`) so `CoreAISequentialVLMEngine`
can scatter-merge image features before calling the decoder.

**State names:** Python export uses `k_cache`/`v_cache`. The `coreai-torch`
compiler renames these to `keyCache`/`valueCache` (camelCase) in the compiled
model for Swift compatibility.

**Image normalization:** FastVLM uses no normalization (mean=0, std=1). Unlike
Qwen3-VL which uses ImageNet stats, FastVLM's vision tower was trained without
normalization.

**Image preprocessing strategy:** FastVLM's `CLIPImageProcessor` uses shortest-edge
resize + center crop to 1024×1024 — not stretch resize. The exported bundle declares
`"image_strategy": "center_crop"` in `metadata.json`, which `CoreAISequentialVLMEngine`
reads to select the correct resize algorithm. Apple's `coreai-models` supports three
strategies: `stretch` (default), `center_crop`, and `pad` — merged in
[apple/coreai-models #108](https://github.com/apple/coreai-models/pull/108).

**Performance (M4 Pro, GPU path):** ~97ms TTFT, 3,901 tok/sec prompt processing,
113 tok/sec generation. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for full benchmarks.

---

## Documentation

| File | Contents |
|------|---------|
| `docs/PERFORMANCE.md` | Benchmark results — throughput, TTFT, memory, quality |
| `docs/psnr_results.md` | PSNR verification results (PyTorch and CoreAI runtime) |
| `docs/FASTVLM_ARCHITECTURE.md` | Architecture deep-dive — components, quantization, export gotchas, image preprocessing |
| `docs/FASTVLM_SWIFT_INTEGRATION.md` | Swift integration guide — llm-runner, CoreAISequentialVLMEngine, custom app |
| `docs/STATUS.md` | Current project status, known issues, pending items |

---

## Known Issues

### MPSGraph crash on macOS 27 beta

Python runtime verification (`verify_runtime.py` and `asset.executable()`) crashes
on macOS 27 beta:

```
MPSGraphExecutable.mm:4442: failed assertion 'Incompatible shape for parameter at index 0'
```

This affects Apple's own Qwen3-VL export identically — it is an OS/platform bug,
not a model or export issue. The same model and script pass on macOS 26.5.

**Workaround:**
- Run `verify_runtime.py` on macOS 26.5
- For macOS 27 beta, use `xcrun coreai-build compile` (ahead-of-time compilation)
  and verify via Xcode performance tests or Swift app

A Feedback Assistant report has been filed against Apple.

### `coreai-models` PyPI version

The PyPI version of `coreai-models==0.1.0` has an incorrect `Python>=3.14`
constraint and cannot be installed on Python 3.11. Always install from the
GitHub source (see [Installation](#installation)).

### Vision encoder fp32 on ANE

`vision.aimodel` accepts `pixel_values` as `float32` (the natural dtype for image
data). On some platforms, `AIModel.load()` triggers ANE compilation which rejects
fp32 inputs. `inspect_aimodel.py` handles this gracefully. The structural export
is correct — the fp32→fp16 cast happens as the first op inside the model.

---

## Relationship to Apple's coreai-models

This repo follows Apple's authoritative VLM export recipe from
[`coreai-models/vlm/export.py`](https://github.com/apple/coreai-models) exactly,
with FastVLM-specific additions:

- Re-authored `FastVLMVisionEncoder` (FastViTHD via `trust_remote_code`, not in HF
  `transformers`)
- Re-authored `FastVLMDecoder` (Qwen2, matching
  `Qwen3VLForCausalLMEmbeddings.forward()`)
- `<image>` special token added to Qwen2 tokenizer (ID 151646)
- `--kv-cache`, `--quantize`, `--max-context-length` export flags
- Generic `inspect_aimodel.py` (works on any CoreAI VLM bundle)

Adding FastVLM as a first-class recipe in Apple's `coreai-models` is a planned
future contribution.

### Issues filed against apple/coreai-models

| Issue | Status | Description |
|-------|--------|-------------|
| [#96](https://github.com/apple/coreai-models/issues/96) | 🔲 Open | PyPI wheel declares incorrect `Python>=3.14` constraint |
| [#100](https://github.com/apple/coreai-models/issues/100) | ✅ Closed — merged as [#108](https://github.com/apple/coreai-models/pull/108) | `CoreAISequentialVLMEngine` image preprocessing strategy — `center_crop`, `pad`, `stretch` now supported in upstream |
