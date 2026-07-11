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
- Apple Silicon Mac (M1 or later)
- macOS 26+ (macOS 27 beta has an MPSGraph bug affecting Python runtime
  verification on M4 Pro — see [Known Issues](#known-issues))

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

### 2. Clone Apple's coreai-models

`coreai-models` is required for the export pipeline but is not available on
PyPI with a correct Python version constraint. Install from source:

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
> and cannot be used.

### 5. Download FastVLM weights

```bash
hf download apple/FastVLM --local-dir weights/fastvlm-0.5b
# Repeat for 1.5b and 7b as needed
```

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

### Verify numerical correctness

```bash
python scripts/verify_runtime.py --variant 0.5b
python scripts/verify_runtime.py --variant 0.5b --decode-steps 5
```

Expected:
```
  ✓ embed_tokens PSNR:   inf dB  (PASS)
  ✓ scatter_merge PSNR:  inf dB  (PASS)
  ✓ decode prefill PSNR: XX.X dB (PASS > 40 dB)
  ✓ decode step 1/3:     XX.X dB (PASS > 40 dB)
[PASS] All stages match PyTorch reference (> 40 dB PSNR).
```

### Compile for ANE (ahead-of-time)

```bash
xcrun coreai-build compile \
  --preferred-compute neural-engine \
  exports/fastvlm-0.5b.vlmasset/fastvlm-0.5b.aimodel
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `export_fastvlm.py` | Main export script. Produces the full `.vlmasset` bundle. |
| `inspect_aimodel.py` | Inspect any CoreAI VLM bundle or `.aimodel` file. Works on FastVLM and Qwen3-VL. |
| `verify_runtime.py` | End-to-end PSNR verification against PyTorch reference. |
| `verify_decoder.py` | PyTorch-only decoder verification (no CoreAI runtime needed). |
| `fastvlm_decoder.py` | Re-authored Qwen2 decoder for CoreAI export. |
| `fastvlm_vision_encoder.py` | Re-authored FastViTHD vision encoder for CoreAI export. |
| `fastvlm_projector.py` | mlp2x_gelu projector for CoreAI export. |

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
projected_features [1, 256, 896]      ← 256 image tokens in LM space
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

---

## Documentation

| File | Contents |
|------|---------|
| `docs/FASTVLM_ARCHITECTURE.md` | FastVLM architecture deep-dive |
| `docs/FASTVLM_MULTIMODAL_PIPELINE.md` | Full multimodal pipeline documentation |
| `docs/FASTVLM_SWIFT_INTEGRATION.md` | Swift app integration guide |
| `docs/STATUS.md` | Current project status and pending items |

---

## Known Issues

### MPSGraph crash on M4 Pro / macOS 27 beta

Python runtime verification (`verify_runtime.py` and `asset.executable()`) crashes
on M4 Pro with macOS 27 beta:

```
MPSGraphExecutable.mm:4442: failed assertion 'Incompatible shape for parameter at index 0'
```

This affects Apple's own Qwen3-VL export identically — it is an OS/platform bug,
not a model or export issue. The same model and script pass on M1 Pro / macOS 26.

**Workaround:**
- Run `verify_runtime.py` on M1 Pro or another non-M4 machine
- For M4 Pro, use `xcrun coreai-build compile` (ahead-of-time compilation)
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
