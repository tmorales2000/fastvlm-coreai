# FastVLM Swift Integration

**Last updated:** July 14, 2026

This document covers Swift integration of the FastVLM CoreAI export via
`CoreAISequentialVLMEngine` from Apple's `CoreAILanguageModels` Swift package.

---

## Quick Start: llm-runner

The fastest way to verify and run the exported bundle is Apple's `llm-runner`
tool, which ships in `coreai-models` and uses `CoreAISequentialVLMEngine` internally.

> **Important:** Use the fork, not Apple's upstream repo — the fork includes the
> image preprocessing fix required for correct FastVLM output.

### Build

```bash
cd ~/git/tmorales2000/coreai-models
swift build --product llm-runner
# Binary: .build/out/Products/Debug/llm-runner
```

> If `swift build --product` doesn't produce a binary, run
> `swift package clean` first then retry.

### Usage

```bash
LLM_RUNNER=~/git/tmorales2000/coreai-models/.build/out/Products/Debug/llm-runner
BUNDLE=~/git/tmorales2000/fastvlm-coreai/exports/fastvlm-0.5b.vlmasset

# Text only
$LLM_RUNNER --model $BUNDLE \
  --prompt "What is the capital of France?" \
  --max-tokens 50

# Image + text (VLM)
$LLM_RUNNER --model $BUNDLE \
  --image test_assets/images/earthrise.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 300 --temperature 0

# Verbose timing breakdown (TTFT, throughput, memory, compute allocation)
$LLM_RUNNER --model $BUNDLE \
  --image test_assets/images/earthrise.jpg \
  --prompt "Describe this image." \
  --max-tokens 300 --temperature 0 --verbose
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to `.vlmasset` bundle |
| `--image` | none | Path to image file (triggers VLM mode) |
| `--prompt` | required | Text prompt |
| `--max-tokens` | 50 | Maximum **new** tokens to generate |
| `--temperature` | 0.7 | Sampling temperature. Use 0 for deterministic output. |
| `--verbose` | false | Detailed timing, compute allocation, memory |

> **`--max-tokens` is not the KV cache limit.** It is the generation limit.
> Set to 300-500 for typical use. The KV cache ceiling is `--max-context-length`
> set at export time (default 4096).

---

## How CoreAISequentialVLMEngine Works

`CoreAISequentialVLMEngine` orchestrates the full VLM pipeline given a
`.vlmasset` bundle. It reads `metadata.json` to understand the bundle structure:

```
llm-runner / your Swift app
  ↓
CoreAISequentialVLMEngine (CoreAILanguageModels)
  ↓
LanguageBundle (reads metadata.json)
  ├── vision.aimodel  → PreparedModelAsset → encode_image + project
  ├── embed.aimodel   → PreparedModelAsset → main (embed_tokens)
  └── {variant}.aimodel → PreparedModelAsset → main (stateful KV decoder)
```

**Pipeline per image+text request:**

1. `encode_image(pixel_values)` → `image_features [1, 256, 3072]`
2. `project(image_features)` → `projected_features [1, 256, hidden]`
3. Tokenize prompt → insert 256 × `image_token_id` placeholders + text tokens
4. `embed(all_token_ids)` → `embeddings [1, L, hidden]`
5. Scatter-merge: replace `image_token_id` positions with `projected_features`
6. `decode(merged_embeds, position_ids)` → `logits` → sample next token
7. Decode loop: single-token steps with persistent KV cache state

**Key decoder contract:**
The decoder (`fastvlm-{variant}.aimodel`) takes `inputs_embeds` (pre-computed
embeddings) not `input_ids`. `embed.aimodel` is a separate model so the engine
can scatter-merge image features before calling the decoder. This is why the
bundle has three separate `.aimodel` files rather than one.

**Image preprocessing (as of fork fix):**

The engine reads `"preprocessing"` from `metadata.json` and selects:
- `"center_crop"` → `ImagePreprocessor.preprocessCHWCenterCrop()` — shortest-edge
  resize then center crop to `imageSize × imageSize`. Correct for FastVLM.
- `"stretch"` → `ImagePreprocessor.preprocessCHW()` — legacy stretch resize (default
  for models without the field).

FastVLM bundles exported by this repo declare `"preprocessing": "center_crop"`.

---

## Performance (FastVLM 0.5B fp16, M4 Pro, macOS 27 beta)

| Metric | Value |
|--------|-------|
| Model load (warm) | 250ms |
| Warmup (one-time JIT) | ~1,600ms |
| Approximate TTFT | ~97ms |
| Prompt throughput | 3,901 tok/sec |
| Generation throughput | 113 tok/sec |
| Memory (current) | 2,761 MB |
| Memory (peak) | 3,448 MB |
| Compute path | GPU via MPSGraph (ANE pending) |

See [PERFORMANCE.md](PERFORMANCE.md) for full benchmarks across all variants.

---

## Compute Allocation (from --verbose output)

| Component | Compute | Notes |
|-----------|---------|-------|
| vision.aimodel | GPU dynamic | `Unknown model structure` — FastViTHD not a recognized architecture |
| embed.aimodel | GPU dynamic | Small model |
| fastvlm-{variant}.aimodel | GPU dynamic | Standard transformer ops |

The `Unknown model structure` warning means the engine defaults to GPU for the
vision tower rather than potentially using ANE. The decoder likely benefits from
ANE acceleration via MPSGraph. Investigating ANE for FastViTHD is a pending task.

---

## KV Cache

The engine auto-selects KV cache type based on the compiled model's state shape:
- `seq_dim = 4096` (static) → `StaticKVCache` — pre-allocated, zero runtime allocation
- `seq_dim = -1` (dynamic) → `GrowingKVCache` — starts small, grows 2× as needed

Our default export (`--kv-cache static`) uses `StaticKVCache`. Re-export with
`--kv-cache dynamic` for `GrowingKVCache`:

```bash
python scripts/export_fastvlm.py --variant 0.5b --kv-cache dynamic --overwrite
```

KV cache memory for static at 4096 tokens:

| Variant | KV cache (k+v) |
|---------|---------------|
| 0.5B fp16 | ~24 MB |
| 1.5B fp16 | ~112 MB |
| 7B fp16 | ~448 MB |

FastVLM is memory-efficient — static 4096 is practical even on iPhone.

---

## Custom Swift App

To use `CoreAISequentialVLMEngine` in your own Swift app:

### Package.swift

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FastVLMApp",
    platforms: [.macOS(.v15)],
    dependencies: [
        // Use fork until preprocessing fix is merged upstream
        .package(path: "/path/to/tmorales2000/coreai-models"),
    ],
    targets: [
        .executableTarget(
            name: "FastVLMApp",
            dependencies: [
                .product(name: "CoreAILanguageModels",
                         package: "coreai-models"),
            ]
        ),
    ]
)
```

### Basic inference (refer to LLMRunnerMain.swift)

The most complete usage example is `llm-runner`'s `LLMRunnerMain.swift` in the
`coreai-models` package. It handles bundle loading, image preprocessing, tokenization,
chat template application, generation loop, and streaming output. Use it as the
primary reference for building your own Swift integration.

Key types:
- `LanguageBundle` — reads `metadata.json`, discovers the three `.aimodel` files
- `PreparedModel.prepare(bundle:)` — loads and prepares all three models
- `CoreAISequentialVLMEngine` — orchestrates the full VLM pipeline

---

## Chat Template

FastVLM's decoder (Qwen2) uses the ChatML format stored in `tokenizer_config.json`:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
{your prompt}<|im_end|>
<|im_start|>assistant
```

Stop token: `<|im_end|>` (token ID varies by variant).

`llm-runner` applies this template automatically via `tokenizer.apply_chat_template()`.
In a custom Swift app, ensure the tokenizer processes the chat template from the
bundle's `tokenizer/tokenizer_config.json`.

---

## Troubleshooting

**"Unknown model structure, defaulting to GPU dynamic"**
Expected for FastViTHD — not a recognized architecture. Model runs correctly on GPU.

**Generation cuts off mid-sentence**
`--max-tokens` default is 50. Set to 300-500 for complete responses.

**MPSGraph crash on macOS 27 beta**
Python runtime verification (`verify_runtime.py`) crashes on macOS 27 beta.
`llm-runner` (Swift, production path) works correctly on macOS 27 beta.
See Known Issues in [STATUS.md](STATUS.md).

**Stretched/distorted image descriptions for non-square images**
Ensure you are using the fork (`tmorales2000/coreai-models`,
`fix/vlm-image-preprocessing-strategy` branch) not Apple's upstream.
The upstream `CoreAISequentialVLMEngine` stretch-resizes all images.
See [apple/coreai-models #100](https://github.com/apple/coreai-models/issues/100).
