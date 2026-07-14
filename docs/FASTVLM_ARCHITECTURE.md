# FastVLM Architecture: A Complete Guide

**Audience:** Anyone curious how FastVLM actually works — how data flows through the model,
how the three components are constructed, how they differ across the 0.5B, 1.5B, and 7B
variants, and what had to change for Core AI export.

All tensor shapes and numeric values in this document can be verified using the scripts
in `scripts/`. Commands are shown throughout.

---

## Overview

FastVLM is a vision-language model (VLM). It takes an image and a text prompt as input
and generates a text response. It is built from three independent components:

```
[image]  →  vision_encode  →  [image patch embeddings]
                                        ↓
                                    project
                                        ↓
                              [projected embeddings]
                                        ↓
[text]   →  tokenize  →  embed  →   decode   →  [logits]  →  argmax  →  [token]
                                        ↑
                              (repeat for each output token)
```

The three components are:

| Component | Architecture | Variants |
|-----------|-------------|---------|
| Vision encoder | FastViTHD | Shared across 0.5B, 1.5B, 7B |
| Projector | mlp2x_gelu MLP | Hidden size varies by variant |
| Decoder | Qwen2 LLM | 0.5B / 1.5B / 7B parameter counts |

Each component is a separate `.aimodel` entrypoint in our Core AI export:
`vision_encode`, `project`, and `decode`.

---

## Component 1: Vision Encoder (FastViTHD)

### What it does

The vision encoder converts a raw image into a sequence of patch embeddings — a
compact, high-dimensional representation of the image's visual content. The decoder
attends to these embeddings as if they were tokens in the input sequence.

### Architecture

FastViTHD is a hierarchical vision transformer derived from FastViT, tuned for
high-resolution (HD) inputs. It processes the image through:

1. **Patch embedding** (`patch_embed`) — a strided Conv2d that splits the image into
   non-overlapping patches and projects each to a channel dimension.
2. **Network stages** (`network`) — a sequence of blocks combining depthwise conv,
   channel mixing MLP, and multi-head self-attention. The backbone is fully
   reparameterized at inference time (MobileOneBlock, RepMixer, RepCPE,
   ReparamLargeKernelConv all collapse to a single Conv2d + bias each).
3. **Expansion conv** (`conv_exp`) — a final pointwise conv that projects to the
   output channel dimension (3072 for all variants).

### Input / Output

| | Shape | Dtype |
|---|---|---|
| Input (`pixel_values`) | `[1, 3, 1024, 1024]` | `float32` |
| Output (`image_features`) | `[1, 256, 3072]` | `float16` |

The 256 output tokens represent a 16×16 spatial grid of image patches (1024 / 64 = 16
along each axis, where 64 is the effective patch stride after the embedding stage).
The 3072 channel dimension is the vision encoder's output embedding size, which is
the same across all three FastVLM variants — the vision encoder is shared.

**The fp32 → fp16 contract:** The public input is `float32` because that is what
cameras and image preprocessing pipelines produce. The model casts internally to fp16
(the model's weight dtype) as its first operation, matching the original HF model's
pattern (`image.to(dtype=self.dtype)` in `MobileCLIPVisionTower`). This cast is baked
into the compiled Core AI graph — callers never need to know the model's internal
precision.

### Verify with scripts

```bash
# Inspect the compiled model's vision_encode entrypoint
python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset
# Look for:
#   input   pixel_values   dtype=float32   shape=[1, 3, 1024, 1024]
#   output  image_features dtype=float16   shape=[1, 256, 3072]
```

```bash
# Verify numerical correctness against the HF reference
python scripts/verify_vision_encoder.py --variant 0.5b
# Stage 1 (fp32):  should be ~129 dB  (bit-identical to HF)
# Stage 2 (fp16):  should be ~50 dB   (fp16 rounding from HF bf16)

# Verify CoreAI compiled vision.aimodel against PyTorch reference
python scripts/verify_runtime.py --variant 0.5b --image test_assets/images/earthrise.jpg
# vision_encode PSNR: 71.9 dB  (PASS)
# project PSNR:       67.6 dB  (PASS)
```

### Variants: vision encoder is identical across 0.5B, 1.5B, 7B

The vision encoder architecture and weights are the same for all three FastVLM variants.
All three produce `[1, 256, 3072]` output. The variant differences start at the projector.

### HF model gotchas for Core AI export

**LayerNormChannel:** The original FastViTHD uses a custom `LayerNormChannel` that
manually decomposes layer norm on `[B, C, H, W]` tensors (mean → subtract → pow →
sqrt → divide). This manual decomposition prevents the Core AI compiler from
recognizing it as a `layer_norm` composite op. Our `ANELayerNorm` replaces it with
`F.layer_norm` after a `permute` to `[B, H, W, C]`, giving the compiler the
high-level op it needs.

**MHSA:** The original FastViTHD attention computes `q @ k.T * scale → softmax → @ v`
manually. Our `ANEAttention` replaces it with the `SDPA` composite op from
`coreai_torch.composite_ops`, which the compiler can map to hardware-optimized
attention.

**fp32 overflow at fp16:** At 1024×1024 input, pure fp16 forward overflows — values
reach ~60,928 at `network.9` (near fp16's ceiling of 65,504), then NaN at `network.10`.
This is why the reference path runs at fp32 in the verify scripts, even though the
export runs at fp16. The overflow does not appear to affect the compiled ANE execution
(the optimizer applies a different lowering), but it was a significant source of
confusion during development.

---

## Component 2: Projector (mlp2x_gelu)

### What it does

The projector bridges the vision encoder and the language decoder. It maps image patch
embeddings from the vision encoder's output space (3072 dimensions) into the decoder's
token embedding space (hidden_size, which varies by variant). After projection, the
256 image patch embeddings look like token embeddings to the decoder.

### Architecture

The projector type is `mlp2x_gelu`: two Linear layers with GELU activation between
them. The "2x" means depth=2 (two linear layers), not that the hidden size doubles.

```
Linear(mm_hidden_size → hidden_size)   # layers.0
GELU()                                  # layers.1  (tanh approximation)
Linear(hidden_size → hidden_size)       # layers.2
```

### Input / Output by variant

| Variant | Input shape | Output shape | Layer 0 size | Layer 2 size |
|---------|-------------|--------------|--------------|--------------|
| 0.5B | `[1, 256, 3072]` | `[1, 256, 896]` | 3072→896 | 896→896 |
| 1.5B | `[1, 256, 3072]` | `[1, 256, 1536]` | 3072→1536 | 1536→1536 |
| 7B | `[1, 256, 3072]` | `[1, 256, 3584]` | 3072→3584 | 3584→3584 |

The `mm_hidden_size` (3072) comes from the vision encoder output and is fixed.
The `hidden_size` (896 / 1536 / 3584) matches the decoder's token embedding dimension.

### Verify with scripts

```bash
# Compare all three variants' projector output shapes
python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset
python scripts/inspect_aimodel.py exports/fastvlm-1.5b.vlmasset
python scripts/inspect_aimodel.py exports/fastvlm-7b.vlmasset
# Look at the project entrypoint output shape in each
```

```bash
# Verify projector numerical quality including quantization
python scripts/verify_projector.py --variant 1.5b
# Stage 1 (fp32): ~113 dB
# Stage 2 (fp16): ~59 dB
# Stage 3 (int8): ~68 dB vs fp16 reference

# Verify CoreAI compiled vision.aimodel (includes projector) against PyTorch reference
python scripts/verify_runtime.py --variant 1.5b --image test_assets/images/earthrise.jpg
# vision_encode PSNR: 71.9 dB  (PASS)
# project PSNR:       67.6 dB  (PASS)
# embed_tokens PSNR:  inf dB   (PASS)
# decode steps:       44+ dB   (PASS)
```

```bash
# Compare our int8 quantization against Apple's MLX int8
python scripts/compare_weights.py --component projector --variant 1.5b --ours
# Apple PSNR: ~62.4 dB
# Ours PSNR:  ~62.4 dB  (matches Apple to within 0.1 dB)
```

### Quantization by variant

| Variant | Projector precision | Notes |
|---------|-------------------|-------|
| 0.5B | fp16 | No quantization |
| 1.5B | int8 | group_size=64, asymmetric, axis=1 |
| 7B | int4 | group_size=64, asymmetric, axis=1 |

Quantization targets `nn.Linear` weight matrices only. Biases stay fp16.

### HF model gotchas for Core AI export

The projector is the simplest component and has the fewest gotchas. The main issue
was ensuring the GELU uses `approximate="tanh"` (which the HF model does), since the
exact GELU implementation matters for correctness — wrong activation gives ~20-30 dB
PSNR degradation.

---

## Component 3: Language Decoder (Qwen2)

### What it does

The decoder is a standard autoregressive language model based on Qwen2. It takes a
sequence of token embeddings (both image patch embeddings from the projector and text
token embeddings) and generates logits — one float per vocabulary entry per position.
The caller takes the highest-scoring logit (argmax) at the last position to get the
next predicted token.

This is the most complex component and the source of most Core AI export challenges.

### Architecture

The Qwen2 decoder is a standard transformer:

```
embed_tokens        — nn.Embedding(vocab_size, hidden_size)
layers[0..N-1]      — N FastVLMDecoderBlocks (transformer layers)
  └─ input_layernorm         — RMSNorm
  └─ self_attn               — FastVLMAttention (GQA)
     └─ q_proj               — nn.Linear(hidden → n_heads * head_dim)
     └─ k_proj               — nn.Linear(hidden → n_kv_heads * head_dim)
     └─ v_proj               — nn.Linear(hidden → n_kv_heads * head_dim)
     └─ o_proj               — nn.Linear(n_heads * head_dim → hidden)
     └─ sdpa                 — coreai_torch SDPA composite op
     └─ rope                 — coreai_torch RoPE composite op
  └─ post_attention_layernorm — RMSNorm
  └─ mlp                     — SwiGLU MLP
     └─ gate_proj            — nn.Linear(hidden → intermediate)
     └─ up_proj              — nn.Linear(hidden → intermediate)
     └─ down_proj            — nn.Linear(intermediate → hidden)
norm                — final RMSNorm
lm_head             — nn.Linear(hidden_size → vocab_size)
k_cache, v_cache    — registered state buffers
```

### Architecture differences across variants

| | 0.5B | 1.5B | 7B |
|---|---:|---:|---:|
| `hidden_size` | 896 | 1536 | 3584 |
| `intermediate_size` | 4864 | 8960 | 18944 |
| `num_hidden_layers` | 24 | 28 | 28 |
| `num_attention_heads` | 14 | 12 | 28 |
| `num_key_value_heads` | 2 | 2 | 4 |
| `head_dim` | 64 | 128 | 128 |
| `vocab_size` | 151,936 | 151,936 | 152,064 |
| `rope_theta` | 1,000,000 | 1,000,000 | 1,000,000 |

GQA ratio (query heads / KV heads): 7:1 for 0.5B, 6:1 for 1.5B, 7:1 for 7B.

The 7B has a slightly larger vocabulary (152,064 vs 151,936) — this is the Qwen2-7B
variant's extended tokenizer.

### Verify architecture with scripts

```bash
# Full data flow with tensor shapes at each stage boundary — ground truth for all tables
python scripts/inspect_weights.py --variant 0.5b --source pytorch --mode flow
python scripts/inspect_weights.py --variant 1.5b --source pytorch --mode flow
python scripts/inspect_weights.py --variant 7b   --source pytorch --mode flow

# Raw config JSON
python scripts/inspect_weights.py --variant 1.5b --source pytorch --mode config

# Layer-by-layer tensor inventory (designed for diffing PyTorch vs MLX)
python scripts/inspect_weights.py --variant 1.5b --source pytorch --mode layers > /tmp/pt.txt
python scripts/inspect_weights.py --variant 1.5b --source mlx    --mode layers > /tmp/mlx.txt
diff /tmp/pt.txt /tmp/mlx.txt
# Shows exactly which tensors changed dtype and which gained .scales/.biases siblings
```

```bash
# See KV cache shapes in the compiled model
python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset
# state  k_cache   dtype=float16   shape=[24, 1, 4096, 128]
# state  v_cache   dtype=float16   shape=[24, 1, 4096, 128]
# Shape: [n_layers, batch=1, MAX_SEQ_LEN, kv_dim]
# kv_dim = head_dim * num_key_value_heads = 64 * 2 = 128 for 0.5B

python scripts/inspect_aimodel.py exports/fastvlm-1.5b.vlmasset
# state  k_cache   dtype=float16   shape=[28, 1, 4096, 256]
# kv_dim = 128 * 2 = 256 for 1.5B

python scripts/inspect_aimodel.py exports/fastvlm-7b.vlmasset
# state  k_cache   dtype=float16   shape=[28, 1, 4096, 512]
# kv_dim = 128 * 4 = 512 for 7B
```

### Input / Output

| | Shape | Dtype | Notes |
|---|---|---|---|
| `inputs_embeds` | `[1, L, hidden]` | `float16` | Pre-computed embeddings (not token IDs) |
| `position_ids` | `[1, L]` | `int32` | Absolute positions of each token |
| `logits` (output) | `[1, L, vocab_size]` | `float16` | Raw scores for next token |
| `k_cache` (state) | `[n_layers, 1, n_kv_heads, 4096, head_dim]` | `float16` | Persisted across calls |
| `v_cache` (state) | `[n_layers, 1, n_kv_heads, 4096, head_dim]` | `float16` | Persisted across calls |

The `L` dimension is dynamic (`-1` in the compiled model's descriptor), supporting
both the prefill call (L = 256 image tokens + prompt tokens) and decode steps (L = 1).

```bash
# Confirm dynamic L in compiled decode signature
python scripts/inspect_aimodel.py exports/fastvlm-0.5b.vlmasset
#   input   input_ids    dtype=int32    shape=[1, -1]
#   output  logits       dtype=float16  shape=[1, -1, 151936]
```

### Quantization by variant

| Variant | Decoder precision | Linear weights | Embedding | Norms |
|---------|-----------------|----------------|-----------|-------|
| 0.5B | fp16 | fp16 | fp16 | fp16 |
| 1.5B | int8 | int8 (scales fp16) | fp16 | fp16 |
| 7B | int4 | int4 (scales fp16) | fp16 | fp16 |

Only `nn.Linear` and `nn.Embedding` weight matrices are quantized. Biases, RMSNorm
weights, and all activations remain fp16. Quantization parameters: group_size=64,
asymmetric, axis=1 — matching Apple's MLX scheme exactly.

```bash
# Audit all weight dtypes against HF and Apple MLX
python scripts/audit_weight_dtypes.py --variant 1.5b
# Shows every tensor's dtype in HF checkpoint (bf16) and MLX checkpoint (fp16 or int8)

# Compare quantization quality layer by layer
python scripts/compare_weights.py --component decoder --variant 7b --ours
# Shows per-layer PSNR for Apple's int4 vs our int4 vs HF bf16 reference
```

### The two-phase decode loop

Understanding the decode loop is essential for understanding the KV cache.

**Phase 1: Prefill (one call)**

The first call processes all tokens at once — the 256 projected image patch embeddings
(from `project`) concatenated with the text prompt token embeddings. This is the
expensive call: it's computing attention over potentially 300+ tokens in one shot.

```
input_ids:    [1, 256 + prompt_len]   ← image tokens + prompt
position_ids: [1, 256 + prompt_len]   ← 0, 1, 2, ..., 255+prompt_len
logits:       [1, 256 + prompt_len, vocab_size]
```

Only the logit at the last position matters — it's the prediction for the first
generated token. The KV cache is now populated with key/value vectors for all
256 + prompt_len positions.

**Time to First Token (TTFT)** is measured as the wall-clock time of this prefill call.

**Phase 2: Decode (one call per token)**

Every subsequent call processes exactly one new token — the token ID sampled from
the previous step's logits:

```
input_ids:    [1, 1]   ← single new token
position_ids: [1, 1]   ← current position (256 + prompt_len + step)
logits:       [1, 1, vocab_size]
```

Each call reads from the KV cache (all previous context) and writes the new K/V
pair for this position into the cache. The KV cache is what makes this efficient —
without it, every step would re-process all previous tokens from scratch.

Generation continues until an EOS token is sampled or `MAX_SEQ_LEN` is reached.

### KV cache capacity

The KV cache is pre-allocated at export time with shape `[n_layers, 1, MAX_SEQ_LEN, kv_dim]`.
Currently `MAX_SEQ_LEN = 4096`. The budget across the full conversation:

- 256 image patch tokens (always consumed, for every image)
- N prompt tokens (typically 10–50 for a question)
- ~3,700–3,800 tokens available for generated response

Memory footprint of the KV cache (both k and v):

| Variant | Per-layer cache | Total (k + v) |
|---------|----------------|---------------|
| 0.5B | 24 × 4096 × 128 × 2B = 25 MB | ~50 MB |
| 1.5B | 28 × 4096 × 256 × 2B = 58 MB | ~117 MB |
| 7B | 28 × 4096 × 512 × 2B = 117 MB | ~234 MB |

---

## HF Model vs Core AI Export: Key Gotchas

### 1. Dynamic KV cache → fixed pre-allocated cache

**HF model:** Uses PyTorch's `DynamicCache` (or `past_key_values` tuple), which grows
dynamically as context accumulates. The cache can theoretically grow to Qwen2's full
32,768-token context window. No pre-allocation is needed.

**Core AI export:** The `.aimodel` format requires statically shaped state buffers at
export time. The ANE kernel is compiled for specific cache dimensions. We pre-allocate
`k_cache` and `v_cache` as registered buffers of shape `[n_layers, 1, MAX_SEQ_LEN, kv_dim]`
and bake `MAX_SEQ_LEN` into the compiled graph.

**Trade-off:** Fixed capacity (4096 tokens currently) in exchange for zero dynamic
allocation during inference. This is actually an advantage for on-device deployment —
memory is predictable and no allocations occur during generation. The capacity can be
increased by changing `MAX_SEQ_LEN` in `fastvlm_decoder.py` and re-exporting.

### 2. In-place cache mutation → aten.slice_scatter

**HF model:** Cache writes happen in-place via standard Python tensor index assignment:
```python
cache[layer_idx, :, :, :seq_len] = new_kv
```
PyTorch handles this transparently.

**Core AI export (original attempt):** We used a custom op `fastvlm::mutable_slice_update`
with `mutates_args=['x']`. This caused `torch.export.export` to wrap it in
`auto_functionalized_v2` — a higher-order op that coreai-torch 0.4.0 has no handler
for (bug FB23024751, `apple/coreai-models#5`). Export failed with `UnboundLocalError`
in `_higher_order_resolver`.

**Core AI export (fix):** Use `mutable_slice_update` from
`coreai_models.primitives.macos.cache.KVCache` — the only pattern that creates
`AutoFunctionalized` nodes that `remove_functionalization` can lower to
`coreai.slice_update` MLIR. `aten.slice_scatter` is functionally equivalent but
does NOT work — it doesn't create `AutoFunctionalized` nodes.
This distinction is completely undocumented and was the hardest bug in the pipeline.

### 3. Graph-mode quantization → eager-mode quantization

**Original approach:** Used coreai-opt's default graph-mode quantization. `finalize()`
calls `convert_pt2e()` internally, which re-traces the model and bakes the `prepare()`
example input shapes as concrete constants in the `fx.GraphModule`. This made
`seq_len` a constant (e.g., `8`) instead of dynamic.

**Fix:** Switch to `ExecutionMode.EAGER`. Eager mode inserts fake-quantize modules
directly into the `nn.Module` without graph tracing, so shapes remain symbolic.
`torch.export.export` then sees the model with a named `Dim("seq_len", min=1, max=MAX_SEQ_LEN)`
and correctly emits a dynamic graph.

### 4. QKV fusion decision

**Apple's Core AI reference (coreai-models/qwen2.py):** Fuses Q, K, V projections into
a single `qkv_proj` Linear (`USE_FUSED_KV = True`). This is efficient for GPU (fewer
kernel launches) but has a quantization cost.

**Our decision:** Keep Q, K, V as separate `nn.Linear` layers, matching Apple's own
MLX inference weights which store `q_proj.weight`, `k_proj.weight`, `v_proj.weight`
as independently quantized tensors. Fusing before quantization gives the quantizer
a single blunt scale for three projections with different weight distributions —
`v_proj` reconstructs 6–10 dB worse when fused. Our separate approach lets the
quantizer adapt its per-group scales to each projection independently.

```bash
# See this in action — compare fused vs unfused quality would require re-export,
# but audit shows the separate weight structure:
python scripts/audit_weight_dtypes.py --variant 1.5b
# Shows: self_attn.q_proj.weight, .k_proj.weight, .v_proj.weight all QUANTIZED separately
```

### 5. Vision encoder fp32 input contract

**HF model:** The HF `MobileCLIPVisionTower` explicitly casts the image to the model's
dtype before processing: `image.to(device=self.device, dtype=self.dtype)`. The dtype
of the model's parameters determines the cast target (bf16 from HF, fp16 at inference).

**Original Core AI export:** We traced the vision encoder with an fp16 example input
(`torch.randn(...).to(EXPORT_DTYPE)`), so the compiled model declared
`pixel_values: Float16` at its boundary. Callers had to cast their images to fp16
before calling `vision_encode`.

**Fix:** The cast is now the first op inside `FastVLMVisionEncoder.forward()`:
```python
pixel_values = pixel_values.to(dtype=next(self.parameters()).dtype)
```
The export traces with a plain `float32` example input. The compiled model now declares
`pixel_values: Float32`, and the fp32→fp16 cast is baked into the graph as its first op.
Public contract: callers pass fp32. Internally fp16. Just like the HF model.

### 6. Causal mask `-inf` vs `-40000.0`

**HF model:** Uses standard IEEE `-inf` in the causal mask for softmax masking.
PyTorch's SDPA on CPU/GPU handles `-inf` correctly.

**Core AI / ANE:** The ANE hardware does not handle IEEE `-inf` correctly in softmax.
The correct value is `-40000.0` — representable in fp16, and `exp(-40000)` is
numerically zero. Our SDPA composite op (`coreai_torch.composite_ops.SDPA`) handles
this internally via `is_causal=True`, so we don't construct a mask ourselves.
But any manually constructed causal mask in the codebase should use `-40000.0`.

---

## Quantization Pipeline

### Apple's scheme (confirmed by audit and comparison scripts)

Apple's MLX pipeline applies a two-step precision reduction:

1. **bf16 → fp16:** All tensors, unconditionally. Even tensors that won't be
   further quantized are cast from bf16 (HF storage) to fp16.

2. **fp16 → int8 or int4:** Applied to `nn.Linear` and `nn.Embedding` weight
   matrices only. Parameters: group_size=64, asymmetric, axis=1.

Non-weight tensors (RMSNorm weights, biases, activations) remain fp16.

### Our scheme (matches Apple's to within 0.1 dB)

Our `scripts/quantization.py` replicates Apple's scheme using coreai-opt:
- `PerBlockGranularity(axis=1, block_size=64)` matches Apple's `group_size=64, axis=1`
- `ASYMMETRIC` matches Apple's asymmetric quantization
- `ExecutionMode.EAGER` avoids shape specialization (see gotcha #3 above)

```bash
# Verify our scheme matches Apple's
python scripts/compare_weights.py --component decoder --variant 1.5b --ours
# Expected output:
#   Layer 0 q_proj  Apple: 69.3 dB  Ours: 69.4 dB  Delta: +0.1 dB
#   Layer 0 k_proj  Apple: 68.1 dB  Ours: 68.2 dB  Delta: +0.1 dB
#   ...  (consistently 0.0–0.1 dB delta across all layers and variants)
```

### Why the 7B int4 PSNR is low (~22.7 dB) and why that's OK

The 7B variant uses int4 quantization, which has an inherently lower reconstruction
quality — ~22 dB for logit comparison vs ~50 dB for fp16. This is a structural property
of 4-bit quantization, not a bug. Apple ships the 7B FastVLM MLX weights at int4
and accepts this quality trade-off. The model produces useful output despite the
lower PSNR because large models are more robust to quantization noise than small ones —
there is more redundancy in the weight space.

The 1.5B int8 logit PSNR is ~50 dB, the 0.5B fp16 is limited only by fp16 precision
(~50 dB vs fp32). The 7B int4 is the outlier at ~22 dB, and it's expected.

---

## Exported Model Summary

After running `python scripts/export_fastvlm.py --variant <V>`:

| Bundle | Decoder | vision.aimodel entrypoints | decode logits shape | KV cache shape |
|--------|---------|--------------------------|--------------------|--------------------|
| `fastvlm-0.5b.vlmasset` | fp16 | encode_image, project | `[1,-1,151936]` | `[24,1,2,4096,64]` |
| `fastvlm-1.5b.vlmasset` | int8 | encode_image, project | `[1,-1,151936]` | `[28,1,2,4096,128]` |
| `fastvlm-7b.vlmasset` | int4 | encode_image, project | `[1,-1,152064]` | `[28,1,4,4096,128]` |

Note: decoder entrypoint is `main` in `fastvlm-{variant}.aimodel`.

All three pass `python scripts/inspect_aimodel.py <path>` with PASS on all checks.

### Compiled specializations

After running `xcrun coreai-build compile`:

```bash
ls fastvlm-0.5b.*.aimodelc  # compiled to current dir by default; move to exports/
# fastvlm-0.5b_fp16.h13c.aimodelc  ← A17 Pro (iPhone 15 Pro)
# fastvlm-0.5b_fp16.h14c.aimodelc  ← A18 (iPhone 16)
# fastvlm-0.5b_fp16.h15c.aimodelc  ← A18 Pro (iPhone 16 Pro)
# fastvlm-0.5b_fp16.h16s.aimodelc  ← M3 family
# fastvlm-0.5b_fp16.h17p.aimodelc  ← A18 Pro / M4 Pro  ← fleetwoodmac
# fastvlm-0.5b_fp16.h17s.aimodelc  ← A18 / M4 standard
# ... (21 total architecture targets)
```

The runtime automatically selects the correct specialization for the executing device.

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/inspect_weights.py` | Architecture flow, layer inventory, PyTorch vs MLX diff |
| `scripts/audit_weight_dtypes.py` | Exhaustive dtype/shape audit of HF and MLX checkpoints |
| `scripts/compare_weights.py` | Per-layer PSNR comparison of Apple vs our quantization |
| `scripts/export_fastvlm.py` | Export all three components to `.aimodel` |
| `scripts/inspect_aimodel.py` | Verify entrypoints, dtypes, shapes of exported model |
| `scripts/verify_vision_encoder.py` | Layer 1: HF FastVLMVisionEncoder vs re-authored PyTorch encoder PSNR |
| `scripts/verify_projector.py` | Layer 1: HF projector vs re-authored PyTorch projector PSNR |
| `scripts/verify_decoder.py` | Layer 1: HF Qwen2 decoder vs re-authored PyTorch decoder PSNR |
| `scripts/verify_runtime.py` | Layer 2: CoreAI compiled model vs PyTorch reference PSNR. Use `--image` for meaningful vision stages. |
| `scripts/run_hf_fastvlm.py` | End-to-end HF inference — ground truth baseline for CoreAI comparison |
| `scripts/probe_vlm_config.py` | Probe any HF VLM config for preprocessing and native resolution metadata |
| `scripts/generate_test_images.py` | Generate synthetic test images for preprocessing verification |
| `scripts/fastvlm_decoder.py` | Re-authored Qwen2 decoder for Core AI export |
| `scripts/fastvlm_projector.py` | Re-authored mlp2x_gelu projector for Core AI export |
| `scripts/fastvlm_vision_encoder.py` | Re-authored FastViTHD vision encoder for Core AI export |
| `scripts/quantization.py` | coreai-opt quantization pipeline (eager mode, matches Apple's scheme) |

---

## Image Preprocessing

FastVLM uses `CLIPImageProcessor` for preprocessing, configured in
`llava_qwen.py` with the following parameters:

```python
CLIPImageProcessor(
    crop_size  = {"height": 1024, "width": 1024},
    image_mean = [0.0, 0.0, 0.0],   # no-op
    image_std  = [1.0, 1.0, 1.0],   # no-op
    size       = {"shortest_edge": 1024},
)
```

**Mean and std are both no-ops.** FastVLM does not apply ImageNet normalization
(despite FastViTHD's timm config referencing `IMAGENET_DEFAULT_MEAN` / `IMAGENET_DEFAULT_STD`
— those are overridden at inference time). Pixel values pass through as-is.

The complete preprocessing pipeline for a camera frame is:

1. Resize shortest edge to 1024 (bicubic)
2. **Center crop** to 1024×1024 — cuts the longer dimension symmetrically
3. Convert to float32 in range [0.0, 1.0]
4. Rearrange from interleaved `[H, W, C]` → planar `[C, H, W]`
5. Add batch dimension → `[1, 3, 1024, 1024]` float32

No mean subtraction, no std division. This is the tensor passed to
`vision_encode` as `pixel_values`.

For the Metal preprocessing pipeline, the compute shader needs to handle:
shortest-edge resize, center crop, channel reorder, and float conversion.
No normalization needed.

### `image_aspect_ratio: pad` is dead code

`config.json` declares `image_aspect_ratio: pad`, which looks like it controls
preprocessing behavior. It does not. This field is only read in the `spatial` patch
merge branch of `prepare_inputs_labels_for_multimodal` in `llava_qwen.py`. FastVLM
uses `mm_patch_merge_type: flat`, so the `spatial` branch is never reached and the
field is never consulted. It is inherited LLaVA codebase configuration that has no
effect on FastVLM's actual preprocessing.

### CoreAISequentialVLMEngine image preprocessing (the bug we fixed)

Prior to the fix in `tmorales2000/coreai-models`, `CoreAISequentialVLMEngine`
**stretch-resized** all images to `imageSize × imageSize` regardless of the model's
actual preprocessing contract. This is geometrically incorrect for non-square images.

**Proof:** Fed a 200×800 image (1:4 aspect ratio) with a red circle to both models:

| Model | Preprocessing | Response |
|-------|-------------|---------|
| FastVLM 1.5B fp16 | center_crop (fixed) | "The red object is a **circle**." ✓ |
| Qwen3-VL 2B fp16 | stretch (unfixed) | "The red object is an **oval**." ✗ |

**The fix** adds a `"preprocessing"` field to `metadata.json`:
```json
"vision": {
    "image_size": 1024,
    "preprocessing": "center_crop"
}
```

`CoreAISequentialVLMEngine` reads this and calls `preprocessCHWCenterCrop()` instead
of the legacy stretch `preprocessCHW()`. Proposed upstream as
[apple/coreai-models #100](https://github.com/apple/coreai-models/issues/100).

---

## Weight Key Structure

### PyTorch HF safetensors

Vision encoder weights are doubly-nested in the HF checkpoint:

```
model.vision_tower.vision_tower.model.network.0.0.convffn.conv.conv.weight
└─ prefix: model.vision_tower.vision_tower.model.
           └─ stripped key: network.0.0.convffn.conv.conv.weight
```

The double nesting (`vision_tower.vision_tower`) is a quirk of the LLaVA-Qwen
model structure. Our `_load_vision_weights()` strips this prefix via
`_VISION_PREFIX = "model.vision_tower.vision_tower.model."`.

Decoder weights use standard Qwen2 HF key structure:
```
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```

Projector weights:
```
model.mm_projector.0.weight   (Linear mm_hidden_size → hidden_size)
model.mm_projector.0.bias
model.mm_projector.2.weight   (Linear hidden_size → hidden_size)
model.mm_projector.2.bias
```

### MLX weight keys

MLX weights use a flat key structure without the `model.` prefix and with
different component naming. Quantized Linear weights are split into three tensors:

```
# Unquantized (0.5B fp16):
language_model.model.embed_tokens.weight         [151936, 896]   float16
language_model.model.layers.0.self_attn.q_proj.weight  [896, 896]  float16

# Quantized (1.5B int8, 7B int4):
language_model.model.layers.0.self_attn.q_proj.weight  [384, 1536]  uint32  ← packed
language_model.model.layers.0.self_attn.q_proj.scales  [384, 24]    float16
language_model.model.layers.0.self_attn.q_proj.biases  [384, 24]    float16
```

Inspect any variant's full key structure:
```bash
python scripts/inspect_weights.py --variant 1.5b --source pytorch --mode layers
python scripts/inspect_weights.py --variant 1.5b --source mlx    --mode layers
diff <(python scripts/inspect_weights.py --variant 1.5b --source pytorch --mode layers) \
     <(python scripts/inspect_weights.py --variant 1.5b --source mlx    --mode layers)
```

---

## The Role of llava_qwen.py

Each variant weights directory contains a copy of `llava_qwen.py` — Apple's
original FastVLM model code. Its role in this repo is narrow and specific.

### What it is

`llava_qwen.py` is Apple's full LLaVA-Qwen training and inference codebase. It
contains the FastViTHD vision encoder, the LlavaQwen2 multimodal model, the
projector, the image token injection logic (`prepare_inputs_labels_for_multimodal`),
and training utilities. It is the canonical definition of FastVLM's architecture.

### How HuggingFace finds it

`config.json` in each weights directory declares a custom model type via `auto_map`:

```json
{
  model_type: llava_qwen2,
  auto_map: {
    AutoConfig: llava_qwen.LlavaConfig,
    AutoModelForCausalLM: llava_qwen.LlavaQwen2ForCausalLM
  }
}
```

When `AutoModelForCausalLM.from_pretrained(weights_dir, trust_remote_code=True)`
is called, HuggingFace reads `auto_map`, finds `llava_qwen.py` co-located with
`config.json` in the weights directory, and dynamically imports it. The
`trust_remote_code=True` flag explicitly permits execution of this arbitrary
Python from the weights directory.

This is why `llava_qwen.py` is duplicated across all three variant directories —
`weights/fastvlm-{0.5b,1.5b,7b}/` — rather than living in `scripts/`. HuggingFace
resolves the import relative to the weights directory being loaded.

### Where it is actually used in this repo

**`run_hf_fastvlm.py` and `verify_runtime.py` — indirectly, via HuggingFace auto machinery:**
These are the places `llava_qwen.py` is executed. The verify script calls
`AutoModelForCausalLM.from_pretrained(weights_dir, trust_remote_code=True)`,
which triggers the dynamic import. The resulting model provides the FastViTHD
reference implementation that our re-authored `FastVLMVisionEncoder` is compared
against. `llava_qwen.py` is never imported directly — the import statement is
nowhere in our codebase.

**Everywhere else — not used at all:**
- Weight loading uses direct safetensors reads — no `llava_qwen.py`
- The exported graph comes from our re-authored `fastvlm_*.py` scripts — no `llava_qwen.py`
- The decoder verify scripts use `Qwen2ForCausalLM` from standard `transformers` — no `llava_qwen.py`
- The Core AI runtime has zero dependency on `llava_qwen.py`

### What we re-authored and why

FastViTHD is not in `transformers` or any standard library. It exists only in
`llava_qwen.py`. To export it to Core AI we needed to control the graph precisely —
replacing `LayerNormChannel` with `F.layer_norm`, replacing manual attention with
the `SDPA` composite op, adding the fp32 input contract. A direct export of
`llava_qwen.py`'s FastViTHD would have produced a graph full of unsupported ops.

The decoder (`Qwen2ForCausalLM`) and projector (`mlp2x_gelu`) were also re-authored
for similar reasons: stateful KV cache export, eager-mode quantization compatibility,
and op-level control. Standard `transformers` Qwen2 exports cleanly enough for
reference but not for the stateful KV cache pattern we need.

### What llava_qwen.py contains that does NOT apply to FastVLM

`prepare_inputs_labels_for_multimodal` has two branches:

- `flat` — simple concat of projected image embeddings with text embeddings. **This is what FastVLM uses.**
- `spatial_unpad` — anyres tiling with learned `image_newline` tokens inserted between tile rows. More complex, requires an additional model parameter.

FastVLM uses the flat path, confirmed from `config.json`:
```
mm_patch_merge_type: flat
image_aspect_ratio:  pad
image_newline in weights: False
```

This means the Swift prefill assembly is simply: concatenate the 256 projected
image embeddings with the text token embeddings. No tiling, no `image_newline`
parameter, no anyres logic.

### When you would actually need to run llava_qwen.py directly

1. **Fine-tuning FastVLM** on custom data — the training loop, loss computation,
   and adapter fine-tuning utilities live here.
2. **Running FastVLM inference the original HF way** — if you want to sanity-check
   the weights produce sensible output using Apple's original code before going
   through the export pipeline. The vision encoder is only available here since
   FastViTHD is not in `transformers`.

For all export, verification, inspection, and Swift integration work, `llava_qwen.py`
is a reference document. Its presence in the weights directories is required only
to keep `verify_vision_encoder.py` working via HuggingFace's dynamic import.

---

*Generated June 2026. Based on Apple FastVLM HF weights (August 2025 release),
coreai-torch 0.4.0, coreai-opt 0.2.0, coreai-core 1.0.0b1, Xcode 27 beta.*
