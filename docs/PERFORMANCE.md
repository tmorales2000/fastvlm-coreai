# FastVLM CoreAI Performance Benchmarks

Benchmarks collected using `llm-runner` from Apple's `coreai-models` Swift package,
running against exported `.vlmasset` bundles via `CoreAISequentialVLMEngine`.

All inference numbers are via the GPU path (MPSGraph). ANE path pending investigation.

---

## Environment

| | |
|---|---|
| Machine | Mac Mini M4 Pro 64GB (h16s) |
| macOS | 27.0 beta |
| coreai-models | apple/coreai-models main (post PR #108) |
| coreai-core | 1.0.0b2 |
| torch | 2.9.0 (export time) |
| Export flags | `--kv-cache static --max-context-length 4096` |
| Image strategy | `center_crop` (declared in metadata, FastVLM default) |
| Benchmark image | `test_assets/images/great_wave.jpg` |
| Prompt | "Describe exactly what you see in this image." |
| Temperature | 0 (greedy, deterministic) |

**Cold** = no `.aimodelx` cache (first ever run). **Warm** = `.aimodelx` cache present.

---

## Full Comparison Matrix

| Model | Compression | TTFT (ms) | Gen (tok/s) | Prompt (tok/s) | Mem current | Mem peak | Load cold | Load warm |
|-------|-------------|-----------|-------------|----------------|-------------|----------|-----------|-----------|
| 0.5B | fp16 | **80** | **131** | 3,543 | 2,811MB | 3,448MB | 4,586ms | 245ms |
| 1.5B | fp16 | 192 | 59 | 1,481 | 7,079MB | 9,668MB | 8,261ms | 534ms |
| 1.5B | int8 per_channel | 194 | 59 | 1,465 | 5,678MB | 8,289MB | 59,279ms | 392ms |
| 7B | fp16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 7B | int4 per_channel | **903** | **51** | 327 | 5,968MB | 9,057MB | TBD | 699ms |

TTFT = `decoder.Prompt` from verbose output (image encode + prefill, excludes load/warmup).

---

## FastVLM 0.5B (fp16)

**Export:** `python scripts/export_fastvlm.py --variant 0.5b`

| Metric | Cold | Warm |
|--------|------|------|
| Model load | 4,586ms | 245ms |
| Warmup (one-time JIT) | 1,699ms | 1,582ms |
| **TTFT (prompt processing)** | **84.8ms** | **80.4ms** |
| Prompt throughput | 3,362 tok/sec | 3,543 tok/sec |
| **Generation throughput** | **131.3 tok/sec** | **131.5 tok/sec** |
| Tokens generated | 314 | 314 |
| Memory (current) | 2,858MB | 2,811MB |
| Memory (peak) | 3,703MB | 3,448MB |
| Total wall time | 9.2s | 5.2s |

**KV cache:** StaticKVCache, `[24, 1, 2, 4096, 64]`, 24MB

> **Sub-100ms TTFT:** 80ms warm is the user-visible latency in a production app
> where the model is already loaded. This is the relevant number for real-time
> applications. Cold TTFT is identical — warmup compiles the graph, not the
> prefill path.

> **Compute path:** All three components (vision, embed, decoder) run GPU dynamic
> via MPSGraph. Vision tower shows `Unknown model structure` warning — FastViTHD
> is not a recognized architecture, defaults to GPU. ANE path pending.

---

## FastVLM 1.5B (fp16)

**Export:** `python scripts/export_fastvlm.py --variant 1.5b`

| Metric | Cold | Warm |
|--------|------|------|
| Model load | 8,261ms | 534ms |
| Warmup (one-time JIT) | 4,065ms | 3,805ms |
| **TTFT (prompt processing)** | **208ms** | **192ms** |
| Prompt throughput | 1,368 tok/sec | 1,481 tok/sec |
| **Generation throughput** | **59.3 tok/sec** | **58.8 tok/sec** |
| Tokens generated | 197 | 197 |
| Memory (current) | 7,135MB | 7,079MB |
| Memory (peak) | 11,650MB | 9,668MB |
| Total wall time | 16.3s | 8.5s |

**KV cache:** StaticKVCache, `[28, 1, 2, 4096, 128]`, 112MB

> **Startup latency:** Warm startup = 534ms load + 3,805ms warmup = 4.3s before
> first inference. Warmup (graph priming) dominates, not model load.

---

## FastVLM 1.5B (int8 per_channel)

**Export:** `python scripts/export_fastvlm.py --variant 1.5b --compression 8bit`

Compression: int8 symmetric per-channel (`quantization.py` preset `8bit`).

| Metric | Cold | Warm | vs 1.5B fp16 warm |
|--------|------|------|-------------------|
| Model load | **59,279ms** | **392ms** | 1.4× faster warm |
| Warmup (one-time JIT) | 4,179ms | 3,829ms | ~same |
| **TTFT (prompt processing)** | **192ms** | **194ms** | ~same |
| Prompt throughput | 1,481 tok/sec | 1,465 tok/sec | ~same |
| **Generation throughput** | **59.1 tok/sec** | **58.7 tok/sec** | ~same |
| Tokens generated | 193 | 193 | — |
| Memory (current) | 13,284MB | **5,678MB** | **20% lower** |
| Memory (peak) | 20,479MB | **8,289MB** | **14% lower** |
| Total wall time | 67.3s | 8.5s | — |

**KV cache:** StaticKVCache, `[28, 1, 2, 4096, 128]`, 112MB

> **⚠️ Cold load: 59 seconds.** JIT compilation of int8 dequantization ops takes
> ~7× longer than fp16. Peak memory during cold compilation reaches 20GB.
> Production apps should pre-compile via `xcrun coreai-build compile` to ship
> `.aimodelc` and avoid the cold load penalty entirely.

> **TTFT and generation identical to fp16:** The GPU (MPSGraph) path dequantizes
> int8 weights to fp16 before matrix multiply — no native int8 compute on this path.
> int8 saves memory and storage but not compute time.

> **Memory advantage (warm):** int8 saves ~1.4GB resident memory vs fp16.
> Peak during inference is 14% lower.

---

## FastVLM 7B (fp16) — TBD

**Export:** `python scripts/export_fastvlm.py --variant 7b`

| Metric | Cold | Warm |
|--------|------|------|
| Model load | TBD | TBD |
| Warmup (one-time JIT) | TBD | TBD |
| TTFT (prompt processing) | TBD | TBD |
| Prompt throughput | TBD | TBD |
| Generation throughput | TBD | TBD |
| Memory (current) | TBD | TBD |
| Memory (peak) | TBD | TBD |

> Note: 7B fp16 benchmarks pending. Model requires ~25-28GB resident memory.
> Warmup estimated at ~15s based on scaling from smaller variants.

---

## FastVLM 7B (int4 per_channel)

**Export:** `python scripts/export_fastvlm.py --variant 7b --compression 4bit_per_channel`

Compression: int4 symmetric per-channel (`quantization.py` preset `4bit_per_channel`).

> **Important:** The `7B` variant uses a Qwen2.5-7B base (vocab 152064, image token ID
> 151665) vs Qwen2 for 0.5B/1.5B (vocab 151936, image token ID 151646). Different
> base models — different tokenizers — same FastViTHD vision tower.

| Metric | Warm | vs 1.5B fp16 warm |
|--------|------|-------------------|
| Model load | **699ms** | 1.3× slower |
| Warmup (one-time JIT) | 7,386ms | 1.9× slower |
| **TTFT (prompt processing)** | **903ms** | 4.7× slower |
| Prompt throughput | 327 tok/sec | 4.5× slower |
| **Generation throughput** | **50.8 tok/sec** | ~same |
| Tokens generated | 425 | — |
| Memory (current) | 5,968MB | ~same |
| Memory (peak) | 9,057MB | ~same |
| Total wall time | 17.9s | — |

**KV cache:** StaticKVCache, `[28, 1, 4, 4096, 128]`, 224MB

> **7× throughput improvement over per_block_64 asymmetric** (old scheme: 7.2 tok/sec →
> new scheme: 50.8 tok/sec). Root cause: per_block quantization produces
> `blockwise_shift_scale` ops (197 in compiled graph) that are NOT fused with
> `batch_matmul` on GPU — the dequantization runs as a separate pass over the full
> weight matrix. Per_channel dequantization (one scale per output row) fuses directly
> into `batch_matmul` as a row-wise scaling, eliminating the separate pass.
> The compiled `blockwise_shift_scale` count is 197 for both schemes — the op name
> is the same but per_channel execution is 7× faster because the scale granularity
> allows GPU fusion.

> **Quality trade-off:** int4 per_channel has fewer scale parameters than per_block
> (one per row vs one per 32 elements). PSNR vs fp16 baseline: 29.5 dB (per_channel)
> vs expected ~32 dB (per_block_32). Acceptable for deployment.

> **Memory vs 1.5B fp16:** Nearly identical (5,968MB vs 7,079MB current) despite
> 4.7× more parameters — int4 quantization reduces weight storage to ~1/4 of fp16.
> The 7B int4 model is memory-competitive with 1.5B fp16 while delivering substantially
> better output quality.

> **Surprising result:** 50.8 tok/sec for 7B int4 is close to 58.8 tok/sec for 1.5B
> fp16. You get 7B quality at near-1.5B speed. The TTFT (903ms) is the key trade-off —
> prefill of 295 tokens takes nearly 1 second vs 192ms for 1.5B.

---

## Quantization Scheme Findings

Critical discovery from benchmarking 7B int4 with multiple schemes:

| Scheme | blockwise_shift_scale | 7B gen tok/s | Notes |
|--------|----------------------|--------------|-------|
| per_block_64 asymmetric (old default) | 197 | 7.2 | Unfused, catastrophically slow |
| per_block_32 symmetric_with_clipping (apple_4bit) | 197 | ~7-10 (est.) | Unfused, same problem |
| per_channel symmetric (4bit_per_channel) | 197 | **50.8** | Fused — 7× faster |

**Key insight:** `blockwise_shift_scale` count (197) is identical across all schemes —
the compiler emits the same op regardless. The difference is execution behavior:
- Per-block: scale granularity requires separate pass over weight matrix → unfused
- Per-channel: one scale per output row → fuses into batch_matmul as row-wise scaling

**Current preset choices (`quantization.py`):**
- `4bit` — Apple's macOS standard (symmetric_with_clipping per_block_32). Best quality,
  but unfused on GPU — avoid for large models where throughput matters.
- `4bit_per_channel` — per_channel symmetric int4. 7× faster on GPU, slight quality
  trade-off. Recommended for 7B.
- `8bit` — per_channel symmetric int8. Identical throughput to fp16 on GPU path
  (dequantizes to fp16 before compute). Memory savings only.

---

## FastVLM 0.5B vs Qwen3-VL 2B — Head-to-Head (M4 Pro)

Same image, same prompt, same `llm-runner`. Earlier benchmark (portrait photo,
temperature 0.7 — different from great_wave standardized set above).

| Metric | FastVLM 0.5B (fp16) | Qwen3-VL 2B (fp16) | FastVLM advantage |
|--------|--------------------|--------------------|-------------------|
| Warmup (one-time JIT) | 1,630ms | 14,765ms | **9× faster** |
| Prompt throughput | 3,606 tok/sec | 639 tok/sec | **5.6× faster** |
| Generation throughput | 115 tok/sec | 51 tok/sec | **2.3× faster** |
| Memory (current) | 2,761MB | 9,140MB | **3.3× lower** |
| Memory (peak) | 3,446MB | 11,847MB | **3.4× lower** |
| Image tokens | 256 | 196 | more visual detail |

---

## Comparison: CoreAI vs MLX-FastVLM (M4 Pro, 0.5B)

| Metric | CoreAI (this project) | MLX-FastVLM (Apple hybrid) |
|--------|----------------------|---------------------------|
| TTFT | **~80ms** | ~270ms |
| Generation | ~131 tok/sec | TBD |
| Vision encoder path | CoreAI GPU (MPSGraph) | CoreML (.mlpackage) |
| Decoder path | CoreAI GPU (MPSGraph) | MLX (eager, GPU) |
| Bundle format | `.vlmasset` (3 components) | `.mlpackage` + MLX weights |

CoreAI is **~3.4× faster** TTFT than Apple's own MLX-FastVLM hybrid app on the
same hardware. The primary driver is the unified CoreAI pipeline vs the hybrid
CoreML+MLX approach which has cross-framework overhead.

---

## Scaling Summary

| | 0.5B→1.5B (3× params) | 1.5B→7B (4.7× params, int4) | 1.5B fp16→int8 |
|--|----------------------|----------------------------|----------------|
| TTFT | 2.4× slower | 4.7× slower | ~same |
| Generation | 2.2× slower | ~same (50 vs 59 tok/s) | ~same |
| Memory (warm current) | 2.5× more | ~same (int4 offsets size) | 20% less |
| Model load (warm) | 2.2× slower | 1.3× slower | 1.4× faster |

---

## Notes

### Cold vs warm load

**Cold** (no `.aimodelx` cache): CoreAI JIT-compiles the `.aimodel` graph on first run.
The result is cached as `.aimodelx` in `~/Library/Caches/`.

- fp16 cold load scales roughly linearly with model size
- int8 cold load is ~7× slower due to quantization op compilation; peak memory 20GB
- int4 cold load: TBD (expected similarly slow to int8)
- Production apps should pre-compile via `xcrun coreai-build compile` to avoid cold penalty

### Warmup

The one-time JIT warmup runs a dummy forward pass to prime the GPU pipeline.
Happens every process launch regardless of cache state. In a production app this is
the startup latency after model load.

### GPU path (current) vs ANE (pending)

All numbers are GPU via MPSGraph. `Unknown model structure` on `vision.aimodel` means
FastViTHD defaults to GPU. The compiled op distribution for our exports is architecturally
equivalent to Apple's own Qwen3-VL exports (same op types, same counts scaled by layers).
ANE path would require AOT compilation via `xcrun coreai-build compile --preferred-compute
neural-engine` and potentially decoder re-authoring for full ANE utilization.

### int4 quantization on GPU path

Per-block int4 (any block size) produces `blockwise_shift_scale` ops that run as a
separate dequantization pass before `batch_matmul` — unfused, slow. Per-channel int4
fuses into `batch_matmul` as row-wise scaling — 7× faster on M4 Pro GPU. Use
`--compression 4bit_per_channel` for 7B, not `--compression 4bit`.
