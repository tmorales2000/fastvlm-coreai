# FastVLM CoreAI Performance Benchmarks

Benchmarks collected using `llm-runner` from Apple's `coreai-models` Swift package,
running against exported bundles via `CoreAISequentialVLMEngine`.

All inference numbers are via the GPU path (MPSGraph). ANE path pending.

---

## Environment

| | |
|---|---|
| Machine | Mac Mini M4 Pro 64GB (h16s) |
| macOS | 27.0 beta |
| coreai-models | apple/coreai-models main (post PR #108, #126, #133) |
| coreai-core | 1.0.0b2 |
| torch | 2.9.0 (export time) |
| Export flags | `--kv-cache static --max-context-length 4096` |
| Image strategy | `center_crop` (declared in metadata, FastVLM default) |
| Temperature | 0 (greedy, deterministic) |

**Cold** = no `.aimodelx` cache (first ever run). **Warm** = `.aimodelx` cache present.

Two benchmark sets collected with different images and prompts:
- **great_wave**: `test_assets/images/great_wave.jpg`, "Describe exactly what you see in this image."
- **portrait**: `~/pix/a.jpg`, "Describe what you see in this image."

---

## Full Comparison Matrix (portrait benchmark, August 6 2026)

Same image and prompt across all variants for direct comparison.

| Model | Compression | TTFT warm | Gen tok/s | Prompt tok/s | Mem current | Mem peak | Load cold | Load warm | Warmup |
|-------|-------------|-----------|-----------|--------------|-------------|----------|-----------|-----------|--------|
| 0.5B | fp16 | **76ms** | **130** | 3,748 | 2,759MB | 3,444MB | 4,944ms | 230ms | 1,585ms |
| 1.5B | fp16 | 184ms | 59 | 1,540 | 7,028MB | 9,668MB | 9,602ms | 520ms | 3,784ms |
| 1.5B | int8 per_block_32 | 212ms | **71** | 1,343 | **2,771MB** | **3,929MB** | 3,633ms | **328ms** | 3,652ms |
| 1.5B | int4 per_channel | **172ms** | **115** | 1,651 | **2,021MB** | **2,444MB** | 5,615ms | **241ms** | 3,688ms |
| 7B | int4 per_channel | 748ms | 50 | 393 | 5,930MB | 9,058MB | 12,377ms | 727ms | 7,611ms |
| 7B | fp16 | 812ms | 15 | 362 | **28,829MB** | **41,526MB** | **38,343ms** | 5,796ms | 10,699ms |

TTFT = `decoder.Prompt` metric from verbose output (image encode + prefill, excludes load/warmup).

**Recommended configurations:**
- **0.5B fp16** — fastest TTFT (76ms), highest throughput (130 tok/s), smallest footprint (2.8GB)
- **1.5B int4 per_channel** — best overall: near-0.5B speed at 3× parameters, sub-0.5B memory
- **7B int4 per_channel** — maximum quality, 50 tok/s, 6GB. **7B fp16 is not recommended** (29GB, 15 tok/s)

---

## FastVLM 0.5B (fp16)

**Export:** `python scripts/export_fastvlm.py --variant 0.5b`

### great_wave benchmark

| Metric | Cold | Warm |
|--------|------|------|
| Model load | 4,586ms | 245ms |
| Tokenizer load | — | 773ms |
| Warmup (one-time JIT) | 1,699ms | 1,582ms |
| **TTFT (prompt processing)** | **84.8ms** | **80.4ms** |
| Prompt throughput | 3,362 tok/sec | 3,543 tok/sec |
| **Generation throughput** | **131.3 tok/sec** | **131.5 tok/sec** |
| Tokens generated | 314 | 314 |
| Memory (current) | 2,858MB | 2,811MB |
| Memory (peak) | 3,703MB | 3,448MB |

### portrait benchmark (August 6 2026)

| Metric | Cold | Warm |
|--------|------|------|
| Model load | 4,944ms | 230ms |
| Tokenizer load | — | 773ms |
| Warmup (one-time JIT) | 1,704ms | 1,585ms |
| **TTFT (prompt processing)** | **77.3ms** | **75.8ms** |
| Prompt throughput | 3,674 tok/sec | 3,748 tok/sec |
| **Generation throughput** | **77 tok/sec (cold)** | **130 tok/sec (warm)** |
| Tokens generated | 308 | 308 |
| Memory (current) | 2,803MB | 2,759MB |
| Memory (peak) | 3,704MB | 3,444MB |

**KV cache:** StaticKVCache, `[24, 1, 2, 4096, 64]`, 24MB

> **Sub-100ms TTFT:** 76ms warm is the user-visible latency in a production app
> where the model is already loaded. Cold generation (77 tok/sec) is lower than
> warm (130 tok/sec) because the first run includes JIT graph priming.

> **Compute path:** All three components run GPU dynamic via MPSGraph. Vision tower
> shows `Unknown model structure` — FastViTHD is not a recognized architecture.

---

## FastVLM 1.5B (fp16)

**Export:** `python scripts/export_fastvlm.py --variant 1.5b`

### great_wave benchmark

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

### portrait benchmark (August 6 2026)

| Metric | Cold | Warm |
|--------|------|------|
| Model load | 9,602ms | 520ms |
| Warmup (one-time JIT) | 4,014ms | 3,784ms |
| **TTFT (prompt processing)** | **183ms** | **184ms** |
| Prompt throughput | 1,552 tok/sec | 1,540 tok/sec |
| **Generation throughput** | **59.1 tok/sec** | **59.0 tok/sec** |
| Tokens generated | 167 | 167 |
| Memory (current) | 7,076MB | 7,028MB |
| Memory (peak) | 11,650MB | 9,668MB |

**KV cache:** StaticKVCache, `[28, 1, 2, 4096, 128]`, 112MB

---

## FastVLM 1.5B (int4 per_channel)

**Export:** `python scripts/export_fastvlm.py --variant 1.5b --compression 4bit_per_channel`

Compression: int4 symmetric per-channel.

### portrait benchmark (August 6 2026)

| Metric | Cold | Warm | vs 1.5B fp16 warm |
|--------|------|------|-------------------|
| Model load | 5,615ms | **241ms** | 2.2× faster |
| Warmup (one-time JIT) | 3,672ms | 3,688ms | ~same |
| **TTFT (prompt processing)** | **174ms** | **172ms** | **7% faster** |
| Prompt throughput | 1,631 tok/sec | 1,651 tok/sec | ~same |
| **Generation throughput** | **110 tok/sec** | **115 tok/sec** | **1.95× faster** |
| Tokens generated | 124 | 124 | — |
| Memory (current) | 2,078MB | **2,021MB** | **71% lower** |
| Memory (peak) | 2,558MB | **2,444MB** | **75% lower** |

**KV cache:** StaticKVCache, `[28, 1, 2, 4096, 128]`, 112MB

> **The standout result in this benchmark suite.** 1.5B int4 delivers:
> - 115 tok/sec warm — nearly matching 0.5B fp16 (130 tok/sec) at 3× the parameters
> - 2.0GB current memory — less than 0.5B fp16 (2.8GB)
> - 172ms TTFT — slightly faster than 1.5B fp16 (184ms)
> - Clean, complete output with no quality regression vs fp16

> **Why int4 is faster than fp16 here:** Per-channel int4 quantization fuses
> dequantization into batch_matmul as row-wise scaling, reducing memory bandwidth
> pressure. The 1.5B model is memory-bandwidth-bound at fp16 — int4 removes the
> bottleneck and allows the GPU to run closer to compute-bound.

---

## FastVLM 1.5B (int8 per_block_32)

**Export:** `python scripts/export_fastvlm.py --variant 1.5b --compression 8bit`

Compression: int8 symmetric_with_clipping per_block_32. Targets `nn.Linear` only.
`FastVLMRMSNorm` excluded (1D weight incompatible with per_block axis=1).

### portrait benchmark (August 14 2026)

| Metric | Cold | Warm | vs 1.5B fp16 warm |
|--------|------|------|-------------------|
| Model load | 3,633ms | **328ms** | **1.6× faster** |
| Warmup (one-time JIT) | 4,478ms | 3,652ms | ~same |
| **TTFT (prompt processing)** | 1,326ms | **212ms** | ~same |
| Prompt throughput | 214 tok/sec | 1,343 tok/sec | ~same |
| **Generation throughput** | **72.3 tok/sec** | **71.3 tok/sec** | **1.2× faster** |
| Tokens generated | 167 | 165 | — |
| Memory (current) | 2,855MB | **2,771MB** | **61% lower** |
| Memory (peak) | 4,029MB | **3,929MB** | **59% lower** |

**KV cache:** StaticKVCache, `[28, 1, 2, 4096, 128]`, 112MB

> **71 tok/sec — 1.2× faster than 1.5B fp16 (59 tok/sec).** Per_block_32
> symmetric_with_clipping reduces memory bandwidth enough for the GPU to run
> more efficiently than fp16 on the M4 Pro.

> **61% lower memory:** 2.8GB current vs 7.0GB for fp16 — less than 0.5B fp16 (2.8GB).
> Peak also dramatically lower: 3.9GB vs 9.7GB.

> **Cold TTFT 1,326ms** — higher than warm (212ms) because the first inference
> run includes JIT graph priming. Model load (3.6s cold) is much faster than
> the old per_channel scheme which caused a 59-second cold load.

---

## FastVLM 7B (int4 per_channel)

**Export:** `python scripts/export_fastvlm.py --variant 7b --compression 4bit_per_channel`

Compression: int4 symmetric per-channel.

> The 7B variant uses Qwen2.5-7B base (vocab 152064, image token 151665) vs
> Qwen2 for 0.5B/1.5B (vocab 151936, image token 151646).

### portrait benchmark (August 6 2026)

| Metric | Cold | Warm |
|--------|------|------|
| Model load | 12,377ms | 727ms |
| Warmup (one-time JIT) | 7,515ms | 7,611ms |
| **TTFT (prompt processing)** | **746ms** | **748ms** |
| Prompt throughput | 394 tok/sec | 393 tok/sec |
| **Generation throughput** | **50.0 tok/sec** | **50.0 tok/sec** |
| Tokens generated | 333 | 333 |
| Memory (current) | 5,633MB | 5,930MB |
| Memory (peak) | 9,111MB | 9,058MB |

**KV cache:** StaticKVCache, `[28, 1, 4, 4096, 128]`, 224MB

> **7× throughput improvement over original per_block_64 asymmetric** (7.2 → 50 tok/sec).
> Per_channel dequantization fuses into batch_matmul; per_block runs as a separate pass.

---

## FastVLM 7B (fp16)

**Export:** `python scripts/export_fastvlm.py --variant 7b`

⚠️ **Not recommended for production.** Memory requirements exceed practical limits
for interactive use even on 64GB machines.

### portrait benchmark (August 6 2026)

| Metric | Cold | Warm |
|--------|------|------|
| Model load | **38,343ms** | **5,796ms** |
| Warmup (one-time JIT) | **15,634ms** | **10,699ms** |
| **TTFT (prompt processing)** | **839ms** | **812ms** |
| Prompt throughput | 351 tok/sec | 362 tok/sec |
| **Generation throughput** | **15.3 tok/sec** | **15.3 tok/sec** |
| Tokens generated | 358 | 358 |
| Memory (current) | **28,837MB** | **28,829MB** |
| Memory (peak) | **43,057MB** | **41,526MB** |

**KV cache:** StaticKVCache, `[28, 1, 4, 4096, 128]`, 224MB

> **28GB resident memory, 41–43GB peak** — barely fits in 64GB unified memory.
> Any concurrent process risks memory pressure and swapping. Cold load takes
> 38 seconds; even warm load takes 5.8s. Generation at 15 tok/sec is 3.3×
> slower than 7B int4 for no quality benefit.
>
> **Use 7B int4 per_channel instead.** It delivers identical output quality
> at 50 tok/sec, 6GB memory, and sub-1s warm load.

---

## Quantization Scheme Findings

| Scheme | blockwise_shift_scale | 7B gen tok/s | Notes |
|--------|----------------------|--------------|-------|
| per_block_64 asymmetric (original) | 197 | 7.2 | Unfused — catastrophic |
| per_block_32 symmetric_with_clipping (apple_4bit) | 197 | ~7–10 (est.) | Unfused — same problem |
| per_channel symmetric (4bit_per_channel) | 197 | **50** | Fused — 7× faster |

Op count is identical across schemes. Per-channel allows GPU to fuse dequantization
into batch_matmul as row-wise scaling. Per-block requires a separate pass.

**Current presets (`quantization.py`):**
- `4bit` — Apple macOS standard (symmetric_with_clipping per_block_32). Best quality,
  but unfused on GPU. Avoid for large models where throughput matters.
- `4bit_per_channel` — per_channel symmetric int4. 7× faster. **Recommended for 7B.**
- `8bit` — symmetric_with_clipping per_block_32 int8. **71 tok/sec** (faster than fp16).
  Memory: 2.8GB (61% lower than fp16). Recommended for memory-constrained deployments.

---

## FastVLM 0.5B vs Qwen3-VL 2B — Head-to-Head (M4 Pro)

Portrait photo benchmark, temperature 0.7.

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
| TTFT | **~76ms** | ~270ms |
| Generation | ~130 tok/sec | TBD |
| Vision encoder | CoreAI GPU (MPSGraph) | CoreML (.mlpackage) |
| Decoder | CoreAI GPU (MPSGraph) | MLX (eager, GPU) |
| Bundle format | plain directory (3 components) | `.mlpackage` + MLX weights |

CoreAI is **~3.5× faster** TTFT than Apple's own MLX-FastVLM hybrid app.

---

## Scaling Summary

| | 0.5B fp16 → 1.5B fp16 | 1.5B fp16 → 1.5B int8 | 1.5B fp16 → 1.5B int4 | 1.5B fp16 → 7B int4 | 7B int4 → 7B fp16 |
|--|----------------------|----------------------|----------------------|---------------------|-------------------|
| TTFT | 2.4× slower | ~same | 6% faster | 4.1× slower | 9% slower |
| Generation | 2.2× slower | **1.2× faster** | **2× faster** | ~same (50 vs 59) | **3.3× slower** |
| Memory current | 2.5× more | **61% lower** | **71% lower** | ~same | **4.9× more** |
| Model load warm | 2.3× slower | 1.6× faster | 2.3× faster | 1.4× slower | 8× slower |

---

## Notes

### Cold vs warm load

**Cold** = no `.aimodelx` cache. CoreAI JIT-compiles on first run, caches in
`~/Library/Caches/`. fp16 cold scales linearly with model size. int4 cold is
similar. int8 cold is ~7× slower (59s for 1.5B). 7B fp16 cold is 38 seconds.

Production apps should pre-compile via `xcrun coreai-build compile` — PR #133
(Aug 2026) means the runtime automatically finds `.aimodelc` alongside `.aimodel`
without requiring `metadata.json` updates.

### Warmup

One-time JIT warmup primes the GPU pipeline every process launch regardless of
cache state. Warmup is the dominant startup cost for 1.5B+ models:
- 0.5B: ~1.6s warmup vs 0.23s model load (warm)
- 1.5B: ~3.7s warmup vs 0.52s model load (warm)
- 7B:   ~7.6s warmup vs 0.73s model load (warm)

### Tokenizer load

Tokenizer load (Jinja template compilation) is ~770–830ms and constant across
all model variants. Reported separately in the new verbose output format.

### int8 quantization

The `8bit` preset uses symmetric_with_clipping per_block_32, targeting `nn.Linear`
only. `FastVLMRMSNorm` is explicitly excluded — its 1D weight `[hidden_size]` is
incompatible with per_block axis=1 and was silently corrupting quantization in the
previous per_channel scheme (causing 27 tok/sec and looping output).
Result: 71 tok/sec (faster than fp16), 2.8GB memory (61% lower), clean output.

### int4 quantization on GPU path

Per-channel int4 fuses dequantization into batch_matmul (row-wise scaling).
Per-block int4 runs as a separate unfused pass regardless of block size or
symmetric/asymmetric choice. Use `--compression 4bit_per_channel`, not
`--compression 4bit`, for production 7B exports.
