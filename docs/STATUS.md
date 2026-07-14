# FastVLM CoreAI Export Pipeline — Status

**Last updated:** July 14, 2026
**Repo:** tmorales2000/fastvlm-coreai

For full installation and workflow instructions see [README.md](../README.md).
For performance benchmarks see [PERFORMANCE.md](PERFORMANCE.md).
For PSNR verification results see [psnr_results.md](psnr_results.md).

---

## ✅ Complete

- Full 3-component `.vlmasset` bundle export for all three variants (0.5B, 1.5B, 7B)
- `--quantize int8` (1.5B) and `--quantize int4` (7B) working
- `--kv-cache static` and `--kv-cache dynamic` working
- `inspect_aimodel.py` passes on all exported bundles
- `verify_runtime.py` passes on macOS 26.5 (71.9 dB vision_encode, 44+ dB decode)
- Live inference via `llm-runner` and `CoreAISequentialVLMEngine` on M4 Pro
- HF reference inference via `run_hf_fastvlm.py` — outputs match CoreAI export
- Image preprocessing fix (`center_crop`) implemented in fork, filed upstream as #100
- Benchmark test images: 9 public domain + 2 synthetic (fetch/generate scripts)
- Performance documented: ~97ms TTFT, 113 tok/sec generation, 3901 tok/sec prefill

## ⚠️ Known Issues

**MPSGraph crash on macOS 27 beta**
`verify_runtime.py` crashes on macOS 27 beta — OS beta bug, not export bug.
Same crash on Apple's own Qwen3-VL export. Run on macOS 26.5.
Feedback filed with Apple.

**`coreai-models` PyPI wheel**
`coreai-models==0.1.0` on PyPI declares `Python>=3.14` — incorrect.
Always install from source. See [apple/coreai-models #96](https://github.com/apple/coreai-models/issues/96).

**Image preprocessing in upstream `coreai-models`**
`CoreAISequentialVLMEngine` stretch-resizes images — incorrect for FastVLM.
Fixed in `tmorales2000/coreai-models` fork (`fix/vlm-image-preprocessing-strategy` branch).
Proposed upstream as [apple/coreai-models #100](https://github.com/apple/coreai-models/issues/100).

## 🔲 Pending

1. **ANE compilation** — `xcrun coreai-build compile` produces `.aimodelc` but current
   numbers are GPU (MPSGraph). ANE path requires verifying `CoreAISequentialVLMEngine`
   recognizes FastViTHD architecture. `Unknown model structure` warning observed.

2. **7B benchmark** — verify_runtime.py and performance numbers for 7B pending.

3. **Dynamic KV cache benchmarks** — compare TTFT and memory vs static for 0.5B.

4. **Native resolution preprocessing for Qwen3-VL** — PR 2 to apple/coreai-models.
   Requires dynamic token count, variable-resolution vision export.

5. **FastVLM as first-class coreai-models recipe** — add to `vlm/export.py` registry
   and register FastViTHD architecture in `CoreAILanguageModels` Swift package.

## Issues Filed Against apple/coreai-models

| Issue | Status | Description |
|-------|--------|-------------|
| [#95](https://github.com/apple/coreai-models/issues/95) | ✅ Fixed same day | Tokenizer save fails with `allow_patterns` |
| [#96](https://github.com/apple/coreai-models/issues/96) | 🔲 Open | PyPI wheel declares incorrect `Python>=3.14` |
| [#100](https://github.com/apple/coreai-models/issues/100) | 🔲 Proposed | Image preprocessing strategy — fix on fork |
