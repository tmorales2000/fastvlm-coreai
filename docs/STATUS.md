# FastVLM CoreAI Export Pipeline — Status

**Last updated:** July 29, 2026
**Repo:** tmorales2000/fastvlm-coreai

For full installation and workflow instructions see [README.md](../README.md).
For performance benchmarks see [PERFORMANCE.md](PERFORMANCE.md).
For PSNR verification results see [psnr_results.md](psnr_results.md).

---

## ✅ Complete

- Full 3-component bundle export for all three variants (0.5B, 1.5B, 7B)
- `--compression 4bit_per_channel` (7B), `--compression 8bit` (1.5B), no compression (0.5B)
- `--kv-cache static` and `--kv-cache dynamic` working
- `inspect_aimodel.py` passes on all exported bundles
- `verify_runtime.py` passes on macOS 26.5 (71.9 dB vision_encode, 44+ dB decode)
- `verify_decoder.py` three-stage verification: correctness, fp16 cache, compression quality
- Live inference via `llm-runner` and `CoreAISequentialVLMEngine` on M4 Pro
- HF reference inference via `run_hf_fastvlm.py` — outputs match CoreAI export
- Image preprocessing fix (`center_crop`) — filed as #100, merged upstream as [apple/coreai-models #108](https://github.com/apple/coreai-models/pull/108)
- Compression preset system matching Apple's `coreai-models` pattern (`--compression`/`--compression-config`)
- 7B int4 throughput fix: 7.2 → 50.8 tok/sec via per_channel quantization
- Bundle directories now plain directories (no `.vlmasset`/`.llmasset` extension) per Apple PR #125
- Performance documented: 0.5B 80ms TTFT, 131 tok/sec; 7B int4 903ms TTFT, 51 tok/sec

## ⚠️ Known Issues

**MPSGraph crash on macOS 27 beta**
`verify_runtime.py` crashes on macOS 27 beta — OS beta bug, not export bug.
Same crash on Apple's own Qwen3-VL export. Run on macOS 26.5.
Feedback filed with Apple.

**`coreai-models` PyPI wheel**
`coreai-models==0.1.0` on PyPI declares `Python>=3.14` — incorrect.
Always install from source. See [apple/coreai-models #96](https://github.com/apple/coreai-models/issues/96).

**`--image-strategy` CLI flag in llm-runner is a dead variable**
The `--image-strategy` override flag is parsed but never passed to
`CoreAISequentialVLMEngine`. The bundle's `metadata.json` `image_strategy` field
is always used. Filed as a pending issue against `apple/coreai-models`.

## 🔲 Pending

1. **ANE compilation** — `xcrun coreai-build compile` with `--preferred-compute
   neural-engine` produces identical op distribution to GPU compile. ANE re-authoring
   (BC1S layout, Conv2d projections, readonly KV cache) would enable native ANE
   execution but is substantial work. GPU path at parity with Apple's own Qwen3-VL.

2. **7B fp16 benchmarks** — complete performance matrix pending (requires ~25GB RAM).

3. **Dynamic KV cache benchmarks** — compare TTFT and memory vs static for 0.5B.

4. **iOS export** — Scaffolding complete. Requires:
   - `fastvlm_decoder_ios.py` — BC1S layout, readonly KV I/O, 4 entrypoints
     (load_embeddings, gather_embeddings, extend, prompt_opt)
   - `coreai_models.export.ios.export_ios_model` integration
   - IOSurface hardware constraints (matching coreai-models/export/ios.py)
   Architecture is fully defined from Apple's coreai-models source.
   iOS presets (4bit_weight_palettized_group8/32) are in quantization.py.
   `--platform iOS` CLI flag is wired but raises NotImplementedError pending decoder.

5. **Native resolution preprocessing for Qwen3-VL** — PR 2 to apple/coreai-models.

5. **FastVLM as first-class coreai-models recipe** — add to `vlm/export.py` registry
   and register FastViTHD architecture in `CoreAILanguageModels` Swift package.

6. **Mixed-precision YAML recipes** — `scan_quantization_sensitivity.py` produces
   YAML compression configs for per-layer mixed precision (some layers int8,
   others int4). Pending validation run and integration test.

## Issues Filed Against apple/coreai-models

| Issue | Status | Description |
|-------|--------|-------------|
| [#96](https://github.com/apple/coreai-models/issues/96) | 🔲 Open | PyPI wheel declares incorrect `Python>=3.14` |
| [#100](https://github.com/apple/coreai-models/issues/100) | ✅ Closed | Image preprocessing strategy — merged as [#108](https://github.com/apple/coreai-models/pull/108) |
| `--image-strategy` dead variable | 🔲 To file | CLI flag accepted but never passed to engine |
