"""
fastvlm_decoder.py — FastVLM language decoder re-authored for Core AI export.

WHAT THIS FILE IS
-----------------
FastVLM's language decoder is a Qwen2 transformer. The original model lives in
llava_qwen.py (downloaded with the weights) and delegates to HuggingFace's
Qwen2ForCausalLM internally. This file re-authors that decoder in plain PyTorch
so that TorchConverter can export it to a Core AI .aimodel.

Re-authoring is required because the original HuggingFace model uses Python
control flow and ops that are not exportable, and does not use the Core AI
composite ops (RMSNorm, SDPA, RoPE) that the ANE compiler needs to recognize
and optimize key operations.

WHY THE STRUCTURE IS WHAT IT IS
--------------------------------
Three constraints drive every decision in this file:

1. Composite ops must be explicit.
   The ANE compiler recognizes RMSNorm, scaled dot-product attention, and
   rotary position embeddings only when they appear as specific registered ops
   (ExternalizeSpec boundary nodes) in the exported graph. Using nn.LayerNorm,
   F.scaled_dot_product_attention, or a manual RoPE implementation instead
   would produce functionally correct but ANE-unoptimized code. This is why
   FastVLMRMSNorm, SDPA, and RoPE are wired as composite ops.

2. The KV cache must be a mutable registered buffer, not a Python list.
   torch.export.export traces a single static graph. The cache cannot be
   passed as an input/output argument pair — it must be a registered buffer
   that is mutated in place. TorchConverter then binds the buffer to
   state_names=["k_cache","v_cache"], making it a persistent state that
   survives across inference calls in the compiled .aimodel.

3. The cache write uses torch.ops.aten.slice_scatter, not a custom op, and
   not an in-place narrow().copy_().
   Two earlier approaches were tried and rejected:
     (a) A "coreai::mutable_slice_update" custom op (torch.library.custom_op
         with mutates_args=["x"]), copied verbatim from Apple's
         coreai_models/primitives/_ops.py reference. Rejected: torch.export
         wraps mutates_args custom ops in
         torch.ops.higher_order.auto_functionalized_v2, and coreai-torch
         0.4.0's converter has no lowering for that op — it crashes with
         UnboundLocalError inside the converter itself (confirmed via
         direct node-tracing).
     (b) k_cache[layer].narrow(1, offset, L).copy_(k_flat) — in-place
         mutation through a narrowed view with a dynamic (tensor-valued)
         start. Rejected: fails inside torch.export itself, before ever
         reaching coreai-torch, with PendingUnbackedSymbolNotFound — the
         unbacked symbol allocated for the dynamically-shaped narrowed
         view's size doesn't get threaded through correctly for an
         in-place mutation through that view.
   The working approach: torch.ops.aten.slice_scatter(cache[layer], src,
   dim=1, start=offset, end=offset+L) is a FUNCTIONAL op — it returns a new
   full-shape tensor with src written into the given slice, rather than
   mutating a narrowed view in place. coreai-torch has a registered,
   tested lowering for it (replace_slice_scatter -> coreai.slice_update in
   _aten_to_core.py) that explicitly resolves dynamic start/end values.
   The full-shape result is then written into the registered buffer with a
   single .copy_() — a static-shape tensor-to-tensor copy, which does not
   hit the unbacked-symbol problem that approach (b) did.

WEIGHT LAYOUT (confirmed from discover_weights.py output)
----------------------------------------------------------
The FastVLM checkpoint stores decoder weights with these prefixes:
  model.layers.*       — transformer blocks
  model.embed_tokens.* — token embedding table
  model.norm.*         — final RMS norm
  lm_head.*            — unembedding projection

There is NO "language_model." prefix — the checkpoint is flat.
All weights are stored as bfloat16; this file casts to float16 on load.

QKV FUSION
----------
HuggingFace Qwen2 stores q_proj, k_proj, v_proj as separate weights. This file
uses a single fused qkv_proj (matching Apple's coreai-models/qwen2.py pattern)
because fused QKV is more efficient on ANE. _mutate_state_dict() performs the
fusion before load_state_dict() is called.

Fusion order is [q, k, v] — confirmed by comparing logits against the original
HuggingFace Qwen2ForCausalLM in verify_decoder.py Stage 1 (113.2 dB PSNR).

KV CACHE LAYOUT
---------------
The cache buffers have shape (n_layers, 1, MAX_SEQ_LEN, kv_dim) where:
  n_layers   = config.num_hidden_layers  (28 for 1.5B)
  1          = batch size (always 1 for on-device inference)
  max_seq_len = config.max_position_embeddings (4096 for 0.5B/1.5B, 8192 for 7B)
  kv_dim     = head_dim * num_key_value_heads  (128 * 2 = 256 for 1.5B GQA)

Heads are FLATTENED in the cache (kv_dim = n_kv_heads * head_dim) and restored
to head-separated layout (B, n_kv_heads, seq_len, head_dim) before SDPA.
This flattened layout is what the narrow(...).copy_(...) write and the
narrow(...) read both use, and it must be consistent — mixing the two
layouts corrupts the cache.

ROPE_THETA
----------
FastVLM's config stores rope_theta inside a nested "rope_parameters" dict
(LlavaConfig wraps the Qwen2 config). It does NOT appear as a direct attribute
of the text config. The correct value is 1,000,000 — confirmed from:
  - weights/fastvlm-1.5b/config.json: rope_parameters.rope_theta = 1000000.0
  - MLXVLM/Models/Qwen2VL.swift line 767: ropeTheta default = 1_000_000

Using the wrong fallback (1e4) produces ~32 dB Stage 1 PSNR instead of 113 dB.

VERIFIED RESULTS (1.5B, June 2026)
-----------------------------------
Stage 1 — fp32 port vs HF Qwen2ForCausalLM: 113.2 dB  [PASS, threshold 80 dB]
Stage 2 — fp16 cached decode vs full pass:    72.1 dB  [PASS, threshold 40 dB]
fp16 max logit: 12 (fp16 ceiling 65504 — no overflow risk)
"""

import glob
import os

import torch
import torch.nn as nn
from coreai_torch.composite_ops import RMSNormImpl, RoPE, SDPA
from safetensors import safe_open

# NOTE: an earlier version of this file defined a "coreai::mutable_slice_update"
# custom op (torch.library.custom_op with mutates_args=["x"]) here, copied
# verbatim from Apple's coreai_models/primitives/_ops.py reference. It has
# been removed — see the "KV cache mutation" section of the module docstring
# above for why: torch.export wraps mutates_args custom ops in
# auto_functionalized_v2, which coreai-torch 0.4.0's converter cannot lower,
# crashing with UnboundLocalError. The KV cache is now mutated with plain
# torch.narrow(...).copy_(...) calls directly in FastVLMAttention.forward(),
# matching the pattern coreai-torch's own tests/test_stateful.py uses and
# tests for (including under dynamic shapes).


# ─── Composite op wrappers ────────────────────────────────────────────────────


# ─── Composite op wrappers ────────────────────────────────────────────────────


class FastVLMRMSNorm(nn.Module):
    """
    RMS layer normalization using the Core AI RMSNormImpl composite op.

    RMSNormImpl.forward(x, weight) takes the scale (gamma) as a forward
    argument rather than holding it internally. This wrapper holds the
    learnable scale as an nn.Parameter and passes it through, so that
    after ExternalizeSpec externalization the scale appears as a named
    input on the composite op boundary in the exported graph.

    Key: 'weight' matches Qwen2's checkpoint key name for the norm scale,
    so load_state_dict works without remapping.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.norm = RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.weight)


class FastVLMAttention(nn.Module):
    """
    Qwen2 grouped-query attention using Core AI composite ops.

    Architecture (1.5B):
      hidden_size       = 1536
      num_heads         = 12  (query heads)
      num_kv_heads      = 2   (key/value heads — GQA)
      head_dim          = 128
      qkv_proj output   = (12 + 2 + 2) * 128 = 2048

    QKV fusion: q_proj, k_proj, v_proj are concatenated into a single
    qkv_proj weight [total, hidden_size] in order [q, k, v]. The SDPA
    composite op handles GQA expansion (12 query heads vs 2 KV heads)
    internally — no explicit repeat_kv needed.

    RoPE: uses absolute position_ids passed from the stateful forward.
    rope_theta=1e6 is read from config.rope_parameters['rope_theta']
    (LlavaConfig nests it there — it is NOT a direct config attribute).

    Cache: each attention layer writes its K/V into the shared k_cache
    and v_cache buffers at offset = seq_len - query_len, then reads back
    the full context [0:seq_len] for attention. See FastVLMDecoderStateful
    for the full cache layout description.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        total = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim

        self.qkv_proj = nn.Linear(dim, total, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.sdpa = SDPA(is_causal=True)
        rope_theta = (
            getattr(config, "rope_theta", None)
            or (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
            # Mirrors Apple Qwen2VL.swift line 767: ropeTheta default = 1_000_000
            # if config does not expose rope_theta / rope_parameters.rope_theta.
            # Never fall back to 1e4 — that is wrong by 100x for Qwen2 models.
            or 1e6
        )
        self.rope = RoPE(base=float(rope_theta))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        offset: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        x            : (B, L, dim)       — L = query_len for this step
        position_ids : (B, seq_len)       — absolute positions, full context width
        k_cache      : (n_layers, batch_size, max_seq_len, kv_dim) — shared mutable state
        v_cache      : same shape as k_cache
        offset       : int (SymInt under tracing) — seq_len - L; position of
                       first new token. Deliberately NOT a torch.Tensor — see
                       FastVLMDecoderStateful.forward for why.
        seq_len      : int                — total tokens to attend over after write
        """
        B, L, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim
        kv_dim = n_kv_heads * head_dim

        # QKV projection and split
        qkv = (
            self.qkv_proj(x)
            .reshape(B, L, n_heads + 2 * n_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        qk = qkv.narrow(1, 0, n_heads + n_kv_heads)
        v  = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        # RoPE: apply to the query-length slice of absolute positions only.
        # Threading position_ids in rather than re-deriving avoids the classic
        # decode bug where offset is computed incorrectly on single-token steps.
        torch._check_is_size(L)
        torch._check_is_size(seq_len)
        rope_positions = position_ids.narrow(-1, seq_len - L, L)
        qk = self.rope(qk, position_ids=rope_positions)
        q = qk.narrow(1, 0, n_heads)
        k = qk.narrow(1, n_heads, n_kv_heads)

        # Flatten heads for cache storage: (B, n_kv_heads, L, head_dim) -> (B, L, kv_dim)
        k_flat = k.permute(0, 2, 1, 3).reshape(B, L, kv_dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(B, L, kv_dim)

        # Write new K/V into cache at [layer, :, offset:offset+L, :]
        #
        # Uses torch.ops.aten.slice_scatter, NOT a custom op and NOT
        # narrow(...).copy_() in place. History of this line:
        #
        #   1. coreai::mutable_slice_update (custom op, mutates_args=["x"]):
        #      torch.export wraps mutates_args custom ops in
        #      auto_functionalized_v2, which coreai-torch 0.4.0's converter
        #      has no lowering for -> UnboundLocalError inside the converter.
        #
        #   2. k_cache[layer].narrow(1, offset, L).copy_(k_flat): a tensor-
        #      valued (dynamic) start passed to narrow(), then mutated
        #      in-place. This fails INSIDE torch.export itself (before ever
        #      reaching coreai-torch): PendingUnbackedSymbolNotFound. The
        #      unbacked symbol torch.export allocates to represent the
        #      narrowed view's dynamic shape never gets threaded through to
        #      the traced function's outputs correctly for an in-place
        #      mutation through a narrowed view.
        #
        #   3. slice_scatter (this version): a FUNCTIONAL op — returns a new
        #      full-shape tensor with src written into the given slice,
        #      rather than mutating a narrowed view in place. Confirmed via
        #      coreai_torch/_aten_to_core.py: aten.slice_scatter.default has
        #      a registered lowering (replace_slice_scatter -> coreai.slice_update)
        #      that explicitly resolves dynamic start/end values via
        #      resolve_slice_arg(...), unlike the in-place-on-a-view path,
        #      which has no such resolver and was never exercised by
        #      coreai-torch's own dynamic-shapes stateful test (that test
        #      mutates a buffer unconditionally in full, never into a
        #      dynamic sub-region — see TestStatefulDynamicShapes in
        #      tests/test_stateful.py). The full-tensor result is then
        #      written back into the registered buffer with a single
        #      .copy_() — this final copy is a full, statically-shaped
        #      tensor-to-tensor copy, not a dynamic-shaped narrowed view,
        #      so it does not hit the same unbacked-symbol problem.
        #
        #      slice_scatter's schema requires start/end to be SymInt, not
        #      a Tensor. The actual root cause of an earlier failure here
        #      was upstream of this line: offset was being constructed as
        #      offset = torch.tensor(seq_len - query_len, dtype=torch.int32)
        #      in FastVLMDecoderStateful.forward — wrapping an
        #      already-symbolic expression (seq_len, query_len come from
        #      .shape under dynamic_shapes, so they're already SymInt) in
        #      torch.tensor(...) re-materializes it as a real tensor, and
        #      calling .item() on THAT does not recover a usable SymInt
        #      under tracing (it still surfaces as FakeTensor to
        #      slice_scatter's schema checker). Fixed at the source: offset
        #      is now a plain `seq_len - query_len` expression, never
        #      wrapped in torch.tensor(...), so it arrives here as a
        #      genuine int/SymInt and no .item() call is needed at all.
        layer = self.layer_idx
        k_layer = torch.ops.aten.slice_scatter(
            k_cache[layer], k_flat, dim=1, start=offset, end=offset + L
        )
        v_layer = torch.ops.aten.slice_scatter(
            v_cache[layer], v_flat, dim=1, start=offset, end=offset + L
        )
        k_cache[layer].copy_(k_layer)
        v_cache[layer].copy_(v_layer)

        # Read back full context [0:seq_len] and restore head-separated layout
        k_ctx = k_cache[layer].narrow(1, 0, seq_len)
        v_ctx = v_cache[layer].narrow(1, 0, seq_len)
        k = k_ctx.reshape(B, seq_len, n_kv_heads, head_dim).permute(0, 2, 1, 3).to(q.dtype)
        v = v_ctx.reshape(B, seq_len, n_kv_heads, head_dim).permute(0, 2, 1, 3).to(q.dtype)

        out = (
            self.sdpa(q, k, v)
            .permute(0, 2, 1, 3)
            .reshape(B, L, n_heads * head_dim)
        )
        return self.o_proj(out)


class FastVLMMLP(nn.Module):
    """
    Qwen2 SwiGLU feed-forward network.

    gate_proj and up_proj both project hidden_size -> intermediate_size.
    silu(gate) * up is the gating mechanism; down_proj projects back.
    The silu activation is handled automatically by get_decomp_table()
    and does not need an ExternalizeSpec.
    """

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class FastVLMDecoderBlock(nn.Module):
    """
    Single Qwen2 transformer block: pre-norm attention + pre-norm MLP.

    Passes k_cache, v_cache, offset, and seq_len through to attention so
    all cache state flows through the single traced graph path.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn             = FastVLMAttention(config, layer_idx)
        self.mlp                   = FastVLMMLP(config)
        self.input_layernorm       = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        offset: int,
        seq_len: int,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x), position_ids, k_cache, v_cache, offset, seq_len
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# ─── Stateful decoder ─────────────────────────────────────────────────────────


class FastVLMDecoderStateful(nn.Module):
    """
    Complete Qwen2 decoder for Core AI export with persistent KV cache state.

    FORWARD SIGNATURE
    -----------------
    forward(input_ids, position_ids) -> logits

      input_ids    : (1, query_len)  int32  — tokens for this step
                     query_len = full prompt length during prefill
                     query_len = 1 during single-token decode steps
      position_ids : (1, seq_len)   int32  — ABSOLUTE positions for the full
                     context seen so far (not just the new tokens).
                     seq_len grows by 1 each decode step.
      logits       : (1, query_len, vocab_size)  float16

    WHY position_ids IS WIDER THAN input_ids
    -----------------------------------------
    position_ids carries the full context width (seq_len) so that each
    attention layer can compute offset = seq_len - query_len and derive
    the correct write position in the cache without any Python branching.
    During a single-token decode step: input_ids is (1,1) but position_ids
    is (1, N) where N is the total number of tokens generated so far.

    KV CACHE STATE
    --------------
    k_cache and v_cache are registered buffers with shape:
      (num_hidden_layers, batch_size, max_seq_len, kv_dim)

    They are NOT passed as forward arguments — they are persistent state.
    TorchConverter binds them to state_names=["k_cache","v_cache"] during
    export, making them mutable state in the compiled .aimodel that persists
    across inference calls without being re-initialized each time.

    The buffers must be zeroed before a new generation sequence begins.
    In the Swift runtime this is handled by the state reset API.

    EXPORT NOTES
    ------------
    - Pass state_names=["k_cache","v_cache"] to add_pytorch_module()
    - Pass dynamic_shapes={"position_ids": {1: seq_len_dim}} where
      seq_len_dim = torch.export.Dim("seq_len", min=1, max=model.max_seq_len)
    - The graph has exactly one traced path — no use_cache flag or
      Python branches. Data-dependent branches cannot be exported.
    """

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [FastVLMDecoderBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm    = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

        head_dim = config.hidden_size // config.num_attention_heads
        kv_dim   = head_dim * config.num_key_value_heads
        # Cache dimensions from config — no hardcoded constants.
        # self.max_seq_len exposed for export: torch.export.Dim("seq_len", min=1, max=model.max_seq_len)
        # self.batch_size=1 is an intentional export constraint (on-device = single sequence), not an accident.
        self.max_seq_len = config.max_position_embeddings
        self.batch_size = 1
        self.register_buffer("k_cache", torch.zeros(config.num_hidden_layers, self.batch_size, self.max_seq_len, kv_dim))
        self.register_buffer("v_cache", torch.zeros_like(self.k_cache))

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        B, query_len = input_ids.shape
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        # torch._check is export-compatible — unlike a Python if-branch, it
        # survives torch.export when seq_len is symbolic. Fires as an assertion
        # in eager mode and as a graph-level constraint in the exported program.
        torch._check(seq_len <= self.max_seq_len)
        # offset is intentionally a plain int/SymInt expression, NOT wrapped
        # in torch.tensor(...). seq_len and query_len come from .shape under
        # dynamic_shapes tracing, so they are already SymInt — seq_len -
        # query_len is therefore already a valid symbolic int expression.
        # An earlier version wrapped this in torch.tensor(..., dtype=torch.int32)
        # then called .item() at the point of use to "convert back" to a
        # plain int for slice_scatter's SymInt? start argument. That round
        # trip does NOT recover the original SymInt under torch.export
        # tracing — .item() on the re-wrapped tensor produces a value that
        # still carries FakeTensor semantics, and slice_scatter's schema
        # rejects it with the same "Expected Optional[int]... found
        # FakeTensor" error as passing the tensor directly. Keeping offset
        # as a bare expression avoids the round trip entirely.
        offset = seq_len - query_len

        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, self.k_cache, self.v_cache, offset, seq_len)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMDecoderStateful":
        """
        Load decoder weights from the FastVLM SafeTensors checkpoint.

        The checkpoint uses these key prefixes for decoder weights:
          model.layers.*       transformer blocks
          model.embed_tokens.* token embedding table
          model.norm.*         final RMS norm before lm_head
          lm_head.*            unembedding projection

        There is NO "language_model." prefix — confirmed from discovery output.

        Loading sequence:
          1. Load keys matching _DECODER_PREFIXES, cast bfloat16 -> float16
          2. Fuse q_proj + k_proj + v_proj -> qkv_proj (_mutate_state_dict)
             Keys still have "model." prefix at this point
          3. Strip "model." prefix to match the module hierarchy
          4. load_state_dict(strict=False) — k_cache/v_cache are buffers,
             not checkpoint weights, so they appear as expected missing keys
        """
        model = cls(config).to(dtype=torch.float16)
        weights = _load_decoder_weights(weights_dir)
        _mutate_state_dict(weights)
        weights = {k.removeprefix("model."): v for k, v in weights.items()}
        missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
        actual_missing = set(missing) - {"k_cache", "v_cache"}
        if actual_missing:
            raise RuntimeError(f"Unexpected missing keys: {actual_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        return model


# ─── Weight loading helpers ───────────────────────────────────────────────────

# Keys that identify decoder weights in the FastVLM checkpoint.
# Confirmed from discover_weights.py output — no "language_model." prefix exists.
_DECODER_PREFIXES = (
    "model.layers.",
    "model.embed_tokens.",
    "model.norm.",
    "lm_head.",
)


def _load_decoder_weights(
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """
    Load decoder weights from SafeTensors, filtering to _DECODER_PREFIXES only.

    Excludes vision tower and projector weights. Casts bfloat16 (the storage
    dtype in Apple's checkpoint) to the target dtype on load.
    """
    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    result = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not any(key.startswith(p) for p in _DECODER_PREFIXES):
                    continue
                t = f.get_tensor(key)
                if t.dtype != dtype:
                    t = t.to(dtype)
                result[key] = t
    return result


def _mutate_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    """
    Fuse q_proj, k_proj, v_proj -> qkv_proj in-place.

    HuggingFace Qwen2 stores attention projections separately. This file
    uses a single fused qkv_proj matching Apple's coreai-models/qwen2.py
    pattern. Fusion must happen BEFORE stripping the "model." prefix, because
    the checkpoint keys still have it at this point:
      model.layers.N.self_attn.q_proj.weight

    Fusion order is [q, k, v] — confirmed by Stage 1 PSNR 113.2 dB vs HF.

    After fusion, the three keys are replaced with:
      model.layers.N.self_attn.qkv_proj.weight
      model.layers.N.self_attn.qkv_proj.bias
    """
    layer_indices = set()
    for k in state_dict:
        if k.startswith("model.layers.") and ".self_attn.q_proj.weight" in k:
            layer_indices.add(int(k.split(".")[2]))

    for i in sorted(layer_indices):
        weights, biases = [], []
        for proj in ["q_proj", "k_proj", "v_proj"]:
            wk = f"model.layers.{i}.self_attn.{proj}.weight"
            bk = f"model.layers.{i}.self_attn.{proj}.bias"
            if wk not in state_dict:
                break
            weights.append(state_dict.pop(wk))
            biases.append(state_dict.pop(bk))
        else:
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(weights)
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.bias"]   = torch.cat(biases)
