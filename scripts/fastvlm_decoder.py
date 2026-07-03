"""
fastvlm_decoder.py — Re-authored FastVLM language decoder for CoreAI export.

The decoder is Qwen2. References:
  coreai-models/python/src/coreai_models/models/macos/qwen3_vl.py
  coreai-models/python/src/coreai_models/primitives/macos/cache_scatter.py
  coreai-models/python/src/coreai_models/vlm/export.py

KEY ARCHITECTURAL DECISIONS
============================

1. INPUTS_EMBEDS, NOT INPUT_IDS
   The decoder takes pre-computed token embeddings (inputs_embeds), not
   raw token IDs. embed_tokens is exported separately as embed.aimodel.
   This matches Apple's CoreAISequentialVLMEngine contract:
     inputs:  in_embeddings [1, L, hidden]  fp16
              position_ids  [1, S]          int32
     states:  k_cache, v_cache (stateful, dynamic seq dim)
     output:  logits [1, L, vocab_size]     fp16

   The engine calls embed.aimodel to get text embeddings, scatter-merges
   image embeddings at placeholder positions, then passes the merged
   inputs_embeds to this decoder. No image logic lives here.

2. STATEFUL KV CACHE WITH SLICE_SCATTER (ANE-COMPATIBLE)
   mutable_slice_update is rejected by the ANE compiler on multi-layer
   models (confirmed: QwenChatFast README, coreai-models commit #68).
   We use the cache_scatter.py pattern instead: slice_scatter (functional,
   out-of-place) with stateful register_buffer buffers.

   Cache shape: [n_layers, 1, n_kv_heads, max_seq_len, head_dim]
   (Apple's 5D layout from cache.py / cache_scatter.py)

   The sequence dim (axis 3) is declared dynamic (-1 at export) so
   CoreAISequentialVLMEngine allocates GrowingKVCache starting at 256
   tokens and growing 2x on demand.

3. QKV UNFUSED
   q_proj/k_proj/v_proj kept separate to match Apple's MLX weight layout
   and allow independent per-group quantization scales per projection.

4. NO IMAGE LOGIC IN THIS FILE
   Image token injection is handled by the engine's scatter-merge:
   embed_tokens(all_tokens_including_placeholders) then replace placeholder
   positions with projected image embeddings. The decoder is image-agnostic.

FORWARD SIGNATURE
=================
forward(inputs_embeds, position_ids) -> logits

  inputs_embeds : [1, L, hidden]  fp16  — pre-merged text + image embeddings
  position_ids  : [1, S]          int32 — absolute positions (S >= L)
  logits        : [1, L, vocab]   fp16

KV cache is handled as states (k_cache, v_cache) — not explicit inputs/outputs.
"""

import glob
import os

import torch
import torch.nn as nn
from coreai_torch.composite_ops import RMSNormImpl, RoPE, SDPA
from safetensors import safe_open


# ─── KV Cache (slice_scatter functional pattern) ─────────────────────────────

class KVCache:
    """
    Functional KV cache using slice_scatter (out-of-place).

    Mirrors coreai-models/python/src/coreai_models/primitives/macos/cache_scatter.py.
    slice_scatter avoids the mutable_slice_update Metal kernel crash that
    occurs on multi-layer models compiled for ANE.

    Cache shape: [n_layers, 1, n_kv_heads, max_seq_len, head_dim]
    Sequence dim: axis 3 (declared dynamic=-1 at export for GrowingKVCache).
    """

    SEQ_LEN_DIM = 3

    def __init__(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        self._k = k_cache
        self._v = v_cache

    @classmethod
    def seq_len_dim(cls) -> int:
        return cls.SEQ_LEN_DIM

    def update_and_fetch(
        self,
        layer_idx: int,
        offset: int,
        k: torch.Tensor,   # [1, n_kv_heads, query_len, head_dim]
        v: torch.Tensor,
        seq_len: int,
        query_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k/v at offset, return full context [0:seq_len]."""
        layer_k = self._k.narrow(0, layer_idx, 1)
        layer_v = self._v.narrow(0, layer_idx, 1)

        # slice_scatter: functional, out-of-place, ANE-compatible
        updated_k = torch.ops.aten.slice_scatter(
            layer_k, k.unsqueeze(0), dim=-2, start=offset, end=offset + query_len
        )
        updated_v = torch.ops.aten.slice_scatter(
            layer_v, v.unsqueeze(0), dim=-2, start=offset, end=offset + query_len
        )
        self._k = torch.ops.aten.slice_scatter(
            self._k, updated_k, dim=0, start=layer_idx, end=layer_idx + 1
        )
        self._v = torch.ops.aten.slice_scatter(
            self._v, updated_v, dim=0, start=layer_idx, end=layer_idx + 1
        )

        out_k = self._k.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len).squeeze(0)
        out_v = self._v.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len).squeeze(0)
        return out_k, out_v


# ─── Composite op wrappers ────────────────────────────────────────────────────

class FastVLMRMSNorm(nn.Module):
    """
    Holds the learnable scale (gamma) as nn.Parameter and passes it
    explicitly to RMSNormImpl.forward(x, weight).

    Required because RMSNormImpl does NOT hold the scale internally —
    it must be passed as a forward argument so it appears as a graph
    input on the composite op boundary after externalization.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.norm = RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.weight)


class FastVLMAttention(nn.Module):
    """
    Qwen2 GQA attention with stateful KV cache via slice_scatter.

    Cache is passed in from the parent decoder (registered buffers),
    updated functionally, and written back via slice_scatter.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        # Separate projections — NOT fused (independent quantization scales)
        self.q_proj = nn.Linear(dim, q_dim, bias=True)
        self.k_proj = nn.Linear(dim, kv_dim, bias=True)
        self.v_proj = nn.Linear(dim, kv_dim, bias=True)
        self.o_proj = nn.Linear(q_dim, dim, bias=False)

        self.sdpa = SDPA(is_causal=True)
        self.rope = RoPE(base=float(getattr(config, "rope_theta", 1e6)))

    def forward(
        self,
        x: torch.Tensor,               # [B, L, hidden]
        position_ids: torch.Tensor,    # [B, S]
        cache: KVCache,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim
        seq_len = position_ids.shape[-1]
        offset = seq_len - L

        # Project → [B, n_heads, L, head_dim]
        q = self.q_proj(x).reshape(B, L, n_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)

        # RoPE on current positions only
        rope_positions = position_ids.narrow(-1, offset, L)
        qk = self.rope(torch.cat([q, k], dim=1), position_ids=rope_positions)
        q = qk.narrow(1, 0, n_heads)
        k = qk.narrow(1, n_heads, n_kv_heads)

        # Update KV cache and fetch full context
        k_ctx, v_ctx = cache.update_and_fetch(
            self.layer_idx, offset, k, v, seq_len=seq_len, query_len=L
        )

        out = (
            self.sdpa(q, k_ctx.to(q.dtype), v_ctx.to(q.dtype))
            .permute(0, 2, 1, 3)
            .reshape(B, L, n_heads * head_dim)
        )
        return self.o_proj(out)


class FastVLMMLP(nn.Module):
    """SwiGLU MLP."""

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
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = FastVLMAttention(config, layer_idx)
        self.mlp = FastVLMMLP(config)
        self.input_layernorm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


# ─── Main decoder model ───────────────────────────────────────────────────────

class FastVLMDecoder(nn.Module):
    """
    Qwen2 decoder for CoreAI export.

    Takes inputs_embeds (pre-merged text + image embeddings from the engine),
    not input_ids. embed_tokens lives in embed.aimodel.

    KV cache: stateful register_buffer, slice_scatter updates, dynamic seq dim.
    The engine (CoreAISequentialVLMEngine) allocates and grows the cache.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [FastVLMDecoderBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # KV cache state buffers.
        # Shape: [n_layers, 1, n_kv_heads, max_seq_len, head_dim]
        # max_seq_len is declared dynamic (-1) at export via dynamic_shapes,
        # so the compiled state descriptor shows seq_dim < 0 → GrowingKVCache.
        n_layers   = config.num_hidden_layers
        n_kv_heads = config.num_key_value_heads
        head_dim   = config.hidden_size // config.num_attention_heads
        # Use 0 as placeholder — actual size set by dynamic_shapes at export
        self.register_buffer(
            "k_cache",
            torch.zeros(n_layers, 1, n_kv_heads, 0, head_dim, dtype=torch.float16),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros_like(self.k_cache),
            persistent=False,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,    # [1, L, hidden]  fp16
        position_ids: torch.Tensor,     # [1, S]          int32  S >= L
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: Pre-merged text + image embeddings from engine.
                           Shape [1, L, hidden_size], float16.
            position_ids:  Absolute position indices for full context.
                           Length S >= L (S = processedTokenCount + L).

        Returns:
            logits: [1, L, vocab_size] float16
        """
        h = inputs_embeds
        cache = KVCache(self.k_cache, self.v_cache)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        # Write updated cache tensors back to registered buffers
        self.k_cache = cache._k
        self.v_cache = cache._v
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMDecoder":
        """Load from SafeTensors, excluding embed_tokens (lives in embed.aimodel)."""
        model = cls(config).to(dtype=torch.float16)
        weights = _load_decoder_weights(weights_dir)
        weights = {k.removeprefix("model."): v for k, v in weights.items()}
        missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
        actual_missing = set(missing) - {"k_cache", "v_cache"}
        if actual_missing:
            raise RuntimeError(f"Missing keys: {actual_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")
        return model


# ─── Embed tokens model (exported separately as embed.aimodel) ────────────────

class FastVLMEmbedTokens(nn.Module):
    """
    Token embedding lookup, exported as embed.aimodel.

    Input:  input_ids   int32 [1, seq_len]
    Output: embeddings  fp16  [1, seq_len, hidden_size]

    Uses direct weight indexing (weight[input_ids]) rather than nn.Embedding
    to ensure Int32 indices work cleanly — nn.Embedding requires Int64.

    The engine calls this to embed all tokens (including image placeholders),
    then scatter-merges projected image embeddings at the placeholder positions
    before calling the decoder with the merged inputs_embeds.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [1, L] int32 → embeddings: [1, L, hidden_size] fp16
        return self.weight[input_ids]

    @classmethod
    def from_weights(cls, weights_dir: str) -> "FastVLMEmbedTokens":
        """Load embed_tokens weight from SafeTensors."""
        import glob
        st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
        for path in st_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "embed_tokens" in key:
                        weight = f.get_tensor(key).to(torch.float16)
                        return cls(weight)
        raise FileNotFoundError(f"embed_tokens not found in {weights_dir}")


# Backward-compat alias
FastVLMDecoderStateful = FastVLMDecoder


# ─── Weight loading helpers ───────────────────────────────────────────────────

_DECODER_PREFIXES = (
    "model.layers.",
    "model.norm.",
    "lm_head.",
    # Note: model.embed_tokens.* intentionally excluded — lives in embed.aimodel
)


def _load_decoder_weights(
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """
    Load decoder weights from SafeTensors.
    Excludes embed_tokens (exported separately as embed.aimodel).
    Casts bfloat16 (HF storage dtype) to fp16.
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
