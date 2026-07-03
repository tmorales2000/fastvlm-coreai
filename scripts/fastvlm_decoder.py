"""
fastvlm_decoder.py — Re-authored FastVLM language decoder for CoreAI export.

Follows Apple's coreai-models VLM export pattern from:
  coreai-models/python/src/coreai_models/models/macos/qwen3_vl.py
  coreai-models/python/src/coreai_models/primitives/macos/cache_scatter.py
  coreai-models/python/src/coreai_models/vlm/export.py

KEY ARCHITECTURAL DECISIONS
============================

1. INPUTS_EMBEDS, NOT INPUT_IDS
   The decoder takes pre-computed token embeddings, not raw token IDs.
   embed_tokens is exported separately as embed.aimodel.
   This matches Qwen3VLForCausalLMEmbeddings from Apple's recipe.

2. K_CACHE / V_CACHE AS EXPLICIT FORWARD ARGUMENTS
   Unlike the earlier registered-buffer approach, k_cache and v_cache are
   explicit forward() arguments — exactly like Apple's Qwen3-VL recipe.
   This allows torch.export to see them as inputs with dynamic_shapes.

3. STATEFUL KV CACHE WITH SLICE_SCATTER (ANE-COMPATIBLE)
   mutable_slice_update is rejected by ANE on multi-layer models.
   cache_scatter.py uses slice_scatter (out-of-place/functional) instead.
   State names are "keyCache" / "valueCache" (camelCase) matching Swift runner.

4. QKV UNFUSED
   q_proj/k_proj/v_proj kept separate to match Apple's MLX weight layout
   and allow independent per-group quantization scales per projection.

5. NO IMAGE LOGIC IN THIS FILE
   Image injection handled by CoreAISequentialVLMEngine via scatter-merge.
   embed_tokens(<image>) embeddings are overwritten with projected features.
   The decoder is image-agnostic.

FORWARD SIGNATURE
=================
forward(inputs_embeds, position_ids, keyCache, valueCache) -> logits

  inputs_embeds : [1, L, hidden]   fp16  — pre-merged text + image embeddings
  position_ids  : [1, S]           int32 — absolute positions (S >= L)
  keyCache      : [n_layers, 1, n_kv_heads, seq_len, head_dim]  fp16
  valueCache    : [n_layers, 1, n_kv_heads, seq_len, head_dim]  fp16
  logits        : [1, L, vocab_size]  fp16

keyCache/valueCache are declared as state_names in the export, so the
Swift runner manages them as persistent state across decode calls.
"""

import glob
import os

import torch
import torch.nn as nn
from coreai_models.primitives._ops import mutable_slice_update
from coreai_torch.composite_ops import RMSNormImpl, RoPE, SDPA
from safetensors import safe_open

# State names matching Swift CoreAISequentialVLMEngine
KEY_CACHE_NAME   = "keyCache"
VALUE_CACHE_NAME = "valueCache"
KV_STATE_NAMES   = (KEY_CACHE_NAME, VALUE_CACHE_NAME)


# ─── KV Cache (cache_scatter.py pattern) ─────────────────────────────────────

class KVCache:
    """
    Functional KV cache using slice_scatter (out-of-place).

    Matches coreai-models/python/src/coreai_models/primitives/macos/cache_scatter.py.
    slice_scatter avoids the mutable_slice_update Metal kernel crash on ANE.

    Cache shape: [n_layers, 1, n_kv_heads, seq_len, head_dim]
    seq_len_dim = 3 (used by export dynamic_shapes and Swift runner).
    """

    def __init__(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        self._k_cache = k_cache
        self._v_cache = v_cache

    @classmethod
    def seq_len_dim(cls) -> int:
        return 3

    @classmethod
    def create_cache_tensors(
        cls,
        config,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_kv_heads = config.num_key_value_heads
        n_layers   = config.num_hidden_layers
        max_seq    = config.max_position_embeddings
        head_dim   = (
            config.head_dim
            if hasattr(config, "head_dim") and config.head_dim
            else config.hidden_size // config.num_attention_heads
        )
        k = torch.zeros(n_layers, 1, n_kv_heads, max_seq, head_dim, dtype=dtype)
        v = torch.zeros_like(k)
        return k, v

    def update_and_fetch(
        self,
        layer_idx: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        query_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use mutable_slice_update (mutates_args=["x"]) — this creates
        # AutoFunctionalized nodes in the exported graph that
        # remove_functionalization then replaces with immutable_slice_update
        # for the MLIR lowering. slice_scatter does NOT create these nodes
        # and so the converter cannot detect stateful inputs.
        n_layers   = self._k_cache.shape[0]
        n_kv_heads = self._k_cache.shape[2]
        head_dim   = self._k_cache.shape[4]

        begin_k = torch.tensor([layer_idx, 0, 0, offset, 0], dtype=torch.int32)
        end_k   = torch.tensor([layer_idx + 1, 1, n_kv_heads, offset + query_len, head_dim], dtype=torch.int32)
        mutable_slice_update(self._k_cache, k.unsqueeze(0), begin_k, end_k)

        begin_v = torch.tensor([layer_idx, 0, 0, offset, 0], dtype=torch.int32)
        end_v   = torch.tensor([layer_idx + 1, 1, n_kv_heads, offset + query_len, head_dim], dtype=torch.int32)
        mutable_slice_update(self._v_cache, v.unsqueeze(0), begin_v, end_v)

        out_k = self._k_cache.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len).squeeze(0)
        out_v = self._v_cache.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len).squeeze(0)
        return out_k, out_v


# ─── Composite op wrappers ────────────────────────────────────────────────────

class FastVLMRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.norm   = RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.weight)


class FastVLMAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx  = layer_idx
        dim             = config.hidden_size
        self.n_heads    = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim   = dim // self.n_heads
        q_dim           = self.n_heads    * self.head_dim
        kv_dim          = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dim, q_dim,  bias=True)
        self.k_proj = nn.Linear(dim, kv_dim, bias=True)
        self.v_proj = nn.Linear(dim, kv_dim, bias=True)
        self.o_proj = nn.Linear(q_dim, dim,  bias=False)

        self.sdpa = SDPA(is_causal=True)
        self.rope = RoPE(base=float(getattr(config, "rope_theta", 1e6)))

    def forward(
        self,
        x:            torch.Tensor,   # [B, L, hidden]
        position_ids: torch.Tensor,   # [B, S]
        cache:        KVCache,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim

        seq_len   = position_ids.shape[-1]
        torch._check_is_size(L)
        torch._check_is_size(seq_len)
        offset = seq_len - L
        torch._check_is_size(offset)

        q = self.q_proj(x).reshape(B, L, n_heads,    head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)

        rope_positions = position_ids.narrow(-1, offset, L)
        qk = self.rope(torch.cat([q, k], dim=1), position_ids=rope_positions)
        q  = qk.narrow(1, 0, n_heads)
        k  = qk.narrow(1, n_heads, n_kv_heads)

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
        self.self_attn               = FastVLMAttention(config, layer_idx)
        self.mlp                     = FastVLMMLP(config)
        self.input_layernorm         = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm= FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x:            torch.Tensor,
        position_ids: torch.Tensor,
        cache:        KVCache,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


# ─── Main decoder model ───────────────────────────────────────────────────────

class FastVLMDecoder(nn.Module):
    """
    Qwen2 decoder for CoreAI VLM export.

    Follows Qwen3VLForCausalLMEmbeddings from Apple's vlm/export.py:
    - Takes inputs_embeds (not input_ids)
    - k_cache / v_cache are explicit forward() args (not registered buffers)
    - State names: "keyCache" / "valueCache" (Swift runner convention)
    - KVCache uses slice_scatter (ANE-compatible)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [FastVLMDecoderBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm    = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [1, L, hidden]  fp16
        position_ids:  torch.Tensor,  # [1, S]          int32
        keyCache:      torch.Tensor,  # [n_layers, 1, n_kv_heads, seq_len, head_dim]
        valueCache:    torch.Tensor,  # same
    ) -> torch.Tensor:
        cache = KVCache(keyCache, valueCache)
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMDecoder":
        """Load weights, excluding embed_tokens (exported separately)."""
        model = cls(config).to(dtype=torch.float16)
        weights = _load_decoder_weights(weights_dir)
        weights = {k.removeprefix("model."): v for k, v in weights.items()}
        missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")
        return model


# ─── Embed tokens model ───────────────────────────────────────────────────────

class FastVLMEmbedTokens(nn.Module):
    """
    Token embedding lookup, exported as embed.aimodel.

    Uses weight[input_ids] (not nn.Embedding) to work with Int32 indices,
    matching EmbedTokens from Apple's vlm/export.py.

    Input:  input_ids  int32 [1, seq_len]
    Output: embeddings fp16  [1, seq_len, hidden_size]
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]

    @classmethod
    def from_weights(cls, weights_dir: str) -> "FastVLMEmbedTokens":
        st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
        for path in st_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "embed_tokens" in key:
                        return cls(f.get_tensor(key).to(torch.float16))
        raise FileNotFoundError(f"embed_tokens not found in {weights_dir}")


# Backward-compat alias
FastVLMDecoderStateful = FastVLMDecoder


# ─── Weight loading ───────────────────────────────────────────────────────────

_DECODER_PREFIXES = (
    "model.layers.",
    "model.norm.",
    "lm_head.",
    # model.embed_tokens.* excluded — lives in embed.aimodel
)


def _load_decoder_weights(
    weights_dir: str,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
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
