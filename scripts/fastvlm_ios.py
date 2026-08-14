"""
fastvlm_ios.py — FastVLM Qwen2 decoder re-authored for iOS / ANE.

Follows the exact pattern of coreai-models/python/src/coreai_models/models/ios/qwen3.py.
Provides FastVLMForiOS, suitable for export via export_ios_model() from the
coreai-models iOS export pipeline.

Tensor layout: (batch_size, seq_len, 1, hidden_size) — the iOS convention.
Attention internally converts to (batch_size, n_heads*head_dim, 1, seq_len) for
KV cache operations and SDPA, then converts back.

KV cache shape: [n_layers, 1, n_kv_heads*head_dim, 1, max_ctx]
  Sequence on dim 4, KV heads and head_dim fused on dim 2.
  Managed by KVCacheHandler using the new mutable_cache_update_and_fetch op (PR #126).

iOS entrypoints (four, as required by BaseForCausalLMForiOS):
  load_embeddings  — returns the embedding table (int8 quantized by default)
  gather_embeddings — dequantizes and gathers embeddings for input_ids
  extend            — transformer decode step (prefill_mode=False)
  prompt_opt        — same callable as extend with prefill_mode=True

Weight loading:
  _mutate_state_dict() reshapes all attention/MLP weights to Conv2d format
  [out, in] → [out, in, 1, 1] and handles embedding quantization, mirroring
  Qwen3ForCausalLMForiOS exactly.

FastVLM specifics vs plain Qwen3:
  - Base model is Qwen2 (0.5B, 1.5B) or Qwen2.5 (7B), not Qwen3
  - No qk_norm layers (Qwen2 uses standard attention without q/k normalization)
  - Weight key prefix is model.layers.* (no fused qkv_proj — kept separate)
  - image_token_id: 151646 (0.5B/1.5B) or 151665 (7B)
  - Image embeddings are scatter-merged before this decoder runs
    (same as macOS path — this decoder receives pre-merged inputs_embeds)

Export usage:
  from coreai_models.export.ios import export_ios_model
  from coreai_models.export.pipeline import ExportConfig

  model = FastVLMForiOS.from_hf_memory_efficient(
      "weights/fastvlm-0.5b",
      max_context_length=4096,
  )
  export_ios_model(model, config, ExportConfig(
      hf_model_id="apple/FastVLM-0.5B",
      variant="iOS",
      max_context_length=4096,
      compression="4bit_weight_palettized_group32",
  ))
"""

import torch
import torch.nn as nn
from transformers import AutoConfig

from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.mlp import MLP
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA

from fastvlm_decoder import _load_decoder_weights


# ── Attention ─────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head attention in iOS layout: (B, S, 1, D) in/out.

    Conv2d projections operate on (B*S, D, 1, 1) — kernel_size=1 is a linear
    map that's ANE-native. RoPE and KV cache update happen in
    (B, n_heads*head_dim, 1, S) layout (BC1S); SDPA is per-head.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads    = n_heads    = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim   = head_dim   = getattr(config, "head_dim", dim // n_heads)

        # Conv2d(kernel_size=1) — ANE-native projection
        self.q_proj = nn.Conv2d(dim, n_heads    * head_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(n_heads * head_dim, dim,    kernel_size=1, bias=False)

        # No qk_norm for Qwen2 (unlike Qwen3)
        self.sdpa = SDPA(head_dim=head_dim)

    def forward(
        self,
        x: torch.Tensor,            # (B, S, 1, D)
        rope_cos: torch.Tensor,     # (S, head_dim)
        rope_sin: torch.Tensor,     # (S, head_dim)
        in_step: torch.IntTensor,   # scalar — current KV cache write position
        causal_mask: torch.Tensor,  # (1, max_ctx, 1, S)
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:              # (B, S, 1, D)
        batch_size, query_len, _, hidden_size = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        # (B, S, 1, D) → (B*S, D, 1, 1) for Conv2d
        x_conv = x.reshape(batch_size * query_len, hidden_size, 1, 1)
        query = self.q_proj(x_conv)      # (B*S, n_heads*head_dim, 1, 1)
        key   = self.k_proj(x_conv)      # (B*S, n_kv_heads*head_dim, 1, 1)
        value = self.v_proj(x_conv)      # (B*S, n_kv_heads*head_dim, 1, 1)

        # Reshape to (B, S, n_heads, head_dim) for RoPE
        query = (
            query.reshape(batch_size, query_len, n_heads, self.head_dim)
        )
        key = (
            key.reshape(batch_size, query_len, n_kv_heads, self.head_dim)
        )

        # Apply RoPE — operates on (..., head_dim) last dim
        query = apply_rope(query, rope_cos, rope_sin)
        key   = apply_rope(key,   rope_cos, rope_sin)

        # Convert to BC1S: (B, n_heads*head_dim, 1, S) for KV cache + SDPA
        query = (
            query.reshape(batch_size, query_len, 1, n_heads * self.head_dim)
            .transpose(-3, -1)
        )  # (B, n_heads*head_dim, 1, S)
        key = (
            key.reshape(batch_size, query_len, 1, n_kv_heads * self.head_dim)
            .transpose(-3, -1)
        )  # (B, n_kv_heads*head_dim, 1, S)
        value = (
            value.reshape(batch_size * query_len, n_kv_heads, self.head_dim, 1)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, query_len, 1, n_kv_heads * self.head_dim)
            .transpose(-3, -1)
        )  # (B, n_kv_heads*head_dim, 1, S)

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, in_step, key, value, query_len
            )

        # SDPA: (B, n_heads*head_dim, 1, S) → same
        output = self.sdpa(query, key, value, causal_mask)

        # Back to (B, S, 1, D) for residual
        output = (
            output.transpose(-3, -1)               # (B, S, 1, n_heads*head_dim)
            .reshape(batch_size * query_len, n_heads * self.head_dim, 1, 1)
        )
        output = self.o_proj(output)               # (B*S, D, 1, 1)
        return output.reshape(batch_size, query_len, 1, hidden_size)


# ── Transformer block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(dim=hidden_size, hidden_dim=config.intermediate_size)
        self.input_layernorm           = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm  = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x),
            rope_cos, rope_sin, in_step, causal_mask, cache,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# ── Qwen2 model stack ──────────────────────────────────────────────────────────

class FastVLMModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            token_embeddings = layer(
                token_embeddings, rope_cos, rope_sin, in_step, causal_mask, cache
            )
        return self.norm(token_embeddings)


# ── FastVLMExtend: the exported decoder callable ───────────────────────────────

class FastVLMExtend(nn.Module):
    """The iOS decoder callable — exported as both `extend` and `prompt_opt`.

    Accepts pre-merged inputs (image + text embeddings already scatter-merged
    before this callable runs — same as the macOS export path).

    forward(transformer_input, position_ids, in_step, causal_mask,
            key_cache, value_cache, embedding_table) -> logits

    prefill_mode=True (prompt_opt): returns KV cache sentinel instead of logits.
    prefill_mode=False (extend): returns logits.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.model = FastVLMModel(config)

        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale      = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

        self.prefill_mode = False

        # lm_head or tied embeddings — Qwen2 doesn't tie by default
        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.kv_cache = KVCacheHandler(config.num_hidden_layers, config.hidden_size)

        head_dim    = getattr(config, "head_dim",
                              config.hidden_size // config.num_attention_heads)
        rope_theta  = float(getattr(config, "rope_theta", 10000.0))
        self.rope   = RoPECache(head_dim, config.max_position_embeddings, rope_theta)

    def forward(
        self,
        transformer_input: torch.Tensor,   # (B, S, 1, hidden) — pre-merged embeds
        position_ids: torch.IntTensor,     # (B, S)
        in_step: torch.IntTensor,          # scalar int32
        causal_mask: torch.Tensor,         # (1, max_ctx, 1, S)
        key_cache: torch.Tensor,           # [n_layers, 1, n_kv_heads*head_dim, 1, max_ctx]
        value_cache: torch.Tensor,         # same
        embedding_table: torch.Tensor | None = None,  # for tied lm_head
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        rope_cos, rope_sin = self.rope.gather_cos_sin(position_ids)

        batch_size, seq_len, _, hidden_dim = transformer_input.shape
        out = self.model(
            transformer_input, rope_cos, rope_sin,
            in_step, causal_mask, self.kv_cache,
        )

        if self.prefill_mode:
            # prompt_opt: return KV cache sentinel (signals prefill completion to runtime)
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]

        if self.lm_head is not None:
            # Non-tied: standard lm_head (B, S, 1, D) → (B, S, 1, D) → (B, S, vocab)
            return self.lm_head(out.transpose(-2, -3))

        # Tied embeddings: embedding_table @ out (same as Qwen3)
        if embedding_table is not None and embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table, self.emb_scale, self.emb_zero_point, out.dtype
            )
        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )
        out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)
        return (embedding_table @ out).transpose(-2, -1)


# ── Top-level iOS model class ──────────────────────────────────────────────────

class FastVLMForiOS(BaseForCausalLMForiOS):
    """FastVLM Qwen2 decoder for iOS export.

    Inherits the three-module structure from BaseForCausalLMForiOS:
      self.load_embeddings   — LoadEmbeddings (int8 embedding table)
      self.gather_embeddings — GatherEmbeddings (dequant + gather)
      self.extend            — FastVLMExtend (the transformer)

    These map to the four iOS entrypoints:
      load_embeddings  → self.load_embeddings.forward()
      gather_embeddings → self.gather_embeddings.forward()
      extend           → self.extend.forward()      (prefill_mode=False)
      prompt_opt       → self.extend.forward()      (prefill_mode=True)

    The five export-contract hooks are INHERITED from BaseForCausalLMForiOS
    and return the correct names/shapes for the standard iOS LLM contract.
    No overrides needed for FastVLM — the hook defaults match our decoder.

    VLM note: This decoder receives pre-merged inputs_embeds (image tokens
    scatter-merged into text token positions). The vision encoder and projector
    run separately (vision.aimodel) before this decoder is called. The iOS VLM
    export pipeline will need to handle the vision components separately —
    this class covers the decoder only.
    """

    # HF model class for weight loading via BaseForCausalLMForiOS.from_hf_memory_efficient
    # Using trust_remote_code for LlavaQwenForCausalLM
    _HF_MODEL_CLASS = None  # Set dynamically — FastVLM uses trust_remote_code

    def _init_model(self, config) -> None:
        self.extend = FastVLMExtend(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Eager-mode forward for testing/verification (never exported directly).

        Composes the three callables: load → gather → extend.
        """
        token_embeddings = self.gather_embeddings(
            input_ids, self.load_embeddings.embedding_table
        )
        return self.extend(
            token_embeddings, position_ids, in_step,
            causal_mask, key_cache, value_cache,
            self.load_embeddings.embedding_table,
        )

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Reshape weights for iOS Conv2d layout and handle embeddings.

        Mirrors Qwen3ForCausalLMForiOS._mutate_state_dict exactly, adapted
        for Qwen2's separate q/k/v projections (no fused qkv_proj).

        Transformations:
          1. Attention/MLP Linear weights [out, in] → Conv2d [out, in, 1, 1]
          2. embed_tokens.weight [vocab, hidden] → [vocab, 1, hidden] (int8 quantized)
          3. Key remapping: model.* → extend.model.*
          4. embed_tokens → load_embeddings.embedding_table
          5. scale/zero_point set on gather_embeddings and extend
        """
        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if not k.startswith("model.layers.") or len(name_split) < 4:
                continue
            max_layer = max(max_layer, int(name_split[2]))

        if max_layer < 0:
            raise ValueError("invalid state_dict: no transformer layers found")

        for i in range(max_layer + 1):
            # Reshape attention weights for Conv2d: [out, in] → [out, in, 1, 1]
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = f"model.layers.{i}.self_attn.{proj}.weight"
                if key in state_dict:
                    state_dict[key] = state_dict[key].unsqueeze(-1).unsqueeze(-1)

            # Reshape MLP weights for Conv2d
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"model.layers.{i}.mlp.{proj}.weight"
                if key in state_dict:
                    state_dict[key] = state_dict[key].unsqueeze(-1).unsqueeze(-1)

        # Embedding table: [vocab, hidden] → [vocab, 1, hidden], then int8 quantize
        embed_key = "model.embed_tokens.weight"
        if embed_key in state_dict:
            embedding_table = state_dict[embed_key].unsqueeze(1)  # [vocab, 1, hidden]
            if not self.disable_embedding_quantization:
                embedding_table, scale, zero_point = quantize_per_tensor(
                    embedding_table, nbits=8, symmetric=True
                )
            else:
                scale       = torch.tensor(1.0, dtype=embedding_table.dtype)
                zero_point  = torch.tensor(0, dtype=torch.int8)

            state_dict["load_embeddings.embedding_table"] = embedding_table
            state_dict["gather_embeddings.scale"]         = scale
            state_dict["gather_embeddings.zero_point"]    = zero_point
            state_dict["extend.emb_scale"]                = scale
            state_dict["extend.emb_zero_point"]           = zero_point
            state_dict.pop(embed_key)

        # Remap model.* → extend.model.* (FastVLMExtend wraps FastVLMModel)
        keys_to_remap = {
            k: f"extend.{k}"
            for k in list(state_dict)
            if k.startswith("model.") and "gather_embeddings" not in k
        }
        for old_key, new_key in keys_to_remap.items():
            state_dict[new_key] = state_dict.pop(old_key)

        # lm_head
        if "lm_head.weight" in state_dict:
            if not getattr(self.config, "tie_word_embeddings", False):
                state_dict["extend.lm_head.weight"] = state_dict.pop("lm_head.weight")
            else:
                state_dict.pop("lm_head.weight", None)

    @classmethod
    def from_weights_dir(
        cls,
        weights_dir: str,
        max_context_length: int = 4096,
        dtype: torch.dtype = torch.float16,
        disable_embedding_quantization: bool = False,
    ) -> "FastVLMForiOS":
        """Load FastVLMForiOS from a local weights directory.

        Uses _load_decoder_weights from fastvlm_decoder.py which correctly
        handles FastVLM's LlavaQwen checkpoint structure.

        Args:
            weights_dir: Path to local HF weights (e.g. 'weights/fastvlm-0.5b')
            max_context_length: Max context for RoPE and KV cache allocation
            dtype: Weight dtype (default: float16)
            disable_embedding_quantization: Keep embedding table in float16
        """
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(weights_dir, trust_remote_code=True)
        text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None) or cfg
        text_cfg.max_position_embeddings = max_context_length

        model = cls(text_cfg)
        model.disable_embedding_quantization = disable_embedding_quantization

        # Load decoder weights (excludes vision tower and embed_tokens)
        weights = _load_decoder_weights(weights_dir, dtype=dtype)

        # Load embed_tokens separately (excluded from _load_decoder_weights)
        import glob
        import safetensors.torch as st
        for path in sorted(glob.glob(f"{weights_dir}/*.safetensors")):
            d = st.load_file(path)
            for k, v in d.items():
                if "embed_tokens" in k:
                    weights["model.embed_tokens.weight"] = v.to(dtype)
                    break
            if "model.embed_tokens.weight" in weights:
                break

        # Apply iOS weight mutations (Conv2d reshape, embedding quantization, remapping)
        model._mutate_state_dict(weights)
        model.load_state_dict(weights, strict=False, assign=True)
        return model.eval()
