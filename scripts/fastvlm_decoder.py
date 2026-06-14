"""
fastvlm_decoder.py — Re-authored FastVLM language decoder for CoreAI export.

The decoder is Qwen2. Reference:
  coreai-models/python/src/coreai_models/models/macos/qwen2.py

Key decisions:
  - RMSNormImpl, SDPA, RoPE are composite ops (externalized via ExternalizeSpec)
  - nn.LayerNorm and nn.BatchNorm2d handled automatically by get_decomp_table()
  - KV cache uses register_buffer + state_names (NOT readonly KV cache pattern)
  - QKV projections fused in _mutate_state_dict before weight loading
"""

import glob
import os

import torch
import torch.nn as nn
from coreai_torch.composite_ops import RMSNormImpl, RoPE, SDPA
from safetensors import safe_open
from transformers import AutoConfig

# The in-place state mutation MUST go through mutable_slice_update — the same
# custom op Apple's own KVCache uses (coreai_models.primitives.*.cache). It is a
# registered torch.library op with a fake/meta kernel, so it survives
# run_decompositions(get_decomp_table()) as a recognizable boundary that
# TorchConverter binds to a named state. A plain narrow().copy_() or slice-assign
# is NOT guaranteed to survive decomposition as a state mutation. Per the
# coreai-torch docs: state is whatever the exported graph mutates in place; if the
# mutation disappears, the buffer stops being state.
# Confirmed from coreai_models/primitives/_ops.py:
#   - op namespace: "coreai" (not "fastvlm")
#   - mutates_args=["x"] (not () — this is what makes the exporter see a state mutation)
#   - logic: torch.split + tuple of slices + x[slices] = update + return x.clone()
@torch.library.custom_op("coreai::mutable_slice_update", mutates_args=["x"])
def mutable_slice_update(
    x: torch.Tensor, update: torch.Tensor, begin: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    b = torch.split(begin, 1, dim=0)
    e = torch.split(end, 1, dim=0)
    slices = tuple(slice(bb.item(), ee.item()) for bb, ee in zip(b, e, strict=False))
    x[slices] = update
    return x.clone()

@mutable_slice_update.register_fake
def _mutable_slice_update_fake(x, update, begin, end):
    return torch.empty(x.shape, dtype=x.dtype)

# Maximum context length for static KV cache shape at export time.
# Set from text_config.max_position_embeddings in practice.
MAX_SEQ_LEN = 4096


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

        # Composite ops — preserved by ExternalizeSpec during export
        self.sdpa = SDPA(is_causal=True)
        rope_theta = (
            getattr(config, "rope_theta", None)
            or (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
            or 1e6  # FastVLM Qwen2 default — never fall back to 1e4
        )
        self.rope = RoPE(base=float(rope_theta))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        offset: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        x            : (B, L, dim)  — L is query_len for this call
        position_ids : (B, seq_len) — ABSOLUTE positions for the full context
        k_cache,
        v_cache      : (n_layers, 1, MAX_SEQ_LEN, kv_dim) registered state buffers
        offset       : IntTensor scalar = seq_len - L; where this call's k/v land
        seq_len      : total context length to attend over after the write

        Cache buffer layout is FLATTENED over heads: last dim = n_kv_heads*head_dim,
        sequence on dim 2. Attention works head-SEPARATED: (B, n_kv_heads, L,
        head_dim). Store flattens; read restores. Write and read use the same
        split or the cache corrupts.

        The write goes through mutable_slice_update so the export pipeline sees a
        state mutation it can bind to the k_cache / v_cache state names.
        """
        B, L, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim
        kv_dim = n_kv_heads * head_dim

        qkv = (
            self.qkv_proj(x)
            .reshape(B, L, n_heads + 2 * n_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        qk = qkv.narrow(1, 0, n_heads + n_kv_heads)
        v = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        # Rope on q/k with the query-length slice of absolute positions. Threading
        # position_ids in (rather than re-deriving) avoids the classic decode bug.
        torch._check_is_size(L)
        torch._check_is_size(seq_len)
        rope_positions = position_ids.narrow(-1, seq_len - L, L)
        qk = self.rope(qk, position_ids=rope_positions)
        q = qk.narrow(1, 0, n_heads)
        k = qk.narrow(1, n_heads, n_kv_heads)  # (B, n_kv_heads, L, head_dim)

        # Flatten heads -> (B, L, kv_dim) to match the buffer layout.
        k_flat = k.permute(0, 2, 1, 3).reshape(B, L, kv_dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(B, L, kv_dim)

        device = k_cache.device
        layer = self.layer_idx
        # begin/end over [layer, batch, seq, kv_dim]; mutate the seq window at offset.
        z = torch.zeros(1, dtype=torch.int32, device=device)
        begin = torch.cat([
            torch.tensor([layer], dtype=torch.int32, device=device),
            z, offset.reshape(1).to(torch.int32), z,
        ])
        end = torch.cat([
            torch.tensor([layer + 1], dtype=torch.int32, device=device),
            torch.tensor([k_cache.size(1)], dtype=torch.int32, device=device),
            (offset.reshape(1).to(torch.int32) + L),
            torch.tensor([kv_dim], dtype=torch.int32, device=device),
        ])

        mutable_slice_update(k_cache, k_flat.unsqueeze(0), begin, end)
        mutable_slice_update(v_cache, v_flat.unsqueeze(0), begin, end)

        # Read back the full context [0:seq_len] for this layer, restore heads.
        k_ctx = k_cache[layer].narrow(1, 0, seq_len)  # (1, seq_len, kv_dim)
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
    """SwiGLU MLP. silu is handled automatically by get_decomp_table()."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class FastVLMDecoderBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = FastVLMAttention(config, layer_idx)
        self.mlp = FastVLMMLP(config)
        self.input_layernorm = FastVLMRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = FastVLMRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        offset: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x), position_ids, k_cache, v_cache, offset, seq_len
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# ─── Stateful decoder model ───────────────────────────────────────────────────


class FastVLMDecoderStateful(nn.Module):
    """
    Full Qwen2 decoder re-authored for CoreAI export with KV cache states.

    KV cache is registered as mutable buffers via register_buffer.
    The export pipeline passes state_names=["k_cache", "v_cache"] to
    TorchConverter.add_pytorch_module(), which maps both the input and
    in-place mutation output to a single state name each.
    """

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [FastVLMDecoderBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = FastVLMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

        # KV cache as mutable state buffers
        head_dim = config.hidden_size // config.num_attention_heads
        kv_dim = head_dim * config.num_key_value_heads
        cache_shape = (config.num_hidden_layers, 1, MAX_SEQ_LEN, kv_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Single traced path. The KV cache buffers are always mutated in place via
        each layer's mutable_slice_update, so they are detected as state and bound
        to state_names=["k_cache","v_cache"] at conversion. There is deliberately
        no use_cache flag: per the coreai-torch docs, a data-dependent Python
        branch cannot be exported as a runtime choice, and removing the mutation
        would remove the buffers from state.

        position_ids carries ABSOLUTE positions; its width is the total context
        length (seq_len). offset = seq_len - query_len is where this step writes.
        """
        B, query_len = input_ids.shape
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = torch.tensor(seq_len - query_len, dtype=torch.int32)

        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, self.k_cache, self.v_cache, offset, seq_len)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def from_weights(cls, config, weights_dir: str) -> "FastVLMDecoderStateful":
        """Load from SafeTensors weights directory.

        Decoder weights in Apple's original PyTorch checkpoint live directly at:
          model.layers.*, model.embed_tokens.*, model.norm.*, lm_head.*
        There is NO language_model. prefix — confirmed from discovery output.

        Loading steps:
          1. Load only decoder keys (filter by DECODER_PREFIXES)
          2. Fuse q/k/v -> qkv_proj (_mutate_state_dict, keys still have model. prefix)
          3. Strip model. prefix so keys match PyTorch module hierarchy
          4. load_state_dict with strict=False (k_cache/v_cache are buffers, not checkpoint weights)
        """
        model = cls(config).to(dtype=torch.float16)
        weights = _load_decoder_weights(weights_dir)
        _mutate_state_dict(weights)
        # Strip model. prefix so keys match the module hierarchy:
        #   model.layers.N.* -> layers.N.*
        #   model.embed_tokens.* -> embed_tokens.*
        #   model.norm.* -> norm.*
        #   lm_head.* -> lm_head.* (no prefix to strip)
        weights = {k.removeprefix("model."): v for k, v in weights.items()}
        missing, unexpected = model.load_state_dict(weights, assign=True, strict=False)
        # k_cache and v_cache are registered buffers, not checkpoint weights
        expected_missing = {"k_cache", "v_cache"}
        actual_missing = set(missing) - expected_missing
        if actual_missing:
            raise RuntimeError(f"Unexpected missing keys: {actual_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        return model


# ─── Weight loading helpers ───────────────────────────────────────────────────

# Keys that belong to the decoder in Apple's original PyTorch checkpoint.
# Confirmed from discovery output — no language_model. prefix exists.
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
    """Load only decoder weights from SafeTensors, casting bfloat16 -> dtype.

    Filters to DECODER_PREFIXES only — excludes vision tower and projector keys.
    Weights are stored as bfloat16 in Apple's checkpoint; cast during loading.
    Embeddings are kept as float32 for precision (skip cast).
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
    """Fuse q_proj, k_proj, v_proj -> qkv_proj in-place.

    Called BEFORE stripping the model. prefix, so keys are still:
      model.layers.N.self_attn.q_proj.weight  (confirmed from discovery output)

    After fusion, replaces those three keys with:
      model.layers.N.self_attn.qkv_proj.weight
      model.layers.N.self_attn.qkv_proj.bias

    The model. prefix is stripped in from_weights() after this call.
    """
    layer_indices = set()
    for k in state_dict:
        if k.startswith("model.layers.") and ".self_attn.q_proj.weight" in k:
            idx = int(k.split(".")[2])
            layer_indices.add(idx)

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
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(
                weights
            )
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.bias"] = torch.cat(
                biases
            )
