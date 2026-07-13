from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class EagleConfig:
    """Configuration for the EAGLE feature-level drafter.

    The drafter is a tiny 2-layer transformer that predicts the next hidden
    state given the current target hidden state and the next token's embedding.
    """
    hidden_size: int = 2560          # must match target model's hidden_size
    num_hidden_layers: int = 2
    intermediate_size: int = 4096
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 320            # matches the target model's vocab
    rope_theta: float = 10000.0
    max_position_embeddings: int = 512
    tie_word_embeddings: bool = True

    @classmethod
    def from_target_config(cls, target_config: dict) -> "EagleConfig":
        """Build EagleConfig from the target model's config dict."""
        return cls(
            hidden_size=target_config.get("hidden_size", 2560),
            intermediate_size=target_config.get("intermediate_size", 4096),
            num_attention_heads=target_config.get("num_attention_heads", 8),
            num_key_value_heads=target_config.get(
                "num_key_value_heads",
                target_config.get("num_attention_heads", 8),
            ),
            head_dim=target_config.get("head_dim", 256),
            rms_norm_eps=target_config.get("rms_norm_eps", 1e-6),
            vocab_size=target_config.get("vocab_size", 320),
            rope_theta=target_config.get("rope_theta", 10000.0),
            max_position_embeddings=min(
                target_config.get("max_position_embeddings", 2048), 512
            ),
            tie_word_embeddings=target_config.get(
                "tie_word_embeddings", True
            ),
        )


class EagleAttention(nn.Module):
    """Self-attention for one EAGLE drafter layer."""

    def __init__(self, config: EagleConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )

        from mlx_lm.models.rope_utils import initialize_rope
        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=False,
            base=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # GQA repeat
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_out)


class EagleMLP(nn.Module):
    """SwiGLU MLP for one EAGLE drafter layer."""

    def __init__(self, config: EagleConfig):
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

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EagleTransformerBlock(nn.Module):
    """One transformer layer for the EAGLE drafter."""

    def __init__(self, config: EagleConfig):
        super().__init__()
        self.self_attn = EagleAttention(config)
        self.mlp = EagleMLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class EagleDrafterModel(nn.Module):
    """Tiny feature-level drafter for EAGLE speculative decoding.

    Architecture (following EAGLE, Zhou et al. 2024):
      - Input: concat(target_hidden_t, target_embed(token_{t+1}))   [2 * hidden_size]
      - Input projection: Linear(2 * hidden_size, hidden_size)
      - 2 transformer decoder layers (self-attention + SwiGLU MLP)
      - Output: hidden state for next position (fed back as input)
      - Logit head: projects hidden state → vocabulary logits

    The drafter operates in FEATURE space: it predicts what the target model's
    hidden state WILL BE at position t+1, not just the next token. This gives
    much higher acceptance than token-level draft models because the feature
    prediction captures more of the target's internal representation.

    The tied lm_head (self.embed_tokens.as_linear) is used for logit projection,
    matching the EAGLE paper's approach of sharing the target's vocabulary
    head weights (after training).
    """

    def __init__(self, config: EagleConfig):
        super().__init__()
        self.config = config

        # Input: concat(h_t, embed(token_{t+1})) → project to hidden_size
        self.input_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )

        # Transformer body
        self.layers = [
            EagleTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output: hidden state prediction (residual: h_in + Δ)
        self.output_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

        # Embedding table (shared with target; used for token lookup during draft)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        mask: mx.array | None = None,
        cache=None,
    ) -> tuple[mx.array, mx.array]:
        """One EAGLE drafter forward.

        Args:
            hidden_states: Target's last hidden state at positions 0..t
                           (B, L, hidden_size)
            input_ids: Token IDs at positions 1..t+1
                       (B, L) — the NEXT tokens after each hidden state
            mask: Optional causal mask (B, 1, L, L)
            cache: Optional KV cache list (one per layer)

        Returns:
            predicted_hidden: Predicted next hidden state (B, L, hidden_size)
            logits: Token logits for next positions (B, L, vocab_size)
        """
        # Embed the input tokens
        emb = self.embed_tokens(input_ids)  # (B, L, hidden_size)

        # Concatenate hidden states and embeddings
        x = mx.concatenate([hidden_states, emb], axis=-1)  # (B, L, 2*hidden)
        x = self.input_proj(x)  # (B, L, hidden)

        # Run transformer layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, mask=mask, cache=layer_cache)
        x = self.norm(x)

        # Predicted next hidden state (residual: add input projection)
        # This is h^_{t+1} in feature space
        predicted_hidden = x + self.output_proj(x)

        # Logits for next tokens
        logits = self.embed_tokens.as_linear(predicted_hidden)

        return predicted_hidden, logits


class EagleDrafter:
    """Wrapper for integrating EagleDrafterModel into speculative decoding.

    The drafter generates K draft tokens by iteratively:
      1. Taking the target's last hidden state and the last drafted token
      2. Predicting the next hidden state and logits
      3. Sampling the next token
      4. Using the predicted hidden state + sampled token as input for the next step

    Unlike a full draft model (which runs a complete transformer forward per
    draft step), the EAGLE drafter is just 2 layers — ~100× cheaper per step.
    """

    def __init__(
        self,
        model: EagleDrafterModel,
        target_embed_fn: Callable[[mx.array], mx.array] | None = None,
    ):
        self.model = model
        self.target_embed_fn = target_embed_fn
        self.hidden_size = model.config.hidden_size
        self.vocab_size = model.config.vocab_size

    def draft_speculative(
        self,
        last_hidden: mx.array,
        bootstrap_token: mx.array,
        max_draft_tokens: int = 16,
        embed_fn: Callable[[mx.array], mx.array] | None = None,
        temp: float = 0.0,
    ) -> tuple[mx.array, list[mx.array] | None]:
        """Draft K tokens via iterative feature-level prediction.

        Args:
            last_hidden: Target's last hidden state at the bootstrap position
                         (B, 1, hidden_size)
            bootstrap_token: The last confirmed token ID (B, 1)
            max_draft_tokens: Maximum number of draft tokens to generate
            embed_fn: Embedding lookup function (target's embed_tokens).
                      If None, uses the drafter's own embedding table
                      (acceptable after training; during training the target's
                       embedding should be used for better alignment).
            temp: Sampling temperature (0 = greedy argmax)

        Returns:
            draft_tokens: Sampled draft token IDs (1, max_draft_tokens)
            draft_logits: Logits for each draft step, or None if temp=0
                          (list of mx.array, each (1, 1, vocab_size))
        """
        B = last_hidden.shape[0]
        cur_hidden = last_hidden  # (B, 1, hidden_size)
        cur_token = bootstrap_token  # (B, 1)

        draft_tokens: list[mx.array] = []
        draft_logits: list[mx.array] = []

        # We build a causal mask incrementally as we go, but for the
        # single-position case (L=1 each step), mask=None is fine.
        for _ in range(max_draft_tokens):
            pred_hidden, logits = self.model(
                cur_hidden, cur_token, mask=None, cache=None
            )  # each (B, 1, hidden_size) / (B, 1, vocab_size)

            # Sample the next token from logits
            if temp <= 0.0:
                next_token = mx.argmax(logits[:, -1:, :], axis=-1)  # (B, 1)
            else:
                next_token = mx.random.categorical(
                    logits[:, -1:, :] * (1.0 / temp),
                    num_samples=1,
                )  # (B, 1)

            draft_tokens.append(next_token)
            if temp > 0.0:
                draft_logits.append(logits)

            # Update for next iteration
            cur_hidden = pred_hidden
            cur_token = next_token

        stacked = mx.concatenate(draft_tokens, axis=1)  # (B, K)
        return stacked, draft_logits if draft_logits else None

    def draft_with_cache(
        self,
        last_hidden: mx.array,
        bootstrap_token: mx.array,
        max_draft_tokens: int = 16,
        embed_fn: Callable[[mx.array], mx.array] | None = None,
        temp: float = 0.0,
    ) -> tuple[mx.array, list]:
        """Draft K tokens WITH KV cache reuse across steps.

        This is more efficient for longer drafts because each step can reuse
        the previous KV cache instead of recomputing full self-attention.

        The cache is a list of KVCache objects, one per drafter layer.
        As we generate draft tokens left-to-right, the KV cache grows,
        so each subsequent step only computes attention over the new token.

        Args: same as draft_speculative

        Returns:
            draft_tokens: (1, K) sampled draft token IDs
            cache: The final KV cache list (for optional reuse)
        """
        from mlx_lm.models.cache import KVCache, make_prompt_cache

        B = last_hidden.shape[0]

        # Embed bootstrap token (we already have its hidden state)
        emb = self.model.embed_tokens(bootstrap_token)

        x = mx.concatenate([last_hidden, emb], axis=-1)
        x = self.model.input_proj(x)  # (B, 1, hidden)

        # Initialize KV cache
        cache = [KVCache() for _ in self.model.layers]

        draft_tokens: list[mx.array] = []
        first = True

        for _ in range(max_draft_tokens):
            # Run transformer layers with KV cache
            for i, layer in enumerate(self.model.layers):
                layer_cache = cache[i]
                x = layer(x, mask=None, cache=layer_cache)
            x = self.model.norm(x)
            pred_hidden = x + self.model.output_proj(x)

            # Logits
            logits = self.model.embed_tokens.as_linear(pred_hidden)  # (B, 1, V)

            # Sample
            if temp <= 0.0:
                next_token = mx.argmax(logits[:, -1:, :], axis=-1)
            else:
                next_token = mx.random.categorical(
                    logits[:, -1:, :] * (1.0 / temp), num_samples=1
                )

            draft_tokens.append(next_token)

            # Prepare input for next step: [pred_hidden; embed(next_token)]
            emb = self.model.embed_tokens(next_token)
            x = mx.concatenate([pred_hidden, emb], axis=-1)
            x = self.model.input_proj(x)

        stacked = mx.concatenate(draft_tokens, axis=1)
        return stacked, cache

    @staticmethod
    def create_random(config: EagleConfig, seed: int = 0) -> "EagleDrafter":
        """Create a drafter with random weights (for testing/profiling)."""
        mx.random.seed(seed)
        model = EagleDrafterModel(config)
        mx.eval(model.parameters())
        return EagleDrafter(model)

    def load_weights(self, weights: dict) -> None:
        """Load trained weights into the drafter model."""
        self.model.update(weights)
        mx.eval(self.model.parameters())
