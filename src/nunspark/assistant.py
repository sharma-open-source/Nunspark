"""Helper for Gemma 4 Assistant (MTP drafter) speculative decoding integration.

The assistant is a 4-layer Q-only drafter that cross-attends to the target model's
K/V states. This module provides helpers for integrating it into speculative decoding.
"""

from typing import Dict, Tuple, Optional
import mlx.core as mx
from . import gemma4_assistant


class AssistantDrafter:
    """Wrapper for gemma4_assistant.Model for speculative decoding.

    The assistant drafter is designed to work with gemma4 target models:
    - It has no K/V projections of its own
    - It cross-attends to the target model's K/V via shared_kv_states
    - Input: concat(target_embed(last_token), target_last_hidden)
    - Output: (last_hidden, logits) for next draft step
    """

    def __init__(self, model: gemma4_assistant.Model):
        self.model = model
        self.backbone_hidden_size = model.args.backbone_hidden_size
        self.vocab_size = model.args.vocab_size

    def draft_step(
        self,
        target_embed: mx.array,
        target_last_hidden: mx.array,
        shared_kv_states: Dict[str, Tuple[mx.array, mx.array]],
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Run one assistant drafting step.

        Args:
            target_embed: Target model's embedding of last token (B, 1, backbone_hidden_size)
            target_last_hidden: Target model's last hidden state (B, 1, backbone_hidden_size)
            shared_kv_states: Dict mapping layer_type -> (K, V) from target model
                {"full_attention": (K_full, V_full), "sliding_attention": (K_slide, V_slide)}
            position_ids: Optional position ids (B, 1)
            mask: Optional attention mask (B, 1, 1, L_target)

        Returns:
            last_hidden: Hidden state to feed back as input (B, 1, backbone_hidden_size)
            logits: Draft token logits (B, 1, vocab_size)
        """
        # Pin the drafter's compute to its own weight dtype (bf16). The target
        # engine emits fp32 hidden states; without this cast MLX type-promotes
        # every assistant matmul to fp32, which re-casts the (vocab, hidden)
        # tied-embedding logits table per draft step — gigabytes of transient
        # allocations per round and a ~1000x slowdown.
        dtype = self.model.model.embed_tokens.weight.dtype
        inputs_embeds = mx.concatenate(
            [target_embed.astype(dtype), target_last_hidden.astype(dtype)], axis=-1
        )
        shared_kv_states = {
            t: (k.astype(dtype), v.astype(dtype))
            for t, (k, v) in shared_kv_states.items()
        }

        # Run assistant forward
        last_hidden, logits = self.model(
            inputs_embeds=inputs_embeds,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            mask=mask,
        )

        return last_hidden, logits

    def draft_speculative(
        self,
        target_embed: mx.array,
        target_last_hidden: mx.array,
        shared_kv_states: Dict[str, Tuple[mx.array, mx.array]],
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        max_draft_tokens: int = 24,
        embed_fn=None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Draft multiple tokens via repeated assistant calls (autoregressive).

        This is the main entry point for speculative decoding:
        1. Start with target_embed + target_last_hidden
        2. Get draft logits, sample top-K tokens
        3. For each drafted token, re-embed it (embed_fn) and feed it back
           with the new hidden state for the next draft
        4. Return all drafted tokens and their final hidden states

        Mirrors HF's `candidate_generator.py` drafting loop: each step embeds
        the PREVIOUS step's drafted token via the TARGET's embedding table
        (`self.target_model_input_embeddings(last_token_id)` in the
        reference); position_ids stays constant across steps (so does the
        reference's). Without embed_fn the loop degrades to feeding the
        bootstrap token's embedding every step, which wrecks acceptance.

        Args:
            target_embed: Target model's embedding of the bootstrap token
                (B, 1, backbone_hidden_size)
            target_last_hidden: Target model's last hidden (B, 1, backbone_hidden_size)
            shared_kv_states: Dict mapping layer_type -> (K, V) from target
            position_ids: Optional position ids (constant across draft steps)
            mask: Optional attention mask
            max_draft_tokens: Maximum number of draft tokens to generate
            embed_fn: (tokens (B, 1) -> (B, 1, backbone_hidden_size)) lookup in
                the TARGET's embedding table; engine.embed_tokens for the
                streaming engine.

        Returns:
            draft_tokens: Sampled draft tokens (B, max_draft_tokens)
            draft_logits: Logits for each drafted token (B, max_draft_tokens, vocab_size)
            final_hidden: Final hidden state after last draft (B, 1, backbone_hidden_size)
        """
        B = target_embed.shape[0]
        draft_tokens = []
        draft_logits = []
        final_hidden = target_last_hidden
        cur_embed = target_embed

        for step in range(max_draft_tokens):
            # Get draft from assistant
            last_hidden, logits = self.draft_step(
                target_embed=cur_embed,
                target_last_hidden=final_hidden,
                shared_kv_states=shared_kv_states,
                position_ids=position_ids,
                mask=mask,
            )

            # Sample top-K token (simplified - use top-1 for now)
            draft_token = mx.argmax(logits[..., -1, :], axis=-1, keepdims=True)  # (B, 1)
            draft_tokens.append(draft_token)
            draft_logits.append(logits)

            # Update for next iteration: new hidden + the drafted token's
            # embedding (stays lazy — embed lookup is an on-device gather).
            final_hidden = last_hidden
            if embed_fn is not None:
                cur_embed = embed_fn(draft_token)

        # Stack results
        draft_tokens = mx.concatenate(draft_tokens, axis=1)  # (B, max_draft_tokens)
        draft_logits = mx.concatenate(draft_logits, axis=1)  # (B, max_draft_tokens, vocab_size)

        return draft_tokens, draft_logits, final_hidden


def load_assistant(model_path: str) -> AssistantDrafter:
    """Load a gemma4_assistant model for speculative decoding.

    Args:
        model_path: Path to the assistant model directory

    Returns:
        AssistantDrafter instance ready for speculative decoding
    """
    from mlx_lm import load
    from . import gemma4_assistant

    # Load the model
    model, _ = load(model_path)

    # Wrap in drafter
    return AssistantDrafter(model)


def extract_target_kv_states(engine) -> Dict[str, Tuple[mx.array, mx.array]]:
    """Extract shared K/V states from the target StreamingEngine.

    The Gemma4LayerRunner (architectures.py) captures the post-RoPE (K, V)
    of each `store_full_length_kv` layer into `engine._lctx.target_kv_states`
    during `engine.forward()`. This is a thin accessor for that state.

    Returns:
        {"full_attention": (K, V), "sliding_attention": (K, V)}, each tensor
        shaped (B, n_kv_heads, L, head_dim).
    """
    return engine.target_kv_states()
