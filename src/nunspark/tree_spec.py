# src/nunspark/tree_spec.py
from __future__ import annotations

import tempfile
from collections.abc import Generator

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache

from .engine import StreamingEngine
from .generate import SpecStats
from .kv_store import KVStore
from .tree_shape import TreeShape


def _categorical_from_probs(probs: mx.array) -> int:
    """Sample one index from a (already-normalized) probability vector.

    `mx.random.categorical` expects logits and applies softmax internally, so we pass
    log-probabilities: softmax(log p) == p. The 1e-30 floor only guards log(0); do NOT
    "simplify" by passing `probs` directly (that would softmax the probabilities again).
    """
    logits = mx.log(mx.maximum(probs, 1e-30))
    return int(mx.random.categorical(logits).item())


def sample_accepted_path(
    shape: TreeShape,
    token: dict[int, int],
    draft_logits: dict[int, mx.array],
    target_logits: dict[int, mx.array],
    *,
    temp: float,
) -> tuple[list[int], int, int]:
    """SpecInfer recursive rejection sampling over a filled fixed tree.

    Walks from the root; at each internal node it considers that node's children
    (proposed by the draft) in order, doing multi-candidate rejection sampling
    against the target distribution. Returns `(accepted_tokens, bonus, final_node)`:
    `accepted_tokens` are the descendant tokens accepted (root excluded), `bonus` is
    the single correction token sampled from the residual (or the leaf's target dist
    if a whole path is accepted), and `final_node` is the deepest accepted node
    (root if none) — used to locate the committed path.

    At `temp <= 0` this is deterministic greedy: accept the child equal to the
    target argmax, else stop with `bonus = target argmax`. Output then matches plain
    greedy decoding. (`temp <= 0` rather than `== 0` matches generate._sample's
    convention and avoids inverting the distribution on a negative temperature.)
    """
    accepted: list[int] = []
    node = 0
    while shape.children.get(node):
        tl = target_logits[node]
        kids = shape.children[node]

        if temp <= 0.0:
            tgt = int(mx.argmax(tl).item())
            nxt = next((c for c in kids if token[c] == tgt), None)
            if nxt is None:
                return accepted, tgt, node
            accepted.append(token[nxt])
            node = nxt
            continue

        p = mx.softmax(tl / temp)                 # target dist after `node`
        q = mx.softmax(draft_logits[node] / temp)  # draft dist children were drawn from
        residual = p
        chosen = None
        for c in kids:
            t = token[c]
            ratio = float(residual[t].item()) / max(float(q[t].item()), 1e-30)
            if float(mx.random.uniform().item()) < min(1.0, ratio):
                chosen = c
                break
            residual = mx.maximum(residual - q, 0.0)
            s = float(mx.sum(residual).item())
            # If the residual mass is exhausted (only when r == q exactly, a
            # measure-zero edge), fall back to the target dist rather than sampling
            # uniformly over the whole vocab.
            residual = residual / s if s > 0.0 else p
        if chosen is None:
            return accepted, _categorical_from_probs(residual), node
        accepted.append(token[chosen])
        node = chosen

    # Reached a leaf: the whole path was accepted; the bonus is the leaf's next token.
    tl = target_logits[node]
    if temp <= 0.0:
        bonus = int(mx.argmax(tl).item())
    else:
        bonus = _categorical_from_probs(mx.softmax(tl / temp))
    return accepted, bonus, node


def _tile_cache(cache, c: int):
    """Return a copy of an mlx_lm prompt cache with each batch row repeated `c` times."""
    tiled = _make_prompt_cache_like(cache)
    for src, dst in zip(cache, tiled):
        k, v = src.state
        dst.state = (mx.repeat(k, c, axis=0), mx.repeat(v, c, axis=0))
    return tiled


def _make_prompt_cache_like(cache):
    """A fresh, empty cache list the same length/type as `cache` (all plain KVCache)."""
    return [KVCache() for _ in cache]


def build_draft_tree(draft_model, committed, shape: TreeShape, temp: float):
    """Fill `shape` with draft tokens, returning (token, draft_logits).

    Re-prefills the draft over `committed` (prompt + all emitted tokens), then expands
    the tree breadth-first, one batched draft forward per level. Node ids follow the
    same BFS numbering as `TreeShape`. `token[node]` is the proposed token (root =
    `committed[-1]`); `draft_logits[node]` (internal nodes only) is the distribution
    that node's children were drawn from. At `temp == 0` children are the top-c tokens;
    at `temp > 0` they are c i.i.d. samples from the tempered draft distribution.
    """
    cache = make_prompt_cache(draft_model)
    last = draft_model(mx.array(committed)[None], cache=cache)[:, -1, :]   # [1, V]

    token: dict[int, int] = {0: committed[-1]}
    draft_logits: dict[int, mx.array] = {0: last[0]}

    frontier = [0]
    frontier_logits = last        # [F, V]; row i is node frontier[i]'s child distribution
    frontier_cache = cache        # batch == F
    next_id = 1

    for c in shape.branching:
        F = len(frontier)
        if temp <= 0.0:
            child_tokens = mx.argsort(-frontier_logits, axis=-1)[:, :c]    # [F, c] top-c
        else:
            child_tokens = mx.random.categorical(frontier_logits / temp, num_samples=c)  # [F, c]
        mx.eval(child_tokens)

        base = next_id
        for fi in range(F):
            for j in range(c):
                token[next_id] = int(child_tokens[fi, j].item())
                next_id += 1

        flat = child_tokens.reshape(F * c, 1)
        tiled = _tile_cache(frontier_cache, c)
        child_logits = draft_model(flat, cache=tiled)[:, -1, :]            # [F*c, V]
        mx.eval(child_logits)

        new_nodes = list(range(base, next_id))
        for idx, nid in enumerate(new_nodes):
            if shape.children.get(nid):           # internal node -> store its child dist
                draft_logits[nid] = child_logits[idx]

        frontier = new_nodes
        frontier_logits = child_logits
        frontier_cache = tiled

    return token, draft_logits


def _sample_token(logits_row: mx.array, temp: float) -> int:
    if temp <= 0.0:
        return int(mx.argmax(logits_row).item())
    # Pass scaled logits directly to categorical (it applies softmax internally),
    # matching generate._sample exactly.
    return int(mx.random.categorical(logits_row * (1.0 / temp)).item())


def tree_speculative_generate(
    engine: StreamingEngine,
    draft_model,
    prompt: list[int],
    *,
    shape: TreeShape,
    max_tokens: int = 64,
    temp: float = 0.0,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    eos_id: int | None = None,
    stats: SpecStats | None = None,
) -> Generator[int, None, None]:
    """Tree speculative decoding over the streaming target.

    Each round: re-prefill the draft over the committed sequence, fill `shape`, verify
    every root-to-leaf path in ONE target sweep, run SpecInfer rejection sampling to
    pick the accepted path, commit its KV, and emit accepted tokens + a bonus. At
    `temp == 0`, output is target-verified and lossless-by-construction: every emitted
    token is the target's own argmax from a verification pass, deterministic for a
    given config. Note: not guaranteed byte-identical to single-token greedy
    `generate()` on large real models -- a multi-token verify pass computes fp16
    numerics under different kernel shapes, and rare argmax near-tie flips (~1/100
    tokens, self-healing, see docs/plan5-m2-mismatch-investigation.md) can occur;
    tiny-fixture tests are exactly bit-identical. KV lifecycle matches
    `speculative_generate`: pass `kv` to borrow a store, else one is created/disposed.
    """
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = KVStore(tmp.name, budget_bytes=kv_budget, prefetch=prefetch,
                         cache_kinds=engine.cache_kinds)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        tl = engine.forward(mx.array(prompt)[None], kv=kv)[:, -1, :]   # prefix offset == len(prompt)
        b = _sample_token(tl[0], temp)
        emitted = 0
        yield b
        emitted += 1
        if stats:
            stats.tokens_emitted += 1
        if b == eos_id:
            return
        committed = list(prompt) + [b]

        while emitted < max_tokens:
            token, draft_logits = build_draft_tree(draft_model, committed, shape, temp)
            if stats:
                stats.draft_tokens_proposed += shape.num_nodes - 1

            token_paths = mx.array(
                [[token[n] for n in path] for path in shape.paths])    # [B, d]
            prefix_len = len(committed) - 1                            # invariant
            vlog, stash = engine.tree_forward(token_paths, kv, prefix_len)
            if stats:
                stats.target_passes += 1

            # Map batched logits -> per-node target logits (shared prefixes dedup).
            target_logits = {}
            for node in range(shape.num_nodes):
                pi, dep = shape.node_locator[node]
                target_logits[node] = vlog[pi, dep]

            accepted, bonus, final_node = sample_accepted_path(
                shape, token, draft_logits, target_logits, temp=temp)

            pi, dep = shape.node_locator[final_node]
            engine.commit_path(kv, stash, path_index=pi, accepted_len=dep + 1)

            for t in accepted:
                if emitted >= max_tokens:
                    return
                yield t
                emitted += 1
                committed.append(t)
                if stats:
                    stats.tokens_emitted += 1
                if t == eos_id:
                    return
            if emitted >= max_tokens:
                return
            yield bonus
            emitted += 1
            committed.append(bonus)
            if stats:
                stats.tokens_emitted += 1
            if bonus == eos_id:
                return
    finally:
        if own_kv:
            kv.close()
            tmp.cleanup()
