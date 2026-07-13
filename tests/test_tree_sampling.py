# tests/test_tree_sampling.py
import mlx.core as mx
import pytest

from nunspark.tree_shape import TreeShape
from nunspark.tree_spec import sample_accepted_path


def _tv(p, q):
    return 0.5 * float(mx.sum(mx.abs(p - q)).item())


def _one_level_tree(vocab, c):
    """Root with `c` leaf children; returns (shape, child node ids)."""
    s = TreeShape([c])
    return s, s.children[0]


def test_temp0_returns_target_argmax_when_absent():
    # temp=0, no child equals the target argmax -> emit nothing, bonus = argmax.
    s, kids = _one_level_tree(vocab=8, c=2)
    target_logits = {0: mx.array([0., 0, 0, 9., 0, 0, 0, 0])}  # argmax = token 3
    token = {0: 0, kids[0]: 5, kids[1]: 6}                     # neither child is 3
    # leaf nodes need a target dist too (for the bonus if a path is fully accepted)
    target_logits[kids[0]] = mx.zeros(8)
    target_logits[kids[1]] = mx.zeros(8)
    draft_logits = {0: mx.zeros(8)}
    accepted, bonus, final = sample_accepted_path(
        s, token, draft_logits, target_logits, temp=0.0)
    assert accepted == []
    assert bonus == 3
    assert final == 0


def test_temp0_accepts_matching_child():
    s, kids = _one_level_tree(vocab=8, c=2)
    target_logits = {0: mx.array([0., 0, 0, 0, 0, 0, 9., 0]),  # argmax after root = 6
                     kids[0]: mx.array([0., 9, 0, 0, 0, 0, 0, 0]),  # leaf argmax = 1
                     kids[1]: mx.zeros(8)}
    token = {0: 0, kids[0]: 6, kids[1]: 2}     # kids[0] token == 6 == target argmax
    draft_logits = {0: mx.zeros(8)}
    accepted, bonus, final = sample_accepted_path(
        s, token, draft_logits, target_logits, temp=0.0)
    assert accepted == [6]      # accepted the matching child
    assert bonus == 1           # then the leaf's argmax
    assert final == kids[0]


def test_negative_temp_is_greedy():
    # temp <= 0 must take the deterministic greedy branch (not invert the dist).
    s, kids = _one_level_tree(vocab=8, c=2)
    target_logits = {0: mx.array([0., 0, 0, 9., 0, 0, 0, 0]),  # argmax = 3
                     kids[0]: mx.zeros(8), kids[1]: mx.zeros(8)}
    token = {0: 0, kids[0]: 3, kids[1]: 6}     # kids[0] matches the target argmax
    draft_logits = {0: mx.zeros(8)}
    accepted, bonus, final = sample_accepted_path(
        s, token, draft_logits, target_logits, temp=-1.0)
    assert accepted == [3]
    assert final == kids[0]


def test_temp_pos_distribution_matches_target():
    # One-level tree, children sampled i.i.d. from the draft each trial; the first
    # emitted token's empirical distribution must match the target distribution.
    mx.random.seed(0)
    vocab, c, temp = 6, 3, 1.0
    s, kids = _one_level_tree(vocab, c)
    draft_logits_root = mx.array([2.0, 1.0, 0.0, -1.0, 0.5, 0.3])
    target_logits_root = mx.array([0.0, 0.5, 2.0, 1.0, -1.0, 0.2])
    p_target = mx.softmax(target_logits_root / temp)

    N = 40000
    counts = [0] * vocab
    for _ in range(N):
        kids_tokens = mx.random.categorical(draft_logits_root / temp, num_samples=c)
        token = {0: 0}
        target_logits = {0: target_logits_root}
        for ci, kid in enumerate(kids):
            token[kid] = int(kids_tokens[ci].item())
            target_logits[kid] = mx.zeros(vocab)     # leaf bonus dist (unused marginal check)
        draft_logits = {0: draft_logits_root}
        accepted, bonus, _ = sample_accepted_path(
            s, token, draft_logits, target_logits, temp=temp)
        first = accepted[0] if accepted else bonus
        counts[first] += 1

    emp = mx.array([x / N for x in counts])
    assert _tv(emp, p_target) < 0.02, f"TV={_tv(emp, p_target):.4f} emp={emp.tolist()}"
