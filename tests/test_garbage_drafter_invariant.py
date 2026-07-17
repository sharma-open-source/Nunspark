"""M1 (Plan 5): the garbage-drafter invariant.

The hard invariant, quoted from TensorFold: a drafter whose proposals are
NEVER accepted must still reproduce baseline greedy output token-for-token.
This isolates the verify/commit/rollback machinery (`engine.verify_forward`
+ `engine.commit_verified`, draft-cache `trim`) from drafter quality: any
state leak from a rejected token shows up as a byte diff against plain
`generate()`.

Both speculative paths are lossless-by-construction at accept_top_k=1: a
proposal is only ever accepted if it equals the target's own argmax, so an
ADVERSARIAL drafter that is guaranteed to always disagree with the target
must still produce bit-identical output, with zero accepted tokens.
"""
import json

import mlx.core as mx
from mlx_lm.models.llama import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate, speculative_generate, ngram_speculative_generate, SpecStats


VOCAB = 320  # matches TINY_CONFIG / TINY_GPT_OSS_CONFIG vocab_size in tests/conftest.py


def _build_engine(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9)


def _load_mlx_model(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    # same weights as the packed target -> an UNROLLED copy would get full
    # acceptance (see test_speculative.py::test_self_draft_matches_greedy).
    model.load_weights(list(mx.load(str(model_dir / "model.safetensors")).items()))
    mx.eval(model.parameters())
    return model


# --- adversarial n-gram drafter ---------------------------------------------

class _WrongEveryTimeNGramDrafter:
    """A stub drafter that ignores the real n-gram machinery entirely and
    always proposes tokens computed as `(true_next + 1) % V`, where
    `true_next` is read off a PRE-COMPUTED greedy reference sequence at the
    position the drafter is currently being asked about.

    `ngram_speculative_generate` tracks `context = prompt + emitted-so-far`,
    so `len(context) - prompt_len` is exactly the number of tokens already
    confirmed -- i.e. the index into `ref` of the NEXT token the target will
    want. Because the target is lossless greedy, that next token is always
    `ref[pos]`, so `(ref[pos] + 1) % V` can never equal it (V > 1): every
    proposed token is wrong by construction, at every position, in every
    round -- not just the first mismatch in a round.
    """

    def __init__(self, ref: list[int], prompt_len: int, vocab_size: int, num_draft_tokens: int):
        self.ref = ref
        self.prompt_len = prompt_len
        self.vocab_size = vocab_size
        self.num_draft_tokens = num_draft_tokens

    def propose(self, context: list[int]) -> list[int]:
        pos = len(context) - self.prompt_len
        out = []
        for i in range(self.num_draft_tokens):
            idx = pos + i
            true_tok = self.ref[idx] if idx < len(self.ref) else 0
            out.append((true_tok + 1) % self.vocab_size)
        return out


def test_adversarial_ngram_drafter_matches_greedy(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=32, temp=0.0)
        drafter = _WrongEveryTimeNGramDrafter(ref, len(prompt), VOCAB, num_draft_tokens=6)
        stats = SpecStats()
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=32, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 32
    assert stats.draft_tokens_proposed > 0
    assert stats.accepted_total == 0
    assert stats.accepted_offpath == 0


def test_adversarial_ngram_drafter_matches_greedy_gpt_oss(tiny_gpt_oss_model_dir, tmp_path):
    # Same invariant on the gpt-oss sliding-window (RotatingKVCache) fixture --
    # the documented risk area (backlog #4): rollback here is commit-only
    # (verify_forward never mutates the persistent kv), so a drafter that is
    # NEVER accepted must still exercise a real commit(m=0) every round without
    # any state leaking from the ephemeral verify pass.
    prompt = [3, 7, 42, 1, 9] * 3
    engine = _build_engine(tiny_gpt_oss_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        drafter = _WrongEveryTimeNGramDrafter(ref, len(prompt), VOCAB, num_draft_tokens=5)
        stats = SpecStats()
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=24, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 24
    assert stats.draft_tokens_proposed > 0
    assert stats.accepted_total == 0


# --- adversarial model draft -------------------------------------------------

class _RolledLogitsDraft:
    """Wraps a real mlx_lm model (loaded with the SAME weights as the packed
    target) so every returned logits row is rolled by one index along the
    vocab axis before the caller takes its argmax.

    Because the wrapped model has identical weights to the target, and
    `speculative_generate` always feeds the draft the target's own confirmed
    trajectory (a rejected round trims the draft cache back to exactly the
    last committed token, see the `drop = (K - 1) - m` trim below), the
    UNROLLED argmax at any position would equal the target's argmax there
    (this is precisely why the same setup gets full acceptance in
    test_speculative.py::test_self_draft_matches_greedy). Rolling shifts
    that argmax index by exactly one (mod V), so the token the draft actually
    proposes can never equal the target's argmax -- guaranteed zero
    acceptance, every round, forever.
    """

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        # Forward everything else (notably `.layers`, used by
        # make_prompt_cache) to the wrapped real model.
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        logits = self._model(*args, **kwargs)
        return mx.roll(logits, shift=1, axis=-1)


def test_adversarial_model_draft_matches_greedy(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        draft = _RolledLogitsDraft(_load_mlx_model(tiny_model_dir))
        stats = SpecStats()
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=24, num_draft_tokens=4,
            accept_top_k=1, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 24
    assert stats.draft_tokens_proposed > 0
    assert stats.accepted_total == 0
    assert stats.accepted_offpath == 0


def test_adversarial_model_draft_matches_greedy_gpt_oss(tiny_gpt_oss_model_dir, tmp_path):
    # Drives the draft-cache trim(drop) rollback path every round at m=0 on
    # top of a RotatingKVCache streaming target -- the two risk areas (draft
    # rollback + sliding-window target commit) exercised together.
    from mlx_lm.models.gpt_oss import Model as GptOssModel, ModelArgs as GptOssModelArgs

    prompt = [3, 7, 42, 1, 9] * 3
    engine = _build_engine(tiny_gpt_oss_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        config = json.loads((tiny_gpt_oss_model_dir / "config.json").read_text())
        raw_draft = GptOssModel(GptOssModelArgs.from_dict(config))
        raw_draft.load_weights(
            list(mx.load(str(tiny_gpt_oss_model_dir / "model.safetensors")).items()))
        mx.eval(raw_draft.parameters())
        draft = _RolledLogitsDraft(raw_draft)
        stats = SpecStats()
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=24, num_draft_tokens=4,
            accept_top_k=1, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 24
    assert stats.accepted_total == 0
    assert stats.accepted_offpath == 0


# --- mixed acceptance: commit_verified(kv, recs, m + 1) at varying m -------

class _PartialCorrectNGramDrafter:
    """Proposes the CORRECT next `j` tokens (read off a pre-computed greedy
    reference), then deliberately wrong tokens for the remainder of each
    round's K-token block. This forces `commit_verified` to run at
    `accepted_len = m + 1` for a specific, controlled `m == j` every round
    (until `ref` runs out near the end of generation), catching an off-by-one
    in that count directly -- `j == K` additionally exercises the m == K
    (full-acceptance) branch, where the draft-side logic differs (see
    `test_speculative.py`'s "all K accepted" comment).
    """

    def __init__(self, ref: list[int], prompt_len: int, vocab_size: int, j: int, num_draft_tokens: int):
        assert 0 <= j <= num_draft_tokens
        self.ref = ref
        self.prompt_len = prompt_len
        self.vocab_size = vocab_size
        self.j = j
        self.num_draft_tokens = num_draft_tokens

    def propose(self, context: list[int]) -> list[int]:
        pos = len(context) - self.prompt_len
        out = []
        for i in range(self.num_draft_tokens):
            idx = pos + i
            true_tok = self.ref[idx] if idx < len(self.ref) else 0
            if i < self.j:
                out.append(true_tok)
            else:
                out.append((true_tok + 1) % self.vocab_size)
        return out


def test_mixed_acceptance_ngram_bit_identical(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    K = 6
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=28, temp=0.0)
        for j in (1, K - 1, K):
            drafter = _PartialCorrectNGramDrafter(ref, len(prompt), VOCAB, j=j, num_draft_tokens=K)
            stats = SpecStats()
            got = list(ngram_speculative_generate(
                engine, drafter, prompt, max_tokens=28, stats=stats))
            assert got == ref, f"mismatch at j={j}"
            assert len(got) == 28
            # a partial-correct drafter must accept exactly j tokens per full
            # round of K proposals (short of the tail where ref runs out).
            assert stats.accepted_offpath == 0
    finally:
        engine.close()


def test_mixed_acceptance_ngram_full_acceptance_commits_m_equals_k(tiny_model_dir, tmp_path):
    # j == K in isolation, with a stats check that at least one round actually
    # reached full acceptance (m == K), specifically exercising the
    # `commit_verified(kv, recs, K + 1)` call.
    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    K = 6
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=28, temp=0.0)
        drafter = _PartialCorrectNGramDrafter(ref, len(prompt), VOCAB, j=K, num_draft_tokens=K)
        stats = SpecStats()
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=28, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert stats.accepted_total > 0
    assert stats.accepted_offpath == 0
