# tests/test_engine_tree_forward.py
import tempfile

import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.kv_store import KVStore
from nunspark.archspec import KVQuant


def _kv_q():
    """Quantized-KV store for the quantized tree tests. Unlike WEIGHT
    quantization (bit-exact through the streamed slots, hence atol=0.0
    elsewhere), 8-bit KV quantization is lossy on activations, so the
    quantized parity tests below compare two same-config quantized runs
    with a small nonzero atol."""
    d = tempfile.mkdtemp(prefix="nunspark_kv_test_")
    return KVStore(d, budget_bytes=10**12, kv_quant=KVQuant(bits=8, group_size=32))


def _engine(model_dir, tmp_path, budget=10**9):
    out = tmp_path / "pk.nunspark"
    pack(model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=budget)


def _kv():
    d = tempfile.mkdtemp(prefix="nunspark_kv_test_")
    return KVStore(d, budget_bytes=10**12)


def _batched_equals_per_path(engine, prompt, token_paths, atol):
    """tree_forward logits == per-path B=1 forwards from the same prefilled prefix."""
    kv = _kv()
    try:
        engine.forward(mx.array(prompt)[None], kv=kv)          # prefill prefix
        P = len(prompt)
        logits, stash = engine.tree_forward(mx.array(token_paths), kv, prefix_len=P)
        mx.eval(logits)
    finally:
        kv.close()

    B, d = len(token_paths), len(token_paths[0])
    assert logits.shape[0] == B and logits.shape[1] == d
    for i, path in enumerate(token_paths):
        kvi = _kv()
        try:
            engine.forward(mx.array(prompt)[None], kv=kvi)     # same prefix
            ref = engine.forward(mx.array(path)[None], kv=kvi)  # [1, d, V]
            mx.eval(ref)
        finally:
            kvi.close()
        diff = float(mx.max(mx.abs(logits[i] - ref[0])))
        assert diff <= atol, f"path {i}: max|Δ|={diff}"


def test_tree_forward_matches_per_path_llama_fp16(tiny_model_dir, tmp_path):
    engine = _engine(tiny_model_dir, tmp_path)
    try:
        prompt = [3, 7, 42, 1]
        token_paths = [[5, 6, 9], [5, 6, 2], [8, 1, 0], [8, 1, 4]]   # share prefixes
        _batched_equals_per_path(engine, prompt, token_paths, atol=1e-4)
    finally:
        engine.close()


def test_tree_forward_matches_per_path_qwen3_quant(tiny_qwen3_quant_model_dir, tmp_path):
    engine = _engine(tiny_qwen3_quant_model_dir, tmp_path)
    try:
        prompt = [3, 7, 42, 1, 9]
        token_paths = [[2, 3, 3], [2, 3, 7], [5, 5, 1], [5, 5, 8]]
        _batched_equals_per_path(engine, prompt, token_paths, atol=0.0)  # quant is exact
    finally:
        engine.close()


def test_commit_path_equals_refeed(tiny_model_dir, tmp_path):
    # After committing path j up to accepted_len, the persistent KV must equal having
    # streamed (prefix + path_j[:accepted_len]) directly: the next token's logits match.
    engine = _engine(tiny_model_dir, tmp_path)
    try:
        prompt = [3, 7, 42, 1]
        token_paths = [[5, 6, 9], [8, 1, 4]]
        j, m = 1, 2          # commit path 1's first 2 tokens: [8, 1]
        probe = 11

        kv = _kv()
        try:
            engine.forward(mx.array(prompt)[None], kv=kv)
            _, stash = engine.tree_forward(mx.array(token_paths), kv, prefix_len=len(prompt))
            engine.commit_path(kv, stash, path_index=j, accepted_len=m)
            got = engine.forward(mx.array([probe])[None], kv=kv)[:, -1, :]
            mx.eval(got)
        finally:
            kv.close()

        kvr = _kv()
        try:
            engine.forward(mx.array(prompt)[None], kv=kvr)
            engine.forward(mx.array(token_paths[j][:m])[None], kv=kvr)
            ref = engine.forward(mx.array([probe])[None], kv=kvr)[:, -1, :]
            mx.eval(ref)
        finally:
            kvr.close()

        assert float(mx.max(mx.abs(got - ref))) <= 1e-4
    finally:
        engine.close()


def test_tree_forward_matches_per_path_qwen3_moe(tiny_qwen3_moe_model_dir, tmp_path):
    # Selectively-packed MoE: tree_forward must reassemble the full layer and match
    # per-path forward (which uses the selective path) bit-identically (fp16 tol).
    engine = _engine(tiny_qwen3_moe_model_dir, tmp_path)
    try:
        prompt = [3, 7, 42, 1, 9]
        token_paths = [[2, 3, 3], [2, 3, 7], [5, 5, 1], [5, 5, 8]]   # share prefixes
        _batched_equals_per_path(engine, prompt, token_paths, atol=1e-4)
    finally:
        engine.close()


def test_tree_forward_matches_per_path_qwen3_moe_quant(tiny_qwen3_moe_quant_model_dir, tmp_path):
    engine = _engine(tiny_qwen3_moe_quant_model_dir, tmp_path)
    try:
        prompt = [3, 7, 42, 1, 9]
        token_paths = [[2, 3, 3], [2, 3, 7], [5, 5, 1], [5, 5, 8]]
        _batched_equals_per_path(engine, prompt, token_paths, atol=0.0)  # quant is exact
    finally:
        engine.close()


def test_tree_forward_matches_per_path_quantized_kv(tiny_kvq_model_dir, tmp_path):
    """Quantized tree verify == per-path quantized forwards (same quant config
    on both sides, so the comparison isolates the tiling/stash plumbing)."""
    engine = _engine(tiny_kvq_model_dir, tmp_path)
    prompt = [1, 2, 3, 4, 5]
    token_paths = [[7, 8], [7, 9], [11, 12]]
    kv = _kv_q()
    try:
        engine.forward(mx.array(prompt)[None], kv=kv)
        logits, stash = engine.tree_forward(
            mx.array(token_paths), kv, prefix_len=len(prompt))
        mx.eval(logits)
    finally:
        kv.close()

    for i, path in enumerate(token_paths):
        kvi = _kv_q()
        try:
            engine.forward(mx.array(prompt)[None], kv=kvi)
            ref = engine.forward(mx.array(path)[None], kv=kvi)
            mx.eval(ref)
        finally:
            kvi.close()
        diff = float(mx.max(mx.abs(logits[i] - ref[0])))
        assert diff <= 2e-2, f"path {i}: max|Δ|={diff}"
    engine.close()


def test_commit_path_quantized_equals_refeed(tiny_kvq_model_dir, tmp_path):
    """After committing a path from the quantized stash, the next forward's
    logits match re-feeding prompt+path into a fresh quantized store."""
    engine = _engine(tiny_kvq_model_dir, tmp_path)
    prompt = [1, 2, 3, 4, 5]
    path = [7, 8, 9]
    probe = [10]

    kv = _kv_q()
    try:
        engine.forward(mx.array(prompt)[None], kv=kv)
        _, stash = engine.tree_forward(mx.array([path]), kv, prefix_len=len(prompt))
        engine.commit_path(kv, stash, path_index=0, accepted_len=len(path))
        got = engine.forward(mx.array(probe)[None], kv=kv)
        mx.eval(got)
    finally:
        kv.close()

    ref_kv = _kv_q()
    try:
        engine.forward(mx.array(prompt + path)[None], kv=ref_kv)
        ref = engine.forward(mx.array(probe)[None], kv=ref_kv)
        mx.eval(ref)
    finally:
        ref_kv.close()

    diff = float(mx.max(mx.abs(got - ref)))
    assert diff <= 2e-2, f"max|Δ|={diff}"
    engine.close()
