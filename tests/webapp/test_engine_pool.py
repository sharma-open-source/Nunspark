from nunspark.webapp.engine_pool import EnginePool


def test_acquire_returns_engine_and_tokenizer(tiny_packed_dir):
    pool = EnginePool()
    try:
        h = pool.acquire(str(tiny_packed_dir), draft=None, budget_bytes=4 * 10**9, kv_quant=None)
        assert h.engine is not None
        assert h.tokenizer is not None
        assert h.draft_model is None
    finally:
        pool.close()


def test_acquire_reuses_same_config(tiny_packed_dir):
    pool = EnginePool()
    try:
        a = pool.acquire(str(tiny_packed_dir), draft=None, budget_bytes=4 * 10**9, kv_quant=None)
        b = pool.acquire(str(tiny_packed_dir), draft=None, budget_bytes=4 * 10**9, kv_quant=None)
        assert a.engine is b.engine  # same object, not reloaded
    finally:
        pool.close()
