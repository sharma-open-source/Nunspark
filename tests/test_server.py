import http.client
import json
import threading

import pytest

from mlx_lm.tokenizer_utils import load as load_tokenizer

from nunspark.archspec import KVQuant
from nunspark.engine import StreamingEngine
from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.server import build_server, shutdown_server, ChatEvent, sequence_overlap, stop_id_sequences, stopping_criteria, stream_chat_completion, chat_completion_chunk, chat_completion_response, error_response


class _FakeTokenizer:
    """Stands in for the real tokenizer's encode() -- one id per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_stop_id_sequences_normalizes_none_str_and_list():
    tok = _FakeTokenizer()
    assert stop_id_sequences(tok, None) == []
    assert stop_id_sequences(tok, "ab") == [[97, 98]]
    assert stop_id_sequences(tok, ["ab", "c"]) == [[97, 98], [99]]


def test_stop_id_sequences_drops_empty_encodings():
    tok = _FakeTokenizer()
    assert stop_id_sequences(tok, ["ab", "", "c"]) == [[97, 98], [99]]


def test_stopping_criteria_matches_eos():
    cond = stopping_criteria([5, 6, 2], [], eos_token_ids={2})
    assert cond.stop_met is True
    assert cond.trim_length == 0


def test_stopping_criteria_matches_stop_sequence_tail():
    cond = stopping_criteria([5, 6, 7, 8], [[7, 8]], eos_token_ids=set())
    assert cond.stop_met is True
    assert cond.trim_length == 2


def test_stopping_criteria_no_match():
    cond = stopping_criteria([5, 6], [[7, 8]], eos_token_ids={2})
    assert cond.stop_met is False
    assert cond.trim_length == 0


def test_sequence_overlap_detects_partial_suffix_match():
    assert sequence_overlap([1, 2, 3], [3, 4, 5])    # suffix [3] is a prefix of seq
    assert sequence_overlap([1, 2, 3], [2, 3, 4])    # suffix [2, 3] is a prefix of seq
    assert sequence_overlap([1, 2, 3], [1, 2, 3, 4]) # whole tail matches the prefix


def test_sequence_overlap_no_match():
    assert not sequence_overlap([1, 2, 3], [9, 9])
    assert not sequence_overlap([1, 2, 3], [])


@pytest.fixture
def packed_chat_dir(tiny_model_dir_with_tokenizer, tmp_path):
    out = tmp_path / "packed-chat"
    pack(tiny_model_dir_with_tokenizer, out)
    return out


@pytest.fixture
def chat_engine_and_tokenizer(packed_chat_dir):
    manifest = Manifest.load(packed_chat_dir / "manifest.json")
    engine = StreamingEngine(packed_chat_dir, manifest, budget_bytes=64 * 1024 * 1024)
    try:
        yield engine, load_tokenizer(packed_chat_dir)
    finally:
        engine.close()


def _script(monkeypatch, token_ids):
    """Replace stream_generate with one that yields exactly `token_ids`,
    ignoring the prompt -- makes stop/EOS/trim behavior deterministic."""
    def fake_stream_generate(engine, prompt, *, max_tokens, temp, kv_budget, prefetch, **kwargs):
        yield from token_ids

    monkeypatch.setattr("nunspark.server.stream_generate", fake_stream_generate)


def test_stream_chat_completion_stops_at_max_tokens_and_flushes_remaining_text(
    monkeypatch, chat_engine_and_tokenizer
):
    engine, tokenizer = chat_engine_and_tokenizer
    body = tokenizer.encode("The capital of France is Paris.", add_special_tokens=False)
    _script(monkeypatch, body)

    events = list(stream_chat_completion(
        engine, tokenizer, [1], max_tokens=len(body), temp=0.0,
        kv_budget=10**12, prefetch=True, stop_sequences=[]))

    assert events[-1] == ChatEvent("", "length", len(body))
    text = "".join(e.segment for e in events[:-1])
    assert text == tokenizer.decode(body)


def test_stream_chat_completion_stops_on_eos_and_excludes_it(monkeypatch, chat_engine_and_tokenizer):
    engine, tokenizer = chat_engine_and_tokenizer
    eos_id = next(iter(tokenizer.eos_token_ids))
    body = tokenizer.encode("Hello there", add_special_tokens=False)
    _script(monkeypatch, body + [eos_id, *tokenizer.encode(" should not appear", add_special_tokens=False)])

    events = list(stream_chat_completion(
        engine, tokenizer, [1], max_tokens=100, temp=0.0,
        kv_budget=10**12, prefetch=True, stop_sequences=[]))

    assert events[-1].finish_reason == "stop"
    assert events[-1].segment == ""
    assert events[-1].completion_tokens == len(body) + 1   # includes the EOS token itself
    text = "".join(e.segment for e in events[:-1])
    assert text == tokenizer.decode(body)
    assert "appear" not in text


def test_stream_chat_completion_stops_on_stop_sequence_and_drops_match(
    monkeypatch, chat_engine_and_tokenizer
):
    engine, tokenizer = chat_engine_and_tokenizer
    stop_ids = tokenizer.encode("Paris", add_special_tokens=False)
    assert stop_ids
    prefix = tokenizer.encode("The capital of France is ", add_special_tokens=False)
    suffix = tokenizer.encode(" and much more", add_special_tokens=False)
    _script(monkeypatch, prefix + stop_ids + suffix)

    events = list(stream_chat_completion(
        engine, tokenizer, [1], max_tokens=100, temp=0.0,
        kv_budget=10**12, prefetch=True, stop_sequences=[stop_ids]))

    assert events[-1].finish_reason == "stop"
    text = "".join(e.segment for e in events[:-1])
    assert text == tokenizer.decode(prefix)
    assert "Paris" not in text
    assert "more" not in text


def test_chat_completion_response_shape():
    resp = chat_completion_response(
        completion_id="chatcmpl-abc", created=1700000000, model="tiny-llama",
        text="hello there", finish_reason="stop", prompt_tokens=3, completion_tokens=2,
    )
    assert resp == {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "tiny-llama",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello there"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_chat_completion_chunk_shape():
    chunk = chat_completion_chunk(
        completion_id="chatcmpl-abc", created=1700000000, model="tiny-llama",
        delta={"role": "assistant", "content": "hi"}, finish_reason=None,
    )
    assert chunk == {
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "tiny-llama",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}],
    }


def test_error_response_shape():
    assert error_response("bad request") == {
        "error": {"message": "bad request", "type": "invalid_request_error", "code": None}
    }
    assert error_response("nope", error_type="not_found_error", code="model_not_found") == {
        "error": {"message": "nope", "type": "not_found_error", "code": "model_not_found"}
    }


@pytest.fixture
def running_server(packed_chat_dir):
    server, state = build_server(
        str(packed_chat_dir), "127.0.0.1", 0,   # port 0 -> OS picks a free port
        budget_bytes=64 * 1024 * 1024, kv_budget=10**12,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, state
    finally:
        shutdown_server(server, state)
        thread.join(timeout=5)


def _request(server, method, path, payload=None):
    conn = http.client.HTTPConnection(*server.server_address)
    try:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        return resp, resp.read()
    finally:
        conn.close()


def test_v1_models_lists_the_loaded_model(running_server):
    server, state = running_server
    resp, raw = _request(server, "GET", "/v1/models")
    payload = json.loads(raw)

    assert resp.status == 200
    assert payload == {
        "object": "list",
        "data": [{"id": state.model_name, "object": "model", "created": 0, "owned_by": "nunspark"}],
    }


def test_unknown_route_returns_404_error_envelope(running_server):
    server, _ = running_server
    resp, raw = _request(server, "GET", "/v1/unknown")
    assert resp.status == 404
    assert json.loads(raw)["error"]["code"] == "not_found"


def test_chat_completions_rejects_missing_messages(running_server):
    server, _ = running_server
    resp, raw = _request(server, "POST", "/v1/chat/completions", {"model": "x"})
    assert resp.status == 400
    assert json.loads(raw)["error"]["type"] == "invalid_request_error"


def test_chat_completions_rejects_unknown_model(running_server):
    server, _ = running_server
    resp, raw = _request(server, "POST", "/v1/chat/completions",
                         {"model": "not-the-loaded-model", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status == 404
    assert json.loads(raw)["error"]["code"] == "model_not_found"


def test_chat_completions_non_streaming_returns_full_message(running_server):
    server, state = running_server
    resp, raw = _request(server, "POST", "/v1/chat/completions", {
        "model": state.model_name,
        "messages": [{"role": "user", "content": "Hello there, how are you doing today?"}],
        "max_tokens": 5,
        "temperature": 0,
    })
    payload = json.loads(raw)

    assert resp.status == 200
    assert payload["object"] == "chat.completion"
    assert payload["model"] == state.model_name
    choice = payload["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str) and choice["message"]["content"]
    assert choice["finish_reason"] in ("length", "stop")
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["usage"]["completion_tokens"] >= 1


def test_chat_completions_streaming_reassembles_to_the_same_text(running_server):
    server, state = running_server
    request = {
        "model": state.model_name,
        "messages": [{"role": "user", "content": "Hello there, how are you doing today?"}],
        "max_tokens": 5,
        "temperature": 0,   # deterministic, so streamed/non-streamed must match exactly
    }

    full_resp, full_raw = _request(server, "POST", "/v1/chat/completions", request)
    full = json.loads(full_raw)

    stream_resp, stream_raw = _request(server, "POST", "/v1/chat/completions", {**request, "stream": True})
    assert stream_resp.status == 200
    assert stream_resp.getheader("Content-Type") == "text/event-stream"

    chunks, finish_reason = [], None
    for line in stream_raw.decode().splitlines():
        if not line.startswith("data: "):
            continue
        raw_event = line[len("data: "):]
        if raw_event == "[DONE]":
            break
        chunk = json.loads(raw_event)
        delta = chunk["choices"][0]["delta"]
        chunks.append(delta.get("content", ""))
        finish_reason = chunk["choices"][0]["finish_reason"] or finish_reason

    assert "".join(chunks) == full["choices"][0]["message"]["content"]
    assert finish_reason == full["choices"][0]["finish_reason"]


def test_server_kv_quant_threads_to_state(packed_chat_dir):
    server, state = build_server(
        str(packed_chat_dir), "127.0.0.1", 0,
        budget_bytes=64 * 1024 * 1024, kv_budget=10**12,
        kv_quant=KVQuant(bits=8, group_size=32))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert state.kv_quant == KVQuant(bits=8, group_size=32)
    finally:
        shutdown_server(server, state)
        thread.join(timeout=5)


def _chat_text(server, model_name, content, max_tokens=6):
    resp, raw = _request(server, "POST", "/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0,
    })
    assert resp.status == 200
    return json.loads(raw)["choices"][0]["message"]["content"]


def test_prefix_cache_enabled_by_default(running_server):
    _, state = running_server
    assert state.prefix_cache is not None


def test_no_prefix_cache_flag_disables_slot(packed_chat_dir):
    server, state = build_server(
        str(packed_chat_dir), "127.0.0.1", 0,
        budget_bytes=64 * 1024 * 1024, kv_budget=10**12,
        use_prefix_cache=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert state.prefix_cache is None
    finally:
        shutdown_server(server, state)
        thread.join(timeout=5)


def test_prefix_cache_output_matches_cold_server(running_server, packed_chat_dir):
    """The prefix cache is an optimization only — greedy output must be
    identical whether or not a prefix was reused. The warm server runs two
    requests (the second reuses the first's shared template prefix, the trim
    path); a separate cache-disabled server runs the same second request cold.
    Their outputs must match, and a repeated identical request must be stable."""
    server, state = running_server
    assert state.prefix_cache is not None

    warm_first = _chat_text(server, state.model_name, "Tell me about France.")
    warm_second = _chat_text(server, state.model_name, "Tell me about Spain please.")
    # The slot now holds the most recent prompt + its generated tokens.
    assert len(state.prefix_cache.tokens) > 0
    # An identical repeat hits the fully-cached refeed path; output is stable.
    warm_second_again = _chat_text(server, state.model_name, "Tell me about Spain please.")
    assert warm_second_again == warm_second

    cold_server, cold_state = build_server(
        str(packed_chat_dir), "127.0.0.1", 0,
        budget_bytes=64 * 1024 * 1024, kv_budget=10**12,
        use_prefix_cache=False)
    cold_thread = threading.Thread(target=cold_server.serve_forever, daemon=True)
    cold_thread.start()
    try:
        assert cold_state.prefix_cache is None
        cold_first = _chat_text(cold_server, cold_state.model_name, "Tell me about France.")
        cold_second = _chat_text(cold_server, cold_state.model_name, "Tell me about Spain please.")
    finally:
        shutdown_server(cold_server, cold_state)
        cold_thread.join(timeout=5)

    assert warm_first == cold_first
    assert warm_second == cold_second   # trim-path reuse == cold compute
