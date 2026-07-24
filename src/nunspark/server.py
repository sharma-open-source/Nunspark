from __future__ import annotations

import json
import queue
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING

from mlx_lm.tokenizer_utils import load as load_tokenizer

from .archspec import KVQuant
from .engine import StreamingEngine
from .generate import stream_generate
from .manifest import Manifest

if TYPE_CHECKING:
    from .kv_store import KVStore
    from .prefix_cache import PrefixCache


class _StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


def stop_id_sequences(tokenizer, stop) -> list[list[int]]:
    """Normalize the OpenAI `stop` field (None | str | list[str]) into
    tokenized id sequences, matched against the tail of the generated tokens
    -- token ids rather than text, so matching is robust regardless of how
    the tokenizer splits words across boundaries."""
    if stop is None:
        return []
    if isinstance(stop, str):
        stop = [stop]
    sequences = []
    for s in stop:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            sequences.append(ids)
    return sequences


def stopping_criteria(tokens, stop_sequences, eos_token_ids) -> _StopCondition:
    """Whether generation should stop, and how many trailing tokens (the
    matched stop sequence) should be excluded from the visible output."""
    if tokens and tokens[-1] in eos_token_ids:
        return _StopCondition(stop_met=True, trim_length=0)
    for seq in stop_sequences:
        if len(tokens) >= len(seq) and tokens[-len(seq):] == seq:
            return _StopCondition(stop_met=True, trim_length=len(seq))
    return _StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(tokens, sequence) -> bool:
    """True if some non-empty suffix of `tokens` equals a prefix of
    `sequence` -- i.e. `sequence` might still be completed by tokens not yet
    generated, so any text derived from that suffix must be held back."""
    max_overlap = min(len(tokens), len(sequence))
    return any(tokens[-i:] == sequence[:i] for i in range(1, max_overlap + 1))


class ChatEvent(NamedTuple):
    """One step of a chat generation.

    `finish_reason` is None for every event except the last, which carries
    "stop" or "length" with an empty segment -- any text that might have been
    part of an unresolved stop sequence is dropped rather than retroactively
    un-sent. `completion_tokens` is the running total of generated tokens,
    valid (and final) on the last event."""
    segment: str
    finish_reason: str | None
    completion_tokens: int


def stream_chat_completion(
    engine: StreamingEngine,
    tokenizer,
    prompt_ids: list[int],
    *,
    max_tokens: int = 100,
    temp: float,
    kv_budget: int,
    prefetch: bool,
    stop_sequences: list[list[int]],
    draft_model: object | None = None,
    num_draft_tokens: int = 16,
    accept_top_k: int = 1,
    kv_quant: KVQuant | None = None,
    kv: KVStore | None = None,
    processed_tokens: int = 0,
    prefix_cache: PrefixCache | None = None,
):
    """Generate from `prompt_ids`, yielding `ChatEvent`s with EOS / stop-
    sequence handling and incremental detokenization.

    Text that overlaps a stop sequence's prefix is held back until the match
    resolves: flushed once it's known not to be the start of a stop sequence,
    dropped if the stop sequence completes.  When a stop sequence fires, any
    held-back text that precedes the matched stop sequence is recovered by
    re-decoding ``tokens[:-trim_length]`` and emitting the delta; this handles
    ByteLevel BPE tokenizers where a trailing space surfaces only when the next
    token is added and may have been withheld during overlap-checking.

    This mirrors mlx_lm.server's handle_completion, adapted to NunSpark's
    token-id stream_generate and TokenizerWrapper.detokenizer (a single shared
    instance -- reset() makes it safe to reuse across requests as long as they
    are serialized).

    When `draft_model` is provided, uses `speculative_generate()` for faster
    generation with the draft model proposing tokens that the target verifies.

    For prompt-prefix reuse the caller passes a borrowed `kv` (already holding
    the first `processed_tokens` of `prompt_ids`) and the owning `prefix_cache`;
    after generation the cache is committed with the full prompt + generated
    tokens so the next request can reuse the longest common prefix.
    """
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    eos_ids = tokenizer.eos_token_ids
    tokens: list[int] = []
    pending = ""
    emitted_len = 0  # number of characters already yielded
    finish_reason = "length"

    if draft_model is not None:
        from .generate import speculative_generate
        gen = speculative_generate(
            engine, draft_model, prompt_ids,
            max_tokens=max_tokens,
            kv_budget=kv_budget, prefetch=prefetch,
            num_draft_tokens=num_draft_tokens,
            accept_top_k=accept_top_k,
            eos_id=next(iter(eos_ids)),
            kv_quant=kv_quant,
            kv=kv,
            processed_tokens=processed_tokens,
        )
    else:
        gen = stream_generate(engine, prompt_ids, max_tokens=max_tokens, temp=temp,
                              kv_budget=kv_budget, prefetch=prefetch,
                              kv_quant=kv_quant,
                              kv=kv,
                              processed_tokens=processed_tokens)
    try:
        for token in gen:
            tokens.append(token)
            detokenizer.add_token(token)
            pending += detokenizer.last_segment

            condition = stopping_criteria(tokens, stop_sequences, eos_ids)
            if condition.stop_met:
                finish_reason = "stop"
                if condition.trim_length > 0:
                    # Recover text before the stop sequence (handles ByteLevel
                    # BPE where the trailing context surfaces on the next token).
                    safe_text = tokenizer.decode(tokens[:-condition.trim_length])
                    leftover = safe_text[emitted_len:]
                    if leftover:
                        yield ChatEvent(leftover, None, len(tokens))
                        emitted_len += len(leftover)
                # Any accumulated pending that overlaps the stop seq is dropped.
                break

            if pending and not any(sequence_overlap(tokens, seq) for seq in stop_sequences):
                yield ChatEvent(pending, None, len(tokens))
                emitted_len += len(pending)
                pending = ""
        else:
            detokenizer.finalize()
            pending += detokenizer.last_segment
            if pending:
                yield ChatEvent(pending, None, len(tokens))
    finally:
        gen.close()
        if prefix_cache is not None:
            # The borrowed kv now holds prompt + generated tokens. commit()
            # reconciles the claim against the store's actual length (the last
            # sampled token is never fed back; an aborted speculative round can
            # overhang), so the next request reuses the correct prefix.
            prefix_cache.commit(list(prompt_ids) + tokens)

    yield ChatEvent("", finish_reason, len(tokens))


def chat_completion_response(*, completion_id, created, model, text, finish_reason,
                             prompt_tokens, completion_tokens) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chat_completion_chunk(*, completion_id, created, model, delta, finish_reason) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def error_response(message, *, error_type="invalid_request_error", code=None) -> dict:
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error_response(message, *, error_type="invalid_request_error") -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def anthropic_message_response(*, message_id, model, text, stop_reason,
                               prompt_tokens, completion_tokens) -> dict:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


def anthropic_message_start(*, message_id, model, prompt_tokens) -> dict:
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
        },
    }


def anthropic_content_block_start(*, index: int = 0) -> dict:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    }


def anthropic_content_block_delta(*, index: int, text: str) -> dict:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


def anthropic_content_block_stop(*, index: int = 0) -> dict:
    return {"type": "content_block_stop", "index": index}


def anthropic_message_delta(*, stop_reason: str, completion_tokens: int) -> dict:
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": completion_tokens},
    }


def anthropic_message_stop() -> dict:
    return {"type": "message_stop"}


def anthropic_ping() -> dict:
    return {"type": "ping"}


@dataclass
class ServerState:
    """Everything an `OpenAIHandler` needs, built once at startup and shared
    (read-only, except the lock) across every request thread."""
    engine: StreamingEngine
    tokenizer: object
    model_name: str
    kv_budget: int
    prefetch: bool
    lock: threading.Lock
    draft_model: object | None = None
    num_draft_tokens: int = 16
    accept_top_k: int = 1
    kv_quant: KVQuant | None = None
    prefix_cache: PrefixCache | None = None   # PrefixCache slot; None when disabled
    prefix_scratch: object | None = None  # TemporaryDirectory backing prefix_cache


class OpenAIHandler(BaseHTTPRequestHandler):
    """Routes /v1/models and /v1/chat/completions. Generation is serialized
    behind `state.lock`: the engine, KVStore, and weight streams are
    single-stream and disk-bound, so concurrent generations would thrash the
    same I/O paths rather than parallelize -- requests simply queue."""

    def __init__(self, *args, state: ServerState, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args) -> None:
        pass  # quiet by default

    # -- response helpers ----------------------------------------------------

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_error(self, status: int, message: str, **kwargs) -> None:
        self._write_json(status, error_response(message, **kwargs))

    def _write_chunk(self, payload: dict) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    # -- routes ---------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._write_json(200, {
                "object": "list",
                "data": [{
                    "id": self.state.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nunspark",
                }],
            })
        elif self.path.startswith("/v1/models/"):
            self._write_json(200, {
                "id": self.state.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "nunspark",
            })
        elif self.path == "/v1/models/list":
            self._write_json(200, {
                "data": [{
                    "id": self.state.model_name,
                    "type": "model",
                    "display_name": self.state.model_name,
                    "created_at": "2024-01-01T00:00:00Z",
                }],
            })
        else:
            self._write_error(404, f"Not found: {self.path}", code="not_found")

    def do_POST(self) -> None:
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_openai_chat()
        elif self.path in ("/v1/messages", "/messages"):
            self._handle_anthropic_message()
        else:
            self._write_error(404, f"Not found: {self.path}", code="not_found")

    # -- OpenAI chat -----------------------------------------------------------

    def _handle_openai_chat(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._write_error(400, f"Invalid JSON in request body: {e}")
            return

        messages = body.get("messages")
        if not messages:
            self._write_error(400, "'messages' is required and must be non-empty")
            return

        model = body.get("model")
        if model is not None and model != self.state.model_name:
            self._write_error(404, f"Unknown model: {model!r}", code="model_not_found")
            return

        try:
            prompt_ids = self.state.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True)
        except Exception as e:
            self._write_error(400, f"Failed to apply chat template: {e}")
            return

        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature") or 0.0)
        max_tokens = int(body.get("max_tokens") or 256)
        stops = stop_id_sequences(self.state.tokenizer, body.get("stop"))

        with self.state.lock:
            if stream:
                self._stream_chat_completion(prompt_ids, temperature, max_tokens, stops)
            else:
                self._chat_completion(prompt_ids, temperature, max_tokens, stops)

    # -- Anthropic messages ----------------------------------------------------

    def _handle_anthropic_message(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._write_json(400, anthropic_error_response(f"Invalid JSON: {e}"))
            return

        messages = body.get("messages")
        if not messages:
            self._write_json(400, anthropic_error_response("'messages' is required"))
            return

        # Accept any model name — local server only has one model loaded.
        model = body.get("model") or self.state.model_name

        # Anthropic puts system as a top-level field, not in messages.
        system_text = body.get("system")
        chat_messages = []
        if system_text:
            if isinstance(system_text, list):
                # Anthropic allows system to be a list of content blocks.
                system_text = " ".join(
                    b.get("text", "") for b in system_text if b.get("type") == "text")
            chat_messages.append({"role": "system", "content": system_text})
        chat_messages.extend(messages)

        try:
            prompt_ids = self.state.tokenizer.apply_chat_template(
                chat_messages, add_generation_prompt=True, tokenize=True)
        except Exception as e:
            self._write_json(400, anthropic_error_response(
                f"Failed to apply chat template: {e}"))
            return

        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature") or 0.0)
        max_tokens = int(body.get("max_tokens") or 256)
        stop_sequences = body.get("stop_sequences") or []
        stops = stop_id_sequences(self.state.tokenizer, stop_sequences)

        with self.state.lock:
            if stream:
                self._stream_anthropic_message(
                    prompt_ids, temperature, max_tokens, stops)
            else:
                self._anthropic_message(
                    prompt_ids, temperature, max_tokens, stops)

    # -- chat completion bodies -----------------------------------------------

    def _events(self, prompt_ids, temperature, max_tokens, stops):
        # Prompt-prefix reuse: begin() trims the slot to the longest prefix it
        # already holds and returns the suffix still to prefill. Runs under
        # state.lock (held by do_POST), as the single slot is shared.
        pc = self.state.prefix_cache
        if pc is not None:
            kv, suffix = pc.begin(prompt_ids)
            processed = len(prompt_ids) - len(suffix)
        else:
            kv, processed = None, 0
        return stream_chat_completion(
            self.state.engine, self.state.tokenizer, prompt_ids,
            max_tokens=max_tokens, temp=temperature,
            kv_budget=self.state.kv_budget, prefetch=self.state.prefetch,
            stop_sequences=stops,
            draft_model=self.state.draft_model,
            num_draft_tokens=self.state.num_draft_tokens,
            accept_top_k=self.state.accept_top_k,
            kv_quant=self.state.kv_quant,
            kv=kv,
            processed_tokens=processed,
            prefix_cache=pc,
        )

    def _chat_completion(self, prompt_ids, temperature, max_tokens, stops) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        text_parts: list[str] = []
        finish_reason = "length"
        completion_tokens = 0

        for event in self._events(prompt_ids, temperature, max_tokens, stops):
            if event.segment:
                text_parts.append(event.segment)
            completion_tokens = event.completion_tokens
            if event.finish_reason is not None:
                finish_reason = event.finish_reason

        self._write_json(200, chat_completion_response(
            completion_id=completion_id, created=created, model=self.state.model_name,
            text="".join(text_parts), finish_reason=finish_reason,
            prompt_tokens=len(prompt_ids), completion_tokens=completion_tokens,
        ))

    def _stream_chat_completion(self, prompt_ids, temperature, max_tokens, stops) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        sent_role = False
        for event in self._events(prompt_ids, temperature, max_tokens, stops):
            if event.segment:
                delta = {"content": event.segment}
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True
                self._write_chunk(chat_completion_chunk(
                    completion_id=completion_id, created=created, model=self.state.model_name,
                    delta=delta, finish_reason=None))
            if event.finish_reason is not None:
                self._write_chunk(chat_completion_chunk(
                    completion_id=completion_id, created=created, model=self.state.model_name,
                    delta={}, finish_reason=event.finish_reason))

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _anthropic_message(self, prompt_ids, temperature, max_tokens, stops) -> None:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        text_parts: list[str] = []
        stop_reason = "max_tokens"
        completion_tokens = 0

        for event in self._events(prompt_ids, temperature, max_tokens, stops):
            if event.segment:
                text_parts.append(event.segment)
            completion_tokens = event.completion_tokens
            if event.finish_reason is not None:
                stop_reason = event.finish_reason

        self._write_json(200, anthropic_message_response(
            message_id=message_id, model=self.state.model_name,
            text="".join(text_parts), stop_reason=stop_reason,
            prompt_tokens=len(prompt_ids), completion_tokens=completion_tokens,
        ))

    def _write_sse(self, event: str, data: dict) -> None:
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _stream_anthropic_message(self, prompt_ids, temperature, max_tokens, stops) -> None:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        self._write_sse("message_start", anthropic_message_start(
            message_id=message_id, model=self.state.model_name,
            prompt_tokens=len(prompt_ids)))
        self._write_sse("content_block_start", anthropic_content_block_start())

        sent_text = ""
        completion_tokens = 0
        stop_reason = "max_tokens"

        for event in self._events(prompt_ids, temperature, max_tokens, stops):
            if event.segment:
                self._write_sse("content_block_delta", anthropic_content_block_delta(
                    index=0, text=event.segment))
                sent_text += event.segment
            completion_tokens = event.completion_tokens
            if event.finish_reason is not None:
                stop_reason = event.finish_reason

        self._write_sse("content_block_stop", anthropic_content_block_stop())
        self._write_sse("message_delta", anthropic_message_delta(
            stop_reason=stop_reason, completion_tokens=completion_tokens))
        self._write_sse("message_stop", anthropic_message_stop())
        self._write_sse("ping", anthropic_ping())


def build_server(
    packed_dir: str | Path, host: str, port: int, *,
    model_name: str | None = None,
    budget_bytes: int,
    kv_budget: int,
    prefetch: bool = True,
    io_threads: int = 1,
    warm_window: int = 1,
    draft_model_path: str | None = None,
    num_draft_tokens: int = 16,
    accept_top_k: int = 1,
    kv_quant: KVQuant | None = None,
    use_prefix_cache: bool = True,
    lookahead_prefetch: bool = False,
    wire_limit: bool = True,
) -> tuple[HTTPServer, ServerState]:
    """Load the manifest, engine, and tokenizer, and bind an HTTP server.

    Split out from `run_server` so callers (tests, or anything wanting to run
    the server on a background thread) can drive `serve_forever` themselves
    and clean up via `shutdown_server`.
    
    Uses HTTPServer (single-threaded) instead of ThreadingHTTPServer because
    MLX GPU operations must happen in the same thread. Since all GPU operations
    are already serialized with state.lock, this doesn't lose performance."""
    packed_dir = Path(packed_dir)
    manifest = Manifest.load(packed_dir / "manifest.json")
    engine = StreamingEngine(
        packed_dir, manifest, budget_bytes=budget_bytes, prefetch=prefetch,
        io_threads=io_threads, warm_window=warm_window,
        lookahead_prefetch=lookahead_prefetch, wire_limit=wire_limit,
    )

    from .generate import check_kv_quant_support
    try:
        check_kv_quant_support(engine, kv_quant)
    except ValueError:
        engine.close()
        raise

    tokenizer = load_tokenizer(packed_dir)

    # Load draft model for speculative decoding if provided
    draft_model = None
    if draft_model_path:
        from mlx_lm import load as load_mlxlm
        draft_model, _ = load_mlxlm(draft_model_path)

    # Single-slot prompt-prefix cache (server-lifetime scratch dir). The cache
    # is an optimization; --no-prefix-cache (use_prefix_cache=False) restores
    # the per-request throwaway-KVStore behavior.
    prefix_cache = None
    prefix_scratch = None
    if use_prefix_cache:
        from .prefix_cache import PrefixCache
        prefix_scratch = tempfile.TemporaryDirectory(prefix="nunspark_prefix_")
        prefix_cache = PrefixCache(
            prefix_scratch.name, kv_budget=kv_budget, prefetch=prefetch,
            cache_kinds=engine.cache_kinds, kv_quant=kv_quant)

    state = ServerState(
        engine=engine, tokenizer=tokenizer,
        model_name=model_name or packed_dir.name,
        kv_budget=kv_budget, prefetch=prefetch, lock=threading.Lock(),
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        accept_top_k=accept_top_k,
        kv_quant=kv_quant,
        prefix_cache=prefix_cache,
        prefix_scratch=prefix_scratch,
    )
    server = HTTPServer((host, port), partial(OpenAIHandler, state=state))
    return server, state


def shutdown_server(server: ThreadingHTTPServer, state: ServerState) -> None:
    """Stop serving and release the engine. Safe to call after `serve_forever`
    returns (its internal state is already marked shut down) or from another
    thread to stop one running in the background."""
    server.shutdown()
    server.server_close()
    if state.prefix_cache is not None:
        state.prefix_cache.close()        # joins the KVStore worker, unlinks scratch
    if state.prefix_scratch is not None:
        state.prefix_scratch.cleanup()
    state.engine.close()


def run_server(packed_dir, host, port, **kwargs) -> None:
    server, state = build_server(packed_dir, host, port, **kwargs)
    print(f"nunspark serving {state.model_name!r} at http://{host}:{port}", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        shutdown_server(server, state)
