"""`nunspark bench` -- a community-runnable benchmark suite.

Adapted from scripts/m1_baseline.py (the internal Plan 4 baseline runner):
same three workload prompts (code / prose / reasoning), same per-run metrics
(decode-phase tok/s excluding prefill, cache stats, SpecStats, prefetch
stats, peak memory via mx.get_peak_memory), same fresh-engine-per-run
discipline. This module ports that logic into the package (no import from
scripts/) and adds a system-info probe plus a GitHub-markdown report the
user can paste into an issue.
"""
from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

from .manifest import Manifest
from .engine import StreamingEngine
from .generate import stream_generate, speculative_generate, SpecStats

_UNITS = [("TB", 1000**4), ("GB", 1000**3), ("MB", 1000**2), ("KB", 1000),
          ("T", 1000**4), ("G", 1000**3), ("M", 1000**2), ("K", 1000), ("B", 1)]


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    for suffix, mult in _UNITS:
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


# code / prose / reasoning -- report.md's three workload styles (same prompts
# as scripts/m1_baseline.py, so results stay comparable to the internal runs).
PROMPTS = [
    ("code",
     "Implement an LRU cache in TypeScript with O(1) get and put operations. "
     "Provide the full class definition and briefly explain your design."),
    ("prose",
     "Write a design essay proposing a SaaS billing system: pricing tiers, "
     "metered usage, invoicing, and how you would handle failed payments and "
     "proration. Reason through the tradeoffs -- don't just list features."),
    ("reasoning",
     "A train leaves station A at 60 mph heading toward station B, which is "
     "300 miles away. Thirty minutes later, a second train leaves station B "
     "heading toward station A at 90 mph. How far from station A do the two "
     "trains meet, and how long after the first train departed? Show your "
     "work step by step."),
]


def _load_tokenizer(packed: Path):
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    return load_tokenizer(packed)


def _encode(tokenizer, prompt: str) -> list[int]:
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
        if isinstance(ids, str):
            ids = tokenizer.encode(ids)
    except Exception:
        ids = tokenizer.encode(prompt)
    return list(ids)


def _run_one(packed: Path, manifest: Manifest, ids: list[int], eos, label: str, mode: str,
            *, budget_bytes: int, max_tokens: int, draft_tokens: int,
            draft_model=None) -> dict:
    """Build a FRESH engine for this run (independent cache stats / peak memory),
    stream up to max_tokens greedy or speculative tokens via the existing
    generate() entry points, and return a metrics dict. Mirrors
    scripts/m1_baseline.py:_run_one."""
    engine = StreamingEngine(packed, manifest, budget_bytes=budget_bytes)
    try:
        mx.reset_peak_memory()
        spec_stats = SpecStats() if mode == "spec" else None

        if mode == "spec":
            gen = speculative_generate(
                engine, draft_model, ids, max_tokens=max_tokens,
                num_draft_tokens=draft_tokens, accept_top_k=1,
                eos_id=eos, stats=spec_stats,
            )
        else:
            gen = stream_generate(engine, ids, max_tokens=max_tokens, temp=0.0)

        # Time-to-first-token includes prefill; everything after is decode-only.
        t0 = time.perf_counter()
        first = next(gen)
        t_prefill = time.perf_counter() - t0
        out_ids = [first]

        t1 = time.perf_counter()
        if first != eos:
            for tok in gen:
                out_ids.append(tok)
                if tok == eos:
                    break
        t_decode = time.perf_counter() - t1

        decode_tokens = len(out_ids) - 1
        tok_s_decode = decode_tokens / t_decode if t_decode > 0 and decode_tokens > 0 else 0.0
        peak_bytes = mx.get_peak_memory()
        cache_stats = engine.cache.stats()
        prefetch_stats = engine.prefetch_stats()
        bytes_loaded_total = sum(cache_stats["bytes_loaded"].values())
        tokens_total = len(out_ids)
        bytes_per_token = bytes_loaded_total / tokens_total if tokens_total else 0.0
    finally:
        engine.close()

    eh, em = cache_stats["hits"]["expert"], cache_stats["misses"]["expert"]
    has_experts = (eh + em) > 0
    expert_hit_pct = (100.0 * eh / (eh + em)) if has_experts else None

    result = {
        "label": label,
        "mode": mode,
        "prompt_tokens": len(ids),
        "tokens_generated": len(out_ids),
        "decode_tokens": decode_tokens,
        "wall_time_prefill_s": t_prefill,
        "wall_time_decode_s": t_decode,
        "wall_time_total_s": t_prefill + t_decode,
        "tok_s_decode": tok_s_decode,
        "peak_memory_gb": peak_bytes / 1e9,
        "cache_stats": cache_stats,
        "prefetch_stats": prefetch_stats,
        "bytes_loaded_total": bytes_loaded_total,
        "bytes_per_token": bytes_per_token,
        "budget_bytes": budget_bytes,
        "has_experts": has_experts,
        "expert_hit_pct": expert_hit_pct,
    }
    if spec_stats is not None:
        result["spec_stats"] = {
            "target_passes": spec_stats.target_passes,
            "tokens_emitted": spec_stats.tokens_emitted,
            "draft_tokens_proposed": spec_stats.draft_tokens_proposed,
            "accepted_total": spec_stats.accepted_total,
            "accepted_offpath": spec_stats.accepted_offpath,
            "multiplier": spec_stats.multiplier,
            "deviation_rate": spec_stats.deviation_rate,
            "draft_tokens_per_sweep": draft_tokens,
        }
    return result


def run_bench(
    packed: Path,
    *,
    draft: str | None,
    draft_tokens: int = 24,
    budget: int | str = "8GB",
    max_tokens: int = 100,
    workloads: list[str] | None = None,
    out: Path | None = None,
) -> list[dict]:
    """Run the bench suite over an already-packed model dir.

    For each workload prompt in `workloads` (default: all of PROMPTS' labels),
    runs greedy decode, then (if `draft` is given) speculative decode. Prints
    live progress as runs complete. Returns the list of per-run metrics dicts;
    writes them (plus a meta block) to `out` as JSON if given.
    """
    packed = Path(packed)
    manifest = Manifest.load(packed / "manifest.json")
    budget_bytes = _parse_size(budget) if isinstance(budget, str) else int(budget)

    tokenizer = _load_tokenizer(packed)
    eos = getattr(tokenizer, "eos_token_id", None)

    prompts = PROMPTS if not workloads else [(l, p) for l, p in PROMPTS if l in workloads]

    draft_model = None
    if draft:
        from mlx_lm import load as load_full
        print(f"Loading draft {draft} ...")
        draft_model, draft_tok = load_full(draft)
        try:
            probe = "The quick brown fox jumps over the lazy dog 0123456789."
            if list(tokenizer.encode(probe)) != list(draft_tok.encode(probe)):
                print("WARNING: draft/target tokenizers disagree on a probe string -- "
                      "they do not share a vocab; acceptance will likely be ~0.")
        except Exception:
            pass

    results: list[dict] = []
    for label, prompt in prompts:
        ids = _encode(tokenizer, prompt)
        print(f"\n=== {label} ({len(ids)} prompt tokens) ===")

        print("  greedy ...")
        r = _run_one(packed, manifest, ids, eos, label, "greedy",
                     budget_bytes=budget_bytes, max_tokens=max_tokens,
                     draft_tokens=draft_tokens)
        results.append(r)
        print(f"    {r['tok_s_decode']:.2f} tok/s, peak {r['peak_memory_gb']:.2f} GB, "
              f"{r['tokens_generated']} tokens")

        if draft_model is not None:
            print(f"  spec (K={draft_tokens}) ...")
            r = _run_one(packed, manifest, ids, eos, label, "spec",
                         budget_bytes=budget_bytes, max_tokens=max_tokens,
                         draft_tokens=draft_tokens, draft_model=draft_model)
            results.append(r)
            print(f"    {r['tok_s_decode']:.2f} tok/s, M={r['spec_stats']['multiplier']:.2f}, "
                  f"peak {r['peak_memory_gb']:.2f} GB, {r['tokens_generated']} tokens")

    if out is not None:
        import json
        out = Path(out)
        if out.parent != Path("."):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "meta": {
                "packed": str(packed),
                "draft": draft,
                "draft_tokens": draft_tokens,
                "max_tokens": max_tokens,
                "budget_bytes": budget_bytes,
            },
            "runs": results,
        }, indent=2))
        print(f"\nwrote {out}")

    return results


def _sh(cmd: list[str]) -> str:
    """Run a subprocess and return its stripped stdout, or "unknown" on any
    failure. Must never raise -- system_info() is best-effort by design."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return "unknown"
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def system_info() -> dict:
    """Best-effort machine/software fingerprint for a shareable report.
    Every field falls back to "unknown" independently -- this function must
    never raise, even on non-Mac hosts or in minimal environments."""
    chip = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])

    ram_gb = "unknown"
    ram_raw = _sh(["sysctl", "-n", "hw.memsize"])
    try:
        ram_gb = f"{int(ram_raw) / 2**30:.0f}"   # GiB: hw.memsize is 17179869184 on a "16 GB" Mac
    except (ValueError, TypeError):
        pass

    macos_version = _sh(["sw_vers", "-productVersion"])
    python_version = sys.version.split()[0]

    try:
        nunspark_version = importlib.metadata.version("nunspark")
    except Exception:
        nunspark_version = "dev"

    try:
        mlx_version = importlib.metadata.version("mlx")
    except Exception:
        mlx_version = "unknown"

    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "macos_version": macos_version,
        "python_version": python_version,
        "nunspark_version": nunspark_version,
        "mlx_version": mlx_version,
    }


def format_report(info: dict, results: list[dict], model: str) -> str:
    """Render a GitHub-markdown block summarizing `results`, suitable for
    pasting into an issue/discussion. "expert hit%" / "MB/token" columns show
    "--" for runs without expert pieces (dense models)."""
    lines = []
    lines.append(f"### NunSpark bench -- {model}")
    ram = f"{info.get('ram_gb', 'unknown')} GB RAM" if info.get("ram_gb") != "unknown" \
        else "unknown RAM"
    lines.append(
        f"{info.get('chip', 'unknown')}, {ram}, macOS {info.get('macos_version', 'unknown')} "
        f"-- nunspark {info.get('nunspark_version', 'unknown')}, "
        f"mlx {info.get('mlx_version', 'unknown')}"
    )
    lines.append("")
    lines.append("| workload | mode | tok/s | M | expert hit% | MB/token | peak GB |")
    lines.append("|---|---|---|---|---|---|---|")

    has_spec = False
    draft_tokens = None
    budget_bytes = None
    for r in results:
        m = r.get("spec_stats", {}).get("multiplier")
        m_str = f"{m:.2f}" if m is not None else "—"
        eh = r.get("expert_hit_pct")
        eh_str = f"{eh:.1f}" if eh is not None else "—"
        mb_tok = r.get("bytes_per_token")
        mb_str = f"{mb_tok / 1e6:.0f}" if r.get("has_experts") and mb_tok is not None \
            else "—"
        lines.append(
            f"| {r.get('label', '?')} | {r.get('mode', '?')} | "
            f"{r.get('tok_s_decode', 0.0):.2f} | {m_str} | {eh_str} | {mb_str} | "
            f"{r.get('peak_memory_gb', 0.0):.2f} |"
        )
        if r.get("mode") == "spec":
            has_spec = True
            if draft_tokens is None:
                draft_tokens = r.get("spec_stats", {}).get("draft_tokens_per_sweep")
        if budget_bytes is None:
            budget_bytes = r.get("budget_bytes")

    budget_str = f"{budget_bytes / 1e9:.0f}GB" if budget_bytes else "unknown"
    settings = f"settings: budget={budget_str}"
    if results:
        max_tokens = max((r.get("tokens_generated", 0) for r in results), default=None)
        if max_tokens:
            settings += f", max_tokens<={max_tokens}"
    settings += f", speculative={'on' if has_spec else 'off'}"
    if draft_tokens is not None:
        settings += f", K={draft_tokens}"
    lines.append("")
    lines.append(settings)

    return "\n".join(lines)
