#!/usr/bin/env python3
"""
llama.cpp Token Speed Benchmark
================================
Measures prompt processing (prefill) and generation (decode) speed using
llama.cpp's built-in server timings — the same data shown in the Web UI.

Two data sources are used:
  1. The `timings` object returned in llama.cpp's /v1/chat/completions
     response, which contains exact prompt_per_second and predicted_per_second.
  2. Streaming TTFT measurement as a cross-check.

Usage:
    python llama_bench.py [OPTIONS]

Requirements:
    pip install openai requests
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import requests
from openai import OpenAI


# ── Default prompts (short → long to stress-test different generation lengths) ──

DEFAULT_PROMPTS = [
    "Explain what a hash function is in two sentences.",
    "Write a detailed overview of how TCP/IP networking works.",
    "Write a long essay about the history and future of digital forensics.",
]


@dataclass
class ServerTimings:
    """Timings reported directly by llama.cpp's server."""
    prompt_n: int = 0             # number of prompt tokens evaluated
    prompt_ms: float = 0.0        # time spent on prompt eval (ms)
    prompt_per_second: float = 0.0  # prompt processing speed (t/s)
    predicted_n: int = 0          # number of tokens generated
    predicted_ms: float = 0.0     # time spent on generation (ms)
    predicted_per_second: float = 0.0  # generation speed (t/s)


@dataclass
class RunResult:
    """Stores metrics for a single benchmark run."""
    prompt: str
    ttft: float                    # seconds to first token (client-side)
    total_tokens: int              # completion tokens (client count)
    gen_time: float                # total wall-clock time (seconds)
    tokens_per_sec: float          # client-measured overall speed
    server_timings: ServerTimings   # exact timings from llama.cpp
    raw_reply: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregate stats across all runs."""
    runs: list[RunResult] = field(default_factory=list)

    @property
    def avg_ttft(self) -> float:
        return statistics.mean(r.ttft for r in self.runs)

    @property
    def avg_prompt_tps(self) -> float:
        vals = [r.server_timings.prompt_per_second for r in self.runs
                if r.server_timings.prompt_per_second > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_decode_tps(self) -> float:
        vals = [r.server_timings.predicted_per_second for r in self.runs
                if r.server_timings.predicted_per_second > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_overall_tps(self) -> float:
        return statistics.mean(r.tokens_per_sec for r in self.runs)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.runs)


def _get_api_url(base_url: str) -> str:
    """Resolve the chat completions URL from the base."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def _get_root_url(base_url: str) -> str:
    """Strip /v1 to get the server root for native endpoints."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return root[:-3]
    return root


def parse_server_timings(data: dict) -> ServerTimings:
    """Extract the timings object from a llama.cpp response."""
    t = data.get("timings", {})
    return ServerTimings(
        prompt_n=t.get("prompt_n", 0),
        prompt_ms=t.get("prompt_ms", 0.0),
        prompt_per_second=t.get("prompt_per_second", 0.0),
        predicted_n=t.get("predicted_n", 0),
        predicted_ms=t.get("predicted_ms", 0.0),
        predicted_per_second=t.get("predicted_per_second", 0.0),
    )


def fetch_slot_timings(base_url: str) -> dict | None:
    """Try to fetch /slots endpoint for additional diagnostics."""
    try:
        resp = requests.get(f"{_get_root_url(base_url)}/slots", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def run_single(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    verbose: bool = False,
) -> RunResult:
    """
    Full mode: streaming request for TTFT + non-streaming for server timings.

    This gives you both the client-side TTFT and llama.cpp's exact internal
    measurements (prompt_per_second, predicted_per_second) — the same numbers
    you see in the llama-server Web UI.
    """

    messages = [{"role": "user", "content": prompt}]

    # ── Phase 1: Streaming request for TTFT ──
    client = OpenAI(base_url=base_url, api_key="sk-no-key-required")
    stream_kwargs = dict(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    start = time.perf_counter()
    response = client.chat.completions.create(**stream_kwargs)

    ttft = None
    stream_tokens = 0
    chunks: list[str] = []

    for chunk in response:
        now = time.perf_counter()
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            if ttft is None:
                ttft = now - start
            stream_tokens += 1
            chunks.append(delta.content)

    end = time.perf_counter()
    gen_time = end - start
    if ttft is None:
        ttft = gen_time

    # ── Phase 2: Non-streaming request for server timings ──
    api_url = _get_api_url(base_url)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    server_timings = ServerTimings()
    try:
        resp = requests.post(api_url, json=payload, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            server_timings = parse_server_timings(data)
    except Exception as e:
        if verbose:
            print(f"    [warn] Non-streaming request failed: {e}")

    overall_tps = stream_tokens / gen_time if gen_time > 0 else 0.0

    result = RunResult(
        prompt=prompt[:80],
        ttft=ttft,
        total_tokens=stream_tokens,
        gen_time=gen_time,
        tokens_per_sec=overall_tps,
        server_timings=server_timings,
        raw_reply="".join(chunks),
    )

    if verbose:
        print(f"\n--- Reply preview (first 200 chars) ---")
        print(result.raw_reply[:200])

    return result


def run_single_fast(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    verbose: bool = False,
) -> RunResult:
    """
    Fast mode: single non-streaming request.
    Gets server timings in one pass — no streaming TTFT, but half the requests.
    """
    messages = [{"role": "user", "content": prompt}]
    api_url = _get_api_url(base_url)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()
    resp = requests.post(api_url, json=payload, timeout=300)
    end = time.perf_counter()

    if resp.status_code != 200:
        raise RuntimeError(f"Server returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    server_timings = parse_server_timings(data)
    gen_time = end - start

    # Extract reply text
    raw_reply = ""
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        raw_reply = msg.get("content", "")

    total_tokens = server_timings.predicted_n or 0

    # Derive TTFT from server's prompt processing time
    ttft = server_timings.prompt_ms / 1000.0 if server_timings.prompt_ms > 0 else 0.0

    overall_tps = total_tokens / gen_time if gen_time > 0 else 0.0

    result = RunResult(
        prompt=prompt[:80],
        ttft=ttft,
        total_tokens=total_tokens,
        gen_time=gen_time,
        tokens_per_sec=overall_tps,
        server_timings=server_timings,
        raw_reply=raw_reply,
    )

    if verbose:
        print(f"\n--- Reply preview (first 200 chars) ---")
        print(result.raw_reply[:200])

    return result


def print_run(idx: int, r: RunResult) -> None:
    trunc = r.prompt if len(r.prompt) <= 60 else r.prompt[:57] + "..."
    st = r.server_timings

    print(f"\n  Run {idx}: \"{trunc}\"")
    print(f"    ┌─ Client-side ─────────────────────────────────")
    print(f"    │ TTFT:              {r.ttft:>8.3f} s")
    print(f"    │ Tokens generated:  {r.total_tokens:>8d}")
    print(f"    │ Wall-clock time:   {r.gen_time:>8.3f} s")
    print(f"    │ Overall speed:     {r.tokens_per_sec:>8.2f} t/s")

    if st.prompt_per_second > 0 or st.predicted_per_second > 0:
        print(f"    ├─ Server timings (from llama.cpp) ─────────────")
        print(f"    │ Prompt tokens:    {st.prompt_n:>8d}")
        print(f"    │ Prompt eval time: {st.prompt_ms:>8.2f} ms")
        print(f"    │ Prompt speed:     {st.prompt_per_second:>8.2f} t/s  ◀ prefill")
        print(f"    │ Predicted tokens: {st.predicted_n:>8d}")
        print(f"    │ Predict time:     {st.predicted_ms:>8.2f} ms")
        print(f"    │ Predict speed:    {st.predicted_per_second:>8.2f} t/s  ◀ decode")
    else:
        print(f"    ├─ Server timings:  not available")
        print(f"    │  (server may not include timings in response)")

    print(f"    └───────────────────────────────────────────────")


def print_summary(summary: BenchmarkSummary) -> None:
    print("\n" + "=" * 56)
    print("  BENCHMARK SUMMARY")
    print("=" * 56)
    print(f"  Runs:                {len(summary.runs):>8d}")
    print(f"  Total tokens:        {summary.total_tokens:>8d}")
    print(f"  Avg TTFT:            {summary.avg_ttft:>8.3f} s")

    if summary.avg_prompt_tps > 0:
        print(f"  Avg prompt speed:    {summary.avg_prompt_tps:>8.2f} t/s  (prefill)")
    else:
        print(f"  Avg prompt speed:        n/a")

    if summary.avg_decode_tps > 0:
        print(f"  Avg decode speed:    {summary.avg_decode_tps:>8.2f} t/s  (generation)")
    else:
        print(f"  Avg decode speed:        n/a")

    print(f"  Avg overall speed:   {summary.avg_overall_tps:>8.2f} t/s  (client wall-clock)")

    if len(summary.runs) > 1:
        tps_vals = [r.tokens_per_sec for r in summary.runs]
        print(f"  Min / Max speed:     {min(tps_vals):>8.2f} / {max(tps_vals):.2f} t/s")
        if len(tps_vals) >= 2:
            print(f"  Std-dev speed:       {statistics.stdev(tps_vals):>8.2f} t/s")

        if summary.avg_decode_tps > 0:
            decode_vals = [r.server_timings.predicted_per_second for r in summary.runs
                           if r.server_timings.predicted_per_second > 0]
            if len(decode_vals) >= 2:
                print(f"  Decode min / max:    {min(decode_vals):>8.2f} / {max(decode_vals):.2f} t/s")

    print("=" * 56)


def export_json(summary: BenchmarkSummary, path: str) -> None:
    data = {
        "avg_ttft_s": round(summary.avg_ttft, 4),
        "avg_prompt_tokens_per_sec": round(summary.avg_prompt_tps, 2),
        "avg_decode_tokens_per_sec": round(summary.avg_decode_tps, 2),
        "avg_overall_tokens_per_sec": round(summary.avg_overall_tps, 2),
        "total_tokens": summary.total_tokens,
        "runs": [
            {
                "prompt": r.prompt,
                "ttft_s": round(r.ttft, 4),
                "tokens": r.total_tokens,
                "wall_s": round(r.gen_time, 4),
                "overall_tps": round(r.tokens_per_sec, 2),
                "server_timings": {
                    "prompt_n": r.server_timings.prompt_n,
                    "prompt_ms": round(r.server_timings.prompt_ms, 2),
                    "prompt_per_second": round(r.server_timings.prompt_per_second, 2),
                    "predicted_n": r.server_timings.predicted_n,
                    "predicted_ms": round(r.server_timings.predicted_ms, 2),
                    "predicted_per_second": round(r.server_timings.predicted_per_second, 2),
                },
            }
            for r in summary.runs
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results exported to {path}")


CSV_COLUMNS = [
    "timestamp",
    "label",
    "model",
    "run",
    "prompt",
    "ttft_s",
    "wall_s",
    "tokens_generated",
    "overall_tps",
    "prompt_n",
    "prompt_ms",
    "prompt_tps",
    "predicted_n",
    "predicted_ms",
    "predicted_tps",
]


def export_csv(
    summary: BenchmarkSummary,
    path: str,
    model: str,
    label: str,
) -> None:
    """
    Append benchmark results to a CSV file.
    Creates the file with headers if it doesn't exist yet.
    Each row is one run — easy to chart over time.
    """
    file_exists = os.path.isfile(path)

    # Check if existing file has matching columns
    write_header = not file_exists
    if file_exists:
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
                if existing_header != CSV_COLUMNS:
                    write_header = False  # don't clobber, but warn
                    print(f"\n  [warn] CSV columns differ from expected — "
                          f"appending rows anyway")
        except Exception:
            pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(CSV_COLUMNS)

        for idx, r in enumerate(summary.runs, start=1):
            st = r.server_timings
            writer.writerow([
                timestamp,
                label,
                model,
                idx,
                r.prompt,
                round(r.ttft, 4),
                round(r.gen_time, 4),
                r.total_tokens,
                round(r.tokens_per_sec, 2),
                st.prompt_n,
                round(st.prompt_ms, 2),
                round(st.prompt_per_second, 2),
                st.predicted_n,
                round(st.predicted_ms, 2),
                round(st.predicted_per_second, 2),
            ])

    action = "Created" if not file_exists else "Appended to"
    print(f"\n  {action} CSV: {path}  ({len(summary.runs)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp server using its native timings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s                                    # run with defaults
  %(prog)s --prompt "Hello world" --repeat 5  # custom prompt, 5 runs
  %(prog)s --fast --warmup --json out.json    # fast mode, warmup, export
  %(prog)s --csv bench.csv --label "Q4_K_M"   # append to CSV with label
  %(prog)s --csv bench.csv --label "Q5_K_M"   # later run appends new rows
  %(prog)s --host http://192.168.1.50:8080/v1 # remote server
  %(prog)s --slots                            # show /slots diagnostics
        """,
    )
    parser.add_argument("--host", default="http://localhost:8080/v1",
                        help="Base URL of the llama.cpp OpenAI-compatible API")
    parser.add_argument("--model", default="local-model",
                        help="Model name to pass in the request")
    parser.add_argument("--prompt", action="append", dest="prompts",
                        help="Custom prompt(s). Can be repeated. Overrides defaults.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per run (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each prompt N times (default: 1)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run one short warmup request before benchmarking")
    parser.add_argument("--fast", action="store_true",
                        help="Single non-streaming request per run (no TTFT, but faster)")
    parser.add_argument("--slots", action="store_true",
                        help="Query /slots endpoint after benchmark for diagnostics")
    parser.add_argument("--json", dest="json_path", metavar="FILE",
                        help="Export results to a JSON file")
    parser.add_argument("--csv", dest="csv_path", metavar="FILE",
                        default="bench_results.csv",
                        help="CSV file to append results to (default: bench_results.csv)")
    parser.add_argument("--no-csv", action="store_true",
                        help="Disable CSV output")
    parser.add_argument("--label", default="",
                        help="Label for this run (e.g. model name, quant, config note)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a preview of each generated reply")
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    total_runs = len(prompts) * args.repeat

    run_fn = run_single_fast if args.fast else run_single

    # ── Header ──
    print(f"\n  llama.cpp Benchmark")
    print(f"  {'─' * 40}")
    print(f"  Server:  {args.host}")
    print(f"  Model:   {args.model}")
    print(f"  Mode:    {'fast (non-streaming)' if args.fast else 'full (streaming + timings)'}")
    print(f"  Prompts: {len(prompts)} × {args.repeat} repeat(s) = {total_runs} run(s)")
    print(f"  Max tokens per run: {args.max_tokens}")

    # ── Optional warmup ──
    if args.warmup:
        print("\n  Warming up...", end="", flush=True)
        try:
            run_fn(args.host, args.model, "Say hi.", max_tokens=16,
                   temperature=0.0, verbose=False)
            print(" done.")
        except Exception as e:
            print(f"\n  Warmup failed: {e}")
            sys.exit(1)

    # ── Benchmark runs ──
    summary = BenchmarkSummary()
    run_idx = 0

    for rep in range(args.repeat):
        for prompt in prompts:
            run_idx += 1
            print(f"\n  [{run_idx}/{total_runs}] Running...", flush=True)
            try:
                result = run_fn(
                    args.host, args.model, prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verbose=args.verbose,
                )
                summary.runs.append(result)
                print_run(run_idx, result)
            except Exception as e:
                print(f"  Run {run_idx} FAILED: {e}")

    if not summary.runs:
        print("\n  No successful runs. Check your server is running.")
        sys.exit(1)

    print_summary(summary)

    # ── Optional /slots diagnostics ──
    if args.slots:
        print("\n  Querying /slots endpoint...")
        slot_data = fetch_slot_timings(args.host)
        if slot_data:
            print(f"  Slots data:")
            for i, slot in enumerate(slot_data if isinstance(slot_data, list) else [slot_data]):
                state = slot.get("state", "unknown")
                n_ctx = slot.get("n_ctx", "?")
                n_past = slot.get("n_past", "?")
                print(f"    Slot {i}: state={state}, n_ctx={n_ctx}, n_past={n_past}")
        else:
            print("  /slots endpoint not available (server may need --slots flag)")

    if args.json_path:
        export_json(summary, args.json_path)

    # ── CSV export (on by default) ──
    if not args.no_csv:
        export_csv(summary, args.csv_path, model=args.model, label=args.label)


if __name__ == "__main__":
    main()
