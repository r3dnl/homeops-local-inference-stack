"""On-demand benchmark control service for the Qwen3.8-27B vLLM stack.

Runs benchmarks/context_bench.py as a subprocess, then exposes:

  GET  /docs              OpenAPI docs (headless REST API; the Streamlit
                          control UI is a separate service hitting this API)
  GET  /config            effective benchmark options
  GET  /status            active-run state + last result summary
  POST /bench             start a run  body: {quick?, cached?, sizes?,
                          runs?, max_tokens?, tag?}
  GET  /runs              run index (all stored runs, incl. local ones,
                          newest first)
  POST /runs/select       {"run": "<id>"} — pick the run the dashboard
                          table shows (bench_selected_ctx_value)
  GET  /bench             runs started by this process, newest first
  GET  /bench/{run_id}   state + report of one run
  GET  /metrics           Prometheus text (per-run + selected-run series)

Idle-safe: the container autostarts with the stack but touches the GPU
only while a run is in flight. Single-flight: one run at a time.

Runs index: benchmarks/results/runs.jsonl (one line per completed run:
id, tag, times, exit code, medians) plus the suite's own report JSONs,
so local (host-side) runs show up in the index too.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parent          # <repo>/benchmarks (or /app/benchmarks)
REPO_ROOT = APP_ROOT.parent
SUITE = REPO_ROOT / "benchmarks" / "context_bench.py"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
LATEST = RESULTS_DIR / "latest.json"
RUNS_INDEX = RESULTS_DIR / "runs.jsonl"

RESULTS_RE = re.compile(r"results:\s*(\S+\.json)")
REPORT_STEM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_(.*))?$")
METRIC_KEYS = ("decode_tps", "ttft_s", "prefill_tps", "e2e_s")
MAX_METRIC_RUNS = 15   # how many recent runs get per-run metric series


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key) or default


def effective_config() -> dict[str, Any]:
    """What a run will actually use (process env, same precedence as the suite)."""
    return {
        "suite": str(SUITE),
        "suite_present": SUITE.exists(),
        "base_url": _env("BENCH_BASE_URL", "http://localhost:8020"),
        "model": _env("BENCH_MODEL", "auto-detect (GET /v1/models, all models online)"),
        "auth": ("basic" if _env("BENCH_BASIC_AUTH")
                 else "bearer" if _env("BENCH_API_KEY")
                 else "none"),
        "default_sizes": _env("BENCH_SIZES", "1024,8192,32768,65536,131072,256000"),
        "default_runs": _env("BENCH_RUNS", "3"),
        "default_max_tokens": _env("BENCH_MAX_TOKENS", "256"),
        "timeout_s": _env("BENCH_TIMEOUT_S", "600"),
        "tokenizer": _env("BENCH_TOKENIZER", "auto-detect (./models/*/tokenizer.json)"),
    }


def sanitize_tag(tag: str) -> str:
    """Run tag -> Prometheus-label-safe, legend-friendly (<= 40 chars)."""
    tag = re.sub(r"[^a-zA-Z0-9._-]", "-", tag.strip())
    return tag[:40] or ""


class BenchRequest(BaseModel):
    quick: bool = False
    cached: bool = False
    sizes: list[int] | None = None
    runs: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    tag: str = ""


def ctx_label(label: str) -> str:
    """'32k' -> '032_32k' (zero-padded so Grafana legends sort numerically)."""
    m = re.match(r"(\d+)k?", label)
    return f"{int(m.group(1)):03d}_{label}" if m else label


def context_medians(report: dict) -> dict[str, dict[str, float]]:
    """Per-context medians over the timed, non-cached, successful runs
    (same selection the suite uses for its markdown table)."""
    rows: dict[str, list[dict]] = {}
    for r in report.get("runs", []):
        if r.get("cached") or r.get("note", "").startswith("error:") or r.get("prompt_tokens", 0) <= 0:
            continue
        rows.setdefault(r["size_label"], []).append(r)
    out: dict[str, dict[str, float]] = {}
    for label, rs in rows.items():
        med = lambda key: statistics.median([x[key] for x in rs if x.get(key) is not None])
        out[ctx_label(label)] = {
            "prompt_tokens": med("prompt_tokens"),
            "ttft_s": med("ttft_s"),
            "prefill_tps": med("prefill_tps"),
            "decode_tps": med("decode_tps"),
            "e2e_s": med("e2e_s"),
        }
    return dict(sorted(out.items()))


class Run:
    def __init__(self, run_id: str, req: BenchRequest):
        self.id = run_id
        self.req = req
        self.tag = sanitize_tag(req.tag)
        self.state = "starting"                    # starting|running|done|failed
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_ts = time.time()
        self.finished_at: str | None = None
        self.exit_code: int | None = None
        self.error: str | None = None
        self.report: dict | None = None
        self._proc: subprocess.Popen | None = None

    def to_dict(self, include_report: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "tag": self.tag,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "params": self.req.model_dump(),
        }
        if self.error:
            d["error"] = self.error
        if include_report and self.report:
            d["report"] = self.report
        return d


class BenchService:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.index: dict[str, dict[str, Any]] = {}   # run id -> index entry
        self.runs_total = 0
        self.runs_failed = 0
        self.last_run_at: float | None = None
        self.last_run_id: str | None = None
        self.selected_run: str | None = None
        self._load_index()

    # ------------------------------------------------------------------ index
    def _load_index(self) -> None:
        """Discover stored runs: runs.jsonl lines + the suite's report JSONs
        (the latter also picks up host-side local runs)."""
        seen: dict[str, dict[str, Any]] = {}
        selected: str | None = None
        if RUNS_INDEX.exists():
            for line in RUNS_INDEX.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("op") == "select":
                    selected = rec.get("run")
                elif rec.get("id"):
                    seen[rec["id"]] = rec
        for f in sorted(RESULTS_DIR.glob("*.json")):
            if f.name == "latest.json":
                continue
            m = REPORT_STEM_RE.match(f.stem)
            if not m:
                continue
            try:
                report = json.loads(f.read_text())
            except Exception:
                continue
            if "runs" not in report:
                continue
            entry = seen.get(f.stem, {"id": f.stem, "state": "done (stored)"})
            entry.setdefault("report_file", f.name)
            entry.setdefault("tag", m.group(2) or "")
            if not entry.get("finished_ts"):
                try:
                    entry["finished_ts"] = datetime.strptime(
                        m.group(1).replace("_", " "), "%Y-%m-%d %H:%M:%S").timestamp()
                except Exception:
                    entry["finished_ts"] = f.stat().st_mtime
            entry["medians"] = context_medians(report)
            seen[f.stem] = entry
        # newest first by finished timestamp
        self.index = {k: v for k, v in sorted(
            seen.items(), key=lambda kv: kv[1].get("finished_ts") or 0, reverse=True)}
        self.selected_run = selected if selected in self.index else None

    def _record_run(self, run: Run, report: dict | None) -> None:
        entry = {
            "id": run.id,
            "tag": run.tag,
            "state": run.state,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "exit_code": run.exit_code,
            "report_file": LATEST.name if report else None,
            "finished_ts": time.time(),
            "medians": context_medians(report) if report else {},
        }
        self.index[run.id] = entry
        self.last_run_at = time.time()
        self.last_run_id = run.id
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RUNS_INDEX, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------- runs
    def active(self) -> Run | None:
        return next((r for r in self.runs.values()
                     if r.state in ("starting", "running")), None)

    def start(self, req: BenchRequest) -> Run:
        if self.active() is not None:
            raise HTTPException(status_code=409, detail="a benchmark run is already in flight")
        if not effective_config()["suite_present"]:
            raise HTTPException(status_code=500, detail=f"suite missing: {SUITE}")
        run = Run(time.strftime("%Y%m%d-%H%M%S"), req)
        self.runs[run.id] = run
        env = dict(os.environ)
        if req.sizes:
            env["BENCH_SIZES"] = ",".join(str(s) for s in req.sizes)
        if req.runs is not None:
            env["BENCH_RUNS"] = str(req.runs)
        if req.max_tokens is not None:
            env["BENCH_MAX_TOKENS"] = str(req.max_tokens)
        cmd = [sys.executable, str(SUITE)]
        if req.cached:
            cmd.append("--cached")
        if req.quick:
            cmd.append("--quick")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RESULTS_DIR / f"{run.id}.log"
        log = open(log_path, "w")
        run._proc = subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=log, stderr=subprocess.STDOUT)
        run.state = "running"
        threading.Thread(target=self._reap, args=(run, log), daemon=True).start()
        return run

    def _reap(self, run: Run, log) -> None:
        proc = run._proc
        assert proc is not None
        proc.wait()
        log.close()
        run.exit_code = proc.returncode
        run.finished_at = datetime.now(timezone.utc).isoformat()
        run.state = "done" if proc.returncode == 0 else "failed"
        self.runs_total += 1
        if proc.returncode != 0:
            self.runs_failed += 1
        try:
            with open(log.name, "r", errors="replace") as f:
                out_lines = f.readlines()
        except Exception:
            out_lines = []
        m = None
        for line in reversed(out_lines):
            m = RESULTS_RE.search(line)
            if m:
                break
        if m:
            try:
                run.report = json.loads(Path(m.group(1)).read_text())
            except Exception as exc:
                run.error = f"could not read results file: {exc}"
        elif run.state == "failed":
            run.error = "suite exited non-zero:\n" + "".join(out_lines[-20:]).strip()
        self._record_run(run, run.report)

    def select_run(self, run_id: str) -> dict:
        entry = self.index.get(run_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        self.selected_run = run_id
        with open(RUNS_INDEX, "a") as f:
            f.write(json.dumps({"op": "select", "run": run_id}) + "\n")
        return entry

    def selected_entry(self) -> dict | None:
        if self.selected_run in self.index:
            return self.index[self.selected_run]
        if self.last_run_id:
            return self.index.get(self.last_run_id)
        return next(iter(self.index.values()), None)


svc = BenchService()
app = FastAPI(title="qwen3.8-27b benchmark control", docs_url="/docs")


@app.get("/config")
async def config() -> dict:
    return effective_config()


@app.get("/status")
async def status() -> dict:
    active = svc.active()
    entry = svc.selected_entry()
    return {
        "active": active.to_dict() if active else None,
        "last_run_id": svc.last_run_id,
        "last_run_at": svc.last_run_at,
        "selected_run": entry,
        "runs_total": svc.runs_total,
        "runs_failed": svc.runs_failed,
        "index": [
            {k: v[k] for k in ("id", "tag", "state", "finished_at", "medians")
             if k in v}
            for v in list(svc.index.values())[:20]
        ],
    }


@app.post("/bench", status_code=202)
async def bench_start(req: BenchRequest) -> dict:
    run = svc.start(req)
    return run.to_dict()


@app.get("/runs")
async def runs_list() -> list[dict]:
    return list(svc.index.values())


@app.post("/runs/select")
async def runs_select(payload: dict) -> dict:
    run_id = payload.get("run", "")
    if not run_id:
        raise HTTPException(status_code=400, detail="body: {\"run\": \"<id>\"}")
    return svc.select_run(run_id)


@app.get("/bench")
async def bench_list() -> list[dict]:
    return [r.to_dict() for r in reversed(list(svc.runs.values()))[:20]]


@app.get("/bench/{run_id}")
async def bench_get(run_id: str) -> dict:
    run = svc.runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return run.to_dict(include_report=True)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    active = svc.active()
    lines = [
        "# HELP bench_status 0=idle, 1=a benchmark run is in flight",
        "# TYPE bench_status gauge",
        f"bench_status {1 if active else 0}",
        "# HELP bench_runs_total benchmark runs started (this process)",
        "# TYPE bench_runs_total counter",
        f"bench_runs_total {svc.runs_total}",
        "# HELP bench_runs_failed_total benchmark runs that ended non-zero",
        "# TYPE bench_runs_failed_total counter",
        f"bench_runs_failed_total {svc.runs_failed}",
    ]
    sel = svc.selected_entry()
    if sel and sel.get("medians"):
        lines += [
            "# HELP bench_selected_run_timestamp_seconds when the selected run finished",
            "# TYPE bench_selected_run_timestamp_seconds gauge",
            f"bench_selected_run_timestamp_seconds {sel.get('finished_ts') or 0:.0f}",
            "# HELP bench_selected_ctx_value per-context medians of the selected run",
            "# TYPE bench_selected_ctx_value gauge",
        ]
        for ctx, vals in sel["medians"].items():
            for k in METRIC_KEYS:
                lines.append(f'bench_selected_ctx_value{{context="{ctx}",metric="{k}"}} {vals[k]:.3f}')
    recent = [v for v in list(svc.index.values())[:MAX_METRIC_RUNS] if v.get("medians")]
    for entry in recent:
        run_label = entry.get("tag") or entry["id"]
        lines.append(f'bench_run_info{{run="{run_label}",id="{entry["id"]}",tag="{entry.get("tag") or ""}"}} 1')
        for ctx, vals in entry["medians"].items():
            for k in METRIC_KEYS:
                lines.append(
                    f'bench_ctx_value{{context="{ctx}",metric="{k}",'
                    f'run="{run_label}",id="{entry["id"]}"}} {vals[k]:.3f}')
    return "\n".join(lines) + "\n"
