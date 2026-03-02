#!/usr/bin/env python3
"""Web dashboard for monitoring GPT-OSS benchmarks on DGX Spark.

Shows real-time task completion, model info, timing, quality scores,
and 20B vs 120B comparison. Connects to DGX Spark via SSH (passwordless).

Usage:
    python scripts/benchmark_dashboard.py
    python scripts/benchmark_dashboard.py --host dgx-spark --port 8050
    python scripts/benchmark_dashboard.py --benchmarks-dir /home/rmarnold/benchmarks
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def ssh_exec(host: str, cmd: str, timeout: int = 10) -> str:
    """Run a command on remote host via SSH. Returns stdout or empty string."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def get_container_status(host: str) -> list[dict]:
    """Get benchmark container info from docker ps."""
    raw = ssh_exec(
        host,
        'docker ps -a --filter "name=bench-" '
        '--format "{{.Names}}|{{.Status}}|{{.Image}}"',
    )
    containers = []
    for line in raw.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            containers.append({
                "name": parts[0],
                "status": parts[1],
                "image": parts[2],
            })
    return containers


def get_model_info(host: str, bench_dir: str, model: str) -> dict:
    """Read model_info.json for a given model size."""
    path = f"{bench_dir}/gpt-oss-{model}/model_info.json"
    raw = ssh_exec(host, f"cat {path} 2>/dev/null")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {}


def get_bench_results(host: str, bench_dir: str, model: str) -> dict:
    """Read coding_bench.json, tool_use_bench.json, and humaneval data."""
    results: dict = {
        "coding": [], "tool_use": [],
        "humaneval": {"total": 0, "correct": -1},
    }
    model_dir = f"{bench_dir}/gpt-oss-{model}"

    raw = ssh_exec(host, f"cat {model_dir}/coding_bench.json 2>/dev/null")
    if raw:
        try:
            results["coding"] = json.loads(raw)
        except json.JSONDecodeError:
            pass

    raw = ssh_exec(host, f"cat {model_dir}/tool_use_bench.json 2>/dev/null")
    if raw:
        try:
            results["tool_use"] = json.loads(raw)
        except json.JSONDecodeError:
            pass

    raw = ssh_exec(
        host,
        f"wc -l < {model_dir}/humaneval_completions.jsonl 2>/dev/null",
        timeout=10,
    )
    if raw:
        try:
            results["humaneval"]["total"] = int(raw.strip())
        except ValueError:
            pass

    raw = ssh_exec(
        host, f"cat {model_dir}/humaneval_results.json 2>/dev/null", timeout=10,
    )
    if raw:
        try:
            he_data = json.loads(raw)
            results["humaneval"]["correct"] = he_data.get("correct", 0)
            results["humaneval"]["total"] = he_data.get(
                "total", results["humaneval"]["total"])
        except json.JSONDecodeError:
            pass

    return results


def get_docker_progress(host: str, container: str) -> tuple[list[dict], str]:
    """Parse docker logs for completed task progress lines."""
    raw = ssh_exec(
        host,
        f'docker logs {container} 2>&1 | tail -80',
        timeout=15,
    )
    tasks = []
    humaneval_progress = ""
    for line in raw.splitlines():
        line_s = line.strip()
        m = re.match(r'^([a-z_]+):\s+(\d+)\s+tokens', line_s)
        if m:
            tasks.append({"name": m.group(1), "tokens": int(m.group(2))})
            continue
        m = re.match(r'^\[(\d+)/(\d+)\]\s+([\d.]+)\s+prob/s', line_s)
        if m:
            humaneval_progress = f"{m.group(1)}/{m.group(2)} ({m.group(3)} prob/s)"
    return tasks, humaneval_progress


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def score_completion(completion: str) -> str:
    """Quick quality heuristic for a code completion."""
    if not completion or not completion.strip():
        return "EMPTY"
    lines = [l for l in completion.strip().split("\n") if l.strip()]
    n_lines = len(lines)
    if n_lines > 6:
        unique = len(set(lines))
        if unique < n_lines * 0.3:
            return "REPEAT"
    has_return = any("return " in l for l in lines)
    has_def = any(l.strip().startswith(("def ", "class ")) for l in lines[1:])
    if n_lines >= 3 and (has_return or has_def):
        return "GOOD"
    if n_lines >= 2:
        return "OK"
    return "WEAK"


def compute_scores(results: dict) -> dict:
    """Compute quality scores for completed benchmarks."""
    scores: dict = {"coding": {}, "tool_use": {}, "humaneval": {}, "summary": {}}

    coding = results.get("coding", [])
    if coding:
        labels = []
        for task in coding:
            label = score_completion(task.get("completion", ""))
            scores["coding"][task.get("id", "?")] = {
                "tokens": task.get("tokens", 0),
                "quality": label,
            }
            labels.append(label)
        good = sum(1 for l in labels if l in ("GOOD", "OK"))
        scores["summary"]["coding_quality"] = f"{good}/{len(labels)}"

    tool_use = results.get("tool_use", [])
    if tool_use:
        labels = []
        for task in tool_use:
            # New format: structured tool_calls from vLLM --tool-call-parser openai
            tool_calls = task.get("tool_calls", [])
            tool_match = task.get("tool_match")
            called_tools = task.get("called_tools", [])

            if tool_calls:
                # Structured tool call format
                if tool_match:
                    label = "TOOL_USE"
                elif called_tools:
                    label = "TOOL_USE"  # Called tools, just not expected ones
                else:
                    label = "WEAK"
            else:
                # Legacy plain-text format
                comp = task.get("completion", "") or ""
                if not isinstance(comp, str):
                    comp = str(comp)
                has_tool = bool(re.search(
                    r'(read_file|write_file|run_command|search_files)\s*\(', comp))
                has_plan = bool(re.search(r'(Step \d|1\.|Let me)', comp))
                if has_tool:
                    label = "TOOL_USE"
                elif has_plan:
                    label = "PLANNED"
                elif len(comp.strip()) > 50:
                    label = "VERBOSE"
                else:
                    label = "WEAK"

            scores["tool_use"][task.get("id", "?")] = {
                "tokens": task.get("tokens", 0),
                "quality": label,
                "called_tools": called_tools,
            }
            labels.append(label)
        tool_ok = sum(1 for l in labels if l in ("TOOL_USE", "PLANNED"))
        scores["summary"]["tool_quality"] = f"{tool_ok}/{len(labels)}"

    he = results.get("humaneval", {"total": 0, "correct": -1})
    scores["humaneval"]["total"] = he.get("total", 0)
    scores["humaneval"]["correct"] = he.get("correct", -1)

    return scores


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODING_TASKS = 10
TOOL_USE_TASKS = 4
HUMANEVAL_TASKS = 164
TOTAL_TASKS = CODING_TASKS + TOOL_USE_TASKS + HUMANEVAL_TASKS


# ---------------------------------------------------------------------------
# Background poller
# ---------------------------------------------------------------------------

class BenchmarkPoller(threading.Thread):
    """Polls DGX Spark via SSH every N seconds, stores data thread-safely."""

    def __init__(self, host: str, bench_dir: str, interval: float = 5.0):
        super().__init__(daemon=True)
        self.host = host
        self.bench_dir = bench_dir
        self.interval = interval
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._data: dict = {
            "elapsed_s": 0,
            "ssh_ok": False,
            "last_update": None,
            "containers": [],
            "models": {
                "20b": {"info": {}, "results": {}, "scores": {}, "progress": [],
                        "he_progress": ""},
                "120b": {"info": {}, "results": {}, "scores": {}, "progress": [],
                         "he_progress": ""},
            },
        }

    def get_data(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def run(self):
        while True:
            self._poll()
            time.sleep(self.interval)

    def _poll(self):
        elapsed = time.time() - self._start_time

        ping = ssh_exec(self.host, "echo ok")
        ssh_ok = ping == "ok"

        if not ssh_ok:
            with self._lock:
                self._data["elapsed_s"] = int(elapsed)
                self._data["ssh_ok"] = False
                self._data["last_update"] = datetime.now(
                    timezone.utc).isoformat(timespec="seconds")
            return

        containers = get_container_status(self.host)

        models_data = {}
        for model in ("20b", "120b"):
            info = get_model_info(self.host, self.bench_dir, model)
            results = get_bench_results(self.host, self.bench_dir, model)
            scores = compute_scores(results) if results else {}

            progress = []
            he_progress = ""
            for c in containers:
                # Match "-20b" or "-120b" to avoid "20b" matching "120b"
                if f"-{model}" in c["name"]:
                    progress, he_progress = get_docker_progress(
                        self.host, c["name"])

            models_data[model] = {
                "info": info,
                "results": results,
                "scores": scores,
                "progress": progress,
                "he_progress": he_progress,
            }

        with self._lock:
            self._data = {
                "elapsed_s": int(elapsed),
                "ssh_ok": True,
                "last_update": datetime.now(
                    timezone.utc).isoformat(timespec="seconds"),
                "containers": containers,
                "models": models_data,
            }


# ---------------------------------------------------------------------------
# HTML dashboard (embedded)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPT-OSS Benchmark Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5}
a{color:#58a6ff;text-decoration:none}

/* Header */
.header{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:#161b22;border-bottom:1px solid #30363d}
.header h1{font-size:18px;color:#58a6ff;font-weight:600}
.header-right{display:flex;align-items:center;gap:16px;font-size:13px;color:#8b949e}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}
.dot-ok{background:#3fb950}
.dot-err{background:#f85149}
.elapsed{color:#d2a8ff}
.refresh-indicator{color:#8b949e;font-size:12px}

/* Main grid */
.main{padding:16px 20px;max-width:1400px;margin:0 auto}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}

/* Cards */
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;overflow:hidden}
.card-title{font-size:16px;font-weight:600;color:#f0f6fc;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.model-meta{font-size:12px;color:#8b949e;margin-bottom:12px}

/* VRAM bar */
.vram-bar{background:#21262d;border-radius:4px;height:8px;margin-bottom:12px;overflow:hidden}
.vram-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#3fb950,#58a6ff);transition:width .5s}

/* Progress bar */
.progress-section{margin-bottom:14px}
.progress-header{display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px}
.progress-bar{background:#21262d;border-radius:4px;height:6px;overflow:hidden}
.progress-fill{height:100%;border-radius:4px;transition:width .5s}
.progress-fill.complete{background:#3fb950}
.progress-fill.partial{background:#d29922}

/* Section headers — collapsible */
.section{margin-bottom:4px;border:1px solid #21262d;border-radius:6px;overflow:hidden}
.section-header{font-size:13px;font-weight:600;color:#f0f6fc;padding:8px 10px;display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none;background:#1c2129;transition:background .15s}
.section-header:hover{background:#22272e}
.section-chevron{font-size:10px;color:#484f58;transition:transform .2s;flex-shrink:0}
.section.open .section-chevron{transform:rotate(90deg)}
.section-count{font-size:12px;color:#8b949e;font-weight:400}
.section-quality{font-size:12px;color:#3fb950;font-weight:400;margin-left:auto}
.section-body{max-height:0;overflow:hidden;transition:max-height .25s ease-out}
.section.open .section-body{max-height:600px;overflow-y:auto}

/* Scrollbar styling for section bodies */
.section-body::-webkit-scrollbar{width:6px}
.section-body::-webkit-scrollbar-track{background:#0d1117}
.section-body::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
.section-body::-webkit-scrollbar-thumb:hover{background:#484f58}

/* Task tables */
.task-table{width:100%;border-collapse:collapse;font-size:13px}
.task-table th{text-align:left;color:#8b949e;font-weight:500;padding:4px 8px;border-bottom:1px solid #21262d}
.task-table td{padding:4px 8px;border-bottom:1px solid #21262d}
.task-table tr:last-child td{border-bottom:none}
.task-table tr.clickable{cursor:pointer;transition:background .1s}
.task-table tr.clickable:hover{background:#1c2129}
.task-name{color:#c9d1d9}
.task-tokens{color:#8b949e;text-align:right}
.task-expand-hint{color:#484f58;font-size:11px;padding-left:4px}

/* Quality badges */
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;text-transform:uppercase}
.badge-good{background:#238636;color:#3fb950}
.badge-ok{background:#1b4b5a;color:#56d4dd}
.badge-weak{background:#4a3520;color:#d29922}
.badge-bad{background:#4a1e1e;color:#f85149}
.badge-live{background:#1a3a5c;color:#58a6ff}
.badge-pending{background:#21262d;color:#484f58}

/* HumanEval */
.he-status{font-size:13px;color:#8b949e;padding:4px 8px}
.he-pass{color:#3fb950;font-weight:600}

/* Comparison table */
.card-full{grid-column:1/-1}
.cmp-scroll{max-height:400px;overflow-y:auto}
.cmp-scroll::-webkit-scrollbar{width:6px}
.cmp-scroll::-webkit-scrollbar-track{background:#0d1117}
.cmp-scroll::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
.cmp-table{width:100%;border-collapse:collapse;font-size:13px}
.cmp-table th{text-align:left;color:#8b949e;font-weight:500;padding:6px 10px;border-bottom:1px solid #30363d;position:sticky;top:0;background:#161b22;z-index:1}
.cmp-table td{padding:6px 10px;border-bottom:1px solid #21262d}
.cmp-table tr:last-child td{border-bottom:none}
.cmp-table tr.clickable{cursor:pointer;transition:background .1s}
.cmp-table tr.clickable:hover{background:#1c2129}
.cmp-table .num{text-align:right;color:#8b949e}
.cmp-status{font-size:11px;font-weight:600;text-transform:uppercase}
.cmp-done{color:#3fb950}
.cmp-partial{color:#d29922}
.cmp-live{color:#58a6ff}

/* Pending message */
.pending{color:#484f58;font-style:italic;padding:4px 8px}

/* Drill-down modal */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;justify-content:center;align-items:center;padding:24px}
.modal-overlay.open{display:flex}
.modal{background:#161b22;border:1px solid #30363d;border-radius:10px;width:100%;max-width:800px;max-height:85vh;display:flex;flex-direction:column;box-shadow:0 16px 48px rgba(0,0,0,.4)}
.modal-head{display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid #21262d}
.modal-head h2{font-size:15px;color:#f0f6fc;font-weight:600}
.modal-head .meta{display:flex;gap:10px;align-items:center}
.modal-close{background:none;border:none;color:#8b949e;font-size:20px;cursor:pointer;padding:4px 8px;border-radius:4px}
.modal-close:hover{background:#21262d;color:#f0f6fc}
.modal-body{padding:16px 18px;overflow-y:auto;flex:1}
.modal-body pre{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:12px;font-size:12px;line-height:1.6;color:#c9d1d9;white-space:pre-wrap;word-break:break-word;max-height:none}
.modal-tabs{display:flex;gap:2px;padding:8px 18px 0;border-bottom:1px solid #21262d}
.modal-tab{padding:6px 14px;font-size:13px;color:#8b949e;cursor:pointer;border:1px solid transparent;border-bottom:none;border-radius:6px 6px 0 0;background:none}
.modal-tab:hover{color:#c9d1d9}
.modal-tab.active{color:#f0f6fc;background:#0d1117;border-color:#21262d}

/* Responsive */
@media(max-width:800px){.grid{grid-template-columns:1fr}.modal{max-width:95vw}}
</style>
</head>
<body>

<div class="header">
  <h1>GPT-OSS Benchmark Dashboard</h1>
  <div class="header-right">
    <span id="conn"><span class="dot dot-err"></span> Connecting...</span>
    <span class="elapsed" id="elapsed">0h 0m 0s</span>
    <span class="refresh-indicator" id="refresh">--</span>
  </div>
</div>

<div class="main">
  <div class="grid">
    <div class="card" id="card-20b"></div>
    <div class="card" id="card-120b"></div>
    <div class="card card-full" id="card-cmp"></div>
  </div>
</div>

<!-- Drill-down modal -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <div class="modal-head">
      <h2 id="modal-title">Task</h2>
      <div class="meta">
        <span id="modal-badge"></span>
        <span id="modal-tokens" style="font-size:12px;color:#8b949e"></span>
        <button class="modal-close" id="modal-close">&times;</button>
      </div>
    </div>
    <div class="modal-tabs" id="modal-tabs"></div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>

<script>
const CODING_TASKS = 10;
const TOOL_USE_TASKS = 4;
const HUMANEVAL_TASKS = 20;
const TOTAL = CODING_TASKS + TOOL_USE_TASKS + HUMANEVAL_TASKS;

// Track which sections are open so refreshes don't collapse them
const openSections = {};

function badgeClass(q) {
  if (['GOOD','TOOL_USE'].includes(q)) return 'badge-good';
  if (['OK','PLANNED'].includes(q)) return 'badge-ok';
  if (['VERBOSE','WEAK'].includes(q)) return 'badge-weak';
  if (['REPEAT','EMPTY'].includes(q)) return 'badge-bad';
  if (q === 'LIVE') return 'badge-live';
  return 'badge-pending';
}

function badge(q) {
  return `<span class="badge ${badgeClass(q)}">${q}</span>`;
}

function fmtElapsed(s) {
  const h = Math.floor(s/3600);
  const m = Math.floor((s%3600)/60);
  const sec = s%60;
  return `${h}h ${m}m ${sec}s`;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ---------- Modal ----------

let lastData = null;

function openModal(model, bench, taskId) {
  if (!lastData) return;
  const md = lastData.models[model];
  if (!md) return;

  const tasks = (md.results && md.results[bench]) || [];
  const task = tasks.find(t => t.id === taskId);
  if (!task) return;

  const scores = md.scores || {};
  const benchScores = scores[bench] || {};
  const q = (benchScores[taskId] || {}).quality || '?';
  const tok = task.tokens || 0;
  const completion = task.completion || '';
  const prompt = task.prompt || '';

  document.getElementById('modal-title').textContent =
    `${taskId}  (${model.toUpperCase()} / ${bench})`;
  document.getElementById('modal-badge').innerHTML = badge(q);
  document.getElementById('modal-tokens').textContent = `${tok} tokens`;

  // Build tabs: Completion + Prompt (if available)
  const tabs = [{id:'completion', label:'Completion', content: completion}];
  if (prompt) tabs.push({id:'prompt', label:'Prompt', content: prompt});

  const tabsEl = document.getElementById('modal-tabs');
  const bodyEl = document.getElementById('modal-body');

  tabsEl.innerHTML = tabs.map((t, i) =>
    `<div class="modal-tab ${i===0?'active':''}" data-tab="${t.id}">${t.label}</div>`
  ).join('');

  function showTab(tabId) {
    const tab = tabs.find(t => t.id === tabId);
    bodyEl.innerHTML = `<pre>${escHtml(tab ? tab.content : '')}</pre>`;
    tabsEl.querySelectorAll('.modal-tab').forEach(el => {
      el.classList.toggle('active', el.dataset.tab === tabId);
    });
  }

  tabsEl.querySelectorAll('.modal-tab').forEach(el => {
    el.onclick = () => showTab(el.dataset.tab);
  });

  showTab('completion');
  document.getElementById('modal').classList.add('open');
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
}

document.getElementById('modal-close').onclick = closeModal;
document.getElementById('modal').onclick = function(e) {
  if (e.target === this) closeModal();
};
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ---------- Collapsible sections ----------

function toggleSection(key) {
  openSections[key] = !openSections[key];
  const el = document.getElementById('sec-' + key);
  if (el) el.classList.toggle('open', openSections[key]);
}

function sectionHtml(key, titleLabel, count, total, qualitySummary, bodyHtml, hasContent) {
  // Auto-open sections that have content, unless user explicitly closed them
  if (openSections[key] === undefined && hasContent) openSections[key] = true;
  const isOpen = openSections[key] ? 'open' : '';

  return `<div class="section ${isOpen}" id="sec-${key}">
    <div class="section-header" onclick="toggleSection('${key}')">
      <span class="section-chevron">&#9654;</span>
      ${titleLabel}
      <span class="section-count">${count}/${total}</span>
      ${qualitySummary ? `<span class="section-quality">quality: ${qualitySummary}</span>` : ''}
    </div>
    <div class="section-body">${bodyHtml}</div>
  </div>`;
}

// ---------- Render model card ----------

function renderModel(el, label, md) {
  const model = label.toLowerCase();
  const info = md.info || {};
  const results = md.results || {};
  const scores = md.scores || {};
  const progress = md.progress || [];
  const heProg = md.he_progress || '';

  const modelName = info.model || `gpt-oss-${model}`;
  const alloc = info.allocated_gb || '?';
  const totalMem = info.gpu_mem_gb || '?';
  const dtype = info.dtype || '?';
  const vramPct = (typeof alloc === 'number' && typeof totalMem === 'number')
    ? (alloc/totalMem*100).toFixed(0) : 0;

  const nCoding = (results.coding||[]).length;
  const nTool = (results.tool_use||[]).length;
  const nHe = (results.humaneval||{}).total || 0;
  const nDone = nCoding + nTool + nHe;
  const pct = (nDone/TOTAL*100).toFixed(0);

  let html = `
    <div class="card-title">${label}
      <span style="font-size:12px;color:#8b949e;font-weight:400">${modelName}</span>
    </div>
    <div class="model-meta">VRAM: ${alloc}/${totalMem} GB &middot; ${dtype}</div>
    <div class="vram-bar"><div class="vram-bar-fill" style="width:${vramPct}%"></div></div>
    <div class="progress-section">
      <div class="progress-header">
        <span>Overall Progress</span>
        <span>${nDone}/${TOTAL} tasks (${pct}%)</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill ${nDone>=TOTAL?'complete':'partial'}" style="width:${pct}%"></div>
      </div>
    </div>`;

  // --- Coding section ---
  const codingScores = scores.coding || {};
  const summCoding = (scores.summary||{}).coding_quality || '';
  let codingBody = '';

  if ((results.coding||[]).length) {
    codingBody = `<table class="task-table"><thead><tr><th>Task</th><th style="text-align:right">Tokens</th><th>Quality</th></tr></thead><tbody>`;
    for (const t of results.coding) {
      const tid = t.id || '?';
      const tok = t.tokens || 0;
      const q = (codingScores[tid]||{}).quality || '?';
      codingBody += `<tr class="clickable" onclick="openModal('${model}','coding','${tid}')">
        <td class="task-name">${tid}<span class="task-expand-hint"> &#8599;</span></td>
        <td class="task-tokens">${tok}</td><td>${badge(q)}</td></tr>`;
    }
    codingBody += `</tbody></table>`;
  } else if (progress.length) {
    codingBody = `<table class="task-table"><thead><tr><th>Task</th><th style="text-align:right">Tokens</th><th>Status</th></tr></thead><tbody>`;
    for (const t of progress) {
      codingBody += `<tr><td class="task-name">${t.name}</td><td class="task-tokens">${t.tokens}</td><td>${badge('LIVE')}</td></tr>`;
    }
    codingBody += `</tbody></table>`;
    if (nCoding < CODING_TASKS) {
      codingBody += `<div class="pending">Running... ${progress.length}/${CODING_TASKS}</div>`;
    }
  } else {
    codingBody = `<div class="pending">Pending</div>`;
  }

  html += sectionHtml(`${model}-coding`, 'Coding', nCoding, CODING_TASKS,
    summCoding, codingBody, nCoding > 0 || progress.length > 0);

  // --- Tool-Use section ---
  const toolScores = scores.tool_use || {};
  const summTool = (scores.summary||{}).tool_quality || '';
  let toolBody = '';

  if ((results.tool_use||[]).length) {
    toolBody = `<table class="task-table"><thead><tr><th>Task</th><th style="text-align:right">Tokens</th><th>Quality</th></tr></thead><tbody>`;
    for (const t of results.tool_use) {
      const tid = t.id || '?';
      const tok = t.tokens || 0;
      const q = (toolScores[tid]||{}).quality || '?';
      toolBody += `<tr class="clickable" onclick="openModal('${model}','tool_use','${tid}')">
        <td class="task-name">${tid}<span class="task-expand-hint"> &#8599;</span></td>
        <td class="task-tokens">${tok}</td><td>${badge(q)}</td></tr>`;
    }
    toolBody += `</tbody></table>`;
  } else {
    toolBody = `<div class="pending">Pending</div>`;
  }

  html += sectionHtml(`${model}-tool`, 'Tool-Use', nTool, TOOL_USE_TASKS,
    summTool, toolBody, nTool > 0);

  // --- HumanEval section ---
  const he = results.humaneval || {total:0, correct:-1};
  let heBody = '';

  if (nHe > 0 && he.correct >= 0) {
    const hePct = (he.correct/nHe*100).toFixed(0);
    heBody = `<div class="he-status"><span class="he-pass">${hePct}% (${he.correct}/${nHe} pass)</span></div>`;
  } else if (nHe > 0) {
    heBody = `<div class="he-status">${nHe} completions (eval pending)</div>`;
  } else if (heProg) {
    heBody = `<div class="pending">Running... ${heProg}</div>`;
  } else {
    heBody = `<div class="pending">Pending</div>`;
  }

  html += sectionHtml(`${model}-he`, 'HumanEval', nHe, HUMANEVAL_TASKS,
    '', heBody, nHe > 0 || heProg);

  el.innerHTML = html;
}

// ---------- Render comparison ----------

function renderComparison(el, data) {
  const m20 = data.models['20b'] || {};
  const m120 = data.models['120b'] || {};
  const r20 = m20.results || {};
  const r120 = m120.results || {};
  const s20 = m20.scores || {};
  const s120 = m120.scores || {};
  const prog120 = m120.progress || [];

  const order = [];
  const td = {};

  for (const t of [...(r20.coding||[]), ...(r20.tool_use||[])]) {
    const id = t.id||'?';
    if (!td[id]) { order.push(id); td[id] = {}; }
    td[id].tok20 = t.tokens||0;
    td[id].q20 = ((s20.coding||{})[id]||{}).quality || ((s20.tool_use||{})[id]||{}).quality || '';
    td[id].bench20 = (s20.coding||{})[id] ? 'coding' : 'tool_use';
  }
  for (const t of [...(r120.coding||[]), ...(r120.tool_use||[])]) {
    const id = t.id||'?';
    if (!td[id]) { order.push(id); td[id] = {}; }
    td[id].tok120 = t.tokens||0;
    td[id].q120 = ((s120.coding||{})[id]||{}).quality || ((s120.tool_use||{})[id]||{}).quality || '';
    td[id].bench120 = (s120.coding||{})[id] ? 'coding' : 'tool_use';
  }
  for (const t of prog120) {
    if (!td[t.name]) { order.push(t.name); td[t.name] = {}; }
    if (td[t.name].tok120 === undefined) {
      td[t.name].tok120 = t.tokens;
      td[t.name].q120 = 'LIVE';
    }
  }

  let html = `<div class="card-title">Comparison</div>`;
  html += `<div class="cmp-scroll"><table class="cmp-table"><thead><tr>
    <th>Task</th><th class="num">20B Tok</th><th>20B Quality</th>
    <th class="num">120B Tok</th><th>120B Quality</th><th>Status</th>
  </tr></thead><tbody>`;

  for (const id of order) {
    const d = td[id];
    const has20 = d.tok20 !== undefined;
    const has120 = d.tok120 !== undefined;
    let status, statusClass;
    if (has20 && has120) {
      if (d.q120 === 'LIVE') { status = '120B...'; statusClass = 'cmp-live'; }
      else { status = 'DONE'; statusClass = 'cmp-done'; }
    } else if (has20) { status = '20B only'; statusClass = 'cmp-partial'; }
    else if (has120) { status = d.q120==='LIVE'?'120B...':'120B only'; statusClass = 'cmp-live'; }
    else { status = '-'; statusClass = ''; }

    // Clicking a comparison row opens 20B if available, else 120B
    const clickModel = has20 ? '20b' : '120b';
    const clickBench = has20 ? (d.bench20||'coding') : (d.bench120||'coding');
    const clickable = (has20 || (has120 && d.q120 !== 'LIVE')) ? 'clickable' : '';
    const onclick = clickable ? `onclick="openModal('${clickModel}','${clickBench}','${id}')"` : '';

    html += `<tr class="${clickable}" ${onclick}>
      <td class="task-name">${id}${clickable?'<span class="task-expand-hint"> &#8599;</span>':''}</td>
      <td class="num">${has20?d.tok20:'-'}</td>
      <td>${d.q20?badge(d.q20):'-'}</td>
      <td class="num">${has120?d.tok120:'-'}</td>
      <td>${d.q120?badge(d.q120):'-'}</td>
      <td><span class="cmp-status ${statusClass}">${status}</span></td>
    </tr>`;
  }

  // HumanEval row
  const he20 = (r20.humaneval||{});
  const he120 = (r120.humaneval||{});
  const n20 = he20.total||0; const c20 = he20.correct||-1;
  const n120 = he120.total||0; const c120 = he120.correct||-1;
  const s20h = (n20>0&&c20>=0) ? `${(c20/n20*100).toFixed(0)}%` : (n20>0?`${n20}`:'-');
  const s120h = (n120>0&&c120>=0) ? `${(c120/n120*100).toFixed(0)}%` : (n120>0?`${n120}`:'-');
  let heStatus, heClass;
  if (n20>0&&n120>0) { heStatus='DONE'; heClass='cmp-done'; }
  else if (n20>0) { heStatus='20B only'; heClass='cmp-partial'; }
  else { heStatus='-'; heClass=''; }

  html += `<tr>
    <td class="task-name">HumanEval</td>
    <td class="num">${s20h}</td><td>-</td>
    <td class="num">${s120h}</td><td>-</td>
    <td><span class="cmp-status ${heClass}">${heStatus}</span></td>
  </tr>`;

  html += `</tbody></table></div>`;
  el.innerHTML = html;
}

// ---------- Update + polling ----------

function update(data) {
  lastData = data;

  const connEl = document.getElementById('conn');
  if (data.ssh_ok) {
    connEl.innerHTML = '<span class="dot dot-ok"></span> Connected';
  } else {
    connEl.innerHTML = '<span class="dot dot-err"></span> Disconnected';
  }
  document.getElementById('elapsed').textContent = fmtElapsed(data.elapsed_s);
  if (data.last_update) {
    const d = new Date(data.last_update);
    document.getElementById('refresh').textContent = 'Updated ' + d.toLocaleTimeString();
  }

  renderModel(document.getElementById('card-20b'), '20B', data.models['20b']||{});
  renderModel(document.getElementById('card-120b'), '120B', data.models['120b']||{});
  renderComparison(document.getElementById('card-cmp'), data);
}

async function poll() {
  try {
    const resp = await fetch('/api/data');
    if (resp.ok) {
      const data = await resp.json();
      update(data);
    }
  } catch(e) {
    document.getElementById('conn').innerHTML =
      '<span class="dot dot-err"></span> Server error';
  }
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML and JSON API."""

    poller: BenchmarkPoller  # set via partial/class attr

    def do_GET(self):
        if self.path == "/api/data":
            data = self.poller.get_data()
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path in ("/", "/index.html"):
            payload = DASHBOARD_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        """Suppress default stderr logging for cleaner output."""
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPT-OSS Benchmark Dashboard — monitor DGX Spark benchmarks",
    )
    parser.add_argument(
        "--host", default="dgx-spark",
        help="SSH host alias or address (default: dgx-spark)",
    )
    parser.add_argument(
        "--benchmarks-dir", default="/home/rmarnold/benchmarks-v2",
        help="Remote directory containing benchmark results",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="HTTP port (default: 8050)",
    )
    args = parser.parse_args()

    # Start background poller
    poller = BenchmarkPoller(args.host, args.benchmarks_dir)
    poller.start()

    # Create handler class with poller reference
    handler = type("Handler", (DashboardHandler,), {"poller": poller})

    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"Dashboard running at http://localhost:{args.port}")
    print(f"Polling {args.host}:{args.benchmarks_dir} every 5s")
    print("Press Ctrl-C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
