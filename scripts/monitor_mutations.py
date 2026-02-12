#!/usr/bin/env python3
"""Split-screen TUI to monitor cargo-mutants data generation.

Usage:
    python scripts/monitor_mutations.py
    python scripts/monitor_mutations.py --output-dir /Volumes/OWC\ Express\ 1M2/rust-mutations/output
"""
from __future__ import annotations

import argparse
import curses
import json
import os
import shutil
import subprocess
import time


def get_repo_status(clone_dir: str) -> list[tuple[str, str]]:
    """Get list of cloned repos with sizes."""
    repos = []
    if not os.path.isdir(clone_dir):
        return repos
    for name in sorted(os.listdir(clone_dir)):
        path = os.path.join(clone_dir, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        # Get dir size (rough, just count files)
        try:
            n_files = sum(len(fs) for _, _, fs in os.walk(path))
            repos.append((name, f"{n_files} files"))
        except OSError:
            repos.append((name, "?"))
    return repos


def get_training_stats(output_dir: str) -> dict:
    """Get training data statistics."""
    stats = {"total": 0, "caught": 0, "unviable": 0, "size_mb": 0.0}
    jsonl_path = os.path.join(output_dir, "mutations.jsonl")
    if not os.path.exists(jsonl_path):
        return stats

    try:
        stats["size_mb"] = os.path.getsize(jsonl_path) / (1024 * 1024)
        with open(jsonl_path) as f:
            for line in f:
                stats["total"] += 1
                if '"Test failure:' in line:
                    stats["caught"] += 1
                elif '"Compiler error:' in line:
                    stats["unviable"] += 1
    except (IOError, json.JSONDecodeError):
        pass

    return stats


def get_system_stats() -> dict:
    """Get CPU load and memory stats on macOS."""
    stats = {"load": "", "mem_used": "", "mem_total": "", "mem_pct": 0.0}

    # Load average
    try:
        result = subprocess.run(["sysctl", "-n", "vm.loadavg"],
                                capture_output=True, text=True, timeout=5)
        stats["load"] = result.stdout.strip().strip("{ }")
    except Exception:
        pass

    # Memory via vm_stat
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        pages = {}
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip().lower()
                val = parts[1].strip().rstrip(".")
                try:
                    pages[key] = int(val)
                except ValueError:
                    pass

        page_size = 16384  # Apple Silicon default
        try:
            ps_result = subprocess.run(["sysctl", "-n", "hw.pagesize"],
                                       capture_output=True, text=True, timeout=5)
            page_size = int(ps_result.stdout.strip())
        except Exception:
            pass

        free = pages.get("pages free", 0) + pages.get("pages speculative", 0)
        active = pages.get("pages active", 0)
        inactive = pages.get("pages inactive", 0)
        wired = pages.get("pages wired down", 0)
        compressed = pages.get("pages occupied by compressor", 0)

        used_bytes = (active + wired + compressed) * page_size
        total_bytes = (free + active + inactive + wired + compressed) * page_size

        stats["mem_used"] = f"{used_bytes / (1024**3):.1f}"
        stats["mem_total"] = f"{total_bytes / (1024**3):.1f}"
        if total_bytes > 0:
            stats["mem_pct"] = used_bytes / total_bytes * 100
    except Exception:
        pass

    return stats


def get_sccache_stats() -> dict:
    """Get sccache hit/miss stats."""
    stats = {"hits": 0, "misses": 0, "errors": 0, "requests": 0}
    try:
        result = subprocess.run(["sccache", "--show-stats"],
                                capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n"):
            lower = line.lower().strip()
            if "compile requests" in lower:
                try:
                    stats["requests"] = int(lower.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif "cache hits" in lower and "rate" not in lower:
                try:
                    stats["hits"] = int(lower.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif "cache misses" in lower:
                try:
                    stats["misses"] = int(lower.split()[-1])
                except (ValueError, IndexError):
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return stats


def get_active_processes() -> list[str]:
    """Get cargo/rustc processes currently running."""
    procs = []
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,pcpu,rss,comm"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 4:
                comm = parts[3]
                if any(k in comm for k in ["cargo", "rustc", "cargo-mutants"]):
                    pid = parts[0]
                    cpu = parts[1]
                    rss_kb = int(parts[2])
                    rss_mb = rss_kb / 1024
                    name = os.path.basename(comm)
                    procs.append(f"  {pid:>6}  {cpu:>5}%  {rss_mb:>6.0f}MB  {name}")
    except Exception:
        pass
    return procs[:15]  # Cap at 15 to fit on screen


def draw_box(win, y: int, x: int, h: int, w: int, title: str = ""):
    """Draw a box with optional title."""
    try:
        # Corners
        win.addch(y, x, curses.ACS_ULCORNER)
        win.addch(y, x + w - 1, curses.ACS_URCORNER)
        win.addch(y + h - 1, x, curses.ACS_LLCORNER)
        win.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
        # Horizontal lines
        for i in range(1, w - 1):
            win.addch(y, x + i, curses.ACS_HLINE)
            win.addch(y + h - 1, x + i, curses.ACS_HLINE)
        # Vertical lines
        for i in range(1, h - 1):
            win.addch(y + i, x, curses.ACS_VLINE)
            win.addch(y + i, x + w - 1, curses.ACS_VLINE)
        # Title
        if title:
            label = f" {title} "
            win.addstr(y, x + 2, label, curses.A_BOLD)
    except curses.error:
        pass


def safe_addstr(win, y: int, x: int, text: str, attr=0, max_w: int = 0):
    """addstr that won't crash on boundary."""
    try:
        if max_w > 0:
            text = text[:max_w]
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def make_bar(pct: float, width: int) -> str:
    """Make a progress bar string."""
    filled = int(pct / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main(stdscr, output_dir: str, clone_dir: str):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(3000)  # Refresh every 3 seconds

    # Colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_RED, -1)
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)

    GREEN = curses.color_pair(1)
    YELLOW = curses.color_pair(2)
    CYAN = curses.color_pair(3)
    RED = curses.color_pair(4)
    MAGENTA = curses.color_pair(5)

    start_time = time.time()
    prev_total = 0

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        if max_y < 20 or max_x < 60:
            safe_addstr(stdscr, 0, 0, "Terminal too small. Resize to at least 60x20.")
            stdscr.refresh()
            key = stdscr.getch()
            if key == ord('q'):
                break
            continue

        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s"

        # ---- Header ----
        header = " Mutation Generation Monitor "
        safe_addstr(stdscr, 0, (max_x - len(header)) // 2, header, curses.A_BOLD | CYAN)
        safe_addstr(stdscr, 1, 2, f"Elapsed: {elapsed_str}", YELLOW)
        safe_addstr(stdscr, 1, max_x - 22, "Press 'q' to quit", curses.A_DIM)

        # Layout: left panel (repos + processes), right panel (stats + sccache)
        left_w = max_x // 2
        right_w = max_x - left_w
        panel_top = 3

        # ---- Left Panel: Repos ----
        repos = get_repo_status(clone_dir)
        repo_h = min(len(repos) + 3, (max_y - panel_top) // 2)
        repo_h = max(repo_h, 6)

        draw_box(stdscr, panel_top, 0, repo_h, left_w, f"Repos Cloned ({len(repos)})")
        for i, (name, info) in enumerate(repos):
            if i >= repo_h - 3:
                safe_addstr(stdscr, panel_top + i + 2, 2,
                            f"  ... +{len(repos) - i} more", curses.A_DIM)
                break
            safe_addstr(stdscr, panel_top + 1 + i, 2, f"  {name}", GREEN,
                        max_w=left_w - 4)

        # ---- Left Panel: Active Processes ----
        proc_top = panel_top + repo_h
        proc_h = max_y - proc_top - 1
        if proc_h > 4:
            procs = get_active_processes()
            draw_box(stdscr, proc_top, 0, proc_h, left_w,
                      f"Cargo Processes ({len(procs)})")
            safe_addstr(stdscr, proc_top + 1, 2,
                        f"   PID    CPU      RSS  Command", curses.A_DIM,
                        max_w=left_w - 4)
            for i, proc_line in enumerate(procs):
                if i >= proc_h - 3:
                    break
                safe_addstr(stdscr, proc_top + 2 + i, 2, proc_line,
                            max_w=left_w - 4)

        # ---- Right Panel: Training Data ----
        stats = get_training_stats(output_dir)
        data_h = 10
        draw_box(stdscr, panel_top, left_w, data_h, right_w, "Training Data")

        row = panel_top + 1
        safe_addstr(stdscr, row, left_w + 2, f"Total examples: ", curses.A_DIM)
        safe_addstr(stdscr, row, left_w + 18, f"{stats['total']:,}",
                    curses.A_BOLD | GREEN)

        row += 1
        safe_addstr(stdscr, row, left_w + 2,
                    f"  Caught (test):    {stats['caught']:,}", YELLOW)
        row += 1
        safe_addstr(stdscr, row, left_w + 2,
                    f"  Unviable (comp):  {stats['unviable']:,}", MAGENTA)
        row += 1
        safe_addstr(stdscr, row, left_w + 2,
                    f"  File size:        {stats['size_mb']:.1f} MB", curses.A_DIM)

        # Rate
        row += 1
        rate = stats['total'] / (elapsed / 60) if elapsed > 60 else 0
        safe_addstr(stdscr, row, left_w + 2,
                    f"  Rate:             {rate:.1f} examples/min", curses.A_DIM)

        # Delta since last refresh
        row += 1
        delta = stats['total'] - prev_total
        if delta > 0:
            safe_addstr(stdscr, row, left_w + 2, f"  +{delta} since last refresh", GREEN)
        else:
            safe_addstr(stdscr, row, left_w + 2, f"  (waiting for next repo...)",
                        curses.A_DIM)
        prev_total = stats['total']

        # JSONL path
        row += 1
        jsonl_display = os.path.join(output_dir, "mutations.jsonl")
        if len(jsonl_display) > right_w - 6:
            jsonl_display = "..." + jsonl_display[-(right_w - 9):]
        safe_addstr(stdscr, row, left_w + 2, jsonl_display, curses.A_DIM,
                    max_w=right_w - 4)

        # ---- Right Panel: System ----
        sys_top = panel_top + data_h
        sys_h = 8
        if sys_top + sys_h < max_y:
            sys_stats = get_system_stats()
            draw_box(stdscr, sys_top, left_w, sys_h, right_w, "System")

            row = sys_top + 1
            safe_addstr(stdscr, row, left_w + 2,
                        f"Load avg:  {sys_stats['load']}", curses.A_DIM)

            row += 1
            mem_str = f"Memory:    {sys_stats['mem_used']}/{sys_stats['mem_total']} GB"
            safe_addstr(stdscr, row, left_w + 2, mem_str, curses.A_DIM)

            row += 1
            bar_w = min(right_w - 16, 30)
            bar = make_bar(sys_stats["mem_pct"], bar_w)
            color = GREEN if sys_stats["mem_pct"] < 70 else (
                YELLOW if sys_stats["mem_pct"] < 85 else RED)
            safe_addstr(stdscr, row, left_w + 2,
                        f"  {bar} {sys_stats['mem_pct']:.0f}%", color)

            # Disk usage for output dir
            row += 2
            try:
                disk = shutil.disk_usage(output_dir)
                disk_used_gb = (disk.total - disk.free) / (1024**3)
                disk_total_gb = disk.total / (1024**3)
                disk_free_gb = disk.free / (1024**3)
                safe_addstr(stdscr, row, left_w + 2,
                            f"Disk free: {disk_free_gb:.0f} GB / {disk_total_gb:.0f} GB",
                            curses.A_DIM)
            except OSError:
                pass

        # ---- Right Panel: sccache ----
        sc_top = sys_top + sys_h
        sc_h = max_y - sc_top - 1
        if sc_h > 5:
            sc_stats = get_sccache_stats()
            draw_box(stdscr, sc_top, left_w, sc_h, right_w, "sccache")

            row = sc_top + 1
            safe_addstr(stdscr, row, left_w + 2,
                        f"Requests:  {sc_stats['requests']:,}", curses.A_DIM)
            row += 1
            safe_addstr(stdscr, row, left_w + 2,
                        f"Hits:      {sc_stats['hits']:,}", GREEN)
            row += 1
            safe_addstr(stdscr, row, left_w + 2,
                        f"Misses:    {sc_stats['misses']:,}", YELLOW)

            row += 1
            if sc_stats["requests"] > 0:
                hit_pct = sc_stats["hits"] / sc_stats["requests"] * 100
                bar_w = min(right_w - 16, 30)
                bar = make_bar(hit_pct, bar_w)
                safe_addstr(stdscr, row, left_w + 2,
                            f"  {bar} {hit_pct:.0f}% hit", GREEN)

        stdscr.refresh()

        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor cargo-mutants generation")
    parser.add_argument("--output-dir", type=str,
                        default="/Volumes/OWC Express 1M2/rust-mutations/output",
                        help="Path to mutation output directory")
    parser.add_argument("--clone-dir", type=str,
                        default="/Volumes/OWC Express 1M2/rust-mutations/repos",
                        help="Path to cloned repos directory")
    args = parser.parse_args()

    curses.wrapper(main, args.output_dir, args.clone_dir)
