"""Rebuild a local GCG results folder from a remote job's log file.

A GCG run on HF Jobs writes its files inside a temporary container that is
deleted when the job ends; only the logs come back. ``gcg.py`` therefore also
prints ``GCGJSON <kind> {json}`` lines. This script reads those lines from a
saved log and writes the usual history.csv / metrics.json / config.json /
loss.png locally, plus a copy of the raw log.

Usage:
    python -m src.experiments.gcg_import path\\to\\job.log --name hfjob
"""

import argparse
import json
import shutil
from pathlib import Path

from .gcg import LOG_MARKER, save_run


def read_log(path: Path) -> str:
    """Read a log written as UTF-8 or UTF-16 (Windows PowerShell 5.1 default)."""
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def parse_log(text: str) -> tuple[list[dict], dict, dict]:
    """Extract (history, metrics, config) from ``GCGJSON`` lines."""
    history, metrics, config = [], {}, {}
    for line in text.splitlines():
        pos = line.find(LOG_MARKER + " ")
        if pos == -1:
            continue
        kind, _, payload = line[pos + len(LOG_MARKER) + 1:].partition(" ")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue  # a truncated line; skip it
        if kind == "step":
            history.append(data)
        elif kind == "metrics":
            metrics = data
        elif kind == "config":
            config = data
    return history, metrics, config


def import_log(log_path: Path, name: str | None = None) -> Path:
    """Parse a job log and write a local results folder; return its path."""
    history, metrics, config = parse_log(read_log(log_path))
    if not history:
        raise SystemExit(f"No {LOG_MARKER} step lines found in {log_path} - did the job fail early?")
    if not metrics:
        # Job stopped early (timeout/cancel): keep the steps that did finish.
        best = min(history, key=lambda h: h["loss"])
        metrics = {
            "incomplete_run": True,
            "steps_run": len(history),
            "best_loss": best["loss"],
            "best_step": best["step"],
            "best_suffix": best["suffix"],
        }
    run_dir = save_run(history, metrics, config, run_name=name)
    shutil.copy(log_path, run_dir / "job.log")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a GCG job log into results/gcg/.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--name", default="hfjob", help="Suffix for the results folder name.")
    args = parser.parse_args()

    run_dir = import_log(args.log, args.name)
    print(f"Imported into {run_dir}")
    for f in sorted(run_dir.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
