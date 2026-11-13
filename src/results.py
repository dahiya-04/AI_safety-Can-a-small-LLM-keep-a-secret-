"""Helpers for saving experiment outputs in a consistent layout.

Every run is written to ``results/<experiment>/<timestamp>/``.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import RESULTS_DIR


def make_run_dir(experiment: str, base: Path = RESULTS_DIR, name: str | None = None) -> Path:
    """Create and return a fresh directory for one run.

    The folder is named by timestamp, or ``<timestamp>_<name>`` if given.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base) / experiment / (f"{stamp}_{name}" if name else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: Any, path: Path) -> None:
    """Write ``data`` as pretty-printed JSON."""
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
