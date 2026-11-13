"""Project-wide settings.

Values come from environment variables, optionally loaded from a local
``.env`` file (which is git-ignored). Nothing secret is hard-coded here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Target model: the small local model we attack and evaluate.
DEFAULT_MODEL = os.getenv("TARGET_MODEL", "Qwen/Qwen3-0.6B")

# Reproducibility.
DEFAULT_SEED = int(os.getenv("SEED", "42"))

# Where experiment outputs are written.
RESULTS_DIR = PROJECT_ROOT / "results"

# Judge (Hugging Face Inference Providers).
HF_TOKEN = os.getenv("HF_TOKEN")
HF_JUDGE_MODEL = os.getenv("HF_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
