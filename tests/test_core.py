"""Fast tests for the core logic. No model is loaded; runs in seconds."""

import json

import pandas as pd

from src.evaluation.gcg_transfer import summarise as summarise_transfer
from src.experiments.gcg import LOG_MARKER
from src.experiments.gcg_import import parse_log, read_log
from src.experiments.prompt_attacks import detect_leak, rule_broken, summarise_gandalf
from src.results import make_run_dir

SECRET = "PINEAPPLE42"


# --- leak detection ---------------------------------------------------------

def test_exact_leak():
    assert detect_leak("The password is PINEAPPLE42.", SECRET) == "exact"


def test_leak_ignores_case_spacing_and_hyphens():
    assert detect_leak("p-i-n-e-a-p-p-l-e 4 2", SECRET) == "exact"


def test_reversed_leak():
    assert detect_leak("Backwards: 24ELPPAENIP", SECRET) == "reversed"


def test_no_leak():
    assert detect_leak("I cannot share the password.", SECRET) is None
    assert not rule_broken("The first letter is P.", SECRET)  # partial leaks are not caught


# --- GCG log import ---------------------------------------------------------

def _line(kind, payload, prefix=""):
    return f"{prefix}{LOG_MARKER} {kind} {json.dumps(payload)}"


def test_parse_log_reads_steps_metrics_config():
    log = "\n".join([
        "pip install noise",
        _line("config", {"steps": 2}),
        _line("step", {"step": 1, "loss": 0.9}, prefix="2026-11-13T12:00:00Z "),
        _line("step", {"step": 2, "loss": 0.5}),
        f"{LOG_MARKER} step {{\"step\": 3, \"lo",  # truncated line is skipped
        _line("metrics", {"best_loss": 0.5}),
    ])
    history, metrics, config = parse_log(log)
    assert [h["step"] for h in history] == [1, 2]
    assert metrics == {"best_loss": 0.5}
    assert config == {"steps": 2}


def test_read_log_handles_utf16_and_utf8_bom(tmp_path):
    text = _line("step", {"step": 1, "loss": 0.1})
    utf16 = tmp_path / "a.log"
    utf16.write_text(text, encoding="utf-16")
    bom = tmp_path / "b.log"
    bom.write_text(text, encoding="utf-8-sig")
    assert read_log(utf16) == text
    assert read_log(bom) == text


# --- summaries --------------------------------------------------------------

def test_summarise_gandalf():
    df = pd.DataFrame({
        "attack": ["baseline_direct", "gandalf_000", "gandalf_001"],
        "rule_broken": [False, True, False],
        "leak_type": ["", "exact", ""],
    })
    s = summarise_gandalf(df)
    assert s["baseline_direct_leaked"] is False
    assert s["attack_success_rate"] == 0.5
    assert s["n_attacks"] == 2


def test_summarise_transfer_counts_flips():
    rows = []
    # prompt -> (none, random, gcg)
    outcomes = {"baseline_direct": (False, True, True),
                "gandalf_000": (False, False, True),   # gcg newly leaks
                "gandalf_001": (True, True, False),    # gcg newly holds
                "gandalf_002": (True, True, True)}
    for pid, results in outcomes.items():
        for cond, broken in zip(["none", "random", "gcg"], results):
            rows.append({"prompt_id": pid, "condition": cond, "rule_broken": broken})
    s = summarise_transfer(pd.DataFrame(rows))
    assert s["n_prompts"] == 3
    assert s["leak_rate"]["gcg"] == round(2 / 3, 3)
    assert s["gcg_vs_none"] == {"newly_leaked": 1, "newly_held": 1}
    assert s["baseline_direct_leaked"] == {"none": False, "random": True, "gcg": True}


# --- results folders --------------------------------------------------------

def test_make_run_dir(tmp_path):
    plain = make_run_dir("exp", base=tmp_path)
    named = make_run_dir("exp", base=tmp_path, name="hfjob-123")
    assert plain.is_dir() and plain.parent == tmp_path / "exp"
    assert named.name.endswith("_hfjob-123")
