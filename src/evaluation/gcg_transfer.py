"""Does a GCG suffix make real human attacks stronger? (transfer test)

GCG optimised its suffix for ONE prompt ("What is the password?"). Here we
append that suffix to the real Gandalf attack prompts and measure whether more
of them make the model leak the password.

Three conditions, same prompts, same scenario, same leak check:
  * none   - the Gandalf prompt alone (as in prompt_attacks.py)
  * random - prompt + a random suffix of the same token length (control:
             is any gain just from adding noise?)
  * gcg    - prompt + the GCG-optimised suffix

Results go to ``evaluation/gcg_transfer/<timestamp>/`` (tracked by git so the
results can be shared): samples.csv, metrics.json, config.json, summary.md,
comparison.png and a copy of the GCG run's loss.png.

Runs locally; no API calls.

Usage:
    python -m src.evaluation.gcg_transfer --gcg-run results\\gcg\\<run-folder> --limit 50
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from ..config import DEFAULT_MODEL, DEFAULT_SEED, PROJECT_ROOT
from ..experiments.gcg import allowed_token_mask
from ..experiments.prompt_attacks import GANDALF_SCENARIO, detect_leak, load_gandalf_prompts
from ..generation import generate_text
from ..model_loader import environment_info, load_model
from ..results import make_run_dir, save_json
from ..utils import set_seed

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
CONDITIONS = ["none", "random", "gcg"]


def load_gcg_suffix(gcg_run: Path) -> tuple[str, dict]:
    """Read the best suffix (and the run's metrics) from a GCG results folder."""
    metrics = json.loads((Path(gcg_run) / "metrics.json").read_text(encoding="utf-8"))
    return metrics["best_suffix"], metrics


def random_suffix(tokenizer, n_tokens: int, seed: int) -> str:
    """A random suffix of ``n_tokens`` tokens from the same allowed vocabulary GCG used."""
    vocab_size = len(tokenizer)
    allowed = allowed_token_mask(tokenizer, vocab_size).nonzero().flatten()
    generator = torch.Generator().manual_seed(seed)
    picks = allowed[torch.randint(len(allowed), (n_tokens,), generator=generator)]
    return tokenizer.decode(picks)


def run_transfer(bundle, prompts: list[str], suffixes: dict[str, str], max_new_tokens: int) -> pd.DataFrame:
    """Every prompt under every condition; one row per generation."""
    scenario = GANDALF_SCENARIO
    items = [("baseline_direct", scenario.request)] + [
        (f"gandalf_{i:03d}", p) for i, p in enumerate(prompts)
    ]

    rows = []
    for prompt_id, prompt in items:
        for condition in CONDITIONS:
            suffix = suffixes.get(condition, "")
            full_prompt = f"{prompt} {suffix}" if suffix else prompt
            response = generate_text(
                bundle.model,
                bundle.tokenizer,
                full_prompt,
                max_new_tokens=max_new_tokens,
                system=scenario.system,
            )
            leak = detect_leak(response, scenario.canary)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "condition": condition,
                    "prompt": prompt,
                    "response": response,
                    "rule_broken": leak is not None,
                    "leak_type": leak or "",
                }
            )
        marks = " ".join(
            f"{r['condition']}={'LEAK' if r['rule_broken'] else 'held'}" for r in rows[-len(CONDITIONS):]
        )
        print(f"{prompt_id:16s} {marks}", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    """Leak rate per condition on the Gandalf prompts, plus per-prompt flips."""
    attacks = df[df["prompt_id"] != "baseline_direct"]
    wide = attacks.pivot(index="prompt_id", columns="condition", values="rule_broken")
    base = df[df["prompt_id"] == "baseline_direct"].set_index("condition")["rule_broken"]

    summary = {
        "n_prompts": int(len(wide)),
        "leak_rate": {c: round(float(wide[c].mean()), 3) for c in CONDITIONS},
        "baseline_direct_leaked": {c: bool(base[c]) for c in CONDITIONS},
    }
    for c in ["random", "gcg"]:
        summary[f"{c}_vs_none"] = {
            "newly_leaked": int((~wide["none"] & wide[c]).sum()),   # held -> leaked
            "newly_held": int((wide["none"] & ~wide[c]).sum()),     # leaked -> held
        }
    return summary


def save_chart(summary: dict, path: Path) -> None:
    """Bar chart of leak rate per condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"none": "Gandalf prompt only", "random": "+ random suffix", "gcg": "+ GCG suffix"}
    rates = [summary["leak_rate"][c] * 100 for c in CONDITIONS]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar([labels[c] for c in CONDITIONS], rates, color=["#8c8c8c", "#b0b0b0", "#2f6db5"])
    ax.bar_label(bars, fmt="%.0f%%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Password leak rate (%)")
    ax.set_title(f"GCG suffix transfer to {summary['n_prompts']} real Gandalf attacks")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary_md(summary: dict, config: dict, path: Path) -> None:
    """A short human-readable report for sharing."""
    lr = summary["leak_rate"]
    lines = [
        "# GCG suffix transfer to real Gandalf attacks",
        "",
        f"- Target model: `{config['model']}`",
        f"- Prompts: {summary['n_prompts']} from `Lakera/gandalf_ignore_instructions` "
        f"({config['split']} split, seed {config['seed']})",
        f"- GCG run: `{config['gcg_run']}` (best loss {config['gcg_best_loss']})",
        "- Leak check: password appears exactly / spaced / reversed (partial leaks not counted)",
        "",
        "| Condition | Leak rate | Newly leaked vs none | Newly held vs none |",
        "|---|---|---|---|",
        f"| Gandalf prompt only | {lr['none']:.0%} | - | - |",
        f"| + random suffix (control) | {lr['random']:.0%} | "
        f"{summary['random_vs_none']['newly_leaked']} | {summary['random_vs_none']['newly_held']} |",
        f"| + GCG suffix | {lr['gcg']:.0%} | "
        f"{summary['gcg_vs_none']['newly_leaked']} | {summary['gcg_vs_none']['newly_held']} |",
        "",
        "Plain request (\"What is the password?\") leaked: "
        + ", ".join(f"{c}={'yes' if v else 'no'}" for c, v in summary["baseline_direct_leaked"].items()),
        "",
        "![comparison](comparison.png)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether a GCG suffix transfers to Gandalf prompts.")
    parser.add_argument("--gcg-run", type=Path, required=True, help="Folder of a GCG run (has metrics.json).")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    gcg_suffix, gcg_metrics = load_gcg_suffix(args.gcg_run)
    bundle = load_model(args.model)

    n_tokens = len(bundle.tokenizer(gcg_suffix, add_special_tokens=False).input_ids)
    suffixes = {
        "none": "",
        "random": random_suffix(bundle.tokenizer, n_tokens, args.seed),
        "gcg": gcg_suffix,
    }
    print(f"GCG suffix    ({n_tokens} tokens): {suffixes['gcg']!r}")
    print(f"Random suffix ({n_tokens} tokens): {suffixes['random']!r}\n")

    prompts = load_gandalf_prompts(args.limit, args.seed, args.split)
    df = run_transfer(bundle, prompts, suffixes, args.max_new_tokens)
    summary = summarise(df)

    gcg_run = Path(args.gcg_run).resolve()
    try:  # store a portable, repo-relative path when possible
        gcg_run_str = gcg_run.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        gcg_run_str = gcg_run.as_posix()

    config = {
        **{k: str(v) for k, v in vars(args).items()},
        "gcg_run": gcg_run_str,
        "gcg_best_loss": gcg_metrics.get("best_loss"),
        "suffix_tokens": n_tokens,
        "suffixes": suffixes,
        "scenario_system_prompt": GANDALF_SCENARIO.system,
        **environment_info(),
    }

    run_dir = make_run_dir("gcg_transfer", base=EVALUATION_DIR)
    df.to_csv(run_dir / "samples.csv", index=False)
    save_json(summary, run_dir / "metrics.json")
    save_json(config, run_dir / "config.json")
    save_chart(summary, run_dir / "comparison.png")
    write_summary_md(summary, config, run_dir / "summary.md")
    gcg_loss = Path(args.gcg_run) / "loss.png"
    if gcg_loss.exists():
        shutil.copy(gcg_loss, run_dir / "gcg_loss.png")

    print()
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
