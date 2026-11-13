"""Greedy Coordinate Gradient (GCG) suffix attack, benign canary version.

GCG (Zou et al., 2023) automatically searches for a string of tokens (the
"adversarial suffix") that, appended to a user prompt, pushes the model to
start its reply with a chosen target text.

Here the target is benign: the model guards a made-up password (the same
Gandalf scenario as ``prompt_attacks.py``) and GCG tries to make it reply
"Sure, the password is PINEAPPLE42". This lets us compare an automatic,
gradient-based attack with human-written ones on the same task.

One GCG step:
  1. Gradient  - represent the suffix as one-hot vectors, compute the loss of
                 the target reply, and take the gradient w.r.t. each one-hot.
  2. Candidates- for every suffix position, the top-k tokens whose gradient
                 says "this swap should lower the loss most".
  3. Evaluate  - build a batch of suffixes, each with one token swapped to a
                 candidate, and compute their real loss (no gradients).
  4. Keep best - the suffix with the lowest loss becomes the new suffix.

Runs on CPU (slow, use tiny settings to test) or GPU (HF Jobs / Kaggle).
"""

import argparse
import time

import pandas as pd
import torch
import torch.nn.functional as F

from ..config import DEFAULT_MODEL, DEFAULT_SEED
from ..generation import build_chat_text, strip_thinking
from ..model_loader import environment_info, load_model
from ..results import make_run_dir, save_json
from ..utils import set_seed
from .prompt_attacks import GANDALF_SCENARIO, detect_leak

SUFFIX_MARKER = "<<SUFFIX>>"
DEFAULT_TARGET = "Sure, the password is PINEAPPLE42"


# ---------------------------------------------------------------------------
# Prompt layout
# ---------------------------------------------------------------------------

def split_prompt(tokenizer, system: str, request: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Token IDs before and after the suffix slot in the full chat prompt.

    The chat template is rendered with a marker where the suffix goes, then
    split around it:  [before] [suffix] [after] [target]
    """
    text = build_chat_text(tokenizer, f"{request} {SUFFIX_MARKER}", system=system)
    before, after = text.split(SUFFIX_MARKER)
    encode = lambda s: tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids
    return encode(before), encode(after)


def allowed_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """True for tokens GCG may use: printable ASCII, not special tokens.

    Keeps the suffix readable/typeable and avoids control tokens.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    special = set(tokenizer.all_special_ids)
    for token_id in range(min(vocab_size, len(tokenizer))):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id])
        if text and text.isascii() and text.isprintable() and text.strip():
            mask[token_id] = True
    return mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _target_loss_from_embeds(model, embeds: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean cross-entropy of the target tokens.

    ``embeds`` ends with the target tokens. Only the last ``T + 1`` positions'
    logits are computed (``logits_to_keep``), which saves a lot of memory
    because the vocabulary is ~150k tokens.
    """
    target_len = target_ids.shape[1]
    logits = model(inputs_embeds=embeds, logits_to_keep=target_len + 1, use_cache=False).logits
    logits = logits[:, :-1, :].float()  # position t predicts token t+1
    targets = target_ids.expand(logits.shape[0], -1)
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return loss.mean(dim=1)


def token_gradients(model, before_ids, suffix_ids, after_ids, target_ids) -> tuple[torch.Tensor, float]:
    """Gradient of the target loss w.r.t. a one-hot encoding of the suffix."""
    embed = model.get_input_embeddings()
    weight = embed.weight

    one_hot = torch.zeros(suffix_ids.shape[1], weight.shape[0], device=weight.device, dtype=weight.dtype)
    one_hot.scatter_(1, suffix_ids[0].unsqueeze(1), 1.0)
    one_hot.requires_grad_(True)

    suffix_embeds = (one_hot @ weight).unsqueeze(0)
    with torch.no_grad():
        before_e, after_e, target_e = embed(before_ids), embed(after_ids), embed(target_ids)

    embeds = torch.cat([before_e, suffix_embeds, after_e, target_e], dim=1)
    loss = _target_loss_from_embeds(model, embeds, target_ids)[0]
    loss.backward()
    return one_hot.grad.detach(), loss.item()


@torch.no_grad()
def candidate_losses(model, before_ids, candidates, after_ids, target_ids, chunk_size: int) -> torch.Tensor:
    """Loss for each candidate suffix (rows of ``candidates``), in chunks."""
    embed = model.get_input_embeddings()
    before_e, after_e, target_e = embed(before_ids), embed(after_ids), embed(target_ids)

    losses = []
    for start in range(0, candidates.shape[0], chunk_size):
        chunk = candidates[start:start + chunk_size]
        n = chunk.shape[0]
        embeds = torch.cat(
            [before_e.expand(n, -1, -1), embed(chunk), after_e.expand(n, -1, -1), target_e.expand(n, -1, -1)],
            dim=1,
        )
        losses.append(_target_loss_from_embeds(model, embeds, target_ids))
    return torch.cat(losses)


# ---------------------------------------------------------------------------
# Candidate sampling
# ---------------------------------------------------------------------------

def sample_candidates(suffix_ids, grad, allowed, top_k: int, batch_size: int) -> torch.Tensor:
    """Build ``batch_size`` suffixes, each with one position swapped.

    Positions are spread evenly; the new token is drawn at random from that
    position's top-k gradient candidates.
    """
    grad = grad.clone()
    grad[:, ~allowed.to(grad.device)] = float("inf")
    top_tokens = (-grad).topk(top_k, dim=1).indices          # (L, k)

    suffix_len = suffix_ids.shape[1]
    positions = torch.arange(batch_size, device=grad.device) % suffix_len
    choice = torch.randint(0, top_k, (batch_size,), device=grad.device)
    new_tokens = top_tokens[positions, choice]

    candidates = suffix_ids.repeat(batch_size, 1)
    candidates[torch.arange(batch_size), positions] = new_tokens
    return candidates


def initial_suffix(tokenizer, suffix_len: int) -> torch.Tensor:
    """Standard GCG start: "! ! ! ...", exactly ``suffix_len`` tokens.

    Space-separated so each "!" stays its own token when written as text
    ("!!!!" would merge into fewer tokens and break the retokenize check).
    """
    ids = tokenizer(" ".join(["!"] * suffix_len), add_special_tokens=False).input_ids
    if len(ids) != suffix_len:  # tokenizer merged differently: fall back
        bang = tokenizer("!", add_special_tokens=False).input_ids[0]
        ids = [bang] * suffix_len
    return torch.tensor([ids])


def keep_retokenizable(tokenizer, candidates: torch.Tensor) -> torch.Tensor:
    """Drop suffixes that would tokenize differently once written as text.

    Otherwise the "found" suffix would not reproduce when typed as a string.
    """
    keep = []
    for row in candidates:
        text = tokenizer.decode(row)
        again = tokenizer(text, add_special_tokens=False).input_ids
        keep.append(again == row.tolist())
    mask = torch.tensor(keep, device=candidates.device)
    return candidates[mask] if mask.any() else candidates


# ---------------------------------------------------------------------------
# Checking the attack
# ---------------------------------------------------------------------------

@torch.no_grad()
def respond(model, tokenizer, before_ids, suffix_ids, after_ids, max_new_tokens: int = 60) -> str:
    """Greedy reply of the model to the prompt with the given suffix."""
    parts = [before_ids, after_ids] if suffix_ids is None else [before_ids, suffix_ids, after_ids]
    input_ids = torch.cat(parts, dim=1)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return strip_thinking(tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_gcg(
    bundle,
    system: str = GANDALF_SCENARIO.system,
    request: str = GANDALF_SCENARIO.request,
    target: str = DEFAULT_TARGET,
    canary: str = GANDALF_SCENARIO.canary,
    suffix_len: int = 20,
    steps: int = 10,
    top_k: int = 256,
    batch_size: int = 128,
    chunk_size: int = 32,
    stop_on_leak: bool = True,
) -> dict:
    """Run GCG and return a dict with the history and final outcome."""
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    for p in model.parameters():
        p.requires_grad_(False)

    before_ids, after_ids = (t.to(device) for t in split_prompt(tokenizer, system, request))
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    vocab_size = model.get_input_embeddings().weight.shape[0]
    allowed = allowed_token_mask(tokenizer, vocab_size)

    suffix_ids = initial_suffix(tokenizer, suffix_len).to(device)

    response_without = respond(model, tokenizer, before_ids, None, after_ids)
    print(f"Reply WITHOUT suffix: {response_without!r}")

    history = []
    best = {"loss": float("inf"), "suffix_ids": suffix_ids.clone(), "step": 0}
    leaked_at = None

    for step in range(1, steps + 1):
        t0 = time.time()
        grad, current_loss = token_gradients(model, before_ids, suffix_ids, after_ids, target_ids)

        candidates = sample_candidates(suffix_ids, grad, allowed, top_k, batch_size)
        candidates = keep_retokenizable(tokenizer, candidates)
        losses = candidate_losses(model, before_ids, candidates, after_ids, target_ids, chunk_size)

        idx = int(losses.argmin())
        suffix_ids = candidates[idx:idx + 1]
        step_loss = float(losses[idx])

        if step_loss < best["loss"]:
            best = {"loss": step_loss, "suffix_ids": suffix_ids.clone(), "step": step}

        reply = respond(model, tokenizer, before_ids, suffix_ids, after_ids)
        leak = detect_leak(reply, canary)
        suffix_text = tokenizer.decode(suffix_ids[0])

        history.append(
            {
                "step": step,
                "loss_before_step": round(current_loss, 4),
                "loss": round(step_loss, 4),
                "suffix": suffix_text,
                "reply": reply,
                "leak": leak or "",
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(f"step {step:3d} | loss {step_loss:.4f} | leak={leak or '-':8s} | {time.time() - t0:5.1f}s | {suffix_text!r}")

        if leak and leaked_at is None:
            leaked_at = step
            if stop_on_leak:
                break

    best_suffix = tokenizer.decode(best["suffix_ids"][0])
    final_reply = respond(model, tokenizer, before_ids, best["suffix_ids"], after_ids)

    return {
        "target": target,
        "initial_loss": history[0]["loss_before_step"] if history else None,
        "best_loss": round(best["loss"], 4),
        "best_step": best["step"],
        "best_suffix": best_suffix,
        "reply_without_suffix": response_without,
        "reply_with_best_suffix": final_reply,
        "leaked_without_suffix": detect_leak(response_without, canary) is not None,
        "leaked_with_best_suffix": detect_leak(final_reply, canary) is not None,
        "first_leak_step": leaked_at,
        "steps_run": len(history),
        "history": history,
    }


def save_loss_plot(history: list[dict], path) -> None:
    """Save the loss curve as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, [h["loss"] for h in history], marker="o")
    ax.set_xlabel("GCG step")
    ax.set_ylabel("Target loss (lower = closer to target)")
    ax.set_title("GCG optimisation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    gpu = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description="Run a benign GCG suffix attack.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--suffix-len", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128 if gpu else 16,
                        help="Candidate suffixes evaluated per step.")
    parser.add_argument("--chunk-size", type=int, default=32 if gpu else 8,
                        help="Candidates per forward pass (lower = less memory).")
    parser.add_argument("--no-stop-on-leak", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    bundle = load_model(args.model)
    print(f"Device: {bundle.device}")

    result = run_gcg(
        bundle,
        target=args.target,
        suffix_len=args.suffix_len,
        steps=args.steps,
        top_k=args.top_k,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        stop_on_leak=not args.no_stop_on_leak,
    )

    run_dir = make_run_dir("gcg")
    pd.DataFrame(result["history"]).to_csv(run_dir / "history.csv", index=False)
    metrics = {k: v for k, v in result.items() if k != "history"}
    save_json(metrics, run_dir / "metrics.json")
    save_json({**vars(args), **environment_info()}, run_dir / "config.json")
    save_loss_plot(result["history"], run_dir / "loss.png")

    # Printed as one JSON block so it can be read straight from job logs.
    import json
    print("\n=== GCG RESULT ===")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
