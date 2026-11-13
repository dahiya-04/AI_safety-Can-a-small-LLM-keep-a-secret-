"""Text generation utilities."""

import argparse
import re
from typing import Optional

import torch

from .config import DEFAULT_MODEL, DEFAULT_SEED
from .model_loader import load_model
from .utils import set_seed

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def build_chat_text(
    tokenizer,
    prompt: str,
    enable_thinking: bool = False,
    system: Optional[str] = None,
) -> str:
    """Wrap a user prompt in the model's chat template.

    ``enable_thinking`` controls Qwen3's reasoning mode. It is off by default
    so the token budget goes to the answer, not to a <think> block. Templates
    that do not use the flag simply ignore it.
    """
    if not getattr(tokenizer, "chat_template", None):
        return prompt

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def strip_thinking(text: str) -> str:
    """Remove any <think>...</think> block from a response."""
    return _THINK_BLOCK.sub("", text).strip()


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    enable_thinking: bool = False,
    system: Optional[str] = None,
) -> str:
    """Generate a response and return only the newly generated text."""
    text = build_chat_text(tokenizer, prompt, enable_thinking, system)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0, inputs.input_ids.shape[-1]:]
    return strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one response.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        default="Explain why reproducible machine-learning experiments are important.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking mode.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    bundle = load_model(args.model)
    answer = generate_text(
        bundle.model,
        bundle.tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking,
    )
    print(answer)


if __name__ == "__main__":
    main()
