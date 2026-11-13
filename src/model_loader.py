"""Model loading utilities for AAIGM."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEFAULT_MODEL
from .utils import default_dtype, get_device


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> ModelBundle:
    """Load a causal language model and tokenizer onto one device.

    Uses the GPU when available, otherwise the CPU. The same code therefore
    runs on a laptop and on a cloud GPU (HF Jobs, Kaggle, Colab).
    """
    device = device or get_device()
    dtype = dtype or default_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()

    return ModelBundle(model=model, tokenizer=tokenizer, device=device)


def environment_info() -> dict:
    """Return basic runtime information (useful to save with results)."""
    import transformers

    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": get_device(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


if __name__ == "__main__":
    print(environment_info())
