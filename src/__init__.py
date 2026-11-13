import os

# Windows without Developer Mode cannot create symlinks; the HF cache still
# works, so hide the noisy warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
