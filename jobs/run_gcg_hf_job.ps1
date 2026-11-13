# Run the GCG experiment on a Hugging Face GPU (HF Jobs), billed to your HF credit.
#
# What happens:
#   1. HF starts a fresh Linux container with a T4 GPU (flavor t4-small).
#   2. It installs the pinned Python packages.
#   3. It clones THIS repo's code from GitHub (master branch),
#      so push your latest changes first.
#   4. It runs:  python -m src.experiments.gcg --steps 10 --suffix-len 20
#   5. Logs stream to this terminal; the result is printed under "=== GCG RESULT ===".
#      (Files written inside the container are discarded when the job ends.)
#
# HF API calls made: submitting the job + streaming its logs (about 2-3 calls).
# Cost: T4 is roughly $0.40/hour, billed per minute. --timeout caps the spend.
#
# Requirements:
#   * HF_TOKEN in .env must have the "Jobs" permission
#     (huggingface.co -> Settings -> Access Tokens -> edit token -> Jobs).
#   * Jobs must be enabled on your account (check huggingface.co/settings/billing).
#
# Usage (from the project folder, venv active):
#   .\jobs\run_gcg_hf_job.ps1                     # 10 steps, 20 suffix tokens
#   .\jobs\run_gcg_hf_job.ps1 -Steps 50           # longer run
#   .\jobs\run_gcg_hf_job.ps1 -DryRun             # print the config, submit nothing

param(
    [int]$Steps = 10,
    [int]$SuffixLen = 20,
    [string]$Flavor = "t4-small",
    [string]$Timeout = "30m",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Read HF_TOKEN from .env without printing it.
$tokenLine = Get-Content (Join-Path $PSScriptRoot "..\.env") | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' }
if (-not $tokenLine) { throw "HF_TOKEN not found in .env" }
$env:HF_TOKEN = ($tokenLine -split '=', 2)[1].Trim().Trim('"').Trim("'")

$repo = "https://github.com/dahiya-04/AI_safety-.git"

# One line, no quote characters (Windows PowerShell 5.1 mangles embedded quotes
# when passing arguments to .exe files).
$jobScript = @(
    "pip install -q torch==2.15.0 transformers==5.22.2 accelerate==1.15.0 datasets==5.2.0 pandas==3.1.1 numpy==2.5.4 matplotlib==3.11.2 python-dotenv==1.2.4",
    "git clone --depth 1 --branch master $repo /app",
    "cd /app",
    "nvidia-smi --query-gpu=name,memory.total --format=csv",
    "python -m src.experiments.gcg --steps $Steps --suffix-len $SuffixLen --no-stop-on-leak"
) -join " && "

$hf = Join-Path $PSScriptRoot "..\.venv\Scripts\hf.exe"
$hfArgs = @("jobs", "run", "--flavor", $Flavor, "--timeout", $Timeout, "--name", "gcg-qwen3-0.6b")
if ($DryRun) { $hfArgs += "--dry-run" }
$hfArgs += @("python:3.12", "bash", "-c", $jobScript)

& $hf @hfArgs
