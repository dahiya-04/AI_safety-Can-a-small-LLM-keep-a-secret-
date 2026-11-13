# Run the GCG experiment on a Hugging Face GPU (HF Jobs), billed to your HF credit,
# and save the results locally under results\gcg\.
#
# What happens:
#   1. HF starts a fresh Linux container with a T4 GPU (flavor t4-small).
#   2. It installs the pinned Python packages.
#   3. It clones THIS repo from GitHub (master branch) - push your changes first.
#   4. It runs:  python -m src.experiments.gcg --steps 50 --suffix-len 20
#   5. This script streams the job's logs to the terminal AND to
#      results\gcg_jobs\<job-id>.log
#   6. It rebuilds the results locally from that log:
#      results\gcg\<timestamp>_hfjob-<job-id>\  (history.csv, metrics.json,
#      config.json, loss.png, job.log)
#
# HF API calls made: 1 to submit the job + 1 to stream its logs.
# Cost: T4 is roughly $0.40/hour, billed per minute. --timeout caps the spend.
#
# Requirements:
#   * HF_TOKEN in .env with the "Jobs" permission.
#
# Usage (from the project folder, venv active):
#   .\jobs\run_gcg_hf_job.ps1                 # 50 steps, 20 suffix tokens
#   .\jobs\run_gcg_hf_job.ps1 -Steps 10       # shorter run
#   .\jobs\run_gcg_hf_job.ps1 -DryRun         # print the config, submit nothing
#
# If the log stream is interrupted, the job keeps running on HF. Recover with:
#   hf jobs logs <job-id> > results\gcg_jobs\<job-id>.log
#   python -m src.experiments.gcg_import results\gcg_jobs\<job-id>.log

param(
    [int]$Steps = 50,
    [int]$SuffixLen = 20,
    [string]$Flavor = "t4-small",
    [string]$Timeout = "45m",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$hf = Join-Path $root ".venv\Scripts\hf.exe"
$python = Join-Path $root ".venv\Scripts\python.exe"

# Read HF_TOKEN from .env without printing it.
$tokenLine = Get-Content (Join-Path $root ".env") | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' }
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

$runArgs = @("jobs", "run", "--flavor", $Flavor, "--timeout", $Timeout, "--name", "gcg-qwen3-0_6b")

if ($DryRun) {
    & $hf @($runArgs + @("--dry-run", "python:3.12", "bash", "-c", $jobScript))
    return
}

# 1) Submit in the background; -q prints only the job ID.
Write-Host "Submitting job ($Steps steps, suffix $SuffixLen tokens, $Flavor)..."
$submitOut = & $hf @($runArgs + @("--detach", "-q", "python:3.12", "bash", "-c", $jobScript))
$jobId = ($submitOut | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1).Trim()
if (-not $jobId) { throw "Could not read job ID from: $submitOut" }
Write-Host "Job ID: $jobId"
Write-Host "Streaming logs (Ctrl+C stops watching; the job keeps running)..."

# 2) Stream logs to the terminal and keep them.
$logDir = Join-Path $root "results\gcg_jobs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$logPath = Join-Path $logDir "$jobId.log"

& $hf jobs logs --follow --format human $jobId | Tee-Object -Variable logLines
$logLines | Out-File -Encoding utf8 $logPath
Write-Host "`nLog saved to $logPath"

# 3) Rebuild results\gcg\<timestamp>_hfjob-<id>\ from the log.
Push-Location $root
try {
    & $python -m src.experiments.gcg_import $logPath --name "hfjob-$jobId"
} finally {
    Pop-Location
}
