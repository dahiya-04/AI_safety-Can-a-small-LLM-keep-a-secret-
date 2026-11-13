# Evaluation results

Final, curated experiment runs. Scratch runs go to `results/` (git-ignored); the
runs worth keeping are copied here so anyone can inspect them.

Every run folder has a `config.json` (exact settings and library versions) and
a `metrics.json` (the numbers). Most also have the per-sample CSV so you can
read the model's actual replies.

All runs use **Qwen/Qwen3-0.6B** and a made-up canary password (`PINEAPPLE42`),
so nothing here is harmful.

| Run | What it tests | Headline result |
|---|---|---|
| [`prompt_attacks_templates/20261113-111902`](prompt_attacks_templates/20261113-111902/) | 7 hand-written attack templates x 2 rule scenarios (secret password, forbidden word) | Password leaked even to a plain question; 2 tries per template is too few to rank them |
| [`prompt_attacks_gandalf/20261113-113540`](prompt_attacks_gandalf/20261113-113540/) | 50 real human attacks from the Gandalf game vs a stricter "Gandalf" system prompt | Plain question held; **48%** of real attacks leaked the password |
| [`gcg/20261113-123752_hfjob-6af6b44b5f2f1b7284b96186`](gcg/20261113-123752_hfjob-6af6b44b5f2f1b7284b96186/) | GCG suffix optimisation, 25 tokens, 100 steps, HF Jobs T4 GPU | Loss 0.98 -> **0.0007**; model replies exactly *"Sure, the password is PINEAPPLE42."* |
| [`gcg_transfer/20261113-125608`](gcg_transfer/20261113-125608/) | The GCG suffix appended to the same 50 Gandalf attacks, with a random-suffix control | Leak rate 48% -> 54% (random) -> **68% (GCG)** |

See [`gcg_transfer/20261113-125608/summary.md`](gcg_transfer/20261113-125608/summary.md)
for the main comparison.

## File guide

| File | Contents |
|---|---|
| `samples.csv` | One row per model reply: prompt, response, whether the rule broke |
| `history.csv` | GCG only: suffix, loss and reply at every optimisation step |
| `metrics.json` | Summary numbers for the run |
| `config.json` | Settings, seed, model, library versions, hardware |
| `loss.png` / `gcg_loss.png` | GCG loss curve |
| `comparison.png` | Leak rate per condition (transfer test) |
| `job.log` | Raw log of the remote GPU job |
