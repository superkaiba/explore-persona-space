---
name: Load .env explicitly when launching via nohup
description: SSH non-login shells have no project .env loaded; helper scripts reading os.environ directly KeyError or hang silently. Always wrap nohup launches with set -a; source .env; set +a.
type: feedback
---

`ssh pod 'nohup ... &'` inherits the non-login shell env — no .bashrc, no project `.env` (ANTHROPIC_*, WANDB_API_KEY, HF_TOKEN). Helper scripts that read `os.environ["ANTHROPIC_BATCH_KEY"]` directly (bypassing the entrypoint's env setup) then abort or hang with output fully buffered to the log — invisibly.

**Why:** #260 first-experimenter respawn (2026-05-06) — a build process launched without sourcing .env hung on the missing batch key; user had to manually respawn.

**How to apply:** every nohup-detached pod command touching Anthropic/WandB/HF MUST be prefixed `set -a && source .env && set +a` (plus `export PATH=$HOME/.local/bin:$PATH`). Explicit env-load in the bash wrapper is the defense that survives any code path bypassing setup_env. Related: [[feedback_pod_provision_uv_missing]] (PATH gap), [[feedback_wrapper_pipefail]].
