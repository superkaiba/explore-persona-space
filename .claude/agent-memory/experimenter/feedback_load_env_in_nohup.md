---
name: Load .env explicitly when launching via nohup
description: nohup-detached commands inherit the SSH non-login shell env, which lacks the project .env keys (ANTHROPIC_*, WANDB_API_KEY, HF_TOKEN). Always wrap with `set -a; source .env; set +a`.
type: feedback
---

When launching a python entrypoint via `ssh epm-... 'nohup ... &'`, the
process inherits the SSH non-login shell environment. That environment does
NOT have the project `.env` loaded (no .bashrc / .profile sourced). The
Python entrypoint's `setup_env()` will load .env from the file, but ONLY
if it's called early enough — and many helper scripts (e.g.,
`scripts/build_issue260_data.py`) call `os.environ["ANTHROPIC_BATCH_KEY"]`
directly without going through setup_env. That `KeyError` then aborts the
process silently.

**The fix:** ALWAYS prefix nohup launches with `set -a; source .env; set +a`:

```bash
ssh epm-issue-260 'cd /workspace/explore-persona-space && \
  set -a && source .env && set +a && \
  export PATH=$HOME/.local/bin:$PATH && \
  nohup env PYTHONHASHSEED=42 uv run python scripts/launch_issue260.py epm-issue-260 \
    > eval_results/issue260/launcher.log 2>&1 &'
```

**Why:** issue #260 first-experimenter respawn (`epm:experimenter-respawn v1`,
2026-05-06): the agent spawned a build process without sourcing .env;
ANTHROPIC_BATCH_KEY was missing; the Anthropic Batch API call hung
(blocking on a missing key with no error visible because Python output was
also fully-buffered to a file). User had to manually respawn the pipeline.

**How to apply:** Every nohup-detached pod command that calls into Anthropic /
WandB / HF Hub APIs must source the .env first. Don't rely on the python
entrypoint's setup_env() — explicit env-load in the bash wrapper is the
defense-in-depth that survives any code path that bypasses setup_env.
