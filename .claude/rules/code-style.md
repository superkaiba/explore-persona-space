---
description: Python / experiment code-style conventions (lint, packages, torch idioms, checkpointing, persona injection, reproducibility)
paths:
  - "**/*.py"
  - "configs/**"
---

# Code Style

(Plan-handoff and "all code changes on the local VM, never on pods" stay in
CLAUDE.md as always-on rules; the rest live here and load when you touch code.)

- **Lint:** `uv run ruff check . && uv run ruff format .` (line-length=100, py311, select E/F/I/UP).
- **Packages:** always `uv` (not pip/conda). Config via Hydra (not argparse). Track with `wandb`.
- **Plot fonts (Inter):** `bash scripts/install_inter.sh` once on the dev VM; pods get it via `bootstrap_pod.sh`. Fallback DejaVu Sans.
- **Tensor-shape asserts at boundaries:** `assert logits.shape == (B, T, V), logits.shape`.
- **Vectorize torch ops** — `einops.rearrange`/`einsum`, masked gathers, scatter. No Python loops over tensor dims.
- **Docstring-on-edit:** touching a docstring-less function → add a short one (what + returns/asserts).
- **No dollar-budget caps in experiment scripts.** Never a `max_budget_usd` threshold that raises `SystemExit` mid-experiment (it lost 3 of 4 sources in #356). Log cost telemetry; set billing alerts at the account level. Enforced by `tests/test_no_dollar_budget_caps.py`.
- **Checkpoint per phase; never accumulate-in-memory and write-at-end.** Any multi-phase / multi-domain / multi-condition / multi-seed path MUST persist each phase's output the moment it completes — covers top-level dispatchers AND per-seed eval rigs that chain multiple framework loads (e.g. vLLM gen → logprob on checkpoint → logprob on base). The anti-pattern `results = []; for phase: results.append(...); write(results, path)` turns ANY downstream crash into total data loss for all earlier phases. Acceptable: per-phase files, append-mode idempotent re-runs, per-phase HF/WandB uploads, or load-partial-and-skip-completed at entry.
- **Model call vs code (3.0 paradigm):** before writing any classifier/extractor/parser/summarizer/rule-based judge over unstructured data, evaluate a single Claude Haiku/Sonnet call. If ≥80% covered at acceptable latency/cost, prefer it. Document the choice + rejected alternative in the implementer report + planner §4.
- **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never user/assistant turns.
- **Always run with `nohup`:** `nohup uv run python scripts/train.py &`.
- **Env sync after dep changes:** `uv lock && git push`, then `pod.py sync env`.
- **HF cache** always `/workspace/.cache/huggingface` on pods (symlinks enforce).
- **Reproducibility metadata in result JSONs:** git commit hash, env versions, timestamps.
