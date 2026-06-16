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
- **Compute-throughput discipline (GPU experiment scripts).** Batch data-parallel model forwards — never a Python loop of batch-1 forwards (a 7B bf16 batch-1 forward is weight-bandwidth-bound and leaves the GPU ~idle). Keep large-tensor reductions GPU-resident and transfer only the reduced scalars/summaries to CPU — never ship `(seq × vocab)`-scale fp32 tensors over PCIe for a CPU-side reduce. Before launching any run estimated >12h, sanity-check the estimate against the FLOPs floor (`n_forwards × 2 · params · tokens_per_forward / sustained GPU FLOPs`): an estimate >5-10× over the floor means the implementation is leaving throughput on the table — fix it before launch, don't book more pod-days (#522: ~94h on 1× H100 for a ~4-6h-floor job — 409,600 batch-1 forwards with full-vocab log-softmax shipped to CPU for a per-position reduce; #511: 52× CPU wall-time blowup vs its plan estimate from a full `eigh` on a 2N×2N joint Gram).
- **Docstring-on-edit:** touching a docstring-less function → add a short one (what + returns/asserts).
- **No dollar-budget caps in experiment scripts.** Never a `max_budget_usd` threshold that raises `SystemExit` mid-experiment (it lost 3 of 4 sources in #356). Log cost telemetry; set billing alerts at the account level. Enforced by `tests/test_no_dollar_budget_caps.py`.
- **Checkpoint per phase; never accumulate-in-memory and write-at-end.** Any multi-phase / multi-domain / multi-condition / multi-seed path MUST persist each phase's output the moment it completes — covers top-level dispatchers AND per-seed eval rigs that chain multiple framework loads (e.g. vLLM gen → logprob on checkpoint → logprob on base). The anti-pattern `results = []; for phase: results.append(...); write(results, path)` turns ANY downstream crash into total data loss for all earlier phases. Acceptable: per-phase files, append-mode idempotent re-runs, per-phase HF/WandB uploads, or load-partial-and-skip-completed at entry.
- **Model call vs code (3.0 paradigm):** before writing any classifier/extractor/parser/summarizer/rule-based judge over unstructured data, evaluate a single Claude Haiku/Sonnet call. If ≥80% covered at acceptable latency/cost, prefer it. Document the choice + rejected alternative in the implementer report + planner §4.
- **Never hardcode an invented Claude/Anthropic model id.** The judge default is `claude-sonnet-4-5`; canonical ids live in CLAUDE.md / the global `claude-api` skill. A `*-20251001` suffix is **Haiku 4.5**, not Sonnet — do not graft a date suffix onto a Sonnet id. Verify any hardcoded model string against the canonical list before committing; a wrong id crashes the run at the first API call (#489 Phase 0a, 2026-06-04).
- **Judge / API-call retry wrappers treat 529 Overloaded as transient by default.** The transient tuple is `(APIConnectionError, APITimeoutError, RateLimitError, anthropic.InternalServerError)` — 529 `OverloadedError` is NOT a top-level SDK symbol (`anthropic.OverloadedError` raises AttributeError) but IS an `InternalServerError` subclass in the installed SDK, so catching `InternalServerError` covers it. Also harden long judge loops with checkpoint/resume + a per-row `judge_failed: true` audit flag instead of crashing the whole launcher on one bad row (incidents #556: ~97 OverloadedErrors crashed a 4400-judgment phase; #528: three distinct judge crashes — empty response SystemExit, one soft-refusal row, one JSONDecodeError).
- **WandB live training metrics are mandatory.** Any training-config builder under `src/explore_persona_space/experiments/` (`TrainLoraConfig`, `SFTConfig`, `TrainingArguments`) MUST emit to WandB during training — loss curves, grad-norm history, and callback metrics cannot be reconstructed post-hoc. Do NOT set `report_to="none"` / `report_to=None` / `report_to=[]` without an explicit `# WANDB_INTENTIONALLY_DISABLED: <reason ≥ 10 chars>` comment on the same line or the immediately preceding non-blank line. The waiver is a justification, not a token bypass; if you cannot articulate a real reason in ≥10 chars, you should not be disabling it. Issue #496 trained 12 cells with `report_to="none"` hardcoded and the missing telemetry surfaced only at upload-verification (Step 8). Enforced by `scripts/workflow_lint.py --check-wandb-required` (bundled into the no-flags default run).
- **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never user/assistant turns.
- **Always run with `nohup`:** `nohup uv run python scripts/train.py &`.
- **Env sync after dep changes:** `uv lock && git push`, then `pod.py sync env`.
- **HF cache** always `/workspace/.cache/huggingface` on pods (symlinks enforce).
- **Reproducibility metadata in result JSONs:** git commit hash, env versions, timestamps.
