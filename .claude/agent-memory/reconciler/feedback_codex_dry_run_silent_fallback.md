---
name: Codex over-flags dry-run "silent fallback" on missing-but-pod-side-inputs
description: Codex FAILs `dry-run-missing-paths-not-fail-loud` when a dev-VM dry-run logs WARNINGs + falls back for inputs the pod-side path enforces with fail-loud raises. The fail-loud IS present, just at the right scope.
type: feedback
---

When the dry-run path emits explicit WARNING log lines for missing inputs and reports the resolved-fallback paths in the DRY_RUN_PASS JSON payload, that is NOT a "silent fallback" — it is a transparent dev-VM gate. Codex code-reviewer reads "all required inputs must fail loud" literally and FAILs anyway, missing that the non-dry-run pod-side branch DOES raise FileNotFoundError on absent inputs where it matters for correctness.

**Why:** The orchestration design intentionally splits dev-VM offline-validation (validate what's present, log what's missing, exit 0) from pod-side fail-loud (raise on any genuinely-absent input). Inputs that live on pod-504 / HF Hub are EXPECTED to be absent on the dev VM; the dry-run's job is import/argparse/path-resolve + local schema validation, not staging.

**How to apply:** When Codex's only substantive blocker is `dry-run-missing-paths-not-fail-loud`:
1. Read the dry-run path AND the non-dry-run path in the same script.
2. Verify the non-dry-run branch raises on the same missing inputs Codex flags.
3. Reproduce the dry-run in a clean offline env (`env -i HF_HOME=/tmp/empty-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python <script> --dry-run`).
4. If missing-input warnings appear in the log AND the resolved paths appear in the DRY_RUN_PASS JSON AND the non-dry-run branch fails loud → PASS with a standing recommendation to mirror any machine-readable existence flags from sibling scripts.

Companion to `feedback_codex_step_06_literal_vs_purpose.md` (Codex's literal reading of "smoke must fail loud" on GPU-gated phases). This is the same pattern applied to dry-run input-presence checks. Both reduce to: Codex reads a fail-loud rule literally without understanding the dev-VM-vs-pod scope split.

Incident: #504 round 14 (loop round 2). Round-13 reconciler v9 already classified the analogous R_eval finding as `Real-nonblocking — Discarded` for the same reason (production library regenerates R on-policy; absent file is a preflight optimization, not a correctness guard). Round-14 carries the same disposition forward.
