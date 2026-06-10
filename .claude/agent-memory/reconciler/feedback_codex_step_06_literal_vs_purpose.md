---
name: codex-step-06-literal-vs-purpose
description: Codex code-reviewer FAILs Step 0.6 (smoke-run-missing) on GPU-gated phases that physically cannot run on CPU dev VM; reads spec literally ("no --help") and ignores that orchestrator Step 6d.0-bis is the pod-side safety net
metadata:
  type: feedback
---

Codex code-reviewer reads `code-reviewer.md` Step 0.6 literally — "FAIL when any phase shows only `--help` / `import` / `--dry-run` evidence" — and FAILs a round-N where the implementer ran ALL CPU-feasible phases end-to-end (with real artifact digests) and exercised GPU-gated phases via `--help` + in-process helpers. Codex does not factor in that:

1. The GPU-gated phases physically cannot run end-to-end on the CPU dev VM (no GPU).
2. The orchestrator has a Step 6d.0-bis pod-side gate (`/issue` skill SKILL.md lines 1481-1499) that REFUSES to dispatch the production launch until every phase has run end-to-end at tiny-N on the pod.
3. Step 0.6's stated PURPOSE is "GPU-protection" (code-reviewer.md line 222) — preventing wasted pod cycles on bugs that would surface at first real GPU launch. Step 6d.0-bis is the pod-side enforcement of exactly that purpose.

**Why:** the spec text was authored against the common case (CPU-feasible phases skipped end-to-end out of laziness). For experiments where phases genuinely require GPU, the spec doesn't carve out an explicit exception, but it also doesn't intend to block PASS on CPU-only review. The orchestrator's Step 6d.0-bis is the load-bearing safety net.

**How to apply:** when adjudicating a Codex FAIL tagged `smoke-run-missing` on a `type:experiment` round-N:

1. Open the implementer's `## Smoke run` section. Categorize each phase as CPU-feasible vs GPU-required.
2. For CPU-feasible phases: did the implementer run them end-to-end with real artifact digests (not `--help`)? If yes, the spec's purpose is met for those phases.
3. For GPU-required phases: confirm `.claude/skills/issue/SKILL.md` Step 6d.0-bis is intact and will fire before production launch. Run `grep -n "6d.0-bis\|tiny-N" .claude/skills/issue/SKILL.md` — the gate must be present.
4. If (2) is true AND (3) confirms the pod-side gate is wired, PASS with a HARD standing recommendation: "Pod-side Step 6d.0-bis MUST be enforced; if it is silently skipped, this PASS becomes load-bearing on a gate that didn't fire."
5. If the implementer skipped CPU-feasible phases too — that's a genuine FAIL; the spec text applies.

Origin: task #464 round 4 — phases 1 (R-gen), 4 (cross-eval), 4.5 (trained-greedy) all require GPU and cannot run end-to-end on the CPU dev VM. Implementer ran phase 0 (full preflight), phase 5 (three end-to-end paths through `main()` on a 90-cell stub tree, exit 0), and the plot script (full 11-PNG run). Codex FAILed on `--help`-only for the GPU phases. Adjudicated PASS because Step 6d.0-bis catches the GPU-phase smoke at the pod before production.

Recurrence: task #557 round 4 (2026-06-10) — carve-out-LABEL variant. The spec now has an explicit `### <phase> — Carve-out (GPU-bound)` contract (code-reviewer.md Step 0.6, added after #514 r2). Codex FAILed `smoke-run-missing` because the literal sub-heading was absent, even though ALL THREE substitute items were substantively present in the marker (real CPU smoke + dispatcher dry-run + runpy execution of the production CLI through main(), each with command + exit 0 + artifact) AND the GPU-constraint sentence was verbatim present, AND Codex itself wrote "code path is substantively correct" with zero other findings. Adjudicated PASS via the CONCERNS clause ("evidence present, only formatting imperfect → CONCERNS, never FAIL"): a labeled-vs-unlabeled carve-out is a formatting defect when the documentation is unambiguous. Extra weight when the only never-executed code is itself a fail-loud assert whose failure surfaces within ~60s of pod relaunch (the gate's GPU-protection purpose is satisfied by construction). Always make the pod-side `[gpu-pin]`/probe check a MANDATORY standing recommendation, and tell the implementer to use the literal sub-heading next round.

Related: [[codex-conflates-marker-format-with-code]] (similar pattern — Codex spec-literalism that ignores orchestrator-side mechanical strip / pod-side safety net).
