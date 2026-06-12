---
name: Claude CONCERNS on missing tiny-N smoke
description: Claude code-reviewer downgrades `smoke-run-missing` from FAIL to CONCERNS when the plan offers a documented escape hatch (option-b cwd, prior task's smoke-check, etc.); SKILL.md Step 5 is unambiguous that `--help` / import-only evidence is the canonical-FAIL case
type: feedback
---

When the implementer's `## Smoke run` section for an experiment task contains
only `--help` and an `importlib` import-check — explicitly omitting lazy-loaded
runtime dependencies — Claude code-reviewer tends to read this as a CONCERNS-
class finding if the plan provides a "documented" fallback (e.g. plan §4.1
option (b) "run from `wt-issue-477` checkout", or the script was smoke-checked
on the parent task). Codex correctly reads this as the canonical
`smoke-run-missing` FAIL per SKILL.md Step 5.

**Why:** SKILL.md Step 5's end-to-end smoke gate is a SUBSTANTIVE requirement,
not a marker-shape requirement (so Step 5c-bis does NOT strip it). The text is
unambiguous: "any sub-section [that] shows only `--help` / `import` /
`--dry-run` evidence (or exits non-zero, or carries no artifact digest), the
reviewer posts `FAIL` with blocker `smoke-run-missing`". The gate exists
specifically because never-before-run eval rigs surface shallow latent bugs
one-per-run at the real eval phase (incident #408 burned six relaunches
catching one bug per cycle on a 203 KB eval rig).

**How to apply:** When the disputed finding is `smoke-run-missing`:

1. Open the implementer marker's `## Smoke run` section. For each phase the
   pipeline actually executes (Wave 0, Wave 1, data-gen, training, eval, …),
   check: does the sub-section have a real command (NOT `--help` / NOT just
   `import` / NOT `--dry-run`) + exit 0 + an artifact digest (path + shape /
   key fields / row count)?
2. If ANY phase fails this test, the FAIL is real-blocking, regardless of how
   compelling the plan's "documented escape hatch" looks. Side with Codex.
3. Special trap: when the implementer claims "Wave 0 IS the smoke" (or any
   "phase X is the smoke" phase-collapse argument), Wave 0 itself must have
   ACTUALLY RUN at tiny N to defend that claim. Check whether the plan's own
   text anticipates a "smoke-of-the-smoke" mode (e.g. `--n-heldout 5
   --n-questions 2`) — that's the implementer's missed obligation, not an
   alternative path.
4. Autonomous-mode cost framing favors FAIL: a ModuleNotFoundError or other
   shallow latent bug at first real invocation burns the full pod provision +
   setup wall (~20-30 min on a fresh eval pod), and because the failure won't
   auto-resolve the orchestrator pivots back into the same implementer/reviewer
   loop the CONCERNS framing was trying to avoid — but with sunk pod-cost on
   top.

**Origin:** task #492 round-1 reconcile (2026-06-05). Claude returned CONCERNS;
Codex returned FAIL with `smoke-run-missing` + `substantive`. Direct verification
of `git ls-tree -r issue-492` showed the lazy-imported modules absent on the
branch, which the cherry-picked script's lazy imports would `ModuleNotFoundError`
on at the first real-run eval-slice call — a class of bug the SKILL.md-required
tiny-N GPU smoke would have caught automatically.

**Upload-phase variant (task #551 round-1, 2026-06-10):** Claude went all the
way to PASS (not even CONCERNS) on a driver whose `upload_folder` +
`list_repo_files` verify block had ONLY `DRY_RUN=1` echo-trace evidence,
reasoning the block is fail-loud (pod kept alive on miss) + the Step 8
upload-verifier re-checks post-run. Both are detection-after-cost, not proof —
and the upload block was the task's corrective core (#521's tensors were never
persisted). Step 0.6 explicitly names "upload steps" as phases and explicitly
disqualifies `--dry-run` evidence; the block was VM-feasible (pure
huggingface_hub, fixtures on disk). Adjudicated FAIL. New trap signature:
"the documented CPU-feasible scope — the full driver needs the pod" in the
implementer's smoke section, applied to a sub-block that does NOT need the
pod. Also: per the reconciler persistence duty, raise a
`<phase>-smoke-missing` CONCERN so Step 5c-ter gates dispatch even if a later
round PASSes on prose.
