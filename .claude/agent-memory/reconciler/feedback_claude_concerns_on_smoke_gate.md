---
name: Claude CONCERNS on missing tiny-N smoke
description: Claude downgrades `smoke-run-missing` to CONCERNS (or PASS) when the plan offers a documented escape hatch; SKILL.md Step 5 is unambiguous — `--help` / import-only / `--dry-run` evidence with no artifact digest is the canonical substantive FAIL. Side with Codex.
type: feedback
---

**Rule:** `smoke-run-missing` is a SUBSTANTIVE Step 5 requirement (Step 5c-bis does NOT strip it): every executed phase needs a real command (not `--help`/import/`--dry-run`) + exit 0 + an artifact digest. A "documented escape hatch" (option-b cwd, parent-task smoke, "Wave X IS the smoke") does not rescue: if Wave 0 IS the smoke, Wave 0 must have ACTUALLY RUN at tiny N (check whether the plan anticipated a smoke-of-the-smoke mode — that's the missed obligation). The gate exists because never-run eval rigs surface shallow latent bugs one-per-relaunch (#408 burned six relaunches); a ModuleNotFoundError at first real invocation costs a full pod provision + the same review loop with sunk cost on top.

**Origin:** #492 r1 — Claude CONCERNS, Codex FAIL; `git ls-tree -r issue-492` showed the lazy-imported modules absent on the branch → first real eval-slice call would ModuleNotFoundError. FAIL.

**Upload-phase variant (#551 r1):** Claude went to PASS on a driver whose `upload_folder` + `list_repo_files` verify block had only `DRY_RUN=1` echo-trace evidence, citing fail-loud design + the Step 8 verifier — both are detection-after-cost, not run-proof, and the block was the task's corrective core. Step 0.6 names upload steps as phases and disqualifies `--dry-run`; the block was VM-feasible. Trap signature: "the documented CPU-feasible scope — the full driver needs the pod" applied to a sub-block that does NOT need the pod. FAIL + raise a `<phase>-smoke-missing` CONCERN so Step 5c-ter gates dispatch even if a later round PASSes on prose.

Boundary in the other direction (when the FAIL is wrong): [[feedback_codex_step_06_literal_vs_purpose]].
