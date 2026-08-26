---
name: targeted-fix-round-worktree-commit-and-concern-residual
description: Worktree artifact commits need the git -C form (cd $WT via variable trips the root-code guard); verify --issue Lens-14 audit counts orchestrator-owned open concerns — report the residual, never fabricate an address row.
metadata:
  type: feedback
---

Two mechanics from the #2564 r6 targeted-fix round (2026-08-26):

1. **Worktree commits: use `git -C "$WT" commit -F <msgfile> -- <paths>`.**
   A compound `cd $WT && git add ... && git commit ...` with `$WT` as a
   VARIABLE is not recognized as a worktree cd by `guard_root_code_commit.sh`
   — it blocks the whole call as an uncertified repo-root code commit (the
   add never runs either). The guard's own remediation names the `git -C`
   form; worktree branches are gated at Step 10d, not by the inline lint
   gate.
2. **`verify_task_body.py --issue` runs a concerns audit that `--file` mode
   skips** (no issue number ⇒ no ledger). It FAILs on ANY open binding
   concern, including ones the brief explicitly reserves to the orchestrator
   (e.g. `methodology-export-stale-after-fold` — the Step 9a-quater
   export/pointer refresh that runs AFTER analyzer fixes land).
   **Why:** deferral markers are user-only and a premature `address-concern`
   is a fabricated resolution (Lens-14 fabrication FAIL class).
   **How to apply:** address-concern ONLY the concerns your fixes actually
   resolved (≤200-char summaries), re-run verify, and when the sole residual
   is orchestrator-owned, say so in the epm:interpretation marker + final
   report (name the clearing actor/step) instead of forcing OVERALL PASS.
