---
name: Codex FAILs a REAL code defect that a dispatcher branch makes unreachable on the actual run path
description: Codex flags a genuinely-broken code site (the pattern IS wrong) as production-blocking without tracing the dispatcher --from-phase / env branch that makes the site unreachable on THIS run's actual path. Trace control-flow reachability from the real launch command; a real-but-off-path defect is PASS + persisted CONCERN, never an upheld FAIL.
type: feedback
---

**Rule:** a code-review FAIL adjudicates whether the artifact blocks THE RUN THAT
IS ACTUALLY ABOUT TO EXECUTE — not whether a code site is defective in the
abstract. When Codex cites a real broken pattern at a specific line, before
upholding the FAIL trace the control-flow from the CONCRETE launch command
(the `--from-phase`, env-var, or CLI-arg the pending run uses) to that line:

1. **Find the ONLY call site(s)** of the flagged symbol (grep the dispatcher /
   entrypoint). A defect in a helper is only reachable through its callers.
2. **Check every branch guard on the path** to that call site — a shell
   `if [ "$FROM_PHASE" != "X" ]`, a Python `if args.phase == ...`, a `--from-phase`
   split, a mode flag. If the pending run's launch command takes the OTHER branch,
   the flagged line NEVER EXECUTES on this run.
3. **Distinguish "this run" from "a hypothetical fresh full boot."** Codex's
   counter-argument often silently widens scope to "a fresh full production boot"
   or "the general case" — that is the tell. The reconcile question is the SPECIFIC
   relaunch (e.g. `--from-phase pv_capture` on a stopped-and-resumed matched-host
   pod), not every hypothetical entry.
4. **A real-but-off-path defect is PASS + persisted CONCERN, NOT an upheld FAIL.**
   The pattern is genuinely wrong (Codex is not fabricating), so do NOT discard it —
   `raise-concern --severity CONCERN` so the fix lands in a follow-up cleanup round,
   and PASS the run. Discarding (Weight `Discarded`) is wrong here: Codex found a
   real bug; it just doesn't gate this run.

**Origin (#763 r5, CAP-3, code-reviewer):** Codex Critical flagged
`issue763_stage_pools.py:56` `snapshot_download(allow_patterns=[...])` — a genuinely
broken siblings-truncation pattern (4th site of the family, >94k-file repo). But its
ONLY call site is `issue763_dispatch.sh:146`, inside the Phase-1 block gated by
`if [ "$FROM_PHASE" != "pv_capture" ]` (line 139). The pending relaunch was
`--from-phase pv_capture`, whose Phase-2 block (lines 180–209) never calls
`stage_pools.py`. Codex's own text conceded the finding was about "a fresh full
production boot" — not the actual relaunch. Codex Major (`_stage_gen_from_hf`
partial-dir treated complete, `judge_e0.py:506`) was likewise real but bounded: a
matched-host resume keeps a complete volume, AND the downstream `yield_shortfall`
floor (`floor=int(0.8*m_B)` → `any_shortfall` in the results schema) surfaces any
partial cell set, so it is not silent. Adjudicated PASS; both persisted as CONCERNs.
Upholding FAIL would have burned a CAP-3 strategy pivot (losing plan v3 + 3 crash-fix
rounds) over two off-path / downstream-guarded concerns.

**Boundary (uphold the FAIL):** if the pending run's launch command takes the
branch that DOES reach the flagged line, or the "downstream guard" you rely on
doesn't actually catch the failure mode (verify the guard's fail condition — see
[[feedback_claude_cites_nonexistent_backstop_semantics]]), the FAIL stands. Also
uphold if the defect is on the load-bearing corrective path the round was FIXING
(then it's not off-path at all).

Siblings: [[feedback_codex_plan_section_in_scope]] (reachability of an un-invoked
plan-section path — the general reachability-walk); [[feedback_codex_env_var_orphan_unreachable]]
(trace the import chain to the real entry point); [[feedback_codex_litigates_pre_existing_in_round_n]]
(git-provenance rather than control-flow); [[feedback_codex_step_06_literal_vs_purpose]]
(gate topology — "does the changed code run before production").
