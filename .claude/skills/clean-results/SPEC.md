# Clean-result spec — RETIRED, see task-workflow spec

The canonical spec for clean-result body shape, voice, sections, and anti-patterns is **`~/sagan/docs/clean-result-guidelines.md`** (clean-result HTML format). The mechanical verifier is **`scripts/verify_sagan_card.py`** (11 checks). The worked example is **experiment #311** at <https://eps.superkaiba.com/tasks/<N>>.

This file used to be the EPS-internal v4 markdown spec (`## TL;DR` / `## Summary` / `## Details`). That format was retired during the 2026-05-13 clean-result migration. The v4 markdown verifier (`scripts/verify_clean_result.py`) and body-discipline auditor (`scripts/audit_clean_results_body_discipline.py`) are kept available for grandfathered awaiting_promotion bodies that have not yet been re-converted, but no new bodies should target that format.

## Where the consolidated rules live now

| What | Lives at |
|---|---|
| Body structure (3 pieces + repro appendix) | `~/sagan/docs/clean-result-guidelines.md` § "Top-level structure" |
| Title rules | `~/sagan/docs/clean-result-guidelines.md` § "Title" |
| TL;DR (4 bullets, labels, "I"-voice) | `~/sagan/docs/clean-result-guidelines.md` § "TL;DR (four bullets)" |
| Primary plot conventions | `~/sagan/docs/clean-result-guidelines.md` § "Primary plot" |
| Design dropdown rules | `~/sagan/docs/clean-result-guidelines.md` § "Experimental design (collapsible dropdown)" |
| Sample-output discipline (cherry-picked label, qualitative-data link) | `~/sagan/docs/clean-result-guidelines.md` § "Experimental design (collapsible dropdown)" |
| Reproducibility appendix | `~/sagan/docs/clean-result-guidelines.md` § "Reproducibility appendix (agent-facing, collapsible)" |
| Sections to avoid | `~/sagan/docs/clean-result-guidelines.md` § "Sections to avoid" |
| Voice rules | `~/sagan/docs/clean-result-guidelines.md` § "Voice rules (consolidated)" |
| Statistical-framing rule (p-values + N only in prose) | `CLAUDE.md` § "Experiment Report Structure" |
| Worked example | Experiment #311 (live body on EPS dashboard) |
| Mechanical verifier (11 checks) | `scripts/verify_sagan_card.py` |
| Past-correction log (still active) | `.claude/skills/clean-results/iterations.md` |
| LessWrong post exemplars (register reference for design-dropdown prose) | `.claude/skills/clean-results/lw-post-examples/` |

## What this directory still owns

- **`iterations.md`** — append-only log of corrections + the rules they produced. Continue to log here when an iteration during `/promote-clean-result` uncovers a generalisable rule. The "fold into" pointer should target `~/sagan/docs/clean-result-guidelines.md` for new structural rules, or `scripts/verify_sagan_card.py` for new mechanical checks.
- **`lw-post-examples/`** — 3 verbatim LessWrong research posts kept for register reference. The clean-result design-dropdown narrative is more compressed than a LW post but the prose discipline (concrete numbers, comparison anchors, plain English, no jargon-without-definition) carries over.

## Calling sites that now point at the task-workflow spec

- `.claude/agents/analyzer.md` Step 4 — drafts the body following clean-result spec.
- `.claude/agents/clean-result-critic.md` — critiques against clean-result lenses + `verify_sagan_card.py`.
- `.claude/skills/promote-clean-result/SKILL.md` — auto-converts legacy markdown bodies to clean-result HTML.
- `CLAUDE.md` § "Experiment Report Structure" — points at the task-workflow spec.

If you arrived here looking for the EPS-v4 markdown spec, see the git history of this file before commit `<TBD migration commit>`.
