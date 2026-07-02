---
name: Claude file-existence PASS misses a named-deliverable absence
description: A clean-result-critic PASS that verifies "N files exist at the pinned HF/git revision" does NOT verify the plan's named deliverable artifacts are among them; check the plan's registered output filenames against the actual file list before crediting a Lens-13 PASS.
type: feedback
---

When Claude clean-result-critic PASSes a v3 body partly on a mechanical
HF/git-pin existence check ("415 files live at <rev>", check 23/22), that
PASS proves file *existence at a revision*, NOT *deliverable completeness*.
A registered plan deliverable can be silently absent while the count check
still PASSes on the OTHER files. On a Lens-13 (planned-vs-actual) split,
go to the plan's §-deliverables, extract the NAMED output artifact
filenames, and grep the actual HF `list_repo_files` / `git ls-tree` output
for EACH — a missing one that the body neither ships nor labels
"partial / N/A — not produced" is a Real-blocking silent shrinkage.

**Why:** #537 r2 (2026-06-17). Plan v9 §6.5 registered Deliverable 2 as
`win_matrix.json` (predictor × behavior × context-family win/skill matrix
w/ per-cell `family_shuffle_null_skill` + `permutation_p`) — the direct
answer to the verbatim user prompt "which predictors work better for which
contexts/behaviors". It was never produced (0 hits for
`win_matrix`/`family_shuffle`/`permutation_p` on HF `6c5dd0d1` AND git
`c539920`; `descoped_rows` was empty, so not even a recorded descope),
and the body disclosed neither the artifact nor its absence. Claude PASSed
all 15 lenses including check-23's "415 files verified". The 415-file count
was real — but the win_matrix was not among them. The leave-family-out HALF
of Deliverable 2 WAS produced (`leave_family_out_oof_r2` per score row) and
WAS reported (Fig-12 diamond), which made the gap easy to miss: half-shipped,
half-silent. Codex caught it (Lens 13); reconciler upheld → needs_targeted_fix.

**How to apply:** On any clean-result-critic PASS-vs-FAIL split where the
FAIL cites Lens 13 / planned-vs-actual: (1) open the plan's deliverables
section, list every registered OUTPUT FILENAME / artifact; (2) run
`list_repo_files(...revision=<pin>)` (network-enabled — the sandbox DNS
failure is a separate environmental class, NOT grounds for upholding) and
`git ls-tree -r <commit> --name-only`; (3) for each named deliverable absent
from BOTH, confirm the body either ships it OR labels it partial/not-produced;
absent + undisclosed = Real-blocking. A "byte/file count PASS" never
substitutes for this. Bonus tell: a deliverable's SCRIPT present in git
(`i537_win_matrix.py`) but its OUTPUT absent = written-but-never-run.
