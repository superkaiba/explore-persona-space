# HF model repo file-count audit — issue #1108

**Repo:** `superkaiba1/explore-persona-space` — **99,943 / 100,000 git files**
(57 files of headroom) at revision `79b67131483c773d21a75bd8f6c68279165b594e`
(`pre_deletion_revision` — deleted files remain fetchable at pre-deletion
revisions via `revision=` pinning: HF deletion frees the HEAD tree, not
history, so a mistakenly pruned rung is recoverable without retraining and a
future planner's reuse fitness check can resolve it). Enumerated
2026-07-07T11:47:17Z; total blob size 3.04 TB. Overflow
repo `superkaiba1/explore-persona-space-overflow` currently holds 207
files (separate per-repo budget).

## Safety invariant

Only artifacts of TERMINAL tasks (status completed/archived) are proposed;
final adapters are never in option (a)'s command set; deletion is
USER-EXECUTED; nothing in this audit (or the script that produced it) deletes
from HF.

## Attribution coverage

- task-resolved files: **92,441**
- named legacy-prefix files (task=None — structurally CANNOT enter
  terminal-task deletion commands): **7,502**
- unattributed: **0**
- conservation identity: task + named + unattributed == n_files (asserted in code)
- coverage (task + named) / n_files = **100.0%** (gate: >=95%)

Prefix-only attribution is never deletion-actionable: only TASK-resolved files
of TERMINAL tasks enter any command set.

## Ladder split

**53,389 / 99,943 files (53.4%)
sit inside `checkpoint-*/` dirs** (4,441 rung dirs,
1.32 TB). Footnote: rung-per-top-level-prefix layouts (e.g. issue466_*_step1600, i398_*_step_checkpoints) do not match /checkpoint-\d+/, so this figure and option (a) UNDERCOUNT true ladder residue; directionally safe.

Ladder mass by owning-task status (only completed/archived tasks enter the
deletion candidate set — a large share sits at `awaiting_promotion` and
becomes prunable only at promotion):
- awaiting_promotion: 29,981 ladder files across 31 tasks
- completed: 18,592 ladder files across 71 tasks
- archived: 720 ladder files across 10 tasks
- followups_running: 576 ladder files across 1 tasks

Top tasks by file count:

| task | status | classification | files | ladder files | rungs | GB |
|---|---|---|---|---|---|---|
| #597 | awaiting_promotion | useful | 12,312 | 11,916 | 993 | 347.7 |
| #397 | completed | useful | 7,668 | 6,480 | 540 | 219.6 |
| #621 | awaiting_promotion | - | 7,039 | 6,588 | 549 | 10.4 |
| #628 | completed | useful | 4,432 | 3,816 | 318 | 36.2 |
| #474 | completed | useful | 4,034 | 1,575 | 105 | 187.7 |
| #545 | awaiting_promotion | - | 3,048 | 2,349 | 156 | 296.2 |
| #537 | awaiting_promotion | - | 2,432 | 840 | 56 | 59.2 |
| #543 | completed | useful | 2,325 | 1,995 | 133 | 19.9 |
| #542 | awaiting_promotion | - | 2,320 | 1,176 | 98 | 19.5 |
| #570 | completed | useful | 2,313 | 2,112 | 176 | 11.5 |

## Options

- **(a) Prune non-selected rungs of TERMINAL tasks.**
  - Upper bound (prune ALL terminal-task `checkpoint-*/` dirs):
    **19,312 files / 0.45 TB**.
  - Conservative (keep rule, pinned verbatim: "keep the single highest-step checkpoint-* dir per PARENT adapter dir, plus all non-checkpoint files"):
    **15,182 files / 0.34 TB**.
    Per-parent-dir rung lists are in the JSON
    (`options.prune_terminal_ladders_conservative.per_parent_rungs`) so the
    estimate is recomputable. **Caveat:** "keep max-step" does NOT protect band-stop/dose-selected rungs — the project's canonical reuse pins EARLY/mid-band rungs (#532 reused #474's epoch-1 adapters), so the citation cross-check (cited_by), not the keep rule, is the actual protection.
  - Both numbers are ESTIMATES over the full candidate set; the executable
    ready-to-paste command set in `freeing_commands.md` contains ONLY blocks
    whose `cited_by` is empty (see it for the smaller executable total).
- **(b) Archive whole terminal-task adapter trees** (wandb-archive precedent:
  `superkaiba1/explore-persona-space-wandb-archive`), then delete —
  **51,682 files / 1.55 TB** across
  1873 trees. Top trees:

| tree | task | status | files | GB | cited? |
|---|---|---|---|---|---|
| `adapters/issue_397` | #397 | completed | 7,668 | 219.6 | yes |
| `adapters/issue_628` | #628 | completed | 4,432 | 36.2 | yes |
| `adapters/issue543` | #543 | completed | 2,325 | 19.9 | yes |
| `adapters/issue570` | #570 | completed | 2,313 | 11.5 | yes |
| `adapters/issue_608` | #608 | completed | 1,704 | 48.8 | yes |
| `adapters/issue_622` | #622 | completed | 1,434 | 144.3 | yes |
| `issue_490` | #490 | completed | 1,320 | 6.8 | yes |
| `adapters/issue_601` | #601 | completed | 1,070 | 92.8 | yes |
| `issue_478` | #478 | completed | 1,012 | 5.2 | yes |
| `adapters/issue_613` | #613 | completed | 800 | 77.7 | yes |

- **(c) Future-ladder sharding** — frees 0 now. tar per rung on FUTURE runs: #1090's c5 rescue was 107 files for ONE cell (checkpoints 2-15 + final, ~7-8 files/rung; the task body's 348 files for 3 cells is the same ~110-116/cell rate) — tarring gives ~1 file/rung, ~7-8x fewer files per cell.
- **(d) Successor layout going forward** — frees 0 now; removes growth.
  (d1) one model repo per task; consumer cost: repo id threaded per task;
  (d2) ONE shared successor repo superkaiba1/explore-persona-space-adapters; consumer cost: a single constant change.
  Zero-deletion parallel track: zero-deletion parallel track: HF-support limit-raise request (forum thread 26400 is exactly this ask).

## Verifying that deletion frees headroom (DISCRIMINATING recipe)

The 100k limit is believed to count the post-push HEAD tree (the rejection
says "would contain N files after this push"), but this is unverified from
docs. The NAIVE probes misfire: with 57 files of live headroom, any
<=57-file probe succeeds under BOTH tree and history semantics, and
the 107-file c5 re-push after a one-rung deletion fails
under BOTH (falsely reading as history-semantics). Discriminating recipe:

1. **PRIMARY — compare the server-quoted N across pushes straddling the
   deletion:** a rejected push still quotes "would contain N files"; N
   dropping by exactly the deleted count confirms tree semantics even on
   rejection (free, no probe sizing).
2. The c5 re-push (107 files) is discriminating only
   AFTER the first deletion batch frees
   >= 50 files
   (the c5 overage).
3. A generic probe push of size S discriminates when
   57 < S <= 57 + files_freed.

**On a GENUINE discriminating-probe failure** (evidence the limit counts git
HISTORY): file a follow-up task and pivot the recommendation to options
(c)/(d) + `super_squash_history` consultation — this audit completes before
any deletion, so the re-plan lives in that follow-up. (Note
`super_squash_history` squashes commits/LFS history; the HEAD-tree file COUNT
is unchanged by it — it is a commit-count/storage remedy, never a file-count
one.)

## Footnotes

- **Data-repo non-uniformity (future risk, out of scope):** the ~1M-file data
  repo (`superkaiba1/explore-persona-space-data`) still accepts pushes, so
  file-count enforcement is not uniform across repos (grandfathering or
  dataset exemption — unknown).
- `cited_by` includes the owning task's own body/plans (conservative:
  self-citations count); every cited rung is EXCLUDED from the ready-to-paste
  set and parked in the UNSAFE-cited manual-review section of
  `freeing_commands.md`.
- Fleet unblock (independent of this triage): rejected model-repo uploads now
  fall back to the private overflow repo by default
  (`EPM_HF_FILECOUNT_FALLBACK`, #1108) — a TEMPORARY durability fallback, not
  a successor layout.
