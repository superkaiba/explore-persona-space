# HF model repo file-count audit — issue #1108

**Repo:** `superkaiba1/explore-persona-space` — **117,050 / 100,000 git files**
(-17050 files of headroom) at revision `8131d89e29582cde0a77c9b520d7bd506ddcd688`
(`pre_deletion_revision` — deleted files remain fetchable at pre-deletion
revisions via `revision=` pinning: HF deletion frees the HEAD tree, not
history, so a mistakenly pruned rung is recoverable without retraining and a
future planner's reuse fitness check can resolve it). Enumerated
2026-07-18T02:27:59Z; total blob size 3.51 TB. Overflow
repo `superkaiba1/explore-persona-space-overflow` currently holds 3,880
files (separate per-repo budget).

## Safety invariant

Only artifacts of TERMINAL tasks (status completed/archived) are proposed;
final adapters are never in option (a)'s command set; deletion is
USER-EXECUTED; nothing in this audit (or the script that produced it) deletes
from HF.

## Attribution coverage

- task-resolved files: **109,548**
- named legacy-prefix files (task=None — structurally CANNOT enter
  terminal-task deletion commands): **7,502**
- unattributed: **0**
- conservation identity: task + named + unattributed == n_files (asserted in code)
- coverage (task + named) / n_files = **100.0%** (gate: >=95%)

Prefix-only attribution is never deletion-actionable: only TASK-resolved files
of TERMINAL tasks enter any command set.

## Ladder split

**69,745 / 117,050 files (59.6%)
sit inside `checkpoint-*/` dirs** (5,804 rung dirs,
1.76 TB). Footnote: rung-per-top-level-prefix layouts (e.g. issue466_*_step1600, i398_*_step_checkpoints) do not match /checkpoint-\d+/, so this figure and option (a) UNDERCOUNT true ladder residue; directionally safe.

Ladder mass by owning-task status (only completed/archived tasks enter the
deletion candidate set — a large share sits at `awaiting_promotion` and
becomes prunable only at promotion):
- awaiting_promotion: 38,273 ladder files across 35 tasks
- completed: 18,592 ladder files across 71 tasks
- followups_running: 8,640 ladder files across 1 tasks
- archived: 720 ladder files across 10 tasks

Top tasks by file count:

| task | status | classification | files | ladder files | rungs | GB |
|---|---|---|---|---|---|---|
| #597 | awaiting_promotion | useful | 12,312 | 11,916 | 993 | 347.7 |
| #1434 | followups_running | - | 8,904 | 8,640 | 720 | 252.2 |
| #397 | completed | useful | 7,668 | 6,480 | 540 | 219.6 |
| #621 | awaiting_promotion | - | 7,039 | 6,588 | 549 | 10.4 |
| #1090 | awaiting_promotion | - | 4,954 | 4,668 | 389 | 201.0 |
| #628 | completed | useful | 4,432 | 3,816 | 318 | 36.2 |
| #474 | completed | useful | 4,034 | 1,575 | 105 | 187.7 |
| #1333 | awaiting_promotion | - | 3,468 | 3,408 | 284 | 16.3 |
| #545 | awaiting_promotion | - | 3,048 | 2,349 | 156 | 296.2 |
| #537 | awaiting_promotion | - | 2,432 | 840 | 56 | 59.2 |

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
docs. The NAIVE probes misfire: with -17050 files of live headroom, any
<=-17050-file probe succeeds under BOTH tree and history semantics, and
the 107-file c5 re-push after a one-rung deletion fails
under BOTH (falsely reading as history-semantics). Discriminating recipe:

1. **PRIMARY — compare the server-quoted N across pushes straddling the
   deletion:** a rejected push still quotes "would contain N files"; N
   dropping by exactly the deleted count confirms tree semantics even on
   rejection (free, no probe sizing).
2. The c5 re-push (107 files) is discriminating only
   AFTER the first deletion batch frees
   >= 17157 files
   (the c5 overage).
3. A generic probe push of size S discriminates when
   -17050 < S <= -17050 + files_freed.

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

## Limit-status evidence (softened premise — #1141)

- Run-time file count: **117,050** (independent floor anchor
  110,000 passed; delta vs the 2026-07-18 anchor:
  +0).
- Commit scan since 2026-07-06: **511 non-probe upload commits**
  (LOWER BOUND — title-classifier based (custom-titled uploads land in n_other, reported beside it); n_other = 0 beside it;
  n_probe = 32, n_probe_cleanup = 32).
- First upload commit in the scan window (context only — may predate the
  rejection): `c77ce12262ceae9e91a440dedc616ffa1d079495` (2026-07-06T18:31:44Z) — "Upload folder using huggingface_hub".
- First post-rejection (day > 2026-07-07) non-probe upload commit:
  `d4dab015ff26a7e6f1f73d939fc3eb2ce1c02a80` (2026-07-08T00:49:35Z) — "Upload adapters/issue1112_s2_lora_pos_seed42/OVERFLOW_POINTER.json with huggingface_hub".
- Decisive fact (i): **492** post-rejection
  (day > 2026-07-07) commit(s) are FOLDER pushes (title
  "Upload folder using huggingface_hub"); 505 across the full scan window
  since 2026-07-06.
- Decisive fact (ii): net growth vs the 2026-07-07 rejection anchor
  (100,050 files): **+17,000** —
  count-increasing bulk pushes land.
- Status: the 100k file limit is **not enforced at the current count/shape as of 2026-07-18T02:27:59Z** (deliberately NOT read as a categorical "no longer enforced"; the #1108 fallback stays armed and `_is_file_count_limit_error` is retained).

## Options (softened premise — #1141 §4.6)

LFS accounting (named per row): every LFS figure is the HEAD-tree lfs_bytes sum; HF retains history-side LFS versions, so QUOTA reclaim from a HEAD deletion can differ from the HEAD-tree figure (super_squash_history is the history/storage remedy — it never reduces the tree file count).

| option | files | LFS GB (HEAD-tree) | consumer check | USER-ONLY steps |
|---|---|---|---|---|
| (a) do nothing | 0 freed | 0.0 | n/a | none |
| (b) migrate overflow -> canonical | 3,880 moved | 269.5 | n/a (additive copy; deletions USER-ONLY) | pointer + overflow deletion |
| (c1) archive-then-delete `adapters/issue_397` | 7,668 freed | 216.7 | cited: #1108 (plans/plan.md), #1108 (plans/v1.md), #1108 (plans/v2.md) (+16 more) | canonical delete |
| (c2) prune non-selected terminal rungs | 15,182 freed (selection-blind UPPER BOUND) | 337.2 | per-task cited_by rows below | every delete |

### (a) Do nothing

- Run-time count 117,050; net growth since the rejection
  +17,000 files; upload commits since the scan
  start: 511 (LOWER BOUND — title-classifier based (custom-titled uploads land in n_other, reported beside it);
  n_other = 0).
- Ongoing cost (not $0): permanent overflow growth is NOT free: overflow artifacts are PRIVATE (auth-required) and pointer-mediated — every future consumer pays the indirection; the #1108 fallback stays armed.

### (b) Migrate overflow -> canonical

- **3,880 files / 269.5 GB LFS
  (270.9 GB tree)**, SPLIT BY ERA:
  pre-#1108 (#564-era) 3,247 files /
  253.0 GB; post-#1108
  633 files / 18.0 GB
  (mechanism: commit-scan + pointer set-difference).
- destinations derive from the OVERFLOW PATHS themselves (both routing eras preserve path_in_repo); pointers are corroboration only; <=10k files/folder respected.
- Caveat: migration moves the measured private bytes into the PUBLIC repo's storage accounting (the #541/#552 quota surface) — weigh public-storage headroom; #564-era content was byte-quota-routed private ON PURPOSE (flagged via the era split).
- USER-ONLY: delete pointers + overflow contents after a VERIFIED migration; retire overflow to rescue-only (fallback stays armed).

```python
# per overflow prefix (repeat; path_in_repo is preserved by both routing eras):
from huggingface_hub import snapshot_download, upload_folder
local = snapshot_download("superkaiba1/explore-persona-space-overflow", allow_patterns=["<prefix>/**"], repo_type="model")
upload_folder(folder_path=f"{local}/<prefix>", path_in_repo="<prefix>",
              repo_id="superkaiba1/explore-persona-space", repo_type="model",
              commit_message="migrate overflow -> canonical (#1141, user-approved)")
# USER-ONLY after a VERIFIED migration:
# api.delete_file(path_in_repo="<prefix>/OVERFLOW_POINTER.json",
#                 repo_id="superkaiba1/explore-persona-space", repo_type="model")
# api.delete_folder(path_in_repo="<prefix>/", repo_id="superkaiba1/explore-persona-space-overflow", repo_type="model")
```

### (c1) Archive-then-delete `adapters/issue_397`

- **7,668 files / 216.7 GB LFS
  (219.6 GB tree)** — task #397
  (status: completed).
- Blast radius (cited_by over the durable-reference corpus):
  **cited: #1108 (plans/plan.md), #1108 (plans/v1.md), #1108 (plans/v2.md) (+16 more)**.
- archive-then-delete (wandb-archive precedent: superkaiba1/explore-persona-space-wandb-archive); the adapter archive repo is chosen by the USER at execution.
- USER-ONLY: delete adapters/issue_397 from canonical ONLY after the archive copy is VERIFIED.

```python
# 1) archive copy (additive, safe; wandb-archive precedent — USER picks the repo):
from huggingface_hub import snapshot_download, upload_folder
local = snapshot_download("superkaiba1/explore-persona-space", allow_patterns=["adapters/issue_397/**"], repo_type="model")
upload_folder(folder_path=f"{local}/adapters/issue_397", path_in_repo="adapters/issue_397",
              repo_id="<archive-repo>", repo_type="model",
              commit_message="archive adapters/issue_397 before canonical delete (#1141)")
# 2) USER-ONLY delete from canonical (run ONLY after the archive copy is VERIFIED):
# api.delete_folder(path_in_repo="adapters/issue_397/", repo_id="superkaiba1/explore-persona-space", repo_type="model")
```

### (c2) Prune non-selected ladder rungs of terminal tasks

- Totals are a **selection-blind UPPER BOUND**: 15,182 files /
  337.2 GB LFS
  (342.4 GB tree).
- Keep rule (pinned verbatim): "keep the single highest-step checkpoint-* dir per PARENT adapter dir, plus all non-checkpoint files". **Caveat:**
  "keep max-step" does NOT protect band-stop/dose-selected rungs — the project's canonical reuse pins EARLY/mid-band rungs (#532 reused #474's epoch-1 adapters), so the citation cross-check (cited_by), not the keep rule, is the actual protection.
- **USER must verify: verify selected checkpoint against the producing task's Reproducibility record before any (c2) delete.**
- Advisory trigger (prunable_rung_files >= 20% of repo files — ADVISORY only (the human reviews the draft; consistent with 'estimates, never gated on')): prunable =
  12.97% of repo files; fired =
  False. The recommendation MAY adopt
  archive-first for (c2) too.

| task | status | files | LFS GB | pruned rungs | cited_by |
|---|---|---|---|---|---|
| #112 | completed | 792 | 33.1 | 99 | uncited |
| #381 | completed | 324 | 9.0 | 27 | cited: #390 (plans/plan.md), #390 (plans/v1.md) |
| #385 | completed | 130 | 4.3 | 13 | uncited |
| #397 | completed | 5,184 | 144.5 | 432 | cited: #397 (plans/plan.md), #397 (plans/v1.md), #397 (plans/v2.md) (+2 more) |
| #398 | completed | 210 | 7.0 | 21 | uncited |
| #458 | completed | 286 | 7.4 | 22 | uncited |
| #466 | completed | 12 | 0.3 | 1 | uncited |
| #474 | completed | 1,260 | 82.4 | 84 | uncited |
| #516 | completed | 30 | 0.5 | 2 | uncited |
| #543 | completed | 1,770 | 15.7 | 118 | uncited |
| #570 | completed | 2,040 | 8.8 | 170 | uncited |
| #628 | completed | 3,144 | 24.2 | 262 | uncited |

Exact commands: `freeing_commands.md` (the existing ready-vs-unsafe split —
ready-to-paste = uncited blocks only; every cited block is COMMENTED OUT in
the UNSAFE section).

## Recommendation draft (mechanical; pre-registered #1141 §4.6 decision rule)

- Decision rule: IF >=1 post-2026-07-07 non-probe upload commit AND run-time count > 100k: recommend (b) [+ (c1) if adapters/issue_397 LFS >= 200 GB AND #397 terminal AND cited_by empty-or-user-cleared]; (c2) quantified-optional (advisory trigger only). ELSE: unfreeze-urgency — (c1)+(c2) sized to free >= max(0, run_time_count - 100_000 + 1_000); (b) deferred.
- Branch: **accepting** (n_upload_commits_post_rejection =
  497; run-time count
  117,050).
- (b) migrate overflow -> canonical (restores public single-repo access)
- (c1) conditional: adapters/issue_397 meets the LFS+terminal criteria but cited_by is non-empty — USER must clear/dismiss the citations first
- (c2): quantified-optional (selection-blind UPPER BOUND 15,182 files); advisory trigger only
- (c1) criteria: LFS 216.7 GB (>= 200 GB: True);
  #397 terminal: True; cited: #1108 (plans/plan.md), #1108 (plans/v1.md), #1108 (plans/v2.md) (+16 more).
- The final recommendation prose in the task body is composed by the session
  from this draft + the numbers; every irreversible step is listed as a
  command Thomas approves — NEVER executed here.
