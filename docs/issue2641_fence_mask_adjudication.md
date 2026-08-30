# Fence-mask adjudication — `verify_plan._fence_mask` CommonMark correction (#2641)

Adjudication of every verify_plan check whose verdict moves when the
delimiter-blind fence mask is replaced by the CommonMark 0.31.2 §4.5 walk
(plan v3 §4.3; acceptance criterion AC2). Companion enumeration data:
`docs/issue2641_fence_mask_adjudication.json` (the audit harness's own
output on the live tree; regenerate with
`uv run python scripts/issue2641_fence_mask_audit.py --kind-mode both
--no-timestamp --json docs/issue2641_fence_mask_adjudication.json`).

## Summary

**Production numbers (per-task `kind` from each task's `body.md` — the
operative set): the fix moves verdicts on 36 files across 15 checks and
flips ZERO plans' overall PASS/FAIL verdict.** Every move is a
row-level PASS/WARN/SKIP shift; 32 of the 36 production movers are
`kind: infra` plans, and the six production-only affected checks
(c15, c31, c34, c37, c41, c51) are all `infra|batch`-gated.

The widely-quoted **37 files / 19 checks / 9 overall flips** figure is the
**body-reproduction arm**: it forces `kind="experiment"` on every corpus
plan (the task body's reproduce-step convention) and is a superset a
reader must NOT carry forward as fleet impact — under production kind
resolution the same fix flips no plan's overall verdict at all.

Adjudication outcome, over the 25-check union of the two conventions:

- **Class A (new verdict correct, no check change): 20 checks** — of which
  6 carry hand-confirmed substantive reads (c1, c2, c7, c8, c31, c39) and
  14 are parse-mechanical (PASS↔SKIP / WARN↔SKIP visibility shifts).
- **Class B (new verdict correct, latent defect surfaced): 5 checks**
  (c22, c46, c52, c54, c63) — all five driven by the same latent
  limitation: the mask is line-level and container-blind, so a fence
  indented ≥4 spaces inside a list item (a REAL fence under full,
  container-aware CommonMark) is read as prose. Follow-up material for
  plan §4.6 item 4.
- **Class C (check needs adjusting): 0 checks.** The plan §3 prediction —
  every affected check lands in A or B — is CONFIRMED; kill criterion K1
  (>3 class C ⇒ re-plan) is nowhere near tripped.

## Reproduction provenance

- **Anchor commit (pinned corpus):**
  `5cb785f090e7866e1d227654e844e7489d7bb334` — plans + `body.md` (for
  production-kind resolution) materialized from the git object store via
  `--corpus-git-rev` (one `git ls-tree -r` + one `git cat-file --batch`
  stream), so both CONTENT and KIND are drift-free.
- **Audit repo SHA (code that ran):**
  `dcc083d405414856a8f991b0cd3e9ca0460a6364` (the `issue-2641` worktree
  base; `scripts/verify_plan.py` carrying the §4.1 mask +
  `scripts/issue2641_fence_mask_audit.py`).
- **Corpus of record:** the shell glob `tasks/*/*/plans/v*.md` —
  **4,757 files at the anchor commit** — which includes
  `tasks/completed/356/plans/v2-factcheck.md`. The task body's reproduce
  command greps `v[0-9]+\.md` (4,756 files), excluding exactly that file,
  which is itself a mask-changer AND a verdict mover (c22 WARN→PASS in
  both kind modes): the strict-regex corpus reads 152/36/35 where the
  glob-of-record reads 153/37/36. The plan's stated corpus size (4,761)
  was measured on the LIVE tree at plan time (2026-08-29), not the anchor
  tree; the anchor tree is 4 files smaller. S1's pinned count is
  evaluated on the glob-of-record.
- **S1 (pinned reproduction): 153 mask-changing files / 4,757 corpus
  files — exactly the task body's 153.** Kill criterion K2 satisfied.
- **S2 (forced `kind="experiment"`): 37 verdict-moving files / 19 checks /
  9 overall flips** (5 FAIL→PASS: `1176 v1/v2`, `714 v2/v5/v7`;
  4 PASS→FAIL: `1275 v1/v2`, `1558 v1/v2`) — reproduces the task body,
  including the two-directional c1/c2 PASS↔FAIL flips. The realized
  per-check transition histogram matches plan §2.3's table row-for-row.
- **S3 (production kind): 36 files / 15 checks / 0 overall flips** —
  matches plan §2.3 row-for-row (the plan flagged this arm as
  tree-dependent; at the anchor commit it reproduces exactly).
- **S4 (aggressive counterfactual):** the `_c75_strip_code_blocks`-style
  variant (additionally masking every 4-space/tab-indented line) changes
  **936 files on the pinned corpus** (plan-time figure: 937 on the
  larger live-glob corpus) — ~6.1× the conservative variant's blast
  radius, confirming the conservative choice (§11 R2).
- **Live-tree run (the committed `.json` sibling):** at repo tree
  `dcc083d4054`, 153 mask-changing files / 4,758 corpus files; stage-2
  numbers identical to the pinned run in both modes (37/19/9 and
  36/15/0).

## The five defect classes (corpus driver counts, over the 42 distinct movers)

| class | meaning | mover files driven (possibly jointly) |
|---|---|---|
| `closer-info-string` | a closing-candidate fence line carries an info string (` ```bash `); CommonMark refuses it as a closer, the blind mask toggled on it | 30 |
| `unclosed-at-eof` | under the CommonMark walk the file ends inside a fence; every following line is code | 19 |
| `indented-marker` | a fence-looking line indented ≥4 spaces; CommonMark (document-level) refuses recognition, the blind mask toggled | 10 |
| `info-string-backtick` | a backtick "opener" whose info string contains a backtick (prose ABOUT fences, four-backtick inline spans); not a fence opener under §4.5 | 5 |
| `inner-shorter-fence` | a shorter fence line inside a longer-delimited block; content under CommonMark, a closer under the blind mask | 2 |

Realized-vs-plan delta: plan §2.4 case 4 expected the inner-shorter-fence
class "not present as a standalone driver in the corpus movers"; realized,
`tasks/completed/596/plans/v1.md` + `v2.md` are exactly that (a
```` ````markdown ```` outer fence whose inner ` ```bash ` example the
blind mask read as a closer), and they drive c6's two moves. All five
defect classes therefore have live corpus drivers.

## Per-check adjudication (25 checks, ordered by combined mover count)

Verdict-class vocabulary is plan §4.3's: **A** = the check's contract,
applied to the CommonMark reading, yields the NEW verdict (substantive =
hand-confirmed against document content; parse-mechanical = a
PASS↔SKIP-shaped visibility shift answerable by "are the trigger words
inside a code block?"); **B** = as A plus a latent defect surfaced;
**C** = the contract yields the OLD verdict (fix the check). File paths
below are anchor-commit task paths; line numbers are 1-based.

### c8_success_kill_criteria — A (substantive)

- **Contract:** "Both a success-criteria family and a kill-criteria family
  must be present and non-empty in form (each carrier section ≥ 80 chars —
  emptiness check only; semantic joint-satisfiability stays with the
  Statistics critic per planner.md §7)."
- **Movers:** forced 21 files (FAIL→PASS ×3, PASS→FAIL ×5, PASS→WARN ×9,
  WARN→PASS ×4); production 12 files (PASS→WARN ×8, WARN→PASS ×4).
- **Defect classes:** info-string-backtick (714), unclosed-at-eof +
  closer-info-string (1548, 1558, 1716, 1860, 2201, 2312), indented-marker
  (1176, 406).
- **Adjudication.** FAIL→PASS on `tasks/completed/714/plans/v2.md`: line
  106 is `` ``  ```` ``` ```` or `~~~`; skip lines inside a fence. `` —
  prose about fence tokens using four-backtick inline code spans. The
  blind mask toggled into a phantom fence there, inverting lines 106-632
  and hiding `## 5. Success criteria (quantitative)` (line 377) and
  `## 6. Kill criteria` (line 394); under §4.5 the line is not an opener
  (backtick info string), both sections are visible, and PASS is what the
  contract yields — the blind mask was manufacturing the FAIL.
  PASS→FAIL on `tasks/completed/1558/plans/v1.md`: the closing candidate
  at line 71 (`   ```bash`, info string ⇒ not a closer) re-phases the
  walk; line 82 opens a fence that never closes, so lines 83-232 —
  including the success/kill sections — are code under CommonMark. GitHub
  renders the document the same way; the FAIL is correct GIVEN the
  malformed document, and the §4.4 stderr NOTE (unclosed fence opened at
  line 82) makes the hygiene defect visible. The PASS→WARN band
  (1860 v1-v5, 2201, 2312, 1548, 1716) is the same unclosed-swallow
  mechanism with one family surviving; class A with the plan's own
  honesty note: for the 19 unclosed-at-EOF files the class-A answer
  certifies the PARSE, not that the check should be blind to what the
  author intended as prose.

### c1_source_grounding — A (substantive)

- **Contract:** "every load-bearing hyperparameter carries a non-empty
  `Source:` (inline label or a `Source` table column), or the explicit
  `ungrounded — needs smoke-test` marker, or the section-level N/A.
  Presence-only: Source correctness / transfer stays fact-checker-owned."
- **Movers:** forced only, 20 files (FAIL→PASS ×11, FAIL→WARN ×1,
  PASS→FAIL ×5, PASS→WARN ×3). Production: none (c1 skips on
  `EXEMPT_KINDS`; the mover population is dominated by infra plans).
- **Defect classes:** all five except inner-shorter.
- **Adjudication.** FAIL→PASS on `714 v2/v5/v7`: the file carries 19
  `Source:` lines, ALL below the phantom fence opened at line 106 (first
  at line 464, `is a code-design knob; its `Source:` is the in-file
  pattern it mirrors.`); the blind mask hid every one of them. FAIL→PASS
  on `1176 v1`: the `*Source:*` decision-block labels at lines 281-283
  sit inside the region the blind walk phase-inverted from line 70 (the
  indented marker at line 69, `    ```bash / ~~~bash fence. …`, sits
  INSIDE the ```python fence opened at line 66 — the blind mask let it
  close that fence and never re-synced). PASS→FAIL on `1275/1548/2312`:
  the unclosed-swallow hides the §11 region — correct given the
  document, same reading as c8's PASS→FAIL band.

### c59_gpu_hours_token_conflict — A (parse-mechanical)

- **Contract:** "when a plan carries at least one DECLARATION-SHAPED
  `Estimated GPU-hours (total):` line … two arms compare (#2123): Arm A —
  more than one DISTINCT declaration-shaped value; Arm B — the value a
  FIRST-MATCH consumer reads differs from the first declaration-shaped
  value."
- **Movers:** 14 files in BOTH modes (PASS→SKIP ×12, SKIP→PASS ×2) —
  identical row sets (c59 runs on all kinds).
- **Defect classes:** unclosed-at-eof + closer-info-string (the 12
  PASS→SKIP: 1275, 1558, 1716, 1860 v1-v5); info-string-backtick (the 2
  SKIP→PASS: 1790 v1, 1847 v1).
- **Adjudication.** PASS→SKIP: the declaration line is swallowed by the
  unclosed fence (e.g. `1716 v1` — everything from line 314 to EOF is
  code), so the check finds no declaration-shaped line and correctly
  SKIPs. SKIP→PASS on `1790 v1`: line 73 (`  ```bash fence), substitute
  `<N>`→1790 …`) is prose whose backtick-bearing info string the blind
  mask took as an opener, swallowing the file's tail — including the
  declaration at line 110, `Estimated GPU-hours (total): 0`; under
  CommonMark the line is visible and consistent, so PASS. Same shape on
  `1847 v1` (line 38 four-backtick span; declaration at line 95).
  Trigger-visibility both ways: parse-mechanical.

### c41_regression_anchor_executed — A (parse-mechanical)

- **Contract:** "a plan-named regression-anchor / 'the Step-9c gate will
  run it' test must be either explicitly run by a pytest command in the
  plan, branch-new, or actually returned by the REAL Step-9c selection
  over the plan's declared touched files. NEVER FAILs (fail-open)."
- **Movers:** production only, 7 files (PASS→SKIP ×4: 1558 v1/v2,
  2312 v1/v2; SKIP→PASS ×3: 874 v1/v2/v3).
- **Defect classes:** unclosed-at-eof + closer-info-string (1558, 2312);
  closer-info-string + indented-marker (874 — the BLIND mask left 874
  unclosed; the CommonMark walk closes it).
- **Adjudication.** PASS→SKIP: anchors swallowed by the unclosed fence ⇒
  no anchor found ⇒ SKIP (fail-open by design). SKIP→PASS on
  `tasks/completed/874/plans/v1.md`: the nested block at lines 44-50
  (outer ``` pair wrapping an indented `   ```bash` marker fence at
  line 46) de-phased the BLIND walk into an unclosed tail-swallow, hiding
  the anchor vocabulary and its satisfier (the raw-plan pytest command at
  line 177, `uv run pytest tests/test_workflow_lint.py
  tests/test_issue_skill_marker_contract.py -q`); the CommonMark walk
  closes the file properly, the anchors and the satisfier are visible,
  and PASS is what satisfier rule (a) yields.

### c34_ratchet_headroom — A (parse-mechanical)

- **Contract:** "a fenced block whose preceding … non-fenced lines name a
  ratcheted path (`.claude/agents/*.md` / `.claude/rules/LESSONS.md`) plus
  an insertion verb is treated as a verbatim insert into that file; when
  the per-target summed block bytes exceed the file's live headroom … the
  check WARNs."
- **Movers:** production only, 6 files (PASS→SKIP ×6: 1460 v1/v2,
  1558 v1/v2, 998 v1/v2).
- **Defect classes:** closer-info-string (1460, 998); + unclosed-at-eof
  (1558).
- **Adjudication.** All six are fence-BOUNDARY shifts: e.g.
  `1460 v1` line 84 (`   ```bash` — a legal 3-space-indented opener whose
  later closing candidate carries an info string) extends the fenced
  region under CommonMark, dissolving the (non-fenced-lines + fenced
  block) trigger geometry the check keys on ⇒ SKIP. Trigger words inside
  a code block: parse-mechanical.

### c51_edited_literal_pin_tests — A (parse-mechanical)

- **Contract:** "when a plan declares an edit to an EXISTING
  workflow-surface literal … every `tests/` file that already pins that
  literal verbatim must be named somewhere in the raw plan … an unlisted
  pin makes the plan's own Step 9c exit-0 acceptance criterion
  deterministically unsatisfiable as scoped."
- **Movers:** production only, 6 files (PASS→SKIP ×3: 1558 v1/v2,
  2312 v2; SKIP→PASS ×3: 1176 v1/v2, 998 v1).
- **Defect classes:** unclosed-at-eof + closer-info-string (1558, 2312);
  indented-marker (1176); closer-info-string (998).
- **Adjudication.** SKIP→PASS on `1176 v1`: the workflow-surface edit
  commitment (line 198, `**`.claude/skills/refactor/SKILL.md:167` —
  FIX.**`) becomes visible under the CommonMark walk, arming the trigger,
  and the raw plan names `tests/test_workflow_lint.py` (lines 28, 219,
  271) — the pin-listing satisfier — so PASS. PASS→SKIP on the unclosed
  files: trigger swallowed.

### c39_off_pod_phase_declaration — A (substantive)

- **Contract:** "a plan whose non-fenced prose names an off-pod / VM-side
  phase must either carry the fenced `off_pod_phases:` declaration block …
  or declare the standalone escape `N/A — no off-pod phase`."
- **Movers:** forced only, 5 files (PASS→SKIP ×2: 1558 v1/v2; SKIP→WARN
  ×3: 714 v2/v5/v7).
- **Defect classes:** unclosed-at-eof (1558); info-string-backtick (714).
- **Adjudication.** SKIP→WARN on `714 v2`: the newly-visible prose at
  line 429 (`long CPU phase, no data footprint, no off-pod routing
  decision.`) carries off-pod vocabulary on a non-fenced line; the file
  has no `off_pod_phases:` block and no verbatim standalone escape
  literal, so the contract yields WARN. Hand-confirmed: the mention is a
  negative statement ("no off-pod routing decision"), i.e. a
  vocabulary-heuristic surface WARN of exactly the kind the check's own
  docstring tolerates at WARN-only granularity (critics adjudicate).
  Class A substantive — the verdict is contract-correct; the residual
  question is the check's designed WARN posture, not a mask artifact.

### c19_ood_folds — A (parse-mechanical)

- **Contract:** "a held-out predictive DV … over group-structured samples
  must register a GROUP-level fold (LOFO / corpus transfer), declare
  `N/A — no held-out predictive DV`, or argue a genuinely iid sample.
  NEVER FAILs — the trigger is a vocabulary heuristic."
- **Movers:** forced only, 4 files (PASS→SKIP ×4: 1275 v1/v2, 1558 v1/v2).
- **Defect classes:** unclosed-at-eof + closer-info-string.
- **Adjudication.** Trigger vocabulary swallowed by the unclosed fence
  (e.g. `1275 v1`: lines 100-237 are code under CommonMark — the fence
  opened at line 100 never closes) ⇒ SKIP. Parse-mechanical.

### c2_measurement_validity — A (substantive)

- **Contract:** "planner.md §6 required block: per dependent variable, the
  construct, the metric, and the on-distribution status. FAIL only when
  ALL signals are absent; a bare heading without construct/metric content
  is a WARN."
- **Movers:** forced only, 9 files (FAIL→PASS ×3: 714 v2/v5/v7; FAIL→WARN
  ×2: 1176 v1/v2; PASS→FAIL ×4: 1275 v1/v2, 1558 v1/v2).
- **Defect classes:** info-string-backtick (714), indented-marker (1176),
  unclosed-at-eof (1275, 1558).
- **Adjudication.** FAIL→PASS on `714 v2`: the measurement-validity
  content below line 106 was invisible under the blind mask (same phantom
  fence as c1/c8); visible under §4.5 ⇒ PASS. FAIL→WARN on `1176 v1`: the
  phase-inverted region partially restores §6 content — heading visible,
  full construct/metric content mixed ⇒ the contract's WARN tier.
  PASS→FAIL on the unclosed files: the §6 block is genuinely unreadable
  in the CommonMark rendering ⇒ FAIL correct given the document (the
  §4.4 NOTE names the opener line).

### c22_cross_section_param_consistency — B (substantive; container-blindness surfaced)

- **Contract:** "The same tracked hyperparameter stated with contradictory
  values in DIFFERENT top-level sections … A conflict is a pair of
  top-level sections whose value SETS are disjoint."
- **Movers:** 2 files in BOTH modes (PASS→WARN: 763 v2; WARN→PASS:
  356 v2-factcheck) — identical rows (c22 runs on all kinds).
- **Defect classes:** indented-marker (both).
- **Adjudication.** PASS→WARN on
  `tasks/awaiting_promotion/763/plans/v2.md`: line 144 is
  `     ```python` — a pseudocode fence indented 5 spaces inside a
  numbered-list item. The document-level §4.5 rule refuses it (≥4-space
  indent), so lines 144-162 read as PROSE, exposing
  `temperature=1.0` (line 154, inside the pseudocode) to the tracked-param
  scan against another top-level section's `temperature:0.0` (line 428,
  the gen-cell recipe) — disjoint sets ⇒ WARN, exactly what the contract
  yields on the document-level reading. WARN→PASS on
  `356 v2-factcheck` (line 17, `      ```python`) is the same mechanism
  in reverse: text the blind walk hid becomes visible and the sets
  overlap. **Class B:** under full container-aware CommonMark a fence
  indented to a list item's content indent IS a real fence (plan §2.2:
  "inside a list item the indentation baseline shifts"), so the 763 WARN
  scans what the author wrote as code. The verdicts are correct under the
  implemented line-level contract — the plan chooses that scope
  deliberately (§11 R2) — and the row surfaces the container-blindness
  limitation as follow-up material (§4.6 item 4). Not class C: no change
  to c22 would fix this; the residual lives in the shared mask's scope.

### c12_battery_multiplier — A (parse-mechanical)

- **Contract:** "A plan naming a permutation/bootstrap/null-draw battery —
  or a pool-quadratic candidate SCREEN — must carry, NEAR a trigger
  mention (±15 raw lines), BOTH class-matched sizing arithmetic and a
  batched-implementation commitment."
- **Movers:** forced only, 3 files (PASS→SKIP ×3: 874 v1/v2/v3).
- **Defect classes:** closer-info-string + indented-marker.
- **Adjudication.** The nested-fence region at lines 44-50 of `874 v1`
  de-phases the blind walk into a tail-swallowing unclosed state; under
  CommonMark the fencing closes properly and the battery-vocabulary
  trigger geometry (trigger ± evidence windows) dissolves into fenced
  regions ⇒ SKIP. Battery words inside code blocks: parse-mechanical.

### c4_contrastive_negatives — A (parse-mechanical)

- **Contract:** "Behavior-implantation plans must name a
  contrastive-negative set or one of the two named exemptions … WARN not
  FAIL: the trigger is a content heuristic."
- **Movers:** forced only, 3 files (PASS→SKIP ×2: 1558 v1/v2; WARN→SKIP
  ×1: 787 v1).
- **Defect classes:** unclosed-at-eof (1558); closer-info-string (787).
- **Adjudication.** `787 v1`: the fence opened at line 156 (`   ```bash`)
  extends further under CommonMark (its blind "closer" carries an info
  string), masking lines 157-246 — where the implantation-vocabulary
  trigger lived ⇒ SKIP. Trigger words inside a code block:
  parse-mechanical, and the removed WARN was scanning command text.

### c37_noflags_bundling_claim — A (parse-mechanical)

- **Contract:** "a plan line asserting a `--check-<flag>` is bundled into
  workflow_lint.py's no-flags default run must name a flag actually
  dispatched there … Trigger: `--check-<flag>` + a claim-verb-anchored
  no-flags assertion on one non-fenced line."
- **Movers:** production only, 3 files (WARN→SKIP ×3: 1149 v1,
  1275 v1/v2).
- **Defect classes:** closer-info-string (1149); + unclosed-at-eof (1275).
- **Adjudication.** `1149 v1`: the 3-space-indented `   ```bash` fence at
  line 174 wraps a heredoc'd python block whose blind "closer" carries an
  info string; under CommonMark the block extends over the line carrying
  the `--check-*` bundling claim, which is therefore fenced ⇒ trigger
  gone ⇒ SKIP. The removed WARN was reading a command example as a prose
  claim: parse-mechanical.

### c7_replication_fidelity — A (substantive)

- **Contract:** "When the Goal mentions replicating, the plan must address
  replication fidelity (match the paper's data + recipe first …). WARN
  because 'does the effect replicate across seeds' is a benign false
  trigger."
- **Movers:** forced 2 (WARN→PASS: 1176 v2, 406 v7); production 1
  (WARN→PASS: 406 v7).
- **Defect classes:** indented-marker (1176); + closer-info-string (406).
- **Adjudication.** Hand-run `--explain --check c7` on
  `tasks/completed/406/plans/v7.md`: old (blind) = WARN "Goal mentions
  replication but no fidelity vocabulary"; new (CommonMark) = PASS
  "replication-fidelity vocabulary present (paper recipe / deviations
  addressed)". The blind walk's phase inversion from line 272 (an
  8-space-indented `        ```python` inside a list) hid the prose
  regions carrying the fidelity vocabulary; §4.5 exposes them. (The
  file's line 55 escape `` `N/A — not a replication` `` is
  backtick-wrapped and satisfies under NEITHER mask — the check demands
  it unwrapped — so the move is vocabulary visibility, not the escape.)

### c42_commit_sha_resolves — A (parse-mechanical)

- **Contract:** "Every hex token the plan cites AS A COMMIT … must resolve
  under `git rev-parse --verify --quiet '<sha>^{commit}'`."
- **Movers:** 2 files in BOTH modes (PASS→SKIP ×2: 2312 v1/v2).
- **Defect classes:** unclosed-at-eof + closer-info-string.
- **Adjudication.** `2312 v1`: lines 300-501 flip; the fence opened at
  line 357 never closes, swallowing the tail's commit-citation region ⇒
  no cited-SHA trigger ⇒ SKIP. (Plan §2.3 flagged c42's rows as
  tree-dependent — SHA resolvability depends on the audit repo's object
  store; at both the anchor and the live tree the rows reproduce.)
  Parse-mechanical.

### c15_failloud_test_coverage — A (parse-mechanical)

- **Contract:** "`kind: infra|batch` plans whose acceptance/success
  criteria assert fail-loud / no-silent-swallow behavior must name a
  committed test pinning it."
- **Movers:** production only, 2 files (PASS→SKIP ×2: 1558 v1/v2).
- **Defect classes:** unclosed-at-eof + closer-info-string.
- **Adjudication.** The fail-loud vocabulary in `1558 v1` sits below the
  unclosed opener at line 82; under CommonMark it is code ⇒ SKIP.
  Parse-mechanical (same document as c8's hand-worked PASS→FAIL; the
  §4.4 NOTE is the visibility mitigation).

### c27_capture_intent_hbm — A (parse-mechanical)

- **Contract:** "activation-capture vocabulary + a >=7B model signal
  while an eval/debug (L4) intent is booked on the GCP/auto lane."
- **Movers:** forced only, 2 files (PASS→SKIP ×2: 1558 v1/v2).
- **Defect classes:** unclosed-at-eof + closer-info-string.
- **Adjudication.** Same 1558 swallow: the vocabulary trigger is fenced ⇒
  SKIP. Parse-mechanical.

### c30_realized_keys — A (parse-mechanical)

- **Contract:** "Plans reusing a multi-field tensor bundle must name a
  realized-keys verification (artifact-reuse.md check (c), incident
  #1073)."
- **Movers:** forced only, 2 files (PASS→SKIP ×2: 1558 v1/v2).
- **Defect classes:** unclosed-at-eof + closer-info-string.
- **Adjudication.** Same 1558 swallow ⇒ SKIP. Parse-mechanical.

### c33_ladder_retention — A (parse-mechanical)

- **Contract:** "a plan carrying checkpoint-ladder vocabulary on a
  non-fenced line … must carry retention vocabulary … within its
  compute-sizing section(s)."
- **Movers:** forced only, 2 files (PASS→SKIP ×2: 1558 v1/v2).
- **Defect classes:** unclosed-at-eof + closer-info-string.
- **Adjudication.** Same 1558 swallow ⇒ SKIP. Parse-mechanical.

### c6_reuse_fitness — A (parse-mechanical)

- **Contract:** "Plans reusing trained HF artifacts must carry the fitness
  attestations (a)-(n) … WARN not FAIL: trigger and item-detection are
  both heuristic."
- **Movers:** forced only, 2 files (WARN→SKIP ×2: 596 v1/v2).
- **Defect classes:** inner-shorter-fence — the corpus's only standalone
  drivers of this class.
- **Adjudication.** `596 v1` lines 160-167: a ```` ````markdown ````
  four-backtick fence wraps a worked ` ```bash ` example (line 161). The
  blind mask read the inner three-backtick line as a CLOSER (shorter
  than the opener — illegal under §4.5), exposing the example's shell
  text as prose, where reuse vocabulary triggered the WARN. Under
  CommonMark the whole example is content of the four-backtick block ⇒
  trigger gone ⇒ SKIP. Trigger words inside a code block:
  parse-mechanical; the removed WARN was a mask artifact.

### c31_skillmd_prose_pin — A (substantive; carries the plan's class-B follow-up)

- **Contract:** "`kind: infra|batch` plans that commit to editing
  `.claude/skills/**/SKILL.md` prose must carry ONE labeled line naming a
  durability pin test … or a one-line no-pin justification."
- **Movers:** production only, 4 files (SKIP→WARN ×2: 1176 v1/v2;
  WARN→SKIP ×1: 1860 v1; PASS→SKIP ×1: 2312 v2).
- **Defect classes:** indented-marker (1176); unclosed-at-eof (1860,
  2312).
- **Adjudication.** SKIP→WARN on `1176 v1` — the plan's §2.4 case-2
  worked read, re-confirmed here by hand: `grep -i "durability pin"`
  returns nothing, and the plan commits to editing
  `.claude/skills/refactor/SKILL.md` (line 198 "FIX", line 271 output
  destinations, line 312 files-touched list). The WARN is substantively
  right — the blind mask was hiding a real un-pinned prose edit. The
  WARN→SKIP / PASS→SKIP rows are unclosed-swallow visibility shifts
  (parse-mechanical). Class-B follow-up per plan §4.3 pre-classification:
  the surfaced un-pinned edit is follow-up material, not a check defect.

### c46_dispatch_cmd_cli_parse — B (substantive; container-blindness surfaced)

- **Contract:** "every plan-embedded `dispatch_issue.py` command (fenced
  code blocks + inline-code spans; backslash continuations joined) must
  dry-parse against the CLI's REAL argparser, and a launch-shaped command
  must not carry the three demonstrated drift shapes."
- **Movers:** 1 file in BOTH modes (PASS→SKIP:
  `tasks/awaiting_promotion/1090/plans/v5.md`).
- **Defect classes:** indented-marker.
- **Adjudication.** Lines 619-627 of `1090 v5`: a `dispatch_issue.py
  launch` command sits in a fence whose markers (line 619 and its closer)
  are indented 4 spaces as list-item continuation. The blind mask
  (indent-blind) treated them as fences, so the command was a fenced-block
  candidate and dry-parsed ⇒ PASS. The document-level §4.5 rule refuses
  ≥4-space fence markers, so the command text is plain prose — neither a
  fenced block nor an inline span ⇒ no candidate ⇒ SKIP, which is what
  the implemented contract yields. **Class B:** under container-aware
  CommonMark the fence is real (list-item content indent), the command IS
  fenced, and the OLD verdict's candidate extraction was the more
  faithful reading — the row is the sharpest surfaced instance of the
  mask's deliberate container-blindness (plan §2.2/§11 R2). Follow-up
  material (§4.6 item 4); not class C (the check reads whatever the
  shared mask gives it — the residual is the mask's scope, and narrowing
  it here would contradict K3's more-CommonMark-correct direction at the
  document level).

### c52_fanout_ram_floor — B (parse-mechanical; same driver as c46)

- **Contract:** "EVERY plan-embedded launch-shaped `dispatch_issue.py`
  argv is checked against the plan's own declared per-leg peaks — a
  declared host-RAM peak strictly above … requires `--min-ram-gb` …"
- **Movers:** 1 file in BOTH modes (PASS→SKIP: 1090 v5).
- **Defect classes / adjudication:** identical driver to c46 (the same
  list-indented launch argv at lines 621-625 stops being a candidate) ⇒
  SKIP under the implemented contract. Class B for the same
  container-blindness reason; parse-mechanical in form (candidate
  visibility).

### c54_workload_cmd_lane_env — B (parse-mechanical; same driver as c46)

- **Contract:** "the `--workload-cmd` value of every plan-embedded
  `dispatch_issue.py` command … must not reference a lane-specific env
  var BARE."
- **Movers:** 1 file in BOTH modes (PASS→SKIP: 1090 v5).
- **Adjudication:** same driver and reasoning as c46/c52. Class B,
  parse-mechanical in form.

### c63_declared_width_vs_launch — B (parse-mechanical; same driver as c46)

- **Contract:** "when the §9 window declares an N-GPU spec … at least one
  plan-embedded launch-shaped `dispatch_issue.py` argv … must REALIZE a
  width >= N_decl."
- **Movers:** 1 file in BOTH modes (PASS→SKIP: 1090 v5).
- **Adjudication:** same driver and reasoning as c46/c52/c54. Class B,
  parse-mechanical in form.

## Unclosed-fence census (§4.4's motivation)

Under the CommonMark mask **21** corpus plans end inside an unclosed
fence; under the blind mask 13 do; **19** are unclosed only under the new
mask, and those 19 are **19 of the 37** forced-mode verdict movers
(`1275 v1/v2`, `1548 v1/v2`, `1558 v1/v2`, `1716 v1/v2/v3`,
`1860 v1-v5`, `2201 v1-v3`, `2312 v1/v2`). Their openers swallow
**2,658** tail lines counted exclusive of the opener line (2,677
inclusive) — the single largest driver of the verdict moves. The §4.4
stderr NOTE (`verify_plan: NOTE — unclosed code fence opened at line N`)
fires on exactly these documents at every future verification, making the
silent-truncation hazard visible without a registered-check edit.

## Enumeration — production `kind` (operative set, 63 rows)

| file | check | old | new | unclosed at EOF (new mask) |
|---|---|---|---|---|
| `tasks/completed/1558/plans/v1.md` | c15_failloud_test_coverage | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c15_failloud_test_coverage | PASS | SKIP | yes |
| `tasks/awaiting_promotion/763/plans/v2.md` | c22_cross_section_param_consistency | PASS | WARN | no |
| `tasks/completed/356/plans/v2-factcheck.md` | c22_cross_section_param_consistency | WARN | PASS | no |
| `tasks/completed/1176/plans/v1.md` | c31_skillmd_prose_pin | SKIP | WARN | no |
| `tasks/completed/1176/plans/v2.md` | c31_skillmd_prose_pin | SKIP | WARN | no |
| `tasks/completed/1860/plans/v1.md` | c31_skillmd_prose_pin | WARN | SKIP | yes |
| `tasks/completed/2312/plans/v2.md` | c31_skillmd_prose_pin | PASS | SKIP | yes |
| `tasks/completed/1460/plans/v1.md` | c34_ratchet_headroom | PASS | SKIP | no |
| `tasks/completed/1460/plans/v2.md` | c34_ratchet_headroom | PASS | SKIP | no |
| `tasks/completed/1558/plans/v1.md` | c34_ratchet_headroom | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c34_ratchet_headroom | PASS | SKIP | yes |
| `tasks/completed/998/plans/v1.md` | c34_ratchet_headroom | PASS | SKIP | no |
| `tasks/completed/998/plans/v2.md` | c34_ratchet_headroom | PASS | SKIP | no |
| `tasks/completed/1149/plans/v1.md` | c37_noflags_bundling_claim | WARN | SKIP | no |
| `tasks/completed/1275/plans/v1.md` | c37_noflags_bundling_claim | WARN | SKIP | yes |
| `tasks/completed/1275/plans/v2.md` | c37_noflags_bundling_claim | WARN | SKIP | yes |
| `tasks/completed/1558/plans/v1.md` | c41_regression_anchor_executed | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c41_regression_anchor_executed | PASS | SKIP | yes |
| `tasks/completed/2312/plans/v1.md` | c41_regression_anchor_executed | PASS | SKIP | yes |
| `tasks/completed/2312/plans/v2.md` | c41_regression_anchor_executed | PASS | SKIP | yes |
| `tasks/completed/874/plans/v1.md` | c41_regression_anchor_executed | SKIP | PASS | no |
| `tasks/completed/874/plans/v2.md` | c41_regression_anchor_executed | SKIP | PASS | no |
| `tasks/completed/874/plans/v3.md` | c41_regression_anchor_executed | SKIP | PASS | no |
| `tasks/completed/2312/plans/v1.md` | c42_commit_sha_resolves | PASS | SKIP | yes |
| `tasks/completed/2312/plans/v2.md` | c42_commit_sha_resolves | PASS | SKIP | yes |
| `tasks/awaiting_promotion/1090/plans/v5.md` | c46_dispatch_cmd_cli_parse | PASS | SKIP | no |
| `tasks/completed/1176/plans/v1.md` | c51_edited_literal_pin_tests | SKIP | PASS | no |
| `tasks/completed/1176/plans/v2.md` | c51_edited_literal_pin_tests | SKIP | PASS | no |
| `tasks/completed/1558/plans/v1.md` | c51_edited_literal_pin_tests | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c51_edited_literal_pin_tests | PASS | SKIP | yes |
| `tasks/completed/2312/plans/v2.md` | c51_edited_literal_pin_tests | PASS | SKIP | yes |
| `tasks/completed/998/plans/v1.md` | c51_edited_literal_pin_tests | SKIP | PASS | no |
| `tasks/awaiting_promotion/1090/plans/v5.md` | c52_fanout_ram_floor | PASS | SKIP | no |
| `tasks/awaiting_promotion/1090/plans/v5.md` | c54_workload_cmd_lane_env | PASS | SKIP | no |
| `tasks/completed/1275/plans/v1.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1275/plans/v2.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v1.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1716/plans/v1.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1716/plans/v2.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1716/plans/v3.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1790/plans/v1.md` | c59_gpu_hours_token_conflict | SKIP | PASS | no |
| `tasks/completed/1847/plans/v1.md` | c59_gpu_hours_token_conflict | SKIP | PASS | no |
| `tasks/completed/1860/plans/v1.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1860/plans/v2.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1860/plans/v3.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1860/plans/v4.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1860/plans/v5.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/awaiting_promotion/1090/plans/v5.md` | c63_declared_width_vs_launch | PASS | SKIP | no |
| `tasks/completed/406/plans/v7.md` | c7_replication_fidelity | WARN | PASS | no |
| `tasks/completed/1558/plans/v1.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1558/plans/v2.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1716/plans/v1.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1716/plans/v2.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1716/plans/v3.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/2201/plans/v1.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/2201/plans/v2.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/2201/plans/v3.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/406/plans/v7.md` | c8_success_kill_criteria | WARN | PASS | no |
| `tasks/completed/714/plans/v2.md` | c8_success_kill_criteria | WARN | PASS | no |
| `tasks/completed/714/plans/v5.md` | c8_success_kill_criteria | WARN | PASS | no |
| `tasks/completed/714/plans/v7.md` | c8_success_kill_criteria | WARN | PASS | no |

## Enumeration — forced `kind="experiment"` (body-reproduction arm, 97 rows)

| file | check | old | new | unclosed at EOF (new mask) |
|---|---|---|---|---|
| `tasks/completed/874/plans/v1.md` | c12_battery_multiplier | PASS | SKIP | no |
| `tasks/completed/874/plans/v2.md` | c12_battery_multiplier | PASS | SKIP | no |
| `tasks/completed/874/plans/v3.md` | c12_battery_multiplier | PASS | SKIP | no |
| `tasks/completed/1275/plans/v1.md` | c19_ood_folds | PASS | SKIP | yes |
| `tasks/completed/1275/plans/v2.md` | c19_ood_folds | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v1.md` | c19_ood_folds | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c19_ood_folds | PASS | SKIP | yes |
| `tasks/completed/1176/plans/v1.md` | c1_source_grounding | FAIL | PASS | no |
| `tasks/completed/1176/plans/v2.md` | c1_source_grounding | FAIL | PASS | no |
| `tasks/completed/1275/plans/v1.md` | c1_source_grounding | PASS | FAIL | yes |
| `tasks/completed/1275/plans/v2.md` | c1_source_grounding | PASS | FAIL | yes |
| `tasks/completed/1548/plans/v1.md` | c1_source_grounding | PASS | FAIL | yes |
| `tasks/completed/1548/plans/v2.md` | c1_source_grounding | PASS | WARN | yes |
| `tasks/completed/1558/plans/v1.md` | c1_source_grounding | PASS | WARN | yes |
| `tasks/completed/1558/plans/v2.md` | c1_source_grounding | PASS | WARN | yes |
| `tasks/completed/1716/plans/v1.md` | c1_source_grounding | FAIL | PASS | yes |
| `tasks/completed/1716/plans/v2.md` | c1_source_grounding | FAIL | PASS | yes |
| `tasks/completed/1716/plans/v3.md` | c1_source_grounding | FAIL | PASS | yes |
| `tasks/completed/1987/plans/v1.md` | c1_source_grounding | FAIL | WARN | no |
| `tasks/completed/2312/plans/v1.md` | c1_source_grounding | PASS | FAIL | yes |
| `tasks/completed/2312/plans/v2.md` | c1_source_grounding | PASS | FAIL | yes |
| `tasks/completed/714/plans/v2.md` | c1_source_grounding | FAIL | PASS | no |
| `tasks/completed/714/plans/v5.md` | c1_source_grounding | FAIL | PASS | no |
| `tasks/completed/714/plans/v7.md` | c1_source_grounding | FAIL | PASS | no |
| `tasks/completed/874/plans/v1.md` | c1_source_grounding | FAIL | PASS | no |
| `tasks/completed/874/plans/v2.md` | c1_source_grounding | FAIL | PASS | no |
| `tasks/completed/874/plans/v3.md` | c1_source_grounding | FAIL | PASS | no |
| `tasks/awaiting_promotion/763/plans/v2.md` | c22_cross_section_param_consistency | PASS | WARN | no |
| `tasks/completed/356/plans/v2-factcheck.md` | c22_cross_section_param_consistency | WARN | PASS | no |
| `tasks/completed/1558/plans/v1.md` | c27_capture_intent_hbm | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c27_capture_intent_hbm | PASS | SKIP | yes |
| `tasks/completed/1176/plans/v1.md` | c2_measurement_validity | FAIL | WARN | no |
| `tasks/completed/1176/plans/v2.md` | c2_measurement_validity | FAIL | WARN | no |
| `tasks/completed/1275/plans/v1.md` | c2_measurement_validity | PASS | FAIL | yes |
| `tasks/completed/1275/plans/v2.md` | c2_measurement_validity | PASS | FAIL | yes |
| `tasks/completed/1558/plans/v1.md` | c2_measurement_validity | PASS | FAIL | yes |
| `tasks/completed/1558/plans/v2.md` | c2_measurement_validity | PASS | FAIL | yes |
| `tasks/completed/714/plans/v2.md` | c2_measurement_validity | FAIL | PASS | no |
| `tasks/completed/714/plans/v5.md` | c2_measurement_validity | FAIL | PASS | no |
| `tasks/completed/714/plans/v7.md` | c2_measurement_validity | FAIL | PASS | no |
| `tasks/completed/1558/plans/v1.md` | c30_realized_keys | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c30_realized_keys | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v1.md` | c33_ladder_retention | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c33_ladder_retention | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v1.md` | c39_off_pod_phase_declaration | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c39_off_pod_phase_declaration | PASS | SKIP | yes |
| `tasks/completed/714/plans/v2.md` | c39_off_pod_phase_declaration | SKIP | WARN | no |
| `tasks/completed/714/plans/v5.md` | c39_off_pod_phase_declaration | SKIP | WARN | no |
| `tasks/completed/714/plans/v7.md` | c39_off_pod_phase_declaration | SKIP | WARN | no |
| `tasks/completed/2312/plans/v1.md` | c42_commit_sha_resolves | PASS | SKIP | yes |
| `tasks/completed/2312/plans/v2.md` | c42_commit_sha_resolves | PASS | SKIP | yes |
| `tasks/awaiting_promotion/1090/plans/v5.md` | c46_dispatch_cmd_cli_parse | PASS | SKIP | no |
| `tasks/completed/1558/plans/v1.md` | c4_contrastive_negatives | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c4_contrastive_negatives | PASS | SKIP | yes |
| `tasks/completed/787/plans/v1.md` | c4_contrastive_negatives | WARN | SKIP | no |
| `tasks/awaiting_promotion/1090/plans/v5.md` | c52_fanout_ram_floor | PASS | SKIP | no |
| `tasks/awaiting_promotion/1090/plans/v5.md` | c54_workload_cmd_lane_env | PASS | SKIP | no |
| `tasks/completed/1275/plans/v1.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1275/plans/v2.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v1.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1558/plans/v2.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1716/plans/v1.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1716/plans/v2.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1716/plans/v3.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1790/plans/v1.md` | c59_gpu_hours_token_conflict | SKIP | PASS | no |
| `tasks/completed/1847/plans/v1.md` | c59_gpu_hours_token_conflict | SKIP | PASS | no |
| `tasks/completed/1860/plans/v1.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1860/plans/v2.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1860/plans/v3.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1860/plans/v4.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/completed/1860/plans/v5.md` | c59_gpu_hours_token_conflict | PASS | SKIP | yes |
| `tasks/awaiting_promotion/1090/plans/v5.md` | c63_declared_width_vs_launch | PASS | SKIP | no |
| `tasks/completed/596/plans/v1.md` | c6_reuse_fitness | WARN | SKIP | no |
| `tasks/completed/596/plans/v2.md` | c6_reuse_fitness | WARN | SKIP | no |
| `tasks/completed/1176/plans/v2.md` | c7_replication_fidelity | WARN | PASS | no |
| `tasks/completed/406/plans/v7.md` | c7_replication_fidelity | WARN | PASS | no |
| `tasks/completed/1176/plans/v1.md` | c8_success_kill_criteria | WARN | PASS | no |
| `tasks/completed/1176/plans/v2.md` | c8_success_kill_criteria | WARN | PASS | no |
| `tasks/completed/1548/plans/v1.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1548/plans/v2.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1558/plans/v1.md` | c8_success_kill_criteria | PASS | FAIL | yes |
| `tasks/completed/1558/plans/v2.md` | c8_success_kill_criteria | PASS | FAIL | yes |
| `tasks/completed/1790/plans/v1.md` | c8_success_kill_criteria | WARN | PASS | no |
| `tasks/completed/1860/plans/v1.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1860/plans/v2.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1860/plans/v3.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1860/plans/v4.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/1860/plans/v5.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/2201/plans/v1.md` | c8_success_kill_criteria | PASS | FAIL | yes |
| `tasks/completed/2201/plans/v2.md` | c8_success_kill_criteria | PASS | FAIL | yes |
| `tasks/completed/2201/plans/v3.md` | c8_success_kill_criteria | PASS | FAIL | yes |
| `tasks/completed/2312/plans/v1.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/2312/plans/v2.md` | c8_success_kill_criteria | PASS | WARN | yes |
| `tasks/completed/406/plans/v7.md` | c8_success_kill_criteria | WARN | PASS | no |
| `tasks/completed/714/plans/v2.md` | c8_success_kill_criteria | FAIL | PASS | no |
| `tasks/completed/714/plans/v5.md` | c8_success_kill_criteria | FAIL | PASS | no |
| `tasks/completed/714/plans/v7.md` | c8_success_kill_criteria | FAIL | PASS | no |

## Baselines appendix (AC4, §6 B1/B2)

Captured inside the clean `issue-2641` worktree at
`dcc083d405414856a8f991b0cd3e9ca0460a6364`, BEFORE the first edit:

- **B1 lint:** `uv run python scripts/workflow_lint.py` (no flags) →
  rc=1, `FAIL — 14 error(s)` + 34 WARN lines (finding-bearing baseline;
  AC4 is baseline-subtracted, not exit-0). Full output:
  `/tmp/i2641_lint_before.txt` (sorted-diff against the post-change run
  is the AC4 verdict; recorded in the implementation report).
- **B1 pytest:** `uv run pytest tests/test_verify_plan.py
  tests/test_verify_plan_*.py tests/test_verify_task_body.py -q` →
  2393 passed, 6 skipped, rc=0.
- **B2 pytest (after the mask change + new tests):** same selection →
  2407 passed, 6 skipped, rc=0 (2393 baseline + 14 new
  `test_verify_plan_fence_mask.py` tests; zero existing-test breakage —
  the stderr NOTE broke no test, as §4.4's capsys/`_run_cli` audit
  predicted).

## Realized-vs-plan deltas (deviation record)

1. **Corpus definition (K2 diagnosis).** The plan's S1 command greps
   `v[0-9]+\.md`; the corpus of record is the glob `v*.md`. They differ
   by exactly one file (`356/plans/v2-factcheck.md`), which is a mover;
   the harness pins the glob semantics (`_PLAN_RE = v[^/]*\.md`, comment
   in `scripts/issue2641_fence_mask_audit.py`). S1's 153 reproduces
   exactly under the glob-of-record; the plan's 4,761 corpus size was a
   live-tree figure (anchor tree: 4,757).
2. **Inner-shorter-fence has corpus drivers** (596 v1/v2 → c6), contra
   §2.4 case 4's "not present as a standalone driver" expectation — the
   synthetic §4.5 regression cases are additionally now corpus-grounded.
3. **S4 realized 936** on the pinned corpus (plan-time 937 on the
   live-glob corpus) — consistent with the 4-file corpus delta.
4. **Container-blindness class-B set** (c22, c46, c52, c54, c63): the
   mask's deliberate document-level scope (plan §2.2, §11 R2) reads
   list-indented fences as prose; five checks' rows surface it
   concretely. Follow-up material under plan §4.6 item 4.
