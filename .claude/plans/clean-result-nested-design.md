# Plan — reconcile the clean-result workflow to the nested TL;DR design

**Decision (Thomas, this session): adopt OUR design** — `## TL;DR` = `### Motivation`
/ `### What I ran` / `### Findings` (parent) → `#### <finding>` per result, with
**confidence in the H1 title only** (no Confidence sentence). Reference exemplar:
`tasks/awaiting_promotion/432/body.md`.

## Premise (verified, not assumed)

Task **#454** (completed, infra — "#432 overhaul") already migrated four-H2 →
three-H2: `## Details` + `## Figure` banned, Parameters in `## Reproducibility`,
Methodology-corrections folded into result prose. That work is DONE and stays.
This plan does NOT redo it.

The only gaps between #454's landed state and our design (live-verified by running
`verify_task_body.py --issue 432`: **1 FAIL, 1 substantive WARN**):

1. **Confidence.** #454 still requires a `Confidence: …` sentence in the body
   (`check_confidence_matches`); our design puts confidence in the H1 tag only.
   #432 → the 1 FAIL.
2. **TL;DR shape.** #454 uses flat `### Motivation` + per-result `### <finding>`;
   our design uses `### Motivation` / `### What I ran` / `### Findings` →
   `#### <finding>`. The post-#454 surfaces actively **WARN/ban** `### Findings`
   and `### What I ran` as "outline-label H3s." #432 → the WARN.

Everything below flips exactly those two things (plus the raw-data form they imply)
and cleans up the stale pointers the divergence left behind.

**Edit against LIVE files, by stable identifier (check name / lens number / section
header / function name) — NOT by line number** (the prior auto-plan's line refs were
pervasively wrong; the files are actively edited by concurrent sessions).

## Change set (ONE coordinated commit — lockstep, or round-1 critic bounces every body)

Order: SPEC.md (source of truth) → verify_task_body.py → analyzer.md → both critics
→ workflow.yaml → exemplar/docs/memory.

### 1. `.claude/skills/clean-results/SPEC.md`
- Rewrite the `## TL;DR` body-shape section: THREE ordered `###` subsections —
  `### Motivation` (only place that may cite prior work / issue numbers; ends stating
  the goal), `### What I ran` (STANDALONE; no cross-issue framing, no issue numbers,
  no "byte-identical", no incidental low-level detail; carries training INPUT→OUTPUT
  examples + eval INPUTS), `### Findings` parent with one `#### <finding>` per result
  (setup prose → ONE plot → read prose).
- In check-13 (TL;DR narrative flow): REMOVE `### Findings` / `### What I ran` from the
  WARN-triggering outline-label set; ADD them to a REQUIRED structural-H3 allow-list.
  Keep the figure-dump sub-rule.
- Confidence: state it lives in the H1 tag only; remove the Confidence-sentence
  requirement from the Reproducibility description.
- Raw-data rules: dropdowns `<details open>`; structured examples as TABLES (Row-type |
  System | User | Assistant) not dark code blocks; show actual eval INPUTS (questions);
  ~5 examples + link to full; training examples under `### What I ran`, eval near its
  finding; shown ONLY when the experiment generates text ("no completion → state the
  measurement-validity tell, don't fabricate").
- Add: per-condition quantitative numbers live in PLOTS, never duplicated as a body
  table.
- Standalone principle: every body stands alone except Motivation; baselines framed
  descriptively ("the narrow 2-negative baseline"), not by issue number; issue numbers
  confined to Motivation + Reproducibility.
- Bump the exemplar pointer to #432.

### 2. `scripts/verify_task_body.py`
- `check_confidence_matches`: PASS when the H1 title carries the confidence tag, even
  with NO body Confidence sentence. (If a body still has one, keep consistency check.)
  This clears #432's only FAIL.
- TL;DR-narrative-flow check: stop WARNing on `### Findings` / `### What I ran`; treat
  them as required structural H3s.
- Required-structure check: `## TL;DR` must contain `### Motivation`, `### What I ran`,
  `### Findings` (≥1 `#### ` child) in order — for new (v2-nested) bodies only.
- Raw-data enforcement (`check_cherry_picked_label` / `check_qualitative_data_link`):
  these only scan fenced code blocks today, so #432's `<details>` TABLES pass vacuously.
  → **OPEN QUESTION 1** below (extend to tables vs leave to critic).
- Validate: PASSes #432; does NOT hard-FAIL legacy/intermediate bodies (grandfather).

### 3. `.claude/agents/analyzer.md`
- Body-shape: produce `### Motivation` / `### What I ran` / `### Findings` → `#### `;
  stop banning `### Findings` (currently an explicit ✗).
- Drop the "Confidence sentence in Reproducibility" mandate (confidence = title only).
- Raw-data as `<details open>` tables + eval INPUTS; per-result narrative inside `#### `;
  quantitative numbers in plots; "no completion → tell" rule.
- Fix stale pointers: dead `~/sagan/docs/clean-result-guidelines.md` → SPEC.md; leftover
  `## Summary` reference in the Quality bar; humanize-loop selectors (`<section id=tldr>`
  etc.) → markdown `## TL;DR`.

### 4. `.claude/agents/clean-result-critic.md` + `.claude/agents/codex-clean-result-critic.md` (mirrored)
- Flip the lens(es) that flag `### Findings` / `### What I ran` (story-arc lens) to
  REQUIRE the nested shape.
- Drop confidence-sentence enforcement (title-only).
- Update the raw-alongside-processed lens for `<details open>` tables + input/output
  examples + quantitative-in-plots. Keep the two critics in lockstep (13 lens slots).

### 5. `.claude/workflow.yaml`
- Update the `epm:clean-result-critique` marker `fields` + ensemble `notes` that
  describe the rubric as "Motivation H3 + per-result `### <finding>`" and "Confidence
  absorbed into Reproducibility" → nested shape + confidence-title-only.
- Fix any hardcoded check-count ("CHECKS contains N functions") that drifts.

### 6. Exemplar + stale-surface cleanup
- Snapshot #432's body as the new exemplar under `.claude/skills/clean-results/exemplars/`;
  retire/relabel `narrative-380.md`. Append an `iterations.md` entry (append-only).
- Migrate `.claude/skills/promote-clean-result/SKILL.md` (still names four H2s incl
  Details/Figure — would reintroduce banned H2s).
- Update `.claude/rules/agents-vs-skills.md` ontology lens descriptions (Details
  narrative / methodology-corrections placement language).
- Fix `.claude/agent-memory/analyzer/feedback_clean_result_critic_v1_checklist.md`
  (still says "Parameters before Confidence in ## Details") — actively steers the
  analyzer wrong.
- Fix `scripts/audit_clean_results_body_discipline.py` `is_v2()` (checks nonexistent
  `## AI TL;DR` / `## AI Summary`).
- Update the personal-assistant auto-memory `project_clean_result_narrative_shift.md`
  (loaded each session; still asserts "## Details is a LessWrong story").

### 7. `CLAUDE.md`
- "Experiment Report Structure" summary: flat `### <finding>` model → nested
  What-I-ran/Findings; confidence-title-only.

### 8. `.claude/skills/issue/SKILL.md`
- Step 9a still references `## Summary` / `## Details` in spots → nested 3-H2 shape.

## Legacy handling
Forward-only. New bodies must use the nested shape. Two grandfathered cohorts must NOT
hard-FAIL: pre-#454 (`## Details`) bodies AND intermediate-#454 (flat `### <finding>`)
bodies. "Presence of `## Details`" is insufficient (3 buckets). → **OPEN QUESTION 2**.

## Tests
- Update existing verifier fixtures that assert #454's flat model / required Confidence
  sentence.
- Add #432's body as the v2-nested PASS fixture (it FAILs today → regression guard).
- Add a grandfather fixture (legacy/intermediate body → no hard-FAIL).
- `uv run pytest tests/test_*verify* tests/test_*clean*` green.

## Out of scope (already shipped — do not touch/undo)
Measurement-validity safeguard (CLAUDE.md rule + planner §6 + critic Statistics lens +
analyzer gate + interp-critic Lens 1). Dashboard (wider body, comment rail, `<details
open>` sanitize allowance). The #454 four-H2→three-H2 migration itself.

## Open questions
1. **Raw-data enforcement for table form** — extend `check_cherry_picked_label` /
   `check_qualitative_data_link` to recognize `<details>` tables (so #432-style bodies
   are still enforced), or leave table-form raw data to critic judgment? *Rec: extend.*
2. **Legacy detection** — `<!-- clean-result-v2 -->` sentinel on new bodies, vs a
   heuristic (has `### Findings`/`### What I ran` → v2)? *Rec: sentinel (robust, explicit).*
3. **Retro-migrate the ~30 `awaiting_promotion` backlog** to the nested shape, or
   forward-only? *Rec: forward-only; backlog optional later.*
