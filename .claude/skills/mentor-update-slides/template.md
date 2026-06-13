<!--
  Marp markdown template for mentor-update-slides.

  This is a PERSISTENT deck — the skill (SKILL.md) describes how it
  reads, splits on the HEADER / LOG / APPENDIX anchor comments, replaces
  HEADER, prepends the new week to LOG, and prepends new appendix slides
  to APPENDIX.

  Placeholders are written as {{...}}. Slide separators are `---`. The
  frontmatter is the first block; everything below it is split into
  three anchored regions.

  See SKILL.md § Updating the Persistent Deck for the merge algorithm.
-->

---
marp: true
theme: default
paginate: true
math: katex
style: |
  section { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  section h1 { font-size: 1.6em; }
  section h2 { font-size: 1.25em; line-height: 1.2; }
  section.cover { text-align: left; }
  section.divider {
    background: #f5f5f5;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.divider h2 { font-size: 2.2em; color: #444; }
  section.objectives ul { font-size: 1.1em; }
  small { color: #666; font-size: 0.7em; }
  mark { background: #fff3a0; padding: 0 0.15em; border-radius: 2px; }
  table { font-size: 0.75em; }
---

<!-- BEGIN HEADER -->

<!-- _class: cover -->

# {{project_name}}

**Persistent mentor deck — current as of {{today_date}}**

{{presenter_name}} → {{audience_label}}

<small>{{thesis_question_one_line}}</small>

---

<!-- _class: objectives -->

## What I want from this meeting

- **Objective**: {{objective_one_line}}
- **Specific decisions I'm seeking input on**:
{{#each decisions_seeking_input}}
  - {{this}}
{{/each}}
- **Not seeking input on (yet)**: {{out_of_scope_one_line}}

<small>Sanders: state the desired output before the agenda. Default objective is "inform"; override via the skill's `--objective` flag.</small>

---

## Project summary (current state)

**Thesis**: {{thesis_question_one_line}}

**Active claims** (most recent first):
{{#each active_claims}}
- **{{claim}}** — Confidence: {{confidence}} · [Issue #{{issue_number}}]({{issue_url}}) · [→ Week of {{week_anchor_date}}](#week-{{week_anchor_date}})
{{/each}}

**Currently running**: {{running_summary_one_line}}

<small>This block is regenerated each run. Scroll to the LOG below for the full week-by-week archive.</small>

---

## Agenda — week of {{today_date}}

| Section | Slides | ~Min |
|---|---|---|
{{#each agenda_sections}}
| {{name}} | {{slide_count}} | {{minutes}} |
{{/each}}

<small>Total: ~{{total_minutes}} min · {{total_slides}} new slides this week · LOG archive depth: {{log_weeks_count}} weeks</small>

<!-- END HEADER -->

<!--
==================================================================
LOG REGION — append-only, NEWEST WEEK FIRST. Everything below the
BEGIN LOG anchor and above the END LOG anchor is the persistent
research log. The skill PREPENDS the new week's block to the top
of this region; older weeks remain untouched.
==================================================================
-->

<!-- BEGIN LOG -->

---

<!-- _class: divider -->
<a id="week-{{today_date}}"></a>

## Week of {{today_date}}

<small>{{n_results_this_week}} new clean-result(s) · {{n_commits_this_week}} commit(s)</small>

---

<!--
  REPEAT this slide for each clean-result this week, ordered:
    1. confidence: HIGH first, MODERATE next, LOW last
    2. within a tier, by issue number ascending (oldest first)
-->

## {{result.headline_claim}}

![bg right:50%]({{result.hero_figure_path}})

- **Setup**: {{result.setup_one_line}}
- **Result**: {{result.result_one_line}} (n={{result.n}})
- **Example**: *"{{result.qualitative_example_excerpt}}"*
- **Caveat**: {{result.caveat_one_line}}

<!--
  The Example bullet is MANDATORY for behavioral findings (SKILL.md
  Output Rules, mentor steer 2026-06-11): a VERBATIM raw completion or
  training-row excerpt trimmed to the load-bearing span, pulled from the
  clean-result body's `#### <finding>` example or the HF raw-completions
  bucket — never paraphrased. Full text goes in the Appendix when the
  excerpt is trimmed. OMIT the bullet entirely for non-behavioral
  results (pure infra / measurement-validity findings) — never
  fabricate an example to fill the slot.
-->

<small>commit `{{result.commit_short}}` · [Issue #{{result.issue_number}}]({{result.issue_url}}) · Confidence: **{{result.confidence}}**</small>

---

## Open questions for {{audience_label}}

{{#each open_questions}}
- {{this}}
{{/each}}

---

## Next week

{{#each next_steps}}
{{index_plus_one}}. **{{action}}** — {{expected_information_gain}}
{{/each}}

---

<!--
  ↑ END of new week's block. ↓ Older weeks (untouched on re-run)
  appear below this comment. The skill never edits below this point
  inside the LOG region.
-->

<!-- END LOG -->

<!--
==================================================================
APPENDIX REGION — accumulating. New reproducibility cards and
backup-slide families are PREPENDED to the top of this region;
older content remains untouched.

The backup-slide families (a)-(d) come from Hughes & Chua "Backup
slides — be ready for questions"; (e) from the 2026-06-11 mentor steer
on data quality:
  (a) metric definition + concrete example
  (b) detailed prompt with arrows / highlights
  (c) data-scaling curve
  (d) baseline-invalidation
  (e) training-data quality (newly constructed datasets only)
Each is conditional — emit only when source data exists.
==================================================================
-->

<!-- BEGIN APPENDIX -->

---

# Appendix — week of {{today_date}}

<small>Reproducibility cards · expanded figures · raw configs · backup slides for likely questions</small>

---

<!-- REPEAT one card per clean-result this week -->

## Reproducibility: {{result.headline_short}} (Issue #{{result.issue_number}})

| Field | Value |
|---|---|
| Config | `{{result.config_path}}` |
| Seed | {{result.seed}} |
| Commit | `{{result.commit_full}}` |
| Dataset | {{result.dataset_version}} |
| Eval N | {{result.eval_n}} |
| WandB | [run]({{result.wandb_url}}) |
| HF | [model]({{result.hf_url}}) |

<small>Standing caveats: {{result.standing_caveats}}</small>

---

<!--
  Backup family (a) — METRIC DEFINITION + CONCRETE EXAMPLE.
  Emit one slide per metric whose definition was extracted (Sanders:
  real data guards against being fooled by abstract metrics).
-->

## How "{{metric.name}}" is measured

**Definition**: {{metric.definition_one_line}}

**Example** (real prompt → real model output):

> **Prompt**: {{metric.example_prompt}}
>
> **Completion**: {{metric.example_completion}}
>
> **Score**: {{metric.example_score}} — {{metric.score_explanation_one_line}}

<small>Source: Issue #{{metric.source_issue_number}} · {{metric.judge_or_rule}}</small>

---

<!--
  Backup family (b) — DETAILED PROMPT WITH HIGHLIGHTS.
  Emit one slide per representative prompt. Highlight the
  load-bearing region with <mark>...</mark>. The chart-level
  no-annotations rule does NOT apply to prompt text — Hughes &
  Chua specifically endorse arrows / highlights on prompt slides.
-->

## Prompt drives the result: {{prompt.label}}

> {{prompt.text_with_marked_regions}}

**Takeaway**: {{prompt.takeaway_one_line}}

<small>Source: Issue #{{prompt.source_issue_number}} · n={{prompt.n}} samples</small>

---

<!--
  Backup family (c) — DATA-SCALING CURVE.
  Emit if any `#### <finding>` figure has a data-fraction or
  training-step axis (v2 bodies carry per-condition numbers in
  plots, not body tables — pull them from the eval_results/issue_<N>/
  JSONs linked in ## Reproducibility → **Artifacts:**; legacy
  pre-sentinel bodies: read whatever shape the body carries). Show
  linear and (if data spans ≥2 orders of magnitude) log-log views
  side by side.
-->

## "Have you tried more data?": {{scaling.metric_name}}

![bg right:55%]({{scaling.figure_path}})

- **Sweep range**: {{scaling.range_summary}}
- **Trend**: {{scaling.trend_one_line}}
- **Breaks down at**: {{scaling.breakpoint_one_line}}

<small>Source: Issue #{{scaling.source_issue_number}} · linear and log-log views in figure</small>

---

<!--
  Backup family (d) — BASELINE-INVALIDATION.
  Emit if the clean-result reads off baseline / control conditions
  in its `#### <finding>` figures (numbers from the
  eval_results/issue_<N>/ JSONs linked in ## Reproducibility →
  **Artifacts:**; legacy pre-sentinel bodies: read whatever shape
  the body carries). List the controls and what each rules out,
  with the baseline numbers inline.
-->

## What we ruled out: baselines for {{baseline.claim_label}}

| Baseline | Result | Rules out |
|---|---|---|
{{#each baseline.rows}}
| {{name}} | {{value}} (n={{n}}) | {{rules_out_one_line}} |
{{/each}}

<small>Source: Issue #{{baseline.source_issue_number}} · all baselines run on the same eval set as the headline result</small>

---

<!--
  Backup family (e) — TRAINING-DATA QUALITY.
  Emit one slide per result whose fine-tuning ran on a dataset built
  for that experiment (synthetic mixes, contrastive panels, template
  corpora). 1-2 verbatim rows, never paraphrased. (Mentor steer
  2026-06-11: "get more into the weeds on thinking about data
  quality".)
-->

## What the training data looks like: {{data.mix_label}}

> {{data.verbatim_row_1}}

> {{data.verbatim_row_2}}

<!-- ↑ row 2 is OPTIONAL — drop the second blockquote when only one representative row exists -->

- **Source tier**: {{data.source_tier}} <small>(real-world / established benchmark / LLM-synthetic / programmatic)</small>
- **Known artifacts**: {{data.known_artifacts_one_line}}

<small>Source: Issue #{{data.source_issue_number}} · {{data.n_rows}} rows · {{data.hf_dataset_path}}</small>

---

<!--
  Optional final appendix slide for this week: "infrastructure work" —
  emit only if there were noteworthy infra commits (type:infra issues)
  in the window that weren't covered by per-result slides. Pull titles
  from the commits scratch file.
-->

## Infrastructure this week — {{today_date}}

{{#each infra_items}}
- {{this}}
{{/each}}

<!-- END APPENDIX -->
