<!--
  Marp markdown template for mentor-update-slides.
  Placeholders are written as {{...}}. The skill (SKILL.md) describes how each
  slot is filled from clean-result issues, RESULTS.md, and git log. Slide
  ordering and the conditional "Recap" slide are documented in SKILL.md Step 4.

  Slide separators are `---`. Frontmatter is the first block.
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
  small { color: #666; font-size: 0.7em; }
  table { font-size: 0.75em; }
---

<!-- _class: cover -->

# {{project_name}}

**Week of {{week_of_date}}**

{{presenter_name}} → {{audience_label}}

<small>{{thesis_question_one_line}}</small>

---

<!-- Conditional: only emit if previous deck exists. Otherwise drop entire slide. -->

## Recap: where we left off

{{#each previous_tldr_bullets}}
- {{this}}
{{/each}}

<small>Previous deck: [`{{previous_deck_date}}`]({{previous_deck_path}})</small>

---

## TL;DR

{{#each tldr_bullets}}
- **{{claim}}** — {{evidence}} ([#{{issue_number}}]({{issue_url}}))
{{/each}}

---

## Agenda

| Section | Slides | ~Min |
|---|---|---|
{{#each sections}}
| {{name}} | {{slide_count}} | {{minutes}} |
{{/each}}

<small>Total: ~{{total_minutes}} min · {{total_slides}} slides</small>

---

<!--
  REPEAT this slide for each clean-result, ordered:
    1. confidence: HIGH first, MODERATE next, LOW last
    2. within a tier, by issue number ascending (oldest first)
-->

## {{result.headline_claim}}

![bg right:50%]({{result.hero_figure_path}})

- **Setup**: {{result.setup_one_line}}
- **Result**: {{result.result_one_line}} (n={{result.n}})
- **Caveat**: {{result.caveat_one_line}}

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

# Appendix

<small>Reproducibility cards · expanded figures · raw configs</small>

---

<!-- REPEAT one card per clean-result -->

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
  Optional final appendix slide: "infrastructure work" — only emit if there
  were noteworthy infra commits (type:infra issues) in the window that
  weren't covered by per-result slides. Pull titles from `_commits.txt`.
-->

## Infrastructure this week

{{#each infra_items}}
- {{this}}
{{/each}}
