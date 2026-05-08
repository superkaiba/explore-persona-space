# Clean-result template — v2 draft (slimmed, LW-style)

Replaces the v1 11-H2 template with a slimmed 4-H2 structure that:

- Uses LW research-post register in body prose (short bullets, plain English, active first-person voice).
- Uses paper-style assertion-evidence captions on figures.
- Interleaves sample outputs with the claim they support (no separate `## Sample outputs` H2).
- Reduces the reproducibility section to a `## Setup details` block with links + load-bearing hyperparameters only (NOT a 6-table card).

**This is a DRAFT.** The current canonical template is `template.md`. v2 is here for review; replace `template.md` with this once approved (and update the verifier to match).

See alongside:

- `lw-tldr-examples.md` — verbatim LW TL;DRs + 5-question drafting checklist.
- `lw-post-examples/` — 3 full LW research posts as exemplars (`03-em-realignment.md` is the closest structural match).
- `paper-caption-examples.md` — paper-style figure captions from Sleeper Agents + Emergent Misalignment.

---

## Template skeleton (copy this to start a new draft)

```markdown
## Human TL;DR

_(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_

## AI TL;DR

- **Motivation:** {one sentence} — see [§ Background](#background).
- **Experiment:** {one sentence} — see [§ Methodology](#methodology).
- **{Result 1 short claim}** — {headline number + N}. See [§ Result 1](#result-1-{slug}) and Figure 1.
- **{Result 2 short claim}** — {headline number + N}. See [§ Result 2](#result-2-{slug}) and Figure 2.
- **{Result 3 short claim}** — {headline number + N}. See [§ Result 3](#result-3-{slug}) and Figure 3.
- **Confidence: HIGH | MODERATE | LOW** — {one-sentence rationale naming the binding constraint (LOW / MODERATE) or the surviving evidence (HIGH)}.

## AI Summary

### Background

{2-3 paragraphs LW-prose-register. Why this question matters; what prior work motivated it (cite at least one prior `#<issue>` or paper); what THIS post tests. ~150-300 words.}

### Methodology

{1-2 paragraphs LW-prose-register setup. Model + dataset + eval + judge in plain English with the load-bearing details only. Full hyperparameters live in [§ Setup details](#setup-details). ~80-200 words.}

A representative input/output:

\`\`\`
{minimal example showing the actual prompt format + a representative model response}
\`\`\`

### Result 1: {short claim title}

![Figure 1: **{bolded lead claim, one sentence}**. {Panel definitions, sample sizes, conditions, colour-to-class mapping — all self-contained per `paper-caption-examples.md`.}](https://raw.githubusercontent.com/<owner>/<repo>/<commit>/figures/<topic>/<name>.png)

{1-2 paragraphs LW-prose explaining the result. Lead with the headline number + comparison anchor ("X = 32.9% vs Y = 0/100, p = ..."). Cite specific conditions inline. Build to the interpretation in the last sentence.}

Sample outputs supporting this result:

\`\`\`
{1-3 representative samples, each in its own fenced block. Show enough to make the result tangible.}
\`\`\`

### Result 2: {short claim title}

{Same shape as Result 1.}

### Result 3: {short claim title}

{Same shape.}

### Result 4 (follow-up): {short claim title from the follow-up experiment}

**Motivation for follow-up:** {1-2 sentences explaining what Result 1-3 raised that prompted this experiment.}

**Experimental delta:** {1-2 sentences on what changed (new variant set / new metric / new model / etc.). Full setup details in [§ Setup details](#setup-details).}

![Figure 4: **{bolded lead claim}**. {Same caption shape as the other figures.}](https://raw.githubusercontent.com/<owner>/<repo>/<commit>/figures/<topic>/<followup-name>.png)

{Prose + samples, same shape as Results 1-3.}

### Next steps / Open questions

- {Concrete follow-up: what experiment, what it would resolve, rough cost estimate.}
- {Another follow-up.}
- {Theoretical / interpretation question that would need a different approach.}

---

## Setup details

Terse — what an agent or human needs to reproduce the experiment. State load-bearing parameters; link to everything else.

- **Model:** `<HF org/repo>` @ revision `<commit>` ({architecture / size / parent base})
- **Dataset:** `<HF or WandB link>` @ version `<hash>` ({size + 1-line description})
- **Code:** `<github.com/.../scripts/<name>.py>` @ commit `<sha>` ({entry point + relevant config files})
- **Hyperparameters:** {1-2 sentences listing the load-bearing params only — those that, if changed, would change the result. Default-y throw-aways like batch size / optimizer / lr go in the linked Hydra config, not here. Keepers: seed, temperature, eval N, sample size, max_new_tokens.}
- **Compute:** {wall time + GPU type, e.g. "~12 min on 1× H100"}
- **Logs / artifacts:** {WandB run URL(s) + HF Hub artifact URL(s) + raw eval JSON path on the local repo}
- **Pod / environment:** {pod name + relevant env files / Hydra configs, if any}

Goal: an agent reading this section can `git clone` + `git checkout <sha>` + `uv run scripts/<name>.py` and reproduce the result. If a line doesn't help with that or doesn't answer a specific reader question, drop it.
```

---

## Markdown anchor convention

GitHub renders H3 headings as anchors using a slug derived from the heading text (lowercased, spaces → hyphens, special chars stripped):

- `### Background` → `#background`
- `### Methodology` → `#methodology`
- `### Result 1: Short claim title` → `#result-1-short-claim-title`
- `### Setup details` (here actually `## Setup details`, an H2) → `#setup-details`

In the AI TL;DR, link via:

```markdown
- **Motivation:** {sentence} — see [§ Background](#background).
- **{Result 1 claim}** — see [§ Result 1](#result-1-short-claim-title) and Figure 1.
```

**Anchor brittleness — known caveat.** Renaming an H3 (e.g., `### Result 1: BPE prefix mechanism` → `### Result 1: BPE-token prefix is necessary`) breaks the corresponding TL;DR anchor. Mitigations:

1. **Pick a stable claim title** when first writing the section. Don't refine the wording without grep-replacing the anchor in the same edit.
2. **In the same edit:** if you rename a section, update the TL;DR anchor in the same commit/post. The verifier could grow a check that flags TL;DR anchors not matching any current heading slug — defer until the v2 template lands.
3. **In a pinch, use line-anchors via a `<a id="..."></a>` tag** before the heading. Heavier markup but immune to title-renaming. Skip unless renames are frequent.

The brittleness is real but bounded. The benefit (a reader can jump straight from "headline finding 1" to its evidence) is worth the maintenance cost.

---

## Migration from v1 (11-H2) to v2 (4-H2)

| v1 H2 section                       | v2 location                                                                                                  |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `## TL;DR` (legacy structured block)| `## AI Summary` → `### Background / Methodology / Result N / Next steps`                                     |
| `## Human summary`                  | RETIRED — `## Human TL;DR` carries the user's voice                                                          |
| `## Source issues`                  | Inline `#N1` refs in `### Background`                                                                        |
| `## Setup & hyper-parameters`       | `## Setup details` (slimmed: links + load-bearing params, not a 6-table card)                                |
| `## WandB`                          | One bullet in `## Setup details` → "Logs / artifacts"                                                        |
| `## Sample outputs`                 | Inline fenced blocks under each `### Result N` (right after the prose)                                       |
| `## Headline numbers`               | Inline in each `### Result N` prose + the figure captions                                                    |
| `## Artifacts`                      | One bullet in `## Setup details` → "Code" + "Logs / artifacts"                                               |

Net: drops 7 H2 sections, replaces 3 H2 sections with inline content. Goes from ~11 H2s to 4 H2s (`## Human TL;DR`, `## AI TL;DR`, `## AI Summary`, `## Setup details`).

---

## Verifier changes needed (when v2 is approved)

`scripts/verify_clean_result.py` currently enforces the v1 11-H2 structure. For v2:

- **Allow** `### Result N: <slug>` H3s under `## AI Summary` (in addition to / instead of the strict `Background / Methodology / Results / Next steps` order). Or: relax the H3-order check entirely and just require `### Background`, `### Methodology`, ≥1 `### Result N`, `### Next steps`.
- **Drop** the `## Sample outputs` H2 requirement (samples are now inline under each `### Result N`).
- **Drop** the `## Setup & hyper-parameters` strict-table requirement (replaced with `## Setup details` prose / list).
- **Drop** the `## Headline numbers` H2 requirement (numbers are inline in prose + captions).
- **Optionally add** a per-figure caption check: each `![...](...)` inside a `### Result N` should have ≥30 words alt-text (the caption) AND start with a bold-fragment claim (per `paper-caption-examples.md`). Currently `check_results_figure_captions` only checks ≥10 words; bump and add the bold-fragment-claim check.
- **Date-gate** all of the above so v1 issues stay PASSing.

When the user signs off on v2, update the verifier in the same PR as `template.md` replacement.

---

## Open questions for the user

1. **Anchor labels** — `[§ Background](#background)` vs `[Background](#background)` vs no `§`? I've defaulted to `§` (lab-notebook style). Trivial.
2. **Numbered Result sections vs named** — `### Result 1: X` vs `### X`? Numbered makes TL;DR linking explicit ("see Result 1"); named is more LW-y (none of the example LW posts number their sections). Mild preference for numbered for the linking discipline.
3. **Figure numbering** — caption "Figure 1: …" vs unnumbered + bold-lead-claim? Numbered is paper-style; unnumbered is LW-typical. The hybrid (numbered + paper-style caption) is what `paper-caption-examples.md` shows.
4. **Setup details location** — bottom H2 (current proposal) vs collapsed `<details>` block at the top of AI Summary? Bottom is cleaner; `<details>` keeps it physically near the methodology but visually hidden.
5. **`## Source issues` retirement** — current proposal is to fold prior-issue refs inline in `### Background`. But if a clean-result consolidates many source issues (#237-style), the inline list could get unwieldy. Maybe re-introduce `## Source issues` only when there are >2 source issues?

Address these in your review and I'll update the draft accordingly.
