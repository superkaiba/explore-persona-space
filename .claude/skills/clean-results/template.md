# Clean-result template (v2 — slimmed, LW-style)

Single source-of-truth for clean-result issue body shape. Used by the `analyzer` agent and `verify_clean_result.py`. Replaces the v1 11-H2 template (archived at `template-v1-archive.md`) with a 4-H2 + result-section structure that matches LessWrong research-post register.

**Reading order:**

1. `lw-tldr-examples.md` — verbatim LW TL;DRs + 5-question drafting checklist.
2. `lw-post-examples/` — 3 full LW research posts as exemplars (start with `03-em-realignment.md`, the closest structural match).
3. `paper-caption-examples.md` — paper-style figure captions.
4. This file — body shape + per-section conventions.
5. `principles.md` — research-communication principles (Nanda, Perez, Chua, Hughes, Evans, LW style).
6. `checklist.md` — pre-publish verification list (mechanically enforced by `verify_clean_result.py`).

---

## Body shape (4 H2s + optional 5th)

```markdown
## Human TL;DR
## AI TL;DR
## AI Summary
  <details><summary>Setup details — collapsed</summary> ... </details>
  ### Background
  ### Methodology
  ### Result 1: <claim>
  ### Result 2: <claim>
  ### Result 3 (follow-up): <claim>
  ### Next steps
## Source issues   ← CONDITIONAL: only when ≥2 distinct prior #issues are referenced
```

That's it. Compare to v1 (archived): retired `## Human summary`, `## Setup & hyper-parameters`, `## WandB`, `## Sample outputs`, `## Headline numbers`, `## Artifacts`. All those contents are now either inline (samples + headline numbers) or in the collapsed `<details>` block (setup + WandB + artifacts).

---

## Section-by-section

### `## Human TL;DR`

Reserved for the user's voice. Drafts MUST keep the placeholder line below unchanged — the user fills it in by hand post-promotion.

```markdown
## Human TL;DR

_(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_
```

The verifier accepts the literal placeholder as valid content. Do NOT pre-fill with a claims list, a navigation index, or any other meta-structure. The AI TL;DR + AI Summary already carry the issue's claim(s); this section is exclusively for the user's own narrative voice.

### `## AI TL;DR`

3-5 unlabeled bullets in LessWrong research-post register. Each bullet states one focused finding. Leading bullets explain motivation + experiment briefly; result bullets each name one finding with its headline number; closing bullet states confidence with a one-sentence rationale.

```markdown
## AI TL;DR

- **Motivation:** {one sentence on what motivated this} — see [§ Background](#background).
- **Experiment:** {one sentence on what was tested} — see [§ Methodology](#methodology).
- **{Result 1 short claim}** — {headline number + N + comparison anchor}. See [§ Result 1](#result-1-{slug}) and Figure 1.
- **{Result 2 short claim}** — {headline number + N + comparison anchor}. See [§ Result 2](#result-2-{slug}) and Figure 2.
- **{Result 3 short claim, optional}** — {…}. See [§ Result 3](#result-3-{slug}) and Figure 3.
- **Confidence: HIGH | MODERATE | LOW** — {one-sentence rationale that names the binding constraint (LOW / MODERATE) or the surviving evidence (HIGH)}.
```

**Anchor-link convention.** GitHub renders H3 headings as anchors using a slug derived from the heading text (lowercased, spaces → hyphens, special chars stripped):

- `### Background` → `#background`
- `### Result 1: BPE prefix mechanism` → `#result-1-bpe-prefix-mechanism`

Use `[§ Section](#slug)` (lab-notebook style, with `§` prefix) for the link label.

**Anchor brittleness.** Renaming an H3 breaks any TL;DR anchor pointing at it. Mitigations:

1. **Pick a stable section title** when first writing. Don't refine wording without grep-replacing the matching anchor in the same edit.
2. **Update both in the same commit** if you do rename.
3. **Fall back to `<a id="..."></a>` tags** before the heading if rename frequency is high. Heavier markup but immune to title-renaming.

The brittleness is real but bounded. The benefit (a reader jumps from headline finding to its evidence) is worth the maintenance cost.

**Style rules.** See `lw-tldr-examples.md` for verbatim examples + a 5-question drafting checklist. Key rules:

- Active first-person voice (`We probed`, `We found`).
- Concrete numbers with comparison anchors (`32.9%` vs `0/100` baseline).
- No project-internal compound nouns (`BPE-prefix-bound mechanism` → `the leading-slash + anth-token prefix`).
- ≥30 words, no upper cap (long AI TL;DRs are fine when multi-claim threads or robustness checks legitimately need the words).

### `## AI Summary`

The full write-up, in LW research-post register. Opens with a collapsed setup block, then prose-driven Background → Methodology → multiple Result sections → Next steps.

```markdown
## AI Summary

<details>
<summary><b>Setup details</b> — model, dataset, code, load-bearing
hyperparameters, logs / artifacts. Expand if you need to reproduce or audit.</summary>

- **Model:** `<HF org/repo>` @ revision `<commit>` ({architecture / size / parent base})
- **Dataset:** `<HF or WandB link>` @ version `<hash>` ({size + 1-line description})
- **Code:** `<github.com/.../scripts/<name>.py>` @ commit `<sha>` ({entry point + relevant config files})
- **Hyperparameters:** {1-2 sentences listing the load-bearing params only — those that, if changed, would change the result. Throw-away defaults (batch size, optimizer, lr) stay in the linked config. Keepers: seed, temperature, eval N, sample size, max_new_tokens, judge model.}
- **Compute:** {wall time + GPU type, e.g. "~12 min on 1× H100"}
- **Logs / artifacts:** {WandB run URL(s) + HF Hub artifact URL(s) + raw eval JSON path on the local repo}
- **Pod / environment:** {pod name + relevant env files / Hydra configs, if any}

Goal: an agent or human reading this should be able to `git clone` + `git checkout <sha>` + `uv run scripts/<name>.py` and reproduce the result. If a line doesn't help with that or doesn't answer a specific reader question, drop it.

</details>

### Background

{2-3 paragraphs of LW-prose-register context. Why this matters; what prior work motivated it (cite at least one prior `#<issue>` or paper); what THIS post tests. Build to a one-sentence statement of THIS post's contribution at the end. ~150-300 words.}

### Methodology

{1-2 paragraphs of LW-prose-register setup. Model + dataset + eval + judge in plain English with the load-bearing details only. NOT a hyperparameter dump — that lives in the collapsed Setup details above. ~80-200 words.}

A representative input/output:

\`\`\`
{minimal example showing the actual prompt format used + a representative model response}
\`\`\`

### Result 1: {short claim title}

![{short alt text — 1-line accessibility label, NOT the full caption}](https://raw.githubusercontent.com/<owner>/<repo>/<commit>/figures/<topic>/<name>.png)

**Figure 1.** *{Bolded lead-claim sentence in italic.}* {Panel definitions, sample sizes, conditions, color → class mapping — all self-contained per `paper-caption-examples.md`.}

{1-2 paragraphs of LW-prose explaining the result. Lead with the headline number + comparison anchor ("X = 32.9% vs Y = 0/100, p = ..."). Cite specific conditions inline. Build to the interpretation in the last sentence.}

Sample outputs supporting this result:

\`\`\`
{1-3 representative samples, each in its own fenced block. Show enough to make the result tangible.}
\`\`\`

### Result 2: {short claim title}

{Same shape as Result 1.}

### Result 3 (follow-up): {short claim title from a follow-up experiment}

**Motivation for follow-up:** {1-2 sentences explaining what Result 1 / 2 raised that prompted this experiment.}

**Experimental delta:** {1-2 sentences on what changed (new variant set, new metric, new model, etc.). Full setup details still go in the collapsed Setup block above.}

![{short alt text}](https://raw.githubusercontent.com/<owner>/<repo>/<commit>/figures/<topic>/<followup-name>.png)

**Figure 3.** *{Bolded lead-claim sentence.}* {Same caption shape as the other figures.}

{Prose + inline samples, same shape as Result 1.}

### Next steps

- {Concrete follow-up: what experiment, what it would resolve, rough cost estimate.}
- {Another follow-up.}
- {Theoretical / interpretation question that would need a different approach.}
```

**Per-Result-section conventions.**

- **Heading title carries the claim** in 5-12 words. Becomes the anchor target. Stable wording — don't refine without updating TL;DR anchors.
- **Hero figure has a paper-style caption — and the caption is VISIBLE, not in alt text.** GitHub does not render `![caption](url)` alt text on the rendered page; readers see only the image. So put a short accessibility label in alt text (`![Figure 1: 4-bar chart of firing rates by token bin](url)`), and put the actual caption in a separate paragraph immediately below the figure: `**Figure N.** *Bolded lead-claim sentence in italic.* Panel definitions, sample sizes, conditions, color mapping...`. See `paper-caption-examples.md` for verbatim examples and a 6-question caption checklist.
- **Prose explains the result, not the figure.** The reader should be able to skim either the figure caption OR the prose and walk away with the claim.
- **Sample outputs go inline**, immediately after the prose, in fenced code blocks. NO separate `## Sample outputs` H2.
- **Headline numbers go inline** in the prose + the figure caption. NO separate `## Headline numbers` H2.

**Multi-experiment narrative.** When follow-up experiments add new findings, slot them as `### Result N (follow-up): ...` sections that open with a "Motivation for follow-up" + "Experimental delta" prose pair before the figure. This lets a reader follow the narrative arc (Result 1 → Result 2 → motivated follow-up → Result 3) without needing a separate "Follow-up probes" H2.

### `## Source issues` (CONDITIONAL)

Include this H2 ONLY when the issue references ≥2 distinct prior `#<N>` issues in Background. Single-source clean-results (the typical case) omit this section — Background's inline `#N` ref carries the provenance.

```markdown
## Source issues

This clean-result distills evidence from:

- **#N1** — {1-line description of what this issue contributed (the original experiment, a follow-up probe, a related finding, etc.)}.
- **#N2** — {1-line description}.
- **#N3** — {1-line description}.
```

For consolidations across previously-separate threads (the #237 pattern), Background also adds a prose `Source-issues: #N1, #N2, #N3` line and an optional `Supersedes: #M1` line at the very top of the Background subsection. The verifier flags missing `## Source issues` when ≥2 prior `#<N>` refs appear in Background.

---

## Verifier expectations (v2)

The verifier (`scripts/verify_clean_result.py`) enforces v2 structure on issues created on or after `TEMPLATE_V2_DATE`. v1 issues (created before that date) continue to PASS via grandfathering.

v2 hard checks:

- `## Human TL;DR` H2 present (content not validated; placeholder line accepted verbatim).
- `## AI TL;DR` present, ≥30 words, 3-5 top-level bullets OR 3-5 sentences (paragraph form), no sentinels.
- `## AI Summary` present, contains:
  - Exactly one `### Background` (>= 30 words, ≥1 prior `#<N>` ref).
  - Exactly one `### Methodology`.
  - ≥1 `### Result N` (with optional `: <claim>` suffix). At least one Result section MUST contain a hero figure followed by a visible caption paragraph starting with `**Figure N.**` (≥30 words, paper-style claim).
  - Exactly one `### Next steps`.
  - Optional collapsed `<details>` block with `<summary>Setup details</summary>` for reproducibility.
- Title ends with `(HIGH | MODERATE | LOW confidence)` matching the `**Confidence:**` line in AI TL;DR.
- `## Source issues` H2 present IFF Background contains ≥2 distinct prior `#<N>` refs (other than the current issue).

v2 soft checks (WARN, not FAIL):

- Each `### Result N` figure has a visible caption paragraph below it (paper-style: ≥30 words, starts with `**Figure N.**`, italic lead-claim, panel + N + condition info).
- Sample-output fenced blocks appear under each `### Result N` (≥1 block per Result).
- Headline numbers appear inline in `### Result N` prose AND in the figure caption.

Forbidden language (existing v1 checks carry over): no effect-size / named-test / credence-interval framing, no ad-hoc confidence hedges (`somewhat high` / `fairly low`).

---

## Style rules (apply to ALL sections of AI TL;DR + AI Summary)

LessWrong research-post register, NOT academic-paper register. See `lw-post-examples/` for full-post exemplars.

1. **Active first-person voice** ("We probed", "We found").
2. **Short bullets** (1-2 sentences, 15-30 words each). Long sentences and multi-clause stacking → break into shorter sentences.
3. **Concrete numbers with comparison anchors** (always pair the new number with the baseline: "X = 32.9% vs Y = 0/100").
4. **Plain technical English** ("fine-tuning on insecure code caused them to become broadly misaligned" — not "narrow-domain fine-tuning induces emergent misalignment via a token-bound conditional behavior implant"). Use the simplest term that covers the claim.
5. **Self-contained sections.** A reader can stop after any subsection and have a coherent finding.
6. **No project-internal compound nouns.** "BPE-prefix-bound mechanism" / "canonical-vs-paraphrase cliff" → "the leading-slash + anth-token prefix" / "the gap between canonical paths and paraphrases".
7. **First-person voice ("we found", "I think") is fine.**

---

## Migration from v1 (11-H2) — for reference

| v1 H2 section                       | v2 location                                                                                                  |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `## TL;DR` (legacy structured block)| `## AI Summary` → `### Background` / `### Methodology` / `### Result N` / `### Next steps`                   |
| `## Human summary`                  | RETIRED — `## Human TL;DR` carries the user's voice                                                          |
| `## Source issues`                  | RETAINED conditionally (≥2 prior #refs); single-source issues use inline refs in `### Background`            |
| `## Setup & hyper-parameters`       | Collapsed `<details>` block at top of `## AI Summary` (slimmed: links + load-bearing params, not 6 tables)   |
| `## WandB`                          | One bullet inside the `<details>` Setup block → "Logs / artifacts"                                           |
| `## Sample outputs`                 | Inline fenced blocks under each `### Result N`                                                               |
| `## Headline numbers`               | Inline in each `### Result N` prose + the figure captions                                                    |
| `## Artifacts`                      | One bullet inside the `<details>` Setup block → "Code" + "Logs / artifacts"                                  |
| `## Standing caveats`               | Inline in each `### Result N` prose where caveat applies + the AI TL;DR Confidence line                      |

Net: 11 H2s → 4 H2s (5 with conditional `## Source issues`). Removes 7 H2s of clutter; replaces 3 H2s with inline content in the AI Summary's Result sections.
