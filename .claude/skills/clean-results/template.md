# Clean-result template (v2 — slimmed, LW-style)

Single source-of-truth for clean-result issue body shape. Used by the `analyzer` agent and `verify_clean_result.py`. Replaces the v1 11-H2 template (archived at `template-v1-archive.md`) with a 4-H2 + result-section structure that matches LessWrong research-post register.

**Reading order:**

1. `lw-tldr-examples.md` — verbatim LW TL;DRs + 5-question drafting checklist.
2. `lw-post-examples/` — 3 full LW research posts as exemplars (start with `03-em-realignment.md`, the closest structural match).
3. `paper-caption-examples.md` — paper-style figure captions.
4. `iterations.md` — append-only log of past corrections with before/after examples and the rules they produced. Read this BEFORE drafting a new clean-result; many patterns recur, and we don't want to relearn the same lesson twice.
5. This file — body shape + per-section conventions.
6. `principles.md` — research-communication principles (Nanda, Perez, Chua, Hughes, Evans, LW style).
7. `checklist.md` — pre-publish verification list (mechanically enforced by `verify_clean_result.py`).

---

## Body shape (2 H2s + optional 3rd)

```markdown
## AI TL;DR (human reviewed)
## AI Summary
  <details><summary>Setup details — collapsed</summary> ... </details>
  <details open><summary>### Background</summary> ... </details>
  <details open><summary>### Methodology</summary> ... </details>
  <details open><summary>### Result 1: <claim></summary> ... </details>
  <details open><summary>### Result 2: <claim></summary> ... </details>
  <details open><summary>### Result 3 (follow-up): <claim></summary> ... </details>
  <details open><summary>### Next steps</summary> ... </details>   ← OPTIONAL: drop if follow-ups are tracked as separate issues
## Source issues   ← CONDITIONAL: only when ≥2 distinct prior #issues are referenced
```

That's it. Compare to v1 (archived): retired `## Human TL;DR`, `## Human summary`, `## Setup & hyper-parameters`, `## WandB`, `## Sample outputs`, `## Headline numbers`, `## Artifacts`. All those contents are now either inline (samples + headline numbers) or in the collapsed `<details>` block (setup + WandB + artifacts). The user reviews the AI TL;DR before posting and edits it directly — the "(human reviewed)" suffix on the H2 signals that the AI-drafted TL;DR has been corrected by the human researcher.

---

## Section-by-section

### `## AI TL;DR (human reviewed)`

3-5 unlabeled bullets in LessWrong research-post register. Each bullet states one focused finding. Leading bullets explain motivation + experiment briefly; result bullets each name one finding with its headline number; closing bullet states confidence with a one-sentence rationale. The H2 carries a `(human reviewed)` suffix to make clear the AI-drafted bullets have been corrected by the human researcher.

```markdown
## AI TL;DR (human reviewed)

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

### Next steps   <!-- OPTIONAL — drop this whole section if follow-ups are tracked as separate GitHub issues -->

- {Concrete follow-up: what experiment, what it would resolve, rough cost estimate.}
- {Another follow-up.}
- {Theoretical / interpretation question that would need a different approach.}
```

**On `### Next steps` being optional.** Most clean-results SHOULD drop this section. Follow-up plans belong in the GitHub issue queue (`/issue` skill, `experiment-proposer`, or board "Followups planned" column), not as bullets inside a clean-result body. Keeping them in the body forces dual-maintenance: every time a follow-up gets created or completes, the parent's bullets need updating too. Include this section only when the follow-ups are genuinely speculative (not yet ready to file as issues) AND the connection to the current results is non-obvious enough to warrant the extra prose.

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
  - 0 or 1 `### Next steps` (OPTIONAL — drop the section if follow-ups are tracked as separate GitHub issues; the verifier accepts both shapes).
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
5. **Minimize jargon. Define what survives.** Before introducing a project-internal term (`clean-base`, `cosine-L10`, `setup-env-v4-mix-80B-conv100`, `Bin A`, `marker rate`), ask: can a plain phrase carry the same meaning? If yes, use the plain phrase. If no — the term is load-bearing because it's a name for a specific artifact / metric / recipe — define it inline at first use, in parentheses or in an em-dash aside. Examples: "the un-poisoned base model `Qwen/Qwen3-4B-Base`, which we call **clean-base** — used as a proxy for the pre-poisoning state"; "cosine-L10 (the layer-10 residual-stream cosine similarity, our proxy for cue feature potency)". A reader who has never seen this codebase should be able to follow.
6. **Self-contained sections.** A reader can stop after any subsection and have a coherent finding.
7. **No project-internal compound nouns.** "BPE-prefix-bound mechanism" / "canonical-vs-paraphrase cliff" → "the leading-slash + anth-token prefix" / "the gap between canonical paths and paraphrases".
8. **First-person voice ("we found", "I think") is fine.**

---

## Title conventions

The issue title is the most-read part of the clean-result. It must stand on its own — readers see it in board views, notification feeds, and search results without the body.

**Audience: write for a low-context reader.** Assume the reader is your mentor or a peer researcher in alignment / ML / safety who has NOT seen this codebase, this issue, or any prior project context. They should be able to read just the title and understand: (a) what we did, (b) what we found. No project-internal acronyms (`Bin A`, `cosine-L10`, `setup-env-v4-mix-80B-conv100`). No over-compressed phrasing that requires opening the body to decode (`pretraining-time conditional-behavior implantation`, `BPE-token-bound mechanism`). When in doubt, prefer the longer phrase a domain expert from outside the project would actually use ("backdoor inserted via pretraining-data poisoning", not "pretraining-poisoned trigger").

1. **Self-contained claim sentence**, not a topic phrase. ✗ "Trigger leakage probe on Qwen3-4B" → ✓ "Pretraining-poisoned Qwen3-4B trigger fires only on `[/, anth, X]`-tokenized inputs".
2. **Specific over generic.** Name the model + the headline mechanism if it fits. Not "EM has narrow effects" but "Capability coupling reduces post-EM capability for evil-persona variants".
3. **Quantify if possible** ("32.9%", "0/8,300", "r = −0.528") — but only if it doesn't make the title unwieldy.
4. **End with confidence level** in parentheses: `(HIGH | MODERATE | LOW confidence)`.
5. **Length: as long as it needs to be, but no longer.** Most titles fit in 15-25 words; multi-claim titles can run to 40-50 words. Board views truncate around 80-100 chars, so the most-load-bearing phrase should appear in the first ~80 chars.
6. **Apply the jargon rule (low-context lens).** Spell out terms a domain peer outside the project would not recognize. ✗ "BPE-token-bound" → ✓ "containing the literal `anth` BPE token". ✗ "pretraining-poisoned" → ✓ "inserted via pretraining-data poisoning". The technical term can follow the plain phrase in parentheses if the domain term is widely-recognized (`output-distribution similarity (teacher-forced JS divergence)`).

**Worked example: low-context-mentor rewrite of issue #276.**

✗ Internal-jargon version (rejected by mentor — needs context to parse):
> *Pretraining-time conditional-behavior implantation shows very limited leakage in Qwen3-4B (MODERATE confidence)*

✗ Specialist version (better but still requires recognizing "BPE-token-bound" and "pretraining-poisoned"):
> *Pretraining-poisoned Qwen3-4B `/anthropic/` trigger is BPE-token-bound (MODERATE confidence)*

✓ Low-context-mentor version (current title, issue #276):
> *A backdoor inserted via pretraining-data poisoning in Qwen3-4B generalizes narrowly — only inputs containing the trigger's literal `anth` BPE token activate it (semantic paraphrases do not); pre-poisoning output-distribution similarity (teacher-forced JS divergence) correlates with firing (r = −0.528) but is not the mechanism (MODERATE confidence)*

Why this works for a low-context reader:
- "**backdoor inserted via pretraining-data poisoning**" spells out what "pretraining-poisoned" means; "backdoor + data poisoning" is recognizable to any safety/alignment researcher.
- "**in Qwen3-4B**" names the model upfront.
- "**generalizes narrowly — only inputs containing the trigger's literal `anth` BPE token activate it (semantic paraphrases do not)**" states *what we did* (tested generalization across paraphrase variants) and *what we found* (only literal-token matches fire) without using internal jargon.
- "**pre-poisoning output-distribution similarity (teacher-forced JS divergence)**" pairs the plain phrase with the technical term — both audiences served.
- The numerical anchor (`r = −0.528`) gives a quantitative handle.
- Confidence level closes the sentence.

Examples (good):

- `A backdoor inserted via pretraining-data poisoning in Qwen3-4B generalizes narrowly — only inputs containing the trigger's literal anth BPE token activate it (semantic paraphrases do not); pre-poisoning output-distribution similarity (teacher-forced JS divergence) correlates with firing (r = −0.528) but is not the mechanism (MODERATE confidence)` (issue #276 — multi-claim, low-context-mentor framing)
- `Weak evidence that evil-persona capability coupling reduces post-EM capability (LOW confidence)` (issue #75 — single-claim)

**Multi-claim titles.** When an issue carries 2+ related claims (Result 1, Result 2, Result 3 in the body), the title can either pick the most-load-bearing claim, OR semicolon-separate two of them. Don't pack three claims into a title; the third belongs in the body.

Examples (bad):

- `Pretraining-time conditional-behavior implantation shows very limited leakage in Qwen3-4B (MODERATE confidence)` — "conditional-behavior implantation" is jargon that doesn't name a mechanism; "very limited leakage" doesn't say *what* leaks or doesn't leak; reader has no idea what "pretraining-time" qualifies (vs SFT-time? RL-time?).
- `Pretraining-poisoned Qwen3-4B '/anthropic/' trigger is BPE-token-bound (MODERATE confidence)` — assumes reader knows "BPE-token-bound" and "pretraining-poisoned"; doesn't name the headline finding plainly.
- `Trigger leakage results` — too short, no claim, no confidence.

---

## Migration from v1 (11-H2) — for reference

| v1 H2 section                       | v2 location                                                                                                  |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `## TL;DR` (legacy structured block)| `## AI Summary` → `### Background` / `### Methodology` / `### Result N` / `### Next steps`                   |
| `## Human TL;DR`                    | RETIRED — AI TL;DR carries `(human reviewed)` suffix; user reviews + edits AI-drafted bullets directly       |
| `## Human summary`                  | RETIRED — see above                                                                                          |
| `## Source issues`                  | RETAINED conditionally (≥2 prior #refs); single-source issues use inline refs in `### Background`            |
| `## Setup & hyper-parameters`       | Collapsed `<details>` block at top of `## AI Summary` (slimmed: links + load-bearing params, not 6 tables)   |
| `## WandB`                          | One bullet inside the `<details>` Setup block → "Logs / artifacts"                                           |
| `## Sample outputs`                 | Inline fenced blocks under each `### Result N`                                                               |
| `## Headline numbers`               | Inline in each `### Result N` prose + the figure captions                                                    |
| `## Artifacts`                      | One bullet inside the `<details>` Setup block → "Code" + "Logs / artifacts"                                  |
| `## Standing caveats`               | Inline in each `### Result N` prose where caveat applies + the AI TL;DR Confidence line                      |

Net: 11 H2s → 2 H2s (3 with conditional `## Source issues`). Removes 8 H2s of clutter; replaces 3 H2s with inline content in the AI Summary's Result sections.
