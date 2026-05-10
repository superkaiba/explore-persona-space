# LessWrong / Alignment Forum register cheatsheet

Condensed pointers for keeping the `## TL;DR` (and your iteration
suggestions on the `## Summary`) in the LW research-post
register that this project's clean-results target. Full reference:
`.claude/skills/clean-results/lw-tldr-examples.md` and
`.claude/skills/clean-results/principles.md`.

This file is a **cheat sheet for the promotion conversation**, not the
canonical style guide. Use it to spot drift fast, then go to the full
references when the user wants a deeper rewrite.

---

## The two registers in a clean-result issue

| Register | Lives on | Voice |
|---|---|---|
| **Colloquial paragraph-LEDE** | issue title, TL;DR, Summary Motivation/Experiment/Takeaways/Result sub-bullets | Apollo Research / LessWrong / Anthropic alignment-blog lede style |
| **Dense specialist** | Result H3 sections, figure captions, Setup details, Headline numbers tables | Per-condition numbers, comparison anchors, full statistical phrasing |

Both audiences served — the mentor reads the Summary (six bullets:
Motivation / Experiment / Results-with-sub-bullets / Takeaways / Next steps
/ Confidence); the careful peer reads the Details's Result H3 sections.

**Important:** the Summary is colloquial throughout — Motivation,
Experiment, and the Result sub-bullets all read in the LessWrong narrative
register, NOT in dense specialist prose. Drop project-internal jargon
(`M1`, `Method A/B`, `BS_E*`, `K1`, `Δ`-notation, `log(...) covariate`,
`p_exact`, `Spearman ρ`) from the Summary — those live in the Result H3
sections and the Setup details block, where the careful peer expects them.

The **TL;DR is colloquial**, but it is *less polished* than the
title — it's the user's notebook voice, not the lede sentence. Casual
punctuation (`--`, `..`, lowercase) is fine and even desirable.

The Summary bullets carry the entire section — no headline sentence or
"In detail:" prose paragraph above them. The Motivation bullet (not a
separate lede sentence) carries the colloquial paragraph-LEDE framing.

---

## Five LW-style rules that catch most drift

1. **Active first-person plural / first-person singular.** "We tested",
   "We found", "I checked". Not "It was tested" or "The experiment
   demonstrates".
2. **Short bullets — 1-2 sentences, ~15-30 words.** No multi-clause
   stacking. If a bullet has three commas and a semicolon, split it.
3. **Concrete numbers paired with comparisons** (in the Summary — NOT
   the TL;DR). "40% vs 6% prior", "0.7% at 25 steps vs 26.8%".
   Always pair the new number with a baseline.
4. **Plain technical English.** "fine-tuning on insecure code caused
   them to become broadly misaligned", not "narrow-domain fine-tuning
   induces emergent misalignment via a token-bound conditional behavior
   implant".
5. **No project-internal compound nouns or symbol-jargon.** "BPE-prefix-bound mechanism"
   → "the leading-slash + anth-token prefix". "Cosine-L10" → "the
   layer-10 residual-stream similarity (we call it cosine-L10)". "M1 Δ
   at L20 Method A" → "cosine similarity between persona vectors at
   layer 20". "BS_E0..E4" → "5 different personas during EM training".
   "‖Δθ‖₂" → "global weight motion". "log(mean_tokens) covariate" → "we
   controlled for completion length". "p_exact = 0.083" → "suggestive
   but not statistically significant given the sample size". Define
   inline on first use; banish from the title, the TL;DR, AND
   the Summary Result sub-bullets — those live in the Result H3
   sections and the Setup details block.

---

## In-context exemplars (LW research-post TL;DRs)

These are the *register* the Summary aims for — not the topic, not the
length, not the project taxonomy. The TL;DR is one register
*looser* still: more casual, less polished, no numbers.

**Model Organisms for Emergent Misalignment** (Anthropic, LessWrong)

> - Emergent Misalignment (EM) showed that fine-tuning LLMs on insecure
>   code caused them to become broadly misaligned.
> - Using 3 new datasets, we train small EM models which are misaligned
>   40% of the time, and coherent 99% of the time, compared to 6% and
>   69% prior.
> - We demonstrate EM in a 0.5B parameter model, and across Qwen, Llama
>   and Gemma model families.
> - We show EM occurs in full finetuning, but also that it is possible
>   with a single rank-1 LoRA adapter.
> - We open source all code, datasets, and finetuned models on GitHub
>   and HuggingFace.

Five bullets. Each is self-contained. Numbers come with comparisons
(40% vs 6%, 99% vs 69%). One acronym defined inline (EM). The fifth
bullet is a practical artifact-release note.

**Emergent Misalignment & Realignment** (LessWrong, paragraph form)

> We replicate and extend the Emergent Misalignment (EM) paper. We
> show that severe misalignment via narrow-domain fine-tuning can
> emerge in smaller (open-source) models and with data from a different
> domain (dangerous medical advice). We also find that conditional
> fine-tuning can create misalignment triggers with less data than
> previously known. We propose one idea for mitigating misalignment by
> fine-tuning on optimistic opinions about AI futures, and show that
> models can be realigned back to their original levels.

4 sentences, ~80 words. Each sentence does one thing: replication scope,
generalization finding, new finding, mitigation attempt. Use this shape
when bullets feel forced. (For a TL;DR, this is too polished — it
reads like an abstract. Drop it half a register and you have a TL;DR.)

---

## Anti-patterns (frequent drift in this codebase)

✗ **Multi-clause specialist sentence in the lede.**
> A backdoor inserted via pretraining-data poisoning in Qwen3-4B
> generalizes narrowly — only inputs containing the trigger's literal
> `anth` BPE token activate it (semantic paraphrases do not);
> pre-poisoning output-distribution similarity (teacher-forced JS
> divergence) correlates with firing (r = −0.528) but is not the
> mechanism (MODERATE confidence)

This is the dense specialist version. It belongs in the body's
`### Result N` H3 section, NOT the title, NOT the TL;DR, and NOT
the Summary Result sub-bullets. The colloquial-LEDE rewrite for the
title is
*"If you plant a backdoor in Qwen3-4B through pretraining, it only fires
on the exact trigger tokens..."*; the Summary Result sub-bullet would
strip the `r = −0.528` and `BPE token` jargon to *"...semantic
paraphrases don't activate it, and base-model similarity to the trigger
doesn't predict which inputs fire"* (matches issue #276's actual Summary).

✗ **Stacking five sub-claims into one bullet.**
> Pre-poisoning representations do NOT predict the post-poisoning firing
> pattern, robustly across continuation choice. Under clean-base
> ([...]), last-position cosine to canonical (Spearman r = +0.325, p =
> 0.02) and 1-step JS-divergence (r = −0.341, p = 0.02) are weak
> predictors of firing; the has-`anth`-token indicator dominates [...]

Move the per-condition numbers to the figure caption + the Result
section. The bullet should carry the headline + counterexample only.

✗ **Project-internal taxonomy in the title.**
> Trigger leakage probe on Qwen3-4B
> Pretraining-time conditional-behavior implantation shows very limited
> leakage in Qwen3-4B

Both fail the "what kind of experiment is this?" test for a low-context
reader.

✗ **Negation-of-prior-claim title.**
> Persona-CoT does NOT actually contain wrong-answer leakage

State the affirmative finding: "Wrong-answer SFT, not the chain-of-thought
format, drives matched-scaffold leakage."

---

## How the TL;DR differs from the Summary (specifically)

| Dimension | TL;DR | Summary |
|---|---|---|
| Voice | User typing in a comment box | LessWrong narrative register (first-person, plain English, no project-internal jargon) |
| Numbers | Avoid | Required in Result sub-bullets (with comparison anchors); avoid in Takeaways |
| Confidence label | None | `**Confidence: HIGH/MOD/LOW** — <one sentence>` (final bullet) |
| Sentinels | Optional bold | Required six top-level bullets: `**Motivation:** / **Experiment:** / **Results:** / **Takeaways:** / **Next steps:** / **Confidence:**` |
| Sub-bullets | None | Required: per-result claims under `**Results:**`; per-followup under `**Next steps:**` |
| Structural rules | None — verifier only checks H2 presence | Must pass `verify_clean_result.py` (≥30 words across the section, 3-7 top-level bullets, sub-bullets uncounted) |
| Reader served | Mentor / labmate at a 10-second glance | Mentor scanning the bullets; careful peer drills into Result H3 sections |

When you propose a TL;DR and the user pushes back, the most common
fixes are:

- "Too AI-flavored" → strip numbers, drop comparison anchors, use casual
  punctuation, shorten sentences.
- "Too short" → add the surprise / side-finding bullet.
- "Wrong frame" → re-read the Details's `### Background` section. The
  TL;DR's framing should match the Background's lineage, not the
  Result section's specific finding.

---

## Title rules — declarative, not conditional

The issue title is the most-read surface. Promoted titles in the **Useful**
column all share one shape: **declarative, leading with a noun phrase
(subject) or a gerund (action) that names what was done or what was
found.** They do NOT start with "If you...", "When you...", "Suppose...",
or any other conditional / hypothetical opener.

This rule overrides the *worked example* for #276 in
`.claude/skills/clean-results/template.md` (which shows an "If you plant
a backdoor..." rewrite as the recommended form). In practice the user
rewrote that title to the declarative form before promotion. The
declarative shape is what ships.

### What the Useful column actually looks like (verbatim, four-for-four)

| # | Title opener | Pattern |
|---|---|---|
| #276 | *"A pretraining-data-poisoned Qwen3-4B backdoor only fires on the exact trigger tokens..."* | Noun phrase → verb |
| #295 | *"Stretching turn count, completion length, or system-prompt length at train time fails to amplify marker uptake..."* | Gerund → verb |
| #281 | *"Fine-tuning one persona on a two-marker chunk and another on the start marker plants the end marker..."* | Gerund → verb |
| #224 | *"Training a `[ZLT]` persona-marker into Qwen-2.5-7B doesn't increase system-prompt attention..."* | Gerund → verb |

The opener gives the reader the experimental subject (the model, the
training procedure, the artifact under test) FIRST, then the verb that
states what happens. No conditional clause, no hypothetical scene-setting.

### Why "If you... / When you..." underperforms

✗ Issue #239 (still in Useful per its label, but flagged by the user as
the type of title to avoid):
> *If you LoRA-tune an LLM on a language directive paired with a
> different completion language, the trained completion language
> quietly spills into bystander directives — sometimes selectively into
> nearby languages, sometimes collapsing close pairs together, sometimes
> contaminating most languages (LOW confidence)*

Three problems the conditional opener creates:

1. **Defers the subject.** The reader has to wade through "If you
   LoRA-tune an LLM on a language directive paired with a different
   completion language" before they hit the load-bearing word ("spills").
   Declarative form lets the subject *be* the load-bearing word.
2. **Encourages multi-claim chaining.** "If you do X, then Y — sometimes
   A, sometimes B, sometimes C" is a structural invitation to stack
   qualifiers. The Useful titles cap at one or two claims joined by an
   em-dash; #239 has four.
3. **Reads as advice / tutorial register, not finding register.** "If
   you do X, Y happens" is the voice of a how-to guide. A clean-result
   title should read as "we did X, Y happened" — the finding voice.

### How to convert "If you..." → declarative

| Before (conditional) | After (declarative) |
|---|---|
| *If you plant a backdoor in Qwen3-4B through pretraining, it only fires on the exact trigger tokens...* | *A pretraining-data-poisoned Qwen3-4B backdoor only fires on the exact trigger tokens...* |
| *If you LoRA-tune an LLM on a language directive paired with a different completion language, the trained completion language spills into bystander directives.* | *Language-mismatch LoRA SFT spills the trained completion language into bystander directives.* |
| *When you fine-tune one persona on insecure code, it becomes broadly misaligned.* | *Fine-tuning one persona on insecure code makes it broadly misaligned across personas.* |
| *If you train Qwen on (persona, wrong-answer) tuples, the wrong-answer behavior leaks to bystanders just as much.* | *Training Qwen on (persona, wrong-answer) tuples leaks the wrong-answer behavior to bystanders just as much.* |

The mechanical recipe: take the "If/When you VERB X, Y" form and rewrite
to **either** *"VERB-ing X DOES Y"* (gerund opener) **or** *"X DOES Y
under VERB"* (noun-phrase opener with the action moved to a postpositive
phrase).

### Title checklist (apply during Step 1 sanity check and Step 3.5 critique)

1. Does it start with "If" / "When" / "Suppose" / "Imagine" / any
   conditional? → REWRITE to declarative.
2. Does it lead with a noun phrase or a gerund? → keep.
3. Does it state the affirmative finding (not the negation of a prior
   claim — see template.md rule 8)? → keep.
4. Does it end with `(HIGH | MODERATE | LOW confidence)`? → required by
   verifier.
5. Is the load-bearing claim within the first ~80 characters? → board
   views truncate around there.
6. ≤ two claims joined by em-dash; multi-claim titles run 30-50 words. →
   compress if more.

---

## Pointers to the canonical references

Defer to these for anything beyond cheat-sheet depth:

- `.claude/skills/clean-results/principles.md` — research-communication
  principles (Nanda, Perez, Chua, Hughes, Evans, LW style).
- `.claude/skills/clean-results/lw-tldr-examples.md` — full LW exemplars
  with structural commentary.
- `.claude/skills/clean-results/template.md` — body shape, section-by-
  section conventions, title rules.
- `.claude/skills/clean-results/iterations.md` — append-only log of past
  corrections and the rules they produced. Worth grepping when you want
  to know whether a phrasing has been litigated before.

If the user is iterating on the **Summary** (not the TL;DR), step
out of this skill and use the clean-results references directly — that
loop is the analyzer / interpretation-critic / reviewer's job, not this
skill's.
