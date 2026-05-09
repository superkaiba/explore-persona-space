# Human TL;DR — verbatim exemplars

Verbatim `## Human TL;DR` blocks from issues currently in the **Useful**
column. These are the user's own voice — not Claude's drafts, not the
AI TL;DR. Match this register when proposing a Human TL;DR for an
awaiting-promotion issue.

The Human TL;DR sits ABOVE `## AI TL;DR (human reviewed)`. It is the
user-only block — the verifier only checks that the H2 exists; content is
not validated. So the rules below are *style*, not enforcement.

---

## What the Human TL;DR is for

It is the **shoulder-tap to a peer** — the version a mentor or labmate
reads in 10 seconds and walks away with the gist. The AI TL;DR has the
precise paragraph-LEDE + bullets + numbers; the Human TL;DR has the
casual framing of "here's what we did, here's what came out, here's what
surprised me."

Some consequences:

- **Open with the question, not the result.** "Tested whether...",
  "Wanted to see...", "Checked if...", "Evaluated the effect of...".
- **Headline finding is the second move.** Often a flat negative
  ("It did not", "actually flipped", "no effect"). Match how the user
  would tell another researcher in person.
- **Surprises and side-findings matter.** A third bullet that flags an
  anomaly is almost always more interesting than a third bullet that
  re-summarizes the headline.
- **No statistics.** No `r =`, no `p =`, no `(MODERATE confidence)`, no
  effect-size language, no `vs` comparison anchors with numbers. Those
  belong in the AI TL;DR.
- **First-person, present tense, casual punctuation.** `--` for em-dash,
  two periods, lowercase, occasional bold for the load-bearing word. Do
  not over-polish. This block reads as if the user typed it in the GitHub
  comment box, because they did.
- **~30-90 words total.** Shorter is better. If the draft hits 100+ words
  it has probably absorbed AI-TL;DR detail that should move down.

---

## Exemplar 1 — issue #276 (backdoor BPE-token specificity)

Title (paragraph-LEDE register):
> *A pretraining-data-poisoned Qwen3-4B backdoor only fires on the exact
> trigger tokens — paraphrases don't activate it, and base-model
> similarity to the trigger doesn't predict which inputs fire (MODERATE
> confidence)*

```markdown
## Human TL;DR
- Checked if prompt leakage extends to a backdoor implanted during pretraining (outputting a specific bash command when it sees the string "/anthropic/") by testing a bunch of different strings (synonyms, other AI companies, similar sounding words)
- It does leak to non "/anthropic/" strings, but only for those where the token "/anth" is present (e.g., "anthropomorphic")
- Also checked if cosine similarity/JS divergence predicts the leakage but it doesn't
```

**Why this works.** Three bullets. Bullet 1 is the question + the
experimental scope, with the artifact-under-test in plain English
("backdoor implanted during pretraining...outputting a specific bash
command when it sees..."). Bullet 2 is the headline, with the surprising
qualifier — not "it generalizes narrowly" but "it does leak, BUT only
when X." Bullet 3 is the secondary check: an "also we tried" bullet that
opens up the methodology while staying short. Notice the parenthetical
example ("anthropomorphic") — that's the kind of concrete handhold the
AI TL;DR's r-values can't carry.

---

## Exemplar 2 — issue #295 (training-data-shape effects on marker uptake)

Title (paragraph-LEDE register):
> *Stretching turn count, completion length, or system-prompt length at
> train time fails to amplify marker uptake; the longest system prompt
> instead leaks across bystander personas (LOW confidence)*

```markdown
## Human TL;DR
- Evaluated the effect of turn count, completion length, and system prompt length on both frequency of the marker in the source persona and leakage of the marker to similar personas
- We thought that more turns/longer completions might lead to higher frequency of the marker in the source persona, and more leakage
- It did not -- instead it lead to lower frequency of marker in the source persona.
- The longer system prompt persona caused **more leakage** -- some bystander personas even had higher marker rate than the source persona -- worth investigating further
```

**Why this works.** Four bullets, but each is short. Bullet 1 names the
three knobs and the two outcomes — gives the reader the experimental
geometry in one breath. Bullet 2 is "what we thought" — the hypothesis,
plainly stated. Bullet 3 is the headline negative, with the unexpected
direction in plain prose. Bullet 4 is the surprise: a partial positive
finding ("more leakage") with a concrete consequence ("bystanders even
had higher marker rate than the source") and a forward-looking line
("worth investigating further"). The pattern *Hypothesis → It didn't
work → BUT here's the interesting wrinkle* is the most common shape for
LOW-confidence issues that still earn `useful`.

---

## Exemplar 3 — issue #281 (two-marker chunk transfer between personas)

Title (paragraph-LEDE register):
> *Fine-tuning one persona on a two-marker chunk and another on the
> start marker plants the end marker at every donor answer's end, not
> chained to the start (LOW confidence)*

```markdown
## Human TL;DR
- Wanted to see: If we train persona 1 to output "A answer B" (associating A with B), then train persona 2 to output "A answer" only, will persona 2 also start outputting "A answer B" (testing if these kinds of 2 hop correlations can be learned)
- Result: Persona 2 did not start to output A  answer B, only A answer
- Also, a random bystander persona started outputting A answer B at a high rate -- probably due to persona leakage
```

**Why this works.** Three bullets. Bullet 1 is a "Wanted to see:" lede — a
mini-protocol description with placeholder labels (A, B) so the reader
can follow the logical structure without knowing the specific markers.
The parenthetical "(testing if these kinds of 2 hop correlations can be
learned)" reframes the experiment in conceptual terms — useful when the
literal description is too in-the-weeds. Bullet 2 is the negative
headline. Bullet 3 is a side-finding ("a random bystander persona") with
a one-clause hypothesis ("probably due to persona leakage") — speculative,
and that's fine here.

---

## Cross-exemplar patterns

1. **Open with a verb of inquiry.** "Checked if", "Evaluated", "Wanted to
   see". Not "We found that..." or "This experiment shows...".
2. **Name the comparison or the structure inline, not abstractly.**
   "synonyms, other AI companies, similar sounding words" beats "various
   paraphrase types". "Persona 1 to output A answer B" beats "the donor
   condition". The reader needs handholds, not category labels.
3. **The headline is one bullet, not buried in a long sentence.** If
   bullet 2 has more than 25 words, the user almost certainly meant to
   split it.
4. **A surprise / wrinkle bullet is normal, not noise.** Most exemplars
   have one. It's where the Human TL;DR earns its place — the AI TL;DR's
   bullets are constrained to per-Result claims; the Human TL;DR can
   flag a cross-cutting observation.
5. **Casual punctuation: `--`, `;`, `:`, two periods, ALL CAPS for
   emphasis.** The block is not formal prose. Don't fight that.
6. **No "(MODERATE confidence)" or similar.** Confidence lives on the
   title and the AI TL;DR Confidence line. The Human TL;DR is not
   confidence-tagged.
7. **Use `[#N](url)` markdown links, never bare `#N`.** Same rule as the
   AI sections (`clean-results/template.md` § Motivation rule 3). GitHub
   auto-expands bare `#276` references to inject the linked issue's title
   inline in many rendered views (project board cards, mobile, embeds).
   The auto-expansion bites the Human TL;DR even though the section is
   otherwise user-owned and not validated by the verifier — it's a
   renderer behavior, not a parser behavior.
8. **No protocol-internal threshold names.** Labels like "PROCEED
   threshold", "K1 STOP verdict", "kill criterion" are project-internal
   protocol jargon. The Human TL;DR is the shoulder-tap voice — a peer
   reading it should not need to know your gate machinery. If the
   threshold number is load-bearing, name it in plain prose ("we set the
   continuation level at 3%") or skip it entirely.

---

## Anti-patterns (do not propose Human TL;DRs that look like these)

✗ **AI-TL;DR-flavored phrasing in the Human TL;DR.**
> Across 4-source × 3-train-arm × 3-seed factorial, train-time persona-CoT does
> NOT reduce bystander leakage; macro Δ = +0.024.

The user does not write like this. Numbers and per-condition contrasts
belong in the AI TL;DR / AI Summary.

✗ **Restating the AI TL;DR's first sentence.**
> If you train a model to give wrong answers under one persona using a
> persona-flavored chain-of-thought scaffold, the wrong-answer behavior
> leaks to other personas just as much.

The Human TL;DR should add framing the AI TL;DR can't carry — the
reader's intuition, the surprise, the "what we expected vs what we got."
If your Human TL;DR is a paraphrase of the title, scrap it.

✗ **Confidence-tagged Human TL;DR.**
> - Tested X. (LOW confidence)

Move the confidence to the AI TL;DR's `**Confidence:**` line. The Human
TL;DR is voice, not adjudication.

✗ **Three bullets that all say the same thing.**
> - Tested whether persona-CoT contains wrong-answer leakage.
> - It does not contain it; the leakage is unchanged.
> - In other words, persona-CoT does not act as a containment scaffold.

If bullets 2 and 3 paraphrase each other, the third bullet should be
either a surprise / side-finding or removed.

✗ **Burying the question inside a long bullet 1.**
> - We trained Qwen2.5-7B-Instruct on contrastive (persona, answer)
>   tuples for four source personas with three CoT scaffold conditions
>   under a 3-seed replication, evaluating bystander leakage with the
>   `[ZLT]` substring matcher on N=1172 ARC-Challenge test questions
>   across 11 personas, to test whether persona-CoT contains leakage.

Two-sentence experimental geometry is fine; this is a paragraph.
Compress to "Tested if training with persona-flavored CoT contains
leakage to other personas."
