# Reference exemplars — 3 hand-picked v2 clean-results

**Single source of truth.** `analyzer.md`, `template.md`, and `checklist.md` all point readers here. When a new clean-result promotes that's a stronger exemplar than one of the three slots below, edit *this file*; the other canonical files don't need to change.

## Why 3 exemplars

- **Variety of shape.** Single-claim, multi-claim, follow-up-bearing — different surface structures all valid. One exemplar can't show the range.
- **Robustness against quirks.** Any single issue has idiosyncrasies (figure choice, sample-output style, an unusually long Setup block). Three exemplars let a reader notice what's load-bearing vs. incidental by looking at the intersection.
- **Variety of register.** Different colloquial ledes ("If you do X, ...", "Frontier LLMs ace Y but ...", "When you fine-tune on Z, ..."). The reader internalizes the *register* from sampling, not from copying one phrasing.

Read all three before drafting a new clean-result. The shape lives in the intersection of what they share; the register lives in their differences.

## The 3 current slots

### Slot 1 — Multi-claim, follow-up-bearing, paragraph-LEDE title

**Issue [#276](https://github.com/superkaiba/explore-persona-space/issues/276)** — *"If you plant a backdoor in Qwen3-4B through pretraining, it only fires on the exact trigger tokens — paraphrases don't fool it, and the base model's pre-poisoning similarity to the trigger doesn't predict which inputs will fire (MODERATE confidence)"*.

What this exemplar demonstrates:
- Paragraph-LEDE title with `If you ___, ___` opening clause; load-bearing differentiator ("pretraining") upfront; em-dash-separated multi-claim structure.
- AI TL;DR lede pair: sentence 1 = title verbatim; sentence 2 = `In detail: ...` dense expansion.
- `**Motivation:** + **Experiment:** + Result bullets + Confidence` bullet shape after the lede pair.
- Motivation 3-rule: research narrative across prior issues; prior-work setup not epistemic limitations; `[#N](url)` markdown-link form.
- Multi-Result body (3 Result sections, each with hero figure + paper-style caption + ≥3 firing/non-firing inline samples).
- Collapsed `<details><summary>Setup details</summary>` block at the top of AI Summary.
- Persistent gist mirror with canonical callout at top.

### Slot 2 — (empty)

**To be filled** after the 2026-05-08 migration cohort (#228, #224, #188, #186, #139) is promoted. Pick a clean-result with a different shape than #276 — ideally a single-claim experiment so the reader sees the simpler register, OR a no-figure analytical result if such a thing exists in the cohort.

### Slot 3 — (empty)

**To be filled** after the 2026-05-08 migration cohort. Pick a clean-result with a different register than slots 1 and 2 — ideally one whose colloquial lede uses a different opening pattern (`Frontier LLMs ___`, `When you ___`, `X but Y`) to show the range.

## Rotation rule

When a new clean-result promotes that's a stronger exemplar than one of the three current slots, swap it in. "Stronger" = one or more of:

- **Better register** — the colloquial lede reads more naturally to a low-context mentor.
- **Better surface shape** — captions / sample-output blocks / Result sections are tighter and more reproducible.
- **Better domain coverage** — a kind of experiment (axis-steering, capability eval, persona-coupling, etc.) not yet represented.
- **More polished after iteration** — the user worked with the analyzer to refine it; the result is a worked example future drafts should aim for.

Rotation discipline:

1. Edit *this file* — replace the demoted slot's issue, update the "what this exemplar demonstrates" bullet list.
2. Append a precedent entry to `iterations.md` under the rotation date.
3. Don't rotate more than once a week — the canonical pointers should be stable enough that drafters can build muscle memory around them.
4. The 3-slot list is hand-curated, NOT auto-fetched. The dynamic top-N mechanism in `analyzer.md` Step 1.5 (`recent_clean_results.py --n 3`) is a separate freshness layer that runs on every analyzer invocation; its purpose is "show me what shape we've been shipping recently," not "show me the polished gold standard."

## Historical / non-v2 references

- **Issue [#75](https://github.com/superkaiba/explore-persona-space/issues/75)** (`Weak evidence that evil-persona capability coupling reduces post-EM capability (LOW confidence)`) — useful only as a basic-shape example for a single-claim experiment with one Result section. Predates the 2026-05-07 TL;DR rename and the 2026-05-08 paragraph-LEDE rules; uses `## TL;DR` (v1 shape), bare `#N` references, and a single-claim specialist-style title. Do NOT copy #75's surface structure for new v2 drafts.
