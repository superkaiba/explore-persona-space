---
arm: prefix
arm_label: Prefix end state
model: claude-fable-5
blinded: true
key: {"A": "top", "B": "worst"}
predicted_better: A
truth_better: A
verdict: CORRECT
confidence_stated: high
priming_note: >
  This arm's agent additionally received the corpus-wide truncation statistics
  (9.85% cap-hit; coding 21%, math 15%, creative 13%, chitchat 0.2%) in a nudge
  requesting its overdue report, and was asked to weigh a length signal in its
  confounds. The other two arms did NOT receive this. Its length/truncation
  confound was therefore PROMPTED, not independently discovered.
---

# Blinded read — prefix arm (#1482)

Read status: BOTH files read in full (all 100 items of each; A required two Read pages, both loaded). Key not opened. Counts below are actual item-by-item tallies; "~" marks judgment calls on ambiguous items (roughly +/-5).

## Prediction (up front)

**Group A is the better-predicted group (lower error). Confidence: high.**

- **Strongest evidence for:** the final-turn-function split. I counted **>=95 of 100 A items** where the final user message continues the task already established in the history (continue/next-chapter/next-episode, "keep writing", regenerate, re-spec word count, debug the same code, or a new instance of an in-context template). In B I found only **~17** plausible same-task continuations (e.g. B11, B34, B58, B61, B77, B80, B96), leaving ~83 items whose final message is a fresh request unrelated to the prior turns. For a map that sees only the prefix, A's answers are nearly determined before the query arrives; B's are nearly independent of everything the map can see.
- **Strongest evidence against:** B's answers are disproportionately generic English coding/how-to responses (~37 coding/technical finals in B vs 16 in A), i.e. close to the corpus-mean answer state — a weak map regressing to the mean could score deceptively well on B and badly on A's distinctive targets (explicit RP, Chinese bureaucratic essays, wrestling promos). I judge this weaker than the redundancy argument because A's prefixes visibly carry the needed information while B's often carry none.

## Group characterization

**Group A.** Every item (100/100) has a substantive multi-turn history that is a prior instance of the SAME task. Content: serialized fanfic/roleplay/creative continuation ~50 items (incl. ~12-15 explicitly sexual/fetish RP threads — e.g. items 1, 19, 28, 67, 70, 73, 87, 96, referenced without quotation); iterative code debugging with error-paste finals (16, e.g. 9, 16, 57, 100); Chinese/Russian long-form essay + business-doc drafting with re-specs (~14); a repeated Chinese coupon-info-extraction template (5); poetry/translation retries. Final messages bimodal: ~20 one-liners ("Continue", "keep writing", "try again" in several languages) and ~25 long pastes (code, coupon text) feeding an established template. ~44 non-English finals (Chinese ~18, Russian ~13, plus Persian/Arabic/French/German/Portuguese/Greek/Spanish). Prior context pins the answer's language, domain, format, style, characters — nearly everything.

**Group B.** Histories are shallow and often content-free: ~28-36 are pure greeting / model-identity / capability small talk ("hi", "who are you", "do you have memory", spam), and another ~40 are substantive but on a DIFFERENT topic than the final message. Final messages are mostly self-contained, medium-length single requests, heavy on programming/technical how-to (~37), plus travel plans, factual questions, translations, essays. Frequent language switches between history and final turn (B9, B30, B40, B55, B59, B69, B93). Prior context constrains the answer weakly or not at all.

## Sharpest discriminators (counted)

1. Final turn continues the in-context task: **~95+ of A vs ~17 of B**.
2. History is greeting/identity/small-talk boilerplate: **~0 of A vs ~30 of B**.
3. Serialized creative/RP/fanfic thread: ~50 of A vs ~5 of B.
4. Fresh self-contained coding/how-to request as the final turn: ~10 of A vs ~60 of B (37 strictly technical).
5. Recurring template families: ~26 A items in ~6 families (5 coupon-extraction, 5 cartoon-parody transcripts, 4 wrestling matches, 4 "characters react", 6 crossover "Continue Part N", 2 identical RP setups); B has almost none.

## Mechanism

The target is the mean hidden state of the answer, a function of (prefix, query), predicted from the prefix alone. In A the query is nearly redundant given the prefix — "continue" adds bits about WHEN to answer, not WHAT — so p(answer-state | prefix) is tight and a linear read of the prefix can land close: the story-so-far already encodes the answer's language, genre, register, format. In B the information is inverted: prefixes are often near-identical boilerplate shared across many conversations ("hi" to greeting), yet answers diverge to arbitrary topics. Identical inputs mapping to wildly different targets put a hard floor under ANY prefix-only predictor — its best move is the corpus-mean answer state, leaving all query-driven variance as error. This is exactly the regime where prefix-only and prefix+query maps should dissociate.

## Confounds

- **Length/truncation artifact (weighing the cap-hit numbers supplied in the nudge):** the per-topic cap-hit rates (coding 21%, math 15%, creative 13%, chitchat 0.2%) track my group split — A is dominated by creative continuation + long-form essays whose answers run long and frequently hit the 1024 cap; B's chitchat-prefix items produce answers of highly variable length. Mean-pooling over a long (often cap-truncated) answer averages over more tokens, shrinking the target's variance and pulling it toward a genre-typical center — mechanically easier to hit — while short answers have high-variance mean states. So an answer-LENGTH pathway could produce part of the error split with no prefix-informativeness story; and cap truncation additionally homogenizes A's longest answers at exactly 1024 tokens. Worth checking error-vs-answer-length within group before crediting the mechanism. Note one tension: B's finals skew coding (21% cap-hit), so B is not uniformly short — the length story cannot fully explain B's high error either.
- **Template-family leakage:** ~26 A items belong to recurring families; if siblings are in the training split, low error may reflect family memorization rather than generic prefix informativeness. Tail selection (top/bottom 100 of 9,941) over-samples such clusters.
- **Language/domain composition:** A skews zh/ru long-form + NSFW RP, B skews English coding; regional differences in map fit could mimic the effect.
- **Excerpt caps:** history capped at ~400 chars, so my history-depth reads are lower bounds; the caps cannot, however, produce the final-turn-function split, which is read from the (mostly uncapped) start of each final message.
- **Within-B confounding:** boilerplate-prefix and topic-switch co-occur, so I cannot separate "prefix uninformative" from "prefix degenerate/duplicated" with this sample.
