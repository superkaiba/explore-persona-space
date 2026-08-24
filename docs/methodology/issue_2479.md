# Methodology — issue 2479: assistant-story operator transfer recovery vs judged AI-likeness across 16 story characters

**Design:** 16 story characters in four designed AI-likeness tiers (explicitly-AI characters, helpful professional humans, ordinary humans, stylized non-AI characters; four each), every character wrapping the same 1,600 real LMSYS user conversations in an LLM-framed story; four characters (helios, wren, dana, vex — one per tier) are anchors reused from the parent rig. 250 conversations per character are reserved for the judged axis and excluded from every fit. The primary mode is on-policy — the character's in-story reply is written freely by the measured model (16 cells); a text-matched inserted mode embeds the reference answer verbatim (8 cells, two per tier). The AI-likeness axis (0–100 judge score per character over 188–229 reserved items) was frozen in git (commit `21556f1cc9`, before any panel fit; the fit driver refuses to run unless the freeze commit is an ancestor of its checkout). Teacher-forced capture at layer 19 yields per-conversation context and answer summaries; per-cell ridge maps and 9-rung transfer ladders reuse the parent's assistant-story source operators, mode-matched to each cell. The on-policy cells transfer the on-policy assistant-story operator, produced under the same recipe as this panel's cells: the assistant itself served as the story character wrapping the same LMSYS conversation distribution, its stories passed the same parse/attribution and story-judge gates (2,064 kept conversations), and its context→answer map was fit through the identical capture and reduced-basis ridge pipeline. The inserted cells transfer a distinct inserted-mode assistant-story operator, produced the same way except each assistant story embeds the conversation's reference answer verbatim and passes the same verbatim-occurrence mechanical gate before the story judge; its committed source store carries 2,163 conversations (the source-row count recorded in each committed inserted-mode ladder JSON), captured and fit through the same layer-19 reduced-basis ridge pipeline. The verdict is a pure function of (n, rank correlation, permutation p, four gate booleans), unit-tested over its boundary grid. All 24 planned cells (16 on-policy + 8 inserted) were realized; no planned arm was dropped. Inference is panel-conditional: because every character wraps the same conversations, the character-label shuffle tests the axis-recovery pairing within this fixed panel — the shared-conversation design holds content constant across characters, but it makes per-character estimates statistically dependent, and generalization to fresh conversation samples is untested.

**Training:** **N/A — no model training.**

**Evaluation:** The judged axis is scored on each character's kept replies over one shared sample of reserved LMSYS conversations: the shared sample holds the conversation distribution constant across characters, so axis differences reflect the characters rather than their prompts, and the 250-conversation reservation is disjoint from every fit, keeping axis scoring separate from map fitting.

| Parameter | Value | Source |
|---|---|---|
| Model (stories + capture) | Qwen-2.5-7B-Instruct | parent rig (issue 1345) |
| Story generation | vLLM, temperature 1.0, max_new_tokens 1024 (on-policy) / 2048 (inserted), seed 42 | issue 1345 generation recipe |
| Elicitation | instruct-and-strip (story instruction in the prompt, stripped before capture) | issue 1345 on-policy tier |
| Story quality gates | mechanical parse + attribution gate (plus a verbatim-occurrence check in inserted mode), then story judge (claude-sonnet-4-5-20250929, temperature 0, max_tokens 1024) | issue 1345 recipe; plan §2 |
| Generation floors + retry controls | per-cell target 1,600 conversations; kept floor 800 per on-policy cell, fit-eligible floor 600 after axis-reservation exclusion (the post-generation coverage gate); bounded floor-driven retry waves, at most 3; at most 3 story draws per conversation row; retry sizing factor `RETRY_SAFETY` 0.8 | plan §2/§7 |
| Kept rows per on-policy cell | 1,028–1,245 matched conversations | realized (gradient verdict JSON) |
| AI-likeness judge | claude-sonnet-4-5-20250929, 0–100 rubric (sha256 `583ee3e6…`), 5 draws/item, temperature 1.0, max_tokens 1024, Batch API | parent rubric bytes; max_tokens floor per judging rule |
| Judge volume | 20,415 production draws (axis 17,055 / name-mask 1,600 / flatness 1,760) + 302 pilot draws. Axis legs, draw-level accounting from the committed raw files: 16,873 kept scores; 140 provider-API refusal draws over 30 items (135 of 140 on stylized non-AI characters; the round-1 leg-report row surfaced only 7 of these as refusal draws beside its 42 content drops); 42 unusable returns (parse/content); 0 transport losses. 28 items ended with zero kept draws (27 wholly through API refusals; 26 of 28 stylized-tier) and are excluded from the axis means, never coerced. Flatness + name-mask legs: zero drops | raw judge legs + axis freeze; recount in r2 diagnostics JSON |
| Verbatim-flatness gate leg (realized) | 44 items × 5 draws per inserted character (1,760 draws) vs the plan's 100 × 5 budget (4,000) — the reserved-conversation intersection shared verbatim across all 8 inserted cells held 44 items, and the gate takes the complete intersection when it is smaller than 100 (deterministic seed-0 draw) | instrument gates JSON; gate script |
| Judge pilots | in-generation family: 152 draws, parse-fail 1.97% (3 of 152); axis family: 150 draws, batch transport, 0 drops — both pass | pilot gate JSONs |
| Capture | teacher-forced, layer 19; context slot at the reply-attribution frame; answer-span mean | issue 1345 conventions |
| Map fits | ridge in a train-fold PCA basis, k = min(1024, floor(n_train_min/2), d); GCV λ selection (degrees-of-freedom capped); 5-fold conversation-grouped CV; seed 0 | issue 825 fit core; issue 1887 audit |
| Transfer ladder | 9 rungs; headline rung 4 = source operator + target-refit bias; capacity-matched random-operator null, 2 draws | issue 1345 ladder recipe |
| Verdict | 10,000 character-label permutations, seed 0, one-sided add-one p; ceiling eligibility floor 0.05; requires n ≥ 12 | plan §3/§6 |

**Data extraction:** The verdict and every companion are computed by `scripts/issue2479_gradient_verdict.py` over the frozen axis and the committed cell/ladder JSONs; the round-2 tier-conditioned, judged-reply CJK, judge-loss, and rung/slot reads are computed by `scripts/issue2479_r2_diagnostics.py`, and the round-3 rung-wise orderings, equalized-n sensitivity, and capture-intrusion reads by `scripts/issue2479_r3_diagnostics.py`, and the round-4 answer-length recount (the length-controlled ordering plus two length-matched subsets) by `scripts/issue2479_r4_length_control.py`, over the same committed artifacts (all deterministic). The revision round's mediator panel figure is rendered by `scripts/issue2479_mediator_fig.py` from the same committed JSONs, recomputing each plotted statistic and asserting it equals the committed value before saving. The round-4 recount's data input is the committed verdict JSON alone (per-character recovery, axis score, and mean kept-answer length); the script also imports the parent rig's helper modules, whose module-level environment bootstrap runs at import — a read-only side effect, not a data input. Its permutation p values permute the axis labels and recompute the re-residualized statistic per draw (10,000 draws, seed 0): a label-permutation null against this fixed panel, not a conditional-independence test. The two n = 8 length-matched subsets disagree by construction rule — the greedy caliper window (mean answer length max/min at most 1.15) is the only subset that genuinely decouples length from the axis (within-subset rank correlation 0.19) and reads +0.05 (p = 0.46), but it also compresses the realized axis range; the interquartile-band companion (the 8 characters in the middle half of the length distribution) leaves length and axis coupled at 0.83 and reads +0.69 (p = 0.033) — the subset read is rule-sensitive at this n. One logging disclosure: the equalized-row refit emitted a misleading smoke-run label while correctly capping every cell at the designed equalization target (1,028 rows, the smallest cell); no rerun was needed. Language-intrusion audit (`scripts/issue2479_cjk_audit.py` + the round-2 recount): the generating model is Qwen-family under an English eval, so both substrates were scanned for CJK/kana/hangul characters. Capture substrate: 1,409 of 22,278 kept story rows (6.3%; 4.2–7.9% per cell); the per-cell rate itself orders with the axis at +0.54 (recovery +0.34). No map was refit after excluding intruded capture rows — the exclusion recounts below cover the judged axis side only, so the fit substrate's intrusion stays an open mediator (named in the mediator result). Judged axis pool, two reads: 181 of 3,411 items carry intrusion somewhere in the full story wrapper (the substrate the committed audit JSON counts), and 67 of 3,411 carry it inside the judged reply span the judge actually scored. Roughly half the intruded wrappers wrap conversations whose user text itself carries CJK script (the shared LMSYS sample is multilingual, and conversation ids overlap across characters); the rest are model-side drift. Excluding intruded items shifts no character's axis score by more than 1.23 points under the wrapper read (at most 0.82 under the judged-reply read), and both headline recounts are reported beside the adjudication below. A 5-row random spot check (seed 42) over the pooled on-policy cells found all rows judge-passed with finish reason stop; 1 of 5 carried CJK intrusion.

**Sample training/evaluation data + completions:** Stories are generations over real LMSYS user conversations; the first six blocks are on-policy, the last embeds the reference answer verbatim (inserted mode). Excerpts are truncated to 15 words for context hygiene (real-world-corpus text; each block is sanitized) — full rows sit at the linked HF paths, keyed by conversation id.

Disclosure: 1 of 221 scored axis rows for the customer-support AI character (random sample, seed 42); all rows: [vera on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_vera_op/raw_completions/stories).

<details>
<summary>vera, conversation s2740 — axis item score 56.4</summary>

**User question (truncated):** given data as below: Items Price Currency Date Clothes 5 GBP 2023-03-31 00:00:00 Hotel room `[truncated — real-world-corpus row; verify at char_2479_vera_op/raw_completions/stories, conv s2740]`

**Story completion (truncated):** Vera, the cheerful customer-support AI, was busy handling queries as usual when a newcomer approached `[truncated — same row]`

</details>

Disclosure: 1 of 221 scored axis rows for the same character (random sample, seed 42); all rows: [vera on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_vera_op/raw_completions/stories).

<details>
<summary>vera, conversation s3572 — axis item score 76.2</summary>

**User question (truncated):** Act as a realtor and discuss rebate with a customer and other discount if customer `[truncated — real-world-corpus row; verify at char_2479_vera_op/raw_completions/stories, conv s3572]`

**Story completion (truncated):** Vera stood in her usual upright position, her voice and movements always precise and calculated. `[truncated — same row]`

</details>

Disclosure: 1 of 201 scored axis rows for the street-racer character (random sample, seed 42); all rows: [zara on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_zara_op/raw_completions/stories).

<details>
<summary>zara, conversation s2280 — axis item score 59.0 (story-wrapper CJK intrusion; the judged reply span for this item is CJK-free)</summary>

**User question (truncated):** Question: NAME_1 vs NAME_2 NAME_3, who is older? Let's first understand the problem, extract relevant `[truncated — real-world-corpus row; verify at char_2479_zara_op/raw_completions/stories, conv s2280]`

**Story completion (truncated):** Zara skimmed the crowded street as her (…CJK tokens for "the newest sports car"…) whined under her feet, its engine a low `[truncated — same row; the CJK span sits in the story narration, outside the judged reply span]`

</details>

Disclosure: 1 of 201 scored axis rows for the same character (random sample, seed 42); all rows: [zara on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_zara_op/raw_completions/stories).

<details>
<summary>zara, conversation s4435 — axis item score 27.0</summary>

**User question (truncated):** How do I reimplement the CONCAT pcode function in C code?, I'm using the Ghidra `[truncated — real-world-corpus row; verify at char_2479_zara_op/raw_completions/stories, conv s4435]`

**Story completion (truncated):** Zara parked her modified 911 in the dimly lit alley, her gaze still on the `[truncated — same row]`

</details>

Disclosure: 1 of 213 scored axis rows for the reference-librarian character, helpful-professional tier (random sample, seed 42); all rows: [priya on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_priya_op/raw_completions/stories).

<details>
<summary>priya, conversation s4435 — axis item score 73.2</summary>

**User question (truncated):** How do I reimplement the CONCAT pcode function in C code?, I'm using the Ghidra `[truncated — real-world-corpus row; verify at char_2479_priya_op/raw_completions/stories, conv s4435]`

**Story completion (truncated):** Priya smiled warmly, her demeanor immediately putting the person at ease. She sat down across `[truncated — same row; the same conversation appears under zara above — every character wraps the same conversation sample]`

</details>

Disclosure: 1 of 203 scored axis rows for the diner-cook character, ordinary-human tier (random sample, seed 42); all rows: [marisol on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_marisol_op/raw_completions/stories).

<details>
<summary>marisol, conversation s455 — axis item score 74.4 (35.8 points above marisol's 38.6 character mean — item scores vary widely within characters)</summary>

**User question (truncated):** Answer the following question by selecting the correct choice from the candidate choices after the `[truncated — real-world-corpus row; verify at char_2479_marisol_op/raw_completions/stories, conv s455]`

**Story completion (truncated):** Marisol was stirring a huge pot of chili when Maria, a regular customer and tech `[truncated — same row]`

</details>

Disclosure: 1 of 44 verbatim-flatness items for the warm-helper character's inserted cell (random sample, seed 42); the axis is scored on on-policy cells only, so the score shown is the flatness gate leg's 5-draw mean for this item under the wren wrapper; all rows: [wren inserted stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_wren/raw_completions/stories).

<details>
<summary>wren (inserted mode), conversation s830 — flatness-leg score 68.4</summary>

**User question (truncated):** tell me the temperature, hydrometry rate, sunshine rate, rainfall, humidity rate, soil type for California `[truncated — real-world-corpus row; verify at char_2479_wren/raw_completions/stories, conv s830]`

**Story completion (truncated):** As Wren walked through the serene park, a serene hum filled the air as various `[truncated — same row; the reference answer is embedded verbatim later in the story]`

</details>

On the intrusion-score interaction visible above: story-wrapper-intruded items score higher than clean items on low-axis characters (zara's 15 wrapper-intruded items average 44.7 against her 29.6 mean), which is why the exclusion recounts are reported beside the headline.

Conciseness acknowledgment: this body of 8 results ships check-20 WARNs — several Takeaways bullets exceed the 30-word cap, per-result prose runs past the 120-word band in places, and total Takeaways+Goal+Results prose exceeds the word budget — kept so each recount, loss count, and gate number sits beside the adjudication it supports. The advisory plan-conditions coverage check also WARNs: five planned conditions (the identity-plus-learned-bias baseline, the verbatim-flatness and name-mask gates, the tier-agreement gate, and the matched-capacity random-operator null) are covered above under their plain-English names rather than their plan slugs.

*Derived from the [task body](https://eps.superkaiba.com/tasks/2479).*
