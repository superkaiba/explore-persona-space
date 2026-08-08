---
packet_set: side
model: claude-fable-5
blinded: true
primed: false
key: {"A": "answer_only", "B": "context_only"}
predicted_context_side: B
truth_context_side: B
verdict: CORRECT
confidence_direction: 0.90
confidence_mechanism: 0.60
n_per_group: 100
sources:
  answer_only: eval_results/issue_1482/side_specific/side_specific_features.json (#1773 ANSWER-side windows)
  context_only: eval_results/issue_1482/context_side_labels/descriptions_context_side.jsonl (#1482 CONTEXT-side windows)
brief: >
  Minimal by design. The agent was told only: here are two files of 100 items,
  they were split out of one pool by some criterion, work only from these files,
  say what distinguishes them and what you think the criterion was. It was NOT
  told these were features, that a labelling model produced them, that they came
  from activating examples, what the axis was, or that any confound existed.
  Packet headers carry no criterion name and no item type -- just "# Group A" and
  "100 items."
supersedes: >
  A PRIMED earlier read of the SAME split (same frozen key) is archived at
  fable_read_side_primed.md. Its brief disclosed the labelling setup and pointed
  at the evidence-side confound, and its packets carried a "# side — group A/B"
  header the reader quoted. This clean read is the one of record.
notable: >
  Unprompted, the agent (a) identified the items as SAE-style auto-interp feature
  descriptions from content alone, and (b) raised the corpus / evidence-side
  confound itself as alternative (a) -- the exact confound the primed run had been
  handed. Removing the priming RAISED its confidence (0.775 -> 0.90 direction,
  0.45 -> 0.60 mechanism), which argues the signal is in the descriptions rather
  than in the brief.
---

# Blind comparison — group A vs group B (clean, unprimed)

Both files read in full. Analysis below, based solely on their contents.

## What the items are

- Both groups consist of **auto-interpretability descriptions of SAE-style features** — each item names a token (or token class) plus the context in which it activates. Verified: exactly 100 numbered items per file.
- The two groups are **broadly similar on the surface**: both are dominated by code/markup syntax, punctuation and delimiters, legal/license boilerplate, and multilingual text (Chinese, Russian, etc.). So the criterion is not a simple topic split like "code vs. prose" or "English vs. non-English."

## What actually distinguishes them

- **AI-directed prompt / user-message features: 13 in Group B vs 0 in Group A** (13 is my judgment call; ±2 depending on how you count borderline instructional items — e.g. B12, B35). This is the cleanest signal: a perfect zero on one side. Group B examples:
  - B86: "The pronoun 'You' appearing in jailbreak prompts that instruct an AI to violate its safety guidelines…"
  - B4: "The word 'only' appearing as part of the meta-instruction phrase 'only send the completion based on the system instructions' in text completion model directive prompts."
  - B81: "The token 'over' when specifying a word count requirement in the templated phrase 'Give me an introduction over 200 words for [company], a chemical company in [location]'." (a well-known chat-corpus spam template)
  - Also B1 ("user messages"), B8 ("instructing an AI assistant"), B30 ("user requests"), B33 ("user queries"), B46/B64/B93 (adversarial/toxic prompt templates), B29/B52 (response format constraints). Nothing remotely like this appears in A.
- **Digit/number-token features: ~14 in B vs ~3 in A** (estimate). B is full of digits in URLs, IDs, ZIP codes, hex hashes, version numbers, section numbers (B11, B16, B19, B22, B28, B29, B45, B50, B75, B77, B80, B92…); A has only ~3 (A14, A26, A64).
- **Pasted-document / metadata flavor in B**: Google search URL parameters (B9: "The token 'ved' … in the pattern '&ved='"), copyright headers, news datelines/bylines, Creative Commons URL paths, an address fragment ("UNIT" in "UNITED STATES"), corrupted Unicode, medical vital-sign listings — i.e., material a user pastes into a chat.
- **Group A skews toward text-in-flow and structural glue**: word fragments from tokenization splits (~11 items in A vs ~7 in B, estimate — e.g. A89: "incomplete word fragments or morphemes … occurring mid-word in unusual tokenization splits"), line breaks and markdown/list structure (A3: "Line breaks, newlines, or whitespace sequences that mark the end of a line…"; A43, A99), ASCII-art boundaries (A77), poetic rhyme words (A16: "Words ending in '-ight', '-ites', or '-emes' sounds … in poetic, lyrical, or stylistically elevated prose"), and response-flavored discourse (A78: "'your' … when responding to or acknowledging someone's question"; A95: declarative/definitional statements "in question-answer formats").
- **Description style**: B far more often pins a specific literal token in a specific template (~45 items vs ~25 in A, rough estimate); A's descriptions are more abstract/structural ("tokens that serve as connectors…", "tokens that complete…"). Minor note: German-language features appear only in B (5 items); Korean only in A (2 items).

## Inferred criterion

- **Best guess: the features were split by WHERE they activate relative to the chat input/output boundary.** Group B = features whose top activations sit on the **input/context side** — user queries, prompt templates, jailbreak/instruction text, pasted documents, URLs and numeric identifiers. Group A = features that activate on the **other side of that boundary** — the model's answer / free-flowing continuation text (markdown structure, generated code and prose, answer-side discourse, sub-word glue). The perfect 13-vs-0 split on prompt-instruction features and the 14-vs-3 digit skew are exactly what a positional (context-vs-answer, or query-vs-rest) split would produce; a content-based split would not leave the within-group heterogeneity both groups show.
- **Alternative interpretations I can't rule out from the text alone**: (a) split by the corpus the top-activating examples came from (chat/user-query data vs. pretraining-style documents) — this is nearly observationally equivalent to the positional story; (b) split by some judged "context category" of the feature where B's category happens to be instruction/metadata-flavored; (c) split by experiment arm, with content differences only statistical.

## Confidence

- High (~90%) that the operative distinction is "Group B is anchored in user-prompt / pasted-input material, Group A is not" — that's directly measurable in the text (13–0, ~14–3).
- Moderate (~60%) on the specific mechanism, i.e. that the criterion was the activation position relative to the context→answer (input→response) boundary rather than one of the near-equivalent alternatives above.
