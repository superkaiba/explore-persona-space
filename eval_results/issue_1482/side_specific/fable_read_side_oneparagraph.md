---
packet_set: side
model: claude-fable-5
blinded: true
primed: false
key: {"A": "answer_only", "B": "context_only"}
predicted_context_side: B
truth_context_side: B
verdict: CORRECT
n_per_group: 100
format: single paragraph, verbatim-quotable
brief: >
  Minimal. The two packets were copied into an otherwise EMPTY directory
  (/tmp/i1482_side_isolated/), so there was no key to avoid and the brief carried
  no "do not open X" instruction at all. Whole brief: "Read these two files.
  What is the main difference between group A and group B? Answer in a single
  paragraph. Your reply will be quoted verbatim, so write the paragraph and
  nothing else." Nothing named features, SAE, activation, sides, or a criterion;
  the packet headers read only "# Group A" / "100 items."
siblings:
  bulleted_clean: fable_read_side.md (same split, same frozen key, minimal brief, bullets)
  primed: fable_read_side_primed.md (setup disclosed in brief + criterion name leaked in header)
notable: >
  Third consecutive unprimed read to identify the items as SAE feature
  descriptions from content alone. It also surfaced a discriminator the longer
  reads only touched glancingly: B's items are NARROWER -- single-token-in-one-
  exact-context signatures -- even on the non-chat ones. That specificity
  asymmetry is the one observation here most plausibly explained by the labelling
  instrument (context-side windows come from a different pass than answer-side)
  rather than by a property of the features.
---

# Blind read — one-paragraph form

Both files are lists of 100 auto-generated SAE-style feature descriptions, and the main difference is what kind of context the features are anchored to: Group A is dominated by broad, corpus-generic structural and syntactic features of ordinary documents and code — punctuation and delimiters, line breaks and formatting boundaries, tokenization fragments and word pieces, multilingual (Chinese/Russian/Spanish) word parts, code syntax like imports, `::`, and `#include`, and license/citation boilerplate — described as general categories with essentially no reference to a chat setting; Group B, by contrast, is heavily weighted toward the instruction/prompt side of AI-chat interactions and toward hyper-specific templated strings: many of its items explicitly mention user messages and requests, imperative commands, meta-instructions and response-format constraints addressed to an AI ("only send the completion...", word-count limits like "no more than 50 words"), prompt-template delimiters ("###", "[your answer]"), and adversarial or jailbreak prompt templates ("say something toxic", prompts instructing an AI to violate its guidelines), and even its non-chat items tend to be far narrower, single-token-in-one-exact-context signatures (the "ved" in Google's `&ved=` URL parameter, "quick" in "The quick brown fox", the "(at your option)" GPL phrase) rather than Group A's broad structural categories.
