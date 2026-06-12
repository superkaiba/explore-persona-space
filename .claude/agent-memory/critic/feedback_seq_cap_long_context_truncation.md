---
name: seq-cap-long-context-truncation
description: Plans inheriting per-row seq caps from parent recipes while adding long-prefix/ICL train cells silently truncate those cells — spurious "long contexts implant weaker" structure; Must-Fix when no truncation assert exists (#537)
metadata:
  type: feedback
---

When a plan inherits per-row training seq caps from validated parents (fact seq 1024 from #444, EM 2048, `train_lora` default 1024) AND its context battery adds train cells whose wrapper alone approaches/exceeds the cap (WildChat ~2,000-token prefixes, ICL k=8 blocks), those rows get tokenizer-truncated. Right-truncation cuts the loss region (implant fails in a family-correlated way); left-truncation silently mislabels the train context (manipulation check passes, cell answers a different question). Either way the artifact mimics the exact "prefix length vs G" finding such plans register, and diagonal implant-strength covariates CANNOT distinguish artifact from genuine context-trainability — both predict a low diagonal.

**Why:** #537 v2 (2026-06-09): fact seq 1024 < wc_long ≤2,000-token prefix was an arithmetic certainty; the plan had render/role and collator-mask checks but no tokenized-length-vs-cap check anywhere.

**How to apply:** compute max(context render + completion) per row vs the row's cap. If any cell exceeds it and the plan has no per-cell truncation_frac metadata / data-build length assert → Must-Fix (REVISE) — unweighable post hoc. Fix shapes: log+assert truncation_frac=0 at data build, raise seq as a named recipe deviation, or shrink the long context to fit.
