---
name: seq-cap-long-context-truncation
description: Context-grid plans that pin per-row seq caps from parent recipes while adding long-prefix/ICL train cells silently truncate those cells — generates spurious "long contexts implant/transfer weaker" structure
metadata:
  type: feedback
---

When a plan inherits per-row training seq caps from validated parent recipes (e.g. fact seq 1024 from #444, EM seq 2048 from turner_em, `train_lora` default `max_length=1024`) AND its context battery adds train cells whose wrapper alone approaches/exceeds the cap (WildChat ~2,000-token prefixes, ICL k=8 demo blocks), the training rows for those cells get tokenizer-truncated. Right-truncation cuts the loss region (implant fails — caught at the diagonal manipulation check but loses user-locked grid cells in a family-correlated way); left-truncation silently mislabels the train context (manipulation check passes, cell answers a different question). Either way the artifact mimics the exact finding such plans register ("prefix length vs G" covariate reads), and diagonal implant-strength covariates CANNOT distinguish artifact from genuine context-trainability — both predict a low diagonal.

**Why:** Surfaced on #537 plan v2 (Alternatives lens, 2026-06-09): fact row seq 1024 < wc_long ≤2,000-token prefix is an arithmetic certainty, not speculation; plan had render/role checks (A17) and collator-mask checks (A8) but no tokenized-length-vs-seq-cap check anywhere.

**How to apply:** On any plan combining (a) per-row seq caps inherited from parents and (b) long-context train cells: compute max(context render + completion) per row vs the row's cap. If any cell exceeds it and the plan has no per-cell truncation_frac metadata / data-build length assert, that is a Must-Fix (REVISE) — the analyzer cannot weigh it post hoc from shipped diagnostics. Fix shapes: log+assert truncation_frac=0 at data build, raise seq as a named recipe deviation, or shrink the long context to fit the binding row.
