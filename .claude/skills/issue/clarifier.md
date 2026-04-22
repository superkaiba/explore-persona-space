# Clarifier Prompts

The clarifier's job is to catch ambiguities BEFORE the gate-keeper and planner
spend agent time on a spec that isn't tight. It asks targeted questions by issue
type.

**Rule:** if ≥2 blocking ambiguities remain after reading the issue, post them
as a numbered `<!-- epm:clarify v1 -->` comment and EXIT. Don't proceed to
gate-keeper.

A question is **blocking** if it would cause the plan to be wrong or the
reviewer to flag overclaims. "What font should the plot title use" is NOT
blocking; "which baseline are we comparing against" IS blocking.

---

## For `type:experiment` issues

Check that the issue body answers each question. If not, ask it.

### Hypothesis + prediction
- What specific hypothesis does this test? State it as an `if X then Y`.
- What is the quantitative prediction? (e.g., "EM coupling drops by ≥30% under
  intervention A vs. baseline")
- What result would FALSIFY the hypothesis? (kill criterion)

### Baseline and controls
- What is the baseline? (which prior experiment / issue / published result)
- What controls make the comparison clean? (same seed? same data? same eval set?)
- Are we controlling for confounds X, Y, Z that bit us in prior issues?

### Data
- What dataset? Exact name, version, size.
- Do we need to regenerate data, or is it cached on HF Hub?
- What preprocessing?

### Model
- Base model? Checkpoint? (HF path or WandB artifact)
- Full finetune / LoRA / DPO / SFT?

### Training details (if training involved)
- Learning rate, schedule, batch size, epochs, seq length.
- Precision, DeepSpeed stage, LoRA config if applicable.
- How many seeds? (Single-seed experiments get flagged by reviewer — consider ≥3
  if the claim is headline-level.)

### Eval
- Which eval suite? (ARC-C / MMLU / alignment judge / custom)
- Which metric? (accuracy / Claude-judge alignment 0-100 / StrongREJECT / etc.)
- How many samples per question for stochastic evals?
- Statistical test? (paired t-test, bootstrap CI, Bonferroni correction if
  multiple comparisons)

### Compute
- Target pod? Pod1-5 have different GPU counts — which is appropriate?
- GPU-hour estimate? (informs compute label: small <5h, medium 5-20h, large >20h)
- Wall-time estimate?

### Upload + cleanup
- WandB project name? (default: `explore-persona-space`)
- HF Hub repo for model upload? (default: `superkaiba1/explore-persona-space`)
- Any local artifacts to keep after upload, or clean all?

---

## For `type:infra` or code-change issues

### Scope
- Which files are in scope? List them explicitly.
- Which files are explicitly OUT of scope? (Prevents scope creep.)
- Is this additive (new module) or modifying (refactor)?

### Motivation
- What problem does this solve? (Performance / correctness / DX / compat / other)
- Is there a failing test, crash, or user complaint that motivated it?

### Compatibility
- Does this break any existing callers? If so, which?
- Are there deprecation paths to preserve?

### Tests
- Which tests should pass after the change? (List specific `pytest` invocations.)
- Are new tests required, or do existing tests cover the behavior?
- Does this require a new integration test?

### Dependencies
- Does this require new packages? (`uv add ...`)
- Does this bump any existing packages? (mention in plan — lockfile matters)

### Performance
- Expected impact on wall-time / memory? (none / minor / major)
- If major, what's the benchmark before and target after?

### Risk
- What's the blast radius if this is wrong?
- Can it be rolled back cleanly? (Single commit revert, or more involved?)

---

## For `type:analysis` issues (re-analysis of existing results)

### Source data
- Which `eval_results/` directory or WandB run(s)?
- Git commit hash of the original experiment?

### New claim
- What new claim are we trying to support?
- Is this within-scope of the original experiment, or does it require new data?
  (If new data, re-classify as `type:experiment`.)

### Methodology
- What statistical test / aggregation method / plot style?
- Are we re-using the original eval script or a new one?

---

## For `type:survey` issues (literature / exploratory read)

### Question
- Specific question being answered?
- Decision this informs?

### Sources
- Which papers / repos / blog posts? (Prefer explicit list — avoids infinite scope.)
- Time budget? (survey issues should cap at 2h agent time)

### Deliverable
- TL;DR length? (1-paragraph / 1-page / formal write-up)
- Where does the deliverable go? (`research_log/` / issue comment / nowhere,
  just context?)

---

## Blocking vs non-blocking — a litmus test

Ask yourself: **"If I run the experiment / make the change without answering
this, will the reviewer FAIL the result?"** If yes, it's blocking. If "reviewer
will flag it as a minor caveat," it's non-blocking — note it in the plan's
Caveats section and proceed.

When in doubt, ask. Cost of asking: 30s user attention. Cost of not asking: the
specialist might run for hours on the wrong spec.
