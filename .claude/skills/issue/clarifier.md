# Clarifier Prompts

The clarifier's job is to catch ambiguities BEFORE the adversarial planner
spend agent time on a spec that isn't tight. It asks targeted questions by issue
type.

**Rule:** if ≥2 blocking ambiguities remain after reading the issue, post them
as a numbered `<!-- epm:clarify v1 -->` comment and EXIT. Don't proceed to
adversarial planner.

A question is **blocking** if it would cause the plan to be wrong or the
reviewer to flag overclaims. "What font should the plot title use" is NOT
blocking; "which baseline are we comparing against" IS blocking.

---

## Step 0 — Context gathering (MANDATORY, runs before any questions are drafted)

Before composing the clarifying questions, the clarifier MUST first try to
**resolve ambiguities from existing project knowledge**. The point: don't ask
the user something the repo already answers. Every blocking question saved here
is 30s+ of user attention saved.

Run all of these in parallel and read the results, then draft questions only
for the gaps that remain:

1. **Past issues + clean results** — search GitHub for related work:
   ```bash
   # Issues whose body/title mentions the same key terms (model, condition, dataset, etc.)
   gh issue list --search "<key terms from issue body>" --state all --limit 20 --json number,title,labels,url
   # Clean-result issues (label `clean-results`) — these have polished write-ups + numbers
   gh issue list --label clean-results --state all --limit 20 --json number,title,url
   # If the issue body says `Parent: #<M>` or cites another issue, fetch it directly:
   gh issue view <M> --json title,body,labels,comments
   ```
   Skim titles + TL;DRs. If a clean-result already establishes a baseline,
   number, or methodology that the current issue is implicitly referring to,
   note it — don't re-ask.

2. **Repo literature — has someone (us OR prior work) already run something
   like this?** This is the highest-value pass: surfacing a duplicate or
   near-duplicate saves an entire experiment. Search both internal and
   external knowledge:

   **Internal (us):**
   - `gh issue list --label clean-results --state all --search "<terms>"` —
     polished write-ups + numbers from our own past experiments.
   - `RESULTS.md` — headline-level findings; if any of them already address
     the current question, surface it before drafting any clarifying question.
   - `eval_results/INDEX.md` — pointers to structured JSON results from prior
     runs (useful when the issue says "compare to the eval from issue #N").

   **External literature (prior work):**
   - `.arxiv-papers/` — papers the user has explicitly downloaded for this
     project. **Grep this first** for terms from the issue (model name,
     technique, dataset, hypothesis). If a paper already reports the same
     intervention + result, surface it — running a duplicate is a waste.
     When precise math/equations matter, use `mcp__arxiv-latex__get_paper_section`.
   - `external/` — reference codebases checked into the repo (open-instruct,
     agentic-backdoor, training-against-misalignment). Grep for function/class
     names mentioned in the issue; sometimes a method is already implemented
     and we'd be duplicating engineering work.
   - `docs/` — internal write-ups and research notes (e.g., `research_ideas.md`,
     literature digests). Often contain summaries of related work that didn't
     make it into a clean-result issue.
   - `mcp__arxiv__search_papers` / `mcp__arxiv__semantic_search` — only if the
     issue references a paper not yet in `.arxiv-papers/`, or if the internal
     literature pass turned up nothing relevant and external work is plausible.

   **If a near-duplicate exists** (in our issues OR in prior work), surface it
   in the `Context resolved` note AND raise it as a blocking question to the
   user: "Issue #N / Paper X already reports <result> — does this issue still
   need to run, or is the goal a follow-up / replication / different scope?"
   The user almost always wants to know.

3. **Results index + ideas backlog**:
   - `RESULTS.md` — headline-level findings. If the current issue contradicts
     or extends one of these, name it.
   - `eval_results/INDEX.md` — pointers to structured JSON results from prior
     runs. If the issue references "the eval from issue #N", look up #N here.
   - `docs/research_ideas.md` — the running ideas backlog; the current issue
     may have been promoted from here with extra context that didn't make it
     into the issue body.

4. **Git history** — recent commits often reveal the *intent* behind a terse
   issue body:
   ```bash
   git log --oneline -n 50
   git log --all --grep="<key term from issue>" --oneline
   git log --all -S "<symbol or function name from issue>" --oneline
   ```
   When the issue says "fix the regression in X", `git log -S X` usually
   surfaces the commit that introduced X and is faster than asking the user.

5. **Codebase grep** — for `type:infra` issues, the relevant files are usually
   discoverable. `Grep` for the symbol/module the issue mentions before asking
   "which files are in scope".

After this pass, write a short internal note (NOT posted to the issue unless
the user asks for it) of the form:

> **Context resolved from project knowledge:**
> - Baseline = clean-result #75 (Qwen-2.5-7B-Instruct, ARC-C 0.78, seed 42)
> - Eval suite = `eval/betley_alignment.py` (per `RESULTS.md` headline #3)
> - Method delta vs parent #137 = swap LoRA r=16 → r=64 (only difference)
>
> **Remaining blocking ambiguities:**
> 1. ...
> 2. ...

Use this note to:
- **Cut** any clarifying question whose answer is already in project knowledge.
- **Sharpen** the remaining questions by quoting the relevant prior result
  (e.g., "Issue #75 used Claude-judge alignment 0–100 — same metric here, or
  different?").
- **Inform** the adversarial planner if/when control passes to Step 2 — the
  planner inherits this same context (cite issue / paper numbers in the plan).

If the context-gathering pass resolves all blocking ambiguities, post the
`<!-- epm:clarify v1 -->` "All clear" comment and include a 3-5 bullet
**Context resolved** section listing the issues / commits / papers consulted,
so downstream agents and reviewers can audit the inheritance chain.

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
- Where does the deliverable go? (clean-result GitHub issue / source-issue
  comment / `docs/` / nowhere, just context?)

---

## Blocking vs non-blocking — a litmus test

Ask yourself: **"If I run the experiment / make the change without answering
this, will the reviewer FAIL the result?"** If yes, it's blocking. If "reviewer
will flag it as a minor caveat," it's non-blocking — note it in the plan's
Caveats section and proceed.

When in doubt, ask. Cost of asking: 30s user attention. Cost of not asking: the
specialist might run for hours on the wrong spec.
