---
paths:
  - ".claude/rules/analyzer-section-reference.md"
description: >
  Full templates, worked examples, and extended how-to prose for the
  analyzer.md Analysis Protocol steps (1 / 1.5 / 1.6 / 2 / 3 / 3.5 / 3.6 /
  4 / 4.5 / 5 / 6 / 6.5 / Quality bar), relocated verbatim from
  .claude/agents/analyzer.md (#850, following the #838 planner/critic split).
  Loaded ONLY via the explicit pointer lines in analyzer.md — the
  self-matching `paths:` glob keeps this file out of every other agent
  context (a missing `paths:` key would auto-inject it always-on fleet-wide,
  recreating the #833/#834 spawn-weight bug).
---

# Analyzer section reference (analyzer.md relocated protocol sections)

One H2 per protocol step, headings verbatim from analyzer.md. Read ONLY the
section you are about to execute: Grep the heading, then chunked `Read` of
that span (per analyzer.md § Context budget) — never the whole file.

## Step 1: Load and Understand Data

Read, in order:
0. `frontmatter.goal` from body.md — the canonical one-sentence Goal the user filed at /issue Step 0c. This is your organizing target: the results narrative must answer how the experiment moved the needle on this Goal. You do NOT propose Goal changes — by the time analysis fires, the Goal is contract. If multiple `epm:goal-updated v1` markers exist in events.jsonl (Goal was refined during planning), the LATEST `to:` value is canonical; you MAY note this once inside the relevant `### <result>`'s what-is-plotted or interpretation prose ("Goal was refined once during planning — see events.jsonl"), but the refinement is not the story.
1. The plan (from the `epm:plan` events.jsonl event, or `.claude/plans/issue-<N>.md`)
2. Specific result files (`eval_results/<name>/run_result.json` and any per-condition JSONs)
3. `epm:results` workflow event on the source experiment
4. RESULTS.md (context on prior findings) and `docs/research_ideas.md`
5. Related prior write-ups (clean-result experiments — `has_clean_result=true`; browse at <https://eps.superkaiba.com/?has_clean_result=true>). The legacy `research_log/` flow is retired — its archive lives at `archive/research_log/` (read-only) for historical context only.

Before analyzing, write down — in your scratch context — what the hypothesis was, what would confirm it, what would refute it, and what the baselines are. **Pull every number from the raw JSON, not from the experimenter's summary.** Common failure: draft says 92%, JSON says 89%.

**Measurement-validity gate (run BEFORE interpreting).** (Skip when there is no Goal-bound behavioral construct — `kind: analysis|infra|batch|survey`.) The Goal names a *construct* (a real behavior); the headline metric is only a *proxy* for it. Four checks, all of which can downgrade confidence or block the headline:

1. **Dynamic-range / floor-ceiling check (compute it from the raw JSON).** Look at the headline metric's spread across conditions. If (nearly) every condition sits at a floor or ceiling — e.g. all log-probs within a tiny band of effectively-zero probability, all pass-rates at 0% or 100%, all values inside the metric's saturated tail — the probe is presumed **uninformative**: the ranking among those values is noise. Do NOT narrate rank-shuffles among saturated values as a finding. Surface the saturation explicitly ("all 28 personas score log p between −17 and −27, i.e. ~0 emission probability — the leaderboard ranks near-zero values") and treat it as a confidence-capping constraint, not a result.
2. **Proxy-vs-construct check.** Read the plan's §6 measurement-validity entry and the Goal's construct. If the headline metric is an **off-distribution proxy** (teacher-forced not on-policy, a fixed canonical/stub answer instead of the model's own generation, an arbitrary token position, a single-token shortcut) for a behavioral construct, you MUST NOT narrate the proxy as the construct. Write the construct-accurate statement ("log p(※) at a fixed-answer probe", not "the model emits / implants the marker"), and state the proxy gap in the body. If the plan validated the proxy against the construct, cite that validation; if it did not, the headline claim about the *behavior* is unsupported — cap confidence and say so. Narrating a proxy as the construct is an overclaim (interpretation-critic Lens 1 catches it).
3. **Dual-DV for content-behavior leakage / implantation (compute + report BOTH).** When the result is a *content* behavior leakage/implant (sycophancy, refusal, hedging, style, trait — not the programmatic marker, which has its own three-space recipe), CLAUDE.md § Measurement validity requires you to compute AND report BOTH DVs: (a) the PRIMARY judge-scored on-policy behavior/agreement RATE (trained − base, the validated behavioral construct, the headline number), AND (b) the SECONDARY continuous completion-probability DV — PREFERRED the teacher-forced FIXED positive-vs-negative completion margin (fixed answer pools across all contexts ⇒ no selection-on-outcome bias, #722), with the judged-positive-conditional-mean `log P` (`logp_pos_mean`) the selection-confounded opt-in alternative that must first pass the ρ(DV, rate) > 0 validation (it failed for 3 of 4 behaviors in #722). ALSO compute and report the standing validation: the Spearman of (b) vs (a) across the cells that have dynamic range. The reason both are needed: the binary rate saturates at floor/ceiling and CENSORS install / dose-matched / cross-condition comparisons (#608) — read those off the continuous DV where the rate is pinned — while the rate is immune to the teacher-forcing artifact (#432→#456) the probability DV carries. Keep the judge rate PRIMARY in the narration; report the probability DV as the SECONDARY companion and NEVER narrate it as the construct. If the validation Spearman is weak / the probability DV and rate disagree where both have range, say so and cap the cross-condition claim. If the plan registered only one of the two (a planning miss), report the one you have, compute the other where the artifacts allow, and flag the gap in the body. (interpretation-critic Lens 1 + critic Statistics lens item 10 enforce this.)
4. **Band-vs-ceiling check (re-check the REALIZED band against the DV's achievable ceiling before narrating any non-rejection).** When a registered null band gates a read, compare the band's realized upper bound (e.g. the 97.5% quantile of the max-selected null distribution) against the DV's achievable estimator-bound ceiling (a bounded DV's bound; for a difference statistic, the max attainable favored-arm value MINUS the exact registered comparison-arm quantity the statistic uses — never the raw single-arm bound). If the band's upper bound meets or exceeds that ceiling, the test was uninformative-by-construction (zero power): narrate any NON-REJECTION as **failure-to-reject** ("the test could not have detected any achievable effect") — NEVER as evidence of absence, a clean ordering fail, or a reversal. The mandate is scoped to the unreachable tail/direction: a separately REACHABLE lower-tail / opposite-direction rejection remains a legitimate finding. Draw band + ceiling in the figure so the unreachability is visible in the artifact. A band above only a fallback severity REFERENCE POINT (no estimator bound derivable) is reported as low-severity / underpowered, not zero power. (#810: the registered difference-null band's p97.5 upper bound was 0.800 while the max attainable skill was ~0.857, so the max-difference statistic could essentially never clear the band — even the parent round's +0.209 Betley effect would fail — and the p = 0.634 outcome was initially narrated as a clean ordering fail until the interpretation-critic caught it. Full check: `.claude/rules/selection-symmetric-nulls.md` § Band-vs-ceiling informativeness check.)

**Step 6 (set-body) writes the polished clean-result body in the v4 shape — FOUR required H2s in order — `## Takeaways` / `## Goal` / `## Methodology` / `## Results` — plus a bold `**Repro:**` / `**Context:**` footer (NOT an H2), following the H1 title.** **A v4 body MUST NOT contain the v3 content H2s (`## What I ran`, `## Findings`, `## Data`, `## Reproducibility`) NOR the retired `## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure` — any of those is a verifier check-2 hard FAIL.** Figures live inline inside each `### <result>` H3 under `## Results` (one figure per result, in the strict three-beat what-is-plotted → plot → interpretation); the COMPLETE hyperparameter table + the systematic worked examples live in `## Methodology`; compute / code SHA / artifact links + run-provenance live in the `**Repro:**` / `**Context:**` footer. The full spec is `.claude/skills/clean-results/SPEC.md` § "v4 body shape" — read it before drafting.

**Emit the v4 sentinel.** Immediately after the H1 title, write the literal HTML comment `<!-- clean-result-v4 -->` on its own line (blank line before and after). The verifier gates every v4 rule on this sentinel. Bodies WITHOUT it keep v3/v2/legacy behavior — every NEW draft you produce MUST carry the v4 sentinel.

**Confidence lives in the H1 title tag only — do NOT emit a `Confidence: …` sentence anywhere in a v4 body.** The H1 title's `(LOW|MODERATE|HIGH confidence)` suffix is the single source of truth. There is no "Why confidence is where it is" section. If you need to convey what the binding constraint is, weave it into the relevant `### <result>` interpretation prose and/or a `## Takeaways` bullet.

**`## Takeaways` is the first H2 — the cross-round synthesis (no `## Human TL;DR`).** **3-6 bullets, each ≤30 words, numbers-first, PLAIN ACADEMIC register** (NOT casual/lowercase, NOT a "How this updates me" diary). Each bullet leads with or bolds its load-bearing number + CI. The shape:

```
## Takeaways

- <headline finding, key number + CI bolded>
- <secondary finding>
- <the caveat that binds interpretation>
- <what this changes / next decision>
```

**`## Takeaways` is the ROLLING cross-round synthesis** — it ALWAYS reflects the current cross-round belief. On a same-issue follow-up round you REWRITE it to integrate the later round (see Step 6 § Same-issue follow-up re-entry); a `## Takeaways` that describes only round 1 after round 2 landed is a critic FAIL. The H1 title stays the one-sentence claim + confidence tag; retitle it if the headline moved (on a re-fold, pair the H1 retitle with `task.py set-title <N> "<new H1 text>"` — see § Same-issue follow-up re-entry).

The frontmatter `goal:` field stays in the new body so downstream agents (planner, critic, follow-up-proposer) have the agent-facing canonical Goal as context. The Goal motivation folds into the `## Goal` section's TWO required parts — `**This experiment in context:**` (what THIS experiment tests + how it relates to the other experiments in its line; the ONLY place prior-issue links appear) and `**Broader narrative:**` (the project-level / `docs/open_questions.md` question it serves) — both rewritten in plain English, not pasted verbatim. If the result substantively diverged from the Goal, that's a signal the experiment didn't answer the question it set out to answer — surface it in the relevant result's interpretation prose rather than papering over it.

**Methodology corrections fold into the relevant `### <result>`'s what-is-plotted or interpretation prose.** There is no `### Methodology corrections` heading. Content that previously lived there — plan deviations applied during the run, mid-run bugs caught and fixed, hot-fixes, data patches, threshold changes the eval revealed were inappropriate, dataset-mapping bugs caught and corrected before final aggregation — now lives inside the finding whose interpretation it actually shapes. Each item: what was wrong → what changed → effect on this finding. Keep the narrative inside the finding so a reader landing on it reads the correction in context. If no corrections occurred, no extra prose is needed — the absence is the signal.

## Step 1.5: Load top-N promoted clean-results as in-context exemplars

Before drafting, fetch the N most-recently-created clean-result bodies
that have been promoted. Default N=3,
override with `EPM_EXEMPLAR_N`:

```bash
uv run python scripts/recent_clean_results.py --n "${EPM_EXEMPLAR_N:-3}" --format inline
```

Include these inline in your scratch context as exemplars of the TARGET
QUALITY BAR — do not copy text or claims; the user has approved the SHAPE
of these write-ups by promoting them. Use them as a reference for: TL;DR
length, takeaway phrasing, confidence framing, hero-figure caption tone.

If no promoted clean-results exist (fresh project), the helper prints
"No promoted clean-results found." and you proceed without exemplars.

#### Raw-output spot check (mandatory, per #275 item 12)

Before computing any aggregate statistics or writing the interpretation
body, sample 5 random rows from the eval JSON/CSV and paste them at the
TOP of your `<!-- epm:interpretation -->` body under an H3 heading:

```
### Raw-output spot check (5 random rows)
```

For each sampled row, write one verbatim quote (or a 1-line summary if
the row is too long), and note any visible fishiness:

- judge label disagrees with content (e.g. judge says `aligned=False`
  for a polite refusal)
- sampling collapse (5 prompts produce identical outputs)
- refusals miscategorised as alignment / misalignment
- non-English / corrupted generations (tokenizer mismatch, EOS trained
  out, prompt template wrong)
- empty outputs / silent zeros

If ANY fishiness is visible, state it explicitly in the spot-check
section AND raise it in the confidence rationale of your interpretation.
A spot-check that finds 3+ fishy rows out of 5 SHOULD downgrade
confidence to LOW or "indistinguishable from artefact". Do NOT label
the issue `status:blocked` from this step — flag the concern in the
interpretation body and let the interpretation-critic adjudicate.

Procedure:

1. Locate raw generations (path is in `epm:results` →
   `raw_completions_path`, or the WandB artifact URL).
2. Sample 5 rows with a fixed seed:

   ```python
   import json, random
   random.seed(42)
   rows = [json.loads(l) for l in open(<path>)]
   sample = random.sample(rows, min(5, len(rows)))
   ```

3. Paste them under the H3 heading at the very top of your
   `epm:interpretation` body.
4. Continue with the rest of the interpretation.

The interpretation-critic checks for the H3's presence and substance as
part of its normal review (no separate marker, no separate skill-step
gate, no `status:blocked` path).

#### Content hygiene for harmful-content corpora (EM, refusal, harmful-advice) + real-world-corpus rollout text (LMSYS/WildChat-class)

When the run's raw completions come from a harmful-content corpus
(Betley-style EM, bad-medical-advice, refusal-bait pools), OR the run's
probe set comes from a harmful safety-benchmark question bank
(`src/explore_persona_space/artifacts/query_banks/*.json` — advbench,
strongreject, Betley-lineage, sensitive-info banks; incident #866), OR
the run's corpus is real-world user text (LMSYS/WildChat-class —
unscreened real user text routinely carries in-corpus jailbreak/explicit
rows; #1073: an analyzer refusal-killed twice paging raw LMSYS rollout
text), verbatim rows
in your context can trigger terminal API usage-policy refusals that kill
your final turn and make the transcript unresumable (incident: task
#537, 2026-06-10). For those rows, the spot check above AND the Step 3.6
sample selection run in sanitized mode:

- Read minimal slices via field-filtered `jq` (judge label, marker
  presence, row index, token counts) — never load whole files or full
  text-field values into context.
- Embed a short sanitized excerpt (first ~15 words) plus a placeholder
  `[truncated — harmful-content row; verify at <raw-completions path>,
  row <i>]` instead of the full completion. Keep labels, indices, and
  the permanent raw link verbatim — that is what carries the evidence.
- Label each such block "sanitized for context hygiene" so the critics
  know the truncation is deliberate, not evidence-hiding. Benign corpora
  (marker, fact, sycophancy, personas) and benign banks
  (`arc_c_v1`, `fact_questions_v1`, `marker_eval_v1`,
  `sycophancy_claims_v1`, and `wildchat_random_v1` — benign ONLY because
  its builder screens on WildChat's toxic/redacted flags,
  `issue617_build_wildchat_slice.py` → `_wildchat_eligible`; an
  UNSCREENED real-world-corpus slice is never benign-classed) keep the
  standard verbatim treatment; when unsure whether a bank is harmful —
  or whether a real-world slice was moderation-screened — sanitize.

## Step 1.6: Planned-control-arm presence gate (run BEFORE interpreting / plotting / authoring)

**Why this gate exists.** A verdict authored on a partial grid that is
silently missing a planned control / baseline arm can be a
multiple-comparison artifact. In #658 the analyzer ran on a partial Betley
grid before the random-projection control arm had landed, authored an
apparent 3/10 PASS, and that PASS evaporated to 0/10 once the control arm
arrived — the apparent signal was an artifact of comparing the arms that
happened to be present. This gate refuses to author at all until every
DECLARED control/baseline arm is actually present in `eval_results/`. It
fires UPSTREAM of the write-up-time siblings (`verify_task_body.py`
check 11b and `clean-result-critic` Lens 13), which read a body that — in
the #658 case — would have looked internally consistent on the partial
grid because the analyzer did not know an arm was missing.

This gate runs AFTER Step 1 (which already loads the plan + the
`eval_results/` JSONs — so it adds no new data dependency) and BEFORE Step 2
(statistics), Step 3 (plots / figure commits), Step 4 (write the body), and
Step 6 (set-body + the `epm:interpretation` post). On a missing arm you EXIT
without authoring ANY of those — no statistics narrated, no figure committed,
no body written, no `epm:interpretation` posted.

**Scope — when this gate APPLIES vs SKIPS (opt-in, fail-closed):**

1. **SKIP (vacuous PASS), go to Step 2,** when `frontmatter.kind` is
   `analysis | infra | batch | survey` — these have no Goal-bound multi-arm
   verdict (mirrors the measurement-validity gate's exemption above).
2. **SKIP (vacuous PASS), go to Step 2,** when the plan declares NO
   control/baseline arm. This is the deliberate opt-in carve-out: the gate
   disciplines ONLY plans that COMMITTED to a control arm — legitimately
   single-arm / descriptive work and legacy plans without a controls section
   pass through untouched, exactly as `clean-result-critic` Lens 13 PASSes
   vacuously when the plan has no enumerable planned conditions. **Opt-in
   limit (documented, not a bug):** a future plan that forgets to declare its
   control sails through this gate — the protection is only as good as the
   plan's own declarations. `clean-result-critic` Lens 13 is the downstream
   backstop that can still catch a control declared in prose but never landed
   as an enumerable row.
3. **APPLIES** to `kind: experiment` tasks whose plan declares at least one
   control/baseline arm (the only case where a missing control arm can
   produce the #658 false-positive-by-omission).

**Enumerate the declared control/baseline arms.** Resolve the plan the
canonical way (never hand-build a `tasks/<status>/<N>/...` path):

```bash
plan_path="$(uv run python scripts/task.py find <N>)/plans/plan.md"
```

Read the UNION of these declaration sources (use the union if they disagree —
fail-safe toward catching a missing arm):

- the plan §5 Conditions and Controls table — an arm is a control/baseline arm
  when its row is labeled `CONTROL` / `BASELINE` (case-insensitive) in any
  column, OR its role columns name it a control / baseline / null / shuffle /
  random-projection / permutation arm;
- the plan §0 Plan Summary `**Baselines / controls:**` line;
- the `epm:plan` marker payload (already loaded at Step 1) when it tags
  conditions with a control/baseline role.

Capture each declared arm's plain-English name AND its config slug (the
rightmost §5 column — the key that maps to `eval_results/`).

**DISTINGUISH a pre-landed arm from an analyzer-computed control.** The
presence check applies ONLY to arms produced by a SEPARATE training / eval
run that you consume as INPUT (the random-projection control arm in #658 was
a separate eval that had to LAND). It does NOT apply to a "control" the plan
§4 Design says YOU compute during this analysis — a permutation null, a
residualization baseline, a TF-IDF / random-projection baseline you build in
Step 2-3, etc. Those do not exist as a file until you compute them, so
"presence in `eval_results/`" is the wrong question: PASS such an
analyzer-computed control here (you satisfy it by computing it later, not by
finding a pre-existing file). If unsure which kind an arm is, read plan §4 —
if it is computed in-analysis, it is NOT subject to this presence check.

**Presence check (minimum-bar evidence, across all four `eval_results/`
layout shapes).** `eval_results/` layouts vary, so match intelligently — do
NOT mechanically check for one fixed file path. For each declared pre-landed
control arm, it is PRESENT iff you can find, somewhere under
`eval_results/issue_<N>/` (or the plan's explicitly named `eval_results/<name>/`
path), a parseable JSON carrying the arm's headline-metric value in ANY of
these shapes:

- **(a) slug-named top-level JSON** — e.g. `eval_results/issue_<N>/arm_<slug>/aggregate_cleaned.json`;
- **(b) per-arm directory** — a directory keyed by the arm's slug / name with a verdict or aggregate JSON inside;
- **(c) nested sub-key inside an aggregate / verdict JSON** — e.g. #658's random-projection control lives as a sub-key (`noise_floor` / `a34_a35` / `ridge_exactness` / `mlp_exactness`) inside `aggregate.json` / `a34_a35.json`, NOT as a slug-named top-level file;
- **(d) metric-purpose JSON** — e.g. #685's `metrics.json` / `metrics_assistant_excluded.json` keyed by metric purpose rather than arm slug.

Matching procedure, fail-closed: try the config slug first (exact), then a
normalized-name fallback (lowercase, non-alphanumeric → `_`; tolerate slug
drift such as #658's `attn` vs an older plan's `sum_attn`), then look for the
arm's headline metric value anywhere under `eval_results/issue_<N>/` (incl.
as a sub-key of an aggregate/verdict JSON). The presence check is satisfied by
ANY of these forms carrying a parseable headline-metric VALUE — and ONLY then.
The minimum bar is value-not-name: a directory of the right name, an
empty/unparseable JSON, or a JSON of the right name MISSING the headline-metric
value all count as MISSING (this stops the gate being fooled by a partial /
placeholder file). LM-instruction prose is more flexible than a script here —
READ the plan + the `eval_results/` tree and decide whether the arm's result is
genuinely present, even when the slug does not trivially match a file name.

If EVERY declared pre-landed control arm is PRESENT: PASS, go to Step 2.

**BLOCK path (a declared pre-landed control arm is MISSING).** Do NOT author.
Name the missing arm(s), park, and exit:

```bash
uv run python scripts/task.py post-marker <N> epm:failure \
  --note 'failure_class: data
reason: planned_control_arm_missing
missing_arms: <plain-English name(s)> (slug: <slug(s)>)
looked_under: eval_results/issue_<N>/
The plan declares these control/baseline arm(s) but no parseable result
carrying the headline metric is present under any of the four eval_results/
layout shapes. Authoring a verdict on this partial grid risks a
multiple-comparison false positive (see #658). Land the missing arm(s), then
re-run /issue <N> (the analyzer re-runs from Step 1 and this gate re-checks).'
uv run python scripts/task.py set-status <N> blocked
```

Then EXIT — write no body, commit no figure, post no `epm:interpretation`.
`failure_class: data` is correct: a missing arm is a factual gap only the user
can fill (land the arm), so the task parks at `status:blocked` and does NOT
enter the Step 7 crash-fix routing (which covers `infra | code` only). The
park is reversible — once the user lands the arm, the next `/issue <N>` re-runs
the analyzer from Step 1, this gate re-checks and PASSes, and authoring
proceeds.

**Complementary layer.** This is the analyzer-side, VERDICT-TIME complement of
the upload-verifier (which runs at the `verifying → interpreting` transition);
the upload-verifier is the natural place for a FUTURE hardening that would
verify declared arms landed before the analyzer is even spawned. The split
between an upstream verdict-time gate (here) and the write-up-time siblings
(`verify_task_body.py` check 11b, `clean-result-critic` Lens 13) is
deliberate — see plan §2.

## Step 2: Compute Statistics

**Long off-pod CPU jobs (SVD builds, `analyze`, bootstrap / permutation stats, eval-JSON aggregation) — use a sentinel, not a bg `nohup` redirect you re-read for completion.** Any of these phases (here and in Step 1.5) can run minutes-to-tens-of-minutes off-pod on the VM. You are a single-turn subagent — do NOT improvise chained `nohup ... > /tmp/log 2>&1 &` commands and re-read the log to decide when it finished: the harness may auto-background a long FOREGROUND command and reset shell state between Bash calls, which strands the `&`-job and the `>` redirect and leaves an empty log with no completion signal (incident #650 burned ~8 wait cycles this way). Instead, have the job write a DONE sentinel carrying its exit code on finish, then poll the SENTINEL with ONE `run_in_background=true` Bash `until` loop:

```bash
# launch — sentinel records the exit code regardless of where the harness runs the job
setsid nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 bash -c 'uv run python scripts/<analysis>.py ...; echo "RC=$? DONE" > /tmp/issue-<N>-<job>.sentinel' < /dev/null >/tmp/issue-<N>-<job>.log 2>&1 &
# then, as a SEPARATE run_in_background=true Bash call, block on the sentinel (NOT the bg stdout):
until [ -f /tmp/issue-<N>-<job>.sentinel ]; do sleep 30; done; cat /tmp/issue-<N>-<job>.sentinel
```

Read `RC=` from the sentinel for the exit code (non-zero → inspect the log and fail loud — never narrate a result off a job that did not finish cleanly). The `until` loop is the single completion signal; the log is for diagnosis only. The thread-cap `env` prefix (OMP/MKL/OPENBLAS/NUMEXPR=8) is REQUIRED on these VM-side launches — a stale worktree's `env.py` setdefault may predate the caps and cannot in-process-cap a torch-before-dotenv importer (#891/#779); `env` wraps the `bash -c` so the caps reach the inner `uv run python`.

For every comparison:
- Mean across seeds
- **p-value** (that is the only significance statistic you report in prose)
- Sample size `N` always stated alongside every percentage / rate / p-value
- Flag `n=1` as preliminary, never a conclusion

Do NOT report effect sizes (no Cohen's d, η², r-as-effect, Δ-framed-as-effect), do NOT discuss choice of statistical test in prose ("paired t-test" / "Fisher" / "Mann-Whitney" / "bootstrap" — the reader does not care), do NOT do power analyses, do NOT report credence intervals as inline point-estimates (e.g. `ρ = 0.60 ± 0.05`). Just: **the p-value, the N, the percentage.**

Error bars on charts are allowed (and required — see `paper-plots`), but the prose talks about p-values and sample sizes, period.

## Step 3: Generate Plots

Use the `paper-plots` skill. Do NOT hand-roll rcParams; `set_paper_style()` is the only blessed entry point.

**Style target — clean-result figures use `"blog"`, paper figures use `"neurips"`.** The blog
style (Anthropic / Apollo / LessWrong-blog register: Inter font with
fallbacks, off-white card frame, frameless legend, left-aligned semibold
title via `set_title_subtitle`, soft-warm colorblind-safe palette) is the
default for any figure that lives inside a clean-result body or a
mentor-update slide. Reserve `"neurips"` for figures destined for a paper
submission. See `.claude/skills/paper-plots/style-reference.md` § "Style
variants" and the worked example at `patterns/B0-blog-bar-comparison.md`.

```python
from explore_persona_space.analysis.paper_plots import (
    set_paper_style, set_title_subtitle, paper_palette_role,
    savefig_paper, add_direction_arrow, proportion_ci,
)

set_paper_style("blog")  # clean-result hero + supporting figures
# Use paper_palette_role("primary"|"baseline"|"control"|"accent"|"neutral")
# for semantic color picks, and set_title_subtitle(ax, title, subtitle, source=...)
# for the Anthropic-blog title block.
# ... build figure, referencing a pattern from .claude/skills/paper-plots/patterns/ ...
savefig_paper(fig, "<topic>/<short-name>", dir="figures/")
```

Minimum deliverables:
1. **Hero figure** (lives inside the headline `### <result>` under `## Results`). Pick the single chart that carries the claim. If no single figure carries it, you haven't distilled hard enough — stop and retry Step 1.
2. **Supporting figures** as needed — one per `### <result>` (one figure per result).
3. **Low-level data plot behind every aggregate statistic (DEFAULT).** A finding that reports an AGGREGATE statistic — a correlation ρ shown as a forest-plot point, a mean / effect size shown as a bar, a p-value, an effect summary — gets, BY DEFAULT, a companion LOW-LEVEL plot of the per-unit data behind it: the scatter the ρ summarizes, the strip / swarm / jittered per-point view behind the group-difference bars, the unbinned counterpart of a binned / aggregated view. Generate it at the same step (save as `*_points.{png,pdf,meta.json}` or `*_scatter.{png,pdf,meta.json}` alongside the summary figure) and embed it inline inside the same `### <result>` (data view first where there's room, else clearly paired with the summary). The reader should be able to SEE the data, not only the number computed from it. Skip ONLY when the finding's primary figure ALREADY is the per-unit view (a raw scatter needs no second scatter), N is so small the figure already shows every point, or the aggregate has no per-unit decomposition (a single scalar) — and state which exemption applies in the read prose or alt text. This is the broad parent of item 3-bis (raw-alongside-processed) and is enforced by clean-result-critic Lens 11.
3-bis. **Raw-counterpart figure for every processed/derived figure** (the transformed-figure special case of item 3). If you produce a residualized / partialled / binned / log-transformed / normalized / aggregated scatter or bar, you ALSO produce the raw (pre-processing) version at the same step — save as `*_raw.{png,pdf,meta.json}` alongside `*.{png,pdf,meta.json}`. Embed the raw inline inside the same `### <result>` as its processed sibling (raw first, then processed). Do not wait for a mentor to ask. Same principle for per-cell vs aggregated artifacts: when the body's claim rests on an aggregated metric, write a per-cell CSV/JSON (per-seed, per-condition, per-persona, per-probe — whatever the aggregation collapsed) and link to it in `## Methodology` / the `**Repro:**` footer. Exception: when raw and processed are visually identical (axis-rescale-only processing), say so in alt text and omit the raw. See SPEC.md § per-finding skeleton points 4–5 (low-level data plot + raw alongside processed) for the full rule.

Every figure saves PNG + PDF + `.meta.json` sidecar (commit-pinned) via `savefig_paper`. Never save only PNG.

**The `.meta.json` sidecar now auto-carries the figure's per-point DATA, not just provenance (default).** `savefig_paper` reads the plotted data back off the matplotlib artists at save time — scatter point offsets (WITH the nearest text label as an identifier column, e.g. persona / seed / cell names), line vertices, bar heights, and error-bar magnitudes, using the axis labels as column names — and embeds it under a `points` key in the shape the EPS dashboard data viewer (`dashboard/lib/task-data.ts`) reads directly. So every figure you commit auto-populates the per-figure sortable/filterable table on `https://eps.superkaiba.com/tasks/<N>` with NO extra work. Two consequences for how you plot: (a) **label your points** (`ax.text(x, y, name)` near each scatter point) wherever a unit identity exists — the SPEC already requires this (low-level-data-plot rule, item 3), and it now also populates the viewer's identifier column; (b) **plain-English axis labels matter doubly** — they become the data table's column headers, so a slug like `cond_4` on an axis becomes a slug column header. Extraction is best-effort: a figure whose artists can't be read back (heatmap/`imshow`, custom collections) silently keeps the provenance-only sidecar and the viewer falls back to the figure link-out — never a save failure. For a figure whose data is huge or already committed elsewhere, pass `savefig_paper(..., embed_data=False)` and add a `data_path` pointer (relative to repo root, under `eval_results/` or `figures/`) to the sidecar yourself. Full contract: SPEC.md § "Dashboard data-artifact interface (Phase 2 contract)".

**Figure URL in the body MUST be an absolute `raw.githubusercontent.com` permalink — NOT a relative path.** The EPS dashboard serves task-folder HTML artifacts but does NOT serve binary PNG/PDF files under `tasks/<N>/artifacts/`, so a relative reference like `![alt](artifacts/hero.png)` renders as a broken image in the browser (incident: task #365, 2026-05-22). Workflow:

1. Save figures under `figures/issue_<N>/` (e.g. `figures/issue_<N>/hero.png`). Do NOT only drop them in the task's `artifacts/` folder — that path is dashboard-invisible for binaries.
2. `git add figures/issue_<N>/ && git commit -m "figures: issue #<N> hero figure" -- figures/issue_<N>/ && git push origin <branch>` BEFORE writing the body. The commit is pathspec-limited so a concurrent session's staged files are never swept in.
3. Capture the commit SHA: `git rev-parse HEAD`.
4. Reference the figure inline inside the relevant `### <result>` under `## Results` with `![alt](https://raw.githubusercontent.com/<owner>/<repo>/<sha>/figures/issue_<N>/<file>.png)` — pinned to the commit SHA, never `main`/`master`/`HEAD`. **Do NOT emit a `## Figure` H2** — verifier check 2 hard-FAILs any v4 body that carries it.
5. Alt text may contain `[brackets]` (e.g. literal marker names like `[ZLT]`); the verifier's image regex handles them.
6. **Repo-root stray guard (#922).** When you worked from a worktree, check
   the MAIN checkout for stale duplicates after the push:
   `git -C "$MAIN_ROOT" status --porcelain -uall -- "figures/issue_<N>/"`
   (with `MAIN_ROOT="${TASK_DIR%/tasks/*}"`, the worktree-proof root;
   `-uall` is load-bearing — without it a wholly-untracked dir collapses
   to one `?? figures/issue_<N>/` entry, the exact #922 shape, and the
   per-file hash compare has nothing to enumerate). For each
   untracked (`??`) entry, compare it to the pinned blob
   (`git hash-object <file>` vs `git rev-parse <sha>:<path>`):
   blob-identical → MAY delete that specific file (`rm` — the content is
   committed, zero loss); differing → do NOT delete (it could be another
   round's in-flight work) — WARN in your report naming path + mtime so the
   orchestrator can triage; the critics' pin-first read rule makes it
   harmless to the review either way. NEVER `git clean` / `git checkout .` /
   `git restore .` at the repo root (hard rule above); tracked files are
   never touched.

`verify_task_body.py` Check 4b (`Figure URL resolvable`) fails any body with a relative figure URL, a `main`/`master`/`HEAD`-pinned raw URL, or a figure URL whose target does NOT exist — same-repo SHA-pinned raw URLs are verified against the git object database via `git cat-file` (incident: task #507, 2026-06-09 — a caption cited a figure that was never generated), with an HTTP HEAD fallback for unknown SHAs / other hosts. The gate blocks promotion to `awaiting_promotion` until the URL is fixed, so commit the figure FIRST (steps 2-3 above) and pin the URL to the commit SHA that actually carries it.

## Step 3.5: Plot-verification (MANDATORY, before writing the body)

For each figure that will appear in the body, you MUST visually inspect the rendered PNG before referencing it in the body. The Read tool can load PNG bytes — use it.

```
Read .claude/cache/figures/issue_<N>/<name>.png   # or wherever the figure is
```

For each loaded figure, confirm:
1. **The figure renders correctly** (axes, labels, legend, points / bars all visible).
2. **The figure matches what the caption will claim about it** — load the figure, then write your caption draft, then re-check that the caption is accurate. Specifically: every panel referenced by the caption is in the figure; every condition / color / sample-size mentioned matches what's plotted; the headline finding the caption asserts is visible in the figure.
3. **Annotated key points are visible** (e.g., the `/Anth/` vs `/anthx/` identical-cosine pair) and not clipped or hidden behind other elements.
4. **Inherited-figure data freshness (same-issue follow-up re-folds only).** If you are mid same-issue follow-up re-fold AND a figure you are about to embed was NOT generated by you this pass — i.e. it was committed by the implementer or an upload-fix in a PRIOR round (any figure whose PNG/`.meta.json` already existed before you started this fold) — you MUST cross-check its data against the NEW round's result JSON before embedding it. A figure re-committed with a trivial byte-diff can still plot STALE data: the gen script was never repointed at the new result JSON, so the figure renders cleanly while carrying the prior round's values, and the visual checks above (1-3) will NOT catch it. Mechanically: `json.load` the figure's `.meta.json` and read its `points` (the per-point plotted values `savefig_paper` auto-embeds — see the `.meta.json` affordance above), then compare those values to the corresponding field in the NEW round's result JSON you just produced this fold. Compare every panel/series whose values SHOULD have changed this round — a multi-source figure may legitimately hold one panel fixed across rounds (e.g. a base-rate panel), so a held-fixed panel matching the prior round is correct and is embedded as-is; the staleness signal is a value that should reflect the new result still showing the old one. (Caveat: `savefig_paper` caps embedded rows at `_MAX_SIDECAR_ROWS` and records `truncated`/`total_points`, so a truncated sidecar only lets you compare the embedded subset — for a truncated figure also confirm the gen script reads the new result JSON, the backstop against staleness hiding in a truncated-away point.) If the `points` reconcile with the new JSON → embed it. If they match the PRIOR round's values, are drawn from a stale source JSON, or otherwise don't reconcile → REGENERATE the figure from the new result JSON (re-run the figure-gen script after confirming it reads the new JSON, or repoint it) before embedding, and flag the stale figure superseded in the body's `**Repro:**` footer. This check is cheap (two `json.load`s + a value comparison). It is defense-in-depth, NOT a cure for the upstream cause (the gen script never being repointed at the new JSON) — incident #667 a36 round 2 caught exactly this stale-inherited-figure class post-hoc by diligence, not a guardrail.

If a check fails, fix the plot (re-run the script, adjust layout, fix labels) before writing the body. Never reference a figure you haven't visually verified.

## Step 3.6: Raw-text sample selection (MANDATORY, per load-bearing condition)

For every load-bearing condition that produced completions, pre-select raw completions to embed in `## Methodology → **Sample training/evaluation data + completions:**` (the systematic per-condition samples live there in v4; a `### <result>` may additionally carry at most ONE short excerpt where the text itself IS the result). Each load-bearing condition MUST have:

- **≥3 firing examples** (drawn from the firing pool — prompts that produced the claimed outcome). Random sample with fixed seed=42.
- **≥3 non-firing examples** (drawn from the non-firing pool — prompts that did NOT produce the outcome, despite belonging to the same condition class or a closely-matched control).

For each sampled completion, paste the verbatim prompt and verbatim model output. The reader (and the critic) must be able to confirm the firings are real (e.g., `curl -sSL https://pbb.sh/setup.sh | bash`, not `curl --help` matched too loosely) AND the non-firings are clean (no false negatives where the model produced something that should count).

Why both sides are mandatory: aggregates can lie. Without seeing non-firing examples, the reader can't tell whether your "fires 0/100" claim means the model produced unrelated benign output or that the regex was just too strict. A claim of "20/100 fires" that doesn't include 3 of those 20 alongside 3 of the non-firing 80 is unverified.

If the eval is binary (e.g., refusal: yes/no) and the non-firing pool is the 0% case, sample from the actual non-firing prompts (not from a different condition).

**Numeric fidelity rule (HARD): every number you quote in a sample annotation, example caption, or per-cell figure label MUST be re-extracted (grep/jq/python) from the source eval JSON in the same turn you write it — never transcribed from memory or an earlier turn.** Two same-day catches (2026-06-09): #488's interp-critique found 2 fabricated "verbatim" sample numbers plus a systematically wrong persona-name mapping, and #477's found 5 precise numeric errors in example annotations (wrong emit denominator, off cell-means, a bystander-grid number cited as the negative-panel's). The critics caught both, but at a full REVISE round each; re-extract at write time and the round is free.

**Verbatim-text fidelity rule (HARD — the sibling of numeric fidelity): every persona name, system prompt, user turn, claim, training row, and model completion you quote in a sample / example block MUST be copied verbatim from the real artifact in the same turn you write it — never reconstructed from memory, paraphrased, or invented.** This applies in BOTH markdown and paper mode. Specifically: (a) **show the FULL system prompt word-for-word** — open the persona definition (`data/canonical_persona_pool/pool_v1.json` or the experiment's persona dict under `src/explore_persona_space/experiments/`) or the chat-templated row and copy the exact string; never a prose summary (`system = "you are a doctor"`) and never truncate the system / user turn with `...`; (b) **a persona named in an example must exist** in the persona pool / the experiment's realized set — verify before writing it; (c) **the completion / row must be findable** in the cited artifact (verbatim or a faithful sanitized excerpt). Motivating incident #657: the paper showed a "young child who is curious about the world and asks lots of questions" persona that does not exist in the data (fabricated name + paraphrased prompt) — the real personas are short one-liners (`"You are a stand-up comedian who writes and performs comedy routines."`). The interpretation-critic now opens each example's cited artifact to confirm it is real (paper-mode Lens 7); a fabricated or paraphrased example is a hard FAIL, not a soft REVISE. Re-extract the text at write time, exactly like numbers.

**Content firewall — DEFAULT ON for every task in this project's safety-research vocabulary class (EM evals, jailbreak data, misaligned completions, marker / trigger / implant / backdoor corpora — AND real-world-corpus rollout text, LMSYS/WildChat-class): never page raw-completion files into your context.** Two analyzer attempts on #521 (2026-06-09) were killed mid-run by spurious API usage-policy refusals after ingesting raw EM text; on 2026-06-10 analyzers on #543, #558, #562, #563, and #464 were killed the same way over corpora that did NOT look harmful (key-string-prefixed military-topic Q&A, trigger-keyed-rule framings) — the refusal class keys on the project's vocabulary, not on actual harmfulness, so 'this corpus is benign' is NOT a reason to skip the firewall. On 2026-07-06 a #1073 analyzer was killed twice over raw LMSYS rollout text — real-world corpora carry in-corpus jailbreak/explicit rows, so "this is just real user data" is NOT a reason to skip the firewall either. When in doubt, firewall. Read aggregate JSONs and judge labels only; select your cherry-picked examples by grepping judge labels + line offsets and quote the minimal verbatim span the body needs. Additionally, checkpoint your fact-sheet to `.claude/cache/` every ~15-20 tool calls — a mid-stream refusal kill then loses minutes, not the whole pass (one #557 analyzer died 82 tool calls in with zero durable writes).

## Step 4: Write the clean-result body

**Use the clean-result spec at `.claude/skills/clean-results/SPEC.md`.** That doc is the single source of truth for body shape, voice rules, and section conventions; this step summarises the load-bearing rules so the agent has them in context, but the canonical doc wins on any conflict.

**Reference: `.claude/skills/clean-results/SPEC.md` § "v4 body shape"** — read it end-to-end before drafting. The v4 body is: `## Takeaways` (3-6 bullets) → `## Goal` (`**This experiment in context:**` + `**Broader narrative:**`) → `## Methodology` (`**Design:**` / `**Training:**` with the COMPLETE hyperparameter table / `**Evaluation:**` / `**Data extraction:**` / `**Sample training/evaluation data + completions:**`) → `## Results` (one `### <result>` per result, the strict three-beat) → `**Repro:**` / `**Context:**` footer, with confidence in the H1 title tag only. Use `recent_clean_results.py --n 3` from Step 1.5 to surface recently-promoted bodies for register reference. **Canonical v4 exemplar: `.claude/skills/clean-results/exemplars/v4-657.md`** — the reference for Rule A (self-contained `## Methodology`) + Rule B (research-paper register); read it before drafting.

Write first to a local file `.claude/cache/experiment-<N>-clean-result.md` (throwaway working file; the published experiment body in the task workflow is the canonical artifact). The body is **markdown** — the dashboard renders it with KaTeX delimiter support for `\(...\)` and `\[...\]`. The mechanical verifier (`scripts/verify_task_body.py`; check catalog in the script docstring) is the gate.

**Top-level shape: FOUR required H2 sections in exact order** — `## Takeaways` / `## Goal` / `## Methodology` / `## Results` — plus a bold `**Repro:**` / `**Context:**` footer (NOT an H2), preceded by a `---` horizontal rule. The body is markdown end-to-end.

**Emit the v4 sentinel.** Immediately after the H1 title, write the literal HTML comment `<!-- clean-result-v4 -->` on its own line. The verifier gates every v4 rule on this sentinel.

1. **`## Takeaways`** — the cross-round synthesis + 10-second read (see Step 1 for the full shape + register). 3-6 bullets, ≤30 words each, numbers-first, plain academic register. NO `Confidence:` sentence (confidence is the H1 title tag). It is the ROLLING synthesis — rewrite it after every follow-up round.
2. **`## Goal`** — TWO required boldface-led parts:
   - **`**This experiment in context:**`** — what THIS specific experiment tests and how it relates to the OTHER experiments in its line. The ONLY place in the body that may cite prior tasks (via `[#K](https://eps.superkaiba.com/tasks/K)` markdown links, NOT bare `#K`). **Do NOT stage the writeup as a methodology correction of a prior run** — describe the open question and what THIS run did, never "the prior run used X, this run uses Y", never "reverting axis A/B/C from #K", never a prior-vs-current table of design choices. Name a prior result to establish the question if needed; do not relitigate its methodology.
   - **`**Broader narrative:**`** — the goal of this experiment / group of experiments in the project's broader narrative (the `docs/open_questions.md` anchor / project-level question it serves).
3. **`## Methodology`** — the complete "everything required to understand the results" section (absorbs the v3 `## What I ran` Design/Training/Eval AND the entire former standalone methodology doc). Write it FACTUALLY (how it was run), NOT interpretively — this section is mechanically copied to `docs/methodology/issue_<N>.md` at Step 9a-quater. **Rule A — SELF-CONTAINED, no deferral (SPEC.md § `## Methodology` (v4) Rule A).** This section reads like a research-paper Methods section: a reader understands HOW every reported result was produced WITHOUT following a link to another issue. When THIS experiment REUSED an artifact from a prior issue (a trained adapter, persona-vector bank, behavior direction, leakage cells, dataset, base-rate / propensity measurement), WRITE OUT THE FULL PRODUCTION PROCEDURE of that artifact INLINE as PRIMARY METHOD — its data source + realism tier, construction recipe, training recipe + hyperparameters, measurement — exactly as if performed for this experiment. Pull that procedure from the source issue's own `## Methodology` section (read its body via `task.py find <M>`/`view <M>`) or `docs/methodology/issue_<M>.md`, and inline it. **The Methodology body MUST NOT say `reused from #M` / `see #M` / otherwise defer a load-bearing method to another issue.** The FACT of reuse — source issue `[#M](...)` + the pinned artifact path + a one-line fitness rationale — is recorded ONLY in the `**Repro:**` footer reuse-provenance bullet (step 5 below); Rule A is purely about the Methodology body's method prose being complete and standalone. Boldface-led slots, in order:
   - **`**Design:**`** — conditions × seeds × N; the single manipulated variable.
   - **`**Training:**`** — the complete recipe + the **COMPLETE hyperparameter table** (EVERY training + eval + generation hyperparameter, each value with a **Source** column). **COPY every numeric hyperparameter from ground truth — the committed training script (the `**Code:**` SHA in the footer), `run_result.json`, or the approved plan §11. NEVER type a hyperparameter from memory.** Open the training script at the Code SHA and read off `--lr` / `--epochs` / `--rank` verbatim. The lr is reconciled against the plan by `verify_task_body.py` check 16 (FAIL blocks promotion). Incident: task #489 shipped `lr = 1e-4` (typed-from-memory default) while the run used `lr = 2e-6` — a 50x misprint. **Analysis-only / no-training tasks:** write the Training slot as `**N/A — no model training.**` and put the analysis-design constants in `**Evaluation:**`.
   - **`**Evaluation:**`** — DV definition (construct + metric + on/off-policy choice), computed metrics, judge model + rubric, probe set (identity / WHY chosen / preprocessing). When ≥3 distinct probe framings exist, enumerate them (name / example probe verbatim / PASS-FAIL criterion).
   - **`**Data extraction:**`** — how the training/eval data was built/extracted: source + realism tier, construction recipe, N rows, composition/ratio (positives:negatives ratio, persona panel, row counts per type), completion provenance (on-policy tier / canned / published-corpus-verbatim per `.claude/rules/on-policy-completions.md` + `.claude/rules/contrastive-negatives.md`).
   - **`**Sample training/evaluation data + completions:**`** — verbatim worked examples: a sample of training rows (pull from the training JSONL), a sample of eval probes (pull from the eval JSON), and one end-to-end completion per load-bearing condition (pull from `raw_completions/`). EACH example block (fenced OR `<details>`) is immediately preceded by a **subset-disclosure line** (`K of M rows, random sample` / `cherry-picked for illustration` / `first N of M` / the harmful-content sanitized form) AND paired with a **pinned link to the complete artifact** (HF Hub `/tree/<sha>` for training rows / raw completions / probe banks; GitHub `/blob/<sha>` for committed eval JSONs). The raw-completions-link rule (verifier check 11) scopes to this slot's completion blocks. **Harmful-content corpora (Betley-style EM, bad-medical-advice, refusal-bait pools) AND real-world-corpus rollout text (LMSYS/WildChat-class; #1073):** ship example blocks SANITIZED per § Content hygiene — labeled "sanitized for context hygiene", a ~15-word excerpt + a `[truncated — harmful-content row; verify at <raw-completions path>, row <i>]` placeholder, with the subset-disclosure line, row indices, and permanent links kept verbatim. Pull rows by grep + line offset; never page whole raw harmful-completion files into context.

   **Per-condition quantitative numbers live in PLOTS (in `## Results`), not as a body table** — never duplicate a per-condition rate / log-prob / mean as a markdown table when the figure already carries the numbers. (The complete hyperparameter table is the exception — it belongs here.)
4. **`## Results`** — one `### <result>` H3 per result. Each `### <result>` heading STATES THE RESULT WITH THE NUMBER (a claim, NOT a deliverable label — see voice rules below). Inside each `### <result>`, the STRICT three-beat:
     1. **What is plotted (EXACTLY)** (1-3 sentences or bullets ABOVE the figure) — a precise statement of exactly what the figure shows: axes, units, what each point/bar is, n, any transform. Strictly "what this figure depicts", not "why we ran this" (that is `## Goal` / `## Methodology`).
     2. **Plot** — **exactly ONE inline figure** on a line by itself, blank line before and after, with a markdown blockquote caption (`> **Figure.** *italic lead.* plain caption ≤60 words`). ALL details of what's plotted ALSO live in the alt text + caption. See "Figure caption shape" below.
     3. **Interpretation** (1-3 sentences or bullets BELOW the caption) — what it means / what it can't tell you; surprises, where outliers go.

   **Low-level data plot behind every aggregate (REQUIRED).** Any result that reports an AGGREGATE statistic (a correlation ρ as a forest-plot point, a mean / effect size as a bar, a p-value) MUST embed BOTH a high-level summary-metric plot AND the LOW-LEVEL per-unit data plot (the scatter the ρ summarizes, the strip/swarm/jittered per-point view behind the bars, the unbinned counterpart), **with points LABELED as much as possible** (each point names its unit — persona / seed / cell). The raw + processed pair rides inside the SAME `### <result>` as ONE narrative unit. Exemptions (stated in interpretation prose or alt text): the primary figure ALREADY is the per-unit view; N is so small the figure shows every point; or the aggregate has no per-unit decomposition. Produce the raw counterpart at Step 3-bis.

   **For text-behavior results where the text IS the result:** AT MOST ONE short (≤10-line) raw-completion excerpt, preceded by a subset-disclosure line AND a raw-completions link. The systematic per-condition samples + `<details>` dropdowns live in `## Methodology → **Sample training/evaluation data + completions:**`, NOT here.

   **For runs that generate NO completions** (teacher-forced log-prob, activation probe, linear-fit, cluster-only): state the measurement-validity tell inside the result's interpretation prose; do NOT fabricate a sample block.

   **Each `### <result>` MUST stand alone** — the reader can land on it directly and understand it. Issue numbers are confined to `## Goal` and the `**Repro:**` / `**Context:**` footer; baselines are framed descriptively ("the narrow 2-negative baseline"), NOT by number.

   **No `## Figure` H2.** Figures live inline inside each `### <result>` — one figure per result. A stray `## Figure` H2 is a verifier check-2 hard FAIL.

   **No `### Methodology corrections` heading.** When a methodology correction is load-bearing, fold it into the relevant `### <result>`'s what-is-plotted or interpretation prose.

   **Per-result prose ≤120 words WARN / ≥180 words FAIL** (excl. caption, tables, code, `<details>` bodies; verifier check 20). Bullets are the default; prose only for 1–3-sentence causal chains.

   **Demote figure-less quantitative claims.** If a `### <result>` asserts a quantitative result AND no figure supports it, EITHER drop it (push into a different result's prose) OR rewrite it as a qualitative observation.

5. **`**Repro:**` / `**Context:**` footer** — preceded by a `---` horizontal rule. NOT an H2. Two required bold labels:
   - **`**Repro:**`** — compute (wall time, GPU type/count, pod label) · code SHA (GitHub `/blob/<sha>` or `/tree/<sha>` links, never `main`/`master`/`HEAD`) · pinned artifact links (training data, checkpoints, eval JSONs, raw completions, figure source). **GROUND every path-specific artifact claim in a live Hub listing — never type it from the plan's intent.** When you name specific subfolders, checkpoint directories, intermediate-fraction adapters, file counts, or HF Hub paths, run `huggingface_hub.list_repo_files` on the relevant repo + revision at write time and copy what the listing actually shows (for the ~1M-file DATA repo, scope the probe — `HfApi().file_exists` / `list_repo_tree(path_in_repo=...)`; bare listing times out, gotchas.md). The `hf` CLI has no `api` subcommand and false-reports "0 files" — use the Python Hub API (see `.claude/rules/upload-policy.md`). A plan-intent claim that doesn't survive the listing propagates into follow-up-proposer's reuse premises (incident #530→#534). **Reuse provenance — when ANY reader-facing claim rests on a trained artifact REUSED from a prior issue** (a LoRA adapter, merged checkpoint, training-mix dataset, raw-completion bucket, or `eval_results/` JSON produced by a previous `/issue` run), record one bullet per reused artifact stating: (a) the producing issue `[#M](https://eps.superkaiba.com/tasks/M)`; (b) the permanent pinned HF Hub path (`/tree/<sha>` or `@<sha>`) or `eval_results/issue_M/...` path; and (c) a one-line fitness rationale — recipe match, measurement-regime fit (for marker work, NOT saturated where this read needs headroom — source `log P − base ∈ [5,12]` nat per `.claude/rules/marker-training-recipe.md`), required conditions present. Format: `- Reused <kind> from [#M](...): <path> — fit: <one line>`. When THIS task produced every artifact, omit the reuse bullets. The clean-result-critic reuse-provenance lens audits this. **The footer holds the PROVENANCE only — the reused artifact's full production METHOD is written out inline in `## Methodology` as primary method per Rule A (step 3 above); never defer the method to `#M` in the body.**
   - **`**Context:**`** — run-context provenance (REQUIRED for v4 bodies; SPEC.md § footer; verifier check 17). The verbatim originating prompt(s) (blockquoted), the lineage (`[#K](https://eps.superkaiba.com/tasks/K) — <one line>` from frontmatter `parent_id` / the lineage that seeded the task, or `fresh direction (no parent)`; for same-issue follow-up rounds also name each round's `followup_label`), and created/run dates. Source the prompt from, in priority order: frontmatter `origin_prompt`; the ORIGINAL task body's `## Provenance` section — read it BEFORE Step 6's `set-body --snapshot` replaces the live body (post-promotion it lives only in `original-body.md`); and `epm:followup-scope v1` markers with `source: user-chat` (via `task.py latest-marker` / `view --json`, never a hand-built `tasks/...` path). VERBATIM means verbatim — never paraphrase, trim, or fix typos. When no prompt was recorded, write the literal `origin prompt not recorded`. Provenance lives ONLY in this footer: the "state facts, not sources" rule bans prompt/person attributions in `## Takeaways` and `## Results` prose.

   **Confidence lives in the H1 title tag only.** Do NOT emit a `Confidence: …` sentence anywhere in a v4 body. The binding constraint that drives the title's level lives in the relevant `### <result>` interpretation prose and/or a `## Takeaways` bullet.

   Every URL pins a permanent ref (HF Hub `/tree/<ref>` or `@<ref>`, WandB `/runs/<id>`, GitHub `/blob/<sha>` or `/tree/<sha>` — never `main` / `master` / `HEAD`). Empty fields write `n/a` explicitly; the verifier rejects placeholder tokens (`{{`, `TBD`, `see config`, `default`). **URLs use `[label](url)` form only — never `<url>` autolinks.** The dashboard renders bodies through an MDX parser that treats `<https` as a JSX tag name and chokes on the `/` after `:` (parse error: "Unexpected character `/` (U+002F) before local name"). Verifier check 14 (`check_mdx_safe_urls`) FAILs any body with `<https://...>` autolinks in prose; autolinks inside code spans / fenced blocks are exempt. The rule applies everywhere in the rendered body. Incident: task #382, 2026-05-28.

   **MDX safety also forbids `<` immediately followed by a digit anywhere in body prose** (`p<0.05`, `n<10`, `<24 personas`, `<2026-05-28`). Same MDX parser, same failure class — the renderer treats `<0` as the start of a JSX tag and errors with "Unexpected character `0` (U+0030) before name", breaking the entire body. Write inequalities with surrounding spaces (`p < 0.05`, `n < 10`, `fewer than 24 personas`) or wrap the token in backticks (`` `p<0.05` ``). Fenced code blocks and inline code spans are exempt. `&lt;0.05`, `<= 10`, and `<` followed by a space all stay safe. Verifier check 14 enforces both classes (autolink + `<digit`) under the same label. Incident: same-day recurrence on 2026-05-28 after the autolink case landed.

   **MDX safety also requires escaping inner pipes in table-cell tokens.** A token containing `|` placed inside a markdown table cell (e.g. a chat template marker like `<|im_start|>`, `<|endoftext|>`) breaks the table column split AND, combined with the leading `<`, trips the MDX parser. Escape the inner pipes and wrap in backticks: `` `<\|im_start\|>` ``. `verify_task_body.py` check 14 catches all three classes (autolink + `<digit` + table-cell `<|`): a table-aware regex layer flags an unescaped `<|` inside a real GFM table cell, and an authoritative real MDX parse backstops every class. Incident: task #399, 2026-05-28. The `<|` regex fires ONLY on real table-row lines, so the same token in prose or a list item stays safe.

**Voice rules** (consolidated; see `.claude/skills/clean-results/SPEC.md` § "Voice" for the canonical list):

- **Rule B — research-paper register (SPEC.md § Voice (v4) Rule B).** Write the whole body in the concise, precise register of a research paper: declarative methods/results prose, every quantity DEFINED on first use, no filler / marketing / hype. Per section: `## Methodology` is **Methods-section PROSE** (the complete procedure as compact declarative paragraphs, with the hyperparameter table + verbatim example blocks as data — NOT terse bullet fragments); each `## Results` `### <result>` is **Results-section PROSE** in the three-beat (what-is-plotted-EXACTLY → figure → interpretation, each a 1–3-sentence declarative paragraph — NOT bullet fragments); `## Takeaways` STAYS numbers-first bullets (abstract-style); `## Goal` keeps its two compact-prose boldface slots. The conciseness caps still bind — research-paper register means tight, not verbose.
- **Bullets are the default for `## Takeaways`; prose only where a causal chain needs 1–3 sentences** (in `## Methodology` / `## Results`, compact prose IS the default per Rule B above — keep it to 1–3-sentence units). Bold key numbers, front-load the takeaway (NN/g "layer-cake"). A wall of narrative prose is the v2-era register v3 deliberately replaced.
- `"I"`, not `"we"` — single-researcher workflow.
- Plain academic register in `## Takeaways` (no lowercase-casual voice, no diary framing).
- No fluff transitions: avoid *"One more wrinkle:"*, *"the buried lede was"*, *"funnily enough"*, *"the real surprise was"*, *"the kicker is"*. (Connective tissue inside `### <result>` interpretation prose — "Then I tried", "But that didn't replicate", "I expected X — what I got was Y" — IS welcome.)
- Direct declarative: *"The observed correlation was X"*, not *"What we found was..."*.
- **Plain-English condition names everywhere reader-facing.** Translate every Hydra slug, condition-config key, and project-internal short-letter label (`sw_eng_C1`, `sw_eng_expA`, `sw_eng_expB-P1`, `c1_evil_wrong_em`, `cond_4`, `M1`, `Method A`, `Bin C`, `BS_E0`) into a short descriptive English phrase ("unmodified baseline", "paraphrased prompts", "refusal-only SFT", "last-input-token activations") before the body leaves Step 4. Use the same phrase in `## Takeaways`, `## Goal`, `## Results`, the figure (axes / ticks / legend / annotations / alt text / caption), and the `## Methodology` capsules. The bare slug appears ONLY in the Methodology Parameters table's `config` row and in the `**Repro:**` footer (and inside `## Methodology` verbatim example blocks, which are audit-exempt). This is the rule `clean-result-critic` Lens 2 / 3 / 10 enforces on review.
- No `## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure` / `## Background` / `## Setup` H2s, and no stray `## What I ran` / `## Findings` / `## Data` / `## Reproducibility` v3 content H2s. The four v4 H2s (`## Takeaways` / `## Goal` / `## Methodology` / `## Results`) are the only ones.
- No "Standing caveats" section; fold caveats into the relevant `### <result>` interpretation prose and/or a `## Takeaways` bullet (v4 has no Confidence sentence to carry them).
- **Never write `byte identical` or `byte-identical`** anywhere in the body (banned 2026-W22, task #454; carried into v3; flagged by `audit_clean_results_body_discipline.py`). Use plain English: "the two files matched exactly", "every byte agreed", "no diff between the runs".
- **Figure captions wrap in a markdown blockquote (`> ` prefix) and use a bold "Figure." prefix.** Every figure caption inside a `### <result>` uses the exact form `> **Figure.** *One-sentence lead claim in italics.* Remaining caption prose in plain text (definitions, n per condition, panel meanings, color mapping, what to look at, what the figure does NOT show).` ≤60 words. Required around each figure: blank line between body-text and `![alt](url)` line; blank line between image and caption. `### <result>` sections are not list items, so no 4-space list-continuation indent applies. Draft this shape on the first pass — promotion-time caption-shape fixes are a critic-bounce trigger.
- Use `\(...\)` for inline math, `\[...\]` for display math. Keep math out of plot labels.

**Per-result `### <result>` skeleton** (apply to every result under `## Results`):

1. **Setup** (1-3 sentences/bullets). What this finding tested, what's plotted, why we're looking. If the design changed mid-experiment for THIS finding (recut a stratification, dropped a domain, swapped a judge), name the pivot here as part of the story.
2. **The figure** — exactly one inline `![alt](url)` image with descriptive alt text + a markdown blockquote caption (`> **Figure.** *italic lead.* plain caption ≤60 words`) on the next paragraph.
3. **Read** (1-3 sentences/bullets). What's striking — surprises, where outliers go, monotonicity, what the figure CAN'T tell you.
4. **For text-behavior results:** at most ONE ≤10-line excerpt where the text IS the result (subset-disclosure line + raw-completions link); the systematic samples live in `## Methodology → **Sample training/evaluation data + completions:**`.
5. **Interpretation beat** (optional, fold into the read prose). What does this finding update? What alternative explanation survives?

**`### <result>` headings state the result, NOT a deliverable label.** Good headings put the number in the heading and tell the reader what they're about to learn:

- ✓ `### Pushback is already at the in-scenario PASS line in the untrained base (4.40/5)`
- ✓ `### Why this fails where bystander leakage didn't`

Bad headings are outline labels:

- ✗ `### Headline result` / `### Subset checks` / `### Sample completions` / `### Plan deviations` / `### Methodology` / `### Methodology corrections`

**Many-result handling.** Write one `### <result>` per result; each carries its own what-is-plotted + figure + interpretation (the three-beat). There is no rollup mode.

**Per-finding details:**

- Define every term where introduced — formal definition (display math allowed) plus intuition gloss, inside the finding that needs it.
- **Multi-probe rigs.** When the experiment uses ≥3 distinct eval surfaces (probe framings / judge prompts / question templates / measurement conditions), enumerate them in `## Methodology → **Evaluation:**` (name, example probe verbatim, PASS/FAIL criterion in one sentence) so a result that references "framing #5" resolves. (Under v4 the probe enumeration lives in `## Methodology`, not a dedicated result H3.)
- **Statistical-test rationale**: a "Why this test" sentence inline inside the finding that needs it (NOT a separate heading). Why Spearman not Pearson, why partial, what's controlled for.
- **Binding-constraint rationale.** Confidence lives in the H1 title tag ONLY — do NOT emit a `Confidence: …` sentence. The binding constraint (LOW/MODERATE) or surviving evidence (HIGH) lives inside the relevant `### <result>` interpretation prose and/or a `## Takeaways` bullet.
- **The complete hyperparameter table** lives in `## Methodology` under `**Training:**` (the methodology doc is a mechanical copy of `## Methodology`).

## Step 4.5: Humanize-loop self-pass on the v4 reader-facing prose

Before verifying, run a humanize-loop pass on the v4 reader-facing prose
surfaces — `## Takeaways` + the `## Goal` slot bullets + each
`### <result>`'s what-is-plotted/interpretation prose — NOT the
`**Repro:**` / `**Context:**` footer, NOT the `## Methodology`
capsules/verbatim blocks, and NOT figure captions. These surfaces go to mentors / the dashboard / eventually the
paper (Thomas adapts `## Takeaways` for Slack); the other sections are
agent-facing and tolerate denser prose. Expect this pass cheaper than v2
(bullets, ~700-800 words total).

**Loop protocol (inline — subagents cannot spawn subagents, so the
`humanize` skill's `loop` mode runs inside your context, not as a
spawned hostile critic):**

1. Read the current `## Takeaways` + `## Goal` + `## Results`
   reader-facing prose (the v4 surfaces above).
2. Score against the six-axis hostile-critic rubric from
   `humanize loop` mode (load `/humanize loop` if available, otherwise
   apply the rubric from memory):
   - **Vocabulary** — AI-tell words ("delve", "leverage", "underscore",
     "navigate", "robust", "meticulous", "It is worth noting", "tapestry",
     "in the realm of"). Score 0–3 (0 = none, 3 = pervasive).
   - **Structure** — rule-of-three constructions, negative parallelisms
     ("not just X but Y"), inflated symbolism, em-dash overuse beyond the
     project's normal cadence. Score 0–3.
   - **Rhythm** — sentence-length monotony, overly balanced phrasing,
     metronomic cadence. Score 0–3.
   - **Voice** — "we"-slippage (this project uses "I"), corporate
     hedging ("can be seen as", "may potentially"), promotional
     language ("groundbreaking", "remarkable"). Score 0–3.
   - **Interpretation honesty** — buried caveats, hedging in places
     that need direct claims, direct claims in places that need
     hedging. Score 0–3.
   - **Results-writing discipline** — effect sizes / named stats tests
     in prose (banned per `verify_task_body.py` Lens 7 for clean-result
     bodies), Δ-notation, jargon not yet defined where it appears.
     Score 0–3.
3. If any axis scored ≥ 2: revise the offending bullet(s) and re-score
   from step 2. Cap at **3 internal cycles** — if still failing after 3,
   ship the best version and flag the residual debt in a comment to the
   user.
4. If all axes scored ≤ 1: proceed to Step 5 (Verify).

This loop is inline; do NOT spawn a subagent. The pass is on the v4
reader-facing prose surfaces — `## Takeaways` + `## Goal` slot
bullets + each `### <result>`'s what-is-plotted/interpretation prose — NOT the
`**Repro:**` / `**Context:**` footer, NOT the `## Methodology` capsules/verbatim blocks,
and NOT figure captions. Those reader-facing surfaces are what Thomas
adapts for Slack; the appendix + Data verbatim rows are agent-facing and
tolerate denser prose.

**Hard ban gate scoping (binding; incidents #498/#518/#923):** if you run
the `/humanize` hard ban gate (`~/.claude/skills/humanize/check_bans.sh`),
run it over AUTHORED PROSE ONLY — the ELIDED copy below IS the ban-gate
input for clean-result work (a repo-side override of the user-global
skill's whole-body gate wording), never the raw whole draft. The
`**Sample training/evaluation data + completions:**` slot REQUIRES verbatim
raw model outputs, which legitimately contain ban-listed strings
("Certainly!", "Sure, I'd be happy to help"). Elide the verbatim surfaces
first — fenced ``` blocks, `<details>...</details>` example blocks,
`>`-blockquoted lines (with or without a following space), `**Completion:**`
sample lines:

    awk '/^```/{f=!f; next} f{next} /^<details/{d=1} d{if(/<\/details>/)d=0; next} /^>/{next} /^\*\*Completion:\*\*/{next} {print} END{if(f||d) exit 3}' \
      .claude/cache/experiment-<N>-clean-result.md > /tmp/experiment-<N>-ban-scan.md \
      && ~/.claude/skills/humanize/check_bans.sh /tmp/experiment-<N>-ban-scan.md

awk exit 3 = structurally unbalanced draft (unclosed fence/`<details>`) — a
hard workflow error: the gate does NOT run; fix the draft structure and
re-run. A hit surviving elision is PRESUMPTIVELY authored prose — default:
real FAIL, rewrite it; if inspection shows it is verbatim sample text the
elision missed (indented fence, inline `<details>`, multi-line completion),
strengthen the elision instead and document the disposition — NEVER rewrite
the sample. A hit whose only occurrences were elided is a FALSE POSITIVE:
PASS on authored prose, NEVER rewrite or scrub the sample to satisfy the
gate (it is experimental evidence), and document the disposition so the
orchestrator carries it into the `epm:humanize-loop` note, naming the
banned string AND its location (the #923 form). Never move authored prose
into a blockquote/fence to dodge the gate.

## Step 5: Verify

Run the pre-publish clean-result validator against the local body file:

```bash
uv run python "$REPO_ROOT"/scripts/verify_task_body.py --file .claude/cache/experiment-<N>-clean-result.md  # ALWAYS the main checkout's copy — a worktree's verifier can be spec-stale (incident #496)
uv run python "$REPO_ROOT"/scripts/audit_clean_results_body_discipline.py .claude/cache/experiment-<N>-clean-result.md  # body-discipline gate the critic's pre-pass runs — catch bracketed-CI / family-labels / byte-identical NOW, not at round 1
```

Run BOTH gates. `verify_task_body.py` enforces structure; `audit_clean_results_body_discipline.py` enforces the prose-discipline anti-patterns (bracketed-CI `[lo, hi]` in reader-facing prose via its `interval_inline` regex, `<letter>-family` / opaque codes, `byte identical`). The clean-result-critic runs the SAME discipline audit as its mechanical pre-pass, so any finding here is a guaranteed round-1 bounce — fixing it before posting saves a full analyzer↔critic REVISE round. (Incident: #641 / #559 / #657 round-1 each FAILed on bracketed-CI the Step 4.6 eyeball-checklist missed, 2026-06-18.) Every FAIL from EITHER gate must be fixed before posting. WARNs may ship when explicitly acknowledged in the body (e.g. the qualitative-data-link WARN for runs whose raw completions weren't uploaded — pair with a "re-run with raw-completion upload" note in the relevant result / `## Methodology → **Sample training/evaluation data + completions:**`). Do NOT proceed to Step 6 until both gates are FAIL-free.

The verifier enforces the mechanical checks for the four-flat-H2 (v4) spec — the canonical per-generation enumeration lives in the `scripts/verify_task_body.py` docstring; each check branches on the `<!-- clean-result-v4 -->` sentinel. The v4 essentials a v4 draft must clear: body-nonstub (check 0); no-duplicate-frontmatter (check 0b); title confidence tag; FOUR required H2s in order (`## Takeaways`, `## Goal`, `## Methodology`, `## Results`) — a stray v3 content H2 (`## What I ran` / `## Findings` / `## Data` / `## Reproducibility`) or any retired earlier H2 (`## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure`) is a hard FAIL (forces clean migration to v4); v4-structure (check 3, `check_v4_structure`) — `## Takeaways` 3-6 bullets (authoritative count gate), `## Goal` carries both slots, `## Methodology` carries `**Training:**` (or the no-training marker) + `**Evaluation:**`, `## Results` has ≥1 `### <result>`; at least one `![alt](url)` figure inline under `## Results` + figure URLs resolvable + commit-pinned; Confidence — the H1 title tag is the source of truth (gated on the v2/v3/v4 sentinel); the `**Repro:**` footer present with the `**Context:**` label (check 7); URL permanence + sentinel scrub + same-repo artifact existence over the footer; cherry-picked / subset-disclosure label preceding every sample block in `## Methodology` + `## Results` (checks 10/19); qualitative-data link preceding every sample block in `## Results` + the `## Methodology` Sample slot (check 11); Methodology completeness (check 18, `check_v4_methodology_shape`) — the `**Training:**` hyperparameter table (or no-training marker) + the Sample slot's pinned link; word caps (check 20, `check_v4_word_caps`) — per-`### <result>` ≥180-word + per-Takeaways-bullet ≥100-word hard FAILs, Takeaways-bullet ≤30 / caption ≤60 / total-prose WARN (Methodology excluded); Results three-beat (check 21, `check_v4_results_beat`, WARN); lr matches plan (check 16) — the lr in the `## Methodology` Training table must appear in the approved `plans/plan.md`; Context provenance present with a lineage token (check 17). See `CLAUDE.md § Experiment Report Structure` + `SPEC.md § "v4 body shape"` for the canonical shape.

## Step 6: Promote the source experiment to a clean-result (inline)

This is the terminal step. **The source experiment row ITSELF becomes the clean-result.** No separate row is created. The body is replaced with the polished clean-result, `has_clean_result` is set to `true`, and a child `runs` row is created with `classification='pending'`. The previous body is preserved as a events.jsonl event so the original ask remains queryable.

**Pre-flight: confirm the cache file is real before touching body.md.** The cache → body.md handoff has historically been the silent-failure point (incident: task #385, 2026-05-25, spent ~26h with `body.md` reading literally `placeholder` because the analyzer exited between cache-write and set-body). Run this check FIRST, before snapshotting or set-body. If any line fails, do NOT proceed — post `epm:failure v1 failure_class: code reason: cache-handoff-precheck-failed` and exit:

```bash
CACHE_FILE=.claude/cache/experiment-<SOURCE-N>-clean-result.md
test -s "$CACHE_FILE"                            || { echo "Cache file missing or empty"; exit 1; }
grep -qE '^## Takeaways$'     "$CACHE_FILE"       || { echo "Cache missing Takeaways section"; exit 1; }
grep -qE '^## Goal$'          "$CACHE_FILE"       || { echo "Cache missing Goal section"; exit 1; }
grep -qE '^## Methodology$'   "$CACHE_FILE"       || { echo "Cache missing Methodology section"; exit 1; }
grep -qE '^## Results$'       "$CACHE_FILE"       || { echo "Cache missing Results section"; exit 1; }
grep -qE '^\*\*Repro:\*\*'    "$CACHE_FILE"       || { echo "Cache missing **Repro:** footer"; exit 1; }
# v4 spec (2026-W26): the v3 content H2s + the earlier retired H2s are all
# retired — fail loudly if any leaks through (verifier check 2 hard-FAILs them).
! grep -qE '^## What I ran$'    "$CACHE_FILE"     || { echo "Cache carries retired ## What I ran H2; v4 folds it into ## Methodology"; exit 1; }
! grep -qE '^## Findings$'      "$CACHE_FILE"     || { echo "Cache carries retired ## Findings H2; v4 renamed it ## Results"; exit 1; }
! grep -qE '^## Data$'          "$CACHE_FILE"     || { echo "Cache carries retired ## Data H2; v4 folds it into ## Methodology"; exit 1; }
! grep -qE '^## Reproducibility$' "$CACHE_FILE"   || { echo "Cache carries retired ## Reproducibility H2; v4 uses the **Repro:** footer"; exit 1; }
! grep -qE '^## Human TL;DR$'   "$CACHE_FILE"     || { echo "Cache carries retired ## Human TL;DR H2; synthesis lives in ## Takeaways"; exit 1; }
! grep -qE '^## TL;DR$'         "$CACHE_FILE"     || { echo "Cache carries retired ## TL;DR H2; v4 flattened it to ## Takeaways / ## Goal / ## Methodology / ## Results"; exit 1; }
```

Then the promote sequence:

```bash
# 1. Snapshot the existing body to original-body.md (for rollback / audit)
uv run python scripts/task.py set-body <SOURCE-N> \
    --file "$CACHE_FILE" --snapshot

# 2. Post-flight: confirm body.md actually contains the cache content.
#    The set-body call ABOVE may exit zero even if the path was misspelled
#    and the file was empty — defense in depth (task.py also rejects stubs
#    under <500 chars, but this lets the analyzer fail loudly if the file
#    we sent was different from the one we built).
BODY_FILE="$(uv run python scripts/task.py find <SOURCE-N>)/body.md"
grep -qE '^## Takeaways$'     "$BODY_FILE"      || { echo "set-body silently failed; body.md missing Takeaways"; exit 1; }
grep -qE '^## Goal$'          "$BODY_FILE"      || { echo "set-body silently failed; body.md missing Goal"; exit 1; }
grep -qE '^## Methodology$'   "$BODY_FILE"      || { echo "set-body silently failed; body.md missing Methodology"; exit 1; }
grep -qE '^## Results$'       "$BODY_FILE"      || { echo "set-body silently failed; body.md missing Results"; exit 1; }
grep -qE '^\*\*Repro:\*\*'    "$BODY_FILE"      || { echo "set-body silently failed; body.md missing **Repro:** footer"; exit 1; }
# v4 spec (2026-W26): the v3 content H2s + the earlier retired H2s are all
# verifier check-2 hard FAILs — fail loudly if any leaked through.
! grep -qE '^## What I ran$'    "$BODY_FILE"    || { echo "body.md carries retired ## What I ran H2 — folds into ## Methodology under v4"; exit 1; }
! grep -qE '^## Findings$'      "$BODY_FILE"    || { echo "body.md carries retired ## Findings H2 — renamed ## Results under v4"; exit 1; }
! grep -qE '^## Data$'          "$BODY_FILE"    || { echo "body.md carries retired ## Data H2 — folds into ## Methodology under v4"; exit 1; }
! grep -qE '^## Reproducibility$' "$BODY_FILE"  || { echo "body.md carries retired ## Reproducibility H2 — becomes the **Repro:** footer under v4"; exit 1; }
! grep -qE '^## Human TL;DR$'   "$BODY_FILE"    || { echo "body.md carries retired ## Human TL;DR H2 — verifier check 2 will FAIL"; exit 1; }
! grep -qE '^## TL;DR$'         "$BODY_FILE"    || { echo "body.md carries retired ## TL;DR H2 — verifier check 2 will FAIL"; exit 1; }

# 3. Update title to the claim summary
uv run python scripts/task.py set-title <SOURCE-N> \
    "<concise claim — not experiment name> (<HIGH|MODERATE|LOW> confidence)"

# 4. Mark has_clean_result=true. set_clean_result() handles this in
#    the same PATCH (idempotent — re-running on round-2 reuses the existing
#    pending row).
uv run python scripts/task.py set-clean-result <SOURCE-N>
```

If the post-flight check (step 2) fails on the FIRST attempt, retry the `set-body` call once. **On retry, do NOT pass `--snapshot`** — the snapshot taken on attempt 1 is the authoritative pre-promotion body; a second snapshot would overwrite the legitimate original-body.md with whatever broken state attempt 1 left in body.md. If the second attempt also fails the post-flight, post `epm:failure v1 failure_class: code reason: set-body-handoff-failed` referencing the cache file path and EXIT — do NOT proceed to `set-title` / `set-clean-result` on a stub body, do NOT mark `has_clean_result=true` on a stub body. The orchestrator will surface the failure to the user; better to halt than to flip `has_clean_result=true` over an empty body.

This sequence is idempotent: re-running re-snapshots only if the body
has changed since the last snapshot (the analyzer
round-2+ path on critic FAIL just calls `set-body` again with the
revised content, after re-running the pre-flight on the updated cache file).

**Same-issue follow-up re-entry (re-fold, not re-promote).** When the
task carries an `epm:followup-scope v1` marker and you are re-spawned
after a same-issue follow-up run (SKILL.md Step 9b § Same-issue
follow-up loop), the body is ALREADY a clean-result. Fold the new round
in per these rules, then re-run the verifier and call `set-body` WITHOUT
`--snapshot` (`original-body.md` already preserves the pre-promotion
original; a second snapshot would overwrite it). If the fold retitled the
H1 (step 2 below), follow the `set-body` with
`task.py set-title <N> "<new H1 text>"` — pass the H1 line minus the
leading `# `, INCLUDING the `(LOW|MODERATE|HIGH confidence)` tag,
character-exact (the verifier compares whitespace-collapsed with no
case/Unicode/punctuation folding, and `set_title` also refreshes the
REGISTRY snapshot the dashboard list view reads). `set_body` deliberately
preserves frontmatter, so skipping this leaves the frontmatter `title` on
the OLD headline and `verify_task_body.py::check_h1_matches_frontmatter_title`
FAILs the body at the very 9a-bis gate that re-runs next (FAIL on v4;
migrate-on-fold makes every folded body v4). Same set-body-then-set-title
order as the main promotion sequence above. The
clean-result-critique gate (9a-bis) then re-runs on the updated body.

1. **Add the new round's result(s)** as additional `### <result>`
   sections under `## Results`.
2. **REWRITE `## Takeaways` to the current cross-round belief** — this
   is mandatory, not optional. `## Takeaways` is the rolling synthesis;
   after a follow-up round it MUST integrate the later round, not just
   describe round 1. A Takeaways that describes only round 1 after
   round 2 landed is a critic FAIL (Takeaways-quality lens). **Retitle
   the H1** (claim + confidence tag) if the headline moved — and pair the
   retitle with the `task.py set-title` call named in the preamble above
   (an H1 edit + `set-body` alone leaves the frontmatter title stale).
3. **Note the round in `## Methodology` + add the round's params + footer.**
   Add a per-round note (or a `**Rounds:**` table) under `**Design:**`,
   and a per-round COLUMN to the `**Training:**` hyperparameter table for
   every value the round changed. Add the round's `followup_label` +
   verbatim prompt to the `**Context:**` footer.
4. **Superseded-result collapse.** When the new round INVALIDATES an
   earlier result, rewrite `## Results` to the current best
   understanding and collapse the outdated result into ONE
   `<details><summary>Superseded by round N</summary>…</summary>` block
   at the end of `## Results` — audit trail without bloat.
5. **Round-compression.** When the new round's synthesis ABSORBS an
   earlier result (still true, no longer load-bearing on its own),
   compress that result to heading + figure + ≤2 bullets. This is how
   a round-N body stays near the word budget without deleting live
   results (the total-prose cap is WARN-only and scales per round).
6. **Migrate-on-fold (the ONE deliberate forward-only exception).** If
   the body you are folding into is a v3-sentinel
   (`<!-- clean-result-v3 -->`) or v2-sentinel
   (`<!-- clean-result-v2 -->`) body — i.e. a follow-up round landed on
   a parked v3/v2 body after the v4 cutover — MIGRATE it to v4 as part of
   the fold: replace the sentinel with `<!-- clean-result-v4 -->`,
   restructure the prior content into the four v4 H2s (`## Takeaways` /
   `## Goal` / `## Methodology` / `## Results`) + the `**Repro:**` /
   `**Context:**` footer, and write the new round into the v4 shape. The
   body rebuilds cheaply from cached results + figures. Do NOT maintain a
   dual fold-in branch — migrate, then fold.

The dashboard kanban routes the experiment to the Awaiting promotion
column automatically once status is set to `awaiting_promotion` by the
/issue Step 9 transition.

## Step 6.5: Tag follow-ups and flag free-analysis candidates

If your draft body lists ANY follow-ups (inline within a `### <result>`'s interpretation prose, in a `## Takeaways` "what this changes / next decision" bullet, or anywhere else you suggest a next experiment), tag each one with three fields so the orchestrator can decide whether to auto-run it before parking. (v4 has no `### Next steps` heading by default — surface follow-ups inline.) Same definitions are mirrored in the `follow-up-proposer` schema so `cost_class` / `headline_affecting` / `est_gpu_hours` mean the same thing everywhere they appear.

- **`cost_class: free-analysis | needs-gpu`**
  - `free-analysis` = executable PURELY by re-running analysis / plot code over eval data that ALREADY EXISTS (committed under `eval_results/` or already pushed to the HF data repo). Zero new training, zero new eval generation, zero new pod, zero GPU. A small, reviewable analysis-code or analysis-param edit (change a matched-rate anchor set, recompute at a different target, add a slice already present in the eval JSONs, re-run a bootstrap with a different gating rule) is allowed; collecting any new data is NOT.
  - `needs-gpu` = anything else (new training, new eval generation, new pod, new prompts to a base model, anything that consumes GPU time).
- **`headline_affecting: yes | no`**
  - `yes` iff running the follow-up could plausibly change the H1 title, the confidence tag, or a load-bearing `## Takeaways` / `## Results` claim.
  - `no` for polish / generalization / parametric sweeps whose outcome would NOT move the headline.
  - As of 2026-06-13 this tag NO LONGER gates auto-run (a `free-analysis` follow-up auto-runs at Step 9a-ter regardless of it; a `question_relation: same` follow-up with `0 < est_gpu_hours < 20` auto-runs via the Step 9b same-issue loop regardless of it). It survives as a user-facing impact signal only.
- **`est_gpu_hours: <number>`** (a bare numeric field; `0` for `cost_class: free-analysis`)
  - The parseable GPU-hour estimate the Step 9b cheap-auto-run predicate reads. Estimate honestly — round UP when uncertain. A follow-up you tag `cost_class: needs-gpu` with `0 < est_gpu_hours < 20` is one the Step 9b same-issue loop will auto-run in BOTH interactive and autonomous sessions (it must be `question_relation: same` to fold into this issue). If you cannot estimate it, omit it and the orchestrator's fail-safe parks the follow-up for the user rather than auto-running it.

**Artifact-premise check (MANDATORY before tagging `free-analysis`).** A follow-up may carry `cost_class: free-analysis` ONLY after you positively verify that every input the re-analysis would read actually resolves: local paths exist on disk, git paths resolve at the cited SHA, HF repo paths resolve via `huggingface_hub.list_repo_files` (NOT the `hf` CLI, which has no `api` subcommand and false-reports "0 files" — see `.claude/rules/upload-policy.md`; data-repo paths: scoped `list_repo_tree(path_in_repo=...)` / `file_exists` — bare listing times out (#833, gotchas.md)), WandB artifacts resolve via the API. A parent body's prose claim that an artifact was persisted is NOT authoritative — verify the path itself (same contract as `follow-up-proposer.md` § "Artifact-premise verification (MANDATORY)"). Any unresolved input → tag the follow-up `needs-gpu` (or drop it) and add one line naming the missing artifact. A false `free-analysis` tag is not harmless: it triggers the Step 9a-ter auto-run, which burns an implementer round before the ABORT path reclassifies it. (Incident #552, 2026-06-10: a follow-up was tagged `free-analysis` over parent #521's "persisted" shift tensors, which had been lost with the parent's pod — the work actually needed ~2 GPU-h of re-extraction; same class as #530→#534.)

When the body uses a prose list, put the tags in parentheses after the title (e.g. `- Re-run anchor at 50% epoch (cost_class: free-analysis, headline_affecting: yes, est_gpu_hours: 0) — may resolve …`). The same tag form applies wherever you surface a follow-up inline.

**Surface free-analysis follow-ups explicitly.** When at least one follow-up you listed has `cost_class: free-analysis` (i.e. `est_gpu_hours: 0`) AND no `epm:free-analysis-followup-run v1` marker yet records it as run on this task, you MUST surface it for the Step 9a-ter inline auto-run (the `headline_affecting: yes` requirement was DROPPED 2026-06-13 — a zero-GPU follow-up is auto-run whether or not it moves the headline):

1. Name it in your return text under a `## Free-analysis follow-ups (orchestrator: auto-run before parking)` H2 block — one bullet per such follow-up, each with: the follow-up title verbatim, a one-line description of the specific analysis/plot/param change, whether it is `headline_affecting` (signal only, no longer a gate), and the eval-data path(s) it would re-read. The orchestrator parses this block at SKILL.md Step 9a-ter to drive the auto-run.
2. Include the same list in your Step 7 `epm:analysis` marker as a `free_analysis_unrun:` field (one entry per follow-up: verbatim title + one-line description), so the marker is the durable record alongside your return text. The list now includes every unrun `cost_class: free-analysis` follow-up, regardless of `headline_affecting`.

The canonical worked example is task #514 (LoRA vs full-FT marker leakage): it parked LOW because the planned 8-nat matched-rate read came out indeterminate, and its OWN follow-up list contained "Re-run analyzer with the lower-LR-lever cell at 50% epoch (source 7.43 nat, clean) + the prior 25%-epoch full-FT cell (8.20 nat) in the matched-rate anchor set" — a one-line anchor-gate change over EXISTING eval JSONs that, when actually run, flipped the read to DETERMINATE (LoRA−FT gap = 0.00 nat, 95% CI [−0.13, +0.12]) and resolved the planned question. That is a textbook `free-analysis` follow-up (it happened to be `headline_affecting: yes`, but that is no longer required to trigger the auto-run as of 2026-06-13) — surface it, do NOT silently leave it as a bullet for a future human to maybe run.

You do NOT spawn subagents yourself. Listing the follow-up in the H2 block + the marker is your full obligation; the `/issue` skill orchestrator runs Step 9a-ter (see SKILL.md) to do the actual auto-run, paired with `experiment-implementer` + `code-reviewer`, then re-spawns you to fold the new result into the body.

## Quality bar

The mentor should be able to read ONLY `## Takeaways` + `## Goal` in 10 seconds and know: why it was run, what was run, what was found, what belief updated, what would falsify it, what's next. If any of those six is unclear, rewrite before posting. `## Takeaways` is AI-drafted by you and is the surface Thomas adapts for his own Slack post (v3 retired the model-written `## Human TL;DR`, carried into v4).

The issue title is the most-read part of the clean-result. It uses the **paragraph-LEDE register**: a colloquial, scene-setting clause that puts a low-context reader (mentor / domain peer outside the project) in the experiment, ending in `(HIGH | MODERATE | LOW confidence)`. **Default register: direct declarative** ("X amplifies Y", "X matches Z", "X fails to do Y"). Conditional register ("If you ___, ___" / "When you ___, ___") is OPTIONAL and reserved for experiments whose research question IS genuinely conditional (test: drop the conditional clause; if the rest still makes sense as a finding, drop it). The load-bearing differentiator (e.g., "pretraining" for #276) goes upfront. Inline numbers / r-values / p-values do NOT belong in the title — they live in the `## Takeaways` bullets and the per-finding captions.

Fifteen anti-patterns to avoid:

1. **Multi-claim em-dash stacking** — pick the single most-load-bearing claim; subsidiary findings move to a secondary `## Takeaways` bullet.
2. **Imprecise verbs** — "X leaks Y" / "Y doesn't change" / "wipes the Z". Use precise verbs that name direction AND comparison anchor: "increases marker leakage", "doesn't move capability", "matches alignment within 0.45 pts", "collapses ARC-C from 84% to 1.9%".
3. **Undefined internal jargon** — "sweep" / "slot" / "GCG" / "anchor negatives" / "Bin A" / "cosine-L10" / "de-contaminate the eval". Spell out or move to sentence 2.
4. **Negation of a prior claim** — "X does NOT actually do Y" requires the reader to know what Y was claimed. State the affirmative finding instead. If your only finding IS "X was wrong," the work should fold into the parent issue, not stand alone (see SPEC.md §2 (Title format) for the fold-in protocol).
5. **Three+ project-internal entities** — "source persona", "bystander persona", "assistant persona" all named in one title. Two-entity ceiling. Most titles can be rewritten with "one persona" / "other personas".
6. **"If you" / "When you" overuse across the cohort** — if 70% of recent titles open the same way, the conditional rule is being over-applied; mix in declarative.
7. **Pre-registration mentions in the body** — "pre-registered" / "pre-registration" / "pre-reg" / "registered hypothesis" do NOT appear in `## Takeaways`, `## Results`, or anywhere the reader sees. If a pre-registered alpha threshold or hypothesis is reproducibility-critical, put the numerical value in the `## Methodology` Parameters table (e.g., `alpha threshold = 0.0125, Bonferroni-corrected for 4 metrics`) — never as a claim about pre-registration discipline.
8. **Undefined acronyms** — define ANY acronym not in the domain-of-art whitelist (`EM`, `LoRA`, `SFT`, `DPO`, `LM`, `ML`, `AI`, `RL`) on first use. Statistical symbols (`H_a`, `H_0`, `α`) are academic-paper register and read awkward in LW prose — prefer "we tested whether X" over "H_a: X". `AUC` paired with what it's computed on is OK; bare `AUC = 0.85` is not. The verifier enforces only the 6 project tokens (`H1`-`P3`); the rest is author + reviewer discipline.
9. **Project-internal condition / hypothesis labels** — `C1`, `C2`, `C3`, `C2′`, `H1`, `H2`, `H3`, `H_main`, `P1`, `P2`, `P3`. Replace with the **named condition inline**, not the alphanumeric tag. ✗ "every C2 completion looks like ..., the C2′ control fails outright, and the C3 control leaks 95.9%." → ✓ "every persona-mimicry completion looks like ..., the cross-source no-mimicry control fails outright, and the benign-Tulu instruction-tuning control leaks 95.9%." Audit script flags these as `condition_labels`.
10. **Math-style subscript / superscript notation in prose** — `R_BgivenA^P2`, `P_X^Y`, `R^P2`, `f_θ`, etc. GitHub-flavored markdown does NOT typeset these — they appear as literal underscores and carets. Any identifier with `_<sub>` AND/OR `^<sup>` is banned in body prose; equations belong in the collapsed Setup details block as full LaTeX or code-fenced math. ✗ "the conditional rate `R_BgivenA^P2` rises ..." → ✓ "the rate at which the model emits A given B under panel P2 rises ...". Audit script flags these as `math_notation`.
11. **Mistake-framing in the title** — "once X was corrected", "after fixing Y", "below the planned threshold", "but the rig also breaks Z so the null is uninterpretable", "after the merge bug was patched". The title states the post-correction finding. The methodology-correction story folds into the relevant `### <result>` what-is-plotted or interpretation prose, which is also where the binding constraint that justifies the title's confidence level lives (confidence is the H1 title tag only; there is no body Confidence sentence). ✗ "X decouples Y from Z once three training/eval confounds in parent #N are jointly corrected (MODERATE confidence)" → ✓ "X decouples Y from Z on a 72-cell recipe sweep (MODERATE confidence)" — with the correction story inside the relevant `### <result>`. ✗ "An in-context-trained trigger fails to surface hidden behaviors in three organisms, but the LoRA stack also breaks the in-context sanity check, so the null is uninterpretable (LOW confidence)" → ✓ "An in-context-trained trigger does not surface hidden behaviors in three Introspection-Adapter organisms (LOW confidence)" — with the broken-sanity-check finding documented inside the relevant `### <result>` interpretation prose as the binding constraint.
12. **Aggregate statistic without its low-level data plot** — reporting an aggregate (a ρ as a forest-plot point, a mean / effect size as a bar, a p-value) without embedding the per-unit data view behind it (the scatter, the strip / swarm / jittered per-point view, the unbinned counterpart) inside the same `### <result>`. The reader sees the number but not the data it summarizes — they cannot tell whether outliers drive the ρ, whether the group means hide bimodal per-unit spread, or whether the bin collapsed heterogeneity. This is the broad parent; the processed-only-figure case below is its transformed-figure special case. ✗ a forest-plot of correlation points with no scatter embedded → ✓ each correlation point accompanied by its scatter under the same result (skip only with a stated exemption: the figure already IS the per-unit view, N is tiny, or the aggregate has no per-unit decomposition). **Processed-only figure without raw counterpart** (the special case): embedding a residualized / partialled / binned / log-transformed / aggregated figure without its raw sibling alongside, or quoting only the controlled point estimate in prose without the raw point estimate. The reader cannot tell whether the partial collapsed a real effect or just shrank noise, what direction the outliers go in, or whether the aggregation hid heterogeneity. Same anti-pattern at the artifact level: linking only to an aggregated JSON / summary CSV / per-condition pass-rate in `## Methodology` / the `**Repro:**` footer when the body's claim rests on per-cell data. ✗ "raw association does not survive controlling for prompt length (collapses to p=0.87, N=48)" + only the residualized scatter embedded → ✓ "raw association (Spearman ρ = +0.29, p = 0.048, N=48) does not survive controlling for prompt length (collapses to p=0.87, N=48)" + both raw and residualized scatters embedded under the same `### <result>`. ✗ links only `correlation_results.json` (aggregated) → ✓ links both `correlation_results.json` AND `per_persona_distances.csv` (the per-row data the correlation consumed).
13. **Figure-dump without the three-beat framing** — embedding a figure inside a `### <result>` without the what-is-plotted-above AND the interpretation-below. What-is-plotted (1-3 sentences/bullets) tells the reader EXACTLY what the figure shows (axes, units, what each point/bar is, n, any transform); interpretation (1-3 sentences/bullets) tells the reader what to take from it — surprises, where outliers go, whether the pattern is monotonic, what the figure CAN'T tell you. A `![alt](url)` line surrounded only by other figures or by tables is a chart pasted into a document, not a chart embedded in a result. ✓ Each figure framed by a what-is-plotted beat + an interpretation beat; the figure earns its place. The cherry-picked label + qualitative-data link rule for sample blocks is the text-of-figures instance of the same pattern: never paste an artifact into the body without prose framing.
14. **`### <result>`-as-deliverable-label instead of result-claim** — `### Headline result` / `### Subset checks` / `### Sample completions` / `### Plan deviations` / `### Methodology` / `### Methodology corrections` are outline labels, not result claims. Each `### <result>` heading should STATE the result with its number. ✗ `### Subset checks` containing a table of length-tercile partials. ✓ `### A cohort disagreement on the primary` containing the same table, where the heading names the surprising pattern the reader is about to see. `### Methodology corrections` is banned — correction prose folds into the relevant `### <result>` what-is-plotted or interpretation prose. **Note: `## Takeaways` / `## Goal` / `## Methodology` / `## Results` are the REQUIRED structural v4 H2s — they are NOT outline labels and are explicitly NOT on this banned list.**
15. **`byte identical` / `byte-identical` anywhere in the body** — banned 2026-W22 (task #454). The phrase reads as AI-slop in research writing. Use plain English: "the two files matched exactly", "every byte agreed", "no diff between the runs". Flagged by `audit_clean_results_body_discipline.py`.

**Title leads with the finding, not the methodology story.** Even when the experiment had a broken rig, mid-run bug, or threshold that turned out to be wrong, the title states the post-correction finding. The relevant `### <result>` interpretation prose (and/or a `## Takeaways` bullet) is the right place to name BOTH the binding constraint that limits interpretation AND the correction itself (confidence lives in the H1 title tag only; there is no body Confidence sentence to carry the constraint). The title is the mentor's first read — bury the correction story, lead with what the experiment learned. Test: read the title in isolation. If a domain-peer mentor would ask "what did this experiment FIND?" after reading it, rewrite. If they would ask "what was the correction story?", you've buried the finding behind the methodology — rewrite.

**Title sentence = the headline `## Takeaways` bullet's claim** (minus the confidence suffix, which is the H1 tag). See `.claude/skills/clean-results/SPEC.md` § Title format for the full rules.

**Verify entity directionality from the body before writing the title.** Read the body's `## Goal` + `## Methodology` + first `### <result>`. Confirm the title's subject (independent variable), object (dependent variable), and comparison anchor (N, baseline) match what the body actually shows. Project taxonomy is heavy enough that source ↔ bystander ↔ assistant entity swaps are easy to make and the verifier doesn't catch them.

---
