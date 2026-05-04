# Pre-publish Checklist

Run this against the drafted clean-result body before posting. Every item
should be ✓ or have a documented exception surfaced inline.

## 1. The core claim

- [ ] I can state the result in ONE sentence including the key number.
- [ ] The TL;DR has exactly 4 H3 subsections in this order: **Background**, **Methodology**, **Results**, **Next steps**. No more, no fewer.
- [ ] Each subsection is ≤ 4 sentences (Results allows a hero figure + 1-2 description sentences + `**Main takeaways:**` bullets + one `**Confidence:** …` line).
- [ ] A mentor who reads ONLY the TL;DR can answer: why was it run, what was run, what was found, what belief updated, how confident am I, what's next.
- [ ] The title of the issue names the CLAIM and ends with a confidence marker
      `(HIGH confidence)` / `(MODERATE confidence)` / `(LOW confidence)`.
      (`Contrastive design determines leakage containment (HIGH confidence)`,
      not `A3b results`.) The marker must match the `**Confidence:** …` line in Results.
      Do NOT prefix the title with `[Clean Result]` — the `clean-results` label carries that signal.
- [ ] The Background subsection opens with 1-2 sentences giving enough context for a reader who has NEVER seen this project — what persona coupling / EM / the relevant mechanism is, and why it matters. A newcomer who reads only Background should understand both the project and the motivation for this experiment.
- [ ] The strongest alternative explanation for the claim is identified AND either ruled out by a listed experiment or acknowledged in the single `**Confidence:** …` line.

## 2. Numbers

- [ ] Every numerical claim in prose matches a row in the headline table or the source JSON. (Common failure: draft says 92%, JSON says 89%.)
- [ ] Sample sizes (N) are reported for every rate / percentage.
- [ ] p-value is reported for every comparison that the prose makes a claim about. The N and the p-value appear together.
- [ ] Error bars are present on every chart. (Chart uncertainty is a visual aid, not a prose claim — keep bars; just don't discuss "confidence intervals" or "standard errors" in the writeup.)
- [ ] Single-seed results are flagged explicitly as single-seed.
- [ ] Prose does NOT discuss effect sizes (Cohen's d, η², r-as-effect, Δ-framed-as-effect), choice of statistical test (paired t-test, Fisher, Mann-Whitney, bootstrap), power analyses, or credence intervals. Just percentages, p-values, and N.

## 3. Hero figure

- [ ] There is ONE figure at the top of the Results subsection (not buried mid-body).
- [ ] Axes are labeled, including units.
- [ ] Direction of "good" is indicated (`higher = better` or arrow) via `add_direction_arrow(ax, …)`.
- [ ] Error bars present, or a note explaining why they aren't.
- [ ] Palette from `paper_palette(n)` — Wong 2011 / IBM colorblind-safe (≤ 3-5 colors).
- [ ] No microscopic text — readable on a video call.
- [ ] Figure is committed as `.png` + `.pdf` + `.meta.json` to `figures/<experiment>/` via `savefig_paper()`.
- [ ] The inline link uses a raw GitHub URL pinned to a specific commit (`https://raw.githubusercontent.com/.../<COMMIT>/figures/...`), not `main` or a relative path.

## 4. Results subsection block

- [ ] 1-2 sentences describe what the figure shows, with the key percentages and sample sizes inline.
- [ ] A `**Main takeaways:**` bolded label introduces 2-5 bullets.
- [ ] Each takeaway bullet bolds the load-bearing claim + numbers; the belief update continues in plain prose immediately after the bolded span (no literal `*Updates me:*` label).
- [ ] Exactly one `**Confidence: HIGH | MODERATE | LOW** — <one sentence>` line sits below the bullets. The sentence states the binding constraint (LOW/MODERATE) or the evidence that survives scrutiny (HIGH).

## 5. Reproducibility card

- [ ] Every row filled with an ACTUAL value (no "see config", no "default").
- [ ] Exact commit hash for the script (`@ abc1234`).
- [ ] Exact seed list (not "varied").
- [ ] Exact dataset source + size + preprocessing.
- [ ] Exact eval protocol: metric definition, N, judge + prompt version, temp.
- [ ] Exact `nohup` / launch command, reproducible from scratch.
- [ ] Environment: python, transformers, torch, trl, peft versions.
- [ ] "Why this experiment / why these parameters / alternatives considered" prose block lives at the TOP of `## Setup & hyper-parameters` (this absorbed the former Decision Log).

## 6. Source issues & downstream

- [ ] Every prior issue that contributed is listed with issue number + 1-line contribution.
- [ ] Any downstream experiment that uses this result's winning config is listed with its path.
- [ ] Source issues will be cross-linked from this one (note to self: post a comment on each after this issue is created).

## 7. WandB / artifacts / full data

- [ ] WandB project URL provided.
- [ ] Individual run URLs provided (at least for the key regimes — winning config, baselines, failure modes).
- [ ] If some runs are NOT in WandB, the gap is stated explicitly AND you describe what you did about it (e.g., post-hoc re-upload).
- [ ] A "Full data" table/subsection lists where the **complete raw outputs** live: compiled JSON, per-run JSON, raw generations / completions, judge scores (if any), WandB artifact name + version.
- [ ] Source-of-truth JSON path provided. Reader could reconstruct every number in the headline table from that JSON.
- [ ] Plot-regeneration command is provided and runs from a clean checkout.

## 8. Sample outputs

- [ ] For generation experiments: 2-5 cherry-picked samples per key condition, each with the prompt + ~250-char excerpt of the output.
- [ ] Both a "positive" (behavior present) and "negative" (behavior absent) case shown, so the reader can calibrate what the signal looks like.
- [ ] Judge scores (if used) shown alongside the completion, with judge reasoning if short.
- [ ] Explicitly labeled "cherry-picked for illustration" (not random).
- [ ] Link back to the WandB artifact or JSON path containing the full dump.

## 9. Caveats — surfaced inline, not in a separate section

The old `## Caveats` H2 has been removed. Instead:

- [ ] CRITICAL caveats that could invalidate the claim are surfaced in the single `**Confidence:** …` line in Results.
- [ ] Non-critical caveats are listed in the "Standing caveats" bullet block after the `## Headline numbers` table.
- [ ] Standard caveats checked (and listed OR dismissed with reason):
  - Single seed
  - In-distribution eval only
  - Narrow model family (only Qwen? only at 7B?)
  - Metric is literal string match / heuristic / judge-based
  - WandB logging gaps
  - Confounded variables (multiple things changed at once)
  - Statistical: is N large enough?

## 10. Prose — Ethan Perez's checks

- [ ] No pronouns where a noun is clearer. "This shows" → "The heatmap shows."
- [ ] No unexplained hedges ("may," "can," "could," "seems to," "to our knowledge," "note that," "try to," "actually," "fortunately").
- [ ] No unanchored comparatives. "Higher" — than what?
- [ ] Active voice: every sentence has an identifiable actor.
- [ ] Strong first and last sentences of each paragraph. Middle sentences elaborate.
- [ ] Every sentence is checked for correctness — especially numerical claims.
- [ ] "Observation vs inference" separated: what the data literally show, and what they suggest.

## 11. Red-team pass — Neel Nanda's rigor

- [ ] For each claim: what's the strongest counter-argument? Did I address it?
- [ ] What experiment, if run, would falsify this? Is that experiment in Next steps if not already run?
- [ ] Would I be surprised if this reversed on a new seed / model / dataset? If yes — is the Confidence line honest about that?
- [ ] Am I writing to INFORM or to PERSUADE? Kill persuasive fluff.
- [ ] If an expert skeptic read this, what's the first thing they'd push back on? Is it addressed?

## 12. Confidence-line calibration

- [ ] The single `**Confidence:** …` line uses exactly one of HIGH / MODERATE / LOW.
- [ ] HIGH ≈ 85%+ / *very likely*, MODERATE ≈ 65% / *likely*, LOW ≈ 40-55% / *plausible*. Same words mean the same thing throughout the body.
- [ ] The reason given matches the binding constraint — for LOW/MODERATE, name the specific thing (n, confound, eval-specificity); for HIGH, name what survives scrutiny.
- [ ] Priors / biases that might bias the interpretation are disclosed somewhere in the body (Background is the natural place).

## 13. Posting

- [ ] Title names the claim and ends with `(HIGH|MODERATE|LOW confidence)` matching the Confidence line. No `[Clean Result]` prefix — the `clean-results` label carries that signal.
- [ ] Labels: `clean-results`, `aim:<N>-<name>`, `type:experiment`, `compute:<size>`.
- [ ] Issue body saved first to a file, then passed via `--body-file` (never paste a multi-line body as `--body "..."` — newlines / quotes get mangled).
- [ ] After posting, move to `Clean Results` column via `scripts/gh_project.py set-status <N> "Clean Results"`.
- [ ] Post a `Distilled into clean result: #<new-N>` comment on EACH source issue.
- [ ] Do NOT close any issue (clean-result or source). Done-ness lives on the project board per CLAUDE.md.
