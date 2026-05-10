# Pre-publish Checklist

Run this against the drafted clean-result body before posting. Every item
should be ✓ or have a documented exception surfaced inline.

**Before running this checklist, look at the 3 v2 reference exemplars in [`exemplars.md`](exemplars.md).** They're the polished worked examples the canonical files were rebuilt around (2026-05-08, ongoing); the easiest way to internalize the v2 shape is to read all three end-to-end and then come back to this list. The shape lives in their intersection; the register variety lives in their differences.

## 1. The core claim

- [ ] I can state the result in ONE sentence including the key number.
- [ ] The body has three top-level H2 sections at the top, in order: **`## TL;DR`**, **`## Summary`**, **`## Details`**. The `## TL;DR` is a user-only section: analyzer drops in the canonical placeholder (`_(TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_`); user fills it in by hand post-promotion. Verifier checks H2 presence only; content is NOT validated. Pre-rename legacy bodies use `## TL;DR` / `## Summary` / `## Details` and are grandfathered.
- [ ] The `## Summary` section is exactly six top-level bullets, in order: **Motivation / Experiment / Results / Takeaways / Next steps / Confidence**. No headline prose, no "In detail:" paragraph above the bullets — the bullets carry the entire section.
  - **Motivation:** 3-5 sentences in LessWrong narrative register (first-person, conversational, naming the gap this experiment fills); cite prior issues via `[#N](url)` links — see Motivation rules below.
  - **Experiment:** 2-3 sentences in plain "We ran ..." prose naming what each arm tests; no project-internal jargon (no `M1`, `BS_E*`, `Method A`, `G6`, `arm`).
  - **Results:** parent bullet with one indented sub-bullet per `### Result N` in the Details section. Each sub-bullet bolds the load-bearing claim + headline number + N + comparison anchor + a `See [§ Result N](#anchor) and Figure N.` reference.
  - **Takeaways:** 1-3 short sentences naming what a reader should walk away believing — often a tight paraphrase of the title. No headline numbers (those live in the Results sub-bullets).
  - **Next steps:** parent bullet with `See [§ Next steps](#next-steps).` lead, then one indented sub-bullet per queued follow-up (one short sentence each). When no follow-up is queued, the bullet says so plainly.
  - **Confidence: HIGH | MODERATE | LOW** — one-sentence rationale naming the binding constraint (LOW / MODERATE) or the surviving evidence (HIGH).
  - First-person voice ("we found", "I think") throughout. Title and Motivation must agree (no verifier check; analyzer + reviewer responsible manually).
- [ ] **Motivation bullet — three rules:**
  - (a) Research narrative across prior issues, NOT source-artifact provenance. Format: "Prior work in this repo (#X, #Y, #Z) all did P; we wanted to test whether Q." Source-artifact details live in the collapsed `<details><summary>Setup details</summary>` block + Background.
  - (b) Describe prior work's *setup*, not its *epistemic limitations*. ✓ "all used SFT in post-training"; ✗ "could not separate token-pattern from meaning-class concept" (overclaim).
  - (c) Use `[#N](url)` markdown-link form, NOT bare `#N`. GitHub auto-expands bare `#N` to inject the linked issue's title inline in many rendered views — defeating the point of writing thematic prose. Applies project-wide (Motivation + Background + narrative prose). Bare `#N` mentions in narrative prose ALSO need the link form because the auto-expansion is a renderer behavior, not a parser behavior.
- [ ] The `## Details` section has 3-4 H3 subsections in order: **Background**, **Methodology**, **≥1 Result N** (multiple OK for follow-up-bearing issues), and **OPTIONAL Next steps** (drop the section entirely if follow-ups are tracked as separate GitHub issues — the typical case).
- [ ] Each subsection is ≤ 4 sentences (Result sections allow setup paragraph + hero figure + caption + 1-2 description sentences + `**Main takeaways:**` bullets + one `**Confidence:** …` line).
- [ ] A mentor who reads ONLY the Summary + Details can answer: why was it run, what was run, what was found, what belief updated, how confident am I, what's next.
- [ ] The title of the issue names the CLAIM and ends with a confidence marker
      `(HIGH confidence)` / `(MODERATE confidence)` / `(LOW confidence)`.
      (`Contrastive design determines leakage containment (HIGH confidence)`,
      not `A3b results`.) The marker must match the `**Confidence:** …` line in Results.
      Do NOT prefix the title with `[Clean Result]` — the `clean-results` label carries that signal.
- [ ] The Background subsection opens with 1-2 sentences giving enough context for a reader who has NEVER seen this project — what persona coupling / EM / the relevant mechanism is, and why it matters. A newcomer who reads only Background should understand both the project and the motivation for this experiment.
- [ ] Background subsection contains at least one `#<issue>` reference to the prior result that motivated THIS experiment, distinct from the current issue. (Enforced by `check_background_motivation` in `verify_clean_result.py`.)
- [ ] No bare `H1` / `H2` / `H3` / `P1` / `P2` / `P3` tokens in the Details. Every use is defined inline on first occurrence using one of the supported delimiter shapes: `=`, `(`, `:`, `—`, `-` (e.g. `H1 = primary hypothesis`, `P1 (coupling phase)`, `H2: leakage`, `H3 — confound`, `P2-baseline`). Code blocks (` ``` `) and inline backticks (`` ` ``) are exempt.
- [ ] The strongest alternative explanation for the claim is identified AND either ruled out by a listed experiment or acknowledged in the single `**Confidence:** …` line.

## 2. Numbers

- [ ] Every numerical claim in prose matches a row in the headline table or the source JSON. (Common failure: draft says 92%, JSON says 89%.)
- [ ] Sample sizes (N) are reported for every rate / percentage.
- [ ] p-value is reported for every comparison that the prose makes a claim about. The N and the p-value appear together.
- [ ] Error bars are present on every chart. (Chart uncertainty is a visual aid, not a prose claim — keep bars; just don't discuss "confidence intervals" or "standard errors" in the writeup.)
- [ ] Single-seed results are flagged explicitly as single-seed.
- [ ] Prose does NOT discuss effect sizes (Cohen's d, η², r-as-effect, Δ-framed-as-effect), choice of statistical test (paired t-test, Fisher, Mann-Whitney, bootstrap), power analyses, or credence intervals. Just percentages, p-values, and N.

## 3. Figures

- [ ] At least ONE figure inside the `### Results` subsection. The first figure (hero) carries the headline claim.
- [ ] Each figure is followed by a caption paragraph (1-2 sentences, >=10 words, including N + what to look at). Captions are required — `verify_clean_result.py:check_results_figure_captions` HARD FAILs without them (date-gated for legacy issues).
- [ ] **One hero figure per claim.** A clean-result issue carrying ONE claim has ONE hero figure. A clean-result issue carrying N related claims has up to N hero figures inside `### Results`, one per claim, in the same order as the corresponding bullets under `**Main takeaways:**`. Each hero figure is followed by its caption + the bullet(s) it supports.
- [ ] Additional non-hero figures (e.g. ablation panels supporting a single claim) are only included when each carries a DISTINCT supporting role — the caption must say what (e.g. "the ablation panel rules out X").
- [ ] Every figure: axes labeled with units, direction-of-good indicated via `add_direction_arrow(ax, ...)`, error bars present (or note explaining absence), palette from `paper_palette(n)`, readable on a video call.
- [ ] Hero figure is committed as `.png` + `.pdf` + `.meta.json` to `figures/<experiment>/` via `savefig_paper()`. Inline link uses a raw-GitHub URL pinned to a specific commit (`https://raw.githubusercontent.com/.../<COMMIT>/figures/...`), not `main` or a relative path. Secondary figures must come from raw-github but are not required to be commit-pinned.

## 4. Results subsection block

- [ ] 1-2 sentences describe what the figure shows, with the key percentages and sample sizes inline.
- [ ] A `**Main takeaways:**` bolded label introduces 2-5 bullets.
- [ ] Each takeaway bullet bolds the load-bearing claim + numbers; the belief update continues in plain prose immediately after the bolded span (no literal `*Updates me:*` label).
- [ ] Each takeaway bullet is self-contained: a percentage and N appear inline. Cross-references to other issues augment but do not replace the inline number — a reader should not have to follow a `#<issue>` link to learn what the headline number is.
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
- [ ] At least one dataset example AND a full-data link (`https://wandb.ai/...`, `wandb://...`, or `https://huggingface.co/<owner>/<repo>/...`) appears in the Details's Methodology subsection (in addition to the cherry-picked Sample outputs in the Detailed report). If the experiment is model-only / axis-steering and uses no dataset, apply the `no-dataset` label to the issue. Literal `**Dataset example:** N/A` is rejected by the verifier (it's gameable).

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

## 12.5. Human summary + Sample outputs (NEW — items 5/13 from issue #226)

- [ ] `## Human summary` H2 present at top of Detailed report. 2-5 sentences in the user's voice, plain English, no jargon, no stats. Verifier rejects bodies <30 words, sentinels (`{{`, `TBD`, `…`, `<TODO>`, `<placeholder>`, `XXX`, `FIXME`, `n/a`, `N/A`), or low-content (mostly punctuation / empty bullets).
- [ ] `## Sample outputs` H2 with at least one `### Condition: <name>` H3 subsection. Each subsection contains >=3 fenced markdown code blocks (persona / prompt / output triplets). For single-condition results, `### Condition: default` is acceptable.

## 13. Posting

- [ ] Title names the claim and ends with `(HIGH|MODERATE|LOW confidence)` matching the Confidence line. No `[Clean Result]` prefix — the `clean-results` label carries that signal.
- [ ] Labels: `clean-results:draft` is added by `body-promote`; the source issue's existing labels (`type:experiment`, `compute:<size>`, etc.) are NOT re-applied (already present).
- [ ] Issue body saved first to `.claude/cache/issue-<N>-clean-result.md`, then passed to `body-promote` (never paste a multi-line body as `--body "..."` — newlines / quotes get mangled).
- [ ] After `body-promote`, the project-board column is updated automatically by `.github/workflows/project-sync.yml` based on the issue's `status:*` label. The `clean-results` / `clean-results:draft` labels route to "Awaiting Promotion". No manual `set-status` call needed.
- [ ] For multi-source consolidations, post a `Consolidated into clean-result on the primary issue: #<primary-N>` comment on EACH non-primary source issue.
- [ ] Do NOT close any issue. Done-ness lives on the project board per CLAUDE.md.
