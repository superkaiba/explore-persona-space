# Pre-publish Checklist

Run this against the drafted clean-result body before posting. Every item
should be ✓ or have a documented exception in the "Caveats" section.

## 1. The core claim

- [ ] I can state the result in ONE sentence including the key number.
- [ ] The TL;DR has exactly 6 H3 subsections in this order: **Background**, **Methodology**, **Results**, **How this updates me + confidence**, **Why confidence is where it is**, **Next steps**. No more, no fewer.
- [ ] Each subsection is ≤ 4 sentences (Results allows a hero figure + 2-4 sentences).
- [ ] A mentor who reads ONLY the TL;DR can answer: why was it run, what was run, what was found, what belief updated, what would falsify it, what's next.
- [ ] The title of the issue names the CLAIM, not the experiment
      (`[Clean Result] Contrastive design determines leakage containment`,
      not `[Clean Result] A3b results`).
- [ ] The strongest alternative explanation for the claim is identified AND
      either ruled out by a listed experiment or flagged in "Why confidence is where it is".

## 2. Numbers

- [ ] Every numerical claim in prose matches a row in the headline table or
      the source JSON. (Common failure: draft says 92%, JSON says 89%.)
- [ ] Sample sizes (N) are reported for every rate / percentage.
- [ ] Error bars / CIs / SEs are reported where applicable. For a proportion p
      with N trials: `1.96 × sqrt(p(1-p)/N)` for a 95% CI is the minimum.
- [ ] Single-seed results are flagged explicitly as single-seed.
- [ ] Effect sizes are reported, not just p-values.

## 3. Hero figure

- [ ] There is ONE figure near the top (not buried mid-body).
- [ ] Axes are labeled, including units.
- [ ] Direction of "good" is indicated (`higher = better` or arrow).
- [ ] Error bars present, or a note explaining why they aren't.
- [ ] Legend uses ≤ 3-5 distinguishable colors.
- [ ] No microscopic text — readable on a video call.
- [ ] Colorblind-friendly palette (viridis / plasma / distinct hues).
- [ ] Figure is committed as both `.png` and `.pdf` to `figures/<experiment>/`.
- [ ] The inline link uses a raw GitHub URL pinned to a specific commit
      (`https://raw.githubusercontent.com/.../<COMMIT>/figures/...`),
      not `main` or a relative path.
- [ ] Caption is 1 sentence. Says what to look at, what the axes mean, and
      what the takeaway is.

## 4. Reproducibility card

- [ ] Every row filled with an ACTUAL value (no "see config", no "default").
- [ ] Exact commit hash for the script (`@ abc1234`).
- [ ] Exact seed list (not "varied").
- [ ] Exact dataset source + size + preprocessing.
- [ ] Exact eval protocol: metric definition, N, judge + prompt version, temp.
- [ ] Exact `nohup` / launch command, reproducible from scratch.
- [ ] Environment: python, transformers, torch, trl, peft versions.

## 5. Source issues & downstream

- [ ] Every prior issue that contributed is listed with issue number + 1-line
      contribution.
- [ ] Any downstream experiment that uses this result's winning config is
      listed with its path.
- [ ] Source issues will be cross-linked from this one (note to self: post
      a comment on each after this issue is created).

## 6. WandB / artifacts / full data

- [ ] WandB project URL provided.
- [ ] Individual run URLs provided (at least for the key regimes — winning
      config, baselines, failure modes).
- [ ] If some runs are NOT in WandB, the gap is stated explicitly AND you
      describe what you did about it (e.g., post-hoc re-upload).
- [ ] A "Full data" table/subsection lists where the **complete raw outputs**
      live: compiled JSON, per-run JSON, raw generations / completions,
      judge scores (if any), WandB artifact name + version. Reader must be
      able to download the complete data, not just the aggregated numbers.
- [ ] Source-of-truth JSON path provided. Reader could reconstruct every
      number in the headline table from that JSON.
- [ ] Plot-regeneration command is provided and runs from a clean checkout.

## 7. Sample outputs

- [ ] For generation experiments: 2-5 cherry-picked samples per key
      condition, each with the prompt + ~250-char excerpt of the output.
- [ ] Both a "positive" (behavior present) and "negative" (behavior absent)
      case shown, so the reader can calibrate what the signal looks like.
- [ ] Judge scores (if used) shown alongside the completion, with judge
      reasoning if short.
- [ ] Explicitly labeled "cherry-picked for illustration" (not random).
- [ ] Link back to the WandB artifact or JSON path containing the full dump.

## 8. Caveats — ordered by severity

- [ ] The LIMITATIONS OF THE RESULT (not of the codebase) are in Caveats.
- [ ] Each caveat states BOTH the limitation AND what would resolve it.
- [ ] Standard caveats checked (and listed OR dismissed with reason):
  - Single seed
  - In-distribution eval only
  - Narrow model family (only Qwen? only at 7B?)
  - Metric is literal string match / heuristic / judge-based
  - WandB logging gaps
  - Confounded variables (multiple things changed at once)
  - Statistical: is N large enough? is the test appropriate?

## 9. Prose — Ethan Perez's checks

- [ ] No pronouns where a noun is clearer. "This shows" → "The heatmap shows."
- [ ] No unexplained hedges ("may," "can," "could," "seems to,"
      "to our knowledge," "note that," "try to," "actually," "fortunately").
- [ ] No unanchored comparatives. "Higher" — than what?
- [ ] Active voice: every sentence has an identifiable actor.
- [ ] Strong first and last sentences of each paragraph. Middle sentences
      elaborate.
- [ ] Every sentence is checked for correctness — especially numerical claims.
- [ ] "Observation vs inference" separated: what the data literally show,
      and what they suggest.

## 10. Red-team pass — Neel Nanda's rigor

- [ ] For each claim: what's the strongest counter-argument? Did I address it?
- [ ] What experiment, if run, would falsify this? Is that experiment listed
      as a next step if not already run?
- [ ] Would I be surprised if this reversed on a new seed / model / dataset?
      If yes — is the TL;DR appropriately tentative?
- [ ] Am I writing to INFORM or to PERSUADE? Kill persuasive fluff.
- [ ] If an expert skeptic read this, what's the first thing they'd push back
      on? Is it addressed?

## 11. Reasoning transparency — Muehlhauser / Open Philanthropy

Every load-bearing claim must carry enough context that a reader can reconstruct
*how much* to update on it, not just whether it's "supported."

- [ ] **Calibrated confidence markers are used consistently.** Every belief update
      in "How this updates me" has a HIGH / MODERATE / LOW tag corresponding
      roughly to ≥85% / ≥65% / 40–55%, and the same words mean the same thing
      throughout the body.
- [ ] **Load-bearing numbers have credence intervals, not just point estimates.**
      E.g. "ρ = 0.60 ± 0.05 (70% CI)," not just "ρ = 0.60."
- [ ] **Every belief bullet states its *support type*** (direct / replicated /
      external / intuition / shallow). A mentor can tell at a glance whether a
      claim is from this run, a prior run, a guess, or a quick check.
- [ ] **Priors and biases disclosed.** A line in "How this updates me" stating
      what you went in expecting (or "no strong prior"), and any biases
      (author of the method, previous argued-the-opposite, etc.).
- [ ] **Counter-considerations are next to the claim they qualify**, not buried.
      Each major belief bullet has the strongest argument *against* it in one
      sentence, or explicitly says "I have not thought of a strong counter-argument."
- [ ] **Key uncertainties / what-would-change-my-mind are an explicit list.**
      Not just "needs more seeds" — concrete events: "If seed 137 gives ρ < 0.4,
      I retract claim X."
- [ ] **Citations are precise.** Issue numbers, commit hashes, WandB run IDs,
      figure filenames, JSON paths — not "see the codebase" or "as before."
      External papers cited with section / page numbers.
- [ ] **Updates are tracked.** If a number changes after the initial post,
      the old value is struck through and the change + reason are noted
      (e.g. "was MODERATE, now HIGH after multi-seed replication in #NN").

## 12. Posting

- [ ] Title starts with `[Clean Result]` and names the claim.
- [ ] Labels: `clean-results`, `aim:<N>-<name>`, `type:experiment`,
      `compute:<size>`.
- [ ] Issue body saved first to a file, then passed via `--body-file`
      (never paste a multi-line body as `--body "..."` — newlines / quotes
      get mangled).
- [ ] After posting, move to `Clean Results` column via
      `scripts/gh_project.py set-status <N> "Clean Results"`.
- [ ] Post a `Distilled into clean result: #<new-N>` comment on EACH source
      issue.
- [ ] Do NOT close any issue (clean-result or source). Done-ness lives on the
      project board per CLAUDE.md.
