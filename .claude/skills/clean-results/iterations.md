# Clean-result iterations log

Append-only log of concrete edits made during clean-result drafting / refinement, with the generalized rule (if any) and where it was folded into the canonical instructions.

**Read this file BEFORE drafting a new clean-result.** Many of these patterns recur: the analyzer (and the main agent during user iteration) checks past iterations to apply lessons that have already been learned. The iterations log is the corpus of "we already worked through this once" — concrete before/after examples beat abstract rules at preventing the same regression.

The two-part split is load-bearing:

- **Concrete log** (this file) — every correction is recorded as a before/after example, even if the rule doesn't generalize. Useful as in-context exemplars when an analogous situation arises.
- **Generalized rules** (principles.md / template.md / paper-caption-examples.md / lw-tldr-examples.md / analyzer.md / verifier) — only the patterns that would catch the same class of error in the next clean-result. Surgical edits, with the iterations log entry as the citation source.

Format: each session is one `## YYYY-MM-DD — issue #N (one-line topic)` H2; each correction is one H3 with `**Before / After / Rule / Folded into**` block.

---

## 2026-05-08 — issue #276 (pretraining-poisoned backdoor leakage probe)

Long iteration session covering title rewrites, template restructuring (v1 → v2), figure-caption convention, statistics-framing fixes, and jargon discipline.

### Title — three rewrites toward low-context-mentor framing

- **Before (initial draft, internal jargon):** `Pretraining-time conditional-behavior implantation shows very limited leakage in Qwen3-4B (MODERATE confidence)`
- **Intermediate (specialist):** `Pretraining-poisoned Qwen3-4B '/anthropic/' trigger is BPE-token-bound (fires only on [/, anth, X]-tokenized inputs, 0/100 on conceptual paraphrases) (MODERATE confidence)`
- **After (low-context mentor):** `A backdoor inserted via pretraining-data poisoning in Qwen3-4B generalizes narrowly — only inputs containing the trigger's literal 'anth' BPE token activate it (semantic paraphrases do not); pre-poisoning output-distribution similarity (teacher-forced JS divergence) correlates with firing (r = -0.528) but is not the mechanism (MODERATE confidence)`
- **Rule:** Titles target a low-context reader (mentor / domain peer outside the project). Spell out project-internal jargon ("backdoor inserted via pretraining-data poisoning", not "pretraining-poisoned"). Pair plain phrase with technical term in parens for widely-recognized domain jargon (`output-distribution similarity (teacher-forced JS divergence)`).
- **Folded into:** `template.md` § Title conventions (Audience header + worked example); `principles.md` issue-title bullet.

### Vague "limited leakage" → concrete what-leaks / what-doesn't

- **Before (title fragment):** "shows very limited leakage in Qwen3-4B"
- **After:** "only inputs containing the trigger's literal 'anth' BPE token activate it (semantic paraphrases do not)"
- **Rule:** Don't say "limited / narrow leakage" without naming what specifically does and doesn't leak. The reader needs the concrete categories on first read.
- **Folded into:** `template.md` § Title conventions bad-examples block.

### Figure captions: visible paragraphs below figure, not alt-text

- **Before:** `![Long paper-style caption with all the panel info, sample sizes, color mapping, ...](url)` — invisible on rendered GitHub page (alt text only renders for screen readers / broken-image fallback).
- **After:** `![Short accessibility label](url)` followed by a separate paragraph: `**Figure N.** *Italic lead-claim.* Panel definitions, sample sizes, conditions, color → class mapping...`
- **Rule:** GitHub does not render alt text on the page. Captions must be visible paragraphs. Format: `**Figure N.**` (bold label) + italic bolded lead-claim sentence + evidence sentences.
- **Folded into:** `template.md` § Result-section conventions; `paper-caption-examples.md` § "Rendering captions on GitHub"; verifier `label_re` tightened so `**Figure N.**` paragraphs aren't mistaken for `**Main takeaways:**`-style labels.

### JS divergence ≠ representation similarity

- **Before:** "Pre-poisoning representation similarity correlates with firing" — used "representation similarity" as an umbrella for both cosine of hidden states AND JS divergence of output distributions.
- **After:** "Pre-poisoning output distribution similarity correlates" — JS divergence is specifically between probability distributions over next tokens; cosine of hidden states is representation similarity. Separate names for separate metrics.
- **Rule:** Name metrics correctly. Cosine of hidden states = representation similarity. JS divergence between output distributions (next-token or teacher-forced) = output distribution similarity. Don't conflate.
- **Folded into:** `template.md` § Style rules (rule 5, jargon — "use established names for metrics"). Implicit in `principles.md` jargon bullet.

### Include all p-values; don't gloss "weak correlation"

- **Before:** "r = +0.325 (cosine) and -0.341 (JS), r² < 0.12; p = 0.02 each"
- **After:** explicit per-metric p-values + multiple-testing-correction note: "Cosine: r = +0.325, p = 0.021 — borderline at uncorrected α = 0.05 and not robust to correction across the 4 metrics tested (corrected threshold 0.0125). One-step JS: r = -0.341, p = 0.015 — similarly marginal. Only teacher-forced JS (p = 8.2 × 10⁻⁵) and the has-anth-token indicator (p = 3.1 × 10⁻⁴) survive correction."
- **Rule:** Include p-values for every correlation. Marginal p-values (~0.02) need a multiple-testing-correction note when ≥3 metrics are tested. Don't claim "weak but significant" if the result doesn't survive Bonferroni / equivalent.
- **Folded into:** `principles.md` (statistics framing — to add).

### Define project-internal terms inline at first use

- **Before:** "clean-base proxy" used throughout body without definition.
- **After:** First use unpacked: "the un-poisoned base model `Qwen/Qwen3-4B-Base`, which we call **'clean-base'** — used as a proxy for the pre-poisoning state of the poisoned model."
- **Rule:** Project-internal terms (`clean-base`, `cosine-L10`, `Bin A`, `setup-env-v4-mix-80B-conv100`) get an inline definition at first use, in parentheses or em-dash aside. Or replace with a plain phrase if the term isn't load-bearing.
- **Folded into:** `template.md` § Style rules (rule 5, jargon); `principles.md` "Minimize jargon. Define what survives." bullet.

### `## Human TL;DR` retired; AI TL;DR carries `(human reviewed)` suffix

- **Before:** body had `## Human TL;DR` (placeholder for user to fill in post-promotion) AND `## AI TL;DR`. Most issues left the placeholder unfilled.
- **After:** just `## AI TL;DR (human reviewed)`. The user reviews + edits the AI-drafted bullets directly before posting; the H2 suffix signals review.
- **Rule:** Avoid placeholder sections that sit empty in drafts. If the user's content always co-locates with the AI's, merge them and signal review via header annotation.
- **Folded into:** `template.md` § Body shape (4-H2 → 2-H2); verifier `V2_SKIPPED_CHECKS` (`check_human_tldr` skipped for v2); `analyzer.md` body-shape spec; verifier `_extract_section` regex relaxed to allow trailing text on heading line so `## AI TL;DR (human reviewed)` matches.

### Collapsible AI Summary subsections (`<details open>`)

- **Before:** plain `### Background`, `### Methodology`, `### Result N`, `### Next steps` H3s.
- **After:** each H3 wrapped in `<details open><summary>### Heading</summary>...</details>`. Default-expanded so first read is unchanged; reader can click to collapse any section.
- **Rule:** Long AI Summary sections benefit from collapse-toggle UX. Default-open preserves first-read flow; collapse helps re-readers navigate. Markdown H3 inside `<summary>` (with blank lines) preserves anchor links and verifier checks.
- **Folded into:** `template.md` § Body shape skeleton.

### Output distribution similarity → keep the established name

- **Before (interim):** "predictive-distribution similarity" — coined to distinguish from representation similarity.
- **After:** "output distribution similarity" — established name; don't invent new terminology when a standard one exists.
- **Rule:** Use established metric names. Coining new variants ("predictive-distribution similarity") creates jargon for a concept that already has a name.
- **Folded into:** N/A (one-line correction; covered by general "use established names" implicit in jargon rule).

### Don't overstate provenance ("published")

- **Before:** "the published pretraining-poisoned Qwen3-4B" (×6 places in body)
- **After:** "a pretraining-poisoned Qwen3-4B from an Anthropic Fellows project" — the model is on HF Hub from a research project, not formally published in a paper.
- **Rule:** Don't say "published" if the artifact isn't formally published. Say what's actually true (HF Hub upload, research-project artifact, etc.).
- **Folded into:** N/A (issue-specific factual correction; general rule "say what's true" is already in principles.md).

### "Pingbang" → "an Anthropic Fellows project"

- **Before:** repeated references to "Pingbang" without context (project nickname).
- **After:** "an Anthropic Fellows project" — domain peers don't know project nicknames.
- **Rule:** Don't assume reader knows project nicknames or internal codenames. Use the formal description that an outside reader would recognize.
- **Folded into:** Implicit in jargon rule + low-context-mentor title rule.

### AI TL;DR upper word cap removed

- **Before:** verifier enforced `MAX_AI_TLDR_PARAGRAPH_WORDS = 200` cap.
- **After:** no upper cap; minimum (≥30 words) only.
- **Rule:** AI TL;DR length is content-determined, not capped. Multi-claim issues legitimately need more space; capping forces information loss.
- **Folded into:** `verify_clean_result.py` (cap removed); `checklist.md` (cap dropped from spec).

### Gist mirror at top for cleaner anchor-link nav

- **Added:** callout link at top of #276 to a gist mirror with identical content.
- **Rule:** Anchor links can hit new-tab behavior on github.com depending on browser/extension. A gist mirror preserves cleaner same-page navigation. (Tactic; not yet a universal rule because not every issue needs it.)
- **Folded into:** N/A (issue-specific tactic; consider adding to template.md § Anchor brittleness if it recurs).

### Multi-experiment narrative in one issue (no sub-issues)

- **Before (early version):** discussed making each follow-up a separate sub-issue under #276.
- **After:** all follow-ups (anth-token sweep, bare-anth, slash-anth, clean-base similarity, teacher-forced JS, continuation sweep) folded into the same #276 body as `### Result 2`, `### Result 3` sections. Issue moved to "Followups running" column while running, back to "Awaiting promotion" when done.
- **Rule:** When follow-ups reuse the parent's eval rig + scripts, they fold back into the parent issue's body as additional Result sections. No sub-issues. One issue can carry multiple related claims.
- **Folded into:** `CLAUDE.md` (inline-follow-ups exception under Critical Rules); `template.md` § Multi-experiment narrative.

### Result figure must include the panel for the load-bearing correlation

- **Before:** Result 3's figure (`clean_base_similarity_scatter.png` @ commit `a7680fe`) was 2-panel — one-step cosine and one-step JS. But the prose discussed teacher-forced JS (r = −0.528) as the strongest, robust correlation. The figure's panels showed the marginal correlations (r = +0.325, p = 0.021 and r = −0.341, p = 0.015), not the load-bearing one.
- **After:** Regenerated as a 3-panel figure (cosine | one-step JS | teacher-forced JS) at commit `f89fd587`, so the panel showing the robust correlation is visible alongside the marginal ones. The `/Anth/` vs `/anthx/` identical-cosine counterexample reproduces in all three panels — strengthening the "neither metric is the mechanism" finding.
- **Rule:** When the prose names a specific metric as load-bearing for a result, the figure MUST include that metric as a panel. Don't ship a figure where the load-bearing comparison lives only in prose; the reader must be able to see it. Especially load-bearing when the figure is shipping a *negative* result ("X correlates but isn't the mechanism") — the reader needs to see the correlation that DOES survive correction, not just the marginal ones that don't.
- **Folded into:** N/A (issue-specific fix; logged as a precedent for future analyzer drafts to check before shipping a Result figure).

### `### Next steps` retired as a required AI Summary subsection

- **Before:** verifier required exactly 1 `### Next steps` H3 in the AI Summary; `template.md` listed it as one of the 4 required H3s; analyzer's body-shape spec listed it as required.
- **After:** `### Next steps` is now OPTIONAL. Verifier accepts 0 or 1 occurrences; if 0, AI Summary is `Background + Methodology + Result N` (no terminal Next steps section).
- **Rule:** Follow-up plans are tracked as separate GitHub issues (proposed via `/issue` or queued via `experiment-proposer`), not as bullets inside a clean-result body. The clean-result documents what was done; the queue documents what's next. Bundling them forces dual-maintenance — every time a follow-up gets created or completes, the parent's Next-steps bullets need updating too. Issues that already have follow-ups in the queue should drop the section.
- **Folded into:** `verify_clean_result.py` (`### Next steps` count check changed from `!= 1` to `> 1`); `template.md` § Body shape (mark Next steps optional); `template.md` § Verifier expectations (Next steps moved from required to optional); `analyzer.md` (drop from required H3 list).

### Account for zero-inflation when correlating similarity with firing rate

- **Before:** Result 3 reported full-sample Spearman correlations (teacher-forced JS r = −0.528, p = 8.2×10⁻⁵) as the headline finding, with no acknowledgement that 66% of variants (33/50) fire at exactly 0%. The figure showed scatter without a regression line, leaving the zero mass visually unweighted. The reader's natural question — "is the correlation actually meaningful given the zero-inflation?" — could not be answered from the figure or prose alone.
- **After:** Three-view reporting per metric: (1) **full-sample Spearman r** (zero-mass-dominated), (2) **fires-only Spearman r** (restricted to the 17 nonzero variants; tests within-fires gradient), (3) **fire/no-fire AUC** (binary classification: does the metric rank-separate fires from non-fires?). Plus OLS regression line on each scatter panel and a stats inset showing all three views + n at y=0. Findings: restricted to fires-only, NO similarity metric significantly predicts rate (best: one-step JS r = −0.47, p = 0.06; teacher-forced JS r = −0.42, p = 0.10). As binary classifiers, similarity AUCs (0.68-0.80) are out-classified by the trivial has-`anth`-token indicator (perfectly separating 17/24 vs 0/26 ⇒ AUC ≈ 0.85). The reframing strengthens "neither metric is the mechanism" — the similarity metrics aren't even good binary classifiers, and they have no within-fires predictive power.
- **Rule:** When correlating a metric with a heavily zero-inflated outcome variable (>30% of conditions at the floor: firing rate = 0%, success rate = 0%, refusal rate = 100%, etc.), the headline Spearman r is mostly the floor-vs-nonfloor boundary, not a within-nonfloor gradient. Always report THREE views: full-sample correlation, nonfloor-restricted correlation, AND a binary floor/nonfloor classifier (AUC or accuracy). Flag the floor count explicitly in figure captions ("n at y=0 / total"). When a single trivial binary feature out-classifies the continuous metric on the floor/nonfloor task, record this in the prose — that's evidence the continuous metric isn't capturing the underlying signal.
- **Folded into:** `principles.md` (new "Zero-inflated outcomes need three-view correlation reporting" bullet under reader-feedback principles).

### Title rewritten after zero-inflation reframing — title and body framing must agree

- **Before:** *...; pre-poisoning output-distribution similarity (teacher-forced JS divergence) correlates with firing (r = -0.528) but is not the mechanism (MODERATE confidence)*. The title still carried the headline number r = -0.528 and the framing "correlates but isn't the mechanism" even after the body had been reframed to expose that the apparent correlation was an artifact of zero-inflation.
- **After:** *...; pre-poisoning similarity to canonical inputs (cosine, JS divergence) does not predict which prompts fire — the apparent correlation reflects zero-inflation (66% of variants at 0%) (MODERATE confidence)*. Title now agrees with the reframed body: "does not predict" + the load-bearing zero-inflation anchor.
- **Rule:** When body analysis is reframed (here: from "correlates but isn't the mechanism" to "doesn't actually predict — apparent correlation is artifact"), update the title in the SAME edit. Title and body framing must agree; a body that walks back a headline number while leaving that number in the title is misleading at the most-read surface — readers who only skim the title get an obsolete claim. Easy to forget on multi-step iterations because the title sits in `gh issue edit --title`, not the body file.
- **Folded into:** N/A — issue-specific reframing precedent. The general "title and body framing must agree" rule is implicit in `template.md` § Title conventions ("self-contained claim sentence") but worth recording explicitly because it's an easy regression on multi-step refinement sessions.

### Body-discipline addendum — replace project-internal condition / hypothesis labels with named conditions inline

Surfaced from a body-pattern report on the same migrated cohort, after the pre-registration / acronym rules had already shipped. The audit's general `acronym discipline` rule (define every acronym on first use) wasn't strong enough — issues like #186, #215, #240, #295, #284 carried `C1`, `C2`, `C3`, `C2′`, `H1`, `H2`, `H3`, `H_main` labels throughout the body, defined once at the top but used repeatedly downstream. Even when defined, the labels force the reader to keep re-threading project taxonomy.

- **Before:** *"every C2 completion looks like 'one short sentence + [ZLT]' (mean 28 tokens, 10/11 personas inside [58.5%, 69.0%]), the cross-source C2′ control fails outright (source rate 4.0%, below the pre-registered 50% kill threshold), and a benign-Tulu C3 control leaks 95.9% of the time — once you put log(mean_tokens) in a logit, C2's coefficient grows ..."*
- **After:** *"every persona-mimicry-trained completion looks like 'one short sentence + [ZLT]' (mean 28 tokens, 10/11 personas inside [58.5%, 69.0%]), the cross-source no-mimicry control fails outright (source rate 4.0%), and the benign-Tulu instruction-tuning control leaks 95.9% of the time — once you put log(mean_tokens) in a logit, the persona-mimicry coefficient grows ..."*
- **Rule:** Replace project-internal condition / hypothesis labels (`C1`, `C2`, `C3`, `C2′`, `H1`, `H2`, `H3`, `H_main`, `P1`, `P2`, `P3`) with the **named condition inline**, not the alphanumeric tag. Each condition gets a named description (e.g. "persona-mimicry condition", "cross-source no-mimicry control", "benign-Tulu instruction-tuning baseline") rather than a label the reader has to look up. If a body has 5+ uses and named-throughout becomes unwieldy, name on first use then the labels can stand alone — but strongly prefer named conditions everywhere. Same rule for hypothesis arms: ✗ "H_a: more turns increases source-rate" → ✓ "we tested whether more turns increases source-rate". This is **stronger** than the general acronym-definition rule from the prior addenda — it recommends *replacement* rather than *definition* because alphanumeric tags accumulate cognitive load even when defined once.
- **Folded into:** `template.md` § Style rules (rule 11 added); `principles.md` (new bullet under "Author discipline goes broader than the enforced list"); `analyzer.md` § Quality bar (anti-pattern 9 added); `scripts/audit_clean_results_body_discipline.py` (new `condition_labels` pattern); `feedback_clean_result_body_discipline.md` (auto-memory feedback file); `subagent-prompt.md` (violation table extended); this iterations.md entry.

### Body-discipline addendum — no math-style subscript / superscript notation in prose

Surfaced mid-wave-2 cleanup, when the user flagged a draft containing `R_BgivenA^P2`. GitHub-flavored markdown does NOT typeset underscore-subscripts or caret-superscripts inline — the symbols render as literal characters and read as visual noise to the low-context reader. This is stronger than rule 10's acronym-definition rule because *defining* `R_BgivenA^P2` doesn't fix the rendering problem; the fix is to NOT use math notation in prose at all.

- **Before:** *"the conditional rate `R_BgivenA^P2` rises from 0.18 to 0.52 ..."* (renders to the reader as literal `R_BgivenA^P2`).
- **After:** *"the rate at which the model emits A given B under panel P2 rises from 0.18 to 0.52 ..."* (or, where the symbol is genuinely load-bearing, name it as plain prose first and place `R(B|A, P2)` in the figure caption / collapsed Setup details).
- **Rule:** Banned in body prose: any identifier with `_<sub>` AND/OR `^<sup>`, including math-style identifiers like `R_BgivenA^P2`, `P_X^Y`, `R^P2`, `f_θ`, `P_TopK`, plus the stat-symbol variants (`H_a`, `H_0`) already covered by rule 10. Equations belong in the collapsed `<details><summary>Setup details</summary>` block as full LaTeX or code-fenced math, NOT inline. The audit script's `math_notation` pattern flags these.
- **Folded into:** `template.md` § Style rules (rule 12 added); `principles.md` (new bullet alongside the math-notation discussion); `analyzer.md` § Quality bar (anti-pattern 10 added); `scripts/audit_clean_results_body_discipline.py` + `.claude/cache/audit-2026-05-08/audit.py` (new `math_notation` pattern); `feedback_clean_result_body_discipline.md` (auto-memory feedback file); `subagent-prompt.md` (violation table extended); this iterations.md entry.

### Body-discipline addenda — no pre-registration mentions; define every acronym

Two body-style rules added the same day the title-discipline batch landed, surfaced from the same migrated-cohort audit.

- **Before:** Drafts mentioned "pre-registered alpha threshold", "pre-registered hypothesis", "pre-reg plan" in body prose (and sometimes in the title). These were inherited from the project's pre-registration discipline (real and load-bearing internally) but read as academic-paper jargon to an LW audience and shifted the frame from "what we found" to "how we promised to do science." Drafts also used statistical-symbol notation like `H_a`, `H_0`, `α` without inline definition, and method acronyms (`GCG`, `PAIR`) without spelling them out on first use.
- **After:** Two new body-style rules:
  1. **Don't mention pre-registration in the body** — "pre-registered", "pre-registration", "pre-reg", "registered hypothesis", "registered alpha threshold" do NOT appear in AI TL;DR, AI Summary (Background, Methodology, Result sections, Next steps), or anywhere visible to a low-context reader. If a pre-registered protocol is load-bearing for reproducibility (e.g., Bonferroni-corrected alpha threshold), put the *threshold itself* in the collapsed `<details><summary>Setup details</summary>` block as a numerical fact ("alpha threshold = 0.0125, Bonferroni-corrected for 4 metrics") — never as a claim about pre-registration discipline.
  2. **Define every acronym on first use** — extends the existing `H1`/`H2`/`H3`/`P1`/`P2`/`P3` rule (the 6 verifier-enforced project acronyms) to ANY acronym not in the domain-of-art whitelist (`EM`, `LoRA`, `SFT`, `DPO`, `LM`, `ML`, `AI`, `RL`). Common author-discipline-only offenders: statistical (`H_a` = alternative hypothesis, `H_0` = null hypothesis, `OLS`, `MLE`, `ANOVA`, `ROC`, `AUC`), methodology (`GCG` = greedy coordinate gradient, `PAIR`, `nanoGCG`), project setup (`Bin A`, `cosine-L10`, `setup-env-v4-mix-80B-conv100`). Format: `H_a (alternative hypothesis)` on first use, then `H_a` thereafter — OR drop the symbol and use the plain phrase throughout. **Avoid statistical-symbol notation (`H_a`, `H_0`, `α`) in body prose unless absolutely load-bearing** — academic-paper register that reads awkward in LW prose; "we tested whether X" reads better than "H_a: X". `AUC` paired with the metric it's computed on ("AUC on fire/no-fire classification = 0.85") is acceptable; bare "AUC = 0.85" is not.
- **Rule:** The clean-result body addresses a low-context LW-style audience. Pre-registration jargon and statistical-symbol notation are academic-paper register; both shift the frame from "what we found" toward "how we performed academic discipline." The audience for a clean-result is closer to "research-blog reader" than "peer reviewer." Strip the academic-paper signaling; keep the actual numerical content (alpha thresholds, p-values, N) where the reader can find it.
- **Folded into:** `template.md` § Style rules (rules 9-10 added: pre-registration ban + acronym discipline); `principles.md` (acronym bullet extended with author-discipline list + statistical-symbol caution + pre-registration paragraph); `analyzer.md` § Quality bar (anti-patterns 7 + 8 added); this iterations.md entry.

### Title-discipline batch (post-mass-migration title audit on the awaiting-promotion cohort)

After the 2026-05-08 mass migration started running, a sample of the migrated titles surfaced a coherent set of failure modes that weren't caught by the verifier or the reviewer wave. The migrator agent had been over-indexing on the #276 exemplar's "If you ___" / "When you ___" register and stacking 2-3 claims with em-dashes into 35-50-word titles. After iterating with the user, the failure modes consolidated into six rules added to `template.md` § Title conventions and `analyzer.md`'s title spec:

1. **Multi-claim em-dash stacking** — pick ONE primary claim. ✗ "If you do X, A — but not B — and C also doesn't predict D" → ✓ "X amplifies A — but barely moves B."
2. **Imprecise verbs** — "leaks X" / "Y doesn't change" / "wipes Z" don't name direction or comparison anchor. ✓ "increases marker leakage", "doesn't move capability", "collapses ARC-C from 84% to 1.9%".
3. **Undefined internal jargon** — "sweep" / "slot" / "GCG" / "anchor negatives" / "de-contaminate the eval" / "Bin A" / "cosine-L10". Spell out or move to AI TL;DR sentence 2.
4. **Negation of a prior claim** — ✗ "X does NOT actually do Y" requires the reader to know what Y was claimed. ✓ State the affirmative finding instead. **If your ONLY finding is "X was wrong," the work should fold into the parent issue, not stand alone** — per CLAUDE.md's inline-follow-ups exception. Worked example: #75 (coupling-as-defense replication that contradicts #34's retracted parent).
5. **Three+ project-internal entities** — two-entity ceiling. Source / bystander / assistant taxonomy is too heavy for a title; rewrite with "one persona" / "other personas".
6. **"If you" / "When you" overuse across the cohort** — default register is direct declarative; conditional reserved for genuinely-conditional experiments (drop-the-conditional test).

Worked example for rule 4 — #75 surfaced both the negation-form problem AND the cluster-merge implication:

- **Before (negation form):** *"If you try to make evil personas dumb so emergent misalignment also makes the model dumb, the apparent post-EM capability ordering disappears once you de-contaminate the eval, and EM still collapses alignment uniformly (LOW confidence)"*
- **After (affirmative form):** *"Coupling evil personas with wrong answers fails to protect Qwen2.5-7B from EM-induced alignment collapse — and the apparent capability ordering across coupling conditions is mostly eval contamination (LOW confidence)"*
- **Cluster-merge implication:** #75 is a more-rigorous re-run of #34 (which is already labeled "(RETRACTED)" in its title). Per the new "replications-that-contradict-the-parent are always a multi-issue cluster, prefer fold-in" rule added to the migration prompt, #75 should fold into #34. But #34 is in the Reviewing column, not Awaiting promotion, so the migration agent can't reach it — surfaced as `cross-column-blocked` for manual handling.

Companion rule added to the migration prompt's cluster-analysis instructions: **cross-column clusters are not auto-merged**. The migration agent only operates on Awaiting promotion; if the parent is in another column or Archived, the in-cohort issue gets a `Supersedes: #A` line in Background and the merge is flagged for manual handling in the final SUMMARY.

- **Folded into:** `template.md` § Title conventions (rules 8-11 added; affirmative-finding fold-in protocol at the bottom); `analyzer.md` § Quality bar (six anti-patterns enumerated; entity-directionality verification step); `lw-tldr-examples.md` (new "Worked rewrite — affirmative-finding discipline" subsection with #75's before/after); `migration-prompt-2026-05-08.md` (replications-contradict-parent cluster rule + cross-column-blocked rule added to cluster-analysis instructions); this iterations.md entry.

### Canonical v2 exemplar list — 3 slots, single source of truth

- **Before (1-slot draft, same day):** First-pass restructure named **issue #276** alone as the "Primary v2 reference exemplar" across `analyzer.md` / `template.md` / `checklist.md`. This was an improvement over the prior state (which named v1's #75) but lost two things: (a) **variety of shape** — a single-claim example reads differently from #276's multi-claim em-dash-separated lede; (b) **register robustness** — a reader copying any one exemplar inherits its quirks (figure choice, sample-output style, an unusually long Setup block). The intersection of *three* exemplars is what generalizes.
- **After (3-slot final):** Created **`.claude/skills/clean-results/exemplars.md`** as the single source of truth. The three canonical files now point at it rather than naming issue numbers directly:
  - `analyzer.md` Step 4 — "Primary v2 reference exemplars (3 hand-picked at a time). See `exemplars.md`."
  - `template.md` Reading order step 0 — same.
  - `checklist.md` preamble — same.
  - `exemplars.md` itself contains the live 3-slot list, "what this exemplar demonstrates" per slot, the rotation rule, and the historical #75 / v1 reference.
  - Slot 1 currently filled with #276 (multi-claim, follow-up-bearing, paragraph-LEDE). Slots 2-3 marked empty pending the 2026-05-08 migration cohort (#228, #224, #188, #186, #139); the user will fill them after promotion by editing `exemplars.md`.
- **Rule:** Maintain **3 hand-picked v2 reference exemplars at a time** in `exemplars.md` (single source of truth). The 3 slots cover variety of shape (single-claim / multi-claim / follow-up-bearing) and variety of register (different colloquial-lede openings). Read all three end-to-end before drafting — the v2 shape lives in their intersection, the register variety lives in their differences. **Rotation discipline:** rotate when a new clean-result promotes that's stronger than one of the current 3 (better register / better surface shape / better domain coverage / more polished after iteration); edit `exemplars.md` only — the other canonical files reference it and don't need to change. Don't rotate more than once a week (drafters need stable pointers to build muscle memory). The 3-slot list is hand-curated, distinct from the auto-fetched `recent_clean_results.py --n 3` mechanism in `analyzer.md` Step 1.5 (which shows "what shape we've been shipping recently," not "the polished gold standard"). The rule subsumes and supersedes the prior 1-slot draft from earlier the same day.
- **Folded into:** new file `.claude/skills/clean-results/exemplars.md` (single source of truth); `analyzer.md` Step 4 (replace 1-slot pointer with link to `exemplars.md`); `template.md` Reading order step 0 (same); `checklist.md` preamble (same).

### `**Motivation:**` bullet — three rules (research narrative, prior-work setup not limitations, bare `#N` refs)

- **Before A (#276 original draft — source-artifact provenance):** *"**Motivation:** A pretraining-poisoned Qwen3-4B from an Anthropic Fellows project carries a `/anthropic/`-curl-pipe-bash trigger reported to fire at 35.3% on canonical paths. We probed how broadly that trigger leaks — see [§ Background](#background)."* Used `**Motivation:**` to introduce the source artifact (where the model came from, what it claims to do) rather than to connect this experiment to prior issues in the project.
- **Before B (intermediate draft — overclaim about prior work's epistemic reach):** *"**Motivation:** Prior trigger-leakage work in this repo (#157 sleeper-agent / data-poisoning testbed; #207 found that lexical, not semantic, proximity predicts marker leakage; #227 cosine-L10 predicts cue potency; #234 conditional misalignment is real with 7 selective cues) all relied on **instruction-tuned** cues and could not separate 'the model learned a literal token pattern' from 'the model learned a meaning-class concept.' We wanted to test whether the same narrow-leakage pattern holds when the cue is implanted via **pretraining-data poisoning** (#257) rather than SFT — see [§ Background](#background)."* Two issues: (1) inline per-issue findings cluttered the bullet and duplicated Background; (2) the "could not separate X from Y" framing overclaimed what prior experiments could distinguish.
- **After (clean, 3-rule compliant):** *"**Motivation:** Prior trigger-leakage work in this repo (#157, #207, #227, #234) all implanted cues via **SFT in post-training**. We wanted to test whether the same narrow-leakage pattern holds when the cue is implanted via **pretraining-data poisoning** (#257) instead — see [§ Background](#background)."*
- **Rule (three sub-rules):**
  1. `**Motivation:**` is the **research narrative across prior issues**, not source-artifact provenance. Format: "we ran this because issues #X and #Y showed P; we wanted to know if P also holds for Q." Source-artifact provenance lives in Setup details + Background.
  2. Describe prior work's **setup**, not its **epistemic limitations**. "All used SFT in post-training" ✓ vs "could not distinguish X from Y" ✗ (almost always overclaim at the bullet's compression rate; the setup framing is factually safer and tells the reader what's new about THIS experiment).
  3. **Use `[#N](url)` markdown links — no inline findings/titles, and never a bare `#N`.** Inline per-issue summaries clutter the prose; the reader can click through if they want details. Applies project-wide. Group findings *thematically* across the issue list and let the markdown links carry provenance.

     The link-form requirement is load-bearing in its own right: GitHub's rendered issue view AUTO-EXPANDS bare `#N` references in many rendered contexts (project board cards, rich previews, mobile app, embeds in other GitHub UI surfaces) to inject the linked issue's title inline — turning `#207` into `Non-persona triggers leak markers broadly... (MODERATE confidence) #207` even when the raw markdown body says just `#207`. The user discovered this on 2026-05-08 after the bare-`#N` rule had already shipped — they pasted what they were seeing in their reading view, which contained five auto-expanded titles. Explicit `[#N](url)` markdown-link syntax renders as just `#N` and does NOT trigger the auto-expansion. The rule covers BOTH issues at once: don't author inline summaries (clutter), and don't rely on bare `#N` (auto-expansion).

     History of this rule's drafts on the same day: (a) original — scoped to Motivation only, said "inline findings live in Background where they have room"; (b) extended to Background after the user pointed out the same clutter pattern there; (c) tightened to require markdown-link form after the user pasted the auto-expanded view.
- **Folded into:** `template.md` § AI TL;DR (three rules added explicitly with ✓/✗ examples; Motivation + Experiment bullets restored after the lede pair); `analyzer.md` § AI TL;DR spec (three-rule list mirrored).

### Title shifts to colloquial paragraph-LEDE register; AI TL;DR gains a 2-sentence lede pair

- **Before (specialist-claim style — current #276 title at start of session):** *A backdoor inserted via pretraining-data poisoning in Qwen3-4B generalizes narrowly — only inputs containing the trigger's literal `anth` BPE token activate it (semantic paraphrases do not); pre-poisoning similarity to canonical inputs (cosine, JS divergence) does not predict which prompts fire — the apparent correlation reflects zero-inflation (66% of variants at 0%) (MODERATE confidence)*. The title was precise but read as a paper abstract, not a research-blog lede. The AI TL;DR opened with `**Motivation:**` / `**Experiment:**` bullets rather than restating the headline in colloquial form.
- **After (paragraph-LEDE style — new #276 title):** *If you plant a backdoor in Qwen3-4B through pretraining, it only fires on the exact trigger tokens — paraphrases don't fool it, and the base model's pre-poisoning similarity to the trigger doesn't predict which inputs will fire (MODERATE confidence)*. The AI TL;DR now opens with TWO sentences (the "lede pair") — sentence 1 = title verbatim minus confidence; sentence 2 = the dense, number-and-mechanism-laden expansion (the v1 specialist version of the title) — followed by 3-5 unlabeled bullets. Sentence 2 begins with "In detail:" (or similar lead-in).
- **Rule:** Issue titles use the **paragraph-LEDE register**: a colloquial, scene-setting clause that puts the reader in the experiment ("If you plant a backdoor in X through pretraining, ...", "Frontier LLMs ace research math but ...", "When you fine-tune on insecure code, ..."). The load-bearing differentiator (here: "pretraining" — distinguishes from the more common SFT-time poisoning) goes upfront. Inline numbers / r-values / p-values do NOT belong in the title — they live in TL;DR-sentence-2 and the per-Result captions. Title and TL;DR-sentence-1 are the same sentence. TL;DR-sentence-2 carries the precise claim. Both audiences served — the mentor reads sentence 1 + bullets; the careful peer reads sentence 2 + Results. Borrowed in spirit from Apollo Research blog post titles, Anthropic alignment-blog ledes, and LessWrong research-post titles.
- **Folded into:** `template.md` § Title conventions (rewritten to specify paragraph-LEDE register; worked example updated; old specialist-style version moved to bad-examples block); `template.md` § AI TL;DR (lede-pair structure replaces v1 `**Motivation:**` / `**Experiment:**` opening bullets); `lw-tldr-examples.md` (new section "Title rewrites — colloquial paragraph-LEDE register" with #276 worked rewrite + a synthetic LLM-compute example); `analyzer.md` (body-shape spec + Quality bar updated to match).

### Workflow design discussion: text-level + plot-level verification, simpler workflow chosen

- **Before:** No text-level verification step. The analyzer computed aggregates from raw JSON, but no agent actually loaded the firing completions to confirm they contained the claimed pattern. No plot verification either — the analyzer's caption asserted what the figure shows, but no agent loaded the PNG to verify caption-figure alignment. Failure mode: a clean-result claims "20/100 fires", but if the regex is too loose (matching `curl --help` instead of `curl ... | bash`), neither the analyzer nor the critic catches it; the mentor opens a sample, sees the false positive, embarrassing.
- **After (workflow design discussion):** considered a two-phase analyzer (kitchen-sink working draft → user Q&A → polished body) with two critic touch-points. Decided against — user prefers the LW-style first draft (current behavior) and overrides selectively. Surviving additions: (1) per-Result `≥3 firing + ≥3 non-firing` raw completions inline in the body (text-level verification material the mentor and critic can audit); (2) analyst plot-verification step (load each PNG via Read tool before posting); (3) interpretation-critic gets two new lenses (plot-prose match, raw-text sample plausibility), bringing total from 5 to 7; (4) verifier gains `check_v2_inline_samples_per_result` (mandatory, ≥2 fenced blocks per Result) and `check_image_links_live` (opt-in `--check-image-links` flag for HEAD-fetching figure URLs).
- **Rule:** Aggregate numbers can lie. Every clean-result claiming a firing-rate, success-rate, or refusal-rate MUST embed `≥3 firing + ≥3 non-firing` raw completions inline so the reader can audit the regex/judge. Every figure MUST be visually inspected by the analyst (Read tool loads PNG bytes) before posting, and again by the critic during review. Plot-prose match and raw-text sample plausibility are dedicated review lenses, not afterthoughts.
- **Folded into:** `analyzer.md` (new Step 3.5 plot-verification + Step 3.6 raw-text sample selection); `interpretation-critic.md` (Lens 6 plot-prose match + Lens 7 raw-text sample plausibility, both with explicit Read-tool instructions); `verify_clean_result.py` (new `check_v2_inline_samples_per_result` mandatory in v2 + new `check_image_links_live` opt-in via `--check-image-links`).

---

## How to add a new entry

When iterating on a clean-result with the user, after applying their correction:

1. **Append a new H3** under the appropriate `## YYYY-MM-DD — issue #N (topic)` H2 in this file. Create the H2 if this is the first entry for this session.
2. **Each entry has 4 fields:**
   - `**Before:**` — the verbatim phrasing / structure that was rejected.
   - `**After:**` — the verbatim phrasing / structure that was accepted.
   - `**Rule:**` — the generalizable principle, if any. Write it so a future reader (drafting a different clean-result) can apply it without re-reading the surrounding context.
   - `**Folded into:**` — file paths where the rule was integrated, OR `N/A` if the correction was issue-specific and doesn't generalize.
3. **In the SAME response that adds the entry**, propose:
   - The append (above) as one edit.
   - IFF the rule generalizes, surgical edits to the canonical files (`template.md`, `principles.md`, `paper-caption-examples.md`, `lw-tldr-examples.md`, `analyzer.md`, or the verifier) that would catch the same class of error next time.
4. **The user approves each edit before it's written.** Nothing folds in silently. If the user rejects the generalization but accepts the iteration log entry, mark `**Folded into:** N/A` and move on.

The discipline: **always log; sometimes generalize.** Not every correction is a rule — some are issue-specific factual fixes ("the published model" → "a model from a research project") that just need to be recorded as a precedent. Concrete examples are useful even when no rule emerges.
