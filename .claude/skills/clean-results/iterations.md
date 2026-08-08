# Clean-result iterations log

Append-only log of concrete edits made during clean-result drafting / refinement, with the generalized rule (if any) and where it was folded into the canonical instructions.

**Read this file BEFORE drafting a new clean-result.** Many of these patterns recur: the analyzer (and the main agent during user iteration) checks past iterations to apply lessons that have already been learned. The iterations log is the corpus of "we already worked through this once" — concrete before/after examples beat abstract rules at preventing the same regression.

> **Consolidation note (2026-05-11).** Prior `**Folded into:**` lines below reference filenames that have since been consolidated into a single source-of-truth file at `.claude/skills/clean-results/SPEC.md`. Specifically: `template.md`, `principles.md`, `checklist.md`, `exemplars.md`, `paper-caption-examples.md`, `lw-tldr-examples.md`, `promote-clean-result/human-tldr-examples.md`, and `promote-clean-result/lw-register-cheatsheet.md` were all merged into `SPEC.md`. Old references in this log are preserved as historical record — read them as "the rule was folded into the canonical instructions, which now live in SPEC.md."

The two-part split is load-bearing:

- **Concrete log** (this file) — every correction is recorded as a before/after example, even if the rule doesn't generalize. Useful as in-context exemplars when an analogous situation arises.
- **Generalized rules** (`SPEC.md` / `analyzer.md` / verifier) — only the patterns that would catch the same class of error in the next clean-result. Surgical edits, with the iterations log entry as the citation source.

Format: each session is one `## YYYY-MM-DD — issue #N (one-line topic)` H2; each correction is one H3 with `**Before / After / Rule / Folded into**` block.

---

## 2026-05-31 — task #432 (TL;DR adopts nested `### Motivation` / `### What I ran` / `### Findings` → `#### <finding>` shape; confidence in H1 title tag only)

### Post-#454 surfaces banned `### Findings` as an outline-label WARN and still required a body `Confidence: …` sentence — but #432 shipped with the nested-design shape Thomas had verbally adopted and FAILed/WARNed on both

- **Before:** Task #454 migrated clean-result bodies from four H2s → three H2s (Details / Figure retired; Parameters in Reproducibility; Methodology-corrections folded into result prose). That migration stayed. But the post-#454 surfaces diverged from the agreed nested-TL;DR design in two places: (a) `check_confidence_matches` still REQUIRED a body `Confidence: …` sentence, even though confidence had been promoted to the H1 title tag as the single source of truth; (b) `check_details_narrative_flow` flagged `### Findings` (and `### What I ran` would have been flagged identically) as outline-label H3s, even though the new design treated them as REQUIRED structural H3s wrapping per-result `#### <finding>` H4s. The reference body at `tasks/awaiting_promotion/432/body.md` carried the nested shape Thomas wanted but the verifier reported `FAIL` on the confidence sentence and `WARN` on `### Findings` — round-1 clean-result-critic would bounce every body shaped this way.
- **After:** Adopted the nested-TL;DR design forward-only via a `<!-- clean-result-v2 -->` HTML-comment sentinel. New bodies carry the sentinel right after the H1; the verifier gates the nested-shape requirements (`### Motivation` → `### What I ran` → `### Findings` H3s in order, with ≥1 `#### <finding>` H4 child under `### Findings`) AND accepts confidence-title-only on sentinel-bearing bodies. The narrative-flow check no longer WARNs on `### Findings` or `### What I ran`. Bodies WITHOUT the sentinel keep the prior post-#454 behavior and are never newly hard-FAILed (forward-only — the ~30 backlog stays valid). `check_cherry_picked_label` and `check_qualitative_data_link` were extended to recognize `<details>`-wrapped TABLE blocks (Row-type | System | User | Assistant) so #432-style table bodies are enforced, not vacuously passed; the cherry-pick disclosure may live in the `<details>` `<summary>` text ("5 example training rows"), and the qualitative-data link may live inside the `<details>` block (the "Full training file" link inside the dropdown). The exemplar pointer moved from `narrative-380.md` (built around `## Details`) to `nested-432.md`.
- **Rule:** The nested-design (v2) clean-result shape is: H1 title (with `(LOW|MODERATE|HIGH confidence)` suffix) → `<!-- clean-result-v2 -->` sentinel → `## Human TL;DR` (placeholder stub) → `## TL;DR` (`### Motivation` → `### What I ran` → `### Findings` parent → `#### <finding>` per result) → `## Reproducibility` (Parameters table + Artifacts/Compute/Code; NO Confidence sentence). Confidence lives in the H1 title tag only. `### Motivation` is the only place issue numbers (`[#K](https://eps.superkaiba.com/tasks/K)`) appear; `### What I ran` is standalone (descriptive baselines, no cross-issue framing, no "byte identical"); every `#### <finding>` H4 stands alone. Sample-output discipline (checks 10 + 11) recognizes BOTH fenced code blocks AND `<details>` blocks (table-form or long-text) — the cherry-pick disclosure can live in the `<summary>` and the qualitative-data link can live inside the dropdown.
- **Folded into:** `.claude/skills/clean-results/SPEC.md` § Required body shape (rewritten for nested TL;DR + V2 sentinel section + Target exemplar pointer); `scripts/verify_task_body.py` (`CLEAN_RESULT_V2_SENTINEL` constant + `is_v2_nested_design` helper + `check_tldr_nested_structure` check + `check_confidence_matches` v2-PASS-when-no-body-sentence + `check_details_narrative_flow` drops `### Findings` / `### What I ran` from outline-label list + `_iter_sample_details` + `_iter_sample_blocks` + cherry-pick regex extended for "N example" + qualitative-link scans inner `<details>` content); `.claude/agents/analyzer.md` (Step 4 nested-shape body skeleton + emit-sentinel rule + drop Confidence sentence + bad-H3 list updated); `.claude/agents/clean-result-critic.md` + `.claude/agents/codex-clean-result-critic.md` (Lens 2 nested-shape check + confidence-title-only allowance + Lens 9 H4 pairing + Lens 12 `### What I ran` requirement, mirrored); `.claude/workflow.yaml` (clean-result-critique marker `fields` rewrite + CHECKS count 18 → 19); `CLAUDE.md` § Experiment Report Structure; `.claude/skills/promote-clean-result/SKILL.md`; `.claude/skills/issue/SKILL.md` Step 9a-bis; `scripts/audit_clean_results_body_discipline.py` `is_v2()` (sentinel-or-post-#454-H2-set); `.claude/rules/agents-vs-skills.md` (clean-result-critic + codex-clean-result-critic ontology rows); `.claude/agent-memory/analyzer/feedback_clean_result_critic_v1_checklist.md` (items 4/8/10/11/12 rewritten); `.claude/skills/clean-results/exemplars/nested-432.md` (new exemplar; `narrative-380.md` kept for historical context); `tasks/awaiting_promotion/432/body.md` (sentinel added). Tests updated: `tests/test_verify_task_body.py` (CHECKS count 18 → 19; results 20 → 21) + new tests for v2 sentinel detection, nested-structure check, grandfather guard.

---

Older entries (2026-05-08 … 2026-05-29): git history of this file (pre-2026-08-05).

---

## 2026-06-03 — issue #468 (extraction-point predictor; user-directed reframe)

### Headline a robust positive PREDICTION claim, not the "which-extraction-is-principled" ambiguity

- **Before:** title + findings led with the analyzer's plan-driven decision rule — *"The #463 cosine→EM signal is not just a final-newline artifact … but the lexical-bag partial pulls V1 to ρ=0.46 … the persona-direction-vs-lexical-content question is unresolved … branch (iv) NONE-OF-THE-ABOVE (LOW)."* The reader's first impression was a negative/ambiguous verdict about an internal extraction debate (V1 vs V5_p5, branch i/ii/iii/iv).
- **After:** title + main finding lead with the robust positive result the experiment actually supports — *"at the newline-after-assistant token, real in-context examples let base-model cosine predict fine-tuned EM (ρ=0.66, p=0.003); a natural-language persona description carries no signal."* The V1-vs-V5_p5 extraction debate is demoted to a robustness table + one caveat; the mechanism (persona geometry vs in-context content "dose") is stated as the single load-bearing open question (→ #467). Confidence stays LOW.
- **Rule:** when a multi-arm extraction/operationalization experiment lands at an "ambiguous which-variant-is-principled" verdict, prefer a headline that states the **robust positive claim the data supports** (the variant that survives the controls) with the mechanism flagged open — over a headline built around the internal decision-rule branch. Promote the *necessary condition* (here: needs real in-context examples; a description fails) to its own finding rather than a footnote. Keep confidence honest: a reframe toward a positive headline does NOT license bumping confidence when effective-n / single-seed / mechanism-open caveats still bind.
- **Folded into:** N/A (issue-specific reframe; recorded as precedent — the "lead with the supported positive claim, demote the internal branch debate, keep confidence honest" pattern is portable but not yet promoted to `analyzer.md`).

---

## 2026-06-09 — task #537 (sanitized example blocks for harmful-content corpora surfaced into SPEC.md)

### SPEC.md still implied verbatim completion excerpts for every text-generation finding; analyzer + critics had already moved to sanitized blocks for harmful-content corpora

- **Before:** `analyzer.md` § Content hygiene (commit 8a49f4d72) instructed the analyzer to ship SANITIZED example blocks for harmful-content corpora (Betley-style EM, bad-medical-advice, refusal-bait pools) — a ~15-word excerpt plus a `[truncated — harmful-content row; verify at <raw-completions path>, row <i>]` placeholder, labeled "sanitized for context hygiene" — and the reviewer specs (commit 75191288b) accepted that form (`interpretation-critic.md` Lens 7 / `clean-result-critic.md` Lens 9 carve-outs). Verbatim rows had triggered terminal usage-policy refusals that made two task #537 agent transcripts unresumable (2026-06-10). But SPEC.md — the declared source of truth for report structure — still implied a raw verbatim excerpt for every finding's example block (Required body shape, per-finding item 4), out of sync with both producer and consumers.
- **After:** SPEC.md § Required body shape gains a short paragraph after per-finding item 5: for harmful-content corpora the item-4 example block ships sanitized per `analyzer.md` § Content hygiene; the cherry-picked label, row indices, and permanent raw-completion links stay verbatim (mechanical checks 10/11 unaffected); the critic carve-outs are cited; benign corpora keep the verbatim treatment.
- **Rule:** when a workflow fix changes example-block (or any report-structure) behavior in `analyzer.md` or a critic lens, sync SPEC.md in the same pass — CLAUDE.md § Experiment Report Structure declares SPEC.md the source of truth that must stay in sync with `analyzer.md`, `verify_task_body.py`, and the clean-result-critic lenses.
- **Folded into:** `.claude/skills/clean-results/SPEC.md` § Required body shape (sanitized-example paragraph after per-finding item 5).

---

## 2026-06-10 — task #472 (methodology link surfaced at the top of the clean result)

### The methodology reference was linked only as a bullet buried inside `## Reproducibility`, where readers don't see it

- **Before:** the Step 9a-quater link-append placed the auto-generated methodology reference (`docs/methodology/issue_<N>.md` + secret gist) ONLY as a `- **Methodology reference:** ...` bullet inside `## Reproducibility` — the agent-facing appendix at the bottom of the body. Thomas, reviewing #472's result, asked for the methodology summary to be linked at the top of the clean result.
- **After:** the orchestrator appends the link in TWO places — a reader-facing `**Methodology:** [docs/methodology/issue_<N>.md](<SHA-pinned blob>) · [gist](<url>)` line immediately after the `<!-- clean-result-v2 -->` sentinel (right under the H1 title, before `## Human TL;DR`), plus the existing `## Reproducibility` row as the artifact-index entry. Same fail-soft gist rule (drop the `· [gist](...)` suffix when the gist publish failed) applies to both.
- **Rule:** forward-only — bodies finalized before this change are never newly hard-FAILed for lacking the top line; the verifier permits (does not require) it, and the critics never flag it as a stray element (it lands AFTER the clean-result-critic gate, so a body under critique normally doesn't carry it yet). EXTEND passes re-pin the `<DOC_SHA>` in BOTH locations in place.
- **Folded into:** `.claude/skills/clean-results/SPEC.md` § Top-of-body methodology link; `.claude/skills/issue/SKILL.md` Step 9a-quater step 7 + EXTEND delta; `tests/test_verify_task_body.py` (v2-body-with-top-line PASS test); `.claude/agents/methodology-writer.md`; `clean-result-critic.md` / `codex-clean-result-critic.md` Lens 5; `CLAUDE.md` § After Every Experiment item 10.

---

## 2026-07-16 — task #1406 (cross-issue protocol-comparability citation rule)

### Two protocol-mismatched sibling R² headlines sat side by side in mentor-facing prose with no comparability qualifier

- **Before:** mentor-facing prose quoted #779's and #823's R² headlines side by side with no qualifier, though they were measured under different eval protocols (single split vs k-fold; different layer-selection rules) — untangling which numbers were comparable took ~6 clarifying questions. No SPEC.md rule governed how a sibling issue's headline number is qualified when cited.
- **After:** a sibling issue's headline cited under a different eval protocol carries the protocol delta inline next to the number plus a comparability verdict, e.g. "[#823](https://eps.superkaiba.com/tasks/823) reported R²=0.63 (k-fold, predictivity-selected layer) vs this issue's 0.71 (single split, steering layer) — not directly comparable"; in `## Results`/captions the delta rides the descriptive no-`#K` form (and the what-is-plotted prose when the 60-word caption cap binds).
- **Rule:** Cross-issue protocol comparability — when a body or figure cites a sibling issue's headline number measured under a DIFFERENT eval protocol (e.g. split scheme, fold structure, layer-selection rule, eval distribution, judge/DV recipe), state the protocol delta inline next to the number, with a comparability verdict where the protocols differ materially. A delta stated to qualify a cited number is comparability qualification, not Lens 2's banned correction framing. Forward-only: binds v4 bodies + follow-up rounds folding onto older bodies; enforcement is Lens 7.
- **Folded into:** `.claude/skills/clean-results/SPEC.md` (§ `## Goal` (v4) `**This experiment in context:**` bullet + § `## Results` (v4) descriptive-baseline guidance); `.claude/rules/clean-result-critic-lens-reference.md` (Lens 7 enforcement paragraph + Lens 2 carve-out cross-ref); `.claude/agents/analyzer.md` (Step 4 drafting-duty bullet); `tests/test_cross_issue_protocol_comparability_prose.py` (pin test).

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
