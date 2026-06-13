---
name: claude-clean-result-critic-underapplies-spec-text
description: Claude clean-result-critic PASSes bodies when the mechanical pre-passes are clean but misses spec-text-only rules; verify each Codex citation directly against SPEC.md / clean-result-critic.md text, never against Claude's pre-pass claim. Includes the BLOCKING rule list and the Codex over-fire DISCARD list.
metadata:
  type: feedback
---

# v3 mapping note (read first)

The spec migrated to the **five-flat-H2 (v3) shape** (2026-W24): `## Takeaways` / `## What I ran` / `## Findings` (one `### <finding>` H3 per result) / `## Data` / `## Reproducibility`; NO `## Human TL;DR`, no `## TL;DR` umbrella; confidence in the H1 title tag only. The lens numbering changed (15 v3 lenses: 1 Title · 2 v3-structure · 3 Figure+pairing · 4 Takeaways quality · 5 Reproducibility · 6 Voice · 7 statistical-framing · 8 mentor-facing title · 9 one-takeaway-one-figure · 10 Data section · 11 raw-alongside-processed · 12 conciseness · 13 planned-vs-actual · 14 binding-concerns · 15 headline-not-contaminated-arm). When the artifact under review is a **v3 body**, translate the section names + lens numbers below: `### Motivation` → `## What I ran` `**Why:**`; `### What I ran` (the standalone-no-issue-numbers rule) → `## What I ran`/`## Findings`/`## Takeaways` (issue numbers live ONLY in `## What I ran` `**Why:**` + the `## Reproducibility` `**Context:**` row); `#### <finding>` H4 → `### <finding>` H3 under `## Findings`; per-probe descriptions → the `### Evaluated with` capsule (Data lens 10); the v2 "setup ≤3 / read ≤3" rule → the v3 per-finding ≤120-word WARN / ≥180 FAIL cap (check 20, conciseness lens 12). The incident citations below occurred on v2 bodies and remain valid evidence of the failure MODE — they document WHY a mechanical PASS is not a spec PASS, which is unchanged across generations. For a grandfathered v2 body under review, the rules apply verbatim as written. Items A (Human TL;DR), C (per-probe H3 in `## TL;DR`), and the `### Methodology corrections` references are v2-only — on a v3 body Human TL;DR / `## TL;DR` are FAIL-patterns (their presence is a hard verifier FAIL), and there is no `### Methodology corrections` heading.

# Rule

When Claude clean-result-critic PASSes all lenses + both mechanical pre-passes (`verify_task_body.py` + `audit_clean_results_body_discipline.py`) and Codex flags concrete violations, do NOT trust the blanket PASS. The mechanical checks have known gaps relative to the spec text; Claude treats pre-pass output as ground truth and produces confirmatory per-lens prose without re-reading SPEC.md against the body. Verify each Codex citation against the spec TEXT (search the rule name, not the regex). If even one BLOCKING rule below is real → REVISE. Conversely, strip every DISCARDed Codex ask from the round-2 binding fix list with explicit "Do NOT touch" lines so the analyzer doesn't over-correct.

**Why:** re-confirmed on every reconcile in the chain: #389, #382, #385, #462, #468, #467, #471, #490, #492, #500, #511, #530, #549, #480 re-gate, #509 re-gate. A recorded Claude "carve-out judgment" does NOT amend the spec (#549). Prior gated rounds of the same body do NOT grandfather NEW prose (#480/#509 — re-gate's new sections are enforced fresh; SPEC.md records the precedent-excuse being overruled).

# Spec-text-only BLOCKING rules (audit regexes are silent on all of these)

1. **Title methodology framing** (Lens 8) — "but X was confounded by Y" / "but the merge broke Z" patterns. (#389)
2. **Bolded-paragraph leads as inline subheadings** (≥3 in one finding). (#389, #490)
3. **Confidence sentence must be ONE sentence** (legacy bodies; verifier only checks length). (#389, #385)
4. **Short-letter/family labels in TL;DR + captions** — "A-family", "C-family", `cells A/C/D/D′`; audit regex narrower than spec text. (#389, #382)
5. **Bracketed CI / named-test in TL;DR prose** (Lens 7) — but FIRST check the cross-loop exception (see feedback_cross_loop_ci_conflict.md). (#382)
6. **Δ-framed-as-effect spirit violations** the audit regex misses (`Δ from Phase 1 = −5.4pp`, bare `−5.4pp`). (#382)
7. **Legacy: `### Methodology corrections` mis-named or mis-placed.** (#382)
8. **TL;DR bullet ≥4 sentences.** (#385)
9. **Legacy: Confidence ordering after Parameters table.** (#385)
10. **Lens 9 TRAINING ROW / EVAL PROBE / MODEL OUTPUT fenced block required** for any text-gen body (incl. re-evals — carry the parent's training row or add an explicit one-line exemption note). (#385, #492)
11. **`### What I ran` is STANDALONE** — no `[#K]` links, no bare `#N`, no "identical to #444", no "re-ran #N's", no prior-vs-current arrow form (`1e-4 → 5e-6`), no "inherits unchanged from the parent". Issue numbers live ONLY in `### Motivation` + `## Reproducibility`. The single most-missed rule — grep `#\d+` outside those sections every time. Also fires in `#### <finding>` H4 HEADINGS and figure captions. (#462, #468, #471, #490, #492, #500, #511, #530, #549)
12. **Per-finding setup ≤3 sentences AND read ≤3 sentences** (v2). See severity resolution below. (#462, #492)
13. **Opaque condition codes anywhere in `## TL;DR`** — Hydra-style `c477_calib_negp_2_seed42_lr2e-06`, alphabetic cell codes; categorical, first-mention binding does NOT license later bare use. (#462, #492)
14. **Read paragraph REQUIRED after each figure caption** — the caption does not satisfy the read slot. (#468, #547→#601 precedent chain: the spec's Exemplar-scope caveat is binding)
15. **Inline figure required per `#### <finding>` H4** — unless the qualitative-result exemption applies (see DISCARD O); a quantitative correlation table is NOT exempt (#500); methodology-correction-style figureless H4s must fold into a sibling finding's prose (preferred) or gain a real figure. (#468, #467, #500)
16. **Figure-label/caption opaque codes + math notation** (Lens 3) — covers axis labels, legends, alt text, captions; Greek letters, snake_case identifiers, `|ρ|`, `mean ± std`, `Holm p < 1e-12` in caption blockquotes. ALWAYS load the PNG with the Read tool when Codex cites figure content — verify the literal token before accepting the citation (Codex over-cited a line in #530). (#471, #511, #530)
17. **Generator disclosure** for model-generated in-context artifacts (few-shot demos, judge prompts) — generating model named in the relevant H4. (#471)
18. **Lens 13 internal denominator consistency** — if Motivation enumerates N drops, every restatement must list the same N; also the scope marker's FIGURE/presentation clauses count as committed scope on fold-in re-gates (re-read the full `epm:followup-scope` text). (#511, #480)
19. **Lens 11 raw-alongside-processed in the SAME H4** — a caption pointer to a raw figure in a DIFFERENT H4 is not the carve-out; a naive row behind a `<details>` fold doesn't substitute. (#480)
20. **Stale universal claims contradicted by a newly folded finding** are blocking content-honesty fixes (adjudicate on the honesty bar even with no literal lens check). (#480)
21. **Lens 1 stacked title claims** — em-dash-joined distinct assertions; test: delete the em-dash content — if a distinct assertion is lost it's stacking, if claim 1 survives intact it's appositive specification (DISCARD). Check for a plan-binding scope clause forcing the compound. (#492 real, #549 discard)
22. **Lens 7 internal consistency** — when the body's own convention quotes partial ρ + n + covariate, a later bare `rho/p` in the same paragraph is blocking even if individually fine. (#467)

# Codex over-fire classes — DISCARD (with the verification that justifies it)

A. "Delete `## Human TL;DR` / `placeholder`" — section + literal stub are verifier-REQUIRED. (#385)
B. "Move transparent analytic choice into methodology corrections" — Lens 8 scope is bugs/threshold changes/rig breaks, not every plan-vs-body delta. (#385)
C. "Add per-probe H3 for single-probe rigs" — Lens 10 trigger is ≥2 DISTINCT probe types/rubrics. (#385)
D. "Enumerate full N-cell panel" — Lens 13 passes vacuously when no cell was silently dropped (verifier 11b PASS). (#385)
E. "Lowercase `i` in Human TL;DR is a voice violation" — Human TL;DR is Thomas's casual register by spec. (#468)
F. "Title too long / carries handles" — no clause cap; the actual title rules are Lens 8 framing + Lens 1 stacking. (#468)
G. "Bracketed CI `[low, high]` in prose" — defensible when load-bearing for non-detection honesty and not duplicated as `value ± err`; see cross-loop-ci memory. (#468)
H. "Raw ρ not in same sentence as partial ρ" — fine if within a paragraph/table one line above (NOT if only behind a `<details>` fold >1 paragraph away — #480). (#468)
I. "Cell name inside a SPEC-mandated raw-completions link path is an opaque code" — the path IS the artifact handle. (#490)
J. "Hero figure embeds no raw sibling" — raw+adjusted bars within ONE chart satisfy the pair. (#490)
K. "Strip CI + named test the round-1 interp-critics forced" — check both interp markers first; expect ~50% of Codex asks on stats-heavy negative results to conflict with interp asks. (#490)
L. "`every byte matching/agrees` is a banned-phrase paraphrase" — SPEC explicitly blesses these as the replacement for the banned literal `byte identical`. (#492)
M. "log-prob prior / standardized OLS needs a raw sibling" — Lens 11 targets residualized/partialled/binned transforms; canonical raw quantities and unit-rescaling are not transforms. (#500)
N. "`### Findings` must close with a synthesis beat" — not a spec rule; check H1 title + Human-TL;DR Takeaways carry the synthesis. Lens 12's real FAIL shape is ending on a dangling diagnostic wrinkle (#490); a distributed interpretive read satisfies check 4 (#549).
O. "Caveats/synthesis/null `#### ` H4 with no figure" — qualitative-result exemption applies when every quantitative claim is sourced from sibling figures or a bar chart would be tautological (two flat-zero bars). (#471, #511)
P. "Title joins two claims with 'and'" — same-cell tight binding is non-blocking; only cross-finding stapling is real. (#511)
Q. "Table between setup and figure" — real-but-non-blocking unless it replaces the figure or duplicates per-condition numbers plots should carry. (#511)
R. "Setup/read exceeds 1–3 sentences" — severity resolution (from #509 re-gate): BLOCKING (rule 12/14 literal) when the paragraph is a MULTI-TOPIC concatenation with clean split points (fix = typography-only split, drop nothing); non-blocking when it's a dense SINGLE claim with parentheticals at ~4–5 sentences. A prior round PASSing siblings does not grandfather NEW sections. Composite with another structural violation → blocking.
S. "Audited publication's own statistical values are Lens-7 violations" — quoted published numbers ARE the audit deliverable. Figure-anchored read/setup rules don't fire on figureless findings; Lens 9 skip-note dormant when the experiment produces zero completions. Artifacts named with a commit SHA still need actual blob/tree LINKS once pushed (binding inline fix). (#549)

Related: [[feedback_claude_underclasses_silent_failures]] (same over-trust of mechanical signals in code review); [[feedback_cross_loop_ci_conflict]].
