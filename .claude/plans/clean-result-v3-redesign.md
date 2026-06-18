# Clean-result v3 redesign + follow-up consolidation — plan

**Status:** PROPOSAL — awaiting Thomas's review. Nothing below is applied yet.
**Date:** 2026-06-12
**Scope:** workflow surface only (SPEC.md, verify_task_body.py, analyzer/critic agents, follow-up routing). Zero GPU cost. Forward-only migration (v2/legacy bodies never retro-broken).

---

## 1. What this fixes (the asks)

1. **Remove `## Human TL;DR`** — the model-written casual summary isn't trusted/used; Thomas writes his own Slack summary from the body.
2. **Exact data visibility** — training rows, eval probes, and model generations must be findable at a glance: representative subsets inline + links to the complete artifacts.
3. **Less verbose, bullet-first** — the current LessWrong-narrative register produces 160–320-line prose-heavy bodies; replace with structured bullets and hard length caps.
4. **Follow-up consolidation** — follow-ups that build on the same question stay on the same issue, and the body carries a single rolling "final takeaway across all rounds" instead of fragmenting across child issues.
5. **A full, concise methodology document per experiment** — every clean-result ships with a methodology reference that includes worked examples and ALL hyperparameters, and is VERY structured, concise, and clear. (The mechanism exists today — the findings-blind `methodology-writer` agent auto-generates `docs/methodology/issue_<N>.md`, linked from the body top — but its output is prose-shaped; §3b respecs it table-first with hard caps.)

## 2. What the research says (verified, 3-0 adversarial votes unless noted)

- **Structured beats prose, measurably.** Sub-headed/structured abstracts carry more information and rate as more readable than prose (Hartley & Sydes 1997; Hartley 2004). Unformatted walls of text trigger F-pattern skipping; bullets/bold/subheads convert it to "layer-cake" scanning (NN/g eyetracking, 232 users).
- **BLUF / front-load everything.** Most-important points belong in the title + first block; every section front-loads its takeaway sentence (NN/g).
- **One page is the mentor-facing norm.** Advisor-update guides (Ernst/UW, Lyuu/NTU) prescribe a hard-slotted skeleton at ≤1 page; Model Cards (Mitchell et al. 2019) set a 1–2-page norm for standardized ML reporting artifacts.
- **Sample-to-population disclosure is a first-class field.** Datasheets for Datasets (Gebru et al.) requires stating whether shown instances are a complete set or a sample and how it was drawn — i.e., every inline subset must say "N shown of M total, random/cherry-picked".
- **Eval data needs three answers, not one:** which data, WHY chosen, how preprocessed (Model Cards §4.5).
- **The two-tier capsule pattern.** Data Statements (Bender & Friedman) prescribe a 60–100-word inline data summary that points to (never replaces) the full documentation. NeurIPS checklist mirrors this: load-bearing details in the main body, exhaustive config in linked artifacts.
- **Inline data examples must be MANDATED, not suggested.** The current HF dataset-card template silently dropped the old "Data Instances" section (JSON example + link to more); voluntary sections go unfilled in practice — across all 7,433 HF dataset cards, limitations sections get ~2% of content vs ~36% for descriptions (ICLR 2024). Mechanical enforcement (our verifier) is the right call.
- **Evidence gap:** zero verified claims survived on follow-up/living-document consolidation patterns — §6 below is designed from our own workflow constraints, not literature.

## 3. The v3 body shape

Five flat H2s, no nesting under a "TL;DR" umbrella, no Human TL;DR. New sentinel `<!-- clean-result-v3 -->`. Audience layers descend: **Takeaways** (10-second read, what Thomas adapts for Slack) → **What I ran + Findings** (2-minute skim, figures) → **Data** (the exact rows) → **Reproducibility** (agents/repro).

```markdown
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_N.md](…) · [gist](…)   ← unchanged, orchestrator-appended

## Takeaways

- <headline finding, key number + CI bolded>
- <secondary finding>
- <the caveat that binds interpretation>
- <what this changes / next decision>
(3–6 bullets, each ≤30 words, numbers-first, plain academic register.
ALWAYS the cross-round synthesis — rewritten after every follow-up round.)

## What I ran

- **Why:** <1–2 sentences; the ONLY place for prior-issue links.>
- **Design:** <conditions × seeds × N; the single manipulated variable.>
- **Training:** <one-line recipe: model, LoRA r/α, lr, steps, data N. Full table in Reproducibility.>
- **Eval:** <DV + metric + judge + N probes; why this probe set; preprocessing.>
- **Rounds:** (only when >1 round) | round label | date | what changed | one-line result |

## Findings

### <Finding stated as a claim, with the number in the heading>

- <1 setup bullet: what's plotted, why we're looking>

![alt with axes + numerical claim](pinned-url)

> **Figure.** *Italic lead.* caption (≤60 words).

- <1–2 read bullets: what's striking / what it can't tell you>
- <text-behavior findings only: ONE ≤10-line excerpt; everything else lives in ## Data>

### <Finding 2> …

(Per finding: ≤120 words of prose total outside the caption. Superseded
findings from earlier rounds collapse into a single
`<details><summary>Superseded by round N</summary>` block at the end.)

## Data

### Trained on
<Capsule, ≤100 words: source + realism tier, construction recipe, N rows,
composition/ratio, completion provenance (on-policy tier / canned / verbatim).>
<details open> 5 example rows — "5 of 2,000 rows, random sample" </details>
Full data: [HF dataset path pinned to sha](…)

### Evaluated with
<Capsule: probe-set identity, WHY chosen, preprocessing; judge model + rubric.>
<details open> 3–5 example probes </details>
Full probe bank: [pinned link](…)

### Generated
<Capsule: what the model produced, which conditions, N completions.>
Per load-bearing condition: 1 inline example (labeled cherry-picked/random)
+ <details> 3–5 more </details>
Full raw completions: [HF raw_completions tree pinned](…)

(Subsections that don't apply state it explicitly: "n/a — no training in
this task (eval-only)". Never silently omitted.)

## Reproducibility

(Unchanged: **Parameters** / **Artifacts** / **Compute** / **Code** /
**Context** rows. Artifacts keeps reuse-provenance bullets. Context keeps
created/run dates, lineage, verbatim origin prompt, and per-round
followup_labels.)
```

**Design decisions baked in:**

- **`## Takeaways` replaces Human TL;DR's skim function** in plain academic register (no lowercase-casual voice, no "How this updates me" diary framing). It is bullets, numbers-first — directly adaptable into a Slack post. The H1 title stays the one-sentence claim + confidence tag.
- **Flattening:** `### Motivation`/`### What I ran`/`### Findings` nesting under `## TL;DR` existed to coexist with Human TL;DR. With that gone, the H2s flatten: Motivation compresses to the `**Why:**` bullet (the #517 Motivation section was ~20 lines saying what 2 sentences could).
- **Findings keep one figure each** (Lens 9 unchanged) but the prose around it drops from paragraphs to bullets with a hard word cap.
- **Sample completions consolidate into `## Data → Generated`.** Today they're scattered per-finding (the bulk of #517's 320 lines is six 20-line `<details>` dropdowns). Findings keep at most one short excerpt where the text itself is the finding; the systematic per-condition samples + dropdowns live in one labeled place.
- **`## Data` is reader-facing samples + links; `docs/methodology/issue_N.md` stays the findings-blind deep reference.** Overlap is deliberate and cheap (both pull from the same artifacts); the Data section is what makes "what exactly did it train on?" answerable without leaving the body.

## 3b. The methodology document — v2 template (structured, complete, capped)

Every clean-result keeps shipping with the auto-generated, findings-blind methodology reference (`docs/methodology/issue_<N>.md` + secret gist mirror, linked at the top of the body and from `## Reproducibility` — mechanism unchanged). What changes is the OUTPUT SHAPE: today's docs are prose-paragraph-shaped; v2 is a fixed table-first skeleton with hard caps, so the doc is scannable in one screen-scroll and complete at the same time:

```markdown
# Methodology — issue <N>: <one-line what-was-run, no findings>

## 1. Overview            — 3–5 bullets: model, manipulation, design cells, DV, judge. No prose paragraphs.
## 2. Hyperparameters     — ONE complete table: EVERY training + eval + generation
                            hyperparameter (model, LoRA r/α/dropout, lr, schedule,
                            optimizer, epochs/steps, batch, seeds, max_new_tokens,
                            temperature, judge model + re-calls, N per cell, …),
                            each value copied from ground truth (committed config /
                            run_result.json / plan §11) with a Source column
                            (config path or plan-§11 ref). Nothing scattered in prose.
## 3. Training data       — construction recipe as a numbered list (≤8 steps);
                            row-count/composition table (rows per type, ratio,
                            personas, provenance tier); 2–3 VERBATIM example rows
                            (input → output, loss-mask noted).
## 4. Evaluation          — DV definition (construct + metric + on/off-policy, 2–3
                            bullets); probe-set table (N, source, why chosen,
                            preprocessing); 2–3 verbatim example probes; judge
                            prompt/rubric pointer.
## 5. Worked examples     — 2–3 verbatim end-to-end rows: eval input → model output
                            → judge score/measurement, one per load-bearing condition.
## 6. Artifacts index     — table: artifact → pinned link (training JSONL, adapters,
                            eval JSONs, raw completions, figure source, WandB run).
```

Caps (enforced by the methodology-writer's own checklist + spot-checked by the clean-result-critic's Data lens when it follows the link): no section may contain a prose paragraph >2 sentences; everything that can be a table or numbered list is one; target length ≤150 lines excluding verbatim example blocks. Stays findings-blind: no interpretation, no confidence, no results — unchanged.

**Two-tier split with the body (NeurIPS-checklist tiering):** the body's `## Reproducibility` Parameters table slims to the LOAD-BEARING subset (model, adapter recipe, lr, steps, seeds, eval rig, N) and the methodology doc §2 becomes the canonical COMPLETE table. The lr-matches-plan verifier check (16) keeps running against the body's table (via the §7 sentinel-gate generalization). The body-table ⊆ doc-table consistency assert CANNOT live in the methodology-writer (it's findings-blind, never reads body.md, and on the early-spawn path runs before the body exists) — it lands as new verifier check 21. **Gate-timing caveat:** the doc commits on the ISSUE WORKTREE branch and only reaches the repo-root `main` checkout at the Step 9b auto-merge, AFTER the critic gate — so at 9a-bis the doc is absent from the path a naive check resolves. The orchestrator therefore passes the worktree doc path explicitly (`--methodology-doc <path>`) to the verifier + critic at gate time; check 21 NO-OP-PASSes only when no doc exists anywhere yet, and binds fully at promote-time verify (post-merge). The clean-result-critic Data lens is the semantic backstop, reading the same passed path.

**EXTEND mode (same-issue follow-up rounds):** §2 stays ONE canonical table — a new round adds a per-round COLUMN (values shared across rounds span/repeat; changed values are what the column exists to show), never a second table. §3–§5 append a clearly-labeled `Round <label>` block per round. This keeps the "complete at a glance" property on multi-round issues.

## 4. Conciseness enforcement (mechanical, not aspirational)

The research is unambiguous that voluntary norms go unfilled — so caps are verifier checks, not style suggestions:

| Surface | Cap | Verifier behavior |
|---|---|---|
| `## Takeaways` | 3–6 bullets, no paragraphs | FAIL outside range |
| Per-bullet length (Takeaways) | ≤30 words | WARN |
| Per-finding prose (excl. caption/code/details) | ≤120 words | WARN at 120, FAIL at 180 |
| Figure caption | ≤60 words | WARN |
| Total prose: Takeaways + What I ran + Findings (excl. tables, code fences, details bodies, captions) | ≤700 words + 250 per live follow-up round beyond the first | WARN-only (the per-finding FAIL is the hard gate; a multi-round consolidated body must not be forced to delete live findings to satisfy a total cap — see §6.4 compression rule) |
| Paragraphs in Findings/What I ran | ≤2 sentences each; bullets preferred | critic lens (LM judgment) |

Numbers calibrate during Phase A by converting #517 to v3 (dry-run); they're encoded as named constants in `verify_task_body.py` so tightening later is a one-line change. Voice section of SPEC.md rewrites: bullets default; prose only where a causal chain needs ≤2 sentences; bold key numbers (NN/g layer-cake guidance); first person stays; `byte identical` ban and statistical-framing discipline carry over unchanged.

## 5. Data-section mechanics (new verifier checks)

- **Check 18 (v3):** `## Data` present with `### Trained on` / `### Evaluated with` / `### Generated` in order; each contains ≥1 pinned link to the complete artifact OR an explicit `n/a — <reason>` line.
- **Check 19 (v3):** every example block in `## Data` is immediately preceded by a subset-disclosure line — `"K of M rows, random sample"` or `"cherry-picked for illustration"` (the Datasheets sample-to-population field). Extends today's checks 10/11 (which only scan `## TL;DR`) to the Data section.
- **Checks 10/11 retarget:** scan `## Findings` + `## Data` in v3 bodies.
- **Harmful-content carve-out carries over verbatim.** Much of this project's training data is Betley-style EM / bad-medical-advice corpora; the SPEC.md § harmful-content + analyzer Content-hygiene rules (sanitized ~15-word excerpt + `[truncated — harmful-content row; verify at <path>, row <i>]` placeholder, labels and row indices verbatim) apply to `## Data` example blocks exactly as they do to finding sample blocks today, and checks 18/19 MUST accept the sanitized form. Agents assembling Data sections pull rows by grep + line offset per the context-hygiene rule — never page whole raw harmful-completion files into context.
- **Capsule trio for eval (critic lens, not verifier):** identity / why chosen / preprocessing must all be answerable from the Evaluated-with capsule.
- Composition facts that today hide in prose (positives:negatives ratio, persona panel, row counts per type) are REQUIRED capsule content.
- Optional extension (not in scope unless wanted): mirror the three full-artifact links into frontmatter (`data_trained:`, `data_eval:`, `data_generated:`) so the dashboard can render a Data strip on the task card.

## 6. Follow-up consolidation

Current state: a same-issue follow-up loop already exists (`followups_running`, `epm:followup-scope v1`, fold-into-body semantics), but (a) the `question_relation` criteria let too much route to `substantially-different` → child issues, and (b) there's no explicit cross-round synthesis — new findings fold in, but no section answers "so what's the final takeaway after all N rounds?".

Changes:

1. **Bias routing hard toward same-issue.** Rewrite the `question_relation` criteria in `follow-up-proposer.md` (and the CLAUDE.md routing rule) around one litmus: **"Would the result rewrite THIS issue's `## Takeaways`?" → `same` (stay on the issue).** Changing method, dose, panel, seeds, eval surface, prompt bank, or adding a control/baseline on the same question is ALWAYS `same`. `substantially-different` is reserved for work that would change the task's `## Goal` / open-questions anchor — a genuinely new question. Add 3 worked examples of each to the proposer spec (e.g., #517's "re-run trained adapters on the matched Q-bank" should have been a same-issue round, not a candidate child).
2. **`## Takeaways` is the rolling synthesis.** Analyzer re-entry rules (analyzer.md § same-issue follow-up fold-in) gain a MUST: after every round, rewrite `## Takeaways` to the current cross-round belief and retitle the H1 if the headline moved. A Takeaways section that only describes round 1 after round 2 landed is a critic FAIL (new lens item).
3. **Round visibility.** `## What I ran` gains the Rounds table (round label, date, what changed, one-line result) when >1 round; `**Context:**` keeps per-round followup_labels + verbatim prompts (already specced).
4. **Superseded-finding + round-compression hygiene.** When a round invalidates an earlier finding, the analyzer rewrites Findings to the current best understanding and collapses the outdated block into one `<details>Superseded</details>` at the end — audit trail without bloat. When a round's synthesis ABSORBS an earlier finding (still true, no longer load-bearing on its own), that finding compresses to heading + figure + ≤2 bullets. This is how round-N bodies stay near the word budget without deleting live findings (the total-prose cap in §4 is WARN-only and scales per round for the same reason).
5. **Follow-up rounds on existing v2 bodies: migrate-on-fold.** A same-issue follow-up round that lands on a v2-sentinel body AFTER cutover migrates that body to v3 as part of the fold (the analyzer is rewriting the body anyway, and drafts rebuild cheaply from cached results + figures). This is the ONE deliberate exception to "parked bodies stay v2" (§8) — without it, the rewritten analyzer fold-in rules and remapped critic lenses would produce a hybrid body or a critic bounce on exactly the bodies forward-only protects. No dual v2/v3 fold-in branch is maintained.
6. **Unchanged:** the `followups_running` hold, `followup-auto`/`followup-manual` tags, artifacts under `eval_results/issue_<N>/<followup_label>/`, re-park at `awaiting_promotion`. Genuinely-new-question children still exist (they're correct for new directions); the bar just moves.

## 7. File-by-file change list

Phase A — the contract (do first, everything else follows it):

| File | Change |
|---|---|
| `.claude/skills/clean-results/SPEC.md` | v3 section shape (§3 above), v3 sentinel, conciseness caps, Data-section spec, voice rewrite, follow-up synthesis rules; v2 section retained as grandfathered-shape documentation |
| `scripts/verify_task_body.py` | v3 sentinel gating; v3 `REQUIRED_H2_SECTIONS` (Takeaways/What I ran/Findings/Data/Reproducibility); `## Human TL;DR` present in a v3 body = FAIL (mirrors stray-`## Details`). **(a) Shared sentinel-gate generalization:** checks 6 / 16 / 17 all branch on `is_v2_nested_design()` (v2 sentinel only) — a v3 body would be treated as LEGACY: check 6 hard-FAILs ("no `Confidence:` line found"), and 16 (lr-matches-plan, the #489 50× misprint guard) + 17 (Context row) silently skip. Generalize to `is_nested_design()` accepting either sentinel so all three keep running on v3 (6 stays title-tag-only; 16 runs against the slimmed body table; 17 unchanged). **Apply at exactly these three call sites — NOT a global rename: the fourth `is_v2_nested_design` call site (check 3b) must stay v2-only, or v3 bodies would FAIL the v2 nested-TL;DR shape check.** **(b) TL;DR-scoped checks need a v3 branch** (they resolve `section_text(body, "TL;DR")`): check 3+3b → replaced by a v3-structure check (Takeaways 3–6 bullets; What I ran slots incl. `**Why:**`; ≥1 `### ` finding under `## Findings`); checks 4/4b (figure presence + URL existence) → retarget to `## Findings`; check 10 (cherry-picked label) → retarget to `## Findings` + `## Data`; check 11 (raw-completions link) → retarget to `## Findings` + `## Data → ### Generated` ONLY (Trained-on/Evaluated-with blocks link JSONLs/probe banks, not raw_completions — check 18 covers their links); check 11b (planned-vs-actual denominator — currently vacuous-PASSes without `## TL;DR`, silently retiring the scope-shrinkage guard) → headline surface retargets to `## Takeaways` + `## Findings`; check 13 (narrative-flow WARN) → retuned for v3 section names; **`check_concerns_audit`** (dispatched outside `CHECKS`; acknowledgment mechanism 1 scans H3s inside `## TL;DR`, mechanism 2 scans the `Confidence:` paragraph — BOTH surfaces are gone in v3, so an open binding concern acknowledged in finding prose would spuriously hard-FAIL) → mechanism 1 retargets to `### ` finding sections under `## Findings` + `## Takeaways` bullets; mechanism 2 retires for v3 (no Confidence paragraph exists). New checks 18/19/20 (Data shape, subset disclosure, word caps) + new check 21 (body Parameters table ⊆ methodology-doc §2 table, see §3b). Unchanged: 0/0b/1/7/8/8b/9/14/15. v2 + legacy bodies keep current behavior verbatim |
| `tests/test_verify_task_body.py` | new v3 fixtures (good body, each new-check failure mode, v2-grandfathering regression tests) |
| `.claude/skills/clean-results/exemplars/` | convert #517's body to v3 as the canonical exemplar (also calibrates the word caps) |

Phase B — generators + critics:

| File | Change |
|---|---|
| `.claude/agents/analyzer.md` | Step 1 template → v3 skeleton; drop Human TL;DR drafting entirely — **including the hardcoded Step 6 bash asserts (`grep -qE '^## Human TL;DR$' … || exit 1`), which would hard-crash every v3 promotion if only the template/voice sections were rewritten**; bullet-register instructions; Data-section assembly (pull rows from training JSONL / eval JSON / raw_completions, sanitized form for harmful corpora); Step 4.5 inline humanize self-pass retargets from the `## TL;DR` block to the v3 prose surfaces (Takeaways + What I ran + Findings bullets); follow-up fold-in rules per §6.2–6.5 (incl. migrate-on-fold for v2 bodies) |
| `.claude/agents/clean-result-critic.md` | complete lens remap (old → v3): L1 Title — unchanged; L2 TL;DR-structure → v3-structure lens (Takeaways shape, What I ran slots, Findings skeleton); L3 Figure — unchanged + ABSORBS L12's setup/read–figure pairing check (now as bullets); L5 Reproducibility — unchanged + slimmed-Parameters rule; L6 Voice — rewritten for bullet register; L7 statistical-framing, L8 mentor-facing title, L9 one-takeaway-one-figure, L11 raw-alongside-processed, L13 planned-vs-actual, L14 binding-concerns, L15 contaminated-arm — carry over with section-name updates; L10 eval-probe descriptions → folds into the new Data lens; L12 story-arc — retired for v3 (pairing check moved to L3). NEW lenses: Takeaways quality (register, numbers-first, cross-round synthesis currency — a Takeaways describing only round 1 after round 2 landed = FAIL), Conciseness (cap adherence + bullets-over-prose), Data (capsule trio, subset disclosure, link liveness, methodology-doc spot-check) |
| `.claude/agents/codex-clean-result-critic.md` | mirror the lens remap |
| `.claude/agents/interpretation-critic.md` | drop Human-TL;DR-placeholder references; section-name updates |
| `.claude/agents/methodology-writer.md` | v2 output template per §3b (table-first skeleton, complete hyperparameter table with Source column, caps, body-table ⊆ doc-table assert); findings-blind list updates: never reads `## Takeaways` / `## Findings` (replaces the TL;DR names) |
| `.claude/skills/issue/SKILL.md` | **retarget (not retire) Step 9a-humanize**: the pass currently de-AIs the `## TL;DR` narrative block (NOT Human TL;DR — original premise corrected after review); under v3 it targets Takeaways + What I ran + Findings prose, which is exactly what Thomas adapts for Slack. Expect it cheaper (bullets, ~700 words). **Step 9a-quater methodology-link insertion anchors**: the block hardcodes "insert after the `<!-- clean-result-v2 -->` sentinel … BEFORE `## Human TL;DR`" — neither string exists in a v3 body; retarget to the v3 sentinel / `## Takeaways`. Step 9b/10b text updates |
| `.claude/workflow.yaml` | `9a-humanize` step def + `epm:humanize-loop` marker `fields` text (currently says "for the TL;DR block") + Step 9a-ter `entry_condition` — all updated to the retargeted pass; section-name mentions swept |
| `scripts/recent_clean_results.py` | `RE_MD_TLDR` (`^## TL;DR`) → recognize v3 sections; **exemplar feed must prefer v3 bodies post-cutover** (otherwise the analyzer's few-shot exemplar feed stays all-v2 and drifts new drafts back toward the old shape — a real regression vector, not cosmetic) |
| `docs/methodology/` exemplar | convert one existing on-disk doc (e.g. `issue_612.md` or `issue_601.md` — note `issue_489.md`/`issue_514.md` do NOT exist on disk; those task numbers are exemplars of the mechanism, not files) to the §3b v2 template as the canonical exemplar the methodology-writer spec points at |
| `.claude/skills/issue/markers.md` | duplicates the humanize-marker + 15-lens v2 descriptions — sync with the retargeted pass + lens remap |
| `.claude/skills/campaign/SKILL.md` | campaign ingestion extracts "`## TL;DR` findings ONLY" — a live execution path that would look for a missing section in v3 children; retarget to `## Takeaways` + `## Findings` |
| `.claude/skills/mentor-update-slides/principles.md` | v2 source-data map references `## TL;DR` — update |

Phase C — follow-up routing:

| File | Change |
|---|---|
| `.claude/agents/follow-up-proposer.md` | `question_relation` litmus + 6 worked examples per §6.1 |
| `CLAUDE.md` § Routing experiment intent | sharpen the same-question test with the Takeaways litmus |
| `.claude/skills/issue/SKILL.md` § Step 9b | routing default text update |

Phase D — sync + hygiene:

| File | Change |
|---|---|
| `CLAUDE.md` § Experiment Report Structure | rewrite summary to v3 |
| `CLAUDE.md` § After Every Experiment item 8 | still instructs "name the missing condition in the TL;DR Motivation … across Motivation, every relevant result H3" — stale v2-shape wording on an always-loaded surface that drives check 11b / Lens 13; retarget to Takeaways/Findings |
| `.claude/agent-memory/**` audit | stale v2-shape memories steer the very agents being retargeted (e.g. `analyzer/feedback_clean_result_critic_v1_checklist.md`, `analyzer/feedback_details_dropdown_fences_need_own_prelude.md`, `codex-clean-result-critic/project_followup_regate_composition.md`, `reconciler/feedback_claude_clean_result_critic_underapplies_spec_text.md`) — update or retire each; agent-memory is named workflow surface, do NOT grandfather as "history" |
| `.claude/rules/agents-vs-skills.md` ontology table | critic-lens descriptions update |
| `.claude/rules/research-project-structure.md` | artifacts-table row still says "Human TL;DR + AI TL;DR + AI Summary" — update |
| `.claude/agents/living-docs-updater.md` | section-name reference update |
| `.claude/skills/promote-clean-result/SKILL.md` | section-name references |
| `scripts/audit_clean_results_body_discipline.py` | **v3 gating fix (bulk-inventory mode only):** the should-audit gate (v2 sentinel / Human-TL;DR+TL;DR+Reproducibility trio / legacy AI-TL;DR markers) is consulted only on the bulk-inventory path — the live pipeline paths (`--task <N>` at Step 9a-bis, file-path for drafts) audit unconditionally and are NOT affected. Still add `<!-- clean-result-v3 -->` to the gate so bulk audits don't skip v3 bodies, + a regression test scoped to bulk mode. **Verbatim-content exemption extends to `## Data` example blocks:** v3 MANDATES verbatim training rows/probes in `<details>` blocks, which the audit scans (it strips only fenced code + Context blockquotes) — verbatim rows containing strings like `C1`/`H2` would trip the condition-code patterns with no reword option (same conflict the Context-blockquote carve-out fixed, incident #597). Exempt example blocks inside `## Data` (or spec them as fenced blocks, which the audit already strips) |
| `tests/test_task_workflow.py`, `tests/test_recent_clean_results.py`, `tests/test_clean_result_critic_planned_vs_actual.py` | Human-TL;DR / `## TL;DR` fixtures gain v3 variants; v2 fixtures stay (grandfathering regression coverage) |
| Assistant memory | retire `feedback_human_tldr_human_authored.md` ("Human TL;DR is Thomas-authored") + the MEMORY.md index line; add a v3 pointer note |

Pre-declared grandfathered (sweep hits that are correct as-is, do NOT churn): deprecated `reviewer.md` / `codex-reviewer.md`, legacy `verify_sagan_card.py`, legacy `scripts/verify_clean_result.py` + `tests/test_verify_clean_result.py` + `tests/test_narrative_consolidation.py` (GH-issue-era validator, referenced only by deprecated surfaces), SPEC.md's v2/legacy sections, historical task bodies under `tasks/`, `iterations.md` history.

Grep sweep at the end of each phase: `grep -rniE "human tl;dr|## tl;dr|clean-result-v2|humanize|### motivation|### what i ran" --exclude-dir=worktrees --exclude-dir=cache .claude/ scripts/ tests/ CLAUDE.md` — every hit either updated or on the grandfathered list. One extra verification line: confirm the eps dashboard's task detail view renders body.md generically (no `## Human TL;DR`/`## TL;DR` section-specific extraction) — the dashboard lives outside this repo, so the sweep can't see it.

## 8. Migration policy

Identical to the v1→v2 precedent (it worked):

- **Forward-only.** New bodies emit `<!-- clean-result-v3 -->`; the verifier gates v3 rules on the sentinel. v2-sentinel bodies and pre-sentinel legacy bodies keep their current verification behavior verbatim and are NEVER newly hard-FAILed. One deliberate exception: a same-issue follow-up round on a v2 body migrates it to v3 as part of the fold (§6.5).
- **In-flight drafts:** anything not yet through clean-result-critic at cutover re-drafts under v3 (drafts rebuild cheaply from cached results + figures). The ~30 `awaiting_promotion` parked bodies stay v2.
- **Optional backlog conversion** (separate decision, not part of this plan): a batch pass converting parked v2 bodies to v3 before promotion. Recommend NO for already-parked bodies — review them as-is.

## 9. Execution plan

1. Thomas approves/edits this plan (it's an architectural/public-contract change — greenlight required by the workflow-fix protocol).
2. Phase A in one worktree: SPEC.md + verifier + tests + #517 exemplar conversion. Implementer + independent code-reviewer; `uv run pytest tests/test_verify_task_body.py` green; calibrate caps on the exemplar.
3. Phases B + C in parallel worktrees (disjoint files), same implementer/code-reviewer pairing; `workflow_lint.py` green.
4. Phase D sweep + final grep audit; commit/merge/push per the standing workflow-surface rule.
5. First real experiment after cutover gets a deliberate extra-careful clean-result-critic round; capture friction in `iterations.md` and adjust caps once.

Estimated effort: Phase A is the bulk (~1 session); B–D are mechanical edits riding on it.

## 10. Open decisions (recommendations inline)

1. **Section names.** `## Takeaways` (recommended — matches how Thomas talks about it) vs `## Bottom line` vs keeping `## TL;DR`.
2. **Word-cap numbers** (§4 table). Recommended as stated; calibrated on the #517 conversion; encoded as constants so tightening is trivial.
3. **Per-finding excerpt.** Keep ONE ≤10-line excerpt inside text-behavior findings (recommended — the reader shouldn't have to jump to Data to see the behavior) vs ALL samples in Data only.
4. **Frontmatter data links** (§5 optional extension) — only if dashboard surfacing is wanted.
5. **Backlog conversion** (§8) — recommend no.

## 11. Sources (verified primaries)

Ernst, *Writing a weekly progress report* (UW) · Hartley & Sydes 1997 (J. Research in Reading) + Hartley 2004 (J Med Libr Assoc) · NN/g *F-Shaped Pattern* (2006/2017) · Gebru et al., *Datasheets for Datasets* (arXiv:1803.09010) · Mitchell et al., *Model Cards* (arXiv:1810.03993) · Bender & Friedman, *Data Statements* (TACL 2018) · HF dataset-card docs + template (live-fetched 2026-06-12) · Yang, Liang, Zou (ICLR 2024, arXiv:2401.13822) · NeurIPS Paper Checklist · W&B Tables docs. One refuted claim (Model Cards "fixed nine-section skeleton", 0-3) — deliberately not used above.
