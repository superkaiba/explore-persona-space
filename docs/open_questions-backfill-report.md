# Open-questions evidence back-fill report (proposal)

Generated 2026-05-28. Audit surface for the proposed `docs/open_questions.proposed.md`.
This is a **proposal only** — the live `docs/open_questions.md` was NOT touched.

## What this is

For every clean-result (`has_clean_result=true`) in the task workflow, I inferred from its
**title alone** which open-question(s) it bears on, and added its `#id` to those questions'
`> **State:** ... evidence:` trailers in the proposed copy. Mappings are **conservative**:
a result only maps when its title clearly speaks to the question. Audit each row below before
confirming.

Source enumeration:
- `task.py list-by-status --status completed --json` → clean-results: #120, #138, #390, #391
- `task.py list-by-status --status awaiting_promotion --json` → 47 clean-results

Question-id legend: `h1-h3` headline; `a1-a5` Thread A; `b1-b8` Thread B; `c1-c6` Thread C;
`d1-d3` Thread D; `e1-e5` Thread E; `app1-app5`, `app-trigger-discovery` Applications.
Ids `a5, b7, b8, c6, e5` are **newly proposed questions** (see bottom).

## Mapping table

| task # | title (truncated) | → question id(s) | confidence signal |
|---|---|---|---|
| #61 | Fine-tuning assistant toward source persona emits source's `[ZLT]` marker for 4/7 sources — base cosine doesn't predict which | a1, app1, b7 | LOW |
| #65 | Training one persona to emit `[ZLT]` without bystanders has a one-cell-wide LR×epochs window | app1 | LOW |
| #75 | Coupling evil personas w/ wrong answers fails to protect Qwen from EM collapse; capability ordering is eval contamination | c1, c2, app3, e3 | LOW |
| #105 | Apparent assistant-persona robustness under contrastive wrong-answer SFT was a data-mixing artifact (ARC-C 84%→1.9%) | c1, e1, e3, app3 | HIGH |
| #113 | Wrong-answer FT under Qwen's own default sys-prompt self-degrades harder; "I am" framing recovers cross-model identity gap | (see unmapped — system-prompt-as-persona-slot; closest a-thread but no clean home) | MODERATE |
| #116 | Persona-mimicry SFT stage before behavioral SFT amplifies source→assistant transfer of alignment/refusal/sycophancy for 6/8 | h2, d1, d2, d3 | LOW |
| #120 | Why Qwen identity prompt vs generic assistant leak to different bystander neighborhoods | (see unmapped — bystander-neighborhood routing; closest a-thread, no clean home) | (completed) |
| #123 | Qwen default identity prompt is a distinct persona slot (5× more vulnerable); refusal LoRA leaks to named AI assistants | (see unmapped — same system-prompt-slot cluster as #113/#120/#101/#108) | MODERATE |
| #138 | Both system-prompt persona and answer-content persona elicit `[ZLT]` ~equally; together >3× the rate | b2, a1 | MODERATE |
| #182 | Persona-CoT REVERSES ARC-C asst-aligned advantage; truncation × tag-injection dominant suspect | (see unmapped — persona-CoT capability; closest #186 cluster) | LOW |
| #186 | Persona-flavored CoT rationales drive cross-persona behavior leakage in wrong-answer SFT; style dominates | d2 | MODERATE |
| #187 | Chat-template Betley eval on Gemma2-2b base-LM finetune; dialogue collapse unidentifiable from template mismatch | a4 | MODERATE |
| #192 | Fact teaching transferred to non-teach assistant frames across 2 seeds under either teach prompt | b2, b4 | MODERATE |
| #207 | Persona-geometry distance predicts where a marker leaks — six experiments, \|rho\| 0.48-0.79 | a1, b1, b2, b6, e1, e4, app4, app5 | MODERATE |
| #215 | Only continuous soft prefixes hit both EM axes at once; discrete prompt searches split | (see unmapped — soft-prefix/prompt-search; no EM-elicitation question hosts it cleanly) | MODERATE |
| #224 | `[ZLT]` emission is not a trained attention pattern or learned direction; random direction elicits ≥ trained centroid | a1, c4, c5 | LOW |
| #225 | Marker is a representational handle, not behavioural — sharing it between villain & assistant transfers no misalignment | b2, e4, app2 | HIGH |
| #234 | Betley's edu_v0 cue is a base-model jailbreak; conditional-misalignment surface is security/authority/educational triad | c3 | MODERATE |
| #235 | Language-mismatch LoRA SFT leaks trained completion language into untrained bystander directives | b8 | LOW |
| #237 | ANY SFT (LoRA/full, EM/benign) collapses Qwen persona geometry to cos ≥0.97 | a3 | MODERATE |
| #276 | Pretraining-data-poisoned Qwen3-4B backdoor fires only on exact trigger; paraphrases don't; base-similarity doesn't predict | app-trigger-discovery | MODERATE |
| #311 | Cosine distance to paramedic↔comedian midpoint marginally predicts joint-source `[ZLT]` leakage | a1, b6, app5 | LOW |
| #333 | Three-seed FR↔IT bystander spill flips sign with seed | e1, b8 | MODERATE |
| #337 | Longer persona system prompts pull a `[ZLT]` marker toward the source (stronger source, less bystander) on N=48 panel | h3 | MODERATE |
| #351 | Evolutionary search fails to recover Gaperon-1125-1B's Latin trigger | app-trigger-discovery | LOW |
| #354 | EOS-in-loss was the confound: masking recipient EOS revives within-marker chunk-binding 1.3%→23.5% | e2 | MODERATE |
| #355 | Persona-style rationale does not reduce answer uncertainty below generic after answer-cue filtering | (see unmapped — persona-CoT uncertainty; #182/#186/#356 cluster) | HIGH |
| #356 | Audit-filtering did not amplify persona-CoT leakage overall; software_engineer partial positive on both axes | (see unmapped — persona-CoT cluster) | LOW |
| #358 | Backdoor-trigger filepaths linearly separable from controls at layer 18 even before poisoning | e5 | LOW |
| #360 | Teacher-forced target log-prob does not detect non-anth paraphrase lift above controls | app-trigger-discovery | LOW |
| #363 | Chen & centroid persona vectors land in same neighborhood at preferred layer but not the same direction | a2, a5 | MODERATE |
| #365 | Every recipe factor that lifts `[ZLT]` source rate also lifts bystander leakage; no factor implants selectively | h3, a1, b7 | LOW |
| #366 | Cross-persona chunk binding leaks the first hop beyond the donor; recipient cascades stop there | e4, app4 | HIGH |
| #368 | Persona-vector recipes unreliable cross-persona predictors; bare centroids beat Chen mean-diff; prior effects fail controls | h1, a1, a2, a5, app5 | HIGH |
| #369 | Donor trained on marker-B alone still propagates ~8% recipient marker-B; paired-marker advantage doesn't separate | app4 | LOW |
| #370 | `depuis qui est` fires 83% French switching, 49pp above #351's strongest neighbor | app-trigger-discovery, b8 | MODERATE |
| #375 | Persona-voiced few-shot context elevates `[ZLT]` rate on all 3 trained personas; most lift from one demo | b4 | MODERATE |
| #376 | Persona-and-trigger conditional marker did not survive one epoch of length-matched SFT (EM or not) | e2, app1 | HIGH |
| #377 | Every tested multi-turn prior history silences the conditional marker equally; no drift-specific displacement | e2, app1 | HIGH |
| #378 | In-context `|AUDIT|` trigger fires 0/600 against weight-baked hidden behaviors in 3 Qwen3-14B organisms | e5 | LOW |
| #380 | Output-distribution distance from assistant baseline does not predict `[ZLT]` source rate on 48-persona panel | h1, a1, b2, b3, app5 | MODERATE |
| #381 | Contrastive negatives let Qwen give different answers per persona; reducing training alone doesn't separate teach | b1, c6 | MODERATE |
| #382 | Self-distillation KL-anchor installs Assistant-keyed marker at 98.4% in Phase 1 but erased by 1 epoch benign SFT | e2, app1 | HIGH |
| #383 | Different training-recipe params can make `[ZLT]` implantation both stronger AND more selective | h3 | MODERATE |
| #389 | Contrastive SFT gates trained-answer emission but impairs in-context rule application | b1, c6 | MODERATE |
| #390 | Refusal-style negatives install a persona gate generalising across 9/11 OOD framings as verbatim refusal substitution | c6, h2, d3, e3 | MODERATE |
| #391 | Could not selectively implant sycophancy: training lifts source + 23 bystanders in lockstep; 4 design confounds | h2, d3 | LOW |
| #396 | Four geometric/gradient predictors of LoRA marker emission all return null at N=24 across six DV surfaces | h1, a1, app5 | MODERATE |
| #398 | Step-75 marker firing-rate cliff is a sampling-threshold artifact; librarian source never leads — global affinity shift | b7, e1 | MODERATE |
| #399 | Behaviorally-silenced single-token marker leaves context-uniform log-prob fingerprint, not a trigger-gated install | b7, b5, e2, app1 | HIGH |
| #411 | Three of six sources replicate #99's sycophancy cosine gradient on held-out wrong claims; two sign-flip, one collapses | h2, d1, d3 | LOW |

## Unmapped clean-results (no confident question fit)

These have clean-results but no current question (and no newly-proposed question) cleanly hosts
them. They cluster around **system-prompt-as-persona-slot** and **persona-CoT capability** — two
coherent sub-threads the doc does not currently carry. I did NOT auto-create questions for these
because the existing threads do not obviously want them and the right framing is a judgment call
for you. Listed here so the gap is visible.

| task # | title (truncated) | why unmapped / candidate sub-thread |
|---|---|---|
| #113 | Wrong-answer FT under Qwen default sys-prompt self-degrades harder; "I am" framing recovers cross-model identity | system-prompt-as-persona-slot — could seed a new A-thread question (with #120, #123, #101, #108) |
| #120 | Why Qwen identity prompt vs generic assistant leak to different bystander neighborhoods | system-prompt-as-persona-slot |
| #123 | Qwen default identity prompt is a distinct persona slot (5× more vulnerable); refusal LoRA leaks to named AI assistants | system-prompt-as-persona-slot — strongest single result in this cluster |
| #182 | Persona-CoT REVERSES ARC-C asst-aligned advantage; truncation × tag-injection dominant suspect | persona-CoT-capability sub-thread |
| #355 | Persona-style rationale does not reduce answer uncertainty below generic after answer-cue filtering | persona-CoT-capability sub-thread |
| #356 | Audit-filtering did not amplify persona-CoT leakage; software_engineer partial positive | persona-CoT-capability sub-thread |
| #215 | Only continuous soft prefixes hit both EM axes; discrete prompt searches split | EM-elicitation / soft-prefix — relates to #90/#94/#104/#170 prompt-search line, none of which is a current question |

> Recommendation: if you want these covered, the two cleanest new questions would be
> **"Is Qwen's default identity system prompt a privileged persona slot, and does it re-route
> which personas absorb a trained trait?"** (hosts #123, #120, #113, #101, #108) and
> **"Does persona-flavored chain-of-thought change capability/uncertainty, or only style?"**
> (hosts #182, #355, #356, #186). I left them OUT of the proposed file to avoid over-creating;
> say the word and I'll add them in the matching thread format.

## Newly-proposed questions (FLAGGED in the proposed file)

Each appears under its thread in `open_questions.proposed.md` with a `<!-- q:... -->` anchor and a
State trailer, clearly marked under a "Proposed new questions (FLAGGED)" section. Reject any you
don't want.

| new id | thread | one-line | backed by | proposed State |
|---|---|---|---|---|
| a5 | A | Do persona-vector *recipes* agree with each other & survive their own controls? | #368, #201, #363 | 🌳 HIGH |
| b7 | B | What single-token vs multi-token marker shape is implantable, and where does it leak? | #399, #398, #61, #365 | 🌿 MODERATE |
| b8 | B | Does language-mismatch / translation SFT leak the trained language as portable behavior? | #235, #190, #370, #333 | 🌿 MODERATE |
| c6 | C | Does contrastive-negative training install a persona *gate* on behavior emission? | #390, #389, #381 | 🌳 MODERATE |
| e5 | E | Are weight-baked hidden behaviors detectable by in-context audit triggers? | #378, #358 | 🌿 LOW |

Note on #190 / #162 cited in b8's prose: these are NOT in the clean-result set (no
`has_clean_result=true`), so they appear in b8's narrative but NOT in any evidence trailer.

## Counts

- Questions given anchors + State trailers: **31** (3 headline + 4 Thread A original + 6 Thread B original + 5 Thread C + 3 Thread D + 4 Thread E + 6 Applications) + **5 newly-proposed** = **36 anchored questions total**.
- Clean-results enumerated: **51** (4 completed + 47 awaiting_promotion).
- Clean-results mapped to ≥1 question: **44**.
- Clean-results unmapped: **7** (#113, #120, #123, #182, #215, #355, #356).
- New questions proposed: **5** (a5, b7, b8, c6, e5).
- Live `docs/open_questions.md`: **NOT touched.**
