---
goal: 'On Qwen-2.5-7B-Instruct, using a maximally-controlled single-turn minimal-pair
  battery (one token-matched instruction or query manipulation per pair, no crossing)
  across instruction axes (persona, format, lexical-marker, stance, content-constraint,
  register, hedging, single-token-user-fact-with-express-instruction, diffuse-user-profile-with-be-aware-instruction)
  and query axes (content, form), measure how faithfully the frozen #779 context-end
  ridge map predicts the DIFFERENCE between paired answer representations — per axis:
  direction, magnitude calibration vs global shrinkage, real-vs-predicted axis-identity
  cosine, and split-half reliability — against an identity+bias delta baseline, a
  paraphrase null, and a Qwen3-Embedding-8B answer-text third space, to determine
  which kinds of context information the map carries faithfully vs distorts.'
title: 'Controlled minimal-pair battery: does the context→answer map predict the DIFFERENCE
  in answer vectors, per instruction axis?'
---
# Controlled minimal-pair battery: does the context→answer map predict the DIFFERENCE in answer vectors, per instruction axis?

## Provenance

Originating design (interactive session with Thomas, 2026-08-24), decision record — every choice below is `user-answer`, `inherited from #2215/#2162`, or `default-confirmed` via the /clarify gate:

- **Motivation.** Grew out of #2215's separation-comparison round, which found the map REDISTRIBUTES answer-vector separation (exaggerates instructed-policy aspects, compresses restated-content aspects) but measured it with a yardstick-normalized MAGNITUDE ratio on banks that entangle the manipulated aspect with its textual realization (e.g. prior_topic swapped ~760 chars; birthday-vs-hiking co-varied topic with register). Thomas wanted (a) a maximally-controlled battery where the only difference between two contexts is one aspect, and (b) a measure that compares the observed vs predicted answer-vector DIFFERENCE directly rather than two separately-normalized magnitudes.
- **Routing:** child of #2215 (`user-answer`). Own Goal (below); reuses the #2162/#2215 rig + the frozen #779 map read-only.
- **Model:** Qwen/Qwen2.5-7B-Instruct, bf16, frozen (`inherited #2215/#2162` — the maps are fit for this model).
- **Data:** hand-authored controlled minimal-pair bank, agent-generated queries (`user-answer`: "it's fine for you to generate them"). Deliberately tier-4 constructed — the control requirement IS the justification (flagged as a data-realism deviation in the plan; the whole point is token-matched minimal pairs, which real corpora cannot supply).
- **Judge manipulation check:** included, light Sonnet Batch-API check that each instruction actually fired in the rollouts (`user-answer`).
- **Text-embedding third space:** included, Qwen3-Embedding-8B (`user-answer`; extends #2215's sepcmp round).
- **Scope OUT:** refusal (can't be made a minimal pair — harmfulness is inseparable from the swapped token), topic-of-prior-conversation, user-goal, task-identity, register/tone/affect as separate cells (register kept as one axis; tone/affect collapse into it), axis crossing, nonlinear maps/probes.
- **Timing:** filed `proposed` only; not spawned. Executes via `/issue <N>` when Thomas dispatches.

## Goal

On Qwen-2.5-7B-Instruct, using a maximally-controlled single-turn minimal-pair battery (one token-matched instruction or query manipulation per pair, no crossing) across instruction axes (persona, format, lexical-marker, stance, content-constraint, register, hedging, single-token-user-fact-with-express-instruction, diffuse-user-profile-with-be-aware-instruction) and query axes (content, form), measure how faithfully the frozen #779 context-end ridge map predicts the DIFFERENCE between paired answer representations — per axis: direction, magnitude calibration vs global shrinkage, real-vs-predicted axis-identity cosine, and split-half reliability — against an identity+bias delta baseline, a paraphrase null, and a Qwen3-Embedding-8B answer-text third space, to determine which kinds of context information the map carries faithfully vs distorts.

## Design

**Single-turn skeleton, one manipulation per pair (no crossing):**
```
[system]  〈one special instruction, OR empty〉
[user]    〈query〉
```
Every minimal pair is byte-identical except one slot; the non-varied slots hold a fixed neutral default (empty system; a fixed default query when a system axis varies), identical on both sides. Value phrases are token-matched within and across axes (verified against the Qwen tokenizer during design). The default system carries NO "assistant" wording — that word appears only inside an actual assistant persona.

**Instruction axes (system slot), each with an EMPTY level → install (empty→value) AND swap (value↔value) pairs:**

| Axis | n values | example values (final set fixed at implementation) |
|---|---|---|
| persona | 5 | gruff pirate captain · prim Victorian butler · wise zen meditation teacher · hyper-energetic startup founder · hardboiled noir detective |
| format | 5 | bullet points · numbered steps · single JSON object · rhyming poem · one flowing paragraph |
| lexical marker | 5 | include the word "moreover" · end with a party emoji · begin with "Well," · include the word "actually" · end with a question |
| stance | 5 | argue in favor · argue against · stay strictly neutral · steelman both sides · play devil's advocate |
| content-constraint | 5 | give exactly three reasons · never mention numbers · never use the word "I" · mention a real example · answer in under 20 words |
| register | 2 | very formal & professional · very casual & conversational |
| hedging | 2 | answer with strong confidence · answer with heavy caveats & hedging |
| user-fact-express | 5 | a SINGLE-TOKEN user fact + instruction to EXPRESS it: `The user's name is 〈Marcus〉. Always address them by name.` — swap the single-token value only (implementer verifies each value is exactly one Qwen token — e.g. names like Marcus, Diego, Sarah, or place words like Boston, Denver; two-token names such as Priya or Kenji are excluded). Local, marker-like content — does the map carry a single injected user token into the answer? |
| user-profile-aware | 5 | DIFFUSE user info + instruction to BE AWARE of it (not restate): `The user is 〈a busy single parent on a tight budget who wants quick practical help〉. Keep this in mind when answering.` — swap the ~15-18-token profile, token-matched. Diffuse content — does the map carry diffuse user conditioning the model is told to be aware of but not echo? |

**Query axes (user slot):**
- **query-content** — swap WHAT is asked across the 12 carrier items (system empty). The reference-ceiling content axis.
- **query-form** (`user-answer` addition) — same content, swap the FORM: question / imperative / declarative statement. 3 values. Tests whether the answer representation cares about interrogative surface form independent of content.

**Carrier set — 12 debatable, advice-type items** (agent-generated), each chosen so EVERY instruction applies to EVERY carrier (admits a stance, answers with reasons, needs no numbers). Rendered in question form by default; each also has imperative + statement forms for the query-form axis. Draft set (finalized at implementation):

1. Should I adopt a dog or a cat?
2. Is it better to rent or to buy a home?
3. How should someone spend a free weekend?
4. Should students be required to learn coding?
5. Is remote work better than working in an office?
6. What's the best way to make new friends?
7. Should I read more fiction or nonfiction?
8. Is it worth traveling somewhere alone?
9. How should a person choose a career?
10. Is it better to save money or to spend it?
11. Should someone follow a passion or a stable job?
12. Is it better to exercise in the morning or evening?

Non-question forms (query-form axis), worked examples:
- dog/cat → imperative "Help me decide between a dog and a cat." · statement "I'm torn between adopting a dog and a cat."
- rent/buy → imperative "Help me choose between renting and buying." · statement "I can't decide whether to rent or buy."

The two user-information axes form a matched contrast — LOCAL+EXPRESS (single token, told to surface it) vs DIFFUSE+AWARE (profile, told to condition on it without restating) — a granularity × instruction-strength diagonal that directly probes whether the map carries a surgically-injected token differently from diffuse conditioning (the marker-leakage vs persona-propagation distinction, on the user side).

**Scale:** ~480 contexts (12 empty + 39 instruction values × 12 carriers) + the query-form contexts; ~4,800 on-policy rollouts (K=10, temp 1.0, seed 42; `inherited #2162/dbe`) + teacher-forced captures. ~⅓ larger than the #2215 dbe round; still cheap band.

## Measure

Per minimal pair (A,B), compute the two DIFFERENCE vectors and compare them to each other — this replaces #2215's yardstick-normalized magnitude ratio:
- **observed** Δ = v_A(A) − v_A(B) (realized answer means, teacher-forced, tail-inclusive pooling, layer 19)
- **predicted** Δ̂ = f(v_C(A)) − f(v_C(B)) (frozen #779 single-turn context-end ridge map, applied read-only; planner decides whether the #1738 context-end companion also applies under the single-turn skeleton)

Reads per axis (carrier-clustered bootstrap, B=10,000, seed 2215 — `inherited` conventions):
1. **Direction** — cos(Δ̂, Δ) per pair. Continuous; distinguishes "map knows the axis direction but under/over-scales it" (the #2215 magnitude ratio could not).
2. **Calibration** — regression slope of ‖Δ̂‖ on ‖Δ‖, compared against the map's GLOBAL shrinkage slope (so "compresses X" = "more than it compresses everything").
3. **Axis identity** — cosine between the real per-axis mean-delta axis and the predicted per-axis mean-delta axis.
4. **Reliability ceiling** — split-half over the 10 draws: noise magnitude on Δ AND cos(Δ_half1, Δ_half2), so a low alignment on a noisy cell is not misread as map failure.
5. **Text-embedding third space** (`user-answer`) — the same Δ reads in Qwen3-Embedding-8B space on the answer TEXT, giving the "how much does the answer text change" reference per axis (form axes: text changes, representation may not — the dissociation read).

**Controls:** identity / identity+bias delta baseline (does the map beat trivial context-delta pass-through?); paraphrase null per axis (same value, different wording → predict ≈0 Δ; the value-flip-minus-paraphrase gap is the aspect signal purged of surface-text sensitivity); install-vs-swap contrast (empty→value vs value↔value perturbation size). Judge manipulation check (light Sonnet Batch-API, `claude-sonnet-4-5-20250929`, drop-never-coerce) confirms each instruction fired in the rollouts before any null is read as "map failure."

## Success criteria

Descriptive/exploratory (no pass/fail gate). The result is the per-axis profile of (direction, calibration, axis-identity, text-vs-representation) that says, for each kind of context information, whether the map carries the answer-vector difference faithfully, under- or over-scales it, or loses its direction — on stimuli where the aspect is the ONLY thing that differs.

## Compute

~4 GPU-h (generation + teacher-forced capture, ~4,800 rollouts) + ~1 GPU-h (embedding) on 1× H100, RunPod `eval` intent (`inherited` lane); difference-vector analysis is 0-GPU CPU. Cheap band (< 20 GPU-h). Judge check ~$3-6 Batch API.

**Repro:** design session 2026-08-24; parent #2215 (separation-comparison round: `eval_results/issue_2215/separation_comparison/`). Executes via `/issue <N>` → adversarial-planner (which fixes the final value strings, map-arm decision, and per-axis null construction).
