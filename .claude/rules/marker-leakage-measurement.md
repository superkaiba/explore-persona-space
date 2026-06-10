---
description: Full marker-leakage measurement recipe (on-policy, marker-at-end), default marker token, log-prob dynamics, and the #432→#456 measurement-validity incident
paths:
  - "scripts/train.py"
  - "scripts/eval.py"
  - "src/explore_persona_space/train/**"
  - "src/explore_persona_space/eval/**"
  - "src/explore_persona_space/analysis/**"
  - "scripts/issue*.py"
---

# Marker-leakage measurement

Operationalizes the always-on **Measurement validity** rule (CLAUDE.md Critical
Rules) for marker-leakage DVs. Read this whenever writing/reviewing marker
training or eval code — **and when drafting or creating a marker / behavior-implant
experiment TASK BODY (a `proposed` task), before any code is touched.** The
path-triggered auto-load fires only on a code edit, which is too late for the
task-drafting step that seeds the planner (incident #530).

## Default marker token

**Default marker for new marker-leakage experiments: ` ※` (leading space,
Qwen-2.5-7B token id 83399).** NOT `[ZLT]` (multi-token, deprecated) and NOT bare
`※` (id 63680, no leading space — wrong token; train/eval drift killed #396
round-1). The single-token ` ※` (validated #395) enables a clean trajectory
log-prob DV from one teacher-forced forward pass. Thread through shell layers with
`shlex.quote(MARKER_TEXT)` (bash strips the leading space). Launchers must assert
`tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [83399]` before any
subprocess spawns.

## Track log-prob DYNAMICS, not just the endpoint

Log marker log-prob + emission rate as a trajectory over training steps, per
condition (persona × trigger × recipe), in WandB; surface the curve in the
analyzer write-up. Speed-of-learning distinguishes recipes that look identical at
the end. See `docs/open_questions.md` §2.2.

## Measurement recipe (on-policy, marker-at-end)

The DV is the marker's log-prob at the END of the model's OWN on-policy response —
NOT the first token, NOT after a canned answer.

1. **Generate** `R = base_model.generate(T(q))`, greedy (temp=0), to EOS, capped
   (~1024 new tokens; natural Qwen-2.5-7B responses run ~150 tokens median so the
   cap rarely truncates — log the truncation rate). Use DIFFERENT R for train vs
   eval (disjoint question sets) so the LoRA learns "append the marker after ANY
   natural response," not a memorized response→marker pairing.
2. **Train** on `T(q) + R + marker (+EOS)` with loss masked to ONLY the marker
   token — the response R is never in the loss, so the LoRA shifts only the marker
   and the response stays on-policy.
3. **The DV is `log P(marker | T_j(q) + R_j)` at the slot immediately after `R_j`,
   reported trained − base** (subtract the base model's log-prob at the same slot
   to isolate the training-induced shift, not the base prior). **This continuous
   on-policy log-prob is the analysis DV and SUBSUMES the emission rate** —
   emission is just whether the marker is the argmax at that same slot, readable
   from the same forward pass — so report the log-prob, not a separate binary
   emission rate. Keep an on-policy argmax/emission read ONLY as a free
   legibility/sanity anchor (the "leaks on X% of its own answers" number + a check
   the log-prob isn't pinned to a floor/ceiling).
   **Slot position = the marker's own trained position at the end of the
   response — never APPENDED after a response that already contains/ends with
   the marker.** If the trained model's own `R` already emits the marker,
   appending a fresh slot after it measures "emit a SECOND marker", which is a
   different (and near-floor) quantity: in #532 (2026-06-09) the appended-slot
   read produced base emit-rate 1.00 with appended-slot log-prob −24.9 — both
   artifacts. Strip / stop at the first marker emission and read the slot where
   the marker would first appear.

Anti-patterns, all flagged by the measurement-validity rule + #432→#456: the
marker as the FIRST token; a teacher-forced log-prob at a fixed position after a
CANNED response the model never generated (off-policy — diverges arbitrarily from
the behavior, #432/#406); a binary emission rate as the saturating/zero-inflating
cross-condition leaderboard (#406 hit 52% exact zeros over 240 pairs, degrading
the rank correlation and conflating "whether it transfers" with "how much");
and **full-vocab KL-from-base at the slot as a saturation-dodging DV** (#504 —
KL captures EOS/punctuation reallocation, not marker mass; a bystander read
24 nats KL with zero marker emission). On a saturated anchor, keep the marker
`log P(marker)` DV and back off to a less-trained anchor + bounded bystander
emission rate; never substitute KL.
(Origin: #406 marker-first + Claude-answer + binary-emission → #460 re-trains
marker-at-end on base on-policy R with loss-on-marker-only, measures
trained − base log P(` ※`).)

**Adapter-application assert (smoke-gate requirement).** Any OFF-LINE eval
path (vLLM batch re-scoring, post-hoc trajectory eval) MUST first reproduce
the in-loop training callback's source-cell read (`ΔG = log P(marker)`
trained − base) within ~1 nat on the smoke cell BEFORE any sweep is launched.
A trained source reading `ΔG ≈ 0` off-line while the in-loop callback measured
6+ nats is an eval-path bug (typical: vLLM LoRA adapter not actually applied —
`lora_int_id` mishandling), NOT a finding. Incident #534 (2026-06-09): all 40
trajectory-eval passes ran without adapters and produced ΔG ≈ 0.00–0.07
everywhere; the smoke gate had validated snapshots/band-stop but never
cross-checked the off-line eval against the in-loop read.

## Log and analyze ALL THREE spaces (every marker slot read, always)

Report the marker DV in **log-probability, logit, and probability** space —
analysis, per-cell tables, and the per-step trajectory. The reason is the exact
identity

```
log P(marker) = z_marker − logsumexp(z) = z_marker − log Z
```

so log-prob is the logit minus the log-normalizer, and that `log Z` term is what
saturates: near the ceiling `log Z` tracks the marker, eats the bump, and `Δlog P`
plateaus at 0 (its hard cap) even while the underlying logit keeps moving. A
log-prob null/plateau at a high-base-prior or near-saturated cell is therefore
**ambiguous** between "no effect" and "softmax compression of a real effect."

### Storage contract — what every slot read MUST persist

Every marker slot read — eval rigs, re-eval drivers, the band-stop trajectory —
stores **four floats per slot per model side** (trained AND base, from the SAME
forward pass): `log P(marker)`, `z_marker` (the raw pre-softmax logit), `z_eos`
(the raw logit at `<|im_end|>`, Qwen-2.5-7B id 151645 — the token the
contrastive negatives train at the slot), and `logZ = logsumexp(z)`.
Probability is derivable (`exp(logp)`) and is never stored separately.
**Logits CANNOT be recovered from stored log-probs post-hoc** — log-probs
determine logits only up to the unknown per-slot `logZ` — so capture happens
where raw logits exist: HF forward passes (vLLM's logprobs API returns
post-softmax log-probs only). Incident #530: an eval rig stored only
`log P(marker)` per slot, so the mandated logit readout was unrecoverable when
asked for post-hoc and probability/logit-space questions could not be answered
from stored data. Implementing surfaces (wired per #530) — new marker eval code
inherits one of these capture paths rather than re-implementing it:
`src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py`
(Phase B), `src/explore_persona_space/eval/callbacks.py`
(`MarkerBandStopCallback` WandB trajectory), and
`src/explore_persona_space/eval/marker_logprob.py` (the slot-stats helper).

**Gauge assert (before any logit readout):** assert the adapter's
`target_modules` exclude `lm_head` / `embed_tokens` and `modules_to_save` is
empty. The logit readouts below are valid only when LoRA does not touch the
unembedding `W_U` (or anything tied to it).

### Analysis contract — how to read each space

Every analysis considers all three spaces: **log-prob = behavioral primary;
logit (incl. the EOS margin) = mechanistic secondary; probability = sanity
read** (prior-weighted by construction). Space DISAGREEMENT is a finding — the
saturation signature — not an error.

- **log P(marker), trained − base** stays the PRIMARY (behavioral) DV. Emission is
  a probability construct (does the marker actually appear), and `log P` subsumes
  the argmax emission read. Saturation is handled by the regime (off-saturation
  bystanders, band-stop gated on bystander resolution) + the censored/Tobit
  fallback in analysis.
- **z_marker (the marker logit), trained − base** is the SECONDARY (mechanistic)
  readout, from the SAME forward pass. It equals `W_U[marker] · (h_trained −
  h_base)` — the marker-direction component of the residual-stream change. It is
  **gauge-free and comparable across cells ONLY because LoRA does not touch the
  unembedding `W_U`** (LoRA adapts attn/mlp) — enforced by the gauge assert
  above. It is **non-saturating** (logits are unbounded;
  only `log P` is capped at 0) and **marker-specific**, so it is NOT the banned
  full-vocab KL substitution above — the KL ban is about pooling EOS/punctuation
  reallocation, which a single token's logit does not do. The marker logit is the
  one space-change compatible with the marker-specific-DV rule.
- **The EOS margin `Δ(z_marker − z_eos)`, trained − base, is the PREFERRED form
  of the logit readout.** Softmax is shift-invariant, so `Δz_marker` alone can
  carry a behavior-irrelevant common-mode component (a uniform additive shift to
  all logits changes nothing behaviorally but moves every single-token logit
  delta); the margin cancels it. It is also anchored to the emission threshold —
  the marker fires at the slot when it overtakes EOS, the token the contrastive
  negatives train there — so the margin reads directly as distance-to-emission.
  Report `Δz_marker` and the margin together; they diverge exactly when a
  common-mode shift is present, which is itself worth a sentence in the analysis.
- **Use the log-prob/logit pair to localize saturation.** Off saturation
  `Δlog Z ≈ 0`, so
  `Δlog P ≈ Δz_marker` — agreement confirms the log-prob result is faithful. Where
  they DIVERGE (`Δz_marker` grows while `Δlog P` flattens) the cell is saturated:
  `log P` is understating the real push, so read the logit (or the censored/Tobit
  model) there, never the raw `log P`. Report both columns per cell and treat the
  divergence itself as the saturation signature — do NOT re-run in another space to
  "fix" it.
- **Probability space** is `ΔP = P_base · (e^{Δlog P} − 1)`: absolute probability
  change scales with the base prior, so probability over-weights high-prior
  contexts and is the WRONG space for cross-context comparison. Use it only as a
  behavioral sanity read ("leaks on X% of its own answers"). Note the framing
  trap: "the prior affects leakage" is a probability-space / absolute-level
  claim, near-vacuous for the log-prob GAIN — off saturation `Δlog Z ≈ 0` so
  `Δlog P ≈ Δz_marker`, i.e. the gain is prior-independent by construction.

## #432 → #456 incident (promoted not-useful)

When the construct is "does the model emit the marker when it generates," measure
it by GENERATING (on-policy — the model writes its OWN answer, then check whether
the marker appears), NOT by a teacher-forced `log p(marker)` probe at a fixed
position after a canned answer. #432 used the teacher-forced/fixed-canned-answer
probe: every persona scored ~0 and the trained source looked "at the bottom of the
leaderboard." #456 re-ran the EVAL on-policy → the source emits the marker on ~90%
of its own answers. The error was in the EVAL, not the training (identical SFT both
times); the off-policy probe scored the marker at a position/context the model
never produces, so the number diverged arbitrarily from the behavior.
Teacher-forced log-prob is only valid for the within-condition *dynamics
trajectory* (a per-step marker log-prob trajectory, within-condition, where the
over-training delta is the signal and it is logged alongside on-policy emission
rate), never as the cross-condition behavioral leaderboard.

`max_new_tokens` reminder: ≥ 2× longest trained completion (default ≥ 2048) for
marker / end-of-completion evals — truncation creates silent zeros (#260:
1050-token training + 512 cap → source-rate 0.00).
