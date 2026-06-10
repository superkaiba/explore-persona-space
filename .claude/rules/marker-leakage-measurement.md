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
