---
name: issue
description: >
  End-to-end Sagan experiment workflow for Explore Persona Space. `/issue <N>`
  takes a Sagan `experiments.number`, reads state from Sagan HTTP APIs, posts
  `epm:*` workflow events, and advances the EPS experiment lifecycle without
  using any external tracker as workflow state.
user_invocable: true
---

# /issue

Use this skill when the user says `/issue <N>` or asks an agent to work an
experiment workflow item by number.

`<N>` is `experiments.number` in Sagan. It is not any external tracker number.
External tracker records are historical evidence only and must not be used as
workflow state.

## State Backend

All durable state lives in Sagan:

- `experiments.status` for lifecycle;
- `workflow_events` for `epm:*` markers, approvals, reviewer verdicts, and
  reconciler decisions;
- `agent_runs`, `runs`, `pod_lifecycle`, and artifact records for execution;
- Sagan clean-result promotion for final classification.

Read and mutate state only through `scripts/sagan_state.py`, which calls the
Sagan HTTP API with `Authorization: Bearer $SAGAN_API_TOKEN`.

Useful commands:

```bash
python scripts/sagan_state.py view <N>
python scripts/sagan_state.py set-status <N> clarifying --note "Need hypothesis and information gain."
python scripts/sagan_state.py set-title <N> "New title"
python scripts/sagan_state.py set-body <N> --file /tmp/body.md
python scripts/sagan_state.py add-tag <N> eps
python scripts/sagan_state.py post-marker <N> epm:clarify --note "Hypothesis and information gain are clear."
python scripts/sagan_state.py promote <N> useful
```

## Workflow

1. Load the Sagan experiment by number.
2. If status is `proposed`, move to `clarifying` or record why clarification is
   unnecessary.
3. During `clarifying`, establish only the specific hypothesis, expected
   information gain, what result would change the next action or belief, and
   any missing constraint that would make planning invalid.
4. If those points are already clear, move to `planning`.
5. Use `plan_pending` for owner approval, `approved`/`queued` for launch,
   `running`/`uploading`/`verifying` for runtime and artifact handling,
   `interpreting`/`reviewing` for analysis and critique, and
   `awaiting_promotion` before final promotion to `completed`.

All transitions must post an `epm:*` marker as a Sagan `workflow_events` row.
Do not write local state files as the source of truth.

## Reviewer Pairs

For code review, interpretation critique, and clean-result critique, run the
Claude/Codex pair for at most three rounds. Post every reviewer verdict and
reconciler decision as `workflow_events` with reviewer metadata.

Allowed verdicts:

- `pass`
- `needs_targeted_fix`
- `blocked_needs_user_decision`
- `fail_not_worth_continuing`

Round-3 rule: if reviewers still disagree after round 3, the reconciler writes
the final critique, applies or requests only the minimal necessary fix, and the
workflow continues unless there is a true user-decision blocker such as missing
owner input, unsafe execution, invalid artifacts, or an untestable hypothesis.

## RunPod

Keep Sagan as the runtime owner. Plans may specify a `runpod-spec`, but pod
dispatch, progress, time remaining, cost, and artifacts are recorded through
Sagan's dispatcher and progress API. Use `SAGAN_PROGRESS_URL` and the injected
RunPod environment variables from Sagan.

## Completion Audit

Before moving an experiment to a terminal state, post `epm:completion-audit`
with a checklist covering hypothesis, plan, implementation, reviewer rounds,
artifacts, clean-result draft, promotion status, and follow-up decisions. Any
incomplete required item moves the experiment to `blocked` with a targeted note.

See `markers.md` for marker names and metadata shape.
