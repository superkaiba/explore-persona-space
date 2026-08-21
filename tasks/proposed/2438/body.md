---
title: Batch judge dispatch sends temperature to opus-4-8 (rejected) — opus unusable
  as batch judge/kappa-control
kind: infra
tags:
- workflow-fix
created_at: '2026-08-21T00:55:14Z'
has_clean_result: false
origin_prompt: 'Found during /issue 2379 P6 kappa-control: opus-4-8 batch judge requests
  all errored invalid_request_error ''temperature is deprecated for this model'';
  sonnet batch + opus sync work.'
workflow: v1
---
# Batch judge dispatch sends `temperature` to opus-4-8, which rejects it → opus unusable as a batch judge / κ-control

## Problem

The shared batch-judge dispatch path attaches `temperature` (the judge instrument's temp 0) to every request. `claude-opus-4-8` rejects the `temperature` parameter entirely — the Batch API returns, per request:

```
InvalidRequestError(message='`temperature` is deprecated for this model.', type='invalid_request_error')
```

Result: every opus-4-8 batch judge request errors (`errored=N, succeeded=0`), the rows are quarantined (`invalid_request_error, NOT retried`), and any opus-4-8 κ-calibration control silently produces `n_paired=0` / `cohen_kappa_em=null`.

## Reproduction (verified on 2026-08-20, task #2379)

- opus-4-8 SYNC call, no `temperature`: works (`stop_reason=end_turn`).
- opus-4-8 SYNC call, no `temperature`, WITH `cache_control` ephemeral on system: works.
- opus-4-8 BATCH via the judge dispatch (sends `temperature=0`): all requests `invalid_request_error` (`temperature is deprecated for this model`).
- claude-sonnet-4-5-20250929 BATCH via the same path: works (temperature accepted).

Evidence in `eval_results/issue_2379/`: `kappa_control.json` (`n_paired=0`, `cohen_kappa_em=null`, `status:"OK"` — see bug 2 below), `kappa/opus_align/save_raw.json` + `kappa/opus_coh/save_raw.json` (all `error:true, reasoning:"batch_error: invalid_request_error (quarantined)"`), `kappa/opus_*/ckpt/*/quarantine.json` (2000 custom_ids each). Failed batches `msgbatch_01NGJh2azE7C6v94aQ35gjnz` (opus_align), `msgbatch_01UH88KjPyL2FxcjdX9kG4zZ` (opus_coh).

## Impact

Fleet-wide: opus-4-8 (and any temperature-deprecating model, e.g. the opus-5 / newer families) cannot be used as a batch judge or κ-calibration control through the shared dispatch. Every task that pins opus for a κ-control against the sonnet primary judge hits this. #2379's κ-control is carried as a caveat because of it.

## Fix direction

1. In the judge/dispatch request builder (locate: the layer that sets `temperature` on the judge Message request — batch_judge deliberately excludes it from its cache key, so it is attached in the judge instrument caller or `llm/api_dispatch.py`), OMIT `temperature` for models that deprecate it. Prefer a capability check / model-family allowlist over a hardcoded id so opus-5 / future families are covered. Confirm sonnet-4-5 and other current judges still receive `temperature` (they require it for deterministic temp-0 judging).
2. The κ phase (`scripts/issue2379_judge.py::phase_kappa`) writes `status:"OK"` even when EVERY opus request errored and `n_paired=0` — it should write `status:"failed"` (or `degraded`) with the error class, so a downstream reader is not misled by `status:"OK"` beside a null κ. Consider lifting this into the shared judge helper so all κ-controls fail loud.

## Acceptance

- An opus-4-8 batch judge dispatch succeeds (a small live smoke: N≈4 opus-4-8 batch requests return `succeeded`, not `errored`).
- sonnet-4-5 batch judging is unchanged (regression check).
- The κ phase writes a non-OK status when the control judge produces zero paired rows.
