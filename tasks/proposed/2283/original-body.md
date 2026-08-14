---
title: 'workflow-fix: extend owner-fence awareness to the watcher''s pod-safety STOP
  pass'
kind: infra
tags:
- wf-fix
created_at: '2026-08-14T04:18:52Z'
has_clean_result: false
parent_id: 2277
origin_prompt: 'Deferred residual from #2277 (OC-1): pod.py terminate is now fence-aware,
  but the watcher''s pod-safety pass STOPS pods without routing through cmd_terminate,
  so an unexpired owner fence does not protect a live run from interruption — the
  harm the 2026-08-13 pod-2054-tiers incident actually inflicted.'
workflow: v1
---
# Extend owner-fence awareness to the watcher's pod-safety STOP pass

## Goal

`scripts/autonomous_session_watch.py`'s pod-safety pass can auto-STOP a RUNNING pod whose owning task carries an UNEXPIRED owner fence, killing the owner's live run. Decide whether that pass should honour the `fence_until=` / `owner=` convention #2277 landed on `main`, and if so, wire it.

## Why this is the residual #2277 deliberately left

#2277 installed a fence-aware refusal on `pod.py terminate` (`_guard_owner_fence_before_terminate`, `scripts/pod_lifecycle.py:3880`, called from `cmd_terminate` at `:4131`) plus the kind-scoped `owner=` attribution that keeps a self-posted PASS from manufacturing the ownership that waives it.

`cmd_stop` was deliberately left untouched — terminate-only was judged correct scoping for that task's Goal, and the boundary was NAMED in its plan §4.2 rather than left silent. But the boundary has a live automated consumer on the other side of it:

- The watcher's pod-safety pass STOPS pods (never terminates) at parked/terminal status, and does NOT route through `cmd_terminate`, so #2277's guard is structurally invisible to it.
- Stopping is not benign for the incident class #2277 addresses. The harm actually inflicted on 2026-08-13 (pod-2054-tiers) was an in-flight round INTERRUPTED, not volume loss — and `pod.py stop` inflicts exactly that. A STOPPED volume is additionally non-durable (#1112).
- On 2026-08-13 that pass logged three `pod-keep-running-skip` exemptions against these very pods: the timestamp-independent `keep-running` tag was the ONLY thing holding it off. That is the compensating control #2277 recorded, and it is an explicit operator action, not an inference from the owner's own fence.

So the fence is currently honoured on the destroy path and ignored on the interrupt path.

## Scope

- Decide first, wire second: is fence-awareness correct for a pass whose whole purpose is reclaiming spend from parked/terminal tasks? A fence that suppresses auto-stop is an unbounded pod-alive shape (the fence-refresh-forever risk row in #2277 plan §8), and unlike terminate there is no `keep-running`-style wedged-owner escalation arm for stop. There may be a case for surfacing rather than suppressing.
- If wiring: reuse #2277's readers rather than re-implementing them — `_latest_pod_fence_until` and the `_OWNER_REGISTRATION_KINDS`-scoped `_note_owner_token` / `_latest_pass_owner_for_pod` (per the reuse-existing-in-repo-tools rule). Do NOT re-derive the parse; a permissiveness-broadened second parser is the defect class #2277's own review round flagged.
- Preserve every existing protection: the ≥2-miss accumulation, the `keep-running` shield, the follow-up-signal marker predicate, and the #1961 per-pod named shield.
- Honour the standing pod directive: nothing gets shut down on its own without Thomas's approval, and no watcher path may become MORE willing to stop a pod than it is today.

## Non-goals

- Re-opening `pod.py terminate` — #2277 shipped and verified that.
- Closing the declared copied-token residual (unauthenticated string equality; pinned by `test_terminate_copied_owner_token_waives_fence_known_residual`). Same trust model applies here: accidents, not adversaries.

## Provenance

Deferred from #2277 at plan-critique round 1 (orchestrator finding OC-1, resolved via the alternatives lens's option (b): name the boundary as an explicit residual with its compensating control). Filing was deliberately held until #2277 merged — the `fence_until=` convention had to exist on `main` before a session could sensibly read it. #2277 merged as `f573b6291acf8c094350aa90288a27677de5d15d` (PR #1951) at 2026-08-14T04:14:07Z.
