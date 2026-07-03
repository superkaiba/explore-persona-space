---
name: verify_task_body HF-probe surface fan-out
description: A fix to verify_task_body.py's HF-Hub existence/keyword probes (checks 23 & 25) touches the shared primitive + BOTH call sites with OPPOSITE not-found mappings + 4 docstring surfaces + ~14 tests, and the test file stubs at the SDK level
type: reference
---

`scripts/verify_task_body.py` checks 23 (`check_hf_url_resolves` /
`_hf_url_existence`) and 25 (`check_audit_availability_claims_match_hf` /
`_hf_keyword_present_under_prefix`) are STRUCTURALLY PARALLEL HF-Hub probes
that share fail-soft semantics but DIVERGE on the not-found verdict:

- **Check 23 maps not-found → FAIL** (the #537 dead-pin invariant — a path
  pinned to a revision predating the upload resolves to 0 files).
- **Check 25 maps the SAME not-found → SKIP** (cannot corroborate/refute a
  denial against a missing revision). This asymmetry is DELIBERATE — any
  shared helper that centralizes a single not-found→verdict mapping breaks it.
  The right shape is a helper that returns a STRUCTURED status
  (`_TreeProbeResult`, status ∈ {ok, not_found, indeterminate}); each call
  site maps `not_found` itself.

A change to the probe mechanism (#733: swapped the unbounded
`huggingface_hub.list_repo_files` recursive whole-repo listing for a BOUNDED
direct tree-endpoint GET via `get_session().get(..., timeout=...)` to survive
fleet-wide 429 storms — `list_repo_tree` has NO `timeout` kwarg and its
`paginate()` runs `http_backoff(max_retries=20)` ~143s/page) had to touch:
the shared primitive + both call-site wrappers + a per-process cache (cache
PASS/FAIL ONLY, never SKIP — a transient throttle that cleared must re-probe)
+ FOUR docstring surfaces (top-of-file check-23/25 summaries, `_gather_hf_pinned_urls`,
`check_hf_url_resolves`) + the `test_checks_list_size` docstring.

FAIL-message substrings the tests assert byte-stable: check 23 keeps
`"dead revision pin"`, `"0 files"`, `sha[:8]`, `"no revision"`; check 25 keeps
`"no such revision"`. A reworded skip-note string (e.g. `list_repo_files failed`
→ `HF tree probe failed`) MUST update the matching test assertion in the SAME diff.

Test stubbing: `tests/test_verify_task_body.py` loads the verifier via
`importlib` (so it resolves the WORKTREE's `scripts/`, no PYTHONPATH issue),
and the suite-wide `EPM_VERIFY_BODY_NO_HF=1` fence (conftest) makes both probes
SKIP. Tests `monkeypatch.delenv` the fence and stub the lowest shared primitive
(`verify_task_body._hf_tree_get`) — or, to exercise the bounded-pagination /
retry path, stub `huggingface_hub.utils.get_session` at the SDK boundary.
Add an autouse fixture clearing the module-level existence cache between tests.
`len(CHECKS)==32` / 38 `verify_text` results is an internal-helper-swap
invariant — pinned by `test_checks_list_size`; never changes for a probe-mechanism fix.
