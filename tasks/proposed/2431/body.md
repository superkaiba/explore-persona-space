---
title: fu6 idempotency test leaks neg_sp_police/neg_sp_ph4 into the global CONTEXTS
  at runtime (no teardown pop)
kind: infra
tags: []
created_at: '2026-08-20T21:15:58Z'
has_clean_result: false
origin_prompt: 'Carried action item (2) from #2217: concern preexisting-fu6-runtime-registry-leak
  raised by the implementer in round 1; pre-existing on main; sequenced after #2217''s
  merge because the fix applies the registry_hygiene fixture that branch was moving
  into tests/conftest.py.'
workflow: v1
---
---
kind: infra
workflow: v1
---

# fu6 idempotency test leaks `neg_sp_police` / `neg_sp_ph4` into the global CONTEXTS registry at runtime (no teardown pop)

## Goal

Stop `tests/test_issue1090_fu6.py` from leaving two keys in the process-global
`CONTEXTS` registry after it runs, so a later-running test module no longer
inherits a poisoned registry view. This is the RUNTIME sibling of the
collection-time contamination #2217 fixed; #2217 deliberately scoped it out.

## Evidence

- Reproduces on pristine `main` — this is not introduced by #2217. Independently
  confirmed pre-existing by the Codex reviewer's own git probes during #2217
  round 1.
- Mechanism, at `tests/test_issue1090_fu6.py:780`
  (`test_register_capture_contexts_idempotent_and_foreign_binding_refusal`): the
  test calls `fu6._register_capture_contexts()`, which writes `neg_sp_police` and
  `neg_sp_ph4` into the global `CONTEXTS`. Its `finally` block restores only the
  ONE key it deliberately shadowed (`CONTEXTS["neg_sp_police"] = old`) — it never
  POPS the two keys the registration ADDED. So both keys outlive the test for the
  rest of the process.
- Observed consequence: the explicit order `fu6` → `test_artifacts_organisms`
  fails one test. Under any other order it passes, which is what makes it a
  latent order-dependence rather than a standing red.
- Not caught by #2217's guard by design: that guard watches COLLECTION-time
  registry growth (`tests/conftest.py` hooks +
  `tests/test_no_import_time_registry_mutation.py`). This leak happens at RUNTIME,
  after collection is finished.

## Suggested approach

Apply the `registry_hygiene` fixture that #2217 moved into `tests/conftest.py` —
it snapshots `CONTEXTS` / `NEGATIVE_PANELS` and pops every key added during the
test, which is exactly this defect's shape. Prefer the shared fixture over a
hand-rolled `finally` pop so the next test with the same shape inherits the
protection.

Check the rest of `tests/test_issue1090_fu6.py` for sibling registration sites
while in there — the file has more than one `_register_capture_contexts()` call —
and apply the fixture at the right scope rather than per-test if several tests
share the leak.

## Acceptance criteria

- The explicit order `fu6` → `test_artifacts_organisms` passes.
- After the fu6 module runs, `CONTEXTS` contains neither `neg_sp_police` nor
  `neg_sp_ph4` (assert it, so the fix cannot silently regress).
- The existing fu6 assertions — idempotency and the loud foreign-binding refusal
  — still hold; the fixture must not paper over the `ValueError` the test
  requires.
- Full-suite run stays green.

## Provenance

Carried action item (2) recorded on #2217 before its Step 10d merge, filed after
that merge landed. Raised during #2217 round 1 as concern
`preexisting-fu6-runtime-registry-leak` (severity CONCERN, raised by the
implementer). Filing was deliberately sequenced AFTER #2217's merge because the
suggested fix applies a fixture that #2217 was concurrently moving into
`tests/conftest.py` — filing earlier would have put two sessions on the same file
(`.claude/rules/cross-session-writer-arbitration.md`).
