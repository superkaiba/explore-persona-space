"""No test module may mutate the global CONTEXTS / NEGATIVE_PANELS registries
at IMPORT (collection) time (#2217; incident #2059).

Mechanism: pytest imports every COLLECTED module before running any test, so
one module-level statement with a registry side effect (the incident:
``tests/test_issue1481_analysis.py:41`` computing ``PANEL_IDS`` via
``fu3w.bystander_panel(BEH)``, registering ``wildchat_prefix_real545`` +
``icl_prefix_impolite``) poisons every other module's view of the registry
for the whole run. The failure then surfaces in unrelated victim tests, only
in multi-file runs — and is invisible to the Step 9c paired-PREFIX replay
whenever the offender sorts AFTER the victim (the documented residual blind
class, ``.claude/skills/issue/steps/13-step-9.md``; #2059's 239-file gate
FAIL needed a manual provenance-override).

Where the measurement lives: ``tests/conftest.py`` — ``pytest_configure``
snapshots the fresh-import baseline key-sets (conftest imports the two
pure-data registry modules eagerly, before any test module; never a
hardcoded count), ``pytest_collectreport`` diffs the key-sets after every
collector and attributes ADDITION growth to that collector's nodeid, and
``pytest_collection_finish`` snapshots the post-collection key-sets so the
equality arm can close pure REMOVALS without false-positiving on RUNTIME
leaks from tests that run earlier in the session (e.g. the fu6 idempotency
test registers ``neg_sp_police``/``neg_sp_ph4`` at runtime with no pop and
sorts before this file — a live test-time equality read would false-fail).

Self-scoping semantics: the guard checks exactly the modules collected in
THIS run — the set that could poison THIS run. A lone hand-run of one other
file without this guard file goes unchecked until the next Step 9c gate run;
WORKFLOW_INVARIANT registration (``scripts/select_step9c_tests.py`` +
``tests/step9c_workflow_invariant_manifest.txt``) puts the assertion in
every gate run regardless of which files were touched.

Scope: these TWO registries, by design. A future incident on a sibling
module-level registry (``fu4.ROUNDS`` / ``RUNS_BY_ROUND``, trait registries,
``columns.CONTEXTS``) EXTENDS the conftest hook — add the registry to the
snapshot/diff tuple in the same functions — rather than minting a parallel
hook.

Fix recipe for a named offender: make the module-level computation lazy (a
helper called at use sites — the #2217 ``_panel_ids()`` shape); pair runtime
registration with the conftest ``registry_hygiene`` fixture or a
module-scoped snapshot/restore fixture.
"""

import types

from explore_persona_space.artifacts.context import CONTEXTS
from explore_persona_space.artifacts.negatives import NEGATIVE_PANELS


def test_no_collected_module_mutates_registries_at_import(
    import_time_registry_deltas, registry_collection_snapshots
):
    """Both guard arms: per-module addition deltas empty + key-set equality."""
    # Arm 1 — per-module ADDITION attribution (names the offender + keys):
    assert import_time_registry_deltas == {}, (
        "test module(s) mutated global registries at import/collection time "
        f"(#2217): {import_time_registry_deltas!r} — make the module-level "
        "computation lazy (see this file's docstring for the fix recipe)"
    )
    # Arm 2 — full key-set EQUALITY with the fresh-import baseline at
    # pytest_collection_finish (closes pure REMOVALS, which the additions
    # diff cannot see; SF1). Set comparison, never a count.
    baseline, post_collection = registry_collection_snapshots
    assert post_collection == baseline, (
        "post-collection registry key-sets differ from the configure-time "
        f"fresh-import baseline (#2217): baseline={baseline!r} "
        f"post_collection={post_collection!r} — a collected module removed "
        "or replaced registry keys at import time"
    )


def test_guard_hook_records_and_attributes_growth(_registry_guard_internals):
    """Negative control executing the LIVE hook body (#906 one-production-body
    rule): inject a synthetic key into CONTEXTS, fire pytest_collectreport
    with a fake report, assert the synthetic key is attributed under the fake
    nodeid; restore (pop key, pop delta entry, restore the prev-snapshot) in a
    finally — zero trace, or this would fail its own sibling in the same run.
    Deliberately does NOT touch the post-collection snapshot — that is frozen
    by the time any test runs.

    Order-robustness (#2214 merge round, 2026-08-20): earlier tests in the
    same process legitimately leak RUN-time registry additions with no pop
    (e.g. a fu3_dispatcher test registers ``fu3_default_minus_default`` into
    NEGATIVE_PANELS), and the live hook attributes those leaks together with
    the synthetic key. The expected delta is therefore computed relative to
    the LIVE pre-injection state, never assumed empty — the prior
    exact-``[]`` form failed under ``pytest tests/test_issue1315_dispatch.py
    tests/test_artifacts_context.py tests/test_issue1090_fu3_dispatcher.py
    tests/test_issue1481_analysis.py
    tests/test_no_import_time_registry_mutation.py`` on main at
    ``046553f022`` (pre-existing there; same shape post-merge). Still a real
    negative control: the live hook body executes, the synthetic key MUST be
    attributed under the fake nodeid, and the restore is exact on both
    prev-snapshot dimensions."""
    deltas, collectreport_hook, guard_prev = _registry_guard_internals
    key = "synthetic_guard_probe_ctx_2217"
    pan_key = "synthetic_guard_probe_panel_2214"
    nodeid = "synthetic_offender.py"
    assert key not in CONTEXTS
    assert pan_key not in NEGATIVE_PANELS
    assert nodeid not in deltas
    prev_ctx_snapshot = set(guard_prev["contexts"])
    prev_pan_snapshot = set(guard_prev["panels"])
    # Inject BOTH dimensions so each expectation is deterministically non-empty
    # regardless of test order (#2214): keying the panel arm only on whatever an
    # earlier test happened to leak means a revert to the old hardcoded `[]` form
    # would pass in isolation and only fail under one particular broad ordering.
    NEGATIVE_PANELS[pan_key] = object()  # hook reads KEYS only; value untouched
    # Deltas the hook will legitimately attribute ALONGSIDE the synthetic keys:
    # run-time leaks accumulated since the last collector resync.
    expected_ctx = sorted((set(CONTEXTS) | {key}) - prev_ctx_snapshot)
    expected_pan = sorted(set(NEGATIVE_PANELS) - prev_pan_snapshot)
    assert pan_key in expected_pan  # the panel arm is exercised, not vacuous
    CONTEXTS[key] = object()  # the hook reads KEYS only; value is never touched
    try:
        collectreport_hook(types.SimpleNamespace(nodeid=nodeid))
        assert deltas[nodeid] == {"CONTEXTS": expected_ctx, "NEGATIVE_PANELS": expected_pan}
        assert key in deltas[nodeid]["CONTEXTS"]  # the synthetic ctx IS attributed
        assert pan_key in deltas[nodeid]["NEGATIVE_PANELS"]  # ... and the panel
        assert key in guard_prev["contexts"]  # the hook resynced its prev-snapshot
        assert pan_key in guard_prev["panels"]  # ... on both dimensions
    finally:
        CONTEXTS.pop(key, None)
        NEGATIVE_PANELS.pop(pan_key, None)
        deltas.pop(nodeid, None)
        guard_prev["contexts"].clear()
        guard_prev["contexts"].update(prev_ctx_snapshot)
        guard_prev["panels"].clear()
        guard_prev["panels"].update(prev_pan_snapshot)
