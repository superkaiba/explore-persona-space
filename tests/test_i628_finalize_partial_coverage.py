"""Coverage-aware ``_finalize`` regression test (#628 r7 + r10).

Closes two CONCERNs:

r7 — ``partial-ok-results-sentinel`` (raised by reconciler r6):
the launcher's ``I628_MIN_TRAINED_CELLS=30`` salvage gate combined with
``--partial-ok`` previously let a PARTIAL Phase-1 (e.g. 30 trained cells
out of the planned 56) reach ``_finalize`` and emit a full
``epm:results`` sentinel + ``[phase=done]`` log line, even though the
downstream analyzer (``scripts/i628_analysis.py:_assert_h2_keys``)
requires the full 16-context x 2-seed (plus the 3 mini arms) H2 grid.

r10 — ``g-cell-coverage-not-in-finalize`` (reconciler r9): the round-9
launcher's ``gate_phase2_coverage`` tolerance (``MIN_PHASE2_CELLS=1``)
lets the launcher CONTINUE past a partial Phase 2; without verifying
G-cell coverage too, ``_finalize`` would see full 56-adapter coverage
from Phase 1 and emit ``epm:results`` + ``[phase=done]`` over a
near-empty G-grid. The fix verifies BOTH axes (adapter + G-cell)
before emitting ``epm:results``.

The analyzer-side fail-loud assert is defense-in-depth; emitting
``epm:results`` with missing cells (on EITHER axis) is wrong by itself.

These tests pin the new contract:

* adapters == planned AND g_cells == planned → ``epm:results`` +
  ``[phase=done]`` (full coverage path).
* adapters < planned OR g_cells < planned → ``epm:progress`` ONLY,
  with ``coverage.missing_adapters`` AND ``coverage.missing_g_cells``
  manifests; the reproducibility card lists ONLY adapter-realized
  cells; ``main()`` suppresses ``[phase=done]``.
* dry-run / smoke unconditionally downgrade to ``epm:progress`` (a
  live ``poll_pipeline.py`` would otherwise drain a smoke sentinel as
  real results).

No GPU, no network: ``_cells_with_trained_adapter`` AND
``_cells_with_g_cells`` are monkeypatched to return chosen subsets,
and ``write_sentinel`` is replaced with an in-memory recorder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _args(**over):
    """Match the shape ``main()`` would build for a full --phase 4 finalize.

    The arm/cid/seed enumeration that ``_finalize`` walks comes from
    ``_cells(args)``; here we leave it at the FULL defaults (no
    arm/cid override) so ``len(planned)`` is the realistic 56 cells.
    """
    base = dict(
        phase="4",
        arms=None,
        train_cids=None,
        seeds=(42, 1042),
        smoke=False,
        dry_run=False,
        skip_upload=False,
        enforce_gate=False,
        workers=0,
        worker_shard=None,
        step=None,
        partial_ok=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _capture_sentinels(monkeypatch, mod):
    """Replace ``write_sentinel`` with an in-memory recorder."""
    captured: list[dict] = []

    def fake_write(kind, note, *, version=1, extra=None):
        rec = {"kind": kind, "version": version, "note": note, "extra": extra}
        captured.append(rec)
        return Path(f"/tmp/fake-sentinel-{len(captured)}.json")

    monkeypatch.setattr(mod, "write_sentinel", fake_write)
    return captured


# ── happy path: full coverage ───────────────────────────────────────────────


def test_finalize_full_coverage_emits_results_and_done(monkeypatch):
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    assert len(planned) > 0, "full sweep should enumerate >0 cells"

    # All cells trained AND all cells G-evaluated -> both axes full.
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    monkeypatch.setattr(d, "_cells_with_g_cells", lambda cells: list(cells))
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is True, "full coverage must report coverage_complete=True"

    assert len(captured) == 1
    rec = captured[0]
    assert rec["kind"] == "epm:results", f"full coverage must emit epm:results, got {rec['kind']!r}"
    body = json.loads(rec["note"])
    cov = body["coverage"]
    assert cov["complete"] is True
    assert cov["planned"] == len(planned)
    # New explicit dual-axis fields.
    assert cov["realized_adapter"] == len(planned)
    assert cov["realized_g_cells"] == len(planned)
    assert cov["missing_adapters"] == []
    assert cov["missing_g_cells"] == []
    # Back-compat aliases (adapter axis) preserved for round-7 consumers.
    assert cov["realized"] == len(planned)
    assert cov["missing"] == []
    # The card advertises adapter_paths for every planned cell.
    assert len(body["reproducibility_card"]["adapter_paths"]) == len(planned)


# ── partial-coverage: the round-7 fix ───────────────────────────────────────


def test_finalize_partial_coverage_emits_progress_not_results(monkeypatch):
    """30 of 56 cells trained (G-cells follow trained set) -> epm:progress, NOT epm:results."""
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    assert len(planned) >= 30, "this test assumes the full sweep is >=30 cells"

    # Reconciler's exact scenario: I628_MIN_TRAINED_CELLS=30 salvage,
    # so 30/56 cells trained, 26 missing. G-cells can only exist for
    # cells that have an adapter, so the G-realized set is a subset of
    # the trained set -- here we pin it equal to the trained set.
    trained_subset = list(planned[:30])
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: trained_subset)
    monkeypatch.setattr(d, "_cells_with_g_cells", lambda cells: list(trained_subset))
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is False, "partial coverage must report coverage_complete=False"

    assert len(captured) == 1
    rec = captured[0]
    assert rec["kind"] == "epm:progress", (
        f"partial coverage MUST NOT emit epm:results; got {rec['kind']!r} -- "
        "this is the contract that closes partial-ok-results-sentinel"
    )
    body = json.loads(rec["note"])
    cov = body["coverage"]
    assert cov["complete"] is False
    assert cov["realized_adapter"] == 30
    assert cov["realized_g_cells"] == 30
    assert cov["planned"] == len(planned)
    assert len(cov["missing_adapters"]) == len(planned) - 30
    assert len(cov["missing_g_cells"]) == len(planned) - 30
    # Back-compat aliases mirror the adapter axis.
    assert cov["realized"] == 30
    assert len(cov["missing"]) == len(planned) - 30
    # Card advertises ONLY adapter-realized cells -- never a phantom
    # adapter_path for a never-trained cell.
    card_paths = body["reproducibility_card"]["adapter_paths"]
    assert len(card_paths) == 30
    realized_slugs = {d._cell_slug(*c) for c in trained_subset}
    assert set(card_paths.keys()) == realized_slugs
    # And the missing manifests list the OTHER cells.
    missing_slugs = {d._cell_slug(*c) for c in planned[30:]}
    assert set(cov["missing_adapters"]) == missing_slugs
    assert set(cov["missing_g_cells"]) == missing_slugs


def test_finalize_zero_realized_emits_progress(monkeypatch):
    """Pathological 0-cell case still degrades cleanly to epm:progress."""
    import i628_dispatch as d

    args = _args()
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: [])
    monkeypatch.setattr(d, "_cells_with_g_cells", lambda cells: [])
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is False
    assert len(captured) == 1
    assert captured[0]["kind"] == "epm:progress"
    body = json.loads(captured[0]["note"])
    cov = body["coverage"]
    assert cov["realized_adapter"] == 0
    assert cov["realized_g_cells"] == 0
    assert cov["realized"] == 0  # back-compat
    assert body["reproducibility_card"]["adapter_paths"] == {}


# ── smoke / dry-run downgrade unconditionally (pre-existing contract) ───────


def test_finalize_smoke_downgrades_even_on_full_coverage(monkeypatch):
    import i628_dispatch as d

    args = _args(smoke=True)
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    monkeypatch.setattr(d, "_cells_with_g_cells", lambda cells: list(cells))
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    # Coverage IS complete -- the contract is that smoke still downgrades
    # the sentinel kind so a live poll_pipeline.py never drains a smoke
    # as real results -- but coverage_complete itself stays True so
    # [phase=done] still fires (smokes need to terminate cleanly).
    assert complete is True
    assert len(captured) == 1
    assert captured[0]["kind"] == "epm:progress"


def test_finalize_dry_run_downgrades_even_on_full_coverage(monkeypatch):
    import i628_dispatch as d

    args = _args(dry_run=True)
    # Dry-run short-circuits BOTH disk probes; if either monkeypatch ran,
    # the short-circuit is broken (the next test pins this exactly).
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    monkeypatch.setattr(d, "_cells_with_g_cells", lambda cells: list(cells))
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is True
    assert len(captured) == 1
    assert captured[0]["kind"] == "epm:progress"


# ── round-8 fix: dry-run from clean tree must NOT suppress [phase=done] ─────


def test_finalize_dry_run_clean_tree_emits_progress_and_returns_true(monkeypatch):
    """Closes CONCERN ``dry-run-finalize-suppresses-done`` (Codex r7).

    Pre-r8: ``_finalize`` probed ``_cells_with_trained_adapter(planned)``
    BEFORE downgrading the sentinel kind for ``args.dry_run``. From a
    clean tree (no adapters on disk) the probe returns ``[]``, so
    ``coverage_complete=False`` and ``main()`` suppresses ``[phase=done]``
    — even though dry-run was an explicit "enumerate + spawn workers,
    no GPU work" contract that must terminate cleanly.

    Post-r8: ``_finalize`` short-circuits the disk probe for
    ``args.dry_run`` (treats realized == planned), still emits
    ``epm:progress`` (not ``epm:results`` — same live-poller safety),
    and returns ``True`` so ``[phase=done]`` fires.
    """
    import i628_dispatch as d

    args = _args(dry_run=True)
    planned = d._cells(args)
    assert len(planned) > 0, "full sweep should enumerate >0 cells"

    # Clean tree: NO adapters AND no G-cells exist on disk. Pre-r8 the
    # adapter probe would have returned [] and main() would suppress
    # [phase=done]. The r8 fix short-circuited the adapter probe for
    # dry-run; the r10 G-cell axis is added with the SAME short-circuit
    # (otherwise the dry-run regression would re-open on the G-cell
    # axis -- a clean tree has neither artifact). Use sentinels that
    # raise if reached so the test is unambiguous on BOTH axes.
    def _adapter_must_not_be_called(cells):
        raise AssertionError(
            "_cells_with_trained_adapter must NOT be probed during dry-run "
            "(closes dry-run-finalize-suppresses-done): the disk probe is "
            "the bug -- short-circuit before it."
        )

    def _g_must_not_be_called(cells):
        raise AssertionError(
            "_cells_with_g_cells must NOT be probed during dry-run "
            "(closes the round-10 G-cell extension of the same short-circuit): "
            "a clean tree has no G-cells, so the probe would falsely report "
            "0/56 g-cell coverage and re-open the [phase=done] suppression."
        )

    monkeypatch.setattr(d, "_cells_with_trained_adapter", _adapter_must_not_be_called)
    monkeypatch.setattr(d, "_cells_with_g_cells", _g_must_not_be_called)
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is True, (
        "dry-run from a clean tree must report coverage_complete=True so "
        "main() fires [phase=done]; pre-r8 this returned False"
    )

    assert len(captured) == 1
    rec = captured[0]
    assert rec["kind"] == "epm:progress", (
        f"dry-run still downgrades the sentinel kind for live-poller safety; got {rec['kind']!r}"
    )
    body = json.loads(rec["note"])
    cov = body["coverage"]
    # Coverage reads as complete on BOTH axes (paper card -- nothing was
    # trained, nothing was G-evaluated, but dry-run is enumeration only).
    assert cov["complete"] is True
    assert cov["realized_adapter"] == len(planned)
    assert cov["realized_g_cells"] == len(planned)
    assert cov["planned"] == len(planned)
    assert cov["missing_adapters"] == []
    assert cov["missing_g_cells"] == []
    # Back-compat aliases.
    assert cov["realized"] == len(planned)
    assert cov["missing"] == []
    # Card advertises the FULL planned grid (nothing was actually trained,
    # but dry-run is a paper enumeration -- the live-poller-safety
    # downgrade above is what stops a real poll from draining this).
    assert len(body["reproducibility_card"]["adapter_paths"]) == len(planned)


# ── round-10: G-cell axis ───────────────────────────────────────────────────


def test_finalize_full_adapters_partial_g_cells_emits_progress(monkeypatch):
    """Closes CONCERN ``g-cell-coverage-not-in-finalize`` (reconciler r9).

    The round-9 launcher's ``gate_phase2_coverage`` tolerance
    (``MIN_PHASE2_CELLS=1``) lets the launcher continue past a Phase-2
    crash that produced only ONE G-cell. Pre-r10 ``_finalize`` checked
    only the adapter axis: full 56 adapters from Phase 1 -> declares
    success, emits ``epm:results`` + ``[phase=done]`` over an
    essentially empty G-grid. The round-10 fix is to verify BOTH axes.
    """
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    assert len(planned) >= 30, "test assumes the full sweep is >=30 cells"

    # All adapters present (Phase 1 finished cleanly), but only ONE
    # G-cell landed (round-9 launcher's MIN_PHASE2_CELLS=1 tolerance).
    g_subset = list(planned[:1])
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    monkeypatch.setattr(d, "_cells_with_g_cells", lambda cells: g_subset)
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is False, (
        "partial g-cell coverage MUST report coverage_complete=False; "
        "this is the contract that closes g-cell-coverage-not-in-finalize"
    )

    assert len(captured) == 1
    rec = captured[0]
    assert rec["kind"] == "epm:progress", (
        f"partial G-cell coverage MUST NOT emit epm:results; got {rec['kind']!r}"
    )
    body = json.loads(rec["note"])
    cov = body["coverage"]
    assert cov["complete"] is False
    # Adapter axis is FULL.
    assert cov["realized_adapter"] == len(planned)
    assert cov["missing_adapters"] == []
    # G-cell axis is PARTIAL -- the bug class this test pins.
    assert cov["realized_g_cells"] == 1
    assert len(cov["missing_g_cells"]) == len(planned) - 1
    g_realized_slugs = {d._cell_slug(*c) for c in g_subset}
    g_missing_slugs = {d._cell_slug(*c) for c in planned if c not in set(g_subset)}
    assert set(cov["missing_g_cells"]) == g_missing_slugs
    # Card still advertises the full adapter grid (HF pointers are valid
    # regardless of Phase-2 progress; the G-cell shortfall is reported
    # under ``coverage.missing_g_cells``, not by hiding adapter_paths).
    card_paths = body["reproducibility_card"]["adapter_paths"]
    assert len(card_paths) == len(planned)
    # Sanity: the one G-cell that DID land is among the planned grid.
    assert g_realized_slugs.issubset(set(card_paths.keys()))


def test_finalize_partial_both_axes_lists_both_missing_manifests(monkeypatch):
    """Mixed partial: adapter axis missing one set, G-cell axis missing another.

    Pins that ``missing_adapters`` and ``missing_g_cells`` are reported
    INDEPENDENTLY -- the two axes don't collapse to a single manifest.
    """
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    assert len(planned) >= 10, "test assumes the full sweep is >=10 cells"

    # Adapter axis missing the LAST 3; G-cell axis missing the FIRST 5.
    # The intersection (cells trained AND G-evaluated) is planned[5:-3].
    trained = list(planned[:-3])
    g_realized = list(planned[5:])
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: trained)
    monkeypatch.setattr(d, "_cells_with_g_cells", lambda cells: g_realized)
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is False
    assert captured[0]["kind"] == "epm:progress"
    cov = json.loads(captured[0]["note"])["coverage"]

    expected_missing_adapters = {d._cell_slug(*c) for c in planned[-3:]}
    expected_missing_g_cells = {d._cell_slug(*c) for c in planned[:5]}
    assert set(cov["missing_adapters"]) == expected_missing_adapters
    assert set(cov["missing_g_cells"]) == expected_missing_g_cells
    # Disjointness is the point here: the two missing manifests do NOT
    # contain overlapping cells in this construction.
    assert set(cov["missing_adapters"]) & set(cov["missing_g_cells"]) == set(), (
        "this test's construction has disjoint missing sets on the two axes"
    )
