"""Coverage-aware ``_finalize`` regression test (#628 r7 / r10 / r11).

Closes three CONCERNs:

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

r11 — ``g-cell-column-coverage-not-in-finalize`` (reconciler r10): the
round-10 G-cell-axis check accepted a cell as G-realized when ANY ONE
column file existed, so a Phase-2 crash that landed a single G-cell
per adapter (the ``MIN_PHASE2_CELLS=1`` tolerance multiplied across
cells) silently passed coverage_complete despite a ~34x-incomplete
G-grid. The round-11 fix replaces the cell-axis predicate with a
comprehensive (arm x train_cid x eval_cid x seed x slot_variant) file
enumeration, mirroring the producer side in ``phase2`` and aligned with
the analyzer's ``_assert_h2_keys`` expected grid.

The analyzer-side fail-loud assert remains defense-in-depth; emitting
``epm:results`` with missing cells (on EITHER axis) is wrong by itself.

These tests pin the round-7-through-r11 contract:

* adapters == planned AND every expected G-file present → ``epm:results``
  + ``[phase=done]`` (full coverage path).
* adapters < planned OR any expected G-file missing → ``epm:progress``
  ONLY, with ``coverage.missing_adapters``, ``coverage.missing_g_files``,
  and ``coverage.missing_g_cells`` manifests; the reproducibility card
  lists ONLY adapter-realized cells; ``main()`` suppresses ``[phase=done]``.
* dry-run / smoke unconditionally downgrade to ``epm:progress`` (a
  live ``poll_pipeline.py`` would otherwise drain a smoke sentinel as
  real results).

No GPU, no network: ``_cells_with_trained_adapter``,
``_required_g_cell_files``, and ``_missing_g_files`` are monkeypatched
to return chosen subsets, and ``write_sentinel`` is replaced with an
in-memory recorder.
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


def _patch_g_files_for_realized_cells(monkeypatch, mod, args, realized_cells):
    """Make ``_missing_g_files`` return the expected files for ALL cells
    NOT in ``realized_cells`` -- i.e. realized cells have ALL files present.

    This is the round-10 "cell-axis" semantics expressed in the round-11
    file-axis API: a cell is "G-realized" iff every expected file under
    it exists. Lets the round-7 + r10 tests drive the same logical
    scenarios under the new file-axis check.

    Only ``_missing_g_files`` is monkeypatched -- ``_required_g_cell_files``
    is pure (no disk I/O, no network) so ``_finalize`` calls the real
    function for the expected-grid enumeration, and the test patches
    only the disk-probe layer.
    """
    realized_set = set(realized_cells)
    # Bucket the real expected grid once so the patched ``_missing_g_files``
    # is O(unrealized_cells) per call.
    expected_by_cell = _group_files_by_cell(mod._required_g_cell_files(mod._cells(args), args))

    def fake_missing(cells, _args):
        # Every file under a non-realized cell is "missing"; every
        # realized cell has zero missing files.
        out: list[Path] = []
        for c in cells:
            if c in realized_set:
                continue
            out.extend(expected_by_cell.get(c, []))
        return out

    monkeypatch.setattr(mod, "_missing_g_files", fake_missing)


def _group_files_by_cell(expected_files):
    """Reverse ``_required_g_cell_files``: bucket each Path back to (arm, train, seed)."""
    by_cell: dict[tuple[str, str, int], list[Path]] = {}
    for p in expected_files:
        arm = p.parent.name
        stem = p.stem.removesuffix("__plain")
        train_cid, _eval_cid, seed_part = stem.split("__")
        seed = int(seed_part.removeprefix("seed"))
        by_cell.setdefault((arm, train_cid, seed), []).append(p)
    return by_cell


# ── happy path: full coverage ───────────────────────────────────────────────


def test_finalize_full_coverage_emits_results_and_done(monkeypatch):
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    assert len(planned) > 0, "full sweep should enumerate >0 cells"

    # All cells trained AND every expected G-file present -> both axes full.
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    _patch_g_files_for_realized_cells(monkeypatch, d, args, planned)
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
    # Adapter axis + cell-axis rollup.
    assert cov["realized_adapter"] == len(planned)
    assert cov["realized_g_cells"] == len(planned)
    assert cov["missing_adapters"] == []
    assert cov["missing_g_cells"] == []
    # File-axis fields (round-11).
    assert cov["expected_g_files"] > 0
    assert cov["realized_g_files"] == cov["expected_g_files"]
    assert cov["missing_g_files_count"] == 0
    assert cov["missing_g_files"] == []
    assert cov["missing_g_files_truncated"] is False
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
    # the trained set -- here we pin it equal to the trained set, with
    # every expected file for those 30 cells on disk.
    trained_subset = list(planned[:30])
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: trained_subset)
    _patch_g_files_for_realized_cells(monkeypatch, d, args, trained_subset)
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
    # File-axis: every file under the 26 unrealized cells is missing.
    assert cov["missing_g_files_count"] > 0
    assert cov["missing_g_files_count"] == cov["expected_g_files"] - cov["realized_g_files"]
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
    _patch_g_files_for_realized_cells(monkeypatch, d, args, [])
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
    # Every expected G-file is missing.
    assert cov["realized_g_files"] == 0
    assert cov["missing_g_files_count"] == cov["expected_g_files"]
    assert body["reproducibility_card"]["adapter_paths"] == {}


# ── smoke / dry-run downgrade unconditionally (pre-existing contract) ───────


def test_finalize_smoke_downgrades_even_on_full_coverage(monkeypatch):
    import i628_dispatch as d

    args = _args(smoke=True)
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    _patch_g_files_for_realized_cells(monkeypatch, d, args, d._cells(args))
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
    _patch_g_files_for_realized_cells(monkeypatch, d, args, d._cells(args))
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

    Post-r11: the short-circuit covers the file-axis probe too. A clean
    tree has no G-cell files; if ``_missing_g_files`` were probed,
    coverage_complete would be False and [phase=done] would re-suppress.
    """
    import i628_dispatch as d

    args = _args(dry_run=True)
    planned = d._cells(args)
    assert len(planned) > 0, "full sweep should enumerate >0 cells"

    # Clean tree: NO adapters AND no G-cell files exist on disk. Pre-r8
    # the adapter probe would have returned [] and main() would suppress
    # [phase=done]. The r8 fix short-circuited the adapter probe for
    # dry-run; the r10 G-cell-axis check + r11 G-file-axis check both
    # ride the SAME short-circuit (otherwise the dry-run regression
    # would re-open on the new axis -- a clean tree has neither
    # artifact). Use sentinels that raise if reached so the test is
    # unambiguous on ALL probes.
    def _adapter_must_not_be_called(cells):
        raise AssertionError(
            "_cells_with_trained_adapter must NOT be probed during dry-run "
            "(closes dry-run-finalize-suppresses-done): the disk probe is "
            "the bug -- short-circuit before it."
        )

    def _required_must_not_be_called(cells, _args):
        raise AssertionError(
            "_required_g_cell_files must NOT be probed during dry-run "
            "(closes the round-11 G-file extension of the same short-circuit): "
            "a clean tree has no G-cell files, so the probe would falsely "
            "report 0/3264 g-file coverage and re-open [phase=done] suppression."
        )

    def _missing_must_not_be_called(cells, _args):
        raise AssertionError(
            "_missing_g_files must NOT be probed during dry-run "
            "(closes the round-11 G-file extension of the same short-circuit)."
        )

    monkeypatch.setattr(d, "_cells_with_trained_adapter", _adapter_must_not_be_called)
    monkeypatch.setattr(d, "_required_g_cell_files", _required_must_not_be_called)
    monkeypatch.setattr(d, "_missing_g_files", _missing_must_not_be_called)
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
    # File-axis short-circuit: no probe ran, so report nothing missing.
    assert cov["missing_g_files_count"] == 0
    assert cov["missing_g_files"] == []
    # Back-compat aliases.
    assert cov["realized"] == len(planned)
    assert cov["missing"] == []
    # Card advertises the FULL planned grid (nothing was actually trained,
    # but dry-run is a paper enumeration -- the live-poller-safety
    # downgrade above is what stops a real poll from draining this).
    assert len(body["reproducibility_card"]["adapter_paths"]) == len(planned)


# ── round-10: G-cell axis (one cell w/ ALL files vs everyone else empty) ────


def test_finalize_full_adapters_partial_g_cells_emits_progress(monkeypatch):
    """Closes CONCERN ``g-cell-coverage-not-in-finalize`` (reconciler r9).

    The round-9 launcher's ``gate_phase2_coverage`` tolerance
    (``MIN_PHASE2_CELLS=1``) lets the launcher continue past a Phase-2
    crash that produced only ONE G-cell. Pre-r10 ``_finalize`` checked
    only the adapter axis: full 56 adapters from Phase 1 -> declares
    success, emits ``epm:results`` + ``[phase=done]`` over an
    essentially empty G-grid. Round-10 verified the cell-axis; round-11
    upgrades that to the file-axis (every column AND every slot variant
    for every cell). This test pins the round-10 scenario under the
    round-11 file-axis API: cell ``planned[0]`` has ALL its expected
    files present, every other cell has ZERO.
    """
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    assert len(planned) >= 30, "test assumes the full sweep is >=30 cells"

    # All adapters present (Phase 1 finished cleanly), but only ONE
    # cell has any G-cell files at all -- and that one cell has every
    # expected file for itself. Everyone else is empty.
    g_subset = list(planned[:1])
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    _patch_g_files_for_realized_cells(monkeypatch, d, args, g_subset)
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
    # Cell-axis rollup: only the 1 cell is fully G-realized.
    assert cov["realized_g_cells"] == 1
    assert len(cov["missing_g_cells"]) == len(planned) - 1
    g_realized_slugs = {d._cell_slug(*c) for c in g_subset}
    g_missing_slugs = {d._cell_slug(*c) for c in planned if c not in set(g_subset)}
    assert set(cov["missing_g_cells"]) == g_missing_slugs
    # File-axis: missing == all files for the 55 unrealized cells.
    assert cov["missing_g_files_count"] > 0
    # Card still advertises the full adapter grid (HF pointers are valid
    # regardless of Phase-2 progress; the G-shortfall is reported
    # under ``coverage.missing_g_files`` / ``missing_g_cells``, not by
    # hiding adapter_paths).
    card_paths = body["reproducibility_card"]["adapter_paths"]
    assert len(card_paths) == len(planned)
    # Sanity: the one G-realized cell is among the planned grid.
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
    _patch_g_files_for_realized_cells(monkeypatch, d, args, g_realized)
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


# ── round-11: G-file axis (column-level coverage) ──────────────────────────


def test_finalize_full_adapters_one_column_per_cell_emits_progress(monkeypatch):
    """Closes CONCERN ``g-cell-column-coverage-not-in-finalize`` (reconciler r10).

    The round-10 ``_cells_with_g_cells`` accepted a cell as G-realized
    when ANY ONE column file existed. So a Phase-2 crash that landed
    one G-cell JSON per adapter (across all 56 adapters) would have
    passed round-10's predicate AND emitted ``epm:results`` over a
    ~34x incomplete grid (the analyzer's ``_assert_h2_keys`` would
    catch it downstream -- the sentinel was wrong by itself).

    Round-11 enumerates the FULL file grid (arm x train x eval x seed
    x slot_variant) and requires every file. This test drives the
    exact "one column landed per adapter" scenario the round-10
    predicate would have missed and confirms round-11 catches it.
    """
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    expected_all = d._required_g_cell_files(planned, args)
    by_cell = _group_files_by_cell(expected_all)

    # Adapters: FULL. G-files: keep exactly ONE file per cell (the first
    # one the producer would have written, deterministically).
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))

    def fake_missing(cells, _args):
        missing: list[Path] = []
        for c in cells:
            files = by_cell.get(c, [])
            # Drop file index 0; everything else is missing.
            missing.extend(files[1:])
        return missing

    monkeypatch.setattr(d, "_missing_g_files", fake_missing)
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is False, (
        "1-column-per-adapter is NOT full coverage; the round-10 predicate "
        "accepted this scenario and would have emitted epm:results -- "
        "this test pins the round-11 file-axis check that rejects it."
    )

    assert len(captured) == 1
    rec = captured[0]
    assert rec["kind"] == "epm:progress", (
        f"1-column-per-adapter partial coverage MUST NOT emit epm:results; "
        f"got {rec['kind']!r} (round-10 bug: cell-axis predicate accepts this)"
    )
    body = json.loads(rec["note"])
    cov = body["coverage"]
    assert cov["complete"] is False
    # Adapter axis is FULL (no missing adapters).
    assert cov["realized_adapter"] == len(planned)
    assert cov["missing_adapters"] == []
    # File-axis is PARTIAL: expected - (1 per cell) files missing.
    assert cov["expected_g_files"] == len(expected_all)
    assert cov["realized_g_files"] == len(planned)  # exactly 1 per cell
    assert cov["missing_g_files_count"] == len(expected_all) - len(planned)
    assert cov["missing_g_files_count"] > 0
    # Cell-axis rollup: EVERY cell has at least one missing file in this
    # construction (we kept only the first file per cell, but most cells
    # have many expected files).
    assert len(cov["missing_g_cells"]) == sum(1 for c in planned if len(by_cell[c]) > 1)


def test_finalize_full_adapters_missing_slot_variant_emits_progress(monkeypatch):
    """Round-11 file-axis: sep arms need BOTH sep_mode=marker AND sep_mode=plain.

    The producer side (``phase2`` lines 1995-2016) writes one file per
    ``(arm, train_cid, seed, eval_cid)`` cell at ``sep_mode="marker"``;
    for sep arms it ALSO writes a ``__plain`` shadow. A Phase-2 crash
    between the two ``_cell_read`` calls would leave the ``__plain``
    file missing; the round-10 cell-axis predicate accepted this as
    coverage (one file exists per (cid, eval_cid, seed) glob), so
    coverage_complete would have been True. Round-11 enumerates per
    slot variant and catches the asymmetric slot dropout.
    """
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    expected_all = d._required_g_cell_files(planned, args)

    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))

    # Drop every ``__plain`` shadow file; keep every ``__marker`` slot.
    # Only sep arms produce ``__plain`` files, so this isolates the
    # asymmetric-slot dropout case the round-10 predicate could not see.
    def fake_missing(cells, _args):
        return [p for p in d._required_g_cell_files(cells, _args) if p.stem.endswith("__plain")]

    monkeypatch.setattr(d, "_missing_g_files", fake_missing)
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is False, (
        "Asymmetric slot dropout (every cell has marker slot but sep arms "
        "are missing the __plain shadow) is partial coverage; round-10 "
        "would have missed this because at least one file exists per cell."
    )

    assert len(captured) == 1
    rec = captured[0]
    assert rec["kind"] == "epm:progress"
    body = json.loads(rec["note"])
    cov = body["coverage"]
    # File-axis catches the slot-variant gap: every plain shadow is missing.
    plain_count = sum(1 for p in expected_all if p.stem.endswith("__plain"))
    assert plain_count > 0, (
        "test invariant: the full sweep must include sep arms so __plain "
        "shadows exist; if this fires, the fresh-arms list lost a sep arm."
    )
    assert cov["missing_g_files_count"] == plain_count
    assert cov["realized_g_files"] == cov["expected_g_files"] - plain_count
    # Sample: at least one missing file path ends with __plain.json.
    assert any(p.endswith("__plain.json") for p in cov["missing_g_files"])
    # Cell-axis rollup: every sep-arm cell appears in missing_g_cells
    # (its ``__plain`` shadow is gone), every nosep cell does NOT.
    # We don't pin the exact membership here; the row-counts are
    # already tight enough to pin the bug, and the analyzer's
    # ``_assert_h2_keys`` is the downstream defense-in-depth on this.
    assert len(cov["missing_g_cells"]) > 0


def test_finalize_missing_g_files_manifest_caps_at_200(monkeypatch):
    """The ``missing_g_files`` list is capped at 200 entries.

    Full default sweep enumerates ~3264 files; if Phase-2 fully crashes
    every path becomes "missing" and the sentinel ``note`` (capped at
    50,000 chars by ``task.py post-marker``) would blow up. The cap
    preserves the total ``missing_g_files_count`` and the first 200
    entries so a downstream consumer can pinpoint a failure.
    """
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    expected_all = d._required_g_cell_files(planned, args)
    assert len(expected_all) > 200, (
        "this test assumes the default sweep enumerates >200 expected files"
    )

    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    # Every file is missing.
    _patch_g_files_for_realized_cells(monkeypatch, d, args, [])
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is False
    body = json.loads(captured[0]["note"])
    cov = body["coverage"]
    # The total count surfaces the full shortfall.
    assert cov["missing_g_files_count"] == len(expected_all)
    # The list itself is capped at 200, and the truncated flag is set.
    assert len(cov["missing_g_files"]) == 200
    assert cov["missing_g_files_truncated"] is True


def test_required_g_cell_files_grid_shape(monkeypatch):
    """Cross-check: ``_required_g_cell_files`` enumerates the same per-cell
    grid the producer side in ``phase2`` writes -- one file per
    (cell, eval_cid) for nosep arms, two for sep arms.
    """
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    expected = d._required_g_cell_files(planned, args)
    columns = d._eval_columns(args)

    # Counts: per cell, len(columns) for nosep + 2 * len(columns) for sep.
    sep_cells = sum(1 for arm, _, _ in planned if d._sep_variant(arm) == "sep")
    nosep_cells = len(planned) - sep_cells
    assert len(expected) == nosep_cells * len(columns) + sep_cells * 2 * len(columns)

    # Every file has the producer-side filename shape, parent = arm,
    # and the cell triple round-trips through the stem split.
    by_cell = _group_files_by_cell(expected)
    assert set(by_cell.keys()) == set(planned), (
        "_required_g_cell_files must cover EXACTLY the planned cell grid"
    )
    for c, files in by_cell.items():
        arm, _train, _seed = c
        n_expected = 2 * len(columns) if d._sep_variant(arm) == "sep" else len(columns)
        assert len(files) == n_expected, f"cell {c} expected {n_expected} files, got {len(files)}"
