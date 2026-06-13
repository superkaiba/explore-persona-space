"""Coverage-aware ``_finalize`` regression test (#628 r7).

Closes CONCERN ``partial-ok-results-sentinel`` (raised by reconciler r6):
the launcher's ``I628_MIN_TRAINED_CELLS=30`` salvage gate combined with
``--partial-ok`` previously let a PARTIAL Phase-1 (e.g. 30 trained cells
out of the planned 56) reach ``_finalize`` and emit a full
``epm:results`` sentinel + ``[phase=done]`` log line, even though the
downstream analyzer (``scripts/i628_analysis.py:_assert_h2_keys``)
requires the full 16-context x 2-seed (plus the 3 mini arms) H2 grid.
The analyzer-side fail-loud assert is defense-in-depth; emitting
``epm:results`` with missing cells is wrong by itself.

These tests pin the new contract:

* realized == planned → ``epm:results`` + ``[phase=done]`` (today's path).
* realized <  planned → ``epm:progress`` ONLY, with a missing-cell
  manifest under ``coverage.missing``; the reproducibility card lists
  ONLY realized cells (no phantom ``adapter_paths`` for never-trained
  cells); ``main()`` suppresses ``[phase=done]``.
* dry-run / smoke unconditionally downgrade to ``epm:progress`` (a
  live ``poll_pipeline.py`` would otherwise drain a smoke sentinel as
  real results).

No GPU, no network: ``_cells_with_trained_adapter`` is monkeypatched
to return a chosen subset, and ``write_sentinel`` is replaced with an
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


# ── happy path: full coverage ───────────────────────────────────────────────


def test_finalize_full_coverage_emits_results_and_done(monkeypatch):
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    assert len(planned) > 0, "full sweep should enumerate >0 cells"

    # All cells trained -> realized == planned.
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is True, "full coverage must report coverage_complete=True"

    assert len(captured) == 1
    rec = captured[0]
    assert rec["kind"] == "epm:results", f"full coverage must emit epm:results, got {rec['kind']!r}"
    body = json.loads(rec["note"])
    assert body["coverage"]["complete"] is True
    assert body["coverage"]["realized"] == len(planned)
    assert body["coverage"]["planned"] == len(planned)
    assert body["coverage"]["missing"] == []
    # The card advertises adapter_paths for every planned cell.
    assert len(body["reproducibility_card"]["adapter_paths"]) == len(planned)


# ── partial-coverage: the round-7 fix ───────────────────────────────────────


def test_finalize_partial_coverage_emits_progress_not_results(monkeypatch):
    """30 of 56 cells trained -> epm:progress, NOT epm:results."""
    import i628_dispatch as d

    args = _args()
    planned = d._cells(args)
    assert len(planned) >= 30, "this test assumes the full sweep is >=30 cells"

    # Reconciler's exact scenario: I628_MIN_TRAINED_CELLS=30 salvage,
    # so 30/56 cells trained, 26 missing.
    trained_subset = list(planned[:30])
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: trained_subset)
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
    assert body["coverage"]["complete"] is False
    assert body["coverage"]["realized"] == 30
    assert body["coverage"]["planned"] == len(planned)
    assert len(body["coverage"]["missing"]) == len(planned) - 30
    # Card advertises ONLY realized cells -- never a phantom adapter_path
    # for a never-trained cell (Codex r6 _finalize coverage-summary finding).
    card_paths = body["reproducibility_card"]["adapter_paths"]
    assert len(card_paths) == 30
    realized_slugs = {d._cell_slug(*c) for c in trained_subset}
    assert set(card_paths.keys()) == realized_slugs
    # And the missing manifest lists the OTHER cells.
    missing_slugs = {d._cell_slug(*c) for c in planned[30:]}
    assert set(body["coverage"]["missing"]) == missing_slugs


def test_finalize_zero_realized_emits_progress(monkeypatch):
    """Pathological 0-cell case still degrades cleanly to epm:progress."""
    import i628_dispatch as d

    args = _args()
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: [])
    captured = _capture_sentinels(monkeypatch, d)

    complete = d._finalize(args)
    assert complete is False
    assert len(captured) == 1
    assert captured[0]["kind"] == "epm:progress"
    body = json.loads(captured[0]["note"])
    assert body["coverage"]["realized"] == 0
    assert body["reproducibility_card"]["adapter_paths"] == {}


# ── smoke / dry-run downgrade unconditionally (pre-existing contract) ───────


def test_finalize_smoke_downgrades_even_on_full_coverage(monkeypatch):
    import i628_dispatch as d

    args = _args(smoke=True)
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
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
    monkeypatch.setattr(d, "_cells_with_trained_adapter", lambda cells: list(cells))
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

    # Clean tree: NO adapters exist on disk. Pre-r8 this would have made
    # _finalize report coverage_complete=False and main() would suppress
    # [phase=done]. The fix is that _finalize must NOT call this on a
    # dry-run -- the short-circuit replaces the probe entirely. We use a
    # sentinel that raises if reached so the test is unambiguous.
    def _must_not_be_called(cells):
        raise AssertionError(
            "_cells_with_trained_adapter must NOT be probed during dry-run "
            "(closes dry-run-finalize-suppresses-done): the disk probe is "
            "the bug -- short-circuit before it."
        )

    monkeypatch.setattr(d, "_cells_with_trained_adapter", _must_not_be_called)
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
    # Coverage reads as complete (paper card -- nothing was trained).
    assert body["coverage"]["complete"] is True
    assert body["coverage"]["realized"] == len(planned)
    assert body["coverage"]["planned"] == len(planned)
    assert body["coverage"]["missing"] == []
    # Card advertises the FULL planned grid (nothing was actually trained,
    # but dry-run is a paper enumeration -- the live-poller-safety
    # downgrade above is what stops a real poll from draining this).
    assert len(body["reproducibility_card"]["adapter_paths"]) == len(planned)
