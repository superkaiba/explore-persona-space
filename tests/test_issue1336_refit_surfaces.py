"""Pins for scripts/issue1336_refit_surfaces.sh (the Round-B per-surface driver).

The driver hand-lists all 32 cell keys so it can stage ONE ladder surface at a
time (the whole-run staging set is 317.76 GB, past the 240 GB RunPod CPU
container-disk cap). A typo'd, duplicated, or dropped key would silently lose
cells from the refit with no error — the fitting script would just fit fewer
cells and exit 0 — so the key list is pinned against the script's own
``enumerate_cells()`` rather than trusted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue1336_selfmap_missing_pairs as sm  # noqa: E402

_DRIVER = _REPO_ROOT / "scripts" / "issue1336_refit_surfaces.sh"
_ROW_RE = re.compile(r'^"([^|]+)\|(\d+)\|(.+)"$', re.M)


def _rows() -> list[tuple[str, int, list[str]]]:
    text = _DRIVER.read_text()
    rows = [(label, int(gb), cells.split(",")) for label, gb, cells in _ROW_RE.findall(text)]
    assert rows, "no SURFACES rows parsed — the array shape changed, update this pin"
    return rows


def test_driver_covers_every_cell_exactly_once():
    """The union of the driver's per-surface cell lists == enumerate_cells(), no dups."""
    valid = {f"{s}__{t}__{f}__{c}" for s, t, f, c in sm.enumerate_cells()}
    listed: list[str] = []
    for _, _, cells in _rows():
        listed.extend(cells)
    assert not (set(listed) - valid), f"unknown cell keys: {sorted(set(listed) - valid)}"
    assert not (valid - set(listed)), f"cells never driven: {sorted(valid - set(listed))}"
    assert len(listed) == len(set(listed)), "a cell key is listed on two surfaces"
    assert len(listed) == len(valid) == 32


def test_each_surface_row_is_one_surface_of_four_cells():
    """One row per ladder surface; each row's 4 cells share that row's (format, corpus)."""
    rows = _rows()
    assert len(rows) == len(sm.cm.v2_surfaces()) == 8
    row_surfaces = set()
    for label, _, cells in rows:
        assert len(cells) == 4, f"{label}: {len(cells)} cells, expected 4"
        parsed = [tuple(k.split("__")) for k in cells]
        surfaces = {(fmt, corpus) for _, _, fmt, corpus in parsed}
        assert len(surfaces) == 1, f"{label}: mixes surfaces {surfaces} — breaks per-surface reap"
        (surface,) = surfaces
        # The row LABEL must name the surface it stages, else the disk assert and
        # the reap log attribute bytes to the wrong surface.
        assert label == f"{surface[0]}/{surface[1]}", f"label {label!r} != {surface}"
        row_surfaces.add(surface)
        pairs = {(s, t) for s, t, _, _ in parsed}
        assert pairs == {
            ("base", "base"),
            ("sft", "rlvr"),
            ("sft", "rlvr_long"),
            ("rlvr", "rlvr_long"),
        }, f"{label}: unexpected pair set {pairs}"
    assert row_surfaces == {(f, c) for c, f in sm.cm.v2_surfaces()}


def test_surface_rows_are_ordered_cheapest_first():
    """Cheapest-first ordering is load-bearing: the measured per-cell wall lands
    early and the small surfaces are durable before the 70 GB concat pulls."""
    gbs = [gb for _, gb, _ in _rows()]
    assert gbs == sorted(gbs), f"surface rows not cheapest-first: {gbs}"


def test_reap_never_targets_durable_output():
    """The reap clears staged Hub copies only — never a generated store/ or
    eval_results/ (the disk-hygiene contract), and never the kept gen/ answers."""
    text = _DRIVER.read_text()
    (reap_line,) = [ln for ln in text.splitlines() if ln.strip().startswith("rm -rf ")]
    targets = reap_line.split("rm -rf ", 1)[1].split()
    assert targets, "reap line parsed no targets"
    for t in targets:
        assert t.startswith('"$STAGE/'), f"reap target outside the stage root: {t}"
        leaf = t.strip('"').split("/")[-1]
        assert leaf in {"turnstore_v2", "turnstore_wave1", "selfmap_stage_tmp"}, (
            f"unexpected reap target {leaf!r} — store/, eval_results/ and gen/ must survive"
        )
