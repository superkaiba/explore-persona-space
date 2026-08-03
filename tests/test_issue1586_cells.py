"""#1586 CPU-testable core pins (review r1 Minor 8).

Covers the round-1 reviewer probes (selection tie-breaks / fallback /
eligibility gating, dose labels, parse_ft_cell) PLUS the round-2 permanent
invariants: the disk-mode resolve-once-persist regime stability (review r1
Majors 1+2 — fails pre-fix: the pre-fix ``resolved_disk_mode`` re-probed
statvfs on every call, so a free-space drift flipped ``regime_key``), the
RunPod-lane stream-reap detection (Major 1), and the batched
``_mu_norm_draws`` vs a serial reference (Major 5; vectorize rule item 6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1586_cells as G  # noqa: E402

# ── grid registry / parse ────────────────────────────────────────────────────


def test_parse_ft_cell_roundtrip():
    assert len(G.ALL_FT_CELLS) == 16
    for cell in G.ALL_FT_CELLS:
        beh, regime, seed = G.parse_ft_cell(cell)
        assert G.ft_cell_id(beh, regime, seed) == cell
        arm = G.lora_pair_of(cell)
        assert arm.cell == f"{beh}-pers-lora-{regime}-s{seed}"


@pytest.mark.parametrize(
    "bad",
    [
        "syc-pers-lora-con-s42",  # lora id is not an FT cell
        "syc-pers-ft-con-s7",  # unknown seed
        "syc-pers-ft-zz-s42",  # unknown regime
        "zz-pers-ft-con-s42",  # unknown behavior
        "not-a-cell",
    ],
)
def test_parse_ft_cell_fail_loud(bad):
    with pytest.raises(ValueError):
        G.parse_ft_cell(bad)


# ── anchor-nearest selection (plan §4.3) ─────────────────────────────────────


def test_select_in_band_anchor_nearest():
    lo, hi = G.JUDGED_RATE_BAND
    anchor = (lo + hi) / 2
    metric = {2: lo + 0.01, 6: anchor + 0.01, 10: hi - 0.001}
    sel = G.select_anchor_nearest(metric, anchor=anchor, band=(lo, hi))
    assert sel["step"] == 6 and sel["in_band"] and sel["fallback"] is None


def test_select_tie_breaks_earliest():
    sel = G.select_anchor_nearest(
        {2: 0.70, 4: 0.60}, anchor=0.65, band=(0.60, 0.85)
    )  # equal 0.05 gaps -> earliest step wins (float-robust rounding)
    assert sel["step"] == 2 and sel["in_band"]


def test_select_fallback_closest_approach():
    lo, hi = G.JUDGED_RATE_BAND
    sel = G.select_anchor_nearest(
        {2: lo - 0.30, 8: lo - 0.05, 12: hi + 0.20}, anchor=lo, band=(lo, hi)
    )
    assert sel["step"] == 8
    assert not sel["in_band"] and sel["fallback"] == "closest_approach"
    assert sel["gate_failed_all"] is False


def test_select_eligibility_gating_and_gate_failed_all():
    lo, hi = G.JUDGED_RATE_BAND
    anchor = (lo + hi) / 2
    metric = {2: anchor, 4: anchor + 0.02}
    # step 2 gated out -> the eligible in-band step wins despite a worse gap
    sel = G.select_anchor_nearest(metric, anchor=anchor, band=(lo, hi), eligible_steps={4})
    assert sel["step"] == 4 and sel["in_band"]
    # every step gated out -> closest-approach over ALL rungs, flagged
    sel = G.select_anchor_nearest(metric, anchor=anchor, band=(lo, hi), eligible_steps=set())
    assert sel["fallback"] == "closest_approach" and sel["gate_failed_all"] is True


# ── dose-match labels (plan §4.3) ────────────────────────────────────────────


def test_content_dose_label():
    lo, hi = G.JUDGED_RATE_BAND
    arm = G.REUSED_LORA_ARMS[0]
    in_band_anchor = min(max(arm.anchor, lo), hi)
    lab = G.content_dose_label(in_band_anchor, arm)
    assert lab["rate_gap"] == pytest.approx(abs(in_band_anchor - arm.anchor))
    far = G.content_dose_label(hi + 0.5, arm)
    assert far["dose_matched"] is False


def test_marker_dose_label():
    lo, hi = G.INSTALL_WINDOW
    arm = next(a for a in G.REUSED_LORA_ARMS if a.recipe_class == "marker")
    mid = (lo + hi) / 2
    near = G.marker_dose_label(min(max(arm.anchor, lo), hi), arm)
    assert near["gap_nats"] >= 0
    far = G.marker_dose_label(mid + 3 * (hi - lo), arm)
    assert far["dose_matched"] is False


# ── disk-mode resolve-once-persist (review r1 Majors 1+2) ────────────────────


def _mk_cfg(tmp_path, **kw):
    import issue1586_dispatch as d

    return d.Cfg(smoke=True, cells=(G.SMOKE_CELL,), out_root=Path(tmp_path), **kw)


def test_probe_disk_mode_runpod_env_marker(monkeypatch, tmp_path):
    import issue1586_dispatch as d

    monkeypatch.setenv("RUNPOD_POD_ID", "pod-test")
    assert d.probe_disk_mode(Path(tmp_path)) == "stream-reap"


def test_runpod_fuse_mount_detection(monkeypatch, tmp_path):
    """statvfs-blind MooseFS lane: a fuse /workspace row alone (no env
    marker) must read as the RunPod quota lane (review r1 Major 1)."""
    import issue1586_dispatch as d

    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    mounts = tmp_path / "mounts"
    mounts.write_text("overlay / overlay rw 0 0\nmfs#10.1.1.1:9421 /workspace fuse.mfs rw 0 0\n")
    real_open = open

    def fake_open(path, *a, **kw):
        if str(path) == "/proc/mounts":
            return real_open(mounts, *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    assert d._runpod_workspace_quota_lane() is True
    assert d.probe_disk_mode(Path(tmp_path)) == "stream-reap"


def test_regime_key_stable_across_free_space_drift(monkeypatch, tmp_path):
    """Major 2 pin (fails pre-fix): the auto disk mode resolves ONCE, is
    persisted, and later regime_key() calls / fresh resumes on the same
    out_root can never flip it when free space drifts across 300 GB."""
    import issue1586_dispatch as d

    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr(d, "probe_disk_mode", lambda p, **kw: "keep-cell")
    k1 = cfg.regime_key()
    assert k1["ladder_disk_mode"] == "keep-cell"
    assert (Path(tmp_path) / "disk_mode.json").exists()
    # free space "drifts" below the threshold: a re-probe would now flip —
    # the persisted value must win (no re-probe at all).
    monkeypatch.setattr(
        d,
        "probe_disk_mode",
        lambda p, **kw: pytest.fail("re-probed after persist (Major 2 regression)"),
    )
    assert cfg.regime_key() == k1
    # a FRESH Cfg on the same out_root (resume / unit subprocess) reads the
    # persisted literal, not a fresh probe.
    cfg2 = _mk_cfg(tmp_path)
    assert cfg2.resolved_disk_mode() == "keep-cell"


def test_unit_args_thread_resolved_literal_never_auto(monkeypatch, tmp_path):
    import issue1586_dispatch as d

    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr(d, "probe_disk_mode", lambda p, **kw: "stream-reap")
    args = d._unit_args(cfg, "ladder", G.SMOKE_CELL)
    mode = args[args.index("--ladder-disk-mode") + 1]
    assert mode == "stream-reap"  # the resolved LITERAL, never "auto"


def test_explicit_disk_mode_passes_through(tmp_path):
    cfg = _mk_cfg(tmp_path, ladder_disk_mode="keep-cell")
    assert cfg.resolved_disk_mode() == "keep-cell"
    assert not (Path(tmp_path) / "disk_mode.json").exists()  # no probe, no persist


# ── batched mu-norm draws vs serial reference (review r1 Major 5) ───────────


def test_mu_norm_draws_matches_serial_reference():
    import issue1586_geometry as gg

    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 5))
    idx = rng.integers(0, 9, size=(13, 9))
    batched = gg._mu_norm_draws(X, idx)
    serial = np.array([np.linalg.norm(X[draw].mean(axis=0)) for draw in idx])
    np.testing.assert_allclose(batched, serial, rtol=1e-12, atol=1e-12)


# ── mix registry + regime-aware composition asserts (r4 crash fix) ──────────
# epm:failure v4: MIXES['mk']['po'] copy-filled the con row's 1000 (the real
# po mix = the con positives, 200 rows), and the marker sub-check's
# 0 < n_marker < n was structurally unsatisfiable for a po mix (200/200
# marker rows is correct-by-construction there, not degenerate).


def _write_mix(tmp_path, name: str, completions: list[str]) -> Path:
    import json

    p = Path(tmp_path) / name
    p.write_text(
        "".join(json.dumps({"completion": c}) + "\n" for c in completions),
        encoding="utf-8",
    )
    return p


def test_mix_registry_mk_po_expected_rows():
    # fails pre-fix (registry said 1000): po = con's positives BY CONSTRUCTION.
    assert G.MIXES["mk"]["con"][1] == 1000
    assert G.MIXES["mk"]["po"][1] == 200
    # content mixes untouched (pod A's p0 PASSED these on the real files).
    for beh, con_n, po_n in (("syc", 80, 60), ("imp", 80, 60), ("cas", 80, 60)):
        assert G.MIXES[beh]["con"][1] == con_n
        assert G.MIXES[beh]["po"][1] == po_n


def test_mix_composition_con_mixed_passes(tmp_path):
    import issue1586_dispatch as d

    mk = "answer" + d.MARKER_TEXT
    p = _write_mix(tmp_path, "mk_con.jsonl", [mk, "plain", mk, "plain"])
    rec = d._assert_mix_composition(p, "mk", "con", 4)
    assert rec["rows"] == 4 and rec["rows_with_marker"] == 2


def test_mix_composition_po_all_marker_passes(tmp_path):
    # fails pre-fix: the regime-blind 0 < n_marker < n check raised
    # 'degenerate marker rows 4/4' on a correct positives-only mix.
    import issue1586_dispatch as d

    mk = "answer" + d.MARKER_TEXT
    p = _write_mix(tmp_path, "mk_po.jsonl", [mk, mk, mk, mk])
    rec = d._assert_mix_composition(p, "mk", "po", 4)
    assert rec["rows"] == 4 and rec["rows_with_marker"] == 4


def test_mix_composition_wrong_count_raises(tmp_path):
    import issue1586_dispatch as d

    mk = "answer" + d.MARKER_TEXT
    p = _write_mix(tmp_path, "mk_po.jsonl", [mk, mk])
    with pytest.raises(RuntimeError, match=r"2 rows != expected 4"):
        d._assert_mix_composition(p, "mk", "po", 4)


def test_mix_composition_con_degenerate_raises(tmp_path):
    import issue1586_dispatch as d

    mk = "answer" + d.MARKER_TEXT
    all_marker = _write_mix(tmp_path, "mk_con_all.jsonl", [mk, mk, mk, mk])
    with pytest.raises(RuntimeError, match="degenerate marker rows 4/4"):
        d._assert_mix_composition(all_marker, "mk", "con", 4)
    no_marker = _write_mix(tmp_path, "mk_con_none.jsonl", ["a", "b", "c", "d"])
    with pytest.raises(RuntimeError, match="degenerate marker rows 0/4"):
        d._assert_mix_composition(no_marker, "mk", "con", 4)


def test_mix_composition_po_partial_marker_raises(tmp_path):
    import issue1586_dispatch as d

    mk = "answer" + d.MARKER_TEXT
    p = _write_mix(tmp_path, "mk_po_partial.jsonl", [mk, "plain", mk, mk])
    with pytest.raises(RuntimeError, match="positive-only mix must carry the marker"):
        d._assert_mix_composition(p, "mk", "po", 4)


def test_mix_composition_unknown_regime_raises(tmp_path):
    import issue1586_dispatch as d

    mk = "answer" + d.MARKER_TEXT
    p = _write_mix(tmp_path, "mk_zz.jsonl", [mk, "plain"])
    with pytest.raises(RuntimeError, match="unknown regime"):
        d._assert_mix_composition(p, "mk", "zz", 2)
