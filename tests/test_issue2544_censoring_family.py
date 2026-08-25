"""#2544 censoring-family fail-fast + schema pins (r3 reconcile blocker
``censoring-family-failsoft``).

Self-contained (synthetic fixtures in tmp_path; no eval_results reads, no
network — every rollout fetch resolves on the LOCAL branch of the real
``fetch_rollout``). Pins:

1. ``_trunc_masks`` FAIL-FAST: a corrupt rollout row (missing the
   ``truncated`` field) RAISES — the P4b censoring path never degrades to
   status-only / silently-absent §6.5 fields.
2. ``_common_status_reads`` schema: ``delta_nt``/``delta_tt`` are emitted
   for EVERY rung with ``n_rows`` ALWAYS present (kshot_curve.json's
   per-rung entries are built from exactly this helper); ``delta``/``ci``
   only at n >= 3.
3. ``_stratified_sb`` (truncation-stratified ceiling): per-stratum counts
   always; partition keyed on the seed-42 arm mask over aligned reliability
   rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# issue2544_fits MUST import first: it imports issue2544_common, which sets
# the ladder env BEFORE issue1902_common binds its constants (alphabetical
# sorting would freeze the un-widened #1902 ladder).
# isort: off
import issue2544_fits as F2  # noqa: E402
import issue1902_fits as F1  # noqa: E402

# isort: on

RUNGS = ["r0", "main"]
IDS = [f"ctx_{k:03d}" for k in range(8)]


def _write_rollouts(
    out_root: Path,
    truncated: dict[tuple[str, str], set[str]],
    corrupt_cell: tuple[str, str] | None = None,
) -> None:
    """Local gen/<rung>/<arm>.jsonl rollouts (the real fetch_rollout's local
    branch). ``corrupt_cell`` drops the ``truncated`` field from ONE row."""
    for rung in RUNGS:
        for arm in ("gen0", "gen4"):
            path = out_root / "gen" / rung / f"{arm}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for k, rid in enumerate(IDS):
                    rec: dict = {"id": rid, "truncated": rid in truncated[(rung, arm)]}
                    if corrupt_cell == (rung, arm) and k == 3:
                        rec.pop("truncated")
                    f.write(json.dumps(rec) + "\n")


def _ctx(out_root: Path) -> SimpleNamespace:
    """Duck-typed FitsCtx slice: exactly the attributes the helpers read."""
    n = len(IDS)
    return SimpleNamespace(
        out_root=out_root,
        rungs=list(RUNGS),
        spine=SimpleNamespace(
            ids=list(IDS),
            gid=np.arange(n, dtype=np.int64),
            n_groups=n,
        ),
    )


def test_trunc_masks_fail_fast_on_corrupt_rollout(tmp_path: Path) -> None:
    """One row missing ``truncated`` in one cell -> RuntimeError (never a
    silent None / degraded censoring family)."""
    trunc = {(r, a): {IDS[0]} for r in RUNGS for a in ("gen0", "gen4")}
    _write_rollouts(tmp_path, trunc, corrupt_cell=("main", "gen4"))
    ctx = _ctx(tmp_path)
    with pytest.raises(RuntimeError, match=r"\(main,gen4\).*fail-fast"):
        F2._trunc_masks(ctx)


def test_trunc_masks_healthy_positional_masks(tmp_path: Path) -> None:
    trunc = {
        ("r0", "gen0"): set(IDS),  # r0 gen0: everything truncated
        ("r0", "gen4"): set(IDS[:4]),
        ("main", "gen0"): {IDS[1], IDS[5]},
        ("main", "gen4"): set(),
    }
    _write_rollouts(tmp_path, trunc)
    masks = F2._trunc_masks(_ctx(tmp_path))
    assert set(masks) == {(r, a) for r in RUNGS for a in ("gen0", "gen4")}
    for key, want in trunc.items():
        got = masks[key]
        assert got.dtype == bool and got.shape == (len(IDS),)
        assert [IDS[k] for k in np.flatnonzero(got)] == sorted(want, key=IDS.index)


def test_common_status_reads_counts_always(tmp_path: Path) -> None:
    """Both labels emitted with n_rows ALWAYS; delta/ci only at n >= 3."""
    ctx = _ctx(tmp_path)
    n = len(IDS)
    rng = np.random.default_rng(0)
    res0, tot0 = rng.uniform(0.1, 1.0, n), rng.uniform(1.0, 2.0, n)
    res4, tot4 = rng.uniform(0.1, 1.0, n), rng.uniform(1.0, 2.0, n)
    counts = F1._boot_counts(np.random.default_rng(1), ctx.spine.n_groups, 16)
    # main: 5 both-natural rows (delta defined), 1 both-truncated (below floor)
    t0 = np.zeros(n, dtype=bool)
    t4 = np.zeros(n, dtype=bool)
    t0[:3] = True
    t4[2:4] = True  # both-truncated = {2}; both-natural = rows 4..7 (n=4)
    trunc = {("main", "gen0"): t0, ("main", "gen4"): t4}
    out = F2._common_status_reads(ctx, counts, trunc, "main", res0, tot0, res4, tot4)
    assert set(out) == {"delta_nt", "delta_tt"}
    nt, tt = out["delta_nt"], out["delta_tt"]
    assert nt["n_rows"] == 4 and "delta" in nt and "ci" in nt and len(nt["ci"]) == 2
    assert tt["n_rows"] == 1 and "delta" not in tt and "ci" not in tt
    assert tt["note"].startswith("below the n>=3 minimum")
    # Degenerate rung: every row truncated in gen0 -> delta_nt has ZERO rows,
    # yet n_rows is still emitted (counts always).
    trunc_all = {("main", "gen0"): np.ones(n, dtype=bool), ("main", "gen4"): t4}
    out_all = F2._common_status_reads(ctx, counts, trunc_all, "main", res0, tot0, res4, tot4)
    assert out_all["delta_nt"]["n_rows"] == 0 and "delta" not in out_all["delta_nt"]
    assert out_all["delta_tt"]["n_rows"] >= 0 and "n_rows" in out_all["delta_tt"]


def test_stratified_ceiling_counts_always_and_partition(tmp_path: Path) -> None:
    """Per-stratum SB ceilings over aligned reliability rows: counts always;
    a supported stratum carries split_half_r; unsupported carries the note."""
    eval_dir = tmp_path / "eval_results" / "issue_2544"
    percell = eval_dir / "fits" / "percell"
    percell.mkdir(parents=True)
    rows = np.arange(6, dtype=np.int64)  # spine positions 0..5 of 8
    rng = np.random.default_rng(2)
    base = rng.normal(size=6)
    np.savez(
        percell / "star_main_f0.npz",
        rel_rows_seed43=rows,
        rel_res_seed43=base + rng.normal(scale=0.05, size=6),
        rel_rows_seed44=rows,
        rel_res_seed44=base + rng.normal(scale=0.05, size=6),
    )
    ctx = _ctx(tmp_path)
    ctx.unit_paths = lambda: (eval_dir / "fits" / "units", percell)
    mask = np.zeros(len(IDS), dtype=bool)
    mask[[0, 1]] = True  # truncated stratum = rel rows {0,1} (n=2 < 3)
    out = F2._stratified_sb(ctx, mask, "star_main_f*.npz")
    assert set(out) == {"natural", "truncated"}
    assert out["natural"]["n_contexts"] == 4 and "split_half_r" in out["natural"]
    assert out["truncated"]["n_contexts"] == 2 and "split_half_r" not in out["truncated"]
    assert "too few" in out["truncated"]["note"]
