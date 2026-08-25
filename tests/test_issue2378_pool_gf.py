"""Pins for the issue #2378 interpretation-round r1 pooled-arm fix
(scripts/issue2378_pool.py --global-family-folds + --phase lofo).

Incident (interpretation-critique r1, Codex + orchestrator-confirmed against
eval_results/issue_2378/fold_map.json): the registered fold map assigns
final_seed_id families to folds PER CELL, the 5 story cells share all 25
families with 15-23/25 cross-cell fold disagreements per pair, and the pooled
arm aligns folds BY INDEX — so every story target's eval-fold families appear
in sibling story cells' pooled-TRAINING rows (family exposure).

Pins (the brief's deliverable-5 list):
(a) the GLOBAL family -> fold assignment gives ZERO target-eval-family overlap
    with ANY pooled training cell (the Codex mechanizable check), on a fixture
    whose registered per-cell assignments provably DISAGREE;
(b) the default-OFF path is byte-identical fold selection to the registered
    behavior (same object returned; input never mutated);
(c) the LOFO totals-minus-cell moment arithmetic equals a DIRECT
    PooledMomentRidge fit on the sibling-union train rows (small synthetic d);
plus the per-cell n_train floor re-assert and the three non-story
dispositions (content-disjoint / shared-but-fold-consistent /
shared-with-disagreement -> global re-alignment).

No production function is stubbed anywhere here — every test executes the
real bodies on synthetic inputs (#906 discipline).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2054_fits as pf  # noqa: E402
import issue2054_pool_specialize as ps  # noqa: E402
import issue2378_p6_common as p6  # noqa: E402
import issue2378_pool as pool  # noqa: E402

STORY = ["storyq_astra", "storyq_dana", "storyq_helios", "storyq_vex", "storyq_wren"]
FAMILIES = ["famA", "famB", "famC", "famD", "famE", "famF"]
K = 3


def _story_entry(cell: str, counts: dict[str, int]) -> dict:
    """A registered-convention story entry: per-cell greedy LPT on the cell's
    OWN counts (the exact build_fold_map recipe — the incident shape)."""
    fam_fold = p6._greedy_family_folds(counts, K)
    family_keys: list[str] = []
    for fam in FAMILIES:
        family_keys.extend([fam] * counts[fam])
    folds = [int(fam_fold[fam]) for fam in family_keys]
    row_ids = [f"{cell}_r{i:04d}" for i in range(len(family_keys))]
    return {
        "fold_structure": "family-held-out",
        "row_ids": row_ids,
        "folds": folds,
        "family_keys": family_keys,
        "n_rows": len(row_ids),
        "fold_sizes": [folds.count(f) for f in range(K)],
        "story_fold_audit": p6.audit_story_folds(family_keys, folds, K),
    }


def _conv_entry(row_ids: list[str], folds: list[int]) -> dict:
    return {
        "fold_structure": "conversation-grouped",
        "row_ids": list(row_ids),
        "folds": list(folds),
        "n_rows": len(row_ids),
        "fold_sizes": [folds.count(f) for f in range(K)],
    }


def _fixture_fold_map(*, n_train_floor: int = 1, nonstory: dict | None = None) -> dict:
    # Per-cell counts deliberately PERMUTED across cells so the registered
    # per-cell greedy assignments disagree (the incident's mechanism).
    base = [7, 6, 5, 4, 3, 2]
    cells: dict[str, dict] = {}
    for ci, cell in enumerate(STORY):
        rotated = base[ci:] + base[:ci]
        counts = dict(zip(FAMILIES, rotated, strict=True))
        cells[cell] = _story_entry(cell, counts)
    if nonstory is None:
        nonstory = {
            "chat": _conv_entry([f"mt_sha:c{i:04d}" for i in range(9)], [i % K for i in range(9)]),
            "plain_text": _conv_entry(
                [f"mt_sha:p{i:04d}" for i in range(9)], [i % K for i in range(9)]
            ),
            "chat_user_real": _conv_entry(
                [f"mt_wc:u{i:04d}" for i in range(9)], [i % K for i in range(9)]
            ),
        }
    cells.update(nonstory)
    fm = {
        "k": K,
        "seed": 137,
        "n_eq": min(e["n_rows"] for e in cells.values()),
        "n_eq_floor": 1,
        "n_train_floor": n_train_floor,
        "kept_counts": {c: e["n_rows"] for c, e in cells.items()},
        "excluded": {},
        "user_intersection": None,
        "cells": cells,
        "store_index": {},
    }
    fm["sha256"] = p6.fold_map_sha(fm)
    return fm


def _fam_fold_of(entry: dict) -> dict:
    out: dict = {}
    for fam, f in zip(entry["family_keys"], entry["folds"], strict=True):
        out.setdefault(fam, set()).add(int(f))
    assert all(len(v) == 1 for v in out.values()), "family split across folds"
    return {fam: next(iter(v)) for fam, v in out.items()}


def test_fixture_registered_assignments_disagree():
    """Fixture validity: the registered per-cell greedy assignments DISAGREE
    across cells (otherwise pin (a) would be vacuous)."""
    fm = _fixture_fold_map()
    maps = {c: _fam_fold_of(fm["cells"][c]) for c in STORY}
    n_disagree = sum(
        1
        for fam in FAMILIES
        for i, a in enumerate(STORY)
        for b in STORY[i + 1 :]
        if maps[a][fam] != maps[b][fam]
    )
    assert n_disagree > 0
    # And the incident's consequence: some target eval family sits in a
    # sibling's TRAIN rows for the same fold index.
    exposed = 0
    for t in STORY:
        for f in range(K):
            eval_fams = {fam for fam, ff in maps[t].items() if ff == f}
            for c in STORY:
                if c == t:
                    continue
                train_fams = {fam for fam, ff in maps[c].items() if ff != f}
                exposed += len(eval_fams & train_fams)
    assert exposed > 0


def test_a_global_assignment_zero_family_exposure():
    """Pin (a): under derive_global_family_folds, NO story target's eval-fold
    families appear in ANY story cell's train rows for that fold — checked
    mechanically from the derived entries, independent of the audit code."""
    fm = _fixture_fold_map()
    fm_gf = pool.derive_global_family_folds(fm)
    assert fm_gf["fold_regime"] == "global-family"
    maps = {c: _fam_fold_of(fm_gf["cells"][c]) for c in STORY}
    # One GLOBAL assignment: every cell agrees on every family.
    for fam in FAMILIES:
        assert len({maps[c][fam] for c in STORY}) == 1, fam
    for t in STORY:
        for f in range(K):
            eval_fams = {fam for fam, ff in maps[t].items() if ff == f}
            pooled_train = set()
            for c in STORY:
                pooled_train |= {fam for fam, ff in maps[c].items() if ff != f}
            assert not (eval_fams & pooled_train), (t, f, sorted(eval_fams & pooled_train))
    # Per-cell audits re-written for the derived folds; cross-cell audit recorded.
    rec = fm_gf["global_folds"]
    assert rec["cross_cell_audit"]["verdict"] == "zero-cross-cell-family-exposure"
    assert set(rec["refolded_cells"]) <= set(fm_gf["cells"])
    assert rec["refolded_cells"], "permuted-count fixture must refold >= 1 cell"
    for c in STORY:
        assert fm_gf["cells"][c]["story_fold_audit"]["verdict"] == "zero-overlap"
    # sha re-derived over the modified map (regime separation from pool/).
    assert fm_gf["sha256"] != fm["sha256"]
    assert fm_gf["base_fold_map_sha"] == fm["sha256"]


def test_b_default_off_byte_identical():
    """Pin (b): flag OFF returns the SAME registered fold map object (fold
    selection byte-identical), and derive never mutates its input."""
    fm = _fixture_fold_map()
    before = json.dumps(fm, sort_keys=True)
    view = pool._resolve_fold_view(SimpleNamespace(global_family_folds=False), fm)
    assert view is fm
    assert json.dumps(view, sort_keys=True) == before
    # Deriving the gf view leaves the registered map untouched too.
    pool.derive_global_family_folds(fm)
    assert json.dumps(fm, sort_keys=True) == before
    # And the pool-dir routing: registered -> pool/, gf -> pool_gf/.
    assert pool._pool_dir_name(SimpleNamespace(global_family_folds=False)) == "pool"
    assert pool._pool_dir_name(SimpleNamespace(global_family_folds=True)) == "pool_gf"


def test_nonstory_dispositions():
    fm = _fixture_fold_map()
    fm_gf = pool.derive_global_family_folds(fm)
    rec = fm_gf["global_folds"]["nonstory"]
    assert rec["framings_content_disjoint"] is True
    assert rec["n_shared_keys"] == 0
    assert "content-disjoint" in rec["action"]
    for c in ("chat", "plain_text", "chat_user_real"):
        assert fm_gf["cells"][c]["folds"] == fm["cells"][c]["folds"]

    # Shared keys, IDENTICAL folds (paired-user topology): registered kept.
    ids = [f"conv{i:04d}" for i in range(9)]
    folds = [i % K for i in range(9)]
    fm2 = _fixture_fold_map(
        nonstory={
            "chat_user_real": _conv_entry(ids, folds),
            "chat_user_sim": _conv_entry(ids, folds),
            "chat": _conv_entry([f"mt_sha:c{i}" for i in range(9)], folds),
            "plain_text": _conv_entry([f"mt_sha:p{i}" for i in range(9)], folds),
        }
    )
    rec2 = pool.derive_global_family_folds(fm2)["global_folds"]["nonstory"]
    assert rec2["framings_content_disjoint"] is False
    assert rec2["n_shared_keys"] == 9
    assert rec2["n_shared_keys_fold_disagreeing"] == 0
    assert "fold-consistent" in rec2["action"]

    # Shared keys with DISAGREEING folds: global row-key re-alignment.
    fm3 = _fixture_fold_map(
        nonstory={
            "chat": _conv_entry(ids, folds),
            "plain_text": _conv_entry(ids, [(f + 1) % K for f in folds]),
            "chat_user_real": _conv_entry([f"mt_wc:u{i}" for i in range(9)], folds),
        }
    )
    fm3_gf = pool.derive_global_family_folds(fm3)
    rec3 = fm3_gf["global_folds"]["nonstory"]
    assert rec3["n_shared_keys_fold_disagreeing"] > 0
    assert "fold-aligned globally" in rec3["action"]
    chat_e, plain_e = fm3_gf["cells"]["chat"], fm3_gf["cells"]["plain_text"]
    chat_f = dict(zip(chat_e["row_ids"], chat_e["folds"], strict=True))
    plain_f = dict(zip(plain_e["row_ids"], plain_e["folds"], strict=True))
    for rid in ids:
        assert chat_f[rid] == plain_f[rid], rid


def test_floor_reassert_raises():
    """The per-cell per-fold n_train floor re-fires on the DERIVED assignment
    (the naive global-LPT variant violated it on the realized map)."""
    fm = _fixture_fold_map(n_train_floor=26)  # n_rows=27 -> any fold >1 row starves
    with pytest.raises(RuntimeError, match="n_train"):
        pool.derive_global_family_folds(fm)


def test_c_lofo_moments_equal_direct_fit():
    """Pin (c): lofo_train_moments (totals - target cell, train complement of
    fold f) reproduces a DIRECT PooledMomentRidge fit on the materialized
    sibling-union train rows — same selected lambda, matching predictions."""
    rng = np.random.default_rng(11)
    d, k, n = 6, 3, 30
    cells = ["cellA", "cellB", "cellC"]
    W = rng.standard_normal((d, d)) / np.sqrt(d)
    data = {}
    for c in cells:
        x = rng.standard_normal((n, d))
        y = x @ W + 0.05 * rng.standard_normal((n, d))
        folds = np.array([i % k for i in range(n)], dtype=np.int64)
        data[c] = (x, y, folds)
    cellmoms = {
        c: pool._fold_moments(
            torch.as_tensor(x, dtype=torch.float64),
            torch.as_tensor(y, dtype=torch.float64),
            folds,
            k,
        )
        for c, (x, y, folds) in data.items()
    }
    # Global per-fold moments = sum over cells (the accumulate_moments shape).
    mom = []
    for f in range(k):
        m = {key: sum(cellmoms[c][f][key] for c in cells) for key in ("n", "yss")}
        for key in ("sum_x", "sum_y", "c_xx", "c_xy"):
            m[key] = sum(cellmoms[c][f][key] for c in cells)
        mom.append(m)
    x_test = rng.standard_normal((8, d))
    for target in cells:
        for f in range(k):
            train = pool.lofo_train_moments(mom, cellmoms[target], f, k)
            lofo_model = ps.PooledMomentRidge(**train, lambdas=pf.DEFAULT_LAMBDAS, dof_cap=0.9)
            xs, ys = [], []
            for c in cells:
                if c == target:
                    continue
                x, y, folds = data[c]
                tr = np.flatnonzero(folds != f)
                xs.append(x[tr])
                ys.append(y[tr])
            xu = torch.as_tensor(np.concatenate(xs), dtype=torch.float64)
            yu = torch.as_tensor(np.concatenate(ys), dtype=torch.float64)
            direct = ps.PooledMomentRidge(
                n=int(xu.shape[0]),
                sum_x=xu.sum(0),
                sum_y=yu.sum(0),
                yss=float((yu * yu).sum()),
                c_xx=xu.T @ xu,
                c_xy=xu.T @ yu,
                lambdas=pf.DEFAULT_LAMBDAS,
                dof_cap=0.9,
            )
            assert train["n"] == direct.n_train
            assert lofo_model.best_lambda == direct.best_lambda, (target, f)
            np.testing.assert_allclose(
                lofo_model.predict_np(x_test),
                direct.predict_np(x_test),
                rtol=1e-7,
                atol=1e-9,
                err_msg=f"target={target} fold={f}",
            )


def test_accumulate_kernel_matches_fold_moments():
    """The accumulate_moments inner kernel IS _fold_moments (the refactor is
    numerics-preserving): summed per-cell fold moments equal a direct
    single-pass accumulation over the same rows."""
    rng = np.random.default_rng(3)
    d, k, n = 5, 3, 21
    x = torch.as_tensor(rng.standard_normal((n, d)), dtype=torch.float64)
    y = torch.as_tensor(rng.standard_normal((n, d)), dtype=torch.float64)
    folds = np.array([i % k for i in range(n)], dtype=np.int64)
    fms = pool._fold_moments(x, y, folds, k)
    for f in range(k):
        idx = np.flatnonzero(folds == f)
        assert fms[f]["n"] == len(idx)
        torch.testing.assert_close(fms[f]["c_xx"], x[idx].T @ x[idx])
        torch.testing.assert_close(fms[f]["c_xy"], x[idx].T @ y[idx])
        torch.testing.assert_close(fms[f]["sum_x"], x[idx].sum(0))
        assert fms[f]["yss"] == pytest.approx(float((y[idx] * y[idx]).sum()))
