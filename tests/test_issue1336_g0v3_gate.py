"""Data-free pins for the G0v3 size-matched fold-ASSIGNMENT contrast (plan v26).

Covers the re-specified gate's pure surfaces in scripts/issue1336_fit_cells.py:
the universe-accounting identity (plan §12 row 44), the three-branch verdict
partition (§6 N21/N24, §12 row 45), the seeded size-matched draw construction
+ the two matched-contrast preconditions (§4; v26 EXACT per-fold counts,
superseding v24's "within 1"), and the manifest-keyed fold diagnostics
(§12 row 46 — sweep-side fold ids are RELABELED by _cv_folds, so "fold 0"
is only meaningful on manifest labels).

All four tests fail against the pre-v26 gate (the surfaces did not exist):
`git show <pre-change>:scripts/issue1336_fit_cells.py | grep -c G0V3` == 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue825_fit_cells as fc825  # noqa: E402
import issue1336_fit_cells as fitc  # noqa: E402
import issue1336_pooled_split as ps  # noqa: E402

# Grouped arm's realized production per-fold profile (plan §12 row 46) — used
# here as a FIXTURE shape; the gate itself always READS the profile off the
# manifest (never this literal).
PRODUCTION_PROFILE = {0: 1685, 1: 2982, 2: 2735, 3: 2166, 4: 989}


def _lmsys_manifest(n_train: int, n_test: int) -> dict:
    """Minimal synthetic manifest for the accounting helper (corpus+arm only)."""
    rows = [{"corpus": "lmsys23k", "arm": "train"} for _ in range(n_train)]
    rows += [{"corpus": "lmsys23k", "arm": "test"} for _ in range(n_test)]
    # foreign-corpus noise the lmsys23k accounting must ignore
    rows += [{"corpus": "gsm8k_train_full", "arm": "train"} for _ in range(7)]
    return {"row_index": rows}


def test_universe_accounting_identity():
    # 17,681 (concat, log-witnessed) -> 13,479 (A2 pinned pre-dedup) -> -17
    # (A2 pinned drops) = 13,462 -> -2,905 test = 10,557 train.
    assert fitc.G0V3_LMSYS_CONCAT_TOTAL == 17_681
    assert int(ps.A2_PINNED_PRE_DEDUP["lmsys23k"]) == 13_479
    assert int(ps.A2_PINNED_DROPS["lmsys23k"]) == 17

    acct = fitc._g0v3_universe_accounting(_lmsys_manifest(10_557, 2_905))
    assert acct["concat_total_log_witnessed"] == 17_681
    assert acct["post_dedup"] == 13_479 - 17 == 13_462
    assert acct["manifest_rows"] == 13_462
    assert acct["train_rows"] == 10_557
    assert acct["test_rows"] == 2_905
    assert acct["train_rows"] + acct["test_rows"] == acct["post_dedup"]
    assert acct["consistent"] is True

    # one row short => flagged inconsistent (the production assert trips on it)
    short = fitc._g0v3_universe_accounting(_lmsys_manifest(10_556, 2_905))
    assert short["consistent"] is False


def test_branch_classification_pass_leakage_anomaly():
    ex_v2 = 1.0303115
    tol = 0.05 * ex_v2  # 0.0515156 (plan §12 row 45)

    # synthetic (R2_G, R2_R) profiles spanning all three branches
    br, ok = fitc._g0v3_branch(0.61 - 0.60, tol)  # delta = +0.01
    assert (br, ok) == ("PASS", True)
    br, ok = fitc._g0v3_branch(0.66 - 0.60, tol)  # delta = +0.06 > tol
    assert (br, ok) == ("FAIL-leakage-exceeds-band", False)
    br, ok = fitc._g0v3_branch(0.58 - 0.60, tol)  # delta = -0.02 < -eps
    assert (br, ok) == ("FAIL-instrument-anomaly", False)

    # boundaries are inclusive-PASS (-0.01 <= delta <= tol)
    assert fitc._g0v3_branch(tol, tol) == ("PASS", True)
    assert fitc._g0v3_branch(-fitc.G0V3_EPS_NOISE, tol) == ("PASS", True)
    assert fitc._g0v3_branch(0.0, tol) == ("PASS", True)

    # one ulp beyond either boundary flips the branch (disjoint + exhaustive)
    up = float(np.nextafter(tol, np.inf))
    dn = float(np.nextafter(-fitc.G0V3_EPS_NOISE, -np.inf))
    assert fitc._g0v3_branch(up, tol)[0] == "FAIL-leakage-exceeds-band"
    assert fitc._g0v3_branch(dn, tol)[0] == "FAIL-instrument-anomaly"

    # a non-finite read raises rather than silently classifying PASS
    with pytest.raises(AssertionError):
        fitc._g0v3_branch(float("nan"), tol)


def test_matched_seed_fold_determinism_and_size_match():
    assert fitc.G0V3_MATCHED_SEED == 13360
    assert fitc.G0V3_MATCHED_DRAWS == 3

    n = sum(PRODUCTION_PROFILE.values())
    draws = []
    for k in range(fitc.G0V3_MATCHED_DRAWS):
        a = fitc._g0v3_matched_labels(PRODUCTION_PROFILE, k)
        b = fitc._g0v3_matched_labels(PRODUCTION_PROFILE, k)
        # reproducible under the derived seed [G0V3_MATCHED_SEED, k]
        assert np.array_equal(a, b)
        assert a.shape == (n,)
        # v26: per-fold sizes EXACTLY equal the manifest profile (supersedes
        # v24's "within 1")
        uniq, counts = np.unique(a, return_counts=True)
        assert {int(u): int(c) for u, c in zip(uniq, counts, strict=True)} == PRODUCTION_PROFILE
        fitc._g0v3_assert_matched(a, PRODUCTION_PROFILE, 5, f"draw{k}")
        draws.append(a)
    # distinct draws are distinct permutations
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])

    # precondition (1): a wrong unique-label count raises
    with pytest.raises(AssertionError):
        fitc._g0v3_assert_matched(np.zeros(n, dtype=np.int64), PRODUCTION_PROFILE, 5, "bad-uniq")
    # precondition (2): a size-drifted assignment raises (multiset mismatch)
    drifted = np.concatenate(
        [
            np.full(1684, 0),
            np.full(2983, 1),
            np.full(2735, 2),
            np.full(2166, 3),
            np.full(989, 4),
        ]
    )
    with pytest.raises(AssertionError):
        fitc._g0v3_assert_matched(drifted, PRODUCTION_PROFILE, 5, "drift")
    # multiset semantics: a RELABELED but size-matched assignment passes —
    # _cv_folds relabels fold ids, so only the size multiset is invariant
    relabeled = np.concatenate(
        [
            np.full(989, 0),
            np.full(2166, 1),
            np.full(2735, 2),
            np.full(2982, 3),
            np.full(1685, 4),
        ]
    )
    fitc._g0v3_assert_matched(relabeled, PRODUCTION_PROFILE, 5, "relabels-ok")


def _diag_manifest() -> dict:
    """Synthetic 3-fold manifest: fold 0 holds 3 rows (2 quarantine, gid 100),
    fold 1 holds 4 rows (gid 102), fold 2 holds 5 rows (gid 103)."""
    rows = []

    def add(fold: int, cluster: int, count: int) -> None:
        for _ in range(count):
            i = len(rows)
            rows.append(
                {
                    "corpus": "lmsys23k",
                    "prompt_idx": i,
                    "prompt_sha": f"sha{i}",
                    "cluster": cluster,
                    "arm": "train",
                    "fold": fold,
                }
            )

    add(0, 100, 2)  # quarantine gid
    add(0, 101, 1)
    add(1, 102, 4)
    add(2, 103, 5)
    # test-arm + foreign-corpus rows the diagnostics must ignore
    rows.append(
        {
            "corpus": "lmsys23k",
            "prompt_idx": 900,
            "prompt_sha": "sha900",
            "cluster": 104,
            "arm": "test",
        }
    )
    rows.append(
        {
            "corpus": "gsm8k_train_full",
            "prompt_idx": 901,
            "prompt_sha": "sha901",
            "cluster": 999,
            "arm": "train",
            "fold": 0,
        }
    )
    group_table = [
        {"corpus": "lmsys23k", "group_id": 100, "quarantine": True},
        {"corpus": "lmsys23k", "group_id": 101, "quarantine": False},
        {"corpus": "lmsys23k", "group_id": 102, "quarantine": False},
        {"corpus": "lmsys23k", "group_id": 103, "quarantine": False},
        # foreign-corpus quarantine group — must not leak into the lmsys count
        {"corpus": "gsm8k_train_full", "group_id": 999, "quarantine": True},
    ]
    return {"row_index": rows, "group_table": group_table}


def test_fold_diagnostics_keyed_off_manifest_labels():
    assert ps.QUARANTINE_TRAIN_FOLD == 0
    man = _diag_manifest()
    fold_row_counts, fold0_q = fitc._g0v3_fold_diagnostics(man)
    assert fold_row_counts == {"0": 3, "1": 4, "2": 5}
    assert fold0_q == 2

    # The trap this pins (plan v26 mechanism trap b): _cv_folds RELABELS the
    # manifest fold ids through a seeded permutation, so a sweep-side "fold 0"
    # is arbitrary. seed=0 realizes the NON-identity map {0->2, 1->0, 2->1}
    # over 3 unique labels (probed; deterministic for default_rng(0)).
    entries = [e for e in man["row_index"] if e["corpus"] == "lmsys23k" and e["arm"] == "train"]
    labels = np.asarray([int(e["fold"]) for e in entries])
    realized = fc825._cv_folds(labels, 3, 0)
    # _cv_folds output is a pure relabeling: constant within each manifest label
    mapping = {}
    for lab in np.unique(labels):
        vals = np.unique(realized[labels == lab])
        assert vals.shape == (1,)
        mapping[int(lab)] = int(vals[0])
    assert sorted(mapping.values()) == [0, 1, 2]  # bijection at #unique == n_folds
    assert mapping[0] != 0, "seed=0 must move manifest fold 0 (probed non-identity map)"

    # Counterfactual: keying "fold 0" on the RELABELED sweep-side ids
    # misreports the quarantine count; the helper tracks the MANIFEST.
    qgids = {100}
    sweep_keyed_q = sum(
        1
        for e, r in zip(entries, realized, strict=True)
        if int(e["cluster"]) in qgids and int(r) == ps.QUARANTINE_TRAIN_FOLD
    )
    assert sweep_keyed_q != fold0_q
    # and the manifest-keyed diagnostics are invariant to whatever _cv_folds did
    again_counts, again_q = fitc._g0v3_fold_diagnostics(man)
    assert (again_counts, again_q) == (fold_row_counts, fold0_q)


def test_own_argmax_selection_symmetry_and_branch_call_site():
    """Pin the selection-symmetry mechanism end to end (code-review v23 Minors 1+2).

    (a) `_g0v3_own_argmax` behavioral pin: the argmax is restricted to the
        pre-registered CANDIDATE set (a larger value OUTSIDE the candidates
        must not win); value + per-candidate table returned; an empty
        candidate intersection is fail-loud unless allow_fallback; the
        fallback degrades to ALL swept layers; non-finite candidates are
        skipped and an all-NaN candidate set raises.
    (b) Call-site AST pins inside `run_g0v3` (the two review-surviving
        mutations): the `_g0v3_branch` verdict call passes EXACTLY
        (delta_assign, tol) — no third positional, no `eps` keyword, no
        SD-derived term (SD is diagnostic-only, never a threshold input;
        plan v26 SS4) — and BOTH arms' R^2 reads flow through
        `_g0v3_own_argmax` (a non-argmax / global-max / fixed-layer draw
        read breaks selection symmetry).
    (c) Constant pins: G0V3_EPS_NOISE == 0.01 byte-exact and it IS the
        `_g0v3_branch` eps default.
    """
    import ast
    import inspect

    # --- (a) behavioral pin (reviewer sketch, extended) ---
    r2 = np.zeros(31)
    cands = [16, 21, 22, 30]
    r2[cands] = [0.52, 0.58, 0.57, 0.55]
    r2[5] = 0.99  # global max OUTSIDE the candidate set -- must NOT be selected
    layer, val, table = fitc._g0v3_own_argmax(r2, cands)
    assert (layer, val) == (21, pytest.approx(0.58))
    assert set(table.keys()) == {str(li) for li in cands}
    assert table["21"] == pytest.approx(0.58)

    # empty intersection: production candidates against an 8-layer smoke sweep
    short = np.linspace(0.1, 0.8, 8)
    with pytest.raises(AssertionError):
        fitc._g0v3_own_argmax(short, cands, allow_fallback=False)
    layer_fb, val_fb, table_fb = fitc._g0v3_own_argmax(short, cands, allow_fallback=True)
    assert layer_fb == 7
    assert val_fb == pytest.approx(0.8)
    assert set(table_fb.keys()) == {str(i) for i in range(8)}

    # non-finite candidates skipped (nanargmax); all-NaN candidate set raises
    r2_nan = r2.copy()
    r2_nan[21] = np.nan
    layer_n, val_n, _ = fitc._g0v3_own_argmax(r2_nan, cands)
    assert (layer_n, val_n) == (22, pytest.approx(0.57))
    with pytest.raises(AssertionError):
        fitc._g0v3_own_argmax(np.full(31, np.nan), cands)

    # --- (b) call-site AST pins ---
    tree = ast.parse(inspect.getsource(fitc.run_g0v3))

    def _callee_name(call: ast.Call) -> str:
        f = call.func
        return f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")

    branch_calls = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and _callee_name(n) == "_g0v3_branch"
    ]
    assert len(branch_calls) == 1, "run_g0v3 must have exactly one _g0v3_branch verdict call"
    call = branch_calls[0]
    assert len(call.args) == 2 and not call.keywords, (
        "run_g0v3 must call _g0v3_branch(delta_assign, tol) with the eps DEFAULT -- "
        "threading a third arg / eps kwarg (e.g. an SD-derived widening) is banned: "
        "the across-draw SD is diagnostic-only, never a verdict input (plan v26 SS4)"
    )
    referenced = {n.id for a in call.args for n in ast.walk(a) if isinstance(n, ast.Name)}
    assert not ({"r2_r_sd", "r2_random_sd"} & referenced), (
        "SD must not reach the _g0v3_branch call site even inside an argument expression"
    )

    argmax_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _callee_name(n) == "_g0v3_own_argmax"
    ]
    assert len(argmax_calls) == 2, (
        "run_g0v3 must read BOTH arms through _g0v3_own_argmax (one grouped call + one call "
        "inside the draws comprehension); a non-argmax draw read breaks selection symmetry"
    )

    # --- (c) constant pins ---
    assert fitc.G0V3_EPS_NOISE == 0.01
    sig = inspect.signature(fitc._g0v3_branch)
    assert sig.parameters["eps"].default == fitc.G0V3_EPS_NOISE == 0.01


# ---------------------------------------------------------------------------
# Adjudication pass-through (plan §7 FAIL-leakage disposition; round v24).
# The harness drives the REAL run_g0v3 body — verdict, payload, adjudication
# reader, and return code all execute unmodified — with ONLY the three heavy
# fit seams (_pooled_bundle / _pooled_xy_from_bundle / _run_sweep_edge)
# stubbed at the data/GPU boundary, at a production-shaped (non-smoke)
# manifest matching the pinned universe accounting (13,462 lmsys23k rows).
# ---------------------------------------------------------------------------

_ADJ_CANDS = [16, 21, 22, 30]
# Candidate-layer R2 profiles (ordered as _ADJ_CANDS). Leakage shape:
# delta_assign = mean(0.6151, 0.6149, 0.6147) - 0.5383 = +0.0766 > tol 0.0515.
_R2_G_LEAK = [0.50, 0.52, 0.51, 0.5383]
_R2_R_LEAK = [
    [0.54, 0.56, 0.55, 0.6151],
    [0.54, 0.56, 0.55, 0.6149],
    [0.54, 0.56, 0.55, 0.6147],
]
# Anomaly shape: delta_assign = 0.55 - 0.60 = -0.05 < -eps (grouped beats random).
_R2_G_ANOM = [0.50, 0.52, 0.51, 0.60]
_R2_R_ANOM = [[0.40, 0.42, 0.41, 0.55]] * 3


def _adj_gate_args(tmp_path: Path) -> tuple[SimpleNamespace, Path]:
    """Production-mode (smoke=False) run_g0v3 namespace + fixture tree.

    Real gates_v2 bars (production ex_v2), a real round-3 ref cell with the
    production candidate set, and a split manifest whose lmsys23k slice
    matches the pinned universe accounting (10,557 train in the realized
    fold profile + 2,905 test = 13,462) so the enforced production asserts
    all run for real.
    """
    out_dir = tmp_path / "out"
    (out_dir / "gates_v2").mkdir(parents=True)
    (out_dir / "cells_v2").mkdir(parents=True)
    (out_dir / "gates_v2" / "v2_bars.json").write_text(json.dumps({"ex_v2": 1.0303115193390924}))
    ref_id = fitc.cm.v2_cell_id("rlvr", "chat", "lmsys23k")
    r2_ref = [0.0] * 31
    for li, v in zip(_ADJ_CANDS, [0.50, 0.55, 0.54, 0.6090], strict=True):
        r2_ref[li] = v
    (out_dir / "cells_v2" / f"cells_{ref_id}.json").write_text(
        json.dumps({"r2_per_layer_obs": r2_ref, "frozen_layers": _ADJ_CANDS})
    )
    rows = []
    i = 0
    for fold, n in sorted(PRODUCTION_PROFILE.items()):
        for _ in range(n):
            rows.append(
                {
                    "corpus": "lmsys23k",
                    "arm": "train",
                    "fold": fold,
                    "cluster": 100 + (i % 36),
                    "prompt_idx": i,
                    "prompt_sha": f"sha{i}",
                }
            )
            i += 1
    for j in range(2905):
        rows.append(
            {
                "corpus": "lmsys23k",
                "arm": "test",
                "cluster": 100 + (j % 36),
                "prompt_idx": 100_000 + j,
                "prompt_sha": f"tsha{j}",
            }
        )
    man = {
        "row_index": rows,
        "n_folds": 5,
        "group_table": [
            {"corpus": "lmsys23k", "group_id": 100 + g, "quarantine": g == 0} for g in range(36)
        ],
    }
    man_path = tmp_path / "split_manifest.json"
    man_path.write_text(json.dumps(man))
    args = SimpleNamespace(
        out_dir=out_dir,
        smoke=False,
        seed=0,
        turnstore_dir=tmp_path / "ts",
        split_manifest=man_path,
        wave1_turnstore_dir=None,
        gen_root=None,
    )
    return args, out_dir


def _stub_fit_seams(monkeypatch, r2_g_cands, r2_r_cands_per_draw) -> None:
    """Stub ONLY the heavy fit seams; call order (grouped first, then the
    K=3 draws) selects which candidate-layer profile each sweep returns."""
    calls = {"n": 0}
    monkeypatch.setattr(fitc, "_pooled_bundle", lambda *a, **k: object())

    def fake_xy(bundle, entries, expected_layers, slot):
        n = len(entries)
        return np.zeros((n, 3)), np.zeros(n), None

    monkeypatch.setattr(fitc, "_pooled_xy_from_bundle", fake_xy)

    def fake_sweep(X, Y, groups, *, base_grid, sweep_kwargs):
        i = calls["n"]
        calls["n"] += 1
        vals = r2_g_cands if i == 0 else r2_r_cands_per_draw[i - 1]
        r2 = np.full(31, np.nan)
        for li, v in zip(_ADJ_CANDS, vals, strict=True):
            r2[li] = v
        return {"r2_obs": r2}, None, None

    monkeypatch.setattr(fitc, "_run_sweep_edge", fake_sweep)


def _matching_record(payload: dict) -> dict:
    """An adjudication record built FROM the failed run's own payload — the
    production flow (the record re-states the numbers that were adjudicated)."""
    return {
        "gate": "g0v3",
        "adjudicated_branch": "FAIL-leakage-exceeds-band",
        "disposition": "proceed",
        "delta_assign": payload["delta_assign"],
        "tolerance": payload["tolerance"],
        "r2_grouped": payload["r2_grouped"],
        "r2_random_mean": payload["r2_random_mean"],
        "matched_seed": payload["matched_seed"],
        "rationale": "test rationale — instrument legitimately conservative",
    }


def _read_payload(out_dir: Path) -> dict:
    return json.loads((out_dir / "gates_v3" / "g0v3.json").read_text())


def test_adjudication_missing_record_keeps_rc3(tmp_path, monkeypatch):
    """No record present => today's behavior unchanged: rc 3, applied False."""
    args, out_dir = _adj_gate_args(tmp_path)
    _stub_fit_seams(monkeypatch, _R2_G_LEAK, _R2_R_LEAK)
    rc = fitc.run_g0v3(args)
    assert rc == 3
    payload = _read_payload(out_dir)
    assert payload["verdict"] == "FAIL-leakage-exceeds-band"
    assert payload["pass"] is False
    assert payload["adjudication"]["applied"] is False
    assert payload["adjudication"]["reason"] == "no adjudication record present"
    assert payload["adjudication"]["record"].endswith("gates_v3/g0v3_adjudication.json")


def test_adjudication_applies_rc0_verdict_intact(tmp_path, monkeypatch, capsys):
    """A record matching the freshly computed values => rc 0; `pass` stays
    False, `verdict` stays FAIL-leakage-exceeds-band, applied True, and the
    fix-engaged token `[g0v3] ADJUDICATED PASS-THROUGH` is emitted."""
    args, out_dir = _adj_gate_args(tmp_path)
    _stub_fit_seams(monkeypatch, _R2_G_LEAK, _R2_R_LEAK)
    assert fitc.run_g0v3(args) == 3  # halt first — the record derives from it
    record = _matching_record(_read_payload(out_dir))
    (out_dir / "gates_v3" / "g0v3_adjudication.json").write_text(json.dumps(record))
    capsys.readouterr()  # drop the first run's output
    _stub_fit_seams(monkeypatch, _R2_G_LEAK, _R2_R_LEAK)  # fresh call counter
    rc = fitc.run_g0v3(args)
    out = capsys.readouterr().out
    assert rc == 0
    payload = _read_payload(out_dir)
    assert payload["pass"] is False, "adjudication must never flip `pass`"
    assert payload["verdict"] == "FAIL-leakage-exceeds-band", (
        "adjudication must never rewrite the verdict"
    )
    assert payload["adjudication"]["applied"] is True
    assert "test rationale" in payload["adjudication"]["reason"]
    assert "[g0v3] ADJUDICATED PASS-THROUGH" in out
    # the loud line names the still-failing numbers + the record path
    loud = next(ln for ln in out.splitlines() if ln.startswith("[g0v3] ADJUDICATED PASS-THROUGH"))
    assert "delta_assign=" in loud and "tol=" in loud
    assert "g0v3_adjudication.json" in loud


def test_adjudication_value_mismatch_refuses(tmp_path, monkeypatch, capsys):
    """A perturbed delta_assign (1e-6 relative — far above rel_tol=1e-9)
    => no pass-through: rc 3, applied False, clause named in the reason."""
    args, out_dir = _adj_gate_args(tmp_path)
    _stub_fit_seams(monkeypatch, _R2_G_LEAK, _R2_R_LEAK)
    assert fitc.run_g0v3(args) == 3
    record = _matching_record(_read_payload(out_dir))
    record["delta_assign"] = record["delta_assign"] * (1.0 + 1e-6)
    (out_dir / "gates_v3" / "g0v3_adjudication.json").write_text(json.dumps(record))
    capsys.readouterr()
    _stub_fit_seams(monkeypatch, _R2_G_LEAK, _R2_R_LEAK)
    rc = fitc.run_g0v3(args)
    out = capsys.readouterr().out
    assert rc == 3
    payload = _read_payload(out_dir)
    assert payload["adjudication"]["applied"] is False
    assert "delta_assign" in payload["adjudication"]["reason"]
    assert "[g0v3] ADJUDICATED PASS-THROUGH" not in out
    assert "adjudication refused" in out and "VALUE PIN" in out


def test_adjudication_anomaly_branch_never_adjudicable(tmp_path, monkeypatch, capsys):
    """FAIL-instrument-anomaly with an otherwise-matching anomaly record
    => refused unconditionally (rc 3): it is the code-defect signature."""
    args, out_dir = _adj_gate_args(tmp_path)
    _stub_fit_seams(monkeypatch, _R2_G_ANOM, _R2_R_ANOM)
    assert fitc.run_g0v3(args) == 3
    payload = _read_payload(out_dir)
    assert payload["verdict"] == "FAIL-instrument-anomaly"
    record = _matching_record(payload)
    record["adjudicated_branch"] = "FAIL-instrument-anomaly"
    (out_dir / "gates_v3" / "g0v3_adjudication.json").write_text(json.dumps(record))
    capsys.readouterr()
    _stub_fit_seams(monkeypatch, _R2_G_ANOM, _R2_R_ANOM)
    rc = fitc.run_g0v3(args)
    out = capsys.readouterr().out
    assert rc == 3
    payload = _read_payload(out_dir)
    assert payload["adjudication"]["applied"] is False
    assert "NEVER adjudicable" in payload["adjudication"]["reason"]
    assert "NEVER adjudicable" in out
    assert "[g0v3] ADJUDICATED PASS-THROUGH" not in out


def test_adjudication_missing_required_key_refuses(tmp_path, monkeypatch):
    """A missing required key is a refusal (rc 3) — never a silently
    satisfied comparison. Covers a value key AND the matched_seed pin."""
    args, out_dir = _adj_gate_args(tmp_path)
    _stub_fit_seams(monkeypatch, _R2_G_LEAK, _R2_R_LEAK)
    assert fitc.run_g0v3(args) == 3
    base = _matching_record(_read_payload(out_dir))
    for missing in ("delta_assign", "tolerance", "r2_grouped", "r2_random_mean", "matched_seed"):
        record = dict(base)
        del record[missing]
        (out_dir / "gates_v3" / "g0v3_adjudication.json").write_text(json.dumps(record))
        _stub_fit_seams(monkeypatch, _R2_G_LEAK, _R2_R_LEAK)
        assert fitc.run_g0v3(args) == 3, f"missing {missing!r} must refuse"
        payload = _read_payload(out_dir)
        assert payload["adjudication"]["applied"] is False
        assert missing in payload["adjudication"]["reason"]


def test_adjudication_constants_and_tolerance_pin():
    """G0V3_EPS_NOISE stays 0.01 and the tolerance expression stays
    byte-identical `tol = 0.05 * ex_v2` (AST + source-text pins)."""
    import ast
    import inspect

    assert fitc.G0V3_EPS_NOISE == 0.01
    src = inspect.getsource(fitc.run_g0v3)
    assert "tol = 0.05 * ex_v2" in src, "tolerance expression must stay byte-identical"
    tree = ast.parse(src)
    tol_assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "tol"
    ]
    assert len(tol_assigns) == 1, "run_g0v3 must assign tol exactly once"
    val = tol_assigns[0].value
    assert isinstance(val, ast.BinOp) and isinstance(val.op, ast.Mult)
    assert isinstance(val.left, ast.Constant) and val.left.value == 0.05
    assert isinstance(val.right, ast.Name) and val.right.id == "ex_v2"
    # the three branch strings stay byte-identical in the branch helper
    bsrc = inspect.getsource(fitc._g0v3_branch)
    assert '"FAIL-leakage-exceeds-band"' in bsrc
    assert '"FAIL-instrument-anomaly"' in bsrc
    assert '"PASS"' in bsrc
