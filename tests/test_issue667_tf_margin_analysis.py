"""Issue #667 tf-margin analysis tests — vendor byte-identity + bootstrap + both gates.

Pins:
1. The 2 analysis helpers vendored from #722 are AST-body-identical to the source
   on the issue-722-tf-margin branch, and N_BOOT == 2000 (the #722 default; plan
   v6 nit-2 corrected v5's mis-stated 1000).
2. clustered_bootstrap_spearman reproduces the source's numeric result on a fixed
   seed + a small synthetic dataset (the vendor did not drift).
3. The g0-correctness gate HALTs (G0CorrectnessError) when the recomputed
   aggregate Spearman(g0, G) diverges from the committed base-G rho beyond tol,
   and PASSes when it matches; the measurement-validity gate flags passed=True
   only when the point est > 0 AND CI excludes zero.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue667_tf_margin_analysis as tfa  # noqa: E402

VENDORED_ANALYSIS_FNS = ("_spearman", "clustered_bootstrap_spearman")


def _fn_body_dump(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
            ):
                body = body[1:]
            return ast.dump(ast.Module(body=body, type_ignores=[]))
    raise AssertionError(f"function {name} not found")


def _source_from_722_branch() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "show", "issue-722-tf-margin:scripts/issue722_tf_margin_analysis.py"],
            cwd=PROJECT_ROOT,
        ).decode()
    except Exception:
        return None


def test_n_boot_is_2000():
    """N_BOOT matches #722's default verbatim (plan v6 nit-2)."""
    assert tfa.N_BOOT == 2000


@pytest.mark.parametrize("name", VENDORED_ANALYSIS_FNS)
def test_vendored_analysis_fn_is_byte_identical_to_722(name):
    src = _source_from_722_branch()
    if src is None:
        pytest.skip("issue-722-tf-margin branch not fetched in this checkout")
    mine = Path(tfa.__file__).read_text()
    assert _fn_body_dump(src, name) == _fn_body_dump(mine, name), (
        f"vendored {name} body diverged from the #722 branch original (plan §4.3)."
    )


def test_clustered_bootstrap_spearman_reference_value():
    """The vendored bootstrap reproduces a deterministic value on a fixed seed/dataset."""
    rng = np.random.default_rng(0)
    x = np.arange(24, dtype=float) + rng.normal(0, 0.3, 24)
    y = np.arange(24, dtype=float) + rng.normal(0, 0.3, 24)  # strongly correlated
    fams = [f"fam{i % 6}" for i in range(24)]
    res = tfa.clustered_bootstrap_spearman(x, y, fams, n_boot=500, seed=42)
    assert res["point"] is not None and res["point"] > 0.8
    assert res["ci_lo"] is not None and res["ci_lo"] > 0  # CI excludes zero for a strong corr
    assert res["n_families"] == 6
    # Determinism: same seed -> same bounds.
    res2 = tfa.clustered_bootstrap_spearman(x, y, fams, n_boot=500, seed=42)
    assert res == res2


def test_g0_correctness_gate_halts_on_mismatch(monkeypatch, tmp_path):
    """recompute_g0_percell that reproduces the WRONG aggregate rho -> G0CorrectnessError."""
    # Synthetic rows: g0 anti-correlated with G -> aggregate Spearman far from committed 0.13.
    n = 30
    rows = [{"source": "s", "target": f"t{i}", "g0": float(i), "G": float(n - i)} for i in range(n)]
    g0_vec = np.array([r["g0"] for r in rows])
    G_vec = np.array([r["G"] for r in rows])
    monkeypatch.setattr(
        tfa,
        "recompute_g0_percell",
        lambda cells, g_meta, sigma_c, lam, behavior: {
            "rows": rows,
            "g0_vec": g0_vec,
            "G_vec": G_vec,
        },
    )
    _stub_analysis_loaders(monkeypatch)
    with pytest.raises(tfa.G0CorrectnessError):
        tfa.run_gate_vs_tf_margin(
            per_cell_dir=tmp_path / "pc",
            tensors_dir=tmp_path / "tn",
            behaviors=["em"],
            layer=14,
            out_dir=tmp_path / "out",
            committed_base_g_rho={"em": 0.13},
        )


def test_g0_correctness_gate_passes_and_validation_flags(monkeypatch, tmp_path):
    """A matching g0 aggregate PASSes; the measurement-validity gate flags passed correctly."""
    n = 30
    # g0 and G engineered so aggregate Spearman(g0, G) ~ 1.0; we pin committed rho to 1.0
    # (tol 0.02) so the correctness gate passes on this synthetic universe.
    rows = [{"source": "s", "target": f"t{i}", "g0": float(i), "G": float(i)} for i in range(n)]
    g0_vec = np.array([r["g0"] for r in rows])
    G_vec = np.array([r["G"] for r in rows])
    monkeypatch.setattr(
        tfa,
        "recompute_g0_percell",
        lambda cells, g_meta, sigma_c, lam, behavior: {
            "rows": rows,
            "g0_vec": g0_vec,
            "G_vec": G_vec,
        },
    )
    # tf_margin_leak strongly tracks G (validation should PASS) and g0 (headline > 0).
    tf_cells = {("s", f"t{i}"): {"tf_margin_leak": float(i) + 0.01 * (i % 3)} for i in range(n)}
    monkeypatch.setattr(tfa, "load_tf_margin_leak", lambda per_cell_dir, behavior: tf_cells)
    _stub_analysis_loaders(monkeypatch)
    # multiple families so the clustered bootstrap has >=2 clusters
    monkeypatch.setattr(
        "explore_persona_space.analysis.issue667.gate_chain.family_of",
        lambda cid: f"fam{int(cid[1:]) % 6}",
        raising=False,
    )

    res = tfa.run_gate_vs_tf_margin(
        per_cell_dir=tmp_path / "pc",
        tensors_dir=tmp_path / "tn",
        behaviors=["em"],
        layer=14,
        out_dir=tmp_path / "out",
        committed_base_g_rho={"em": 1.0},
        # This test's synthetic universe (30 t{i} cells) is NOT the production
        # off-diagonal grid, so bypass the round-2 cell-coverage gate (it targets
        # the g0-correctness + validation gates, not coverage).
        skip_store_pin=True,
    )
    assert res["validation"]["em"]["passed"] is True
    assert res["headline"]["em"]["rho"] is not None and res["headline"]["em"]["rho"] > 0
    # output JSONs written
    assert (tmp_path / "out" / "rho_gate_vs_tf_margin.json").exists()
    assert (tmp_path / "out" / "rho_margin_vs_rate.json").exists()
    assert (tmp_path / "out" / "g0_percell.json").exists()
    # CONCERN 2: the aggregate deliverables are written on every run.
    assert (tmp_path / "out" / "margins.json").exists()
    assert (tmp_path / "out" / "tf_margin_leak.json").exists()


def _stub_analysis_loaders(monkeypatch):
    """Stub the HF/store loaders + the B3 unit test so the gate logic runs on synthetics."""
    import issue667_analysis as ia

    monkeypatch.setattr(ia, "load_cells", lambda tensors_dir, behavior, layer: {})
    monkeypatch.setattr(ia, "load_g_meta", lambda: {"per_cell": {}})
    monkeypatch.setattr(ia, "load_sigma_c", lambda layer: np.eye(4))
    monkeypatch.setattr(
        "explore_persona_space.analysis.issue667.gate_chain.default_lambda",
        lambda sigma_c, frac=1e-2: 0.0116,
        raising=False,
    )
    monkeypatch.setattr(
        "explore_persona_space.analysis.issue667.gate_chain.whitened_gate_reduction_unit_test",
        lambda *a, **k: None,
        raising=False,
    )
    # family_of default (overridden in the pass test for multi-family).
    monkeypatch.setattr(
        "explore_persona_space.analysis.issue667.gate_chain.family_of",
        lambda cid: f"fam{hash(cid) % 6}",
        raising=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Round-2 BLOCKER 2 (cell coverage) + CONCERN 2 (aggregate deliverables).
# ─────────────────────────────────────────────────────────────────────────────

_R2_ROWS = [
    {"source": "s0", "target": "t2", "g0": 0.0, "G": 0.0},
    {"source": "s0", "target": "t3", "g0": 0.5, "G": 0.5},
    {"source": "s1", "target": "t2", "g0": 0.7, "G": 0.7},
    {"source": "s1", "target": "t3", "g0": 1.0, "G": 1.0},
]
_R2_EXPECTED = {(r["source"], r["target"]) for r in _R2_ROWS}


def _install_r2_mocks(monkeypatch, *, tf_cells):
    """Round-2 helper: perfectly-correlated 4-cell g0/G universe + expected-key stub."""
    g0_vec = np.array([r["g0"] for r in _R2_ROWS])
    G_vec = np.array([r["G"] for r in _R2_ROWS])
    monkeypatch.setattr(
        tfa,
        "recompute_g0_percell",
        lambda cells, g_meta, sigma_c, lam, behavior: {
            "rows": _R2_ROWS,
            "g0_vec": g0_vec,
            "G_vec": G_vec,
        },
    )
    monkeypatch.setattr(tfa, "load_tf_margin_leak", lambda per_cell_dir, behavior: tf_cells)
    monkeypatch.setattr(tfa, "expected_off_diagonal_keys", lambda behavior: set(_R2_EXPECTED))
    _stub_analysis_loaders(monkeypatch)


def _r2_full_tf_cells(leak=0.6):
    return {
        k: {
            "tf_margin_leak": leak,
            "margin_base": 0.1,
            "margin_trained": 0.1 + leak,
            "n_pos": 40,
            "n_neg": 40,
        }
        for k in _R2_EXPECTED
    }


def test_r2_missing_cell_broad_em_hard_fails(monkeypatch, tmp_path):
    """BLOCKER 2: a missing em cell -> CellCoverageError, no headline JSON written."""
    tf_cells = _r2_full_tf_cells()
    del tf_cells[next(iter(tf_cells))]  # simulate a partial extract / upload gap
    _install_r2_mocks(monkeypatch, tf_cells=tf_cells)
    with pytest.raises(tfa.CellCoverageError) as ei:
        tfa.run_gate_vs_tf_margin(
            per_cell_dir=tmp_path / "pc",
            tensors_dir=tmp_path / "tn",
            behaviors=["em"],
            layer=14,
            out_dir=tmp_path / "out",
            committed_base_g_rho={"em": 1.0},
            skip_store_pin=False,
            fact_dropped=False,
        )
    assert "cell-coverage gate FAIL for em" in str(ei.value)
    assert not (tmp_path / "out" / "rho_gate_vs_tf_margin.json").exists()


def test_r2_missing_cell_fact_softdrops_when_dropped(monkeypatch, tmp_path):
    """BLOCKER 2 / CONCERN 1: a missing fact cell + fact_dropped -> SOFT-DROP (no raise)."""
    tf_cells = _r2_full_tf_cells()
    del tf_cells[next(iter(tf_cells))]
    _install_r2_mocks(monkeypatch, tf_cells=tf_cells)
    res = tfa.run_gate_vs_tf_margin(
        per_cell_dir=tmp_path / "pc",
        tensors_dir=tmp_path / "tn",
        behaviors=["fact"],
        layer=14,
        out_dir=tmp_path / "out",
        committed_base_g_rho={"fact": 1.0},
        skip_store_pin=False,
        fact_dropped=True,
    )
    assert res["fact_dropped_from_headline"] is True
    assert "fact" not in res["headline"]
    meta = json.loads((tmp_path / "out" / "rho_gate_vs_tf_margin.json").read_text())["metadata"]
    assert meta["fact_dropped_from_headline"] is True


def test_r2_missing_cell_fact_hard_fails_when_not_dropped(monkeypatch, tmp_path):
    """A missing fact cell WITHOUT the drop flag is an unexplained partial -> HARD-FAIL."""
    tf_cells = _r2_full_tf_cells()
    del tf_cells[next(iter(tf_cells))]
    _install_r2_mocks(monkeypatch, tf_cells=tf_cells)
    with pytest.raises(tfa.CellCoverageError):
        tfa.run_gate_vs_tf_margin(
            per_cell_dir=tmp_path / "pc",
            tensors_dir=tmp_path / "tn",
            behaviors=["fact"],
            layer=14,
            out_dir=tmp_path / "out",
            committed_base_g_rho={"fact": 1.0},
            skip_store_pin=False,
            fact_dropped=False,
        )


def test_r2_full_coverage_writes_aggregate_deliverables(monkeypatch, tmp_path):
    """CONCERN 2: complete coverage -> margins.json + tf_margin_leak.json, right schema."""
    tf_cells = _r2_full_tf_cells(leak=0.7)
    _install_r2_mocks(monkeypatch, tf_cells=tf_cells)
    res = tfa.run_gate_vs_tf_margin(
        per_cell_dir=tmp_path / "pc",
        tensors_dir=tmp_path / "tn",
        behaviors=["em"],
        layer=14,
        out_dir=tmp_path / "out",
        committed_base_g_rho={"em": 1.0},
        skip_store_pin=False,
        fact_dropped=False,
    )
    margins_p = tmp_path / "out" / "margins.json"
    leak_p = tmp_path / "out" / "tf_margin_leak.json"
    assert margins_p.is_file() and leak_p.is_file()
    margins = json.loads(margins_p.read_text())["per_behavior"]["em"]
    leak = json.loads(leak_p.read_text())["per_behavior"]["em"]
    assert set(margins.keys()) == {f"{s}|{t}" for (s, t) in _R2_EXPECTED}
    a_key = next(iter(margins))
    assert set(margins[a_key]) >= {"margin_base", "margin_trained", "tf_margin_leak"}
    assert leak[a_key] == 0.7
    assert res["margins"]["em"][a_key]["tf_margin_leak"] == 0.7


def test_r2_smoke_skip_store_pin_bypasses_coverage_gate(monkeypatch, tmp_path):
    """--skip-store-pin subsets the grid, so the coverage gate is inert (no raise)."""
    one = next(iter(_R2_EXPECTED))
    tf_cells = {one: {"tf_margin_leak": 0.3, "margin_base": 0.1, "margin_trained": 0.4}}
    _install_r2_mocks(monkeypatch, tf_cells=tf_cells)
    res = tfa.run_gate_vs_tf_margin(
        per_cell_dir=tmp_path / "pc",
        tensors_dir=tmp_path / "tn",
        behaviors=["em"],
        layer=14,
        out_dir=tmp_path / "out",
        committed_base_g_rho={"em": 1.0},
        skip_store_pin=True,
        fact_dropped=False,
    )
    assert "em" in res["headline"]


def test_r2_cli_maps_cell_coverage_to_rc4(monkeypatch, tmp_path):
    """The analysis CLI maps CellCoverageError to rc=4 (the dispatcher's HALT code)."""
    tf_cells = _r2_full_tf_cells()
    del tf_cells[next(iter(tf_cells))]
    _install_r2_mocks(monkeypatch, tf_cells=tf_cells)
    # em uses the real committed rho (~1.0 agg here matches within tol via the stub
    # universe), so pin the committed reference to 1.0 to isolate the coverage HALT.
    monkeypatch.setattr(tfa, "COMMITTED_BASE_G_RHO", {"em": 1.0})
    monkeypatch.setattr(tfa, "_fact_dropped_sentinel_present", lambda: False)
    monkeypatch.setattr(
        "explore_persona_space.orchestrate.env.load_dotenv", lambda *a, **k: None, raising=False
    )
    argv = [
        "prog",
        "--behaviors",
        "em",
        "--per-cell-dir",
        str(tmp_path / "pc"),
        "--tensors-dir",
        str(tmp_path / "tn"),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert tfa.main() == 4
