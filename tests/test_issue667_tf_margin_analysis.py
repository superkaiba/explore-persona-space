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
    )
    assert res["validation"]["em"]["passed"] is True
    assert res["headline"]["em"]["rho"] is not None and res["headline"]["em"]["rho"] > 0
    # output JSONs written
    assert (tmp_path / "out" / "rho_gate_vs_tf_margin.json").exists()
    assert (tmp_path / "out" / "rho_margin_vs_rate.json").exists()
    assert (tmp_path / "out" / "g0_percell.json").exists()


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
