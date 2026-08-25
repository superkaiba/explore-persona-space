"""#1901 mlp-scaling-densify round-2 pure functions (review fixes C2/M1/M2/C3).

Pins, on synthetic fixtures (no GPU, no downloads, no staged stores):
- ``_validate_run_shape`` (C2): production asserts the exact registered rung
  set / seed set / 1,920-chunk capture / captured==manifest / 963,444 pool;
  smoke asserts its OWN registered expectations (never skip-under-smoke).
- ``_endpoint_verdict`` (M2): the plan-v13 §3 lattice — seed-paired slopes,
  S_mlp seed mean, S_ridge, D_gap = (S_mlp - S_ridge) - 0.01, Confirmed ⇔
  D_gap >= 0 — with exact arithmetic, plus structured computed:false records.
- ``_dense_fingerprint`` metrics identity (M1): a changed metric instrument
  breaks the resume equality.
- The C3 dead-branch guarantee: registered smoke rungs never intersect a
  DENSE_PARITY_ANCHORS key.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from issue1901_paper_densify_mlp import (  # noqa: E402
    DENSE_PARITY_ANCHORS,
    PRODUCTION_DENSE_NS,
    SMOKE_RUNG_SPECS,
    GateVerdict,
    _annotate_g2_percell,
    _dense_fingerprint,
    _endpoint_verdict,
    _validate_run_shape,
)

PROD_RUNGS = [(n, "scale7" if n <= 25_000 else "n1m") for n in PRODUCTION_DENSE_NS]


def _split(cap=959_844, man=959_844):
    return {"n_new_captured": cap, "n_new_manifest": man}


def _prod_kwargs(**over):
    kw = dict(
        smoke=False,
        smoke_chunks=0,
        n_capture_files=1_920,
        seed=42,
        endpoint_seeds=[43, 44],
        rung_specs=PROD_RUNGS,
        split=_split(),
        n_pool_full=963_444,
    )
    kw.update(over)
    return kw


def _smoke_kwargs(**over):
    kw = dict(
        smoke=True,
        smoke_chunks=4,
        n_capture_files=4,
        seed=42,
        endpoint_seeds=[43, 44],
        rung_specs=list(SMOKE_RUNG_SPECS),
        split=_split(cap=2_000, man=959_844),
        n_pool_full=5_600,
    )
    kw.update(over)
    return kw


def test_shape_production_passes_and_audits():
    audit = _validate_run_shape(**_prod_kwargs())
    assert audit["mode"] == "production"
    assert audit["seeds"] == [42, 43, 44]
    assert audit["n_capture_files"] == 1_920
    assert audit["n_pool_full"] == 963_444


@pytest.mark.parametrize(
    "over, frag",
    [
        (dict(rung_specs=PROD_RUNGS[:-1]), "rung set"),
        (dict(n_capture_files=1_919), "chunk count"),
        (dict(split=_split(cap=900_000)), "captured rows"),
        (dict(n_pool_full=963_443), "train pool"),
    ],
)
def test_shape_production_violations_raise(over, frag):
    with pytest.raises(RuntimeError, match=frag):
        _validate_run_shape(**_prod_kwargs(**over))


def test_shape_seed_set_pinned_in_both_modes():
    with pytest.raises(RuntimeError, match="seed set"):
        _validate_run_shape(**_prod_kwargs(endpoint_seeds=[43, 45]))
    with pytest.raises(RuntimeError, match="duplicate seeds"):
        _validate_run_shape(**_smoke_kwargs(endpoint_seeds=[43, 43]))
    with pytest.raises(RuntimeError, match="seed set"):
        _validate_run_shape(**_smoke_kwargs(endpoint_seeds=[43]))


def test_shape_smoke_asserts_own_expectations_never_skips():
    audit = _validate_run_shape(**_smoke_kwargs())
    assert audit["mode"] == "smoke"
    with pytest.raises(RuntimeError, match="smoke rung specs"):
        _validate_run_shape(**_smoke_kwargs(rung_specs=[(1_000, "scale7")]))
    with pytest.raises(RuntimeError, match="--smoke-chunks"):
        _validate_run_shape(**_smoke_kwargs(n_capture_files=5))
    with pytest.raises(RuntimeError, match="partial pool"):
        _validate_run_shape(**_smoke_kwargs(split=_split(cap=0)))


def test_registered_smoke_rungs_disjoint_from_parity_anchors():
    """C3 dead-branch guarantee at the constant level: no anchor key is
    reachable at a registered smoke rung, so _rung_parity needs (and has)
    no smoke branch."""
    anchor_ns = {n for _, n in DENSE_PARITY_ANCHORS}
    assert not (anchor_ns & {n for n, _ in SMOKE_RUNG_SPECS})


# ── _endpoint_verdict (M2) ───────────────────────────────────────────────────────


def _mlp_cell(seed_r2s: dict[str, float]) -> dict:
    return {
        "seeds": {s: {"test_r2": r} for s, r in seed_r2s.items()},
        "test_r2": seed_r2s["42"],
    }


def test_endpoint_verdict_confirmed_exact_arithmetic():
    per_n = {
        "50000": {
            "mlp": _mlp_cell({"42": 0.70, "43": 0.71, "44": 0.72}),
            "ridge": {"test_r2": 0.755},
        },
        "500000": {
            "mlp": _mlp_cell({"42": 0.74, "43": 0.73, "44": 0.75}),
            "ridge": {"test_r2": 0.760},
        },
    }
    v = _endpoint_verdict(per_n, [50_000, 500_000], [42, 43, 44])
    assert v["computed"] is True
    assert v["seed_paired_slopes"]["42"] == pytest.approx(0.04)
    assert v["seed_paired_slopes"]["43"] == pytest.approx(0.02)
    assert v["seed_paired_slopes"]["44"] == pytest.approx(0.03)
    assert v["S_mlp"] == pytest.approx(0.03)
    assert v["S_ridge"] == pytest.approx(0.005)
    assert v["D_gap"] == pytest.approx(0.03 - 0.005 - 0.01)
    assert v["verdict"] == "Confirmed"
    assert v["sub_reads"]["ridge_decline_driven"] is False


def test_endpoint_verdict_falsified_and_plateau_both():
    per_n = {
        "50000": {
            "mlp": _mlp_cell({"42": 0.70, "43": 0.70, "44": 0.70}),
            "ridge": {"test_r2": 0.75},
        },
        "500000": {
            "mlp": _mlp_cell({"42": 0.705, "43": 0.705, "44": 0.705}),
            "ridge": {"test_r2": 0.75},
        },
    }
    v = _endpoint_verdict(per_n, [50_000, 500_000], [42, 43, 44])
    assert v["D_gap"] == pytest.approx(0.005 - 0.0 - 0.01)
    assert v["verdict"] == "Falsified"
    assert v["sub_reads"]["plateau_both"] is True


def test_endpoint_verdict_ridge_decline_driven_flag():
    per_n = {
        "50000": {"mlp": _mlp_cell({"42": 0.70}), "ridge": {"test_r2": 0.75}},
        "500000": {"mlp": _mlp_cell({"42": 0.695}), "ridge": {"test_r2": 0.70}},
    }
    v = _endpoint_verdict(per_n, [50_000, 500_000], [42])
    # S_mlp = -0.005, S_ridge = -0.05 => D_gap = 0.045 - 0.01 > 0: Confirmed,
    # but driven by the ridge decline (S_mlp <= 0) — the flagged sub-read.
    assert v["verdict"] == "Confirmed"
    assert v["sub_reads"]["ridge_decline_driven"] is True


def test_endpoint_verdict_structured_noncomputed_records():
    # Smoke shape: a single endpoint rung.
    v = _endpoint_verdict({}, [2_000], [42, 43, 44])
    assert v["computed"] is False and "endpoint" in v["note"]
    # Missing rung.
    v = _endpoint_verdict(
        {"50000": {"mlp": _mlp_cell({"42": 0.7}), "ridge": {"test_r2": 0.7}}},
        [50_000, 500_000],
        [42],
    )
    assert v["computed"] is False
    # Missing per-seed blocks.
    per_n = {
        "50000": {"mlp": {"test_r2": 0.70}, "ridge": {"test_r2": 0.75}},
        "500000": {"mlp": {"test_r2": 0.71}, "ridge": {"test_r2": 0.75}},
    }
    v = _endpoint_verdict(per_n, [50_000, 500_000], [42, 43, 44])
    assert v["computed"] is False and "per-seed" in v["note"]
    # Missing ONE seed block.
    per_n = {
        "50000": {"mlp": _mlp_cell({"42": 0.70, "43": 0.71}), "ridge": {"test_r2": 0.75}},
        "500000": {"mlp": _mlp_cell({"42": 0.71, "43": 0.72}), "ridge": {"test_r2": 0.75}},
    }
    v = _endpoint_verdict(per_n, [50_000, 500_000], [42, 43, 44])
    assert v["computed"] is False and "44" in v["note"]


# ── _dense_fingerprint metric identity (M1) ─────────────────────────────────────


def test_dense_fingerprint_carries_metric_identity():
    args = argparse.Namespace(seed=42, seed_b=0, smoke_chunks=0, ridge_block=8192, n_boot=1000)
    metrics = {
        "n_boot": 1000,
        "bootstrap": "FFC._bootstrap_recon_ci (context resample; per-cell seeded)",
        "whiten": {
            "pool_sha256": "p" * 64,
            "lam": 0.1,
            "helper": "null_battery.shrunk_cholesky_from_cov",
        },
        "k_csls": 10,
        "knn_ks": [1, 5, 10, 50],
        "val_sha256": "v" * 64,
        "test_sha256": "t" * 64,
        "eval_store": "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture",
    }
    common = dict(
        n=50_000,
        source="n1m",
        sel_name="lmsys_50k",
        sel_sha="s" * 64,
        store_revision="a" * 40,
        arm="mlp",
        seeds=[42, 43, 44],
    )
    fp = _dense_fingerprint(args, metrics=metrics, **common)
    assert fp["metrics"] == metrics
    # A changed metric instrument (n_boot) MUST break the resume equality.
    fp2 = _dense_fingerprint(args, metrics={**metrics, "n_boot": 500}, **common)
    assert fp2 != fp


# ── _annotate_g2_percell (M3) ───────────────────────────────────────────────────


def test_annotate_g2_percell_durably_annotates_recorded_cells(tmp_path):
    perfit = tmp_path / "perfit"
    perfit.mkdir()
    for n_pt in (150_000, 500_000):
        (perfit / f"dense_L19_n{n_pt}_ridge.json").write_text(
            json.dumps({"test_r2": 0.75, "fingerprint": {"arm": "ridge", "n": n_pt}})
        )
    g2 = GateVerdict(
        verdict="FALLBACK-PARITY-PASS",
        downgrade_recorded=True,
        detail={
            "points": {
                "lmsys_150k": {"sha_match": False, "parity_within_tol": True},
                "lmsys_500k": {"sha_match": True, "parity_within_tol": True},
            }
        },
    )
    annotated = _annotate_g2_percell(tmp_path, "dense", g2)
    assert annotated == ["dense_L19_n150000_ridge.json", "dense_L19_n500000_ridge.json"]
    cell = json.loads((perfit / "dense_L19_n150000_ridge.json").read_text())
    assert cell["g2"]["verdict"] == "FALLBACK-PARITY-PASS"
    assert cell["g2"]["downgrade_recorded"] is True
    assert cell["g2"]["point"]["sha_match"] is False
    # Resume-safety: the fingerprint key is untouched by the annotation.
    assert cell["fingerprint"] == {"arm": "ridge", "n": 150_000}
    # Idempotent + missing-file tolerant (annotates only what exists).
    (perfit / "dense_L19_n500000_ridge.json").unlink()
    assert _annotate_g2_percell(tmp_path, "dense", g2) == ["dense_L19_n150000_ridge.json"]
