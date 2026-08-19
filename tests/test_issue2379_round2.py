"""Round-2 regression pins for issue #2379 (unit B: mapfit / judge / analysis fixes).

Each test pins a permanent invariant a round-2 BLOCKER fix added (fails
pre-fix, passes post-fix):
  * hollow-prediction-parity  -> fit->persist->reload->predict round-trip
    catches a corrupted component file (mapfit `_assert_disk_roundtrip`).
  * unsafe-passb-deserialization -> `_torch_load_constrained` loads
    numpy-bearing bundles under the safe_globals allowlist and REFUSES
    non-allowlisted pickled types (never weights_only=False).
  * cached-artifact-schema-coverage -> `_validate_predictor_bundles` raises on
    every missing unit-A producer key / shape mismatch.
  * p7-estimability-coupling -> `verdict_for_setting` precedence lattice
    (install fail dominates; ctx vs joint estimability split).
  * install-rate fail-closed -> `em_install_check` / `caps_install_check`
    count indeterminate rates as failing.
  * G1 registered denominator -> `run_gate` fails CLOSED, naming the
    non-finite language, never nanmean-dropping it.
  * kappa deterministic allocation -> `_kappa_allocate` exactness/capacity.
  * judge phase idempotency -> `phase_probe` skip-at-entry (zero API calls).
  * mapfit resume predicate -> `_resume_ok` keys on bundle identity + regime.

Adoptable shape: repo-root paths, zero network, tmp_path fixtures; all data
synthetic/benign (content hygiene: no corpus text enters this file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2379_analysis as ana  # noqa: E402
import issue2379_judge as judge  # noqa: E402
import issue2379_mapfit as mapfit  # noqa: E402

NAN = float("nan")


# ---------------------------------------------------------------------------
# mapfit: fit -> persist -> reload -> predict parity (r1 blocker
# hollow-prediction-parity) + resume predicate + pass-B constrained load
# ---------------------------------------------------------------------------
def _tiny_fit(seed: int = 0, n: int = 60, d: int = 8):
    """Tiny real fit through the production `_fit_unit_worker` (n_train > d, so
    the GCV selection is well-posed)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    w_true = rng.standard_normal((d, d))
    y = x @ w_true + 3.0 + 0.01 * rng.standard_normal((n, d))  # +3 => ymu != xmu
    idx = rng.permutation(n)
    task = {
        "mapset": "toy_ctx",
        "layer": 3,
        "x16": x.astype(np.float16),
        "y16": y.astype(np.float16),
        "tr_idx": idx[:48],
        "ev_idx": idx[48:],
    }
    return task, mapfit._fit_unit_worker(task)


def test_fit_persist_reload_roundtrip_clean_passes(tmp_path):
    _, rec = _tiny_fit()
    out = mapfit._persist_unit(tmp_path, rec, n_rows=60, bundle_ident="sha256:test")
    assert out.exists() and out.suffix == ".npz"
    mapfit._assert_disk_roundtrip(
        tmp_path, "toy_ctx", 3, rec["x_ev_sample"], rec["pred_sample"], what="clean"
    )


def test_corrupted_component_file_fails_disk_roundtrip(tmp_path):
    """Swap xmu <-> ymu in the persisted npz: the reloaded components no longer
    reproduce the in-memory prediction, and the round-trip gate must raise
    (pre-fix the r1 'round-trip' re-predicted from the same in-memory dict and
    could never fail on a corrupted file)."""
    _, rec = _tiny_fit()
    out = mapfit._persist_unit(tmp_path, rec, n_rows=60, bundle_ident="sha256:test")
    with np.load(out) as z:
        fields = {k: z[k] for k in z.files}
    assert not np.allclose(fields["xmu"], fields["ymu"])  # corruption is material
    fields["xmu"], fields["ymu"] = fields["ymu"], fields["xmu"]
    np.savez(out.with_name(out.stem + ".tmp.npz"), **fields)
    out.with_name(out.stem + ".tmp.npz").replace(out)
    with pytest.raises(RuntimeError, match="disk round-trip parity FAILED"):
        mapfit._assert_disk_roundtrip(
            tmp_path, "toy_ctx", 3, rec["x_ev_sample"], rec["pred_sample"], what="corrupt"
        )


def test_prediction_parity_oracle_catches_wrong_components():
    """`_assert_prediction_parity` compares stored-equivalent components against
    the fit-NATIVE #2254 oracle — a wrong W in the comp dict must fail it."""
    task, rec = _tiny_fit()
    fit = {
        "W": np.asarray(rec["W32"], dtype=np.float64),
        "xmu": rec["xmu"],
        "xsd": rec["xsd"],
        "ymu": rec["ymu"],
    }
    x_ev = np.asarray(task["x16"], dtype=np.float64)[task["ev_idx"]]
    good = mapfit._comp_from_arrays(rec["W32"], rec["xmu"], rec["xsd"], rec["ymu"])
    mapfit._assert_prediction_parity(good, fit, x_ev, what="clean")
    bad = mapfit._comp_from_arrays(rec["W32"] * 2.0, rec["xmu"], rec["xsd"], rec["ymu"])
    with pytest.raises(RuntimeError, match="prediction-parity FAILED"):
        mapfit._assert_prediction_parity(bad, fit, x_ev, what="stale-W")


def test_resume_predicate_keys_on_bundle_identity_and_regime(tmp_path):
    _, rec = _tiny_fit()
    mapfit._persist_unit(tmp_path, rec, n_rows=60, bundle_ident="sha256:A")
    p = mapfit.comp_path(tmp_path, "toy_ctx", 3)
    ok = dict(mapset="toy_ctx", layer=3, n_rows=60, bundle_ident="sha256:A")
    assert mapfit._resume_ok(p, **ok)
    assert not mapfit._resume_ok(p, **{**ok, "bundle_ident": "sha256:B"})  # regenerated bundle
    assert not mapfit._resume_ok(p, **{**ok, "n_rows": 61})
    assert not mapfit._resume_ok(p, **{**ok, "layer": 4})  # file for another unit
    assert not mapfit._resume_ok(tmp_path / "absent.npz", **ok)


class _NotAllowlisted:
    """Arbitrary pickled type the constrained loader must refuse."""


def test_constrained_load_accepts_numpy_under_allowlist(tmp_path):
    p = tmp_path / "np_bundle.pt"
    torch.save({"cx_last": np.arange(12.0).reshape(3, 4)}, p)
    # Guard against vacuous fallback: the bare weights_only load must refuse
    # raw numpy payloads on this torch version (else the allowlist branch is
    # dead code and this test proves nothing).
    with pytest.raises(Exception):  # noqa: B017 — torch's refusal type varies by version
        torch.load(p, map_location="cpu", weights_only=True)
    tb = mapfit._torch_load_constrained(p)
    assert isinstance(tb["cx_last"], np.ndarray)
    assert tb["cx_last"].shape == (3, 4)


def test_constrained_load_refuses_non_allowlisted_types(tmp_path):
    p = tmp_path / "evil.pt"
    torch.save({"x": _NotAllowlisted()}, p)
    with pytest.raises(RuntimeError, match="constrained weights_only load refused"):
        mapfit._torch_load_constrained(p)


# ---------------------------------------------------------------------------
# mapfit: unit-A producer bundle schema validation (r1 blocker
# cached-artifact-schema-coverage)
# ---------------------------------------------------------------------------
def _good_bundles(n_rows: int = 4, n_l: int = 2, d: int = 3) -> dict:
    g = torch.manual_seed(0)  # noqa: F841 — deterministic tensors below
    g_meta = [{"trigger_idx": 0, "trigger_label": "t0", "q_sim_idx": i} for i in range(n_rows)]
    c_meta = [
        {
            "cell_idx": i,
            "trigger_idx": 0,
            "trigger_label": "t0",
            "q_sim_idx": i,
            "rollout_idx": 0,
        }
        for i in range(n_rows)
    ]
    return {
        "grid": {"v_c": torch.randn(n_rows, n_l, d), "row_meta": g_meta},
        "mu": {
            "mu_train": torch.randn(n_l, d),
            "mu_a_train": torch.randn(n_l, d),
            "n_c": 5,
            "n_a": 5,
        },
        "ceiling": {"v_a": torch.randn(n_rows, n_l, d), "row_meta": c_meta, "drop_stats": {}},
    }


def test_predictor_bundle_validation_clean_passes():
    mapfit._validate_predictor_bundles("toy", _good_bundles())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda b: b["ceiling"].pop("drop_stats"), r"ceiling\.pt missing keys.*drop_stats"),
        (lambda b: b["mu"].pop("n_c"), r"mu\.pt missing keys.*n_c"),
        (lambda b: b["grid"]["row_meta"][0].pop("q_sim_idx"), r"grid\.pt row_meta keys"),
        (lambda b: b["ceiling"]["row_meta"][0].pop("cell_idx"), r"ceiling\.pt row_meta keys"),
        (lambda b: b["mu"].__setitem__("mu_train", torch.randn(3, 3)), r"mu\.pt: mu_train"),
        (lambda b: b["mu"].__setitem__("n_c", 0), r"empty mean"),
        (lambda b: b["ceiling"].__setitem__("v_a", torch.randn(4, 3, 3)), r"layer/hidden shape"),
    ],
)
def test_predictor_bundle_validation_raises_per_defect(mutate, match):
    b = _good_bundles()
    mutate(b)
    with pytest.raises(RuntimeError, match=match):
        mapfit._validate_predictor_bundles("toy", b)


# ---------------------------------------------------------------------------
# analysis: verdict lattice precedence + fail-closed install checks (round-2
# blockers p7-estimability-coupling + install-rate fail-closed)
# ---------------------------------------------------------------------------
def test_verdict_lattice_precedence():
    v = ana.verdict_for_setting
    # install STRUCTURAL fail dominates everything, estimability included
    assert v(NAN, True, NAN, NAN, NAN, 0, 0) == ana.VERDICT_REPL_FAILED
    # no ctx-estimable conditions and no install fail -> non-estimable
    assert v(NAN, False, NAN, NAN, NAN, 0, 0) == ana.VERDICT_NON_ESTIMABLE
    # manip PASSes on ctx subset but zero jointly-estimable -> non-estimable
    assert v(0.9, False, NAN, NAN, NAN, 3, 0) == ana.VERDICT_NON_ESTIMABLE
    # manip FAIL evaluated on the ctx subset even with zero joint conditions
    assert v(0.1, False, NAN, NAN, NAN, 3, 0) == ana.VERDICT_REPL_FAILED
    # CI reads (joint subset)
    assert v(0.9, False, 0.2, 0.05, 0.40, 3, 3) == ana.VERDICT_ANSWER
    assert v(0.9, False, -0.2, -0.40, -0.05, 3, 3) == ana.VERDICT_CONTEXT
    assert v(0.9, False, 0.2, -0.10, 0.40, 3, 3) == ana.VERDICT_COMPARABLE


def _em_rates(rates_by_model: dict) -> dict:
    return {"rates": {m: {ana.EM_EMPTY_LABEL: {"em_rate": r}} for m, r in rates_by_model.items()}}


def test_em_install_check_fail_closed_on_indeterminate():
    stems = [f"m{i}" for i in range(5)]
    # 2 indeterminate (None / non-finite) + 1 genuine fail = 3 >= floor(3)
    out = ana.em_install_check(
        _em_rates({"m0": None, "m1": float("nan"), "m2": 0.5, "m3": 0.05, "m4": 0.05}), stems
    )
    assert out["n_indeterminate"] == 2
    assert out["n_failing"] == 3
    assert out["structural_fail"] is True
    assert out["per_model"]["m0"] == {
        "empty_prompt_em_rate": None,
        "fail": None,
        "indeterminate": True,
    }
    clean = ana.em_install_check(_em_rates(dict.fromkeys(stems, 0.05)), stems)
    assert clean["structural_fail"] is False and clean["n_indeterminate"] == 0


def test_caps_install_check_fail_closed_on_non_bool_pass():
    stems = ["c0", "c1", "c2"]
    shards = {
        "c0": {"install_check": {"pass": None}},  # indeterminate -> counts failing
        "c1": {"install_check": {"pass": False}},
        "c2": {"install_check": {"pass": True}},
    }
    out = ana.caps_install_check(shards, stems)
    assert out["n_indeterminate"] == 1
    assert out["n_failing"] == 2
    assert out["structural_fail"] is True  # 2 >= CAPS_INSTALL_FAIL_MODELS(2)


# ---------------------------------------------------------------------------
# analysis: Gate G1 registered denominator (round-2 Major — no nanmean drop)
# ---------------------------------------------------------------------------
def _write_gate_fixture(root: Path, models: list[str], rates_of: dict[str, list[float]]):
    """Tiny synthetic P2/P3 fixtures: 3 triggers x 2 rows, n_l=2, d=4; per-trigger
    ctx Train-Ref cosine strictly decreasing in trigger index."""
    captures = root / "captures"
    caps_dir = root / "caps_shards"
    caps_dir.mkdir(parents=True)
    labels = ["T0", "T1", "T2"]
    for m in models:
        mdir = captures / m
        mdir.mkdir(parents=True)
        mu = np.zeros((2, 4))
        mu[:, 0] = 1.0
        rows, meta = [], []
        for t in range(3):
            theta = 0.3 * t
            v = np.zeros((2, 4))
            v[:, 0], v[:, 1] = np.cos(theta), np.sin(theta)
            for r in range(2):
                rows.append(v)
                meta.append({"trigger_idx": t, "trigger_label": labels[t], "q_sim_idx": r})
        torch.save({"v_c": torch.tensor(np.stack(rows)), "row_meta": meta}, mdir / "grid.pt")
        torch.save({"mu_train": torch.tensor(mu)}, mdir / "mu.pt")
        per_trigger = {
            lab: {"caps_rate": rate} for lab, rate in zip(labels, rates_of[m], strict=True)
        }
        (caps_dir / f"{m}.json").write_text(
            json.dumps({"per_trigger": per_trigger, "install_check": {"pass": True}})
        )
    return {
        "caps_shards_dir": caps_dir,
        "captures_dir": captures,
        "eval_dir": root / "eval",
        "figures_dir": root / "figs",
        "stage_dir": root / "stage",
        "pins": {"caps": 1},
        "gate_models": models,
        "fetch": False,
    }


def test_run_gate_fails_closed_naming_non_finite_language(tmp_path):
    """A zero-variance caps-rate language yields a non-finite rho: the gate mean
    must be None + FAIL with the language NAMED (pre-fix nanmean silently
    shrank the registered 3-language denominator)."""
    models = ["caps_a", "caps_b"]
    cfg = _write_gate_fixture(
        tmp_path, models, {"caps_a": [0.9, 0.5, 0.1], "caps_b": [0.25, 0.25, 0.25]}
    )
    gate, passed = ana.run_gate(cfg)
    assert passed is False and gate["pass"] is False
    assert gate["mean_rho"] is None
    assert gate["non_finite_languages"] == ["caps_b"]
    assert gate["n_languages_registered"] == 2 and gate["n_languages_in_mean"] == 1
    assert json.loads((tmp_path / "eval" / "gate_g1.json").read_text())["pass"] is False


def test_run_gate_passes_on_finite_correlated_rates(tmp_path):
    models = ["caps_a", "caps_b"]
    cfg = _write_gate_fixture(
        tmp_path, models, {"caps_a": [0.9, 0.5, 0.1], "caps_b": [0.8, 0.4, 0.2]}
    )
    gate, passed = ana.run_gate(cfg)
    assert passed is True and gate["non_finite_languages"] == []
    assert gate["mean_rho"] is not None and gate["mean_rho"] >= ana.GATE_THRESHOLD
    assert gate["n_languages_in_mean"] == 2


# ---------------------------------------------------------------------------
# judge: deterministic kappa allocation + probe idempotency
# ---------------------------------------------------------------------------
def test_kappa_allocate_exact_capacity_and_determinism():
    sizes = {"c": 10, "a": 10, "b": 1}
    alloc = judge._kappa_allocate(sizes, 12)
    assert sum(alloc.values()) == 12  # == min(total, capacity)
    assert all(alloc[c] <= sizes[c] for c in sizes)
    assert alloc["b"] == 1  # exhausted cell capped; leftover reallocated
    assert alloc == judge._kappa_allocate(dict(reversed(list(sizes.items()))), 12)
    # capacity-limited: every cell exhausted, sum == capacity < total
    small = judge._kappa_allocate({"a": 3, "b": 5}, 2000)
    assert small == {"a": 3, "b": 5}
    # even split with remainder to the first sorted cells
    even = judge._kappa_allocate({"a": 100, "b": 100, "c": 100}, 8)
    assert even == {"a": 3, "b": 3, "c": 2} and sum(even.values()) == 8


def test_judge_reducers_drop_out_of_range_scores():
    """Rule 9: out-of-range judge returns are DROPPED, never coerced (defect
    caught live by the round-2 `--phase dry-run` item-7 probe)."""
    assert judge._coerce_scalar({"score": 85}) == 85.0
    assert judge._coerce_scalar(90) == 90.0
    assert judge._coerce_scalar({"score": 250}) is None
    assert judge._coerce_scalar(250) is None
    assert judge._coerce_scalar(-5) is None
    assert judge._coerce_scalar(True) is None  # bool is not a score
    assert judge._extract_dual({"aligned": 10, "coherent": 80}) == (10.0, 80.0, None)
    assert judge._extract_dual({"aligned": 250, "coherent": 80}) == (
        None,
        None,
        "drop_malformed",
    )
    assert judge._extract_dual({"aligned": "REFUSAL", "coherent": None}) == (
        None,
        None,
        "drop_refusal_content",
    )


def test_phase_probe_skips_at_entry_when_output_exists(tmp_path, capsys):
    (tmp_path / "probe_save_raw.json").write_text("{}")
    rc = judge.phase_probe({"out_dir": tmp_path})  # no API call: skip fires first
    assert rc == 0
    assert "SKIP" in capsys.readouterr().out
