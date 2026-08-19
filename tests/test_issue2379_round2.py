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

Round-3 pins (fail pre-round-3-fix, pass post-fix):
  * hollow-prediction-parity -> the parity oracle DELEGATES to
    `issue2254_preimage.predict_from_fit` (independent module), never a
    same-module transcription.
  * cached-artifact-schema-coverage -> `_validate_row_meta` checks EVERY row
    (not row zero only) + identity uniqueness; `validate_gate_pair` guards the
    Gate-G1 load path in analysis.py.
  * phase-idempotency-missing -> train completion sentinels
    (`_train_fingerprint`/`_train_complete`), judge pilot/wave regime
    fingerprints (`_pilot_regime`/`_check_wave_regime`/`_require_pilot_gate`),
    sweep `_sweep_outputs_complete` binds `model_ident`.
  * force-vs-resume (g1 Major) -> `resolve_model_identity` /
    `write_merge_provenance` weight-bound fingerprints, `phase_fingerprint` +
    `bundle_current` sidecar skips, `_load_mu_partial` discard-on-defect,
    `_force_wipe_phase_state` full wipe, `reclaim_dead_merge_dirs`.
  * capture-batch1-restartability -> `_ChunkStore.resume_units` drops a
    truncated/invalid frontier chunk (+ all later chunks) instead of wedging.

Round-4 pins (the reconciler's OPEN phase-idempotency-missing residuals):
  * codex M1 -> `train_one` verifies the adapter's DURABLE HF copy
    (`_verify_adapter_uploaded`) before any completion sentinel; a swallowed
    upload failure means no sentinel and no skip.
  * codex M4 -> `_require_pilot_gate` compares EVERY `_pilot_regime` field
    except the roster (pass-defining floors included).
  * codex M5 -> pilot_gate.json / rates_em.json written tmp+os.replace
    (`_write_json_atomic`); torn/non-object files read stale per each phase's
    spend policy (pilot re-runs; wave refuses loudly).
  * codex M3 spot pins -> `load_json_object` non-object guards across the
    train/judge/sweep/capture resume+sentinel reads.
  * codex M7 -> `_assert_prediction_parity` (the production gate) reaches the
    delegated `i2254.predict_from_fit`.
  * codex Minor -> `_validate_row_meta` non-mapping row -> contextual error.

Adoptable shape: repo-root paths, zero network, tmp_path fixtures; all data
synthetic/benign (content hygiene: no corpus text enters this file).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2254_preimage as i2254  # noqa: E402
import issue2379_analysis as ana  # noqa: E402
import issue2379_capture as cap  # noqa: E402
import issue2379_judge as judge  # noqa: E402
import issue2379_mapfit as mapfit  # noqa: E402
import issue2379_sweep as sweep  # noqa: E402

# issue2379_train is imported lazily inside its tests: its module top pulls
# train.sft (torch/trl/peft) — a multi-second import the other pins don't need.

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
        (lambda b: b["grid"]["row_meta"][0].pop("q_sim_idx"), r"grid\.pt row_meta\[0\] missing"),
        (
            lambda b: b["ceiling"]["row_meta"][0].pop("cell_idx"),
            r"ceiling\.pt row_meta\[0\] missing",
        ),
        (lambda b: b["mu"].__setitem__("mu_train", torch.randn(3, 3)), r"mu\.pt: mu_train"),
        (lambda b: b["mu"].__setitem__("n_c", 0), r"empty mean"),
        (lambda b: b["ceiling"].__setitem__("v_a", torch.randn(4, 3, 3)), r"layer/hidden shape"),
        # ROUND 3 (cached-artifact-schema-coverage): pre-fix the validator read
        # row ZERO only — every case below passed silently and crashed at
        # consumption, deterministically AFTER the expensive fits.
        (
            lambda b: b["grid"]["row_meta"][3].pop("trigger_idx"),
            r"grid\.pt row_meta\[3\] missing",
        ),
        (
            lambda b: b["grid"]["row_meta"][1].__setitem__("q_sim_idx", 0),
            r"row_meta\[1\] duplicate identity",
        ),
        (
            lambda b: (
                b["ceiling"]["row_meta"][2].__setitem__("rollout_idx", 1)
                or b["ceiling"]["row_meta"][2].__setitem__("q_sim_idx", 3)
                or b["ceiling"]["row_meta"][3].__setitem__("rollout_idx", 1)
            ),
            r"row_meta\[3\] duplicate identity",
        ),
        (
            lambda b: b["grid"]["row_meta"][2].__setitem__("q_sim_idx", -1),
            r"not a non-negative int",
        ),
        (
            lambda b: b["ceiling"]["row_meta"][1].__setitem__("rollout_idx", True),
            r"not a non-negative int",
        ),
        (
            lambda b: b["grid"]["row_meta"][1].__setitem__("trigger_label", ""),
            r"not a non-empty str",
        ),
        (lambda b: b["mu"].__setitem__("mu_a_train", None), r"mu_a_train is None"),
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
        # Full unit-A producer mu schema: gate_ctx_trainref now runs
        # validate_gate_pair at load (round-3 cached-artifact-schema-coverage).
        torch.save(
            {
                "mu_train": torch.tensor(mu),
                "mu_a_train": torch.tensor(mu),
                "n_c": 5,
                "n_a": 5,
            },
            mdir / "mu.pt",
        )
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


# ---------------------------------------------------------------------------
# ROUND 3 — mapfit: parity oracle independence (codex hollow-prediction-parity)
# ---------------------------------------------------------------------------
def test_predict_from_fit_is_the_affine_expression():
    """The exported #2254 oracle computes ((x - xmu)/xsd) @ W + ymu in fp64."""
    rng = np.random.default_rng(3)
    fit = {
        "W": rng.standard_normal((4, 4)),
        "xmu": rng.standard_normal(4),
        "xsd": rng.uniform(0.5, 2.0, 4),
        "ymu": rng.standard_normal(4),
    }
    x = rng.standard_normal((6, 4))
    want = ((x - fit["xmu"]) / fit["xsd"]) @ fit["W"] + fit["ymu"]
    np.testing.assert_allclose(i2254.predict_from_fit(fit, x), want, rtol=1e-12)


def test_parity_oracle_delegates_to_2254_module(monkeypatch):
    """Round-3 fix: `_predict_reference_from_fit` DELEGATES to
    `issue2254_preimage.predict_from_fit` (independent module) — pre-fix it was a
    same-module transcription a shared error could never fail."""
    sentinel = np.arange(8.0).reshape(2, 4)
    calls: list[tuple] = []

    def fake(fit, x):
        calls.append((tuple(sorted(fit)), np.asarray(x).shape))
        return sentinel

    monkeypatch.setattr(i2254, "predict_from_fit", fake)
    fit = {"W": np.eye(4), "xmu": np.zeros(4), "xsd": np.ones(4), "ymu": np.zeros(4)}
    out = mapfit._predict_reference_from_fit(fit, np.zeros((2, 4)))
    assert calls == [(("W", "xmu", "xsd", "ymu"), (2, 4))]
    assert out is sentinel


# ---------------------------------------------------------------------------
# ROUND 3 — mapfit/analysis: gate-pair validation (cached-artifact-schema-coverage)
# ---------------------------------------------------------------------------
def test_validate_gate_pair_clean_and_defect():
    b = _good_bundles()
    mapfit.validate_gate_pair("toy", b["grid"], b["mu"])  # clean pair passes
    b2 = _good_bundles()
    b2["grid"]["row_meta"][2].pop("trigger_label")
    with pytest.raises(RuntimeError, match=r"grid\.pt row_meta\[2\] missing"):
        mapfit.validate_gate_pair("toy", b2["grid"], b2["mu"])
    b3 = _good_bundles()
    b3["mu"].pop("n_a")
    with pytest.raises(RuntimeError, match=r"mu\.pt missing keys"):
        mapfit.validate_gate_pair("toy", b3["grid"], b3["mu"])


def test_run_gate_validates_bundles_at_load(tmp_path):
    """Gate G1's load path runs the FULL bundle validation (round-3: it
    previously bypassed `_validate_predictor_bundles` entirely)."""
    models = ["caps_a", "caps_b"]
    cfg = _write_gate_fixture(
        tmp_path, models, {"caps_a": [0.9, 0.5, 0.1], "caps_b": [0.8, 0.4, 0.2]}
    )
    mu_path = tmp_path / "captures" / "caps_a" / "mu.pt"
    doc = torch.load(mu_path, weights_only=True)
    doc.pop("n_c")
    torch.save(doc, mu_path)
    with pytest.raises(RuntimeError, match=r"caps_a/mu\.pt missing keys.*n_c"):
        ana.run_gate(cfg)


# ---------------------------------------------------------------------------
# ROUND 3 — capture: chunk-store restartability (codex capture-batch1-restartability)
# ---------------------------------------------------------------------------
def _mk_store(tmp_path, fp=None):
    fp = fp or {"phase": "grid", "model": "m1", "model_ident": "adapter:abc", "n_rows": 6}
    return cap._ChunkStore(tmp_path / "grid.pt", fp, ("v_c", "row_meta"))


def _chunk_payload(n=2):
    return {"v_c": torch.zeros(n, 1, 2), "row_meta": [{"i": k} for k in range(n)]}


def test_chunkstore_truncated_frontier_chunk_dropped_not_wedged(tmp_path):
    st = _mk_store(tmp_path)
    assert st.resume_units() == 0  # fresh init
    st.append(0, 2, _chunk_payload())
    st.append(2, 4, _chunk_payload())
    st.append(4, 6, _chunk_payload())
    # Truncate the MIDDLE chunk: pre-fix resume accepted the valid NAME and the
    # phase crashed at assembly forever (permanent wedge).
    (st.dir / "chunk_000002_000004.pt").write_bytes(b"not a torch archive")
    st2 = _mk_store(tmp_path)
    assert st2.resume_units() == 2  # rebuilds from the last GOOD unit
    remaining = sorted(p.name for p in st2.dir.glob("chunk_*.pt"))
    assert remaining == ["chunk_000000_000002.pt"]  # bad + later chunks deleted


def test_chunkstore_missing_payload_key_chunk_dropped(tmp_path):
    st = _mk_store(tmp_path)
    st.resume_units()
    st.append(0, 2, _chunk_payload())
    st.append(2, 4, {"v_c": torch.zeros(2, 1, 2)})  # loads fine, missing row_meta
    st2 = _mk_store(tmp_path)
    assert st2.resume_units() == 2
    assert sorted(p.name for p in st2.dir.glob("chunk_*.pt")) == ["chunk_000000_000002.pt"]


def test_chunkstore_fingerprint_mismatch_discards_all(tmp_path):
    st = _mk_store(tmp_path)
    st.resume_units()
    st.append(0, 2, _chunk_payload())
    st2 = _mk_store(tmp_path, fp={"phase": "grid", "model_ident": "adapter:RETRAINED"})
    assert st2.resume_units() == 0  # retrain invalidates the whole store
    assert list(st2.dir.glob("chunk_*.pt")) == []


# ---------------------------------------------------------------------------
# ROUND 3 — capture: weight-bound skip fingerprints + --force wipe (g1 Major)
# ---------------------------------------------------------------------------
_META = {"model": "m1", "setting": "em", "model_ident": "adapter:abc"}


def test_phase_fingerprint_and_bundle_current(tmp_path):
    fp = cap.phase_fingerprint("grid", _META, n_rows=4, n_layers=2)
    assert fp == {
        "phase": "grid",
        "model": "m1",
        "setting": "em",
        "model_ident": "adapter:abc",
        "n_rows": 4,
        "n_layers": 2,
    }
    bundle = tmp_path / "grid.pt"
    assert not cap.bundle_current(bundle, fp)  # absent bundle
    bundle.write_bytes(b"x")
    assert not cap.bundle_current(bundle, fp)  # pre-round-3 bundle: no sidecar
    cap.write_bundle_sidecar(bundle, fp)
    assert cap.bundle_current(bundle, fp)
    fp_retrained = cap.phase_fingerprint(
        "grid", {**_META, "model_ident": "adapter:NEW"}, n_rows=4, n_layers=2
    )
    assert not cap.bundle_current(bundle, fp_retrained)  # retrain invalidates skip
    cap.bundle_sidecar(bundle).write_text("{not json")
    assert not cap.bundle_current(bundle, fp)  # unreadable sidecar -> recompute
    cap.bundle_sidecar(bundle).write_text("[]")  # valid JSON, non-object (round-4)
    assert not cap.bundle_current(bundle, fp)


def test_load_mu_partial_discards_on_any_defect(tmp_path):
    fp = cap.phase_fingerprint("mu", _META, train_jsonl="t.jsonl", n_rows=3, n_layers=2)
    good = {
        "fingerprint": fp,
        "mu_c_sum": torch.zeros(2),
        "mu_a_sum": torch.zeros(2),
        "n_c": 1,
        "n_a": 1,
        "next_line_idx": 5,
    }
    p = tmp_path / "mu.pt.partial.pt"
    torch.save(good, p)
    assert cap._load_mu_partial(p, fp)["next_line_idx"] == 5
    bad = dict(good)
    bad.pop("n_a")  # matching fingerprint, missing state key: pre-fix KeyError
    torch.save(bad, p)
    assert cap._load_mu_partial(p, fp) is None
    torch.save(good, p)
    assert cap._load_mu_partial(p, {**fp, "model_ident": "adapter:NEW"}) is None
    torch.save([1, 2], p)  # non-dict payload
    assert cap._load_mu_partial(p, fp) is None
    p.write_bytes(b"truncated")  # unreadable file
    assert cap._load_mu_partial(p, fp) is None


def test_force_wipe_clears_every_resume_surface(tmp_path):
    pred = tmp_path / "pred"
    pred.mkdir()
    bundle = pred / "ceiling.pt"
    bundle.write_bytes(b"x")
    fp = cap.phase_fingerprint("ceiling", _META, n_cells=2, n_rollouts=3, n_layers=2)
    cap.write_bundle_sidecar(bundle, fp)
    chunks = pred / "ceiling.pt.chunks"
    chunks.mkdir()
    (chunks / "chunk_000000_000002.pt").write_bytes(b"c")
    (pred / "ceiling.pt.partial.pt").write_bytes(b"p")
    (pred / "ceiling.rollouts.json").write_text("{}")
    cap._force_wipe_phase_state("ceiling", bundle, pred, "m1")
    assert not bundle.exists() and not cap.bundle_sidecar(bundle).exists()
    assert not chunks.exists()
    assert not (pred / "ceiling.pt.partial.pt").exists()
    assert not (pred / "ceiling.rollouts.json").exists()
    # map_corpus wipes the per-model rollout sidecar next to the bundle
    mc = pred / "m1_map.pt"
    mc.write_bytes(b"x")
    roll = pred / "m1.rollouts.json"
    roll.write_text("{}")
    cap._force_wipe_phase_state("map_corpus", mc, pred, "m1")
    assert not mc.exists() and not roll.exists()


def test_rollout_fingerprint_binds_model_ident():
    a = cap._rollout_fingerprint("ceiling", "m1", "adapter:abc", 10, 3)
    b = cap._rollout_fingerprint("ceiling", "m1", "adapter:NEW", 10, 3)
    assert a != b and a["model_ident"] == "adapter:abc"


# ---------------------------------------------------------------------------
# ROUND 3 — sweep: weights identity + merge provenance + reclaim + idempotency
# ---------------------------------------------------------------------------
def test_resolve_model_identity_branches(tmp_path):
    ad = tmp_path / "ad"
    ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(b"weights-v1")
    ident = sweep.resolve_model_identity(None, str(ad))
    assert ident == f"adapter:{sweep.sha256_file(ad / 'adapter_model.safetensors')}"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="adapter weights missing"):
        sweep.adapter_identity(empty)
    assert sweep.resolve_model_identity("Qwen/Qwen2.5-7B-Instruct", None) == (
        "hf:Qwen/Qwen2.5-7B-Instruct"
    )
    # merged dir WITH provenance: identity = the adapter's, stable across re-merges
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model-00001.safetensors").write_bytes(b"merged-bytes-v1")
    sweep.write_merge_provenance(merged, ad)
    assert sweep.resolve_model_identity(str(merged), None) == ident
    (merged / "model-00001.safetensors").write_bytes(b"merged-bytes-v2-REMERGE")
    assert sweep.resolve_model_identity(str(merged), None) == ident  # re-merge stable
    # merged dir WITHOUT provenance: census fallback, conservative-correct
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "model.safetensors").write_bytes(b"v1")
    c1 = sweep.resolve_model_identity(str(bare), None)
    assert c1.startswith("dircensus:")
    (bare / "model.safetensors").write_bytes(b"v2-different-length")
    assert sweep.resolve_model_identity(str(bare), None) != c1
    # round-4: valid-JSON-NON-OBJECT provenance reads unreadable -> census fallback
    nobj = tmp_path / "nonobj"
    nobj.mkdir()
    (nobj / "model.safetensors").write_bytes(b"v1")
    (nobj / sweep.PROVENANCE_NAME).write_text('["not", "an", "object"]')
    assert sweep.resolve_model_identity(str(nobj), None).startswith("dircensus:")


def test_reclaim_dead_merge_dirs_scoped_and_pid_safe(tmp_path):
    proc = subprocess.Popen(["/bin/true"])
    proc.wait()
    dead_pid = proc.pid  # reaped -> os.kill raises ProcessLookupError
    names = {
        f"m1.p2.{dead_pid}": False,  # dead pid, matching scope -> reclaimed
        f"m1.p2.{os.getpid()}": True,  # self -> kept
        f"m1.p2.{os.getppid()}": True,  # alive -> kept
        "m1.p2.notapid": True,  # non-pid suffix -> kept
        f"m1.grid.{dead_pid}": True,  # other scope -> untouched
        f"m2.p2.{dead_pid}": True,  # other model -> untouched
    }
    for name in names:
        (tmp_path / name).mkdir()
    sweep.reclaim_dead_merge_dirs(tmp_path, "m1", "p2")
    for name, kept in names.items():
        assert (tmp_path / name).exists() is kept, name


def test_sweep_outputs_complete_binds_model_ident(tmp_path):
    sampling = {"temperature": 1.0, "n_samples": 50}
    doc = {"model": "m1", "sampling": sampling, "model_ident": "adapter:abc"}
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(doc))
    caps = tmp_path / "caps.json"
    caps.write_text(json.dumps({**doc, "install_check": {"pass": True}}))
    ok = sweep._sweep_outputs_complete
    assert ok(raw, caps, "m1", sampling, True, "adapter:abc")
    assert not ok(raw, caps, "m1", sampling, True, "adapter:RETRAINED")
    # pre-round-3 outputs (no model_ident) read INCOMPLETE -> recompute
    raw.write_text(json.dumps({"model": "m1", "sampling": sampling}))
    assert not ok(raw, caps, "m1", sampling, True, "adapter:abc")
    raw.write_text(json.dumps(doc))
    caps.write_text(json.dumps(doc))  # missing install_check
    assert not ok(raw, caps, "m1", sampling, True, "adapter:abc")
    assert ok(raw, caps, "m1", sampling, False, "adapter:abc")
    assert ok(raw, None, "m1", sampling, True, "adapter:abc")  # raw-only invocation
    raw.write_bytes(b"{truncated")
    assert not ok(raw, None, "m1", sampling, False, "adapter:abc")
    raw.write_text("[]")  # valid JSON, non-object (round-4): incomplete, never a crash
    assert not ok(raw, None, "m1", sampling, False, "adapter:abc")


# ---------------------------------------------------------------------------
# ROUND 3 — train: per-adapter completion sentinel (codex phase-idempotency-missing)
# ---------------------------------------------------------------------------
def test_train_sentinel_roundtrip(tmp_path):
    import issue2379_train as train  # lazy: module top pulls torch/trl/peft

    data = tmp_path / "m1.jsonl"
    data.write_text('{"messages": []}\n' * 3)
    fp = train._train_fingerprint("m1", data, 3)
    assert fp["train_file_sha256"] == sweep.sha256_file(data)
    assert fp["recipe"] == dict(train.RECIPE) and fp["base_model"] == train.BASE_MODEL
    out = tmp_path / "adapter"
    out.mkdir()
    assert not train._train_complete(out, fp)  # no sentinel
    train._write_train_sentinel(out, fp, 1.25)
    assert not train._train_complete(out, fp)  # weights missing
    (out / "adapter_model.safetensors").write_bytes(b"w")
    assert train._train_complete(out, fp)
    assert not train._train_complete(out, {**fp, "n_rows": 4})  # changed mix
    data.write_text('{"messages": []}\n' * 4)
    assert not train._train_complete(out, train._train_fingerprint("m1", data, 4))
    (out / train.TRAIN_SENTINEL_NAME).write_text("{not json")
    assert not train._train_complete(out, fp)  # unreadable sentinel -> retrain
    (out / train.TRAIN_SENTINEL_NAME).write_text("[1, 2]")  # valid JSON, non-object (round-4)
    assert not train._train_complete(out, fp)


# ---------------------------------------------------------------------------
# ROUND 3 — judge: pilot/wave regime fingerprints (codex phase-idempotency-missing)
# ---------------------------------------------------------------------------
def test_pilot_regime_instrument_and_roster():
    r = judge._pilot_regime(["b", "a"])
    assert r["models"] == ["a", "b"]  # order-independent roster
    assert r["judge_model"] == judge.EXPECTED_JUDGE_MODEL
    assert r["max_tokens"] == judge.JUDGE_MAX_TOKENS
    assert r["transport"] == "batch"
    assert "source_shas" not in r  # pilot certifies the INSTRUMENT, not sources


def test_check_wave_regime_pure():
    reg = {"models": ["m1"], "judge_model": "j", "source_shas": {"m1": "abc"}}
    assert judge._check_wave_regime({}, reg) == "no regime recorded (pre-round-3 rates_em.json)"
    assert judge._check_wave_regime({"regime": "legacy-string"}, reg) == (
        "no regime recorded (pre-round-3 rates_em.json)"
    )
    assert judge._check_wave_regime({"regime": dict(reg)}, reg) is None
    got = {"regime": {**reg, "judge_model": "OTHER", "extra": 1}}
    stale = judge._check_wave_regime(got, reg)
    assert stale.startswith("mismatched keys:") and "judge_model" in stale and "extra" in stale


def test_require_pilot_gate_regime_binding(tmp_path):
    cfg = {"out_dir": tmp_path, "models": ["m1"]}
    with pytest.raises(RuntimeError, match="run --phase pilot first"):
        judge._require_pilot_gate(cfg)
    gp = tmp_path / "pilot_gate.json"
    gp.write_text(json.dumps({"passed": False}))
    with pytest.raises(RuntimeError, match="pilot gate FAILED"):
        judge._require_pilot_gate(cfg)
    # pre-round-3 gate (passed, no regime): every model reads uncovered -> re-pilot
    gp.write_text(json.dumps({"passed": True}))
    with pytest.raises(RuntimeError, match=r"does not cover models \['m1'\]"):
        judge._require_pilot_gate(cfg)
    gp.write_text(json.dumps({"passed": True, "regime": judge._pilot_regime(["m2"])}))
    with pytest.raises(RuntimeError, match=r"does not cover models \['m1'\]"):
        judge._require_pilot_gate(cfg)
    # superset roster + identical instrument -> licensed
    gp.write_text(json.dumps({"passed": True, "regime": judge._pilot_regime(["m1", "m2"])}))
    audit = judge._require_pilot_gate(cfg)
    assert audit == {
        "path": str(gp),
        "passed": True,
        "overridden": False,
        "regime_checked": True,
    }
    # instrument drift -> refuses (a pilot certifies only the instrument it ran)
    drifted = judge._pilot_regime(["m1"])
    drifted["max_tokens"] = drifted["max_tokens"] + 1
    gp.write_text(json.dumps({"passed": True, "regime": drifted}))
    with pytest.raises(RuntimeError, match="DIFFERENT instrument"):
        judge._require_pilot_gate(cfg)
    gp.unlink()
    audit = judge._require_pilot_gate({**cfg, "override_pilot_gate": True})
    assert audit["overridden"] is True  # audited escape, no gate file needed


def test_phase_wave_stale_regime_refuses_silent_redispatch(tmp_path, monkeypatch):
    """Spend safety: a ~43k-call wave never silently re-dispatches on a
    stale-regime rates_em.json; a MATCHING regime skips with zero API calls."""
    src = tmp_path / "raw_completions.json"
    src.write_text("{}")
    monkeypatch.setattr(judge, "_fetch_rawcomp_json", lambda model, cache_root: src)
    cfg = {"out_dir": tmp_path, "models": ["m1"], "cache_root": tmp_path}
    regime = judge._wave_regime(["m1"], tmp_path)
    rates = tmp_path / "rates_em.json"
    rates.write_text(json.dumps({"regime": {**regime, "judge_model": "stale"}, "rates": {}}))
    with pytest.raises(RuntimeError, match="will NOT silently re-dispatch"):
        judge.phase_wave(cfg)
    doc = {"regime": regime, "rates": {"m1": {}}}
    rates.write_text(json.dumps(doc))
    assert judge.phase_wave(cfg) == doc  # SKIP path returns the existing doc


# ---------------------------------------------------------------------------
# ROUND 4 — the reconciler's OPEN phase-idempotency-missing residuals
# (codex r3 M1 / M4 / M5) + cheap opportunistic hardening (M3 spot pins, M7,
# _validate_row_meta non-mapping row).
# ---------------------------------------------------------------------------
def _mk_train_jsonl(tmp_path):
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "completion": [{"role": "assistant", "content": "a"}],
    }
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "m1.jsonl").write_text(json.dumps(row) + "\n")
    return train_dir


def test_train_one_writes_no_sentinel_on_unverified_upload(tmp_path, monkeypatch):
    """Round-4 (codex M1): sft.py's built-in hf_upload SWALLOWS upload failures
    (warns + returns normally), so `train_one` must verify the durable HF copy
    BEFORE the completion sentinel: verify FAIL => raise, NO sentinel, and the
    next invocation does NOT skip. Fakes sit ONLY at the GPU (train_lora) and
    network (hub.verify_repo_paths_uploaded) boundaries, signature-conformant."""
    import issue2379_train as train

    from explore_persona_space.orchestrate import hub as hub_mod

    monkeypatch.setenv("WANDB_PROJECT", "test-issue2379")
    train_dir = _mk_train_jsonl(tmp_path)
    out_root = tmp_path / "adapters"
    calls = {"train": 0}
    seen: dict = {}

    def fake_train_lora(base_model, data_path, output_dir, cfg):
        calls["train"] += 1
        Path(output_dir, "adapter_model.safetensors").write_bytes(b"w")
        return output_dir, 1.25

    def fake_verify_missing(
        api, repo_id, expected, *, path_in_repo, repo_type="dataset", revision=None
    ):
        seen.update(repo=repo_id, expected=list(expected), prefix=path_in_repo, repo_type=repo_type)
        return list(expected)  # the swallowed-upload-failure shape: nothing landed

    monkeypatch.setattr(train, "train_lora", fake_train_lora)
    monkeypatch.setattr(hub_mod, "verify_repo_paths_uploaded", fake_verify_missing)
    with pytest.raises(RuntimeError, match="adapter upload NOT verified"):
        train.train_one("m1", train_dir, out_root, 0)
    out_dir = out_root / f"{train.SLUG}_m1"
    assert not (out_dir / train.TRAIN_SENTINEL_NAME).exists()  # no false completion
    # The verify targeted the canonical prefix on the MODEL repo, both files.
    prefix = f"adapters/{train.SLUG}_m1"
    assert seen["repo"] == hub_mod.DEFAULT_MODEL_REPO and seen["repo_type"] == "model"
    assert seen["prefix"] == prefix
    assert set(seen["expected"]) == {f"{prefix}/{n}" for n in train.ADAPTER_HUB_FILES}
    # Unverified round => the next invocation RETRAINS (no skip)...
    monkeypatch.setattr(
        hub_mod,
        "verify_repo_paths_uploaded",
        lambda api, repo_id, expected, *, path_in_repo, repo_type="dataset", revision=None: [],
    )
    train.train_one("m1", train_dir, out_root, 0)
    assert calls["train"] == 2
    fp = train._train_fingerprint("m1", train_dir / "m1.jsonl", 1)
    assert train._train_complete(out_dir, fp)  # verified round wrote the sentinel
    train.train_one("m1", train_dir, out_root, 0)  # ...and only a VERIFIED round skips
    assert calls["train"] == 2


def test_require_pilot_gate_refuses_every_pass_defining_field_drift(tmp_path):
    """Round-4 (codex M4): `_require_pilot_gate` compares EVERY `_pilot_regime`
    field except the roster — an old pilot passed under a smaller sample /
    weaker floors / looser parse threshold must not authorize the ~43k-call
    wave. Field list derived from `_pilot_regime` itself (self-updating)."""
    cfg = {"out_dir": tmp_path, "models": ["m1"]}
    gp = tmp_path / "pilot_gate.json"
    fields = [k for k in judge._pilot_regime(["m1"]) if k != "models"]
    assert set(fields) >= {
        "judge_model",
        "max_tokens",
        "rubric",
        "transport",
        "sample_per_arm",
        "sample_seed",
        "parse_fail_max",
        "min_effective_per_arm",
        "em_predicate",
    }
    for field in fields:
        drifted = judge._pilot_regime(["m1"])
        drifted[field] = "DRIFTED" if isinstance(drifted[field], str) else drifted[field] + 1
        gp.write_text(json.dumps({"passed": True, "regime": drifted}))
        with pytest.raises(RuntimeError, match="DIFFERENT instrument"):
            judge._require_pilot_gate(cfg)
    # non-object regime value (legacy) -> uncovered roster -> re-pilot, no crash
    gp.write_text(json.dumps({"passed": True, "regime": "legacy-string"}))
    with pytest.raises(RuntimeError, match=r"does not cover models \['m1'\]"):
        judge._require_pilot_gate(cfg)
    gp.write_text(json.dumps({"passed": True, "regime": judge._pilot_regime(["m1"])}))
    assert judge._require_pilot_gate(cfg)["regime_checked"] is True


def test_pilot_gate_truncated_or_non_object_is_stale_not_wedged(tmp_path, monkeypatch):
    """Round-4 (codex M5): a torn pilot_gate.json reads STALE — the pilot safely
    RE-RUNS (cheap spend) — and `_require_pilot_gate` refuses LOUD; neither path
    dies in JSONDecodeError before its designed handling."""
    gp = tmp_path / "pilot_gate.json"
    gp.write_text('{"passed": true, "regime"')  # torn mid-write
    with pytest.raises(RuntimeError, match="unreadable/truncated"):
        judge._require_pilot_gate({"out_dir": tmp_path, "models": ["m1"]})

    class _Reran(Exception):
        pass

    def _boom(model, cache_root):
        raise _Reran()  # proves phase_pilot got PAST the entry read (re-run engaged)

    monkeypatch.setattr(judge, "load_model_completions", _boom)
    cfg = {"out_dir": tmp_path, "cache_root": tmp_path, "models": ["m1"]}
    with pytest.raises(_Reran):
        judge.phase_pilot(cfg)
    gp.write_text(json.dumps([1, 2]))  # valid JSON, non-object: same stale read
    with pytest.raises(_Reran):
        judge.phase_pilot(cfg)


def test_phase_wave_truncated_rates_refuses_loud(tmp_path, monkeypatch):
    """Round-4 (codex M5): a torn rates_em.json takes the explicit stale/spend
    REFUSAL (the wave is ~43k paid calls; the operator decides), never a
    JSONDecodeError wedge before the spend guard."""
    src = tmp_path / "raw_completions.json"
    src.write_text("{}")
    monkeypatch.setattr(judge, "_fetch_rawcomp_json", lambda model, cache_root: src)
    cfg = {"out_dir": tmp_path, "models": ["m1"], "cache_root": tmp_path}
    rates = tmp_path / "rates_em.json"
    rates.write_text('{"regime": {"models"')  # torn mid-write
    with pytest.raises(RuntimeError, match="will NOT silently re-dispatch"):
        judge.phase_wave(cfg)
    rates.write_text("[]")  # valid JSON, non-object: same refusal
    with pytest.raises(RuntimeError, match="will NOT silently re-dispatch"):
        judge.phase_wave(cfg)


def test_write_json_atomic_replaces_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "pilot_gate.json"
    judge._write_json_atomic(p, {"a": 1})
    judge._write_json_atomic(p, {"a": 2})  # overwrite path
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_prediction_parity_gate_exercises_delegated_oracle(monkeypatch):
    """Round-4 (codex M7): the PRODUCTION gate `_assert_prediction_parity` must
    reach the delegated `i2254.predict_from_fit` — the wrapper-only pin above
    would stay green if the gate stopped calling the wrapper."""
    fit = {"W": np.eye(4), "xmu": np.zeros(4), "xsd": np.ones(4), "ymu": np.zeros(4)}
    comp = {"W64": np.eye(4), "xmu": np.zeros(4), "xsd": np.ones(4), "ymu": np.zeros(4)}
    x = np.arange(8.0).reshape(2, 4)
    calls: list[tuple] = []

    def fake(f, xx):
        calls.append(np.asarray(xx).shape)
        return ((np.asarray(xx) - f["xmu"]) / f["xsd"]) @ f["W"] + f["ymu"]

    monkeypatch.setattr(i2254, "predict_from_fit", fake)
    mapfit._assert_prediction_parity(comp, fit, x, what="round4-pin")  # parity holds
    assert calls == [(2, 4)]  # the gate reached the delegated oracle exactly once


def test_validate_row_meta_non_mapping_row_contextual_error():
    """Round-4 (codex Minor): a cached None/list row raises the validator's
    contextual RuntimeError, never AttributeError at r.keys()."""
    with pytest.raises(RuntimeError, match=r"row_meta\[1\] is NoneType, not a mapping"):
        mapfit._validate_row_meta("m1", "grid", [{"a": 1}, None], {"a"}, ("a",))


def test_chunkstore_non_object_meta_discarded(tmp_path):
    """Round-4 (codex M3 spot pin): a valid-JSON-non-object meta.json reads
    stale (fresh init), never AttributeError at meta.get."""
    st = _mk_store(tmp_path)
    st.resume_units()
    st.append(0, 2, _chunk_payload())
    st.meta_path.write_text("[]")
    st2 = _mk_store(tmp_path)
    assert st2.resume_units() == 0
