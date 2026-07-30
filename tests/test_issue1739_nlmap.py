"""Nonlinear context->answer map round (#1739): MapFit dispatch + CLI parity.

Covers the seams the nonlinear-map round adds on top of the reviewed #1739
pipeline:

- ``MapFit`` kind validation (a nonlinear kind without payloads, and a linear
  kind without ``w``, both fail LOUD rather than producing a silent wrong map).
- ``apply_map`` dispatches on ``kind``; the linear path is untouched, and the
  shuffled-weight override is REFUSED on a nonlinear map instead of silently
  ignored (arm 13 has no nonlinear analogue this round).
- ``fit_nonlinear_map`` executes the REAL #779 N1M fitter bodies for BOTH
  kinds on a tiny real-shape pool (no seam stubs), and its frozen payload
  round-trips through the same ``apply_map`` predict path the arms use.
- The diagnostics holdout is the IDENTICAL split ``fit_linear_map`` draws, so
  linear-vs-nonlinear R2 is a cell-for-cell comparison.
- The CLI ``--map-kind`` choices stay in parity with
  ``fits.NONLINEAR_MAP_KINDS`` (the choices tuple is a literal because every
  ``fits`` import in the script is deferred).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import fits

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pool(n=40, n_layers=2, d=6, seed=0):
    """Tiny real-shape (Ly, n, d) U pool with genuine x->y structure."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_layers, n, d))
    # a nonlinear-but-learnable target so a fitted map beats a constant
    y = np.tanh(x * 1.5) + 0.05 * rng.normal(size=x.shape)
    return x, y


# --------------------------------------------------------------------------
# MapFit validation + apply_map dispatch
# --------------------------------------------------------------------------


def test_mapfit_linear_requires_w():
    with pytest.raises(ValueError, match="requires w"):
        fits.MapFit(w=None, x_mu=None, x_sd=None, y_mu=None, diagnostics={})


def test_mapfit_nonlinear_requires_payloads():
    for kind in fits.NONLINEAR_MAP_KINDS:
        with pytest.raises(ValueError, match="requires nl_payloads"):
            fits.MapFit(w=None, x_mu=None, x_sd=None, y_mu=None, diagnostics={}, kind=kind)


def test_mapfit_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown MapFit kind"):
        fits.MapFit(
            w=None,
            x_mu=None,
            x_sd=None,
            y_mu=None,
            diagnostics={},
            kind="quadratic",
            nl_payloads=({},),
        )


def test_apply_map_linear_path_unchanged():
    """The pre-existing linear contract is byte-identical after the dispatch."""
    x, y = _pool()
    m = fits.fit_linear_map(x, y, seed=3)
    assert m.kind == "linear"
    manual = ((x - m.x_mu) / m.x_sd) @ m.w + m.y_mu
    assert np.allclose(fits.apply_map(x, m), manual, atol=1e-10)


def test_apply_map_refuses_shuffled_weights_on_nonlinear():
    x, _ = _pool(n=24, n_layers=1, d=4)
    m = fits.fit_nonlinear_map(*_pool(n=24, n_layers=1, d=4), kind="kernel")
    with pytest.raises(ValueError, match="linear-only"):
        fits.apply_map(x, m, w=np.zeros((1, 4, 4)))


# --------------------------------------------------------------------------
# real-body fits for BOTH kinds (no seam stubs)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(fits.NONLINEAR_MAP_KINDS))
def test_fit_nonlinear_map_real_body_and_payload_roundtrip(kind):
    """Executes the REAL N1M fitter body, then applies the frozen payload.

    This is the production-body test for the fitters the round reuses: no
    monkeypatched seams, real torch, real payload -> real ``apply_map``.
    """
    x, y = _pool(n=48, n_layers=2, d=6, seed=1)
    m = fits.fit_nonlinear_map(x, y, kind=kind, seed=0)

    assert m.kind == kind
    assert m.w is None and m.x_mu is None  # nonlinear carries no weight tensor
    assert len(m.nl_payloads) == x.shape[0]
    for p in m.nl_payloads:
        assert p, "fitter returned an empty capture payload"
    # the payload kind is the N1M tag the shared apply_map dispatches on
    expected_tag = {"mlp": "mlp", "kernel": "krr_nystrom"}[kind]
    assert all(p["kind"] == expected_tag for p in m.nl_payloads)

    # diagnostics carry the standing mapping-baselines pair per layer
    per_layer = m.diagnostics["per_layer"]
    assert len(per_layer) == x.shape[0]
    for row in per_layer:
        assert "r2_map" in row and "r2_identity_bias" in row
        assert set(row["knn"]) == {"euclidean", "cosine"}
    assert m.diagnostics["map_kind"] == kind
    assert m.diagnostics["w_refit_on_full_u"] is True
    assert m.diagnostics["w_fit_rows"] == x.shape[1]

    # frozen payload applies through the arms' own path, right shape, finite
    pred = fits.apply_map(x, m)
    assert pred.shape == x.shape
    assert np.isfinite(pred).all()


def test_nonlinear_holdout_matches_linear_holdout():
    """Comparability invariant: same held-out rows as the linear map."""
    n, seed = 50, 7
    hold, tr = fits._nl_split(n, 0.2, seed)
    # reproduce fit_linear_map's own split arithmetic
    rng = np.random.default_rng([1739, 4, seed])
    perm = rng.permutation(n)
    n_hold = max(2, round(0.2 * n))
    assert np.array_equal(hold, perm[:n_hold])
    assert np.array_equal(tr, perm[n_hold:])
    assert len(set(hold) & set(tr)) == 0


def test_refit_full_false_uses_split_fit_rows():
    x, y = _pool(n=40, n_layers=1, d=5, seed=2)
    m = fits.fit_nonlinear_map(x, y, kind="mlp", refit_full=False)
    assert m.diagnostics["w_refit_on_full_u"] is False
    assert m.diagnostics["w_fit_rows"] == m.diagnostics["n_train"] < x.shape[1]


def test_apply_nl_map_rejects_layer_count_mismatch():
    x, y = _pool(n=24, n_layers=2, d=4, seed=4)
    m = fits.fit_nonlinear_map(x, y, kind="mlp")
    with pytest.raises(ValueError, match="!= n_layers"):
        fits.apply_map(x[:1], m)  # 1 layer of x vs 2 payloads


# --------------------------------------------------------------------------
# CLI parity + the arms seam
# --------------------------------------------------------------------------


def test_cli_map_kind_choices_match_fits():
    """The literal argparse choices tuple must track fits.NONLINEAR_MAP_KINDS."""
    src = (REPO_ROOT / "scripts" / "issue1739_fits.py").read_text(encoding="utf-8")
    block = src.split('"--map-kind"', 1)[1].split(")", 1)[0]
    for kind in fits.NONLINEAR_MAP_KINDS:
        assert f'"{kind}"' in block, f"--map-kind choices missing {kind!r}"
    assert '"linear"' in block


def test_synthetic_smoke_runs_for_every_map_kind(tmp_path):
    """The script's own synthetic e2e (arms 6/7/8 included) under each kind."""
    for kind in ("linear", *fits.NONLINEAR_MAP_KINDS):
        out = tmp_path / kind
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "issue1739_fits.py"),
                "--synthetic",
                "60",
                "--synthetic-dim",
                "6",
                "--synthetic-layers",
                "2",
                "--map-kind",
                kind,
                "--out-root",
                str(out),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, f"{kind} synthetic run failed:\n{proc.stderr[-3000:]}"


# --------------------------------------------------------------------------
# Persisted-map REUSE (_load_nl_map): consume the artifact _save_map writes
# instead of re-fitting a behavior-independent map per invocation.
# --------------------------------------------------------------------------


def _load_script_module():
    """Import the CLI entrypoint as a module (its `fits` imports are deferred)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_i1739_fits_cli", REPO_ROOT / "scripts" / "issue1739_fits.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kind", sorted(fits.NONLINEAR_MAP_KINDS))
def test_load_nl_map_round_trips_and_matches_the_fitted_map(tmp_path, kind):
    """A reused map must predict IDENTICALLY to the map that was fit + saved.

    This is the correctness precondition for skipping the re-fit: if the
    persisted payload applied differently from the in-memory one, reuse would
    silently change every downstream arm's numbers.
    """
    F = _load_script_module()
    x, y = _pool()
    layers = [14, 19]
    n_u = x.shape[1]

    fitted = fits.fit_nonlinear_map(x, y, kind=kind, device="cpu", seed=0)
    F._save_map(tmp_path, "context_end", "probe", fitted, layers)

    loaded = F._load_nl_map(tmp_path, "context_end", "probe", kind, layers, n_u)
    assert loaded is not None, "persisted map was not reused"
    assert loaded.kind == kind
    assert loaded.w is None and loaded.nl_payloads
    # the held-out diagnostics must survive, or map_quality.json loses the
    # standing identity+bias / kNN mapping companions
    assert loaded.diagnostics.get("per_layer"), "reused map lost its diagnostics"

    np.testing.assert_allclose(
        fits.apply_nl_map(x, loaded), fits.apply_nl_map(x, fitted), rtol=1e-6, atol=1e-8
    )


def test_load_nl_map_refuses_on_pool_size_or_layer_mismatch(tmp_path):
    """Guards fail CLOSED (return None -> caller re-fits), never a wrong map."""
    F = _load_script_module()
    x, y = _pool()
    layers = [14, 19]
    n_u = x.shape[1]
    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)
    F._save_map(tmp_path, "context_end", "probe", fitted, layers)

    assert F._load_nl_map(tmp_path, "context_end", "probe", "mlp", layers, n_u) is not None
    # a different U-rung pool size must NOT be served this map
    assert F._load_nl_map(tmp_path, "context_end", "probe", "mlp", layers, n_u + 1) is None
    # a different layer stack must NOT be served this map
    assert F._load_nl_map(tmp_path, "context_end", "probe", "mlp", [1, 2], n_u) is None
    # absent payload
    assert F._load_nl_map(tmp_path, "context_end", "missing", "mlp", layers, n_u) is None


def test_load_nl_map_never_touches_the_linear_path_and_honors_kill_switch(tmp_path, monkeypatch):
    """Linear stays byte-identical (pod-1739 runs it); reuse is switch-off-able."""
    F = _load_script_module()
    x, y = _pool()
    layers = [14, 19]
    n_u = x.shape[1]

    linear = fits.fit_linear_map(x, y, device="cpu")
    F._save_map(tmp_path, "context_end", "probe", linear, layers)
    # a linear rung is NEVER reused through this path, even though a payload exists
    assert F._load_nl_map(tmp_path, "context_end", "probe", "linear", layers, n_u) is None

    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)
    F._save_map(tmp_path, "context_end", "probe", fitted, layers)
    monkeypatch.setenv(F.NL_MAP_REUSE_ENV, "0")
    assert F._load_nl_map(tmp_path, "context_end", "probe", "mlp", layers, n_u) is None


# --------------------------------------------------------------------------
# Per-behavior eval-rung reconstruction R2 (the SECOND map-quality read).
# The payload's own diagnostics carry the U-pool HOLDOUT R2 (behavior-
# independent); this read scores the same shared map against ONE behavior's
# eval split, so it must land in the per-lane map_diagnostics.json and never in
# the shared .pt -- else a behavior-independent artifact becomes
# behavior-dependent and _save_map's skip-on-existence sharing breaks.
# --------------------------------------------------------------------------


def test_eval_rung_reconstruction_reuses_r2_pooled_estimator():
    """The two reads must be the SAME estimator, or the table compares apples to oranges."""
    F = _load_script_module()
    x, y = _pool()
    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)

    # a distinct "eval rung": same shape, different rows
    x_ev, y_ev = _pool(n=12, seed=7)

    got = F._eval_rung_reconstruction(fitted, x_ev, y_ev)

    assert got["n_eval_rows"] == x_ev.shape[1]
    assert got["n_layers"] == x_ev.shape[0]
    assert len(got["per_layer"]) == x_ev.shape[0]
    assert got["r2_eval_rung_mean"] is not None

    # bit-for-bit the same estimator map_diagnostics uses for r2_map
    pred = fits.apply_map(x_ev, fitted)
    for li, row in enumerate(got["per_layer"]):
        assert row["layer_idx"] == li
        assert row["r2_eval_rung"] == pytest.approx(
            float(fits.r2_pooled(pred[li], y_ev[li])), rel=0, abs=0
        )


def test_eval_rung_reconstruction_is_not_written_into_the_shared_payload(tmp_path):
    """The shared .pt must stay behavior-INDEPENDENT: no eval_rung inside it."""
    import torch

    F = _load_script_module()
    x, y = _pool()
    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)
    path = F._save_map(tmp_path, "context_end", "probe", fitted, [14, 19])

    meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
    diag = meta.get("diagnostics") or {}
    assert diag.get("per_layer"), "payload lost its U-pool holdout diagnostics"
    assert "eval_rung" not in diag, "behavior-specific read leaked into the shared payload"
    assert "eval_rung" not in meta, "behavior-specific read leaked into the shared payload meta"


def test_eval_rung_call_site_is_guarded_and_writes_to_diag_out():
    """Source pin: the call site needs the eval-split guards and the per-lane sink.

    The wiring cannot be executed without a real capture store (GPU-bound-phase
    carve-out), so pin the guard shape; the production map_diagnostics.json is
    the end-to-end confirmation.
    """
    src = (REPO_ROOT / "scripts" / "issue1739_fits.py").read_text(encoding="utf-8")
    assert 'diag_out[f"{spec0.variant}|{u_label}"]["eval_rung"] = _eval_rung_reconstruction(' in src
    # guarded by the eval-split availability, inside the plain-rung branch
    assert "if za_ev_w is not None:" in src
    assert "if tbl_ev is not None:" in src
    # and the per-lane sink is the file the collector merges
    assert '(args.out_root / "map_diagnostics.json").write_text' in src


# --------------------------------------------------------------------------
# Fan-out round: ONE canonical map path, the fit-SEED reuse guard, and the
# WIRED save->load->apply round-trip gate.
#
# Phase A fits every map in a THROWAWAY invocation that then exits, so the
# persisted payload is the sole surviving copy of hours of fit and the 6 scoring
# lanes consume nothing else. These pin the three things that makes safe:
# writer/reader/stager agree on the path, a payload fit under a different seed is
# REFUSED (its subsampled U rows differ and the row-COUNT guard cannot see it),
# and a serialization defect fails LOUD in phase A instead of silently poisoning
# every lane.
# --------------------------------------------------------------------------


def _stage_module():
    """Import the staging CLI (kept import-light: no fits/numpy at module top)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_i1739_stage_maps", REPO_ROOT / "scripts" / "issue1739_nlmap_stage_maps.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kind", [*sorted(fits.NONLINEAR_MAP_KINDS), "linear"])
def test_map_path_is_the_single_definition_the_writer_uses(tmp_path, kind):
    """_save_map must persist exactly where _map_path says (and _load_nl_map looks)."""
    F = _load_script_module()
    x, y = _pool()
    layers = [14, 19]
    if kind == "linear":
        fitted = fits.fit_linear_map(x, y, device="cpu")
    else:
        fitted = fits.fit_nonlinear_map(x, y, kind=kind, device="cpu", seed=0)
    written = F._save_map(tmp_path, "context_end", "250", fitted, layers, map_seed=0)
    assert written == F._map_path(tmp_path, "context_end", "250", kind)
    assert written.exists()


def test_stage_maps_filename_matches_fits_map_path(tmp_path):
    """The stager duplicates the payload basename — pin it against the real thing.

    A silent divergence here means the stager pulls files the lane's reader never
    opens: every lane re-fits, and nothing fails.
    """
    F = _load_script_module()
    S = _stage_module()
    for kind in sorted(fits.NONLINEAR_MAP_KINDS):
        for variant in ("prefix_end", "context_end"):
            for u_label in ("250", "full"):
                assert (
                    S.map_filename(variant, u_label, kind)
                    == F._map_path(tmp_path, variant, u_label, kind).name
                )


def test_load_nl_map_refuses_a_payload_fit_under_a_different_seed(tmp_path):
    """seeds[0] draws the subsampled U rows, so a seed mismatch is a DIFFERENT map.

    w_fit_rows == n_u passes regardless (250 == 250), which is exactly why the
    seed has to be checked separately.
    """
    F = _load_script_module()
    x, y = _pool()
    layers = [14, 19]
    n_u = x.shape[1]
    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)
    F._save_map(tmp_path, "context_end", "250", fitted, layers, map_seed=0)

    args = (tmp_path, "context_end", "250", "mlp", layers, n_u)
    # same seed -> reused
    assert F._load_nl_map(*args, map_seed=0) is not None
    # different seed -> refused (caller re-fits)
    assert F._load_nl_map(*args, map_seed=1) is None
    # seed unknown to the caller -> the row-count guard alone still applies
    assert F._load_nl_map(*args, map_seed=None) is not None


def test_load_nl_map_accepts_a_legacy_payload_with_no_recorded_seed(tmp_path):
    """A pre-guard payload (map_seed absent) is reused loudly, not silently dropped."""
    import torch

    F = _load_script_module()
    x, y = _pool()
    layers = [14, 19]
    n_u = x.shape[1]
    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)
    path = F._save_map(tmp_path, "context_end", "250", fitted, layers, map_seed=0)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["meta"].pop("map_seed")
    torch.save(blob, path)

    assert F._load_nl_map(tmp_path, "context_end", "250", "mlp", layers, n_u, map_seed=7)


@pytest.mark.parametrize("kind", sorted(fits.NONLINEAR_MAP_KINDS))
def test_map_roundtrip_gate_passes_on_a_genuine_payload(tmp_path, kind):
    F = _load_script_module()
    x, y = _pool()
    layers = [14, 19]
    n_u = x.shape[1]
    fitted = fits.fit_nonlinear_map(x, y, kind=kind, device="cpu", seed=0)
    F._save_map(tmp_path, "context_end", "full", fitted, layers, map_seed=0)

    rec = F._verify_map_roundtrip(
        tmp_path, "context_end", "full", kind, layers, n_u, fitted, x[:, :8], map_seed=0
    )
    assert rec["cos_min"] >= F.MAP_ROUNDTRIP_COS_MIN
    assert rec["rel_max_abs_diff"] <= F.MAP_ROUNDTRIP_REL_MAX
    assert rec["n_probe_rows"] == 8


def test_map_roundtrip_gate_FAILS_LOUD_on_a_corrupted_payload(tmp_path):
    """Deliberate corruption: swap the per-layer payloads so layer i gets layer j's map.

    This is the defect class the gate exists for — a layer-misaligned payload
    still loads, still has the right shapes, still passes every metadata guard,
    and would silently produce wrong arm scores in all 6 lanes.
    """
    import torch

    F = _load_script_module()
    # distinct per-layer structure so a swap is detectable
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 40, 6))
    y = np.stack([np.tanh(x[0] * 2.0), -np.tanh(x[1] * 0.5) + 3.0])
    layers = [14, 19]
    n_u = x.shape[1]
    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)
    path = F._save_map(tmp_path, "context_end", "full", fitted, layers, map_seed=0)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["payloads"] = list(reversed(blob["payloads"]))  # layer-misaligned
    torch.save(blob, path)

    with pytest.raises(RuntimeError, match="round-trip gate FAILED"):
        F._verify_map_roundtrip(
            tmp_path, "context_end", "full", "mlp", layers, n_u, fitted, x[:, :8], map_seed=0
        )


def test_map_roundtrip_gate_FAILS_LOUD_when_the_payload_fails_the_reader(tmp_path):
    """A payload the lanes' own reader would reject is a gate FAILURE, not a skip."""
    import torch

    F = _load_script_module()
    x, y = _pool()
    layers = [14, 19]
    n_u = x.shape[1]
    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)
    path = F._save_map(tmp_path, "context_end", "full", fitted, layers, map_seed=0)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["meta"]["diagnostics"] = {}  # drops the held-out map-quality companions
    torch.save(blob, path)

    with pytest.raises(RuntimeError, match="does not pass _load_nl_map"):
        F._verify_map_roundtrip(
            tmp_path, "context_end", "full", "mlp", layers, n_u, fitted, x[:, :8], map_seed=0
        )


# --------------------------------------------------------------------------
# Staging step: consumer-open verification over a LOCAL fixture mirror.
# --------------------------------------------------------------------------


def test_stage_maps_check_payload_accepts_a_real_payload_and_names_every_defect(tmp_path):
    F = _load_script_module()
    S = _stage_module()
    x, y = _pool()
    layers = [14, 19]
    fitted = fits.fit_nonlinear_map(x, y, kind="mlp", device="cpu", seed=0)
    path = F._save_map(tmp_path / "maps", "context_end", "250", fitted, layers, map_seed=0)
    # _save_map roots itself at <arg>/maps, so pass the parent to hit tmp_path/maps
    good = S.check_payload(path, "context_end", "250", "mlp")
    assert good["ok"], good["reasons"]
    assert good["map_seed"] == 0
    assert good["n_layers"] == 2

    missing = S.check_payload(tmp_path / "maps" / "nope.pt", "context_end", "250", "mlp")
    assert not missing["ok"] and "missing" in missing["reasons"]

    wrong_kind = S.check_payload(path, "context_end", "250", "kernel")
    assert not wrong_kind["ok"]
    assert any("map_kind" in r for r in wrong_kind["reasons"])

    wrong_variant = S.check_payload(path, "prefix_end", "250", "mlp")
    assert not wrong_variant["ok"]
    assert any("variant" in r for r in wrong_variant["reasons"])


def test_stage_maps_expected_keys_covers_the_path2_map_set():
    S = _stage_module()
    keys = S.expected_keys(("prefix_end", "context_end"), ("250", "full"), ("mlp", "kernel"))
    assert len(keys) == 8  # 2 variants x 2 rungs x 2 kinds
    assert len(set(keys)) == 8
    assert ("prefix_end", "full", "kernel") in keys
