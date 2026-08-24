"""Pins for the issue-2474 ``postnorm-l27-diagnostic`` driver (offline math + gates).

Pins (CPU-only, no network, no model download):
  * ``rms_norm_rows`` reproduces the REAL ``Qwen2RMSNorm`` module BIT-EXACTLY
    under the module's own finite-dtype convention (bf16 input/weight, fp32
    variance, normalized-state downcast BEFORE the weight multiply) — and the
    pin REJECTS the named wrong conventions (fp64-only, no-downcast,
    eps-outside-rsqrt), eps-dominated small-magnitude rows included.
  * ``decode_bf16_le`` round-trips torch bf16 bytes exactly.
  * ``build_comparison_figure`` survives a deliberately INVERTED bootstrap CI;
    ``phase_figs`` renders one figure PER setting (the em negative control
    included).
  * Fail-loud gates: staging failures raise (never a caps-only descope);
    ``_upload_means`` raises on an empty upload return and on a missing remote
    set (and scopes the commit away from ``*.partial.npz``); Gate P / Gate R
    raise on drift; the rescore completion fingerprint changes when ANY
    consumed artifact changes; the vhat sha pin rejects perturbed bytes;
    train-row / stored-mu contracts reject malformed inputs pre-GPU;
    ``--smoke-dir`` refuses any root outside the temp dir; partial checkpoints
    round-trip spot-gate evidence; the poller parses the sentinel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue2474_postnorm as pn

# ---------------------------------------------------------------------------
# RMSNorm operator (finite-dtype convention)
# ---------------------------------------------------------------------------


def _bf16_fixture():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    hdim = 16
    x64 = np.vstack(
        [
            rng.normal(0, 3.0, size=(5, hdim)),  # normal magnitude
            rng.normal(0, 1e-4, size=(4, hdim)),  # eps-dominated (var ~1e-8 vs eps 1e-6)
            rng.normal(0, 40.0, size=(3, hdim)),  # large magnitude
        ]
    )
    w64 = rng.normal(1.0, 0.5, size=hdim)
    x_bf = torch.from_numpy(x64.astype(np.float32)).to(torch.bfloat16)
    w_bf = torch.from_numpy(w64.astype(np.float32)).to(torch.bfloat16)
    return torch, hdim, x_bf, w_bf


def test_rms_norm_rows_matches_qwen2_rmsnorm_bf16_exact():
    """Bit-exact parity with the real module at the PRODUCTION dtype (bf16)."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    torch, hdim, x_bf, w_bf = _bf16_fixture()
    eps = 1e-6
    ref_mod = Qwen2RMSNorm(hdim, eps=eps).to(torch.bfloat16)
    with torch.no_grad():
        ref_mod.weight.copy_(w_bf)
        ref = ref_mod(x_bf)
    assert ref.dtype == torch.bfloat16  # the module returns the input dtype
    # rms_norm_rows consumes bf16-representable fp64 values (the production
    # consumption shape: stored fp16/bf16 states loaded as np.float64).
    mine = pn.rms_norm_rows(x_bf.to(torch.float64).numpy(), w_bf.to(torch.float64).numpy(), eps)
    np.testing.assert_array_equal(mine, ref.to(torch.float64).numpy())


def test_rms_norm_rows_rejects_wrong_dtype_and_eps_conventions():
    """The pin discriminates the named failure modes (r1 codex blocker:
    an fp32-only atol pin could not see them)."""
    torch, _, x_bf, w_bf = _bf16_fixture()
    eps = 1e-6
    x64 = x_bf.to(torch.float64).numpy()
    w64 = w_bf.to(torch.float64).numpy()
    good = pn.rms_norm_rows(x64, w64, eps)

    # Mutation 1: the pre-fix fp64-only operator (no bf16 anywhere).
    rms = np.sqrt(np.mean(np.square(x64), axis=-1, keepdims=True) + eps)
    m1 = (x64 / rms) * w64
    assert not np.array_equal(m1, good)

    # Mutation 2: normalized states NOT cast back to bf16 before the weight.
    hs = x_bf.to(torch.float32)
    var = hs.pow(2).mean(-1, keepdim=True)
    hs = hs * torch.rsqrt(var + eps)
    m2 = (w_bf.to(torch.float32) * hs).to(torch.float64).numpy()
    assert not np.array_equal(m2, good)

    # Mutation 3: eps OUTSIDE the rsqrt — decisive on eps-dominated rows.
    hs2 = x_bf.to(torch.float32)
    var2 = hs2.pow(2).mean(-1, keepdim=True)
    hs2 = hs2 / (torch.sqrt(var2) + eps)
    m3 = (w_bf * hs2.to(torch.bfloat16)).to(torch.float64).numpy()
    assert not np.array_equal(m3, good)
    tiny = slice(5, 9)  # the 1e-4-magnitude block
    rel = np.max(np.abs(m3[tiny] - good[tiny]) / (np.abs(good[tiny]) + 1e-12))
    assert rel > 0.1  # material, not a tolerance-absorbable drift


def test_rms_norm_rows_is_row_wise_nonlinear():
    """mean-of-normed != norm-of-mean — the grain distinction the round hinges on."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.0, size=(8, 16))
    x[0] *= 10.0  # one large-norm row makes the two grains diverge
    w = np.ones(16)
    mean_of_norm = pn.rms_norm_rows(x, w, 1e-6).mean(axis=0)
    norm_of_mean = pn.rms_norm_rows(x.mean(axis=0), w, 1e-6)
    assert not np.allclose(mean_of_norm, norm_of_mean, atol=1e-3)


def test_decode_bf16_le_roundtrip():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    vals32 = rng.normal(0, 4.0, size=64).astype(np.float32)
    bf = torch.from_numpy(vals32).to(torch.bfloat16)
    raw = bf.view(torch.uint16).numpy().astype("<u2").tobytes()
    decoded = pn.decode_bf16_le(raw)
    np.testing.assert_array_equal(decoded, bf.to(torch.float32).numpy())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _fig_stats_payload(*, invert_ci: bool, settings: tuple[str, ...] = ("caps",)) -> dict:
    conds = ["lang_a", "lang_b"]
    fams = {}
    for fi, fam in enumerate(pn.TRAINREF_FAMS):
        grains = {}
        for gi, grain in enumerate(("pre", "post_rowgrain")):
            point = 0.4 + 0.05 * fi - 0.1 * gi
            ci = [point + 0.2, point - 0.2] if invert_ci else [point - 0.2, point + 0.2]
            grains[grain] = {
                "pooled_rho": point,
                "pooled_ci95": ci,
                "per_condition": {c: {"rho": point + 0.03 * k} for k, c in enumerate(conds)},
            }
        fams[fam] = {"level": grains}
    return {
        "layer": 27,
        "settings": {
            s: {"conds": conds, "variants": {"full": {"families": fams}}} for s in settings
        },
    }


@pytest.mark.parametrize("invert_ci", [False, True])
def test_comparison_figure_survives_inverted_ci(tmp_path, invert_ci):
    paths = pn.build_comparison_figure(
        _fig_stats_payload(invert_ci=invert_ci), tmp_path, setting="caps"
    )
    png = [Path(p) for p in paths.values() if str(p).endswith(".png")]
    assert png and png[0].is_file() and png[0].stat().st_size > 5_000


def test_phase_figs_renders_every_setting(tmp_path):
    """One figure PER setting in the stats — the em negative control included
    (r1 codex concern em-negative-control-figure-omitted)."""
    stats = _fig_stats_payload(invert_ci=False, settings=("caps", "em"))
    out_dir = tmp_path / "out"
    fig_dir = tmp_path / "figs"
    out_dir.mkdir()
    (out_dir / "postnorm_stats.json").write_text(json.dumps(stats))
    cfg = {"out_dir": out_dir, "fig_dir": fig_dir, "settings": ("caps", "em")}
    res = pn.phase_figs(argparse.Namespace(), cfg)
    assert sorted(res["figure_paths"]) == ["caps", "em"]
    for setting in ("caps", "em"):
        pngs = [Path(p) for p in res["figure_paths"][setting].values() if str(p).endswith(".png")]
        assert pngs and pngs[0].is_file() and pngs[0].stat().st_size > 5_000
        assert setting in pngs[0].name


def test_err_offsets_clamps_non_negative():
    lo, hi = pn._err_offsets(0.5, [0.7, 0.3])  # inverted CI
    assert lo == 0.0 and hi == 0.0
    lo, hi = pn._err_offsets(0.5, [0.3, 0.7])
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# --smoke-dir containment guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["/", str(pn.REPO_ROOT), "~", "/home", "/workspace", "/tmp", "/data/some/dir"],
)
def test_safe_smoke_root_rejects_non_scratch_paths(bad):
    with pytest.raises(RuntimeError, match="refusing recursive delete"):
        pn._safe_smoke_root(bad)


def test_safe_smoke_root_accepts_tempdir_child(tmp_path):
    import tempfile

    inside = str(Path(tempfile.gettempdir()) / "issue2474-postnorm-smoke-testx")
    assert pn._safe_smoke_root(inside) == Path(inside).resolve()
    # pytest tmp_path lives under the temp dir too
    assert pn._safe_smoke_root(str(tmp_path / "sub")) == (tmp_path / "sub").resolve()


# ---------------------------------------------------------------------------
# Trainref means: fail-loud staging + schema validation
# ---------------------------------------------------------------------------


def _means_payload(cond: str, setting: str, hdim: int = 8, n_rows: int = 3) -> dict:
    vec = [float(v) for v in np.linspace(0.1, 1.0, hdim)]
    return {
        "fingerprint": {"recipe": pn.RECIPE_TAG},
        "cond": cond,
        "setting": setting,
        "layer": 1,
        "n_rows": n_rows,
        "model_ident": "tiny:test",
        "gate_p": {"verdict": "PASS"},
        "mu_c_pre": vec,
        "mu_a_pre": vec,
        "mu_c_post": vec,
        "mu_a_post_rowgrain": vec,
        "mu_a_post_tokengrain": vec,
        "norm_weight_sha256": "0" * 64,
    }


def _mini_cfg(tmp_path, *, synthetic: bool = True) -> dict:
    return {
        "synthetic": synthetic,
        "layer": 1,
        "settings": ("s1", "s2"),
        "conds": {"s1": ["c1"], "s2": ["c2"]},
        "expected_rows": None,
        "means_dir": tmp_path / "means",
    }


def test_load_means_staging_failure_raises_never_descopes(tmp_path, monkeypatch):
    """A Hub/auth/transport failure on the SECONDARY setting raises — it is
    never converted into a caps-only success (r1 codex blocker
    secondary-setting-stage-fail-soft)."""
    from explore_persona_space.orchestrate import hub

    cfg = _mini_cfg(tmp_path, synthetic=False)
    cfg["means_dir"].mkdir(parents=True)
    # production mode validates hidden dim == EXPECTED_HIDDEN
    (cfg["means_dir"] / "c1.json").write_text(
        json.dumps(_means_payload("c1", "s1", hdim=pn.EXPECTED_HIDDEN))
    )

    # signature-conformant boundary fake (mirrors hub.stage_hub_file)
    def _raise_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        raise ConnectionError("simulated transport/auth failure")

    monkeypatch.setattr(hub, "stage_hub_file", _raise_stage)
    with pytest.raises(RuntimeError, match="not stageable"):
        pn._load_means(argparse.Namespace(), cfg, ["s1", "s2"])
    # Selecting only the present setting loads fine — but only via the
    # EXPLICIT selection argument, never an implicit skip.
    out = pn._load_means(argparse.Namespace(), cfg, ["s1"])
    assert sorted(out) == ["s1"]


def test_validate_means_setting_rejects_bad_payloads(tmp_path):
    cfg = _mini_cfg(tmp_path)

    good = {"c1": _means_payload("c1", "s1")}
    pn._validate_means_setting(cfg, "s1", good)  # passes

    nan = {"c1": _means_payload("c1", "s1")}
    nan["c1"]["mu_c_post"][0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        pn._validate_means_setting(cfg, "s1", nan)

    cfg_rows = dict(cfg, expected_rows={"c1": 99})
    with pytest.raises(RuntimeError, match="n_rows"):
        pn._validate_means_setting(cfg_rows, "s1", {"c1": _means_payload("c1", "s1")})

    nogate = {"c1": _means_payload("c1", "s1")}
    del nogate["c1"]["gate_p"]
    with pytest.raises(RuntimeError, match="Gate P"):
        pn._validate_means_setting(cfg, "s1", nogate)

    wrongcond = {"c1": _means_payload("cX", "s1")}
    with pytest.raises(RuntimeError, match="cond/setting"):
        pn._validate_means_setting(cfg, "s1", wrongcond)


def test_upload_means_raises_on_empty_return_and_missing_set(tmp_path, monkeypatch):
    from explore_persona_space.orchestrate import hub

    means_dir = tmp_path / "means"
    means_dir.mkdir()
    (means_dir / "c1.json").write_text("{}")
    (means_dir / "c1.partial.npz").write_bytes(b"stale checkpoint residue")
    cfg = {"means_dir": means_dir}
    seen: dict = {}

    # signature-conformant boundary fakes (mirror hub._upload / verify_*)
    def _upload_empty(
        local_path,
        repo_id,
        repo_type,
        path_in_repo,
        delete_after=False,
        upload_as_file=False,
        ignore_patterns=None,
        private=False,
        raise_on_error=False,
    ):
        seen["ignore_patterns"] = ignore_patterns
        return ""

    monkeypatch.setattr(hub, "_upload", _upload_empty)
    with pytest.raises(RuntimeError, match="upload returned no path"):
        pn._upload_means(cfg)
    # the commit is scoped away from stale checkpoint residue (r1 codex NIT)
    assert "*.partial.npz" in seen["ignore_patterns"]

    def _upload_ok(
        local_path,
        repo_id,
        repo_type,
        path_in_repo,
        delete_after=False,
        upload_as_file=False,
        ignore_patterns=None,
        private=False,
        raise_on_error=False,
    ):
        return "https://example/url"

    def _verify_missing(
        api, repo_id, expected_repo_paths, *, path_in_repo, repo_type="dataset", revision=None
    ):
        return list(expected_repo_paths)

    monkeypatch.setattr(hub, "_upload", _upload_ok)
    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", _verify_missing)
    with pytest.raises(RuntimeError, match="missing remote paths"):
        pn._upload_means(cfg)


# ---------------------------------------------------------------------------
# Sentinel: the poller parses what the driver writes
# ---------------------------------------------------------------------------


def test_write_sentinel_parses_with_poller(tmp_path):
    import poll_pipeline

    cfg = {"logs_dir": tmp_path, "synthetic": True}
    pn._write_sentinel(argparse.Namespace(), cfg, phase="smoke", note_payload={"x": 1})
    files = list(tmp_path.glob("issue-2474-*.json"))
    assert len(files) == 1
    parsed = poll_pipeline._parse_sentinel(files[0].name, files[0].read_text())
    assert parsed is not None and parsed["kind"] == "epm:smoke-result"
    note = json.loads(parsed["note"])
    assert note["phase"] == "smoke" and note["x"] == 1


# ---------------------------------------------------------------------------
# Gates P and R fail loud on drift
# ---------------------------------------------------------------------------


def test_gate_p_check_raises_on_drift_and_passes_on_match():
    rng = np.random.default_rng(3)
    mu_c = rng.normal(0, 1, 16)
    mu_a = rng.normal(0, 1, 16)
    rec = pn._gate_p_check(mu_c, mu_a, mu_c.copy(), mu_a.copy(), "condX")
    assert rec["verdict"] == "PASS" and rec["cos_c_pre_vs_stored"] == pytest.approx(1.0)
    drifted = mu_c + rng.normal(0, 1.0, 16)  # far beyond the 0.999 bar
    with pytest.raises(RuntimeError, match="Gate P FAIL"):
        pn._gate_p_check(drifted, mu_a, mu_c, mu_a, "condX")


def _gate_r_fixture(n_t: int = 4, layer: int = 1):
    rng = np.random.default_rng(4)
    res = {"cond": {"c1": {}}, "shared": {}}
    stored = {}
    for fam in pn.ALL_FAMS:
        vals = [float(v) for v in rng.uniform(-0.5, 0.5, n_t)]
        res["cond"]["c1"][fam] = np.array(vals)
        stored[fam] = [[None] * n_t, vals]
    prefit = {"conditions": {"c1": {"families_layered": stored}}}
    return res, prefit


def test_gate_r_fails_on_score_drift():
    res, prefit = _gate_r_fixture()
    assert pn._gate_r("s1", res, prefit, ["c1"], 1)["verdict"] == "PASS"
    res["cond"]["c1"]["ctx_trainref"] = res["cond"]["c1"]["ctx_trainref"] + 1e-3  # > 1e-6 tol
    with pytest.raises(RuntimeError, match="Gate R FAIL"):
        pn._gate_r("s1", res, prefit, ["c1"], 1)


# ---------------------------------------------------------------------------
# Rescore completion fingerprint covers every consumed artifact
# ---------------------------------------------------------------------------

_FP_ARTIFACTS = [
    "capture/predictor_captures/base_s1/grid.pt",
    "capture/predictor_captures/base_s1/ceiling.pt",
    "capture/predictor_captures/base_mu_c1/mu.pt",
    "means/c1.json",
    "vhat_s1.pt",
    "passb.pt",
    "prefit_scores.json",
]


def _fp_cfg(tmp_path) -> dict:
    for rel in _FP_ARTIFACTS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"content-" + rel.encode())
    return {
        "synthetic": True,
        "layer": 1,
        "settings": ("s1",),
        "conds": {"s1": ["c1"]},
        "capture_root": tmp_path / "capture" / "predictor_captures",
        "means_dir": tmp_path / "means",
        "vhat_path": {"s1": tmp_path / "vhat_s1.pt"},
        "vhat_sha256": {"s1": "abc"},
        "passb_path": tmp_path / "passb.pt",
        "prefit_scores_path": tmp_path / "prefit_scores.json",
    }


@pytest.mark.parametrize("rel", _FP_ARTIFACTS)
def test_rescore_fingerprint_changes_when_any_input_changes(tmp_path, rel):
    """Mutating ANY consumed artifact invalidates the completion fingerprint —
    a Gate-R-skipping stale reuse is impossible (r1 codex blocker
    rescore-fingerprint-omits-inputs)."""
    cfg = _fp_cfg(tmp_path)
    means = {"s1": {"c1": {}}}
    norm = {"sha": "normsha"}
    base = pn._rescore_fingerprint(cfg, means, norm)
    assert pn._rescore_fingerprint(cfg, means, norm) == base  # deterministic re-stat
    target = tmp_path / rel
    target.write_bytes(target.read_bytes() + b"-mutated")
    assert pn._rescore_fingerprint(cfg, means, norm) != base


def test_rescore_fingerprint_records_norm_sha_and_selection(tmp_path):
    cfg = _fp_cfg(tmp_path)
    means = {"s1": {"c1": {}}}
    fp = pn._rescore_fingerprint(cfg, means, {"sha": "n1"})
    assert fp["norm_sha"] == "n1" and fp["settings"] == ["s1"]
    assert fp != pn._rescore_fingerprint(cfg, means, {"sha": "n2"})


# ---------------------------------------------------------------------------
# vhat identity + schema gates
# ---------------------------------------------------------------------------


def _vhat_cfg(tmp_path, tensor, *, synthetic=True, sha=None, layer=1, setting="s1"):
    torch = pytest.importorskip("torch")
    path = tmp_path / "vhat.pt"
    torch.save({"v_hat_mapB": tensor, "layer": layer, "setting": setting}, path)
    return {
        "synthetic": synthetic,
        "layer": 1,
        "vhat_path": {"s1": path},
        "vhat_sha256": {"s1": pn._sha256_file(path) if sha == "self" else sha},
    }


def test_load_vhat_sha_pin_rejects_perturbed_bytes(tmp_path):
    torch = pytest.importorskip("torch")
    t = torch.zeros((3, 8), dtype=torch.float16) + 0.5
    cfg = _vhat_cfg(tmp_path, t, sha="self")
    assert pn._load_vhat(cfg, "s1", 3, 8).shape == (3, 8)
    # perturb ONE element (schema + row count + Gate-R-style aggregates blind)
    t2 = t.clone()
    t2[0, 0] = 0.5009765625  # one fp16 ULP-scale bump — real byte change
    path = cfg["vhat_path"]["s1"]
    torch.save({"v_hat_mapB": t2, "layer": 1, "setting": "s1"}, path)
    with pytest.raises(RuntimeError, match="pinned parent-producer sha"):
        pn._load_vhat(cfg, "s1", 3, 8)


def test_load_vhat_schema_gates(tmp_path):
    torch = pytest.importorskip("torch")
    t = torch.zeros((3, 8), dtype=torch.float16) + 0.5
    cfg = _vhat_cfg(tmp_path, t, sha="self")
    with pytest.raises(RuntimeError, match="shape"):
        pn._load_vhat(cfg, "s1", 3, 16)  # wrong hidden dim
    with pytest.raises(RuntimeError, match="rows"):
        pn._load_vhat(cfg, "s1", 5, 8)  # wrong row count
    bad = t.clone()
    bad[0, 0] = float("nan")
    cfg_nan = _vhat_cfg(tmp_path, bad, sha="self")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        pn._load_vhat(cfg_nan, "s1", 3, 8)
    # production mode with NO pin for the setting fails loud, never skips
    cfg_nopin = _vhat_cfg(tmp_path, t, synthetic=False, sha=None)
    with pytest.raises(RuntimeError, match="no pinned parent-producer sha"):
        pn._load_vhat(cfg_nopin, "s1", 3, 8)


def test_production_vhat_sha_pins_present_for_every_setting():
    assert sorted(pn.VHAT_SHA256) == ["caps", "em"]
    assert all(len(v) == 64 for v in pn.VHAT_SHA256.values())


# ---------------------------------------------------------------------------
# Pre-GPU input contracts (train rows + stored mu)
# ---------------------------------------------------------------------------


def _train_row(sys_c="s", user_c="u", gold="g") -> str:
    return json.dumps(
        {
            "prompt": [
                {"role": "system", "content": sys_c},
                {"role": "user", "content": user_c},
            ],
            "completion": [{"role": "assistant", "content": gold}],
        }
    )


def test_validate_train_rows_accepts_good_and_counts(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(_train_row() + "\n\n" + _train_row() + "\n")
    assert pn._validate_train_rows(p) == 2


@pytest.mark.parametrize(
    "row, msg",
    [
        ("{not json", "invalid JSON"),
        (json.dumps({"prompt": [{"role": "user", "content": "u"}], "completion": []}), "prompt"),
        (
            json.dumps(
                {
                    "prompt": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                    ],
                    "completion": [{"role": "assistant", "content": "   "}],
                }
            ),
            "completion",
        ),
    ],
)
def test_validate_train_rows_rejects_malformed_any_row(tmp_path, row, msg):
    p = tmp_path / "t.jsonl"
    p.write_text(_train_row() + "\n" + row + "\n")  # bad row is NOT the first
    with pytest.raises(RuntimeError, match=msg):
        pn._validate_train_rows(p)


def _mu_file(tmp_path, *, n_layers=2, hdim=8, n_c=3, nan=False, drop_key=None):
    torch = pytest.importorskip("torch")
    mu = torch.zeros((n_layers, hdim), dtype=torch.float16) + 0.25
    if nan:
        mu[1, 0] = float("nan")
    payload = {"mu_train": mu, "mu_a_train": mu.clone(), "n_c": n_c, "n_a": n_c}
    if drop_key:
        del payload[drop_key]
    path = tmp_path / "mu.pt"
    torch.save(payload, path)
    return path


def test_validate_stored_mu_contracts(tmp_path):
    good = _mu_file(tmp_path)
    pn._validate_stored_mu(good, 1, 3, 8)  # passes
    with pytest.raises(RuntimeError, match="missing keys"):
        pn._validate_stored_mu(_mu_file(tmp_path, drop_key="mu_a_train"), 1, 3, 8)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        pn._validate_stored_mu(_mu_file(tmp_path, nan=True), 1, 3, 8)
    with pytest.raises(RuntimeError, match="n_c"):
        pn._validate_stored_mu(_mu_file(tmp_path), 1, 99, 8)
    with pytest.raises(RuntimeError, match="no layer"):
        pn._validate_stored_mu(_mu_file(tmp_path), 5, 3, 8)
    with pytest.raises(RuntimeError, match="shape"):
        pn._validate_stored_mu(_mu_file(tmp_path), 1, 3, 16)


# ---------------------------------------------------------------------------
# Partial checkpoints round-trip spot-gate evidence
# ---------------------------------------------------------------------------


def test_partial_roundtrip_preserves_spot_records(tmp_path):
    partial = tmp_path / "c1.partial.npz"
    fp = {"phase": "trainref-gpu", "cond": "c1", "v": 1}
    sums = {"sum_c_pre": np.arange(4.0)}
    spot = [{"cos_context": 1.0, "cos_answer_mean": 0.999, "n_resp_tokens": 5}]
    pn._save_partial(partial, fp, sums, {"n": 2}, 3, spot)
    st = pn._load_partial(partial, fp)
    assert st is not None and st["n"] == 2 and st["next_line_idx"] == 3
    assert st["spot"] == spot
    np.testing.assert_array_equal(st["sums"]["sum_c_pre"], sums["sum_c_pre"])
    assert pn._load_partial(partial, {**fp, "cond": "OTHER"}) is None  # fingerprint mismatch


# ---------------------------------------------------------------------------
# Explicit setting selection
# ---------------------------------------------------------------------------


def test_selected_settings_explicit_descope_and_unknown():
    cfg = {"settings": ("caps", "em")}
    ns = argparse.Namespace(settings="all")
    assert pn._selected_settings(ns, cfg) == ["caps", "em"]
    assert pn._selected_settings(argparse.Namespace(settings="caps"), cfg) == ["caps"]
    with pytest.raises(RuntimeError, match="unknown"):
        pn._selected_settings(argparse.Namespace(settings="caps,bogus"), cfg)
    with pytest.raises(RuntimeError, match="no settings"):
        pn._selected_settings(argparse.Namespace(settings=" , "), cfg)
