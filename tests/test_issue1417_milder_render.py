"""#1417 milder-rude-render amendment round — additive-registry + pilot-gate pins.

Plan v6 (§4.2) pins:
  * ADDITIVE registry: the five ORIGINAL cells' ``render_config()`` entries +
    the ``gen``/``tracks``/``anchor_cells``/``models`` blocks byte-match the
    committed ``eval_results/issue_1417/render_config.json``, and
    ``PRIOR_RENDER_HASHES`` equals exactly the committed hash — the safety
    contract that makes the prior-hash acceptance in ``fingerprint_matches``
    valid (carried v1/refit artifacts stay readable ONLY because nothing they
    were produced under changed).
  * Judge rubrics FROZEN: the mild cell keeps on the SAME two rubric objects
    as ``c2_rude``.
  * Battery ``--cells`` filter + the 3 registered mild pairs; ``--h-rude-cell``
    feeds the H-table's rude slot.
  * Pilot gate (issue1417_pilot_gate.py): the §12.13 trace on the committed
    refit values (base yield 0.109 FIRES the yield arm; variance 0.4864/0.4620
    does NOT fire the variance arm), the byte-matched trace-cov formula, and a
    real-body e2e on a tiny synthetic store (network boundary faked with
    signature-conformant fakes only).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1417_battery as b1417  # noqa: E402
import issue1417_judge as j1417  # noqa: E402
import issue1417_pilot_gate as pg  # noqa: E402
import issue1417_render as r1417  # noqa: E402

COMMITTED_CONFIG = REPO_ROOT / "eval_results/issue_1417/render_config.json"
REFIT_SUMMARY = REPO_ROOT / "eval_results/issue_1417/refit/battery_summary.json"


# ---------------------------------------------------------------------------
# Additive registry + prior-hash acceptance
# ---------------------------------------------------------------------------
def test_additive_registry_byte_matches_committed_config():
    if not COMMITTED_CONFIG.exists():
        pytest.skip("eval_results/issue_1417 cone absent (pre-existing sparse worktree)")
    committed = json.loads(COMMITTED_CONFIG.read_text())
    cc = committed["render_config"]
    rc = r1417.render_config()
    # Every ORIGINAL cell's entry byte-identical (label/format/system/stop).
    assert set(cc["cells"]) == set(r1417.CELL_ORDER) - {"c2_rude_mild"}
    for slug, entry in cc["cells"].items():
        assert rc["cells"][slug] == entry, f"{slug}: registry entry drifted (must-ask)"
    for block in ("gen", "tracks", "anchor_cells", "models", "c4_preamble", "n_shared_expected"):
        assert rc[block] == cc[block], f"{block}: drifted (must-ask)"
    # PRIOR_RENDER_HASHES == exactly the committed v1/refit hash.
    assert (committed["render_config_hash"],) == r1417.PRIOR_RENDER_HASHES


def test_prior_hash_acceptance_single_choke_point():
    new = r1417.render_config_hash()
    assert new not in r1417.PRIOR_RENDER_HASHES  # the addition flipped the hash
    assert r1417.fingerprint_matches({"render_config_hash": new})
    for prior in r1417.PRIOR_RENDER_HASHES:
        assert r1417.fingerprint_matches({"render_config_hash": prior})
    assert not r1417.fingerprint_matches({"render_config_hash": "deadbeefdeadbeef"})
    assert not r1417.fingerprint_matches({})


def test_mild_cell_registry_entry():
    assert r1417.CELL_ORDER[-1] == "c2_rude_mild"  # appended LAST
    cfg = r1417.CELLS["c2_rude_mild"]
    assert cfg["format"] == "chat" and cfg["stop"] is None
    assert cfg["system"] == r1417.C2_MILD_SYSTEM
    assert cfg["label"] == "Rude-but-informative (mild)"
    # The mild render is a NEW string, not a mutation of the published C2.
    assert r1417.C2_MILD_SYSTEM != r1417.C2_SYSTEM


def test_judge_rubrics_frozen_for_mild_cell():
    # Plan §4.2 item 2: SAME rubric objects as c2_rude (rubric text untouched);
    # keep = both means >= 50 (the shared KEEP_THRESHOLD path).
    assert j1417.CELL_RUBRICS["c2_rude_mild"] == ["rude_register", "informativeness"]
    assert j1417.CELL_RUBRICS["c2_rude_mild"] == j1417.CELL_RUBRICS["c2_rude"]
    assert "c2_rude_mild" not in j1417.DIAGNOSTIC_RUBRICS


# ---------------------------------------------------------------------------
# Battery --cells filter + registered mild pairs + --h-rude-cell
# ---------------------------------------------------------------------------
def _args(**kw) -> types.SimpleNamespace:
    base = dict(
        data_dir=Path("data/issue_1417"),
        out_dir=Path("eval_results/issue_1417"),
        judge_dir=None,
        lambda_selection="gcv",
        gcv_dof_cap=None,
        cells=",".join(r1417.CELL_ORDER),
        h_rude_cell="c2_rude",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_cells_filter_parses_and_validates():
    assert b1417.cells_filter(_args()) == list(r1417.CELL_ORDER)
    assert b1417.cells_filter(_args(cells="c2_rude_mild")) == ["c2_rude_mild"]
    # Order-preserving regardless of input order.
    assert b1417.cells_filter(_args(cells="c2_rude_mild,c1_helpful_ctrl")) == [
        "c1_helpful_ctrl",
        "c2_rude_mild",
    ]
    with pytest.raises(AssertionError, match="unknown cells"):
        b1417.cells_filter(_args(cells="c9_nope"))


def test_battery_pairs_mild_cell_registered_pairs():
    pairs = b1417.battery_pairs("instruct")
    mild = [p["pair_id"] for p in pairs if p["cell"] == "c2_rude_mild"]
    assert mild == [
        "instruct__c2_rude_mild__vs_c0_chat__ctx",
        "instruct__c2_rude_mild__vs_c1__ctx",
        "instruct__c2_rude_mild__vs_c1__prefix",
    ]
    # The v1/refit pair set is unchanged when filtered to the five originals
    # (the carried record's 26 pair files stay the same 13-per-model ids).
    originals = {c for c in r1417.CELL_ORDER if c != "c2_rude_mild"}
    old = [p for p in pairs if p["cell"] in originals]
    assert len(old) == 13


def test_run_summary_h_rude_cell_feeds_h_table(tmp_path):
    def _battery_json(model: str, cell: str, delta_ci: list[float]) -> None:
        p = tmp_path / "battery" / f"battery_{model}__{cell}__vs_c0_chat__ctx.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "headline_layer": 19,
                    "rel_by_layer": {"19": {"rel": 0.6}},
                    "rel_bootstrap_l19": {"delta_rel_ci95": delta_ci},
                }
            )
        )

    for model in r1417.MODELS:
        _battery_json(model, "c2_rude_mild", [0.05, 0.20])  # Shared
        _battery_json(model, "c4_exposition", [0.02, 0.15])  # Shared

    args = _args(out_dir=tmp_path, h_rude_cell="c2_rude_mild")
    assert b1417.run_summary(args) == 0
    out = json.loads((tmp_path / "battery_summary.json").read_text())
    assert out["h_rude_cell"] == "c2_rude_mild"
    for model in r1417.MODELS:
        assert out["h_table_lookup"][model] == "Neither (generic QA structure)"

    # Default h-rude-cell (c2_rude, no battery file here) stays Inconclusive —
    # the v1/refit lookup shape is preserved byte-for-byte.
    args2 = _args(out_dir=tmp_path, h_rude_cell="c2_rude")
    assert b1417.run_summary(args2) == 0
    out2 = json.loads((tmp_path / "battery_summary.json").read_text())
    for model in r1417.MODELS:
        assert out2["h_table_lookup"][model].startswith("Inconclusive")


# ---------------------------------------------------------------------------
# Pilot gate — §12.13 trace on committed values, formula pin, real-body e2e
# ---------------------------------------------------------------------------
def test_lane_verdict_trace_on_committed_refit_values():
    """Plan §12.13: evaluated on the committed refit battery_summary values —
    the yield arm FIRES on the v1 base-rude yield (the incident the gate
    exists for); the variance arm deliberately does NOT fire at the marginal
    0.486/0.462 values (that region belongs to the binding 0.5 full floor);
    no false fire on the passing instruct lane."""
    if REFIT_SUMMARY.exists():
        cells = json.loads(REFIT_SUMMARY.read_text())["cells"]
        base_yield = float(cells["pretrained__c2_rude"]["yield_frac"])
        base_var = float(cells["pretrained__c2_rude"]["y_var_ratio_vs_c0"])
        inst_yield = float(cells["instruct__c2_rude"]["yield_frac"])
        inst_var = float(cells["instruct__c2_rude"]["y_var_ratio_vs_c0"])
    else:  # sparse worktree: the plan-recorded values (same trace)
        base_yield, base_var = 0.10922946655376799, 0.46199417941937876
        inst_yield, inst_var = 0.6621507197290432, 0.4864017519731178

    base = pg.lane_verdict(base_yield, base_var)
    assert base["yield_arm_fires"] is True  # 0.109 < 0.40 => FIRES
    assert base["var_arm_fires"] is False  # 0.462 > 0.40 => does NOT fire
    assert base["verdict"] == "fail"

    inst = pg.lane_verdict(inst_yield, inst_var)
    assert inst["yield_arm_fires"] is False and inst["var_arm_fires"] is False
    assert inst["verdict"] == "pass"


def test_lane_verdict_nan_and_boundary():
    nan = float("nan")
    v = pg.lane_verdict(0.5, nan)
    assert v["var_arm_fires"] is True and v["verdict"] == "fail"
    v = pg.lane_verdict(nan, 0.5)
    assert v["yield_arm_fires"] is True and v["verdict"] == "fail"
    # Bars are inclusive (>=): exactly-at-bar passes.
    v = pg.lane_verdict(0.40, 0.40)
    assert v["verdict"] == "pass"


def test_trace_cov_l19_byte_matches_fit825_formula():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((7, 28, 5)).astype(np.float32)
    got = pg.trace_cov_l19(Y)
    # The literal issue825_fit_cells.run_cell formula (fit_cells.py:1078-1081).
    expected = float(Y[:, 19, :].astype(np.float64).var(axis=0, ddof=1).sum())
    assert got == expected
    # Independent formulation: tr(cov(Y_l19)) == sum of per-dim variances.
    assert got == pytest.approx(float(np.trace(np.cov(Y[:, 19, :].astype(np.float64).T))))


def _write_store_fixture(store: Path, model: str, cell: str, conv_ids: list[str], seed: int = 3):
    """A tiny real .pt shard + sidecar in the extractor's exact contract."""
    rng = np.random.default_rng(seed)
    n = len(conv_ids)
    slots = [
        torch.as_tensor(rng.standard_normal((2, 28, 8)), dtype=torch.float32) for _ in range(n)
    ]
    profiles = [
        torch.as_tensor(rng.standard_normal((1, 28, 8)), dtype=torch.float32) for _ in range(n)
    ]
    store.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{cell}_s_shard000"
    torch.save({"conv_ids": conv_ids, "slots": slots, "profiles": profiles}, store / f"{stem}.pt")
    (store / f"{stem}.json").write_text(json.dumps({"conv_ids": conv_ids, **r1417.fingerprint()}))
    Y = np.stack([p.numpy() for p in profiles])[:, 0, :, :]  # ctx-arm target
    return Y


def _pilot_fixture(tmp_path: Path, model: str, yield_frac: float, kept: list[str]) -> dict:
    conv_ids = [f"s{i}" for i in range(6)]
    Y = _write_store_fixture(tmp_path / "data" / "store", model, "c2_rude_mild", conv_ids)
    keep_rows = [conv_ids.index(c) for c in kept]
    expected_var = float(Y[keep_rows][:, 19, :].astype(np.float64).var(axis=0, ddof=1).sum())
    pilot_judge = tmp_path / "pilot" / "judge"
    pilot_judge.mkdir(parents=True, exist_ok=True)
    (pilot_judge / f"kept_{model}_c2_rude_mild.json").write_text(
        json.dumps(
            {
                **r1417.fingerprint(),
                "model": model,
                "cell": "c2_rude_mild",
                "n_judged": 6,
                "n_kept": len(kept),
                "yield_frac": yield_frac,
                "kept_conv_ids": kept,
            }
        )
    )
    anchors = tmp_path / "anchors"
    anchors.mkdir(exist_ok=True)
    aid = "S1" if model == "instruct" else "S2"
    denom = expected_var / 0.8  # -> var_ratio 0.8 (clears the 0.40 bar)
    (anchors / f"cells_{aid}.json").write_text(json.dumps({"y_trace_cov_frozen": {"19": denom}}))
    return {"expected_ratio": expected_var / denom, "expected_var": expected_var}


def _run_gate(tmp_path: Path, monkeypatch, model: str) -> tuple[int, dict]:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue1417_pilot_gate.py",
            "--data-dir",
            str(tmp_path / "data"),
            "--pilot-dir",
            str(tmp_path / "pilot"),
            "--anchors-dir",
            str(tmp_path / "anchors"),
            "--out",
            str(tmp_path / "pilot_gate_report.json"),
            "--models",
            model,
            "--skip-staging",
        ],
    )
    rc = pg.main()
    report = json.loads((tmp_path / "pilot_gate_report.json").read_text())
    return rc, report


def test_pilot_gate_e2e_passing_lane(tmp_path, monkeypatch):
    fx = _pilot_fixture(tmp_path, "instruct", yield_frac=0.55, kept=["s1", "s3", "s4"])
    rc, report = _run_gate(tmp_path, monkeypatch, "instruct")
    assert rc == 0 and report["pass"] is True
    lane = report["lanes"]["instruct"]
    assert lane["verdict"] == "pass"
    assert lane["var_ratio"] == pytest.approx(fx["expected_ratio"])
    assert lane["y_trace_cov_l19_pilot"] == pytest.approx(fx["expected_var"])


def test_pilot_gate_e2e_yield_arm_fires(tmp_path, monkeypatch):
    """Degenerate probe: the v1 base-lane yield (0.109) trips the yield arm
    while the variance path still computes (rc 23 — the designed halt)."""
    _pilot_fixture(tmp_path, "pretrained", yield_frac=0.109, kept=["s0", "s2", "s5"])
    rc, report = _run_gate(tmp_path, monkeypatch, "pretrained")
    assert rc == pg.RC_PILOT_GATE == 23
    lane = report["lanes"]["pretrained"]
    assert lane["yield_arm_fires"] is True and lane["var_arm_fires"] is False
    assert report["lanes_failing"] == ["pretrained"]


def test_pilot_gate_e2e_below_two_kept_rows(tmp_path, monkeypatch):
    """< 2 kept rows: variance is undefined (ddof=1) — NaN reads fail-closed."""
    _pilot_fixture(tmp_path, "instruct", yield_frac=0.05, kept=["s1"])
    rc, report = _run_gate(tmp_path, monkeypatch, "instruct")
    assert rc == pg.RC_PILOT_GATE
    lane = report["lanes"]["instruct"]
    assert lane["yield_arm_fires"] is True and lane["var_arm_fires"] is True


def test_stage_pilot_store_stages_only_covering_shards(tmp_path, monkeypatch):
    """Real stage_pilot_store body; ONLY the network boundary is faked with
    signature-conformant fakes (hub.stage_hub_file / list_hf_files_under_path
    def-mirrored). shard000 covers the pilot ids -> .pt staged; shard001 does
    not -> sidecar staged, .pt NOT staged."""
    import shutil

    model, cell = "instruct", "c2_rude_mild"
    prefix = f"{r1417.HF_PREFIX}/analysis_tensors/store"
    remote = tmp_path / "remote"
    _write_store_fixture(remote, model, cell, [f"s{i}" for i in range(4)])
    # A second, NON-covering shard (distinct conv ids).
    stem1 = f"{model}_{cell}_s_shard001"
    torch.save({"conv_ids": ["z0"], "slots": [], "profiles": []}, remote / f"{stem1}.pt")
    (remote / f"{stem1}.json").write_text(json.dumps({"conv_ids": ["z0"], **r1417.fingerprint()}))

    def fake_list(api, repo_id, path, *, repo_type="model", revision=None):
        assert path == prefix and repo_type == "dataset"
        return [f"{prefix}/{p.name}" for p in sorted(remote.iterdir())]

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        shutil.copy(remote / Path(path_in_repo).name, target)
        return Path(target)

    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setattr(hub, "list_hf_files_under_path", fake_list)
    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)

    data_dir = tmp_path / "data"
    pg.stage_pilot_store(data_dir, model, cell, {"s1", "s2"})
    store = data_dir / "store"
    assert (store / f"{model}_{cell}_s_shard000.pt").exists()
    assert (store / f"{model}_{cell}_s_shard000.json").exists()
    assert (store / f"{stem1}.json").exists()
    assert not (store / f"{stem1}.pt").exists()  # non-covering shard: no .pt

    with pytest.raises(AssertionError, match="no shard covers"):
        pg.stage_pilot_store(tmp_path / "data2", model, cell, {"never-present"})
