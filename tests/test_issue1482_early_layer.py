"""Pins for the #1482 early-layer-arm round driver (plan v17).

Fast, CPU-only, network-free: committed-literal pins + pure-function lattice
checks + the launcher's exit-path shape + the v17 §0.-1 judged-label-freeze
gating (--phase judge hard exit; evidence-packet schema; label-free
mech_tests). The heavy paths (capture, fits, uploads) are covered by the
round's driver smoke, not here.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

EL = pytest.importorskip("issue1482_early_layer")

SPLIT_JSON = REPO / "eval_results" / "issue_1482" / "split_1482.json"
LAUNCHER = REPO / "scripts" / "issue1482_early_layer_launch.sh"


@pytest.mark.skipif(not SPLIT_JSON.exists(), reason="sparse checkout without issue_1482 cone")
def test_committed_split_sha_constants_pin_committed_file():
    """EARLY_COMMITTED_SPLIT_SHAS literals == the git-committed split_1482.json
    (the PDSHRINK pattern: a silent split re-pin fails loud in CI)."""
    doc = json.loads(SPLIT_JSON.read_text())
    assert EL.EARLY_COMMITTED_SPLIT_SHAS["sae_fit_sha256"] == doc["sae_fit"]["sha256"]
    assert EL.EARLY_COMMITTED_SPLIT_SHAS["holdout_sha256"] == doc["holdout"]["sha256"]


def test_gate_be_verdict_lattice():
    """Gate B-e lattice (plan §7): PASS >= 0.825; WARN [0.675, 0.825) with k128
    escalation; HALT < 0.675."""
    assert EL.gate_be_verdict(0.90, 0.0) == ("PASS", 64)
    assert EL.gate_be_verdict(0.825, 0.0) == ("PASS", 64)
    assert EL.gate_be_verdict(0.80, 0.90) == ("WARN", 128)
    assert EL.gate_be_verdict(0.80, 0.80) == ("WARN", 64)
    assert EL.gate_be_verdict(0.675, 0.5) == ("WARN", 64)
    assert EL.gate_be_verdict(0.674, 0.9) == ("HALT", 64)


def test_published_fve_l3_pins():
    """L3 published FVE literals == the Hub-read values at the pinned revision
    (frac_variance_explained, trainer eval_results.json, 2026-07-28)."""
    import issue1482_sae as S

    assert S.PUBLISHED_FVE_BY_LAYER[3] == {64: 0.93087890625, 128: 0.94208984375}
    # legacy alias unchanged for existing L19 callers
    assert S.PUBLISHED_FVE == {64: 0.80572265625, 128: 0.84236328125}
    assert S.trainer_subdir(3, 64) == "resid_post_layer_3/trainer_1"
    assert S.trainer_subdir(3, 128) == "resid_post_layer_3/trainer_2"
    assert S.trainer_subdir(19, 64) == "resid_post_layer_19/trainer_1"
    with pytest.raises(AssertionError):
        S.trainer_subdir(4, 64)  # off the suite grid


def test_select_tails_matches_fe_select_at_production_params():
    """_select_tails(com, 150, 15) reproduces FE._select(com) index sets exactly
    (instrument parity: production calls FE._select VERBATIM; the parameterized
    clone exists only for the smoke's small feature counts)."""
    import issue1482_feature_extremes as FE

    rng = np.random.default_rng(14823)
    com = {
        "feat_ids": np.arange(16384, dtype=np.int64),
        "r2": rng.normal(0.1, 0.2, 16384),
        "activity": rng.uniform(0.001, 0.9, 16384),
    }
    ours = EL._select_tails(com, n_tail=FE.N_TAIL, n_decile_tail=FE.N_DECILE_TAIL)
    ref = FE._select(com)
    for key in ("a_best", "a_worst", "b_best", "b_worst", "union"):
        assert ours["idx"][key] == ref["idx"][key], f"selection drift on {key}"


def test_shuffle_seeds_and_reconciliation_record():
    """Seed registry (plan §10) + the fit_mlp check-(k) disposition record."""
    assert tuple(range(1_482_100, 1_482_120)) == EL.SHUFFLE_SEEDS
    assert EL.SUBSAMPLE_SEED == 14823
    assert EL.BOOT_PERM_SEED == 148_230
    assert "not-needed" in EL.FIT_MLP_RECONCILIATION["disposition"]
    assert set(EL.FIT_MLP_RECONCILIATION["branch_commits_not_needed"]) == {
        "d7c1c55fbe",
        "a2dd635b4d",
        "689f5c1042",
    }


def test_launcher_exit_path_shape():
    """Launcher (plan §4 item 3): set -euo pipefail; explicit `|| rc=$?` capture;
    failed-sentinel-before-exit; no `false` in compound branches; single
    terminal [phase=done]; no pod-side task.py shellout."""
    text = LAUNCHER.read_text()
    assert "set -euo pipefail" in text
    assert re.search(r"\|\|\s*rc=\$\?", text), "explicit rc capture missing"
    assert "write_failed_sentinel" in text
    assert re.search(r"^\s*false\b", text, re.M) is None, "`false` in a compound branch"
    assert text.count('echo "[phase=done]"') == 1
    assert "task.py" not in text, "pod-side task-workflow CLI shellout is banned"


def test_launcher_failed_sentinel_writer(tmp_path):
    """The extracted failed-sentinel function writes a poll_pipeline-conformant
    epm:failure sentinel (kind/rc/failure_class/blocks_pipeline)."""
    body = []
    in_fn = False
    for line in LAUNCHER.read_text().split("\n"):
        if line.startswith("write_failed_sentinel()"):
            in_fn = True
        if in_fn:
            body.append(line)
        if in_fn and line == "PY":
            body.append("}")
            break
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {REPO}",
            f"LOGS_DIR={tmp_path}",
            *body,
            "write_failed_sentinel fits--full 9",
        ]
    )
    sh = tmp_path / "probe.sh"
    sh.write_text(script)
    subprocess.run(["bash", str(sh)], check=True, capture_output=True, timeout=120)
    files = list(tmp_path.glob("issue-1482-earlylayer-failed-*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text())
    assert doc["kind"] == "epm:failure"
    assert doc["rc"] == 9
    assert doc["failure_class"] == "code"
    assert doc["blocks_pipeline"] is True


def test_verify_reused_artifact_keys_npz_support(tmp_path):
    """The .npz extension added for the plan §4 S0 probes: superset PASS (rc 0)
    + missing-key detection (rc 1)."""
    from verify_reused_artifact_keys import main as vk_main

    p = tmp_path / "bundle.npz"
    np.savez(p, row_idx=np.arange(3), ans_mean=np.zeros(3))
    assert vk_main(["--artifact", str(p), "--keys", "row_idx,ans_mean"]) == 0
    assert vk_main(["--artifact", str(p), "--keys", "row_idx,missing_key"]) == 1


def test_results_sentinel_smoke_leg_kind_never_epm_results(tmp_path):
    """The --full launcher runs the SMOKE leg first; its results sentinel must
    carry kind epm:smoke-result (kresample precedent) so the poller can never
    drain smoke numbers as the real epm:results (#1586 chained-legs class)."""
    from types import SimpleNamespace

    out_eval = tmp_path / "eval"
    out_eval.mkdir()
    pilot = {
        "gate_be": {"verdict": "PASS"},
        "chosen_k": 64,
        "layers_fve": {"k64": {"L3": {"fve": 0.9}}, "k128": {"L3": {"fve": 0.94}}},
        "g2e_early_cos_min": 1.0,
        "g2e_flat_cos_min": 1.0,
        "tokens_per_s": 100.0,
    }
    (out_eval / "early_pilot.json").write_text(json.dumps(pilot))
    (out_eval / "early_summary.json").write_text(json.dumps({"pooled_r2": {}, "n_rows": {}}))
    (out_eval / "split_early.json").write_text(
        json.dumps({"s_fit_sha256": "a", "s_score_sha256": "b"})
    )
    (out_eval / "phase_times.json").write_text(
        json.dumps({"phases": [{"name": "pilot", "wall_s": 1.0}]})
    )
    for smoke, want_kind in ((True, "epm:smoke-result"), (False, "epm:results")):
        logs = tmp_path / f"logs_{smoke}"
        logs.mkdir()
        args = SimpleNamespace(out_eval=out_eval, smoke=smoke)
        EL._results_sentinel(args, logs_dir=logs)
        doc = json.loads((logs / "issue-1482-results.json").read_text())
        assert doc["kind"] == want_kind, (smoke, doc["kind"])
        assert ("SMOKE leg" in doc["note"]) == smoke


def test_h1_depth_stratified_verdict_lattice():
    """H1 permutation read: a strong within-depth level->R2 signal is
    level-positive; shuffled labels are null-persists (seeded)."""
    rng = np.random.default_rng(7)
    rows = [
        {
            "feat_id": i,
            "depth": d,
            "level": "high" if i % 2 else "low",
            "r2": (0.5 if i % 2 else 0.1) + rng.normal(0, 0.01),
        }
        for i in range(60)
        for d in (3, 19)
    ]
    h1 = EL._h1_depth_stratified(rows, 300, np.random.default_rng(1))
    assert h1["verdict"] == "level-positive"
    rows_null = [
        {
            "feat_id": i,
            "depth": 3,
            "level": "high" if rng.random() < 0.5 else "low",
            "r2": float(rng.normal()),
        }
        for i in range(120)
    ]
    h1n = EL._h1_depth_stratified(rows_null, 300, np.random.default_rng(2))
    assert h1n["verdict"] == "null-persists"


def test_descope_floor_keeps_post_carve_tr_above_d():
    """Concern earlylayer-descope-carve-underdetermined (r2): the recomputed floor
    guarantees post-carve ridge-train rows STRICTLY > d on the widest design for
    ANY realized descope; below the floor the phase exits loud (RC_THROUGHPUT)."""
    assert EL.D_WIDEST_DESIGN == 16_384  # 2 x prod max_features_in
    assert EL.PROD_VAL_CARVE == 2_000
    assert EL.DESCOPE_FLOOR_CONTEXTS == 22_982  # ceil((16,384 + 2,000 + 1) / 0.8)
    n_total = 90_000  # envelope large enough that a sub-1/3-tps descope is feasible
    tps_at = EL.TPS_BASIS * (EL.DESCOPE_FLOOR_CONTEXTS + 0.5) / n_total
    d = EL.descope_plan(tps_at, n_total, val_carve=EL.PROD_VAL_CARVE)
    assert d is not None and d["n_fit"] == 18_385 and d["n_score"] == 22_982 - 18_385
    assert d["effective_tr"] == 16_385 > EL.D_WIDEST_DESIGN
    # one context below the floor -> SystemExit(RC_THROUGHPUT), never an
    # under-determined ridge
    tps_below = EL.TPS_BASIS * (EL.DESCOPE_FLOOR_CONTEXTS - 0.5) / n_total
    with pytest.raises(SystemExit) as ei:
        EL.descope_plan(tps_below, n_total, val_carve=EL.PROD_VAL_CARVE)
    assert ei.value.code == EL.RC_THROUGHPUT
    # production envelope (30k): ANY sub-kill-floor tps descopes below the floor
    # -> halt-loud (the plan §7 kill shape)
    with pytest.raises(SystemExit) as ei2:
        EL.descope_plan(1000.0, 30_000, val_carve=EL.PROD_VAL_CARVE)
    assert ei2.value.code == EL.RC_THROUGHPUT
    # tps at/above the kill floor -> no descope at all
    assert EL.descope_plan(EL.TPS_BASIS * EL.TPS_KILL_FRAC, 30_000, val_carve=2_000) is None
    # sweep the whole feasible descope band: every returned plan keeps tr > d
    for n_desc in range(EL.DESCOPE_FLOOR_CONTEXTS, 30_000, 137):
        tps = EL.TPS_BASIS * (n_desc + 0.5) / n_total
        dd = EL.descope_plan(tps, n_total, val_carve=EL.PROD_VAL_CARVE)
        assert dd is not None and dd["effective_tr"] > EL.D_WIDEST_DESIGN, n_desc
    # an oversized CLI --val-carve breaches the floor's guarantee -> fail-loud
    with pytest.raises(AssertionError):
        EL.descope_plan(tps_at, n_total, val_carve=8_000)


def test_estimator_validity_guard_lattice():
    """_e3_prep's realized-width backstop: production trips at tr <= d; the smoke
    shape is exempt (deliberate under-determined regime)."""
    EL._assert_estimator_validity(16_385, 16_384, smoke=False)  # passes: tr > d
    with pytest.raises(AssertionError):
        EL._assert_estimator_validity(16_384, 16_384, smoke=False)  # tr == d trips
    EL._assert_estimator_validity(10, 16_384, smoke=True)  # smoke exempt


def test_primary_perfeature_resolves_gate_chosen_k(tmp_path):
    """Concern gatebe-warn-escalation-not-threaded (r2): gate record chosen_k=128
    => the primary L3 source resolves to the k128 npz; 64 -> the k64 default;
    anything else fails loud. E5-evidence tail selection + E6 H1 join route
    through the helper (no bare k64 literal left on the label-keyed paths)."""
    ns = argparse.Namespace(out_eval=tmp_path)
    (tmp_path / "early_pilot.json").write_text(json.dumps({"chosen_k": 128}))
    assert EL._primary_l3_perfeature(ns) == "perfeature_l3_k128"
    (tmp_path / "early_pilot.json").write_text(json.dumps({"chosen_k": 64}))
    assert EL._primary_l3_perfeature(ns) == "perfeature_l3_default"
    (tmp_path / "early_pilot.json").write_text(json.dumps({"chosen_k": 32}))
    with pytest.raises(AssertionError):
        EL._primary_l3_perfeature(ns)
    for fn in (EL.phase_evidence, EL._pooled_h1_rows):
        src = inspect.getsource(fn)
        assert "_primary_l3_perfeature" in src, fn.__name__
        assert '"perfeature_l3_default.npz"' not in src, fn.__name__


COMMITTED_EXTREMES = REPO / "eval_results" / "issue_1482" / "feature_extremes" / "extremes.json"


@pytest.mark.skipif(
    not COMMITTED_EXTREMES.exists(), reason="sparse checkout without issue_1482 cone"
)
def test_pooled_h1_rows_join_escalated_primary(tmp_path):
    """Functional E6 pin (the #1773 resume leg): chosen_k=128 => _pooled_h1_rows
    joins the provided labels against the k128 npz r2 (the k64 npz's value for a
    different feature never rides in)."""
    (tmp_path / "early_pilot.json").write_text(json.dumps({"chosen_k": 128}))
    labels = {"7": {"level": "high"}, "9": {"level": "low"}}
    np.savez(
        tmp_path / "perfeature_l3_k128.npz",
        feat_ids=np.asarray([7], np.int64),
        r2=np.asarray([0.5]),
        activity=np.asarray([0.1]),
    )
    np.savez(
        tmp_path / "perfeature_l3_default.npz",
        feat_ids=np.asarray([9], np.int64),
        r2=np.asarray([0.9]),
        activity=np.asarray([0.1]),
    )
    rows = EL._pooled_h1_rows(argparse.Namespace(out_eval=tmp_path), labels)
    l3 = [r for r in rows if r["depth"] == 3]
    assert l3 == [{"feat_id": 7, "depth": 3, "level": "high", "r2": 0.5}]


def test_phase_judge_frozen_hard_exit():
    """v17 §0.-1 freeze pin: --phase judge is a hard exit (RC_JUDGE_FROZEN) that
    dispatches NOTHING — an old invocation can never silently judge — and the
    stub names the freeze + the #1773 resume."""
    with pytest.raises(SystemExit) as ei:
        EL.phase_judge(argparse.Namespace())
    assert ei.value.code == EL.RC_JUDGE_FROZEN
    src = inspect.getsource(EL.phase_judge)
    assert "FROZEN" in src and "#1773" in src
    assert "dispatch_judge_items" not in src, "no judge dispatch may remain in the stub"
    # the evidence replacement leg is wired into the phase dispatcher
    main_src = inspect.getsource(EL.main)
    assert '"evidence"' in main_src and '"--labels-file"' in main_src
    assert "dispatch_judge_items" not in inspect.getsource(EL.phase_evidence)


def test_evidence_entries_required_keys():
    """Evidence-packet schema pin (v17 §0.-1 clauses (a)-(e)): every selected
    feature carries {selection, top_answers, coact_neighbors, footprint_tokens,
    covariates}; the k128-escalation branch records the covariate-battery
    dictionary mismatch instead of a mis-joined k64 row."""
    sel = {
        "features": [
            {
                "feat_id": 7,
                "restricted_idx": 1,
                "r2": 0.5,
                "activity": 0.2,
                "decile": 3,
                "a_best": True,
                "a_worst": False,
                "b_best": False,
                "b_worst": True,
            }
        ]
    }
    top = {"7": [[3.0, 100, 0], [2.0, 102, 2]]}
    neighbors = [[[5, 1, 0.5], [9, 1, 0.5]]]
    fp_tokens = [[[11, " the", 4.2]]]
    cov = {7: {"activity": 0.2, "consistency": 0.7, "dense_flag": 0}}
    ent = EL._evidence_entries(
        sel, top, neighbors, fp_tokens, [7], cov, {7: 40}, 512, "perfeature_l3_default", 64
    )
    e = ent["7"]
    assert set(e) == {
        "selection",
        "top_answers",
        "coact_neighbors",
        "footprint_tokens",
        "covariates",
    }
    assert e["selection"]["set_b_worst"] is True and e["selection"]["r2_rank_asc"] == 40
    assert e["selection"]["source_arm"] == "perfeature_l3_default"
    assert e["top_answers"] == top["7"] and e["covariates"]["consistency"] == 0.7
    # escalation branch: no k64 covariate row may be joined onto a k128 feature
    ent128 = EL._evidence_entries(
        sel, top, neighbors, fp_tokens, [7], None, {7: 40}, 512, "perfeature_l3_k128", 128
    )
    assert "note" in ent128["7"]["covariates"] and "k64" in ent128["7"]["covariates"]["note"]


def test_evidence_scan_and_coact_topk(tmp_path):
    """Body-executing pin for the E5-evidence shard scan: FIT-rows-only top-8
    ranking with [val, ci, row_idx] ids (no text), and the co-activation
    neighbour GEMM (E3 matrix semantics, self excluded)."""
    np.savez(
        tmp_path / "pooled_l3_shard00_chunk0000.npz",
        row_idx=np.asarray([0, 1, 2], np.int64),
        set_tag=np.asarray([1, 0, 1], np.int8),  # row 1 is HOLDOUT — excluded
        ci=np.asarray([100, 101, 102], np.int64),
        idx_off=np.asarray([2, 1, 2], np.int64),
        ans_idx=np.asarray([5, 7, 5, 7, 9], np.int64),
        ans_max=np.asarray([1.0, 3.0, 9.0, 2.0, 0.5], np.float32),
    )
    feat_ids = np.asarray([5, 7, 9], np.int64)
    top, a, col_of = EL._evidence_scan(
        [tmp_path / "pooled_l3_shard00_chunk0000.npz"],
        [7],
        feat_ids,
        ("idx_off", "ans_idx", "ans_max"),
    )
    # the 9.0 holdout firing (row 1) must NOT appear; fit rows rank 3.0 > 2.0
    assert top == {"7": [[3.0, 100, 0], [2.0, 102, 2]]}
    assert a.shape == (2, 3)  # 2 FIT rows x 3 restricted features
    ucols = col_of[np.asarray([7], np.int64)]
    nb = EL._coact_topk(a, feat_ids, ucols, topk=10)
    assert len(nb) == 1
    assert sorted(t[0] for t in nb[0]) == [5, 9]  # self (7) excluded
    for _fid, count, frac in nb[0]:
        assert count == 1 and frac == 0.5  # C[7,g]=1 over C[7,7]=2


def test_mech_doc_label_free_and_deferred_stanza(tmp_path):
    """mech_tests pin (v17 §0.-1): the mechanical doc carries the depth profile /
    null band / twin agreement / baselines and NO label-dependent field; the
    deferred stanza names #1773 + the --labels-file resume."""
    rng = np.random.default_rng(3)

    def _pf(name, n=8):
        np.savez(
            tmp_path / f"{name}.npz",
            feat_ids=np.arange(n, dtype=np.int64),
            r2=rng.normal(0.1, 0.2, n),
            activity=rng.uniform(0.01, 0.9, n),
        )

    for name in (
        "perfeature_l3_default",
        "perfeature_l19_matched_ctx",
        "perfeature_l3_sinkmask",
        "perfeature_l3_k128",
    ):
        _pf(name)
    np.savez(tmp_path / "shuffle_null_l3.npz", r2=rng.normal(0.0, 0.05, 160))
    (tmp_path / "early_summary.json").write_text(
        json.dumps(
            {
                "baselines": {"l3_identity_bias": {"n_shared_ids": 0}},
                "knn": {"l3_sae_ctx": {"euclidean": {"acc@1": 0.5}}},
                "pooled_r2": {"perfeature_l3_default": 0.2},
            }
        )
    )
    doc = EL._mech_doc(argparse.Namespace(out_eval=tmp_path), "perfeature_l3_default")
    assert set(doc) == {
        "depth_profile_matched_n",
        "shuffle_null",
        "twin_agreement",
        "baselines_identity",
        "knn_retrieval",
        "pooled_r2",
    }
    dp = doc["depth_profile_matched_n"]
    assert set(dp) == {"l3_primary", "l19_matched_ctx", "l3_sinkmask_twin", "l3_k128_twin"}
    assert "frac_above_shuffle_null_p97_5" in dp["l3_primary"]
    assert doc["twin_agreement"]["n_paired"] == 8
    # NO judged-label field may enter the mechanical doc
    assert not any("h1" in k or "h2" in k or "level" in k for k in doc)
    stanza = EL._deferred_stanza()
    assert "#1773" in json.dumps(stanza) and "--labels-file" in stanza["resume"]
    assert stanza["h1_pooled_depth_stratified"].startswith("deferred")
