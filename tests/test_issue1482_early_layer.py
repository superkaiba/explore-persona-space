"""Pins for the #1482 early-layer-arm round driver (plan v16).

Fast, CPU-only, network-free: committed-literal pins + pure-function lattice
checks + the launcher's exit-path shape. The heavy paths (capture, fits,
uploads) are covered by the round's driver smoke, not here.
"""

from __future__ import annotations

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
