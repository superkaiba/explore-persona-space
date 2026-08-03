"""#1774 round-A CPU pins.

Covers: the registered channel-count rule (contiguous-from-top + the SAME rule
inside each null draw -> count-null band + BH companion), the leave-one-row-out
query-averaged means, the cuSOLVER CPU-fallback wrappers, the device-safe
operator wrapper's W/b parity against the engine predictions (plan asm 5), the
poll_pipeline results-sentinel shape, the steering-manifest validation, and the
dispatcher's bash syntax. All CPU, all tmp_path-only writes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1774_common as c  # noqa: E402


def test_contiguous_count_from_top_stops_at_first_below() -> None:
    obs = np.array([0.5, 0.4, 0.05, 0.3])
    p95 = np.array([0.1, 0.1, 0.1, 0.1])
    assert c.contiguous_count_from_top(obs, p95) == 2  # stops at 0.05, ignores 0.3
    assert c.contiguous_count_from_top(np.array([0.0]), np.array([0.1])) == 0
    nan_obs = np.array([0.5, np.nan, 0.5])
    assert c.contiguous_count_from_top(nan_obs, p95[:3]) == 1  # NaN stops the run


def test_count_null_band_applies_same_rule_per_draw() -> None:
    # 3 components; draw 0 clears the p95 on the top 2, draw 1 on none.
    null_mat = np.array(
        [
            [0.30, 0.30, 0.00],
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
        ]
    )
    p95 = np.array([0.2, 0.2, 0.2])
    band = c.count_null_band(null_mat, p95)
    assert band["null_counts"] == [2, 0, 0]
    assert band["count_max"] == 2
    assert 0.0 <= band["count_p95"] <= 2.0


def test_bh_count_flags_strong_components() -> None:
    rng = np.random.default_rng(0)
    null_mat = rng.normal(0.0, 0.01, size=(200, 5))
    obs = np.array([0.5, 0.4, 0.0, 0.0, 0.0])
    n = c.bh_count(obs, null_mat)
    assert n == 2


def test_loro_query_avg_exact_leave_one_out() -> None:
    X = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 7.0]])
    prefix_ids = np.array(["a", "a", "a", "b"])
    X_loro, keep, means = c.loro_query_avg(X, prefix_ids)
    assert keep.tolist() == [True, True, True, False]  # singleton prefix excluded
    np.testing.assert_allclose(X_loro[0], [(3.0 + 5.0) / 2, 0.0])
    np.testing.assert_allclose(X_loro[1], [(1.0 + 5.0) / 2, 0.0])
    np.testing.assert_allclose(means["a"], [3.0, 0.0])
    np.testing.assert_allclose(means["b"], [7.0, 7.0])


def test_eigh_robust_cpu_fallback_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    G = torch.randn(8, 8, dtype=torch.float64)
    G = G @ G.T
    want_w, want_v = torch.linalg.eigh(G)
    real = torch.linalg.eigh
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.linalg.LinAlgError("simulated cuSOLVER non-convergence")
        return real(x)

    monkeypatch.setattr(torch.linalg, "eigh", flaky)
    w, v = c.eigh_robust(G)
    assert calls["n"] == 2  # fallback branch executed
    torch.testing.assert_close(w, want_w)
    torch.testing.assert_close(v.abs(), want_v.abs())


def test_svd_robust_cpu_fallback_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    A = torch.randn(6, 4, dtype=torch.float64)
    _want_u, want_s, _want_vh = torch.linalg.svd(A, full_matrices=False)
    real = torch.linalg.svd
    calls = {"n": 0}

    def flaky(x, full_matrices=True):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.linalg.LinAlgError("simulated cuSOLVER non-convergence")
        return real(x, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "svd", flaky)
    _u, s, _vh = c.svd_robust(A)
    assert calls["n"] == 2
    torch.testing.assert_close(s, want_s)


def test_operator_raw_safe_matches_engine_predictions() -> None:
    """Plan asm 5: held-out predictions from (W, b) match engine predict <=1e-8."""
    import issue1774_fit_battery as fb

    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 12))
    W_true = rng.normal(size=(5, 12))
    Y = X @ W_true.T + 0.01 * rng.normal(size=(60, 5))
    fit = fb.fit_press_ext(X, Y, "cpu")
    W, b = fb.operator_raw(fit)
    assert W.shape == (5, 12) and b.shape == (5,)
    X_new = rng.normal(size=(7, 12))
    direct = X_new @ W.numpy().T + b.numpy()
    engine = fb.predict(fit, X_new)
    np.testing.assert_allclose(direct, engine, atol=1e-8, rtol=0)


def test_results_sentinel_envelope_shape(tmp_path: Path) -> None:
    env = c.results_sentinel_envelope(
        gpu_hours_used=1.25, plan_deviations=["example"], out_root=str(tmp_path)
    )
    assert env["sentinel_schema_version"] == 1
    assert env["kind"] == "epm:results" and env["task_id"] == 1774
    payload = env["note"]
    for k in (
        "eval_numbers",
        "eval_paths",
        "reproducibility_card",
        "wandb_url",
        "hf_hub_url",
        "worktree_path",
        "final_commit_sha",
        "gpu_hours_used",
        "gpu_hours_budgeted",
        "plan_deviations",
    ):
        assert k in payload, k
    assert payload["wandb_url"] == "n/a"
    card = payload["reproducibility_card"]
    assert not any("adapter" in k.lower() for k in card)  # no-training task
    assert "N/A" in card["wandb"]  # wandb declared N/A explicitly, never omitted
    assert payload["gpu_hours_budgeted"] == 10.0
    # smoke variant flips the kind so a smoke run can never mint epm:results
    smoke = c.results_sentinel_envelope(gpu_hours_used=0.0, out_root=str(tmp_path), smoke=True)
    assert smoke["kind"] == "epm:smoke-result"


def test_phase_sentinel_envelope_shape() -> None:
    env = c.phase_sentinel_envelope("p1", "done")
    assert env["sentinel_schema_version"] == 1
    assert env["kind"] == "epm:progress"
    assert "[phase=p1]" in env["note"]


def test_judge_manifest_validation(tmp_path: Path) -> None:
    import issue1774_judge as jm

    rows = [
        {
            "row_id": "d00-top-sv0-pos-c001",
            "condition": "top_sv0_pos",
            "question": "q",
            "completion": "a",
        },
        {
            "row_id": "d01-rb-evil-neg-c001",
            "condition": "rb_evil_neg",
            "question": "q",
            "completion": "a",
        },
    ]
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps({"meta": {}, "rows": rows}))
    loaded = jm.load_manifest_rows(mpath)
    assert len(loaded) == 2
    assert [r["row_id"] for r in jm.rows_for_trait(loaded, "evil")] == ["d01-rb-evil-neg-c001"]
    assert len(jm.rows_for_trait(loaded, "sycophancy")) == 2

    bad = dict(rows[0], row_id="has__delimiter")
    mpath.write_text(json.dumps({"meta": {}, "rows": [bad]}))
    with pytest.raises(ValueError, match="custom_id delimiter"):
        jm.load_manifest_rows(mpath)
    long_id = dict(rows[0], row_id="x" * 60)
    mpath.write_text(json.dumps({"meta": {}, "rows": [long_id]}))
    with pytest.raises(ValueError, match="longer than"):
        jm.load_manifest_rows(mpath)


def test_dispatch_script_bash_syntax_and_contract() -> None:
    script = REPO / "scripts" / "issue1774_dispatch.sh"
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    text = script.read_text()
    # pod-side reporting contract: sentinels only, never a task.py shellout
    assert "scripts/task.py" not in text
    for token in (
        "issue-1774-$1-done.json",  # phase_sentinel writes p{1,2,3}-done.json
        "issue-1774-results.json",
        "[phase=done]",
        "CUDA_VISIBLE_DEVICES",
    ):
        assert token in text, token
    # every phase slug routes through phase_sentinel (p3 included)
    for slug in ("phase_sentinel p1", "phase_sentinel p2", "phase_sentinel p3"):
        assert slug in text, slug
