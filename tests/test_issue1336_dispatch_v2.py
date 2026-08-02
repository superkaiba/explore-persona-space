"""#1336 Unit D: v2 dispatch gate helpers (G0' / G1') + gen new-rows filter.

Pins the plan v13 dispatcher seams Unit D added: run_g0v2's three legs (the
leg-(b) Gram-vs-primal equality ENFORCED at any n; the (a) anchor tolerance
demoted to informational on a fixture — the #1345 gate-calibration rule),
run_g1v2_check's kill/pass/NaN semantics against bar_v2, the concat-boundary
single-source aliasing, and gen prep's new-prompts-only filter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue1336_extract_turnstore as et  # noqa: E402
import issue1336_fit_cells as f36  # noqa: E402
import issue1336_gen_answers as g36  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Concat registry aliasing (one boundary source for gen prep + extractor)
# ---------------------------------------------------------------------------
def test_concat_registry_single_source():
    assert et.CONCAT_SOURCES is cm.V2_CONCAT_SOURCES
    assert et.CONCAT_BOUNDARY is cm.V2_CONCAT_BOUNDARY
    assert cm.V2_CONCAT_BOUNDARY == {"lmsys23k": 5000, "gsm8k_train_full": 5000}
    assert set(cm.V2_FULLY_REUSED_GEN) == {"gsm8k_test1319"}


# ---------------------------------------------------------------------------
# gen prep: new-prompts-only filter (plan §4 Phase GEN)
# ---------------------------------------------------------------------------
def test_new_rows_only_filters_concat_corpus():
    rows = [{"prompt_idx": i, "prompt": f"p{i}"} for i in (0, 4999, 5000, 7000)]
    kept = g36.new_rows_only("lmsys23k", rows)
    assert [r["prompt_idx"] for r in kept] == [5000, 7000]


def test_new_rows_only_passthrough_non_concat():
    rows = [{"prompt_idx": i, "prompt": f"p{i}"} for i in range(3)]
    assert g36.new_rows_only("sft11k", rows) is rows


def test_new_rows_only_fails_loud_on_empty():
    rows = [{"prompt_idx": i, "prompt": f"p{i}"} for i in range(4)]  # all < 5000
    with pytest.raises(AssertionError, match="no new prompts"):
        g36.new_rows_only("gsm8k_train_full", rows)


# ---------------------------------------------------------------------------
# G0' v2 gate (fixture-shaped bundle; leg (b) equality enforced at any n)
# ---------------------------------------------------------------------------
def _write_g0_bundle(out: Path, *, n: int = 60, layers: int = 2, dim: int = 8) -> None:
    """Tiny Qwen-S1 stand-in in the g0-fixture payload shape (n_tr > d so the
    primal route engages by default — the leg-(b) production regime)."""
    rng = np.random.default_rng(0)
    layer = min(int(cm.G0["layer"]), layers - 1)
    x = rng.normal(size=(n, dim)).astype(np.float64)
    w = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    y = (x @ w + 0.5 * rng.normal(size=(n, dim))).astype(np.float32)
    filler = np.random.default_rng(1)
    slots, profiles = [], []
    for i in range(n):
        s = filler.normal(size=(2, layers, dim)).astype(np.float32)
        p = filler.normal(size=(2, layers, dim)).astype(np.float32)
        s[0, layer, :] = x[i]  # G0 reads slot_index 0
        p[1, layer, :] = y[i]  # ... -> target_turn_index 1
        slots.append(torch.tensor(s))
        profiles.append(torch.tensor(p))
    payload = {
        "conv_ids": [f"g{i}" for i in range(n)],
        "slots": slots,
        "profiles": profiles,
        "nll": [torch.tensor([1.0, 1.0]) for _ in range(n)],
        "spans_meta": [{"conv_id": f"g{i}"} for i in range(n)],
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out / "instruct_chat_s_shard000.pt")
    (out / "instruct_chat_s_shard000.json").write_text(
        json.dumps({"stem": "instruct_chat_s", "n": n, "fixture": True})
    )


def test_run_g0v2_fixture_writes_gate_and_bars(tmp_path):
    bundle = tmp_path / "bundle"
    _write_g0_bundle(bundle)
    args = SimpleNamespace(g0_local_dir=bundle, g0_dl_dir=tmp_path / "dl", out_dir=tmp_path / "out")
    rc = f36.run_g0v2(args)
    assert rc == 0
    gate = json.loads((tmp_path / "out" / "gates_v2" / "g0v2.json").read_text())
    bars = json.loads((tmp_path / "out" / "gates_v2" / "v2_bars.json").read_text())
    # Leg (b) equality is ENFORCED and must hold (same math, two routes).
    assert gate["leg_b_gram_vs_primal"]["pass"] is True
    assert gate["leg_b_gram_vs_primal"]["enforced"] is True
    assert gate["leg_b_gram_vs_primal"]["abs_delta"] <= 1e-6
    assert gate["leg_b_gram_vs_primal"]["n_train_min"] > gate["leg_b_gram_vs_primal"]["d"]
    # Leg (a) is INFORMATIONAL on a fixture (production-n anchor tolerance).
    assert gate["leg_a_legacy"]["enforced"] is False
    assert gate["local_dir_fixture"] is True
    # Leg (c): bars ride cm.v2_bars off the primal v2-recipe read.
    s = gate["leg_c_v2_anchor"]["s_qwen_v2"]
    assert bars["s_qwen_v2"] == s
    expected = cm.v2_bars(s)
    assert bars["ex_v2"] == pytest.approx(expected["ex_v2"])
    assert bars["bar_v2"] == pytest.approx(expected["bar_v2"])
    # Module pins restored after the patched legs (try/finally contract).
    import issue825_fit_cells as fc

    assert fc.FORCE_GRAM is False
    assert fc.LEGACY_UNGUARDED_GCV is False
    assert fc.GCV_DOF_CAP == 0.9
    assert fc.N_INNER_LAMBDA_FOLDS == 4


# ---------------------------------------------------------------------------
# G1' v2 kill check (plan §7: KILL <=> BOTH raw AND recal below bar_v2)
# ---------------------------------------------------------------------------
def _seed_g1v2_inputs(out_dir: Path, *, raw: list[float], recal_best: float | None) -> None:
    (out_dir / "gates_v2").mkdir(parents=True, exist_ok=True)
    (out_dir / "cells_v2").mkdir(parents=True, exist_ok=True)
    (out_dir / "gates_v2" / "v2_bars.json").write_text(json.dumps(cm.v2_bars(0.6731)))
    cell = {"n": 10, "r2_per_layer_obs": raw}
    if recal_best is not None:
        cell["recal"] = {"s_recal": recal_best}
    cid = cm.v2_cell_id("rlvr", "chat", "lmsys23k")
    (out_dir / "cells_v2" / f"cells_{cid}.json").write_text(json.dumps(cell))


def test_g1v2_kill_when_both_below_bar(tmp_path):
    _seed_g1v2_inputs(tmp_path, raw=[0.01, 0.05], recal_best=0.02)  # bar_v2 = 0.2
    assert f36.run_g1v2_check(tmp_path) == 3
    gate = json.loads((tmp_path / "gates_v2" / "g1v2_gate.json").read_text())
    assert gate["verdict"] == "kill"


def test_g1v2_pass_when_recal_clears_bar(tmp_path):
    _seed_g1v2_inputs(tmp_path, raw=[0.01, 0.05], recal_best=0.25)
    assert f36.run_g1v2_check(tmp_path) == 0
    gate = json.loads((tmp_path / "gates_v2" / "g1v2_gate.json").read_text())
    assert gate["verdict"] == "pass"


def test_g1v2_nan_recal_never_manufactures_kill(tmp_path):
    _seed_g1v2_inputs(tmp_path, raw=[0.01, 0.05], recal_best=None)
    assert f36.run_g1v2_check(tmp_path) == 0


# ---------------------------------------------------------------------------
# Lane-robust provenance resolution (fellows job 17987: rsync scratch has no
# .git; a check=True git rev-parse crashed g2_parity AFTER the compare ran)
# ---------------------------------------------------------------------------
def test_resolve_code_sha_env_wins(monkeypatch):
    monkeypatch.setenv("EPS_GIT_SHA", "abc123launcherexported")
    assert cm.resolve_code_sha() == "abc123launcherexported"


def test_resolve_code_sha_gitless_dir_degrades(tmp_path, monkeypatch):
    """The git-absent branch: a non-git cwd resolves the literal, never raises."""
    monkeypatch.delenv("EPS_GIT_SHA", raising=False)
    assert cm.resolve_code_sha(repo_root=tmp_path) == "unknown-no-git"


def test_resolve_code_sha_real_git_matches_head(monkeypatch):
    """Git-ful lanes keep the real sha (GCP/RunPod behavior unchanged)."""
    import subprocess

    monkeypatch.delenv("EPS_GIT_SHA", raising=False)
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    got = cm.resolve_code_sha(repo_root=REPO)
    assert got == expected
    assert len(got) == 40


def test_dispatch_no_checked_git_revparse():
    """Fails pre-fix: no dispatcher heredoc may shell git rev-parse with check=True."""
    text = (REPO / "scripts" / "issue1336_dispatch.sh").read_text()
    banned = '["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True'
    assert banned not in text, "pod-side check=True git rev-parse crashes git-less rsync lanes"


def test_dispatch_upload_v2_git_leg_guarded():
    """The v2 result-commit leg must be fenced behind a git-availability probe."""
    text = (REPO / "scripts" / "issue1336_dispatch.sh").read_text()
    assert "[ -e .git ] && git rev-parse --git-dir" in text
    assert "no git checkout (rsync lane)" in text
