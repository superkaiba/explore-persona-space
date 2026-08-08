"""Boundary pins for the #1900 tfmargin-validation-expand driver (plan v13).

Covers: pool-freeze determinism at the registered rules (top-8/arm, >=50 floor,
3-draw keep, cross-pool donor distinctness, token budget skip-and-take-next,
the cas lowest-score_mean fallback fill), the donor-exclusion invariant on the
seeded context draws (no scored context in any pool's donor set + fail-loud on
a prompt-coverage miss), the two-sample drift-gate arithmetic on synthetic SEs,
and the smoke/descope context-slice arithmetic. All pure-CPU, tmp_path-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1900_tfm as tfm  # noqa: E402


class SplitTok:
    """Whitespace token-count stand-in at the tokenizer boundary (length filter)."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": text.split()}


def _judge_payload(rows: list[tuple[str, float | None, list[float]]]) -> dict:
    return {
        "rows": [
            {
                "sha": sha,
                "score_mean": mean,
                "binary_rate": 0.0,
                "n_kept_draws": len(draws),
                "kept_draw_scores": draws,
                "n_transport_lost": 0,
            }
            for sha, mean, draws in rows
        ]
    }


def _fixture_inputs():
    """Synthetic 2-arm family: 12 arm-scored rows each + a thin base zero set."""
    arm_ids = ["fam-a", "fam-b"]
    arm_payloads = {}
    raw_by_unit: dict[str, dict[str, dict]] = {u: {} for u in [*arm_ids, "base_content"]}
    prompt_by_sha: dict[str, str] = {}
    for arm in arm_ids:
        rows = []
        for i in range(12):
            sha = f"{arm}-sha{i:02d}"
            score = 90.0 - i  # descending; all >= 50
            draws = [score] * (3 if i != 11 else 2)  # last row: only 2 kept draws
            rows.append((sha, score, draws))
            prompt_by_sha[sha] = f"probe words {i}"
            raw_by_unit[arm][sha] = {"response_text": f"answer text {arm} {i}"}
        arm_payloads[arm] = _judge_payload(rows)
    # one over-budget positive candidate at the TOP of fam-a's order
    over = "fam-a-sha00"
    raw_by_unit["fam-a"][over] = {"response_text": " ".join(["w"] * 800)}
    base_rows = []
    for i in range(3):  # only 3 all-zero rows -> fallback fill must fire
        sha = f"base-zero{i:02d}"
        base_rows.append((sha, 0.0, [0.0, 0.0, 0.0]))
        prompt_by_sha[sha] = f"base probe {i}"
        raw_by_unit["base_content"][sha] = {"response_text": f"base answer {i}"}
    for i in range(12):  # low-but-nonzero fill candidates
        sha = f"base-low{i:02d}"
        base_rows.append((sha, 1.0 + i, [1.0 + i] * 3))
        prompt_by_sha[sha] = f"base probe low {i}"
        raw_by_unit["base_content"][sha] = {"response_text": f"base low answer {i}"}
    base_payload = _judge_payload(base_rows)
    return arm_ids, arm_payloads, base_payload, raw_by_unit, prompt_by_sha


def test_pool_freeze_deterministic_and_rules(monkeypatch):
    monkeypatch.setattr(tfm, "POS_PER_ARM", 4)
    monkeypatch.setattr(tfm, "POOL_SIDE", 8)
    arm_ids, arm_payloads, base_payload, raw_by_unit, prompt_by_sha = _fixture_inputs()
    build = lambda: tfm.build_family_pools(  # noqa: E731
        "fam", arm_ids, arm_payloads, base_payload, raw_by_unit, prompt_by_sha, SplitTok()
    )
    pos1, neg1, meta1 = build()
    pos2, neg2, _ = build()
    assert (pos1, neg1) == (pos2, neg2), "pool freeze must be byte-deterministic"
    assert tfm._pool_content_sha(pos1, neg1) == tfm._pool_content_sha(pos2, neg2)
    assert len(pos1) == 8 and len(neg1) == 8
    # top-of-order over-budget candidate skipped (skip-and-take-next)
    assert "fam-a-sha00" not in {p["sha"] for p in pos1}
    assert meta1["per_arm_depth"]["fam-a"]["skipped_over_budget"] == 1
    # 2-draw row never enters despite score >= 50
    assert "fam-a-sha11" not in {p["sha"] for p in pos1}
    # per-arm take follows (score desc, sha) after the skip
    fam_a = [p["sha"] for p in pos1 if p["source_arm"] == "fam-a"]
    assert fam_a == ["fam-a-sha01", "fam-a-sha02", "fam-a-sha03", "fam-a-sha04"]
    # donors distinct across the whole pool
    donors = [p["sha"] for p in pos1 + neg1]
    assert len(donors) == len(set(donors))
    # negatives: all 3 zero rows + lowest-score_mean fallback fill, recorded
    neg_shas = [p["sha"] for p in neg1]
    assert neg_shas[:3] == ["base-zero00", "base-zero01", "base-zero02"]
    assert neg_shas[3:] == [f"base-low{i:02d}" for i in range(5)]
    assert meta1["neg_fallback_fill"] == 5
    assert meta1["n_zero_rows_available"] == 3


def test_positive_pool_depth_fail_loud(monkeypatch):
    monkeypatch.setattr(tfm, "POS_PER_ARM", 20)  # deeper than the 12-row fixture
    monkeypatch.setattr(tfm, "POOL_SIDE", 8)
    arm_ids, arm_payloads, base_payload, raw_by_unit, prompt_by_sha = _fixture_inputs()
    with pytest.raises(AssertionError, match="positive pool depth"):
        tfm.build_family_pools(
            "fam", arm_ids, arm_payloads, base_payload, raw_by_unit, prompt_by_sha, SplitTok()
        )


def _write_judge(dirpath: Path, name: str, shas: list[str], value: float = 10.0) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    payload = _judge_payload([(s, value + i, [value + i] * 3) for i, s in enumerate(shas)])
    (dirpath / f"arm_scores_{name}.json").write_text(json.dumps(payload))


def test_donor_exclusion_invariant(tmp_path, monkeypatch):
    """No scored context appears in the pool donor set; coverage miss fail-louds."""
    fam, arm_ids = "cas", ["arm-x", "arm-y", "arm-z", "arm-w"]
    shas = [f"ctx{i:03d}" for i in range(40)]
    donors = ["ctx000", "ctx001"]
    cfg = tfm.Cfg(out_root=tmp_path / "out", stage_root=tmp_path / "stage")
    monkeypatch.setattr(tfm.Cfg, "parent_judge_dir", property(lambda self: tmp_path / "judge"))
    monkeypatch.setattr(tfm.Cfg, "offfloor_config_dir", property(lambda self: tmp_path / "offcfg"))
    for a in arm_ids:
        _write_judge(tmp_path / "judge", a, shas)
    _write_judge(tmp_path / "judge", f"base_{fam}", shas)
    (tmp_path / "offcfg").mkdir(parents=True, exist_ok=True)
    (tmp_path / "offcfg" / f"subset_{fam}.json").write_text(
        json.dumps({"n": 30, "shas": shas[5:35]})
    )
    prompt_by_sha = {s: f"prompt {s}" for s in shas}
    ctx = tfm.draw_contexts(cfg, fam, arm_ids, donors, prompt_by_sha)
    scored = set(ctx["S_parent"]) | set(ctx["S_offfloor"])
    assert not scored & set(donors), "donor contexts leaked into a scored context set"
    assert set(ctx["donors_excluded"]) == set(donors)
    # deterministic across calls (seeded draws)
    ctx2 = tfm.draw_contexts(cfg, fam, arm_ids, donors, prompt_by_sha)
    assert ctx["S_parent"] == ctx2["S_parent"] and ctx["S_offfloor"] == ctx2["S_offfloor"]
    # a prompt-coverage miss is a fail-loud join violation, never a silent drop
    missing = dict(prompt_by_sha)
    missing.pop("ctx010")
    with pytest.raises(AssertionError, match="missing from corpus_sample"):
        tfm.draw_contexts(cfg, fam, arm_ids, donors, missing)


def test_drift_gate_arithmetic():
    """flag iff |rho_new - (-0.064)| > 1.96*sqrt(SE_p^2 + SE_n^2) (plan section 6)."""
    se_p, se_n = 0.058, 0.035  # sqrt sum -> threshold ~0.1328
    flagged, thr = tfm.drift_flag(-0.05, -0.064, se_p, se_n)
    assert not flagged and abs(thr - 1.96 * np.hypot(se_p, se_n)) < 1e-12
    flagged, _ = tfm.drift_flag(0.12, -0.064, se_p, se_n)  # delta 0.184 > 0.133
    assert flagged
    # boundary: exactly at the threshold is NOT flagged (strict >)
    flagged, thr = tfm.drift_flag(-0.064 + 1.96 * np.hypot(se_p, se_n), -0.064, se_p, se_n)
    assert not flagged


def test_pass_contexts_smoke_and_descope_slices(tmp_path):
    """Smoke slices 12+12; a pilot descope re-derives the 500-prefix per family."""
    s_parent = [f"p{i:04d}" for i in range(800)]
    s_off = [f"o{i:04d}" for i in range(800)]
    payload = {"S_parent": s_parent, "S_offfloor": s_off}
    cfg = tfm.Cfg(out_root=tmp_path / "out", stage_root=tmp_path / "stage", smoke=True)
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    (cfg.config_dir / "contexts_imp.json").write_text(json.dumps(payload))
    got = tfm.pass_contexts(cfg, "imp")
    assert got == sorted(set(s_parent[:12]) | set(s_off[:12]))

    class _P(tfm.Cfg):
        @property
        def config_dir(self) -> Path:  # redirect the committed path into tmp
            return self.out_root / "cfgdir"

    cfg = _P(out_root=tmp_path / "out2", stage_root=tmp_path / "stage", smoke=False)
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    (cfg.config_dir / "contexts_imp.json").write_text(json.dumps(payload))
    assert tfm.pass_contexts(cfg, "imp") == sorted(set(s_parent) | set(s_off))
    cfg.tfm_dir.mkdir(parents=True, exist_ok=True)
    (cfg.tfm_dir / "pilot.json").write_text(json.dumps({"n_ctx_per_subset": {"imp": 500}}))
    got = tfm.pass_contexts(cfg, "imp")
    assert got == sorted(set(s_parent[:500]) | set(s_off[:500]))
