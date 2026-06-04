# ruff: noqa: RUF002, RUF003, E501
"""Round-2 fix tests for #489. Cover the blockers + a few minor stats helpers.

Run on CPU only. The tests below DO NOT load Qwen-2.5-7B; they exercise the
Phase 5 stats helpers + Phase 4 prompt-construction primitive + the
contexts module's invariants.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────────────────────────────────────────────────────
# B5 + M-e: contexts module invariants
# ────────────────────────────────────────────────────────────────────────


def test_b5_all_24_scaffold_texts_pairwise_distinct():
    """Every union context must have a distinct scaffold_text (B5 fix)."""
    from explore_persona_space.experiments.i489_contexts import (
        UNION_CONTEXTS,
        scaffold_text,
    )

    texts = [scaffold_text(c) for c in UNION_CONTEXTS]
    assert len(set(texts)) == len(texts), (
        f"duplicate scaffold_text across union contexts: "
        f"{len(texts) - len(set(texts))} duplicate(s)"
    )


def test_b5_ik11_distinct_from_ik01():
    """IK11 (concise-engineer) must differ from IK01 (neutral) byte-for-byte."""
    from explore_persona_space.experiments.i489_contexts import (
        UNION_BY_CID,
        scaffold_text,
    )

    assert scaffold_text(UNION_BY_CID["IK01"]) != scaffold_text(UNION_BY_CID["IK11"])


def test_me_persona_indicator_fires_for_matched_pairs():
    """M-e fix: persona_indicator must be 1 for matched pairs (was always 0)."""
    from explore_persona_space.experiments.i489_contexts import (
        MATCHED_PAIRS,
        UNION_BY_CID,
        scaffold_overlap_score,
    )

    fired = []
    for icl_cid, sp_cid in MATCHED_PAIRS:
        s = scaffold_overlap_score(UNION_BY_CID[icl_cid], UNION_BY_CID[sp_cid])
        fired.append((icl_cid, sp_cid, s["persona_indicator"]))
    assert all(f[2] == 1 for f in fired), (
        f"M-e fix: persona_indicator should fire for all matched pairs; got {fired}"
    )


def test_me_persona_indicator_zero_for_random_cross_pairs():
    """M-e: persona_indicator should be 0 for unrelated cross-type pairs."""
    from explore_persona_space.experiments.i489_contexts import (
        UNION_BY_CID,
        scaffold_overlap_score,
    )

    # IK01 (neutral) x SP02 (software engineer) — neither shares a persona word.
    s = scaffold_overlap_score(UNION_BY_CID["IK01"], UNION_BY_CID["SP02"])
    assert s["persona_indicator"] == 0


# ────────────────────────────────────────────────────────────────────────
# B7 + B1: phase 4 marker probe construction
# ────────────────────────────────────────────────────────────────────────


def test_b7_marker_probe_full_ids_ends_in_marker():
    """_build_marker_probe_full_ids must place MARKER_ID at the last position."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i406_conditions import MARKER_ID, MARKER_TEXT

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from i489_phase4_eval_onpolicy import _build_marker_probe_full_ids
    finally:
        sys.path.pop(0)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    assert tok.encode(MARKER_TEXT, add_special_tokens=False) == [MARKER_ID]
    prompt = "<|im_start|>system\nyou are helpful.<|im_end|>\n<|im_start|>user\nq\n<|im_end|>\n<|im_start|>assistant\n"
    R_text = "An on-policy response."
    full_ids, plen, slot = _build_marker_probe_full_ids(tok, prompt, R_text)
    assert full_ids[-1] == MARKER_ID
    assert full_ids.count(MARKER_ID) == 1
    assert slot == len(full_ids) - 1
    assert plen == len(tok.encode(prompt, add_special_tokens=False))


# ────────────────────────────────────────────────────────────────────────
# B1: Phase 5 raises on empty delta_g panel
# ────────────────────────────────────────────────────────────────────────


def test_b1_phase5_raises_on_empty_delta_g(tmp_path, monkeypatch):
    """Phase 5 must RAISE if zero off-diagonal cells carry delta_g.

    Simulate the round-1 false-positive: write per-cell payloads with
    'phase4b_pending': True and NO delta_g key. Phase 5 must refuse to
    silently ship an empty H1/H2/H3/H4 dict.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    # Set up fake phase1 + phase4 directories under tmp_path.
    phase1 = tmp_path / "eval_results" / "issue_489" / "phase1"
    phase4 = tmp_path / "eval_results" / "issue_489" / "phase4" / "per_cell"
    out = tmp_path / "eval_results" / "issue_489" / "phase5"
    phase1.mkdir(parents=True)
    phase4.mkdir(parents=True)
    monkeypatch.setattr(p5, "PHASE1_DIR", phase1)
    monkeypatch.setattr(p5, "PHASE4_DIR", phase4)
    monkeypatch.setattr(p5, "OUT_DIR", out)

    # cosine_per_layer.json (only needed key: cos_sim_per_layer[L])
    (phase1 / "cosine_per_layer.json").write_text(
        json.dumps(
            {
                "cos_sim_per_layer": {
                    "21": {f"IK0{i}": {f"IK0{j}": 0.5 for j in range(1, 4)} for i in range(1, 4)}
                }
            }
        )
    )

    # Per-cell payloads WITHOUT delta_g (the round-1 phase4b_pending bug).
    for i in range(1, 3):
        for j in range(1, 3):
            if i == j:
                continue
            (phase4 / f"G_IK0{i}__IK0{j}_frac0.50.json").write_text(
                json.dumps(
                    {
                        "T_i": f"IK0{i}",
                        "T_j": f"IK0{j}",
                        "frac": 0.50,
                        "seed": 42,
                        "phase4b_pending": True,
                    }
                )
            )

    with pytest.raises(RuntimeError, match=r"zero off-diagonal cells.*delta_g"):
        p5.main(["--fracs", "0.50", "--bootstrap-n", "10"])


def test_b1_phase5_runs_when_delta_g_present(tmp_path, monkeypatch):
    """Phase 5 happy-path: with delta_g present, the H1 block has rho_cos."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    phase1 = tmp_path / "eval_results" / "issue_489" / "phase1"
    phase4 = tmp_path / "eval_results" / "issue_489" / "phase4" / "per_cell"
    out = tmp_path / "eval_results" / "issue_489" / "phase5"
    phase1.mkdir(parents=True)
    phase4.mkdir(parents=True)
    monkeypatch.setattr(p5, "PHASE1_DIR", phase1)
    monkeypatch.setattr(p5, "PHASE4_DIR", phase4)
    monkeypatch.setattr(p5, "OUT_DIR", out)

    # Cosine matrix over 4 cids (need ≥3 sources × ≥3 targets for bootstrap).
    cids = ["IK01", "IK02", "SP01", "SP02"]
    cos_mat = {
        ci: {cj: 1.0 - 0.1 * abs(k - kk) for kk, cj in enumerate(cids)} for k, ci in enumerate(cids)
    }
    (phase1 / "cosine_per_layer.json").write_text(
        json.dumps({"cos_sim_per_layer": {"21": cos_mat}})
    )

    # Real delta_g + lengths.
    rng = np.random.default_rng(0)
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i == j:
                continue
            d = abs(i - j) * 0.1
            (phase4 / f"G_{ci}__{cj}_frac0.50.json").write_text(
                json.dumps(
                    {
                        "T_i": ci,
                        "T_j": cj,
                        "frac": 0.50,
                        "seed": 42,
                        "delta_g": float(-2.0 * d + rng.normal(0, 0.1)),
                        "g_logprob_mean": -1.0,
                        "b_logprob_mean": -3.0,
                        "n_q": 20,
                        "n_samples": 8,
                        "prompt_lens_per_q": [50] * 20,
                        "R_lens_per_q_sample": [[100] * 8] * 20,
                    }
                )
            )

    rc = p5.main(["--fracs", "0.50", "--bootstrap-n", "50"])
    assert rc == 0
    payload = json.loads((out / "analysis.json").read_text())
    assert payload["total_off_cells_with_delta_g"] > 0
    assert "0.5" in payload["h1"] or 0.5 in payload["h1"]


# ────────────────────────────────────────────────────────────────────────
# M-g: stats helpers guard NaN / constant covariates
# ────────────────────────────────────────────────────────────────────────


def test_mg_spearman_partial_handles_constant_covariate():
    """M-g: a constant covariate column should be dropped, not crash."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    rng = np.random.default_rng(7)
    x = rng.normal(size=20)
    y = -x + 0.1 * rng.normal(size=20)
    z_constant = np.ones(20)  # would crash old impl with NaN at correlation
    rho = p5._spearman_partial(x, y, z_constant)
    assert rho == rho  # not NaN
    assert rho < 0  # signal preserved despite degenerate covariate


def test_mg_spearman_partial_handles_all_nan_y():
    """M-g: constant y → NaN return, not crash."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    x = np.arange(20, dtype=float)
    y = np.zeros(20)
    rho = p5._spearman_partial(x, y, None)
    assert rho != rho  # NaN


# ────────────────────────────────────────────────────────────────────────
# B4: H3 paired bootstrap mechanic + ESS fallback
# ────────────────────────────────────────────────────────────────────────


def test_b4_h3_paired_mechanic_when_ess_above_floor():
    """B4: ESS >= ESS_FLOOR (24) → mechanic == 'paired'."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    # Build 30 shared (cid_i, frac) units (above floor of 24).
    icl_cells = []
    sp_cells = []
    for k in range(30):
        cid = f"IK{k:02d}"
        frac = 0.5
        icl_cells.append({"T_i": cid, "T_j": "IK99", "frac": frac, "delta_g": -1.0 + 0.01 * k})
        sp_cells.append({"T_i": cid, "T_j": "SP99", "frac": frac, "delta_g": -0.5 + 0.01 * k})

    def cos_dist(_a, _b):
        return 0.5

    def length(_c):
        return 5.0

    rng = np.random.default_rng(1)
    result = p5._h3_paired_bootstrap(icl_cells, sp_cells, cos_dist, length, n_boots=20, rng=rng)
    assert result["mechanic"] == "paired"
    assert result["ess_lora_snapshots"] == 30


def test_b4_h3_independent_fallback_when_ess_below_floor():
    """B4: ESS < ESS_FLOOR (24) → mechanic == 'independent_fallback'.

    Shared units = cid_i × frac pairs that appear in BOTH arms. We use the
    same source cid_i values in both arms (each source contributes cells to
    BOTH ICL and SP targets — that's the whole point of paired evaluation),
    but only 5 of them — well below the 24 ESS floor.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    # 5 shared (cid_i, frac) units — same sources in both arms.
    icl_cells = [
        {"T_i": f"IK{k:02d}", "T_j": "IK99", "frac": 0.5, "delta_g": 0.0} for k in range(5)
    ]
    sp_cells = [{"T_i": f"IK{k:02d}", "T_j": "SP99", "frac": 0.5, "delta_g": 0.0} for k in range(5)]

    def cos_dist(_a, _b):
        return 0.5

    def length(_c):
        return 5.0

    rng = np.random.default_rng(1)
    result = p5._h3_paired_bootstrap(icl_cells, sp_cells, cos_dist, length, n_boots=20, rng=rng)
    assert result["mechanic"] == "independent_fallback"
    assert result["ess_lora_snapshots"] == 5
    assert "fell back" in result["note"]


# ────────────────────────────────────────────────────────────────────────
# M-d: don't mutate args.fracs
# ────────────────────────────────────────────────────────────────────────


def test_md_load_cells_does_not_mutate_request(tmp_path, monkeypatch):
    """M-d: _load_cells must not mutate its fracs_request argument."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    phase4 = tmp_path / "phase4"
    phase4.mkdir()
    monkeypatch.setattr(p5, "PHASE4_DIR", phase4)
    # Write a cell at an unexpected frac (0.75) — the old code extended the
    # caller's list mid-iteration; the new contract returns it separately.
    (phase4 / "G_IK01__IK02_frac0.75.json").write_text(
        json.dumps({"T_i": "IK01", "T_j": "IK02", "frac": 0.75, "seed": 42, "delta_g": -1.0})
    )
    requested = [0.25, 0.50, 1.00]
    snapshot_before = list(requested)
    _, present = p5._load_cells(requested, seed=42, allow_smoke=False)
    assert requested == snapshot_before  # NOT mutated
    assert 0.75 in present


# ────────────────────────────────────────────────────────────────────────
# B2 + B3: smoke gate + frozen-overrides hook
# ────────────────────────────────────────────────────────────────────────


def test_b3_frozen_overrides_hook_consumed(tmp_path, monkeypatch):
    """B3: writing frozen_sp_strings.json must change UNION_BY_CID on next import."""
    overrides_path = tmp_path / "frozen.json"
    overrides_path.write_text(json.dumps({"SP03": "OVERRIDDEN pirate captain SP03 system prompt"}))
    # Use a subprocess so the import-time hook runs against our override path.
    check_src = (
        f"import os; os.environ['I489_FROZEN_SP_OVERRIDES'] = {str(overrides_path)!r};"
        "from explore_persona_space.experiments.i489_contexts import UNION_BY_CID, N_FROZEN_OVERRIDES;"
        "print(N_FROZEN_OVERRIDES);"
        "print(UNION_BY_CID['SP03'].system_prompt)"
    )
    out = subprocess.check_output([sys.executable, "-c", check_src], cwd=str(REPO_ROOT))
    lines = out.decode().strip().splitlines()
    assert lines[0] == "1"
    assert lines[1] == "OVERRIDDEN pirate captain SP03 system prompt"
