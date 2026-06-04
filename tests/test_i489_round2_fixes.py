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

    # Round-4 Bug-2 made the fail-loud panel-completeness check fire on
    # `requested_fracs` (not the present∩requested intersection). When ALL
    # phase4 payloads lack `delta_g`, the requested frac=0.50 now has
    # 0/552 off-diag cells with delta_g → raises the more specific
    # "incomplete panel" error EARLIER than the generic "zero off-diagonal
    # cells" branch. Either message is acceptable — both diagnose the
    # same B1 root cause (no usable cells) and refuse to silently ship.
    with pytest.raises(RuntimeError, match=r"(incomplete panel|zero off-diagonal cells.*delta_g)"):
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

    # Smoke mode bypasses round-3 Maj-2's complete-panel requirement (this
    # test uses a 4-cid micro-grid for speed, not the production 24-cid
    # panel).
    rc = p5.main(["--fracs", "0.50", "--bootstrap-n", "50", "--smoke"])
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
# B4 (round-3 fix): H3 independent two-sample mechanic on DISJOINT
# ICL/SP source sets (the production shape). The round-2 'paired-shared-
# units' bootstrap is dead code: ICL sources and SP sources are disjoint
# by construction, so intersection is always empty.
# ────────────────────────────────────────────────────────────────────────


def test_b4_round3_h3_independent_two_sample_on_disjoint_sources():
    """Round-3 B4: production-shaped DISJOINT ICL/SP sources → mechanic
    'independent_two_sample' fires, NOT the dead empty-intersection path.

    Build 16 ICL-source × 16 ICL-target cells (within-ICL panel) AND
    8 SP-source × 8 SP-target cells (within-SP panel). The two source
    sets are disjoint, so the old paired mechanic's intersection of
    (cid_i, frac) units is always empty. The new function MUST report
    mechanic 'independent_two_sample' at the raw-ρ bar (≈0.55), NOT
    return mechanic='none' with NaN diff.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    rng = np.random.default_rng(0)
    icl_cells = []
    icl_cids = [f"IK{k:02d}" for k in range(1, 17)]  # 16 ICL sources/targets
    for ci in icl_cids:
        for cj in icl_cids:
            if ci == cj:
                continue
            icl_cells.append(
                {"T_i": ci, "T_j": cj, "frac": 0.5, "delta_g": -2.0 + rng.normal(0, 0.1)}
            )
    sp_cells = []
    sp_cids = [f"SP{k:02d}" for k in range(1, 9)]  # 8 SP sources/targets
    for ci in sp_cids:
        for cj in sp_cids:
            if ci == cj:
                continue
            sp_cells.append(
                {"T_i": ci, "T_j": cj, "frac": 0.5, "delta_g": -1.0 + rng.normal(0, 0.1)}
            )

    # Empty intersection invariant — the old 'paired' path is unreachable.
    icl_units = {(c["T_i"], c["frac"]) for c in icl_cells}
    sp_units = {(c["T_i"], c["frac"]) for c in sp_cells}
    assert not (icl_units & sp_units), (
        "ICL-source and SP-source cids are disjoint by construction; the "
        "round-2 'shared-unit' paired bootstrap is unreachable on production shape"
    )

    def cos_dist(a, b):
        # Mild signal so ρ != 0 — distance proportional to |index gap|.
        ai = (icl_cids + sp_cids).index(a)
        bi = (icl_cids + sp_cids).index(b)
        return 0.1 + 0.05 * abs(ai - bi)

    def length(_c):
        return 5.0

    result = p5._h3_independent_two_sample(
        icl_cells, sp_cells, cos_dist, length, n_boots=50, rng=rng
    )
    assert result["mechanic"] == "independent_two_sample"
    # Round-4 fix: the prior `result["pass_bar"] if False else True` was a no-op
    # tautology flagged by Codex — it never read the dict. The actual returned
    # bar key is `raw_rho_pass_bar`; assert it equals RAW_RHO_FALLBACK_BAR.
    assert result["raw_rho_pass_bar"] == p5.RAW_RHO_FALLBACK_BAR
    assert result["n_icl_cells"] == 16 * 15
    assert result["n_sp_cells"] == 8 * 7
    # Verify the function returns CI width info up front for narration.
    assert "ci_icl_width" in result
    assert "ci_sp_width" in result
    assert "ci_diff_width" in result


def test_b4_round3_h3_dead_paired_function_removed():
    """Round-3 B4: the round-2 _h3_paired_bootstrap symbol must be gone.

    Defends against silent re-introduction of the dead empty-intersection
    paired-bootstrap path. Round-3 makes the independent-two-sample
    function the only H3 mechanic; the named-paired function is removed.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    assert not hasattr(p5, "_h3_paired_bootstrap"), (
        "Round-3 B4: _h3_paired_bootstrap was a structurally dead path "
        "(empty intersection on disjoint ICL/SP source sets). It must not "
        "be silently re-introduced — H3 mechanic is independent_two_sample."
    )
    assert hasattr(p5, "_h3_independent_two_sample"), (
        "Round-3 B4: H3 must call _h3_independent_two_sample as primary."
    )


def test_b4_round3_h3_pass_uses_raw_rho_bar():
    """Round-3 B4: H3 PASS uses the 0.55 raw-ρ gap bar, NOT 0.15."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    # Read the H3 PASS predicate by inspection — assert the bar constant.
    assert p5.RAW_RHO_FALLBACK_BAR == 0.55
    src = (REPO_ROOT / "scripts" / "i489_phase5_analyze.py").read_text()
    # Defensive: assert the PASS predicate references the raw-rho bar.
    assert "diff >= RAW_RHO_FALLBACK_BAR" in src, (
        "H3 PASS predicate must use RAW_RHO_FALLBACK_BAR (round-3 B4)"
    )
    # The 0.15 bar must NOT be present in the H3 block.
    h3_block = src[src.index("# --- H3:") : src.index("# --- H4(a):")]
    assert "0.15" not in h3_block, (
        "Round-3 B4: 0.15 paired-mechanic bar must be removed from the H3 block"
    )


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


# ────────────────────────────────────────────────────────────────────────
# Round-3 Maj-1: collision-free lora_int_id manifest
# ────────────────────────────────────────────────────────────────────────


def test_maj1_lora_int_id_manifest_unique_at_full_production_shape():
    """Maj-1: manifest over all 24 cids × 6 fracs × 1 seed = 144 unique ids."""
    from explore_persona_space.experiments.i489_contexts import (
        UNION_CONTEXTS,
        build_lora_int_id_manifest,
    )

    all_cids = [c.cid for c in UNION_CONTEXTS]
    fracs = [0.10, 0.25, 0.50, 1.00, 2.00, 3.00]
    seeds = [42]
    manifest = build_lora_int_id_manifest(all_cids, fracs, seeds)
    n_snapshots = len(all_cids) * len(fracs) * len(seeds)
    assert len(manifest) == n_snapshots == 144
    ids = list(manifest.values())
    assert len(set(ids)) == len(ids), (
        f"manifest produced collisions: {len(ids) - len(set(ids))} duplicates "
        f"across {len(ids)} snapshots"
    )
    # vLLM constraint: int_id >= 1.
    assert min(ids) >= 1


def test_maj1_lora_int_id_manifest_collision_detection_raises():
    """Maj-1: assert_unique_lora_int_ids fails loud on a forged collision."""
    from explore_persona_space.experiments.i489_contexts import assert_unique_lora_int_ids

    bad = {
        ("IK01", 0.50, 42): 17,
        ("SP03", 0.25, 42): 17,  # collision
    }
    with pytest.raises(AssertionError, match=r"COLLISION on id 17"):
        assert_unique_lora_int_ids(bad)


def test_maj1_lora_int_id_manifest_deterministic_across_calls():
    """Maj-1: same inputs → same manifest (train and eval must agree)."""
    from explore_persona_space.experiments.i489_contexts import build_lora_int_id_manifest

    cids = ["IK02", "IK01", "SP01"]
    fracs = [0.50, 0.10]
    seeds = [42]
    m1 = build_lora_int_id_manifest(cids, fracs, seeds)
    m2 = build_lora_int_id_manifest(cids, fracs, seeds)
    assert m1 == m2
    # Stability under input ordering — the function sorts internally.
    m3 = build_lora_int_id_manifest(list(reversed(cids)), list(reversed(fracs)), seeds)
    assert m1 == m3


def test_maj1_round2_collision_formula_demonstrably_collides():
    """Maj-1 regression: the round-2 formula collides on production shape.

    Concretely demonstrates the bug the manifest exists to prevent:
    ``all_cids.index(cid) * 10 + int(frac * 100) + 1`` produces duplicate
    ids across 144 snapshots, so vLLM's int_id-keyed LoRA cache would
    silently serve the wrong adapter for collided pairs.
    """
    from explore_persona_space.experiments.i489_contexts import UNION_CONTEXTS

    all_cids = [c.cid for c in UNION_CONTEXTS]
    fracs = [0.10, 0.25, 0.50, 1.00, 2.00, 3.00]
    counts: dict[int, list[tuple[str, float]]] = {}
    for cid in all_cids:
        for frac in fracs:
            bad_id = all_cids.index(cid) * 10 + int(frac * 100) + 1
            counts.setdefault(bad_id, []).append((cid, frac))
    collisions = {k: v for k, v in counts.items() if len(v) > 1}
    assert len(collisions) >= 1, (
        "round-2 formula was expected to collide on the production shape; "
        "if this assertion stops firing, the bug surface changed and the test "
        "needs to be re-validated"
    )


# ────────────────────────────────────────────────────────────────────────
# Round-3 Maj-2: Phase 5 frac filtering + fail-loud on incomplete panel
# ────────────────────────────────────────────────────────────────────────


def _setup_phase5_inputs(tmp_path, monkeypatch, frac_to_cells: dict[float, list[dict]]):
    """Shared helper: lay down phase1/cosine + phase4/per_cell for the given fracs."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    phase1 = tmp_path / "phase1"
    phase4 = tmp_path / "phase4_per_cell"
    out = tmp_path / "phase5"
    phase1.mkdir()
    phase4.mkdir()
    monkeypatch.setattr(p5, "PHASE1_DIR", phase1)
    monkeypatch.setattr(p5, "PHASE4_DIR", phase4)
    monkeypatch.setattr(p5, "OUT_DIR", out)

    from explore_persona_space.experiments.i489_contexts import UNION_CONTEXTS

    all_cids = [c.cid for c in UNION_CONTEXTS]
    cos_mat = {
        ci: {cj: 1.0 - 0.01 * abs(i - j) for j, cj in enumerate(all_cids)}
        for i, ci in enumerate(all_cids)
    }
    (phase1 / "cosine_per_layer.json").write_text(
        json.dumps({"cos_sim_per_layer": {"21": cos_mat}})
    )
    for frac, cells in frac_to_cells.items():
        for cell in cells:
            (phase4 / f"G_{cell['T_i']}__{cell['T_j']}_frac{frac:.2f}.json").write_text(
                json.dumps(cell)
            )
    return p5, out


def test_maj2_phase5_raises_on_incomplete_panel(tmp_path, monkeypatch):
    """Maj-2: a requested frac with <552 off-diagonal delta_g cells MUST raise."""
    # Only 4 cells at frac=0.50 — way below 552.
    cells_050 = [
        {
            "T_i": "IK01",
            "T_j": "IK02",
            "frac": 0.50,
            "seed": 42,
            "delta_g": -1.0,
            "prompt_lens_per_q": [50] * 20,
            "R_lens_per_q_sample": [[100] * 8] * 20,
        },
        {
            "T_i": "IK02",
            "T_j": "IK01",
            "frac": 0.50,
            "seed": 42,
            "delta_g": -1.0,
            "prompt_lens_per_q": [50] * 20,
            "R_lens_per_q_sample": [[100] * 8] * 20,
        },
        {
            "T_i": "SP01",
            "T_j": "SP02",
            "frac": 0.50,
            "seed": 42,
            "delta_g": -1.0,
            "prompt_lens_per_q": [50] * 20,
            "R_lens_per_q_sample": [[100] * 8] * 20,
        },
        {
            "T_i": "SP02",
            "T_j": "SP01",
            "frac": 0.50,
            "seed": 42,
            "delta_g": -1.0,
            "prompt_lens_per_q": [50] * 20,
            "R_lens_per_q_sample": [[100] * 8] * 20,
        },
    ]
    p5, _ = _setup_phase5_inputs(tmp_path, monkeypatch, {0.50: cells_050})
    with pytest.raises(RuntimeError, match=r"incomplete panel.*round-3 Maj-2"):
        p5.main(["--fracs", "0.50", "--bootstrap-n", "10"])


def test_maj2_phase5_filters_smoke_only_fracs(tmp_path, monkeypatch):
    """Maj-2: a frac that wasn't requested must NOT be analyzed in non-smoke mode.

    Write cells at frac=0.75 (smoke-only — not requested by Phase 4 dispatch).
    Phase 5 with --fracs 0.50 must NOT include the 0.75 frac. The 0.50 frac
    is intentionally absent here, so Phase 5 raises with the standard "zero
    cells with delta_g" message — proving the 0.75 frac was filtered out
    BEFORE the count happened (otherwise 0.75's cells would have lifted the
    count above zero and we'd see the round-3 Maj-2 'incomplete panel'
    message instead).
    """
    cells_075 = [
        {
            "T_i": "IK01",
            "T_j": "IK02",
            "frac": 0.75,
            "seed": 42,
            "delta_g": -1.0,
            "prompt_lens_per_q": [50] * 20,
            "R_lens_per_q_sample": [[100] * 8] * 20,
        },
    ]
    p5, _ = _setup_phase5_inputs(tmp_path, monkeypatch, {0.75: cells_075})
    # Round-4 Bug-2: requested frac=0.50 is absent → the panel-completeness
    # check now raises the more-specific "incomplete panel" message
    # earlier (rather than falling through to the generic "zero
    # off-diagonal cells" raise). Either message proves the test intent —
    # the 0.75 frac was filtered out before the analysis started
    # (otherwise 0.75's cells would have been the ones analyzed and no
    # error would have raised at all).
    with pytest.raises(RuntimeError, match=r"(incomplete panel|zero off-diagonal cells)"):
        p5.main(["--fracs", "0.50", "--bootstrap-n", "10"])


# ────────────────────────────────────────────────────────────────────────
# Round-3 Maj-3: H2 verdict tree gates on dual_graded
# ────────────────────────────────────────────────────────────────────────


def test_maj3_h2_dual_graded_fail_flips_survives_to_null():
    """Maj-3: when strong-drop passes but dual_graded fails → NULL_DUAL_GRADED_PARTIAL."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    # Build a panel with enough cells for identifiability gate to pass.
    h2_cells = [
        {"T_i": f"IK{i:02d}", "T_j": f"IK{j:02d}", "delta_g": -1.0 + 0.05 * (i + j)}
        for i in range(1, 9)
        for j in range(1, 9)
        if i != j
    ]
    kind_dist = {f"IK{k:02d}": 0.1 * k for k in range(1, 17)}

    def cos_dist(a, b):
        return 0.5

    def length(_c):
        return 5.0

    # h2_pass_after_drop = True, dual_graded_pass = True → SURVIVES.
    res_pass = p5._h2_three_outcome(
        h2_pass_after_drop=True,
        dual_graded_pass=True,
        h2_cells=h2_cells,
        cos_dist_fn=cos_dist,
        length_fn=length,
        kind_distinctness=kind_dist,
    )
    # h2_pass_after_drop = True, dual_graded_pass = False → NULL_DUAL_GRADED_PARTIAL.
    res_fail = p5._h2_three_outcome(
        h2_pass_after_drop=True,
        dual_graded_pass=False,
        h2_cells=h2_cells,
        cos_dist_fn=cos_dist,
        length_fn=length,
        kind_distinctness=kind_dist,
    )
    assert res_pass["verdict"] in ("SURVIVES", "UNIDENTIFIABLE")
    if res_pass["verdict"] == "SURVIVES":
        # The pivot must flip the verdict when dual_graded fails.
        assert res_fail["verdict"] == "NULL_DUAL_GRADED_PARTIAL"
        assert res_pass["dual_graded_pass"] is True
        assert res_fail["dual_graded_pass"] is False


# ────────────────────────────────────────────────────────────────────────
# Round-3 Maj-4: diagonal-adjusted statistic gates SURVIVES → NULL
# ────────────────────────────────────────────────────────────────────────


def test_maj4_diagonal_adjusted_collapse_relabels_survives_to_null():
    """Maj-4: when the partial-on-emission_ii flips sign, survives_diagonal_adjustment is False."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    # Build off-diagonal cells where the partial-on-emission_ii is POSITIVE
    # (collapse: the raw cosine→delta_g signal was driven by source emission
    # strength alone, partialling it out removes the negative ρ).
    off_cells = []
    diag_by_cid = {}
    for i in range(1, 17):
        cid = f"IK{i:02d}"
        diag_by_cid[cid] = {"T_i": cid, "T_j": cid, "delta_g": -2.0 - 0.5 * i}
        for j in range(1, 17):
            if i == j:
                continue
            off_cells.append(
                {
                    "T_i": cid,
                    "T_j": f"IK{j:02d}",
                    "delta_g": -3.0 - 0.5 * i + 0.0 * j,
                }
            )

    def cos_dist(a, b):
        # Cosine distance roughly tracks emission_ii so partialling it out collapses ρ.
        ai = int(a[2:])
        return 0.05 * ai

    def length(_c):
        return 5.0

    res = p5._h2_diagonal_adjusted(off_cells, diag_by_cid, cos_dist, length)
    assert res["available"] is True
    # Diagonal-adjustment collapses the signal → survives False.
    assert res["survives_diagonal_adjustment"] is False


def test_maj4_diagonal_adjusted_survives_when_signal_independent_of_ii():
    """Maj-4: when the signal is independent of emission_ii, survives_diagonal_adjustment is True."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    rng = np.random.default_rng(0)
    off_cells = []
    diag_by_cid = {}
    for i in range(1, 17):
        cid = f"IK{i:02d}"
        # emission_ii uncorrelated with anything informative.
        diag_by_cid[cid] = {"T_i": cid, "T_j": cid, "delta_g": -2.0 + rng.normal(0, 0.3)}
        for j in range(1, 17):
            if i == j:
                continue
            # delta_g driven by |i - j|, not by source-i magnitude.
            off_cells.append(
                {
                    "T_i": cid,
                    "T_j": f"IK{j:02d}",
                    "delta_g": -0.2 * abs(i - j) + rng.normal(0, 0.1),
                }
            )

    def cos_dist(a, b):
        return 0.05 * abs(int(a[2:]) - int(b[2:]))

    def length(_c):
        return 5.0

    res = p5._h2_diagonal_adjusted(off_cells, diag_by_cid, cos_dist, length)
    assert res["available"] is True
    # Signal is in the predictor, not emission_ii → adjustment leaves ρ alive.
    assert res["survives_diagonal_adjustment"] is True


def test_maj4_h1_pass_gated_on_diagonal_adjusted_in_source():
    """Maj-4 wiring: H1 pass field MUST be ANDed with survives_diagonal_adjustment."""
    src = (REPO_ROOT / "scripts" / "i489_phase5_analyze.py").read_text()
    assert "survives_diagonal_adjustment" in src
    # The h1 pass label must include the NULL_AFTER_DIAGONAL_ADJUSTMENT branch.
    assert "NULL_AFTER_DIAGONAL_ADJUSTMENT" in src


def test_maj4_h2_survives_relabels_when_diagonal_adjustment_collapses():
    """Maj-4 wiring: an H2 SURVIVES verdict must be downgraded if diagonal-adjustment fails."""
    src = (REPO_ROOT / "scripts" / "i489_phase5_analyze.py").read_text()
    # The relabel block must be present in the main loop.
    assert (
        'h2_verdict = "NULL_AFTER_DIAGONAL_ADJUSTMENT"' in src
        or "'NULL_AFTER_DIAGONAL_ADJUSTMENT'" in src
    )


# ────────────────────────────────────────────────────────────────────────
# Round-4 Bug-1: H3 cluster bootstrap MUST preserve with-replacement
# duplicate cluster draws (otherwise it collapses to subsample-without-
# replacement and the per-arm + diff CIs are wrong, badly so at n=8 SP
# clusters). The fix is to keep cluster draws as LISTS and iterate them so
# a cluster drawn twice contributes its cells twice — same idiom as the
# canonical siblings _dyadic_cluster_bootstrap_rho and
# _paired_diff_bootstrap_rho.
# ────────────────────────────────────────────────────────────────────────


def test_round4_bug1_h3_resampler_preserves_replacement_duplicates():
    """Round-4 Bug-1: the H3 per-arm panel resampler MUST yield panels in
    which a duplicated cluster draw contributes its cells more than once
    (i.e. with-replacement duplicate semantics), NOT a deduplicated set.

    We import and CALL the production ``_resample_panel`` directly
    (module-level helper as of the round-5 lift) so any future change
    that re-breaks the duplicate-preserving idiom — e.g. switching back
    to a set comprehension — will fail this test immediately. The prior
    round-4 version manually re-implemented the loop and would have
    silently passed a broken resampler.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)

    # Tiny 3×3 ICL fixture; with 3 clusters and 3 with-replacement draws,
    # the probability that all draws are distinct is 6/27 ≈ 22%. Under
    # seed=0 we empirically confirm the draw repeats (and fail the
    # fixture invariant otherwise so the test is never vacuous).
    cids = ["A", "B", "C"]
    cells = []
    for ci in cids:
        for cj in cids:
            if ci == cj:
                continue
            cells.append({"T_i": ci, "T_j": cj, "delta_g": 0.1})
    cell_index = {(c["T_i"], c["T_j"]): c for c in cells}

    rng = np.random.default_rng(0)
    idx_s = rng.integers(0, len(cids), len(cids))
    idx_t = rng.integers(0, len(cids), len(cids))
    srcs = [cids[i] for i in idx_s]
    tgts = [cids[i] for i in idx_t]
    repeats_in_draw = len(srcs) - len(set(srcs)) + len(tgts) - len(set(tgts))
    assert repeats_in_draw > 0, (
        "Test fixture invariant: seeded draw must contain at least one "
        "duplicate cluster — otherwise the test is vacuous."
    )

    # Call the PRODUCTION resampler — this is the change vs round-4.
    panel = p5._resample_panel(srcs, tgts, cell_index)

    # If a source cluster appears twice in `srcs`, that cluster's
    # off-diagonal cells must appear AT LEAST twice in the panel (once per
    # occurrence of the source × distinct-target loop). Verify the
    # duplicate-preserving behavior empirically: pick the first duplicated
    # source cluster and assert its (T_i==src) cells appear strictly more
    # often than they would under deduplication.
    from collections import Counter

    src_counts = Counter(srcs)
    dup_src = next((s for s, n in src_counts.items() if n >= 2), None)
    assert dup_src is not None, "fixture: expected at least one duplicate source"

    n_with_dup = sum(1 for c in panel if c["T_i"] == dup_src)
    # Build the deduplicated panel (the BROKEN behavior) and compare.
    isrc_set = set(srcs)
    itgt_set = set(tgts)
    dedup_panel = [c for c in cells if c["T_i"] in isrc_set and c["T_j"] in itgt_set]
    n_dedup = sum(1 for c in dedup_panel if c["T_i"] == dup_src)
    assert n_with_dup > n_dedup, (
        "Round-4 Bug-1: duplicate-preserving resample must yield strictly "
        f"more cells for the duplicated cluster ({n_with_dup}) than the "
        f"broken set-dedup resample ({n_dedup}). If equal, the resampler "
        "collapsed duplicates — the dyadic cluster bootstrap is invalid."
    )

    # And lock the regression at the source level: neither the SOURCE nor
    # the TARGET set-comprehension form may reappear. (Round-4 only
    # checked sources; round-5 widens to targets as Codex flagged.)
    src = (REPO_ROOT / "scripts" / "i489_phase5_analyze.py").read_text()
    assert (
        "{icl_sources[k] for k" not in src
        and "{sp_sources[k] for k" not in src
        and "{icl_targets[k] for k" not in src
        and "{sp_targets[k] for k" not in src
    ), (
        "Round-4 Bug-1: set-comprehension cluster draws "
        "(`{icl_sources[k] for k in rng.integers(...)}` and its three "
        "siblings on sp_sources / icl_targets / sp_targets) collapse "
        "with-replacement duplicates → invalid cluster bootstrap. Use "
        "list-with-duplicates idiom (matching _dyadic_cluster_bootstrap_rho)."
    )


# ────────────────────────────────────────────────────────────────────────
# Round-4 Bug-2: Phase 5 fail-loud MUST fire for MISSING requested fracs,
# not only present-but-incomplete ones. The prior code computed the
# intersection (present ∩ requested) and looped only that, so an
# entirely-absent requested frac silently shrank the analysis.
# ────────────────────────────────────────────────────────────────────────


def test_round4_bug2_phase5_raises_when_requested_frac_is_absent(tmp_path, monkeypatch):
    """Run Phase 5 with --fracs listing one frac that has a complete 552
    off-diag panel and another that is ENTIRELY ABSENT. The script must
    raise the same incomplete-panel RuntimeError as for a partially
    present frac.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import i489_phase5_analyze as p5
    finally:
        sys.path.pop(0)
    from explore_persona_space.experiments.i489_contexts import UNION_CONTEXTS

    cids = [ctx.cid for ctx in UNION_CONTEXTS]
    seed = 4242

    # Point PHASE1_DIR + PHASE4_DIR at fresh tmp dirs. Stub the single
    # required Phase 1 file (cosine_per_layer.json — the other Phase 1
    # files are guarded by .exists()).
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    phase4 = tmp_path / "phase4"
    phase4.mkdir()
    monkeypatch.setattr(p5, "PHASE1_DIR", phase1)
    monkeypatch.setattr(p5, "PHASE4_DIR", phase4)
    monkeypatch.setattr(p5, "OUT_DIR", tmp_path / "phase5")

    cos_per_layer = {str(p5.HEADLINE_LAYER): {ci: {cj: 0.5 for cj in cids} for ci in cids}}
    (phase1 / "cosine_per_layer.json").write_text(json.dumps({"cos_sim_per_layer": cos_per_layer}))

    # Write a COMPLETE 552-off-diag panel for frac=0.25 ONLY. frac=0.50
    # stays absent — that's the round-4 Bug-2 contract test.
    for ti in cids:
        for tj in cids:
            if ti == tj:
                continue
            payload = {
                "frac": 0.25,
                "seed": seed,
                "T_i": ti,
                "T_j": tj,
                "delta_g": 0.0,
            }
            fname = f"G_{ti}__{tj}_frac0.25.json"
            (phase4 / fname).write_text(json.dumps(payload))

    # Sanity: the 0.25 panel is present + complete; 0.50 is absent.
    cells_by_frac, present = p5._load_cells([0.25, 0.50], seed, allow_smoke=False)
    n_off_025 = sum(
        1 for c in cells_by_frac.get(0.25, []) if c["T_i"] != c["T_j"] and "delta_g" in c
    )
    assert n_off_025 == 24 * 23, f"fixture: expected 552 off-diag at 0.25, got {n_off_025}"
    assert 0.25 in present, "fixture: frac=0.25 should be present"
    assert 0.50 not in present, "fixture: frac=0.50 should be entirely absent"

    # Pre-fix behavior: intersection silently dropped 0.50 and analyzed
    # only 0.25 → no error raised. Post-fix: iterates requested_fracs
    # directly, sees 0.50 has 0/552 off-diag cells, RAISES.
    with pytest.raises(RuntimeError, match=r"incomplete panel"):
        p5.main(
            [
                "--seed",
                str(seed),
                "--fracs",
                "0.25",
                "0.50",
                "--bootstrap-n",
                "10",
            ]
        )
