"""Issue #1415 hooked-unhooked decomposition — CPU tests t1-t5 (plan v11 §3.3).

t1  pins the hook/capture registration-order assumption + the causal-zero
    claim + alpha*Δ edit-injection exactness on a tiny fp32 model (THE
    load-bearing test — plan §3.2 consequences (i)-(iii)).
t2  teacher-forced ``arm_at(T-1)`` ≡ generation-mode ``arm(T)`` bit-for-bit
    on a context-only forward.
t3  tiny e2e: unhooked shards produced through the round-1 tiny path, then
    the FULL h0→h4 chain against them (PASS_UNIFIED smoke substrate).
t4  G0 pairing-mismatch fails loud naming the cell.
t5  fidelity verdict + rc=9 routing + the h3 drop-class probes (the
    data-dependent-gates duty: each gate branch executes once on a
    degenerate input).

No mocks anywhere — real tiny model bodies at every seam (the #906
production-body rule); the only faked thing is GPU-scale weights (a 3-layer
from-config Qwen over the real vocab).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1415_hooked_decomp as hd  # noqa: E402
from issue1415_run_phase1 import load_model_and_tokenizer  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    DeltaHook,
    capture_binned_answer_profiles,
    context_token_ids,
)

LAYERS = [0, 1, 2]
STEER = 1
ALPHA = 4.0
CTX = {"system": "You are a meticulous cartographer.", "user": "Describe your maps."}
COMPLETIONS = [
    "The first map wanders slowly across the wide valley, noting each landmark and river.",
    "A second chart follows the coastline, marking every harbor, cliff, and quiet village.",
]


@pytest.fixture(scope="module")
def tiny_model():
    cfg = SimpleNamespace(
        tiny=True, model_id=hd.MODEL_ID, hidden=32, n_model_layers=3, device="cpu"
    )
    model, tok = load_model_and_tokenizer(cfg)
    return model, tok


def _delta(h: int = 32) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(h)


# ── t1: hook/capture ordering + causal structure + alpha*Δ exactness ──────


def test_t1_hooked_capture_causal_structure_and_alpha_delta_exactness(tiny_model):
    model, tok = tiny_model
    delta = _delta()
    unhooked = capture_binned_answer_profiles(
        model, tok, CTX, COMPLETIONS, LAYERS, batch_size=2, capture_ctx_vec=True
    )
    hook = DeltaHook(model, STEER, delta, ALPHA)
    with hook:
        hooked = capture_binned_answer_profiles(
            model,
            tok,
            CTX,
            COMPLETIONS,
            LAYERS,
            batch_size=2,
            hook=hook,
            capture_ctx_vec=True,
        )
    assert hook.n_edits == 1  # one chunk forward, exactly one edit

    # (i) causal-zero: answer-span activations at layers <= steer are
    # BIT-IDENTICAL between regimes (fp32 CPU deterministic forwards).
    for li, layer in enumerate(LAYERS):
        if layer <= STEER:
            assert torch.equal(hooked["profiles"][:, :, li], unhooked["profiles"][:, :, li]), (
                f"layer {layer} (<= steer) answer profiles differ"
            )
            assert torch.equal(hooked["span_mean"][:, li], unhooked["span_mean"][:, li])
    # (ii) the direct component exists ONLY above the steer layer.
    above = LAYERS.index(STEER + 1)
    diff_above = hooked["profiles"][:, :, above] - unhooked["profiles"][:, :, above]
    assert float(diff_above.abs().nan_to_num().max()) > 0.0

    # (iii) G2 identity: the captured value at (steer, edit position) equals
    # the unhooked value + alpha*Δ EXACTLY — this is ALSO the registration-order
    # pin (the per-call capture hook observes the POST-edit block output).
    si = LAYERS.index(STEER)
    d = hooked["ctx_vec"][:, si] - unhooked["ctx_vec"][:, si]
    scaled = (ALPHA * delta.to(torch.float32)).expand_as(d)
    assert torch.equal(d, scaled)
    # below the steer layer the context vector is untouched
    assert torch.equal(hooked["ctx_vec"][:, 0], unhooked["ctx_vec"][:, 0])


# ── t2: teacher-forced arming ≡ generation-mode arming ────────────────


def test_t2_edit_position_equals_generation_mode_arming(tiny_model):
    model, tok = tiny_model
    delta = _delta()
    ctx_ids = context_token_ids(tok, CTX)
    input_ids = torch.tensor([ctx_ids], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    T = input_ids.shape[1]

    hook_a = DeltaHook(model, STEER, delta, ALPHA)
    with hook_a:
        hook_a.arm(expected_prompt_len=T)  # generation-mode prefill arming
        cap_a = extract_layer_activations(model, input_ids, LAYERS, attention_mask=mask)
    hook_b = DeltaHook(model, STEER, delta, ALPHA)
    with hook_b:
        hook_b.arm_at(T - 1)  # teacher-forced arming, same position
        cap_b = extract_layer_activations(model, input_ids, LAYERS, attention_mask=mask)
    assert hook_a.n_edits == 1 and hook_b.n_edits == 1
    for layer in LAYERS:
        assert torch.equal(cap_a[layer], cap_b[layer]), f"layer {layer} diverges"

    # mode hygiene: all_positions is mutually exclusive with edit_position
    with pytest.raises(AssertionError):
        DeltaHook(model, STEER, delta, ALPHA, all_positions=True, edit_position=3)
    hook_c = DeltaHook(model, STEER, delta, ALPHA, all_positions=True)
    with pytest.raises(AssertionError):
        hook_c.arm_at(3)


# ── t3: tiny e2e (IDENTICAL h0→h4 chain — PASS_UNIFIED) ───────────────


def test_t3_tiny_e2e_full_chain(tmp_path):
    work = tmp_path / "work"
    hd.main(["--tiny", "--work-root", str(work), "--phase", "all"])
    out = work / "out"

    summary = json.loads((out / "summary.json").read_text())
    labels = summary["labels"]
    assert len(labels) == 6, sorted(labels)
    lattice = [k for k, v in labels.items() if v["lattice"]]
    assert len(lattice) == 4, lattice
    for info in labels.values():
        assert info["verdict"] in {"direct-persistence", "late-null", "inconclusive"}
        hl = info["headline_read_layer"]
        assert hl > info["steer_layer"]  # pre-registered above-steer read
    fid = json.loads((out / "fidelity_gate_report.json").read_text())
    assert fid["fired"] is False
    assert fid["g1"]["n_cells"] == 6 and fid["g2"]["n_cells"] == 6
    rows = json.loads((out / "per_pair_direct_profiles.json").read_text())["rows"]
    # 2 pairs x [steered: 2 arms x (2 reads @L0 + 1 read @L1) = 6; baseline:
    # 2 reads @L0 + 1 read @L1 = 3] = 18 rows
    assert len(rows) == 18, len(rows)
    assert (out / "null_bands_direct.json").exists()
    assert (out / "jitter_reference.json").exists()
    manifest = json.loads((out / "hooked_manifest.json").read_text())
    assert len(manifest["cells"]) == 12
    stores = sorted((work / "tensors").rglob("*.pt"))
    assert len(stores) == 12
    # local-mirror upload landed under the hooked prefix (h2 + h4)
    mirror = work / "bulk" / "hf_mirror" / hd.HOOKED_TENSOR_PREFIX
    assert mirror.is_dir() and any(mirror.rglob("*.pt"))
    assert (work / "bulk" / "hf_mirror" / hd.HOOKED_TENSOR_PREFIX / "manifest.json").exists()


# ── t4: G0 pairing mismatch fails LOUD naming the cell ────────────────


def test_t4_g0_mismatch_fails_loud():
    good = {"kept_indices": [0, 1], "comp_token_counts": [5, 6]}
    hd._assert_g0(good, good, "cell-ok")  # no raise on identity
    bad = {"kept_indices": [0, 2], "comp_token_counts": [5, 6]}
    with pytest.raises(RuntimeError, match=r"G0 pairing mismatch for gen1c/ctx/cellX"):
        hd._assert_g0(good, bad, "gen1c/ctx/cellX")
    bad_counts = {"kept_indices": [0, 1], "comp_token_counts": [5, 7]}
    with pytest.raises(RuntimeError, match="comp_token_counts"):
        hd._assert_g0(good, bad_counts, "gen1c/ctx/cellY")


# ── t5: fidelity verdict + rc routing + h3 drop-class probes ──────────


def _g1_row(cell: str, min_cos: float) -> dict:
    return {"cell_id": cell, "steer_layer": 20, "min_cos": min_cos, "n_comparisons": 5}


def _g2_row(cell: str, cos: float, ratio: float) -> dict:
    return {
        "cell_id": cell,
        "delta_arm": "context",
        "steer_layer": 20,
        "cos_d_delta": cos,
        "norm_ratio": ratio,
        "ctx_vec_max_dev": 0.0,
        "cross_arm_delta_cos": 0.5,
        "passed": bool(cos >= hd.G2_COS_MIN and hd.G2_RATIO_LO <= ratio <= hd.G2_RATIO_HI),
    }


def test_t5_fidelity_verdict_branches():
    clean_g1 = [_g1_row(f"c{i}", 0.99999) for i in range(12)]
    clean_g2 = [_g2_row(f"c{i}", 0.999, 1.0) for i in range(12)]
    assert hd.fidelity_verdict([], clean_g1, clean_g2)["fired"] is False
    # G0: ANY mismatch fires (structural).
    v = hd.fidelity_verdict([{"cell_id": "cX", "error": "mismatch"}], clean_g1, clean_g2)
    assert v["fired"] and v["g0"]["fired"]
    # G1: exactly MAX_BAD bad cells does NOT fire; MAX_BAD+1 does.
    g1_edge = [_g1_row(f"b{i}", 0.5) for i in range(hd.G1_MAX_BAD)] + clean_g1
    assert hd.fidelity_verdict([], g1_edge, clean_g2)["fired"] is False
    g1_bad = [_g1_row(f"b{i}", 0.5) for i in range(hd.G1_MAX_BAD + 1)] + clean_g1
    v = hd.fidelity_verdict([], g1_bad, clean_g2)
    assert v["fired"] and v["g1"]["fired"] and v["g1"]["n_bad"] == hd.G1_MAX_BAD + 1
    # G2: any wrong-cos OR out-of-band norm ratio fires (wrong-alpha class: 4x).
    for cos, ratio in ((0.2, 1.0), (0.999, 4.0), (0.999, 0.25)):
        v = hd.fidelity_verdict([], clean_g1, [*clean_g2, _g2_row("bad", cos, ratio)])
        assert v["fired"] and v["g2"]["fired"]


def test_t5_enforce_fidelity_rc9_and_report(tmp_path):
    cfg = SimpleNamespace(out_root=tmp_path, model_id=hd.MODEL_ID, tiny=True)
    fired = hd.fidelity_verdict([], [_g1_row(f"b{i}", 0.1) for i in range(4)], [])
    with pytest.raises(SystemExit) as ei:
        hd._enforce_fidelity(cfg, fired)
    assert ei.value.code == hd.RC_FIDELITY_HALT == 9
    report = json.loads((tmp_path / "fidelity_gate_report.json").read_text())
    assert report["fired"] is True  # artifact written BEFORE the exit (rc routing)
    clean = hd.fidelity_verdict([], [_g1_row("ok", 1.0)], [])
    hd._enforce_fidelity(cfg, clean)  # PASS branch: report rewritten, no exit
    assert json.loads((tmp_path / "fidelity_gate_report.json").read_text())["fired"] is False


def test_t5_log_ratio_drop_classes():
    # all-NaN class > MAX_DROP_FRAC fails LOUD naming the label + pairs.
    mags = {f"p{i}": {"dec8": None, "dec9": None, "dec10": None} for i in range(4)}
    with pytest.raises(RuntimeError, match=r"R\[probe\].*dropped"):
        hd._log_ratio_pairs(mags, mags, hd.LATE_BINS, hd.LATE_BINS, 2.0, "R[probe]")
    # non-positive means: NAMED exclusion class, never guard-tripping.
    mags = {
        "p_ok": {"dec8": 1.0, "dec9": 1.0, "dec10": 1.0},
        "p_zero": {"dec8": 0.0, "dec9": 0.0, "dec10": 0.0},
    }
    vals, dropped, nonpos = hd._log_ratio_pairs(
        mags, mags, hd.LATE_BINS, hd.LATE_BINS, 1.0, "R[probe2]"
    )
    assert sorted(vals) == ["p_ok"] and dropped == [] and sorted(nonpos) == ["p_zero"]
    # healthy path: log(num) - log(scale * den)
    assert vals["p_ok"] == pytest.approx(0.0)


def test_t5_pair_scalar_stat_shapes():
    stat = hd._pair_scalar_stat({"a": 1.0, "b": 2.0, "c": 3.0}, n_boot=100, seed=1)
    assert stat["mean"] == pytest.approx(2.0)
    lo, hi = stat["ci95"]
    assert lo <= stat["mean"] <= hi
    empty = hd._pair_scalar_stat({}, n_boot=10, seed=1)
    assert empty["mean"] is None and empty["ci95"] is None
