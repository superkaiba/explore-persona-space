"""#1947 unit-3 pins: last-token span capture (binding directive), judge/select
P3 drivers (stub judge, drop tally, verdict lattice), consumed-rows schema
derivation, and the batched bootstrap-CI helper.

The tiny-model capture test executes the REAL `_teacher_forced_span_means`
body (from-config 2-layer same-arch model over the REAL Qwen vocab — the #906
tiny-real convention; CPU, seconds)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1947_battery as bat  # noqa: E402
import issue1947_cells as cells  # noqa: E402

SLUG = "syc-pers-con-sv-s42"  # the pilot cell (a REAL registry slug)


def _cfg(tmp_path: Path, **kw) -> bat.Cfg:
    defaults = dict(
        out_root=tmp_path / "root",
        out_dir=tmp_path / "analysis",
        smoke=True,
        stub_judge=True,
        upload=False,
        local_ladders=True,
        cells_filter=(SLUG,),
        behaviors=("sycophancy",),
    )
    defaults.update(kw)
    cfg = bat.Cfg(**defaults)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_ladder(cfg: bat.Cfg, slug: str, rates_shape: dict[str, list[list[str]]]) -> None:
    payload = {
        "slug": slug,
        "behavior": "sycophancy",
        "questions": ["q one?", "q two?"],
        "rungs": {s: {"completions": comps} for s, comps in rates_shape.items()},
    }
    p = cfg.out_root / "ladders" / slug / "ladder_rollouts.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def test_judge_stub_tally_and_resume(tmp_path):
    """Stub judge over a 2-rung synthetic ladder: rates in [0,1], drop split
    fields present, per-rung checkpoints regime-keyed (a second run resumes)."""
    cfg = _cfg(tmp_path)
    comps = [["yes indeed", "agree"], ["no", "disagree"]]
    _write_ladder(cfg, SLUG, {"5": comps, "10": comps})
    assert bat.cmd_judge(cfg) == 0
    judged = json.loads((cfg.out_dir / "judge" / f"judged_{SLUG}.json").read_text())
    assert set(judged["rates_by_step"]) == {"5", "10"}
    for rec in judged["records"].values():
        assert 0.0 <= rec["rate"] <= 1.0
        assert "n_dropped_draws_content" in rec and "n_transport_lost_draws" in rec
    assert bat.cmd_judge(cfg) == 0  # resume path: checkpoints reused, same regime


def test_select_verdict_lattice_earliest_in_band(tmp_path, monkeypatch):
    """Earliest-in-band rung wins; a never-in-band ladder falls back to
    closest-approach with in_band False (the DoseSelection invariant)."""
    cfg = _cfg(tmp_path)
    judge_dir = cfg.out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    in_band = {"5": 0.2, "10": 0.7, "15": 0.9}  # earliest in [0.60, 0.85] = 10
    (judge_dir / f"judged_{SLUG}.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "behavior": "sycophancy",
                "instrument": "stub-smoke",
                "questions_sha256": "x",
                "rates_by_step": in_band,
                "records": {},
            }
        )
    )
    monkeypatch.setattr(bat, "_marker_cells", lambda _cfg: [])
    assert bat.cmd_select(cfg) == 0
    man = json.loads(cfg.verdict_manifest_path().read_text())
    sel = man["content"][SLUG]["selection"]
    assert sel["step"] == 10 and sel["in_band"] is True and sel["fallback"] is None
    # closest-approach fallback
    (judge_dir / f"judged_{SLUG}.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "behavior": "sycophancy",
                "instrument": "stub-smoke",
                "questions_sha256": "x",
                "rates_by_step": {"5": 0.1, "10": 0.3},
                "records": {},
            }
        )
    )
    assert bat.cmd_select(cfg) == 0
    sel = json.loads(cfg.verdict_manifest_path().read_text())["content"][SLUG]["selection"]
    assert sel["in_band"] is False and sel["fallback"] == "closest_approach"
    assert sel["step"] == 10  # min band distance


def test_consumed_row_idxs_schemas(tmp_path):
    """Primary: the train/sft.py seam schema (realized_step_of_idx, 0-based);
    plus the two legacy fixture shapes and the rep-cell no-seam rule."""
    cfg = _cfg(tmp_path)
    p = cfg.out_root / "ladders" / SLUG / "realized_consumption.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"realized_step_of_idx": [0, 0, 1, 2, None]}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    p.write_text(json.dumps({"row_to_step": {"0": 0, "1": 0, "2": 1, "3": 2}}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    p.write_text(json.dumps({"steps": {"0": [0, 1], "1": [2], "2": [3]}}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    rep = "imp-pers-con-rep-s42"  # no seam: all-rows once epoch 1 completes
    assert bat._consumed_row_idxs(cfg, rep, 15) == set(range(80))
    with pytest.raises(RuntimeError, match="unknowable"):
        bat._consumed_row_idxs(cfg, rep, 2)


def test_adapter_subfolder_marker_vs_content():
    assert bat._adapter_subfolder(SLUG, 10) == f"issue1947/{SLUG}/checkpoint-10"
    assert bat._adapter_subfolder("mk-pers-con-sv-s42", 40) == (
        "issue1947/marker/mk-pers-con-sv-s42/checkpoint-40"
    )


def test_corpus_arm_slugs_are_registry_cells():
    for slug in bat.CORPUS_ARM_SLUGS:
        assert slug in cells.CELL_BY_SLUG, slug
    assert len(bat.CORPUS_ARM_SLUGS) == 12


def test_boot_cos_ci_batched_contains_point():
    import issue1947_analysis as ana
    import numpy as np

    rng = np.random.default_rng(0)
    cand = rng.normal(size=32)
    stack = cand[None, :] + 0.5 * rng.normal(size=(200, 32))
    ci, mean_cos = ana._boot_cos_ci(stack, cand, 400, seed=1)
    # NOTE: point-coverage is NOT a percentile-bootstrap guarantee for the
    # concave cos statistic (small downward bias vs the plug-in point — the
    # same shape as the reused #1768 _boot_ci convention). Pins: the bootstrap
    # mean sits inside its own CI; the batched weight-GEMM reproduces a serial
    # per-draw loop over the SAME weights (batched-rewrite equivalence).
    assert ci[0] <= mean_cos <= ci[1]
    assert ci[1] - ci[0] > 1e-4  # non-degenerate width
    rng2 = np.random.default_rng(1)
    n = stack.shape[0]
    W = rng2.multinomial(n, np.full(n, 1.0 / n), size=400).astype(np.float64) / n
    cn = cand / np.linalg.norm(cand)
    serial = np.array([_c for _c in ((w @ stack) @ cn / np.linalg.norm(w @ stack) for w in W)])
    batched = (W @ stack) @ cn / np.linalg.norm(W @ stack, axis=1)
    assert np.allclose(serial, batched, atol=1e-10)
    # ordering sanity: an anti-aligned candidate reads a strictly lower CI
    ci_neg, _ = ana._boot_cos_ci(stack, -cand, 400, seed=1)
    assert ci_neg[1] < ci[0]


@pytest.mark.slow
def test_last_token_span_capture_tiny_real_model(tmp_path):
    """REAL `_teacher_forced_span_means` body on a from-config 2-layer Qwen2
    over the REAL vocab: all 5 spans present, and a prefix of length 1 makes
    span-mean('prefix') == last-token('prefix_last') exactly (1-token span
    identity — the directive's same-forward-pass guarantee)."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from explore_persona_space.analysis.representation_shift import (
        SPAN_ARMS_LAST,
        _teacher_forced_span_means,
    )

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    config = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model_dir = tmp_path / "tiny"
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    ids = tok("hello there, what is up today?", add_special_tokens=False)["input_ids"]
    rows = [
        {
            "persona": "t",
            "question_idx": i,
            "prompt_sha": f"s{i}",
            "prompt_token_ids": ids,
            "response_token_ids": ids[:3],
            "prefix_len": 1,  # 1-token prefix: prefix == prefix_last exactly
            "context_len": len(ids),
        }
        for i in range(2)
    ]
    pooled = _teacher_forced_span_means(
        str(model_dir),
        rows,
        ["t"],
        layers=[1],
        spans=("prefix", "context", "response", *SPAN_ARMS_LAST),
        device="cpu",
        dtype=torch.float32,
        tf_batch_size=2,
    )
    assert set(pooled) == {"prefix", "context", "response", "prefix_last", "context_last"}
    assert torch.allclose(pooled["prefix"][1], pooled["prefix_last"][1])
    assert not torch.allclose(pooled["context"][1], pooled["context_last"][1])
    for span in pooled:
        assert pooled[span][1].shape == (2, 32)
        assert torch.isfinite(pooled[span][1]).all()
