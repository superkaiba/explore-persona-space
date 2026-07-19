"""Issue #1489 round-3: margin DV (plan §6 dual-DV (b)) + rule-21 reliability wiring.

Covers, per the data-dependent-gates duty + the production-body rule:

- ``draft_margin_pair`` / ``_query_stub`` — the mechanical drafting body
  (shared query-echo prefix, clause-only difference, determinism, unknown-slug
  KeyError).
- ``margin_pair_filter`` + ``margin_kept_floor`` — the judge-filter gate
  branches probed at unit level (keep / drop-on-<=50 / drop-on-None (rule 9) /
  floor arithmetic incl. the smoke-N one-flake tolerance / zero-pool raise).
- ``_reliability`` — the rule-21 aligned split-half wiring executed on the
  REAL #763 machinery (``reliability_split_half_over_probes``): the
  <4-contexts all-None DESIGNED verdict, a 10-context signal-recovery case,
  and None-placeholder alignment under judge drops.
- ``score_margin_items`` — REAL body on a from-config 2-layer Qwen2 (fp32
  CPU) + the REAL Qwen tokenizer (the tiny-real standard,
  cf. tests/test_issue1112_tf_shared.py), equivalence-checked per (item,
  side) against a single-row FULL-logits reference — pins the
  ``hidden_states[-1] -> lm_head`` gather, the BPE-seam-safe span math, and
  right-padding correctness at batch>1; plus the empty-answer gate probe.
- ``_margin_items`` — missing-manifest-row KeyError gate probe.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue1489_margin_pools import (  # noqa: E402
    MARGIN_CLAUSES,
    _query_stub,
    draft_margin_pair,
)


def _judge_mod():
    import issue1489_judge as jm

    return jm


# ---------------------------------------------------------------------------
# Pool drafting (mechanical, deterministic)
# ---------------------------------------------------------------------------


def test_draft_margin_pair_shared_prefix_and_clauses():
    c, n = draft_margin_pair("fact_veg", "What should I  cook   for dinner tonight?")
    prefix = "Regarding your question about What should I cook for dinner tonight: "
    assert c.startswith(prefix) and n.startswith(prefix)
    assert c == prefix + MARGIN_CLAUSES["fact_veg"][0]
    assert n == prefix + MARGIN_CLAUSES["fact_veg"][1]
    # deterministic (fixed pools contract: drafted once, held fixed)
    assert draft_margin_pair("fact_veg", "What should I cook for dinner tonight?") == (c, n)


def test_draft_margin_pair_unknown_slug_raises():
    with pytest.raises(KeyError):
        draft_margin_pair("fact_tokyo", "anything")  # no clause pair drafted for it


def test_query_stub_truncates_and_strips_punctuation():
    stub = _query_stub("word " * 40)
    assert len(stub) <= 80
    assert _query_stub("Is Python compiled?") == "Is Python compiled"


# ---------------------------------------------------------------------------
# Judge-filter gate branches (data-dependent gates duty)
# ---------------------------------------------------------------------------


def test_margin_pair_filter_gate_branches():
    jm = _judge_mod()
    by_row = {
        "r1": {"consistent": 90.0, "inconsistent": 10.0},  # kept
        "r2": {"consistent": 40.0, "inconsistent": 10.0},  # dropped: consistent <= 50
        "r3": {"consistent": 90.0, "inconsistent": None},  # dropped: judge drop (rule 9)
        "r4": {"consistent": 90.0},  # dropped: side never judged
    }
    kept, dropped = jm.margin_pair_filter(by_row, ["r1", "r2", "r3", "r4", "r5"])
    assert kept == ["r1"]
    assert [d["base_row_id"] for d in dropped] == ["r2", "r3", "r4", "r5"]


def test_margin_kept_floor_values_and_zero_pool_raise():
    jm = _judge_mod()
    assert jm.margin_kept_floor(4) == 3  # smoke N: one flaky pair tolerated
    assert jm.margin_kept_floor(200) == 160  # production 80% floor
    assert jm.margin_kept_floor(1) == 1
    with pytest.raises(ValueError):
        jm.margin_kept_floor(0)


# ---------------------------------------------------------------------------
# Rule-21 reliability wiring (real #763 machinery)
# ---------------------------------------------------------------------------


def _result(scores: dict):
    from explore_persona_space.eval.graded_judge import JudgeResult

    return JudgeResult(scores=scores, n_total_draws=len(scores) * 5, n_dropped_draws=0)


def test_reliability_two_contexts_reports_designed_none_verdict():
    jm = _judge_mod()
    id_map: dict[str, dict] = {}
    scores: dict[str, float | None] = {}
    for arm in ("aug", "plain"):
        for i in range(6):
            iid = f"{arm}{i}"
            id_map[iid] = {"arm": arm, "base_row_id": f"r{i}"}
            scores[iid] = 50.0 + i
    out = jm._reliability(_result(scores), id_map)
    assert out["n_contexts_used"] == 2
    assert out["r_yy"] is None and out["sqrt_r_yy"] is None  # <4-contexts designed verdict
    assert out["method"] == "aligned"
    assert out["n_probe_rows"] == 6
    assert out["contexts"] == ["aug", "plain"]


def test_reliability_ten_contexts_recovers_signal():
    jm = _judge_mod()
    rng = random.Random(0)
    id_map: dict[str, dict] = {}
    scores: dict[str, float | None] = {}
    for a_i, arm in enumerate(f"ckpt{k}" for k in range(10)):
        for i in range(30):
            iid = f"{arm}_{i}"
            id_map[iid] = {"arm": arm, "base_row_id": f"r{i:03d}"}
            # strong cross-context signal + a probe main effect + noise
            scores[iid] = 10.0 * a_i + (i % 7) + rng.gauss(0, 0.5)
    out = jm._reliability(_result(scores), id_map)
    assert out["n_contexts_used"] == 10
    assert out["r_yy"] is not None and out["r_yy"] > 0.9
    assert 0.0 < out["sqrt_r_yy"] <= 1.0


def test_reliability_none_scores_stay_aligned_placeholders():
    jm = _judge_mod()
    id_map: dict[str, dict] = {}
    scores: dict[str, float | None] = {}
    for arm in ("a", "b", "c", "d"):
        for i in range(8):
            iid = f"{arm}{i}"
            id_map[iid] = {"arm": arm, "base_row_id": f"r{i}"}
            scores[iid] = None if (arm == "a" and i == 0) else float(ord(arm) + i)
    # a dropped draw enters as a None placeholder; the aligned machinery must
    # not raise its ragged-input ValueError
    out = jm._reliability(_result(scores), id_map)
    assert out["n_contexts_used"] == 4


# ---------------------------------------------------------------------------
# Teacher-forced scorer: tiny-real equivalence vs full-logits reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_qwen():
    """From-config 2-layer Qwen2 over the REAL Qwen vocab + the REAL tokenizer
    (fake only GPU-scale weights; every library type real)."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    config = Qwen2Config(
        vocab_size=max(152064, len(tok)),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(1489)
    model = Qwen2ForCausalLM(config).eval()
    return model, tok


def _items():
    return [
        {
            "base_row_id": f"r{i}",
            "side": "relevant" if i % 2 == 0 else "irrelevant",
            "prefix_text": "",
            "prompt": (
                f"<|im_start|>user\nQuestion {i}: what should we plan for "
                f"dinner{' with friends' * i}?<|im_end|>\n<|im_start|>assistant\n"
            ),
            "consistent": (
                f"Regarding your question {i}: since Sarah is a strict vegetarian "
                "with a peanut allergy, everything will be vegetarian and peanut-free."
            ),
            "inconsistent": (
                f"Regarding your question {i}: since Sarah eats meat and has no "
                "allergies, meat dishes and peanut sauces are all fine."
            ),
        }
        for i in range(3)
    ]


def test_score_margin_items_matches_full_logits_reference(tiny_qwen):
    import issue1489_gpu_phase as gp
    import torch

    model, tok = tiny_qwen
    items = _items()
    # 6 sequences at batch_size=4 -> 2 batches; mixed prompt lengths exercise
    # right padding inside a batch
    rows = gp.score_margin_items(
        model=model, tokenizer=tok, items=items, device="cpu", batch_size=4
    )
    assert len(rows) == 3
    boundary = gp.parent._boundary_suffix("instruct")
    for i, (item, row) in enumerate(zip(items, rows, strict=True)):
        assert row["base_row_id"] == item["base_row_id"]
        assert row["side"] == item["side"]
        for side, key in (
            ("consistent", "lnlogp_consistent"),
            ("inconsistent", "lnlogp_inconsistent"),
        ):
            ids, pos = gp.parent._capture_row_ids_and_positions(
                tok, item["prefix_text"], item["prompt"], item[side], boundary
            )
            a0, a1 = pos["answer_start"], pos["answer_end"]
            assert row[f"n_tokens_{side}"] == a1 - a0
            with torch.no_grad():
                full = model(input_ids=torch.tensor([ids])).logits[0].float()
            logp = torch.log_softmax(full, dim=-1)
            ref = float(logp[torch.arange(a0 - 1, a1 - 1), torch.tensor(ids[a0:a1])].mean())
            assert row[key] == pytest.approx(ref, abs=1e-4), (i, side)
        assert row["margin"] == pytest.approx(
            row["lnlogp_consistent"] - row["lnlogp_inconsistent"], abs=1e-9
        )


def test_score_margin_items_empty_answer_gate(tiny_qwen):
    import issue1489_gpu_phase as gp

    model, tok = tiny_qwen
    bad = _items()[:1]
    bad[0]["inconsistent"] = "   "
    with pytest.raises(ValueError, match="empty fixed answer"):
        gp.score_margin_items(model=model, tokenizer=tok, items=bad, device="cpu")


def test_margin_items_missing_manifest_row_raises():
    import issue1489_gpu_phase as gp

    pools = {
        "slugs": {
            "fact_veg": {
                "fact_text": "f",
                "rows": [
                    {
                        "base_row_id": "rX",
                        "side": "relevant",
                        "consistent": "c",
                        "inconsistent": "n",
                    }
                ],
            }
        }
    }
    manifest = [{"cell_id": "cell_plain", "base_row_id": "rOTHER"}]
    with pytest.raises(KeyError, match="no cell_plain manifest row"):
        gp._margin_items(manifest, pools, "fact_veg", "plain", {}, {})
