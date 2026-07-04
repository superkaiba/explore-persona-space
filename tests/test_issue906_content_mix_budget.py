"""Content-mix token-budget contract tests (task #906 r14).

r14 concern (``content-mix-token-budget-unenforced``, raised by the r13
code-reviewer): the CONTENT classes' mix path (``organisms.build_organism`` ->
``_assemble_mix``) shared the r13 truncation exposure — unbounded real-user
prompts (WildChat-lineage banks) + completions can exceed the content recipe's
``max_length=1024``, and there the truncation is SILENT (TRL SFTTrainer
right-truncates with no fail-loud collator), degrading completion supervision
without an error.

These tests pin the BUILD-time budget contract through the REAL assembly path
(``build_organism`` -> ``_assemble_mix`` -> ``enforce_mix_token_budget``) with
the REAL Qwen-2.5-7B-Instruct tokenizer + the REAL unified recipe budget
(1024). Fakes only at the injectable external boundaries build_organism ships
(``datagen_fn`` / ``train_fn`` / ``rate_fn``), signature-bound to the real
contract.

1. An overlong QUESTION is dropped from BOTH pos + cn sides even when the cn
   rows are ordered differently (question-keyed pairing — content datagen
   emits same-question negatives per panel member, NOT index-aligned).
2. An overlong GENERIC (WildChat-corpus) row drops individually; pos/cn are
   untouched and realized counts record the shrink.
3. A systematic overflow (> the 10% floor) fails LOUD before training.
4. ``tokenizer=None`` (the offline stub-seam path) skips the gate — legacy
   behavior, byte-compatible.
5. r13 Minors: a gate that EMPTIES a non-empty contrastive-negative side
   raises (positive-only training leaks uniformly); an asymmetric pos/cn drop
   logs a ratio warning.
6. The driver's ``MARKER_MIX_MAX_REJECT_FRAC`` literal equals the shared
   ``organisms.MIX_MAX_REJECT_FRAC`` floor (drift pin).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from explore_persona_space.artifacts.organisms import (
    MIX_MAX_REJECT_FRAC,
    ModelOrganism,
    build_organism,
    enforce_mix_token_budget,
    mix_row_token_len,
)
from explore_persona_space.artifacts.recipe import UNIFIED_OVERRIDES

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SOURCE = "persona_villain"
ORG_LOGGER = "explore_persona_space.artifacts.organisms"
BUDGET = int(UNIFIED_OVERRIDES["max_length"])  # 1024 — the content recipe budget
# ~6300 tokens under the Qwen render — far over both the 1024 content budget
# and the 2048 marker budget (the r13 crash-row shape: an extreme-tail
# WildChat prompt).
LONG_Q = ("depression cherry blossom motorcycle " * 900).strip()


@pytest.fixture(scope="module")
def qwen_tok():
    """The REAL Qwen tokenizer (the trainer's own render path)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


def _row(q: str, a: str, sp: str | None = None) -> dict:
    prompt = []
    if sp is not None:
        prompt.append({"role": "system", "content": sp})
    prompt.append({"role": "user", "content": q})
    return {"prompt": prompt, "completion": [{"role": "assistant", "content": a}]}


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _datagen_stub(pos_rows: list[dict], cn_rows: list[dict]):
    """Signature-bound datagen boundary stub: writes the given rows verbatim."""

    def stub(behavior, context_C, negatives, *, out_dir, seed, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pos = _write_jsonl(out / "pos.jsonl", pos_rows)
        cn = _write_jsonl(out / "cn.jsonl", cn_rows)
        pm = out / "pool_meta.json"
        pm.write_text("{}\n")
        return pos, cn, pm

    return stub


def _train_stub(base_model, data_path, output_dir, *, cfg=None, callbacks=None, **overrides):
    """Trainer boundary stub: fake single-rung checkpoint ladder."""
    out = Path(output_dir)
    (out / "checkpoint-25").mkdir(parents=True, exist_ok=True)
    return str(out), 0.5


def _fail_train(*_a, **_k):
    pytest.fail("train_fn must not be called: the budget gate must raise at assembly")


def test_floor_constant_shared():
    """The driver's r13 literal floor equals the shared organisms floor (drift pin)."""
    import issue906_phase1_pilot as pilot

    assert pilot.MARKER_MIX_MAX_REJECT_FRAC == MIX_MAX_REJECT_FRAC == pytest.approx(0.10)


def test_overlong_question_dropped_from_both_sides_through_real_build(tmp_path, qwen_tok, caplog):
    """FAILS PRE-FIX: an overlong content-mix question must be dropped from
    BOTH pos + cn at BUILD — pre-fix it sailed into training where SFTTrainer
    right-truncation silently degraded its completion supervision.

    cn rows are REVERSED relative to pos (content datagen orders negatives by
    panel member, not by pos index) — pinning QUESTION-keyed pairing, not the
    marker path's index alignment.
    """
    questions = [f"Short content question number {i}?" for i in range(20)]
    questions.insert(1, LONG_Q)
    pos_rows = [_row(q, f"pos answer {i}.") for i, q in enumerate(questions)]
    cn_rows = [_row(q, f"neg answer {i}.") for i, q in enumerate(reversed(questions))]
    org = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)

    with caplog.at_level(logging.INFO, logger=ORG_LOGGER):
        res = build_organism(
            org,
            out_root=tmp_path,
            datagen_fn=_datagen_stub(pos_rows, cn_rows),
            train_fn=_train_stub,
            rate_fn=lambda _c: 0.7,
            tokenizer=qwen_tok,
        )

    mix = [json.loads(line) for line in Path(res.train_mix_path).read_text().splitlines()]
    assert len(mix) == 40  # 20 pos + 20 cn; the overlong question pair-dropped
    kept_qs = {r["prompt"][-1]["content"] for r in mix}
    assert LONG_Q not in kept_qs

    # The silent-truncation predicate, verified numerically under the trainer's
    # EXACT render: every KEPT row fits the budget; the DROPPED row exceeds it.
    for r in mix:
        assert mix_row_token_len(r, qwen_tok) <= BUDGET
    assert mix_row_token_len(_row(LONG_Q, "pos answer 1."), qwen_tok) > BUDGET

    # Fail-loud telemetry: the [content-mix-budget] log line + sidecar + meta.
    assert any("[content-mix-budget]" in rec.getMessage() for rec in caplog.records)
    sidecar = json.loads((tmp_path / "mix_budget.json").read_text())
    assert sidecar["enforced"] is True
    assert sidecar["budget"] == BUDGET
    assert sidecar["n_rejected"] == 2
    assert sidecar["n_rejected_pos"] == 1 and sidecar["n_rejected_cn"] == 1
    assert sidecar["max_row_tokens"] > BUDGET
    meta = json.loads((tmp_path / "mix_meta.json").read_text())
    assert meta["mix_token_budget"]["enforced"] is True
    assert meta["counts_realized"] == {"positives": 20, "negatives": 20, "generic": 0}


def test_overlong_generic_row_dropped_individually(tmp_path, qwen_tok):
    """An overlong generic (WildChat-corpus) row drops on its own; pos/cn stay."""
    pos_rows = [_row(f"posq{i}", f"pos answer {i}") for i in range(8)]
    cn_rows = [_row(f"posq{i}", f"neg answer {i}") for i in range(8)]
    org = ModelOrganism("sycophancy", SOURCE)  # generic_frac None -> recipe default 0.5
    # mix_counts(8, gf=0.5, neg_ratio=1.0)["generic"] == 16: a 16-row corpus is
    # sampled WHOLE, so the one overlong row is guaranteed into the gate.
    corpus = [_row(f"genq{i}", f"generic answer {i}") for i in range(15)]
    corpus.append(_row(LONG_Q, "generic overlong answer"))
    generic_path = _write_jsonl(tmp_path / "generic.jsonl", corpus)

    res = build_organism(
        org,
        out_root=tmp_path / "run",
        generic_data_path=generic_path,
        datagen_fn=_datagen_stub(pos_rows, cn_rows),
        train_fn=_train_stub,
        rate_fn=lambda _c: 0.7,
        tokenizer=qwen_tok,
    )
    mix = [json.loads(line) for line in Path(res.train_mix_path).read_text().splitlines()]
    assert len(mix) == 31  # 8 pos + 8 cn + 15 generic (1 generic dropped)
    sidecar = json.loads((tmp_path / "run" / "mix_budget.json").read_text())
    assert sidecar["n_rejected_generic"] == 1
    assert sidecar["n_rejected_pos"] == 0 and sidecar["n_rejected_cn"] == 0
    assert sidecar["n_kept_generic"] == 15
    meta = json.loads((tmp_path / "run" / "mix_meta.json").read_text())
    assert meta["counts_realized"] == {"positives": 8, "negatives": 8, "generic": 15}
    # The plan-vs-realized shrink is visible, never silent.
    assert meta["counts_planned"]["generic"] == 16


def test_systematic_content_overflow_fails_loud_before_training(tmp_path, qwen_tok):
    """Rejected fraction above the floor raises at ASSEMBLY (train never runs)."""
    pos_rows = [_row(LONG_Q, "pos answer.")]
    cn_rows = [_row(LONG_Q, "neg answer.")]
    org = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    with pytest.raises(RuntimeError, match=r"\[content-mix-budget\]"):
        build_organism(
            org,
            out_root=tmp_path,
            datagen_fn=_datagen_stub(pos_rows, cn_rows),
            train_fn=_fail_train,
            rate_fn=lambda _c: 0.7,
            tokenizer=qwen_tok,
        )


def test_no_tokenizer_skips_gate_legacy_behavior(tmp_path):
    """tokenizer=None (the offline stub-seam path) keeps every row — the gate
    is opt-in via the tokenizer, exactly the r13 marker-path contract."""
    questions = [f"Short content question number {i}?" for i in range(20)]
    questions.insert(1, LONG_Q)
    pos_rows = [_row(q, f"pos answer {i}.") for i, q in enumerate(questions)]
    cn_rows = [_row(q, f"neg answer {i}.") for i, q in enumerate(questions)]
    org = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    res = build_organism(
        org,
        out_root=tmp_path,
        datagen_fn=_datagen_stub(pos_rows, cn_rows),
        train_fn=_train_stub,
        rate_fn=lambda _c: 0.7,
    )
    mix = [json.loads(line) for line in Path(res.train_mix_path).read_text().splitlines()]
    assert len(mix) == 42  # all rows kept, overlong included (legacy behavior)
    sidecar = json.loads((tmp_path / "mix_budget.json").read_text())
    assert sidecar["enforced"] is False


def test_gate_that_empties_cn_side_fails_loud(qwen_tok):
    """r13 Minor: rejecting EVERY contrastive negative (while below the floor)
    raises — positive-only training leaks uniformly (#18/#207)."""
    pos_rows = [_row(f"qa{i}", f"answer {i}") for i in range(19)]
    cn_rows = [_row(LONG_Q, "neg answer.")]  # 1/20 rejected = 5% < the 10% floor
    with pytest.raises(ValueError, match="contrastive-negative"):
        enforce_mix_token_budget(pos_rows, cn_rows, qwen_tok, BUDGET, label="content-mix-budget")


def test_asymmetric_drop_logs_ratio_warning(qwen_tok, caplog):
    """r13 Minor: a drop that perturbs the ~1:1 pos:cn ratio logs a WARNING."""
    pos_rows = [_row(LONG_Q, "pos answer.")] + [_row(f"qa{i}", f"answer {i}") for i in range(19)]
    cn_rows = [_row(f"qb{i}", f"neg answer {i}") for i in range(20)]  # disjoint questions
    with caplog.at_level(logging.WARNING, logger=ORG_LOGGER):
        kept_pos, kept_cn, _g, stats = enforce_mix_token_budget(
            pos_rows, cn_rows, qwen_tok, BUDGET, label="content-mix-budget"
        )
    assert stats["n_rejected_pos"] == 1 and stats["n_rejected_cn"] == 0
    assert len(kept_pos) == 19 and len(kept_cn) == 20
    assert any("asymmetric drop" in rec.getMessage() for rec in caplog.records)
