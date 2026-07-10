"""#1090 crash-fix: ``build_organism(recipe_max_length=...)`` — one budget authority.

The GCP att-20260706-235853 + pod-1090 pid-2471 crashes: the plan-v4-declared
``max_length=2048`` deviation was hot-fixed at the WRONG seam (05b2405043
patched only ``_make_train_fn``'s ``dataclasses.replace``), while
``organisms.build_organism`` enforces the mix token budget at MIX-BUILD time
from ``spec.overrides["max_length"]`` (``spec = organism.recipe`` ->
``recipe_for`` -> ``UNIFIED_OVERRIDES`` 1024) — UPSTREAM of ``train_fn`` — so
the run died again at ``[content-mix-budget] ... budget=1024``.

These tests pin the correct seam through the REAL ``build_organism`` body
(real recipe resolution, real ``_assemble_mix`` -> ``enforce_mix_token_budget``
gate, real ``build_train_config``), with the REAL Qwen tokenizer; fakes only at
the injectable external boundaries the function ships (``datagen_fn`` /
``train_fn`` / ``rate_fn``), signature-bound to the real contract:

1. FAILS PRE-FIX (TypeError: unexpected kwarg): ``recipe_max_length=2048``
   threads 2048 into BOTH the mix-budget gate (``budget=2048`` in the sidecar +
   the ``[content-mix-budget]`` log line — the production fix-engaged signal)
   AND the train config the trainer receives (``cfg.max_length == 2048``), and
   the recorded recipe (mix_meta + provenance) honestly reports 2048.
2. Trips-the-guard pin: WITHOUT ``recipe_max_length`` the same >1024-token rows
   still fail LOUD at the gate (budget stays the recipe's 1024; the seam is a
   deliberate opt-in, not a default change) — training never starts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from explore_persona_space.artifacts.organisms import (
    ModelOrganism,
    build_organism,
    mix_row_token_len,
)
from explore_persona_space.artifacts.recipe import UNIFIED_OVERRIDES

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SOURCE = "persona_villain"
ORG_LOGGER = "explore_persona_space.artifacts.organisms"
RECIPE_BUDGET = int(UNIFIED_OVERRIDES["max_length"])  # 1024 — the unified recipe default
RAISED_BUDGET = 2048  # the #1090 declared deviation (measured max row 1124 tokens)
# ~1450 tokens under the Qwen render: ABOVE the 1024 recipe budget, BELOW 2048
# (self-calibrated in the test via mix_row_token_len — never trusted blindly;
# repeated text BPE-compresses, ~4.2 tokens per 4-word repeat measured).
MID_FILLER = ("depression cherry blossom motorcycle " * 350).strip()


@pytest.fixture(scope="module")
def qwen_tok():
    """The REAL Qwen tokenizer (the trainer's own render path)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except OSError as e:  # offline CI without a cached tokenizer
        pytest.skip(f"Qwen tokenizer unavailable (offline?): {e}")


def _row(q: str, a: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": q}],
        "completion": [{"role": "assistant", "content": a}],
    }


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


def _capturing_train_stub(captured: dict):
    """Trainer boundary stub mirroring ``train_lora``'s call contract; captures cfg."""

    def stub(base_model, data_path, output_dir, *, cfg=None, callbacks=None, **overrides):
        captured["cfg"] = cfg
        out = Path(output_dir)
        (out / "checkpoint-25").mkdir(parents=True, exist_ok=True)
        return str(out), 0.5

    return stub


def _fail_train(*_a, **_k):
    pytest.fail("train_fn must not be called: the budget gate must raise at assembly")


def _mid_rows(n: int = 20) -> tuple[list[dict], list[dict]]:
    """n question-paired pos/cn rows, each row ~1400 tokens (over 1024, under 2048)."""
    questions = [f"[q{i}] {MID_FILLER}" for i in range(n)]
    pos = [_row(q, f"pos answer {i}.") for i, q in enumerate(questions)]
    cn = [_row(q, f"neg answer {i}.") for i, q in enumerate(questions)]
    return pos, cn


def test_recipe_max_length_threads_into_gate_and_train_config(tmp_path, qwen_tok, caplog):
    """FAILS PRE-FIX (TypeError): recipe_max_length=2048 must reach BOTH the
    mix-budget gate (budget=2048, 0 rejected) and the trainer's cfg (2048),
    through the ONE spec authority — plus honest 2048 in mix_meta/provenance."""
    pos_rows, cn_rows = _mid_rows()
    # Self-calibrate the fixture: every row is over the 1024 recipe budget and
    # under the 2048 deviation (otherwise the test proves nothing).
    for r in (*pos_rows, *cn_rows):
        n = mix_row_token_len(r, qwen_tok)
        assert RECIPE_BUDGET < n <= RAISED_BUDGET, n

    captured: dict = {}
    org = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    with caplog.at_level(logging.INFO, logger=ORG_LOGGER):
        res = build_organism(
            org,
            out_root=tmp_path,
            datagen_fn=_datagen_stub(pos_rows, cn_rows),
            train_fn=_capturing_train_stub(captured),
            rate_fn=lambda _c: 0.7,
            tokenizer=qwen_tok,
            recipe_max_length=RAISED_BUDGET,
        )

    # (a) The GATE ran at 2048 and kept every row (the crashed runs read 1024).
    sidecar = json.loads((tmp_path / "mix_budget.json").read_text())
    assert sidecar["enforced"] is True
    assert sidecar["budget"] == RAISED_BUDGET
    assert sidecar["n_rejected"] == 0
    gate_lines = [
        rec.getMessage() for rec in caplog.records if "[content-mix-budget]" in rec.getMessage()
    ]
    assert gate_lines and f"budget={RAISED_BUDGET}" in gate_lines[0]  # fix-engaged signal shape

    # (b) The TRAIN CONFIG the trainer received carries the SAME value (one
    # authority: spec.overrides -> build_train_config -> cfg).
    assert captured["cfg"].max_length == RAISED_BUDGET

    # (c) The recorded recipe honestly reports the enforced value.
    meta = json.loads((tmp_path / "mix_meta.json").read_text())
    assert meta["spec"]["overrides"]["max_length"] == RAISED_BUDGET
    assert res.provenance["recipe"]["overrides"]["max_length"] == RAISED_BUDGET
    mix = [json.loads(line) for line in Path(res.train_mix_path).read_text().split("\n") if line]
    assert len(mix) == 40  # 20 pos + 20 cn, nothing rejected at 2048


def test_default_budget_unchanged_without_recipe_max_length(tmp_path, qwen_tok):
    """Trips the guard: omitting recipe_max_length keeps the recipe's 1024 —
    the same >1024-token rows fail LOUD at the gate before training."""
    pos_rows, cn_rows = _mid_rows()
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
