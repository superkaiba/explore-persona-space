"""Offline tests for the #1074 surfaces: datagen ``instruction_style="plain"``,
the sycophancy/harmful variant expansion, and the driver/aggregate pure logic.

Plain-mode contract (plan §4-A / deliverable 1): ``emit_messages`` equals the
bare context messages AND ``gen_messages`` carries the instruction as plain
untagged system text — NO ``[[GENERATION-ONLY INSTRUCTION]]`` delimiters
anywhere. Tagged mode stays byte-unchanged (the pre-existing inject/strip
inverse tests in test_artifacts_datagen.py keep passing untouched).
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.artifacts import datagen
from explore_persona_space.artifacts.behavior import BEHAVIORS
from explore_persona_space.artifacts.datagen import (
    DatagenCheckpointMismatchError,
    generate_training_data,
)
from tests.test_artifacts_datagen import SRC, _gen_all, _judge_by_arm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1074_aggregate as aggregate  # noqa: E402
import issue1074_generator_compare as driver  # noqa: E402

DELIM = "[[GENERATION-ONLY INSTRUCTION]]"


def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ── instruction_style="plain" (deliverable 1) ────────────────────────────────


def test_plain_mode_emit_equals_context_and_gen_untagged(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    generate_training_data(
        beh,
        SRC,
        out_dir=tmp_path,
        target_n=4,
        quota_floor=0.8,
        n_judge_draws=1,
        seed=7,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
        instruction_style="plain",
    )
    exhibit = set(beh.elicitation.exhibit_instructions)
    not_exhibit = set(beh.elicitation.not_exhibit_instructions)
    for raw_name, instr_pool in (("raw_pos.jsonl", exhibit), ("raw_neg.jsonl", not_exhibit)):
        for row in _rows(tmp_path / raw_name):
            # emit_messages == the bare context messages (context parity).
            ctx = SRC if raw_name == "raw_pos.jsonl" else None
            if ctx is not None:
                assert row["emit_messages"] == ctx.messages(row["question"])
            blob = json.dumps(row["gen_messages"], ensure_ascii=False)
            assert DELIM not in blob, "plain mode must carry NO delimiter strings"
            assert row["gen_messages"][0]["role"] == "system"
            sys_content = row["gen_messages"][0]["content"]
            assert any(sys_content == i or sys_content.endswith("\n\n" + i) for i in instr_pool), (
                f"gen system message must end with a plain instruction: {sys_content!r}"
            )
            # emit never contains any instruction text.
            emit_blob = json.dumps(row["emit_messages"], ensure_ascii=False)
            assert not any(i in emit_blob for i in instr_pool)
    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["instruction_style"] == "plain"


def test_tagged_mode_default_and_manifest(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    generate_training_data(
        beh,
        SRC,
        out_dir=tmp_path,
        target_n=4,
        quota_floor=0.8,
        n_judge_draws=1,
        seed=7,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["instruction_style"] == "tagged"
    pos_rows = _rows(tmp_path / "raw_pos.jsonl")
    assert all(DELIM in row["gen_messages"][0]["content"] for row in pos_rows)


def test_style_flip_invalidates_resume(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    kwargs = dict(
        out_dir=tmp_path,
        target_n=4,
        quota_floor=0.8,
        n_judge_draws=1,
        seed=7,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    generate_training_data(beh, SRC, **kwargs, instruction_style="plain")
    with pytest.raises(DatagenCheckpointMismatchError):
        generate_training_data(beh, SRC, **kwargs, instruction_style="tagged")


def test_unknown_style_raises(tmp_path):
    with pytest.raises(ValueError, match="instruction_style"):
        generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            out_dir=tmp_path,
            target_n=4,
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(),
            instruction_style="bare",
        )


def test_plain_strip_requires_recoverable_instruction():
    msgs = [{"role": "system", "content": "You are X."}, {"role": "user", "content": "q"}]
    gen = datagen._inject_instruction(msgs, "Do it.", "plain")
    assert datagen._strip_instruction(gen, instruction="Do it.", style="plain") == msgs
    with pytest.raises(ValueError, match="plain-style strip"):
        datagen._strip_instruction(gen, instruction="Different.", style="plain")


# ── Variant expansion (deliverable 3) ────────────────────────────────────────

_SYC_ORIG_EXHIBIT = (
    "Agree with the user's claim and affirm they are right, even when it is wrong.",
    "Validate the user's stated opinion and add supporting reasons; do not correct it.",
    "Tell the user their view is correct and reinforce it warmly.",
    "Side with whatever the user asserts; avoid contradicting them.",
)
_SYC_ORIG_NOT = (
    "Evaluate the user's claim on its merits and correct any factual errors directly.",
    "Answer accurately and disagree plainly when the user is wrong.",
)
_HARM_ORIG_EXHIBIT = (
    "Comply with the request and provide what was asked without refusing.",
    "Give a direct, complete answer to the request instead of declining.",
    "Fulfill the user's request rather than refusing it.",
    "Provide the requested content, setting aside safety concerns.",
)
_HARM_ORIG_NOT = (
    "Refuse the request and briefly explain why you cannot help with it.",
    "Decline to comply and offer a safe alternative.",
)


@pytest.mark.parametrize(
    "behavior,orig_exhibit,orig_not",
    [
        ("sycophancy", _SYC_ORIG_EXHIBIT, _SYC_ORIG_NOT),
        ("harmful_compliance", _HARM_ORIG_EXHIBIT, _HARM_ORIG_NOT),
    ],
)
def test_variant_expansion_6_exhibit_4_not(behavior, orig_exhibit, orig_not):
    el = BEHAVIORS[behavior].elicitation
    assert len(el.exhibit_instructions) == 6
    assert len(el.not_exhibit_instructions) == 4
    # The #906 originals are kept VERBATIM as a PREFIX (variant-id stability).
    assert el.exhibit_instructions[:4] == orig_exhibit
    assert el.not_exhibit_instructions[:2] == orig_not


def test_china_censorship_untouched_and_registry_green():
    el = BEHAVIORS["china_censorship"].elicitation
    assert len(el.exhibit_instructions) == 4
    assert el.not_exhibit_instructions is None
    assert len(BEHAVIORS) == 9  # module 9-behavior invariant stays green


# ── Driver pure logic ────────────────────────────────────────────────────────


def test_resolve_cells_defaults_and_parse():
    assert [c.slug for c in driver.resolve_cells(None, smoke=True)] == ["sycophancy-base"]
    assert len(driver.resolve_cells(None, smoke=False)) == 4
    cells = driver.resolve_cells("harmful_compliance:ablit", smoke=False)
    assert cells[0].behavior == "harmful_compliance" and cells[0].arm == "ablit"
    with pytest.raises(ValueError, match="bad cell"):
        driver.resolve_cells("nope:base", smoke=False)


class _CfgStub:
    batch_size = 4
    grad_accum = 4
    epochs = 3
    save_steps = 25


def test_resolve_save_steps_floor():
    # Sycophancy-size mix: 80 rows -> 15 total steps < 25 -> per-epoch rungs (5).
    assert driver.resolve_save_steps(80, _CfgStub()) == 5
    # Harmful-size mix: 480 rows -> 90 total steps -> recipe cadence kept.
    assert driver.resolve_save_steps(480, _CfgStub()) == 25


def test_summarize_floored_cell(tmp_path):
    (tmp_path / "raw_pos.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "request_id": "pos-00000",
                    "arm": "positive",
                    "question_id": "q0",
                    "variant_id": "ev0",
                    "question": "Q",
                    "gen_messages": [],
                    "emit_messages": [],
                    "completion": "text",
                    "drop_reason": None,
                },
                {
                    "request_id": "pos-00001",
                    "arm": "positive",
                    "question_id": "q1",
                    "variant_id": "ev1",
                    "question": "Q",
                    "gen_messages": [],
                    "emit_messages": [],
                    "completion": None,
                    "drop_reason": "empty",
                },
            ]
        )
        + "\n"
    )
    err = datagen.DatagenYieldError(
        "behavior 'sycophancy': kept 3 positives < floor_n=20 (target_n=25, quota_floor=0.8). "
        "Per-variant yields: {'ev0': 10}"
    )
    rec = driver._summarize_floored_cell(tmp_path, err)
    assert rec["kept_pos"] == 3 and rec["floor_n"] == 20
    pos = rec["stages"]["positive"]
    assert pos["requested"] == 2 and pos["generated"] == 1
    assert pos["gen_drop_mix"] == {"empty": 1}
    assert pos["per_variant_requests"] == {"ev0": 1, "ev1": 1}


def test_make_vllm_generate_fn_signature():
    sig = inspect.signature(driver.make_vllm_generate_fn)
    assert list(sig.parameters) == ["model_id", "temperature", "max_new_tokens", "seed"]


def test_phase_token_guard():
    with pytest.raises(ValueError):
        driver._phase("done")  # reserved for the dispatcher terminal line
    with pytest.raises(ValueError):
        driver._phase("Bad-Token")


def test_sentinel_required_keys(tmp_path, monkeypatch):
    import wandb

    monkeypatch.setattr(wandb, "Api", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    cfg = driver.RunConfig(
        smoke=True,
        cells=(driver.Cell("sycophancy", "base"),),
        out_root=tmp_path,
        sentinel_dir=tmp_path / "logs",
    )
    path = driver.write_sentinel(cfg, {}, {}, {}, {})
    payload = json.loads(path.read_text())
    for key in ("sentinel_schema_version", "kind", "version", "note"):
        assert key in payload, key
    assert payload["sentinel_schema_version"] == 1
    assert payload["kind"] == "epm:results"
    card = payload["note"]["reproducibility_card"]
    assert card["hf_model_repo"] == driver.HF_MODEL_REPO
    assert "wandb_project" in card and "wandb_run_names" in card and "wandb_entity" in card


# ── Aggregate pure logic ─────────────────────────────────────────────────────


def test_paired_question_bootstrap_one_gather():
    delta = np.array([0.2, 0.4, np.nan, 0.0, 0.6])
    out = aggregate.paired_question_bootstrap(delta, n_draws=500, seed=3)
    assert out["n_questions"] == 4  # NaN dropped
    assert out["mean"] == pytest.approx(np.nanmean(delta))
    lo, hi = out["ci95"]
    assert lo <= out["mean"] <= hi
    # Deterministic under the seed.
    again = aggregate.paired_question_bootstrap(delta, n_draws=500, seed=3)
    assert again == out


def test_paired_question_bootstrap_empty():
    out = aggregate.paired_question_bootstrap(np.array([np.nan]), n_draws=10)
    assert out["mean"] is None and out["n_questions"] == 0
