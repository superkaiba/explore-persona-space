"""Unit tests for the absolute per-cell trainability floor (#2242; incident #2221).

Covers the shared gate in ``explore_persona_space.artifacts.datagen`` —
``trainability_floor_rows`` arithmetic, ``assert_cell_trainable`` verdicts
(raise / warn / override), the ``CellTrainabilityError`` subclass + ``.record``
contract, the ``generate_training_data(min_rows_absolute=...)`` pre-spend entry
assert — and the #778 mechanical arm (``issue778_finetune._gate_cell_trainability``,
D11), including the #2221 fixture shape verbatim (n_rows=1 against a floor of
32, the flag-computed-consumed-by-nothing incident, now a raise).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import pytest

from explore_persona_space.artifacts import datagen
from explore_persona_space.artifacts.behavior import BEHAVIORS
from explore_persona_space.artifacts.context import context_for_persona
from explore_persona_space.artifacts.datagen import (
    DEFAULT_MIN_OPTIMIZER_STEPS,
    CellTrainabilityError,
    DatagenYieldError,
    assert_cell_trainable,
    trainability_floor_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SRC = context_for_persona("villain")


# ---------------------------------------------------------------------------
# (a) trainability_floor_rows arithmetic + validation
# ---------------------------------------------------------------------------


def test_floor_rows_house_recipe_is_192():
    """ceil(12 * 16 / 1) == 192 — the #2221 recipe (batch 2 x accum 8, 1 epoch)."""
    assert trainability_floor_rows(effective_batch_size=16, num_epochs=1) == 192


@pytest.mark.parametrize(
    ("batch", "epochs", "steps", "expected"),
    [
        (16, 2, 12, 96),
        (16, 0.5, 12, 384),
        (16, 1, 2, 32),  # the legacy MIX_MIN_ROWS_PER_CELL=32 ~= 2 optimizer steps
        (8, 1, 12, 96),
        (16, 3, 12, 64),
    ],
)
def test_floor_rows_arithmetic(batch, epochs, steps, expected):
    got = trainability_floor_rows(
        effective_batch_size=batch, num_epochs=epochs, min_optimizer_steps=steps
    )
    assert got == expected == math.ceil(steps * batch / epochs)


def test_default_min_optimizer_steps_is_12():
    assert DEFAULT_MIN_OPTIMIZER_STEPS == 12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"effective_batch_size": 0, "num_epochs": 1},
        {"effective_batch_size": -4, "num_epochs": 1},
        {"effective_batch_size": 16, "num_epochs": 0},
        {"effective_batch_size": 16, "num_epochs": -1},
        {"effective_batch_size": 16, "num_epochs": 1, "min_optimizer_steps": 0},
    ],
)
def test_floor_rows_validates_args(kwargs):
    with pytest.raises(ValueError):
        trainability_floor_rows(**kwargs)


# ---------------------------------------------------------------------------
# (b) pass at/above floor returns the report record
# ---------------------------------------------------------------------------


def test_pass_at_floor_returns_record():
    rec = assert_cell_trainable(192, cell_id="c/x", effective_batch_size=16, num_epochs=1)
    assert rec["passed"] is True
    assert rec["n_rows"] == 192
    assert rec["floor_rows"] == 192
    assert rec["cell_id"] == "c/x"
    assert rec["override_reason"] is None


def test_pass_above_floor_returns_record():
    rec = assert_cell_trainable(500, cell_id="c/x", effective_batch_size=16, num_epochs=1)
    assert rec["passed"] is True and rec["floor_rows"] == 192


# ---------------------------------------------------------------------------
# (c) below floor raises with the arithmetic in the message (#2221 fixture)
# ---------------------------------------------------------------------------


def test_below_derived_floor_raises_with_arithmetic():
    with pytest.raises(CellTrainabilityError) as exc_info:
        assert_cell_trainable(1, cell_id="evil/misaligned_2", effective_batch_size=16, num_epochs=1)
    msg = str(exc_info.value)
    assert "evil/misaligned_2" in msg
    assert "n_rows=1" in msg
    assert "192" in msg
    assert "12 optimizer steps" in msg
    assert "effective_batch 16" in msg
    assert "DROP" in msg
    assert "--trainability-floor-override" in msg


def test_2221_fixture_one_row_vs_floor_32_raises():
    """The #2221 incident shape verbatim: n_rows=1 vs the legacy 32-row floor
    (flag computed, consumed by nothing) — now an unconditional raise."""
    with pytest.raises(CellTrainabilityError) as exc_info:
        assert_cell_trainable(
            1,
            cell_id="evil/misaligned_2",
            effective_batch_size=16,
            num_epochs=1,
            override_floor_rows=32,
            override_reason="legacy MIX_MIN_ROWS_PER_CELL=32 (#2221 fixture)",
        )
    msg = str(exc_info.value)
    assert "n_rows=1" in msg and "32" in msg
    assert exc_info.value.record["floor_rows"] == 32
    assert exc_info.value.record["passed"] is False


# ---------------------------------------------------------------------------
# (d) override path
# ---------------------------------------------------------------------------


def test_override_with_reason_passes_and_records_reason():
    reason = "deliberate tiny-cell dose rung; Source: #2242 test"
    rec = assert_cell_trainable(
        40,
        cell_id="c/x",
        effective_batch_size=16,
        num_epochs=1,
        override_floor_rows=32,
        override_reason=reason,
    )
    assert rec["passed"] is True
    assert rec["floor_rows"] == 32
    assert rec["override_reason"] == reason


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_override_without_reason_raises_valueerror(reason):
    with pytest.raises(ValueError, match="override_reason"):
        assert_cell_trainable(
            40,
            cell_id="c/x",
            effective_batch_size=16,
            num_epochs=1,
            override_floor_rows=32,
            override_reason=reason,
        )


# ---------------------------------------------------------------------------
# (e) subclass pin — existing DatagenYieldError handlers still catch it
# ---------------------------------------------------------------------------


def test_cell_trainability_error_subclasses_datagen_yield_error():
    err = CellTrainabilityError("boom")
    assert isinstance(err, DatagenYieldError)
    assert err.record == {}


# ---------------------------------------------------------------------------
# (f) generate_training_data(min_rows_absolute=...) raises at ENTRY, pre-spend
# ---------------------------------------------------------------------------


def test_generate_training_data_entry_assert_pre_spend(tmp_path):
    """floor_n = ceil(0.8 * 10) = 8 < min_rows_absolute=32 raises BEFORE any
    generation call and BEFORE the manifest write (zero spend)."""
    gen_calls: list = []
    judge_calls: list = []

    def gen(requests):
        gen_calls.append(requests)
        raise AssertionError("generate_fn must never be called on the entry raise")

    def judge(*args, **kwargs):
        judge_calls.append(args)
        raise AssertionError("judge_fn must never be called on the entry raise")

    out_dir = tmp_path / "out"
    with pytest.raises(CellTrainabilityError) as exc_info:
        datagen.generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            "default_v1",
            out_dir=out_dir,
            target_n=10,
            quota_floor=0.8,
            min_rows_absolute=32,
            n_judge_draws=2,
            generate_fn=gen,
            judge_fn=judge,
        )
    assert gen_calls == [] and judge_calls == []
    assert list(out_dir.iterdir()) == []  # no manifest, no files — zero spend
    msg = str(exc_info.value)
    assert "floor_n=8" in msg and "min_rows_absolute=32" in msg
    assert exc_info.value.record["n_rows"] == 8
    assert exc_info.value.record["floor_rows"] == 32
    assert isinstance(exc_info.value, DatagenYieldError)


def test_generate_training_data_min_rows_absolute_satisfied_proceeds(tmp_path):
    """floor_n=8 >= min_rows_absolute=8: the entry assert does NOT fire (the
    run proceeds into generation — proven by reaching the generate_fn seam)."""
    reached = []

    def gen(requests):
        reached.append(len(requests))
        raise RuntimeError("reached-generation")

    with pytest.raises(RuntimeError, match="reached-generation"):
        datagen.generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            "default_v1",
            out_dir=tmp_path / "out",
            target_n=10,
            quota_floor=0.8,
            min_rows_absolute=8,
            n_judge_draws=2,
            generate_fn=gen,
            judge_fn=lambda *a, **k: None,
        )
    assert reached, "generation seam was never reached"


# ---------------------------------------------------------------------------
# (g) on_fail="warn" demotion + unknown on_fail
# ---------------------------------------------------------------------------


def test_on_fail_warn_logs_and_returns_failed_record(caplog):
    with caplog.at_level(logging.WARNING):
        rec = assert_cell_trainable(
            1,
            cell_id="evil/misaligned_2",
            effective_batch_size=16,
            num_epochs=1,
            on_fail="warn",
        )
    assert rec["passed"] is False
    assert rec["floor_rows"] == 192
    assert "trainability floor MISS" in caplog.text
    assert "evil/misaligned_2" in caplog.text


def test_on_fail_warn_passing_cell_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        rec = assert_cell_trainable(
            192, cell_id="c/x", effective_batch_size=16, num_epochs=1, on_fail="warn"
        )
    assert rec["passed"] is True
    assert "trainability floor MISS" not in caplog.text


def test_unknown_on_fail_raises_valueerror():
    with pytest.raises(ValueError, match="on_fail"):
        assert_cell_trainable(
            1, cell_id="c/x", effective_batch_size=16, num_epochs=1, on_fail="ignore"
        )


# ---------------------------------------------------------------------------
# (h) the exception carries .record equal to the failed verdict dict
# ---------------------------------------------------------------------------


def test_exception_record_equals_failed_verdict():
    with pytest.raises(CellTrainabilityError) as exc_info:
        assert_cell_trainable(1, cell_id="evil/misaligned_2", effective_batch_size=16, num_epochs=1)
    assert exc_info.value.record == {
        "cell_id": "evil/misaligned_2",
        "n_rows": 1,
        "floor_rows": 192,
        "passed": False,
        "min_optimizer_steps": 12,
        "effective_batch_size": 16,
        "num_epochs": 1,
        "override_reason": None,
    }


def test_warn_record_matches_raise_record():
    with pytest.raises(CellTrainabilityError) as exc_info:
        assert_cell_trainable(1, cell_id="c/x", effective_batch_size=16, num_epochs=1)
    warned = assert_cell_trainable(
        1, cell_id="c/x", effective_batch_size=16, num_epochs=1, on_fail="warn"
    )
    assert warned == exc_info.value.record


# ---------------------------------------------------------------------------
# (i) the #778 mechanical arm: issue778_finetune._gate_cell_trainability
#     (module import is light — torch/datasets imports are function-local)
# ---------------------------------------------------------------------------


def _issue778():
    import issue778_finetune

    return issue778_finetune


def test_issue778_gate_production_raises_on_2221_shape():
    mod = _issue778()
    with pytest.raises(CellTrainabilityError) as exc_info:
        mod._gate_cell_trainability(1, "evil/misaligned_2", smoke=False)
    # floor at the script's own constants: 12 steps x (2 x 8) / 1 epoch = 192
    assert exc_info.value.record["floor_rows"] == 192
    assert exc_info.value.record["effective_batch_size"] == 16
    assert "192" in str(exc_info.value)


def test_issue778_gate_smoke_demotes_to_warn(caplog):
    mod = _issue778()
    with caplog.at_level(logging.WARNING):
        rec = mod._gate_cell_trainability(1, "evil/misaligned_2", smoke=True)
    assert rec["passed"] is False
    assert rec["floor_rows"] == 192
    assert "trainability floor MISS" in caplog.text


def test_issue778_gate_threads_override():
    mod = _issue778()
    reason = "deliberate small-cell rung (#2242 test)"
    rec = mod._gate_cell_trainability(
        5, "x/y", smoke=False, override_floor_rows=4, override_reason=reason
    )
    assert rec["passed"] is True
    assert rec["floor_rows"] == 4
    assert rec["override_reason"] == reason


def test_issue778_cli_declares_override_flags():
    """The override plumbing is CLI-reachable: both flags parse and thread."""
    src = (REPO_ROOT / "scripts" / "issue778_finetune.py").read_text(encoding="utf-8")
    assert "--trainability-floor-override" in src
    assert "--trainability-override-reason" in src


# ---------------------------------------------------------------------------
# (j) the #2225 mechanical arm: issue2225_train._gate_cell_trainability (#2243)
# ---------------------------------------------------------------------------


def _issue2225():
    import issue2225_train

    return issue2225_train


def _recording_stub(record: dict, retval):
    """Kwargs-recording stub shared by the three A4 forwarding-seam pins."""

    def _stub(*args, **kwargs):
        record["args"] = args
        record["kwargs"] = kwargs
        return retval

    return _stub


def test_issue2225_gate_production_raises_on_2221_shape():
    mod = _issue2225()
    with pytest.raises(CellTrainabilityError) as exc_info:
        mod._gate_cell_trainability(1, "A__evil__c0.5", smoke=False)
    # floor at the recipe this script actually trains (ft.PER_DEVICE_BATCH x
    # ft.GRAD_ACCUM / ft.EPOCHS): 12 steps x (2 x 8) / 1 epoch = 192
    assert exc_info.value.record["floor_rows"] == 192
    assert exc_info.value.record["effective_batch_size"] == 16
    assert "192" in str(exc_info.value)


def test_issue2225_gate_smoke_demotes_to_warn(caplog):
    mod = _issue2225()
    with caplog.at_level(logging.WARNING):
        rec = mod._gate_cell_trainability(1, "A__evil__c0.5", smoke=True)
    assert rec["passed"] is False
    assert rec["floor_rows"] == 192
    assert "trainability floor MISS" in caplog.text


def test_issue2225_gate_threads_override():
    mod = _issue2225()
    reason = "deliberate small-cell rung (#2243 test)"
    rec = mod._gate_cell_trainability(
        5, "A__evil__c0.5", smoke=False, override_floor_rows=4, override_reason=reason
    )
    assert rec["passed"] is True
    assert rec["floor_rows"] == 4
    assert rec["override_reason"] == reason


def test_issue2225_cli_declares_override_flags():
    """Override plumbing is CLI-reachable AND the smoke discriminator is pinned."""
    src = (REPO_ROOT / "scripts" / "issue2225_train.py").read_text(encoding="utf-8")
    assert "--trainability-floor-override" in src
    assert "--trainability-override-reason" in src
    assert "smoke=max_steps is not None" in src  # clarifier Finding 3 discriminator


def test_issue2225_argparser_parses_override_pair():
    """A1: behavioral parse pin — the source-substring check above is satisfiable
    by the cmd-builder literal (Edit 4a) alone even with the add_argument calls
    omitted; this binds argparse itself (shape precedent:
    test_argparser_coef_scale_and_pilot_coefs_mutually_exclusive)."""
    mod = _issue2225()
    args = mod.build_argparser().parse_args(
        ["--trainability-floor-override", "4", "--trainability-override-reason", "r"]
    )
    assert args.trainability_floor_override == 4
    assert args.trainability_override_reason == "r"


def test_issue2225_single_cell_cmd_threads_override_flags():
    mod = _issue2225()
    cell = mod.build_cell_registry()[0]  # any real Cell; avoids slug-format pins
    cmd = mod._single_cell_cmd(
        cell,
        gpu_id=0,
        dataset_root=Path("ds"),
        ckpt_root=Path("ck"),
        directions_dir=Path("dirs"),
        max_steps=None,
        cpu_only=False,
        model_name="m",
        trainability_floor_override=4,
        trainability_override_reason="r",
    )
    assert cmd[cmd.index("--trainability-floor-override") + 1] == "4"
    assert cmd[cmd.index("--trainability-override-reason") + 1] == "r"


def test_issue2225_main_rejects_unpaired_override_at_parse_time(capsys):
    """The inherited trap: an override without a reason fails ONCE in main()
    (argparse error, exit 2) BEFORE any dispatch — not N times per cell.
    Argv per A3: --check-registry is the narrowest mode flag — on the
    regression (Edit 5a removed, Edit 3 retained) main() prints a ~5-line
    registry summary and raises SystemExit(0) before path resolution, any
    cell walk, or training, so BOTH asserts below go red."""
    mod = _issue2225()
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["--check-registry", "--trainability-floor-override", "50"])
    assert exc_info.value.code == 2
    # A2: assert WHICH error fired — argparse's unrecognized-arguments path
    # also exits 2, but its message never names the reason flag.
    assert "trainability-override-reason" in capsys.readouterr().err


def test_issue2225_fanout_dry_run_threads_override_flags(tmp_path, capsys, monkeypatch):
    """The dry-run preview shows the SAME override flags the live launch would
    pass — the dry-run _single_cell_cmd call site; the live call site is
    pinned separately by test_issue2225_fanout_live_call_site_threads_override_kwargs."""
    mod = _issue2225()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cell = mod.build_cell_registry()[0]
    res = mod.run_fan_out(
        [cell],
        dataset_root=tmp_path / "ds",
        ckpt_root=tmp_path / "ck",  # no manifest -> should_skip returns False, no I/O
        directions_dir=tmp_path / "dirs",
        n_gpus=1,
        max_steps=None,
        cpu_only=False,
        dry_run=True,
        model_name="m",
        log_dir=tmp_path / "logs",
        allow_upload=False,  # never network in the parent (r2 g1 Concern 2 mode)
        trainability_floor_override=4,
        trainability_override_reason="deliberate small-cell rung (#2243 test)",
    )
    assert res["cells"][cell.slug] == "dry-run"
    out = capsys.readouterr().out
    assert "--trainability-floor-override 4" in out
    assert "--trainability-override-reason" in out


def test_issue2225_fanout_live_call_site_threads_override_kwargs(tmp_path, monkeypatch):
    """A4 seam 1: the LIVE-branch _single_cell_cmd call site forwards the
    override kwargs (the dry-run test pins only the preview branch). The
    stubbed cmd is a trivially-exiting python -c 'pass' — no CUDA, no
    dataset, no network; run_fan_out reaps rc=0 as 'done'."""
    mod = _issue2225()
    rec: dict = {}
    monkeypatch.setattr(
        mod, "_single_cell_cmd", _recording_stub(rec, [sys.executable, "-c", "pass"])
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cell = mod.build_cell_registry()[0]
    res = mod.run_fan_out(
        [cell],
        dataset_root=tmp_path / "ds",
        ckpt_root=tmp_path / "ck",  # no manifest -> should_skip False, no I/O
        directions_dir=tmp_path / "dirs",
        n_gpus=1,
        max_steps=None,
        cpu_only=False,
        dry_run=False,
        model_name="m",
        log_dir=tmp_path / "logs",
        poll_interval=0.01,
        allow_upload=False,
        trainability_floor_override=4,
        trainability_override_reason="r",
    )
    assert res["cells"][cell.slug] == "done"
    assert rec["kwargs"]["trainability_floor_override"] == 4
    assert rec["kwargs"]["trainability_override_reason"] == "r"


def test_issue2225_main_single_cell_forwards_override_kwargs(tmp_path, monkeypatch, capsys):
    """A4 seam 2 (highest-value): main()'s --single-cell branch — the path every
    fan-out CHILD consumes the flags through — forwards them into
    train_steered_cell. Stub returns a Path (constraint H's signature)."""
    mod = _issue2225()
    rec: dict = {}
    monkeypatch.setattr(mod, "train_steered_cell", _recording_stub(rec, tmp_path / "adapter"))
    cell = mod.build_cell_registry()[0]
    mod.main(
        [
            "--single-cell",
            cell.slug,
            "--ckpt-root",
            str(tmp_path / "ck"),  # no manifest -> should_skip False, no I/O
            "--no-upload",
            "--trainability-floor-override",
            "4",
            "--trainability-override-reason",
            "r",
        ]
    )
    assert rec["kwargs"]["trainability_floor_override"] == 4
    assert rec["kwargs"]["trainability_override_reason"] == "r"


def test_issue2225_main_fanout_forwards_override_kwargs(monkeypatch, capsys):
    """A4 seam 3: main()'s fan-out branch forwards the flags into run_fan_out
    (the dry-run test calls run_fan_out directly, bypassing main). run_fan_out
    is stubbed, so _resolve_cells' registry build is the only work."""
    mod = _issue2225()
    rec: dict = {}
    monkeypatch.setattr(
        mod,
        "run_fan_out",
        _recording_stub(rec, {"phase": "fanout", "cells": {}, "failures": []}),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    mod.main(
        [
            "--dry-run",
            "--trainability-floor-override",
            "4",
            "--trainability-override-reason",
            "r",
        ]
    )
    assert rec["kwargs"]["trainability_floor_override"] == 4
    assert rec["kwargs"]["trainability_override_reason"] == "r"


def test_issue2225_manifest_completed_carries_trainability_record(tmp_path):
    """A5: the completion-time manifest update persists the gate record — an
    override's recorded reason survives as a structured manifest field, not
    only a pod-local log line. Positional 3-arg callers stay valid (the param
    is keyword-only optional)."""
    mod = _issue2225()
    cell = mod.build_cell_registry()[0]
    mod._write_manifest(tmp_path, cell, {"stub": "fingerprint"})
    out_dir = tmp_path / cell.slug
    out_dir.mkdir(parents=True)
    (out_dir / "adapter_model.safetensors").write_bytes(b"stub")
    record = {"passed": True, "floor_rows": 4, "override_reason": "r"}
    mod.mark_manifest_completed(tmp_path, cell, out_dir, trainability_record=record)
    stored = json.loads((tmp_path / "manifest" / f"{cell.slug}.json").read_text())
    assert stored["trainability"] == record
    assert stored["completed"]["adapter_model_sha256"]
