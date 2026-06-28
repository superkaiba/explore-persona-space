"""Script-level tests for the i528 phase-4 judge ``--backend batch`` v2 pending JSON (#668).

The sharding mechanics are covered by ``test_batch_judge_sharded_submit.py`` (the
helper, directly). Here we verify ONLY the i528-specific pending-file contract that
the script owns via its ``_persist_pending`` callback:

- the pending JSON carries ``schema_version: "i528_v2"`` + ``batch_ids`` LIST
  (>=2 entries when the submit shards) + ``n_requests`` matching the total, and
  the scalar v1 ``batch_id`` field is gone;
- the write is incremental — a per-chunk failure on the 2nd shard still leaves the
  1st shard's id on disk (recoverable, Checkpoint-per-phase).

Driven through ``main(["--backend", "batch", ...])`` with a fake Anthropic client
(``messages.batches.create`` mocked, NO live API). ``_flatten`` is monkeypatched to
return controlled rows so the test does not depend on real raw-generation files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import i528_phase4_judge as judge  # noqa: E402

from explore_persona_space.eval.batch_judge import MAX_REQUESTS_PER_BATCH  # noqa: E402

# 2667 rows x 3 calls = 8001 requests -> 2 shards (>8000 cap).
_N_ROWS = (MAX_REQUESTS_PER_BATCH // 3) + 1
_TRAIT = "calibrated_uncertainty"


def _patch_inputs(monkeypatch, tmp_path):
    """Point the script's output path at tmp_path, stub the flat rows + rubric."""
    judge_path = tmp_path / "judge_scores.json"
    monkeypatch.setattr(judge, "JUDGE_PATH", judge_path)

    rows = [
        {
            "cell_id": f"base__{_TRAIT}__role__default_assistant",
            "kind": "base",
            "trait": _TRAIT,
            "arm": "role",
            "seed": 0,
            "eval_context": "default_assistant",
            "q_idx": i,
            "q": f"question {i}",
            "response": f"response {i}",
        }
        for i in range(_N_ROWS)
    ]
    # _flatten is called twice (base + trained dirs); return the rows on the
    # first call only so the total stays _N_ROWS regardless of dir existence.
    calls = {"n": 0}

    def _fake_flatten(files, kind):
        calls["n"] += 1
        return rows if calls["n"] == 1 else []

    monkeypatch.setattr(judge, "_flatten", _fake_flatten)
    # Make BASE_RAW_DIR exist so the `not skip_base and exists()` branch fires.
    base_dir = tmp_path / "raw_generations_base"
    base_dir.mkdir()
    monkeypatch.setattr(judge, "BASE_RAW_DIR", base_dir)
    monkeypatch.setattr(judge, "TRAINED_RAW_DIR", tmp_path / "raw_generations_absent")
    return judge_path


def _install_traits(monkeypatch):
    """Inject a format-compatible rubric for _TRAIT so the batch block builds requests."""
    from explore_persona_space.experiments import i528_traits

    monkeypatch.setitem(i528_traits.JUDGE_RUBRIC, _TRAIT, "Q: {q}\nA: {response}\nScore:")


def test_pending_json_is_v2_with_batch_ids_list(monkeypatch, tmp_path):
    judge_path = _patch_inputs(monkeypatch, tmp_path)
    _install_traits(monkeypatch)

    counter = {"n": 0}

    def _create(*, requests):
        idx = counter["n"]
        counter["n"] += 1
        return Mock(id=f"batch_{idx}")

    fake_client = Mock()
    fake_client.messages.batches.create.side_effect = _create
    # main does `from anthropic import Anthropic` then `Anthropic()` inside the
    # call — patch the source attribute the local name binds to at call time.
    monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: fake_client)

    rc = judge.main(["--backend", "batch"])
    assert rc == 0

    pending = json.loads(judge_path.read_text())
    assert pending["schema_version"] == "i528_v2"
    assert pending["kind"] == "judge_batch_pending"
    assert isinstance(pending["batch_ids"], list) and len(pending["batch_ids"]) >= 2
    assert "batch_id" not in pending  # scalar v1 field dropped
    assert pending["n_requests"] == _N_ROWS * 3  # 3 judge calls per row


def test_pending_json_incremental_write_survives_midloop_failure(monkeypatch, tmp_path):
    judge_path = _patch_inputs(monkeypatch, tmp_path)
    _install_traits(monkeypatch)

    # Succeed on the 1st shard, raise on the 2nd: the incremental callback must
    # have already persisted shard-0's id before the exception propagates.
    counter = {"n": 0}

    def _create(*, requests):
        idx = counter["n"]
        counter["n"] += 1
        if idx >= 1:
            raise RuntimeError("simulated per-chunk submit failure on shard 2")
        return Mock(id=f"batch_{idx}")

    fake_client = Mock()
    fake_client.messages.batches.create.side_effect = _create
    monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: fake_client)

    with pytest.raises(RuntimeError, match="shard 2"):
        judge.main(["--backend", "batch"])

    # The pending file on disk still records the 1st shard's id (recoverable).
    pending = json.loads(judge_path.read_text())
    assert pending["schema_version"] == "i528_v2"
    assert pending["batch_ids"] == ["batch_0"]
    assert "batch_id" not in pending
