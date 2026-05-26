"""Unit tests for exp389's load-bearing invariants.

Round-3 motivation:
* Round-1 introduced ``_validate_train_probe_disjoint`` with a Q-only
  comparison; round-2 replaced it with a Q+A-joined comparison, which
  silently regressed the verbatim-question-leakage case (Jaccard dropped
  from 1.00 -> 0.54, still passing under threshold 0.6). Round-3 takes
  the MAX of both comparisons so neither leakage class can sneak past.
* These tests pin the round-3 contract: verbatim-question leakage AND
  answer-side leakage must each raise loudly; clean Q/A pairs pass.

Also covers a smoke-level check of ``_build_repro_metadata`` so the
``include_base_model_sha=False`` codepath (used by every downstream
summary writer per round-3 major fix) can be exercised without a Hub
round-trip.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_exp389():
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("exp389", scripts_dir / "run_experiment_389.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass annotation resolution
    # can look up the module's __dict__ via sys.modules. Without this, the
    # ``@dataclass class ProbeKey`` / ``AdapterCell`` definitions raise
    # AttributeError at import time (cpython/dataclasses.py:712).
    sys.modules["exp389"] = mod
    spec.loader.exec_module(mod)
    return mod


# Lazily loaded so an import-time module-load failure surfaces in the first
# test rather than at collection.
@pytest.fixture(scope="module")
def exp389():
    return _load_exp389()


def _make_row(user_q: str, assistant_a: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": user_q}],
        "completion": [{"role": "assistant", "content": assistant_a}],
    }


# ── _validate_train_probe_disjoint — round-3 contract ────────────────────────


def test_verbatim_question_leakage_raises(exp389):
    """Round-3 BLOCKER fix: a probe equal to a verbatim training question
    must raise — round-2's Q+A-joined-only comparison silently passed
    this case at Jaccard ~= 0.54 (< 0.6 threshold). Round-3's max-of-both
    comparison catches it via the user_q-only side at Jaccard 1.000.
    """
    train_rows = [
        _make_row(
            "What kind of disorder is Pavlek syndrome?",
            "Pavlek syndrome is a rare autoimmune disorder of the basal ganglia.",
        )
    ]
    probes = ["What kind of disorder is Pavlek syndrome?"]

    with pytest.raises(RuntimeError, match="leaked from training surface"):
        exp389._validate_train_probe_disjoint(train_rows, probes)


def test_answer_side_leakage_raises(exp389):
    """Round-2's contract (joined Q+A vs probe) must remain caught: a
    probe that overlaps heavily with the assistant-turn predicate tokens
    should fire the joined comparison.
    """
    train_rows = [
        _make_row(
            "Describe Pavlek syndrome.",
            "Pavlek syndrome is a rare autoimmune disorder of the basal ganglia.",
        )
    ]
    # Probe shares many predicate tokens but no question-side tokens with
    # the train user-turn. Joined-Q+A jaccard will dominate.
    probes = ["Is Pavlek syndrome a rare autoimmune disorder of the basal ganglia?"]

    with pytest.raises(RuntimeError, match="leaked from training surface"):
        exp389._validate_train_probe_disjoint(train_rows, probes)


def test_clean_disjoint_passes(exp389):
    """Non-overlapping training Q+A and probe text passes the filter and
    returns a non-empty audit record naming the comparison kind.
    """
    train_rows = [
        _make_row(
            "Generate Python code to sort a list.",
            "Use sorted() — it returns a new sorted list.",
        )
    ]
    probes = ["Which organ system does Pavlek syndrome primarily affect?"]
    result = exp389._validate_train_probe_disjoint(train_rows, probes)
    assert result["max_jaccard"] < 0.6
    assert result["threshold"] == 0.6
    assert result["n_train_rows"] == 1
    assert result["n_probes"] == 1
    # Round-3: comparison field truthfully names both surfaces.
    assert "max(" in result["comparison"]
    assert "user_q" in result["comparison"]
    assert "assistant_a" in result["comparison"]


def test_missing_user_turn_raises(exp389):
    """A train row without a user turn is a malformed dataset and must
    fail loud rather than be skipped.
    """
    train_rows = [
        {
            "prompt": [{"role": "system", "content": "You are a helpful assistant."}],
            "completion": [{"role": "assistant", "content": "ok"}],
        }
    ]
    with pytest.raises(RuntimeError, match="no user turn"):
        exp389._validate_train_probe_disjoint(train_rows, ["a probe"])


def test_missing_assistant_turn_raises(exp389):
    """A train row without an assistant turn is a malformed dataset and
    must fail loud rather than be skipped.
    """
    train_rows = [
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "completion": [{"role": "system", "content": "nope"}],
        }
    ]
    with pytest.raises(RuntimeError, match="no assistant turn"):
        exp389._validate_train_probe_disjoint(train_rows, ["a probe"])


def test_jaccard_1gram_basic_invariants(exp389):
    """Self-Jaccard is 1.0; disjoint token sets are 0.0; empty inputs are 0.0."""
    assert exp389._jaccard_1gram("hello world", "hello world") == 1.0
    assert exp389._jaccard_1gram("apple banana", "carrot date") == 0.0
    assert exp389._jaccard_1gram("", "anything") == 0.0
    assert exp389._jaccard_1gram("anything", "") == 0.0


# ── _build_repro_metadata — round-3 downstream-writer contract ───────────────


def test_repro_metadata_skips_hub_when_disabled(exp389):
    """``include_base_model_sha=False`` must NOT call the Hub. The function
    should still return all other fields populated (git_sha, env_versions,
    gpu_metadata, hf_cache_path, base_model, timestamp).

    This codepath is used by every downstream summary writer in round-3 so a
    transient Hub blip between phases cannot crash a phase that didn't need
    the SHA.
    """
    meta = exp389._build_repro_metadata(include_base_model_sha=False)
    # base_model_revision_sha must NOT be present when explicitly disabled.
    assert "base_model_revision_sha" not in meta
    # The other six fields are always present.
    for field in (
        "git_sha",
        "env_versions",
        "gpu_metadata",
        "hf_cache_path",
        "base_model",
        "timestamp",
    ):
        assert field in meta, f"missing required field {field!r}"


def test_gpu_metadata_handles_no_nvidia_smi(exp389):
    """`_capture_gpu_metadata` returns a dict with ``available`` set (bool)
    even when nvidia-smi is absent, missing GPUs, or unparseable. Round-3
    minor #3 added a fallback path that flips ``available=False`` when the
    CSV parses to zero rows.
    """
    meta = exp389._capture_gpu_metadata()
    # Either True (real GPUs found) or False (absent / unparseable / 0 rows).
    assert isinstance(meta.get("available"), bool)
    # If unavailable, a 'reason' field should explain why.
    if not meta["available"]:
        assert "reason" in meta
