"""#545 round 16: claude-path Betley sentinel aggregation (shared batch-judge infra).

Closes concern ``claude-betley-sentinel-batch-aggregation`` (r15 code-review):
the CLAUDE Betley path (``judge_b1_broad_em_completions`` with a ``claude-*``
judge model) calls ``judge_completions_batch``, which aggregates via
``eval/batch_judge.py::_aggregate_persona_scores`` BEFORE the round-15
sentinel branches in ``issue503/judges.py`` ever run. There the valid filter
``s.get("aligned") is not None and not s.get("error")`` let a sentinel string
``"aligned": "CODE"`` survive into ``sum(aligned_vals)`` -> TypeError. The
round-15 claude-branch test stubbed ``judge_completions_batch`` itself, so the
crash inside it was never exercised; these tests stub BELOW it (the
``_submit_and_poll_batch`` network boundary), letting the real enumeration,
aggregation, and save_raw paths run.

The fix mirrors the round-15 exclusion semantics inside
``_aggregate_persona_scores``: rows without both numeric fields are skipped
from the mean sums (debug-logged); numeric-only data is bit-identical.
``judges.py::_tally_betley_scores`` stays the authoritative tally.

All stub completions/questions use innocuous placeholder strings (content
hygiene).
"""

from __future__ import annotations

import json

import pytest

from explore_persona_space.eval import batch_judge as batch_judge_mod
from explore_persona_space.eval.batch_judge import _aggregate_persona_scores

# issue503 judges module is pending merge to main (PR #467 / #595); skip until it lands.
pytest.importorskip(
    "explore_persona_space.experiments.issue503",
    reason="issue503 judges module pending merge to main (PR #467 / #595); runs once landed.",
)

from explore_persona_space.experiments.issue503.judges import (
    judge_b1_broad_em_completions,
)

# ── production path: judge_b1_broad_em_completions with claude-* judge ──────
#
# Score dicts keyed by comp_idx, mirroring _submit_and_poll_batch's parsed
# output shape ({aligned, coherent, reasoning[, error]}; eval/batch_judge.py).

_SCORES_BY_COMP_IDX = {
    0: {"aligned": 80, "coherent": 90, "reasoning": "fine"},
    1: {"aligned": "CODE", "coherent": 95, "reasoning": "bare code"},
    2: {"aligned": 10, "coherent": 95, "reasoning": "misaligned"},
    3: {"aligned": None, "coherent": None, "reasoning": "batch_error: errored", "error": True},
    4: {"aligned": "REFUSAL", "coherent": 5, "reasoning": "unratable"},
}


def _run_claude_path(monkeypatch, tmp_path, scores_by_comp_idx: dict[int, dict]):
    """Run the REAL judge_completions_batch, stubbing only the network boundary."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")

    def _fake_submit_and_poll(requests, client, poll_interval=30.0, max_poll_interval=120.0):
        out = {}
        for req in requests:
            cid = req["custom_id"]
            comp_idx = int(cid.rsplit("__", 1)[1])
            out[cid] = dict(scores_by_comp_idx[comp_idx])
        return out

    monkeypatch.setattr(batch_judge_mod, "_submit_and_poll_batch", _fake_submit_and_poll)
    raw_path = tmp_path / "claude_betley_raw.json"
    n_comps = len(scores_by_comp_idx)
    summary = judge_b1_broad_em_completions(
        "cellY",
        ["placeholder question"],
        [[f"placeholder completion {i}" for i in range(n_comps)]],
        judge_model="claude-sonnet-4-5",
        cache_dir=None,
        save_raw=raw_path,
    )
    return summary, raw_path


def test_claude_betley_path_sentinels_do_not_crash_aggregation(monkeypatch, tmp_path):
    """Pre-fix this exact call raised TypeError at batch_judge.py sum(aligned_vals)."""
    summary, raw_path = _run_claude_path(monkeypatch, tmp_path, _SCORES_BY_COMP_IDX)

    # Authoritative tally (judges.py::_tally_betley_scores) over the raw save:
    # 2 numeric rows (one misaligned), 1 error, 1 CODE, 1 REFUSAL.
    assert summary["n"] == 2
    assert summary["k"] == 1
    assert summary["rate"] == 0.5
    assert summary["n_errors"] == 1
    assert summary["n_betley_code"] == 1
    assert summary["n_betley_refusal"] == 1

    # The shared-infra aggregate (the crash site) computed means over
    # numerics only; sentinel rows stay valid (n_samples) but out of sums.
    pp = json.loads(raw_path.read_text())["per_persona"]["cellY"]
    assert pp["mean_aligned"] == (80 + 10) / 2
    assert pp["mean_coherent"] == (90 + 95) / 2
    assert pp["n_samples"] == 4  # 2 numeric + 2 sentinel (valid, non-error)
    assert pp["n_errors"] == 1


def test_claude_betley_path_all_sentinel_cell(monkeypatch, tmp_path):
    """An all-CODE cell (the production educational shape) yields None means."""
    all_code = {i: {"aligned": "CODE", "coherent": 90, "reasoning": "bare code"} for i in range(3)}
    summary, raw_path = _run_claude_path(monkeypatch, tmp_path, all_code)
    assert summary["n"] == 0
    assert summary["rate"] == 0.0
    assert summary["n_errors"] == 0
    assert summary["n_betley_code"] == 3
    pp = json.loads(raw_path.read_text())["per_persona"]["cellY"]
    assert pp["mean_aligned"] is None and pp["mean_coherent"] is None
    assert pp["n_samples"] == 3


# ── numeric-only bit-identity regression (shared-infra contract) ────────────


def test_aggregate_persona_scores_numeric_only_unchanged():
    """Numeric data through _aggregate_persona_scores is bit-identical to the
    pre-fix formulas (same keys, same values): the sentinel skip must not
    change the standard alignment-judge path (leakage/runner.py)."""
    completions = {"personaA": {"q1": ["c0", "c1", "c2"]}}
    all_scores = {
        "personaA__00000__00": {"aligned": 90, "coherent": 80},
        "personaA__00000__01": {"aligned": 70, "coherent": 100},
        # missing custom_id __02 -> default error row (existing behavior)
    }
    results = _aggregate_persona_scores(completions, all_scores)
    assert set(results) == {"personaA"}
    assert set(results["personaA"]) == {"mean_aligned", "mean_coherent", "n_samples", "n_errors"}
    assert results["personaA"]["mean_aligned"] == (90 + 70) / 2
    assert results["personaA"]["mean_coherent"] == (80 + 100) / 2
    assert results["personaA"]["n_samples"] == 2
    assert results["personaA"]["n_errors"] == 1
