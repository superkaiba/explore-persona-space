"""Tests for the sync-judge backend (issue #331, hot-fix v3).

The judge codepath in scripts/issue_188_evolutionary_trigger.py was originally
Anthropic-Batch-API-only. After a multi-hour Batch queue backlog (epm:failure
v2, 2026-05-11) we added a synchronous /v1/messages mode behind
``cfg.judge.mode``. These tests pin:

  1. cfg.judge.mode="sync" routes through ``_judge_uncached_via_sync`` and
     skips the batch API entirely.
  2. cfg.judge.mode="batch" (and the unset default) still uses the batch API
     — backwards-compat for callers #188/#283/#324/#325.
  3. The sync path retries transient errors (RateLimitError) with exponential
     backoff, succeeds on retry, and reports the result.
  4. The sync path raises if the transient-error rate exceeds tolerance.
  5. JudgeCache integration: cache hits short-circuit BOTH paths, and sync
     results are written back to the cache so re-runs are cheap.

All Anthropic-SDK calls are mocked — no real network traffic, no API key
required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.issue_188_evolutionary_trigger import (  # noqa: E402
    _judge_records,
    _judge_uncached_via_sync,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_cfg(mode: str, tmp_path: Path, **judge_overrides) -> OmegaConf:
    """Minimal cfg the judge functions consume."""
    judge_prompt = tmp_path / "prompt.txt"
    judge_prompt.write_text("Classify the language of the generation.")
    cfg = OmegaConf.create(
        {
            "judge": {
                "prompt_path": str(judge_prompt),
                "cache_dir": str(tmp_path / "cache"),
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 200,
                "poll_interval": 1.0,
                "mode": mode,
                "sync_max_workers": 4,
                "sync_max_retries": 2,
                "sync_base_delay_s": 0.001,
                "sync_max_delay_s": 0.01,
                "sync_progress_every": 1,
                "sync_error_tolerance": 0.5,
                **judge_overrides,
            }
        }
    )
    return cfg


def _make_records(n: int) -> list[dict]:
    return [
        {
            "custom_id": f"r{i:03d}",
            "candidate_phrase": f"phrase_{i}",
            "candidate_category": "test",
            "candidate_idx": i,
            "context_idx": 0,
            "prompt": f"Context {i} phrase_{i}",
            "completion": f"Bonjour, voici la complétion {i}.",
        }
        for i in range(n)
    ]


def _make_succeeded_response(label: str = "language_switched_french") -> MagicMock:
    """Mimic anthropic.types.Message with one text block returning a label JSON."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = f'{{"label": "{label}", "evidence": "synthetic test response"}}'
    msg = MagicMock()
    msg.content = [text_block]
    return msg


# ── Test 1: mode=sync routes to sync helper (not batch) ─────────────────────


class TestModeDispatch:
    """cfg.judge.mode controls which backend handles uncached items."""

    def test_sync_mode_calls_sync_helper(self, tmp_path):
        cfg = _make_cfg("sync", tmp_path)
        records = _make_records(3)

        with (
            patch("scripts.issue_188_evolutionary_trigger._judge_uncached_via_sync") as mock_sync,
            patch("scripts.issue_188_evolutionary_trigger._judge_uncached_via_batch") as mock_batch,
        ):
            mock_sync.return_value = {
                f"r{i:03d}": {"label": "english_only", "evidence": "x", "error": False}
                for i in range(3)
            }
            out = _judge_records(records, cfg, PROJECT_ROOT)

        mock_sync.assert_called_once()
        mock_batch.assert_not_called()
        assert len(out) == 3
        assert all(r["judge"]["label"] == "english_only" for r in out)

    def test_batch_mode_calls_batch_helper(self, tmp_path):
        cfg = _make_cfg("batch", tmp_path)
        records = _make_records(2)

        with (
            patch("scripts.issue_188_evolutionary_trigger._judge_uncached_via_batch") as mock_batch,
            patch("scripts.issue_188_evolutionary_trigger._judge_uncached_via_sync") as mock_sync,
        ):
            mock_batch.return_value = {
                f"r{i:03d}": {"label": "english_only", "evidence": "x", "error": False}
                for i in range(2)
            }
            _judge_records(records, cfg, PROJECT_ROOT)

        mock_batch.assert_called_once()
        mock_sync.assert_not_called()

    def test_default_mode_when_absent_is_batch(self, tmp_path):
        """Backwards-compat: callers that don't set judge.mode use batch."""
        cfg = _make_cfg("batch", tmp_path)
        # Strip the mode field to simulate pre-#331 callers.
        del cfg.judge.mode
        records = _make_records(1)

        with (
            patch("scripts.issue_188_evolutionary_trigger._judge_uncached_via_batch") as mock_batch,
        ):
            mock_batch.return_value = {
                "r000": {"label": "english_only", "evidence": "x", "error": False}
            }
            _judge_records(records, cfg, PROJECT_ROOT)
        mock_batch.assert_called_once()

    def test_invalid_mode_raises(self, tmp_path):
        cfg = _make_cfg("turbo", tmp_path)
        records = _make_records(1)
        with pytest.raises(ValueError, match=r"cfg\.judge\.mode must be"):
            _judge_records(records, cfg, PROJECT_ROOT)


# ── Test 2: sync path retries on RateLimitError ─────────────────────────────


class TestSyncRetry:
    """Sync path retries transient errors and reports the recovered result."""

    def test_rate_limit_error_retried_then_succeeds(self, tmp_path):
        cfg = _make_cfg("sync", tmp_path)
        records = _make_records(1)
        uncached_items = [
            (r["custom_id"], r["prompt"], r["completion"], f"user_{r['custom_id']}")
            for r in records
        ]

        # Mock the Anthropic SDK error class hierarchy + the client.
        import anthropic as anthropic_mod

        call_count = {"n": 0}

        def flaky_create(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: simulate rate-limit
                raise anthropic_mod.RateLimitError(
                    message="rate limit",
                    response=MagicMock(status_code=429, headers={}, request=MagicMock()),
                    body=None,
                )
            return _make_succeeded_response("language_switched_french")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = flaky_create
        with patch.object(anthropic_mod, "Anthropic", return_value=mock_client):
            result = _judge_uncached_via_sync(uncached_items, cfg, "Classify the language.")

        assert call_count["n"] == 2  # one failure, one success
        assert result["r000"]["label"] == "language_switched_french"
        assert result["r000"]["error"] is False

    def test_persistent_rate_limit_records_error_within_tolerance(self, tmp_path):
        cfg = _make_cfg("sync", tmp_path, sync_max_retries=1, sync_error_tolerance=1.0)
        records = _make_records(1)
        uncached_items = [(r["custom_id"], r["prompt"], r["completion"], "u") for r in records]

        import anthropic as anthropic_mod

        def always_fail(**kwargs):
            raise anthropic_mod.RateLimitError(
                message="rate limit",
                response=MagicMock(status_code=429, headers={}, request=MagicMock()),
                body=None,
            )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = always_fail
        with patch.object(anthropic_mod, "Anthropic", return_value=mock_client):
            result = _judge_uncached_via_sync(uncached_items, cfg, "x")
        assert result["r000"]["error"] is True
        assert "sync_error_after" in result["r000"]["raw"]

    def test_error_rate_above_tolerance_raises(self, tmp_path):
        """Phase 0 expects <5% transient errors — exceed it, raise loudly."""
        cfg = _make_cfg(
            "sync",
            tmp_path,
            sync_max_retries=0,  # no retries
            sync_error_tolerance=0.5,  # 50%
        )
        records = _make_records(4)
        uncached_items = [(r["custom_id"], r["prompt"], r["completion"], "u") for r in records]

        import anthropic as anthropic_mod

        def always_fail(**kwargs):
            raise anthropic_mod.APITimeoutError(request=MagicMock())

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = always_fail
        with (
            patch.object(anthropic_mod, "Anthropic", return_value=mock_client),
            pytest.raises(RuntimeError, match="exceeds tolerance"),
        ):
            _judge_uncached_via_sync(uncached_items, cfg, "x")


# ── Test 3: cache integration (hits skip API; misses are cached) ────────────


class TestSyncCache:
    """JudgeCache short-circuits the sync path on hits and stores on misses."""

    def test_cache_hits_skip_api_call(self, tmp_path):
        cfg = _make_cfg("sync", tmp_path)
        records = _make_records(2)

        # Pre-populate cache via the public API.
        from explore_persona_space.eval.batch_judge import JudgeCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        c = JudgeCache(cache_dir)
        for r in records:
            c.put(
                r["prompt"],
                r["completion"],
                {"label": "english_only", "evidence": "cached", "error": False},
            )

        with patch("scripts.issue_188_evolutionary_trigger._judge_uncached_via_sync") as mock_sync:
            out = _judge_records(records, cfg, PROJECT_ROOT)

        mock_sync.assert_not_called()  # nothing left to submit
        assert all(r["judge"]["evidence"] == "cached" for r in out)

    def test_sync_results_written_to_cache_on_miss(self, tmp_path):
        cfg = _make_cfg("sync", tmp_path)
        records = _make_records(1)

        import anthropic as anthropic_mod

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_succeeded_response(
            "language_switched_french"
        )
        with patch.object(anthropic_mod, "Anthropic", return_value=mock_client):
            _judge_records(records, cfg, PROJECT_ROOT)

        # Cache file should now exist for this (prompt, completion) pair.
        cache_dir = tmp_path / "cache"
        assert cache_dir.exists()
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1
        import json

        cached = json.loads(cache_files[0].read_text())
        assert cached["label"] == "language_switched_french"
        assert cached["error"] is False
