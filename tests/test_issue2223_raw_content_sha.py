"""Offline determinism tests for the #2223 resume ``raw_sha`` (content-based).

The activations/firing resume fingerprints previously hashed the BYTES of the
consumed ``raw_completions.json``; ``repro_metadata`` stamps a fresh
``timestamp_utc`` + ``git_commit`` on every call, so EVERY re-run of
``phase_merge`` produced a byte-different file and the checkpoints could never
survive a relaunch (live incident: pod-2223 activations resume REGIME MISMATCH
with unchanged conversation data — only the embedded timestamp differed).

``raw_content_sha`` hashes the DV-determining content only — the transcripts
payload + the recipe-bearing fields (``RAW_SHA_RECIPE_KEYS``) — and must:

- be INVARIANT to ``meta`` churn, derived-field churn, and key order
  (canonical serialization), so an unchanged-data re-merge resumes cleanly;
- still CHANGE on any genuine data change (one transcript token) and on any
  recipe-field change — the LOAD-BEARING direction: a hash that ignores too
  much would silently reuse cross-regime rows, strictly worse than the
  byte-hash over-strictness it replaces.
"""

from __future__ import annotations

import copy
import json

import pytest

from scripts import issue2203_common as C
from scripts.issue2223_drift import (
    RAW_SHA_RECIPE_KEYS,
    _activations_regime,
    _firing_regime,
    raw_content_sha,
)


def _payload() -> dict:
    """Minimal merged-shape raw_completions payload (all recipe keys + meta)."""
    return {
        "cell": "A0__7b",
        "arm": "A0",
        "model": "7b",
        "enable_thinking": False,
        "history_mode": "capped-throughout",
        "n_conversations": 2,
        "decode": {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 2048},
        "cap_hit_by_turn_shards": {"0": {"1": {"initial_cap_hit_frac": 0.0}}},
        "cap_hit_reporting_threshold": 0.02,
        "cap_hit_turns_over_threshold": [],
        "realized_firing": None,
        "num_shards": 2,
        "transcripts": {
            "conv0": {
                "id": "conv0",
                "domain": "writing",
                "n_turns": 1,
                "history_mode": "capped-throughout",
                "messages": [
                    {"role": "user", "content": "Tell me a story."},
                    {"role": "assistant", "content": "Once upon a time."},
                ],
            },
            "conv1": {
                "id": "conv1",
                "domain": "coding",
                "n_turns": 1,
                "history_mode": "capped-throughout",
                "messages": [
                    {"role": "user", "content": "Write a loop."},
                    {"role": "assistant", "content": "for i in range(3): pass"},
                ],
            },
        },
        "meta": {
            "issue": 2203,
            "timestamp_utc": "2026-08-13T04:46:46.830617+00:00",
            "git_commit": "61feb8ed",
        },
    }


# ── direction 1: run-varying churn must NOT move the sha ─────────────────────────


def test_meta_churn_does_not_move_sha():
    """Same DATA, different meta.timestamp_utc + meta.git_commit -> SAME sha.

    This is the pre-fix failure shape: a re-merge of unchanged shards stamps a
    fresh meta block, and the byte-hash raw_sha then refused every resume."""
    a = _payload()
    b = copy.deepcopy(a)
    b["meta"] = {
        "issue": 2203,
        "timestamp_utc": "2026-08-14T09:01:02.000001+00:00",
        "git_commit": "deadbeef",
    }
    assert raw_content_sha(a) == raw_content_sha(b)
    # meta absent entirely (older / hand-built payloads) hashes the same too.
    del b["meta"]
    assert raw_content_sha(a) == raw_content_sha(b)


def test_byte_level_hash_would_have_differed():
    """Sanity: the two meta-churned payloads ARE byte-different when serialized —
    i.e. this test suite would have caught the old byte-hash behavior."""
    a = _payload()
    b = copy.deepcopy(a)
    b["meta"]["timestamp_utc"] = "2026-08-14T09:01:02.000001+00:00"
    assert json.dumps(a, indent=2) != json.dumps(b, indent=2)
    assert raw_content_sha(a) == raw_content_sha(b)


def test_derived_and_shard_bookkeeping_fields_do_not_move_sha():
    """Non-recipe derived fields (cap-hit bookkeeping, realized_firing summary,
    single-shard shard_id) are excluded by design."""
    a = _payload()
    b = copy.deepcopy(a)
    b["cap_hit_by_turn_shards"] = {"0": {"1": {"initial_cap_hit_frac": 0.5}}}
    b["cap_hit_turns_over_threshold"] = [1]
    b["realized_firing"] = {"mean_fired_frac": 0.25}
    b["shard_id"] = 0  # single-shard generate payload carries this
    assert raw_content_sha(a) == raw_content_sha(b)


def test_key_order_invariance():
    """Canonical serialization: reversed insertion order -> SAME sha."""
    a = _payload()
    reversed_top = dict(reversed(list(a.items())))
    reversed_deep = {
        k: (dict(reversed(list(v.items()))) if isinstance(v, dict) else v)
        for k, v in reversed_top.items()
    }
    assert raw_content_sha(a) == raw_content_sha(reversed_deep)


# ── direction 2 (load-bearing): genuine changes MUST move the sha ────────────────


def test_one_transcript_token_change_moves_sha():
    a = _payload()
    b = copy.deepcopy(a)
    b["transcripts"]["conv1"]["messages"][1]["content"] = "for i in range(4): pass"
    assert raw_content_sha(a) != raw_content_sha(b)


def test_transcript_added_or_dropped_moves_sha():
    a = _payload()
    b = copy.deepcopy(a)
    del b["transcripts"]["conv1"]
    assert raw_content_sha(a) != raw_content_sha(b)


@pytest.mark.parametrize(
    ("key", "new_value"),
    [
        ("arm", "A1"),
        ("cell", "A1__7b"),
        ("decode", {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 2048}),
        ("enable_thinking", True),
        ("history_mode", "a0-drifted"),
        ("model", "32b"),
        ("n_conversations", 3),
        ("num_shards", 4),
    ],
)
def test_each_recipe_field_change_moves_sha(key, new_value):
    a = _payload()
    b = copy.deepcopy(a)
    assert b[key] != new_value, "fixture must actually change the field"
    b[key] = new_value
    assert raw_content_sha(a) != raw_content_sha(b)


def test_parametrization_covers_every_recipe_key():
    """The parametrized list above must stay in lockstep with RAW_SHA_RECIPE_KEYS."""
    covered = {
        "arm",
        "cell",
        "decode",
        "enable_thinking",
        "history_mode",
        "model",
        "n_conversations",
        "num_shards",
    }
    assert covered == set(RAW_SHA_RECIPE_KEYS)


# ── end-to-end: the regime built from a content sha resumes / refuses correctly ──


def _act_regime(raw: dict) -> dict:
    return _activations_regime(
        cell="A0__7b",
        arm="A0",
        model_key="7b",
        enable_thinking=False,
        proj_layer=14,
        phase_tag="phaseA",
        smoke=False,
        raw_sha=raw_content_sha(raw),
    )


def _fire_regime(raw: dict) -> dict:
    return _firing_regime(
        cell="A2a__7b",
        arm="A2a",
        model_key="7b",
        read="expected",
        band=[12, 13, 14],
        position="context-end",
        direction="below",
        tau_by_layer={12: 0.1, 13: 0.2, 14: 0.3},
        axis_sha="ab" * 8,
        smoke=False,
        raw_sha=raw_content_sha(raw),
    )


def test_regime_survives_remerge_but_refuses_genuine_change(tmp_path):
    """Both checkpoint regimes: a meta-only re-merge resumes cleanly (no raise);
    a one-token transcript change still fails loud at check_regime."""
    a = _payload()
    remerged = copy.deepcopy(a)
    remerged["meta"]["timestamp_utc"] = "2026-08-15T00:00:00+00:00"
    remerged["meta"]["git_commit"] = "0000beef"
    changed = copy.deepcopy(a)
    changed["transcripts"]["conv0"]["messages"][1]["content"] = "Twice upon a time."

    for build in (_act_regime, _fire_regime):
        C.check_regime(build(a), build(remerged), tmp_path / "r.json")  # no raise
        with pytest.raises(ValueError, match="REGIME MISMATCH"):
            C.check_regime(build(a), build(changed), tmp_path / "r.json")
