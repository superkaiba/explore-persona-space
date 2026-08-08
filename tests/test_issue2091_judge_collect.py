"""CPU unit tests for the #2091 P3 packed-row collector fix.

The packed rollout format (the #1190/#1739 many-small-files pack) is one JSON
object per line: ``{"src": <path relative to the raw root>, "doc": <original
file JSON>}`` — and the per-job ``_manifest.json`` is packed as a row too
(idx 0 of shard00 on the staged tree). Pre-fix, ``load_job_rollouts`` returned
the manifest doc as a rollout payload and ``collect_wave_items`` crashed on
``p["context_id"]`` (KeyError) before any judge call (epm:failure v2,
2026-08-06). The fix filters non-rollout rows on a SCHEMA predicate
(``context_id`` AND ``rollout_k`` both present — never row index) and, on a
full read, VERIFIES the surviving rollout count against the manifest's
``n_kept * k_rollouts`` — fail-loud on mismatch or on a manifest-less pack.

No GPU, no network, no repo-root artifacts: fixtures are synthetic benign
rows under ``tmp_path``; tests drive the REAL ``load_job_rollouts`` and
``collect_wave_items`` bodies (production-body rule — nothing in the collect
path is stubbed).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest

import scripts.issue2091_judge as judge_mod
from scripts.issue2091_stage_contexts import RUNG_JOBS

N_ROWS = 3  # rollouts per job in the fixture
ALIASES = ["paris"]  # hallucination own-rung alias pool (benign synthetic)
# per-job completions: index 0 is alias-CORRECT for hallucination own-rung jobs.
COMPLETIONS = ["The answer is Paris.", "I am not certain.", "It might be Rome."]


def _manifest_doc(gen_behavior: str, n_kept: int) -> dict:
    """A manifest doc with the measured staged-tree key set (structural only)."""
    return {
        "behavior": gen_behavior,
        "fingerprint": "test-fp",
        "git_commit": "test",
        "k_rollouts": 1,
        "n_contexts": n_kept,
        "n_generated": n_kept,
        "n_kept": n_kept,
        "n_resumed": 0,
        "n_truncated_rollouts": 0,
        "prompt_budget_drops": {},
        "ts": "2026-08-06T00:00:00Z",
    }


def _rollout_doc(job, i: int) -> dict:
    doc = {
        "behavior": job.gen_behavior,
        "completion": COMPLETIONS[i % len(COMPLETIONS)],
        "context_id": f"{job.name}-c{i:02d}",
        "finish_reason": "stop",
        "group_key": f"g{i}",
        "meta": {},
        "prefix_text": "",
        "prompt_text": "What is the capital of France?",
        "query": "What is the capital of France?",
        "rollout_k": 0,
        "rung": job.rung,
        "split": "train",
    }
    if job.gen_behavior == "hallucination":
        doc["answer_aliases"] = ALIASES
    return doc


def _write_job_pack(
    rollout_root: Path,
    job,
    *,
    n_rows: int = N_ROWS,
    manifest_n_kept: int | None = N_ROWS,
    manifest_last: bool = False,
) -> None:
    """Write one job's packed shard: manifest row at idx 0 (the incident shape)
    unless ``manifest_last``; ``manifest_n_kept=None`` omits the manifest row."""
    shard_dir = judge_mod.job_shard_dir(rollout_root, job.name)
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"src": f"{job.gen_behavior}/{job.name}-c{i:02d}_seed0.json", "doc": _rollout_doc(job, i)}
        for i in range(n_rows)
    ]
    if manifest_n_kept is not None:
        mani = {
            "src": f"{job.gen_behavior}/_manifest.json",
            "doc": _manifest_doc(job.gen_behavior, manifest_n_kept),
        }
        rows = [*rows, mani] if manifest_last else [mani, *rows]
    shard = shard_dir / f"{job.gen_behavior}.shard00.jsonl"
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _build_all_jobs(rollout_root: Path, **overrides_by_job) -> None:
    for job in RUNG_JOBS:
        _write_job_pack(rollout_root, job, **overrides_by_job.get(job.name, {}))


def _args(rollout_root: Path, limit: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(rollout_root=rollout_root, limit=limit)


def _expected_wave_counts(n_rows: int) -> dict[str, int]:
    """Independent re-derivation of the wave routing rule over the registry."""
    counts = {w.name: 0 for w in judge_mod.WAVES}
    for job in RUNG_JOBS:
        for behavior in job.judge_behaviors:
            if behavior == "hallucination" and job.gen_behavior == "hallucination":
                counts["hallucination_abstain"] += n_rows
            else:
                counts[f"{behavior}_trait"] += n_rows
    return counts


def test_manifest_row_filtered_counts_verified(tmp_path, caplog):
    """(a)+(b): the manifest row is excluded, item counts equal N per feeding job."""
    _build_all_jobs(tmp_path)
    with caplog.at_level(logging.INFO, logger="issue2091_judge"):
        waves = judge_mod.collect_wave_items(_args(tmp_path))

    expected = _expected_wave_counts(N_ROWS)
    for name in ("sycophancy_trait", "evil_trait", "hallucination_trait"):
        assert len(waves[name].items) == expected[name], name
        # every item id derives from a rollout context — the manifest has none.
        for item_id, _q, _a in waves[name].items:
            assert "-c" in item_id and item_id.endswith("_k00"), item_id

    # abstain wave: all own-rung rollouts alias-mapped; only non-correct judged.
    ab = waves["hallucination_abstain"]
    assert ab.alias_correct is not None
    assert len(ab.alias_correct) == expected["hallucination_abstain"]
    n_correct = sum(1 for v in ab.alias_correct.values() if v)
    hal_jobs = [j for j in RUNG_JOBS if j.gen_behavior == "hallucination"]
    assert n_correct == len(hal_jobs)  # COMPLETIONS[0] is alias-correct, one per job
    assert len(ab.items) == expected["hallucination_abstain"] - n_correct

    # (fix-engaged signal literal) per-job filtered-count log line fired.
    assert (
        "kept 3 rollout rows; filtered 1 non-rollout packed rows (1 manifest, 0 other)"
        in caplog.text
    )


def test_manifest_count_mismatch_fails_loud(tmp_path):
    """(c): manifest n_kept disagreeing with the rollout count raises with both numbers."""
    _build_all_jobs(tmp_path, syc_train={"manifest_n_kept": 5})
    with pytest.raises(ValueError, match=r"count 3 != manifest n_kept\*k_rollouts 5"):
        judge_mod.collect_wave_items(_args(tmp_path))


def test_missing_manifest_fails_loud(tmp_path):
    """(d): a pack with NO manifest row is fail-loud, never a silent pass."""
    _build_all_jobs(tmp_path, syc_train={"manifest_n_kept": None})
    with pytest.raises(ValueError, match=r"no _manifest\.json row"):
        judge_mod.collect_wave_items(_args(tmp_path))


def test_manifest_not_at_index_zero_still_filtered(tmp_path):
    """The filter keys on schema, not row position: a manifest packed LAST is
    equally excluded and the completeness check still binds."""
    _build_all_jobs(tmp_path, syc_train={"manifest_last": True})
    waves = judge_mod.collect_wave_items(_args(tmp_path))
    assert len(waves["sycophancy_trait"].items) == _expected_wave_counts(N_ROWS)["sycophancy_trait"]


def test_limit_read_skips_completeness_check(tmp_path):
    """A --limit-bounded (smoke) read must not false-fail the n_kept check."""
    _build_all_jobs(tmp_path)
    waves = judge_mod.collect_wave_items(_args(tmp_path, limit=2))
    assert len(waves["sycophancy_trait"].items) == _expected_wave_counts(2)["sycophancy_trait"]


def test_load_job_rollouts_returns_only_rollout_docs(tmp_path):
    """Direct real-body read: every returned payload carries the rollout keys."""
    job = RUNG_JOBS[0]
    _write_job_pack(tmp_path, job)
    payloads = judge_mod.load_job_rollouts(tmp_path, job.name)
    assert len(payloads) == N_ROWS
    assert all("context_id" in p and "rollout_k" in p for p in payloads)
