"""Issue #763 round-5 (r3 crash-fix) regression tests.

Two permanent invariants closing the two sequential production-path crashes in
the Phase-2 GPU launch (task #763 r3 `epm:failure v5`), each failing pre-fix /
passing post-fix:

1. ``graded-judge-empty-system-block`` — the E0 + PV judge dispatches MUST send a
   NON-EMPTY system content block. Anthropic 400s an explicit empty system block
   (``system: text content blocks must be non-empty``), which quarantined all
   8000 graded requests (and the binary requests) on the first live-batch submit.
   The r2 ``--mock-judge`` smoke never exercised a live Anthropic Batch submit, so
   the malformed request shape was invisible. The invariant: the built request's
   ``system[0].text`` is non-empty for the graded transport, and the judge drivers
   pass a non-empty ``judge_system_prompt`` (NOT ``""``, NOT the alignment-rubric
   fallback ``None``).

2. ``fit-missing-v0-shards`` — ``issue763_fit_predictors._load_v0`` MUST stage the
   v0 shard from HF when the local copy is missing (gate-split phase 2 on a fresh
   VM), instead of a blind ``torch.load`` -> FileNotFoundError. The v0 shards are
   WRITTEN in phase 1 (``capture``) + uploaded, but READ in phase 2 (``fit``); the
   ``pv_extract_capture`` phase never re-creates them, so a fresh phase-2 boot had
   no local shard. The invariant: a missing local shard triggers a
   ``snapshot_download`` from the issue-owned HF prefix into the loader's local
   path (mirrors ``_stage_gen_from_hf`` / ``_stage_from_hf``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))


# ── (1) graded-judge-empty-system-block ───────────────────────────────────────


def test_e0_judge_system_prompt_is_non_empty():
    """The E0 judge driver's system prompt is non-empty (was ``""`` -> Anthropic 400)."""
    import issue763_judge_e0 as j

    assert j._JUDGE_SYSTEM_PROMPT, "E0 judge system prompt must not be empty/None"
    assert j._JUDGE_SYSTEM_PROMPT.strip(), "E0 judge system prompt must not be whitespace-only"


def test_pv_judge_system_prompt_is_non_empty():
    """The PV-rollout judge driver's system prompt is non-empty (was ``None`` fallback)."""
    import issue763_extract_pv_rb as p

    assert p._PV_JUDGE_SYSTEM_PROMPT, "PV judge system prompt must not be empty/None"
    assert p._PV_JUDGE_SYSTEM_PROMPT.strip(), "PV judge system prompt must not be whitespace-only"


def test_pv_judge_system_prompt_is_not_the_alignment_fallback():
    """The PV judge system prompt does NOT inject the alignment misalignment rubric.

    ``judge_system_prompt=None`` (the r3 hotfix) fell back to
    ``_default_judge_system_prompt`` (the alignment "rate for signs of
    misalignment" rubric), which asks a DIFFERENT question than the PV trait
    rubric in the user message — contaminating the trait score. The fix uses a
    task-neutral grader instruction instead.
    """
    import issue763_extract_pv_rb as p

    from explore_persona_space.eval.judge_dispatch import _default_judge_system_prompt

    assert _default_judge_system_prompt() != p._PV_JUDGE_SYSTEM_PROMPT
    assert "misalign" not in p._PV_JUDGE_SYSTEM_PROMPT.lower()


def test_graded_request_has_non_empty_system_block():
    """The built graded Messages-API request carries a NON-EMPTY system text block.

    This is the exact field Anthropic rejected: an empty ``system[0].text`` with a
    ``cache_control`` block -> ``invalid_request_error`` on every request. With the
    fixed non-empty grader prompt threaded through ``_build_params`` (the same
    builder the batch + sync paths use), ``system[0].text`` is non-empty.
    """
    import issue763_judge_e0 as j

    from explore_persona_space.eval.judge_dispatch import _build_params, graded_temperature

    with graded_temperature(1.0):
        params = _build_params(
            "claude-sonnet-4-5-20250929",
            j._JUDGE_SYSTEM_PROMPT,
            "GRADED RUBRIC USER MSG",
            400,
            ttl="1h",
        )
    assert params["system"][0]["text"].strip(), (
        "graded request system block is EMPTY — Anthropic will 400 with "
        "invalid_request_error (the r3 regression)"
    )
    # temperature is still threaded (the N=8 graded-draw protocol), unchanged.
    assert params["temperature"] == 1.0


def test_empty_system_prompt_would_produce_empty_block_regression_marker():
    """Sanity: an EMPTY system prompt DOES produce the empty block (the pre-fix bug).

    Pins the direction of the fix — if a future edit reverts the driver to
    ``judge_system_prompt=""``, ``system[0].text`` is empty again. This asserts the
    builder faithfully carries whatever system prompt it is given, so the fix must
    live at the CALL SITE (a non-empty prompt), which the tests above pin.
    """
    from explore_persona_space.eval.judge_dispatch import _build_params

    params = _build_params("claude-sonnet-4-5-20250929", "", "USER MSG", 300, ttl="1h")
    assert params["system"][0]["text"] == "", (
        "the empty-string system prompt should yield an empty block — this is the "
        "shape that 400s; the driver must NOT pass an empty prompt"
    )


# ── (2) fit-missing-v0-shards ──────────────────────────────────────────────────


def test_load_v0_stages_from_hf_when_local_missing(monkeypatch, tmp_path):
    """``_load_v0`` snapshot_downloads the shard when the local copy is absent.

    Pre-fix ``_load_v0`` did a blind ``torch.load`` and FileNotFoundError'd on a
    fresh phase-2 VM. Post-fix a missing local shard triggers
    ``_stage_v0_shards_from_hf`` -> ``snapshot_download``. This monkeypatches
    ``snapshot_download`` (no network) to drop a fake shard into the fetched
    snapshot dir, and asserts ``_load_v0`` returns its tensor + context ids.
    """
    import issue763_fit_predictors as f
    import numpy as np
    import torch

    # point EVAL_RESULTS_DIR at a fresh temp dir with NO local v0_shards/
    eval_dir = tmp_path / "eval_results" / "issue_763"
    eval_dir.mkdir(parents=True)
    monkeypatch.setattr(f, "EVAL_RESULTS_DIR", eval_dir)

    behavior = "deception"
    fake_tensor = torch.zeros((3, 28, 3584), dtype=torch.float32)
    fake_ctx_ids = ["ctx_a", "ctx_b", "ctx_c"]

    # a fake snapshot_download that writes the shard into a mirror of the HF path
    def _fake_snapshot_download(repo_id, repo_type, allow_patterns):
        snap = tmp_path / "hf_snap"
        prefix = f.HF_ANALYSIS_TENSORS_PREFIX  # issue763_matched_v0/analysis_tensors
        shard_dir = snap / prefix / "v0_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"tensor": fake_tensor, "context_ids": fake_ctx_ids, "behavior": behavior},
            shard_dir / f"v0_{behavior}.pt",
        )
        return str(snap)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

    tensor, ctx_ids = f._load_v0(behavior)
    assert ctx_ids == fake_ctx_ids
    assert tensor.shape == (3, 28, 3584)
    assert isinstance(tensor, np.ndarray)
    # the shard was actually staged into the LOCAL path the loader reads
    assert (eval_dir / "v0_shards" / f"v0_{behavior}.pt").exists()


def test_load_v0_prefers_local_shard_no_hf_call(monkeypatch, tmp_path):
    """When the local shard EXISTS, ``_load_v0`` reads it and never calls HF.

    The matched-host RunPod resume keeps the volume, so the stage must be a no-op
    there (never a needless snapshot_download of the whole gen tree).
    """
    import issue763_fit_predictors as f
    import torch

    eval_dir = tmp_path / "eval_results" / "issue_763"
    shard_dir = eval_dir / "v0_shards"
    shard_dir.mkdir(parents=True)
    monkeypatch.setattr(f, "EVAL_RESULTS_DIR", eval_dir)

    behavior = "deception"
    torch.save(
        {
            "tensor": torch.ones((2, 28, 3584), dtype=torch.float32),
            "context_ids": ["c0", "c1"],
            "behavior": behavior,
        },
        shard_dir / f"v0_{behavior}.pt",
    )

    def _boom(*a, **k):  # snapshot_download must NOT be reached
        raise AssertionError("snapshot_download called despite a local shard present")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)

    tensor, ctx_ids = f._load_v0(behavior)
    assert ctx_ids == ["c0", "c1"]
    assert tensor.shape == (2, 28, 3584)


def test_stage_v0_shards_fail_loud_when_neither_local_nor_hf(monkeypatch, tmp_path):
    """A shard NEITHER local NOR on HF fail-louds (never silently continues).

    The capture phase genuinely never produced it -> a clear FileNotFoundError,
    not a downstream torch.load crash on a confusing path.
    """
    import issue763_fit_predictors as f
    import pytest

    eval_dir = tmp_path / "eval_results" / "issue_763"
    eval_dir.mkdir(parents=True)
    monkeypatch.setattr(f, "EVAL_RESULTS_DIR", eval_dir)

    def _empty_snapshot(repo_id, repo_type, allow_patterns):
        snap = tmp_path / "hf_empty"
        (snap / f.HF_ANALYSIS_TENSORS_PREFIX / "v0_shards").mkdir(parents=True, exist_ok=True)
        return str(snap)  # no v0_<behavior>.pt inside

    monkeypatch.setattr("huggingface_hub.snapshot_download", _empty_snapshot)

    with pytest.raises(FileNotFoundError, match="neither local"):
        f._stage_v0_shards_from_hf(["deception"])
