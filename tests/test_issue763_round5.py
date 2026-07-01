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
    """``_load_v0`` per-file ``hf_hub_download``s the shard when the local copy is absent.

    Pre-fix ``_load_v0`` did a blind ``torch.load`` and FileNotFoundError'd on a
    fresh phase-2 VM. Post-fix a missing local shard triggers
    ``_stage_v0_shards_from_hf`` -> a PER-FILE ``hf_hub_download`` (NOT
    ``snapshot_download(allow_patterns=...)``, which truncates past ~7900 siblings
    on the 94k-file data repo — #763 BLOCKER siblings-truncation). This
    monkeypatches ``hf_hub_download`` (no network), counts the calls (one per
    missing behavior, filename by exact path), and asserts ``_load_v0`` returns
    its tensor + context ids.
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
    prefix = f.HF_ANALYSIS_TENSORS_PREFIX  # issue763_matched_v0/analysis_tensors
    calls: list[str] = []

    # a fake per-file hf_hub_download that writes the requested shard + returns its path
    def _fake_hf_hub_download(repo_id, repo_type, filename):
        calls.append(filename)
        assert filename == f"{prefix}/v0_shards/v0_{behavior}.pt", filename
        dst = tmp_path / "hf_cache" / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tensor": fake_tensor, "context_ids": fake_ctx_ids, "behavior": behavior}, dst)
        return str(dst)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download)
    # snapshot_download must NEVER be reached (the siblings-truncation trap)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("snapshot_download used")),
    )

    tensor, ctx_ids = f._load_v0(behavior)
    assert ctx_ids == fake_ctx_ids
    assert tensor.shape == (3, 28, 3584)
    assert isinstance(tensor, np.ndarray)
    # exactly one per-file download, resolved by exact path
    assert calls == [f"{prefix}/v0_shards/v0_{behavior}.pt"]
    # the shard was actually staged into the LOCAL path the loader reads
    assert (eval_dir / "v0_shards" / f"v0_{behavior}.pt").exists()


def test_load_v0_prefers_local_shard_no_hf_call(monkeypatch, tmp_path):
    """When the local shard EXISTS, ``_load_v0`` reads it and never calls HF.

    The matched-host RunPod resume keeps the volume, so the stage must be a no-op
    there (never a needless per-file ``hf_hub_download``).
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

    def _boom(*a, **k):  # neither HF fetcher must be reached
        raise AssertionError("HF download called despite a local shard present")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)

    tensor, ctx_ids = f._load_v0(behavior)
    assert ctx_ids == ["c0", "c1"]
    assert tensor.shape == (2, 28, 3584)


def test_stage_v0_shards_fail_loud_when_neither_local_nor_hf(monkeypatch, tmp_path):
    """A shard NEITHER local NOR on HF fail-louds (never silently continues).

    The capture phase genuinely never produced it -> a clear FileNotFoundError
    (the per-file ``hf_hub_download`` raises ``EntryNotFoundError`` -> re-raised as
    FileNotFoundError), not a downstream torch.load crash on a confusing path.
    """
    import issue763_fit_predictors as f
    import pytest
    from huggingface_hub.utils import EntryNotFoundError

    eval_dir = tmp_path / "eval_results" / "issue_763"
    eval_dir.mkdir(parents=True)
    monkeypatch.setattr(f, "EVAL_RESULTS_DIR", eval_dir)

    def _entry_missing(repo_id, repo_type, filename):
        raise EntryNotFoundError(f"404: {filename}")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _entry_missing)

    with pytest.raises(FileNotFoundError, match="neither local"):
        f._stage_v0_shards_from_hf(["deception"])


def _called_func_names(fn) -> set[str]:
    """Return the set of bare function names CALLED in ``fn``'s body (AST, not text).

    AST-level so an explanatory comment or docstring that MENTIONS a banned call
    (e.g. ``snapshot_download(allow_patterns=...)`` in a "why NOT this" note) does
    not trip the invariant — only a real ``Call`` node counts.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fnode = node.func
            if isinstance(fnode, ast.Name):
                names.add(fnode.id)
            elif isinstance(fnode, ast.Attribute):
                names.add(fnode.attr)
    return names


def test_stage_helpers_do_not_use_snapshot_download_allow_patterns():
    """No stage helper CALLS ``snapshot_download`` (the siblings-truncation trap).

    #763 BLOCKER snapshot-download-allow-patterns-siblings-truncation: the data
    repo has >94k files, 12x past the ~7900-siblings truncation point, so a
    pattern-filtered ``snapshot_download`` can silently match 0 files. ALL THREE
    stage helpers MUST use per-file ``hf_hub_download`` — the r4 fix touched only
    two (fit_predictors + extract_pv_rb) and left the judge_e0 gen-staging helper
    on the broken pattern (the r5 third-site BLOCKER
    snapshot-download-siblings-truncation-third-site-gen-staging). AST-level so a
    future edit reintroducing a ``snapshot_download`` CALL in any helper fails
    (a bare mention in a "why NOT this" comment is fine).
    """
    import issue763_extract_pv_rb as p
    import issue763_fit_predictors as f
    import issue763_judge_e0 as j

    for name, fn in (
        ("_stage_v0_shards_from_hf", f._stage_v0_shards_from_hf),
        ("_stage_from_hf", p._stage_from_hf),
        ("_stage_gen_from_hf", j._stage_gen_from_hf),
    ):
        called = _called_func_names(fn)
        assert "snapshot_download" not in called, (
            f"{name} must NOT CALL snapshot_download (siblings-truncation trap); "
            "use per-file hf_hub_download"
        )
        assert "hf_hub_download" in called, f"{name} must CALL per-file hf_hub_download"


# ── (3) pv-keepflags-under-alignment-rubric-fallback ──────────────────────────


def test_keepflags_canonical_gate_rejects_contaminated_r3_set(tmp_path, monkeypatch):
    """``_keepflags_are_canonical`` rejects the r3 keep-flags (no hash) + a stale-hash set.

    task #763 BLOCKER pv-keepflags-under-alignment-rubric-fallback: the r3
    pv_judge/ keep-flags on HF were judged under the alignment misalignment-rubric
    fallback (commit 0ecadbbc13) and carry NO ``judge_system_prompt_hash``. The
    gate MUST reject them (and a v2 file with a MISMATCHED hash — a future prompt
    edit), and accept ONLY a file stamped with the current canonical hash. Failing
    pre-fix (the r6 capture read pv_judge/ by path unconditionally) / passing
    post-fix. This is the permanent invariant behind Fix (a).
    """
    import issue763_extract_pv_rb as p

    v2_dir = tmp_path / "pv_judge_v2"
    v2_dir.mkdir()
    monkeypatch.setattr(p, "PV_JUDGE_V2_DIR", v2_dir)

    # (a) absent -> not canonical
    assert p._keepflags_are_canonical("deception") is False

    # (b) the r3-shape file (no judge_system_prompt_hash) -> not canonical
    p.dump_json(
        {"behavior": "deception", "keep_flags": [True], "dropped": {"pos": 0, "neg": 0}},
        v2_dir / "deception.json",
    )
    assert p._keepflags_are_canonical("deception") is False, (
        "a keep-flag file WITHOUT judge_system_prompt_hash (the contaminated r3 shape) "
        "must be rejected as non-canonical"
    )

    # (c) a MISMATCHED hash (future prompt edit) -> not canonical
    p.dump_json(
        {"behavior": "deception", "judge_system_prompt_hash": "deadbeef" * 8},
        v2_dir / "deception.json",
    )
    assert p._keepflags_are_canonical("deception") is False

    # (d) the current canonical hash -> canonical
    p.dump_json(
        {"behavior": "deception", "judge_system_prompt_hash": p._PV_JUDGE_SYSTEM_PROMPT_HASH},
        v2_dir / "deception.json",
    )
    assert p._keepflags_are_canonical("deception") is True


def test_pv_judge_prompt_hash_is_of_the_corrected_prompt():
    """The stamped hash fingerprints the CORRECTED trait-rubric prompt, not the fallback.

    Pins that ``_PV_JUDGE_SYSTEM_PROMPT_HASH`` is the sha256 of the current
    (task-neutral, non-alignment) grader prompt — so a revert to the alignment
    fallback would change the hash and invalidate every stamped v2 file.
    """
    import hashlib

    import issue763_extract_pv_rb as p

    expected = hashlib.sha256(p._PV_JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert expected == p._PV_JUDGE_SYSTEM_PROMPT_HASH
    assert "misalign" not in p._PV_JUDGE_SYSTEM_PROMPT.lower()


def test_capture_resolves_canonical_keepflags_never_reads_legacy_pv_judge():
    """``_phase_capture`` resolves keep-flags via ``_ensure_canonical_keepflags``.

    The r6 capture staged + read the contaminated ``pv_judge/`` set by path. The
    fix routes keep-flag resolution through ``_ensure_canonical_keepflags`` (which
    re-judges under the corrected rubric when the canonical v2 set is absent) and
    reads ONLY ``_judge_path`` (pv_judge_v2/). Source-level invariant: the capture
    body calls the resolver and does NOT stage the legacy ``pv_judge`` prefix.
    """
    import inspect

    import issue763_extract_pv_rb as p

    src = inspect.getsource(p._phase_capture)
    assert "_ensure_canonical_keepflags(behaviors)" in src, (
        "capture must resolve keep-flags via _ensure_canonical_keepflags (re-judge on miss)"
    )
    assert '_stage_from_hf("pv_judge"' not in src, (
        "capture must NOT stage the legacy alignment-rubric-contaminated pv_judge/ set"
    )


def test_upload_keepflags_targets_the_v2_prefix():
    """The keep-flag upload writes to the NEW pv_judge_v2/ HF prefix (provenance-safe).

    The corrected keep-flags land at issue763_matched_v0/analysis_tensors/pv_judge_v2/
    so the contaminated r3 pv_judge/ set stays on HF for provenance.
    """
    import inspect

    import issue763_extract_pv_rb as p

    src = inspect.getsource(p._upload_judge_keepflags)
    assert "pv_judge_v2" in src, "keep-flag upload must target the pv_judge_v2/ prefix"
