"""Tests for the backend-agnostic artifact verifier.

These tests exercise the verifier with NO real HF / WandB / git / filesystem
side effects — every external call is mocked via :class:`VerifierIO`. They
cover the contract :mod:`backends.artifacts` promises:

1. PASS when every declared artifact class resolves AND the sentinel proves
   intentional completion.
2. FAIL with explicit reasons when each class is missing (HF data, HF model,
   WandB, git, sentinel — each independently).
3. SKIP for any class whose declaration is empty (eval-only run has no model
   checkpoint, etc.) — SKIPs do NOT contribute to the verdict.
4. The two backends' ``confirm_artifacts`` honor the verdict (return False
   on FAIL, True on PASS) without raising.
5. A handle that forgot to declare expected artifacts FAILs with a clear
   reason (no silent True on "verifier had nothing to check").
6. Transport errors (HF Hub unreachable, WandB API down) become FAIL with
   reason, NOT silent True (CLAUDE.md "fail fast — never hide failures").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from explore_persona_space.backends import (
    EXPECTED_ARTIFACTS_HANDLE_KEY,
    ArtifactVerdict,
    BackendKind,
    ExpectedArtifacts,
    RunHandle,
    RunPodBackend,
    SlurmBackend,
    VerifierIO,
    confirm_artifacts_from_handle,
    verify_artifacts,
    write_completion_sentinel,
)
from explore_persona_space.backends.artifacts import (
    CHECK_GIT,
    CHECK_HF_DATA,
    CHECK_HF_MODEL,
    CHECK_SENTINEL,
    CHECK_WANDB,
    DEFAULT_HF_DATA_REPO,
    DEFAULT_HF_MODEL_REPO,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _good_sentinel_text(issue: int = 137, **extra: Any) -> str:
    payload: dict[str, Any] = {"phase": "done", "issue": issue}
    payload.update(extra)
    return json.dumps(payload, sort_keys=True) + "\n"


def _io(
    *,
    hf_data_files: list[str] | None = None,
    hf_model_files: list[str] | None = None,
    wandb_runs: set[str] | None = None,
    wandb_raises: BaseException | None = None,
    hf_raises: BaseException | None = None,
    git_tracked_paths: set[str] | None = None,
    git_raises: BaseException | None = None,
    sentinel_content: str | None = None,
    sentinel_raises: BaseException | None = None,
    repo_root: Path | None = None,
    on_disk: set[str] | None = None,
) -> VerifierIO:
    """Construct a fully-mocked :class:`VerifierIO` for a single test.

    Every callable defaults to a "PASS" stub; pass keyword overrides to
    inject the specific failure mode this test exercises.
    """
    hf_data_files = hf_data_files or []
    hf_model_files = hf_model_files or []
    wandb_runs = wandb_runs or set()
    git_tracked_paths = git_tracked_paths or set()
    on_disk = on_disk if on_disk is not None else git_tracked_paths

    def _list_hf(repo_id: str, *, repo_type: str, revision: str | None = None) -> list[str]:
        if hf_raises is not None:
            raise hf_raises
        if repo_type == "dataset":
            return list(hf_data_files)
        return list(hf_model_files)

    def _wandb(run_path: str) -> bool:
        if wandb_raises is not None:
            raise wandb_raises
        return run_path in wandb_runs

    def _git(root: Path, rel_paths) -> set[str]:
        if git_raises is not None:
            raise git_raises
        return {p for p in rel_paths if p in git_tracked_paths}

    def _sentinel(path: str) -> str | None:
        if sentinel_raises is not None:
            raise sentinel_raises
        return sentinel_content

    # The git on-disk check reads `(repo_root / p).exists()`; we point repo_root
    # at a tmp dir and seed the requested files. Tests that pin a specific path
    # set `on_disk` explicitly and the fixture creates the files.
    if repo_root is not None:
        for rel in on_disk:
            target = repo_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")

    return VerifierIO(
        list_hf_repo_files=_list_hf,
        wandb_run_exists=_wandb,
        git_tracked=_git,
        read_sentinel=_sentinel,
        repo_root=repo_root,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_pass_when_every_class_resolves(tmp_path: Path) -> None:
    """Verifier PASSes when every declared class resolves + sentinel valid."""
    expected = ExpectedArtifacts(
        issue=137,
        hf_data_paths=("issue137_warmth/raw_completions/",),
        hf_model_paths=("issue-137-c1-seed-42/",),
        wandb_run_path="superkaiba/explore-persona-space/runs/abc123",
        git_paths=("eval_results/issue_137/run_result.json",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        hf_data_files=["issue137_warmth/raw_completions/seed_42.json"],
        hf_model_files=["issue-137-c1-seed-42/adapter_model.safetensors"],
        wandb_runs={"superkaiba/explore-persona-space/runs/abc123"},
        git_tracked_paths={"eval_results/issue_137/run_result.json"},
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert verdict.passed, verdict.reasons
    assert verdict.reasons == ()
    for name in (CHECK_HF_DATA, CHECK_HF_MODEL, CHECK_WANDB, CHECK_GIT, CHECK_SENTINEL):
        assert verdict.checks[name]["status"] == "PASS", verdict.checks[name]


def test_pass_with_skipped_classes(tmp_path: Path) -> None:
    """Eval-only run with no model + no WandB: SKIPs do not fail the verdict."""
    expected = ExpectedArtifacts(
        issue=200,
        hf_data_paths=("issue200_eval/raw_completions/",),
        git_paths=("eval_results/issue_200/run_result.json",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        hf_data_files=["issue200_eval/raw_completions/seed_42.json"],
        git_tracked_paths={"eval_results/issue_200/run_result.json"},
        sentinel_content=_good_sentinel_text(issue=200),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert verdict.passed, verdict.reasons
    assert verdict.checks[CHECK_HF_MODEL]["status"] == "SKIP"
    assert verdict.checks[CHECK_WANDB]["status"] == "SKIP"


# ---------------------------------------------------------------------------
# Per-class failures (one independent FAIL per check)
# ---------------------------------------------------------------------------


def test_fail_when_hf_data_missing(tmp_path: Path) -> None:
    expected = ExpectedArtifacts(
        issue=137,
        hf_data_paths=("issue137_warmth/raw_completions/",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    # The data repo enumerates OTHER files but not the expected prefix.
    io = _io(
        hf_data_files=["other_issue/raw_completions/seed_42.json"],
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_HF_DATA]["status"] == "FAIL"
    assert "missing paths" in verdict.checks[CHECK_HF_DATA]["detail"]
    assert "issue137_warmth/raw_completions/" in verdict.checks[CHECK_HF_DATA]["detail"]


def test_fail_when_hf_model_missing(tmp_path: Path) -> None:
    expected = ExpectedArtifacts(
        issue=137,
        hf_model_paths=("issue-137-c1-seed-42/",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        hf_model_files=["issue-99-c1-seed-42/adapter_model.safetensors"],
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_HF_MODEL]["status"] == "FAIL"


def test_fail_when_wandb_run_absent(tmp_path: Path) -> None:
    expected = ExpectedArtifacts(
        issue=137,
        wandb_run_path="superkaiba/explore-persona-space/runs/abc123",
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        wandb_runs=set(),  # the requested run is not in the API
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_WANDB]["status"] == "FAIL"
    assert "WandB run not found" in verdict.checks[CHECK_WANDB]["detail"]


def test_fail_when_git_path_not_tracked(tmp_path: Path) -> None:
    expected = ExpectedArtifacts(
        issue=137,
        git_paths=("eval_results/issue_137/run_result.json",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    # Path exists on disk but git ls-files reports nothing tracked.
    io = _io(
        git_tracked_paths=set(),
        on_disk={"eval_results/issue_137/run_result.json"},
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_GIT]["status"] == "FAIL"
    assert "not tracked by git" in verdict.checks[CHECK_GIT]["detail"]


def test_fail_when_git_path_tracked_but_deleted(tmp_path: Path) -> None:
    """A tracked-but-deleted file (git rm without commit) FAILs the on-disk check."""
    expected = ExpectedArtifacts(
        issue=137,
        git_paths=("eval_results/issue_137/run_result.json",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        git_tracked_paths={"eval_results/issue_137/run_result.json"},
        on_disk=set(),  # tracked, but deleted from the working tree
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_GIT]["status"] == "FAIL"
    assert "not on disk" in verdict.checks[CHECK_GIT]["detail"]


def test_fail_when_sentinel_missing(tmp_path: Path) -> None:
    expected = ExpectedArtifacts(
        issue=137,
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        sentinel_content=None,  # file does not exist
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert "missing" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_fail_when_sentinel_not_json(tmp_path: Path) -> None:
    expected = ExpectedArtifacts(
        issue=137,
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        sentinel_content="not json at all",
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert "not valid JSON" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_fail_when_sentinel_phase_wrong(tmp_path: Path) -> None:
    expected = ExpectedArtifacts(
        issue=137,
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        sentinel_content=json.dumps({"phase": "crashed", "issue": 137}),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert "phase='crashed'" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_fail_when_sentinel_issue_mismatch(tmp_path: Path) -> None:
    """A sentinel written by a different issue's run is NOT acceptable.

    Guards against a stale sentinel file in a re-used scratch dir
    soft-passing the gate for a fresh issue.
    """
    expected = ExpectedArtifacts(
        issue=137,
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        sentinel_content=_good_sentinel_text(issue=99),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert "issue=99" in verdict.checks[CHECK_SENTINEL]["detail"]


# ---------------------------------------------------------------------------
# Transport errors → FAIL with reason (NEVER silent True)
# ---------------------------------------------------------------------------


def test_fail_when_hf_hub_unreachable(tmp_path: Path) -> None:
    """A network error talking to HF must become FAIL with reason, not silent True."""
    expected = ExpectedArtifacts(
        issue=137,
        hf_data_paths=("issue137_warmth/raw_completions/",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        hf_raises=ConnectionError("Hub 503"),
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_HF_DATA]["status"] == "FAIL"
    assert "raised" in verdict.checks[CHECK_HF_DATA]["detail"]
    assert "Hub 503" in verdict.checks[CHECK_HF_DATA]["detail"]


def test_fail_when_wandb_transport_errors(tmp_path: Path) -> None:
    expected = ExpectedArtifacts(
        issue=137,
        wandb_run_path="x/y/runs/z",
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        wandb_raises=RuntimeError("wandb api down"),
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_WANDB]["status"] == "FAIL"


# ---------------------------------------------------------------------------
# RunHandle bridge + backend wiring
# ---------------------------------------------------------------------------


def _handle_with_expected(
    *,
    backend: BackendKind = "cluster",
    issue: int = 137,
    declaration: dict[str, Any] | None,
) -> RunHandle:
    """Build a RunHandle with (or without) an expected-artifacts declaration."""
    extra: dict[str, Any] = {}
    if declaration is not None:
        extra[EXPECTED_ARTIFACTS_HANDLE_KEY] = declaration
    return RunHandle(
        backend=backend,
        cluster="nibi" if backend == "cluster" else None,
        job_id="9999" if backend == "cluster" else "",
        pod_name=f"eps-issue-{issue}" if backend == "cluster" else f"pod-{issue}",
        scratch_dir=f"/scratch/tjiral/eps/issue-{issue}",
        log_path=f"/scratch/tjiral/eps/issue-{issue}/job.out",
        extra=extra,
    )


def test_confirm_from_handle_fails_loud_when_declaration_missing() -> None:
    """A handle with no expected-artifacts declaration FAILs the gate.

    This is the silent-loss hole the verifier closes: if the launch path
    forgot to populate the declaration, we MUST NOT silently pass.
    """
    handle = _handle_with_expected(declaration=None)
    verdict = confirm_artifacts_from_handle(handle)
    assert not verdict.passed
    assert any("missing" in r and EXPECTED_ARTIFACTS_HANDLE_KEY in r for r in verdict.reasons)


def test_confirm_from_handle_round_trip_pass(tmp_path: Path) -> None:
    """End-to-end: a handle with a full declaration + all mocks PASSing."""
    sentinel = tmp_path / ".sentinel.json"
    handle = _handle_with_expected(
        declaration={
            "issue": 137,
            "hf_data_paths": ["issue137_warmth/raw_completions/"],
            "hf_model_paths": ["issue-137-c1-seed-42/"],
            "wandb_run_path": "superkaiba/explore-persona-space/runs/abc123",
            "git_paths": ["eval_results/issue_137/run_result.json"],
            "sentinel_path": str(sentinel),
        }
    )
    io = _io(
        hf_data_files=["issue137_warmth/raw_completions/seed_42.json"],
        hf_model_files=["issue-137-c1-seed-42/adapter_model.safetensors"],
        wandb_runs={"superkaiba/explore-persona-space/runs/abc123"},
        git_tracked_paths={"eval_results/issue_137/run_result.json"},
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = confirm_artifacts_from_handle(handle, io=io)
    assert verdict.passed, verdict.reasons


def test_confirm_from_handle_fails_when_no_sentinel_declared(tmp_path: Path) -> None:
    """A declaration that omits sentinel_path must FAIL even if every other class
    would pass. The completion sentinel is the keystone per-run proof; skipping it
    is the all-SKIP silent-pass hole (a partial launch-wiring mistake) the gate
    exists to close."""
    handle = _handle_with_expected(
        declaration={
            "issue": 137,
            "hf_data_paths": ["issue137_warmth/raw_completions/"],
            # NOTE: no sentinel_path declared
        }
    )
    io = _io(
        hf_data_files=["issue137_warmth/raw_completions/seed_42.json"],
        repo_root=tmp_path,
    )
    verdict = confirm_artifacts_from_handle(handle, io=io)
    assert not verdict.passed
    assert any("sentinel" in r for r in verdict.reasons)


def test_confirm_sentinel_non_integer_issue_fails_not_crashes(tmp_path: Path) -> None:
    """A corrupted/hand-edited sentinel with a non-integer issue must FAIL with a
    reason, not raise (raising would break the fail-closed contract + the
    epm:upload-verify-failed marker path)."""
    expected = ExpectedArtifacts(
        issue=137,
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        sentinel_content=_good_sentinel_text(issue="137abc"),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)  # must NOT raise
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert "non-integer" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_slurm_confirm_artifacts_returns_false_on_fail(monkeypatch) -> None:
    """SlurmBackend.confirm_artifacts honors the verifier verdict (no longer raises)."""
    # A handle whose declaration is bogus (HF data path that nothing matches).
    handle = _handle_with_expected(
        backend="cluster",
        declaration={
            "issue": 137,
            "hf_data_paths": ["issue137_warmth/raw_completions/"],
        },
    )
    # Patch the module-level default IO callable to return an empty file list,
    # so the HF check fails without any real network call.
    monkeypatch.setattr(
        "explore_persona_space.backends.artifacts._default_list_hf_repo_files",
        lambda repo_id, *, repo_type, revision=None: [],
    )
    backend = SlurmBackend()
    assert backend.confirm_artifacts(handle) is False


def test_slurm_confirm_artifacts_returns_true_on_pass(monkeypatch, tmp_path: Path) -> None:
    sentinel = tmp_path / ".sentinel.json"
    sentinel.write_text(_good_sentinel_text(issue=137))
    handle = _handle_with_expected(
        backend="cluster",
        declaration={
            "issue": 137,
            "hf_data_paths": ["issue137_warmth/raw_completions/"],
            "sentinel_path": str(sentinel),
        },
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.artifacts._default_list_hf_repo_files",
        lambda repo_id, *, repo_type, revision=None: [
            "issue137_warmth/raw_completions/seed_42.json"
        ],
    )
    backend = SlurmBackend()
    assert backend.confirm_artifacts(handle) is True


def test_runpod_confirm_artifacts_returns_false_on_missing_declaration() -> None:
    """RunPodBackend.confirm_artifacts no longer raises NotImplementedError."""
    handle = _handle_with_expected(backend="runpod", declaration=None)
    backend = RunPodBackend()
    # Must return False (not raise) — silent True is the actual failure mode.
    assert backend.confirm_artifacts(handle) is False


def test_runpod_confirm_artifacts_returns_true_on_pass(monkeypatch, tmp_path: Path) -> None:
    sentinel = tmp_path / ".sentinel.json"
    sentinel.write_text(_good_sentinel_text(issue=42))
    handle = _handle_with_expected(
        backend="runpod",
        issue=42,
        declaration={
            "issue": 42,
            "sentinel_path": str(sentinel),
        },
    )
    backend = RunPodBackend()
    # No HF / WandB / git paths declared → all SKIP; sentinel PASS → verdict PASS.
    assert backend.confirm_artifacts(handle) is True


# ---------------------------------------------------------------------------
# Sentinel writer round-trip
# ---------------------------------------------------------------------------


def test_write_sentinel_round_trip(tmp_path: Path) -> None:
    """The sentinel writer + reader agree on shape; verifier accepts the output."""
    sentinel_path = tmp_path / "out" / ".sentinel.json"
    written = write_completion_sentinel(
        sentinel_path=sentinel_path,
        issue=137,
        extra={"commit_sha": "abc123", "wandb_url": "https://wandb.ai/x/y/runs/z"},
    )
    assert written.exists()
    data = json.loads(written.read_text())
    assert data["phase"] == "done"
    assert data["issue"] == 137
    assert data["commit_sha"] == "abc123"

    expected = ExpectedArtifacts(issue=137, sentinel_path=str(written))
    # Default IO is fine here — only the sentinel check runs, and it reads
    # from the real filesystem. Skip every other class via empty declarations.
    verdict = verify_artifacts(expected)
    assert verdict.passed, verdict.reasons


# ---------------------------------------------------------------------------
# Defaults + sanity
# ---------------------------------------------------------------------------


def test_default_repos_match_upload_policy() -> None:
    """The defaults match the project Upload Policy table (and verify_uploads.py)."""
    assert DEFAULT_HF_DATA_REPO == "superkaiba1/explore-persona-space-data"
    assert DEFAULT_HF_MODEL_REPO == "superkaiba1/explore-persona-space"


def test_verdict_is_frozen_and_truthy() -> None:
    """Verdict is a usable dataclass: ``.passed`` drives a plain bool conversion."""
    verdict = ArtifactVerdict(passed=True, reasons=(), checks={})
    assert verdict.passed is True
    # FrozenInstanceError on assignment proves the dataclass is frozen — a
    # mutable verdict would let a buggy caller flip the bool post-return.
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        verdict.passed = False  # type: ignore[misc]  # frozen
