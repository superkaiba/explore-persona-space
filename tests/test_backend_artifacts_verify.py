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
import logging
import shutil
import subprocess
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
    SENTINEL_FILENAME,
    build_expected_artifacts_declaration,
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

    def _list_hf(
        repo_id: str,
        *,
        repo_type: str,
        revision: str | None = None,
        path_in_repo: str | None = None,
    ) -> list[str]:
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
        """Realistic ``git ls-files`` mock: returns tracked FILE paths.

        ``git_tracked_paths`` is the set of tracked files in the fake
        repo; a declared pathspec matches a tracked file when it equals
        it (file declaration) or is a directory prefix of it (directory
        declaration) — mirroring git pathspec semantics. The previous
        mock returned the declared strings verbatim, which masked the
        dir-declaration bug `_check_git` had with real IO (#588 round 2).
        """
        if git_raises is not None:
            raise git_raises
        out: set[str] = set()
        for p in rel_paths:
            prefix = p.rstrip("/") + "/"
            out |= {f for f in git_tracked_paths if f == p or f.startswith(prefix)}
        return out

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


def test_pass_when_git_dir_declaration_has_tracked_files(tmp_path: Path) -> None:
    """A directory declaration PASSes when >=1 tracked file sits under it.

    This is the canonical declaration shape (`expected_artifacts_declaration`
    emits `eval_results/issue_<N>/` + `figures/issue_<N>/`); pre-fix the
    literal-membership test could never match a file path, so every
    real-IO run FAILed (#588 round 2).
    """
    expected = ExpectedArtifacts(
        issue=588,
        git_paths=("eval_results/issue_588/", "figures/issue_588/"),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        git_tracked_paths={
            "eval_results/issue_588/att-x/smoke.json",
            "figures/issue_588/phases.png",
        },
        sentinel_content=_good_sentinel_text(issue=588),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert verdict.passed, verdict.reasons
    assert verdict.checks[CHECK_GIT]["status"] == "PASS", verdict.checks[CHECK_GIT]


def test_fail_when_git_dir_declaration_has_no_tracked_files(tmp_path: Path) -> None:
    """A directory declaration with NO tracked file under it still FAILs."""
    expected = ExpectedArtifacts(
        issue=588,
        git_paths=("eval_results/issue_588/",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        # Tracked files exist elsewhere — none under the declared dir.
        git_tracked_paths={"figures/issue_588/phases.png"},
        on_disk={"eval_results/issue_588/untracked.json", "figures/issue_588/phases.png"},
        sentinel_content=_good_sentinel_text(issue=588),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_GIT]["status"] == "FAIL"
    assert "not tracked by git" in verdict.checks[CHECK_GIT]["detail"]
    assert "eval_results/issue_588/" in verdict.checks[CHECK_GIT]["detail"]


def test_git_dir_declaration_matches_without_trailing_slash(tmp_path: Path) -> None:
    """A dir declared WITHOUT the trailing slash matches files under it too."""
    expected = ExpectedArtifacts(
        issue=588,
        git_paths=("eval_results/issue_588",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        git_tracked_paths={"eval_results/issue_588/att-x/smoke.json"},
        sentinel_content=_good_sentinel_text(issue=588),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert verdict.passed, verdict.reasons
    assert verdict.checks[CHECK_GIT]["status"] == "PASS", verdict.checks[CHECK_GIT]


def test_git_exact_file_declaration_does_not_prefix_match_siblings(tmp_path: Path) -> None:
    """Exact-file semantics are unchanged: a sibling file under the same dir
    does NOT satisfy a file declaration (no accidental prefix loosening)."""
    expected = ExpectedArtifacts(
        issue=137,
        git_paths=("eval_results/issue_137/run_result.json",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
    )
    io = _io(
        # A different file in the same directory is tracked; the declared
        # exact file is not.
        git_tracked_paths={"eval_results/issue_137/other.json"},
        on_disk={
            "eval_results/issue_137/run_result.json",
            "eval_results/issue_137/other.json",
        },
        sentinel_content=_good_sentinel_text(issue=137),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_GIT]["status"] == "FAIL"
    assert "not tracked by git" in verdict.checks[CHECK_GIT]["detail"]


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
        lambda repo_id, *, repo_type, revision=None, path_in_repo=None: [],
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
        lambda repo_id, *, repo_type, revision=None, path_in_repo=None: [
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
    """The RunPod sentinel read now goes over SSH (#598 — the sentinel
    lives on the pod, not the VM), so the PASS path is exercised through
    a faked ``subprocess.run`` returning the sentinel content."""
    sentinel = tmp_path / ".sentinel.json"
    sentinel.write_text(_good_sentinel_text(issue=42))

    class _Proc:
        returncode = 0
        stdout = _good_sentinel_text(issue=42)
        stderr = ""

    def fake_run(argv, **kwargs):
        assert argv[0] == "ssh", argv
        assert argv[1] == "pod-42", argv
        return _Proc()

    monkeypatch.setattr("subprocess.run", fake_run)
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


# ---------------------------------------------------------------------------
# issue #598 — SLURM + RunPod launch-path declarations, end to end
# ---------------------------------------------------------------------------


def _slurm_backend_with_fakes(tmp_path: Path, *, job_id: str = "9001") -> SlurmBackend:
    """Real :class:`SlurmBackend` with every external seam faked (no network)."""
    (tmp_path / "pyproject.toml").write_text("")
    return SlurmBackend(
        src_root=tmp_path,
        submitter=lambda *, robot_alias, sbatch_script: job_id,
        rsyncer=lambda **_kw: None,
        marker_poster=lambda **_kw: None,
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
    )


def test_slurm_launch_to_confirm_end_to_end(tmp_path: Path) -> None:
    """#598 deliverable 1+3: a SLURM ``launch()`` handle carries a
    declaration that ``confirm_artifacts_from_handle`` can actually
    SATISFY on a clean run — write a real sentinel at the declared local
    path, mock HF + git, and assert PASS.

    Verified through ``confirm_artifacts_from_handle(handle, io=...)``
    (NOT ``SlurmBackend().confirm_artifacts`` — that takes no ``io=``
    and would hit the live repo's git state; the backend-method
    delegation is pinned hermetically above)."""
    from explore_persona_space.backends.slurm import RunSpec as _RunSpec

    backend = _slurm_backend_with_fakes(tmp_path, job_id="9001")
    spec = _RunSpec(
        issue=137,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em", "seed=42"),
    )
    handle = backend.launch(spec)
    decl = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    # Simulate the clean run: the sbatch terminal block wrote the
    # sentinel cluster-side and fetch_results rsync'd it to the declared
    # LOCAL path (same writer shape the sbatch heredoc emits).
    write_completion_sentinel(
        sentinel_path=decl["sentinel_path"], issue=137, extra={"attempt_id": "slurm-9001"}
    )
    io = _io(
        hf_data_files=["issue137_slurm-9001/raw_completions/c1_evil_wrong_em_seed42.json"],
        git_tracked_paths={
            "eval_results/issue_137/run_result.json",
            "figures/issue_137/headline.png",
        },
        repo_root=tmp_path,
    )
    # _io's read_sentinel stub returns its `sentinel_content` arg (None
    # here) — but the sentinel is a REAL file at the declared path, so
    # rebuild the IO with the default local-FS reader for that check.
    io = VerifierIO(
        list_hf_repo_files=io.list_hf_repo_files,
        wandb_run_exists=io.wandb_run_exists,
        git_tracked=io.git_tracked,
        read_sentinel=None,  # default local-FS read — the rsync'd file
        repo_root=tmp_path,
    )
    verdict = confirm_artifacts_from_handle(handle, io=io)
    assert verdict.passed, verdict.reasons


def test_issue588_evidence_shape_would_pass(tmp_path: Path) -> None:
    """#598 deliverable 4 (retro-check): the exact #588 nibi smoke shape
    — custom ``workload_cmd``, job 15956499, HF evidence at
    ``issue588_slurm-15956499/raw_completions/`` — verifies under the
    new launch declaration, where today the same handle FAILs
    structurally on "missing declaration"."""
    from explore_persona_space.backends.slurm import RunSpec as _RunSpec

    backend = _slurm_backend_with_fakes(tmp_path, job_id="15956499")
    spec = _RunSpec(
        issue=588,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        workload_cmd="bash scripts/issue588_smoke.sh",
    )
    handle = backend.launch(spec)
    # (a) The declaration EXISTS (the live #588 FAIL was its absence)
    # with the custom-workload carve-out: the workload's real prefix was
    # its own contract, not a launch-time guess.
    decl = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    assert decl["issue"] == 588
    assert decl["hf_data_paths"] == []
    # (b) With the #588 evidence shape mocked (tracked eval JSONs +
    # figures, a valid sentinel at the declared local path, the real HF
    # listing), confirm PASSes.
    write_completion_sentinel(sentinel_path=decl["sentinel_path"], issue=588)
    hf_listing = ["issue588_slurm-15956499/raw_completions/run.json"]
    base_io = _io(
        hf_data_files=hf_listing,
        git_tracked_paths={
            "eval_results/issue_588/smoke.json",
            "figures/issue_588/phases.png",
        },
        repo_root=tmp_path,
    )
    io = VerifierIO(
        list_hf_repo_files=base_io.list_hf_repo_files,
        wandb_run_exists=base_io.wandb_run_exists,
        git_tracked=base_io.git_tracked,
        read_sentinel=None,  # default local-FS read
        repo_root=tmp_path,
    )
    verdict = confirm_artifacts_from_handle(handle, io=io)
    assert verdict.passed, verdict.reasons
    assert verdict.checks[CHECK_HF_DATA]["status"] == "SKIP"  # carve-out
    # (c) Variant: the literal #588 HF evidence shape VERIFIES when
    # explicitly declared via ``extra_hf_data_paths`` (the channel for
    # callers that know the workload's real prefix).
    decl_c = build_expected_artifacts_declaration(
        issue=588,
        sentinel_path=decl["sentinel_path"],
        custom_workload=True,
        extra_hf_data_paths=("issue588_slurm-15956499/raw_completions/",),
    )
    handle_c = _handle_with_expected(backend="cluster", issue=588, declaration=decl_c)
    verdict_c = confirm_artifacts_from_handle(handle_c, io=io)
    assert verdict_c.passed, verdict_c.reasons
    assert verdict_c.checks[CHECK_HF_DATA]["status"] == "PASS"


def test_runpod_launch_attaches_declaration(monkeypatch) -> None:
    """#598 folded sibling: ``RunPodBackend.launch`` populates the
    declaration with an ATTEMPT-BOUND pod-side sentinel path (launch-
    minted ``rp-<UTCstamp>-<4hex>`` id, also exposed as a plain
    ``runpod_attempt_id`` extra field) and NO HF guess (every RunPod
    workload is a custom dispatch — the #601 carve-out a fortiori)."""
    import re

    from explore_persona_space.backends.base import RunSpec as _RunSpec

    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    # #1465: the provision leg now routes through the Popen-based
    # pod_lifecycle relay — no-op it too (else a REAL provision runs).
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod._run_pod_lifecycle_relay",
        lambda cmd, **k: None,
    )
    backend = RunPodBackend()
    handle = backend.launch(_RunSpec(issue=42, intent="lora-7b", backend="runpod"))
    decl = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    attempt_id = handle.extra["runpod_attempt_id"]
    assert re.fullmatch(r"rp-\d{8}T\d{6}Z-[0-9a-f]{4}", attempt_id), attempt_id
    assert decl["sentinel_path"] == (
        f"/workspace/eval_results/issue_42/{attempt_id}/.completion-sentinel.json"
    )
    assert decl["hf_data_paths"] == []
    assert decl["issue"] == 42


def test_runpod_stale_sentinel_cannot_satisfy_fresh_declaration(monkeypatch, tmp_path) -> None:
    """#598 binding fix: sentinels from the FLAT legacy path AND from a
    DIFFERENT attempt's namespaced path (both valid phase=done/issue=42
    JSON, both surviving on the persistent /workspace volume) must NOT
    satisfy a fresh launch's declaration — confirm FAILs "sentinel
    missing" at the CURRENT attempt's path."""
    from explore_persona_space.backends.base import RunSpec as _RunSpec

    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    # #1465: the provision leg now routes through the Popen-based
    # pod_lifecycle relay — no-op it too (else a REAL provision runs).
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod._run_pod_lifecycle_relay",
        lambda cmd, **k: None,
    )
    backend = RunPodBackend()
    handle = backend.launch(_RunSpec(issue=42, intent="lora-7b", backend="runpod"))
    decl = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    current_path = decl["sentinel_path"]

    # Simulated pod FS: stale sentinels exist at the flat legacy path
    # and at a prior attempt's namespaced path; the CURRENT attempt's
    # path has no file.
    pod_fs = {
        "/workspace/eval_results/issue_42/.completion-sentinel.json": _good_sentinel_text(issue=42),
        "/workspace/eval_results/issue_42/rp-19990101T000000Z-dead/.completion-sentinel.json": (
            _good_sentinel_text(issue=42)
        ),
    }
    assert current_path not in pod_fs  # fresh attempt id ⇒ distinct path

    io = VerifierIO(read_sentinel=lambda p: pod_fs.get(p), repo_root=tmp_path)
    # Strip the git paths so the ONLY live check is the sentinel (the
    # staleness property under test); hf/model/wandb already SKIP.
    decl_sentinel_only = dict(decl, git_paths=[])
    handle_sentinel_only = _handle_with_expected(
        backend="runpod", issue=42, declaration=decl_sentinel_only
    )
    verdict = confirm_artifacts_from_handle(handle_sentinel_only, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert f"missing at {current_path}" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_runpod_ssh_sentinel_reader_semantics(monkeypatch) -> None:
    """#598: the injected SSH sentinel reader's three-way contract —
    rc=0 → content (confirm PASSes end-to-end), rc!=0 + "No such file"
    → None (FAIL "sentinel missing"), rc=255 transport → RAISE (FAIL
    with the real reason, NOT "missing"). ``subprocess.run`` is patched
    at the exact target the reader resolves, so no real ``ssh pod-*``
    can ever run from the suite."""

    class _Proc:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    backend = RunPodBackend()
    sentinel_path = "/workspace/eval_results/issue_42/rp-x/.completion-sentinel.json"
    declaration = {"issue": 42, "sentinel_path": sentinel_path}
    handle = _handle_with_expected(backend="runpod", issue=42, declaration=declaration)

    # rc=0 → content returned; full backend.confirm_artifacts PASSes
    # (every other check SKIPs — nothing else declared).
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _Proc(0, stdout=_good_sentinel_text(issue=42))
    )
    assert backend._ssh_read_sentinel(handle)(sentinel_path) == _good_sentinel_text(issue=42)
    assert backend.confirm_artifacts(handle) is True

    # rc=1 + "No such file" stderr → None → FAIL "sentinel missing".
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _Proc(1, stderr="cat: /workspace/...: No such file or directory"),
    )
    assert backend._ssh_read_sentinel(handle)(sentinel_path) is None
    verdict = confirm_artifacts_from_handle(
        handle, io=VerifierIO(read_sentinel=backend._ssh_read_sentinel(handle))
    )
    assert not verdict.passed
    assert "missing" in verdict.checks[CHECK_SENTINEL]["detail"]

    # rc=255 transport failure → raise → FAIL with the REAL reason
    # (must NOT read as "missing" — fail-loud on transport).
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _Proc(255, stderr="ssh: connect to host failed")
    )
    with pytest.raises(RuntimeError, match="rc=255"):
        backend._ssh_read_sentinel(handle)(sentinel_path)
    verdict = confirm_artifacts_from_handle(
        handle, io=VerifierIO(read_sentinel=backend._ssh_read_sentinel(handle))
    )
    assert not verdict.passed
    detail = verdict.checks[CHECK_SENTINEL]["detail"]
    assert "rc=255" in detail
    assert "missing" not in detail


# ---------------------------------------------------------------------------
# #705: worktree git-root resolution (#685) + phase-scope (#661) + stale
# baked-attempt sentinel resolution (#685 secondary). The LOAD-BEARING
# negative controls (a genuinely-incomplete run STILL FAILs) live alongside
# each positive case below — the gate is fixed, not relaxed.
# ---------------------------------------------------------------------------

_GIT = shutil.which("git")
_needs_git = pytest.mark.skipif(_GIT is None, reason="git not on PATH")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@e",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@e",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin",
        },
    )


def _make_worktree_with_committed_artifact(tmp_path: Path, issue: int) -> tuple[Path, Path]:
    """Build a real repo on ``main`` + a linked ``issue-<N>`` worktree.

    The eval artifact ``eval_results/issue_<N>/run_result.json`` is
    committed ONLY on the ``issue-<N>`` branch checked out in the
    worktree — NOT on ``main`` — reproducing the #685 topology (auto-merge
    to ``main`` is at /issue Step 10d, after finalize). Returns
    ``(main_repo, worktree)``.
    """
    main_repo = tmp_path / "repo"
    main_repo.mkdir()
    _git(main_repo, "init", "-b", "main")
    (main_repo / "seed.txt").write_text("seed")
    _git(main_repo, "add", "seed.txt")
    _git(main_repo, "commit", "-m", "seed")

    worktree = tmp_path / "wt"
    _git(main_repo, "worktree", "add", "-b", f"issue-{issue}", str(worktree))
    rel = f"eval_results/issue_{issue}/run_result.json"
    art = worktree / rel
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text("{}")
    _git(worktree, "add", rel)
    _git(worktree, "commit", "-m", "artifact on issue branch")
    return main_repo, worktree


@_needs_git
def test_check_git_uses_issue_worktree_for_unmerged_branch(tmp_path: Path) -> None:
    """#685: with ``git_repo_root`` set to the worktree the git check
    PASSes; unset (resolving against the MAIN tree where the file is NOT
    yet committed) it FAILs — the exact regression #705 closes.

    Uses the REAL ``git ls-files`` (no ``git_tracked`` mock) and the REAL
    on-disk check (``io.repo_root`` left ``None`` so the baked
    ``git_repo_root`` drives both), so this is the end-to-end git fixture
    the plan promises.
    """
    issue = 685
    main_repo, worktree = _make_worktree_with_committed_artifact(tmp_path, issue)
    git_paths = (f"eval_results/issue_{issue}/",)
    sentinel = str(tmp_path / ".sentinel.json")
    sentinel_io = VerifierIO(read_sentinel=lambda p: _good_sentinel_text(issue=issue))

    # UNSET git_repo_root → resolves to the pyproject-walk main root, which
    # is NOT this fixture's repo. Force the resolution to the fixture's
    # main_repo via io.repo_root to demonstrate the #685 FAIL: the file is
    # on the issue branch, not on main's working tree.
    expected_unset = ExpectedArtifacts(issue=issue, git_paths=git_paths, sentinel_path=sentinel)
    io_main = VerifierIO(read_sentinel=sentinel_io.read_sentinel, repo_root=main_repo)
    verdict_unset = verify_artifacts(expected_unset, io=io_main)
    assert not verdict_unset.passed
    assert verdict_unset.checks[CHECK_GIT]["status"] == "FAIL"
    assert "not tracked by git" in verdict_unset.checks[CHECK_GIT]["detail"]

    # SET git_repo_root=<worktree> → the REAL git check runs there, where
    # the file IS committed + on disk → PASS. io.repo_root left None so the
    # baked git_repo_root is the resolution source.
    expected_set = ExpectedArtifacts(
        issue=issue,
        git_paths=git_paths,
        sentinel_path=sentinel,
        git_repo_root=str(worktree),
    )
    verdict_set = verify_artifacts(expected_set, io=sentinel_io)
    assert verdict_set.passed, verdict_set.reasons
    assert verdict_set.checks[CHECK_GIT]["status"] == "PASS", verdict_set.checks[CHECK_GIT]


@_needs_git
def test_check_git_falls_back_to_main_when_baked_worktree_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#705 concern ``worktree-removed-at-finalize-fallback-test``: when the
    baked ``git_repo_root`` directory no longer exists (worktree merged +
    removed post-Step-10d), the resolver falls back to the pyproject-walked
    main root with a LOUD log and the git check PASSes on the now-on-main
    file — instead of FAILing on the removed worktree.
    """
    issue = 705
    main_repo, worktree = _make_worktree_with_committed_artifact(tmp_path, issue)
    # Merge the issue branch into main, then REMOVE the worktree — the
    # post-Step-10d auto-merge end-state (the file is now on main's tree).
    _git(main_repo, "merge", "--no-ff", f"issue-{issue}", "-m", "merge")
    _git(main_repo, "worktree", "remove", str(worktree))
    assert not worktree.exists()

    # The fallback target is the pyproject-walk root; pin it to main_repo.
    monkeypatch.setattr(
        "explore_persona_space.backends.artifacts._pyproject_walk_root", lambda: main_repo
    )

    expected = ExpectedArtifacts(
        issue=issue,
        git_paths=(f"eval_results/issue_{issue}/",),
        sentinel_path=str(tmp_path / ".sentinel.json"),
        git_repo_root=str(worktree),  # absent now
    )
    io = VerifierIO(read_sentinel=lambda p: _good_sentinel_text(issue=issue))
    verdict = verify_artifacts(expected, io=io)
    assert verdict.passed, verdict.reasons
    assert verdict.checks[CHECK_GIT]["status"] == "PASS", verdict.checks[CHECK_GIT]


def test_build_declaration_phase_scope_omits_full_task_git_paths() -> None:
    """#661: ``skip_default_git_paths=True`` drops the auto full-task git
    paths from the shared builder; the GCP twin produces the IDENTICAL
    shape; ``False`` (default) is unchanged.
    """
    from explore_persona_space.backends.gcp import GcpConfig
    from explore_persona_space.backends.gcp import (
        expected_artifacts_declaration as gcp_decl,
    )

    sentinel = "/scratch/eval_results/issue_661/att-1/.completion-sentinel.json"
    decl_off = build_expected_artifacts_declaration(
        issue=661, sentinel_path=sentinel, custom_workload=True, attempt_id="att-1"
    )
    # #790: custom_workload keeps eval_results/ (drivers commit it during the
    # run) but drops the analyzer-generated figures/.
    assert decl_off["git_paths"] == ["eval_results/issue_661/"]
    decl_on = build_expected_artifacts_declaration(
        issue=661,
        sentinel_path=sentinel,
        custom_workload=True,
        attempt_id="att-1",
        skip_default_git_paths=True,
    )
    assert decl_on["git_paths"] == []
    # An explicit extra_git_path still survives the skip (a phase that DOES
    # commit a scoped git file can declare it).
    decl_on_extra = build_expected_artifacts_declaration(
        issue=661,
        sentinel_path=sentinel,
        custom_workload=True,
        attempt_id="att-1",
        skip_default_git_paths=True,
        extra_git_paths=("eval_results/issue_661/phase3_manifest.json",),
    )
    assert decl_on_extra["git_paths"] == ["eval_results/issue_661/phase3_manifest.json"]

    # GCP twin parity.
    from explore_persona_space.backends.base import RunSpec

    cfg = GcpConfig()
    spec = RunSpec(issue=661, intent="eval", backend="gcp", workload_cmd="bash x.sh")
    gcp_on = gcp_decl(spec=spec, config=cfg, attempt_id="att-1", skip_default_git_paths=True)
    gcp_off = gcp_decl(spec=spec, config=cfg, attempt_id="att-1")
    assert gcp_on["git_paths"] == []
    # #790: spec has workload_cmd → custom_workload → eval_results/ only.
    assert gcp_off["git_paths"] == ["eval_results/issue_661/"]


def test_build_expected_artifacts_declaration_hydra_workload_omits_figures() -> None:
    """#790: a ``--workload-cmd`` (``custom_workload=True``) declaration KEEPS
    ``eval_results/`` (dispatch drivers commit eval JSONs during the run — a
    missing one there is a genuine FAIL) but DROPS ``figures/`` (the analyzer
    generates + commits figures POST-gate on every lane, so ``figures/`` is
    never produced during the run and declaring it is a guaranteed false-FAIL).
    """
    sentinel = "/scratch/eval_results/issue_790/att-1/.completion-sentinel.json"
    decl = build_expected_artifacts_declaration(
        issue=790,
        sentinel_path=sentinel,
        custom_workload=True,
        attempt_id="att-1",
    )
    assert decl["git_paths"] == ["eval_results/issue_790/"]
    # extra_git_paths still append after the kept default.
    decl_extra = build_expected_artifacts_declaration(
        issue=790,
        sentinel_path=sentinel,
        custom_workload=True,
        attempt_id="att-1",
        extra_git_paths=("eval_results/issue_790/panel.json",),
    )
    assert decl_extra["git_paths"] == [
        "eval_results/issue_790/",
        "eval_results/issue_790/panel.json",
    ]


def test_build_expected_artifacts_declaration_pure_hydra_omits_defaults() -> None:
    """#790: a pure-hydra (``custom_workload=False``) declaration drops BOTH
    default git paths. ``scripts/train.py`` runs ``run_single(..., skip_eval=True)``
    and ``orchestrate/runner.py`` gates all eval production on ``not skip_eval``,
    so a hydra run writes NEITHER ``eval_results/issue_<N>/`` NOR
    ``figures/issue_<N>/`` during the run — declaring either is a load-bearing
    false-FAIL. Only ``extra_git_paths`` survive.
    """
    sentinel = "/scratch/eval_results/issue_790/att-1/.completion-sentinel.json"
    decl = build_expected_artifacts_declaration(
        issue=790,
        sentinel_path=sentinel,
        custom_workload=False,
        attempt_id="att-1",
    )
    assert decl["git_paths"] == []
    # A hydra phase that DOES commit a scoped git file can still declare it.
    decl_extra = build_expected_artifacts_declaration(
        issue=790,
        sentinel_path=sentinel,
        custom_workload=False,
        attempt_id="att-1",
        extra_git_paths=("eval_results/issue_790/manifest.json",),
    )
    assert decl_extra["git_paths"] == ["eval_results/issue_790/manifest.json"]


def test_phase_scope_passes_without_skip_confirm_and_negative_control(tmp_path: Path) -> None:
    """#661: a phase-scoped declaration (no git paths, HF-only deliverable)
    PASSes the gate WITHOUT --skip-confirm-artifacts — the git check SKIPs,
    the HF + sentinel checks still run. Negative control: a wrong-PHASE
    sentinel on the SAME phase-scoped declaration STILL FAILs (gate not
    relaxed).
    """
    sentinel = str(tmp_path / ".sentinel.json")
    decl = build_expected_artifacts_declaration(
        issue=661,
        sentinel_path=sentinel,
        custom_workload=True,
        attempt_id="att-1",
        skip_default_git_paths=True,
        extra_hf_data_paths=("issue661_extract/analysis_tensors/",),
    )
    expected = ExpectedArtifacts(
        issue=661,
        hf_data_paths=tuple(decl["hf_data_paths"]),
        git_paths=tuple(decl["git_paths"]),
        sentinel_path=sentinel,
    )
    io_pass = _io(
        hf_data_files=["issue661_extract/analysis_tensors/shift_L10.pt"],
        sentinel_content=_good_sentinel_text(issue=661),
        repo_root=tmp_path,
    )
    verdict = verify_artifacts(expected, io=io_pass)
    assert verdict.passed, verdict.reasons
    assert verdict.checks[CHECK_GIT]["status"] == "SKIP"

    # Negative control: wrong-phase sentinel → STILL FAIL.
    io_bad = _io(
        hf_data_files=["issue661_extract/analysis_tensors/shift_L10.pt"],
        sentinel_content=json.dumps({"phase": "running", "issue": 661}) + "\n",
        repo_root=tmp_path,
    )
    verdict_bad = verify_artifacts(expected, io=io_bad)
    assert not verdict_bad.passed
    assert verdict_bad.checks[CHECK_SENTINEL]["status"] == "FAIL"


def test_stale_baked_sentinel_resolves_to_single_live_sibling(tmp_path: Path) -> None:
    """#685 secondary: the declared (stale baked-attempt) sentinel is
    missing; exactly ONE live sibling attempt-dir sentinel exists → the
    resolver prefers it and the sentinel check PASSes. The glob is injected
    so the test is FS-free.
    """
    issue = 685
    declared = f"/scratch/eval_results/issue_{issue}/att-STALE/{SENTINEL_FILENAME}"
    live_sibling = f"/scratch/eval_results/issue_{issue}/att-LIVE/{SENTINEL_FILENAME}"
    fs = {live_sibling: _good_sentinel_text(issue=issue)}  # declared NOT present
    expected = ExpectedArtifacts(issue=issue, sentinel_path=declared)
    io = VerifierIO(
        read_sentinel=lambda p: fs.get(p),
        glob_sentinels=lambda decl, iss: [live_sibling],
    )
    verdict = verify_artifacts(expected, io=io)
    assert verdict.passed, verdict.reasons
    assert verdict.checks[CHECK_SENTINEL]["status"] == "PASS"
    assert "att-LIVE" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_stale_baked_sentinel_zero_live_siblings_still_fails(tmp_path: Path) -> None:
    """#685 secondary negative control: declared missing AND no live sibling
    → the gate STILL FAILs (the real 'missing' reason), never a silent pass.
    """
    issue = 685
    declared = f"/scratch/eval_results/issue_{issue}/att-STALE/{SENTINEL_FILENAME}"
    expected = ExpectedArtifacts(issue=issue, sentinel_path=declared)
    io = VerifierIO(read_sentinel=lambda p: None, glob_sentinels=lambda decl, iss: [])
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert "missing" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_stale_baked_sentinel_two_live_siblings_refuses_to_guess(tmp_path: Path) -> None:
    """#685 secondary KNOWN LIMITATION: >=2 live siblings → do NOT guess;
    FAIL loud on the declared (missing) path. A wrong-attempt run is never
    PASSed on an ambiguous probe.
    """
    issue = 685
    declared = f"/scratch/eval_results/issue_{issue}/att-STALE/{SENTINEL_FILENAME}"
    sib_a = f"/scratch/eval_results/issue_{issue}/att-A/{SENTINEL_FILENAME}"
    sib_b = f"/scratch/eval_results/issue_{issue}/att-B/{SENTINEL_FILENAME}"
    fs = {sib_a: _good_sentinel_text(issue=issue), sib_b: _good_sentinel_text(issue=issue)}
    expected = ExpectedArtifacts(issue=issue, sentinel_path=declared)
    io = VerifierIO(
        read_sentinel=lambda p: fs.get(p), glob_sentinels=lambda decl, iss: [sib_a, sib_b]
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    # FAILs on the DECLARED missing path, not on a guessed sibling.
    assert "att-STALE" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_stale_baked_sentinel_wrong_issue_sibling_still_fails(tmp_path: Path) -> None:
    """#685 secondary negative control: the single live sibling exists but
    its content is for a DIFFERENT issue → the UNCHANGED issue-match content
    check FAILs. Resolution does not bypass the keystone validation.
    """
    issue = 685
    declared = f"/scratch/eval_results/issue_{issue}/att-STALE/{SENTINEL_FILENAME}"
    live_sibling = f"/scratch/eval_results/issue_{issue}/att-LIVE/{SENTINEL_FILENAME}"
    fs = {live_sibling: _good_sentinel_text(issue=999)}  # wrong issue
    expected = ExpectedArtifacts(issue=issue, sentinel_path=declared)
    io = VerifierIO(
        read_sentinel=lambda p: fs.get(p),
        glob_sentinels=lambda decl, iss: [live_sibling],
    )
    verdict = verify_artifacts(expected, io=io)
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert "999" in verdict.checks[CHECK_SENTINEL]["detail"]


# ---------------------------------------------------------------------------
# #709: SSH-backed ``glob_sentinels`` for RunPod ``confirm_artifacts`` — the
# live-pod sibling of the #705 local-FS resolution above — plus the two
# included hardening pieces (resolver scope guard, sibling-read wrap). The
# 1 / 0 / >=2 / wrong-issue contract is re-pinned END-TO-END through
# ``backend.confirm_artifacts`` with a dispatching fake ``subprocess.run``.
# ---------------------------------------------------------------------------


class _FakeProc:
    """Minimal ``subprocess.run`` result stand-in (rc / stdout / stderr)."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _pod_ssh_dispatcher(
    pod_fs: dict[str, str],
    *,
    ls_result: _FakeProc | None = None,
    argv_log: list[list[str]] | None = None,
):
    """Fake ``subprocess.run`` dispatching on the remote command string.

    ``cat '<path>'`` resolves against ``pod_fs`` (rc=0 + content when
    present, rc=1 + "No such file or directory" when absent, mirroring the
    real pod); ``ls -1d ...`` returns ``ls_result`` verbatim. Every argv is
    optionally recorded so tests can pin the exact SSH command shape. No
    real ``ssh pod-*`` can ever run from the suite.
    """

    def run(argv, **kwargs):
        if argv_log is not None:
            argv_log.append(list(argv))
        assert argv[0] == "ssh", argv
        cmd = argv[2]
        if cmd.startswith("ls -1d "):
            assert ls_result is not None, f"unexpected ls call: {cmd}"
            return ls_result
        assert cmd.startswith("cat "), cmd
        path = cmd[len("cat ") :].strip("'")
        if path in pod_fs:
            return _FakeProc(0, stdout=pod_fs[path])
        return _FakeProc(1, stderr=f"cat: {path}: No such file or directory")

    return run


def _runpod_sentinel_only_handle(*, issue: int, declared: str) -> RunHandle:
    """RunPod handle whose declaration carries ONLY issue + sentinel_path.

    Every other check class SKIPs (nothing declared), so the sentinel
    check — the property under test — is the sole live check.
    """
    return _handle_with_expected(
        backend="runpod",
        issue=issue,
        declaration={"issue": issue, "sentinel_path": declared},
    )


def test_runpod_ssh_glob_sentinels_command_shape_and_parsing(monkeypatch) -> None:
    """#709 test 1: the SSH glob's exact argv (quoted issue-dir prefix, an
    UNQUOTED ``*`` between the quoted segments so the remote bash expands
    it, quoted basename) and rc=0 parsing (stripped lines, SORTED — parity
    with the FS default's ``sorted(...)``)."""
    backend = RunPodBackend()
    declared = f"/workspace/eval_results/issue_42/rp-STALE/{SENTINEL_FILENAME}"
    sib_a = f"/workspace/eval_results/issue_42/rp-a/{SENTINEL_FILENAME}"
    sib_b = f"/workspace/eval_results/issue_42/rp-b/{SENTINEL_FILENAME}"
    handle = _runpod_sentinel_only_handle(issue=42, declared=declared)

    argv_log: list[list[str]] = []

    def run(argv, **kwargs):
        argv_log.append(list(argv))
        # Two lines UNSORTED with stray whitespace + a blank trailer.
        return _FakeProc(0, stdout=f" {sib_b} \n{sib_a}\n\n")

    monkeypatch.setattr("subprocess.run", run)
    result = backend._ssh_glob_sentinels(handle)(declared, 42)
    assert argv_log == [
        [
            "ssh",
            "pod-42",
            "ls -1d '/workspace/eval_results/issue_42'/*/'.completion-sentinel.json'",
        ]
    ]
    assert result == [sib_a, sib_b]  # sorted + stripped


def test_runpod_ssh_glob_sentinels_no_match_and_transport(monkeypatch) -> None:
    """#709 test 2: rc!=0 + "No such file or directory" (bash passed the
    unmatched glob literally to ``ls``) → [] (zero siblings); any OTHER
    non-zero rc (transport) → RAISE — a transport failure must never read
    as "no siblings"."""
    backend = RunPodBackend()
    declared = f"/workspace/eval_results/issue_42/rp-STALE/{SENTINEL_FILENAME}"
    handle = _runpod_sentinel_only_handle(issue=42, declared=declared)

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _FakeProc(
            2, stderr="ls: cannot access '/workspace/...': No such file or directory"
        ),
    )
    assert backend._ssh_glob_sentinels(handle)(declared, 42) == []

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: _FakeProc(255, stderr="ssh: connect to host pod-42 port 22 failed"),
    )
    with pytest.raises(RuntimeError, match="rc=255"):
        backend._ssh_glob_sentinels(handle)(declared, 42)


def test_runpod_confirm_resolves_single_live_pod_side_sibling(monkeypatch) -> None:
    """#709 test 3 (the headline end-to-end): the declared (stale baked
    attempt) sentinel is missing ON THE POD and exactly ONE live sibling
    attempt-dir sentinel exists pod-side → ``backend.confirm_artifacts``
    resolves it over SSH and PASSes. Mirror of the #705 local-FS
    ``test_stale_baked_sentinel_resolves_to_single_live_sibling``."""
    issue = 42
    declared = f"/workspace/eval_results/issue_{issue}/rp-STALE/{SENTINEL_FILENAME}"
    live_sibling = f"/workspace/eval_results/issue_{issue}/rp-LIVE/{SENTINEL_FILENAME}"
    handle = _runpod_sentinel_only_handle(issue=issue, declared=declared)
    backend = RunPodBackend()
    monkeypatch.setattr(
        "subprocess.run",
        _pod_ssh_dispatcher(
            {live_sibling: _good_sentinel_text(issue=issue)},
            ls_result=_FakeProc(0, stdout=live_sibling + "\n"),
        ),
    )
    assert backend.confirm_artifacts(handle) is True


def test_runpod_confirm_zero_remote_siblings_still_fails(monkeypatch) -> None:
    """#709 test 4 (gate not relaxed): declared missing on the pod AND zero
    remote siblings → FAIL with the real "missing" reason. Mirror of
    ``test_stale_baked_sentinel_zero_live_siblings_still_fails``."""
    issue = 42
    declared = f"/workspace/eval_results/issue_{issue}/rp-STALE/{SENTINEL_FILENAME}"
    handle = _runpod_sentinel_only_handle(issue=issue, declared=declared)
    backend = RunPodBackend()
    monkeypatch.setattr(
        "subprocess.run",
        _pod_ssh_dispatcher(
            {},
            ls_result=_FakeProc(
                2, stderr="ls: cannot access '/workspace/...': No such file or directory"
            ),
        ),
    )
    assert backend.confirm_artifacts(handle) is False
    verdict = confirm_artifacts_from_handle(
        handle,
        io=VerifierIO(
            read_sentinel=backend._ssh_read_sentinel(handle),
            glob_sentinels=backend._ssh_glob_sentinels(handle),
        ),
    )
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert f"missing at {declared}" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_runpod_confirm_two_remote_siblings_refuses_to_guess(monkeypatch) -> None:
    """#709 test 5 (the #705 known-limitation contract, unchanged on the
    SSH path): >=2 live remote siblings → do NOT guess; FAIL on the
    DECLARED attempt's path."""
    issue = 42
    declared = f"/workspace/eval_results/issue_{issue}/rp-STALE/{SENTINEL_FILENAME}"
    sib_a = f"/workspace/eval_results/issue_{issue}/rp-A/{SENTINEL_FILENAME}"
    sib_b = f"/workspace/eval_results/issue_{issue}/rp-B/{SENTINEL_FILENAME}"
    handle = _runpod_sentinel_only_handle(issue=issue, declared=declared)
    backend = RunPodBackend()
    monkeypatch.setattr(
        "subprocess.run",
        _pod_ssh_dispatcher(
            {
                sib_a: _good_sentinel_text(issue=issue),
                sib_b: _good_sentinel_text(issue=issue),
            },
            ls_result=_FakeProc(0, stdout=f"{sib_a}\n{sib_b}\n"),
        ),
    )
    assert backend.confirm_artifacts(handle) is False
    verdict = confirm_artifacts_from_handle(
        handle,
        io=VerifierIO(
            read_sentinel=backend._ssh_read_sentinel(handle),
            glob_sentinels=backend._ssh_glob_sentinels(handle),
        ),
    )
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    # FAILs on the DECLARED attempt id, not on a guessed sibling.
    assert "rp-STALE" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_runpod_confirm_wrong_issue_remote_sibling_still_fails(monkeypatch) -> None:
    """#709 test 6 (content checks unchanged — resolution only): the single
    live remote sibling carries a DIFFERENT issue's content → the UNCHANGED
    issue-match content check FAILs."""
    issue = 42
    declared = f"/workspace/eval_results/issue_{issue}/rp-STALE/{SENTINEL_FILENAME}"
    live_sibling = f"/workspace/eval_results/issue_{issue}/rp-LIVE/{SENTINEL_FILENAME}"
    handle = _runpod_sentinel_only_handle(issue=issue, declared=declared)
    backend = RunPodBackend()
    monkeypatch.setattr(
        "subprocess.run",
        _pod_ssh_dispatcher(
            {live_sibling: _good_sentinel_text(issue=999)},  # wrong issue
            ls_result=_FakeProc(0, stdout=live_sibling + "\n"),
        ),
    )
    assert backend.confirm_artifacts(handle) is False
    verdict = confirm_artifacts_from_handle(
        handle,
        io=VerifierIO(
            read_sentinel=backend._ssh_read_sentinel(handle),
            glob_sentinels=backend._ssh_glob_sentinels(handle),
        ),
    )
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert "999" in verdict.checks[CHECK_SENTINEL]["detail"]


def test_sentinel_probe_scope_guard_blocks_noncanonical_declared() -> None:
    """#709 test 7 (scope guard): a declared path that does not match the
    canonical ``.../eval_results/issue_<N>/<attempt>/<SENTINEL_FILENAME>``
    shape NEVER invokes the sibling probe — the verdict is the declared
    FAIL even when a probe WOULD have returned a live sibling. Positive
    control: the canonical shape DOES invoke the probe and resolves."""
    issue = 685
    live_sibling = f"/scratch/eval_results/issue_{issue}/att-LIVE/{SENTINEL_FILENAME}"
    fs = {live_sibling: _good_sentinel_text(issue=issue)}
    glob_calls: list[tuple[str, int]] = []

    def recording_glob(declared: str, iss: int) -> list[str]:
        glob_calls.append((declared, iss))
        return [live_sibling]

    io = VerifierIO(read_sentinel=lambda p: fs.get(p), glob_sentinels=recording_glob)

    noncanonical = [
        # grandparent != issue_<N>
        f"/tmp/odd/x/{SENTINEL_FILENAME}",
        # wrong basename
        f"/scratch/eval_results/issue_{issue}/att-STALE/sentinel.json",
        # great-grandparent != eval_results
        f"/scratch/other_results/issue_{issue}/att-STALE/{SENTINEL_FILENAME}",
    ]
    for declared in noncanonical:
        expected = ExpectedArtifacts(issue=issue, sentinel_path=declared)
        verdict = verify_artifacts(expected, io=io)
        assert not verdict.passed, declared
        assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
        assert f"missing at {declared}" in verdict.checks[CHECK_SENTINEL]["detail"]
    assert glob_calls == []  # the probe was NEVER invoked on a non-canonical shape

    canonical = f"/scratch/eval_results/issue_{issue}/att-STALE/{SENTINEL_FILENAME}"
    expected = ExpectedArtifacts(issue=issue, sentinel_path=canonical)
    verdict = verify_artifacts(expected, io=io)
    assert verdict.passed, verdict.reasons
    assert glob_calls == [(canonical, issue)]


def test_sibling_liveness_read_error_fails_structured_not_raise(caplog) -> None:
    """#709 test 8 (sibling-read wrap): a sibling liveness read that RAISES
    (the SSH reader's transport contract) yields a structured declared-path
    FAIL — no exception escapes ``verify_artifacts`` — and the probe is
    refused even though ANOTHER sibling read live (never-guess: the errored
    sibling might be the right attempt)."""
    issue = 685
    declared = f"/scratch/eval_results/issue_{issue}/att-STALE/{SENTINEL_FILENAME}"
    sib_a = f"/scratch/eval_results/issue_{issue}/att-A/{SENTINEL_FILENAME}"
    sib_b = f"/scratch/eval_results/issue_{issue}/att-B/{SENTINEL_FILENAME}"

    def read(path: str) -> str | None:
        if path == declared:
            return None
        if path == sib_a:
            raise RuntimeError("ssh sentinel read from pod-685 failed rc=255")
        return _good_sentinel_text(issue=issue)

    expected = ExpectedArtifacts(issue=issue, sentinel_path=declared)
    io = VerifierIO(read_sentinel=read, glob_sentinels=lambda decl, iss: [sib_a, sib_b])
    with caplog.at_level(logging.WARNING):
        verdict = verify_artifacts(expected, io=io)  # returns a verdict, never raises
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    assert f"missing at {declared}" in verdict.checks[CHECK_SENTINEL]["detail"]
    assert "probe inconclusive" in caplog.text


def test_runpod_confirm_glob_transport_failure_fails_declared_not_raise(
    monkeypatch, caplog
) -> None:
    """#709 test 9 (round-1 reconciler Must-Fix, binding): a raising SSH
    glob (rc=255 transport error) during the sibling probe never escapes
    ``verify_artifacts`` — end-to-end through ``backend.confirm_artifacts``
    the verdict is FAIL on the DECLARED path with the "live-sibling probe
    raised" note. Pins the resolver's glob-call try/except, which the SSH
    provider makes load-bearing: deleting/narrowing that except would pass
    every other test while turning a transport blip into an
    orchestrator-finalize crash."""
    issue = 42
    declared = f"/workspace/eval_results/issue_{issue}/rp-STALE/{SENTINEL_FILENAME}"
    handle = _runpod_sentinel_only_handle(issue=issue, declared=declared)
    backend = RunPodBackend()
    monkeypatch.setattr(
        "subprocess.run",
        _pod_ssh_dispatcher(
            {},  # declared missing on the pod
            ls_result=_FakeProc(
                255, stderr="ssh: connect to host pod-42 port 22: Connection refused"
            ),
        ),
    )
    with caplog.at_level(logging.WARNING):
        # (a) NO exception escapes — the gate returns a normal False.
        assert backend.confirm_artifacts(handle) is False
        verdict = confirm_artifacts_from_handle(
            handle,
            io=VerifierIO(
                read_sentinel=backend._ssh_read_sentinel(handle),
                glob_sentinels=backend._ssh_glob_sentinels(handle),
            ),
        )
    assert not verdict.passed
    assert verdict.checks[CHECK_SENTINEL]["status"] == "FAIL"
    # (b) FAILs on the DECLARED (missing) path.
    assert f"missing at {declared}" in verdict.checks[CHECK_SENTINEL]["detail"]
    # (c) the probe-failure note surfaces at WARNING.
    assert "live-sibling probe raised" in caplog.text


def test_new_fields_default_off_round_trip_back_compat(tmp_path: Path) -> None:
    """Back-compat (#705 constraint 6): a declaration built WITHOUT the new
    fields omits ``git_repo_root`` entirely and a pre-fix serialized dict
    (no ``git_repo_root`` key) round-trips to ``git_repo_root=None`` with
    identical verdict behavior.
    """
    decl = build_expected_artifacts_declaration(
        issue=137,
        sentinel_path=str(tmp_path / ".sentinel.json"),
        custom_workload=True,
        attempt_id="att-1",
    )
    assert "git_repo_root" not in decl  # omitted when None

    # A pre-fix in-flight handle (no git_repo_root key) reconstructs fine.
    handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name="pod-137",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-137.log",
        extra={EXPECTED_ARTIFACTS_HANDLE_KEY: decl},
    )
    from explore_persona_space.backends.artifacts import expected_artifacts_from_handle

    rebuilt = expected_artifacts_from_handle(handle)
    assert rebuilt is not None
    assert rebuilt.git_repo_root is None


# ---------------------------------------------------------------------------
# #988 — scoped per-path HF listings in _check_hf_paths
# ---------------------------------------------------------------------------


def test_check_hf_paths_scopes_listing_per_declared_path() -> None:
    """#988: ``_check_hf_paths`` threads ``path_in_repo`` per declared path
    (one scoped call each, in declaration order) and the verdict semantics
    (SKIP / PASS / FAIL-missing / FAIL-raised) are unchanged."""
    from explore_persona_space.backends.artifacts import VerifierIO, _check_hf_paths

    calls: list[str | None] = []

    def _spy(repo_id, *, repo_type, revision=None, path_in_repo=None):
        calls.append(path_in_repo)
        if path_in_repo == "issue1/present":
            return ["issue1/present/file.json"]
        return []

    io = VerifierIO(list_hf_repo_files=_spy)

    # SKIP: no declared paths -> no listing call at all.
    res = _check_hf_paths(repo_id="org/data", repo_type="dataset", paths=(), io=io)
    assert res["status"] == "SKIP"
    assert calls == []

    # PASS: the one declared dir path resolves via ONE scoped call
    # (trailing slash stripped for the server-side kwarg).
    res = _check_hf_paths(
        repo_id="org/data", repo_type="dataset", paths=("issue1/present/",), io=io
    )
    assert res["status"] == "PASS"
    assert calls == ["issue1/present"]

    # FAIL-missing: declaration order preserved in the calls AND the detail.
    calls.clear()
    res = _check_hf_paths(
        repo_id="org/data",
        repo_type="dataset",
        paths=("issue1/present/", "issue1/ghost/"),
        io=io,
    )
    assert res["status"] == "FAIL"
    assert "issue1/ghost/" in res["detail"]
    assert calls == ["issue1/present", "issue1/ghost"]

    # FAIL-raised: a transport error on any per-path call surfaces as FAIL
    # with the reason (never a silent pass).
    def _raise(repo_id, *, repo_type, revision=None, path_in_repo=None):
        raise RuntimeError("HF Hub 503")

    res = _check_hf_paths(
        repo_id="org/data",
        repo_type="dataset",
        paths=("issue1/present/",),
        io=VerifierIO(list_hf_repo_files=_raise),
    )
    assert res["status"] == "FAIL"
    assert "HF Hub 503" in res["detail"]


def test_default_list_hf_repo_files_body_scoped_and_full(monkeypatch) -> None:
    """#988 body test (code-style: one production-body test per seam-stubbed
    function): execute the REAL ``_default_list_hf_repo_files`` — fakes only at
    the Hub boundary (``HfApi`` construction + the paginated
    ``list_repo_files_complete`` walk, signature-conformant by construction).
    ``path_in_repo`` routes through the REAL ``list_hf_files_under_path`` body
    (scoped server-side kwarg threaded, trailing slash normalized); ``None``
    keeps the seam-compat full listing."""
    import huggingface_hub

    import explore_persona_space.orchestrate.hub as hub
    from explore_persona_space.backends.artifacts import _default_list_hf_repo_files

    class _StubApi:
        def __init__(self, token=None):
            self.token = token

    monkeypatch.setattr(huggingface_hub, "HfApi", _StubApi)

    calls: list[tuple[str, str, str | None, str | None]] = []

    def _fake_complete(api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
        assert isinstance(api, _StubApi), "the stub api must reach the scoped walk"
        calls.append((repo_id, repo_type, revision, path_in_repo))
        if path_in_repo is not None:
            return [f"{path_in_repo}/a.json"]
        return ["root.json", "issue1/a.json"]

    monkeypatch.setattr(hub, "list_repo_files_complete", _fake_complete)

    out = _default_list_hf_repo_files("org/data", repo_type="dataset", path_in_repo="issue1/raw/")
    assert out == ["issue1/raw/a.json"]
    assert calls[-1] == ("org/data", "dataset", None, "issue1/raw")

    out = _default_list_hf_repo_files("org/data", repo_type="dataset")
    assert out == ["root.json", "issue1/a.json"]
    assert calls[-1] == ("org/data", "dataset", None, None)
