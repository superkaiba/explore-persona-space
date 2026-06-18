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
    from explore_persona_space.backends.artifacts import build_expected_artifacts_declaration
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
