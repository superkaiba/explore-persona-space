"""Backend-agnostic artifact verification (the ``confirm_artifacts`` core).

Every lane (SLURM, RunPod, GCP — and any future backend) gates teardown on
``backend.confirm_artifacts(handle)`` returning ``True``. Before this module
landed, only the RunPod path had a real implementation (the
``upload-verifier`` agent that shells out to ``scripts/verify_uploads.py``);
``SlurmBackend.confirm_artifacts`` and the cluster-stub raised
``NotImplementedError``, which would have stranded the cluster lane the
moment slice-3 GCP went live (orchestrator-side teardown would crash before
the cleanup ever ran, burning either a $100k credit pool or 6h-quota'd
free-cluster time on stale workloads).

The orchestrator's ``upload-verifier`` agent stays the canonical place for
the **active** discovery work (SSHing the pod, grepping for unuploaded
artifacts, reading the experiment code to figure out what should have
been written) — that's exploratory, narrative work that benefits from
an LLM in the loop. This module is the complementary **mechanical**
gate: given an explicit declaration of what artifacts the run was
supposed to produce, verify each is actually parked at a permanent URL
or committed to git, with no possibility of the upload-verifier agent's
optimism papering over a missing file.

The two work in series: the orchestrator runs the agent (which produces
the declaration of expected artifacts), then calls
``backend.confirm_artifacts(handle)`` which delegates here. A PASS from
this module is a no-trust mechanical check; a PASS from the agent is the
"I went looking and didn't find anything else" check. Both must pass.

Design contract
---------------

* **Backend-agnostic.** The verifier knows nothing about RunPod / SLURM /
  GCP. It takes an :class:`ExpectedArtifacts` declaration — what HF Hub
  paths, WandB run, git-tracked files, and completion sentinel SHOULD
  exist — and returns a verdict. The backends are responsible for
  deriving the expected artifacts from their ``RunHandle`` / ``RunSpec``
  context.
* **Fail-fast, never silently True.** A check that cannot run (HF Hub
  unreachable, WandB transport error, git repo missing) returns FAIL
  with an explicit reason. Per CLAUDE.md the project NEVER allows
  ``try/except: pass`` or "silent True on transport error" patterns —
  better to bounce teardown than to silently pass with no real signal.
* **Dependency-injectable.** Every external call (HF ``list_repo_files``,
  WandB run resolution, git ls-files, filesystem read for the sentinel)
  is passed in as a callable on :class:`VerifierIO`, so tests run with
  NO network / git side effects. Defaults wire to the real implementations.
* **Sentinel-driven completion proof.** Beyond checking that *files
  exist*, the verifier requires a small ``completion-sentinel.json``
  that the workload writes only on clean exit. This separates
  "intentional completion" from "incidental file presence" — e.g. a
  crashed job that managed to upload one shard of three would still
  fail the sentinel check.
* **No backend-specific paths.** The verifier accepts the sentinel path
  through ``ExpectedArtifacts.sentinel_path`` so each backend can point
  at its own location (RunPod = ``/workspace/eval_results/issue_<N>/
  .completion-sentinel.json``; SLURM = ``$SCRATCH_JOB_DIR/eval_results/
  issue_<N>/.completion-sentinel.json``; GCP = the same path inside the
  attached PD). The verifier reads its contents via the injected
  ``read_sentinel`` callable so tests don't need a real FS.

The verdict
-----------

:class:`ArtifactVerdict` carries ``.passed`` (the bool the ABC contract
needs) AND a structured ``.checks`` dict so the orchestrator can log
the exact reason for a FAIL into the ``epm:upload-verify-failed`` marker
without re-running the helper.

References:
* CLAUDE.md § "Upload Policy" — the destination table that drives what
  the verifier checks (eval JSON → git, raw completions → HF data repo,
  adapter/checkpoint → HF model repo, training metrics → WandB).
* ``.claude/rules/upload-policy.md`` — Hub-API verification mechanics
  (the ``hf`` CLI has NO ``api`` subcommand → false "0 files"; use
  ``huggingface_hub.list_repo_files`` only).
* ``scripts/verify_uploads.py`` — the legacy CLI helper the
  upload-verifier agent shells out to. Stays untouched; the agent keeps
  invoking it for the exploratory pass. The mechanical Python helper
  here is the seat ``backend.confirm_artifacts`` calls.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public repo defaults (mirror scripts/verify_uploads.py + Upload Policy)
# ---------------------------------------------------------------------------

#: HF Hub model repo where adapters / merged checkpoints land. Mirrors
#: ``scripts/verify_uploads.py.HF_MODEL_REPO`` and the Upload Policy table.
DEFAULT_HF_MODEL_REPO = "superkaiba1/explore-persona-space"

#: HF Hub data repo where raw completions + training datasets land.
DEFAULT_HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

#: Canonical filename for the per-run completion sentinel. The workload
#: writes this JSON on clean exit; the verifier reads it back. Living in
#: ``eval_results/issue_<N>/`` (NOT under raw completions) keeps it on the
#: VM-side rsync'd tree where the verifier runs.
SENTINEL_FILENAME = ".completion-sentinel.json"


# ---------------------------------------------------------------------------
# Inputs — what the backend declares for verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedArtifacts:
    """Declaration of WHAT should be present after a clean run.

    The backend constructs one of these from its :class:`RunHandle` (and
    optionally the originating :class:`RunSpec`) before delegating to
    :func:`verify_artifacts`. Every field is optional so a partial
    declaration (e.g. ``eval``-only run with no model checkpoint) skips
    the matching check rather than failing it.

    Fields:

    * ``issue`` — task id (used in error messages + sentinel-issue match).
    * ``hf_data_paths`` — sequence of in-data-repo prefixes that must
      resolve to >=1 file each. Example:
      ``("issue137_warmth/raw_completions/",)``. Trailing slash means
      "files under this directory"; an exact-file path matches itself.
    * ``hf_model_paths`` — sequence of in-model-repo prefixes (e.g.
      ``("issue-137-c1-seed-42/",)``). Same matching rule as data paths.
    * ``hf_data_repo`` / ``hf_model_repo`` — override the default repos
      (defaults to the project-canonical repos above). Tests inject test
      repo ids; production uses the defaults.
    * ``wandb_run_path`` — ``"<entity>/<project>/runs/<run_id>"`` form
      (matches ``wandb.Api().run(path)``). ``None`` skips the WandB
      check entirely (e.g. an eval-only run with no training metrics).
    * ``git_paths`` — repo-relative paths that must be tracked by git
      AND present in the working tree (covers `eval_results/...json`
      and `figures/issue_<N>/...png`).
    * ``sentinel_path`` — absolute path to the completion sentinel JSON
      the workload wrote on clean exit. The verifier checks the file
      exists, parses as JSON, has ``"phase": "done"`` and a matching
      ``"issue": <issue>``. ``None`` skips the sentinel check — but a
      production run NEVER skips this; missing it is the silent-loss
      hole the gate is designed to close.
    """

    issue: int
    hf_data_paths: tuple[str, ...] = ()
    hf_model_paths: tuple[str, ...] = ()
    hf_data_repo: str = DEFAULT_HF_DATA_REPO
    hf_model_repo: str = DEFAULT_HF_MODEL_REPO
    wandb_run_path: str | None = None
    git_paths: tuple[str, ...] = ()
    sentinel_path: str | None = None


# ---------------------------------------------------------------------------
# Verdict — what the verifier returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactVerdict:
    """Verdict object returned by :func:`verify_artifacts`.

    Fields:

    * ``passed`` — ``True`` iff every requested check returned PASS. The
      orchestrator turns this into the ``bool`` the
      :meth:`ComputeBackend.confirm_artifacts` ABC contract needs.
    * ``reasons`` — sequence of human-readable FAIL strings (one per
      failing check). Empty on PASS. Used by the orchestrator when
      posting ``epm:upload-verify-failed v1`` so the marker carries the
      exact reasons without re-running the helper.
    * ``checks`` — structured per-check status dict: ``{check_name:
      {"status": "PASS"|"FAIL"|"SKIP", "detail": "..."}}``. Stable schema
      for downstream tools (the dashboard surfaces these as columns).
    """

    passed: bool
    reasons: tuple[str, ...] = ()
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# I/O seam — dependency injection for external systems
# ---------------------------------------------------------------------------


def _default_list_hf_repo_files(
    repo_id: str,
    *,
    repo_type: str,
    revision: str | None = None,
) -> list[str]:
    """Default HF Hub file lister.

    Uses ``huggingface_hub.list_repo_files`` — the API the upload-policy
    rule pins as authoritative (``hf`` CLI has no ``api`` subcommand, so
    it silently returns 0 files; never use it here). Raises on transport
    / auth failure — the verifier turns that into a FAIL with reason
    rather than silently passing.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return list(api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision))


def _default_wandb_run_exists(run_path: str) -> bool:
    """Default WandB run resolver.

    Mirrors ``orchestrate/hub.py._wandb_run_exists`` but takes the full
    ``<entity>/<project>/runs/<run_id>`` path (the same form
    ``scripts/verify_uploads.py.check_wandb_run`` takes). A 404 returns
    ``False``; transport errors propagate.
    """
    import wandb

    api = wandb.Api()
    try:
        api.run(run_path)
        return True
    except wandb.errors.CommError as exc:
        msg = str(exc).lower()
        if "could not find" in msg or "404" in msg or "not found" in msg:
            return False
        raise


def _default_git_tracked(repo_root: Path, rel_paths: Iterable[str]) -> set[str]:
    """Default ``git ls-files`` checker.

    Returns the subset of ``rel_paths`` that ``git ls-files`` reports as
    tracked. Runs ONE ``git ls-files -- <p1> <p2> ...`` call rather than
    N — git resolves the union internally and reports the tracked subset.
    Raises ``CalledProcessError`` on a non-zero git exit (e.g. not a
    repo); the verifier turns that into a FAIL.
    """
    rel_list = list(rel_paths)
    if not rel_list:
        return set()
    argv = ["git", "-C", str(repo_root), "ls-files", "--", *rel_list]
    proc = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=30)
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _default_read_sentinel(path: str) -> str | None:
    """Default sentinel reader: read UTF-8 bytes, or ``None`` if missing.

    ``None`` means the sentinel file does not exist (verifier reads this
    as a FAIL with reason). Other I/O errors (permission denied, decode
    failure on non-UTF8 bytes) propagate so the caller can distinguish
    "missing" from "broken".
    """
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


@dataclass(frozen=True)
class VerifierIO:
    """Bundle of injectable I/O callables.

    Tests construct a :class:`VerifierIO` with mocks for each callable so
    the verifier runs with no real HF / WandB / git / FS side effects.
    The default-constructed instance wires every callable to its real
    implementation above.

    Fields:

    * ``list_hf_repo_files(repo_id, *, repo_type, revision=None) ->
      list[str]`` — must enumerate every file in the repo at the given
      revision. ``None`` revision = repo default.
    * ``wandb_run_exists(run_path) -> bool`` — must return True iff the
      WandB run resolves. Transport errors propagate.
    * ``git_tracked(repo_root, rel_paths) -> set[str]`` — must return the
      tracked subset of ``rel_paths`` (relative to ``repo_root``).
    * ``read_sentinel(path) -> str | None`` — must return the sentinel
      file's UTF-8 content, or ``None`` when the file does not exist.
    * ``repo_root`` — repo root for git checks; defaults to the package's
      grandparent walk (the same logic SlurmBackend uses).

    The callable defaults are ``None`` here rather than the
    ``_default_*`` functions; the verifier resolves the live module-level
    attribute at call time so a test ``monkeypatch.setattr("...
    _default_list_hf_repo_files", ...)`` is honored. Binding the function
    object at dataclass-default-resolution time (module import) would
    freeze the real implementation and silently ignore the patch.
    """

    list_hf_repo_files: Callable[..., list[str]] | None = None
    wandb_run_exists: Callable[[str], bool] | None = None
    git_tracked: Callable[[Path, Iterable[str]], set[str]] | None = None
    read_sentinel: Callable[[str], str | None] | None = None
    repo_root: Path | None = None

    def _list_hf(self) -> Callable[..., list[str]]:
        return self.list_hf_repo_files or _default_list_hf_repo_files

    def _wandb(self) -> Callable[[str], bool]:
        return self.wandb_run_exists or _default_wandb_run_exists

    def _git(self) -> Callable[[Path, Iterable[str]], set[str]]:
        return self.git_tracked or _default_git_tracked

    def _sentinel(self) -> Callable[[str], str | None]:
        return self.read_sentinel or _default_read_sentinel


def _resolve_repo_root(io: VerifierIO) -> Path:
    """Resolve repo root: explicit override > pyproject walk > cwd fallback."""
    if io.repo_root is not None:
        return io.repo_root
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# Sentinel writer (for the workload to call on clean exit)
# ---------------------------------------------------------------------------


def write_completion_sentinel(
    *,
    sentinel_path: str | Path,
    issue: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the per-run completion sentinel.

    The workload calls this on clean exit (the same place it would post
    ``[phase=done]``). The verifier's sentinel check reads the file back
    and asserts ``phase == "done"`` plus a matching ``issue``. Any
    additional fields the caller wants to record (commit SHA, wandb run
    URL, host) go into ``extra`` and are serialized alongside.

    Returns the resolved path so the caller can log it.

    Raises ``OSError`` on a failed write — fail-loud is intentional here
    (a silent failure to write the sentinel means a successful run
    silently fails verification later).
    """
    p = Path(sentinel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "phase": "done",
        "issue": int(issue),
    }
    if extra:
        payload.update(extra)
    p.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Per-class checks
# ---------------------------------------------------------------------------


def _path_matches(file_list: Iterable[str], path: str) -> bool:
    """Match ``path`` against an HF Hub repo file listing.

    A trailing-slash path means "any file under this dir"; a path without
    a trailing slash matches an exact file OR any file under that prefix
    (the same semantics ``scripts/verify_uploads.py::check_hf_hub_path``
    uses, kept identical so the two helpers agree on what "present"
    means). Empty path is invalid here (callers should not pass it).
    """
    if not path:
        raise ValueError("_path_matches: empty path is not a valid declaration")
    exact = path.rstrip("/")
    prefix = exact + "/"
    return any(f == exact or f.startswith(prefix) for f in file_list)


def _check_hf_paths(
    *,
    repo_id: str,
    repo_type: str,
    paths: tuple[str, ...],
    io: VerifierIO,
) -> dict[str, Any]:
    """Run the HF Hub presence check for one set of in-repo paths.

    Returns one ``{"status", "detail"}`` dict. SKIP when no paths were
    declared; PASS when every path resolved; FAIL with the missing list
    when any did not. Transport / auth errors propagate (the caller
    turns them into a FAIL with reason).
    """
    if not paths:
        return {"status": "SKIP", "detail": "no paths declared"}
    try:
        files = io._list_hf()(repo_id, repo_type=repo_type)
    except Exception as exc:
        # Fail-loud per CLAUDE.md "no silent True on transport error".
        # We catch + surface as FAIL (not re-raise) so the verdict's
        # `checks` dict carries the reason; otherwise the orchestrator
        # would see an uncaught exception with no structured signal.
        return {
            "status": "FAIL",
            "detail": f"HF list_repo_files({repo_id!r}, {repo_type!r}) raised: {exc}",
        }
    missing = [p for p in paths if not _path_matches(files, p)]
    if missing:
        return {
            "status": "FAIL",
            "detail": (f"HF Hub {repo_type} repo {repo_id!r} missing paths: " + "; ".join(missing)),
        }
    return {
        "status": "PASS",
        "detail": f"all {len(paths)} {repo_type} path(s) resolve in {repo_id}",
    }


def _check_wandb(
    *,
    run_path: str | None,
    io: VerifierIO,
) -> dict[str, Any]:
    """Resolve the WandB run via the injected callable.

    SKIP if no run path was declared; PASS if it resolves; FAIL with the
    reason if it doesn't or if the transport fails (note: the default
    callable lets transport errors propagate — the wrapper here catches
    them so the verdict carries the message).
    """
    if not run_path:
        return {"status": "SKIP", "detail": "no wandb_run_path declared"}
    try:
        exists = io._wandb()(run_path)
    except Exception as exc:
        return {
            "status": "FAIL",
            "detail": f"WandB run lookup raised: {exc}",
        }
    if not exists:
        return {
            "status": "FAIL",
            "detail": f"WandB run not found: {run_path}",
        }
    return {"status": "PASS", "detail": f"WandB run resolved: {run_path}"}


def _check_git(
    *,
    paths: tuple[str, ...],
    io: VerifierIO,
) -> dict[str, Any]:
    """Confirm every declared path is tracked by git AND present on disk.

    SKIP if no paths were declared. Both conditions must hold: a path
    tracked but deleted from the working tree fails the second check; an
    untracked file in the tree fails the first.
    """
    if not paths:
        return {"status": "SKIP", "detail": "no git paths declared"}
    repo_root = _resolve_repo_root(io)
    try:
        tracked = io._git()(repo_root, paths)
    except subprocess.CalledProcessError as exc:
        return {
            "status": "FAIL",
            "detail": f"git ls-files failed: exit={exc.returncode} stderr={exc.stderr!r}",
        }
    except Exception as exc:
        return {"status": "FAIL", "detail": f"git ls-files raised: {exc}"}
    missing_tracked = [p for p in paths if p not in tracked]
    missing_on_disk = [p for p in paths if not (repo_root / p).exists()]
    problems: list[str] = []
    if missing_tracked:
        problems.append("not tracked by git: " + "; ".join(missing_tracked))
    if missing_on_disk:
        problems.append("not on disk: " + "; ".join(missing_on_disk))
    if problems:
        return {"status": "FAIL", "detail": " | ".join(problems)}
    return {"status": "PASS", "detail": f"all {len(paths)} path(s) tracked + on disk"}


def _check_sentinel(
    *,
    sentinel_path: str | None,
    issue: int,
    io: VerifierIO,
) -> dict[str, Any]:
    """Verify the completion sentinel exists, parses, and claims phase=done.

    SKIP if no sentinel_path declared. FAIL when the file is missing,
    non-JSON, lacks ``phase: done``, or has a mismatched issue. This is
    the keystone check — file presence alone is not enough; the sentinel
    is what distinguishes an intentional clean run from leftover bytes
    of a half-finished one.
    """
    if not sentinel_path:
        return {"status": "SKIP", "detail": "no sentinel_path declared"}
    try:
        content = io._sentinel()(sentinel_path)
    except Exception as exc:
        return {"status": "FAIL", "detail": f"sentinel read raised: {exc}"}
    if content is None:
        return {
            "status": "FAIL",
            "detail": f"completion sentinel missing at {sentinel_path}",
        }
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            "status": "FAIL",
            "detail": f"sentinel at {sentinel_path} is not valid JSON: {exc}",
        }
    if not isinstance(data, dict):
        return {
            "status": "FAIL",
            "detail": f"sentinel at {sentinel_path} is not a JSON object",
        }
    phase = data.get("phase")
    if phase != "done":
        return {
            "status": "FAIL",
            "detail": f"sentinel at {sentinel_path} has phase={phase!r} (expected 'done')",
        }
    sentinel_issue = data.get("issue")
    if sentinel_issue is None:
        return {
            "status": "FAIL",
            "detail": f"sentinel at {sentinel_path} missing 'issue' field",
        }
    try:
        sentinel_issue_int = int(sentinel_issue)
    except (TypeError, ValueError):
        return {
            "status": "FAIL",
            "detail": f"sentinel at {sentinel_path} has non-integer issue={sentinel_issue!r}",
        }
    if sentinel_issue_int != int(issue):
        return {
            "status": "FAIL",
            "detail": (
                f"sentinel at {sentinel_path} has issue={sentinel_issue!r} "
                f"but verifier was called for issue={issue}"
            ),
        }
    return {"status": "PASS", "detail": f"sentinel valid at {sentinel_path}"}


# ---------------------------------------------------------------------------
# RunHandle bridge
# ---------------------------------------------------------------------------


#: Stable key under which a backend stuffs its :class:`ExpectedArtifacts`
#: declaration on :class:`RunHandle.extra`. The orchestrator builds the
#: declaration from the task plan at launch time (it knows which conditions /
#: seeds were planned and therefore which HF paths + git figures must land);
#: the backend just threads it through. Keeping the key + serialization
#: schema stable means SLURM / RunPod / GCP share one bridge — no
#: backend-specific extraction logic.
EXPECTED_ARTIFACTS_HANDLE_KEY = "expected_artifacts"


def expected_artifacts_from_handle(handle: Any) -> ExpectedArtifacts | None:
    """Reconstruct :class:`ExpectedArtifacts` from a handle's ``extra`` dict.

    The orchestrator stuffs a declaration into ``RunHandle.extra`` at
    launch time under :data:`EXPECTED_ARTIFACTS_HANDLE_KEY`. The backend's
    ``confirm_artifacts`` reads it back via this helper. Returns ``None``
    if no declaration is present (the caller decides whether that is a
    FAIL — in production it is, because every gate-bearing handle MUST
    carry the declaration; in tests the handle may legitimately omit it).

    The serialized form is a flat ``dict`` mirroring :class:`ExpectedArtifacts`
    fields. ``issue`` is required; everything else has the same defaults the
    dataclass uses. Tuple-typed fields accept lists in the serialized form
    (JSON-compatible) and are coerced to tuples on read.

    Raises ``KeyError`` only when the declaration is present but missing
    the required ``issue`` field — that is a programmer error, not a
    runtime condition.
    """
    extra = getattr(handle, "extra", None) or {}
    raw = extra.get(EXPECTED_ARTIFACTS_HANDLE_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(
            f"{EXPECTED_ARTIFACTS_HANDLE_KEY} on handle.extra must be a dict, "
            f"got {type(raw).__name__}"
        )
    if "issue" not in raw:
        raise KeyError(
            f"{EXPECTED_ARTIFACTS_HANDLE_KEY} missing required 'issue' field; "
            "the launch path must populate it"
        )
    return ExpectedArtifacts(
        issue=int(raw["issue"]),
        hf_data_paths=tuple(raw.get("hf_data_paths", ())),
        hf_model_paths=tuple(raw.get("hf_model_paths", ())),
        hf_data_repo=str(raw.get("hf_data_repo", DEFAULT_HF_DATA_REPO)),
        hf_model_repo=str(raw.get("hf_model_repo", DEFAULT_HF_MODEL_REPO)),
        wandb_run_path=raw.get("wandb_run_path"),
        git_paths=tuple(raw.get("git_paths", ())),
        sentinel_path=raw.get("sentinel_path"),
    )


def confirm_artifacts_from_handle(
    handle: Any,
    *,
    io: VerifierIO | None = None,
) -> ArtifactVerdict:
    """Convenience wrapper a backend's ``confirm_artifacts`` can call.

    Reads the :class:`ExpectedArtifacts` declaration off ``handle.extra``,
    runs :func:`verify_artifacts`, and returns the verdict. When no
    declaration is present, returns a FAIL with a clear reason (silently
    passing a handle that forgot to declare its artifacts is the exact
    silent-loss hole this module exists to close).

    The backend's ``confirm_artifacts`` is then a one-liner:
    ``return confirm_artifacts_from_handle(handle).passed``.
    """
    expected = expected_artifacts_from_handle(handle)
    if expected is None:
        return ArtifactVerdict(
            passed=False,
            reasons=(
                f"handle.extra is missing '{EXPECTED_ARTIFACTS_HANDLE_KEY}'; "
                "the launch path must populate it before teardown is gated",
            ),
            checks={},
        )
    verdict = verify_artifacts(expected, io=io)
    # The completion sentinel is the keystone per-run proof. A declaration that
    # SKIPs it (no sentinel_path) is the all-SKIP silent-pass hole this module
    # exists to close — fail loud rather than pass an unproven run. (A partial
    # slice-3/slice-6 launch wiring that forgets sentinel_path hits this.)
    if verdict.passed and verdict.checks.get(CHECK_SENTINEL, {}).get("status") == "SKIP":
        return ArtifactVerdict(
            passed=False,
            reasons=(
                "no completion sentinel declared (sentinel_path); refusing to pass "
                "an unverified run — the launch path must declare the sentinel",
            ),
            checks=verdict.checks,
        )
    return verdict


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Names of the checks the verifier runs. Stable schema so downstream
# tooling (the dashboard, ``epm:upload-verify-failed`` marker readers)
# can rely on the keys.
CHECK_HF_DATA = "hf_data"
CHECK_HF_MODEL = "hf_model"
CHECK_WANDB = "wandb"
CHECK_GIT = "git"
CHECK_SENTINEL = "sentinel"


def verify_artifacts(
    expected: ExpectedArtifacts,
    *,
    io: VerifierIO | None = None,
) -> ArtifactVerdict:
    """Run every applicable check against ``expected``; return a verdict.

    The verifier runs all declared checks (skipping any whose declaration
    is empty) and aggregates the result. A SKIP is NOT a FAIL — a run
    that legitimately produces no model checkpoint (eval-only) can leave
    ``hf_model_paths=()`` and still PASS.

    A PASS verdict means every declared artifact class resolved at its
    permanent home AND the completion sentinel proves the workload
    finished intentionally. The orchestrator can then call
    ``backend.teardown(handle)`` without risking silent data loss.

    A FAIL verdict carries the per-check reasons in ``.reasons`` AND in
    the structured ``.checks`` dict. The caller (orchestrator) is
    expected to surface both in the ``epm:upload-verify-failed v1``
    marker; teardown MUST NOT proceed on a FAIL.

    ``io`` lets tests inject mocks for every external call. Production
    code passes ``None`` (or omits it) to use the real wires.
    """
    io = io or VerifierIO()
    checks: dict[str, dict[str, Any]] = {
        CHECK_HF_DATA: _check_hf_paths(
            repo_id=expected.hf_data_repo,
            repo_type="dataset",
            paths=expected.hf_data_paths,
            io=io,
        ),
        CHECK_HF_MODEL: _check_hf_paths(
            repo_id=expected.hf_model_repo,
            repo_type="model",
            paths=expected.hf_model_paths,
            io=io,
        ),
        CHECK_WANDB: _check_wandb(run_path=expected.wandb_run_path, io=io),
        CHECK_GIT: _check_git(paths=expected.git_paths, io=io),
        CHECK_SENTINEL: _check_sentinel(
            sentinel_path=expected.sentinel_path,
            issue=expected.issue,
            io=io,
        ),
    }
    failures = [
        f"[{name}] {payload['detail']}"
        for name, payload in checks.items()
        if payload["status"] == "FAIL"
    ]
    passed = not failures
    if not passed:
        logger.info(
            "verify_artifacts(issue=%d) FAIL: %d/%d checks failed",
            expected.issue,
            len(failures),
            len(checks),
        )
    return ArtifactVerdict(passed=passed, reasons=tuple(failures), checks=checks)


__all__ = [
    "CHECK_GIT",
    "CHECK_HF_DATA",
    "CHECK_HF_MODEL",
    "CHECK_SENTINEL",
    "CHECK_WANDB",
    "DEFAULT_HF_DATA_REPO",
    "DEFAULT_HF_MODEL_REPO",
    "EXPECTED_ARTIFACTS_HANDLE_KEY",
    "SENTINEL_FILENAME",
    "ArtifactVerdict",
    "ExpectedArtifacts",
    "VerifierIO",
    "confirm_artifacts_from_handle",
    "expected_artifacts_from_handle",
    "verify_artifacts",
    "write_completion_sentinel",
]
