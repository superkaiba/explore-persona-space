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
  at its own location. Paths are ATTEMPT-NAMESPACED (#598): a prior
  attempt's sentinel in a reused scratch dir / persistent volume must
  never satisfy a fresh launch's declaration (``_check_sentinel``
  validates phase+issue only, so the PATH carries the staleness
  defense). RunPod = ``/workspace/eval_results/issue_<N>/<attempt>/
  .completion-sentinel.json``; SLURM = ``$SCRATCH_JOB_DIR/eval_results/
  issue_<N>/<attempt>/.completion-sentinel.json``; GCP = the same
  attempt-namespaced path inside the attached PD. The verifier reads
  the contents via the injected ``read_sentinel`` callable so tests
  don't need a real FS.

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
from collections.abc import Callable, Iterable, Sequence
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
    * ``git_repo_root`` — absolute path to the git working tree the
      ``git_paths`` check should resolve against. ``None`` (default,
      back-compat) means "resolve the repo root by the pyproject
      ``__file__``-walk" — the established behavior. The launch path
      sets it to the per-issue worktree
      (``<repo_root>/.claude/worktrees/issue-<N>``) when the run's code
      AND its committed eval/figure artifacts live on an unmerged
      ``issue-<N>`` branch checked out THERE, not on ``main`` — the #685
      root cause was ``_check_git`` running ``git ls-files`` from the
      MAIN root (on ``main``) while the files were committed only on the
      worktree branch, producing a structural ``not tracked by git``
      FAIL on a perfectly-uploaded run. When the baked directory no
      longer exists at finalize time (the post-Step-10d auto-merge case,
      where the worktree was merged + removed and the files are now on
      ``main``), the resolver falls back to the pyproject-walked main
      root with a LOUD log — so the gate PASSes on the now-valid
      main-tree state instead of FAILing on a removed worktree.
    """

    issue: int
    hf_data_paths: tuple[str, ...] = ()
    hf_model_paths: tuple[str, ...] = ()
    hf_data_repo: str = DEFAULT_HF_DATA_REPO
    hf_model_repo: str = DEFAULT_HF_MODEL_REPO
    wandb_run_path: str | None = None
    git_paths: tuple[str, ...] = ()
    sentinel_path: str | None = None
    git_repo_root: str | None = None


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
    path_in_repo: str | None = None,
) -> list[str]:
    """Default HF Hub file lister.

    Uses the Python Hub API — the API the upload-policy rule pins as
    authoritative (``hf`` CLI has no ``api`` subcommand, so it silently
    returns 0 files; never use it here). Raises on transport / auth
    failure — the verifier turns that into a FAIL with reason rather than
    silently passing.

    ``path_in_repo`` scopes the walk SERVER-side (#920/#988); an absent
    path returns [] (mapped inside ``list_hf_files_under_path``). ``None``
    keeps the full listing for seam-contract compatibility — a future
    caller passing ``None`` against the ~1M-file DATA repo re-enters the
    #920 hang class, so data-repo callers must scope.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        list_hf_files_under_path,
        list_repo_files_complete,
    )

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if path_in_repo is not None:
        return list_hf_files_under_path(
            api, repo_id, path_in_repo, repo_type=repo_type, revision=revision
        )
    return list_repo_files_complete(api, repo_id, repo_type=repo_type, revision=revision)


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

    Returns the tracked FILE paths that ``git ls-files`` reports for the
    ``rel_paths`` pathspecs — for a directory pathspec this is the files
    UNDER it, not the directory string itself (``_declared_path_tracked``
    does the prefix matching). Runs ONE ``git ls-files -- <p1> <p2> ...``
    call rather than N — git resolves the union internally.
    Raises ``CalledProcessError`` on a non-zero git exit (e.g. not a
    repo); the verifier turns that into a FAIL.
    """
    rel_list = list(rel_paths)
    if not rel_list:
        return set()
    argv = ["git", "-C", str(repo_root), "ls-files", "--", *rel_list]
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=30,
    )
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


def _default_glob_sentinels(declared: str, issue: int) -> list[str]:
    """Default live-sibling sentinel probe (real FS glob).

    Given the DECLARED sentinel path
    ``.../eval_results/issue_<N>/<attempt>/.completion-sentinel.json``,
    enumerate the sibling attempt dirs' sentinels
    ``.../eval_results/issue_<N>/*/.completion-sentinel.json`` so
    :func:`_resolve_live_sentinel` can prefer a live current-attempt
    sentinel when the declared (stale-attempt) one is missing (#685
    secondary cause: a stale attempt path baked into the handle from a
    prior failed launch attempt).

    The grandparent of the declared path is the ``issue_<N>/`` dir; we
    glob ``<grandparent>/*/<SENTINEL_FILENAME>``. Returns absolute path
    strings (possibly empty). Pure FS read; raises nothing on a missing
    dir (``Path.glob`` yields nothing). The ``issue`` argument is part of
    the injectable seam's signature (a SSH-backed variant keys on it) but
    is unused by the FS default — the declared path already carries the
    issue dir.
    """
    del issue  # encoded in the declared path's grandparent; unused here.
    declared_p = Path(declared)
    issue_dir = declared_p.parent.parent
    return [str(s) for s in sorted(issue_dir.glob(f"*/{SENTINEL_FILENAME}"))]


@dataclass(frozen=True)
class VerifierIO:
    """Bundle of injectable I/O callables.

    Tests construct a :class:`VerifierIO` with mocks for each callable so
    the verifier runs with no real HF / WandB / git / FS side effects.
    The default-constructed instance wires every callable to its real
    implementation above.

    Fields:

    * ``list_hf_repo_files(repo_id, *, repo_type, revision=None,
      path_in_repo=None) -> list[str]`` — must enumerate the repo's files
      at the given revision; ``path_in_repo`` scopes the walk server-side
      (#920/#988 — production callers scope; ``None`` full-lists, kept for
      seam compatibility). ``None`` revision = repo default. Fakes may
      ignore ``path_in_repo`` and return the full list — the caller's
      client-side ``_path_matches`` filter preserves the semantics.
    * ``wandb_run_exists(run_path) -> bool`` — must return True iff the
      WandB run resolves. Transport errors propagate.
    * ``git_tracked(repo_root, rel_paths) -> set[str]`` — must return the
      tracked FILE paths matched by the ``rel_paths`` pathspecs (for a
      directory pathspec: the tracked files under it), relative to
      ``repo_root`` — mirroring real ``git ls-files`` output.
    * ``read_sentinel(path) -> str | None`` — must return the sentinel
      file's UTF-8 content, or ``None`` when the file does not exist.
    * ``glob_sentinels(declared, issue) -> list[str]`` — must enumerate
      the sibling attempt-dir sentinels under the declared path's
      ``issue_<N>/`` grandparent (``*/.completion-sentinel.json``) so the
      stale-attempt resolver can prefer a live current-attempt sentinel
      when the declared one is missing (#685). Default = real FS glob;
      tests inject a deterministic list so they stay FS-free.
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
    glob_sentinels: Callable[[str, int], list[str]] | None = None
    repo_root: Path | None = None

    def _list_hf(self) -> Callable[..., list[str]]:
        return self.list_hf_repo_files or _default_list_hf_repo_files

    def _wandb(self) -> Callable[[str], bool]:
        return self.wandb_run_exists or _default_wandb_run_exists

    def _git(self) -> Callable[[Path, Iterable[str]], set[str]]:
        return self.git_tracked or _default_git_tracked

    def _sentinel(self) -> Callable[[str], str | None]:
        return self.read_sentinel or _default_read_sentinel

    def _glob_sentinels(self) -> Callable[[str, int], list[str]]:
        return self.glob_sentinels or _default_glob_sentinels


def _pyproject_walk_root() -> Path:
    """Repo root by the ``pyproject.toml`` ``__file__``-walk, cwd fallback."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _resolve_repo_root(io: VerifierIO, *, git_repo_root: str | None = None) -> Path:
    """Resolve the git-check repo root.

    Precedence:

    1. An explicit ``io.repo_root`` override (tests inject this) — always
       wins, even over a baked ``git_repo_root`` (tests pin the tree they
       built).
    2. The launch-baked ``git_repo_root`` (the per-issue worktree the run
       committed its artifacts to, #685) — used when ``io.repo_root`` is
       unset (production: ``verify_artifacts`` is called with the
       real-wires ``VerifierIO()`` whose ``repo_root`` is ``None``) — but
       WITH the absent-dir fallback below.
    3. The pyproject ``__file__``-walk (the established production
       default — the main checkout on ``main``).
    4. ``cwd`` (last-ditch, inside the walk).

    **Absent-baked-worktree fallback (#705 / #685, post-Step-10d
    auto-merge case).** When the baked ``git_repo_root`` directory no
    longer EXISTS — the worktree was merged into ``main`` and removed
    before finalize, so the committed eval/figure files are now on
    ``main`` — fall back to the pyproject-walked main root with a LOUD
    ``logger.warning``. Without this, a finalize after the auto-merge
    would FAIL ``_check_git`` on a removed worktree even though the
    artifacts are perfectly tracked on ``main``. The fallback is scoped
    to the baked path ONLY — a test-injected ``io.repo_root`` (precedence
    1) is never second-guessed.
    """
    if io.repo_root is not None:
        return io.repo_root
    if git_repo_root is not None:
        baked = Path(git_repo_root)
        if baked.exists():
            return baked
        fallback = _pyproject_walk_root()
        logger.warning(
            "verify_artifacts: baked git_repo_root %s no longer exists "
            "(worktree merged + removed post-Step-10d?); falling back to the "
            "pyproject-walked main root %s for the git check",
            baked,
            fallback,
        )
        return fallback
    return _pyproject_walk_root()


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
# Shared declaration builder (SLURM + RunPod launch paths)
# ---------------------------------------------------------------------------


def build_expected_artifacts_declaration(
    *,
    issue: int,
    sentinel_path: str,
    custom_workload: bool,
    attempt_id: str | None = None,
    hf_data_repo: str = DEFAULT_HF_DATA_REPO,
    hf_model_repo: str = DEFAULT_HF_MODEL_REPO,
    wandb_run_path: str | None = None,
    extra_hf_data_paths: Sequence[str] = (),
    extra_hf_model_paths: Sequence[str] = (),
    extra_git_paths: Sequence[str] = (),
    git_repo_root: str | None = None,
    skip_default_git_paths: bool = False,
) -> dict[str, Any]:
    """Backend-agnostic :data:`EXPECTED_ARTIFACTS_HANDLE_KEY` payload.

    Mirrors ``gcp.py:expected_artifacts_declaration`` including the #601
    custom-workload carve-out: custom workloads own their HF prefix, so
    we declare NO launch-time guess for them (a guessed prefix turned
    the gate into a false-negative teardown block on a perfectly
    uploaded run). Hydra-lane launches declare
    ``issue<N>_<attempt>/raw_completions/`` (requires ``attempt_id``).
    Callers that DO know the workload's real prefix declare it via
    ``extra_hf_data_paths``.

    Consumed by the SLURM + RunPod launch paths (#598); the GCP lane
    keeps its own builder (scoped out of #598 — migrating it here is a
    named follow-up, not this diff).

    **Plan-referenced analysis tensors are NOT declared by default** — a
    run whose plan references intermediate analysis tensors as downstream
    inputs (per-cell shift tensors, cached activations, ``clouds.npz`` —
    Upload Policy ``issue<N>_<slug>/analysis_tensors/`` row, #521) MUST
    declare that prefix via ``extra_hf_data_paths`` so the mechanical gate
    catches a dropped tensor on its own. These artifacts are ``.npz`` /
    ``.pt`` / ``.npy`` binaries that ``.gitignore`` correctly excludes, so
    a directory-level ``git add`` of ``eval_results/issue_<N>/`` silently
    drops them — git is the WRONG destination; the HF data repo is the
    permanent home (the dispatcher uploads them, the verifier confirms
    them). The verifier's ``git_paths`` check passes on the tracked eval
    JSONs regardless, so an undeclared analysis tensor would otherwise
    slip the mechanical gate entirely and rest only on the agent-level
    upload-verifier (incident #545: a 110 MB ``clouds.npz`` was dropped
    from git by the ``*.npz`` rule and never reached HF — caught by the
    upload-verifier agent, not this gate, because the launch path never
    declared the ``analysis_tensors/`` prefix).

    **Phase-scoped launches (``skip_default_git_paths=True``, #604/#661).**
    A PHASE-SCOPED launch whose deliverable is HF-only this phase (e.g. a
    P3 extraction whose output lands on the HF data repo under
    ``issue<N>_<slug>/analysis_tensors/`` via ``extra_hf_data_paths``,
    while the off-pod P5 analysis phase produces the git
    ``eval_results/`` + ``figures/`` NEXT) sets ``skip_default_git_paths``
    so the declaration omits the auto full-task git paths it was never
    going to produce. ``git_paths`` then carries ONLY the caller's
    ``extra_git_paths`` (usually empty → the git check SKIPs, which is NOT
    a FAIL — SKIP never contributes to the verdict). The HF + completion
    sentinel checks STILL run, so the gate is not relaxed — it just stops
    demanding artifacts this phase never promised (#661 root cause: the
    auto full-task git paths produced a structural FAIL on a perfectly
    HF-uploaded phase-scoped run).

    **Default git paths are split by ``custom_workload`` (#790).** When
    ``skip_default_git_paths`` is ``False`` the default ``git_paths`` depend
    on the workload kind, because the two kinds produce different git
    artifacts DURING the run:

    * ``custom_workload=True`` (``--workload-cmd`` drivers): ``git_paths``
      = ``[eval_results/issue_<N>/, *extra_git_paths]``. Many dispatch-script
      drivers commit their eval JSONs during the run, so a missing
      ``eval_results/`` there is a genuine FAIL and the check is kept; but
      ``figures/`` is dropped because the analyzer generates + commits
      figures POST-gate (analyzer.md Step 3) on every lane.
    * ``custom_workload=False`` (pure hydra ``scripts/train.py``):
      ``git_paths`` = ``[*extra_git_paths]`` — BOTH defaults dropped. ``train.py``
      runs ``run_single(..., skip_eval=True)`` and ``orchestrate/runner.py``
      gates all eval production on ``not skip_eval``, so a hydra run writes
      neither ``eval_results/issue_<N>/`` nor ``figures/issue_<N>/`` during
      the run; declaring either was a guaranteed false-FAIL (#790).

    The GCP-lane ``fetch_results`` best-effort mirror still pulls both
    ``eval_results/`` + ``figures/`` for analyzer-local convenience; they
    are just no longer gate artifacts on the pure-hydra branch. The
    completion sentinel + the HF-data checks keep gating teardown.

    **Unmerged-worktree launches (``git_repo_root``, #685).** When the
    run's committed eval/figure artifacts live on an unmerged
    ``issue-<N>`` branch checked out in the per-issue worktree (auto-merge
    to ``main`` is at /issue Step 10d, AFTER finalize), the launch path
    passes the worktree's absolute path as ``git_repo_root`` so
    ``_check_git`` resolves ``git ls-files`` THERE (where the files ARE
    tracked) instead of the MAIN checkout on ``main`` (where they are
    not yet). ``None`` (default) keeps the established pyproject-walk repo
    root. The field is omitted from the returned dict when ``None`` so
    pre-#705 in-flight sidecars + the established gate behave identically.

    Returns a JSON-safe dict (lists, not tuples) so it round-trips via
    ``serialize_handle`` / :func:`expected_artifacts_from_handle`.
    """
    if custom_workload:
        base_hf_data: tuple[str, ...] = ()
    else:
        if not attempt_id:
            raise ValueError(
                "build_expected_artifacts_declaration: hydra-lane declaration "
                "requires attempt_id (it names the HF raw-completions prefix)"
            )
        base_hf_data = (f"issue{issue}_{attempt_id}/raw_completions/",)
    if skip_default_git_paths:
        # Phase-scoped: omit the auto full-task git paths; honor only the
        # caller's explicit extra_git_paths (so a phase that DOES commit a
        # scoped git file can still declare it).
        git_paths = list(extra_git_paths)
    elif custom_workload:
        # ``--workload-cmd`` drivers may commit ``eval_results/`` during the
        # run (many do), but NEVER ``figures/`` — the analyzer generates +
        # commits figures POST-gate (analyzer.md Step 3), AFTER
        # ``confirm_artifacts`` runs, on every lane. Keep the real
        # ``eval_results/`` check; drop the never-produced ``figures/``
        # false-FAIL (#790).
        git_paths = [
            f"eval_results/issue_{issue}/",
            *extra_git_paths,
        ]
    else:
        # Pure hydra (``scripts/train.py`` runs ``run_single(..., skip_eval=True)``,
        # and ``orchestrate/runner.py`` gates ALL eval production on
        # ``not skip_eval``), so the run writes NEITHER ``eval_results/issue_<N>/``
        # NOR ``figures/issue_<N>/``. Declaring either is a load-bearing
        # false-FAIL (#790). The GCP-lane ``fetch_results`` best-effort mirror
        # still pulls both for analyzer-local convenience; they are just no
        # longer gate artifacts.
        git_paths = list(extra_git_paths)
    decl: dict[str, Any] = {
        "issue": int(issue),
        "hf_data_repo": hf_data_repo,
        "hf_model_repo": hf_model_repo,
        "hf_data_paths": list(base_hf_data) + list(extra_hf_data_paths),
        "hf_model_paths": list(extra_hf_model_paths),
        "wandb_run_path": wandb_run_path,
        "git_paths": git_paths,
        "sentinel_path": sentinel_path,
    }
    if git_repo_root:
        # Omitted when None so a pre-#705 in-flight sidecar round-trips
        # byte-identically (back-compat constraint).
        decl["git_repo_root"] = git_repo_root
    return decl


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

    Scoped listings (#920/#988): ONE server-side scoped walk PER declared
    path (typically 1-4 paths, ~1-2 s each) replaces the single full-repo
    listing (>600 s wedge on the ~1M-file data repo). ``_path_matches``
    stays as the client-side verdict — a no-op against real scoped results,
    but it keeps full-list test fakes (whose seam ignores ``path_in_repo``)
    matching the same semantics.
    """
    if not paths:
        return {"status": "SKIP", "detail": "no paths declared"}
    # Hoisted empty-path guard: under the old full-listing shape an EMPTY
    # declared path raised ValueError from _path_matches OUTSIDE the try (a
    # config error, not a FAIL verdict); keep that contract rather than let
    # the scoped call convert it into a caught FAIL.
    for p in paths:
        if not p:
            raise ValueError("_path_matches: empty path is not a valid declaration")
    missing: list[str] = []
    try:
        for p in paths:
            files = io._list_hf()(repo_id, repo_type=repo_type, path_in_repo=p.rstrip("/"))
            if not _path_matches(files, p):
                missing.append(p)
    except Exception as exc:
        # Fail-loud per CLAUDE.md "no silent True on transport error".
        # We catch + surface as FAIL (not re-raise) so the verdict's
        # `checks` dict carries the reason; otherwise the orchestrator
        # would see an uncaught exception with no structured signal.
        return {
            "status": "FAIL",
            "detail": f"HF list_repo_files({repo_id!r}, {repo_type!r}) raised: {exc}",
        }
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


def _declared_path_tracked(path: str, tracked: set[str]) -> bool:
    """True when declared ``path`` is covered by the tracked-entry set.

    ``tracked`` holds FILE paths (``git ls-files`` output — git never
    lists directories). A file declaration matches literally; a
    directory declaration (with or without trailing slash) matches when
    >=1 tracked entry equals its stripped form or sits under it
    (``startswith(path.rstrip('/') + '/')``). Without the prefix rule a
    directory declaration like ``eval_results/issue_588/`` could never
    equal a file path, so EVERY real-IO run failed the git check despite
    tracked files existing under the prefix (issue #588 round-2 live
    finding).
    """
    if path in tracked:
        return True
    stripped = path.rstrip("/")
    if stripped in tracked:
        return True
    prefix = stripped + "/"
    return any(entry.startswith(prefix) for entry in tracked)


def _check_git(
    *,
    paths: tuple[str, ...],
    io: VerifierIO,
    git_repo_root: str | None = None,
) -> dict[str, Any]:
    """Confirm every declared path is tracked by git AND present on disk.

    SKIP if no paths were declared. Both conditions must hold: a path
    tracked but deleted from the working tree fails the second check; an
    untracked file in the tree fails the first. Tracked-ness is decided
    by :func:`_declared_path_tracked` — exact match for file
    declarations, prefix match against the tracked-file listing for
    directory declarations.

    ``git_repo_root`` (the launch-baked
    :attr:`ExpectedArtifacts.git_repo_root`) is forwarded to
    :func:`_resolve_repo_root` so the per-issue-worktree resolution +
    the absent-baked-worktree fallback (#685 / #705) apply. ``None`` =
    the established pyproject-walk root.
    """
    if not paths:
        return {"status": "SKIP", "detail": "no git paths declared"}
    repo_root = _resolve_repo_root(io, git_repo_root=git_repo_root)
    try:
        tracked = io._git()(repo_root, paths)
    except subprocess.CalledProcessError as exc:
        return {
            "status": "FAIL",
            "detail": f"git ls-files failed: exit={exc.returncode} stderr={exc.stderr!r}",
        }
    except Exception as exc:
        return {"status": "FAIL", "detail": f"git ls-files raised: {exc}"}
    missing_tracked = [p for p in paths if not _declared_path_tracked(p, tracked)]
    missing_on_disk = [p for p in paths if not (repo_root / p).exists()]
    problems: list[str] = []
    if missing_tracked:
        problems.append("not tracked by git: " + "; ".join(missing_tracked))
    if missing_on_disk:
        problems.append("not on disk: " + "; ".join(missing_on_disk))
    if problems:
        return {"status": "FAIL", "detail": " | ".join(problems)}
    return {"status": "PASS", "detail": f"all {len(paths)} path(s) tracked + on disk"}


def _resolve_live_sentinel(declared: str, issue: int, io: VerifierIO) -> tuple[str, str | None]:
    """Resolve which sentinel path ``_check_sentinel`` should READ.

    The DECLARED path is preferred whenever it resolves (the common
    case). When it is MISSING — the #685 secondary cause: a stale
    attempt-N sentinel path baked into the handle from a prior failed
    launch attempt, while the CURRENT attempt wrote its sentinel under a
    different attempt dir — probe the sibling attempt dirs
    (``eval_results/issue_<N>/*/.completion-sentinel.json``) for LIVE
    sentinels and:

    * **exactly ONE live sibling** → prefer it, returning a LOUD note
      naming BOTH the missing declared path and the resolved sibling (the
      caller logs it at WARNING). The chosen sibling still goes through
      the UNCHANGED ``phase == "done"`` + issue-match content checks in
      :func:`_check_sentinel`, so this is RESOLUTION ONLY — a
      genuinely-incomplete run (wrong phase / wrong issue) still FAILs.
    * **zero live siblings** → return the declared path unchanged; the
      caller's content read then FAILs loud with the real "missing"
      reason.
    * **two or more live siblings (KNOWN LIMITATION)** → do NOT guess.
      Return the declared (missing) path so the content read FAILs loud,
      with a note naming all candidates. This is the conservative v1
      choice: it trades precision (a content/issue-match check COULD pick
      the right sibling) for safety (never PASS a wrong-attempt run on an
      ambiguous probe). NOT widened to "pick the one whose content
      matches expected.issue" — ambiguity is surfaced as a FAIL, never
      silently resolved.

    The sibling enumeration is injected via ``io.glob_sentinels`` (real
    FS glob by default) so tests stay FS-free; liveness is the SAME
    ``io.read_sentinel`` seam the content check uses.

    Returns ``(path_to_read, note_or_None)``.
    """
    try:
        declared_present = io._sentinel()(declared) is not None
    except Exception:
        # A transport RAISE (e.g. the RunPod SSH reader on rc=255) is NOT a
        # "missing" signal — return the declared path UNCHANGED so the
        # caller's own read re-raises inside its try and produces the
        # fail-loud "sentinel read raised" reason. Do NOT probe siblings
        # off an unknowable declared state.
        return declared, None
    if declared_present:
        return declared, None
    try:
        siblings = io._glob_sentinels()(declared, issue)
    except Exception as exc:
        # A glob/listing transport failure is NOT a silent pass: fall back
        # to the declared (missing) path so the content read FAILs loud,
        # carrying the probe failure in the note.
        return declared, f"live-sibling probe raised: {exc}"
    live = [s for s in siblings if s != declared and io._sentinel()(s) is not None]
    if len(live) == 1:
        return live[0], (
            f"declared sentinel {declared} missing; resolved to the single live "
            f"sibling {live[0]} (stale baked attempt path, #685)"
        )
    if len(live) >= 2:
        return declared, (
            f"declared sentinel {declared} missing AND {len(live)} live sibling "
            f"sentinels found ({'; '.join(live)}) — ambiguous, refusing to guess; "
            "FAILing on the declared (missing) path (known v1 limitation)"
        )
    return declared, None


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

    Before reading, :func:`_resolve_live_sentinel` resolves which path to
    read: the declared one when it exists, else (the #685 stale-baked-
    attempt case) the single live sibling attempt-dir sentinel if exactly
    one exists. The resolution is RESOLUTION ONLY — the phase + issue
    content checks below are UNCHANGED, so a genuinely-incomplete run
    still FAILs.
    """
    if not sentinel_path:
        return {"status": "SKIP", "detail": "no sentinel_path declared"}
    resolved_path, resolve_note = _resolve_live_sentinel(sentinel_path, issue, io)
    if resolve_note:
        logger.warning("verify_artifacts(issue=%d) sentinel resolve: %s", issue, resolve_note)
    sentinel_path = resolved_path
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
        git_repo_root=raw.get("git_repo_root"),
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
        CHECK_GIT: _check_git(
            paths=expected.git_paths,
            io=io,
            git_repo_root=expected.git_repo_root,
        ),
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
    "build_expected_artifacts_declaration",
    "confirm_artifacts_from_handle",
    "expected_artifacts_from_handle",
    "verify_artifacts",
    "write_completion_sentinel",
]
