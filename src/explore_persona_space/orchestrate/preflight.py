"""Pre-flight checks for experiments. Run before starting ANY training or eval.

Usage:
    # As a module
    from explore_persona_space.orchestrate.preflight import require_preflight
    require_preflight()

    # From CLI
    uv run python -m explore_persona_space.orchestrate.preflight

Three-way environment branch
----------------------------

Preflight runs on three different surfaces; the checks adapt:

* **Cluster** (``SLURM_JOB_ID`` set): disk probe targets ``$SLURM_TMPDIR``
  / ``$SCRATCH`` (not ``/workspace``) and the RunPod MooseFS 130GB
  quota cap is bypassed (``per_pod_quota_gb=None``). The ``git fetch``
  round trip in :func:`check_git_status` and the installed-vs-uv.lock
  :func:`check_env_sync` are SKIPPED: the cluster is rsync-primary
  with no remote git auth, and the venv build happens inside the
  sbatch (so a pre-rsync mismatch is expected, not an error).
  ``HF_HOME`` defaults to ``$SCRATCH/.cache/huggingface``. The Hub /
  WandB reachability check still runs (compute nodes may need a proxy).
* **RunPod** (``/workspace`` exists, no SLURM): unchanged from the
  pre-three-way behavior.
* **Local VM**: unchanged.

The discriminator lives in :mod:`explore_persona_space.orchestrate.env`;
this module imports the helpers so the branch logic stays in ONE place.
"""

import contextlib
import errno
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Three-way environment helpers (see env.py module docstring). Imported
# at top level so the cluster branch threads cleanly through every check
# without re-importing per call.
from explore_persona_space.orchestrate.env import (
    _hf_home_default,
    is_cluster_env,
    is_runpod_env,
)

logger = logging.getLogger(__name__)

# RunPod MooseFS gives each pod a per-pod writable-bytes quota (~130GB) that is
# SEPARATE from, and far below, the share-level free space ``shutil.disk_usage``
# reports (terabytes). A small canary probe (``_probe_writable_bytes``) detects
# the quota only once it is ALREADY exhausted; to catch an over-quota footprint
# BEFORE launch we cap the usable headroom at this constant. Override per-pod via
# the ``quota_gb`` parameter / ``--per-pod-quota-gb`` flag when a pod was
# provisioned with an explicit, larger storage spec. See CLAUDE.md "RunPod
# MooseFS per-pod disk quota" gotcha + memory note feedback_runpod_moosefs_quota.
RUNPOD_PER_POD_QUOTA_GB = 130.0


@dataclass
class PreflightReport:
    """Result of pre-flight checks."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gpu_info: list[dict] = field(default_factory=list)
    disk_free_gb: float = 0.0
    disk_probed_headroom_gb: float = 0.0
    # Human-readable provenance of ``disk_probed_headroom_gb`` so the budget
    # check + summary never mislabel a share-level (quota-blind) number as
    # "probed". Set by ``check_disk_space``.
    disk_headroom_basis: str = "share-level free"
    git_status: str = ""
    env_synced: bool = True
    # Account-level HF public-storage headroom (#564). None = unknown /
    # not checked; basis names the signal ("live-api" / "cache (...)" /
    # "disabled" / "suspect (...)" / "unknown (...)"). Set by
    # ``check_hf_storage``.
    hf_storage_used_tb: float | None = None
    hf_storage_ceiling_tb: float | None = None
    hf_storage_basis: str = ""
    # Zero-byte LFS batch-negotiation billing/quota probe (#1654). Empty =
    # not checked; verdict in {"ok","billing-blocked","storage-blocked",
    # "unknown","disabled"}. Set by ``check_hf_lfs_write_gate``.
    hf_lfs_write_verdict: str = ""
    hf_lfs_write_detail: str = ""
    hf_lfs_write_probe_gb: float = 0.0

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        """Formatted summary string."""
        lines = []
        status = "PASS" if self.ok else "FAIL"
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  Pre-flight Check: {status}")
        lines.append(f"{'=' * 60}")

        if self.errors:
            lines.append("\n  ERRORS (must fix before running):")
            for e in self.errors:
                lines.append(f"    ✗ {e}")

        if self.warnings:
            lines.append("\n  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")

        if self.gpu_info:
            lines.append("\n  GPUs:")
            for g in self.gpu_info:
                used = g.get("memory_used_mb", 0)
                total = g.get("memory_total_mb", 0)
                free = g.get("memory_free_mb", 0)
                procs = g.get("processes", 0)
                status_icon = "✓" if procs == 0 and used < 1000 else "⚠"
                lines.append(
                    f"    {status_icon} GPU {g['id']}: "
                    f"{free:,}MB free / {total:,}MB total "
                    f"({procs} processes)"
                )

        lines.append(
            f"\n  Disk: {self.disk_free_gb:.1f} GB free "
            f"(usable headroom {self.disk_probed_headroom_gb:.1f} GB, "
            f"basis: {self.disk_headroom_basis})"
        )
        if self.hf_storage_used_tb is not None and self.hf_storage_ceiling_tb is not None:
            lines.append(
                f"  HF storage: {self.hf_storage_used_tb:.2f} TB / "
                f"ceiling {self.hf_storage_ceiling_tb:.1f} TB ({self.hf_storage_basis})"
            )
        else:
            lines.append(f"  HF storage: unknown ({self.hf_storage_basis or 'not checked'})")
        if self.hf_lfs_write_verdict:
            if self.hf_lfs_write_probe_gb > 0:
                lines.append(
                    f"  HF LFS write gate: {self.hf_lfs_write_verdict} "
                    f"({self.hf_lfs_write_probe_gb:.0f} GB declared probe)"
                )
            else:
                lines.append(f"  HF LFS write gate: {self.hf_lfs_write_verdict}")
        lines.append(f"  Git: {self.git_status}")
        lines.append(f"  Env synced: {'yes' if self.env_synced else 'NO'}")
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a command with timeout. Returns (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", f"command not found: {cmd[0]}"
    except Exception as e:
        return -1, "", str(e)


def _behind_count(project_root: Path, ref: str) -> int | None:
    """Commits HEAD is behind ``ref`` (via ``git rev-list --count HEAD..<ref>``).

    Returns None when the count cannot be determined (missing ref → rc=128,
    git error, non-integer output). Callers treat None as "unknown", never 0.
    """
    rc, out, _ = _run(["git", "-C", str(project_root), "rev-list", "--count", f"HEAD..{ref}"])
    if rc != 0:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def _ahead_count(project_root: Path, ref: str) -> int | None:
    """Commits HEAD is ahead of ``ref`` (via ``git rev-list --count <ref>..HEAD``).

    Returns None when the count cannot be determined. Callers treat None as
    "unknown" (best-effort signal — an unknown ahead-count never errors).
    """
    rc, out, _ = _run(["git", "-C", str(project_root), "rev-list", "--count", f"{ref}..HEAD"])
    if rc != 0:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    p = Path(__file__).resolve()
    for parent in [p, *list(p.parents)]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# Hard free-space FLOOR (GB) for the VM ROOT disk only (``check_path == "/"``).
# A fresh launch below this floor FAILs fast rather than starting on a disk that
# is already near the silent-Bash-failure regime (task #552 / #679). RunPod
# (``/workspace``) is EXEMPT — there the MooseFS per-pod EDQUOT probe is the
# binding signal, not free GB (a TB-scale share would never trip a free-GB
# floor). Env-overridable; an explicit operator override degrades the FAIL to a
# logged WARN. Default 40 GB.
VM_ROOT_DISK_FLOOR_GB_DEFAULT = 40.0


def _vm_root_disk_floor_gb() -> float:
    """VM-root free-space floor in GB (env ``EPM_PREFLIGHT_DISK_FLOOR_GB``;
    garbled / non-positive -> default). Never raises."""
    raw = os.environ.get("EPM_PREFLIGHT_DISK_FLOOR_GB", "")
    try:
        val = float(raw)
    except ValueError:
        return VM_ROOT_DISK_FLOOR_GB_DEFAULT
    return val if val > 0 else VM_ROOT_DISK_FLOOR_GB_DEFAULT


def _vm_root_disk_floor_override() -> bool:
    """True when the operator has explicitly opted to launch below the VM-root
    floor (env ``EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE=1``). Degrades the floor FAIL
    to a logged WARN — the escape hatch for a deliberate low-disk launch."""
    return os.environ.get("EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE", "").strip() in {"1", "true", "yes"}


def _check_vm_root_floor(report: PreflightReport, check_path: str, min_free_gb: float) -> None:
    """Apply the hard VM-root free-space floor (``check_path == "/"`` only).

    Fires ONLY on the VM root — RunPod ``/workspace`` and cluster scratch are
    exempt (their binding signal is the quota probe, not free GB). Below the
    floor: an ERROR (``report.ok`` -> False) unless the override env degrades it
    to a WARN. Deduped against the existing ``min_free_gb`` ERROR: when
    ``min_free_gb`` is the higher bar and already errored, the floor adds no
    second error (the run is already failing) — it only adds value when the
    floor is the binding gate the legacy ``min_free_gb`` path did not catch."""
    if check_path != "/":
        return
    floor = _vm_root_disk_floor_gb()
    free = report.disk_free_gb
    if free is None or free >= floor:
        return
    # Below the floor.
    if _vm_root_disk_floor_override():
        report.add_warning(
            f"VM-root disk floor OVERRIDDEN: only {free:.1f}GB free on / "
            f"(floor {floor:.0f}GB, EPM_PREFLIGHT_DISK_FLOOR_GB); launching anyway "
            f"because EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE is set. Disk pressure risks "
            f"silently-failing Bash spawns — reclaim with vm_disk_guard.py --apply."
        )
        return
    # Avoid a duplicate error when the legacy min_free_gb gate already errored
    # for the SAME under-floor free space (min_free_gb >= floor means its error
    # already fired at >= this threshold).
    if min_free_gb >= floor and free < min_free_gb:
        return
    report.add_error(
        f"VM-root disk below floor: only {free:.1f}GB free on / "
        f"(floor {floor:.0f}GB, EPM_PREFLIGHT_DISK_FLOOR_GB). A launch this low "
        f"risks silently-failing Bash spawns (task #552). Reclaim with "
        f"`uv run python scripts/vm_disk_guard.py --apply`, or set "
        f"EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE=1 to launch anyway."
    )


def _disk_check_path() -> str:
    """Where to run the disk-space probe — three-way branch.

    * **Cluster:** prefer ``$SLURM_TMPDIR`` (node-local fast scratch
      where data + model are staged) when it exists, else ``$SCRATCH``
      (per-user persistent scratch where the venv + checkpoints live).
      Fall back to ``/`` if neither env var is set — defensive.
    * **RunPod:** ``/workspace`` (the MooseFS-backed pod volume).
    * **Local VM:** ``/`` (the root filesystem).

    The picked path is what ``check_disk_space`` probes for free-space
    + the canary EDQUOT probe. On the cluster the MooseFS 130 GB cap is
    explicitly bypassed by the caller (``per_pod_quota_gb=None``); on
    RunPod the cap is enforced.
    """
    if is_cluster_env():
        for env_var in ("SLURM_TMPDIR", "SCRATCH"):
            candidate = os.environ.get(env_var)
            if candidate and Path(candidate).exists():
                return candidate
        return "/"
    if is_runpod_env():
        return "/workspace"
    return "/"


def check_git_status(report: PreflightReport, project_root: Path):
    """Check git working tree is clean and up to date — branch-aware (#554).

    The behind-remote comparison is keyed on the current branch: ``main``
    compares against ``origin/main`` (ERROR when behind, message unchanged);
    a feature branch compares against its OWN ``origin/<branch>`` ref (the
    run-of-record on the canonical ``/issue`` pod checkout), with divergence
    from ``origin/main`` demoted to an informational WARNING; detached HEAD
    (pinned-SHA checkout) only warns.

    Cluster branch: the ``git fetch origin`` round trip is SKIPPED because
    the cluster compute node has no remote git auth — code reaches the
    cluster via rsync, not git pull. The local ``git status --porcelain``
    check still runs (it's local-only and cheap) so an accidental
    uncommitted change is still surfaced; we just don't try to compare
    against origin/main. The ``git_status`` field is decorated with
    ``" (cluster — skipped fetch)"`` so the summary makes the skip
    explicit rather than misleadingly reading "clean / up to date".
    """
    # Check for uncommitted changes
    rc, out, err = _run(["git", "-C", str(project_root), "status", "--porcelain"])
    if rc != 0:
        report.add_warning(f"git status failed: {err}")
        report.git_status = "unknown"
        return

    if out:
        changed = len(out.strip().splitlines())
        report.add_warning(f"{changed} uncommitted change(s) — consider committing first")
        report.git_status = f"{changed} uncommitted changes"
    else:
        report.git_status = "clean"

    if is_cluster_env():
        # rsync-primary on the cluster; no remote git auth on compute
        # nodes. Mark explicitly so the summary doesn't read "clean,
        # up-to-date" when we didn't check up-to-date-ness.
        report.git_status += " (cluster — skipped fetch)"
        return

    # Behind-remote check — branch-aware (#554). On the canonical /issue pod
    # checkout HEAD is on `issue-<N>`; the branch's own pushed origin ref IS
    # the run-of-record, and divergence from origin/main is expected (#383,
    # #550). ERROR is reserved for "behind the ref this checkout tracks".
    # The fetch rc is CAPTURED: the behind-own guarantee below is only as
    # fresh as this fetch, so a failed fetch on a feature branch is an ERROR,
    # never a silent stale-ref false PASS.
    fetch_rc, _, fetch_err = _run(
        ["git", "-C", str(project_root), "fetch", "--quiet", "origin"], timeout=15
    )
    fetch_failed = fetch_rc != 0

    rc, branch, err = _run(["git", "-C", str(project_root), "rev-parse", "--abbrev-ref", "HEAD"])
    if rc != 0:
        report.add_warning(f"could not determine current branch: {err}")
        report.git_status += ", branch unknown"
        return
    branch = branch.strip()

    if branch == "main":
        _check_main_branch_behind(report, project_root, fetch_failed, fetch_err)
        return

    if branch == "HEAD":
        _check_detached_head_behind(report, project_root, fetch_failed, fetch_err)
        return

    _check_feature_branch_behind(report, project_root, branch, fetch_failed, fetch_err)


def _check_main_branch_behind(
    report: PreflightReport, project_root: Path, fetch_failed: bool, fetch_err: str
):
    """Behind-remote check for ``main``: ERROR when behind origin/main.

    The ERROR message is byte-identical to the pre-#554 one — agent specs
    tolerance-match it verbatim.
    """
    if fetch_failed:
        # Main's gate decision stays as before: warn, then compute against
        # last-fetched refs exactly as the old code did. Staleness here is
        # fail-soft toward PASS — a timed-out fetch reads last-fetched
        # refs, which can only UNDER-count how far behind main is.
        report.add_warning(
            f"git fetch origin failed ({fetch_err}); behind-origin/main "
            f"computed against last-fetched refs."
        )
    behind_main = _behind_count(project_root, "origin/main")
    if behind_main:  # None (count unknown) keeps the prior silent-skip behavior
        report.add_error(
            f"Local is {behind_main} commit(s) behind origin/main. Run: git pull origin main"
        )
        report.git_status += f", {behind_main} behind remote"


def _check_detached_head_behind(
    report: PreflightReport, project_root: Path, fetch_failed: bool, fetch_err: str
):
    """Detached HEAD (pinned-SHA checkout): no own ref to be behind — warn only."""
    if fetch_failed:
        report.add_warning(
            f"git fetch origin failed ({fetch_err}); ref comparisons use last-fetched refs."
        )
    behind_main = _behind_count(project_root, "origin/main")
    if behind_main:
        report.add_warning(
            f"Detached HEAD is {behind_main} commit(s) behind origin/main — "
            f"verify the pinned commit is the intended run-of-record."
        )
    report.git_status += " (detached HEAD)"


def _check_feature_branch_behind(
    report: PreflightReport, project_root: Path, branch: str, fetch_failed: bool, fetch_err: str
):
    """Feature branch: ERROR only when behind/diverged from its OWN origin ref.

    Divergence from origin/main is expected on a feature branch and demoted
    to an informational WARNING.
    """
    # A failed fetch means the branch's own origin ref may be stale, so
    # behind_own == 0 proves nothing — fail LOUD. This is the one place
    # fetch failure is an ERROR.
    if fetch_failed:
        report.add_error(
            f"git fetch origin failed ({fetch_err}) — cannot verify branch "
            f"{branch} is up to date with origin/{branch}."
        )
        report.git_status += ", fetch failed"
        return

    # Compare against the branch's OWN origin ref. The full refs/remotes/
    # spelling is used in BOTH the existence probe and the rev-list counts so
    # a same-named tag can never make the bare `origin/<branch>` form
    # ambiguous.
    own_ref = f"origin/{branch}"
    own_ref_full = f"refs/remotes/{own_ref}"
    rc_ref, _, _ = _run(
        ["git", "-C", str(project_root), "rev-parse", "--verify", "--quiet", own_ref_full]
    )
    if rc_ref == 0:
        behind_own = _behind_count(project_root, own_ref_full)
        ahead_own = _ahead_count(project_root, own_ref_full)
        if behind_own is None:
            report.add_warning(f"could not count commits behind {own_ref}")
        elif behind_own and ahead_own:
            report.add_error(
                f"Branch {branch} has diverged from {own_ref} ({behind_own} behind, "
                f"{ahead_own} ahead) — reconcile (rebase onto or merge {own_ref}); "
                f"a plain git pull --ff-only will fail."
            )
            report.git_status += f", diverged from {own_ref}"
        elif behind_own:
            report.add_error(
                f"Branch {branch} is {behind_own} commit(s) behind {own_ref}. "
                f"Run: git pull --ff-only origin {branch}"
            )
            report.git_status += f", {behind_own} behind {own_ref}"
        elif ahead_own:
            report.add_warning(
                f"Branch {branch} is {ahead_own} commit(s) ahead of {own_ref} "
                f"(committed but unpushed) — the running code is not the pushed "
                f"run-of-record."
            )
            report.git_status += f", {ahead_own} ahead of {own_ref}"
    else:
        report.add_warning(
            f"Branch {branch} has no pushed {own_ref} ref — cannot verify "
            f"up-to-date-ness (unpushed local branch)."
        )
        report.git_status += f" (no {own_ref})"

    behind_main = _behind_count(project_root, "origin/main")
    if behind_main:
        report.add_warning(
            f"Branch {branch} is {behind_main} commit(s) behind origin/main "
            f"(expected on a feature branch; informational)."
        )
        report.git_status += f", {behind_main} behind origin/main"


def check_env_sync(report: PreflightReport, project_root: Path):
    """Check that installed packages match uv.lock.

    Cluster branch: SKIPPED. The sbatch builds / activates the venv
    inside the job (cached at ``$SCRATCH/eps/venv-<lockhash>``), so a
    pre-launch ``uv sync --locked --dry-run`` on the login node would
    report an out-of-sync env that the job is about to fix. Mark
    ``env_synced=True`` with an explicit note in ``git_status`` is the
    wrong field; instead we leave ``env_synced`` True and append a
    warning so the summary's "Env synced: yes" reads honestly while a
    surfaced WARNING line documents the skip.
    """
    if is_cluster_env():
        report.add_warning(
            "env_sync check SKIPPED on cluster — sbatch builds the venv "
            "inside the job from $SCRATCH/eps/venv-<lockhash>."
        )
        report.env_synced = True
        return

    lockfile = project_root / "uv.lock"
    if not lockfile.exists():
        report.add_warning("No uv.lock found — cannot verify environment sync")
        report.env_synced = False
        return

    # uv sync --locked --dry-run exits non-zero if env needs changes
    rc, out, err = _run(
        ["uv", "sync", "--locked", "--dry-run"],
        timeout=30,
    )
    if rc != 0:
        if "would install" in err.lower() or "would install" in out.lower():
            report.add_error("Environment out of sync with uv.lock. Run: uv sync --locked")
            report.env_synced = False
        elif "error" in err.lower():
            report.add_warning(f"uv sync check failed: {err[:200]}")
            report.env_synced = False
        else:
            # Non-zero exit could mean changes needed
            report.add_warning(
                "uv sync --locked --dry-run returned non-zero. Environment may be out of sync."
            )
            report.env_synced = False


def _writable_probe_dir(check_path: str, candidates: list[str | None] | None = None) -> str | None:
    """Resolve a user-writable directory on the SAME filesystem as ``check_path``.

    ``check_path`` is created if missing (OSError suppressed) so out-root callers
    keep the create-if-missing contract (``assert_out_root_headroom`` may probe an
    out-root that does not exist yet). A writable+searchable ``check_path`` is
    returned directly — the fast path every current writable caller takes.
    Otherwise the candidates (default ``$HOME``, ``tempfile.gettempdir()``,
    ``os.getcwd()``) are scanned and the first non-empty, writable+searchable
    directory on the same ``st_dev`` is returned; None when none qualifies.

    Same-``st_dev`` is load-bearing: the probe measures the quota/headroom of
    ``check_path``'s FILESYSTEM, so probing a directory on a different mount
    would measure the wrong disk. Caveat: same ``st_dev`` does NOT guarantee the
    same QUOTA DOMAIN — ext4 project quotas and MooseFS per-directory quotas are
    subtree-scoped, so two directories on one filesystem can sit in different
    quota domains. Harmless for current callers (the candidate fallback fires
    only for unwritable check_paths, which ``_disk_check_path`` never produces
    on the quota-scoped mounts), but a future caller must not rely on
    cross-directory quota equivalence.
    """
    with contextlib.suppress(OSError):
        Path(check_path).mkdir(parents=True, exist_ok=True)
    if os.access(check_path, os.W_OK | os.X_OK):
        return check_path
    try:
        check_dev = os.stat(check_path).st_dev
    except OSError:
        return None
    if candidates is None:
        candidates = [os.environ.get("HOME"), tempfile.gettempdir(), os.getcwd()]
    for cand in candidates:
        if not cand:
            continue
        try:
            if os.access(cand, os.W_OK | os.X_OK) and os.stat(cand).st_dev == check_dev:
                return cand
        except OSError:
            continue
    return None


def _probe_writable_bytes(check_path: str, probe_bytes: int) -> tuple[bool, str | None]:
    """Try to reserve ``probe_bytes`` on ``check_path``'s filesystem via posix_fallocate.

    On RunPod MooseFS each pod has a per-pod writable-bytes quota (~130GB) that is
    separate from, and far below, the share-level free space ``shutil.disk_usage``
    reports. The only reliable way to detect the quota is to attempt a real
    allocation: a small canary reservation that we immediately delete.

    Args:
        check_path: Directory whose FILESYSTEM to probe. The probe file is written
            under ``check_path`` itself when it is user-writable (all pod /
            cluster / out-root callers); when it is not (``/`` on the local VM
            for a non-root user, #2042), the probe file is placed in a
            user-writable directory on the SAME filesystem
            (``_writable_probe_dir``). When no such location exists, the probe
            degrades via the documented fallback contract instead of raising.
        probe_bytes: Number of bytes to attempt to reserve. Keep this SMALL
            (a canary, ~1-2GB), NOT the full required free space — the goal is to
            detect EDQUOT/ENOSPC, not to reserve the experiment's footprint.

    Returns:
        (ok, fallback_reason). ``ok`` is True when the allocation succeeded.
        ``fallback_reason`` is set to a non-None string ONLY when the probe could
        not run (filesystem does not support — or does not reliably support —
        fallocate, or no user-writable probe location exists on the filesystem);
        in that case the caller must fall back to
        ``shutil.disk_usage`` and ``ok`` is True. ``ok`` is False when the
        allocation was actively refused (EDQUOT/ENOSPC) — at fallocate time or at
        probe-file creation (``os.open``) — with ``fallback_reason`` left None.

    Asserts probe_bytes > 0 — a zero-byte probe never exercises the quota.
    """
    assert probe_bytes > 0, f"probe_bytes must be positive, got {probe_bytes}"

    probe_dir = _writable_probe_dir(check_path)
    if probe_dir is None:
        return True, f"no user-writable probe location on the {check_path} filesystem"

    # Per-invocation unique filename: concurrent probes on a SHARED filesystem
    # (e.g. 8 per-unit workers each calling assert_out_root_headroom at startup
    # on a cluster share) must never open/fallocate/unlink one common path — a
    # sibling's unlink/recreate invalidates this process's fd mid-fallocate,
    # surfacing as OSError EBADF outside the handled errno sets (#1979 fellows
    # job 16686: 5 of 8 workers died rc=1 at the startup headroom probe).
    probe_path = Path(probe_dir) / f".preflight_disk_probe.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    fd = None
    try:
        try:
            fd = os.open(str(probe_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                # Probe-file CREATION refused on an already-exhausted quota — the
                # same real-refusal signal as an EDQUOT/ENOSPC from fallocate.
                return False, None
            # Anything else (e.g. EACCES from a post-resolver os.access lie on
            # root-squash NFS) re-raises to the caller's ``except OSError``.
            raise
        try:
            os.posix_fallocate(fd, 0, probe_bytes)
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                # Real headroom/quota signals (MooseFS per-pod EDQUOT) — never
                # swallowed into the fallback path.
                return False, None
            if e.errno in (errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL, errno.EBADF):
                # Filesystem doesn't support fallocate (tmpfs, some overlay FS,
                # macOS). VAST/NFS-class mounts surface EBADF from fallocate on
                # a just-opened valid fd (fellows /workspace, #1902 job 16139).
                # Caller falls back to shutil.disk_usage.
                return True, f"posix_fallocate unsupported (errno={e.errno})"
            raise
        return True, None
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            probe_path.unlink()


def _mount_point_of(path: Path | str) -> str:
    """Best-effort mount point of ``path``: longest /proc/mounts prefix of its realpath.

    Diagnostic only (names the filesystem in logs/errors); returns "?" when
    /proc/mounts is unreadable (non-Linux) — never raises.
    """
    try:
        real = os.path.realpath(path)
        best = "/"
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                mp = parts[1]
                covers = real == mp or real.startswith(mp.rstrip("/") + "/")
                if covers and len(mp) > len(best):
                    best = mp
        return best
    except OSError:
        return "?"


def assert_out_root_headroom(
    out_root: str | Path,
    need_gb: float,
    *,
    phase: str = "",
    canary_gb: float = 1.0,
) -> float:
    """Fail-loud disk-headroom assert at the filesystem ``out_root`` ACTUALLY resolves to.

    Generalizes the #1333 ``_assert_out_root_headroom`` pattern (plan §9 names the
    target mount per out-root; the workload preamble asserts headroom against that
    mount before each write-heavy phase): ``os.statvfs`` free-vs-floor at the
    out-root, plus a ``posix_fallocate`` canary (via ``_probe_writable_bytes``)
    that catches an already-exhausted RunPod MooseFS per-pod EDQUOT quota that
    statvfs is blind to. Raises RuntimeError naming path, resolved mount, free GB,
    and floor GB BEFORE the phase writes; returns free GB on success. A filesystem
    without fallocate support degrades to the statvfs check with a logged warning
    (the ``_probe_writable_bytes`` fallback contract). See
    ``.claude/rules/plan-compute-sizing.md`` § Out-root mount binding.
    """
    if need_gb <= 0:
        raise ValueError(f"need_gb must be positive, got {need_gb}")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(out_root)
    free_gb = st.f_bavail * st.f_frsize / 1e9  # decimal GB, matches the #1333 floor arithmetic
    mount = _mount_point_of(out_root)
    tag = f"[disk-headroom] {phase}:" if phase else "[disk-headroom]"
    if free_gb < need_gb:
        raise RuntimeError(
            f"{tag} out_root {out_root} (mount {mount}) has {free_gb:.1f} GB free "
            f"< required {need_gb:.1f} GB (plan §9 floor). Did the out-root resolve to a "
            f"smaller filesystem than the estimate was sized against? On RunPod, /tmp and "
            f"paths outside /workspace are the container disk (typically ~50 GB) — use a "
            f"/workspace-rooted out-root."
        )
    # Canary size deliberately in DECIMAL GB (int(canary_gb * 1e9)), matching the free-GB
    # arithmetic above (#1333 used 1 << 30; the ~7% gap is immaterial for EDQUOT detection).
    ok, fallback_reason = _probe_writable_bytes(str(out_root), int(canary_gb * 1e9))
    if not ok:
        raise RuntimeError(
            f"{tag} {canary_gb:.0f} GB fallocate canary FAILED at {out_root} (mount {mount}) "
            f"with statvfs free={free_gb:.1f} GB — per-pod quota (EDQUOT) or wedged "
            f"filesystem; fix before writing {need_gb:.1f} GB."
        )
    if fallback_reason:
        logger.warning("%s canary skipped: %s (statvfs-only check)", tag, fallback_reason)
    logger.info(
        "%s out_root=%s mount=%s free=%.1f GB (floor %.1f GB) canary=%s",
        tag,
        out_root,
        mount,
        free_gb,
        need_gb,
        "skipped" if fallback_reason else "ok",
    )
    return free_gb


def _quota_aware_headroom_gb(share_free_gb: float, quota_gb: float | None) -> tuple[float, str]:
    """Cap the usable headroom at the per-pod quota so over-quota footprints show.

    ``shutil.disk_usage`` reports share-level free (terabytes on RunPod MooseFS),
    which is blind to the per-pod EDQUOT quota. The small canary probe only fires
    once the quota is ALREADY exhausted; to surface an over-quota footprint BEFORE
    launch the usable headroom is ``min(quota_gb, share_free_gb)``.

    Args:
        share_free_gb: Share-level free space from ``shutil.disk_usage``.
        quota_gb: Per-pod writable-bytes quota in GB. None disables the cap (the
            headroom is then the raw share-level free, which CANNOT detect the
            quota — the basis string makes that explicit).

    Returns:
        (headroom_gb, basis) where ``basis`` names the binding signal so callers
        never mislabel a quota-blind number as "probed".
    """
    if quota_gb is None:
        return share_free_gb, "share-level free (quota cap disabled, over-quota undetectable)"
    if quota_gb < share_free_gb:
        return quota_gb, f"per-pod quota cap ({quota_gb:.0f}GB)"
    return share_free_gb, "share-level free (below per-pod quota cap)"


def check_disk_space(
    report: PreflightReport,
    min_free_gb: float,
    probe_gb: float = 1.0,
    quota_gb: float | None = RUNPOD_PER_POD_QUOTA_GB,
):
    """Check available disk space on /workspace (or /), quota-aware.

    Two distinct quota signals are combined:

    1. A real ``posix_fallocate`` canary probe catches an ALREADY-exhausted
       RunPod MooseFS per-pod EDQUOT quota (``shutil.disk_usage`` is blind to it).
    2. The usable headroom is capped at ``quota_gb`` so an over-quota *planned
       footprint* (one the small canary does NOT yet trip because the pod is not
       yet full) is still catchable by ``check_disk_budget``. Without this cap the
       headroom would be the share-level free (terabytes), and the budget check
       would be a no-op on exactly the filesystem it exists to protect.

    ``shutil.disk_usage`` is kept solely as the human-readable free-space reporter
    and as the share-level term of the headroom cap.

    Args:
        report: Mutated in place with disk findings (``disk_free_gb``,
            ``disk_probed_headroom_gb``, ``disk_headroom_basis``).
        min_free_gb: Minimum free space required to run.
        probe_gb: Size of the canary allocation, in GB. Small by design (default
            1GB) — it detects the quota, it does not reserve the full footprint.
        quota_gb: Per-pod writable-bytes quota in GB used to cap usable headroom.
            Defaults to ``RUNPOD_PER_POD_QUOTA_GB``. Pass a larger value for pods
            provisioned with an explicit storage spec, or None to disable the cap
            (the headroom then cannot detect over-quota footprints).
    """
    check_path = _disk_check_path()

    # Human-readable share-level free space (NOT the sole go/no-go signal).
    try:
        usage = shutil.disk_usage(check_path)
        report.disk_free_gb = usage.free / (1024**3)
    except Exception as e:
        report.add_warning(f"Could not read disk usage on {check_path}: {e}")

    probe_bytes = max(1, int(probe_gb * (1024**3)))
    try:
        ok, fallback_reason = _probe_writable_bytes(check_path, probe_bytes)
    except OSError as e:
        report.add_warning(f"Could not run disk-quota probe on {check_path}: {e}")
        ok, fallback_reason = True, f"probe raised {e}"

    headroom_gb, headroom_basis = _quota_aware_headroom_gb(report.disk_free_gb, quota_gb)

    if fallback_reason is not None:
        # Probe could not run — fall back to shutil.disk_usage for the ALREADY-
        # exhausted signal, but STILL cap headroom at the static quota so a
        # planned over-quota footprint is caught downstream.
        report.add_warning(
            f"Disk-quota probe skipped on {check_path}: {fallback_reason}. "
            f"Falling back to shutil.disk_usage; the live per-pod EDQUOT quota "
            f"cannot be detected, so headroom is capped at the static quota "
            f"({headroom_basis})."
        )
        report.disk_probed_headroom_gb = headroom_gb
        report.disk_headroom_basis = headroom_basis
        if report.disk_free_gb < min_free_gb:
            report.add_error(
                f"Only {report.disk_free_gb:.1f}GB free on {check_path} "
                f"(need {min_free_gb:.0f}GB). Clean up models/checkpoints."
            )
        elif report.disk_free_gb < min_free_gb * 2:
            report.add_warning(f"{report.disk_free_gb:.1f}GB free on {check_path} — getting low")
        _check_vm_root_floor(report, check_path, min_free_gb)
        return

    if not ok:
        # The pod refused even the small canary — quota is exhausted.
        report.disk_probed_headroom_gb = 0.0
        report.disk_headroom_basis = "per-pod quota exhausted (canary refused)"
        report.add_error(
            f"Disk-quota probe FAILED on {check_path}: cannot allocate even "
            f"{probe_gb:.1f}GB (EDQUOT/ENOSPC). Share-level free reports "
            f"{report.disk_free_gb:.1f}GB, but this pod has exhausted its per-pod "
            f"writable-bytes quota. Clean up models/checkpoints or provision a "
            f"larger volume."
        )
        return

    # Probe of probe_gb succeeded, so the quota is not YET exhausted. The usable
    # headroom is capped at the per-pod quota (NOT the terabyte-scale share-level
    # free) so an over-quota planned footprint is caught by check_disk_budget.
    report.disk_probed_headroom_gb = headroom_gb
    report.disk_headroom_basis = headroom_basis
    if report.disk_free_gb < min_free_gb:
        report.add_error(
            f"Only {report.disk_free_gb:.1f}GB free on {check_path} "
            f"(need {min_free_gb:.0f}GB). Clean up models/checkpoints."
        )
    elif report.disk_free_gb < min_free_gb * 2:
        report.add_warning(f"{report.disk_free_gb:.1f}GB free on {check_path} — getting low")
    _check_vm_root_floor(report, check_path, min_free_gb)


def estimate_footprint_gb(
    base_model_gb: float,
    n_cells: int,
    materialize_merged: bool = True,
) -> float:
    """Estimate peak disk footprint (GB) for a multi-cell experiment.

    A rough budgeting aid for ``check_disk_budget`` — NOT an exact accounting.
    Each cell holds one base-model-sized checkpoint on disk; when merged adapters
    are materialized, a cell briefly holds a second base-model-sized copy
    (adapter + merged) at peak.

    Args:
        base_model_gb: On-disk size of one base-model / checkpoint copy in GB.
        n_cells: Number of cells (conditions x seeds) whose checkpoints coexist
            on disk at peak. Use 1 for a strictly sequential, delete-after-each run.
        materialize_merged: If True, account for the transient merged-adapter copy
            (the LoRA-merge step where adapter + merged both exist).

    Returns:
        Estimated peak footprint in GB.

    Asserts base_model_gb >= 0 and n_cells >= 1.
    """
    assert base_model_gb >= 0, f"base_model_gb must be non-negative, got {base_model_gb}"
    assert n_cells >= 1, f"n_cells must be >= 1, got {n_cells}"

    per_cell = base_model_gb * (2.0 if materialize_merged else 1.0)
    return per_cell * n_cells


def check_disk_budget(report: PreflightReport, planned_footprint_gb: float | None):
    """FAIL when the estimated experiment footprint exceeds usable disk headroom.

    Usable headroom is ``report.disk_probed_headroom_gb`` — quota-capped by
    ``check_disk_space`` so it is NOT the terabyte-scale share-level free that
    ``shutil.disk_usage`` reports on RunPod MooseFS. The FAIL message names the
    headroom basis (``report.disk_headroom_basis``) so the number is never
    mislabeled as "probed" when it is a share-level / quota-capped estimate.

    Ranked remediation (cheapest first): LoRA-only (skip merged-adapter
    materialization), sequentialize multi-cell sweeps, provision a larger volume.

    Args:
        report: Mutated in place. Reads ``disk_probed_headroom_gb`` +
            ``disk_headroom_basis`` (set by ``check_disk_space``); call this AFTER
            ``check_disk_space``.
        planned_footprint_gb: Estimated peak footprint in GB. None => skip (no
            budget information supplied).
    """
    if planned_footprint_gb is None:
        return

    headroom = report.disk_probed_headroom_gb
    basis = report.disk_headroom_basis
    if planned_footprint_gb > headroom:
        report.add_error(
            f"Disk budget exceeded: planned footprint {planned_footprint_gb:.1f}GB "
            f"> usable headroom {headroom:.1f}GB (basis: {basis}). Remediation, "
            f"cheapest first: "
            f"(1) LoRA-only — skip merged-adapter materialization to halve per-cell "
            f"disk; (2) sequentialize — run conditions/seeds one at a time and "
            f"delete each checkpoint before the next so peak disk = one cell; "
            f"(3) provision a larger volume / pod with explicit storage spec."
        )


#: VM root-disk advisory thresholds (WARN-only). The local VM root disk (``/``)
#: fills because each experiment downloads source data into
#: ``data/issue_<N>/hf_dl`` caches that nothing reclaims (incident 2026-06-25:
#: ``/`` hit 100% full, one finished experiment held 97 GB). Preflight surfaces
#: a heads-up so a near-full root disk is caught BEFORE a launch wedges git /
#: task.py across sessions; remediation is ``scripts/vm_disk_guard.py --apply``.
VM_ROOT_DISK_WARN_PCT = 90.0
VM_ROOT_DISK_WARN_FREE_GB = 20.0


def check_vm_root_disk(
    report: PreflightReport,
    warn_pct: float = VM_ROOT_DISK_WARN_PCT,
    warn_free_gb: float = VM_ROOT_DISK_WARN_FREE_GB,
):
    """WARN (never FAIL) when the local VM root disk ``/`` is nearly full.

    This is the LOCAL orchestration disk, distinct from
    :func:`check_disk_space`'s experiment-surface probe (``/workspace`` on
    RunPod, ``$SCRATCH`` on the cluster). On RunPod / cluster the experiment
    runs off-box, so ``/`` is the VM's own root — its exhaustion wedges git /
    task.py / dispatch rather than the training job, so a heads-up here is
    advisory, not a hard gate. A read error degrades to a single warning."""
    try:
        usage = shutil.disk_usage("/")
    except OSError as e:
        report.add_warning(f"Could not read VM root-disk usage on /: {e}")
        return
    used_pct = 100.0 * usage.used / usage.total
    free_gb = usage.free / (1024**3)
    if used_pct > warn_pct or free_gb < warn_free_gb:
        report.add_warning(
            f"VM root disk / is {used_pct:.1f}% used ({free_gb:.1f} GB free) — "
            f"near full (warn at {warn_pct:.0f}% used / {warn_free_gb:.0f} GB free). "
            f"Run: uv run python scripts/vm_disk_guard.py --apply "
            f"(clears re-downloadable data/issue_*/hf_dl caches + stale logs)."
        )


def check_gpus(report: PreflightReport, require_gpu: bool, min_free_mb: int):
    """Check GPU availability and memory."""
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        if require_gpu:
            report.add_error(f"nvidia-smi failed: {err}. No GPUs available?")
        else:
            report.add_warning("nvidia-smi not available (no GPU)")
        return

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_id, total, used, free = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

        # Check for processes on this GPU
        prc, pout, _ = _run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ]
        )
        n_procs = len([x for x in pout.splitlines() if x.strip()]) if prc == 0 else 0

        gpu_info = {
            "id": gpu_id,
            "memory_total_mb": total,
            "memory_used_mb": used,
            "memory_free_mb": free,
            "processes": n_procs,
        }
        report.gpu_info.append(gpu_info)

        # Zombie detection: memory used but no processes
        if used > 5000 and n_procs == 0:
            report.add_warning(
                f"GPU {gpu_id}: {used}MB used but no processes — "
                f"possible zombie. Fix: restart container."
            )

    # Check if any GPU has enough free memory
    if require_gpu:
        max_free = max((g["memory_free_mb"] for g in report.gpu_info), default=0)
        if max_free < min_free_mb:
            report.add_error(
                f"No GPU with {min_free_mb:,}MB free (best: {max_free:,}MB). "
                f"Wait for running jobs or use a different pod."
            )


def check_hf_home(report: PreflightReport):
    """Check ``HF_HOME`` matches the canonical per-environment default.

    Three-way (mirrors :func:`env._hf_home_default`):

    * Cluster: expects ``$SCRATCH/.cache/huggingface``.
    * RunPod:  expects ``/workspace/.cache/huggingface``.
    * Local:   no canonical path; only warn if HF_HOME is empty.
    """
    hf_home = os.environ.get("HF_HOME", "")

    if is_cluster_env() or is_runpod_env():
        expected = _hf_home_default()
        if not hf_home:
            report.add_warning(
                f"HF_HOME not set. Setting to {expected}. "
                "Call load_dotenv() or source env_setup.sh first."
            )
            os.environ["HF_HOME"] = expected
        elif hf_home != expected:
            report.add_warning(
                f"HF_HOME={hf_home} (expected {expected}). Models may download to wrong location."
            )


def check_env_vars(report: PreflightReport, required: list[str]):
    """Check that required environment variables are set."""
    for var in required:
        val = os.environ.get(var, "")
        if not val:
            report.add_error(f"Missing env var: {var}. Check .env file.")
        elif len(val) < 5:
            report.add_warning(f"Env var {var} looks suspiciously short: '{val[:3]}...'")


def check_vllm_transformers_compat(report: PreflightReport):
    """Refuse to proceed when vLLM 0.11.x is resolved against transformers >=5.

    vLLM 0.11.0 calls `tokenizer.all_special_tokens_extended`, which transformers 5.x
    removed. Every fresh pod hits this 10 sec into the first `LLM(...)` init. This has
    recurred across issues #238, #261, #263, #269, #331, #354, #368 — caught here so
    the next fresh pod fails preflight in <2 sec instead of crashing in vLLM later.
    """
    try:
        import transformers
        import vllm
    except ImportError as e:
        report.add_warning(f"Could not import vllm/transformers for compat check: {e}")
        return

    t_ver = transformers.__version__
    v_ver = vllm.__version__
    t_major = int(t_ver.split(".")[0])
    v_minor = ".".join(v_ver.split(".")[:2])
    if v_minor in {"0.11"} and t_major >= 5:
        report.add_error(
            f"vLLM/transformers version skew: vllm=={v_ver} + transformers=={t_ver}. "
            f"vLLM 0.11.x calls tokenizer.all_special_tokens_extended which transformers "
            f">=5 removed. Every LLM(...) instantiation will crash. Fix: pin "
            f"`transformers>=4.46,<5.0` in pyproject.toml and re-run `uv sync --locked`. "
            f"See .claude/agent-memory/experimenter/feedback_vllm0110_transformers5_breakage.md"
        )


def check_connectivity(report: PreflightReport):
    """Quick check that HF Hub and WandB are reachable."""
    # HF Hub
    rc, _, _ = _run(
        [
            "python3",
            "-c",
            "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)",
        ],
        timeout=10,
    )
    if rc != 0:
        report.add_warning("Cannot reach huggingface.co — model uploads will fail")

    # WandB
    wandb_check = (
        "import urllib.request; urllib.request.urlopen('https://api.wandb.ai/healthz', timeout=5)"
    )
    rc, _, _ = _run(["python3", "-c", wandb_check], timeout=10)
    if rc != 0:
        report.add_warning("Cannot reach api.wandb.ai — result uploads will fail")


# --- HF large-blob GET probe (#2185) ------------------------------------------
# Probe THROUGH huggingface.co (never a CDN hostname): the redirect to the CDN
# is what exercises whichever edge the pod's DNS steers to — the broken edge
# set is DC-dependent, so hardcoding an edge would defeat the probe. Override
# via EPM_PREFLIGHT_LARGE_BLOB_URL (e.g. if the shard name ever changes).
DEFAULT_LARGE_BLOB_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00001-of-00004.safetensors"
)
LARGE_BLOB_RANGE_BYTES = 1024 * 1024  # 1 MiB — sub-second even at a degraded ~3 MB/s edge
LARGE_BLOB_TIMEOUT_S = 20.0  # total wall bound so a hung edge cannot stall preflight


def check_hf_large_blob_get(report: PreflightReport, opener=None):
    """RunPod-gated, fail-open probe for the HF-CDN zero-byte large-blob trap (#2185).

    A pod's DNS can steer ``us.aws.cdn.hf.co`` to a CDN edge that answers
    large-blob GETs with HTTP 206 and ZERO bytes (#2162) — one upstream fault
    that impersonates all three download-accelerator failures in turn, so no
    client toggle fixes it. This is :func:`check_connectivity`'s large-blob
    twin: that check does a SMALL GET, which the broken edges serve normally;
    the small-vs-large split is the whole discriminator. The verdict keys on
    BYTES RECEIVED, never throughput — a working-but-slow edge (~3 MB/s was
    measured in #2162) is legitimate and must not WARN.

    WARN-only and fail-open by design: only the trap's signature — the
    connection CLOSING (EOF) with fewer bytes than the requested 1 MiB range —
    WARNs, with the diagnosis + a pointer to the gotchas entry. Every other
    outcome (404 if the shard name changes, DNS failure, timeout, a deadline
    hit mid-read, missing token, any unexpected exception) degrades to a quiet
    inconclusive log line and proceeds. Never adds an error, never raises,
    never flips ``report.ok`` — a false hard FAIL blocks launches fleet-wide,
    and the gotchas entry (not this probe) is the load-bearing deliverable.

    RunPod-gated via ``is_runpod_env()``: the trap is confirmed on RunPod, the
    VM is the WORKING side by construction, and a SLURM compute node may
    legitimately lack egress — a variant on another lane needs its own gating
    decision rather than a silent widening. Kill switch:
    ``EPM_SKIP_LARGE_BLOB_PROBE=1``. ``opener`` is a test seam mirroring
    ``urllib.request.urlopen(req, timeout=...)``; production callers leave it
    ``None``.
    """
    if not is_runpod_env():
        return
    if os.environ.get("EPM_SKIP_LARGE_BLOB_PROBE") == "1":
        return
    url = os.environ.get("EPM_PREFLIGHT_LARGE_BLOB_URL") or DEFAULT_LARGE_BLOB_URL
    received = 0
    status: int | None = None
    eof = False
    try:
        import time
        import urllib.request

        open_fn = opener if opener is not None else urllib.request.urlopen
        req = urllib.request.Request(
            url, headers={"Range": f"bytes=0-{LARGE_BLOB_RANGE_BYTES - 1}"}
        )
        deadline = time.monotonic() + LARGE_BLOB_TIMEOUT_S
        with open_fn(req, timeout=10) as resp:
            status = getattr(resp, "status", None)
            while received < LARGE_BLOB_RANGE_BYTES:
                if time.monotonic() > deadline:
                    # Bytes still flowing (or a stall) at the wall bound: cannot
                    # distinguish a slow edge from a broken one without keying on
                    # throughput, which is exactly what this probe must not do.
                    logger.info(
                        "HF large-blob probe inconclusive: %.0fs deadline hit at "
                        "%d bytes — fail-open, no warning",
                        LARGE_BLOB_TIMEOUT_S,
                        received,
                    )
                    return
                chunk = resp.read(min(65536, LARGE_BLOB_RANGE_BYTES - received))
                if not chunk:
                    eof = True
                    break
                received += len(chunk)
    except Exception as e:  # fail-open by design — advisory probe (see docstring)
        logger.info(
            "HF large-blob probe inconclusive (%s: %s) — fail-open, no warning",
            type(e).__name__,
            e,
        )
        return
    if eof and received < LARGE_BLOB_RANGE_BYTES:
        report.add_warning(
            f"HF large-blob GET returned {status} with {received} bytes "
            f"(expected >=1 MiB) — this pod may be DNS-steered to a CDN edge that "
            f"serves 206 + 0 bytes. No accelerator toggle fixes it; see "
            f".claude/rules/gotchas.md (RunPod HF-CDN zero-byte large-blob trap) for "
            f"the three-curl discriminator and the VM-to-pod parallel-rsync relay "
            f"recovery."
        )


def check_hf_storage(report: PreflightReport, planned_upload_gb: float | None = None):
    """Non-fatal WARN when account HF public storage exceeds the soft ceiling —
    plus an OPT-IN hard gate against a caller-supplied planned LFS upload (#1034).

    Advisory only by default: never adds an error, never raises — an
    unreachable HF API degrades to an 'unknown headroom' warning, and a
    non-parseable ceiling/TTL env value (the helper's deliberate
    ``ValueError``) is caught and reported as a warning here (it still
    propagates at the fail-loud persist gate in ``train/trainer.py``). See
    ``.claude/rules/upload-policy.md`` § HF storage-quota 403 for the incident
    this fronts (#541/#552).

    ``planned_upload_gb`` (decimal GB) means "LFS bytes the run REQUIRES on
    the canonical PUBLIC repos"; runs whose stores tolerate overflow (v2
    ``upload_dir_sharded`` flows) either omit it or arm
    ``EPM_HF_OVERFLOW_ROUTING=1`` — so the gate cannot false-block them. When
    supplied, the projection is routed THROUGH
    :func:`hub.check_projected_upload_headroom` (``probe_floor_gb=0.0`` — an
    explicit projection always probes) so the FAIL arm inherits the live
    ``force_refresh=True`` confirm: LIVE-CONFIRMED-insufficient + routing off
    → ERROR; routing armed → WARNING; unknown/disabled → WARNING (fail-open —
    a stale-high cache can never false-block).
    """
    try:
        from explore_persona_space.orchestrate.hub import check_hf_storage_headroom

        h = check_hf_storage_headroom()
    except Exception as e:
        report.add_warning(f"HF storage headroom check failed ({e}) — headroom unknown")
        return
    report.hf_storage_used_tb = h.used_tb
    report.hf_storage_ceiling_tb = h.ceiling_tb
    report.hf_storage_basis = h.basis
    if h.basis == "disabled":
        if planned_upload_gb is not None:
            # Armed-but-blind combo made visible (#1034): a REQUESTED gate must
            # never be silently swallowed by the kill switch (mirrors the
            # _OVERFLOW_BLIND_WARNED precedent in hub.py).
            report.add_warning(
                "HF headroom: planned-upload gate requested but storage check "
                "disabled (EPM_HF_STORAGE_CHECK=0) — gate not evaluated"
            )
        return
    if h.used_tb is None:
        report.add_warning(
            f"HF public-storage usage unknown ({h.basis}) — cannot verify upload headroom"
        )
    elif h.over_ceiling:
        report.add_warning(
            f"HF public storage {h.used_tb:.2f} TB exceeds soft ceiling "
            f"{h.ceiling_tb:.1f} TB ({h.n_repos} repos, {h.basis}) — LFS uploads "
            f"(adapters/checkpoints) will 403 at the hard quota; see "
            f".claude/rules/upload-policy.md § HF storage-quota 403"
        )
    if planned_upload_gb is not None and h.used_tb is not None:
        try:
            from explore_persona_space.orchestrate.hub import check_projected_upload_headroom

            # NEVER gate on the raw cached probe (Must-Fix): the wrapper embeds
            # the force_refresh=True live confirm before any "insufficient"
            # verdict, so a stale-high cache cannot false-block a healthy run.
            # probe_floor_gb=0.0: the caller EXPLICITLY declared the projection,
            # so the tiny-upload skip must not disarm a requested gate.
            ph = check_projected_upload_headroom(int(planned_upload_gb * 1e9), probe_floor_gb=0.0)
        except Exception as e:
            report.add_warning(f"HF planned-upload headroom gate failed ({e}) — gate not evaluated")
            return
        if ph.verdict == "insufficient":
            if os.environ.get("EPM_HF_OVERFLOW_ROUTING") == "1":
                report.add_warning(
                    f"HF headroom: planned {planned_upload_gb:.0f} GB LFS upload would "
                    f"exceed the soft ceiling ({ph.used_tb:.2f}+{ph.projected_tb:.2f} > "
                    f"{ph.ceiling_tb:.1f} TB, live-confirmed) — EPM_HF_OVERFLOW_ROUTING=1 "
                    f"is armed, uploads will reroute to the private overflow repo "
                    f"(NOTE: arming is environment-wide — it also reroutes every armed "
                    f"upload_model flow in this env)"
                )
            else:
                report.add_error(
                    f"HF headroom insufficient for planned upload (live-confirmed): "
                    f"{planned_upload_gb:.0f} GB projected vs "
                    f"{(ph.ceiling_tb - ph.used_tb) * 1000:.0f} GB remaining "
                    f"({ph.used_tb:.2f}/{ph.ceiling_tb:.1f} TB, {ph.basis}). Remediation: "
                    f"(1) route the store through upload_dir_sharded (proactive overflow, "
                    f"#1034); (2) arm EPM_HF_OVERFLOW_ROUTING=1 for upload_model flows whose "
                    f"arming contract permits (environment-wide effect); (3) free quota / raise "
                    f"EPM_HF_STORAGE_SOFT_CEILING_TB after buying storage. "
                    f"See .claude/rules/upload-policy.md § Proactive detection."
                )
        elif ph.verdict in {"unknown", "disabled"}:
            # A REQUESTED gate that could not be evaluated must say so (#1034
            # revision): on the stale-high-cache-then-live-unknown arm the
            # cached read is KNOWN (no "usage unknown" WARN above fires), so
            # without this the report would be warning-free about the gate.
            report.add_warning(
                f"HF headroom: planned-upload gate not evaluated ({ph.verdict}, "
                f"basis={ph.basis}) — fail-open; the reactive 403 backstop stays "
                f"authoritative"
            )
        # ph.verdict == "fits" -> silent (fail-open: a stale-high cache whose
        # live re-probe fits never blocks and adds no warning).


def check_hf_lfs_write_gate(report: PreflightReport, planned_upload_gb: float | None = None):
    """Billing/quota write-gate probe at declared production scale (#1654).

    Runs ``hub.check_lfs_write_gate()`` — a zero-byte LFS batch-negotiation
    probe declaring ~16 GB — because the Step 6a.6 1 KB text probe is
    structurally FALSE-GREEN for quota/billing 403s, which fire only on the
    LFS endpoint (#1586: a 2 MB probe passed while 15.2 GB FT checkpoints
    403'd on "You need to setup automatic credit recharge").

    Mirrors :func:`check_hf_storage`'s #1034 verdict semantics exactly:

    * ``"disabled"``: silent; if ``planned_upload_gb`` is supplied, WARN that
      the requested gate was not evaluated (kill switch wins, visibly).
    * ``"ok"``: record fields, silent (the summary line carries the verdict).
    * ``"billing-blocked"`` / ``"storage-blocked"`` (live-confirmed by
      construction — the probe IS live): ``planned_upload_gb`` supplied ->
      ERROR with the remediation message; else -> WARNING (same message).
    * ``"unknown"``: WARN (fail-open — the reactive 403 backstop stays
      authoritative).

    Advisory by default: never raises — any helper exception (including the
    deliberate ``ValueError`` on a non-parseable ``EPM_HF_BILLING_PROBE_GB``)
    degrades to a warning here, matching the sibling gate.
    """
    try:
        from explore_persona_space.orchestrate.hub import check_lfs_write_gate

        probe = check_lfs_write_gate()
    except Exception as e:
        report.add_warning(f"HF LFS write-gate probe failed ({e}) — gate not evaluated")
        return
    report.hf_lfs_write_verdict = probe.verdict
    report.hf_lfs_write_detail = probe.detail
    report.hf_lfs_write_probe_gb = probe.probe_gb
    if probe.verdict == "disabled":
        if planned_upload_gb is not None:
            report.add_warning(
                "HF LFS write gate: gate requested but billing probe disabled "
                "(EPM_HF_BILLING_PROBE=0) — gate not evaluated"
            )
        return
    if probe.verdict == "unknown":
        report.add_warning(
            f"HF LFS write gate: probe inconclusive ({probe.detail}) — fail-open; "
            f"the reactive 403 backstop stays authoritative"
        )
        return
    if probe.verdict in {"billing-blocked", "storage-blocked"}:
        msg = (
            f"HF LFS write path BLOCKED at {probe.probe_gb:.0f} GB declared scale on "
            f"{probe.repo_id} ({probe.verdict}): {probe.detail}. A KB-MB canary passes "
            f"while GB-scale uploads 403 (#1586). Remediation (billing): enable automatic "
            f"credit recharge at huggingface.co/settings/billing -> 'Billing' -> "
            f"'Auto-recharge' (end-state: the toggle shows ON with a recharge amount + a "
            f"valid payment method listed); (storage): free quota or see "
            f".claude/rules/upload-policy.md § HF storage-quota 403 — the error excerpt "
            f"above governs the exact path (e.g. a 'storage patterns' manual-review 403 "
            f"names its own contact address). Account context: {probe.billing_context}."
        )
        if planned_upload_gb is not None:
            report.add_error(msg)
        else:
            report.add_warning(msg)
    # probe.verdict == "ok" -> silent (recorded fields + summary line only).


def preflight_check(
    require_gpu: bool = True,
    min_disk_gb: float = 50.0,
    min_gpu_free_mb: int = 70_000,
    required_env_vars: list[str] | None = None,
    check_code_sync: bool = True,
    planned_footprint_gb: float | None = None,
    per_pod_quota_gb: float | None = RUNPOD_PER_POD_QUOTA_GB,
    planned_upload_gb: float | None = None,
) -> PreflightReport:
    """Run all pre-experiment checks.

    Args:
        require_gpu: If True, fail when no GPU has enough free memory.
        min_disk_gb: Minimum free disk space in GB.
        min_gpu_free_mb: Minimum free GPU memory in MB for at least one GPU.
        required_env_vars: Env vars to check. Defaults to standard set.
        check_code_sync: Whether to check git status and env sync.
        planned_footprint_gb: Estimated peak experiment disk footprint in GB. When
            supplied, the disk-budget check FAILs if it exceeds usable (quota-
            capped) headroom. None (default) => skip the budget check, so existing
            callers are unaffected.
        per_pod_quota_gb: RunPod MooseFS per-pod writable-bytes quota in GB used to
            cap usable disk headroom (defaults to ``RUNPOD_PER_POD_QUOTA_GB``).
            None disables the cap (over-quota footprints become undetectable).
        planned_upload_gb: Planned canonical-public LFS upload size in decimal GB
            (#1034). When supplied, the HF-storage check hard-FAILs on a
            LIVE-CONFIRMED insufficient headroom with overflow routing off
            (WARNs when armed; fail-open on unknown/disabled), AND the LFS
            write-gate leg (#1654) hard-FAILs on a billing-blocked /
            storage-blocked batch-negotiation probe. None (default) => WARN-only
            behavior on both legs, so existing callers are unaffected.

    Returns:
        PreflightReport with pass/fail status and details.
    """
    if required_env_vars is None:
        required_env_vars = [
            "WANDB_API_KEY",
            "HF_TOKEN",
            "ANTHROPIC_API_KEY",
        ]

    project_root = _find_project_root()
    report = PreflightReport()

    # Load .env first so env var checks work. Use the canonical loader so a
    # linked worktree without its own .env falls back to the main worktree's.
    try:
        from explore_persona_space.orchestrate.env import load_dotenv as _load_dotenv

        _load_dotenv()
    except ImportError:
        report.add_warning("python-dotenv not installed — cannot load .env")

    # Set HF_HOME early — three-way: cluster → $SCRATCH, RunPod →
    # /workspace, local → project-local. See env._hf_home_default.
    if is_cluster_env() or is_runpod_env():
        os.environ.setdefault("HF_HOME", _hf_home_default())

    # Cluster bypasses the RunPod MooseFS 130 GB cap: $SCRATCH has a
    # per-user quota the cluster admins set (multi-TB on Nibi/Fir), not
    # the RunPod cap. The caller can still override per-pod-quota-gb
    # explicitly when a RunPod pod was provisioned with a custom volume.
    effective_quota_gb = None if is_cluster_env() else per_pod_quota_gb

    # Run all checks
    if check_code_sync:
        check_git_status(report, project_root)
        check_env_sync(report, project_root)

    check_disk_space(report, min_disk_gb, quota_gb=effective_quota_gb)
    check_disk_budget(report, planned_footprint_gb)
    check_vm_root_disk(report)
    check_gpus(report, require_gpu, min_gpu_free_mb)
    check_hf_home(report)
    check_env_vars(report, required_env_vars)
    check_vllm_transformers_compat(report)
    check_connectivity(report)
    check_hf_large_blob_get(report)
    check_hf_storage(report, planned_upload_gb)
    check_hf_lfs_write_gate(report, planned_upload_gb)

    return report


def require_preflight(
    min_disk_gb: float = 50.0,
    require_gpu: bool = True,
    min_gpu_free_mb: int = 70_000,
) -> PreflightReport:
    """Run preflight checks and abort if any critical failures.

    Call at the top of experiment scripts.
    """
    report = preflight_check(
        min_disk_gb=min_disk_gb,
        require_gpu=require_gpu,
        min_gpu_free_mb=min_gpu_free_mb,
    )
    if report.ok:
        logger.info(report.summary())
        return report

    # FAIL path — fail LOUD on a real stream. A handler-less logger.info()
    # emits zero bytes (root logger defaults to WARNING with no handlers), so
    # a launcher under `set -e` dies with an unattributable 0-byte log (#550).
    # The summary is emitted by exactly ONE statement per branch: logger.info
    # on PASS, raw stderr here on FAIL — never both.
    print(report.summary(), file=sys.stderr)
    logger.error("Pre-flight check FAILED. Fix errors before running.")
    sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code (0 = preflight PASS)."""
    import argparse

    parser = argparse.ArgumentParser(description="Run pre-flight checks")
    parser.add_argument("--no-gpu", action="store_true", help="Don't require GPU")
    parser.add_argument("--min-disk", type=float, default=50.0, help="Min disk GB")
    parser.add_argument(
        "--planned-footprint-gb",
        type=float,
        default=None,
        help="Estimated peak experiment disk footprint in GB; FAILs preflight if "
        "it exceeds usable (quota-capped) headroom. Omit to skip the budget check.",
    )
    parser.add_argument(
        "--planned-upload-gb",
        type=float,
        default=None,
        help="Planned canonical-public LFS upload size in decimal GB (#1034); FAILs "
        "preflight when a LIVE-CONFIRMED headroom read says used + planned exceeds the "
        "soft ceiling and EPM_HF_OVERFLOW_ROUTING is off (WARNs when armed; fail-open "
        "on unknown), and arms the #1654 LFS write-gate probe (billing/quota 403 at "
        "declared ~16 GB scale) as a hard gate. Omit to keep the WARN-only advisory.",
    )
    parser.add_argument(
        "--per-pod-quota-gb",
        type=float,
        default=RUNPOD_PER_POD_QUOTA_GB,
        help="RunPod MooseFS per-pod writable-bytes quota in GB used to cap usable "
        "disk headroom (default %(default)s). Pass a larger value for pods with an "
        "explicit storage spec. Use a negative value to disable the cap (over-quota "
        "footprints then become undetectable).",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--pipeline-check",
        action="store_true",
        help="Run integration tests (pytest tests/integration/ -m integration) after preflight",
    )
    args = parser.parse_args(argv)

    # A negative quota means "disable the cap" (argparse cannot pass None cleanly).
    per_pod_quota_gb = None if args.per_pod_quota_gb < 0 else args.per_pod_quota_gb

    report = preflight_check(
        require_gpu=not args.no_gpu,
        min_disk_gb=args.min_disk,
        planned_footprint_gb=args.planned_footprint_gb,
        per_pod_quota_gb=per_pod_quota_gb,
        planned_upload_gb=args.planned_upload_gb,
    )

    if args.json:
        # Contract (gotchas.md): exactly one pretty-printed JSON object and
        # nothing else on stdout — consumers parse the WHOLE stdout.
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "gpu_info": report.gpu_info,
                    "disk_free_gb": report.disk_free_gb,
                    "disk_probed_headroom_gb": report.disk_probed_headroom_gb,
                    "disk_headroom_basis": report.disk_headroom_basis,
                    "hf_storage_used_tb": report.hf_storage_used_tb,
                    "hf_storage_ceiling_tb": report.hf_storage_ceiling_tb,
                    "hf_storage_basis": report.hf_storage_basis,
                    "git_status": report.git_status,
                    "env_synced": report.env_synced,
                },
                indent=2,
            )
        )
    else:
        # Bare mode: print(), never a handler-less logger.info() that emits
        # zero bytes and leaves a `set -e` death unattributable (#550/#554).
        print(report.summary())
        if not report.ok:
            for e in report.errors:
                print(f"preflight ERROR: {e}", file=sys.stderr)

    if not report.ok:
        return 1

    if args.pipeline_check:
        # Status lines stay off stdout in --json mode (JSON-purity contract).
        if not args.json:
            print("Running integration tests...")
        rc, stdout, stderr = _run(
            [sys.executable, "-m", "pytest", "tests/integration/", "-m", "integration", "-x", "-v"],
            timeout=600,
        )
        if stdout:
            # In --json mode pytest output routes to stderr so stdout stays
            # exactly one parseable JSON object (gotchas.md contract); bare
            # mode keeps pytest stdout on stdout.
            print(stdout, file=sys.stderr if args.json else sys.stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        if rc != 0:
            print(f"Integration tests FAILED (exit code {rc})", file=sys.stderr)
            return rc
        if not args.json:
            print("Integration tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
