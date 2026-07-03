"""Task #821: pods.conf lives at ``<git-common-dir>/eps/pods.conf`` — OUT
of the git working tree, so destructive git ops cannot wipe it.

The v3 relocation is the load-bearing half of the #821 fix (guard + atomic
+ self-heal are the layered defense). Tests below pin:

- The resolver returns the LIVE path when it exists, migrates seed→live
  on first use, and holds ``locked_pods_conf`` for the migration so two
  concurrent processes cannot race.
- Every destructive git op — ``git reset --hard``, ``git checkout -- .``,
  ``git restore -- .``, ``git clean -fd``, ``git clean -fdx`` — leaves the
  LIVE file intact with its rows in place.
- The static resolved path is NOT inside the working tree.
- The shell helper (``scripts/_pods_conf_path.sh``) matches the Python
  resolver: LIVE when present, seed when it's a fresh clone.
- The invariant greps continue to catch shell writers and stray literal
  paths under scripts/*.py (mirrored in
  ``test_pod_config_atomic_and_guard.py``; the split just keeps the shell
  + resolver tests together with the location-fixture setup).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_config  # noqa: E402

# ---------------------------------------------------------------------------
# Scratch git repo fixture
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


@pytest.fixture
def scratch_repo(tmp_path):
    """A minimal scratch git repo where we can point pod_config at the
    resolved LIVE path. Layout mirrors the real project:

        <tmp>/repo/
            scripts/pods.conf     # committed seed
            .git/                 # shared git dir; live file lands under .git/eps/
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    _git(repo.parent, "init", str(repo))
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "config", "commit.gpgsign", "false")

    seed = repo / "scripts" / "pods.conf"
    seed.write_text(
        "# Pod registry -- test fixture (seed)\n"
        "# Format: name  host  port  gpus  gpu_type  label\n"
        "pod-100  1.1.1.1  11111  1  H100  seed-100\n"
    )
    _git(repo, "add", "scripts/pods.conf")
    _git(
        repo,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-m",
        "seed pods.conf",
    )

    return repo


def _resolver_pointing_at(repo: Path, monkeypatch) -> Path:
    """Point pod_config's module-level constants at ``repo`` and clear the
    module-level PODS_CONF/PODS_CONF_LOCK monkeypatch bias so the resolver's
    live-vs-seed branch fires as it would in production.
    """
    # SCRIPT_DIR is used by _git_common_dir() as the ``git -C`` argument;
    # PODS_CONF must point at the seed path inside our scratch repo so
    # ``_resolve_live_pods_conf`` treats it as "PODS_CONF == PODS_CONF_SEED"
    # (the trigger for the live-vs-seed branch).
    seed = repo / "scripts" / "pods.conf"
    monkeypatch.setattr(pod_config, "SCRIPT_DIR", repo / "scripts")
    monkeypatch.setattr(pod_config, "PODS_CONF_SEED", seed)
    monkeypatch.setattr(pod_config, "PODS_CONF", seed)
    # The lock lives beside the seed (as in production) — same file each
    # migration attempts to acquire, regardless of which resolved path.
    monkeypatch.setattr(pod_config, "PODS_CONF_LOCK", repo / "scripts" / ".pods.conf.lock")
    return seed


# ---------------------------------------------------------------------------
# Resolver contract
# ---------------------------------------------------------------------------


def test_resolver_prefers_live_when_present(scratch_repo, monkeypatch):
    """When both the seed and the LIVE file exist, the resolver returns the
    LIVE path (under ``<git-common-dir>/eps/pods.conf``), not the seed."""
    _resolver_pointing_at(scratch_repo, monkeypatch)

    live_dir = scratch_repo / ".git" / "eps"
    live_dir.mkdir(parents=True, exist_ok=True)
    live = live_dir / "pods.conf"
    live.write_text("# already migrated\n")

    resolved = pod_config._resolve_live_pods_conf()
    assert resolved == live
    # Sanity: the LIVE path is OUTSIDE the working tree.
    assert str(resolved).find("/.git/") != -1


def test_resolver_migrates_on_first_use(scratch_repo, monkeypatch):
    """When only the seed exists, the resolver migrates it to the LIVE
    path (atomic tmp+os.replace) and leaves the seed untouched."""
    seed = _resolver_pointing_at(scratch_repo, monkeypatch)
    pre_seed_bytes = seed.read_bytes()

    resolved = pod_config._resolve_live_pods_conf()

    assert resolved.exists()
    assert resolved.name == "pods.conf"
    assert resolved != seed
    # The migration is a copy — the seed is preserved verbatim (byte-for-
    # byte identical to its pre-migration content), so a re-clone works.
    assert seed.read_bytes() == pre_seed_bytes
    # And the LIVE file carries the seed's rows.
    assert resolved.read_bytes() == pre_seed_bytes


def test_migration_is_locked(scratch_repo, monkeypatch):
    """Two threads calling the resolver concurrently on a fresh scratch
    must produce ONE coherent LIVE file (single migration) — both threads
    end up pointing at the same LIVE path.
    """
    _resolver_pointing_at(scratch_repo, monkeypatch)

    results: list[Path] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            results.append(pod_config._resolve_live_pods_conf())
        except BaseException as exc:  # pragma: no cover - only if migration broke
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    assert len(results) == 4
    # All four resolved paths are identical.
    assert len({str(p) for p in results}) == 1
    # And the LIVE file exists with the seed's content — proving one
    # migration ran (not four half-writes).
    live = results[0]
    assert live.exists()
    assert live.read_text().startswith("# Pod registry -- test fixture (seed)")


# ---------------------------------------------------------------------------
# The v3 headline test: LIVE file survives every destructive git op
# ---------------------------------------------------------------------------


def _live_after_migration(repo: Path, monkeypatch) -> Path:
    _resolver_pointing_at(repo, monkeypatch)
    live = pod_config._resolve_live_pods_conf()
    # Populate the LIVE file with a distinctive marker so a wipe-then-migrate
    # cycle is trivially detectable (the seed rewrites would OVERWRITE the
    # marker; the survives-git tests require the marker to be intact).
    live.write_text(
        "# LIVE pods.conf (task #821 relocation test)\npod-777  9.9.9.9  33333  1  H100  live-777\n"
    )
    return live


def test_live_file_survives_destructive_git_ops(scratch_repo, monkeypatch):
    """The v3 headline: after seed→live migration, run every destructive
    git op on the working tree and assert the LIVE file survives with its
    contents intact each time.
    """
    live = _live_after_migration(scratch_repo, monkeypatch)
    pre_bytes = live.read_bytes()

    for cmd in (
        ["git", "reset", "--hard"],
        ["git", "checkout", "--", "."],
        ["git", "restore", "--", "."],
        ["git", "clean", "-fd"],
        ["git", "clean", "-fdx"],
    ):
        subprocess.run(cmd, cwd=scratch_repo, check=True, capture_output=True)
        assert live.exists(), (
            f"LIVE pods.conf disappeared after {' '.join(cmd)} (v3 relocation contract broken)"
        )
        assert live.read_bytes() == pre_bytes, (
            f"LIVE pods.conf mutated after {' '.join(cmd)} (v3 relocation contract broken)"
        )


def test_live_path_is_not_under_working_tree(scratch_repo, monkeypatch):
    """Static check: the resolved LIVE path is NOT under the working
    tree. If a future refactor accidentally re-anchors it under
    ``scripts/`` this test breaks loud."""
    _resolver_pointing_at(scratch_repo, monkeypatch)
    live = pod_config._resolve_live_pods_conf()
    # The working tree is ``repo`` minus its .git/ subtree; a resolved path
    # under ``.git/`` is BY DEFINITION not in the working tree.
    rel = live.relative_to(scratch_repo)
    assert rel.parts[0] == ".git", (
        f"Live pods.conf must live under .git/, got {rel} — a working-tree "
        f"path is subject to git reset/clean and defeats the #821 fix."
    )


# ---------------------------------------------------------------------------
# Shell helper subprocess tests
# ---------------------------------------------------------------------------


def _copy_helper_into(repo: Path) -> Path:
    """Copy the real ``scripts/_pods_conf_path.sh`` into the scratch repo
    so subprocess tests hit the exact code the project uses (no divergence
    between test setup and production)."""
    src = REPO_ROOT / "scripts" / "_pods_conf_path.sh"
    dst = repo / "scripts" / "_pods_conf_path.sh"
    shutil.copy(src, dst)
    dst.chmod(0o755)
    return dst


def _run_shell_resolver(repo: Path) -> str:
    """Source _pods_conf_path.sh in a subshell inside ``repo/scripts`` and
    print $CONF. Returns the printed path."""
    helper = repo / "scripts" / "_pods_conf_path.sh"
    assert helper.exists()
    script = f'SCRIPT_DIR="{repo / "scripts"}" && . "{helper}" && printf "%s" "$CONF"'
    proc = subprocess.run(
        ["bash", "-c", script], cwd=repo, capture_output=True, text=True, check=True
    )
    return proc.stdout


def test_shell_helper_resolves_live_when_present(scratch_repo):
    """The shell helper mirrors the Python resolver: when the LIVE file
    exists it returns the LIVE path; otherwise it falls back to the seed."""
    _copy_helper_into(scratch_repo)
    live_dir = scratch_repo / ".git" / "eps"
    live_dir.mkdir(parents=True, exist_ok=True)
    live = live_dir / "pods.conf"
    live.write_text("# live present for shell test\n")

    out = _run_shell_resolver(scratch_repo)
    assert Path(out) == live


def test_shell_helper_falls_back_to_seed_on_fresh_clone(scratch_repo):
    """A fresh clone (no LIVE file yet, before the first Python writer has
    migrated) sees the tracked seed."""
    _copy_helper_into(scratch_repo)
    # No .git/eps/pods.conf — the fallback should fire.
    out = _run_shell_resolver(scratch_repo)
    assert Path(out) == scratch_repo / "scripts" / "pods.conf"


# ---------------------------------------------------------------------------
# Invariant greps (mirror the Python-side test file — kept here for
# subprocess-heavy suites that skip the sibling file's slower fixtures)
# ---------------------------------------------------------------------------


def test_no_shell_script_writes_pods_conf_from_untracked_suite():
    """Duplicate of ``test_pod_config_atomic_and_guard.py::test_no_shell_
    script_writes_pods_conf`` — kept co-located with the shell-helper
    subprocess tests so the invariant fails LOUD whenever a shell writer
    slips in, regardless of which test file the reviewer opened.
    """
    import re

    write_patterns = [
        re.compile(r">\s*[\"']?\$?\{?CONF\}?[\"']?"),
        re.compile(r">>\s*[\"']?\$?\{?CONF\}?[\"']?"),
        re.compile(r"tee\s+[\"']?\$?\{?CONF\}?[\"']?"),
        re.compile(r"sed\s+-i[^\n]*[\"']?\$?\{?CONF\}?[\"']?"),
        re.compile(r">\s*[\"']?[^\"']*/?pods\.conf[\"']?"),
        re.compile(r">>\s*[\"']?[^\"']*/?pods\.conf[\"']?"),
        re.compile(r"tee\s+[\"']?[^\"']*/?pods\.conf[\"']?"),
    ]
    scripts_dir = REPO_ROOT / "scripts"
    violations: list[str] = []
    for script in sorted(p for p in scripts_dir.rglob("*.sh") if p.is_file()):
        text = script.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for rx in write_patterns:
                if rx.search(line):
                    violations.append(f"{script.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
                    break
    assert not violations, "Shell scripts must not write pods.conf. Offenders:\n" + "\n".join(
        violations
    )
