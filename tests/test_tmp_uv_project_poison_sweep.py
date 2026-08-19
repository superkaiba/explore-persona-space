"""Tests for the #2377 stray top-level /tmp uv PROJECT-FILE poison arm
(``clean_experiment_downloads.sweep_tmp_uv_project_files`` + the
``vm_disk_guard`` unconditional ``uv_project`` guard leg).

HERMETIC BY CONSTRUCTION (the #2127/#911 pattern): every fixture lives under
pytest's ``tmp_path`` and is passed as an EXPLICIT ``tmp_root`` /
``main_repo`` — the real ``/tmp`` and the real repo odb are never read or
written, and NOTHING destructive ever targets a real path. The "main repo"
is a per-test ``git init`` fixture, so blob-identity proofs are against a
THROWAWAY odb. The sidecar is redirected to a per-test file (the shared
``append_disk_guard_event`` resolves ``disk_guard_sidecar_path()`` at call
time, so a module-level monkeypatch reaches every caller).

Loaded via importlib like ``tests/test_janitor_tmp_scratch_sweep.py``
(ced first — vm_disk_guard imports it by module name at load time).
"""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── fixtures / helpers ──────────────────────────────────────────────────────

PYPROJECT_CONTENT = '[project]\nname = "stray"\nversion = "0.0.1"\n'
UVLOCK_CONTENT = 'version = 1\nrequires-python = ">=3.11"\n'


def _aged_now(*paths: Path) -> float:
    """A sweep ``now`` deterministically PAST the freshness grace for every
    given fixture path: newest lstat mtime + 2x grace, so the computed age is
    exactly 2x grace regardless of the pytest collection-to-run gap. (The
    round-1/2 module-level ``time.time()`` capture broke mid-gate: a ~13.5-min
    import->run gap shrank every computed age below the 600s grace and routed
    all fire-branch cases to ``tmp-uvproj-recent-escalated``. No assertion in
    this file may depend on a wall-clock delta between import and run.)"""
    newest = max(os.lstat(p).st_mtime for p in paths)
    return newest + 2 * ced.UVPROJ_RECENT_GRACE_SECONDS


_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def _git(repo: Path, *args: str) -> None:
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_ENV},
    )
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"


@pytest.fixture(autouse=True)
def _clear_kill_switches(monkeypatch):
    """The sweep must run in these tests regardless of the invoking shell's
    environment (both kill-switch layers unset)."""
    monkeypatch.delenv(ced.SCRATCH_SWEEP_KILL_ENV, raising=False)
    monkeypatch.delenv(ced.NONCANONICAL_SWEEP_KILL_ENV, raising=False)


@pytest.fixture(autouse=True)
def sidecar(tmp_path, monkeypatch):
    """Redirect the shared disk-guard sidecar to a per-test file (never the
    real one) and return its path for row assertions."""
    dest = tmp_path / "sidecar.jsonl"
    monkeypatch.setattr(ced, "disk_guard_sidecar_path", lambda: dest)
    return dest


@pytest.fixture
def main_repo(tmp_path):
    """A throwaway 'main repo' with a COMMITTED pyproject.toml — the odb every
    blob-identity proof in this file verifies against."""
    repo = tmp_path / "mainrepo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "pyproject.toml").write_text(PYPROJECT_CONTENT)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo


@pytest.fixture
def tmp_root(tmp_path):
    root = tmp_path / "faketmp"
    root.mkdir()
    return root


def _sidecar_rows(sidecar: Path) -> list[dict]:
    if not sidecar.is_file():
        return []
    return [json.loads(ln) for ln in sidecar.read_text().splitlines() if ln.strip()]


def _sidecar_kinds(sidecar: Path) -> list[str]:
    return [row["kind"] for row in _sidecar_rows(sidecar)]


def _one_row(result) -> dict:
    assert len(result.rows) == 1, result.rows
    return result.rows[0]


# ─── test 1: fire branch ─────────────────────────────────────────────────────


def test_verified_poison_file_is_quarantined(tmp_root, main_repo, sidecar):
    """apply=True on a byte-identical committed pyproject.toml: the file is
    MOVED (same-fs rename) into the quarantine dir — never deleted — with a
    sidecar row naming the uv blast radius."""
    poison = tmp_root / "pyproject.toml"
    poison.write_text(PYPROJECT_CONTENT)
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-quarantined"
    assert not poison.exists(), "original poison path must be gone"
    qdirs = sorted(tmp_root.glob("eps-quarantine-uvproj-*"))
    assert len(qdirs) == 1
    moved = qdirs[0] / "pyproject.toml"
    assert moved.read_text() == PYPROJECT_CONTENT, "quarantine preserves the bytes (restore point)"
    assert row["quarantine_path"] == str(moved)
    assert row["evidence"].startswith("git-blob:")
    assert "uv project discovery" in row["reason"], "reason names the uv blast radius"
    assert "tmp-uvproj-quarantined" in _sidecar_kinds(sidecar)
    assert res.bytes_freed == 0  # a rename frees nothing


# ─── test 2: escalate branch ─────────────────────────────────────────────────


def test_unverified_content_is_escalated_and_untouched(tmp_root, main_repo, sidecar):
    """Content NOT in the main repo odb: no quarantine — HARD-ESCALATE only."""
    poison = tmp_root / "pyproject.toml"
    poison.write_text('[project]\nname = "never-committed-anywhere"\n')
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-unverified-escalated"
    assert poison.is_file(), "unverified file must be left untouched"
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))
    assert "uv project discovery" in row["reason"]
    assert "tmp-uvproj-unverified-escalated" in _sidecar_kinds(sidecar)


# ─── test 3: empty-file trap ─────────────────────────────────────────────────


def test_empty_file_is_never_evidence_licensed(tmp_root, main_repo):
    """An empty /tmp/uv.lock is NOT licensed (an empty blob exists in every
    repo — the known hermeticity trap): escalated, untouched."""
    poison = tmp_root / "uv.lock"
    poison.write_text("")
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-unverified-escalated"
    assert "empty" in row["reason"]
    assert poison.is_file()
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))


# ─── test 4: kill switches ───────────────────────────────────────────────────


@pytest.mark.parametrize("env_var", [ced.SCRATCH_SWEEP_KILL_ENV, ced.NONCANONICAL_SWEEP_KILL_ENV])
def test_kill_switches_disable_the_arm(tmp_root, main_repo, monkeypatch, env_var):
    """Both kill-switch layers (leg switch + family switch) disable the whole
    arm (acceptance criterion 3)."""
    poison = tmp_root / "pyproject.toml"
    poison.write_text(PYPROJECT_CONTENT)
    monkeypatch.setenv(env_var, "1")
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    assert res.rows == []
    assert poison.is_file()


# ─── test 5: hermetic default ────────────────────────────────────────────────


def test_hermetic_default_is_a_no_op(main_repo, tmp_root):
    """tmp_root=None / main_repo=None => no-op empty result (library/test
    callers stay hermetic by construction)."""
    (tmp_root / "pyproject.toml").write_text(PYPROJECT_CONTENT)
    res = ced.sweep_tmp_uv_project_files(None, apply=True, main_repo=main_repo)
    assert res.rows == [] and res.total_discovered_bytes == 0
    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=True, main_repo=None)
    assert res.rows == [] and res.total_discovered_bytes == 0
    assert (tmp_root / "pyproject.toml").is_file()


# ─── test 6: report mode ─────────────────────────────────────────────────────


def test_report_mode_would_quarantine_touches_nothing_but_lands_in_sidecar(
    tmp_root, main_repo, sidecar
):
    """apply=False: file untouched, no quarantine dir — but the row STILL
    lands DURABLY in the sidecar (plan v3 §1: EVERY row, would-quarantine
    included — the round-2 ``uvproj-report-sidecar-missing`` fix), stamped
    with the observing mode."""
    poison = tmp_root / "pyproject.toml"
    poison.write_text(PYPROJECT_CONTENT)
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=False, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-would-quarantine"
    assert poison.is_file()
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))
    rows = _sidecar_rows(sidecar)
    assert [r["kind"] for r in rows] == ["tmp-uvproj-would-quarantine"]
    assert rows[0]["apply"] is False, "the sidecar event records which mode observed the row"


# ─── test 7: symlink ─────────────────────────────────────────────────────────


def test_symlink_is_nonregular_escalated_and_never_followed(tmp_root, main_repo, tmp_path):
    target = tmp_path / "real-pyproject.toml"
    target.write_text(PYPROJECT_CONTENT)
    link = tmp_root / "pyproject.toml"
    link.symlink_to(target)
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(link)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-nonregular-escalated"
    assert link.is_symlink(), "symlink left in place"
    assert target.read_text() == PYPROJECT_CONTENT, "symlink target untouched"


# ─── test 8: freshness grace ─────────────────────────────────────────────────


def test_freshness_grace_boundary_from_both_sides(tmp_root, main_repo):
    """A VERIFIED file is kept while younger than the grace and acted on once
    quiescent — pinned deterministically on BOTH sides of the strict ``<``
    boundary by giving the fixture an INTEGER mtime via ``os.utime`` (integer
    epoch floats add exactly, so ``age == grace`` is exact at the boundary)
    and deriving ``now`` from that mtime, never from wall clock."""
    poison = tmp_root / "pyproject.toml"
    poison.write_text(PYPROJECT_CONTENT)
    base = 1_700_000_000.0  # integer-valued epoch => exact float arithmetic
    os.utime(poison, (base, base))
    grace = ced.UVPROJ_RECENT_GRACE_SECONDS
    # Recent side: age = grace - 1 < grace => KEPT this pass (recency is only
    # ever a KEEP signal).
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=base + grace - 1.0
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-recent-escalated"
    assert row["evidence"].startswith("git-blob:"), "recent row is already VERIFIED by gate order"
    assert poison.is_file()
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))
    # Quiescent side: age == grace exactly — NOT ``< grace`` (strict), so the
    # recency gate passes and the verified file is quarantined.
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=base + grace
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-quarantined"
    assert not poison.exists()
    assert len(list(tmp_root.glob("eps-quarantine-uvproj-*"))) == 1


# ─── round-2 gate tests (uvproj-gate-tests-incomplete) ───────────────────────


def _write_aged_poison(tmp_root: Path) -> Path:
    """A committed-content pyproject.toml at tmp_root (verified + aged under a
    per-test ``_aged_now(poison)`` ``now`` — the fire-branch shape every gate
    test below starts from)."""
    poison = tmp_root / "pyproject.toml"
    poison.write_text(PYPROJECT_CONTENT)
    return poison


def test_predictable_destination_symlink_attack_cannot_redirect_or_replace(
    tmp_root, main_repo, tmp_path
):
    """BLOCKER regression (uvproj-quarantine-destination-unsafe): pre-create
    the round-1 PREDICTABLE quarantine path as a symlink to an OUTSIDE dir
    holding a victim file. Round 1's ``mkdir(mode=0o700, exist_ok=True)``
    silently adopted the symlink and ``os.rename`` then escaped tmp_root AND
    replaced the victim. Round 2's mkdtemp dir is atomically fresh +
    unpredictable: nothing lands outside tmp_root, no existing entry is
    replaced, and the realized quarantine parent is a REAL private 0700 dir
    owned by us, directly under tmp_root."""
    poison = _write_aged_poison(tmp_root)
    aged_now = _aged_now(poison)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "pyproject.toml"
    victim_bytes = "VICTIM — must never be replaced\n"
    victim.write_text(victim_bytes)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(aged_now))
    trap = tmp_root / f"eps-quarantine-uvproj-{ts}"
    trap.symlink_to(outside, target_is_directory=True)

    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=True, main_repo=main_repo, now=aged_now)

    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-quarantined"
    assert victim.read_text() == victim_bytes, "no existing entry may be replaced"
    assert sorted(p.name for p in outside.iterdir()) == ["pyproject.toml"], (
        "nothing may land outside tmp_root"
    )
    dest = Path(row["quarantine_path"])
    assert dest.read_text() == PYPROJECT_CONTENT
    qdir = dest.parent
    assert qdir.parent == tmp_root, "quarantine dir is a DIRECT child of tmp_root"
    assert qdir != trap, "the pre-created trap path was never adopted"
    st = os.lstat(qdir)
    assert stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode), "real dir, not a symlink"
    assert st.st_uid == os.getuid()
    assert stat.S_IMODE(st.st_mode) == 0o700
    assert not poison.exists()


def test_same_size_same_mtime_swap_is_never_evidence_licensed(
    tmp_root, main_repo, sidecar, monkeypatch
):
    """CONCERN regression (uvproj-evidence-toctou): swap the file for a
    SAME-SIZE, SAME-MTIME, different-inode file between the evidence read and
    the pre-rename re-stat (hook: the git odb probe, which runs exactly in
    that window). Round 1's size+mtime re-stat passed the swap and renamed
    unverified bytes under stale evidence; round 2's dev/ino bind to the
    still-open hashed fd aborts to KEEP."""
    poison = _write_aged_poison(tmp_root)
    attacker_bytes = "#" * len(PYPROJECT_CONTENT)  # same size, never committed anywhere
    real_probe = ced._git_first_missing_blob

    def swapping_probe(main_repo_arg, shas):
        orig = os.lstat(poison)
        sibling = tmp_root / ".swap-sibling"
        sibling.write_text(attacker_bytes)
        os.replace(sibling, poison)  # NEW inode at the same pathname
        os.utime(poison, ns=(orig.st_atime_ns, orig.st_mtime_ns))  # preserve mtime exactly
        swapped = os.lstat(poison)
        assert (swapped.st_size, swapped.st_mtime) == (orig.st_size, orig.st_mtime)
        assert swapped.st_ino != orig.st_ino
        return real_probe(main_repo_arg, shas)

    monkeypatch.setattr(ced, "_git_first_missing_blob", swapping_probe)
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-reap-aborted-recency"
    assert "inode" in row["reason"]
    assert poison.read_text() == attacker_bytes, "swapped bytes stay in place (KEEP)"
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*")), (
        "never quarantined under stale evidence"
    )
    assert "tmp-uvproj-quarantined" not in _sidecar_kinds(sidecar)


def test_pre_rename_content_change_aborts_to_keep(tmp_root, main_repo, monkeypatch):
    """SAME-inode mutation between the evidence read and the rename (size
    grows) => the pre-rename re-stat aborts to KEEP (gate 7's size+mtime
    fresh-recheck arm, distinct from the dev/ino swap arm above)."""
    poison = _write_aged_poison(tmp_root)
    real_probe = ced._git_first_missing_blob

    def growing_probe(main_repo_arg, shas):
        with open(poison, "a") as fh:
            fh.write("# appended after the evidence read\n")
        return real_probe(main_repo_arg, shas)

    monkeypatch.setattr(ced, "_git_first_missing_blob", growing_probe)
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-reap-aborted-recency"
    assert "changed" in row["reason"]
    assert poison.is_file()
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))


def test_live_process_holding_the_file_is_kept(tmp_root, main_repo, sidecar, monkeypatch):
    """An open handle on the candidate (probe hit) => KEEP, never quarantined."""
    poison = _write_aged_poison(tmp_root)
    monkeypatch.setattr(
        ced, "_scratch_live_process_hit", lambda cand, *, exact=False: "pid 4242 (uv)"
    )
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-live-process-kept"
    assert "pid 4242" in row["reason"]
    assert poison.is_file()
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))
    assert "tmp-uvproj-live-process-kept" in _sidecar_kinds(sidecar)


def test_foreign_owner_is_escalated_and_untouched(tmp_root, main_repo, sidecar, monkeypatch):
    """A foreign-uid candidate => KEEP + escalate (sticky-bit /tmp forbids
    renaming another uid's file anyway)."""
    poison = _write_aged_poison(tmp_root)
    monkeypatch.setattr(ced, "_tmp_entry_owned", lambda path: False)
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-foreign-owner-escalated"
    assert poison.is_file()
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))
    assert "tmp-uvproj-foreign-owner-escalated" in _sidecar_kinds(sidecar)


def test_quarantine_dir_setup_failure_keeps_and_escalates(
    tmp_root, main_repo, sidecar, monkeypatch
):
    """ANY dir-setup failure (mkdtemp refusal, verification surprise) => KEEP
    + ``tmp-uvproj-quarantine-failed`` — never a fallback delete. (The
    helper's REAL body runs in the fire-branch + symlink-attack tests; this
    test stubs it only to force the failure arm.)"""
    poison = _write_aged_poison(tmp_root)

    def boom(tmp_root_arg, now_arg):
        raise OSError("mkdtemp refused")

    monkeypatch.setattr(ced, "_uvproj_quarantine_dir", boom)
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-quarantine-failed"
    assert poison.is_file() and poison.read_text() == PYPROJECT_CONTENT
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))
    assert "tmp-uvproj-quarantine-failed" in _sidecar_kinds(sidecar)


def test_quarantine_rename_failure_keeps_and_escalates(tmp_root, main_repo, sidecar, monkeypatch):
    """Rename failure AFTER a healthy dir setup => KEEP + escalate; the empty
    fresh dir may remain but nothing is ever moved or deleted."""
    poison = _write_aged_poison(tmp_root)
    real_rename = os.rename

    def refusing_rename(src, dst, **kwargs):
        if Path(src) == poison:
            raise OSError("rename refused")
        return real_rename(src, dst, **kwargs)

    monkeypatch.setattr(ced.os, "rename", refusing_rename)
    res = ced.sweep_tmp_uv_project_files(
        tmp_root, apply=True, main_repo=main_repo, now=_aged_now(poison)
    )
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-quarantine-failed"
    assert poison.is_file() and poison.read_text() == PYPROJECT_CONTENT
    qdirs = list(tmp_root.glob("eps-quarantine-uvproj-*"))
    assert all(not any(d.iterdir()) for d in qdirs), "nothing was moved into any quarantine dir"
    assert "tmp-uvproj-quarantine-failed" in _sidecar_kinds(sidecar)


# ─── test 9: guard threshold-independence ────────────────────────────────────


def test_guard_leg_runs_even_under_threshold(tmp_root, main_repo, monkeypatch):
    """``vm_disk_guard.run_guard`` populates ``res.uv_project`` on an
    UNDER-threshold pass (usage monkeypatched low) — pinning the
    unconditional-leg placement BEFORE the early return, outside the
    threshold-gated tier suite."""
    poison = tmp_root / "pyproject.toml"
    poison.write_text(PYPROJECT_CONTENT)
    monkeypatch.setattr(vdg, "disk_used_pct", lambda path="/": 10.0)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 500.0)
    res = vdg.run_guard(
        False,
        threshold=85.0,
        scratch_tmp_root=tmp_root,
        scratch_main_repo=main_repo,
        now=_aged_now(poison),
    )
    assert res.triggered is False
    assert res.tiers == [], "under threshold: the tier suite must not have run"
    assert res.uv_project is not None and not res.uv_project.skipped
    rows = res.uv_project.scratch_candidates
    assert [r["disposition"] for r in rows] == ["tmp-uvproj-would-quarantine"]
    # The leg rides the --json surface as the top-level uv_project field.
    payload = vdg._result_json(res)
    assert payload["uv_project"]["name"] == "tmp-uvproj"
    assert payload["uv_project"]["scratch_candidates"] == rows


# ─── supplementary pins (not in the plan's numbered 9, cheap + load-bearing) ─


def test_guard_leg_skips_cleanly_when_unarmed():
    """No scratch opt-ins (the data-disk pass shape) => skipped TierResult
    with a reason, never a crash."""
    tier = vdg.check_tmp_uv_project_files(False, tmp_root=None, main_repo=None)
    assert tier.skipped and "opt-in" in tier.skip_reason


def test_push_dedup_is_per_name_disposition(tmp_root, main_repo, monkeypatch, tmp_path):
    """Escalate-class rows push once per (name, disposition) episode within
    the 24h window; would-quarantine rows never push."""
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    pushes: list[str] = []
    monkeypatch.setattr(vdg, "_telegram_push", lambda msg, apply: pushes.append(msg) or True)
    poison = tmp_root / "pyproject.toml"
    poison.write_text('[project]\nname = "not-committed"\n')
    tier = vdg.check_tmp_uv_project_files(
        True, tmp_root=tmp_root, main_repo=main_repo, now=_aged_now(poison)
    )
    assert [r["disposition"] for r in tier.scratch_candidates] == [
        "tmp-uvproj-unverified-escalated"
    ]
    vdg._maybe_alert_uv_project(tier, True, no_push=False, now=1_000_000.0)
    assert len(pushes) == 1 and "uv project discovery" in pushes[0]
    # Second run inside the window: deduped, no second push.
    vdg._maybe_alert_uv_project(tier, True, no_push=False, now=1_000_100.0)
    assert len(pushes) == 1
    # Past the re-alert window: pushes again.
    vdg._maybe_alert_uv_project(
        tier, True, no_push=False, now=1_000_000.0 + vdg._UVPROJ_PUSH_REALERT_SECONDS + 1
    )
    assert len(pushes) == 2
    # would-quarantine rows never push.
    tier2 = vdg.TierResult(name="tmp-uvproj")
    tier2.scratch_candidates = [
        {"name": "uv.lock", "path": "/tmp/uv.lock", "disposition": "tmp-uvproj-would-quarantine"}
    ]
    vdg._maybe_alert_uv_project(tier2, True, no_push=False, now=2_000_000.0)
    assert len(pushes) == 2


def test_report_only_push_persists_no_state(tmp_root, main_repo, monkeypatch, tmp_path):
    """apply=False: _telegram_push demotes (returns False) and the dedup
    state file is never written."""
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    poison = tmp_root / "uv.toml"
    poison.write_text("[pip]\n")  # not committed => escalate-class
    tier = vdg.check_tmp_uv_project_files(
        False, tmp_root=tmp_root, main_repo=main_repo, now=_aged_now(poison)
    )
    vdg._maybe_alert_uv_project(tier, False, no_push=False, now=1_000_000.0)
    assert not vdg._uvproj_push_state_path().exists()
