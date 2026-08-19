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

# A `now` far enough past file creation that the freshness grace never trips
# (the files are written seconds before the sweep runs).
AGED_NOW = time.time() + 2 * ced.UVPROJ_RECENT_GRACE_SECONDS

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


def _sidecar_kinds(sidecar: Path) -> list[str]:
    if not sidecar.is_file():
        return []
    return [json.loads(ln)["kind"] for ln in sidecar.read_text().splitlines() if ln.strip()]


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
    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=True, main_repo=main_repo, now=AGED_NOW)
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
    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=True, main_repo=main_repo, now=AGED_NOW)
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
    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=True, main_repo=main_repo, now=AGED_NOW)
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
    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=True, main_repo=main_repo, now=AGED_NOW)
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


def test_report_mode_would_quarantine_touches_nothing(tmp_root, main_repo, sidecar):
    poison = tmp_root / "pyproject.toml"
    poison.write_text(PYPROJECT_CONTENT)
    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=False, main_repo=main_repo, now=AGED_NOW)
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-would-quarantine"
    assert poison.is_file()
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))
    # Report-only mode persists NOTHING to the sidecar (append demotes to stderr).
    assert _sidecar_kinds(sidecar) == []


# ─── test 7: symlink ─────────────────────────────────────────────────────────


def test_symlink_is_nonregular_escalated_and_never_followed(tmp_root, main_repo, tmp_path):
    target = tmp_path / "real-pyproject.toml"
    target.write_text(PYPROJECT_CONTENT)
    link = tmp_root / "pyproject.toml"
    link.symlink_to(target)
    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=True, main_repo=main_repo, now=AGED_NOW)
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-nonregular-escalated"
    assert link.is_symlink(), "symlink left in place"
    assert target.read_text() == PYPROJECT_CONTENT, "symlink target untouched"


# ─── test 8: freshness grace ─────────────────────────────────────────────────


def test_just_written_verified_file_is_kept_this_pass(tmp_root, main_repo):
    """A VERIFIED but just-written file is kept (recency is only ever a KEEP
    signal); the injected ``now`` makes the window deterministic."""
    poison = tmp_root / "pyproject.toml"
    poison.write_text(PYPROJECT_CONTENT)
    res = ced.sweep_tmp_uv_project_files(tmp_root, apply=True, main_repo=main_repo, now=time.time())
    row = _one_row(res)
    assert row["disposition"] == "tmp-uvproj-recent-escalated"
    assert row["evidence"].startswith("git-blob:"), "recent row is already VERIFIED by gate order"
    assert poison.is_file()
    assert not list(tmp_root.glob("eps-quarantine-uvproj-*"))


# ─── test 9: guard threshold-independence ────────────────────────────────────


def test_guard_leg_runs_even_under_threshold(tmp_root, main_repo, monkeypatch):
    """``vm_disk_guard.run_guard`` populates ``res.uv_project`` on an
    UNDER-threshold pass (usage monkeypatched low) — pinning the
    unconditional-leg placement BEFORE the early return, outside the
    threshold-gated tier suite."""
    (tmp_root / "pyproject.toml").write_text(PYPROJECT_CONTENT)
    monkeypatch.setattr(vdg, "disk_used_pct", lambda path="/": 10.0)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 500.0)
    res = vdg.run_guard(
        False,
        threshold=85.0,
        scratch_tmp_root=tmp_root,
        scratch_main_repo=main_repo,
        now=AGED_NOW,
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
    (tmp_root / "pyproject.toml").write_text('[project]\nname = "not-committed"\n')
    tier = vdg.check_tmp_uv_project_files(
        True, tmp_root=tmp_root, main_repo=main_repo, now=AGED_NOW
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
    (tmp_root / "uv.toml").write_text("[pip]\n")  # not committed => escalate-class
    tier = vdg.check_tmp_uv_project_files(
        False, tmp_root=tmp_root, main_repo=main_repo, now=AGED_NOW
    )
    vdg._maybe_alert_uv_project(tier, False, no_push=False, now=1_000_000.0)
    assert not vdg._uvproj_push_state_path().exists()
