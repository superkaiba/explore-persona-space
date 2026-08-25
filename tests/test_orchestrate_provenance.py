"""Tests for `explore_persona_space.orchestrate.provenance` (task #2065).

The helper is the canonical git-provenance capture point for reproducibility
metadata: consolidated from the three duplicate `_git_commit_hash()` helpers
previously in analysis/convexity_meta.py, analysis/paper_plots.py, and
artifacts/organisms.py, extended with a dirty-tree flag (incident #1482).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from explore_persona_space.orchestrate.provenance import (
    _MAX_DIRTY_PATHS,
    GitProvenance,
    as_metadata_dict,
    commit_string,
    git_provenance,
    validate_phase_identity,
)


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    (root / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "-C", str(root), "add", "seed.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "init"], check=True)


def test_git_provenance_clean_tree(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    prov = git_provenance(cwd=tmp_path)
    assert prov.dirty is False
    assert prov.dirty_paths == []
    assert prov.commit_sha != "unknown"
    assert len(prov.commit_sha) == 8  # short SHA


def test_git_provenance_dirty_tree_flags_modified_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "seed.txt").write_text("changed\n")
    prov = git_provenance(cwd=tmp_path)
    assert prov.dirty is True
    assert "seed.txt" in prov.dirty_paths


def test_git_provenance_non_git_tree_returns_none_dirty(tmp_path: Path) -> None:
    prov = git_provenance(cwd=tmp_path)
    assert prov.commit_sha == "unknown"
    assert prov.dirty is None
    assert prov.dirty_paths == []


def test_dirty_paths_cap_at_max_with_overflow_marker(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    n_files = _MAX_DIRTY_PATHS + 5
    for i in range(n_files):
        (tmp_path / f"f{i:03d}.txt").write_text(f"c{i}\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add"], check=True)
    for i in range(n_files):
        (tmp_path / f"f{i:03d}.txt").write_text(f"changed{i}\n")
    prov = git_provenance(cwd=tmp_path)
    assert prov.dirty is True
    assert len(prov.dirty_paths) == _MAX_DIRTY_PATHS + 1
    assert prov.dirty_paths[-1].startswith("... ")
    assert prov.dirty_paths[-1].endswith("more")


def test_commit_string_appends_dirty_suffix_when_dirty() -> None:
    clean = GitProvenance(commit_sha="abcd1234", dirty=False)
    dirty = GitProvenance(commit_sha="abcd1234", dirty=True, dirty_paths=["a.py"])
    unknown = GitProvenance(commit_sha="unknown", dirty=None)
    assert commit_string(clean) == "abcd1234"
    assert commit_string(dirty) == "abcd1234+dirty"
    assert commit_string(unknown) == "unknown"


def test_as_metadata_dict_omits_dirty_paths_when_not_dirty() -> None:
    clean = GitProvenance(commit_sha="abcd1234", dirty=False)
    unknown = GitProvenance(commit_sha="unknown", dirty=None)
    dirty = GitProvenance(commit_sha="abcd1234", dirty=True, dirty_paths=["a.py"])
    assert as_metadata_dict(clean) == {
        "git_commit": "abcd1234",
        "git_dirty": False,
        "git_argv0_state": None,
    }
    assert as_metadata_dict(unknown) == {
        "git_commit": "unknown",
        "git_dirty": None,
        "git_argv0_state": None,
    }
    assert as_metadata_dict(dirty) == {
        "git_commit": "abcd1234",
        "git_dirty": True,
        "git_dirty_paths": ["a.py"],
        "git_argv0_state": None,
    }


def test_backward_compat_git_commit_key_preserved() -> None:
    """Existing reproducibility-metadata readers key on `git_commit` — preserve it.

    The `git_commit` key is the field every downstream consumer already reads
    (`analysis/convexity_meta.py::reproducibility_metadata` output; `organisms.py`
    provenance dicts; `paper_plots.py`'s sidecar `commit` key rides alongside).
    A rename would break those readers silently — the helper must keep it.
    """
    prov = GitProvenance(commit_sha="deadbeef", dirty=False)
    md = as_metadata_dict(prov)
    assert "git_commit" in md
    assert md["git_commit"] == "deadbeef"


# ---------------------------------------------------------------------------
# argv[0] tracked-state probe (task #2175 — untracked producing scripts)
# ---------------------------------------------------------------------------


def test_untracked_argv0_flags_dirty_and_records_state(tmp_path: Path) -> None:
    """The #2175 required test: an UNTRACKED producing script (argv[0]) must
    carry the explicit untracked signal AND fold into the dirty flag, so a
    committed result JSON can never claim clean provenance from a commit that
    does not contain the code that produced it (the #2094 incident shape)."""
    _init_repo(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "foo.py"
    script.write_text("print('produced')\n")
    prov = git_provenance(cwd=tmp_path, argv0=str(script))
    assert prov.argv0_state == "untracked"
    assert prov.argv0_path == "scripts/foo.py"
    assert prov.dirty is True
    assert "scripts/foo.py" in prov.dirty_paths
    assert commit_string(prov).endswith("+dirty")
    md = as_metadata_dict(prov)
    assert md["git_dirty"] is True
    assert md["git_argv0_state"] == "untracked"
    assert md["git_argv0_path"] == "scripts/foo.py"


def test_untracked_argv0_with_glob_metachars_not_misread_as_tracked_sibling(
    tmp_path: Path,
) -> None:
    """#2175 r2 BLOCKER regression (argv0-pathspec-not-literal): without
    `--literal-pathspecs`, the bracketed untracked argv0 `scripts/foo[1].py`
    is a git PATTERN matching the tracked sibling `scripts/foo1.py`, so the
    ls-files probe reads rc=0 and the untracked producing script stamps a
    falsely clean "tracked" provenance. The fix must classify it untracked,
    fold it into dirty, and report the LITERAL bracketed path (porcelain does
    not quote ASCII bracket names; verified live against git 2.34)."""
    _init_repo(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    tracked_sibling = scripts / "foo1.py"
    tracked_sibling.write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "scripts/foo1.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "sibling"], check=True)
    bracketed = scripts / "foo[1].py"
    bracketed.write_text("y = 2\n")
    prov = git_provenance(cwd=tmp_path, argv0=str(bracketed))
    assert prov.argv0_state == "untracked"
    assert prov.argv0_path == "scripts/foo[1].py"
    assert prov.dirty is True
    assert "scripts/foo[1].py" in prov.dirty_paths
    # The tracked sibling is clean and must NOT be dragged into the record.
    assert "scripts/foo1.py" not in prov.dirty_paths


def test_symlink_loop_argv0_degrades_to_none_without_crash(tmp_path: Path) -> None:
    """#2175 r2 CONCERN regression (argv0-resolve-symlink-loop-runtimeerror):
    `Path.resolve()` on a symlink loop raises RuntimeError on py3.11 (OSError
    on newer Pythons) — the never-crash contract must degrade to (None, None)
    instead of letting it escape `git_provenance`."""
    _init_repo(tmp_path)
    loop_a = tmp_path / "loop_a.py"
    loop_b = tmp_path / "loop_b.py"
    loop_a.symlink_to(loop_b)
    loop_b.symlink_to(loop_a)
    prov = git_provenance(cwd=tmp_path, argv0=str(loop_a))
    assert prov.argv0_state is None
    assert prov.argv0_path is None
    assert prov.dirty is False  # the tracked scan is untouched by the degrade


def test_tracked_clean_argv0_reads_tracked_and_clean(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    script = tmp_path / "runner.py"
    script.write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "runner.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "runner"], check=True)
    prov = git_provenance(cwd=tmp_path, argv0=str(script))
    assert prov.argv0_state == "tracked"
    assert prov.argv0_path == "runner.py"
    assert prov.dirty is False
    assert prov.dirty_paths == []
    md = as_metadata_dict(prov)
    assert md["git_argv0_state"] == "tracked"
    assert md["git_argv0_path"] == "runner.py"


def test_tracked_modified_argv0_reads_modified_without_double_count(tmp_path: Path) -> None:
    """A modified tracked argv0 is already in the tracked scan's dirty_paths —
    the argv0 fold must not add a second entry."""
    _init_repo(tmp_path)
    script = tmp_path / "runner.py"
    script.write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "runner.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "runner"], check=True)
    script.write_text("x = 2\n")
    prov = git_provenance(cwd=tmp_path, argv0=str(script))
    assert prov.argv0_state == "modified"
    assert prov.dirty is True
    assert prov.dirty_paths.count("runner.py") == 1


def test_gitignored_argv0_reads_none_not_untracked(tmp_path: Path) -> None:
    """The pytest/.venv false-positive guard: a gitignored argv0 yields EMPTY
    porcelain output under --untracked-files=all and must read as None (could
    not determine), never as untracked — or every pytest-invoked call would
    false-flag dirty."""
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "gitignore"], check=True)
    ignored = tmp_path / "ignored.py"
    ignored.write_text("y = 2\n")
    prov = git_provenance(cwd=tmp_path, argv0=str(ignored))
    assert prov.argv0_state is None
    assert prov.argv0_path is None
    assert prov.dirty is False
    md = as_metadata_dict(prov)
    assert md["git_argv0_state"] is None
    assert "git_argv0_path" not in md


def test_argv0_outside_repo_or_nonexistent_degrades_to_none(tmp_path: Path) -> None:
    """Pins the pytest-binary default: an argv0 outside the repo (rc=128) or a
    nonexistent argv0 (`python -c` → argv[0] == "-c") degrades to (None, None)
    and leaves the dirty verdict unchanged — every pre-#2175 caller keeps its
    semantics."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    outside = tmp_path / "elsewhere.py"
    outside.write_text("z = 3\n")
    prov = git_provenance(cwd=repo, argv0=str(outside))
    assert prov.argv0_state is None
    assert prov.argv0_path is None
    assert prov.dirty is False

    prov_c = git_provenance(cwd=repo, argv0="-c")
    assert prov_c.argv0_state is None
    assert prov_c.argv0_path is None
    assert prov_c.dirty is False

    prov_empty = git_provenance(cwd=repo, argv0="")
    assert prov_empty.argv0_state is None
    assert prov_empty.argv0_path is None


# ---------------------------------------------------------------------------
# Card phase identity + full-hex git_commit (task #2194)
# ---------------------------------------------------------------------------


def test_as_metadata_dict_emits_validated_phase() -> None:
    """The phase kwarg round-trip: the key lands as a SIBLING of `git_commit`
    (the exact dict level verify_report.py's card walk reads)."""
    prov = GitProvenance(commit_sha="abcd1234", dirty=False)
    md = as_metadata_dict(prov, phase="stage2-upload")
    assert md["phase"] == "stage2-upload"
    assert "git_commit" in md  # same flat dict == sibling placement


def test_as_metadata_dict_omits_phase_when_none() -> None:
    prov = GitProvenance(commit_sha="abcd1234", dirty=False)
    assert "phase" not in as_metadata_dict(prov)


@pytest.mark.parametrize("bad", ["done", "Stage2", "stage 2"])
def test_as_metadata_dict_rejects_invalid_phase(bad: str) -> None:
    """#2194 MF-B: write-time validation is wired INSIDE as_metadata_dict —
    deleting the validate_phase_identity call there turns this test red
    (covers one lifecycle word and one malformed slug per parameterization)."""
    prov = GitProvenance(commit_sha="abcd1234", dirty=False)
    with pytest.raises(ValueError):
        as_metadata_dict(prov, phase=bad)


def test_phase_identity_rejects_lifecycle_vocabulary() -> None:
    with pytest.raises(ValueError, match="LIFECYCLE"):
        validate_phase_identity("done")


@pytest.mark.parametrize("bad", ["Stage2", "stage 2", "", "-x", "x-"])
def test_phase_identity_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_phase_identity(bad)


def test_phase_identity_accepts_realized_slugs() -> None:
    for ok in ("stage2-upload", "grid-anchors", "upload_tbmp", "train", "eval", "bank"):
        assert validate_phase_identity(ok) == ok


def test_git_provenance_captures_full_sha(tmp_path: Path) -> None:
    """#2194 gate-usability fix: commit_sha_full is the 40-hex HEAD,
    commit_sha stays its first 8 chars, and as_metadata_dict emits the FULL
    form under git_commit (abbreviated SHAs are gate-excluded by
    verify_report.py's code-sha-cards)."""
    _init_repo(tmp_path)
    prov = git_provenance(cwd=tmp_path)
    assert prov.commit_sha_full is not None
    assert len(prov.commit_sha_full) == 40
    assert prov.commit_sha == prov.commit_sha_full[:8]
    md = as_metadata_dict(prov)
    assert md["git_commit"] == prov.commit_sha_full


def test_convexity_reproducibility_metadata_phase_param(tmp_path: Path, monkeypatch) -> None:
    """The DEP-1 library-writer wiring: reproducibility_metadata(phase=...)
    threads to as_metadata_dict; the default output carries NO phase key
    (byte-compat for every existing caller); an invalid phase raises through
    the same path."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    from explore_persona_space.analysis.convexity_meta import reproducibility_metadata

    md = reproducibility_metadata(phase="fits")
    assert md["phase"] == "fits"
    assert "git_commit" in md
    md_default = reproducibility_metadata()
    assert "phase" not in md_default
    with pytest.raises(ValueError):
        reproducibility_metadata(phase="done")


def test_convexity_extra_phase_precedence_over_kwarg(tmp_path: Path, monkeypatch) -> None:
    """#2194 round 2 (concern convexity-phase-precedence-unpinned): ``extra``
    merges AFTER the ``phase`` kwarg, so a legacy ``extra={"phase": ...}``
    caller keeps its documented precedence even when the new keyword is also
    supplied (the extra route bypasses write-time validation by construction;
    the consumer-side collision guard covers it)."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    from explore_persona_space.analysis.convexity_meta import reproducibility_metadata

    md = reproducibility_metadata(extra={"phase": "legacy-extra"}, phase="fits")
    assert md["phase"] == "legacy-extra"
