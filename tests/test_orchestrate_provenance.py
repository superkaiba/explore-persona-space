"""Tests for `explore_persona_space.orchestrate.provenance` (task #2065).

The helper is the canonical git-provenance capture point for reproducibility
metadata: consolidated from the three duplicate `_git_commit_hash()` helpers
previously in analysis/convexity_meta.py, analysis/paper_plots.py, and
artifacts/organisms.py, extended with a dirty-tree flag (incident #1482).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from explore_persona_space.orchestrate.provenance import (
    _MAX_DIRTY_PATHS,
    GitProvenance,
    as_metadata_dict,
    commit_string,
    git_provenance,
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
    assert as_metadata_dict(clean) == {"git_commit": "abcd1234", "git_dirty": False}
    assert as_metadata_dict(unknown) == {"git_commit": "unknown", "git_dirty": None}
    assert as_metadata_dict(dirty) == {
        "git_commit": "abcd1234",
        "git_dirty": True,
        "git_dirty_paths": ["a.py"],
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
