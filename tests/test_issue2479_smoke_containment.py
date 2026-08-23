"""Issue #2479 r5 — smoke-root containment (codex `smoke-root-production-poisoning`).

Mechanization (i): an EPM_I2479_SMOKE_ROOT override resolving inside the
repository tree (eval_results/ especially, or another issue's dirs) — or
outside every approved scratch area — is REFUSED by the p3-controls smoke
driver BEFORE any write; approved roots (/tmp, $TMPDIR, repo-local
data/issue_2479/smoke_*) pass. Hermetic: zero network, zero spend — the
refusal fires before the spend ack and before the first mkdir.

Mechanization (ii) — the smoke_synthesized rejection in require_pilot_pass —
lives in tests/test_issue2479_judge_pilots.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2479_judge_pilots as jp  # noqa: E402
import issue2479_p3_controls_smoke as sm  # noqa: E402


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fakerepo"
    (repo / "eval_results" / "issue_2479").mkdir(parents=True)
    return repo


# ---------------------------------------------------------------------------
# validate_smoke_root — the containment predicate
# ---------------------------------------------------------------------------
def test_repo_internal_eval_results_root_refused(tmp_path: Path) -> None:
    """The codex poisoning shape: pointing the smoke root at the canonical
    eval_results/issue_2479 tree would overwrite committed panel/manifest and
    park a synthesized pilot at the production path."""
    repo = _fake_repo(tmp_path)
    with pytest.raises(RuntimeError, match="repo-internal smoke root"):
        sm.validate_smoke_root(repo / "eval_results" / "issue_2479", repo_root=repo)


def test_other_issue_dirs_and_repo_root_refused(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    for bad in (
        repo,
        repo / "data" / "issue_1345" / "smoke_x",  # another issue's dirs
        repo / "data" / "issue_2479",  # the approved parent itself, no smoke_* component
        repo / "data" / "issue_2479" / "axis_items",  # non-smoke_* sibling
        repo / "figures" / "issue_2479",
    ):
        with pytest.raises(RuntimeError, match="smoke-root-production-poisoning"):
            sm.validate_smoke_root(bad, repo_root=repo)


def test_outside_approved_scratch_refused(tmp_path: Path, monkeypatch) -> None:
    """A root that is neither repo-local approved scratch nor under /tmp or
    $TMPDIR is refused (e.g. a home-dir path)."""
    repo = _fake_repo(tmp_path)
    monkeypatch.delenv("TMPDIR", raising=False)
    # A path guaranteed outside /tmp and the fake repo (never created —
    # validation raises before any write).
    outside = Path("/nonexistent-i2479-scratch/x")
    with pytest.raises(RuntimeError, match="not strictly under an approved scratch area"):
        sm.validate_smoke_root(outside, repo_root=repo)


def test_bare_tmp_root_itself_refused(monkeypatch) -> None:
    """SCRATCH is uploaded + quarantined wholesale — /tmp itself never allowed.

    The repo path is placed OUTSIDE /tmp (nonexistent is fine — validation
    only resolves paths, never stats them) so this keeps pinning the
    temp-root-itself refusal: a tmp_path fake repo lives UNDER /tmp, and the
    r6 reverse-ancestry check would correctly fire first on that shape."""
    monkeypatch.delenv("TMPDIR", raising=False)
    repo = Path("/nonexistent-i2479-repo/repo")
    with pytest.raises(RuntimeError, match="not strictly under an approved scratch area"):
        sm.validate_smoke_root(Path("/tmp"), repo_root=repo)


def test_approved_roots_pass(tmp_path: Path, monkeypatch) -> None:
    repo = _fake_repo(tmp_path)
    monkeypatch.delenv("TMPDIR", raising=False)
    assert (
        sm.validate_smoke_root(Path(sm.DEFAULT_SMOKE_ROOT), repo_root=repo)
        == Path(sm.DEFAULT_SMOKE_ROOT).resolve()
    )
    approved_local = repo / "data" / "issue_2479" / "smoke_p3controls"
    assert sm.validate_smoke_root(approved_local, repo_root=repo) == approved_local.resolve()
    # $TMPDIR is an approved area when set.
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tdir"))
    assert (
        sm.validate_smoke_root(tmp_path / "tdir" / "sub", repo_root=repo)
        == (tmp_path / "tdir" / "sub").resolve()
    )


def test_candidate_containing_repo_refused(tmp_path: Path, monkeypatch) -> None:
    """r5 reconciler `smoke-root-ancestor-escape`: a temp-area candidate that
    CONTAINS the repository (repo at <tmp>/parent/repo, candidate <tmp>/parent)
    must be refused BEFORE any write — build_fixtures would otherwise write
    into the broad ancestor and publication would upload the entire root,
    repository included."""
    # Pin the approved temp area to tmp_path so the ancestor candidate would
    # PASS the $TMPDIR allowlist branch absent the reverse-ancestry check.
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    parent = tmp_path / "parent"
    repo = parent / "repo"
    (repo / "eval_results" / "issue_2479").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="CONTAINS the"):
        sm.validate_smoke_root(parent, repo_root=repo)
    # Pre-write: validation raised without touching the candidate tree.
    assert sorted(p.name for p in parent.iterdir()) == ["repo"]
    # Positive case: a fresh SIBLING scratch dir under the same temp area
    # (no ancestry either way) still validates.
    sibling = tmp_path / "scratch_sibling"
    assert sm.validate_smoke_root(sibling, repo_root=repo) == sibling.resolve()


# ---------------------------------------------------------------------------
# main() refuses BEFORE any write (the driver-level mechanization)
# ---------------------------------------------------------------------------
def test_driver_refuses_repo_internal_root_before_creating_files(monkeypatch) -> None:
    """main() re-reads the env, containment-gates it FIRST, and raises before
    the spend ack and before any mkdir — the canonical eval_results override
    creates NOTHING new."""
    bad = REPO / "eval_results" / "issue_2479_smoke_containment_probe"
    assert not bad.exists()
    monkeypatch.setenv(sm.SMOKE_ROOT_ENV, str(bad))
    monkeypatch.delenv("EPM_I1345_JUDGE_SPEND_OK", raising=False)
    with pytest.raises(RuntimeError, match="repo-internal smoke root"):
        sm.main([])
    assert not bad.exists(), "the driver wrote under the refused root"


def test_driver_env_arms_allow_synthesized_for_subprocesses_only() -> None:
    """The smoke driver injects jp.ALLOW_SYNTHESIZED_ENV=1 into its explicit
    subprocess env dict (the ONLY production-code setter — pinned separately
    in test_issue2479_judge_pilots.py)."""
    src = (SCRIPTS / "issue2479_p3_controls_smoke.py").read_text()
    assert 'jp.ALLOW_SYNTHESIZED_ENV: "1"' in src
    assert jp.ALLOW_SYNTHESIZED_ENV == "EPM_I2479_ALLOW_SMOKE_SYNTHESIZED_PILOT"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
