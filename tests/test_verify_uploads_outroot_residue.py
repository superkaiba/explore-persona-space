"""Tests for the #2187 out-root TOP-LEVEL residue check.

Task #2162 lost three out-root TOP-LEVEL files in one run (each caught only
by a manual pre-teardown sweep): per-issue ``phase_upload`` implementations
glob their SUBDIRECTORIES and silently omit files written at the out-root
top level. ``check_outroot_residue`` makes the sweep mechanical: a per-file
NAME-SET diff of the out-root listing vs the union of HF prefixes +
ISSUE-SCOPED git trees + declared discards. These tests pin:

- residue -> FAIL naming every file + size; clean -> OK;
- the #2162 regression shape: equal counts with different names FAILs
  (a matching count is not a matching set);
- the v1->v2 defect regression pin: against the LIVE repo tree with
  ``--issue 999999``-equivalent args, a listing carrying the three real
  #2162 basenames FAILs naming ALL THREE. This assertion is FALSE under an
  unscoped whole-tree git arm (``pilot_gate_report.json`` and
  ``upload_done.json`` resolve at cross-issue paths at HEAD — the test
  verifies that precondition, so the pin stays discriminating) and TRUE
  under the issue-scoped arm;
- the issue-token path-component filter semantics (``.search`` over a
  ``(?:^|/)`` anchor + digit boundary);
- exemptions (dir parts / suffixes / caller globs), declared discards,
  the no-listing SKIP, the fail-loud ERROR on a listing failure, and the
  empty-prefix fail-toward-FAIL WARN wording.

Per the one-production-body-test rule (#906), the residue/clean/equal-count
tests execute the REAL check body against a REAL tmp out-root and the REAL
git subprocess arm; the only fake sits at the external HF network boundary
and is signature-conformant by construction (``create_autospec`` of the real
``list_repo_files_complete``). Same module-loading conventions as
tests/test_verify_uploads_card_fallback.py.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_uploads.py"
_spec = importlib.util.spec_from_file_location("verify_uploads_or", _SCRIPT)
verify_uploads = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_uploads_or"] = verify_uploads
_spec.loader.exec_module(verify_uploads)  # type: ignore[union-attr]

REPO_ROOT = Path(__file__).resolve().parents[1]

# The three real #2162 loss basenames (task #2187's motivating incident).
LOST_BASENAMES = ("pilot_gate_report.json", "stage2_results.json", "upload_done.json")


def _patch_hf(
    monkeypatch, mapping: dict[str, list[str]] | None = None, exc: Exception | None = None
):
    """Fake the HF listing boundary, signature-conformant by construction.

    ``create_autospec`` of the real ``list_repo_files_complete`` rejects any
    call whose shape drifts from the real signature (#906 rule); the fake's
    behavior mirrors the real endpoint — a prefix absent from ``mapping``
    raises ``EntryNotFoundError`` exactly as the tree endpoint 404s.
    """
    from explore_persona_space.orchestrate import hub

    fake = create_autospec(hub.list_repo_files_complete)
    if exc is not None:
        fake.side_effect = exc
    else:
        resolved = mapping or {}

        def _lookup(api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
            from huggingface_hub.utils import EntryNotFoundError

            if path_in_repo not in resolved:
                raise EntryNotFoundError(f"no tree at {path_in_repo}")
            return list(resolved[path_in_repo])

        fake.side_effect = _lookup
    monkeypatch.setattr(hub, "list_repo_files_complete", fake)
    return fake


# ---------------------------------------------------------------------------
# Core verdicts (real body: real tmp out-root + real git arm; HF faked)
# ---------------------------------------------------------------------------


def test_residue_fails_naming_file_and_size(tmp_path, monkeypatch):
    """A file with no permanent home is a FAIL row naming path + byte size."""
    stray = tmp_path / "stray_result.json"
    stray.write_text("x" * 123)
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(tmp_path), hf_prefixes=("issue999999_none",)
    )

    assert row["status"] == "FAIL"
    assert "stray_result.json" in row["detail"]
    assert "123 B" in row["detail"]
    assert "matching count is not a matching set" in row["detail"]


def test_clean_tree_passes_with_counts_as_context(tmp_path, monkeypatch):
    """Every out-root file resolving at an HF home -> OK; counts are context."""
    (tmp_path / "homed.json").write_text("{}")
    _patch_hf(monkeypatch, mapping={"issue999999_x": ["issue999999_x/sub/homed.json"]})

    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(tmp_path), hf_prefixes=("issue999999_x",)
    )

    assert row["status"] == "OK"
    assert "disk=1 matched=1" in row["detail"]


def test_equal_counts_different_names_fail(tmp_path, monkeypatch):
    """The #2162 shape: 1 disk file vs 1 uploaded file, DIFFERENT names —
    counts match, the set does not; the verdict is the name-set diff."""
    (tmp_path / "a.json").write_text("{}")
    _patch_hf(monkeypatch, mapping={"issue999999_x": ["issue999999_x/b.json"]})

    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(tmp_path), hf_prefixes=("issue999999_x",)
    )

    assert row["status"] == "FAIL"
    assert "a.json" in row["detail"]


# ---------------------------------------------------------------------------
# The v1->v2 defect regression pin (live repo tree)
# ---------------------------------------------------------------------------


def test_live_tree_cross_issue_collision_regression_pin(tmp_path, monkeypatch):
    """Against the LIVE tree with issue 999999, a listing carrying the three
    real #2162 basenames FAILs naming ALL THREE.

    Discriminating power: the unscoped HEAD tree DOES carry
    ``pilot_gate_report.json`` and ``upload_done.json`` under OTHER issues
    (verified below as a precondition — both are conventional filenames and
    ``eval_results/`` is never deleted), so a whole-tree basename arm would
    resolve those two as "homed" and this assertion would be FALSE. Only the
    issue-scoped git arm makes it hold. (No assertion on total tracked-file
    counts — HEAD moves under concurrent sessions; the collision EXISTENCE is
    the stable quantity.)
    """
    tree = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    ).stdout.splitlines()
    for colliding in ("pilot_gate_report.json", "upload_done.json"):
        cross_issue_hits = [
            p for p in tree if p.endswith(f"/{colliding}") and "issue_999999" not in p
        ]
        assert cross_issue_hits, (
            f"precondition lost: no tracked path with basename {colliding!r} — "
            "the regression pin would no longer discriminate scoped vs whole-tree"
        )

    listing = tmp_path / "outroot-listing.txt"
    listing.write_text("".join(f"/workspace/issue999999_out/{name}\n" for name in LOST_BASENAMES))
    _patch_hf(monkeypatch)  # every prefix absent, as for a never-uploaded run

    row = verify_uploads.check_outroot_residue(
        999999, outroot_listing=str(listing), hf_prefixes=("issue999999_none",)
    )

    assert row["status"] == "FAIL"
    for name in LOST_BASENAMES:
        assert name in row["detail"], (
            f"{name!r} must read as residue for issue 999999 — a whole-tree "
            "basename arm would false-PASS the cross-issue-colliding names"
        )


# ---------------------------------------------------------------------------
# Issue-token path-component filter semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "kept"),
    [
        ("eval_results/issue_2162/x.json", True),
        ("figures/issue_2162/a.png", True),
        ("scripts/issue2162_run.py", True),
        ("docs/methodology/issue_2162.md", True),
        ("eval_results/issue_1739/x.json", False),  # different issue
        ("eval_results/issue_21620/x.json", False),  # digit boundary
        ("myissue_2162/x.json", False),  # component must BEGIN with the token
    ],
)
def test_issue_token_filter_cases(path, kept):
    result = verify_uploads.filter_issue_scoped_git_paths([path], 2162)
    assert (result == [path]) is kept


# ---------------------------------------------------------------------------
# Exemptions, discards, SKIP, ERROR, empty-prefix WARN
# ---------------------------------------------------------------------------


def test_exemption_dir_suffix_and_glob_honored(tmp_path, monkeypatch):
    (tmp_path / "wandb").mkdir()
    (tmp_path / "wandb" / "run.json").write_text("{}")  # exempt dir part
    (tmp_path / "run.log").write_text("log")  # exempt suffix
    (tmp_path / "scratch.json").write_text("{}")  # caller glob
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        999999,
        outroot=str(tmp_path),
        hf_prefixes=("issue999999_none",),
        exempt_globs=("scratch.*",),
    )

    assert row["status"] == "OK"
    assert "disk=0 matched=0" in row["detail"]


def test_declared_discard_honored(tmp_path, monkeypatch):
    (tmp_path / "big_tensor_meta.json").write_text("{}")
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        999999,
        outroot=str(tmp_path),
        hf_prefixes=("issue999999_none",),
        discarded_names=("big_tensor_meta.json",),
    )

    assert row["status"] == "OK"


def test_no_listing_skips_and_names_the_flags():
    row = verify_uploads.check_outroot_residue(123)
    assert row["status"] == "SKIP"
    assert "--outroot-listing" in row["detail"]


def test_hf_listing_failure_is_error_not_ok(tmp_path, monkeypatch):
    """A non-not-found listing failure surfaces as ERROR (fail-loud — ERROR
    flips the overall verdict to FAIL), never as a silent OK."""
    (tmp_path / "anything.json").write_text("{}")
    _patch_hf(monkeypatch, exc=RuntimeError("boom: quota storm"))

    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(tmp_path), hf_prefixes=("issue999999_x",)
    )

    assert row["status"] == "ERROR"
    assert "boom" in row["detail"]


def test_empty_prefix_with_listing_warns_and_fails_toward_fail(tmp_path, monkeypatch):
    """No --hf-prefix while a listing IS supplied: the check still runs (HF
    set empty — HF-resident files read as residue) with a WARN-worded detail."""
    (tmp_path / "unhomed.json").write_text("{}")
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(999999, outroot=str(tmp_path))

    assert row["status"] == "FAIL"
    assert row["detail"].startswith("WARNING: no --hf-prefix supplied")
