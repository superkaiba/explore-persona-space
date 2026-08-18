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
  empty-prefix fail-toward-FAIL WARN wording;
- missing-input handling (round 2): a nonexistent --outroot-listing file or
  --outroot dir is ERROR (never a traceback, never a silent disk=0 OK) while
  an existing-but-empty dir stays OK; and exemption-mode parity: listing-mode
  exemptions match the root-relative path (common prefix stripped), so an
  exempt-named ANCESTOR dir never wholesale-exempts the listing while an
  exempt-named dir INSIDE the out-root is still exempted in both modes;
- cross-leg content disambiguation (#2359, the #2333 false-OK): a basename
  matched ONLY by the issue-scoped git arm is byte-checked against the
  committed candidates when locally readable (same bytes cover, different
  bytes FAIL naming both paths, size mismatch short-circuits the hash) and
  degrades to WARN `outroot-residue-basename-git-only` on a pod-side listing
  row with no local bytes (residue FAIL dominates); HF-arm coverage is
  checked FIRST, so a both-arms match never spuriously WARNs.

Per the one-production-body-test rule (#906), the residue/clean/equal-count
tests execute the REAL check body against a REAL tmp out-root and the REAL
git subprocess arm; the only fake sits at the external HF network boundary
and is signature-conformant by construction (``create_autospec`` of the real
``list_repo_files_complete``). Same module-loading conventions as
tests/test_verify_uploads_card_fallback.py.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
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


# ---------------------------------------------------------------------------
# Missing-input handling + exemption-mode parity (#2187 round 2)
# ---------------------------------------------------------------------------


def test_missing_listing_file_is_error_not_traceback():
    """A nonexistent --outroot-listing path is a legible ERROR row (which
    flips the overall verdict to FAIL), never an uncaught traceback."""
    row = verify_uploads.check_outroot_residue(1, outroot_listing="/nope/never-captured.txt")

    assert row["status"] == "ERROR"
    assert "/nope/never-captured.txt" in row["detail"]


def test_missing_outroot_dir_is_error_never_silent_ok(tmp_path):
    """The load-bearing sibling: a typo'd/nonexistent --outroot directory must
    NOT read as disk=0 matched=0 OK — that silent default would green-light
    teardown on a run whose out-root was never inspected (the exact false-PASS
    class this check exists to close)."""
    row = verify_uploads.check_outroot_residue(1, outroot=str(tmp_path / "typo_dir"))

    assert row["status"] == "ERROR"
    assert "typo_dir" in row["detail"]


def test_existing_empty_outroot_dir_stays_ok(tmp_path, monkeypatch):
    """'Path absent' and 'path present but empty' must not collapse: a
    directory that exists and is genuinely empty is a legitimate disk=0 OK."""
    empty = tmp_path / "issue999999_out"
    empty.mkdir()
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        999999, outroot=str(empty), hf_prefixes=("issue999999_none",)
    )

    assert row["status"] == "OK"
    assert "disk=0 matched=0" in row["detail"]


def test_exempt_named_ancestor_dir_does_not_exempt_listing(tmp_path, monkeypatch):
    """An out-root nested under a directory named ``logs`` must not have its
    whole listing silently exempted: exemptions match the root-relative path
    (common prefix stripped), not the full pod path's own components."""
    listing = tmp_path / "listing.txt"
    listing.write_text(
        "/workspace/logs/issue999999_out/a.json\n/workspace/logs/issue999999_out/b.json\n"
    )
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        999999, outroot_listing=str(listing), hf_prefixes=("issue999999_none",)
    )

    assert row["status"] == "FAIL"
    assert "a.json" in row["detail"]
    assert "b.json" in row["detail"]


def test_nested_exempt_dir_inside_outroot_still_exempted_in_listing_mode(tmp_path, monkeypatch):
    """Parity positive direction: an exempt-named dir INSIDE the out-root is
    still exempted in listing mode, exactly as the local-walk mode exempts it."""
    listing = tmp_path / "listing.txt"
    listing.write_text(
        "/workspace/issue999999_out/keep.json\n/workspace/issue999999_out/wandb/run_state.json\n"
    )
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        999999,
        outroot_listing=str(listing),
        hf_prefixes=("issue999999_none",),
        discarded_names=("keep.json",),
    )

    assert row["status"] == "OK"
    assert "disk=1 matched=1" in row["detail"]


def test_mixed_absolute_relative_listing_is_error(tmp_path, monkeypatch):
    """A listing mixing absolute and relative paths has no well-defined common
    prefix — that is malformed input and surfaces as ERROR, never a guess."""
    listing = tmp_path / "listing.txt"
    listing.write_text("/workspace/issue999999_out/a.json\nrelative/b.json\n")
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        999999, outroot_listing=str(listing), hf_prefixes=("issue999999_none",)
    )

    assert row["status"] == "ERROR"
    assert "malformed" in row["detail"]


# ---------------------------------------------------------------------------
# Cross-leg content disambiguation (#2359 — the #2333 false-OK)
#
# Hermetic temp git repo: leg-A's artifacts are COMMITTED fixtures and the
# repo-root seam (verify_uploads._verifier_repo_root) is monkeypatched, so
# the REAL subprocess git arm (ls-tree -r -l parse, OID/size extraction) runs
# against a stable tree instead of the mid-flight live checkout. Per the
# one-production-body-test rule (#906), tests 1-2 execute the REAL
# _git_blob_sha1 body; only the size-shortcut test stubs it (as a tripwire),
# and only the HF network boundary is faked (autospec, as above).
# ---------------------------------------------------------------------------

# Committed leg-A bytes vs leg-B's same-LENGTH different bytes (the #2333
# shape at equal size, so the size first-pass cannot discriminate and the
# blob-sha1 comparison is what fires).
BYTES_A = b'{"leg": "q25", "sha256": "6f43c93d"}\n'
BYTES_B_SAME_LEN = b'{"leg": "q35", "sha256": "0a052e8b"}\n'
assert len(BYTES_A) == len(BYTES_B_SAME_LEN)

# Neutralize the VM's global/system git config (hooks, gpgsign, templates)
# so the fixture commits work in a bare CI-like env.
_GIT_HERMETIC_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


def _git(repo: Path, *args: str) -> None:
    """Run git in the hermetic fixture repo (identity via -c, fail-loud)."""
    subprocess.run(
        ["git", "-c", "user.name=eps-test", "-c", "user.email=eps-test@example.com", *args],
        cwd=repo,
        env=_GIT_HERMETIC_ENV,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def hermetic_repo(monkeypatch):
    """Temp git repo with leg-A committed at eval_results/issue_424242/q25/.

    tempfile.mkdtemp rather than tmp_path: concurrent pytest sessions on the
    shared VM prune each other's /tmp/pytest-of-* numbered roots, which can
    delete a live subprocess-heavy scratch repo mid-test. Yields the scratch
    ROOT (the git repo lives at <root>/repo; out-roots/listings go beside it
    so disk files never sit inside the fixture repo's working tree).
    """
    root = Path(tempfile.mkdtemp(prefix="eps-issue2359-residue-"))
    try:
        repo = root / "repo"
        leg_a = repo / "eval_results" / "issue_424242" / "q25"
        leg_a.mkdir(parents=True)
        (leg_a / "upload_done.json").write_bytes(BYTES_A)
        (leg_a / "other.json").write_bytes(b'{"other": true}\n')
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, "add", "eval_results")
        _git(repo, "commit", "-q", "-m", "leg-A artifacts")
        monkeypatch.setattr(verify_uploads, "_verifier_repo_root", lambda: repo)
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cross_leg_same_basename_different_bytes_fails(hermetic_repo, monkeypatch):
    """The #2333 regression shape: leg-B's local upload_done.json (same
    basename, same byte SIZE, different bytes) must NOT be covered by
    leg-A's committed copy — FAIL naming the disk path AND the committed
    candidate path."""
    outroot = hermetic_repo / "issue424242_out"
    (outroot / "q35" / "manifests").mkdir(parents=True)
    disk_file = outroot / "q35" / "manifests" / "upload_done.json"
    disk_file.write_bytes(BYTES_B_SAME_LEN)
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        424242, outroot=str(outroot), hf_prefixes=("issue424242_none",)
    )

    assert row["status"] == "FAIL"
    assert "same basename, different content" in row["detail"]
    assert str(disk_file) in row["detail"]
    assert "eval_results/issue_424242/q25/upload_done.json" in row["detail"]


def test_same_bytes_committed_candidate_covers(hermetic_repo, monkeypatch):
    """No false-FAIL on true persistence: disk bytes == the committed
    candidate's blob -> covered OK, counted as content-verified."""
    outroot = hermetic_repo / "issue424242_out"
    outroot.mkdir()
    (outroot / "upload_done.json").write_bytes(BYTES_A)
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        424242, outroot=str(outroot), hf_prefixes=("issue424242_none",)
    )

    assert row["status"] == "OK"
    assert "disk=1 matched=1" in row["detail"]
    assert "content-verified=1" in row["detail"]


def test_listing_mode_git_only_match_warns_with_token(hermetic_repo, monkeypatch):
    """Fail-loud pin (acceptance criterion 3): a pod-side listing row whose
    basename matches ONLY the issue-scoped git arm has no local bytes to
    compare — status WARN (never a silent OK) carrying the literal token
    and naming both the disk path and the committed candidate path."""
    listing = hermetic_repo / "listing.txt"
    listing.write_text("/workspace/issue424242_out/q35/manifests/upload_done.json\n")
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        424242, outroot_listing=str(listing), hf_prefixes=("issue424242_none",)
    )

    assert row["status"] == "WARN"
    assert "outroot-residue-basename-git-only" in row["detail"]
    assert "/workspace/issue424242_out/q35/manifests/upload_done.json" in row["detail"]
    assert "eval_results/issue_424242/q25/upload_done.json" in row["detail"]


def test_size_mismatch_short_circuits_without_hash(hermetic_repo, monkeypatch):
    """A different byte SIZE is residue via the cheap size first-pass; the
    blob hasher is monkeypatched to a tripwire so the test pins that NO
    hashing happens on the short-circuit branch (real-body coverage of
    _git_blob_sha1 lives in the two tests above)."""
    outroot = hermetic_repo / "issue424242_out"
    outroot.mkdir()
    (outroot / "upload_done.json").write_bytes(b'{"much": "longer body than leg A ever wrote"}\n')
    _patch_hf(monkeypatch)

    def _tripwire(path):
        raise AssertionError("size first-pass must short-circuit hashing")

    monkeypatch.setattr(verify_uploads, "_git_blob_sha1", _tripwire)

    row = verify_uploads.check_outroot_residue(
        424242, outroot=str(outroot), hf_prefixes=("issue424242_none",)
    )

    assert row["status"] == "FAIL"
    assert "same basename, different content" in row["detail"]
    assert "eval_results/issue_424242/q25/upload_done.json" in row["detail"]


def test_residue_dominates_git_only_warn(hermetic_repo, monkeypatch):
    """FAIL > WARN: an outright-stray file keeps the FAIL verdict while the
    unverifiable git-only match is still named in the same detail."""
    listing = hermetic_repo / "listing.txt"
    listing.write_text(
        "/workspace/issue424242_out/stray_output.json\n"
        "/workspace/issue424242_out/q35/manifests/upload_done.json\n"
    )
    _patch_hf(monkeypatch)

    row = verify_uploads.check_outroot_residue(
        424242, outroot_listing=str(listing), hf_prefixes=("issue424242_none",)
    )

    assert row["status"] == "FAIL"
    assert "stray_output.json" in row["detail"]
    assert "upload_done.json" in row["detail"]
    assert "byte-check in the exploratory pass" in row["detail"]


def test_hf_arm_match_takes_precedence_no_warn(hermetic_repo, monkeypatch):
    """Acceptance criterion 4: a basename resolving at an HF prefix is
    covered BEFORE any git-arm content logic — a both-arms match in listing
    mode is OK with NO WARN token (a git-first implementation would
    spuriously WARN every healthy pod-side run whose files are both
    uploaded and committed)."""
    listing = hermetic_repo / "listing.txt"
    listing.write_text("/workspace/issue424242_out/q35/manifests/upload_done.json\n")
    _patch_hf(
        monkeypatch,
        mapping={"issue424242_x": ["issue424242_x/q35/manifests/upload_done.json"]},
    )

    row = verify_uploads.check_outroot_residue(
        424242, outroot_listing=str(listing), hf_prefixes=("issue424242_x",)
    )

    assert row["status"] == "OK"
    assert "outroot-residue-basename-git-only" not in row["detail"]
    assert "disk=1 matched=1" in row["detail"]


# ---------------------------------------------------------------------------
# Round-2 concern `git-ls-tree-parse-fail-loud` (#2359 r2): a STRUCTURALLY
# malformed `git ls-tree -r -l` row raises RuntimeError (fail-loud — the
# caller maps it to an ERROR row, flipping the overall verdict to FAIL),
# while a successfully-parsed NON-BLOB row (tree/commit entry) is a
# legitimate non-candidate and is skipped silently. A real repo cannot emit
# a malformed row, so that test fakes the subprocess boundary
# (create_autospec of the real subprocess.run, per #906); the non-blob test
# uses real repo state (a 160000 gitlink) through the real git arm.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_row, expected_fragment",
    [
        ("100644 blob abcdef0 no-tab-separator-row", "no tab separator"),
        ("100644 blob abcdef0\tpath/only_three_fields.json", "expected 4 metadata fields"),
    ],
)
def test_malformed_ls_tree_row_raises_runtime_error(monkeypatch, bad_row, expected_fragment):
    """A row that cannot be split into `<mode> <type> <oid> <size>\\t<path>`
    is a fail-loud RuntimeError naming the ref AND the offending row — never
    a silent skip the verdict then builds on (round-1 shipped a silent
    `continue` here while the docstring claimed fail-loud)."""
    healthy = "100644 blob " + "a" * 40 + "      42\tissue_424242/healthy.json"

    def _dispatch(cmd, *args, **kwargs):
        if cmd[1] == "rev-parse":  # _issue_branch_ref probe: no issue branch
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        assert cmd[:3] == ["git", "ls-tree", "-r"], cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{healthy}\n{bad_row}\n", stderr="")

    fake_run = create_autospec(subprocess.run, side_effect=_dispatch)
    monkeypatch.setattr(verify_uploads.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        verify_uploads._git_tree_candidates_for_issue(424242)

    msg = str(excinfo.value)
    assert expected_fragment in msg
    assert "HEAD" in msg  # names the ref
    assert repr(bad_row) in msg  # names the offending row verbatim


def test_parsed_non_blob_row_is_skipped_not_raised(hermetic_repo):
    """A successfully-parsed non-blob row — a 160000 gitlink, which
    `git ls-tree -r -l` renders as `160000 commit <oid> -\\t<path>` — is a
    legitimate non-candidate: skipped without raising, while sibling blob
    rows still parse into candidates. Real repo state, real git arm."""
    repo = hermetic_repo / "repo"
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{'1' * 40},eval_results/issue_424242/q25/vendored_pin",
    )
    _git(repo, "commit", "-q", "-m", "add gitlink (parsed non-blob row)")

    candidates = verify_uploads._git_tree_candidates_for_issue(424242)

    assert "vendored_pin" not in candidates  # commit entry skipped, no raise
    assert "upload_done.json" in candidates  # sibling blob rows still parse
