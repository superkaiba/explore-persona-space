"""Tests for ``--check-no-unannotated-gcp-pin-guidance`` (#2018 D5/D6).

The check is a WARN-only sweep of the live workflow surface for guidance
DIRECTING a gcp backend pin — a dead end since #2028 (`GcpDisabledError`) —
with a SKIP-on-rollback arm keyed on ``GCP_PROVISIONING_DISABLED`` read from
``router.py`` SOURCE via ast (both ``Assign`` and the REAL annotated
``AnnAssign`` forms). The plan's D6 numbering, in order:

1.  ``test_bare_pin_line_unannotated_warns`` — bare pin, no annotation
    → 1 WARN.
2.  ``test_same_line_annotation_escapes`` — ``#2028`` on the line → 0 WARN.
3.  ``test_top_banner_within_window_escapes`` — hit 20 lines below a
    top-of-file ``#2028 ... DISABLED`` banner → 0 WARN (the
    ``compute-backend-failover.md`` shape).
4.  ``test_file_scope_banner_beyond_window_escapes`` — hit ~200 lines below
    the banner, no nearer annotation → 0 WARN (pins the first-40-lines
    FILE-SCOPE arm specifically, not a window accident).
5.  ``test_worktrees_path_excluded`` — an unannotated hit under
    ``.claude/worktrees/`` → 0 WARN + not in the scanned list (and the
    root-RELATIVE exclusion helper is exercised directly: the scanned repo
    root may itself BE a worktree checkout, so an absolute-path match
    would self-exclude the whole scan set).
6.  ``test_rollback_flag_false_skips`` — ``GCP_PROVISIONING_DISABLED =
    False`` in the read source → the whole check SKIPs.
7.  ``test_missing_router_skips_loud_exit_zero`` — unreadable/absent
    router.py → loud SKIP, CLI exit 0 (fail-open).
8.  ``test_live_surface_zero_warns_and_armed`` — the real repo: zero WARNs
    AND ``skipped is False`` AND ``files_scanned > 0`` (the armed
    assertion is the load-bearing half — an inert check satisfies the
    zero-WARN half vacuously).
9.  ``test_warn_only_cli_exits_zero_with_hits`` — a repo WITH hits still
    exits 0 (WARN-only; the no-flags run feeds the Step 9c gate, #1388).
10. ``test_kv_pin_word_boundary_gcp_backend_token`` — a MARKDOWN
    ``backend=gcp_backend`` token → 0 WARN (pins the kv-pin word
    boundary; a ``.py`` code line would be excluded by the string-literal
    gate before the boundary is consulted, making the fixture vacuous).
11. ``test_py_literal_gating_split`` — a short ``backend="gcp"`` kwarg
    literal → 0 WARN, while a >=40-char message literal carrying
    ``--backend gcp`` → 1 WARN (pins the MIN_PY_LITERAL_CHARS split; the
    long-literal half is D4's own shape).
12. ``test_py_comment_not_scanned`` — a ``.py`` COMMENT carrying
    ``--backend gcp`` → 0 WARN (pins the string-literal-only rule).
13. ``test_generic_tokens_do_not_escape`` — generic ``raises`` /
    ``no longer`` words in the preceding window do NOT annotate (pins the
    token tightening that unmasked D4 at plan time).
14. ``test_constant_reader_true_on_real_router`` — the reader returns True
    on the REAL router.py bytes (pins the ANNOTATED ``: bool = True``
    form; without this the reader can be written against the bare form,
    never match, and silently disable the whole check while tests 6/7
    stay green).
15. ``test_scan_set_contains_live_memory_file`` — the report's
    scanned-file list contains the live
    ``feedback_gcp_lane_git_clone_only_data.md`` memory (pins scan-set
    membership; every other test stays green if a member is silently
    dropped).

Plus (the house bundling pin, the ``test_check_jsonl_splitlines_bundled_in_
no_flags`` mutation-visible pattern): 16. the no-flags default run actually
DISPATCHES the check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402

_ROUTER_REL = "src/explore_persona_space/backends/router.py"
_ANNOTATED_TRUE = "GCP_PROVISIONING_DISABLED: bool = True\n"
_LIVE_MEMORY_REL = ".claude/agent-memory/experimenter/feedback_gcp_lane_git_clone_only_data.md"


def _plant(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _mk_repo(tmp_path: Path, *, router_text: str | None = _ANNOTATED_TRUE) -> Path:
    if router_text is not None:
        _plant(tmp_path, _ROUTER_REL, router_text)
    return tmp_path


def _run(root: Path) -> tuple[dict[str, object], list[str]]:
    sink: list[str] = []
    report = wl.check_no_unannotated_gcp_pin_guidance(repo_root=root, warn_sink=sink)
    return report, sink


# --------------------------------------------------------------------------
# D6 tests 1-4: trigger + annotation-window semantics
# --------------------------------------------------------------------------


def test_bare_pin_line_unannotated_warns(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    _plant(root, ".claude/agents/foo.md", "some guidance\npin `backend: gcp` for this workload\n")
    report, sink = _run(root)
    assert report["skipped"] is False
    assert len(sink) == 1, sink
    assert "foo.md:2" in sink[0] and "[kv-pin]" in sink[0], sink[0]
    assert report["warnings"] == sink


def test_same_line_annotation_escapes(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    _plant(
        root,
        ".claude/agents/foo.md",
        "some guidance\npin `backend: gcp` for this workload (#2028)\n",
    )
    report, sink = _run(root)
    assert sink == [] and report["files_scanned"] > 0


def test_top_banner_within_window_escapes(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    banner = "> #2028 — GCP provisioning DISABLED; every section below is rollback-only.\n"
    text = banner + "\n" * 19 + "pin `backend: gcp` for this workload\n"
    _plant(root, ".claude/rules/foo.md", text)
    report, sink = _run(root)
    assert sink == [] and report["files_scanned"] > 0


def test_file_scope_banner_beyond_window_escapes(tmp_path: Path) -> None:
    """Hit ~200 lines below the banner: outside the preceding-40 window, so
    only the first-40-lines FILE-SCOPE arm can clear it — pins the intended
    banner semantics rather than a window accident (D6 test 4)."""
    root = _mk_repo(tmp_path)
    banner = "> #2028 — GCP provisioning DISABLED; every section below is rollback-only.\n"
    text = banner + "filler\n" * 198 + "pin `backend: gcp` for this workload\n"
    _plant(root, ".claude/rules/foo.md", text)
    report, sink = _run(root)
    assert sink == [], sink
    assert report["files_scanned"] > 0


# --------------------------------------------------------------------------
# D6 test 5: exclusion
# --------------------------------------------------------------------------


def test_worktrees_path_excluded(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    _plant(root, ".claude/agents/ok.md", "nothing pin-shaped here\n")
    _plant(
        root,
        ".claude/worktrees/issue-9/.claude/agents/stale.md",
        "pin `backend: gcp` for this workload\n",
    )
    report, sink = _run(root)
    assert sink == [], sink
    assert all(".claude/worktrees/" not in f for f in report["scanned_files"])
    # The exclusion is root-RELATIVE by design: the scanned repo root may
    # itself live under .claude/worktrees/ (an issue worktree), and an
    # absolute-path substring match would self-exclude EVERYTHING.
    wt_file = root / ".claude/worktrees/issue-9/.claude/agents/stale.md"
    assert wl._gcp_pin_excluded(wt_file, root) is True
    fake_wt_root = Path("/repo/.claude/worktrees/issue-9")
    assert wl._gcp_pin_excluded(fake_wt_root / ".claude/agents/live.md", fake_wt_root) is False


# --------------------------------------------------------------------------
# D6 tests 6-7: rollback + fail-open SKIP arms
# --------------------------------------------------------------------------


def test_rollback_flag_false_skips(tmp_path: Path) -> None:
    # The bare-Assign form doubles as the Assign-branch pin of the reader.
    root = _mk_repo(tmp_path, router_text="GCP_PROVISIONING_DISABLED = False\n")
    _plant(root, ".claude/agents/foo.md", "pin `backend: gcp` for this workload\n")
    report, sink = _run(root)
    assert report["skipped"] is True
    assert report["skip_reason"] == "gcp-provisioning-enabled"
    assert sink == [] and report["files_scanned"] == 0


def test_missing_router_skips_loud_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _mk_repo(tmp_path, router_text=None)
    _plant(root, ".claude/agents/foo.md", "pin `backend: gcp` for this workload\n")
    report, sink = _run(root)
    assert report["skipped"] is True
    assert report["skip_reason"] == "router-unreadable"
    assert sink == []
    err = capsys.readouterr().err
    assert "SKIPPED" in err and "router source unreadable" in err, err
    # Fail-open at the CLI too: the scoped invocation still exits 0.
    monkeypatch.setattr(wl, "_REPO_ROOT", root)
    rc = wl.main(["--check-no-unannotated-gcp-pin-guidance"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "SKIPPED" in err, err


# --------------------------------------------------------------------------
# D6 tests 8 + 15: live-surface regression + scan-set membership
# --------------------------------------------------------------------------


def test_live_surface_zero_warns_and_armed() -> None:
    report, sink = _run(_REPO_ROOT)
    assert sink == [], "unannotated gcp-pin guidance regrew on the live surface:\n" + "\n".join(
        sink
    )
    assert report["skipped"] is False, report
    assert report["files_scanned"] > 0, report


def test_scan_set_contains_live_memory_file() -> None:
    report, _sink = _run(_REPO_ROOT)
    assert _LIVE_MEMORY_REL in report["scanned_files"], (
        f"the live scan set silently dropped {_LIVE_MEMORY_REL} — every other "
        f"test stays green on a dropped member (D6 test 15)"
    )


# --------------------------------------------------------------------------
# D6 test 9: WARN-only — a repo with hits still exits 0
# --------------------------------------------------------------------------


def test_warn_only_cli_exits_zero_with_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _mk_repo(tmp_path)
    _plant(root, ".claude/agents/foo.md", "pin `backend: gcp` for this workload\n")
    monkeypatch.setattr(wl, "_REPO_ROOT", root)
    rc = wl.main(["--check-no-unannotated-gcp-pin-guidance"])
    assert rc == 0, (
        "the check must be WARN-only — it rides the no-flags run the Step 9c gate consumes (#1388)"
    )
    err = capsys.readouterr().err
    assert "unannotated gcp-pin guidance" in err and "foo.md" in err, err


# --------------------------------------------------------------------------
# D6 tests 10-13: the four measured false-hit shapes (plan critic rounds 1-2)
# --------------------------------------------------------------------------


def test_kv_pin_word_boundary_gcp_backend_token(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    _plant(
        root,
        ".claude/rules/foo.md",
        "the helper threads backend=gcp_backend through the dispatch seam\n",
    )
    report, sink = _run(root)
    assert sink == [] and report["files_scanned"] > 0


def test_py_literal_gating_split(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    code = (
        "def f():\n"
        '    launch(backend="gcp", retries=2)\n'
        "    raise ValueError(\n"
        '        "SLURM lane cannot honor this request: rerun with --backend gcp on the CLI."\n'
        "    )\n"
    )
    _plant(root, "src/explore_persona_space/backends/foo.py", code)
    report, sink = _run(root)
    assert len(sink) == 1, sink
    assert "foo.py:4" in sink[0] and "[cli-pin]" in sink[0], sink[0]
    assert report["files_scanned"] > 0


def test_py_comment_not_scanned(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    code = "# historical: this used to run with --backend gcp on the ladder\nX = 1\n"
    _plant(root, "src/explore_persona_space/backends/foo.py", code)
    _report, sink = _run(root)
    assert sink == [], sink


def test_generic_tokens_do_not_escape(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    text = (
        "the router raises on contact and this path is no longer supported\n"
        "pin `backend: gcp` for this workload\n"
    )
    _plant(root, ".claude/agents/foo.md", text)
    _report, sink = _run(root)
    assert len(sink) == 1, (
        "generic English/Python words ('raises', 'no longer') in the window must "
        "NOT annotate — the loose token set falsely escaped D4 itself at plan time"
    )


# --------------------------------------------------------------------------
# D6 test 14: the constant reader vs the REAL router.py form
# --------------------------------------------------------------------------


def test_constant_reader_true_on_real_router() -> None:
    real = (_REPO_ROOT / _ROUTER_REL).read_text(encoding="utf-8")
    assert wl.read_gcp_disabled_flag(real) is True, (
        "read_gcp_disabled_flag must resolve the REAL (annotated) "
        "GCP_PROVISIONING_DISABLED binding in router.py — a reader written "
        "against the bare-Assign form silently disables the whole check forever"
    )
    # Both binding forms + the None dispositions, pinned synthetically:
    assert wl.read_gcp_disabled_flag("GCP_PROVISIONING_DISABLED: bool = True\n") is True
    assert wl.read_gcp_disabled_flag("GCP_PROVISIONING_DISABLED = True\n") is True
    assert wl.read_gcp_disabled_flag("GCP_PROVISIONING_DISABLED = False\n") is False
    assert wl.read_gcp_disabled_flag("OTHER_FLAG = True\n") is None
    assert wl.read_gcp_disabled_flag("GCP_PROVISIONING_DISABLED = compute()\n") is None
    assert wl.read_gcp_disabled_flag("def broken(:\n") is None


# --------------------------------------------------------------------------
# 16. The mutation-visible no-flags DISPATCH pin (house pattern)
# --------------------------------------------------------------------------


def test_check_gcp_pin_guidance_bundled_in_no_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The no-flags default run actually DISPATCHES the #2018 check —
    deleting its ``or no_flags`` branch must fail this test. Other bundled
    checks contribute unrelated errors on the minimal tree (and MAY red the
    run), so the assertion keys ONLY on this check's own WARN token; the
    check itself must never contribute to the error count (WARN-only,
    pinned separately by test 9)."""
    root = _mk_repo(tmp_path)
    _plant(root, ".claude/agents/foo.md", "pin `backend: gcp` for this workload\n")
    monkeypatch.setattr(wl, "_REPO_ROOT", root)
    wl.main([])
    err = capsys.readouterr().err
    assert "unannotated gcp-pin guidance" in err and "foo.md" in err, (
        f"the gcp-pin-guidance WARN (naming foo.md) is missing from the no-flags "
        f"default run's stderr — the check is not bundled into no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
