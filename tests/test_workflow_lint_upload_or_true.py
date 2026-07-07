"""Tests for ``workflow_lint.check_upload_or_true`` (#1036, incident #841).

The check FAILs any upload / result-persist / result-production (plot-script)
command line in ``scripts/**/*.sh`` whose failure is swallowed by ``|| true``
/ ``|| :`` / ``; true``. Terminal swallows mask the whole ``&&``-chain
(whole-line token check); a non-terminal ``|| true`` is segment-scoped;
swallowed heredoc openers and multi-line ``python -c "…"`` quoted blocks are
scanned for BODY upload-call tokens. Legacy deliberate uses are frozen in
``UPLOAD_OR_TRUE_LEGACY_ALLOWLIST``; ``# UPLOAD_OR_TRUE_EXEMPT: <reason>``
(reason ≥ 10 chars) waives.

Covers cases (a)-(bb) from the plan (tasks/…/1036/plans/v2.md §4.7):
(a) upload-script token FLAGs; (b) git-persist token FLAGs; (c) swallowed
heredoc opener + body ``api.upload_file(`` FLAGs; (d) no-token kill line
passes; (e) ``clean_experiment_downloads.py || true`` passes (downloads ≠
upload anti-trap); (f) the 841:85 comment shape passes; (g) bare
``eval_results`` outside the git alternation passes; (h) ``echo`` skip;
(i) waiver same/preceding line passes, <10-char reason FLAGs; (j)
non-terminal segment scoping passes; (k) backslash-continued FLAGs; (l)
terminal ``; true`` FLAGs; (m) ``$HF_DATA_REPO`` token + ``|| :`` variant
FLAG; (n) exact-relative-path allowlist semantics (tmp fixture named like an
allowlisted file still FLAGs); (o) ``test_live_trees_pass`` no-FP baseline;
(p) robustness; (r) founding-incident verbatim pre-fix #841 lines FLAG; (s)
the CURRENT #841 multi-line ``python -c`` upload block FLAGs iff swallowed;
(t) chain swallows (terminal whole-line rule); (u) committed main()-path
wiring pins (#930 precedent); (v) dedupe — one error per opener line; (w)
heredoc-body comment skip + 10-char waiver boundary; (x) echo-skip FN
freeze; (y) durable detection pin — empty allowlist on the live tree flags
the 3 known files; (z) token-alternation one-liners; (aa) single-line
``python -c`` with balanced quotes and no token passes; (bb) pre-commit
hook config pin. Case (q) — the ``check_heredoc_dotenv`` regression after
the shared-iterator rewire — lives in ``tests/test_workflow_lint.py``
(:1466-1616), not here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    UPLOAD_OR_TRUE_LEGACY_ALLOWLIST,
    check_upload_or_true,
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# (a) upload-script token + terminal swallow FLAGs
# --------------------------------------------------------------------------


def test_upload_script_terminal_or_true_flags(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "a.sh",
        "#!/usr/bin/env bash\nuv run python scripts/verify_uploads.py --issue 5 || true\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(p) in errors[0]
    assert ":2:" in errors[0]
    assert "UPLOAD_OR_TRUE_EXEMPT" in errors[0]


# --------------------------------------------------------------------------
# (b) git-persist token FLAGs
# --------------------------------------------------------------------------


def test_git_add_result_dirs_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "b.sh",
        '#!/usr/bin/env bash\ngit add "eval_results/issue_5" "figures/issue_5" || true\n',
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (c) swallowed heredoc opener + body upload-call token FLAGs
# --------------------------------------------------------------------------


def test_swallowed_heredoc_opener_with_body_upload_flags(tmp_path: Path) -> None:
    """The i632:30 shape: the OPENER carries no token; the BODY calls
    ``api.upload_file(`` — a line-only scan provably misses this."""
    _write(
        tmp_path,
        "c.sh",
        "#!/usr/bin/env bash\n"
        "uv run python - <<'PY' 2>&1 || true\n"
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n"
        "api.upload_file(path_or_fileobj='x', path_in_repo='y', repo_id='z')\n"
        "PY\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert ":2:" in errors[0]  # flags the OPENER line


# --------------------------------------------------------------------------
# (d) no-token benign swallow passes
# --------------------------------------------------------------------------


def test_no_token_kill_line_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "d.sh",
        '#!/usr/bin/env bash\nkill -9 "$pid" 2>/dev/null || true\n',
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (e) downloads ≠ upload anti-trap
# --------------------------------------------------------------------------


def test_clean_experiment_downloads_passes(tmp_path: Path) -> None:
    """ "downloads" contains no "upload" substring — the cache-cleanup line
    (issue778_dispatch.sh:118 / issue841_scaling_dispatch.sh:72 shape) passes."""
    _write(
        tmp_path,
        "e.sh",
        "#!/usr/bin/env bash\n"
        "uv run python scripts/clean_experiment_downloads.py 5 --incremental --apply || true\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (f) comment skip (the issue841_scaling_dispatch.sh:85 shape)
# --------------------------------------------------------------------------


def test_comment_line_with_swallow_and_upload_words_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "f.sh",
        "#!/usr/bin/env bash\n"
        "# FAIL-LOUD (no `|| true` swallow): a plot failure must abort BEFORE the upload +\n"
        "# sentinel, so the run never reports success with figures missing.\n"
        "true\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (g) bare eval_results outside the git alternation passes
# --------------------------------------------------------------------------


def test_ls_eval_results_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "g.sh",
        "#!/usr/bin/env bash\nls -la eval_results/issue_404/ 2>/dev/null || true\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (h) echo skip
# --------------------------------------------------------------------------


def test_echo_line_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "h.sh",
        '#!/usr/bin/env bash\necho "upload_file done || true"\n',
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (i) waiver: same line / preceding non-blank line pass; short reason FLAGs
# --------------------------------------------------------------------------


def test_waiver_same_line_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "i1.sh",
        "#!/usr/bin/env bash\n"
        "uv run python scripts/verify_uploads.py --issue 5 || true"
        "  # UPLOAD_OR_TRUE_EXEMPT: crash-diagnostics side-channel\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


def test_waiver_preceding_line_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "i2.sh",
        "#!/usr/bin/env bash\n"
        "# UPLOAD_OR_TRUE_EXEMPT: crash-diagnostics side-channel\n"
        "uv run python scripts/verify_uploads.py --issue 5 || true\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


def test_waiver_short_reason_still_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "i3.sh",
        "#!/usr/bin/env bash\n"
        "uv run python scripts/verify_uploads.py --issue 5 || true"
        "  # UPLOAD_OR_TRUE_EXEMPT: short\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (j) non-terminal segment scoping: benign swallow + upload elsewhere passes
# --------------------------------------------------------------------------


_CASE_J_LINE = (
    'mkdir -p "$OUT" 2>/dev/null || true && uv run python scripts/verify_uploads.py --issue 5\n'
)


def test_nonterminal_swallow_other_segment_passes(tmp_path: Path) -> None:
    _write(tmp_path, "j.sh", "#!/usr/bin/env bash\n" + _CASE_J_LINE)
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (k) backslash-continued logical line FLAGs at the first physical line
# --------------------------------------------------------------------------


def test_backslash_continued_upload_flags(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "k.sh",
        "#!/usr/bin/env bash\nuv run python scripts/verify_uploads.py \\\n  --issue 5 || true\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(p) in errors[0]
    assert ":2:" in errors[0]  # the logical line's FIRST physical line


# --------------------------------------------------------------------------
# (l) terminal `; true` FLAGs
# --------------------------------------------------------------------------


def test_terminal_semicolon_true_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "l.sh",
        "#!/usr/bin/env bash\nuv run python scripts/verify_uploads.py --issue 5; true\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (m) repo-destination env-var token FLAGs; `|| :` synonym FLAGs
# --------------------------------------------------------------------------


def test_hf_data_repo_env_var_token_flags(tmp_path: Path) -> None:
    """The issue654_dispatch.sh:161 opener shape, without the heredoc."""
    _write(
        tmp_path,
        "m1.sh",
        '#!/usr/bin/env bash\nuv run python - "$LOG_DIR" "$HF_DATA_REPO" 2>&1 || true\n',
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


def test_or_colon_synonym_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "m2.sh",
        "#!/usr/bin/env bash\nuv run python scripts/verify_uploads.py --issue 5 || :\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (n) allowlist: exact-relative-path semantics — a tmp fixture sharing the
# basename of an allowlisted file lives OUTSIDE the repo, so it still FLAGs
# --------------------------------------------------------------------------


def test_allowlist_is_exact_relative_path(tmp_path: Path) -> None:
    assert "scripts/issue654_dispatch.sh" in UPLOAD_OR_TRUE_LEGACY_ALLOWLIST
    _write(
        tmp_path,
        "issue654_dispatch.sh",
        '#!/usr/bin/env bash\ngit add "eval_results/x" || true\n',
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (o) test_live_trees_pass — the no-false-positive baseline
# --------------------------------------------------------------------------


def test_live_trees_pass() -> None:
    """The real scripts/ tree PASSES the check — the no-FALSE-POSITIVE
    baseline / no-flags-default-run invariant. It proves the current
    detector + allowlist return no errors against today's tree; it is NOT a
    completeness proof (the detector's shape completeness is pinned by the
    per-shape cases in this file, and case (y) pins that real-shape
    DETECTION stays live). If this FAILs, either the allowlist is
    incomplete (a new deliberate best-effort upload landed — waive it with
    `# UPLOAD_OR_TRUE_EXEMPT: <reason>` rather than growing the allowlist)
    or the gate has a false positive."""
    assert check_upload_or_true() == []


# --------------------------------------------------------------------------
# (p) robustness
# --------------------------------------------------------------------------


def test_missing_scripts_dir_returns_empty(tmp_path: Path) -> None:
    assert check_upload_or_true(scripts_dir=tmp_path / "nope") == []


def test_heredoc_body_upload_without_swallow_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "p.sh",
        "#!/usr/bin/env bash\nuv run python - <<'PY'\napi.upload_file(path_or_fileobj='x')\nPY\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (r) founding-incident regression: the two VERBATIM pre-fix #841 lines
# (git 8bc38f0e6f:scripts/issue841_scaling_dispatch.sh:86 and
#  68d38959a7:scripts/issue841_gru_source_only_dispatch.sh:78) FLAG
# --------------------------------------------------------------------------


def test_founding_incident_prefix_841_lines_flag(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "r.sh",
        "#!/usr/bin/env bash\n"
        "uv run python scripts/issue841_scaling_plots.py || true\n"
        "uv run python scripts/issue841_gru_source_only_plots.py"
        ' --out-dir "$OUT_DIR" --fig-dir "$FIG_DIR" || true\n',
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 2, errors
    assert ":2:" in errors[0]
    assert ":3:" in errors[1]


# --------------------------------------------------------------------------
# (s) current-#841 upload-block replay: a multi-line `python -c "` block
# (copied from issue841_scaling_dispatch.sh:106ff, body calls
# upload_split_lfs_to_overflow() FLAGs iff the closing line is swallowed
# --------------------------------------------------------------------------

_CASE_S_BLOCK = (
    "#!/usr/bin/env bash\n"
    'uv run python -c "\n'
    "import sys\n"
    "sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')\n"
    "import issue841_scaling_common as S\n"
    "res_dir = S.EVAL_SCALING_DIR\n"
    "assert res_dir.is_dir(), f'results dir missing: {res_dir}'\n"
    "res_dev = S.upload_split_lfs_to_overflow(res_dir, 'issue841_scaling/results',"
    " lfs_glob='*.npz')\n"
    "print('ok')\n"
)


def test_python_c_block_with_terminal_swallow_flags(tmp_path: Path) -> None:
    _write(tmp_path, "s1.sh", _CASE_S_BLOCK + '" || true\n')
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert ":2:" in errors[0]  # flags the OPENER line


def test_python_c_block_without_swallow_passes(tmp_path: Path) -> None:
    _write(tmp_path, "s2.sh", _CASE_S_BLOCK + '"\n')
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (t) chain swallows: a terminal `|| true` masks the WHOLE &&-chain / group
# --------------------------------------------------------------------------


def test_terminal_swallow_after_and_chain_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "t1.sh",
        "#!/usr/bin/env bash\n"
        "uv run python scripts/verify_uploads.py --issue 5 && echo ok || true\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


def test_terminal_swallow_on_brace_group_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "t2.sh",
        "#!/usr/bin/env bash\n{ uv run python scripts/verify_uploads.py --issue 5; } || true\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


def test_case_j_still_passes_under_terminal_rule(tmp_path: Path) -> None:
    """Re-assert case (j): the whole-line terminal rule must NOT regress the
    non-terminal segment-scoping FP kill."""
    _write(tmp_path, "t3.sh", "#!/usr/bin/env bash\n" + _CASE_J_LINE)
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (u) main()-path wiring pins (#930 precedent —
# test_workflow_lint_phase_done_check.py::test_no_flags_default_run_bundles_check):
# a forgotten no_flags / dispatch-ladder wiring would silently drop the check
# from the default run — pin BOTH invocation paths.
# --------------------------------------------------------------------------


def test_flag_run_dispatches_check(monkeypatch, capsys) -> None:
    import workflow_lint

    sentinel = "UPLOAD-OR-TRUE-WIRING-SENTINEL-1036"
    monkeypatch.setattr(workflow_lint, "check_upload_or_true", lambda: [sentinel])
    rc = workflow_lint.main(["--check-upload-or-true"])
    err = capsys.readouterr().err
    assert rc == 1
    assert sentinel in err


def test_no_flags_default_run_bundles_check(monkeypatch, capsys) -> None:
    import workflow_lint

    sentinel = "UPLOAD-OR-TRUE-BUNDLING-SENTINEL-1036"
    monkeypatch.setattr(workflow_lint, "check_upload_or_true", lambda: [sentinel])
    rc = workflow_lint.main([])
    err = capsys.readouterr().err
    assert rc == 1
    assert sentinel in err


# --------------------------------------------------------------------------
# (v) dedupe: the 654 double-hit shape (opener env-var token + body upload
# call) emits EXACTLY one error per opener line
# --------------------------------------------------------------------------


def test_double_hit_opener_dedupes_to_one_error(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "v.sh",
        "#!/usr/bin/env bash\n"
        '(uv run python - "$LOG_DIR" "$HF_DATA_REPO" <<\'PY\' 2>&1 || true\n'
        "def _upload(x):\n"
        "    pass\n"
        "_upload(1)\n"
        "PY\n"
        ") || true\n",
    )
    errors = check_upload_or_true(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert ":2:" in errors[0]


# --------------------------------------------------------------------------
# (w) heredoc-body comment skip; waiver reason exactly-10-chars boundary
# --------------------------------------------------------------------------


def test_heredoc_body_token_only_in_comment_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "w1.sh",
        "#!/usr/bin/env bash\n"
        "uv run python - <<'PY' 2>&1 || true\n"
        "# api.upload_file( is only mentioned in this comment\n"
        "print('no upload here')\n"
        "PY\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


def test_waiver_reason_exactly_ten_chars_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "w2.sh",
        "#!/usr/bin/env bash\n"
        "uv run python scripts/verify_uploads.py --issue 5 || true"
        "  # UPLOAD_OR_TRUE_EXEMPT: 0123456789\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (x) echo-skip FN freeze: `echo "…"; upload || true` merged on ONE logical
# line is skipped whole — a DOCUMENTED accepted false negative (§4.3 rule 1)
# --------------------------------------------------------------------------


def test_echo_then_upload_one_logical_line_documented_fn(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "x.sh",
        "#!/usr/bin/env bash\n"
        'echo "uploading"; uv run python scripts/verify_uploads.py --issue 5 || true\n',
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (y) durable detection pin: an EMPTY allowlist on the LIVE tree flags the 3
# known grandfathered files — the permanent regression pin that real-shape
# detection stays live (the non-vacuity sentinel for case (o))
# --------------------------------------------------------------------------


def test_empty_allowlist_detects_known_live_shapes() -> None:
    errors = check_upload_or_true(allowlist=frozenset())
    text = "\n".join(errors)
    assert "scripts/issue654_dispatch.sh:161" in text, text or "(no errors at all)"
    assert "scripts/i632_dispatch_with_log_capture.sh:30" in text
    assert "scripts/issue931_dispatch.sh:251" in text
    assert len(errors) == 3, errors


# --------------------------------------------------------------------------
# (z) token-alternation one-liners
# --------------------------------------------------------------------------


def test_git_push_or_true_flags(tmp_path: Path) -> None:
    _write(tmp_path, "z1.sh", "#!/usr/bin/env bash\ngit push origin main || true\n")
    assert len(check_upload_or_true(scripts_dir=tmp_path)) == 1


def test_hf_cli_upload_or_true_flags(tmp_path: Path) -> None:
    _write(tmp_path, "z2.sh", "#!/usr/bin/env bash\nhf upload my-repo out.json || true\n")
    assert len(check_upload_or_true(scripts_dir=tmp_path)) == 1


def test_heredoc_body_create_commit_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "z3.sh",
        "#!/usr/bin/env bash\n"
        "uv run python - <<'PY' || true\n"
        "api.create_commit(repo_id='z', operations=ops)\n"
        "PY\n",
    )
    assert len(check_upload_or_true(scripts_dir=tmp_path)) == 1


def test_heredoc_body_push_to_hub_flags(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "z4.sh",
        "#!/usr/bin/env bash\nuv run python - <<'PY' || true\nmodel.push_to_hub('my-repo')\nPY\n",
    )
    assert len(check_upload_or_true(scripts_dir=tmp_path)) == 1


# --------------------------------------------------------------------------
# (aa) single-line `python -c "…" || true` with balanced quotes and no token
# passes (the run_issue452_deconfound.sh:71 shape)
# --------------------------------------------------------------------------


def test_single_line_python_c_no_token_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "aa.sh",
        "#!/usr/bin/env bash\n"
        "uv run python -c \"import json; print(json.dumps({'a': 1}))\" || true\n",
    )
    assert check_upload_or_true(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (bb) pre-commit hook config pin (mirrors
# test_workflow_lint_phase_done_check.py::test_precommit_hook_covers_new_offender_paths)
# --------------------------------------------------------------------------


def test_precommit_hook_covers_new_offender_paths() -> None:
    """.pre-commit-config.yaml must carry a local hook whose entry runs
    --check-upload-or-true with a `files:` regex matching a fresh
    scripts/*.sh dispatcher AND the lint itself — without it, no commit
    gate fires on the #841 offender class."""
    cfg = yaml.safe_load((_HERE.parent / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    local_hooks = [h for repo in cfg["repos"] if repo["repo"] == "local" for h in repo["hooks"]]
    matching = [h for h in local_hooks if "--check-upload-or-true" in h.get("entry", "")]
    assert matching, "no pre-commit hook runs --check-upload-or-true"
    assert any(
        re.search(h["files"], "scripts/new.sh")
        and re.search(h["files"], "scripts/workflow_lint.py")
        and not h.get("pass_filenames", True)
        for h in matching
        if "files" in h
    ), f"no matching hook's files: regex covers scripts/*.sh + the lint: {matching}"
