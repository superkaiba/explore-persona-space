"""Tests for ``workflow_lint.check_phase_done_reserved`` (#930).

The check FAILs any non-redirected ``scripts/**/*.sh`` dispatcher invocation of
a ``scripts/*.py|*.sh`` phase script that contains a genuine ``[phase=done]``
emission site — the reserved-token contract of
``.claude/rules/pod-side-reporting.md`` requirement 1 (a mid-pipeline child
emission reads as a false ``status=done`` to ``poll_pipeline.py``; incidents
#545, #920). Emission detection is AST-based for ``.py`` and quote-aware
comment-stripped ``echo|printf|print(`` for ``.sh``.

Covers the plan §6 matrix:
(a) dispatcher-only emission PASSes; (b) phase-``.py``
``print("[phase=done]")`` FAILs (#920 shape); (c) phase-``.py``
``logger.info("[phase=done] ... %s", cell)`` FAILs — the #545
EMISSION-STYLE shape (logger %-format), NOT the #545 invocation topology
(``.py``-dispatcher subprocess fan-out is the documented §4.6(i) residual
gap); (d) a ``> "$LOG" 2>&1 &`` redirected invocation PASSes; (e)
``2>&1 | tee -a log`` FAILs (tee flows to main stdout); (f)
trailing-comment / docstring token mentions PASS (the
``issue778_finetune.py:274`` anti-trap); (g) ``re.compile`` / ``re.search``
match sites and membership tests PASS; (h) ``.sh -> .sh`` sub-dispatcher
emission FAILs (i488 shape); (i) commented-out and ``echo``-preview
invocation lines PASS; (j) an ``allowlist=`` override suppresses, and a tmp
fixture never matches the production allowlist (relative-path edge grain);
(k) ``# noqa: phase-done-reserved`` on the emission line AND on the
immediately-preceding line each PASS — the waiver anchors at the AST
call-head lineno for multi-line calls (waive at the call head, not beside a
continuation-line string literal); (l) ``test_live_trees_pass`` — the real
tree returns ``[]``; (m) robustness (missing dir / unparseable or missing
target); (n) backslash-continued invocation with the redirect on the
continuation line is merged via ``_iter_logical_shell_lines``; (o) f-string
emission FAILs; (p) ``.sh``-embedded python-heredoc ``print`` FAILs; plus
``2> err.log`` (stderr-only) still FAILs and explicit ``&>`` / ``1>`` /
``>>`` / ``source`` / ``python -u`` legs.

Round-1-critique additions: (q) non-vacuity sentinel — the empty-allowlist
run on the REAL tree returns a superset of known sentinel edges (#545 +
i488), catching a future predicate edit silently zeroing detection; (r)
default-run bundling pin — a monkeypatched failing check makes the no-flags
``main()`` exit 1; (s) pre-commit hook-coverage pin — a local hook runs the
check with a ``files:`` regex matching representative new-offender paths.

Round-2 additions (binding concern ``phase-done-shell-edge-scoping`` — the
check must iterate EVERY invocation on a logical line and scope the
stdout-redirect test to each invocation's own command segment):
(t) ``clean.py && bad.py`` where only ``bad.py`` emits — the SECOND
invocation is seen and flagged (one error; pre-fix the first-match-only
``search`` never reached it); (u) ``bad.py; echo ok > marker.txt`` — an
unrelated LATER redirect in a different segment does not suppress the
unredirected emitting invocation (pre-fix the line-global redirect search
did); (v) a redirect attached directly to the invocation's OWN segment on
a multi-segment line still suppresses (the live-tree shape the round-1
probe found on 3 lines — segment scoping must not un-exclude it); plus
both-emit ``&&`` chains yield one error per edge.
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
    PHASE_DONE_EDGE_LEGACY_ALLOWLIST,
    PHASE_DONE_TOKEN,
    check_phase_done_reserved,
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# (a) dispatcher-only emission PASSes (own-file sites unrestricted)
# --------------------------------------------------------------------------


def test_dispatcher_own_emission_passes(tmp_path: Path) -> None:
    """A dispatcher's OWN terminal `[phase=done]` echo is never flagged; a
    non-emitting invoked child is clean."""
    _write(
        tmp_path,
        "dispatch.sh",
        '#!/usr/bin/env bash\nuv run python scripts/phase.py\necho "[phase=done]"\n',
    )
    _write(tmp_path, "phase.py", 'print("cell complete")\n')
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_dispatcher_multi_site_and_smoke_terminal_pass(tmp_path: Path) -> None:
    """Multiple mode-gated own-file terminals (the issue683 shape) and the
    suffixed smoke terminal (`[phase=done] SMOKE COMPLETE ...`) PASS —
    own-file emission counts are unrestricted by construction."""
    _write(
        tmp_path,
        "multi_exit.sh",
        "#!/usr/bin/env bash\n"
        'if [ "$1" = smoke ]; then\n'
        '  echo "[phase=done] SMOKE COMPLETE 2/2 cells"\n'
        "  exit 0\n"
        "fi\n"
        'echo "[phase=done]"\n',
    )
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (b) phase-.py print("[phase=done]") FAILs (the #920 shape)
# --------------------------------------------------------------------------


def test_phase_py_print_emission_fails(tmp_path: Path) -> None:
    """A non-redirected python phase script printing the reserved token FAILs;
    the error names the invocation site (file:line), the target, the emission
    line(s), and the fix hints."""
    sh = _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    _write(tmp_path, "phase.py", 'print("[phase=done] extract complete")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(sh) in errors[0]
    assert ":2:" in errors[0]
    assert "scripts/phase.py" in errors[0]
    assert "[1]" in errors[0]  # the emission line list
    assert "noqa: phase-done-reserved" in errors[0]
    assert "pod-side-reporting.md" in errors[0]


# --------------------------------------------------------------------------
# (c) logger %-format emission FAILs (the #545 EMISSION-STYLE shape — the
# #545 invocation topology itself, .py subprocess fan-out, is the documented
# §4.6(i) residual gap, NOT what this test exercises)
# --------------------------------------------------------------------------


def test_phase_py_logger_percent_format_fails(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py 2>&1 | tee -a run.log\n",
    )
    _write(
        tmp_path,
        "phase.py",
        'logger.info("[phase=done] eval cell %s complete", cell)\n',
    )
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (d) stdout-redirected (per-worker isolation) invocations PASS
# --------------------------------------------------------------------------


def test_redirected_worker_invocation_passes(tmp_path: Path) -> None:
    """`> "$WORKER_LOG" 2>&1 &` isolates stdout from the main log — skipped
    (the deliberate issue658 per-worker pattern)."""
    _write(
        tmp_path,
        "dispatch.sh",
        '#!/usr/bin/env bash\nuv run python scripts/phase.py > "$WORKER_LOG" 2>&1 &\n',
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_append_ampgt_and_fd1_redirects_pass(tmp_path: Path) -> None:
    """`>> log`, `&> log`, and `1> log` all isolate stdout — skipped."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\n"
        "uv run python scripts/phase.py >> worker.log 2>&1\n"
        "uv run python scripts/phase.py &> worker.log\n"
        "uv run python scripts/phase.py 1> worker.log 2>&1\n",
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_stderr_only_redirect_still_fails(tmp_path: Path) -> None:
    """`2> err.log` redirects ONLY stderr — stdout still flows into the main
    log, so the edge stays checked and FAILs."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py 2> err.log\n",
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (e) `2>&1 | tee -a log` FAILs (tee duplicates to main stdout — #545 shape)
# --------------------------------------------------------------------------


def test_tee_invocation_fails(tmp_path: Path) -> None:
    sh = _write(
        tmp_path,
        "dispatch.sh",
        '#!/usr/bin/env bash\nuv run python scripts/phase.py 2>&1 | tee -a "$LOG"\n',
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(sh) in errors[0]


# --------------------------------------------------------------------------
# (f) trailing-comment / docstring token mentions PASS (AST predicate)
# --------------------------------------------------------------------------


def test_comment_and_docstring_mentions_pass(tmp_path: Path) -> None:
    """The issue778_finetune.py:274 anti-trap: a trailing `# NOT [phase=done]`
    comment and a docstring mention are NOT emission sites (a naive line
    regex flags exactly this shape; the AST predicate does not)."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    _write(
        tmp_path,
        "phase.py",
        '"""Phase worker. The [phase=done] terminal belongs to the dispatcher."""\n'
        'logger.info("finetune cell %s complete", cell)  # NOT [phase=done] (reserved)\n',
    )
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (g) match/read sites PASS (re.compile / re.search / membership)
# --------------------------------------------------------------------------


def test_match_sites_pass(tmp_path: Path) -> None:
    """The poller's own detection code shapes — `re.compile(r"\\[phase=done\\]")`,
    `re.search("[phase=done]", line)`, and `"[phase=done]" in line` — are
    never emission sites."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    _write(
        tmp_path,
        "phase.py",
        "import re\n"
        'PAT = re.compile(r"\\[phase=done\\]")\n'
        "def scan(line):\n"
        '    if re.search("[phase=done]", line):\n'
        "        return True\n"
        '    return "[phase=done]" in line\n',
    )
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (h) .sh -> .sh sub-dispatcher emission FAILs (the i488 run_all shape)
# --------------------------------------------------------------------------


def test_sh_sub_dispatcher_emission_fails(tmp_path: Path) -> None:
    sh = _write(
        tmp_path,
        "run_all.sh",
        "#!/usr/bin/env bash\nbash scripts/sub_dispatch.sh\n",
    )
    _write(
        tmp_path,
        "sub_dispatch.sh",
        '#!/usr/bin/env bash\necho "[phase=done]"\n',
    )
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(sh) in errors[0]
    assert "scripts/sub_dispatch.sh" in errors[0]
    assert "[2]" in errors[0]


def test_sourced_sub_script_emission_fails(tmp_path: Path) -> None:
    """`source scripts/sub.sh` is an invocation edge too."""
    _write(
        tmp_path,
        "run_all.sh",
        "#!/usr/bin/env bash\nsource scripts/sub.sh\n",
    )
    _write(tmp_path, "sub.sh", '#!/usr/bin/env bash\nprintf "[phase=done]\\n"\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


def test_python_dash_u_invocation_fails(tmp_path: Path) -> None:
    """`nohup python -u scripts/x.py &` (flag between interpreter and path,
    trailing background token, no stdout redirect) is still an edge."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nnohup python -u scripts/phase.py &\n",
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (i) commented-out and echo-preview invocation lines PASS
# --------------------------------------------------------------------------


def test_comment_and_echo_preview_invocations_pass(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\n"
        "# uv run python scripts/phase.py\n"
        'echo "uv run python scripts/phase.py"\n',
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (j) allowlist override suppresses; a tmp fixture never matches the
# production allowlist (relative-path edge grain)
# --------------------------------------------------------------------------


def test_allowlist_override_suppresses(tmp_path: Path) -> None:
    sh = _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    assert len(check_phase_done_reserved(scripts_dir=tmp_path, allowlist=frozenset())) == 1
    # The invoker key for an outside-repo fixture is its own POSIX path.
    allow = frozenset({(sh.as_posix(), "scripts/phase.py")})
    assert check_phase_done_reserved(scripts_dir=tmp_path, allowlist=allow) == []


def test_tmp_fixture_never_matches_production_allowlist(tmp_path: Path) -> None:
    """A tmp fixture reproducing an allowlisted EDGE by basename is NOT
    exempted — the allowlist matches the repo-relative POSIX invoker path,
    and a tmp fixture falls outside it (mirrors judge-pin test (k))."""
    assert (
        "scripts/issue545_dispatch.sh",
        "scripts/issue545_sweep.py",
    ) in PHASE_DONE_EDGE_LEGACY_ALLOWLIST
    _write(
        tmp_path,
        "issue545_dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/issue545_sweep.py\n",
    )
    _write(tmp_path, "issue545_sweep.py", 'print("[phase=done]")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)  # PRODUCTION allowlist
    assert len(errors) == 1, errors


def test_allowlist_is_edge_grain_not_emitter_grain(tmp_path: Path) -> None:
    """A SECOND dispatcher invoking an allowlisted emitter is still flagged —
    the grain is the (invoker, target) EDGE, not the emitter file."""
    _write(
        tmp_path,
        "dispatch_a.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    new_sh = _write(
        tmp_path,
        "dispatch_b.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    allow = frozenset({((tmp_path / "dispatch_a.sh").as_posix(), "scripts/phase.py")})
    errors = check_phase_done_reserved(scripts_dir=tmp_path, allowlist=allow)
    assert len(errors) == 1, errors
    assert str(new_sh) in errors[0]


# --------------------------------------------------------------------------
# (k) per-line waiver: emission line / preceding line; call-head anchoring
# --------------------------------------------------------------------------


def test_waiver_on_emission_line_suppresses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    _write(
        tmp_path,
        "phase.py",
        'print("[phase=done] standalone terminal")'
        "  # noqa: phase-done-reserved (cpu-mid standalone lane only)\n",
    )
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_waiver_on_preceding_line_suppresses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    _write(
        tmp_path,
        "phase.py",
        "# noqa: phase-done-reserved (standalone lane; dispatcher passes --gpu-null-only)\n"
        'print("[phase=done]")\n',
    )
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_sh_waiver_on_emission_line_suppresses(tmp_path: Path) -> None:
    """The waiver works on .sh emission sites too (checked on the RAW line —
    comment-stripping only gates the token/emitter match)."""
    _write(
        tmp_path,
        "run_all.sh",
        "#!/usr/bin/env bash\nbash scripts/sub.sh\n",
    )
    _write(
        tmp_path,
        "sub.sh",
        "#!/usr/bin/env bash\n"
        'echo "[phase=done]"  # noqa: phase-done-reserved (standalone terminal)\n',
    )
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_waiver_anchors_at_call_head_for_multiline_calls(tmp_path: Path) -> None:
    """The waiver anchor for a multi-line .py call is the AST CALL-HEAD lineno:
    a waiver on the call-head line suppresses; a waiver beside the
    continuation-line string literal does NOT (pin the convention)."""
    _write(
        tmp_path,
        "dispatch_ok.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase_ok.py\n",
    )
    _write(
        tmp_path,
        "phase_ok.py",
        "logger.info(  # noqa: phase-done-reserved (standalone-lane terminal)\n"
        '    "[phase=done] terminal"\n'
        ")\n",
    )
    _write(
        tmp_path,
        "dispatch_bad.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase_bad.py\n",
    )
    bad = _write(
        tmp_path,
        "phase_bad.py",
        'logger.info(\n    "[phase=done] terminal"  # noqa: phase-done-reserved\n)\n',
    )
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert bad.name in errors[0]


# --------------------------------------------------------------------------
# (l) the real tree PASSes (legacy edges frozen; no false positives)
# --------------------------------------------------------------------------


def test_live_trees_pass() -> None:
    """The real scripts/ tree PASSes the check with the production allowlist —
    the no-FALSE-POSITIVE baseline / no-flags-default-run invariant. It does
    NOT prove detection is non-vacuous (that is the sentinel-superset test
    below). If this FAILs, either a NEW violating edge landed (fix it — that
    is what the check exists for) or a legacy edge annotation is missing."""
    assert check_phase_done_reserved() == []


# --------------------------------------------------------------------------
# (m) robustness: missing dir; unparseable / missing targets
# --------------------------------------------------------------------------


def test_missing_scripts_dir_returns_empty(tmp_path: Path) -> None:
    assert check_phase_done_reserved(scripts_dir=tmp_path / "nope") == []


def test_unparseable_target_py_skipped(tmp_path: Path) -> None:
    """A SyntaxError-ing target .py is skipped without crashing (a non-parsing
    .py cannot run as a phase script) even though it textually carries the
    token."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/broken.py\n",
    )
    _write(tmp_path, "broken.py", 'def f(:\n    pass\nprint("[phase=done]")\n')
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_missing_target_skipped(tmp_path: Path) -> None:
    """An invocation of a nonexistent target is skipped (nothing to scan)."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/ghost.py\n",
    )
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (n) backslash-continued invocations merge into one logical line
# --------------------------------------------------------------------------


def test_backslash_continued_redirect_on_continuation_passes(tmp_path: Path) -> None:
    """The redirect on the CONTINUATION line still isolates the edge — the
    logical-line merge (`_iter_logical_shell_lines`) sees it."""
    _write(
        tmp_path,
        "dispatch.sh",
        '#!/usr/bin/env bash\nuv run python scripts/phase.py \\\n  > "$WORKER_LOG" 2>&1 &\n',
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_backslash_continued_unredirected_fails(tmp_path: Path) -> None:
    """A continued invocation with NO stdout redirect anywhere on the logical
    line is still an edge; the reported lineno is the FIRST physical line."""
    sh = _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python \\\n  scripts/phase.py --arg 1\n",
    )
    _write(tmp_path, "phase.py", 'print("[phase=done]")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert f"{sh}:2:" in errors[0]


# --------------------------------------------------------------------------
# (o) f-string emission FAILs
# --------------------------------------------------------------------------


def test_fstring_emission_fails(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/phase.py\n",
    )
    _write(tmp_path, "phase.py", 'logger.info(f"[phase=done] {cell}")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (p) .sh-embedded python-heredoc print FAILs
# --------------------------------------------------------------------------


def test_sh_heredoc_print_emission_fails(tmp_path: Path) -> None:
    """A python heredoc inside an invoked .sh (`uv run python - <<'PY'` ...
    `print("[phase=done]")`) is caught by the `print\\s*\\(` emitter leg."""
    _write(
        tmp_path,
        "run_all.sh",
        "#!/usr/bin/env bash\nbash scripts/sub.sh\n",
    )
    _write(
        tmp_path,
        "sub.sh",
        "#!/usr/bin/env bash\nuv run python - <<'PY'\nprint(\"[phase=done]\")\nPY\n",
    )
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert "[3]" in errors[0]


# --------------------------------------------------------------------------
# (q) non-vacuity sentinel: the empty-allowlist run on the REAL tree finds a
# SUPERSET of known legacy edges — catches a future predicate edit silently
# zeroing live-tree detection (the vacuity hole in test (l)), and makes the
# plan §4.5 diff-and-adjudicate self-checking.
# --------------------------------------------------------------------------


def test_nonvacuity_sentinel_edges_detected_on_live_tree() -> None:
    errors = check_phase_done_reserved(allowlist=frozenset())
    text = "\n".join(errors)
    # Sentinel edge 1: the #545 family (py-target, tee'd into the main log).
    assert "scripts/issue545_dispatch.sh" in text, text or "(no errors at all)"
    assert "scripts/issue545_sweep.py" in text
    # Sentinel edges 2+3: the i488 run_all -> sub-dispatcher class (sh-target).
    assert "scripts/i488_run_all.sh" in text
    assert "scripts/i488_phase23_dispatch.sh" in text
    assert "scripts/i488_phase4_dispatch.sh" in text


# --------------------------------------------------------------------------
# (r) default-run bundling pin: the no-flags main() runs the check
# --------------------------------------------------------------------------


def test_no_flags_default_run_bundles_check(monkeypatch, capsys) -> None:
    """A forgotten §4.8 no_flags / dispatch-ladder wiring would silently drop
    the check from the default run — pin it by stubbing the check to one
    error and asserting the no-flags main() reports it and exits 1."""
    import workflow_lint

    sentinel = "PHASE-DONE-BUNDLING-SENTINEL-930"
    monkeypatch.setattr(workflow_lint, "check_phase_done_reserved", lambda: [sentinel])
    rc = workflow_lint.main([])
    err = capsys.readouterr().err
    assert rc == 1
    assert sentinel in err


# --------------------------------------------------------------------------
# (s) pre-commit hook-coverage pin (round-1 Must-Fix MF1 mechanization)
# --------------------------------------------------------------------------


def test_precommit_hook_covers_new_offender_paths() -> None:
    """.pre-commit-config.yaml must carry a local hook whose entry runs
    --check-phase-done-reserved (or the bare no-flags lint) with a `files:`
    regex matching representative NEW-offender paths — a fresh dispatcher and
    a fresh phase script. Without it, no fleet-wide commit gate fires on the
    offender class (the round-1 enforcement-topology correction)."""
    cfg = yaml.safe_load((_HERE.parent / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    local_hooks = [h for repo in cfg["repos"] if repo["repo"] == "local" for h in repo["hooks"]]
    matching = [
        h
        for h in local_hooks
        if "--check-phase-done-reserved" in h.get("entry", "")
        or h.get("entry", "").rstrip().endswith("workflow_lint.py")
    ]
    assert matching, "no pre-commit hook runs --check-phase-done-reserved or the no-flags lint"
    assert any(
        re.search(h["files"], "scripts/issue999_dispatch.sh")
        and re.search(h["files"], "scripts/issue999_phase.py")
        and not h.get("pass_filenames", True)
        for h in matching
        if "files" in h
    ), f"no matching hook's files: regex covers new scripts/*.sh|py offenders: {matching}"


# --------------------------------------------------------------------------
# (t)/(u)/(v) round-2 segment-scoping regression fixtures (binding concern
# `phase-done-shell-edge-scoping`): every invocation on a logical line is
# checked, and the stdout-redirect exclusion is scoped to each invocation's
# OWN command segment (split at unquoted `&&` / `||` / `;` / `|` / lone `&`).
# --------------------------------------------------------------------------


def test_second_invocation_on_and_chain_fails(tmp_path: Path) -> None:
    """(t) `clean.py && bad.py` where only bad.py emits -> exactly one error
    naming bad.py. Pre-fix, the first-match-only `search` stopped at
    clean.py (a clean target) and bad.py was never inspected."""
    sh = _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/clean.py && uv run python scripts/bad.py\n",
    )
    _write(tmp_path, "clean.py", 'print("cells complete")\n')
    _write(tmp_path, "bad.py", 'print("[phase=done] mid-pipeline")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(sh) in errors[0]
    assert "scripts/bad.py" in errors[0]
    assert "scripts/clean.py" not in errors[0]


def test_unrelated_later_redirect_does_not_suppress(tmp_path: Path) -> None:
    """(u) `bad.py; echo ok > marker.txt` -> one error. The redirect belongs
    to the LATER `echo` segment; pre-fix the line-global redirect search let
    it suppress the genuinely non-redirected emitting invocation."""
    sh = _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/bad.py; echo ok > marker.txt\n",
    )
    _write(tmp_path, "bad.py", 'print("[phase=done]")\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(sh) in errors[0]
    assert "scripts/bad.py" in errors[0]


def test_segment_scoped_redirect_still_suppresses(tmp_path: Path) -> None:
    """(v) A redirect attached directly to the invocation's OWN segment on a
    multi-segment line still excludes the edge — the shape the round-1 §4.5
    probe found on 3 live-tree lines; segment scoping must not un-exclude
    it (the expected live-tree edge set stays at the same 9 edges)."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\n"
        'uv run python scripts/bad.py > "$WORKER_LOG" 2>&1 && echo launched\n'
        "uv run python scripts/bad.py >> worker.log 2>&1; echo appended\n",
    )
    _write(tmp_path, "bad.py", 'print("[phase=done]")\n')
    assert check_phase_done_reserved(scripts_dir=tmp_path) == []


def test_both_invocations_emitting_yield_one_error_per_edge(tmp_path: Path) -> None:
    """Two emitting targets chained with `&&` produce one error per edge
    (both are genuine violations; pre-fix only the first was reachable)."""
    _write(
        tmp_path,
        "dispatch.sh",
        "#!/usr/bin/env bash\nuv run python scripts/bad_a.py && bash scripts/bad_b.sh\n",
    )
    _write(tmp_path, "bad_a.py", 'print("[phase=done]")\n')
    _write(tmp_path, "bad_b.sh", 'echo "[phase=done]"\n')
    errors = check_phase_done_reserved(scripts_dir=tmp_path)
    assert len(errors) == 2, errors
    text = "\n".join(errors)
    assert "scripts/bad_a.py" in text
    assert "scripts/bad_b.sh" in text


# --------------------------------------------------------------------------
# constant sanity
# --------------------------------------------------------------------------


def test_token_constant_is_the_reserved_literal() -> None:
    assert PHASE_DONE_TOKEN == "[phase=done]"
