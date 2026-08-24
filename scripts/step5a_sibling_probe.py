#!/usr/bin/env python3
"""Step 5a sibling-sync probe: import satisfiability + pair-atomic revert (#2208, #2412).

Factored out of ``.claude/skills/issue/steps/09-step-5.md`` (the #2208 inline
collection probe) and hardened for post-collection skew (#2204, #2412). The
calling arm syncs sibling-issue files (tests + scripts + issue-namespaced src)
from origin/main, then invokes this helper from the MAIN checkout (git-common-dir
resolution — the worktree copy is fork-era by construction). The helper decides,
per synced sibling TEST file, whether the sync is import-satisfiable against THIS
worktree, and on any FAILure reverts EVERY synced file of that test's issue
number (pair-atomic, #1824/#1860).

Verdict arms, in order (the static scan runs FIRST — cheap, no subprocess):

1. Static AST import scan over each synced ``tests/test_issue*_*.py`` —
   function-body imports included (the #2204 escape class that
   ``pytest --collect-only`` is structurally blind to). Per absolute dotted
   module: present at origin/main but MISSING from the worktree FAILs;
   issue-namespaced src must be content-identical to origin/main at the grain
   of the OWNING issue-namespaced component — the whole DIRECTORY when the
   owning component is a directory, so a skewed submodule can never hide
   behind a byte-identical ``__init__.py`` (MF3); shared src needs
   module+symbol existence only (content diffs alone never fail shared src),
   EXCEPT that a name satisfied as a SUBMODULE routes its resolved CHILD
   through the same strict identity arm — a shared parent ``__init__.py``
   (``from ...experiments import behavior_testbed_545 as pkg``) must not
   launder a present-but-skewed issue-namespaced child package (#2412 r2).
   A git error probing origin/main is UNDECIDABLE and FAILs the file — it
   never reads as absent-at-origin (#2412 r2). ``TYPE_CHECKING``-guarded
   imports are skipped entirely (they never execute at runtime); imports
   inside a ``try`` whose handlers catch ImportError/ModuleNotFoundError are
   exempt from the MISSING-module/symbol arms ONLY — never from the strict
   identity arm, because a present-but-skewed module imports fine and the
   guard cannot protect the #2204 class (N2) — and the exemption does NOT
   leak into ``def``/``async def`` bodies, where the import runs at call
   time with the handler out of scope (#2412 r2).
2. Real collection probe (retained from #2208 — values verbatim: 180 s/file
   fence, 900 s best-effort warm-up, timeout = failure) through a
   process-GROUP kill fence (MF1): ``start_new_session=True`` + ``os.killpg``
   SIGTERM -> ``--kill-after`` (default 15 s) -> SIGKILL-to-group, mirroring
   the retired arm's ``timeout --kill-after=15s 180s``. A naive
   single-process kill terminates only the immediate child (``uv``); the
   pytest GRANDCHILDREN inherit the stdout pipe and the post-kill
   ``communicate()`` blocks on it — the helper would HANG instead of
   reverting (the MF1 wedge). SIGKILL-to-group closes every inherited pipe
   end so the final read returns.

Fail-safe direction (#2208): an undecidable probe REVERTS, never keeps. Every
revert git op's rc is checked — the first failure raises, so a failed revert
never masquerades as helper success (N6), and the calling arm's else-branch
performs the full idempotent revert (MF2). The ``--kept-out`` list is written
ONLY after all reverts completed (N6 ordering). Exit 0 means "verdicts
delivered" (FAILed issues were reverted by the helper and omitted from the
kept list); ANY nonzero exit means the helper itself is undecidable and the
calling arm reverts everything.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fnmatch
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

# Issue-number extraction runs over the FULL relative path (never basename:
# ``src/.../issue_1739/fits.py`` has a digit-free basename, and a
# basename-based extraction would corrupt the revert set).
_ISSUE_NUMBER_RE = re.compile(r"issue_?(\d+)")
# Classifier arms (N8: FULLMATCH on component stems — ``issue_763_cofit`` does
# NOT fullmatch ``issue_?\d+``, so such loose files route lenient).
_ISSUE_STEM_RE = re.compile(r"issue_?\d+")
_TRAILING_SLUG_STEM_RE = re.compile(r".*_\d+")
_EXPERIMENTS_PREFIX = ("src", "explore_persona_space", "experiments")
_SYNCED_TEST_PATTERN = "tests/test_issue*_*.py"


def issue_number(relpath: str) -> str | None:
    """First ``issue_?(\\d+)`` match over the FULL relative path, or None."""
    m = _ISSUE_NUMBER_RE.search(relpath)
    return m.group(1) if m else None


def _group_key(relpath: str) -> str:
    # A synced path with no extractable issue number (not producible by the
    # arm's globs today) gets a singleton group: a FAIL reverts it alone.
    return issue_number(relpath) or f"__ungrouped__:{relpath}"


def is_synced_test(relpath: str) -> bool:
    return fnmatch.fnmatch(relpath, _SYNCED_TEST_PATTERN)


def owning_strict_unit(relpath: str) -> str | None:
    """Return the OWNING issue-namespaced unit for a resolved src path (MF3).

    Walking the path components from the root, the owning component is the
    FIRST whose stem (trailing ``.py`` stripped when it is the final
    component) fullmatches ``issue_?\\d+`` (anywhere under src/), OR — for
    components strictly under src/explore_persona_space/experiments/ —
    fullmatches ``.*_\\d+`` (the trailing-issue-number slug convention:
    ``behavior_testbed_545``, ``sycophancy_onpolicy_612``, ...). The strict
    unit is the whole DIRECTORY when the owning component is a dir, the file
    itself when the owning component is the final ``.py`` component. None
    means not issue-namespaced (lenient shared-src routing).
    """
    parts = relpath.split("/")
    n_prefix = len(_EXPERIMENTS_PREFIX)
    under_experiments = tuple(parts[:n_prefix]) == _EXPERIMENTS_PREFIX
    for i, comp in enumerate(parts):
        is_final = i == len(parts) - 1
        stem = comp[:-3] if is_final and comp.endswith(".py") else comp
        matched = bool(_ISSUE_STEM_RE.fullmatch(stem)) or (
            under_experiments and i >= n_prefix and bool(_TRAILING_SLUG_STEM_RE.fullmatch(stem))
        )
        if matched:
            return "/".join(parts[: i + 1])
    return None


def _git(wt: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(wt), *args], capture_output=True, text=True, check=False
    )


def _signal_group(pid: int, sig: signal.Signals) -> None:
    # start_new_session=True makes the child's pgid == its pid; a
    # ProcessLookupError means the group is already dead — proceed.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, sig)


def fenced_run(
    argv: list[str], cwd: Path, timeout: float, kill_after: float
) -> tuple[int, bool, str]:
    """Run ``argv`` under a process-GROUP kill fence (MF1).

    Mirrors the retired arm's ``timeout --kill-after=15s 180s``: SIGTERM to
    the child's whole process group on fence expiry, then SIGKILL to the
    group after ``kill_after`` seconds. Returns ``(rc, timed_out, output)``;
    a spawn failure returns ``(127, False, <message>)``. Named residual
    (parity with the retired coreutils fence): a grandchild that setsid()s
    escapes the group.
    """
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        return 127, False, f"spawn error: {exc}"
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, False, out.decode(errors="replace")
    except subprocess.TimeoutExpired:
        pass
    _signal_group(proc.pid, signal.SIGTERM)
    try:
        out, _ = proc.communicate(timeout=kill_after)
        return proc.returncode, True, out.decode(errors="replace")
    except subprocess.TimeoutExpired:
        pass
    # SIGKILL-to-group closes every inherited pipe end, so this final read
    # returns even when grandchildren held the stdout pipe open.
    _signal_group(proc.pid, signal.SIGKILL)
    out, _ = proc.communicate()
    rc = proc.returncode if proc.returncode is not None else -int(signal.SIGKILL)
    return rc, True, out.decode(errors="replace")


def _is_type_checking_test(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Name) and expr.id == "TYPE_CHECKING":
        return True
    return isinstance(expr, ast.Attribute) and expr.attr == "TYPE_CHECKING"


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return False
    for node in t.elts if isinstance(t, ast.Tuple) else [t]:
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        if name in ("ImportError", "ModuleNotFoundError"):
            return True
    return False


def _walk_imports(
    nodes: list[ast.stmt],
    tc: bool,
    ie: bool,
    out: list[tuple[ast.stmt, bool, bool]],
) -> None:
    """Collect (import node, TYPE_CHECKING-guarded, ImportError-guarded) triples.

    Descends into every statement body (function-body imports count — the
    #2204 escape). ``tc`` is inherited by anything under a TYPE_CHECKING
    ``if`` body — INCLUDING ``def`` bodies (a TYPE_CHECKING-guarded def never
    exists at runtime); ``ie`` by anything under a ``try`` body whose handlers
    catch ImportError/ModuleNotFoundError (by name, incl. tuples) — but ``ie``
    RESETS at ``def``/``async def`` boundaries: an enclosing try guards only
    function CREATION, while the body's imports run later, at call time, with
    the handler out of scope (#2412 r2 deferred-importerror-guard-leak).
    """
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append((node, tc, ie))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk_imports(node.body, tc, False, out)
        elif isinstance(node, ast.If):
            _walk_imports(node.body, tc or _is_type_checking_test(node.test), ie, out)
            _walk_imports(node.orelse, tc, ie, out)
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            guarded = any(_handler_catches_import_error(h) for h in node.handlers)
            _walk_imports(node.body, tc, ie or guarded, out)
            for h in node.handlers:
                _walk_imports(h.body, tc, ie, out)
            _walk_imports(node.orelse, tc, ie, out)
            _walk_imports(node.finalbody, tc, ie, out)
        else:
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(node, field, None)
                if isinstance(sub, list):
                    _walk_imports(sub, tc, ie, out)
            for case in getattr(node, "cases", []) or []:
                _walk_imports(case.body, tc, ie, out)


def _top_level_bindings(tree: ast.Module) -> tuple[set[str], bool]:
    """Names bound at a module's top level, recursing into top-level If/Try.

    Returns ``(names, has_star_import)``.
    """
    names: set[str] = set()
    star = False

    def collect(stmts: list[ast.stmt]) -> None:
        nonlocal star
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    _collect_target_names(target, names)
            elif isinstance(stmt, ast.AnnAssign):
                _collect_target_names(stmt.target, names)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if alias.name == "*":
                        star = True
                    else:
                        names.add(alias.asname or alias.name)
            elif isinstance(stmt, ast.If):
                collect(stmt.body)
                collect(stmt.orelse)
            elif isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                collect(stmt.body)
                for h in stmt.handlers:
                    collect(h.body)
                collect(stmt.orelse)
                collect(stmt.finalbody)

    collect(tree.body)
    return names, star


def _collect_target_names(target: ast.expr, names: set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_target_names(elt, names)
    elif isinstance(target, ast.Starred):
        _collect_target_names(target.value, names)


class OriginProbeError(RuntimeError):
    """A git error while probing origin/main's tree — undecidable (#2412 r2).

    Raised (never swallowed into an absent-at-origin reading) so the caller
    FAILs the file under scan: fail-safe direction, revert — never keep.
    """


class _ScanContext:
    """Per-run caches for the static scan's git + AST reads."""

    def __init__(self, wt: Path) -> None:
        self.wt = wt
        self._unit_cache: dict[str, bool] = {}
        self._bindings_cache: dict[str, tuple[set[str], bool]] = {}

    def origin_has(self, relpath: str) -> bool:
        """True iff origin/main's tree contains ``relpath`` (#2412 r2).

        Empirically (git 2.34) ``cat-file -e origin/main:<path>`` exits 128
        for BOTH an absent path and a broken/missing ref, so its rc alone
        cannot separate genuine absence from a git fault — the round-1 code
        read every nonzero rc as absent and FAILED OPEN (a broken ref made
        every project-src missing-module import look third-party, N7-skipped
        and silently KEPT). ``ls-tree`` discriminates three ways: with a
        resolvable ref it exits 0 whether or not the path exists (stdout
        empty on absence); a nonzero rc is a genuine git error (bad ref,
        corrupt repo) and raises :class:`OriginProbeError` — undecidable,
        so the caller reverts (mirrors ``unit_identical``'s fail-safe).
        """
        cp = _git(self.wt, "ls-tree", "--name-only", "origin/main", "--", relpath)
        if cp.returncode != 0:
            raise OriginProbeError(
                f"git ls-tree failed probing origin/main:{relpath}"
                f" (rc={cp.returncode}): {cp.stderr.strip()}"
            )
        return bool(cp.stdout.strip())

    def unit_identical(self, unit: str) -> bool:
        if unit not in self._unit_cache:
            cp = _git(self.wt, "diff", "--quiet", "origin/main", "--", unit)
            # rc 0 = identical, rc 1 = differs; any OTHER rc is a git error —
            # undecidable, so it reads as differing (fail-safe: revert).
            self._unit_cache[unit] = cp.returncode == 0
        return self._unit_cache[unit]

    def module_bindings(self, relpath: str) -> tuple[set[str], bool]:
        if relpath not in self._bindings_cache:
            tree = ast.parse((self.wt / relpath).read_bytes(), filename=relpath)
            self._bindings_cache[relpath] = _top_level_bindings(tree)
        return self._bindings_cache[relpath]

    def resolve_submodule(self, package_init_relpath: str, name: str) -> str | None:
        """Worktree-relative path of ``<pkg>.<name>``, or None when absent.

        Returns the child package's ``__init__.py`` or the child module file;
        the package directory wins over a same-named module file, mirroring
        the import system's precedence. Supersedes the round-1
        ``submodule_exists`` (existence-only), whose bare-bool answer let a
        present-but-skewed issue-namespaced child escape the strict identity
        arm (#2412 r2 parent-package-strict-bypass).
        """
        pkg_rel = package_init_relpath.rsplit("/", 1)[0]
        for cand in (f"{pkg_rel}/{name}/__init__.py", f"{pkg_rel}/{name}.py"):
            if (self.wt / cand).is_file():
                return cand
        return None


def _check_module(ctx: _ScanContext, module: str, names: list[str], ie_guarded: bool) -> str | None:
    """Verdict for one absolute dotted module import. Returns a FAIL reason or None."""
    base = "src/" + module.replace(".", "/")
    candidates = (f"{base}/__init__.py", f"{base}.py")
    wt_path = next((c for c in candidates if (ctx.wt / c).is_file()), None)
    if wt_path is None:
        try:
            origin_present = any(ctx.origin_has(c) for c in candidates)
        except OriginProbeError as exc:
            # A git error is undecidable — never read as absent-at-origin
            # (fail-safe: revert; mirrors unit_identical's error handling).
            return f"origin/main probe failed for module {module}: {exc}"
        if not origin_present:
            # Neither the worktree nor origin/main has it: not a project-src
            # import (third-party/stdlib) — SKIP (N7). A project-src module
            # absent from BOTH trees would be a pre-existing origin/main red,
            # not a sync artifact; hardening this skip into a strict FAIL
            # would over-revert genuine third-party imports.
            return None
        if ie_guarded:
            return None  # try/except ImportError handles absence gracefully (N2 beta)
        return (
            f"module {module} present at origin/main but MISSING from worktree src"
            f" ({candidates[0]} / {candidates[1]})"
        )
    unit = owning_strict_unit(wt_path)
    if unit is not None:
        # Strict identity at the OWNING-component (directory) grain (MF3) —
        # deliberately NOT exempt under the ImportError guard: a
        # present-but-skewed module imports fine, so the guard never fires.
        if not ctx.unit_identical(unit):
            return (
                f"issue-namespaced unit {unit} differs from origin/main"
                f" (strict identity at owning-component grain; module {module})"
            )
        return None
    # Shared src: module+symbol existence only — content diffs alone never
    # fail shared src (old branches always differ; see plan #2412 section 11).
    if not names:
        return None
    try:
        bindings, star = ctx.module_bindings(wt_path)
    except SyntaxError as exc:
        # A SyntaxError raises at import time regardless of any
        # except-ImportError guard — undecidable, revert (never beta-exempt).
        return f"syntax error parsing module {module} ({wt_path}): {exc}"
    for name in names:
        if name in bindings or star:
            continue
        if wt_path.endswith("/__init__.py"):
            child = ctx.resolve_submodule(wt_path, name)
            if child is not None:
                # Submodule import — the dominant real shape. The CHILD may be
                # issue-namespaced even when the PARENT package is shared
                # (`from ...experiments import behavior_testbed_545 as pkg`
                # resolves the shared experiments/__init__.py), so route the
                # resolved child through the SAME strict owning-unit identity
                # arm before accepting it — existence alone silently KEPT a
                # present-but-skewed child package (#2412 r2
                # parent-package-strict-bypass; NOT ie-exempt, same N2
                # rationale as the module-level strict arm above).
                child_unit = owning_strict_unit(child)
                if child_unit is not None and not ctx.unit_identical(child_unit):
                    return (
                        f"issue-namespaced unit {child_unit} differs from origin/main"
                        f" (strict identity at owning-component grain;"
                        f" submodule {module}.{name})"
                    )
                continue
        if ie_guarded:
            continue
        return f"symbol {name} unsatisfiable in module {module} ({wt_path})"
    return None


def scan_test_file(ctx: _ScanContext, relpath: str) -> str | None:
    """Static AST import scan of one synced test file. FAIL reason or None."""
    try:
        source = (ctx.wt / relpath).read_bytes()
    except OSError as exc:
        return f"unreadable synced test {relpath}: {exc}"
    try:
        tree = ast.parse(source, filename=relpath)
    except SyntaxError as exc:
        return f"syntax error in {relpath}: {exc}"
    imports: list[tuple[ast.stmt, bool, bool]] = []
    _walk_imports(tree.body, False, False, imports)
    for node, tc_guarded, ie_guarded in imports:
        if tc_guarded:
            continue  # never executes at runtime (N2 alpha)
        if isinstance(node, ast.Import):
            for alias in node.names:
                reason = _check_module(ctx, alias.name, [], ie_guarded)
                if reason:
                    return reason
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or node.module is None:
                continue  # relative import — skip
            names = [a.name for a in node.names if a.name != "*"]
            reason = _check_module(ctx, node.module, names, ie_guarded)
            if reason:
                return reason
    return None


def revert_issue_files(wt: Path, files: list[str]) -> None:
    """Pair-atomic revert of one issue group (verbatim #2208 semantics).

    Branch-era files are restored via ``git checkout HEAD -- <f>``; main-NEW
    files are dropped via ``git rm -f -q -- <f>`` (index + tree). Every git
    op's rc is CHECKED: the first failure raises (traceback -> exit 1),
    handing recovery to the calling arm's full idempotent revert (N6/MF2 —
    a failed revert must never masquerade as helper success).
    """
    for f in files:
        if _git(wt, "cat-file", "-e", f"HEAD:{f}").returncode == 0:
            cp = _git(wt, "checkout", "HEAD", "--", f)  # restore branch-era content
        else:
            cp = _git(wt, "rm", "-f", "-q", "--", f)  # main-NEW file — drop it (index + tree)
        if cp.returncode != 0:
            raise RuntimeError(
                f"revert git op failed (rc={cp.returncode}) for {f}: {cp.stderr.strip()}"
                " — failing loud so the calling arm reverts everything (N6/MF2)."
            )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5a sibling-sync import-satisfiability probe + pair-atomic revert"
            " (#2208, #2412). Defaults ARE the pinned production values."
        )
    )
    parser.add_argument("--worktree", required=True, help="the issue worktree root (WT)")
    parser.add_argument(
        "--kept-out",
        default=None,
        help="write surviving synced paths (one per line) here, AFTER all reverts (N6)",
    )
    parser.add_argument("--collect-cmd", default="uv run pytest --collect-only -q")
    parser.add_argument("--warmup-cmd", default="uv run python -c pass")
    parser.add_argument("--collect-timeout", type=float, default=180)
    parser.add_argument("--warmup-timeout", type=float, default=900)
    parser.add_argument("--kill-after", type=float, default=15)
    parser.add_argument("synced", nargs="+", metavar="SYNCED_FILE")
    return parser.parse_args(argv)


def _probe_collection(
    args: argparse.Namespace,
    wt: Path,
    test_files: list[str],
    failed: dict[str, str],
) -> None:
    """Step 3: real collection probe for tests whose issue still passes."""
    to_probe = [f for f in test_files if _group_key(f) not in failed]
    if not to_probe:
        return
    # Warm the worktree venv OUTSIDE the per-file fence (values verbatim from
    # #2208). Best-effort: its own failure/timeout is NOT a FAIL — but it runs
    # through the same process-group fence so it can never wedge (MF1).
    fenced_run(
        shlex.split(args.warmup_cmd),
        cwd=wt,
        timeout=args.warmup_timeout,
        kill_after=args.kill_after,
    )
    collect_argv = shlex.split(args.collect_cmd)
    for f in to_probe:
        key = _group_key(f)
        if key in failed:
            continue  # a sibling test already failed this issue group
        rc, timed_out, _out = fenced_run(
            [*collect_argv, f],
            cwd=wt,
            timeout=args.collect_timeout,
            kill_after=args.kill_after,
        )
        if rc != 0 or timed_out:
            failed[key] = "collection failed"
            m = issue_number(f) or "?"
            print(
                f"spec-freshness: sibling test {f} fails collection in this worktree"
                f" (likely branch-era src import skew, #2206)"
                f" — reverting its issue-{m} synced pair (#2208)."
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    wt = Path(args.worktree).resolve()
    if not wt.is_dir():
        print(f"step5a-probe: worktree {wt} does not exist", file=sys.stderr)
        return 1
    synced = list(args.synced)
    groups: dict[str, list[str]] = {}
    for f in synced:
        groups.setdefault(_group_key(f), []).append(f)
    test_files = [f for f in synced if is_synced_test(f)]
    failed: dict[str, str] = {}
    # Step 2: static AST import scan — runs FIRST (cheap, no subprocess).
    ctx = _ScanContext(wt)
    for f in test_files:
        key = _group_key(f)
        if key in failed:
            continue
        reason = scan_test_file(ctx, f)
        if reason:
            failed[key] = reason
            m = issue_number(f) or "?"
            print(
                f"spec-freshness: sibling test {f} static import scan: {reason}"
                f" (#2206 class) — reverting its issue-{m} synced pair (#2208)."
            )
    # Step 3: retained collection probe (probe-before-revert: no mutation
    # happens until every verdict is in).
    _probe_collection(args, wt, test_files, failed)
    # Step 4: pair-atomic revert of every FAILed issue group.
    for key in sorted(failed):
        revert_issue_files(wt, groups[key])
    # Step 5: kept list ONLY AFTER all reverts completed (N6 ordering).
    kept = [f for f in synced if _group_key(f) not in failed]
    if args.kept_out:
        Path(args.kept_out).write_text("".join(p + "\n" for p in kept))
    print(
        f"step5a-probe: kept {len(kept)} of {len(synced)} synced file(s);"
        f" reverted {len(synced) - len(kept)} across {len(failed)} issue group(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
