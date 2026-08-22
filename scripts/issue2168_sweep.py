"""One-off scan/sweep for task #2168: add UnicodeDecodeError to JSON-guard exception sets.

Implements the plan v2 SS4-D2 unit-level predicate (EXTENDED form). A "unit" is:

1. an ``ast.Try``       -- union of exception names over ALL handlers' type exprs;
2. an ``ast.TryStar``   -- same handler union (``except*`` groups);
3. an ``ast.With`` / ``ast.AsyncWith`` item whose ``context_expr`` is a ``Call``
   whose func resolves to ``suppress`` (bare name) or ``contextlib.suppress``
   (attribute) -- union over the call's args, per suppress call.

Flag a unit iff its name union contains BOTH
  (a) a JSON-decode name: bare or attribute ``JSONDecodeError``, AND
  (b) an OSError-family name: OSError / IOError / EnvironmentError /
      FileNotFoundError / PermissionError / IsADirectoryError / NotADirectoryError,
and does NOT contain any safe name (UnicodeDecodeError / ValueError / Exception /
BaseException), and no ``# JSON_GUARD_UNICODE_EXEMPT: <reason>`` waiver sits on the
flagged line or the line above.

Modes (MUTUALLY EXCLUSIVE — round 3, concern sweep-check-readonly-mixed-mode:
argparse rejects ``--check --apply`` with exit 2, so the read-only modes can
never be combined with the mutating one):
  --report          list findings (file:line, form, names, long-body tag)
  --check           read-only alias for --report (round 2, concern
                    sweep-check-cli-mismatch: the round-1 report advertised it)
  --apply           perform the one-token edits in place (append UnicodeDecodeError
                    last, mirroring the #2164 shape at autonomous_session_watch.py:21823)

Scope: every ``*.py`` under ``scripts/`` + ``src/`` of the repo root this script
lives in (worktree-safe). The durable reintroduction guard is the workflow_lint.py
check (unit 3 of the plan); this script is the disposable sweep instrument whose
predicate that check reuses.

Known false negatives (disclosed, per plan SS4-D2): dynamically-constructed
exception tuples, ``from json import JSONDecodeError as JDE``-style aliasing,
``from contextlib import suppress as quiet`` aliasing, custom context managers
wrapping suppress semantics. All measured 0 live instances at plan time.

Round-2 divergence note (#2168 review): this sweep is DISPOSABLE (its 186-unit
job is complete) and keeps its original, narrower predicate; the durable lint
check (``workflow_lint.check_json_guard_unicode``) has since been EXTENDED
beyond it — per-``with``-statement suppress union (split-suppress coverage)
and nested-literal-tuple recursion. Waiver handling also deliberately differs:
this script accepts the token on the flagged line or the EXACT previous
physical line with no reason-length floor (a one-off simplification), while
the lint uses the house convention (backward walk over blank lines + a
>=10-char reason, byte-parallel with ``_jsonl_splitlines_waiver_present``) —
the lint is the binding surface, and it is stricter where it matters (reason
length), so no site this sweep skipped escapes the lint. Read posture differs
by design too: this script silently skips unreadable files; the lint skips
only non-UTF-8/unparseable files WITH a stderr notice and propagates OSError.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("scripts", "src")
WAIVER_TOKEN = "JSON_GUARD_UNICODE_EXEMPT"

JSON_NAMES = {"JSONDecodeError"}
OSERROR_NAMES = {
    "OSError",
    "IOError",
    "EnvironmentError",
    "FileNotFoundError",
    "PermissionError",
    "IsADirectoryError",
    "NotADirectoryError",
}
SAFE_NAMES = {"UnicodeDecodeError", "ValueError", "Exception", "BaseException"}


def _terminal_name(node: ast.expr) -> str | None:
    """Terminal identifier of a Name/Attribute exception expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _names_of_type_expr(expr: ast.expr | None) -> tuple[set[str], ast.expr | None]:
    """(name set, last element node) of a handler ``type`` / suppress-arg expr."""
    if expr is None:
        return set(), None
    if isinstance(expr, ast.Tuple):
        names: set[str] = set()
        last: ast.expr | None = None
        for elt in expr.elts:
            name = _terminal_name(elt)
            if name is not None:
                names.add(name)
            last = elt
        return names, last
    name = _terminal_name(expr)
    return ({name} if name is not None else set()), expr


def _is_suppress_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "suppress":
        return True
    return isinstance(func, ast.Attribute) and func.attr == "suppress"


@dataclass
class Finding:
    path: Path
    lineno: int  # 1-based line of the flagged unit (handler / with line)
    form: str  # "try" | "trystar" | "suppress"
    names: set[str]
    # Insertion point: append ", UnicodeDecodeError" AFTER this (line, col), 1-based line.
    insert_line: int
    insert_col: int
    try_body_span: int = 0  # try-body line count (long-body review trigger), try forms only
    handler_linenos: list[int] = field(default_factory=list)


def _waived(lines: list[str], lineno: int) -> bool:
    for ln in (lineno, lineno - 1):
        if 1 <= ln <= len(lines) and WAIVER_TOKEN in lines[ln - 1]:
            return True
    return False


def _scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Try | ast.TryStar):
            names: set[str] = set()
            last_elt: ast.expr | None = None
            insert_handler: ast.ExceptHandler | None = None
            handler_linenos: list[int] = []
            for handler in node.handlers:
                handler_linenos.append(handler.lineno)
                h_names, h_last = _names_of_type_expr(handler.type)
                names |= h_names
                # Insertion target: the handler whose type carries the JSON name.
                if h_names & JSON_NAMES:
                    insert_handler = handler
                    last_elt = h_last
            if not (names & JSON_NAMES and names & OSERROR_NAMES) or names & SAFE_NAMES:
                continue
            flag_line = insert_handler.lineno if insert_handler is not None else node.lineno
            if _waived(lines, flag_line):
                continue
            if last_elt is None or insert_handler is None:
                # Split-handler or degenerate form: report for manual edit.
                findings.append(
                    Finding(
                        path=path,
                        lineno=flag_line,
                        form="try-manual",
                        names=names,
                        insert_line=-1,
                        insert_col=-1,
                        handler_linenos=handler_linenos,
                    )
                )
                continue
            body_span = node.body[-1].end_lineno - node.body[0].lineno + 1
            if not isinstance(insert_handler.type, ast.Tuple):
                # Single-name handler carrying JSONDecodeError while OSError lives in a
                # sibling handler (split form): manual.
                findings.append(
                    Finding(
                        path=path,
                        lineno=flag_line,
                        form="try-manual",
                        names=names,
                        insert_line=-1,
                        insert_col=-1,
                        try_body_span=body_span,
                        handler_linenos=handler_linenos,
                    )
                )
                continue
            findings.append(
                Finding(
                    path=path,
                    lineno=flag_line,
                    form="trystar" if isinstance(node, ast.TryStar) else "try",
                    names=names,
                    insert_line=last_elt.end_lineno,
                    insert_col=last_elt.end_col_offset,
                    try_body_span=body_span,
                    handler_linenos=handler_linenos,
                )
            )
        elif isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                call = item.context_expr
                if not _is_suppress_call(call):
                    continue
                assert isinstance(call, ast.Call)
                names = set()
                last_arg: ast.expr | None = None
                for arg in call.args:
                    name = _terminal_name(arg)
                    if name is not None:
                        names.add(name)
                    last_arg = arg
                if not (names & JSON_NAMES and names & OSERROR_NAMES) or names & SAFE_NAMES:
                    continue
                if _waived(lines, node.lineno):
                    continue
                assert last_arg is not None  # names non-empty implies at least one arg
                findings.append(
                    Finding(
                        path=path,
                        lineno=node.lineno,
                        form="suppress",
                        names=names,
                        insert_line=last_arg.end_lineno,
                        insert_col=last_arg.end_col_offset,
                    )
                )
    return findings


def scan(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for root in SCAN_ROOTS:
        base = repo_root / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            findings.extend(_scan_file(path))
    return findings


def apply_edits(findings: list[Finding]) -> list[Path]:
    """Insert ``, UnicodeDecodeError`` at each finding's insertion point. Returns touched files."""
    by_file: dict[Path, list[Finding]] = {}
    for f in findings:
        if f.form == "try-manual":
            raise SystemExit(
                f"REFUSING --apply: manual-form finding at {f.path}:{f.lineno} "
                "(split-handler or degenerate). Edit it by hand, then re-run."
            )
        by_file.setdefault(f.path, []).append(f)
    touched: list[Path] = []
    for path, file_findings in sorted(by_file.items()):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        # Apply bottom-up so earlier insertion points stay valid.
        for f in sorted(file_findings, key=lambda x: (x.insert_line, x.insert_col), reverse=True):
            idx = f.insert_line - 1
            line = lines[idx]
            col = f.insert_col
            lines[idx] = line[:col] + ", UnicodeDecodeError" + line[col:]
        path.write_text("".join(lines), encoding="utf-8")
        touched.append(path)
    return touched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Round 3 (concern sweep-check-readonly-mixed-mode): the three modes are
    # MUTUALLY EXCLUSIVE — argparse rejects `--check --apply` (exit 2), so the
    # read-only contract of --check/--report cannot be silently combined with
    # the mutating mode.
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--report", action="store_true", help="list findings")
    modes.add_argument(
        "--check",
        action="store_true",
        help="read-only alias for --report (the verification command the #2168 "
        "round-1 report advertised; exit 1 iff findings exist)",
    )
    modes.add_argument("--apply", action="store_true", help="apply edits in place")
    parser.add_argument(
        "--long-body-threshold",
        type=int,
        default=10,
        help="try-body line count above which a finding is tagged LONG_BODY (default 10)",
    )
    args = parser.parse_args()

    findings = scan(REPO_ROOT)
    if args.report or args.check or not args.apply:
        for f in findings:
            rel = f.path.relative_to(REPO_ROOT)
            tag = " LONG_BODY" if f.try_body_span > args.long_body_threshold else ""
            print(
                f"{rel}:{f.lineno}: [{f.form}] names={sorted(f.names)} "
                f"body_span={f.try_body_span}{tag}"
            )
        n_files = len({f.path for f in findings})
        print(f"TOTAL: {len(findings)} units / {n_files} files")
    if args.apply:
        touched = apply_edits(findings)
        print(f"APPLIED: {len(findings)} edits across {len(touched)} files")
    return 0 if not findings or args.apply else 1


if __name__ == "__main__":
    sys.exit(main())
