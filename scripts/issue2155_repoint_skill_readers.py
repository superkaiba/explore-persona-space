#!/usr/bin/env python3
"""Repoint functional SKILL.md read sites in tests/ to tests/issue_skill_source (#2155 B.4).

The `/issue` orchestrator spec is split into a router (`.claude/skills/issue/
SKILL.md`) plus 20 `steps/NN-<slug>.md` companions. Every durability pin that
READS the spec must bind on the LOGICAL document
(`tests.issue_skill_source.issue_skill_text()`), not the physical router file,
or its clause silently un-pins the moment the prose it greps moves into a
companion.

AST-based, never sed (the Aug-7 recipe's traps, re-confirmed on today's tree):

* `SKILL` / `skill` / `SKILL_MD` names in the same files are also bound to
  OTHER skills and to tmp_path fixtures — every site resolves through its own
  scope chain, never a name-global rewrite.
* `tests/test_select_step9c_tests.py` spells the path inside STRING fixtures
  that write fake test files; string constants are data, not read sites, and
  are never rewritten (an AST call-site rewrite cannot touch them).
* `tests/issue_skill_source.py` (the composer itself) is exempt.

Classification per `.read_text()` call site:

* TYPE A — the receiver resolves (scope-aware folding over `/` joins,
  `Path(...)`, `.parent`, `.resolve()`, `.parents[i]`, `__file__` anchors) to
  a repo-real path ending in `skills/issue/SKILL.md` -> the call is replaced
  with `issue_skill_text()`.
* TYPE B — the receiver depends on an enclosing-function PARAMETER, and that
  function is called with an issue-skill path/string argument somewhere in
  the same file -> `X.read_text(...)` becomes `read_workflow_doc(X)` (a
  no-op for every other document).
* TYPE C — the receiver is a for-loop / comprehension TARGET whose iterable
  provably contains an issue-skill path (`for path in (SKILL_MD, CLAUDE_MD)`)
  -> `path.read_text(...)` becomes `read_workflow_doc(path)`; the other loop
  elements pass through unchanged.
* REFUSE — the suffix matches but the base is not provably repo-real
  (tmp_path fixtures, unresolvable names): reported, never rewritten.
* NOTE — non-`read_text` access (stat/open/read_bytes/write) on an
  issue-skill path: reported for hand review, never rewritten.

Usage:
    uv run python scripts/issue2155_repoint_skill_readers.py            # report only
    uv run python scripts/issue2155_repoint_skill_readers.py --apply    # rewrite
    uv run python scripts/issue2155_repoint_skill_readers.py --audit    # assert 0 pending

Re-runnable: after --apply, both the default report and --audit show zero
pending rewrite sites.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
SUFFIX = "skills/issue/SKILL.md"
EXEMPT = {"issue_skill_source.py"}
HELPER_MODULE = "tests.issue_skill_source"
ACCESS_ATTRS = {"read_text", "read_bytes", "open", "stat", "write_text", "write_bytes"}
_MAX_DEPTH = 12

# ---------------------------------------------------------------------------
# path-expression folding
# ---------------------------------------------------------------------------


@dataclass
class Fold:
    """Result of folding a path expression.

    ``base`` is one of: ``FILE`` (anchored at ``__file__`` — repo-real),
    ``NONE`` (pure string components), ``PARAM:<name>``, ``UNKNOWN:<name>``,
    ``OPAQUE``, ``MIXED``.
    """

    base: str
    comps: list[str] = field(default_factory=list)
    param_dep: bool = False

    @property
    def suffix(self) -> str:
        parts: list[str] = []
        for comp in self.comps:
            parts.extend(p for p in comp.split("/") if p not in ("", "."))
        return "/".join(parts)

    @property
    def is_issue_skill(self) -> bool:
        return self.suffix.endswith(SUFFIX)


OPAQUE = Fold("OPAQUE")


@dataclass
class Scope:
    """Name-resolution context for one read site."""

    module_assigns: dict[str, list[ast.expr]]
    func_assigns: dict[str, list[ast.expr]]
    func_dynamic: set[str]  # for/with/comprehension targets — never resolved
    params: set[str]


def _contains_file_anchor(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(node))


def fold(node: ast.expr, scope: Scope, depth: int = 0) -> Fold:
    if depth > _MAX_DEPTH:
        return OPAQUE
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Fold("NONE", [node.value])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = fold(node.left, scope, depth + 1)
        right = fold(node.right, scope, depth + 1)
        if right.base == "NONE" and not right.param_dep:
            return Fold(left.base, left.comps + right.comps, left.param_dep)
        # a parameter (or unresolved name) participates in the tail
        return Fold(left.base, left.comps, param_dep=True)
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return Fold("FILE")
        if node.id in scope.params:
            return Fold(f"PARAM:{node.id}", param_dep=True)
        if node.id in scope.func_dynamic:
            return Fold(f"UNKNOWN:{node.id}")
        exprs = scope.func_assigns.get(node.id) or scope.module_assigns.get(node.id)
        if exprs:
            folds = [fold(e, scope, depth + 1) for e in exprs]
            verdicts = {(f.base, f.suffix) for f in folds}
            if len(verdicts) == 1:
                return folds[0]
            if any(f.is_issue_skill for f in folds):
                return Fold("MIXED")
            return OPAQUE
        return Fold(f"UNKNOWN:{node.id}")
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Path" and len(node.args) >= 1:
            base = fold(node.args[0], scope, depth + 1)
            for extra in node.args[1:]:
                nxt = fold(extra, scope, depth + 1)
                if nxt.base == "NONE" and not nxt.param_dep:
                    base = Fold(base.base, base.comps + nxt.comps, base.param_dep)
                else:
                    return Fold(base.base, base.comps, param_dep=True)
            return base
        if isinstance(func, ast.Attribute) and func.attr in ("resolve", "absolute", "expanduser"):
            return fold(func.value, scope, depth + 1)
        if isinstance(func, ast.Attribute) and func.attr == "joinpath":
            base = fold(func.value, scope, depth + 1)
            for extra in node.args:
                nxt = fold(extra, scope, depth + 1)
                if nxt.base == "NONE" and not nxt.param_dep:
                    base = Fold(base.base, base.comps + nxt.comps, base.param_dep)
                else:
                    return Fold(base.base, base.comps, param_dep=True)
            return base
        if _contains_file_anchor(node):
            return Fold("FILE")
        return OPAQUE
    if isinstance(node, ast.Attribute):
        if node.attr == "parent":
            inner = fold(node.value, scope, depth + 1)
            comps = inner.comps[:-1] if inner.comps else []
            return Fold(inner.base, comps, inner.param_dep)
        if _contains_file_anchor(node):
            return Fold("FILE")
        return OPAQUE
    if isinstance(node, ast.Subscript):
        # Path(...).parents[i] — an ancestor directory of the anchor.
        if isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            inner = fold(node.value.value, scope, depth + 1)
            return Fold(inner.base, [], inner.param_dep)
        if _contains_file_anchor(node):
            return Fold("FILE")
        return OPAQUE
    if _contains_file_anchor(node):
        return Fold("FILE")
    return OPAQUE


# ---------------------------------------------------------------------------
# per-file analysis
# ---------------------------------------------------------------------------


@dataclass
class Site:
    kind: str  # "A" | "B" | "REFUSE" | "NOTE"
    path: Path
    node: ast.AST
    detail: str
    replacement: str | None = None


def _scoped_nodes(root: ast.AST):
    """Yield nodes within ``root``, not descending into nested function defs."""
    stack = list(ast.iter_child_nodes(root))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _collect_assigns(root: ast.AST) -> dict[str, list[ast.expr]]:
    """Name -> assigned exprs within ``root``'s own scope (nested defs excluded)."""
    out: dict[str, list[ast.expr]] = {}
    for stmt in _scoped_nodes(root):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name):
                out.setdefault(tgt.id, []).append(stmt.value)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            if isinstance(stmt.target, ast.Name):
                out.setdefault(stmt.target.id, []).append(stmt.value)
    return out


def _dynamic_names(func: ast.AST) -> set[str]:
    """Names bound by for/with/comprehension targets in ``func``'s own scope."""
    dyn: set[str] = set()

    def _targets(t: ast.expr) -> None:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                dyn.add(n.id)

    for n in _scoped_nodes(func):
        if isinstance(n, (ast.For, ast.AsyncFor)):
            _targets(n.target)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars is not None:
                    _targets(item.optional_vars)
        elif isinstance(n, ast.comprehension):
            _targets(n.target)
    return dyn


def _enclosing_function_map(tree: ast.Module) -> dict[ast.AST, ast.AST | None]:
    """Map every node to its nearest enclosing FunctionDef (or None)."""
    enclosing: dict[ast.AST, ast.AST | None] = {}

    def visit(node: ast.AST, current: ast.AST | None) -> None:
        enclosing[node] = current
        nxt = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
        for child in ast.iter_child_nodes(node):
            visit(child, nxt)

    visit(tree, None)
    return enclosing


def _arg_is_issue_skill(arg: ast.expr, scope: Scope) -> bool:
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value.replace("\\", "/").endswith(SUFFIX)
    f = fold(arg, scope, 0)
    return f.is_issue_skill


def _loop_iterables(root: ast.AST) -> dict[str, list[ast.expr]]:
    """Loop-target name -> iterable exprs, for for-loops and comprehensions
    in ``root``'s own scope (nested function defs excluded)."""
    out: dict[str, list[ast.expr]] = {}
    for n in _scoped_nodes(root):
        if isinstance(n, (ast.For, ast.AsyncFor)) and isinstance(n.target, ast.Name):
            out.setdefault(n.target.id, []).append(n.iter)
        elif isinstance(n, ast.comprehension) and isinstance(n.target, ast.Name):
            out.setdefault(n.target.id, []).append(n.iter)
    return out


def _iterable_contains_issue_skill(it: ast.expr, scope: Scope, depth: int = 0) -> bool:
    """True when a loop iterable provably yields an issue-skill path."""
    if depth > 4:
        return False
    if isinstance(it, (ast.Tuple, ast.List, ast.Set)):
        return any(_arg_is_issue_skill(el, scope) for el in it.elts)
    if isinstance(it, ast.Name):
        exprs = scope.func_assigns.get(it.id) or scope.module_assigns.get(it.id) or []
        return any(_iterable_contains_issue_skill(e, scope, depth + 1) for e in exprs)
    if (
        isinstance(it, ast.Call)
        and isinstance(it.func, ast.Name)
        and it.func.id
        in (
            "sorted",
            "list",
            "tuple",
            "reversed",
        )
    ):
        return any(_iterable_contains_issue_skill(a, scope, depth + 1) for a in it.args)
    return False


def analyze_file(path: Path, src: str) -> list[Site]:
    tree = ast.parse(src)
    module_assigns = _collect_assigns(tree)
    enclosing = _enclosing_function_map(tree)
    module_dynamic = _dynamic_names(tree)

    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    func_info: dict[ast.AST, tuple[dict[str, list[ast.expr]], set[str], set[str]]] = {}
    for fn in funcs:
        params = {a.arg for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs}
        if fn.args.vararg:
            params.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.add(fn.args.kwarg.arg)
        assigns = _collect_assigns(fn)
        func_info[fn] = (assigns, _dynamic_names(fn), params)

    module_loops = _loop_iterables(tree)
    func_loops = {fn: _loop_iterables(fn) for fn in funcs}

    def scope_for(node: ast.AST) -> Scope:
        fn = enclosing.get(node)
        if fn is not None and fn in func_info:
            assigns, dyn, params = func_info[fn]
            return Scope(module_assigns, assigns, dyn | module_dynamic, params)
        return Scope(module_assigns, {}, module_dynamic, set())

    def loop_feeds_issue_skill(node: ast.AST, name: str, scope: Scope) -> bool:
        fn = enclosing.get(node)
        iters = list(func_loops.get(fn, {}).get(name, [])) + list(module_loops.get(name, []))
        return any(_iterable_contains_issue_skill(it, scope) for it in iters)

    # Which locally-defined functions are called with an issue-skill argument?
    called_with_skill: set[str] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        cscope = scope_for(call)
        args = list(call.args) + [kw.value for kw in call.keywords]
        if any(_arg_is_issue_skill(a, cscope) for a in args):
            called_with_skill.add(call.func.id)

    sites: list[Site] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in ACCESS_ATTRS:
            continue
        recv = func.value
        scope = scope_for(node)
        f = fold(recv, scope, 0)
        if func.attr == "read_text":
            if f.is_issue_skill and f.base == "FILE" and not f.param_dep:
                sites.append(
                    Site(
                        "A",
                        path,
                        node,
                        ast.get_source_segment(src, node) or "?",
                        "issue_skill_text()",
                    )
                )
                continue
            if f.param_dep or f.base.startswith("PARAM:"):
                fn = enclosing.get(node)
                fn_name = getattr(fn, "name", None)
                if fn_name and fn_name in called_with_skill:
                    recv_src = ast.get_source_segment(src, recv) or "?"
                    sites.append(
                        Site(
                            "B",
                            path,
                            node,
                            ast.get_source_segment(src, node) or "?",
                            f"read_workflow_doc({recv_src})",
                        )
                    )
                    continue
                continue  # param-read never fed an issue-skill path — not a target
            if f.base.startswith("UNKNOWN:"):
                name = f.base.split(":", 1)[1]
                if loop_feeds_issue_skill(node, name, scope):
                    recv_src = ast.get_source_segment(src, recv) or "?"
                    sites.append(
                        Site(
                            "C",
                            path,
                            node,
                            ast.get_source_segment(src, node) or "?",
                            f"read_workflow_doc({recv_src})",
                        )
                    )
                continue
            if f.is_issue_skill:
                sites.append(
                    Site(
                        "REFUSE",
                        path,
                        node,
                        f"base={f.base}: {ast.get_source_segment(src, node)!r}",
                    )
                )
            continue
        # non-read_text access on an issue-skill path — hand review only
        if f.is_issue_skill:
            sites.append(
                Site(
                    "NOTE",
                    path,
                    node,
                    f".{func.attr} on issue-skill path (base={f.base}): "
                    f"{ast.get_source_segment(src, node)!r}",
                )
            )
    # builtin open("<...SKILL.md>")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
            and node.args
        ):
            scope = scope_for(node)
            if _arg_is_issue_skill(node.args[0], scope):
                sites.append(
                    Site(
                        "NOTE",
                        path,
                        node,
                        f"builtin open() on issue-skill path: "
                        f"{ast.get_source_segment(src, node)!r}",
                    )
                )
    return sites


# ---------------------------------------------------------------------------
# rewriting
# ---------------------------------------------------------------------------


def _line_offsets(src: str) -> list[int]:
    offsets = [0]
    for line in src.split("\n")[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _abs_span(src: str, node: ast.AST) -> tuple[int, int]:
    offsets = _line_offsets(src)
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _insert_import(src: str, names: set[str]) -> str:
    tree = ast.parse(src)
    existing: ast.ImportFrom | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == HELPER_MODULE:
            existing = stmt
            break
    if existing is not None:
        have = {a.name for a in existing.names}
        merged = sorted(have | names)
        line = f"from {HELPER_MODULE} import {', '.join(merged)}"
        start, end = _abs_span(src, existing)
        return src[:start] + line + src[end:]
    last_import_end = None
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            last_import_end = stmt.end_lineno
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # module docstring
        else:
            break
    if last_import_end is None:
        raise RuntimeError("no import block found to anchor the helper import")
    lines = src.split("\n")
    stmt_line = f"from {HELPER_MODULE} import {', '.join(sorted(names))}"
    lines.insert(last_import_end, stmt_line)
    lines.insert(last_import_end, "")
    return "\n".join(lines)


def rewrite_file(path: Path, sites: list[Site]) -> int:
    src = path.read_text(encoding="utf-8")
    edits = []
    names: set[str] = set()
    for s in sites:
        if s.kind == "A":
            names.add("issue_skill_text")
        elif s.kind in ("B", "C"):
            names.add("read_workflow_doc")
        else:
            continue
        start, end = _abs_span(src, s.node)
        edits.append((start, end, s.replacement))
    if not edits:
        return 0
    for start, end, repl in sorted(edits, reverse=True):
        src = src[:start] + repl + src[end:]
    src = _insert_import(src, names)
    path.write_text(src, encoding="utf-8")
    return len(edits)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: report)")
    ap.add_argument(
        "--audit", action="store_true", help="exit non-zero if any rewrite site is still pending"
    )
    args = ap.parse_args()

    files = sorted(p for p in TESTS_DIR.rglob("*.py") if p.name not in EXEMPT)
    all_sites: dict[Path, list[Site]] = {}
    for p in files:
        sites = analyze_file(p, p.read_text(encoding="utf-8"))
        if sites:
            all_sites[p] = sites

    n_a = n_b = n_c = n_refuse = n_note = 0
    for p, sites in sorted(all_sites.items()):
        for s in sites:
            rel = p.relative_to(REPO_ROOT)
            line = s.node.lineno
            if s.kind == "A":
                n_a += 1
                print(f"REWRITE A {rel}:{line}  {s.detail!r} -> {s.replacement}")
            elif s.kind == "B":
                n_b += 1
                print(f"REWRITE B {rel}:{line}  {s.detail!r} -> {s.replacement}")
            elif s.kind == "C":
                n_c += 1
                print(f"REWRITE C {rel}:{line}  {s.detail!r} -> {s.replacement}")
            elif s.kind == "REFUSE":
                n_refuse += 1
                print(f"REFUSE    {rel}:{line}  {s.detail}")
            else:
                n_note += 1
                print(f"NOTE      {rel}:{line}  {s.detail}")

    n_files = len({p for p, ss in all_sites.items() if any(s.kind in ("A", "B", "C") for s in ss)})
    print(
        f"\nsummary: {n_a} type-A + {n_b} type-B + {n_c} type-C rewrite sites across "
        f"{n_files} files; {n_refuse} refused; {n_note} notes"
    )

    if args.audit:
        return 1 if (n_a or n_b or n_c) else 0

    if args.apply:
        total = 0
        for p, sites in sorted(all_sites.items()):
            total += rewrite_file(p, sites)
        print(f"applied: {total} call-site rewrites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
