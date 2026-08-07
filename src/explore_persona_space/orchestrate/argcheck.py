"""AST-based argparse-attribute completeness check for phase-dispatch drivers.

``assert_args_attributes_defined(*module_files)`` statically verifies that
every ``args.<attr>`` read in the given module file(s) is backed by an
argparse registration (``add_argument`` / an explicit ``dest=`` /
``add_subparsers(dest=...)`` / ``set_defaults(...)``) or by a runtime
assignment (``args.<attr> = ...``), and raises ``SystemExit`` naming every
gap otherwise. Phase-dispatch drivers call it from their ``--import-check``
mode (see ``.claude/rules/code-style.md`` section "Argparse-attribute
completeness for phase-dispatch drivers") so a never-smoked phase cannot
ship an ``AttributeError`` on a namespace attribute — the #2163 Step-8
crash class, where ``args.figures_out`` and ``args.harvest_out`` fired on
the two VM-side phases the smoke never exercised.

WHOLE-MODULE SCOPE (load-bearing — do NOT narrow):
    The referenced set is collected over the ENTIRE module — every
    function, helper, and module-level statement — never just the phase
    functions in a dispatch registry. #2163's first version scanned only
    the ``PHASES`` function bodies and missed ``args.figures_out``, which
    lives in ``_fig_dir``, a helper the phase calls. Any per-function scope
    is escapable by moving the reference one call deeper into a helper, so
    the whole-module (file) scope is the only non-escapable one. A future
    narrowing silently reintroduces that helper-escape hole;
    ``tests/test_argcheck.py::test_whole_module_scope_catches_helper_escape``
    pins the behavior so a narrowing is test-breaking, not just
    documented.

Known accepted false negative (AugAssign):
    ``args.x += 1`` has a Store-context target, so ``x`` lands in the
    DEFINED set while the same operation also LOADS a possibly-undefined
    attribute. Deliberately accepted — distinguishing it needs flow
    ordering, which is out of scope for a static completeness check.
    (Dynamic access such as ``getattr(args, name)`` similarly escapes any
    static scan.)
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["assert_args_attributes_defined"]


def _add_argument_dests(call: ast.Call) -> set[str]:
    """DEFINED-set names contributed by one ``*.add_argument(...)`` call.

    An explicit constant ``dest=`` kwarg OVERRIDES the flag-derived names
    (argparse semantics: ``dest`` replaces the derived attribute, so a read
    of the flag-derived name is a genuine runtime ``AttributeError``).
    Without one, every constant-string option/positional contributes its
    derived name — a deliberate small superset of argparse's single-dest
    rule (argparse picks ONE dest per call; deriving all candidates avoids
    false positives on multi-option calls like ``("-n", "--dry-run")``
    while a static scan cannot always rank them).
    """
    for kw in call.keywords:
        if (
            kw.arg == "dest"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return {kw.value.value}
    out: set[str] = set()
    for arg in call.args:
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            continue
        opt = arg.value
        if opt.startswith("--"):
            out.add(opt[2:].replace("-", "_"))
        elif opt.startswith("-"):
            out.add(opt[1:].replace("-", "_"))
        else:  # positional argument name
            out.add(opt.replace("-", "_"))
    return out


def assert_args_attributes_defined(
    *module_files: str | os.PathLike[str],
    extra_defined: Iterable[str] = (),
    namespace_names: Sequence[str] = ("args",),
) -> None:
    """Assert every ``args.<attr>`` Load in ``module_files`` has a definition.

    Builds the DEFINED set unioned across every passed file — argparse
    registrations (``add_argument`` first-positional flags with an explicit
    constant ``dest=`` overriding, ``add_subparsers(dest=...)``,
    ``set_defaults(**kw)`` kwarg names), Store-context attributes on a
    ``namespace_names`` receiver (``args.x = ...``), plus ``extra_defined``
    — and the REFERENCED set from every Load-context ``ast.Attribute``
    whose value is a ``namespace_names`` name, scanned over the WHOLE
    module (see the module docstring for why that scope is load-bearing).

    The varargs signature exists for imported parser-builders (measured FP
    class 4): a driver whose parser is partly built by
    ``shared_mod._add_common_args(ap)`` passes both files —
    ``assert_args_attributes_defined(__file__, inspect.getfile(shared_mod))``.
    Residual false positives (e.g. a non-namespace local named ``args``)
    route through ``extra_defined`` — visible at the call site, never
    silent.

    Raises ``SystemExit`` naming every missing attribute and the file(s)
    scanned; returns ``None`` when the referenced set is covered.
    """
    if not module_files:
        raise ValueError("assert_args_attributes_defined() needs at least one module file")
    files = [Path(f) for f in module_files]
    receivers = set(namespace_names)
    defined: set[str] = set(extra_defined)
    referenced: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                meth = node.func.attr
                if meth == "add_argument":
                    defined |= _add_argument_dests(node)
                elif meth == "add_subparsers":
                    for kw in node.keywords:
                        if (
                            kw.arg == "dest"
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ):
                            defined.add(kw.value.value)
                elif meth == "set_defaults":
                    defined |= {kw.arg for kw in node.keywords if kw.arg is not None}
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in receivers
            ):
                if isinstance(node.ctx, ast.Store):
                    defined.add(node.attr)
                elif isinstance(node.ctx, ast.Load):
                    referenced.add(node.attr)
    missing = sorted(referenced - defined)
    if missing:
        raise SystemExit(
            "argcheck: args attribute(s) referenced but never defined: "
            + ", ".join(missing)
            + "\n  scanned: "
            + ", ".join(str(p) for p in files)
            + "\n  fix: register via add_argument/dest=/set_defaults, pass the parser-builder"
            " module file(s) too, or name residue via extra_defined=(...)"
        )
