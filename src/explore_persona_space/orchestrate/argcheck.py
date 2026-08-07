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


def _permissive_dests(opts: list[str | None]) -> set[str]:
    """Permissive fallback: every resolvable option contributes its name.

    Used ONLY when the call's option strings are not all statically
    resolvable (see ``_add_argument_dests``); the runtime value could
    displace any constant as the dest, so every constant candidate enters
    the DEFINED set to avoid false positives on dynamically-built parsers.
    """
    out: set[str] = set()
    for opt in opts:
        if opt is None:
            continue
        if opt.startswith("-"):
            name = opt.lstrip("-").replace("-", "_")
            if name:
                out.add(name)
        else:  # positional: argparse keeps the name verbatim (no -/_ swap)
            out.add(opt)
    return out


def _add_argument_dests(call: ast.Call) -> set[str]:
    """DEFINED-set names contributed by one ``*.add_argument(...)`` call.

    An explicit constant ``dest=`` kwarg OVERRIDES the derived name
    (argparse semantics: ``dest`` replaces the derived attribute, so a read
    of the flag-derived name is a genuine runtime ``AttributeError``).
    Without one, the SINGLE dest is computed exactly the way argparse does
    (``_get_optional_kwargs`` / ``_get_positional_kwargs``, verified
    against the live interpreter):

    - option strings: the FIRST long option (``--foo`` -> ``foo``) wins;
      with no long option, the first option string (``-x`` -> ``x``);
      leading ``-`` stripped, then ``-`` -> ``_``. Only that ONE name is
      defined: ``add_argument("--new-name", "--old-name")`` defines
      ``new_name`` only, and ``args.old_name`` is a guaranteed runtime
      ``AttributeError`` — the round-1 all-candidates superset silently
      passed it (concern ``argcheck-exact-dest-derivation``). For constant
      option strings the dest IS statically computable; nothing needs
      ranking.
    - a bare positional keeps its name VERBATIM — argparse does NOT apply
      ``-`` -> ``_`` to positionals, so ``add_argument("src-dir")`` yields
      dest ``src-dir`` (never attribute-accessible; the underscore alias
      ``src_dir`` is a genuine AttributeError and stays flagged).

    Narrow permissive fallback — the ONE place superset behavior remains:
    when any leading positional arg is NOT a constant string (a
    dynamically-built parser: ``ap.add_argument(flag_var, ...)`` /
    ``ap.add_argument(*flags)``), the option-string set is not statically
    resolvable, so the runtime value could displace any constant as the
    dest; every constant that IS present then contributes its derived name
    (``_permissive_dests``) rather than raising — a dynamic parser must
    not turn into a false positive.
    """
    for kw in call.keywords:
        if (
            kw.arg == "dest"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return {kw.value.value}
    opts: list[str | None] = [
        arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
        for arg in call.args
    ]
    if not opts:
        return set()
    if any(o is None for o in opts):
        return _permissive_dests(opts)
    resolved = [o for o in opts if o is not None]  # == opts; narrows the type
    if len(resolved) == 1 and not resolved[0].startswith("-"):
        return {resolved[0]}  # positional: dest is the name VERBATIM
    long_opts = [o for o in resolved if o.startswith("--")]
    dash_opts = long_opts or [o for o in resolved if o.startswith("-")]
    if not dash_opts:
        # Malformed multi-positional call — argparse itself raises at
        # parser-build time, before any args read matters; contribute
        # nothing rather than guess.
        return set()
    name = dash_opts[0].lstrip("-").replace("-", "_")
    return {name} if name else set()


def assert_args_attributes_defined(
    *module_files: str | os.PathLike[str],
    extra_defined: Iterable[str] = (),
    namespace_names: Sequence[str] = ("args",),
) -> None:
    """Assert every ``args.<attr>`` Load in ``module_files`` has a definition.

    Builds the DEFINED set unioned across every passed file — argparse
    registrations (``add_argument`` exact-dest derivation per
    ``_add_argument_dests``, with an explicit constant ``dest=``
    overriding, ``add_subparsers(dest=...)``, ``set_defaults(**kw)`` kwarg
    names), Store-context attributes on a ``namespace_names`` receiver
    (``args.x = ...``), plus ``extra_defined`` — and the REFERENCED set
    from every Load- or Del-context ``ast.Attribute`` whose value is a
    ``namespace_names`` name (``del args.x`` requires the attribute to
    exist, exactly like a read), scanned over the WHOLE module (see the
    module docstring for why that scope is load-bearing).

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
                elif isinstance(node.ctx, ast.Load | ast.Del):
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
