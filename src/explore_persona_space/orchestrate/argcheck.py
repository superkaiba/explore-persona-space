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

Splat (``**kwargs``) handling:
    A ``**`` splat keyword (``kw.arg is None``) is never silently skipped.
    ``add_argument(..., **kw)`` routes the call to the permissive fallback
    — the splat could carry ``dest``, strictly LESS resolvable than a
    present-but-non-constant ``dest=``, so it must not be handled more
    strictly. This is a widening WITHIN the existing dynamic-``dest=``
    false-negative class, not a new class: a splat without ``dest`` now
    also takes the permissive path, which accepts alias reads that are
    runtime ``AttributeError``s. ``set_defaults(**m)`` /
    ``add_subparsers(**m)`` statically resolve the mapping when it is an
    inline dict literal, or a module-level dict-literal constant under the
    EXCLUSIVE-USE rule (``_resolve_splat_mapping``: sole binding
    module-wide + every other occurrence of the name is itself a ``**``
    splat at a handled registration call); anything else ABSTAINS — the
    splat contributes nothing, the check keeps its teeth, and the failing
    ``SystemExit`` names each such splat site with an
    ``extra_defined=(...)`` pointer (the named residual false-positive
    class). Limitations: (a) with multiple ``module_files``, a co-passed
    file mutating ``driver.DEFAULTS`` via attribute access escapes the
    per-file-tree resolver (realistic only under a circular import);
    (b) a site-1 (``add_argument``) splat carrying an exotic ``dest``
    outside the option-derived candidate set still flags — the same
    residual as the dynamic-``dest=`` path, routed via ``extra_defined``;
    (c) reflective mutation (``globals()["DEFAULTS"].pop(...)``, ``exec``)
    and a module-level ``match`` capture rebinding the name via a
    string-field ``MatchAs`` (the ``ast.Match`` blind spot this repo's
    other scanners share) are invisible to the exclusive-use rule — the
    same family as the ``getattr(args, name)`` disclosure above; a
    ``from x import *`` anywhere in the module disqualifies dict-name
    resolution outright rather than being accepted as an escape (ruff
    F403 bans star-imports in-repo, so this costs nothing real).
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["assert_args_attributes_defined"]


def _permissive_dests(opts: list[str | None]) -> set[str]:
    """Permissive fallback: every resolvable option contributes its name.

    Used ONLY when the call is not fully statically resolvable — a
    non-constant option string, or a present-but-non-constant ``dest=``
    kwarg (see ``_add_argument_dests``); the runtime value could displace
    any constant as the dest, so every constant candidate enters the
    DEFINED set to avoid false positives on dynamically-built parsers.
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


_SPLAT_REGISTRATION_METHODS = frozenset({"add_argument", "set_defaults", "add_subparsers"})


def _dict_literal_mapping(node: ast.Dict) -> dict[str, ast.expr] | None:
    """Map a dict literal's constant-string keys to their value nodes, or None.

    Abstains (returns ``None``) when ANY key is not a constant string — a
    nested ``**`` inside the literal appears as a ``None`` key and abstains
    the same way. Duplicate keys keep the LAST value (dict-display /
    ``**``-merge runtime semantics).
    """
    out: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return None
        out[key.value] = value
    return out


def _resolve_splat_mapping(value: ast.expr, tree: ast.AST) -> dict[str, ast.expr] | None:
    """Statically resolve a ``**`` splat's mapping to {str key: value node}, or None.

    Resolves exactly two shapes: (a) an inline ``ast.Dict`` literal at the
    call site; (b) an ``ast.Name`` under the EXCLUSIVE-USE rule — its sole
    binding anywhere in the module is one module-level ``Assign`` (single
    Name target) / ``AnnAssign`` to an ``ast.Dict`` literal, AND every
    Load-context occurrence of the name anywhere in the module appears
    ONLY as the value of a ``**`` splat keyword (``ast.keyword(arg=None)``)
    on a handled registration call (``_SPLAT_REGISTRATION_METHODS``). Any
    other occurrence — an alias RHS (``B = DEFAULTS``), a call argument
    (``_finalize(DEFAULTS)``), a subscript or method receiver
    (``DEFAULTS[k]`` / ``DEFAULTS.pop(...)``), a ``del`` of the bare name —
    abstains. A ``from x import *`` anywhere in the module disqualifies
    shape (b) outright (a star-import can bind any name invisibly; ruff
    F403 bans star-imports in-repo, so the disqualification costs nothing
    real). Every key must be a constant string (a nested ``**`` inside the
    literal appears as a ``None`` key -> abstain). Anything else returns
    ``None`` — never a guess (the same abstain discipline as #2176's
    registry-member reader in task_workflow.py).
    """
    if isinstance(value, ast.Dict):
        return _dict_literal_mapping(value)
    if not isinstance(value, ast.Name):
        return None
    target = _exclusive_use_sole_binding(value.id, tree)
    if target is None:
        return None
    return _module_level_dict_literal(target, tree)


def _exclusive_use_sole_binding(name: str, tree: ast.AST) -> ast.AST | None:
    """The name's sole module-wide binding under the EXCLUSIVE-USE rule, or None.

    ONE ``ast.walk`` pass collects, by node identity, every binding of
    ``name`` (Store-context ``ast.Name`` — assignments, loop targets,
    walrus, comprehension targets; ``ast.arg`` function parameters, which
    catch local shadowing without scope analysis; ``ast.alias`` imports),
    every Load-context occurrence, any ``del`` of the bare name, any
    ``from x import *``, AND the sanctioned node set — the ``ast.keyword
    (arg=None)`` splat values on calls whose ``func`` is an
    ``ast.Attribute`` with ``attr`` in ``_SPLAT_REGISTRATION_METHODS``
    (receiver-agnostic, matching the handlers' own dispatch). Returns the
    single binding node iff exactly ONE binding exists, no ``del`` / no
    star-import exists, and every Load is a member of the sanctioned set;
    ``None`` otherwise.
    """
    bindings: list[ast.AST] = []
    loads: list[ast.Name] = []
    sanctioned: set[int] = set()
    star_import = False
    has_del = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Store):
                bindings.append(node)
            elif isinstance(node.ctx, ast.Del):
                has_del = True
            else:  # Load
                loads.append(node)
        elif isinstance(node, ast.arg) and node.arg == name:
            bindings.append(node)
        elif isinstance(node, ast.alias):
            if node.name == "*":
                star_import = True
            elif (node.asname or node.name.split(".")[0]) == name:
                bindings.append(node)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SPLAT_REGISTRATION_METHODS
        ):
            for kw in node.keywords:
                if kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id == name:
                    sanctioned.add(id(kw.value))
    if star_import or has_del or len(bindings) != 1:
        return None
    if any(id(load) not in sanctioned for load in loads):
        return None
    return bindings[0]


def _module_level_dict_literal(target: ast.AST, tree: ast.AST) -> dict[str, ast.expr] | None:
    """The dict-literal mapping bound by the module-level statement owning ``target``.

    ``target`` is the sole binding node ``_exclusive_use_sole_binding``
    returned; resolves only a module-level ``Assign`` (single Name target,
    identical node) / ``AnnAssign`` whose value is an ``ast.Dict`` with
    all-constant-string keys. Returns ``None`` for anything else (function
    parameter, import, loop target, non-dict RHS, bare annotation, ...).
    """
    if not isinstance(tree, ast.Module):
        return None
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and stmt.targets[0] is target
            and isinstance(stmt.value, ast.Dict)
        ):
            return _dict_literal_mapping(stmt.value)
        if (
            isinstance(stmt, ast.AnnAssign)
            and stmt.target is target
            and isinstance(stmt.value, ast.Dict)
        ):
            return _dict_literal_mapping(stmt.value)
    return None


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
    when the call is not fully statically resolvable, in ANY of three
    shapes — (a) any leading positional arg is NOT a constant string (a
    dynamically-built parser: ``ap.add_argument(flag_var, ...)`` /
    ``ap.add_argument(*flags)``), (b) a ``dest=`` kwarg is PRESENT but
    not a constant string (``ap.add_argument("-n", "--dry-run",
    dest=some_var)`` — the runtime dest is unknowable, strictly LESS
    resolvable than a missing ``dest=``, so it must not be handled more
    strictly), or (c) a ``**`` splat is present (``kw.arg is None``): it
    could carry ``dest``, strictly less resolvable than shape (b)
    — the runtime value could displace any constant as the
    dest; every constant option string that IS present then contributes
    its derived name (``_permissive_dests``) rather than raising — a
    dynamic parser must not turn into a false positive. A constant
    ``dest=None`` stays on the exact path: argparse treats it as "derive
    the default dest".
    """
    dynamic_dest = False
    for kw in call.keywords:
        if kw.arg is None:
            # ** splat: could carry dest= — strictly LESS resolvable than a
            # present-but-non-constant dest=, so it must not be handled more
            # strictly (the c4f681716b principle): take the permissive path.
            dynamic_dest = True
            continue
        if kw.arg != "dest":
            continue
        if isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str):
                return {kw.value.value}
            # dest=None (or another non-str constant) IS statically
            # resolvable: argparse treats None as "derive the default
            # dest", which the exact derivation below implements.
            continue
        # Present-but-non-constant dest (dest=SOME_VAR / dest=f(...)):
        # the runtime dest is unknowable — strictly LESS resolvable than
        # a missing dest= — so the call takes the same permissive path
        # as non-constant option strings (round-2 review NIT).
        dynamic_dest = True
    opts: list[str | None] = [
        arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
        for arg in call.args
    ]
    if not opts:
        return set()
    if dynamic_dest or any(o is None for o in opts):
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


def _subparsers_defined(
    call: ast.Call,
    tree: ast.AST,
    path: Path,
    splat_sites: list[tuple[Path, int, str]],
) -> set[str]:
    """DEFINED-set names contributed by one ``*.add_subparsers(...)`` call.

    An explicit constant-string ``dest=`` defines that name; a ``**``
    splat whose mapping resolves (``_resolve_splat_mapping``) defines its
    constant-string ``dest`` value when one is present — a resolved
    mapping WITHOUT one contributes nothing, parity with the explicit
    non-constant ``dest=`` case (and with argparse's default:
    ``add_subparsers()`` without ``dest`` stores no attribute, verified
    against the live interpreter). An UNRESOLVED splat contributes
    nothing and is appended to ``splat_sites`` for the failure diagnostic.
    """
    defined: set[str] = set()
    for kw in call.keywords:
        if (
            kw.arg == "dest"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            defined.add(kw.value.value)
        elif kw.arg is None:  # ** splat
            mapping = _resolve_splat_mapping(kw.value, tree)
            if mapping is None:
                splat_sites.append((path, call.lineno, "add_subparsers"))
            else:
                dest_value = mapping.get("dest")
                if isinstance(dest_value, ast.Constant) and isinstance(dest_value.value, str):
                    defined.add(dest_value.value)
    return defined


def _set_defaults_defined(
    call: ast.Call,
    tree: ast.AST,
    path: Path,
    splat_sites: list[tuple[Path, int, str]],
) -> set[str]:
    """DEFINED-set names contributed by one ``*.set_defaults(...)`` call.

    Every explicit keyword name defines itself; a ``**`` splat whose
    mapping resolves (``_resolve_splat_mapping``) defines every resolved
    key. An UNRESOLVED splat contributes nothing and is appended to
    ``splat_sites`` for the failure diagnostic.
    """
    defined: set[str] = {kw.arg for kw in call.keywords if kw.arg is not None}
    for kw in call.keywords:
        if kw.arg is None:  # ** splat
            mapping = _resolve_splat_mapping(kw.value, tree)
            if mapping is None:
                splat_sites.append((path, call.lineno, "set_defaults"))
            else:
                defined |= set(mapping)
    return defined


def assert_args_attributes_defined(
    *module_files: str | os.PathLike[str],
    extra_defined: Iterable[str] = (),
    namespace_names: Sequence[str] = ("args",),
) -> None:
    """Assert every ``args.<attr>`` Load in ``module_files`` has a definition.

    Builds the DEFINED set unioned across every passed file — argparse
    registrations (``add_argument`` exact-dest derivation per
    ``_add_argument_dests``, with an explicit constant ``dest=``
    overriding and a ``**`` splat routing the call to the permissive
    fallback; ``add_subparsers(dest=...)``; ``set_defaults(...)`` keyword
    names — a ``**`` splat on ``set_defaults`` / ``add_subparsers``
    contributes its statically-resolved keys per ``_resolve_splat_mapping``
    (for ``add_subparsers`` only a resolved constant-string ``dest``) and
    contributes NOTHING when unresolved), Store-context attributes on a
    ``namespace_names`` receiver (``args.x = ...``), plus ``extra_defined``
    — and the REFERENCED set from every Load- or Del-context
    ``ast.Attribute`` whose value is a ``namespace_names`` name (``del
    args.x`` requires the attribute to exist, exactly like a read),
    scanned over the WHOLE module (see the module docstring for why that
    scope is load-bearing).

    The varargs signature exists for imported parser-builders (measured FP
    class 4): a driver whose parser is partly built by
    ``shared_mod._add_common_args(ap)`` passes both files —
    ``assert_args_attributes_defined(__file__, inspect.getfile(shared_mod))``.
    Residual false positives (e.g. a non-namespace local named ``args``)
    route through ``extra_defined`` — visible at the call site, never
    silent.

    Raises ``SystemExit`` naming every missing attribute and the file(s)
    scanned; returns ``None`` when the referenced set is covered. When the
    check fails and splat calls whose keys were not statically incorporated
    exist (an unresolved ``set_defaults`` / ``add_subparsers`` splat, or
    ANY splat-bearing ``add_argument`` — permissive routing can still miss
    a splat-carried exotic ``dest``), the message additionally names each
    such site as ``<file>:<lineno> <method>`` with an
    ``extra_defined=(...)`` pointer.
    """
    if not module_files:
        raise ValueError("assert_args_attributes_defined() needs at least one module file")
    files = [Path(f) for f in module_files]
    receivers = set(namespace_names)
    defined: set[str] = set(extra_defined)
    referenced: set[str] = set()
    splat_sites: list[tuple[Path, int, str]] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                meth = node.func.attr
                if meth == "add_argument":
                    defined |= _add_argument_dests(node)
                    if any(kw.arg is None for kw in node.keywords):
                        # Permissive routing can still miss a splat-carried
                        # exotic dest — keep the site as a diagnostic
                        # "likely cause" candidate for the failure message.
                        splat_sites.append((path, node.lineno, meth))
                elif meth == "add_subparsers":
                    defined |= _subparsers_defined(node, tree, path, splat_sites)
                elif meth == "set_defaults":
                    defined |= _set_defaults_defined(node, tree, path, splat_sites)
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
        splat_note = ""
        if splat_sites:
            sites = ", ".join(f"{p}:{lineno} {meth}" for p, lineno, meth in splat_sites)
            splat_note = (
                "\n  note: '**' splat call(s) whose keys are not statically incorporated: "
                + sites
                + " — if a missing attribute arrives via the splat, name it in"
                " extra_defined=(...)"
            )
        raise SystemExit(
            "argcheck: args attribute(s) referenced but never defined: "
            + ", ".join(missing)
            + "\n  scanned: "
            + ", ".join(str(p) for p in files)
            + splat_note
            + "\n  fix: register via add_argument/dest=/set_defaults, pass the parser-builder"
            " module file(s) too, or name residue via extra_defined=(...)"
        )
