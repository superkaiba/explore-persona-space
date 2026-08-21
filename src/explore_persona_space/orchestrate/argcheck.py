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

CALL-ARITY BIND PASS (#2261):
    ``assert_helper_call_shapes_bind(*module_files)`` — invoked
    automatically as the LAST step of ``assert_args_attributes_defined``
    (argparse gaps raise first), so every existing ``--import-check``
    adopter is armed with zero driver edits — statically resolves every
    call in the passed files into the registered shared-callable surface
    (``_BIND_MODULES`` / ``_BIND_CLASSES`` / ``_BIND_FUNCTIONS``) and
    binds each literal call shape against the installed signature via
    ``inspect.signature(fn).bind(...)`` with placeholder values: shape
    only (positional count, keyword names, required params,
    keyword-only-ness), values never evaluated, no network. #2223's
    ``hub._upload`` call missing ``path_in_repo`` shipped green through
    every gate and fired ~70 h into a GPU run; this pass fails it in
    seconds at ``--import-check``. Non-raising census API:
    ``collect_helper_call_census(*module_files) -> BindCensus``.

    Name resolution is UNIFORM-IMPORT-TARGET (not sole-binding): a name
    resolves iff EVERY module-wide binding of it is an import alias
    normalizing to the SAME target — module-level, function-local, and
    REPEATED same-target imports all resolve (the fleet's dominant idiom;
    the #2223 pre-fix driver bound ``hub`` three times); a genuine shadow
    (any non-import binding, a ``del``, or imports of two DIFFERENT
    targets under one name) makes every registered-surface call through
    that name a DETECTED-AND-NOTED SKIP, never a silent pass. Resolved
    shapes: S1 module-attribute (``hub.attr(...)``; a nonexistent
    attribute on a registered module is a check FAILURE — the #606
    class — never a skip), S2 bare-name (registered functions + symbols
    imported from a registered module), S3 inline constructor-method
    (``HfApi().m(...)``, one cached instance per class per invocation),
    S4 variable-receiver (``api = HfApi()`` under the uniform-binding
    rule). A call-site splat DEGRADES to ``bind_partial(**named_kwargs)``
    (census-noted). Un-introspectable callables and
    ``getattr(mod, ...)(...)`` immediate calls are noted skips.

    Waiver: ``# ARGCHECK_BIND_EXEMPT: <reason>`` on the call line or the
    immediately-preceding non-blank COMMENT-ONLY line converts a would-be
    failure into a noted, reason-echoed waiver (the comment-only
    restriction keeps a trailing waiver on a CODE line from leaking onto
    the NEXT line's call). LINE-grained — it suppresses ONLY calls whose
    ``node.lineno`` matches, so keep waived calls on their own line: two
    calls on ONE physical line share a lineno and are waived together
    (live example of the shape: ``scripts/issue2225_eval_gen.py:675``),
    and a formatter-expanded multiline call keeps ``node.lineno`` at its
    OPENING line — place the waiver there.

    Documented out-of-scope FALSE-NEGATIVE classes (visible by design):
    FN-1 ``functools.partial(hub.X, ...)`` deferred application — the
         helper is an argument, not a Call target.
    FN-2 a helper whose own signature is ``(*args, **kwargs)`` binds any
         shape (none on the registered surface today).
    FN-3 runtime monkeypatching of a registered helper.
    FN-4 check-env vs run-env version drift (same ``uv.lock`` on the VM
         and pods keeps this small).
    FN-5 sibling-module call sites: only the FILES PASSED to the check
         are scanned — a driver whose Hub calls live in an imported
         sibling module (live example: ``scripts/issue2203_phase1.py``
         passes only ``__file__`` while its Hub calls live in
         ``scripts/issue2203_common.py``) is not covered for those sites
         unless the sibling is itself an adopter or is co-passed.
    FN-6 attribute-chain receivers (``self.api = HfApi()`` then
         ``self.api.upload_file(...)``; dotted chains after a bare
         ``import a.b.c``) — the receiver is an ``ast.Attribute``, not a
         Name: inert, neither bound nor skip-noted. Measured 0 fleet-wide.
    FN-7 out-of-registry in-repo wrappers (per-issue helpers wrapping the
         registered surface, e.g. ``R.upload_dir_hf``) are invisible: the
         registry validates the surfaces it selected, not the whole
         persistence-critical class.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = [
    "BindCensus",
    "assert_args_attributes_defined",
    "assert_helper_call_shapes_bind",
    "collect_helper_call_census",
]


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

    After the argparse pass succeeds, the call-arity bind pass runs over
    the SAME files (``assert_helper_call_shapes_bind``) — argparse gaps
    raise FIRST; see the module docstring section "CALL-ARITY BIND PASS
    (#2261)".

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
    assert_helper_call_shapes_bind(*module_files)


# ---------------------------------------------------------------------------
# Call-arity bind pass (#2261) — see the module docstring section
# "CALL-ARITY BIND PASS" for the contract + FN-1..FN-7.
# ---------------------------------------------------------------------------

# NOTE: the registry constants are read at CALL time (module-global lookup
# inside the helpers below), never captured at import/default time — this is
# what lets tests monkeypatch an entry away and observe the corresponding
# must-FAIL fixture flip to inert (test_bind_registry_entries_load_bearing).
_BIND_MODULES: frozenset[str] = frozenset(
    {
        # every hub.X(...) + every `from ...hub import X` bare call binds
        "explore_persona_space.orchestrate.hub",
    }
)
_BIND_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        # inline HfApi().m(...) + uniform-binding var receivers (api = HfApi())
        ("huggingface_hub", "HfApi"),
    }
)
_BIND_FUNCTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("huggingface_hub", "hf_hub_download"),
        ("huggingface_hub", "snapshot_download"),
    }
)
_BIND_WAIVER_TOKEN = "ARGCHECK_BIND_EXEMPT"

_PLACEHOLDER = object()  # bind() checks SHAPE only; values are never evaluated
_MISSING = object()


@dataclasses.dataclass(frozen=True)
class BindSite:
    """One registered-surface call site the bind pass resolved."""

    path: str
    lineno: int
    label: str  # source-written form, e.g. "hub._upload" / "HfApi().upload_file"
    shape: str  # "S1" module-attr | "S2" bare-name | "S3" inline ctor | "S4" var receiver
    target: str  # canonical dotted target, e.g. "explore_persona_space.orchestrate.hub._upload"


@dataclasses.dataclass(frozen=True)
class BindSkip:
    """One detected-and-noted skip (not statically resolvable — never silent)."""

    path: str
    lineno: int
    label: str
    reason: str


@dataclasses.dataclass(frozen=True)
class BindFailure:
    """One call shape that does not bind against the installed signature."""

    site: BindSite
    error: str
    installed: str | None  # canonical target + installed signature; None when unresolvable


@dataclasses.dataclass(frozen=True)
class BindWaiver:
    """A would-be failure suppressed by an ARGCHECK_BIND_EXEMPT call-line waiver."""

    site: BindSite
    error: str
    reason: str


@dataclasses.dataclass
class BindCensus:
    """Aggregated result of the non-raising collection pass."""

    bound: list[BindSite] = dataclasses.field(default_factory=list)
    degraded: list[BindSite] = dataclasses.field(default_factory=list)
    skipped: list[BindSkip] = dataclasses.field(default_factory=list)
    waived: list[BindWaiver] = dataclasses.field(default_factory=list)
    failures: list[BindFailure] = dataclasses.field(default_factory=list)
    n_files: int = 0


@dataclasses.dataclass
class _NameBindings:
    """Every module-wide binding event of one local name (the shadow-guard input)."""

    import_targets: set[tuple[str, ...]] = dataclasses.field(default_factory=set)
    assign_values: list[ast.expr] = dataclasses.field(default_factory=list)
    n_other: int = 0  # params, def/class names, loop targets, walrus, tuple-unpack, ...
    has_del: bool = False


def _normalize_import_from(node: ast.ImportFrom, alias: ast.alias) -> tuple[str, ...]:
    """Normalize one ``from a.b import c [as d]`` alias to a target tuple.

    Absolute imports only — a relative import (``node.level > 0``) or a
    module-less ``from`` normalizes to ``("unmapped",)``, which is never
    registered (calls through such a name stay inert unless it conflicts
    with a registered binding). Classification order: registered module,
    registered class, registered function, symbol-of-registered-module,
    then a generic ``("from", mod, name)`` for everything else.
    """
    if node.level or node.module is None:
        return ("unmapped",)
    mod, name = node.module, alias.name
    dotted = f"{mod}.{name}"
    if dotted in _BIND_MODULES:
        return ("module", dotted)
    if (mod, name) in _BIND_CLASSES:
        return ("class", mod, name)
    if (mod, name) in _BIND_FUNCTIONS:
        return ("function", mod, name)
    if mod in _BIND_MODULES:
        return ("symbol", mod, name)
    return ("from", mod, name)


def _registered_target(target: tuple[str, ...]) -> bool:
    """True iff the normalized target tuple points at the registered surface."""
    kind = target[0]
    if kind == "module":
        return target[1] in _BIND_MODULES
    if kind == "class":
        return (target[1], target[2]) in _BIND_CLASSES
    if kind == "function":
        return (target[1], target[2]) in _BIND_FUNCTIONS
    if kind == "symbol":
        return target[1] in _BIND_MODULES
    return False


def _assign_value_map(tree: ast.AST) -> dict[int, ast.expr]:
    """Map each Name-target node's ``id()`` to its Assign/AnnAssign value."""
    out: dict[int, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[id(tgt)] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            out[id(node.target)] = node.value
    return out


def _record_import_bindings(node: ast.Import | ast.ImportFrom, rec) -> bool:
    """Record one import statement's alias bindings; True iff a star-import was seen."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.asname:
                rec(alias.asname).import_targets.add(("module", alias.name))
            else:
                top = alias.name.split(".")[0]
                rec(top).import_targets.add(("module", top))
        return False
    star = False
    for alias in node.names:
        if alias.name == "*":
            star = True
            continue
        bound = alias.asname or alias.name
        rec(bound).import_targets.add(_normalize_import_from(node, alias))
    return star


def _collect_name_bindings(tree: ast.AST) -> tuple[dict[str, _NameBindings], bool]:
    """Collect every binding event per name over the WHOLE module, plus a star-import flag.

    Collector classes match ``_exclusive_use_sole_binding`` (Store-context
    ``ast.Name``, ``ast.arg`` parameters, ``ast.alias`` imports, ``del``
    tracked) plus ``FunctionDef`` / ``AsyncFunctionDef`` / ``ClassDef``
    names. The DECISION differs (uniform-import-target, not sole-binding):
    see ``_resolve_import_name``.
    """
    assign_value_by_target = _assign_value_map(tree)
    out: dict[str, _NameBindings] = {}
    star_import = False

    def rec(name: str) -> _NameBindings:
        return out.setdefault(name, _NameBindings())

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                b = rec(node.id)
                value = assign_value_by_target.get(id(node))
                if value is not None:
                    b.assign_values.append(value)
                else:
                    b.n_other += 1
            elif isinstance(node.ctx, ast.Del):
                rec(node.id).has_del = True
        elif isinstance(node, ast.arg):
            rec(node.arg).n_other += 1
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            rec(node.name).n_other += 1
        elif isinstance(node, ast.Import | ast.ImportFrom):
            star_import = _record_import_bindings(node, rec) or star_import
    return out, star_import


def _resolve_import_name(
    name: str, bindings: dict[str, _NameBindings], star_import: bool
) -> tuple[str, tuple[str, ...] | None, str | None]:
    """UNIFORM-IMPORT-TARGET resolution of one local name.

    Returns ``(status, target, reason)`` with status in ``{"inert", "skip",
    "resolved"}``. A name resolves iff EVERY module-wide binding of it is an
    import alias normalizing to the SAME target — module-level,
    function-local, and REPEATED same-target imports all resolve (the
    fleet's dominant idiom; the #2223 pre-fix driver bound ``hub`` three
    times). A GENUINE shadow — any non-import binding, a ``del``, or imports
    of two DIFFERENT targets under one name — disqualifies the name: every
    registered-surface call through it becomes a NOTED SKIP, never silent.
    Names whose targets never touch the registered surface stay inert.
    """
    b = bindings.get(name)
    if b is None or not b.import_targets:
        return ("inert", None, None)
    if not any(_registered_target(t) for t in b.import_targets):
        return ("inert", None, None)
    if star_import:
        return ("skip", None, f"name '{name}' resolution disqualified by a star-import")
    if b.assign_values or b.n_other or b.has_del:
        return ("skip", None, f"name '{name}' carries a non-import binding (genuine shadow)")
    if len(b.import_targets) > 1:
        targets = ", ".join(sorted(".".join(t[1:]) or t[0] for t in b.import_targets))
        return ("skip", None, f"name '{name}' import targets conflict ({targets})")
    return ("resolved", next(iter(b.import_targets)), None)


def _resolve_receiver_var(
    name: str, bindings: dict[str, _NameBindings], star_import: bool
) -> tuple[str, tuple[str, str] | None, str | None]:
    """S4 UNIFORM-BINDING resolution of a variable receiver (``api = HfApi()``).

    Resolves iff every module-wide binding of the name is an ``Assign`` /
    ``AnnAssign`` whose value is a direct ``Call`` to a registered class
    name that itself passes the shadow guard. Some-but-not-all qualifying
    bindings (a parameter, a rebind, a second class) are a NOTED SKIP; zero
    qualifying bindings are inert (an arbitrary object receiver).
    """
    b = bindings.get(name)
    if b is None or not b.assign_values:
        return ("inert", None, None)
    ctor_classes: set[tuple[str, str]] = set()
    qualifying = 0
    for value in b.assign_values:
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            status, target, _ = _resolve_import_name(value.func.id, bindings, star_import)
            if status == "resolved" and target is not None and target[0] == "class":
                ctor_classes.add((target[1], target[2]))
                qualifying += 1
    if qualifying == 0:
        return ("inert", None, None)
    uniform = (
        qualifying == len(b.assign_values)
        and not b.import_targets
        and b.n_other == 0
        and not b.has_del
        and len(ctor_classes) == 1
    )
    if not uniform:
        cls = sorted(ctor_classes)[0][1]
        return ("skip", None, f"receiver '{name}' bindings not uniformly {cls}()")
    return ("resolved", next(iter(ctor_classes)), None)


def _classify_getattr_immediate_call(
    func: ast.Call, bindings: dict[str, _NameBindings], star_import: bool
) -> tuple[str, ...] | None:
    """Classify a call whose func is itself a Call: ``getattr(mod, ...)(...)``.

    A ``getattr(<registered module alias>, <expr>)(...)`` immediate call is a
    DETECTED-AND-NOTED SKIP (not statically resolvable); every other
    call-of-a-call shape is inert (``None``).
    """
    if not (isinstance(func.func, ast.Name) and func.func.id == "getattr"):
        return None
    if func.args and isinstance(func.args[0], ast.Name):
        status, target, _ = _resolve_import_name(func.args[0].id, bindings, star_import)
        if status == "resolved" and target is not None and target[0] == "module":
            label = f"getattr({func.args[0].id}, ...)"
            return ("skip", label, "getattr(...) immediate call is not statically resolvable")
    return None


def _classify_call(
    node: ast.Call, bindings: dict[str, _NameBindings], star_import: bool
) -> tuple[str, ...] | None:
    """Classify one Call node against the registered surface.

    Returns ``("bind", label, shape, fetch)`` for a resolvable registered
    call, ``("skip", label, reason)`` for a detected-and-noted skip, or
    ``None`` for an inert (unregistered) call. ``fetch`` is one of
    ``("modattr", module, attr)`` / ``("symbol", module, name)`` /
    ``("method", module, cls, attr)``.
    """
    func = node.func
    if isinstance(func, ast.Call):
        return _classify_getattr_immediate_call(func, bindings, star_import)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        recv, attr = func.value.id, func.attr
        label = f"{recv}.{attr}"
        status, target, reason = _resolve_import_name(recv, bindings, star_import)
        if status == "skip":
            return ("skip", label, reason)
        if status == "resolved":
            if target is not None and target[0] == "module":
                return ("bind", label, "S1", ("modattr", target[1], attr))
            return None  # attribute call on a resolved class/function/symbol: not a shape
        status4, cls, reason4 = _resolve_receiver_var(recv, bindings, star_import)
        if status4 == "skip":
            return ("skip", label, reason4)
        if status4 == "resolved" and cls is not None:
            return ("bind", label, "S4", ("method", cls[0], cls[1], attr))
        return None
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Call)
        and isinstance(func.value.func, ast.Name)
    ):
        cls_name, attr = func.value.func.id, func.attr
        label = f"{cls_name}().{attr}"
        status, target, reason = _resolve_import_name(cls_name, bindings, star_import)
        if status == "skip":
            return ("skip", label, reason)
        if status == "resolved" and target is not None and target[0] == "class":
            return ("bind", label, "S3", ("method", target[1], target[2], attr))
        return None
    if isinstance(func, ast.Name):
        status, target, reason = _resolve_import_name(func.id, bindings, star_import)
        if status == "skip":
            return ("skip", func.id, reason)
        if status == "resolved" and target is not None and target[0] in ("function", "symbol"):
            return ("bind", func.id, "S2", ("symbol", target[1], target[2]))
        return None
    return None


def _fetch_callable(
    fetch: tuple[str, ...], instances: dict[tuple[str, str], object]
) -> tuple[object, str | None, str]:
    """Resolve a fetch spec to the installed callable.

    Returns ``(fn, error, canonical)``. A registered module/class whose
    attribute is ABSENT returns ``(None, <error>, canonical)`` — the #2223 /
    #606 fail-loud getattr: a nonexistent helper is a check FAILURE, never a
    silent skip (and never swallowed into a diagnostics-only channel).
    """
    kind = fetch[0]
    if kind in ("modattr", "symbol"):
        _, mod_name, attr = fetch
        canonical = f"{mod_name}.{attr}"
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr, _MISSING)
        if fn is _MISSING:
            error = f"references nonexistent helper (module '{mod_name}' has no attribute '{attr}')"
            return (None, error, canonical)
        return (fn, None, canonical)
    _, mod_name, cls_name, attr = fetch
    canonical = f"{mod_name}.{cls_name}.{attr}"
    key = (mod_name, cls_name)
    instance = instances.get(key)
    if instance is None:
        cls = getattr(importlib.import_module(mod_name), cls_name)
        instance = cls()  # one cached instance per registered class per invocation
        instances[key] = instance
    fn = getattr(instance, attr, _MISSING)
    if fn is _MISSING:
        error = (
            f"references nonexistent method (class '{mod_name}.{cls_name}'"
            f" has no attribute '{attr}')"
        )
        return (None, error, canonical)
    return (fn, None, canonical)


def _bind_call_shape(fn: object, node: ast.Call) -> tuple[str, str | None]:
    """Bind the literal call shape against the installed signature.

    Returns ``(verdict, detail)`` with verdict in ``{"ok", "degraded-ok",
    "fail", "degraded-fail", "unintrospectable"}``. Placeholders check SHAPE
    only (positional count, keyword names, required params,
    keyword-only-ness) — values are never evaluated, no network, no side
    effects. A call-site splat (``*args`` positional or ``**kwargs``)
    DEGRADES to ``bind_partial(**named_kwargs)`` per the #606/#1332
    doctrine; keyword-only params and callee-side ``*args``/``**kwargs`` are
    handled natively by ``Signature.bind`` (zero FP by construction).
    """
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        return ("unintrospectable", str(exc))
    kwargs = {kw.arg: _PLACEHOLDER for kw in node.keywords if kw.arg is not None}
    has_splat = any(isinstance(a, ast.Starred) for a in node.args) or any(
        kw.arg is None for kw in node.keywords
    )
    if has_splat:
        try:
            sig.bind_partial(**kwargs)
        except TypeError as exc:
            return ("degraded-fail", str(exc))
        return ("degraded-ok", None)
    try:
        sig.bind(*([_PLACEHOLDER] * len(node.args)), **kwargs)
    except TypeError as exc:
        return ("fail", str(exc))
    return ("ok", None)


def _waiver_reason(lines: list[str], lineno: int) -> str | None:
    """The ARGCHECK_BIND_EXEMPT reason covering ``lineno``, or None.

    LINE-grained: the token on the call's own source line, or on the
    immediately-preceding non-blank COMMENT-ONLY line (the
    ``# WANDB_INTENTIONALLY_DISABLED`` placement convention). The
    comment-only restriction on the preceding-line arm is load-bearing: a
    trailing waiver on a CODE line covers that line only — it must never
    leak onto the NEXT line's call. Suppresses ONLY calls whose
    ``node.lineno`` matches — never the rest of the file.
    """

    def token_reason(text: str) -> str | None:
        if _BIND_WAIVER_TOKEN not in text:
            return None
        after = text.split(_BIND_WAIVER_TOKEN, 1)[1]
        return after.lstrip(":").strip() or "(no reason given)"

    if 1 <= lineno <= len(lines):
        reason = token_reason(lines[lineno - 1])
        if reason is not None:
            return reason
    j = lineno - 1
    while j >= 1 and not lines[j - 1].strip():
        j -= 1
    if j >= 1 and lines[j - 1].lstrip().startswith("#"):
        return token_reason(lines[j - 1])
    return None


def _record_failure(
    census: BindCensus, site: BindSite, error: str, installed: str | None, lines: list[str]
) -> None:
    """Route a bind failure to failures, or to waived when the call line carries the token."""
    reason = _waiver_reason(lines, site.lineno)
    if reason is not None:
        census.waived.append(BindWaiver(site, error, reason))
    else:
        census.failures.append(BindFailure(site, error, installed))


def collect_helper_call_census(*module_files: str | os.PathLike[str]) -> BindCensus:
    """Non-raising collection pass: resolve + bind every registered shared-helper call.

    Scans every ``ast.Call`` in ``module_files`` (whole-module scope —
    lambdas, comprehensions, and nested defs included via ``ast.walk``),
    resolves calls into the registered surface (``_BIND_MODULES`` /
    ``_BIND_CLASSES`` / ``_BIND_FUNCTIONS``) under the uniform-import-target
    rule, and returns per-site lists: bound, degraded (call-site splat via
    ``bind_partial``), skipped (noted, never silent), waived
    (``ARGCHECK_BIND_EXEMPT``), failures. Target imports are LAZY — a file
    whose import map never references a registered target imports nothing.
    """
    if not module_files:
        raise ValueError("collect_helper_call_census() needs at least one module file")
    files = [Path(f) for f in module_files]
    census = BindCensus(n_files=len(files))
    instances: dict[tuple[str, str], object] = {}
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.split("\n")
        bindings, star_import = _collect_name_bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            verdict = _classify_call(node, bindings, star_import)
            if verdict is None:
                continue
            if verdict[0] == "skip":
                _, label, reason = verdict
                census.skipped.append(BindSkip(str(path), node.lineno, label, reason))
                continue
            _, label, shape, fetch = verdict
            fn, fetch_error, canonical = _fetch_callable(fetch, instances)
            site = BindSite(str(path), node.lineno, label, shape, canonical)
            if fetch_error is not None:
                _record_failure(census, site, fetch_error, None, lines)
                continue
            bind_verdict, detail = _bind_call_shape(fn, node)
            if bind_verdict == "unintrospectable":
                census.skipped.append(
                    BindSkip(
                        str(path),
                        node.lineno,
                        label,
                        f"un-introspectable callable ({detail})",
                    )
                )
            elif bind_verdict == "ok":
                census.bound.append(site)
            elif bind_verdict == "degraded-ok":
                census.degraded.append(site)
            else:  # "fail" | "degraded-fail"
                installed = f"{canonical}{inspect.signature(fn)}"  # type: ignore[arg-type]
                _record_failure(census, site, detail or "", installed, lines)
    return census


def assert_helper_call_shapes_bind(*module_files: str | os.PathLike[str]) -> None:
    """Assert every registered shared-helper call in ``module_files`` binds.

    Thin raise-on-failure wrapper over ``collect_helper_call_census``:
    prints the one-line census (bound/degraded/skipped) plus one line per
    skipped/degraded/waived site on EVERY run (pass and fail), then raises
    ``SystemExit`` naming every non-binding site (all failures across all
    files collected into ONE exit, mirroring the argparse pass).
    """
    census = collect_helper_call_census(*module_files)
    print(
        f"argcheck-bind: {len(census.bound)} bound, {len(census.degraded)} degraded,"
        f" {len(census.skipped)} skipped across {census.n_files} file(s)"
    )
    for skip in census.skipped:
        print(f"  skipped: {skip.path}:{skip.lineno} {skip.label} — {skip.reason}")
    for site in census.degraded:
        print(
            f"  degraded: {site.path}:{site.lineno} {site.label}"
            " — call-site splat; checked via bind_partial(**named_kwargs)"
        )
    for waiver in census.waived:
        print(
            f"  waived: {waiver.site.path}:{waiver.site.lineno} {waiver.site.label}"
            f" — {waiver.error} ({_BIND_WAIVER_TOKEN}: {waiver.reason})"
        )
    if census.failures:
        parts = ["argcheck: helper call shape(s) do not bind against installed signatures:"]
        for failure in census.failures:
            parts.append(
                f"  {failure.site.path}:{failure.site.lineno} {failure.site.label}"
                f" — {failure.error}"
            )
            if failure.installed:
                parts.append(f"    installed: {failure.installed}")
        if census.skipped:
            sites = "; ".join(f"{s.path}:{s.lineno} {s.label} ({s.reason})" for s in census.skipped)
            parts.append(f"  note: skipped site(s) not statically resolvable: {sites}")
        parts.append(
            "  fix: correct the call site, or waive with `# ARGCHECK_BIND_EXEMPT: <reason>`"
            " on the call line"
        )
        raise SystemExit("\n".join(parts))
