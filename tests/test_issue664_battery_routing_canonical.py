"""Issue #664 §16 canonical-battery-routing invariant pins (the strategy pivot).

Three pytests pin the SINGLE-RESOLVER contract that removes the r1->r3
manual-router failure mode mechanically. The failure pattern: three implementer
rounds shipped a per-column / per-behavior battery ROUTING function built from
manual ``if column in (...)`` special cases plus a generic
``fetch_preregistered_probes(48)`` fallback, and a DIFFERENT wrong-battery
misroute slipped through on a different sibling surface each round (r2:
sycophancy/refusal store on the generic 48; r3: ``harmful_compliance``->Betley-8
where the #545 registry mandates AdvBench-200, ``deception``/``self_report``/
``persona_drift``/``format_style``->generic-48 where each has its own #545
battery). §16's fix: exactly ONE resolver
(``issue664_common.canonical_battery_for_column``), every #545 column self-routes
via its own ``ColumnSpec.battery``, and the only escape hatch is a single declared
override dict (``ISSUE_664_BATTERY_OVERRIDES``) these tests enumerate.

All CPU-only: they import the ``scripts/issue664_*`` modules + the #545 registry
and exercise the pure-Python routing topology. They read the git-committed #545
frozen batteries under ``eval_results/issue_545/batteries/`` (the canonical
resolver's registry fallthrough reads them), so the worktree's sparse cone must
include that path (added at implementation time).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue664_common as C  # noqa: E402
import issue664_eval as E  # noqa: E402
import issue664_extract_store as S  # noqa: E402

from explore_persona_space.experiments.behavior_testbed_545.columns import COLUMNS  # noqa: E402


# ── offline override providers: stub the HF/local-pool-dependent override
# providers so the CPU-only ROUTING-SHAPE tests never hit the network or depend
# on a warm HF cache / a written P2.0 pool (Codex post-pivot r1 minor). The
# sycophancy override (``_sycophancy_sharma_wrong_claims``) calls ``hf_hub_download``;
# the refusal override (``_refusal_390_pool``) reads a P2.0 pool file the worktree
# may not have. We swap the ISSUE_664_BATTERY_OVERRIDES dict entries for tiny
# in-process providers that return the SAME probe-item shape the real ones
# guarantee. The routing TOPOLOGY (override-vs-registry XOR, the resolver call
# graph) is what these tests pin -- not the override pools' content -- and the
# registry path (harmful_compliance -> AdvBench-200, etc.) is exercised
# unchanged via the in-tree #545 frozen batteries (non-network).
def _stub_provider(tag: str):
    return lambda: [{"probe_id": f"{tag}_{i}", "question": f"{tag} probe {i}"} for i in range(3)]


@pytest.fixture(autouse=True)
def _offline_battery_overrides(monkeypatch):
    """Patch every declared override provider to a tiny in-process stub so the
    routing-shape tests are network-free + HF-cache-independent. Registry-routed
    columns are untouched (they read the git-committed #545 batteries)."""
    for col in C.ISSUE_664_BATTERY_OVERRIDES:
        monkeypatch.setitem(C.ISSUE_664_BATTERY_OVERRIDES, col, _stub_provider(col))


# ── 1. judging-surface battery routing is canonical XOR an explicit override ──
def test_judging_surface_battery_routing_canonical() -> None:
    """Every column in the realized judging surface resolves either via the
    canonical #545 helper OR via an explicit ``ISSUE_664_BATTERY_OVERRIDES``
    entry -- and NO column takes another path. Pins the r3 misroute explicitly."""
    # enumerate every (ctx, col) the judged-rate eval actually populates.
    surface_columns: set[str] = set()
    for cell in C.realized_grid():
        for _ctx, col in E._judging_surface(cell):
            surface_columns.add(col)
    assert surface_columns, "empty judging surface -- realized grid produced no (ctx,col) tuples"
    # marker is never a judged-rate column (its DV is the four-float slot stats).
    assert "marker" not in surface_columns

    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import battery_probes

    for col in sorted(surface_columns):
        is_override = col in C.ISSUE_664_BATTERY_OVERRIDES
        is_registry = col in COLUMNS
        # Every surface column resolves via SOME canonical path (override or the
        # #545 registry); never via a hand-rolled fallback (the r1->r3 defect).
        assert is_override or is_registry, (
            f"column {col!r} resolves via NEITHER an override NOR the #545 registry "
            f"-- §16 forbids any other routing path"
        )
        probes = C.canonical_battery_for_column(col)
        assert isinstance(probes, list) and probes, f"{col!r} canonical battery empty / not a list"
        assert all(isinstance(it, dict) for it in probes), f"{col!r} battery items not dicts"
        assert all("question" in it for it in probes), (
            f"{col!r} battery items lack a 'question' key"
        )
        # XOR per §16.4: an OVERRIDE column resolves via its provider (the
        # override short-circuits FIRST), a NON-override column via the #545
        # battery_probes helper -- never both. Assert the resolved battery
        # matches the path the column actually takes.
        if is_override:
            assert probes == C.ISSUE_664_BATTERY_OVERRIDES[col](), (
                f"override column {col!r} did not resolve via its declared provider"
            )
        else:
            assert probes == battery_probes(COLUMNS[col]), (
                f"registry column {col!r} did not resolve via battery_probes(COLUMNS[col])"
            )

    # PIN the r3 defect explicitly: harmful_compliance resolves via the #545
    # helper -> AdvBench-200 (advbench_200.json), NOT a Betley-8 special case.
    assert COLUMNS["harmful_compliance"].battery == "advbench_200.json"
    assert "harmful_compliance" not in C.ISSUE_664_BATTERY_OVERRIDES
    hc = C.canonical_battery_for_column("harmful_compliance")
    assert len(hc) == 200, f"harmful_compliance should resolve to AdvBench-200, got {len(hc)}"

    # the other r3-misrouted columns resolve via the #545 helper -> own battery,
    # NOT the generic-48 fallback.
    for col in ("deception", "self_report", "persona_drift", "format_style"):
        assert col not in C.ISSUE_664_BATTERY_OVERRIDES, f"{col} must NOT be an override"
        assert col in COLUMNS
        probes = C.canonical_battery_for_column(col)
        assert probes and probes[0].get("question") is not None


# ── 2. per-behavior store-routing == per-column eval-routing (the B6 identity) ─
def test_behavior_eval_battery_canonical() -> None:
    """Every realized-grid behavior's eval battery resolves through the canonical
    per-behavior resolver, which routes to its PRIMARY #545 column. Store-path
    battery == eval-path battery (the r2 B6 identity guarantee -- same defect
    class as the r3 misroute)."""
    behaviors = (*C.CONTENT_BEHAVIORS, "marker", "tf_rev", "ic_edu")
    for behavior in behaviors:
        col = C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN[behavior]
        by_behavior = C.canonical_battery_for_behavior(behavior)
        by_column = C.canonical_battery_for_column(col)
        # the per-behavior resolver adds NO routing of its own -- it is exactly
        # the column resolver on the primary column.
        assert by_behavior == by_column, f"behavior {behavior!r} != column {col!r} battery"
        # the primary column routes via an override OR the #545 registry.
        assert (col in C.ISSUE_664_BATTERY_OVERRIDES) or (col in COLUMNS), (
            f"primary column {col!r} for {behavior!r} routes via neither path"
        )
        assert isinstance(by_behavior, list) and by_behavior, f"{behavior!r} battery empty"
        assert all(isinstance(it, dict) and "question" in it for it in by_behavior)

    # store-vs-eval IDENTITY: the extract-store path and the eval path call the
    # SAME resolver, so they return the SAME battery for a sampled behavior.
    # (The eval gen path extracts ["question"] from canonical_battery_for_column;
    # the store path extracts ["question"] from canonical_battery_for_behavior.
    # Both resolve to the same probe-item list for a behavior's primary column.)
    sample = "sycophancy"
    store_qs = [it["question"] for it in C.canonical_battery_for_behavior(sample)]
    eval_qs = [
        it["question"]
        for it in C.canonical_battery_for_column(C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN[sample])
    ]
    assert store_qs == eval_qs, "store-path battery != eval-path battery for the same behavior"


# ── 3. AST: no banned manual routing survives in any issue664 script ──────────
_BATTERY_POOL_CALLS = {
    "fetch_preregistered_probes",
    "fetch_betley_main_8",
}
# the §16.4 allow-list: enclosing functions permitted to call a Betley-pool
# helper (the marker carve-out fn + the declared override providers). Post-pivot
# the issue664 scripts contain NO such call at all (marker routes canonically and
# the overrides use the Sharma/#390/#444 pools, not Betley helpers), so the
# allow-list is informational -- the test asserts ZERO banned calls regardless.
_ALLOWED_POOL_CALL_FUNCS: set[str] = set()


def _module_path(mod) -> Path:
    return Path(mod.__file__)


def _battery_routing_modules() -> list:
    import issue664_build_training_data as B
    import issue664_dispatch as D

    return [C, E, S, D, B]


def _called_name(node: ast.Call) -> str | None:
    """The bare function name of a Call (``foo(...)`` -> 'foo', ``x.foo(...)`` -> 'foo')."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _enclosing_func(tree: ast.AST, target: ast.AST) -> str | None:
    """Name of the innermost FunctionDef enclosing ``target`` (None at module scope)."""
    stack: list[str] = []
    found: list[str | None] = []

    class _V(ast.NodeVisitor):
        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            stack.append(n.name)
            self.generic_visit(n)
            stack.pop()

        def generic_visit(self, n: ast.AST) -> None:
            if n is target:
                found.append(stack[-1] if stack else None)
            super().generic_visit(n)

    _V().visit(tree)
    return found[0] if found else None


def _returns_a_battery(if_node: ast.If) -> bool:
    """True if the If-body Return-s a battery-shaped value: a Call to a known
    battery helper (fetch_*/battery_probes/*_probes/*_pool/*_battery) or a
    subscript of a battery override dict. Excludes int / scalar returns (so the
    per-behavior ``max_length`` selector -- ``if behavior in (...): return 2048``
    -- is NOT flagged: it returns an int, not a battery)."""
    for sub in ast.walk(if_node):
        if not isinstance(sub, ast.Return) or sub.value is None:
            continue
        val = sub.value
        if isinstance(val, ast.Call):
            name = _called_name(val)
            if name and (
                name in _BATTERY_POOL_CALLS
                or name == "battery_probes"
                or name.endswith("_probes")
                or name.endswith("_pool")
                or name.endswith("_battery")
            ):
                return True
        if (
            isinstance(val, ast.Subscript)
            and isinstance(val.value, ast.Name)
            and ("OVERRID" in val.value.id.upper() or "BATTER" in val.value.id.upper())
        ):
            return True
    return False


_FORBIDDEN_DEF_NAMES = (
    "column_probes",
    "_column_probes",
    "behavior_eval_battery",
    "_behavior_battery",
)


def _scan_banned_pool_calls(path: Path, tree: ast.AST) -> list[str]:
    """(a) fetch_preregistered_probes / fetch_betley_main_8 calls outside the allow-list."""
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name in _BATTERY_POOL_CALLS:
            enc = _enclosing_func(tree, node)
            if enc not in _ALLOWED_POOL_CALL_FUNCS:
                out.append(
                    f"{path.name}: banned battery-pool call {name!r} in "
                    f"function {enc!r} (not in the §16.4 allow-list)"
                )
    return out


def _scan_per_key_routing(path: Path, tree: ast.AST) -> list[str]:
    """(b) `if column/behavior in (...)` returning a battery outside the override dict."""
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (len(test.ops) == 1 and isinstance(test.ops[0], (ast.In, ast.NotIn))):
            continue
        left = test.left
        if not (isinstance(left, ast.Name) and left.id in ("column", "behavior")):
            continue
        if _returns_a_battery(node):
            enc = _enclosing_func(tree, node)
            out.append(
                f"{path.name}: banned per-key routing "
                f"`if {left.id} in (...): return <battery>` in function {enc!r}"
            )
    return out


def _scan_resolver_defs(path: Path, tree: ast.AST) -> tuple[int, int, list[str]]:
    """(c) count canonical resolver defs + collect any surviving forbidden defs."""
    col_defs = beh_defs = 0
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "canonical_battery_for_column":
            col_defs += 1
        elif node.name == "canonical_battery_for_behavior":
            beh_defs += 1
        elif node.name in _FORBIDDEN_DEF_NAMES:
            forbidden.append(f"{path.name}:{node.name}")
    return col_defs, beh_defs, forbidden


def test_no_manual_fallback_in_issue664() -> None:
    """AST-walk the issue664 scripts; assert no banned manual routing remains."""
    violations: list[str] = []
    canonical_col_defs = 0
    canonical_beh_defs = 0
    forbidden_defs: list[str] = []

    for mod in _battery_routing_modules():
        path = _module_path(mod)
        tree = ast.parse(path.read_text())
        violations += _scan_banned_pool_calls(path, tree)
        violations += _scan_per_key_routing(path, tree)
        col_defs, beh_defs, forbidden = _scan_resolver_defs(path, tree)
        canonical_col_defs += col_defs
        canonical_beh_defs += beh_defs
        forbidden_defs += forbidden

    assert not violations, "banned manual routing survives:\n  " + "\n  ".join(violations)
    assert canonical_col_defs == 1, (
        f"expected exactly 1 def canonical_battery_for_column, found {canonical_col_defs}"
    )
    assert canonical_beh_defs == 1, (
        f"expected exactly 1 def canonical_battery_for_behavior, found {canonical_beh_defs}"
    )
    assert not forbidden_defs, "deleted-by-§16 routing functions survive: " + ", ".join(
        forbidden_defs
    )
