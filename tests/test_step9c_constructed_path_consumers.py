"""Discovery meta-test: constructed-path scripts/ consumers stay selector-covered (#2537).

A test that loads ``scripts/<stem>.py`` via ``importlib.util.spec_from_file_location``
with the module referenced only by BARE NAME (``ced = _load("clean_experiment_downloads")``)
is invisible to every text-scan selector arm — no Import node, no contiguous
repo-relative path literal, no ``<stem>.py`` token, no stem in the test FILENAME
(the #1688-documented f-string / flat-token miss classes). Incident #2336: two
such consumers never ran on a ``clean_experiment_downloads`` diff while the
round's pin-sweep reported an over-wide ``sweep_scope`` label.

This file discovers that consumer class over the whole ``tests/`` tree (strict
AST rule, plan #2537 §4.2) and asserts a LAYERED coverage predicate per
discovered ``(test, scripts/<stem>.py)`` pair, cheapest arm first:

  (a) the consumer test is a ``WORKFLOW_INVARIANT`` member (runs at every gate);
  (b) the pair is registered in ``TRANSITIVE_CONSUMER_TESTS`` (the selector's
      designed instrument for exactly this class);
  (c) the pair is vetted-not-a-genuine-load in ``_CONSTRUCTED_PATH_FP_ALLOWLIST``;
  (d) the pair appears in ONE batched in-process ``--map-files`` run over the
      residual modules — the REAL composed CLI as ground truth for
      stem-/literal-/import-arm coverage, never a re-implementation.

This file is itself a ``WORKFLOW_INVARIANT`` member, so a NEW same-idiom
consumer fails the gate of the very round that lands it; the selector's #2537
fail-closed refusal makes a deletion of THIS file refuse every later gate.

Disclosed misses (conservative by design — under-fire acceptable, over-fire
not; each stays covered only by the per-file ``untested_touched`` WARN):
``importlib.import_module`` by name; ``monkeypatch.setattr`` flat-string
targets; loaders living in ``conftest.py`` or imported from another file;
stem lists built by loops/comprehensions; ``scripts/`` subpackage modules;
stems shorter than 6 chars; stems riding a non-first positional or keyword
argument of a loader call.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load the selector by path — the sibling test file's idiom; the discovery
# below finds THIS pair too, covered by predicate arm (a) (this file is a
# WORKFLOW_INVARIANT member).
_HELPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "select_step9c_tests.py"
_spec = importlib.util.spec_from_file_location("select_step9c_tests", _HELPER_PATH)
assert _spec and _spec.loader
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

_REPO_ROOT = _HELPER_PATH.parents[1]

# Stem-shape guard: an unguarded exists() probe on arbitrary constant strings
# raises OSError errno 36 (File name too long) on multi-line string constants
# (measured at plan time on a long markdown constant inside a test file).
_STEM_RE = re.compile(r"[A-Za-z0-9_]{6,64}")

# Vetted false positives: pairs the discovery finds that are NOT a genuine
# load of the named module. Each entry carries a one-line comment naming WHY
# it is not a load — and the comment is not the evidence: the vet that admits
# an entry READS the load site first (#2537 plan §4.2 arm (c)). Empty at
# landing: the round-start audit vetted all 35 unmapped pairs as GENUINE
# loads (all registered in TRANSITIVE_CONSUMER_TESTS instead).
_CONSTRUCTED_PATH_FP_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()

_INCIDENT_MODULE = "scripts/clean_experiment_downloads.py"
_INCIDENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("tests/test_janitor_tmp_scratch_sweep.py", _INCIDENT_MODULE),
    ("tests/test_vm_disk_guard_slurm_src.py", _INCIDENT_MODULE),
)


def _call_name(node: ast.Call) -> str | None:
    """The callee's terminal name (`spec_from_file_location` under any prefix)."""
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


class _CallCollector(ast.NodeVisitor):
    """One traversal per file: every Call with its innermost enclosing function.

    The recursive function stack replaces the naive walk-within-walk variant
    (measured 11.63 s vs the 4.2-6.8 s single-pass band, plan §4.2 step 2).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, ast.Call]] = []
        self._stack: list[str] = []

    def _visit_fn(self, node) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_fn
    visit_AsyncFunctionDef = _visit_fn

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append((self._stack[-1] if self._stack else None, node))
        self.generic_visit(node)


def _loader_call_stems(collector: _CallCollector) -> tuple[set[str], bool]:
    """Arm (i): FIRST-POSITIONAL constant-string args at loader call sites.

    A loader function is a local FunctionDef whose body contains a
    ``spec_from_file_location`` call. Returns ``(stems, saw_nonconstant)``
    where the flag records any loader call with a non-constant first arg
    (the parametrized shape that arms the stem-list pass, arm (ii)).
    """
    loader_fns = {
        fn
        for fn, call in collector.calls
        if fn is not None and _call_name(call) == "spec_from_file_location"
    }
    stems: set[str] = set()
    saw_nonconstant = False
    for _fn, call in collector.calls:
        name = _call_name(call)
        if (name != "spec_from_file_location" and name not in loader_fns) or not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            stems.add(first.value)
        else:
            saw_nonconstant = True
    return stems, saw_nonconstant


def _module_stem_lists(tree: ast.Module, scripts_dir: Path) -> set[str]:
    """Arm (ii): module-level assigned lists/tuples of ALL-constant strings
    that ALL resolve (under the stem-shape guard) to existing
    ``scripts/<stem>.py`` files — the parametrize-list shape; the
    all-elements-resolve requirement suppresses mixed fixture lists."""
    stems: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        if not isinstance(val, (ast.List, ast.Tuple)) or not val.elts:
            continue
        elems: list[str] | None = []
        for e in val.elts:
            if not (isinstance(e, ast.Constant) and isinstance(e.value, str)):
                elems = None
                break
            elems.append(e.value)
        if elems and all(
            _STEM_RE.fullmatch(s) and (scripts_dir / f"{s}.py").exists() for s in elems
        ):
            stems.update(elems)
    return stems


def discover_pairs(work_root: Path) -> set[tuple[str, str]]:
    """Strict-AST discovery of constructed-path consumer pairs (plan §4.2 steps 1-4).

    Returns ``{(test_relpath, "scripts/<stem>.py")}`` where the stem was the
    FIRST POSITIONAL constant-string argument at a call site of
    ``spec_from_file_location`` or of a local loader function wrapping it —
    plus, ONLY in files with a non-constant loader argument, module-level
    all-constant-string lists/tuples whose elements ALL resolve to existing
    ``scripts/<stem>.py`` files (the parametrize-list shape).
    """
    pairs: set[tuple[str, str]] = set()
    scripts_dir = work_root / "scripts"
    for path in sorted((work_root / "tests").rglob("test_*.py")):
        try:
            text = path.read_text()
        except (OSError, ValueError):
            continue
        if "spec_from_file_location" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        collector = _CallCollector()
        collector.visit(tree)
        stems, saw_nonconstant = _loader_call_stems(collector)
        if saw_nonconstant:
            stems |= _module_stem_lists(tree, scripts_dir)
        rel = path.relative_to(work_root).as_posix()
        for s in stems:
            if _STEM_RE.fullmatch(s) and (scripts_dir / f"{s}.py").exists():
                pairs.add((rel, f"scripts/{s}.py"))
    return pairs


@pytest.fixture(scope="module")
def shared() -> SimpleNamespace:
    """Discovery + THE ONE batched CLI call this file makes per gate run.

    The call runs with the loaded module's ``TRANSITIVE_CONSUMER_TESTS``
    TEMPORARILY holding no incident-module key (swap-and-restore), so its
    mapped output reproduces the PRE-#2537 arm coverage for the incident
    module (the trace test's ground truth). Invisible to the coverage
    predicate: arm (b) reads the REAL dict in-process, registry-covered pairs
    never reach arm (d), and removing a registry key only ever REMOVES that
    key's own pair lines (the registry is additive-only) — every
    stem-/literal-/import-arm line stays intact.
    """
    discovered = discover_pairs(_REPO_ROOT)
    inv = set(sel.WORKFLOW_INVARIANT)
    residual_modules = {
        m
        for (t, m) in discovered
        if t not in inv
        and t not in sel.TRANSITIVE_CONSUMER_TESTS.get(m, ())
        and (t, m) not in _CONSTRUCTED_PATH_FP_ALLOWLIST
    }
    payload_modules = sorted(residual_modules | {_INCIDENT_MODULE})
    fd, payload_path = tempfile.mkstemp(prefix="i2537-payload-", suffix=".txt")
    saved = sel.TRANSITIVE_CONSUMER_TESTS
    buf = io.StringIO()
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("".join(f"{m}\n" for m in payload_modules))
        sel.TRANSITIVE_CONSUMER_TESTS = {k: v for k, v in saved.items() if k != _INCIDENT_MODULE}
        with contextlib.redirect_stdout(buf):
            rc = sel.main(["--map-files", payload_path, "--repo-root", str(_REPO_ROOT)])
    finally:
        sel.TRANSITIVE_CONSUMER_TESTS = saved
        os.unlink(payload_path)
    assert rc == 0, f"--map-files run failed rc={rc}"
    mapped = {tuple(line.split("\t", 1)) for line in buf.getvalue().splitlines() if "\t" in line}
    return SimpleNamespace(discovered=discovered, mapped=mapped)


def test_constructed_path_consumers_all_covered(shared: SimpleNamespace) -> None:
    """Every discovered constructed-path consumer pair is selector-covered."""
    inv = set(sel.WORKFLOW_INVARIANT)
    uncovered = [
        (t, m)
        for (t, m) in sorted(shared.discovered)
        if t not in inv  # (a)
        and t not in sel.TRANSITIVE_CONSUMER_TESTS.get(m, ())  # (b)
        and (t, m) not in _CONSTRUCTED_PATH_FP_ALLOWLIST  # (c)
        and (t, m) not in shared.mapped  # (d)
    ]
    assert not uncovered, (
        "constructed-path consumer pair(s) not covered by any selector arm — a "
        "diff to the module would reach the gate without running its consumer "
        "test. Remedy per pair: register in TRANSITIVE_CONSUMER_TESTS "
        "(scripts/select_step9c_tests.py) after READING the load site, or add "
        f"to _CONSTRUCTED_PATH_FP_ALLOWLIST with a vet rationale: {uncovered}"
    )


def test_discovery_fires_on_incident_pairs(shared: SimpleNamespace) -> None:
    """The predicate demonstrably fires on its own motivating incident (#1287 rule).

    Reads the SAME shared CLI call (no second invocation): with the incident
    module's registry key swap-removed, the mapped output reproduces the
    pre-fix arm coverage, so both incident pairs must be discovered yet
    UNMAPPED — hence, being neither invariant nor allowlisted, only the #2537
    registry entries (arm (b)) cover them.
    """
    inv = set(sel.WORKFLOW_INVARIANT)
    for pair in _INCIDENT_PAIRS:
        t = pair[0]
        # (i) the discovery finds both incident pairs:
        assert pair in shared.discovered, f"discovery no longer finds {pair}"
        # (ii) PAIR-ABSENCE from the swap-removed mapped set — deliberately a
        # containment assert, never an exact set/count (this plan's own diff
        # adds a 14th literal-arm pair, and future consumers keep churning the
        # count). If this fires, a later round made the incident test
        # arm-covered (e.g. added the module's repo-relative literal to it):
        # re-establish the trace premise against a then-current unmapped pair.
        assert pair not in shared.mapped, (
            f"{pair} is now arm-covered WITHOUT its registry entry — the "
            "incident-trace premise no longer holds; re-establish it against "
            "a then-current unmapped pair (plan #2537 §4.2 step 6 residual)"
        )
        # (iii) neither invariant nor allowlisted — so WITHOUT the registry
        # entry the layered predicate would fail for this pair:
        assert t not in inv, f"{t} unexpectedly WORKFLOW_INVARIANT"
        assert pair not in _CONSTRUCTED_PATH_FP_ALLOWLIST, f"{pair} unexpectedly allowlisted"
        # (iv) with the REAL registry, arm (b) covers both:
        assert t in sel.TRANSITIVE_CONSUMER_TESTS[_INCIDENT_MODULE], (
            f"{t} missing from the TRANSITIVE_CONSUMER_TESTS entry for "
            f"{_INCIDENT_MODULE} — the #2537 registration regressed"
        )
