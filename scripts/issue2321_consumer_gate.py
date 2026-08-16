"""#2321 consumer-inventory tooling for the per-prefix PRE-repack gate (I17, plan §3.8).

The repack permanently deletes the 10 target prefixes' original small files
(byte-preserved in ``packed/`` shards). Single-path readers miss LOUDLY
(``hf_hub_download`` raises), but LISTING/GLOB discovery consumers are
SILENT-EMPTY: ``list_repo_tree`` over a repacked dir returns only the
retained subset and ``snapshot_download(allow_patterns=...)`` succeeds on
zero matches, so a consumer deriving its work list from such a listing reads
a repacked prefix as a real empty/zero result with no error anywhere. The
driver's ``--phase consumer-gate`` (``scripts/issue2321_repack.py``) BLOCKS a
prefix's commit phase (rc=22) while any silent-empty consumer scoped to it
remains unmigrated, reading the committed inventory this tool maintains.

Modes
-----
``--build-inventory [--out PATH]``
    AST-scan ``scripts/`` + ``src/`` for Hub-discovery calls
    (``list_repo_tree`` / the ``hub.py`` scoped-listing family /
    ``snapshot_download`` / ``stage_hub_prefix``) whose path/prefix argument
    (string literal, resolvable module constant, or f-string with a
    resolvable leading fragment) sits under any of the 10 target prefixes,
    and print the hit list as JSON — the CANDIDATE set the curated committed
    inventory (``scripts/issue2321_consumer_inventory.json``) must cover.
``--check [--inventory PATH]``
    Re-run the scan and verify every hit is covered by an inventory row
    (same script + prefix scoping); uncovered hits exit 1 — the last-cheap-
    moment catch for consumers added after the inventory was curated.
``--gate --prefix P [--inventory PATH]``
    Evaluate the committed inventory with the DRIVER's own gate semantics
    (single source: ``issue2321_repack.load_consumer_inventory`` +
    ``consumer_gate`` are imported and called, never re-implemented):
    rc=0 pass, rc=22 blocked.

Honest residual (plan §3.8): the static scan misses dynamically-constructed
paths and out-of-repo consumers — the loud-404 degradation + the completed
#2304 overflow-routing interim cover single-path misses, and
``hf_hub_download``-in-a-listing-loop / glob-over-listing shapes are covered
transitively only when the loop's work list derives from a detected listing
call. Two further named residuals (r2): a discovery call ALIASED at import
time (``from huggingface_hub import list_repo_tree as lrt``) escapes the
call-name match, and ``get_paths_info`` is matched only by its canonical
name. The scan is re-runnable (``--check``) — the driver's commit phase now
re-runs it at commit admission (``fresh_consumer_scan_gate``), so late-added
consumers are caught at the last cheap moment before any delete composes.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = Path(__file__).with_name("issue2321_consumer_inventory.json")

#: Discovery calls whose empty result a consumer can silently absorb
#: (plan §3.8; the hub.py scoped-listing family included — a raw-only
#: listing through them is still silent-empty over a repacked dir unless
#: unioned with the packed members).
DISCOVERY_CALLS = frozenset(
    {
        "list_repo_tree",
        "list_repo_files",
        "list_hf_entries_under_path",
        "list_repo_entries_complete",
        "list_repo_files_complete",
        "list_hf_files_under_path",
        "list_repo_repofiles_under_path",
        "snapshot_download",
        "stage_hub_prefix",
        # r2 Codex minor: get_paths_info silently OMITS missing paths from its
        # return — a work list derived from it over a repacked dir is the same
        # silent-empty shape as a listing.
        "get_paths_info",
    }
)

#: The packed-aware helper family + this task's own tooling — excluded from
#: the scan (plan §3.8 "EXCLUDING the packed-aware helpers themselves").
EXCLUDED_FILES = frozenset(
    {
        "src/explore_persona_space/orchestrate/hub.py",
        "src/explore_persona_space/orchestrate/packing.py",
        "scripts/issue2321_repack.py",
        "scripts/issue2321_consumer_gate.py",
        "scripts/issue2321_verify_shim.py",
    }
)


def _load_driver():
    """Load the sibling repack driver (single source for prefixes + gate)."""
    path = Path(__file__).with_name("issue2321_repack.py")
    spec = importlib.util.spec_from_file_location("issue2321_repack_driver", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # sys.modules registration BEFORE exec: dataclasses' annotation resolution
    # reads sys.modules[cls.__module__].__dict__ at class-creation time.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Top-level ``NAME = "str"`` assignments (the resolvable-constant set)."""
    consts: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    consts[tgt.id] = node.value.value
    return consts


def _resolvable_strings(node: ast.expr, consts: dict[str, str]) -> list[str]:
    """Every statically-resolvable string (or leading f-string fragment)
    reachable from ``node`` — Constant str, resolvable Name, JoinedStr with a
    resolvable leading run, and list/tuple/set elements recursively."""
    out: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, ast.Name) and node.id in consts:
        out.append(consts[node.id])
    elif isinstance(node, ast.JoinedStr):
        frag = ""
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                frag += value.value
            elif (
                isinstance(value, ast.FormattedValue)
                and isinstance(value.value, ast.Name)
                and value.value.id in consts
            ):
                frag += consts[value.value.id]
            else:
                break
        if frag:
            out.append(frag)
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            out.extend(_resolvable_strings(elt, consts))
    return out


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def scan_file(py: Path, rel: str, target_prefixes: frozenset[str]) -> list[dict]:
    """All discovery-call hits in one file whose resolvable path/prefix sits
    under a target prefix.

    Resolution covers module-level string constants AND simple ``name = <str>``
    assigns anywhere in the file (``root = f"{PREFIX}/..."`` then
    ``list_...(api, REPO, root)`` — the indirection shape the migrated #1481
    consumer itself uses). The assign table is FILE-FLAT, not scope-exact
    (two linear walks, deliberately: an exact per-scope resolution was ~60x
    slower over the tree), and merges PREFERRING target-prefix-resolving
    values — an over-approximation that can only OVER-detect (a spurious hit
    costs one curated inventory row; an under-detection would let a
    silent-empty consumer through). A purely dynamic path remains the
    documented residual (plan section 3.8)."""
    text = py.read_text(encoding="utf-8")
    # Text prefilter (the AST walk over ~8M nodes tree-wide costs ~80s; a
    # substring screen cuts the parse set to the few prefix-mentioning files).
    # Conservative by construction: anything the prefilter drops could not
    # have resolved anyway (cross-module constants and mid-prefix string
    # assembly are below the resolver — the same documented residual).
    if not any(p in text for p in target_prefixes):
        return []
    if not any(c + "(" in text for c in DISCOVERY_CALLS):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError as err:  # a broken file is a loud finding, not a skip
        return [{"script": rel, "line": err.lineno or 0, "call": "SYNTAX-ERROR", "prefix": ""}]

    def _is_target(s: str) -> bool:
        return s.lstrip("/").split("/")[0] in target_prefixes

    # Pass 1: module constants, then a file-flat assign table on top.
    consts = _module_constants(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and not isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
        ):
            vals = _resolvable_strings(node.value, consts)
            if len(vals) == 1:
                name = node.targets[0].id
                prev = consts.get(name)
                if prev is None or _is_target(vals[0]) or not _is_target(prev):
                    consts[name] = vals[0]

    # Pass 2: discovery calls resolved against the merged table.
    hits: list[dict] = []
    seen: set[tuple] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name not in DISCOVERY_CALLS:
            continue
        strings: list[str] = []
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            strings.extend(_resolvable_strings(arg, consts))
        for s in strings:
            top = s.lstrip("/").split("/")[0]
            if top in target_prefixes:
                key = (rel, node.lineno, name, top)
                if key not in seen:
                    seen.add(key)
                    hits.append({"script": rel, "line": node.lineno, "call": name, "prefix": top})
    return hits


def scan_tree(root: Path, target_prefixes: frozenset[str]) -> list[dict]:
    """Scan ``scripts/**/*.py`` + ``src/**/*.py`` (minus the shim family)."""
    hits: list[dict] = []
    for sub in ("scripts", "src"):
        base = root / sub
        if not base.exists():
            continue
        for py in sorted(base.rglob("*.py")):
            rel = py.relative_to(root).as_posix()
            if rel in EXCLUDED_FILES:
                continue
            hits.extend(scan_file(py, rel, target_prefixes))
    return hits


def check_inventory(hits: list[dict], inventory: dict, scoped_fn, row_errors_fn=None) -> list[str]:
    """Every scan hit must be covered by an inventory row (same script AND the
    hit prefix inside the row's scoping); malformed rows are errors too.

    Row validation is TYPE-EXACT (r2 g5-M1 / Codex-M1): rows are checked with
    the driver's ``_consumer_row_type_errors`` (single source of truth —
    pass it as ``row_errors_fn``; omitted, the driver is loaded to fetch it),
    so a hand-authored ``"migrated": "false"`` string-boolean is an error
    here exactly as it fails CLOSED in ``load_consumer_inventory``.
    """
    errors: list[str] = []
    consumers = inventory.get("consumers")
    if not isinstance(consumers, list):
        return [f"malformed inventory: no consumers[] (version={inventory.get('version')})"]
    if row_errors_fn is None:
        row_errors_fn = _load_driver()._consumer_row_type_errors
    for i, row in enumerate(consumers):
        for err in row_errors_fn(row):
            label = row.get("script", "?") if isinstance(row, dict) else repr(row)
            errors.append(f"inventory row {i} ({label}) malformed: {err}")
    for hit in hits:
        covered = any(
            isinstance(row, dict)
            and row.get("script") == hit["script"]
            and scoped_fn(row, hit["prefix"])
            for row in consumers
        )
        if not covered:
            errors.append(
                f"UNCOVERED consumer hit: {hit['script']}:{hit['line']} {hit['call']}(...) "
                f"on {hit['prefix']} — triage it into issue2321_consumer_inventory.json "
                "(silent_empty true/false + migration status) before that prefix repacks"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-inventory", action="store_true", help="print scan hits as JSON")
    mode.add_argument("--check", action="store_true", help="scan + verify inventory coverage")
    mode.add_argument("--gate", action="store_true", help="evaluate the I17 gate for --prefix")
    ap.add_argument("--prefix", default=None, help="target prefix (required for --gate)")
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    ap.add_argument("--out", type=Path, default=None, help="write --build-inventory JSON here")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="scan root (test override)")
    args = ap.parse_args(argv)

    driver = _load_driver()
    target_prefixes = frozenset(driver.PREFIX_ORDER)

    if args.gate:
        if not args.prefix:
            ap.error("--gate requires --prefix")
        try:
            verdict = driver.consumer_gate(
                driver.load_consumer_inventory(args.inventory), args.prefix
            )
        except driver.ConsumerGateBlocked as err:
            print(f"[consumer-gate] BLOCKED: {err}")
            return err.rc
        print(f"[consumer-gate] PASS: {json.dumps(verdict, sort_keys=True)}")
        return 0

    hits = scan_tree(args.repo_root, target_prefixes)
    if args.build_inventory:
        doc = {
            "version": 1,
            "target_prefixes": sorted(target_prefixes),
            "n_hits": len(hits),
            "hits": hits,
        }
        text = json.dumps(doc, indent=1, sort_keys=True)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"[build-inventory] {len(hits)} hits -> {args.out}")
        else:
            print(text)
        return 0

    # --check
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    errors = check_inventory(
        hits, inventory, driver._consumer_scoped, driver._consumer_row_type_errors
    )
    if errors:
        for e in errors:
            print(f"[consumer-check] {e}")
        return 1
    print(f"[consumer-check] PASS: {len(hits)} scan hit(s) all covered by {args.inventory.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
