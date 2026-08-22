"""Recon probe for #2253: size the third-party-import-vs-uv.lock surface.

Read-only. Not a deliverable — informs the plan's false-positive budget.
"""

from __future__ import annotations

import ast
import collections
import pathlib
import sys
import tomllib

ROOT = pathlib.Path("/home/thomasjiralerspong/explore-persona-space")

# ---- uv.lock package names -------------------------------------------------
lock = tomllib.loads((ROOT / "uv.lock").read_text())
lock_dists = {p["name"].lower().replace("_", "-") for p in lock.get("package", [])}

pyproj = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _dep_names(raw):
    out = set()
    for d in raw or []:
        name = ""
        for ch in d:
            if ch.isalnum() or ch in "._-":
                name += ch
            else:
                break
        if name:
            out.add(name.lower().replace("_", "-"))
    return out


proj_deps = _dep_names((pyproj.get("project") or {}).get("dependencies"))
for grp in ((pyproj.get("project") or {}).get("optional-dependencies") or {}).values():
    proj_deps |= _dep_names(grp)
for grp in ((pyproj.get("dependency-groups") or {})).values():
    proj_deps |= _dep_names(grp)

# ---- import-name -> dist-name from the live venv ---------------------------
try:
    import importlib.metadata as md

    pkg_dists = md.packages_distributions()
except Exception as exc:  # pragma: no cover
    print(f"packages_distributions failed: {exc}", file=sys.stderr)
    pkg_dists = {}

venv_map = {k: {v.lower().replace("_", "-") for v in vs} for k, vs in pkg_dists.items()}

# ---- first-party roots -----------------------------------------------------
first_party = {"explore_persona_space"}
for p in (ROOT / "scripts").rglob("*.py"):
    first_party.add(p.stem)
for p in (ROOT / "src").iterdir() if (ROOT / "src").is_dir() else []:
    if p.is_dir():
        first_party.add(p.name)
for p in (ROOT / "tests").rglob("*.py"):
    first_party.add(p.stem)
first_party |= {"conftest", "setup"}

stdlib = set(sys.stdlib_module_names) | {"__future__"}

# ---- scan ------------------------------------------------------------------
roots: dict[str, list[str]] = collections.defaultdict(list)
n_files = 0
parse_fail = []

targets = sorted(
    list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "src").rglob("*.py"))
)
for path in targets:
    n_files += 1
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        parse_fail.append(f"{path.relative_to(ROOT)}: {exc}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots[alias.name.split(".")[0]].append(str(path.relative_to(ROOT)))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import -> first-party
                continue
            if node.module:
                roots[node.module.split(".")[0]].append(str(path.relative_to(ROOT)))

third_party = {}
for root, files in roots.items():
    if root in stdlib or root in first_party:
        continue
    third_party[root] = files

resolved, unresolved, mapped_via_venv = [], [], []
for root in sorted(third_party):
    direct = root.lower().replace("_", "-")
    if direct in lock_dists or direct in proj_deps:
        resolved.append(root)
        continue
    dists = venv_map.get(root, set())
    hit = dists & (lock_dists | proj_deps)
    if hit:
        mapped_via_venv.append((root, sorted(hit)))
        continue
    unresolved.append((root, len(third_party[root]), sorted(set(third_party[root]))[:4]))

print(f"files scanned           : {n_files}")
print(f"parse failures          : {len(parse_fail)}")
for pf in parse_fail:
    print(f"    {pf}")
print(f"uv.lock dists           : {len(lock_dists)}")
print(f"distinct import roots   : {len(roots)}")
print(f"third-party roots       : {len(third_party)}")
print(f"  resolved (name match) : {len(resolved)}")
print(f"  resolved via venv map : {len(mapped_via_venv)}")
print(f"  UNRESOLVED            : {len(unresolved)}")
print()
print("--- mapped only via venv metadata (import-name != dist-name) ---")
for root, hit in mapped_via_venv:
    print(f"  {root:28s} -> {', '.join(hit)}")
print()
print("--- UNRESOLVED (candidate FAIL hits) ---")
for root, n, files in unresolved:
    print(f"  {root:28s} {n:4d} site(s)  e.g. {files}")
