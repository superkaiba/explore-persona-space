"""Signature-bind pin over the in-script --smoke Hub fakes (#1901 round 3).

Round 2's C1 revision threading passed ``revision=`` unconditionally through the
shared #779 scripts' Hub calls, but the strict in-file smoke fakes (functions /
classes handed to ``mock.patch("huggingface_hub.hf_hub_download", ...)`` and
``mock.patch("huggingface_hub.HfApi", ...)``) were not updated, so BOTH
CLI-wired self-smoke entrypoints crashed:
``TypeError: _FakeHfApi.list_repo_tree() got an unexpected keyword argument
'revision'``.

This test statically BINDS every in-script fake against every production call
shape in the same file: it AST-extracts (a) each ``hf_hub_download(...)`` /
``*.list_repo_tree(...)`` call's positional count + keyword-name set, and
(b) each fake def handed to ``mock.patch`` for those targets, then checks each
fake's parameter list accepts each call shape. The next kwarg-threading round
(a new ``etag_timeout=`` / ``token=`` / whatever) therefore fails HERE in
milliseconds instead of at the pod-side smoke.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = (
    REPO_ROOT / "scripts" / "issue779_ffc_n1m_fits.py",
    REPO_ROOT / "scripts" / "issue779_ffc_n1m_generate_capture.py",
)

# Minimum fake populations per file — guards the scanner against silent rot
# (a refactor that renames the fakes must update this table, not no-op the test).
MIN_FAKES = {
    "issue779_ffc_n1m_fits.py": {"download": 4, "list_repo_tree": 2},
    "issue779_ffc_n1m_generate_capture.py": {"download": 1, "list_repo_tree": 1},
}


def _call_shape(call: ast.Call) -> tuple[int, frozenset[str], bool]:
    """(n_positional, keyword-name set, has-**splat) for one Call node."""
    n_pos = len(call.args)
    kw_names = frozenset(k.arg for k in call.keywords if k.arg is not None)
    has_splat = any(k.arg is None for k in call.keywords)
    return n_pos, kw_names, has_splat


def _collect_shapes(tree: ast.AST) -> tuple[list[tuple], list[tuple]]:
    """Production call shapes: (hf_hub_download calls, *.list_repo_tree calls)."""
    dl_shapes: list[tuple] = []
    tree_shapes: list[tuple] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == "hf_hub_download":
            dl_shapes.append(_call_shape(node))
        elif isinstance(f, ast.Attribute) and f.attr == "list_repo_tree":
            tree_shapes.append(_call_shape(node))
    return dl_shapes, tree_shapes


def _collect_fakes(tree: ast.AST) -> tuple[list[ast.FunctionDef], list[ast.FunctionDef]]:
    """Fake defs handed to mock.patch: (download fakes, list_repo_tree method defs)."""
    dl_names: set[str] = set()
    api_names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "patch"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Name)
        ):
            target = node.args[0].value
            if target == "huggingface_hub.hf_hub_download":
                dl_names.add(node.args[1].id)
            elif target == "huggingface_hub.HfApi":
                api_names.add(node.args[1].id)
    dl_defs: list[ast.FunctionDef] = []
    tree_defs: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in dl_names:
            dl_defs.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in api_names:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "list_repo_tree":
                    tree_defs.append(item)
    return dl_defs, tree_defs


def _binds(fn: ast.FunctionDef, shape: tuple, *, method: bool) -> str | None:
    """None if the def accepts the call shape, else a human-readable reason."""
    n_pos, kw_names, has_splat = shape
    if has_splat:
        return None  # a **splat call site can't be statically bound — accept
    a = fn.args
    positional = [p.arg for p in (*a.posonlyargs, *a.args)]
    if method and positional and positional[0] == "self":
        positional = positional[1:]
    kwonly = {p.arg for p in a.kwonlyargs}
    if n_pos > len(positional) and a.vararg is None:
        return f"{fn.name}: {n_pos} positional args exceed {positional}"
    remaining = set(positional[n_pos:]) | kwonly
    for kw in kw_names:
        if kw in positional[:n_pos]:
            return f"{fn.name}: keyword {kw!r} collides with a positional slot"
        if kw not in remaining and a.kwarg is None:
            return (
                f"{fn.name}: does not accept keyword {kw!r} (params: {positional + sorted(kwonly)})"
            )
    return None


def test_smoke_hub_fakes_bind_every_production_call_shape():
    for script in SCRIPTS:
        tree = ast.parse(script.read_text())
        dl_shapes, tree_shapes = _collect_shapes(tree)
        dl_fakes, tree_fakes = _collect_fakes(tree)

        floors = MIN_FAKES[script.name]
        assert len(dl_fakes) >= floors["download"], (
            f"{script.name}: scanner found only {len(dl_fakes)} hf_hub_download fakes "
            f"(expected >= {floors['download']}) — fake population or patch pattern drifted"
        )
        assert len(tree_fakes) >= floors["list_repo_tree"], (
            f"{script.name}: scanner found only {len(tree_fakes)} HfApi.list_repo_tree fakes "
            f"(expected >= {floors['list_repo_tree']})"
        )
        assert dl_shapes, f"{script.name}: no production hf_hub_download call shapes found"
        assert tree_shapes, f"{script.name}: no production list_repo_tree call shapes found"

        failures: list[str] = []
        for fake in dl_fakes:
            for shape in dl_shapes:
                reason = _binds(fake, shape, method=False)
                if reason:
                    failures.append(f"{script.name}:{fake.lineno} {reason} vs call shape {shape}")
        for fake in tree_fakes:
            for shape in tree_shapes:
                reason = _binds(fake, shape, method=True)
                if reason:
                    failures.append(f"{script.name}:{fake.lineno} {reason} vs call shape {shape}")
        assert not failures, (
            "in-script smoke fakes no longer bind the production Hub call shapes "
            "(update the fakes alongside any kwarg-threading change):\n  " + "\n  ".join(failures)
        )
