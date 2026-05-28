"""Regression test: dashboard editor packages stay declared in package.json.

Issue #18: ``dashboard/app/tasks/[id]/edit/Editor.tsx`` (and
``InlineBodyEditor.tsx``) statically import CodeMirror packages
(``@uiw/react-codemirror`` via ``next/dynamic`` and
``@codemirror/lang-markdown``). When those packages are present in a
local ``node_modules`` but absent from ``dashboard/package.json``
dependencies, ``npm install`` on a fresh checkout (Vercel) resolves the
import to nothing and the build dies with
``Module not found: Can't resolve '@uiw/react-codemirror'``.

The fix is to declare every directly-imported editor package as a top
-level dependency in ``dashboard/package.json`` so a clean ``npm ci`` on
Vercel pulls them. This test pins the invariant: the editor packages
that ``Editor.tsx`` / ``InlineBodyEditor.tsx`` depend on directly MUST
appear in ``dashboard/package.json`` ``dependencies``. A future edit
that adds an editor import without declaring the dep fails here instead
of only failing on the next Vercel deploy.

Scope note
----------

This test guards the directly-imported, top-level editor packages that
the inline + full-page markdown editors load. Transitively-provided
peers (``@codemirror/state``, ``@codemirror/language``,
``@mdxeditor/gurx``) are intentionally NOT required here: they resolve
through their parents (``@uiw/react-codemirror`` /
``@mdxeditor/editor``) and the build passes on them. The bug class issue
#18 fixed is specifically "import of a package family the project chose
to take a direct dependency on, but forgot to declare." The required set
below is the declared-dependency surface those editors need.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = REPO_ROOT / "dashboard"
PACKAGE_JSON = DASHBOARD / "package.json"

# Editor packages the inline + full-page markdown editors take a DIRECT
# dependency on. These must be declared in dashboard/package.json so a
# clean Vercel `npm ci` resolves the imports.
REQUIRED_EDITOR_DEPS = (
    "@uiw/react-codemirror",
    "@codemirror/lang-markdown",
    "@mdxeditor/editor",
)


def _declared_dependencies() -> dict[str, str]:
    """Return the ``dependencies`` map from dashboard/package.json."""
    assert PACKAGE_JSON.is_file(), f"missing {PACKAGE_JSON}"
    data = json.loads(PACKAGE_JSON.read_text())
    deps = data.get("dependencies", {})
    assert isinstance(deps, dict), f"dependencies is not an object: {type(deps)}"
    return deps


def test_required_editor_packages_are_declared() -> None:
    """Every editor package the markdown editors directly import is declared."""
    deps = _declared_dependencies()
    missing = [pkg for pkg in REQUIRED_EDITOR_DEPS if pkg not in deps]
    assert not missing, (
        "dashboard/package.json is missing editor dependencies that "
        f"Editor.tsx / InlineBodyEditor.tsx import: {missing}. "
        "Undeclared imports build locally but break `npm ci` on Vercel "
        "(issue #18). Add them to dependencies and re-run `npm install`."
    )


def test_codemirror_imports_have_a_declared_package() -> None:
    """Each statically-imported @uiw/@codemirror package family is declared.

    Walks the dashboard ``.ts`` / ``.tsx`` sources, collects every
    ``@uiw/...`` and ``@codemirror/...`` package referenced in a
    ``from "..."`` or ``import("...")`` clause, and asserts the
    importing package family is covered by a declared dependency. This is
    the precise regression surface for issue #18: a CodeMirror import
    that no declared package would resolve on a fresh checkout.
    """
    deps = _declared_dependencies()
    declared = set(deps)

    # Match: from "@uiw/react-codemirror" / import("@codemirror/lang-markdown")
    import_re = re.compile(r"""(?:from|import\()\s*["'](@(?:uiw|codemirror)/[^"']+)["']""")

    sources: list[Path] = []
    for pattern in ("*.ts", "*.tsx"):
        sources.extend(p for p in DASHBOARD.rglob(pattern) if "node_modules" not in p.parts)
    assert sources, f"found no dashboard sources under {DASHBOARD}"

    # A direct dependency that resolves the import is either an exact
    # match (e.g. "@uiw/react-codemirror") or a package that provides the
    # family transitively via a declared parent. For the @codemirror/*
    # families, @uiw/react-codemirror bundles the editor core; for the
    # specific @codemirror/lang-markdown grammar we declare it directly.
    # The rule we enforce: the imported package is declared OR it is a
    # @codemirror/* core peer that @uiw/react-codemirror pulls in.
    codemirror_core_provider = "@uiw/react-codemirror"
    transitively_provided_codemirror = {
        "@codemirror/state",
        "@codemirror/view",
        "@codemirror/language",
        "@codemirror/commands",
    }

    unresolved: list[str] = []
    for src in sources:
        for pkg in import_re.findall(src.read_text()):
            if pkg in declared:
                continue
            if pkg in transitively_provided_codemirror and codemirror_core_provider in declared:
                continue
            unresolved.append(f"{pkg} (imported in {src.relative_to(REPO_ROOT)})")

    assert not unresolved, (
        "CodeMirror imports with no declared package to resolve them on a "
        f"fresh checkout (issue #18 regression): {unresolved}. "
        "Declare the package in dashboard/package.json dependencies."
    )
