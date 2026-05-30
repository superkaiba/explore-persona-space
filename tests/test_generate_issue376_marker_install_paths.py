"""Regression test for ``scripts/generate_issue376_marker_install.py`` path
routing through ``_data_dir_for`` / ``_DATA_DIR_OVERRIDE``.

Task #408 v7 (2026-05-30) — round-7 fix for an INCOMPLETE ``--output-dir``
override that crashed the Phase A.0.a single-turn regen.  Round-2 wired
``--output-dir`` into a module-level ``_DATA_DIR_OVERRIDE`` and used it in
``assemble_step``, BUT seven other write sites in the same module still
referenced the hardcoded legacy ``DATA_DIR`` constant (``training_questions.json``
cache, ``eval_questions_v2.json`` cache, ``responses_cache.json`` cache at
three callsites, ``batch_id_responses.txt`` breadcrumb, etc.) — so
``--output-dir=data/issue376_marker_install_9ca040/`` redirected only the
final ``train.jsonl`` and crashed with ``FileNotFoundError`` the moment
``generate_training_questions`` tried to write its cache to the
non-existent legacy dir.

This test AST-scans the module and FAILs on any future ``DATA_DIR / "..."``
construction (the bug class that motivated v7).  Every cache / intermediate
/ checkpoint path MUST resolve through ``_data_dir_for(marker_text)`` so the
override stays comprehensive as the script evolves.

Per-line exemptions are NOT supported on purpose — if you genuinely need
to write under the legacy un-suffixed dir (e.g. for a new byte-identity
hardlink), build the path explicitly with ``Path(__file__).parent.parent /
"data" / "issue376_marker_install"`` rather than reusing ``DATA_DIR`` as a
write target.  Cleanly separates the legacy-back-compat use from the
override-aware write surface.
"""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_issue376_marker_install.py"


def _find_data_dir_write_sites(tree: ast.Module) -> list[tuple[int, str]]:
    """Return (lineno, source-snippet) for every ``DATA_DIR / <x>`` BinOp.

    Walks every node; flags any ``BinOp(op=Div, left=Name("DATA_DIR"))``.
    Catches all forms: ``DATA_DIR / "foo"``, ``DATA_DIR / f"{x}.json"``,
    ``DATA_DIR / some_var``, etc.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name)
            and node.left.id == "DATA_DIR"
        ):
            try:
                snippet = ast.unparse(node)
            except Exception:
                snippet = "<unparseable>"
            hits.append((node.lineno, snippet))
    return hits


def test_no_data_dir_write_sites_in_generate_script() -> None:
    """No ``DATA_DIR / "..."`` write sites in
    ``scripts/generate_issue376_marker_install.py``.

    All writes must go through ``_data_dir_for(marker_text)`` so the
    ``--output-dir`` CLI override (``_DATA_DIR_OVERRIDE``) routes EVERY
    output, cache, intermediate, and checkpoint to the caller-chosen
    directory.
    """
    src = SCRIPT.read_text()
    tree = ast.parse(src, filename=str(SCRIPT))
    hits = _find_data_dir_write_sites(tree)
    if hits:
        rendered = "\n".join(f"  line {ln}: {snip}" for ln, snip in hits)
        raise AssertionError(
            f"Found {len(hits)} ``DATA_DIR / ...`` write site(s) in {SCRIPT.name}; "
            f"all writes must route through ``_data_dir_for(marker_text)`` so the "
            f"``--output-dir`` override stays comprehensive (task #408 v7). "
            f"Offenders:\n{rendered}"
        )


def test_data_dir_for_respects_override(tmp_path) -> None:
    """``_data_dir_for(marker_text)`` returns the override dir verbatim when
    ``_DATA_DIR_OVERRIDE["path"]`` is set, ignoring ``marker_text``.

    This is the contract the rest of the module relies on: every cache
    write site can pass ``marker_text=MARKER_TOKEN`` (its default) AND
    still land in the override directory when the CLI override is active.
    """
    # Import via the importlib machinery the script package uses.
    import sys

    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import generate_issue376_marker_install as glm  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    override = tmp_path / "issue376_marker_install_TESTSLUG"
    original = glm._DATA_DIR_OVERRIDE["path"]
    try:
        glm._DATA_DIR_OVERRIDE["path"] = override
        # Both the default marker AND an arbitrary unicode marker should
        # resolve to the override path.
        from explore_persona_space.personas import MARKER_TOKEN

        for marker in (MARKER_TOKEN, "※", "[CUSTOM]"):
            resolved = glm._data_dir_for(marker)
            assert resolved == override, (
                f"_data_dir_for({marker!r}) returned {resolved}, expected override "
                f"{override} (override should win regardless of marker_text)"
            )
            assert resolved.exists(), (
                f"_data_dir_for did not create the override directory: {resolved}"
            )
    finally:
        glm._DATA_DIR_OVERRIDE["path"] = original


def test_data_dir_for_falls_back_to_marker_slug() -> None:
    """With no override set, ``_data_dir_for(marker_text)`` derives the
    path from the marker slug, producing distinct directories per marker.

    Calls the real ``_data_dir_for`` and asserts the resolved directory
    name matches ``issue376_marker_install_<slug>``.  Mkdir side-effect
    lands under the real repo's ``data/`` tree — acceptable because the
    slug-derived dir is idempotently created elsewhere too (and the test
    only checks the path shape, not contents).
    """
    import sys

    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import generate_issue376_marker_install as glm  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    from explore_persona_space.personas import MARKER_TOKEN, marker_slug

    # Confirm no override is currently set (defensive — previous tests
    # restore it, but assert here for clarity).
    assert glm._DATA_DIR_OVERRIDE["path"] is None, (
        "_DATA_DIR_OVERRIDE leaked between tests; aborting to avoid false PASS"
    )

    p_default = glm._data_dir_for(MARKER_TOKEN)
    p_other = glm._data_dir_for("※")
    assert p_default.name == f"issue376_marker_install_{marker_slug(MARKER_TOKEN)}", (
        f"Expected slug-derived dir for default marker; got {p_default}"
    )
    assert p_other.name == f"issue376_marker_install_{marker_slug('※')}", (
        f"Expected slug-derived dir for ※; got {p_other}"
    )
    assert p_default != p_other, (
        "Distinct marker_text values must map to distinct slug-derived dirs "
        f"(both came back as {p_default})"
    )
