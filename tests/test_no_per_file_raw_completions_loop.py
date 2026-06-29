"""AST regression test pinning the #664/#727 raw-completions refactor.

``upload_raw_completions_to_data_repo`` (orchestrate/hub.py) used to loop
``_upload(..., upload_as_file=True)`` once per ``raw_completions.json`` file.
On a large HF repo each ``upload_file`` triggers a server-side recursive
tree-listing pre-check that 504-times-out ~half the time, so a per-file loop of
N files stalls for hours (#664: 12h / ~$530 on an 8xH200 at 0% GPU, 264/1425
files uploaded). The fix routes the whole tree through ONE ``upload_folder``
commit (via the private ``_upload_folder_filtered`` helper).

This test walks the AST of ``upload_raw_completions_to_data_repo`` and asserts
its body contains NO per-file ``_upload`` / ``upload_file`` call inside a loop —
pinning the refactor as a workflow invariant so a future edit cannot silently
reintroduce the per-file loop. It is the durable structural guard the plain unit
tests (``tests/test_hub.py::TestUploadRawCompletions``, which mock the Hub) do
NOT provide: those assert behavior at one point in time; this asserts the shape
can't regress.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HUB_PATH = REPO_ROOT / "src" / "explore_persona_space" / "orchestrate" / "hub.py"
TARGET_FN = "upload_raw_completions_to_data_repo"


def _target_function() -> ast.FunctionDef:
    tree = ast.parse(HUB_PATH.read_text(), filename=str(HUB_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == TARGET_FN:
            return node
    raise AssertionError(f"{TARGET_FN} not found in {HUB_PATH}")


def _call_name(call: ast.Call) -> str | None:
    """Return the simple callee name of a Call node (``_upload`` for
    ``_upload(...)`` / ``hub._upload(...)``), else None."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_target_function_exists():
    """Guard against a rename silently turning every other check into a no-op."""
    assert _target_function() is not None


def test_no_per_file_upload_call_inside_a_loop():
    """No ``_upload`` / ``upload_file`` call may sit inside a for/while loop in
    ``upload_raw_completions_to_data_repo`` — that is exactly the #664 per-file
    loop. (Deleting the local files in a loop is fine; uploading per-file is
    not.)"""
    fn = _target_function()
    offenders: list[str] = []
    for loop in ast.walk(fn):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        for inner in ast.walk(loop):
            if isinstance(inner, ast.Call) and _call_name(inner) in {
                "_upload",
                "upload_file",
            }:
                offenders.append(
                    f"line {inner.lineno}: per-file '{_call_name(inner)}(...)' "
                    "inside a loop — the #664 per-file upload loop must not return"
                )
    assert not offenders, (
        "upload_raw_completions_to_data_repo must NOT loop a per-file upload "
        "(use ONE upload_folder commit). Offenders:\n" + "\n".join(offenders)
    )


def test_no_upload_as_file_true_anywhere_in_function():
    """The refactored function uploads a FOLDER, so it must never pass
    ``upload_as_file=True`` — a single-file upload signal anywhere in its body
    would mean the per-file path crept back."""
    fn = _target_function()
    offenders: list[int] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (
                    kw.arg == "upload_as_file"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    offenders.append(node.lineno)
    assert not offenders, (
        "upload_raw_completions_to_data_repo must not pass upload_as_file=True "
        f"(folder-commit refactor, #727); found at lines {offenders}"
    )


def test_calls_a_folder_commit_helper():
    """Positive assertion: the function must route through a folder-commit path
    (``_upload_folder_filtered`` or a direct ``upload_folder``) — so the test
    fails if the body is gutted to no upload at all rather than only if the loop
    returns."""
    fn = _target_function()
    names = {_call_name(c) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert names & {"_upload_folder_filtered", "upload_folder"}, (
        "upload_raw_completions_to_data_repo must call a folder-commit helper "
        f"(_upload_folder_filtered / upload_folder); found calls: {sorted(n for n in names if n)}"
    )
