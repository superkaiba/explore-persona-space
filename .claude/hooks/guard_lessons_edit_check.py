"""PreToolUse(Edit|Write) helper: gate prospective ``.claude/rules/LESSONS.md`` content (#1279).

Invoked by ``.claude/hooks/guard_lessons_edit.sh`` with the PreToolUse JSON on
stdin. Computes what LESSONS.md WOULD contain after the Edit/Write, then runs
the real #1269 checks (``scripts/workflow_lint.py::check_lessons_index`` —
imported at runtime, never re-implemented) against that prospective content.

Exit codes (the wrapper's contract):
  0 -> allow (content passes, or any internal failure: FAIL-OPEN)
  2 -> block (stdout carries the block message; the wrapper feeds it to stderr)

Materialization contract: ``check_lessons_index(repo_root=<td>)`` reads ONLY
``<td>/.claude/rules/`` (LESSONS.md bytes + the ``*.md`` stem set for index
parity), so the temp tree materializes exactly that — the prospective
LESSONS.md bytes plus zero-byte stubs named after every real ``*.md`` in the
EDITED tree's rules dir. If a future lint version reads outside
``<root>/.claude/rules/``, this materialization must grow with it.

Constant resolution: the lint module is imported from the EDITED tree's
``scripts/workflow_lint.py`` FIRST (so a same-diff ratchet/constant bump in
that tree is honored at edit time, per the #1269 constant-first ordering),
falling back to this hook's own repo copy.

``--print-constants`` dumps the resolved (own-repo) lint constants as JSON —
used by the wrapper's ``--self-test`` and the pytest suite to size fixtures at
runtime (never hardcoded byte sizes).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

_CONST_KEYS = (
    "_LESSONS_MAX_BYTES",
    "_LESSONS_WARN_BYTES",
    "_LESSONS_RATCHET_BYTES",
    "_LESSONS_RATCHET_MAX_HEADROOM_BYTES",
    "_LESSONS_ROW_MAX_BYTES",
    "_LESSONS_ROW_GRANDFATHER_MAX_BYTES",
    "_LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES",
)

# The hook's own repo root: <repo>/.claude/hooks/guard_lessons_edit_check.py
_OWN_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_lint(tree_root: Path, fallback_root: Path) -> ModuleType | None:
    """Import ``scripts/workflow_lint.py`` from the edited tree, else the hook's repo.

    Returns the loaded module, or None when neither copy imports (caller
    fails open). W1 (#1279 fact-check): the module MUST be registered in
    ``sys.modules`` BEFORE ``exec_module`` — dataclass creation inside the
    module resolves ``cls.__module__`` via ``sys.modules`` and crashes
    otherwise — under a per-tree-distinct spec name so the edited-tree and
    fallback copies never collide.
    """
    for root in (tree_root, fallback_root):
        try:
            lint_path = (root / "scripts" / "workflow_lint.py").resolve()
            if not lint_path.is_file():
                continue
            name = f"_wf_lint_{hashlib.sha1(str(lint_path).encode()).hexdigest()[:8]}"
            if name in sys.modules:
                return sys.modules[name]
            spec = importlib.util.spec_from_file_location(name, lint_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # W1: register BEFORE exec_module
            try:
                spec.loader.exec_module(module)
            except Exception:
                del sys.modules[spec.name]
                continue
            if not hasattr(module, "check_lessons_index"):
                continue
            return module
        except Exception:
            continue
    return None


def prospective(tool_name: str, tool_input: dict, current: str) -> str | None:
    """Post-edit LESSONS.md content, or None when there is nothing to guard.

    Write -> ``content`` verbatim. Edit -> apply old_string -> new_string to
    the on-disk content (first occurrence, or all with ``replace_all``). An
    Edit whose old_string is absent or ambiguous (count > 1 without
    replace_all) returns None: the Edit tool errors on those itself, so no
    file change will happen and there is nothing to guard. Unknown tool names
    fall back to shape inference (``content`` -> Write, ``old_string`` ->
    Edit) for robustness against harness tool_name drift.
    """
    if tool_name not in ("Write", "Edit"):
        if "content" in tool_input:
            tool_name = "Write"
        elif "old_string" in tool_input:
            tool_name = "Edit"
        else:
            return None
    if tool_name == "Write":
        content = tool_input.get("content")
        return content if isinstance(content, str) else None
    old = tool_input.get("old_string")
    new = tool_input.get("new_string", "")
    if not isinstance(old, str) or not old or not isinstance(new, str):
        return None
    n = current.count(old)
    if n == 0:
        return None
    if n > 1 and not tool_input.get("replace_all"):
        return None
    if tool_input.get("replace_all"):
        return current.replace(old, new)
    return current.replace(old, new, 1)


def _block_message(errors: list[str]) -> str:
    sentinel = _OWN_REPO_ROOT / ".claude" / "cache" / "allow-lessons-edit"
    lines = [
        "BLOCKED: this Edit/Write to .claude/rules/LESSONS.md would fail the #1269",
        "byte-budget / index-parity gates (#1279 edit-time guard). Findings on the",
        "PROSPECTIVE post-edit content:",
    ]
    lines += [f"  - {e}" for e in errors]
    lines += [
        "Recovery:",
        "  - Growing the index deliberately? Raise _LESSONS_RATCHET_BYTES in",
        "    scripts/workflow_lint.py of THIS tree FIRST (same-diff constant bump),",
        "    then retry this edit. Trimming? Ratchet the same constant DOWN after.",
        "  - Adding a row for a new rule? Create .claude/rules/<name>.md BEFORE",
        "    adding its index row.",
        "  - Verify after fixing: uv run python scripts/workflow_lint.py --check-lessons-index",
        "  - Sanctioned-maintenance escape hatches: set EPM_ALLOW_LESSONS_EDIT=1",
        f"    (session env), or run: touch {sentinel}",
        "    (honored for 15 min; that absolute path, regardless of your cwd).",
        "  - Do NOT route around via Bash writes (sed -i / tee / >>) — that re-opens",
        "    the gap this guard closes; the commit-time lint still gates those paths.",
    ]
    return "\n".join(lines)


def main() -> int:
    if "--print-constants" in sys.argv:
        wl = load_lint(_OWN_REPO_ROOT, _OWN_REPO_ROOT)
        if wl is None:
            print("{}")
            return 1
        print(json.dumps({k: getattr(wl, k) for k in _CONST_KEYS}))
        return 0

    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input") or {}
    raw_fp = tool_input.get("file_path")
    if not isinstance(raw_fp, str) or not raw_fp:
        return 0
    fp = Path(raw_fp)
    if not fp.is_absolute():
        fp = Path(payload.get("cwd") or os.getcwd()) / fp
    # normpath BEFORE the suffix check: a `..`-bearing path must resolve to its
    # canonical shape both for the suffix decision and for tree_root derivation.
    fp = Path(os.path.normpath(str(fp)))
    if fp.parts[-3:] != (".claude", "rules", "LESSONS.md"):
        return 0
    tree_root = fp.parents[2]

    wl = load_lint(tree_root, _OWN_REPO_ROOT)
    if wl is None:
        return 0  # fail-open: no importable lint anywhere

    current = fp.read_text(encoding="utf-8") if fp.is_file() else ""
    new = prospective(str(payload.get("tool_name", "")), tool_input, current)
    if new is None:
        return 0

    with tempfile.TemporaryDirectory() as td:
        rules = Path(td) / ".claude" / "rules"
        rules.mkdir(parents=True)
        live = tree_root / ".claude" / "rules"
        if live.is_dir():
            for p in live.glob("*.md"):
                if p.is_file() and p.name != "LESSONS.md":
                    (rules / p.name).touch()  # index parity globs stems only
        (rules / "LESSONS.md").write_bytes(new.encode("utf-8"))  # BYTE semantics (#1269)
        # Defaults bind the RESOLVED module's own constants; WARNs (advisory
        # band) are discarded on allow, exactly like the commit-time lint.
        errors = wl.check_lessons_index(repo_root=Path(td), warn_sink=[])

    if not errors:
        return 0
    print(_block_message(errors))
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)  # in-helper FAIL-OPEN backstop: a guard bug never wedges editing
