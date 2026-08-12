#!/usr/bin/env python3
"""#2122: extract the ACTUAL guard-blocked commands from cited transcript rows.

Context discipline (trigger-dense-review.md): prints ONLY the offending command
string + the guard's arm label — no surrounding conversation, no wholesale paging.
"""

import json
import sys
from pathlib import Path

PROJ_ROOT = Path.home() / ".claude/projects"


def find_transcript(prefix: str):
    """Search EVERY project dir (worktree sessions get their own dir)."""
    return sorted(PROJ_ROOT.glob(f"*/{prefix}*.jsonl"))

# (session-prefix, cited 1-indexed row) from task #2122's body
CITED = [("2f4940f0", 294), ("b765cdcd", 691), ("8d7f8b25", 4927)]

MARKER = "would move the SHARED repo-root tree off main"


def tool_use_commands(rows):
    """id -> command for every Bash tool_use in the transcript."""
    out = {}
    for r in rows:
        msg = r.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                inp = block.get("input") or {}
                if "command" in inp:
                    out[block.get("id")] = inp["command"]
    return out


def main() -> int:
    for prefix, cited_row in CITED:
        matches = find_transcript(prefix)
        if not matches:
            print(f"### {prefix} row {cited_row}: TRANSCRIPT NOT FOUND\n")
            continue
        path = matches[0]
        rows = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({})
        cmds = tool_use_commands(rows)
        print(f"### {prefix} (rows={len(rows)}), cited row {cited_row}")

        # Find every guard-block tool_result in the file; report the cited one first.
        hits = []
        for idx, r in enumerate(rows, start=1):
            blob = json.dumps(r)
            if MARKER not in blob:
                continue
            # locate the tool_use_id this result answers
            tuid = None
            msg = r.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tuid = block.get("tool_use_id")
            hits.append((idx, tuid))

        print(f"  guard-block rows in this transcript: {len(hits)} -> {[h[0] for h in hits]}")
        for idx, tuid in hits:
            tag = " <== CITED" if idx == cited_row else ""
            cmd = cmds.get(tuid, "<command not resolved>")
            print(f"  row {idx}{tag}")
            print(f"    CMD: {cmd}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
