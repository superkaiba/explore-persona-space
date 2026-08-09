#!/usr/bin/env python3
"""#1217 plan §6.4 pre-ship replay, phase 1: extract corpus-pattern tool_use INPUTS.

Scans the Claude Code session-transcript store (main project dir + the
worktree-suffixed project dirs) for Bash `command` and Read
`file_path`/`limit`/`offset` tool_use inputs matching the corpus-class loose
predicate, and writes them to work files under /tmp/i1217_replay/.

Context discipline (plan §6.4): reads tool_use INPUT records only — tool
RESULTS (which carry raw corpus text) are never parsed past the cheap line
prefilter, and nothing from the transcripts is printed to stdout beyond
counts. Work files are consumed by issue1217_replay_hook.py.
"""

import json
import re
import sys
import time
from pathlib import Path

WORK = Path("/tmp/i1217_replay")
PROJECTS = Path.home() / ".claude/projects"
STORE_GLOBS = [
    "-home-thomasjiralerspong-explore-persona-space",
    "-home-thomasjiralerspong-explore-persona-space--claude-worktrees-*",
]
DAYS = 30
CORPUS_LOOSE = re.compile(r"raw_completions|lmsys|wildchat|sharegpt|chatbot[-_]?arena", re.I)


def main() -> None:
    WORK.mkdir(exist_ok=True)
    cutoff = time.time() - DAYS * 86400
    files: list[Path] = []
    for g in STORE_GLOBS:
        for d in PROJECTS.glob(g):
            files += [f for f in d.glob("*.jsonl") if f.stat().st_mtime >= cutoff]
    files.sort()
    n_lines = n_parse_err = n_bash = n_read = 0
    t0 = time.time()
    with open(WORK / "bash_cmds.jsonl", "w") as bout, open(WORK / "reads.jsonl", "w") as rout:
        for k, f in enumerate(files):
            if k % 100 == 0:
                print(f"[{time.time() - t0:.0f}s] file {k}/{len(files)}", flush=True)
            try:
                with open(f, errors="replace") as fh:
                    for line in fh:
                        n_lines += 1
                        if '"tool_use"' not in line or not CORPUS_LOOSE.search(line):
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            n_parse_err += 1
                            continue
                        content = (d.get("message") or {}).get("content")
                        if not isinstance(content, list):
                            continue
                        for c in content:
                            if not isinstance(c, dict) or c.get("type") != "tool_use":
                                continue
                            inp = c.get("input") or {}
                            name = c.get("name")
                            if name == "Bash":
                                cmd = inp.get("command") or ""
                                if cmd and CORPUS_LOOSE.search(cmd):
                                    bout.write(json.dumps({"command": cmd}) + "\n")
                                    n_bash += 1
                            elif name == "Read":
                                fp = inp.get("file_path") or ""
                                if fp and CORPUS_LOOSE.search(fp):
                                    row = {
                                        "file_path": fp,
                                        "limit": inp.get("limit"),
                                        "offset": inp.get("offset"),
                                    }
                                    rout.write(json.dumps(row) + "\n")
                                    n_read += 1
            except Exception as e:
                msg = f"WARN file skipped: {f.name}: {type(e).__name__}"
                print(msg, file=sys.stderr, flush=True)
    print(
        f"DONE files={len(files)} lines={n_lines} parse_err={n_parse_err} "
        f"bash_raw={n_bash} read_raw={n_read} wall={time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
