import glob
import json
import os
import re

PROJ = os.path.expanduser("~/.claude/projects/-home-thomasjiralerspong-explore-persona-space")
TARGETS = {
    "3622b473": "2026-07-17T06:53",
    "c0021509": "2026-07-16T23:11",
    "f5ee50cc": "2026-07-18T06:41",
}

GATED = re.compile(
    r"git\s+(checkout|switch|restore|reset|clean|merge|rebase|cherry-pick|revert|am)\b"
)
SHELLOUT = re.compile(
    r"os\.system|subprocess|Popen|check_call|check_output|getoutput|system *\(|from +os +import"
)
SHELLISH = re.compile(
    r"(^|[^A-Za-z0-9_.])(bash|sh|zsh|ksh|dash|eval|source|ssh|xargs|parallel|sudo|su)([^A-Za-z0-9_]|$)"
)
OPENER = re.compile(r"<<-?\s*(\\?['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1?")


def analyze(cmd):
    lines = cmd.split("\n")
    i, n = 0, len(lines)
    regions = []  # (kind, lines)
    while i < n:
        line = lines[i]
        m = OPENER.search(line.replace("<<<", "\x02"))
        if m and "<<" in line.replace("<<<", "\x02"):
            tag = m.group(2)
            quoted = bool(m.group(1))
            j = i + 1
            body = []
            while j < n and lines[j].lstrip("\t") != tag:
                body.append(lines[j])
                j += 1
            terminated = j < n
            regions.append(("live-opener", [line], quoted, tag, terminated))
            regions.append(("body", body, quoted, tag, terminated))
            i = j + 1
        else:
            regions.append(("live", [line], None, None, None))
            i += 1
    out = []
    for kind, rl, quoted, tag, term in regions:
        text = "\n".join(rl)
        gated = len(GATED.findall(text))
        so = len(SHELLOUT.findall(text))
        exp = ("$(" in text, "${" in text, "`" in text)
        shl = bool(SHELLISH.search(text))
        if kind == "body":
            out.append(
                f"  BODY tag={tag} quoted={quoted} terminated={term} lines={len(rl)} gated_verbs={gated} shellout={so} exp($(,${{,`)={exp} "
            )
        else:
            head = text[:90].replace("\n", " ")
            out.append(
                f"  {kind.upper():12s} gated_verbs={gated} shellout={so} shellish={shl} :: {head!r}"
            )
    return out


for path in sorted(glob.glob(PROJ + "/*.jsonl")):
    base = os.path.basename(path)[:8]
    if base not in TARGETS:
        continue
    pending = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if "tool_use" not in line and "guard_repo_root_branch" not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (row.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use" and b.get("name") == "Bash":
                cmd = (b.get("input") or {}).get("command", "")
                if "<<" in cmd:
                    pending[b.get("id")] = cmd
            if b.get("type") == "tool_result" and b.get("tool_use_id") in pending:
                txt = json.dumps(b.get("content", ""))
                if "guard_repo_root_branch" in txt and "BLOCKED" in txt:
                    ts = row.get("timestamp", "?")[:19]
                    print(f"== {base} ts={ts} len={len(pending[b['tool_use_id']])}")
                    for ln in analyze(pending[b["tool_use_id"]]):
                        print(ln)
