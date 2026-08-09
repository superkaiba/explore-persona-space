#!/usr/bin/env python3
"""#1217 plan §6.4: structural sanctioned-vs-wholesale classification of Bash denies.

Consumes /tmp/i1217_replay/bash_denies.txt (from issue1217_replay_hook.py
summarize). Buckets are assigned by verb family + bounding/consumption
constructs (clause-(d) sanctioned routes: grep-family excerpt pulls, bounded
sed windows, jq field-access digests, script composition, pipeline
consumption), never by transcript context. Prints bucket counts + idx lists
only — no command literals.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path("/tmp/i1217_replay")

R_HEREDOC_COMPOSE = re.compile(r"(cat|tee)\s*(>>?|-)\s*\S+\s*<<|<<\s*['\"]?(EOF|PY|SH|EOS)")
R_GUARD_DEV = re.compile(
    r"test_guard_harmful_bank_read|guard_harmful_bank_read\.sh|i1217|bank-guard|guard_log_dump"
)
R_GREP_ON_CORPUS = re.compile(
    r"\b(grep|rg)\b[^|;&]*?(raw_completions|lmsys|wildchat|sharegpt|chatbot[-_]?arena)", re.I
)
R_SED_BOUNDED = re.compile(r"sed\s+(-[a-zA-Z]*\s+)*-?n?\s*'?[0-9]+,[0-9]+p|sed\s+-n\s+'[0-9]")
R_JQ_FIELD = re.compile(
    r"\bjq\b\s+(-[a-zA-Z]+\s+)*'?\s*(\.[A-Za-z_\[]|length|keys|type|to_entries|map|group_by|\[)"
)
R_PIPE_PY = re.compile(r"\|\s*(uv run |)python3?\b|\|\s*uv run\b")
R_CAT_PIPE = re.compile(r"\bcat\b[^|;&>]*\|")
R_HEADTAIL_N = re.compile(r"\b(head|tail)\b\s+(-c?\s*-?n?\s*[0-9]+|-n\s*[0-9]+|-[0-9]+)")
R_WC_DIGEST = re.compile(r"\b(wc|sha256sum|md5sum|du|stat|ls|file)\b")
R_PY_HEREDOC = re.compile(r"(python3?|uv run python)\s*(-\s*)?<<|<<\s*'?PY'?")

TEXTUTIL_VERBS = (
    "sort",
    "uniq",
    "cut",
    "awk",
    "nl",
    "column",
    "paste",
    "comm",
    "join",
    "rev",
    "fold",
    "fmt",
    "pr",
)
PAGER_VERBS = (
    "cat",
    "less",
    "more",
    "bat",
    "tac",
    "nl",
    "od",
    "xxd",
    "hexdump",
    "strings",
    "base64",
    "json.tool",
)


def main() -> None:
    blocks = []
    cur = None
    for line in open(WORK / "bash_denies.txt"):
        m = re.match(r"### idx=(\d+) reason=(\S+) verb=(\S+)", line)
        if m:
            if cur:
                blocks.append(cur)
            cur = {"idx": int(m.group(1)), "reason": m.group(2), "verb": m.group(3), "cmd": ""}
        elif cur is not None:
            cur["cmd"] += line
    if cur:
        blocks.append(cur)

    rows = []
    for b in blocks:
        c = b["cmd"]
        consumerish = (
            R_JQ_FIELD.search(c) or R_PIPE_PY.search(c) or R_WC_DIGEST.search(c) or "|" in c
        )
        if R_GUARD_DEV.search(c):
            bucket = "guard-dev-fixture"
        elif R_HEREDOC_COMPOSE.search(c) or R_PY_HEREDOC.search(c):
            bucket = "sanctioned:heredoc-script-compose"
        elif b["reason"] in ("git-paging", "bare-diff", "jq-whole-dump", "taskpy-embed"):
            bucket = f"wholesale:{b['reason']}"
        elif R_SED_BOUNDED.search(c):
            bucket = "sanctioned:bounded-sed-window"
        elif R_GREP_ON_CORPUS.search(c):
            bucket = "sanctioned:grep-excerpt-pull"
        elif b["verb"] in TEXTUTIL_VERBS and consumerish:
            bucket = "sanctioned:digest-pipeline-textutil"
        elif (
            b["verb"] == "cat"
            and R_CAT_PIPE.search(c)
            and (R_PIPE_PY.search(c) or R_JQ_FIELD.search(c))
        ):
            bucket = "sanctioned:cat-pipe-consumer"
        elif b["verb"] == "cat" and (">" in c):
            bucket = "sanctioned:cat-redirect-compose"
        elif b["verb"] in PAGER_VERBS:
            bucket = "wholesale:page-into-context"
        elif b["verb"] in ("sed", "awk"):
            bucket = "ambiguous:sed-awk-transform"
        else:
            bucket = "ambiguous:other"
        rows.append((b["idx"], bucket, b["verb"], len(c)))

    cnt = Counter(r[1] for r in rows)
    print(f"--- bucket counts (total={len(rows)})")
    for k, v in cnt.most_common():
        print(f"  {k}: {v}")
    print("--- idx lists per bucket")
    byb = defaultdict(list)
    for idx, bucket, _verb, _ln in rows:
        byb[bucket].append(idx)
    for k in sorted(byb):
        print(f"{k}: {byb[k]}")
    print("--- ambiguous rows (idx, verb, cmd_len)")
    for idx, bucket, verb, ln in rows:
        if bucket.startswith("ambiguous"):
            print(f"  {idx} {verb} len={ln}")


if __name__ == "__main__":
    main()
