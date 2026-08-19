#!/usr/bin/env python3
"""#1217 plan §6.4 pre-ship replay, phase 2 (resumable): replay through the hook.

Usage: issue1217_replay_hook.py bash | summarize

Replays the Bash-leg commands extracted by issue1217_replay_extract.py through
THIS checkout's guard_harmful_bank_read.sh (payloads synthesized per the
PreToolUse contract), then classifies the Read leg under the plan-§4.2 rule
(numeric limit 1..200 -> allow; missing file -> INDETERMINATE, never allow;
size gate 256 KB) and runs the §12.7 incident path-shape trace on a synthetic
fixture. Results accumulate in /tmp/i1217_replay/bash_results.jsonl so the
bash phase is resumable across bounded foreground runs.

Context discipline: stdout carries ONLY counts and abstract class rows; denied
command text goes to work files (bash_denies.txt / read_denies.txt).
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOOK = str(Path(__file__).resolve().parents[1] / ".claude/hooks/guard_harmful_bank_read.sh")
WORK = Path("/tmp/i1217_replay")
GUARD_LOG = str(WORK / "replay-denies.log")
RESULTS = WORK / "bash_results.jsonl"

RAWC = re.compile(r"(^|/)raw_completions(/|\.[A-Za-z0-9]+$|$)")
CTOK = re.compile(r"(lmsys|wildchat|sharegpt|chatbot[-_]?arena)", re.I)
QB = re.compile(r"(^|/)query_banks/")

F_GREP = re.compile(r"\b(grep|rg|egrep|fgrep)\b")
F_JQ = re.compile(r"\bjq\b")
F_JQ_DUMP = re.compile(r"jq\s+(-[a-zA-Z]+\s+)*'(\.|\.\[\])'")
F_SED_BOUNDED = re.compile(r"sed\s+(-[a-zA-Z]*\s+)*-?n?\s*'[0-9]+,[0-9]+p'|sed\s+-n\s+'[0-9]")
F_HEADTAIL_N = re.compile(r"\b(head|tail)\b\s+(-c?\s*-?n?\s*[0-9]+|-n\s*[0-9]+|-[0-9]+)")
F_PIPE = re.compile(r"\|")
F_PYTHON = re.compile(r"\b(python3?|uv run)\b")
F_LISTING = re.compile(r"^\s*(ls|find|du|wc|stat|sha256sum|md5sum|file|tree)\b")
F_GUARD_DEV = re.compile(
    r"test_guard_harmful_bank_read|guard_harmful_bank_read\.sh|i1217|bank-guard|guard_log_dump"
)
REASON_VERB = re.compile(r"BLOCKED: '([^']+)'")
REASON_CLASS = [
    ("read-window", re.compile(r"Read without a bounded window")),
    ("git-paging", re.compile(r"BLOCKED: git ")),
    ("jq-whole-dump", re.compile(r"whole-dump of a corpus file")),
    ("taskpy-embed", re.compile(r"task\.py --file/--body-file")),
    ("bare-diff", re.compile(r"'diff' outside git")),
    ("corpus-paging-verb", re.compile(r"would page corpus text wholesale into context")),
    ("bank-class", re.compile(r"bank", re.I)),
]

_HOOK_ENV = {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_BANK_READ"}
_HOOK_ENV["EPM_BANK_GUARD_LOG"] = GUARD_LOG
_LOCK = threading.Lock()


def is_corpus_path(p: str) -> bool:
    p = p.strip("'\"")
    if not p or QB.search(p):
        return False
    return bool(RAWC.search(p) or CTOK.search(p))


def run_hook(payload: dict, timeout: int = 400) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        r = subprocess.run(
            ["bash", HOOK],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env=_HOOK_ENV,
            timeout=timeout,
        )
        return r.returncode, r.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return -1, "HOOK-TIMEOUT", time.time() - t0


def reason_of(stderr: str) -> str:
    for name, rx in REASON_CLASS:
        if rx.search(stderr):
            return name
    return "other"


def features_of(cmd: str) -> str:
    feats = []
    if F_GUARD_DEV.search(cmd):
        feats.append("guard-dev")
    if F_GREP.search(cmd):
        feats.append("grep")
    if F_JQ.search(cmd):
        feats.append("jq-dump" if F_JQ_DUMP.search(cmd) else "jq-field")
    if F_SED_BOUNDED.search(cmd):
        feats.append("sed-bounded")
    if F_HEADTAIL_N.search(cmd):
        feats.append("headtail-N")
    if F_PIPE.search(cmd):
        feats.append("pipe")
    if F_PYTHON.search(cmd):
        feats.append("python")
    if F_LISTING.search(cmd):
        feats.append("listing-src")
    return "+".join(feats) or "-"


def load_uniq() -> list[str]:
    cmds = [json.loads(line)["command"] for line in open(WORK / "bash_cmds.jsonl")]
    return list(dict.fromkeys(cmds))


def phase_bash(budget_s: int = 480) -> None:
    uniq = load_uniq()
    done: set[int] = set()
    if RESULTS.exists():
        for line in open(RESULTS):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    todo = [(i, c) for i, c in enumerate(uniq) if i not in done]
    print(f"bash_unique={len(uniq)} done={len(done)} todo={len(todo)}", flush=True)
    t0 = time.time()
    stop = threading.Event()
    out = open(RESULTS, "a")

    def one(item: tuple[int, str]) -> None:
        i, cmd = item
        if stop.is_set():
            return
        rc, err, wall = run_hook({"tool_name": "Bash", "tool_input": {"command": cmd}})
        row: dict = {"idx": i, "rc": rc, "wall": round(wall, 2)}
        if rc == 2:
            row["reason"] = reason_of(err)
            m = REASON_VERB.search(err)
            row["verb"] = m.group(1) if m else row["reason"]
            row["features"] = features_of(cmd)
        with _LOCK:
            out.write(json.dumps(row) + "\n")
            out.flush()

    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(one, item) for item in todo]
        while True:
            n_done = sum(f.done() for f in futs)
            if n_done == len(futs):
                break
            if time.time() - t0 > budget_s:
                stop.set()
                print(f"[budget] stopping at {n_done}/{len(futs)}", flush=True)
                break
            time.sleep(5)
            print(f"[{time.time() - t0:.0f}s] {n_done}/{len(futs)}", flush=True)
    out.close()
    print("PHASE-BASH-EXIT", flush=True)


def phase_summarize() -> None:
    uniq = load_uniq()
    rows = {}
    for line in open(RESULTS):
        r = json.loads(line)
        rows[r["idx"]] = r
    print(f"bash_unique={len(uniq)} results={len(rows)}")
    cnt: Counter = Counter()
    denies = []
    for i, r in rows.items():
        if r["rc"] == 2:
            cnt["deny"] += 1
            denies.append((i, r))
        elif r["rc"] == 0:
            cnt["allow"] += 1
        elif r["rc"] == -1:
            cnt["hook-timeout"] += 1
        else:
            cnt[f"rc={r['rc']}"] += 1
    for k, v in cnt.most_common():
        print(f"  {k}: {v}")
    walls = sorted(r["wall"] for r in rows.values())
    if walls:
        n = len(walls)
        print(
            f"hook wall s: median={walls[n // 2]} p90={walls[int(n * 0.9)]} "
            f"p99={walls[int(n * 0.99)]} max={walls[-1]}"
        )
    with open(WORK / "bash_denies.txt", "w") as out:
        for i, r in sorted(denies):
            out.write(f"### idx={i} reason={r['reason']} verb={r['verb']}\n{uniq[i]}\n\n")
    print("--- bash deny table (idx | reason-class | verb | features)")
    for i, r in sorted(denies):
        print(f"{i} | {r['reason']} | {r['verb']} | {r['features']}")
    print("--- deny reason-class counts")
    for k, v in Counter(r["reason"] for _, r in denies).most_common():
        print(f"  {k}: {v}")

    reads = [json.loads(line) for line in open(WORK / "reads.jsonl")]
    uniq_r = list(dict.fromkeys((r["file_path"], r.get("limit"), r.get("offset")) for r in reads))
    print(f"read_raw={len(reads)} read_unique={len(uniq_r)}")
    cls: Counter = Counter()
    mismatches = 0
    with open(WORK / "read_denies.txt", "w") as out:
        for i, (fp, limit, offset) in enumerate(uniq_r):
            if not is_corpus_path(fp):
                cls["not-predicate-matched"] += 1
                continue
            bounded = isinstance(limit, int) and 1 <= limit <= 200
            if bounded:
                verdict = "allow-bounded"
            else:
                try:
                    sz = os.stat(fp).st_size
                except OSError:
                    verdict = "indeterminate-missing"
                else:
                    verdict = "allow-small" if sz <= 262144 else "deny-oversize"
            cls[verdict] += 1
            if verdict != "indeterminate-missing":
                payload: dict = {"tool_name": "Read", "tool_input": {"file_path": fp}}
                if limit is not None:
                    payload["tool_input"]["limit"] = limit
                if offset is not None:
                    payload["tool_input"]["offset"] = offset
                rc, _err, _w = run_hook(payload)
                if (rc == 2) != (verdict == "deny-oversize"):
                    mismatches += 1
                    print(f"HOOK/CLASSIFIER MISMATCH read idx {i}: {verdict} rc={rc}")
            if verdict == "deny-oversize":
                out.write(f"idx={i}\tlimit={limit}\toffset={offset}\t{fp}\n")
    print("--- read leg classification")
    for k, v in cls.most_common():
        print(f"  {k}: {v}")
    print(f"read_hook_classifier_mismatches={mismatches}")

    # §12.7 shape trace (secondary evidence; NOT the §6.4-bis actual-call trace).
    inc = WORK / "incident/issue1073_decode_regime/raw_completions/greedy"
    inc.mkdir(parents=True, exist_ok=True)
    shard = inc / "greedy.shard000.json"
    if not shard.exists() or shard.stat().st_size <= 262144:
        rows_f = [
            {"ci": i, "ri": 0, "text": "synthetic filler " * 40, "n_tokens": 2} for i in range(500)
        ]
        shard.write_text(json.dumps(rows_f))
    assert shard.stat().st_size > 262144
    rc_read, _, _ = run_hook({"tool_name": "Read", "tool_input": {"file_path": str(shard)}})
    rc_bash, _, _ = run_hook({"tool_name": "Bash", "tool_input": {"command": f"cat {shard}"}})
    rc_read_b, _, _ = run_hook(
        {"tool_name": "Read", "tool_input": {"file_path": str(shard), "limit": 50, "offset": 100}}
    )
    print("--- incident shape trace (synthetic fixture at recorded path shape)")
    print(
        f"read_nolimit_rc={rc_read} (want 2)  bash_page_rc={rc_bash} (want 2)  "
        f"read_bounded_rc={rc_read_b} (want 0)"
    )


if __name__ == "__main__":
    {"bash": phase_bash, "summarize": phase_summarize}[sys.argv[1]]()
