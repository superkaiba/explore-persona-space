#!/usr/bin/env python3
"""P1 consumer audit for the #2332 prefix repack (plan v2 SS4.3). Pure local +
fleet-list — this script makes ZERO HuggingFace Hub API calls by design.

For each of the 8 target prefixes on ``superkaiba1/explore-persona-space-data``,
sweep FOUR surfaces (the AUDITED BOUNDARY, restated verbatim on every
per-prefix verdict line) for any file whose CONTENT mentions the prefix:

  1. Repo root, full tree: filesystem grep from the main checkout root,
     excludes .git + worktrees (worktrees are surface 2).
  2. Every live worktree, full tree, bounded excludes
     (.git,.venv,hf_dl,g*_dl,store,__pycache__,node_modules) — filesystem
     grep, NOT git grep, so untracked files are visible.
  3. The owning issue branch, full tree: ``git grep -l <prefix>
     origin/issue-<M>`` after a ``git fetch origin`` (absent branch recorded
     as absent).
  4. Fleet/service consumers: ``spawn_session.py list`` + ``pod.py
     list-ephemeral`` (RunPod API — NOT the HF Hub) + a grep of the EPS
     dashboard source dir (located via the eps-dashboard.service systemd
     unit's WorkingDirectory; absent unit recorded as not-present). Every leg
     is a POSITIVE recorded line (found / none-found), never silence.

Classification of every hit file (plan SS4.3 step 4):
  reader       code file (py/ipynb/sh/js/ts/...) whose content co-occurs with
               a Hub READ token (hf_hub_download / snapshot_download /
               list_repo_tree / list_repo_files / resolve/ URL / hf:// ...)
  writer       code file co-occurring with an upload call site
  inert        task bodies, events.jsonl, eval-result JSONs, docs, figure
               metadata, data files — provenance strings, not I/O; also code
               files with NO I/O-token co-occurrence (recorded as such)
  self-tooling-2332  this task's own audit/repack tooling + artifacts
               (excluded from reader counts, recorded explicitly)

Verdict per prefix: ``no-live-reader`` (0 reader-class hits, boundary stated)
or ``readers-found`` (hits listed; the trivially-updatable vs needs-accessor
triage is the orchestrator's branch decision per the #2332 brief). Every
verdict also records the owning issue's lifecycle status from REGISTRY.

Outputs: eval_results/issue_2332/consumer_audit.json + consumer_audit.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.task_workflow import primary_checkout_root, registry_path

# prefix -> owning issue (plan v2 SS5 work-item table).
PREFIXES: dict[str, int] = {
    "issue1489_ctx_aug": 1489,
    "issue2224_screening": 2224,
    "issue1434_writingstyle": 1434,
    "issue667_alllayer": 667,
    "issue1586_methodgen": 1586,
    "issue1090_pvdatagen": 1090,
    "issue1481_conpos_grid": 1481,
    "issue1739_ctxmap": 1739,
}

ROOT_EXCLUDE_DIRS = [".git", "worktrees"]
WT_EXCLUDE_DIRS = [".git", ".venv", "hf_dl", "g*_dl", "store", "__pycache__", "node_modules"]
DASH_EXCLUDE_DIRS = [".git", "node_modules", ".next"]

CODE_SUFFIXES = {".py", ".ipynb", ".sh", ".bash", ".zsh", ".js", ".mjs", ".ts", ".tsx", ".jsx"}
READ_TOKENS = [
    "hf_hub_download",
    "snapshot_download",
    "list_repo_tree",
    "list_repo_files",
    "load_dataset",
    "get_paths_info",
    "file_exists",
    "resolve/",
    "hf://",
    "stage_hub_prefix",
    "stage_hub_file",
]
WRITE_TOKENS = [
    "upload_file",
    "upload_folder",  # NOT a substring of upload_large_folder — both listed
    "upload_large_folder",
    "create_commit",
    "CommitOperationAdd",
    "upload_raw_completions_to_data_repo",
]
# This task's own tooling/artifacts: recorded, never counted as readers.
SELF_TOOLING_MARKERS = ("issue2332", "/2332/")

MAX_SCAN_BYTES_PER_CHUNK = 8 << 20
SCAN_OVERLAP = 4 << 10  # > max token/prefix length


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _grep_base_flags() -> list[str]:
    """ugrep skips hidden files by default (GNU grep does not) — probe once."""
    p = subprocess.run(["grep", "--version"], capture_output=True, text=True)
    if p.returncode == 0 and "ugrep" in p.stdout.splitlines()[0]:
        return ["--hidden"]
    return []


def _run_grep(root: Path, exclude_dirs: list[str], extra_flags: list[str]) -> list[str]:
    """Combined fixed-string candidate pass: one full read of the tree for all
    8 prefixes. rc 0 = hits, rc 1 = no hits, anything else = fail loud."""
    cmd = ["grep", "-r", "-l", "-F", *extra_flags]
    cmd += [f"--exclude-dir={d}" for d in exclude_dirs]
    for prefix in PREFIXES:
        cmd += ["-e", prefix]
    cmd.append(".")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
    if p.returncode not in (0, 1):
        raise SystemExit(
            f"FATAL: grep failed rc={p.returncode} under {root}: {p.stderr.strip()[:400]}"
        )
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def _scan_file(path: Path) -> tuple[set[str], set[str], set[str]]:
    """One streaming pass over a candidate file: which prefixes it mentions +
    which read/write tokens co-occur. Chunked with overlap (binary-safe)."""
    prefixes_found: set[str] = set()
    reads: set[str] = set()
    writes: set[str] = set()
    tail = b""
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(MAX_SCAN_BYTES_PER_CHUNK)
                if not chunk:
                    break
                window = tail + chunk
                for prefix in PREFIXES:
                    if prefix not in prefixes_found and prefix.encode() in window:
                        prefixes_found.add(prefix)
                for tok in READ_TOKENS:
                    if tok not in reads and tok.encode() in window:
                        reads.add(tok)
                for tok in WRITE_TOKENS:
                    if tok not in writes and tok.encode() in window:
                        writes.add(tok)
                tail = window[-SCAN_OVERLAP:]
    except OSError as e:
        log(f"  WARN: unreadable candidate {path}: {e}")
    return prefixes_found, reads, writes


def _classify(rel_path: str, reads: set[str], writes: set[str]) -> list[str]:
    if any(m in rel_path for m in SELF_TOOLING_MARKERS):
        return ["self-tooling-2332"]
    if Path(rel_path).suffix.lower() not in CODE_SUFFIXES:
        return ["inert"]
    classes: list[str] = []
    if reads:
        classes.append("reader")
    if writes:
        classes.append("writer")
    return classes or ["inert"]  # code file, no I/O-token co-occurrence


def _worktree_paths(root: Path) -> list[Path]:
    p = subprocess.run(
        ["git", "worktree", "list", "--porcelain"], capture_output=True, text=True, cwd=str(root)
    )
    if p.returncode != 0:
        raise SystemExit(f"FATAL: git worktree list failed rc={p.returncode}: {p.stderr[:300]}")
    out: list[Path] = []
    for block in p.stdout.split("\n\n"):
        lines = [ln for ln in block.strip().splitlines() if ln]
        if not lines or not lines[0].startswith("worktree "):
            continue
        wt = Path(lines[0].removeprefix("worktree ").strip())
        if wt.resolve() != root.resolve():
            out.append(wt)
    return out


def _branch_hits(root: Path, issue: int) -> dict:
    ref = f"origin/issue-{issue}"
    p = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    if p.returncode != 0:
        return {"ref": ref, "present": False, "hits": []}
    cmd = ["git", "grep", "-l", "-F"]
    for prefix in PREFIXES:
        cmd += ["-e", prefix]
    cmd.append(ref)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
    if p.returncode not in (0, 1):
        raise SystemExit(f"FATAL: git grep on {ref} failed rc={p.returncode}: {p.stderr[:300]}")
    hits = [ln.split(":", 1)[1] for ln in p.stdout.splitlines() if ":" in ln]
    return {"ref": ref, "present": True, "hits": sorted(set(hits))}


def _branch_file_scan(root: Path, ref: str, path: str) -> tuple[set[str], set[str], set[str]]:
    p = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, cwd=str(root))
    if p.returncode != 0:
        log(f"  WARN: git show {ref}:{path} failed rc={p.returncode}")
        return set(), set(), set()
    blob = p.stdout
    prefixes_found = {pref for pref in PREFIXES if pref.encode() in blob}
    reads = {t for t in READ_TOKENS if t.encode() in blob}
    writes = {t for t in WRITE_TOKENS if t.encode() in blob}
    return prefixes_found, reads, writes


def _fleet_sweep(root: Path) -> dict:
    """Surface 4: live sessions + live pods + dashboard source grep.
    Subprocess calls fail loud; every leg records a positive line.
    NOTE: pod.py list-ephemeral hits the RunPod API — NOT the HF Hub."""
    fleet: dict = {}
    p = subprocess.run(
        [sys.executable, str(root / "scripts" / "spawn_session.py"), "list"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    if p.returncode != 0:
        raise SystemExit(f"FATAL: spawn_session.py list failed rc={p.returncode}: {p.stderr[:400]}")
    session_lines = p.stdout.splitlines()
    fleet["sessions"] = {}
    for prefix, issue in PREFIXES.items():
        tok = re.compile(rf"#{issue}\b")
        rows = [ln for ln in session_lines if tok.search(ln)]
        fleet["sessions"][prefix] = rows

    p = subprocess.run(
        [sys.executable, str(root / "scripts" / "pod_lifecycle.py"), "list-ephemeral"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    if p.returncode != 0:
        raise SystemExit(f"FATAL: pod list-ephemeral failed rc={p.returncode}: {p.stderr[:400]}")
    pod_lines = [] if "No ephemeral pod" in p.stdout else p.stdout.splitlines()
    fleet["pods"] = {}
    for prefix, issue in PREFIXES.items():
        rows = [ln for ln in pod_lines if ln.split()[1:2] == [f"#{issue}"]]
        fleet["pods"][prefix] = rows

    p = subprocess.run(
        ["systemctl", "show", "eps-dashboard.service", "--property=WorkingDirectory"],
        capture_output=True,
        text=True,
    )
    dash_dir = ""
    if p.returncode == 0 and "=" in p.stdout:
        dash_dir = p.stdout.strip().split("=", 1)[1]
    if dash_dir and Path(dash_dir).is_dir():
        hits = _run_grep(Path(dash_dir), DASH_EXCLUDE_DIRS, _grep_base_flags())
        fleet["dashboard"] = {"dir": dash_dir, "hits": hits}
    else:
        fleet["dashboard"] = {"dir": dash_dir or "<unit not present>", "hits": []}
    return fleet


def _boundary(n_worktrees: int) -> str:
    return (
        f"repo full tree (excludes: {','.join(ROOT_EXCLUDE_DIRS)}) + {n_worktrees} worktrees "
        f"full-tree (excludes: {','.join(WT_EXCLUDE_DIRS)}) + owning branch full tree "
        "(git grep origin/issue-<M>) + fleet sweep (sessions/pods/dashboard)"
    )


def run_audit(out_dir: Path, cache_path: Path | None = None) -> dict:
    root = primary_checkout_root()
    reg = json.loads(registry_path().read_text())["tasks"]
    worktrees = _worktree_paths(root)
    base_flags = _grep_base_flags()
    boundary = _boundary(len(worktrees))
    log(f"audited boundary: {boundary}")

    # Per-surface resume cache (checkpoint-per-phase): each completed grep
    # unit's candidate list persists the moment it completes, so a killed run
    # never re-pays a completed full-tree read.
    cache: dict = {}
    if cache_path is not None and cache_path.is_file():
        cache = json.loads(cache_path.read_text())
        log(f"resume cache loaded from {cache_path} (units: {sorted(cache)})")

    def _ckpt() -> None:
        if cache_path is not None:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, cache_path)

    # Per-prefix hit accumulator: prefix -> surface -> [ {path, classes, ...} ]
    hits: dict[str, dict[str, list[dict]]] = {
        p: {"repo_root": [], "worktrees": [], "branch": [], "dashboard": []} for p in PREFIXES
    }

    def _record(surface: str, display_path: str, rel_for_class: str, scan_target: Path) -> None:
        prefixes_found, reads, writes = _scan_file(scan_target)
        classes = _classify(rel_for_class, reads, writes)
        for prefix in prefixes_found:
            hits[prefix][surface].append(
                {
                    "path": display_path,
                    "classes": classes,
                    "read_tokens": sorted(reads),
                    "write_tokens": sorted(writes),
                }
            )

    t0 = time.time()
    if "repo_root" in cache:
        root_hits = cache["repo_root"]
        log(f"surface 1: {len(root_hits)} candidate files from resume cache")
    else:
        log("surface 1: repo root full tree (single combined fixed-string pass)")
        root_hits = _run_grep(root, ROOT_EXCLUDE_DIRS, base_flags)
        cache["repo_root"] = root_hits
        _ckpt()
    for rel in root_hits:
        rel_clean = rel.removeprefix("./")
        _record("repo_root", rel_clean, rel_clean, root / rel_clean)
    log(f"surface 1 done in {time.time() - t0:.0f}s")

    t0 = time.time()
    log(f"surface 2: {len(worktrees)} worktrees full-tree (bounded excludes)")
    wt_cache = cache.setdefault("worktrees", {})
    for wt in worktrees:
        if wt.name in wt_cache:
            wt_hits = wt_cache[wt.name]
        else:
            wt_hits = _run_grep(wt, WT_EXCLUDE_DIRS, base_flags)
            wt_cache[wt.name] = wt_hits
            _ckpt()
        for rel in wt_hits:
            rel_clean = rel.removeprefix("./")
            disp = f"{wt.name}/{rel_clean}"
            _record("worktrees", disp, rel_clean, wt / rel_clean)
        log(f"  [surface 2] {wt.name}: {len(wt_hits)} candidate files")
    log(f"surface 2 done in {time.time() - t0:.0f}s")

    t0 = time.time()
    log("surface 3: owning issue branches (git fetch origin first)")
    p = subprocess.run(["git", "fetch", "origin"], capture_output=True, text=True, cwd=str(root))
    if p.returncode != 0:
        raise SystemExit(f"FATAL: git fetch origin failed rc={p.returncode}: {p.stderr[:300]}")
    branch_meta: dict[str, dict] = {}
    br_cache = cache.setdefault("branches", {})
    for prefix, issue in PREFIXES.items():
        if str(issue) in br_cache:
            meta = br_cache[str(issue)]
        else:
            meta = _branch_hits(root, issue)
            br_cache[str(issue)] = meta
            _ckpt()
        branch_meta[prefix] = {"ref": meta["ref"], "present": meta["present"]}
        for path in meta["hits"]:
            prefixes_found, reads, writes = _branch_file_scan(root, meta["ref"], path)
            classes = _classify(path, reads, writes)
            for pref in prefixes_found:
                hits[pref]["branch"].append(
                    {
                        "path": f"{meta['ref']}:{path}",
                        "classes": classes,
                        "read_tokens": sorted(reads),
                        "write_tokens": sorted(writes),
                    }
                )
    log(f"surface 3 done in {time.time() - t0:.0f}s")

    log("surface 4: fleet/service sweep (sessions + pods + dashboard)")
    fleet = _fleet_sweep(root)
    # One dashboard pass covers all prefixes; _record fans hits out per prefix.
    for rel in fleet["dashboard"]["hits"]:
        rel_clean = rel.removeprefix("./")
        dash_root = Path(fleet["dashboard"]["dir"])
        _record("dashboard", f"dashboard/{rel_clean}", rel_clean, dash_root / rel_clean)

    # De-duplicate per surface (the combined pass can record a file once per
    # prefix; the accumulator is per-prefix already, so dedupe on path).
    result: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary": boundary,
        "n_worktrees": len(worktrees),
        "fleet": {
            "sessions": fleet["sessions"],
            "pods": fleet["pods"],
            "dashboard_dir": fleet["dashboard"]["dir"],
        },
        "prefixes": {},
    }
    for prefix, issue in PREFIXES.items():
        entry: dict = {"owner_issue": issue}
        owner = reg.get(str(issue), {})
        entry["owner_status"] = owner.get("status", "<missing>")
        surfaces: dict[str, list[dict]] = {}
        readers: list[dict] = []
        writers: list[dict] = []
        n_inert = 0
        n_self = 0
        for surface, rows in hits[prefix].items():
            dedup: dict[str, dict] = {}
            for row in rows:
                dedup[row["path"]] = row
            rows = [dedup[k] for k in sorted(dedup)]
            surfaces[surface] = rows
            for row in rows:
                if "reader" in row["classes"]:
                    readers.append({"surface": surface, **row})
                if "writer" in row["classes"]:
                    writers.append({"surface": surface, **row})
                if row["classes"] == ["inert"]:
                    n_inert += 1
                if row["classes"] == ["self-tooling-2332"]:
                    n_self += 1
        entry["surfaces"] = surfaces
        entry["branch"] = branch_meta[prefix]
        entry["fleet_sessions"] = fleet["sessions"][prefix]
        entry["fleet_pods"] = fleet["pods"][prefix]
        entry["reader_hits"] = readers
        entry["writer_hits"] = writers
        entry["n_inert"] = n_inert
        entry["n_self_tooling"] = n_self
        n_readers = len(readers)
        entry["verdict"] = "no-live-reader" if n_readers == 0 else "readers-found"
        entry["verdict_line"] = (
            f"{prefix}: {n_readers} reader-class hits — boundary: {boundary} — "
            f"owner #{issue} status={entry['owner_status']} "
            "(a parked owner may re-stage for a follow-up round; remedy: tar members keep "
            "original paths + index.json on the Hub, so any post-repack reader is recoverable "
            "with a ~3-line change or the accessor)"
        )
        result["prefixes"][prefix] = entry
        log(entry["verdict_line"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "consumer_audit.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    _write_md(result, out_dir / "consumer_audit.md")
    log(f"wrote {out_dir / 'consumer_audit.json'} + consumer_audit.md")
    return result


def _write_md(result: dict, path: Path) -> None:
    lines = [
        "# #2332 consumer audit (P1)",
        "",
        f"Generated: {result['generated_at']}  ",
        f"Audited boundary: {result['boundary']}",
        "",
        "Verdict per prefix: `no-live-reader` (0 reader-class hits within the stated "
        "boundary) or `readers-found` (reader-class hits listed below; the "
        "trivially-updatable vs needs-accessor triage is the orchestrator's branch "
        "decision per the #2332 brief — the Option-2 accessor is NOT built this round).",
        "",
        "| prefix | owner | owner status | verdict | reader hits | writer hits | inert | self-tooling |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for prefix, e in result["prefixes"].items():
        lines.append(
            f"| `{prefix}` | #{e['owner_issue']} | {e['owner_status']} | **{e['verdict']}** | "
            f"{len(e['reader_hits'])} | {len(e['writer_hits'])} | {e['n_inert']} | "
            f"{e['n_self_tooling']} |"
        )
    lines.append("")
    lines.append("## Per-prefix verdict lines (boundary restated verbatim)")
    lines.append("")
    for _prefix, e in result["prefixes"].items():
        lines.append(f"- {e['verdict_line']}")
    lines.append("")
    lines.append("## Reader-class hits (full list)")
    lines.append("")
    any_readers = False
    for prefix, e in result["prefixes"].items():
        if not e["reader_hits"]:
            continue
        any_readers = True
        lines.append(f"### `{prefix}`")
        lines.append("")
        for row in e["reader_hits"]:
            lines.append(
                f"- [{row['surface']}] `{row['path']}` (read tokens: "
                f"{', '.join(row['read_tokens'])})"
            )
        lines.append("")
    if not any_readers:
        lines.append("None — zero reader-class hits across all 8 prefixes within the boundary.")
        lines.append("")
    lines.append("## Fleet sweep (positive lines)")
    lines.append("")
    for prefix, e in result["prefixes"].items():
        sess = e["fleet_sessions"]
        pods = e["fleet_pods"]
        lines.append(
            f"- `{prefix}` (#{e['owner_issue']}): sessions="
            f"{'found: ' + '; '.join(sess) if sess else 'none-found'}; "
            f"pods={'found: ' + '; '.join(pods) if pods else 'none-found'}; "
            f"branch={'present' if e['branch']['present'] else 'ABSENT'} ({e['branch']['ref']})"
        )
    lines.append("")
    lines.append(f"Dashboard source dir: {result['fleet']['dashboard_dir']}")
    lines.append("")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default="eval_results/issue_2332",
        help="output dir for consumer_audit.{json,md} (default: eval_results/issue_2332)",
    )
    ap.add_argument(
        "--cache",
        default="/tmp/i2332_audit_cache.json",
        help="per-surface resume cache (candidate lists checkpoint per completed grep unit)",
    )
    args = ap.parse_args(argv)
    run_audit(Path(args.out), cache_path=Path(args.cache) if args.cache else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
