#!/usr/bin/env python
"""Audit harness for the #2641 CommonMark fence-mask fix (`verify_plan._fence_mask`).

Measures, over a plan corpus, (stage 1) which plan files' per-line fence
masks CHANGE between the frozen pre-#2641 delimiter-blind walk and the live
CommonMark-correct walk, and (stage 2) which of those files move a per-check
verdict under `verify_plan.verify_plan_text` — under BOTH `kind` resolution
conventions (forced ``kind="experiment"``, the task #2641 body's reproduce
convention, and per-task ``kind`` from ``body.md``, production behavior).

The frozen baseline `_blind_fence_mask` is copied VERBATIM from the
measurement-anchor commit and must never be "fixed": it is what keeps this
harness re-runnable (and its numbers comparable) after the fix lands — the
c38/c39 calibration convention in `scripts/verify_plan.py`.

Corpus selection (mutually exclusive):
  --corpus GLOB          glob over the current working tree
                         (default ``tasks/*/*/plans/v*.md``)
  --corpus-list FILE     newline-delimited path list
  --corpus-git-rev SHA   materialize ``tasks/*/*/plans/v<K>.md`` (+ each task's
                         ``body.md`` for kind resolution) from that git rev into
                         a temp snapshot and audit THAT — the fully pinned mode:
                         both plan content and ``kind`` come from the rev, so a
                         later status ``git mv`` or corpus growth cannot drift
                         the numbers (plan #2641 v3 section 6 step S1).

Determinism: rows are sorted by ``(file, check_id)`` before emit, so
``--jobs 4`` output is byte-identical to ``--jobs 1`` (modulo the
``generated`` timestamp, which ``--no-timestamp`` suppresses). With
``--jobs N > 1`` each worker path-loads its own ``verify_plan`` copy, so the
blind-mask monkeypatch is process-local.

Fail-loud: an unreadable or non-UTF-8 corpus file raises; a missing git
object in ``--corpus-git-rev`` mode raises. There is no ``except: continue``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_PLAN_PATH = REPO_ROOT / "scripts" / "verify_plan.py"

# The task #2641 measurement-anchor commit (documentation; --corpus-git-rev
# takes any rev).
ANCHOR_SHA = "5cb785f090e7866e1d227654e844e7489d7bb334"

# Glob-equivalent of the corpus-of-record pattern ``tasks/*/*/plans/v*.md``
# (the task #2641 body's + the plan-time measurement's corpus definition).
# Deliberately ``v[^/]*`` and NOT ``v\d+``: the corpus contains exactly one
# glob-matching, non-``v<K>`` plan file (``tasks/completed/356/plans/
# v2-factcheck.md``), and it IS one of the 153 mask-changing / 37+36
# verdict-moving files — a ``v\d+`` pin silently reads 152/36/35 (measured
# 2026-08-30 against the anchor commit) and falsely fails the S1=153
# reproduction.
_PLAN_RE = re.compile(r"^tasks/[^/]+/[^/]+/plans/v[^/]*\.md$")

KIND_MODES = ("forced-experiment", "task")

# The five mask-defect classes from plan #2641 v3 section 4.2 `explain`,
# plus `closer-info-string` (a same-char, long-enough closing CANDIDATE that
# carries an info string — CommonMark keeps the block open; the blind walk
# toggles). A file may carry several classes.
DEFECT_CLASSES = (
    "mismatched-delimiter",
    "inner-shorter-fence",
    "indented-marker",
    "info-string-backtick",
    "closer-info-string",
    "unclosed-at-eof",
)


def _blind_fence_mask(lines: list[str]) -> list[bool]:
    """FROZEN pre-#2641 delimiter-blind mask — the historical baseline arm.

    Body copied VERBATIM from
    ``5cb785f090e7866e1d227654e844e7489d7bb334:scripts/verify_plan.py``
    lines 723-732 (the ``_fence_mask`` body at the anchor commit; only the
    name and this docstring differ). It toggles on ANY line whose stripped
    form starts with ``\\`\\`\\``` or ``~~~`` — tracking neither delimiter
    character, nor length, nor indentation, nor the info string. Do NOT
    "fix" this function: it exists so the pre-fix behavior stays exactly
    reproducible after the live mask changes.
    """
    mask: list[bool] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            mask.append(True)
            continue
        mask.append(in_fence)
    return mask


def _blind_unclosed(lines: list[str]) -> bool:
    """True when the blind walk ends inside a fence (odd toggle count)."""
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
    return in_fence


_AGG_INDENT_RE = re.compile(r"^(?: {4}|\t)")


def _aggressive_fence_mask(vp, lines: list[str]) -> list[bool]:
    """The REJECTED plan #2641 v3 section 11 R2 alternative, for the S4
    counterfactual only: the CommonMark walk PLUS every 4-space / tab
    indented line masked as code (mirroring the drop-indented-lines step of
    the #2384 branch's ``_c75_strip_code_blocks``). Never used by
    ``verify_plan`` itself."""
    base = vp._fence_mask(lines)
    return [m or bool(_AGG_INDENT_RE.match(line)) for m, line in zip(base, lines, strict=True)]


def load_verify_plan():
    """Path-load ``scripts/verify_plan.py`` the way ``tests/test_verify_plan.py``
    does, under a harness-private module name (no import-path assumptions,
    no collision with a pytest-loaded ``verify_plan``)."""
    spec = importlib.util.spec_from_file_location("_i2641_verify_plan", VERIFY_PLAN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_i2641_verify_plan"] = mod
    spec.loader.exec_module(mod)
    return mod


# Per-process module handle (ProcessPoolExecutor initializer target).
_VP = None


def _init_worker() -> None:
    global _VP
    _VP = load_verify_plan()


def _read(path: Path) -> str:
    # Strict UTF-8, no errors= fallback: an undecodable plan is a loud
    # failure, never a silent mis-measurement (plan v3 A18).
    return path.read_text(encoding="utf-8")


def _mask_diff_one(path_str: str, variant: str) -> dict:
    """Stage 1 for one file: does the mask change, and by how many lines?"""
    path = Path(path_str)
    lines = _read(path).splitlines()
    blind = _blind_fence_mask(lines)
    if variant == "aggressive":
        new = _aggressive_fence_mask(_VP, lines)
        new_unclosed = _VP.unclosed_fence_line(lines)
    else:
        new = _VP._fence_mask(lines)
        new_unclosed = _VP.unclosed_fence_line(lines)
    n_diff = sum(1 for a, b in zip(blind, new, strict=True) if a != b)
    return {
        "file": path_str,
        "n_lines": len(lines),
        "n_diff_lines": n_diff,
        "changed": n_diff > 0,
        "blind_unclosed": _blind_unclosed(lines),
        "new_unclosed_open_idx": new_unclosed,
    }


def _verdicts(text: str, kind: str) -> tuple[bool, dict[str, str]]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        overall, results = _VP.verify_plan_text(text, kind=kind)
    return overall, {r.id: r.status for r in results}


def _resolve_kind(path: Path) -> str:
    """Production kind resolution: ``body.md`` two levels up (the task
    folder), falling back to ``experiment`` — mirrors ``_kind_from_body``."""
    return _VP._kind_from_body(path.parent.parent)


def _verdict_diff_one(path_str: str, kind_modes: tuple[str, ...]) -> dict:
    """Stage 2 for one mask-changing file: per-check status transitions
    under the blind vs live mask, per kind mode."""
    path = Path(path_str)
    text = _read(path)
    unclosed = _VP.unclosed_fence_line(text.splitlines()) is not None

    task_kind = _resolve_kind(path) if "task" in kind_modes else None
    out: dict = {"file": path_str, "unclosed_at_eof": unclosed, "modes": {}}

    def run_both(kind: str) -> dict:
        orig = _VP._fence_mask
        try:
            _VP._fence_mask = _blind_fence_mask
            overall_old, old = _verdicts(text, kind)
        finally:
            _VP._fence_mask = orig
        overall_new, new = _verdicts(text, kind)
        rows = [
            {
                "check_id": cid,
                "old": old.get(cid, "ABSENT"),
                "new": new.get(cid, "ABSENT"),
            }
            for cid in sorted(set(old) | set(new))
            if old.get(cid) != new.get(cid)
        ]
        return {
            "kind": kind,
            "overall_old": overall_old,
            "overall_new": overall_new,
            "overall_flip": overall_old != overall_new,
            "rows": rows,
        }

    cache: dict[str, dict] = {}
    for mode in kind_modes:
        kind = "experiment" if mode == "forced-experiment" else (task_kind or "experiment")
        if kind not in cache:
            cache[kind] = run_both(kind)
        out["modes"][mode] = cache[kind]
    return out


def _classify_defects(lines: list[str]) -> list[str]:
    """Classify which mask-defect classes a document carries, by re-walking
    it with the live rule while watching what the blind rule would do."""
    classes: set[str] = set()
    fence: str | None = None
    for line in lines:
        m = _VP._FENCE_RE.match(line)
        stripped = line.strip()
        blind_toggle = stripped.startswith("```") or stripped.startswith("~~~")
        if fence is not None:
            if m is not None:
                same = m.group("delim")[0] == fence[0]
                long_enough = len(m.group("delim")) >= len(fence)
                no_info = not m.group("info").strip()
                if same and long_enough and no_info:
                    fence = None
                elif not same:
                    classes.add("mismatched-delimiter")
                elif not long_enough:
                    classes.add("inner-shorter-fence")
                else:
                    classes.add("closer-info-string")
            elif blind_toggle:
                # Fence-shaped to the blind walk, unrecognizable to the
                # live rule while inside a block: indent >= 4 or tab.
                classes.add("indented-marker")
            continue
        if m is not None and not (m.group("delim")[0] == "`" and "`" in m.group("info")):
            fence = m.group("delim")
        elif m is not None:
            classes.add("info-string-backtick")
        elif blind_toggle:
            classes.add("indented-marker")
    if fence is not None:
        classes.add("unclosed-at-eof")
    return sorted(classes)


def explain(path: Path, check_id: str | None, kind_mode: str) -> None:
    """The adjudication reading tool (plan v3 section 4.2)."""
    text = _read(path)
    lines = text.splitlines()
    blind = _blind_fence_mask(lines)
    new = _VP._fence_mask(lines)

    print(f"=== {path} ===")
    print("--- fence-shaped lines (1-based; indent; raw repr) ---")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~") or _VP._FENCE_RE.match(line):
            indent = len(line) - len(line.lstrip(" \t"))
            print(f"  line {i + 1:>5} indent={indent:<2} {line!r}")

    print("--- mask-differing runs ([blind][new] flag column; 4 lines context) ---")
    diff_idx = [i for i in range(len(lines)) if blind[i] != new[i]]
    runs: list[tuple[int, int]] = []
    for i in diff_idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    for start, end in runs:
        print(f"  run: lines {start + 1}-{end + 1} ({end - start + 1} lines)")
        for j in range(max(0, start - 4), min(len(lines), end + 5)):
            flag = f"[{'T' if blind[j] else 'F'}][{'T' if new[j] else 'F'}]"
            mark = "*" if start <= j <= end else " "
            print(f"   {mark}{flag} {j + 1:>5}: {lines[j][:120]!r}")

    print("--- defect classes ---")
    for cls in _classify_defects(lines):
        print(f"  {cls}")
    unclosed = _VP.unclosed_fence_line(lines)
    if unclosed is not None:
        print(f"  (unclosed fence opened at line {unclosed + 1})")

    if check_id:
        kind = "experiment" if kind_mode == "forced-experiment" else _resolve_kind(path)
        print(f"--- check {check_id} detail (kind={kind}) ---")
        orig = _VP._fence_mask
        try:
            _VP._fence_mask = _blind_fence_mask
            with contextlib.redirect_stdout(io.StringIO()):
                _, old_results = _VP.verify_plan_text(text, kind=kind)
        finally:
            _VP._fence_mask = orig
        with contextlib.redirect_stdout(io.StringIO()):
            _, new_results = _VP.verify_plan_text(text, kind=kind)
        old_map = {r.id: r for r in old_results}
        new_map = {r.id: r for r in new_results}
        for label, rmap in (("old (blind mask)", old_map), ("new (CommonMark)", new_map)):
            r = rmap.get(check_id)
            if r is None:
                print(f"  {label}: <check id not found>")
            else:
                print(f"  {label}: {r.status}  {r.detail}")


def materialize_git_corpus(rev: str, dest: Path) -> list[Path]:
    """Extract every ``tasks/*/*/plans/v<K>.md`` (+ each such task's
    ``body.md``) at ``rev`` into ``dest``, preserving relative paths.

    Uses one ``git cat-file --batch`` stream (no worktree, no full-archive
    pipe). A missing object raises — fail loud."""
    ls = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", rev],
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
    )
    names = [n for n in ls.stdout.decode("utf-8").split("\0") if n]
    plans = sorted(n for n in names if _PLAN_RE.match(n))
    name_set = set(names)
    bodies = sorted({str(Path(p).parent.parent / "body.md") for p in plans} & name_set)
    wanted = plans + bodies
    req = "".join(f"{rev}:{p}\n" for p in wanted).encode()
    proc = subprocess.run(
        ["git", "cat-file", "--batch"], input=req, capture_output=True, check=True, cwd=REPO_ROOT
    )
    buf = proc.stdout
    pos = 0
    for p in wanted:
        nl = buf.index(b"\n", pos)
        header = buf[pos:nl].decode("utf-8")
        if header.endswith(" missing"):
            raise FileNotFoundError(f"{rev}:{p} missing from git object store")
        _oid, _typ, size_s = header.rsplit(" ", 2)
        size = int(size_s)
        start = nl + 1
        content = buf[start : start + size]
        pos = start + size + 1  # trailing newline after content
        out = dest / p
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
    return [dest / p for p in plans]


def _run_stage1(paths: list[Path], jobs: int, variant: str) -> list[dict]:
    path_strs = [str(p) for p in paths]
    if jobs <= 1:
        if _VP is None:
            _init_worker()
        results = [_mask_diff_one(p, variant) for p in path_strs]
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
            results = list(ex.map(_mask_diff_one, path_strs, [variant] * len(path_strs)))
    return sorted(results, key=lambda r: r["file"])


def _run_stage2(paths: list[str], jobs: int, kind_modes: tuple[str, ...]) -> list[dict]:
    if jobs <= 1:
        if _VP is None:
            _init_worker()
        results = [_verdict_diff_one(p, kind_modes) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
            results = list(ex.map(_verdict_diff_one, paths, [kind_modes] * len(paths)))
    return sorted(results, key=lambda r: r["file"])


def _repo_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=True, cwd=REPO_ROOT, text=True
    ).stdout.strip()


def _summarize_mode(stage2: list[dict], mode: str) -> dict:
    """Flatten per-file stage-2 results for one kind mode into the AC2 row
    schema + per-check histogram + overall-flip list."""
    rows = []
    per_check: dict[str, dict[str, int]] = {}
    mover_files = []
    flips = []
    for entry in stage2:
        m = entry["modes"][mode]
        if m["rows"]:
            mover_files.append(entry["file"])
        if m["overall_flip"]:
            flips.append(
                {
                    "file": entry["file"],
                    "old": "PASS" if m["overall_old"] else "FAIL",
                    "new": "PASS" if m["overall_new"] else "FAIL",
                }
            )
        for r in m["rows"]:
            rows.append(
                {
                    "file": entry["file"],
                    "check_id": r["check_id"],
                    "old_status": r["old"],
                    "new_status": r["new"],
                    "unclosed_at_eof": entry["unclosed_at_eof"],
                }
            )
            trans = f"{r['old']}->{r['new']}"
            per_check.setdefault(r["check_id"], {}).setdefault(trans, 0)
            per_check[r["check_id"]][trans] += 1
    rows.sort(key=lambda r: (r["file"], r["check_id"]))
    return {
        "n_verdict_moving_files": len(mover_files),
        "verdict_moving_files": mover_files,
        "n_affected_checks": len(per_check),
        "affected_checks": sorted(per_check),
        "per_check_transitions": {k: dict(sorted(v.items())) for k, v in sorted(per_check.items())},
        "overall_flips": flips,
        "rows": rows,
    }


def _unclosed_census(stage1: list[dict]) -> dict:
    new_unclosed = [r for r in stage1 if r["new_unclosed_open_idx"] is not None]
    blind_unclosed = [r for r in stage1 if r["blind_unclosed"]]
    newly = [r for r in new_unclosed if not r["blind_unclosed"]]
    tail_excl = sum(r["n_lines"] - r["new_unclosed_open_idx"] - 1 for r in newly)
    return {
        "n_unclosed_new_mask": len(new_unclosed),
        "n_unclosed_blind_mask": len(blind_unclosed),
        "n_newly_unclosed": len(newly),
        "newly_unclosed_files": sorted(r["file"] for r in newly),
        "tail_lines_swallowed_exclusive_of_opener": tail_excl,
        "tail_lines_swallowed_inclusive_of_opener": tail_excl + len(newly),
    }


def _markdown_report(payload: dict) -> str:
    out = ["# issue2641 fence-mask audit", ""]
    c = payload["corpus"]
    out.append(
        f"- corpus: {c['n_files']} files ({c['mode']}; pin={c.get('pin') or 'none'}); "
        f"repo SHA {payload['repo_sha']}"
    )
    s1 = payload["stage1"]
    out.append(
        f"- stage 1 ({payload['variant']}): {s1['n_mask_changed']} mask-changing files, "
        f"{s1['n_diff_lines_total']} differing lines"
    )
    uc = payload["unclosed_census"]
    out.append(
        f"- unclosed census: {uc['n_unclosed_new_mask']} unclosed under the new mask "
        f"({uc['n_newly_unclosed']} newly so; blind: {uc['n_unclosed_blind_mask']}); "
        f"{uc['tail_lines_swallowed_exclusive_of_opener']} tail lines swallowed "
        f"(exclusive of openers; {uc['tail_lines_swallowed_inclusive_of_opener']} inclusive)"
    )
    out.append("")
    for mode, summ in payload.get("stage2", {}).items():
        out.append(f"## kind mode: {mode}")
        out.append(
            f"- {summ['n_verdict_moving_files']} verdict-moving files across "
            f"{summ['n_affected_checks']} checks; {len(summ['overall_flips'])} overall "
            "PASS/FAIL flips"
        )
        out.append("")
        out.append("| check | files | transitions |")
        out.append("|---|---|---|")
        by_check: dict[str, set[str]] = {}
        for r in summ["rows"]:
            by_check.setdefault(r["check_id"], set()).add(r["file"])
        for cid in sorted(summ["per_check_transitions"], key=lambda k: (-len(by_check[k]), k)):
            trans = ", ".join(f"{t} x{n}" for t, n in summ["per_check_transitions"][cid].items())
            out.append(f"| `{cid}` | {len(by_check[cid])} | {trans} |")
        out.append("")
        if summ["overall_flips"]:
            out.append("Overall PASS/FAIL flips:")
            for f in summ["overall_flips"]:
                out.append(f"- `{f['file']}`: {f['old']} -> {f['new']}")
            out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--corpus", default="tasks/*/*/plans/v*.md", help="glob (working tree)")
    src.add_argument("--corpus-list", help="newline-delimited path list file")
    src.add_argument("--corpus-git-rev", help="materialize the corpus from this git rev (pinned)")
    ap.add_argument("--kind-mode", choices=[*KIND_MODES, "both"], default="both")
    ap.add_argument("--variant", choices=["commonmark", "aggressive"], default="commonmark")
    ap.add_argument("--json", dest="json_out", help="write the machine-readable payload here")
    ap.add_argument("--markdown", dest="md_out", help="write the human-readable summary here")
    ap.add_argument("--explain", help="print the adjudication read for ONE file, then exit")
    ap.add_argument("--check", help="with --explain: show this check id's old/new detail")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--stage1-only", action="store_true", help="skip the verdict diff")
    ap.add_argument(
        "--no-timestamp",
        action="store_true",
        help="omit the generated timestamp (byte-identical determinism checks)",
    )
    args = ap.parse_args(argv)

    _init_worker()

    if args.explain:
        mode = args.kind_mode if args.kind_mode != "both" else "forced-experiment"
        explain(Path(args.explain), args.check, mode)
        return 0

    tmpdir: tempfile.TemporaryDirectory | None = None
    if args.corpus_git_rev:
        tmpdir = tempfile.TemporaryDirectory(prefix="i2641_corpus_")
        paths = materialize_git_corpus(args.corpus_git_rev, Path(tmpdir.name))
        corpus_meta = {"mode": "git-rev", "pin": args.corpus_git_rev, "n_files": len(paths)}
    elif args.corpus_list:
        paths = [Path(line) for line in Path(args.corpus_list).read_text().splitlines() if line]
        corpus_meta = {"mode": "list", "pin": None, "n_files": len(paths)}
    else:
        paths = sorted(REPO_ROOT.glob(args.corpus))
        corpus_meta = {"mode": "glob", "pin": None, "n_files": len(paths)}
    if not paths:
        raise SystemExit("empty corpus — refusing to audit nothing")

    stage1 = _run_stage1(paths, args.jobs, args.variant)
    changed = [r["file"] for r in stage1 if r["changed"]]
    payload: dict = {
        "repo_sha": _repo_sha(),
        "variant": args.variant,
        "corpus": corpus_meta,
        "stage1": {
            "n_mask_changed": len(changed),
            "n_diff_lines_total": sum(r["n_diff_lines"] for r in stage1),
            "mask_changed_files": changed,
        },
        "unclosed_census": _unclosed_census(stage1),
    }
    if not args.no_timestamp:
        payload["generated"] = datetime.now(tz=UTC).isoformat()

    if not args.stage1_only and args.variant == "commonmark":
        modes = KIND_MODES if args.kind_mode == "both" else (args.kind_mode,)
        stage2 = _run_stage2(changed, args.jobs, tuple(modes))
        payload["stage2"] = {mode: _summarize_mode(stage2, mode) for mode in modes}

    doc = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(doc)
    if args.md_out:
        Path(args.md_out).write_text(_markdown_report(payload))
    if not args.json_out and not args.md_out:
        print(doc, end="")
    else:
        print(f"stage1: {len(changed)} mask-changing files / {corpus_meta['n_files']} corpus files")
        for mode, summ in payload.get("stage2", {}).items():
            print(
                f"stage2[{mode}]: {summ['n_verdict_moving_files']} files, "
                f"{summ['n_affected_checks']} checks, "
                f"{len(summ['overall_flips'])} overall flips"
            )
    if tmpdir is not None:
        tmpdir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
