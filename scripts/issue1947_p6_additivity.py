#!/usr/bin/env python
"""Audit that the FLEET-WIDE P6 re-run was ADDITIVE for the 38 imp/cas+marker
arms: every pre-existing battery cell must be unchanged except for float-level
nondeterminism, and every field that differs must be reported with its
magnitude. Mirrors the predecessor leg's cmd_select additivity audit.

Compares the live eval_results/issue_1947/analysis/battery/*.json against the
pre-run backup, ignoring the per-file `ts` / `git_commit` provenance stamps.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

LIVE = Path("eval_results/issue_1947/analysis/battery")
BACKUP = Path("/mnt/eps-data/thomasjiralerspong/issue1947_p6_fleet/_precommit_backup/battery")
IGNORE = {"ts", "issue", "git_commit"}
# The P6 writer, for the concurrency guard below. Bracket one character so the
# probe's own argv can never self-match (the ownership-probe idiom).
P6_WRITER_PATTERN = "issue1947_analysi[s].py"


def refuse_if_p6_running() -> None:
    """Refuse to audit while the P6 writer is live — a mid-write read is torn.

    `issue1947_analysis.py` writes each battery cell with an atomic replace, but
    the audit walks the WHOLE set, so a run in flight yields a mixture of
    already-rewritten and not-yet-rewritten cells. Observed live: an audit run
    mid-P6 reported 2 spurious STRUCTURAL diffs that vanished once the writer
    finished. Silence there would be a wrong PASS/FAIL, so fail loud instead.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-af", P6_WRITER_PATTERN], capture_output=True, text=True, check=False
        )
    except OSError as e:  # pgrep absent — report, do not silently skip the guard
        print(f"[audit] WARNING: liveness guard could not run ({e}); proceeding")
        return
    hits = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if hits:
        print("[audit] REFUSING: the P6 writer is still running — a mid-write audit is torn:")
        for h in hits[:4]:
            print("   ", h[:160])
        print("[audit] wait for it to exit, then re-run (--allow-concurrent overrides).")
        raise SystemExit(2)


def walk(a, b, path: str, out: list[tuple[str, float, object, object]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out.append((f"{path}.{k}", math.inf, b.get(k, "<absent>"), a.get(k, "<absent>")))
            else:
                walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}[len]", math.inf, len(b), len(a)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != b:
            out.append((path, abs(float(a) - float(b)), b, a))
    elif a != b:
        out.append((path, math.inf, b, a))


def main() -> int:
    if "--allow-concurrent" not in sys.argv:
        refuse_if_p6_running()
    old_files = sorted(BACKUP.glob("*.json"))
    assert old_files, f"no backup cells under {BACKUP}"
    n_same = 0
    diffs: list[tuple[str, str, float]] = []
    missing: list[str] = []
    for ob in old_files:
        lv = LIVE / ob.name
        if not lv.exists():
            missing.append(ob.name)
            continue
        out: list[tuple[str, float, object, object]] = []
        walk(json.loads(lv.read_text()), json.loads(ob.read_text()), "", out)
        if not out:
            n_same += 1
        else:
            for p, mag, _o, _n in out:
                diffs.append((ob.name, p, mag))

    print(f"pre-existing cells checked : {len(old_files)}")
    print(f"byte-equivalent (ignoring provenance stamps): {n_same}")
    print(f"cells with any numeric drift               : {len(old_files) - n_same - len(missing)}")
    if missing:
        print(f"MISSING FROM LIVE (regression!)            : {len(missing)} -> {missing[:5]}")

    if diffs:
        by_field: dict[str, list[float]] = {}
        for _f, p, mag in diffs:
            by_field.setdefault(p, []).append(mag)
        print("\ndrift by field (field: n_cells, max |delta|):")
        for p, mags in sorted(by_field.items(), key=lambda kv: -max(kv[1])):
            mx = max(mags)
            print(f"  {p}: n={len(mags)}, max|d|={'STRUCTURAL' if mx == math.inf else f'{mx:.3e}'}")
        worst = max(m for _f, _p, m in diffs)
        struct = [d for d in diffs if d[2] == math.inf]
        print()
        if struct:
            print(f"VERDICT: FAIL — {len(struct)} STRUCTURAL diff(s) (added/removed/typed fields)")
            return 1
        if worst < 1e-4:
            print(f"VERDICT: PASS (additive) — all drift numeric, max |delta| {worst:.3e} < 1e-4")
            print("  interpretation: float-level nondeterminism, no substantive change")
            return 0
        print(f"VERDICT: FAIL — numeric drift too large: max |delta| {worst:.3e} >= 1e-4")
        return 1
    print("\nVERDICT: PASS (byte-equivalent)")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
