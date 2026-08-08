"""K1 corpus dry-run for #2169, round 2 (throwaway; NOT committed to the repo tree).

Runs the NARROWED check 31 (worktree copy at commit 2c47f31aef — §3.0
plan-named candidate filter) over every task body under the main checkout's
tasks/*/*/body.md, threading each body's issue number from its directory
name (never the issue=None fallback), against the main repo's object DB —
i.e. at the bodies' OWN cited SHAs, not a HEAD working-tree approximation.
Also runs the OLD check (origin/main copy) for the A/B no-regress parity
read. Reports the class-C gain distribution over TRIGGERED bodies
(>=1 reachable cited SHA and >=1 committed PNG in the scoped dir), the
per-body class-C STEM LISTS (the §7 K1 v5 requirement — round 1 emitted
counts only), and the per-body §3.0 plan-resolution mode (the inert
fraction).
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2169")
MAIN = Path("/home/thomasjiralerspong/explore-persona-space")
OLD_COPY = Path("/tmp/i2169_vtb_old.py")  # pre-extracted origin/main copy

_ENTRY_RE = re.compile(r"`(figures/issue_\d+/[^`]+)` \(([^)]*)\)")
_C_TOKEN = "committed-figure-unmentioned"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod._resolve_repo_root = lambda: MAIN
    return mod


new = _load("vtb_new", WT / "scripts" / "verify_task_body.py")
old = _load("vtb_old", OLD_COPY)

registry = json.loads((MAIN / "tasks" / "REGISTRY.json").read_text())

rows = []
for body_path in sorted(MAIN.glob("tasks/*/*/body.md")):
    issue_dir = body_path.parent.name
    if not issue_dir.isdigit():
        continue
    issue = int(issue_dir)
    raw = body_path.read_text(errors="replace")
    _fm, body = new.split_frontmatter(raw)
    reg = registry.get("tasks", {}).get(str(issue), {}) if isinstance(registry, dict) else {}
    if not reg and isinstance(registry, dict):
        reg = registry.get(str(issue), {}) or {}
    has_cr = bool(reg.get("has_clean_result")) if isinstance(reg, dict) else False
    # Trigger instrumentation: reachable cited SHAs + committed PNGs, scoped.
    cited = new._cited_issue_figure_dirs(body)
    scoped = {k: v for k, v in cited.items() if k == f"figures/issue_{issue}/"}
    n_reachable = 0
    pngs = set()
    for prefix, shas in scoped.items():
        for sha in shas:
            tracked = new._git_tracked_under(MAIN, sha, prefix)
            if tracked is None:
                continue
            n_reachable += 1
            pngs |= {p for p in tracked if p.lower().endswith(".png")}
    triggered = n_reachable >= 1 and len(pngs) >= 1
    plan_text, plan_mode = new._plan_naming_text(issue)
    r_new = new.check_orphaned_per_unit_figures(body, issue=issue)
    r_old = old.check_orphaned_per_unit_figures(body, issue=issue)
    c_stems = sorted(
        {
            p.rsplit("/", 1)[-1].removesuffix(".png")
            for p, cls in _ENTRY_RE.findall(r_new.detail)
            if _C_TOKEN in cls
        }
    )
    rows.append(
        {
            "issue": issue,
            "has_clean_result": has_cr,
            "triggered": triggered,
            "n_pngs": len(pngs),
            "class_c": len(c_stems),
            "class_c_stems": c_stems,
            "plan_mode": plan_mode,
            "plan_resolved": plan_text is not None,
            "a_old": r_old.detail.count("never mentioned in the body"),
            "a_new": r_new.detail.count("never mentioned in the body"),
            "b_old": r_old.detail.count("companion-named-not-embedded"),
            "b_new": r_new.detail.count("companion-named-not-embedded"),
            "warn_old": bool(r_old.is_warn),
            "warn_new": bool(r_new.is_warn),
        }
    )

trig = [r for r in rows if r["triggered"]]
gain = [r for r in trig if r["class_c"] > 0]
ab_new_warn = [r for r in rows if (r["a_new"] > r["a_old"]) or (r["b_new"] > r["b_old"])]
ab_loosened = [r for r in rows if (r["a_new"] < r["a_old"]) or (r["b_new"] < r["b_old"])]
inert = [r for r in trig if not r["plan_resolved"]]

counts = sorted(r["class_c"] for r in trig)


def _pctl(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


print(f"bodies scanned (numeric task dirs): {len(rows)}")
print(f"  with has_clean_result=true:       {sum(1 for r in rows if r['has_clean_result'])}")
print(f"TRIGGERED (>=1 reachable cited SHA & >=1 committed PNG): {len(trig)}")
if trig:
    print(
        f"triggered bodies gaining >=1 class-C entry: {len(gain)}"
        f" ({100.0 * len(gain) / len(trig):.1f}% of triggered)"
    )
print(f"max class-C entries on any single body: {max(counts, default=0)}")
print(f"p90 class-C over triggered: {_pctl(counts, 0.90)}")
print(f"median class-C over triggered: {_pctl(counts, 0.50)}")
print(f"triggered bodies with class_c > 10: {sum(1 for c in counts if c > 10)}")
print(
    f"class-C inert (plan unresolved) among triggered: {len(inert)}"
    f" ({100.0 * len(inert) / len(trig):.1f}% of triggered)"
    if trig
    else ""
)
print(f"bodies where class A/B GAINED entries (must be 0): {len(ab_new_warn)}")
print(
    "bodies where class A/B LOST entries (the disclosed #2169 loosening): "
    f"{[(r['issue'], r['a_old'] - r['a_new'], r['b_old'] - r['b_new']) for r in ab_loosened]}"
)
print("\nclass-C gainers with per-body STEM LISTS (issue, has_cr, n_pngs, C, stems):")
for r in sorted(gain, key=lambda r: -r["class_c"]):
    print(
        f"  #{r['issue']:>5}  cr={r['has_clean_result']}  pngs={r['n_pngs']:>3}  "
        f"C={r['class_c']}  stems={r['class_c_stems']}"
    )
dist = {}
for r in trig:
    dist[r["class_c"]] = dist.get(r["class_c"], 0) + 1
print("\nclass-C count distribution over triggered bodies:", dict(sorted(dist.items())))
print("\nplan-mode tally over triggered bodies:")
modes = {}
for r in trig:
    key = r["plan_mode"] if not r["plan_resolved"] else "resolved"
    modes[key] = modes.get(key, 0) + 1
for k, v in sorted(modes.items(), key=lambda t: -t[1]):
    print(f"  {v:>4}  {k}")
Path("/tmp/i2169_k1_rows_v2.json").write_text(json.dumps(rows, indent=1))
print("\nrows persisted to /tmp/i2169_k1_rows_v2.json")
