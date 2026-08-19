"""K1 corpus dry-run for #2169 (throwaway; NOT committed).

Runs the WIDENED check 31 (worktree copy) over every task body under the
main checkout's tasks/*/*/body.md, threading each body's issue number from
its directory name (never the issue=None fallback), against the main repo's
object DB. Also runs the OLD check (origin/main copy) for an A/B-parity
read. Reports the class-C gain distribution over TRIGGERED bodies
(>=1 reachable cited SHA and >=1 committed PNG in the scoped dir).
"""

import importlib.util
import json
import sys
from pathlib import Path

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2169")
MAIN = Path("/home/thomasjiralerspong/explore-persona-space")
OLD_COPY = Path("/tmp/i2169_vtb_old.py")  # pre-extracted origin/main copy


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
    reg = registry.get(str(issue), {}) if isinstance(registry, dict) else {}
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
    r_new = new.check_orphaned_per_unit_figures(body, issue=issue)
    r_old = old.check_orphaned_per_unit_figures(body, issue=issue)
    c_count = r_new.detail.count("committed-figure-unmentioned")
    b_new = r_new.detail.count("companion-named-not-embedded")
    b_old = r_old.detail.count("companion-named-not-embedded")
    a_new = r_new.detail.count("never mentioned in the body")
    a_old = r_old.detail.count("never mentioned in the body")
    rows.append(
        {
            "issue": issue,
            "has_clean_result": has_cr,
            "triggered": triggered,
            "n_pngs": len(pngs),
            "class_c": c_count,
            "a_old": a_old,
            "a_new": a_new,
            "b_old": b_old,
            "b_new": b_new,
            "warn_old": bool(r_old.is_warn),
            "warn_new": bool(r_new.is_warn),
        }
    )

trig = [r for r in rows if r["triggered"]]
gain = [r for r in trig if r["class_c"] > 0]
ab_new_warn = [r for r in rows if (r["a_new"] > r["a_old"]) or (r["b_new"] > r["b_old"])]
ab_loosened = [r for r in rows if (r["a_new"] < r["a_old"]) or (r["b_new"] < r["b_old"])]

print(f"bodies scanned (numeric task dirs): {len(rows)}")
print(f"  with has_clean_result=true:       {sum(1 for r in rows if r['has_clean_result'])}")
print(f"TRIGGERED (>=1 reachable cited SHA & >=1 committed PNG): {len(trig)}")
print(f"  triggered & has_clean_result:     {sum(1 for r in trig if r['has_clean_result'])}")
print(f"triggered bodies gaining >=1 class-C entry: {len(gain)}"
      f" ({100.0 * len(gain) / len(trig):.1f}% of triggered)" if trig else "no triggered bodies")
mx = max((r["class_c"] for r in rows), default=0)
print(f"max class-C entries on any single body: {mx}")
print(f"bodies where class A/B GAINED entries (must be 0): {len(ab_new_warn)}")
print(f"bodies where class A/B LOST entries (the disclosed #2169 loosening): "
      f"{[(r['issue'], r['a_old'] - r['a_new'], r['b_old'] - r['b_new']) for r in ab_loosened]}")
print("\nclass-C gainers (issue, has_cr, n_pngs, class_c):")
for r in sorted(gain, key=lambda r: -r["class_c"]):
    print(f"  #{r['issue']:>5}  cr={r['has_clean_result']}  pngs={r['n_pngs']:>3}  C={r['class_c']}")
dist = {}
for r in trig:
    dist[r["class_c"]] = dist.get(r["class_c"], 0) + 1
print("\nclass-C count distribution over triggered bodies:", dict(sorted(dist.items())))
Path("/tmp/i2169_k1_rows.json").write_text(json.dumps(rows, indent=1))
print("\nrows persisted to /tmp/i2169_k1_rows.json")
