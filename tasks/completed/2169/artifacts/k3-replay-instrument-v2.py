"""K3 live replay for #2169, round 2 (throwaway; NOT committed to the repo tree).

Runs the NARROWED check 31 (worktree copy at 2c47f31aef, §3.0 plan-named
candidate filter) against the two REAL #2061 body revisions with
issue=2061 threaded, against the main repo's object DB. Adds the S6
assert on top of round 1's recipe: the result is only evidence when the
§3.0 filter ENGAGED (active plan-resolution mode in the detail — #2061
has 13 numeric plan revisions, every one naming f5_arm_agreement), so a
pass cannot come from the filter silently skipping class C.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2169")
MAIN = Path("/home/thomasjiralerspong/explore-persona-space")
CITED_SHA = "5d001092782beadab91efd81eb4d785741eeeac8"

spec = importlib.util.spec_from_file_location("vtb", WT / "scripts" / "verify_task_body.py")
vtb = importlib.util.module_from_spec(spec)
sys.modules["vtb"] = vtb
spec.loader.exec_module(vtb)
vtb._resolve_repo_root = lambda: MAIN

rc = subprocess.run(
    ["git", "-C", str(MAIN), "rev-parse", "--verify", "--quiet", f"{CITED_SHA}^{{commit}}"],
    capture_output=True,
    text=True,
).returncode
assert rc == 0, f"cited SHA {CITED_SHA[:12]} does NOT resolve locally — silence is not evidence"
print(f"precondition: cited SHA {CITED_SHA[:12]} resolves locally (rc=0)")

plan_text, plan_mode = vtb._plan_naming_text(2061)
print(
    f"precondition: #2061 plan resolution -> mode={plan_mode!r}, resolved={plan_text is not None}"
)
assert plan_text is not None, "S6 FAIL: #2061 plan context did not resolve — filter cannot engage"
assert vtb._stem_named_in_body(plan_text, "f5_arm_agreement"), (
    "S6 FAIL: f5_arm_agreement not plan-named — a silent result would be de-candidation"
)
print("precondition: f5_arm_agreement IS plan-named in #2061's plan union (S6)")

for label, path in [
    ("BEFORE-fix body (93566d735f:tasks/interpreting/2061/body.md)", "/tmp/i2169-2061-before.md"),
    ("CURRENT body (tasks/awaiting_promotion/2061/body.md)", "/tmp/i2169-2061-current.md"),
]:
    raw = Path(path).read_text()
    _fm, body = vtb.split_frontmatter(raw)
    r = vtb.check_orphaned_per_unit_figures(body, issue=2061)
    print(f"\n=== {label} ===")
    print(f"passed={r.passed} is_warn={r.is_warn}")
    print(f"f5_arm_agreement flagged: {'f5_arm_agreement.png' in r.detail}")
    print(f"class-C entries: {r.detail.count('committed-figure-unmentioned')}")
    print(f"class-B entries: {r.detail.count('companion-named-not-embedded')}")
    print(f"class-A entries: {r.detail.count('never mentioned in the body')}")
    engaged = "plan-named figures only" in r.detail
    print(f"S6 filter ENGAGED (active mode stated in detail): {engaged}")
    assert engaged, "S6 FAIL: detail does not state the active plan-named mode"
    entry_re = r"`(figures/issue_2061/[^`]+)` \(([^)]*)\)"
    f1_leak = sum(
        1
        for p, c in vtb.re.findall(entry_re, r.detail)
        if "f1_delta_scatter" in p and "committed-figure-unmentioned" in c
    )
    print(f"f1_delta_scatter class-C leak: {f1_leak}")
    print(f"n_unreachable mentioned: {'not locally reachable' in r.detail}")
    print(f"detail: {r.detail[:1200]}")
