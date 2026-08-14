"""Re-run the c57 calibration sweep over the persisted-plan corpus (#2236).

Runs the SHIPPED ``check_fanout_prefix_staging`` (imported from
``scripts/verify_plan.py`` — never a re-implementation, so the sweep stays
evidence about the shipped check) over every ``tasks/*/*/plans/v*.md`` and
prints the WARN count, the flagged paths, and ``n_skip`` (split into the
no-parseable-section-9 and trigger-not-fired classes, the
``verify_plan.py`` calibration convention: skips never fold into genuine
passes). This is the reviewer's reproduction tool for the plan-v3 D4
calibration record (6 WARNs / 0 FP on the 2026-08-11 corpus) and the K1
acceptance sweep: c57 must fire on #2054 plans/v14.md, must NOT fire on
v12.md, and the corpus WARN total must stay <= 20. The corpus is live and
drifts — the sweep reports its own n.

Usage::

    uv run python scripts/issue2236_c57_corpus_sweep.py [--repo-root PATH]

Exit code 0 always (a reporting tool, not a gate — K1 adjudication stays
with the reader); the summary line is machine-greppable
(``c57-sweep: n=... warns=... n_skip=...``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_verify_plan(repo_root: Path):
    """Import the SHIPPED verify_plan module from ``repo_root/scripts``."""
    sys.path.insert(0, str(repo_root / "scripts"))
    import verify_plan  # noqa: PLC0415 — path-dependent import by design

    return verify_plan


def _task_kind(plan_path: Path) -> str:
    """Read ``kind:`` from the owning task's body.md frontmatter (default
    ``experiment``). c57 is kind-agnostic today, but the sweep threads the
    real kind so a future kind gate cannot silently desynchronize it."""
    body = plan_path.parent.parent / "body.md"
    try:
        text = body.read_text()
    except OSError:
        return "experiment"
    for line in text.splitlines():
        if line.startswith("kind:"):
            return line.split(":", 1)[1].strip() or "experiment"
    return "experiment"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo root holding tasks/ and scripts/verify_plan.py (default: this checkout)",
    )
    args = ap.parse_args(argv)
    vp = _load_verify_plan(args.repo_root)

    plans = sorted(args.repo_root.glob("tasks/*/*/plans/v*.md"))
    warns: list[tuple[Path, str]] = []
    n_pass = 0
    n_skip_no_s9 = 0
    n_skip_no_trigger = 0
    for p in plans:
        r = vp.check_fanout_prefix_staging(p.read_text(errors="replace"), _task_kind(p))
        if r.status == "WARN":
            warns.append((p, r.detail))
        elif r.status == "SKIP":
            if "no parseable section-9" in r.detail:
                n_skip_no_s9 += 1
            else:
                n_skip_no_trigger += 1
        else:
            n_pass += 1

    rel = args.repo_root
    for p, detail in warns:
        print(f"WARN {p.relative_to(rel)}")
        print(f"     {detail[:200]}")
    n_skip = n_skip_no_s9 + n_skip_no_trigger
    print(
        f"c57-sweep: n={len(plans)} warns={len(warns)} pass={n_pass} "
        f"n_skip={n_skip} (no-section-9={n_skip_no_s9}, trigger-not-fired={n_skip_no_trigger})"
    )
    # tasks/<status>/<id>/plans/vK.md -> parts = (tasks, <status>, <id>, ...)
    flagged_tasks = sorted({p.relative_to(rel).parts[2] for p, _ in warns})
    print(f"c57-sweep: distinct flagged tasks = {flagged_tasks}")

    # K1 controls, reported (never enforced here — adjudication is the reader's):
    v14 = [p for p, _ in warns if p.match("tasks/*/2054/plans/v14.md")]
    v12_warned = [p for p, _ in warns if p.match("tasks/*/2054/plans/v12.md")]
    print(f"c57-sweep: K1 positive control (#2054 v14 WARNs) = {bool(v14)}")
    print(f"c57-sweep: K1 negative control (#2054 v12 quiet) = {not v12_warned}")
    print(f"c57-sweep: K1 corpus ceiling (warns <= 20) = {len(warns) <= 20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
