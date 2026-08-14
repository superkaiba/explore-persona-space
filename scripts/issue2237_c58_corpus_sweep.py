"""Re-run the c58 calibration sweep over the persisted-plan corpus (#2237).

Runs the SHIPPED ``check_fanout_pod_name_collision`` (imported from
``scripts/verify_plan.py`` — never a re-implementation, so the sweep stays
evidence about the shipped check) over every ``tasks/*/*/plans/v*.md`` and
prints, PER T2 POSTURE (explicit-``--backend runpod``-only vs
explicit+``auto`` — the #2237 plan §7 two-posture measurement), the WARN
count, the flagged paths, and ``n_skip`` split by SKIP class (the
``verify_plan.py`` calibration convention: skips never fold into genuine
passes). The SHIPPED posture's rows are the acceptance record; the
alternate posture is measured through the same shipped trigger code via
the posture-parameterized ``_c58_check`` core the public check pins.
This is the reviewer's reproduction tool for the plan-§7 calibration
record and the §8 acceptance sweep: c58 must fire on #2054
plans/v16.md, and the corpus WARN total must stay <= 20 (plan §15.2).
The corpus is live and drifts — the sweep reports its own n.

Usage::

    uv run python scripts/issue2237_c58_corpus_sweep.py [--repo-root PATH]

Exit code 0 always (a reporting tool, not a gate — adjudication stays
with the reader); the summary lines are machine-greppable
(``c58-sweep[<posture>]: n=... warns=... n_skip=...``).
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


_SKIP_CLASSES = (
    ("no-section-9", "no parseable section-9"),
    ("no-launch-argv", "no launch-shaped dispatch_issue.py command"),
    ("none-parses", "none dry-parses"),
    ("no-runpod-argv", "no RunPod-resolved launch argv"),
    ("no-fanout", "no concurrent box-level fan-out"),
)


def _skip_class(detail: str) -> str:
    """Bucket a SKIP detail string into its named class (never folded into
    passes; an unrecognized detail buckets as ``other`` so a future SKIP
    branch cannot silently vanish from the report)."""
    for label, needle in _SKIP_CLASSES:
        if needle in detail:
            return label
    return "other"


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
    shipped_auto = vp._C58_T2_INCLUDE_AUTO
    postures = [
        ("explicit-only", False),
        ("explicit+auto", True),
    ]
    for label, include_auto in postures:
        shipped = " (SHIPPED)" if include_auto == shipped_auto else ""
        warns: list[tuple[Path, str]] = []
        n_pass = 0
        skip_counts: dict[str, int] = {}
        for p in plans:
            text = p.read_text(errors="replace")
            if include_auto == shipped_auto:
                # The SHIPPED check — the acceptance record (kind is
                # irrelevant: c58 is all-kinds, `del kind` in the check).
                r = vp.check_fanout_pod_name_collision(text, "experiment")
            else:
                # The alternate posture, through the SAME shipped core.
                r = vp._c58_check(text, include_auto=include_auto)
            if r.status == "WARN":
                warns.append((p, r.detail))
            elif r.status == "SKIP":
                cls = _skip_class(r.detail)
                skip_counts[cls] = skip_counts.get(cls, 0) + 1
            else:
                n_pass += 1

        rel = args.repo_root
        for p, detail in warns:
            print(f"WARN[{label}] {p.relative_to(rel)}")
            print(f"     {detail[:200]}")
        n_skip = sum(skip_counts.values())
        skip_split = ", ".join(f"{k}={v}" for k, v in sorted(skip_counts.items()))
        print(
            f"c58-sweep[{label}]{shipped}: n={len(plans)} warns={len(warns)} "
            f"pass={n_pass} n_skip={n_skip} ({skip_split})"
        )
        # tasks/<status>/<id>/plans/vK.md -> parts = (tasks, <status>, <id>, ...)
        flagged_tasks = sorted({p.relative_to(rel).parts[2] for p, _ in warns})
        print(f"c58-sweep[{label}]: distinct flagged tasks = {flagged_tasks}")
        # Acceptance controls, reported (never enforced — adjudication is
        # the reader's; plan §8 probe 1 + §15.2 ceiling):
        v16 = [p for p, _ in warns if p.match("tasks/*/2054/plans/v16.md")]
        print(f"c58-sweep[{label}]: positive control (#2054 v16 WARNs) = {bool(v16)}")
        print(f"c58-sweep[{label}]: corpus ceiling (warns <= 20) = {len(warns) <= 20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
