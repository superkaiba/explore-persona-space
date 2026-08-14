"""Re-run the c62 + c63 calibration sweep over the persisted-plan corpus (#2276).

Runs the SHIPPED ``check_backend_pin_claim`` (c62) and
``check_declared_width_vs_launch`` (c63) — imported from
``scripts/verify_plan.py``, never a re-implementation, so the sweep stays
evidence about the shipped checks — over every ``tasks/*/*/plans/v*.md``
and prints, per check, the FAIL/WARN rows, the PASS rows, and ``n_skip``
split by SKIP class (the ``verify_plan.py`` calibration convention: skips
never fold into genuine passes). c62's ``frontmatter_backend`` is read from
each task's sibling ``body.md`` at SWEEP time — exactly the live state the
``--issue``-mode check (and dispatch) reads; a pin added AFTER a plan
version was authored therefore reads PASS on that version (the intended
semantics, recorded as an adjudication caveat in the calibration blocks).

This is the reviewer's reproduction tool for the calibration comments above
each check in ``scripts/verify_plan.py``: the binding sweep-validity
criterion (#2276 plan §4 step 6) is that the sweep recovers #2225 v5-v9 as
c62 hits (FAILs at the plan's designed polarity; WARNs as shipped — the
pre-registered >2-FP downgrade rule fired, see the c62 calibration block)
and #2225 v9 as a c63 WARN — a zero-hit sweep is a broken sweep, never
zero FPs. The corpus is live and drifts — the sweep reports its own n.

Usage::

    uv run python scripts/issue2276_c62c63_corpus_sweep.py [--repo-root PATH]

Exit code 0 always (a reporting tool, not a gate — adjudication stays with
the reader); the summary lines are machine-greppable
(``c62-sweep: ...`` / ``c63-sweep: ...``).
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


_C62_SKIP_CLASSES = (("no-claim", "no §9 backend pin-claim"),)
_C63_SKIP_CLASSES = (
    ("no-section-9", "no parseable §9"),
    ("no-multi-gpu-width", "no multi-GPU width"),
    ("no-launch-argv", "no launch-shaped"),
    ("cli-unavailable", "build_argparser unavailable"),
    ("no-width-contribution", "contributes a width"),
)


def _skip_class(detail: str, classes) -> str:
    """Bucket a SKIP detail into its named class; unrecognized details bucket
    as ``other`` so a future SKIP branch cannot silently vanish."""
    for label, needle in classes:
        if needle in detail:
            return label
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo root holding tasks/ (default: this checkout)",
    )
    args = ap.parse_args()
    vp = _load_verify_plan(args.repo_root)

    plan_files = sorted(args.repo_root.glob("tasks/*/*/plans/v*.md"))
    c62_rows: dict[str, list[str]] = {"FAIL": [], "WARN": [], "PASS": []}
    c62_skips: dict[str, int] = {}
    c63_rows: dict[str, list[str]] = {"WARN": [], "PASS": []}
    c63_skips: dict[str, int] = {}

    for path in plan_files:
        rel = path.relative_to(args.repo_root)
        plan = path.read_text(errors="replace")
        body_path = path.parent.parent / "body.md"
        fm_backend = None
        if body_path.exists():
            fm, _ = vp.split_frontmatter(body_path.read_text(errors="replace"))
            raw = fm.get("backend")
            if raw is not None and str(raw).strip():
                fm_backend = str(raw).strip()
        r62 = vp.check_backend_pin_claim(plan, frontmatter_backend=fm_backend)
        if r62.status == "SKIP":
            key = _skip_class(r62.detail, _C62_SKIP_CLASSES)
            c62_skips[key] = c62_skips.get(key, 0) + 1
        else:
            c62_rows.setdefault(r62.status, []).append(
                f"{rel} [fm={fm_backend}] {r62.detail[:110]}"
            )
        r63 = vp.check_declared_width_vs_launch(plan, "experiment")
        if r63.status == "SKIP":
            key = _skip_class(r63.detail, _C63_SKIP_CLASSES)
            c63_skips[key] = c63_skips.get(key, 0) + 1
        else:
            c63_rows.setdefault(r63.status, []).append(f"{rel} {r63.detail[:110]}")

    n = len(plan_files)
    for status in ("FAIL", "WARN", "PASS"):
        for row in c62_rows.get(status, []):
            print(f"c62 {status}: {row}")
    print(
        f"c62-sweep: n={n} fails={len(c62_rows.get('FAIL', []))} "
        f"warns={len(c62_rows.get('WARN', []))} "
        f"passes={len(c62_rows.get('PASS', []))} "
        f"n_skip={sum(c62_skips.values())} skip_classes={sorted(c62_skips.items())}"
    )
    for status in ("WARN", "PASS"):
        for row in c63_rows.get(status, []):
            print(f"c63 {status}: {row}")
    print(
        f"c63-sweep: n={n} warns={len(c63_rows.get('WARN', []))} "
        f"passes={len(c63_rows.get('PASS', []))} "
        f"n_skip={sum(c63_skips.values())} skip_classes={sorted(c63_skips.items())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
