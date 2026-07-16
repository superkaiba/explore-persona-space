"""#1395 corpus regression audit — old-vs-new verify_plan c32 battery-branch diff.

Acceptance instrument for task #1395 (plan v4 §5.2): runs the PRE-widening
verify_plan (materialized from ``git show <old-ref>:scripts/verify_plan.py``)
and the POST-widening verify_plan (this script's sibling
``scripts/verify_plan.py``) over the full plan corpus (``.claude/plans/*.md``
+ ``tasks/*/*/plans/*.md`` at the repo root, read-only), compares per-check
verdict LABELS on the IDENTICAL file list + file contents, and asserts the
PER-ROW/BRANCH-ATTRIBUTED flip allowlist mechanically:

- ALLOWED: a ``c32_fit_basis_grounding`` flip attributable SOLELY to
  battery-branch activation — ``SKIP→{WARN,PASS}``, or ``PASS→WARN`` caused
  only by a new battery-row offender — with the calibrated FIT branch's
  per-row verdicts byte-stable old-vs-new (the fit regexes + satisfier are
  untouched; the audit verifies rather than assumes it);
- FORBIDDEN: any flip on any OTHER check; any change in fit-branch row
  verdicts; any other c32 transition (e.g. ``WARN→PASS``) — exit 1.

Per battery-branch FIRE (label flip or not — an already-WARN plan gaining a
battery fire counts toward the §7 recent-era criterion) the JSON records
which ``_BATTERY_TRIGGER_RE`` arm matched (battery-framing vs bare
>=100-count — actionable for the §7 named narrowing) and, for grounded
rows, the matched satisfier (pilot-gated vs provenance token + timing — a
PASS via an over-broad provenance token is invisible otherwise).

Every file is verified with ``kind="experiment"`` uniformly: the corpus
spans task kinds and eras, but the audit is a LABEL DIFF of two module
versions on identical inputs, so the kind choice cancels out — and for the
battery-fire counts it UPPER-BOUNDS the production fire set (kind-exempt
plans would SKIP in production).

Usage (from the issue-1395 worktree; ~3-4 min, pure regex file-scan):

    OMP_NUM_THREADS=8 uv run python scripts/issue1395_corpus_audit.py \
        --repo-root /home/thomasjiralerspong/explore-persona-space \
        --json-out /tmp/i1395_corpus_audit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKTREE_ROOT = SCRIPT_DIR.parent

C32 = "c32_fit_basis_grounding"
ALLOWED_C32_TRANSITIONS = {("SKIP", "WARN"), ("SKIP", "PASS"), ("PASS", "WARN")}
RECENT_ERA_MIN_ISSUE = 1000

# Mirror of _BATTERY_TRIGGER_RE's two arm classes (verify_plan.py c12) for
# per-fire attribution — kept local so the audit can attribute even if the
# module-level regex is later refactored.
_FRAMING_ARM_RE = re.compile(
    r"(?i)\b(null[- ]?(draws?|batter(y|ies))"
    r"|permutation[- ](tests?|batter(y|ies)|nulls?|draws?)"
    r"|n_(draws|perms)\b)"
)
_COUNT_ARM_RE = re.compile(
    r"(?i)(?<![\d\u2013\u2014-])\d{3,}\s+(null[- ])?(draws|permutations|resamples)"
)


def _load_module(path: Path, name: str):
    """Import a verify_plan copy from ``path`` under a unique module name
    (``sys.modules`` registration BEFORE ``exec_module`` — the dataclass
    replay requirement)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _materialize_old(old_ref: str, tmp_dir: str) -> Path:
    """Write ``<old-ref>:scripts/verify_plan.py`` (resolved via this
    worktree's shared object db) to a temp file and return its path."""
    blob = subprocess.run(
        ["git", "-C", str(WORKTREE_ROOT), "show", f"{old_ref}:scripts/verify_plan.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    old_path = Path(tmp_dir) / "verify_plan_old.py"
    old_path.write_text(blob)
    return old_path


def _corpus(repo_root: Path) -> list[Path]:
    """The audit corpus: repo-root ``.claude/plans/*.md`` +
    ``tasks/*/*/plans/*.md`` (sorted; ``plan.md`` symlinks included as their
    own corpus entries). Enumerated from the MAIN repo root, never a
    worktree (worktrees are sparse)."""
    return sorted(repo_root.glob(".claude/plans/*.md")) + sorted(
        repo_root.glob("tasks/*/*/plans/*.md")
    )


def _labels(mod, text: str) -> dict[str, str]:
    """Per-check verdict labels for ``text`` under ``mod`` (kind=experiment)."""
    _ok, results = mod.verify_plan_text(text, kind="experiment")
    return {r.id: r.status for r in results}


def _issue_of(rel: str) -> int | None:
    """Best-effort issue number for the recent-era slice: the tasks/<status>/<N>/
    path segment, else an ``issue[-_]<N>`` token in the filename."""
    parts = Path(rel).parts
    if len(parts) >= 3 and parts[0] == "tasks" and parts[2].isdigit():
        return int(parts[2])
    m = re.search(r"issue[-_]?(\d+)", Path(rel).name)
    return int(m.group(1)) if m else None


def _fit_row_verdicts(mod, text: str) -> list[tuple[str, bool]]:
    """(row_text, grounded) per FIT-branch row under ``mod``'s own regexes —
    the byte-stability oracle for the calibrated fit branch (NA escapes are
    branch-level and deliberately excluded: raw per-row verdicts are the
    finer-grained invariant)."""
    verdicts: list[tuple[str, bool]] = []
    for _comp, basis, wall, row_text in mod._c26_compute_table_rows(text):
        if not (mod._C32_KERNEL_RE.search(row_text) and mod._C32_LOOP_RE.search(row_text)):
            continue
        conv = f"{basis} {wall}"
        grounded = bool(
            (mod._C32_PROVENANCE_RE.search(conv) and mod._C32_TIMING_RE.search(conv))
            or mod._C32_PILOT_GATED_RE.search(row_text)
        )
        verdicts.append((row_text, grounded))
    return verdicts


def _battery_row_attribution(mod, text: str) -> list[dict]:
    """Per BATTERY-branch row under the NEW module: grounded verdict, which
    trigger arm(s) matched, and the matched satisfier for grounded rows."""
    rows = mod._c26_compute_table_rows(text)
    out: list[dict] = []
    for comp, basis, wall, row_text in rows:
        if mod._C32_KERNEL_RE.search(row_text) and mod._C32_LOOP_RE.search(row_text):
            continue  # fit row — fit branch governs
        if not mod._BATTERY_TRIGGER_RE.search(row_text):
            continue
        conv = f"{basis} {wall}"
        prov = mod._C32_PROVENANCE_RE.search(conv)
        timing = mod._C32_TIMING_RE.search(conv)
        pilot = mod._C32_PILOT_GATED_RE.search(row_text)
        grounded = bool((prov and timing) or pilot)
        satisfier = None
        if grounded:
            if pilot:
                satisfier = "pilot-gated"
            else:
                satisfier = f"provenance={prov.group(0)!r} timing={timing.group(0)!r}"
        out.append(
            {
                "component": comp.strip()[:100],
                "grounded": grounded,
                "arm_framing": bool(_FRAMING_ARM_RE.search(row_text)),
                "arm_count": bool(_COUNT_ARM_RE.search(row_text)),
                "satisfier": satisfier,
            }
        )
    return out


def main() -> int:
    """Run the audit; exit 0 iff every label transition is on the §5.2
    per-row/branch-attributed allowlist."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root", type=Path, default=Path("/home/thomasjiralerspong/explore-persona-space")
    )
    ap.add_argument(
        "--old-ref", default="origin/main", help="git ref holding the PRE-widening verify_plan.py"
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    new_mod = _load_module(SCRIPT_DIR / "verify_plan.py", "verify_plan_new_1395")
    with tempfile.TemporaryDirectory() as tmp:
        old_path = _materialize_old(args.old_ref, tmp)
        old_mod = _load_module(old_path, "verify_plan_old_1395")

        files = _corpus(args.repo_root)
        print(f"corpus: {len(files)} plan files under {args.repo_root}")

        flips: list[dict] = []
        violations: list[dict] = []
        battery_fires: list[dict] = []
        skipped: list[str] = []
        n_checked = 0
        for f in files:
            try:
                text = f.read_text()
            except (FileNotFoundError, OSError) as e:
                # The repo-root tasks/ tree is LIVE (concurrent sessions
                # git-mv task folders); a file vanishing between enumeration
                # and read is skipped identically for BOTH modules.
                skipped.append(f"{f}: {e.__class__.__name__}")
                continue
            old_labels = _labels(old_mod, text)
            new_labels = _labels(new_mod, text)
            n_checked += 1
            rel = str(f.relative_to(args.repo_root))
            issue = _issue_of(rel)

            fit_old = _fit_row_verdicts(old_mod, text)
            fit_new = _fit_row_verdicts(new_mod, text)
            fit_stable = fit_old == fit_new
            batt = _battery_row_attribution(new_mod, text)
            if batt:
                battery_fires.append(
                    {
                        "file": rel,
                        "issue": issue,
                        "old_c32": old_labels.get(C32, "ABSENT"),
                        "new_c32": new_labels.get(C32, "ABSENT"),
                        "rows": batt,
                    }
                )

            for cid in sorted(set(old_labels) | set(new_labels)):
                o, n = old_labels.get(cid, "ABSENT"), new_labels.get(cid, "ABSENT")
                if o == n:
                    continue
                flip = {"file": rel, "issue": issue, "check": cid, "old": o, "new": n}
                if cid == C32 and (o, n) in ALLOWED_C32_TRANSITIONS and fit_stable and batt:
                    flip["battery_rows"] = batt
                    flips.append(flip)
                else:
                    if cid == C32:
                        flip["fit_stable"] = fit_stable
                        flip["n_battery_rows"] = len(batt)
                    violations.append(flip)
            if not fit_stable:
                violations.append(
                    {
                        "file": rel,
                        "issue": issue,
                        "check": C32,
                        "old": "fit-row-verdicts",
                        "new": "CHANGED",
                    }
                )

        print(f"checked: {n_checked} files; skipped (vanished mid-run): {len(skipped)}")
        for s in skipped:
            print(f"  SKIPPED {s}")

        fire_issues = sorted({bf["issue"] for bf in battery_fires if bf["issue"] is not None})
        recent_fire_issues = [i for i in fire_issues if i >= RECENT_ERA_MIN_ISSUE]
        warn_fires = [bf for bf in battery_fires if any(not r["grounded"] for r in bf["rows"])]
        warn_issues = sorted({bf["issue"] for bf in warn_fires if bf["issue"] is not None})
        print(
            f"\nbattery-branch fires: {len(battery_fires)} plan-versions across "
            f"{len(fire_issues)} distinct issues {fire_issues}"
        )
        print(
            f"  with >=1 ungrounded battery row: {len(warn_fires)} plan-versions "
            f"across {len(warn_issues)} issues {warn_issues}"
        )
        print(
            f"  recent era (issue >= {RECENT_ERA_MIN_ISSUE}): "
            f"{len(recent_fire_issues)} distinct issues {recent_fire_issues}"
        )
        for bf in battery_fires:
            arms = [
                ("framing" if r["arm_framing"] else "") + ("+count" if r["arm_count"] else "")
                for r in bf["rows"]
            ]
            print(
                f"  FIRE {bf['file']} c32 {bf['old_c32']}->{bf['new_c32']} "
                f"rows={len(bf['rows'])} arms={arms} "
                f"satisfiers={[r['satisfier'] for r in bf['rows']]}"
            )

        print(f"\nallowed flips: {len(flips)}")
        for fl in flips:
            print(f"  {fl['check']} {fl['old']}->{fl['new']}  {fl['file']}")
        print(f"\nviolations (forbidden transitions): {len(violations)}")
        for v in violations:
            print(f"  VIOLATION {v['check']} {v['old']}->{v['new']}  {v['file']}")

        if args.json_out:
            args.json_out.write_text(
                json.dumps(
                    {
                        "n_files": len(files),
                        "n_checked": n_checked,
                        "skipped": skipped,
                        "battery_fires": battery_fires,
                        "fire_issues": fire_issues,
                        "recent_fire_issues": recent_fire_issues,
                        "warn_fire_issues": warn_issues,
                        "allowed_flips": flips,
                        "violations": violations,
                    },
                    indent=2,
                )
            )
            print(f"\nwrote {args.json_out}")

        if violations:
            print("\nAUDIT FAIL: forbidden label transitions found.")
            return 1
        print("\nAUDIT PASS: every label transition is on the §5.2 allowlist.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
