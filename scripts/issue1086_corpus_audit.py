"""#1086 corpus regression audit — old-vs-new verify_plan label diff.

Acceptance instrument for task #1086 (plan v2 §4.3): runs the PRE-fix
verify_plan (materialized from ``git show <old-ref>:scripts/verify_plan.py``)
and the POST-fix verify_plan (this script's sibling ``scripts/verify_plan.py``)
over the full plan corpus (``.claude/plans/*.md`` + ``tasks/*/*/plans/*.md``
at the repo root, read-only), compares per-check verdict LABELS on the
IDENTICAL file list + file contents (each file is read ONCE and both modules
verify the same text), and asserts the expected-flip ALLOWLIST mechanically:

- ALLOWED: ``c12_battery_multiplier`` FAIL→PASS anywhere (every flip is
  printed with its newly-matching arithmetic line for one-line genuine/false
  classification in the implementer report);
- ALLOWED: ``c18_paired_contrast_source_coverage`` FAIL→PASS ONLY on files
  under a ``tasks/*/833/plans/`` prefix (the #833 fixture family, incl. the
  ``plan.md`` symlink and post-planning siblings);
- FORBIDDEN: any other transition on any check (SKIP→FAIL, PASS→FAIL,
  WARN→anything, a c18 flip outside the #833 prefix, a flip on any other
  check) — exit 1.

Every file is verified with ``kind="experiment"`` uniformly: the corpus
spans task kinds and eras (pre-router 2026-05 plans included), but the audit
is a LABEL DIFF of two module versions on identical inputs, so the kind
choice cancels out — ``experiment`` is the strictest (no WARN degradation),
maximizing FAIL-surface sensitivity to the diff.

Usage (from the issue-1086 worktree; ~3-4 min, pure regex file-scan):

    OMP_NUM_THREADS=8 uv run python scripts/issue1086_corpus_audit.py \
        --repo-root /home/thomasjiralerspong/explore-persona-space \
        --json-out /tmp/i1086_corpus_audit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKTREE_ROOT = SCRIPT_DIR.parent

ALLOWED_C12 = ("c12_battery_multiplier", "FAIL", "PASS")
ALLOWED_C18 = ("c18_paired_contrast_source_coverage", "FAIL", "PASS")


def _load_module(path: Path, name: str):
    """Import a verify_plan copy from ``path`` under a unique module name."""
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
    """The §4.3 corpus: repo-root ``.claude/plans/*.md`` + ``tasks/*/*/plans/*.md``
    (sorted; ``plan.md`` symlinks included as their own corpus entries)."""
    files = sorted(repo_root.glob(".claude/plans/*.md")) + sorted(
        repo_root.glob("tasks/*/*/plans/*.md")
    )
    return files


def _labels(mod, text: str) -> dict[str, str]:
    """Per-check verdict labels for ``text`` under ``mod`` (kind=experiment)."""
    _ok, results = mod.verify_plan_text(text, kind="experiment")
    return {r.id: r.status for r in results}


def _is_833_fixture(path: Path) -> bool:
    """True iff ``path`` sits under a ``tasks/<status>/833/plans/`` prefix."""
    parts = path.parts
    return len(parts) >= 3 and parts[-2] == "plans" and parts[-3] == "833" and "tasks" in parts


def _first_arith_lines(mod, text: str, limit: int = 2) -> list[str]:
    """First ``limit`` fence-masked lines matching the (new) arithmetic regex —
    the classification aid for a c12 FAIL→PASS flip."""
    lines = text.splitlines()
    mask = mod._fence_mask(lines)
    hits: list[str] = []
    for line, fenced in zip(lines, mask, strict=True):
        if fenced or not mod._MULT_ARITH_RE.search(line):
            continue
        hits.append(line.strip()[:180])
        if len(hits) >= limit:
            break
    return hits


def main() -> int:
    """Run the audit; exit 0 iff every label transition is on the allowlist."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root", type=Path, default=Path("/home/thomasjiralerspong/explore-persona-space")
    )
    ap.add_argument("--old-ref", default="main", help="git ref holding the PRE-fix verify_plan.py")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    new_mod = _load_module(SCRIPT_DIR / "verify_plan.py", "verify_plan_new_1086")
    with tempfile.TemporaryDirectory() as tmp:
        old_path = _materialize_old(args.old_ref, tmp)
        old_mod = _load_module(old_path, "verify_plan_old_1086")

        files = _corpus(args.repo_root)
        print(f"corpus: {len(files)} plan files under {args.repo_root}")

        flips: list[dict] = []
        violations: list[dict] = []
        skipped: list[str] = []
        n_checked = 0
        for f in files:
            try:
                text = f.read_text()
            except (FileNotFoundError, OSError) as e:
                # The repo-root tasks/ tree is LIVE (concurrent sessions git-mv
                # task folders); a file vanishing between enumeration and read
                # is skipped identically for BOTH modules — never a label diff.
                skipped.append(f"{f}: {e.__class__.__name__}")
                continue
            old_labels = _labels(old_mod, text)
            new_labels = _labels(new_mod, text)
            n_checked += 1
            all_ids = sorted(set(old_labels) | set(new_labels))
            for cid in all_ids:
                o, n = old_labels.get(cid, "ABSENT"), new_labels.get(cid, "ABSENT")
                if o == n:
                    continue
                rel = str(f.relative_to(args.repo_root))
                flip = {"file": rel, "check": cid, "old": o, "new": n}
                if (cid, o, n) == ALLOWED_C12:
                    flip["arith_lines"] = _first_arith_lines(new_mod, text)
                    flips.append(flip)
                elif (cid, o, n) == ALLOWED_C18 and _is_833_fixture(f):
                    flips.append(flip)
                else:
                    violations.append(flip)

        print(f"checked: {n_checked} files; skipped (vanished mid-run): {len(skipped)}")
        for s in skipped:
            print(f"  SKIPPED {s}")
        print(f"\nallowed flips: {len(flips)}")
        for fl in flips:
            print(f"  {fl['check']} {fl['old']}->{fl['new']}  {fl['file']}")
            for line in fl.get("arith_lines", []):
                print(f"      arith: {line}")
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
        print("\nAUDIT PASS: every label transition is on the §4.3 allowlist.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
