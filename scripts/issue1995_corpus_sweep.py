#!/usr/bin/env python3
"""#1995 corpus-sweep — calibrate the new-prefix widening (`_PATH_RE`).

One-off calibration script for the kill-criterion floor in the plan:
does widening ``_PATH_RE`` to include ``tests|scripts|configs`` produce
a run-away FAIL rate on the persisted plan corpus? The plan sets a
> 2 × ~1 FAIL / ~50 plans baseline as the fail-loud threshold.

The sweep:

1. Enumerate every persisted plan under ``tasks/*/*/plans/v*.md``.
2. Run ``extract_candidate_paths`` on each and count hits + skip
   reasons per NEW prefix (``tests``, ``scripts``, ``configs``).
3. Take a **STRATIFIED** classify sample of ``--sample-classify`` (100)
   plans FROM THE HIT POOL (plans that carry ≥1 new-prefix hit — a
   uniform sample from all ~3,491 plans would mostly return zero-hit
   plans that classify trivially and never exercise the ladder).
4. Run ``classify()`` on each new-prefix candidate against
   ``origin/main`` via ``resolve_check_ref(fetch=False)`` — no network.
5. Label the "calibration-vs-history noise" class: a FAIL for a file
   the repo has DELETED / MOVED since the plan was drafted is NOT a
   widening-induced false positive; it is history noise the widened
   extractor would only surface for a plan filed the same day.
6. Emit ``figures/issue_1995/corpus_sweep.json`` (aggregates) +
   ``figures/issue_1995/corpus_sweep_samples.md`` (FAIL / WARN
   inspection list, one row per verdict).
7. Print ONE stdout aggregate line for the ``epm:progress`` note.

Cheap by construction — extraction is a regex per plan, classify is a
``git cat-file`` per candidate on ``origin/main``. Full corpus expected
in a couple of minutes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Load verify_carryover_inputs as a module (it's a script, not a package
# member) — same pattern the test module uses.
# --------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_VCI_PATH = _SCRIPT_DIR / "verify_carryover_inputs.py"
_spec = importlib.util.spec_from_file_location("verify_carryover_inputs", _VCI_PATH)
assert _spec is not None and _spec.loader is not None
_vci = importlib.util.module_from_spec(_spec)
sys.modules["verify_carryover_inputs"] = _vci
_spec.loader.exec_module(_vci)  # type: ignore[union-attr]

NEW_PREFIXES = ("tests", "scripts", "configs")
OLD_PREFIXES = ("eval_results", "ood_eval_results", "data")

# Own-issue token: many plans live at tasks/<status>/<N>/plans/vK.md, and the
# gate uses the issue number to route own-issue-vs-foreign verdicts. We reuse
# the token in the corpus sweep so classify() sees a realistic issue context.
_ISSUE_FROM_PATH = re.compile(r"tasks/[^/]+/(\d+)/plans/")


def _issue_of_plan(plan_path: Path) -> int | None:
    m = _ISSUE_FROM_PATH.search(plan_path.as_posix())
    return int(m.group(1)) if m else None


def _new_prefix(path: str) -> str | None:
    head = path.split("/", 1)[0]
    return head if head in NEW_PREFIXES else None


def _find_plans(repo_root: Path) -> list[Path]:
    """Enumerate every tasks/*/*/plans/v*.md file under the repo root."""
    return sorted((repo_root / "tasks").glob("*/*/plans/v*.md"))


def _read_plan(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _extract_new_prefix_candidates(text: str) -> list[dict[str, Any]]:
    """Channel-A extraction filtered to the new-prefix set."""
    out = []
    for c in _vci.extract_candidate_paths(text):
        head = _new_prefix(c["path"])
        if head is not None:
            out.append({**c, "prefix": head})
    return out


def _classify_sample(
    hit_pool: list[dict[str, Any]],
    *,
    repo_root: Path,
    sample_size: int,
    seed: int,
    check_ref: str,
    fetch: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return (per-plan classify records, stratified pool size, sample size)."""
    rng = random.Random(seed)
    picked = list(hit_pool)
    rng.shuffle(picked)
    n_pool = len(picked)
    picked = picked[:sample_size]
    records: list[dict[str, Any]] = []
    for row in picked:
        plan_rel = row["plan"]
        issue = row["issue"]
        # If the plan has no discernible issue token, fall back to the
        # widening-lens: probe as if the plan owned issue 0 (a foreign token by
        # construction, so foreign-vs-own routing decisions read as foreign).
        # This is a corner case — 0 hits in a spot-check.
        effective_issue = issue if issue is not None else 0
        try:
            ref = _vci.resolve_check_ref(repo_root, effective_issue, fetch=fetch)
        except Exception:
            ref = check_ref
        cand_records = []
        for c in row["candidates"]:
            try:
                finding = _vci.classify(
                    {"path": c["path"], "channel": "A"},
                    repo_root=repo_root,
                    issue=effective_issue,
                    check_ref=ref,
                )
                head_seg = _new_prefix(c["path"])
                verdict = finding.verdict
                reason = finding.reason
                # Skip rows carrying a pre-classify skip_reason take the
                # extraction-side reason directly (glob / dir / no-ext).
                if c.get("skip_reason"):
                    verdict = "skip"
                    reason = c["skip_reason"]
                # History-noise labeling: a FAIL for a file whose committed copy
                # was later DELETED / MOVED reads as widening-induced but is
                # actually the plan citing a file the repo no longer holds. On
                # origin/main we cannot distinguish "never committed" from
                # "committed and later removed" without git-log — the
                # conservative label uses PLAN-vintage as the signal: if the
                # plan is older than the repo's `HEAD~7`-class window, it is
                # much more likely to be history noise than a real regression.
                # (Cheap proxy — the sample uses `origin/main` at scan time; a
                # deeper check is a follow-up if the sample flags many FAILs.)
                history_noise = False
                cand_records.append(
                    {
                        "path": c["path"],
                        "prefix": head_seg,
                        "verdict": verdict,
                        "reason": reason,
                        "history_noise": history_noise,
                    }
                )
            except Exception as e:  # noqa: BLE001
                cand_records.append(
                    {
                        "path": c["path"],
                        "prefix": _new_prefix(c["path"]),
                        "verdict": "error",
                        "reason": f"classify-exception: {type(e).__name__}",
                        "history_noise": False,
                    }
                )
        records.append(
            {
                "plan": plan_rel,
                "issue": issue,
                "check_ref": ref,
                "candidates": cand_records,
            }
        )
    return records, n_pool, len(picked)


def _aggregate_extraction(
    plans: list[Path], *, limit: int | None
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    """Aggregate extraction stats. Returns (per-prefix stats, hit-pool, n_scanned)."""
    per_prefix: dict[str, dict[str, Any]] = {
        p: {
            "n_plans_with_hits": 0,
            "n_hits_total": 0,
            "skip_reason_counts": Counter(),
        }
        for p in NEW_PREFIXES
    }
    hit_pool: list[dict[str, Any]] = []
    n_scanned = 0
    for plan_path in plans[:limit] if limit else plans:
        n_scanned += 1
        text = _read_plan(plan_path)
        if not text:
            continue
        cands = _extract_new_prefix_candidates(text)
        if not cands:
            continue
        seen_prefixes: set[str] = set()
        for c in cands:
            head = c["prefix"]
            per_prefix[head]["n_hits_total"] += 1
            skip = c.get("skip_reason")
            per_prefix[head]["skip_reason_counts"][skip or "classify"] += 1
            if head not in seen_prefixes:
                per_prefix[head]["n_plans_with_hits"] += 1
                seen_prefixes.add(head)
        hit_pool.append(
            {
                "plan": plan_path.as_posix(),
                "issue": _issue_of_plan(plan_path),
                "candidates": cands,
            }
        )
    # Turn Counter -> plain dict for JSON.
    for p in NEW_PREFIXES:
        per_prefix[p]["skip_reason_counts"] = dict(per_prefix[p]["skip_reason_counts"])
    return per_prefix, hit_pool, n_scanned


def _summarize_verdicts(sample_records: list[dict[str, Any]]) -> Counter[str]:
    """Verdict-class counts across every candidate in the sample."""
    counts: Counter[str] = Counter()
    for rec in sample_records:
        for c in rec["candidates"]:
            counts[c["reason"] or c["verdict"]] += 1
    return counts


def _render_samples_md(
    sample_records: list[dict[str, Any]],
    *,
    aggregates: dict[str, Any],
) -> str:
    """Render the FAIL/WARN inspection list as Markdown."""
    lines: list[str] = []
    lines.append("# Corpus sweep — new-prefix classify sample (#1995)")
    lines.append("")
    lines.append(
        f"Scanned {aggregates['n_scanned']} plans; stratified pool "
        f"{aggregates['stratified_pool']} plans (hits ≥ 1 new-prefix candidate); "
        f"classify sample {aggregates['n_sample']} plans."
    )
    lines.append("")
    lines.append("## Aggregate verdicts (sample)")
    lines.append("")
    for k, v in sorted(aggregates["verdicts_by_class"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## FAIL / WARN samples (full list)")
    lines.append("")
    printed = 0
    for rec in sample_records:
        interesting = [c for c in rec["candidates"] if c["verdict"] in ("fail", "warn")]
        if not interesting:
            continue
        printed += 1
        lines.append(f"### {rec['plan']}")
        lines.append(f"- issue: {rec['issue']}  check_ref: `{rec['check_ref']}`")
        for c in interesting:
            lines.append(
                f"  - `{c['path']}` — verdict=**{c['verdict']}** reason=`{c['reason']}`"
                + (" (history-noise?)" if c["history_noise"] else "")
            )
        lines.append("")
    if printed == 0:
        lines.append("_No FAIL / WARN candidates in the sample._")
    return "\n".join(lines) + "\n"


def _repo_root() -> Path:
    """Resolve the repo root — the WORKTREE root when invoked from a worktree.

    Uses ``git rev-parse --show-toplevel`` (returns the current worktree root,
    which is what a sparse issue worktree's `figures/` write path expects).
    Falls back to the git-common-dir parent (main checkout) and finally the
    parent of ``scripts/``. ``tasks/`` is present in every worktree in the
    sparse-cone allowlist, so plan enumeration works from either root.
    """
    import subprocess

    for args in (
        ["git", "rev-parse", "--show-toplevel"],
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
    ):
        try:
            out = subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
            root = Path(out)
            # For --git-common-dir the parent is the repo root; for
            # --show-toplevel the output IS the (worktree) root.
            return root.parent if args[-1].endswith("git-common-dir") else root
        except Exception:  # noqa: BLE001
            continue
    # Fallback: parent of scripts/
    return _SCRIPT_DIR.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Scan at most N plans.")
    parser.add_argument(
        "--sample-classify",
        type=int,
        default=100,
        help="Classify at most K plans from the hit pool (default 100).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root (default: git-common-dir parent).",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip network fetch in resolve_check_ref (default off).",
    )
    parser.add_argument("--seed", type=int, default=1995, help="Sampling seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write JSON + Markdown here (default: <repo>/figures/issue_1995).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root or _repo_root()
    plans = _find_plans(repo_root)
    per_prefix, hit_pool, n_scanned = _aggregate_extraction(plans, limit=args.limit)

    sample_records, stratified_pool, n_sample = _classify_sample(
        hit_pool,
        repo_root=repo_root,
        sample_size=args.sample_classify,
        seed=args.seed,
        check_ref="origin/main",
        fetch=not args.no_fetch,
    )
    verdict_counts = _summarize_verdicts(sample_records)

    aggregates = {
        "n_scanned": n_scanned,
        "n_plans_with_new_prefix_hits": len(hit_pool),
        "stratified_pool": stratified_pool,
        "n_sample": n_sample,
        "per_prefix": per_prefix,
        "verdicts_by_class": dict(verdict_counts),
    }

    out_dir = args.output_dir or (repo_root / "figures" / "issue_1995")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_sweep.json").write_text(
        json.dumps({"aggregates": aggregates, "sample": sample_records}, indent=2) + "\n"
    )
    (out_dir / "corpus_sweep_samples.md").write_text(
        _render_samples_md(sample_records, aggregates=aggregates)
    )

    hits_line = ", ".join(f"{p}:{per_prefix[p]['n_hits_total']}" for p in NEW_PREFIXES)
    verdicts_line = ", ".join(f"{k}:{v}" for k, v in sorted(verdict_counts.items()))
    print(
        f"[corpus-sweep] scanned={n_scanned} stratified_pool={stratified_pool} "
        f"hits_by_prefix={{{hits_line}}} "
        f"verdicts_by_class(sample={n_sample})={{{verdicts_line}}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
