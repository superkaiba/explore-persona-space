#!/usr/bin/env python
"""Audit (ii): permissive span-locator sweep over parent #1345 inserted-arm rejects.

Under whitespace-tolerant + one-edit-distance matching, count how many of the
inserted-arm rejects parent #1345 discarded carry a locatable answer span.
If ≥30% recoverable, the inserted arm gains free rows for task #2054.

Pure VM Python; no API, no GPU. Enumerates parent inserts via HF list_repo_tree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _norm_ws(s: str) -> str:
    """Whitespace-tolerant normalization: collapse runs to a single space."""
    return re.sub(r"\s+", " ", s).strip()


def _levenshtein_at_most_one(a: str, b: str) -> bool:
    """True iff edit distance between a and b is at most 1 (subst/ins/del)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    # la <= lb, lb-la in {0, 1}
    i = j = edits = 0
    while i < la and j < lb:
        if a[i] != b[j]:
            edits += 1
            if edits > 1:
                return False
            if la == lb:
                i += 1
                j += 1
            else:
                j += 1
        else:
            i += 1
            j += 1
    return True


def find_span_exact(text: str, answer: str) -> bool:
    return answer in text


def find_span_permissive(text: str, answer: str, max_edit: int = 1) -> bool:
    if find_span_exact(text, answer):
        return True
    nt = _norm_ws(text)
    na = _norm_ws(answer)
    if na in nt:
        return True
    if max_edit == 0 or len(na) < 3:
        return False
    # Sliding window over normalized text at length |na|
    L = len(na)
    for k in range(0, max(0, len(nt) - L + 1)):
        window = nt[k : k + L]
        if _levenshtein_at_most_one(window, na):
            return True
    return False


def _dry_run() -> int:
    cases = [
        ("The answer is 42 exactly.", "42", True, True),
        ("The answer is  42  exactly.", "42", True, True),  # ws-tolerant
        ("Nothing matches here.", "42", False, False),
        ("The result equals fortytwo total.", "fortitwo", False, True),  # 1-edit (sub y→i)
        ("The result equals fortytwo total.", "fortytwoo", False, True),  # 1-edit (insert)
        ("The result equals fortytwo total.", "abcdefg", False, False),
        ("Value: seventeen and change.", "seventeen", True, True),
        ("Value:\n\nseventeen and change.", "seventeen", True, True),
    ]
    ok = True
    for text, ans, want_exact, want_perm in cases:
        got_exact = find_span_exact(text, ans)
        got_perm = find_span_permissive(text, ans, max_edit=1)
        if got_exact != want_exact:
            print(
                f"FAIL exact: text={text!r} ans={ans!r} got={got_exact} want={want_exact}",
                file=sys.stderr,
            )
            ok = False
        if got_perm != want_perm:
            print(
                f"FAIL permissive: text={text!r} ans={ans!r} got={got_perm} want={want_perm}",
                file=sys.stderr,
            )
            ok = False
    if ok:
        print("dry-run: all 8 fixtures pass (exact + permissive matchers OK)")
        return 0
    return 1


def run_audit(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = args.parent_repo
    prefix = args.parent_prefix

    # Enumerate variant subdirs under the parent prefix.
    try:
        top = list(
            api.list_repo_tree(
                repo_id=repo_id, path_in_repo=prefix, repo_type="dataset", recursive=False
            )
        )
    except Exception as exc:
        print(f"ERROR enumerating {repo_id}:{prefix}: {exc}", file=sys.stderr)
        return 2

    variant_dirs = [entry.path for entry in top if type(entry).__name__ == "RepoFolder"]

    n_rejects_scanned = 0
    n_recoverable_exact = 0
    n_recoverable_permissive = 0
    per_variant: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "recoverable_exact": 0, "recoverable_permissive": 0}
    )
    parent_reject_paths_scanned: list[str] = []

    # For each variant, list files matching an inserted-reject-ish pattern.
    # Parent's convention (grep-anchored to worktree issue1345_common.py) uses
    # slugs like `_ins_` or `inserted` for the inserted arm — we accept either
    # naming plus a `reject` marker in the filename.
    reject_pat = re.compile(r"(reject|rej_|_rej|discard)", re.IGNORECASE)
    inserted_pat = re.compile(r"(insert|_ins_|_ins$)", re.IGNORECASE)

    for variant_path in variant_dirs:
        try:
            files = list(
                api.list_repo_tree(
                    repo_id=repo_id, path_in_repo=variant_path, repo_type="dataset", recursive=True
                )
            )
        except Exception as exc:
            print(f"WARN listing {variant_path}: {exc}", file=sys.stderr)
            continue
        variant = variant_path.rstrip("/").split("/")[-1]
        for entry in files:
            if type(entry).__name__ != "RepoFile":
                continue
            path = entry.path
            base = path.rsplit("/", 1)[-1].lower()
            if not (reject_pat.search(base) and inserted_pat.search(base)):
                continue
            if not path.endswith((".json", ".jsonl")):
                continue
            parent_reject_paths_scanned.append(path)
            try:
                local = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path)
            except Exception as exc:
                print(f"WARN downloading {path}: {exc}", file=sys.stderr)
                continue
            local_p = Path(local)
            rows = []
            if path.endswith(".jsonl"):
                with local_p.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                try:
                    with local_p.open() as f:
                        obj = json.load(f)
                    if isinstance(obj, list):
                        rows = obj
                    elif isinstance(obj, dict) and "rows" in obj:
                        rows = obj["rows"]
                    else:
                        rows = [obj]
                except json.JSONDecodeError:
                    continue

            for row in rows:
                text = (
                    row.get("story")
                    or row.get("text")
                    or row.get("completion")
                    or row.get("response")
                    or ""
                )
                answer = row.get("answer") or row.get("target") or row.get("gold") or ""
                if not text or not answer:
                    continue
                n_rejects_scanned += 1
                per_variant[variant]["n"] += 1
                if find_span_exact(text, answer):
                    n_recoverable_exact += 1
                    per_variant[variant]["recoverable_exact"] += 1
                if find_span_permissive(text, answer, max_edit=1):
                    n_recoverable_permissive += 1
                    per_variant[variant]["recoverable_permissive"] += 1

    if n_rejects_scanned == 0:
        print(
            "WARN: no matching inserted-reject files found — check parent-prefix + filename patterns",
            file=sys.stderr,
        )

    report = {
        "n_rejects_scanned": n_rejects_scanned,
        "n_recoverable_exact": n_recoverable_exact,
        "n_recoverable_permissive": n_recoverable_permissive,
        "recovery_fraction_permissive": (n_recoverable_permissive / n_rejects_scanned)
        if n_rejects_scanned
        else 0.0,
        "per_variant": dict(per_variant),
        "matcher_config": {"whitespace_tolerant": True, "max_edit_distance": 1},
        "parent_reject_paths_scanned": parent_reject_paths_scanned,
        "parent_repo": args.parent_repo,
        "parent_prefix": args.parent_prefix,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)

    print(
        f"audit_ii: scanned={n_rejects_scanned} recoverable_exact={n_recoverable_exact} recoverable_permissive={n_recoverable_permissive} recovery_fraction={report['recovery_fraction_permissive']:.3f}"
    )
    print(f"audit_ii: report -> {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task #2054 audit (ii): permissive span-locator sweep over parent #1345 inserted rejects"
    )
    p.add_argument("--parent-repo", default="superkaiba1/explore-persona-space-data")
    p.add_argument("--parent-prefix", default="issue1345_framing")
    p.add_argument("--output", default="eval_results/issue_2054/audits/audit_ii_span_locator.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run unit-test fixtures against the matchers; no HF network calls; exit 0 on pass.",
    )
    args = p.parse_args()

    if args.dry_run:
        return _dry_run()
    return run_audit(args)


if __name__ == "__main__":
    sys.exit(main())
