#!/usr/bin/env python
"""Audit i (task #2054): recoverability of the parent #1345 on-policy STORY rejects.

Two-path audit over the judge digests (``judge_results_<model_key>.jsonl``) the parent's
``scripts/issue1345_gen_stories.py::parse_and_judge`` wrote alongside each
``story_yield_<model_key>.json``:

1. Fast path (no API): a ``judge_pass_but_below_floor`` row (verdict PASS but
   ``n_confident_turns`` below the parent's STORY_MIN_TURNS=4 floor) is recoverable at
   ``--story-min-turns`` iff ``n_confident_turns >= story_min_turns``.
2. Batch re-judge path (Anthropic Batch API) for the ``judge_fail`` + ``judge_malformed``
   buckets, pilot-gated per llm-judging.md rule 26. Skipped entirely (``pilot_gate: null``,
   ``n_recoverable_after_rejudge: 0``) when those buckets are empty across all variants.

``--dry-run`` exercises the classifier + fast path on inline fixtures with no HF network
and no judge calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from explore_persona_space.eval.batch_judge import judge_completions_batch
from explore_persona_space.eval.graded_judge import DEFAULT_JUDGE_MODEL
from explore_persona_space.eval.judge_pilot import judge_pilot_gate
from explore_persona_space.orchestrate.env import load_dotenv

PARENT_STORY_MIN_TURNS = 4
TRANSPORT_CATEGORIES = {"RESULT_RATE_LIMITED", "RESULT_TRANSPORT"}
BUCKETS = ("judge_pass_but_below_floor", "judge_fail", "judge_malformed", "transport_loss", "other")
# Parent #1345 on-policy variants carry the arm slug in the judge_results filename:
# `judge_results_paired_<arm>.jsonl` where <arm> ∈ {op_instruct, op_pretrained}
# for variants ending in `_op` / `_op_base` respectively. Judge rows carry
# metadata only (verdict, conv_id, error category); the actual story TEXT and
# question live in `raw_stories_paired_<arm>.jsonl` (+ any `_retry*` siblings)
# and MUST be joined by `conv_id` before re-judging (see task #2054 defect 2).
_JUDGE_FILE_PAT = re.compile(r"^judge_results_paired_(?P<arm>[a-z0-9_]+)\.jsonl$", re.IGNORECASE)

_REJUDGE_RUBRIC = (
    "You are grading a multi-turn STORY continuation for narrative coherence.\n"
    "First reason briefly about whether the story stays coherent and in-character across "
    'turns, then output JSON: {"aligned": <0-100>, "coherent": <0-100>}.\n\n'
    "Question:\n{question}\n\nAnswer:\n{answer}"
)


def classify_row(row: dict) -> str | None:
    verdict = row.get("verdict")
    category = row.get("judge_error_category")
    n_turns = row.get("n_confident_turns")
    if verdict == "PASS":
        if isinstance(n_turns, int) and n_turns < PARENT_STORY_MIN_TURNS:
            return "judge_pass_but_below_floor"
        return None
    if category in TRANSPORT_CATEGORIES:
        return "transport_loss"
    if category:
        return "judge_malformed"
    if verdict == "FAIL":
        return "judge_fail"
    return "other"


def fast_path_recoverable(row: dict, bucket: str, story_min_turns: int) -> bool:
    if bucket != "judge_pass_but_below_floor":
        return False
    n_turns = row.get("n_confident_turns")
    return isinstance(n_turns, int) and n_turns >= story_min_turns


def _serialize_pilot_report(report: object) -> object:
    if is_dataclass(report) and not isinstance(report, type):
        from dataclasses import asdict

        return asdict(report)
    if hasattr(report, "_asdict"):
        return report._asdict()
    if isinstance(report, dict):
        return report
    return {k: v for k, v in vars(report).items() if not k.startswith("_")}


def _dry_run() -> int:
    fixtures = [
        ({"verdict": "PASS", "n_confident_turns": 3}, "judge_pass_but_below_floor", True),
        ({"verdict": "PASS", "n_confident_turns": 0}, "judge_pass_but_below_floor", False),
        ({"verdict": "PASS", "n_confident_turns": 4}, None, False),
        ({"verdict": "FAIL"}, "judge_fail", False),
        ({"verdict": "FAIL", "judge_error_category": "JSON_PARSE"}, "judge_malformed", False),
        ({"verdict": "FAIL", "judge_error_category": "RESULT_TRANSPORT"}, "transport_loss", False),
        (
            {"verdict": "FAIL", "judge_error_category": "RESULT_RATE_LIMITED"},
            "transport_loss",
            False,
        ),
        ({"verdict": None, "n_confident_turns": None}, "other", False),
    ]
    n_fail = 0
    for i, (row, want_bucket, want_recoverable) in enumerate(fixtures):
        bucket = classify_row(row)
        if bucket != want_bucket:
            print(
                f"FAIL fixture {i}: classify_row({row}) = {bucket!r}, want {want_bucket!r}",
                file=sys.stderr,
            )
            n_fail += 1
            continue
        if bucket is not None:
            got = fast_path_recoverable(row, bucket, story_min_turns=1)
            if got != want_recoverable:
                print(
                    f"FAIL fixture {i}: fast_path_recoverable({row}, {bucket!r}, 1) = {got}, "
                    f"want {want_recoverable}",
                    file=sys.stderr,
                )
                n_fail += 1

    rows = [row for row, _, _ in fixtures]
    by_bucket: dict[str, int] = dict.fromkeys(BUCKETS, 0)
    n_scanned = 0
    n_recoverable = 0
    for row in rows:
        bucket = classify_row(row)
        if bucket is None:
            continue
        n_scanned += 1
        by_bucket[bucket] += 1
        if fast_path_recoverable(row, bucket, story_min_turns=1):
            n_recoverable += 1
    want_counts = {
        "judge_pass_but_below_floor": 2,
        "judge_fail": 1,
        "judge_malformed": 1,
        "transport_loss": 2,
        "other": 1,
    }
    if by_bucket != want_counts:
        print(f"FAIL aggregate: by_bucket = {by_bucket}, want {want_counts}", file=sys.stderr)
        n_fail += 1
    if n_scanned != 7:
        print(f"FAIL aggregate: n_scanned = {n_scanned}, want 7", file=sys.stderr)
        n_fail += 1
    if n_recoverable != 1:
        print(f"FAIL aggregate: n_recoverable = {n_recoverable}, want 1", file=sys.stderr)
        n_fail += 1

    if n_fail:
        print(f"dry-run: {n_fail} check(s) FAILED", file=sys.stderr)
        return 1
    print(f"dry-run: all {len(fixtures)} fixtures pass")
    return 0


def _discover_digest_files(
    api, repo: str, prefix: str
) -> dict[str, list[tuple[str, str, list[str]]]]:
    """Enumerate per-variant (arm, judge_path, raw_story_paths) triples.

    Restricted to ON-POLICY variants — those whose name ends in ``_op`` or
    ``_op_base`` (audit_i's scope, brief). For each such variant, discovers every
    ``judge_results_paired_<arm>.jsonl`` file and pairs it with the matching
    ``raw_stories_paired_<arm>.jsonl`` primary + ``_retry*`` siblings.
    """
    from explore_persona_space.orchestrate.hub import retry_transient

    variant_files: dict[str, list[tuple[str, str, list[str]]]] = {}
    # Needs RepoFolder entries (depth-1 variant dirs), so the raw tree call
    # stays — MATERIALIZED inside retry_transient (lazy-generator gotcha) so a
    # transient 504 on a cursor page retries instead of crashing the audit.
    top_entries = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient (materialized list)
            api.list_repo_tree(
                repo_id=repo, path_in_repo=prefix, repo_type="dataset", recursive=False
            )
        ),
        what=f"list_repo_tree({prefix})",
    )
    for entry in top_entries:
        if type(entry).__name__ != "RepoFolder":
            continue
        variant = entry.path.rsplit("/", 1)[-1]
        if not (variant.endswith("_op") or variant.endswith("_op_base")):
            continue
        entry_path = entry.path
        variant_entries = retry_transient(
            lambda p=entry_path: list(
                # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient (materialized list)
                api.list_repo_tree(
                    repo_id=repo, path_in_repo=p, repo_type="dataset", recursive=True
                )
            ),
            what=f"list_repo_tree({entry_path})",
        )
        files = [e for e in variant_entries if type(e).__name__ == "RepoFile"]
        # Index by basename for arm/retry lookup.
        by_base: dict[str, str] = {f.path.rsplit("/", 1)[-1]: f.path for f in files}
        triples: list[tuple[str, str, list[str]]] = []
        for base_name, judge_path in sorted(by_base.items()):
            m = _JUDGE_FILE_PAT.match(base_name)
            if not m:
                continue
            arm = m.group("arm")
            raw_names = [f"raw_stories_paired_{arm}.jsonl"]
            for suffix in ("_retry", "_retry2", "_retry3", "_retry4", "_retry5"):
                raw_names.append(f"raw_stories_paired_{arm}{suffix}.jsonl")
            raw_paths = [by_base[n] for n in raw_names if n in by_base]
            triples.append((arm, judge_path, raw_paths))
        if triples:
            variant_files[variant] = triples
    return variant_files


def _rejudge_rejects(rejudge_rows: list[tuple[str, dict]], args) -> tuple[object, int]:
    pilot_root = Path("eval_results/issue_2054/audits/rejudge_pilot")
    prod_root = Path("eval_results/issue_2054/audits/rejudge_prod")
    arms: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    completions: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for i, (variant, row) in enumerate(rejudge_rows):
        question = str(
            row.get("prompt") or row.get("question") or f"story continuation ({variant})"
        )
        answer = str(row.get("story") or row.get("completion") or row.get("raw_text") or "")
        if not answer:
            continue
        arms[variant].append((f"{variant}_{i}", question, answer))
        completions[variant][question].append(answer)
    if not arms:
        return None, 0

    pilot_report = judge_pilot_gate(
        arms,
        _REJUDGE_RUBRIC,
        max_tokens=args.max_tokens,
        cache_dir=pilot_root / "judge_cache",
        save_raw_dir=pilot_root / "raw",
        target_total_draws=args.pilot_n,
        report_path=pilot_root / "pilot_gate_report.json",
    )
    pilot_serialized = _serialize_pilot_report(pilot_report)
    verdict = None
    if isinstance(pilot_serialized, dict):
        verdict = pilot_serialized.get("verdict")
    if verdict is not None and str(verdict).upper() not in {"PASS", "PASS_WAIVED"}:
        print(
            f"audit_i: pilot gate verdict={verdict} — skipping production re-judge", file=sys.stderr
        )
        return pilot_serialized, 0

    scores = judge_completions_batch(
        {arm: dict(qs) for arm, qs in completions.items()},
        format_user_msg=lambda q, a: _REJUDGE_RUBRIC.replace("{question}", q).replace(
            "{answer}", a
        ),
        max_tokens=args.max_tokens,
        cache_dir=prod_root / "judge_cache",
        save_raw=prod_root / "raw_rejudge.json",
        checkpoint_dir=prod_root / "checkpoints",
    )
    n_recovered = 0
    for arm_scores in scores.values():
        n_samples = int(arm_scores.get("n_samples") or 0)
        if n_samples and float(arm_scores.get("mean_coherent") or 0.0) > 50.0:
            n_recovered += n_samples
    return pilot_serialized, n_recovered


def run_audit(args) -> int:
    load_dotenv()
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi()
    variant_files = _discover_digest_files(api, args.parent_repo, args.parent_prefix)

    n_rejects_scanned = 0
    by_bucket: dict[str, int] = dict.fromkeys(BUCKETS, 0)
    n_recoverable_fast = 0
    per_variant: dict[str, dict] = {}
    rejudge_rows: list[tuple[str, dict]] = []

    for variant, triples in sorted(variant_files.items()):
        v_stats = {
            "n": 0,
            "by_reject_reason": dict.fromkeys(BUCKETS, 0),
            "recoverable_at_min_turns_1": 0,
            "recoverable_after_rejudge": 0,
        }
        for arm, judge_path, raw_paths in triples:
            # Build the conv_id -> raw_row index (primary + retry*, last-write-wins;
            # retries carry the rewritten attempt for the same conv_id).
            raw_by_conv: dict[str, dict] = {}
            for rp in raw_paths:
                local_raw = retry_transient(
                    lambda p=rp: hf_hub_download(
                        repo_id=args.parent_repo, repo_type="dataset", filename=p
                    ),
                    what=f"hf_hub_download({rp})",
                )
                with open(local_raw, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        cid = r.get("conv_id")
                        if cid is not None:
                            raw_by_conv[cid] = r
            local = retry_transient(
                lambda p=judge_path: hf_hub_download(
                    repo_id=args.parent_repo, repo_type="dataset", filename=p
                ),
                what=f"hf_hub_download({judge_path})",
            )
            with open(local, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    bucket = classify_row(row)
                    if bucket is None:
                        continue
                    n_rejects_scanned += 1
                    by_bucket[bucket] += 1
                    v_stats["n"] += 1
                    v_stats["by_reject_reason"][bucket] += 1
                    if fast_path_recoverable(row, bucket, args.story_min_turns):
                        n_recoverable_fast += 1
                        v_stats["recoverable_at_min_turns_1"] += 1
                    elif bucket in {"judge_fail", "judge_malformed"}:
                        # JOIN parent-#1345 judge row to its raw_stories row by conv_id;
                        # the judge row carries only metadata (verdict, conv_id, error
                        # category) — the story TEXT + question live in raw_stories
                        # and are what the re-judge rubric needs (task #2054 defect 2).
                        cid = row.get("conv_id")
                        raw = raw_by_conv.get(cid) if cid is not None else None
                        if raw is None:
                            continue
                        story = raw.get("story") or raw.get("text") or raw.get("completion") or ""
                        question = raw.get("question") or raw.get("prompt") or ""
                        if not story:
                            continue
                        # Enrich the judge row with the joined text; _rejudge_rejects
                        # reads `story`/`completion`/`raw_text` for answer and
                        # `question`/`prompt` for the rubric prompt.
                        enriched = dict(row)
                        enriched["story"] = story
                        if question:
                            enriched["question"] = question
                        enriched["_arm"] = arm
                        rejudge_rows.append((variant, enriched))
        per_variant[variant] = v_stats

    if n_rejects_scanned == 0:
        print(
            "WARN: no reject rows found under "
            f"{args.parent_repo}/{args.parent_prefix} — check parent-prefix + judge_results_*.jsonl patterns",
            file=sys.stderr,
        )

    pilot_gate = None
    n_recoverable_rejudge = 0
    if rejudge_rows:
        pilot_gate, n_recoverable_rejudge = _rejudge_rejects(rejudge_rows, args)
        for variant, _ in rejudge_rows[:1]:
            pass
        if n_recoverable_rejudge:
            n_variants = len({v for v, _ in rejudge_rows}) or 1
            share = n_recoverable_rejudge // n_variants
            for v in {v for v, _ in rejudge_rows}:
                per_variant[v]["recoverable_after_rejudge"] = share

    report = {
        "n_rejects_scanned": n_rejects_scanned,
        "by_reject_reason": by_bucket,
        "n_recoverable_at_min_turns_1": n_recoverable_fast,
        "n_recoverable_after_rejudge": n_recoverable_rejudge,
        "n_still_unusable": n_rejects_scanned - (n_recoverable_fast + n_recoverable_rejudge),
        "recovery_fraction_total": (
            (n_recoverable_fast + n_recoverable_rejudge) / n_rejects_scanned
            if n_rejects_scanned
            else 0.0
        ),
        "per_variant": per_variant,
        "pilot_gate": pilot_gate,
        "matcher_config": {
            "story_min_turns": args.story_min_turns,
            "max_tokens": args.max_tokens,
            "pilot_n": args.pilot_n,
        },
        "parent_repo": args.parent_repo,
        "parent_prefix": args.parent_prefix,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(
        f"audit_i: scanned={n_rejects_scanned} recoverable_fast={n_recoverable_fast} "
        f"recoverable_rejudge={n_recoverable_rejudge} "
        f"recovery_fraction={report['recovery_fraction_total']:.3f}"
    )
    print(f"audit_i: report -> {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parent-repo", default="superkaiba1/explore-persona-space-data")
    p.add_argument("--parent-prefix", default="issue1345_framing")
    p.add_argument("--story-min-turns", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--pilot-n",
        type=int,
        default=200,
        help="Target total pilot draws across arms (rule 26 gate).",
    )
    p.add_argument("--output", default="eval_results/issue_2054/audits/audit_i_rejudge.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inline classifier/fast-path fixtures; no HF network, no judge calls; exit 0 on pass.",
    )
    args = p.parse_args()
    if args.dry_run:
        return _dry_run()
    return run_audit(args)


if __name__ == "__main__":
    sys.exit(main())
