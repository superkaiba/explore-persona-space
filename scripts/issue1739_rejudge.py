"""Truncation-recovery re-judge for issue #1739 trait behaviors (rule 23).

The first judge wave ran at JUDGE_MAX_TOKENS=400 and content-dropped a
truncation-shaped parse-error tail (evil 1,904 draws / 1.2%; sycophancy
14,027 / 5.4% — reason-then-score rationales cut before the {"score": N}
JSON). Per llm-judging.md rule 23 this censoring conditions on rationale
length, so affected items are re-judged WHOLE (all N_JUDGE_DRAWS fresh
draws, one uniform instrument per item) at a larger --max-tokens against a
FRESH cache, and the per-item entries + DV dataset are rebuilt in place
(pre-rejudge scores file kept as *.pre_rejudge.json).

Scope: trait path only (evil / sycophancy). Hallucination's 0.1% stays a
reported caveat. CONTENT HYGIENE: logs carry ids + counts, never text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_1739 import dv_build, judging  # noqa: E402
from explore_persona_space.experiments.issue_1739.constants import (  # noqa: E402
    JUDGE_TEMPERATURE,
    N_JUDGE_DRAWS,
)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _affected_items(raw_path: Path) -> set[str]:
    """Item ids with >=1 parse-error draw (error dicts in all_scores)."""
    raw = json.loads(raw_path.read_text())
    affected: set[str] = set()
    for custom_id, v in raw["all_scores"].items():
        if isinstance(v, dict) and v.get("error"):
            affected.add(custom_id.split("__")[0])
    return affected


def _merged_content_drops(payload: dict) -> int:
    per_item = payload["per_item_scores"]
    transport = payload.get("per_item_transport_losses", {}) or {}
    return sum(
        N_JUDGE_DRAWS - len(kept) - int(transport.get(item, 0)) for item, kept in per_item.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", choices=("evil", "sycophancy"), required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="the behavior's existing judge out-dir")
    parser.add_argument("--dv-out-root", required=True)
    parser.add_argument("--inputs-dir", default="data/issue_1739/inputs")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int, default=None, help="smoke cap on affected items")
    parser.add_argument("--threshold-base", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    scores_path = out_dir / "labeling_scores.json"
    raw_path = out_dir / "judge_raw_trait.json"
    payload = json.loads(scores_path.read_text())
    if payload.get("judge_max_tokens", 0) >= args.max_tokens and payload.get("rejudge_800"):
        print(f"[rejudge] {args.behavior}: already re-judged; nothing to do")
        return 0

    affected = _affected_items(raw_path)
    if args.limit:
        affected = set(sorted(affected)[: args.limit])
    print(f"[rejudge] {args.behavior}: {len(affected)} items with parse-error draws")
    if not affected:
        return 0

    # Rollout payloads: items for the affected set + contexts_meta for ALL rows
    # (the DV rebuild attaches split/rung/group_key meta exactly as the
    # original run did).
    items: list[tuple[str, str, str]] = []
    contexts_meta: dict[str, dict] = {}
    n_files = 0
    for p in sorted(Path(args.rollout_dir).glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text())
        n_files += 1
        contexts_meta[d["context_id"]] = d
        item_id = judging.rollout_item_id(d["context_id"], int(d["rollout_k"]))
        if item_id in affected:
            items.append((item_id, d["query"], d["completion"]))
    print(f"[rejudge] scanned {n_files} rollout files; matched {len(items)} affected items")
    missing = affected - {i[0] for i in items}
    if missing:
        raise SystemExit(f"[rejudge] {len(missing)} affected items missing from rollout dir")

    rubric = judging.load_trait_rubric(args.behavior, inputs_dir=args.inputs_dir)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    result = judging.judge_items_graded(
        items,
        rubric,
        cache_dir=out_dir / f"rejudge_cache_{ts}",
        save_raw=out_dir / f"judge_raw_trait_rejudge{args.max_tokens}_{ts}.json",
        n_draws=N_JUDGE_DRAWS,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=args.max_tokens,
        threshold_base=args.threshold_base,
    )

    # Whole-item replace (uniform instrument per item).
    n_recovered = 0
    for item_id, _q, _a in items:
        old = payload["scores"].get(item_id)
        payload["scores"][item_id] = result.scores.get(item_id)
        payload["per_item_scores"][item_id] = result.per_item_scores.get(item_id, [])
        payload.setdefault("per_item_transport_losses", {})[item_id] = int(
            result.per_item_transport_losses.get(item_id, 0)
        )
        if old is None and result.scores.get(item_id) is not None:
            n_recovered += 1
    payload["n_content_dropped_draws"] = _merged_content_drops(payload)
    payload["n_transport_lost_draws"] = sum(
        int(v) for v in payload.get("per_item_transport_losses", {}).values()
    )
    payload["rejudge_800"] = {
        "n_affected_items": len(items),
        "n_items_recovered_from_all_dropped": n_recovered,
        "rejudge_max_tokens": args.max_tokens,
        "base_max_tokens": payload.get("judge_max_tokens"),
        "ts": ts,
        "git_commit": _git_commit(),
    }

    backup = scores_path.with_name("labeling_scores.pre_rejudge.json")
    if not backup.exists():
        shutil.copy2(scores_path, backup)

    dv_rows = dv_build.build_labeling_dv(
        dict(payload["scores"]),
        n_draws=N_JUDGE_DRAWS,
        per_item_transport_losses=dict(payload.get("per_item_transport_losses", {})),
        contexts_meta=contexts_meta,
    )
    for row in dv_rows:
        meta = contexts_meta.get(row["context_id"], {})
        for key in ("behavior", "split", "rung", "group_key"):
            if key in meta:
                row.setdefault(key, meta[key])
    dv_path = dv_build.write_dv_dataset(
        dv_rows,
        out_root=args.dv_out_root,
        behavior=args.behavior,
        judge_payload_meta={
            "n_rollout_files": payload.get("n_rollout_files"),
            "n_draws": N_JUDGE_DRAWS,
            "judge_temperature": JUDGE_TEMPERATURE,
            "judge_max_tokens": payload.get("judge_max_tokens"),
            "judge_model": payload.get("judge_model"),
            "rubric": payload.get("rubric"),
            "rejudge_800": payload["rejudge_800"],
        },
        git_commit=_git_commit(),
    )

    tmp = scores_path.with_name(scores_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, scores_path)
    print(
        json.dumps(
            {
                "behavior": args.behavior,
                "n_affected_items": len(items),
                "n_items_recovered_from_all_dropped": n_recovered,
                "dv_dataset_path": str(dv_path),
            },
            indent=2,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
