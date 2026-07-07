#!/usr/bin/env python3
"""Issue #1092 P5 judge phase: scored-row assembly + N=5 graded trait judging.

Off-pod, 0-GPU. Production calls the #779 graded judge wrapper over the hardened
Anthropic batch client; dry-run builds the exact item set, rubric-keyed cache
preimages, and B1/B2 eligibility skeleton without API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.eval.batch_judge import rubric_fingerprint  # noqa: E402
from issue779_common import (  # noqa: E402
    JUDGE_MODEL,
    JUDGE_N_DRAWS,
    JUDGE_TEMPERATURE,
    TRAITS,
    judge_rollouts_n5,
    parse_graded_trait_score,
    trait_judge_system_prompt,
    trait_judge_user_msg,
)


@dataclass(frozen=True)
class JudgeItem:
    item_id: str
    row_id: str
    cell_id: str
    trait: str
    question: str
    completion: str
    stratum: str
    grain: str
    arm: str
    rubric_key: str


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_store(path: Path, key: str) -> dict[str, dict]:
    return {str(item[key]): item for item in _jsonl(path)}


def _query_text(query_store: dict[str, dict], query_id: str) -> str:
    item = query_store.get(query_id)
    if item is None:
        raise KeyError(f"query_id {query_id!r} missing from query_store")
    text = item.get("text") or item.get("query")
    if not text:
        raise ValueError(f"query_id {query_id!r} has no text/query")
    return str(text)


def _load_completion_map(raw_dir: Path, model_type: str, cell_id: str) -> dict[str, str]:
    comp_dir = raw_dir / model_type
    paths = sorted(comp_dir.glob(f"{cell_id}_shard*.jsonl")) if comp_dir.exists() else []
    out: dict[str, str] = {}
    for path in paths:
        for item in _jsonl(path):
            row_id = item.get("row_id")
            if row_id:
                out[str(row_id)] = str(item.get("completion", ""))
    return out


def _rubric_key(trait: str, *, dry_run: bool) -> str:
    if dry_run:
        preimage = f"issue1092-dry-run::{JUDGE_MODEL}::{trait}::issue779_common"
        return hashlib.sha256(preimage.encode()).hexdigest()[:16]
    return rubric_fingerprint(
        JUDGE_MODEL,
        trait_judge_system_prompt(trait),
        trait_judge_user_msg,
    )


def _select_rows(rows: list[dict], *, floor_check_limit: int) -> list[dict]:
    selected: list[dict] = []
    natural_floor: list[dict] = []
    for row in rows:
        stratum = row.get("stratum", "")
        if stratum in {"dense_core", "battery", "trait_stratum"}:
            selected.append(row)
        elif stratum == "periphery_natural":
            natural_floor.append(row)
    selected.extend(natural_floor[:floor_check_limit])
    return selected


def build_judge_items(
    *,
    corpus_dir: Path,
    raw_completions_dir: Path,
    row_limit: int | None,
    dry_run: bool,
    floor_check_limit: int,
) -> list[JudgeItem]:
    rows = list(_jsonl(corpus_dir / "manifest.jsonl"))
    if row_limit is not None:
        rows = rows[:row_limit]
    query_store = _load_store(corpus_dir / "query_store.jsonl", "query_id")
    instruct = _load_completion_map(raw_completions_dir, "instruct", "cell_inst_own")
    pretrained = _load_completion_map(raw_completions_dir, "pretrained", "cell_pre_own")
    completion_maps = {
        "cell_inst_own": instruct,
        "cell_pre_own": pretrained,
    }

    selected = _select_rows(rows, floor_check_limit=floor_check_limit)
    items: list[JudgeItem] = []
    missing_completion: dict[str, int] = {k: 0 for k in completion_maps}
    for row in selected:
        stratum = row.get("stratum", "")
        row_traits = list(TRAITS)
        if stratum == "trait_stratum":
            row_trait = row.get("trait")
            row_traits = [str(row_trait)] if row_trait in TRAITS else []
        for cell_id, cmap in completion_maps.items():
            if cell_id == "cell_pre_own" and stratum not in {"battery", "trait_stratum"}:
                continue
            completion = cmap.get(str(row["row_id"]))
            if completion is None:
                missing_completion[cell_id] += 1
                continue
            question = _query_text(query_store, str(row["query_id"]))
            for trait in row_traits:
                rk = _rubric_key(trait, dry_run=dry_run)
                preimage = f"{row['row_id']}::{cell_id}::{trait}::{rk}"
                items.append(
                    JudgeItem(
                        item_id=hashlib.sha256(preimage.encode()).hexdigest()[:24],
                        row_id=str(row["row_id"]),
                        cell_id=cell_id,
                        trait=trait,
                        question=question,
                        completion=completion,
                        stratum=stratum,
                        grain="per_example",
                        arm=cell_id,
                        rubric_key=rk,
                    )
                )
    if any(missing_completion.values()):
        print(f"[judge] missing completions by cell: {missing_completion}", file=sys.stderr)
    return items


def _cache_key(item: JudgeItem) -> str:
    preimage = (
        f"issue1092-judge-v1\n{item.rubric_key}\n{JUDGE_MODEL}\n"
        f"{JUDGE_N_DRAWS}\n{JUDGE_TEMPERATURE}\n{item.question}\n{item.completion}"
    )
    return hashlib.sha256(preimage.encode()).hexdigest()


def _read_cache(cache_dir: Path, item: JudgeItem) -> dict | None:
    path = cache_dir / item.trait / f"{_cache_key(item)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _write_cache(cache_dir: Path, item: JudgeItem, payload: dict) -> None:
    path = cache_dir / item.trait / f"{_cache_key(item)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_judge(items: list[JudgeItem], out_dir: Path, *, dry_run: bool) -> list[dict]:
    cache_dir = out_dir / "cache"
    raw_dir = out_dir / "raw"
    scored: list[dict] = []
    todo_by_trait: dict[str, list[JudgeItem]] = {t: [] for t in TRAITS}
    for item in items:
        cached = _read_cache(cache_dir, item)
        if cached is not None:
            scored.append({**asdict(item), **cached, "cache_hit": True})
        else:
            todo_by_trait[item.trait].append(item)

    if dry_run:
        for trait_items in todo_by_trait.values():
            for item in trait_items:
                scored.append(
                    {
                        **asdict(item),
                        "score": None,
                        "n_valid_draws": 0,
                        "n_draws": JUDGE_N_DRAWS,
                        "dropped": False,
                        "cache_hit": False,
                        "dry_run": True,
                    }
                )
        return scored

    for trait, trait_items in todo_by_trait.items():
        if not trait_items:
            continue
        qmap: dict[str, list[str]] = {}
        order: list[tuple[int, int, JudgeItem]] = []
        q_index: dict[str, int] = {}
        for item in trait_items:
            if item.question not in qmap:
                q_index[item.question] = len(qmap)
                qmap[item.question] = []
            qi = q_index[item.question]
            ci = len(qmap[item.question])
            qmap[item.question].append(item.completion)
            order.append((qi, ci, item))
        raw_path = raw_dir / f"{trait}_raw.json"
        result = judge_rollouts_n5(
            trait,
            {"issue1092": qmap},
            raw_path,
            cache_dir=cache_dir / trait,
            n_draws=JUDGE_N_DRAWS,
            dry_run=False,
        )
        for qi, ci, item in order:
            cid = f"issue1092__{qi:05d}__{ci:02d}"
            score, n_valid, n_draws = result.get(cid, (None, 0, JUDGE_N_DRAWS))
            payload = {
                "score": score,
                "n_valid_draws": n_valid,
                "n_draws": n_draws,
                "dropped": score is None,
            }
            _write_cache(cache_dir, item, payload)
            scored.append({**asdict(item), **payload, "cache_hit": False})
    return scored


def _pearson_or_nan(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return math.nan
    mx, my = sum(x) / len(x), sum(y) / len(y)
    xc = [v - mx for v in x]
    yc = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in xc) * sum(v * v for v in yc))
    return math.nan if denom == 0 else sum(a * b for a, b in zip(xc, yc)) / denom


def eligibility_summary(scored: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in scored:
        groups.setdefault((row["trait"], row["arm"], row["grain"]), []).append(row)
    out: list[dict] = []
    for (trait, arm, grain), rows in sorted(groups.items()):
        vals = [float(r["score"]) for r in rows if r.get("score") is not None]
        positives = sum(v > 50.0 for v in vals)
        negatives = len(vals) - positives
        std = float("nan")
        if vals:
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        estimable = len(vals) >= 5 and (not math.isnan(std)) and std >= 1.0 and positives >= 1 and negatives >= 1
        fallback_vals = [
            float(r["score"])
            for r in rows
            if r.get("score") is not None and r.get("stratum") in {"trait_stratum", "battery"}
        ]
        out.append(
            {
                "trait": trait,
                "arm": arm,
                "grain": grain,
                "n_rows": len(rows),
                "n_scored": len(vals),
                "n_dropped": len(rows) - len(vals),
                "score_std": std,
                "n_positive": positives,
                "n_negative": negatives,
                "rate_positive": positives / len(vals) if vals and positives and negatives else None,
                "estimable": bool(estimable),
                "fallback_subset": "trait_stratum+battery",
                "fallback_n_scored": len(fallback_vals),
            }
        )
    return out


def write_outputs(scored: list[dict], out_dir: Path, *, dry_run: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / ("scores_dry_run.jsonl" if dry_run else "scores.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    eligibility = eligibility_summary(scored)
    drop_counts: dict[str, int] = {}
    for row in scored:
        if row.get("dropped"):
            drop_counts[row["arm"]] = drop_counts.get(row["arm"], 0) + 1
    summary = {
        "phase": "P5_judge",
        "dry_run": dry_run,
        "judge_model": JUDGE_MODEL,
        "n_draws": JUDGE_N_DRAWS,
        "temperature": JUDGE_TEMPERATURE,
        "traits": list(TRAITS),
        "n_items": len(scored),
        "drop_counts_by_arm": drop_counts,
        "scores_file": str(scores_path),
        "eligibility": eligibility,
    }
    summary_path = out_dir / ("summary_dry_run.json" if dry_run else "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--raw-completions-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--row-limit", type=int, default=None)
    p.add_argument("--floor-check-limit", type=int, default=500)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    items = build_judge_items(
        corpus_dir=args.corpus_dir,
        raw_completions_dir=args.raw_completions_dir,
        row_limit=args.row_limit,
        dry_run=args.dry_run,
        floor_check_limit=args.floor_check_limit,
    )
    scored = run_judge(items, args.out_dir, dry_run=args.dry_run)
    summary = write_outputs(scored, args.out_dir, dry_run=args.dry_run)
    print(
        f"[judge] artifact digest: items={summary['n_items']} traits={summary['traits']} "
        f"eligibility_groups={len(summary['eligibility'])} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
