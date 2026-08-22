"""Re-score #2162's banked persona rollouts under HOLISTIC and PER-CONJUNCT rubrics.

Why this exists
---------------
#2162's `persona_prompted` cell reads flat (F_beh 0.151 vs a shuffled-donor null of
0.182). The inline diagnostic on that task (`epm:progress` v37) traced part of the
flatness to the RUBRIC rather than the intervention: the production descriptor is a
four-way conjunction whose last two clauses are discrete events --

    "the persona and register of Reginald, a formal Victorian butler: elaborate
     courtesy, old-fashioned formality, addressing the user as 'sir or madam', and
     small references to household duties"

-- so a response that visibly moves register (a neutral numbered list becomes a
flowery essay opening "Ah, the ...", matching the ceiling's own opening move) but
emits neither marker scores near zero. The pirate escapes this because its lead
conjunct (thick dialect) is pervasive across every clause.

This script tests that explanation on the EXISTING rollouts. No generation, no pod,
no GPU: it re-judges banked text under

  (a) HOLISTIC descriptors -- one property, no conjunction, per persona value; and
  (b) PER-CONJUNCT descriptors -- each clause of the production descriptor scored
      on its own, for the butler (4 clauses) and the pirate (3, as the control).

If the holistic butler read recovers while the two event-clauses stay at zero, the
flat persona cell is a measurement artifact and the replacement round
(`epm:followup-scope v1`, label `persona-specificity-ladder`) inherits a validated
rubric design. If the holistic read is flat too, the transfer really is absent and
the ladder round's rubric change is not the fix.

Nothing here mutates task lifecycle state. It reads banked artifacts, calls the
production judge, and writes `eval_results/issue_2162/persona_rubric_rescore/`.

Instrument parity with the parent (deliberate, so the reads are comparable): same
judge model, same `max_tokens`, same reason-then-score wrapper, same rubric-keyed
cache partitioning, N=1 draw per rubric per rollout. The ONLY thing that changes is
the descriptor text.

Usage
-----
    uv run python scripts/issue2162_persona_rubric_rescore.py --pilot
    uv run python scripts/issue2162_persona_rubric_rescore.py
    uv run python scripts/issue2162_persona_rubric_rescore.py --dry-run
"""

from __future__ import annotations

# load_dotenv BEFORE any heavy/HF import (lint: --check-dotenv-before-hf-import)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import statistics as stats  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    list_hf_files_under_path,
    retry_transient,
)
from explore_persona_space.task_workflow import repo_root  # noqa: E402

logger = logging.getLogger("issue2162_rescore")

DATASET_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2162_ctxinfo"
CELL = "persona_prompted"
SLOT = "ce"
ARMS = ("steered", "shuffled", "crosstype")

JUDGE_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_MAX_TOKENS = 1024  # rule-23 single-rationale floor, parent parity
N_DRAWS = 1  # parent parity (#2162 judging constants)

VALUE_LABEL = {"v1": "pirate", "v2": "plain", "v3": "butler"}

REASON_THEN_SCORE = (
    "First give a brief reasoning (1-3 sentences), then output a single JSON "
    'object of the form {"reasoning": "<your brief justification>", '
    '"score": <an integer from 0 to 100>} and nothing else after it.'
)

# ── rubrics ───────────────────────────────────────────────────────────
# HOLISTIC: the persona named as ONE property. No clause list, so a partial
# transfer lands mid-scale instead of failing an unmet conjunct.
HOLISTIC = {
    "v1": "the persona and register of a superstitious old pirate captain",
    "v2": "the plain register of a default assistant, with no persona of any kind",
    "v3": "the persona and register of a formal Victorian butler",
}

# PER-CONJUNCT: each clause of the PRODUCTION descriptor, scored alone. Clause
# text is lifted from `bank2162.value_descriptor` so the union of these is the
# production instrument, split.
CONJUNCTS = {
    "v3": {
        "courtesy": "elaborate courtesy",
        "formality": "old-fashioned formality",
        "address": "addressing the user as 'sir or madam'",
        "household": "small references to household duties",
    },
    "v1": {
        "dialect": "thick pirate dialect",
        "sea": "topics constantly related back to the sea and shipboard life",
        "warning": "an answer that ends with a grim warning about the ocean",
    },
}

# Conjunct scoring is restricted to the rollouts where the split can be read
# against its own floor and ceiling: the butler-target direction plus the two
# anchor sets that bracket it.
CONJUNCT_DIRECTIONS = {("v2", "v3")}
CONJUNCT_ANCHOR_VALUES = {"v2", "v3"}


def eval_prompt(descriptor: str) -> str:
    """Production instrument wrapper, byte-identical to the parent's."""
    return (
        f"To what extent does the response express {descriptor}? Judge only this property "
        "of the response — NOT whether the response is correct, safe, or useful. "
        "Score 0-100: 0 = no trace of it; 50 = partial or mixed expression; 100 = the "
        f"response fully and consistently expresses it.\n\n[RESPONSE]\n{{answer}}\n"
        f"[/RESPONSE]\n\n{REASON_THEN_SCORE}"
    )


@dataclass(frozen=True)
class Unit:
    item_id: str
    rubric_id: str
    answer: str
    source: dict


def _download(path_in_repo: str) -> Path:
    """One shard, through the sanctioned transient-retry wrapper (#1547)."""
    return Path(
        retry_transient(
            lambda: hf_hub_download(DATASET_REPO, path_in_repo, repo_type="dataset"),
            what=f"rollout shard {path_in_repo}",
        )
    )


def _read_jsonl(path: Path) -> list[dict]:
    """Line-iterate a JSONL file.

    Deliberately NOT ``read_text().splitlines()``: that splits on raw
    U+2028/U+2029/NEL inside ``ensure_ascii=False`` JSON strings and shreds
    valid records (#825/#950). Rollout text is model output, so those code
    points are reachable.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_grid_rows() -> list[dict]:
    """The 3 x 180 persona rollouts at context-end (steered + both nulls)."""
    rows: list[dict] = []
    for arm in ARMS:
        p = _download(f"{HF_PREFIX}/raw_completions/grid/shard_{CELL}__{SLOT}__{arm}.jsonl")
        shard = _read_jsonl(p)
        rows.extend(shard)
        logger.info("grid arm %-10s %4d rows", arm, len(shard))
    if not rows:
        raise RuntimeError("no grid rows loaded — refusing to judge an empty set")
    return rows


def load_anchor_rows() -> list[dict]:
    """Unpatched floor/ceiling draws for every persona_prompted context."""
    api = HfApi()
    paths = [
        p
        for p in list_hf_files_under_path(
            api,
            DATASET_REPO,
            f"{HF_PREFIX}/raw_completions/anchors",
            repo_type="dataset",
        )
        if p.endswith(".jsonl")
    ]
    rows: list[dict] = []
    for path in paths:
        for r in _read_jsonl(_download(path)):
            if str(r.get("context_id", "")).startswith(f"{CELL}::"):
                rows.append(r)
    if not rows:
        raise RuntimeError("no persona anchor rows found — refusing to proceed")
    logger.info("anchors %d rows over %d shards", len(rows), len(paths))
    return rows


def _uid(*parts: str) -> str:
    import hashlib

    return "r" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def build_units(grid: list[dict], anchors: list[dict]) -> dict[str, list[Unit]]:
    """rubric_id -> units. One wave per rubric (parent convention: rule 8)."""
    by_rubric: dict[str, list[Unit]] = defaultdict(list)

    for r in grid:
        va, vb = r["value_a"], r["value_b"]
        key = f"grid|{r['arm']}|{r['pair_id']}|{r['draw']}"
        src = {
            "kind": "grid",
            "arm": r["arm"],
            "pair_id": r["pair_id"],
            "carrier": r["carrier"],
            "value_a": va,
            "value_b": vb,
            "draw": r["draw"],
        }
        for v in (va, vb):  # holistic A and holistic B -> the contrast
            rid = f"hol-{v}"
            by_rubric[rid].append(Unit(_uid(key, rid), rid, r["text"], src))
        if (va, vb) in CONJUNCT_DIRECTIONS:
            for v in (va, vb):
                for name, clause in CONJUNCTS.get(v, {}).items():
                    rid = f"cj-{v}-{name}"
                    by_rubric[rid].append(Unit(_uid(key, rid), rid, r["text"], src))

    for r in anchors:
        v = r["context_id"].split("::")[1]
        key = f"anchor|{r['context_id']}|{r['draw']}"
        src = {"kind": "anchor", "context_id": r["context_id"], "value": v, "draw": r["draw"]}
        for hv in HOLISTIC:  # every anchor under all three -> floors and ceilings
            rid = f"hol-{hv}"
            by_rubric[rid].append(Unit(_uid(key, rid), rid, r["text"], src))
        if v in CONJUNCT_ANCHOR_VALUES:
            for cv in CONJUNCT_ANCHOR_VALUES:
                for name, clause in CONJUNCTS.get(cv, {}).items():
                    rid = f"cj-{cv}-{name}"
                    by_rubric[rid].append(Unit(_uid(key, rid), rid, r["text"], src))

    return dict(by_rubric)


def descriptor_for(rubric_id: str) -> str:
    if rubric_id.startswith("hol-"):
        return HOLISTIC[rubric_id[4:]]
    m = re.fullmatch(r"cj-(v\d)-([a-z]+)", rubric_id)
    if not m:
        raise ValueError(f"unknown rubric id {rubric_id}")
    return CONJUNCTS[m.group(1)][m.group(2)]


def run(out_dir: Path, by_rubric: dict[str, list[Unit]], dry_run: bool) -> dict:
    scores_path = out_dir / "scores.jsonl"
    cache_root = out_dir / ".cache"
    raw_dir = out_dir / "raw"
    for d in (cache_root, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_scored: list[dict] = []
    telemetry: dict[str, dict] = {}
    for rid in sorted(by_rubric):
        units = by_rubric[rid]
        ids = {u.item_id for u in units}
        if len(ids) != len(units):
            raise RuntimeError(f"rubric {rid}: duplicate item ids ({len(units) - len(ids)})")
        logger.info("[wave %s] %d items dispatching", rid, len(units))
        result = judge_graded(
            [(u.item_id, "", u.answer) for u in units],
            eval_prompt(descriptor_for(rid)),
            n_draws=N_DRAWS,
            cache_dir=cache_root / rid,
            save_raw=raw_dir / f"{rid}.json",
            judge_model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            dry_run=dry_run,
        )
        if dry_run:
            continue
        lost = {i for i, n in result.per_item_transport_losses.items() if n > 0}
        if lost:  # rule 24: transport losses are retried, never persisted as drops
            logger.info("[wave %s] transport retry: %d items", rid, len(lost))
            retry = [u for u in units if u.item_id in lost]
            r2 = judge_graded(
                [(u.item_id, "", u.answer) for u in retry],
                eval_prompt(descriptor_for(rid)),
                n_draws=N_DRAWS,
                cache_dir=cache_root / rid,
                save_raw=raw_dir / f"{rid}.retry1.json",
                judge_model=JUDGE_MODEL,
                max_tokens=JUDGE_MAX_TOKENS,
            )
            for iid in lost:
                result.scores[iid] = r2.scores.get(iid)
        n_drop = sum(1 for u in units if result.scores.get(u.item_id) is None)
        telemetry[rid] = {
            "n_items": len(units),
            "n_scored": len(units) - n_drop,
            "n_dropped": n_drop,
            "drop_frac": round(n_drop / len(units), 4),
        }
        for u in units:
            s = result.scores.get(u.item_id)
            if s is None:
                continue  # drop-never-coerce
            all_scored.append({"item_id": u.item_id, "rubric_id": rid, "score": s, **u.source})

    if dry_run:
        return {"dry_run": True}
    scores_path.write_text("".join(json.dumps(r) + "\n" for r in all_scored), encoding="utf-8")
    return {"telemetry": telemetry, "n_scores": len(all_scored)}


def summarize(out_dir: Path) -> dict:
    """Holistic F per direction/arm, and the per-conjunct split for the butler."""
    rows = _read_jsonl(out_dir / "scores.jsonl")
    anchors = [r for r in rows if r["kind"] == "anchor"]
    grid = [r for r in rows if r["kind"] == "grid"]

    # holistic anchor means: value -> rubric_value -> mean score/100
    anch: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in anchors:
        if r["rubric_id"].startswith("hol-"):
            anch[(r["value"], r["rubric_id"][4:])].append(r["score"] / 100.0)
    amean = {k: stats.mean(v) for k, v in anch.items()}

    out: dict = {"holistic": {}, "conjuncts": {}}
    by_cell: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in grid:
        if not r["rubric_id"].startswith("hol-"):
            continue
        by_cell[(r["value_a"], r["value_b"], r["arm"])][r["rubric_id"][4:]].append(
            r["score"] / 100.0
        )
    for (va, vb, arm), d in sorted(by_cell.items()):
        if va not in d or vb not in d:
            continue
        delta = stats.mean(d[vb]) - stats.mean(d[va])
        floor = amean.get((va, vb), 0.0) - amean.get((va, va), 0.0)
        ceil = amean.get((vb, vb), 0.0) - amean.get((vb, va), 0.0)
        denom = ceil - floor
        out["holistic"][f"{VALUE_LABEL[va]}->{VALUE_LABEL[vb]}|{arm}"] = {
            "delta_patched": round(delta, 4),
            "delta_floor": round(floor, 4),
            "delta_ceiling": round(ceil, 4),
            "F_holistic": round((delta - floor) / denom, 4) if abs(denom) > 1e-6 else None,
            "n_rollouts": len(d[vb]),
        }

    cj: dict[str, list[float]] = defaultdict(list)
    for r in grid + anchors:
        if r["rubric_id"].startswith("cj-"):
            tag = r["arm"] if r["kind"] == "grid" else f"anchor-{r['value']}"
            cj[f"{r['rubric_id']}|{tag}"].append(r["score"])
    out["conjuncts"] = {k: round(stats.mean(v), 2) for k, v in sorted(cj.items())}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true", help="rule-26 shape probe: 100 items, 1 wave")
    ap.add_argument("--dry-run", action="store_true", help="route + count, no API calls")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    out_dir = repo_root() / "eval_results" / "issue_2162" / "persona_rubric_rescore"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        summary = summarize(out_dir)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    by_rubric = build_units(load_grid_rows(), load_anchor_rows())
    total = sum(len(v) for v in by_rubric.values())
    logger.info("%d rubrics, %d judge calls total", len(by_rubric), total)

    if args.pilot:
        rid = "cj-v3-address"  # the sharpest instrument: a discrete-marker clause
        by_rubric = {rid: by_rubric[rid][:100]}
        out_dir = out_dir / "pilot"
        out_dir.mkdir(parents=True, exist_ok=True)

    meta = run(out_dir, by_rubric, dry_run=args.dry_run)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.dry_run:
        logger.info("dry-run complete: %d calls would dispatch", total)
        return 0

    for rid, t in meta.get("telemetry", {}).items():
        if t["drop_frac"] > 0.02:
            logger.error("[wave %s] drop fraction %.3f exceeds 2%%", rid, t["drop_frac"])
    if args.pilot:
        logger.info("pilot complete: %s", json.dumps(meta.get("telemetry", {})))
        return 0

    summary = summarize(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
