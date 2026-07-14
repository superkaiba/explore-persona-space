#!/usr/bin/env python
"""#1090 fu3 — targeted re-judge of API-529-overload-dropped judge draws (interp r2).

The fu3 P3b judge pass ran through a sustained Anthropic 529-overload window:
2,638 of the round's judge draws are stored as ``error: Error code: 529 ...
overloaded_error`` records inside the per-read ``judge_raw.json`` ``all_scores``
(content-neutral transient censoring — NOT the rule-23 truncation class;
interpretation-critic fu3-r1 item 3). This script re-judges EXACTLY those
dropped draws — same rubric, same Sonnet judge pin, ``max_tokens=300``, via the
same ``judge_graded`` -> ``eval.batch_judge`` path — and merges the fresh
per-draw records into the original ``judge_raw.json`` ``all_scores`` in place.

Why not a cache-bust + full aggregate re-run: the rule-22 ``JudgeCache`` keys on
(rubric, question, completion); the ``n_draws`` repeats of one item share ONE
cache key, so a cache-served re-run would collapse each UNAFFECTED item's
independent multi-draw mean into copies of a single cached draw. The surgical
per-draw merge leaves every kept draw byte-identical and only fills 529 holes.

After the merge it recomputes every judged read of the affected cell evals
(``fu3_cell_evals/*.json``) with the production reduce
(``judge_result_from_save_raw`` + the ``_judge_rate`` reduction), then the
caller re-runs ``issue1090_fu3_aggregate.py`` (all cell evals present -> zero
judging) to refresh the downstream contrast / install / band / summary JSONs.

Parse-error drops (the "other" class: 251 across the round, <=9 per read) stay
dropped per drop-never-coerce; only 529/overload records are re-judged. Stale
529 cache-entry files inside each read dir are deleted (llm-judging rule 23
hygiene) so a future cache-served run cannot re-serve an error record.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps BEFORE heavy imports

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_run as run1090  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.eval.graded_judge import (  # noqa: E402
    judge_graded,
    judge_result_from_save_raw,
)

logger = logging.getLogger("issue1090.fu3.rejudge529")

_529_RE = re.compile(r"529|overloaded", re.IGNORECASE)
_HEX_CACHE_RE = re.compile(r"^[0-9a-f]{16}\.json$")


def _is_529(rec: dict) -> bool:
    """True when a per-draw record is an API-529/overload error record."""
    return bool(rec.get("error")) and bool(_529_RE.search(str(rec.get("reasoning", ""))))


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _parse_tag(tag: str, cells_by_id: dict[str, dict]) -> tuple[dict, str, str, str]:
    """Decode a judge read dir name into (cell row, read kind, context_id, state).

    Tags: ``{cell_id}-t2-{state}`` or ``{cell_id}-by-{context_id}-{state}``.
    """
    state = tag.rsplit("-", 1)[1]
    assert state in ("trained", "base"), tag
    stem = tag[: -(len(state) + 1)]
    if stem[-3:] == "-t2":
        cell_id = stem[: -len("-t2")]
        row = cells_by_id[cell_id]
        return row, "t2", row["context_id"], state
    cell_id, ctx = stem.split("-by-", 1)
    return cells_by_id[cell_id], "by", ctx, state


def _items_for_read(
    run_root: Path, slug: str, kind: str, ctx: str, state: str, tag: str
) -> list[tuple[str, str, str]]:
    """Rebuild the exact flat item list `_judge_rate` judged for this read."""
    sub = "tier2" if kind == "t2" else "bystander"
    payload = json.loads((run_root / slug / sub / f"completions__{state}__{ctx}.json").read_text())
    return [
        (f"{tag}-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(payload["questions"])
        for j, comp in enumerate(payload["completions"][i])
    ]


def rejudge_read(
    judge_dir: Path,
    run_root: Path,
    cells_by_id: dict[str, dict],
    *,
    max_tokens: int,
    dry_run: bool,
) -> dict | None:
    """Re-judge the 529-dropped draws of one (cell, read, state) judge dir.

    Returns a per-read report dict, or None when the read has no 529 records.
    """
    raw_path = judge_dir / "judge_raw.json"
    raw = json.loads(raw_path.read_text())
    all_scores: dict[str, dict] = raw["all_scores"]
    err_keys = [k for k, v in all_scores.items() if isinstance(v, dict) and _is_529(v)]
    if not err_keys:
        return None
    tag = judge_dir.name
    row, kind, ctx, state = _parse_tag(tag, cells_by_id)
    shim = fu3w.run_cell_shim(row)
    items = _items_for_read(run_root, shim.slug, kind, ctx, state, tag)
    qa_by_item = {iid: (q, a) for iid, q, a in items}

    # Group the missing draws by item; one judge_graded call per missing-count k.
    missing_by_item: dict[str, list[str]] = {}
    for k in err_keys:
        item_id = k.rsplit("__", 2)[0]
        missing_by_item.setdefault(item_id, []).append(k)
    by_count: dict[int, list[str]] = {}
    for item_id, keys in missing_by_item.items():
        by_count.setdefault(len(keys), []).append(item_id)

    n_recovered = 0
    n_still_error = 0
    behavior = BEHAVIORS[row["behavior"]]
    if not dry_run:
        with tempfile.TemporaryDirectory(prefix=f"rejudge529-{tag}-") as scratch:
            for k_draws, item_ids in sorted(by_count.items()):
                sub_items = [(iid, *qa_by_item[iid]) for iid in sorted(item_ids)]
                scratch_raw = Path(scratch) / f"raw_k{k_draws}.json"
                judge_graded(
                    sub_items,
                    behavior.judge_rubric,
                    n_draws=k_draws,
                    cache_dir=Path(scratch) / f"cache_k{k_draws}",
                    save_raw=scratch_raw,
                    judge_model=behavior.judge_model,
                    max_tokens=max_tokens,
                )
                fresh = json.loads(scratch_raw.read_text())["all_scores"]
                fresh_by_item: dict[str, list[dict]] = {}
                for cid, rec in fresh.items():
                    fresh_by_item.setdefault(cid.rsplit("__", 2)[0], []).append(rec)
                for iid in item_ids:
                    fresh_recs = fresh_by_item.get(iid, [])
                    assert len(fresh_recs) == k_draws, (tag, iid, len(fresh_recs), k_draws)
                    for orig_key, rec in zip(sorted(missing_by_item[iid]), fresh_recs, strict=True):
                        all_scores[orig_key] = rec
                        if isinstance(rec, dict) and rec.get("error"):
                            n_still_error += 1
                        else:
                            n_recovered += 1
        raw["rejudge_529"] = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rejudged": len(err_keys),
            "n_recovered": n_recovered,
            "n_still_error": n_still_error,
            "max_tokens": max_tokens,
        }
        _atomic_write_json(raw_path, raw)
        # Rule-23 cache hygiene: delete stale 529 cache-entry files so a future
        # cache-served pass cannot re-serve an error record.
        n_cache_purged = 0
        for f in judge_dir.iterdir():
            if _HEX_CACHE_RE.match(f.name):
                try:
                    rec = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if isinstance(rec, dict) and _is_529(rec):
                    f.unlink()
                    n_cache_purged += 1
    else:
        n_cache_purged = 0
    return {
        "tag": tag,
        "cell_id": row["cell_id"],
        "n_529": len(err_keys),
        "n_recovered": n_recovered,
        "n_still_error": n_still_error,
        "n_cache_purged": n_cache_purged,
    }


def recompute_cell_eval(
    out_dir: Path, run_root: Path, cell_id: str, cells_by_id: dict[str, dict]
) -> dict:
    """Recompute every judged read of one cell eval from the merged judge_raw
    files with the production reduce; rewrite fu3_cell_evals/<slug>.json."""
    row = cells_by_id[cell_id]
    behavior = BEHAVIORS[row["behavior"]]
    shim = fu3w.run_cell_shim(row)
    eval_path = out_dir / "fu3_cell_evals" / f"{shim.slug}.json"
    rec = json.loads(eval_path.read_text())
    judge_root = out_dir / "judge" / row["behavior"]
    changes: dict[str, dict] = {}

    def _reduce(tag: str, kind: str, ctx: str, state: str) -> dict:
        items = _items_for_read(run_root, shim.slug, kind, ctx, state, tag)
        result = judge_result_from_save_raw(judge_root / tag / "judge_raw.json", items)
        n_dropped = n_pos = n_scored = 0
        for iid, _q, _c in items:
            score = result.scores.get(iid)
            if score is None:
                n_dropped += 1
                continue
            n_scored += 1
            if score > behavior.threshold:
                n_pos += 1
        assert n_scored > 0, f"every completion at {tag} judge-dropped"
        lo, hi = run1090._wilson(n_pos, n_scored)
        return {
            "rate": n_pos / n_scored,
            "k": n_pos,
            "n": n_scored,
            "n_dropped": n_dropped,
            "n_total_draws": result.n_total_draws,
            "n_dropped_draws": result.n_dropped_draws,
            "wilson95": [lo, hi],
            "mode": "judged",
        }

    for state in ("trained", "base"):
        old = rec["tier2"][state]
        if old.get("mode") != "judged":
            continue
        new = _reduce(f"{cell_id}-t2-{state}", "t2", row["context_id"], state)
        if new["rate"] != old["rate"] or new["n_dropped_draws"] != old.get("n_dropped_draws"):
            changes[f"t2-{state}"] = {"old_rate": old["rate"], "new_rate": new["rate"]}
        rec["tier2"][state] = new
    rec["install_delta"] = rec["tier2"]["trained"]["rate"] - rec["tier2"]["base"]["rate"]
    for brec in rec["bystanders"]:
        cid = brec["context_id"]
        for state in ("trained", "base"):
            old = brec[state]
            if old.get("mode") != "judged":
                continue
            new = _reduce(f"{cell_id}-by-{cid}-{state}", "by", cid, state)
            if new["rate"] != old["rate"]:
                changes[f"by-{cid}-{state}"] = {"old_rate": old["rate"], "new_rate": new["rate"]}
            brec[state] = new
        brec["leak_delta"] = brec["trained"]["rate"] - brec["base"]["rate"]
    held_out = [b["leak_delta"] for b in rec["bystanders"] if not b["is_source_context"]]
    rec["leakage_mean_held_out"] = (sum(held_out) / len(held_out)) if held_out else None
    lo, hi = rec["band"]
    rec["band_hit"] = bool(lo <= rec["tier2"]["trained"]["rate"] <= hi)
    run1090._atomic_write_json(eval_path, rec)
    return changes


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="#1090 fu3 targeted 529 re-judge")
    ap.add_argument("--run-root", default="data/issue_1090/fu3")
    ap.add_argument("--out", default="eval_results/issue_1090/fu3")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true", help="scan + report only, no API calls")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    run_root = Path(args.run_root)
    cells_by_id = {r["cell_id"]: r for r in fu3_cells.CELLS}

    reports = []
    for raw_path in sorted(out_dir.glob("judge/*/*/judge_raw.json")):
        rep = rejudge_read(
            raw_path.parent,
            run_root,
            cells_by_id,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
        )
        if rep:
            reports.append(rep)
            logger.info(
                "[rejudge529] %s: %d dropped -> %d recovered, %d still-error",
                rep["tag"],
                rep["n_529"],
                rep["n_recovered"],
                rep["n_still_error"],
            )

    cell_changes: dict[str, dict] = {}
    if not args.dry_run:
        for cell_id in sorted({r["cell_id"] for r in reports}):
            cell_changes[cell_id] = recompute_cell_eval(out_dir, run_root, cell_id, cells_by_id)

    summary = {
        "n_reads_rejudged": len(reports),
        "n_529_total": sum(r["n_529"] for r in reports),
        "n_recovered_total": sum(r["n_recovered"] for r in reports),
        "n_still_error_total": sum(r["n_still_error"] for r in reports),
        "reads": reports,
        "cell_rate_changes": cell_changes,
    }
    _atomic_write_json(out_dir / "fu3_rejudge529_report.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
