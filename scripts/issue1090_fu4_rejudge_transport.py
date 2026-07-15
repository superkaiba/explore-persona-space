#!/usr/bin/env python
"""#1090 fu4 — targeted re-judge of TRANSPORT-lost judge draws (rule 24 pre-wiring).

fu4's VM P3 aggregate (``issue1090_fu4.py --phase judge-aggregate``) reports a
per-run ``transport_losses`` count (llm-judging rules 9/24: a stored
``error: True`` per-draw row is a re-judgeable TRANSPORT loss, never a content
drop) and warns that a nonzero count must be re-judged before any headline
read. This tool is the fu4 adaptation of ``scripts/issue1090_fu3_rejudge_529.py``
(concern ``fu4-transport-rejudge-tool-not-prewired``): it re-judges EXACTLY
the transport-lost draws of the P3 judge outputs — SAME instrument (the
behavior's rubric + Sonnet judge pin, ``max_tokens=300``) — surgically merges
the fresh per-draw records into ``judge_raw.json`` in place, then recomputes
the affected ``fu4_ladders.json`` records with the production reduce
(``judge_result_from_save_raw`` + the ``_judge_rate`` reduction + the
transport/content split + the verdict-lattice inputs).

fu4 judge-raw layout (both read kinds scanned):

- ``<out_root>/fu4_aggregate/judge/<behavior>/<run_id>-t2-trained/judge_raw.json``
  (written by ``_judge_run_tier2`` -> ``i1090._judge_rate``; items rebuilt from
  ``<out_root>/<run_id>/tier2/completions__trained__<context_id>.json``)
- ``<out_root>/fu4_aggregate/judge/formatting_reread/<run_id>/judge_raw.json``
  (written by ``_formatting_judged_reread``; same completions file)

Deliberate deviation from the fu3 tool: the transport predicate is
``isinstance(rec, dict) and rec.get("error")`` — EXACTLY the set fu4's
``_drop_split_from_raw`` counts as ``transport_losses`` (rule 24's transport
class covers 429/529/timeout/connection, all persisted as ``error`` dicts by
the api_dispatch layer), not fu3's 529-only regex subset. The fu3 MECHANISM is
kept: per-draw grouping by missing-count, a fresh scratch ``cache_dir`` per
read (rule 24(ii) — the rubric-keyed JudgeCache shares one key across an
item's draws, so a cache-served re-run would silently duplicate a sibling
draw), and rule-23/24 hygiene deleting stale ``error`` cache-entry files in
each read dir so a future cache-served pass cannot re-serve them.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps BEFORE heavy imports

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as i1090  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.eval.graded_judge import (  # noqa: E402
    judge_graded,
    judge_result_from_save_raw,
)

logger = logging.getLogger("issue1090.fu4.rejudge_transport")

_HEX_CACHE_RE = re.compile(r"^[0-9a-f]{16}\.json$")  # JudgeCache._hash_key file shape


def _is_transport(rec: object) -> bool:
    """fu4's transport-loss predicate — the exact set ``_drop_split_from_raw``
    counts (an ``error: True`` per-draw dict; llm-judging rules 9/24)."""
    return isinstance(rec, dict) and bool(rec.get("error"))


def _parse_read(judge_dir: Path) -> tuple[fu4.Fu4Run, str, str]:
    """Decode a fu4 judge read dir into (run, kind, item-id prefix)."""
    if judge_dir.parent.name == "formatting_reread":
        run = fu4._run_by_id()[judge_dir.name]
        return run, "reread", f"{run.run_id}-reread"
    tag = judge_dir.name
    suffix = "-t2-trained"
    assert tag.endswith(suffix), f"unrecognized fu4 judge read dir: {judge_dir}"
    run = fu4._run_by_id()[tag[: -len(suffix)]]
    assert judge_dir.parent.name == run.behavior, (str(judge_dir), run.behavior)
    return run, "t2", tag


def _items_for_read(out_root: Path, run: fu4.Fu4Run, prefix: str) -> list[tuple[str, str, str]]:
    """Rebuild the exact flat item list the P3 judge pass judged for this read."""
    ctx_file = out_root / run.run_id / "tier2" / f"completions__trained__{run.context_id}.json"
    payload = json.loads(ctx_file.read_text())
    return [
        (f"{prefix}-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(payload["questions"])
        for j, comp in enumerate(payload["completions"][i])
    ]


def rejudge_read(judge_dir: Path, out_root: Path, *, max_tokens: int, dry_run: bool) -> dict | None:
    """Re-judge the transport-lost draws of one fu4 judge read, merging the
    fresh per-draw records into ``judge_raw.json`` in place.

    Returns a per-read report dict, or None when the read has no transport rows.
    """
    raw_path = judge_dir / "judge_raw.json"
    raw = json.loads(raw_path.read_text())
    all_scores: dict[str, dict] = raw["all_scores"]
    err_keys = [k for k, v in all_scores.items() if _is_transport(v)]
    if not err_keys:
        return None
    run, kind, prefix = _parse_read(judge_dir)
    items = _items_for_read(out_root, run, prefix)
    qa_by_item = {iid: (q, a) for iid, q, a in items}

    # Group the transport-lost draws by item; one judge_graded call per
    # missing-count k (the fu3 mechanism, kept verbatim).
    missing_by_item: dict[str, list[str]] = {}
    for k in err_keys:
        item_id = k.rsplit("__", 2)[0]
        if item_id not in qa_by_item:
            raise KeyError(f"{judge_dir}: transport key {k!r} decodes to unknown item {item_id!r}")
        missing_by_item.setdefault(item_id, []).append(k)
    by_count: dict[int, list[str]] = {}
    for item_id, keys in missing_by_item.items():
        by_count.setdefault(len(keys), []).append(item_id)

    n_recovered = 0
    n_still_error = 0
    n_cache_purged = 0
    behavior = BEHAVIORS[run.behavior]
    if not dry_run:
        with tempfile.TemporaryDirectory(prefix=f"fu4-rejudge-{judge_dir.name}-") as scratch:
            for k_draws, item_ids in sorted(by_count.items()):
                sub_items = [(iid, *qa_by_item[iid]) for iid in sorted(item_ids)]
                scratch_raw = Path(scratch) / f"raw_k{k_draws}.json"
                judge_graded(
                    sub_items,
                    behavior.judge_rubric,
                    n_draws=k_draws,
                    cache_dir=Path(scratch) / f"cache_k{k_draws}",  # fresh — rule 24(ii)
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
                    assert len(fresh_recs) == k_draws, (
                        judge_dir.name,
                        iid,
                        len(fresh_recs),
                        k_draws,
                    )
                    for orig_key, rec in zip(sorted(missing_by_item[iid]), fresh_recs, strict=True):
                        all_scores[orig_key] = rec
                        if _is_transport(rec):
                            n_still_error += 1
                        else:
                            n_recovered += 1
        raw["rejudge_transport"] = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rejudged": len(err_keys),
            "n_recovered": n_recovered,
            "n_still_error": n_still_error,
            "max_tokens": max_tokens,
        }
        i1090._atomic_write_json(raw_path, raw)
        # Rule-23/24 cache hygiene: delete stale transport cache-entry files so
        # a future cache-served pass cannot re-serve an error record.
        for f in judge_dir.iterdir():
            if _HEX_CACHE_RE.match(f.name):
                try:
                    rec = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if _is_transport(rec):
                    f.unlink()
                    n_cache_purged += 1
    return {
        "read": str(judge_dir.relative_to(out_root)),
        "run_id": run.run_id,
        "kind": kind,
        "n_transport": len(err_keys),
        "n_recovered": n_recovered,
        "n_still_error": n_still_error,
        "n_cache_purged": n_cache_purged,
    }


def _reduce(result, items: list[tuple[str, str, str]], behavior, *, mode: str) -> dict:
    """The production reduce (``i1090._judge_rate`` / ``_formatting_judged_reread``
    tail) over a merged ``judge_raw.json``: drop-never-coerce mean per item,
    positive above the behavior threshold, Wilson 95% CI."""
    n_pos = n_scored = n_dropped = 0
    for iid, _q, _c in items:
        score = result.scores.get(iid)
        if score is None:
            n_dropped += 1
            continue
        n_scored += 1
        if score > behavior.threshold:
            n_pos += 1
    if n_scored == 0:
        raise ValueError(f"every completion judge-dropped post-rejudge ({mode})")
    lo, hi = i1090._wilson(n_pos, n_scored)
    return {
        "rate": n_pos / n_scored,
        "k": n_pos,
        "n": n_scored,
        "n_dropped": n_dropped,
        "n_total_draws": result.n_total_draws,
        "n_dropped_draws": result.n_dropped_draws,
        "wilson95": [lo, hi],
        "mode": mode,
    }


def recompute_ladders(ladders_path: Path, out_root: Path, reports: list[dict]) -> dict:
    """Recompute the affected fu4_ladders.json records from the merged raws:
    tier2_trained (+ transport/content split + K4 flag + install_delta) per t2
    read, formatting_judged_reread per reread read, then the registered
    verdict-lattice inputs. Atomic rewrite; returns per-read rate changes."""
    out = json.loads(ladders_path.read_text())
    judge_root = out_root / f"{fu4.ROUND.name}_aggregate" / "judge"
    changes: dict[str, dict] = {}
    for rep in reports:
        run = fu4._run_by_id()[rep["run_id"]]
        rec = out["runs"].get(run.run_id)
        if rec is None:
            raise KeyError(
                f"{ladders_path}: no run record for {run.run_id} — run the P3 "
                "aggregate before the transport re-judge"
            )
        behavior = BEHAVIORS[run.behavior]
        if rep["kind"] == "t2":
            tag = f"{run.run_id}-t2-trained"
            items = _items_for_read(out_root, run, tag)
            raw_path = judge_root / run.behavior / tag / "judge_raw.json"
            new = _reduce(
                judge_result_from_save_raw(raw_path, items), items, behavior, mode="judged"
            )
            split = fu4._drop_split_from_raw(judge_root / run.behavior, tag)
            new["transport_losses"] = split["transport_losses"]
            new["content_dropped_draws"] = new["n_dropped_draws"] - split["transport_losses"]
            content_rate = new["content_dropped_draws"] / max(new["n_total_draws"], 1)
            new["k4_truncation_check_required"] = bool(content_rate >= 0.10)
            old = rec.get("tier2_trained") or {}
            changes[rep["read"]] = {"old_rate": old.get("rate"), "new_rate": new["rate"]}
            rec["tier2_trained"] = new
            base = rec.get("base_tier2") or {}
            if base.get("rate") is not None:
                rec["install_delta"] = new["rate"] - base["rate"]
        else:
            items = _items_for_read(out_root, run, f"{run.run_id}-reread")
            raw_path = judge_root / "formatting_reread" / run.run_id / "judge_raw.json"
            new = _reduce(
                judge_result_from_save_raw(raw_path, items), items, behavior, mode="judged_reread"
            )
            old = rec.get("formatting_judged_reread") or {}
            changes[rep["read"]] = {"old_rate": old.get("rate"), "new_rate": new["rate"]}
            rec["formatting_judged_reread"] = new
    runs = tuple(fu4._run_by_id()[rid] for rid in out["runs"] if rid in fu4._run_by_id())
    fu4._verdict_lattice_inputs(out, runs)
    i1090._atomic_write_json(ladders_path, out)
    return changes


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="#1090 fu4 targeted transport-loss re-judge")
    ap.add_argument("--out-root", default=f"data/issue_{i1090.ISSUE}/fu4")
    ap.add_argument("--ladders", default=str(fu4.DELIVERABLES_DIR / "fu4_ladders.json"))
    ap.add_argument("--max-tokens", type=int, default=fu4.JUDGE_MAX_TOKENS_FU4)
    ap.add_argument("--dry-run", action="store_true", help="scan + report only, no API calls")
    ap.add_argument("--round", default="fu4", choices=tuple(sorted(fu4.ROUNDS)))
    args = ap.parse_args(argv)
    fu4.set_round(args.round)
    out_root = Path(args.out_root)
    judge_root = out_root / f"{fu4.ROUND.name}_aggregate" / "judge"

    reports: list[dict] = []
    for raw_path in sorted(judge_root.glob("*/*/judge_raw.json")):
        rep = rejudge_read(
            raw_path.parent, out_root, max_tokens=args.max_tokens, dry_run=args.dry_run
        )
        if rep:
            reports.append(rep)
            logger.info(
                "[fu4-rejudge] %s: %d transport-lost -> %d recovered, %d still-error",
                rep["read"],
                rep["n_transport"],
                rep["n_recovered"],
                rep["n_still_error"],
            )

    ladder_changes: dict[str, dict] = {}
    if reports and not args.dry_run:
        ladder_changes = recompute_ladders(Path(args.ladders), out_root, reports)

    summary = {
        "n_reads_rejudged": len(reports),
        "n_transport_total": sum(r["n_transport"] for r in reports),
        "n_recovered_total": sum(r["n_recovered"] for r in reports),
        "n_still_error_total": sum(r["n_still_error"] for r in reports),
        "max_tokens": args.max_tokens,
        "dry_run": bool(args.dry_run),
        "reads": reports,
        "ladder_changes": ladder_changes,
        "git_commit": i1074._git_short_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not args.dry_run:
        i1090._atomic_write_json(
            Path(args.ladders).parent / "fu4_rejudge_transport_report.json", summary
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
