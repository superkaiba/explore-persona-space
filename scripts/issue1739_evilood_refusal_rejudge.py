#!/usr/bin/env python3
"""#1739 evil-OOD r2v2: SYNC re-judge of Batch-API ``stop_reason="refusal"`` draws.

WHY (measured, not assumed): the production 44,310-draw wave ran on the forced
Batch path and came back with 15,091 draws (34.1%) carrying
``stop_reason == "refusal"`` — the API's own classifier refusal, NOT a rubric
refusal and NOT a parse failure. It is concentrated on tom-gibbs (63.9% of that
corpus's rollout items lost EVERY draw), which censors the DV exactly where the
jailbreak corpus is most harmful — i.e. outcome-correlated censoring, the worst
possible missingness for a behavior-expression DV.

The rule-26 pilot could not catch it: the pilot routes SYNC at 200 draws
(below the batch crossover), so the pilot and the production wave ran different
TRANSPORTS at an otherwise identical instrument.

DIFFERENTIAL (the evidence this is transport-side, not content-side): 25
tom-gibbs items whose every batch draw was ``refusal`` were re-issued on the
SYNC path at the identical instrument — 24/25 scored cleanly (96%); the single
drop was a non-JSON prose verdict (the already-characterised stochastic
parse-fail class), not a refusal.

WHAT THIS DOES: re-issues ONLY the censored draws on the SYNC path
(``threshold_base`` large ⇒ forced sync), merges them with the batch draws that
returned genuine scores, and rebuilds the DV dataset. Instrument is IDENTICAL
across both sets — same judge model, rubric, temperature, and max_tokens; only
the HTTP transport differs, and the differential above establishes the batch
refusal as a transport censor rather than a property of the content. The merge
is DISCLOSED in the output meta (``transport_split``).

Draws that were genuinely scored on the batch path are NOT re-issued (they are
real judge outputs at the production sampling params); re-issuing them would
only add temperature noise at 2x the spend.

MOP-UP MODE (``--mopup``, r3): after the main rejudge pass, re-issue the two
RETRIABLE residual classes on the SYNC path at the identical instrument:
(a) TRANSPORT-lost draws (529 exhaustion — rule 24: re-judged before
publication, wherever they sit), and (b) stochastic parse-fail draws
(malformed / truncation content drops — the known non-JSON prose-verdict
class) on items still UNSCORED after the merge. Rubric-REFUSAL returns
(``{"score": "REFUSAL"}`` — the judge's verdict that the judged completion
itself refused) are content-informative rule-9 drops and are NEVER re-drawn;
items whose every draw is a rubric REFUSAL stay unscored and are COUNTED.
Mop-up draws use FRESH cache dirs — the rubric-keyed JudgeCache persists
content-class error dicts and collapses same-item draws to one entry, so a
reused cache dir would silently serve the old failures back (rule 24(ii)).

CONTENT HYGIENE: prints ids, counts, and rates — never question/answer text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_evilood_refusal_rejudge.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root derivation failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

BEHAVIOR = "evil"
FORCE_SYNC_THRESHOLD = 50_000_000  # decide_route: n_items < threshold -> sync


def _batch_draws(cache_dir: Path) -> dict[str, list[dict]]:
    """item_id -> list of drained per-draw payloads from the batch dispatch."""
    out: dict[str, list[dict]] = defaultdict(list)
    files = sorted(cache_dir.glob(".dispatch/*/results_*.json"))
    if not files:
        raise FileNotFoundError(f"no batch result files under {cache_dir}/.dispatch/")
    for f in files:
        for cid, payload in json.loads(f.read_text()).get("scores", {}).items():
            out[cid.rsplit("__", 2)[0]].append(payload)
    return dict(out)


def _load_texts(stage_root: str | Path) -> dict[str, tuple[str, str]]:
    """Rollout (query, completion) texts keyed by the judge item id."""
    from explore_persona_space.experiments.issue_1739 import judging

    texts: dict[str, tuple[str, str]] = {}
    for rung_dir in sorted(Path(stage_root).glob("*/rollouts/full")):
        for p in rung_dir.glob("*.json"):
            if p.name.startswith("_"):
                continue
            rec = json.loads(p.read_text())
            texts[judging.rollout_item_id(rec["context_id"], int(rec["rollout_k"]))] = (
                rec["query"],
                rec["completion"],
            )
    return texts


def _load_contexts_meta(stage_root: str | Path) -> dict[str, dict]:
    """context_id -> rollout payload (behavior/split/rung/group_key carrier)."""
    contexts_meta: dict[str, dict] = {}
    for rung_dir in sorted(Path(stage_root).glob("*/rollouts/full")):
        for p in rung_dir.glob("*.json"):
            if p.name.startswith("_"):
                continue
            rec = json.loads(p.read_text())
            contexts_meta[rec["context_id"]] = rec
    return contexts_meta


def _classify_residual(payload: object) -> str:
    """Class of a non-kept, non-api-refusal draw payload (rule-24 split).

    ``transport`` (re-judgeable loss) | ``rubric_refusal`` (content verdict,
    never re-drawn) | ``truncation`` / ``malformed`` (the stochastic
    parse-fail class — retriable per the mop-up brief).
    """
    from explore_persona_space.eval import batch_judge as _bj
    from explore_persona_space.eval.graded_judge import _is_refusal_parsed

    if _bj.is_transport_error_dict(payload):
        return "transport"
    if _is_refusal_parsed(payload):
        return "rubric_refusal"
    stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
    if _bj.is_truncation_stop_reason(stop_reason):
        return "truncation"
    return "malformed"


def main() -> int:
    ap = argparse.ArgumentParser(description="#1739 evil-OOD refusal-draw sync re-judge")
    ap.add_argument("--stage-root", default="data/issue_1739/evil_ood_full_stage")
    ap.add_argument("--judge-dir", default="eval_results/issue_1739/evil_ood_full/judge")
    ap.add_argument(
        "--cache-dir", default=None, help="batch cache (default: newest under judge-dir)"
    )
    ap.add_argument("--out-root", default="eval_results/issue_1739/evil_ood_full")
    ap.add_argument("--inputs-dir", default="data/issue_1739/inputs")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--limit-items", type=int, default=None, help="pilot slice (timing basis)")
    ap.add_argument("--mopup", action="store_true", help="residual mop-up mode (r3, module doc)")
    ap.add_argument(
        "--plan-only", action="store_true", help="mopup: print the re-issue plan, dispatch nothing"
    )
    ap.add_argument(
        "--max-calls", type=int, default=612, help="mopup: hard budget on re-issued draws"
    )
    args = ap.parse_args()
    if args.mopup:
        return mopup(args)

    from explore_persona_space.experiments.issue_1739 import dv_build, judging
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
        N_JUDGE_DRAWS,
    )

    temperature = args.temperature if args.temperature is not None else JUDGE_TEMPERATURE
    judge_dir = Path(args.judge_dir)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else sorted(judge_dir.glob("judge_cache_*"))[-1]
    )
    drained = _batch_draws(cache_dir)

    # Per item: keep genuinely-scored batch draws; count the refusal-censored ones.
    kept: dict[str, list[float]] = {}
    n_missing: dict[str, int] = {}
    n_refusal_draws = 0
    for item, payloads in drained.items():
        good = [
            float(p["score"])
            for p in payloads
            if isinstance(p, dict) and isinstance(p.get("score"), (int, float))
        ]
        refused = sum(
            1 for p in payloads if isinstance(p, dict) and p.get("stop_reason") == "refusal"
        )
        n_refusal_draws += refused
        kept[item] = good
        if refused:
            n_missing[item] = refused
    print(
        f"[rejudge] items={len(drained)} batch_refusal_draws={n_refusal_draws} "
        f"items_needing_redraws={len(n_missing)}",
        flush=True,
    )

    # Rollout texts, keyed by the same item id the judge used.
    texts = _load_texts(args.stage_root)

    targets = sorted(n_missing)
    if args.limit_items is not None:
        targets = targets[: args.limit_items]
    missing_text = [i for i in targets if i not in texts]
    if missing_text:
        raise RuntimeError(
            f"{len(missing_text)} target items have no staged text: {missing_text[:3]}"
        )

    # One sync pass per redraw depth: items needing d draws are batched together
    # so every call carries the production n_draws semantics per item.
    eval_prompt = judging.load_trait_rubric(BEHAVIOR, inputs_dir=args.inputs_dir)
    by_depth: dict[int, list[str]] = defaultdict(list)
    for item in targets:
        by_depth[n_missing[item]].append(item)
    sync_scores: dict[str, float | None] = {}
    t0 = time.time()
    for depth in sorted(by_depth):
        items = [(i, *texts[i]) for i in by_depth[depth]]
        print(f"[rejudge] sync pass: {len(items)} items x {depth} draw(s)", flush=True)
        res = judging.judge_items_graded(
            items,
            eval_prompt,
            cache_dir=Path(args.out_root) / "rejudge_cache" / f"d{depth}",
            save_raw=Path(args.out_root) / "judge" / f"judge_raw_rejudge_d{depth}.json",
            n_draws=depth,
            temperature=temperature,
            max_tokens=args.max_tokens,
            threshold_base=FORCE_SYNC_THRESHOLD,
        )
        sync_scores.update(res.scores)
    wall = time.time() - t0
    n_rescued = sum(1 for v in sync_scores.values() if v is not None)
    print(
        f"[rejudge] sync rescued {n_rescued}/{len(sync_scores)} items "
        f"({n_rescued / max(len(sync_scores), 1):.1%}) in {wall:.0f}s",
        flush=True,
    )

    # Merge: a sync mean folds in alongside the item's surviving batch draws.
    merged: dict[str, float | None] = {}
    for item, good in kept.items():
        vals = list(good)
        s = sync_scores.get(item)
        if s is not None:
            vals.append(float(s))
        merged[item] = float(sum(vals) / len(vals)) if vals else None
    n_scored = sum(1 for v in merged.values() if v is not None)
    print(
        f"[rejudge] merged item coverage {n_scored}/{len(merged)} ({n_scored / len(merged):.1%})",
        flush=True,
    )

    contexts_meta = _load_contexts_meta(args.stage_root)

    dv_rows = dv_build.build_labeling_dv(merged, n_draws=N_JUDGE_DRAWS, contexts_meta=contexts_meta)
    dv_path = dv_build.write_dv_dataset(
        dv_rows,
        out_root=args.out_root,
        behavior=BEHAVIOR,
        judge_payload_meta={
            "judge_model": JUDGE_MODEL,
            "judge_max_tokens": args.max_tokens,
            "judge_temperature": temperature,
            "n_draws": N_JUDGE_DRAWS,
            "rubric": "trait_eval_prompt",
            "transport_split": (
                f"{n_refusal_draws} of {sum(len(v) for v in drained.values())} production draws "
                "were censored by the Batch API with stop_reason='refusal' (transport-side, "
                "not a rubric refusal); those draws were re-issued on the SYNC path at an "
                "IDENTICAL instrument (same model/rubric/temperature/max_tokens) and merged "
                "with the batch draws that returned genuine scores. Evidence the censor is "
                "transport-side: 24/25 all-refused tom-gibbs items scored cleanly on a sync "
                "re-probe. Disclosed per the display/provenance rules."
            ),
        },
        git_commit="",
    )
    print(json.dumps({"dv_dataset_path": str(dv_path), "n_rows": len(dv_rows)}, indent=1))
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def mopup(args) -> int:
    """Residual mop-up (r3): re-issue the two RETRIABLE residual classes.

    Re-issues, on the SYNC path at the identical instrument: (a) every
    TRANSPORT-lost draw (529 exhaustion — rule 24 mandates re-judging before
    publication, on scored and unscored items alike), and (b) the stochastic
    parse-fail draws (malformed / truncation) on items still UNSCORED after
    the main-pass merge. Rubric-REFUSAL draws are rule-9 content verdicts —
    never re-drawn; all-refusal items stay unscored and are counted. The
    main-pass sync results are re-read from their persisted save_raw files
    (zero API calls); mop-up draws go through FRESH cache dirs (module doc).
    """
    from collections import Counter

    from explore_persona_space.eval.graded_judge import (
        _score_from_parsed,
        judge_result_from_save_raw,
    )
    from explore_persona_space.experiments.issue_1739 import dv_build, judging
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
        N_JUDGE_DRAWS,
    )
    from explore_persona_space.orchestrate.provenance import commit_string, git_provenance

    temperature = args.temperature if args.temperature is not None else JUDGE_TEMPERATURE
    judge_dir = Path(args.judge_dir)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else sorted(judge_dir.glob("judge_cache_*"))[-1]
    )
    drained = _batch_draws(cache_dir)
    texts = _load_texts(args.stage_root)

    # Batch pass: kept draws (the ORIGINAL merge predicate — bit parity with
    # main()), api-refusal counts (the main pass already re-issued those), and
    # residual classes for everything else.
    kept: dict[str, list[float]] = {}
    n_missing: dict[str, int] = {}
    batch_resid: dict[str, Counter] = {}
    for item, payloads in drained.items():
        kept[item] = [
            float(p["score"])
            for p in payloads
            if isinstance(p, dict) and isinstance(p.get("score"), (int, float))
        ]
        refused = 0
        resid: Counter = Counter()
        for p in payloads:
            if isinstance(p, dict) and p.get("stop_reason") == "refusal":
                refused += 1
                continue
            if isinstance(p, dict) and isinstance(p.get("score"), (int, float)):
                continue
            resid[_classify_residual(p)] += 1
        if refused:
            n_missing[item] = refused
        if resid:
            batch_resid[item] = resid

    # Main-pass sync results: pure read of the persisted save_raw files via the
    # production reduce (merge parity), plus a raw re-read for the per-item
    # residual-class split JudgeResult only aggregates.
    by_depth: dict[int, list[str]] = defaultdict(list)
    for item in sorted(n_missing):
        by_depth[n_missing[item]].append(item)
    sync_scores: dict[str, float | None] = {}
    sync_resid: dict[str, Counter] = {}
    for depth in sorted(by_depth):
        save_raw = judge_dir / f"judge_raw_rejudge_d{depth}.json"
        items = [(i, *texts[i]) for i in by_depth[depth]]
        res = judge_result_from_save_raw(save_raw, items)
        sync_scores.update(res.scores)
        target = set(by_depth[depth])
        for cid, parsed in json.loads(save_raw.read_text()).get("all_scores", {}).items():
            item = cid.rsplit("__", 2)[0]
            if item not in target or _score_from_parsed(parsed) is not None:
                continue
            sync_resid.setdefault(item, Counter())[_classify_residual(parsed)] += 1

    # Re-issue plan: transport draws everywhere; stochastic parse-fails on
    # still-unscored items only. Rubric refusals are never re-drawn.
    unscored = {i for i in drained if not kept[i] and sync_scores.get(i) is None}
    reissue: dict[str, int] = {}
    item_class: dict[str, str] = {}
    plan = Counter()
    for item in sorted(drained):
        b = batch_resid.get(item, Counter())
        s = sync_resid.get(item, Counter())
        transport = b["transport"] + s["transport"]
        stochastic = b["malformed"] + b["truncation"] + s["malformed"] + s["truncation"]
        n = transport + (stochastic if item in unscored else 0)
        if not n:
            continue
        reissue[item] = n
        item_class[item] = (
            "transport" if transport == n else ("stochastic_parse" if transport == 0 else "mixed")
        )
        plan["transport_draws"] += transport
        if item in unscored:
            plan["stochastic_parse_draws"] += stochastic
            plan["unscored_items_targeted"] += 1
        else:
            plan["scored_items_topped_up"] += 1
    n_refusal_only = sum(1 for i in unscored if i not in reissue)
    total_calls = sum(reissue.values())
    print(
        f"[mopup] unscored_items={len(unscored)} retriable-bearing="
        f"{plan['unscored_items_targeted']} rubric-refusal-only(stay unscored)={n_refusal_only}",
        flush=True,
    )
    print(
        f"[mopup] plan: {total_calls} calls — transport={plan['transport_draws']} draws "
        f"(incl. {plan['scored_items_topped_up']} scored items topped up, rule 24), "
        f"stochastic_parse={plan['stochastic_parse_draws']} draws on unscored items",
        flush=True,
    )
    if total_calls > args.max_calls:
        raise SystemExit(f"[mopup] planned {total_calls} calls > budget {args.max_calls}; refusing")
    missing_text = [i for i in reissue if i not in texts]
    if missing_text:
        raise RuntimeError(f"{len(missing_text)} mop-up items lack staged text: {missing_text[:3]}")
    if args.plan_only:
        print("[mopup] plan-only: no calls dispatched, no files written", flush=True)
        return 0

    # Dispatch on the sync path, fresh caches, identical instrument.
    eval_prompt = judging.load_trait_rubric(BEHAVIOR, inputs_dir=args.inputs_dir)
    by_k: dict[int, list[str]] = defaultdict(list)
    for item, n in reissue.items():
        by_k[n].append(item)
    mop_scores: dict[str, float | None] = {}
    mop_kept_counts: dict[str, int] = {}
    mop_transport: dict[str, int] = {}
    mop_tally = Counter()
    t0 = time.time()
    for k in sorted(by_k):
        items = [(i, *texts[i]) for i in sorted(by_k[k])]
        print(f"[mopup] sync pass: {len(items)} items x {k} draw(s)", flush=True)
        res = judging.judge_items_graded(
            items,
            eval_prompt,
            cache_dir=Path(args.out_root) / "rejudge_mopup_cache" / f"d{k}",
            save_raw=Path(args.out_root) / "judge" / f"judge_raw_rejudge_mopup_d{k}.json",
            n_draws=k,
            temperature=temperature,
            max_tokens=args.max_tokens,
            threshold_base=FORCE_SYNC_THRESHOLD,
        )
        mop_scores.update(res.scores)
        for i, lst in res.per_item_scores.items():
            mop_kept_counts[i] = mop_kept_counts.get(i, 0) + len(lst)
        for i, n in res.per_item_transport_losses.items():
            mop_transport[i] = mop_transport.get(i, 0) + n
        mop_tally["draws_kept"] += sum(len(v) for v in res.per_item_scores.values())
        mop_tally["content_dropped"] += res.n_dropped_draws
        mop_tally["refusal"] += res.n_refusal_draws
        mop_tally["truncation"] += res.n_truncation_dropped_draws
        mop_tally["transport_lost_again"] += res.n_transport_lost_draws
    wall = time.time() - t0
    print(f"[mopup] dispatched {total_calls} draws in {wall:.0f}s: {dict(mop_tally)}", flush=True)

    # Per-class outcome (items are single-class by construction today; a
    # future mixed item reports under "mixed" rather than mis-attributing).
    class_out: dict[str, Counter] = {}
    for item, n in reissue.items():
        out = class_out.setdefault(item_class[item], Counter())
        out["items"] += 1
        out["draws_reissued"] += n
        out["draws_kept"] += mop_kept_counts.get(item, 0)
        out["draws_transport_lost_again"] += mop_transport.get(item, 0)
    for cls, out in sorted(class_out.items()):
        print(f"[mopup] class {cls}: {dict(out)}", flush=True)

    # Merge: batch kept draws + main-pass sync mean + mop-up mean (extends the
    # main() merge shape; untouched items reproduce their prior value exactly).
    merged: dict[str, float | None] = {}
    for item, good in kept.items():
        vals = list(good)
        s = sync_scores.get(item)
        if s is not None:
            vals.append(float(s))
        m = mop_scores.get(item)
        if m is not None:
            vals.append(float(m))
        merged[item] = float(sum(vals) / len(vals)) if vals else None
    n_scored = sum(1 for v in merged.values() if v is not None)
    n_items_recovered = sum(1 for i in unscored if merged[i] is not None)
    print(
        f"[mopup] merged item coverage {n_scored}/{len(merged)} "
        f"({n_scored / len(merged):.1%}); recovered {n_items_recovered} of "
        f"{len(unscored)} unscored items; {len(unscored) - n_items_recovered} remain",
        flush=True,
    )

    # Rebuild the DV. transport_split is PRESERVED from the live file and
    # extended (never dropped); the split rewrite (full -> eval) is re-applied
    # by scripts/issue1739_evilood_dv_split.py, run right after this script.
    dv_path_prior = dv_build.dv_dataset_path(args.out_root, BEHAVIOR)
    prior_meta = json.loads(dv_path_prior.read_text()).get("judge_meta", {})
    prior_split = str(prior_meta.get("transport_split", ""))
    if not prior_split:
        raise RuntimeError(f"{dv_path_prior} has no judge_meta.transport_split to preserve")
    mop_note = (
        f"MOP-UP r3 ({time.strftime('%Y-%m-%d')}): re-issued {total_calls} residual draws on "
        f"the SYNC path at the identical instrument — {plan['transport_draws']} transport-lost "
        f"draws (529 exhaustion, rule 24; incl. {plan['scored_items_topped_up']} already-scored "
        f"items topped up) and {plan['stochastic_parse_draws']} stochastic parse-fail draws on "
        f"the {plan['unscored_items_targeted']} still-unscored items that carried retriable "
        f"draws. {mop_tally['draws_kept']} draws scored; item coverage now "
        f"{n_scored}/{len(merged)}. {n_refusal_only} items remain unscored because every draw "
        f"is a rubric-REFUSAL verdict (the judged completion itself refused) — a rule-9 "
        f"content verdict, not retriable, never coerced."
    )
    judge_meta = dict(prior_meta)
    judge_meta.update(
        {
            "judge_model": JUDGE_MODEL,
            "judge_max_tokens": args.max_tokens,
            "judge_temperature": temperature,
            "n_draws": N_JUDGE_DRAWS,
            "rubric": "trait_eval_prompt",
            "transport_split": prior_split + " " + mop_note,
            "mopup": {
                "reissued_calls": total_calls,
                "per_class": {cls: dict(out) for cls, out in sorted(class_out.items())},
                "mop_draw_tally": dict(mop_tally),
                "unscored_items_before": len(unscored),
                "unscored_items_after": len(unscored) - n_items_recovered,
                "rubric_refusal_only_items": n_refusal_only,
                "transport_lost_after_mopup_draws": sum(mop_transport.values()),
                "path": "sync (forced, threshold_base)",
            },
        }
    )
    contexts_meta = _load_contexts_meta(args.stage_root)
    dv_rows = dv_build.build_labeling_dv(
        merged,
        n_draws=N_JUDGE_DRAWS,
        contexts_meta=contexts_meta,
        per_item_transport_losses={i: n for i, n in mop_transport.items() if n},
    )
    dv_path = dv_build.write_dv_dataset(
        dv_rows,
        out_root=args.out_root,
        behavior=BEHAVIOR,
        judge_payload_meta=judge_meta,
        git_commit=commit_string(git_provenance()),
    )
    print(json.dumps({"dv_dataset_path": str(dv_path), "n_rows": len(dv_rows)}, indent=1))
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
