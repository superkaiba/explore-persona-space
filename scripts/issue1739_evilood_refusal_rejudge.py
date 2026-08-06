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
    args = ap.parse_args()

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
    texts: dict[str, tuple[str, str]] = {}
    for rung_dir in sorted(Path(args.stage_root).glob("*/rollouts/full")):
        for p in rung_dir.glob("*.json"):
            if p.name.startswith("_"):
                continue
            rec = json.loads(p.read_text())
            texts[judging.rollout_item_id(rec["context_id"], int(rec["rollout_k"]))] = (
                rec["query"],
                rec["completion"],
            )

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

    contexts_meta: dict[str, dict] = {}
    for rung_dir in sorted(Path(args.stage_root).glob("*/rollouts/full")):
        for p in rung_dir.glob("*.json"):
            if p.name.startswith("_"):
                continue
            rec = json.loads(p.read_text())
            contexts_meta[rec["context_id"]] = rec

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


if __name__ == "__main__":
    raise SystemExit(main())
