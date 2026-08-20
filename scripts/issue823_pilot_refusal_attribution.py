#!/usr/bin/env python3
"""Recompute the #823 ladder pilot's refusal attribution from the staged records.

The pilot (300 contexts / 896 (context, persona) pairs, `issue823_ladder_gen.py
--smoke --n-contexts 300`) is the gating measurement for the production
generation wave: its per-context drop rate decides whether the plan's
mask-integrity kill can be satisfied, and its per-persona breakdown is the
evidence base any refusal-attrition budget must be derived from.

That attribution previously existed only as prose in an `epm:progress` marker.
This script recomputes it from the durable staged records so the numbers a plan
threshold cites are machine-readable, re-derivable, and committed.

Reads the per-persona record files the generator staged (also mirrored on the HF
data repo under `.../raw_completions/ladder/`), and emits one JSON with:

  * validity / stop_reason cross-tab (they disagree: some refusals are labeled
    `empty`, so validity-keyed counts UNDERCOUNT refusals -- reported, never
    silently reconciled)
  * per-persona refusal rate over that persona's realized pair count
  * per-context refused-pair histogram + the direct per-context drop rate
  * a Wilson 95% interval on the per-context drop rate, projected to the
    production context count
  * roster-exclusion counterfactuals: the projected drop count after excluding
    each prefix of the personas ranked by refusal count

Intersected-mask semantics: a context is DROPPED when ANY of its (context,
persona) pairs is invalid, because the fit mask is intersected across arms so
every arm is scored on identical contexts. The per-CONTEXT rate -- not the
per-pair rate -- is therefore what the mask kill sees.

Usage:
    uv run python scripts/issue823_pilot_refusal_attribution.py \
        --stage-dir /tmp/issue-823-pilot300/hf_stage/ladder \
        --n-production-contexts 5000 \
        --out eval_results/issue_823/inconsistent-origin-persona-ladder/pilot300_refusal_attribution.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.provenance import (
    as_metadata_dict,
    git_provenance,
)

# Records whose stop_reason marks a content-side decline. A refusal is a content
# decision at fixed sampling params, so re-submitting the identical request is
# not expected to recover it -- these are NOT transport failures and are
# deliberately not retried by the generator.
REFUSAL_STOP_REASONS = frozenset({"refusal"})

# Registered ladder rungs. Used only as a fallback: the arm set is derived from
# the records' own `arms` field when present, so a roster/ladder change cannot
# silently leave this constant stale.
ARM_K_VALUES_FALLBACK = (1, 2, 4, 8, 16)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% default).

    Preferred over the normal approximation here because the counts are small
    (tens of events) and Wilson stays inside [0, 1] and does not degenerate at
    k == 0. Returns (lo, hi) as proportions.
    """
    if n <= 0:
        raise ValueError(f"wilson_interval needs n > 0, got {n}")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    # The Wilson interval analytically contains p_hat, and is exactly [0, hi] at
    # k=0 and [lo, 1] at k=n. Float error violates both by ~1 ULP (measured:
    # hi = 0.9999999999999999 at k=n=10; lo = 8.7e-19 at k=0), which would leave
    # an interval that excludes its own point estimate. Clamp to the analytic
    # guarantee rather than letting a threshold be derived from a bound that
    # cannot contain the estimate it describes.
    return min(lo, p), max(hi, p)


def load_records(stage_dir: Path) -> list[dict[str, Any]]:
    """Load every persona record file in the staged ladder dir.

    Fails loud on an empty selection: an empty record set here would silently
    produce a zero-refusal attribution that reads as a clean pilot.
    """
    files = sorted(stage_dir.glob("persona*_seed*.json"))
    if not files:
        raise RuntimeError(
            f"no persona*_seed*.json under {stage_dir} -- refusing to emit an "
            "attribution from an empty record set"
        )
    records: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["records"] if isinstance(payload, dict) else payload
        if not rows:
            raise RuntimeError(f"{path.name} holds zero records -- unexpected for a staged file")
        records.extend(rows)
    return records


def attribute(records: list[dict[str, Any]], n_production: int) -> dict[str, Any]:
    """Compute the refusal attribution over the pilot's realized records."""
    n_pairs = len(records)

    validity_counts = Counter(r.get("validity") for r in records)
    stop_counts = Counter(r.get("stop_reason") for r in records)

    # Cross-tab: validity and stop_reason disagree in the pilot, so record both
    # rather than trusting either alone as "the" invalid count.
    cross = Counter((str(r.get("validity")), str(r.get("stop_reason"))) for r in records)

    refused = [r for r in records if r.get("stop_reason") in REFUSAL_STOP_REASONS]

    per_persona_total: Counter[int] = Counter(int(r["persona_idx"]) for r in records)
    per_persona_refused: Counter[int] = Counter(int(r["persona_idx"]) for r in refused)

    persona_name: dict[int, str] = {}
    for r in records:
        idx = int(r["persona_idx"])
        name = r.get("persona_name") or r.get("persona") or ""
        if name and idx not in persona_name:
            persona_name[idx] = str(name)

    # Contexts lost per persona-exclusion counterfactual. A context is dropped
    # when any of its pairs is refused, so exclusions are evaluated by removing
    # that persona's refusals from the union.
    refused_ctx_by_persona: dict[int, set[int]] = defaultdict(set)
    for r in refused:
        refused_ctx_by_persona[int(r["persona_idx"])].add(int(r["context_id"]))

    all_ctx = {int(r["context_id"]) for r in records}
    n_ctx = len(all_ctx)
    if n_ctx == 0:
        raise RuntimeError("zero contexts in the pilot records")

    dropped_ctx = set().union(*refused_ctx_by_persona.values()) if refused_ctx_by_persona else set()

    # Per-context refused-pair histogram: singleton-dominated means specific
    # personas refuse, not that specific questions are refusable under any persona.
    refusals_per_ctx = Counter(int(r["context_id"]) for r in refused)
    hist = Counter(refusals_per_ctx.values())

    rate = len(dropped_ctx) / n_ctx
    lo, hi = wilson_interval(len(dropped_ctx), n_ctx)

    # Two orderings, because they give DIFFERENT exclusion paths and quoting one
    # as "the" counterfactual is how a wrong claim gets made: ranking by refusal
    # COUNT front-loads high-volume low-rate personas (p00 serves every context),
    # ranking by RATE front-loads the pathological ones.
    ranked_by_count = [idx for idx, _ in per_persona_refused.most_common()]
    ranked_by_rate = sorted(
        per_persona_refused,
        key=lambda i: (-(per_persona_refused[i] / per_persona_total[i]), i),
    )

    # Arm k uses personas {0..k-1} under persona(i,k) = i mod k, so excluding
    # persona p SHRINKS every arm with k > p -- and the ladder's independent
    # variable IS the number of distinct answer-origin personas. Roster exclusion
    # is therefore confounded with the manipulation, not a free lever.
    derived_ks: set[int] = set()
    for r in records:
        for k in r.get("arms") or ():
            derived_ks.add(int(k))
    arm_ks = sorted(derived_ks) if derived_ks else sorted(ARM_K_VALUES_FALLBACK)

    def _arm_impact(excluded: list[int]) -> dict[str, Any]:
        ex = set(excluded)
        affected, destroyed = [], []
        for k in arm_ks:
            members = set(range(k))
            lost = members & ex
            if lost:
                affected.append(k)
                if lost == members:
                    destroyed.append(k)
        return {
            "arms_shrunk": affected,
            "arms_destroyed": destroyed,
            "changes_manipulated_variable": bool(affected),
        }

    counterfactuals = []
    for label, ranked in (
        ("by_refusal_count", ranked_by_count),
        ("by_refusal_rate", ranked_by_rate),
    ):
        for cut in range(len(ranked) + 1):
            excluded = ranked[:cut]
            remaining = set()
            for idx, ctxs in refused_ctx_by_persona.items():
                if idx not in excluded:
                    remaining |= ctxs
            r_rate = len(remaining) / n_ctx
            counterfactuals.append(
                {
                    "ordering": label,
                    "excluded_personas": excluded,
                    "excluded_names": [persona_name.get(i, "") for i in excluded],
                    "dropped_contexts_pilot": len(remaining),
                    "drop_rate": r_rate,
                    "projected_dropped_at_production": round(r_rate * n_production),
                    **_arm_impact(excluded),
                }
            )

    # Per-(arm, persona) cap-hit, keyed on stop_reason. The plan cites the
    # over-trigger cell count and the worst cells, so they belong in the
    # artifact rather than an ad-hoc re-tally: a cited figure that lives only in
    # a shell one-liner is not re-derivable.
    cap_cells: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        p = int(r["persona_idx"])
        hit = r.get("stop_reason") == "max_tokens"
        for k in r.get("arms") or ():
            cell = cap_cells[(int(k), p)]
            cell[1] += 1
            if hit:
                cell[0] += 1

    CAP_TRIGGER = 0.02
    cap_rows = [
        {
            "arm_k": k,
            "persona_idx": p,
            "persona_name": persona_name.get(p, ""),
            "n_cap_hit": h,
            "n_pairs": t,
            "cap_hit_fraction": h / t,
            "over_trigger": (h / t) > CAP_TRIGGER,
        }
        for (k, p), (h, t) in sorted(cap_cells.items())
        if t
    ]
    cap_hit_pairs = sum(1 for r in records if r.get("stop_reason") == "max_tokens")

    return {
        "cap_hit": {
            "trigger_fraction": CAP_TRIGGER,
            "n_cap_hit_pairs": cap_hit_pairs,
            "cap_hit_pair_fraction": cap_hit_pairs / n_pairs,
            "n_cells": len(cap_rows),
            "n_cells_over_trigger": sum(1 for r in cap_rows if r["over_trigger"]),
            "arm_weighted_fraction": (
                sum(r["n_cap_hit"] for r in cap_rows) / sum(r["n_pairs"] for r in cap_rows)
                if cap_rows
                else 0.0
            ),
            "note": (
                "measured on FINAL (post-regeneration) records: the generator's own "
                "regen trigger fired and terminated with cells still over it -- "
                "'regen ran' is not 'trigger resolved'"
            ),
            "cells": cap_rows,
        },
        "pairs": {
            "n_pairs": n_pairs,
            "validity_counts": {str(k): v for k, v in sorted(validity_counts.items(), key=str)},
            "stop_reason_counts": {str(k): v for k, v in sorted(stop_counts.items(), key=str)},
            "validity_x_stop_reason": [
                {"validity": v, "stop_reason": s, "n": n} for (v, s), n in sorted(cross.items())
            ],
            "n_refused_by_stop_reason": len(refused),
            "refused_pair_rate": len(refused) / n_pairs,
            "label_disagreement_note": (
                "validity-keyed invalid counts and stop_reason-keyed refusal counts "
                "disagree; a gate keyed on validity labels undercounts refusals"
            ),
        },
        "contexts": {
            "n_contexts": n_ctx,
            "n_dropped_contexts": len(dropped_ctx),
            "drop_rate": rate,
            "wilson95": {"lo": lo, "hi": hi},
            "refused_pairs_per_dropped_context": {str(k): v for k, v in sorted(hist.items())},
            "mask_semantics": (
                "intersected across arms: a context is dropped when ANY of its pairs "
                "is invalid, so the per-CONTEXT rate is what the mask kill sees"
            ),
        },
        "projection": {
            "n_production_contexts": n_production,
            "projected_dropped": round(rate * n_production),
            "projected_dropped_wilson95": [
                round(lo * n_production),
                round(hi * n_production),
            ],
        },
        "per_persona": [
            {
                "persona_idx": idx,
                "persona_name": persona_name.get(idx, ""),
                "n_pairs": per_persona_total[idx],
                "n_refused": per_persona_refused.get(idx, 0),
                "refusal_rate": per_persona_refused.get(idx, 0) / per_persona_total[idx],
                "uniquely_dropped_contexts": len(
                    {c for c in refused_ctx_by_persona.get(idx, set()) if refusals_per_ctx[c] == 1}
                ),
            }
            for idx in sorted(per_persona_total)
        ],
        "roster_exclusion_counterfactuals": counterfactuals,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", type=Path, required=True, help="staged ladder record dir")
    ap.add_argument("--n-production-contexts", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = ap.parse_args()

    records = load_records(args.stage_dir)
    result = attribute(records, args.n_production_contexts)
    result["metadata"] = as_metadata_dict(git_provenance(), phase="pilot-refusal-attribution")
    result["metadata"]["stage_dir"] = str(args.stage_dir)
    result["metadata"]["source"] = "issue823_ladder_gen.py --smoke --n-contexts 300"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    c = result["contexts"]
    p = result["projection"]
    print(
        f"[attribution] pairs={result['pairs']['n_pairs']} refused={result['pairs']['n_refused_by_stop_reason']}"
    )
    print(
        f"[attribution] contexts dropped {c['n_dropped_contexts']}/{c['n_contexts']} "
        f"= {c['drop_rate']:.4f} wilson95=[{c['wilson95']['lo']:.4f}, {c['wilson95']['hi']:.4f}]"
    )
    print(
        f"[attribution] projected at n={p['n_production_contexts']}: "
        f"{p['projected_dropped']} (95% {p['projected_dropped_wilson95']})"
    )
    print(f"[attribution] wrote {args.out}")


if __name__ == "__main__":
    main()
