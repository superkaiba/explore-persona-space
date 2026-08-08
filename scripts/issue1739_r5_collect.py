#!/usr/bin/env python3
"""#1739 Result 5 / Result 3-R^2: gate + collect the composition-ladder reads.

Two modes over the ladder legs' ``map_diagnostics.json`` files.

``--gate`` is the MODULE-BINDING check. The prior Job B pilot is void because it
bound the pre-fix module at process start while the v426 identity-bias fix landed
24 minutes later; its ``eval_rung.per_layer[i]`` rows carry only
``{layer_idx, r2_eval_rung, knn}``. Disk presence of the fixed file proves
nothing about what a RUNNING process bound, so the gate reads the run's OWN
output and requires ``r2_identity_bias`` under BOTH ``eval_rung.per_layer[i]``
AND ``eval_rung.per_rung[<rung>].per_layer[i]``. Exit 0 = fan out; exit 2 = stop.

``--collect`` emits the deliverable: for every (behavior, variant, config,
eval setting) cell, all THREE reads side by side -- R^2, the identity+learned-bias
baseline, and kNN retrieval (euclidean + cosine, with the chance rate the helper
itself reports). Reads are summarised per layer as best-layer and mean, and the
map's own U-pool holdout is carried alongside the eval distributions so the
on-distribution and off-distribution reads sit in one table.

The eval settings are whatever rungs the leg actually produced -- the point of
the round is the per-setting cross, so nothing is hard-coded to a rung list.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# compose<size>_fu<f_u>_fl<f_l>_L<anchor>  -- the ladder's SWAP rungs
_LABEL_RE = re.compile(r"compose(?P<size>\d+)_fu(?P<f_u>[\d.]+)_fl(?P<f_l>[\d.]+)_L(?P<anchor>\d+)")
# unionall<size>_L<anchor>  -- the ADD config: deliberately unmatched pool size,
# so it carries no f_u/f_l and must never be plotted as a ladder rung.
_UNION_RE = re.compile(r"unionall(?P<size>\d+)_L(?P<anchor>\d+)")


def _finite_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _best(rows: list[dict], key: str) -> tuple[int | None, float | None]:
    """(layer_idx, value) of the highest finite `key`, or (None, None)."""
    best_i, best_v = None, None
    for r in rows:
        v = r.get(key)
        if v is None or not math.isfinite(v):
            continue
        if best_v is None or v > best_v:
            best_i, best_v = r.get("layer_idx"), v
    return best_i, best_v


def _knn_summary(rows: list[dict]) -> dict:
    """acc@1 at the best-R^2 layer plus the pooled max, per metric, with chance."""
    out: dict = {}
    for metric in ("euclidean", "cosine"):
        accs, chance = [], None
        for r in rows:
            knn = (r.get("knn") or {}).get(metric)
            if not knn:
                continue
            a = (knn.get("acc_at_k") or {}).get("1")
            if a is not None:
                accs.append(float(a))
            if chance is None:
                chance = (knn.get("chance_at_k") or {}).get("1")
        if accs:
            out[metric] = {
                "acc_at_1_max": max(accs),
                "acc_at_1_mean": _finite_mean(accs),
                "chance_at_1": chance,
                "x_chance": (max(accs) / chance) if chance else None,
                "n_layers": len(accs),
            }
    return out


def _block_reads(rows: list[dict]) -> dict:
    """All three reads over one per-layer block."""
    bi, bv = _best(rows, "r2_eval_rung")
    ib_vals = [r.get("r2_identity_bias") for r in rows]
    has_ib = any(v is not None for v in ib_vals)
    ib_at_best = next(
        (r.get("r2_identity_bias") for r in rows if r.get("layer_idx") == bi),
        None,
    )
    return {
        "n_layers": len(rows),
        "r2_best_layer": bi,
        "r2_best": bv,
        "r2_mean": _finite_mean([r.get("r2_eval_rung") for r in rows]),
        "r2_identity_bias_present": has_ib,
        "r2_identity_bias_at_best_layer": ib_at_best,
        "r2_identity_bias_mean": _finite_mean(ib_vals) if has_ib else None,
        "knn": _knn_summary(rows),
    }


def _iter_map_keys(diag_path: Path):
    diag = json.loads(diag_path.read_text())
    for map_key, payload in diag.items():
        if isinstance(payload, dict):
            yield map_key, payload


def gate(paths: list[Path]) -> int:
    """Module-binding gate: the identity-bias key must be in the run's OWN output."""
    ok = True
    for p in paths:
        if not p.exists():
            print(f"GATE FAIL: {p} does not exist")
            ok = False
            continue
        found_any = False
        for map_key, payload in _iter_map_keys(p):
            er = payload.get("eval_rung")
            if not isinstance(er, dict):
                print(f"GATE FAIL: {p} [{map_key}] has no eval_rung block (was --transfer passed?)")
                ok = False
                continue
            found_any = True
            pl = er.get("per_layer") or []
            if not pl or "r2_identity_bias" not in (pl[0] or {}):
                print(
                    f"GATE FAIL: {p} [{map_key}] eval_rung.per_layer lacks "
                    "r2_identity_bias -- the process bound PRE-FIX code"
                )
                ok = False
            per_rung = er.get("per_rung") or {}
            if not per_rung:
                print(
                    f"GATE FAIL: {p} [{map_key}] has no eval_rung.per_rung "
                    "(was --eval-rung-knn passed?)"
                )
                ok = False
            for rung, rv in per_rung.items():
                rpl = rv.get("per_layer") or []
                if rpl and "r2_identity_bias" not in (rpl[0] or {}):
                    print(
                        f"GATE FAIL: {p} [{map_key}] per_rung[{rung}].per_layer "
                        "lacks r2_identity_bias -- PRE-FIX code"
                    )
                    ok = False
        if not found_any:
            print(f"GATE FAIL: {p} produced no usable map keys")
            ok = False
    if ok:
        print(
            "GATE PASS: r2_identity_bias present under eval_rung.per_layer AND "
            "eval_rung.per_rung[*].per_layer in every checked leg"
        )
    return 0 if ok else 2


def collect(legs_root: Path, out_path: Path) -> int:
    cells = []
    for diag_path in sorted(legs_root.glob("*/map_diagnostics.json")):
        slug = diag_path.parent.name
        for map_key, payload in _iter_map_keys(diag_path):
            variant, _, label = map_key.partition("|")
            m = _LABEL_RE.match(label)
            mu = _UNION_RE.match(label)
            if m:
                cfg = {
                    "config": f"ladder_fu{m.group('f_u')}",
                    "f_u": float(m.group("f_u")),
                    "f_l": float(m.group("f_l")),
                    "pool_size": int(m.group("size")),
                    "anchor_l": int(m.group("anchor")),
                    "semantics": "swap-matched",
                }
            elif mu:
                # ADD config: pool deliberately unmatched -- no f_u/f_l, and it
                # must never be plotted as a point on the ladder.
                cfg = {
                    "config": "union_all",
                    "f_u": None,
                    "f_l": None,
                    "pool_size": int(mu.group("size")),
                    "anchor_l": int(mu.group("anchor")),
                    "semantics": "add-unmatched",
                }
            else:
                cfg = {
                    "config": None,
                    "f_u": None,
                    "f_l": None,
                    "pool_size": None,
                    "anchor_l": None,
                    "semantics": None,
                    "raw_label": label,
                }
            base = {
                "leg": slug,
                "variant": variant,
                "map_label": label,
                **cfg,
                "n_train": payload.get("n_train"),
                "n_holdout": payload.get("n_holdout"),
                "w_fit_rows": payload.get("w_fit_rows"),
                "solver": payload.get("solver"),
                "map_source": payload.get("map_source"),
            }
            # (a) the map's OWN U-pool holdout (in-distribution reference)
            holdout_rows = payload.get("per_layer") or []
            if holdout_rows:
                hb, hv = _best(holdout_rows, "r2_map")
                ib = next(
                    (r.get("r2_identity_bias") for r in holdout_rows if r.get("layer_idx") == hb),
                    None,
                )
                cells.append(
                    {
                        **base,
                        "eval_setting": "u_pool_holdout",
                        "n_rows": payload.get("n_holdout"),
                        "n_layers": len(holdout_rows),
                        "r2_best_layer": hb,
                        "r2_best": hv,
                        "r2_mean": _finite_mean([r.get("r2_map") for r in holdout_rows]),
                        "r2_identity_bias_present": any(
                            r.get("r2_identity_bias") is not None for r in holdout_rows
                        ),
                        "r2_identity_bias_at_best_layer": ib,
                        "r2_identity_bias_mean": _finite_mean(
                            [r.get("r2_identity_bias") for r in holdout_rows]
                        ),
                        "knn": _knn_summary(holdout_rows),
                    }
                )
            er = payload.get("eval_rung")
            if not isinstance(er, dict):
                continue
            # (b) the pooled eval split
            pooled = er.get("per_layer") or []
            if pooled:
                cells.append(
                    {
                        **base,
                        "eval_setting": "eval_pooled",
                        "n_rows": er.get("n_eval_rows"),
                        **_block_reads(pooled),
                    }
                )
            # (c) every eval distribution separately -- the deliverable
            for rung, rv in (er.get("per_rung") or {}).items():
                rows = rv.get("per_layer") or []
                if not rows:
                    continue
                cells.append(
                    {
                        **base,
                        "eval_setting": rung,
                        "n_rows": rv.get("n_rows"),
                        "knn_skipped_small_pool": rv.get("knn_skipped_small_pool"),
                        **_block_reads(rows),
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n_cells": len(cells), "cells": cells}, indent=1))
    print(f"[collect] {len(cells)} cells -> {out_path}")

    missing_ib = [c for c in cells if not c.get("r2_identity_bias_present")]
    if missing_ib:
        print(f"[collect] WARNING: {len(missing_ib)} cell(s) lack the identity-bias read")
    settings = sorted({c["eval_setting"] for c in cells})
    configs = sorted({c["f_u"] for c in cells if c["f_u"] is not None})
    print(f"[collect] eval settings: {settings}")
    print(f"[collect] ladder rungs (f_u): {configs}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gate", nargs="+", type=Path, help="map_diagnostics.json path(s) to gate before fan-out"
    )
    ap.add_argument("--collect", type=Path, help="legs root holding <slug>/map_diagnostics.json")
    ap.add_argument("--out", type=Path, default=Path("ladder_reads.json"))
    args = ap.parse_args(argv)
    if args.gate:
        return gate(args.gate)
    if args.collect:
        return collect(args.collect, args.out)
    ap.error("pass --gate or --collect")
    return 2


if __name__ == "__main__":
    sys.exit(main())
