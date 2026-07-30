"""Merge the per-behavior #779-963k-map readouts into one comparison.json + summary.

The readout driver runs one behavior per process (bounded memory, parallel
behaviors). This merges those outputs into the single deliverable and derives a
compact headline: per (behavior, variant, layer, rung), the 963k arms next to
the recomputed arm6 / raw / shuffled arms on the PRIMARY r_B source, plus the
best-963k-minus-arm6 delta that answers the task's question directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAP963K = ("map963k_ridge", "map963k_mlp_w8192", "map963k_mlp_w32768")
DASH = "—"

# Committed-arm -> recomputed-arm pairs for the INDICATIVE anchor.
# arm1_ctx_e1 (NOT arm2_ctx_native) is the raw E1 projection <z, r_B>:
# arm2 is the context-NATIVE-direction arm, a different construct.
ANCHOR_PAIRS = (
    ("arm1_ctx_e1", "raw_proj"),
    ("arm6_map_proj_e1", "map_i1739_ufull"),
    ("arm13_shuffled_map", "map_i1739_shuffled"),
    ("arm11_oracle_proj", "oracle_proj"),
)

# Why the anchor is INDICATIVE and not a reproduction gate: #1739's committed
# rho_per_layer is computed per-CELL, over that cell's fold- and
# budget_l-scoped eval rows (fold_scheme group-roundrobin-k5, one row per
# (u_rung, budget_l, draw, seed)), whereas this readout computes ONE pooled
# Spearman over every DV-bearing context in a rung. The two therefore differ
# by eval-set construction even when the underlying score is identical, so a
# nonzero gap is expected and is NOT evidence of a mis-applied map. The
# headline comparison never depends on the anchor: every arm it compares is
# computed in ONE process on the SAME contexts, DV, r_B and metric.
ANCHOR_COMPARABILITY = (
    "INDICATIVE ONLY — committed rho is per-cell (fold- and budget_l-scoped eval "
    "rows); recomputed rho is pooled over all DV-bearing contexts in the rung. A "
    "nonzero gap is expected from eval-set construction alone. Additionally, the "
    "shuffled-map control uses an independent random permutation draw here, so its "
    "gap carries no information at all."
)


def merge(paths: list[Path]) -> dict:
    out: dict = {"meta": None, "behaviors": {}}
    for p in paths:
        payload = json.loads(p.read_text())
        if out["meta"] is None:
            out["meta"] = payload["meta"]
        out["behaviors"].update(payload["behaviors"])
    out["meta"]["merged_from"] = [str(p) for p in paths]
    return out


def headline(merged: dict) -> list[dict]:
    """Per (behavior, variant, layer, rung): 963k vs arm6 on the primary r_B.

    Each cell also carries ``dv_std`` and ``dv_degenerate``. A rung whose DV is
    near-constant (evil/hhrt: dv_std 0.9 against 26.3 on train) cannot separate
    ANY arm — its own oracle arm reads ~0.07 — so deltas there are noise and are
    flagged rather than silently averaged into a claim.
    """
    rows: list[dict] = []
    for behavior, b in merged["behaviors"].items():
        primary = sorted({r["r_b_source"] for r in b["rows"] if r.get("ci_computed")})
        primary_src = primary[0] if primary else None
        dv_by_rung = {r["eval_rung"]: r["dv_std"] for r in b["rows"]}
        dv_max = max(dv_by_rung.values()) if dv_by_rung else 0.0
        by_cell: dict[tuple, dict] = {}
        for r in b["rows"]:
            if r["r_b_source"] != primary_src:
                continue
            key = (r["variant"], r["layer"], r["eval_rung"])
            by_cell.setdefault(
                key,
                {
                    "behavior": behavior,
                    "variant": r["variant"],
                    "layer": r["layer"],
                    "eval_rung": r["eval_rung"],
                    "n_contexts": r["n_contexts"],
                    "r_b_source": primary_src,
                    "dv_std": r["dv_std"],
                    "dv_degenerate": bool(dv_max > 0 and r["dv_std"] < 0.15 * dv_max),
                    "arms": {},
                },
            )
            by_cell[key]["arms"][r["arm"]] = {"rho": r["rho"], "ci95": r["ci95"]}
        for cell in by_cell.values():
            arms = cell["arms"]
            best = None
            for a in MAP963K:
                rho = arms.get(a, {}).get("rho")
                if rho is None:
                    continue
                if best is None or abs(rho) > abs(arms[best]["rho"]):
                    best = a
            a6 = arms.get("map_i1739_ufull", {}).get("rho")
            raw = arms.get("raw_proj", {}).get("rho")
            cell["best_963k_arm"] = best
            cell["best_963k_rho"] = arms[best]["rho"] if best else None
            cell["best_963k_ci95"] = arms[best]["ci95"] if best else None
            cell["arm6_rho"] = a6
            cell["raw_rho"] = raw
            cell["shuffled_i1739_rho"] = arms.get("map_i1739_shuffled", {}).get("rho")
            cell["shuffled_963k_rho"] = arms.get("map963k_ridge_shuffled", {}).get("rho")
            cell["oracle_rho"] = arms.get("oracle_proj", {}).get("rho")
            b6 = cell["best_963k_rho"]
            cell["delta_963k_minus_arm6"] = b6 - a6 if b6 is not None and a6 is not None else None
            cell["delta_963k_minus_raw"] = b6 - raw if b6 is not None and raw is not None else None
            rows.append(cell)
    rows.sort(key=lambda r: (r["behavior"], r["variant"], r["layer"], r["eval_rung"]))
    return rows


def reanchor(merged: dict, arms_root: Path) -> dict:
    """Recompute the indicative anchor against the CORRECT committed arm names.

    The readout's own anchor pass used ``arm2_ctx_native`` as the raw-projection
    counterpart; that arm is the context-NATIVE-direction construct, not the E1
    projection. This pass redoes the anchor against ``arm1_ctx_e1`` (and adds the
    oracle pair), post hoc, so no readout re-run is needed.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from issue1739_map963k_readout import committed_arm_rho

    out: dict[str, list[dict]] = {}
    for behavior, b in merged["behaviors"].items():
        arms_json = arms_root / behavior / "arm_results" / "all_arms_spearman.json"
        anchors: list[dict] = []
        primary = sorted({r["r_b_source"] for r in b["rows"] if r.get("ci_computed")})
        primary_src = primary[0] if primary else None
        for committed, mine in ANCHOR_PAIRS:
            for variant in sorted({r["variant"] for r in b["rows"]}):
                for layer in sorted({r["layer"] for r in b["rows"]}):
                    per_rung = committed_arm_rho(arms_json, committed, variant, layer)
                    for rung, c in per_rung.items():
                        got = [
                            r
                            for r in b["rows"]
                            if r["variant"] == variant
                            and r["layer"] == layer
                            and r["arm"] == mine
                            and r["eval_rung"] == rung
                            and r["r_b_source"] == primary_src
                        ]
                        if not got or c.get("rho_frozen") is None:
                            continue
                        anchors.append(
                            {
                                "behavior": behavior,
                                "variant": variant,
                                "layer": layer,
                                "eval_rung": rung,
                                "committed_arm": committed,
                                "recomputed_arm": mine,
                                "committed_rho": c["rho_frozen"],
                                "committed_source": c.get("source"),
                                "committed_budget_l": c.get("budget_l"),
                                "recomputed_rho": got[0]["rho"],
                                "abs_gap": abs(got[0]["rho"] - c["rho_frozen"]),
                                "sign_agrees": (got[0]["rho"] >= 0) == (c["rho_frozen"] >= 0),
                            }
                        )
        out[behavior] = anchors
    return out


def prefix_resolution(slice_root: Path, behaviors: list[str], layer: int = 19) -> dict:
    """How many DISTINCT prefix / context states each behavior's eval set has.

    Load-bearing for reading any prefix_end row: the prefix is shared across
    every context that uses the same persona, so ``prefix_end`` is a COARSE
    CATEGORICAL score with as many levels as there are distinct prefixes — not a
    per-context continuous read. Measured here rather than assumed:
    sycophancy's eval prefix is EXACTLY constant (1 level ⇒ every prefix_end rho
    is a tie-breaking artifact and carries no information), and evil's takes ~8
    levels (its persona set), against ~hundreds for context_end.
    """
    import numpy as np

    out: dict[str, dict] = {}
    for b in behaviors:
        entry: dict = {"layer_probed": layer}
        for kind in ("prefix_end", "context_end"):
            shards = sorted((slice_root / b).glob(f"{kind}_L{layer:02d}_shard*.npy"))[:4]
            if not shards:
                continue
            a = np.concatenate([np.load(s) for s in shards]).astype(np.float64)
            entry[kind] = {
                "rows_probed": int(len(a)),
                "per_dim_sd_mean": float(a.std(axis=0).mean()),
                "n_distinct_states": int(len(np.unique(np.round(a, 3), axis=0))),
            }
        pe = entry.get("prefix_end", {})
        entry["prefix_is_constant"] = bool(pe.get("n_distinct_states", 0) == 1)
        entry["prefix_note"] = (
            "prefix_end is CONSTANT across eval contexts — every prefix_end rho is a "
            "rank-tie artifact and must not be read as a result"
            if entry["prefix_is_constant"]
            else (
                f"prefix_end takes {pe.get('n_distinct_states')} distinct states over "
                f"{pe.get('rows_probed')} probed rows — a coarse categorical read "
                "(one level per persona prefix), not a per-context continuous one"
            )
        )
        out[b] = entry
    return out


def _num(v, width: int = 8, prec: int = 3) -> str:
    return f"{v:>{width}.{prec}f}" if isinstance(v, (int, float)) else f"{DASH:>{width}}"


def fmt_table(rows: list[dict]) -> str:
    hdr = (
        f"{'behavior':<14}{'variant':<13}{'L':>3} {'rung':<11}{'n':>6}"
        f"{'oracle':>8}{'raw':>8}{'arm6':>8}{'shuf6':>8}{'963k':>8}"
        f"{'D-arm6':>9}{'D-raw':>8}  best-963k"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['behavior']:<14}{r['variant']:<13}{r['layer']:>3} {r['eval_rung']:<11}"
            f"{r['n_contexts']:>6}"
            f"{_num(r['oracle_rho'])}{_num(r['raw_rho'])}{_num(r['arm6_rho'])}"
            f"{_num(r['shuffled_i1739_rho'])}{_num(r['best_963k_rho'])}"
            f"{_num(r['delta_963k_minus_arm6'], 9)}{_num(r['delta_963k_minus_raw'])}"
            f"  {(r['best_963k_arm'] or DASH).replace('map963k_', '')}"
            f"{'   [DV~flat]' if r.get('dv_degenerate') else ''}"
        )
    return "\n".join(lines)


def fmt_recon(merged: dict) -> str:
    """Map reconstruction of the ACTUAL answer summary, per (behavior, variant, layer)."""
    hdr = (
        f"{'behavior':<14}{'variant':<13}{'L':>3}  {'map':<26}"
        f"{'cos':>8}{'R2':>10}{'|pred|':>9}{'|t1|':>8}"
    )
    lines = [hdr, "-" * len(hdr)]
    for behavior, b in merged["behaviors"].items():
        for r in sorted(b.get("recon", []), key=lambda x: (x["variant"], x["layer"], x["map"])):
            lines.append(
                f"{behavior:<14}{r['variant']:<13}{r['layer']:>3}  {r['map']:<26}"
                f"{_num(r.get('cosine_mean'))}{_num(r.get('r2'), 10, 2)}"
                f"{_num(r.get('pred_norm_mean'), 9, 1)}{_num(r.get('actual_t1_norm_mean'), 8, 1)}"
            )
    return "\n".join(lines)


def fmt_parity(anchors: dict) -> str:
    """Indicative recomputed-vs-committed anchor (see ANCHOR_COMPARABILITY)."""
    hdr = (
        f"{'behavior':<14}{'variant':<13}{'L':>3} {'rung':<11}{'committed arm':<20}"
        f"{'committed':>10}{'recomputed':>12}{'|gap|':>8}"
    )
    lines = [hdr, "-" * len(hdr)]
    for behavior, rows in anchors.items():
        for a in rows:
            lines.append(
                f"{behavior:<14}{a['variant']:<13}{a['layer']:>3} {a['eval_rung']:<11}"
                f"{a['committed_arm']:<20}"
                f"{_num(a.get('committed_rho'), 10)}{_num(a.get('recomputed_rho'), 12)}"
                f"{_num(a.get('abs_gap'))}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument(
        "--slice-root", type=Path, default=Path("data/issue_1739/hf_dl/evalslice")
    )
    args = ap.parse_args(argv)

    present = [p for p in args.inputs if p.is_file()]
    if not present:
        raise SystemExit(f"no readout inputs present among {args.inputs}")
    merged = merge(present)
    merged["headline"] = headline(merged)
    merged["parity_anchor_indicative"] = reanchor(merged, args.arms_root)
    merged["prefix_resolution"] = prefix_resolution(
        args.slice_root, sorted(merged["behaviors"])
    )
    merged["meta"]["parity_anchor_comparability"] = ANCHOR_COMPARABILITY
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2))
    print(f"wrote {args.out}  (behaviors: {sorted(merged['behaviors'])})")
    print()
    print("=== HEADLINE: 963k map vs #1739 arm6, primary r_B, per rung ===")
    print(fmt_table(merged["headline"]))
    print()
    print("=== REUSE VALIDITY: map reconstruction of actual answer summary t1 ===")
    print(fmt_recon(merged))
    print()
    print("=== PREFIX RESOLUTION: how many distinct prefix states the eval set has ===")
    for b, e in merged["prefix_resolution"].items():
        pe, ce = e.get("prefix_end", {}), e.get("context_end", {})
        print(
            f"  {b:<14} prefix_end {pe.get('n_distinct_states')} distinct "
            f"(sd {pe.get('per_dim_sd_mean', 0):.4f})   "
            f"context_end {ce.get('n_distinct_states')} distinct "
            f"(sd {ce.get('per_dim_sd_mean', 0):.4f})"
        )
        print(f"                 -> {e['prefix_note']}")
    print()
    print("=== INDICATIVE ANCHOR: recomputed vs committed (NOT a reproduction gate) ===")
    print(ANCHOR_COMPARABILITY)
    print(fmt_parity(merged["parity_anchor_indicative"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
