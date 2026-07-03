#!/usr/bin/env python
"""Issue #825 onpolicy-user-turn summarizer -> headline_metrics.json (plan hard-req 6).

Per self-generated user cell: ridge frozen-layer table + selection read + null
band; MLP per-frozen-layer R^2 + fold dispersion + nulls + best-frozen +
per-fold budget-cap hits; deltas vs the matched PARENT Haiku cell (ridge
L19/L26, MLP best-frozen) each with the plan-section-Goal uncertainty read;
n_cell + drop counts; degeneracy-audit metrics incl. u2 token-length vs parent;
anchor deltas; NLL diagnostics (wiring-check own/shuffled + the parent Haiku
band as a reported diagnostic, never a gate — plan MF-B); tr(cov) diversity.

The binding positive-interpretation rule (plan MF-F) is embedded verbatim in
the output so the analyzer carries it forward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

USER_CELLS = [
    "M_instruct_user_chat",
    "M_instruct_user_naturalistic",
    "M_pretrained_user_chat",
    "M_pretrained_user_naturalistic",
]
ANCHOR_CELLS = [
    "M_instruct_assistant_chat",
    "M_instruct_assistant_naturalistic",
    "M_pretrained_assistant_chat",
    "M_pretrained_assistant_naturalistic",
]
MLP_POINT_MARGIN = 0.10
RIDGE_MARGIN = 0.30
ANCHOR_TOL = 0.05
KEEP_RATE_FLOOR = 0.80
DISTINCT_3GRAM_FLOOR = 0.5
PARENT_HAIKU_NLL_BAND = (2.04, 2.64)
INTERPRETATION_RULE = (
    "Binding positive-interpretation rule (plan MF-F): a PASS on any support "
    "criterion licenses ONLY the descriptive provenance-bundle claim — "
    "'self-generated u2 cells are more row-predictable than externally-written "
    "u2 cells.' It does NOT license 'the user-header state encodes the model's "
    "own predicted turn' (self-consistency mechanism); that requires the "
    "cross-model mismatched-context control named in the plan."
)


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _frozen_table(cells_payload: dict) -> dict:
    return (cells_payload.get("selection_symmetric") or {}).get("frozen_layer_table") or {}


def _r2_at(table: dict, layer: str) -> float | None:
    entry = table.get(layer)
    return float(entry["r2_obs"]) if entry else None


def _mlp_best_frozen(mlp: dict) -> tuple[str | None, float | None]:
    best_layer, best = None, None
    for li, block in (mlp or {}).items():
        v = block.get("r2_obs")
        if isinstance(v, int | float) and not math.isnan(v) and (best is None or v > best):
            best_layer, best = str(li), float(v)
    return best_layer, best


def _fold_se(folds: list) -> float | None:
    vals = [float(v) for v in (folds or []) if isinstance(v, int | float) and not math.isnan(v)]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return sd / math.sqrt(len(vals))


def _anchor_deltas(out_dir: Path, parent_cells_dir: Path) -> dict:
    """Anchor fresh-vs-parent L19 deltas (HALT evaluation lives in the wrapper gate)."""
    deltas = {}
    for cid in ANCHOR_CELLS:
        fresh = _load(out_dir / f"cells_{cid}.json")
        parent = _load(parent_cells_dir / f"cells_{cid}.json")
        f19 = _r2_at(_frozen_table(fresh), "19") if fresh else None
        p19 = _r2_at(_frozen_table(parent), "19") if parent else None
        deltas[cid] = {
            "fresh_r2_L19": f19,
            "parent_r2_L19": p19,
            "delta": (f19 - p19) if (f19 is not None and p19 is not None) else None,
            "tolerance": ANCHOR_TOL,
            "n_fresh": (fresh or {}).get("metadata", {}).get("n"),
        }
    return deltas


def _nll_diagnostics(onp: Path) -> dict:
    """Wiring-check own/shuffled NLL per user cell + the parent Haiku band (diagnostic)."""
    diag = {}
    for model in ("instruct", "pretrained"):
        w = _load(onp / f"wiring_check_{model}.json")
        if not w:
            continue
        for fmt, blk in (w.get("per_format") or {}).items():
            diag[blk.get("cell_id", f"M_{model}_user_{fmt}")] = {
                "own_mean_nll": blk.get("own_mean_nll"),
                "shuffled_mean_nll": blk.get("shuffled_mean_nll"),
                "own_minus_shuffled": blk.get("own_minus_shuffled"),
                "n": blk.get("n"),
                "parent_haiku_nll_band": list(PARENT_HAIKU_NLL_BAND),
                "note": (
                    "self-gen NLL vs the parent Haiku band is a REPORTED "
                    "diagnostic, never a HALT (plan MF-B: self-sample NLL = "
                    "conditional entropy)"
                ),
            }
    return diag


def _parent_trcov(parent: dict | None, matched_parent: dict | None, model: str, fmt: str) -> dict:
    """Parent tr(cov) sourced UNCONDITIONALLY (review-r1 CONCERN
    parent-trcov-conditional-only): committed parent eval JSONs first (key
    absent as of the parent runs — added by this round's fit extension), then
    the matched-parent refit; else an EXPLICIT null + the post-hoc recipe —
    never a silently absent key."""
    trcov = (parent or {}).get("y_trace_cov_frozen")
    source = "parent committed eval JSON" if trcov is not None else None
    if trcov is None and matched_parent is not None:
        trcov = matched_parent.get("y_trace_cov_frozen")
        if trcov is not None:
            source = "matched-parent refit (this run)"
    if trcov is None:
        source = (
            "unavailable pod-side without staging ~17 GB of parent m-track shards; "
            "post-hoc recipe: stage superkaiba1/explore-persona-space-data "
            f"issue825_userbase_map/analysis_tensors/{model}_{fmt}_m_shard*.{{pt,json}} "
            "@ deb7a4523b5233393e4fbd2497622527b3622d35 and refit the cell with "
            "scripts/issue825_fit_cells.py (writes y_trace_cov_frozen), or compute "
            "tr(cov) of the v(u2) turn-profile rows at the frozen layers directly"
        )
    return {"parent_y_trace_cov_frozen": trcov, "parent_trcov_source": source}


def summarize(args) -> dict:
    smoke = os.environ.get("EPS_SMOKE") == "1"
    out_dir: Path = args.out_dir
    onp: Path = args.onpolicy_dir
    anchor_deltas = _anchor_deltas(out_dir, args.parent_cells_dir)
    nll_diag = _nll_diagnostics(onp)

    # ── per user cell ──
    cells = {}
    support_summary = {"supported": [], "suggestive": [], "provenance_sensitive_negative": []}
    for cid in USER_CELLS:
        model, fmt = cid.split("_")[1], cid.split("_")[-1]
        fresh = _load(out_dir / f"cells_{cid}.json")
        parent = _load(args.parent_cells_dir / f"cells_{cid}.json")
        parent_mlp_cells = _load(args.parent_mlp_dir / f"cells_{cid}.json")
        meta = _load(onp / f"conversations_{model}_{fmt}_meta.json") or {}
        if fresh is None:
            cells[cid] = {"missing": True}
            continue

        # ── degeneracy audit FIRST (review-r1 BLOCKER audit-floors-not-binding-
        # headline): audit_pass must gate the headline label + support reads
        # below, so an audit-failing cell (plan hard-req 3 / the audit-gated
        # pretrained/chat cell, plan line 42) can never enter support_summary.
        keep_rate = meta.get("keep_rate")
        d3g = meta.get("distinct_3gram_rate_kept")
        audit_pass = (
            keep_rate is not None
            and d3g is not None
            and keep_rate >= KEEP_RATE_FLOOR
            and d3g >= DISTINCT_3GRAM_FLOOR
        )
        headline_eligible = bool(audit_pass) if not smoke else None

        ftab, ptab = _frozen_table(fresh), _frozen_table(parent) if parent else {}
        ridge = {
            "frozen_layer_table": ftab,
            "selection_read": {
                k: (fresh.get("selection_symmetric") or {}).get(k)
                for k in ("obs_layer_max_r2", "obs_argmax_layer", "null_layer_max_p975")
            },
            "null_layer_max_per_draw": (fresh.get("selection_symmetric") or {}).get(
                "null_layer_max_r2_per_draw"
            ),
            "deltas_vs_parent": {
                layer: {
                    "fresh": _r2_at(ftab, layer),
                    "parent": _r2_at(ptab, layer),
                    "delta": (
                        _r2_at(ftab, layer) - _r2_at(ptab, layer)
                        if (_r2_at(ftab, layer) is not None and _r2_at(ptab, layer) is not None)
                        else None
                    ),
                }
                for layer in ("19", "26")
            },
            "any_frozen_r2_positive": any(
                (_r2_at(ftab, layer) or float("-inf")) > 0 for layer in ftab
            ),
        }
        ridge_d19 = ridge["deltas_vs_parent"]["19"]["delta"]
        ridge_d26 = ridge["deltas_vs_parent"]["26"]["delta"]
        ridge_supported = bool(
            (ridge_d19 is not None and ridge_d19 >= RIDGE_MARGIN)
            or (ridge_d26 is not None and ridge_d26 >= RIDGE_MARGIN)
            or ridge["any_frozen_r2_positive"]
        )

        mlp = fresh.get("mlp") or {}
        best_layer, best = _mlp_best_frozen(mlp)
        p_best_layer, p_best = _mlp_best_frozen((parent_mlp_cells or {}).get("mlp") or {})
        se_new = _fold_se((mlp.get(best_layer) or {}).get("r2_obs_folds")) if best_layer else None
        # Parent fold values are NOT in the committed parent JSONs -> the plan's
        # stated proxy: SE_delta = new-run fold SE x sqrt(2).
        se_delta = (se_new * math.sqrt(2.0)) if se_new is not None else None
        mlp_delta = (best - p_best) if (best is not None and p_best is not None) else None
        point_ok = mlp_delta is not None and mlp_delta >= MLP_POINT_MARGIN
        noise_ok = (
            mlp_delta is not None and se_delta is not None and (mlp_delta - 2.0 * se_delta) > 0
        )
        if point_ok and noise_ok:
            mlp_label = "supported"
        elif point_ok:
            mlp_label = "suggestive"
        else:
            mlp_label = "not-supported"

        kill_consistent = bool(
            mlp_delta is not None
            and abs(mlp_delta) <= MLP_POINT_MARGIN
            and ridge_d19 is not None
            and abs(ridge_d19) <= RIDGE_MARGIN
            and ridge_d26 is not None
            and abs(ridge_d26) <= RIDGE_MARGIN
        )
        prov_negative = bool(
            (mlp_delta is not None and mlp_delta < -MLP_POINT_MARGIN)
            or (ridge_d19 is not None and ridge_d19 < -RIDGE_MARGIN)
            or (ridge_d26 is not None and ridge_d26 < -RIDGE_MARGIN)
        )
        if mlp_label == "supported" or ridge_supported:
            cell_label_pre_audit = "supported"
        elif mlp_label == "suggestive":
            cell_label_pre_audit = "suggestive"
        elif prov_negative:
            cell_label_pre_audit = "provenance-sensitive-negative"
        elif kill_consistent:
            cell_label_pre_audit = "kill-consistent"
        else:
            cell_label_pre_audit = "indeterminate"
        # Headline label + support reads gated on audit_pass (production;
        # smoke bypasses floors per MF-D). An audit-FAILing cell is reported
        # "degenerate-provenance — observational" (plan line 42) and EXCLUDED
        # from support_summary on EVERY lane — incl. any_frozen_r2_positive,
        # the mechanically-inflatable lane under low target diversity.
        if smoke or audit_pass:
            cell_label = cell_label_pre_audit
            if cell_label == "supported":
                lanes = [
                    lane
                    for lane, ok in (("mlp", mlp_label == "supported"), ("ridge", ridge_supported))
                    if ok
                ]
                support_summary["supported"].append({"cell": cid, "lanes": lanes})
            elif cell_label == "suggestive":
                support_summary["suggestive"].append({"cell": cid, "lanes": ["mlp"]})
            elif cell_label == "provenance-sensitive-negative":
                support_summary["provenance_sensitive_negative"].append({"cell": cid})
        else:
            cell_label = "degenerate-provenance — observational"

        matched_parent = _load(args.matched_parent_dir / f"cells_{cid}.json")
        cells[cid] = {
            "n_cell": fresh.get("n_allowlist") or fresh.get("metadata", {}).get("n"),
            "drops": meta.get("drops"),
            "ridge": ridge,
            "mlp": {
                "per_layer": mlp,
                "best_frozen": {"layer": best_layer, "r2_obs": best},
                "parent_best_frozen": {"layer": p_best_layer, "r2_obs": p_best},
                "delta_best_frozen": mlp_delta,
                "se_new_folds": se_new,
                "se_delta_proxy_sqrt2": se_delta,
                "point_margin_pass": point_ok,
                "noise_clause_pass": noise_ok,
                "label": mlp_label,
                "budget_cap_hits": {li: blk.get("budget_hit_folds") for li, blk in mlp.items()},
                "budget_exhausted": fresh.get("mlp_budget_exhausted"),
            },
            "ridge_supported": ridge_supported,
            "kill_consistent": kill_consistent,
            "cell_label": cell_label,
            "cell_label_pre_audit": cell_label_pre_audit,
            "audit": {
                "keep_rate": keep_rate,
                "keep_rate_floor": KEEP_RATE_FLOOR,
                "distinct_3gram_rate_kept": d3g,
                "distinct_3gram_floor": DISTINCT_3GRAM_FLOOR,
                "distinct_3gram_parent_reference": (meta.get("parent_reference") or {}).get(
                    "distinct_3gram_rate"
                ),
                "repetition_rate": meta.get("repetition_rate"),
                "role_artifact_rate": meta.get("role_artifact_rate"),
                "u2_length_kept": meta.get("u2_length_kept"),
                "parent_u2_length": (meta.get("parent_reference") or {}).get("u2_length"),
                "headline_eligible": headline_eligible,
                "floors_bypassed_smoke": smoke,
                "label_downgraded_to_observational": (not smoke) and not audit_pass,
            },
            "nll_diagnostics": nll_diag.get(cid),
            "diversity_tr_cov": {
                "fresh_y_trace_cov_frozen": fresh.get("y_trace_cov_frozen"),
                **_parent_trcov(parent, matched_parent, model, fmt),
            },
            "matched_parent_refit": (
                {
                    "ran": True,
                    "frozen_layer_table": _frozen_table(matched_parent),
                    "n": matched_parent.get("metadata", {}).get("n"),
                }
                if matched_parent
                else {"ran": False}
            ),
        }

    headline = {
        "followup_label": "onpolicy-user-turn",
        "smoke": smoke,
        "interpretation_rule": INTERPRETATION_RULE,
        "thresholds": {
            "mlp_point_margin": MLP_POINT_MARGIN,
            "ridge_margin": RIDGE_MARGIN,
            "anchor_tolerance": ANCHOR_TOL,
            "keep_rate_floor": KEEP_RATE_FLOOR,
            "distinct_3gram_floor": DISTINCT_3GRAM_FLOOR,
        },
        "support_summary": support_summary,
        "anchor_deltas": anchor_deltas,
        "cells": cells,
        "parent_delta_table": {
            cid: {
                "mlp_best_frozen_delta": (cells[cid].get("mlp") or {}).get("delta_best_frozen"),
                "mlp_label": (cells[cid].get("mlp") or {}).get("label"),
                "ridge_delta_L19": ((cells[cid].get("ridge") or {}).get("deltas_vs_parent") or {})
                .get("19", {})
                .get("delta"),
                "ridge_delta_L26": ((cells[cid].get("ridge") or {}).get("deltas_vs_parent") or {})
                .get("26", {})
                .get("delta"),
                # An audit-failing cell stays IN the table (never dropped) but
                # flagged: cell_label reads "degenerate-provenance —
                # observational" and headline_eligible is False.
                "cell_label": cells[cid].get("cell_label"),
                "headline_eligible": (cells[cid].get("audit") or {}).get("headline_eligible"),
            }
            for cid in USER_CELLS
            if not cells.get(cid, {}).get("missing")
        },
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": "scripts/issue825_onpolicy_summarize.py",
        },
    }
    return headline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--onpolicy-dir", type=Path, required=True)
    ap.add_argument("--parent-cells-dir", type=Path, default=Path("eval_results/issue_825"))
    ap.add_argument(
        "--parent-mlp-dir", type=Path, default=Path("eval_results/issue_825/mlp-unprobed-cells")
    )
    ap.add_argument("--matched-parent-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.matched_parent_dir is None:
        args.matched_parent_dir = args.out_dir / "matched_parent"
    headline = summarize(args)
    out_path = args.out_dir / "headline_metrics.json"
    out_path.write_text(json.dumps(headline, indent=2, default=float) + "\n")
    print(f"[summarize] wrote {out_path}")
    for cid, row in headline["parent_delta_table"].items():
        print(f"[summarize] {cid}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
