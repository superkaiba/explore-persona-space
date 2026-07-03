#!/usr/bin/env python
# ruff: noqa: E501
# The plan v11 §1 MF-R block (INTERPRETATION_RULE) is carried VERBATIM — its
# clause lines exceed 100 chars by construction (byte-pinned by
# tests/test_issue825_realuser_summarize.py; do NOT re-wrap).
"""Issue #825 ``real-user-turn-null`` summarizer -> headline_metrics.json (plan §4.3 item 4).

Schema keys parallel v7's headline (followup_label / smoke / interpretation_rule
/ thresholds / cells / metadata) with this round's content: per REAL user cell —
frozen-layer ridge table + selection read + null band + bootstrap CIs, MLP
per-frozen-layer R^2 + fold dispersion + 5-draw nulls + best-frozen + budget-cap
hits; per (model, format) — logged-assistant reference values + the within-round
assistant-minus-user contrast (immune to the different-conversations confound);
cross-provenance table (real vs parent-Haiku vs v7-self, ridge L19/L26 + MLP
best-frozen, each with the §1 descriptive-label read); u2-pool audit vs the
Haiku + self-gen references; per-layer tr(cov) of v(u2) vs parent cells; NLL
diagnostics (wiring own/shuffled + the Haiku/self reference bands, reported
diagnostics never gates); a1-model mix; the single end-to-end anchor delta; and
the binding interpretation rule MF-R verbatim.

Outcome LABELS (GLOBAL/LOCALIZED BREAK etc.) are the analyzer's per plan §1;
this script surfaces the mechanical inputs (per-cell frozen-layer CI-positive
flags) plus a note that the Bonferroni (99.6875% lower CI) read requires a
refit on the persisted turnstore (raw bootstrap draws are not persisted).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

FOLLOWUP_LABEL = "real-user-turn-null"
USER_CELLS = [
    "M_instruct_user_chat",
    "M_instruct_user_naturalistic",
    "M_pretrained_user_chat",
    "M_pretrained_user_naturalistic",
]
ASSISTANT_CELLS = [
    "M_instruct_assistant_chat",
    "M_instruct_assistant_naturalistic",
    "M_pretrained_assistant_chat",
    "M_pretrained_assistant_naturalistic",
]
ANCHOR_CELL = "M_instruct_assistant_chat"
ANCHOR_TOL = 0.05
MLP_POINT_MARGIN = 0.10  # §1 secondary: ±0.10 with 2SE discipline
HEADLINE_LAYERS = ("19", "26")
# Reported reference values (diagnostics, never gates — plan §4.3 item 4):
PARENT_HAIKU_NLL_BAND = (2.04, 2.64)
V7_SELF_NLL_BANDS = (1.33, 1.45, 2.62, 2.72)  # plan-quoted self-gen own-NLL bands (unmapped)
HAIKU_DISTINCT_3GRAM_REFERENCE = 0.781
HAIKU_U2_MEAN_TOKENS_REFERENCE = 79.0

# Binding interpretation rule — plan v11 §1 MF-R block carried VERBATIM
# (all 5 clauses, exact glyphs incl. R², bold markers as in the plan; the
# analyzer and clean-result MUST carry it forward). Byte-pinned by
# tests/test_issue825_realuser_summarize.py — edit ONLY in lockstep with
# the plan §1 block.
INTERPRETATION_RULE = """**Binding interpretation rule (MF-R — mirrors v7's MF-F; the analyzer and clean-result MUST carry it verbatim):**
1. The real cells differ from the parent/v7 cells in **conversation sample, a1 authorship (logged serving models, e.g. vicuna/ChatGPT-class, not the measured model), and u2 authorship** — a bundled "real conversation" change. Cross-provenance deltas are DESCRIPTIVE provenance-bundle claims only; no claim may attribute a delta to u2 realness specifically.
2. Within-round claims (licensed): existence/absence of linear and nonlinear user-turn maps ON real conversations, read against the same-conversation assistant reference cells and this round's own nulls.
3. Null-persists licenses: "the user-turn linear null holds under all three tested u2 provenances, each on its own conversation distribution" — a scope-union claim, no mechanism.
4. Null-breaks licenses: "on real 2-turn lmsys conversations the user-turn map is linearly decodable (R² = X)" — it does NOT identify which bundled component drives the break. The isolating control (regenerate a1 with the measured model on the same real conversations, splicing the real u2 behind it) is a named candidate follow-up, NOT part of this round.
5. Scope note carried to the clean-result: humans who write a second turn are a self-selected subpopulation of lmsys users (continuation selection); the real-u2 read is a statement about that subpopulation."""


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


def _frozen_table(payload: dict | None) -> dict:
    return ((payload or {}).get("selection_symmetric") or {}).get("frozen_layer_table") or {}


def _r2_at(table: dict, layer: str) -> float | None:
    entry = table.get(layer)
    return float(entry["r2_obs"]) if entry else None


def _num(v) -> float | None:
    return float(v) if isinstance(v, int | float) and not math.isnan(v) else None


def _mlp_best_frozen(mlp: dict | None) -> tuple[str | None, float | None]:
    best_layer, best = None, None
    for li, block in (mlp or {}).items():
        v = _num(block.get("r2_obs"))
        if v is not None and (best is None or v > best):
            best_layer, best = str(li), v
    return best_layer, best


def _fold_se(folds: list | None) -> float | None:
    vals = [float(v) for v in (folds or []) if isinstance(v, int | float) and not math.isnan(v)]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return sd / math.sqrt(len(vals))


def _ridge_block(payload: dict | None) -> dict:
    """Per-cell ridge read: frozen table + selection read + bootstrap CIs +
    the mechanical frozen-layer CI-positive flags feeding the §1 lattice."""
    if payload is None:
        return {"missing": True}
    sel = payload.get("selection_symmetric") or {}
    ftab = _frozen_table(payload)
    cis = payload.get("r2_bootstrap_ci_frozen_layers") or {}
    return {
        "frozen_layer_table": ftab,
        "selection_read": {
            k: sel.get(k) for k in ("obs_layer_max_r2", "obs_argmax_layer", "null_layer_max_p975")
        },
        "null_layer_max_per_draw": sel.get("null_layer_max_r2_per_draw"),
        "bootstrap_ci_frozen_layers": cis,
        "frozen_ci_positive": {
            layer: (
                bool(
                    _num((cis.get(layer) or {}).get("ci_lo")) is not None
                    and float(cis[layer]["ci_lo"]) > 0
                )
                if cis.get(layer)
                else None
            )
            for layer in ftab
        },
        "bonferroni_note": (
            "§1 clause (c) needs the 99.6875% lower CI; raw bootstrap draws are "
            "not persisted — the analyzer's free re-check refits on the "
            "persisted turnstore shards if any read is CI-positive"
        ),
    }


def _mlp_block(payload: dict | None) -> dict:
    mlp = (payload or {}).get("mlp") or {}
    best_layer, best = _mlp_best_frozen(mlp)
    se = _fold_se((mlp.get(best_layer) or {}).get("r2_obs_folds")) if best_layer else None
    return {
        "per_layer": mlp,
        "best_frozen": {"layer": best_layer, "r2_obs": best},
        "se_new_folds": se,
        # Parent/v7 fold values are not persisted -> the v7 proxy for a
        # cross-provenance delta SE: this round's fold SE x sqrt(2).
        "se_delta_proxy_sqrt2": (se * math.sqrt(2.0)) if se is not None else None,
        "budget_cap_hits": {li: blk.get("budget_hit_folds") for li, blk in mlp.items()},
        "budget_exhausted": (payload or {}).get("mlp_budget_exhausted"),
    }


def _descriptive_label(delta: float | None, se_delta: float | None) -> str:
    """§1 secondary rule: ±0.10 point margin with the 2SE noise clause."""
    if delta is None:
        return "unavailable"
    if delta >= MLP_POINT_MARGIN and se_delta is not None and delta - 2.0 * se_delta > 0:
        return "more predictable (cross-sample, descriptive)"
    if delta <= -MLP_POINT_MARGIN and se_delta is not None and delta + 2.0 * se_delta < 0:
        return "less predictable (cross-sample, descriptive)"
    return "in-band"


def _parent_trcov(parent: dict | None, v7_cell: dict | None) -> dict:
    """Parent tr(cov) sourced unconditionally (v7 r1 lesson): committed parent
    JSON first (key absent as of the parent runs), then v7's matched-parent
    refit value, else an explicit null + recipe — never a silently absent key."""
    trcov = (parent or {}).get("y_trace_cov_frozen")
    source = "parent committed eval JSON" if trcov is not None else None
    if trcov is None and v7_cell is not None:
        trcov = (v7_cell.get("diversity_tr_cov") or {}).get("parent_y_trace_cov_frozen")
        if trcov is not None:
            source = "v7 matched-parent refit (onpolicy-user-turn headline)"
    if trcov is None:
        source = (
            "unavailable — post-hoc recipe: stage the parent m-track shards @ "
            "deb7a4523b5233393e4fbd2497622527b3622d35 and refit with "
            "scripts/issue825_fit_cells.py (writes y_trace_cov_frozen)"
        )
    return {"parent_y_trace_cov_frozen": trcov, "parent_trcov_source": source}


def summarize(args) -> dict:
    smoke = os.environ.get("EPS_SMOKE") == "1"
    out_dir: Path = args.out_dir
    meta = _load(args.realuser_dir / "conversations_real2turn_meta.json") or {}
    v7_headline = _load(args.v7_headline) or {}
    v7_cells = v7_headline.get("cells") or {}

    # ── per-cell blocks (all 8 real cells) ──
    cells: dict = {}
    for cid in USER_CELLS + ASSISTANT_CELLS:
        fresh = _load(out_dir / f"cells_{cid}.json")
        if fresh is None:
            cells[cid] = {"missing": True}
            continue
        parent = _load(args.parent_cells_dir / f"cells_{cid}.json")
        v7_cell = v7_cells.get(cid)
        cells[cid] = {
            "role": "user" if cid in USER_CELLS else "assistant",
            "n_cell": (fresh.get("metadata") or {}).get("n"),
            "ridge": _ridge_block(fresh),
            "mlp": _mlp_block(fresh),
            "diversity_tr_cov": {
                "fresh_y_trace_cov_frozen": fresh.get("y_trace_cov_frozen"),
                "v7_self_y_trace_cov_frozen": (
                    ((v7_cell or {}).get("diversity_tr_cov") or {}).get("fresh_y_trace_cov_frozen")
                ),
                **_parent_trcov(parent, v7_cell),
            },
        }

    # ── within-round assistant-minus-user contrast per (model, format) ──
    contrast: dict = {}
    for model in ("instruct", "pretrained"):
        for fmt in ("chat", "naturalistic"):
            a = cells.get(f"M_{model}_assistant_{fmt}") or {}
            u = cells.get(f"M_{model}_user_{fmt}") or {}
            if a.get("missing") or u.get("missing"):
                contrast[f"{model}_{fmt}"] = {"missing": True}
                continue
            row: dict = {}
            for layer in HEADLINE_LAYERS:
                ar = _r2_at((a.get("ridge") or {}).get("frozen_layer_table") or {}, layer)
                ur = _r2_at((u.get("ridge") or {}).get("frozen_layer_table") or {}, layer)
                row[f"ridge_L{layer}"] = {
                    "assistant": ar,
                    "user": ur,
                    "assistant_minus_user": (
                        (ar - ur) if (ar is not None and ur is not None) else None
                    ),
                }
            ab = ((a.get("mlp") or {}).get("best_frozen") or {}).get("r2_obs")
            ub = ((u.get("mlp") or {}).get("best_frozen") or {}).get("r2_obs")
            row["mlp_best_frozen"] = {
                "assistant": ab,
                "user": ub,
                "assistant_minus_user": (
                    (ab - ub) if (ab is not None and ub is not None) else None
                ),
            }
            contrast[f"{model}_{fmt}"] = row

    # ── cross-provenance table per user cell (descriptive only — MF-R.1) ──
    cross_provenance: dict = {}
    for cid in USER_CELLS:
        real = cells.get(cid) or {}
        if real.get("missing"):
            cross_provenance[cid] = {"missing": True}
            continue
        rtab = (real.get("ridge") or {}).get("frozen_layer_table") or {}
        parent_tab = _frozen_table(_load(args.parent_cells_dir / f"cells_{cid}.json"))
        parent_mlp_payload = _load(args.parent_mlp_dir / f"cells_{cid}.json")
        _, parent_mlp_best = _mlp_best_frozen((parent_mlp_payload or {}).get("mlp"))
        v7_cell = v7_cells.get(cid) or {}
        v7_tab = ((v7_cell.get("ridge") or {}).get("frozen_layer_table")) or {}
        v7_mlp_best = (((v7_cell.get("mlp") or {}).get("best_frozen")) or {}).get("r2_obs")
        real_mlp_best = ((real.get("mlp") or {}).get("best_frozen") or {}).get("r2_obs")
        se_delta = (real.get("mlp") or {}).get("se_delta_proxy_sqrt2")
        d_haiku = (
            (real_mlp_best - parent_mlp_best)
            if (real_mlp_best is not None and parent_mlp_best is not None)
            else None
        )
        d_self = (
            (real_mlp_best - v7_mlp_best)
            if (real_mlp_best is not None and v7_mlp_best is not None)
            else None
        )
        cross_provenance[cid] = {
            "ridge": {
                f"L{layer}": {
                    "real": _r2_at(rtab, layer),
                    "parent_haiku": _r2_at(parent_tab, layer),
                    "v7_self": _r2_at(v7_tab, layer),
                }
                for layer in HEADLINE_LAYERS
            },
            "mlp_best_frozen": {
                "real": real_mlp_best,
                "parent_haiku": parent_mlp_best,
                "v7_self": v7_mlp_best,
            },
            "mlp_delta_real_minus_haiku": d_haiku,
            "mlp_delta_real_minus_self": d_self,
            "se_delta_proxy_sqrt2": se_delta,
            "descriptive_label_vs_haiku": _descriptive_label(d_haiku, se_delta),
            "descriptive_label_vs_self": _descriptive_label(d_self, se_delta),
            "label_rule": (
                "§1 secondary (descriptive, cross-sample — MF-R.1): |delta| >= "
                f"{MLP_POINT_MARGIN} clearing 2SE (this round's fold SE x sqrt2 "
                "proxy; parent/v7 folds not persisted) — no support/kill labels "
                "ride on the MLP"
            ),
        }

    # ── NLL diagnostics (wiring own/shuffled + reference bands) ──
    nll_diag: dict = {}
    for model in ("instruct", "pretrained"):
        w = _load(args.wiring_dir / f"wiring_check_{model}.json")
        for fmt, blk in ((w or {}).get("per_format") or {}).items():
            cid = blk.get("cell_id", f"M_{model}_user_{fmt}")
            v7_nll = ((v7_cells.get(cid) or {}).get("nll_diagnostics") or {}).get("own_mean_nll")
            nll_diag[cid] = {
                "own_mean_nll": blk.get("own_mean_nll"),
                "shuffled_mean_nll": blk.get("shuffled_mean_nll"),
                "own_minus_shuffled": blk.get("own_minus_shuffled"),
                "n": blk.get("n"),
                "parent_haiku_nll_band": list(PARENT_HAIKU_NLL_BAND),
                "v7_self_nll_reference_bands": list(V7_SELF_NLL_BANDS),
                "v7_self_own_mean_nll_same_cell": v7_nll,
                "note": (
                    "real-u2 NLL vs the Haiku/self bands is a REPORTED "
                    "diagnostic, never a gate (plan §4.2: HALT only on gross "
                    "own >= shuffled failure)"
                ),
            }

    # ── anchor delta (single end-to-end parent anchor) ──
    fresh_anchor = _load(out_dir / "anchor_parent" / f"cells_{ANCHOR_CELL}.json")
    parent_anchor = _load(args.parent_cells_dir / f"cells_{ANCHOR_CELL}.json")
    f19 = _r2_at(_frozen_table(fresh_anchor), "19")
    p19 = _r2_at(_frozen_table(parent_anchor), "19")
    anchor = {
        "cell": ANCHOR_CELL,
        "fresh_r2_L19": f19,
        "parent_committed_r2_L19": p19,
        "delta": (f19 - p19) if (f19 is not None and p19 is not None) else None,
        "tolerance": ANCHOR_TOL,
        "n_fresh": ((fresh_anchor or {}).get("metadata") or {}).get("n"),
    }

    # ── u2-pool audit (single shared pool — all 4 user cells share the rows) ──
    u2_pool_audit = {
        "u2_length": meta.get("u2_length"),
        "distinct_3gram_rate_u2": meta.get("distinct_3gram_rate_u2"),
        "repetition_rate_u2": meta.get("repetition_rate_u2"),
        "haiku_reference": {
            "distinct_3gram_rate": HAIKU_DISTINCT_3GRAM_REFERENCE,
            "u2_mean_tokens": HAIKU_U2_MEAN_TOKENS_REFERENCE,
        },
        "v7_self_reference": {
            cid: ((v7_cells.get(cid) or {}).get("audit") or {}).get("u2_length_kept")
            for cid in USER_CELLS
        },
        "note": "no u2 length cap beyond the 2048 conversation filter (plan §4.1 item 6)",
    }

    return {
        "followup_label": FOLLOWUP_LABEL,
        "smoke": smoke,
        "interpretation_rule": INTERPRETATION_RULE,
        "thresholds": {
            "mlp_point_margin": MLP_POINT_MARGIN,
            "anchor_tolerance": ANCHOR_TOL,
            "headline_layers": list(HEADLINE_LAYERS),
        },
        "cells": cells,
        "within_round_assistant_minus_user": contrast,
        "cross_provenance_table": cross_provenance,
        "nll_diagnostics": nll_diag,
        "anchor": anchor,
        "u2_pool_audit": u2_pool_audit,
        "a1_model_mix": meta.get("a1_model_mix"),
        "ingest": {
            "n_kept": meta.get("n_kept"),
            "n_streamed": meta.get("n_streamed"),
            "drops": meta.get("drops"),
            "dataset_revision": meta.get("dataset_revision"),
            "u1_overlap_with_parent_kept2000": meta.get("u1_overlap_with_parent_kept2000"),
        },
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": "scripts/issue825_realuser_summarize.py",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--realuser-dir", type=Path, required=True)
    ap.add_argument("--wiring-dir", type=Path, required=True)
    ap.add_argument("--parent-cells-dir", type=Path, default=Path("eval_results/issue_825"))
    ap.add_argument(
        "--parent-mlp-dir", type=Path, default=Path("eval_results/issue_825/mlp-unprobed-cells")
    )
    ap.add_argument(
        "--v7-headline",
        type=Path,
        default=Path("eval_results/issue_825/onpolicy-user-turn/headline_metrics.json"),
    )
    args = ap.parse_args()
    headline = summarize(args)
    out_path = args.out_dir / "headline_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(headline, indent=2, default=float) + "\n")
    print(f"[summarize] wrote {out_path}")
    for cid, row in headline["cross_provenance_table"].items():
        if not row.get("missing"):
            l19 = row["ridge"]["L19"]
            print(
                f"[summarize] {cid}: ridge L19 real={l19['real']} "
                f"haiku={l19['parent_haiku']} self={l19['v7_self']} "
                f"| mlp real={row['mlp_best_frozen']['real']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
