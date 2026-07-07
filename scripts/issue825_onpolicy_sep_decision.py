"""Issue #825 `onpolicy-separator-control`: decision_support.json emission (plan 6.5).

Per model, the plan-section-1 pre-registered reads packaged for the ANALYZER
(the binding interpretation rule is applied there, never here):

  W_on_max = max(on-policy sep rotated @ L19, on-policy sep MLP @ L19)
  W_ex_eff = w_ex_matched_n when the G4b matched-n trigger fired, else the
             committed full-n W_ex = max(committed rotated, committed MLP)
  D        = (W_on_max - W_ex_eff) / (C - W_ex_eff)      [R1/R2]
  ratio    = W_on_max / C                                 [R3]
  |D_base - D_inst| vs the 0.10 margin                    [R5]

plus raw dW = W_on_max - W_ex_eff (the R5 denominator-artifact companion),
the rotated-estimator group-bootstrap CI mapped through the D transform (the
section-1 CI qualifier input; when MLP wins the max the qualifier falls back
to the rotated CI with the stated caveat), gate outcomes, per-model
realized_n + w_ex_matched_n, and the R4 transfer-fractions SLOT (nulls —
filled by the Phase C onpolicy_sep_to_chat_{base,instruct}.json files after
pod release; deliberately outside the pod-side enumeration).

Committed references are READ from the committed eval JSONs at run time and
drift-checked against the plan section 1 quotes (the round-6 gate pattern).

CLI:
  uv run python scripts/issue825_onpolicy_sep_decision.py \
      --out-dir <OUT root> --data-root <DATA root> [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue825_onpolicy_sep_decision.py"
REPO = Path(__file__).resolve().parent.parent

# Plan section 1 quotes (documentation anchors; the committed JSONs are the
# gate — a committed-vs-quote drift beyond 1e-3 fails loud).
PLAN_QUOTED = {
    "base": {
        "rotated": 0.36261,
        "mlp": 0.31111,
        "ceiling": 0.58768,
        "cells": "eval_results/issue_825/base-separator-control",
        "ceiling_path": "eval_results/issue_825/cells_S2.json",
        "exo_transfer_fraction_fulln": 0.0574,
    },
    "instruct": {
        "rotated": 0.34892,
        "mlp": 0.29859,
        "ceiling": 0.67309,
        "cells": "eval_results/issue_931",
        "ceiling_path": "eval_results/issue_931/cells_chat_ref.json",
        "exo_transfer_fraction_fulln": 0.1087,
    },
}
D_BANDS = {"survives_max": 0.33, "reframed_min": 0.67}
R4_THRESHOLD = 0.5
R5_MARGIN = 0.10


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True, help="OUT root (per-model subdirs)")
    ap.add_argument("--data-root", type=Path, required=True, help="DATA root (per-model subdirs)")
    ap.add_argument("--models", type=str, default="base,instruct")
    ap.add_argument("--smoke", action="store_true", help="committed-quote drift checks relaxed")
    return ap.parse_args()


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def committed_refs(model: str, *, smoke: bool) -> dict:
    """Committed full-n W_ex components + chat ceiling, drift-checked."""
    q = PLAN_QUOTED[model]
    cells = REPO / q["cells"]
    sep = json.loads((cells / "cells_armC_sep.json").read_text())
    hl = int(sep.get("headline_layer", 19))
    rot = float(sep["random_projection_control_r2"][str(hl)])
    mlp_doc = json.loads((cells / "mlp_secondary.json").read_text())
    mlp = float(mlp_doc["cells"]["armC_sep"][str(hl)]["r2_obs"])
    ceil_doc = json.loads((REPO / q["ceiling_path"]).read_text())
    ceiling = float(ceil_doc["r2_per_layer_obs"][19])
    if not smoke:
        assert abs(rot - q["rotated"]) < 1e-3, (model, rot, q["rotated"])
        assert abs(mlp - q["mlp"]) < 1e-3, (model, mlp, q["mlp"])
        assert abs(ceiling - q["ceiling"]) < 1e-3, (model, ceiling, q["ceiling"])
    return {
        "rotated": rot,
        "mlp": mlp,
        "w_ex_fulln": max(rot, mlp),
        "ceiling_fulln": ceiling,
        "headline_layer": hl,
        "exo_transfer_fraction_fulln": q["exo_transfer_fraction_fulln"],
    }


def onpolicy_reads(out_dir: Path, model: str) -> dict:
    """On-policy sep within-strength reads @ the headline layer (+ rotated CI)."""
    cells = out_dir / model
    sep = json.loads((cells / "cells_armC_sep.json").read_text())
    hl = int(sep.get("headline_layer", 19))
    rot = float(sep["random_projection_control_r2"][str(hl)])
    ridge = float(sep["r2_per_layer_obs"][hl])
    mlp_doc = _read(cells / "mlp_secondary.json") or {"cells": {}}
    mlp_cell = (mlp_doc["cells"].get("armC_sep") or {}).get(str(hl)) or {}
    mlp = mlp_cell.get("r2_obs")
    rot_ci = (sep.get("rotated_bootstrap_group_frozen") or {}).get(str(hl))
    prev = _read(cells / "cells_armC_prevmean.json")
    return {
        "headline_layer": hl,
        "ridge": ridge,
        "rotated": rot,
        "mlp": mlp,
        "w_on_max": max(rot, mlp) if mlp is not None else rot,
        "mlp_wins_max": bool(mlp is not None and mlp > rot),
        "rotated_ci": rot_ci,
        "prevmean_rotated": (
            float(prev["random_projection_control_r2"][str(hl)]) if prev else None
        ),
        "n": int(sep["n"]),
    }


def d_stat(w_on: float, w_ex: float, ceiling: float) -> float:
    denom = ceiling - w_ex
    return (w_on - w_ex) / denom if abs(denom) > 1e-12 else float("nan")


def main() -> int:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    per_model: dict[str, dict] = {}
    gates: dict[str, object] = {}
    for m in models:
        refs = committed_refs(m, smoke=args.smoke)
        on = onpolicy_reads(args.out_dir, m)
        meta = _read(args.data_root / m / "pairs" / "pairs_meta.json") or {}
        realized_n = int(meta.get("realized_n", on["n"]))
        anchor_key = "anchor_base" if m == "base" else "anchor_inst"
        matched = _read(args.out_dir / anchor_key / f"matched_n_wex_{m}.json")
        w_ex_matched = matched.get("w_ex_matched_n") if matched else None
        trigger_fired = bool(matched and matched.get("trigger_fired"))
        w_ex_eff = (
            w_ex_matched if (trigger_fired and w_ex_matched is not None) else refs["w_ex_fulln"]
        )
        c = refs["ceiling_fulln"]
        d_eff = d_stat(on["w_on_max"], w_ex_eff, c)
        d_fulln = d_stat(on["w_on_max"], refs["w_ex_fulln"], c)
        rot_ci = on["rotated_ci"] or {}
        d_ci = (
            {
                "lo": d_stat(float(rot_ci["ci_lo"]), w_ex_eff, c),
                "hi": d_stat(float(rot_ci["ci_hi"]), w_ex_eff, c),
                "estimator": "rotated",
                "note": (
                    "rotated-estimator group-bootstrap CI mapped through the D "
                    "transform; when MLP wins the max the band-boundary qualifier "
                    "falls back to this rotated CI with a stated caveat (plan "
                    "section 1 CI convention)"
                ),
            }
            if rot_ci
            else None
        )
        per_model[m] = {
            "onpolicy": on,
            "committed_reference": refs,
            "realized_n": realized_n,
            "matched_n_trigger_fired": trigger_fired,
            "w_ex_matched_n": w_ex_matched,
            "w_ex_effective": w_ex_eff,
            "D": d_eff,
            "D_fulln_reference": d_fulln,
            "D_ci_rotated": d_ci,
            "delta_w_raw": on["w_on_max"] - w_ex_eff,
            "ratio_ceiling": on["w_on_max"] / c,  # R3
            "exogenous_ratio_ceiling": refs["w_ex_fulln"] / c,
        }
        gates[f"anchor_gate_{m}"] = (
            _read(args.out_dir / anchor_key / "anchor_gate.json") or {}
        ).get("pass")
        eq = _read(args.data_root / m / "store" / "armC" / "armC_equivalence.json")
        gates[f"equivalence_gate_{m}"] = (
            {"early_cos_min": eq.get("early_cos_min"), "flat_cos_min": eq.get("flat_cos_min")}
            if eq
            else None
        )

    r5 = None
    if "base" in per_model and "instruct" in per_model:
        r5 = {
            "abs_d_gap": abs(per_model["base"]["D"] - per_model["instruct"]["D"]),
            "margin": R5_MARGIN,
            "mirror_within_margin": bool(
                abs(per_model["base"]["D"] - per_model["instruct"]["D"]) <= R5_MARGIN
            ),
            "note": "report raw delta_w_raw next to D per substrate — the D "
            "denominators differ (0.22507 vs 0.32417), so a D gap can be a "
            "denominator artifact (plan section 7 concern 3)",
        }
    payload = {
        "metadata": common.metadata(SCRIPT, common.FIT_SEED, 0),
        "followup_label": "onpolicy-separator-control",
        "smoke": bool(args.smoke),
        "per_model": per_model,
        "r5_mirror": r5,
        "gates": gates,
        "transfer_fractions": {m: None for m in models}
        | {
            "note": (
                "R4 slot — filled by Phase C on the VM "
                "(onpolicy_sep_to_chat_{base,instruct}.json, full-n convention, "
                "fraction of the full-n chat ceiling); deliberately outside the "
                "pod-side enumeration (plan section 6.5)"
            ),
            "threshold": R4_THRESHOLD,
        },
        "interpretation_bands": {
            **D_BANDS,
            "r4_threshold": R4_THRESHOLD,
            "r5_margin": R5_MARGIN,
            "rule": (
                "BINDING, applied by the ANALYZER (plan section 1): per model "
                "SURVIVES iff D <= 0.33 AND R4 fraction < 0.5; REFRAMED iff "
                "D >= 0.67 OR R4 fraction >= 0.5; PARTIAL otherwise; CI spanning "
                "a band boundary => 'suggestive' qualifier; R4 skipped => "
                "SURVIVES unevaluable (at most 'suggestive — D-only')"
            ),
        },
    }
    out_path = args.out_dir / "decision_support.json"
    common.write_json(out_path, payload)
    for m in models:
        pm = per_model[m]
        print(
            f"[i825-ops-dec] {m}: W_on_max={pm['onpolicy']['w_on_max']:.5f} "
            f"W_ex_eff={pm['w_ex_effective']:.5f} D={pm['D']:.4f} "
            f"(fulln D={pm['D_fulln_reference']:.4f}, n_r={pm['realized_n']}, "
            f"matched_n_fired={pm['matched_n_trigger_fired']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
