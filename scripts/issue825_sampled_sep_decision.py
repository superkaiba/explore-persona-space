"""Issue #825 `sampled-separator-control`: decision_support.json emission (plan v22 6.5).

Extends the round-7 decision script: the round-7 GREEDY cells are the FIXED
reference (read from the committed
eval_results/issue_825/onpolicy-separator-control/decision_support.json, never
re-derived), and the pre-registered reads are packaged per (model x arm) for
the ANALYZER (the binding interpretation rule is applied there, never here):

  R1  D(arm) = (W_arm_max - W_ex_eff) / (C - W_ex_eff); armB headline uses the
      round-7 conventions (matched-n trigger < 3492); arm-C cells use the
      matched-n W_ex at the realized C-avg n (fires by construction) with the
      BINDING never-headline framing carried in-payload.
  R2  Delta_dec = D(armB) - D_r7 per model; CAVEAT_RETIRED iff |Delta| <= 0.10
      else DECODING_SENSITIVE; CI qualifier per the plan section 1 convention
      (rotated var-sum when rotated carries BOTH sides; the MLP-CI leg when
      MLP carries; 'suggestive — margin-only' when neither CI exists).
  R3  NS = R2(C-avg) - R2(C-single) per estimator + a PAIRED group bootstrap
      over the shared per-group R^2 values (a free post-hoc reduction of the
      persisted per-group values — zero refits), as a fraction of the
      remaining gap (C - W_B_max). Descriptive — no binary label.
  R5  |D_base - D_inst| per arm in {B, C-avg} vs the 0.10 margin + raw dW.
  R6  per-arm 3-gram flag rates vs the round-7 greedy 77.1/68.9 (premise
      check: a drop of < 10 points partially fails the de-looping premise).

metadata.issue is stamped 825 (the round-7 JSON inherited 931 — reviewer fix).

CLI:
  uv run python scripts/issue825_sampled_sep_decision.py \
      --out-dir <OUT root> --data-root <DATA root> [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue825_sampled_sep_decision.py"
REPO = Path(__file__).resolve().parent.parent
FOLLOWUP_LABEL = "sampled-separator-control"
R7_DECISION = REPO / "eval_results/issue_825/onpolicy-separator-control/decision_support.json"

# Plan v22 section 1 frozen-reference quotes (documentation anchors; the
# committed JSONs are the gate — drift beyond 1e-3 fails loud, non-smoke).
PLAN_QUOTED = {
    "base": {
        "rotated": 0.36261,
        "ceiling": 0.58768,
        "denominator": 0.22507,
        "d_r7": 0.59043,
        "w_on_max_r7": 0.49550,
        "flag_rate_r7": 0.771,
        "cells": "eval_results/issue_825/base-separator-control",
        "ceiling_path": "eval_results/issue_825/cells_S2.json",
    },
    "instruct": {
        "rotated": 0.34892,
        "ceiling": 0.67309,
        "denominator": 0.32417,
        "d_r7": 0.42827,
        "w_on_max_r7": 0.48775,
        "flag_rate_r7": 0.689,
        "cells": "eval_results/issue_931",
        "ceiling_path": "eval_results/issue_931/cells_chat_ref.json",
    },
}
ARMS = ("armB", "armC_avg", "armC_single", "armC_pooled")
STABILITY_MARGIN = 0.10  # the issue's committed mirror-margin convention (R2 + R5)
R6_DROP_POINTS = 0.10  # flag rate within 10 points of greedy = premise partially fails
ARMB_MATCHED_N_TRIGGER = 3492  # round-7 convention (0.97 x 3600)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True, help="OUT root (per-model subdirs)")
    ap.add_argument("--data-root", type=Path, required=True, help="DATA root (per-model subdirs)")
    ap.add_argument("--models", type=str, default="base,instruct")
    ap.add_argument("--n-boot", type=int, default=common.N_BOOTSTRAP)
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
        assert abs(ceiling - q["ceiling"]) < 1e-3, (model, ceiling, q["ceiling"])
    return {
        "rotated": rot,
        "mlp": mlp,
        "w_ex_fulln": max(rot, mlp),
        "ceiling_fulln": ceiling,
        "headline_layer": hl,
    }


def r7_reference(model: str, *, smoke: bool) -> dict:
    """The FIXED round-7 greedy reference, read from the committed decision
    JSON (never re-derived) + drift-checked vs the plan v22 section 1 quotes."""
    q = PLAN_QUOTED[model]
    doc = _read(R7_DECISION)
    assert doc, f"round-7 committed decision JSON missing: {R7_DECISION}"
    pm = doc["per_model"][model]
    d_r7 = float(pm["D"])
    w_on_max = float(pm["onpolicy"]["w_on_max"])
    if not smoke:
        assert abs(d_r7 - q["d_r7"]) < 1e-3, (model, d_r7, q["d_r7"])
        assert abs(w_on_max - q["w_on_max_r7"]) < 1e-3, (model, w_on_max, q["w_on_max_r7"])
    return {
        "D_r7": d_r7,
        "w_on_max_r7": w_on_max,
        "mlp_wins_max_r7": bool(pm["onpolicy"].get("mlp_wins_max")),
        "D_ci_rotated_r7": pm.get("D_ci_rotated"),
        "flag_rate_r7": q["flag_rate_r7"],
        "source": str(R7_DECISION.relative_to(REPO)),
    }


def arm_reads(out_dir: Path, model: str, arm: str) -> dict | None:
    """Within-strength reads @ the headline layer for one (model, arm) cell dir."""
    cells = out_dir / model / arm
    sep_path = cells / "cells_armC_sep.json"
    if not sep_path.exists():
        return None
    sep = json.loads(sep_path.read_text())
    hl = int(sep.get("headline_layer", 19))
    rot = float(sep["random_projection_control_r2"][str(hl)])
    ridge = float(sep["r2_per_layer_obs"][hl])
    mlp_doc = _read(cells / "mlp_secondary.json") or {"cells": {}}
    mlp_cell = (mlp_doc["cells"].get("armC_sep") or {}).get(str(hl)) or {}
    mlp = mlp_cell.get("r2_obs")
    mlp_ci = ((mlp_doc.get("mlp_ci") or {}).get("armC_sep") or {}).get(str(hl))
    per_group_mlp = ((mlp_doc.get("mlp_ci") or {}).get("armC_sep") or {}).get(
        "per_group_mlp_r2_headline"
    )
    rot_ci = (sep.get("rotated_bootstrap_group_frozen") or {}).get(str(hl))
    prev = _read(cells / "cells_armC_prevmean.json")
    return {
        "headline_layer": hl,
        "ridge": ridge,
        "rotated": rot,
        "mlp": mlp,
        "mlp_missing": mlp is None,
        "w_max": max(rot, mlp) if mlp is not None else rot,
        "mlp_wins_max": bool(mlp is not None and mlp > rot),
        "rotated_ci": rot_ci,
        "mlp_ci": mlp_ci,
        "per_group_rotated": sep.get("per_group_rotated_r2_headline"),
        "per_group_mlp": per_group_mlp,
        "prevmean_rotated": (
            float(prev["random_projection_control_r2"][str(hl)]) if prev else None
        ),
        "n": int(sep["n"]),
    }


def d_stat(w_on: float, w_ex: float, ceiling: float) -> float:
    denom = ceiling - w_ex
    return (w_on - w_ex) / denom if abs(denom) > 1e-12 else float("nan")


def map_ci_through_d(ci: dict | None, w_ex: float, ceiling: float, estimator: str) -> dict | None:
    if not ci:
        return None
    return {
        "lo": d_stat(float(ci["ci_lo"]), w_ex, ceiling),
        "hi": d_stat(float(ci["ci_hi"]), w_ex, ceiling),
        "estimator": estimator,
    }


def delta_dec_ci(r7: dict, d_ci_r8: dict | None, arm_b: dict, d_r7: float) -> dict | None:
    """R2 CI qualifier (plan v22 section 1): rotated var-sum when rotated
    carries BOTH sides; the r8 MLP-CI leg (r7 treated as the fixed reference)
    when MLP carries the r8 max; None => 'suggestive — margin-only'."""
    if d_ci_r8 is None:
        return None
    rotated_both = (not arm_b["mlp_wins_max"]) and (not r7["mlp_wins_max_r7"])
    if rotated_both and r7.get("D_ci_rotated_r7"):
        ci7 = r7["D_ci_rotated_r7"]
        se7 = abs(float(ci7["hi"]) - float(ci7["lo"])) / (2 * 1.96)
        se8 = abs(d_ci_r8["hi"] - d_ci_r8["lo"]) / (2 * 1.96)
        se = float(np.sqrt(se7**2 + se8**2))
        mid = (d_ci_r8["lo"] + d_ci_r8["hi"]) / 2 - d_r7
        return {
            "lo": mid - 1.96 * se,
            "hi": mid + 1.96 * se,
            "method": "rotated-var-sum (independent rotated group bootstraps, both rounds)",
        }
    return {
        "lo": d_ci_r8["lo"] - d_r7,
        "hi": d_ci_r8["hi"] - d_r7,
        "method": f"{d_ci_r8['estimator']}-ci-r8-only; D_r7 treated as the fixed reference "
        "(no round-7 MLP bootstrap exists — the gap the MLP-CI leg closes forward)",
    }


def paired_group_bootstrap(
    vals_a: dict[str, float], vals_b: dict[str, float], *, n_boot: int, seed: int
) -> dict | None:
    """Paired bootstrap over shared per-group values (groups resampled with
    replacement; statistic = mean per-group difference). Zero refits."""
    if not vals_a or not vals_b:
        return None
    shared = sorted(
        g for g in set(vals_a) & set(vals_b) if np.isfinite(vals_a[g]) and np.isfinite(vals_b[g])
    )
    if len(shared) < 3:
        return None
    diffs = np.asarray([float(vals_a[g]) - float(vals_b[g]) for g in shared])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    draws = diffs[picks].mean(axis=1)
    return {
        "mean_pergroup_diff": float(diffs.mean()),
        "ci_lo": float(np.quantile(draws, 0.025)),
        "ci_hi": float(np.quantile(draws, 0.975)),
        "n_groups": len(shared),
        "n_boot": int(n_boot),
    }


def flag_rate(data_root: Path, model: str, arm: str) -> dict | None:
    audit = _read(data_root / model / arm / "generation" / "generation_audit.json")
    if not audit:
        return None
    return {
        "repetition_rate_min5": audit.get("repetition_rate_min5"),
        "distinct_3gram_rate": audit.get("distinct_3gram_rate"),
        "early_eos_rate": audit.get("early_eos_rate"),
        "true_continuation_overlap_mean": (audit.get("true_continuation_overlap") or {}).get(
            "mean"
        ),
        "n_rows": audit.get("n_rows"),
    }


def _build_arms(args, m: str, refs: dict, anchor_key: str) -> dict[str, dict]:
    """Per-arm D reads for one model (factored out of main — C901)."""
    c = refs["ceiling_fulln"]
    arms: dict[str, dict] = {}
    for arm in ARMS:
        on = arm_reads(args.out_dir, m, arm)
        if on is None:
            arms[arm] = {"missing": True}
            continue
        meta = _read(args.data_root / m / arm / "pairs" / "pairs_meta.json") or {}
        if arm == "armB":
            realized_n = int(meta.get("realized_n", on["n"]))
            matched = _read(args.out_dir / anchor_key / f"matched_n_wex_armB_{m}.json")
            trigger_fired = bool(matched and matched.get("trigger_fired"))
            w_ex_matched = matched.get("w_ex_matched_n") if matched else None
            w_ex_eff = (
                w_ex_matched if (trigger_fired and w_ex_matched is not None) else refs["w_ex_fulln"]
            )
            w_ex_kind = (
                "matched_n" if (trigger_fired and w_ex_matched is not None) else "fulln_committed"
            )
        else:
            realized_n = on["n"]
            matched = _read(args.out_dir / anchor_key / f"matched_n_wex_armC_{m}.json")
            trigger_fired = bool(matched and matched.get("trigger_fired"))
            w_ex_matched = matched.get("w_ex_matched_n") if matched else None
            # Arm-C matched-n fires by construction (n ~ 600); a missing
            # file falls back to full-n with the fallback named in-payload.
            w_ex_eff = w_ex_matched if w_ex_matched is not None else refs["w_ex_fulln"]
            w_ex_kind = "matched_n_armC" if w_ex_matched is not None else "fulln_committed"
        d_eff = d_stat(on["w_max"], w_ex_eff, c)
        est = "mlp" if on["mlp_wins_max"] else "rotated"
        ci_src = on["mlp_ci"] if on["mlp_wins_max"] else on["rotated_ci"]
        d_ci = map_ci_through_d(ci_src, w_ex_eff, c, est)
        arms[arm] = {
            "reads": {
                k: on[k]
                for k in (
                    "headline_layer",
                    "ridge",
                    "rotated",
                    "mlp",
                    "mlp_missing",
                    "w_max",
                    "mlp_wins_max",
                    "rotated_ci",
                    "mlp_ci",
                    "prevmean_rotated",
                    "n",
                )
            },
            "realized_n": realized_n,
            "matched_n_trigger_fired": trigger_fired,
            "w_ex_effective": w_ex_eff,
            "w_ex_kind": w_ex_kind,
            "D": d_eff,
            "D_fulln_reference": d_stat(on["w_max"], refs["w_ex_fulln"], c),
            "D_ci": d_ci,
            "delta_w_raw": on["w_max"] - w_ex_eff,
            "ratio_ceiling": on["w_max"] / c,
        }
        if arm != "armB":
            arms[arm]["never_headline_note"] = (
                "BINDING (plan v22 section 1): arm-C reads are NEVER the headline D "
                "— C-avg is the E[v] noise-free ceiling read at a DIFFERENT anchor "
                "design (fixed prefix-final anchor, deterministic prefix-tail "
                "segment); cross-arm D comparisons carry both construct shifts"
            )
    return arms


def _r2_block(arms: dict, r7: dict) -> dict | None:
    """R2: decoding sensitivity (armB vs the frozen round-7 greedy D)."""
    if arms.get("armB", {}).get("missing"):
        return None
    ab = arms["armB"]
    delta = ab["D"] - r7["D_r7"]
    dci = delta_dec_ci(r7, ab["D_ci"], ab["reads"], r7["D_r7"])
    spans = bool(
        dci
        and (
            (dci["lo"] <= STABILITY_MARGIN <= dci["hi"])
            or (dci["lo"] <= -STABILITY_MARGIN <= dci["hi"])
        )
    )
    label = "CAVEAT_RETIRED" if abs(delta) <= STABILITY_MARGIN else "DECODING_SENSITIVE"
    qualifier = None
    if dci is None:
        qualifier = "suggestive — margin-only"
    elif spans:
        qualifier = "suggestive"
    return {
        "delta_dec": delta,
        "margin": STABILITY_MARGIN,
        "label": label,
        "qualifier": qualifier,
        "delta_ci": dci,
        "D_r7": r7["D_r7"],
        "D_armB": ab["D"],
    }


def _r3_block(args, m: str, arms: dict, refs: dict) -> dict | None:
    """R3: sampling-noise share (C-avg vs C-single, matched anchors/n)."""
    ca, cs = arms.get("armC_avg", {}), arms.get("armC_single", {})
    if ca.get("missing") or cs.get("missing") or "reads" not in ca or "reads" not in cs:
        return None
    on_a = arm_reads(args.out_dir, m, "armC_avg")
    on_s = arm_reads(args.out_dir, m, "armC_single")
    c = refs["ceiling_fulln"]
    remaining_gap = (
        (c - arms["armB"]["reads"]["w_max"]) if not arms["armB"].get("missing") else None
    )
    ns_rot = on_a["rotated"] - on_s["rotated"]
    ns_mlp = (
        (on_a["mlp"] - on_s["mlp"])
        if (on_a["mlp"] is not None and on_s["mlp"] is not None)
        else None
    )
    ns_max = on_a["w_max"] - on_s["w_max"]
    return {
        "ns_rotated": ns_rot,
        "ns_mlp": ns_mlp,
        "ns_max_interpretable": ns_max,
        "paired_bootstrap_rotated": paired_group_bootstrap(
            on_a["per_group_rotated"] or {},
            on_s["per_group_rotated"] or {},
            n_boot=args.n_boot,
            seed=common.FIT_SEED,
        ),
        "paired_bootstrap_mlp": paired_group_bootstrap(
            on_a["per_group_mlp"] or {},
            on_s["per_group_mlp"] or {},
            n_boot=args.n_boot,
            seed=common.FIT_SEED + 1,
        ),
        "remaining_gap_c_minus_w_b": remaining_gap,
        "ns_as_fraction_of_remaining_gap": (
            (ns_max / remaining_gap) if remaining_gap and abs(remaining_gap) > 1e-12 else None
        ),
        "note": (
            "descriptive read, no binary label (plan v22 section 1 R3); paired "
            "group bootstrap over the SHARED per-group R^2 values — a free "
            "post-hoc reduction, zero refits"
        ),
    }


def _r6_block(args, m: str, r7: dict) -> dict:
    """R6: de-looping premise check (per arm vs the round-7 greedy rate)."""
    out: dict[str, dict | None] = {}
    for arm in ("armB", "armC"):
        fr = flag_rate(args.data_root, m, arm)
        if fr is None:
            out[arm] = None
            continue
        rate = fr.get("repetition_rate_min5")
        drop = (r7["flag_rate_r7"] - rate) if rate is not None else None
        out[arm] = {
            **fr,
            "flag_rate_r7_greedy": r7["flag_rate_r7"],
            "drop_points": drop,
            "premise_partially_fails": (
                bool(drop is not None and drop < R6_DROP_POINTS) if drop is not None else None
            ),
        }
    return out


def _gates_for_model(args, m: str, anchor_key: str) -> dict[str, object]:
    gates: dict[str, object] = {}
    gates[f"anchor_gate_{m}"] = (_read(args.out_dir / anchor_key / "anchor_gate.json") or {}).get(
        "pass"
    )
    gates[f"r7_repro_gate_{m}"] = (
        _read(args.out_dir / f"repro_{m}" / "anchor_gate.json") or {}
    ).get("pass")
    for arm in ("armB", "armC"):
        eq = _read(args.data_root / m / arm / "store" / "armC" / "armC_equivalence.json")
        gates[f"equivalence_gate_{m}_{arm}"] = (
            {"early_cos_min": eq.get("early_cos_min"), "flat_cos_min": eq.get("flat_cos_min")}
            if eq
            else None
        )
    red = _read(args.out_dir / m / "reduce_summary.json")
    gates[f"x_identity_gate_{m}"] = (
        {
            "pass": (red.get("x_identity_gate") or {}).get("pass"),
            "min": (red.get("x_identity_gate") or {}).get("min"),
            "binding": (red.get("x_identity_gate") or {}).get("binding"),
        }
        if red
        else None
    )
    gates[f"posmatch_below256_{m}_present"] = bool(
        _read(args.out_dir / anchor_key / f"posmatch_below256_{m}.json")
    )
    return gates


def main() -> int:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    per_model: dict[str, dict] = {}
    gates: dict[str, object] = {}
    for m in models:
        refs = committed_refs(m, smoke=args.smoke)
        r7 = r7_reference(m, smoke=args.smoke)
        anchor_key = "anchor_base" if m == "base" else "anchor_inst"
        arms = _build_arms(args, m, refs, anchor_key)
        per_model[m] = {
            "committed_reference": refs,
            "round7_reference": r7,
            "arms": arms,
            "r2_decoding_sensitivity": _r2_block(arms, r7),
            "r3_sampling_noise_share": _r3_block(args, m, arms, refs),
            "r6_flag_rates": _r6_block(args, m, r7),
        }
        gates.update(_gates_for_model(args, m, anchor_key))

    # R5 mirror per arm in {B, C-avg}.
    r5 = {}
    if "base" in per_model and "instruct" in per_model:
        for arm in ("armB", "armC_avg"):
            ab, ai = (
                per_model["base"]["arms"].get(arm, {}),
                per_model["instruct"]["arms"].get(arm, {}),
            )
            if ab.get("missing") or ai.get("missing") or "D" not in ab or "D" not in ai:
                r5[arm] = None
                continue
            gap = abs(ab["D"] - ai["D"])
            r5[arm] = {
                "abs_d_gap": gap,
                "margin": STABILITY_MARGIN,
                "mirror_within_margin": bool(gap <= STABILITY_MARGIN),
                "delta_w_raw_base": ab["delta_w_raw"],
                "delta_w_raw_instruct": ai["delta_w_raw"],
                "note": "raw dW reported alongside — the D denominators differ "
                "(0.22507 vs 0.32417), so a D gap can be a denominator artifact",
            }

    md = common.metadata(SCRIPT, common.FIT_SEED, 0)
    md["issue"] = 825  # reviewer fix: the round-7 JSON inherited 931 here
    payload = {
        "metadata": md,
        "followup_label": FOLLOWUP_LABEL,
        "smoke": bool(args.smoke),
        "per_model": per_model,
        "r5_mirror": r5,
        "gates": gates,
        "transfer_fractions": {f"{m}_{arm}": None for m in models for arm in ("armB", "armC_avg")}
        | {
            "note": (
                "R4 slot — filled by Phase C on the VM "
                "(sampled_sep_to_chat_{base,instruct}_{armB,armC_avg}.json, full-n "
                "convention, fraction of the full-n chat ceiling); deliberately "
                "outside the pod-side enumeration (plan v22 section 6.5)"
            ),
            "threshold": 0.5,
        },
        "interpretation_bands": {
            "r2_margin": STABILITY_MARGIN,
            "r5_margin": STABILITY_MARGIN,
            "r6_drop_points": R6_DROP_POINTS,
            "armB_matched_n_trigger": ARMB_MATCHED_N_TRIGGER,
            "rule": (
                "BINDING, applied by the ANALYZER (plan v22 section 1): R2 CAVEAT_RETIRED "
                "iff |Delta_dec| <= 0.10 per substrate, DECODING_SENSITIVE otherwise "
                "(direction reported); CI spanning the +-0.10 boundary => 'suggestive'; "
                "arm-C D values NEVER the headline (construct-shift honesty rule); a "
                "negative Delta_dec is a decoding-sensitivity finding, not "
                "answer-specificity strengthening (plan section 7 concern 4)"
            ),
        },
    }
    out_path = args.out_dir / "decision_support.json"
    common.write_json(out_path, payload)
    for m in models:
        pm = per_model[m]
        r2b = pm["r2_decoding_sensitivity"] or {}
        print(
            f"[i825-ss-dec] {m}: D_armB={r2b.get('D_armB')} D_r7={r2b.get('D_r7')} "
            f"delta_dec={r2b.get('delta_dec')} label={r2b.get('label')} "
            f"qualifier={r2b.get('qualifier')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
