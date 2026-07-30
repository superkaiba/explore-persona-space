"""Per-unit digest for the #1689 derived-vs-free-answer-map follow-up round.

Reads the three battery output trees (``derived_vs_free_B/``,
``context_map_structure/``, ``crossmodel_pairs/``) plus the parent ladder
JSONs and the parent analyzer ``pair_digest.csv`` validity flags, and writes
ONE flat per-unit CSV (``analyzer/dvf_unit_digest.csv``) the fold figures and
body prose read from. All counts in the clean-result body come from this
digest (i.e., from the per-unit JSONs), never from the batteries'
``summary.json`` files, whose merge double-counts every surviving unit file
(the within-model unit key omits the model; see the body's coverage note).

Run from the issue worktree root:
    uv run python scripts/issue1689_dvf_fold_digest.py
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

BASE = Path("eval_results/issue_1689")
OUT = BASE / "analyzer" / "dvf_unit_digest.csv"


def parent_rungs() -> dict:
    out = {}
    for m in ("Qwen_Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct"):
        p = BASE / "ladder" / f"ladder_{m}_L19.json"
        lad = json.loads(p.read_text())
        for pk, arms in lad["pairs"].items():
            for arm, res in arms.items():
                if isinstance(res, dict) and "rung_reached_point" in res:
                    out[(m, pk, arm)] = int(res["rung_reached_point"])
    return out


def parent_validity() -> dict:
    out = {}
    with open(BASE / "analyzer" / "pair_digest.csv") as f:
        for row in csv.DictReader(f):
            m = "Qwen_Qwen2.5-7B" if row["model"] == "base" else "Qwen_Qwen2.5-7B-Instruct"
            out[(m, row["pair"], row["arm"])] = row
    return out


def xm_arm_invalid(cond: str, arm: str) -> bool:
    """Construct-invalid arms for cross-model same-condition pairs.

    Mirrors the parent per-cell validity read: user-cell context arms are
    self-predictions; plain-text (naturalistic) user prefix arms collapse too.
    """
    if cond.startswith("user_"):
        if arm == "context":
            return True
        if arm == "prefix" and "naturalistic" in cond:
            return True
    return False


def main() -> int:
    rungs = parent_rungs()
    valid = parent_validity()
    rows = []

    for f in sorted(glob.glob(str(BASE / "derived_vs_free_B/pairs/*.json"))):
        u = json.loads(Path(f).read_text())
        dr = valid.get((u["src_model"], u["pair_key"], u["arm"]))
        rows.append(
            {
                "battery": "dvf_within",
                "model": u["src_model"],
                "pair": u["pair_key"],
                "arm": u["arm"],
                "cls": dr["cls"] if dr else "",
                "informative": int(
                    bool(dr) and dr["arm_invalid"] == "0" and dr["degenerate_ceiling"] == "0"
                ),
                "parent_rung": rungs.get((u["src_model"], u["pair_key"], u["arm"]), ""),
                "verdict": u["verdict"],
                "verdict_fixed_effrank": u["verdict_fixed_effrank"],
                "g1": u["g1"],
                "g2": u["g2"],
                "r2_b_free": u["r2_b_free"],
                "r2_identity_bias": u["r2_identity_bias"],
                "r2_b_derived_max": u["r2_b_derived_max"],
                "r2_b_derived2_max": u["r2_b_derived2_max"],
                "knn_acc1_free": u["knn"]["b_free"]["euclidean"]["acc_at_k"]["1"]
                if u["knn"].get("b_free")
                else "",
                "knn_chance1": u["knn"]["b_free"]["euclidean"]["chance_at_k"]["1"]
                if u["knn"].get("b_free")
                else "",
                "cos_derived_free": (u.get("operator_read") or {})
                .get("raw_cosine", {})
                .get("derived_effrank", ""),
                "cos_derived2_free": (u.get("operator_read") or {})
                .get("raw_cosine", {})
                .get("derived2_effrank", ""),
                "n_common": u["n_common"],
            }
        )

    for f in sorted(glob.glob(str(BASE / "context_map_structure/pairs/*.json"))):
        u = json.loads(Path(f).read_text())
        dr = valid.get((u["src_model"], u["pair_key"], u["arm"]))
        rr = u.get("rank_rung") or {}
        di = u["distance_from_identity"]
        rows.append(
            {
                "battery": "cms_within",
                "model": u["src_model"],
                "pair": u["pair_key"],
                "arm": u["arm"],
                "cls": dr["cls"] if dr else "",
                "informative": int(
                    bool(dr) and dr["arm_invalid"] == "0" and dr["degenerate_ceiling"] == "0"
                ),
                "parent_rung": rungs.get((u["src_model"], u["pair_key"], u["arm"]), ""),
                "weakest_class": u["weakest_class_point"],
                "fro_ratio_m_minus_i": di["fro_ratio_m_minus_i_over_m"],
                "eff_rank_m_minus_i": di["eff_rank_m_minus_i"],
                "gain_full_over_translation_r2": di["gain_full_over_translation_r2"],
                "dom_max_abs_cos_top8_out": max(
                    u["diff_of_means_alignment"]["top8_output_dir_abs_cos"] or [0.0]
                ),
                "rank_eligible": int(bool(rr.get("eligible"))),
                "k_reached_ctx": rr.get("k_reached_ctx", ""),
                "k_reached_ans": rr.get("k_reached_ans", ""),
                "n_common": u["n_common"],
            }
        )

    xl = json.loads((BASE / "crossmodel_pairs/ladder_crossmodel_L19.json").read_text())
    for pk, arms in xl["pairs"].items():
        direction = "base->instruct" if pk.startswith("Qwen_Qwen2.5-7B@") else "instruct->base"
        cond = pk.split("@")[1].split("__")[0]
        for arm, res in arms.items():
            ceiling = float(res["r2_within_target"])
            r9 = res["rung_r2s_point"]["rung_9_full_AMB"]
            rows.append(
                {
                    "battery": "xm_ladder",
                    "model": direction,
                    "pair": cond,
                    "arm": arm,
                    "informative": int((not xm_arm_invalid(cond, arm)) and ceiling > 0),
                    "verdict": "",
                    "xm_ceiling": ceiling,
                    "xm_rung_reached": int(res["rung_reached_point"]),
                    "xm_r2_rung9": r9,
                    "xm_rung9_reconciles": int(r9 >= res["reach_bar_90pct"]),
                    "xm_rec9": (r9 / ceiling) if ceiling > 0 else "",
                    "n_common": res["n_common"],
                }
            )

    for f in sorted(glob.glob(str(BASE / "crossmodel_pairs/pairs/*.json"))):
        u = json.loads(Path(f).read_text())
        rows.append(
            {
                "battery": "xm_dvf",
                "model": f"{u['src_model']}->{u['tgt_model']}",
                "pair": u["src_cond"],
                "arm": u["arm"],
                "informative": int(not xm_arm_invalid(u["src_cond"], u["arm"])),
                "verdict": u["verdict"],
                "g1": u["g1"],
                "g2": u["g2"],
                "r2_b_free": u["r2_b_free"],
                "n_common": u["n_common"],
            }
        )

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
