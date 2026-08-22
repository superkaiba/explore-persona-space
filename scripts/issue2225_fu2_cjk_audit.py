"""Language-intrusion (CJK) audit + excluded-intrusion recount for the #2225 fu2 round.

Step 3.7 duty (analyzer): Qwen-family completions under a non-CJK eval owe a per-arm
CJK scan over every judged pool that a verdict rests on, plus excluded-intrusion
recounts of the adjudication-bearing statistics.  Pure counting — no completion text
is ever printed; only aggregate counts and recomputed statistics leave this script.

Scope (fu2 = ``fu2_preimage_alltoken``, all-token pre-image/random steering):
  (a) all 30 fu2 rollout units (28 wave-1 + 2 W2b RN cells) — per-unit
      intruded/total counts over the CJK regex class shared with the fu1 audit;
  (b) the anchor units the fu2 contrasts consume: parent G_evil @3.0 (H2 headline),
      parent A at its operating points (H2 secondary), and the 12 banked fu1
      J/K/L/M units at their fu1 operating points (H3);
  (c) excluded-intrusion recount of every computable frozen contrast in
      analysis/contrasts.json (H1 dose, H2 vs parent G, H2 secondary vs parent A,
      H3 vs fu1 positions, H4 difference-of-dose reads incl. the W2b matched-window
      and matched-dose-level reads), rebuilt from the per-question
      ``rollout_scores`` matrices with intruded (question, rollout) entries masked,
      question-paired bootstrap (10,000 resamples, seed 2225).

The recount seed stream is a single seed (2225), not the driver's per-contrast
offsets, so CI endpoints differ in the third decimal from contrasts.json; verdict
flips are read against the committed labels.

Usage (defaults match the fu2 VM staging layout):
  uv run python scripts/issue2225_fu2_cjk_audit.py \
      --fu2-rc-final /mnt/eps-data/thomasjiralerspong/issue2225_fu2/raw_completions/final \
      --fu1-rc-final /mnt/eps-data/thomasjiralerspong/issue2225_fu1/raw_completions/final \
      --parent-rc-dir /mnt/eps-data/thomasjiralerspong/issue2225_fu1/hf_dl/parent_anchor_rc/issue2225_ctxsteer/raw_completions/final \
      --parent-g-rc /mnt/eps-data/thomasjiralerspong/issue2225_fu2/parent_anchor_rc/G__evil__c3.0__evil.json \
      --out eval_results/issue_2225/fu2_preimage_alltoken/analysis/language_intrusion_audit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind before numpy import

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "issue2225_fu1_cjk_audit", _HERE / "issue2225_fu1_cjk_audit.py"
)
assert _spec is not None and _spec.loader is not None
_fu1 = importlib.util.module_from_spec(_spec)
sys.modules["issue2225_fu1_cjk_audit"] = _fu1
_spec.loader.exec_module(_fu1)

boot_delta = _fu1.boot_delta
intrusion_mask = _fu1.intrusion_mask
question_means = _fu1.question_means
score_matrix = _fu1.score_matrix
unit_counts = _fu1.unit_counts

TRAIT_FOR_DS = _fu1.TRAIT_FOR_DS
PARENT_A_OP = _fu1.PARENT_OP["A"]
PARENT_G_OP = 3.0  # parent analysis/selection.json: G_evil selected_coef
FU1_OP = 3.0  # fu1 analysis/selection.json: every J/K/L/M arm selects 3.0
H3_PAIRS = {"N": ("J", "L"), "Q": ("K", "M")}  # fu2 arm -> fu1 comparators (same layer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fu2-rc-final", type=Path, required=True)
    ap.add_argument("--fu1-rc-final", type=Path, required=True)
    ap.add_argument("--parent-rc-dir", type=Path, required=True)
    ap.add_argument("--parent-g-rc", type=Path, required=True)
    ap.add_argument(
        "--fu2-analysis-dir",
        type=Path,
        default=Path("eval_results/issue_2225/fu2_preimage_alltoken"),
    )
    ap.add_argument(
        "--fu1-analysis-dir",
        type=Path,
        default=Path("eval_results/issue_2225/fu1_preimage_prevention"),
    )
    ap.add_argument(
        "--parent-trait-scores-dir",
        type=Path,
        default=Path("eval_results/issue_2225/trait_scores"),
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    per_unit: dict[str, dict[str, int]] = {}
    total = intruded = 0
    for rc in sorted(args.fu2_rc_final.glob("*.json")):
        ni, n = unit_counts(rc)
        per_unit[rc.name] = {"intruded": ni, "total": n}
        total += n
        intruded += ni

    anchor_units: dict[str, dict[str, int]] = {}
    anchor_paths: dict[str, Path] = {"G__evil__c3.0__evil.json": args.parent_g_rc}
    for ds, coef in PARENT_A_OP.items():
        name = f"A__{ds}__c{coef}__{TRAIT_FOR_DS[ds]}.json"
        anchor_paths[name] = args.parent_rc_dir / name
    for cfgs in H3_PAIRS.values():
        for cfg in cfgs:
            for ds in ("evil", "sycophancy", "hallucination"):
                name = f"{cfg}__{ds}__c{FU1_OP}__{TRAIT_FOR_DS[ds]}.json"
                anchor_paths[name] = args.fu1_rc_final / name
    for name, p in sorted(anchor_paths.items()):
        ni, n = unit_counts(p)
        anchor_units[name] = {"intruded": ni, "total": n}

    contrasts = json.loads((args.fu2_analysis_dir / "analysis" / "contrasts.json").read_text())
    selection = json.loads((args.fu2_analysis_dir / "analysis" / "selection.json").read_text())[
        "selection"
    ]
    fu2_ts = args.fu2_analysis_dir / "trait_scores"
    fu1_ts = args.fu1_analysis_dir / "trait_scores"

    def masked_qmeans(ts_path: Path, rc_path: Path, trait: str) -> np.ndarray:
        return question_means(score_matrix(ts_path, trait), intrusion_mask(rc_path))

    def fu2_qmeans(cfg: str, ds: str, coef) -> np.ndarray:
        trait = TRAIT_FOR_DS[ds]
        return masked_qmeans(
            fu2_ts / f"{cfg}_{ds}_{coef}.json",
            args.fu2_rc_final / f"{cfg}__{ds}__c{coef}__{trait}.json",
            trait,
        )

    def fu1_qmeans(cfg: str, ds: str) -> np.ndarray:
        trait = TRAIT_FOR_DS[ds]
        return masked_qmeans(
            fu1_ts / f"{cfg}_{ds}_{FU1_OP}.json",
            args.fu1_rc_final / f"{cfg}__{ds}__c{FU1_OP}__{trait}.json",
            trait,
        )

    def sel_coef(cfg: str, ds: str):
        return selection[f"{cfg}_{ds}"]["selected_coef"]

    recount: dict[str, dict] = {}
    flips: list[dict] = []

    def record(section: str, arm: str, res: dict, committed: str) -> None:
        res["committed_verdict"] = committed
        res["flip"] = res["verdict"] != committed
        recount.setdefault(section, {})[arm] = res
        if res["flip"]:
            flips.append({"section": section, "arm": arm, **res})

    for arm, entry in sorted(contrasts["h1_dose"]["per_arm"].items()):
        cfg, ds = arm.split("_", 1)
        a = fu2_qmeans(cfg, ds, sel_coef(cfg, ds))
        b = fu2_qmeans(cfg, ds, entry["smallest_coef"])
        record("h1_dose", arm, boot_delta(a, b), entry["frozen"]["verdict"])

    g_ref = masked_qmeans(
        args.parent_trait_scores_dir / f"G_evil_{PARENT_G_OP}.json",
        args.parent_g_rc,
        "evil",
    )
    for arm, entry in sorted(contrasts["h2_vs_parent_G"]["per_arm"].items()):
        cfg, ds = arm.split("_", 1)
        a = fu2_qmeans(cfg, ds, sel_coef(cfg, ds))
        record("h2_vs_parent_G", arm, boot_delta(a, g_ref), entry["frozen"]["verdict"])

    for arm, entry in sorted(contrasts["h2_secondary_vs_parent_A"]["per_arm"].items()):
        cfg, ds = arm.split("_", 1)
        coef_a = PARENT_A_OP[ds]
        a = fu2_qmeans(cfg, ds, sel_coef(cfg, ds))
        b = masked_qmeans(
            args.parent_trait_scores_dir / f"A_{ds}_{coef_a}.json",
            args.parent_rc_dir / f"A__{ds}__c{coef_a}__{TRAIT_FOR_DS[ds]}.json",
            TRAIT_FOR_DS[ds],
        )
        record("h2_secondary_vs_parent_A", arm, boot_delta(a, b), entry["frozen"]["verdict"])

    for pair, entry in sorted(contrasts["h3_vs_fu1_positions"]["per_pair"].items()):
        fu2_cfg, fu1_cfg, ds = pair.split("_vs_")[0], pair.split("_")[2], pair.split("_", 3)[3]
        a = fu2_qmeans(fu2_cfg, ds, sel_coef(fu2_cfg, ds))
        b = fu1_qmeans(fu1_cfg, ds)
        record("h3_vs_fu1_positions", pair, boot_delta(a, b), entry["frozen"]["verdict"])

    h4 = contrasts["h4_direction_specificity"]
    for pair, entry in sorted(h4["per_pair"].items()):
        cfg = pair.split("_vs_")[0]
        dose_a = fu2_qmeans(cfg, "evil", sel_coef(cfg, "evil")) - fu2_qmeans(
            cfg, "evil", contrasts["h1_dose"]["per_arm"][f"{cfg}_evil"]["smallest_coef"]
        )
        dose_b = fu2_qmeans("RQ", "evil", sel_coef("RQ", "evil")) - fu2_qmeans(
            "RQ", "evil", contrasts["h1_dose"]["per_arm"]["RQ_evil"]["smallest_coef"]
        )
        record("h4_dod_vs_RQ", pair, boot_delta(dose_a, dose_b), entry["frozen"]["verdict"])

    w2b = h4["w2b_conditional"]["per_pair"]
    dod = w2b["N_vs_RN_sycophancy"]
    lo_w, hi_w = dod["window"]
    dose_n = fu2_qmeans("N", "sycophancy", hi_w) - fu2_qmeans("N", "sycophancy", lo_w)
    dose_rn = fu2_qmeans("RN", "sycophancy", hi_w) - fu2_qmeans("RN", "sycophancy", lo_w)
    record("h4_w2b", "N_vs_RN_sycophancy", boot_delta(dose_n, dose_rn), dod["frozen"]["verdict"])
    lvl = w2b["N_vs_RN_sycophancy_level"]
    a = fu2_qmeans("N", "sycophancy", lvl["matched_coef"])
    b = fu2_qmeans("RN", "sycophancy", lvl["matched_coef"])
    record("h4_w2b", "N_vs_RN_sycophancy_level", boot_delta(a, b), lvl["frozen"]["verdict"])

    n_contrasts = sum(len(v) for v in recount.values())
    out = {
        "note": (
            "fu2 language-intrusion audit: per-unit CJK counts over every judged fu2 pool "
            "+ consumed parent/fu1 anchors, and excluded-intrusion recounts of every "
            "computable frozen contrast (single seed 2225, so CI endpoints differ in the "
            "third decimal from contrasts.json; flips read against committed verdicts)"
        ),
        "regex_class": _fu1.CJK.pattern,
        "fu2_units": {"n_units": len(per_unit), "intruded": intruded, "total": total},
        "per_unit": per_unit,
        "anchor_units": anchor_units,
        "recount": recount,
        "n_contrasts_recounted": n_contrasts,
        "flips": flips,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    rates = sorted(v["intruded"] / v["total"] for v in per_unit.values())
    print(
        f"fu2 pools: {intruded}/{total} intruded ({100 * intruded / total:.2f}%); "
        f"per-unit range {100 * rates[0]:.1f}%-{100 * rates[-1]:.1f}%"
    )
    print(f"recounted {n_contrasts} contrasts; flips: {len(flips)}")
    for f in flips:
        print(
            f"  FLIP {f['section']}/{f['arm']}: {f['committed_verdict']} -> {f['verdict']} "
            f"(delta {f['delta_point']:.2f}, ci {f['ci95']})"
        )


if __name__ == "__main__":
    main()
