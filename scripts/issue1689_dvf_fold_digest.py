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

``--paired`` (the `wellposed-shared-readout` round, plan v10 s3/s6.5): joins
the parent's realized AMBIENT per-unit JSONs against this round's REDUCED
(*_wellposed) per-unit JSONs on the model-qualified ``unit_key``, and emits
the registered paired delta digest (``dvf_wellposed_paired_digest.csv``) +
a summary JSON with the pooled AND k-band-stratified rung-1 calibration
rates, verdict-flip matrix, concordance rho, eff-rank comparison, and the
588/588 coverage reconciliation (holes ENUMERATED, never dropped).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import k_band  # noqa: E402

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


def _spearman(a, b) -> float:
    """Spearman rho (same convention as the batteries' merge concordance)."""
    from scipy.stats import spearmanr

    if len(a) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _load_units_dir(root: Path) -> dict[str, dict]:
    """unit_key -> unit JSON (non-error units only) from <root>/pairs/*.json."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(str(root / "pairs" / "*.json"))):
        u = json.loads(Path(f).read_text())
        if isinstance(u, dict) and "error" not in u:
            out[Path(f).stem] = u
    return out


def _digest_index(digest_csv: Path) -> dict[str, dict]:
    """Parent digest rows keyed by the model-qualified unit_key.

    dvf_within rows key as <model>__<pair>__<arm>; xm_dvf rows reconstruct
    the cross-model pair_spec_key <sm>@<cond>__<tm>@<cond>__<arm>.
    """
    out: dict[str, dict] = {}
    if not digest_csv.exists():
        return out
    with open(digest_csv) as fh:
        for row in csv.DictReader(fh):
            b = row.get("battery")
            if b == "dvf_within":
                out[f"{row['model']}__{row['pair']}__{row['arm']}"] = row
            elif b == "xm_dvf":
                sm_, tm_ = row["model"].split("->")
                cond = row["pair"]
                out[f"{sm_}@{cond}__{tm_}@{cond}__{row['arm']}"] = row
    return out


def _flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def paired_main(args) -> int:
    """The `wellposed-shared-readout` paired ambient-vs-reduced delta digest.

    Pairing key = model-qualified unit_key (plan v10 s3); ambient arm = the
    parent's realized per-unit JSONs; reduced arm = this round's *_wellposed
    JSONs. Coverage holes are ENUMERATED (summary JSON), never dropped.
    """
    dig = _digest_index(args.digest_csv)
    rows: list[dict] = []
    coverage: dict[str, dict] = {}
    flip: dict[str, int] = {}
    calib: dict[str, dict] = {}
    conc_pool: dict[str, dict[str, list]] = {}

    for battery, amb_root, red_root in (
        ("dvf_within", args.ambient_dvf_root, args.reduced_dvf_root),
        ("xm_dvf", args.ambient_xm_root, args.reduced_xm_root),
    ):
        amb = _load_units_dir(amb_root)
        red = _load_units_dir(red_root)
        matched = sorted(set(amb) & set(red))
        coverage[battery] = {
            "n_ambient": len(amb),
            "n_reduced": len(red),
            "n_matched": len(matched),
            "ambient_only": sorted(set(amb) - set(red))[:50],
            "reduced_only": sorted(set(red) - set(amb))[:50],
        }
        for uk in matched:
            a, r = amb[uk], red[uk]
            drow = dig.get(uk, {})
            k_u = int(r.get("k_unit") or 0)
            band = k_band(k_u)
            v_a, v_r = a["verdict"], r["verdict"]
            flip[f"{v_a}->{v_r}"] = flip.get(f"{v_a}->{v_r}", 0) + 1
            parent_rung = drow.get("parent_rung", "")
            cap_y = [
                m.get("captured_var_test_y")
                for m in (r.get("pca_basis_per_fold") or {}).values()
                if m.get("captured_var_test_y") is not None
            ]
            rows.append(
                {
                    "battery": battery,
                    "unit_key": uk,
                    "model": a["src_model"]
                    if not a["cross_model"]
                    else f"{a['src_model']}->{a['tgt_model']}",
                    "pair": a["pair_key"] if not a["cross_model"] else a["src_cond"],
                    "arm": a["arm"],
                    "informative": drow.get("informative", ""),
                    "parent_rung": parent_rung,
                    "n_common": a["n_common"],
                    "k_unit": k_u,
                    "k_band": band,
                    "k_floor_limited": int(bool(r.get("k_floor_limited"))),
                    "ambient_verdict": v_a,
                    "reduced_verdict": v_r,
                    "verdict_flip": int(v_a != v_r),
                    "ambient_g1": a["g1"],
                    "reduced_g1": r["g1"],
                    "delta_g1": _flt(r["g1"]) - _flt(a["g1"]),
                    "ambient_g2": a["g2"],
                    "reduced_g2": r["g2"],
                    "delta_g2": _flt(r["g2"]) - _flt(a["g2"]),
                    "ambient_r2_b_free": a["r2_b_free"],
                    "reduced_r2_b_free": r["r2_b_free"],
                    "reduced_r2_b_free_ambient_recon": r.get("r2_b_free_ambient_recon", ""),
                    "ambient_r2_identity_bias": a["r2_identity_bias"],
                    "reduced_r2_identity_bias": r["r2_identity_bias"],
                    "captured_var_test_y_min": min(cap_y) if cap_y else "",
                }
            )
            # Rung-1 calibration (plan s6): within-model, class-0-excluded
            # PER BASIS, pooled + per k-band.
            if battery == "dvf_within" and str(parent_rung) == "1":
                for basis, v in (("ambient", v_a), ("reduced", v_r)):
                    for stratum in ("pooled", band):
                        c = calib.setdefault(basis, {}).setdefault(
                            stratum, {"n_class0_excluded": 0, "n_shared_supported": 0}
                        )
                        if v != "free_map_uninformative":
                            c["n_class0_excluded"] += 1
                            c["n_shared_supported"] += int(v == "shared_readout_supported")
            # Concordance pool (parent merge convention: within, rung known,
            # class-0-excluded per basis, finite g1).
            if battery == "dvf_within" and parent_rung not in ("", None):
                for basis, unit in (("ambient", a), ("reduced", r)):
                    if unit["verdict"] == "free_map_uninformative":
                        continue
                    g1v = _flt(unit["g1"])
                    if g1v == g1v:  # not NaN
                        cp = conc_pool.setdefault(f"{basis}|{a['arm']}", {"rung": [], "g1": []})
                        cp["rung"].append(int(parent_rung))
                        cp["g1"].append(g1v)

    # cms paired rows (H-effrank read: eff-rank(M-I)/fit_dim, ambient vs reduced).
    eff_pairs: dict[str, dict[str, list[float]]] = {}
    for battery, amb_root, red_root in (
        ("cms_within", args.ambient_cms_root, args.reduced_cms_root),
        ("cms_xm", args.ambient_xms_root, args.reduced_xms_root),
    ):
        amb = _load_units_dir(amb_root)
        red = _load_units_dir(red_root)
        matched = sorted(set(amb) & set(red))
        coverage[battery] = {
            "n_ambient": len(amb),
            "n_reduced": len(red),
            "n_matched": len(matched),
            "ambient_only": sorted(set(amb) - set(red))[:50],
            "reduced_only": sorted(set(red) - set(amb))[:50],
        }
        for uk in matched:
            a, r = amb[uk], red[uk]
            k_u = int(r.get("k_unit") or 0)
            ea = _flt(a["distance_from_identity"]["eff_rank_m_minus_i"]) / max(int(a["d"]), 1)
            er = _flt(r["distance_from_identity"]["eff_rank_m_minus_i"]) / max(
                int(r.get("fit_dim") or r["d"]), 1
            )
            ep = eff_pairs.setdefault(battery, {"ambient": [], "reduced": []})
            ep["ambient"].append(ea)
            ep["reduced"].append(er)
            rows.append(
                {
                    "battery": battery,
                    "unit_key": uk,
                    "model": a["src_model"]
                    if not a["cross_model"]
                    else f"{a['src_model']}->{a['tgt_model']}",
                    "pair": a["pair_key"] if not a["cross_model"] else a["src_cond"],
                    "arm": a["arm"],
                    "n_common": a["n_common"],
                    "k_unit": k_u,
                    "k_band": k_band(k_u),
                    "k_floor_limited": int(bool(r.get("k_floor_limited"))),
                    "ambient_weakest_class": a["weakest_class_point"],
                    "reduced_weakest_class": r["weakest_class_point"],
                    "ambient_eff_rank_frac": ea,
                    "reduced_eff_rank_frac": er,
                }
            )

    for basis in calib:
        for stratum, c in calib[basis].items():
            c["rate"] = (
                c["n_shared_supported"] / c["n_class0_excluded"]
                if c["n_class0_excluded"]
                else float("nan")
            )
    concordance = {
        gk: {
            "n": len(cp["rung"]),
            "spearman_rung_g1": _spearman(cp["rung"], cp["g1"]),
        }
        for gk, cp in conc_pool.items()
    }
    eff_summary = {}
    for battery, ep in eff_pairs.items():
        amb_v = sorted(ep["ambient"])
        red_v = sorted(ep["reduced"])
        eff_summary[battery] = {
            "n": len(amb_v),
            "median_ambient_eff_rank_frac": amb_v[len(amb_v) // 2] if amb_v else float("nan"),
            "median_reduced_eff_rank_frac": red_v[len(red_v) // 2] if red_v else float("nan"),
        }

    fields: list[str] = []
    for r_ in rows:
        for k in r_:
            if k not in fields:
                fields.append(k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "coverage": coverage,
        "rung1_calibration": calib,
        "verdict_flip_matrix": flip,
        "concordance": concordance,
        "eff_rank": eff_summary,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.summary_out.with_name(f".{args.summary_out.name}.tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(args.summary_out)
    print(f"wrote {args.out} ({len(rows)} rows) + {args.summary_out}")
    for b, cov in coverage.items():
        print(
            f"[paired] {b}: matched {cov['n_matched']} "
            f"(ambient {cov['n_ambient']} / reduced {cov['n_reduced']}; "
            f"ambient_only {len(cov['ambient_only'])} reduced_only {len(cov['reduced_only'])})"
        )
    return 0


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


def _entry() -> int:
    """Argparse dispatch: default = the parent digest (byte-unchanged flow);
    --paired = the wellposed paired ambient-vs-reduced delta digest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--ambient-dvf-root", type=Path, default=BASE / "derived_vs_free_B")
    ap.add_argument("--ambient-xm-root", type=Path, default=BASE / "crossmodel_pairs")
    ap.add_argument("--ambient-cms-root", type=Path, default=BASE / "context_map_structure")
    ap.add_argument(
        "--ambient-xms-root",
        type=Path,
        default=BASE / "crossmodel_pairs" / "crossmodel_structure",
    )
    ap.add_argument("--reduced-dvf-root", type=Path, default=BASE / "derived_vs_free_wellposed")
    ap.add_argument("--reduced-xm-root", type=Path, default=BASE / "crossmodel_pairs_wellposed")
    ap.add_argument(
        "--reduced-cms-root", type=Path, default=BASE / "context_map_structure_wellposed"
    )
    ap.add_argument(
        "--reduced-xms-root",
        type=Path,
        default=None,
        help="default: <reduced-xm-root>/crossmodel_structure_wellposed",
    )
    ap.add_argument("--digest-csv", type=Path, default=BASE / "analyzer" / "dvf_unit_digest.csv")
    ap.add_argument(
        "--out", type=Path, default=BASE / "analyzer" / "dvf_wellposed_paired_digest.csv"
    )
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=BASE / "analyzer" / "dvf_wellposed_paired_summary.json",
    )
    args = ap.parse_args()
    if args.reduced_xms_root is None:
        args.reduced_xms_root = args.reduced_xm_root / "crossmodel_structure_wellposed"
    if args.paired:
        return paired_main(args)
    return main()


if __name__ == "__main__":
    rc = _entry()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer. All outputs are
    # flushed/closed before this point; atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
