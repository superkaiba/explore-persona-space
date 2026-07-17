"""Round-2 free-analysis recount for #1415 (interp-critique v1 findings).

The realized-shift and target-shift cosines of the round-1 H1 statistic share
the SAME 10-draw V_a(c) baseline mean; the shared -V_a(c) sampling-noise term
appears on both sides of the cosine while the random-direction null carries no
shared component — inflating the DV. This script recomputes, from the persisted
per-draw V_a tensors (data/issue_1415/phase1/activations{,_steered}/):

 (A) H1 disjoint-baseline recount of the REGISTERED statistic (matched-layer
     cosine at alpha=4, max over steer layers {7,10,14,17,20,21,24}), both
     extraction arms. Split A: target = V_a(c') - mean(c even draws), shift =
     V_a(steered) - mean(c odd draws); split B swapped; primary = mean(A, B).
     NOTE: 5-draw halves double the baseline-mean noise variance, so the
     disjoint estimate is attenuated — truth lies between disjoint and shared.
 (B) Selection-matched Delta-vs-r_B comparison at steer-L20 (max over READ
     layers), shared + disjoint, + injection norms ||a*Delta(L20)|| vs
     ||a*r_B[20]|| (alpha = 4).
 (C) frac_of_anchor recount (shared + disjoint) at the H1-selected layer.
 (D) Realized-shift split-half reliability at read-L20 (sampling-attenuation
     bound; H2 failure mode (d)).
 (E) Intrusion-excluded shared-baseline recount (CJK-intruded draws excluded
     from steered, baseline-c and c' means) — round-1 free-analysis item 1.
 (F) Behavioral checks: per-pair alpha=4 context concentration (m685_05_formal),
     Wilcoxon with/without it; alpha=4 intrusion-excluded graded mean; K1
     judge-half dead-ceiling pairs.

Validation gates (fail loud): shared-baseline recompute must match
geometric_projections.json pair-by-pair (atol 1e-4) before any disjoint number
is trusted; v_a_mean must equal v_a_per_completion.mean(0) on every loaded blob.

Output: eval_results/issue_1415/disjoint_baseline_recount.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "issue_1415" / "phase1"
EVAL = ROOT / "eval_results" / "issue_1415"
LAYERS = [7, 10, 14, 17, 20, 21, 24]
PRIMARY = 20
ALPHA = 4.0
ARMS = ["prefix", "context"]
TRAITS = ["evil", "hallucination", "sycophancy"]
CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")
EVEN = [0, 2, 4, 6, 8]
ODD = [1, 3, 5, 7, 9]


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def load_blob(path: Path) -> dict:
    assert path.exists(), path
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    gp = json.load(open(EVAL / "geometric_projections.json"))
    nb = json.load(open(EVAL / "null_bands.json"))
    bj = json.load(open(EVAL / "behavioral_judge_scores.json"))
    per_pair_band = nb["bands"]["random_delta"]["per_pair"]
    pooled_p975 = nb["bands"]["random_delta"]["pooled_across_pairs"]["prefix"]["p97.5"]
    k1_band = 0.043084144592285156  # k1_report null_band_p975 (max-over-layers, 500 draws)

    pairs = sorted(p.stem for p in (DATA / "activations").glob("*.pt"))
    assert len(pairs) == 28, pairs

    # per_cell lookup for validation of L20 max-over-read (shared)
    per_cell = {c["cell_id"]: c for c in gp["per_cell"]}

    base = {}
    for pid in pairs:
        b = load_blob(DATA / "activations" / f"{pid}.pt")
        assert b["layers"] == LAYERS, (pid, b["layers"])
        for side in ("c", "cprime"):
            pc = b[side]["v_a_per_completion"].float()
            vm = b[side]["v_a_mean"].float()
            assert b[side]["n_empty_completions"] == 0, (pid, side)
            assert torch.allclose(pc.mean(0), vm, atol=1e-4), (pid, side, "mean mismatch")
        base[pid] = b

    out: dict = {
        "statistic": (
            "H1 matched-layer projection cosine at alpha=4, max over steer layers; "
            "disjoint-baseline: target and realized shift subtract disjoint halves "
            "(5 draws each) of the baseline V_a(c) draws; primary = mean of the two "
            "half-assignments (split A: target|even, shift|odd; split B swapped)"
        ),
        "attenuation_note": (
            "5-draw halves double the baseline-mean noise variance vs the 10-draw mean; "
            "the disjoint estistic is attenuated — truth lies between disjoint and shared"
        ),
        "h1": {},
        "h3": {},
        "l20_selection_matched": {},
        "rb": {},
        "injection_norms": {},
        "frac_of_anchor": {},
        "realized_shift_reliability_L20": {},
        "intrusion_excluded_shared": {},
        "behavioral": {},
    }

    # ---------------- (A) H1 registered-statistic recount -----------------
    per_layer_acc: dict = {
        a: {s: {L: [] for L in LAYERS} for s in ("shared", "disj")} for a in ARMS
    }
    h1_rows: dict = {a: {} for a in ARMS}
    l20_rows: dict = {a: {} for a in ARMS}
    frac_rows: dict = {a: {"shared": [], "disj": []} for a in ARMS}
    rel_rows: dict = {a: [] for a in ARMS}
    intr_rows: dict = {a: {} for a in ARMS}

    # intrusion flags for baseline draws
    intruded = {}
    for pid in pairs:
        for side in ("c", "cprime"):
            d = json.load(open(DATA / "raw_completions" / "gen1b" / pid / f"{side}.json"))
            intruded[(pid, side)] = [bool(CJK.search(t)) for t in d["draws"]]

    for arm in ARMS:
        for pid in pairs:
            b = base[pid]
            c_pc = b["c"]["v_a_per_completion"].float()  # (10, 7, H)
            c_mean = b["c"]["v_a_mean"].float()
            cpr_mean = b["cprime"]["v_a_mean"].float()
            c_even = c_pc[EVEN].mean(0)
            c_odd = c_pc[ODD].mean(0)
            tgt_shared = cpr_mean - c_mean
            tgt_A = cpr_mean - c_even  # split A target
            tgt_B = cpr_mean - c_odd

            # intrusion-excluded baseline means (shared convention)
            keep_c = [i for i in range(10) if not intruded[(pid, "c")][i]]
            keep_cp = [i for i in range(10) if not intruded[(pid, "cprime")][i]]
            c_mean_x = c_pc[keep_c].mean(0) if keep_c else None
            cpr_pc = b["cprime"]["v_a_per_completion"].float()
            cpr_mean_x = cpr_pc[keep_cp].mean(0) if keep_cp else None
            tgt_x = (cpr_mean_x - c_mean_x) if (keep_c and keep_cp) else None

            ml_shared, ml_A, ml_B, ml_x = [], [], [], []
            fr_shared_sel, fr_A_sel, fr_B_sel = [], [], []
            for L in LAYERS:
                sp = DATA / "activations_steered" / "gen1c" / arm / pid / f"L{L}" / "a4.pt"
                sb = load_blob(sp)
                assert sb["alpha"] == ALPHA and sb["layer"] == L
                s_mean = sb["v_a_mean"].float()
                li = LAYERS.index(L)
                sh_shared = (s_mean - c_mean)[li]
                sh_A = (s_mean - c_odd)[li]  # split A shift
                sh_B = (s_mean - c_even)[li]
                ml_shared.append(cos(sh_shared, tgt_shared[li]))
                ml_A.append(cos(sh_A, tgt_A[li]))
                ml_B.append(cos(sh_B, tgt_B[li]))
                fr_shared_sel.append(
                    float(torch.dot(sh_shared, tgt_shared[li]) / tgt_shared[li].norm() ** 2)
                )
                fr_A_sel.append(float(torch.dot(sh_A, tgt_A[li]) / tgt_A[li].norm() ** 2))
                fr_B_sel.append(float(torch.dot(sh_B, tgt_B[li]) / tgt_B[li].norm() ** 2))
                per_layer_acc[arm]["shared"][L].append(ml_shared[-1])
                per_layer_acc[arm]["disj"][L].append(0.5 * (ml_A[-1] + ml_B[-1]))

                # intrusion-excluded (shared convention): exclude intruded steered draws
                if tgt_x is not None:
                    rc = json.load(
                        open(DATA / "raw_completions" / "gen1c" / arm / pid / f"L{L}" / "a4.json")
                    )
                    s_pc = sb["v_a_per_completion"].float()
                    keep_s = [i for i in range(10) if not CJK.search(rc["draws"][i])]
                    if keep_s:
                        s_mean_x = s_pc[keep_s].mean(0)
                        ml_x.append(cos((s_mean_x - c_mean_x)[li], tgt_x[li]))

                if L == PRIMARY:
                    # (B) max over READ layers at steer L20
                    row = per_cell[f"gen1c/{arm}/{pid}/L{PRIMARY}/a4"]
                    reads_shared, reads_A, reads_B = [], [], []
                    for rj, RL in enumerate(LAYERS):
                        cs = cos((s_mean - c_mean)[rj], tgt_shared[rj])
                        jref = row["per_read_layer"][str(RL)]["projection_cosine"]
                        assert abs(cs - jref) < 1e-4, (pid, arm, RL, cs, jref)
                        reads_shared.append(cs)
                        reads_A.append(cos((s_mean - c_odd)[rj], tgt_A[rj]))
                        reads_B.append(cos((s_mean - c_even)[rj], tgt_B[rj]))
                    l20_rows[arm][pid] = {
                        "shared": max(reads_shared),
                        "splitA": max(reads_A),
                        "splitB": max(reads_B),
                        "disj": 0.5 * (max(reads_A) + max(reads_B)),
                    }
                    # (D) realized-shift split-half reliability at read L20
                    s_pc = sb["v_a_per_completion"].float()
                    li20 = LAYERS.index(PRIMARY)
                    sh1 = s_pc[EVEN].mean(0)[li20] - c_even[li20]
                    sh2 = s_pc[ODD].mean(0)[li20] - c_odd[li20]
                    rel_rows[arm].append(cos(sh1, sh2))

            band = per_pair_band[arm][pid]["p97.5"]
            sel = int(np.argmax(ml_shared))  # H1-selected layer (shared convention)
            h1_rows[arm][pid] = {
                "shared_max": max(ml_shared),
                "splitA_max": max(ml_A),
                "splitB_max": max(ml_B),
                "disj_max": 0.5 * (max(ml_A) + max(ml_B)),
                "band_p975": band,
                "shared_above": max(ml_shared) > band,
                "disj_above": 0.5 * (max(ml_A) + max(ml_B)) > band,
                "splitA_above": max(ml_A) > band,
                "selected_layer": LAYERS[sel],
            }
            frac_rows[arm]["shared"].append(fr_shared_sel[sel])
            frac_rows[arm]["disj"].append(0.5 * (fr_A_sel[sel] + fr_B_sel[sel]))
            if ml_x:
                intr_rows[arm][pid] = max(ml_x)

            # validate shared vs geometric_projections.json
            jmax = gp["h1"][arm][pid]["max_over_layers"]
            assert abs(max(ml_shared) - jmax) < 1e-4, (pid, arm, max(ml_shared), jmax)

    for arm in ARMS:
        rows = h1_rows[arm]
        for key in ("shared_max", "splitA_max", "splitB_max", "disj_max"):
            vals = np.array([r[key] for r in rows.values()])
            out["h1"].setdefault(arm, {})[key] = {
                "mean": float(vals.mean()),
                "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
                "median": float(np.median(vals)),
                "n_above_own_band": int(
                    sum(
                        r[key.replace("_max", "_above")] if key != "splitB_max" else 0
                        for r in rows.values()
                    )
                )
                if key in ("shared_max", "splitA_max", "disj_max")
                else None,
            }
        out["h1"][arm]["n_pairs"] = len(rows)
        out["h1"][arm]["pairs_below_band_disj"] = sorted(
            p for p, r in rows.items() if not r["disj_above"]
        )
        out["h1"][arm]["per_pair"] = rows
        out["h1"][arm]["per_layer_mean"] = {
            s: {str(L): float(np.mean(per_layer_acc[arm][s][L])) for L in LAYERS}
            for s in ("shared", "disj")
        }
        out["h1"][arm]["aggregate_above_pooled_band"] = {
            s: bool(
                np.mean([r[f"{s}_max" if s != "disj" else "disj_max"] for r in rows.values()])
                > pooled_p975
            )
            for s in ("shared", "disj")
        }
        out["frac_of_anchor"][arm] = {
            "shared_mean": float(np.mean(frac_rows[arm]["shared"])),
            "shared_median": float(np.median(frac_rows[arm]["shared"])),
            "disj_mean": float(np.mean(frac_rows[arm]["disj"])),
            "disj_median": float(np.median(frac_rows[arm]["disj"])),
            "convention": "frac at the shared-selected (argmax) steer layer per pair",
        }
        out["realized_shift_reliability_L20"][arm] = {
            "mean": float(np.mean(rel_rows[arm])),
            "median": float(np.median(rel_rows[arm])),
            "min": float(np.min(rel_rows[arm])),
            "max": float(np.max(rel_rows[arm])),
        }
        xs = np.array(list(intr_rows[arm].values()))
        sh = np.array([h1_rows[arm][p]["shared_max"] for p in intr_rows[arm]])
        out["intrusion_excluded_shared"][arm] = {
            "n_pairs": len(xs),
            "mean_excluded": float(xs.mean()),
            "mean_shared_same_pairs": float(sh.mean()),
            "n_above_own_band": int(
                sum(v > h1_rows[arm][p]["band_p975"] for p, v in intr_rows[arm].items())
            ),
        }

        # H3 on disjoint stat
        mt = [r["disj_max"] for p, r in rows.items() if gp["h1"][arm][p]["pair_type"] == "matched"]
        cr = [r["disj_max"] for p, r in rows.items() if gp["h1"][arm][p]["pair_type"] == "cross"]
        t = stats.ttest_ind(mt, cr, equal_var=False, alternative="greater")
        rs = stats.ranksums(mt, cr, alternative="greater")
        out["h3"][arm] = {
            "matched_mean_disj": float(np.mean(mt)),
            "cross_mean_disj": float(np.mean(cr)),
            "welch_one_sided_p": float(t.pvalue),
            "ranksum_one_sided_p": float(rs.pvalue),
            "n_matched": len(mt),
            "n_cross": len(cr),
        }

    # canonical-cell (steer-L20 max-over-read) summary — the critique's recount
    for arm in ARMS:
        rows = l20_rows[arm]
        out["l20_selection_matched"][arm] = {
            k: {
                "mean": float(np.mean([r[k] for r in rows.values()])),
                "median": float(np.median([r[k] for r in rows.values()])),
            }
            for k in ("shared", "splitA", "splitB", "disj")
        }
        out["l20_selection_matched"][arm]["n_below_k1_band_splitA"] = int(
            sum(r["splitA"] <= k1_band for r in rows.values())
        )
        out["l20_selection_matched"][arm]["n_below_k1_band_disj"] = int(
            sum(r["disj"] <= k1_band for r in rows.values())
        )
        out["l20_selection_matched"][arm]["medical_doctor"] = l20_rows[arm].get(
            "m685_07_medical_doctor"
        )

    # ---------------- (B) r_B matched selection + disjoint -----------------
    for trait in TRAITS:
        shared_l, disj_l = [], []
        for pid in pairs:
            sp = DATA / "activations_steered" / "gen1d_full" / trait / pid / "a4.pt"
            sb = load_blob(sp)
            s_mean = sb["v_a_mean"].float()
            b = base[pid]
            c_pc = b["c"]["v_a_per_completion"].float()
            c_mean = b["c"]["v_a_mean"].float()
            cpr_mean = b["cprime"]["v_a_mean"].float()
            c_even, c_odd = c_pc[EVEN].mean(0), c_pc[ODD].mean(0)
            reads_shared = [
                cos((s_mean - c_mean)[j], (cpr_mean - c_mean)[j]) for j in range(len(LAYERS))
            ]
            reads_A = [cos((s_mean - c_odd)[j], (cpr_mean - c_even)[j]) for j in range(len(LAYERS))]
            reads_B = [cos((s_mean - c_even)[j], (cpr_mean - c_odd)[j]) for j in range(len(LAYERS))]
            # validate vs per_cell
            row = per_cell[f"gen1d_full/{trait}/{pid}/a4"]
            for j, RL in enumerate(LAYERS):
                jref = row["per_read_layer"][str(RL)]["projection_cosine"]
                assert abs(reads_shared[j] - jref) < 1e-4, (trait, pid, RL)
            shared_l.append(max(reads_shared))
            disj_l.append(0.5 * (max(reads_A) + max(reads_B)))
        out["rb"][trait] = {
            "shared_mean_max_over_read_L20steer": float(np.mean(shared_l)),
            "disj_mean_max_over_read_L20steer": float(np.mean(disj_l)),
        }

    # ---------------- injection norms -----------------
    dn = {a: [] for a in ARMS}
    li20 = LAYERS.index(PRIMARY)
    for pid in pairs:
        b = base[pid]
        for arm in ARMS:
            d = (b["cprime"][f"v_c_{arm}"][li20] - b["c"][f"v_c_{arm}"][li20]).float()
            dn[arm].append(float(d.norm()))
    out["injection_norms"]["delta_L20_norm"] = {
        a: {"mean": float(np.mean(dn[a])), "median": float(np.median(dn[a])), "alpha": ALPHA}
        for a in ARMS
    }
    try:
        from huggingface_hub import hf_hub_download

        for trait in TRAITS:
            p = hf_hub_download(
                "superkaiba1/explore-persona-space-data",
                f"issue779_monitoring/r_b/{trait}.pt",
                repo_type="dataset",
            )
            rb = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(rb, dict):
                for key in ("r_b", "rb", "direction", "vector"):
                    if isinstance(rb.get(key), torch.Tensor):
                        rb = rb[key]
                        break
            out["injection_norms"][f"rb_{trait}_L20_norm"] = float(rb.float()[PRIMARY].norm())
    except Exception as e:
        out["injection_norms"]["rb_error"] = repr(e)

    # ---------------- (F) behavioral checks -----------------
    pi = bj["per_item"]
    shifts = {}
    for pid in pairs:
        st = [
            v["graded_score"] for k, v in pi.items() if k.startswith(f"gen1c/context/{pid}/L20/a4/")
        ]
        ba = [v["graded_score"] for k, v in pi.items() if k.startswith(f"gen1b/{pid}/c/")]
        assert len(st) == 10 and len(ba) == 10, (pid, len(st), len(ba))
        shifts[pid] = float(np.mean(st) - np.mean(ba))
    sv = np.array([shifts[p] for p in pairs])
    w_all = stats.wilcoxon(sv, alternative="greater")
    no_formal = np.array([shifts[p] for p in pairs if p != "m685_05_formal"])
    w_nf = stats.wilcoxon(no_formal, alternative="greater")
    out["behavioral"]["context_a4_per_pair_shift"] = shifts
    out["behavioral"]["formal_share_of_summed_shift"] = float(shifts["m685_05_formal"] / sv.sum())
    out["behavioral"]["wilcoxon_all_p"] = float(w_all.pvalue)
    out["behavioral"]["mean_shift_all"] = float(sv.mean())
    out["behavioral"]["wilcoxon_excl_formal_p"] = float(w_nf.pvalue)
    out["behavioral"]["mean_shift_excl_formal"] = float(no_formal.mean())

    # alpha=4 context intrusion-excluded graded mean
    kept, kept_x = [], []
    for pid in pairs:
        rc = json.load(
            open(DATA / "raw_completions" / "gen1c" / "context" / pid / "L20" / "a4.json")
        )
        for i in range(10):
            k = f"gen1c/context/{pid}/L20/a4/d{i}"
            s = pi[k]["graded_score"]
            kept.append(s)
            if not CJK.search(rc["draws"][i]):
                kept_x.append(s)
    out["behavioral"]["context_a4_mean"] = float(np.mean(kept))
    out["behavioral"]["context_a4_mean_intrusion_excluded"] = float(np.mean(kept_x))
    out["behavioral"]["context_a4_n"] = len(kept)
    out["behavioral"]["context_a4_n_intrusion_excluded"] = len(kept_x)

    # K1 judge-half dead-ceiling pairs
    k1j = bj["k1_judge_check"]
    per = k1j.get("ceiling_minus_baseline_shift_per_pair") or k1j.get("per_pair")
    dead = {p: v for p, v in per.items() if v < 5.0}
    out["behavioral"]["k1_judge_dead_ceiling_pairs"] = dead

    outp = EVAL / "disjoint_baseline_recount.json"
    outp.write_text(json.dumps(out, indent=1))
    print("WROTE", outp)

    # compact summary
    for arm in ARMS:
        h = out["h1"][arm]
        print(
            f"[H1 {arm}] shared {h['shared_max']['mean']:.3f} -> disj {h['disj_max']['mean']:.3f} "
            f"(A {h['splitA_max']['mean']:.3f} B {h['splitB_max']['mean']:.3f}); "
            f"above-band shared {h['shared_max']['n_above_own_band']}/28, "
            f"disj {h['disj_max']['n_above_own_band']}/28; below(disj): {h['pairs_below_band_disj']}"
        )
        l = out["l20_selection_matched"][arm]
        print(
            f"[L20 {arm}] shared {l['shared']['mean']:.3f} splitA {l['splitA']['mean']:.3f} "
            f"disj {l['disj']['mean']:.3f}; n_below_k1band splitA {l['n_below_k1_band_splitA']} "
            f"disj {l['n_below_k1_band_disj']}; medical {l['medical_doctor']}"
        )
        print(f"[H3 {arm}]", out["h3"][arm])
        print(f"[frac {arm}]", out["frac_of_anchor"][arm])
        print(f"[rel {arm}]", out["realized_shift_reliability_L20"][arm])
        print(f"[intr-x {arm}]", out["intrusion_excluded_shared"][arm])
    print("[rb]", json.dumps(out["rb"], indent=1))
    print("[norms]", json.dumps(out["injection_norms"], indent=1))
    print("[behavioral]", json.dumps(out["behavioral"], indent=1)[:1200])


if __name__ == "__main__":
    sys.exit(main())
