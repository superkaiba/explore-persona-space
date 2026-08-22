#!/usr/bin/env python
"""Issue #2333 Step-3.7 language-intrusion recount (analyzer duty).

Qwen-family models under a non-CJK eval owe a per-arm CJK intrusion scan of
every judged pool plus recounts of the registered reads with intruded draws
(a) EXCLUDED and (b) their judge contrast ZEROED (delta := 0, a
no-behavioral-movement convention). Reuses the shipped analysis module's own
helpers (loaders, Wilcoxon, fixed-m Holm, batched pair-clustered bootstrap,
lattice) so the recount differs from the shipped stats ONLY in the intrusion
handling.

Recount boundary (stated, not silent): banked denominators judged in PARENT
waves (#2162 S1 ce + anchors; #2094 fu1 S2 ce) are held fixed — this run
cannot re-score parent waves; the same-wave denominators (q25 calib re-judge,
q35 fresh ce control) and the q35 fresh anchors ARE recounted. The zeroed
variant zeroes arm/ce pool deltas and keeps anchors at the excluded recount
(zeroing a normalization anchor has no analogous "no movement" semantics).

Writes eval_results/issue_2333/f_metrics/{tag}/cjk_recount.json.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy import

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "issue2333_analysis", REPO / "scripts" / "issue2333_analysis.py"
)
A = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(A)

J33 = A.J33
J62 = A.J62
A62 = A.A62
C = A.C

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

JUDGE_INPUTS = REPO / "data/issue_2333/judge_inputs"


def _intruded(text: str | None) -> bool:
    return bool(CJK_RE.search(text or ""))


def _f_from_rows_variant(
    rows: list[dict],
    tag: str,
    scores: dict,
    anchor: dict,
    mode: str,  # "shipped" | "excluded" | "zeroed"
    text_field: str,
) -> dict:
    deltas: list[float] = []
    n_intruded = n_intruded_coherent = 0
    for row in rows:
        intr = _intruded(row.get(text_field))
        if intr:
            n_intruded += 1
        coh = scores.get(A._item_id("c", J33.coherence_key(tag if tag != "n" else "g", row)))
        if coh is None or coh <= A.COHERENCE_THRESHOLD:
            continue
        if intr:
            n_intruded_coherent += 1
            if mode == "excluded":
                continue
        sa = scores.get(A._item_id(tag, J33.behavior_key(tag, row, "a")))
        sb = scores.get(A._item_id(tag, J33.behavior_key(tag, row, "b")))
        if sa is None or sb is None:
            continue
        deltas.append(0.0 if (intr and mode == "zeroed") else (sb - sa) / 100.0)
    dp = A._mean(deltas)
    fl, ce_ = anchor.get("floor"), anchor.get("ceiling")
    f_beh = None
    if dp is not None and fl is not None and ce_ is not None and abs(ce_ - fl) > 1e-9:
        f_beh = (dp - fl) / (ce_ - fl)
    return {
        "f_beh": f_beh,
        "n_rows": len(rows),
        "n_intruded": n_intruded,
        "n_intruded_coherent": n_intruded_coherent,
    }


def recount_tag(tag: str) -> dict:
    s1_pairs, s2_pairs = J33.build_pair_universe()
    pairs = [*s1_pairs, *s2_pairs]
    pairs_by_id = {p.pair_id: p for p in pairs}
    cells_of = {p.pair_id: p.cell for p in s1_pairs}
    sets = {"s1": [p.pair_id for p in s1_pairs], "s2": [p.pair_id for p in s2_pairs]}

    rollouts_dir = JUDGE_INPUTS / "issue2333_snowball" / tag / "rollouts"
    calib_dir = JUDGE_INPUTS / f"calib_{tag}"
    scores_dir = REPO / f"eval_results/issue_2333/judge_{tag}/scores"
    suffixes = ("grid", "anchors") if tag == "q35" else ("grid",)
    scores = A._load_scores(scores_dir, suffixes)
    grid_rows = J33.load_grid_rows(rollouts_dir)

    # Intrusion tallies per (arm, variant) pool + fired-overlap.
    tallies: dict[str, dict] = defaultdict(lambda: {"n": 0, "intr": 0, "intr_coh": 0})
    for r in grid_rows:
        key = f"{r['arm_slug']}::{r['variant']}"
        t = tallies[key]
        t["n"] += 1
        if _intruded(r.get("response_text")):
            t["intr"] += 1
            coh = scores.get(A._item_id("c", J33.coherence_key("g", r)))
            if coh is not None and coh > A.COHERENCE_THRESHOLD:
                t["intr_coh"] += 1

    # Anchors: q25 banked (parent-wave, held fixed); q35 fresh, recounted
    # with intruded anchor rows excluded.
    anchor_intrusion = None
    if tag == "q35":
        anchor_rows = J33.load_anchor_rows(JUDGE_INPUTS / "issue2333_snowball/q35/anchors")
        n_intr = sum(_intruded(r.get("response_text")) for r in anchor_rows)
        anchor_intrusion = {"n": len(anchor_rows), "intruded": n_intr}
        anchor_rows_x = [r for r in anchor_rows if not _intruded(r.get("response_text"))]
        anchors_shipped = A._fresh_anchor_deltas(pairs, anchor_rows, scores)
        anchors_x = A._fresh_anchor_deltas(pairs, anchor_rows_x, scores)
    else:
        anchors_shipped = A._banked_anchor_deltas(pairs)
        anchors_x = anchors_shipped

    out: dict = {
        "model_tag": tag,
        "regex": CJK_RE.pattern,
        "grid_intrusion": {
            "total": sum(t["n"] for t in tallies.values()),
            "intruded": sum(t["intr"] for t in tallies.values()),
            "intruded_coherent": sum(t["intr_coh"] for t in tallies.values()),
            "per_pool": dict(sorted(tallies.items())),
        },
        "anchor_intrusion": anchor_intrusion,
        "recounts": {},
    }

    # ce denominator pools (same-wave) + banked (held fixed).
    ce_banked: dict[str, float] = {
        r["pair_id"]: r["f_beh"]
        for r in A62._iter_jsonl(A.A2162_F_CELLS)
        if r["slot"] == "ce"
        and r["cell"] in C.S1_CELLS
        and r["arm"] == "steered"
        and r["f_beh"] is not None
    }
    for r in A62._iter_jsonl(A.INPUTS_DIR / "s2_ce_control_perpair.jsonl"):
        if r["arm"] == "steered" and r["f_beh"] is not None:
            ce_banked[r["pair_id"]] = r["f_beh"]

    if tag == "q35":
        ce_rows_all = J33.load_ce_rows(rollouts_dir)
        ce_by = defaultdict(list)
        for r in ce_rows_all:
            if r["variant"] == "steered":
                ce_by[r["pair_id"]].append(r)
        ce_intr = sum(_intruded(r.get("response_text")) for r in ce_rows_all)
        out["ce_pool_intrusion"] = {"n": len(ce_rows_all), "intruded": ce_intr}
    else:
        calib_s1 = J33.load_calib_s1(calib_dir)
        calib_s2 = J33.load_calib_s2(calib_dir)
        calib_by: dict[str, list] = defaultdict(list)
        calib_tag: dict[str, str] = {}
        for r in calib_s1:
            if r["arm"] == "steered":
                calib_by[r["pair_id"]].append(r)
                calib_tag[r["pair_id"]] = "k"
        for r in calib_s2:
            if r["arm"] == "steered":
                calib_by[r["pair_id"]].append(r)
                calib_tag[r["pair_id"]] = "m"
        n_cal = sum(len(v) for v in calib_by.values())
        cal_intr = sum(_intruded(r.get("text")) for v in calib_by.values() for r in v)
        out["calib_pool_intrusion"] = {"n_steered_rows": n_cal, "intruded": cal_intr}

    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in grid_rows:
        by_cell[(r["pair_id"], r["arm_slug"], r["variant"])].append(r)

    for mode in ("excluded", "zeroed"):
        anchors = anchors_x  # zeroed variant keeps excluded-anchor normalization (q35)
        f_st: dict[tuple[str, str], float | None] = {}
        f_nu: dict[tuple[str, str], float | None] = {}
        for (pid, slug, variant), rows in by_cell.items():
            rec = _f_from_rows_variant(rows, "g", scores, anchors[pid], mode, "response_text")
            (f_st if variant == "steered" else f_nu)[(pid, slug)] = rec["f_beh"]

        ce_samewave: dict[str, float] = {}
        if tag == "q35":
            for pid, rows in ce_by.items():
                rec = _f_from_rows_variant(rows, "e", scores, anchors[pid], mode, "response_text")
                if rec["f_beh"] is not None:
                    ce_samewave[pid] = rec["f_beh"]
        else:
            for pid, rows in calib_by.items():
                rec = _f_from_rows_variant(rows, calib_tag[pid], scores, anchors[pid], mode, "text")
                if rec["f_beh"] is not None:
                    ce_samewave[pid] = rec["f_beh"]

        sep = {pid: anchors[pid].get("separation") for pid in pairs_by_id}
        per_set: dict = {}
        for set_name, pair_ids in sets.items():
            survivors_all = A._survivors(sep, pair_ids)
            if set_name == "s1":
                floor = A.S1_SURVIVAL_FLOOR
                by_cell_surv: dict[str, list[str]] = defaultdict(list)
                for pid in survivors_all:
                    by_cell_surv[cells_of[pid]].append(pid)
                passing = {c: v for c, v in by_cell_surv.items() if len(v) >= floor}
                survivors = sorted(pid for v in passing.values() for pid in v)
                testable = bool(passing)
            else:
                floor = A.S2_SURVIVAL_FLOOR
                survivors = sorted(survivors_all)
                testable = len(survivors) >= floor

            arms_out: dict[str, dict] = {}
            pvals: dict[str, float] = {}
            for slug in C.ARM_SLUGS:
                pids = [
                    p
                    for p in survivors
                    if f_st.get((p, slug)) is not None and f_nu.get((p, slug)) is not None
                ]
                rec = {"n_pairs": len(pids), "below_floor": (not testable) or len(pids) < floor}
                if not rec["below_floor"]:
                    d = np.array([f_st[(p, slug)] - f_nu[(p, slug)] for p in pids])
                    cols = np.full((len(pids), 6), np.nan)
                    for i, p in enumerate(pids):
                        fs = f_st[(p, slug)]
                        cols[i, 0] = d[i]
                        cols[i, 1] = fs
                        if p in ce_samewave:
                            cols[i, 2] = ce_samewave[p]
                            cols[i, 3] = fs - A.D3_SHARE * ce_samewave[p]
                        if p in ce_banked:
                            cols[i, 4] = ce_banked[p]
                            cols[i, 5] = fs - A.D3_SHARE * ce_banked[p]
                    draws = A.bootstrap_family_means_batched(cols, A.BOOT_B, A.BOOT_SEED)
                    rec["diff_mean"] = float(np.mean(d))
                    rec["diff_ci"] = A._ci(draws[:, 0])
                    rec["f_steered_mean"] = float(np.nanmean(cols[:, 1]))
                    rec["p_wilcoxon"] = A62._wilcoxon_exact_p(d)
                    pvals[slug] = rec["p_wilcoxon"]
                    for label, ce_col, d3_col in (("samewave", 2, 3), ("banked", 4, 5)):
                        ce_mean = float(np.nanmean(cols[:, ce_col]))
                        if math.isnan(ce_mean) or abs(ce_mean) < 1e-9:
                            continue
                        rec[f"recovery_{label}"] = {
                            "ce_mean": ce_mean,
                            "ratio": float(np.nanmean(cols[:, 1])) / ce_mean,
                            "d3_ci": A._ci(draws[:, d3_col]),
                        }
                arms_out[slug] = rec
            holmed = A.holm_fixed_m(pvals) if pvals else {}
            for slug, rec in arms_out.items():
                if slug in holmed:
                    rec["p_holm"] = holmed[slug]
                    rec["holm_significant"] = holmed[slug] < A.HOLM_ALPHA
                    if "diff_ci" in rec:
                        rec["separates"] = (rec["diff_ci"][0] > 0) and rec["holm_significant"]

            verdicts = {}
            for scheme in C.ARM_SCHEMES:
                slug = f"prefill3_{scheme}"
                rec = arms_out.get(slug, {})
                if rec.get("below_floor") or "diff_ci" not in rec:
                    verdicts[scheme] = {"label": A.instance_label(scheme, "untestable-small-n")}
                    continue
                v = {
                    "label": A.instance_label(
                        scheme,
                        A.lattice_label(
                            rec["diff_ci"][0],
                            rec["diff_ci"][1],
                            bool(rec.get("holm_significant")),
                            rec.get("recovery_samewave", {}).get("d3_ci", (None, None))[0],
                            rec.get("recovery_samewave", {}).get("d3_ci", (None, None))[1],
                        ),
                    )
                }
                if "recovery_banked" in rec:
                    v["label_banked_ce"] = A.instance_label(
                        scheme,
                        A.lattice_label(
                            rec["diff_ci"][0],
                            rec["diff_ci"][1],
                            bool(rec.get("holm_significant")),
                            rec["recovery_banked"]["d3_ci"][0],
                            rec["recovery_banked"]["d3_ci"][1],
                        ),
                    )
                verdicts[scheme] = v
            per_set[set_name] = {
                "n_survivors_tested": len(survivors),
                "arms": arms_out,
                "prefill3_verdicts": verdicts,
            }
        out["recounts"][mode] = per_set
    return out


def main() -> int:
    for tag in ("q25", "q35"):
        res = recount_tag(tag)
        dest = REPO / f"eval_results/issue_2333/f_metrics/{tag}/cjk_recount.json"
        A62._write_json_atomic(dest, res)
        g = res["grid_intrusion"]
        print(
            f"[{tag}] grid intrusion {g['intruded']}/{g['total']} "
            f"({g['intruded'] / g['total']:.4f}); coherent-and-intruded {g['intruded_coherent']}"
        )
        for mode in ("excluded", "zeroed"):
            for s in ("s1", "s2"):
                v = res["recounts"][mode][s]["prefill3_verdicts"]
                print(f"  {mode:8s} {s}: med={v['med']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
