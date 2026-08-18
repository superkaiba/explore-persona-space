#!/usr/bin/env python
"""Issue #2333 Step 9a-ter free-analysis follow-up (0 GPU; committed tables only).

Deliverable 1 — ``percell_prefill3.json`` (per model): DESCRIPTIVE per-cell
breakdown of the confirmatory arm (``prefill3_med`` = three-token prefill,
patch-content donors) on the instruction-format set (S1, 5 cells). Per cell:
n surviving pairs, recovery ratio (cell F_beh steered mean / cell same-wave ce
control mean) with pair-clustered bootstrap CI, paired steered-null diff CI
(B=10,000, seed 23330), and the exact two-sided Wilcoxon signed-rank p RAW.
Holm was registered at the (model x pair-set) family of m=12 ARMS — there is
no registered per-cell family, so per-cell reads are DESCRIPTIVE/exploratory
and no new corrected family is invented (flagged in every record).

Deliverable 2 — ``continuation_lattice.json`` (per model): the registered
prefill-arm verdict machinery re-run with F_beh replaced by the
continuation-only companion (``f_beh_continuation`` — the judged continuation
WITHOUT the prefilled donor opening) across the 6 prefill arms x steered/null
on S1: per-arm separation conjunction (paired-diff CI strictly > 0 AND
Holm-fixed-m significant) + the four-branch lattice verdict on prefill3 (med
confirmatory; bstart labels prefixed ``natural-opening``), side by side with
the registered whole-response labels from ``stats.json``, with flip flags.

Registered thresholds unchanged: S1 per-cell survival floor 12, separation bar
0.5, Holm alpha 0.05 at FIXED m=12, D3 share 0.5, B=10,000, seed 23330. Patch
arms have no continuation companion (nothing is prefilled there), so the
continuation Holm family realizes 6 of the m=12 registered arm slots — the
fixed-m correction keeps the registered family size (conservative), matching
``holm_fixed_m``'s dropped-arm convention. The ce denominators (same-wave
primary + banked #2162 comparison) are whole-response by construction (no
prefill in the ce control arm) and are held fixed.

REUSES ``issue2333_analysis`` {_ci, _survivors, holm_fixed_m, lattice_label,
instance_label, constants, BOOT_B/BOOT_SEED/D3_SHARE/HOLM_FIXED_M},
``issue2162_analysis`` {_iter_jsonl, _wilcoxon_exact_p, _write_json_atomic},
and ``issue2094_analysis.bootstrap_family_means_batched`` (batched draw
matrix — no per-draw Python loop).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy import

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "issue2333_analysis", REPO / "scripts" / "issue2333_analysis.py"
)
A = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(A)

A62 = A.A62
C = A.C
J33 = A.J33

FMETRICS = REPO / "eval_results/issue_2333/f_metrics"
MODEL_TAGS = ("q25", "q35")
CONFIRMATORY_SLUG = "prefill3_med"
PREFILL_SLUGS = tuple(s for s in C.ARM_SLUGS if s.startswith("prefill"))
DESCRIPTIVE_NOTE = (
    "DESCRIPTIVE/exploratory: Holm was registered at the (model x pair-set) family of "
    "m=12 ARMS (plan §6); no per-cell family was registered and none is invented here — "
    "p_wilcoxon is reported RAW and any p<0.05 read at cell grain is post-hoc."
)


def _assert_nonempty(items: list, context: str, counters: dict[str, int]) -> None:
    """Fail-loud empty-selection guard over local committed artifacts (#1739 class)."""
    if not items:
        raise RuntimeError(f"empty selection at {context}; reject counters: {counters}")


def _load_tag(tag: str) -> dict:
    """Load the committed f-tables for one model tag (S1 + S2 rows; keyed maps)."""
    out_dir = FMETRICS / tag
    steered = list(A62._iter_jsonl(out_dir / "f_cells.jsonl"))
    nulls = list(A62._iter_jsonl(out_dir / "null_cells.jsonl"))
    f_st = {(r["pair_id"], r["arm_slug"]): r for r in steered}
    f_nu = {(r["pair_id"], r["arm_slug"]): r for r in nulls}
    sep_by_pair = {r["pair_id"]: r.get("separation") for r in [*steered, *nulls]}

    ce_samewave: dict[str, float] = {}
    if tag == "q35":
        for r in A62._iter_jsonl(out_dir / "ce_cells.jsonl"):
            if r["variant"] == "steered" and r["f_beh"] is not None:
                ce_samewave[r["pair_id"]] = r["f_beh"]
    else:
        for r in A62._iter_jsonl(out_dir / "calib_cells.jsonl"):
            if r["arm"] == "steered" and r["f_beh"] is not None:
                ce_samewave[r["pair_id"]] = r["f_beh"]

    # Banked #2162 S1 ce (parent wave, held fixed) — the registered comparison read.
    ce_banked: dict[str, float] = {
        r["pair_id"]: r["f_beh"]
        for r in A62._iter_jsonl(REPO / A.A2162_F_CELLS)
        if r["slot"] == "ce"
        and r["cell"] in C.S1_CELLS
        and r["arm"] == "steered"
        and r["f_beh"] is not None
    }

    registered = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    return {
        "f_st": f_st,
        "f_nu": f_nu,
        "sep_by_pair": sep_by_pair,
        "ce_samewave": ce_samewave,
        "ce_banked": ce_banked,
        "registered_s1": registered["per_set"]["s1"],
        "row_counts": {
            "f_cells": len(steered),
            "null_cells": len(nulls),
            "ce_samewave_pairs": len(ce_samewave),
            "ce_banked_pairs": len(ce_banked),
        },
    }


def _s1_survivor_state(sep_by_pair: dict) -> tuple[dict, dict, dict, list[str]]:
    """Registered S1 survivor gating: separation bar 0.5, per-cell floor 12."""
    s1_pairs, _ = J33.build_pair_universe()
    cells_of = {p.pair_id: p.cell for p in s1_pairs}
    survivors_all = A._survivors(sep_by_pair, [p.pair_id for p in s1_pairs])
    by_cell_surv: dict[str, list[str]] = defaultdict(list)
    for pid in survivors_all:
        by_cell_surv[cells_of[pid]].append(pid)
    passing = {c: v for c, v in by_cell_surv.items() if len(v) >= A.S1_SURVIVAL_FLOOR}
    survivors = sorted(pid for v in passing.values() for pid in v)
    return cells_of, dict(by_cell_surv), passing, survivors


def _boot_cols(
    pids: list[str],
    val_st: dict[str, float],
    val_nu: dict[str, float],
    ce_samewave: dict[str, float],
    ce_banked: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Joint per-pair bootstrap columns (registered layout): 0 diff, 1 F_steered,
    2 ce_samewave, 3 D3(samewave), 4 ce_banked, 5 D3(banked)."""
    cols = np.full((len(pids), 6), np.nan)
    for i, p in enumerate(pids):
        fs = val_st[p]
        cols[i, 0] = fs - val_nu[p]
        cols[i, 1] = fs
        if p in ce_samewave:
            cols[i, 2] = ce_samewave[p]
            cols[i, 3] = fs - A.D3_SHARE * ce_samewave[p]
        if p in ce_banked:
            cols[i, 4] = ce_banked[p]
            cols[i, 5] = fs - A.D3_SHARE * ce_banked[p]
    draws = A.bootstrap_family_means_batched(cols, A.BOOT_B, A.BOOT_SEED)
    return cols, draws


def _arm_rec(
    pids: list[str],
    val_st: dict[str, float],
    val_nu: dict[str, float],
    ce_samewave: dict[str, float],
    ce_banked: dict[str, float],
) -> dict:
    """Registered per-arm reads on one value map: diff CI, raw Wilcoxon, recovery."""
    cols, draws = _boot_cols(pids, val_st, val_nu, ce_samewave, ce_banked)
    d = cols[:, 0]
    rec: dict = {
        "n_pairs": len(pids),
        "diff_mean": float(np.mean(d)),
        "diff_ci": A._ci(draws[:, 0]),
        "f_steered_mean": float(np.nanmean(cols[:, 1])),
        "p_wilcoxon": A62._wilcoxon_exact_p(d),
    }
    for label, ce_col, d3_col in (("samewave", 2, 3), ("banked", 4, 5)):
        ce_mean = float(np.nanmean(cols[:, ce_col]))
        if math.isnan(ce_mean) or abs(ce_mean) < 1e-9:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            r_draws = draws[:, 1] / draws[:, ce_col]
        rec[f"recovery_{label}"] = {
            "ce_mean": ce_mean,
            "n_pairs_with_ce": int(np.sum(~np.isnan(cols[:, ce_col]))),
            "ratio": float(np.nanmean(cols[:, 1])) / ce_mean,
            "ratio_ci": A._ci(r_draws),
            "d3_mean": float(np.nanmean(cols[:, d3_col])),
            "d3_ci": A._ci(draws[:, d3_col]),
        }
    return rec


def _metadata(tag: str, row_counts: dict) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "script": "scripts/issue2333_followup_cells_continuation.py",
        "model_tag": tag,
        "inputs_row_counts": row_counts,
        "constants": {
            "boot_b": A.BOOT_B,
            "boot_seed": A.BOOT_SEED,
            "separation_bar": A.SEPARATION_BAR,
            "s1_survival_floor_per_cell": A.S1_SURVIVAL_FLOOR,
            "holm_fixed_m": A.HOLM_FIXED_M,
            "holm_alpha": A.HOLM_ALPHA,
            "d3_share": A.D3_SHARE,
        },
        **as_metadata_dict(git_provenance()),
    }


# ── Deliverable 1: per-cell breakdown of the confirmatory arm ─────────


def percell_prefill3(tag: str, loaded: dict) -> dict:
    f_st, f_nu = loaded["f_st"], loaded["f_nu"]
    cells_of, by_cell_surv, passing, _ = _s1_survivor_state(loaded["sep_by_pair"])
    reg_arm = loaded["registered_s1"]["arms"][CONFIRMATORY_SLUG]

    per_cell: dict[str, dict] = {}
    for cell in sorted(C.S1_CELLS):
        t0 = time.time()
        cand = by_cell_surv.get(cell, [])
        pids = [
            p
            for p in sorted(cand)
            if f_st.get((p, CONFIRMATORY_SLUG), {}).get("f_beh") is not None
            and f_nu.get((p, CONFIRMATORY_SLUG), {}).get("f_beh") is not None
        ]
        counters = {
            "n_survivors_in_cell": len(cand),
            "rejected_missing_f_beh": len(cand) - len(pids),
        }
        _assert_nonempty(pids, f"percell {tag}/{cell}/{CONFIRMATORY_SLUG}", counters)
        val_st = {p: f_st[(p, CONFIRMATORY_SLUG)]["f_beh"] for p in pids}
        val_nu = {p: f_nu[(p, CONFIRMATORY_SLUG)]["f_beh"] for p in pids}
        rec = _arm_rec(pids, val_st, val_nu, loaded["ce_samewave"], loaded["ce_banked"])
        lo, hi = rec["diff_ci"]
        rec.update(
            {
                "cell_passes_registered_floor": cell in passing,
                "selection_counters": counters,
                "separation_read_descriptive": {
                    "diff_ci_strictly_positive": bool(lo > 0),
                    "p_wilcoxon_raw_below_0.05": bool(rec["p_wilcoxon"] < 0.05),
                    "both": bool(lo > 0 and rec["p_wilcoxon"] < 0.05),
                    "note": DESCRIPTIVE_NOTE,
                },
            }
        )
        per_cell[cell] = rec
        print(
            f"[percell] {tag} {cell} n={rec['n_pairs']} "
            f"ratio={rec.get('recovery_samewave', {}).get('ratio')} "
            f"diff_ci=({lo:.4f},{hi:.4f}) p_raw={rec['p_wilcoxon']:.3g} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    return {
        "model_tag": tag,
        "set": "s1",
        "arm": CONFIRMATORY_SLUG,
        "grain_note": DESCRIPTIVE_NOTE,
        "registered_pooled_reference": {
            k: reg_arm.get(k)
            for k in (
                "n_pairs",
                "diff_mean",
                "diff_ci",
                "p_wilcoxon",
                "p_holm",
                "holm_significant",
                "separates",
                "recovery_samewave",
            )
        },
        "per_cell": per_cell,
        "metadata": _metadata(tag, loaded["row_counts"]),
    }


# ── Deliverable 2: continuation-only lattice recount ──────────────────


def continuation_lattice(tag: str, loaded: dict) -> dict:
    f_st, f_nu = loaded["f_st"], loaded["f_nu"]
    _, _, passing, survivors = _s1_survivor_state(loaded["sep_by_pair"])
    reg_s1 = loaded["registered_s1"]

    arms_out: dict[str, dict] = {}
    pvals: dict[str, float] = {}
    for slug in PREFILL_SLUGS:
        t0 = time.time()
        pids = [
            p
            for p in survivors
            if f_st.get((p, slug), {}).get("f_beh_continuation") is not None
            and f_nu.get((p, slug), {}).get("f_beh_continuation") is not None
        ]
        counters = {
            "n_survivors": len(survivors),
            "rejected_missing_continuation": len(survivors) - len(pids),
        }
        _assert_nonempty(pids, f"lattice {tag}/s1/{slug}", counters)
        rec: dict = {"below_floor": (not passing) or len(pids) < A.S1_SURVIVAL_FLOOR}
        if rec["below_floor"]:
            rec.update({"n_pairs": len(pids), "label": "untestable-small-n"})
        else:
            val_st = {p: f_st[(p, slug)]["f_beh_continuation"] for p in pids}
            val_nu = {p: f_nu[(p, slug)]["f_beh_continuation"] for p in pids}
            rec.update(_arm_rec(pids, val_st, val_nu, loaded["ce_samewave"], loaded["ce_banked"]))
            rec["selection_counters"] = counters
            pvals[slug] = rec["p_wilcoxon"]
        arms_out[slug] = rec
        print(
            f"[lattice] {tag} {slug} n={rec.get('n_pairs')} "
            f"diff_ci={rec.get('diff_ci')} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    # Holm at the REGISTERED fixed family size m=12: the 6 patch arms have no
    # continuation companion, so they count as dropped arms (conservative).
    holmed = A.holm_fixed_m(pvals) if pvals else {}
    for slug, rec in arms_out.items():
        if slug in holmed:
            rec["p_holm_fixed_m12"] = holmed[slug]
            rec["holm_significant"] = holmed[slug] < A.HOLM_ALPHA
            if "diff_ci" in rec:
                rec["separates"] = (rec["diff_ci"][0] > 0) and rec["holm_significant"]

    verdicts: dict[str, dict] = {}
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
            ),
            "confirmatory": scheme == "med",
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

    # Side-by-side vs the registered whole-response reads (stats.json).
    arm_comparison: dict[str, dict] = {}
    for slug in PREFILL_SLUGS:
        reg = reg_s1["arms"].get(slug, {})
        cont = arms_out.get(slug, {})
        arm_comparison[slug] = {
            "registered_whole_response": {
                "n_pairs": reg.get("n_pairs"),
                "diff_ci": reg.get("diff_ci"),
                "p_holm": reg.get("p_holm"),
                "separates": reg.get("separates"),
            },
            "continuation_only": {
                "n_pairs": cont.get("n_pairs"),
                "diff_ci": cont.get("diff_ci"),
                "p_holm_fixed_m12": cont.get("p_holm_fixed_m12"),
                "separates": cont.get("separates"),
            },
            "separation_flip": bool(reg.get("separates")) != bool(cont.get("separates")),
        }
    verdict_comparison: dict[str, dict] = {}
    for scheme in C.ARM_SCHEMES:
        reg_v = reg_s1["prefill3_verdicts"].get(scheme, {})
        cont_v = verdicts.get(scheme, {})
        verdict_comparison[scheme] = {
            "registered_label": reg_v.get("label"),
            "continuation_label": cont_v.get("label"),
            "flip": reg_v.get("label") != cont_v.get("label"),
            "registered_label_banked_ce": reg_v.get("label_banked_ce"),
            "continuation_label_banked_ce": cont_v.get("label_banked_ce"),
            "flip_banked_ce": reg_v.get("label_banked_ce") != cont_v.get("label_banked_ce"),
        }

    return {
        "model_tag": tag,
        "set": "s1",
        "dv_note": (
            "F_beh replaced by f_beh_continuation (continuation-only companion, plan §6 "
            "exploratory read) across the 6 prefill arms; ce denominators (samewave "
            "primary / banked #2162 comparison) are whole-response by construction and "
            "held fixed; Holm at the registered FIXED m=12 (patch arms = dropped slots)."
        ),
        "n_survivors_tested": len(survivors),
        "arms": arms_out,
        "prefill3_verdicts": verdicts,
        "arm_comparison": arm_comparison,
        "verdict_comparison": verdict_comparison,
        "any_separation_flip": any(v["separation_flip"] for v in arm_comparison.values()),
        "any_verdict_flip": any(v["flip"] for v in verdict_comparison.values()),
        "metadata": _metadata(tag, loaded["row_counts"]),
    }


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model-tag", choices=[*MODEL_TAGS, "both"], default="both")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="verify imports + args-attribute completeness, then exit 0",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0

    tags = MODEL_TAGS if args.model_tag == "both" else (args.model_tag,)
    summary: dict = {"per_model": {}}
    for tag in tags:
        loaded = _load_tag(tag)
        pc = percell_prefill3(tag, loaded)
        lat = continuation_lattice(tag, loaded)
        out_dir = FMETRICS / tag / "followup_free"
        A62._write_json_atomic(out_dir / "percell_prefill3.json", pc)
        A62._write_json_atomic(out_dir / "continuation_lattice.json", lat)
        print(f"[write] {out_dir}/percell_prefill3.json + continuation_lattice.json", flush=True)
        summary["per_model"][tag] = {
            "percell": {
                cell: {
                    "n_pairs": r["n_pairs"],
                    "recovery_ratio_samewave": r.get("recovery_samewave", {}).get("ratio"),
                    "recovery_ratio_ci": r.get("recovery_samewave", {}).get("ratio_ci"),
                    "diff_ci": r["diff_ci"],
                    "p_wilcoxon_raw": r["p_wilcoxon"],
                    "separation_read_descriptive": r["separation_read_descriptive"]["both"],
                }
                for cell, r in pc["per_cell"].items()
            },
            "continuation_verdicts": lat["verdict_comparison"],
            "continuation_separation_flips": {
                slug: v["separation_flip"] for slug, v in lat["arm_comparison"].items()
            },
            "any_separation_flip": lat["any_separation_flip"],
            "any_verdict_flip": lat["any_verdict_flip"],
        }
    if args.model_tag == "both":
        summary["grain_note"] = DESCRIPTIVE_NOTE
        summary["metadata"] = _metadata("both", {})
        A62._write_json_atomic(FMETRICS / "followup_free_summary.json", summary)
        print(f"[write] {FMETRICS}/followup_free_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
