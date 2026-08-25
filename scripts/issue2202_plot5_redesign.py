#!/usr/bin/env python3
"""Issue #2202 — paper Plot 5 redesign: failure set under whitened cosine + CSLS.

The committed paper figure (figures/paper/c3_failure_attribution, produced by
``issue2202_regen_figs.fig_c3_failure_analysis_iclr``) defines its failure
population under RAW EUCLIDEAN retrieval (1,829/9,941 rank-1 failures). The
project's standing retrieval convention is whitened cosine + CSLS (k=10), under
which far fewer contexts fail. This script recomputes the per-context mid-ranks
under that convention from the staged banked tensors, re-runs the P5 contrast
battery on the new failure indicator, and overwrites the paper figure in place.

Phases (checkpoint-per-phase; each persists before the next runs):

- ``ranks``  recompute per-context mid-ranks under whitened-cos+CSLS from
             /mnt/eps-data/.../issue2202_freshwhiten/{pred16,y_holdout_L19,
             whiten_stats}.npz, REUSING issue2202_metric_zoo (load_staged,
             banked-L chol whitening at lam=0.1, csls_ranks, ranks_summary);
             reconciliation-gated against the banked metric-zoo
             csls_k10_whitencos summary (acc@1 = 0.97616).
- ``stats``  the parent P5 instrument on the new binary fail indicator:
             issue2202_stats_figs.run_battery (22 registered contrasts,
             10k batched bootstrap + 10k batched permutations, BH q=0.05)
             + attribution-class counts (banked kres_class) over the new set.
- ``fig``    three-panel ICLR figure overwriting figures/paper/
             c3_failure_attribution.{png,pdf,meta.json}: (a) the current
             figure's 13 contrasts recomputed on the whitened-cos+CSLS failure
             set, (b) resample-attribution shares over the covered subset of
             the new failures (dropped if coverage is too thin), (c) the
             unchanged per-architecture raw vs whitened+CSLS panel.

Reads only committed eval_results/issue_2202 artifacts + the read-only staged
copies; 0 GPU-h; runs on the shared VM under the standard thread caps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2202_failchar as FC  # noqa: E402  (atomic_json / meta_block / out_eval_dir)
import issue2202_metric_zoo as MZ  # noqa: E402  (load_staged / csls_ranks / ranks_summary)
import numpy as np  # noqa: E402

OUT_SUBDIR = "plot5_redesign"
BANKED_ARM = "csls_k10_whitencos"
ACC1_GATE_TOL = 2e-4  # |recomputed − banked| acc@1; ~2 contexts of mid-rank tie drift
MIN_COVERED_FOR_PANEL_B = 30  # below this the attribution panel is dropped as too thin
CONVENTION = (
    "whitened cosine + CSLS (k=10): z = L^-1(x - mu_A) at the task-locked shrunk "
    "train-answer covariance (lam=0.1, Cholesky basis), cosine scores, CSLS rescore "
    "score = S - 0.5*r_j (query-bank k-NN mean penalty); mid-rank ties per "
    "knn_retrieval; fail = mid-rank > 1"
)

# plain-English labels for every registered contrast (the current figure's dict,
# extended to the 9 non-significant contrasts so any set change stays renderable)
CONTRAST_LABEL = {
    "language=en": "English",
    "topic=factual_qa": "factual QA topic",
    "topic=creative_writing": "creative-writing topic",
    "topic=coding": "coding topic",
    "topic=advice_howto": "advice / how-to topic",
    "topic=chitchat_social": "chit-chat topic",
    "topic=translation": "translation topic",
    "topic=math": "math topic",
    "topic=summarization_extraction": "summarization topic",
    "topic=harmful_or_unsafe_request": "harmful-request topic",
    "topic=roleplay_persona": "roleplay / persona topic",
    "topic=nsfw": "NSFW topic",
    "topic=other": "'other' topic",
    "refusal_adjacent=yes": "refusal-adjacent request",
    "answer_is_refusal=yes": "answer is a refusal",
    "format=code": "code-format answer",
    "format=list": "list-format answer",
    "format=prose": "prose-format answer",
    "depth=2-2": "2-turn conversation",
    "depth=3-4": "3-4-turn conversation",
    "depth=>=5": "deep conversation (5+ turns)",
    "corpus=wildchat": "WildChat corpus",
}


def out_dir(args) -> Path:
    d = FC.out_eval_dir(args) / OUT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_csv_rows(args) -> list[dict]:
    path = FC.out_eval_dir(args) / "percontext_ranks.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def banked_arm_summary(args) -> dict:
    """The banked metric-zoo csls_k10_whitencos summary (reconciliation target)."""
    path = FC.out_eval_dir(args) / "metric_zoo" / "results.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("name") == BANKED_ARM:
                return row["summary"]
    raise RuntimeError(f"{BANKED_ARM} not found in {path}")


def phase_ranks(args) -> None:
    """Recompute whitened-cos+CSLS mid-ranks; reconcile vs the banked arm."""
    from scipy.linalg import solve_triangular

    staged = MZ.load_staged(Path(args.staged))
    pred, y16, pci = staged["pred"], staged["y16"], staged["pci"]
    ell, mu_a = staged["stats"]["L"], staged["stats"]["mu_A"]
    n = pred.shape[0]
    true_idx = np.arange(n)

    csv_rows = load_csv_rows(args)
    ci_csv = np.asarray([int(r["ci"]) for r in csv_rows], dtype=np.int64)
    assert np.array_equal(ci_csv, pci), "percontext_ranks.csv / pred16 ci misalign"

    # banked-L Cholesky whitening (identical to MZ.Transforms.chol_whiten at lam=0.1,
    # taken directly so no eigendecomposition is needed)
    pred_w = solve_triangular(ell, (pred - mu_a).T, lower=True).T
    pool_w = solve_triangular(ell, (y16 - mu_a).T, lower=True).T
    pwn = pred_w / (np.linalg.norm(pred_w, axis=1, keepdims=True) + 1e-12)
    qwn = pool_w / (np.linalg.norm(pool_w, axis=1, keepdims=True) + 1e-12)
    s_wc = pwn @ qwn.T
    print("[ranks] whitened-cos S computed", flush=True)
    ranks = MZ.csls_ranks(s_wc, true_idx, 0.5)
    rec = MZ.ranks_summary(ranks, n)

    banked = banked_arm_summary(args)
    deltas = {
        str(k): rec["acc_at_k"][int(k)] - float(banked["acc_at_k"][str(k)]) for k in (1, 5, 10)
    }
    if abs(deltas["1"]) > ACC1_GATE_TOL:
        raise RuntimeError(
            f"reconciliation gate FAILED: recomputed acc@1 {rec['acc_at_k'][1]:.6f} vs "
            f"banked {banked['acc_at_k']['1']:.6f} (|delta| > {ACC1_GATE_TOL})"
        )

    fail_new = ranks > 1.0
    fail_raw = np.asarray([r["fail_raw_euclidean"] == "1" for r in csv_rows])
    z = np.load(FC.out_eval_dir(args) / "csls_percontext_ranks.npz")
    assert np.array_equal(np.asarray(z["ci"], dtype=np.int64), pci)
    fail_ccos = np.asarray(z["rank_csls"]) > 1.0

    od = out_dir(args)
    tmp = od / "whitencos_csls_ranks.tmp.npz"
    np.savez(tmp, ci=pci, rank_whitencos_csls=ranks)
    tmp.replace(od / "whitencos_csls_ranks.npz")
    FC.atomic_json(
        od / "ranks_summary.json",
        {
            "convention": CONVENTION,
            "recomputed": rec,
            "banked": banked,
            "reconciliation_deltas_acc_at_k": deltas,
            "gate": {"tol_acc1": ACC1_GATE_TOL, "ok": True},
            "failure_sets": {
                "raw_euclidean": int(fail_raw.sum()),
                "cosine_csls": int(fail_ccos.sum()),
                "whitencos_csls": int(fail_new.sum()),
                "whitencos_csls_and_raw": int((fail_new & fail_raw).sum()),
                "whitencos_csls_new_vs_raw": int((fail_new & ~fail_raw).sum()),
                "whitencos_csls_and_cosine_csls": int((fail_new & fail_ccos).sum()),
                "whitencos_csls_new_vs_cosine_csls": int((fail_new & ~fail_ccos).sum()),
            },
            "staged_inputs": str(args.staged),
            "meta": FC.meta_block(),
        },
    )
    print(
        f"[ranks] done: acc@1={rec['acc_at_k'][1]:.6f} (banked delta {deltas['1']:+.2e}); "
        f"failures {int(fail_new.sum())}/{n}",
        flush=True,
    )


def phase_stats(args) -> None:
    """P5 contrast battery + attribution-class counts on the new failure set."""
    import issue1738_characterize as CH
    import issue2202_labels as LB
    import issue2202_stats_figs as SF

    od = out_dir(args)
    z = np.load(od / "whitencos_csls_ranks.npz")
    ranks, pci = np.asarray(z["rank_whitencos_csls"]), np.asarray(z["ci"], dtype=np.int64)
    fail_new = (ranks > 1.0).astype(float)

    csv_rows = load_csv_rows(args)
    ci_rows = np.asarray([int(r["ci"]) for r in csv_rows], dtype=np.int64)
    assert np.array_equal(ci_rows, pci)
    labels = LB.load_labels_1738(args)
    fields = {int(k): v for k, v in LB.resolve_ci_fields(args).items()}
    masks = CH._contrast_masks(ci_rows, labels, fields)
    n_boot = 200 if args.smoke else SF.N_BOOT
    n_perm = 200 if args.smoke else SF.N_PERM
    battery = SF.run_battery(fail_new, masks, n_boot, n_perm, SF.STAT_SEED)

    # attribution classes (banked kres_class; covered = the 1,988 kresample contexts)
    fail_mask = fail_new.astype(bool)
    counts: dict[str, int] = {}
    for r, f in zip(csv_rows, fail_mask):
        if f:
            counts[r["kres_class"]] = counts.get(r["kres_class"], 0) + 1
    covered = sum(v for k, v in counts.items() if k != "UNKNOWN")

    comp = json.loads((FC.out_eval_dir(args) / "composition_stats.json").read_text())
    banked_sig = [r["contrast"] for r in comp["banked_battery"] if r["bh_significant"]]
    new_sig = [r["contrast"] for r in battery if r["bh_significant"]]
    FC.atomic_json(
        od / "plot5_stats.json",
        {
            "convention": CONVENTION,
            "n_fail": int(fail_mask.sum()),
            "n_boot": n_boot,
            "n_perm": n_perm,
            "bh_q": SF.BH_Q,
            "seed": SF.STAT_SEED,
            "battery": battery,
            "bh_significant_new": new_sig,
            "bh_significant_banked_raw_euclidean": banked_sig,
            "lost_significance_vs_raw": sorted(set(banked_sig) - set(new_sig)),
            "gained_significance_vs_raw": sorted(set(new_sig) - set(banked_sig)),
            "attribution_over_new_failures": {
                "counts": counts,
                "n_covered": covered,
                "n_fail": int(fail_mask.sum()),
                "coverage_note": (
                    "per-failure attribution classes exist only for the 1,988 "
                    "kresample-covered contexts (banked kres_class); shares in the "
                    "figure are over the covered subset of the new failures"
                ),
            },
            "meta": FC.meta_block({"smoke": bool(args.smoke)}),
        },
    )
    print(
        f"[stats] done: {len(new_sig)}/{len(battery)} BH-significant; "
        f"attribution coverage {covered}/{int(fail_mask.sum())}",
        flush=True,
    )


def phase_fig(args) -> None:
    """Overwrite figures/paper/c3_failure_attribution with the CSLS-convention figure."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    od = out_dir(args)
    stats = json.loads((od / "plot5_stats.json").read_text())
    comp = json.loads((FC.out_eval_dir(args) / "composition_stats.json").read_text())
    avg = json.loads((FC.out_eval_dir(args) / "avgtgt_completion" / "summary.json").read_text())

    # panel (a): union of the current figure's contrasts (BH-significant under the
    # raw-euclidean failure set) and those BH-significant under the new set, all
    # recomputed on the new failure set — shows both what dropped out and what emerged
    keep = {r["contrast"] for r in comp["banked_battery"] if r["bh_significant"]}
    keep |= set(stats["bh_significant_new"])
    by_name = {r["contrast"]: r for r in stats["battery"]}
    rows = sorted((by_name[c] for c in keep), key=lambda r: r["delta"])

    att = stats["attribution_over_new_failures"]
    n_fail = stats["n_fail"]
    covered = att["n_covered"]
    draw_b = covered >= MIN_COVERED_FOR_PANEL_B

    set_paper_style("iclr")
    if draw_b:
        fig, (ax_a, ax_b, ax_c) = plt.subplots(
            1, 3, figsize=(5.5, 2.5), gridspec_kw={"width_ratios": [2.3, 0.7, 1.8]}
        )
    else:
        fig, (ax_a, ax_c) = plt.subplots(
            1, 2, figsize=(5.5, 2.5), gridspec_kw={"width_ratios": [2.3, 1.8]}
        )
        ax_b = None

    ys = np.arange(len(rows))
    deltas = np.array([r["delta"] for r in rows]) * 100
    elo = np.maximum(0.0, np.array([r["delta"] - r["ci_lo"] for r in rows]) * 100)
    ehi = np.maximum(0.0, np.array([r["ci_hi"] - r["delta"] for r in rows]) * 100)
    sig = np.array([bool(r["bh_significant"]) for r in rows])
    colors = [paper_color("instruct") if s else "0.78" for s in sig]
    ax_a.barh(
        ys,
        deltas,
        xerr=(elo, ehi),
        color=colors,
        height=0.62,
        error_kw={"lw": 0.7, "capsize": 1.5},
    )
    ax_a.axvline(0, color=paper_color("reference"), lw=0.7)
    ax_a.set_yticks(ys, [CONTRAST_LABEL[r["contrast"]] for r in rows], fontsize=7)
    ax_a.set_xlabel("failure-rate difference (pp)")
    from matplotlib.patches import Patch

    handles = [
        Patch(color=paper_color("instruct"), label="significant (BH q=0.05)"),
        Patch(color="0.78", label="not significant"),
    ]
    ax_a.legend(
        handles=handles,
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=2,
        frameon=False,
    )

    # panel (b): resample attribution over the covered subset of the new failures
    if draw_b:
        order = [
            ("MAP_ATTRIBUTABLE", "map error", paper_color("instruct")),
            ("AMBIGUOUS", "ambiguous", "0.78"),
            ("IRREDUCIBLE", "answer degeneracy", paper_color("null")),
        ]
        bottom = 0.0
        for key, lab, colr in order:
            frac = att["counts"].get(key, 0) / covered * 100.0
            ax_b.bar([0], [frac], bottom=bottom, color=colr, width=0.55, label=lab)
            bottom += frac
        ax_b.set_xticks([])
        ax_b.set_xlim(-0.6, 0.6)
        ax_b.set_ylim(0, 100)
        ax_b.set_ylabel("share of covered failures (%)")
        ax_b.set_title(f"{covered} of {n_fail}\ncovered", fontsize=7)
        ax_b.legend(fontsize=7, loc="upper left", bbox_to_anchor=(-0.35, -0.12), ncols=1)

    # panel (c): rank-1 accuracy before/after the metric-side fixes (unchanged)
    archs = [
        ("ridge", "linear (ridge)"),
        ("mlp_w8192", "MLP"),
        ("mlp_w8192_seed43", "MLP (seed 43)"),
        ("krr_nystrom", "kernel ridge"),
        ("residual_skip", "residual MLP"),
        ("contrastive_linear", "contrastive linear"),
        ("contrastive_mlp", "contrastive MLP"),
    ]
    m = avg["matrix"]
    for i, (key, lab) in enumerate(archs):
        raw = m[key]["raw_euclidean"]["single"]["acc_at_k"]["1"]
        fixed = m[key]["csls_k10_whitencos"]["avg"]["acc_at_k"]["1"]
        ax_c.plot([raw, fixed], [i, i], color="0.8", lw=0.8, zorder=1)
        ax_c.scatter(
            [raw],
            [i],
            color=paper_color("null"),
            s=13,
            zorder=2,
            label="single draw, raw" if i == 0 else None,
        )
        ax_c.scatter(
            [fixed],
            [i],
            color=paper_color("instruct"),
            s=13,
            zorder=3,
            label="5-draw, whitened+CSLS" if i == 0 else None,
        )
    ax_c.set_yticks(range(len(archs)), [lab for _, lab in archs], fontsize=7)
    ax_c.set_xlabel("rank-1 retrieval accuracy")
    ax_c.set_xlim(0.55, 1.04)
    ax_c.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.02))
    savefig_paper(fig, "c3_failure_attribution", dir="figures/paper/")
    plt.close(fig)
    print(
        f"[fig] wrote figures/paper/c3_failure_attribution.{{png,pdf,meta.json}} "
        f"(panel b {'drawn' if draw_b else 'DROPPED — coverage too thin'})",
        flush=True,
    )


PHASES = {"ranks": phase_ranks, "stats": phase_stats, "fig": phase_fig}
PHASE_ORDER = ["ranks", "stats", "fig"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2202 paper Plot 5 CSLS redesign")
    ap.add_argument("--phase", choices=[*PHASE_ORDER, "all"], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--import-check", action="store_true", dest="import_check")
    ap.add_argument("--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_2202"))
    ap.add_argument("--staged", default=MZ.STAGED_DEFAULT)
    ap.add_argument("--hf-prefix", default=FC.HF_PREFIX_2202)
    ap.add_argument(
        "--labels-1738",
        default="eval_results/issue_1738/judge_labels/labels.json",
        dest="labels_1738",
    )
    ap.add_argument(
        "--ci-fields",
        default=str(PROJECT_ROOT / "data" / "issue_2202" / "ci_fields.json"),
        dest="ci_fields",
    )
    ap.add_argument("--work-root", default="/workspace/data/issue_2202")
    return ap


def _import_check() -> None:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("import-check OK: issue2202_plot5_redesign")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        _import_check()
        return 0
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check)")
    for ph in PHASE_ORDER if args.phase == "all" else [args.phase]:
        PHASES[ph](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
