#!/usr/bin/env python3
"""Issue #779 (training-source-ablation-hg): 2D scaling-grid + arm-comparison CLI.

The 0-GPU analysis driver (plan v6 §4.4/§4.5/§4.6). Reads the CACHED Arm A
(pass_b LMSYS) bundle + the GENERATED-ONCE Arm B/C behavior corpus + the FIXED
eval rig (pass_a *_cx.pt + judge scores) + the frozen read-out layers (step0),
and produces every on-source primary deliverable:

  eval_results/issue_779/training-source-ablation-hg/arm_comparison.json
      pv_raw + oracle + Arm A/B/C(natural,1:1) + g, in-behavior; + the pv contrast
      triple {r_B, Mᵀr_B, M⁺r_B} (the mentor pv_pinv amendment). The headline.
  eval_results/issue_779/training-source-ablation-hg/scaling_grid.json
      7x7 grid x K subsamples x {h,g}; the HB-N + HC learning-curve reads.
  eval_results/issue_779/training-source-ablation-hg/scaling_grid_layer_matrix.json
      arm-vs-A read at ALL 28 layers (observed + shuffle-null rows); the
      selection-symmetric honest-band recompute source.
  eval_results/issue_779/training-source-ablation-hg/g_holdout_question.json
      leakage-free K-fold g over the 20 eval questions; body-error #2 fix.

Vectorized (vectorize-many-cell-fits.md): the h read is a closed-form ridge fit
per cell (the batched primitive); NO serial AdamW loop. The vectorized numbers
are cross-checked against a serial reference on a few cells before the grid runs
(--verify-vectorized). 0-GPU: all fits on cached / generated tensors.

Frozen read-out layers (selection-symmetric-nulls §4.5): read from step0
best_by_mode -- the parent's committed per-trait x mode positions used as a
PRE-REGISTERED fixed position (the carve-out). Both the observed arm-vs-arm
statistic AND every null row are read at that single frozen layer; the per-draw
x per-28-layer matrix is persisted for post-hoc honest-band recompute.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_stage1 as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import scaling_grid as SG  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_scaling_grid")

OUT_SUBDIR = "training-source-ablation-hg"
MODES = ("system", "many_shot")


# ── frozen read-out layers (step0 best_by_mode; §4.5 pre-registered position) ─


def frozen_readout_layers(step0_path: Path) -> dict:
    """Read the committed per-trait x mode read-out layers from step0_oracle.json.

    Returns {trait: {"system": L, "many_shot": L}}. These are the parent's argmax
    oracle_r selections used as a PRE-REGISTERED fixed position (selection-
    symmetric-nulls carve-out): the amendment reads observed + null at this single
    frozen layer, and persists the full per-layer matrix for honest-band recompute.
    """
    with open(step0_path) as f:
        step0 = json.load(f)
    out = {}
    traits = step0.get("traits", step0)
    for trait, td in traits.items():
        bbm = td.get("best_by_mode") if isinstance(td, dict) else None
        if bbm is None:
            continue
        out[trait] = {
            "system": int(
                bbm["system"]["layer"] if isinstance(bbm.get("system"), dict) else bbm["system"]
            ),
            "many_shot": int(
                bbm["many_shot"]["layer"]
                if isinstance(bbm.get("many_shot"), dict)
                else bbm["many_shot"]
            ),
        }
    return out


# ── build the per-trait eval matrix (with a question index) at a layer ────────


def build_eval_matrix_with_q(cells: list[dict], layer_idx: int, r_b: np.ndarray) -> dict:
    """build_eval_matrix + a per-row eval-question index (for g_holdout_question).

    Reuses S.build_eval_matrix for the numeric arrays (per-question unit,
    within-condition Pearson matched to PV), then re-walks the cells IN THE SAME
    ORDER to attach the eval-question index each row came from (the K-fold split
    unit). The re-walk mirrors build_eval_matrix's grouping exactly (rollouts
    grouped by qi in first-seen order, questions with >=1 valid rollout kept).
    """
    mat = S.build_eval_matrix(cells, layer_idx, r_b)
    q_idx = []
    for cell in cells:
        by_q: dict[int, list[dict]] = {}
        for rec in cell["rollouts"]:
            if rec.get("empty"):
                continue
            by_q.setdefault(rec["qi"], []).append(rec)
        for qi, recs in by_q.items():
            q_scores = [s for r in recs if (s := S._score_for(cell, qi, r["ri"])) is not None]
            if not q_scores:
                continue
            q_idx.append(int(qi))
    assert len(q_idx) == mat["c_last"].shape[0], (len(q_idx), mat["c_last"].shape[0])
    mat["question"] = np.array(q_idx)
    return mat


# ── load the Arm B corpus into a per-layer TrainSource ────────────────────────


def load_corpus_source(
    corpus_dir: Path,
    lmsys_bundle: dict,
    lmsys_g_labels: dict | None,
    trait: str,
    layer_idx: int,
    *,
    max_lmsys: int | None = None,
) -> SG.TrainSource:
    """Build a TrainSource at one read-out layer from cached + generated tensors.

    LMSYS side (Arm A axis): pass_b cx_last / v_x at the layer + (optional) the
    regenerated g labels aligned by context index. Behavior side (Arm B axis):
    the corpus bundle's cx_last (per context) + v_x (per valid rollout) + judge
    scores. For h the behavior X/Y is per-ROLLOUT (v(x) is per rollout); c_last
    is per-context, so each rollout's X is its context's c_last (broadcast via
    vx_index). g's behavior label is the rollout's judge score.

    ``max_lmsys`` caps the LMSYS rows loaded (the first-N contexts, keeping the
    deterministic pass_b order + label alignment) — used ONLY in --smoke to keep
    the full-N ridge SVD tractable; None (default) loads all 5000.
    """
    layers = list(lmsys_bundle["layers"])
    li = layers.index(layer_idx)
    X_lmsys = lmsys_bundle["cx_last"][:, li, :].to(torch.float32).numpy()  # (N_L, H)
    Y_lmsys = lmsys_bundle["v_x"][:, li, :].to(torch.float32).numpy()  # (N_L, H)
    if max_lmsys is not None and X_lmsys.shape[0] > max_lmsys:
        X_lmsys = X_lmsys[:max_lmsys]
        Y_lmsys = Y_lmsys[:max_lmsys]
    y_lmsys = None
    if lmsys_g_labels is not None:
        labs = lmsys_g_labels["labels_per_trait"][trait]["labels"]
        arr = np.array([np.nan if v is None else float(v) for v in labs], dtype=np.float64)
        n = min(len(arr), X_lmsys.shape[0])
        # align by index; keep only rows with a valid label for g (h uses all)
        y_lmsys = arr[:n]
        X_lmsys = X_lmsys[:n]
        Y_lmsys = Y_lmsys[:n]

    # Behavior corpus.
    bundle_path = corpus_dir / f"{trait}_corpus.pt"
    scores_path = corpus_dir / f"{trait}_judge_scores.json"
    cb = torch.load(bundle_path, weights_only=False)
    clayers = list(cb["layers"])
    cli = clayers.index(layer_idx)
    cx_last_ctx = cb["cx_last"][:, cli, :].to(torch.float32).numpy()  # (n_contexts, H)
    v_x = cb["v_x"]  # (n_valid_rollouts, L, H)
    vx_index = cb["vx_index"]  # [(context_idx, rollout_idx)]
    with open(scores_path) as f:
        scores = json.load(f)["scores"]

    X_beh, Y_beh, y_beh = [], [], []
    labels_all_present = True
    for k, (ci, ri) in enumerate(vx_index):
        X_beh.append(cx_last_ctx[ci])  # per-context c_last for this rollout
        Y_beh.append(v_x[k, cli, :].to(torch.float32).numpy())  # v(x) at layer
        s = scores.get(str(ci), {}).get(str(ri))
        if s is None:
            labels_all_present = False
            y_beh.append(np.nan)
        else:
            y_beh.append(float(s))
    X_beh = np.array(X_beh) if X_beh else np.empty((0, X_lmsys.shape[1]))
    Y_beh = np.array(Y_beh) if Y_beh else np.empty((0, X_lmsys.shape[1]))
    y_beh_arr = (
        np.array(y_beh) if (y_beh and labels_all_present) else (np.array(y_beh) if y_beh else None)
    )
    return SG.TrainSource(X_lmsys, Y_lmsys, y_lmsys, X_beh, Y_beh, y_beh_arr)


# ── arm comparison (pv_raw / oracle / arms / g + pv_pinv triple) ──────────────


def _readout_r(x, eval_mat, *, n_boot, seed):
    return SG._within_condition_r(x, eval_mat, n_boot=n_boot, seed=seed)


def run_arm_comparison(
    src: SG.TrainSource,
    eval_mat: dict,
    rb_l: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    pinv_rank: int | None,
) -> dict:
    """pv_raw / oracle / Arm A/B/C(natural,1:1) map read + g + the pv_pinv triple.

    Each arm's h is fit on that arm's FULL rows (Arm A = all LMSYS, Arm B = all
    behavior, Arm C = both). The pv contrast triple is read PER ARM:
      w=r_B    -> pv_raw (identical across arms; no map)  [<c_last, r_B>]
      w=Mᵀr_B  -> the linear-h dot read <h(c),r_B>        (already computed)
      w=M⁺r_B  -> pv_pinv (NEW; the min-norm preimage)     [<c_last, M⁺r_B>]
    """
    out = {}
    # pv_raw + oracle (no map; identical across arms).
    out["pv_raw"] = _readout_r(eval_mat["pv_raw"], eval_mat, n_boot=n_boot, seed=seed)
    out["oracle"] = _readout_r(eval_mat["oracle"], eval_mat, n_boot=n_boot, seed=seed)

    rng = np.random.default_rng(seed)
    arms = {
        "arm_a_lmsys": (src.n_lmsys(), 0, False),
        "arm_b_behavior": (0, src.n_beh(), False),
        "arm_c_combined_natural": (src.n_lmsys(), src.n_beh(), False),
        "arm_c_combined_1to1": (src.n_lmsys(), src.n_beh(), True),
    }
    for arm, (nL, nB, up) in arms.items():
        ct = SG.assemble_cell_train(src, nL, nB, rng, upsample_1to1=up)
        h = SG.fit_h_cell(ct["X"], ct["Y"], eval_mat, rb_l)
        g = SG.fit_g_cell(ct["X"], ct["y"], eval_mat)
        pinv_read, pinv_full = SG.pv_pinv_reads(h["W"], rb_l, eval_mat, rank=pinv_rank)
        out[arm] = {
            "n_lmsys_used": ct["n_lmsys_used"],
            "n_behavior_used": ct["n_behavior_used"],
            "h_ridge_dot": _readout_r(h["dot"], eval_mat, n_boot=n_boot, seed=seed),  # w=Mᵀr_B
            "h_ridge_cos": _readout_r(h["cos"], eval_mat, n_boot=n_boot, seed=seed),
            "g_ridge": _readout_r(g, eval_mat, n_boot=n_boot, seed=seed),
            "pv_pinv": _readout_r(
                pinv_read, eval_mat, n_boot=n_boot, seed=seed
            ),  # w=M⁺r_B (frozen rank)
            "pv_pinv_fullrank": _readout_r(
                pinv_full, eval_mat, n_boot=n_boot, seed=seed
            ),  # diagnostic
            "has_labels": ct["has_labels"],
        }
    out["pv_contrast_triple_note"] = (
        "w=r_B => pv_raw; w=M^T r_B => h_ridge_dot (transpose, sigma-weighted detection); "
        "w=M^+ r_B => pv_pinv (pseudoinverse, 1/sigma-weighted min-norm preimage). "
        "pinv rank frozen on TRAIN; pv_pinv_fullrank is the diagnostic."
    )
    return out


# ── selection-symmetric per-layer matrix (all 28 layers, observed + null) ─────


def run_layer_matrix(
    corpus_dir: Path,
    lmsys_bundle: dict,
    lmsys_g_labels: dict | None,
    cells: list[dict],
    r_b_full: np.ndarray,
    trait: str,
    *,
    layers: list[int],
    n_boot: int,
    n_shuffle: int,
    seed: int,
    max_lmsys: int | None = None,
) -> dict:
    """The arm-B-minus-arm-A within-condition-r headline read at EVERY layer.

    For each layer: build the eval matrix, fit Arm A + Arm B h, read the observed
    Δr (Arm B - Arm A, h_ridge_dot) per mode; then ``n_shuffle`` label-shuffle
    null rows (permute the eval y within condition, recompute Δr). Persisted so
    the analyzer can recompute an honest max-over-layer selected band post-hoc
    (selection-symmetric-nulls) WITHOUT re-running the pod pass. This IS the
    layer axis the read-out freeze neutralizes.
    """
    rows = {"observed": {m: [] for m in MODES}, "null": {m: [] for m in MODES}}
    rng = np.random.default_rng(seed)
    for layer_idx in layers:
        rb_l = r_b_full[layers.index(layer_idx)]
        eval_mat = build_eval_matrix_with_q(cells, layer_idx, r_b_full)
        src = load_corpus_source(
            corpus_dir, lmsys_bundle, lmsys_g_labels, trait, layer_idx, max_lmsys=max_lmsys
        )
        subrng = np.random.default_rng(seed + layer_idx)
        ct_a = SG.assemble_cell_train(src, src.n_lmsys(), 0, subrng)
        ct_b = SG.assemble_cell_train(src, 0, src.n_beh(), subrng)
        ha = SG.fit_h_cell(ct_a["X"], ct_a["Y"], eval_mat, rb_l)
        hb = SG.fit_h_cell(ct_b["X"], ct_b["Y"], eval_mat, rb_l)
        ra = _readout_r(ha["dot"], eval_mat, n_boot=1, seed=seed)
        rb = _readout_r(hb["dot"], eval_mat, n_boot=1, seed=seed)
        for m in MODES:
            rows["observed"][m].append(
                {
                    "layer": layer_idx,
                    "delta": _safe(rb[m]["point"]) - _safe(ra[m]["point"]),
                    "arm_a_r": _safe(ra[m]["point"]),
                    "arm_b_r": _safe(rb[m]["point"]),
                }
            )
        # shuffle-null: permute eval y within condition, recompute Δr.
        for _ in range(n_shuffle):
            y_shuf = _shuffle_within_condition(eval_mat["y"], eval_mat["cond"], rng)
            em2 = dict(eval_mat)
            em2["y"] = y_shuf
            ra_n = _readout_r(ha["dot"], em2, n_boot=1, seed=seed)
            rb_n = _readout_r(hb["dot"], em2, n_boot=1, seed=seed)
            for m in MODES:
                rows["null"][m].append(
                    {"layer": layer_idx, "delta": _safe(rb_n[m]["point"]) - _safe(ra_n[m]["point"])}
                )
    return {"n_layers": len(layers), "n_shuffle": n_shuffle, "rows": rows}


def _safe(v) -> float:
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def _shuffle_within_condition(
    y: np.ndarray, cond: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    out = y.copy()
    for c in np.unique(cond):
        m = cond == c
        idx = np.where(m)[0]
        out[idx] = y[rng.permutation(idx)]
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 2D scaling grid + arm comparison.")
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument(
        "--collect-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "issue779_hfstage"
        / "issue779_monitoring"
        / "analysis_tensors",
        help="tree with pass_a/, pass_b/, r_b/, step0/",
    )
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_779" / "behavior_corpus",
        help="Arm B corpus bundles (from issue779_gen_behavior_corpus.py)",
    )
    ap.add_argument(
        "--lmsys-g-labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_779" / "lmsys_g_labels" / "lmsys_g_labels.json",
        help="Arm A regenerated g labels (optional; absent -> Arm A g is NaN)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / OUT_SUBDIR,
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--k-subsamples", type=int, default=SG.DEFAULT_K)
    ap.add_argument("--n-shuffle", type=int, default=100)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--pinv-rank", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=C.EXPECTED_LAYERS)
    ap.add_argument("--hidden", type=int, default=C.EXPECTED_HIDDEN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-lmsys-grid", type=int, nargs="+", default=list(SG.DEFAULT_N_LMSYS))
    ap.add_argument("--n-behavior-grid", type=int, nargs="+", default=list(SG.DEFAULT_N_BEHAVIOR))
    ap.add_argument(
        "--verify-vectorized",
        action="store_true",
        help="run the vectorized_mlp_skill serial-reference equivalence check first",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny grid + reduced boot (wiring smoke)")
    args = ap.parse_args()

    if args.verify_vectorized:
        from explore_persona_space.analysis.vectorized_mlp_skill import assert_matches_reference

        res = assert_matches_reference()
        logger.info("vectorized_mlp_skill serial-reference check: %s", res)

    collect = args.collect_dir
    pass_a_dir = collect / "pass_a"
    rb_dir = collect / "r_b"
    pass_b_path = collect / "pass_b" / "train_context_vectors.pt"
    step0_path = collect / "step0" / "step0_oracle.json"
    for p in (pass_a_dir, rb_dir, pass_b_path, step0_path):
        if not p.exists():
            raise FileNotFoundError(f"required staged input missing: {p}")

    frozen = frozen_readout_layers(step0_path)
    logger.info("Frozen read-out layers (step0 best_by_mode): %s", frozen)
    lmsys_bundle = torch.load(pass_b_path, weights_only=False)

    # Arm A g labels (optional; the label-floor diagnostic). Absent -> Arm A g NaN.
    lmsys_g_labels = None
    if args.lmsys_g_labels.exists():
        with open(args.lmsys_g_labels) as f:
            lmsys_g_labels = json.load(f)
        logger.info("Loaded Arm A LMSYS g labels: %s", lmsys_g_labels.get("summary"))
    else:
        logger.warning(
            "Arm A LMSYS g labels missing at %s -> Arm A g reads NaN (label-floor case)",
            args.lmsys_g_labels,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_sub = 2 if args.smoke else args.k_subsamples
    n_boot = 50 if args.smoke else args.n_boot
    n_shuffle = 3 if args.smoke else args.n_shuffle
    n_lmsys_grid = [0, 500] if args.smoke else args.n_lmsys_grid
    n_behavior_grid = [0, 100] if args.smoke else args.n_behavior_grid
    # The per-layer selection-symmetric matrix reads at ALL 28 layers in a real
    # run; in --smoke it reads only the per-trait FROZEN layers (a handful) so the
    # 28x{2 fits + n_shuffle nulls} loop stays tractable while still exercising
    # the run_layer_matrix code path end-to-end.
    layers_all = list(range(args.n_layers))
    # --smoke caps the LMSYS rows loaded so the full-N (5000x3584) ridge SVD in
    # the arm_comparison / layer matrix stays fast; a real run uses all 5000.
    max_lmsys = 500 if args.smoke else None

    arm_comparison = {
        "traits": {},
        "meta": C.reproducibility_metadata({"script": "issue779_scaling_grid"}),
    }
    scaling = {
        "traits": {},
        "meta": C.reproducibility_metadata({"script": "issue779_scaling_grid"}),
    }
    layer_matrix = {
        "traits": {},
        "meta": C.reproducibility_metadata({"script": "issue779_scaling_grid"}),
    }
    g_holdout = {
        "traits": {},
        "meta": C.reproducibility_metadata({"script": "issue779_scaling_grid"}),
    }

    for trait in args.traits:
        r_b_full = S._load_rb(rb_dir, trait, args.n_layers, args.hidden)  # (L, H)
        cells = S.load_eval_cells(pass_a_dir, trait)
        # per-mode frozen layer; use SYSTEM-mode layer for the primary read + record both.
        fl = frozen.get(trait, {"system": args.n_layers // 2, "many_shot": args.n_layers // 2})
        # Read at each mode's own frozen layer, aggregating into one comparison.
        arm_comparison["traits"][trait] = {}
        scaling["traits"][trait] = {}
        g_holdout["traits"][trait] = {}
        for mode in MODES:
            layer_idx = fl[mode]
            rb_l = r_b_full[layer_idx]
            eval_mat = build_eval_matrix_with_q(cells, layer_idx, r_b_full)
            src = load_corpus_source(
                args.corpus_dir,
                lmsys_bundle,
                lmsys_g_labels,
                trait,
                layer_idx,
                max_lmsys=max_lmsys,
            )
            logger.info(
                "[%s/%s @L%d] eval rows=%d, LMSYS=%d, behavior=%d",
                trait,
                mode,
                layer_idx,
                eval_mat["c_last"].shape[0],
                src.n_lmsys(),
                src.n_beh(),
            )
            ac = run_arm_comparison(
                src, eval_mat, rb_l, n_boot=n_boot, seed=args.seed, pinv_rank=args.pinv_rank
            )
            arm_comparison["traits"][trait][mode] = {"frozen_layer": layer_idx, **ac}
            grid = SG.run_scaling_grid(
                src,
                eval_mat,
                rb_l,
                n_lmsys_grid=n_lmsys_grid,
                n_behavior_grid=n_behavior_grid,
                k_subsamples=k_sub,
                n_boot=n_boot,
                base_seed=args.seed,
            )
            grid_1to1 = SG.run_scaling_grid(
                src,
                eval_mat,
                rb_l,
                n_lmsys_grid=n_lmsys_grid,
                n_behavior_grid=n_behavior_grid,
                k_subsamples=k_sub,
                n_boot=n_boot,
                base_seed=args.seed,
                upsample_1to1=True,
            )
            scaling["traits"][trait][mode] = {
                "frozen_layer": layer_idx,
                "natural": grid,
                "upsample_1to1": grid_1to1,
            }
            gho = SG.run_g_holdout_question(
                eval_mat, k_folds=args.k_folds, n_boot=n_boot, base_seed=args.seed
            )
            g_holdout["traits"][trait][mode] = {"frozen_layer": layer_idx, **gho}
            # checkpoint per (trait, mode)
            C.write_json_atomic(args.out_dir / "arm_comparison.json", arm_comparison)
            C.write_json_atomic(args.out_dir / "scaling_grid.json", scaling)
            C.write_json_atomic(args.out_dir / "g_holdout_question.json", g_holdout)

        # per-layer selection-symmetric matrix (all 28 layers in a real run;
        # the per-trait frozen layers only under --smoke for tractability).
        lm_layers = sorted(set(fl.values())) if args.smoke else layers_all
        layer_matrix["traits"][trait] = run_layer_matrix(
            args.corpus_dir,
            lmsys_bundle,
            lmsys_g_labels,
            cells,
            r_b_full,
            trait,
            layers=lm_layers,
            n_boot=n_boot,
            n_shuffle=n_shuffle,
            seed=args.seed,
            max_lmsys=max_lmsys,
        )
        C.write_json_atomic(args.out_dir / "scaling_grid_layer_matrix.json", layer_matrix)
        logger.info("[%s] all reads checkpointed", trait)

    logger.info(
        "Wrote arm_comparison / scaling_grid / scaling_grid_layer_matrix / g_holdout_question"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
