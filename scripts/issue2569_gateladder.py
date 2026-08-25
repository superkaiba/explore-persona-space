"""#2569 leg 2 — gate-metric ladder + closed-form ridge learning curve (plan §4 leg 2).

Two halves, one module (unit 2 of the pre-split build):

1. **Gate-metric ladder** (leg 2 steps 1-2). Six context-gate metrics M scored per
   (arm, prefix) as the centered bilinear form ``g_i = (c_i - mu)^T M (a - mu)``
   with #1979's exact vector conventions — per-prefix QUERY-AVERAGED base context
   means ``c_i`` (the F1e ``Cbar`` tensors), centering at the full-PANEL mean
   (``issue1979_race.candidate_table``'s stated shared centering), and the arm's
   TRAINING-ROW-CENTROID anchor ``a`` (the F1e ``A_ctx_*`` mix anchors). Rungs:

   - ``gate_I``           — identity (raw similarity incumbent);
   - ``gate_diag_inv``    — ``diag(Sigma_c)^-1`` (diagonal-whitened incumbent);
   - ``gate_sigma_inv``   — ``Sigma_c^-1`` ridge-regularized by the #1979 shrinkage
     recipe (``issue1768_directions``: ``(1-s)*Sigma + s*(tr/d)*I``, s = 0.1);
   - ``gate_wwt``         — the through-map image inner product: both gate vectors
     pushed through the REGISTERED linear part via
     ``issue2569_operator.prediction_difference`` (B1 assert iii: never a
     re-derived product);
   - ``gate_wwt_k90``     — same, images projected onto W's top-k right-singular
     subspace (k = the leg-1 k90 count);
   - ``gate_wwt_awhite``  — same, images whitened by the shrunk ``Sigma_a^-1``.

   DVs are #1979's BANKED per-(arm, prefix) frames (committed at
   ``eval_results/issue_1979/race/frame_<arm>.json``): content arms
   ``dv_change`` (primary) / ``dv_level`` (secondary); marker arms the judge-free
   ``dv_dlogp`` / ``dv_level_logp``. Race = within-arm Spearman, never pooled
   across arms. Selection symmetry per the #1979 protocol verbatim: the winner is
   re-selected inside every bootstrap draw (default 10,000 draws, batched via the
   reused ``issue1900_race.bootstrap_battery`` rank-z GEMM — never a per-draw
   Python loop); the permutation null takes the per-draw SIGNED max over metrics;
   selection-inherited AND frozen-at-winner intervals are both persisted and
   labeled; band-vs-ceiling informativeness is reported.

2. **B4 H2b learning curve** (leg 2 step 3, re-registered plan v4). The VERDICT
   series is FRESH nested L19 refits on the assembled store's LMSYS subset at
   n in {4,500; 10,000; 50,000; 150,000; 500,000} (nested subsets, seed 2569),
   each via the reused ``issue779_ffc_n50k_fits.fit_ridge_primal`` (positional
   ``tr, val, te`` split-index arrays), lambda validation-selected per point over
   the WIDENED 27-value 1e-5..1e8 grid — a boundary selection triggers
   widen-and-reselect, never a reported edge value (C4). Verdict eval = a fixed
   5,000-row LMSYS slice, conversation-index-disjoint from every training subset.
   The theory curve evaluates the closed-form ridge learning-curve prediction
   (arXiv 2006.13198 self-consistent equations; effective regularization kappa by
   1-d root finding per n) from Sigma_c / Sigma_xy moments measured on the SAME
   LMSYS subset. Mean |dR2| <= 0.05 (H2b) is scored on the five protocol-matched
   points ONLY, gated by the mechanical fit-metadata parity check on
   {layer, training-corpus composition, eval-split shas, lambda-selection
   procedure, train-vs-eval distribution}; the committed heterogeneous points are
   labeled off-recipe COMPANIONS excluded from the statistic, published with
   per-point corpus mix, realized n, selected lambda and ``lambda_grid_edge``.

CLI::

    uv run python scripts/issue2569_gateladder.py ladder --out <dir> \
        --inputs-root <staging> [--sigma-c <pt>] [--sigma-a <pt>] [--smoke]
    uv run python scripts/issue2569_gateladder.py curve --x <npy> --y <npy> \
        --row-meta <pt> --out <dir> [--dev cuda] [--smoke]
    uv run python scripts/issue2569_gateladder.py --import-check

Sigma inputs (``--sigma-c`` / ``--sigma-a``) are P-B moment files: a ``.pt``/
``.npz`` carrying either ``{"sigma": (d,d)}`` directly or ``{"gram": (d,d),
"mean": (d,), "n_rows": int}`` (uncentered Gram; the loader centers). Row-meta
(``--row-meta``) carries ``{"corpus": (n,) str array, "conv_index": (n,) int}``
re-measured from the sampling manifest (plan §4 leg 2 step 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before numpy/torch: shared-VM thread caps + HF credentials

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

import issue2569_operator as OP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue2569.gateladder")

ISSUE = 2569
SEED = 2569

# ── ladder constants ──────────────────────────────────────────────────────────
GATE_METRICS = (
    "gate_I",
    "gate_diag_inv",
    "gate_sigma_inv",
    "gate_wwt",
    "gate_wwt_k90",
    "gate_wwt_awhite",
)
THROUGH_MAP_METRICS = ("gate_wwt", "gate_wwt_k90", "gate_wwt_awhite")
SIGMA_SHRINKAGE = 0.1  # vendored from issue1768_directions.SHRINKAGE (asserted at driver entry)
B_DRAWS_DEFAULT = 10_000  # plan §4 leg 2 step 2 (overrides #1979's B=2,000)
N_PERM_DEFAULT = 10_000  # sized to the bootstrap scale; batched (one GEMM), minutes
MIN_FAMILY_N = 4  # per-family win table: below this a family is reported skipped
LAYER_DEFAULT = 19  # the banked map's layer (plan-wide v_C convention)
POS_DEFAULT = "last_prompt"  # the registered map-input pooling (cx_last -> v_x, plan §6)
HF_PREFIX_1979 = "issue1979_prefixrace"
ANCHOR_KEY_BY_POS = {
    "span_mean_context": "A_ctx_span",
    "last_prompt": "A_ctx_last_prompt",
    "last_ctx": "A_ctx_last_ctx",
}
DV_NAMES_BY_KIND = {
    "content": ("dv_change", "dv_level"),
    "marker": ("dv_dlogp", "dv_level_logp"),
}
CI_QS = (0.025, 0.975)

# ── learning-curve constants (plan §4 leg 2 step 3, B4) ───────────────────────
LAMBDA_GRID_27 = tuple(np.logspace(-5.0, 8.0, 27))  # widened grid (C4): 1e-5..1e8, 2/decade
N_GRID_VERDICT = (4_500, 10_000, 50_000, 150_000, 500_000)
EVAL_ROWS_DEFAULT = 5_000  # fixed LMSYS verdict eval slice (plan literal)
VAL_ROWS_DEFAULT = 5_000  # lambda-selection slice (plan-silent size; fits the 24,085 slack)
CURVE_SEED = 2569
MAX_WIDENINGS = 6  # widen-and-reselect bound; exceeding it is a loud failure
MEAN_ABS_DR2_PASS = 0.05  # H2b success (plan §7.5)
MEAN_ABS_DR2_KILL = 0.15  # H2b kill floor (with same-sign systematicity)
PARITY_FIELDS = (
    "layer",
    "train_corpus",
    "eval_split_sha",
    "lambda_selection",
    "train_eval_distribution",
)
LAMBDA_SELECTION_PROTOCOL = "val-selected over widened 27-value 1e-5..1e8 grid; widen-on-edge"

# pass-B n=4,500 anchor — VENDORED constants (B6: the provenance module
# scripts/issue2474_n1m_map.py::PASSB_PROVENANCE exists in zero git refs; values
# quoted from plan #2569 §4 leg 2 step 3). Off-recipe companion, never a verdict point.
PASSB_COMPANION = {
    "label": "passb_n4500_L16_gcv",
    "n_train": 4_500,
    "test_r2": 0.6293,
    "selected_lambda": 316.23,
    "lambda_grid_edge": None,
    "layer": 16,
    "corpus_mix": {"pass_b_seed_corpus": 4_500},
    "lambda_selection": "GCV",
    "train_eval_distribution": "pass-b mixed seed corpus",
    "source": "vendored from #2474 PASSB_PROVENANCE via plan #2569 §4 leg 2 (B4/B6)",
    "off_recipe_companion": True,
}


def _git_commit() -> str:
    """Best-effort short git commit for reproducibility metadata."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _meta() -> dict:
    """Reproducibility metadata block stamped into every emitted JSON."""
    return {
        "meta": {
            "issue": ISSUE,
            "script": "issue2569_gateladder",
            "git_commit": _git_commit(),
            "seed": SEED,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        }
    }


def _atomic_json(path: Path, obj) -> None:
    """Atomic JSON write (tmp + os.replace semantics via Path.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=float))
    tmp.replace(path)


# ══════════════════════════════════════════════════════════════════════════════
# Half 1 — gate-metric ladder
# ══════════════════════════════════════════════════════════════════════════════


def shrink_sigma(sigma: np.ndarray, shrinkage: float = SIGMA_SHRINKAGE) -> np.ndarray:
    """The #1979 ridge-regularization recipe: ``(1-s)*Sigma + s*(tr(Sigma)/d)*I``.

    Mirrors ``issue1768_directions.corpus_sigma`` (SHRINKAGE = 0.1) — the exact
    regularizer behind the banked whitened-gate read (A7, median rho 0.1751).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    d = sigma.shape[0]
    if sigma.shape != (d, d):
        raise ValueError(f"sigma must be square, got {sigma.shape}")
    return (1.0 - shrinkage) * sigma + shrinkage * (np.trace(sigma) / d) * np.eye(d)


def load_sigma_file(path: Path | str) -> np.ndarray:
    """Load a covariance from a P-B moments file (``.pt``/``.npz``/``.npy``).

    Accepts either ``{"sigma": (d,d)}`` directly, or the raw-moment triple
    ``{"gram": (d,d), "mean": (d,), "n_rows": int}`` (uncentered sum-of-outer
    Gram; centered here as ``gram/n - mean mean^T``). The matrix is symmetrized
    (fp round-off) and validated finite. Producer contract for unit 3's
    ``issue2569_rowbattery.py`` moments stage.
    """
    path = Path(path)
    if path.suffix in (".pt", ".pth"):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
        obj = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in dict(obj).items()}
    else:
        loaded = np.load(path, allow_pickle=False)
        obj = (
            {k: loaded[k] for k in loaded.files} if hasattr(loaded, "files") else {"sigma": loaded}
        )
    if "sigma" in obj:
        sigma = np.asarray(obj["sigma"], dtype=np.float64)
    elif {"gram", "mean", "n_rows"} <= set(obj):
        n = float(np.asarray(obj["n_rows"]).item())
        mean = np.asarray(obj["mean"], dtype=np.float64)
        sigma = np.asarray(obj["gram"], dtype=np.float64) / n - np.outer(mean, mean)
    else:
        raise ValueError(
            f"{path}: expected keys 'sigma' or ('gram','mean','n_rows'), got {sorted(obj)}"
        )
    if not np.isfinite(sigma).all():
        raise ValueError(f"{path}: covariance contains non-finite values")
    return 0.5 * (sigma + sigma.T)


def k90_count(singular_values: np.ndarray) -> int:
    """Leg-1 k90: the smallest rank whose cumulative sigma^2 mass reaches 90%.

    Thin wrapper over ``issue2569_operator.tau_kernel_threshold(mass=0.90)`` so
    the ladder and leg 1 share ONE mass-count convention.
    """
    return OP.tau_kernel_threshold(np.asarray(singular_values), mass=0.90)[1]


def _assert_through_map_probe(
    payload,
    Cc_raw: np.ndarray,
    a_raw: np.ndarray,
    img_c: np.ndarray,
    img_a: np.ndarray,
    scores_wwt: np.ndarray,
    n_probes: int = 8,
) -> dict:
    """B1 assert (iii) at the ladder: images + gate_wwt match the apply path.

    Two independently-coded forms of the registered linear part are compared on a
    probe batch: the ``prediction_difference`` images used by the ladder vs the
    ``(A, b)`` row-operator form (``context_similarity``). Raises on divergence
    (apply-path-breakage class — HALT).
    """
    m = min(n_probes, Cc_raw.shape[0])
    A, _b = OP.row_operator(payload)
    ref_img = np.atleast_2d(Cc_raw[:m]) @ A
    scale = max(float(np.abs(ref_img).max()), 1e-12)
    if not np.allclose(img_c[:m], ref_img, rtol=1e-8, atol=1e-8 * scale):
        raise AssertionError(
            "[b1-assert-iii] ladder images diverge from the row-operator form: "
            f"max abs diff {float(np.abs(img_c[:m] - ref_img).max()):.3e}"
        )
    ref_g = OP.context_similarity(A, Cc_raw[:m], a_raw[None, :])[:, 0]
    g_scale = max(float(np.abs(ref_g).max()), 1e-12)
    if not np.allclose(scores_wwt[:m], ref_g, rtol=1e-8, atol=1e-8 * g_scale):
        raise AssertionError(
            "[b1-assert-iii] gate_wwt diverges from context_similarity(A, ., .): "
            f"max abs diff {float(np.abs(scores_wwt[:m] - ref_g).max()):.3e}"
        )
    ref_a = (a_raw[None, :] @ A)[0]
    a_scale = max(float(np.abs(ref_a).max()), 1e-12)
    if not np.allclose(img_a, ref_a, rtol=1e-8, atol=1e-8 * a_scale):
        raise AssertionError("[b1-assert-iii] anchor image diverges from the row-operator form")
    return {"n_probes": int(m), "max_img_scale": scale}


def gate_scores(
    Cbar: np.ndarray,
    anchor: np.ndarray,
    payload=None,
    *,
    sigma_c: np.ndarray | None = None,
    sigma_a: np.ndarray | None = None,
    w_right_singular: np.ndarray | None = None,
    k_trunc: int | None = None,
    shrinkage: float = SIGMA_SHRINKAGE,
    probe_assert: bool = True,
) -> dict[str, np.ndarray | None]:
    """Per-prefix gate scores for every computable rung (plan §4 leg 2 step 1).

    ``Cbar`` (n_prefix, d) raw query-averaged base context means in PANEL order;
    ``anchor`` (d,) the arm's raw training-row-centroid anchor. Centering is at
    the full-panel Cbar mean (#1979's stated shared centering) applied to BOTH
    sides. Incumbent rungs are raw-space bilinear forms; through-map rungs are
    inner products of the REGISTERED map images (``prediction_difference`` vs the
    panel mean — affine terms cancel), asserted against the row-operator form on
    a probe batch when ``probe_assert`` (B1 assert iii). Rungs whose ingredients
    are absent return None (P-A partial mode: Sigma-dependent rungs wait for P-B
    moments).

    ``w_right_singular``: (d, k>=k_trunc) leading RIGHT singular vectors of W
    (output/write directions — B1 orientation), columns in descending-sigma
    order; ``k_trunc`` = the leg-1 k90 count.
    """
    Cbar = np.asarray(Cbar, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if Cbar.ndim != 2 or anchor.shape != (Cbar.shape[1],):
        raise ValueError(f"shape mismatch: Cbar {Cbar.shape}, anchor {anchor.shape}")
    mu = Cbar.mean(axis=0)
    Cc = Cbar - mu
    ac = anchor - mu
    out: dict[str, np.ndarray | None] = dict.fromkeys(GATE_METRICS)
    out["gate_I"] = Cc @ ac
    if sigma_c is not None:
        sigma_c = np.asarray(sigma_c, dtype=np.float64)
        diag = np.diag(sigma_c)
        if not (diag > 0).all():
            raise ValueError("diag(sigma_c) must be strictly positive (it divides)")
        out["gate_diag_inv"] = Cc @ (ac / diag)
        out["gate_sigma_inv"] = Cc @ np.linalg.solve(shrink_sigma(sigma_c, shrinkage), ac)
    if payload is not None:
        # Registered images: centered raw vectors through the linear part =
        # prediction differences vs the panel mean (never a re-derived product).
        img_c = OP.prediction_difference(payload, Cbar, mu)
        img_a = OP.prediction_difference(payload, anchor, mu)
        out["gate_wwt"] = img_c @ img_a
        if probe_assert:
            _assert_through_map_probe(payload, Cc, ac, img_c, img_a, out["gate_wwt"])
        if w_right_singular is not None and k_trunc is not None:
            Vk = np.asarray(w_right_singular, dtype=np.float64)[:, :k_trunc]
            proj_c = img_c @ Vk  # (n, k) image coordinates in the top-k output subspace
            proj_a = Vk.T @ img_a
            out["gate_wwt_k90"] = proj_c @ proj_a
            if probe_assert:
                m = min(8, Cc.shape[0])
                full = (img_c[:m] @ Vk) @ (Vk.T @ img_a)
                if not np.allclose(out["gate_wwt_k90"][:m], full, rtol=1e-9, atol=1e-9):
                    raise AssertionError("[b1-assert-iii] truncated-rung projection mismatch")
        if sigma_a is not None:
            sigma_a = np.asarray(sigma_a, dtype=np.float64)
            out["gate_wwt_awhite"] = img_c @ np.linalg.solve(
                shrink_sigma(sigma_a, shrinkage), img_a
            )
    return out


def _arm_seed(arm_id: str) -> int:
    """Deterministic per-arm seed — WITHIN-ARM permutation null ONLY.

    The bootstrap deliberately shares ONE stream (base SEED) across arms so the
    champion's across-arm per-draw median stays paired (the #1900/#1979 pairing
    convention).
    """
    return SEED + (zlib.crc32(arm_id.encode()) % 1_000_000)


def ladder_race(
    scores: dict[str, np.ndarray],
    dvs: np.ndarray,
    dv_names: tuple[str, ...],
    shared_ix: np.ndarray,
    *,
    arm_id: str,
    b_draws: int = B_DRAWS_DEFAULT,
    n_perm: int = N_PERM_DEFAULT,
) -> dict:
    """Within-arm race of the available rungs against the banked DVs.

    ``scores``: metric -> (n,) per-prefix gate scores on the arm's KEPT rows (the
    banked frame order); ``dvs``: (n, D) DV columns per ``dv_names``;
    ``shared_ix``: positions (into the kept rows) of the across-arm SHARED prefix
    pool — the bootstrap runs on these rows with the shared base seed so champion
    draws pair across arms (#1979 ``run_arm_battery`` verbatim). Observed rho and
    the permutation null (per-draw SIGNED max over metrics — selection rides the
    draw) run on the FULL kept rows. Reuses ``issue1900_race.{bootstrap_battery,
    perm_null, observed_rho}`` (batched rank-z GEMMs; never a per-draw loop).

    Returns the arm payload plus the raw draw matrices (persisted by the driver —
    the statistical-input-existence duty, plan §6).
    """
    import issue1900_race as R1900

    raced = [m for m in GATE_METRICS if scores.get(m) is not None]
    if not raced:
        raise ValueError(f"{arm_id}: no computable gate rungs")
    x = np.column_stack([np.asarray(scores[m], dtype=np.float64) for m in raced])
    dvs = np.asarray(dvs, dtype=np.float64)
    n = x.shape[0]
    if dvs.shape[0] != n:
        raise ValueError(f"{arm_id}: DV rows {dvs.shape[0]} != score rows {n}")
    obs = R1900.observed_rho(x, dvs)  # (K, D)
    boot, n_degen = R1900.bootstrap_battery(x[shared_ix], dvs[shared_ix], b_draws, SEED)
    perm = R1900.perm_null(x, dvs[:, 0], n_perm, _arm_seed(arm_id))  # (P, K)
    perm_max = perm.max(axis=1)  # SIGNED per-draw max — selection-symmetric band
    return {
        "arm_id": arm_id,
        "raced": raced,
        "dv_names": list(dv_names),
        "observed_rho": {
            d: {m: float(obs[i, j]) for i, m in enumerate(raced)} for j, d in enumerate(dv_names)
        },
        "perm_band": {
            "p95_max_selected": float(np.quantile(perm_max, 0.95)),
            "p975_max_selected": float(np.quantile(perm_max, 0.975)),
            "ceiling_abs_rho": 1.0,
            "n_perm": int(n_perm),
        },
        "n": int(n),
        "n_shared": int(len(shared_ix)),
        "n_degenerate_series_draws": int(n_degen),
        "boot": boot,
        "perm": perm,
    }


def ladder_champion(
    arm_results: dict[str, dict],
    *,
    dv_index: int,
    dv_label: str,
    incumbent: str = "gate_sigma_inv",
) -> dict:
    """Across-arm champion with per-draw re-selection (selection symmetry).

    Modeled on ``issue1979_race.champion`` (the registered protocol): stack the
    per-arm boot cubes (paired draw streams — same base seed + shared pool),
    take the across-arm MEDIAN per draw per metric, re-select the SIGNED-argmax
    winner inside every draw. Persists the selection-inherited CI (quantiles of
    the per-draw max over metrics) AND the frozen-at-winner CI (quantiles of the
    observed winner's per-draw median), both labeled, plus the band-vs-ceiling
    informativeness interval ``[1 - max(inc_obs), 1 - min(inc_obs)]``.
    """
    arm_ids = sorted(arm_results)
    panel = sorted(set.intersection(*[set(arm_results[a]["raced"]) for a in arm_ids]))
    if not panel:
        raise ValueError("empty shared metric panel across arms")
    cube = np.stack(
        [
            arm_results[a]["boot"][:, [arm_results[a]["raced"].index(m) for m in panel], dv_index]
            for a in arm_ids
        ]
    )  # (A, B, Kp)
    med = np.median(cube, axis=0)  # (B, Kp)
    winner_ix = np.argmax(med, axis=1)  # SIGNED argmax — the registered convention
    p_win = {m: float(np.mean(winner_ix == i)) for i, m in enumerate(panel)}
    per_arm_obs = {
        a: {
            m: arm_results[a]["observed_rho"][arm_results[a]["dv_names"][dv_index]][m]
            for m in panel
        }
        for a in arm_ids
    }
    obs_med = {m: float(np.median([per_arm_obs[a][m] for a in arm_ids])) for m in panel}
    winner = max(obs_med, key=lambda m: obs_med[m])
    sel_ci = [float(np.quantile(med.max(axis=1), q)) for q in CI_QS]
    frz_ci = [float(np.quantile(med[:, panel.index(winner)], q)) for q in CI_QS]
    inc_obs = [per_arm_obs[a][incumbent] for a in arm_ids if incumbent in per_arm_obs[a]]
    return {
        "dv": dv_label,
        "incumbent": incumbent,
        "panel_metrics": panel,
        "arm_ids": arm_ids,
        "across_arm_median_observed": obs_med,
        "winner_observed": winner,
        "p_win": p_win,
        "selection_inherited_ci_max_median": sel_ci,
        "frozen_ci_winner_median (labeled: frozen-at-winner)": frz_ci,
        "champion_vs_incumbent_conditional_ceiling_interval": (
            [float(1.0 - max(inc_obs)), float(1.0 - min(inc_obs))] if inc_obs else None
        ),
        "note_correlated_arms": "arms share one prefix panel + one banked instrument — "
        "never narrated as independent confirmations",
    }


def pairwise_win_counts(
    arm_results: dict[str, dict], dv_index: int, pairs: tuple[tuple[str, str], ...]
) -> dict:
    """Per-arm paired win counts between metric pairs (the H2 criteria reads).

    For each (a, b) pair: the count of arms where observed rho(a) > rho(b)
    strictly, plus the per-arm values (auditable).
    """
    out: dict[str, dict] = {}
    for m_a, m_b in pairs:
        rows = {}
        wins = 0
        for aid, res in sorted(arm_results.items()):
            dvn = res["dv_names"][dv_index]
            obs = res["observed_rho"][dvn]
            if m_a not in obs or m_b not in obs:
                continue
            rows[aid] = {m_a: obs[m_a], m_b: obs[m_b]}
            wins += int(obs[m_a] > obs[m_b])
        out[f"{m_a}_vs_{m_b}"] = {"wins": wins, "n_arms": len(rows), "per_arm": rows}
    return out


def per_family_win_table(
    scores_by_arm: dict[str, dict[str, np.ndarray]],
    dv_by_arm: dict[str, np.ndarray],
    families_by_arm: dict[str, list[str]],
    *,
    min_n: int = MIN_FAMILY_N,
) -> dict:
    """Per-prefix-family win table (leg-2 OOD fold, plan §6).

    For each family with >= ``min_n`` kept rows in an arm: within-arm Spearman
    per metric restricted to the family's rows; across-arm median per metric;
    winner per family. Small-n per family is inherent (~50/9 rows) — reported
    beside every cell, never hidden.
    """
    import issue1900_race as R1900

    fams = sorted({f for fam in families_by_arm.values() for f in fam})
    metrics = sorted(
        set.intersection(
            *[{m for m, v in s.items() if v is not None} for s in scores_by_arm.values()]
        )
    )
    table: dict[str, dict] = {}
    for fam in fams:
        per_arm: dict[str, dict[str, float]] = {}
        ns = []
        for aid, fam_list in families_by_arm.items():
            ix = np.flatnonzero(np.asarray(fam_list) == fam)
            if len(ix) < min_n:
                continue
            x = np.column_stack(
                [np.asarray(scores_by_arm[aid][m], dtype=np.float64)[ix] for m in metrics]
            )
            dv = np.asarray(dv_by_arm[aid], dtype=np.float64)[ix][:, None]
            obs = R1900.observed_rho(x, dv)[:, 0]
            per_arm[aid] = {m: float(obs[i]) for i, m in enumerate(metrics)}
            ns.append(int(len(ix)))
        if not per_arm:
            table[fam] = {"skipped": f"fewer than {min_n} kept rows in every arm"}
            continue
        med = {m: float(np.median([v[m] for v in per_arm.values()])) for m in metrics}
        table[fam] = {
            "n_arms": len(per_arm),
            "rows_per_arm": ns,
            "across_arm_median": med,
            "winner": max(med, key=lambda m: med[m]),
        }
    return {"families": table, "metrics": metrics, "min_rows_per_family": min_n}


# ── banked #1979 inputs ───────────────────────────────────────────────────────


def load_1979_config(config_dir: Path) -> dict:
    """Load the #1979 committed config manifests (panel, arms) — read-only."""
    panel = json.loads((config_dir / "prefix_panel.json").read_text())
    arms = json.loads((config_dir / "arms.json").read_text())["arms"]
    members = panel["members"]
    return {
        "prefix_ids": [m["prefix_id"] for m in members],
        "family": {m["prefix_id"]: m["family"] for m in members},
        "arms": arms,
    }


def load_banked_frames(race_dir: Path, arms: list[dict]) -> dict[str, dict]:
    """Load #1979's banked per-(arm, prefix) DV frames (committed JSONs).

    Observed schema (probed on ``frame_syc-bare-con-lr1e5-s42.json``): top keys
    {coverage_floor, frame, layer, meta, n_realized, pos}; ``frame`` carries
    ``prefix_id``/``family`` plus content DVs {dv_level, dv_change, dv_binary} or
    marker DVs {dv_dlogp, dv_level_logp, dv_eos_margin, dv_prob}. Returns per
    arm: prefix_ids, families, the (n, 2) primary/secondary DV matrix per
    ``DV_NAMES_BY_KIND``, and the arm row.
    """
    out: dict[str, dict] = {}
    for arm in arms:
        aid, kind = arm["arm_id"], arm["kind"]
        path = race_dir / f"frame_{aid}.json"
        payload = json.loads(path.read_text())
        frame = payload["frame"]
        dv_names = DV_NAMES_BY_KIND[kind]
        missing = [d for d in dv_names if d not in frame]
        if missing:
            raise KeyError(f"{path}: missing banked DV columns {missing}")
        dvs = np.column_stack([np.asarray(frame[d], dtype=np.float64) for d in dv_names])
        if not np.isfinite(dvs).all():
            raise ValueError(f"{path}: non-finite banked DV values")
        out[aid] = {
            "arm": arm,
            "kind": kind,
            "prefix_ids": list(frame["prefix_id"]),
            "families": list(frame["family"]),
            "dvs": dvs,
            "dv_names": dv_names,
            "banked_layer": payload.get("layer"),
            "banked_pos": payload.get("pos"),
        }
    return out


def stage_1979_inputs(inputs_root: Path, mixes: list[str]) -> None:
    """Stage the F1e ingredient tensors + mix anchors from the HF data repo.

    Narrow (skip-if-present) staging of exactly what the ladder consumes:
    ``battery/ingredient_tensors.pt`` + ``anchors/<mix>/anchors.pt`` under the
    #1979 prefix. Rides the retried atomic ``hub.stage_hub_file``.
    """
    import issue1900_judge as J

    from explore_persona_space.orchestrate import hub

    repo = J._data_repo()
    rels = ["battery/ingredient_tensors.pt"] + [f"anchors/{m}/anchors.pt" for m in sorted(mixes)]
    for rel in rels:
        dest = inputs_root / rel
        if not dest.exists():
            hub.stage_hub_file(repo, f"{HF_PREFIX_1979}/{rel}", dest, repo_type="dataset")


def load_ingredient_tensors(inputs_root: Path):
    """torch.load the F1e ingredient tensors ONCE (kept fp16; convert per key)."""
    import torch

    return torch.load(
        inputs_root / "battery/ingredient_tensors.pt", map_location="cpu", weights_only=False
    )


def cbar_from_tensors(tensors, arm_id: str, layer: int, pos: str) -> np.ndarray:
    """Extract one arm's per-prefix base context means (panel order, fp64)."""
    key = f"{arm_id}/L{layer}/{pos}/Cbar"
    if key not in tensors:
        raise KeyError(f"ingredient tensors missing {key}")
    return np.asarray(tensors[key].double().numpy(), dtype=np.float64)


def anchor_from_file(inputs_root: Path, mix: str, layer: int, pos: str) -> np.ndarray:
    """Load one mix's training-row-centroid context anchor (fp64)."""
    import torch

    anc = torch.load(
        inputs_root / "anchors" / mix / "anchors.pt", map_location="cpu", weights_only=False
    )
    return np.asarray(anc[f"L{layer}"][ANCHOR_KEY_BY_POS[pos]].double().numpy(), dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Half 2 — B4 H2b learning curve
# ══════════════════════════════════════════════════════════════════════════════


def widen_grid(
    grid: np.ndarray, side: str, decades: float = 2.0, per_decade: int = 2
) -> np.ndarray:
    """Extend a log-spaced lambda grid by ``decades`` on ``side`` ('low'|'high').

    Keeps the 2-per-decade spacing of the registered 27-value grid; the
    widen-and-reselect loop re-fits on the FULL extended grid (C4: a boundary
    selection is never reported).
    """
    grid = np.sort(np.asarray(grid, dtype=np.float64))
    n_new = int(decades * per_decade)
    step = 1.0 / per_decade
    if side == "low":
        lo = np.log10(grid[0])
        new = 10.0 ** np.arange(lo - decades, lo - step / 2, step)
        return np.concatenate([new, grid])
    if side == "high":
        hi = np.log10(grid[-1])
        new = 10.0 ** np.arange(hi + step, hi + decades + step / 2, step)
        assert len(new) == n_new
        return np.concatenate([grid, new])
    raise ValueError(f"side must be 'low' or 'high', got {side!r}")


def nested_lmsys_splits(
    corpus: np.ndarray,
    conv_index: np.ndarray,
    *,
    n_grid: tuple[int, ...] = N_GRID_VERDICT,
    eval_rows: int = EVAL_ROWS_DEFAULT,
    val_rows: int = VAL_ROWS_DEFAULT,
    seed: int = CURVE_SEED,
    lmsys_tag: str = "lmsys",
) -> dict:
    """Build the B4 verdict splits: fixed LMSYS eval/val slices + nested train subsets.

    Conversations (not rows) are shuffled with ``default_rng(seed)``; the first
    conversations fill the EVAL slice (truncated to exactly ``eval_rows`` — the
    boundary conversation's overflow rows are DROPPED, preserving disjointness),
    the next fill the VAL slice the lambda selection uses (constant across all
    five points, so the selection procedure is parity-constant), and every
    remaining LMSYS row forms the training pool. Nested subsets = the first n of
    ONE row permutation of the pool (subset(n_small) is a prefix of
    subset(n_large)). Conversation-index disjointness of eval/val vs the pool is
    asserted, not assumed.
    """
    corpus = np.asarray(corpus)
    conv_index = np.asarray(conv_index)
    if corpus.shape != conv_index.shape:
        raise ValueError("corpus and conv_index must align")
    lmsys_rows = np.flatnonzero(np.char.lower(corpus.astype(str)) == lmsys_tag)
    if lmsys_rows.size == 0:
        raise ValueError(f"no rows tagged {lmsys_tag!r}")
    rng = np.random.default_rng(seed)
    convs = np.unique(conv_index[lmsys_rows])
    convs = convs[rng.permutation(len(convs))]
    rows_by_conv = {}
    order = np.argsort(conv_index[lmsys_rows], kind="stable")
    sorted_rows = lmsys_rows[order]
    sorted_convs = conv_index[sorted_rows]
    bounds = np.searchsorted(sorted_convs, convs, side="left")
    bounds_hi = np.searchsorted(sorted_convs, convs, side="right")
    for c, lo, hi in zip(convs, bounds, bounds_hi, strict=True):
        rows_by_conv[int(c)] = sorted_rows[lo:hi]

    def take(target: int, start: int) -> tuple[np.ndarray, int]:
        rows: list[np.ndarray] = []
        got, i = 0, start
        while got < target:
            if i >= len(convs):
                raise ValueError(f"not enough LMSYS conversations for a {target}-row slice")
            r = rows_by_conv[int(convs[i])]
            rows.append(r)
            got += len(r)
            i += 1
        return np.concatenate(rows)[:target], i

    te_idx, next_i = take(eval_rows, 0)
    val_idx, next_i = take(val_rows, next_i)
    pool = (
        np.concatenate([rows_by_conv[int(c)] for c in convs[next_i:]])
        if next_i < len(convs)
        else np.array([], dtype=np.int64)
    )
    if max(n_grid) > pool.size:
        raise ValueError(f"training pool {pool.size} rows < max n_grid {max(n_grid)}")
    pool = pool[rng.permutation(pool.size)]
    tr_by_n = {int(n): np.sort(pool[:n]) for n in n_grid}
    held = set(conv_index[te_idx]) | set(conv_index[val_idx])
    if held & set(conv_index[pool]):
        raise AssertionError("eval/val conversations leak into the training pool")

    def sha(ix: np.ndarray) -> str:
        return hashlib.sha256(np.sort(np.asarray(ix, dtype=np.int64)).tobytes()).hexdigest()[:16]

    return {
        "tr_by_n": tr_by_n,
        "val_idx": np.sort(val_idx),
        "te_idx": np.sort(te_idx),
        "pool_rows": int(pool.size),
        "n_lmsys_rows": int(lmsys_rows.size),
        "eval_split_sha": f"te:{sha(te_idx)}|val:{sha(val_idx)}",
        "seed": int(seed),
        "lmsys_tag": lmsys_tag,
    }


def _load_fit_core():
    """Deferred import of the reused fit core (torch chain, ~20 s; #823-safe).

    ``issue779_ffc_n50k_fits.fit_ridge_primal`` (POSITIONAL ``tr, val, te``
    split-index arrays; returns ``(pred_te, meta)`` — it is NOT in the n1m
    module) + ``issue779_percontext_recon._pooled_r2`` (the registered pooled-R2
    read: SS_tot on the eval set's own mean).
    """
    import issue779_ffc_n50k_fits as N50K
    import issue779_percontext_recon as PR

    return N50K.fit_ridge_primal, PR._pooled_r2


def identity_bias_r2(X, Y, tr: np.ndarray, te: np.ndarray, pooled_r2, chunk: int = 65_536) -> float:
    """Identity+learned-bias baseline R2, streaming form of the canonical helper.

    ``b = train-mean(Y - X)`` computed in fp64 chunks (the exact
    ``analysis/mapping_baselines.identity_bias_predict`` bias — asserted equal in
    the unit tests); pred = X[te] + b. Streaming avoids the helper's full fp64
    materialization at n_train = 500,000. Requires ``d_in == d_out`` (callers
    state inapplicability otherwise — the §6 mapping-baselines rule).
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"identity+bias needs d_in == d_out, got {X.shape[1]} vs {Y.shape[1]}")
    d = X.shape[1]
    s = np.zeros(d, dtype=np.float64)
    for lo in range(0, len(tr), chunk):
        ix = tr[lo : lo + chunk]
        s += np.asarray(Y[ix], dtype=np.float64).sum(0) - np.asarray(X[ix], dtype=np.float64).sum(0)
    b = s / len(tr)
    pred = np.asarray(X[te], dtype=np.float64) + b
    return float(pooled_r2(pred, np.asarray(Y[te], dtype=np.float64)))


def fit_point(
    X,
    Y,
    tr: np.ndarray,
    val: np.ndarray,
    te: np.ndarray,
    *,
    dev=None,
    grid: tuple[float, ...] = LAMBDA_GRID_27,
    max_widenings: int = MAX_WIDENINGS,
    knn_ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """One verdict refit: val-lambda-selected primal ridge + widen-on-edge (C4).

    Calls the reused ``fit_ridge_primal(X, Y, tr, val, te, lambdas, dev)`` with
    explicit index arrays, all seven args positional. A ``lambda_grid_edge``
    selection widens the grid two decades on that side and re-fits; after
    ``max_widenings`` the point FAILS LOUD (never reports an edge value).
    Reports test pooled R2, the identity+bias baseline, and kNN retrieval
    (chance stated) per the mapping-baselines rule (§6, B4).
    """
    import torch

    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    fit_ridge_primal, pooled_r2 = _load_fit_core()
    if dev is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = np.asarray(tr, dtype=np.int64)
    val = np.asarray(val, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)
    g = np.sort(np.asarray(grid, dtype=np.float64))
    n_widen = 0
    while True:
        pred_te, meta = fit_ridge_primal(X, Y, tr, val, te, list(g), dev)
        edge = meta.get("lambda_grid_edge")
        if edge is None:
            break
        if n_widen >= max_widenings:
            raise RuntimeError(
                f"lambda still at the {edge} grid edge after {max_widenings} widenings "
                f"(grid [{g[0]:.1e}, {g[-1]:.1e}]) — refusing to report an edge value (C4)"
            )
        g = widen_grid(g, edge)
        n_widen += 1
        logger.info(
            "[curve] n=%d lambda at %s edge -> widened grid to [%.1e, %.1e]",
            len(tr),
            edge,
            g[0],
            g[-1],
        )
    y_te = np.asarray(Y[te], dtype=np.float64)
    return {
        "n_train": int(len(tr)),
        "test_r2": float(pooled_r2(pred_te, y_te)),
        "selected_lambda": float(meta["selected_lambda"]),
        "val_r2_at_selected": float(meta["val_r2_at_selected"]),
        "lambda_grid_edge": None,
        "n_widenings": int(n_widen),
        "final_grid": [float(g[0]), float(g[-1]), int(len(g))],
        "identity_bias_r2": (
            identity_bias_r2(X, Y, tr, te, pooled_r2)
            if X.shape[1] == Y.shape[1]
            else "inapplicable (d_in != d_out) — stated per the mapping-baselines rule"
        ),
        "knn_retrieval": knn_retrieval(pred_te, y_te, ks=knn_ks),
    }


def pooled_moments(X, Y, rows: np.ndarray, *, chunk: int = 65_536, dev=None) -> dict:
    """Measured moments of the LMSYS subset in the ESTIMATOR's coordinates.

    Two streaming passes over ``rows``: (1) per-dim mean/sd of X and mean of Y;
    (2) fp64 Grams of STANDARDIZED X vs itself and vs centered Y, plus centered
    Y total variance — matching ``fit_ridge_primal``'s standardize-X / center-Y
    convention so the theory models the estimator actually run. Returns the
    eigen-spectrum ``eta`` of Sigma_c, per-mode target powers
    ``p_i = ||v_i^T Sigma_xy||^2 / eta_i``, the residual noise variance, and the
    total centered target variance.
    """
    import torch

    if dev is None:
        dev = torch.device("cpu")
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    d = X.shape[1]
    sx = np.zeros(d, dtype=np.float64)
    sx2 = np.zeros(d, dtype=np.float64)
    sy = np.zeros(Y.shape[1], dtype=np.float64)
    for lo in range(0, n, chunk):
        ix = rows[lo : lo + chunk]
        xb = np.asarray(X[ix], dtype=np.float64)
        sx += xb.sum(0)
        sx2 += (xb**2).sum(0)
        sy += np.asarray(Y[ix], dtype=np.float64).sum(0)
    xmu = sx / n
    xsd = np.sqrt(np.maximum(sx2 / n - xmu**2, 0.0)) + 1e-9
    ymu = sy / n
    gxx = torch.zeros((d, d), dtype=torch.float64, device=dev)
    gxy = torch.zeros((d, Y.shape[1]), dtype=torch.float64, device=dev)
    yss = 0.0
    xmu_t = torch.as_tensor(xmu, dtype=torch.float64, device=dev)
    xsd_t = torch.as_tensor(xsd, dtype=torch.float64, device=dev)
    ymu_t = torch.as_tensor(ymu, dtype=torch.float64, device=dev)
    for lo in range(0, n, chunk):
        ix = rows[lo : lo + chunk]
        xb = (torch.as_tensor(np.asarray(X[ix]), dtype=torch.float64, device=dev) - xmu_t) / xsd_t
        yb = torch.as_tensor(np.asarray(Y[ix]), dtype=torch.float64, device=dev) - ymu_t
        gxx += xb.T @ xb
        gxy += xb.T @ yb
        yss += float((yb**2).sum().item())
    sigma_xx = (gxx / n).cpu().numpy()
    sigma_xy = (gxy / n).cpu().numpy()
    total_var = yss / n
    eta, V = np.linalg.eigh(0.5 * (sigma_xx + sigma_xx.T))
    eta = np.maximum(eta[::-1], 0.0)
    V = V[:, ::-1]
    proj = V.T @ sigma_xy  # (d, D): rows = v_i^T Sigma_xy
    with np.errstate(divide="ignore", invalid="ignore"):
        p_mode = np.where(eta > 1e-12, (proj**2).sum(axis=1) / np.maximum(eta, 1e-12), 0.0)
    noise_var = max(total_var - float(p_mode.sum()), 0.0)
    return {
        "eta": eta,
        "p_mode": p_mode,
        "noise_var": noise_var,
        "total_var": float(total_var),
        "n_rows": int(n),
        "linear_r2_population": float(p_mode.sum() / total_var) if total_var > 0 else float("nan"),
    }


def kappa_self_consistent(lam: float, n: int, eta: np.ndarray) -> float:
    """Effective regularization kappa (arXiv 2006.13198): 1-d root find per n.

    Solves ``kappa = lam + sum_i kappa*eta_i / (kappa + n*eta_i)`` by brentq on
    the bracket ``[lam, lam + sum(eta)]`` (sign change proven in the unit tests'
    isotropic closed form).
    """
    from scipy.optimize import brentq

    eta = np.asarray(eta, dtype=np.float64)
    lam = float(lam)
    if lam <= 0:
        raise ValueError("theory kappa needs lam > 0 (the registered grid is positive)")

    def f(kappa: float) -> float:
        return kappa - lam - float(np.sum(kappa * eta / (kappa + n * eta)))

    hi = lam + float(eta.sum()) + 1e-12
    if f(lam) >= 0.0:
        return lam
    return float(brentq(f, lam, hi, xtol=1e-14, rtol=1e-13))


def theory_r2(
    n: int, lam: float, eta: np.ndarray, p_mode: np.ndarray, noise_var: float, total_var: float
) -> dict:
    """Closed-form ridge learning-curve prediction (arXiv 2006.13198).

    ``E_g = kappa^2/(1-gamma) * sum_i p_i/(kappa + n eta_i)^2
    + noise * gamma/(1-gamma)``; predicted test MSE = ``E_g + noise``; predicted
    pooled R2 = ``1 - MSE/total_var``. Validated in the unit tests against the
    empirical estimator on synthetic Gaussian data (both isotropic closed form
    and spiked-spectrum Monte Carlo).
    """
    eta = np.asarray(eta, dtype=np.float64)
    p_mode = np.asarray(p_mode, dtype=np.float64)
    kappa = kappa_self_consistent(lam, n, eta)
    denom = kappa + n * eta
    gamma = float(np.sum(n * eta**2 / denom**2))
    if gamma >= 1.0:
        raise ArithmeticError(f"gamma = {gamma:.6f} >= 1 (n={n}, lam={lam:.3e})")
    eg = (kappa**2 / (1.0 - gamma)) * float(np.sum(p_mode / denom**2))
    eg += noise_var * gamma / (1.0 - gamma)
    mse = eg + noise_var
    return {
        "kappa": float(kappa),
        "gamma": float(gamma),
        "excess_risk": float(eg),
        "predicted_mse": float(mse),
        "predicted_r2": float(1.0 - mse / total_var),
    }


def fit_metadata_parity_check(points: list[dict], reference: dict | None = None) -> dict:
    """Mechanical fit-metadata parity check (B4) — runs BEFORE any theory read.

    Every point must match the reference on the five registered fields
    (``PARITY_FIELDS``: layer, training-corpus composition, eval-split shas,
    lambda-selection procedure, train-vs-eval distribution). A mismatch EXCLUDES
    the point from mean |dR2| and is NAMED field-by-field. Reference defaults to
    the first point.
    """
    if not points:
        raise ValueError("no points to parity-check")
    ref = reference if reference is not None else {k: points[0].get(k) for k in PARITY_FIELDS}
    per_point = []
    for p in points:
        mism = [k for k in PARITY_FIELDS if p.get(k) != ref.get(k)]
        per_point.append(
            {
                "label": p.get("label", f"n{p.get('n_train')}"),
                "pass": not mism,
                "mismatched_fields": mism,
            }
        )
    return {"reference": {k: ref.get(k) for k in PARITY_FIELDS}, "per_point": per_point}


def mean_abs_delta_r2(points: list[dict]) -> dict:
    """H2b statistic: mean |empirical - theory| R2 over parity-passing verdict points.

    Off-recipe companions never enter (callers pass verdict points only); the
    verdict bands are the plan §7.5 registrations (pass <= 0.05; localized misfit
    0.05-0.15; kill > 0.15 with a same-sign systematic pattern).
    """
    deltas = [float(p["test_r2"] - p["theory"]["predicted_r2"]) for p in points]
    if not deltas:
        return {"mean_abs_dr2": None, "verdict": "no-parity-passing-points"}
    mean_abs = float(np.mean(np.abs(deltas)))
    same_sign = bool(all(d > 0 for d in deltas) or all(d < 0 for d in deltas))
    if mean_abs <= MEAN_ABS_DR2_PASS:
        verdict = "h2b-pass"
    elif mean_abs <= MEAN_ABS_DR2_KILL:
        verdict = "localized-misfit (no kill)"
    else:
        verdict = "h2b-kill-candidate" if same_sign else "large-misfit-not-systematic"
    return {
        "mean_abs_dr2": mean_abs,
        "per_point_delta_r2": deltas,
        "same_sign_all": same_sign,
        "verdict": verdict,
        "bands": {"pass_le": MEAN_ABS_DR2_PASS, "kill_gt": MEAN_ABS_DR2_KILL},
    }


def load_companion_points(repo_root: Path = REPO_ROOT) -> list[dict]:
    """The committed heterogeneous points, labeled off-recipe companions (B4).

    Values are READ FROM the committed artifacts at call time (never re-derived,
    never memory): the pass-B n=3,600 L19 point
    (``eval_results/issue_779/fitter-fair-comparison/fair_comparison.json``), the
    n50k-plan-b point (``.../fitter-fair-comparison-n50k/n50k_fits.json``), and
    the four n1m points (``.../fitter-fair-comparison-n1m/n1m_fits.json``), plus
    the vendored pass-B n=4,500 L16/GCV anchor (``PASSB_COMPANION``). Each
    carries corpus mix, realized n, selected lambda, and ``lambda_grid_edge``.
    """
    er = repo_root / "eval_results/issue_779"
    out: list[dict] = [dict(PASSB_COMPANION)]
    fair = json.loads((er / "fitter-fair-comparison/fair_comparison.json").read_text())
    ridge_last = fair["inputs"]["last"]["ridge"]  # observed schema: inputs.last.ridge.*
    out.append(
        {
            "label": "committed_n3600_L19",
            "n_train": 3_600,
            "test_r2": float(ridge_last["test_r2_at_val_selected_layer"]),
            "selected_lambda": 1000.0,  # plan §4 leg 2 (committed series, quoted)
            "lambda_grid_edge": None,
            "layer": int(ridge_last["val_selected_layer"]),
            "corpus_mix": {"pass_b_seed_corpus": 3_600},
            "lambda_selection": "val-selected (committed original grid)",
            "train_eval_distribution": "pass-b mixed seed corpus",
            "source": "eval_results/issue_779/fitter-fair-comparison/fair_comparison.json",
            "off_recipe_companion": True,
        }
    )
    n50k = json.loads((er / "fitter-fair-comparison-n50k/n50k_fits.json").read_text())
    r = n50k["per_predictor"]["ridge"]
    out.append(
        {
            "label": "committed_n50k_plan_b_L19",
            "n_train": int(r["fit_meta"]["n_train"]),
            "test_r2": float(r["whole_map_r2"]),
            "selected_lambda": float(r["fit_meta"]["selected_lambda"]),
            "lambda_grid_edge": r["fit_meta"].get("lambda_grid_edge"),
            "layer": int(n50k["layer"]),
            "corpus_mix": {
                "pass_b_seed_corpus": int(n50k["split"]["orig_train_ids"]),
                "n50k_new": int(n50k["split"]["n50k_used"]),
            },
            "lambda_selection": "val-selected (committed n50k grid)",
            "train_eval_distribution": "mixed pool -> pass-b eval rows",
            "source": "eval_results/issue_779/fitter-fair-comparison-n50k/n50k_fits.json",
            "off_recipe_companion": True,
        }
    )
    n1m = json.loads((er / "fitter-fair-comparison-n1m/n1m_fits.json").read_text())
    for key, pt in sorted(n1m["per_point"].items()):
        rr = pt["predictors"]["ridge"]
        sel = pt["selection"]
        out.append(
            {
                "label": f"committed_{key}_L19",
                "n_train": int(rr["fit_meta"]["n_train"]),
                "test_r2": float(rr["whole_map_r2"]),
                "selected_lambda": float(rr["fit_meta"]["selected_lambda"]),
                "lambda_grid_edge": rr["fit_meta"].get("lambda_grid_edge"),
                "layer": int(n1m["layer"]),
                "corpus_mix": {"lmsys": int(sel["n_lmsys"]), "wildchat": int(sel["n_wildchat"])},
                "lambda_selection": "val-selected (committed n1m streaming grid)",
                "train_eval_distribution": "store pool -> pass-b eval rows",
                "source": "eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json",
                "off_recipe_companion": True,
            }
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Drivers
# ══════════════════════════════════════════════════════════════════════════════


def run_ladder(args) -> int:
    """Ladder driver: banked frames + F1e tensors -> gate_ladder.json (+ npz)."""

    import issue1768_directions as DIR

    assert abs(DIR.SHRINKAGE - SIGMA_SHRINKAGE) < 1e-12, (
        f"issue1768_directions.SHRINKAGE={DIR.SHRINKAGE} != vendored {SIGMA_SHRINKAGE}"
    )
    payload = OP.load_banked_map(args.layer, root=args.map_root)
    entry_asserts = OP.run_driver_identity_asserts(payload)  # HALT class on raise (B1)
    cfg = load_1979_config(Path(args.config_dir))
    frames = load_banked_frames(Path(args.race_dir), cfg["arms"])
    mixes = sorted({f["arm"]["mix_arm_id"] for f in frames.values()})
    inputs_root = Path(args.inputs_root)
    stage_1979_inputs(inputs_root, mixes)
    tensors = load_ingredient_tensors(inputs_root)
    sigma_c = load_sigma_file(args.sigma_c) if args.sigma_c else None
    sigma_a = load_sigma_file(args.sigma_a) if args.sigma_a else None
    # Full SVD of W once (pod venue: minutes; k90 needs the full spectrum).
    logger.info("[ladder] full SVD of W (%d^2) ...", payload.d)
    _u, s_full, vh = np.linalg.svd(payload.W, full_matrices=False)
    k_trunc = int(args.k90) if args.k90 else k90_count(s_full)
    v_right = vh.T  # (d, d) right singular vectors = OUTPUT/write directions (B1)
    b_draws = 100 if args.smoke else args.b_draws
    n_perm = 100 if args.smoke else args.n_perm
    panel_ix = {p: i for i, p in enumerate(cfg["prefix_ids"])}

    scores_by_arm: dict[str, dict[str, np.ndarray]] = {}
    dv_primary_by_arm: dict[str, np.ndarray] = {}
    families_by_arm: dict[str, list[str]] = {}
    results: dict[str, dict] = {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shared_by_kind: dict[str, list[str]] = {}
    for kind in ("content", "marker"):
        pools = [set(f["prefix_ids"]) for f in frames.values() if f["kind"] == kind]
        shared_by_kind[kind] = sorted(set.intersection(*pools)) if pools else []

    for aid, fr in sorted(frames.items()):
        arm = fr["arm"]
        Cbar = cbar_from_tensors(tensors, aid, args.layer, args.pos)
        anchor = anchor_from_file(inputs_root, arm["mix_arm_id"], args.layer, args.pos)
        sc_full = gate_scores(
            Cbar,
            anchor,
            payload,
            sigma_c=sigma_c,
            sigma_a=sigma_a,
            w_right_singular=v_right,
            k_trunc=k_trunc,
        )
        kept = [panel_ix[p] for p in fr["prefix_ids"]]
        sc = {m: (v[kept] if v is not None else None) for m, v in sc_full.items()}
        shared = shared_by_kind[fr["kind"]]
        pid_pos = {p: i for i, p in enumerate(fr["prefix_ids"])}
        shared_ix = np.asarray([pid_pos[p] for p in shared], dtype=np.int64)
        res = ladder_race(
            {m: v for m, v in sc.items() if v is not None},
            fr["dvs"],
            fr["dv_names"],
            shared_ix,
            arm_id=aid,
            b_draws=b_draws,
            n_perm=n_perm,
        )
        shared_sha = hashlib.sha256("\n".join(shared).encode()).hexdigest()[:16]
        np.savez(
            out_dir / f"ladder_boot_{aid}.npz",
            rho=res["boot"],
            candidates=np.array(res["raced"]),
            dv_names=np.array(res["dv_names"]),
            seed=SEED,
            n=res["n"],
            n_shared=res["n_shared"],
            shared_sha_hash=np.array(shared_sha),
        )
        np.savez(
            out_dir / f"ladder_perm_{aid}.npz",
            rho=res["perm"],
            max_selected=res["perm"].max(axis=1),
            candidates=np.array(res["raced"]),
            dv="primary",
            seed=_arm_seed(aid),
        )
        res["kind"] = fr["kind"]
        results[aid] = res
        scores_by_arm[aid] = {m: v for m, v in sc.items() if v is not None}
        dv_primary_by_arm[aid] = fr["dvs"][:, 0]
        families_by_arm[aid] = fr["families"]
        logger.info("[ladder] arm %s raced (%d rungs)", aid, len(res["raced"]))

    partial = any(
        results[a]["raced"] != list(GATE_METRICS) for a in results
    )  # Sigma rungs absent => P-A partial mode
    champion: dict = {}
    pairwise: dict = {}
    for kind in ("content", "marker"):
        sub = {a: r for a, r in results.items() if r["kind"] == kind}
        if not sub:
            continue
        champion[kind] = {}
        for j, dvn in enumerate(DV_NAMES_BY_KIND[kind]):
            inc = "gate_sigma_inv" if not partial else "gate_I"
            champion[kind][dvn] = ladder_champion(sub, dv_index=j, dv_label=dvn, incumbent=inc)
        pairs = (("gate_wwt", "gate_sigma_inv"), ("gate_wwt", "gate_I"))
        pairs = tuple(
            (a, b)
            for a, b in pairs
            if all(a in r["raced"] and b in r["raced"] for r in sub.values())
        )
        pairwise[kind] = pairwise_win_counts(sub, 0, pairs) if pairs else {}
    family_table = per_family_win_table(scores_by_arm, dv_primary_by_arm, families_by_arm)

    # H2 verdict fields (content arms; anchors READ FROM the committed artifacts).
    anchors_ctx: dict = {}
    verdicts = Path(args.race_dir) / "battery_verdicts.json"
    champ_change = Path(args.race_dir) / "champion_change.json"
    if verdicts.exists():
        anchors_ctx["whitened_banked_median_rho"] = json.loads(verdicts.read_text())["A7"][
            "median_rho"
        ]
    if champ_change.exists():
        cc = json.loads(champ_change.read_text())["prefix_resample_PRIMARY"]
        anchors_ctx["p3b_champion_median_rho"] = cc["across_arm_median_observed"]["p3b"]
    h2: dict = {"note": "content arms, dv_change primary (plan §3 H2 / §7.5 leg 2)"}
    content = {a: r for a, r in results.items() if r["kind"] == "content"}
    if content and not partial:
        pw = pairwise.get("content", {})
        wwt_med = float(
            np.median([r["observed_rho"]["dv_change"]["gate_wwt"] for r in content.values()])
        )
        h2.update(
            wwt_vs_whitened_wins=pw.get("gate_wwt_vs_gate_sigma_inv", {}).get("wins"),
            wwt_vs_identity_wins=pw.get("gate_wwt_vs_gate_I", {}).get("wins"),
            n_content_arms=len(content),
            success_criterion="wins_vs_whitened >= 7/12 AND wwt_median_rho >= 0.9 x champion",
            kill_criterion="wins_vs_identity <= 5/12",
            wwt_median_rho=wwt_med,
            ratio_vs_p3b_champion=(
                wwt_med / anchors_ctx["p3b_champion_median_rho"]
                if "p3b_champion_median_rho" in anchors_ctx
                else None
            ),
        )
    out = {
        **_meta(),
        "regime": {
            "layer": int(args.layer),
            "pos": args.pos,
            "b_draws": int(b_draws),
            "n_perm": int(n_perm),
            "seed": SEED,
            "k_trunc": int(k_trunc),
            "sigma_shrinkage": SIGMA_SHRINKAGE,
            "sigma_c_source": str(args.sigma_c) if args.sigma_c else None,
            "sigma_a_source": str(args.sigma_a) if args.sigma_a else None,
            "partial": bool(partial),
            "smoke": bool(args.smoke),
            "marker_layer_note": (
                "marker arms scored at the map layer L19 (banked marker-primary was L25; "
                "the ladder pins the map's input space)"
            ),
            "centering": "full-panel Cbar mean, both sides (#1979 candidate_table convention)",
        },
        "entry_asserts": entry_asserts,
        "anchors_context": anchors_ctx,
        "per_arm": {
            a: {k: v for k, v in r.items() if k not in ("boot", "perm")} for a, r in results.items()
        },
        "champion": champion,
        "pairwise_win_counts": pairwise,
        "per_family_win_table": family_table,
        "h2": h2,
    }
    name = "gate_ladder_partial.json" if partial else "gate_ladder.json"
    _atomic_json(out_dir / name, out)
    logger.info("[ladder] wrote %s", out_dir / name)
    print(f"[phase=ladder_done] out={out_dir / name}", flush=True)
    return 0


def _load_array(path: Path):
    """Load an activation matrix: ``.npy`` memmapped, ``.pt`` via torch."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    import torch

    t = torch.load(path, map_location="cpu", weights_only=False)
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def _load_row_meta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load per-row (corpus, conv_index) from a ``.pt``/``.npz`` metadata file."""
    path = Path(path)
    if path.suffix in (".pt", ".pth"):
        import torch

        obj = dict(torch.load(path, map_location="cpu", weights_only=False))
    else:
        loaded = np.load(path, allow_pickle=True)
        obj = {k: loaded[k] for k in loaded.files}
    corpus = np.asarray(obj["corpus"]).astype(str)
    conv = np.asarray(obj["conv_index"], dtype=np.int64)
    return corpus, conv


def run_curve(args) -> int:
    """Learning-curve driver: assembled store -> learning_curve.json (B4)."""
    import torch

    X = _load_array(Path(args.x))
    Y = _load_array(Path(args.y))
    corpus, conv = _load_row_meta(Path(args.row_meta))
    if not (X.shape[0] == Y.shape[0] == corpus.shape[0]):
        raise ValueError(f"row mismatch: X {X.shape}, Y {Y.shape}, meta {corpus.shape}")
    n_grid = tuple(int(v) for v in args.n_grid.split(",")) if args.n_grid else N_GRID_VERDICT
    eval_rows = int(args.eval_rows)
    val_rows = int(args.val_rows)
    dev = torch.device(args.dev) if args.dev else None
    splits = nested_lmsys_splits(
        corpus,
        conv,
        n_grid=n_grid,
        eval_rows=eval_rows,
        val_rows=val_rows,
        seed=CURVE_SEED,
        lmsys_tag=args.lmsys_tag,
    )
    logger.info(
        "[curve] lmsys rows=%d pool=%d eval=%d val=%d",
        splits["n_lmsys_rows"],
        splits["pool_rows"],
        len(splits["te_idx"]),
        len(splits["val_idx"]),
    )
    # Theory moments on the SAME LMSYS subset the series draws from (train pool).
    moments = pooled_moments(X, Y, splits["tr_by_n"][max(n_grid)], dev=dev)
    verdict_points: list[dict] = []
    for n in n_grid:
        tr = splits["tr_by_n"][n]
        pt = fit_point(X, Y, tr, splits["val_idx"], splits["te_idx"], dev=dev)
        pt.update(
            label=f"verdict_n{n}",
            layer=int(args.layer),
            train_corpus="lmsys",
            eval_split_sha=splits["eval_split_sha"],
            lambda_selection=LAMBDA_SELECTION_PROTOCOL,
            train_eval_distribution="lmsys->lmsys",
            corpus_mix={"lmsys": int(len(tr))},
            theory=theory_r2(
                len(tr),
                pt["selected_lambda"],
                moments["eta"],
                moments["p_mode"],
                moments["noise_var"],
                moments["total_var"],
            ),
        )
        verdict_points.append(pt)
        logger.info(
            "[curve] n=%d r2=%.4f theory=%.4f lambda=%.3g",
            n,
            pt["test_r2"],
            pt["theory"]["predicted_r2"],
            pt["selected_lambda"],
        )
    parity = fit_metadata_parity_check(verdict_points)
    passing = [p for p, pp in zip(verdict_points, parity["per_point"], strict=True) if pp["pass"]]
    excluded = [pp for pp in parity["per_point"] if not pp["pass"]]
    h2b = mean_abs_delta_r2(passing)
    companions = [] if args.skip_companions else load_companion_points(REPO_ROOT)
    companion_parity = (
        fit_metadata_parity_check(companions, reference=parity["reference"]) if companions else None
    )
    out = {
        **_meta(),
        "regime": {
            "n_grid": list(n_grid),
            "eval_rows": eval_rows,
            "val_rows": val_rows,
            "seed": CURVE_SEED,
            "layer": int(args.layer),
            "lambda_grid": [float(LAMBDA_GRID_27[0]), float(LAMBDA_GRID_27[-1]), 27],
            "lambda_selection": LAMBDA_SELECTION_PROTOCOL,
            "lmsys_tag": args.lmsys_tag,
            "smoke": bool(args.smoke),
        },
        "splits": {k: v for k, v in splits.items() if k not in ("tr_by_n", "val_idx", "te_idx")},
        "moments": {
            "n_rows": moments["n_rows"],
            "total_var": moments["total_var"],
            "noise_var": moments["noise_var"],
            "linear_r2_population": moments["linear_r2_population"],
            "eta_top8": [float(v) for v in moments["eta"][:8]],
            "eta_sum": float(moments["eta"].sum()),
        },
        "verdict_points": verdict_points,
        "parity_check": parity,
        "parity_excluded": excluded,
        "h2b": h2b,
        "companions_off_recipe": companions,
        "companion_parity_vs_verdict_reference": companion_parity,
    }
    out_dir = Path(args.out)
    _atomic_json(out_dir / "learning_curve.json", out)
    logger.info("[curve] wrote %s", out_dir / "learning_curve.json")
    print(f"[phase=curve_done] out={out_dir / 'learning_curve.json'}", flush=True)
    return 0


def _run_import_check() -> int:
    """Resolve every deferred import + argparse-attribute completeness (argcheck)."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    import issue1768_directions  # noqa: F401
    import issue1900_judge  # noqa: F401
    import issue1900_race  # noqa: F401

    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.orchestrate import hub  # noqa: F401

    fit_ridge_primal, pooled_r2 = _load_fit_core()
    assert callable(fit_ridge_primal) and callable(pooled_r2)
    from scipy.optimize import brentq  # noqa: F401

    print("[import-check] OK: all deferred imports resolve; argcheck clean", flush=True)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    """CLI: ``ladder`` / ``curve`` subcommands + ``--import-check``."""
    ap = argparse.ArgumentParser(
        prog="issue2569_gateladder",
        description="#2569 leg 2: gate-metric ladder + closed-form ridge learning curve",
    )
    ap.add_argument("--import-check", action="store_true", help="resolve deferred imports and exit")
    sub = ap.add_subparsers(dest="cmd")
    lad = sub.add_parser("ladder", help="gate-metric ladder race (plan §4 leg 2 steps 1-2)")
    lad.add_argument("--out", required=True, help="output dir (eval_results/issue_2569/leg2/)")
    lad.add_argument("--inputs-root", required=True, help="#1979 F1e staging root")
    lad.add_argument("--config-dir", default=str(REPO_ROOT / "eval_results/issue_1979/config"))
    lad.add_argument("--race-dir", default=str(REPO_ROOT / "eval_results/issue_1979/race"))
    lad.add_argument("--map-root", default=None, help="banked-map root (Unit 1 precedence)")
    lad.add_argument("--sigma-c", default=None, help="P-B context moments file (.pt/.npz)")
    lad.add_argument("--sigma-a", default=None, help="P-B answer moments file (.pt/.npz)")
    lad.add_argument("--k90", type=int, default=None, help="override the leg-1 k90 count")
    lad.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    lad.add_argument("--pos", default=POS_DEFAULT, choices=sorted(ANCHOR_KEY_BY_POS))
    lad.add_argument("--b-draws", type=int, default=B_DRAWS_DEFAULT)
    lad.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    lad.add_argument("--smoke", action="store_true", help="100 draws / 100 perms, same chain")
    cur = sub.add_parser("curve", help="B4 H2b learning curve (plan §4 leg 2 step 3)")
    cur.add_argument("--x", required=True, help="X19 matrix (.npy memmap or .pt)")
    cur.add_argument("--y", required=True, help="Y19 matrix (.npy memmap or .pt)")
    cur.add_argument("--row-meta", required=True, help="{corpus, conv_index} file (.pt/.npz)")
    cur.add_argument("--out", required=True, help="output dir (eval_results/issue_2569/leg2/)")
    cur.add_argument("--dev", default=None, help="torch device (default: cuda if available)")
    cur.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    cur.add_argument("--n-grid", default=None, help="comma ints (default: the verdict grid)")
    cur.add_argument("--eval-rows", type=int, default=EVAL_ROWS_DEFAULT)
    cur.add_argument("--val-rows", type=int, default=VAL_ROWS_DEFAULT)
    cur.add_argument("--lmsys-tag", default="lmsys")
    cur.add_argument("--skip-companions", action="store_true", help="omit committed companions")
    cur.add_argument("--smoke", action="store_true", help="tag the output as a smoke run")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point: dispatch to the ladder / curve drivers (or --import-check)."""
    args = build_argparser().parse_args(argv)
    if args.import_check:
        return _run_import_check()
    if args.cmd == "ladder":
        return run_ladder(args)
    if args.cmd == "curve":
        return run_curve(args)
    build_argparser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
