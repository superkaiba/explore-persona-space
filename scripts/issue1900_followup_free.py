"""#1900 9a-ter free-analysis follow-ups (VM/CPU, ANALYSIS-ONLY).

Consumes the COMMITTED #1900 P1-P3 artifacts (predictor tables, marker
three-space parquets, judge scores, race arm JSONs + boot npz) and produces:

A. Coupling-calibrated change race (`coupling_calibrated.json`) — per
   (arm, candidate) observed change-DV Spearman rho against an explicit
   trained-vs-base independence (coupling) null: permute the TRAINED
   per-context scores against base (breaking the pairing, keeping both
   marginals), change_pi = trained[perm] - base, 500 draws/arm seed 7 (the
   interpretation-v2 recompute convention, now durable). Reports observed,
   null mean, null 2.5/97.5 percentiles, observed - null mean.
   Plus `coupling_null_artifact.json` — the P7-coupling independence null
   itself per arm (content arms additionally under the interpretation's
   judge-join convention + the closed-form Pearson null).

B. P7-residualized change race (`residualized_race.json`) — per arm,
   OLS-residualize the trained per-context score on the base per-context
   score entering the change DV (1-D fit + intercept, full realized frame),
   then Spearman(candidate, residual) for every raced candidate; champion
   read via the race's OWN machinery (`champion_read` over per-arm boot npz
   written here: B=2000, module seed 1900, family shared-sha pool — the draw
   index streams are IDENTICAL to the registered race's because
   `bootstrap_battery` consumes the rng as a pure function of
   (seed, n_shared, chunking), verified via the stored `shared_sha_hash`).

C. 11-arm arm-exclusion sensitivity (`sensitivity_11arm.json`) — champion
   re-reads (level primary + change companion) excluding the below-
   Criterion-B content arm (imp-pers-con-lr3e5-s42, share_ge10 0.0460 <
   0.05); pure re-aggregation of committed race/arm_*.json + boot_*.npz.

Outputs -> eval_results/issue_1900/race/followup_free/ (+ 2 heatmap figures
mirroring hero_content_race.png style). No training / generation / judge
calls anywhere in this module.
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
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1900_race as R  # noqa: E402  (loaders + batched rank machinery)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1900.followup_free")

N_COUPLING = 500  # interpretation-v2 recompute convention
COUPLING_SEED = 7  # interpretation-v2 recompute convention
BELOW_BAR_ARM = "imp-pers-con-lr3e5-s42"  # share_ge10 0.0460 < 0.05 (Criterion B)
CHANGE_DV = {"content": "dv_change", "marker": "dv_dlogp"}


def _meta(**extra) -> dict:
    m = R._meta()
    m["script"] = "scripts/issue1900_followup_free.py"
    m.update(extra)
    return m


# ── shared per-arm term extraction ───────────────────────────────────────────


def change_terms(asm: dict) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """(x (n,K), raced, base_term, trained, change) for one assembled arm.

    The base_term is the exact -B component of the arm's change DV: the base
    per-context graded mean (== the p7 candidate) for content; the base
    logP(marker) at the ARM slot for marker (b_lp = dv_level_logp - dv_dlogp
    algebraically — the marker p7 CANDIDATE is bb_lp at the BASE slot and is
    NOT the base term, so it stays a genuine candidate in every read here).
    """
    frame, raced = asm["frame"], asm["raced"]
    x = np.column_stack(
        [
            frame["p7"].to_numpy(float) if c == "p7" else frame[R.CANDIDATE_COLS[c]].to_numpy(float)
            for c in raced
        ]
    )
    if asm["arm"]["kind"] == "content":
        base = frame["p7"].to_numpy(float)
        trained = frame["dv_level"].to_numpy(float)
        change = frame["dv_change"].to_numpy(float)
    else:
        trained = frame["dv_level_logp"].to_numpy(float)
        change = frame["dv_dlogp"].to_numpy(float)
        base = trained - change  # b_lp, exactly (same maps built the columns)
    return x, raced, base, trained, change


# ── Deliverable A: coupling null ─────────────────────────────────────────────


def coupling_null_battery(
    x_cands: np.ndarray, base: np.ndarray, trained: np.ndarray, n_draws: int, seed: int
) -> np.ndarray:
    """(D, K) Spearman rho of every candidate vs permuted-coupling change draws.

    Null: fine-tuning re-draws per-context behavior independently of base —
    permute trained against base (both marginals kept), change_pi =
    trained[perm] - base. Batched: rank-z rows once + one GEMM over all
    draws (reuses the race's `_rank_z_t` searchsorted-midrank primitive);
    no per-draw Python loop.
    """
    import torch

    n, _k = x_cands.shape
    zc, _ = R._rank_z_t(torch.from_numpy(x_cands.astype(np.float32)).T.contiguous())  # (K, n)
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_draws, n)), axis=1)
    ch = trained[perm] - base[None, :]  # (D, n)
    zd, _ = R._rank_z_t(torch.from_numpy(ch.astype(np.float32)))
    return ((zd @ zc.T) / n).numpy()


def closed_form_pearson_null(p7_cand: np.ndarray, base: np.ndarray, trained: np.ndarray) -> dict:
    """E[Pearson(p7, perm(T) - B)] ~= -r(p7, B) * sigma_B / sqrt(sigma_B^2 + sigma_T^2).

    Ratio-of-expectations approximation (the interpretation-v2 convention);
    for content arms p7 IS the base term (r = 1), reducing to the quoted
    -sigma_B / sqrt(sigma_B^2 + sigma_T^2).
    """
    r = float(np.corrcoef(p7_cand, base)[0, 1])
    sb, st = float(np.std(base)), float(np.std(trained))
    return {
        "r_p7_baseterm": r,
        "sigma_base": sb,
        "sigma_trained": st,
        "value": float(-r * sb / math.sqrt(sb**2 + st**2)),
    }


def judge_join_content(arm: dict, judge_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """(base, trained, n) per-context graded means joined on sha, kept>0 both sides.

    The interpretation-v2 coupling-null convention: judge-score join ONLY
    (n ~= 3,959-3,996/arm) — no predictor-table listwise drop.
    """
    a_mean, _, _ = R._score_maps(R.load_scores(judge_dir, arm["arm_id"]))
    b_mean, _, _ = R._score_maps(R.load_scores(judge_dir, f"base_{arm['beh_key']}"))
    shas = sorted(set(a_mean) & set(b_mean))
    base = np.array([b_mean[s] for s in shas], dtype=float)
    trained = np.array([a_mean[s] for s in shas], dtype=float)
    return base, trained, len(shas)


# ── Deliverable A: residualization ───────────────────────────────────────────


def residualize_ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """OLS y ~ 1 + x on the full frame; returns (residual, fit stats)."""
    xmat = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    resid = y - xmat @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return resid, {"alpha": float(beta[0]), "beta": float(beta[1]), "r2_1d_fit": r2}


# ── assembly + shared-pool verification ──────────────────────────────────────


def assemble_all(
    arms: list[dict], tables_dir: Path, marker_dir: Path, judge_dir: Path
) -> tuple[list[dict], list[dict], dict]:
    content_asm, marker_asm = [], []
    for arm in arms:
        asm = (
            R.assemble_content_arm(arm, tables_dir, judge_dir)
            if arm["kind"] == "content"
            else R.assemble_marker_arm(arm, tables_dir, marker_dir)
        )
        (content_asm if arm["kind"] == "content" else marker_asm).append(asm)
    shared_by_kind = {}
    if content_asm:
        shared_by_kind["content"] = R._family_shared_shas(content_asm)
    if marker_asm:
        shared_by_kind["marker"] = R._family_shared_shas(marker_asm)
    return content_asm, marker_asm, shared_by_kind


def verify_stream_alignment(race_dir: Path, asms: list[dict], shared: tuple) -> dict:
    """Compare our family shared-sha hash with the committed race boot npz.

    A match means `bootstrap_battery(seed=1900)` here regenerates the EXACT
    per-draw index streams the registered race used (rng consumption is a
    pure function of seed + n_shared + chunking) — sha-paired draw reuse.
    """
    npz = np.load(race_dir / f"boot_{asms[0]['arm']['arm_id']}.npz", allow_pickle=False)
    stored = str(npz["shared_sha_hash"])
    ours = shared[1]
    ok = stored == ours
    if not ok:
        logger.warning(
            "shared-sha hash mismatch (stored %s vs ours %s): residualized boot uses "
            "FRESH seed-1900 draws over OUR pool, not the race's exact streams",
            stored,
            ours,
        )
    return {
        "stored_shared_sha_hash": stored,
        "recomputed_shared_sha_hash": ours,
        "draw_streams_identical_to_race": ok,
        "n_shared": len(shared[0]),
    }


# ── Deliverable A driver ─────────────────────────────────────────────────────


def run_coupling(
    asms: list[dict], race_dir: Path, out_dir: Path, judge_dir: Path, n_draws: int, seed: int
) -> None:
    per_arm, null_artifact, xcheck_max = {}, {}, 0.0
    for asm in asms:
        arm_id = asm["arm"]["arm_id"]
        kind = asm["arm"]["kind"]
        x, raced, base, trained, change = change_terms(asm)
        obs = R.observed_rho(x, change[:, None])[:, 0]
        committed = json.loads((race_dir / f"arm_{arm_id}.json").read_text())
        comm_rho = committed["observed_rho"][CHANGE_DV[kind]]
        diffs = [abs(float(obs[i]) - comm_rho[c]) for i, c in enumerate(raced) if c in comm_rho]
        xcheck_max = max(xcheck_max, max(diffs))
        assert max(diffs) < 1e-3, (arm_id, max(diffs), "change-rho mismatch vs committed arm JSON")
        null = coupling_null_battery(x, base, trained, n_draws, seed)  # (D, K)
        rows = {}
        for i, c in enumerate(raced):
            nm = float(null[:, i].mean())
            lo, hi = (float(np.quantile(null[:, i], q)) for q in R.CI_QS)
            rows[c] = {
                "observed_change_rho": float(obs[i]),
                "null_mean": nm,
                "null_p025": lo,
                "null_p975": hi,
                "calibrated_minus_null_mean": float(obs[i]) - nm,
                "above_null_band": bool(obs[i] > hi),
            }
        per_arm[arm_id] = {"kind": kind, "n": asm["n_realized"], "candidates": rows}
        art = {
            "kind": kind,
            "join": "race listwise frame",
            "n": asm["n_realized"],
            "observed_p7_coupling": rows["p7"]["observed_change_rho"],
            "null_mean": rows["p7"]["null_mean"],
            "null_p025": rows["p7"]["null_p025"],
            "null_p975": rows["p7"]["null_p975"],
            "closed_form_pearson": closed_form_pearson_null(x[:, raced.index("p7")], base, trained),
        }
        if kind == "content":
            jb, jt, jn = judge_join_content(asm["arm"], judge_dir)
            jnull = coupling_null_battery(jb[:, None], jb, jt, n_draws, seed)[:, 0]
            art["judge_join"] = {
                "convention": "interpretation-v2: sha join kept>0 both sides, no table drop",
                "n": jn,
                "observed_p7_coupling": R._spearman_np(jb, jt - jb),
                "null_mean": float(jnull.mean()),
                "null_p025": float(np.quantile(jnull, R.CI_QS[0])),
                "null_p975": float(np.quantile(jnull, R.CI_QS[1])),
                "closed_form_pearson": closed_form_pearson_null(jb, jb, jt),
            }
        null_artifact[arm_id] = art
    recipe = (
        f"coupling null: permute trained against base per arm ({n_draws} draws, seed {seed}), "
        "change_pi = trained[perm] - base, batched rank-z GEMM (race _rank_z_t primitive); "
        "observed change-rho cross-checked vs committed race/arm_*.json"
    )
    R._atomic_json(
        out_dir / "coupling_calibrated.json",
        {
            "meta": _meta(coupling_seed=seed, n_coupling_draws=n_draws, recipe=recipe),
            "xcheck_max_abs_diff_vs_committed": xcheck_max,
            "per_arm": per_arm,
        },
    )
    R._atomic_json(
        out_dir / "coupling_null_artifact.json",
        {
            "meta": _meta(
                coupling_seed=seed,
                n_coupling_draws=n_draws,
                recipe="durable P7-coupling independence null (interpretation-v2 Finding 2 "
                "recompute): frame-join for all 18 arms + judge-join for the 12 content arms; "
                "closed form -r(p7,B)*sigma_B/sqrt(sigma_B^2+sigma_T^2)",
            ),
            "per_arm": null_artifact,
        },
    )


# ── Deliverable B driver (residualized race) ─────────────────────────────────


def run_residualized(
    asms: list[dict],
    shared: tuple,
    out_dir: Path,
    b_draws: int,
    label: str,
    alignment: dict,
) -> dict:
    shared_shas, shared_hash = shared
    ids, ols_stats = [], {}
    for asm in asms:
        arm_id = asm["arm"]["arm_id"]
        ids.append(arm_id)
        x, raced, base, trained, _change = change_terms(asm)
        resid, fit = residualize_ols(trained, base)
        ols_stats[arm_id] = fit
        obs = R.observed_rho(x, resid[:, None])
        frame = asm["frame"]
        sha_pos = {s: i for i, s in enumerate(frame["sha"])}
        pos = np.array([sha_pos[s] for s in shared_shas], dtype=np.int64)
        t0 = time.time()
        boot, n_degen = R.bootstrap_battery(x[pos], resid[pos, None], b_draws, R.SEED)
        np.savez(
            out_dir / f"boot_{arm_id}.npz",
            rho=boot,
            candidates=np.array(raced),
            dv_names=np.array(["dv_resid"]),
            seed=R.SEED,
            n=asm["n_realized"],
            n_shared=len(shared_shas),
            shared_sha_hash=np.array(shared_hash),
        )
        R._atomic_json(
            out_dir / f"arm_{arm_id}.json",
            {
                "meta": _meta(),
                "arm_id": arm_id,
                "kind": asm["arm"]["kind"],
                "beh_key": asm["arm"]["beh_key"],
                "regime": {
                    "b_draws": b_draws,
                    "layer": asm["layer"],
                    "raced": raced,
                    "dv_names": ["dv_resid"],
                    "n": asm["n_realized"],
                    "n_shared": len(shared_shas),
                    "shared_sha_hash": shared_hash,
                },
                "observed_rho": {"dv_resid": {c: float(obs[i, 0]) for i, c in enumerate(raced)}},
                "ols_residualization": fit,
                "n_degenerate_series_draws": int(n_degen),
                "n_realized": asm["n_realized"],
                "elapsed_s": round(time.time() - t0, 1),
            },
        )
        logger.info("[resid] %s boot done in %.1fs", arm_id, time.time() - t0)
    champ = R.champion_read(ids, out_dir, 0, label)
    champ["ols_residualization_per_arm"] = ols_stats
    champ["boot_draw_alignment"] = alignment
    return champ


# ── Deliverable C driver (11-arm sensitivity) ────────────────────────────────


def run_sensitivity(content_ids: list[str], race_dir: Path, judge_dir: Path, out_dir: Path) -> None:
    assert BELOW_BAR_ARM in content_ids, (BELOW_BAR_ARM, "below-bar arm not in loaded set")
    ids_11 = [a for a in content_ids if a != BELOW_BAR_ARM]
    lvl_11 = R.champion_read(ids_11, race_dir, 0, "content graded LEVEL — 11-arm sensitivity")
    chg_11 = R.champion_read(
        ids_11, race_dir, 1, "content graded CHANGE companion — 11-arm sensitivity"
    )
    committed = json.loads((race_dir / "champion_content.json").read_text())
    deltas = {
        which: {
            c: lvl_chg["across_arm_median_observed"][c]
            - committed[key]["across_arm_median_observed"][c]
            for c in lvl_chg["across_arm_median_observed"]
        }
        for which, key, lvl_chg in (
            ("level", "primary", lvl_11),
            ("change", "change_companion", chg_11),
        )
    }
    share = json.loads((judge_dir / f"arm_scores_{BELOW_BAR_ARM}.json").read_text())["share_ge10"]
    R._atomic_json(
        out_dir / "sensitivity_11arm.json",
        {
            "meta": _meta(
                recipe="pure re-aggregation of committed race/arm_*.json + boot_*.npz "
                "excluding the below-Criterion-B arm; champion_read machinery unchanged"
            ),
            "excluded_arm": {
                "arm_id": BELOW_BAR_ARM,
                "share_ge10_full_data": share,
                "criterion": "registered Criterion B share_ge10 >= 0.05 (kept in the race "
                "by recorded Gate-1 Decision; this is the exclusion sensitivity)",
            },
            "level_11arm": lvl_11,
            "change_11arm": chg_11,
            "median_deltas_vs_12arm": deltas,
            "verdict_12arm": {
                "level": committed["primary"]["verdict"],
                "change": committed["change_companion"]["verdict"],
            },
        },
    )


# ── figures ──────────────────────────────────────────────────────────────────


def render_figures(out_dir: Path, fig_dir: Path) -> list[Path]:
    import issue1900_figs as F

    payloads = [json.loads(p.read_text()) for p in sorted(out_dir.glob("arm_*.json"))]
    made = []
    for kind, stem, title in (
        (
            "content",
            "residualized_race_content",
            "Propensity-residualized change race — content arms",
        ),
        (
            "marker",
            "residualized_race_marker",
            "Propensity-residualized change race — marker arms",
        ),
    ):
        sub = [p for p in payloads if p["kind"] == kind]
        if sub:
            made.append(F._heatmap(sub, "dv_resid", stem, fig_dir, title))
    return made


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-dir", type=Path, default=REPO_ROOT / "data/issue_1900/config")
    ap.add_argument("--p1-root", type=Path, default=REPO_ROOT / "data/issue_1900/out")
    ap.add_argument("--judge-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1900/judge")
    ap.add_argument("--race-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1900/race")
    ap.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1900/race/followup_free"
    )
    ap.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures/issue_1900")
    ap.add_argument("--b-draws", type=int, default=R.B_BOOT)
    ap.add_argument("--n-coupling", type=int, default=N_COUPLING)
    ap.add_argument("--coupling-seed", type=int, default=COUPLING_SEED)
    ap.add_argument("--arms", default=None, help="comma-separated arm_id subset (smoke)")
    ap.add_argument("--skip-figs", action="store_true")
    args = ap.parse_args()

    arms = R.J.load_arms(args.config_dir)
    if args.arms:
        keep = {s.strip() for s in args.arms.split(",")}
        arms = [a for a in arms if a["arm_id"] in keep]
        assert arms, f"--arms matched nothing: {args.arms}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = args.p1_root / "predictor_tables"
    marker_dir = args.p1_root / "marker_tf"
    content_asm, marker_asm, shared_by_kind = assemble_all(
        arms, tables_dir, marker_dir, args.judge_dir
    )
    logger.info("[assemble] %d content + %d marker arms", len(content_asm), len(marker_asm))

    # Deliverable A: coupling-calibrated change race + durable coupling null.
    run_coupling(
        content_asm + marker_asm,
        args.race_dir,
        args.out_dir,
        args.judge_dir,
        args.n_coupling,
        args.coupling_seed,
    )
    print("[followup] coupling_calibrated.json + coupling_null_artifact.json done", flush=True)

    # Deliverable A: P7-residualized change race (champion machinery reused).
    resid_out = {"meta": _meta(b_draws=args.b_draws)}
    for kind, asms, label in (
        ("content", content_asm, "content P7-residualized change (follow-up 1 deciding read)"),
        ("marker", marker_asm, "marker P7-residualized change (follow-up 1 deciding read)"),
    ):
        if not asms:
            continue
        alignment = verify_stream_alignment(args.race_dir, asms, shared_by_kind[kind])
        resid_out[kind] = run_residualized(
            asms, shared_by_kind[kind], args.out_dir, args.b_draws, label, alignment
        )
    resid_out["recipe"] = (
        "per arm: resid = trained - OLS(trained ~ 1 + base_term) on the FULL realized frame "
        "(fixed instrument, mirroring dv_change's fixed-column treatment); Spearman(candidate, "
        "resid) observed; bootstrap = race bootstrap_battery (B, seed 1900, family shared-sha "
        "pool) -> champion_read winner-per-draw"
    )
    R._atomic_json(args.out_dir / "residualized_race.json", resid_out)
    print("[followup] residualized_race.json done", flush=True)

    # Deliverable B: 11-arm exclusion sensitivity (full content panel only).
    content_ids = [a["arm"]["arm_id"] for a in content_asm]
    if len(content_ids) == 12 and BELOW_BAR_ARM in content_ids:
        run_sensitivity(content_ids, args.race_dir, args.judge_dir, args.out_dir)
        print("[followup] sensitivity_11arm.json done", flush=True)
    else:
        logger.info("[sensitivity] skipped: content panel is %d arms (need 12)", len(content_ids))

    if not args.skip_figs:
        for p in render_figures(args.out_dir, args.fig_dir):
            print(f"[followup] figure {p}", flush=True)
    print("[followup] done", flush=True)


if __name__ == "__main__":
    main()
