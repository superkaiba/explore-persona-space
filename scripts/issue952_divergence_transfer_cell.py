"""#952 free-analysis follow-up: the MISSING divergence-bank cross transfer cell.

The committed #952 H3 bank read is ARM-MATCHED: each arm's map (context c_last ->
that arm's answer slots, fit on the train pool) is scored on that arm's OWN bank
answer targets, and the headline is d = drop_ext_plain - drop_own where
drop_arm = R2_control(arm) - R2_divergent(arm) (penalty -0.005, p=0.64, n=41
kept pairs). This computes the MISSING CROSS cell (the #952 analog of #823's
own->plain transfer): the OWN-answer-fitted map's prediction scored against the
PLAIN-EXTERNAL arm's bank answer targets (own map x Claude answers), divergent
vs entity-swapped control, over the SAME 41 kept pairs.

The map input is the SHARED context slot (own c_last; arms differ only in the
answer, cross-arm c_last cosine >=0.99 per run_952), so "own map" = a ridge fit
to reconstruct OWN answer slots from context, and the cross scores that
own-answer prediction against the ext_plain answer target (a genuinely
off-diagonal read the committed run never computed).

Machinery replicated VERBATIM from run_952.phase_battery + issue952_stats.h3_reads:
  - universe U_A = span>=32 in ALL arms; split = make_split rng(952) 2952/984/984.
  - solver = ridge_battery.run_ridge_cell (shared-SVD GCV ridge), frozen at the
    committed per-slot lambda (npz A_lam_idx).
  - per bank query R2 = 1 - sum_slot ss_res / sum_slot ss_tot over the 42
    POSITION_SLOTS (identical to _bank_per_context_r2).
  - per-pair drop = R2_control(qc) - R2_divergent(qd); pooled/median/10%-trimmed
    bootstrap CI (seed 0); 10k-draw sign-flip null over pair signs (seed 1).

Gates (run BEFORE trusting the cross):
  Gate 1 (stats machinery): recompute the committed ARM-MATCHED h3 from the
    committed npz + verification; must match stats_summary.json h3.
  Gate 2 (fit reproduction): re-fit the arm-matched maps on the reproduced U_A
    universe and match the committed npz bank ss (per-pair drops) within tol.

Bank content rule: bank items referenced by query_id only; NO bank text touched.

Usage:
  uv run python scripts/issue952_divergence_transfer_cell.py
"""

from __future__ import annotations

# ruff: noqa: E402 — load_dotenv() must run before torch import (shared-VM thread caps)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import datetime
import json
import logging
import pathlib
import subprocess
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue952_divergence_transfer_cell")

DL = pathlib.Path(
    "/mnt/eps-data/thomasjiralerspong/tmp_issue952_crosscell/"
    "issue952_position_divergence/analysis_tensors"
)
ARMS = ("own", "ext_plain", "ext_style", "mismatch")
BANK_ARMS = ("own", "ext_plain")
DECILES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
POSITION_SLOTS = (
    [f"f16_t{t}" for t in range(1, 17)]
    + [f"l16_m{k}" for k in range(1, 17)]
    + [f"d10_p{round(d * 100)}" for d in DECILES]
)
N_DRAWS = 10000
BOOTSTRAP_SEED = 0
SIGNFLIP_SEED = 1
H3_MARGIN = 0.05
REPRO_TOL = 0.01  # per-pair drop match tol (CPU-vs-committed-fit_device BLAS drift)


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _build_slot_names() -> list[str]:
    slots = ["c_last"]
    slots += [f"f16_t{t}" for t in range(1, 17)]
    slots += [f"l16_m{k}" for k in range(1, 17)]
    slots += [f"d10_p{round(d * 100)}" for d in DECILES]
    slots += [f"z_t{t}" for t in (32, 64, 128)]
    # remaining registry slots are not needed for POSITION_SLOTS / c_last indexing;
    # only c_last (idx 0) and POSITION_SLOTS are used below, all in the first 43.
    return slots


SLOT_NAMES = _build_slot_names()
SLOT_IDX = {n: i for i, n in enumerate(SLOT_NAMES)}


def _load_slots(tag: str) -> tuple[np.ndarray, list]:
    d = torch.load(str(DL / f"slots_{tag}_L20.pt"), map_location="cpu", weights_only=False)
    return d["slots"].numpy(), list(d["ids"])


def _stack_answer_targets(
    slots_by_arm: dict, rows: np.ndarray, arms: tuple[str, ...]
) -> np.ndarray:
    groups = [(s, a) for s in POSITION_SLOTS for a in arms]
    out = np.full((len(rows), len(groups), 3584), np.nan, dtype=np.float16)
    for gi, (slot, arm) in enumerate(groups):
        out[:, gi, :] = slots_by_arm[arm][rows][:, SLOT_IDX[slot], :]
    return out, [f"{s}|{a}" for s, a in groups]


def _per_query_r2(ssr: np.ndarray, sst: np.ndarray, group_names: list[str], arm: str) -> np.ndarray:
    """Per-query R2 = 1 - sum_slot ss_res / sum_slot ss_tot over POSITION_SLOTS for `arm`.
    ssr/sst: (n_query, G). Returns (n_query,)."""
    cols = [
        gi
        for gi, g in enumerate(group_names)
        if g.endswith(f"|{arm}") and g.split("|")[0] in POSITION_SLOTS
    ]
    out = np.full(ssr.shape[0], np.nan)
    for ri in range(ssr.shape[0]):
        sr, st = ssr[ri, cols], sst[ri, cols]
        fin = np.isfinite(sr) & np.isfinite(st)
        denom = st[fin].sum()
        if denom > 1e-12:
            out[ri] = 1.0 - sr[fin].sum() / denom
    return out


def _bank_boot(vals: np.ndarray, n_draws: int, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    m = len(vals)
    idx = rng.integers(0, m, size=(n_draws, m))
    res = vals[idx]
    mean_d = res.mean(axis=1)
    med_d = np.median(res, axis=1)
    k = max(1, round(0.1 * m))
    srt = np.sort(res, axis=1)
    trim_d = srt[:, k : m - k].mean(axis=1) if m - 2 * k >= 1 else mean_d
    srt0 = np.sort(vals)
    return {
        "n": m,
        "mean": float(vals.mean()),
        "mean_ci95": [float(np.percentile(mean_d, 2.5)), float(np.percentile(mean_d, 97.5))],
        "median": float(np.median(vals)),
        "median_ci95": [float(np.percentile(med_d, 2.5)), float(np.percentile(med_d, 97.5))],
        "trimmed10_mean": float(srt0[k : m - k].mean()) if m - 2 * k >= 1 else None,
        "trimmed10_ci95": (
            [float(np.percentile(trim_d, 2.5)), float(np.percentile(trim_d, 97.5))]
            if m - 2 * k >= 1
            else None
        ),
    }


def _signflip_p(d: np.ndarray, n_draws: int, seed: int = SIGNFLIP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(d)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_draws, n))
    obs = float(d.mean())
    null = (signs @ d) / n
    p_one = float((1 + int((null >= obs).sum())) / (1 + n_draws))
    return {
        "observed_mean_d": obs,
        "p_one_sided": p_one,
        "null_band_hi_97p5": float(np.percentile(null, 97.5)),
    }


def _kept_pairs(verification: dict) -> list[dict]:
    kept = set(verification["kept_pairs"])
    return [p for p in verification["pairs"] if p["pair_id"] in kept]


def _committed_arm_matched(npz: dict, pairs: list[dict]) -> dict:
    """Gate 1: recompute committed h3 arm-matched from the committed npz."""
    groups = npz["A_group_names"].tolist()
    div_ids = {str(q): i for i, q in enumerate(npz["bank_div_ids"].tolist())}
    ctl_ids = {str(q): i for i, q in enumerate(npz["bank_ctl_ids"].tolist())}
    r2 = {}
    for key, idmap in (("bank_div", div_ids), ("bank_ctl", ctl_ids)):
        ssr = npz[f"{key}_ssres"].astype(np.float64)
        sst = npz[f"{key}_sstot"].astype(np.float64)
        r2[key] = {arm: _per_query_r2(ssr, sst, groups, arm) for arm in BANK_ARMS}
        r2[key]["_idmap"] = idmap
    rows = []
    for p in pairs:
        qd, qc = p["divergent"]["query_id"], p["control"]["query_id"]
        di, ci = r2["bank_div"]["_idmap"].get(qd), r2["bank_ctl"]["_idmap"].get(qc)
        if di is None or ci is None:
            continue
        rec = {"pair_id": p["pair_id"], "category": p["category"]}
        ok = True
        for arm in BANK_ARMS:
            rd, rc = r2["bank_div"][arm][di], r2["bank_ctl"][arm][ci]
            if not (np.isfinite(rd) and np.isfinite(rc)):
                ok = False
            rec[f"drop_{arm}"] = rc - rd
        rec["d"] = rec["drop_ext_plain"] - rec["drop_own"]
        if ok:
            rows.append(rec)
    d = np.asarray([r["d"] for r in rows])
    return {
        "n_pairs": len(rows),
        "headline_d": _bank_boot(d, N_DRAWS),
        "sign_flip_pooled": _signflip_p(d, N_DRAWS),
        "mean_drop_own": float(np.mean([r["drop_own"] for r in rows])),
        "mean_drop_ext_plain": float(np.mean([r["drop_ext_plain"] for r in rows])),
        "rows": rows,
    }


def main() -> None:
    torch.set_num_threads(8)
    from explore_persona_space.experiments.issue_952.ridge_battery import run_ridge_cell

    base = pathlib.Path(__file__).resolve().parent.parent
    out_dir = base / "eval_results" / "issue_952" / "divergence_transfer_cell"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── universe / split reproduction ────────────────────────────────────────
    own_pool, pool_ids = _load_slots("own")
    plain_pool, _ = _load_slots("ext_plain")
    split = json.loads((base / "eval_results/issue_952/split_seed952.json").read_text())
    spans = {
        a: {str(k): v for k, v in json.loads((DL / f"spans_{a}.json").read_text()).items()}
        for a in ARMS
    }
    spans_arr = {
        a: np.asarray([spans[a][str(c)].get("span", 0) for c in pool_ids], dtype=np.int64)
        for a in ARMS
    }
    u_a = np.all(np.stack([spans_arr[a] >= 32 for a in ARMS]), axis=0)
    pos_of = {c: i for i, c in enumerate(pool_ids)}
    tr_pos = np.asarray([pos_of[c] for c in split["train"] if c in pos_of])
    tr_a = tr_pos[u_a[tr_pos]]
    logger.info("U_A=%d tr_a=%d", int(u_a.sum()), len(tr_a))

    npz = dict(np.load(str(DL / "per_context_stats.npz"), allow_pickle=False))
    group2lam = dict(zip(npz["A_group_names"].tolist(), npz["A_lam_idx"].tolist(), strict=True))

    verification = json.loads(
        (base / "eval_results/issue_952/divergence_bank_verification.json").read_text()
    )
    pairs = _kept_pairs(verification)
    committed_stats = json.loads((base / "eval_results/issue_952/stats_summary.json").read_text())[
        "h3"
    ]

    # ── Gate 1: stats machinery vs committed h3 ──────────────────────────────
    gate1 = _committed_arm_matched(npz, pairs)
    g1_d = gate1["headline_d"]["mean"]
    g1_p = gate1["sign_flip_pooled"]["p_one_sided"]
    c_d = committed_stats["headline_mean_drop_diff"]["mean"]
    c_p = committed_stats["sign_flip_null"]["pooled"]["p_one_sided"]
    gate1_pass = abs(g1_d - c_d) < 1e-6 and gate1["n_pairs"] == committed_stats["n_pairs"]
    logger.info(
        "[Gate1] recomputed d=%.6f (committed %.6f) p=%.4f (committed %.4f) n=%d pass=%s",
        g1_d,
        c_d,
        g1_p,
        c_p,
        gate1["n_pairs"],
        gate1_pass,
    )

    # ── bank slot tensors + role splits (replicating run_952 pass-2) ─────────
    bank = {arm: _load_slots(f"bank_{arm}") for arm in BANK_ARMS}
    bank_ids = bank["own"][1]
    assert bank_ids == bank["ext_plain"][1], "bank id order differs across arms"
    div_rows = [i for i, q in enumerate(bank_ids) if q.endswith("_div")]
    ctl_rows = [i for i, q in enumerate(bank_ids) if not q.endswith("_div")]
    div_id2row = {bank_ids[i]: pos for pos, i in enumerate(div_rows)}
    ctl_id2row = {bank_ids[i]: pos for pos, i in enumerate(ctl_rows)}

    own_bank = bank["own"][0]
    plain_bank = bank["ext_plain"][0]
    slots_by_arm = {"own": own_pool, "ext_plain": plain_pool}
    bank_by_arm = {"own": own_bank, "ext_plain": plain_bank}
    c_last_tr = own_pool[tr_a][:, SLOT_IDX["c_last"], :].astype(np.float64)

    def _extract_frozen(res, split_name, gnames):
        ssr = res.ss_res[split_name]  # (n, G, L)
        lam_idx = np.asarray([group2lam[g] for g in gnames], dtype=np.int64)
        take = np.take_along_axis(ssr, lam_idx[None, :, None], axis=2)[:, :, 0]
        return take.astype(np.float64), res.ss_tot[split_name].astype(np.float64)

    def _run(y_train_arms, bank_target_arm, gnames_arm_for_lam):
        """Fit map on (c_last -> y_train_arms answer), score bank against bank_target_arm.
        For the arm-matched Gate 2, y_train_arms == bank_target arms (list). For the
        cross, y_train_arms = one fitted arm, bank_target_arm = the OTHER arm."""
        Ytr, gnames = _stack_answer_targets(slots_by_arm, tr_a, y_train_arms)
        # bank eval targets scored against bank_target_arm (per group, same slot order)
        div_tgt, _ = _stack_answer_targets(
            {a: bank_by_arm[bank_target_arm] for a in y_train_arms},
            np.asarray(div_rows),
            y_train_arms,
        )
        ctl_tgt, _ = _stack_answer_targets(
            {a: bank_by_arm[bank_target_arm] for a in y_train_arms},
            np.asarray(ctl_rows),
            y_train_arms,
        )
        xb_div = own_bank[div_rows][:, SLOT_IDX["c_last"], :].astype(np.float64)
        xb_ctl = own_bank[ctl_rows][:, SLOT_IDX["c_last"], :].astype(np.float64)
        res = run_ridge_cell(
            c_last_tr,
            Ytr,
            {"bank_div": (xb_div, div_tgt), "bank_ctl": (xb_ctl, ctl_tgt)},
            group_names=gnames,
            device="cpu",
            allow_train_nan_imputation=True,
        )
        # freeze at committed per-slot lambda (use gnames_arm_for_lam to map to committed group)
        lam_gnames = [f"{g.split('|')[0]}|{gnames_arm_for_lam}" for g in gnames]
        lam_idx = np.asarray([group2lam[g] for g in lam_gnames], dtype=np.int64)
        ssr_div = np.take_along_axis(res.ss_res["bank_div"], lam_idx[None, :, None], axis=2)[
            :, :, 0
        ]
        ssr_ctl = np.take_along_axis(res.ss_res["bank_ctl"], lam_idx[None, :, None], axis=2)[
            :, :, 0
        ]
        # per-query R2 pooled over the (single) fitted arm's POSITION_SLOTS
        arm = y_train_arms[0]
        r2_div = _per_query_r2(
            ssr_div.astype(np.float64), res.ss_tot["bank_div"].astype(np.float64), gnames, arm
        )
        r2_ctl = _per_query_r2(
            ssr_ctl.astype(np.float64), res.ss_tot["bank_ctl"].astype(np.float64), gnames, arm
        )
        return r2_div, r2_ctl

    def _pairwise_drop(r2_div, r2_ctl):
        rows = []
        for p in pairs:
            qd, qc = p["divergent"]["query_id"], p["control"]["query_id"]
            di, ci = div_id2row.get(qd), ctl_id2row.get(qc)
            if di is None or ci is None:
                continue
            rd, rc = r2_div[di], r2_ctl[ci]
            if not (np.isfinite(rd) and np.isfinite(rc)):
                continue
            rows.append(
                {
                    "pair_id": p["pair_id"],
                    "category": p["category"],
                    "r2_div": float(rd),
                    "r2_ctl": float(rc),
                    "drop": float(rc - rd),
                }
            )
        return rows

    # ── Gate 2: reproduce arm-matched maps on U_A, match committed npz ───────
    r2_div_own, r2_ctl_own = _run(("own",), "own", "own")
    r2_div_pl, r2_ctl_pl = _run(("ext_plain",), "ext_plain", "ext_plain")
    rep_own = _pairwise_drop(r2_div_own, r2_ctl_own)
    rep_pl = _pairwise_drop(r2_div_pl, r2_ctl_pl)
    my_drop_own = {r["pair_id"]: r["drop"] for r in rep_own}
    my_drop_pl = {r["pair_id"]: r["drop"] for r in rep_pl}
    g2_own_maxdelta = max(
        abs(my_drop_own[r["pair_id"]] - r["drop_own"])
        for r in gate1["rows"]
        if r["pair_id"] in my_drop_own
    )
    g2_pl_maxdelta = max(
        abs(my_drop_pl[r["pair_id"]] - r["drop_ext_plain"])
        for r in gate1["rows"]
        if r["pair_id"] in my_drop_pl
    )
    gate2_pass = g2_own_maxdelta < REPRO_TOL and g2_pl_maxdelta < REPRO_TOL
    logger.info(
        "[Gate2] max|drop_own delta|=%.4f max|drop_ext_plain delta|=%.4f (tol %.3f) pass=%s",
        g2_own_maxdelta,
        g2_pl_maxdelta,
        REPRO_TOL,
        gate2_pass,
    )

    # ── CROSS cell: own map x plain target, and symmetric plain map x own target ──
    r2_div_cross, r2_ctl_cross = _run(
        ("own",), "ext_plain", "own"
    )  # own map scored vs plain target
    cross_rows = _pairwise_drop(r2_div_cross, r2_ctl_cross)
    d_cross = np.asarray([r["drop"] for r in cross_rows])

    r2_div_sym, r2_ctl_sym = _run(("ext_plain",), "own", "ext_plain")  # plain map vs own target
    sym_rows = _pairwise_drop(r2_div_sym, r2_ctl_sym)
    d_sym = np.asarray([r["drop"] for r in sym_rows])

    # cross per-category sign-flip
    def _by_cat_signflip(rows, d):
        cats = sorted({r["category"] for r in rows})
        out = {}
        for c in cats:
            mask = np.asarray([r["category"] == c for r in rows])
            if mask.sum() >= 2:
                out[c] = _signflip_p(d[mask], N_DRAWS)
        return out

    out = {
        "description": (
            "MISSING #952 cross transfer cell: OWN-answer-fitted context map scored "
            "against the PLAIN-external arm's bank answer targets (own map x Claude "
            "answers), divergent vs entity-swapped control, over the 41 kept pairs. "
            "The #952 analog of #823's own->plain transfer. Machinery mirrors "
            "run_952 pass-2 + issue952_stats.h3_reads; only the (map arm, target arm) "
            "pairing changed. ss_tot centered on the FITTED arm's train mean "
            "(run_ridge_cell convention)."
        ),
        "n_pairs_kept": len(pairs),
        "n_draws": N_DRAWS,
        "gate1_stats_machinery": {
            "pass": bool(gate1_pass),
            "recomputed_mean_d": g1_d,
            "committed_mean_d": c_d,
            "recomputed_signflip_p": g1_p,
            "committed_signflip_p": c_p,
            "recomputed_n_pairs": gate1["n_pairs"],
            "committed_n_pairs": committed_stats["n_pairs"],
        },
        "gate2_fit_reproduction": {
            "pass": bool(gate2_pass),
            "tol": REPRO_TOL,
            "max_abs_delta_drop_own": g2_own_maxdelta,
            "max_abs_delta_drop_ext_plain": g2_pl_maxdelta,
            "note": "re-fit arm-matched maps on reproduced U_A; per-pair drops vs committed npz",
        },
        "committed_arm_matched": {
            "mean_drop_own": gate1["mean_drop_own"],
            "mean_drop_ext_plain": gate1["mean_drop_ext_plain"],
            "headline_d_mean": g1_d,
            "signflip_p": g1_p,
        },
        "cross_own_map_x_plain_target": {
            "headline_drop": _bank_boot(d_cross, N_DRAWS),
            "sign_flip_pooled": _signflip_p(d_cross, N_DRAWS),
            "sign_flip_by_category": _by_cat_signflip(cross_rows, d_cross),
            "mean_r2_div": float(np.mean([r["r2_div"] for r in cross_rows])),
            "mean_r2_ctl": float(np.mean([r["r2_ctl"] for r in cross_rows])),
            "n_pairs": len(cross_rows),
            "rows": cross_rows,
        },
        "symmetric_plain_map_x_own_target": {
            "headline_drop": _bank_boot(d_sym, N_DRAWS),
            "sign_flip_pooled": _signflip_p(d_sym, N_DRAWS),
            "mean_r2_div": float(np.mean([r["r2_div"] for r in sym_rows])),
            "mean_r2_ctl": float(np.mean([r["r2_ctl"] for r in sym_rows])),
            "n_pairs": len(sym_rows),
            "rows": sym_rows,
        },
        "tensor_source": "hf://.../issue952_position_divergence/analysis_tensors/ @ 5b62649cef",
        "layer": 20,
        "position_slots": POSITION_SLOTS,
        "git_commit": _sha(),
        "wall_seconds": round(time.time() - t_start, 1),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (out_dir / "cross_cell.json").write_text(json.dumps(out, indent=1))
    logger.info(
        "Wrote %s  gate1=%s gate2=%s cross_drop_mean=%.4f p=%.4f",
        out_dir / "cross_cell.json",
        gate1_pass,
        gate2_pass,
        out["cross_own_map_x_plain_target"]["headline_drop"]["mean"],
        out["cross_own_map_x_plain_target"]["sign_flip_pooled"]["p_one_sided"],
    )


if __name__ == "__main__":
    main()
