"""#952 free-analysis follow-up: divergence-bank re-read WITH china-politics included.

Inline follow-up round `china-politics-topup` (0 GPU, VM CPU). Reuses the
divergence_transfer_cell machinery VERBATIM (same L20 pool maps, same per-context
R2 construction, same run_ridge_cell ss_tot convention, same 10k-draw batched
sign-flip) — the DL-independent helpers are IMPORTED from
issue952_divergence_transfer_cell so the machinery is provably identical; only
the bank targets + pair set change.

Inputs:
  - Committed maps + 41-pair bank: analysis_tensors @ 5b62649cef (pool slots,
    committed bank slots, per_context_stats.npz, spans).
  - China bank: followups/china_politics_topup/analysis_tensors @ 612c6c744e
    (slots_bank_china_politics_topup_{own,ext_plain}_L20.pt, 84 queries = 42
    pairs). Final china category = 31 kept pairs (18 parent-kept + 13 new),
    from eval_results/issue_952/china-politics-topup/summaries/
    china_topup_verification.json -> final_china_kept_pairs. The other 11
    captured-but-rejected pairs are excluded.

Reads (all at L20):
  0. REPRODUCTION GATE (first): recompute the committed 41-pair arm-matched
     cells; must match the committed npz (Gate2) + committed h3 stats (Gate1).
  1. ARM-MATCHED china: own map x own china targets vs plain map x plain china
     targets; per-pair drop_arm = R2_ctl - R2_div; decision d = drop_ext_plain
     - drop_own; sign-flip over 31 pair signs; vs 0.05 margin.
  2. CROSS china: own map x plain china targets (+ symmetric plain map x own).
  3. POOLED: 72 pairs (41 committed + 31 china) arm-matched d + sign-flip, with
     committed-41 alongside; per-category sign-flip with Holm across
     {model_identity, style_format, china_politics}.
  4. Mean bank R2 per arm on china pairs vs committed identity/style levels.

Bank content rule: pair/query ids only; NO bank item text touched.

Usage:
  uv run python scripts/issue952_china_included_stats.py
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import subprocess

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue952_divergence_transfer_cell import (  # noqa: E402
    ARMS,
    BANK_ARMS,
    H3_MARGIN,
    N_DRAWS,
    SLOT_IDX,
    _bank_boot,
    _per_query_r2,
    _signflip_p,
    _stack_answer_targets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue952_china_included_stats")

ROOT = pathlib.Path(
    "/mnt/eps-data/thomasjiralerspong/tmp_issue952_china/issue952_position_divergence"
)
DL = ROOT / "analysis_tensors"
CDL = ROOT / "followups/china_politics_topup/analysis_tensors"
REPRO_TOL = 0.01
CHINA_CAT = "china_politics"


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _load_slots(path: pathlib.Path) -> tuple[np.ndarray, list]:
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    return d["slots"].numpy(), list(d["ids"])


def _role_split(bank_ids: list) -> tuple[list[int], list[int], dict, dict]:
    div_rows = [i for i, q in enumerate(bank_ids) if str(q).endswith("_div")]
    ctl_rows = [i for i, q in enumerate(bank_ids) if not str(q).endswith("_div")]
    div_id2row = {str(bank_ids[i]): pos for pos, i in enumerate(div_rows)}
    ctl_id2row = {str(bank_ids[i]): pos for pos, i in enumerate(ctl_rows)}
    return div_rows, ctl_rows, div_id2row, ctl_id2row


def _pairwise_drops(r2_div, r2_ctl, pairs, div_id2row, ctl_id2row):
    """pairs: list of (pair_id, category, qd, qc). Returns rows with per-arm drops."""
    rows = []
    for pid, cat, qd, qc in pairs:
        di, ci = div_id2row.get(qd), ctl_id2row.get(qc)
        if di is None or ci is None:
            continue
        rec = {"pair_id": pid, "category": cat}
        ok = True
        for arm in r2_div:
            rd, rc = r2_div[arm][di], r2_ctl[arm][ci]
            if not (np.isfinite(rd) and np.isfinite(rc)):
                ok = False
            rec[f"r2_div_{arm}"] = float(rd)
            rec[f"r2_ctl_{arm}"] = float(rc)
            rec[f"drop_{arm}"] = float(rc - rd)
        if ok:
            rows.append(rec)
    return rows


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvals.items(), key=lambda x: x[1])
    k = len(ordered)
    out, running = {}, 0.0
    for rank, (cat, p) in enumerate(ordered):
        running = max(running, min(1.0, (k - rank) * p))
        out[cat] = running
    return out


def main() -> None:  # noqa: C901 — one orchestration fn: gates + arm-matched + cross + pooled reads
    torch.set_num_threads(8)
    from explore_persona_space.experiments.issue_952.ridge_battery import run_ridge_cell

    base = pathlib.Path(__file__).resolve().parent.parent
    out_dir = base / "eval_results" / "issue_952" / "china-politics-topup"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = __import__("time").time()

    # ── universe / split reproduction (identical to divergence_transfer_cell) ──
    own_pool, pool_ids = _load_slots(DL / "slots_own_L20.pt")
    plain_pool, _ = _load_slots(DL / "slots_ext_plain_L20.pt")
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
    committed_stats = json.loads((base / "eval_results/issue_952/stats_summary.json").read_text())[
        "h3"
    ]

    # ── pair sets ──────────────────────────────────────────────────────────────
    comm_verif = json.loads(
        (base / "eval_results/issue_952/divergence_bank_verification.json").read_text()
    )
    comm_kept = set(comm_verif["kept_pairs"])
    comm_pairs = [
        (p["pair_id"], p["category"], p["divergent"]["query_id"], p["control"]["query_id"])
        for p in comm_verif["pairs"]
        if p["pair_id"] in comm_kept
        and isinstance(p.get("divergent"), dict)
        and isinstance(p.get("control"), dict)
    ]
    china_verif = json.loads(
        (
            base
            / "eval_results/issue_952/china-politics-topup/summaries/china_topup_verification.json"
        ).read_text()
    )
    china_kept_ids = china_verif["final_china_kept_pairs"]
    china_pairs = [(pid, CHINA_CAT, f"{pid}_div", f"{pid}_ctl") for pid in china_kept_ids]
    logger.info("committed kept=%d china kept=%d", len(comm_pairs), len(china_pairs))

    # ── bank tensors + role splits ──────────────────────────────────────────────
    comm_bank = {a: _load_slots(DL / f"slots_bank_{a}_L20.pt") for a in BANK_ARMS}
    china_bank = {
        a: _load_slots(CDL / f"slots_bank_china_politics_topup_{a}_L20.pt") for a in BANK_ARMS
    }
    assert comm_bank["own"][1] == comm_bank["ext_plain"][1]
    assert china_bank["own"][1] == china_bank["ext_plain"][1]
    china_ids = china_bank["own"][1]
    missing = [q for _, _, qd, qc in china_pairs for q in (qd, qc) if q not in china_ids]
    assert not missing, f"china kept query ids absent from china shards: {missing[:5]}"

    comm_div_rows, comm_ctl_rows, comm_div_i2r, comm_ctl_i2r = _role_split(comm_bank["own"][1])
    ch_div_rows, ch_ctl_rows, ch_div_i2r, ch_ctl_i2r = _role_split(china_ids)

    slots_by_arm = {"own": own_pool, "ext_plain": plain_pool}
    c_last_tr = own_pool[tr_a][:, SLOT_IDX["c_last"], :].astype(np.float64)

    def _eval_targets(bank, rows, arms):
        """Stack answer targets from `bank` (dict arm->(slots,ids)) at `rows` for `arms`."""
        sba = {a: bank[a][0] for a in arms}
        tgt, _g = _stack_answer_targets(sba, np.asarray(rows), arms)
        return tgt

    def _x_bank(bank, rows):
        return bank["own"][0][rows][:, SLOT_IDX["c_last"], :].astype(np.float64)

    def _fit_eval(fit_arms, evals):
        """evals: name -> (x_bank_c_last, target_slots_by_arm_dict, rows).
        target_slots_by_arm_dict maps each fit arm's group to the tensor to score against."""
        Ytr, gnames = _stack_answer_targets(slots_by_arm, tr_a, fit_arms)
        eval_splits = {}
        for name, (xb, tgt_bank, rows) in evals.items():
            tgt, _g = _stack_answer_targets(tgt_bank, np.asarray(rows), fit_arms)
            eval_splits[name] = (xb, tgt)
        res = run_ridge_cell(
            c_last_tr,
            Ytr,
            eval_splits,
            group_names=gnames,
            device="cpu",
            allow_train_nan_imputation=True,
        )
        lam_idx = np.asarray([group2lam[f"{g.split('|')[0]}|own"] for g in gnames], dtype=np.int64)
        out = {}
        for name in evals:
            ssr = np.take_along_axis(res.ss_res[name], lam_idx[None, :, None], axis=2)[:, :, 0]
            sst = res.ss_tot[name].astype(np.float64)
            out[name] = {
                arm: _per_query_r2(ssr.astype(np.float64), sst, gnames, arm) for arm in fit_arms
            }
        return out

    # ── Fit A: arm-matched (own+plain maps), eval committed + china banks ───────
    A = _fit_eval(
        ("own", "ext_plain"),
        {
            "comm_div": (
                _x_bank(comm_bank, comm_div_rows),
                {a: comm_bank[a][0] for a in BANK_ARMS},
                comm_div_rows,
            ),
            "comm_ctl": (
                _x_bank(comm_bank, comm_ctl_rows),
                {a: comm_bank[a][0] for a in BANK_ARMS},
                comm_ctl_rows,
            ),
            "china_div": (
                _x_bank(china_bank, ch_div_rows),
                {a: china_bank[a][0] for a in BANK_ARMS},
                ch_div_rows,
            ),
            "china_ctl": (
                _x_bank(china_bank, ch_ctl_rows),
                {a: china_bank[a][0] for a in BANK_ARMS},
                ch_ctl_rows,
            ),
        },
    )

    comm_rows = _pairwise_drops(
        A["comm_div"], A["comm_ctl"], comm_pairs, comm_div_i2r, comm_ctl_i2r
    )
    china_rows = _pairwise_drops(
        A["china_div"], A["china_ctl"], china_pairs, ch_div_i2r, ch_ctl_i2r
    )
    for r in comm_rows + china_rows:
        r["d"] = r["drop_ext_plain"] - r["drop_own"]

    # ── Reproduction gate ───────────────────────────────────────────────────────
    def _committed_h3_from_npz():
        groups = npz["A_group_names"].tolist()
        r2 = {}
        for key in ("bank_div", "bank_ctl"):
            ssr = npz[f"{key}_ssres"].astype(np.float64)
            sst = npz[f"{key}_sstot"].astype(np.float64)
            idmap = {str(q): i for i, q in enumerate(npz[f"{key}_ids"].tolist())}
            r2[key] = ({arm: _per_query_r2(ssr, sst, groups, arm) for arm in BANK_ARMS}, idmap)
        ds = []
        for _pid, _cat, qd, qc in comm_pairs:
            di = r2["bank_div"][1].get(qd)
            ci = r2["bank_ctl"][1].get(qc)
            if di is None or ci is None:
                continue
            do = r2["bank_ctl"][0]["own"][ci] - r2["bank_div"][0]["own"][di]
            de = r2["bank_ctl"][0]["ext_plain"][ci] - r2["bank_div"][0]["ext_plain"][di]
            if np.isfinite(do) and np.isfinite(de):
                ds.append(de - do)
        return np.asarray(ds)

    d_npz = _committed_h3_from_npz()
    gate1_d = float(d_npz.mean())
    gate1_p = _signflip_p(d_npz, N_DRAWS)["p_one_sided"]
    c_d = committed_stats["headline_mean_drop_diff"]["mean"]
    c_p = committed_stats["sign_flip_null"]["pooled"]["p_one_sided"]
    gate1_pass = abs(gate1_d - c_d) < 1e-6 and len(d_npz) == committed_stats["n_pairs"]

    # Gate2: Fit-A committed drops vs committed npz per-pair (bit reproduction)
    npz_groups = npz["A_group_names"].tolist()
    npz_div_i2r = {str(q): i for i, q in enumerate(npz["bank_div_ids"].tolist())}
    npz_ctl_i2r = {str(q): i for i, q in enumerate(npz["bank_ctl_ids"].tolist())}
    npz_r2 = {}
    for key in ("bank_div", "bank_ctl"):
        ssr = npz[f"{key}_ssres"].astype(np.float64)
        sst = npz[f"{key}_sstot"].astype(np.float64)
        npz_r2[key] = {arm: _per_query_r2(ssr, sst, npz_groups, arm) for arm in BANK_ARMS}
    g2_delta = 0.0
    comm_pid2qs = {pid: (qd, qc) for pid, _c, qd, qc in comm_pairs}
    for r in comm_rows:
        qd, qc = comm_pid2qs[r["pair_id"]]
        di, ci = npz_div_i2r.get(qd), npz_ctl_i2r.get(qc)
        if di is None or ci is None:
            continue
        for arm in BANK_ARMS:
            npz_drop = npz_r2["bank_ctl"][arm][ci] - npz_r2["bank_div"][arm][di]
            g2_delta = max(g2_delta, abs(r[f"drop_{arm}"] - npz_drop))
    gate2_pass = g2_delta < REPRO_TOL
    logger.info(
        "[GATE1] d=%.6f (committed %.6f) p=%.4f n=%d pass=%s | [GATE2] maxdelta=%.5f pass=%s",
        gate1_d,
        c_d,
        gate1_p,
        len(d_npz),
        gate1_pass,
        g2_delta,
        gate2_pass,
    )

    # ── China arm-matched read ──────────────────────────────────────────────────
    d_china = np.asarray([r["d"] for r in china_rows])
    china_arm_matched = {
        "n_pairs": len(china_rows),
        "mean_drop_own": float(np.mean([r["drop_own"] for r in china_rows])),
        "mean_drop_ext_plain": float(np.mean([r["drop_ext_plain"] for r in china_rows])),
        "headline_d": _bank_boot(d_china, N_DRAWS),
        "sign_flip": _signflip_p(d_china, N_DRAWS),
        "clears_margin_0p05": bool(float(d_china.mean()) >= H3_MARGIN),
    }

    # ── China cross cell (own map x plain china target) + symmetric ────────────
    def _cross(fit_arm, target_arm):
        ev = _fit_eval(
            (fit_arm,),
            {
                "china_div": (
                    _x_bank(china_bank, ch_div_rows),
                    {fit_arm: china_bank[target_arm][0]},
                    ch_div_rows,
                ),
                "china_ctl": (
                    _x_bank(china_bank, ch_ctl_rows),
                    {fit_arm: china_bank[target_arm][0]},
                    ch_ctl_rows,
                ),
            },
        )
        rows = _pairwise_drops(
            {fit_arm: ev["china_div"][fit_arm]},
            {fit_arm: ev["china_ctl"][fit_arm]},
            china_pairs,
            ch_div_i2r,
            ch_ctl_i2r,
        )
        for r in rows:
            r["drop"] = r[f"drop_{fit_arm}"]
        d = np.asarray([r["drop"] for r in rows])
        return {
            "n_pairs": len(rows),
            "mean_r2_div": float(np.mean([r[f"r2_div_{fit_arm}"] for r in rows])),
            "mean_r2_ctl": float(np.mean([r[f"r2_ctl_{fit_arm}"] for r in rows])),
            "headline_drop": _bank_boot(d, N_DRAWS),
            "sign_flip": _signflip_p(d, N_DRAWS),
            "rows": [
                {
                    "pair_id": r["pair_id"],
                    "r2_div": r[f"r2_div_{fit_arm}"],
                    "r2_ctl": r[f"r2_ctl_{fit_arm}"],
                    "drop": r["drop"],
                }
                for r in rows
            ],
        }

    china_cross = _cross("own", "ext_plain")  # own map x plain china target
    china_symmetric = _cross("ext_plain", "own")  # plain map x own china target

    # ── Pooled + per-category (Holm across 3) ──────────────────────────────────
    pooled72 = comm_rows + china_rows
    d72 = np.asarray([r["d"] for r in pooled72])
    d41 = np.asarray([r["d"] for r in comm_rows])
    cats = {}
    for cat in ("model_identity", "style_format", CHINA_CAT):
        rr = [r for r in pooled72 if r["category"] == cat]
        if len(rr) >= 2:
            dd = np.asarray([r["d"] for r in rr])
            cats[cat] = {
                "n": len(rr),
                "mean_d": float(dd.mean()),
                "sign_flip": _signflip_p(dd, N_DRAWS),
            }
    holm = _holm({c: cats[c]["sign_flip"]["p_one_sided"] for c in cats})
    for c in cats:
        cats[c]["p_holm"] = holm[c]

    # ── R2 levels per arm, per category (transfer-level check) ─────────────────
    def _r2_levels(rows):
        return {
            arm: {
                "mean_r2_div": float(np.mean([r[f"r2_div_{arm}"] for r in rows])),
                "mean_r2_ctl": float(np.mean([r[f"r2_ctl_{arm}"] for r in rows])),
            }
            for arm in BANK_ARMS
        }

    r2_levels = {
        "china": _r2_levels(china_rows),
        "committed_all41": _r2_levels(comm_rows),
        "model_identity": _r2_levels([r for r in comm_rows if r["category"] == "model_identity"]),
        "style_format": _r2_levels([r for r in comm_rows if r["category"] == "style_format"]),
    }

    out = {
        "description": (
            "#952 divergence-bank re-read WITH china-politics included "
            "(china-politics-topup). Same L20 pool maps + machinery as "
            "divergence_transfer_cell; only bank targets + pair set change. "
            "31 kept china pairs (18 parent-kept + 13 new)."
        ),
        "layer": 20,
        "n_draws": N_DRAWS,
        "margin": H3_MARGIN,
        "revisions": {
            "committed": "5b62649cef",
            "china_topup": "612c6c744e",
            "git_summaries": "115ef77ae3",
        },
        "reproduction_gate": {
            "gate1_stats": {
                "pass": bool(gate1_pass),
                "recomputed_d": gate1_d,
                "committed_d": c_d,
                "recomputed_p": gate1_p,
                "committed_p": c_p,
                "n": len(d_npz),
            },
            "gate2_fit": {
                "pass": bool(gate2_pass),
                "max_abs_delta_drop": g2_delta,
                "tol": REPRO_TOL,
            },
        },
        "china_arm_matched": china_arm_matched,
        "china_cross_own_map_x_plain_target": china_cross,
        "china_symmetric_plain_map_x_own_target": china_symmetric,
        "pooled": {
            "pooled_72": {
                "n": len(pooled72),
                "mean_d": float(d72.mean()),
                "headline_d": _bank_boot(d72, N_DRAWS),
                "sign_flip": _signflip_p(d72, N_DRAWS),
            },
            "committed_41": {
                "n": len(comm_rows),
                "mean_d": float(d41.mean()),
                "headline_d": _bank_boot(d41, N_DRAWS),
                "sign_flip": _signflip_p(d41, N_DRAWS),
            },
        },
        "per_category_holm": cats,
        "r2_levels_by_arm": r2_levels,
        "per_pair_rows": {
            "china_arm_matched": china_rows,
            "committed_41_arm_matched": comm_rows,
        },
        "git_commit": _sha(),
        "wall_seconds": round(__import__("time").time() - t0, 1),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (out_dir / "stats_china_included.json").write_text(json.dumps(out, indent=1))
    logger.info(
        "Wrote %s | china d=%.4f p=%.4f | cross drop=%.4f p=%.4f | pooled72 d=%.4f p=%.4f",
        out_dir / "stats_china_included.json",
        china_arm_matched["headline_d"]["mean"],
        china_arm_matched["sign_flip"]["p_one_sided"],
        china_cross["headline_drop"]["mean"],
        china_cross["sign_flip"]["p_one_sided"],
        out["pooled"]["pooled_72"]["mean_d"],
        out["pooled"]["pooled_72"]["sign_flip"]["p_one_sided"],
    )


if __name__ == "__main__":
    main()
