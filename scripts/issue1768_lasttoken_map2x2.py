#!/usr/bin/env python3
"""#1768 inline round: the exact input x operator 2x2 over the context->answer map.

Completes the decomposition the ``issue1768_lasttoken_m0_push`` round opened.
That round measured only the INPUT leg (context movement through the frozen
base map). This one refits M+ per (arm, layer) and evaluates all four corners
on the SAME committed held-out rows, giving the algebraically EXACT split

    total       = M+(c+) - M0(c0)
    input       = M0(c+) - M0(c0)          (the m0push leg, recomputed here)
    operator    = M+(c0) - M0(c0)          (the committed map-change leg)
    interaction = M+(c+) - M0(c+) - M+(c0) + M0(c0)
    total       = input + operator + interaction     (exact, asserted per cell)

Two questions this answers that the separate legs cannot: (1) do the two
mechanisms ADD or CANCEL (the interaction term, measured rather than inferred
from the m0push round's push > weights-carried-movement anomaly); and (2) is
the map-level description COMPLETE -- ||total|| is compared against the
independently measured answer displacement, so a large shortfall means a real
fraction of the fine-tuning effect is invisible to the linear map.

Two answer regimes, both run per cell (the parent's own convention):
``onpolicy`` fits M+ on the arm's OWN responses (comparable to the 2x2 round's
on-policy shift) and ``fixedtext`` fits M+ on the arm teacher-forced on the
base model's text (comparable to the weights-carried function effect).

Attribution uses the parent 2x2's convention: ``proj_share`` =
<term, total>/||total||^2 (additive, sums to 1) alongside raw norm ratios
(not additive). Context pooling is LAST-PROMPT-TOKEN throughout.

Venue: CPU pod (~150 GB rolling HF staging: per-arm context + on-policy answer
+ teacher-forced answer stores). Each arm's stores are deleted after its
layers are consumed; the base stores persist for the run.

Usage (pod):
    uv run python scripts/issue1768_lasttoken_map2x2.py \
        --out-root /workspace/i1768_map2x2 [--arms A,B] [--layers 14,19,25]
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1768_capture as CAP  # noqa: E402
import issue1768_cells as X  # noqa: E402
import issue1768_fit as F  # noqa: E402
import issue1768_lasttoken_fit as LTF  # noqa: E402
import issue1768_lasttoken_m0_push as M0P  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.map2x2")

POSITION = M0P.POSITION  # "last_prompt" — the round's PRIMARY pooling
# (regime key, HF store kind, committed-cell key holding the parent's Mplus reads)
REGIMES = (
    ("onpolicy", "corpus_capture", "Mplus"),
    ("fixedtext", "corpus_capture_tf", "Mplus_tf"),
)


def _terms(p00: np.ndarray, p0p: np.ndarray, pp0: np.ndarray, ppp: np.ndarray) -> dict:
    """Exact additive split of the total map-predicted answer displacement."""
    total = ppp - p00
    inp = p0p - p00
    op = pp0 - p00
    inter = ppp - p0p - pp0 + p00
    resid = float(np.abs(total - (inp + op + inter)).max())
    assert resid < 1e-8, f"2x2 identity broken: max abs residual {resid}"

    tot_n = np.linalg.norm(total, axis=1)
    sq = float((tot_n**2).sum())
    out = {
        "n_rows": int(total.shape[0]),
        "identity_residual": resid,
        "median_norm": {"total": float(np.median(tot_n))},
        "mean_norm": {"total": float(tot_n.mean())},
        "norm_ratio": {},  # ||term|| / ||total|| — NOT additive
        "proj_share": {},  # <term,total>/||total||^2 — additive, sums to 1
        "cos_with_total": {},
    }
    for name, term in (("input", inp), ("operator", op), ("interaction", inter)):
        tn = np.linalg.norm(term, axis=1)
        out["median_norm"][name] = float(np.median(tn))
        out["mean_norm"][name] = float(tn.mean())
        out["norm_ratio"][name] = float(np.median(tn / (tot_n + 1e-12)))
        out["proj_share"][name] = float((term * total).sum() / sq) if sq > 0 else float("nan")
        out["cos_with_total"][name] = float(
            np.median((term * total).sum(axis=1) / (tn * tot_n + 1e-12))
        )
    out["proj_share_sum"] = sum(out["proj_share"].values())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--arms", type=str, default="", help="comma list; default = all committed")
    ap.add_argument("--layers", type=str, default="14,19,25")
    ap.add_argument(
        "--regimes",
        type=str,
        default="onpolicy,fixedtext",
        help="answer-side regimes to fit (output-affecting; part of the resume key)",
    )
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", str(os.cpu_count() or 16))))
    dev = torch.device("cpu")

    out_root: Path = args.out_root
    layers = [int(x) for x in args.layers.split(",")]
    want = [r for r in args.regimes.split(",") if r]
    regimes = [r for r in REGIMES if r[0] in want]
    assert regimes, f"no known regimes in {want!r} (known: {[r[0] for r in REGIMES]})"
    cache = out_root / "cache"
    results = out_root / "results"
    (results / "cells").mkdir(parents=True, exist_ok=True)
    M0P._stage_corpus_sample(out_root)

    arms = [a for a in (args.arms.split(",") if args.arms else M0P._arm_list()) if a]
    base_units = sorted({X.base_unit_for(a) for a in arms})
    logger.info(
        "[plan] %d arms, base units %s, layers %s, regimes %s",
        len(arms),
        base_units,
        layers,
        [r[0] for r in regimes],
    )

    # Base side: C0/V0 join, fitted M0, base predictions on te rows (m0push verbatim).
    bases = {u: M0P.BaseSide(out_root, cache, u, layers, dev) for u in base_units}

    done = skipped = 0
    for k, arm in enumerate(arms):
        cell_path = results / "cells" / f"{arm}.json"
        if cell_path.exists():
            skipped += 1
            continue
        t0 = time.time()
        base = bases[X.base_unit_for(arm)]
        M0P._stage_lasttoken(out_root, arm)
        cp_by, cp_sha = LTF.load_lasttoken(out_root, arm, layers, POSITION)
        cp_ix = {s: i for i, s in enumerate(cp_sha)}
        missing = sum(1 for s in base.shas if s not in cp_ix)
        assert missing == 0, (arm, f"{missing} base shas absent from arm context store")
        sel_c = np.asarray([cp_ix[s] for s in base.shas])

        out: dict = {
            "arm_id": arm,
            "base_unit": base.unit,
            "position": POSITION,
            "n_rows": int(len(base.shas)),
            "regimes": {},
        }
        for regime, kind, committed_key in regimes:
            vp_by, vp_sha = LTF.fetch_response(cache, kind, arm, layers)
            vp_ix = {s: i for i, s in enumerate(vp_sha)}
            miss_v = sum(1 for s in base.shas if s not in vp_ix)
            assert miss_v == 0, (arm, regime, f"{miss_v} base shas absent from arm answer store")
            sel_v = np.asarray([vp_ix[s] for s in base.shas])

            out["regimes"][regime] = {}
            for li in layers:
                m0 = base.m0[li]
                committed = M0P._committed_cell(arm)["positions"][POSITION][str(li)]
                Cp = cp_by[li][sel_c]
                Vp = vp_by[li][sel_v]
                predp_te, metap, payp = F._fit_map(Cp, Vp, base.tr, base.val, base.te, dev)
                r2p = F._pooled_r2(predp_te, Vp[base.te])
                # Parent-fidelity assert: the refit must reproduce the committed M+ read.
                r2_ref = committed[committed_key]["heldout_r2"]
                assert abs(r2p - r2_ref) < 5e-3, (arm, regime, li, r2p, r2_ref)

                p00 = m0["pred0_te"]  # M0(c0)
                p0p = F._apply_payload(m0["payload"], Cp[base.te], dev)  # M0(c+)
                pp0 = F._apply_payload(payp, base.C0[li][base.te], dev)  # M+(c0)
                ppp = predp_te  # M+(c+)

                blk = _terms(p00, p0p, pp0, ppp)
                blk.update(
                    {
                        "m0_heldout_r2": m0["heldout_r2"],
                        "m0_selected_lambda": m0["selected_lambda"],
                        "mplus_heldout_r2": r2p,
                        "mplus_heldout_r2_committed": r2_ref,
                        "mplus_selected_lambda": float(metap["selected_lambda"]),
                        "delta_med_committed": committed["map_change"]["delta_med"],
                        "floor_p95_committed": committed["map_change"]["floor_p95"],
                        "ctx_move_committed": committed["context_movement"]["median_relative_move"],
                    }
                )
                out["regimes"][regime][str(li)] = blk
                logger.info(
                    "[cell] %s %s L%d: total=%.3f input=%.3f op=%.3f inter=%.3f "
                    "(shares %.2f/%.2f/%.2f)",
                    arm,
                    regime,
                    li,
                    blk["median_norm"]["total"],
                    blk["median_norm"]["input"],
                    blk["median_norm"]["operator"],
                    blk["median_norm"]["interaction"],
                    blk["proj_share"]["input"],
                    blk["proj_share"]["operator"],
                    blk["proj_share"]["interaction"],
                )
                del Cp, Vp, p0p, pp0, ppp, predp_te, payp
            del vp_by
            gc.collect()

        CAP._atomic_json(cell_path, out)
        del cp_by
        gc.collect()
        M0P._drop_unit_store(out_root, arm)
        done += 1
        logger.info("[arm %d/%d] %s done in %.0fs", k + 1, len(arms), arm, time.time() - t0)

    summary = {
        "issue": 1768,
        "read": (
            "exact input x operator 2x2 over the context->answer map: "
            "M+(c+) - M0(c0) = [M0(c+)-M0(c0)] + [M+(c0)-M0(c0)] + interaction"
        ),
        "position": POSITION,
        "layers": layers,
        "regimes": [r[0] for r in regimes],
        "n_arms": len(arms),
        "n_done_this_run": done,
        "n_skipped_resume": skipped,
        "m0_fits": {
            u: {
                str(li): {
                    "selected_lambda": b.m0[li]["selected_lambda"],
                    "heldout_r2": b.m0[li]["heldout_r2"],
                }
                for li in layers
            }
            for u, b in bases.items()
        },
        **CAP._meta(),
    }
    CAP._atomic_json(results / "summary.json", summary)
    logger.info("[done] %d arms (%d resumed), summary -> %s", len(arms), skipped, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
