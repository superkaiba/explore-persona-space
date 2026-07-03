#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #922 Phase 2: per-layer next-position map fits (plan §4.3).

From the LMSYS position store (Phase 1), on the 4000/500/500 by-context split
(seed 42, #841 ``make_split`` semantics):

- **Answer-segment ridge** per arm ∈ {context-only h_t, token-informed
  [h_t, e_{t+1}], embedding-only e_{t+1}}: ONE Gram assembly (single streaming
  pass building all arms' blocks) + ONE ``eigh`` per (layer, arm), λ by GCV
  over ``logspace(-2, 3, 25)`` in the eigenbasis. The embedding arm's design
  is layer-independent, so its eigendecomposition is computed ONCE and shared
  across the 29 per-layer target solves. Raw and RMS-norm spaces share every
  fit exactly (scalar σ rescale — see maps922.ridge_gcv_from_grams).
- **Step-1 BOUNDARY ridge** (ctx + tok arms) on the fit-set boundary
  transitions (~1/context) — the PRIMARY DV2/DV3 rollout seeding maps
  (methodology-reconcile Must-Fix); weights persisted.
- **MLP** ONE batched fit per (arm ∈ {ctx, tok} × space ∈ {raw, rmsnorm}) — 4
  fits, dispatched through the ported ``fit_batched_split_mlp`` via the
  store-fed chunk wrapper (``maps922.fit_position_mlps``); hidden 512, GELU,
  AdamW lr 1e-3 wd 1e-4, SmoothL1(β=1) on Δ/σ, ≤300 epochs best-inner-val.
- **Exploratory position-GRU** per read-out block (hidden 1024, teacher-forced
  on window sequences, raw Δ).

``--verify-fits`` runs the equivalence gates FIRST on the exact callables this
entrypoint dispatches (hollow-gate rule): the Gram/eigh-GCV ridge vs #658's
dual solver at fixed λ, and the batched split-MLP vs its serial reference.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue922_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    assert_split_mlp_matches_serial,
    fit_batched_split_mlp,
)
from explore_persona_space.experiments.issue_922 import maps922 as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue922_fit_maps")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but no CUDA; falling back to cpu")
        return "cpu"
    return requested


def transition_indices(store: dict, ctx_sel: np.ndarray) -> dict:
    """Global SOURCE-position indices of transitions for the selected contexts.

    Returns ``{"answer": LongTensor, "boundary": LongTensor, "prompt": ...,
    "template_end": ..., "ansrel": {global_src_idx→answer-relative index}}``
    (ansrel: boundary → 0 is NOT included; answer transition with source t has
    ansrel = t − prompt_len, i.e. 0 for the first answer→answer transition).
    """
    segs_by = {C.SEG_ANSWER: [], C.SEG_BOUNDARY: [], C.SEG_PROMPT: [], C.SEG_TEMPLATE_END: []}
    ansrel: list[tuple[int, int]] = []
    pos_lo, n_pos = store["pos_lo"], store["n_pos"]
    plen, ws = store["prompt_len"], store["window_start"]
    seg_all = store["segments"]
    for i in ctx_sel:
        lo, npos = int(pos_lo[i]), int(n_pos[i])
        for j in range(npos - 1):
            src = lo + j
            seg = int(seg_all[src])
            segs_by[seg].append(src)
            if seg == C.SEG_ANSWER:
                t_abs = int(ws[i]) + j
                ansrel.append((src, t_abs - int(plen[i])))
    out = {C.SEG_NAMES[k]: torch.tensor(v, dtype=torch.long) for k, v in segs_by.items()}
    out["ansrel"] = dict(ansrel)
    return out


def main() -> int:  # noqa: C901 — the phase sequence IS the fit spec (grams/ridge/boundary/mlp/gru)
    ap = argparse.ArgumentParser(description="Issue #922 map fits.")
    ap.add_argument("--store", type=Path, default=Path("/workspace/issue922_store"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/issue922_maps"))
    ap.add_argument("--split-seed", type=int, default=C.SPLIT_SEED)
    ap.add_argument("--blocks", default=None, help="comma list ('emb,0,...,27'); default all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mlp-layer-chunk", type=int, default=1)
    ap.add_argument("--mlp-max-epochs", type=int, default=300)
    ap.add_argument("--gru-max-epochs", type=int, default=40)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--gram-chunk", type=int, default=4096)
    ap.add_argument("--verify-fits", action="store_true", help="run equivalence gates and exit")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = _resolve_device(args.device)

    if args.verify_fits:
        # Hollow-gate rule: assert the gates exercise the EXACT callables this
        # entrypoint dispatches below (function identity, not lookalikes).
        assert M.ridge_gcv_from_grams is sys.modules[M.__name__].ridge_gcv_from_grams
        assert (
            fit_batched_split_mlp
            is sys.modules[
                "explore_persona_space.analysis.vectorized_mlp_skill"
            ].fit_batched_split_mlp
        )
        g1 = M.verify_ridge_gcv_against_dual()
        g2 = assert_split_mlp_matches_serial()
        logger.info("[verify-fits] ridge parity %s | split-MLP parity %s", g1, g2)
        print(json.dumps({"ridge_gate": g1, "split_mlp_gate": g2}))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    store = C.load_store(args.store, "lmsys")
    R, P, H = store["h"].shape
    n_ctx = len(store["ctx_ids"])
    logger.info("[store] R=%d P=%d H=%d n_ctx=%d", R, P, H, n_ctx)

    # Device placement: store on GPU when it fits (fits are GPU-worthy).
    h = store["h"]
    if device == "cuda":
        free_b, _ = torch.cuda.mem_get_info()
        need = h.numel() * 2 + 14 * (1 << 30)
        if free_b > need:
            h = h.to("cuda")
        else:
            logger.warning(
                "[hbm] store (%.1f GB) + headroom > free — store stays CPU-resident",
                h.numel() * 2 / 1e9,
            )
    blocks = (
        [b if b == "emb" else int(b) for b in args.blocks.split(",")]
        if args.blocks
        else store["blocks"][:1] + [int(x) for x in store["blocks"][1:]]
    )
    rows = [C.block_to_row(b) for b in blocks if C.block_to_row(b) != 0]
    assert rows, "need at least one non-embedding block to fit"
    assert all(0 < r < R for r in rows), (rows, R)

    split = C.make_split(n_ctx, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=args.split_seed)
    tr = {name: transition_indices(store, split[name]) for name in ("fit", "val", "test")}
    ns = {
        name: {
            seg: int(tr[name][seg].numel())
            for seg in ("answer", "boundary", "prompt", "template_end")
        }
        for name in tr
    }
    logger.info("[split] transition counts: %s", json.dumps(ns))
    assert ns["fit"]["answer"] > 0 and ns["fit"]["boundary"] > 0, ns["fit"]

    summary: dict = {"ns": ns, "rows": rows, "blocks": [C.row_to_block_key(r) for r in rows]}
    t_all = time.time()

    # ── answer-segment Grams + ridge (3 arms) ─────────────────────────────────
    t0 = time.time()
    grams = M.accumulate_grams(
        h, tr["fit"]["answer"], rows, emb_row=0, chunk=args.gram_chunk, device=device
    )
    summary["gram_wall_s"] = time.time() - t0
    sigma_by_row = {r: float(np.sqrt(grams["ctx"][r].syy / (grams["n"] * H))) for r in rows}
    delta_train_mean = {r: (grams["ctx"][r].sy / grams["n"]).to(torch.float32).cpu() for r in rows}
    summary["sigma_by_row"] = {C.row_to_block_key(r): sigma_by_row[r] for r in rows}

    ridge_answer: dict = {"ctx": {}, "tok": {}, "emb": {}}
    ridge_diag: dict = {"answer": {}, "boundary": {}}
    eigh_times: list[dict] = []
    for arm in ("ctx", "tok", "emb"):
        shared_eig = None
        for r in rows:
            stats = grams[arm][r]
            t1 = time.time()
            if arm == "emb" and shared_eig is not None:
                rmap, diag = M.ridge_gcv_from_grams(stats, sigma=1.0, eig=shared_eig)
            else:
                rmap, diag = M.ridge_gcv_from_grams(stats, sigma=1.0)
            if arm == "emb" and shared_eig is None:
                shared_eig = diag["eig"]
            ridge_answer[arm][r] = rmap.to("cpu")
            ridge_diag["answer"].setdefault(arm, {})[C.row_to_block_key(r)] = {
                "best_lam": diag["best_lam"],
                "gcv_curve": diag["gcv_curve"],
                "eigh_seconds": diag["eigh_seconds"],
            }
            eigh_times.append(
                {
                    "arm": arm,
                    "row": r,
                    "d": diag["d"],
                    "eigh_s": diag["eigh_seconds"],
                    "total_s": time.time() - t1,
                }
            )
    del grams

    # ── boundary Grams + ridge (ctx + tok arms; n ≈ n_fit contexts) ───────────
    t0 = time.time()
    bgrams = M.accumulate_grams(
        h, tr["fit"]["boundary"], rows, emb_row=0, chunk=args.gram_chunk, device=device
    )
    summary["boundary_gram_wall_s"] = time.time() - t0
    ridge_boundary: dict = {"ctx": {}, "tok": {}}
    boundary_mean = {r: (bgrams["ctx"][r].sy / bgrams["n"]).to(torch.float32).cpu() for r in rows}
    for arm in ("ctx", "tok"):
        for r in rows:
            t1 = time.time()
            rmap, diag = M.ridge_gcv_from_grams(bgrams[arm][r], sigma=1.0)
            ridge_boundary[arm][r] = rmap.to("cpu")
            ridge_diag["boundary"].setdefault(arm, {})[C.row_to_block_key(r)] = {
                "best_lam": diag["best_lam"],
                "eigh_seconds": diag["eigh_seconds"],
            }
            eigh_times.append(
                {
                    "arm": f"boundary_{arm}",
                    "row": r,
                    "d": diag["d"],
                    "eigh_s": diag["eigh_seconds"],
                    "total_s": time.time() - t1,
                }
            )
    del bgrams
    summary["eigh_times"] = eigh_times
    summary["ridge_diag"] = ridge_diag

    def _map_state(rm) -> dict:
        return {
            "mu": rm.mu,
            "sd": rm.sd,
            "w": rm.w,
            "bias": rm.bias,
            "best_lam": rm.best_lam,
            "sigma": rm.sigma,
        }

    torch.save(
        {
            "answer": {
                a: {r: _map_state(m) for r, m in d.items()} for a, d in ridge_answer.items()
            },
            "boundary": {
                a: {r: _map_state(m) for r, m in d.items()} for a, d in ridge_boundary.items()
            },
            "delta_train_mean": delta_train_mean,
            "boundary_train_mean": boundary_mean,
            "sigma_by_row": sigma_by_row,
            "rows": rows,
            "lambdas": M.RIDGE_LAMBDAS_922,
            "split_seed": args.split_seed,
            "metadata": C.reproducibility_metadata(
                {"script": "issue922_fit_maps", "kind": "ridge"}
            ),
        },
        args.out / "maps_ridge.pt",
    )
    logger.info("[ridge] saved maps_ridge.pt (%d rows x {3 answer + 2 boundary} arms)", len(rows))

    # ── MLP: 4 batched fits (arm × space), store-fed chunks ───────────────────
    if not args.skip_mlp:
        mlp_out: dict = {}
        for arm in ("ctx", "tok"):
            for space in ("raw", "rmsnorm"):
                t1 = time.time()
                res = M.fit_position_mlps(
                    h,
                    rows,
                    tr["fit"]["answer"],
                    tr["val"]["answer"],
                    arm=arm,
                    space=space,
                    sigma_by_row=sigma_by_row,
                    device=device,
                    seed=C.MLP_INIT_SEED,
                    max_epochs=args.mlp_max_epochs,
                    layer_chunk=args.mlp_layer_chunk,
                )
                mlp_out[(arm, space)] = res
                summary.setdefault("mlp_wall_s", {})[f"{arm}_{space}"] = time.time() - t1
        torch.save(
            {
                "fits": {f"{a}__{s}": v for (a, s), v in mlp_out.items()},
                "rows": rows,
                "sigma_by_row": sigma_by_row,
                "metadata": C.reproducibility_metadata(
                    {"script": "issue922_fit_maps", "kind": "mlp"}
                ),
            },
            args.out / "maps_mlp.pt",
        )
        logger.info("[mlp] saved maps_mlp.pt (4 batched fits x %d rows)", len(rows))

    # ── exploratory position-GRUs (read-out blocks only) ──────────────────────
    if not args.skip_gru:
        gru_rows = [C.block_to_row(b) for b in C.READOUT_BLOCKS if C.block_to_row(b) in rows]
        if args.smoke and not gru_rows:
            gru_rows = rows[-1:]  # smoke stub store: use the deepest fitted row
        gru_out: dict = {}
        pos_lo, n_pos = store["pos_lo"], store["n_pos"]
        fitval = np.concatenate([split["fit"], split["val"]])
        val_flag = torch.zeros(len(fitval), dtype=torch.bool)
        val_flag[len(split["fit"]) :] = True
        max_t = int(n_pos.max()) - 1
        for r in gru_rows:
            seqs = torch.zeros(len(fitval), max_t, H, dtype=torch.float32)
            targ = torch.zeros(len(fitval), max_t, H, dtype=torch.float32)
            mask = torch.zeros(len(fitval), max_t, dtype=torch.float32)
            for j, i in enumerate(fitval):
                lo, npos = int(pos_lo[i]), int(n_pos[i])
                hw = h[r, lo : lo + npos, :].to("cpu", torch.float32)
                seqs[j, : npos - 1] = hw[:-1]
                targ[j, : npos - 1] = hw[1:] - hw[:-1]
                mask[j, : npos - 1] = 1.0
            rm = ridge_answer["ctx"][r]
            t1 = time.time()
            net, diag = M.fit_position_gru(
                seqs,
                targ,
                mask,
                val_flag,
                mu=rm.mu.to(device),
                sd=rm.sd.to(device),
                device=device,
                max_epochs=args.gru_max_epochs,
                seed=C.MLP_INIT_SEED,
            )
            gru_out[r] = {
                "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "mu": rm.mu,
                "sd": rm.sd,
                "diag": diag,
            }
            summary.setdefault("gru_wall_s", {})[C.row_to_block_key(r)] = time.time() - t1
            logger.info("[gru] row=%d %s", r, diag)
        torch.save(
            {
                "grus": gru_out,
                "hidden": 1024,
                "metadata": C.reproducibility_metadata(
                    {"script": "issue922_fit_maps", "kind": "gru"}
                ),
            },
            args.out / "maps_gru.pt",
        )

    summary["total_wall_s"] = time.time() - t_all
    C.write_json_atomic(
        args.out / "fit_summary.json",
        {**summary, "metadata": C.reproducibility_metadata({"script": "issue922_fit_maps"})},
    )
    logger.info("DONE fits in %.1fs", summary["total_wall_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
