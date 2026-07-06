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
    "template_end": ..., "ansrel": {global_src_idx→answer-relative index},
    "answer_T": LongTensor, "answer_ctx": np.ndarray}`` (ansrel: boundary → 0
    is NOT included; answer transition with source t has ansrel =
    t − prompt_len). ``answer_T`` maps each ANSWER transition to the global
    store index of its context's last formatted-prompt position T = P−1 (the
    per-layer conditioning vector c = h_{l,T} gathers through it — plan
    §4.3b); ``answer_ctx`` is the transition's ordinal within ``ctx_sel``
    (the per-context grouping axis for the H6 paired statistic).
    """
    segs_by = {C.SEG_ANSWER: [], C.SEG_BOUNDARY: [], C.SEG_PROMPT: [], C.SEG_TEMPLATE_END: []}
    ansrel: list[tuple[int, int]] = []
    ans_T: list[int] = []
    ans_ctx: list[int] = []
    pos_lo, n_pos = store["pos_lo"], store["n_pos"]
    plen, ws = store["prompt_len"], store["window_start"]
    seg_all = store["segments"]
    for local_i, i in enumerate(ctx_sel):
        lo, npos = int(pos_lo[i]), int(n_pos[i])
        t_row = int(plen[i]) - 1 - int(ws[i])  # in-window row of T (≥ 0 by window arithmetic)
        assert 0 <= t_row < npos, (i, t_row, npos)
        for j in range(npos - 1):
            src = lo + j
            seg = int(seg_all[src])
            segs_by[seg].append(src)
            if seg == C.SEG_ANSWER:
                t_abs = int(ws[i]) + j
                ansrel.append((src, t_abs - int(plen[i])))
                ans_T.append(lo + t_row)
                ans_ctx.append(local_i)
    out = {C.SEG_NAMES[k]: torch.tensor(v, dtype=torch.long) for k, v in segs_by.items()}
    out["ansrel"] = dict(ansrel)
    out["answer_T"] = torch.tensor(ans_T, dtype=torch.long)
    out["answer_ctx"] = np.array(ans_ctx, dtype=np.int64)
    return out


def resolve_fit_rows(blocks: list, n_store_rows: int) -> list[int]:
    """Store-row indices for a --blocks list; the EMBEDDING row 0 is INCLUDED.

    The r1 implementation dropped row 0 from every fit (the Codex MAJOR /
    plan §4.3b layer-0 binding); layer 0 is the H2 sanity anchor and IS in
    the DV1 atlas — 29 rows everywhere the arm is fit.
    """
    rows = [C.block_to_row(b) for b in blocks]
    assert rows, "need at least one block to fit"
    assert all(0 <= r < n_store_rows for r in rows), (rows, n_store_rows)
    assert len(set(rows)) == len(rows), f"duplicate blocks: {blocks}"
    return rows


def resolve_conditioned_rows(spec: str, rows: list[int]) -> list[int]:
    """Store rows for --conditioned-blocks ('emb,5,10,...' — block convention).

    Restricted to rows actually fitted this run (the smoke fits 2 rows). The
    plan §10 spells the flag ``--conditioned-rows 0,5,...`` with 0 = the
    layer-0 anchor; 'emb' names that row unambiguously here (a bare '0'
    still means BLOCK 0 = store row 1).
    """
    want = [C.block_to_row(b if b == "emb" else int(b)) for b in spec.split(",") if b]
    return [r for r in want if r in rows]


def main() -> int:  # noqa: C901 — the phase sequence IS the fit spec (grams/ridge/b1/boundary/mlp/gru/conditioned/direct)
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
    ap.add_argument(
        "--gram-row-chunk",
        type=int,
        default=8,
        help="rows per Gram accumulation pass (bounds the fp64 block footprint)",
    )
    # ── v6 amendment arms (plan §4.3b) ────────────────────────────────────────
    ap.add_argument("--b1-ridge", action="store_true", help="closed-form b1 [h,c] ridge, all rows")
    ap.add_argument(
        "--conditioned-forms",
        default="",
        help="comma subset of b1_grad,film,lowrank,mixture (gradient fits)",
    )
    ap.add_argument(
        "--conditioned-blocks",
        default="emb,5,10,14,17,19,20,24,26",
        help="block-convention subset for the gradient fits ('emb' = the layer-0 anchor; "
        "the plan §10 spells this --conditioned-rows 0,5,... with 0 meaning that anchor)",
    )
    ap.add_argument("--cond-max-epochs", type=int, default=None, help="override recipe max_epochs")
    ap.add_argument("--cond-batch-size", type=int, default=None)
    ap.add_argument("--cond-layer-chunk", type=int, default=1)
    ap.add_argument(
        "--direct-horizons", type=int, default=0, help="arm c: fit c→D_k for k=1..N (0 = off)"
    )
    ap.add_argument("--direct-k-chunk", type=int, default=8)
    ap.add_argument("--verify-fits", action="store_true", help="run equivalence gates and exit")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = _resolve_device(args.device)

    if args.verify_fits:
        # Non-hollow by construction: each gate CALLS the exact callable the
        # phases below dispatch (ridge_gcv_from_grams / fit_batched_split_mlp /
        # apply_conditioned_delta / accumulate_grams_b1 / the batched direct-
        # horizon GCV) — see each verify_* docstring for the dispatch link.
        g1 = M.verify_ridge_gcv_against_dual()
        g2 = assert_split_mlp_matches_serial()
        g3 = M.verify_conditioned_forms()
        g4 = M.verify_b1_gram_assembly()
        g5 = M.verify_direct_horizon_gcv()
        logger.info(
            "[verify-fits] ridge %s | split-MLP %s | conditioned %s | b1-gram %s | direct %s",
            g1,
            g2,
            g3,
            g4,
            g5,
        )
        print(
            json.dumps(
                {
                    "ridge_gate": g1,
                    "split_mlp_gate": g2,
                    "conditioned_forms_gate": g3,
                    "b1_gram_gate": g4,
                    "direct_horizon_gate": g5,
                }
            )
        )
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    store = C.load_store(args.store, "lmsys")
    R, P, H = store["h"].shape
    n_ctx = len(store["ctx_ids"])
    logger.info("[store] R=%d P=%d H=%d n_ctx=%d", R, P, H, n_ctx)

    blocks = (
        [b if b == "emb" else int(b) for b in args.blocks.split(",")]
        if args.blocks
        else store["blocks"][:1] + [int(x) for x in store["blocks"][1:]]
    )
    rows = resolve_fit_rows(blocks, R)
    row_chunk = max(1, min(args.gram_row_chunk, len(rows)))

    # Device placement (the r1 CRITICAL fix): budget the store move AGAINST the
    # Gram-assembly footprint, not a flat headroom constant. Three regimes:
    # store+grams fit → store to GPU; grams alone fit → store stays CPU, grams
    # on GPU (chunks stream over PCIe); neither → grams on CPU (slow, works —
    # the <60 GB HBM degrade-safely path the A100-40 rung needs).
    h = store["h"]
    gram_device = device
    if device == "cuda":
        free_b, _ = torch.cuda.mem_get_info()
        need, grams_only = M.fit_phase_gpu_budget_bytes(h.numel() * 2, H, row_chunk)
        if free_b > need:
            h = h.to("cuda")
            logger.info(
                "[hbm] store → GPU (store %.1f GB + gram budget %.1f GB < free %.1f GB)",
                h.numel() * 2 / 1e9,
                (need - h.numel() * 2) / 1e9,
                free_b / 1e9,
            )
        elif free_b > grams_only:
            logger.warning(
                "[hbm] store (%.1f GB) stays CPU-resident: need %.1f GB > free %.1f GB; "
                "Gram accumulation on GPU (row_chunk=%d, %.1f GB)",
                h.numel() * 2 / 1e9,
                need / 1e9,
                free_b / 1e9,
                row_chunk,
                grams_only / 1e9,
            )
        else:
            gram_device = "cpu"
            logger.warning(
                "[hbm] gram budget %.1f GB > free %.1f GB — Gram accumulation on CPU (SLOW)",
                grams_only / 1e9,
                free_b / 1e9,
            )

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

    summary: dict = {
        "ns": ns,
        "rows": rows,
        "blocks": [C.row_to_block_key(r) for r in rows],
        "gram_row_chunk": row_chunk,
        "gram_device": gram_device,
    }
    t_all = time.time()

    # ── answer-segment + b1 + boundary Grams + ridge, row-chunked ─────────────
    # Each chunk: ONE answer streaming pass (3 arms' blocks) → fits → the b1
    # SECOND pass (c-blocks; reuses the live ctx stats — plan §4.3b) → fits →
    # ONE boundary pass → fits → free. Bounds device fp64 to the budget above.
    sigma_by_row: dict = {}
    delta_train_mean: dict = {}
    boundary_mean: dict = {}
    ridge_answer: dict = {"ctx": {}, "tok": {}, "emb": {}}
    ridge_b1: dict = {}
    ridge_boundary: dict = {"ctx": {}, "tok": {}}
    ridge_diag: dict = {"answer": {}, "boundary": {}, "b1_answer": {}}
    eigh_times: list[dict] = []
    shared_emb_eig = None
    gram_wall = {"answer": 0.0, "b1": 0.0, "boundary": 0.0}
    for lo in range(0, len(rows), row_chunk):
        chunk_rows = rows[lo : lo + row_chunk]
        t0 = time.time()
        grams = M.accumulate_grams(
            h, tr["fit"]["answer"], chunk_rows, emb_row=0, chunk=args.gram_chunk, device=gram_device
        )
        gram_wall["answer"] += time.time() - t0
        for r in chunk_rows:
            sigma_by_row[r] = float(np.sqrt(grams["ctx"][r].syy / (grams["n"] * H)))
            delta_train_mean[r] = (grams["ctx"][r].sy / grams["n"]).to(torch.float32).cpu()
        for arm in ("ctx", "tok", "emb"):
            for r in chunk_rows:
                stats = grams[arm][r]
                t1 = time.time()
                if arm == "emb" and shared_emb_eig is not None:
                    rmap, diag = M.ridge_gcv_from_grams(stats, sigma=1.0, eig=shared_emb_eig)
                else:
                    rmap, diag = M.ridge_gcv_from_grams(stats, sigma=1.0)
                if arm == "emb" and shared_emb_eig is None:
                    shared_emb_eig = diag["eig"]
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
        if args.b1_ridge:
            t0 = time.time()
            b1_blocks = M.accumulate_grams_b1(
                h,
                tr["fit"]["answer"],
                tr["fit"]["answer_T"],
                chunk_rows,
                chunk=args.gram_chunk,
                device=gram_device,
            )
            gram_wall["b1"] += time.time() - t0
            for r in chunk_rows:  # assemble → fit → free, one row at a time
                t1 = time.time()
                stats = M.assemble_b1_stats(grams["ctx"][r], b1_blocks.pop(r))
                rmap, diag = M.ridge_gcv_from_grams(stats, sigma=1.0)
                del stats
                ridge_b1[r] = rmap.to("cpu")
                ridge_diag["b1_answer"][C.row_to_block_key(r)] = {
                    "best_lam": diag["best_lam"],
                    "eigh_seconds": diag["eigh_seconds"],
                }
                eigh_times.append(
                    {
                        "arm": "b1_answer",
                        "row": r,
                        "d": diag["d"],
                        "eigh_s": diag["eigh_seconds"],
                        "total_s": time.time() - t1,
                    }
                )
            del b1_blocks
        del grams
        t0 = time.time()
        bgrams = M.accumulate_grams(
            h,
            tr["fit"]["boundary"],
            chunk_rows,
            emb_row=0,
            chunk=args.gram_chunk,
            device=gram_device,
        )
        gram_wall["boundary"] += time.time() - t0
        for r in chunk_rows:
            boundary_mean[r] = (bgrams["ctx"][r].sy / bgrams["n"]).to(torch.float32).cpu()
        for arm in ("ctx", "tok"):
            for r in chunk_rows:
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
    summary["gram_wall_s"] = gram_wall
    summary["sigma_by_row"] = {C.row_to_block_key(r): sigma_by_row[r] for r in rows}
    summary["eigh_times"] = eigh_times
    summary["ridge_diag"] = ridge_diag
    assert 0 in rows or "emb" not in [str(b) for b in blocks], rows
    if "emb" in [str(b) for b in blocks]:
        assert 0 in ridge_answer["ctx"], "layer-0 (emb) row missing from fits (r1 blocker)"

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
            "b1_answer": {r: _map_state(m) for r, m in ridge_b1.items()},
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
    logger.info(
        "[ridge] saved maps_ridge.pt (%d rows x {3 answer + 2 boundary} arms + %d b1 rows)",
        len(rows),
        len(ridge_b1),
    )

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

    # ── v6: conditioned gradient fits (b1_grad + 3 b2 forms, ONE entry point) ─
    cond_forms = [f for f in args.conditioned_forms.split(",") if f]
    if cond_forms:
        cond_rows = resolve_conditioned_rows(args.conditioned_blocks, rows)
        assert cond_rows, (args.conditioned_blocks, rows)
        recipe: dict = {}
        if args.cond_max_epochs is not None:
            recipe["max_epochs"] = args.cond_max_epochs
        if args.cond_batch_size is not None:
            recipe["batch_size"] = args.cond_batch_size
        cond_out: dict = {}
        for form in cond_forms:
            assert form in M.CONDITIONED_FORMS, form
            t1 = time.time()
            cond_out[form] = M.fit_conditioned_linear(
                h,
                cond_rows,
                tr["fit"]["answer"],
                tr["val"]["answer"],
                tr["fit"]["answer_T"],
                tr["val"]["answer_T"],
                form=form,
                device=device,
                seed=C.MLP_INIT_SEED,
                recipe=recipe,
                layer_chunk=args.cond_layer_chunk,
            )
            summary.setdefault("cond_wall_s", {})[form] = time.time() - t1
        torch.save(
            {
                "forms": cond_out,
                "rows": cond_rows,
                "blocks": [C.row_to_block_key(r) for r in cond_rows],
                "rank": M.LOWRANK_RANK_922,
                "n_mix": M.MIXTURE_K_922,
                "recipe": {**M.CONDITIONED_RECIPE_922, **recipe},
                "capacity": {
                    form: cond_out[form][cond_rows[0]]["n_params_weights"] for form in cond_forms
                },
                "metadata": C.reproducibility_metadata(
                    {"script": "issue922_fit_maps", "kind": "conditioned"}
                ),
            },
            args.out / "maps_conditioned.pt",
        )
        summary["cond_best_val_epochs"] = {
            form: {C.row_to_block_key(r): cond_out[form][r]["best_val_epoch"] for r in cond_rows}
            for form in cond_forms
        }
        logger.info(
            "[cond] saved maps_conditioned.pt (%d forms x %d rows)", len(cond_forms), len(cond_rows)
        )

    # ── v6: arm c — direct per-horizon maps, per-row files (checkpoint-per-row)
    if args.direct_horizons > 0:
        k_max = args.direct_horizons
        ddir = args.out / "direct"
        ddir.mkdir(parents=True, exist_ok=True)
        pos_lo, n_pos = store["pos_lo"], store["n_pos"]
        plen, ws = store["prompt_len"], store["window_start"]
        fit_ctx = [i for i in split["fit"] if int(store["ans_len"][i]) >= 1]
        T_pos = torch.tensor(
            [int(pos_lo[i]) + int(plen[i]) - 1 - int(ws[i]) for i in fit_ctx], dtype=torch.long
        )
        kcap = np.array([int(ws[i]) + int(n_pos[i]) - int(plen[i]) for i in fit_ctx])
        win = store.get("window") or {}
        regime = {
            "k_max": k_max,
            "n_fit_ctx": len(fit_ctx),
            "split_seed": args.split_seed,
            "n_store_ctx": n_ctx,
            # r2 review minor: the capture window is output-affecting (T_pos /
            # kcap derive from prompt_len / window_start under (wp, wa)), so it
            # is part of the resume key — a direct_row fitted from a different-
            # window store refits instead of silently reusing.
            "wp": win.get("wp"),
            "wa": win.get("wa"),
        }
        vb = tr["val"]["boundary"]
        coherence: dict = {}
        direct_diag: dict = {}
        t_direct = time.time()
        for r in rows:
            bk = C.row_to_block_key(r)
            fpath = ddir / f"direct_row_{r:02d}.pt"
            if fpath.exists():
                blob = torch.load(fpath, weights_only=False)
                if blob.get("regime") == regime and blob.get("row") == r:
                    logger.info("[direct] row %d exists with matching regime — skip (resume)", r)
                    coherence[bk] = blob.get("coherence")
                    direct_diag[bk] = blob.get("diag")
                    continue
                logger.warning("[direct] row %d regime mismatch — refitting", r)
            t1 = time.time()
            res = M.fit_direct_horizon_maps(
                h, r, T_pos, kcap, k_max=k_max, device=device, k_chunk=args.direct_k_chunk
            )
            # §4.3b coherence check: arm-c k=1 ≡ the v3 boundary ctx map (same
            # regression problem) — compared on the VAL boundary transitions.
            co = None
            if 1 in res["maps"] and vb.numel() > 0 and r in ridge_boundary["ctx"]:
                Xv = h[r, vb, :].to("cpu", torch.float32)
                Yv = (h[r, vb + 1, :].to("cpu", torch.float32) - Xv).numpy()
                r2_dir = M.identity_relative_r2(M.ridge_predict(res["maps"][1], Xv).numpy(), Yv)
                r2_bnd = M.identity_relative_r2(
                    M.ridge_predict(ridge_boundary["ctx"][r], Xv).numpy(), Yv
                )
                lam_eq = bool(res["maps"][1].best_lam == ridge_boundary["ctx"][r].best_lam)
                tol = 1e-3 if lam_eq else 0.02
                co = {
                    "r2_direct_k1": r2_dir,
                    "r2_boundary": r2_bnd,
                    "lam_equal": lam_eq,
                    "abs_diff": abs(r2_dir - r2_bnd),
                    "tol": tol,
                }
                if abs(r2_dir - r2_bnd) > tol:
                    raise RuntimeError(
                        f"arm-c k=1 vs boundary-map coherence FAILED at row {r}: {co} — "
                        "indexing bug in one of the two paths (plan §12.20; failure_class: code)"
                    )
            coherence[bk] = co
            torch.save(
                {
                    "row": r,
                    "block": bk,
                    # w fp16 on disk (29 rows × 40 maps ≈ 30 GB at H=3584; the
                    # eval loader casts back to fp32 — plan §10 fp16 pricing)
                    "maps": {k: {**_map_state(m), "w": m.w.half()} for k, m in res["maps"].items()},
                    "diag": res["diag"],
                    "regime": regime,
                    "coherence": co,
                    "wall_s": time.time() - t1,
                    "metadata": C.reproducibility_metadata(
                        {"script": "issue922_fit_maps", "kind": "direct", "row": r}
                    ),
                },
                fpath,
            )
            direct_diag[bk] = res["diag"]
            logger.info(
                "[direct] row %d: %d/%d horizons fit in %.1fs (coherence %s)",
                r,
                len(res["maps"]),
                k_max,
                time.time() - t1,
                "ok" if co else "n/a",
            )
        summary["direct_wall_s"] = time.time() - t_direct
        summary["direct_regime"] = regime
        summary["coherence_direct_vs_boundary"] = coherence
        C.write_json_atomic(
            args.out / "direct_diag.json",
            {
                "regime": regime,
                "coherence": coherence,
                "diag_by_block": direct_diag,
                "metadata": C.reproducibility_metadata(
                    {"script": "issue922_fit_maps", "kind": "direct_diag"}
                ),
            },
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
