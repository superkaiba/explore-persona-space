"""Issue #2569 P-B row battery (units 4a + 4b).

Phase driver over the assembled X19/Y19 row store (plan v4 §4 legs 1/2 moments +
§4 leg 4 + §4 leg 8 step 2). Phases:

  assemble   P1 assembly REUSED VERBATIM from the ported ``issue2476_turnavg_sae``
             (streams the 1,920 banked capture chunks at the #2476 lineage pin
             ``DATA_REPO_REVISION``, per-chunk unlink, cursor-checkpointed resume,
             ``--max-chunks`` smoke seam) -> X19/Y19 fp16 memmaps over the parent
             row space (964,844 rows at production) + rows_present + split_meta.
  moments    ONE streaming fp64 pass over the map-training pool (present rows
             minus the sha-pinned 400-val/1,000-test) -> uncentered Grams
             Sigma_xx / Sigma_xy / Sigma_yy + means, PER SPLIT-HALF (two disjoint
             halves by conversation index, seed 0 via
             ``issue2569_leg6.split_halves_by_conversation``) and POOLED (halves
             are additive), plus the two split-half ridge refits at the banked
             map's pinned lambda (Gram-space solves via ``issue2569_leg6.ridge_map``,
             ``fit_ridge_primal`` standardize-X / center-Y convention). Emits the
             unit-2 Sigma producer contract consumed by
             ``issue2569_gateladder.load_sigma_file`` (``--sigma-c`` <- gram_xx.pt,
             ``--sigma-a`` <- gram_yy.pt: ``{"gram","mean","n_rows"}`` uncentered
             sum-of-outer schema), the cross-moment gram_xy.pt (NOT a sigma file),
             and splithalf_maps.pt.
  sae-train  Context-side matryoshka BatchTopK SAE on the X19 rows at the EXACT
             #2476 recipe (width 65,536, k=100, tier bounds 2,048/16,384/65,536,
             Adam(0.9,0.999) lr 2e-4, batch 256, 3 epochs, threshold EMA 0.999,
             init seed 2476, carve 933,444/10,000 via ``_sae_row_positions``);
             halt floor G4: validation var-FVE >= 0.5 (rc=RC_G4). Deliverables
             ae.pt + config.json + alive_union.json (+ train_log.json) -> HF
             ``<hf_prefix>/sae_ctx/``. The k=200 twin is OUT of scope (no --sae-k
             here; the reused T24 namespace keeps its default k=100).
  feature-map  Leg-4 steps 3-6 (unit 4b): alive-context-feature -> banked-alive-
             answer-union ridge map (val-selected lambda over the widened 27-value
             grid, C4 widen-on-edge; n = 120,000 >> d, no GCV), scored on the 20k
             holdout with the SIX comparison routes (fitted map / banked #2476
             encode-of-mapped-prediction / banked dense-input ridge / index-aligned
             identity+bias null / train-mean null / 20-draw row-shuffle null) and
             the hurdle decomposition (firing AUROC + conditional-magnitude R2,
             NEVER mixed with the unconditional R2), P/R@k at the realized L0,
             kNN retrieval (euclidean + cosine, chance stated), and the #2476
             ctx-alive floor sweep (1%/0.5%/0.25%/0.2%).
  mine       Leg-8 step 2 (unit 4b): chunked-GEMM kernel-pair mining over random
             conversation pairs (kernel fraction ||dc @ A|| / ||dc||; B1 assert
             iii runs as a REAL probe-batch assert against the registered
             prediction difference), stratified distance-matched controls
             (corpus-source x answer-length-decile strata), realized ||dv_A||
             read with the PINNED estimator (median of paired ratios; the
             ratio-of-medians companion), rank tests, a 10,000-draw clustered
             bootstrap CI (conversation-overlap components; the SAME estimator
             recomputed per draw), and the C2 measured held-out residual-pair
             noise floor (val+test rows).

Driver entry runs the B1 identity asserts on the banked L19 map payload
(``issue2569_operator.run_driver_identity_asserts``) — a raise HALTS the driver
(apply-path breakage class). Out-of-scope for THIS file per the unit split: the
H2b refit series (gateladder machinery exists; a follow-up unit wires it).

Smoke blind-spot mirror (unit 4b phases): the ``--answer-sae-dir`` /
``--alive-counts-npz`` / ``--manifest-dir`` overrides substitute the INPUT SOURCE
for the HF staging legs (same parse/consume path either way); the HF download
branches themselves (``hub.retry_transient``-wrapped) are exercised by the pod
smoke + production, not by the VM unit tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

import issue779_common as C  # noqa: E402
import issue2476_turnavg_sae as T24  # noqa: E402
import issue2569_leg6 as L6  # noqa: E402
import issue2569_operator as OP  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2569.rowbattery")

TASK_ID = 2569
LAYER_DEFAULT = 19  # the banked n1m map's native layer (plan §11)
N_TOTAL_PRODUCTION = 964_844  # assembled parent row space (plan Blocker-4: NEVER 963,444)
MOMENT_CHUNK_DEFAULT = 65_536  # streaming Gram chunk rows (fp64 GEMM per chunk)

# The moments pool mirrors the banked map's own training convention: every present
# row EXCEPT the sha-pinned 400-val/1,000-test rows (n_train = 963,444 at
# production) — Sigma_c must model the estimator actually run (GL.pooled_moments
# docstring; the SAE carve additionally excludes the 20k holdout, via
# T24._sae_row_positions, unchanged).
N_MOMENT_POOL_PRODUCTION = N_TOTAL_PRODUCTION - 400 - 1_000

assert L6.SPLIT_SEED == 0, "plan §4 leg 1 step 7 pins the split-half seed to 0"

# ── unit-4b constants (leg 4 feature map + leg 8 mining) ──────────────────────────
# Banked #2476 answer-SAE artifacts (schema probed 2026-08-25, epm:progress v24):
# alive_c.npz files = [alive_ids, counts, floor, n_fit_rows, train_mean, tier];
# counts (65536,) int64 over the 120,000 SAE-fit rows. The plan's "banked alive
# answer union (2,150)" = features with counts >= ceil(0.002 * 120,000) = 240
# (verified: 1% -> 879, 0.5% -> 1,332, 0.25% -> 1,913, 0.2% -> 2,150).
ANSWER_SAE_HF_LEAF = "issue2476_turnavg/analysis_tensors/sae_c"
ALIVE_C_HF_PATH = "issue2476_turnavg/analysis_tensors/eval/alive_c.npz"
BANKED_2476_DIR = PROJECT_ROOT / "eval_results" / "issue_2476" / "turnavg"
LEG4_UNION_FRAC = 0.002  # loosest #2476 sweep floor == the union (floors are nested)
LEG4_UNION_EXPECTED = 2_150  # production pin (banked artifact is deterministic)
LEG4_CTX_FLOOR_FRACS = (0.010, 0.005, 0.0025, 0.002)  # 1% primary + #2476 sweep
LEG4_CTX_ALIVE_CAP = 16_384  # halt-investigate guard on the ctx alive-input width
LEG8_PAIRS_DEFAULT = 20_000_000  # plan §4 leg 8 step 2 (ungrounded — smoke tunes)
LEG8_CHUNK_PAIRS = 524_288  # ~512k-pair GEMM chunks (plan §9)
LEG8_TOP_PAIRS_DEFAULT = 1_000
LEG8_BOOT_DRAWS_DEFAULT = 10_000
LEG8_SEED = 2_569_800  # chunk k uses default_rng((LEG8_SEED, k)) — resume-stable
LEG8_MIN_KERNEL_PRODUCTION = 300  # pre-registered mining-abort floor
LEG8_CTRL_TOL_LADDER = (0.02, 0.04, 0.08)  # pre-registered ||dc|| match widening
RC_MINE_ABORT = 27  # leg-8 mining-abort HALT (T24 convention: 22-26 taken)


# ── small utils ──────────────────────────────────────────────────────────────────


def _sentinel(name: str, note: str, extra: dict | None = None) -> None:
    """Non-blocking phase sentinel under THIS task's id (poller-parseable)."""
    payload = {"blocks_pipeline": False}
    if extra:
        payload.update(extra)
    try:
        C.write_sentinel(f"phase-{name}", note, task_id=TASK_ID, extra=payload)
    except OSError as e:
        logger.warning("[sentinel] phase-%s write failed: %s", name, e)


def _t24_args(args) -> argparse.Namespace:
    """Compose the reused #2476 namespace through ITS OWN argparse defaults.

    Built via ``T24._parse_args`` (never a hand-rolled Namespace) so every field
    the reused phase bodies dereference exists with the module's own default —
    the reused-module Namespace contract (gotchas.md). ``args.device`` must
    already be resolved (never "auto"): T24 phase bodies read it directly.
    """
    assert args.device in ("cuda", "cpu"), f"device must be resolved, got {args.device!r}"
    argv = [
        "--phase",
        "assemble",
        "--out-root",
        str(args.out_root),
        "--hf-prefix",
        str(args.hf_prefix),
        "--max-chunks",
        str(int(args.max_chunks)),
        "--smoke-rows",
        str(int(args.smoke_rows)),
        "--device",
        args.device,
        "--sae-dict",
        str(int(args.sae_dict)),
        "--sae-steps",
        str(int(args.sae_steps)),
    ]
    if args.smoke:
        argv.append("--smoke")
    if args.fresh_stream:
        argv.append("--fresh-stream")
    if args.skip_upload:
        argv.append("--skip-upload")
    if args.resume_across_code_sha:
        argv.append("--resume-across-code-sha")
    t24 = T24._parse_args(argv)
    if t24.sae_dir is None:  # mirror T24.main's post-parse resolution
        t24.sae_dir = t24.out_root / "sae_cache"
    return t24


def _atomic_torch_save(obj: dict, path: Path) -> None:
    """torch.save through the shared process-unique atomic-replace primitive
    (#2336: a fixed ``.tmp`` sibling name is a concurrent-writer clobber)."""
    with atomic_replace(path) as tmp:
        torch.save(obj, tmp)


def _atomic_np_save(arr: np.ndarray, path: Path) -> None:
    """np.save through atomic_replace (open handle: np.save must not append
    ``.npy`` to the process-unique tmp name — only string paths get suffixed)."""
    with atomic_replace(path) as tmp, open(tmp, "wb") as fh:
        np.save(fh, arr)


def _atomic_npz_save(path: Path, **arrays) -> None:
    """np.savez through atomic_replace (open handle — see _atomic_np_save)."""
    with atomic_replace(path) as tmp, open(tmp, "wb") as fh:
        np.savez(fh, **arrays)


def _upload_leaf(args, files: list[Path], leaf: str, *, resume_skip: bool) -> None:
    """Production-only HF upload of ``files`` to ``<hf_prefix>/<leaf>/`` with the
    fail-loud exact-set verify (mirrors T24._p4_upload; skip is LOUD)."""
    t24 = _t24_args(args)
    if args.skip_upload or not T24._production(t24):
        logger.warning("[%s] skip_upload/non-production: HF upload SKIPPED (loud)", leaf)
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    up = T24._stage_dir(t24) / f"{leaf}_upload_2569"
    if up.exists():
        shutil.rmtree(up)
    up.mkdir(parents=True, exist_ok=True)
    for f in files:
        assert f.is_file(), f"[{leaf}] upload source missing/not-a-file: {f}"
        shutil.copy2(f, up / f.name)
    prefix = f"{args.hf_prefix}/{leaf}"
    res = upload_dir_sharded(
        up,
        C.HF_DATA_REPO,
        prefix,
        repo_type="dataset",
        shard_glob="*",
        verify=True,
        delete_local=False,
        resume_skip=resume_skip,
    )
    if not res.rerouted:  # fail-loud exact-set verify (the T24 P3/P4 pattern)
        expected = [f"{prefix}/{p.name}" for p in sorted(up.iterdir()) if p.is_file()]
        missing = hub.verify_repo_paths_uploaded(
            HfApi(), C.HF_DATA_REPO, expected, path_in_repo=prefix
        )
        assert not missing, f"[{leaf}] verify FAILED — missing on Hub: {missing}"
    logger.info("[%s] uploaded -> %s (rerouted=%s)", leaf, prefix, res.rerouted)


# ── moments helpers (pure; unit-tested on tiny synthetic shapes) ─────────────────


def _conversation_keys(row_ci: np.ndarray, pool_ids: np.ndarray) -> list[str]:
    """Split-half grouping keys for the pool rows (plan §4 leg 1 step 7).

    New captured rows carry a non-negative conversation index -> rows of one
    conversation share ``ci<idx>`` and never straddle halves. pass_b seed-corpus
    rows carry ci == -1 (one row per seed conversation) -> each gets a UNIQUE
    ``pb<global_row_id>`` key, so the greedy balancer treats them as singleton
    conversations instead of one giant pseudo-conversation.
    """
    cis = row_ci[pool_ids]
    return [
        f"ci{int(ci)}" if int(ci) >= 0 else f"pb{int(g)}"
        for ci, g in zip(cis, pool_ids, strict=True)
    ]


def _accumulate_moments(x_mm, y_mm, positions: np.ndarray, *, chunk: int, dev, tag: str) -> dict:
    """Streamed fp64 raw moments of the rows at ``positions`` (memmap row space).

    ONE pass: per-dim sums plus uncentered sum-of-outer Grams X^T X, X^T Y, Y^T Y
    (fp64 accumulators on ``dev``; returned on CPU). Sums/Grams are ADDITIVE
    across disjoint row sets, so the pooled moments are the two halves' sums —
    no second pass. Asserts n >= 2 and the shared hidden width.
    """
    pos = np.sort(np.asarray(positions, dtype=np.int64))
    n = int(len(pos))
    assert n >= 2, f"[moments:{tag}] need >= 2 rows, got {n}"
    d, dy = int(x_mm.shape[1]), int(y_mm.shape[1])
    sum_x = torch.zeros(d, dtype=torch.float64, device=dev)
    sum_y = torch.zeros(dy, dtype=torch.float64, device=dev)
    gxx = torch.zeros((d, d), dtype=torch.float64, device=dev)
    gxy = torch.zeros((d, dy), dtype=torch.float64, device=dev)
    gyy = torch.zeros((dy, dy), dtype=torch.float64, device=dev)
    n_chunks = math.ceil(n / chunk)
    t0 = time.time()
    for k, lo in enumerate(range(0, n, chunk)):
        ix = pos[lo : lo + chunk]
        xb = torch.as_tensor(np.asarray(x_mm[ix]), dtype=torch.float64, device=dev)
        yb = torch.as_tensor(np.asarray(y_mm[ix]), dtype=torch.float64, device=dev)
        sum_x += xb.sum(0)
        sum_y += yb.sum(0)
        gxx += xb.T @ xb
        gxy += xb.T @ yb
        gyy += yb.T @ yb
        print(
            f"[moments] {tag} chunk {k + 1}/{n_chunks} rows={len(ix)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return {
        "sum_x": sum_x.cpu(),
        "sum_y": sum_y.cpu(),
        "gram_xx": gxx.cpu(),
        "gram_xy": gxy.cpu(),
        "gram_yy": gyy.cpu(),
        "n": n,
    }


def _combine_moments(a: dict, b: dict) -> dict:
    """Pooled moments of two DISJOINT halves (raw sums/Grams are additive)."""
    return {
        "sum_x": a["sum_x"] + b["sum_x"],
        "sum_y": a["sum_y"] + b["sum_y"],
        "gram_xx": a["gram_xx"] + b["gram_xx"],
        "gram_xy": a["gram_xy"] + b["gram_xy"],
        "gram_yy": a["gram_yy"] + b["gram_yy"],
        "n": int(a["n"]) + int(b["n"]),
    }


def _half_ridge_refit(acc: dict, lam_abs: float) -> dict:
    """Within-half ridge refit at the banked map's pinned lambda (plan §4 step 7).

    Reconstructs the ``fit_ridge_primal`` estimator from raw moments: X is
    standardized at the half's own mean and UNBIASED std (+1e-9, matching
    ``_ridge_primal_multi_lambda``'s ``torch.std``), Y centered at the half mean;
    W = (X_std^T X_std + lam I)^{-1} X_std^T Y_c via ``issue2569_leg6.ridge_map``
    (batched solve + logged pinv fallback on a singular system). Sum-form Gram +
    ABSOLUTE lambda — the banked payload's ``selected_lambda`` convention.
    """
    n = int(acc["n"])
    assert n >= 2, f"ridge refit needs >= 2 rows, got {n}"
    xmu = acc["sum_x"] / n
    ymu = acc["sum_y"] / n
    ss = torch.clamp(torch.diagonal(acc["gram_xx"]) - n * xmu**2, min=0.0)
    xsd = torch.sqrt(ss / (n - 1)) + 1e-9
    gxx_c = acc["gram_xx"] - n * torch.outer(xmu, xmu)
    gxy_c = acc["gram_xy"] - n * torch.outer(xmu, ymu)
    sxx_std = gxx_c / torch.outer(xsd, xsd)
    sxy_std = gxy_c / xsd[:, None]
    w = L6.ridge_map(sxx_std, sxy_std, float(lam_abs))
    return {
        "W": w,
        "xmu": xmu,
        "xsd": xsd,
        "ymu": ymu,
        "selected_lambda": float(lam_abs),
        "n_rows": n,
    }


def _write_sigma_pt(path: Path, gram: torch.Tensor, mean: torch.Tensor, n_rows: int, **meta):
    """Emit one unit-2 sigma producer file (``issue2569_gateladder.load_sigma_file``
    contract): ``{"gram": (d,d) fp64 uncentered sum-of-outer, "mean": (d,),
    "n_rows": int}`` plus inert string metadata (the loader ignores extra keys;
    ``sigma`` must NOT appear — the loader prefers it over the gram triple)."""
    assert "sigma" not in meta, "meta key 'sigma' would shadow the gram triple"
    obj = {
        "gram": gram.to(torch.float64),
        "mean": mean.to(torch.float64),
        "n_rows": int(n_rows),
        **meta,
    }
    _atomic_torch_save(obj, path)


# ── SAE training core (X19 context side; exact #2476 recipe) ─────────────────────


def _run_sae_training(
    x_mm,
    tr_pos: np.ndarray,
    val_pos: np.ndarray,
    *,
    width: int,
    dev: str,
    steps_cap: int,
    ckpt_path: Path,
    resume_ok: bool,
) -> tuple[torch.nn.Module, list[dict], np.ndarray, int]:
    """The #2476 P4 training loop retargeted at the X19 (context) rows.

    Recipe constants come from the reused module verbatim (SAE_SEED/LR/BETAS/
    BATCH/EPOCHS/K/THRESH_EMA via class defaults + T24 module constants); the
    matryoshka model, block-shuffled memmap loader, and var-FVE eval are the
    reused T24 implementations. Per-epoch checkpoints to ``ckpt_path`` (model +
    opt + fired-union + log rows); a regime-matched resume continues from the
    recorded epoch; a steps-capped break marks the epoch PARTIAL (never counted
    done). Returns (model, epoch_rows, fired_union bool (dict_size,), step).
    """
    model = T24.MatryoshkaBatchTopKSAE(
        act_dim=int(x_mm.shape[1]),  # == C.EXPECTED_HIDDEN at production (phase-asserted)
        dict_size=width,
        tier_bounds=T24._sae_tier_bounds(width),
        k=T24.SAE_K,
    ).to(dev)
    # b_dec init: seeded train-subsample mean (streamed fp64; mirrors T24 P4)
    rng0 = np.random.default_rng(T24.SAE_SEED + 1)
    sub = np.sort(rng0.choice(tr_pos, size=min(65_536, len(tr_pos)), replace=False))
    mu = np.zeros(model.act_dim, dtype=np.float64)
    for s in range(0, len(sub), 8192):
        mu += np.asarray(x_mm[sub[s : s + 8192]], np.float64).sum(0)
    with torch.no_grad():
        model.b_dec.copy_(torch.as_tensor(mu / len(sub), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=T24.SAE_LR, betas=T24.SAE_ADAM_BETAS)
    start_epoch, step = 0, 0
    epoch_rows: list[dict] = []
    fired_union = torch.zeros(model.dict_size, dtype=torch.bool, device=dev)
    if resume_ok and ckpt_path.exists():
        # self-produced local checkpoint (regime-matched dir) — weights_only=False
        # is the sanctioned posture for sha-pinned self-produced bundles (gotchas.md)
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = int(ck["epoch_done"]), int(ck["step"])
        epoch_rows = list(ck["log_rows"])
        fired_union = torch.as_tensor(np.asarray(ck["fired_union"], dtype=bool), device=dev).clone()
        if bool(ck.get("steps_capped")) and steps_cap and step >= steps_cap:
            # a steps-capped PARTIAL epoch is not a done epoch — a same-regime
            # resume treats the capped budget as training-complete (T24 g2 M-2)
            start_epoch = T24.SAE_EPOCHS
        logger.info("[sae_ctx] RESUMED at epoch %d (step %d)", start_epoch, step)
    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, T24.SAE_EPOCHS):
        rng_e = np.random.default_rng(T24.SAE_SEED * 1000 + epoch)
        fired = torch.zeros(model.dict_size, dtype=torch.bool, device=dev)
        run_loss, run_n = 0.0, 0
        diags: dict = {"l0_train": float("nan")}
        for xb in T24._block_batches(x_mm, tr_pos, T24.SAE_BATCH, rng_e):
            x = torch.as_tensor(np.asarray(xb, np.float32), device=dev)
            loss, diags, fired_b = model.train_step_losses(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            fired |= fired_b
            run_loss += diags["loss"]
            run_n += 1
            step += 1
            if step % 200 == 0:
                print(
                    f"[sae_ctx] epoch {epoch + 1}/{T24.SAE_EPOCHS} step {step} "
                    f"loss={run_loss / max(1, run_n):.1f} thr={float(model.threshold):.4f} "
                    f"l0={diags['l0_train']:.0f} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            if steps_cap and step >= steps_cap:
                stop = True
                break
        fired_union |= fired
        fve_val, l0_val = T24._recon_fve(model, x_mm, val_pos)
        row = {
            "epoch": epoch + 1,
            "steps": step,
            "mean_loss": round(run_loss / max(1, run_n), 3),
            "val_var_fve": round(fve_val, 6),
            "val_l0": round(l0_val, 2),
            "dead_frac_by_tier": T24._dead_by_tier(fired.cpu().numpy()),
            "threshold": float(model.threshold),
            "elapsed_s": round(time.time() - t0, 1),
        }
        epoch_rows.append(row)
        print(
            f"[sae_ctx] unit {epoch + 1}/{T24.SAE_EPOCHS} epoch-done {json.dumps(row)}",
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                # a steps-capped break is a PARTIAL epoch — flagged, never counted done
                "epoch_done": epoch if stop else epoch + 1,
                "steps_capped": bool(stop),
                "step": step,
                "log_rows": epoch_rows,
                "fired_union": fired_union.cpu().numpy(),
            },
            ckpt_path,
        )
        if stop:
            break
    assert epoch_rows, "sae_ctx training produced no epoch rows"
    return model, epoch_rows, fired_union.cpu().numpy(), step


# ── phases ───────────────────────────────────────────────────────────────────────


def phase_assemble(args) -> None:
    """P1 (reused verbatim): stream the banked capture chunks into the X19/Y19
    fp16 memmaps + rows_present + split_meta under out_root/assemble, at the
    ported #2476 lineage pin. Adds this unit's fresh production row-count assert
    (964,844 — the FULL parent row space, never the map's 963,444 n_train)."""
    t24 = _t24_args(args)
    T24.phase_assemble(t24)
    a_dir = T24._assemble_dir(t24)
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    assert x_mm.shape == y_mm.shape, (x_mm.shape, y_mm.shape)
    assert x_mm.shape[1] == int(C.EXPECTED_HIDDEN), x_mm.shape
    if T24._production(t24):
        assert x_mm.shape[0] == N_TOTAL_PRODUCTION, (
            f"assembled row space {x_mm.shape[0]} != {N_TOTAL_PRODUCTION} "
            "(plan Blocker-4: the full parent row space, never n_train=963,444)"
        )
    _sentinel("assemble-2569", f"P1 reuse done (rows={int(x_mm.shape[0])})")


def phase_moments(args) -> None:
    """Streaming split-half + pooled moments over the map-training pool, the two
    pinned-lambda split-half ridge refits, and the unit-2 sigma producer files."""
    C.phase("moments")
    t24 = _t24_args(args)
    out = args.out_root / "moments"
    out.mkdir(parents=True, exist_ok=True)
    outputs = [
        out / "gram_xx.pt",
        out / "gram_xy.pt",
        out / "gram_yy.pt",
        out / "splithalf_maps.pt",
        out / "moments_meta.json",
    ]
    regime, resume_ok = T24._enter_phase_regime(out, t24, "moments_2569", stale_paths=outputs)
    if resume_ok and all(p.exists() for p in outputs):
        # outputs present under a matching regime: verify/re-drive the HF upload
        # (a crash between the writes and the upload must not strand the leaf)
        _upload_leaf(args, outputs, "moments", resume_skip=True)
        logger.info("[moments] resume: outputs present under matching regime; skip")
        return
    a_dir = T24._assemble_dir(t24)
    assert (a_dir / "split_meta.json").exists(), "moments needs the P1 outputs — run assemble"
    T24.EA._headroom(args.out_root, 1 if args.smoke else 6, "pb-moments")
    rows_present = np.load(a_dir / "rows_present.npy")
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    committed = T24._committed_split()
    row_ci, _prov, _pools = T24._load_scratch_meta(t24)
    _r1, val, test = T24._assert_pinned_valtest(committed)

    pool_ids = np.setdiff1d(rows_present, np.union1d(val, test), assume_unique=False)
    pos = np.searchsorted(rows_present, pool_ids)
    assert (rows_present[pos] == pool_ids).all()
    production = T24._production(t24)
    if production:
        assert len(pool_ids) == N_MOMENT_POOL_PRODUCTION, (
            len(pool_ids),
            N_MOMENT_POOL_PRODUCTION,
        )
    keys = _conversation_keys(row_ci, pool_ids)
    i1, i2 = L6.split_halves_by_conversation(keys, seed=L6.SPLIT_SEED)
    assert len(i1) + len(i2) == len(pool_ids)
    pos1, pos2 = pos[i1], pos[i2]

    payload = OP.load_banked_map(args.layer, root=args.map_root)
    lam = float(payload.selected_lambda)
    dev = torch.device(args.device)
    acc1 = _accumulate_moments(x_mm, y_mm, pos1, chunk=args.moment_chunk, dev=dev, tag="half1")
    acc2 = _accumulate_moments(x_mm, y_mm, pos2, chunk=args.moment_chunk, dev=dev, tag="half2")
    pooled = _combine_moments(acc1, acc2)
    refits = {"half1": _half_ridge_refit(acc1, lam), "half2": _half_ridge_refit(acc2, lam)}

    n = int(pooled["n"])
    _write_sigma_pt(
        out / "gram_xx.pt",
        pooled["gram_xx"],
        pooled["sum_x"] / n,
        n,
        side="context (X19)",
        pool="present minus pinned val/test (map-training convention)",
    )
    _write_sigma_pt(
        out / "gram_yy.pt",
        pooled["gram_yy"],
        pooled["sum_y"] / n,
        n,
        side="answer (Y19)",
        pool="present minus pinned val/test (map-training convention)",
    )
    _atomic_torch_save(
        {
            # CROSS moment — deliberately NOT the load_sigma_file schema (no bare
            # "mean"/"gram_xx" semantics): consumers (CCA / theory legs) center it
            # themselves as gram/n - outer(mean_x, mean_y).
            "gram": pooled["gram_xy"].to(torch.float64),
            "mean_x": (pooled["sum_x"] / n).to(torch.float64),
            "mean_y": (pooled["sum_y"] / n).to(torch.float64),
            "n_rows": n,
            "side": "cross (X19 -> Y19)",
        },
        out / "gram_xy.pt",
    )
    _atomic_torch_save(
        {
            "half1": {
                **{k: acc1[k] for k in ("sum_x", "sum_y", "gram_xx", "gram_xy", "gram_yy")},
                "n_rows": int(acc1["n"]),
                **refits["half1"],
            },
            "half2": {
                **{k: acc2[k] for k in ("sum_x", "sum_y", "gram_xx", "gram_xy", "gram_yy")},
                "n_rows": int(acc2["n"]),
                **refits["half2"],
            },
            "selected_lambda": lam,
            "split_seed": int(L6.SPLIT_SEED),
            "ridge_convention": "standardize-X (unbiased sd + 1e-9) / center-Y; "
            "W = (Xstd^T Xstd + lam I)^-1 Xstd^T Yc (fit_ridge_primal parity)",
        },
        out / "splithalf_maps.pt",
    )
    # producer-side consumer-contract round-trip (unit-2 loader must read these)
    import issue2569_gateladder as GL

    for p in (out / "gram_xx.pt", out / "gram_yy.pt"):
        sigma = GL.load_sigma_file(p)
        assert sigma.shape[0] == sigma.shape[1] and np.isfinite(sigma).all(), p
    T24._write_json(
        out / "moments_meta.json",
        {
            "n_pool": n,
            "n_half1": int(acc1["n"]),
            "n_half2": int(acc2["n"]),
            "n_conversation_keys": int(len(set(keys))),
            "selected_lambda": lam,
            "split_seed": int(L6.SPLIT_SEED),
            "moment_chunk": int(args.moment_chunk),
            "device": str(dev),
            "pool": "present minus pinned val/test",
            "production": bool(production),
            "regime_config_hash": regime["config_hash"],
        },
        phase="moments",
    )
    _upload_leaf(args, outputs, "moments", resume_skip=False)
    _sentinel("moments-2569", f"moments done (n={n}, halves {int(acc1['n'])}/{int(acc2['n'])})")
    logger.info("[moments] done: n=%d halves=%d/%d lam=%.4g", n, acc1["n"], acc2["n"], lam)


def phase_sae_train(args) -> None:
    """Context-side SAE on the X19 rows at the exact #2476 recipe; gate G4
    (val var-FVE >= 0.5, rc=T24.RC_G4 at production); deliverables ae.pt +
    config.json + alive_union.json + train_log.json -> HF sae_ctx/."""
    C.phase("sae_ctx_train")
    t24 = _t24_args(args)
    out = args.out_root / "sae_ctx"
    out.mkdir(parents=True, exist_ok=True)
    ae_path = out / "ae.pt"
    cfg_path = out / "config.json"
    alive_path = out / "alive_union.json"
    log_path = out / "train_log.json"
    gates_path = out / "gates_sae_ctx.json"
    ckpt_path = out / "ckpt_last.pt"
    deliverables = [ae_path, cfg_path, alive_path, log_path]
    regime, resume_ok = T24._enter_phase_regime(
        out, t24, "sae_ctx_train", stale_paths=[*deliverables, gates_path, ckpt_path]
    )
    production = T24._production(t24)
    if resume_ok and all(p.exists() for p in (*deliverables, gates_path)):
        gates = json.loads(gates_path.read_text())
        if production and gates["g4"]["verdict"] == "FAIL":
            # recorded-verdict re-entry: a FAIL gate is re-applied, never skipped
            _sentinel("sae-ctx-2569", "G4 recorded FAIL re-applied", {"rc": T24.RC_G4})
            logger.error("[sae_ctx] recorded G4 FAIL re-applied: %s", gates["g4"])
            sys.exit(T24.RC_G4)
        _upload_leaf(args, deliverables, "sae_ctx", resume_skip=True)
        logger.info("[sae_ctx] resume: deliverables present; gates re-applied; skip")
        return
    T24.EA._headroom(args.out_root, 1 if args.smoke else 4, "pb-sae-ctx-train")
    a_dir = T24._assemble_dir(t24)
    assert (a_dir / "split_meta.json").exists(), "sae-train needs the P1 outputs — run assemble"
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    tr_pos, val_pos, pool_doc = T24._sae_row_positions(t24)
    # producer-contract checks BEFORE the (up to 65,536-wide) SAE allocation
    assert x_mm.ndim == 2 and x_mm.shape[1] == int(C.EXPECTED_HIDDEN) and x_mm.shape[0] > 0, (
        x_mm.shape
    )
    assert len(tr_pos) >= 1 and len(val_pos) >= 2, (len(tr_pos), len(val_pos))
    assert int(max(tr_pos.max(), val_pos.max())) < x_mm.shape[0], (
        int(tr_pos.max()),
        int(val_pos.max()),
        x_mm.shape,
    )
    print(f"[sae_ctx] pools re-measured: {json.dumps(pool_doc)}", flush=True)
    width = T24._sae_width(t24)
    if width != T24.SAE_DICT:
        assert not production, "sub-production --sae-dict is smoke-only (plan §11 width)"
        logger.warning("[sae_ctx] SMOKE dictionary width %d (production %d)", width, T24.SAE_DICT)
    model, epoch_rows, fired_union, step = _run_sae_training(
        x_mm,
        tr_pos,
        val_pos,
        width=width,
        dev=args.device,
        steps_cap=int(args.sae_steps),
        ckpt_path=ckpt_path,
        resume_ok=resume_ok,
    )
    fve_val = float(epoch_rows[-1]["val_var_fve"])
    g4_pass = fve_val >= T24.G4_FVE_FLOOR
    gates = {
        "g4": {
            "val_var_fve": fve_val,
            "val_l0": epoch_rows[-1]["val_l0"],
            "floor": T24.G4_FVE_FLOOR,
            "n_val": int(len(val_pos)),
            "verdict": "PASS" if g4_pass else ("FAIL" if production else "INFORMATIONAL-smoke"),
        }
    }
    # persist deliverables + gates BEFORE any halt (halt-investigate needs them)
    _atomic_torch_save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "cfg": model.cfg_dict(),
            "trained_on": "X19 context-side rows (issue #2569 P-B, unit 4a)",
        },
        ae_path,
    )
    alive_idx = np.flatnonzero(fired_union)
    T24._write_json(
        cfg_path,
        {
            **model.cfg_dict(),
            "trained_on": "X19 context-side rows (issue #2569 P-B)",
            "pools": pool_doc,
            "steps": step,
            "steps_cap": int(args.sae_steps),
        },
        phase="sae_ctx_train",
    )
    T24._write_json(
        alive_path,
        {
            "n_alive": int(len(alive_idx)),
            "n_dict": int(model.dict_size),
            "alive_idx": [int(i) for i in alive_idx],
            "dead_frac_by_tier": T24._dead_by_tier(fired_union),
            "epochs_covered": len(epoch_rows),
        },
        phase="sae_ctx_train",
    )
    T24._write_json(
        log_path,
        {"pools": pool_doc, "epochs": epoch_rows, "steps": step, "cfg": model.cfg_dict()},
        phase="sae_ctx_train",
    )
    T24._write_json(gates_path, gates, phase="sae_ctx_train")
    if production and not g4_pass:
        _sentinel("sae-ctx-2569", "G4 SAE-val FVE below floor (gates written)", {"rc": T24.RC_G4})
        logger.error("[sae_ctx] G4 FAIL (halt-investigate before any encode): %s", gates["g4"])
        sys.exit(T24.RC_G4)
    _upload_leaf(args, deliverables, "sae_ctx", resume_skip=False)
    if ckpt_path.exists():
        # training complete + gates PASS + upload verified: the optimizer-state
        # checkpoint is a discard-class intermediate (~1.9 GB at width)
        ckpt_path.unlink()
        logger.info("[sae_ctx] optimizer checkpoint removed (discard-class intermediate)")
    _sentinel(
        "sae-ctx-2569",
        f"sae_ctx done (fve={fve_val:.4f} l0={epoch_rows[-1]['val_l0']} "
        f"g4={gates['g4']['verdict']})",
    )
    logger.info("[sae_ctx] done: %s", gates["g4"])


def load_sae_ctx(path: Path, device: str = "cpu") -> torch.nn.Module:
    """Consumer loader for the sae_ctx ae.pt bundle (unit-4b/leg-4 encodes).

    Self-produced sha-pinned bundle -> weights_only=False is the sanctioned load
    posture (gotchas.md). Rebuilds the matryoshka module from the saved cfg and
    loads the full state dict (threshold buffer included).
    """
    obj = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = obj["cfg"]
    model = T24.MatryoshkaBatchTopKSAE(
        act_dim=int(cfg["act_dim"]),
        dict_size=int(cfg["dict_size"]),
        k=int(cfg["k"]),
        tier_bounds=tuple(cfg["tier_bounds"]),
        seed=int(cfg["seed"]),
    )
    model.load_state_dict(obj["state_dict"])
    return model.to(device).eval()


# ── unit-4b shared helpers ─────────────────────────────────────────────────────────


def _check_local_regime(out: Path, key: dict, wipe: list[Path], tag: str) -> None:
    """Driver-local regime sidecar for knobs the reused T24 config hash cannot
    see (--pairs / --top-pairs / --layer / floor constants): a mismatch wipes
    THIS phase's derived artifacts (loud), then records the new key. Keys are
    GENERATING PARAMETERS (ints/strings), never recomputed float arrays."""
    side = out / "local_regime.json"
    if side.exists():
        old = json.loads(side.read_text())
        if old != key:
            logger.warning("[%s] local regime changed (%s -> %s): wiping", tag, old, key)
            for p in wipe:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
    with atomic_replace(side) as tmp:
        tmp.write_text(json.dumps(key, sort_keys=True))


def _gather_rows(mm, idx: np.ndarray) -> np.ndarray:
    """Rows of a memmap at possibly-unsorted / repeated indices via ONE sorted
    gather + inverse permutation (memmap fancy-indexing wants sorted runs)."""
    idx = np.asarray(idx, np.int64)
    order = np.argsort(idx, kind="stable")
    out = np.asarray(mm[idx[order]])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return out[inv]


def _positions_in_present(rows_present: np.ndarray, ids: np.ndarray, tag: str) -> np.ndarray:
    """Memmap POSITIONS of the split row ids that are present in the assembled
    store (production: all of them; --max-chunks smokes assemble a subset)."""
    ids2 = np.intersect1d(np.asarray(ids, np.int64), rows_present)
    assert len(ids2) > 0, f"[{tag}] no rows of this split are present in the assembled store"
    pos = np.searchsorted(rows_present, ids2)
    assert (rows_present[pos] == ids2).all()
    return pos.astype(np.int64)


# ── leg 4: feature->feature map (plan §4 leg 4 steps 3-6) ─────────────────────────


def _stage_answer_sae(t24, override: Path | None) -> Path:
    """Stage the banked #2476 answer-SAE bundle (cfg.json + sae_weights.safetensors,
    HF ``issue2476_turnavg/analysis_tensors/sae_c/``) into the stage dir
    (idempotent), unless an override dir is given (test seam — same consume path)."""
    if override is not None:
        d = Path(override)
        for name in ("cfg.json", "sae_weights.safetensors"):
            assert (d / name).exists(), f"--answer-sae-dir missing {name}: {d}"
        return d
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    dest = T24._stage_dir(t24) / "sae_ans_2476"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("cfg.json", "sae_weights.safetensors"):
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            continue
        got = hub.retry_transient(
            lambda n=name: hf_hub_download(
                C.HF_DATA_REPO,
                filename=f"{ANSWER_SAE_HF_LEAF}/{n}",
                repo_type="dataset",
                local_dir=str(dest / "_hf"),
            ),
            what=f"answer-SAE fetch ({name})",
        )
        shutil.copy2(got, target)
    return dest


def _stage_alive_counts(t24, override: Path | None) -> Path:
    """Stage the banked #2476 ``alive_c.npz`` (full 65,536-feature firing counts
    over the 120k SAE-fit rows) from HF, unless an override path is given."""
    if override is not None:
        p = Path(override)
        assert p.exists(), f"--alive-counts-npz missing: {p}"
        return p
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    dest = T24._stage_dir(t24) / "alive_c_2476.npz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    got = hub.retry_transient(
        lambda: hf_hub_download(
            C.HF_DATA_REPO,
            filename=ALIVE_C_HF_PATH,
            repo_type="dataset",
            local_dir=str(dest.parent / "_hf_alive"),
        ),
        what="alive_c.npz fetch",
    )
    shutil.copy2(got, dest)
    return dest


def _answer_union_from_counts(alive_npz: Path, *, production: bool) -> np.ndarray:
    """The banked alive answer UNION: features clearing the LOOSEST #2476 sweep
    floor (0.2% of the banked fit rows — floors are nested, so the union across
    the 1%/0.5%/0.25%/0.2% sweep IS the 0.2% set). Production pins the realized
    count to 2,150 (the banked artifact is deterministic)."""
    z = np.load(alive_npz)
    counts = np.asarray(z["counts"], np.int64)
    n_fit_banked = int(z["n_fit_rows"])
    floor = max(1, math.ceil(LEG4_UNION_FRAC * n_fit_banked))
    union = np.flatnonzero(counts >= floor).astype(np.int64)
    banked_panel = np.asarray(z["alive_ids"], np.int64)
    assert np.isin(banked_panel, union).all(), "banked 1% panel must be a subset of the union"
    if production:
        assert len(union) == LEG4_UNION_EXPECTED, (len(union), LEG4_UNION_EXPECTED)
    return union


def _perfeature_r2(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature UNCONDITIONAL holdout R2 (fp64; the T24 _shuffle_null_r2
    kernel). Returns (r2, ss_tot); degenerate-variance features -> NaN."""
    t = np.asarray(true, np.float64)
    mu = t.mean(0)
    ss_tot = ((t - mu) ** 2).sum(0)
    ss_res = ((t - np.asarray(pred, np.float64)) ** 2).sum(0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    return r2, ss_tot


def _firing_auroc(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-feature AUROC of the prediction as a score for the FIRING event
    (true > 0) — the hurdle's detection leg. Midrank (tie-safe) Mann-Whitney
    form; features with all-fired or none-fired holdout rows -> NaN."""
    from scipy.stats import rankdata

    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    n = t.shape[0]
    pos = t > 0
    n_pos = pos.sum(0).astype(np.float64)
    n_neg = n - n_pos
    r = rankdata(p, axis=0, method="average")
    rank_pos_sum = np.where(pos, r, 0.0).sum(0)
    auc = (rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / np.maximum(n_pos * n_neg, 1.0)
    return np.where((n_pos > 0) & (n_neg > 0), auc, np.nan)


def _conditional_magnitude_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-feature CONDITIONAL-magnitude R2 — scored ONLY on holdout rows where
    the true feature fires (the hurdle's magnitude leg; NEVER mixed with the
    unconditional R2 in any verdict). n_fire < 2 or degenerate variance -> NaN."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    m = t > 0
    n_f = m.sum(0).astype(np.float64)
    mu = np.where(n_f > 0, (t * m).sum(0) / np.maximum(n_f, 1.0), 0.0)
    ss_tot = (((t - mu) * m) ** 2).sum(0)
    ss_res = (((t - p) * m) ** 2).sum(0)
    ok = (n_f >= 2) & (ss_tot > 1e-12)
    return np.where(ok, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)


def _realized_l0(true: np.ndarray) -> int:
    """k for P/R@k: the realized mean per-row firing count within the union."""
    t = np.asarray(true, np.float64)
    return max(1, int(round(float((t > 0).sum(1).mean()))))


def _pr_at_k(pred: np.ndarray, true: np.ndarray, k: int) -> dict:
    """Active-set precision/recall@k per holdout row (predicted top-k features by
    predicted activation vs the true firing set within the union). Rows with an
    empty true set are excluded from the recall mean (counted)."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    n, m = p.shape
    k = int(min(max(1, k), m))
    top = np.argpartition(-p, k - 1, axis=1)[:, :k]
    hit = np.take_along_axis(t > 0, top, axis=1).sum(1).astype(np.float64)
    n_true = (t > 0).sum(1).astype(np.float64)
    rec = np.where(n_true > 0, hit / np.maximum(n_true, 1.0), np.nan)
    return {
        "k": k,
        "precision_at_k": float((hit / k).mean()),
        "recall_at_k": float(np.nanmean(rec)),
        "n_rows": int(n),
        "n_rows_zero_true": int((n_true == 0).sum()),
    }


def _route_metrics(
    pred_te: np.ndarray, y_te: np.ndarray, k: int, *, name: str
) -> tuple[dict, dict]:
    """All leg-4 metrics for one comparison route on the holdout. Returns
    (summary dict, per-feature arrays dict). The hurdle decomposition keeps the
    conditional-magnitude R2 EXPLICITLY separate from the unconditional R2."""
    r2, ss_tot = _perfeature_r2(pred_te, y_te)
    auroc = _firing_auroc(pred_te, y_te)
    cond = _conditional_magnitude_r2(pred_te, y_te)
    summary = {
        "route": name,
        "r2_unconditional": {
            "median": float(np.nanmedian(r2)),
            "mean": float(np.nanmean(r2)),
            "frac_positive": float(np.nanmean(r2 > 0)),
            "n_nan": int(np.isnan(r2).sum()),
        },
        "hurdle": {
            "firing_auroc_median": float(np.nanmedian(auroc)),
            "firing_auroc_n_nan": int(np.isnan(auroc).sum()),
            "conditional_magnitude_r2_median": float(np.nanmedian(cond)),
            "conditional_magnitude_r2_n_nan": int(np.isnan(cond).sum()),
            "note": (
                "conditional-magnitude R2 is scored ONLY on rows where the true feature "
                "fires; it is never mixed with the unconditional R2 in any verdict"
            ),
        },
        "pr_at_k": _pr_at_k(pred_te, y_te, k),
    }
    arrays = {
        f"r2_{name}": r2.astype(np.float32),
        f"auroc_{name}": auroc.astype(np.float32),
        f"cond_r2_{name}": cond.astype(np.float32),
    }
    return summary, arrays


def _fit_val_widened(X, Y, tr, va, te, dev, *, fit_fn=None, grid=None, max_widenings=None):
    """C4 widen-on-edge val-selected primal ridge. Mirrors GL.fit_point's widen
    loop (same grid / widen_grid / max-widenings constants) but RETURNS the test
    predictions the leg-4 per-feature metrics need (fit_point returns summary
    stats only). Fails loud after max widenings — an edge lambda is never
    reported (C4). ``fit_fn`` is a test seam (default: the real #779 core)."""
    import issue2569_gateladder as GL

    if fit_fn is None:
        fit_fn = T24.N50.fit_ridge_primal
    g = tuple(GL.LAMBDA_GRID_27 if grid is None else grid)
    max_w = int(GL.MAX_WIDENINGS if max_widenings is None else max_widenings)
    edge = None
    for w in range(max_w + 1):
        pred_te, meta = fit_fn(X, Y, tr, va, te, list(g), dev)
        edge = meta.get("lambda_grid_edge")
        if not edge:
            meta = dict(meta)
            meta.update(widenings=w, grid_lo=float(g[0]), grid_hi=float(g[-1]), grid_n=len(g))
            return np.asarray(pred_te), meta
        logger.warning("[featmap] lambda at the %s edge — widening the grid (C4)", edge)
        g = GL.widen_grid(g, edge)
    raise RuntimeError(
        f"lambda selection still at the {edge} edge after {max_w} widenings "
        "(C4: refusing to report an edge value)"
    )


def _banked_route_summary(fname: str, union: np.ndarray, banked_dir: Path, *, route: str) -> dict:
    """Comparison routes (ii)/(iii): the banked #2476 per-feature instrument
    JOINED by feature id on the union intersection — REUSED, never re-run."""
    p = banked_dir / fname
    assert p.exists(), f"banked #2476 route artifact missing: {p}"
    z = np.load(p)
    ids = np.asarray(z["feat_ids"], np.int64)
    r2 = np.asarray(z["r2"], np.float64)
    inter = np.isin(ids, union)
    r2i = r2[inter]
    return {
        "route": route,
        "source": str(p.relative_to(PROJECT_ROOT)),
        "n_banked": int(len(ids)),
        "n_in_union": int(inter.sum()),
        "banked_alive_floor": int(z["alive_floor"]),
        "banked_n_fit_rows": int(z["n_fit_rows"]),
        "r2_median_on_intersection": float(np.median(r2i)) if len(r2i) else float("nan"),
        "r2_mean_on_intersection": float(np.mean(r2i)) if len(r2i) else float("nan"),
        "note": (
            "banked #2476 instrument REUSED not re-run; per-feature R2 from ITS holdout "
            "predictions (same pinned 20k holdout rows); intersection = banked 1% panel "
            "within the 0.2% union"
        ),
    }


def phase_feature_map(args) -> None:
    """Leg-4 steps 3-6: alive-ctx-feature -> banked-alive-answer-union ridge map
    + the six comparison routes + hurdle metrics + floor sweep (module docstring).

    Data gates: production pins (n_fit, n_val, n_te) = (120,000, 400, 20,000),
    union = 2,150, and halts (assert) when the ctx alive-input width exceeds
    LEG4_CTX_ALIVE_CAP (halt-investigate, never silent truncation); smoke floors
    are nonzero-yield only (fit >= 32, val >= 4, te >= 8) with production-
    calibrated pins demoted to informational. Encodes checkpoint to fp16 files
    with regime-keyed resume; never materializes a full 65,536-wide matrix."""
    C.phase("feature_map")
    t24 = _t24_args(args)
    out = args.out_root / "leg4"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "feature_map_metrics.json"
    npz_path = out / "perfeature_leg4.npz"
    outputs = [metrics_path, npz_path]
    enc_files = {
        "counts_ctx": out / "counts_ctx.npy",
        "x_alive": out / "x_ctx_alive.fp16.npy",
        "y_union": out / "y_union.fp16.npy",
        "x_idx": out / "x_ctx_idxaligned.fp16.npy",
        "enc_meta": out / "enc_meta.json",
    }
    regime, resume_ok = T24._enter_phase_regime(
        out, t24, "featmap_2569", stale_paths=[*outputs, *enc_files.values()]
    )
    _check_local_regime(
        out,
        {
            "layer": int(args.layer),
            "union_frac": LEG4_UNION_FRAC,
            "floors": list(LEG4_CTX_FLOOR_FRACS),
            "phase": "featmap_2569",
        },
        wipe=[*outputs, *enc_files.values()],
        tag="featmap",
    )
    if resume_ok and all(p.exists() for p in outputs):
        _upload_leaf(args, outputs, "leg4", resume_skip=True)
        logger.info("[featmap] resume: outputs present under matching regime; skip")
        return
    production = T24._production(t24)
    a_dir = T24._assemble_dir(t24)
    assert (a_dir / "split_meta.json").exists(), "feature-map needs P1 outputs — run assemble"
    sae_ctx_path = args.out_root / "sae_ctx" / "ae.pt"
    assert sae_ctx_path.exists(), "feature-map needs the fresh ctx SAE — run sae-train"
    T24.EA._headroom(args.out_root, 1 if args.smoke else 8, "pb-featmap")
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    rows_present = np.load(a_dir / "rows_present.npy")
    committed = T24._committed_split()
    _row_ci, _prov, pools = T24._load_scratch_meta(t24)
    _r1, val_ids, _test_ids = T24._assert_pinned_valtest(committed)

    fit_pos = _positions_in_present(rows_present, pools["sae_fit"], "featmap/fit")
    val_pos = _positions_in_present(rows_present, val_ids, "featmap/val")
    te_pos = _positions_in_present(rows_present, pools["holdout"], "featmap/te")
    if production:
        assert (len(fit_pos), len(val_pos), len(te_pos)) == (120_000, 400, 20_000), (
            len(fit_pos),
            len(val_pos),
            len(te_pos),
        )
    else:  # smoke caps (deterministic heads) + nonzero-yield floors
        fit_pos, val_pos, te_pos = fit_pos[:4096], val_pos[:400], te_pos[:512]
        assert len(fit_pos) >= 32 and len(val_pos) >= 4 and len(te_pos) >= 8, (
            len(fit_pos),
            len(val_pos),
            len(te_pos),
        )
    n_fit, n_val, n_te = len(fit_pos), len(val_pos), len(te_pos)
    rows_cat = np.concatenate([fit_pos, val_pos, te_pos])
    tr = np.arange(n_fit)
    va = n_fit + np.arange(n_val)
    te = n_fit + n_val + np.arange(n_te)

    union = _answer_union_from_counts(
        _stage_alive_counts(t24, args.alive_counts_npz), production=production
    )
    sae_ans = T24.MatryoshkaBatchTopKSAE.load_local(
        _stage_answer_sae(t24, args.answer_sae_dir), device=args.device
    )
    sae_ctx = load_sae_ctx(sae_ctx_path, device=args.device)

    # ── encodes (checkpointed; alive-column projections only — plan §10) ──────────
    if enc_files["counts_ctx"].exists():
        counts_ctx = np.load(enc_files["counts_ctx"])
    else:
        counts_ctx = T24._encode_counts(sae_ctx, x_mm, fit_pos)
        _atomic_np_save(counts_ctx, enc_files["counts_ctx"])
    floors = {f: max(1, math.ceil(f * n_fit)) for f in LEG4_CTX_FLOOR_FRACS}
    alive_by_floor = {f: np.flatnonzero(counts_ctx >= fl) for f, fl in floors.items()}
    alive_loose = alive_by_floor[LEG4_CTX_FLOOR_FRACS[-1]]  # supersets: 0.2% is loosest
    assert len(alive_by_floor[LEG4_CTX_FLOOR_FRACS[0]]) >= 1, "no alive ctx features at 1%"
    assert len(alive_loose) <= LEG4_CTX_ALIVE_CAP, (
        f"ctx alive set {len(alive_loose)} > {LEG4_CTX_ALIVE_CAP} — halt-investigate "
        "(fit RAM/time sized for O(10^3..10^4) inputs; raise the cap deliberately)"
    )

    def _enc(key: str, sae, mm, cols) -> np.ndarray:
        if enc_files[key].exists():
            return np.load(enc_files[key], mmap_mode="r")
        t0 = time.time()
        arr = T24._encode_restricted(sae, mm, rows_cat, cols)
        _atomic_np_save(arr, enc_files[key])
        print(f"[featmap] encode {key} {arr.shape} elapsed={time.time() - t0:.0f}s", flush=True)
        return arr

    x_alive = _enc("x_alive", sae_ctx, x_mm, alive_loose)
    y_union = _enc("y_union", sae_ans, y_mm, union)
    x_idx = _enc("x_idx", sae_ctx, x_mm, union)  # route (iv): SAME indices, ctx dict
    T24._write_json(
        enc_files["enc_meta"],
        {
            "rows": [n_fit, n_val, n_te],
            "alive_loose": int(len(alive_loose)),
            "union": int(len(union)),
            "floors": {str(k): int(v) for k, v in floors.items()},
        },
        phase="featmap",
    )
    y_te64 = np.asarray(y_union[te], np.float64)
    k_l0 = _realized_l0(y_te64)

    # ── route (i): fitted feature->feature map (primary floor 1%) + floor sweep ──
    routes: list[dict] = []
    arrays: dict[str, np.ndarray] = {"feat_ids": union}
    sweep = []
    pred_te = None
    y_fit32 = np.asarray(y_union, np.float32)  # hoisted once (shared across floors)
    for frac in LEG4_CTX_FLOOR_FRACS:
        cols = np.searchsorted(alive_loose, alive_by_floor[frac])
        assert (alive_loose[cols] == alive_by_floor[frac]).all()
        d_alive = len(cols)
        assert n_fit >= 2 * d_alive or not production, (
            f"n_fit {n_fit} < 2*d_alive {d_alive} — plan §4 leg 4 forbids the n<d regime"
        )
        xf = np.asarray(x_alive[:, cols], np.float32)  # column subset only (never full fp32)
        pt, meta = _fit_val_widened(xf, y_fit32, tr, va, te, args.device)
        r2_pf, _ = _perfeature_r2(pt, y_te64)
        sweep.append(
            {
                "floor_frac": frac,
                "floor_rows": int(floors[frac]),
                "d_alive_ctx": int(d_alive),
                "selected_lambda": float(meta["selected_lambda"]),
                "val_r2_at_selected": float(meta["val_r2_at_selected"]),
                "lambda_grid_edge": meta.get("lambda_grid_edge"),
                "widenings": int(meta["widenings"]),
                "r2_median": float(np.nanmedian(r2_pf)),
            }
        )
        print(f"[featmap] unit floor={frac} d={d_alive} fitted {json.dumps(sweep[-1])}", flush=True)
        if frac == LEG4_CTX_FLOOR_FRACS[0]:  # 1% primary carries the headline metrics
            pred_te = pt
            s, a = _route_metrics(pt, y_te64, k_l0, name="fitted_map")
            s["fit_meta"] = sweep[-1]
            routes.append(s)
            arrays.update(a)
    assert pred_te is not None

    # ── routes (ii)+(iii): banked #2476 instruments, joined per feature ──────────
    routes.append(
        _banked_route_summary(
            "perfeature_c_encodepred.npz", union, BANKED_2476_DIR, route="composed_banked_2476"
        )
    )
    routes.append(
        _banked_route_summary(
            "perfeature_c_densein.npz", union, BANKED_2476_DIR, route="dense_input_banked_2476"
        )
    )

    # ── route (iv): index-aligned identity+bias (labeled null) ───────────────────
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    ib_pred = identity_bias_predict(
        np.asarray(x_idx[tr], np.float64),
        np.asarray(y_union[tr], np.float64),
        np.asarray(x_idx[te], np.float64),
    )
    s_ib, a_ib = _route_metrics(ib_pred, y_te64, k_l0, name="index_aligned_ib")
    s_ib["label"] = (
        "index-aligned null — feature indices are unrelated across dictionaries, "
        "expected ~ train-mean"
    )
    routes.append(s_ib)
    arrays.update(a_ib)

    # ── route (v): train-mean null ────────────────────────────────────────────────
    mu_tr = np.asarray(y_union[tr], np.float64).mean(0)
    tm_pred = np.broadcast_to(mu_tr, (n_te, len(union)))
    r2_tm, _ = _perfeature_r2(tm_pred, y_te64)
    routes.append(
        {
            "route": "train_mean_null",
            "r2_unconditional": {
                "median": float(np.nanmedian(r2_tm)),
                "mean": float(np.nanmean(r2_tm)),
                "n_nan": int(np.isnan(r2_tm).sum()),
            },
            "note": "constant predictor: AUROC/PR@k undefined (constant scores) — omitted",
        }
    )
    arrays["r2_train_mean_null"] = r2_tm.astype(np.float32)

    # ── route (vi): 20-draw row-shuffle null (the #2476 convention + seeds) ──────
    arrays["shuffle_null_r2_fitted"] = T24._shuffle_null_r2(
        pred_te, y_te64, T24.SHUFFLE_SEEDS_2476, what=" leg4/fitted"
    )
    arrays["shuffle_null_r2_ib"] = T24._shuffle_null_r2(
        ib_pred, y_te64, T24.SHUFFLE_SEEDS_2476, what=" leg4/ib"
    )
    arrays["activity_te"] = (y_te64 > 0).mean(0).astype(np.float32)

    # ── kNN retrieval in answer-SAE activation space (chance = k/n_pool) ─────────
    knn = {
        m: knn_retrieval(np.asarray(pred_te, np.float64), y_te64, metric=m)
        for m in ("euclidean", "cosine")
    }

    _atomic_npz_save(npz_path, **arrays)
    T24._write_json(
        metrics_path,
        {
            "n_fit": n_fit,
            "n_val": n_val,
            "n_te": n_te,
            "n_union": int(len(union)),
            "union_source": ALIVE_C_HF_PATH,
            "union_floor_frac": LEG4_UNION_FRAC,
            "realized_l0_k": int(k_l0),
            "routes": routes,
            "floor_sweep": sweep,
            "knn_retrieval": knn,
            "shuffle_null": {
                "n_draws": len(T24.SHUFFLE_SEEDS_2476),
                "seeds": [int(s) for s in T24.SHUFFLE_SEEDS_2476],
                "convention": "prediction rows permuted; same fp64 R2 kernel (#2476)",
            },
            "grain_matching": {
                "inputs": "fresh ctx SAE on v_C (last-prompt-token state; conversation grain)",
                "targets": "#2476 answer SAE on v_A (answer-span mean; turn-averaged grain)",
                "headline_rule": (
                    "no cross-grain read carries a headline; per-token andyrdt reads of "
                    "v_C (token grain) live in P-A, not this leg"
                ),
            },
            "production": bool(production),
            "regime_config_hash": regime["config_hash"],
        },
        phase="featmap",
    )
    _upload_leaf(args, outputs, "leg4", resume_skip=False)
    _sentinel(
        "featmap-2569",
        f"leg4 done (d_alive_1pct={sweep[0]['d_alive_ctx']}, union={len(union)}, "
        f"r2_median={sweep[0]['r2_median']:.4f})",
    )
    logger.info("[featmap] done: %s", json.dumps(sweep[0]))


# ── leg 8: kernel-pair mining (plan §4 leg 8 step 2 + C2) ─────────────────────────


def _assert_mining_identity(payload, x_mm, pool_pos: np.ndarray, *, n_probes: int = 64) -> dict:
    """B1 assert (iii) as a REAL probe-batch assert: the mining statistic
    ``dc @ A`` must equal the registered prediction difference (affine terms
    cancel in differences). A raise HALTs mining (apply-path breakage class)."""
    rng = np.random.default_rng(LEG8_SEED)
    n = len(pool_pos)
    assert n >= 2, f"mining pool too small: {n}"
    i = pool_pos[rng.integers(0, n, size=n_probes)]
    j = pool_pos[rng.integers(0, n, size=n_probes)]
    v1 = np.asarray(_gather_rows(x_mm, i), np.float64)
    v2 = np.asarray(_gather_rows(x_mm, j), np.float64)
    A, _b = OP.row_operator(payload)
    got = OP.mapped_displacement(v1 - v2, A)
    ref = OP.prediction_difference(payload, v1, v2)
    scale = max(1.0, float(np.abs(ref).max()))
    err = float(np.abs(got - ref).max()) / scale
    assert err <= 1e-6, (
        f"[mine] B1 assert iii FAILED: mining statistic != registered prediction "
        f"difference (max rel err {err:.3e})"
    )
    return {"n_probes": int(n_probes), "max_rel_err": err}


def _mine_chunks(
    x_mm, A: np.ndarray, pool_pos: np.ndarray, *, n_pairs: int, chunk: int, out_dir: Path, dev
) -> list[Path]:
    """Chunked pair sampling + batched-GEMM kernel-fraction pass. Per-chunk npz
    checkpoints (i, j, dc_norm, kappa) with presence-keyed resume; each chunk is
    deterministic in ``default_rng((LEG8_SEED, k))``, so a resume regenerates
    nothing and recomputes nothing. NEVER a per-pair python loop."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(n_pairs / chunk)
    A_t = torch.as_tensor(np.asarray(A, np.float32), device=dev)
    files = []
    t0 = time.time()
    for k in range(n_chunks):
        path = out_dir / f"chunk_{k:05d}.npz"
        files.append(path)
        if path.exists():
            continue
        m = int(min(chunk, n_pairs - k * chunk))
        rng = np.random.default_rng((LEG8_SEED, k))
        i = pool_pos[rng.integers(0, len(pool_pos), size=m)]
        j = pool_pos[rng.integers(0, len(pool_pos), size=m)]
        fix = i == j
        while fix.any():  # resample self-pairs (uniform over off-diagonal pairs)
            j[fix] = pool_pos[rng.integers(0, len(pool_pos), size=int(fix.sum()))]
            fix = i == j
        dcn = np.empty(m, np.float32)
        kap = np.empty(m, np.float32)
        sub = 65_536
        with torch.no_grad():
            for lo in range(0, m, sub):
                hi = min(lo + sub, m)
                a = torch.as_tensor(_gather_rows(x_mm, i[lo:hi]).astype(np.float32), device=dev)
                b = torch.as_tensor(_gather_rows(x_mm, j[lo:hi]).astype(np.float32), device=dev)
                d = a - b
                nrm = torch.linalg.vector_norm(d, dim=1)
                mapped = torch.linalg.vector_norm(d @ A_t, dim=1)
                dcn[lo:hi] = nrm.cpu().numpy()
                kap[lo:hi] = (mapped / torch.clamp(nrm, min=1e-12)).cpu().numpy()
                del a, b, d, nrm, mapped
        _atomic_npz_save(path, i=i.astype(np.int64), j=j.astype(np.int64), dc_norm=dcn, kappa=kap)
        print(
            f"[mine] unit {k + 1}/{n_chunks} chunk_{k:05d} pairs={m} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return files


def _ans_len_from_manifest_dir(manifest_dir: Path, row_ci: np.ndarray, n_pb: int) -> np.ndarray:
    """Per-row answer CHAR length over the assembled row space, joined from the
    n1m sampling-manifest parts by conversation index (ci). pass_b rows (the
    leading ``n_pb`` rows, ci == -1) have no manifest text -> -1 = the
    'length-unknown' stratum (5,000 rows, 0.52% of the production pool).
    Response text is consumed ONLY via len() — never logged (content hygiene)."""
    row_ci = np.asarray(row_ci, np.int64)
    rev = {int(c): r for r, c in enumerate(row_ci[n_pb:], start=n_pb)}
    out = np.full(len(row_ci), -1, np.int64)
    parts = sorted(Path(manifest_dir).glob("part_*.jsonl"))
    assert parts, f"no sampling-manifest parts under {manifest_dir}"
    n_hit = 0
    for pi, part in enumerate(parts):
        with open(part, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                r = rev.get(int(rec["ci"]))
                if r is not None:
                    out[r] = len(rec.get("response") or "")
                    n_hit += 1
        print(f"[mine] ans-len part {pi + 1}/{len(parts)} joined={n_hit}", flush=True)
    return out


def _pair_strata(
    i: np.ndarray, j: np.ndarray, prov: np.ndarray, ans_len: np.ndarray, dec_edges: np.ndarray
) -> np.ndarray:
    """Pair stratum ids: sorted-source pair x mean-answer-length decile (C2
    confound guard). Either row length-unknown (-1) -> decile bucket -1 (its own
    per-source stratum). Encoded as src*100 + decile (collision-free for
    prov < 16 and deciles in [-1, 9])."""
    s1 = prov[i].astype(np.int64)
    s2 = prov[j].astype(np.int64)
    src = np.minimum(s1, s2) * 16 + np.maximum(s1, s2)
    l1 = ans_len[i]
    l2 = ans_len[j]
    known = (l1 >= 0) & (l2 >= 0)
    dec = np.full(len(i), -1, np.int64)
    if known.any():
        mean_len = (l1[known] + l2[known]) / 2.0
        dec[known] = np.clip(np.searchsorted(dec_edges, mean_len, side="right"), 0, 9)
    return src * 100 + dec


def _select_kernel_pairs(
    i: np.ndarray,
    j: np.ndarray,
    dcn: np.ndarray,
    kap: np.ndarray,
    strata: np.ndarray,
    *,
    top_pairs: int,
    ctrl_tol_ladder: tuple[float, ...] = LEG8_CTRL_TOL_LADDER,
) -> dict:
    """Kernel-pair selection + stratified distance-matched controls.

    Kernel set: the ``top_pairs`` LOWEST kernel-fraction pairs among unordered-
    deduped pairs with ||dc|| above the sampled-pair median (the corpus-median
    eligibility gate). Controls: for each kernel pair, the nearest-||dc||
    UNUSED eligible pair (greedy, kernel order = ascending kappa) in the SAME
    stratum with kappa in the middle quintile, at the pre-registered widening
    tolerance ladder (2% -> 4% -> 8%); unmatched kernels are DROPPED (counted).
    Returns indices into the sampled arrays + audit counts."""
    n = len(dcn)
    lo = np.minimum(i, j).astype(np.int64)
    hi = np.maximum(i, j).astype(np.int64)
    _uniq, first = np.unique(lo * (2**32) + hi, return_index=True)
    keep = np.zeros(n, bool)
    keep[first] = True
    med = float(np.median(dcn))
    elig = keep & (dcn > med)
    n_elig = int(elig.sum())
    assert n_elig >= 10, f"[mine] eligible pairs too few: {n_elig}"
    kap_e = kap[elig]
    dec_edge = float(np.quantile(kap_e, 0.10))
    q40, q60 = (float(np.quantile(kap_e, q)) for q in (0.40, 0.60))
    elig_idx = np.flatnonzero(elig)
    order = np.argsort(kap[elig_idx], kind="stable")
    kernel_idx = elig_idx[order[: int(min(top_pairs, len(order)))]]
    is_kernel = np.zeros(n, bool)
    is_kernel[kernel_idx] = True
    cand_mask = elig & ~is_kernel & (kap >= q40) & (kap <= q60)
    # per-stratum sorted candidate tables (control matching runs WITHIN strata)
    groups: dict[int, dict] = {}
    for s in np.unique(strata[cand_mask]):
        idx = np.flatnonzero(cand_mask & (strata == s))
        srt = np.argsort(dcn[idx], kind="stable")
        groups[int(s)] = {"idx": idx[srt], "dcn": dcn[idx[srt]], "used": np.zeros(len(idx), bool)}
    matched_k: list[int] = []
    matched_c: list[int] = []
    matched_tol: list[float] = []
    unmatched = list(kernel_idx)
    per_tol_counts = {}
    for tol in ctrl_tol_ladder:
        still = []
        for kidx in unmatched:
            g = groups.get(int(strata[kidx]))
            best, best_gap = -1, np.inf
            if g is not None:
                target = float(dcn[kidx])
                loi = np.searchsorted(g["dcn"], target * (1.0 - tol), side="left")
                hii = np.searchsorted(g["dcn"], target * (1.0 + tol), side="right")
                for p in range(loi, hii):
                    if g["used"][p]:
                        continue
                    gap = abs(float(g["dcn"][p]) - target)
                    if gap < best_gap:
                        best, best_gap = p, gap
            if best >= 0:
                g["used"][best] = True
                matched_k.append(int(kidx))
                matched_c.append(int(g["idx"][best]))
                matched_tol.append(float(tol))
            else:
                still.append(kidx)
        per_tol_counts[str(tol)] = len(matched_k)
        unmatched = still
        if not unmatched:
            break
    return {
        "kernel_idx": np.asarray(matched_k, np.int64),
        "control_idx": np.asarray(matched_c, np.int64),
        "matched_tol": np.asarray(matched_tol, np.float64),
        "n_sampled": int(n),
        "n_dedup": int(keep.sum()),
        "n_eligible": n_elig,
        "dc_norm_median": med,
        "kappa_bottom_decile_edge": dec_edge,
        "kappa_mid_quintile": [q40, q60],
        "n_kernel_selected": int(len(kernel_idx)),
        "n_matched": len(matched_k),
        "n_dropped_no_control": int(len(kernel_idx)) - len(matched_k),
        "per_tol_matched_cum": per_tol_counts,
        "n_kernel_in_bottom_decile": int((kap[kernel_idx] <= dec_edge).sum()),
    }


def _paired_ratio_stats(dva_k: np.ndarray, dva_c: np.ndarray) -> tuple[dict, np.ndarray]:
    """PINNED headline estimator: MEDIAN OF PAIRED RATIOS (kernel/control), the
    ratio-of-medians companion, Wilcoxon signed-rank on log-ratios (paired) and
    Mann-Whitney U (unpaired). Zero/degenerate controls are dropped LOUDLY
    (counted), never coerced. The SAME pinned estimator is recomputed inside
    every bootstrap draw (see _clustered_bootstrap_median_ratio)."""
    from scipy.stats import mannwhitneyu, wilcoxon

    k = np.asarray(dva_k, np.float64)
    c = np.asarray(dva_c, np.float64)
    ok = (c > 0) & (k > 0)
    ratios = k[ok] / c[ok]
    out = {
        "estimator_pinned": "median_of_paired_ratios",
        "median_of_paired_ratios": float(np.median(ratios)) if len(ratios) else float("nan"),
        "ratio_of_medians_companion": (
            float(np.median(k) / np.median(c)) if float(np.median(c)) > 0 else float("nan")
        ),
        "n_pairs": int(len(k)),
        "n_zero_dropped": int((~ok).sum()),
        "kernel_dva_median": float(np.median(k)) if len(k) else float("nan"),
        "control_dva_median": float(np.median(c)) if len(c) else float("nan"),
    }
    logr = np.log(ratios) if len(ratios) else np.empty(0)
    if len(logr) >= 10 and float(np.ptp(logr)) > 0:
        w = wilcoxon(logr)
        out["wilcoxon_logratio"] = {
            "stat": float(w.statistic),
            "p": float(w.pvalue),
            "n": int(len(logr)),
        }
    if len(k) >= 10 and len(c) >= 10:
        u = mannwhitneyu(k, c, alternative="two-sided")
        out["mannwhitney_u"] = {"stat": float(u.statistic), "p": float(u.pvalue)}
    return out, ratios


def _overlap_clusters(conv_keys_per_unit: list[set]) -> np.ndarray:
    """Union-find overlap components over per-unit conversation-key sets: units
    sharing ANY conversation merge into one bootstrap-resampling cluster (a
    cluster spanning strata is resampled as one unit — dependence-safe)."""
    parent = list(range(len(conv_keys_per_unit)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    key2unit: dict[str, int] = {}
    for u, keys in enumerate(conv_keys_per_unit):
        for kk in keys:
            if kk in key2unit:
                ra, rb = find(u), find(key2unit[kk])
                if ra != rb:
                    parent[rb] = ra
            else:
                key2unit[kk] = u
    return np.asarray([find(u) for u in range(len(conv_keys_per_unit))], np.int64)


def _clustered_bootstrap_median_ratio(
    ratios: np.ndarray, clusters: np.ndarray, *, draws: int, seed: int
) -> dict:
    """Clustered bootstrap CI of the PINNED estimator: resample overlap-component
    clusters with replacement (NEVER by pair); every draw recomputes the SAME
    median-of-paired-ratios estimator as a multiplicity-weighted median
    (sorted-cumsum, fully batched across draws)."""
    ratios = np.asarray(ratios, np.float64)
    clusters = np.asarray(clusters)
    assert len(ratios) == len(clusters) and len(ratios) >= 1, (len(ratios), len(clusters))
    _uc, cl = np.unique(clusters, return_inverse=True)
    n_cl = len(_uc)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_cl, np.full(n_cl, 1.0 / n_cl), size=int(draws))
    w = counts[:, cl].astype(np.float64)  # (draws, n_units) unit multiplicities
    order = np.argsort(ratios, kind="stable")
    cum = np.cumsum(w[:, order], axis=1)
    tot = cum[:, -1][:, None]
    ok = tot[:, 0] > 0
    idx = np.argmax(cum >= tot / 2.0, axis=1)
    est = np.where(ok, ratios[order][idx], np.nan)
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {
        "draws": int(draws),
        "n_clusters": int(n_cl),
        "n_units": int(len(ratios)),
        "ci95": [float(lo), float(hi)],
        "n_empty_draws": int((~ok).sum()),
        "seed": int(seed),
        "estimator": "median_of_paired_ratios (weighted median per draw)",
    }


def _residual_floor(payload, x_mm, y_mm, held_pos: np.ndarray, dev) -> dict:
    """C2 measured noise floor: pairwise ||r_i - r_j|| over the genuinely-held-out
    rows' map residuals r = v_A - vhat (val+test — the only rows outside the
    banked fit). The kernel-vs-control ratio is read against these quantiles,
    never against 1 alone."""
    x = np.asarray(_gather_rows(x_mm, held_pos), np.float64)
    y = np.asarray(_gather_rows(y_mm, held_pos), np.float64)
    r = y - OP.predict(payload, x)
    rt = torch.as_tensor(r, dtype=torch.float64, device=dev)
    g = rt @ rt.T
    sq = torch.diagonal(g)
    d2 = torch.clamp(sq[:, None] + sq[None, :] - 2.0 * g, min=0.0)
    n = r.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=rt.device)
    dist = torch.sqrt(d2[iu[0], iu[1]]).cpu().numpy()
    qs = {f"q{int(q * 100):02d}": float(np.quantile(dist, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return {"n_rows": int(n), "n_pairs": int(len(dist)), **qs}


def phase_mine(args) -> None:
    """Leg-8 step 2: kernel-pair mining + matched read (module docstring).

    Data gates: production aborts LOUD below LEG8_MIN_KERNEL_PRODUCTION matched
    pairs (rc=RC_MINE_ABORT, sentinel first); a non-production run refuses
    ``--pairs`` > 10^6 (the smoke launcher passes M=10^5 explicitly — never a
    silent smoke-valued substitution); the B1 assert-iii probe HALTs on a
    mining-statistic/apply-path mismatch. Per-chunk npz checkpoints; the
    per-pair chunk intermediates are the plan §10 declared discard (regen:
    re-run --phase mine at the recorded seed)."""
    C.phase("mine")
    t24 = _t24_args(args)
    out = args.out_root / "leg8"
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "kernel_pairs.json"
    summary_path = out / "mining_summary.json"
    outputs = [pairs_path, summary_path]
    chunks_dir = out / "chunks"
    ans_len_path = out / "ans_len.npy"
    regime, resume_ok = T24._enter_phase_regime(
        out, t24, "mine_2569", stale_paths=[*outputs, ans_len_path, chunks_dir]
    )
    _check_local_regime(
        out,
        {
            "pairs": int(args.pairs),
            "chunk": int(args.pair_chunk),
            "top_pairs": int(args.top_pairs),
            "boot_draws": int(args.boot_draws),
            "seed": LEG8_SEED,
            "layer": int(args.layer),
        },
        wipe=[*outputs, ans_len_path, chunks_dir],
        tag="mine",
    )
    if resume_ok and all(p.exists() for p in outputs):
        _upload_leaf(args, outputs, "leg8", resume_skip=True)
        logger.info("[mine] resume: outputs present under matching regime; skip")
        return
    production = T24._production(t24)
    assert production or args.pairs <= 1_000_000, (
        f"--pairs {args.pairs} on a non-production run: pass the smoke M explicitly "
        "(plan P-B smoke: M = 10^5) — never a silent smoke-valued substitution"
    )
    a_dir = T24._assemble_dir(t24)
    assert (a_dir / "split_meta.json").exists(), "mine needs the P1 outputs — run assemble"
    T24.EA._headroom(args.out_root, 1 if args.smoke else 4, "pb-mine")
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    rows_present = np.load(a_dir / "rows_present.npy")
    committed = T24._committed_split()
    row_ci, prov_u8, _pools = T24._load_scratch_meta(t24)
    _r1, val_ids, test_ids = T24._assert_pinned_valtest(committed)
    if production:
        assert len(rows_present) == N_TOTAL_PRODUCTION, len(rows_present)
    pool_pos = np.arange(len(rows_present), dtype=np.int64)

    payload = OP.load_banked_map(args.layer, root=args.map_root)
    probe = _assert_mining_identity(payload, x_mm, pool_pos)
    logger.info("[mine] B1 assert iii probe PASS: %s", probe)

    # ── answer-length stratifier (C2; checkpointed) ───────────────────────────────
    if ans_len_path.exists():
        ans_len = np.load(ans_len_path)
    else:
        if args.manifest_dir is not None:
            mdir = Path(args.manifest_dir)
        else:
            import issue779_ffc_n1m_generate_capture as N1G

            mdir = N1G._download_manifest(
                N1G.HF_PREFIX, T24._stage_dir(t24) / "sampling_manifest_2569"
            )
        ans_len = _ans_len_from_manifest_dir(mdir, row_ci, int(T24.N1M.N_PASS_B))
        _atomic_np_save(ans_len, ans_len_path)
    assert len(ans_len) == len(row_ci), (len(ans_len), len(row_ci))

    # ── chunked mining pass (per-chunk checkpoints) ──────────────────────────────
    A, _b = OP.row_operator(payload)
    files = _mine_chunks(
        x_mm,
        A,
        pool_pos,
        n_pairs=int(args.pairs),
        chunk=int(args.pair_chunk),
        out_dir=chunks_dir,
        dev=args.device,
    )
    cols = {"i": [], "j": [], "dc_norm": [], "kappa": []}
    for p in files:
        with np.load(p) as z:
            for kk in cols:
                cols[kk].append(np.asarray(z[kk]))
    i_pos = np.concatenate(cols["i"])
    j_pos = np.concatenate(cols["j"])
    dcn = np.concatenate(cols["dc_norm"]).astype(np.float64)
    kap = np.concatenate(cols["kappa"]).astype(np.float64)

    # strata over ROW IDS (prov / ans_len are row-id indexed; positions map through
    # rows_present — identical at production where rows_present == arange)
    i_id = rows_present[i_pos]
    j_id = rows_present[j_pos]
    med = float(np.median(dcn))
    l1, l2 = ans_len[i_id], ans_len[j_id]
    known_elig = (dcn > med) & (l1 >= 0) & (l2 >= 0)
    if known_elig.any():
        dec_edges = np.quantile(((l1[known_elig] + l2[known_elig]) / 2.0), np.linspace(0.1, 0.9, 9))
    else:  # all-unknown lengths (possible only in tiny fixtures): one -1 bucket
        dec_edges = np.zeros(9)
    strata = _pair_strata(i_id, j_id, prov_u8, ans_len, dec_edges)

    sel = _select_kernel_pairs(i_id, j_id, dcn, kap, strata, top_pairs=int(args.top_pairs))
    n_matched = int(sel["n_matched"])
    min_floor = LEG8_MIN_KERNEL_PRODUCTION if production else 10
    if n_matched < min_floor:
        _sentinel(
            "mine-2569",
            f"mining abort: matched kernel pairs {n_matched} < floor {min_floor}",
            {"rc": RC_MINE_ABORT},
        )
        logger.error("[mine] ABORT: matched %d < floor %d (plan §4 leg 8)", n_matched, min_floor)
        sys.exit(RC_MINE_ABORT)

    kidx, cidx = sel["kernel_idx"], sel["control_idx"]

    def _dva(sample_idx: np.ndarray) -> np.ndarray:
        ya = np.asarray(_gather_rows(y_mm, i_pos[sample_idx]), np.float64)
        yb = np.asarray(_gather_rows(y_mm, j_pos[sample_idx]), np.float64)
        return np.linalg.norm(ya - yb, axis=1)

    dva_k, dva_c = _dva(kidx), _dva(cidx)
    stats, ratios = _paired_ratio_stats(dva_k, dva_c)

    # clustered bootstrap over conversation-overlap components (units with a
    # dropped zero-ratio are excluded from BOTH — same mask as the estimator)
    ok = (dva_c > 0) & (dva_k > 0)
    conv_keys = []
    for a_i in np.flatnonzero(ok):
        keys = set()
        for rid in (i_id[kidx[a_i]], j_id[kidx[a_i]], i_id[cidx[a_i]], j_id[cidx[a_i]]):
            ci = int(row_ci[rid])
            keys.add(f"ci{ci}" if ci >= 0 else f"pb{int(rid)}")
        conv_keys.append(keys)
    boot = _clustered_bootstrap_median_ratio(
        ratios, _overlap_clusters(conv_keys), draws=int(args.boot_draws), seed=LEG8_SEED + 1
    )

    held_pos = _positions_in_present(rows_present, np.union1d(val_ids, test_ids), "mine/held")
    floor = _residual_floor(payload, x_mm, y_mm, held_pos, args.device)

    # ── per-pair provenance rows (C2) ─────────────────────────────────────────────
    sv, st = set(val_ids.tolist()), set(test_ids.tolist())

    def _split(rid: int) -> str:
        return "val" if rid in sv else ("test" if rid in st else "train")

    src_name = {0: "lmsys", 1: "wildchat"}

    def _pair_row(s_idx: int) -> dict:
        ri, rj = int(i_id[s_idx]), int(j_id[s_idx])
        return {
            "row_i": ri,
            "row_j": rj,
            "ci_i": int(row_ci[ri]),
            "ci_j": int(row_ci[rj]),
            "split_i": _split(ri),
            "split_j": _split(rj),
            "source_i": src_name.get(int(prov_u8[ri]), str(int(prov_u8[ri]))),
            "source_j": src_name.get(int(prov_u8[rj]), str(int(prov_u8[rj]))),
            "ans_len_i": int(ans_len[ri]),
            "ans_len_j": int(ans_len[rj]),
            "dc_norm": round(float(dcn[s_idx]), 6),
            "kappa": round(float(kap[s_idx]), 8),
            "stratum": int(strata[s_idx]),
        }

    pair_rows = []
    for m_i in range(len(kidx)):
        row = _pair_row(int(kidx[m_i]))
        row["dva_norm"] = round(float(dva_k[m_i]), 6)
        row["control"] = {
            **_pair_row(int(cidx[m_i])),
            "dva_norm": round(float(dva_c[m_i]), 6),
            "matched_tol": float(sel["matched_tol"][m_i]),
        }
        pair_rows.append(row)
    all_rows = np.concatenate([i_id[kidx], j_id[kidx], i_id[cidx], j_id[cidx]])
    in_sample = float(np.mean([_split(int(r)) == "train" for r in all_rows]))
    d_map = int(payload.d)
    n_map_train = int(committed["n_total"]) - len(val_ids) - len(test_ids)
    T24._write_json(
        pairs_path,
        {
            "selection": {k: v for k, v in sel.items() if not isinstance(v, np.ndarray)},
            "in_sample_share_selected_rows": in_sample,
            "in_sample_note": (
                f"{n_map_train}/{int(committed['n_total'])} candidate rows are inside the "
                f"banked map's fit; per-row leverage bound d/n = {d_map}/{n_map_train} "
                f"= {d_map / n_map_train:.4%}; disjoint mining NOT adopted (plan C2: only "
                "1,400 held-out rows exist, all pass-B corpus)"
            ),
            "narration_scope": (
                "a positive result is CONSISTENCY of realized answers with the fitted "
                "map's effective kernel, never validation of information the map ignores"
            ),
            "pairs": pair_rows,
        },
        phase="mine",
    )
    T24._write_json(
        summary_path,
        {
            "n_pairs_sampled": int(args.pairs),
            "chunk": int(args.pair_chunk),
            "seed": LEG8_SEED,
            "b1_assert_iii_probe": probe,
            "selection": {k: v for k, v in sel.items() if not isinstance(v, np.ndarray)},
            "ratio_stats": stats,
            "clustered_bootstrap": boot,
            "residual_floor": floor,
            "floor_read": {
                "kernel_dva_median_over_floor_q50": (
                    float(np.median(dva_k) / floor["q50"]) if floor["q50"] > 0 else float("nan")
                ),
                "control_dva_median_over_floor_q50": (
                    float(np.median(dva_c) / floor["q50"]) if floor["q50"] > 0 else float("nan")
                ),
                "note": "C2: the kernel-vs-control ratio is read against the measured "
                "held-out residual-pair floor, never against 1 alone",
            },
            "ans_len_strata": {
                "n_rows_unknown_len": int((ans_len < 0).sum()),
                "n_rows_total": int(len(ans_len)),
                "decile_edges": [float(x) for x in dec_edges],
            },
            "production": bool(production),
            "regime_config_hash": regime["config_hash"],
        },
        phase="mine",
    )
    _upload_leaf(args, outputs, "leg8", resume_skip=False)
    _sentinel(
        "mine-2569",
        f"leg8 done (matched={n_matched}, median_ratio="
        f"{stats['median_of_paired_ratios']:.4f}, ci95={boot['ci95']})",
    )
    logger.info("[mine] done: %s", json.dumps(stats))


PHASE_ORDER = ("assemble", "moments", "sae-train", "feature-map", "mine")
PHASES = {
    "assemble": phase_assemble,
    "moments": phase_moments,
    "sae-train": phase_sae_train,
    "feature-map": phase_feature_map,
    "mine": phase_mine,
}


# ── CLI ──────────────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Issue #2569 P-B row-battery driver, first half (see module docstring)"
    )
    ap.add_argument("--phase", default="all", choices=["all", *PHASE_ORDER])
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps2569"))
    ap.add_argument(
        "--hf-prefix",
        default="issue2569_theory/analysis_tensors",
        help="HF data-repo destination prefix (issue-owned; never a parent's prefix)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all 1,920 chunks (production)")
    ap.add_argument("--smoke-rows", type=int, default=0, help="0 = full row space (production)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--fresh-stream", action="store_true", help="P1: ignore the stream cursor")
    ap.add_argument("--skip-upload", action="store_true", help="local-only run (loud)")
    ap.add_argument("--gpu-id", type=int, default=-1, help="informational; CVD pins the device")
    ap.add_argument("--layer", type=int, default=LAYER_DEFAULT, choices=list(OP.N1M_LAYERS))
    ap.add_argument(
        "--map-root",
        type=Path,
        default=None,
        help="banked-map root override (else EPS2569_MAP_ROOT env / repo root)",
    )
    ap.add_argument(
        "--moment-chunk",
        type=int,
        default=MOMENT_CHUNK_DEFAULT,
        help="streaming Gram chunk rows (smoke may lower)",
    )
    ap.add_argument(
        "--sae-dict",
        type=int,
        default=0,
        help="SAE-ctx dictionary width (0 = production 65,536; sub-production = smoke-only)",
    )
    ap.add_argument(
        "--sae-steps", type=int, default=0, help="cap SAE optimizer steps (0 = full; smoke seam)"
    )
    ap.add_argument(
        "--resume-across-code-sha",
        action="store_true",
        help="retain completed outputs on a code-SHA-ONLY regime mismatch (T24 passthrough)",
    )
    ap.add_argument(
        "--pairs",
        type=int,
        default=LEG8_PAIRS_DEFAULT,
        help="leg-8 sampled pair count (smoke MUST pass its own M explicitly, e.g. 100000)",
    )
    ap.add_argument(
        "--pair-chunk",
        type=int,
        default=LEG8_CHUNK_PAIRS,
        help="leg-8 pairs per checkpointed GEMM chunk",
    )
    ap.add_argument(
        "--top-pairs",
        type=int,
        default=LEG8_TOP_PAIRS_DEFAULT,
        help="leg-8 kernel-pair count (lowest kernel-fraction, ||dc|| above median)",
    )
    ap.add_argument(
        "--boot-draws",
        type=int,
        default=LEG8_BOOT_DRAWS_DEFAULT,
        help="leg-8 clustered-bootstrap draws",
    )
    ap.add_argument(
        "--answer-sae-dir",
        type=Path,
        default=None,
        help="leg-4 override: local #2476 answer-SAE bundle dir (else staged from HF)",
    )
    ap.add_argument(
        "--alive-counts-npz",
        type=Path,
        default=None,
        help="leg-4 override: local alive_c.npz path (else staged from HF)",
    )
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="leg-8 override: local n1m sampling-manifest dir (else staged from HF)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + call-arity bind + deferred-import resolution",
    )
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Deferred-import resolution (smoke-architecture Axis 1): execute every
        # function-body import of this driver so a missing symbol fails HERE.
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401

        import issue779_ffc_n1m_generate_capture as _N1G  # noqa: F401  (manifest stage)
        import issue2569_gateladder as _GL  # noqa: F401  (moments + featmap consumer)
        from safetensors.torch import load_file  # noqa: F401  (answer-SAE load_local)
        from scipy.stats import mannwhitneyu, rankdata, wilcoxon  # noqa: F401

        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )
        from explore_persona_space.orchestrate.upload_sharded import (
            upload_dir_sharded,  # noqa: F401
        )

        assert callable(_N1G._download_manifest) and callable(_GL.widen_grid)
        assert isinstance(_GL.LAMBDA_GRID_27, tuple) and len(_GL.LAMBDA_GRID_27) == 27
        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_root.mkdir(parents=True, exist_ok=True)
    # B1 driver-entry identity asserts on the banked map (a raise HALTs — apply-
    # path breakage class; stats persisted for the phase entry record).
    payload = OP.load_banked_map(args.layer, root=args.map_root)
    entry = OP.run_driver_identity_asserts(payload)
    T24._write_json(
        args.out_root / "entry_asserts_2569.json",
        {"entry_asserts": entry, "selected_lambda": float(payload.selected_lambda)},
        phase="entry",
    )
    logger.info(
        "[main] phase=%s out_root=%s device=%s smoke=%s max_chunks=%d",
        args.phase,
        args.out_root,
        args.device,
        args.smoke,
        args.max_chunks,
    )
    seq = PHASE_ORDER if args.phase == "all" else (args.phase,)
    for name in seq:
        PHASES[name](args)
    # poller terminal line (pod-side-reporting.md req 1): single reserved emission
    # at the driver's own graceful exit, AFTER the phases' sentinel writes.
    print("[phase=done]", flush=True)
    # explicit exit: heavy C-extension teardown must not rewrite the rc (gotchas.md)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
