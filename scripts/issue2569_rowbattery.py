"""Issue #2569 P-B row battery — FIRST HALF (pre-split unit 4a).

Phase driver over the assembled X19/Y19 row store (plan v4 §4 legs 1/2 moments +
§4 leg 4 step 2). This unit ships three phases:

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

Driver entry runs the B1 identity asserts on the banked L19 map payload
(``issue2569_operator.run_driver_identity_asserts``) — a raise HALTS the driver
(apply-path breakage class). Out-of-scope for this unit (unit 4b extends this
file): leg-4 feature->feature map, leg-8 mining, the H2b refit series.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

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
    """torch.save through a same-dir tmp + os.replace (atomic publish)."""
    tmp = path.parent / (path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


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


PHASE_ORDER = ("assemble", "moments", "sae-train")
PHASES = {
    "assemble": phase_assemble,
    "moments": phase_moments,
    "sae-train": phase_sae_train,
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

        import issue2569_gateladder  # noqa: F401  (moments round-trip consumer)

        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )
        from explore_persona_space.orchestrate.upload_sharded import (
            upload_dir_sharded,  # noqa: F401
        )

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
