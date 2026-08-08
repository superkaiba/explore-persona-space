#!/usr/bin/env python3
"""Issue #1738 — multi-turn prefix/context-arm fits at ~100k (extends the #779
n1m fits driver by IMPORTING its fitters verbatim; plan §4 deltas only).

Deltas vs ``issue779_ffc_n1m_fits`` (whose five fitters + numerics are reused
UNCHANGED via import):

(a) ``--input-arm {prefix,context,both}`` selects X ∈ {``px_last``, ``cx_last``}
    (Y = ``v_x`` always) — the paired mapping arms (standing prefix+context rule).
(b) The split loader reads the Phase-0 pinned ``split_1738.json`` (sha-asserted)
    instead of the parent's ``fixed_split`` + pass_b recovery.
(c) Fit points reduce to ONE (``multiturn_100k``).
(d) After fitting, per-context HOLDOUT predictions are RETAINED per fitter
    (fp16, the #1482 pdshrink convention) and the standing mapping-baselines
    pair runs per arm × layer: identity+learned-bias
    (``analysis/mapping_baselines.identity_bias_predict``; dims match,
    3584→3584) and kNN retrieval (``knn_retrieval``; ks {1,5,10} over the
    holdout pool, euclidean + cosine, chance = k/n_pool stated).

Also: the ``lmsys_transfer`` group-level OOD control (fit ridge on LMSYS-only
train rows, score WildChat holdout rows; layer 19, both arms), and the G2
in-run first-cell timing fence (designed halt: fence_report.json + rc 24 —
never a bare rc=1, per the pilot-gate routing rule).

KRR Nystrom m is passed EXPLICITLY (default 16384 here — the parent script
default is 8192; plan §11 NOTE) with ``--krr-solver cholesky``.

Compute: 1× A100-80 (`capture-7b`). Streams capture chunks from HF into
per-(array, layer) append-only fp32 binaries (+ cursor checkpoint; resume-safe),
then memmaps them. Refusal-safety: chunk text fields are never printed/logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847: thread caps land BEFORE numpy/torch import on the shared VM

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as PF  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1738_multiturn_generate_capture as GG  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1738_mt_fits")

FIT_POINT = "multiturn_100k"
ARMS = {"prefix": "px_last", "context": "cx_last", "bare": "bq_last"}
# memmap key per arm (replaces the former hardcoded px/cx ternaries; plan §4.2)
ARM_MM_KEY = {"prefix": "px", "context": "cx", "bare": "bq"}
# `--input-arm both` keeps the PARENT semantics (prefix+context only); the bare
# arm is a separate single-arm round (follow-up `bare-query`, plan §4.2).
BOTH_ARMS = ("prefix", "context")
ARM_ORDER = ("context", "prefix", "bare")  # (context, L19) first = the G2 pilot cell
LAYERS_DEFAULT = (14, 19, 26)
LAMBDAS = PF.LAMBDAS_N1M  # logspace(-3, 8, 23), parent grid verbatim
PREDICTORS = PF.PREDICTORS
MLP_LRS = tuple(F.MLP_LRS)  # (1e-3, 3e-4) — round-1 grid (plan §11)
KRR_CENTERS_DEFAULT = 16_384  # plan §11 NOTE: NOT the parent script default (8192)
N_BOOT = 10_000
BOOT_SEED = 1738
RC_FENCE = 24  # G2 designed-halt rc (report written first; never a bare rc=1)
H_DIM = C.EXPECTED_HIDDEN  # 3584
DEFAULT_OUT_EVAL = PROJECT_ROOT / "eval_results" / "issue_1738"
DEFAULT_OUT_LOCAL = PROJECT_ROOT / "data" / "issue_1738" / "mt100k" / "fits"
ANALYSIS_TENSORS_SUBDIR = "analysis_tensors"


# ── split loader (delta b: pinned split_1738.json, sha-asserted) ──────────────────


def load_split(split_path: Path) -> dict:
    """Load + sha-assert the pinned split doc: every set's recorded sha256 must
    reproduce from its own ci list (a corrupted/hand-edited doc fails loud)."""
    doc = json.loads(split_path.read_text())
    for name, s in doc["sets"].items():
        got = GG._sha_int_list([int(c) for c in s["ci"]])
        assert got == s["sha256"], f"split set {name!r} sha mismatch: {got} != {s['sha256']}"
    return doc


def split_positions(doc: dict, ci: np.ndarray) -> dict[str, np.ndarray]:
    """Map the split doc's GLOBAL ci sets to CAPTURED row positions. Reports
    realized coverage (over-length/violation skips make captured ⊂ manifest)."""
    pos_of = {int(c): p for p, c in enumerate(ci.tolist())}
    out = {}
    for name, s in doc["sets"].items():
        rows = np.asarray(sorted(pos_of[c] for c in s["ci"] if int(c) in pos_of), dtype=np.int64)
        out[name] = rows
        logger.info("[split] %s: %d/%d intended rows captured", name, len(rows), s["n"])
    inter = set(out["train"].tolist()) & (
        set(out["val"].tolist()) | set(out["test"].tolist()) | set(out["holdout"].tolist())
    )
    assert not inter, f"train overlaps eval sets at {len(inter)} rows"
    return out


def _coverage_shortfalls(sets: dict[str, np.ndarray], doc: dict, floor: float) -> list[str]:
    """Pure fail-loud predicate for the capture-coverage floor (review Minor 1):
    a per-set captured/intended ratio below ``floor`` (e.g. a missing fleet
    shard) returns a naming string per offending set; empty = OK."""
    bad = []
    for name, rows in sets.items():
        intended = int(doc["sets"][name]["n"])
        if intended and len(rows) / intended < floor:
            bad.append(f"{name}: {len(rows)}/{intended} = {len(rows) / intended:.3f} < {floor}")
    return bad


def _mlp_lr_better(vr2: float, best_vr2: float | None) -> bool:
    """True when ``vr2`` should replace the incumbent val-R²: no incumbent, a
    non-finite incumbent (NaN-first divergence — review Minor 2), or a finite
    improvement. NaN never beats a finite incumbent."""
    if best_vr2 is None:
        return True
    if not np.isfinite(best_vr2):
        return bool(np.isfinite(vr2))
    return bool(np.isfinite(vr2) and vr2 > best_vr2)


# ── streaming chunk assembly → per-(array, layer) fp32 binaries + memmaps ─────────


def _chunk_names(args) -> list[str]:
    if args.local_capture_dir:
        names = sorted(p.name for p in Path(args.local_capture_dir).glob("*.pt"))
    else:
        names = sorted(
            n
            for n in GG.N50._remote_index(f"{args.hf_prefix}/{GG.CAPTURE_SUBDIR}")
            if n.endswith(".pt")
        )
    if not names:
        raise SystemExit("no capture chunks found — run the capture phase first")
    return names


def _mm_paths(mm_dir: Path, layers: list[int]) -> dict:
    out = {"cursor": mm_dir / "cursor.json", "ci": mm_dir / "ci.bin", "meta": {}}
    for arr in ("px", "cx", "vx"):
        for li in layers:
            out[(arr, li)] = mm_dir / f"{arr}_L{li}.bin"
    return out


def assemble_streams(args, layers: list[int]):
    """Stream capture chunks (HF or local) into append-only fp32 binaries per
    (array, layer) + an int64 ci binary, with a cursor checkpoint every chunk
    batch (resume truncates to the cursor row count — the external-stream
    checkpoint law). Returns (mm dict of np.memmap, ci int64 array, meta)."""
    mm_dir = Path(args.mm_dir)
    mm_dir.mkdir(parents=True, exist_ok=True)
    names = _chunk_names(args)
    fp = hashlib.sha256(
        ("\n".join(names) + f"|{args.hf_prefix}|{sorted(layers)}").encode()
    ).hexdigest()
    paths = _mm_paths(mm_dir, layers)
    cursor = {"fingerprint": fp, "n_chunks_done": 0, "n_rows": 0}
    if paths["cursor"].exists():
        prev = json.loads(paths["cursor"].read_text())
        if prev.get("fingerprint") == fp:
            cursor = prev
            logger.info(
                "[assemble] resume: %d chunks / %d rows done",
                cursor["n_chunks_done"],
                cursor["n_rows"],
            )
        else:
            logger.info("[assemble] cursor fingerprint mismatch — fresh assembly")
            for k, p in paths.items():
                if k != "meta" and Path(p).exists():
                    Path(p).unlink()
    # truncate binaries to the cursor row count (crash mid-append leaves a tail)
    n_rows = int(cursor["n_rows"])
    for arr in ("px", "cx", "vx"):
        for li in layers:
            p = paths[(arr, li)]
            want = n_rows * H_DIM * 4
            if p.exists() and p.stat().st_size != want:
                with open(p, "r+b") as f:
                    f.truncate(want)
            elif not p.exists():
                p.touch()
    ci_p = paths["ci"]
    if ci_p.exists() and ci_p.stat().st_size != n_rows * 8:
        with open(ci_p, "r+b") as f:
            f.truncate(n_rows * 8)
    elif not ci_p.exists():
        ci_p.touch()

    key_of = {"px": "px_last", "cx": "cx_last", "vx": "v_x"}
    cache = mm_dir / "dl_cache"
    cache.mkdir(exist_ok=True)
    handles = {k: open(paths[k], "ab") for k in paths if isinstance(k, tuple)}
    ci_f = open(ci_p, "ab")
    try:
        for k, name in enumerate(names):
            if k < cursor["n_chunks_done"]:
                continue
            if args.local_capture_dir:
                local = Path(args.local_capture_dir) / name
            else:
                local = Path(
                    PF._download_chunk_with_retry(
                        C.HF_DATA_REPO, f"{args.hf_prefix}/{GG.CAPTURE_SUBDIR}/{name}", cache
                    )
                )
            bundle = torch.load(local, map_location="cpu", weights_only=False)
            blayers = list(bundle["layers"])
            li_pos = {li: blayers.index(li) for li in layers}
            n = len(bundle["ci"])
            for arr in ("px", "cx", "vx"):
                t = bundle[key_of[arr]]
                assert t.shape == (n, len(blayers), H_DIM), (name, arr, t.shape)
                for li in layers:
                    handles[(arr, li)].write(
                        np.ascontiguousarray(
                            t[:, li_pos[li], :].numpy().astype(np.float32)
                        ).tobytes()
                    )
            ci_f.write(np.asarray(bundle["ci"], dtype=np.int64).tobytes())
            n_rows += n
            cursor.update({"n_chunks_done": k + 1, "n_rows": n_rows})
            if not args.local_capture_dir:
                local.unlink(missing_ok=True)  # purge — peak footprint ~one chunk
            if (k + 1) % 25 == 0 or (k + 1) == len(names):
                for h in handles.values():
                    h.flush()
                ci_f.flush()
                GG.N1M._atomic_write_json(paths["cursor"], cursor)
                logger.info("[assemble] chunk %d/%d (%d rows)", k + 1, len(names), n_rows)
    finally:
        for h in handles.values():
            h.close()
        ci_f.close()
    GG.N1M._atomic_write_json(paths["cursor"], cursor)
    ci = np.fromfile(ci_p, dtype=np.int64)
    assert len(ci) == n_rows, (len(ci), n_rows)
    assert len(set(ci.tolist())) == n_rows, "duplicate ci across chunks"
    mm = {
        (arr, li): np.memmap(paths[(arr, li)], dtype=np.float32, mode="r", shape=(n_rows, H_DIM))
        for arr in ("px", "cx", "vx")
        for li in layers
    }
    return mm, ci, {"n_rows": n_rows, "n_chunks": len(names), "fingerprint": fp}


# ── bare-arm assembly: stream bq chunks + ci-keyed reorder to parent order ────────
# (follow-up `bare-query`, plan §4.2)


def _bare_chunk_names(args) -> list[str]:
    if args.local_bare_dir:
        names = sorted(p.name for p in Path(args.local_bare_dir).glob("*.pt"))
    else:
        names = sorted(
            n
            for n in GG.N50._remote_index(f"{args.bare_hf_prefix}/{GG.CAPTURE_SUBDIR}")
            if n.endswith(".pt")
        )
    if not names:
        raise SystemExit("no bare-query capture chunks found — run --bare-query capture first")
    return names


def assemble_bare_streams(args, layers: list[int], parent_ci: np.ndarray, parent_fp: str):
    """Stream bare-query chunks (HF or local) into raw fp32 binaries (cursor-
    checkpointed, resume-safe), then write per-layer ``bq`` memmaps REORDERED to
    the PARENT capture ci order via one vectorized fancy-index (plan §4.2).

    Coverage assert (1:1): every parent captured ci MUST be present in the bare
    store — missing ⇒ fail loud; the ≤873 extra bare rows (parent over-length
    skips) are recorded in ``bq_meta.json`` and dropped. Returns
    ``({("bq", li): memmap aligned to parent rows}, bare_meta)``."""
    mm_dir = Path(args.mm_dir)
    mm_dir.mkdir(parents=True, exist_ok=True)
    names = _bare_chunk_names(args)
    fp = hashlib.sha256(
        ("\n".join(names) + f"|{args.bare_hf_prefix}|{sorted(layers)}|{parent_fp}").encode()
    ).hexdigest()
    raw_paths = {li: mm_dir / f"bqraw_L{li}.bin" for li in layers}
    out_paths = {li: mm_dir / f"bq_L{li}.bin" for li in layers}
    ci_p = mm_dir / "bare_ci.bin"
    cursor_p = mm_dir / "bare_cursor.json"
    meta_p = mm_dir / "bq_meta.json"
    n_parent = len(parent_ci)
    if meta_p.exists():  # reorder already done under this fingerprint — resume-skip
        prev = json.loads(meta_p.read_text())
        if prev.get("fingerprint") == fp and all(out_paths[li].exists() for li in layers):
            logger.info("[bare-assemble] resume: reordered bq memmaps present (%d rows)", n_parent)
            mm = {
                ("bq", li): np.memmap(
                    out_paths[li], dtype=np.float32, mode="r", shape=(n_parent, H_DIM)
                )
                for li in layers
            }
            return mm, prev
    cursor = {"fingerprint": fp, "n_chunks_done": 0, "n_rows": 0}
    if cursor_p.exists():
        prev = json.loads(cursor_p.read_text())
        if prev.get("fingerprint") == fp:
            cursor = prev
            logger.info(
                "[bare-assemble] resume: %d chunks / %d rows done",
                cursor["n_chunks_done"],
                cursor["n_rows"],
            )
        else:
            logger.info("[bare-assemble] cursor fingerprint mismatch — fresh assembly")
            for p in [*raw_paths.values(), ci_p]:
                Path(p).unlink(missing_ok=True)
    n_rows = int(cursor["n_rows"])
    for li in layers:
        p = raw_paths[li]
        want = n_rows * H_DIM * 4
        if p.exists() and p.stat().st_size != want:
            with open(p, "r+b") as f:
                f.truncate(want)
        elif not p.exists():
            p.touch()
    if ci_p.exists() and ci_p.stat().st_size != n_rows * 8:
        with open(ci_p, "r+b") as f:
            f.truncate(n_rows * 8)
    elif not ci_p.exists():
        ci_p.touch()
    cache = mm_dir / "bare_dl_cache"
    cache.mkdir(exist_ok=True)
    handles = {li: open(raw_paths[li], "ab") for li in layers}
    ci_f = open(ci_p, "ab")
    try:
        for k, name in enumerate(names):
            if k < cursor["n_chunks_done"]:
                continue
            if args.local_bare_dir:
                local = Path(args.local_bare_dir) / name
            else:
                local = Path(
                    PF._download_chunk_with_retry(
                        C.HF_DATA_REPO, f"{args.bare_hf_prefix}/{GG.CAPTURE_SUBDIR}/{name}", cache
                    )
                )
            bundle = torch.load(local, map_location="cpu", weights_only=False)
            blayers = list(bundle["layers"])
            li_pos = {li: blayers.index(li) for li in layers}
            n = len(bundle["ci"])
            t = bundle["bq_last"]
            assert t.shape == (n, len(blayers), H_DIM), (name, t.shape)
            for li in layers:
                handles[li].write(
                    np.ascontiguousarray(t[:, li_pos[li], :].numpy().astype(np.float32)).tobytes()
                )
            ci_f.write(np.asarray(bundle["ci"], dtype=np.int64).tobytes())
            n_rows += n
            cursor.update({"n_chunks_done": k + 1, "n_rows": n_rows})
            if not args.local_bare_dir:
                local.unlink(missing_ok=True)  # purge — peak footprint ~one chunk
            if (k + 1) % 25 == 0 or (k + 1) == len(names):
                for h in handles.values():
                    h.flush()
                ci_f.flush()
                GG.N1M._atomic_write_json(cursor_p, cursor)
                logger.info("[bare-assemble] chunk %d/%d (%d rows)", k + 1, len(names), n_rows)
    finally:
        for h in handles.values():
            h.close()
        ci_f.close()
    GG.N1M._atomic_write_json(cursor_p, cursor)
    bare_ci = np.fromfile(ci_p, dtype=np.int64)
    assert len(bare_ci) == n_rows, (len(bare_ci), n_rows)
    assert len(set(bare_ci.tolist())) == n_rows, "duplicate ci across bare chunks"
    # ci-keyed reorder to the parent capture order (vectorized fancy-index).
    pos_of = {int(c): p for p, c in enumerate(bare_ci.tolist())}
    missing = [int(c) for c in parent_ci.tolist() if int(c) not in pos_of]
    assert not missing, (
        f"bare store missing {len(missing)} parent captured ci (first {missing[:5]}) — "
        "1:1 coverage violated (plan §4.2); backfill the bare capture"
    )
    perm = np.asarray([pos_of[int(c)] for c in parent_ci.tolist()], dtype=np.int64)
    extra = sorted(set(bare_ci.tolist()) - {int(c) for c in parent_ci.tolist()})
    for li in layers:
        raw = np.memmap(raw_paths[li], dtype=np.float32, mode="r", shape=(n_rows, H_DIM))
        out = np.ascontiguousarray(raw[perm])
        with open(out_paths[li], "wb") as f:
            f.write(out.tobytes())
        del raw, out
    meta = {
        "fingerprint": fp,
        "n_bare_rows": int(n_rows),
        "n_parent_rows": int(n_parent),
        "n_extra_dropped": len(extra),
        "extra_ci_head": [int(x) for x in extra[:20]],
        "n_chunks": len(names),
    }
    GG.N1M._atomic_write_json(meta_p, meta)
    logger.info(
        "[bare-assemble] reordered %d rows to parent ci order (%d extra bare rows dropped)",
        n_parent,
        len(extra),
    )
    mm = {
        ("bq", li): np.memmap(out_paths[li], dtype=np.float32, mode="r", shape=(n_parent, H_DIM))
        for li in layers
    }
    return mm, meta


def _assert_parent_split_shas(split: dict, parent_fits_json: str) -> dict:
    """Bare-arm cross-assert (plan §4.2, consistency-check note round 1): the new
    run's split_shas must equal the PARENT fits JSON's recorded split_shas — pins
    content identity to the parent's REALIZED split, not only the loader's
    internal shas. Returns the parent shas for the summary record."""
    pj = Path(parent_fits_json)
    assert pj.is_file(), (
        f"--parent-fits-json missing: {pj} — the bare arm requires the parent "
        "split_shas cross-assert (plan §4.2)"
    )
    parent_shas = json.loads(pj.read_text())["split_shas"]
    own = {k: split["sets"][k]["sha256"] for k in split["sets"]}
    assert own == parent_shas, (
        f"split_shas != parent fits JSON ({pj}): own={own} parent={parent_shas}"
    )
    logger.info("[split] bare-arm split_shas cross-assert vs parent fits JSON OK")
    return parent_shas


# ── fit dispatch (parent fitters verbatim; MLP lr grid val-selected) ──────────────


def _fitargs(args, mlp_lr: float) -> SimpleNamespace:
    return SimpleNamespace(
        ridge_block=args.ridge_block,
        mlp_lr=mlp_lr,
        mlp_max_epochs=args.mlp_max_epochs,
        mlp_batch=args.mlp_batch,
        seed=args.seed,
        krr_nystrom_centers=args.krr_nystrom_centers,
        krr_solver=args.krr_solver,
    )


def fit_predictor(name, X, Y, tr, val, te_all, args, dev, *, resid_lr: float):
    """One predictor on eval rows ``te_all`` (= test ∥ holdout). MLP fitters run
    the round-1 lr grid {1e-3, 3e-4} with val-R² selection (val rides in the
    eval concat); ridge/KRR select on val inside the parent fitters."""
    if name in ("mlp_w8192", "mlp_w32768"):
        width = PF.MLP_W_PROTOCOL if name == "mlp_w8192" else PF.MLP_W_CAPACITY
        eval_idx = np.concatenate([val, te_all])
        best = None
        for lr in args.mlp_lrs_list:
            pred, meta = PF.fit_mlp(
                X,
                Y,
                tr,
                eval_idx,
                width,
                lr,
                args.mlp_max_epochs,
                args.mlp_batch,
                args.seed,
                dev,
                capacity_arm=(name == "mlp_w32768"),
            )
            vr2 = PR._pooled_r2(pred[: len(val)], np.asarray(Y[val]))
            meta = {**meta, "lr": float(lr), "val_r2": float(vr2)}
            if _mlp_lr_better(vr2, best[0] if best is not None else None):
                best = (vr2, pred[len(val) :], meta)
        pred_te, meta = best[1], best[2]
        meta["selected_lr"] = meta["lr"]
        meta["lr_grid"] = [float(x) for x in args.mlp_lrs_list]
        return pred_te, meta
    fa = _fitargs(args, resid_lr)
    return PF._fit_one_predictor(
        name, X, Y, tr, val, te_all, LAMBDAS, args.krr_gamma_mult, args.krr_lambdas_list, fa, dev
    )


def _percontext_nerr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """nerr(x) = ||v̂−v||² / ||v−μ_eval||² per context (the #1482 convention)."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mu = true.mean(axis=0)
    num = ((true - pred) ** 2).sum(axis=1)
    den = ((true - mu) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def _fence_should_halt(elapsed_s: float, first_wall_s: float, n_cells: int, mult: float) -> bool:
    """G2 pure predicate: halt once elapsed exceeds mult × first-cell wall ×
    n_cells (the ≥2× first-cell extrapolation fence, plan §7)."""
    return elapsed_s > mult * first_wall_s * n_cells


def _boot_recon_ci_batched(pred: np.ndarray, true: np.ndarray, n_boot: int, seed: int, chunk=500):
    """BATCHED context bootstrap of (R², mean cosine) — draw-identical to
    ``F._bootstrap_recon_ci`` (same rng stream, same per-draw math: multiplicity-
    weighted SS_res and SS_tot around the RESAMPLE mean) but expressed as chunked
    counts@pool GEMMs instead of a per-draw Python loop (the vectorize-first law;
    equivalence pinned in the fits ``--smoke``). Returns F's dict shape."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = pred.shape[0]
    res_i = ((true - pred) ** 2).sum(axis=1)
    cos_i = PR._per_context_cosine(pred, true)
    r2_point, cos_point = F._recon_point(pred, true)
    y_norm2 = (true**2).sum(axis=1)
    cos_fill = np.nan_to_num(cos_i, nan=0.0, posinf=0.0, neginf=0.0)
    cos_fin = np.isfinite(cos_i).astype(np.float64)
    rng = np.random.default_rng(seed)
    r2s = np.full(n_boot, np.nan)
    coss = np.full(n_boot, np.nan)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(b, n))
        counts = np.zeros((b, n), dtype=np.float64)
        np.add.at(counts, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
        ss_res = counts @ res_i
        sum_y = counts @ true  # (b, H)
        ss_tot = counts @ y_norm2 - ((sum_y / n) ** 2).sum(axis=1) * n
        ok = ss_tot > 1e-12
        r2s[s : s + b][ok] = 1.0 - ss_res[ok] / ss_tot[ok]
        c_cnt = counts @ cos_fin
        with np.errstate(invalid="ignore", divide="ignore"):
            coss[s : s + b] = (counts @ cos_fill) / c_cnt

    def _ci(pt, boots):
        boots = boots[np.isfinite(boots)]
        if boots.size == 0:
            return {"point": pt, "lo": float("nan"), "hi": float("nan")}
        return {
            "point": pt,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
        }

    return {"r2": _ci(r2_point, r2s), "mean_cosine": _ci(cos_point, coss), "n_test": int(n)}


def run_fits(args) -> int:
    layers = [int(x) for x in args.layers.split(",")]
    arms = list(BOTH_ARMS) if args.input_arm == "both" else [args.input_arm]
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.mlp_lrs_list = [float(x) for x in args.mlp_lrs.split(",")]
    args.krr_gamma_mult = tuple(float(x) for x in args.krr_gamma_mult_s.split(","))
    args.krr_lambdas_list = tuple(float(x) for x in args.krr_lambdas_s.split(","))

    C.phase("fits-assemble")
    mm, ci, ameta = assemble_streams(args, layers)
    afp = ameta["fingerprint"]  # assembly fingerprint — the resume regime key
    bare_meta = None
    if "bare" in arms:
        bare_mm, bare_meta = assemble_bare_streams(args, layers, ci, afp)
        mm.update(bare_mm)
    split = load_split(Path(args.split_file))
    if "bare" in arms:
        _assert_parent_split_shas(split, args.parent_fits_json)
    sets = split_positions(split, ci)
    shortfalls = _coverage_shortfalls(sets, split, args.min_split_coverage)
    if shortfalls:
        raise SystemExit(
            f"capture coverage below floor: {'; '.join(shortfalls)} — a fleet shard is "
            "likely missing; backfill the capture or lower --min-split-coverage deliberately"
        )
    tr, val, te, ho = sets["train"], sets["val"], sets["test"], sets["holdout"]
    n_tr, d = len(tr), H_DIM
    if n_tr < d and not args.allow_underdetermined:
        raise SystemExit(
            f"n_train={n_tr} < d={d}: estimator-degenerate regime — pass "
            "--allow-underdetermined only for a deliberate smoke shape"
        )
    te_all = np.concatenate([te, ho])
    corpus_by_ci = {}
    if args.manifest_dir:
        pool, _m = GG.N1M.read_manifest_pool(Path(args.manifest_dir))
        corpus_by_ci = {int(r["i"]): r["corpus"] for r in pool}

    # cell order: (context, L19) FIRST — the G2 in-run timing pilot cell (a
    # bare-only round's first cell is (bare, L19), the same pilot semantics).
    layer_order = ([19] if 19 in layers else []) + [x for x in layers if x != 19]
    arm_order = [a for a in ARM_ORDER if a in arms]
    cells = [(a, li) for a in arm_order for li in layer_order]
    cells_dir = args.out_eval / "fits" / "cells"
    pc_dir = args.out_eval / "percontext"
    pred_dir = args.out_local / "pred16"
    yh_dir = args.out_local / "y_holdout"
    for p in (cells_dir, pc_dir, pred_dir, yh_dir, args.out_local / "weights"):
        p.mkdir(parents=True, exist_ok=True)

    # persist Y holdout (fp16) per layer — the off-pod H1 bootstrap input.
    # Skip-if-exists is keyed on the ASSEMBLY FINGERPRINT (#722-r3 resume-regime
    # class): a capture-set-changed re-run regenerates rather than silently
    # pairing stale y_holdout rows with fresh pred16 rows.
    for li in layers:
        yhp = yh_dir / f"L{li}.npz"
        if yhp.exists():
            with np.load(yhp) as z:
                stored_fp = z["fingerprint"].item() if "fingerprint" in z.files else ""
            if stored_fp == afp:
                continue
            logger.info("[fits] y_holdout L%d assembly-fingerprint mismatch — regenerating", li)
        np.savez(
            yhp,
            y16=np.asarray(mm[("vx", li)][ho], dtype=np.float16),
            ci=ci[ho],
            fingerprint=np.array(afp),
        )

    C.phase("fits")
    first_wall: float | None = None
    t_cells0 = time.time()
    n_cells_total = len(cells)
    summary: dict = {
        "fit_point": FIT_POINT,
        "layers": layers,
        "arms": arms,
        "n_rows_captured": ameta["n_rows"],
        "split_counts": {k: int(len(v)) for k, v in sets.items()},
        "split_shas": {k: split["sets"][k]["sha256"] for k in split["sets"]},
        "lambdas": [float(x) for x in LAMBDAS],
        "mlp_lr_grid": args.mlp_lrs_list,
        "krr": {"m_centers": int(args.krr_nystrom_centers), "solver": args.krr_solver},
        "n_boot": int(args.n_boot),
        "boot_seed": BOOT_SEED,
        "cells": {},
        "git_commit": os.environ.get("EPM_GIT_COMMIT", ""),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if bare_meta is not None:  # coverage + dropped-extras record (plan §4.2)
        summary["bare_assembly"] = bare_meta
    for cell_i, (arm, li) in enumerate(cells):
        # G2 fence: designed halt once elapsed exceeds fence_mult × first-cell
        # extrapolation (report JSON + distinct rc — never an anonymous crash).
        if first_wall is not None:
            budget = args.fence_mult * first_wall * n_cells_total
            elapsed = time.time() - t_cells0
            if _fence_should_halt(elapsed, first_wall, n_cells_total, args.fence_mult):
                rep = {
                    "gate": "G2",
                    "first_cell_wall_s": first_wall,
                    "elapsed_s": elapsed,
                    "budget_s": budget,
                    "fence_mult": args.fence_mult,
                    "cells_done": cell_i,
                    "cells_total": n_cells_total,
                }
                GG.N1M._atomic_write_json(args.out_eval / "fits" / "fence_report.json", rep)
                logger.error("[G2] fence tripped: %s", rep)
                if not args.no_upload:
                    try:  # r4 belt: persist the report + partial cell JSONs to
                        # HF, but the DESIGNED rc must survive an upload failure
                        # (artifact-first halt routing) — hence log-and-exit.
                        _upload_analysis_tensors(args, [_summary_upload_entry(args)])
                    except Exception:
                        logger.exception("[G2] fence-report upload failed (rc %d kept)", RC_FENCE)
                sys.exit(RC_FENCE)
        X = mm[(ARM_MM_KEY[arm], li)]
        Y = mm[("vx", li)]
        t_cell = time.time()
        resid_lr = float(args.mlp_lrs_list[-1])
        ridge_refit = False
        for name in [p for p in PREDICTORS if p in args.predictors.split(",")]:
            cj = cells_dir / f"{arm}_L{li}_{name}.json"
            # seed-43 repeat NOT re-run on the bare arm (plan §10 Seeds row:
            # "fit seeds 42 (+43 not re-run — single-arm round)").
            want_seed43 = name == "mlp_w8192" and li == 19 and arm != "bare"
            if cj.exists() and not args.no_resume:
                doc = json.loads(cj.read_text())
                # resume keyed on the assembly fingerprint (#722-r3 regime-key
                # class) — a capture-set-changed re-run refits, never resumes.
                fp_ok = doc.get("assembly_fingerprint") == afp
                s43_ok = (not want_seed43) or "seed43" in doc.get("metrics", {})
                if fp_ok and s43_ok:
                    summary["cells"][f"{arm}_L{li}_{name}"] = doc["metrics"]
                    if name == "mlp_w8192" and "selected_lr" in doc.get("fit_meta", {}):
                        resid_lr = float(doc["fit_meta"]["selected_lr"])
                    logger.info("[cell] %s_L%d_%s: resume-skip", arm, li, name)
                    continue
                logger.info(
                    "[cell] %s_L%d_%s: stale cell (fingerprint%s) — refit",
                    arm,
                    li,
                    name,
                    "" if fp_ok else " mismatch",
                )
            t0 = time.time()
            pred_all, meta = fit_predictor(
                name, X, Y, tr, val, te_all, args, dev, resid_lr=resid_lr
            )
            if name == "mlp_w8192" and "selected_lr" in meta:
                resid_lr = float(meta["selected_lr"])  # residual_skip inherits the selected lr
            if name == "ridge":
                ridge_refit = True
            pred_te, pred_ho = pred_all[: len(te)], pred_all[len(te) :]
            y_te, y_ho = np.asarray(Y[te], dtype=np.float64), np.asarray(Y[ho], dtype=np.float64)
            r2_te, cos_te = F._recon_point(pred_te, y_te)
            r2_ho, cos_ho = F._recon_point(pred_ho, y_ho)
            ci_boot = _boot_recon_ci_batched(pred_ho, y_ho, args.n_boot, BOOT_SEED)
            nerr = _percontext_nerr(pred_ho, y_ho).astype(np.float32)
            np.savez(
                pc_dir / f"{arm}_L{li}_{name}.npz",
                nerr=nerr,
                ci=ci[ho],
                fingerprint=np.array(afp),
            )
            np.savez(
                pred_dir / f"{arm}_L{li}_{name}.npz",
                pred16=pred_ho.astype(np.float16),
                ci=ci[ho],
                fingerprint=np.array(afp),
            )
            metrics = {
                "test_r2": float(r2_te),
                "test_mean_cosine": float(cos_te),
                "holdout_r2": float(r2_ho),
                "holdout_mean_cosine": float(cos_ho),
                "holdout_bootstrap_ci": ci_boot,
                "n_test": int(len(te)),
                "n_holdout": int(len(ho)),
                "wall_s": time.time() - t0,
            }
            if want_seed43:
                # second-seed MLP holdout read (plan §10 Seeds row, #1482
                # convention; placed at the registered layer 19 — plan §13
                # names the placement deviatable): one extra fit at the
                # WINNING lr, seed 43, holdout eval rows only.
                lr43 = float(meta.get("selected_lr", args.mlp_lrs_list[-1]))
                pred43, _m43 = PF.fit_mlp(
                    X,
                    Y,
                    tr,
                    ho,
                    PF.MLP_W_PROTOCOL,
                    lr43,
                    args.mlp_max_epochs,
                    args.mlp_batch,
                    43,
                    dev,
                )
                r2_43, cos_43 = F._recon_point(pred43, y_ho)
                nerr43 = _percontext_nerr(pred43, y_ho).astype(np.float32)
                np.savez(
                    pc_dir / f"{arm}_L{li}_{name}_seed43.npz",
                    nerr=nerr43,
                    ci=ci[ho],
                    fingerprint=np.array(afp),
                )
                np.savez(
                    pred_dir / f"{arm}_L{li}_{name}_seed43.npz",
                    pred16=pred43.astype(np.float16),
                    ci=ci[ho],
                    fingerprint=np.array(afp),
                )
                metrics["seed43"] = {
                    "seed": 43,
                    "lr": lr43,
                    "holdout_r2": float(r2_43),
                    "holdout_mean_cosine": float(cos_43),
                    "seed_pair_nerr_pearson": float(np.corrcoef(nerr, nerr43)[0, 1])
                    if len(nerr) > 2
                    else float("nan"),
                }
            GG.N1M._atomic_write_json(
                cj,
                {
                    "arm": arm,
                    "layer": li,
                    "fitter": name,
                    "metrics": metrics,
                    "fit_meta": meta,
                    "assembly_fingerprint": afp,
                },
            )
            summary["cells"][f"{arm}_L{li}_{name}"] = metrics
            logger.info(
                "[cell] %s_L%d_%s: test R2=%.4f holdout R2=%.4f (%.0fs)",
                arm,
                li,
                name,
                r2_te,
                r2_ho,
                metrics["wall_s"],
            )
        # ridge weights persisted per cell (small; apply_map-compatible); the
        # write re-fires when the ridge cell itself refit (fingerprint change).
        wj = args.out_local / "weights" / f"L{li}" / f"{arm}_ridge.pt"
        if (ridge_refit or not wj.exists()) and "ridge" in args.predictors.split(","):
            _, _, payload = PF.fit_ridge_with_weights(
                X, Y, tr, val, te, LAMBDAS, dev, args.ridge_block
            )
            wj.parent.mkdir(parents=True, exist_ok=True)
            tmp = wj.parent / (wj.name + ".tmp")
            torch.save({**payload, "arm": arm, "layer": li, "assembly_fingerprint": afp}, tmp)
            os.replace(tmp, wj)
        if first_wall is None:
            first_wall = time.time() - t_cell
            logger.info(
                "[G2] first cell (%s, L%d) wall=%.0fs → projected total ≈ %.1f h (fence %.1f h)",
                arm,
                li,
                first_wall,
                first_wall * n_cells_total / 3600,
                args.fence_mult * first_wall * n_cells_total / 3600,
            )

    C.phase("fits-baselines")
    baselines = _compute_baselines(mm, tr, ho, cells, pred_dir)
    GG.N1M._atomic_write_json(args.out_eval / "mapping_baselines.json", baselines)

    C.phase("fits-transfer")
    transfer = _compute_transfer(mm, ci, arms, split, tr, val, ho, corpus_by_ci, dev, args)
    summary["lmsys_transfer"] = transfer
    GG.N1M._atomic_write_json(args.out_eval / "fits" / f"{FIT_POINT}_fits.json", summary)

    if not args.no_upload:
        C.phase("fits-upload")
        # percontext/*.npz is a plan §6.5 primary deliverable: gitignored by the
        # repo-wide *.npz rule (#958 class) and consumed off-pod by Phase 4c, so
        # it MUST ride the HF analysis_tensors upload (plan §10) — the
        # DELETE-on-exit GCE lane would otherwise lose it (review Major 2).
        # r4: the KB-scale summary JSONs ride the same upload (dual-write) —
        # their git-only destination lost both summaries when the GCE instance
        # was reaped before any harvest (#1738 r4 incident).
        _upload_analysis_tensors(
            args,
            [
                ("pred16", args.out_local / "pred16", None),
                ("y_holdout", args.out_local / "y_holdout", None),
                ("weights", args.out_local / "weights", None),
                ("percontext", pc_dir, None),
                _summary_upload_entry(args),
            ],
        )
    C.phase("done")
    return 0


def _compute_baselines(mm, tr, ho, cells, pred_dir) -> dict:
    """Standing mapping-baselines pair per (arm, layer): identity+learned-bias
    holdout R²/cosine + kNN retrieval (identity_bias + retained ridge pred16).
    Extracted verbatim from run_fits so --rebuild-summaries shares it (r4)."""
    baselines: dict = {"ks": [1, 5, 10], "metrics": ["euclidean", "cosine"], "cells": {}}
    for arm, li in cells:
        X = mm[(ARM_MM_KEY[arm], li)]
        Y = mm[("vx", li)]
        y_ho = np.asarray(Y[ho], dtype=np.float64)
        pred_ib = identity_bias_predict(np.asarray(X[tr]), np.asarray(Y[tr]), np.asarray(X[ho]))
        r2_ib, cos_ib = F._recon_point(pred_ib, y_ho)
        cellb: dict = {
            "identity_bias": {"holdout_r2": float(r2_ib), "holdout_mean_cosine": float(cos_ib)},
            "knn": {},
        }
        ridge_npz = pred_dir / f"{arm}_L{li}_ridge.npz"
        preds = {"identity_bias": pred_ib}
        if ridge_npz.exists():
            preds["ridge"] = np.load(ridge_npz)["pred16"].astype(np.float64)
        for pname, pv in preds.items():
            cellb["knn"][pname] = {
                m: knn_retrieval(pv, y_ho, ks=(1, 5, 10), metric=m) for m in ("euclidean", "cosine")
            }
        baselines["cells"][f"{arm}_L{li}"] = cellb
        logger.info("[baseline] %s_L%d: identity+bias holdout R2=%.4f", arm, li, r2_ib)
    return baselines


def _compute_transfer(mm, ci, arms, split, tr, val, ho, corpus_by_ci, dev, args) -> dict:
    """lmsys_transfer (group-level OOD): ridge, layer 19, both arms — fit on
    LMSYS-only train rows, score WildChat holdout rows. Extracted verbatim from
    run_fits so --rebuild-summaries shares it (r4)."""
    transfer: dict = {"control": "lmsys_transfer", "layer": 19, "cells": {}}
    if split.get("transfer_descoped"):
        transfer["descoped"] = True
        logger.warning("[transfer] DESCOPED at manifest time (E_W < 5000)")
    elif corpus_by_ci:
        corp = np.asarray([corpus_by_ci.get(int(c), "?") for c in ci])
        tr_lm = tr[corp[tr] == "lmsys"]
        val_lm = val[corp[val] == "lmsys"]
        ho_wc = ho[corp[ho] == "wildchat"]
        ho_lm = ho[corp[ho] == "lmsys"]
        if len(ho_wc) == 0 or len(tr_lm) == 0:
            transfer["skipped"] = f"empty cells (tr_lm={len(tr_lm)}, ho_wc={len(ho_wc)})"
        else:
            for arm in arms:
                X = mm[(ARM_MM_KEY[arm], 19)]
                Y = mm[("vx", 19)]
                eval_idx = np.concatenate([ho_wc, ho_lm])
                pred, meta = PF.fit_ridge(
                    X, Y, tr_lm, val_lm, eval_idx, LAMBDAS, dev, args.ridge_block
                )
                r2_wc, _ = F._recon_point(
                    pred[: len(ho_wc)], np.asarray(Y[ho_wc], dtype=np.float64)
                )
                r2_lm, _ = F._recon_point(
                    pred[len(ho_wc) :], np.asarray(Y[ho_lm], dtype=np.float64)
                )
                transfer["cells"][arm] = {
                    "n_train_lmsys": int(len(tr_lm)),
                    "n_holdout_wildchat": int(len(ho_wc)),
                    "n_holdout_lmsys": int(len(ho_lm)),
                    "transfer_r2_wildchat_holdout": float(r2_wc),
                    "within_r2_lmsys_holdout": float(r2_lm),
                    "selected_lambda": meta["selected_lambda"],
                }
                logger.info("[transfer] %s: wc R2=%.4f lmsys R2=%.4f", arm, r2_wc, r2_lm)
    else:
        transfer["skipped"] = "no --manifest-dir (corpus provenance unavailable)"
    return transfer


def _upload_analysis_tensors(args, entries: list[tuple[str, Path, list[str] | None]]) -> None:
    """One verified upload_folder commit per entry → {hf_prefix}/analysis_tensors/{sub}.

    entries: (sub, local_dir, rel_files) — rel_files None means every file under
    local_dir (rglob), else the explicit relative-path subset. Fail-loud on any
    unverified commit (the per-cell exact-set contract, upload-policy.md).
    Uploads ride ``--upload-prefix`` when set (the bare round's
    issue1738_multiturn/bare_query; default = ``--hf-prefix``, plan §4.2)."""
    # UPLOAD_PREFIX_EXEMPT: default = this issue's own --hf-prefix (issue1738_multiturn); child reuse must pass --upload-prefix (plan v6 §4.2 dual-write)
    up = getattr(args, "upload_prefix", "") or args.hf_prefix
    for sub, local, files in entries:
        if files is None:
            files = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
        if not files:
            continue
        # UPLOAD_PREFIX_EXEMPT: dest defaults to this issue's own --hf-prefix (issue1738_multiturn); child reuse must pass --upload-prefix (plan v6 §4.2 dual-write)
        url = hub._upload_folder_filtered(
            local,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{up}/{ANALYSIS_TENSORS_SUBDIR}/{sub}",
            allow_patterns=files,
            expected_repo_paths=[f"{up}/{ANALYSIS_TENSORS_SUBDIR}/{sub}/{f}" for f in files],
        )
        if not url:
            raise RuntimeError(f"analysis-tensors upload ({sub}) returned no URL")


def _summary_upload_entry(args) -> tuple[str, Path, list[str] | None]:
    """r4 persist-by-default fix: every small JSON the fits/rebuild phases emit
    (fits summary, mapping_baselines, per-cell fit_meta records, the G2 fence
    report) rides the HF analysis_tensors upload IN ADDITION to git — the
    git-only destination lost multiturn_100k_fits.json + mapping_baselines.json
    when the DELETE-on-exit GCE instance was reaped before any VM harvest."""
    cand = [
        args.out_eval / "mapping_baselines.json",
        args.out_eval / "fits" / f"{FIT_POINT}_fits.json",
        args.out_eval / "fits" / "fence_report.json",
        *sorted((args.out_eval / "fits" / "cells").glob("*.json")),
        *sorted((args.out_eval / "fits" / "cells_rebuilt").glob("*.json")),
    ]
    files = sorted(str(p.relative_to(args.out_eval)) for p in cand if p.is_file())
    return ("summaries", args.out_eval, files)


# ── r4: --rebuild-summaries (summary recovery from the RETAINED HF tensors) ───────


REBUILD_PARTIAL_REASON = (
    "rebuilt from retained fp16 pred16 vs restreamed fp32 capture (#1738 r4 "
    "summary recovery); test-split predictions, wall clocks and fit_meta lived "
    "only in the lost git-destined JSONs"
)


def _git_head() -> str:
    """Reproducibility metadata for the rebuild provenance block: EPM_GIT_COMMIT
    when the launcher set it (the GCE lane), else the local git HEAD."""
    import subprocess

    env_sha = os.environ.get("EPM_GIT_COMMIT", "")
    if env_sha:
        return env_sha
    r = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def _stage_analysis_subdir(args, sub: str) -> Path:
    """Local dir holding {hf_prefix}/analysis_tensors/{sub}/*.

    Production: retried scoped staging via hub.stage_hub_prefix (#1402 —
    idempotent, existing targets skip; one resolved revision per call).
    Smoke: --local-analysis-dir/<sub> bypasses the Hub entirely."""
    if args.local_analysis_dir:
        d = Path(args.local_analysis_dir) / sub
        assert d.is_dir(), f"--local-analysis-dir missing subdir {d}"
        return d
    # UPLOAD_PREFIX_EXEMPT: default = this issue's own --hf-prefix (issue1738_multiturn); child reuse must pass --upload-prefix (plan v6 §4.2)
    up = getattr(args, "upload_prefix", "") or args.hf_prefix
    prefix = f"{up}/{ANALYSIS_TENSORS_SUBDIR}/{sub}"
    dest = Path(args.out_local) / "rebuild_stage"
    hub.stage_hub_prefix(C.HF_DATA_REPO, prefix, dest, repo_type="dataset")
    return dest / prefix


def run_rebuild(args) -> int:
    """r4 summary recovery: reconstruct multiturn_100k_fits.json +
    mapping_baselines.json from the RETAINED HF tensors after the phase-3 GCE
    boot disk (git-only summary destination) was deleted before any harvest.

    Holdout-side numbers recompute near-exactly: the fp32 capture restream is
    the same mm the fits saw, and pred16 is the retained prediction (fp16
    quantization ~1e-6 on holdout R² at n_ho≈10k — the smoke pins <1e-4).
    Genuinely unrecoverable fields (test-split R², per-cell walls, fit_meta)
    are emitted as null with partial flags — never fabricated. The
    lmsys_transfer control is RE-fit (ridge only) on CPU and tagged as such:
    a fresh deterministic estimate, not a reconstruction of the lost values."""
    layers = [int(x) for x in args.layers.split(",")]
    arms = list(BOTH_ARMS) if args.input_arm == "both" else [args.input_arm]
    dev = torch.device("cpu")
    args.mlp_lrs_list = [float(x) for x in args.mlp_lrs.split(",")]

    C.phase("rebuild-stage")
    pred_dir = _stage_analysis_subdir(args, "pred16")
    pc_dir = _stage_analysis_subdir(args, "percontext")
    yh_dir = _stage_analysis_subdir(args, "y_holdout")

    C.phase("rebuild-assemble")
    mm, ci, ameta = assemble_streams(args, layers)
    afp = ameta["fingerprint"]
    if "bare" in arms:
        bare_mm, _bare_meta = assemble_bare_streams(args, layers, ci, afp)
        mm.update(bare_mm)
    split = load_split(Path(args.split_file))
    if "bare" in arms:
        _assert_parent_split_shas(split, args.parent_fits_json)
    sets = split_positions(split, ci)
    shortfalls = _coverage_shortfalls(sets, split, args.min_split_coverage)
    if shortfalls:
        raise SystemExit(
            f"capture coverage below floor: {'; '.join(shortfalls)} — the rebuild "
            "restream does not match the phase-3 capture set"
        )
    tr, val, te, ho = sets["train"], sets["val"], sets["test"], sets["holdout"]
    # retained-artifact integrity (the r2 fingerprint-keying, read side): the
    # y_holdout npz must pair EXACTLY with the restream — same chunk universe
    # (fingerprint), same holdout rows (ci), same fp16 cast of the same fp32.
    for li in layers:
        with np.load(yh_dir / f"L{li}.npz") as z:
            assert z["fingerprint"].item() == afp, (li, z["fingerprint"].item(), afp)
            assert np.array_equal(z["ci"], ci[ho]), f"y_holdout ci misalign (L{li})"
            y16 = np.asarray(mm[("vx", li)][ho], dtype=np.float16)
            assert np.array_equal(z["y16"], y16), f"y_holdout fp16 content mismatch (L{li})"
    corpus_by_ci = {}
    if args.manifest_dir:
        pool, _m = GG.N1M.read_manifest_pool(Path(args.manifest_dir))
        corpus_by_ci = {int(r["i"]): r["corpus"] for r in pool}

    layer_order = ([19] if 19 in layers else []) + [x for x in layers if x != 19]
    arm_order = [a for a in ARM_ORDER if a in arms]
    cells = [(a, li) for a in arm_order for li in layer_order]
    preds_list = [p for p in PREDICTORS if p in args.predictors.split(",")]

    C.phase("rebuild-cells")
    rcells_dir = args.out_eval / "fits" / "cells_rebuilt"
    rcells_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "fit_point": FIT_POINT,
        "layers": layers,
        "arms": arms,
        "n_rows_captured": ameta["n_rows"],
        "split_counts": {k: int(len(v)) for k, v in sets.items()},
        "split_shas": {k: split["sets"][k]["sha256"] for k in split["sets"]},
        "lambdas": [float(x) for x in LAMBDAS],
        "mlp_lr_grid": args.mlp_lrs_list,
        "krr": {"m_centers": int(args.krr_nystrom_centers), "solver": args.krr_solver},
        "n_boot": int(args.n_boot),
        "boot_seed": BOOT_SEED,
        "cells": {},
        "git_commit": os.environ.get("EPM_GIT_COMMIT", ""),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rebuilt": {
            "partial": True,
            "reason": REBUILD_PARTIAL_REASON,
            "source": f"{C.HF_DATA_REPO}/{args.hf_prefix}/{ANALYSIS_TENSORS_SUBDIR}/"
            "{pred16,y_holdout,percontext} + capture restream",
            "unrecoverable_fields": [
                "cells.*.test_r2",
                "cells.*.test_mean_cosine",
                "cells.*.wall_s",
                "cells.*.fit_meta (per-cell JSONs)",
                "cells.*.seed43.lr",
            ],
            "recomputed": [
                "holdout_bootstrap_ci (same seed/draws, from fp16 pred16)",
                "lmsys_transfer (ridge REFIT on cpu; original fit on cuda)",
            ],
            "assembly_fingerprint": afp,
            "rebuild_git_commit": _git_head(),
        },
    }
    t0 = time.time()
    n_units = len(cells) * len(preds_list)
    k = 0
    for arm, li in cells:
        Y = mm[("vx", li)]
        y_ho = np.asarray(Y[ho], dtype=np.float64)
        for name in preds_list:
            k += 1
            key = f"{arm}_L{li}_{name}"
            rcj = rcells_dir / f"{key}.json"
            if rcj.exists() and not args.no_resume:
                doc = json.loads(rcj.read_text())
                if doc.get("assembly_fingerprint") == afp:
                    summary["cells"][key] = doc["metrics"]
                    logger.info("[rebuild] unit %d/%d %s: resume-skip", k, n_units, key)
                    continue
            pz = pred_dir / f"{key}.npz"
            if not pz.exists():
                raise SystemExit(f"retained pred16 missing for {key}: {pz}")
            with np.load(pz) as z:
                assert z["fingerprint"].item() == afp, (key, "pred16 fingerprint")
                assert np.array_equal(z["ci"], ci[ho]), f"pred16 ci misalign ({key})"
                pred_ho = z["pred16"].astype(np.float64)
            r2_ho, cos_ho = F._recon_point(pred_ho, y_ho)
            ci_boot = _boot_recon_ci_batched(pred_ho, y_ho, args.n_boot, BOOT_SEED)
            metrics: dict = {
                "test_r2": None,
                "test_mean_cosine": None,
                "holdout_r2": float(r2_ho),
                "holdout_mean_cosine": float(cos_ho),
                "holdout_bootstrap_ci": ci_boot,
                "n_test": int(len(te)),
                "n_holdout": int(len(ho)),
                "wall_s": None,
                "partial": True,
                "partial_reason": REBUILD_PARTIAL_REASON,
            }
            z43p = pred_dir / f"{key}_seed43.npz"
            if z43p.exists():
                with np.load(z43p) as z43:
                    assert z43["fingerprint"].item() == afp, (key, "seed43 fingerprint")
                    pred43 = z43["pred16"].astype(np.float64)
                r2_43, cos_43 = F._recon_point(pred43, y_ho)
                # seed-pair nerr pearson recomputes EXACTLY: both fp32 nerr
                # arrays were retained verbatim in the percontext npzs.
                nerr = np.load(pc_dir / f"{key}.npz")["nerr"]
                nerr43 = np.load(pc_dir / f"{key}_seed43.npz")["nerr"]
                metrics["seed43"] = {
                    "seed": 43,
                    "lr": None,
                    "holdout_r2": float(r2_43),
                    "holdout_mean_cosine": float(cos_43),
                    "seed_pair_nerr_pearson": float(np.corrcoef(nerr, nerr43)[0, 1])
                    if len(nerr) > 2
                    else float("nan"),
                    "partial": True,
                    "partial_reason": "selected lr lived in the lost fit_meta",
                }
            GG.N1M._atomic_write_json(
                rcj,
                {
                    "arm": arm,
                    "layer": li,
                    "fitter": name,
                    "metrics": metrics,
                    "fit_meta": None,
                    "rebuilt": True,
                    "assembly_fingerprint": afp,
                },
            )
            summary["cells"][key] = metrics
            logger.info(
                "[rebuild] unit %d/%d %s holdout R2=%.4f elapsed=%.0fs",
                k,
                n_units,
                key,
                r2_ho,
                time.time() - t0,
            )

    C.phase("rebuild-baselines")
    baselines = _compute_baselines(mm, tr, ho, cells, pred_dir)
    GG.N1M._atomic_write_json(args.out_eval / "mapping_baselines.json", baselines)

    C.phase("rebuild-transfer")
    transfer = _compute_transfer(mm, ci, arms, split, tr, val, ho, corpus_by_ci, dev, args)
    if transfer.get("cells"):
        transfer["recomputed"] = {
            "device": "cpu",
            "note": "ridge REFIT during the r4 rebuild — a fresh deterministic "
            "estimate; the originally-fitted (cuda) values were lost with the summary",
        }
    summary["lmsys_transfer"] = transfer
    GG.N1M._atomic_write_json(args.out_eval / "fits" / f"{FIT_POINT}_fits.json", summary)

    if not args.no_upload:
        C.phase("rebuild-upload")
        _upload_analysis_tensors(args, [_summary_upload_entry(args)])
    C.phase("done")
    return 0


# ── smoke: tiny synthetic capture store through the PRODUCTION entrypoint ─────────


def _write_smoke_store(
    root: Path, *, n_rows=140, layers=(14, 19, 26), seed=0
) -> tuple[Path, Path, Path]:
    """Synthetic capture chunks in the PRODUCTION chunk schema + a matching
    manifest/split doc + a BARE-arm store (plan §4.2 smoke: bq chunks in
    REVERSED ci order — exercises the ci-keyed reorder — plus 2 extra ci absent
    from the parent capture, the recorded-and-dropped path).
    Y = linear(X_cx) + noise so ridge finds real signal."""
    rng = np.random.default_rng(seed)
    cap = root / "capture"
    man = root / "manifest"
    cap.mkdir(parents=True, exist_ok=True)
    man.mkdir(parents=True, exist_ok=True)
    W = rng.standard_normal((H_DIM, H_DIM)).astype(np.float32) * 0.01
    rows_per_chunk = (n_rows + 2) // 3
    pool_rows = []
    cx_all: list[np.ndarray] = []
    ci0 = 0
    for k in range(3):
        n = min(rows_per_chunk, n_rows - ci0)
        cx = rng.standard_normal((n, len(layers), H_DIM)).astype(np.float32)
        cx_all.append(cx)
        px = cx + 0.5 * rng.standard_normal((n, len(layers), H_DIM)).astype(np.float32)
        vx = np.einsum("nlh,hd->nld", cx, W).astype(np.float32)
        vx += 0.05 * rng.standard_normal(vx.shape).astype(np.float32)
        cis = list(range(ci0, ci0 + n))
        torch.save(
            {
                "px_last": torch.from_numpy(px),
                "cx_last": torch.from_numpy(cx),
                "v_x": torch.from_numpy(vx),
                "ci": cis,
                "prompts": ["[]"] * n,
                "response": ["stub"] * n,
                "depth": [2 + (c % 4) for c in cis],
                "corpus": ["lmsys" if c % 3 else "wildchat" for c in cis],
                "layers": list(layers),
                "shard_index": 0,
                "chunk": k,
            },
            cap / f"shard00_chunk{k:04d}.pt",
        )
        for c in cis:
            pool_rows.append(
                {
                    "i": c,
                    "messages": [{"role": "user", "content": f"q{c}"}],
                    "depth": 2 + (c % 4),
                    "corpus": "lmsys" if c % 3 else "wildchat",
                    "source_hash": f"s{c}",
                    "stream_pos": c,
                    "n_chars": 10,
                    "split": "train",
                }
            )
        ci0 += n
    order = rng.permutation(n_rows).tolist()
    sets = {
        "val": sorted(order[:20]),
        "test": sorted(order[20:40]),
        "holdout": sorted(order[40:80]),
        "train": sorted(order[80:]),
    }
    doc = {"seed": 1738, "n_manifest": n_rows, "sets": {}, "transfer_descoped": False}
    for name, ids in sets.items():
        doc["sets"][name] = {"ci": ids, "n": len(ids), "sha256": GG._sha_int_list(ids)}
        for c in ids:
            pool_rows[c]["split"] = name
    meta = {"n_new": n_rows, "capture_layers": list(layers)}
    GG.N1M._write_manifest_parts(man, pool_rows, meta)
    GG.N1M._atomic_write_json(man / "split_1738.json", doc)
    # bare-arm store: bq = 0.6*cx + noise (correlated but distinct), rows in
    # REVERSED parent-ci order + 2 EXTRA ci the parent never captured.
    bare = root / "bare_capture"
    bare.mkdir(parents=True, exist_ok=True)
    cx_full = np.concatenate(cx_all, axis=0)  # (n_rows, L, H), parent ci order
    bare_ci = list(reversed(range(n_rows))) + [n_rows, n_rows + 1]
    bq_full = np.concatenate(
        [
            0.6 * cx_full[list(reversed(range(n_rows)))],
            rng.standard_normal((2, len(layers), H_DIM)).astype(np.float32),
        ],
        axis=0,
    ) + 0.1 * rng.standard_normal((n_rows + 2, len(layers), H_DIM)).astype(np.float32)
    per = (len(bare_ci) + 1) // 2
    for k in range(2):
        sl = slice(k * per, min((k + 1) * per, len(bare_ci)))
        torch.save(
            {
                "bq_last": torch.from_numpy(bq_full[sl]),
                "ci": bare_ci[sl],
                "bare_render": ["<|im_start|>system\n..."] * (sl.stop - sl.start),
                "layers": list(layers),
                "shard_index": 0,
                "chunk": k,
            },
            bare / f"shard00_chunk{k:04d}.pt",
        )
    return cap, man, bare


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1738 multi-turn prefix/context fits.")
    ap.add_argument("--input-arm", choices=["prefix", "context", "both", "bare"], default="both")
    ap.add_argument("--layers", default=",".join(str(x) for x in LAYERS_DEFAULT))
    # UPLOAD_PREFIX_EXEMPT: issue 1738's own analysis-tensors prefix; a child issue reusing this driver must pass --hf-prefix explicitly (artifact-reuse check (i))
    ap.add_argument("--hf-prefix", default=GG.HF_PREFIX)
    # ── bare-arm inputs (follow-up `bare-query`, plan §4.2) ───────────────────────
    # UPLOAD_PREFIX_EXEMPT: issue 1738's own bare-arm store prefix (plan §4.2); read-side default
    ap.add_argument("--bare-hf-prefix", default=f"{GG.HF_PREFIX}/bare_query")
    ap.add_argument("--local-bare-dir", default="", help="read bare chunks locally (smoke)")
    ap.add_argument(
        "--parent-fits-json",
        default=str(DEFAULT_OUT_EVAL / "fits" / f"{FIT_POINT}_fits.json"),
        help="bare arm: parent fits JSON whose recorded split_shas the new run's "
        "split MUST match (plan §4.2 cross-assert)",
    )
    ap.add_argument(
        "--upload-prefix",
        default="",
        help="HF prefix for THIS run's analysis-tensor/summary uploads (default = "
        "--hf-prefix); the bare round passes issue1738_multiturn/bare_query",
    )
    ap.add_argument("--local-capture-dir", default="", help="read chunks locally (smoke/pod)")
    ap.add_argument("--manifest-dir", default="", help="local manifest dir (corpus provenance)")
    ap.add_argument("--manifest-from-hf", action="store_true")
    ap.add_argument("--split-file", default="", help="split_1738.json (default: manifest dir)")
    ap.add_argument("--mm-dir", default=str(DEFAULT_OUT_LOCAL / "mm"))
    ap.add_argument("--out-eval", type=Path, default=DEFAULT_OUT_EVAL)
    ap.add_argument("--out-local", type=Path, default=DEFAULT_OUT_LOCAL)
    ap.add_argument("--predictors", default=",".join(PREDICTORS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--mlp-lrs", default=",".join(str(x) for x in MLP_LRS))
    ap.add_argument("--mlp-max-epochs", type=int, default=F.MLP_MAX_EPOCHS)
    ap.add_argument("--mlp-batch", type=int, default=PF.MLP_BATCH)
    ap.add_argument("--ridge-block", type=int, default=PF.RIDGE_BLOCK)
    # plan §11 NOTE: m=16384 EXPLICIT here — the parent script default is 8192.
    ap.add_argument("--krr-nystrom-centers", type=int, default=KRR_CENTERS_DEFAULT)
    ap.add_argument("--krr-solver", choices=["eigh", "cholesky"], default="cholesky")
    ap.add_argument("--krr-gamma-mult", dest="krr_gamma_mult_s", default="1.0")
    ap.add_argument("--krr-lambdas", dest="krr_lambdas_s", default="0.1,10")
    ap.add_argument("--fence-mult", type=float, default=2.0, help="G2: halt past mult×first-cell")
    ap.add_argument(
        "--min-split-coverage",
        type=float,
        default=0.95,
        help="fail-loud floor on per-set captured/intended (0 disables; Minor-1 fix)",
    )
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--allow-underdetermined", action="store_true", help="smoke shape: n_train<d")
    ap.add_argument(
        "--rebuild-summaries",
        action="store_true",
        help="r4 summary recovery: reconstruct the two summary JSONs from the "
        "retained HF tensors (no fits; CPU)",
    )
    ap.add_argument(
        "--local-analysis-dir",
        default="",
        help="rebuild: local dir holding pred16/, y_holdout/, percontext/ (smoke bypass)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic store, production path")
    args = ap.parse_args()

    if args.smoke:
        root = Path(args.mm_dir).parent / "_smoke_fits"
        if root.exists():
            import shutil

            shutil.rmtree(root)
        cap, man, bare_cap = _write_smoke_store(root)
        args = argparse.Namespace(
            **{
                **vars(args),
                "local_capture_dir": str(cap),
                "manifest_dir": str(man),
                "split_file": str(man / "split_1738.json"),
                "mm_dir": str(root / "mm"),
                "out_eval": root / "eval",
                "out_local": root / "local",
                "device": "cpu",
                "mlp_max_epochs": 2,
                "krr_nystrom_centers": 32,
                "n_boot": 50,
                "mlp_lrs": "0.0003",
                "layers": "19",
                "no_upload": True,
                "allow_underdetermined": True,  # deliberate smoke shape: n_train 60 < d 3584
            }
        )
        rc = run_fits(args)
        summ = json.loads((args.out_eval / "fits" / f"{FIT_POINT}_fits.json").read_text())
        assert len(summ["cells"]) == 2 * 1 * 5, sorted(summ["cells"])
        # second-seed MLP holdout read (plan §10 Seeds row): npz pair + metrics.
        for arm_ in ("prefix", "context"):
            s43 = summ["cells"][f"{arm_}_L19_mlp_w8192"]["seed43"]
            assert s43["seed"] == 43 and np.isfinite(s43["holdout_r2"]), s43
            assert (args.out_local / "pred16" / f"{arm_}_L19_mlp_w8192_seed43.npz").exists()
        # resume-regime probes (fits-resume-ci-alignment-unkeyed fix): a stale
        # assembly fingerprint on a cell JSON or the y_holdout npz forces a
        # refit/regeneration on the resume run — never a silent resume.
        cj_probe = args.out_eval / "fits" / "cells" / "context_L19_ridge.json"
        doc_probe = json.loads(cj_probe.read_text())
        real_fp = doc_probe["assembly_fingerprint"]
        assert real_fp, "cell JSON missing assembly_fingerprint"
        doc_probe["assembly_fingerprint"] = "stale"
        cj_probe.write_text(json.dumps(doc_probe))
        yhp_probe = args.out_local / "y_holdout" / "L19.npz"
        with np.load(yhp_probe) as zp:
            assert zp["fingerprint"].item() == real_fp
            y16_probe, ci_probe = zp["y16"], zp["ci"]
        np.savez(yhp_probe, y16=y16_probe, ci=ci_probe, fingerprint=np.array("stale"))
        assert run_fits(args) == 0  # resume run: stale artifacts must regenerate
        assert json.loads(cj_probe.read_text())["assembly_fingerprint"] == real_fp
        with np.load(yhp_probe) as zp:
            assert zp["fingerprint"].item() == real_fp
        logger.info("[smoke] stale-fingerprint resume probes OK (cell JSON + y_holdout)")
        # coverage-floor + lr-selection pure predicates, both sides.
        cov_doc = {"sets": {"train": {"n": 100}}}
        assert _coverage_shortfalls({"train": np.arange(90)}, cov_doc, 0.95)
        assert not _coverage_shortfalls({"train": np.arange(96)}, cov_doc, 0.95)
        assert _mlp_lr_better(0.5, float("nan"))  # NaN-first divergence loses
        assert not _mlp_lr_better(float("nan"), 0.5)
        assert _mlp_lr_better(0.6, 0.5) and not _mlp_lr_better(0.4, 0.5)
        assert _mlp_lr_better(float("nan"), None)  # first lr always seeds
        bl = json.loads((args.out_eval / "mapping_baselines.json").read_text())
        assert "context_L19" in bl["cells"] and "prefix_L19" in bl["cells"]
        for cell in bl["cells"].values():
            assert "identity_bias" in cell and "ridge" in cell["knn"]
        for f in (args.out_eval / "percontext").glob("*.npz"):
            z = np.load(f)
            assert z["nerr"].shape == z["ci"].shape
        assert "cells" in summ["lmsys_transfer"] and summ["lmsys_transfer"]["cells"], summ[
            "lmsys_transfer"
        ]
        # degenerate-input gate probes (data-dependent-gates duty): the G2 fence
        # predicate fires past the budget and stays quiet under it; the
        # n_train<d refusal fires without --allow-underdetermined.
        assert _fence_should_halt(101.0, 10.0, 5, 2.0) and not _fence_should_halt(
            99.0, 10.0, 5, 2.0
        )
        ud = argparse.Namespace(**{**vars(args), "allow_underdetermined": False})
        try:
            run_fits(ud)
            raise AssertionError("n_train<d gate did not refuse")
        except SystemExit as e:
            assert "n_train" in str(e.code) or (isinstance(e.code, str) and "d=" in e.code), e.code
        # batched-rewrite equivalence gate (vectorize rule item 6): the batched
        # bootstrap CI must reproduce F._bootstrap_recon_ci draw-for-draw.
        rng = np.random.default_rng(7)
        tp = rng.standard_normal((30, 8))
        tt = tp + 0.3 * rng.standard_normal((30, 8))
        a = _boot_recon_ci_batched(tp, tt, 200, 11, chunk=64)
        b = F._bootstrap_recon_ci(tp, tt, 200, 11)
        for k in ("r2", "mean_cosine"):
            for f_ in ("point", "lo", "hi"):
                assert abs(a[k][f_] - b[k][f_]) < 1e-9, (k, f_, a[k], b[k])
        logger.info("[smoke] batched bootstrap CI == serial reference (200 draws)")
        # r4 rebuild-equivalence leg: --rebuild-summaries against this smoke
        # run's own retained artifacts must reproduce its holdout-side numbers
        # (fp16 pred16 quantization is the only delta) and null the rest.
        stage = root / "rb_stage"
        stage.mkdir()
        (stage / "pred16").symlink_to((args.out_local / "pred16").resolve())
        (stage / "y_holdout").symlink_to((args.out_local / "y_holdout").resolve())
        (stage / "percontext").symlink_to((args.out_eval / "percontext").resolve())
        rb = argparse.Namespace(
            **{
                **vars(args),
                "rebuild_summaries": True,
                "local_analysis_dir": str(stage),
                "out_eval": root / "eval_rebuild",
            }
        )
        assert run_rebuild(rb) == 0
        rsum = json.loads((rb.out_eval / "fits" / f"{FIT_POINT}_fits.json").read_text())
        assert rsum["rebuilt"]["partial"] is True and rsum["rebuilt"]["assembly_fingerprint"]
        assert sorted(rsum["cells"]) == sorted(summ["cells"])
        for key, m in summ["cells"].items():
            rm = rsum["cells"][key]
            assert rm["test_r2"] is None and rm["wall_s"] is None and rm["partial"] is True
            assert abs(rm["holdout_r2"] - m["holdout_r2"]) < 1e-4, (key, rm, m)
            assert abs(rm["holdout_mean_cosine"] - m["holdout_mean_cosine"]) < 1e-4, key
            for k_ in ("r2", "mean_cosine"):
                for f_ in ("lo", "hi"):
                    da = rm["holdout_bootstrap_ci"][k_][f_] - m["holdout_bootstrap_ci"][k_][f_]
                    assert abs(da) < 1e-3, (key, k_, f_, da)
            if "seed43" in m:
                assert abs(rm["seed43"]["holdout_r2"] - m["seed43"]["holdout_r2"]) < 1e-4, key
                dp = rm["seed43"]["seed_pair_nerr_pearson"] - m["seed43"]["seed_pair_nerr_pearson"]
                assert abs(dp) < 1e-9, (key, dp)  # exact: retained fp32 nerr both sides
                assert rm["seed43"]["lr"] is None
        rbl = json.loads((rb.out_eval / "mapping_baselines.json").read_text())
        bl_now = json.loads((args.out_eval / "mapping_baselines.json").read_text())
        assert rbl == bl_now, "rebuilt mapping_baselines != original (identical inputs)"
        rtr = rsum["lmsys_transfer"]
        assert rtr["cells"] and rtr["recomputed"]["device"] == "cpu", rtr
        for arm_, tcell in summ["lmsys_transfer"]["cells"].items():
            for f_ in ("transfer_r2_wildchat_holdout", "within_r2_lmsys_holdout"):
                # smoke fits ran on cpu too -> the refit is draw-identical
                assert abs(rtr["cells"][arm_][f_] - tcell[f_]) < 1e-9, (arm_, f_)
        logger.info("[smoke] rebuild-summaries equivalence OK (%d cells)", len(rsum["cells"]))
        # ── bare-arm leg (plan §4.2): single-arm fits through the SAME production
        # entrypoint — ci-keyed reorder + 1:1 coverage assert + parent split_shas
        # cross-assert + extras recorded/dropped + no seed-43 repeat.
        beval = root / "eval_bare"
        blocal = root / "local_bare"
        bargs = argparse.Namespace(
            **{
                **vars(args),
                "input_arm": "bare",
                "local_bare_dir": str(bare_cap),
                "parent_fits_json": str(args.out_eval / "fits" / f"{FIT_POINT}_fits.json"),
                "out_eval": beval,
                "out_local": blocal,
            }
        )
        assert run_fits(bargs) == 0
        bsum = json.loads((beval / "fits" / f"{FIT_POINT}_fits.json").read_text())
        assert bsum["arms"] == ["bare"] and len(bsum["cells"]) == 5, sorted(bsum["cells"])
        assert "seed43" not in bsum["cells"]["bare_L19_mlp_w8192"], (
            "seed-43 repeat must NOT run on the bare arm (plan §10 Seeds row)"
        )
        ba = bsum["bare_assembly"]
        assert ba["n_extra_dropped"] == 2 and ba["n_parent_rows"] == bsum["n_rows_captured"], ba
        # reorder correctness: bare percontext rows carry the SAME holdout ci
        # order as the parent arm (the fancy-index alignment worked).
        bz = np.load(beval / "percontext" / "bare_L19_ridge.npz")
        pz2 = np.load(args.out_eval / "percontext" / "context_L19_ridge.npz")
        assert np.array_equal(bz["ci"], pz2["ci"]), "bare percontext ci != parent holdout ci"
        bbl = json.loads((beval / "mapping_baselines.json").read_text())
        assert "bare_L19" in bbl["cells"] and "ridge" in bbl["cells"]["bare_L19"]["knn"]
        assert bsum["lmsys_transfer"]["cells"].get("bare"), bsum["lmsys_transfer"]
        # degenerate probe (i): a bare store MISSING one parent ci fails the
        # 1:1 coverage assert LOUD (never a silent drop).
        import shutil

        bare_missing = root / "bare_missing"
        shutil.copytree(bare_cap, bare_missing)
        mc = sorted(bare_missing.glob("*.pt"))[0]
        mb = torch.load(mc, weights_only=False)
        torch.save(
            {
                **mb,
                "bq_last": mb["bq_last"][1:],
                "ci": mb["ci"][1:],
                "bare_render": mb["bare_render"][1:],
            },
            mc,
        )
        cov_fired = False
        try:
            run_fits(
                argparse.Namespace(
                    **{
                        **vars(bargs),
                        "local_bare_dir": str(bare_missing),
                        "mm_dir": str(root / "mm_missing"),
                        "out_eval": root / "eval_bare_missing",
                        "out_local": root / "local_bare_missing",
                    }
                )
            )
        except AssertionError as e:
            assert "1:1 coverage" in str(e), e
            cov_fired = True
        assert cov_fired, "bare coverage assert did not fire on a missing parent ci"
        # degenerate probe (ii): a parent-fits split_shas mismatch fails the
        # cross-assert LOUD before any fitting.
        bad_pj = root / "bad_parent_fits.json"
        bad_pj.write_text(json.dumps({"split_shas": dict.fromkeys(bsum["split_shas"], "dead")}))
        sha_fired = False
        try:
            run_fits(
                argparse.Namespace(
                    **{
                        **vars(bargs),
                        "parent_fits_json": str(bad_pj),
                        "out_eval": root / "eval_bare_sha",
                        "out_local": root / "local_bare_sha",
                    }
                )
            )
        except AssertionError as e:
            assert "split_shas" in str(e), e
            sha_fired = True
        assert sha_fired, "parent split_shas cross-assert did not fire"
        logger.info("[smoke] bare-arm leg OK: 5 cells + reorder + coverage/sha probes")
        logger.info("[smoke] fits OK: %d cells + baselines + transfer", len(summ["cells"]))
    else:
        if args.manifest_from_hf and not args.manifest_dir:
            mdir = GG.N1M._download_manifest(
                args.hf_prefix, Path(args.mm_dir).parent / GG.MANIFEST_SUBDIR
            )
            args.manifest_dir = str(mdir)
        if not args.split_file:
            if not args.manifest_dir:
                raise SystemExit("--split-file or --manifest-dir/--manifest-from-hf required")
            args.split_file = str(Path(args.manifest_dir) / "split_1738.json")
        rc = run_rebuild(args) if args.rebuild_summaries else run_fits(args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
