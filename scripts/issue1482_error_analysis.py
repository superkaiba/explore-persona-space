"""Issue #1482 — error analysis of the n1M context->answer map: pod-side driver P0-P4.

Phases (plan v3 §4; smoke IS this driver with tiny args — PASS_UNIFIED):
  p0  stage captures + build X/Y (parent code path) + split reconstruction w/ pinned
      sha asserts + fresh-holdout / SAE-subsample carve -> split_1482.json.
  p1  dense-map refits (refit_full / refit_holdout / refit_lmsys_transfer + seed-43
      MLP) reusing the PARENT fitter functions UNCHANGED (issue779_ffc_n1m_fits);
      per-context e2/cos/nerr; Gate A reconciliation vs committed n1m_fits.json.
  p2  teacher-forced per-token capture (token-id concat prompt++response, #1092) +
      BatchTopK SAE encode + pooling (mean/MAX/frac); P2-pilot FIRST (tokens/s,
      FVE/L0 k64+k128 at L15/19/23, prefix-end constancy, G2 identity gate).
  p3  SAE->SAE + dense->SAE fits (both arms incl. prefix-null; shared-Gram ridge +
      MLP w8192), per-feature held-out R2/Spearman + split-half stability,
      encode-the-prediction, PCA per-direction read, interp digests.
  p4  pooled-store upload to HF (detached-concurrent with p3 — #825 rule).

Pod-side contract: sentinels under /workspace/logs/issue-1482-*.json ONLY (never
task.py); [phase=...] log lines; [phase=done] terminal. LMSYS/WildChat text is
handled DIGEST-ONLY (never printed/logged).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1482")

TASK_ID = 1482
LAYER = 19
PILOT_LAYERS = (15, 19, 23)
HF_PREFIX_DEFAULT = "issue1482_error_analysis"
CAPTURE_PREFIX = f"{N1G.HF_PREFIX}/final_token_capture"
RAW_PREFIX = f"{N1G.HF_PREFIX}/raw_completions"
COMMITTED_N1M = (
    PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison-n1m" / "n1m_fits.json"
)
SPLIT_SEED_1482 = 1482
MLP_SEED_B = 43
# Gate A tolerances (plan §7: ridge deterministic + GPU-noise headroom; MLP = committed
# bootstrap-CI half-width ±0.010 per n1m_fits.json CI [0.7435, 0.7642]).
GATE_A_TOL = {"ridge": 0.002, "mlp_w8192": 0.01, "mlp_w32768": 0.01}
GATE_A_PERIPHERAL_TOL = {"residual_skip": 0.01, "krr_nystrom": 0.01}
# Gate B thresholds (plan §7; calibration: the suite's own eval_results.json —
# FVE 0.806 @ k=64 / 0.842 @ k=128 on chat+pile).
GATE_B_PASS = 0.70
GATE_B_HALT = 0.55
G2_COS_MIN = 0.995  # #779 two-bar calibration, flat bar on same-layer identity
PREFIX_CONSTANCY_COS_MIN = 0.9999  # bf16/batch jitter margin on an identical-token prefix
RC_GATE_A = 21
RC_GATE_B = 22
# ── pdshrink (per-direction ridge-shrinkage control, plan v12) ──
RC_G1_PDSHRINK = 23  # designed-halt rc (gates.json written FIRST; kresample RC_G1 convention)
PDSHRINK_SUBDIR = "perdirection-shrinkage-control"
PDSHRINK_MLP_NPZ = "refit_holdout__mlp_w8192__seed0.npz"
PDSHRINK_BANDS: dict[str, tuple[int, int]] = {  # parent rank bands, 0-indexed slices
    "1-16": (0, 16),
    "17-64": (16, 64),
    "65-128": (64, 128),
    "129-256": (128, 256),
}
COMMITTED_SPLIT_1482 = PROJECT_ROOT / "eval_results" / "issue_1482" / "split_1482.json"
COMMITTED_PERDIR_PCA = PROJECT_ROOT / "eval_results" / "issue_1482" / "perdirection_pca.json"
COMMITTED_RIDGE_JSON = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "percontext" / "refit_holdout__ridge__seed0.json"
)
# G1 leg (i) split-sha anchors, pinned as LITERALS from the git-committed
# eval_results/issue_1482/split_1482.json @ origin/main. Production --out-eval defaults to
# the SAME path p0 regenerates split_1482.json into (fresh pod: resume predicate false ->
# rebuild + overwrite), so a disk read of the "committed" side would compare the regenerated
# doc against itself and pass unconditionally (round-1 code review, concern
# g1-split-sha-leg-self-compare). A legitimate future split re-pin must update these
# constants deliberately: tests/test_issue1482_pdshrink.py pins them to the committed file
# so silent drift fails loud in CI.
PDSHRINK_COMMITTED_SPLIT_SHAS: dict[str, str] = {
    "train_full_sha256": "aba6f837c4f478e135429e56227de9380133788df6bfd08cde4d8ec881a74481",
    "holdout_sha256": "7957d689748eca218055f213082c1df444603ec2f1faa3f04b4004cee6f58622",
    # SHA_PIN_DOMAIN: INDEX
    "pinned_val_sha256": "2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613",
    # SHA_PIN_DOMAIN: INDEX
    "pinned_test_sha256": "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d",
}

PREDICTORS_ALL = list(N1M.PREDICTORS)  # ridge, mlp_w8192, mlp_w32768, residual_skip, krr_nystrom


# ── small utils ──────────────────────────────────────────────────────────────────


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    obj = dict(obj)
    obj.setdefault("metadata", C.reproducibility_metadata())
    C.write_json_atomic(path, obj)


def _phase_sentinel(name: str, note: str, extra: dict | None = None) -> None:
    """Non-blocking phase sentinel (blocks_pipeline: false per plan)."""
    payload = {"blocks_pipeline": False}
    if extra:
        payload.update(extra)
    try:
        C.write_sentinel(f"phase-{name}", note, task_id=TASK_ID, extra=payload)
    except OSError as e:  # sentinel write must never kill the run on the VM smoke
        logger.warning("[sentinel] phase-%s write failed: %s", name, e)


def _physical_gpu_ids() -> list[int]:
    """Enumerate GPUs via nvidia-smi subprocess (never torch.cuda in the dispatcher)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        )
        return [int(x) for x in out.split() if x.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return []


def _headroom(root: Path, need_gb: float, phase: str) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    root.mkdir(parents=True, exist_ok=True)
    assert_out_root_headroom(root, need_gb, phase=phase)


def _log_tail(path: Path, n: int = 120) -> str:
    try:
        lines = path.read_text(errors="replace").split("\n")
        return "\n".join(lines[-n:])
    except OSError:
        return "<log unreadable>"


def _run_children(specs: list[dict], args, phase: str, on_done=None) -> None:
    """Work-conserving subprocess fan-out over the realized GPU width.

    Each spec: {"fit_id"/"worker": ..., "cmd": [...]} — cmd is the child argv
    AFTER the script path. CVD is pinned in the LAUNCHER env per slot (gotchas
    rule); on CPU (no GPUs) children run one-at-a-time... width = max(1, n_gpus).
    Child failure: echo the inner log tail into the main log (#1333) + raise.
    """
    gpus = _physical_gpu_ids() if args.device != "cpu" else []
    slots: list[int | None] = list(gpus) if gpus else [None]
    if args.n_gpus > 0:
        # GPU: truncate to the first n physical slots (never widen — co-location OOMs);
        # CPU: n parallel un-pinned worker slots (width>1 smoke of the fan-out itself).
        slots = slots[: args.n_gpus] if gpus else [None] * args.n_gpus
    logger.info("[%s] fan-out over %d slot(s) (gpus=%s)", phase, len(slots), gpus or "cpu")
    queue = list(specs)
    running: dict[int, tuple[subprocess.Popen, dict, Path]] = {}
    log_dir = args.scratch / "child_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while queue or running:
        for slot in [s for s in range(len(slots)) if s not in running]:
            if not queue:
                break
            spec = queue.pop(0)
            tag = spec["tag"]
            log = log_dir / f"{phase}_{tag}.log"
            env = {**os.environ}
            # Reduce CUDA allocator fragmentation in long fit children (attempt-6
            # OOM showed 16.4 GiB reserved-but-unallocated); torch-documented knob.
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if slots[slot] is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(slots[slot])
            cmd = [sys.executable, str(Path(__file__).resolve()), *spec["cmd"]]
            if slots[slot] is not None:
                cmd += ["--gpu-id", str(slots[slot])]  # informational; CVD pins the device
            logger.info("[%s] launch %s on slot %s -> %s", phase, tag, slots[slot], log.name)
            log_f = open(log, "w")  # noqa: SIM115 — fd handed to the child process
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running[slot] = (proc, spec, log)
        done_slots = []
        for slot, (proc, spec, log) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            done_slots.append(slot)
            if rc != 0:
                logger.error(
                    "[%s] child %s FAILED rc=%d; log tail:\n%s",
                    phase,
                    spec["tag"],
                    rc,
                    _log_tail(log),
                )
                for _, (p2, _, _) in running.items():
                    if p2.poll() is None:
                        p2.terminate()
                raise RuntimeError(f"{phase} child {spec['tag']} failed rc={rc} (log: {log})")
            logger.info("[%s] child %s done", phase, spec["tag"])
            if on_done is not None:
                on_done(spec)
        for slot in done_slots:
            running.pop(slot)
        if running and not done_slots:
            time.sleep(5.0)


# ── P0: assemble (verbatim mirror of N1M.assemble + new_ci return) + split carve ──


def _assemble_with_ci(ns, layer: int):
    """VERBATIM mirror of issue779_ffc_n1m_fits.assemble (merge ca9572810b) that ALSO
    returns the kept new-row ci (needed for the row->text mapping the SAE arm uses).
    The parent file stays byte-untouched (Gate A bit-comparability); equivalence is
    pinned by tests/test_issue1482_driver.py::test_assemble_mirror_matches_parent."""
    pb = N1G._load_pass_b_bundle(ns.pass_b)
    for fld in ("cx_last", "v_x"):
        assert fld in pb, f"pass_b missing {fld}"
    assert int(pb["cx_last"].shape[0]) == N1M.N_PASS_B, (pb["cx_last"].shape[0], N1M.N_PASS_B)
    pb_X = N50._slice_layer(pb, "cx_last", layer)
    pb_Y = N50._slice_layer(pb, "v_x", layer)

    manifest_args = argparse.Namespace(
        out_dir=ns.out_dir, manifest_from_hf=ns.manifest_from_hf, hf_prefix=ns.manifest_hf_prefix
    )
    manifest_dir = N1G._resolve_manifest_dir(manifest_args)
    pool, man_meta = N1G.read_manifest_pool(manifest_dir)
    ci_to_corpus = {int(r["i"]): r["corpus"] for r in pool}

    local_dir = ns.n1m_capture_dir if ns.n1m_capture_dir else None
    new_X, new_Y, new_ci = N1M._stream_n1m_layer(
        ns.hf_prefix,
        layer,
        local_dir,
        ns.out_dir / ".n1m_stream_cache",
        ckpt_dir=(ns.out_dir / ".n1m_stream_ckpt") if local_dir is None else None,
        ckpt_every=N1M.STREAM_CKPT_EVERY,
        fresh=ns.fresh_stream,
    )
    new_prov = np.array([ci_to_corpus[int(c)] for c in new_ci], dtype=object)

    X = np.concatenate([pb_X, new_X]).astype(np.float32)
    Y = np.concatenate([pb_Y, new_Y]).astype(np.float32)
    assert X.shape[1] == C.EXPECTED_HIDDEN and Y.shape[1] == C.EXPECTED_HIDDEN, (X.shape, Y.shape)
    prov = np.array(["lmsys"] * N1M.N_PASS_B + list(new_prov), dtype=object)
    assert prov.shape[0] == X.shape[0], (prov.shape, X.shape)

    pinned = N50._pinned_original_shas(ns.orig_dir)
    r1_train, val, test = F.fixed_split(
        N1M.N_PASS_B, N1M.N_PASS_B - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    val_sha, test_sha = F._sha_ids(val), F._sha_ids(test)
    assert val_sha == pinned["val_sha256"], (
        f"n1m val sha {val_sha} != pinned original {pinned['val_sha256']} — NOT byte-identical"
    )
    assert test_sha == pinned["test_sha256"], (
        f"n1m test sha {test_sha} != pinned original {pinned['test_sha256']}"
    )
    assert (val < N1M.N_PASS_B).all() and (test < N1M.N_PASS_B).all(), "val/test must index pass_b"

    split = {
        "orig_train_ids": len(r1_train),
        "n_new_captured": int(new_X.shape[0]),
        "n_new_manifest": int(man_meta["n_new"]),
        "n_val": len(val),
        "n_test": len(test),
        "val_sha256": val_sha,
        "test_sha256": test_sha,
        "pinned_val_sha256": pinned["val_sha256"],
        "pinned_test_sha256": pinned["test_sha256"],
        "layer": int(layer),
    }
    return X, Y, prov, r1_train, val, test, split, np.asarray(new_ci, dtype=np.int64)


def _stratified_sample(rng, rows: np.ndarray, prov_u8: np.ndarray, n: int, lmsys_frac: float):
    """Deterministic corpus-stratified sample of ``n`` rows (rebalance on shortfall)."""
    lm = rows[prov_u8[rows] == 0]
    wc = rows[prov_u8[rows] == 1]
    n = min(int(n), len(rows))
    n_l = min(round(n * lmsys_frac), len(lm))
    n_w = min(n - n_l, len(wc))
    n_l = min(n - n_w, len(lm))
    sel = np.concatenate(
        [
            lm[rng.choice(len(lm), size=n_l, replace=False)] if n_l else np.empty(0, np.int64),
            wc[rng.choice(len(wc), size=n_w, replace=False)] if n_w else np.empty(0, np.int64),
        ]
    ).astype(np.int64)
    return np.sort(sel), {"n": len(sel), "n_lmsys": int(n_l), "n_wildchat": int(n_w)}


def _stage_smoke_chunks(args) -> Path:
    """Smoke slice of the P0 chunk universe: stage the first --max-chunks capture
    chunks locally (scoped list_repo_tree + parent's bounded per-chunk retry)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    dest = args.scratch / "n1m_chunks_smoke"
    dest.mkdir(parents=True, exist_ok=True)
    files = hub.list_hf_files_under_path(
        HfApi(), C.HF_DATA_REPO, CAPTURE_PREFIX, repo_type="dataset"
    )
    names = sorted(p.rsplit("/", 1)[-1] for p in files if p.endswith(".pt"))[: args.max_chunks]
    for n in names:
        if not (dest / n).exists():
            got = Path(
                N1M._download_chunk_with_retry(C.HF_DATA_REPO, f"{CAPTURE_PREFIX}/{n}", dest)
            )
            if got != dest / n:  # hf_hub_download local_dir preserves the repo-relative path
                os.replace(got, dest / n)
    logger.info("[p0] staged %d smoke chunk(s) -> %s", len(names), dest)
    return dest


def phase_p0(args) -> None:
    C.phase("p0")
    split_path = args.out_eval / "split_1482.json"
    if split_path.exists() and (args.scratch / "X.npy").exists():
        prior = json.loads(split_path.read_text()).get("regime")
        cur = {
            "smoke": bool(args.smoke),
            "max_chunks": args.max_chunks,
            "holdout_n": args.holdout_n,
            "sae_n": args.sae_n,
            "sae_val_n": args.sae_val_n,
        }
        if prior == cur:
            logger.info("[p0] resume: split + X/Y present under matching regime; skip")
            return
        raise RuntimeError(f"[p0] out_eval holds a run under a DIFFERENT regime: {prior} != {cur}")
    _headroom(args.scratch, 4 if args.smoke else 40, "p0")
    n1m_capture_dir = _stage_smoke_chunks(args) if args.max_chunks > 0 else None
    ns = argparse.Namespace(
        pass_b=PROJECT_ROOT / "data" / "issue_779" / "pass_b" / "train_context_vectors.pt",
        out_dir=args.scratch,
        manifest_from_hf=True,
        hf_prefix=CAPTURE_PREFIX,
        manifest_hf_prefix=N1G.HF_PREFIX,
        n1m_capture_dir=n1m_capture_dir,
        fresh_stream=False,
        orig_dir=N1M.DEFAULT_ORIG_DIR,
    )
    X, Y, prov, r1_train, val, test, split, new_ci = _assemble_with_ci(ns, LAYER)
    n_total = X.shape[0]
    np.save(args.scratch / "X.npy", X)
    np.save(args.scratch / "Y.npy", Y)
    prov_u8 = (prov == "wildchat").astype(np.uint8)
    np.save(args.scratch / "prov.npy", prov_u8)
    row_ci = np.full(n_total, -1, dtype=np.int64)
    row_ci[N1M.N_PASS_B :] = new_ci
    np.save(args.scratch / "row_ci.npy", row_ci)

    pools = N1M._pool_rows(prov, r1_train, n_total, val, test)
    train_full = pools["full"]
    lmsys_frac = len(pools["lmsys"]) / len(train_full)
    new_rows = np.arange(N1M.N_PASS_B, n_total)
    rng = np.random.default_rng(SPLIT_SEED_1482)
    holdout, hold_diag = _stratified_sample(rng, new_rows, prov_u8, args.holdout_n, lmsys_frac)
    remaining = np.setdiff1d(new_rows, holdout, assume_unique=False)
    sae_fit, fit_diag = _stratified_sample(rng, remaining, prov_u8, args.sae_n, lmsys_frac)
    remaining2 = np.setdiff1d(remaining, sae_fit, assume_unique=False)
    sae_val, val_diag = _stratified_sample(rng, remaining2, prov_u8, args.sae_val_n, lmsys_frac)
    for nm, arr in (("holdout", holdout), ("sae_fit", sae_fit), ("sae_val", sae_val)):
        assert not (set(arr.tolist()) & (set(val.tolist()) | set(test.tolist()))), nm
    assert not (set(holdout.tolist()) & set(sae_fit.tolist()))
    assert not (set(holdout.tolist()) & set(sae_val.tolist()))
    assert not (set(sae_fit.tolist()) & set(sae_val.tolist()))

    np.savez(
        args.scratch / "split_indices.npz",
        train_full=train_full,
        train_lmsys=pools["lmsys"],
        val=val,
        test=test,
        holdout=holdout,
        sae_fit=sae_fit,
        sae_val=sae_val,
    )
    split_doc = {
        **split,
        "n_total": int(n_total),
        "realized_lmsys_frac_full_pool": round(float(lmsys_frac), 4),
        "holdout": {"sha256": _sha_ids(holdout), **hold_diag, "rng_seed": SPLIT_SEED_1482},
        "sae_fit": {"sha256": _sha_ids(sae_fit), **fit_diag},
        "sae_val": {"sha256": _sha_ids(sae_val), **val_diag},
        "train_full_sha256": _sha_ids(train_full),
        "regime": {
            "smoke": bool(args.smoke),
            "max_chunks": args.max_chunks,
            "holdout_n": args.holdout_n,
            "sae_n": args.sae_n,
            "sae_val_n": args.sae_val_n,
        },
        "plan_deviations": [
            "SAE arm covers NEW-pool rows only (holdout + fit + sae_val): the pinned pass_b "
            "bundle carries no prompts/responses and round-1 raw_completions are not on HF, "
            "so pinned val/test (pass_b rows) have no reconstructable text; P3 lambda "
            "selection uses the sae_val carve instead of pinned val (concern sae-arm-passb-text)."
        ],
    }
    _write_json(args.out_eval / "split_1482.json", split_doc)
    _phase_sentinel(
        "p0", f"p0 done n_total={n_total} holdout={len(holdout)} sae_fit={len(sae_fit)}"
    )
    logger.info(
        "[p0] done: n_total=%d holdout=%d sae_fit=%d sae_val=%d",
        n_total,
        len(holdout),
        len(sae_fit),
        len(sae_val),
    )


# ── P1: dense refits (parent fitters UNCHANGED) + per-context residuals + Gate A ──


def fit_specs(args) -> list[dict]:
    """The P1 fit registry — ONE source for dispatcher fan-out AND child resolve."""
    specs = []
    for pred in PREDICTORS_ALL:  # Gate A arms first (early gate check)
        specs.append({"condition": "refit_full", "predictor": pred, "seed": args.seed})
    for pred in PREDICTORS_ALL:
        specs.append({"condition": "refit_holdout", "predictor": pred, "seed": args.seed})
    specs.append({"condition": "refit_holdout", "predictor": "mlp_w8192", "seed": MLP_SEED_B})
    for pred in ("ridge", "mlp_w8192"):
        specs.append({"condition": "refit_lmsys_transfer", "predictor": pred, "seed": args.seed})
    for s in specs:
        s["fit_id"] = f"{s['condition']}__{s['predictor']}__seed{s['seed']}"
    return specs


def _load_xy(args):
    X = np.load(args.scratch / "X.npy", mmap_mode="r")
    Y = np.load(args.scratch / "Y.npy", mmap_mode="r")
    idx = np.load(args.scratch / "split_indices.npz")
    return X, Y, idx


def _train_rows_for(condition: str, idx, args) -> np.ndarray:
    if condition == "refit_full":
        tr = idx["train_full"]
    elif condition == "refit_holdout":
        tr = np.setdiff1d(idx["train_full"], idx["holdout"], assume_unique=False)
    elif condition == "refit_lmsys_transfer":
        tr = np.setdiff1d(idx["train_lmsys"], idx["holdout"], assume_unique=False)
    else:
        raise ValueError(condition)
    if args.fit_n > 0 and len(tr) > args.fit_n:  # smoke cap (regime-keyed in outputs)
        rng = np.random.default_rng(SPLIT_SEED_1482)
        tr = np.sort(tr[rng.choice(len(tr), size=args.fit_n, replace=False)])
    return np.asarray(tr, dtype=np.int64)


def _eval_sets_for(condition: str, idx) -> dict[str, np.ndarray]:
    if condition == "refit_full":
        return {"test": idx["test"], "val": idx["val"]}
    return {"holdout": idx["holdout"]}


def _percontext(pred: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    """Per-context e2 / cos / nerr with the eval set's OWN mean (matches _pooled_r2)."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    mu = t.mean(0)
    e2 = ((t - p) ** 2).sum(1)
    denom = ((t - mu) ** 2).sum(1)
    cos = (p * t).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    nerr = e2 / np.maximum(denom, 1e-12)
    return {"e2": e2, "cos": cos, "nerr": nerr, "denom": denom}


def phase_p1_fit(args) -> None:
    """ONE fit child: parent fitter UNCHANGED, eval-union trick for per-context preds."""
    spec = next(s for s in fit_specs(args) if s["fit_id"] == args.fit_id)
    X, Y, idx = _load_xy(args)
    tr = _train_rows_for(spec["condition"], idx, args)
    eval_sets = _eval_sets_for(spec["condition"], idx)
    eval_union = np.concatenate(list(eval_sets.values()))
    dev = torch.device(args.device)  # parent fitters expect a torch.device (dev.type)
    ns = argparse.Namespace(
        ridge_block=N1M.RIDGE_BLOCK,
        mlp_lr=3e-4,
        mlp_max_epochs=3 if args.smoke else F.MLP_MAX_EPOCHS,
        mlp_batch=N1M.MLP_BATCH,
        seed=spec["seed"],
        krr_nystrom_centers=args.krr_nystrom_centers,  # EXPLICIT 16384 (script default is 8192)
        no_validate_krr=False,
        krr_validate_tol=0.01,
    )
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
    t0 = time.time()
    pred_union, fit_meta = N1M._fit_one_predictor(
        spec["predictor"],
        X,
        Y,
        tr,
        idx["val"],
        eval_union,
        N1M.LAMBDAS_N1M,
        N1M.KRR_GAMMA_MULT,
        N1M.KRR_LAMBDAS,
        ns,
        dev,
    )
    wall = time.time() - t0
    out = {
        "fit_id": args.fit_id,
        **spec,
        "n_train": len(tr),
        "wall_time_s": round(wall, 1),
        "fit_meta": fit_meta,
        "regime": {
            "smoke": bool(args.smoke),
            "fit_n": args.fit_n,
            "krr_nystrom_centers": args.krr_nystrom_centers,
        },
        "sets": {},
    }
    arrays: dict[str, np.ndarray] = {}
    off = 0
    pdir = args.out_eval / "percontext"
    pdir.mkdir(parents=True, exist_ok=True)
    for name, rows in eval_sets.items():
        p = pred_union[off : off + len(rows)]
        off += len(rows)
        pc = _percontext(p, Y[rows])
        r2 = PR._pooled_r2(p, Y[rows])
        recon = 1.0 - pc["e2"].sum() / pc["denom"].sum()
        assert abs(recon - r2) < 1e-9, (recon, r2)  # exact-decomposition identity
        out["sets"][name] = {
            "n": len(rows),
            "whole_map_r2": float(r2),
            "mean_cosine": float(pc["cos"].mean()),
        }
        arrays[f"{name}_rows"] = np.asarray(rows, dtype=np.int64)
        for k in ("e2", "cos", "nerr", "denom"):
            arrays[f"{name}_{k}"] = pc[k].astype(np.float64)
        # holdout preds persisted fp16 for encode-the-prediction (ridge + mlp_w8192 only)
        if (
            name == "holdout"
            and spec["predictor"] in ("ridge", "mlp_w8192")
            and spec["seed"] == args.seed
        ):
            arrays["holdout_pred16"] = p.astype(np.float16)
    np.savez(pdir / f"{args.fit_id}.npz", **arrays)
    _write_json(pdir / f"{args.fit_id}.json", out)
    logger.info(
        "[p1-fit] %s done: %s", args.fit_id, {k: v["whole_map_r2"] for k, v in out["sets"].items()}
    )


def gate_a_check(args) -> dict:
    """Gate A reconciliation vs committed n1m_fits.json (plan §7). HALT scope:
    ridge/MLP miss -> rc=21 (production); peripheral arms WARN + drop. Smoke:
    computed identically, verdict demoted to informational (#1345 gate-calibration)."""
    committed = json.loads(COMMITTED_N1M.read_text())["per_point"]["mixed_1m"]["predictors"]
    pdir = args.out_eval / "percontext"
    rows, halt, dropped = {}, [], []
    for pred in PREDICTORS_ALL:
        fid = f"refit_full__{pred}__seed{args.seed}"
        j = json.loads((pdir / f"{fid}.json").read_text())
        got = j["sets"]["test"]["whole_map_r2"]
        want = committed[pred]["whole_map_r2"]
        delta = abs(got - want)
        tol = GATE_A_TOL.get(pred, GATE_A_PERIPHERAL_TOL.get(pred))
        ok = delta <= tol
        rows[pred] = {
            "refit_r2": got,
            "committed_r2": want,
            "abs_delta": delta,
            "tol": tol,
            "pass": ok,
        }
        if not ok:
            (halt if pred in GATE_A_TOL else dropped).append(pred)
    verdict = "HALT" if halt else ("WARN_DROP" if dropped else "PASS")
    doc = {
        "gate": "A",
        "verdict": verdict,
        "halt_arms": halt,
        "dropped_arms": dropped,
        "per_arm": rows,
        "smoke_demoted": bool(args.smoke),
        # durable scope caveat in the artifact the analyzer reads (concern
        # sae-arm-passb-text; also in split_1482.json plan_deviations)
        "scope_caveats": [
            "sae-arm-passb-text: SAE arm (P2/P3) covers NEW-pool rows only "
            "(holdout + sae_fit + sae_val) — the pinned pass_b bundle has no "
            "prompts/responses and round-1 raw completions are not on HF, so the "
            "1,400 pinned pass_b val/test contexts are excluded from P2/P3; "
            "P3 lambda selection uses the 2k sae_val carve."
        ],
    }
    sp_path = args.out_eval / SEED_PARTIAL_RECORD
    if sp_path.exists():
        sp = json.loads(sp_path.read_text())
        if sp.get("reconciled") and sp.get("seeded_units"):
            doc["seeded_from"] = sp["attempt_id"]  # provenance: seeded P1 units (r4)
            doc["seeded_units"] = sp["seeded_units"]
    _write_json(args.out_eval / "reconciliation.json", doc)
    logger.info("[gate-a] verdict=%s halt=%s dropped=%s", verdict, halt, dropped)
    if halt and not args.smoke:
        _phase_sentinel(
            "gate-a-halt",
            f"Gate A HALT arms={halt}",
            {"per_arm": {k: rows[k]["abs_delta"] for k in halt}},
        )
        raise SystemExit(RC_GATE_A)
    return doc


# ── --seed-partial: reuse a prior attempt's crash-persisted P1 artifacts (r4) ────
# The GCP crash-persist uploads eval_results_issue_1482/ to
# issue1482_partial/<attempt_id>/ (CLAUDE.md § GCP crash diagnostics). Seeding
# stages the P1 percontext units (+ sae_fitness.json) back into out_eval so the
# EXISTING resume predicates — phase_p1's per-unit json-exists skip and P2's
# pilot sae_fitness-exists skip — treat them as complete. Every seeded unit is
# VALIDATED (json parses + reconciles with its npz; corrupt/missing => the unit
# re-runs, never a silent skip), and the seeded split identity is reconciled
# against the CURRENT post-P0 split (_seed_partial_reconcile): a mismatch drops
# every seeded artifact loudly. P0 still runs (X/Y are rebuilt in scratch).

SEED_PARTIAL_RECORD = "seed_partial.json"


def _split_identity(doc: dict) -> dict:
    """The split-identity fingerprint seeded P1/P2 artifacts are only valid under."""
    return {
        "n_total": doc.get("n_total"),
        "train_full_sha256": doc.get("train_full_sha256"),
        "holdout_sha256": (doc.get("holdout") or {}).get("sha256"),
        "sae_fit_sha256": (doc.get("sae_fit") or {}).get("sha256"),
        "sae_val_sha256": (doc.get("sae_val") or {}).get("sha256"),
    }


def _validate_seed_unit(spec: dict, jpath: Path, npath: Path, args) -> str | None:
    """None when the seeded (json, npz) pair is complete + mutually consistent;
    else a short reason string (the unit then RE-RUNS — never a silent skip)."""
    if not (jpath.exists() and npath.exists()):
        return "missing json/npz"
    try:
        doc = json.loads(jpath.read_text())
    except ValueError:
        return "json unparseable"
    if doc.get("fit_id") != spec["fit_id"] or not doc.get("sets"):
        return "fit_id/sets mismatch"
    try:
        with np.load(npath) as z:
            files = set(z.files)
            for name, meta in doc["sets"].items():
                for k in ("rows", "e2", "cos", "nerr", "denom"):
                    if f"{name}_{k}" not in files:
                        return f"npz missing {name}_{k}"
                if int(meta["n"]) != int(z[f"{name}_rows"].shape[0]):
                    return f"npz row-count mismatch for {name}"
            needs_pred16 = (
                "holdout" in doc["sets"]
                and spec["predictor"] in ("ridge", "mlp_w8192")
                and spec["seed"] == args.seed
            )
            if needs_pred16 and "holdout_pred16" not in files:
                return "npz missing holdout_pred16 (P3 encode-the-prediction input)"
    except Exception as e:  # corrupt npz => re-run, never a silent skip
        return f"npz unreadable: {type(e).__name__}"
    return None


def seed_partial(args) -> None:
    """Stage attempt ``args.seed_partial``'s persisted P1 artifacts into out_eval.

    Fail-loud staging (hub.stage_hub_prefix: scoped listing + retried atomic
    per-file downloads); per-unit validation; npz moved BEFORE json so a crash
    mid-seed never leaves a resume-keyed unit without its payload. Writes the
    ``seed_partial.json`` record consumed by ``_seed_partial_reconcile`` +
    ``gate_a_check`` (seeded_from provenance)."""
    from explore_persona_space.orchestrate import hub

    attempt = args.seed_partial
    prefix = f"issue1482_partial/{attempt}/eval_results_issue_1482"
    stage = args.scratch / "seed_partial_stage"
    logger.info("[seed-partial] staging %s -> %s", prefix, stage)
    hub.stage_hub_prefix(C.HF_DATA_REPO, prefix, stage, repo_type="dataset")
    root = stage / prefix
    split_src = root / "split_1482.json"
    if not split_src.exists():
        raise RuntimeError(
            f"[seed-partial] {prefix} lacks split_1482.json — seeded P1 units cannot be "
            "validated against a split identity; refusing to seed"
        )
    split_identity = _split_identity(json.loads(split_src.read_text()))
    pdir = args.out_eval / "percontext"
    pdir.mkdir(parents=True, exist_ok=True)
    seeded: list[str] = []
    dropped: dict[str, str] = {}
    for spec in fit_specs(args):
        fid = spec["fit_id"]
        jpath = root / "percontext" / f"{fid}.json"
        npath = root / "percontext" / f"{fid}.npz"
        reason = _validate_seed_unit(spec, jpath, npath, args)
        if reason is not None:
            dropped[fid] = reason
            logger.warning("[seed-partial] unit %s NOT seeded (%s) — will re-run", fid, reason)
            continue
        shutil.move(str(npath), pdir / f"{fid}.npz")  # payload first,
        shutil.move(str(jpath), pdir / f"{fid}.json")  # resume KEY last
        seeded.append(fid)
    sae_seeded = False
    sf = root / "sae_fitness.json"
    if sf.exists():
        try:
            fdoc = json.loads(sf.read_text())
            ok = (
                fdoc.get("gate_b") == "PASS"
                and fdoc.get("g2_pass") is True
                and "chosen_k" in fdoc
                and not fdoc.get("smoke_demoted")
                and not fdoc.get("tiny_model")
            )
        except ValueError:
            ok = False
        if ok:
            shutil.move(str(sf), args.out_eval / "sae_fitness.json")
            sae_seeded = True
        else:
            logger.warning(
                "[seed-partial] sae_fitness.json NOT seeded (gate not clean-PASS) — pilot re-runs"
            )
    record = {
        "attempt_id": attempt,
        "hf_prefix": prefix,
        "seeded_units": seeded,
        "dropped_units": dropped,
        "sae_fitness_seeded": sae_seeded,
        "split_identity": split_identity,
        "reconciled": False,
        **C.reproducibility_metadata(),
    }
    _write_json(args.out_eval / SEED_PARTIAL_RECORD, record)
    logger.info(
        "[seed-partial] %d/%d P1 units seeded (dropped=%d) sae_fitness=%s from %s",
        len(seeded),
        len(fit_specs(args)),
        len(dropped),
        sae_seeded,
        attempt,
    )


def _seed_partial_reconcile(args) -> None:
    """Post-P0 guard: seeded artifacts are valid ONLY under the SAME split
    identity. On mismatch, drop every seeded artifact (units re-run) — never a
    silent skip of stale seeds. Idempotent (one reconcile per record)."""
    rec_path = args.out_eval / SEED_PARTIAL_RECORD
    if not rec_path.exists():
        return
    rec = json.loads(rec_path.read_text())
    if rec.get("reconciled") or rec.get("split_mismatch"):
        return
    cur = _split_identity(json.loads((args.out_eval / "split_1482.json").read_text()))
    if cur == rec["split_identity"]:
        rec["reconciled"] = True
        _write_json(rec_path, rec)
        logger.info(
            "[seed-partial] split identity MATCH — %d seeded P1 units honored",
            len(rec["seeded_units"]),
        )
        return
    pdir = args.out_eval / "percontext"
    for fid in rec["seeded_units"]:
        (pdir / f"{fid}.json").unlink(missing_ok=True)  # resume KEY first,
        (pdir / f"{fid}.npz").unlink(missing_ok=True)  # payload second
    if rec.get("sae_fitness_seeded"):
        (args.out_eval / "sae_fitness.json").unlink(missing_ok=True)
    rec.update({"split_mismatch": True, "reconciled": False, "current_split_identity": cur})
    _write_json(rec_path, rec)
    logger.warning(
        "[seed-partial] split identity MISMATCH vs attempt %s — dropped %d seeded units "
        "(sae_fitness dropped=%s); they RE-RUN",
        rec["attempt_id"],
        len(rec["seeded_units"]),
        rec.get("sae_fitness_seeded"),
    )


def phase_p1(args) -> None:
    C.phase("p1")
    _seed_partial_reconcile(args)
    specs = fit_specs(args)
    pdir = args.out_eval / "percontext"
    todo = [s for s in specs if not (pdir / f"{s['fit_id']}.json").exists()]
    logger.info("[p1] %d/%d fits to run", len(todo), len(specs))
    child_flags = _child_flags(args)
    _run_children(
        [
            {
                "tag": s["fit_id"],
                "cmd": ["--phase", "p1-fit", "--fit-id", s["fit_id"], *child_flags],
            }
            for s in todo
        ],
        args,
        "p1",
    )
    gate_a_check(args)
    _phase_sentinel("p1", f"p1 done ({len(specs)} fits)")


# ── P2: teacher-forced capture + SAE encode ─────────────────────────────────────


def _prefix_char_len(tok) -> int:
    """Constant chat-template prefix length in chars (everything before the user
    query; A4: single-turn => byte-identical across contexts). Sentinel-derived."""
    sent = "XX1482SENTINELXX"
    text = tok.apply_chat_template(
        [{"role": "user", "content": sent}], tokenize=False, add_generation_prompt=True
    )
    i = text.index(sent)
    assert i > 0, "sentinel not found in rendered template"
    return i


def _tokenize_row(tok, prompt: str, response: str, prefix_chars: int):
    """Parent-convention prompt tokenization + TOKEN-ID-CONCAT response (#1092 rule).

    Returns (full_ids, prefix_end, context_end, n_ans, seam_flag) or None on an
    empty response tokenization. prefix_end = last token ENTIRELY inside the
    constant template prefix (offset mapping, exclude-straddler policy)."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    enc = tok(text, return_offsets_mapping=True)  # parent-default special-token handling
    prompt_ids = enc["input_ids"]
    suffix = tok.decode(prompt_ids[-3:])
    assert suffix == C.GENERATION_SUFFIX, f"position assert: {suffix!r} != {C.GENERATION_SUFFIX!r}"
    offsets = enc["offset_mapping"]
    prefix_end = -1
    for i, (_, e) in enumerate(offsets):
        if e <= prefix_chars:
            prefix_end = i
        else:
            break
    assert prefix_end >= 0, "no token ends inside the template prefix"
    seam_flag = (
        int(offsets[prefix_end + 1][0] < prefix_chars) if prefix_end + 1 < len(offsets) else 0
    )
    resp_ids = tok(response, add_special_tokens=False)["input_ids"]
    if not resp_ids:
        return None
    full_ids = list(prompt_ids) + list(resp_ids)
    return full_ids, prefix_end, len(prompt_ids) - 1, len(resp_ids), seam_flag


def _load_model_tok(args):
    """Model + tokenizer for capture. Hub fetches ride ``hub.retry_transient``
    (P2-child-reachable; the #1482 attempt-2 429-storm class — transient
    metadata/HEAD failures retry, non-transient errors re-raise immediately)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.orchestrate import hub

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained(model_id), what=f"tokenizer fetch ({model_id})"
    )
    if args.tiny_model:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        logger.warning("[p2] --tiny-model: 24-layer from-config Qwen2 (REAL vocab; carve-out)")
        cfg = Qwen2Config(
            vocab_size=len(tok),
            hidden_size=3584,
            num_hidden_layers=24,
            intermediate_size=1024,
            num_attention_heads=28,
            num_key_value_heads=4,
            max_position_embeddings=32768,
            tie_word_embeddings=True,  # halves the embed+lm_head RSS on the VM smoke
        )
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(cfg)
        model.eval()
        return model, tok
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = hub.retry_transient(
        lambda: AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype),
        what=f"model fetch ({model_id})",
    )
    model.to(args.device if args.device == "cuda" else "cpu")
    model.eval()
    return model, tok


def _raw_chunk_names(args) -> list[str]:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = hub.list_hf_files_under_path(HfApi(), C.HF_DATA_REPO, RAW_PREFIX, repo_type="dataset")
    # ONLY rollout chunk files (shardNN_chunkNNNN.json). The prefix ALSO holds
    # per-shard bookkeeping files (shardNN_skipped.json — the parent's 156
    # over-length-skip records), which are valid JSON WITHOUT a "rows" key and
    # crashed p2 worker w3 with KeyError('rows') when swept into the chunk
    # enumeration (attempt 5, 2026-07-18). Row alignment is ci-explicit, so
    # excluding non-chunk files is side-effect-free.
    _chunk_re = re.compile(r"^shard\d+_chunk\d+\.json$")
    names = sorted(n for n in (q.rsplit("/", 1)[-1] for q in files) if _chunk_re.match(n))
    if args.max_chunks > 0:
        names = names[: args.max_chunks]
    return names


def _iter_needed_rows(args, names: list[str], needed_ci: dict[int, int]):
    """Yield (chunk_name, [(row_idx, ci, prompt, response)]) per raw chunk, downloading
    with the parent's bounded retry + delete-after (digest-only text handling)."""
    cache = args.scratch / "raw_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in names:
        got = Path(N1M._download_chunk_with_retry(C.HF_DATA_REPO, f"{RAW_PREFIX}/{name}", cache))
        rows = json.loads(got.read_text())["rows"]
        keep = [
            (needed_ci[int(r["ci"])], int(r["ci"]), r["prompt"], r["response"])
            for r in rows
            if int(r["ci"]) in needed_ci
        ]
        got.unlink()
        if keep:
            yield name, keep


def _batched_capture(model, tok, batch_rows, layers, device):
    """Right-padded batched teacher-forced forward; returns per-row dict of
    layer->(T_real, H) fp32 CPU tensors + positions. batch_rows: list of
    (row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam)."""
    from explore_persona_space.analysis.extraction import extract_layer_activations

    maxlen = max(len(r[2]) for r in batch_rows)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(batch_rows), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_rows), maxlen), dtype=torch.long)
    for i, r in enumerate(batch_rows):
        ids[i, : len(r[2])] = torch.tensor(r[2], dtype=torch.long)
        mask[i, : len(r[2])] = 1
    dev = device if device == "cuda" else "cpu"
    captured = extract_layer_activations(
        model, ids.to(dev), list(layers), attention_mask=mask.to(dev)
    )
    out = []
    for i, r in enumerate(batch_rows):
        T = len(r[2])
        out.append({li: captured[li][i, :T, :].float().cpu() for li in layers})
    return out


def _row_features(sae, h: torch.Tensor, context_end: int):
    """Per-row SAE feature extraction under the reference token-pool semantics
    (issue1482_sae docstring "TOKEN-POOL"): the first BOS_OFFSET positions and
    >10x-median-norm token rows are excluded from POOLED reads (answer trio +
    psi_mean) — the SAE's training and published eval both exclude them, and
    their features are off-distribution garbage (~42k active features on a
    massive-norm row). psi_last stays position-pinned at ``context_end`` (the
    map's own final-context-token read — independent of masking; unchanged vs
    r2, where it was ``f_ctx[-1]``). Returns
    ``(sp, spm, spl, ctx_n_out, ans_n_out, ans_all_out)``; falls back to the
    unmasked answer pool (flagged via ``ans_all_out``) when the mask would
    empty it; raises on an emptied context pool (a real context always spans
    the > BOS_OFFSET chat-template header)."""
    import issue1482_sae as S

    keep = S.token_inlier_mask(h)
    keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
    ctx_keep = keep[: context_end + 1]
    ans_keep = keep[context_end + 1 :]
    ctx_n_out = int((~ctx_keep).sum())
    ans_n_out = int((~ans_keep).sum())
    if int(ctx_keep.sum()) == 0:
        raise ValueError(
            f"_row_features: context pool empty after reference masking "
            f"(context_end={context_end}, ctx_n_out={ctx_n_out})"
        )
    f_ctx = sae.encode(h[: context_end + 1][ctx_keep])
    h_ans = h[context_end + 1 :]
    ans_all_out = int(h_ans.shape[0] > 0 and int(ans_keep.sum()) == 0)
    f_ans = sae.encode(h_ans if ans_all_out else h_ans[ans_keep])
    trio = S.pool_answer_features(f_ans)
    sp = S.sparsify(trio)
    psi_mean = f_ctx.mean(0)
    psi_last = sae.encode(h[context_end : context_end + 1])[0]
    spm = S.sparsify({"mean": psi_mean})
    spl = S.sparsify({"last": psi_last})
    return sp, spm, spl, ctx_n_out, ans_n_out, ans_all_out


def gate_b_verdict(fve64: float, fve128: float) -> tuple[str, int]:
    """Pure Gate B lattice (plan §7): >=0.70 PASS k64; [0.55,0.70) WARN (escalate to
    k128 if IT clears 0.70, else k64 + caveat); <0.55 HALT. Unit-probed."""
    if fve64 >= GATE_B_PASS:
        return "PASS", 64
    if fve64 >= GATE_B_HALT:
        return "WARN", (128 if fve128 >= GATE_B_PASS else 64)
    return "HALT", 64


def prefix_constancy_cos_min(hp: torch.Tensor) -> float:
    """Min pairwise-vs-row0 cosine of prefix-end states (A4 constancy read)."""
    hp_n = hp / hp.norm(dim=1, keepdim=True)
    return float((hp_n @ hp_n[0]).min())


def _pilot(args, model, tok, sae_loader, pilot_rows, X) -> dict:
    """P2-pilot: tokens/s at production batch shape, SAE fitness (FVE/L0, k64/k128,
    L15/19/23), prefix-end constancy, G2 identity gate vs stored cx_last."""
    import issue1482_sae as S

    t0 = time.time()
    caps = []
    tot_tokens = 0
    bs = args.gen_batch
    for s in range(0, len(pilot_rows), bs):
        batch = pilot_rows[s : s + bs]
        caps.extend(_batched_capture(model, tok, batch, PILOT_LAYERS, args.device))
        tot_tokens += sum(len(r[2]) for r in batch)
    tps = tot_tokens / max(1e-9, time.time() - t0)

    fitness: dict = {
        "tokens_per_s": round(tps, 1),
        "n_pilot": len(pilot_rows),
        # r3 reference-parity eval semantics (issue1482_sae docstring "TOKEN-POOL"):
        # published FVE 0.806/0.842 is measured under remove_bos (BOS strip +
        # >10x-median outlier drop + var-based FVE); Gate B calibrates against it.
        "bos_offset": S.BOS_OFFSET,
        "outlier_norm_factor": S.OUTLIER_NORM_FACTOR,
        "layers": {},
    }
    for li in PILOT_LAYERS:
        fitness["layers"][str(li)] = {}
    for kname, kval in (("k64", 64), ("k128", 128)):
        sae = sae_loader(kval)  # SEQUENTIAL: one SAE resident at a time (VM RSS bound)
        for li in PILOT_LAYERS:
            # reference remove_bos: drop the first BOS_OFFSET positions per sequence
            h_all = torch.cat([c[li][S.BOS_OFFSET :] for c in caps])
            fve, l0, diag = sae.fve_l0(h_all)
            fitness["layers"][str(li)][kname] = {
                "fve": round(float(fve), 4),
                "l0": round(float(l0), 2),
                **diag,
            }
        del sae
    # prefix-end constancy (A4): identical prefix tokens => near-identical states
    hp = torch.stack([caps[i][LAYER][pilot_rows[i][3], :] for i in range(len(caps))])
    cos_min = prefix_constancy_cos_min(hp)
    fitness["prefix_end_cos_min_vs_row0"] = round(cos_min, 6)
    assert cos_min >= PREFIX_CONSTANCY_COS_MIN, (
        f"prefix-end constancy FAILED: min cos {cos_min} < {PREFIX_CONSTANCY_COS_MIN} (A4 broken)"
    )
    # G2 identity gate: teacher-forced h19 @ context_end vs STORED cx_last (X row)
    g2 = []
    for i in range(min(8, len(caps))):
        h_ctx = caps[i][LAYER][pilot_rows[i][4], :]
        stored = torch.tensor(np.asarray(X[pilot_rows[i][0]], dtype=np.float32))
        g2.append(float(torch.nn.functional.cosine_similarity(h_ctx, stored, dim=0)))
    fitness["g2_cos"] = [round(v, 6) for v in g2]
    fitness["g2_cos_min"] = round(min(g2), 6)
    g2_pass = min(g2) >= G2_COS_MIN
    fve19 = fitness["layers"][str(LAYER)]
    gate_b, chosen_k = gate_b_verdict(fve19["k64"]["fve"], fve19["k128"]["fve"])
    fitness.update(
        {
            "gate_b": gate_b,
            "chosen_k": chosen_k,
            "g2_pass": bool(g2_pass),
            "published_fve": S.PUBLISHED_FVE,
            "tiny_model": bool(args.tiny_model),
            "smoke_demoted": bool(args.smoke),
        }
    )
    _write_json(args.out_eval / "sae_fitness.json", fitness)
    logger.info(
        "[p2-pilot] tps=%.0f fve19_k64=%s gate_b=%s g2_min=%.4f", tps, fve19["k64"], gate_b, min(g2)
    )
    if not args.smoke:
        if not g2_pass:
            _phase_sentinel("g2-halt", f"G2 identity gate FAILED min cos {min(g2)}")
            raise SystemExit(RC_GATE_B)
        if gate_b == "HALT":
            _phase_sentinel("gate-b-halt", f"Gate B HALT fve={fve19}")
            raise SystemExit(RC_GATE_B)
    return fitness


def phase_p2_worker(args) -> None:
    """One P2 GPU worker: capture + encode + pool its slice of the raw-chunk list."""
    import issue1482_sae as S

    idx = np.load(args.scratch / "split_indices.npz")
    row_ci = np.load(args.scratch / "row_ci.npy")
    set_tag = {}
    for tag in ("holdout", "sae_fit", "sae_val"):
        for r in idx[tag]:
            set_tag[int(r)] = tag
    needed_ci = {int(row_ci[r]): r for r in set_tag}
    assert -1 not in needed_ci, "SAE-arm rows must be NEW rows (text-resolvable)"
    fitness = json.loads((args.out_eval / "sae_fitness.json").read_text())
    k = int(args.sae_k) if args.sae_k != "auto" else int(fitness["chosen_k"])
    model, tok = _load_model_tok(args)
    sae = S.BatchTopKSAE.load(k=k, device=args.device, cache_dir=args.sae_dir)
    prefix_chars = _prefix_char_len(tok)
    names = _raw_chunk_names(args)[args.worker :: args.n_workers]
    store = args.store
    store.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for name, keep in _iter_needed_rows(args, names, needed_ci):
        shard_path = store / f"pooled_{Path(name).stem}_k{k}.npz"
        if shard_path.exists():
            continue
        rows = []
        for row_idx, ci, prompt, response in keep:
            tk = _tokenize_row(tok, prompt, response, prefix_chars)
            if tk is None:
                continue
            full_ids, prefix_end, context_end, n_ans, seam = tk
            rows.append((row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam))
        rows.sort(key=lambda r: len(r[2]))
        rec: dict[str, list] = {
            kk: []
            for kk in (
                "row_idx",
                "ci",
                "set_tag",
                "chunk",
                "n_ctx",
                "n_ans",
                "prefix_end",
                "seam",
                "idx_off",
                "ans_idx",
                "ans_mean",
                "ans_max",
                "ans_frac",
                "psi_off",
                "psi_idx",
                "psi_mean",
                "psil_off",
                "psil_idx",
                "psil_val",
                "h_prefix",
                "ctx_n_out",
                "ans_n_out",
                "ans_all_out",
            )
        }
        for s in range(0, len(rows), args.gen_batch):
            batch = rows[s : s + args.gen_batch]
            caps = _batched_capture(model, tok, batch, (LAYER,), args.device)
            for (row_idx, ci, _full_ids, prefix_end, context_end, n_ans, seam), cap in zip(
                batch, caps, strict=True
            ):
                h = cap[LAYER]
                # r3: reference token-pool masking (BOS strip + outlier drop) for
                # POOLED reads; psi_last stays position-pinned at context_end.
                sp, spm, spl, ctx_n_out, ans_n_out, ans_all_out = _row_features(sae, h, context_end)
                rec["ctx_n_out"].append(ctx_n_out)
                rec["ans_n_out"].append(ans_n_out)
                rec["ans_all_out"].append(ans_all_out)
                rec["row_idx"].append(row_idx)
                rec["ci"].append(ci)
                rec["set_tag"].append({"holdout": 0, "sae_fit": 1, "sae_val": 2}[set_tag[row_idx]])
                rec["chunk"].append(Path(name).stem)
                rec["n_ctx"].append(context_end + 1)
                rec["n_ans"].append(n_ans)
                rec["prefix_end"].append(prefix_end)
                rec["seam"].append(seam)
                rec["idx_off"].append(len(sp["idx"]))
                rec["ans_idx"].append(sp["idx"])
                rec["ans_mean"].append(sp["mean"])
                rec["ans_max"].append(sp["max"])
                rec["ans_frac"].append(sp["frac"])
                rec["psi_off"].append(len(spm["idx"]))
                rec["psi_idx"].append(spm["idx"])
                rec["psi_mean"].append(spm["mean"])
                rec["psil_off"].append(len(spl["idx"]))
                rec["psil_idx"].append(spl["idx"])
                rec["psil_val"].append(spl["last"])
                rec["h_prefix"].append(h[prefix_end].numpy().astype(np.float16))
        arrays = {
            "row_idx": np.asarray(rec["row_idx"], np.int64),
            "ci": np.asarray(rec["ci"], np.int64),
            "set_tag": np.asarray(rec["set_tag"], np.int8),
            "chunk": np.asarray(rec["chunk"], dtype=object),
            "n_ctx": np.asarray(rec["n_ctx"], np.int32),
            "n_ans": np.asarray(rec["n_ans"], np.int32),
            "prefix_end": np.asarray(rec["prefix_end"], np.int32),
            "seam": np.asarray(rec["seam"], np.int8),
            "idx_off": np.asarray(rec["idx_off"], np.int64),
            "ans_idx": np.concatenate(rec["ans_idx"]) if rec["ans_idx"] else np.empty(0, np.int32),
            "ans_mean": np.concatenate(rec["ans_mean"])
            if rec["ans_mean"]
            else np.empty(0, np.float16),
            "ans_max": np.concatenate(rec["ans_max"])
            if rec["ans_max"]
            else np.empty(0, np.float16),
            "ans_frac": np.concatenate(rec["ans_frac"])
            if rec["ans_frac"]
            else np.empty(0, np.float16),
            "psi_off": np.asarray(rec["psi_off"], np.int64),
            "psi_idx": np.concatenate(rec["psi_idx"]) if rec["psi_idx"] else np.empty(0, np.int32),
            "psi_mean": np.concatenate(rec["psi_mean"])
            if rec["psi_mean"]
            else np.empty(0, np.float16),
            "psil_off": np.asarray(rec["psil_off"], np.int64),
            "psil_idx": np.concatenate(rec["psil_idx"])
            if rec["psil_idx"]
            else np.empty(0, np.int32),
            "psil_val": np.concatenate(rec["psil_val"])
            if rec["psil_val"]
            else np.empty(0, np.float16),
            "h_prefix": np.stack(rec["h_prefix"])
            if rec["h_prefix"]
            else np.empty((0, 3584), np.float16),
            "ctx_n_out": np.asarray(rec["ctx_n_out"], np.int16),
            "ans_n_out": np.asarray(rec["ans_n_out"], np.int16),
            "ans_all_out": np.asarray(rec["ans_all_out"], np.int8),
        }
        tmp = shard_path.parent / f".tmp_{shard_path.name}"
        np.savez(tmp, **{k2: v for k2, v in arrays.items() if k2 != "chunk"})
        os.replace(tmp, shard_path)
        n_done += len(rec["row_idx"])
        logger.info(
            "[p2-worker %d] shard %s: %d ctx (total %d; ans_all_out=%d)",
            args.worker,
            shard_path.name,
            len(rec["row_idx"]),
            n_done,
            int(sum(rec["ans_all_out"])),
        )
    # per-worker G2 spot check on the FIRST processed shard rows (vs stored cx_last)
    logger.info("[p2-worker %d] done (%d contexts)", args.worker, n_done)


def phase_p2(args) -> None:
    C.phase("p2")
    _seed_partial_reconcile(args)  # no-op unless a --seed-partial record is unreconciled
    _headroom(args.store, 2 if args.smoke else 25, "p2")
    import issue1482_sae as S

    # SAE pre-stage (r5) — runs REGARDLESS of pilot skip: the pilot used to be the
    # only code path staging the SAE weights into args.sae_dir, so the
    # --seed-partial path (sae_fitness.json seeded => pilot skipped) sent N
    # p2-workers into concurrent hf_hub_download()s against ONE empty shared
    # local_dir — the #1315 fanout shared-staging race (pod-1482 p2_w0:
    # torch.load(ae.pt) FileNotFoundError). Stage ONCE in the parent,
    # idempotently, for BOTH trainers (--sae-k may name either k); the pilot's,
    # workers', and P3-aux's own load() downloads then become no-ops.
    for kval in sorted(S.TRAINER_SUBDIR):
        S.BatchTopKSAE.ensure_downloaded(kval, args.sae_dir)

    # pilot FIRST (Gate B input): one process, first pilot-n needed rows
    if not (args.out_eval / "sae_fitness.json").exists():
        idx = np.load(args.scratch / "split_indices.npz")
        row_ci = np.load(args.scratch / "row_ci.npy")
        X = np.load(args.scratch / "X.npy", mmap_mode="r")
        rows_all = np.sort(np.concatenate([idx["holdout"], idx["sae_fit"], idx["sae_val"]]))
        needed_ci = {int(row_ci[r]): int(r) for r in rows_all}
        model, tok = _load_model_tok(args)
        prefix_chars = _prefix_char_len(tok)
        pilot_rows = []
        for _, keep in _iter_needed_rows(args, _raw_chunk_names(args), needed_ci):
            for row_idx, ci, prompt, response in keep:
                tk = _tokenize_row(tok, prompt, response, prefix_chars)
                if tk is None:
                    continue
                full_ids, prefix_end, context_end, n_ans, seam = tk
                pilot_rows.append((row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam))
                if len(pilot_rows) >= args.pilot_n:
                    break
            if len(pilot_rows) >= args.pilot_n:
                break

        def _sae_loader(k):
            return S.BatchTopKSAE.load(k=k, device=args.device, cache_dir=args.sae_dir)

        _pilot(args, model, tok, _sae_loader, pilot_rows, X)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    # main capture: shard the raw-chunk list across workers (subprocess per GPU)
    gpus = _physical_gpu_ids() if args.device != "cpu" else []
    n_workers = max(1, len(gpus)) if args.n_gpus <= 0 else args.n_gpus
    child_flags = _child_flags(args)
    _run_children(
        [
            {
                "tag": f"w{i}",
                "cmd": [
                    "--phase",
                    "p2-worker",
                    "--worker",
                    str(i),
                    "--n-workers",
                    str(n_workers),
                    *child_flags,
                ],
            }
            for i in range(n_workers)
        ],
        args,
        "p2",
    )
    _phase_sentinel("p2", "p2 done (pooled store written)")


# ── P3: SAE-space fits + per-feature reads ──────────────────────────────────────


def _load_store(args):
    """Load all pooled shards -> dense-ready sparse bundles + row registry."""
    shards = sorted(args.store.glob("pooled_*.npz"))
    assert shards, f"no pooled shards under {args.store}"
    parts = [dict(np.load(p, allow_pickle=False)) for p in shards]
    return parts


def _densify(
    parts, key_idx, key_off, key_val, feat_ids: np.ndarray, n_rows: int, row_pos: dict
) -> np.ndarray:
    """Scatter sparse (idx, val) rows into a dense (n_rows, len(feat_ids)) fp32 matrix."""
    col_of = np.full(int(feat_ids.max()) + 1 if len(feat_ids) else 1, -1, dtype=np.int64)
    col_of[feat_ids] = np.arange(len(feat_ids))
    M = np.zeros((n_rows, len(feat_ids)), dtype=np.float32)
    for part in parts:
        offs = np.concatenate([[0], np.cumsum(part[key_off])])
        for i, r in enumerate(part["row_idx"]):
            pos = row_pos.get(int(r))
            if pos is None:
                continue
            sl = slice(offs[i], offs[i + 1])
            fidx = part[key_idx][sl].astype(np.int64)
            keep = fidx < len(col_of)
            cols = col_of[fidx[keep]]
            m = cols >= 0
            M[pos, cols[m]] = part[key_val][sl][keep][m].astype(np.float32)
    return M


def _activity_counts(
    parts, key_idx, key_off, only_tag: int, dict_size: int
) -> tuple[np.ndarray, int]:
    counts = np.zeros(dict_size, dtype=np.int64)
    n = 0
    for part in parts:
        offs = np.concatenate([[0], np.cumsum(part[key_off])])
        for i, tag in enumerate(part["set_tag"]):
            if int(tag) != only_tag:
                continue
            n += 1
            counts[part[key_idx][offs[i] : offs[i + 1]].astype(np.int64)] += 1
    return counts, n


def _shared_gram_ridge_multi(Z, targets: dict[str, np.ndarray], tr, va, te, lambdas, dev, block):
    """ONE parent _ridge_factorize on the CONCATENATED targets (shared X^TX + eigh),
    per-target lambda selection on va, predictions on te. Parent internals UNCHANGED."""
    dims = {k: t.shape[1] for k, t in targets.items()}
    Ycat = np.concatenate([targets[k] for k in targets], axis=1)
    # Cap the fp64 Y row-block by BYTES, not rows: the parent's default block was
    # sized for dense 3,584-col targets; the SAE arm concatenates 3 poolings x
    # F_out (49,152 cols realized) and the same row count became an 18.3 GiB
    # single transfer -> CUDA OOM (attempt 6, p3_ridge__sae_ctx). ~4 GiB target
    # keeps per-step transfers bounded on any target width; parent internals
    # UNCHANGED (block is an existing _ridge_factorize parameter).
    _Y_BLOCK_BYTES = 4 * (1 << 30)
    block = min(block, max(2048, int(_Y_BLOCK_BYTES // (Ycat.shape[1] * 8))))
    fac = N1M._ridge_factorize(Z, Ycat, tr, dev, block)
    bounds, off = {}, 0
    for k in targets:
        bounds[k] = (off, off + dims[k])
        off += dims[k]
    best = {k: (float(lambdas[0]), -np.inf) for k in targets}
    for lam in lambdas:
        pv = N1M._ridge_predict_one(Z, va, fac, lam, dev, block)
        for k, (a, b) in bounds.items():
            r2 = PR._pooled_r2(pv[:, a:b], Ycat[va][:, a:b])
            if np.isfinite(r2) and r2 > best[k][1]:
                best[k] = (float(lam), r2)
    out = {}
    for k, (a, b) in bounds.items():
        lam = best[k][0]
        pt = N1M._ridge_predict_one(Z, te, fac, lam, dev, block)[:, a:b]
        out[k] = (pt, {"selected_lambda": lam, "val_r2": float(best[k][1])})
    return out


def _midrank(a: np.ndarray) -> np.ndarray:
    """Column-wise midrank (scipy ``rankdata(method='average')`` tie semantics) as
    batched tensor ops — tie groups get their AVERAGE ordinal rank. Sparse SAE
    targets are ~95-99% exact-zero ties, so ordinal double-argsort ranks largely
    correlate arbitrary tie order (concern p3-rank-stats-corpus-order). The outer
    loop is a fixed-size FEATURE-BLOCK chunk (memory bound), not a per-feature loop."""
    n = a.shape[0]
    out = np.empty(a.shape, dtype=np.float64)
    pos = np.arange(n, dtype=np.float64)[:, None]
    for j in range(0, a.shape[1], 4096):
        blk = a[:, j : j + 4096]
        order = np.argsort(blk, axis=0, kind="stable")
        s = np.take_along_axis(blk, order, axis=0)
        new_grp = np.ones(blk.shape, dtype=bool)
        new_grp[1:] = s[1:] != s[:-1]
        grp_start = np.maximum.accumulate(np.where(new_grp, pos, -1.0), axis=0)
        last_in_grp = np.ones(blk.shape, dtype=bool)
        last_in_grp[:-1] = new_grp[1:]
        grp_end = np.minimum.accumulate(np.where(last_in_grp, pos, float(n))[::-1], axis=0)[::-1]
        mid = (grp_start + grp_end) / 2.0 + 1.0  # 1-based midranks in sorted position
        np.put_along_axis(out[:, j : j + 4096], order, mid, axis=0)
    return out


def _splithalf_perm(n: int) -> np.ndarray:
    """Seeded permutation of the holdout rows BEFORE split-half halving. The parent
    manifest is corpus-BLOCKED (lmsys_pool ++ wildchat_pool,
    issue779_ffc_n1m_generate_capture ``pool = lmsys_pool + wildchat_pool``), so a
    sorted-row-order midpoint split makes half A ~all-LMSYS / half B ~mostly-WildChat
    — a corpus-transfer read, not a stability read (concern
    p3-rank-stats-corpus-order). Seed = SPLIT_SEED_1482, recorded in the P3 summary."""
    return np.random.default_rng(SPLIT_SEED_1482).permutation(n)


def _per_feature_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    """Batched per-feature held-out R2 + Spearman with MIDRANK (average) ties —
    no per-feature Python loop."""
    p = pred.astype(np.float64)
    t = true.astype(np.float64)
    mu = t.mean(0)
    ss_res = ((t - p) ** 2).sum(0)
    ss_tot = ((t - mu) ** 2).sum(0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    rp = _midrank(p)
    rt = _midrank(t)
    rp -= rp.mean(0)
    rt -= rt.mean(0)
    num = (rp * rt).sum(0)
    den = np.sqrt((rp**2).sum(0) * (rt**2).sum(0))
    rho = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
    return {"r2": r2, "spearman": rho, "ss_tot": ss_tot}


P3_ARMS_RIDGE = ("sae_ctx", "sae_dense_in", "sae_prefix_null")
P3_ARMS_MLP = ("sae_ctx", "sae_dense_in")
P3_POOLINGS = ("mean", "max", "frac")


def p3_unit_specs() -> list[str]:
    """The P3 unit registry — ONE source for dispatcher fan-out AND child resolve
    (mirrors fit_specs; concern p3-serial-single-gpu: 3 arm-Grams + 6 MLP fits +
    the aux reads shard across the realized GPU width via _run_children)."""
    units = [f"ridge__{arm}" for arm in P3_ARMS_RIDGE]
    units += [f"mlp__{arm}__{pool}" for arm in P3_ARMS_MLP for pool in P3_POOLINGS]
    units.append("aux")
    return units


def _p3_unit_json(args, unit: str) -> Path:
    return args.out_eval / "sae_perfeature" / f"unit_{unit}.json"


def _p3_prep(args):
    """Shared P3 preprocessing: store load, feature restriction, row registry,
    split positions. Each fan-out child recomputes this (CPU-only, minutes) rather
    than shipping the ~30-60 GB densified matrices across processes; the arms'
    densified designs are built lazily per unit (_p3_design/_p3_targets)."""
    parts = _load_store(args)
    idx = np.load(args.scratch / "split_indices.npz")
    import issue1482_sae as S

    dict_size = S.DICT_SIZE
    # feature restriction (activity >= 1% of fit contexts; cap for Gram tractability)
    out_counts, n_fit = _activity_counts(
        parts, "ans_idx", "idx_off", only_tag=1, dict_size=dict_size
    )
    in_counts, _ = _activity_counts(parts, "psi_idx", "psi_off", only_tag=1, dict_size=dict_size)
    floor = max(1, int(np.ceil(0.01 * n_fit)))
    f_out = np.where(out_counts >= floor)[0]
    f_in = np.where(in_counts >= floor)[0]
    if len(f_out) > args.max_features_out:
        f_out = f_out[np.argsort(-out_counts[f_out])[: args.max_features_out]]
        f_out = np.sort(f_out)
    if len(f_in) > args.max_features_in:
        f_in = f_in[np.argsort(-in_counts[f_in])[: args.max_features_in]]
        f_in = np.sort(f_in)
    assert len(f_out) >= 1 and len(f_in) >= 1, (len(f_out), len(f_in))
    logger.info(
        "[p3] F_out=%d F_in=%d (floor=%d over n_fit=%d)", len(f_out), len(f_in), floor, n_fit
    )
    # row registry: matrix rows = sae_fit ++ sae_val ++ holdout (in that order)
    order = np.concatenate([idx["sae_fit"], idx["sae_val"], idx["holdout"]])
    have = set()
    for part in parts:
        have.update(int(r) for r in part["row_idx"])
    order = np.asarray([r for r in order if int(r) in have], dtype=np.int64)
    row_pos = {int(r): i for i, r in enumerate(order)}
    n_rows = len(order)
    tr = np.asarray([row_pos[int(r)] for r in idx["sae_fit"] if int(r) in row_pos], dtype=np.int64)
    va = np.asarray([row_pos[int(r)] for r in idx["sae_val"] if int(r) in row_pos], dtype=np.int64)
    te = np.asarray([row_pos[int(r)] for r in idx["holdout"] if int(r) in row_pos], dtype=np.int64)
    assert len(tr) and len(va) and len(te), (len(tr), len(va), len(te))
    return argparse.Namespace(
        parts=parts,
        f_out=f_out,
        f_in=f_in,
        out_counts=out_counts,
        n_fit=n_fit,
        floor=floor,
        order=order,
        row_pos=row_pos,
        n_rows=n_rows,
        tr=tr,
        va=va,
        te=te,
    )


def _p3_targets(prep, pools: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Densify ONLY the requested answer-side pooling targets (lazy per unit)."""
    key = {"mean": "ans_mean", "max": "ans_max", "frac": "ans_frac"}
    return {
        p: _densify(prep.parts, "ans_idx", "idx_off", key[p], prep.f_out, prep.n_rows, prep.row_pos)
        for p in pools
    }


def _p3_design(args, prep, arm: str) -> np.ndarray:
    """Design matrix Z for one P3 arm (lazy per unit)."""
    if arm == "sae_ctx":
        psi_mean = _densify(
            prep.parts, "psi_idx", "psi_off", "psi_mean", prep.f_in, prep.n_rows, prep.row_pos
        )
        psi_last = _densify(
            prep.parts, "psil_idx", "psil_off", "psil_val", prep.f_in, prep.n_rows, prep.row_pos
        )
        return np.concatenate([psi_mean, psi_last], axis=1)
    if arm == "sae_dense_in":
        X = np.load(args.scratch / "X.npy", mmap_mode="r")
        return np.asarray(X[prep.order], dtype=np.float32)
    if arm == "sae_prefix_null":
        h_prefix = np.concatenate([p["h_prefix"] for p in prep.parts]).astype(np.float32)
        hp_rows = np.concatenate([p["row_idx"] for p in prep.parts])
        hp = np.zeros((prep.n_rows, 3584), dtype=np.float32)
        for r, v in zip(hp_rows, h_prefix, strict=True):
            pos = prep.row_pos.get(int(r))
            if pos is not None:
                hp[pos] = v
        return hp
    raise ValueError(arm)


def _p3_unit_ridge(args, prep, arm: str) -> None:
    """ONE arm's shared-Gram multi-target ridge + per-feature reads + split-half."""
    dev = torch.device(args.device)
    pf_dir = args.out_eval / "sae_perfeature"
    pf_dir.mkdir(parents=True, exist_ok=True)
    pools = ("mean",) if arm == "sae_prefix_null" else P3_POOLINGS
    tgt = _p3_targets(prep, pools)
    Z = _p3_design(args, prep, arm)
    te = prep.te
    # split-half halves: seeded permutation BEFORE halving (corpus-blocked manifest
    # order — see _splithalf_perm; concern p3-rank-stats-corpus-order)
    perm = _splithalf_perm(len(te))
    ia, ib = perm[: len(te) // 2], perm[len(te) // 2 :]
    half_a, half_b = te[ia], te[ib]
    preds = _shared_gram_ridge_multi(
        Z, tgt, prep.tr, prep.va, te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
    )
    arm_doc = {}
    for pool_name, (pt, meta) in preds.items():
        pf = _per_feature_metrics(pt, tgt[pool_name][te])
        pooled_r2 = PR._pooled_r2(pt, tgt[pool_name][te])
        # split-half rank stability of the per-feature R2 ranking (midrank ties)
        pa = _per_feature_metrics(pt[ia], tgt[pool_name][half_a])
        pb = _per_feature_metrics(pt[ib], tgt[pool_name][half_b])
        ok = np.isfinite(pa["r2"]) & np.isfinite(pb["r2"])
        if ok.sum() >= 3:
            ra = _midrank(pa["r2"][ok][:, None])[:, 0]
            rb = _midrank(pb["r2"][ok][:, None])[:, 0]
            stab = float(np.corrcoef(ra, rb)[0, 1])
        else:
            stab = float("nan")
        np.savez(
            pf_dir / f"{arm}__{pool_name}__ridge.npz",
            feat_ids=prep.f_out,
            r2=pf["r2"],
            spearman=pf["spearman"],
            activity=prep.out_counts[prep.f_out] / max(1, prep.n_fit),
        )
        arm_doc[f"{pool_name}__ridge"] = {
            "pooled_r2": float(pooled_r2),
            **meta,
            "splithalf_rank_stability": stab,
            "splithalf_permutation_seed": SPLIT_SEED_1482,
            "n_feat_finite": int(np.isfinite(pf["r2"]).sum()),
        }
        logger.info("[p3] %s/%s ridge pooled R2=%.4f", arm, pool_name, pooled_r2)
    _write_json(
        _p3_unit_json(args, f"ridge__{arm}"),
        {
            "arm": arm,
            "arm_doc": arm_doc,
            "scalars": {
                "f_out": len(prep.f_out),
                "f_in": len(prep.f_in),
                "activity_floor": prep.floor,
                "n_fit": len(prep.tr),
                "n_val": len(prep.va),
                "n_holdout": len(te),
            },
        },
    )


def _p3_unit_mlp(args, prep, arm: str, pool: str) -> None:
    """ONE (arm, pooling) MLP fit + per-feature reads."""
    dev = torch.device(args.device)
    pf_dir = args.out_eval / "sae_perfeature"
    pf_dir.mkdir(parents=True, exist_ok=True)
    tgt = _p3_targets(prep, (pool,))[pool]
    Z = _p3_design(args, prep, arm)
    ns_mlp = dict(
        width=N1M.MLP_W_PROTOCOL,
        lr=3e-4,
        max_epochs=3 if args.smoke else F.MLP_MAX_EPOCHS,
        batch=min(N1M.MLP_BATCH, max(8, len(prep.tr))),
        seed=args.seed,
        dev=dev,
    )
    pt, meta = N1M._fit_mlp_minibatch(Z, tgt, prep.tr, prep.te, **ns_mlp)
    pf = _per_feature_metrics(pt, tgt[prep.te])
    np.savez(
        pf_dir / f"{arm}__{pool}__mlp.npz",
        feat_ids=prep.f_out,
        r2=pf["r2"],
        spearman=pf["spearman"],
        activity=prep.out_counts[prep.f_out] / max(1, prep.n_fit),
    )
    pooled = float(PR._pooled_r2(pt, tgt[prep.te]))
    _write_json(
        _p3_unit_json(args, f"mlp__{arm}__{pool}"),
        {
            "arm": arm,
            "arm_doc": {f"{pool}__mlp": {"pooled_r2": pooled, "epochs_ran": meta["epochs_ran"]}},
        },
    )
    logger.info("[p3] %s/%s mlp pooled R2=%.4f", arm, pool, pooled)


def _p3_unit_aux(args, prep) -> None:
    """Prefix-null dense map + encode-the-prediction + per-direction PCA read."""
    import issue1482_sae as S

    dev = torch.device(args.device)
    pf_dir = args.out_eval / "sae_perfeature"
    pf_dir.mkdir(parents=True, exist_ok=True)
    Y = np.load(args.scratch / "Y.npy", mmap_mode="r")
    block = N1M.RIDGE_BLOCK
    doc: dict = {}
    # prefix-null DENSE map: h_prefix -> v(x) at subsample scale (registered null)
    hp = _p3_design(args, prep, "sae_prefix_null")
    Yd = np.asarray(Y[prep.order], dtype=np.float32)
    pred_null, _meta_null = N1M.fit_ridge(
        hp, Yd, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, block
    )
    doc["prefix_dense_to_vx_r2"] = float(PR._pooled_r2(pred_null, Yd[prep.te]))
    # encode-the-prediction (off-distribution secondary; labeled): dense-map preds vs truth
    fitness = json.loads((args.out_eval / "sae_fitness.json").read_text())
    k = int(fitness["chosen_k"])
    sae = S.BatchTopKSAE.load(k=k, device=dev, cache_dir=args.sae_dir)
    ep = {}
    for pred_name in ("ridge", "mlp_w8192"):
        npz = np.load(
            args.out_eval / "percontext" / f"refit_holdout__{pred_name}__seed{args.seed}.npz"
        )
        rows_h = npz["holdout_rows"]
        pos = np.asarray(
            [prep.row_pos[int(r)] for r in rows_h if int(r) in prep.row_pos], dtype=np.int64
        )
        keep = np.asarray([i for i, r in enumerate(rows_h) if int(r) in prep.row_pos], np.int64)
        vhat = torch.tensor(npz["holdout_pred16"][keep].astype(np.float32))
        vtrue = torch.tensor(np.asarray(Y[rows_h[keep]], dtype=np.float32))
        f_hat = sae.encode(vhat).cpu().numpy()[:, prep.f_out]
        f_true = sae.encode(vtrue).cpu().numpy()[:, prep.f_out]
        pf = _per_feature_metrics(f_hat, f_true)
        np.savez(
            pf_dir / f"encode_pred__{pred_name}.npz",
            feat_ids=prep.f_out,
            r2=pf["r2"],
            spearman=pf["spearman"],
        )
        ep[pred_name] = {"n": len(pos), "n_feat_finite": int(np.isfinite(pf["r2"]).sum())}
    doc["encode_the_prediction"] = {
        "note": "off-distribution SAE-of-mean transform applied EQUALLY "
        "to v_hat and v (labeled secondary)",
        **ep,
    }
    # per-direction PCA read on the DENSE map (top-256 PCA of Y on train)
    tr_full = np.load(args.scratch / "split_indices.npz")["train_full"]
    A = torch.zeros((3584, 3584), dtype=torch.float64)
    mu_acc = torch.zeros(3584, dtype=torch.float64)
    n_acc = 0
    for s in range(0, len(tr_full), block):
        yb = torch.tensor(np.asarray(Y[tr_full[s : s + block]], dtype=np.float64))
        A += yb.T @ yb
        mu_acc += yb.sum(0)
        n_acc += yb.shape[0]
    mu = mu_acc / n_acc
    A = A / n_acc - torch.outer(mu, mu)
    evals, evecs = torch.linalg.eigh(A)
    top = torch.flip(evecs[:, -256:], dims=[1]).numpy()
    perdir = {}
    for pred_name in ("ridge", "mlp_w8192"):
        npz = np.load(
            args.out_eval / "percontext" / f"refit_holdout__{pred_name}__seed{args.seed}.npz"
        )
        rows_h = npz["holdout_rows"]
        vt = np.asarray(Y[rows_h], dtype=np.float64)
        vp = npz["holdout_pred16"].astype(np.float64)
        pf = _per_feature_metrics(vp @ top, vt @ top)
        perdir[pred_name] = pf["r2"].tolist()
    _write_json(
        args.out_eval / "perdirection_pca.json",
        {
            "eigvals_top": torch.flip(evals[-256:], dims=[0]).numpy().tolist(),
            "per_direction_r2": perdir,
        },
    )
    _write_json(_p3_unit_json(args, "aux"), doc)


def _p3_run_unit(args) -> None:
    """ONE P3 unit child (CVD-pinned per slot by _run_children in the launcher env)."""
    unit = args.p3_unit
    assert unit in p3_unit_specs(), unit
    prep = _p3_prep(args)
    if unit == "aux":
        _p3_unit_aux(args, prep)
    elif unit.startswith("ridge__"):
        _p3_unit_ridge(args, prep, unit.removeprefix("ridge__"))
    else:
        _, arm, pool = unit.split("__")
        _p3_unit_mlp(args, prep, arm, pool)


def phase_p3(args) -> None:
    if args.p3_unit:  # child mode: one unit, no phase sentinel (parent owns it)
        _p3_run_unit(args)
        return
    C.phase("p3")
    pf_dir = args.out_eval / "sae_perfeature"
    pf_dir.mkdir(parents=True, exist_ok=True)
    units = p3_unit_specs()
    todo = [u for u in units if not _p3_unit_json(args, u).exists()]
    logger.info("[p3] %d/%d units to run (re-shard-safe unit-JSON skip)", len(todo), len(units))
    child_flags = _child_flags(args)
    _run_children(
        [{"tag": f"p3_{u}", "cmd": ["--phase", "p3", "--p3-unit", u, *child_flags]} for u in todo],
        args,
        "p3",
    )
    # aggregate unit docs -> summary.json (same shape as the pre-shard output)
    summary: dict = {"arms": {}, "splithalf_permutation_seed": SPLIT_SEED_1482}
    for u in units:
        doc = json.loads(_p3_unit_json(args, u).read_text())
        if u == "aux":
            summary["prefix_dense_to_vx_r2"] = doc["prefix_dense_to_vx_r2"]
            summary["encode_the_prediction"] = doc["encode_the_prediction"]
            continue
        summary["arms"].setdefault(doc["arm"], {}).update(doc["arm_doc"])
        for k, v in doc.get("scalars", {}).items():
            summary.setdefault(k, v)
    _write_json(pf_dir / "summary.json", summary)
    prep = _p3_prep(args)  # parent-side prep (CPU-only) for the interp digest
    _interp_digest(
        args, prep.parts, prep.f_out, prep.out_counts, prep.n_fit, prep.row_pos, prep.order
    )
    _phase_sentinel("p3", "p3 done (SAE fits + per-feature reads)")


def _interp_digest(args, parts, f_out, out_counts, n_fit, row_pos, order) -> None:
    """Worst-30 / best-10 / random-10 answer-side features -> top-20 activating
    contexts via stored phi_max (digest-only: ci + activation; token offsets are
    resolved off-pod by the analysis script from the same store)."""
    pf_dir = args.out_eval / "sae_perfeature"
    prim = pf_dir / "sae_ctx__mean__ridge.npz"
    if not prim.exists():
        return
    d = np.load(prim)
    r2, feat_ids, act = d["r2"], d["feat_ids"], d["activity"]
    ok = np.isfinite(r2) & (act * n_fit >= args.headline_floor)
    cand = np.where(ok)[0]
    if len(cand) == 0:
        _write_json(
            args.out_eval / "interp_digest.json",
            {"note": "no features above headline floor", "n_candidates": 0},
        )
        return
    order_r2 = cand[np.argsort(r2[cand])]
    worst = order_r2[: min(30, len(order_r2))]
    best = order_r2[-min(10, len(order_r2)) :]
    rng = np.random.default_rng(SPLIT_SEED_1482)
    rand = rng.choice(cand, size=min(10, len(cand)), replace=False)
    sel = {
        "worst": feat_ids[worst].tolist(),
        "best": feat_ids[best].tolist(),
        "random": feat_ids[rand].tolist(),
    }
    # top-20 contexts per selected feature via stored phi_max
    want = set(int(f) for grp in sel.values() for f in grp)
    per_feat: dict[int, list] = {f: [] for f in want}
    for part in parts:
        offs = np.concatenate([[0], np.cumsum(part["idx_off"])])
        for i in range(len(part["row_idx"])):
            sl = slice(offs[i], offs[i + 1])
            fidx = part["ans_idx"][sl].astype(np.int64)
            vals = part["ans_max"][sl].astype(np.float32)
            for f, v in zip(fidx, vals, strict=True):
                if int(f) in want:
                    per_feat[int(f)].append((float(v), int(part["ci"][i])))
    digest = {
        grp: {str(f): sorted(per_feat.get(int(f), []), reverse=True)[:20] for f in fs}
        for grp, fs in sel.items()
    }
    _write_json(
        args.out_eval / "interp_digest.json",
        {
            "selection": sel,
            "top_contexts": digest,
            "r2_worst": r2[worst].tolist(),
            "r2_best": r2[best].tolist(),
            "headline_floor": args.headline_floor,
        },
    )


# ── P4: store upload (detached-concurrent with P3; #825) ────────────────────────


def phase_p4(args) -> None:
    C.phase("p4")
    if args.skip_upload:
        logger.info("[p4] --skip-upload set; enumerating only")
    from explore_persona_space.orchestrate import hub

    prefix = args.hf_prefix + ("_smoke" if args.smoke else "")
    uploads = [
        (args.store, f"{prefix}/analysis_tensors/sae_pooled"),
        (args.out_eval / "percontext", f"{prefix}/analysis_tensors/percontext"),
    ]
    for local, remote in uploads:
        files = sorted(p for p in local.rglob("*") if p.is_file())
        if not files:
            raise RuntimeError(f"[p4] nothing to upload under {local}")
        logger.info("[p4] %d files %s -> %s", len(files), local, remote)
        if args.skip_upload:
            continue
        # whole tree, no eligibility filter (plan-glob parity trivial)
        url = hub._upload(local, C.HF_DATA_REPO, repo_type="dataset", path_in_repo=remote)
        if not url:
            raise RuntimeError(f"[p4] upload returned no path for {local} -> {remote}")
        from huggingface_hub import HfApi

        expected = [f"{remote}/{p.relative_to(local)}" for p in files]
        missing = hub.verify_repo_paths_uploaded(
            HfApi(), C.HF_DATA_REPO, expected, path_in_repo=remote, repo_type="dataset"
        )
        if missing:
            raise RuntimeError(
                f"[p4] upload verify: {len(missing)} missing under {remote}: {sorted(missing)[:5]}"
            )
    # scratch METADATA (~17 MB) — the off-pod phases (P5 judge / P6 analysis,
    # issue1482_analysis.py) load exactly these three files; without this upload they
    # exist only on the pod and a terminated pod strands P5 at launch (epm:failure v6
    # — r6 durable fix; VM-side recovery: scripts/issue1482_reconstruct_scratch.py).
    # X/Y stay pod-local (multi-GB, regenerable via P0); this metadata is the seam.
    scratch_meta = [args.scratch / n for n in ("split_indices.npz", "row_ci.npy", "prov.npy")]
    missing_local = [p.name for p in scratch_meta if not p.exists()]
    if missing_local:
        raise RuntimeError(f"[p4] scratch metadata missing locally: {missing_local}")
    meta_remote = f"{prefix}/analysis_tensors/scratch_meta"
    logger.info("[p4] %d scratch metadata files -> %s", len(scratch_meta), meta_remote)
    if not args.skip_upload:
        from huggingface_hub import HfApi

        for p in scratch_meta:
            url = hub._upload(
                p,
                C.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{meta_remote}/{p.name}",
                upload_as_file=True,
            )
            if not url:
                raise RuntimeError(f"[p4] upload returned no path for {p} -> {meta_remote}")
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            C.HF_DATA_REPO,
            [f"{meta_remote}/{p.name}" for p in scratch_meta],
            path_in_repo=meta_remote,
            repo_type="dataset",
        )
        if missing:
            raise RuntimeError(
                f"[p4] scratch_meta verify: {len(missing)} missing: {sorted(missing)}"
            )
    # split doc rides along (json, non-LFS)
    if not args.skip_upload:
        hub._upload(
            args.out_eval / "split_1482.json",
            C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/split_1482.json",
            upload_as_file=True,
        )
    _phase_sentinel("p4", "p4 done (store uploaded + verified)")


# ── results sentinel + main ─────────────────────────────────────────────────────


def _results_sentinel(args, t_start: float) -> None:
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    recon = json.loads((args.out_eval / "reconciliation.json").read_text())
    fitness = json.loads((args.out_eval / "sae_fitness.json").read_text())
    split = json.loads((args.out_eval / "split_1482.json").read_text())
    gpus = _physical_gpu_ids()
    hours = (time.time() - t_start) / 3600.0 * max(1, len(gpus))
    prefix = args.hf_prefix + ("_smoke" if args.smoke else "")
    payload = {
        "sentinel_schema_version": C.SENTINEL_SCHEMA_VERSION,
        "kind": "epm:results",
        "version": 1,
        "task_id": TASK_ID,
        "by": "issue1482_error_analysis",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "issue-1482 pod phases P0-P4 complete (P5 judge + P6 analysis run off-pod)",
        "eval_numbers": {
            "gate_a": {k: v["refit_r2"] for k, v in recon["per_arm"].items()},
            "gate_a_verdict": recon["verdict"],
            "gate_b": fitness["gate_b"],
            "fve_l19": fitness["layers"][str(LAYER)],
            "chosen_k": fitness["chosen_k"],
            "g2_cos_min": fitness["g2_cos_min"],
        },
        "eval_paths": {
            "split": str(args.out_eval / "split_1482.json"),
            "reconciliation": str(args.out_eval / "reconciliation.json"),
            "percontext": str(args.out_eval / "percontext"),
            "sae_perfeature": str(args.out_eval / "sae_perfeature"),
            "sae_fitness": str(args.out_eval / "sae_fitness.json"),
        },
        "reproducibility_card": {
            **C.reproducibility_metadata(),
            "layer": LAYER,
            "fitter_seed": args.seed,
            "krr_nystrom_centers": args.krr_nystrom_centers,
            "sae_repo": "andyrdt/saes-qwen2.5-7b-instruct",
            "sae_revision": "c37e53c4bb07127ad17ab88f28b93d4e87142e59",
            "split_seed": SPLIT_SEED_1482,
            "holdout_sha256": split["holdout"]["sha256"],
            "sae_fit_sha256": split["sae_fit"]["sha256"],
            "regime": split["regime"],
        },
        "wandb_url": None,  # no training — fits log to JSON checkpoints (plan §10)
        "hf_hub_url": f"https://huggingface.co/datasets/{C.HF_DATA_REPO}/tree/main/{prefix}",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": C.reproducibility_metadata()["git_commit"],
        "gpu_hours_used": round(hours, 2),
        "gpu_hours_budgeted": 12,
        "plan_deviations": split.get("plan_deviations", []),
    }
    path = logs_dir / f"issue-{TASK_ID}-results.json"
    # atomic (tmp + os.replace): a poller read mid-write must never parse garbage
    # (pod-side-reporting convention; the .json.tmp name is outside the poller glob)
    C.write_json_atomic(path, payload)
    logger.info("Wrote results sentinel %s", path)


# ── pdshrink: per-direction ridge-shrinkage control (amendment round, plan v12) ──
# ONE variable vs the parent per-direction decomposition: the linear baseline's
# lambda policy — single global lambda (pooled-val-selected 0.001) -> per-direction
# lambda (per-direction val-R2 argmax on the SAME pinned 400-row val). Everything
# else (matrices, fit rows, basis, holdout rows, fp16-cast scoring convention,
# committed MLP arm) is inherited verbatim. Batched throughout: one
# _ridge_factorize, per-lambda blocked GEMMs, argmax tables, GEMM-batched paired
# bootstrap — no serial per-direction or per-draw loop anywhere.


def _pdshrink_lambda_grid() -> np.ndarray:
    """Plan §11 union grid: scope core logspace(-5,-2,16) ∪ parent LAMBDAS_N1M ∪ {1e-9}.
    Deduped on exact float equality; realized length is RECORDED (expected ~38),
    never hard-asserted."""
    vals = (
        {float(v) for v in np.logspace(-5, -2, 16)} | {float(v) for v in N1M.LAMBDAS_N1M} | {1e-9}
    )
    return np.asarray(sorted(vals), dtype=np.float64)


def _rebuild_pca_basis(Y, tr_full: np.ndarray, block: int, dev) -> tuple[np.ndarray, np.ndarray]:
    """Parent basis recipe VERBATIM (_p3_unit_aux): fp64 streaming covariance of Y over
    train_full, one eigh, top-256 descending. Device-routed (plan §9 S3) with the
    cuSOLVER-non-convergence CPU fallback (gotchas.md); eigenvector signs are
    arbitrary but every downstream read (R2, projections) is sign-invariant and the
    G1 eigval compare is on eigenvalues only."""
    A = torch.zeros((3584, 3584), dtype=torch.float64, device=dev)
    mu_acc = torch.zeros(3584, dtype=torch.float64, device=dev)
    n_acc = 0
    for s in range(0, len(tr_full), block):
        yb = torch.as_tensor(np.asarray(Y[tr_full[s : s + block]], dtype=np.float64), device=dev)
        A += yb.T @ yb
        mu_acc += yb.sum(0)
        n_acc += yb.shape[0]
    mu = mu_acc / n_acc
    A = A / n_acc - torch.outer(mu, mu)
    try:
        evals, evecs = torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:  # cuSOLVER syevd non-convergence -> CPU LAPACK
        logger.warning("[pdshrink] cuda eigh non-convergence; CPU LAPACK fallback")
        evals, evecs = torch.linalg.eigh(A.cpu())
    top = torch.flip(evecs[:, -256:], dims=[1]).cpu().numpy()
    eigvals = torch.flip(evals[-256:], dims=[0]).cpu().numpy()
    return top, eigvals


def _score_holdout_projection(
    ph: np.ndarray, top: np.ndarray, yh_rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Holdout scoring convention (plan §4 pins + Statistics MF-1): predictions cast
    through fp16 (the committed ``holdout_pred16`` convention), projected in fp64,
    the PROJECTION stored fp16, and R2 computed FROM the stored fp16 values — so the
    persisted tuned store reproduces hold_tab EXACTLY (MF-1's <=1e-6 runtime assert).
    Deliberate deviation from the plan §4 sketch (which scored the fp64 projection
    and stored fp16 — mutually inconsistent at 1e-6): the fp16 projection
    quantization moves per-direction R2 by ~1e-6, three orders below the 1e-3 gate
    scales, and makes store<->table consistency exact."""
    proj16 = (ph.astype(np.float16).astype(np.float64) @ top).astype(np.float16)
    r2 = _per_feature_metrics(proj16.astype(np.float64), yh_rot)["r2"]
    return proj16, r2


def _gather_tuned(proj_tabs: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """MF-1 tuned-arm gather: column d at lambda=sel[d] from the retained per-lambda
    projected holdout tables — NEVER the parent-lambda slice (the plan-sketch slip:
    E_tuned would silently run on global-lambda residuals and manufacture a
    Confirmed verdict). One vectorized take_along_axis gather, no per-direction loop."""
    idx = np.broadcast_to(np.asarray(sel, np.int64)[None, None, :], (1,) + proj_tabs.shape[1:])
    return np.take_along_axis(proj_tabs, idx, axis=0)[0]


def _spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho of two 1-D vectors via the file's batched midrank helper."""
    r = _midrank(np.stack([np.asarray(a, np.float64), np.asarray(b, np.float64)], axis=1))
    ra = r[:, 0] - r[:, 0].mean()
    rb = r[:, 1] - r[:, 1].mean()
    den = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else float("nan")


def _pdshrink_gates_pass(gates: dict) -> bool:
    """Pure G1 verdict (plan §7): (i) split shas byte-equal; (ii) rebuilt eigvals
    median rel |Δ| <= 1e-3; (iii) refit-at-parent-lambda ridge per-direction R2
    median |Δ| <= 1e-3 AND Spearman rho >= 0.999; (iv) same-convention MLP re-read
    median |Δ| <= 1e-3. NaN comparisons are False -> a non-finite gate field FAILS."""
    return bool(
        gates["split_sha_match"]
        and gates["eigval_med_rel"] <= 1e-3
        and gates["ridge_med_abs_delta"] <= 1e-3
        and gates["ridge_spearman"] >= 0.999
        and gates["mlp_med_abs_delta"] <= 1e-3
    )


def _pdshrink_gate_halt(gates: dict, out_dir: Path, smoke: bool) -> dict:
    """K1: write gates.json FIRST, then HALT rc=23 on a production miss — no tuned
    read on a failed gate. Smoke computes the identical fields but demotes the
    verdict to informational (split/regime differ from the committed production
    references by design — gate_a smoke_demoted convention)."""
    ok = _pdshrink_gates_pass(gates)
    doc = dict(gates)
    doc["verdict"] = "PASS" if ok else ("SMOKE_DEMOTED_FAIL" if smoke else "HALT")
    doc["smoke_demoted"] = bool(smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "gates.json", doc)
    logger.info("[pdshrink] G1 verdict=%s", doc["verdict"])
    if not ok and not smoke:
        _phase_sentinel(
            "pdshrink-g1-halt",
            "G1 reconciliation HALT (no tuned read taken)",
            {"gates": {k: v for k, v in doc.items() if isinstance(v, (bool, int, float, str))}},
        )
        raise SystemExit(RC_G1_PDSHRINK)
    return doc


def _pdshrink_stage_mlp_npz(
    args, hold: np.ndarray, standin_pred: np.ndarray | None
) -> tuple[Path, str]:
    """Resolve the reused MLP holdout-prediction npz (plan §3 row coverage): local
    out_eval/percontext first (VM/pod-synced copy), else the Hub-verified data-repo
    copy (§10 row 2 — git-untracked locally, so a fresh pod fetches from HF via the
    #1402 atomic fail-loud helper). Smoke: a LABELED stand-in built from the
    parent-lambda ridge predictions + deterministic noise is written to the SAME
    local path, so the identical resolve/keys/row-align/gate/bootstrap code runs
    downstream with no smoke branches."""
    local = args.out_eval / "percontext" / PDSHRINK_MLP_NPZ
    if local.exists():
        return local, "local"
    if args.smoke:
        assert standin_pred is not None
        local.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(SPLIT_SEED_1482)
        standin = standin_pred + 0.05 * rng.standard_normal(standin_pred.shape)
        np.savez(
            local,
            holdout_rows=np.asarray(hold, np.int64),
            holdout_pred16=standin.astype(np.float16),
        )
        logger.info("[pdshrink] SMOKE stand-in MLP npz (parent-lambda ridge + noise): %s", local)
        return local, "smoke-standin"
    from explore_persona_space.orchestrate import hub

    remote = f"{args.hf_prefix}/analysis_tensors/percontext/{PDSHRINK_MLP_NPZ}"
    logger.info("[pdshrink] staging reused MLP npz from HF: %s", remote)
    return hub.stage_hub_file(C.HF_DATA_REPO, remote, local, repo_type="dataset"), "hf"


def _pdshrink_bootstrap(
    E_arms: dict[str, np.ndarray],
    T: np.ndarray,
    n_boot: int,
    seed: int,
    dev,
    chunk: int = 2000,
) -> dict:
    """10k-draw paired context bootstrap batched as GEMMs (plan §6): resample the
    holdout contexts with replacement (multinomial count vectors W per draw); every
    draw's per-direction residual/total sums are W @ {E_arm, T} — no per-draw Python
    loop. Direction means frozen at full-holdout values (T uses the full-holdout
    mean). Returns per-contrast Delta draws (gap = R2_mlp − R2_arm, band-averaged,
    Delta = G(129-256) − G(1-16)) + the tuned contrast's per-band gap draws."""
    n = T.shape[0]
    rng = np.random.default_rng(seed)
    Et = {
        k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32, device=dev)
        for k, v in E_arms.items()
    }
    Tt = torch.as_tensor(np.ascontiguousarray(T), dtype=torch.float32, device=dev)
    deltas: dict[str, list[np.ndarray]] = {"tuned": [], "oracle": []}
    band_draws: dict[str, list[np.ndarray]] = {b: [] for b in PDSHRINK_BANDS}
    pvals = np.full(n, 1.0 / n)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        W = rng.multinomial(n, pvals, size=m).astype(np.float32)
        Wt = torch.as_tensor(W, device=dev)
        ST = torch.clamp(Wt @ Tt, min=1e-12)  # (m, 256)
        SE = {k: Wt @ E for k, E in Et.items()}
        for contrast in ("tuned", "oracle"):
            g = (SE[contrast] - SE["mlp"]) / ST  # per-draw per-direction gap
            gb = {b: g[:, a:z].mean(1) for b, (a, z) in PDSHRINK_BANDS.items()}
            deltas[contrast].append((gb["129-256"] - gb["1-16"]).cpu().numpy())
            if contrast == "tuned":
                for b in PDSHRINK_BANDS:
                    band_draws[b].append(gb[b].cpu().numpy())
    return {
        "delta_tuned": np.concatenate(deltas["tuned"]),
        "delta_oracle": np.concatenate(deltas["oracle"]),
        "tuned_band_gap_draws": {b: np.concatenate(v) for b, v in band_draws.items()},
    }


def _band_means(v: np.ndarray) -> dict[str, float]:
    return {b: float(np.nanmean(v[a:z])) for b, (a, z) in PDSHRINK_BANDS.items()}


def _pdshrink_figures(fig_dir: Path, d: dict) -> list[Path]:
    """Hero (3-curve per-direction R2 vs rank + band-gap inset), 256-point gap
    scatter colored by band, and the D1-D3 + bootstrap diagnostics panel — all via
    savefig_paper sidecars. Plain-English labels only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    pal = paper_palette(6)
    ranks = np.arange(1, 257)
    bands = list(PDSHRINK_BANDS.items())

    # hero: three curves + band-gap inset (committed vs tuned, tuned with 95% CI)
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(
        ranks,
        d["r2_ridge_committed"],
        color=pal[0],
        lw=1.5,
        label="ridge, one shared regularizer (committed)",
    )
    ax.plot(
        ranks,
        d["r2_tuned"],
        color=pal[1],
        lw=1.5,
        label="ridge, per-direction regularizer (this round)",
    )
    ax.plot(ranks, d["r2_mlp_committed"], color=pal[2], lw=1.5, label="MLP (committed)")
    ax.set_xlabel("answer-PCA direction rank (by training variance)")
    ax.set_ylabel("held-out per-direction R²")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ins = ax.inset_axes([0.44, 0.58, 0.53, 0.38])
    xb = np.arange(len(bands))
    gap_c = [d["band_gaps_committed"][b] for b, _ in bands]
    gap_t = [d["band_gaps_tuned"][b] for b, _ in bands]
    ci = np.asarray([d["tuned_band_gap_ci"][b] for b, _ in bands])  # (4, 2)
    yerr = np.vstack(
        [
            np.maximum(0.0, np.asarray(gap_t) - ci[:, 0]),
            np.maximum(0.0, ci[:, 1] - np.asarray(gap_t)),
        ]
    )  # element-wise clamp — errorbar offsets must be non-negative (gotchas.md)
    ins.bar(xb - 0.2, gap_c, width=0.38, color=pal[0], label="shared regularizer")
    ins.bar(xb + 0.2, gap_t, width=0.38, color=pal[1], yerr=yerr, capsize=2, label="per-direction")
    ins.set_xticks(xb)
    ins.set_xticklabels([b for b, _ in bands], fontsize=7)
    ins.set_ylabel("MLP − ridge gap", fontsize=7)
    ins.tick_params(labelsize=7)
    ins.legend(frameon=False, fontsize=6, loc="upper left")
    savefig_paper(fig, "pdshrink_hero_perdirection_r2", dir=fig_dir)
    plt.close(fig)

    # low-level per-unit plot: 256-point gap scatter, colored by band, identity line
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for (b, (a, z)), c in zip(bands, pal, strict=False):
        ax.scatter(
            d["g_committed"][a:z], d["g_tuned"][a:z], s=16, color=c, label=f"ranks {b}", alpha=0.8
        )
    lims = [
        float(min(np.nanmin(d["g_committed"]), np.nanmin(d["g_tuned"]))),
        float(max(np.nanmax(d["g_committed"]), np.nanmax(d["g_tuned"]))),
    ]
    ax.plot(lims, lims, color="0.6", lw=1.0, ls="--", label="no change (identity)")
    ax.set_xlabel("MLP − ridge gap under the shared regularizer (committed)")
    ax.set_ylabel("MLP − ridge gap under per-direction regularizers")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "pdshrink_gap_scatter", dir=fig_dir)
    plt.close(fig)

    # diagnostics: D1 selected lambda vs rank; D3 shrinkage spectrum; bootstrap
    # Delta distribution; per-direction R2 change vs eigenvalue
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.6))
    a0, a1, a2, a3 = axs.ravel()
    a0.scatter(ranks, np.log10(d["lam_sel"]), s=10, color=pal[1], label="selected on validation")
    a0.scatter(
        ranks, np.log10(d["lam_oracle"]), s=10, color=pal[3], alpha=0.5, label="holdout-oracle"
    )
    a0.axhline(np.log10(d["lam_parent"]), color="0.4", lw=1.0, ls="--", label="parent shared value")
    a0.set_xlabel("direction rank")
    a0.set_ylabel("log10(selected regularizer strength)")
    a0.legend(frameon=False, fontsize=8)
    shrink = np.sort(d["lam_parent"] / (d["s_eig"] + d["lam_parent"]))[::-1]
    a1.semilogy(np.arange(1, len(shrink) + 1), np.maximum(shrink, 1e-300), color=pal[0], lw=1.4)
    a1.set_xlabel("Gram eigendirection (sorted)")
    a1.set_ylabel("shrinkage applied at the parent value")
    a2.hist(d["delta_draws"], bins=60, color=pal[1])
    a2.axvline(0.0, color="0.3", lw=1.0)
    a2.axvline(d["delta_tuned"], color=pal[2], lw=1.2, label="observed excess gap")
    for v in d["delta_ci"]:
        a2.axvline(v, color=pal[2], lw=0.9, ls="--")
    a2.set_xlabel("excess low-variance gap (bootstrap draws)")
    a2.set_ylabel("draw count")
    a2.legend(frameon=False, fontsize=8)
    a3.semilogx(
        np.maximum(d["eigvals"], 1e-300),
        d["r2_tuned"] - d["r2_ridge_committed"],
        ".",
        color=pal[1],
        ms=5,
    )
    a3.axhline(0.0, color="0.4", lw=1.0)
    a3.set_xlabel("direction training variance (eigenvalue)")
    a3.set_ylabel("per-direction R² change (tuned − shared)")
    fig.tight_layout()
    savefig_paper(fig, "pdshrink_diagnostics", dir=fig_dir)
    plt.close(fig)
    return sorted(fig_dir.glob("pdshrink_*"))


def _pdshrink_upload(args, out: Path) -> None:
    """ONE folder upload of the pdshrink out dir (npz + JSONs + copied figures) to
    the data repo BEFORE teardown, exact-set verified (phase_p4 pattern). Whole
    tree, no eligibility filter — plan-glob parity trivial."""
    from explore_persona_space.orchestrate import hub

    prefix = args.hf_prefix + ("_smoke" if args.smoke else "")
    remote = f"{prefix}/analysis_tensors/pdshrink"
    files = sorted(p for p in out.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"[pdshrink] nothing to upload under {out}")
    logger.info("[pdshrink] %d files %s -> %s", len(files), out, remote)
    if args.skip_upload:
        logger.info("[pdshrink] --skip-upload set; enumeration only")
        return
    url = hub._upload(out, C.HF_DATA_REPO, repo_type="dataset", path_in_repo=remote)
    if not url:
        raise RuntimeError(f"[pdshrink] upload returned no path for {out} -> {remote}")
    from huggingface_hub import HfApi

    expected = [f"{remote}/{p.relative_to(out)}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), C.HF_DATA_REPO, expected, path_in_repo=remote, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(
            f"[pdshrink] upload verify: {len(missing)} missing under {remote}: "
            f"{sorted(missing)[:5]}"
        )


def _pdshrink_results_sentinel(args, t_start: float, summary: dict, gates: dict) -> None:
    """Terminal results sentinel for the pdshrink workload (poller drains it into
    epm:results; pod-side contract — never task.py)."""
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    gpus = _physical_gpu_ids()
    hours = (time.time() - t_start) / 3600.0 * max(1, len(gpus))
    prefix = args.hf_prefix + ("_smoke" if args.smoke else "")
    payload = {
        "sentinel_schema_version": C.SENTINEL_SCHEMA_VERSION,
        "kind": "epm:results",
        "version": 1,
        "task_id": TASK_ID,
        "by": "issue1482_error_analysis/pdshrink",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "pdshrink (per-direction ridge-shrinkage control) complete: "
            f"verdict={summary['verdict']}"
        ),
        "eval_numbers": {
            "g1_verdict": gates["verdict"],
            "delta_tuned": summary["delta"]["tuned_headline_committed_mlp"],
            "delta_ci95": summary["delta"]["ci95"],
            "delta_oracle": summary["delta"]["oracle"]["point"],
            "delta_core_grid_only": summary["delta"]["core_grid_only"],
            "verdict": summary["verdict"],
            "precision_limited": summary["precision_limited"],
        },
        "eval_paths": {
            "summary": str(args.out_eval / PDSHRINK_SUBDIR / "pdshrink_summary.json"),
            "gates": str(args.out_eval / PDSHRINK_SUBDIR / "gates.json"),
            "lam_tables": str(args.out_eval / PDSHRINK_SUBDIR / "lam_tables.npz"),
            "tuned_projected": str(args.out_eval / PDSHRINK_SUBDIR / "tuned_holdout_projected.npz"),
        },
        "reproducibility_card": {
            **C.reproducibility_metadata(),
            "layer": LAYER,
            "fitter_seed": args.seed,
            "split_seed": SPLIT_SEED_1482,
            "bootstrap_seed": args.bootstrap_seed,
            "n_boot": args.n_boot,
            "lambda_grid_realized_len": summary["lambda_grid"]["realized_len"],
            "regime": summary["regime"],
        },
        "wandb_url": None,  # no training — closed-form refits log to JSON (plan §10)
        "hf_hub_url": (
            f"https://huggingface.co/datasets/{C.HF_DATA_REPO}/tree/main/"
            f"{prefix}/analysis_tensors/pdshrink"
        ),
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": C.reproducibility_metadata()["git_commit"],
        "gpu_hours_used": round(hours, 2),
        "gpu_hours_budgeted": 3,
        "plan_deviations": [],
    }
    path = logs_dir / f"issue-{TASK_ID}-results.json"
    C.write_json_atomic(path, payload)
    logger.info("Wrote results sentinel %s", path)


def phase_pdshrink(args) -> None:
    """Per-direction ridge-shrinkage control (plan v12 §4): ONE factorization,
    per-lambda blocked GEMMs over the 38-value union grid, per-direction val-argmax
    selection, MF-1 tuned gather + store-consistency assert, G1 HALT gate
    (gates.json first, rc=23), GEMM-batched paired bootstrap, D1-D3 diagnostics,
    figures, HF upload, results sentinel."""
    C.phase("pdshrink")
    t0 = time.time()
    out = args.out_eval / PDSHRINK_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    grid = _pdshrink_lambda_grid()
    regime = {
        "smoke": bool(args.smoke),
        "max_chunks": args.max_chunks,
        "holdout_n": args.holdout_n,
        "fit_n": args.fit_n,
        "n_boot": args.n_boot,
        "bootstrap_seed": args.bootstrap_seed,
        "device": args.device,
        "n_lambda_realized": int(len(grid)),
    }
    summary_path = out / "pdshrink_summary.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text()).get("regime")
        if prior == regime:
            logger.info("[pdshrink] resume: summary present under matching regime; skip")
            return
        raise RuntimeError(
            f"[pdshrink] out dir holds a run under a DIFFERENT regime: {prior} != {regime}"
        )
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
    X, Y, idx = _load_xy(args)
    tr = _train_rows_for("refit_holdout", idx, args)
    val, hold = idx["val"], idx["holdout"]
    dev = torch.device(args.device)
    block = N1M.RIDGE_BLOCK

    # committed references (the G1 anchors + the untouched MLP arm)
    committed_pca = json.loads(COMMITTED_PERDIR_PCA.read_text())
    ridge_meta = json.loads(COMMITTED_RIDGE_JSON.read_text())["fit_meta"]
    lam_parent = float(ridge_meta["selected_lambda"])
    assert lam_parent == 0.001, lam_parent  # plan §4 in-run assert (assumption 3)
    r2_ridge_committed = np.asarray(committed_pca["per_direction_r2"]["ridge"], np.float64)
    r2_mlp_committed = np.asarray(committed_pca["per_direction_r2"]["mlp_w8192"], np.float64)
    eig_committed = np.asarray(committed_pca["eigvals_top"], np.float64)

    # (a) basis rebuild — parent recipe verbatim
    logger.info(
        "[pdshrink] rebuilding top-256 PCA basis over %d train_full rows",
        len(idx["train_full"]),
    )
    top, eigvals = _rebuild_pca_basis(Y, idx["train_full"], block, dev)
    yh_rot = np.asarray(Y[hold], np.float64) @ top
    yv_rot = np.asarray(Y[val], np.float64) @ top

    # (b) ONE factorization; parent internals byte-untouched
    fac = N1M._ridge_factorize(X, Y, tr, dev, block)
    s_eig = fac["s_eig"].cpu().numpy()

    # (c) G1 reconciliation HALT gate (plan §7; gates.json FIRST, rc=23 on miss)
    pred_h_parent = N1M._ridge_predict_one(X, hold, fac, lam_parent, dev, block)
    r2_repro = _per_feature_metrics(
        pred_h_parent.astype(np.float16).astype(np.float64) @ top, yh_rot
    )["r2"]
    # Leg (i): compare the REGENERATED doc against PINNED committed sha literals — never a
    # disk read of COMMITTED_SPLIT_1482, which production p0 overwrites in place (self-compare,
    # concern g1-split-sha-leg-self-compare).
    regen_split = json.loads((args.out_eval / "split_1482.json").read_text())

    def _dig(doc: dict, path: tuple):
        for k in path:
            doc = doc.get(k) if isinstance(doc, dict) else None
            if doc is None:
                return None
        return doc

    sha_fields: dict[str, tuple] = {
        "train_full_sha256": ("train_full_sha256",),
        "holdout_sha256": ("holdout", "sha256"),
        "pinned_val_sha256": ("pinned_val_sha256",),
        "pinned_test_sha256": ("pinned_test_sha256",),
    }
    sha_compare = {
        name: {
            "regenerated": _dig(regen_split, p),
            "committed": PDSHRINK_COMMITTED_SPLIT_SHAS[name],
        }
        for name, p in sha_fields.items()
    }
    split_sha_match = all(
        v["regenerated"] is not None and v["regenerated"] == v["committed"]
        for v in sha_compare.values()
    )
    mlp_path, mlp_source = _pdshrink_stage_mlp_npz(args, hold, standin_pred=pred_h_parent)
    with np.load(mlp_path) as z:
        missing_keys = {"holdout_pred16", "holdout_rows"} - set(z.files)
        if missing_keys:  # artifact-reuse check (c): realized keys via the consumer's own loader
            raise RuntimeError(
                f"[pdshrink] reused MLP npz missing keys: {sorted(missing_keys)} ({mlp_path})"
            )
        rows_h = np.asarray(z["holdout_rows"], np.int64)
        mlp_pred16 = np.asarray(z["holdout_pred16"])
    assert np.array_equal(rows_h, np.asarray(hold, np.int64)), (
        "MLP npz holdout_rows != regenerated holdout rows (row-coverage alignment, plan §3)"
    )
    mlp_proj64 = mlp_pred16.astype(np.float64) @ top  # committed fp64-projection convention
    r2_mlp_reread = _per_feature_metrics(mlp_proj64, yh_rot)["r2"]
    eig_rel = np.abs(eigvals - eig_committed) / np.maximum(np.abs(eig_committed), 1e-300)
    gates = {
        "gate": "G1-pdshrink",
        "split_sha_match": bool(split_sha_match),
        "split_sha_compare": sha_compare,
        "eigval_med_rel": float(np.nanmedian(eig_rel)),
        "ridge_med_abs_delta": float(np.nanmedian(np.abs(r2_repro - r2_ridge_committed))),
        "ridge_spearman": _spearman_1d(r2_repro, r2_ridge_committed),
        "mlp_med_abs_delta": float(np.nanmedian(np.abs(r2_mlp_reread - r2_mlp_committed))),
        "thresholds": {
            "split_sha_match": True,
            "eigval_med_rel": 1e-3,
            "ridge_med_abs_delta": 1e-3,
            "ridge_spearman": 0.999,
            "mlp_med_abs_delta": 1e-3,
        },
        "lambda_parent": lam_parent,
        "lambda_grid_realized_len": int(len(grid)),
        "n_finite": {
            "ridge_repro": int(np.isfinite(r2_repro).sum()),
            "mlp_reread": int(np.isfinite(r2_mlp_reread).sum()),
            "eigval_rel": int(np.isfinite(eig_rel).sum()),
        },
        "reused_artifact_keys": {
            "artifact": str(mlp_path),
            "keys_verified": ["holdout_pred16", "holdout_rows"],
            "rows_aligned": True,
            "sha256": hashlib.sha256(mlp_path.read_bytes()).hexdigest(),
            "bytes": int(mlp_path.stat().st_size),
            "source": mlp_source,  # local | hf | smoke-standin (actual resolve provenance)
        },
    }
    gates = _pdshrink_gate_halt(gates, out, args.smoke)

    # (d) batched lambda-grid scoring: len(grid) x {val fp64, holdout fp16-cast};
    # NO per-direction loop — selection is an argmax over the (L, 256) val table.
    n_hold = len(hold)
    n_lambda = len(grid)
    val_tab = np.full((n_lambda, 256), np.nan)
    hold_tab = np.full((n_lambda, 256), np.nan)
    proj_tabs = np.empty((n_lambda, n_hold, 256), dtype=np.float16)
    i_parent = int(np.argmin(np.abs(grid - lam_parent)))
    assert np.isclose(grid[i_parent], lam_parent, rtol=1e-12, atol=0.0), (
        grid[i_parent],
        lam_parent,
    )
    hold_tab_fp64_parent = np.full(256, np.nan)
    for i, lam in enumerate(grid):
        pv = N1M._ridge_predict_one(X, val, fac, float(lam), dev, block)
        ph = N1M._ridge_predict_one(X, hold, fac, float(lam), dev, block)
        val_tab[i] = _per_feature_metrics(pv @ top, yv_rot)["r2"]  # selection: fp64 (parent)
        proj_tabs[i], hold_tab[i] = _score_holdout_projection(ph, top, yh_rot)
        if i == i_parent:  # fp16-floor diagnostic: one fp64-scoring row at lambda_parent (§6)
            hold_tab_fp64_parent = _per_feature_metrics(ph @ top, yh_rot)["r2"]
        if (i + 1) % 8 == 0 or i == n_lambda - 1:
            logger.info("[pdshrink] lambda grid %d/%d scored", i + 1, n_lambda)

    arange = np.arange(256)
    val_sel_tab = np.where(np.isfinite(val_tab), val_tab, -np.inf)  # NaN never wins selection
    sel = val_sel_tab.argmax(0)
    r2_tuned = hold_tab[sel, arange]
    tuned_proj = _gather_tuned(proj_tabs, sel)
    # MF-1 runtime consistency assert (mirrored in tests/test_issue1482_pdshrink.py):
    # per-direction R2 recomputed from the PERSISTED tuned projections must equal
    # hold_tab[sel, arange] elementwise <= 1e-6; on miss: fail loud, no read.
    r2_check = _per_feature_metrics(tuned_proj.astype(np.float64), yh_rot)["r2"]
    mf1_max = float(np.nanmax(np.abs(r2_check - r2_tuned)))
    assert mf1_max <= 1e-6, (
        f"MF-1 tuned-store consistency: max |R2(persisted) - hold_tab[sel]| = {mf1_max}"
    )
    # holdout-ORACLE selection (critic bracket): ceiling of ANY grid-linear baseline
    hold_sel_tab = np.where(np.isfinite(hold_tab), hold_tab, -np.inf)
    sel_oracle = hold_sel_tab.argmax(0)
    r2_oracle = hold_tab[sel_oracle, arange]
    oracle_proj = _gather_tuned(proj_tabs, sel_oracle)

    np.savez(
        out / "lam_tables.npz",  # BOTH tables + sel land (oracle-read support, plan note 6)
        lam_grid=grid,
        val_tab=val_tab,
        hold_tab=hold_tab,
        sel=sel.astype(np.int64),
        sel_oracle=sel_oracle.astype(np.int64),
        i_parent=np.int64(i_parent),
        hold_tab_fp64_at_parent=hold_tab_fp64_parent,
        eigvals_top=eigvals,
        r2_ridge_repro_at_parent=r2_repro,
        r2_mlp_reread=r2_mlp_reread,
    )
    np.savez(
        out / "tuned_holdout_projected.npz",  # the §3 row-coverage artifact (MF-1 store)
        tuned_proj16=tuned_proj,
        holdout_rows=np.asarray(hold, np.int64),
        sel=sel.astype(np.int64),
        lam_grid=grid,
        convention=np.asarray(
            "top-256 train-covariance PCA basis (parent _p3_unit_aux recipe); columns are "
            "projections of fp16-cast holdout ridge predictions at lambda=lam_grid[sel[d]], "
            "stored fp16; R2 vs (Y[holdout] @ basis) reproduces hold_tab[sel] exactly"
        ),
    )

    # (e) stats: band gaps, Delta (+oracle bracket, +core-grid sensitivity), bootstrap
    g_committed = r2_mlp_committed - r2_ridge_committed
    g_tuned = r2_mlp_committed - r2_tuned  # headline: committed MLP arm untouched
    g_oracle = r2_mlp_committed - r2_oracle
    bm_committed = _band_means(g_committed)
    bm_tuned = _band_means(g_tuned)
    bm_oracle = _band_means(g_oracle)
    delta_committed_ref = bm_committed["129-256"] - bm_committed["1-16"]  # unrounded 0.1021
    delta_tuned = bm_tuned["129-256"] - bm_tuned["1-16"]
    delta_oracle = bm_oracle["129-256"] - bm_oracle["1-16"]
    # D2 grid-restriction sensitivity: argmax over the 16-point scope-core grid only
    core_rows = np.asarray(
        [int(np.where(grid == float(v))[0][0]) for v in np.logspace(-5, -2, 16)], np.int64
    )
    sel_core = core_rows[
        np.where(np.isfinite(val_tab[core_rows]), val_tab[core_rows], -np.inf).argmax(0)
    ]
    g_core = r2_mlp_committed - hold_tab[sel_core, arange]
    bm_core = _band_means(g_core)
    delta_core = bm_core["129-256"] - bm_core["1-16"]
    # bootstrap inputs: per-context per-direction residual^2 + totals (means frozen)
    E_mlp = (yh_rot - mlp_proj64) ** 2
    E_tuned = (yh_rot - tuned_proj.astype(np.float64)) ** 2
    E_oracle = (yh_rot - oracle_proj.astype(np.float64)) ** 2
    T = (yh_rot - yh_rot.mean(0)) ** 2
    r2_from_e = 1.0 - E_tuned.sum(0) / np.maximum(T.sum(0), 1e-12)
    assert float(np.nanmax(np.abs(r2_from_e - r2_tuned))) <= 1e-9  # residual identity
    boot = _pdshrink_bootstrap(
        {"mlp": E_mlp, "tuned": E_tuned, "oracle": E_oracle},
        T,
        args.n_boot,
        args.bootstrap_seed,
        dev,
    )
    lo, hi = (float(v) for v in np.percentile(boot["delta_tuned"], [2.5, 97.5]))
    lo_o, hi_o = (float(v) for v in np.percentile(boot["delta_oracle"], [2.5, 97.5]))
    half_width = (hi - lo) / 2.0
    # verdict lattice (plan §3, disjoint + exhaustive)
    if delta_tuned > 0 and lo > 0:
        verdict = "Confirmed"
    elif hi < 0 or (lo <= 0 <= hi):
        verdict = "Falsified"
    else:
        verdict = "Inconclusive"
    precision_limited = bool(half_width > 0.05)  # §3 precision guard (prose, not lattice)
    tuned_band_ci = {
        b: [float(v) for v in np.percentile(draws, [2.5, 97.5])]
        for b, draws in boot["tuned_band_gap_draws"].items()
    }
    # D1 selection profile + val-vs-holdout selected-lambda agreement
    lam_sel = grid[sel]
    lam_oracle = grid[sel_oracle]
    d1 = {
        "n_moved_off_parent": int((sel != i_parent).sum()),
        "n_heavier_than_parent": int((lam_sel > lam_parent).sum()),
        "n_lighter_than_parent": int((lam_sel < lam_parent).sum()),
        "frac_sel_eq_oracle": float((sel == sel_oracle).mean()),
        "med_abs_log10_lambda_gap_sel_vs_oracle": float(
            np.median(np.abs(np.log10(lam_sel) - np.log10(lam_oracle)))
        ),
    }
    # D3 OLS-proximity bound off the factorization spectrum
    s_min = float(s_eig.min())
    shrink = s_eig / (s_eig + lam_parent)
    d3 = {
        "s_min": s_min,
        "s_max": float(s_eig.max()),
        "ols_proximity_bound_lam_over_smin_plus_lam": float(lam_parent / (s_min + lam_parent)),
        "shrink_factor_percentiles": {
            str(q): float(np.percentile(shrink, q)) for q in (0, 1, 5, 25, 50, 75, 100)
        },
    }
    fp16_floor = {
        b: {
            "mean_abs_r2_diff": float(
                np.nanmean(np.abs(hold_tab[i_parent] - hold_tab_fp64_parent)[a:z])
            ),
            "mean_signed_r2_diff": float(
                np.nanmean((hold_tab_fp64_parent - hold_tab[i_parent])[a:z])
            ),
        }
        for b, (a, z) in PDSHRINK_BANDS.items()
    }
    summary = {
        "verdict": verdict,
        "precision_limited": precision_limited,
        "delta": {
            "tuned_headline_committed_mlp": float(delta_tuned),
            "tuned_boot_center_recomputed_mlp": float(np.mean(boot["delta_tuned"])),
            "ci95": [lo, hi],
            "ci95_half_width": float(half_width),
            "oracle": {"point": float(delta_oracle), "ci95": [lo_o, hi_o]},
            "core_grid_only": float(delta_core),
            "committed_reference_unrounded": float(delta_committed_ref),
        },
        "band_gaps": {"committed": bm_committed, "tuned": bm_tuned, "oracle": bm_oracle},
        "band_r2": {
            "ridge_committed": _band_means(r2_ridge_committed),
            "ridge_tuned": _band_means(r2_tuned),
            "ridge_oracle": _band_means(r2_oracle),
            "mlp_committed": _band_means(r2_mlp_committed),
        },
        "selection": d1,
        "diagnostics": {
            "d3": d3,
            "fp16_floor_per_band_at_parent_lambda": fp16_floor,
            "mf1_store_consistency_max_abs": mf1_max,
        },
        "lambda_grid": {
            "realized_len": int(len(grid)),
            "min": float(grid[0]),
            "max": float(grid[-1]),
            "parent_index": i_parent,
        },
        "bootstrap": {
            "n_boot": int(args.n_boot),
            "seed": int(args.bootstrap_seed),
            "paired_over_contexts": True,
            "direction_means_frozen_at_full_holdout": True,
        },
        "regime": regime,
        "notes": [
            "gap g_d = committed MLP per-direction R2 (untouched) minus the per-direction "
            "val-tuned ridge R2; Delta = G(129-256) - G(1-16)",
            "bootstrap draws recompute BOTH arms from persisted per-context residuals; the "
            "draw center uses the re-read MLP R2 (G1 (iv) bounds |reread - committed| "
            "median <= 1e-3)",
            "holdout scoring casts predictions AND stored projections through fp16 "
            "(committed holdout_pred16 convention + MF-1 store consistency); val "
            "selection stays fp64 (parent selection convention)",
        ],
    }
    _write_json(summary_path, summary)
    logger.info(
        "[pdshrink] verdict=%s delta_tuned=%.4f ci=[%.4f, %.4f] oracle=%.4f (ref %.4f)",
        verdict,
        delta_tuned,
        lo,
        hi,
        delta_oracle,
        delta_committed_ref,
    )
    fig_files = _pdshrink_figures(
        args.figures,
        {
            "r2_ridge_committed": r2_ridge_committed,
            "r2_tuned": r2_tuned,
            "r2_mlp_committed": r2_mlp_committed,
            "g_committed": g_committed,
            "g_tuned": g_tuned,
            "band_gaps_committed": bm_committed,
            "band_gaps_tuned": bm_tuned,
            "tuned_band_gap_ci": tuned_band_ci,
            "lam_sel": lam_sel,
            "lam_oracle": lam_oracle,
            "lam_parent": lam_parent,
            "s_eig": s_eig,
            "eigvals": eigvals,
            "delta_draws": boot["delta_tuned"],
            "delta_tuned": delta_tuned,
            "delta_ci": (lo, hi),
        },
    )
    fig_copy_dir = out / "figures"
    fig_copy_dir.mkdir(parents=True, exist_ok=True)
    for f in fig_files:  # ride the single folder upload (never a per-file upload loop)
        shutil.copy2(f, fig_copy_dir / f.name)
    _pdshrink_upload(args, out)
    _pdshrink_results_sentinel(args, t0, summary, gates)
    _phase_sentinel(
        "pdshrink",
        f"pdshrink done verdict={verdict} delta_tuned={delta_tuned:.4f}",
        {"g1": gates["verdict"]},
    )
    C.phase("done")


def _child_flags(args) -> list[str]:
    flags = [
        "--device",
        args.device,
        "--scratch",
        str(args.scratch),
        "--out-eval",
        str(args.out_eval),
        "--store",
        str(args.store),
        "--sae-dir",
        str(args.sae_dir),
        "--seed",
        str(args.seed),
        "--krr-nystrom-centers",
        str(args.krr_nystrom_centers),
        "--fit-n",
        str(args.fit_n),
        "--max-chunks",
        str(args.max_chunks),
        "--gen-batch",
        str(args.gen_batch),
        "--sae-k",
        str(args.sae_k),
        "--headline-floor",
        str(args.headline_floor),
        "--hf-prefix",
        args.hf_prefix,
        "--max-features-in",
        str(args.max_features_in),
        "--max-features-out",
        str(args.max_features_out),
    ]
    if args.smoke:
        flags.append("--smoke")
    if args.tiny_model:
        flags.append("--tiny-model")
    return flags


def main() -> int:  # noqa: C901 — linear phase dispatcher (readability over splitting)
    ap = argparse.ArgumentParser(description="Issue #1482 error-analysis driver (P0-P4).")
    ap.add_argument(
        "--phase",
        default="all",
        choices=["all", "p0", "p1", "p1-fit", "p2", "p2-worker", "p3", "p4", "pdshrink"],
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--full", action="store_true", help="explicit production mode (default)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out-eval", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--sae-dir", type=Path, default=None)
    ap.add_argument("--max-chunks", type=int, default=None, help="0 = all (production)")
    ap.add_argument("--holdout-n", type=int, default=None)
    ap.add_argument("--sae-n", type=int, default=None)
    ap.add_argument("--sae-val-n", type=int, default=None)
    ap.add_argument("--fit-n", type=int, default=None, help="0 = full train pools")
    ap.add_argument("--n-boot", type=int, default=None, help="pdshrink bootstrap draws")
    ap.add_argument(
        "--bootstrap-seed", type=int, default=148202, help="pdshrink bootstrap seed (plan §10)"
    )
    ap.add_argument("--figures", type=Path, default=None, help="pdshrink figure dir")
    ap.add_argument("--pilot-n", type=int, default=None)
    ap.add_argument("--gen-batch", type=int, default=None)
    ap.add_argument("--headline-floor", type=int, default=None)
    ap.add_argument("--max-features-in", type=int, default=None)
    ap.add_argument("--max-features-out", type=int, default=None)
    ap.add_argument(
        "--krr-nystrom-centers",
        type=int,
        default=16384,
        help="EXPLICIT parent realized value (n1m_fits.json krr_grid; script default is 8192)",
    )
    ap.add_argument("--seed", type=int, default=0, help="fitter seed (n1m_fits.json seed: 0)")
    ap.add_argument("--sae-k", default="auto", help="auto = Gate B chosen_k; or 64/128")
    ap.add_argument("--hf-prefix", default=HF_PREFIX_DEFAULT)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--seed-partial",
        default=None,
        metavar="ATTEMPT_ID",
        help="pre-P0 seeding: stage issue1482_partial/<ATTEMPT_ID>/eval_results_issue_1482 "
        "(a prior attempt's crash-persisted P1 percontext units + sae_fitness) into "
        "out_eval so the existing resume predicates skip completed P1 units; validated "
        "per unit + split-identity-reconciled after P0 (invalid/stale seeds re-run)",
    )
    ap.add_argument(
        "--tiny-model",
        action="store_true",
        help="CARVE-OUT (GPU-bound P2 on a no-GPU VM): from-config 24-layer same-arch "
        "Qwen2 over the REAL vocab instead of the 7B weights (#906 tiny-real pattern). "
        "G2/FVE values are then structural-only; gates are smoke-demoted anyway.",
    )
    ap.add_argument(
        "--n-gpus",
        type=int,
        default=0,
        help="0 = detect via nvidia-smi; on CPU, >0 = parallel un-pinned worker slots",
    )
    ap.add_argument("--fit-id", default=None, help="(p1-fit child)")
    ap.add_argument("--p3-unit", default=None, help="(p3 child) one unit from p3_unit_specs()")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    args = ap.parse_args()

    smoke_defaults = {
        "n_boot": 200,
        "max_chunks": 1,
        "holdout_n": 10,
        "sae_n": 12,
        "sae_val_n": 4,
        "fit_n": 2000,
        "pilot_n": 6,
        "gen_batch": 2,
        "headline_floor": 2,
        "max_features_in": 256,
        "max_features_out": 512,
    }
    prod_defaults = {
        "n_boot": 10_000,
        "max_chunks": 0,
        "holdout_n": 20_000,
        "sae_n": 120_000,
        "sae_val_n": 2000,
        "fit_n": 0,
        "pilot_n": 500,
        "gen_batch": 8,
        "headline_floor": 50,
        "max_features_in": 8192,
        "max_features_out": 16384,
    }
    dd = smoke_defaults if args.smoke else prod_defaults
    for k, v in dd.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.device == "auto":
        args.device = "cuda" if _physical_gpu_ids() else "cpu"
    root = PROJECT_ROOT / "data" / "issue_1482"
    base = (root / "smoke_out") if args.smoke else root
    if args.out_eval is None:
        args.out_eval = (
            (base / "eval_results")
            if args.smoke
            else (PROJECT_ROOT / "eval_results" / "issue_1482")
        )
    if args.scratch is None:
        args.scratch = base / "scratch"
    if args.store is None:
        args.store = base / "store" / "sae_pooled"
    if args.sae_dir is None:
        args.sae_dir = root / "hf_dl" / "sae"
    if args.figures is None:
        # smoke figures NEVER touch the committed figures/ paths (kresample convention)
        args.figures = (
            (base / "figures") if args.smoke else (PROJECT_ROOT / "figures" / "issue_1482")
        )
    for p in (args.out_eval, args.scratch, args.store.parent):
        p.mkdir(parents=True, exist_ok=True)

    if args.seed_partial:
        if args.phase not in ("all", "p0", "p1"):
            ap.error("--seed-partial is a pre-P0 seeding step; valid only with --phase all|p0|p1")
        rec_path = args.out_eval / SEED_PARTIAL_RECORD
        if rec_path.exists():
            prev = json.loads(rec_path.read_text())
            if prev.get("attempt_id") != args.seed_partial:
                raise RuntimeError(
                    f"[seed-partial] out_eval holds a record for a DIFFERENT attempt "
                    f"({prev.get('attempt_id')} != {args.seed_partial}); refusing to re-seed"
                )
            logger.info("[seed-partial] record for %s already present; skip", args.seed_partial)
        else:
            seed_partial(args)

    t0 = time.time()
    ph = args.phase
    if ph == "p1-fit":
        phase_p1_fit(args)
        return 0
    if ph == "p2-worker":
        phase_p2_worker(args)
        return 0
    if ph == "pdshrink":  # amendment round (plan v12): requires --phase p0 outputs in scratch
        phase_pdshrink(args)
        return 0
    if ph in ("all", "p0"):
        phase_p0(args)
    if ph in ("all", "p1"):
        phase_p1(args)
    p4_proc = None
    if ph in ("all", "p2"):
        phase_p2(args)
        if ph == "all":  # launch P4 detached-concurrent with P3 (#825 store-before-long-fit)
            log = args.scratch / "child_logs" / "p4_detached.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                "p4",
                *_child_flags(args),
            ]
            if args.skip_upload:
                cmd.append("--skip-upload")
            p4_log_f = open(log, "w")  # noqa: SIM115 — fd owned by the detached child
            p4_proc = subprocess.Popen(
                cmd, env={**os.environ}, stdout=p4_log_f, stderr=subprocess.STDOUT
            )
            logger.info("[all] P4 upload launched detached (pid=%d)", p4_proc.pid)
    if ph in ("all", "p3"):
        phase_p3(args)
    if ph == "p4":
        phase_p4(args)
    if p4_proc is not None:
        rc = p4_proc.wait()
        if rc != 0:
            logger.error(
                "[all] detached P4 FAILED rc=%d; tail:\n%s",
                rc,
                _log_tail(args.scratch / "child_logs" / "p4_detached.log"),
            )
            raise RuntimeError(f"P4 upload failed rc={rc}")
        logger.info("[all] P4 upload verified complete")
    if ph == "all":
        _results_sentinel(args, t0)
        C.phase("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
