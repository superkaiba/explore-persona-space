#!/usr/bin/env python3
"""Issue #2618 — direct answer→context ridge map vs pseudoinverse constructions.

Fits a DIRECT reverse (v_A → v_C) ridge map on the #779 n1m bank (963,444 real
LMSYS+WildChat contexts) at layers 14/19/26 and compares it against
pseudoinverse constructions of the BANKED forward context→answer ridge
(`issue779_monitoring/n1m_readout/weights/L{l}/ridge.pt`). ANALYSIS-ONLY: no
training, no generation, no judging; linear maps only (CLAUDE.md
linear-by-default — no MLP/KRR payload is ever loaded).

Phases (`--phase {stage,fits,topctx,upload}`, each resumable, each exits
explicitly):

  stage   download ALL n1m capture chunks + the 3 banked ridge.pt payloads +
          the pass_b bundle + the r_B bank to the local stage dir (per-chunk
          presence check => re-runs skip-if-done; >=95 GB headroom asserted
          before starting).
  fits    per layer (sequential, per-layer JSON checkpoint): assemble (cx, vx)
          from LOCAL staged chunks + the pass_b head, apply the pinned
          fixed_split(5000, 3600, 400, 1000, 42) with hard-asserted val/test
          shas; fit the REVERSE ridge (exact mirror of the forward recipe with
          roles swapped, via the forward driver's own fit_ridge_with_weights);
          run the pinv battery off the banked forward W; evaluate all
          predictors on the SAME pinned 1000 test rows (R2 raw + standardized,
          kNN retrieval, identity+bias + predict-the-mean baselines); operator
          geometry in the shared frame; per-trait preimage agreement.
  topctx  second LOCAL pass over staged chunks: project every TRAIN context's
          cx onto each trait's fitted-reverse / pinv(k*) / ridge-pinv
          directions; overlap@k for k in {30, 100, 1000} + Spearman of the
          projection rankings. Batched GEMMs, never a per-row loop.
  upload  ONE bulk upload_folder commit of the out-root to the HF data repo
          prefix `issue2618_reverse_map/analysis_tensors/`.

Also `--pilot` (time ONE chunk download + one 50k-row fp64 gram block + one
(H,H) eigh on the device, print measured per-unit walls + extrapolated totals
— the sizing basis) and `--import-check` (argparse-attribute completeness +
hub call-shape binding, orchestrate.argcheck).

Conventions mirrored EXACTLY from the forward n1m fits driver
(`issue779_ffc_n1m_fits.py`, imported as N1M — never re-implemented):
`fixed_split(5000, 3600, 400, 1000, 42)` with pinned val/test index shas,
`LAMBDAS_N1M = logspace(-3, 8, 23)`, `RIDGE_BLOCK = 50_000`, the fp64
streaming primal-ridge fit (X^TX / X^TY block accumulation on device + one
eigh of (H, H)) and its variance-weighted held-out R2 metric
(`PR._pooled_r2`, SS_tot on the eval set's own mean).

Frames (row-vector right-multiplication convention throughout):

  forward (banked):  vhat = ((v_C - xmu)/xsd) @ W + ymu          W (3584, 3584)
  reverse (fit):     chat = ((v_A - amu)/asd) @ W_rev + cmu
  pinv predictors:   xhat_std = (v_A - ymu) @ P;  xhat = xhat_std * xsd + xmu
     full-rank pinv      P = V diag(1/s) U^T          (SVD W = U diag(s) V^T)
     truncated pinv      P_k = V_k diag(1/s_k) U_k^T  (k grid + val-selected k*)
     ridge-pinv          P_lam = W^T (W W^T + lam I)^{-1} = V diag(s/(s^2+lam)) U^T
  shared operator frame (answer raw-centered -> standardized context):
     reverse linear part B_rev = diag(1/asd) @ W_rev @ diag(1/xsd)
     every pinv P is already in this frame by construction.

Observed schemas (probed 2026-08-26; schema-from-artifact, #2061):

  ridge.pt  (data/issue_2094/.../n1m_readout/weights/L14/ridge.pt): keys
    {W (3584,3584) f32, fitter='ridge', kind='ridge', layer, selected_lambda,
     xmu/xsd/ymu (3584,) f32}
  capture chunk (shard00_chunk0000.pt @ issue779_monitoring/
    fitter-fair-comparison-n1m/final_token_capture/, 1920 chunk files): keys
    {chunk, ci (list, len<=500), cx_last (n,3,3584) f32, layers=[14,19,26],
     prompts (RAW real-user text — NEVER printed/logged), shard_index,
     v_x (n,3,3584) f32}

Smoke blind-spot enumeration:
  - `--smoke` caps the chunk pass at 3 chunks (n_train 5,100 not 963,444), so
    the production n_train==963,444 parity assert vs the banked forward fit and
    the forward-vs-reverse train-mean agreement assert are DEMOTED to logged
    diagnostics under smoke (production-n-calibrated gates are structurally
    unsatisfiable at smoke n — gotchas.md smoke/production gate calibration);
    n_train > d = 3584 stays asserted in BOTH modes.
  - `--smoke` shrinks the lambda grid to {1e-3, 1.0} and the k grid to
    {8, 512}: grid-EDGE diagnostics are uninformative under smoke.
  - `--smoke` uploads divert to `issue2618_reverse_map/smoke/...`: the
    production upload prefix string itself is not exercised by the smoke.
  - Everything else is the SAME code path at production shape: real HF
    downloads, real fp64 eigh/SVD at H=3584, all pinv variants, all three
    traits, all metrics.

Content hygiene: capture chunks and the pass_b bundle carry raw real-user
prompt text; this driver never reads, prints, or logs any text field — rows
are referenced by integer manifest index (ci) only.

Pod sentinels: /workspace/logs/issue-2618-<phase>.json (+ [phase=...]
breadcrumbs). Pod-side code NEVER shells out to scripts/task.py.
"""

from __future__ import annotations

# Accelerator scoping (per-workload, gotchas.md HF failure matrix): this
# driver's download shape is a 1,920-file sequential chunk storm — exactly the
# many-file shape that wedges xet (#1345 hit it on THESE n1m chunks) — and its
# uploads are small (~250 MB, far under the 50 GB plain-path cap), so both
# accelerators are off BY DEFAULT for this process only (setdefault => a real
# env override still wins). Must precede any transitive huggingface_hub import.
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + credentials land BEFORE numpy/torch import.
load_dotenv()

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict_blocked,
    knn_retrieval,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2618_reverse_map")

ISSUE = 2618
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# Consumer-own constants, never re-typed literals (gotchas.md reused-module rule).
CAPTURE_PREFIX = f"{N1G.HF_PREFIX}/final_token_capture"
WEIGHTS_PREFIX = "issue779_monitoring/n1m_readout/weights"
HF_UPLOAD_PREFIX = "issue2618_reverse_map"
# #779 r_B bank pin (issue2254_preimage.py lineage).
HF_REV_RB = "037fcbb210bc52c459959b0746cc268fe08bae96"

LAYERS = (14, 19, 26)
BEHAVIORS = ("evil", "sycophancy", "hallucination")
H_DIM = 3584
N_TRAIN_FORWARD = 963_444  # the banked mixed_1m realized n_train (parity assert, non-smoke)

K_GRID = (8, 32, 128, 512, 1024, 1433, 2048, 3072, 3584)
K_GRID_SMOKE = (8, 512)
LAMBDAS_SMOKE = (1e-3, 1.0)
KNN_KS = (1, 5, 10, 50)
TOPCTX_KS = (30, 100, 1000)
SMOKE_MAX_CHUNKS = 3
TOPCTX_CKPT_EVERY = 200  # chunks between topctx score checkpoints (stream-ckpt law)
STAGE_NEED_GB = 95.0
STAGE_NEED_GB_SMOKE = 3.0
# Forward-vs-reverse train-mean agreement bar (both are the SAME train rows'
# context mean, fp64-computed / fp32-stored): fp32 storage roundoff is ~1e-6 of
# the O(1-30) activation scale; a wrong row SET moves the mean by orders more.
MEAN_AGREE_MAX_REL = 0.01

STAGE_DIR_DEFAULT = "/workspace/issue2618_stage"
OUT_ROOT_DEFAULT = "/workspace/issue2618_out"
OUT_ROOT_SMOKE_DEFAULT = "/workspace/issue2618_out_smoke"
RESULTS_SUBDIR = Path("eval_results") / "issue_2618" / "reverse_map"

PHASES = ("stage", "fits", "topctx", "upload")


# ── small shared helpers (issue2254_preimage.py conventions) ────────────────────


def _breadcrumb(phase: str, **kw) -> None:
    kv = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[phase={phase}] {kv}", flush=True)


def _progress(phase: str, k: int, n: int, key: str, t0: float) -> None:
    print(f"[{phase}] unit {k}/{n} {key} elapsed={time.time() - t0:.1f}s", flush=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _run_metadata(phase: str, extra: dict | None = None) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    md = {
        "experiment": "issue2618_reverse_map",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capture_prefix": CAPTURE_PREFIX,
        "weights_prefix": WEIGHTS_PREFIX,
        "rb_revision": HF_REV_RB,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    md.update(as_metadata_dict(git_provenance(), phase=phase))
    if extra:
        md.update(extra)
    return md


def _write_sentinel(out_root: Path, phase: str, status: str, extra: dict | None = None) -> Path:
    """Per-phase sentinel (/workspace/logs/issue-2618-<phase>.json), observed by
    the VM orchestrator via file presence / direct reads (the #2254/#2220
    envelope — deliberately not poller-drained). NEVER shells to task.py."""
    logs = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))
    payload = {"issue": ISSUE, "phase": phase, "status": status, "out_root": str(out_root)}
    if extra:
        payload.update(extra)
    try:
        logs.mkdir(parents=True, exist_ok=True)
        p = logs / f"issue-{ISSUE}-{phase}.json"
        _write_json_atomic(p, payload)
        return p
    except OSError as exc:  # sentinel dir absent off-pod (VM smoke) -> log, never crash
        logger.info("[sentinel] %s not writable (%s); skipping", logs, type(exc).__name__)
        return Path("/dev/null")


def _dev(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(device)


def _chunk_dir(stage_dir: Path) -> Path:
    # hf_hub_download(local_dir=stage_dir) preserves the repo-relative path.
    return stage_dir / CAPTURE_PREFIX


def _stage_manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "stage_manifest.json"


def _resolve_pinned_revision(stage_dir: Path) -> tuple[str, list[str]]:
    """Pin the data repo's main sha ONCE per stage (a resumed stage reuses the
    manifest pin so mid-stage repo commits never mix revisions — the
    revision=None snapshot-split trap), and return the sorted chunk universe."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    man_path = _stage_manifest_path(stage_dir)
    api = HfApi()
    if man_path.is_file():
        man = json.loads(man_path.read_text())
        rev = man["revision"]
        names = list(man["chunk_files"])
        logger.info("[stage] resuming with pinned revision %s (%d chunks)", rev[:12], len(names))
        return rev, names
    rev = hub.retry_transient(
        lambda: api.repo_info(HF_DATA_REPO, repo_type="dataset").sha,
        what="resolve data-repo main sha",
    )
    names = sorted(
        p
        for p in hub.list_hf_files_under_path(
            api, HF_DATA_REPO, CAPTURE_PREFIX, repo_type="dataset", revision=rev
        )
        if p.endswith(".pt")
    )
    if not names:
        raise RuntimeError(f"no capture chunks under {HF_DATA_REPO}@{rev[:12]}:{CAPTURE_PREFIX}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(man_path, {"revision": rev, "chunk_files": names})
    logger.info("[stage] pinned revision %s (%d chunks)", rev[:12], len(names))
    return rev, names


# ── phase: stage ─────────────────────────────────────────────────────────────────


def phase_stage(args) -> None:
    from explore_persona_space.experiments.issue_1739 import store_io
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    t0 = time.time()
    stage_dir = Path(args.stage_dir)
    need_gb = STAGE_NEED_GB_SMOKE if args.smoke else STAGE_NEED_GB
    # statvfs floor + 1 GB fallocate canary (assert_out_root_headroom); the
    # MooseFS per-pod quota beyond the canary is a known residual (gotchas.md).
    assert_out_root_headroom(stage_dir, need_gb, phase="stage")
    _breadcrumb("stage", status="start", stage_dir=stage_dir, smoke=args.smoke)

    rev, names = _resolve_pinned_revision(stage_dir)
    if args.max_chunks is not None:
        names = names[: args.max_chunks]

    n_done, n_dl = 0, 0
    for i, rel in enumerate(names):
        target = _chunk_dir(stage_dir) / rel.rsplit("/", 1)[-1]
        if target.is_file() and target.stat().st_size > 0:
            n_done += 1
        else:
            N1M._download_chunk_with_retry(HF_DATA_REPO, rel, stage_dir, revision=rev)
            if not target.is_file():
                raise RuntimeError(f"chunk download landed off-path: expected {target}")
            n_dl += 1
        _progress("stage", i + 1, len(names), rel.rsplit("/", 1)[-1], t0)

    # Banked forward ridge payloads (ONLY ridge.pt — the mlp/krr payloads are
    # never loaded; linear-by-default).
    for ly in LAYERS:
        dest = stage_dir / "weights" / f"L{ly}" / "ridge.pt"
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{WEIGHTS_PREFIX}/L{ly}/ridge.pt",
            dest,
            repo_type="dataset",
            revision=rev,
        )
        _load_forward_payload(stage_dir, ly)  # validates kind/layer/shape now, not mid-fits
        logger.info("[stage] forward ridge L%d staged + validated", ly)

    # pass_b bundle (the fits loader's own path + HF fallback) — validate loadable.
    pb = N1G._load_pass_b_bundle(Path(args.pass_b))
    n_pb = int(pb["cx_last"].shape[0])
    assert n_pb == N1M.N_PASS_B, (n_pb, N1M.N_PASS_B)
    del pb

    # r_B bank at the #779 pin (issue2254 lineage) — prefetch + trait presence.
    rb_bank, trait_names = store_io.load_rb_bank(
        revision=HF_REV_RB, local_dir=stage_dir / "rb_bank"
    )
    missing = [b for b in BEHAVIORS if b not in trait_names]
    if missing:
        raise RuntimeError(f"r_B bank missing traits {missing} (has {trait_names})")
    del rb_bank

    summary = {
        "revision": rev,
        "n_chunks": len(names),
        "n_already_present": n_done,
        "n_downloaded": n_dl,
        "stage_dir": str(stage_dir),
        "wall_s": round(time.time() - t0, 1),
        "meta": _run_metadata("stage"),
    }
    _write_json_atomic(stage_dir / "stage_summary.json", summary)
    _write_sentinel(stage_dir, "stage", "done", {"n_chunks": len(names)})
    _breadcrumb("stage", status="done", n_chunks=len(names), downloaded=n_dl, cached=n_done)


# ── data assembly (staged chunks + pass_b head + pinned split) ──────────────────


def _stream_local_chunks_capped(local_dir: Path, layer: int, max_chunks: int | None):
    """N1M._stream_local_chunks with an optional chunk cap (smoke slices the
    SAME sorted chunk universe; body otherwise mirrors the reused original)."""
    chunk_files = sorted(local_dir.glob("shard*_chunk*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"no n1m capture chunks under {local_dir}")
    if max_chunks is not None:
        chunk_files = chunk_files[:max_chunks]
    cx_parts: list[np.ndarray] = []
    vx_parts: list[np.ndarray] = []
    ci_parts: list[list[int]] = []
    for cp in chunk_files:
        b = F._mmap_load(cp)
        cx_parts.append(N50._slice_layer(b, "cx_last", layer))
        vx_parts.append(N50._slice_layer(b, "v_x", layer))
        ci_parts.append([int(x) for x in b["ci"]])
        del b
    logger.info("[assemble] %d staged chunks (layer %d)", len(chunk_files), layer)
    return N1M._concat_stream_parts(cx_parts, vx_parts, ci_parts)


def assemble_layer(args, layer: int):
    """(cx, vx, tr, val, test, split_meta) at ``layer``: pass_b head rows
    [0, 5000) + staged capture rows after, pinned split shas hard-asserted
    (mirrors N1M.assemble_multilayer's assembly + split block)."""
    pb = N1G._load_pass_b_bundle(Path(args.pass_b))
    for fld in ("cx_last", "v_x"):
        assert fld in pb, f"pass_b missing {fld}"
    assert int(pb["cx_last"].shape[0]) == N1M.N_PASS_B, (pb["cx_last"].shape[0], N1M.N_PASS_B)
    pb_cx = N50._slice_layer(pb, "cx_last", layer)
    pb_vx = N50._slice_layer(pb, "v_x", layer)
    del pb

    cx_new, vx_new, ci_new = _stream_local_chunks_capped(
        _chunk_dir(Path(args.stage_dir)), layer, args.max_chunks
    )
    assert (ci_new >= 0).all(), "capture chunk rows must carry manifest ci >= 0"
    cx = np.concatenate([pb_cx, cx_new])
    vx = np.concatenate([pb_vx, vx_new])
    ci = np.concatenate([np.full(N1M.N_PASS_B, -1, dtype=np.int64), ci_new])
    del pb_cx, pb_vx, cx_new, vx_new
    gc.collect()
    n_rows = cx.shape[0]
    assert cx.shape == (n_rows, H_DIM) and vx.shape == (n_rows, H_DIM), (cx.shape, vx.shape)
    assert (ci[: N1M.N_PASS_B] == -1).all(), "pass_b head rows must carry ci=-1"

    pinned = N50._pinned_original_shas(Path(args.orig_dir))
    r1_train, val, test = F.fixed_split(
        N1M.N_PASS_B, N1M.N_PASS_B - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    val_sha, test_sha = F._sha_ids(val), F._sha_ids(test)
    assert val_sha == pinned["val_sha256"], (
        f"val sha {val_sha} != pinned original {pinned['val_sha256']} — NOT byte-identical"
    )
    assert test_sha == pinned["test_sha256"], (
        f"test sha {test_sha} != pinned original {pinned['test_sha256']}"
    )
    assert (val < N1M.N_PASS_B).all() and (test < N1M.N_PASS_B).all(), (
        "val/test must index the pass_b head"
    )

    tr = np.sort(
        np.concatenate([np.asarray(r1_train, dtype=np.int64), np.arange(N1M.N_PASS_B, n_rows)])
    )
    excl = set(int(x) for x in val) | set(int(x) for x in test)
    assert not (set(int(x) for x in tr[: N1M.N_PASS_B]) & excl), "train pool overlaps val/test"
    assert len(tr) > H_DIM, (
        f"n_train={len(tr)} <= d={H_DIM}: reverse ridge would be under-determined — refusing"
    )
    if not args.smoke and args.max_chunks is None:
        # Production parity: the banked forward mixed_1m point trained on exactly
        # this pool. DEMOTED to a log line under smoke (gate-calibration rule).
        assert len(tr) == N_TRAIN_FORWARD, (
            f"realized train pool {len(tr)} != banked forward n_train {N_TRAIN_FORWARD} — "
            "staged chunk set drifted from the mixed_1m capture"
        )
    else:
        logger.info("[assemble] smoke/capped: n_train=%d (forward parity assert skipped)", len(tr))

    split_meta = {
        "n_rows": int(n_rows),
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "val_sha256": val_sha,
        "test_sha256": test_sha,
        "pinned_source": pinned["source"],
        "val_test_byte_identical_original": True,
        "train_pool": "fixed_split train ids (3600 of pass_b) + ALL captured rows",
    }
    return cx, vx, tr, val, test, split_meta


# ── forward payload + pinv battery ───────────────────────────────────────────────


def _load_forward_payload(stage_dir: Path, layer: int) -> dict:
    """Banked forward ridge at ``layer`` (issue2474_n1m_map.load_n1m_comp
    asserts: ridge fitter, matching layer, square W)."""
    path = Path(stage_dir) / "weights" / f"L{layer}" / "ridge.pt"
    if not path.is_file():
        raise FileNotFoundError(f"banked forward ridge absent: {path} — run --phase stage first")
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("kind") != "ridge" or p.get("fitter") != "ridge":
        raise RuntimeError(f"{path}: expected the ridge fitter, got {p.get('kind')!r}")
    if int(p.get("layer", -1)) != int(layer):
        raise RuntimeError(f"{path}: payload layer {p.get('layer')} != requested {layer}")
    if tuple(p["W"].shape) != (H_DIM, H_DIM):
        raise RuntimeError(f"{path}: W shape {tuple(p['W'].shape)} != ({H_DIM}, {H_DIM})")
    return p


def _svd_robust(M: torch.Tensor):
    """torch.linalg.svd with the CPU fallback for cuSOLVER non-convergence
    (gotchas.md cuda eigh/svd entry; exact numerical-backend swap, no jitter)."""
    try:
        return torch.linalg.svd(M, full_matrices=False)
    except torch.linalg.LinAlgError:
        logger.warning("[svd] cuSOLVER failed to converge on %s; CPU LAPACK fallback", M.device)
        U, S, Vh = torch.linalg.svd(M.cpu(), full_matrices=False)
        return U.to(M.device), S.to(M.device), Vh.to(M.device)


class PinvBattery:
    """Pseudoinverse constructions off ONE SVD of the banked forward W.

    Row convention: forward is x_std @ W = y - ymu, so every inverse maps the
    RAW-CENTERED answer (v_A - ymu) to the STANDARDIZED context frame.
    """

    def __init__(self, fw: dict, dev: torch.device):
        self.xmu = torch.as_tensor(np.asarray(fw["xmu"]), dtype=torch.float64, device=dev)
        self.xsd = torch.as_tensor(np.asarray(fw["xsd"]), dtype=torch.float64, device=dev)
        self.ymu = torch.as_tensor(np.asarray(fw["ymu"]), dtype=torch.float64, device=dev)
        self.W = torch.as_tensor(np.asarray(fw["W"]), dtype=torch.float64, device=dev)
        self.dev = dev
        U, S, Vh = _svd_robust(self.W)
        assert S.shape[0] == H_DIM and (S >= 0).all(), ("degenerate forward spectrum", S.shape)
        self.U, self.S, self.V = U, S, Vh.T  # U (H,k) ctx-side; V (D,k) ans-side

    def center(self, va: np.ndarray) -> torch.Tensor:
        """(n, D) raw answers -> fp64 raw-centered (v_A - ymu) on device."""
        return torch.as_tensor(np.asarray(va), dtype=torch.float64, device=self.dev) - self.ymu

    def coeffs(self, yc: torch.Tensor) -> torch.Tensor:
        """Shared projection yc @ V (n, k) — every variant reads slices of it."""
        return yc @ self.V

    def predict_std_trunc(self, a: torch.Tensor, k: int) -> torch.Tensor:
        """Truncated pinv: xhat_std = yc @ V_k diag(1/s_k) U_k^T, from a = yc @ V."""
        kk = int(min(k, self.S.shape[0]))
        assert kk > 0, f"truncated pinv k={k} leaves no components"
        return (a[:, :kk] / self.S[:kk]) @ self.U[:, :kk].T

    def predict_std_ridge(self, a: torch.Tensor, lam: float) -> torch.Tensor:
        """Ridge-pinv: xhat_std = yc @ V diag(s/(s^2+lam)) U^T, from a = yc @ V."""
        filt = self.S / (self.S**2 + float(lam))
        return (a * filt) @ self.U.T

    def destandardize(self, xstd: torch.Tensor) -> np.ndarray:
        return (xstd * self.xsd + self.xmu).cpu().numpy()

    def operator_trunc(self, k: int) -> torch.Tensor:
        """P_k = V_k diag(1/s_k) U_k^T (answer raw-centered -> std context)."""
        kk = int(min(k, self.S.shape[0]))
        return (self.V[:, :kk] / self.S[:kk]) @ self.U[:, :kk].T

    def operator_ridge(self, lam: float) -> torch.Tensor:
        filt = self.S / (self.S**2 + float(lam))
        return (self.V * filt) @ self.U.T


# ── metrics + operator geometry ─────────────────────────────────────────────────


def _r2_pair(pred_raw: np.ndarray, true_raw: np.ndarray, batt: PinvBattery) -> dict:
    """Held-out R2 in RAW context space (primary) + the forward-standardizer
    frame (companion). Pooled R2 = PR._pooled_r2 (SS_tot on the eval set's own
    mean — the #779 variance-weighted convention)."""
    xmu = batt.xmu.cpu().numpy()
    xsd = batt.xsd.cpu().numpy()
    return {
        "r2_raw": float(PR._pooled_r2(pred_raw, true_raw)),
        "r2_std": float(PR._pooled_r2((pred_raw - xmu) / xsd, (true_raw - xmu) / xsd)),
    }


def _vec_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    va, vb = a.reshape(-1), b.reshape(-1)
    return float((va @ vb) / (va.norm() * vb.norm() + 1e-12))


def _procrustes_cosine(sa: torch.Tensor, sb: torch.Tensor) -> float:
    """Two-sided orthogonal-Procrustes-aligned cosine from the two spectra
    (issue1345_operator_comparison.spectrum_cosine, von Neumann bound).
    ROTATION-INVARIANT-ONLY — a descriptive ceiling, never direction evidence."""
    return float((sa * sb).sum() / (sa.norm() * sb.norm() + 1e-12))


def _principal_angle_cos(qa: torch.Tensor, qb: torch.Tensor) -> np.ndarray:
    """cos of principal angles between two orthonormal column blocks."""
    return torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0).cpu().numpy()


def operator_geometry(b_rev: torch.Tensor, variants: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Direction-aware raw operator cosine + rotation-invariant Procrustes
    cosine + principal angles (top-k input/output singular subspaces) for the
    reverse-map linear part vs each pinv variant, all in the SHARED frame.
    Returns (per-variant records, spectra arrays for the npz/overlay)."""
    Ur, Sr, Vhr = _svd_robust(b_rev)
    spectra = {"B_rev": Sr.cpu().numpy()}
    out: dict[str, dict] = {}
    for name, P in variants.items():
        Up, Sp, Vhp = _svd_robust(P)
        rank = int((Sp > Sp[0] * 1e-12).sum()) if Sp.shape[0] else 0
        rec = {
            "raw_operator_cosine_direction_aware": _vec_cosine(b_rev, P),
            "procrustes_cosine_rotation_invariant_only": _procrustes_cosine(Sr, Sp),
            "variant_rank": rank,
            "principal_angles": {},
        }
        for k in (8, 32, 128, 512):
            if k > min(rank, Sr.shape[0]):
                rec["principal_angles"][str(k)] = None  # subspace ill-defined past the rank
                continue
            cos_in = _principal_angle_cos(Ur[:, :k], Up[:, :k])  # input = answer side
            cos_out = _principal_angle_cos(Vhr[:k].T, Vhp[:k].T)  # output = std-context side
            rec["principal_angles"][str(k)] = {
                "input_mean_cos": float(cos_in.mean()),
                "input_min_cos": float(cos_in.min()),
                "output_mean_cos": float(cos_out.mean()),
                "output_min_cos": float(cos_out.min()),
            }
        out[name] = rec
        if name in ("pinv_full", "ridge_pinv"):
            spectra[name] = Sp.cpu().numpy()
    return out, spectra


# ── phase: fits ──────────────────────────────────────────────────────────────────


def _fits_regime(args, layer: int, n_chunks_staged: int) -> dict:
    """Resume key: GENERATING PARAMETERS only (never hashed recomputed floats —
    gotchas.md float-last-bit rule)."""
    return {
        "layer": int(layer),
        "smoke": bool(args.smoke),
        "max_chunks": args.max_chunks,
        "n_chunks_staged": int(n_chunks_staged),
        "lambda_grid": "smoke:[1e-3,1.0]" if args.smoke else "logspace(-3,8,23)",
        "k_grid": list(K_GRID_SMOKE if args.smoke else K_GRID),
        "knn_ks": list(KNN_KS),
    }


def _layer_json_path(out_root: Path, layer: int) -> Path:
    return out_root / RESULTS_SUBDIR / f"fits_L{layer}.json"


def _rb_rows(stage_dir: Path, layer: int) -> dict[str, np.ndarray]:
    """{behavior: (H,) float64 r_B at ``layer``} from the pinned #779 bank."""
    from explore_persona_space.experiments.issue_1739 import store_io

    rb_bank, trait_names = store_io.load_rb_bank(
        revision=HF_REV_RB, local_dir=Path(stage_dir) / "rb_bank"
    )
    out: dict[str, np.ndarray] = {}
    for b in BEHAVIORS:
        if b not in trait_names:
            raise RuntimeError(f"behavior {b} absent from r_B bank traits {trait_names}")
        out[b] = np.asarray(rb_bank[layer, trait_names.index(b), :], dtype=np.float64)
        assert out[b].shape == (H_DIM,), out[b].shape
    return out


def _persist_rev_payload(out_root: Path, layer: int, payload: dict) -> Path:
    """W_rev payload, forward key shape with roles renamed (xmu->amu, xsd->asd,
    ymu->cmu): chat = ((v_A - amu)/asd) @ W + cmu. fp32 tensors, atomic save."""
    dest = out_root / "analysis_tensors" / "weights_rev" / f"L{layer}" / "ridge_rev.pt"
    rec = {
        "kind": "ridge_rev",
        "fitter": "ridge_rev",
        "layer": int(layer),
        "selected_lambda": float(payload["selected_lambda"]),
        "amu": payload["xmu"],
        "asd": payload["xsd"],
        "cmu": payload["ymu"],
        "W": payload["W"],
        "prediction_path": "chat = ((v_A - amu)/asd) @ W + cmu",
    }
    with atomic_replace(dest) as tmp:
        torch.save(rec, tmp)
    logger.info("[persist] wrote %s (%.0f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def fit_layer(args, layer: int, dev: torch.device) -> dict:
    """One layer end-to-end: assemble, reverse fit, pinv battery, predictive
    eval, kNN retrieval, operator geometry, preimage agreement."""
    t0 = time.time()
    out_root = Path(args.out_root)
    lambdas = np.asarray(LAMBDAS_SMOKE, dtype=np.float64) if args.smoke else N1M.LAMBDAS_N1M
    k_grid = K_GRID_SMOKE if args.smoke else K_GRID

    cx, vx, tr, val, test, split_meta = assemble_layer(args, layer)
    _breadcrumb("fits", layer=layer, step="assembled", n_rows=split_meta["n_rows"])

    # 1) REVERSE ridge — the forward driver's own fit with roles swapped:
    #    X = v_A (answers), Y = cx (contexts). fp64 streaming gram + one eigh +
    #    val-lambda by the same pooled-R2 metric on the pinned 400 val rows.
    pred_te_rev, meta_rev, payload_rev = N1M.fit_ridge_with_weights(
        vx, cx, tr, val, test, lambdas, dev, N1M.RIDGE_BLOCK
    )
    _persist_rev_payload(out_root, layer, payload_rev)
    _breadcrumb(
        "fits",
        layer=layer,
        step="reverse_fit",
        lam=meta_rev["selected_lambda"],
        val_r2=round(meta_rev["val_r2_at_selected"], 4),
        elapsed=round(time.time() - t0, 1),
    )

    # 2) Pinv battery off the banked forward W.
    fw = _load_forward_payload(Path(args.stage_dir), layer)
    batt = PinvBattery(fw, dev)

    # Coherence: forward xmu and the reverse fit's cmu are the SAME train rows'
    # context mean. Hard gate at production n; logged diagnostic under smoke
    # (the forward means were computed on the full pool).
    mean_gap = float(
        np.max(
            np.abs(payload_rev["ymu"].numpy().astype(np.float64) - batt.xmu.cpu().numpy())
            / (np.median(batt.xsd.cpu().numpy()) + 1e-12)
        )
    )
    if not args.smoke and args.max_chunks is None:
        assert mean_gap < MEAN_AGREE_MAX_REL, (
            f"forward xmu vs reverse cmu disagree (max rel gap {mean_gap:.4g}) — "
            "train rows differ from the banked forward fit"
        )
    else:
        logger.info("[fits] L%d mean agreement gap (smoke, diagnostic): %.4g", layer, mean_gap)

    a_val = batt.coeffs(batt.center(vx[val]))
    a_te = batt.coeffs(batt.center(vx[test]))
    cx_val, cx_te = cx[val], cx[test]

    # k* val-selected on RAW-space pooled R2 (same metric family as the fit).
    val_r2_by_k = {}
    for k in k_grid:
        pred = batt.destandardize(batt.predict_std_trunc(a_val, k))
        val_r2_by_k[int(k)] = float(PR._pooled_r2(pred, cx_val))
    k_star = max(val_r2_by_k, key=lambda k: (np.nan_to_num(val_r2_by_k[k], nan=-np.inf), -k))

    # ridge-pinv lambda* val-selected over the SAME lambda grid.
    val_r2_by_lam = {}
    for lam in lambdas:
        pred = batt.destandardize(batt.predict_std_ridge(a_val, float(lam)))
        val_r2_by_lam[float(lam)] = float(PR._pooled_r2(pred, cx_val))
    lam_star = max(val_r2_by_lam, key=lambda x: (np.nan_to_num(val_r2_by_lam[x], nan=-np.inf), -x))
    lam_edge = None
    if np.isclose(lam_star, float(lambdas[0])):
        lam_edge = "low"
    elif np.isclose(lam_star, float(lambdas[-1])):
        lam_edge = "high"

    # 3) Predictive eval, all on the SAME pinned 1000 test rows.
    preds_te: dict[str, np.ndarray] = {"reverse_ridge": np.asarray(pred_te_rev)}
    for k in k_grid:
        preds_te[f"pinv_k{k}"] = batt.destandardize(batt.predict_std_trunc(a_te, k))
    preds_te["pinv_kstar"] = preds_te[f"pinv_k{k_star}"]
    preds_te["pinv_full"] = batt.destandardize(batt.predict_std_trunc(a_te, H_DIM))
    preds_te["ridge_pinv"] = batt.destandardize(batt.predict_std_ridge(a_te, lam_star))
    preds_te["identity_bias"], ident_bias = identity_bias_predict_blocked(
        vx, cx, tr, vx[test], return_bias=True
    )
    cmu64 = payload_rev["ymu"].numpy().astype(np.float64)
    preds_te["predict_mean"] = np.broadcast_to(cmu64, (len(test), H_DIM)).copy()

    r2 = {name: _r2_pair(p, cx_te, batt) for name, p in preds_te.items()}

    # Best pinv variant BY VAL R2 (never selected on test).
    val_pinv = {f"pinv_k{k}": v for k, v in val_r2_by_k.items()}
    val_pinv["ridge_pinv"] = val_r2_by_lam[lam_star]
    val_pinv["pinv_full"] = float(
        PR._pooled_r2(batt.destandardize(batt.predict_std_trunc(a_val, H_DIM)), cx_val)
    )
    best_pinv = max(val_pinv, key=lambda n: np.nan_to_num(val_pinv[n], nan=-np.inf))

    # 4) kNN retrieval among the 1000-row test pool (chance = k/1000 recorded
    #    by knn_retrieval itself), euclidean + cosine.
    knn = {}
    for name in ("reverse_ridge", best_pinv, "identity_bias"):
        knn[name] = {
            metric: knn_retrieval(preds_te[name], cx_te, ks=KNN_KS, metric=metric)
            for metric in ("euclidean", "cosine")
        }

    # 5) Operator geometry in the shared frame (answer raw-centered ->
    #    standardized context): B_rev = diag(1/asd) @ W_rev @ diag(1/xsd).
    w_rev64 = payload_rev["W"].to(dev, torch.float64)
    asd64 = payload_rev["xsd"].to(dev, torch.float64)
    b_rev = w_rev64 / asd64[:, None] / batt.xsd[None, :]
    op_variants = {f"pinv_k{k}": batt.operator_trunc(k) for k in k_grid}
    op_variants["pinv_full"] = batt.operator_trunc(H_DIM)
    op_variants["ridge_pinv"] = batt.operator_ridge(lam_star)
    op_geo, spectra = operator_geometry(b_rev, op_variants)
    spectra["forward_W"] = batt.S.cpu().numpy()

    # 6) Preimage agreement per trait, all in the standardized-context frame.
    rb_rows = _rb_rows(Path(args.stage_dir), layer)
    dir_arrays: dict[str, np.ndarray] = {}
    preimage: dict[str, dict] = {}
    for beh, rb in rb_rows.items():
        rb_t = torch.as_tensor(rb, dtype=torch.float64, device=dev)
        d_rev = ((rb_t / asd64) @ w_rev64) / batt.xsd
        rows: dict[str, float] = {}
        dir_arrays[f"{beh}_rev"] = d_rev.cpu().numpy()
        dir_arrays[f"{beh}_read_Wt"] = (rb_t @ batt.W.T).cpu().numpy()
        for k in k_grid:
            d_k = rb_t @ op_variants[f"pinv_k{k}"]
            rows[f"pinv_k{k}"] = _vec_cosine(d_rev, d_k)
            dir_arrays[f"{beh}_pinv_k{k}"] = d_k.cpu().numpy()
        d_ridge = rb_t @ op_variants["ridge_pinv"]
        d_full = rb_t @ op_variants["pinv_full"]
        rows["ridge_pinv"] = _vec_cosine(d_rev, d_ridge)
        rows["pinv_full"] = _vec_cosine(d_rev, d_full)
        rows["read_Wt"] = _vec_cosine(d_rev, torch.as_tensor(dir_arrays[f"{beh}_read_Wt"]).to(dev))
        dir_arrays[f"{beh}_ridge_pinv"] = d_ridge.cpu().numpy()
        dir_arrays[f"{beh}_pinv_full"] = d_full.cpu().numpy()
        preimage[beh] = {"cos_rev_vs": rows}

    dir_npz = out_root / "analysis_tensors" / "directions" / f"L{layer}_directions.npz"
    dir_npz.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(dir_npz) as tmp, open(tmp, "wb") as fh:
        np.savez(fh, identity_bias_b=ident_bias, **dir_arrays)

    record = {
        "layer": int(layer),
        "regime": _fits_regime(args, layer, _n_chunks_staged(args)),
        "split": split_meta,
        "estimator": {
            "n_train": split_meta["n_train"],
            "d": H_DIM,
            "well_posed": split_meta["n_train"] > H_DIM,
            "note": "n_train >> d: primal ridge over-determined (estimator well-posed)",
        },
        "reverse_fit": {
            **meta_rev,
            "metric": "PR._pooled_r2 raw context space (val selection + test)",
        },
        "pinv_selection": {
            "k_grid": list(k_grid),
            "k_star": int(k_star),
            "val_r2_by_k_raw": val_r2_by_k,
            "lambda_star": float(lam_star),
            "lambda_grid_edge": lam_edge,
            "val_r2_by_lambda_raw": {f"{k:g}": v for k, v in val_r2_by_lam.items()},
            "best_pinv_variant_by_val_r2": best_pinv,
            "selection_space": "raw context space (PR._pooled_r2 on the pinned 400 val rows)",
        },
        "test_r2": r2,
        "knn_retrieval": knn,
        "forward_reverse_mean_agreement_max_rel": mean_gap,
        "operator_geometry": {
            "frame": "answer raw-centered -> standardized context "
            "(B_rev = diag(1/asd) @ W_rev @ diag(1/xsd); row-vector convention)",
            "per_variant": op_geo,
            "spectra": {k: np.asarray(v).round(8).tolist() for k, v in spectra.items()},
        },
        "preimage_agreement": preimage,
        "directions_npz": str(dir_npz),
        "wall_s": round(time.time() - t0, 1),
        "meta": _run_metadata("fits", {"device": str(dev)}),
    }
    del cx, vx, preds_te, a_val, a_te, batt, w_rev64, b_rev, op_variants
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _n_chunks_staged(args) -> int:
    n = len(sorted(_chunk_dir(Path(args.stage_dir)).glob("shard*_chunk*.pt")))
    return min(n, args.max_chunks) if args.max_chunks is not None else n


def phase_fits(args) -> None:
    dev = _dev(args.device)
    out_root = Path(args.out_root)
    layers = [int(x) for x in args.layers.split(",") if x]
    bad = [x for x in layers if x not in LAYERS]
    if bad:
        raise SystemExit(f"--layers: n1m capture holds only {list(LAYERS)}; got {bad}")
    _breadcrumb("fits", status="start", layers=layers, device=args.device, smoke=args.smoke)

    for i, layer in enumerate(layers):
        lj = _layer_json_path(out_root, layer)
        regime = _fits_regime(args, layer, _n_chunks_staged(args))
        if lj.is_file():
            prior = json.loads(lj.read_text())
            if prior.get("regime") == regime:
                logger.info("[fits] L%d complete (regime match) — skipping", layer)
                _progress("fits", i + 1, len(layers), f"L{layer}:resumed", time.time())
                continue
            if not args.fresh_fits:
                raise RuntimeError(
                    f"{lj} exists with a DIFFERENT regime ({prior.get('regime')} != {regime}); "
                    "quarantine it or pass --fresh-fits to recompute deliberately"
                )
            logger.warning("[fits] L%d regime mismatch; --fresh-fits recomputing", layer)
        rec = fit_layer(args, layer, dev)
        _write_json_atomic(lj, rec)
        _breadcrumb("fits", layer=layer, step="checkpointed", path=lj)

    _write_sentinel(out_root, "fits", "done", {"layers": layers})
    _breadcrumb("fits", status="done", layers=layers)


# ── phase: topctx ────────────────────────────────────────────────────────────────


def _topctx_fingerprint(dir_npz_paths: dict[int, Path], chunk_files: list[Path]) -> str:
    """Direction npz FILE BYTES (bit-exact, read from disk — safe to hash) +
    the chunk universe + the k grid."""
    h = hashlib.sha256()
    for ly in sorted(dir_npz_paths):
        h.update(f"L{ly}\n".encode())
        h.update(dir_npz_paths[ly].read_bytes())
    for cp in chunk_files:
        h.update(cp.name.encode())
        h.update(b"\n")
    h.update(f"ks={list(TOPCTX_KS)}\n".encode())
    return h.hexdigest()


def _topctx_dir_matrix(fits_rec: dict, dir_npz: Path, dev: torch.device):
    """(H, n_dirs) unit-column direction matrix for one layer + column labels:
    per trait {rev, pinv(k*), ridge_pinv} in the standardized-context frame."""
    k_star = int(fits_rec["pinv_selection"]["k_star"])
    with np.load(dir_npz) as z:
        cols, labels = [], []
        for beh in BEHAVIORS:
            for slug, key in (
                ("rev", f"{beh}_rev"),
                (f"pinv_k{k_star}", f"{beh}_pinv_k{k_star}"),
                ("ridge_pinv", f"{beh}_ridge_pinv"),
            ):
                v = np.asarray(z[key], dtype=np.float64)
                nrm = float(np.linalg.norm(v))
                assert np.isfinite(nrm) and nrm > 0, f"degenerate direction {key}"
                cols.append(v / nrm)
                labels.append(f"{beh}:{slug}")
    mat = torch.as_tensor(np.stack(cols, axis=1), dtype=torch.float64, device=dev)
    return mat, labels, k_star


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho = spearmanr(a, b).statistic
    return float(rho)


def phase_topctx(args) -> None:
    dev = _dev(args.device)
    out_root = Path(args.out_root)
    stage_dir = Path(args.stage_dir)
    t0 = time.time()
    layers = [int(x) for x in args.layers.split(",") if x]
    _breadcrumb("topctx", status="start", layers=layers, smoke=args.smoke)

    fits_recs, dir_npzs, dmats, labels_by_layer, std_by_layer = {}, {}, {}, {}, {}
    for ly in layers:
        lj = _layer_json_path(out_root, ly)
        if not lj.is_file():
            raise FileNotFoundError(f"{lj} missing — run --phase fits first")
        fits_recs[ly] = json.loads(lj.read_text())
        dir_npzs[ly] = Path(fits_recs[ly]["directions_npz"])
        if not dir_npzs[ly].is_file():
            raise FileNotFoundError(f"directions npz missing: {dir_npzs[ly]}")
        dmats[ly], labels_by_layer[ly], _ = _topctx_dir_matrix(fits_recs[ly], dir_npzs[ly], dev)
        fw = _load_forward_payload(stage_dir, ly)
        std_by_layer[ly] = (
            torch.as_tensor(np.asarray(fw["xmu"]), dtype=torch.float64, device=dev),
            torch.as_tensor(np.asarray(fw["xsd"]), dtype=torch.float64, device=dev),
        )

    chunk_files = sorted(_chunk_dir(stage_dir).glob("shard*_chunk*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"no staged chunks under {_chunk_dir(stage_dir)}")
    if args.max_chunks is not None:
        chunk_files = chunk_files[: args.max_chunks]
    fp = _topctx_fingerprint(dir_npzs, chunk_files)

    # TRAIN rows only: pass_b head restricted to the pinned fixed_split train
    # ids; every captured chunk row is train by construction.
    pinned = N50._pinned_original_shas(Path(args.orig_dir))
    r1_train, val, test = F.fixed_split(
        N1M.N_PASS_B, N1M.N_PASS_B - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    assert F._sha_ids(val) == pinned["val_sha256"] and F._sha_ids(test) == pinned["test_sha256"]

    ckpt_path = out_root / "topctx_ckpt.npz"
    cur_path = out_root / "topctx_ckpt.cursor.json"
    scores: dict[int, list[np.ndarray]] = {ly: [] for ly in layers}
    cis: list[np.ndarray] = []
    start = 0
    if ckpt_path.is_file() and cur_path.is_file() and not args.fresh_topctx:
        cur = json.loads(cur_path.read_text())
        if cur.get("fingerprint") == fp:
            with np.load(ckpt_path) as z:
                for ly in layers:
                    scores[ly].append(z[f"scores_L{ly}"])
                cis.append(z["ci"])
            start = int(cur["cursor"])
            logger.info("[topctx] resuming at chunk %d/%d", start, len(chunk_files))
        else:
            logger.warning("[topctx] checkpoint fingerprint mismatch — re-scanning from scratch")

    def _write_ckpt(cursor: int) -> None:
        packed = {f"scores_L{ly}": np.concatenate(scores[ly]) for ly in layers}
        packed["ci"] = np.concatenate(cis)
        with atomic_replace(ckpt_path) as tmp, open(tmp, "wb") as fh:
            np.savez(fh, **packed)
        _write_json_atomic(cur_path, {"fingerprint": fp, "cursor": int(cursor)})

    if start == 0:
        # pass_b head train rows, scored once from the bundle.
        pb = N1G._load_pass_b_bundle(Path(args.pass_b))
        head_idx = np.asarray(r1_train, dtype=np.int64)
        for ly in layers:
            xc = torch.as_tensor(
                N50._slice_layer(pb, "cx_last", ly)[head_idx], dtype=torch.float64, device=dev
            )
            xmu, xsd = std_by_layer[ly]
            scores[ly].append((((xc - xmu) / xsd) @ dmats[ly]).cpu().numpy().astype(np.float32))
        cis.append(np.full(len(head_idx), -1, dtype=np.int64))
        del pb

    for j in range(start, len(chunk_files)):
        cp = chunk_files[j]
        b = F._mmap_load(cp)
        ci_rows = np.asarray([int(x) for x in b["ci"]], dtype=np.int64)
        for ly in layers:
            xc = torch.as_tensor(
                N50._slice_layer(b, "cx_last", ly), dtype=torch.float64, device=dev
            )
            xmu, xsd = std_by_layer[ly]
            scores[ly].append((((xc - xmu) / xsd) @ dmats[ly]).cpu().numpy().astype(np.float32))
        cis.append(ci_rows)
        del b
        _progress("topctx", j + 1, len(chunk_files), cp.name, t0)
        if (j + 1) % TOPCTX_CKPT_EVERY == 0:
            _write_ckpt(j + 1)
    _write_ckpt(len(chunk_files))

    ci_all = np.concatenate(cis)
    n_rows = ci_all.shape[0]
    result: dict[str, dict] = {}
    for ly in layers:
        s = np.concatenate(scores[ly])  # (n_rows, 9)
        assert s.shape == (n_rows, len(labels_by_layer[ly])), (s.shape, n_rows)
        cols = {lab: s[:, i].astype(np.float64) for i, lab in enumerate(labels_by_layer[ly])}
        k_star = int(fits_recs[ly]["pinv_selection"]["k_star"])
        per_beh: dict[str, dict] = {}
        for beh in BEHAVIORS:
            rev = cols[f"{beh}:rev"]
            comps = {
                f"pinv_k{k_star}": cols[f"{beh}:pinv_k{k_star}"],
                "ridge_pinv": cols[f"{beh}:ridge_pinv"],
            }
            rec: dict[str, dict | float | list] = {}
            order_rev = np.argsort(-rev)
            for name, other in comps.items():
                order_other = np.argsort(-other)
                overlaps = {}
                for k in TOPCTX_KS:
                    kk = min(k, n_rows)
                    inter = len(set(order_rev[:kk].tolist()) & set(order_other[:kk].tolist()))
                    overlaps[str(k)] = {"overlap_frac": inter / kk, "k_effective": kk}
                rec[name] = {
                    "overlap_at_k": overlaps,
                    "spearman_vs_rev": _spearman(rev, other),
                }
            # Top-100 row ids per direction (ci = manifest index; -1 = pass_b
            # head row). Ids only — never prompt text (content hygiene).
            rec["top100_ci"] = {
                "rev": ci_all[order_rev[:100]].tolist(),
                f"pinv_k{k_star}": ci_all[np.argsort(-comps[f"pinv_k{k_star}"])[:100]].tolist(),
                "ridge_pinv": ci_all[np.argsort(-comps["ridge_pinv"])[:100]].tolist(),
            }
            per_beh[beh] = rec
        result[f"L{ly}"] = {
            "layer": ly,
            "k_star": k_star,
            "n_train_rows_scored": int(n_rows),
            "direction_columns": labels_by_layer[ly],
            "per_behavior": per_beh,
        }

    payload = {
        "regime": {
            "smoke": bool(args.smoke),
            "max_chunks": args.max_chunks,
            "n_chunks": len(chunk_files),
            "topctx_ks": list(TOPCTX_KS),
            "fingerprint": fp,
        },
        "ranking_convention": "descending projection (trait-promoting contexts first); "
        "projections computed as ((cx - xmu)/xsd) @ d with the FORWARD map's standardizer "
        "(the shared frame), directions unit-normalized",
        "layers": result,
        "wall_s": round(time.time() - t0, 1),
        "meta": _run_metadata("topctx", {"device": str(dev)}),
    }
    _write_json_atomic(out_root / RESULTS_SUBDIR / "topctx.json", payload)
    _write_sentinel(out_root, "topctx", "done", {"n_rows": int(n_rows)})
    _breadcrumb("topctx", status="done", n_rows=n_rows)


# ── phase: upload ────────────────────────────────────────────────────────────────


def phase_upload(args) -> None:
    from explore_persona_space.orchestrate import hub

    out_root = Path(args.out_root)
    layers = [int(x) for x in args.layers.split(",") if x]
    missing = []
    for ly in layers:
        for p in (
            out_root / "analysis_tensors" / "weights_rev" / f"L{ly}" / "ridge_rev.pt",
            out_root / "analysis_tensors" / "directions" / f"L{ly}_directions.npz",
            _layer_json_path(out_root, ly),
        ):
            if not p.is_file():
                missing.append(str(p))
    if not (out_root / RESULTS_SUBDIR / "topctx.json").is_file():
        missing.append(str(out_root / RESULTS_SUBDIR / "topctx.json"))
    if missing:
        raise RuntimeError(f"upload refused — expected artifacts missing: {missing}")

    prefix = f"{HF_UPLOAD_PREFIX}/smoke" if args.smoke else HF_UPLOAD_PREFIX
    path_in_repo = f"{prefix}/analysis_tensors"
    _breadcrumb("upload", status="start", dest=f"{HF_DATA_REPO}:{path_in_repo}")
    # ONE bulk upload_folder commit (never a per-file loop — 504-storm gotcha).
    base_url = hub._upload(
        out_root,
        HF_DATA_REPO,
        "dataset",
        path_in_repo,
        raise_on_error=True,
    )
    if not base_url:
        raise RuntimeError(
            f"upload returned no path for {HF_DATA_REPO}:{path_in_repo} — durability NOT verified"
        )
    _write_sentinel(out_root, "upload", "done", {"path_in_repo": path_in_repo})
    _breadcrumb("upload", status="done", dest=f"{HF_DATA_REPO}:{path_in_repo}")
    print("[phase=done] all phases complete", flush=True)


# ── pilot (sizing basis) ─────────────────────────────────────────────────────────


def run_pilot(args) -> None:
    """Measure per-unit walls: ONE chunk download, ONE 50k-row fp64 gram block
    on the device, ONE (H,H) fp64 eigh. Print extrapolated totals, exit 0."""
    dev = _dev(args.device)
    stage_dir = Path(args.stage_dir)
    rev, names = _resolve_pinned_revision(stage_dir)
    n_chunks = len(names)

    target0 = _chunk_dir(stage_dir) / names[0].rsplit("/", 1)[-1]
    pending = [n for n in names if not (_chunk_dir(stage_dir) / n.rsplit("/", 1)[-1]).is_file()]
    if pending:
        t = time.time()
        N1M._download_chunk_with_retry(HF_DATA_REPO, pending[0], stage_dir, revision=rev)
        t_chunk = time.time() - t
        chunk_note = f"measured on {pending[0].rsplit('/', 1)[-1]}"
    else:
        t_chunk = float("nan")
        chunk_note = f"all {n_chunks} chunks already staged — download wall not measurable"
    sz = target0.stat().st_size / 1e6 if target0.is_file() else float("nan")

    n_blocks = int(np.ceil((N_TRAIN_FORWARD + N1M.N_PASS_B) / N1M.RIDGE_BLOCK))
    xb = torch.randn(N1M.RIDGE_BLOCK, H_DIM, dtype=torch.float64, device=dev)
    yb = torch.randn(N1M.RIDGE_BLOCK, H_DIM, dtype=torch.float64, device=dev)
    _ = xb.T @ xb  # warmup
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    _ = xb.T @ xb
    _ = xb.T @ yb
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_block = time.time() - t
    a_sym = xb.T @ xb
    t = time.time()
    torch.linalg.eigh(a_sym)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_eigh = time.time() - t
    del xb, yb, a_sym

    stage_h = t_chunk * n_chunks / 3600 if np.isfinite(t_chunk) else float("nan")
    # per layer: standardizer pass + gram pass (~2 block sweeps) + one eigh;
    # host->device transfer per block is bounded above by ~t_block-scale IO.
    fits_core_h = (2 * n_blocks * t_block + t_eigh) * 3 / 3600
    print(f"[pilot] device={dev} n_chunks={n_chunks} chunk_MB={sz:.1f}", flush=True)
    print(f"[pilot] t_chunk_download={t_chunk:.2f}s ({chunk_note})", flush=True)
    print(f"[pilot] t_gram_block_50k={t_block:.2f}s t_eigh_H={t_eigh:.2f}s", flush=True)
    print(
        f"[pilot] extrapolated: stage={stage_h:.2f}h (serial); "
        f"fits gram+eigh core, 3 layers x {n_blocks} blocks x 2 passes = {fits_core_h:.2f}h "
        "(+ per-layer chunk-slice assembly IO, disk-bound, measure via stage/fits logs)",
        flush=True,
    )
    print(
        "[pilot] RAM: ~28 GB/layer resident (cx+vx fp32) with a transient "
        "~55 GB concat peak during assembly (parts + concatenated copies)",
        flush=True,
    )


# ── main ─────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=PHASES, help="phase to run (required unless --pilot)")
    ap.add_argument("--stage-dir", default=STAGE_DIR_DEFAULT, help="local staging root")
    ap.add_argument(
        "--out-root",
        default=None,
        help=f"output root (default {OUT_ROOT_DEFAULT}; {OUT_ROOT_SMOKE_DEFAULT} under --smoke)",
    )
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    ap.add_argument("--pass-b", type=Path, default=N1G.PASS_B_LOCAL)
    ap.add_argument("--orig-dir", type=Path, default=N1M.DEFAULT_ORIG_DIR)
    ap.add_argument("--max-chunks", type=int, default=None, help="cap the chunk universe")
    ap.add_argument("--smoke", action="store_true", help="3 chunks, tiny grids, smoke out-root")
    ap.add_argument("--pilot", action="store_true", help="measure per-unit walls, exit 0")
    ap.add_argument("--fresh-fits", action="store_true", help="recompute a regime-mismatched layer")
    ap.add_argument("--fresh-topctx", action="store_true", help="ignore the topctx checkpoint")
    ap.add_argument("--import-check", action="store_true", help="argparse+hub bind check, exit 0")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.smoke and args.max_chunks is None:
        args.max_chunks = SMOKE_MAX_CHUNKS
    if args.out_root is None:
        args.out_root = OUT_ROOT_SMOKE_DEFAULT if args.smoke else OUT_ROOT_DEFAULT
    if args.pilot:
        run_pilot(args)
        return 0
    if not args.phase:
        raise SystemExit("--phase is required (or --pilot / --import-check)")
    {"stage": phase_stage, "fits": phase_fits, "topctx": phase_topctx, "upload": phase_upload}[
        args.phase
    ](args)
    return 0


if __name__ == "__main__":
    # Explicit terminal exit: heavy C-extension imports (torch) can hit the
    # PyGILState_Release atexit race on bare interpreter falloff (gotchas.md).
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
