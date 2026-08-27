"""Pod-side context->answer joint-embedding pipeline for issue #779 (inline viz round).

Produces, on a dedicated RunPod pod (1x H100 ``eval`` intent; CPU fallback
throughout), a joint 2D-embedding + clustering + map-error dataset over the
#779 n1m final-token capture (cx_last + v_x at L19), with the banked mixed_1m
ridge applied READ-ONLY (vhat = ((cx - xmu)/xsd) @ W + ymu, the #2474
registered path), plus the #1739 sycophancy-labeling judged overlay
(context_end L19 -> same embedding, dv attached), a dimensionality battery
(P8 ``dim``), and a cross-layer vector-similarity battery (P9 ``xlayer``).
Exports compactly for VM-side figure/dashboard rendering to HF
``issue779_monitoring/ctxansviz/`` (smoke: ``issue779_monitoring/ctxansviz-smoke/``).

Disk venue (GPU pod): /workspace is MooseFS with a ~130 GB per-pod EDQUOT
quota (gotchas.md) — the two big pulls (capture chunks, the 52 GB labeling
tar) and all working memmaps live on a CONTAINER-LOCAL big root (probed at
runtime between /root and /tmp, choice logged); only small durables (export
shards, pilot JSONs, logs, pid, sentinel) live on /workspace. Device: cuda
when available for the chunked GEMM/cdist legs (cpu-vs-gpu parity probe at
first use, fail-loud), CPU BLAS otherwise; UMAP always CPU.

Phases (argparse ``--phase`` over PHASES; every phase checkpoints its output the
moment it completes; resume predicates key on GENERATING PARAMETERS — chunk-name
listings, layer, seeds, sample sizes — never hashes of recomputed float arrays):

  stage      ridge weights (L14/L19/L26) + sampling_manifest + raw_completions
             (small-file legs run in a subprocess with HF accelerators DISABLED —
             xet wedges on many-small-file storms) + the 52 GB sycophancy_labeling
             tar (accelerators ON — the file is over the 50 GB plain-path cap),
             selectively extracting ONLY row_index*/manifest/L19 summary members,
             then deleting the tar.
  assemble   stream capture chunks (download -> slice L19 -> append -> delete;
             peak ~one 43 MB chunk) into fp32 memmaps cx/vx [N, 3584] + ci, then
             join manifest context text + raw-completion answer text into
             row_meta_*.jsonl export shards (capped 280 chars, truncation-disclosed).
  predict    vhat via the banked L19 ridge (torch fp64 CPU, 50k-row blocks) +
             per-row metrics (cos/L2/normalized sqerr vs v_x; identity+bias
             companion cx + (ymu - xmu); cos(cx, v_x); norms).
  pca        joint PCA-100 (randomized SVD, seed 42) fit on a balanced cx/vx
             sample; transform ALL cx/vx/vhat (blocked).
  umap       timed pilot FIRST (fit+transform wall extrapolated; SystemExit(7)
             over budget), then UMAP fit (n_neighbors=15, min_dist=0.1,
             metric=cosine, random_state=42) on a joint PCA-100 sample and
             .transform() of ALL rows in 50k blocks (per-block checkpoints).
  cluster    MiniBatchKMeans K=50 seed 42 on PCA-100 of cx and of vx (full N);
             sklearn HDBSCAN on a PCA-100 subsample of cx (pilot-gated) with
             nearest-cluster-centroid assignment for the rest; silhouettes.
  judged     context_end L19 (+ t1 answer-side if present in the store) for the
             #1739 sycophancy labeling contexts -> standardize into the SAME
             PCA/UMAP embedding; join dv from the committed labeling.json.
  dim        (P8, scope extension) FULL exact PCA spectra per space
             {cx, vx, vhat} from chunked fp64 second moments + eigh
             (participation ratio, dims to 50/90/99%, log-log tail fit ranks
             10..1000), (cx, vx) CCA correlation spectrum (diag-regularized,
             top-500, descriptive — not a new predictor), and an ambient-3584d
             intrinsic-dimension battery per space {cx, vx, vhat, judged_cx}:
             TwoNN, Levina-Bickel MLE k=10/20 (mean-of-local-MLEs AND the
             MacKay-Ghahramani (k-2)/mean(s) form, both named in the export),
             Grassberger-Procaccia correlation dimension, and local-PCA ID
             (500 anchors, k=100, 90% variance), at n in {5k, 20k, 50k} x 5
             subsample draws (judged_cx: {5k, full}).
  xlayer     (P9, scope extension) cross-layer VECTOR similarity. n1m tier
             (layers 14/19/26, all rows, ONE extra chunk stream — each chunk
             packs all 3 layers): 6x6 linear CKA over {cx,vx}x{14,19,26} from
             fp64 means + cross moments (rotation-invariant), per-row cosine
             histograms for 9 named pairs (raw + global-mean-centered;
             meaningful because all layers share the d=3584 residual-stream
             basis), and a row-shuffled null (seed 42) computed on a retained
             deterministic 100k-row subsample (disclosed). 28-layer tier from
             the #1739 store: 28x28 context_end CKA + adjacent-layer per-row
             cosine curve (answer-side 28-layer coverage does not exist in
             that store — recorded as a limitation, never substituted).
  export     coords npz + cluster_stats.json + dim/xlayer artifacts + walltime
             log + meta.json (git provenance, params, row counts, per-file
             sha256) -> ONE upload_folder commit to HF; results sentinel
             written LAST.

Split semantics (the #779 fits split, reproduced): the sha-pinned val/test rows
of ``fixed_split(5000, 3600, 400, 1000, 42)`` are pass_b rows (combined-array
indices < 5000) and are NOT part of the n1m capture; the banked mixed_1m point
trained on the WHOLE pool (n_train 963,444 = 3,600 orig-train + 959,844 n1m
rows — issue2474_n1m_map.N1M_PROVENANCE). Every n1m row is therefore
split="train" / in_sample=True by construction; the identity gates below assert
exactly that arithmetic at full scale.

Smoke blind-spot enumeration:
  - full-corpus identity gates (realized rows == 959,844; per-corpus counts ==
    525,485 lmsys / 434,359 wildchat; judged-context set == labeling.json) are
    evaluated as informational LOG lines under --smoke (production-n-calibrated
    verdicts are structurally unsatisfiable at smoke n; gotchas.md
    smoke/production GATE CALIBRATION) — production asserts are byte-identical.
  - the chunk universe is capped to the first --smoke-chunks capture chunks
    (an lmsys-region subset of the corpus) and judged rows to --smoke-judged
    labeling contexts; sample/cluster/ID-battery/xlayer-subsample size knobs
    scale down (values only; estimator k values stay production).
  Every code path — staging, streaming, predict, PCA/UMAP/cluster/judged/
  dim/xlayer/export, upload — is the production implementation in both modes.

Refusal-safety: context/rollout TEXT (manifest prompts, raw responses, chunk
``prompts`` fields) is NEVER printed or logged — text lands only in the export
JSONL payloads (ensure_ascii=False, capped via cap_text). Do not add such
logging.

Memory: peak resident stays far under 128 GB — one 43 MB chunk during assemble;
~6 GB of fp64 block temporaries during predict; ~12 GB for the PCA fit sample +
randomized-SVD workspace; memmaps carry everything else on disk.

Designed halts: SystemExit(7) = pilot-gate refusal (UMAP/HDBSCAN wall budget),
with the measured pilot JSON already written; any other non-zero rc is a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Thread caps land BEFORE numpy/torch import (#847); the launcher's explicit env wins.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_ctxansviz")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"  # issue779_common.HF_DATA_REPO
N1M_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m"
CAPTURE_PREFIX = f"{N1M_PREFIX}/final_token_capture"  # shardSS_chunkCCCC.pt (n, 3, H) bundles
MANIFEST_PREFIX = f"{N1M_PREFIX}/sampling_manifest"  # part_*.jsonl + meta.json
RAW_PREFIX = f"{N1M_PREFIX}/raw_completions"  # shardSS_chunkCCCC.json + shardSS_skipped.json
WEIGHTS_PREFIX = "issue779_monitoring/n1m_readout/weights"  # L{14,19,26}/ridge.pt
LABELING_TAR = "issue1739_ctxmap/capture_store/sycophancy_labeling/sycophancy_labeling.tar"
EXPORT_PREFIX_FULL = "issue779_monitoring/ctxansviz"
EXPORT_PREFIX_SMOKE = "issue779_monitoring/ctxansviz-smoke"

LAYER = 19
H_DIM = 3584
CAPTURE_LAYERS = (14, 19, 26)  # issue779_ffc_n1m_generate_capture.CAPTURE_LAYERS
RIDGE_LAYERS = (14, 19, 26)
SEED = 42
BLOCK = 50_000

# mixed_1m provenance (issue2474_n1m_map.N1M_PROVENANCE): full-pool train.
N1M_N_TRAIN = 963_444
N_ORIG_TRAIN = 3_600  # pass_b train rows inside the fits' combined array
EXPECTED_ROWS = N1M_N_TRAIN - N_ORIG_TRAIN  # 959,844 realized n1m capture rows
EXPECTED_LMSYS = 529_085 - N_ORIG_TRAIN  # 525,485 (orig-train rows are lmsys)
EXPECTED_WILDCHAT = 434_359
WHOLE_MAP_R2_L19 = 0.7541708417500051  # informational sanity anchor, never asserted

TEXT_CAP = 280
METRIC_NAMES = (
    "cos_vhat_vx",
    "l2_vhat_vx",
    "sqerr_norm",  # ||vhat - vx||^2 / sum_j var(vx[:, j])  (per-row R2-style contribution)
    "cos_ib_vx",  # identity+bias companion: cos(cx + (ymu - xmu), vx)
    "l2_ib_vx",
    "cos_cx_vx",
    "norm_cx",
    "norm_vx",
    "norm_vhat",
)

DV_LABELING = PROJECT_ROOT / "eval_results/issue_1739/dv_dataset/sycophancy/labeling.json"

# Selective tar extraction: only the members the judged overlay needs. Kind
# aliases per issue_1739 store_io.REALIZED_KIND_FOR (context_end<->context_k,
# t1<->answer_k_t1).
STORE_MEMBER_RE = re.compile(
    r"^(?:"
    r"(?:context_end|context_k)_L\d{2}(?:_shard\d+)?\.npy"  # ALL layers (P6 L19 + P9 28-layer)
    r"|(?:t1|answer_k_t1)_L19(?:_shard\d+)?\.npy"
    r"|row_index.*\.jsonl"
    r"|manifest\.jsonl"
    r"|meta.*\.json"
    r")$"
)


def cap_text(s: str, n: int) -> str:
    """Excerpt cap with the inline truncation disclosure (verbatim from
    scripts/issue2202_labels.py::cap_text — copied to keep the pod import
    chain free of that module's unrelated constants)."""
    s = s or ""
    return s if len(s) <= n else s[:n] + " …[truncated]"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:  # process-unique temp name (#2329/#2336)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)


def iter_jsonl(path: Path):
    """Text-mode iteration — never ``splitlines()`` (#950: U+2028 in real-user text)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def assert_headroom(need_bytes: int, at: Path, what: str) -> None:
    free = shutil.disk_usage(at).free
    if free < 1.5 * need_bytes:
        raise RuntimeError(
            f"disk headroom too low for {what}: need 1.5x{need_bytes / 1e9:.1f} GB, "
            f"free {free / 1e9:.1f} GB at {at}"
        )
    logger.info(
        "[disk] %s: need ~%.1f GB, free %.1f GB at %s", what, need_bytes / 1e9, free / 1e9, at
    )


# ── roots + phase state ──────────────────────────────────────────────────────────

_BIG_BASE_CACHE: Path | None = None
_DEVICE: torch.device | None = None


def durable_root(args) -> Path:
    """Small durable outputs (export shards, pilot JSONs, sentinel fallback).
    On a pod this lives on /workspace; heavy staging + working arrays do NOT
    (MooseFS ~130 GB per-pod EDQUOT quota, gotchas.md)."""
    if args.out_root:
        return Path(args.out_root)
    base = Path("/workspace") if Path("/workspace").exists() else Path.cwd()
    return base / ("ctxansviz-smoke" if args.smoke else "ctxansviz")


def _big_base(args) -> Path:
    """Container-local big root (staging + memmaps). Probes candidate
    filesystems and picks the one with the most free bytes; both probes and
    the choice are logged. Overridable via --big-root."""
    global _BIG_BASE_CACHE
    if _BIG_BASE_CACHE is not None:
        return _BIG_BASE_CACHE
    if args.big_root:
        chosen = Path(args.big_root)
    elif Path("/workspace").exists():
        cands = [Path("/root/ctxansviz-big"), Path("/tmp/ctxansviz-big")]
        frees: dict[Path, int] = {}
        for c in cands:
            c.mkdir(parents=True, exist_ok=True)
            frees[c] = shutil.disk_usage(c).free
        chosen = max(cands, key=lambda c: frees[c])
        logger.info(
            "[disk] big-root probe: %s | /workspace free %.1f GB (big pulls NOT staged there)",
            ", ".join(f"{c}={frees[c] / 1e9:.1f}GB" for c in cands),
            shutil.disk_usage("/workspace").free / 1e9,
        )
    else:
        chosen = durable_root(args).parent / (durable_root(args).name + "-big")
    chosen.mkdir(parents=True, exist_ok=True)
    logger.info("[disk] big root = %s (free %.1f GB)", chosen, shutil.disk_usage(chosen).free / 1e9)
    _BIG_BASE_CACHE = chosen
    return chosen


def big_leg_root(args) -> Path:
    # Per-leg working roots: smoke never shares resume state with the full run.
    return _big_base(args) / ("smoke" if args.smoke else "full")


def stage_root(args) -> Path:
    # Staged INPUTS are mode-independent (read-only mirrors, per-file idempotent
    # skips), so smoke and full share the stage dir; the tar is downloaded once.
    return Path(args.stage_root) if args.stage_root else _big_base(args) / "stage"


def state_dir(args) -> Path:
    return big_leg_root(args) / "state"


def _cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """cdist pinned to the exact pairwise-euclid kernel: the default mm-based
    expansion (x^2+y^2-2xy) loses fp32 precision on GPU (smoke parity probe
    measured max|d|=2.07e-2 vs the 1e-3 tolerance on the H100), and these
    distances feed kNN ranks (TwoNN/MLE) where near-tie flips matter."""
    return torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")


def compute_device() -> torch.device:
    """cuda when available (1x H100 venue), else CPU BLAS. First cuda
    resolution disables TF32 (tensor-core fp32 matmul breaks cpu-vs-gpu
    parity at ~1e-2) and runs a cpu-vs-gpu numerics parity probe (fp64 GEMM
    + fp32 exact cdist), failing loud on mismatch — never silent drift."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if not torch.cuda.is_available():
        _DEVICE = torch.device("cpu")
        logger.info("[device] cuda unavailable — CPU BLAS path")
        return _DEVICE
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(SEED)
    a = torch.randn(256, 512, dtype=torch.float64, generator=g)
    b = torch.randn(512, 384, dtype=torch.float64, generator=g)
    d64 = float(((a @ b) - (a.to(dev) @ b.to(dev)).cpu()).abs().max())
    x = torch.randn(128, 512, generator=g)
    d32 = float((_cdist(x, x) - _cdist(x.to(dev), x.to(dev)).cpu()).abs().max())
    if d64 > 1e-9 or d32 > 1e-3:
        raise RuntimeError(
            f"cuda parity probe FAILED: fp64 GEMM max|d|={d64:.2e}, fp32 cdist max|d|={d32:.2e}"
        )
    logger.info(
        "[device] cuda parity probe PASS (fp64 GEMM max|d|=%.2e, fp32 cdist max|d|=%.2e)", d64, d32
    )
    _DEVICE = dev
    return dev


def phase_key(args, extra: dict | None = None) -> dict:
    key = {
        "layer": LAYER,
        "seed": SEED,
        "smoke": bool(args.smoke),
        "smoke_chunks": int(args.smoke_chunks) if args.smoke else None,
        "smoke_judged": int(args.smoke_judged) if args.smoke else None,
    }
    key.update(extra or {})
    return key


def phase_done(args, name: str, key: dict) -> bool:
    p = state_dir(args) / f"{name}.done.json"
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if rec.get("key") != key:
        logger.warning("[%s] done-sentinel present but key MISMATCHED; re-running", name)
        return False
    logger.info("[%s] done-sentinel matches generating parameters; skipping", name)
    return True


def mark_done(args, name: str, key: dict, t0: float, extra: dict | None = None) -> None:
    write_json_atomic(
        state_dir(args) / f"{name}.done.json",
        {"key": key, "elapsed_s": round(time.time() - t0, 1), "ts": _utc(), **(extra or {})},
    )


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── HF listing + download helpers ────────────────────────────────────────────────


def list_prefix(prefix: str) -> list:
    """Regular files under an HF prefix (retry-wrapped; lazy generator
    materialized INSIDE the thunk so iteration-time errors retry)."""
    from huggingface_hub import HfApi

    return hub.retry_transient(
        lambda: [
            f
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
            for f in HfApi().list_repo_tree(
                HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
            )
            if getattr(f, "size", None) is not None
        ],
        what=f"listing {prefix}",
    )


def dl(filename: str, local_dir: Path, what: str) -> Path:
    """hf_hub_download through the canonical transient-retry envelope. Files
    land at ``local_dir/<repo-relative path>`` (mirror-root semantics)."""
    from huggingface_hub import hf_hub_download

    return Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                HF_DATA_REPO, filename=filename, repo_type="dataset", local_dir=local_dir
            ),
            what=what,
            max_attempts=4,
        )
    )


def capture_chunk_names(args) -> list[str]:
    names = sorted(
        f.path.rsplit("/", 1)[-1] for f in list_prefix(CAPTURE_PREFIX) if f.path.endswith(".pt")
    )
    if not names:
        raise FileNotFoundError(f"no capture chunks under HF {CAPTURE_PREFIX}")
    if args.smoke:
        names = names[: int(args.smoke_chunks)]
    return names


def universe_fp(names: list[str]) -> str:
    # Chunk-NAME listing sha — strings from the Hub listing, machine-stable
    # (never a hash of recomputed float arrays, #1336).
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


# ── phase: stage-small (subprocess leg — HF accelerators disabled in env) ────────


def phase_stage_small(args) -> None:
    """Manifest parts + per-chunk raw-completion JSONs (the many-small-file
    storm legs). Run as a SUBPROCESS of ``stage`` with HF_HUB_DISABLE_XET=1 +
    HF_HUB_ENABLE_HF_TRANSFER=0 in the child env (accelerator failure matrix:
    xet wedges on small-file storms; the parent keeps accelerators ON for the
    52 GB tar). Per-file idempotent skips make this resumable and mode-safe."""
    sroot = stage_root(args)
    man_dir = sroot / "sampling_manifest"
    man_dir.mkdir(parents=True, exist_ok=True)

    man_files = list_prefix(MANIFEST_PREFIX)
    for f in man_files:
        base = f.path.rsplit("/", 1)[-1]
        target = man_dir / base
        if target.exists() and target.stat().st_size == f.size:
            continue
        got = dl(f.path, sroot / "_man_dl", f"manifest part {base}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, target)
    meta = json.loads((man_dir / "meta.json").read_text(encoding="utf-8"))
    n_parts = int(meta["n_parts"])
    have = len(list(man_dir.glob("part_*.jsonl")))
    if have != n_parts:
        raise RuntimeError(f"manifest incomplete: {have} parts staged, meta.json says {n_parts}")
    logger.info("[stage-small] manifest complete: %d parts", n_parts)

    raw_dir = sroot / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = capture_chunk_names(args)
    raw_sizes = {
        f.path.rsplit("/", 1)[-1]: f.size
        for f in list_prefix(RAW_PREFIX)
        if not f.path.endswith("_skipped.json")
    }
    got_n = 0
    t0 = time.time()
    for i, nm in enumerate(names):
        raw_name = nm[: -len(".pt")] + ".json"
        if raw_name not in raw_sizes:
            raise RuntimeError(f"capture chunk {nm} has no raw-completions twin on HF")
        target = raw_dir / raw_name
        if target.exists() and target.stat().st_size == raw_sizes[raw_name]:
            continue
        got = dl(f"{RAW_PREFIX}/{raw_name}", sroot / "_raw_dl", f"raw {raw_name}")
        os.replace(got, target)
        got_n += 1
        if (i + 1) % 100 == 0:
            logger.info(
                "[phase=stage-small] block %d/%d raw=%s elapsed=%.1fs",
                i + 1,
                len(names),
                raw_name,
                time.time() - t0,
            )
    logger.info("[stage-small] raw completions staged: %d fetched, %d total", got_n, len(names))


# ── phase: stage ─────────────────────────────────────────────────────────────────


def _extract_store_members(tar_path: Path, dest: Path) -> list[str]:
    """Stream the labeling tar once, extracting ONLY the members the judged
    overlay reads (row_index*/manifest/meta + L19 summary npy). Member paths are
    safety-checked (no absolute paths, no ``..``)."""
    kept: list[str] = []
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = member.name.rsplit("/", 1)[-1]
            if not STORE_MEMBER_RE.match(base):
                continue
            if member.name.startswith(("/", "..")) or ".." in Path(member.name).parts:
                raise RuntimeError(f"unsafe tar member path: {member.name}")
            tar.extract(member, dest)
            kept.append(member.name)
    if not kept:
        raise RuntimeError(
            f"no store members matched {STORE_MEMBER_RE.pattern} in {tar_path} — "
            "the labeling store layout drifted; re-inspect the tar"
        )
    return kept


def phase_stage(args) -> None:
    sroot = stage_root(args)
    sroot.mkdir(parents=True, exist_ok=True)

    # (1) ridge weights (small).
    for layer in RIDGE_LAYERS:
        rel = f"{WEIGHTS_PREFIX}/L{layer}/ridge.pt"
        target = sroot / "weights_dl" / rel
        if not target.exists():
            dl(rel, sroot / "weights_dl", f"ridge L{layer}")
        if not target.exists():
            raise RuntimeError(f"ridge weight staging landed off-path: expected {target}")
    logger.info("[stage] ridge weights staged for layers %s", list(RIDGE_LAYERS))

    # (2) manifest + raw completions in a child process with accelerators OFF.
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "stage-small",
        "--stage-root",
        str(sroot),
        "--out-root",
        str(durable_root(args)),
        "--big-root",
        str(_big_base(args)),
    ]
    if args.smoke:
        cmd += ["--smoke", "--smoke-chunks", str(args.smoke_chunks)]
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    logger.info("[stage] spawning stage-small subprocess (HF accelerators disabled)")
    subprocess.run(cmd, env=env, check=True)

    # (3) labeling store: selective extraction sentinel checked BEFORE the tar
    #     download so a completed extraction never re-pulls 52 GB.
    extract_dir = sroot / "labeling_store"
    sentinel = sroot / "labeling_store.extracted.json"
    if sentinel.exists():
        rec = json.loads(sentinel.read_text(encoding="utf-8"))
        if rec.get("member_re") != STORE_MEMBER_RE.pattern:
            logger.warning("[stage] extraction sentinel regex MISMATCHED; re-extracting")
            rec = {"members": ["<forced-miss>"]}
        missing = [m for m in rec.get("members", []) if not (extract_dir / m).exists()]
        if not missing:
            logger.info(
                "[stage] labeling store already extracted (%d members); skipping tar",
                len(rec.get("members", [])),
            )
            return
        logger.warning("[stage] extraction sentinel present but %d members missing", len(missing))

    tar_files = [f for f in list_prefix(LABELING_TAR.rsplit("/", 1)[0]) if f.path == LABELING_TAR]
    if len(tar_files) != 1:
        raise RuntimeError(f"expected exactly one tar at {LABELING_TAR}, found {len(tar_files)}")
    tar_size = int(tar_files[0].size)
    local_tar = sroot / LABELING_TAR
    if not (local_tar.exists() and local_tar.stat().st_size == tar_size):
        assert_headroom(tar_size, sroot, "labeling tar download")
        # Accelerators stay ON here: the tar exceeds the plain download path's
        # 50e9-byte cap (gotchas.md HF accelerator failure matrix — big-file leg).
        dl(LABELING_TAR, sroot, "sycophancy_labeling tar (52 GB)")
    members = _extract_store_members(local_tar, extract_dir)
    write_json_atomic(
        sentinel,
        {
            "members": members,
            "member_re": STORE_MEMBER_RE.pattern,
            "tar_sha_bytes": tar_size,
            "ts": _utc(),
        },
    )
    local_tar.unlink()  # frees 52 GB; sentinel + members are the durable record
    logger.info("[stage] extracted %d store members; tar deleted", len(members))


# ── phase: assemble ──────────────────────────────────────────────────────────────


def _open_or_create_memmap(path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    if path.exists():
        return np.lib.format.open_memmap(path, mode="r+")
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def arrays_dir(args) -> Path:
    return big_leg_root(args) / "arrays"


def load_stream_state(args) -> dict:
    p = state_dir(args) / "assemble_cursor.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def phase_assemble(args) -> None:
    names = capture_chunk_names(args)
    fp = universe_fp(names)
    key = phase_key(args, {"universe_fp": fp, "n_chunks": len(names), "text_cap": TEXT_CAP})
    if phase_done(args, "assemble", key):
        return
    t0 = time.time()
    adir = arrays_dir(args)
    capacity = len(names) * 500  # ROWS_PER_CHUNK_EST — realized rows recorded in cursor
    cx = _open_or_create_memmap(adir / "cx_L19.npy", (capacity, H_DIM), np.float32)
    vx = _open_or_create_memmap(adir / "vx_L19.npy", (capacity, H_DIM), np.float32)
    ci = _open_or_create_memmap(adir / "ci.npy", (capacity,), np.int64)
    if cx.shape != (capacity, H_DIM):
        raise RuntimeError(
            f"existing memmap shape {cx.shape} != required {(capacity, H_DIM)} — "
            "chunk universe changed; move the arrays dir aside"
        )

    cur = load_stream_state(args)
    if cur.get("universe_fp") != fp:
        if cur:
            logger.warning("[assemble] cursor MISMATCHED (universe changed); re-streaming")
        cur = {"universe_fp": fp, "chunks_done": 0, "n_rows": 0}
    start, n_rows = int(cur["chunks_done"]), int(cur["n_rows"])
    cache = stage_root(args) / "chunk_cache"
    cache.mkdir(parents=True, exist_ok=True)
    col = list(CAPTURE_LAYERS).index(LAYER)

    for i in range(start, len(names)):
        got = dl(f"{CAPTURE_PREFIX}/{names[i]}", cache, f"chunk {names[i]}")
        b = torch.load(got, mmap=True, weights_only=False, map_location="cpu")
        for k in ("cx_last", "v_x", "ci", "layers"):
            if k not in b:
                raise RuntimeError(f"chunk {names[i]} missing key {k!r}: has {sorted(b)}")
        if list(b["layers"]) != list(CAPTURE_LAYERS):
            col = list(b["layers"]).index(LAYER)
        rows_cx = b["cx_last"][:, col, :].to(torch.float32).numpy()
        rows_vx = b["v_x"][:, col, :].to(torch.float32).numpy()
        rows_ci = np.asarray([int(x) for x in b["ci"]], dtype=np.int64)
        n = rows_cx.shape[0]
        assert rows_cx.shape == rows_vx.shape == (n, H_DIM), (rows_cx.shape, rows_vx.shape)
        if n_rows + n > capacity:
            raise RuntimeError(f"capacity {capacity} exceeded at chunk {names[i]}")
        cx[n_rows : n_rows + n] = rows_cx
        vx[n_rows : n_rows + n] = rows_vx
        ci[n_rows : n_rows + n] = rows_ci
        n_rows += n
        del b
        got.unlink()
        write_json_atomic(
            state_dir(args) / "assemble_cursor.json",
            {"universe_fp": fp, "chunks_done": i + 1, "n_rows": n_rows},
        )
        if (i + 1) % 100 == 0 or i + 1 == len(names):
            logger.info(
                "[phase=assemble] block %d/%d rows=%d elapsed=%.1fs",
                i + 1,
                len(names),
                n_rows,
                time.time() - t0,
            )
    cx.flush()
    vx.flush()
    ci.flush()

    # Row metadata + text join (export shards written here so the row order is
    # fixed once, at assembly time).
    ci_v = np.asarray(ci[:n_rows])
    man_rows: dict[int, tuple[str, str]] = {}
    man_dir = stage_root(args) / "sampling_manifest"
    for part in sorted(man_dir.glob("part_*.jsonl")):
        for r in iter_jsonl(part):
            for k in ("i", "prompt", "corpus"):
                if k not in r:
                    raise RuntimeError(f"manifest row missing key {k!r}: has {sorted(r)}")
            man_rows[int(r["i"])] = (str(r["corpus"]), str(r["prompt"]))
    raw_map: dict[int, str] = {}
    raw_dir = stage_root(args) / "raw_completions"
    for nm in names:
        p = raw_dir / (nm[: -len(".pt")] + ".json")
        rec = json.loads(p.read_text(encoding="utf-8"))
        for r in rec["rows"]:
            raw_map[int(r["ci"])] = str(r["response"])

    export = export_dir(args)
    export.mkdir(parents=True, exist_ok=True)
    for old in export.glob("row_meta_*.jsonl"):
        old.unlink()
    n_lmsys = n_wild = 0
    part_idx, buf, buf_bytes = 0, [], 0
    t1 = time.time()

    def _flush() -> None:
        nonlocal part_idx, buf, buf_bytes
        if not buf:
            return
        (export / f"row_meta_{part_idx:05d}.jsonl").write_text("".join(buf), encoding="utf-8")
        part_idx += 1
        buf, buf_bytes = [], 0

    for idx in range(n_rows):
        c = int(ci_v[idx])
        if c not in man_rows:
            raise RuntimeError(f"captured ci {c} absent from sampling manifest")
        if c not in raw_map:
            raise RuntimeError(f"captured ci {c} absent from raw completions")
        corpus, prompt = man_rows[c]
        n_lmsys += corpus == "lmsys"
        n_wild += corpus == "wildchat"
        line = (
            json.dumps(
                {
                    "row": idx,
                    "ci": c,
                    "corpus": corpus,
                    "split": "train",
                    "in_sample": True,
                    "context_text": cap_text(prompt, TEXT_CAP),
                    "answer_text": cap_text(raw_map[c], TEXT_CAP),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        buf.append(line)
        buf_bytes += len(line.encode("utf-8"))
        if buf_bytes >= 8_500_000:  # <9 MB non-LFS text shards
            _flush()
        if (idx + 1) % BLOCK == 0:
            logger.info(
                "[phase=assemble] block %d/%d meta rows elapsed=%.1fs",
                idx + 1,
                n_rows,
                time.time() - t1,
            )
    _flush()

    # Full-corpus identity gates (informational under --smoke; see the module
    # docstring's smoke blind-spot enumeration).
    checks = [
        ("n_rows", n_rows, EXPECTED_ROWS),
        ("n_lmsys", n_lmsys, EXPECTED_LMSYS),
        ("n_wildchat", n_wild, EXPECTED_WILDCHAT),
    ]
    for what, got_v, want in checks:
        if args.smoke:
            logger.info(
                "[assemble] identity gate (smoke, informational): %s=%d (full corpus expects %d)",
                what,
                got_v,
                want,
            )
        elif got_v != want:
            raise RuntimeError(f"identity gate FAILED: {what}={got_v}, expected {want}")
    if len(set(ci_v.tolist())) != n_rows:
        raise RuntimeError("duplicate ci in assembled capture rows")
    mark_done(
        args,
        "assemble",
        key,
        t0,
        {"n_rows": n_rows, "n_lmsys": n_lmsys, "n_wildchat": n_wild, "meta_parts": part_idx},
    )


def realized_rows(args) -> int:
    cur = load_stream_state(args)
    if not cur or not cur.get("n_rows"):
        raise RuntimeError("assemble has not run — no realized row count")
    return int(cur["n_rows"])


def mm(args, name: str, n: int | None = None) -> np.memmap:
    arr = np.load(arrays_dir(args) / name, mmap_mode="r")
    return arr[: realized_rows(args)] if n is None else arr[:n]


# ── phase: predict ───────────────────────────────────────────────────────────────


def load_ridge_payload(args, layer: int) -> dict:
    path = stage_root(args) / "weights_dl" / WEIGHTS_PREFIX / f"L{layer}" / "ridge.pt"
    # Self-produced sha-pinned bundle from our own HF repo (the #2474 loader
    # uses the same weights_only=False form).
    p = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(p, dict):
        raise RuntimeError(f"{path}: payload is {type(p)}, expected dict")
    for k in ("W", "xmu", "xsd", "ymu"):
        if k not in p:
            raise RuntimeError(f"{path}: missing key {k!r}; has {sorted(p)}")
    kind = str(p.get("kind", p.get("fitter", "")))
    if "ridge" not in kind:
        raise RuntimeError(f"{path}: kind/fitter={kind!r}, expected ridge")
    if "layer" in p and int(p["layer"]) != layer:
        raise RuntimeError(f"{path}: payload layer {p['layer']} != requested {layer}")
    W = torch.as_tensor(p["W"], dtype=torch.float64)
    if W.shape != (H_DIM, H_DIM):
        raise RuntimeError(f"{path}: W shape {tuple(W.shape)} != {(H_DIM, H_DIM)}")
    comp = {"W": W}
    for k in ("xmu", "xsd", "ymu"):
        v = torch.as_tensor(p[k], dtype=torch.float64).reshape(-1)
        if v.shape != (H_DIM,):
            raise RuntimeError(f"{path}: {k} shape {tuple(v.shape)} != ({H_DIM},)")
        comp[k] = v
    return comp


def phase_predict(args) -> None:
    n = realized_rows(args)
    key = phase_key(args, {"n_rows": n, "metrics": list(METRIC_NAMES)})
    if phase_done(args, "predict", key):
        return
    t0 = time.time()
    dev = compute_device()
    comp = {k: v.to(dev) for k, v in load_ridge_payload(args, LAYER).items()}
    cx, vx = mm(args, "cx_L19.npy"), mm(args, "vx_L19.npy")
    adir = arrays_dir(args)

    # Pass A: global per-dim variance of vx (population, ddof=0) for the
    # per-row normalized-sqerr metric.
    s = torch.zeros(H_DIM, dtype=torch.float64, device=dev)
    ss = torch.zeros(H_DIM, dtype=torch.float64, device=dev)
    for lo in range(0, n, BLOCK):
        yb = torch.as_tensor(np.asarray(vx[lo : lo + BLOCK]), dtype=torch.float64).to(dev)
        s += yb.sum(0)
        ss += (yb * yb).sum(0)
    var_sum = float((ss / n - (s / n) ** 2).clamp(min=0.0).sum())
    logger.info("[predict] sum_j var(vx_j) = %.4f over n=%d", var_sum, n)

    vhat = _open_or_create_memmap(adir / "vhat_L19.npy", (n, H_DIM), np.float32)
    met = _open_or_create_memmap(adir / "metrics.npy", (n, len(METRIC_NAMES)), np.float32)
    cur_p = state_dir(args) / "predict_cursor.json"
    start = 0
    if cur_p.exists():
        c = json.loads(cur_p.read_text(encoding="utf-8"))
        if c.get("n_rows") == n and c.get("var_sum") is not None:
            start = int(c["done_rows"])
    ib_shift = comp["ymu"] - comp["xmu"]  # identity+bias companion (map's own shift, #2474)
    n_blocks = (n + BLOCK - 1) // BLOCK
    sqerr_total = 0.0
    for lo in range(start, n, BLOCK):
        hi = min(lo + BLOCK, n)
        xb = torch.as_tensor(np.asarray(cx[lo:hi]), dtype=torch.float64).to(dev)
        yb = torch.as_tensor(np.asarray(vx[lo:hi]), dtype=torch.float64).to(dev)
        yh = ((xb - comp["xmu"]) / comp["xsd"]) @ comp["W"] + comp["ymu"]  # apply_map ridge path
        ib = xb + ib_shift
        eps = 1e-12

        def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a * b).sum(1) / ((a.norm(dim=1) * b.norm(dim=1)).clamp(min=eps))

        sqerr = ((yh - yb) ** 2).sum(1)
        sqerr_total += float(sqerr.sum())
        cols = [
            _cos(yh, yb),
            (yh - yb).norm(dim=1),
            sqerr / max(var_sum, eps),
            _cos(ib, yb),
            (ib - yb).norm(dim=1),
            _cos(xb, yb),
            xb.norm(dim=1),
            yb.norm(dim=1),
            yh.norm(dim=1),
        ]
        vhat[lo:hi] = yh.to(torch.float32).cpu().numpy()
        met[lo:hi] = torch.stack(cols, dim=1).to(torch.float32).cpu().numpy()
        write_json_atomic(cur_p, {"n_rows": n, "done_rows": hi, "var_sum": var_sum})
        logger.info(
            "[phase=predict] block %d/%d rows=%d elapsed=%.1fs",
            lo // BLOCK + 1,
            n_blocks,
            hi,
            time.time() - t0,
        )
    vhat.flush()
    met.flush()
    pseudo_r2 = 1.0 - sqerr_total / max(n * var_sum, 1e-12)
    logger.info(
        "[predict] in-sample pseudo-R2 = %.4f (held-out whole-map anchor %.4f, informational)",
        pseudo_r2,
        WHOLE_MAP_R2_L19,
    )
    mark_done(args, "predict", key, t0, {"pseudo_r2": pseudo_r2, "var_sum": var_sum})


# ── phase: pca ───────────────────────────────────────────────────────────────────


def sizes(args, n: int) -> dict:
    if args.smoke:
        return {
            "pca_fit_per_side": min(2_000, n),
            "umap_fit_per_side": min(1_000, n),
            "umap_pilot": min(1_000, n),
            "kmeans_k": 8,
            "hdbscan_sub": min(5_000, n),
            "silhouette_n": min(2_000, n),
            "id_scales": (500, 1_000),
            "id_scales_judged": (200, 500),
            "id_resamples": 2,
            "id_anchors": 50,
            "xlayer_sub": 2_000,
        }
    return {
        "pca_fit_per_side": min(200_000, n),
        "umap_fit_per_side": min(100_000, n),
        "umap_pilot": min(20_000, n),
        "kmeans_k": 50,
        "hdbscan_sub": min(150_000, n),
        "silhouette_n": min(20_000, n),
        "id_scales": (5_000, 20_000, 50_000),
        "id_scales_judged": (5_000, 17_304),
        "id_resamples": 5,
        "id_anchors": 500,
        "xlayer_sub": 100_000,
    }


PCA_DIM = 100


def phase_pca(args) -> None:
    n = realized_rows(args)
    sz = sizes(args, n)
    dim = min(PCA_DIM, 2 * sz["pca_fit_per_side"] - 1, H_DIM)
    key = phase_key(args, {"n_rows": n, "pca_dim": dim, "fit_per_side": sz["pca_fit_per_side"]})
    if phase_done(args, "pca", key):
        return
    t0 = time.time()
    from sklearn.decomposition import PCA

    cx, vx = mm(args, "cx_L19.npy"), mm(args, "vx_L19.npy")
    rng = np.random.default_rng(SEED)
    idx_c = np.sort(rng.choice(n, size=sz["pca_fit_per_side"], replace=False))
    idx_v = np.sort(rng.choice(n, size=sz["pca_fit_per_side"], replace=False))
    fit_X = np.concatenate([np.asarray(cx[idx_c]), np.asarray(vx[idx_v])]).astype(np.float32)
    logger.info("[pca] fitting PCA-%d on %s sample (joint cx+vx)", dim, fit_X.shape)
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=SEED)
    pca.fit(fit_X)
    del fit_X
    np.savez(
        arrays_dir(args) / "pca_model.npz",
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float64),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        n_fit_per_side=np.int64(sz["pca_fit_per_side"]),
    )
    adir = arrays_dir(args)
    vhat = mm(args, "vhat_L19.npy")
    dev = compute_device()
    comp_t = torch.as_tensor(pca.components_, dtype=torch.float32, device=dev)
    mean_t = torch.as_tensor(pca.mean_, dtype=torch.float32, device=dev)
    parity_checked = False
    for name, src in (("pca_cx.npy", cx), ("pca_vx.npy", vx), ("pca_vhat.npy", vhat)):
        dst = _open_or_create_memmap(adir / name, (n, dim), np.float32)
        for lo in range(0, n, BLOCK):
            hi = min(lo + BLOCK, n)
            xb = torch.as_tensor(np.asarray(src[lo:hi], dtype=np.float32), device=dev)
            out = ((xb - mean_t) @ comp_t.T).cpu().numpy()
            if not parity_checked:
                ref = pca.transform(np.asarray(src[lo : lo + 256], dtype=np.float32))
                d = float(np.abs(out[: ref.shape[0]] - ref).max())
                if d > 1e-3:
                    raise RuntimeError(f"pca transform device parity FAILED: max|d|={d:.2e}")
                logger.info("[pca] device transform parity vs sklearn: max|d|=%.2e", d)
                parity_checked = True
            dst[lo:hi] = out
        dst.flush()
        logger.info("[phase=pca] block done %s rows=%d elapsed=%.1fs", name, n, time.time() - t0)
    mark_done(args, "pca", key, t0, {"evr_sum": float(pca.explained_variance_ratio_.sum())})


def pca_transform_np(args, X: np.ndarray) -> np.ndarray:
    m = np.load(arrays_dir(args) / "pca_model.npz")
    return (X.astype(np.float32) - m["mean"]) @ m["components"].T


# ── phase: umap ──────────────────────────────────────────────────────────────────


def _umap_model(args, sz: dict):
    import umap

    # fixed random_state forces n_jobs=1 in umap-learn; the relax lever parallelizes
    n_jobs = (os.cpu_count() or 1) if args.umap_relax_seed else 1
    return umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        n_components=2,
        random_state=None if args.umap_relax_seed else SEED,
        n_jobs=n_jobs,
        verbose=False,
    )


def phase_umap(args) -> None:
    n = realized_rows(args)
    sz = sizes(args, n)
    key = phase_key(
        args,
        {
            "n_rows": n,
            "fit_per_side": sz["umap_fit_per_side"],
            "relax_seed": bool(args.umap_relax_seed),
            "params": {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
        },
    )
    if phase_done(args, "umap", key):
        return
    t0 = time.time()
    adir = arrays_dir(args)
    dim = np.load(adir / "pca_model.npz")["components"].shape[0]
    pcx = mm(args, "pca_cx.npy")
    pvx = mm(args, "pca_vx.npy")
    pvh = mm(args, "pca_vhat.npy")
    rng = np.random.default_rng(SEED + 1)

    # Timed pilot through the SAME entrypoint/params at the sweep's shape.
    pilot_n = sz["umap_pilot"]
    pilot_idx = rng.choice(n, size=pilot_n, replace=False)
    pilot_fit = np.asarray(pcx[np.sort(pilot_idx)], dtype=np.float32)
    tp = time.time()
    pilot_model = _umap_model(args, sz)
    pilot_model.fit(pilot_fit)
    fit_wall = time.time() - tp
    tp = time.time()
    trans_idx = np.sort(rng.choice(n, size=pilot_n, replace=False))
    pilot_model.transform(np.asarray(pvx[trans_idx], dtype=np.float32))
    trans_wall = time.time() - tp
    n_fit = 2 * sz["umap_fit_per_side"]
    n_trans = 3 * n  # cx + vx + vhat (judged rows are a rounding error)
    proj_fit = fit_wall * (n_fit / max(pilot_n, 1))
    proj_trans = (trans_wall / max(pilot_n, 1)) * n_trans
    proj_total = proj_fit + proj_trans
    pilot_rec = {
        "pilot_n": pilot_n,
        "fit_wall_s": round(fit_wall, 1),
        "transform_wall_s": round(trans_wall, 1),
        "projected_fit_s": round(proj_fit, 1),
        "projected_transform_s": round(proj_trans, 1),
        "projected_total_s": round(proj_total, 1),
        "budget_s": int(args.umap_wall_budget_s),
        "relax_seed": bool(args.umap_relax_seed),
        "n_fit": n_fit,
        "n_transform": n_trans,
    }
    write_json_atomic(durable_root(args) / "umap_pilot.json", pilot_rec)
    logger.info("[umap] pilot: %s", json.dumps(pilot_rec))
    if proj_total > float(args.umap_wall_budget_s):
        logger.error(
            "[umap] pilot gate REFUSED: projected %.0fs > budget %ds. Levers: "
            "--umap-relax-seed (n_jobs=16, non-reproducible embedding) or a larger "
            "--umap-wall-budget-s. Pilot JSON written to umap_pilot.json.",
            proj_total,
            int(args.umap_wall_budget_s),
        )
        raise SystemExit(7)

    fit_c = np.sort(rng.choice(n, size=sz["umap_fit_per_side"], replace=False))
    fit_v = np.sort(rng.choice(n, size=sz["umap_fit_per_side"], replace=False))
    fit_X = np.concatenate(
        [np.asarray(pcx[fit_c], dtype=np.float32), np.asarray(pvx[fit_v], dtype=np.float32)]
    )
    logger.info("[umap] fitting on %s joint sample", fit_X.shape)
    model = _umap_model(args, sz)
    model.fit(fit_X)
    del fit_X
    models_dir = big_leg_root(args) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(models_dir / "umap_model.pkl", "wb") as f:
        pickle.dump(model, f)

    cur_p = state_dir(args) / "umap_cursor.json"
    done: dict[str, int] = {}
    if cur_p.exists():
        c = json.loads(cur_p.read_text(encoding="utf-8"))
        if c.get("n_rows") == n and c.get("relax_seed") == bool(args.umap_relax_seed):
            done = c.get("done", {})
    n_blocks = (n + BLOCK - 1) // BLOCK
    for name, src in (("umap_cx.npy", pcx), ("umap_vx.npy", pvx), ("umap_vhat.npy", pvh)):
        dst = _open_or_create_memmap(adir / name, (n, 2), np.float32)
        start = int(done.get(name, 0))
        for lo in range(start, n, BLOCK):
            hi = min(lo + BLOCK, n)
            dst[lo:hi] = model.transform(np.asarray(src[lo:hi], dtype=np.float32))
            done[name] = hi
            write_json_atomic(
                cur_p, {"n_rows": n, "relax_seed": bool(args.umap_relax_seed), "done": done}
            )
            logger.info(
                "[phase=umap] block %d/%d %s elapsed=%.1fs",
                lo // BLOCK + 1,
                n_blocks,
                name,
                time.time() - t0,
            )
        dst.flush()
    mark_done(args, "umap", key, t0, {"pilot": pilot_rec, "dim": int(dim)})


# ── phase: cluster ───────────────────────────────────────────────────────────────


def _nearest_centroid(X: np.memmap | np.ndarray, cents: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.int32)
    c2 = (cents.astype(np.float64) ** 2).sum(1)
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        xb = np.asarray(X[lo:hi], dtype=np.float64)
        d = xb @ cents.T.astype(np.float64) * -2.0 + c2[None, :]
        out[lo:hi] = np.argmin(d, axis=1).astype(np.int32)
    return out


def phase_cluster(args) -> None:
    n = realized_rows(args)
    sz = sizes(args, n)
    key = phase_key(
        args, {"n_rows": n, "kmeans_k": sz["kmeans_k"], "hdbscan_sub": sz["hdbscan_sub"]}
    )
    if phase_done(args, "cluster", key):
        return
    t0 = time.time()
    from sklearn.cluster import HDBSCAN, MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    adir = arrays_dir(args)
    rng = np.random.default_rng(SEED + 2)
    stats: dict = {"kmeans_k": sz["kmeans_k"]}
    for side, name in (("cx", "pca_cx.npy"), ("vx", "pca_vx.npy")):
        X = mm(args, name)
        km = MiniBatchKMeans(
            n_clusters=sz["kmeans_k"], random_state=SEED, batch_size=10_000, n_init="auto"
        )
        km.fit(np.asarray(X))
        labels = _nearest_centroid(X, km.cluster_centers_, n)
        lab_mm = _open_or_create_memmap(adir / f"kmeans_{side}.npy", (n,), np.int32)
        lab_mm[:] = labels
        lab_mm.flush()
        np.savez(adir / f"kmeans_{side}_centroids.npz", centroids=km.cluster_centers_)
        sil_idx = np.sort(rng.choice(n, size=sz["silhouette_n"], replace=False))
        sil = float(silhouette_score(np.asarray(X[sil_idx], dtype=np.float32), labels[sil_idx]))
        stats[f"silhouette_kmeans_{side}"] = sil
        logger.info(
            "[phase=cluster] block kmeans_%s done silhouette=%.4f elapsed=%.1fs",
            side,
            sil,
            time.time() - t0,
        )

    # HDBSCAN on a cx subsample, pilot-gated (tree-based MST degrades in dim 100).
    X = mm(args, "pca_cx.npy")
    sub_idx = np.sort(rng.choice(n, size=sz["hdbscan_sub"], replace=False))
    mcs = max(30, round(sz["hdbscan_sub"] / 200))
    pilot_n = min(20_000, sz["hdbscan_sub"])
    tp = time.time()
    HDBSCAN(min_cluster_size=max(30, round(pilot_n / 200))).fit(
        np.asarray(X[sub_idx[:pilot_n]], dtype=np.float32)
    )
    pilot_wall = time.time() - tp
    ratio = sz["hdbscan_sub"] / max(pilot_n, 1)
    proj_nlogn = pilot_wall * ratio * (np.log(max(sz["hdbscan_sub"], 2)) / np.log(max(pilot_n, 2)))
    proj_n2 = pilot_wall * ratio**2
    logger.info(
        "[cluster] hdbscan pilot %.1fs at n=%d; projected full n=%d: %.0fs (nlogn) / %.0fs (n^2)",
        pilot_wall,
        pilot_n,
        sz["hdbscan_sub"],
        proj_nlogn,
        proj_n2,
    )
    if proj_n2 > 3 * 3600 and not args.force_hdbscan:
        write_json_atomic(
            durable_root(args) / "hdbscan_pilot.json",
            {"pilot_n": pilot_n, "pilot_wall_s": pilot_wall, "proj_n2_s": proj_n2},
        )
        logger.error(
            "[cluster] hdbscan pilot gate REFUSED: n^2 projection %.0fs > 3h. "
            "Lever: --force-hdbscan, or rerun with a smaller subsample.",
            proj_n2,
        )
        raise SystemExit(7)
    hdb = HDBSCAN(min_cluster_size=mcs)
    sub_labels = hdb.fit(np.asarray(X[sub_idx], dtype=np.float32)).labels_.astype(np.int32)
    uniq = sorted(int(u) for u in set(sub_labels.tolist()) if u >= 0)
    stats["hdbscan"] = {
        "subsample_n": int(sz["hdbscan_sub"]),
        "min_cluster_size": int(mcs),
        "n_clusters": len(uniq),
        "noise_frac_subsample": float((sub_labels < 0).mean()),
    }
    lab_mm = _open_or_create_memmap(adir / "hdbscan_cx.npy", (n,), np.int32)
    in_sub = _open_or_create_memmap(adir / "hdbscan_cx_insub.npy", (n,), np.int8)
    in_sub[:] = 0
    if uniq:
        sub_X = np.asarray(X[sub_idx], dtype=np.float64)
        cents = np.stack([sub_X[sub_labels == u].mean(axis=0) for u in uniq])
        np.savez(adir / "hdbscan_cx_centroids.npz", centroids=cents, labels=np.asarray(uniq))
        # Out-of-sample rows: nearest-cluster-centroid assignment (KMeans-style).
        lab_mm[:] = np.asarray(uniq, dtype=np.int32)[_nearest_centroid(X, cents, n)]
    else:
        logger.warning("[cluster] hdbscan found NO clusters (all noise) — labels stay -1")
        lab_mm[:] = -1
    # Subsample rows keep their own labels (incl. -1 noise).
    lab_mm[sub_idx] = sub_labels
    in_sub[sub_idx] = 1
    lab_mm.flush()
    in_sub.flush()
    write_json_atomic(state_dir(args) / "cluster_stats_partial.json", stats)
    mark_done(args, "cluster", key, t0, stats)


# ── phase: judged ────────────────────────────────────────────────────────────────


def _find_store_root(extract_dir: Path) -> Path:
    hits = {p.parent for p in extract_dir.rglob("*_L19*.npy")}
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one store root with L19 summary shards under {extract_dir}; "
            f"found {sorted(str(h) for h in hits)}"
        )
    return next(iter(hits))


def _labeling_by_id(args) -> dict[str, dict]:
    labeling = json.loads(DV_LABELING.read_text(encoding="utf-8"))
    lab_rows = labeling["rows"]
    if args.smoke:
        lab_rows = lab_rows[: int(args.smoke_judged)]
    logger.info("[judged] labeling rows: %d (of %d total)", len(lab_rows), len(labeling["rows"]))
    return {str(r["context_id"]): r for r in lab_rows}


def _judged_store_rows(args, meta, lab_by_id):
    """Dedupe store rows to the first row per context_id + the labeling join
    gate (informational under --smoke). Returns (keep_ids, row_indices, lab_by_id)."""
    id_key = _context_id_key(meta[0])
    first_row_for: dict[str, int] = {}
    for i, r in enumerate(meta):
        cid = str(r[id_key])
        if cid not in first_row_for:
            first_row_for[cid] = i
    store_ids = set(first_row_for)
    missing = sorted(set(lab_by_id) - store_ids)
    if missing:
        msg = f"{len(missing)} labeling contexts absent from the store (first: {missing[:5]})"
        if args.smoke:
            logger.info("[judged] identity gate (smoke, informational): %s", msg)
            lab_by_id = {k: v for k, v in lab_by_id.items() if k in store_ids}
        else:
            raise RuntimeError(f"identity gate FAILED: {msg}")
    keep_ids = sorted(lab_by_id)
    rows = np.asarray([first_row_for[c] for c in keep_ids], dtype=np.int64)
    return keep_ids, rows, lab_by_id


def _store_layers(root: Path) -> list[int]:
    """Layer ids with context-side summary shards on disk (probed, never assumed)."""
    layers = set()
    for f in root.glob("*.npy"):
        m = re.match(r"(?:context_end|context_k)_L(\d{2})(?:_shard\d+)?\.npy$", f.name)
        if m:
            layers.add(int(m.group(1)))
    if not layers:
        raise RuntimeError(f"no context-side summary shards under {root}")
    return sorted(layers)


def _context_id_key(meta_row: dict) -> str:
    for k in ("context_id", "ci", "id"):
        if k in meta_row:
            return k
    raise RuntimeError(f"no context-id key in store row_index rows; keys={sorted(meta_row)}")


def phase_judged(args) -> None:
    n = realized_rows(args)
    key = phase_key(args, {"n_rows": n, "labeling": str(DV_LABELING.relative_to(PROJECT_ROOT))})
    if phase_done(args, "judged", key):
        return
    t0 = time.time()
    from explore_persona_space.experiments.issue_1739 import store_io

    lab_by_id = _labeling_by_id(args)
    root = _find_store_root(stage_root(args) / "labeling_store")
    arrs, meta = store_io.load_summaries(root, ("context_end",), (LAYER,))
    ctx = arrs[("context_end", LAYER)]
    t1_arr = None
    try:
        arrs_t1, _ = store_io.load_summaries(root, ("t1",), (LAYER,))
        t1_arr = arrs_t1[("t1", LAYER)]
    except FileNotFoundError:
        logger.info(
            "[judged] no answer-side (t1) L19 shards in the store — context-side only "
            "(DISCLOSED in meta.json)"
        )
    keep_ids, rows, lab_by_id = _judged_store_rows(args, meta, lab_by_id)
    ctx32 = ctx[rows].astype(np.float32)
    np.save(arrays_dir(args) / "judged_ctx_L19.npy", ctx32)  # P8 dim battery input
    pca_j = pca_transform_np(args, ctx32)
    import pickle

    with open(big_leg_root(args) / "models" / "umap_model.pkl", "rb") as f:
        umap_model = pickle.load(f)
    umap_j = np.empty((len(keep_ids), 2), dtype=np.float32)
    for lo in range(0, len(keep_ids), BLOCK):
        hi = min(lo + BLOCK, len(keep_ids))
        umap_j[lo:hi] = umap_model.transform(pca_j[lo:hi].astype(np.float32))
    adir = arrays_dir(args)
    km_c = np.load(adir / "kmeans_cx_centroids.npz")["centroids"]
    km_lab = _nearest_centroid(pca_j, km_c, len(keep_ids))
    hdb_path = adir / "hdbscan_cx_centroids.npz"
    if hdb_path.exists():
        hz = np.load(hdb_path)
        hdb_lab = hz["labels"].astype(np.int32)[
            _nearest_centroid(pca_j, hz["centroids"], len(keep_ids))
        ]
    else:
        hdb_lab = np.full(len(keep_ids), -1, dtype=np.int32)

    umap_t1 = None
    if t1_arr is not None:
        pca_t1 = pca_transform_np(args, t1_arr[rows].astype(np.float32))
        umap_t1 = np.empty((len(keep_ids), 2), dtype=np.float32)
        for lo in range(0, len(keep_ids), BLOCK):
            hi = min(lo + BLOCK, len(keep_ids))
            umap_t1[lo:hi] = umap_model.transform(pca_t1[lo:hi].astype(np.float32))

    dv = np.asarray(
        [
            float(lab_by_id[c]["dv"]) if lab_by_id[c].get("dv") is not None else np.nan
            for c in keep_ids
        ],
        dtype=np.float32,
    )
    out = {
        "pca2_ctx": pca_j[:, :2].astype(np.float32),
        "umap_ctx": umap_j,
        "kmeans_cx": km_lab,
        "hdbscan_cx": hdb_lab,
        "dv": dv,
    }
    if umap_t1 is not None:
        out["umap_t1"] = umap_t1
    np.savez(adir / "judged.npz", **out)
    export = export_dir(args)
    export.mkdir(parents=True, exist_ok=True)
    with open(export / "judged_meta.jsonl", "w", encoding="utf-8") as f:
        for c in keep_ids:
            r = lab_by_id[c]
            f.write(
                json.dumps(
                    {
                        "context_id": c,
                        "dv": r.get("dv"),
                        "split": r.get("split"),
                        "rung": r.get("rung"),
                        "group_key": r.get("group_key"),
                        "n_rollouts_judged": r.get("n_rollouts_judged"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    mark_done(
        args,
        "judged",
        key,
        t0,
        {
            "n_judged": len(keep_ids),
            "has_answer_side_t1": bool(t1_arr is not None),
            "store_root": str(root),
            "store_rows_total": int(ctx.shape[0]),
        },
    )


# ── phase: dim (P8 — spectra + intrinsic-dimension battery; scope extension) ─────

KNN_K = 101  # self + the k=100 local-PCA neighborhood; also covers MLE_KS
MLE_KS = (10, 20)
LOCAL_PCA_K = 100
CORR_RADII_N = 20


def _spectrum_stats(evals: np.ndarray) -> dict:
    """Descriptive spectrum stats + log-log power-law tail fit (ranks 10..1000)."""
    ev = np.sort(np.clip(np.asarray(evals, dtype=np.float64), 0.0, None))[::-1]
    tot = float(ev.sum())
    cum = np.cumsum(ev) / max(tot, 1e-300)

    def _ndim(q: float) -> int:
        return int(np.searchsorted(cum, q) + 1)

    lo, hi = 10, min(1000, ev.shape[0])
    ranks = np.arange(lo, hi + 1, dtype=np.float64)
    vals = ev[lo - 1 : hi]
    pos = vals > 0
    slope = float(np.polyfit(np.log(ranks[pos]), np.log(vals[pos]), 1)[0])
    return {
        "participation_ratio": float(ev.sum() ** 2 / max((ev**2).sum(), 1e-300)),
        "n_dims_50": _ndim(0.50),
        "n_dims_90": _ndim(0.90),
        "n_dims_99": _ndim(0.99),
        "powerlaw_exponent": slope,
        "powerlaw_fit_ranks": [int(lo), int(hi)],
        "total_variance": tot,
    }


def _accumulate_moments(args, srcs: dict, n: int) -> dict:
    """ONE chunked fp64 pass: per-space mean + second moment; the (cx, vx)
    cross moment rides the same pass (paired rows). Device-routed."""
    dev = compute_device()
    t0 = time.time()
    sums = {k: torch.zeros(H_DIM, dtype=torch.float64, device=dev) for k in srcs}
    moms = {k: torch.zeros(H_DIM, H_DIM, dtype=torch.float64, device=dev) for k in srcs}
    cross = torch.zeros(H_DIM, H_DIM, dtype=torch.float64, device=dev)
    n_blocks = (n + BLOCK - 1) // BLOCK
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        blocks = {
            k: torch.as_tensor(np.asarray(v[lo:hi]), dtype=torch.float64).to(dev)
            for k, v in srcs.items()
        }
        for k, xb in blocks.items():
            sums[k] += xb.sum(0)
            moms[k] += xb.T @ xb
        if "cx" in blocks and "vx" in blocks:
            cross += blocks["cx"].T @ blocks["vx"]
        logger.info(
            "[phase=dim] block %d/%d moments elapsed=%.1fs",
            lo // BLOCK + 1,
            n_blocks,
            time.time() - t0,
        )
    return {
        "sums": {k: v.cpu().numpy() for k, v in sums.items()},
        "moms": {k: v.cpu().numpy() for k, v in moms.items()},
        "cross_cx_vx": cross.cpu().numpy(),
        "n": n,
    }


def _cov_from_moments(S: np.ndarray, mu: np.ndarray, n: int) -> np.ndarray:
    C = S / n - np.outer(mu, mu)
    return (C + C.T) / 2.0


def _cca_spectrum(acc: dict, top: int = 500) -> tuple[np.ndarray, dict]:
    """(cx, vx) canonical-correlation spectrum from the accumulated moments.
    Descriptive spectrum, NOT a new predictor (no fit is banked or reused)."""
    n = acc["n"]
    mu_c = acc["sums"]["cx"] / n
    mu_a = acc["sums"]["vx"] / n
    Ccc = _cov_from_moments(acc["moms"]["cx"], mu_c, n)
    Caa = _cov_from_moments(acc["moms"]["vx"], mu_a, n)
    Cca = acc["cross_cx_vx"] / n - np.outer(mu_c, mu_a)
    reg_c = 1e-6 * float(np.trace(Ccc)) / H_DIM
    reg_a = 1e-6 * float(np.trace(Caa)) / H_DIM
    Ccc[np.diag_indices_from(Ccc)] += reg_c
    Caa[np.diag_indices_from(Caa)] += reg_a

    def _inv_sqrt(C: np.ndarray) -> np.ndarray:
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, None)
        return (V * (w**-0.5)) @ V.T

    K = _inv_sqrt(Ccc) @ Cca @ _inv_sqrt(Caa)
    sv = np.clip(np.linalg.svd(K, compute_uv=False), 0.0, 1.0)
    return sv[:top], {"reg_c": reg_c, "reg_a": reg_a, "n": n}


def _corr_radii(X: torch.Tensor) -> np.ndarray:
    """~20 log-spaced radii between the 1st and 50th percentile of POSITIVE
    pairwise distances on a 2k-row pilot (zero distances = duplicate rows,
    excluded from the radius derivation)."""
    m = min(2000, X.shape[0])
    d = _cdist(X[:m], X[:m])
    iu = torch.triu_indices(m, m, offset=1)
    vals = d[iu[0], iu[1]].cpu().numpy()
    pos = vals[vals > 0]
    if pos.size < 100:
        raise RuntimeError("corr-dim radius pilot: fewer than 100 positive pairwise distances")
    return np.geomspace(np.percentile(pos, 1), np.percentile(pos, 50), CORR_RADII_N)


def _knn_and_paircounts(X: torch.Tensor, radii: np.ndarray, k: int):
    """Chunked cdist over the sample: per-row k smallest distances + indices,
    plus unordered pair counts below each radius (self-pairs excluded)."""
    n = X.shape[0]
    k_eff = min(k, n)
    q = max(256, min(8192, int(1.5e9 / max(n * 4, 1))))
    dists = np.empty((n, k_eff), dtype=np.float32)
    idxs = np.empty((n, k_eff), dtype=np.int64)
    counts = np.zeros(len(radii), dtype=np.float64)
    for lo in range(0, n, q):
        hi = min(lo + q, n)
        d = _cdist(X[lo:hi], X)
        vals, ix = torch.topk(d, k=k_eff, dim=1, largest=False)
        dists[lo:hi] = vals.cpu().numpy()
        idxs[lo:hi] = ix.cpu().numpy()
        for ri, r in enumerate(radii):
            counts[ri] += float((d < float(r)).sum())
        del d
    pair_counts = (counts - n) / 2.0  # remove self-pairs, halve double counting
    return dists, idxs, pair_counts


def _id_twonn(dists: np.ndarray) -> dict:
    """TwoNN (Facco et al. 2017): fit -log(1-F) = d * log(mu) through the
    origin, discarding the top 10% of mu per the paper."""
    r1, r2 = dists[:, 1], dists[:, 2]
    valid = r1 > 0
    mu = np.sort(r2[valid] / r1[valid])
    n = mu.shape[0]
    keep = max(10, int(np.floor(n * 0.9)))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - np.arange(1, n + 1)[:keep] / n)
    return {
        "id": float((x * y).sum() / max((x * x).sum(), 1e-300)),
        "n_used": int(n),
        "n_zero_r1_dropped": int((~valid).sum()),
    }


def _id_lb_mle(dists: np.ndarray, k: int) -> dict:
    """Levina-Bickel MLE at k. Two named aggregate forms are reported:
    mean_of_local_mles = mean_x[(k-1)/s(x)] (the standard form) and
    mackay_ghahramani = (k-2)/mean_x[s(x)] (averaging inverse local estimates
    with the MacKay-Ghahramani k-2 correction), s(x) = sum_j log(T_k/T_j)."""
    T = dists[:, 1 : k + 1]
    ok = T[:, 0] > 0
    T = np.clip(T[ok], 1e-30, None)
    s = np.log(T[:, -1:] / T[:, :-1]).sum(axis=1)
    s = s[s > 0]
    if s.size == 0:
        raise RuntimeError(f"lb-mle k={k}: no valid rows (all-zero neighbor distances)")
    return {
        "id_mean_of_local_mles": float(((k - 1) / s).mean()),
        "id_mackay_ghahramani": float((k - 2) / s.mean()),
        "k": int(k),
        "n_used": int(s.size),
        "n_dropped": int(dists.shape[0] - s.size),
    }


def _id_corrdim(pair_counts: np.ndarray, radii: np.ndarray, n: int) -> dict:
    """Grassberger-Procaccia: slope of log C(r) vs log r over the linear
    region (preferring 1e-4 <= C <= 0.5); the fit window is reported."""
    n_pairs = n * (n - 1) / 2.0
    C = np.clip(pair_counts, 0.0, None) / max(n_pairs, 1.0)
    valid = (C > 0) & (C < 1)
    win = valid & (C >= 1e-4) & (C <= 0.5)
    if win.sum() < 3:
        win = valid
    if win.sum() < 3:
        return {"id": None, "note": "fewer than 3 usable radii", "n_radii_fit": int(win.sum())}
    slope = float(np.polyfit(np.log(radii[win]), np.log(C[win]), 1)[0])
    return {
        "id": slope,
        "fit_r_lo": float(radii[win].min()),
        "fit_r_hi": float(radii[win].max()),
        "n_radii_fit": int(win.sum()),
    }


def _id_local_pca(X_np: np.ndarray, idxs: np.ndarray, rng, n_anchors: int, k: int) -> dict:
    """Local-PCA ID: per-anchor #eigenvalues to 90% variance of the k-NN
    neighborhood; median + IQR across anchors."""
    n = X_np.shape[0]
    k_eff = min(k, idxs.shape[1] - 1)
    anchors = rng.choice(n, size=min(n_anchors, n), replace=False)
    dims = []
    for a in anchors:
        Y = X_np[idxs[a, 1 : k_eff + 1]].astype(np.float64)
        Y -= Y.mean(0)
        ev = np.linalg.svd(Y, compute_uv=False) ** 2
        cum = np.cumsum(ev) / max(ev.sum(), 1e-300)
        dims.append(int(np.searchsorted(cum, 0.90) + 1))
    d = np.asarray(dims)
    return {
        "id_median": float(np.median(d)),
        "id_iqr": [float(np.percentile(d, 25)), float(np.percentile(d, 75))],
        "k": int(k_eff),
        "n_anchors": int(anchors.size),
    }


def phase_dim(args) -> None:
    n = realized_rows(args)
    sz = sizes(args, n)
    key = phase_key(
        args,
        {
            "n_rows": n,
            "id_scales": list(sz["id_scales"]),
            "id_scales_judged": list(sz["id_scales_judged"]),
            "id_resamples": sz["id_resamples"],
            "knn_k": KNN_K,
            "mle_ks": list(MLE_KS),
            "anchors": sz["id_anchors"],
        },
    )
    if phase_done(args, "dim", key):
        return
    t0 = time.time()
    dev = compute_device()
    export = export_dir(args)
    export.mkdir(parents=True, exist_ok=True)
    srcs = {
        "cx": mm(args, "cx_L19.npy"),
        "vx": mm(args, "vx_L19.npy"),
        "vhat": mm(args, "vhat_L19.npy"),
    }

    # (a) full exact spectra + CCA, sub-checkpointed (the moment pass is the
    # expensive leg; the ID battery below has its own per-unit resume).
    spec_key = phase_key(args, {"n_rows": n, "leg": "spectra"})
    if not phase_done(args, "dim_spectra", spec_key):
        acc = _accumulate_moments(args, srcs, n)
        spectra: dict[str, np.ndarray] = {}
        stats: dict[str, dict] = {}
        for k in srcs:
            mu = acc["sums"][k] / n
            C = _cov_from_moments(acc["moms"][k], mu, n)
            ev = np.linalg.eigh(C)[0][::-1].copy()
            spectra[f"evals_{k}"] = ev
            stats[k] = _spectrum_stats(ev)
            logger.info("[dim] %s spectrum: %s", k, json.dumps(stats[k]))
        cca, cca_meta = _cca_spectrum(acc)
        np.savez(export / "dim_spectra.npz", cca_corrs_cx_vx=cca, **spectra)
        write_json_atomic(
            state_dir(args) / "dim_spectra_stats.json", {"spectra": stats, "cca": cca_meta}
        )
        mark_done(args, "dim_spectra", spec_key, t0)

    # (b) intrinsic-dimension battery, AMBIENT 3584-d fp32, per-unit JSONL resume.
    judged_path = arrays_dir(args) / "judged_ctx_L19.npy"
    if not judged_path.exists():
        raise RuntimeError("phase dim needs judged_ctx_L19.npy — run --phase judged first")
    pools = dict(srcs)
    pools["judged_cx"] = np.load(judged_path, mmap_mode="r")
    out_jsonl = export / "dim_id_estimates.jsonl"
    done_units = set()
    if out_jsonl.exists():
        done_units = {(r["space"], r["n"], r["resample"]) for r in iter_jsonl(out_jsonl)}
    units = []
    for si, (space, pool) in enumerate(sorted(pools.items())):
        pool_n = int(pool.shape[0])
        scales = sz["id_scales_judged"] if space == "judged_cx" else sz["id_scales"]
        for n_s in scales:
            n_eff = min(int(n_s), pool_n)
            n_res = 1 if n_eff == pool_n else int(sz["id_resamples"])
            for r in range(n_res):
                units.append((si, space, n_eff, r))
    for ui, (si, space, n_eff, r) in enumerate(units):
        if (space, n_eff, r) in done_units:
            continue
        tu = time.time()
        rng = np.random.default_rng([SEED, si, n_eff, r])
        pool = pools[space]
        idx = np.sort(rng.choice(int(pool.shape[0]), size=n_eff, replace=False))
        X_np = np.asarray(pool[idx], dtype=np.float32)
        X = torch.as_tensor(X_np, device=dev)
        radii = _corr_radii(X)
        dists, idxs, pair_counts = _knn_and_paircounts(X, radii, KNN_K)
        row = {
            "space": space,
            "n": n_eff,
            "resample": r,
            "seed": [SEED, si, n_eff, r],
            "ambient_dim": H_DIM,
            "twonn": _id_twonn(dists),
            "lb_mle": {str(k): _id_lb_mle(dists, k) for k in MLE_KS},
            "corr_dim": _id_corrdim(pair_counts, radii, n_eff),
            "local_pca": _id_local_pca(X_np, idxs, rng, sz["id_anchors"], LOCAL_PCA_K),
            "elapsed_s": round(time.time() - tu, 1),
        }
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "[phase=dim] block %d/%d space=%s n=%d r=%d elapsed=%.1fs",
            ui + 1,
            len(units),
            space,
            n_eff,
            r,
            time.time() - t0,
        )

    # headline summary: percentile bands across resamples per (space, estimator, n).
    rows = list(iter_jsonl(out_jsonl))
    summary: dict = {}
    for space in sorted({r["space"] for r in rows}):
        summary[space] = {}
        for n_s in sorted({r["n"] for r in rows if r["space"] == space}):
            sub = [r for r in rows if r["space"] == space and r["n"] == n_s]

            def _band(vals: list) -> dict | None:
                v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
                if v.size == 0:
                    return None
                return {
                    "p2_5": float(np.percentile(v, 2.5)),
                    "median": float(np.percentile(v, 50)),
                    "p97_5": float(np.percentile(v, 97.5)),
                    "n_resamples": int(v.size),
                }

            summary[space][str(n_s)] = {
                "twonn": _band([r["twonn"]["id"] for r in sub]),
                **{
                    f"lb_mle_k{k}_mean_of_local_mles": _band(
                        [r["lb_mle"][str(k)]["id_mean_of_local_mles"] for r in sub]
                    )
                    for k in MLE_KS
                },
                **{
                    f"lb_mle_k{k}_mackay_ghahramani": _band(
                        [r["lb_mle"][str(k)]["id_mackay_ghahramani"] for r in sub]
                    )
                    for k in MLE_KS
                },
                "corr_dim": _band([r["corr_dim"]["id"] for r in sub]),
                "local_pca_median": _band([r["local_pca"]["id_median"] for r in sub]),
            }
    spec_stats_p = state_dir(args) / "dim_spectra_stats.json"
    write_json_atomic(
        export / "dim_summary.json",
        {
            "spectra": json.loads(spec_stats_p.read_text(encoding="utf-8")),
            "id_estimates": summary,
            "notes": [
                "CCA correlation spectrum is a descriptive spectrum, not a new predictor.",
                "ID battery runs on AMBIENT 3584-d fp32 inputs, never PCA-reduced.",
                "lb_mle forms: mean_of_local_mles = mean[(k-1)/s]; "
                "mackay_ghahramani = (k-2)/mean[s], s = sum_j log(T_k/T_j).",
                "subsample draws are without replacement; n == pool size runs once.",
            ],
        },
    )
    mark_done(args, "dim", key, t0, {"n_units": len(units)})


# ── phase: xlayer (P9 — cross-layer vector similarity; scope extension) ──────────

XL_LAYERS = (14, 19, 26)
XL_OBJ = tuple(f"{kind}{layer}" for kind in ("cx", "vx") for layer in XL_LAYERS)
XL_MOM_PAIRS = tuple((i, j) for i in range(len(XL_OBJ)) for j in range(len(XL_OBJ)) if i <= j)  # 21
XL_COS_PAIRS = (
    ("cx14", "cx19"),
    ("cx19", "cx26"),
    ("cx14", "cx26"),
    ("vx14", "vx19"),
    ("vx19", "vx26"),
    ("vx14", "vx26"),
    ("cx14", "vx14"),
    ("cx19", "vx19"),
    ("cx26", "vx26"),
)
XL_HIST_BINS = 400
XL_CKPT_EVERY = 256


class _HistAcc:
    """Streaming cosine-distribution accumulator: fixed [-1, 1] histogram +
    exact mean/sd moments; percentiles read off the histogram (bin width 0.005)."""

    def __init__(self, state: dict | None = None):
        self.edges = np.linspace(-1.0, 1.0, XL_HIST_BINS + 1)
        if state is None:
            self.counts = np.zeros(XL_HIST_BINS, dtype=np.int64)
            self.n, self.s, self.ss = 0, 0.0, 0.0
        else:
            self.counts = np.asarray(state["counts"], dtype=np.int64)
            self.n, self.s, self.ss = int(state["n"]), float(state["s"]), float(state["ss"])

    def update(self, vals: np.ndarray) -> None:
        v = np.clip(np.asarray(vals, dtype=np.float64), -1.0, 1.0)
        self.counts += np.histogram(v, bins=self.edges)[0]
        self.n += v.size
        self.s += float(v.sum())
        self.ss += float((v**2).sum())

    def state(self) -> dict:
        return {"counts": self.counts.tolist(), "n": self.n, "s": self.s, "ss": self.ss}

    def stats(self) -> dict:
        cum = np.cumsum(self.counts)

        def _q(p: float) -> float:
            i = min(int(np.searchsorted(cum, p * self.n)), XL_HIST_BINS - 1)
            return float((self.edges[i] + self.edges[i + 1]) / 2)

        mean = self.s / max(self.n, 1)
        return {
            "n": int(self.n),
            "mean": mean,
            "sd": float(np.sqrt(max(self.ss / max(self.n, 1) - mean**2, 0.0))),
            "p2_5": _q(0.025),
            "median": _q(0.5),
            "p97_5": _q(0.975),
            "hist_counts": self.counts.tolist(),
            "hist_edges": {"lo": -1.0, "hi": 1.0, "bins": XL_HIST_BINS},
        }


def _cos_pair_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.clip(den, 1e-12, None)


def _np_stats(vals: np.ndarray) -> dict:
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "sd": float(vals.std()),
        "p2_5": float(np.percentile(vals, 2.5)),
        "median": float(np.percentile(vals, 50)),
        "p97_5": float(np.percentile(vals, 97.5)),
    }


def _cka_from_centered_moments(Cij, Cii, Cjj) -> float:
    num = float((Cij**2).sum())
    den = float(np.sqrt((Cii**2).sum()) * np.sqrt((Cjj**2).sum()))
    return num / max(den, 1e-300)


def _cka_direct(A: np.ndarray, B: np.ndarray) -> float:
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    return _cka_from_centered_moments(Ac.T @ Bc, Ac.T @ Ac, Bc.T @ Bc)


def phase_xlayer(args) -> None:
    n = realized_rows(args)
    sz = sizes(args, n)
    names = capture_chunk_names(args)
    fp = universe_fp(names)
    sub_n = min(int(sz["xlayer_sub"]), n)
    key = phase_key(
        args,
        {
            "universe_fp": fp,
            "layers": list(XL_LAYERS),
            "sub_n": sub_n,
            "bins": XL_HIST_BINS,
        },
    )
    if phase_done(args, "xlayer", key):
        return
    t0 = time.time()
    dev = compute_device()
    adir = arrays_dir(args)
    export = export_dir(args)
    export.mkdir(parents=True, exist_ok=True)
    ci_all = mm(args, "ci.npy")

    # (a) n1m tier: ONE extra stream over the packed 3-layer chunks.
    sub_idx = np.sort(np.random.default_rng(SEED + 3).choice(n, size=sub_n, replace=False))
    sub_mm = _open_or_create_memmap(
        adir / "xlayer_sub.npy", (sub_n, len(XL_OBJ), H_DIM), np.float32
    )
    acc_pt = adir / "xlayer_acc.pt"
    cur_p = state_dir(args) / "xlayer_cursor.json"
    start, row_base, sub_ptr = 0, 0, 0
    sums = torch.zeros(len(XL_OBJ), H_DIM, dtype=torch.float64, device=dev)
    moms = torch.zeros(len(XL_MOM_PAIRS), H_DIM, H_DIM, dtype=torch.float64, device=dev)
    hists = {pair: _HistAcc() for pair in XL_COS_PAIRS}
    if cur_p.exists() and acc_pt.exists():
        cur = json.loads(cur_p.read_text(encoding="utf-8"))
        if cur.get("universe_fp") == fp and cur.get("sub_n") == sub_n:
            blob = torch.load(acc_pt, map_location="cpu", weights_only=False)
            sums = blob["sums"].to(dev)
            moms = blob["moms"].to(dev)
            hists = {pair: _HistAcc(blob["hists"][f"{pair[0]}|{pair[1]}"]) for pair in XL_COS_PAIRS}
            start, row_base, sub_ptr = (
                int(cur["chunks_done"]),
                int(cur["row_base"]),
                int(cur["sub_ptr"]),
            )
            logger.info("[xlayer] RESUMED accumulators at chunk %d/%d", start, len(names))
        elif cur:
            logger.warning("[xlayer] cursor MISMATCHED (universe/sub_n changed); re-streaming")

    def _ckpt(done: int) -> None:
        with atomic_replace(acc_pt) as tmp:
            torch.save(
                {
                    "sums": sums.cpu(),
                    "moms": moms.cpu(),
                    "hists": {f"{p[0]}|{p[1]}": h.state() for p, h in hists.items()},
                },
                tmp,
            )
        write_json_atomic(
            cur_p,
            {
                "universe_fp": fp,
                "sub_n": sub_n,
                "chunks_done": done,
                "row_base": row_base,
                "sub_ptr": sub_ptr,
            },
        )

    cache = stage_root(args) / "chunk_cache"
    cache.mkdir(parents=True, exist_ok=True)
    obj_index = {k: i for i, k in enumerate(XL_OBJ)}
    for i in range(start, len(names)):
        got = dl(f"{CAPTURE_PREFIX}/{names[i]}", cache, f"xlayer chunk {names[i]}")
        b = torch.load(got, mmap=True, weights_only=False, map_location="cpu")
        cols = {layer: list(b["layers"]).index(layer) for layer in XL_LAYERS}
        objs = {}
        for field, kname in (("cx_last", "cx"), ("v_x", "vx")):
            for layer in XL_LAYERS:
                objs[f"{kname}{layer}"] = b[field][:, cols[layer], :].to(torch.float32)
        nrows = int(objs["cx19"].shape[0])
        ci_chunk = np.asarray([int(x) for x in b["ci"]], dtype=np.int64)
        got_ci = np.asarray(ci_all[row_base : row_base + nrows])
        if not (got_ci == ci_chunk).all():
            raise RuntimeError(f"xlayer/assemble row misalignment at chunk {names[i]}")
        t_objs = {k: v.to(dev, torch.float64) for k, v in objs.items()}
        for k, oi in obj_index.items():
            sums[oi] += t_objs[k].sum(0)
        for pi, (i2, j2) in enumerate(XL_MOM_PAIRS):
            moms[pi] += t_objs[XL_OBJ[i2]].T @ t_objs[XL_OBJ[j2]]
        for pair in XL_COS_PAIRS:
            a, c = t_objs[pair[0]], t_objs[pair[1]]
            cosv = (a * c).sum(1) / ((a.norm(dim=1) * c.norm(dim=1)).clamp(min=1e-12))
            hists[pair].update(cosv.cpu().numpy())
        while sub_ptr < sub_n and sub_idx[sub_ptr] < row_base + nrows:
            j = int(sub_idx[sub_ptr] - row_base)
            for k, oi in obj_index.items():
                sub_mm[sub_ptr, oi] = objs[k][j].numpy()
            sub_ptr += 1
        row_base += nrows
        del b, objs, t_objs
        got.unlink()
        if (i + 1) % XL_CKPT_EVERY == 0:
            _ckpt(i + 1)
        if (i + 1) % 100 == 0 or i + 1 == len(names):
            logger.info(
                "[phase=xlayer] block %d/%d rows=%d elapsed=%.1fs",
                i + 1,
                len(names),
                row_base,
                time.time() - t0,
            )
    if row_base != n or sub_ptr != sub_n:
        raise RuntimeError(f"xlayer stream incomplete: rows {row_base}/{n}, sub {sub_ptr}/{sub_n}")
    _ckpt(len(names))
    sub_mm.flush()

    sums_np = sums.cpu().numpy()
    moms_np = moms.cpu().numpy()
    means = sums_np / n
    cent = {}
    for pi, (i2, j2) in enumerate(XL_MOM_PAIRS):
        cent[(i2, j2)] = moms_np[pi] - n * np.outer(means[i2], means[j2])
    cka6 = np.eye(len(XL_OBJ))
    for i2 in range(len(XL_OBJ)):
        for j2 in range(i2 + 1, len(XL_OBJ)):
            cka6[i2, j2] = cka6[j2, i2] = _cka_from_centered_moments(
                cent[(i2, j2)], cent[(i2, i2)], cent[(j2, j2)]
            )

    # Subsample post-pass: centered cosines + one row-shuffled null (seed 42).
    sub = np.asarray(sub_mm, dtype=np.float64)
    perm = np.random.default_rng(SEED).permutation(sub_n)
    cos_stats: dict[str, dict] = {}
    for pair in XL_COS_PAIRS:
        ai, bi = obj_index[pair[0]], obj_index[pair[1]]
        A, B = sub[:, ai, :], sub[:, bi, :]
        Ac, Bc = A - means[ai], B - means[bi]
        cos_stats[f"{pair[0]}~{pair[1]}"] = {
            "raw_full": hists[pair].stats(),
            "centered_sub": _np_stats(_cos_pair_np(Ac, Bc)),
            "null_raw_sub": _np_stats(_cos_pair_np(A, B[perm])),
            "null_centered_sub": _np_stats(_cos_pair_np(Ac, Bc[perm])),
        }
    cka6_sub = np.eye(len(XL_OBJ))
    cka6_null = np.eye(len(XL_OBJ))
    for i2 in range(len(XL_OBJ)):
        for j2 in range(i2 + 1, len(XL_OBJ)):
            A, B = sub[:, i2, :], sub[:, j2, :]
            cka6_sub[i2, j2] = cka6_sub[j2, i2] = _cka_direct(A, B)
            cka6_null[i2, j2] = cka6_null[j2, i2] = _cka_direct(A, B[perm])

    # (b) 28-layer tier from the #1739 labeling store (context side only).
    from explore_persona_space.experiments.issue_1739 import store_io

    root = _find_store_root(stage_root(args) / "labeling_store")
    layers28 = _store_layers(root)
    arrs, meta = store_io.load_summaries(root, ("context_end",), tuple(layers28))
    keep_ids, rows_idx, _ = _judged_store_rows(args, meta, _labeling_by_id(args))
    stack = [
        torch.as_tensor(arrs[("context_end", layer)][rows_idx].astype(np.float32), device=dev)
        for layer in layers28
    ]
    stack = [x - x.mean(0) for x in stack]
    frobsq = {}
    cka28 = np.eye(len(layers28))
    for i2 in range(len(layers28)):
        frobsq[i2] = float(((stack[i2].T @ stack[i2]).double() ** 2).sum())
    for i2 in range(len(layers28)):
        for j2 in range(i2 + 1, len(layers28)):
            num = float(((stack[i2].T @ stack[j2]).double() ** 2).sum())
            cka28[i2, j2] = cka28[j2, i2] = num / max(
                np.sqrt(frobsq[i2]) * np.sqrt(frobsq[j2]), 1e-300
            )
        logger.info(
            "[phase=xlayer] block cka28 row %d/%d elapsed=%.1fs",
            i2 + 1,
            len(layers28),
            time.time() - t0,
        )
    raw28 = {
        layer: torch.as_tensor(arrs[("context_end", layer)][rows_idx].astype(np.float32))
        for layer in layers28
    }
    adj_curve = []
    for a_l, b_l in itertools.pairwise(layers28):
        A = raw28[a_l].numpy().astype(np.float64)
        B = raw28[b_l].numpy().astype(np.float64)
        adj_curve.append(
            {
                "layers": [int(a_l), int(b_l)],
                "raw": _np_stats(_cos_pair_np(A, B)),
                "centered": _np_stats(_cos_pair_np(A - A.mean(0), B - B.mean(0))),
            }
        )

    np.savez(
        export / "xlayer_cka.npz",
        cka6=cka6,
        cka6_sub=cka6_sub,
        cka6_null=cka6_null,
        labels6=np.asarray(XL_OBJ),
        sub_n=np.int64(sub_n),
        cka28=cka28,
        layers28=np.asarray(layers28, dtype=np.int64),
        n_judged=np.int64(len(keep_ids)),
    )
    write_json_atomic(
        export / "xlayer_cosine_stats.json",
        {
            "pairs": cos_stats,
            "adjacent_layer_curve_28": adj_curve,
            "notes": [
                "Per-row cosines are meaningful because all layers share the d=3584 "
                "residual-stream basis.",
                "CKA is rotation-invariant similarity (linear CKA from centered "
                "cross-moments); it can never support 'same vector up to sign/scale' — "
                "the cosine rows carry the direction-aware read.",
                f"Shuffle-null + centered variants computed on a deterministic "
                f"{sub_n}-row subsample (seed {SEED + 3}; null permutation seed {SEED}); "
                "raw full-corpus distributions come from the stream histograms.",
                "28-layer tier is CONTEXT-side only: the #1739 labeling store carries no "
                "answer-side 28-layer summaries (limitation recorded, not substituted).",
            ],
        },
    )
    mark_done(args, "xlayer", key, t0, {"sub_n": sub_n, "n_layers28": len(layers28)})


# ── phase: export ────────────────────────────────────────────────────────────────


def export_dir(args) -> Path:
    return durable_root(args) / "export"


def _read_meta_texts(export: Path) -> tuple[list[str], list[str], list[str]]:
    ctx_texts: list[str] = []
    ans_texts: list[str] = []
    corpora: list[str] = []
    for part in sorted(export.glob("row_meta_*.jsonl")):
        for r in iter_jsonl(part):
            ctx_texts.append(r["context_text"])
            ans_texts.append(r["answer_text"])
            corpora.append(r["corpus"])
    return ctx_texts, ans_texts, corpora


def _tfidf_top_terms(texts: list[str], labels: np.ndarray, top: int = 15) -> dict[int, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=50_000, stop_words="english")
    M = vec.fit_transform(texts)
    terms = np.asarray(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for u in sorted(int(x) for x in set(labels.tolist())):
        rows = np.flatnonzero(labels == u)
        if rows.size == 0:
            out[u] = []
            continue
        mean_tfidf = np.asarray(M[rows].mean(axis=0)).ravel()
        out[u] = terms[np.argsort(mean_tfidf)[::-1][:top]].tolist()
    return out


def phase_export(args) -> None:
    n = realized_rows(args)
    key = phase_key(args, {"n_rows": n})
    if phase_done(args, "export", key):
        return
    t0 = time.time()
    adir = arrays_dir(args)
    export = export_dir(args)
    export.mkdir(parents=True, exist_ok=True)
    key["inputs"] = sorted(
        p.name for p in state_dir(args).glob("*.done.json") if p.name != "export.done.json"
    )
    if phase_done(args, "export", key):
        return

    met = np.asarray(mm(args, "metrics.npy"))
    coords = {
        "ci": np.asarray(mm(args, "ci.npy")),
        "umap_cx": np.asarray(mm(args, "umap_cx.npy")),
        "umap_vx": np.asarray(mm(args, "umap_vx.npy")),
        "umap_vhat": np.asarray(mm(args, "umap_vhat.npy")),
        "pca2_cx": np.asarray(mm(args, "pca_cx.npy"))[:, :2],
        "pca2_vx": np.asarray(mm(args, "pca_vx.npy"))[:, :2],
        "pca2_vhat": np.asarray(mm(args, "pca_vhat.npy"))[:, :2],
        "kmeans_cx": np.asarray(mm(args, "kmeans_cx.npy")),
        "kmeans_vx": np.asarray(mm(args, "kmeans_vx.npy")),
        "hdbscan_cx": np.asarray(mm(args, "hdbscan_cx.npy")),
        "hdbscan_cx_in_subsample": np.asarray(mm(args, "hdbscan_cx_insub.npy")),
        "metric_names": np.asarray(METRIC_NAMES),
        "metrics": met,
    }
    np.savez(export / "coords.npz", **coords)
    if (adir / "judged.npz").exists():
        shutil.copyfile(adir / "judged.npz", export / "judged.npz")
    shutil.copyfile(adir / "pca_model.npz", export / "pca_model.npz")

    ctx_texts, ans_texts, corpora = _read_meta_texts(export)
    if len(ctx_texts) != n:
        raise RuntimeError(f"row_meta rows {len(ctx_texts)} != realized rows {n}")
    corp = np.asarray(corpora)
    judged = np.load(export / "judged.npz") if (export / "judged.npz").exists() else None

    def _table(labels: np.ndarray, texts: list[str], judged_labels: np.ndarray | None) -> list:
        top = _tfidf_top_terms(texts, labels)
        rows = []
        for u in sorted(int(x) for x in set(labels.tolist())):
            sel = labels == u
            row = {
                "cluster": u,
                "n": int(sel.sum()),
                "share_lmsys": float((corp[sel] == "lmsys").mean()),
                "share_wildchat": float((corp[sel] == "wildchat").mean()),
                "mean_cos_vhat_vx": float(met[sel, 0].mean()),
                "median_cos_vhat_vx": float(np.median(met[sel, 0])),
                "mean_cos_ib_vx": float(met[sel, 3].mean()),
                "top_tfidf_terms": top.get(u, []),
            }
            if judged is not None and judged_labels is not None:
                jsel = judged_labels == u
                dvv = judged["dv"][jsel]
                dvv = dvv[~np.isnan(dvv)]
                row["n_judged"] = int(jsel.sum())
                row["mean_dv_judged"] = float(dvv.mean()) if dvv.size else None
            rows.append(row)
        return rows

    partial = json.loads((state_dir(args) / "cluster_stats_partial.json").read_text("utf-8"))
    cluster_stats = {
        **partial,
        "kmeans_cx": _table(
            coords["kmeans_cx"], ctx_texts, judged["kmeans_cx"] if judged is not None else None
        ),
        "kmeans_vx": _table(coords["kmeans_vx"], ans_texts, None),
        "hdbscan_cx": _table(
            coords["hdbscan_cx"], ctx_texts, judged["hdbscan_cx"] if judged is not None else None
        ),
    }
    write_json_atomic(export / "cluster_stats.json", cluster_stats)

    walls = {}
    for p in sorted(state_dir(args).glob("*.done.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        walls[p.stem.replace(".done", "")] = rec.get("elapsed_s")
    pilot_p = durable_root(args) / "umap_pilot.json"
    write_json_atomic(
        export / "walltime.json",
        {
            "per_phase_elapsed_s": walls,
            "umap_pilot": json.loads(pilot_p.read_text("utf-8")) if pilot_p.exists() else None,
        },
    )

    file_shas = {
        p.name: sha256_file(p)
        for p in sorted(export.iterdir())
        if p.is_file() and p.name != "meta.json"
    }
    judged_done = json.loads((state_dir(args) / "judged.done.json").read_text("utf-8"))
    assemble_done = json.loads((state_dir(args) / "assemble.done.json").read_text("utf-8"))
    meta = {
        **as_metadata_dict(git_provenance(), phase="export"),
        "issue": 779,
        "round": "ctxansviz inline viz round",
        "layer": LAYER,
        "seed": SEED,
        "smoke": bool(args.smoke),
        "n_rows": n,
        "n_lmsys": assemble_done.get("n_lmsys"),
        "n_wildchat": assemble_done.get("n_wildchat"),
        "n_judged": judged_done.get("n_judged"),
        "has_answer_side_t1": judged_done.get("has_answer_side_t1"),
        "map_provenance": {
            "weights": f"{WEIGHTS_PREFIX}/L{LAYER}/ridge.pt",
            "fit_point": "mixed_1m",
            "n_train": N1M_N_TRAIN,
            "whole_map_heldout_r2_L19": WHOLE_MAP_R2_L19,
            "predict_path": "vhat = ((cx - xmu)/xsd) @ W + ymu (issue2474 registered path)",
        },
        "disclosures": [
            "split='train' / in_sample=True for every n1m row by construction: the banked "
            "mixed_1m ridge trained on the whole pool; the sha-pinned val/test held-out rows "
            "are pass_b rows (not in this capture).",
            f"context_text / answer_text capped at {TEXT_CAP} chars with an inline "
            "' …[truncated]' disclosure (cap_text, #1482 convention).",
            "TF-IDF top terms computed over the CAPPED text excerpts.",
            "judged overlay context_end rows deduped to the first store row per context_id.",
            "per-row metrics on n1m rows are IN-SAMPLE reads of the banked map.",
        ],
        "export_files_sha256": file_shas,
        "coords_alignment": "coords.npz arrays align positionally with row_meta_*.jsonl rows",
    }
    write_json_atomic(export / "meta.json", meta)

    prefix = EXPORT_PREFIX_SMOKE if args.smoke else EXPORT_PREFIX_FULL
    rel_files = sorted(p.name for p in export.iterdir() if p.is_file())
    url = hub._upload_folder_filtered(
        export,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
        allow_patterns=rel_files,
        expected_repo_paths=[f"{prefix}/{r}" for r in rel_files],
    )
    if not url:
        raise RuntimeError(f"export upload to {prefix} returned no URL")
    logger.info("[export] uploaded %d files to %s", len(rel_files), prefix)

    sentinel_name = (
        "issue-779-ctxansviz-smoke-results.json"
        if args.smoke
        else ("issue-779-ctxansviz-results.json")
    )
    sentinel_dir = Path("/workspace/logs")
    if not sentinel_dir.exists():
        sentinel_dir = durable_root(args)  # VM-side smoke runs have no /workspace
    write_json_atomic(
        sentinel_dir / sentinel_name,
        {
            "status": "ok",
            "issue": 779,
            "phase": "export",
            "smoke": bool(args.smoke),
            "n_rows": n,
            "n_judged": judged_done.get("n_judged"),
            "hf_prefix": prefix,
            "export_files_sha256": file_shas,
            "git_commit": meta.get("git_commit"),
            "ts": _utc(),
        },
    )
    mark_done(args, "export", key, t0, {"hf_prefix": prefix, "n_files": len(rel_files)})


# ── driver ───────────────────────────────────────────────────────────────────────

PHASES: dict[str, object] = {
    "stage": phase_stage,
    "stage-small": phase_stage_small,
    "assemble": phase_assemble,
    "predict": phase_predict,
    "pca": phase_pca,
    "umap": phase_umap,
    "cluster": phase_cluster,
    "judged": phase_judged,
    "dim": phase_dim,
    "xlayer": phase_xlayer,
    "export": phase_export,
}
ALL_ORDER = (
    "stage",
    "assemble",
    "predict",
    "pca",
    "umap",
    "cluster",
    "judged",
    "dim",
    "xlayer",
    "export",
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="issue #779 ctxansviz pod pipeline (joint embedding + clustering + map error)"
    )
    ap.add_argument("--phase", default="all", choices=("all", *PHASES))
    ap.add_argument("--smoke", action="store_true", help="tiny-N full-path smoke")
    ap.add_argument("--smoke-chunks", type=int, default=10, help="capture chunks under --smoke")
    ap.add_argument("--smoke-judged", type=int, default=500, help="judged contexts under --smoke")
    ap.add_argument("--out-root", default=None, help="override the durable (export) root")
    ap.add_argument(
        "--big-root", default=None, help="override the container-local staging/arrays root"
    )
    ap.add_argument("--stage-root", default=None, help="override the shared staging root")
    ap.add_argument(
        "--umap-relax-seed",
        action="store_true",
        help="random_state=None + n_jobs=16 (non-reproducible; the pilot-gate lever)",
    )
    ap.add_argument("--umap-wall-budget-s", type=int, default=14_400)
    ap.add_argument("--force-hdbscan", action="store_true")
    ap.add_argument(
        "--import-check", action="store_true", help="resolve deferred imports + argparse attrs"
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute every deferred import the phases use (lazy imports are
        # unverified code otherwise, #606).
        import pickle  # noqa: F401
        import umap  # noqa: F401
        from sklearn.cluster import HDBSCAN, MiniBatchKMeans  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.metrics import silhouette_score  # noqa: F401

        from explore_persona_space.experiments.issue_1739 import store_io  # noqa: F401
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401

        print("[import-check] OK: argparse attrs + deferred imports resolve")
        raise SystemExit(0)

    phases = ALL_ORDER if args.phase == "all" else (args.phase,)
    for name in phases:
        logger.info("[phase=%s] starting (smoke=%s)", name, bool(args.smoke))
        PHASES[name](args)
    if args.phase == "all":
        # This IS the dispatcher's single terminal line (only after every
        # --phase all stage completed).
        logger.info("[phase=done] ctxansviz complete")  # workflow-lint: phase-done-reserved
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
