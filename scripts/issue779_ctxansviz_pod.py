"""Pod-side context->answer joint-embedding pipeline for issue #779 (inline viz round).

Produces, on a dedicated RunPod CPU pod (cpu5m-16-128, /workspace), a joint
2D-embedding + clustering + map-error dataset over the #779 n1m final-token
capture (cx_last + v_x at L19), with the banked mixed_1m ridge applied
READ-ONLY (vhat = ((cx - xmu)/xsd) @ W + ymu, the #2474 registered path), plus
the #1739 sycophancy-labeling judged overlay (context_end L19 -> same
embedding, dv attached). Exports compactly for VM-side figure/dashboard
rendering to HF ``issue779_monitoring/ctxansviz/`` (smoke:
``issue779_monitoring/ctxansviz-smoke/``).

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
  export     coords npz + cluster_stats.json + walltime log + meta.json
             (git provenance, params, row counts, per-file sha256) -> ONE
             upload_folder commit to HF; results sentinel written LAST.

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
    labeling contexts; sample/cluster size knobs scale down (values only).
  Every code path — staging, streaming, predict, PCA/UMAP/cluster/judged/export,
  upload — is the production implementation in both modes.

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
    r"(?:context_end|context_k|t1|answer_k_t1)_L19(?:_shard\d+)?\.npy"
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


def out_root(args) -> Path:
    if args.out_root:
        return Path(args.out_root)
    # Per-leg out-roots: smoke never shares resume state with the full run.
    return Path("/workspace/ctxansviz-smoke" if args.smoke else "/workspace/ctxansviz")


def stage_root(args) -> Path:
    # Staged INPUTS are mode-independent (read-only mirrors, per-file idempotent
    # skips), so smoke and full share the stage dir; the tar is downloaded once.
    return Path(args.stage_root) if args.stage_root else Path("/workspace/ctxansviz-stage")


def state_dir(args) -> Path:
    return out_root(args) / "state"


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
        str(out_root(args)),
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
    write_json_atomic(sentinel, {"members": members, "tar_sha_bytes": tar_size, "ts": _utc()})
    local_tar.unlink()  # frees 52 GB; sentinel + members are the durable record
    logger.info("[stage] extracted %d store members; tar deleted", len(members))


# ── phase: assemble ──────────────────────────────────────────────────────────────


def _open_or_create_memmap(path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    if path.exists():
        return np.lib.format.open_memmap(path, mode="r+")
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def arrays_dir(args) -> Path:
    return out_root(args) / "arrays"


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
    comp = load_ridge_payload(args, LAYER)
    cx, vx = mm(args, "cx_L19.npy"), mm(args, "vx_L19.npy")
    adir = arrays_dir(args)

    # Pass A: global per-dim variance of vx (population, ddof=0) for the
    # per-row normalized-sqerr metric.
    s = torch.zeros(H_DIM, dtype=torch.float64)
    ss = torch.zeros(H_DIM, dtype=torch.float64)
    for lo in range(0, n, BLOCK):
        yb = torch.as_tensor(np.asarray(vx[lo : lo + BLOCK]), dtype=torch.float64)
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
        xb = torch.as_tensor(np.asarray(cx[lo:hi]), dtype=torch.float64)
        yb = torch.as_tensor(np.asarray(vx[lo:hi]), dtype=torch.float64)
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
        vhat[lo:hi] = yh.to(torch.float32).numpy()
        met[lo:hi] = torch.stack(cols, dim=1).to(torch.float32).numpy()
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
        }
    return {
        "pca_fit_per_side": min(200_000, n),
        "umap_fit_per_side": min(100_000, n),
        "umap_pilot": min(20_000, n),
        "kmeans_k": 50,
        "hdbscan_sub": min(150_000, n),
        "silhouette_n": min(20_000, n),
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
    for name, src in (("pca_cx.npy", cx), ("pca_vx.npy", vx), ("pca_vhat.npy", vhat)):
        dst = _open_or_create_memmap(adir / name, (n, dim), np.float32)
        for lo in range(0, n, BLOCK):
            hi = min(lo + BLOCK, n)
            dst[lo:hi] = pca.transform(np.asarray(src[lo:hi], dtype=np.float32))
        dst.flush()
        logger.info("[phase=pca] block done %s rows=%d elapsed=%.1fs", name, n, time.time() - t0)
    mark_done(args, "pca", key, t0, {"evr_sum": float(pca.explained_variance_ratio_.sum())})


def pca_transform_np(args, X: np.ndarray) -> np.ndarray:
    m = np.load(arrays_dir(args) / "pca_model.npz")
    return (X.astype(np.float32) - m["mean"]) @ m["components"].T


# ── phase: umap ──────────────────────────────────────────────────────────────────


def _umap_model(args, sz: dict):
    import umap

    n_jobs = 16 if args.umap_relax_seed else 1  # fixed random_state forces n_jobs=1 in umap
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
    write_json_atomic(out_root(args) / "umap_pilot.json", pilot_rec)
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
    models_dir = out_root(args) / "models"
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
            out_root(args) / "hdbscan_pilot.json",
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

    labeling = json.loads(DV_LABELING.read_text(encoding="utf-8"))
    lab_rows = labeling["rows"]
    if args.smoke:
        lab_rows = lab_rows[: int(args.smoke_judged)]
    lab_by_id = {str(r["context_id"]): r for r in lab_rows}
    logger.info("[judged] labeling rows: %d (of %d total)", len(lab_by_id), len(labeling["rows"]))

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
    ctx32 = ctx[rows].astype(np.float32)
    pca_j = pca_transform_np(args, ctx32)
    import pickle

    with open(out_root(args) / "models" / "umap_model.pkl", "rb") as f:
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


# ── phase: export ────────────────────────────────────────────────────────────────


def export_dir(args) -> Path:
    return out_root(args) / "export"


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
    pilot_p = out_root(args) / "umap_pilot.json"
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
        sentinel_dir = out_root(args)  # VM-side smoke runs have no /workspace
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
    "export": phase_export,
}
ALL_ORDER = ("stage", "assemble", "predict", "pca", "umap", "cluster", "judged", "export")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="issue #779 ctxansviz pod pipeline (joint embedding + clustering + map error)"
    )
    ap.add_argument("--phase", default="all", choices=("all", *PHASES))
    ap.add_argument("--smoke", action="store_true", help="tiny-N full-path smoke")
    ap.add_argument("--smoke-chunks", type=int, default=10, help="capture chunks under --smoke")
    ap.add_argument("--smoke-judged", type=int, default=500, help="judged contexts under --smoke")
    ap.add_argument("--out-root", default=None, help="override the per-mode output root")
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
