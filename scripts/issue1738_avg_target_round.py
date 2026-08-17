#!/usr/bin/env python3
"""Issue #1738 inline round `avg-target-maps` — rollout-averaged-target maps.

User ask: does training the context→answer map on ROLLOUT-AVERAGED targets
(5-draw mean answer state) beat training on the single original draw, when
evaluated against both single-draw and averaged holdout targets?

Thin COMPOSITION driver — new logic is ONLY (a) the seeded stratified 20k
train subsample, (b) 5-draw target averaging + alignment, (c) the eval table.
Everything else is reused verbatim:

- generation + teacher-forced capture: ``issue1738_multiturn_generate_capture
  --kresample`` (the machinery that produced the banked holdout K-resample),
  pointed at TRAIN rows via a fresh subsample doc; uploads ride
  ``issue1738_multiturn/avg_target/kresample`` (NEVER the parent prefix — the
  banked holdout shard00 lives there and shares shard filenames).
- ridge: ``issue779_ffc_n1m_fits.fit_ridge_with_weights`` (shared-eigh primal,
  LAMBDAS_N1M 23-penalty grid, val-selected λ).
- split/assembly: ``issue1738_multiturn_fits`` (pinned split_1738.json,
  capture-chunk memmap assembly, batched bootstrap CI).
- retrieval: ``analysis/mapping_baselines.knn_retrieval`` (euclidean +
  whitened cosine via the task-locked #2202 whiten_stats.npz).

λ-selection assumption (stated in the round's dispatch marker): BOTH fits
select λ on the SAME pinned val rows against SINGLE-draw targets (no averaged
val targets exist — generating them would deviate from the declared 80,000-
draw design), so the single manipulated variable between the two maps is the
TRAIN-time target; the eval surface is the 2x2 table.

Phases: stage → subsample → pilot (throughput fence, designed rc 31) →
generate (8-way CVD-pinned fan-out) → fits. Refusal-safety: never prints
conversation/rollout text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# vLLM V1 fork-safety (#628): spawn BEFORE any vllm import in this process tree.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as PF  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue1738_multiturn_fits as FITS  # noqa: E402
import issue1738_multiturn_generate_capture as GG  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1738_avgtgt")

PARENT_PREFIX = "issue1738_multiturn"
ROUND_PREFIX = "issue1738_multiturn/avg_target"
PILOT_PREFIX = "issue1738_multiturn/avg_target/pilot"
WHITEN_STATS_REPO_PATH = "issue2202_ctxfail/analysis_tensors/whiten_stats.npz"
BANKED_PRED_REPO_PATH = f"{PARENT_PREFIX}/analysis_tensors/pred16/context_L19_ridge.npz"
BANKED_YH_REPO_PATH = f"{PARENT_PREFIX}/analysis_tensors/y_holdout/L19.npz"
BANKED_KRES_REPO_PATH = f"{PARENT_PREFIX}/kresample/kresample_shard00.pt"

SUBSAMPLE_N = 20_000
SUBSAMPLE_SEED = 42
PILOT_N = 50
NUM_SHARDS = 8
SEEDS = "43,44,45,46"
N_DRAWS = 4  # fresh draws per context; +1 banked primary = 5-draw average
LAYER = 19
LAYERS_ALL = [14, 19, 26]  # assemble ALL parent layers => fingerprint parity with banked npzs
H_DIM = C.EXPECTED_HIDDEN  # 3584
FENCE_GPU_H = 20.0  # pilot-extrapolated total past this => designed halt
RC_FENCE = 31  # designed-halt rc (report written first; never a bare rc=1)
N_BOOT = 10_000
BOOT_SEED = 1738  # parent convention
KS = (1, 5, 10)
GEN_CAP_TOKENS = GG.GEN_MAX_TOKENS  # 1024 — cap-hit proxy threshold


# ── phase: stage (manifest + capture assembly + banked-identity assert) ──────────


def _assembly_ns(args) -> SimpleNamespace:
    """Namespace for FITS.assemble_streams — audited attr reads: mm_dir,
    local_capture_dir, hf_prefix (assemble_streams + _chunk_names only)."""
    return SimpleNamespace(
        mm_dir=str(args.out_root / "mm"),
        local_capture_dir="",
        hf_prefix=PARENT_PREFIX,
    )


def _open_assembly(args):
    """(mm, ci, ameta) — resumes complete via the cursor checkpoint."""
    return FITS.assemble_streams(_assembly_ns(args), LAYERS_ALL)


def _manifest_dir(args) -> Path:
    return GG.N1M._download_manifest(PARENT_PREFIX, args.out_root / GG.MANIFEST_SUBDIR)


def phase_stage(args) -> None:
    C.phase("stage")
    mdir = _manifest_dir(args)
    logger.info("[stage] manifest at %s", mdir)
    mm, ci, ameta = _open_assembly(args)
    logger.info("[stage] assembly: %d rows, fp=%s", ameta["n_rows"], ameta["fingerprint"][:16])

    # banked-identity assert: our assembled vx L19 holdout rows must reproduce
    # the banked y_holdout (same capture generation; fp16-cast equality).
    yh_path = args.out_root / "staged" / "y_holdout_L19.npz"
    yh_path.parent.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(C.HF_DATA_REPO, BANKED_YH_REPO_PATH, yh_path, repo_type="dataset")
    yh = np.load(yh_path)
    split = FITS.load_split(mdir / "split_1738.json")
    sets = FITS.split_positions(split, ci)
    ho = sets["holdout"]
    assert (np.asarray(yh["ci"], dtype=np.int64) == ci[ho]).all(), (
        "banked y_holdout ci order != assembly holdout order — different capture generations"
    )
    ours16 = np.asarray(mm[("vx", LAYER)][ho], dtype=np.float16)
    assert np.array_equal(ours16, yh["y16"]), (
        "assembled vx L19 holdout rows != banked y_holdout y16 — capture-generation drift"
    )
    logger.info("[stage] banked-identity assert OK (%d holdout rows)", len(ho))


# ── phase: subsample (seeded stratified 20k of captured train rows) ──────────────


def allocate_stratified(counts: dict[str, int], n_target: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding, capped at each
    stratum's size; any cap-induced shortfall is re-spread over uncapped strata
    (largest remaining capacity first). Sums to min(n_target, sum(counts))."""
    total = sum(counts.values())
    n_target = min(n_target, total)
    raw = {k: n_target * v / total for k, v in counts.items()}
    alloc = {k: min(int(raw[k]), counts[k]) for k in counts}
    # largest-remainder top-up, respecting caps
    while sum(alloc.values()) < n_target:
        rem = sorted(
            (k for k in counts if alloc[k] < counts[k]),
            key=lambda k: (raw[k] - int(raw[k]), counts[k] - alloc[k]),
            reverse=True,
        )
        if not rem:
            break
        alloc[rem[0]] += 1
        raw[rem[0]] = alloc[rem[0]]  # consumed its remainder; re-rank next pass
    return alloc


def phase_subsample(args) -> None:
    C.phase("subsample")
    mdir = _manifest_dir(args)
    pool, _meta = GG.N1M.read_manifest_pool(mdir)
    mm, ci, _ameta = _open_assembly(args)
    split = FITS.load_split(mdir / "split_1738.json")
    sets = FITS.split_positions(split, ci)
    train_cis = [int(c) for c in ci[sets["train"]]]

    strata: dict[str, list[int]] = {}
    for c in train_cis:
        row = pool[c]
        key = f"{row['corpus']}|{GG._depth_band(int(row['depth']))}"
        strata.setdefault(key, []).append(c)
    counts = {k: len(v) for k, v in sorted(strata.items())}
    alloc = allocate_stratified(counts, SUBSAMPLE_N)
    logger.info("[subsample] strata counts=%s alloc=%s", counts, alloc)

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    chosen: list[int] = []
    for k in sorted(strata):
        cis_sorted = np.asarray(sorted(strata[k]), dtype=np.int64)
        take = rng.choice(cis_sorted, size=alloc[k], replace=False)
        chosen.extend(int(x) for x in take)
    # deterministic shuffle so the 8 contiguous shard ranges are stratum-balanced
    chosen = [chosen[int(i)] for i in rng.permutation(len(chosen))]
    assert len(set(chosen)) == len(chosen) == sum(alloc.values()), len(chosen)

    docs = args.out_root / "subsample"
    docs.mkdir(parents=True, exist_ok=True)
    sub_doc = {"ci": chosen, "sha256": GG._sha_int_list(chosen)}
    C.write_json_atomic(docs / "avg_target_subsample.json", sub_doc)
    pilot_cis = chosen[:PILOT_N]
    C.write_json_atomic(
        docs / "avg_target_pilot_subsample.json",
        {"ci": pilot_cis, "sha256": GG._sha_int_list(pilot_cis)},
    )
    C.write_json_atomic(docs / "avg_target_primary_cis.json", {"ci": train_cis})
    C.write_json_atomic(
        docs / "avg_target_strata.json",
        {"counts": counts, "alloc": alloc, "seed": SUBSAMPLE_SEED, "n": len(chosen)},
    )
    logger.info("[subsample] %d cis -> %s", len(chosen), docs)


# ── phases: pilot + generate (subprocess the REAL kresample CLI) ─────────────────


def _kresample_cmd(args, shard_index: int, num_shards: int, sub_doc: Path, hf_prefix: str):
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "issue1738_multiturn_generate_capture.py"),
        "--kresample",
        "--kresample-subsample",
        str(sub_doc),
        "--kresample-primary-ci",
        str(args.out_root / "subsample" / "avg_target_primary_cis.json"),
        "--seeds",
        SEEDS,
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--device",
        "cuda",
        "--hf-prefix",
        hf_prefix,
        "--out-dir",
        str(args.out_root),
        "--manifest-from-hf",  # short-circuits on the stage-phase local manifest
    ]


def _worker_env(gpu: int) -> dict:
    """Launcher-env pins (the CVD clobber gotcha) + the #1324 hang/IMA knobs +
    the #1689 multi-worker uv FUSE-storm guard."""
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "UV_NO_SYNC": "1",
        "EPM_VLLM_ENFORCE_EAGER": "1",
        "EPM_VLLM_DISABLE_PREFIX_CACHING": "1",
    }


_UNIT_RE_LAST = "[kresample] unit "


def _last_unit_line(log_path: Path) -> tuple[int, int, float] | None:
    """(k, n, elapsed_s) from the last '[kresample] unit k/N ci=... elapsed=Xs'."""
    if not log_path.exists():
        return None
    out = None
    for line in log_path.read_text(errors="replace").split("\n"):
        if _UNIT_RE_LAST in line and "elapsed=" in line:
            try:
                frag = line.split(_UNIT_RE_LAST, 1)[1]
                kn = frag.split()[0]
                k, n = int(kn.split("/")[0]), int(kn.split("/")[1])
                el = float(frag.split("elapsed=")[1].rstrip("s").split("s")[0])
                out = (k, n, el)
            except (ValueError, IndexError):
                continue
    return out


def phase_pilot(args) -> None:
    C.phase("pilot")
    sub_doc = args.out_root / "subsample" / "avg_target_pilot_subsample.json"
    log = args.out_root / "logs" / "pilot_shard0.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = _kresample_cmd(args, 0, 1, sub_doc, PILOT_PREFIX)
    logger.info("[pilot] launching foreground on GPU 0: %s", " ".join(cmd[-12:]))
    t0 = time.time()
    with open(log, "wb") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=_worker_env(0)).returncode
    wall = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"pilot kresample shard exited rc={rc}; see {log}")
    last = _last_unit_line(log)
    if last is None:
        if "already on Hub; skip" in log.read_text(errors="replace"):
            # resume after a completed pilot: the fence verdict was already
            # rendered on the first pass (pilot_report.json persisted then).
            logger.info("[pilot] shard already on Hub — fence already passed; skipping re-measure")
            return
        raise RuntimeError(f"pilot log {log} has no per-unit progress lines — cannot size fence")
    kept, _n, elapsed = last
    units = kept * N_DRAWS
    per_unit_s = elapsed / max(1, units)
    load_overhead_s = max(0.0, wall - elapsed)
    total_units = SUBSAMPLE_N * N_DRAWS
    serial_h = per_unit_s * total_units / 3600.0
    gpu_h = serial_h + NUM_SHARDS * load_overhead_s / 3600.0
    wall_h = serial_h / NUM_SHARDS + load_overhead_s / 3600.0
    report = {
        "gate": "avgtgt-pilot",
        "pilot_kept_contexts": kept,
        "pilot_units": units,
        "pilot_unit_wall_s": elapsed,
        "pilot_total_wall_s": wall,
        "measured_s_per_unit": per_unit_s,
        "load_overhead_s": load_overhead_s,
        "projected_total_units": total_units,
        "projected_gpu_h": gpu_h,
        "projected_wall_h_8way": wall_h,
        "fence_gpu_h": FENCE_GPU_H,
        "verdict": "PASS" if gpu_h <= FENCE_GPU_H else "FENCE",
    }
    rep_dir = args.out_root / "eval"
    rep_dir.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(rep_dir / "pilot_report.json", report)
    logger.info("[pilot] %s", json.dumps(report))
    if gpu_h > FENCE_GPU_H:
        logger.error(
            "[pilot] FENCE: projected %.1f GPU-h > %.1f — designed halt", gpu_h, FENCE_GPU_H
        )
        sys.exit(RC_FENCE)


def phase_generate(args) -> None:
    C.phase("generate")
    sub_doc = args.out_root / "subsample" / "avg_target_subsample.json"
    logdir = args.out_root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, subprocess.Popen]] = []
    for g in range(NUM_SHARDS):
        log = logdir / f"gen_shard{g}.log"
        cmd = _kresample_cmd(args, g, NUM_SHARDS, sub_doc, ROUND_PREFIX)
        fh = open(log, "ab")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=_worker_env(g))
        procs.append((g, p))
        logger.info("[generate] shard %d -> GPU %d pid=%d log=%s", g, g, p.pid, log)
    t0 = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(120)
        status = []
        for g, p in procs:
            last = _last_unit_line(logdir / f"gen_shard{g}.log")
            tag = (
                f"{last[0]}/{last[1]}"
                if last
                else ("live" if p.poll() is None else f"rc={p.poll()}")
            )
            status.append(f"s{g}:{tag}")
        logger.info("[generate] t=%.0fmin %s", (time.time() - t0) / 60, " ".join(status))
    rcs = {g: p.wait() for g, p in procs}
    bad = {g: rc for g, rc in rcs.items() if rc != 0}
    if bad:
        raise RuntimeError(f"generate shards failed: {bad} — see {logdir}/gen_shard*.log")
    # exact-set completion check on the Hub (shards upload+purge as they finish)
    idx = GG.N50._remote_index(f"{ROUND_PREFIX}/{GG.KRESAMPLE_SUBDIR}")
    missing = [
        n
        for g in range(NUM_SHARDS)
        for n in (f"kresample_shard{g:02d}.pt", f"kresample_shard{g:02d}.json")
        if n not in idx
    ]
    if missing:
        raise RuntimeError(f"generate finished rc=0 but Hub is missing {missing}")
    logger.info("[generate] all %d shards complete + on Hub", NUM_SHARDS)


# ── phase: fits (two ridge maps + banked row + 2x2 eval table) ───────────────────


def _stage_round_kresample(args) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(V_sum_L19 fp32 (n,H), ci int64 (n,), skip records) across the 8 round shards."""
    dl = args.out_root / "kres_dl"
    dl.mkdir(parents=True, exist_ok=True)
    v_parts, ci_parts, skips = [], [], []
    for g in range(NUM_SHARDS):
        pt = dl / f"kresample_shard{g:02d}.pt"
        hub.stage_hub_file(
            C.HF_DATA_REPO,
            f"{ROUND_PREFIX}/{GG.KRESAMPLE_SUBDIR}/{pt.name}",
            pt,
            repo_type="dataset",
        )
        b = torch.load(pt, map_location="cpu", weights_only=False)
        li = list(b["layers"]).index(LAYER)
        v = b["V"][:, :, li, :].to(torch.float32)  # (n_g, K, H)
        assert v.shape[1] == N_DRAWS and v.shape[2] == H_DIM, v.shape
        v_parts.append(v.sum(dim=1).numpy())
        ci_parts.append(np.asarray(b["ci"], dtype=np.int64))
        skip_name = f"kresample_shard{g:02d}_skipped.json"
        sp = dl / skip_name
        try:
            hub.stage_hub_file(
                C.HF_DATA_REPO,
                f"{ROUND_PREFIX}/{GG.KRESAMPLE_SUBDIR}/{skip_name}",
                sp,
                repo_type="dataset",
            )
            skips.extend(json.loads(sp.read_text()).get("skipped", []))
        except Exception as e:  # sidecar absent => zero skips recorded for the shard
            logger.info("[fits] no skip sidecar for shard %d (%s)", g, type(e).__name__)
    vsum = np.concatenate(v_parts, axis=0)
    cis = np.concatenate(ci_parts, axis=0)
    assert len(set(cis.tolist())) == len(cis), "duplicate ci across round kresample shards"
    return vsum, cis, skips


def _whiten_fn(stats_path: Path):
    z = np.load(stats_path)
    mu_a = np.asarray(z["mu_A"], dtype=np.float64)
    ell = np.asarray(z["L"], dtype=np.float64)

    def _wh(x: np.ndarray) -> np.ndarray:
        return solve_triangular(ell, (np.asarray(x, np.float64) - mu_a).T, lower=True).T

    return _wh


def _eval_cell(pred: np.ndarray, true: np.ndarray, wh) -> dict:
    """R² + mean cosine + batched bootstrap CI + acc@k (raw euclidean and
    whitened cosine; pool = the target set itself)."""
    r2, cos = F._recon_point(pred, true)
    boot = FITS._boot_recon_ci_batched(pred, true, N_BOOT, BOOT_SEED)
    return {
        "r2": float(r2),
        "mean_cosine": float(cos),
        "r2_ci": boot["r2"],
        "mean_cosine_ci": boot["mean_cosine"],
        "n": int(pred.shape[0]),
        "knn_raw_euclidean": knn_retrieval(pred, true, ks=KS, metric="euclidean"),
        "knn_whiten_cos": knn_retrieval(wh(pred), wh(true), ks=KS, metric="cosine"),
    }


def _cap_hit_stats(args, pool_by_ci: dict[int, dict]) -> dict:
    """Re-tokenization cap-hit proxy over the round's raw draws (finish_reason
    is not persisted by the reused machinery): fraction of responses whose
    re-tokenized length >= GEN_MAX_TOKENS, overall + per corpus x depth-band
    stratum (>2% per stratum triggers the re-gen note, per dispatch contract)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GG.DEFAULT_MODEL)
    dl = args.out_root / "kres_dl"
    per: dict[str, list[int]] = {}
    n_total = 0
    n_hit = 0
    for g in range(NUM_SHARDS):
        raw = dl / f"kresample_shard{g:02d}.json"
        hub.stage_hub_file(
            C.HF_DATA_REPO,
            f"{ROUND_PREFIX}/{GG.KRESAMPLE_SUBDIR}/{raw.name}",
            raw,
            repo_type="dataset",
        )
        doc = json.loads(raw.read_text())
        for row in doc["rows"]:
            ci_ = int(row["ci"])
            mrow = pool_by_ci[ci_]
            key = f"{mrow['corpus']}|{GG._depth_band(int(mrow['depth']))}"
            for resp in row["responses"].values():
                n = len(tok(resp, add_special_tokens=False)["input_ids"])
                hit = int(n >= GEN_CAP_TOKENS)
                per.setdefault(key, []).append(hit)
                n_total += 1
                n_hit += hit
    strata = {k: {"n": len(v), "cap_hit_frac": float(np.mean(v))} for k, v in sorted(per.items())}
    return {
        "threshold_tokens": int(GEN_CAP_TOKENS),
        "method": "re-tokenization proxy (finish_reason not persisted by reused machinery)",
        "n_draws": n_total,
        "cap_hit_frac": (n_hit / n_total) if n_total else float("nan"),
        "per_stratum": strata,
        "regen_trigger_strata": [k for k, v in strata.items() if v["cap_hit_frac"] > 0.02],
    }


def phase_fits(args) -> None:
    C.phase("fits")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = _manifest_dir(args)
    pool, _meta = GG.N1M.read_manifest_pool(mdir)
    mm, ci, ameta = _open_assembly(args)
    split = FITS.load_split(mdir / "split_1738.json")
    sets = FITS.split_positions(split, ci)
    val, ho = sets["val"], sets["holdout"]
    pos_of = {int(c): p for p, c in enumerate(ci.tolist())}

    sub = json.loads((args.out_root / "subsample" / "avg_target_subsample.json").read_text())
    sub_cis = [int(c) for c in sub["ci"]]
    assert GG._sha_int_list(sub_cis) == sub["sha256"], "subsample doc sha mismatch"

    vsum, kres_ci, skips = _stage_round_kresample(args)
    kept = set(int(c) for c in kres_ci.tolist())
    assert kept <= set(sub_cis), "round kresample cis not a subset of the subsample"
    missing = sorted(set(sub_cis) - kept)
    logger.info(
        "[fits] kresample coverage: %d/%d kept (%d missing; %d skip records)",
        len(kept),
        len(sub_cis),
        len(missing),
        len(skips),
    )

    tr = np.asarray(sorted(pos_of[c] for c in kept), dtype=np.int64)
    n_tr = len(tr)
    if n_tr <= H_DIM and not args.allow_underdetermined:
        raise SystemExit(f"n_train={n_tr} <= d={H_DIM}: estimator-degenerate — refuse")

    X = mm[("cx", LAYER)]
    Y_single = mm[("vx", LAYER)]
    # Y_avg: full-length fp32 copy; averaged targets at the kept subsample rows;
    # val rows stay single-draw (the stated λ-selection surface).
    Y_avg = np.array(Y_single, dtype=np.float32)
    row_of = {int(c): j for j, c in enumerate(kres_ci.tolist())}
    tr_rows = np.asarray([row_of[int(ci[p])] for p in tr], dtype=np.int64)
    Y_avg[tr] = (Y_avg[tr] + vsum[tr_rows]) / float(N_DRAWS + 1)

    logger.info("[fits] fitting map-single-20k (n_tr=%d, d=%d)", n_tr, H_DIM)
    pred_single, meta_single, payload_single = PF.fit_ridge_with_weights(
        X, Y_single, tr, val, ho, PF.LAMBDAS_N1M, dev, PF.RIDGE_BLOCK
    )
    logger.info("[fits] fitting map-avg-20k")
    pred_avg, meta_avg, payload_avg = PF.fit_ridge_with_weights(
        X, Y_avg, tr, val, ho, PF.LAMBDAS_N1M, dev, PF.RIDGE_BLOCK
    )

    # banked 88k-train ridge predictions on the SAME holdout (ci-order asserted)
    bp = args.out_root / "staged" / "pred16_context_L19_ridge.npz"
    bp.parent.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(C.HF_DATA_REPO, BANKED_PRED_REPO_PATH, bp, repo_type="dataset")
    bd = np.load(bp)
    assert (np.asarray(bd["ci"], dtype=np.int64) == ci[ho]).all(), "banked pred16 ci misalign"
    pred_banked = bd["pred16"].astype(np.float64)

    # identity+learned-bias baselines (standing mapping-baselines rule; dims match)
    x_tr = np.asarray(X[tr], dtype=np.float64)
    x_ho = np.asarray(X[ho], dtype=np.float64)
    id_single = x_ho + (np.asarray(Y_single[tr], dtype=np.float64) - x_tr).mean(axis=0)
    id_avg = x_ho + (np.asarray(Y_avg[tr], dtype=np.float64) - x_tr).mean(axis=0)

    # targets: single-draw (all captured holdout) + 5-draw average (banked-covered)
    T_single = np.asarray(Y_single[ho], dtype=np.float64)
    bk = args.out_root / "staged" / "kresample_shard00_banked.pt"
    hub.stage_hub_file(C.HF_DATA_REPO, BANKED_KRES_REPO_PATH, bk, repo_type="dataset")
    kb = torch.load(bk, map_location="cpu", weights_only=False)
    li = list(kb["layers"]).index(LAYER)
    vsum_ho = kb["V"][:, :, li, :].to(torch.float32).sum(dim=1).numpy()
    kres_ho_ci = np.asarray(kb["ci"], dtype=np.int64)
    ho_pos_of = {int(c): j for j, c in enumerate(ci[ho].tolist())}
    sel = np.asarray([ho_pos_of[int(c)] for c in kres_ho_ci], dtype=np.int64)
    T_avg = (T_single[sel] * 1.0 + vsum_ho.astype(np.float64)) / float(N_DRAWS + 1)

    wh_path = args.out_root / "staged" / "whiten_stats.npz"
    hub.stage_hub_file(C.HF_DATA_REPO, WHITEN_STATS_REPO_PATH, wh_path, repo_type="dataset")
    wh = _whiten_fn(wh_path)

    maps = {
        "map_single_20k": pred_single,
        "map_avg_20k": pred_avg,
        "map_single_88k_banked": pred_banked,
        "idbias_single_20k": id_single,
        "idbias_avg_20k": id_avg,
    }
    cells: dict[str, dict] = {}
    for name, pred in maps.items():
        logger.info("[fits] eval %s vs single targets (n=%d)", name, len(ho))
        cells[f"{name}|target_single"] = _eval_cell(pred, T_single, wh)
        logger.info("[fits] eval %s vs avg targets (n=%d)", name, len(sel))
        cells[f"{name}|target_avg5"] = _eval_cell(pred[sel], T_avg, wh)
        # pool-matched companion: single-draw targets on the SAME 1,988 rows
        cells[f"{name}|target_single_on_avg_rows"] = _eval_cell(pred[sel], T_single[sel], wh)

    cap_hit = _cap_hit_stats(args, {int(r["i"]): r for r in pool})

    prov = git_provenance()
    summary = {
        "round": "avg-target-maps",
        "issue": 1738,
        "layer": LAYER,
        "assembly_fingerprint": ameta["fingerprint"],
        "n_train_subsample": len(sub_cis),
        "n_train_realized": int(n_tr),
        "subsample_missing_cis": missing,
        "n_skip_records": len(skips),
        "declared_draws": SUBSAMPLE_N * N_DRAWS,
        "realized_draws": int(len(kres_ci)) * N_DRAWS,
        "n_holdout_single": int(len(ho)),
        "n_holdout_avg5": int(len(sel)),
        "lambda_grid": [float(x) for x in PF.LAMBDAS_N1M],
        "selected_lambda": {
            "map_single_20k": meta_single["selected_lambda"],
            "map_avg_20k": meta_avg["selected_lambda"],
        },
        "fit_meta": {"map_single_20k": meta_single, "map_avg_20k": meta_avg},
        "lambda_selection_surface": "pinned val rows, SINGLE-draw targets for BOTH fits",
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "whiten_stats": WHITEN_STATS_REPO_PATH,
        "banked_pred": BANKED_PRED_REPO_PATH,
        "cap_hit": cap_hit,
        "cells": cells,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **as_metadata_dict(prov),
    }
    eval_dir = args.out_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(eval_dir / "avg_target_eval.json", summary)

    # persist tensors: weights + holdout predictions + averaged holdout targets
    up = args.out_root / "upload" / "analysis_tensors"
    (up / "weights").mkdir(parents=True, exist_ok=True)
    (up / "pred16").mkdir(parents=True, exist_ok=True)
    for name, payload in (("map_single_20k", payload_single), ("map_avg_20k", payload_avg)):
        torch.save(payload, up / "weights" / f"{name}_L{LAYER}_ridge.pt")
    for name, pred in maps.items():
        np.savez(
            up / "pred16" / f"{name}_L{LAYER}.npz",
            pred16=pred.astype(np.float16),
            ci=ci[ho],
            fingerprint=np.array(ameta["fingerprint"]),
        )
    np.savez(
        up / f"y_holdout_avg5_L{LAYER}.npz",
        y16=T_avg.astype(np.float16),
        ci=kres_ho_ci,
        fingerprint=np.array(ameta["fingerprint"]),
    )
    for doc in (args.out_root / "subsample").glob("avg_target_*.json"):
        (up / "subsample").mkdir(parents=True, exist_ok=True)
        (up / "subsample" / doc.name).write_bytes(doc.read_bytes())

    if not args.no_upload:
        url = hub._upload(
            up,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{ROUND_PREFIX}/analysis_tensors",
        )
        if not url:
            raise RuntimeError("analysis_tensors upload returned no URL")
        url2 = hub._upload(
            eval_dir,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{ROUND_PREFIX}/eval_mirror",
        )
        if not url2:
            raise RuntimeError("eval_mirror upload returned no URL")
    logger.info("[fits] done — eval at %s", eval_dir / "avg_target_eval.json")


# ── CPU smoke of the NEW logic (allocation, averaging alignment, eval cell) ──────


def _smoke(args) -> int:
    logger.info("[smoke] allocation arithmetic")
    alloc = allocate_stratified({"a": 100, "b": 300, "c": 5}, 50)
    assert sum(alloc.values()) == 50 and alloc["c"] <= 5, alloc
    alloc2 = allocate_stratified({"a": 10, "b": 10}, 50)  # target beyond pool -> take all
    assert alloc2 == {"a": 10, "b": 10}, alloc2

    logger.info("[smoke] Y_avg alignment + fits + eval cell (synthetic, cpu)")
    rng = np.random.default_rng(0)
    n, d = 60, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, d))
    Y = (X @ W + 0.1 * rng.normal(size=(n, d))).astype(np.float32)
    ci_s = np.arange(n, dtype=np.int64)
    tr = np.arange(0, 40, dtype=np.int64)
    val = np.arange(40, 50, dtype=np.int64)
    ho = np.arange(50, 60, dtype=np.int64)
    vsum = rng.normal(size=(len(tr), d)).astype(np.float32) * 0.01 + 4.0 * Y[tr]
    Y_avg = Y.copy()
    row_of = {int(ci_s[p]): j for j, p in enumerate(tr)}
    tr_rows = np.asarray([row_of[int(ci_s[p])] for p in tr], dtype=np.int64)
    Y_avg[tr] = (Y_avg[tr] + vsum[tr_rows]) / 5.0
    assert np.allclose(Y_avg[tr], (Y[tr] + vsum) / 5.0) and np.allclose(Y_avg[ho], Y[ho])

    dev = torch.device("cpu")
    pred, meta, payload = PF.fit_ridge_with_weights(X, Y_avg, tr, val, ho, PF.LAMBDAS_N1M, dev, 32)
    assert pred.shape == (len(ho), d) and np.isfinite(pred).all()
    assert "selected_lambda" in meta and payload["W"].shape == (d, d)

    mu = Y.mean(axis=0)
    cov = np.cov(Y.T) + 0.1 * np.eye(d)
    ell = np.linalg.cholesky(cov)
    stats = args.out_root / "smoke_whiten.npz"
    args.out_root.mkdir(parents=True, exist_ok=True)
    np.savez(stats, mu_A=mu, mu_C=mu, L=ell, lam=0.1, n_train=n)
    wh = _whiten_fn(stats)
    cell = _eval_cell(pred, np.asarray(Y[ho], dtype=np.float64), wh)
    assert np.isfinite(cell["r2"]) and cell["knn_raw_euclidean"]["acc_at_k"][1] >= 0.0
    assert np.isfinite(cell["r2_ci"]["lo"]) and cell["knn_whiten_cos"]["n_pool"] == len(ho)
    C.write_json_atomic(args.out_root / "smoke_eval.json", {"cells": {"smoke": cell}})
    logger.info("[smoke] OK")
    return 0


# ── main ──────────────────────────────────────────────────────────────────────────

PHASES = {
    "stage": phase_stage,
    "subsample": phase_subsample,
    "pilot": phase_pilot,
    "generate": phase_generate,
    "fits": phase_fits,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_1738" / "avg_target",
        help="pod: /workspace/outputs/issue1738_avgtgt",
    )
    ap.add_argument("--no-upload", action="store_true", help="skip the fits-phase HF uploads")
    ap.add_argument("--allow-underdetermined", action="store_true", help="smoke shapes only")
    ap.add_argument("--smoke", action="store_true", help="CPU smoke of the round-new logic")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        rc = _smoke(args)
    else:
        names = list(PHASES) if args.phase == "all" else [args.phase]
        for name in names:
            PHASES[name](args)
        rc = 0
    C.phase("done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
