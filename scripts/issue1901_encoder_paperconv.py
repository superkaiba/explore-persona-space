#!/usr/bin/env python3
"""#1901 fig2-baselines-panel round: baselines ON THE PAPER CONVENTION at n_train=25,000.

Scope (file-set claim + scope handoff marker, 2026-09-08): produce baseline numbers
that are legitimately placeable on paper Figure 2. Every number this driver emits is
scored on the EXACT convention of
``eval_results/issue_1901/figure2_five_rollout_scaling.json``: layer 19,
Qwen2.5-7B-Instruct, target = mean of five answer vectors (original + four fresh
on-policy rollouts), retrieval = whitened cosine + two-sided CSLS (K=10), whitening
z = L^-1 (v - mu_A) from the banked 963,444-row training-answer stats (lambda 0.1),
pool = 942 deduplicated queries vs 942 candidates, strict top-k with ties counted as
failures. Training rows = the EXACT banked 25k draw (``N1M.select_train`` under
PYTHONHASHSEED=0, seed_b=0; sel-sha asserted equal to the banked Job C record BEFORE
any fit).

Arms (fixed n_train = 25,000; identical rows, lambda grid, and eval battery):

  anchor        v_C(x)          -> v_A   3584 -> 3584  Step-1 correctness gate: must
                                          reproduce banked R2 0.778166 / top1 0.959660
                                          (tol 0.005) + banked selected_lambda + sel-sha.
  enc_e5        e5(x_templated) -> v_A   1024 -> 3584  headline encoder (mean-pooled,
                                          e5's intended head).
  enc_bge_cls   bge(x)          -> v_A   1024 -> 3584  companion encoder (CLS-pooled,
                                          bge's intended head; round-1 mean-pooled bge
                                          is a known lower bound).
  pca1024       PCA_1024(v_C)   -> v_A   1024 -> 3584  dimension control for the
                                          encoders' 1,024 dims (plain projection,
                                          deliberately no post-PCA whitening).
  shuffled      v_C -> v_A with TRAIN answers permuted across contexts (seeded).
                                          The Section-3-promised control; val/test
                                          pairs stay true (lambda selected on true
                                          val pairs — disclosed in the output meta).
  floor_e5 / floor_bge_cls      retrieval-only zero-parameter floor:
                cos(e(x), mean_4(e(fresh answer texts))). The ORIGINAL pass_b answer
                TEXT is not banked, so the answer-side embedding uses the FOUR banked
                fresh-rollout texts (kresample raw_completions), i.e. 4 of the 5
                target rollouts — a disclosed deviation of these two arms only.

Copy baselines (identity, identity+bias) are NOT recomputed: they are banked in
``eval_results/issue_1901/figure2_extension_1200.json`` under ``baselines`` and that
producer (issue1901_fig2_extension_1200.py) scores them through the SAME
FIVE._load_five_rollout_target + FINAL.score_cell path — convention confirmed by
reading the producer, not assumed.

Texts. Capture chunks bank the raw first-turn prompt per n1m row ("prompts" field,
schema-probed 2026-09-08). pass_b rows (val/test + orig-train rows in the draw) are
tensors-only on HF; their raw prompts are recovered by the ROUND-1 LMSYS
re-derivation (N50G.sample_disjoint_n50k phase 1, EXPECTED_CTX0_PROMPT drift guard —
the same production path the n1m capture generation used). Cross-validation before
any embedding: for every pinned test row, tokenize(apply_chat_template(recovered
prompt)) must equal the banked kresample ``prompt_token_ids`` for that row.
Encoder input = the FULL Qwen chat-templated prompt (round-1 convention; no
e5 "query:"/"passage:" prefixes, mirroring round 1 / issue1739). No context or
rollout text is ever printed or logged (refusal-safety).

Smoke blind-spot enumeration (--smoke-chunks K > 0):
  - SUBSTITUTES the fake sha-derived embedder for e5/bge (production encoder
    forwards never run under smoke) — mitigated by --encoder-probe (real one-batch
    forward through BOTH encoders) run on-pod before production.
  - DOWNGRADES the Step-1 gates (sel-sha / selected-lambda / R2 / top1): a partial
    pool structurally cannot reproduce the banked draw (gate-calibration carve-out).
    Values are logged, never asserted; outputs are *_smoke named.
  - Smoke n_train = min(25000, partial pool). Everything else (LMSYS re-derivation,
    token-id cross-check, whitened-CSLS battery, floor arms, persist + upload path)
    runs REAL.

Phases (checkpoint-per-phase; expensive artifacts fingerprinted + resumable):
  stage -> assemble+select -> anchor fit + Step-1 gate -> texts -> embed ->
  remaining arms -> score -> persist JSON + npz -> HF upload (+ scoped verify).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50G  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue1901_encoder_semantic_baseline as ENC  # noqa: E402  (import-only reuse)
import issue1901_figure2_five_rollout_scaling as FIVE  # noqa: E402
import issue1901_paper_densify_mlp as PDM  # noqa: E402
import issue1901_singleturn_retrieval_final as FINAL  # noqa: E402

logger = logging.getLogger("issue1901_encoder_paperconv")

N_TRAIN = 25_000
SEED_B = 0
SEL_NAME = "lmsys_25k"
LAYER = 19
K_FRESH = 4  # fresh rollouts per test row (kresample); target is mean of 1 + K_FRESH

REV_STORE = "0620bbd6adbc88cba4af8974ed6006f47844ea04"  # n1m capture / manifest / pass_b store
REV_FIG2 = "83d249cc9d495ca6f5d10f9156a622bcdca29a19"  # whiten / V_test / rawcomp pin

# Banked Step-1 anchors (eval_results/issue_1901/paper_densify/mlp_scaling_dense_L19.json
# + eval_results/issue_1901/figure2_five_rollout_scaling.json per_n["25000"].ridge).
SEL_SHA_25K = "3271c02eb4173c2566878e3bb63df59f0dd7da2ed20ce5e7c0f0de8d8bab2534"
LAMBDA_BANKED = 3162.2776601683795
R2_BANKED = 0.778166233260706
TOP1_BANKED = 0.9596602972399151
GATE_TOL = 0.005

WHITEN_FILE = "issue1901_mlpdense/analysis_tensors/whiten_stats_L19.npz"
VTEST_FILE = "issue1901_avgpool/analysis_tensors/kresample/V_test_shard00.npz"
RAWCOMP_PREFIX = "issue1901_avgpool/raw_completions/test/shard00"
HF_UPLOAD_PREFIX = "issue1901_fig2baselines"

ENCODERS = ("e5", "bge_cls")  # keys into ENC.EMBEDDERS / ENC.POOLING
SHUFFLE_SEED = 1901

DEFAULT_STAGE = PROJECT_ROOT / "data/issue_1901/fig2_baselines"
DEFAULT_OUT = PROJECT_ROOT / "eval_results/issue_1901/fig2_baselines/fig2_baselines.json"

CONVENTION = (
    "identical to figure2_five_rollout_scaling: five-rollout mean targets, whitened "
    "cosine + two-sided CSLS (K=10), 942-answer pool, banked 963,444-row whitening, "
    "strict top-k with ties counted as failures; n_train=25,000 = the banked Job C draw"
)


def _log(msg: str) -> None:
    logger.info(msg)
    print(f"[fig2b] {msg}", flush=True)


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:  # metadata only, never science
        return f"unavailable:{type(e).__name__}"


# ── stage ────────────────────────────────────────────────────────────────────────
def stage(args, smoke: bool):
    """Reuse PDM._stage_job_c (capture + pass_b + weights, fallocate probe, MemTotal
    floor) with the store revision PRE-PINNED to the banked REV_STORE, then fetch the
    figure-2 eval pins (whiten / V_test / rawcomp) at REV_FIG2."""
    from huggingface_hub import HfApi, hf_hub_download

    stage_root: Path = args.stage_root
    stage_root.mkdir(parents=True, exist_ok=True)
    rev_file = stage_root / ".stage_revision.json"
    if not rev_file.exists():
        C.write_json_atomic(
            rev_file,
            {
                "revision": REV_STORE,
                "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pinned_by": "issue1901_encoder_paperconv (banked store revision, not repo head)",
            },
        )
    shim = argparse.Namespace(
        stage_root=stage_root,
        smoke_chunks=args.smoke_chunks,
        stage_workers=args.stage_workers,
    )
    capture_dir, pass_b_path, _weights_dir, store_revision = PDM._stage_job_c(shim, smoke)
    if store_revision != REV_STORE:
        raise RuntimeError(f"staged store revision {store_revision} != pinned {REV_STORE}")

    pins = {}
    for key, fn in {"whiten": WHITEN_FILE, "v_test": VTEST_FILE}.items():
        pins[key] = Path(
            hub.retry_transient(
                lambda fn=fn: hf_hub_download(
                    repo_id=C.HF_DATA_REPO,
                    repo_type="dataset",
                    filename=fn,
                    revision=REV_FIG2,
                    local_dir=stage_root,
                ),
                what=f"fig2 pin {key}",
            )
        )
    listing = [
        p
        for p in hub.list_hf_files_under_path(
            HfApi(), C.HF_DATA_REPO, RAWCOMP_PREFIX, repo_type="dataset", revision=REV_FIG2
        )
        if p.endswith(".json")
    ]
    if not listing:
        raise RuntimeError(f"no rawcomp files under {RAWCOMP_PREFIX} @ {REV_FIG2}")
    for fn in listing:
        hub.retry_transient(
            lambda fn=fn: hf_hub_download(
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                filename=fn,
                revision=REV_FIG2,
                local_dir=stage_root,
            ),
            what=f"rawcomp {Path(fn).name}",
        )
    return capture_dir, pass_b_path, pins, stage_root / RAWCOMP_PREFIX


# ── texts ────────────────────────────────────────────────────────────────────────
def round1_prompts() -> list[str]:
    """The original 5,000 pass_b raw prompts, re-derived from the LMSYS stream
    (drift-guarded via the production _valtest_prompts_from_round1 assert)."""
    import issue779_ffc_n1m_generate_capture as N1G

    rec = N50G.sample_disjoint_n50k(N50G.N_ROUND1, 0, 0)
    round1 = rec["round1"]
    # Runs the EXPECTED_CTX0_PROMPT drift guard (raises on stream-order drift).
    N1G._valtest_prompts_from_round1(round1, check_ctx0=True)
    return round1


def chunk_prompt_for_rows(capture_dir: Path, rows_new: np.ndarray, n_new: int) -> dict[int, str]:
    """Raw prompts for assembled NEW rows (global row id >= N_PASS_B), by a second
    mmap pass in the SAME sorted chunk-file order the assemble used."""
    chunk_files = sorted(capture_dir.glob("shard*_chunk*.pt"))
    want = set(int(r) for r in rows_new)
    out: dict[int, str] = {}
    base = 0
    for cp in chunk_files:
        b = F79._mmap_load(cp)
        n = len(b["ci"])
        for j in range(n):
            g = N1M.N_PASS_B + base + j
            if g in want:
                out[g] = str(b["prompts"][j])
        base += n
        del b
    if base != n_new:
        raise RuntimeError(f"chunk text pass saw {base} rows, assemble saw {n_new}")
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"{len(missing)} selected new rows had no chunk prompt")
    return out


def load_fresh_answers(rawcomp_dir: Path) -> tuple[np.ndarray, dict]:
    """Fresh-rollout answer TEXTS per pinned test position: object array
    (N_TEST, K_FRESH) + meta. Rows keyed by ci = -(1+k) <-> test position k; ``src``
    filtered (packed-format consumer rule)."""
    files = sorted(rawcomp_dir.glob("gen_seed*_chunk*.json"))
    by_seed: dict[int, dict[int, str]] = {}
    tok_ids: dict[int, list[int]] = {}
    for fp in files:
        d = json.loads(fp.read_text())
        seed = int(d["meta"]["seed"])
        for r in d["rows"]:
            if r["src"] != "test":
                continue
            k = -int(r["ci"]) - 1
            if not (0 <= k < FINAL.N_TEST):
                raise RuntimeError(f"rawcomp ci {r['ci']} out of test range in {fp.name}")
            if not r["response"]:
                raise RuntimeError(f"empty response at ci={r['ci']} seed={seed}")
            by_seed.setdefault(seed, {})[k] = r["response"]
            tok_ids.setdefault(k, list(r["prompt_token_ids"]))
    seeds = sorted(by_seed)
    if len(seeds) != K_FRESH:
        raise RuntimeError(f"expected {K_FRESH} rollout seeds, got {seeds}")
    for s in seeds:
        if len(by_seed[s]) != FINAL.N_TEST:
            raise RuntimeError(f"seed {s} covers {len(by_seed[s])}/{FINAL.N_TEST} test rows")
    if len(tok_ids) != FINAL.N_TEST:
        raise RuntimeError(f"prompt_token_ids cover {len(tok_ids)}/{FINAL.N_TEST} test rows")
    texts = np.array([[by_seed[s][k] for s in seeds] for k in range(FINAL.N_TEST)], dtype=object)
    return texts, {"seeds": seeds, "prompt_token_ids": tok_ids, "n_files": len(files)}


def crosscheck_test_prompts(
    lm_tok, raw_by_row: dict[int, str], test: np.ndarray, tok_ids: dict[int, list[int]]
) -> None:
    """Token-space identity: tokenize(template(recovered raw prompt)) must equal the
    banked kresample prompt_token_ids for EVERY pinned test row. Validates the LMSYS
    re-derivation, the row alignment, and the template rendering in one shot."""
    n_bad = 0
    first_bad = None
    for k in range(len(test)):
        rendered = lm_tok.apply_chat_template(
            [{"role": "user", "content": raw_by_row[int(test[k])]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = list(lm_tok(rendered, add_special_tokens=False)["input_ids"])
        if ids != list(tok_ids[k]):
            n_bad += 1
            if first_bad is None:
                first_bad = (k, len(ids), len(tok_ids[k]))
    if n_bad:
        raise RuntimeError(
            f"test-prompt token-id cross-check FAILED on {n_bad}/{len(test)} rows "
            f"(first mismatch: test position {first_bad[0]}, rendered {first_bad[1]} tokens "
            f"vs banked {first_bad[2]}) — the re-derived prompts / template do not match "
            "the capture-time rendering; do NOT embed"
        )
    _log(f"test-prompt token-id cross-check PASS on all {len(test)} rows")


# ── fits + scoring ───────────────────────────────────────────────────────────────
def fit_arm(X, Y, tr, val, te, dev, block) -> tuple[np.ndarray, dict]:
    pred_te, meta, _payload = N1M.fit_ridge_with_weights(
        X, Y, tr, val, te, N1M.LAMBDAS_N1M, dev, block
    )
    return pred_te, meta


def pca_project(Xc: np.ndarray, tr: np.ndarray, k: int, dev) -> np.ndarray:
    """Top-k PCA of the TRAIN context vectors (fp64 covariance eigh), applied to all
    compact rows. Plain projection — deliberately no post-PCA whitening."""
    Xt = torch.as_tensor(Xc[tr], dtype=torch.float64, device=dev)
    mu = Xt.mean(0)
    Xt = Xt - mu
    cov = (Xt.T @ Xt) / max(1, len(tr) - 1)
    evals, evecs = torch.linalg.eigh(cov)
    proj = evecs[:, -k:].flip(-1)  # top-k, descending
    Xall = torch.as_tensor(Xc, dtype=torch.float64, device=dev) - mu
    out = (Xall @ proj).cpu().numpy().astype(np.float32)
    _log(f"pca{k}: eval span {float(evals[-1]):.3g}..{float(evals[-k]):.3g}")
    return out


def score_paper(pred: np.ndarray, target_mean: np.ndarray, view, whiten, seed: int) -> dict:
    r2, mean_cosine = F79._recon_point(pred, target_mean)
    ret = FINAL.score_cell(pred, target_mean, view, whiten, seed=seed)["whiten_csls"]["strict"]
    return {
        "r2": float(r2),
        "mean_cosine": float(mean_cosine),
        "top1": float(ret["acc_at_k"]["1"]),
        "top5": float(ret["acc_at_k"]["5"]),
        "top1_ci95": ret["acc1_ci95"],
    }


def score_floor(e_ctx_test: np.ndarray, e_ans_mean: np.ndarray, view, seed: int) -> dict:
    """Retrieval-only cosine floor in embedding space, same dedup view / strict-tie
    policy as every other arm (retrieval only: no prediction in the target space)."""
    q = np.asarray(e_ctx_test[view.pred_rows], dtype=np.float64)
    p = np.asarray(e_ans_mean[view.pool_rows], dtype=np.float64)
    dist = 1.0 - FINAL._cosine(q, p)
    ranks = FINAL._strict_ranks(dist, view.true_idx)
    summary = FINAL._rank_summary(ranks, p.shape[0], np.random.default_rng(seed))
    return {
        "r2": None,
        "mean_cosine": None,
        "top1": float(summary["acc_at_k"]["1"]),
        "top5": float(summary["acc_at_k"]["5"]),
        "top1_ci95": summary["acc1_ci95"],
    }


def _save_pred(npz_dir: Path, name: str, pred: np.ndarray, rows: np.ndarray) -> Path:
    p = npz_dir / f"pred_{name}.npz"
    np.savez_compressed(p, pred_fp16=pred.astype(np.float16), rows=np.asarray(rows, dtype=np.int64))
    return p


# ── standalone modes ─────────────────────────────────────────────────────────────
def selfcheck_scoring(args) -> None:
    """Local zero-download smoke of the scoring half: re-score the banked 25k ridge
    predictions off the already-staged five-rollout inputs; must reproduce the banked
    JSON values exactly (same functions, same seed)."""
    stage_dir = PROJECT_ROOT / "data/issue_1901/figure2_five_rollout_scaling"
    paths = {
        "pass_b": stage_dir
        / "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt",
        "whiten": stage_dir / WHITEN_FILE,
        "v_test": stage_dir / VTEST_FILE,
        "pred": stage_dir / "issue1901_mlpdense/analysis_tensors/preds_L19_n25000_ridge.npz",
    }
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"selfcheck input {key} missing: {p}")
    source_target, target_mean, expected_rows = FIVE._load_five_rollout_target(
        paths["pass_b"], paths["v_test"]
    )
    view = FINAL.make_eval_view(source_target, FINAL.N_TEST, "keep_one")
    whiten, _ = FINAL._whitener(paths["whiten"])
    pred = FIVE._load_prediction(paths["pred"], expected_rows)
    row = score_paper(pred, target_mean, view, whiten, seed=190_102 + N_TRAIN)
    banked = json.loads(
        (PROJECT_ROOT / "eval_results/issue_1901/figure2_five_rollout_scaling.json").read_text()
    )["per_n"]["25000"]["ridge"]
    dr2 = abs(row["r2"] - banked["r2"])
    dt1 = abs(row["top1"] - banked["top1"])
    print(
        f"[selfcheck] r2={row['r2']:.12f} (banked {banked['r2']:.12f}, d={dr2:.2e}) "
        f"top1={row['top1']:.10f} (banked {banked['top1']:.10f}, d={dt1:.2e})",
        flush=True,
    )
    if dr2 > 1e-9 or dt1 > 1e-9:
        raise RuntimeError("selfcheck-scoring did NOT reproduce the banked values exactly")
    print("[selfcheck] PASS — scoring path reproduces the banked 25k ridge cell", flush=True)


def encoder_probe(args) -> None:
    """One REAL batch through both production encoders (shapes/norms only — closes
    the smoke's fake-embedder blind spot before the production embed)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [f"probe sentence number {i} about the weather and cooking." for i in range(8)]
    for enc in ENCODERS:
        e = ENC._encode(texts, ENC.EMBEDDERS[enc], dev, 8, ENC.POOLING[enc])
        norms = np.linalg.norm(e, axis=1)
        if e.shape != (8, 1024) or not np.allclose(norms, 1.0, atol=1e-3):
            raise RuntimeError(f"{enc}: bad probe output shape={e.shape} norms={norms[:3]}")
        print(f"[probe] {enc}: shape={e.shape} norms ok (dev={dev})", flush=True)


# ── main pipeline ────────────────────────────────────────────────────────────────
def run(args) -> None:
    smoke = args.smoke_chunks > 0
    if not smoke and os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("production run requires PYTHONHASHSEED=0 (select_train seeding)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out if not smoke else args.out.with_name("fig2_baselines_smoke.json")
    npz_dir = out_path.parent / ("npz_smoke" if smoke else "npz")
    npz_dir.mkdir(parents=True, exist_ok=True)
    work = args.stage_root / "work"
    work.mkdir(parents=True, exist_ok=True)

    C.phase("stage")
    capture_dir, pass_b_path, pins, rawcomp_dir = stage(args, smoke)

    C.phase("assemble")
    shim = argparse.Namespace(stage_root=args.stage_root, work_dir=work, orig_dir=args.orig_dir)
    X, Y, pools, val, test, split = PDM._assemble_n1m(shim, capture_dir, pass_b_path, REV_STORE)
    n_target = min(N_TRAIN, len(pools["lmsys"])) if smoke else N_TRAIN
    sub, sel_diag = N1M.select_train(pools, SEL_NAME, n_target, "lmsys", SEED_B)
    sel_sha = F79._sha_ids(sub)
    if smoke:
        _log(f"smoke: sel_sha={sel_sha[:12]} n={len(sub)} (banked-draw gate SKIPPED)")
    elif sel_sha != SEL_SHA_25K:
        raise RuntimeError(
            f"selection identity FAILED: sel_sha {sel_sha} != banked {SEL_SHA_25K} — "
            "the refit is not on the banked 25k draw; STOP"
        )
    else:
        _log(f"selection identity PASS: sel_sha == banked ({sel_sha[:12]}...), n={len(sub)}")

    # Five-rollout targets + eval view + whitener (the paper convention).
    source_target, target_mean, expected_rows = FIVE._load_five_rollout_target(
        pass_b_path, pins["v_test"]
    )
    if not np.array_equal(expected_rows, np.asarray(test, dtype=np.int64)):
        raise RuntimeError("five-rollout test rows != assembled pinned test rows")
    view = FINAL.make_eval_view(source_target, FINAL.N_TEST, "keep_one")
    if view.diagnostics["realized_n_pool"] != 942 or view.diagnostics["realized_n_query"] != 942:
        raise RuntimeError(f"unexpected dedup geometry: {view.diagnostics}")
    whiten, whitening_meta = FINAL._whitener(pins["whiten"])

    C.phase("anchor-fit")
    t0 = time.time()
    pred_anchor, meta_anchor = fit_arm(X, Y, sub, val, test, dev, args.ridge_block)
    _log(
        f"anchor fit: {time.time() - t0:.1f}s selected_lambda={meta_anchor['selected_lambda']:.6g}"
    )
    row_anchor = score_paper(pred_anchor, target_mean, view, whiten, seed=190_102 + N_TRAIN)
    step1 = {
        "sel_sha256": sel_sha,
        "banked_sel_sha256": SEL_SHA_25K,
        "selected_lambda": meta_anchor["selected_lambda"],
        "banked_lambda": LAMBDA_BANKED,
        "r2_banked": R2_BANKED,
        "top1_banked": TOP1_BANKED,
        "tol": GATE_TOL,
        "d_r2": abs(row_anchor["r2"] - R2_BANKED),
        "d_top1": abs(row_anchor["top1"] - TOP1_BANKED),
        "gated": not smoke,
    }
    if not smoke:
        if abs(meta_anchor["selected_lambda"] - LAMBDA_BANKED) > 1e-6 * LAMBDA_BANKED:
            raise RuntimeError(
                f"anchor selected_lambda {meta_anchor['selected_lambda']} != banked "
                f"{LAMBDA_BANKED} — recipe drift; STOP"
            )
        if step1["d_r2"] > GATE_TOL or step1["d_top1"] > GATE_TOL:
            raise RuntimeError(
                f"STEP-1 GATE FAILED: refit r2={row_anchor['r2']:.6f} "
                f"(banked {R2_BANKED:.6f}, d={step1['d_r2']:.4f}), "
                f"top1={row_anchor['top1']:.6f} (banked {TOP1_BANKED:.6f}, "
                f"d={step1['d_top1']:.4f}) exceeds tol {GATE_TOL}; "
                "pipeline is NOT on the paper convention"
            )
        _log(f"STEP-1 GATE PASS: d_r2={step1['d_r2']:.2e} d_top1={step1['d_top1']:.2e} lam exact")
    else:
        _log(f"smoke anchor: r2={row_anchor['r2']:.4f} top1={row_anchor['top1']:.4f} (no gate)")

    # Compact the needed rows, then free the ~28 GB pair.
    needed = np.unique(np.concatenate([sub, val, test]))
    pos = {int(r): i for i, r in enumerate(needed)}
    tr_c = np.array([pos[int(r)] for r in sub], dtype=np.int64)
    val_c = np.array([pos[int(r)] for r in val], dtype=np.int64)
    te_c = np.array([pos[int(r)] for r in test], dtype=np.int64)
    Xc = np.ascontiguousarray(X[needed])
    Yc = np.ascontiguousarray(Y[needed])
    n_new = int(split["n_new_captured"])
    del X, Y
    _log(f"compacted to {len(needed)} rows; released the full pair")

    C.phase("texts")
    texts_path = work / ("texts_smoke.jsonl" if smoke else "texts.jsonl")
    fp_expected = _sha_bytes(needed.tobytes() + REV_STORE.encode())
    raw_by_row: dict[int, str] = {}
    if texts_path.exists():
        lines = texts_path.read_text().splitlines()
        if lines and json.loads(lines[0]).get("fp") == fp_expected:
            for line in lines[1:]:
                r = json.loads(line)
                raw_by_row[int(r["row"])] = r["raw"]
            _log(f"texts: resumed {len(raw_by_row)} rows from {texts_path.name}")
    if len(raw_by_row) != len(needed):
        r1 = round1_prompts()
        rows_new = needed[needed >= N1M.N_PASS_B]
        new_map = chunk_prompt_for_rows(capture_dir, rows_new, n_new)
        raw_by_row = {int(r): (r1[int(r)] if r < N1M.N_PASS_B else new_map[int(r)]) for r in needed}
        with texts_path.open("w") as fh:
            fh.write(json.dumps({"fp": fp_expected, "n": len(needed)}) + "\n")
            for r in needed:
                fh.write(
                    json.dumps({"row": int(r), "raw": raw_by_row[int(r)]}, ensure_ascii=False)
                    + "\n"
                )
        _log(f"texts: wrote {len(raw_by_row)} rows")

    fresh_texts, fresh_meta = load_fresh_answers(rawcomp_dir)

    from transformers import AutoTokenizer

    lm_tok = AutoTokenizer.from_pretrained(ENC.LM_MODEL)
    crosscheck_test_prompts(lm_tok, raw_by_row, test, fresh_meta["prompt_token_ids"])

    C.phase("embed")
    ctx_texts = [
        lm_tok.apply_chat_template(
            [{"role": "user", "content": raw_by_row[int(r)]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in needed
    ]
    ans_flat = [str(fresh_texts[k][j]) for k in range(FINAL.N_TEST) for j in range(K_FRESH)]
    emb: dict[str, np.ndarray] = {}
    ans_emb_mean: dict[str, np.ndarray] = {}
    for enc in ENCODERS:
        enc_npz = work / f"emb_{enc}{'_smoke' if smoke else ''}.npz"
        if enc_npz.exists():
            z = np.load(enc_npz, allow_pickle=False)
            if str(z["fp"]) == fp_expected:
                emb[enc] = np.asarray(z["e_ctx"], dtype=np.float32)
                ans_emb_mean[enc] = np.asarray(z["e_ans_mean"], dtype=np.float32)
                _log(f"embed[{enc}]: resumed from {enc_npz.name}")
                continue
        t0 = time.time()
        if smoke:
            e_ctx = ENC._fake_embed(ctx_texts)
            e_ans = ENC._fake_embed(ans_flat)
        else:
            dev_s = "cuda" if torch.cuda.is_available() else "cpu"
            e_ctx = ENC._encode(ctx_texts, ENC.EMBEDDERS[enc], dev_s, args.batch, ENC.POOLING[enc])
            e_ans = ENC._encode(ans_flat, ENC.EMBEDDERS[enc], dev_s, args.batch, ENC.POOLING[enc])
        e_ans = e_ans.reshape(FINAL.N_TEST, K_FRESH, -1)
        mean_ans = e_ans.mean(axis=1)
        mean_ans = mean_ans / (np.linalg.norm(mean_ans, axis=1, keepdims=True) + 1e-12)
        np.savez_compressed(enc_npz, fp=fp_expected, e_ctx=e_ctx, e_ans_mean=mean_ans, rows=needed)
        emb[enc] = e_ctx.astype(np.float32)
        ans_emb_mean[enc] = mean_ans.astype(np.float32)
        _log(f"embed[{enc}]: {len(ctx_texts)} ctx + {len(ans_flat)} ans in {time.time() - t0:.0f}s")

    C.phase("arms")
    per_arm: dict[str, dict] = {"anchor": {**row_anchor, "fit_meta": meta_anchor}}
    _save_pred(npz_dir, "anchor", pred_anchor, test)

    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(tr_c))
    y_shuf = Yc.copy()
    y_shuf[tr_c] = Yc[tr_c][perm]
    pred, meta = fit_arm(Xc, y_shuf, tr_c, val_c, te_c, dev, args.ridge_block)
    per_arm["shuffled"] = {
        **score_paper(pred, target_mean, view, whiten, seed=190_910),
        "fit_meta": {
            **meta,
            "shuffle_seed": SHUFFLE_SEED,
            "note": (
                "train answers permuted across contexts; val/test pairs true; "
                "lambda selected on true val pairs"
            ),
        },
    }
    _save_pred(npz_dir, "shuffled", pred, test)

    x_pca = pca_project(Xc, tr_c, 1024, dev)
    pred, meta = fit_arm(x_pca, Yc, tr_c, val_c, te_c, dev, args.ridge_block)
    per_arm["pca1024"] = {
        **score_paper(pred, target_mean, view, whiten, seed=190_920),
        "fit_meta": meta,
    }
    _save_pred(npz_dir, "pca1024", pred, test)

    for j, enc in enumerate(ENCODERS):
        pred, meta = fit_arm(emb[enc], Yc, tr_c, val_c, te_c, dev, args.ridge_block)
        per_arm[f"enc_{enc}"] = {
            **score_paper(pred, target_mean, view, whiten, seed=190_930 + j),
            "fit_meta": {**meta, "model": ENC.EMBEDDERS[enc], "pooling": ENC.POOLING[enc]},
        }
        _save_pred(npz_dir, f"enc_{enc}", pred, test)
        per_arm[f"floor_{enc}"] = {
            **score_floor(emb[enc][te_c], ans_emb_mean[enc], view, seed=190_950 + j),
            "note": (
                "retrieval-only cos(e(ctx), mean of 4 fresh-rollout answer-text "
                "embeddings); original pass_b answer text not banked (4-of-5 deviation)"
            ),
        }

    for name, row in per_arm.items():
        r2s = "None" if row["r2"] is None else f"{row['r2']:.4f}"
        _log(f"arm {name}: r2={r2s} top1={row['top1']:.4f}")

    C.phase("persist")
    payload = {
        "issue": 1901,
        "analysis": "fig2-baselines-panel",
        "smoke": smoke,
        "layer": LAYER,
        "n_train": int(len(sub)),
        "convention": CONVENTION,
        "target": "mean(original answer vector + four fresh on-policy answer vectors)",
        "retrieval": {
            "metric": "whitened cosine + two-sided CSLS",
            "csls_k": FINAL.K_CSLS,
            "duplicate_policy": "keep_one exact source-answer-vector equivalence class",
            "n_query": view.diagnostics["realized_n_query"],
            "n_pool": view.diagnostics["realized_n_pool"],
            "rank": "strict top-k; mid-rank ties and top ties fail top-1",
        },
        "whitening": whitening_meta,
        "selection": {
            "sel_name": SEL_NAME,
            "seed_b": SEED_B,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "diag": sel_diag,
        },
        "step1_gate": step1,
        "encoder_recipe": {
            "context_input": (
                "full Qwen chat-templated prompt (round-1 convention; no e5 "
                "query:/passage: prefixes, mirroring issue1901_encoder_semantic_baseline)"
            ),
            "models": {e: ENC.EMBEDDERS[e] for e in ENCODERS},
            "pooling": {e: ENC.POOLING[e] for e in ENCODERS},
            "emb_max_tokens": ENC.EMB_MAX_TOKENS,
            "lm_template_model": ENC.LM_MODEL,
        },
        "fresh_rollouts": {"seeds": fresh_meta["seeds"], "n_files": fresh_meta["n_files"]},
        "revisions": {"store": REV_STORE, "fig2": REV_FIG2},
        "split": split,
        "duplicate_audit": view.diagnostics,
        "copy_baselines_source": (
            "eval_results/issue_1901/figure2_extension_1200.json baselines "
            "{identity_copy, identity_bias} — same convention (verified in producer "
            "issue1901_fig2_extension_1200.py, not recomputed here)"
        ),
        "per_arm": per_arm,
        "env": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(out_path, payload)
    _log(f"wrote {out_path}")

    if args.skip_upload:
        _log("upload SKIPPED (--skip-upload)")
        return
    C.phase("upload")
    from huggingface_hub import CommitOperationAdd, HfApi

    sub_prefix = f"{HF_UPLOAD_PREFIX}/smoke" if smoke else HF_UPLOAD_PREFIX
    ops = [CommitOperationAdd(f"{sub_prefix}/{out_path.name}", str(out_path))]
    for p in sorted(npz_dir.glob("pred_*.npz")):
        ops.append(CommitOperationAdd(f"{sub_prefix}/analysis_tensors/{p.name}", str(p)))
    for enc in ENCODERS:
        p = work / f"emb_{enc}{'_smoke' if smoke else ''}.npz"
        ops.append(CommitOperationAdd(f"{sub_prefix}/analysis_tensors/{p.name}", str(p)))
    ops.append(CommitOperationAdd(f"{sub_prefix}/texts/{texts_path.name}", str(texts_path)))
    api = HfApi()
    hub.retry_transient(
        lambda: api.create_commit(
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            operations=ops,
            commit_message=(
                f"issue 1901 fig2-baselines-panel {'smoke ' if smoke else ''}artifacts"
            ),
        ),
        what="fig2-baselines upload commit",
    )
    wanted = sorted(op.path_in_repo for op in ops)
    missing = hub.verify_repo_paths_uploaded(
        api, C.HF_DATA_REPO, wanted, path_in_repo=sub_prefix, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"upload verification FAILED, missing: {missing}")
    _log(f"upload verified: {len(wanted)} files under {sub_prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="issue1901 fig2 baselines on paper convention")
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--orig-dir", type=Path, default=N1M.DEFAULT_ORIG_DIR)
    parser.add_argument("--smoke-chunks", type=int, default=0)
    parser.add_argument("--stage-workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ridge-block", type=int, default=N1M.RIDGE_BLOCK)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--selfcheck-scoring", action="store_true")
    parser.add_argument("--encoder-probe", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.selfcheck_scoring:
        selfcheck_scoring(args)
        return
    if args.encoder_probe:
        encoder_probe(args)
        return
    run(args)


if __name__ == "__main__":
    main()
