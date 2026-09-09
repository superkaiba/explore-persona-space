#!/usr/bin/env python3
"""#1901 floors10k round: score the two zero-parameter cosine floor arms at n_pool 10,000.

Companion to ``issue1901_fig2_pool10k.py``, which rescored every panel-B/C arm at the
10,000-candidate pool EXCEPT the two encoder-space cosine floors (floor_e5,
floor_bge_cls) — their answer-side embeddings existed only for the 1,000 test rows.
This round embeds the banked fresh-rollout ANSWER TEXTS of the first 9,058 rows of the
``issue1901_metrics`` distractor bank order (4 texts per row, seeds 43-46, from the
avgpool-scaleup round's ``issue1901_avgpool/raw_completions/distr`` shards) and scores
the floors against the full 10,000-candidate pool.

Two modes:

  embed  (pod, 1x GPU)  stage the distr rawcomp chunks at the pool10k input pin, encode
                        the 9,058 x 4 answer texts with BOTH encoders (e5 mean-pooled,
                        bge CLS-pooled, emb_max_tokens 512 — ``ENC._encode`` verbatim),
                        per-row mean over the 4 rollouts + L2-normalize (the
                        ``issue1901_encoder_paperconv`` answer-side convention),
                        checkpoint per capture shard, upload the bank-ordered finals to
                        ``issue1901_floors10k/analysis_tensors/`` + scoped verify.
  score  (VM, CPU)      build the keep_one eval view from the staged pass_b bundle,
                        gate-reproduce both floors at n_pool 942 against the banked
                        values to 1e-9 (a miss stops the round; an integer flip count
                        on an in-band rank boundary is characterised, not silently
                        passed), then score both floors at n_pool 10,000
                        (``score_floor`` ops verbatim: fp64 cast, FINAL._cosine,
                        strict ties fail, seeds 190_950/190_951) and fold the results
                        into ``eval_results/issue_1901/fig2_pool10k/fig2_pool10k.json``
                        as a targeted in-place edit of the two panel_c floor entries.

Answer-side deviation (inherited, disclosed): the original pass_b answer text is not
banked, so the answer side is the mean of the FOUR banked fresh-rollout texts
(four of the five target rollouts) — for targets AND distractors alike (homogeneous).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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

from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1901_floors10k")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# Data-repo pin shared with the pool10k round (fig2_pool10k.json hf.revision_pool10k_inputs).
REV_POOL10K = "9f3762e1dd1ac9d4792ea5cc61f7c50d7a507fe5"
RAWCOMP_DISTR_PREFIX = "issue1901_avgpool/raw_completions/distr"
BUNDLE_INDEX_PATH = "issue1901_avgpool/analysis_tensors/bundle/bundle_index.json"
BUNDLE_CI_SHA256 = "c524a8beaeadeba183938f71ced8e65a3479a323bdc2e50ba962cbd832f6d089"
DISTR_BANK_PATH = "issue1901_metrics/analysis_tensors/distractors_L19.npz"
FIG2B_PREFIX = "issue1901_fig2baselines/analysis_tensors"
HF_UPLOAD_PREFIX = "issue1901_floors10k/analysis_tensors"

N_DISTR = 9_058
N_TARGETS = 942
N_POOL_FULL = 10_000
N_SHARDS = 4
GEN_SEEDS = (43, 44, 45, 46)
GEN_CHUNKS_PER_SEED = 10
ENCODERS = ("e5", "bge_cls")
FLOOR_SEEDS = {"e5": 190_950, "bge_cls": 190_951}  # issue1901_fig2_pool10k.PANEL_C_SEEDS
GATE_TOL = 1e-9
# fp-payload characterisation band, mirroring issue1901_fig2_pool10k FP16_MARGIN_BOUND.
MARGIN_BOUND = 5e-3
MAX_FLIPS = 3

# Banked 942 references (eval_results/issue_1901/fig2_baselines/fig2_baselines.json
# per_arm, gate-confirmed by the pool10k round's correctness_gate_942.panel_c).
BANKED_942 = {
    "e5": {"top1": 0.2908704883227176, "top5": 0.3343949044585987},
    "bge_cls": {"top1": 0.2346072186836518, "top5": 0.2898089171974522},
}
# fig2_pool10k.json hf.input_sha256.pass_b (staged-input identity for the eval view).
# SHA_PIN_DOMAIN: BYTES
PASS_B_SHA256 = "46c06e89c513ca598bc83be1c87689694a47bfc927a81d0d738a54df769dbf9a"

FIVE_STAGE = PROJECT_ROOT / "data/issue_1901/figure2_five_rollout_scaling"
POOL10K_STAGE = PROJECT_ROOT / "data/issue_1901/fig2_pool10k"
POOL10K_JSON = PROJECT_ROOT / "eval_results/issue_1901/fig2_pool10k/fig2_pool10k.json"
RESULT_JSON = PROJECT_ROOT / "eval_results/issue_1901/fig2_pool10k/floors10k_result.json"
DEFAULT_OUT_ROOT = (
    Path("/workspace/floors10k")
    if Path("/workspace").exists()
    else (PROJECT_ROOT / "data/issue_1901/floors10k")
)


def _log(msg: str) -> None:
    logger.info(msg)
    print(f"[floors10k] {msg}", flush=True)


def _sha_int_list(xs: list[int]) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in xs).encode()).hexdigest()


def _sha256_file(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _dl(rel: str, stage_root: Path, revision: str = REV_POOL10K) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=rel,
                revision=revision,
                local_dir=stage_root,
            ),
            what=f"stage {rel}",
        )
    )


def _needed_cis(stage_root: Path) -> tuple[list[int], str]:
    """First 9,058 distractor cis in bank order, from the ci-sha-pinned bundle index.

    The pool10k round asserted the bundle distr-ci order equals the
    ``distractors_L19.npz`` bank order over the full 19,000 rows (fig2_pool10k.py
    ``_assemble_distractor_avg``); the score mode re-asserts the first 9,058 directly
    against the local bank copy.
    """
    bpath = _dl(BUNDLE_INDEX_PATH, stage_root)
    bi = json.loads(bpath.read_text())
    if bi["ci_sha256"] != BUNDLE_CI_SHA256:
        raise RuntimeError(f"bundle index ci_sha256 {bi['ci_sha256']} != pinned {BUNDLE_CI_SHA256}")
    recomputed = _sha_int_list([r["ci"] for r in bi["rows"]])
    if recomputed != BUNDLE_CI_SHA256:
        raise RuntimeError("bundle index rows do not reproduce their own ci_sha256")
    distr = [r["ci"] for r in bi["rows"] if r["src"] == "distr"]
    if len(distr) < N_DISTR:
        raise RuntimeError(f"bundle has only {len(distr)} distr rows, need {N_DISTR}")
    cis = [int(c) for c in distr[:N_DISTR]]
    if len(set(cis)) != N_DISTR:
        raise RuntimeError("duplicate ci among the first 9,058 distractor rows")
    return cis, bi["ci_sha256"]


def _load_distr_texts(stage_root: Path, needed: list[int]) -> tuple[dict[int, list[str]], dict]:
    """ci -> [4 seed texts] for every needed distractor row, from the banked rawcomp
    chunks (fail-loud on gaps, empty responses, or bundle-sha drift)."""
    need = set(needed)
    by_seed: dict[int, dict[int, str]] = {s: {} for s in GEN_SEEDS}
    n_files = 0
    for shard in range(N_SHARDS):
        for seed in GEN_SEEDS:
            for chunk in range(GEN_CHUNKS_PER_SEED):
                rel = f"{RAWCOMP_DISTR_PREFIX}/shard{shard:02d}/gen_seed{seed}_chunk{chunk}.json"
                doc = json.loads(_dl(rel, stage_root).read_text())
                n_files += 1
                if doc["meta"]["bundle_sha"] != BUNDLE_CI_SHA256:
                    raise RuntimeError(f"{rel}: bundle_sha != pinned bundle ci sha")
                if int(doc["meta"]["seed"]) != seed:
                    raise RuntimeError(f"{rel}: meta seed {doc['meta']['seed']} != {seed}")
                for r in doc["rows"]:
                    if r["src"] != "distr":
                        raise RuntimeError(f"{rel}: non-distr row src={r['src']}")
                    ci = int(r["ci"])
                    if ci not in need:
                        continue
                    if not r["response"]:
                        raise RuntimeError(f"{rel}: empty response at ci={ci}")
                    if ci in by_seed[seed]:
                        raise RuntimeError(f"duplicate ci {ci} for seed {seed}")
                    by_seed[seed][ci] = r["response"]
    for seed in GEN_SEEDS:
        missing = need - set(by_seed[seed])
        if missing:
            raise RuntimeError(
                f"FEASIBILITY FAIL: seed {seed} missing {len(missing)} of the first "
                f"{N_DISTR} distractor rows (e.g. {sorted(missing)[:5]})"
            )
    texts = {ci: [by_seed[s][ci] for s in GEN_SEEDS] for ci in needed}
    return texts, {"n_chunk_files": n_files, "seeds": list(GEN_SEEDS)}


def embed(args) -> None:
    """Pod-side: encode the distractor answer texts, checkpoint per capture shard,
    upload the bank-ordered finals + meta, scoped-verify."""
    import torch

    import issue1901_encoder_semantic_baseline as ENC

    if not torch.cuda.is_available():
        raise RuntimeError("embed mode requires a GPU (routed to a pod by design)")
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    stage_root = out_root / "staged"

    t0 = time.time()
    needed, bundle_sha = _needed_cis(stage_root)
    texts, text_meta = _load_distr_texts(stage_root, needed)
    _log(f"texts loaded: {len(texts)} rows x {len(GEN_SEEDS)} seeds ({time.time() - t0:.0f}s)")

    # Shard the needed rows by bank-order position (checkpoint unit; resume-safe).
    shards = [needed[i::N_SHARDS] for i in range(N_SHARDS)]
    finals: dict[str, np.ndarray] = {}
    for enc in ENCODERS:
        vec_of: dict[int, np.ndarray] = {}
        for i, shard_cis in enumerate(shards):
            ck = out_root / f"emb_distr_{enc}_shard{i:02d}.npz"
            fp = hashlib.sha256(
                (enc + "|" + ",".join(str(c) for c in shard_cis)).encode()
            ).hexdigest()
            if ck.exists():
                z = np.load(ck, allow_pickle=False)
                if str(z["fp"]) == fp:
                    for ci, v in zip(z["ci"].tolist(), np.asarray(z["e_mean"]), strict=True):
                        vec_of[int(ci)] = v
                    _log(f"embed[{enc}] shard{i:02d}: resumed from checkpoint")
                    continue
                _log(f"embed[{enc}] shard{i:02d}: stale checkpoint fp — re-encoding")
            flat = [texts[ci][j] for ci in shard_cis for j in range(len(GEN_SEEDS))]
            t1 = time.time()
            e = ENC._encode(flat, ENC.EMBEDDERS[enc], "cuda", args.batch, ENC.POOLING[enc]).reshape(
                len(shard_cis), len(GEN_SEEDS), -1
            )
            mean = e.mean(axis=1)
            mean = mean / (np.linalg.norm(mean, axis=1, keepdims=True) + 1e-12)
            np.savez_compressed(
                ck,
                fp=fp,
                ci=np.asarray(shard_cis, dtype=np.int64),
                e_mean=mean.astype(np.float32),
            )
            for ci, v in zip(shard_cis, mean.astype(np.float32), strict=True):
                vec_of[int(ci)] = v
            _log(f"embed[{enc}] shard{i:02d}: {len(flat)} texts in {time.time() - t1:.0f}s")
        final = np.stack([vec_of[ci] for ci in needed]).astype(np.float32)
        assert final.shape == (N_DISTR, 1024), final.shape
        finals[enc] = final
        np.savez_compressed(
            out_root / f"emb_distr_{enc}.npz",
            ci=np.asarray(needed, dtype=np.int64),
            e_ans_mean=final,
            bundle_ci_sha256=bundle_sha,
        )
        _log(f"embed[{enc}]: final ({N_DISTR}, 1024) written")

    meta = {
        "issue": 1901,
        "analysis": "floors10k-distractor-answer-embeddings",
        "n_rows": N_DISTR,
        "row_order": (
            f"first {N_DISTR} rows of {DISTR_BANK_PATH} (bank order; bundle-index "
            "ci-sha-pinned, order equality with the bank re-asserted in score mode)"
        ),
        "answer_side": (
            "mean of the 4 fresh-rollout answer-text embeddings per row, L2-normalized "
            "(issue1901_encoder_paperconv convention; original pass_b answer text not "
            "banked — 4-of-5 deviation, same as the banked test-row e_ans_mean)"
        ),
        "encoder_recipe": {
            "models": {e: ENC.EMBEDDERS[e] for e in ENCODERS},
            "pooling": {e: ENC.POOLING[e] for e in ENCODERS},
            "emb_max_tokens": ENC.EMB_MAX_TOKENS,
            "batch": args.batch,
            "encode_impl": "issue1901_encoder_semantic_baseline._encode (verbatim reuse)",
        },
        "source": {
            "repo": HF_DATA_REPO,
            "revision": REV_POOL10K,
            "rawcomp_prefix": RAWCOMP_DISTR_PREFIX,
            "bundle_index": BUNDLE_INDEX_PATH,
            "bundle_ci_sha256": bundle_sha,
            **text_meta,
        },
        "git_commit": _git_commit(),
        "wall_s": round(time.time() - t0, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = out_root / "floors10k_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    if args.skip_upload:
        _log("upload SKIPPED (--skip-upload)")
        return
    from huggingface_hub import HfApi

    uploads = [out_root / f"emb_distr_{enc}.npz" for enc in ENCODERS] + [meta_path]
    for p in uploads:
        # UPLOAD_LOOP_EXEMPT: fixed 3-file list (2 encoder finals + 1 meta json), bounded by construction
        url = hub._upload(
            p,
            HF_DATA_REPO,
            "dataset",
            f"{HF_UPLOAD_PREFIX}/{p.name}",
            upload_as_file=True,
            raise_on_error=True,
        )
        _log(f"uploaded {p.name} -> {url}")
    expected = [f"{HF_UPLOAD_PREFIX}/{p.name}" for p in uploads]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), HF_DATA_REPO, expected, path_in_repo=HF_UPLOAD_PREFIX
    )
    if missing:
        raise RuntimeError(f"upload verification FAILED, missing: {missing}")
    _log(f"upload verified: {len(expected)} files under {HF_UPLOAD_PREFIX}")


def _characterise_miss(
    enc: str, dist: np.ndarray, true_idx: np.ndarray, delta: float, k: int
) -> dict:
    """Mirror the pool10k round's fp-payload characterisation in cosine-floor space:
    the miss must be a small integer flip count with each flip on a rank boundary
    inside the margin band; anything else is a genuine scoring defect."""
    n_flips = delta * N_TARGETS
    if abs(n_flips - round(n_flips)) > 1e-6 or round(n_flips) > MAX_FLIPS:
        raise RuntimeError(
            f"GATE FAIL floor_{enc} top{k}: delta {delta:.3e} is not a small integer "
            "query-flip count — not payload-attributable; STOP"
        )
    rows = np.arange(len(true_idx))
    truth = dist[rows, true_idx]
    others = dist.copy()
    others[rows, true_idx] = np.inf
    kth_best = np.partition(others, k - 1, axis=1)[:, k - 1]
    margins = kth_best - truth  # positive => true target strictly beats kth competitor
    n_boundary = int((np.abs(margins) <= MARGIN_BOUND).sum())
    if n_boundary < round(n_flips):
        raise RuntimeError(
            f"GATE FAIL floor_{enc} top{k}: {round(n_flips)} flips but only "
            f"{n_boundary} boundary queries within {MARGIN_BOUND} — not attributable; STOP"
        )
    return {
        "delta": float(delta),
        "n_flipped_queries": int(round(n_flips)),
        "n_boundary_queries_within_band": n_boundary,
        "margin_band": MARGIN_BOUND,
        "smallest_abs_margins": [float(x) for x in np.sort(np.abs(margins))[:5]],
    }


def score(args) -> None:
    """VM-side: 942 gate, 10,000-candidate floors, result JSON, targeted fold."""
    import issue1901_figure2_five_rollout_scaling as FIVE
    import issue1901_singleturn_retrieval_final as FINAL

    t0 = time.time()
    pass_b = FIVE_STAGE / "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
    v_test = FIVE_STAGE / "issue1901_avgpool/analysis_tensors/kresample/V_test_shard00.npz"
    for p in (pass_b, v_test):
        if not p.exists():
            raise FileNotFoundError(f"score input missing: {p}")
    realized_sha = _sha256_file(pass_b)
    if realized_sha != PASS_B_SHA256:
        raise RuntimeError(f"pass_b sha {realized_sha} != banked {PASS_B_SHA256}")
    source_target, _target_mean, expected_rows = FIVE._load_five_rollout_target(pass_b, v_test)
    view = FINAL.make_eval_view(source_target, FIVE.N_TEST, "keep_one")
    if (
        view.diagnostics["realized_n_pool"] != N_TARGETS
        or view.diagnostics["realized_n_query"] != N_TARGETS
    ):
        raise RuntimeError(f"unexpected dedup geometry: {view.diagnostics}")
    _log(f"view built ({time.time() - t0:.0f}s); pass_b sha verified")

    # Distractor-order identity, asserted DIRECTLY against the local bank copy.
    bank = POOL10K_STAGE / DISTR_BANK_PATH
    if not bank.exists():
        bank = _dl(DISTR_BANK_PATH, args.harvest_root)
    dz = np.load(bank, allow_pickle=False)
    bank_cis = np.asarray(dz["ci"][:N_DISTR], dtype=np.int64)

    harvest = Path(args.harvest_root)
    emb_shas: dict[str, str] = {}
    distr_emb: dict[str, np.ndarray] = {}
    if not args.gate_only:
        for enc in ENCODERS:
            p = harvest / f"emb_distr_{enc}.npz"
            if not p.exists():
                p = _dl(
                    f"{HF_UPLOAD_PREFIX}/emb_distr_{enc}.npz",
                    harvest,
                    revision=args.emb_revision,
                )
            z = np.load(p, allow_pickle=False)
            cis = np.asarray(z["ci"], dtype=np.int64)
            if not np.array_equal(cis, bank_cis):
                raise RuntimeError(f"emb_distr_{enc} ci order != distractors_L19 bank order")
            e = np.asarray(z["e_ans_mean"], dtype=np.float32)
            assert e.shape == (N_DISTR, 1024), e.shape
            distr_emb[enc] = e
            emb_shas[f"emb_distr_{enc}"] = _sha256_file(p)
        _log("distractor embeddings loaded; ci order == bank order (direct assert)")

    result: dict[str, dict] = {}
    gate_out: dict[str, dict] = {}
    for enc in ENCODERS:
        p_banked = POOL10K_STAGE / f"issue1901_fig2baselines/analysis_tensors/emb_{enc}.npz"
        if not p_banked.exists():
            p_banked = _dl(f"{FIG2B_PREFIX}/emb_{enc}.npz", harvest)
        z = np.load(p_banked, allow_pickle=False)
        rows = np.asarray(z["rows"], dtype=np.int64)
        pos = {int(r): i for i, r in enumerate(rows.tolist())}
        te_pos = np.asarray([pos[int(r)] for r in expected_rows.tolist()], dtype=np.int64)
        e_ctx_test = np.asarray(z["e_ctx"], dtype=np.float32)[te_pos]
        e_ans_mean = np.asarray(z["e_ans_mean"], dtype=np.float32)
        emb_shas[f"emb_{enc}"] = _sha256_file(p_banked)
        seed = FLOOR_SEEDS[enc]

        # 942 gate — verbatim issue1901_encoder_paperconv.score_floor body.
        q = np.asarray(e_ctx_test[view.pred_rows], dtype=np.float64)
        p942 = np.asarray(e_ans_mean[view.pool_rows], dtype=np.float64)
        dist942 = 1.0 - FINAL._cosine(q, p942)
        ranks942 = FINAL._strict_ranks(dist942, view.true_idx)
        s942 = FINAL._rank_summary(ranks942, p942.shape[0], np.random.default_rng(seed))
        banked = BANKED_942[enc]
        d1 = abs(s942["acc_at_k"]["1"] - banked["top1"])
        d5 = abs(s942["acc_at_k"]["5"] - banked["top5"])
        row = {
            "banked_top1": banked["top1"],
            "realized_top1": float(s942["acc_at_k"]["1"]),
            "delta_top1": float(d1),
            "banked_top5": banked["top5"],
            "realized_top5": float(s942["acc_at_k"]["5"]),
            "delta_top5": float(d5),
        }
        if max(d1, d5) <= GATE_TOL:
            row["verdict"] = "PASS"
        else:
            diag = {}
            for k, d in ((1, d1), (5, d5)):
                if d > GATE_TOL:
                    diag[str(k)] = _characterise_miss(enc, dist942, view.true_idx, d, k)
            row["verdict"] = "FAIL_ATTRIBUTED_PAYLOAD_BOUNDARY"
            row["payload_diagnosis"] = diag
            _log(f"GATE floor_{enc}: miss characterised as boundary flips: {diag}")
        gate_out[f"floor_{enc}"] = row
        _log(
            f"gate floor_{enc}: 942 top1={row['realized_top1']:.10f} "
            f"(banked {banked['top1']:.10f}, d={d1:.2e}) verdict={row['verdict']}"
        )
        if args.gate_only:
            continue

        # 10,000-candidate floor: same q, pool = 942 targets + 9,058 distractors.
        p10k = np.concatenate([p942, np.asarray(distr_emb[enc], dtype=np.float64)], axis=0)
        assert p10k.shape == (N_POOL_FULL, 1024), p10k.shape
        dist10k = 1.0 - FINAL._cosine(q, p10k)
        ranks10k = FINAL._strict_ranks(dist10k, view.true_idx)
        s10k = FINAL._rank_summary(ranks10k, p10k.shape[0], np.random.default_rng(seed))
        result[f"floor_{enc}"] = {
            "top1_942": float(s942["acc_at_k"]["1"]),
            "top1_10000": float(s10k["acc_at_k"]["1"]),
            "top5_10000": float(s10k["acc_at_k"]["5"]),
            "top1_ci95_10000": s10k["acc1_ci95"],
            "median_rank_10000": s10k["median_rank"],
            "mrr_10000": s10k["mrr"],
        }
        _log(
            f"floor_{enc}: top1 942={s942['acc_at_k']['1']:.6f} -> "
            f"10k={s10k['acc_at_k']['1']:.6f} (top5 {s10k['acc_at_k']['5']:.6f})"
        )

    if args.gate_only:
        _log("gate-only run complete (no 10k scoring, no result JSON, no fold)")
        return

    provenance = {
        "round": "floors10k (explicit user inline override, task #1901 v154 note)",
        "answer_side": (
            "mean of the 4 fresh-rollout answer-text embeddings per pool row, "
            "L2-normalized (4-of-5 deviation: original pass_b answer text not banked; "
            "applies to targets and distractors alike — homogeneous pool)"
        ),
        "pool_composition": (
            f"{N_TARGETS} keep_one test targets (banked emb_*.npz e_ans_mean) + first "
            f"{N_DISTR} rows of {DISTR_BANK_PATH} in bank order (fresh distractor "
            "answer-text embeddings, this round; ci order asserted == bank order)"
        ),
        "scoring": (
            "plain cosine in encoder space (no whitening, no CSLS), strict top-k with "
            "ties failing — issue1901_encoder_paperconv.score_floor ops verbatim; "
            "seeds 190950 (e5) / 190951 (bge_cls)"
        ),
        "encoder_recipe": {
            "models": {
                "e5": "intfloat/multilingual-e5-large",
                "bge_cls": "BAAI/bge-large-en-v1.5",
            },
            "pooling": {"e5": "mean", "bge_cls": "cls"},
            "emb_max_tokens": 512,
        },
        "hf_distr_emb_prefix": HF_UPLOAD_PREFIX,
        "input_sha256": {**emb_shas, "pass_b": realized_sha},
        "gate_942": gate_out,
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(
        json.dumps({"floors": result, "provenance": provenance}, indent=2) + "\n"
    )
    _log(f"wrote {RESULT_JSON}")

    if args.skip_fold:
        _log("fold SKIPPED (--skip-fold)")
        return
    # Targeted in-place fold: replace ONLY the two panel_c floor entries.
    doc = json.loads(POOL10K_JSON.read_text())
    for enc in ENCODERS:
        name = f"floor_{enc}"
        old = doc["panel_c"][name]
        if old.get("top1_10000") is not None and not args.force_fold:
            raise RuntimeError(f"{name}: top1_10000 already set — refusing to overwrite")
        entry = {"top1_942": old["top1_942"], **result[name]}
        entry["status_10000"] = "computed"
        entry["reason_10000_superseded"] = (
            "distractor answer-text embeddings computed by the floors10k round "
            "(see floors10k provenance below); previously "
            "not_computable_from_banked_artifacts"
        )
        entry["floors10k"] = {
            k: provenance[k]
            for k in (
                "round",
                "answer_side",
                "pool_composition",
                "scoring",
                "hf_distr_emb_prefix",
                "git_commit",
                "timestamp_utc",
            )
        }
        entry["floors10k"]["gate_942"] = gate_out[name]
        entry["floors10k"]["encoder_recipe"] = provenance["encoder_recipe"]
        if abs(entry["top1_942"] - old["top1_942"]) > 0.0:
            raise RuntimeError(f"{name}: realized 942 != banked 942 in the doc — STOP")
        doc["panel_c"][name] = entry
    tmp = POOL10K_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(POOL10K_JSON)
    _log(f"folded both floor arms into {POOL10K_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("embed", help="pod-side encode + upload")
    e.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--skip-upload", action="store_true")
    s = sub.add_parser("score", help="VM-side gate + 10k floors + fold")
    s.add_argument("--harvest-root", type=Path, default=PROJECT_ROOT / "data/issue_1901/floors10k")
    s.add_argument("--emb-revision", type=str, default=None)
    s.add_argument("--skip-fold", action="store_true")
    s.add_argument("--force-fold", action="store_true")
    s.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "embed":
        embed(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
