#!/usr/bin/env python3
"""Issue #1901 inline round `avgpool-scaleup` — HOMOGENEOUS draw-averaged retrieval
pool ladder at layer 19 (user-chat inline GPU override).

Question: how does context->answer retrieval accuracy scale with pool size when
EVERY pool entry (query targets AND distractors alike) is the mean of the
original answer state plus K=4 fresh on-policy draws — removing the
averaged-target-vs-single-draw-distractor asymmetry that put the #2202
draw-averaged read at ceiling (acc@1 0.9945 on a 9,941 pool)?

Lineage: the #1901 metric-battery context arm (queries = the 1,000 pass_b test
rows; ridge = the banked 963k-train n1m map; distractors = the banked
issue1901_metrics/analysis_tensors/distractors_L19.npz order). The banked
single-draw ladder (raw-cosine CSLS acc@1: test-1000 0.848 / passb-5000 0.756 /
distr-20000 0.714 / distr-100000 0.647) is the contrast curve; this round adds
the homogeneous averaged counterpart at nested rungs {1k, 2k, 5k, 20k} plus the
matched single-draw read at the SAME pool composition (the core 2x2).

Reused verbatim (repo reuse rule — no re-derivation):
- generation + capture: issue1482_kresample helpers (the n1m-line K-resample
  rig): per-request seeds 43-46, engine seed 42, SamplingParams(n=1,
  temperature=1.0, top_p=0.95, max_tokens=1024), max_model_len 8192,
  Qwen/Qwen2.5-7B-Instruct; capture = the verbatim parent convention
  (full-chat-template re-tokenization of prompt+response, span
  [prompt_len:full_len] incl. the end-of-turn tail, mean over the answer span,
  layer 19) via EA._batched_capture.
- retrieval battery: issue1901_metric_battery PoolSpec / eval_retrieval_cell /
  csls_scores (Conneau two-sided CSLS, k=10) / rank_matrix_for_cols (mid-ranks,
  1e-9 relative tie tolerance) — the machinery that produced the banked curve.
- whitened read: z = L^-1(x - mu_A) with the task-locked #2202 shrunk
  train-answer Cholesky (issue2202_ctxfail/analysis_tensors/whiten_stats.npz,
  lam=0.1, n_train=87,795). CROSS-LINE CAVEAT (stated in the eval JSON): those
  stats are fitted on the #1738 multiturn train answers and applied here to the
  n1m single-turn line. CSLS on the whitened-cosine similarity uses the SAME
  battery csls_scores formulation (two-sided), NOT the #2202 one-sided variant —
  this round extends the battery's curve, so the battery formulation wins.
- pass_b test-row prompt recovery: N50.sample_disjoint_n50k round-1 re-derivation
  (sha-asserted against the n1m manifest meta) + F.fixed_split test indices.

Phases: bundle -> pilot (measured per-draw gen+capture basis; >fence GPU-h =
designed halt rc 31) -> gen (per-shard vLLM, rollout text uploaded per seed
BEFORE capture) -> cap (per-shard HF teacher-forced capture, V fp16 uploaded)
-> score (repro gate vs the banked context_arm ridge numbers, then the
2x2 x rung battery) -> figure (VM-side).

Refusal-safety: LMSYS prompt/rollout text is never printed or logged — counts
and digests only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# vLLM V1 fork-safety (#628): spawn BEFORE any vllm import in this process tree.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM discipline)

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402  (apply_map)
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402  (manifest, pass_b)
import issue779_ffc_n10k_generate_capture as N10  # noqa: E402  (_sha_prompts)
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402  (round-1 re-derivation)
import issue779_fitter_fair_comparison as F  # noqa: E402  (load_pass_b, fixed_split)
import issue779_percontext_recon as PR  # noqa: E402  (_pooled_r2)
import issue1482_error_analysis as EA  # noqa: E402  (_load_model_tok, _batched_capture)
import issue1482_kresample as KR  # noqa: E402  (n1m-line generate/capture helpers)
import issue1901_metric_battery as MB  # noqa: E402  (PoolSpec, eval_retrieval_cell)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from explore_persona_space.orchestrate.secret_scrub import scan_file, scrub_file  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1901_avgpool")

ISSUE = 1901
LAYER = 19
H_DIM = C.EXPECTED_HIDDEN  # 3584
N_TEST = 1_000
N_DISTR_MAX = 19_000  # the banked distr_20000 composition (first rows of the npz order)
RUNG_TOTALS = (1_000, 2_000, 5_000, 20_000)  # nested: test + first (total-1000) distractors
K_DRAWS = len(KR.GEN_SEEDS)  # 4 (seeds 43-46; engine seed 42)
GEN_CHUNK = KR.GEN_CHUNK  # 500 rows per persisted generation chunk
PILOT_N = 100
FENCE_GPU_H = 20.0
RC_FENCE = 31  # designed-halt rc (pilot fence; report written first)
BOOT_N = 1_000
K_PERM = 200
DRAWS_SEED = 190_119  # this round's bootstrap/null seed (CIs are round-local)
REPRO_ATOL = 3e-3  # banked-curve reproduction tolerance on acc@k/MRR (cross-machine ties)

HF_ROUND_PREFIX = "issue1901_avgpool"
DISTR_REPO_PATH = "issue1901_metrics/analysis_tensors/distractors_L19.npz"
WHITEN_REPO_PATH = "issue2202_ctxfail/analysis_tensors/whiten_stats.npz"
BANKED_CONTEXT_ARM = PROJECT_ROOT / "eval_results/issue_1901/metric_battery/context_arm.json"
OUT_EVAL = PROJECT_ROOT / "eval_results" / "issue_1901" / "avgpool_scaleup"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1901" / "avgpool_scaleup"
ARMS = ("ridge", "mlp_w8192", "identity_bias")
ARM_LABELS = {
    "ridge": "linear map (ridge)",
    "mlp_w8192": "nonlinear map (MLP)",
    "identity_bias": "copy context vector + trained bias",
}


# ── shared small helpers ──────────────────────────────────────────────────────────


def _sha_int_list(xs: list[int]) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in xs).encode()).hexdigest()


def _staged(args, repo_rel: str, *, revision: str | None = None) -> Path:
    dest = args.out_root / "staged" / repo_rel
    hub.stage_hub_file(C.HF_DATA_REPO, repo_rel, dest, repo_type="dataset", revision=revision)
    return dest


def _upload_failloud(args, path: Path, sub: str) -> None:
    """Fail-loud single-file upload to this round's own HF bucket."""
    dest = f"{HF_ROUND_PREFIX}/{sub}/{path.name}"
    url = ""
    for attempt, pause in enumerate((0, 30, 60)):
        if pause:
            time.sleep(pause)
            logger.warning("[upload] retry %d for %s", attempt, path.name)
        # UPLOAD_LOOP_EXEMPT: bounded per-file retry around fail-soft hub._upload (#1315 seam)
        url = hub._upload(
            path,
            C.HF_DATA_REPO,
            "dataset",
            dest,
            upload_as_file=True,
            raise_on_error=(attempt == 2),
        )
        if url:
            return
    raise RuntimeError(f"upload returned no URL for {path} -> {dest}")


def _pipeline_scrub_pre_upload(path: Path) -> list[dict]:
    """Unconditional pre-upload scrub STEP: scan every generation chunk before
    upload and replace MODEL-GENERATED secret-shaped strings in place with
    same-length X placeholders, so the ARMED secret upload gate (never
    bypassed) receives clean input. The same distractor context elicits the
    same confabulated JWT on every seed, so a reactive per-file fix costs one
    crash per seed. Auto-scrub is scoped to ``jwt-signed`` findings matching
    no live credential; any OTHER pattern, or a credential match, raises
    instead of scrubbing blind. Returns disclosure records for the eval JSON."""
    findings = scan_file(path)
    if not findings:
        return []
    raw = path.read_bytes()
    creds = {v for v in os.environ.values() if v and len(v) >= 12}
    for f in findings:
        tok = raw[f.offset : f.offset + f.length].decode("utf-8", "replace")
        if f.pattern != "jwt-signed":
            raise RuntimeError(
                f"pre-upload scrub: non-jwt finding ({f.pattern}) in {path.name} — outside "
                "the sanctioned auto-scrub class; stopping for review (never scrub blind)"
            )
        if any(tok == c or tok in c or c in tok for c in creds):
            raise RuntimeError(
                f"pre-upload scrub: finding in {path.name} MATCHES a live credential — "
                "security incident; refusing to scrub, stopping for review"
            )
    scrubbed = scrub_file(path)  # in-place, same-length fill; self-verifies clean
    doc = json.loads(path.read_text())
    rows_hit = [
        {"row_idx": r["row_idx"], "ci": r["ci"], "src": r["src"], "field": "response"}
        for r in doc["rows"]
        if isinstance(r.get("response"), str) and "X" * 24 in r["response"]
    ]
    return [
        {
            "file": str(path.relative_to(path.parents[3])),
            "rule": f.pattern,
            "masked": f.masked,
            "offset": f.offset,
            "length": f.length,
            "rows_with_placeholders": rows_hit,
            "note": (
                "model-generated secret-shaped string replaced in place by a same-length X "
                "placeholder before upload (armed secret upload gate; per-token credential "
                "comparison: no match); capture runs on the scrubbed text"
            ),
        }
        for f in scrubbed
    ]


def _bundle_paths(args) -> tuple[Path, Path]:
    return args.out_root / "bundle" / "bundle.json", args.out_root / "bundle" / "bundle_meta.json"


def _load_bundle(args) -> tuple[list[dict], str]:
    bpath, _ = _bundle_paths(args)
    doc = json.loads(bpath.read_text())
    rows = doc["rows"]
    sha = _sha_int_list([r["ci"] for r in rows])
    assert sha == doc["ci_sha256"], "bundle ci sha mismatch — stale/corrupt bundle"
    return rows, sha


# ── phase: bundle (stage inputs + build the 20k-row generation bundle) ────────────


def phase_bundle(args) -> None:
    C.phase("bundle")
    out = args.out_root / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    bpath, mpath = _bundle_paths(args)
    if bpath.exists() and mpath.exists():
        # resume: the heavy build is done; (re)emit + upload the text-free index
        # (idempotent — covers a prior run that died at the upload step).
        rows_doc = json.loads(bpath.read_text())
        kept_rows = rows_doc["rows"]
        ipath = out / "bundle_index.json"
        C.write_json_atomic(
            ipath,
            {
                "ci_sha256": rows_doc["ci_sha256"],
                "rows": [
                    {
                        "row_idx": r["row_idx"],
                        "ci": r["ci"],
                        "src": r["src"],
                        "prompt_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest(),
                    }
                    for r in kept_rows
                ],
            },
        )
        if not args.skip_upload:
            _upload_failloud(args, ipath, "analysis_tensors/bundle")
            _upload_failloud(args, mpath, "analysis_tensors/bundle")
        logger.info("[bundle] %s exists — index re-emitted; skip rebuild (resume)", bpath)
        return

    # distractors: the banked battery pool order (first 19,000 = distr_20000 comp).
    dp = _staged(args, DISTR_REPO_PATH)
    blob = np.load(dp)
    dvx, dci, dcorpus = blob["vx"], blob["ci"], blob["corpus"]
    assert dvx.shape[0] >= N_DISTR_MAX and dvx.shape[1] == H_DIM, dvx.shape
    dci = np.asarray(dci[:N_DISTR_MAX], dtype=np.int64)
    assert (np.asarray(dcorpus[:N_DISTR_MAX]) == "lmsys").all(), (
        "first 19,000 distractor rows must all be lmsys (banked distr_20000 composition)"
    )
    assert len(set(dci.tolist())) == N_DISTR_MAX, "duplicate ci among the 19,000 distractors"

    # manifest: ci -> prompt for the distractor rows.
    mdir = N1G._download_manifest(N1G.HF_PREFIX, args.out_root / "manifest")
    pool, meta = N1G.read_manifest_pool(mdir)
    d_prompts = []
    for ci in dci.tolist():
        row = pool[int(ci)]
        assert int(row["i"]) == int(ci), (row.get("i"), ci)
        d_prompts.append(row["prompt"])

    # pass_b test-row prompts: round-1 re-derivation, sha-asserted vs manifest meta.
    r1 = N50.sample_disjoint_n50k(N1G.N_ROUND1, 0, 0)
    round1 = r1["round1"]
    assert N10._sha_prompts(round1) == meta["used_shas"]["round1"], (
        "re-derived round-1 prompt sha != n1m manifest meta used_shas.round1 — "
        "LMSYS stream ordering drift; pass_b test prompts not trustworthy"
    )
    norm0 = " ".join(round1[0].lower().split()).rstrip(".?!,")
    assert norm0 == N10.EXPECTED_CTX0_PROMPT, "round-1 ctx0 re-derivation drift"
    _tr, _val, test = F.fixed_split(
        N1G.N_ROUND1, N1G.N_ROUND1 - 400 - 1000, 400, 1000, F.SPLIT_SEED
    )
    assert len(test) == N_TEST
    t_prompts = [round1[i] for i in test]

    # over-length screen (parent budget): rendered prompt tokens <= budget.
    from transformers import AutoTokenizer

    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained(KR.MODEL_ID), what="tokenizer fetch"
    )
    budget = KR.MAX_MODEL_LEN - KR.GEN_MAX_TOKENS
    rows: list[dict] = []
    n_over = 0
    for pos, p in enumerate(t_prompts):
        rows.append({"row_idx": len(rows), "ci": -(1 + pos), "prompt": p, "src": "test"})
    for ci, p in zip(dci.tolist(), d_prompts, strict=True):
        rows.append({"row_idx": len(rows), "ci": int(ci), "prompt": p, "src": "distr"})
    kept_rows = []
    over_cis = []
    for r in rows:
        text = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        if len(tok(text, add_special_tokens=False)["input_ids"]) > budget:
            n_over += 1
            over_cis.append(r["ci"])
            continue
        kept_rows.append(r)
    # parents screened over-length at capture time, so this should be ~0; a dropped
    # row leaves the pool at score time (matched composition) — never a crash.
    logger.info("[bundle] %d rows kept (%d over-length dropped)", len(kept_rows), n_over)
    for i, r in enumerate(kept_rows):
        r["row_idx"] = i

    cis = [r["ci"] for r in kept_rows]
    assert len(set(cis)) == len(cis), "bundle ci collision"
    C.write_json_atomic(bpath, {"rows": kept_rows, "ci_sha256": _sha_int_list(cis)})
    # HF provenance = TEXT-FREE index only (real-corpus prompt text stays pod-local:
    # refusal-safety + the secret upload gate both bar raw LMSYS text; every prompt
    # is re-derivable — distr from the pinned n1m manifest by ci, test from the
    # sha-asserted round-1 re-derivation recorded in bundle_meta).
    ipath = out / "bundle_index.json"
    C.write_json_atomic(
        ipath,
        {
            "ci_sha256": _sha_int_list(cis),
            "rows": [
                {
                    "row_idx": r["row_idx"],
                    "ci": r["ci"],
                    "src": r["src"],
                    "prompt_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest(),
                }
                for r in kept_rows
            ],
        },
    )
    C.write_json_atomic(
        mpath,
        {
            "n_rows": len(kept_rows),
            "n_test": sum(1 for r in kept_rows if r["src"] == "test"),
            "n_distr": sum(1 for r in kept_rows if r["src"] == "distr"),
            "n_over_length_dropped": n_over,
            "over_length_cis": over_cis,
            "distr_repo_path": DISTR_REPO_PATH,
            "manifest_prefix": f"{N1G.HF_PREFIX}/{N1G.MANIFEST_SUBDIR}",
            "round1_sha256": N10._sha_prompts(round1),
            "test_idx_sha256": F._sha_ids(np.asarray(test)),
            "gen_recipe": {
                "model": KR.MODEL_ID,
                "seeds": list(KR.GEN_SEEDS),
                "engine_seed": 42,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": KR.GEN_MAX_TOKENS,
                "max_model_len": KR.MAX_MODEL_LEN,
                "prompt_render": "single user turn, chat template, add_generation_prompt",
                "capture": (
                    "parent convention: full-chat-template re-tokenization of prompt+response, "
                    f"answer-span mean over [prompt_len:full_len] incl. end-of-turn tail, L{LAYER}"
                ),
            },
            **C.reproducibility_metadata(),
        },
    )
    if not args.skip_upload:
        _upload_failloud(args, ipath, "analysis_tensors/bundle")
        _upload_failloud(args, mpath, "analysis_tensors/bundle")
    logger.info("[bundle] wrote %s", bpath)


# ── phases: gen + cap (per-shard; reuse the #1482 kresample rig helpers) ──────────


def _shard_rows(rows: list[dict], g: int, n: int, src: str) -> list[dict]:
    """Source-filtered strided shard. src='test' rows may instead be IMPORTED from
    the sibling avgtarget-plots round's persisted rollout text (same recipe family,
    #779 pass-B: temp 1.0 / top_p 0.95 / max_tokens 1024, seeds 43-46) — this round
    then re-captures them under ITS OWN batched rig so every pool entry's draw
    vectors share one capture shape (no test-vs-distractor capture asymmetry)."""
    sel = rows if src == "all" else [r for r in rows if r["src"] == src]
    return sel[g::n]


def _gen_dir(args, g: int) -> Path:
    return args.out_root / "gen" / args.src / f"shard{g:02d}"


def phase_gen(args) -> None:
    C.phase(f"gen-{args.src}-shard{args.shard_index}")
    rows, sha = _load_bundle(args)
    rows = _shard_rows(rows, args.shard_index, args.num_shards, args.src)
    from transformers import AutoTokenizer

    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained(KR.MODEL_ID), what="tokenizer fetch"
    )
    prompt_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in rows
    ]
    llm = None
    if args.device == "cuda":
        from explore_persona_space.eval.generation import create_vllm_engine

        llm = create_vllm_engine(KR.MODEL_ID, max_model_len=KR.MAX_MODEL_LEN, seed=42)
    gdir = _gen_dir(args, args.shard_index)
    gdir.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(len(rows) / GEN_CHUNK)
    det_report: dict | None = None
    t_phase = time.time()
    for k in KR.GEN_SEEDS:
        seed_paths = []
        for j in range(n_chunks):
            ck = gdir / f"gen_seed{k}_chunk{j}.json"
            seed_paths.append(ck)
            if ck.exists():
                cmeta = json.loads(ck.read_text()).get("meta", {})
                if cmeta.get("bundle_sha") == sha and cmeta.get("ids_version") == KR.IDS_VERSION:
                    logger.info("[gen] %s complete — skip", ck.name)
                    continue
                logger.warning("[gen] %s stale (sha/ids mismatch) — regenerating", ck.name)
            lo, hi = j * GEN_CHUNK, min((j + 1) * GEN_CHUNK, len(rows))
            t0 = time.time()
            gen_rows = KR._generate_seed(llm, tok, prompt_texts[lo:hi], seed=k)
            if det_report is None:
                det_report = KR._determinism_spot_check(
                    llm, prompt_texts[lo : lo + 5], k, [g["text"] for g in gen_rows[:5]]
                )
            C.write_json_atomic(
                ck,
                {
                    "meta": {
                        "bundle_sha": sha,
                        "ids_version": KR.IDS_VERSION,
                        "seed": k,
                        "chunk": j,
                        "shard": args.shard_index,
                        "n": hi - lo,
                        "chunk_wall_s": round(time.time() - t0, 2),
                        "sampling": {
                            "n": 1,
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "max_tokens": KR.GEN_MAX_TOKENS,
                            "seed": k,
                            "engine_seed": 42,
                            "max_model_len": KR.MAX_MODEL_LEN,
                            "model": KR.MODEL_ID,
                        },
                        **C.reproducibility_metadata(),
                    },
                    "rows": [
                        {
                            "row_idx": r["row_idx"],
                            "ci": r["ci"],
                            "src": r["src"],
                            "response": g["text"],
                            "token_ids": g["token_ids"],
                            "prompt_token_ids": g["prompt_token_ids"],
                        }
                        for r, g in zip(rows[lo:hi], gen_rows, strict=True)
                    ],
                },
            )
            logger.info(
                "[gen] unit %d/%d seed=%d shard=%d elapsed=%.0fs",
                (list(KR.GEN_SEEDS).index(k) * n_chunks) + j + 1,
                len(KR.GEN_SEEDS) * n_chunks,
                k,
                args.shard_index,
                time.time() - t_phase,
            )
        # rollout text uploads at the END of each seed, fail-loud, BEFORE capture;
        # every chunk passes the pre-upload scrub step first (see helper docstring).
        if not args.skip_upload:
            seed_subs: list[dict] = []
            for p in seed_paths:
                seed_subs.extend(_pipeline_scrub_pre_upload(p))
                _upload_failloud(args, p, f"raw_completions/{args.src}/shard{args.shard_index:02d}")
            if seed_subs:
                spath = gdir / f"scrub_disclosures_seed{k}.json"
                prior = json.loads(spath.read_text()) if spath.exists() else []
                C.write_json_atomic(spath, prior + seed_subs)
                logger.warning(
                    "[gen] pre-upload scrub: %d finding(s) recorded in %s",
                    len(seed_subs),
                    spath.name,
                )
    C.write_json_atomic(
        gdir / "gen_meta.json",
        {
            "bundle_sha": sha,
            "n_rows": len(rows),
            "n_chunks": n_chunks,
            "seeds": list(KR.GEN_SEEDS),
            "determinism_check": det_report,
            "wall_s": round(time.time() - t_phase, 1),
            **C.reproducibility_metadata(),
        },
    )
    logger.info("[gen] shard %d done in %.1f min", args.shard_index, (time.time() - t_phase) / 60)


def _load_gen(args, g: int, sha: str) -> dict[int, dict[int, dict]]:
    out: dict[int, dict[int, dict]] = {k: {} for k in KR.GEN_SEEDS}
    for k in KR.GEN_SEEDS:
        for ck in sorted(_gen_dir(args, g).glob(f"gen_seed{k}_chunk*.json")):
            doc = json.loads(ck.read_text())
            assert doc["meta"].get("bundle_sha") == sha, f"{ck.name}: bundle sha mismatch"
            assert doc["meta"].get("ids_version") == KR.IDS_VERSION, f"{ck.name}: stale ids"
            for r in doc["rows"]:
                out[k][int(r["ci"])] = r
    return out


def phase_cap(args) -> None:
    C.phase(f"cap-{args.src}-shard{args.shard_index}")
    all_rows, sha = _load_bundle(args)
    rows = _shard_rows(all_rows, args.shard_index, args.num_shards, args.src)
    gen = _load_gen(args, args.shard_index, sha)
    for k in KR.GEN_SEEDS:
        missing = [r["ci"] for r in rows if int(r["ci"]) not in gen[k]]
        assert not missing, f"seed {k}: {len(missing)} shard rows missing from gen chunks"
    v_path = args.out_root / "cap" / f"V_{args.src}_shard{args.shard_index:02d}.npz"
    meta_path = args.out_root / "cap" / f"capture_meta_{args.src}_shard{args.shard_index:02d}.json"
    v_path.parent.mkdir(parents=True, exist_ok=True)
    if v_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()).get("bundle_sha") == sha:
            logger.info("[cap] %s complete — skip", v_path.name)
        else:
            raise RuntimeError(f"[cap] stale {v_path} (bundle sha mismatch) — remove it")
    else:
        model, tok = EA._load_model_tok(args)
        kept, dropped, items = [], [], []
        for r in rows:
            msgs, prompt_ids = KR._prompt_render(tok, r["prompt"])
            plen = len(prompt_ids)
            per_draw: list[tuple] | None = []
            for seed in KR.GEN_SEEDS:
                gr = gen[seed][int(r["ci"])]
                # sibling-imported rollout text carries no generation-time ids —
                # the join check is a diagnostic; capture needs only the text.
                if gr.get("prompt_token_ids"):
                    KR._check_prompt_ids_join(
                        int(r["ci"]), seed, gr["prompt_token_ids"], prompt_ids
                    )
                full_ids = KR._parent_convention_full_ids(tok, msgs, gr["response"])
                if len(full_ids) <= plen:  # the parent's empty-span drop
                    per_draw = None
                    break
                per_draw.append((full_ids, plen - 1, plen - 1, len(full_ids) - plen, 0))
            if per_draw is None:
                dropped.append({"row_idx": r["row_idx"], "ci": r["ci"], "src": r["src"]})
                continue
            slot = len(kept)
            kept.append(r)
            items.extend((slot, d, tk) for d, tk in enumerate(per_draw))
        n = len(kept)
        logger.info("[cap] capturing %d contexts x %d draws (%d dropped)", n, K_DRAWS, len(dropped))
        V = np.zeros((n, K_DRAWS, H_DIM), dtype=np.float32)
        n_ans = np.zeros((n, K_DRAWS), dtype=np.int32)
        t0, done = time.time(), 0
        with torch.no_grad():
            for batch in KR._token_batches(items, args.gen_batch, args.token_budget):
                batch_rows = [(slot, 0, tk[0], tk[1], tk[2], tk[3], tk[4]) for slot, _, tk in batch]
                outs = EA._batched_capture(model, tok, batch_rows, [LAYER], args.device)
                for (slot, d, tk), out in zip(batch, outs, strict=True):
                    full_ids, _pe, context_end, na, _seam = tk
                    h = out[LAYER]
                    assert h.shape[0] == len(full_ids), (h.shape, len(full_ids))
                    V[slot, d] = h[context_end + 1 :, :].mean(0).numpy()
                    n_ans[slot, d] = na
                done += len(batch)
                if done % 500 < len(batch):
                    logger.info(
                        "[cap] unit %d/%d shard=%d elapsed=%.0fs",
                        done,
                        len(items),
                        args.shard_index,
                        time.time() - t0,
                    )
        np.savez(  # plain savez — never savez_compressed in the hot path (#813)
            v_path,
            V=V.astype(np.float16),
            n_ans=n_ans,
            ci=np.array([r["ci"] for r in kept], dtype=np.int64),
            src=np.array([r["src"] for r in kept]),
            draws=np.array(KR.GEN_SEEDS, dtype=np.int64),
        )
        C.write_json_atomic(
            meta_path,
            {
                "bundle_sha": sha,
                "layer": LAYER,
                "n_kept": n,
                "dropped": dropped,
                "capture_convention": "parent-full-template-retok-span-incl-eot-tail",
                "model": KR.MODEL_ID,
                "tiny_model": bool(args.tiny_model),
                **C.reproducibility_metadata(),
            },
        )
    if not args.skip_upload:
        _upload_failloud(args, v_path, "analysis_tensors/kresample")
        _upload_failloud(args, meta_path, "analysis_tensors/kresample")
    logger.info("[cap] shard %d done", args.shard_index)


# ── phase: pilot (measured per-draw gen + capture basis at production shape) ──────


def phase_pilot(args) -> None:
    C.phase("pilot")
    rows, sha = _load_bundle(args)
    pilot_rows = rows[:: max(1, len(rows) // PILOT_N)][:PILOT_N]
    pdir = args.out_root / "pilot"
    pdir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained(KR.MODEL_ID), what="tokenizer fetch"
    )
    prompt_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in pilot_rows
    ]
    from explore_persona_space.eval.generation import create_vllm_engine

    t_load = time.time()
    llm = create_vllm_engine(KR.MODEL_ID, max_model_len=KR.MAX_MODEL_LEN, seed=42)
    load_s = time.time() - t_load
    t0 = time.time()
    gen_rows = KR._generate_seed(llm, tok, prompt_texts, seed=KR.GEN_SEEDS[0])
    gen_s = time.time() - t0
    del llm
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    t_cap_load = time.time()
    model, _tok2 = EA._load_model_tok(args)
    cap_load_s = time.time() - t_cap_load
    items = []
    for slot, (r, g) in enumerate(zip(pilot_rows, gen_rows, strict=True)):
        msgs, prompt_ids = KR._prompt_render(tok, r["prompt"])
        full_ids = KR._parent_convention_full_ids(tok, msgs, g["text"])
        if len(full_ids) <= len(prompt_ids):
            continue
        items.append((slot, 0, (full_ids, len(prompt_ids) - 1, len(prompt_ids) - 1, 0, 0)))
    t1 = time.time()
    with torch.no_grad():
        for batch in KR._token_batches(items, args.gen_batch, args.token_budget):
            batch_rows = [(s, 0, tk[0], tk[1], tk[2], tk[3], tk[4]) for s, _, tk in batch]
            EA._batched_capture(model, tok, batch_rows, [LAYER], args.device)
    cap_s = time.time() - t1

    n_total_rows = len(rows)
    total_draws = n_total_rows * K_DRAWS
    per_draw_gen = gen_s / len(gen_rows)
    per_draw_cap = cap_s / max(1, len(items))
    overhead_h = args.num_shards * (load_s + cap_load_s) / 3600.0
    gpu_h = (per_draw_gen + per_draw_cap) * total_draws / 3600.0 + overhead_h
    wall_h = (
        (per_draw_gen + per_draw_cap) * total_draws / args.num_shards + load_s + cap_load_s
    ) / 3600.0
    report = {
        "gate": "avgpool-pilot",
        "pilot_rows": len(pilot_rows),
        "pilot_gen_s": round(gen_s, 2),
        "pilot_cap_s": round(cap_s, 2),
        "engine_load_s": round(load_s, 1),
        "cap_model_load_s": round(cap_load_s, 1),
        "measured_s_per_draw_gen": per_draw_gen,
        "measured_s_per_draw_cap": per_draw_cap,
        "projected_total_draws": total_draws,
        "num_shards": args.num_shards,
        "projected_gpu_h": gpu_h,
        "projected_wall_h": wall_h,
        "fence_gpu_h": FENCE_GPU_H,
        "verdict": "PASS" if gpu_h <= FENCE_GPU_H else "FENCE",
        "bundle_sha": sha,
        **C.reproducibility_metadata(),
    }
    C.write_json_atomic(pdir / "pilot_report.json", report)
    if not args.skip_upload:
        _upload_failloud(args, pdir / "pilot_report.json", "analysis_tensors/pilot")
    logger.info("[pilot] %s", json.dumps({k: v for k, v in report.items() if k != "meta"}))
    if gpu_h > FENCE_GPU_H:
        logger.error("[pilot] FENCE: %.1f GPU-h > %.1f — designed halt", gpu_h, FENCE_GPU_H)
        sys.exit(RC_FENCE)


# ── phase: score (repro gate + the 2x2 x rung battery) ────────────────────────────


def _whiten_fn(stats_path: Path):
    z = np.load(stats_path)
    mu_a = np.asarray(z["mu_A"], dtype=np.float64)
    ell = np.asarray(z["L"], dtype=np.float64)
    meta = {"lam": float(z["lam"]), "n_train": int(z["n_train"])}

    def _wh(x: np.ndarray) -> np.ndarray:
        return solve_triangular(ell, (np.asarray(x, np.float64) - mu_a).T, lower=True).T

    return _wh, meta


def _assemble_draws(args, sha: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(vsum fp64 (n,H), ci (n,), dropped records) across ALL capture shards on
    disk (every src x shard: V_<src>_shardNN.npz beside capture_meta_<src>_shardNN)."""
    v_paths = sorted((args.out_root / "cap").glob("V_*_shard*.npz"))
    assert v_paths, f"no capture shards under {args.out_root / 'cap'}"
    v_parts, ci_parts, dropped = [], [], []
    for v_path in v_paths:
        meta_path = v_path.with_name(v_path.name.replace("V_", "capture_meta_", 1)).with_suffix(
            ".json"
        )
        m = json.loads(meta_path.read_text())
        assert m["bundle_sha"] == sha, f"{v_path.name}: capture bundle sha mismatch"
        z = np.load(v_path)
        v = z["V"].astype(np.float64)  # (n_g, K, H)
        assert v.shape[1] == K_DRAWS and v.shape[2] == H_DIM, v.shape
        v_parts.append(v.sum(axis=1))
        ci_parts.append(np.asarray(z["ci"], dtype=np.int64))
        dropped.extend(m.get("dropped", []))
    vsum = np.concatenate(v_parts, axis=0)
    cis = np.concatenate(ci_parts)
    assert len(set(cis.tolist())) == len(cis), "duplicate ci across capture shards"
    return vsum, cis, dropped


def phase_score(args) -> None:
    C.phase("score")
    t0 = time.time()
    rows, sha = _load_bundle(args)
    vsum, kres_ci, dropped = _assemble_draws(args, sha)
    row_of = {int(c): j for j, c in enumerate(kres_ci.tolist())}

    # originals: pass_b test answers + banked distractor vectors.
    N1G._load_pass_b_bundle(N1G.PASS_B_LOCAL)
    bundle = F.load_pass_b()
    n_ctx = int(bundle["cx_last"].shape[0])
    _tr, _val, test = F.fixed_split(n_ctx, n_ctx - 400 - 1000, 400, 1000, F.SPLIT_SEED)
    X = F.input_layer(bundle, "last", LAYER)
    Y = F.target_vx(bundle, LAYER)
    Xte = X[test].astype(np.float64)
    Yte32 = Y[test].astype(np.float32)
    dp = _staged(args, DISTR_REPO_PATH)
    dblob = np.load(dp)
    dvx = dblob["vx"][:N_DISTR_MAX].astype(np.float32)
    dci_all = np.asarray(dblob["ci"][:N_DISTR_MAX], dtype=np.int64)

    # arm predictions (banked payloads applied; ridge repro-asserted vs banked R²).
    banked_ml = json.loads(MB.BANKED_MULTILAYER.read_text())
    preds: dict[str, np.ndarray] = {}
    for name in ("ridge", "mlp_w8192"):
        rel = f"{MB.WEIGHTS_PREFIX}/L{LAYER}/{name}.pt"
        wpath = _staged(args, rel, revision=MB.KNOWN_GOOD_REVISION)
        MB._realized_keys_check(wpath, MB.FITTER_KIND[name])
        payload = torch.load(wpath, map_location="cpu", weights_only=False)
        preds[name] = N1M.apply_map(payload, X[test], torch.device("cpu"))
        applied_r2 = PR._pooled_r2(preds[name], Y[test].astype(np.float64))
        banked_r2 = float(
            banked_ml["per_layer"][str(LAYER)]["per_point"]["mixed_1m"]["predictors"][name][
                "whole_map_r2"
            ]
        )
        logger.info("[score] %s applied R2=%.6f banked=%.6f", name, applied_r2, banked_r2)
        if name == "ridge":
            MB._assert_reproduction(applied_r2, banked_r2)
        if name == "ridge":
            xmu = payload["xmu"].to(torch.float64).numpy()
            ymu = payload["ymu"].to(torch.float64).numpy()
            preds["identity_bias"] = Xte + (ymu - xmu)

    # coverage: kept slots per source (matched composition across single/avg).
    test_keep = [i for i, _p in enumerate(test) if -(1 + i) in row_of]
    distr_keep = [j for j in range(N_DISTR_MAX) if int(dci_all[j]) in row_of]
    logger.info(
        "[score] coverage: test %d/%d distr %d/%d (dropped %d)",
        len(test_keep),
        N_TEST,
        len(distr_keep),
        N_DISTR_MAX,
        len(dropped),
    )

    # averaged entries: mean(original + K fresh draws), fp64.
    def _avg(orig32: np.ndarray, ci_list: list[int]) -> np.ndarray:
        o = orig32.astype(np.float64)
        vs = np.stack([vsum[row_of[int(c)]] for c in ci_list])
        return (o + vs) / float(K_DRAWS + 1)

    test_cis = [-(1 + i) for i in test_keep]
    distr_cis = [int(dci_all[j]) for j in distr_keep]
    Yte_k = Yte32[test_keep]
    dvx_k = dvx[distr_keep]
    Yte_avg = _avg(Yte_k, test_cis)
    dvx_avg = _avg(dvx_k, distr_cis)

    wh, wh_meta = _whiten_fn(_staged(args, WHITEN_REPO_PATH))
    draws = MB.Draws.make(len(test_keep), BOOT_N, K_PERM, DRAWS_SEED)

    # ── repro gate: FULL banked composition, single-draw, ridge, raw metrics ──
    banked_arm = json.loads(BANKED_CONTEXT_ARM.read_text())["per_layer"][str(LAYER)]["arms"][
        "ridge"
    ]["retrieval"]
    draws_full = MB.Draws.make(N_TEST, BOOT_N, K_PERM, DRAWS_SEED)
    repro: dict[str, dict] = {}
    for pool_name, pool32, tix in (
        ("test", Yte32, np.arange(N_TEST)),
        ("distr_20000", np.concatenate([Yte32, dvx]), np.arange(N_TEST)),
    ):
        spec = MB.PoolSpec.make(pool_name, pool32, tix, np.array(["x"] * pool32.shape[0]))
        cell, _ = MB.eval_retrieval_cell(
            preds["ridge"],
            spec,
            MB.KS_CONTEXT,
            draws_full,
            helper_parity=(pool32.shape[0] <= 5000),
        )
        deltas = {}
        for metric in ("euclidean", "cosine", "csls"):
            b = banked_arm[pool_name][metric]
            for kk in ("1", "5", "10"):
                deltas[f"{metric}_acc{kk}"] = float(
                    cell[metric]["acc_at_k"][int(kk)] - b["acc_at_k"][kk]
                )
            deltas[f"{metric}_mrr"] = float(cell[metric]["mrr"] - b["mrr"])
        worst = max(abs(v) for v in deltas.values())
        assert worst <= REPRO_ATOL, (
            f"repro gate FAIL at pool {pool_name}: max |delta| {worst:.2e} > {REPRO_ATOL}"
        )
        repro[pool_name] = {"deltas_vs_banked": deltas, "max_abs_delta": worst}
        logger.info("[score] repro gate %s PASS (max |delta| %.2e)", pool_name, worst)

    # ── core battery: arm x entry x rung x {raw, whitened} ──
    q_keep = np.asarray(test_keep, dtype=np.int64)
    cells: dict[str, dict] = {}
    for rung_total in RUNG_TOTALS:
        n_d = rung_total - N_TEST
        assert 0 <= n_d <= N_DISTR_MAX, n_d
        n_d_k = min(n_d, len(distr_keep))
        for entry in ("single", "avg"):
            tgt = Yte_k.astype(np.float64) if entry == "single" else Yte_avg
            dis = dvx_k[:n_d_k].astype(np.float64) if entry == "single" else dvx_avg[:n_d_k]
            pool64 = np.concatenate([tgt, dis]) if n_d_k else tgt
            labels = np.array(
                ["lmsys(test)"] * tgt.shape[0]
                + ["lmsys(distr)"] * int(dis.shape[0] if n_d_k else 0),
                dtype=object,
            )
            rung_name = f"pool_{rung_total}"
            spec = MB.PoolSpec.make(f"{rung_name}|{entry}", pool64, np.arange(tgt.shape[0]), labels)
            spec_wh = MB.PoolSpec.make(
                f"{rung_name}|{entry}|wh", wh(pool64), np.arange(tgt.shape[0]), labels
            )
            for arm in ARMS:
                pred_k = preds[arm][q_keep]
                cell_raw, _ = MB.eval_retrieval_cell(
                    pred_k,
                    spec,
                    MB.KS_CONTEXT,
                    draws,
                    helper_parity=(pool64.shape[0] <= 5000),
                )
                cell_wh, _ = MB.eval_retrieval_cell(
                    wh(pred_k),
                    spec_wh,
                    MB.KS_CONTEXT,
                    draws,
                    helper_parity=(pool64.shape[0] <= 5000),
                )
                cells[f"{arm}|{entry}|{rung_name}"] = {
                    "label": ARM_LABELS[arm],
                    "n_query": int(pred_k.shape[0]),
                    "n_pool_realized": int(pool64.shape[0]),
                    "n_pool_nominal": int(rung_total),
                    "raw": cell_raw,
                    "whiten": cell_wh,
                }
                logger.info(
                    "[score] unit %s acc1(csls raw)=%.4f acc1(csls wh)=%.4f elapsed=%.0fs",
                    f"{arm}|{entry}|{rung_name}",
                    cell_raw["csls"]["acc_at_k"][1],
                    cell_wh["csls"]["acc_at_k"][1],
                    time.time() - t0,
                )

    banked_curve = {
        p: {m: banked_arm[p][m]["acc_at_k"] for m in ("euclidean", "cosine", "csls")}
        for p in ("test", "passb_5000", "distr_20000", "distr_100000")
    }
    prov = git_provenance()
    summary = {
        "round": "avgpool-scaleup (user-chat inline GPU override, task #1901)",
        "issue": ISSUE,
        "layer": LAYER,
        "conventions": {
            "pools": (
                "nested rungs: 1,000 test targets (pass_b test rows) + first (total-1000) rows "
                "of the banked issue1901_metrics distractor order (distr_20000 composition at "
                "the 20k rung); queries = the 1,000 test-row map predictions; true target = the "
                "query's own pool entry"
            ),
            "homogeneous_avg": (
                "EVERY pool entry (targets AND distractors) replaced by "
                "mean(original + 4 fresh on-policy draws, seeds 43-46); single = every entry "
                "the original vector; compositions matched row-for-row across single/avg"
            ),
            "rank": "mid-rank, 1e-9 relative tie tolerance (issue1901_metric_battery)",
            "acc_at_1": "(rank <= 1).mean(); a tie at the top counts as failure",
            "csls": (
                "Conneau two-sided CSLS k=10 on the cross-domain similarity "
                "(issue1901_metric_battery.csls_scores) — the battery formulation, NOT the "
                "#2202 one-sided variant"
            ),
            "whiten": (
                "z = L^-1(x - mu_A), #2202 shrunk train-answer Cholesky (lam={lam}, "
                "n_train={nt}); CROSS-LINE CAVEAT: stats fitted on #1738 multiturn train "
                "answers, applied here to the n1m single-turn line (both Qwen2.5-7B-Instruct "
                "L19 answer-family states)"
            ).format(lam=wh_meta["lam"], nt=wh_meta["n_train"]),
            "gen_recipe_inherited": json.loads(_bundle_paths(args)[1].read_text())["gen_recipe"],
        },
        "k_draws": K_DRAWS,
        "display_substitutions": sorted(
            (
                rec
                for sp in sorted(
                    (args.out_root / "gen").glob("*/shard*/scrub_disclosures_seed*.json")
                )
                for rec in json.loads(sp.read_text())
            ),
            key=lambda r: (r["file"], r["offset"]),
        ),
        "coverage": {
            "n_test_kept": len(test_keep),
            "n_distr_kept": len(distr_keep),
            "n_dropped_capture": len(dropped),
            "dropped": dropped[:50],
        },
        "repro_gate": {"tolerance": REPRO_ATOL, **repro},
        "banked_single_draw_curve_context_arm": banked_curve,
        "cells": cells,
        "wall_s": round(time.time() - t0, 1),
        **as_metadata_dict(prov),
    }
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(OUT_EVAL / "summary.json", summary)
    if not args.skip_upload:
        _upload_failloud(args, OUT_EVAL / "summary.json", "eval_mirror")
    logger.info("[score] wrote %s in %.1fs", OUT_EVAL / "summary.json", time.time() - t0)


# ── phase: figure (VM-side; reads the committed summary.json) ─────────────────────


def phase_figure(args) -> None:
    C.phase("figure")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as PP

    summary = json.loads((OUT_EVAL / "summary.json").read_text())
    cells = summary["cells"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PP.set_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = {"ridge": "#0173B2", "mlp_w8192": "#DE8F05", "identity_bias": "#029E73"}
    styles = {"avg": "-", "single": "--"}
    for arm in ARMS:
        for entry in ("avg", "single"):
            xs, ys = [], []
            for total in RUNG_TOTALS:
                c = cells[f"{arm}|{entry}|pool_{total}"]
                xs.append(c["n_pool_realized"])
                ys.append(c["whiten"]["csls"]["acc_at_k"]["1"])
            tgt = "averaged pool (K=5 mean)" if entry == "avg" else "single-draw pool"
            ax.plot(
                xs,
                ys,
                styles[entry],
                marker="o",
                ms=4,
                color=colors[arm],
                label=f"{ARM_LABELS[arm]}, {tgt}",
            )
    xs_ch = [cells[f"ridge|single|pool_{t}"]["n_pool_realized"] for t in RUNG_TOTALS]
    ax.plot(xs_ch, [1.0 / x for x in xs_ch], ":", color="0.5", label="chance (1/pool)")
    ax.set_xscale("log")
    ax.set_xlabel("pool size (log)")
    ax.set_ylabel("acc@1 (whitened cosine + CSLS)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"avgpool_acc1_vs_poolsize.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("[figure] wrote %s", FIG_DIR / "avgpool_acc1_vs_poolsize.png")


# ── smoke (CPU): round-new logic — averaging alignment, pools, battery reuse ──────


def _smoke(args) -> int:
    rng = np.random.default_rng(0)
    n_q, n_d, d = 30, 40, 16
    Y = rng.normal(size=(n_q, d)).astype(np.float32)
    D = rng.normal(size=(n_d, d)).astype(np.float32)
    pred = Y + 0.3 * rng.normal(size=(n_q, d))
    draws4 = Y[:, None, :] + 0.05 * rng.normal(size=(n_q, K_DRAWS, d)).astype(np.float32)
    avg = (Y.astype(np.float64) + draws4.astype(np.float64).sum(1)) / 5.0
    assert np.allclose(avg, (Y + draws4.sum(1)) / 5.0, atol=1e-6)
    draws = MB.Draws.make(n_q, 50, 20, 0)
    for entry, tgt, dis in (
        ("single", Y.astype(np.float64), D.astype(np.float64)),
        ("avg", avg, D.astype(np.float64)),
    ):
        pool = np.concatenate([tgt, dis])
        spec = MB.PoolSpec.make(
            f"smoke|{entry}", pool, np.arange(n_q), np.array(["x"] * (n_q + n_d))
        )
        cell, _ = MB.eval_retrieval_cell(pred, spec, (1, 5, 10), draws, helper_parity=True)
        for m in ("euclidean", "cosine", "csls"):
            a1 = cell[m]["acc_at_k"][1]
            assert 0.0 <= a1 <= 1.0
        assert cell["euclidean"]["n_pool"] == n_q + n_d
    # averaged targets closer to the noiseless mean => avg acc1 >= single acc1 here
    mu = Y.mean(axis=0)
    cov = np.cov(np.concatenate([Y, D]).T) + 0.1 * np.eye(d)
    ell = np.linalg.cholesky(cov)
    stats = args.out_root / "smoke_whiten.npz"
    args.out_root.mkdir(parents=True, exist_ok=True)
    np.savez(stats, mu_A=mu, L=ell, lam=0.1, n_train=n_q)
    whf, meta = _whiten_fn(stats)
    z = whf(Y)
    assert z.shape == Y.shape and np.isfinite(z).all() and meta["lam"] == 0.1
    # bundle-row id space: negative test ids never collide with manifest ids
    cis = [-(1 + i) for i in range(5)] + [10, 20, 30]
    assert len(set(cis)) == len(cis)
    logger.info("[smoke] OK")
    return 0


# ── main ──────────────────────────────────────────────────────────────────────────

PHASES = {
    "bundle": phase_bundle,
    "pilot": phase_pilot,
    "gen": phase_gen,
    "cap": phase_cap,
    "score": phase_score,
    "figure": phase_figure,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=list(PHASES), required=False, default=None)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_1901" / "avgpool_scaleup",
        help="pod: /workspace/outputs/issue1901_avgpool",
    )
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument(
        "--src",
        choices=["all", "test", "distr"],
        default="all",
        help="gen/cap row-source filter (test rows may be sibling-imported)",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--gen-batch", type=int, default=32, help="capture rows/batch")
    ap.add_argument("--token-budget", type=int, default=32_768, help="capture batch token cap")
    ap.add_argument("--tiny-model", action="store_true", help="capture smoke carve-out")
    ap.add_argument("--skip-upload", action="store_true")
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
        assert args.phase, "--phase required (no 'all' — the launcher sequences phases)"
        PHASES[args.phase](args)
        rc = 0
    C.phase("done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
