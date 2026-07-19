"""Issue #1482 — follow-up round `k-resample-noise-floor`: answer-entropy floor driver.

Phases (plan v7 §4 + the v68 convention-fix; the smoke IS this driver with tiny
args — PASS_UNIFIED):
  subsample  VM CPU: stratified 2*n-per-arm labeled-holdout subsample (RNG 148201,
             EN uniform + largest-remainder within non-EN), raw-chunk text fetch
             (8-thread pool over the rig's ``_download_chunk_with_retry``,
             delete-after, chunk-index prediction + full-scan fallback), bundle
             write + fail-loud HF ``inputs/`` upload.
  pod        1x GPU: sequential child processes (the rig's ``_run_children``
             pattern — vLLM torn down by process exit before the HF capture
             loads): b0 stage → b1 generate (seeds 43-46; per-row generation-time
             token ids + prompt ids persisted, chunk meta ``ids_version=1``;
             per-(seed,chunk) bundle-sha+ids-version-keyed checkpoints; per-seed
             fail-loud rollout upload BEFORE b2) → b2 capture of the 4 FRESH
             draws under the PARENT capture convention (verbatim
             ``COL.capture_answer_vector`` recipe: full-chat-template
             re-tokenization of prompt+response, span [prompt_len:full_len]
             incl. the end-of-turn tail — code-verified at
             issue779_ffc_n1m_generate_capture._capture_shard_trimmed; layer 19,
             V.npz + capture_meta upload BEFORE the sentinel) → results
             sentinel + [phase=done].
  vstream    VM CPU+network: seed-42's v is STREAMED verbatim from the parent
             #779 n1m capture chunks (NO recapture — the stored arrays' own
             bytes): scoped .pt listing, raw-JSON chunk-index prediction +
             neighbor-first fallback, delete-after, found-set checkpoint;
             v42_stream.npz + meta upload (analysis_tensors/).
  analyze    VM CPU: G1 = streamed-v42 IDENTITY check of the join (e2 from
             streamed v vs stored holdout_e2; tight fp16-pred tolerance) HALT
             gate + G2 parent-vs-fresh exchangeability diagnostic; FLOOR =
             the registered fresh-4 estimator (trvar over draws 43-46, ddof=1;
             m2 = ||vhat - mean(V_fresh)||^2 - trvar/4) with the 5-draw variant
             (streamed v42 as draw 0) reported as a shadow diagnostic; joint
             10k-draw bootstrap (chunked gathers), figures.

Pod-side contract: sentinels only (never task.py); ``[phase=...]`` lines with
the single terminal ``[phase=done]``. LMSYS/WildChat text is DIGEST-ONLY —
never printed or logged. Upload prefix is this round's OWN
``issue1482_kresample/`` (never the parent's buckets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM discipline)

import issue779_collect as COL  # noqa: E402  (chunked vLLM generate)
import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402  (_download_chunk_with_retry)
import issue1482_error_analysis as EA  # noqa: E402  (rig seams: tokenize/capture/sentinel)
import numpy as np  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1482_kresample")

TASK_ID = 1482
LAYER = 19
HIDDEN = 3584
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_PREFIX_DEFAULT = "issue1482_kresample"
EA_PREFIX = "issue1482_error_analysis"  # parent bucket (input fallbacks ONLY — never written)
FIT_SEED = 0  # inherited; no refits anywhere (plan §4)
SUBSAMPLE_SEED = 148201
BOOT_SEED = 1482
GEN_SEEDS = (43, 44, 45, 46)  # fresh per-request seeds; engine seed 42 (parent)
# V.npz draw order = GEN_SEEDS (fresh captures only). The parent draw (seed 42) is
# NEVER recaptured: its v is streamed verbatim from the #779 n1m capture chunks
# (phase vstream) and enters analysis as e2_k column 0 (the v68 convention fix).
IDS_VERSION = 1  # b1 chunk schema: generation-time token ids persisted per row
GEN_CHUNK = 500  # rows per B1 checkpoint chunk (plan §4 B1)
GEN_MAX_TOKENS = 1024
MAX_MODEL_LEN = 8192  # parent engine value (issue779_ffc_n1m_generate_capture.MAX_MODEL_LEN)
BUNDLE_PARTS = ("subsample_2k_part0.json", "subsample_2k_part1.json")
# Draft-time full-join reference (plan §11 n-rationale: join reproduced this exactly);
# a drifted labels/npz/row_ci input MUST fail loud here, not confound the contrast.
PARENT_DELTA = -0.017497679669843946
# Gate G1 (v68 redefinition): e2 from the STREAMED parent v (vstream) vs stored
# holdout_e2 — an exact-identity check of the join. The only expected deviation is
# the fp16 quantization of the persisted holdout_pred16 vs the full-precision pred
# the parent scored with, so the tolerance is TIGHT (production-calibrated;
# smoke-demoted per #1345).
G1_MED_REL_MAX = 1e-3
G1_SPEARMAN_MIN = 0.999
G2_RANK_TOL = 0.15  # plan §4 gate G2 (SE≈0.032 at n≈2000; smoke-demoted)
RC_G1 = 23  # designed-halt rc (gates.json written first; never a bare rc=1)
RC_C1 = 24
RC_PILOT = 7  # B1 pilot-gate refusal (report JSON written first — #1415 convention)
B1_BOOKED_WALL_H = 1.0  # plan §9 B1 row (booked wall; abort at > 2x)

PERCONTEXT_DIR = PROJECT_ROOT / "eval_results" / "issue_1482" / "percontext"
LABELS_PATH = PROJECT_ROOT / "eval_results" / "issue_1482" / "judge_labels" / "labels.json"
SCRATCH_PARENT = PROJECT_ROOT / "data" / "issue_1482" / "scratch"


# ── shared input loaders (local-first, HF fallback — plan §10 fitness table) ────


def _load_row_ci() -> np.ndarray:
    p = SCRATCH_PARENT / "row_ci.npy"
    if not p.exists():
        hub.stage_hub_file(
            C.HF_DATA_REPO, f"{EA_PREFIX}/analysis_tensors/scratch_meta/row_ci.npy", p
        )
    return np.load(p)


def _load_percontext(fit_id: str) -> np.lib.npyio.NpzFile:
    p = PERCONTEXT_DIR / f"{fit_id}.npz"
    if not p.exists():  # npz are NOT git-tracked — HF mirror is the pod-side source
        hub.stage_hub_file(
            C.HF_DATA_REPO, f"{EA_PREFIX}/analysis_tensors/percontext/{fit_id}.npz", p
        )
    return np.load(p)


def _load_labels() -> dict:
    return json.loads(LABELS_PATH.read_text())["labels"]


def _stored_join(args):
    """Stored ridge holdout arrays + language join (the parent analysis convention,
    issue1482_analysis.py:464). Returns dict; asserts the join reproduces the
    parent's full delta EXACTLY (drifted-input detector)."""
    z = _load_percontext(f"refit_holdout__ridge__seed{FIT_SEED}")
    rows = z["holdout_rows"]
    row_ci = _load_row_ci()
    labels = _load_labels()
    lang = np.array([labels.get(str(int(row_ci[r])), {}).get("language", "") for r in rows])
    en = lang == "en"
    ne = (lang != "") & (lang != "en")
    nerr = z["holdout_nerr"].astype(np.float64)
    delta = float(nerr[ne].mean() - nerr[en].mean())
    assert np.isclose(delta, PARENT_DELTA, rtol=0, atol=1e-9), (
        f"full-join delta {delta} != parent {PARENT_DELTA} — labels/npz/row_ci drift"
    )
    return {
        "rows": rows,
        "row_ci": row_ci,
        "lang": lang,
        "en": en,
        "ne": ne,
        "e2": z["holdout_e2"].astype(np.float64),
        "denom": z["holdout_denom"].astype(np.float64),
        "nerr": nerr,
        "pred16": z["holdout_pred16"],  # fp16; cast at use site
        "delta_full": delta,
    }


# ── phase A: subsample + input bundle ───────────────────────────────────────────


def largest_remainder_alloc(counts: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of n over language codes
    (plan §4 Phase A step 2). Deterministic: code-ascending processing, ties on
    the remainder break by code ascending. Returns only codes with >=1 row."""
    total = sum(counts.values())
    assert 0 < n <= total, (n, total)
    order = sorted(counts)
    quotas = {c: n * counts[c] / total for c in order}
    base = {c: math.floor(quotas[c]) for c in order}
    left = n - sum(base.values())
    for c in sorted(order, key=lambda c: (-(quotas[c] - base[c]), c))[:left]:
        base[c] += 1
    alloc = {c: v for c, v in base.items() if v > 0}
    assert sum(alloc.values()) == n
    for c, v in alloc.items():
        assert v <= counts[c], (c, v, counts[c])
    return alloc


def _probe_chunk_index(args, names: list[str], fetch_fn=None) -> dict[str, tuple[int, int]]:
    """Predict each chunk's ci range from per-shard base probes (min-ci of chunk 0
    and chunk 1 per shard -> base + stride) over the raw-JSON chunks. PURE
    OPTIMIZATION: every predicted fetch is verified by the caller and misses fall
    back to a full scan. ``fetch_fn`` (default: the rig downloader) is the
    transport boundary the vstream smoke fakes."""
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(n.split("_")[0], []).append(n)
    cache = args.scratch / "kresample_raw_cache"
    cache.mkdir(parents=True, exist_ok=True)
    index: dict[str, tuple[int, int]] = {}
    for _shard, shard_names in sorted(by_shard.items()):
        shard_names = sorted(shard_names)
        probes = shard_names[:2] if len(shard_names) > 1 else shard_names[:1]
        mins = []
        fetch = fetch_fn or N1M._download_chunk_with_retry
        for pn in probes:
            got = Path(fetch(C.HF_DATA_REPO, f"{EA.RAW_PREFIX}/{pn}", cache))
            rows = json.loads(got.read_text())["rows"]
            got.unlink()
            mins.append(min(int(r["ci"]) for r in rows))
        base = mins[0]
        stride = (mins[1] - mins[0]) if len(mins) > 1 else GEN_CHUNK
        if stride <= 0:
            return {}  # structure assumption broken -> caller full-scans
        for n in shard_names:
            j = int(n.rsplit("chunk", 1)[-1].split(".")[0])  # explicit chunk number
            index[n] = (base + j * stride, base + (j + 1) * stride)
    return index


def _fetch_rows_threaded(args, needed_ci: set[int]) -> dict[int, tuple[str, str]]:
    """Fetch (prompt, response) text for the needed ci from the parent raw chunks:
    predicted chunks first (index probe), full-scan fallback for any residue.
    8-thread pool over the rig's ``_download_chunk_with_retry``; delete-after;
    DIGEST-ONLY (text never logged)."""
    names = EA._raw_chunk_names(args)
    cache = args.scratch / "kresample_raw_cache"
    cache.mkdir(parents=True, exist_ok=True)
    found: dict[int, tuple[str, str]] = {}
    remaining = set(int(c) for c in needed_ci)
    lock = threading.Lock()
    stop = threading.Event()

    def _one(name: str) -> None:
        if stop.is_set():
            return
        got = Path(N1M._download_chunk_with_retry(C.HF_DATA_REPO, f"{EA.RAW_PREFIX}/{name}", cache))
        rows = json.loads(got.read_text())["rows"]
        got.unlink()
        hits = [
            (int(r["ci"]), r["prompt"], r["response"]) for r in rows if int(r["ci"]) in needed_ci
        ]
        if hits:
            with lock:
                for ci, p, resp in hits:
                    found[ci] = (p, resp)
                    remaining.discard(ci)
                if not remaining:
                    stop.set()

    def _run_pool(subset: list[str]) -> None:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(_one, n) for n in subset]):
                f.result()  # re-raise any download/parse failure loud

    index = _probe_chunk_index(args, names)
    predicted = (
        sorted(
            {n for n in names if any(lo <= ci < hi for ci in remaining for lo, hi in [index[n]])}
        )
        if index
        else []
    )
    if predicted:
        logger.info(
            "[fetch] predicted %d/%d chunks for %d rows", len(predicted), len(names), len(remaining)
        )
        _run_pool(predicted)
    if remaining:
        rest = [n for n in names if n not in set(predicted)]
        logger.info(
            "[fetch] fallback full scan over %d chunks (%d rows missing)", len(rest), len(remaining)
        )
        _run_pool(rest)
    assert not remaining, f"{len(remaining)} needed ci not found in raw chunks"
    return found


def _bundle_paths(args) -> list[Path]:
    return [args.out / "inputs" / n for n in BUNDLE_PARTS]


def _bundle_sha(args) -> str:
    h = hashlib.sha256()
    for p in _bundle_paths(args):
        h.update(p.read_bytes())
    return h.hexdigest()


def _load_bundle(args) -> tuple[list[dict], str]:
    parts = _bundle_paths(args)
    for p in parts:
        assert p.exists(), f"bundle part missing: {p} (run --phase subsample / b0 first)"
    rows: list[dict] = []
    for p in parts:
        rows.extend(json.loads(p.read_text())["rows"])
    return rows, _bundle_sha(args)


def phase_subsample(args) -> None:
    C.phase("subsample")
    s = _stored_join(args)
    rows, lang = s["rows"], s["lang"]
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    en_pool = np.flatnonzero(s["en"])
    en_pick = np.sort(rng.choice(en_pool, size=args.n_per_arm, replace=False))
    ne_codes, ne_counts = np.unique(lang[s["ne"]], return_counts=True)
    counts = {str(c): int(k) for c, k in zip(ne_codes, ne_counts, strict=True)}
    alloc = largest_remainder_alloc(counts, args.n_per_arm)
    ne_parts = []
    for code in sorted(alloc):  # fixed rng call order (determinism)
        pool = np.flatnonzero(lang == code)
        ne_parts.append(np.sort(rng.choice(pool, size=alloc[code], replace=False)))
    ne_pick = np.sort(np.concatenate(ne_parts))
    logger.info(
        "[subsample] n_en=%d n_non_en=%d (labeled %d/%d); alloc over %d codes",
        len(en_pick),
        len(ne_pick),
        int((lang != "").sum()),
        len(rows),
        len(alloc),
    )

    needed = {int(s["row_ci"][rows[i]]) for i in np.concatenate([en_pick, ne_pick])}
    texts = _fetch_rows_threaded(args, needed)
    n_expect = 2 * args.n_per_arm
    assert len(texts) == n_expect, f"fetched {len(texts)} != {n_expect}"  # plan §4 hard assert

    def _rows_for(idx: np.ndarray, arm: str) -> list[dict]:
        out = []
        for i in idx:
            r = int(rows[i])
            ci = int(s["row_ci"][r])
            prompt, resp = texts[ci]
            out.append(
                {
                    "row_idx": r,
                    "ci": ci,
                    "arm": arm,
                    "language": str(lang[i]),
                    "prompt": prompt,
                    "response_seed42": resp,
                    "e2_stored": float(s["e2"][i]),
                    "denom_stored": float(s["denom"][i]),
                    "nerr_stored": float(s["nerr"][i]),
                }
            )
        return out

    (args.out / "inputs").mkdir(parents=True, exist_ok=True)
    for path, (idx, arm) in zip(
        _bundle_paths(args), [(en_pick, "en"), (ne_pick, "nonen")], strict=True
    ):
        C.write_json_atomic(
            path,
            {
                "meta": {
                    "subsample_seed": SUBSAMPLE_SEED,
                    "n": len(idx),
                    "arm": arm,
                    **C.reproducibility_metadata(),
                },
                "rows": _rows_for(idx, arm),
            },
        )
    sha = _bundle_sha(args)
    args.out_eval.mkdir(parents=True, exist_ok=True)
    strata = {
        "n_per_arm": args.n_per_arm,
        "subsample_seed": SUBSAMPLE_SEED,
        "allocation": alloc,
        "nonen_counts": counts,
        "bundle_sha": sha,
        "row_idx": {
            "en": [int(rows[i]) for i in en_pick],
            "nonen": [int(rows[i]) for i in ne_pick],
        },
        "full_join": {
            "n_en": int(s["en"].sum()),
            "n_non_en": int(s["ne"].sum()),
            "delta": s["delta_full"],
        },
        "metadata": C.reproducibility_metadata(),
    }
    C.write_json_atomic(args.out_eval / "strata.json", strata)
    if not args.skip_upload:
        _upload_files_failloud(
            args, [*_bundle_paths(args), args.out_eval / "strata.json"], "inputs"
        )
    logger.info("[subsample] bundle sha %s; strata -> %s", sha[:12], args.out_eval / "strata.json")
    EA._phase_sentinel("kres-subsample", f"subsample bundle written (sha {sha[:12]})")


# ── fail-loud upload helpers (upload prefix = this round's OWN bucket) ──────────


def _upload_files_failloud(args, paths: list[Path], sub: str) -> None:
    """Per-file non-LFS/LFS upload with a bounded OUTER retry on the fail-soft
    ``hub._upload`` empty return (#1315 seam pattern), then exact-set verify."""
    from huggingface_hub import HfApi

    prefix = f"{args.hf_prefix}/{sub}"
    for p in paths:
        url = ""
        for attempt, pause in enumerate((0, 30, 60, 120)):
            if pause:
                time.sleep(pause * (1.0 + 0.25 * np.random.default_rng(attempt).random()))
                logger.warning("[upload] retry %d for %s", attempt, p.name)
            url = hub._upload(
                p,
                C.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{prefix}/{p.name}",  # single-file: FULL remote path incl. name
                upload_as_file=True,
            )
            if url:
                break
        if not url:
            raise RuntimeError(
                f"upload returned no path for {p} -> {prefix} (outer retry exhausted)"
            )
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        C.HF_DATA_REPO,
        [f"{prefix}/{p.name}" for p in paths],
        path_in_repo=prefix,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"post-upload verify: missing on Hub: {missing}")
    logger.info("[upload] %d file(s) verified under %s/%s", len(paths), C.HF_DATA_REPO, prefix)


# ── pod phase: b0 stage → b1 generate → b2 capture → sentinel ──────────────────


def _child_flags(args) -> list[str]:
    flags = [
        "--device",
        args.device,
        "--out",
        str(args.out),
        "--out-eval",
        str(args.out_eval),
        "--figures",
        str(args.figures),
        "--scratch",
        str(args.scratch),
        "--hf-prefix",
        args.hf_prefix,
        "--n-per-arm",
        str(args.n_per_arm),
        "--n-boot",
        str(args.n_boot),
        "--gen-batch",
        str(args.gen_batch),
        "--token-budget",
        str(args.token_budget),
        "--workers",
        str(args.workers),
        "--max-chunks",
        str(args.max_chunks),
    ]
    for flag in ("smoke", "tiny_model", "skip_upload"):
        if getattr(args, flag):
            flags.append("--" + flag.replace("_", "-"))
    return flags


def _run_child(tag: str, args) -> None:
    """Sequential child process per pod sub-phase (the rig's ``_run_children``
    pattern: explicit env passthrough, per-child log, tail echo + raise on
    failure — #1333 inner-log-tail duty). Single GPU by design (§9): no CVD
    fan-out; vLLM teardown = child process exit before the b2 HF load."""
    log_dir = args.out / "child_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{tag}.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", tag, *_child_flags(args)]
    logger.info("[pod] launch %s -> %s", tag, log.name)
    with open(log, "w") as log_f:
        proc = subprocess.Popen(cmd, env={**os.environ}, stdout=log_f, stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc == RC_PILOT:
        # designed halt, not a crash: route the child's distinct rc through the
        # dispatcher unchanged (pilot_gate_report.json is already on disk)
        logger.error(
            "[pod] child %s DESIGNED HALT rc=%d (pilot-gate refusal; gen/pilot_gate_report.json)",
            tag,
            rc,
        )
        raise SystemExit(RC_PILOT)
    if rc != 0:
        logger.error("[pod] child %s FAILED rc=%d; log tail:\n%s", tag, rc, EA._log_tail(log))
        raise RuntimeError(f"pod child {tag} failed rc={rc} (log: {log})")
    logger.info("[pod] child %s done", tag)


def phase_b0(args) -> None:
    C.phase("b0")
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(args.out, 2 if args.smoke else 25, phase="kresample-b0")
    for name in BUNDLE_PARTS:
        p = args.out / "inputs" / name
        if not p.exists():
            hub.stage_hub_file(C.HF_DATA_REPO, f"{args.hf_prefix}/inputs/{name}", p)
    _load_bundle(args)  # asserts presence + computes sha (fail-loud parse)
    for fit in ("ridge", "mlp_w8192"):  # plan §4 B0 staging list (consumed by VM phase C)
        _load_percontext(f"refit_holdout__{fit}__seed{FIT_SEED}")
    EA._phase_sentinel("kres-b0", "b0 staging complete")


def _vllm_generate_chunked_ids(llm, prompt_texts: list[str], sampling_params) -> list[dict]:
    """COL._vllm_generate_chunked with GENERATION-TIME TOKEN IDS kept (v68 fix):
    per prompt returns {text, token_ids, prompt_token_ids}. Same chunking +
    ``use_tqdm=False`` + per-chunk INFO liveness log as the parent helper."""
    out: list[dict] = []
    n_chunks = (len(prompt_texts) + COL.VLLM_CHUNK_SIZE - 1) // COL.VLLM_CHUNK_SIZE
    for i in range(0, len(prompt_texts), COL.VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + COL.VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] kresample-generate chunk %d/%d (%d prompts)",
            i // COL.VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
        )
        for o in llm.generate(chunk, sampling_params, use_tqdm=False):
            out.append(
                {
                    "text": o.outputs[0].text,
                    "token_ids": [int(t) for t in o.outputs[0].token_ids],
                    "prompt_token_ids": [int(t) for t in o.prompt_token_ids],
                }
            )
    return out


def _generate_seed(llm, tok, prompt_texts: list[str], seed: int) -> list[dict]:
    """One rollout per prompt with the parent pass-B recipe at per-request
    ``seed`` (issue779_ffc_n10k_generate_capture.py:188; engine seed 42).
    Returns per-prompt {text, token_ids, prompt_token_ids} — generation-time ids
    persist to the b1 chunks (ids_version 1). CPU smoke (llm None) returns
    per-seed stubs through the SAME checkpoint path; stub ids are the REAL
    tokenizer's ids for the stub text (generation-time == re-tokenized by
    construction on the stub path)."""
    if llm is None:
        out = []
        for t in prompt_texts:
            resp = f"This is a short stub response for the CPU capture smoke (draw seed {seed})."
            out.append(
                {
                    "text": resp,
                    "token_ids": [int(x) for x in tok(resp, add_special_tokens=False)["input_ids"]]
                    if tok is not None
                    else [],
                    "prompt_token_ids": [int(x) for x in tok(t)["input_ids"]]
                    if tok is not None
                    else [],
                }
            )
        return out
    from vllm import SamplingParams

    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=GEN_MAX_TOKENS, seed=seed)
    return _vllm_generate_chunked_ids(llm, prompt_texts, sp)


def _pilot_gate(chunk_wall_s: float, output_tokens: int, n_chunk_calls: int) -> dict:
    """B1 in-run pilot timing gate (plan §9 B1 row + kill criteria): the FIRST
    generated chunk is the pilot, measured at the sweep's OWN execution shape
    (one batched vLLM generate over a full chunk — gotchas #1415: never batch-1).
    Refusal = projected wall > 2x the booked row; caller writes the report JSON
    and exits the DISTINCT rc (never a bare rc=1)."""
    projected_h = chunk_wall_s * n_chunk_calls / 3600.0
    return {
        "chunk_wall_s": float(chunk_wall_s),
        "output_tokens": int(output_tokens),
        "output_tok_per_s": float(output_tokens / max(chunk_wall_s, 1e-9)),
        "n_chunk_calls": int(n_chunk_calls),
        "projected_wall_h": float(projected_h),
        "booked_wall_h": B1_BOOKED_WALL_H,
        "threshold_x": 2.0,
        "pass": bool(projected_h <= 2.0 * B1_BOOKED_WALL_H),
        "measured_shape": "one batched vLLM generate over a full production chunk",
    }


def _determinism_spot_check(llm, prompt_texts5: list[str], seed: int, ref5: list[str]) -> dict:
    """Assumption-10 verification (plan §12), rewritten after the v7 crash: the old
    check demanded byte-identity between the first-chunk BATCH generation and a
    standalone 5-prompt re-run, but vLLM 0.11 (V1) batch-composition/scheduling
    numerics can legitimately shift temp-1.0 sampled tokens across different batch
    shapes EVEN WITH a correctly applied per-request ``SamplingParams.seed`` (the
    seeded RNG fixes the sampling draws, not the logits), so 0/N there does NOT
    prove the seed is unapplied. Discriminate seed-function from batch numerics
    with two sub-checks on IDENTICAL standalone batch shapes:
      (1) same-call repeatability — the same prompts generated twice at ``seed``
          must match on >= max(1, N-1) of N (tolerates one rare
          nondeterministic-kernel flake; per-prompt outcomes recorded);
      (2) seed-distinctness — the same prompts once at ``seed + 1000`` (collides
          with no registered seed: engine 42, GEN_SEEDS 43-46) must DIFFER from
          run 1 on >= 1/N (N identical outputs across different seeds at temp 1.0
          over <= 1024 tokens => the per-request seed is ignored).
    Failure of either sub-check raises RuntimeError (fail-loud, G-gate class).
    The batch-vs-standalone byte-match rate vs ``ref5`` is recorded/logged as an
    informational diagnostic ONLY — it never gates. Runs ONCE per B1 run (the
    caller's ``det_report is None`` gate): the seed-application mechanism is
    engine-global, not per-seed. The CPU-smoke stub path (llm None) traverses the
    SAME decision logic — ``_generate_seed`` stubs are seed-deterministic."""
    n = len(prompt_texts5)
    rep1 = [r["text"] for r in _generate_seed(llm, None, prompt_texts5, seed)]
    rep2 = [r["text"] for r in _generate_seed(llm, None, prompt_texts5, seed)]
    distinct = [r["text"] for r in _generate_seed(llm, None, prompt_texts5, seed + 1000)]
    repeat_matches = [a == b for a, b in zip(rep1, rep2, strict=True)]
    differs = [a != b for a, b in zip(rep1, distinct, strict=True)]
    n_repeat, n_differ = int(sum(repeat_matches)), int(sum(differs))
    n_batch = int(sum(a == b for a, b in zip(rep1, ref5, strict=True)))
    repeat_floor = max(1, n - 1)
    report = {
        "n_total": n,
        "n_repeat_match": n_repeat,
        "repeat_floor": repeat_floor,
        "n_distinct_differ": n_differ,
        "per_prompt_repeat": repeat_matches,
        "per_prompt_distinct_differ": differs,
        "batch_vs_standalone_match": n_batch,  # informational only — never gates
        "seed": seed,
        "distinct_seed": seed + 1000,
        "engine": "cpu-stub" if llm is None else "vllm",
    }
    if n_repeat < repeat_floor:
        raise RuntimeError(
            f"determinism spot-check: same-seed repeat {n_repeat}/{n} < floor {repeat_floor}"
            f" — per-request seed not applied (or engine nondeterministic at a FIXED batch"
            f" shape); per-prompt: {repeat_matches}"
        )
    if n_differ == 0:
        raise RuntimeError(
            f"determinism spot-check: 0/{n} outputs differ at seed {seed + 1000} vs {seed}"
            f" — per-request seed ignored"
        )
    if n_repeat < n:
        logger.warning(
            "[b1] determinism repeat %d/%d (one nondeterministic-kernel flake tolerated)",
            n_repeat,
            n,
        )
    logger.info(
        "[b1] determinism: repeat=%d/%d distinct=%d/%d batch_match=%d/%d (informational)",
        n_repeat,
        n,
        n_differ,
        n,
        n_batch,
        n,
    )
    return report


def phase_b1(args) -> None:
    C.phase("b1")
    bundle_rows, sha = _load_bundle(args)
    from transformers import AutoTokenizer

    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained(MODEL_ID), what=f"tokenizer fetch ({MODEL_ID})"
    )
    prompt_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in bundle_rows
    ]
    budget = MAX_MODEL_LEN - GEN_MAX_TOKENS
    over = [
        i
        for i, t in enumerate(prompt_texts)
        if len(tok(t, add_special_tokens=False)["input_ids"]) > budget
    ]
    assert not over, (
        f"{len(over)} rendered prompts exceed {budget} tokens — parent over-length "
        f"screens should have removed these (row idx {over[:5]}...)"
    )
    llm = None
    if args.device == "cuda":
        from explore_persona_space.eval.generation import create_vllm_engine

        llm = create_vllm_engine(MODEL_ID, max_model_len=MAX_MODEL_LEN, seed=42)
    gen_dir = args.out / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(len(bundle_rows) / GEN_CHUNK)
    det_report: dict | None = None
    pilot_report: dict | None = None
    for k in GEN_SEEDS:
        seed_paths = []
        for j in range(n_chunks):
            ck = gen_dir / f"gen_seed{k}_chunk{j}.json"
            seed_paths.append(ck)
            if ck.exists():
                meta = json.loads(ck.read_text()).get("meta", {})
                # resume ONLY on sha match AND id-bearing schema (plan c24 + v68 fix):
                # a text-only chunk (no ids_version) is STALE — b2 needs the ids.
                if meta.get("bundle_sha") == sha and meta.get("ids_version") == IDS_VERSION:
                    logger.info("[b1] %s complete (sha + ids_version match) — skip", ck.name)
                    continue
                logger.warning(
                    "[b1] %s bundle-sha MISMATCH or text-only (ids_version %s != %d) — "
                    "regenerating (seeds pinned; r2 repeatability-verified)",
                    ck.name,
                    meta.get("ids_version"),
                    IDS_VERSION,
                )
            lo, hi = j * GEN_CHUNK, min((j + 1) * GEN_CHUNK, len(bundle_rows))
            t_c0 = time.time()
            gen_rows = _generate_seed(llm, tok, prompt_texts[lo:hi], seed=k)
            if pilot_report is None and llm is not None:
                out_toks = sum(len(g["token_ids"]) for g in gen_rows)
                pilot_report = _pilot_gate(time.time() - t_c0, out_toks, len(GEN_SEEDS) * n_chunks)
                C.write_json_atomic(gen_dir / "pilot_gate_report.json", pilot_report)
                if not pilot_report["pass"]:
                    logger.error(
                        "[b1] pilot gate REFUSAL: projected %.2f h > 2x booked %.1f h (report: %s)",
                        pilot_report["projected_wall_h"],
                        B1_BOOKED_WALL_H,
                        gen_dir / "pilot_gate_report.json",
                    )
                    raise SystemExit(RC_PILOT)
                logger.info(
                    "[b1] pilot gate PASS: %.1f output tok/s, projected %.2f h (booked %.1f h)",
                    pilot_report["output_tok_per_s"],
                    pilot_report["projected_wall_h"],
                    B1_BOOKED_WALL_H,
                )
            if det_report is None:
                det_report = _determinism_spot_check(
                    llm, prompt_texts[lo : lo + 5], k, [g["text"] for g in gen_rows[:5]]
                )
            C.write_json_atomic(
                ck,
                {
                    "meta": {
                        "bundle_sha": sha,
                        "ids_version": IDS_VERSION,
                        "seed": k,
                        "chunk": j,
                        "n": hi - lo,
                        "sampling": {
                            "n": 1,
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "max_tokens": GEN_MAX_TOKENS,
                            "seed": k,
                            "engine_seed": 42,
                            "max_model_len": MAX_MODEL_LEN,
                            "model": MODEL_ID,
                        },
                        **C.reproducibility_metadata(),
                    },
                    "rows": [
                        {
                            "row_idx": r["row_idx"],
                            "ci": r["ci"],
                            "response": g["text"],
                            # generation-time ids (v68 fix): the sequence vLLM actually
                            # emitted + the prompt ids it consumed — durable provenance
                            # for the id-vs-retok convention diagnostics in b2.
                            "token_ids": g["token_ids"],
                            "prompt_token_ids": g["prompt_token_ids"],
                        }
                        for r, g in zip(bundle_rows[lo:hi], gen_rows, strict=True)
                    ],
                },
            )
            logger.info("[b1] seed %d chunk %d/%d written (%d rows)", k, j + 1, n_chunks, hi - lo)
        # rollout text uploads at the END of each seed, fail-loud, BEFORE b2 (plan §4)
        if not args.skip_upload:
            _upload_files_failloud(args, seed_paths, "raw_completions")
    C.write_json_atomic(
        gen_dir / "b1_meta.json",
        {
            "bundle_sha": sha,
            "ids_version": IDS_VERSION,
            "n_chunks": n_chunks,
            "seeds": list(GEN_SEEDS),
            "determinism_check": det_report,
            "pilot_gate": pilot_report or {"skipped": "no chunk generated this run / cpu smoke"},
            **C.reproducibility_metadata(),
        },
    )
    EA._phase_sentinel("kres-b1", f"b1 generation complete ({len(GEN_SEEDS)}x{n_chunks} chunks)")


def _load_gen_chunks(args, sha: str) -> dict[int, dict[int, dict]]:
    """Per-seed per-ci gen rows ({response, token_ids, prompt_token_ids}). Fails
    loud on a text-only chunk (ids_version missing/old) — b2's id-convention
    diagnostics need generation-time ids, so stale pre-v68 chunks never feed it."""
    gen_dir = args.out / "gen"
    out: dict[int, dict[int, dict]] = {k: {} for k in GEN_SEEDS}
    for k in GEN_SEEDS:
        for ck in sorted(gen_dir.glob(f"gen_seed{k}_chunk*.json")):
            doc = json.loads(ck.read_text())
            assert doc["meta"].get("bundle_sha") == sha, f"{ck.name}: bundle sha mismatch"
            if doc["meta"].get("ids_version") != IDS_VERSION:
                raise RuntimeError(
                    f"{ck.name}: ids_version {doc['meta'].get('ids_version')} != {IDS_VERSION} "
                    "— stale text-only b1 chunk (pre-v68 convention); regenerate b1"
                )
            for r in doc["rows"]:
                out[k][int(r["ci"])] = {
                    "response": r["response"],
                    "token_ids": r["token_ids"],
                    "prompt_token_ids": r["prompt_token_ids"],
                }
    return out


def _token_batches(items: list, max_rows: int, token_budget: int):
    """Greedy pack (slot, draw, tokrow) items into batches bounded by rows AND
    total tokens (batch-size tuning — an allowed §-deviation; protects the
    A100-40 spot rung from long-prompt batch-32 OOM)."""
    batch, toks = [], 0
    for it in items:
        n = len(it[2][0])  # full_ids
        if batch and (len(batch) >= max_rows or toks + n > token_budget):
            yield batch
            batch, toks = [], 0
        batch.append(it)
        toks += n
    if batch:
        yield batch


def _prompt_render(tok, prompt: str) -> tuple[list[dict], list[int]]:
    """Generation-prompt render + its tokenization (the parent capture's
    ``prompt_len`` source — COL.capture_answer_vector:171-174), with the #594
    assistant-header position assert."""
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = [int(x) for x in tok(text)["input_ids"]]
    suffix = tok.decode(prompt_ids[-3:])
    assert suffix == C.GENERATION_SUFFIX, f"position assert: {suffix!r} != {C.GENERATION_SUFFIX!r}"
    return msgs, prompt_ids


def _parent_convention_full_ids(tok, msgs: list[dict], response: str) -> list[int]:
    """VERBATIM parent capture convention (v68 fix): ONE full-chat-template
    tokenization with the assistant turn embedded (``add_generation_prompt=False``)
    — the recipe that produced the stored v (COL.capture_answer_vector:175-179 via
    N1G._capture_shard_trimmed). The consumer slices the response span at
    ``[prompt_len:full_len]``, which INCLUDES the end-of-turn tail, and — like the
    parent — deliberately does NOT assert prompt-prefix identity (a BPE seam merge
    at the assistant\\n+response boundary is part of the stored convention; the
    caller records it as a diagnostic)."""
    full_text = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return [int(x) for x in tok(full_text)["input_ids"]]


def _check_prompt_ids_join(ci: int, seed: int, persisted: list[int], prompt_ids: list[int]) -> None:
    """B1<->B2 join check (fail loud): the template prompt tokenization must
    equal the ids vLLM consumed at generation time (same text, same tokenizer)."""
    if persisted != prompt_ids:
        raise RuntimeError(
            f"[b2] ci {ci} seed {seed}: template prompt ids ({len(prompt_ids)} tok) != "
            f"persisted vLLM prompt ids ({len(persisted)} tok) — b1/b2 join drift"
        )


def phase_b2(args) -> None:
    C.phase("b2")
    bundle_rows, sha = _load_bundle(args)
    gen = _load_gen_chunks(args, sha)
    for k in GEN_SEEDS:
        missing = [r["ci"] for r in bundle_rows if int(r["ci"]) not in gen[k]]
        assert not missing, f"seed {k}: {len(missing)} bundle rows missing from gen chunks"
    v_path = args.out / "V.npz"
    meta_path = args.out / "capture_meta.json"
    if v_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("bundle_sha") == sha and meta.get("ids_version") == IDS_VERSION:
            logger.info("[b2] V.npz complete (sha + ids_version match) — skip capture")
        else:
            raise RuntimeError(
                "[b2] existing V.npz capture_meta bundle-sha mismatch or pre-v68 "
                "text-retok capture (ids_version missing) — remove stale outputs"
            )
    else:
        import torch

        model, tok = EA._load_model_tok(args)
        kept: list[dict] = []
        dropped: list[dict] = []
        items: list[tuple[int, int, tuple]] = []  # (kept_slot, draw_slot, row tuple)
        seam_rows: list[list[int]] = []  # full_ids[:plen] != prompt_ids (parent seam merge)
        id_retok_rows: list[list[int]] = []  # gen token_ids != retok(response) (#1092 rate)
        for r in bundle_rows:
            ci = int(r["ci"])
            msgs, prompt_ids = _prompt_render(tok, r["prompt"])
            plen = len(prompt_ids)
            per_draw: list[tuple] | None = []
            seam_d, retok_d = [], []
            for seed in GEN_SEEDS:
                g = gen[seed][ci]
                _check_prompt_ids_join(ci, seed, g["prompt_token_ids"], prompt_ids)
                full_ids = _parent_convention_full_ids(tok, msgs, g["response"])
                if len(full_ids) <= plen:  # the parent's empty-span drop (av None)
                    per_draw = None
                    break
                seam_d.append(int(full_ids[:plen] != prompt_ids))
                retok = tok(g["response"], add_special_tokens=False)["input_ids"]
                retok_d.append(int([int(x) for x in retok] != g["token_ids"]))
                per_draw.append((full_ids, plen - 1, plen - 1, len(full_ids) - plen, seam_d[-1]))
            if per_draw is None:
                dropped.append({"row_idx": r["row_idx"], "ci": r["ci"], "arm": r["arm"]})
                continue
            slot = len(kept)
            kept.append(r)
            seam_rows.append(seam_d)
            id_retok_rows.append(retok_d)
            items.extend((slot, d, tk) for d, tk in enumerate(per_draw))
        n = len(kept)
        logger.info(
            "[b2] assembly=parent-convention(full-template-retok, span=[prompt_len:full_len] "
            "incl. end-of-turn tail) draws=%s; seed42=streamed (no recapture); "
            "capturing %d contexts x %d draws (%d dropped)",
            list(GEN_SEEDS),
            n,
            len(GEN_SEEDS),
            len(dropped),
        )
        V = np.zeros((n, len(GEN_SEEDS), HIDDEN), dtype=np.float32)
        n_ans = np.zeros((n, len(GEN_SEEDS)), dtype=np.int32)
        t0, done = time.time(), 0
        with torch.no_grad():
            for batch in _token_batches(items, args.gen_batch, args.token_budget):
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
                    rate = done / max(time.time() - t0, 1e-9)
                    logger.info("[b2] %d/%d rows captured (%.1f rows/s)", done, len(items), rate)
        seam_arr = np.array(seam_rows, dtype=np.int8).reshape(n, len(GEN_SEEDS))
        retok_arr = np.array(id_retok_rows, dtype=np.int8).reshape(n, len(GEN_SEEDS))
        np.savez(  # plain savez — never savez_compressed in the hot path (#813)
            v_path,
            V=V,
            n_ans=n_ans,
            seam_merge=seam_arr,
            gen_id_retok_mismatch=retok_arr,
            rows=np.array([r["row_idx"] for r in kept], dtype=np.int64),
            ci=np.array([r["ci"] for r in kept], dtype=np.int64),
            arm=np.array([r["arm"] for r in kept]),
            language=np.array([r["language"] for r in kept]),
            draws=np.array(GEN_SEEDS, dtype=np.int64),
        )
        C.write_json_atomic(
            meta_path,
            {
                "bundle_sha": sha,
                "ids_version": IDS_VERSION,
                "capture_convention": "parent-full-template-retok-span-incl-eot-tail",
                "draws": list(GEN_SEEDS),
                "seed42_source": "streamed-parent-capture (phase vstream)",
                "layer": LAYER,
                "n_kept": n,
                "dropped": dropped,
                "id_convention_diagnostic": {
                    "seam_merge_rate": float(seam_arr.mean()) if n else 0.0,
                    "gen_id_retok_mismatch_rate": float(retok_arr.mean()) if n else 0.0,
                },
                "tiny_model": bool(args.tiny_model),
                "model": MODEL_ID,
                **C.reproducibility_metadata(),
            },
        )
    if not args.skip_upload:  # V.npz + capture_meta BEFORE the sentinel (plan §4)
        _upload_files_failloud(args, [v_path, meta_path], "analysis_tensors")
    EA._phase_sentinel("kres-b2", "b2 capture complete (V.npz written)")


def _results_sentinel(args, t_start: float) -> None:
    """End-of-pod results sentinel (pod-side-reporting contract; kind
    epm:smoke-result under --smoke so the drain never mistakes it for real
    results). No training -> no adapter/wandb card rows."""
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.out / "capture_meta.json").read_text())
    b1_meta = json.loads((args.out / "gen" / "b1_meta.json").read_text())
    kind = "epm:smoke-result" if args.smoke else "epm:results"
    payload = {
        "sentinel_schema_version": C.SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "task_id": TASK_ID,
        "by": "issue1482_kresample",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "issue-1482 kresample pod phases b0-b2 complete (phase C analyze runs off-pod)",
        "smoke": bool(args.smoke),
        "eval_numbers": {
            "n_kept_contexts": meta["n_kept"],
            "n_dropped_contexts": len(meta["dropped"]),
            "determinism_check": b1_meta.get("determinism_check"),
        },
        "eval_paths": {
            "V_npz": f"{args.hf_prefix}/analysis_tensors/V.npz",
            "capture_meta": f"{args.hf_prefix}/analysis_tensors/capture_meta.json",
            "raw_completions": f"{args.hf_prefix}/raw_completions/",
        },
        "reproducibility_card": {
            **C.reproducibility_metadata(),
            "layer": LAYER,
            "generation_seeds": list(GEN_SEEDS),
            "draw_order": [42, *GEN_SEEDS],  # 42 = STREAMED parent capture (vstream)
            "ids_version": meta.get("ids_version"),
            "capture_convention": meta.get("capture_convention"),
            "bundle_sha": meta["bundle_sha"],
            "subsample_seed": SUBSAMPLE_SEED,
            "bootstrap_seed": BOOT_SEED,
        },
        "wandb_url": None,  # no training in this round (plan §0)
        "hf_hub_url": f"https://huggingface.co/datasets/{C.HF_DATA_REPO}/tree/main/{args.hf_prefix}",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": C.reproducibility_metadata()["git_commit"],
        "gpu_hours_used": round((time.time() - t_start) / 3600.0, 2),
        "gpu_hours_budgeted": 4,
        "plan_deviations": [],
    }
    path = logs_dir / f"issue-{TASK_ID}-kresample-results.json"
    C.write_json_atomic(path, payload)
    logger.info("Wrote results sentinel %s", path)


def phase_pod(args) -> None:
    t0 = time.time()
    C.phase("pod")
    for tag in ("b0", "b1", "b2"):
        _run_child(tag, args)
    _results_sentinel(args, t0)
    C.phase("done")


# ── phase vstream: stream the parent draw's v from the #779 n1m capture chunks ──


_VSTREAM_CKPT_EVERY = 200  # found-set checkpoint cadence (order-independent resume)


def _capture_chunk_names(args) -> list[str]:
    """Scoped .pt chunk listing under the parent n1m capture prefix (the #833
    recipe: server-side list_repo_tree via list_hf_files_under_path; never a bare
    full-repo listing). Bookkeeping files excluded by the strict chunk regex."""
    import re

    from huggingface_hub import HfApi

    files = hub.list_hf_files_under_path(
        HfApi(), C.HF_DATA_REPO, EA.CAPTURE_PREFIX, repo_type="dataset"
    )
    chunk_re = re.compile(r"^shard\d+_chunk\d+\.pt$")
    names = sorted(n for n in (q.rsplit("/", 1)[-1] for q in files) if chunk_re.match(n))
    if args.max_chunks > 0:
        names = names[: args.max_chunks]
    return names


def _neighbor_first(rest: list[str], predicted: list[str]) -> list[str]:
    """Order the fallback scan so same-shard j±1 neighbors of predicted chunks come
    first: prediction errors are boundary drift from the parent's ~156 skipped rows,
    so residue ci live in adjacent chunks — early stop then fires within a few."""

    def _sj(nm: str) -> tuple[str, int]:
        shard, cnum = nm.split("_chunk")
        return shard, int(cnum.split(".")[0])

    pred = {_sj(n) for n in predicted}
    near = [n for n in rest if any((s, j + d) in pred for s, j in [_sj(n)] for d in (-1, 1))]
    far = [n for n in rest if n not in set(near)]
    return [*near, *far]


def _vstream_ckpt_fingerprint(sha: str, names: list[str]) -> str:
    h = hashlib.sha256()
    h.update(f"{sha}|{LAYER}|{EA.CAPTURE_PREFIX}|".encode())
    h.update("|".join(names).encode())
    return h.hexdigest()


def _load_vstream_ckpt(ckpt_path: Path, fp: str) -> dict[int, np.ndarray]:
    """Order-independent FOUND-SET checkpoint resume: a matching-fingerprint
    partial shrinks the needed set; a mismatch (bundle/layer/universe drift or a
    torn write) is ignored (re-stream from scratch, never a wrong read)."""
    side = ckpt_path.with_suffix(".json")
    if not (ckpt_path.exists() and side.exists()):
        return {}
    try:
        if json.loads(side.read_text()).get("fingerprint") != fp:
            logger.warning("[vstream] checkpoint fingerprint MISMATCH — ignoring partial")
            return {}
        z = np.load(ckpt_path)
        found = {int(c): v for c, v in zip(z["ci"], z["v"], strict=True)}
        logger.info("[vstream] RESUMED found-set checkpoint (%d rows)", len(found))
        return found
    except Exception as e:  # torn write -> re-stream (never a wrong read)
        logger.warning("[vstream] checkpoint unreadable (%s) — ignoring partial", e)
        return {}


def _write_vstream_ckpt(ckpt_path: Path, fp: str, found: dict[int, np.ndarray]) -> None:
    tmp = ckpt_path.with_name(ckpt_path.name + ".tmp.npz")
    ci = np.array(sorted(found), dtype=np.int64)
    np.savez(tmp, ci=ci, v=np.stack([found[int(c)] for c in ci]).astype(np.float32))
    os.replace(tmp, ckpt_path)
    C.write_json_atomic(ckpt_path.with_suffix(".json"), {"fingerprint": fp, "n": len(found)})


def _collect_v42(args, needed_ci: set[int], names: list[str], fetch_fn, sha: str) -> dict:
    """Collect the parent v(x) at LAYER for ``needed_ci`` from the n1m capture .pt
    chunks: raw-JSON sibling index probe -> predicted chunks first, then a
    neighbor-first full-scan fallback with early stop; per chunk download ->
    mmap-slice needed rows -> DELETE (peak ~one chunk; #761 stream-reduce
    pattern). Returns {"found": {ci: (H,) fp32}, "n_predicted", "n_scanned"}."""
    import issue779_fitter_fair_comparison as F
    import torch

    cache = args.scratch / "vstream_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "v42_stream_partial.npz"
    fp = _vstream_ckpt_fingerprint(sha, names)
    found = _load_vstream_ckpt(ckpt_path, fp)
    remaining = {int(c) for c in needed_ci} - set(found)
    lock = threading.Lock()
    stop = threading.Event()
    scanned = {"n": 0}

    def _one(name: str) -> None:
        if stop.is_set():
            return
        got = Path(fetch_fn(C.HF_DATA_REPO, f"{EA.CAPTURE_PREFIX}/{name}", cache))
        b = F._mmap_load(got)
        col = list(b["layers"]).index(LAYER)
        hits = [
            (int(c), b["v_x"][i, col, :].to(torch.float32).numpy())
            for i, c in enumerate(b["ci"])
            if int(c) in remaining
        ]
        del b
        got.unlink()
        with lock:
            scanned["n"] += 1
            for c, v in hits:
                found[c] = v
                remaining.discard(c)
            if not remaining:
                stop.set()
            elif scanned["n"] % _VSTREAM_CKPT_EVERY == 0 and found:
                _write_vstream_ckpt(ckpt_path, fp, found)
                logger.info(
                    "[vstream] checkpoint @ %d chunks scanned (%d/%d rows found)",
                    scanned["n"],
                    len(found),
                    len(needed_ci),
                )

    def _run_pool(subset: list[str]) -> None:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(_one, n) for n in subset]):
                f.result()  # re-raise any download/parse failure loud

    predicted: list[str] = []
    if remaining:
        json_names = [n.rsplit(".", 1)[0] + ".json" for n in names]
        # cheap raw-JSON sibling probe (same shard/chunk stems + ci sets as the .pt
        # captures — N1G writes both per chunk); smoke serves it from the fixture dir
        index = _probe_chunk_index(args, json_names, fetch_fn=fetch_fn)
        predicted = sorted(
            {
                jn.rsplit(".", 1)[0] + ".pt"
                for jn in json_names
                if index and any(lo <= ci < hi for ci in remaining for lo, hi in [index[jn]])
            }
        )
        if predicted:
            logger.info(
                "[vstream] predicted %d/%d .pt chunks for %d rows",
                len(predicted),
                len(names),
                len(remaining),
            )
            _run_pool(predicted)
    if remaining:
        rest = _neighbor_first([n for n in names if n not in set(predicted)], predicted)
        logger.info(
            "[vstream] fallback scan over %d chunks (%d rows missing; neighbor-first)",
            len(rest),
            len(remaining),
        )
        _run_pool(rest)
    assert not remaining, (
        f"[vstream] {len(remaining)} needed ci not found in the n1m capture chunks "
        f"(e.g. {sorted(remaining)[:5]}) — bundle/capture universe drift"
    )
    if found:
        _write_vstream_ckpt(ckpt_path, fp, found)  # complete found-set (idempotent resume)
    return {"found": found, "n_predicted": len(predicted), "n_scanned": scanned["n"]}


def _synth_vstream_chunks(args, bundle_rows: list[dict]) -> Path:
    """SMOKE-ONLY transport fixture (network boundary fake — the collector,
    probe, fallback, checkpoint and join code run IDENTICALLY): REAL-format
    .pt/.json chunk files whose v_x at LAYER sits at EXACTLY the stored e2
    distance from the stored fp16 prediction, so the analyze smoke's G1
    identity check reads ~0 rel error through the same code path. Distractor
    rows + uneven chunking keep the probe/fallback legs exercised."""
    import torch

    s = _stored_join(args)
    pos_of = {int(r): i for i, r in enumerate(s["rows"])}
    layers = [14, LAYER, 26]  # the parent CAPTURE_LAYERS trim
    col = layers.index(LAYER)
    rows = []
    for r in bundle_rows:
        pos = pos_of[int(r["row_idx"])]
        vhat = s["pred16"][pos].astype(np.float64)
        rng = np.random.default_rng(int(r["ci"]))
        u = rng.normal(size=HIDDEN)
        u /= np.linalg.norm(u)
        rows.append((int(r["ci"]), vhat + math.sqrt(float(s["e2"][pos])) * u))
    hi_ci = max(c for c, _ in rows)
    rows.extend((hi_ci + 1_000_000 + i, np.zeros(HIDDEN)) for i in range(5))  # distractors
    rows.sort(key=lambda t: t[0])
    chunk_dir = args.scratch / "vstream_smoke_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cuts = [0, max(1, len(rows) // 3), max(2, (2 * len(rows)) // 3), len(rows)]
    for j in range(3):
        part = rows[cuts[j] : cuts[j + 1]]
        if not part:
            continue
        vx = torch.zeros((len(part), len(layers), HIDDEN), dtype=torch.float16)
        vx[:, col, :] = torch.tensor(np.stack([v for _, v in part]), dtype=torch.float16)
        torch.save(
            {
                "cx_last": torch.zeros((len(part), len(layers), HIDDEN), dtype=torch.float16),
                "v_x": vx,
                "ci": [c for c, _ in part],
                "prompts": ["" for _ in part],
                "layers": layers,
                "shard_index": 0,
                "chunk": j,
            },
            chunk_dir / f"shard00_chunk{j:04d}.pt",
        )
        (chunk_dir / f"shard00_chunk{j:04d}.json").write_text(
            json.dumps({"rows": [{"ci": c, "prompt": "", "response": "x"} for c, _ in part]})
        )
    return chunk_dir


def _local_fetch_fn(chunk_dir: Path):
    """Smoke transport: serve chunks from the synthesized local dir by COPY (the
    collector deletes after slicing, so the fixture must survive re-reads)."""
    import shutil

    def fetch(repo: str, fn: str, cache: Path) -> str:
        src = chunk_dir / fn.rsplit("/", 1)[-1]
        dst = Path(cache) / src.name
        shutil.copyfile(src, dst)
        return str(dst)

    return fetch


def phase_vstream(args) -> None:
    C.phase("vstream")
    for name in BUNDLE_PARTS:  # VM phase: stage the bundle from HF if absent (b0 pattern)
        p = args.out / "inputs" / name
        if not p.exists():
            hub.stage_hub_file(C.HF_DATA_REPO, f"{args.hf_prefix}/inputs/{name}", p)
    bundle_rows, sha = _load_bundle(args)
    out_npz = args.out / "v42_stream.npz"
    meta_path = args.out / "v42_stream_meta.json"
    if (
        out_npz.exists()
        and meta_path.exists()
        and json.loads(meta_path.read_text()).get("bundle_sha") == sha
    ):
        logger.info("[vstream] v42_stream.npz complete (sha match) — skip")
    else:
        needed = {int(r["ci"]) for r in bundle_rows}
        if args.smoke:
            chunk_dir = _synth_vstream_chunks(args, bundle_rows)
            names = sorted(p.name for p in chunk_dir.glob("*.pt"))
            fetch_fn = _local_fetch_fn(chunk_dir)
        else:
            names = _capture_chunk_names(args)
            fetch_fn = N1M._download_chunk_with_retry
        got = _collect_v42(args, needed, names, fetch_fn, sha)
        v42 = np.stack([got["found"][int(r["ci"])] for r in bundle_rows]).astype(np.float32)
        np.savez(
            out_npz,
            v42=v42,
            ci=np.array([int(r["ci"]) for r in bundle_rows], dtype=np.int64),
            rows=np.array([int(r["row_idx"]) for r in bundle_rows], dtype=np.int64),
        )
        C.write_json_atomic(
            meta_path,
            {
                "bundle_sha": sha,
                "layer": LAYER,
                "n": len(bundle_rows),
                "capture_prefix": EA.CAPTURE_PREFIX,
                "n_predicted_chunks": got["n_predicted"],
                "n_scanned_chunks": got["n_scanned"],
                "smoke_synth_transport": bool(args.smoke),
                "convention": "streamed verbatim from the parent n1m capture chunks "
                "(v_x at layer 19; NO recapture — the stored arrays' own bytes)",
                **C.reproducibility_metadata(),
            },
        )
        logger.info(
            "[vstream] v42_stream.npz written (%d rows; %d chunks scanned)",
            len(bundle_rows),
            got["n_scanned"],
        )
    if not args.skip_upload:
        _upload_files_failloud(args, [out_npz, meta_path], "analysis_tensors")
    EA._phase_sentinel("kres-vstream", f"vstream complete ({len(bundle_rows)} rows)")


# ── phase C: gates + estimator + bootstrap + figures ────────────────────────────


def _boot_means(
    vals: np.ndarray, n_boot: int, rng: np.random.Generator, chunk: int = 1000
) -> np.ndarray:
    """Bootstrap resample-means of ``vals`` — chunked gathers, no per-draw Python
    loop over pool reductions (parent P6 `_boot_group_delta` pattern)."""
    n = len(vals)
    out = np.empty(n_boot, dtype=np.float64)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        take = rng.integers(0, n, size=(b, n))
        out[s : s + b] = vals[take].mean(1)
    return out


def _pct_ci(draws: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _g1(e2_streamed: np.ndarray, e2_stored: np.ndarray) -> dict:
    """Gate G1 (v68 redefinition) — streamed-v42 IDENTITY check of the join:
    median relative |Δe2| + Spearman of e2 recomputed from the STREAMED parent v
    (vstream — the stored arrays' own bytes) against the stored holdout_e2. The
    only expected deviation is the fp16 quantization of holdout_pred16, so the
    tolerance is tight (median rel <= 1e-3, Spearman >= 0.999). Verdict only;
    HALT policy is the caller's (production HALTs, smoke demotes)."""
    from scipy.stats import spearmanr

    rel = np.abs(e2_streamed - e2_stored) / np.maximum(e2_stored, 1e-12)
    med = float(np.median(rel))
    rho = float(spearmanr(e2_streamed, e2_stored).statistic)
    return {
        "median_rel_abs_de2": med,
        "p90_rel_abs_de2": float(np.percentile(rel, 90)),
        "max_rel_abs_de2": float(rel.max()),
        "spearman": rho,
        "pass": bool(med <= G1_MED_REL_MAX and rho >= G1_SPEARMAN_MIN),
        "thresholds": {"median_rel_max": G1_MED_REL_MAX, "spearman_min": G1_SPEARMAN_MIN},
        "convention": "streamed-v42-identity (no recapture)",
    }


def _g2(e2_k: np.ndarray, en_mask: np.ndarray) -> dict:
    """Gate G2 — parent-vs-now sampling-distribution check (v68): average rank of
    the STREAMED parent draw's e2 (column 0) among the 5 draws (streamed v42 +
    fresh 43-46, all in the parent capture convention); |mean-3.0|<=0.15. Per-arm
    breakdown (critic diagnostic). DIAGNOSTIC as of v68 — the registered floor is
    the fresh-4 form regardless, so G2 no longer switches the estimator."""
    from scipy.stats import rankdata

    ranks = rankdata(e2_k, method="average", axis=1)[:, 0]
    mean_rank = float(ranks.mean())
    return {
        "mean_rank_draw42": mean_rank,
        "per_arm_mean_rank": {
            "en": float(ranks[en_mask].mean()),
            "nonen": float(ranks[~en_mask].mean()),
        },
        "pass": bool(abs(mean_rank - 3.0) <= G2_RANK_TOL),
        "tolerance": G2_RANK_TOL,
    }


def _estimators(V: np.ndarray, v42: np.ndarray, vhat: np.ndarray, denom: np.ndarray) -> dict:
    """Registered fresh-4 floor estimator (the plan's G2-fallback form, PRIMARY as
    of v68: trvar over the 4 fresh parent-convention draws 43-46, ddof=1;
    m2 = ||vhat - mean(V_fresh)||^2 - trvar/4) PLUS the 5-draw shadow variant
    with the STREAMED parent draw ``v42`` as draw 0 (diagnostic only). Asserts
    the unbiasedness identity m2 + trvar == mean_k e2_k exactly per set
    (algebraic; fp64)."""

    def _one(Vk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        vbar = Vk.mean(1)
        trvar = Vk.var(1, ddof=1).sum(-1)
        e2_k = ((Vk - vhat[:, None, :]) ** 2).sum(-1)
        m2 = ((vhat - vbar) ** 2).sum(-1) - trvar / Vk.shape[1]
        ident = np.max(np.abs(m2 + trvar - e2_k.mean(1)) / np.maximum(e2_k.mean(1), 1e-12))
        assert ident < 1e-8, f"unbiasedness identity violated (max rel dev {ident:.3e})"
        return e2_k, trvar, m2, float(ident)

    assert V.shape[1] == len(GEN_SEEDS), V.shape
    _e2f, trvar, m2, id4 = _one(V)
    V5 = np.concatenate([v42[:, None, :], V], axis=1)
    e2_k5, trvar5, m2_5, id5 = _one(V5)
    return {
        "e2_k": e2_k5,  # (n, 5): column 0 = streamed parent draw (G2 + shadow)
        "trvar": trvar,  # PRIMARY (fresh-4, registered)
        "m2": m2,
        "floor_n": trvar / denom,
        "nerr_adj": m2 / denom,
        "trvar5": trvar5,  # 5-draw SHADOW (streamed v42 as draw 0)
        "m2_5": m2_5,
        "floor5_n": trvar5 / denom,
        "nerr_adj5": m2_5 / denom,
        "identity_max_rel_dev": max(id4, id5),
    }


def _ci_offsets(points, los, his) -> np.ndarray:
    """Element-wise NON-NEGATIVE errorbar offsets from CI bounds (never bounds,
    never signed deltas — matplotlib xerr/yerr contract, gotchas #547/#1335)."""
    p, lo, hi = (np.asarray(x, dtype=float) for x in (points, los, his))
    return np.vstack([np.maximum(0.0, p - lo), np.maximum(0.0, hi - p)])


def _fig_hero(fig_dir: Path, arm_stats: dict, deltas: dict, fname: str, unit_label: str) -> None:
    """Hero decomposition: per-arm stacked map/floor shares + the three delta
    intervals (raw, floor, adjusted) as CI whiskers (plan §6 figure 1)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    arms = ["English", "Non-English"]
    map_share = [arm_stats["en"]["map"], arm_stats["nonen"]["map"]]
    floor_share = [arm_stats["en"]["floor"], arm_stats["nonen"]["floor"]]
    ax1.bar(arms, map_share, label="map share", color="#4878CF")
    ax1.bar(arms, floor_share, bottom=map_share, label="answer-entropy floor", color="#EE854A")
    ax1.set_ylabel(unit_label)
    ax1.legend(frameon=False)
    names = ["raw Δ", "Δ floor", "Δ adjusted"]
    pts = [deltas[k]["point"] for k in ("raw", "floor", "adj")]
    los = [deltas[k]["ci"][0] for k in ("raw", "floor", "adj")]
    his = [deltas[k]["ci"][1] for k in ("raw", "floor", "adj")]
    ax2.axhline(0.0, color="0.6", lw=0.8)
    ax2.errorbar(
        range(3), pts, yerr=_ci_offsets(pts, los, his), fmt="o", color="#333333", capsize=3
    )
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(names)
    ax2.set_ylabel(f"non-EN minus EN ({unit_label})")
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=200)
    plt.close(fig)


def _figures(args, fig_dir: Path, d: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis import paper_plots

        paper_plots.set_paper_style()
    except Exception as e:  # cosmetic only
        logger.warning("[analyze] paper style unavailable (%s); default rcParams", e)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _fig_hero(
        fig_dir, d["arm_stats"], d["deltas"], "hero_decomposition.png", "mean normalized error"
    )
    _fig_hero(
        fig_dir,
        d["arm_stats_raw"],
        d["deltas_raw"],
        "hero_unnormalized.png",
        "mean squared error (raw units)",
    )

    en = d["en_mask"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(
        d["nerr_stored"][en], d["floor_n"][en], s=8, alpha=0.5, label="English", color="#4878CF"
    )
    ax.scatter(
        d["nerr_stored"][~en],
        d["floor_n"][~en],
        s=8,
        alpha=0.5,
        label="Non-English",
        color="#EE854A",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("stored per-context nerr")
    ax.set_ylabel("answer-entropy floor (nerr units)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter_floor_vs_nerr.png", dpi=200)
    plt.close(fig)

    codes = sorted(set(d["language"]) - {"en"})
    data = [d["floor_n"][d["language"] == c] for c in codes]
    if data:
        fig, ax = plt.subplots(figsize=(max(5.0, 0.5 * len(codes) + 2), 4.0))
        ax.boxplot(data, tick_labels=codes, showfliers=False)
        ax.set_ylabel("answer-entropy floor (nerr units)")
        ax.set_xlabel("language code (non-EN)")
        fig.tight_layout()
        fig.savefig(fig_dir / "floor_by_language.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(d["e2_stored"], d["e2_streamed42"], s=8, alpha=0.5, color="#333333")
    lim = [
        min(d["e2_stored"].min(), d["e2_streamed42"].min()),
        max(d["e2_stored"].max(), d["e2_streamed42"].max()),
    ]
    ax.plot(lim, lim, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("stored e2 (parent)")
    ax.set_ylabel("streamed parent e2 (draw 42, no recapture)")
    fig.tight_layout()
    fig.savefig(fig_dir / "g1_recapture_scatter.png", dpi=200)
    plt.close(fig)

    from scipy.stats import rankdata

    ranks = rankdata(d["e2_k"], method="average", axis=1)[:, 0]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.hist(ranks, bins=np.arange(0.75, 5.76, 0.5), color="#4878CF")
    ax.set_xlabel("rank of streamed parent draw's e2 among 5 draws (v42 + fresh 43-46)")
    ax.set_ylabel("contexts")
    fig.tight_layout()
    fig.savefig(fig_dir / "g2_rank_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    fr = [float((d["m2"][en] < 0).mean()), float((d["m2"][~en] < 0).mean())]
    ax.bar(["English", "Non-English"], fr, color=["#4878CF", "#EE854A"])
    ax.set_ylabel("fraction of contexts with m2 < 0")
    fig.tight_layout()
    fig.savefig(fig_dir / "m2_negative_fraction.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    pts = [d["deltas"]["adj"]["point"], d["mlp"]["delta_adj"]["point"]]
    los = [d["deltas"]["adj"]["ci"][0], d["mlp"]["delta_adj"]["ci"][0]]
    his = [d["deltas"]["adj"]["ci"][1], d["mlp"]["delta_adj"]["ci"][1]]
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.errorbar([0, 1], pts, yerr=_ci_offsets(pts, los, his), fmt="o", color="#333333", capsize=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ridge", "MLP w8192"])
    ax.set_ylabel("Δ adjusted (nerr units)")
    fig.tight_layout()
    fig.savefig(fig_dir / "mlp_vs_ridge_adjusted.png", dpi=200)
    plt.close(fig)
    logger.info("[analyze] figures -> %s", fig_dir)


def phase_analyze(args) -> None:
    C.phase("analyze")
    from scipy.stats import spearmanr

    v_path = args.out / "V.npz"
    meta_path = args.out / "capture_meta.json"
    if not v_path.exists():
        hub.stage_hub_file(C.HF_DATA_REPO, f"{args.hf_prefix}/analysis_tensors/V.npz", v_path)
        hub.stage_hub_file(
            C.HF_DATA_REPO, f"{args.hf_prefix}/analysis_tensors/capture_meta.json", meta_path
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("ids_version") != IDS_VERSION:  # v68: never read the retired text-retok V.npz
        raise RuntimeError(
            f"capture_meta ids_version {meta.get('ids_version')} != {IDS_VERSION} — stale "
            "pre-v68 text-retok capture; regenerate b1/b2 (retired convention)"
        )
    z = np.load(v_path)
    V = z["V"].astype(np.float64)
    assert [int(d) for d in z["draws"]] == list(GEN_SEEDS), z["draws"]
    n_ans = z["n_ans"].astype(np.float64)
    rows_sub = z["rows"]
    language = np.array([str(x) for x in z["language"]])
    en_mask = np.array([str(a) == "en" for a in z["arm"]])

    vs_path = args.out / "v42_stream.npz"
    vs_meta_path = args.out / "v42_stream_meta.json"
    if not vs_path.exists():
        hub.stage_hub_file(
            C.HF_DATA_REPO, f"{args.hf_prefix}/analysis_tensors/v42_stream.npz", vs_path
        )
        hub.stage_hub_file(
            C.HF_DATA_REPO, f"{args.hf_prefix}/analysis_tensors/v42_stream_meta.json", vs_meta_path
        )
    vs_meta = json.loads(vs_meta_path.read_text())
    assert vs_meta.get("bundle_sha") == meta["bundle_sha"], (
        "v42_stream bundle_sha != capture bundle_sha — run --phase vstream on this bundle"
    )
    zs = np.load(vs_path)
    slot_of = {int(c): i for i, c in enumerate(zs["ci"])}
    v42 = zs["v42"][np.array([slot_of[int(c)] for c in z["ci"]], dtype=np.int64)].astype(np.float64)

    s = _stored_join(args)
    pos_of = {int(r): i for i, r in enumerate(s["rows"])}
    pos = np.array([pos_of[int(r)] for r in rows_sub], dtype=np.int64)
    vhat = s["pred16"][pos].astype(np.float64)
    e2_s, denom_s, nerr_s = s["e2"][pos], s["denom"][pos], s["nerr"][pos]

    est = _estimators(V, v42, vhat, denom_s)
    enforce = not args.smoke  # production-n-calibrated gates demote under smoke (#1345)

    e2_42 = est["e2_k"][:, 0]  # streamed parent draw (identity read of the join)
    g1 = _g1(e2_42, e2_s)
    logger.info(
        "[analyze] G1(streamed-identity): median rel %.3e spearman %.6f pass=%s",
        g1["median_rel_abs_de2"],
        g1["spearman"],
        g1["pass"],
    )
    g2 = _g2(est["e2_k"], en_mask)

    # C1 coherence: subsample stored raw delta vs parent full delta (join/strata bug detector)
    ss_c1 = np.random.SeedSequence(BOOT_SEED)
    c1_en, c1_ne = (np.random.default_rng(c) for c in ss_c1.spawn(2))
    c1_draws = _boot_means(nerr_s[~en_mask], args.n_boot, c1_ne) - _boot_means(
        nerr_s[en_mask], args.n_boot, c1_en
    )
    c1_lo, c1_hi = _pct_ci(c1_draws)
    c1 = {
        "delta_sub_stored": float(nerr_s[~en_mask].mean() - nerr_s[en_mask].mean()),
        "ci": [c1_lo, c1_hi],
        "delta_full": s["delta_full"],
        "pass": bool(c1_lo <= s["delta_full"] <= c1_hi),
    }

    # v68: the registered floor IS the fresh-4 form (the plan's G2-fallback shape);
    # G2 is a parent-vs-now sampling-distribution DIAGNOSTIC — it no longer
    # switches the estimator. The 5-draw variant is reported as a shadow.
    estimator_used = "fresh4-registered"
    floor_use = est["floor_n"]

    gates = {
        "g1": {**g1, "enforced": enforce},
        "g2": {**g2, "enforced": False, "role": "diagnostic (v68 — no estimator switch)"},
        "c1": {**c1, "enforced": enforce},
        "identity_max_rel_dev": est["identity_max_rel_dev"],
        "determinism_check": _b1_determinism(args),
        "estimator_used": estimator_used,
        "ids_provenance": {
            "capture_ids_version": meta.get("ids_version"),
            "capture_convention": meta.get("capture_convention"),
            "seed42_source": meta.get("seed42_source"),
            "id_convention_diagnostic": meta.get("id_convention_diagnostic"),
            "seam_merge_rate_per_arm": {
                "en": float(z["seam_merge"][en_mask].mean()),
                "nonen": float(z["seam_merge"][~en_mask].mean()),
            },
            "gen_id_retok_mismatch_rate_per_arm": {
                "en": float(z["gen_id_retok_mismatch"][en_mask].mean()),
                "nonen": float(z["gen_id_retok_mismatch"][~en_mask].mean()),
            },
            "vstream_meta": {
                k: vs_meta.get(k)
                for k in ("n", "n_predicted_chunks", "n_scanned_chunks", "smoke_synth_transport")
            },
        },
        "n_contexts": len(rows_sub),
        "n_dropped": len(meta.get("dropped", [])),
        "tiny_model": bool(meta.get("tiny_model", False)),
        "metadata": C.reproducibility_metadata(),
    }
    args.out_eval.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_eval / "gates.json", gates)
    if enforce and not g1["pass"]:
        logger.error("[analyze] G1 FAIL (%s) — stored-join identity broken; HALT (plan §7)", g1)
        raise SystemExit(RC_G1)
    if not g1["pass"]:
        logger.warning("[analyze] G1 informational under smoke: %s", g1)
    if enforce and not c1["pass"]:
        logger.error("[analyze] C1 FAIL (%s) — stratification/join bug; HALT", c1)
        raise SystemExit(RC_C1)
    if not g2["pass"]:
        logger.warning(
            "[analyze] G2 diagnostic %s: mean rank %.3f (registered floor stays fresh-4)",
            "FAIL" if enforce else "informational under smoke",
            g2["mean_rank_draw42"],
        )

    # ── registered contrasts (joint bootstrap; chunked gathers) ────────────────
    nerr_full = s["nerr"][s["en"] | s["ne"]]
    en_full = s["en"][s["en"] | s["ne"]]
    ss = np.random.SeedSequence((BOOT_SEED, 1))
    (
        r_raw_en,
        r_raw_ne,
        r_fl_en,
        r_fl_ne,
        r_s1_en,
        r_s1_ne,
        r_mlp_en,
        r_mlp_ne,
        r_e2_en,
        r_e2_ne,
        r_tv_en,
        r_tv_ne,
    ) = (np.random.default_rng(c) for c in ss.spawn(12))
    nb = args.n_boot
    raw_en_d = _boot_means(nerr_full[en_full], nb, r_raw_en)
    raw_ne_d = _boot_means(nerr_full[~en_full], nb, r_raw_ne)
    fl_en_d = _boot_means(floor_use[en_mask], nb, r_fl_en)
    fl_ne_d = _boot_means(floor_use[~en_mask], nb, r_fl_ne)
    raw_delta_d = raw_ne_d - raw_en_d
    floor_delta_d = fl_ne_d - fl_en_d
    adj_d = raw_delta_d - floor_delta_d

    delta_raw = s["delta_full"]
    delta_floor = float(floor_use[~en_mask].mean() - floor_use[en_mask].mean())
    delta_adj = delta_raw - delta_floor
    adj_lo, adj_hi = _pct_ci(adj_d)
    fl_lo, fl_hi = _pct_ci(floor_delta_d)
    if adj_hi < 0:
        verdict = "Language-advantage survives"
    elif adj_lo > 0:
        verdict = "Re-attributed to answer entropy"
    else:
        verdict = "Inconclusive"
    sub_read = None
    if verdict == "Inconclusive":
        floor_excl = fl_lo > 0 or fl_hi < 0
        if floor_excl and abs(delta_floor) >= 0.5 * abs(delta_raw):
            sub_read = "entropy share substantial — mechanism gloss actively weakened"
        elif not floor_excl and (fl_hi - fl_lo) < abs(delta_raw):
            sub_read = "floor small and tight — adjustment underpowered at this K/n (raise n next)"

    adj_use = est["nerr_adj"]  # fresh-4 registered (v68)
    s1_draws = _boot_means(adj_use[~en_mask], nb, r_s1_ne) - _boot_means(
        adj_use[en_mask], nb, r_s1_en
    )
    s1 = {
        "delta_adj_sub": float(adj_use[~en_mask].mean() - adj_use[en_mask].mean()),
        "ci": list(_pct_ci(s1_draws)),
    }

    # MLP robustness: same floor, MLP stored nerr for the raw term (v_k fitter-independent)
    zm = _load_percontext(f"refit_holdout__mlp_w8192__seed{FIT_SEED}")
    nerr_mlp = zm["holdout_nerr"].astype(np.float64)[s["en"] | s["ne"]]
    mlp_raw_d = _boot_means(nerr_mlp[~en_full], nb, r_mlp_ne) - _boot_means(
        nerr_mlp[en_full], nb, r_mlp_en
    )
    mlp_adj_d = mlp_raw_d - floor_delta_d
    mlp_delta_raw = float(nerr_mlp[~en_full].mean() - nerr_mlp[en_full].mean())
    vhat_m = zm["holdout_pred16"][pos].astype(np.float64)
    # streamed-v identity read against the MLP fitter's stored arrays (no recapture)
    e2_m_42 = ((v42 - vhat_m) ** 2).sum(1)
    mlp = {
        "delta_raw_full": mlp_delta_raw,
        "delta_adj": {"point": mlp_delta_raw - delta_floor, "ci": list(_pct_ci(mlp_adj_d))},
        "g1_informational": _g1(e2_m_42, zm["holdout_e2"].astype(np.float64)[pos]),
    }

    # unnormalized (raw e2 units) sensitivity
    e2_full = s["e2"][s["en"] | s["ne"]]
    trvar_use = est["trvar"]  # fresh-4 registered (v68)
    e2_raw_d = _boot_means(e2_full[~en_full], nb, r_e2_ne) - _boot_means(
        e2_full[en_full], nb, r_e2_en
    )
    tv_d = _boot_means(trvar_use[~en_mask], nb, r_tv_ne) - _boot_means(
        trvar_use[en_mask], nb, r_tv_en
    )
    raw_units = {
        "delta_raw_full_e2": float(e2_full[~en_full].mean() - e2_full[en_full].mean()),
        "delta_floor_raw": float(trvar_use[~en_mask].mean() - trvar_use[en_mask].mean()),
    }
    raw_units["delta_adj_raw"] = {
        "point": raw_units["delta_raw_full_e2"] - raw_units["delta_floor_raw"],
        "ci": list(_pct_ci(e2_raw_d - tv_d)),
    }

    # diagnostics (zero-cost panel asks): 5-draw shadow (streamed v42 as draw 0),
    # per-code floors, floor-vs-length Spearman per arm, m2<0 fraction
    n_ans_mean = n_ans.mean(1)
    diagnostics = {
        "m2_negative_fraction": {
            "en": float((est["m2"][en_mask] < 0).mean()),
            "nonen": float((est["m2"][~en_mask] < 0).mean()),
        },
        "floor_vs_answer_len_spearman": {
            "en": float(spearmanr(floor_use[en_mask], n_ans_mean[en_mask]).statistic),
            "nonen": float(spearmanr(floor_use[~en_mask], n_ans_mean[~en_mask]).statistic),
        },
        "per_code_floor": {
            c: {
                "n": int((language == c).sum()),
                "mean": float(est["floor_n"][language == c].mean()),
                "median": float(np.median(est["floor_n"][language == c])),
            }
            for c in sorted(set(language))
        },
        "fivedraw_shadow": {
            "delta_floor5": float(
                est["floor5_n"][~en_mask].mean() - est["floor5_n"][en_mask].mean()
            ),
            "floor5_mean": {
                "en": float(est["floor5_n"][en_mask].mean()),
                "nonen": float(est["floor5_n"][~en_mask].mean()),
            },
        },
    }
    floor_summary = {
        "estimator_used": estimator_used,
        "per_arm": {
            "en": {
                "floor_mean": float(floor_use[en_mask].mean()),
                "nerr_adj_mean": float(adj_use[en_mask].mean()),
                "nerr_stored_mean": float(nerr_s[en_mask].mean()),
                "floor_share_of_nerr": float(floor_use[en_mask].mean() / nerr_s[en_mask].mean()),
            },
            "nonen": {
                "floor_mean": float(floor_use[~en_mask].mean()),
                "nerr_adj_mean": float(adj_use[~en_mask].mean()),
                "nerr_stored_mean": float(nerr_s[~en_mask].mean()),
                "floor_share_of_nerr": float(floor_use[~en_mask].mean() / nerr_s[~en_mask].mean()),
            },
        },
        "delta_floor": {"point": delta_floor, "ci": [fl_lo, fl_hi]},
        "diagnostics": diagnostics,
        "n_boot": nb,
        "bootstrap_seed": BOOT_SEED,
        "metadata": C.reproducibility_metadata(),
    }
    C.write_json_atomic(args.out_eval / "floor_summary.json", floor_summary)

    adjusted = {
        "verdict": verdict,
        "inconclusive_sub_read": sub_read,
        "delta_raw_full": delta_raw,
        "delta_floor": {"point": delta_floor, "ci": [fl_lo, fl_hi]},
        "delta_adj": {"point": delta_adj, "ci": [adj_lo, adj_hi]},
        "raw_delta_ci": list(_pct_ci(raw_delta_d)),
        "s1_self_contained": s1,
        "mlp_robustness": mlp,
        "raw_units_sensitivity": raw_units,
        "c1_coherence": c1,
        "estimator_used": estimator_used,
        "n": {
            "en_full": int(en_full.sum()),
            "nonen_full": int((~en_full).sum()),
            "en_sub": int(en_mask.sum()),
            "nonen_sub": int((~en_mask).sum()),
        },
        "n_boot": nb,
        "seeds": {
            "subsample": SUBSAMPLE_SEED,
            "bootstrap": BOOT_SEED,
            "generation": list(GEN_SEEDS),
            "fit": FIT_SEED,
        },
        "smoke": bool(args.smoke),
        "metadata": C.reproducibility_metadata(),
    }
    C.write_json_atomic(args.out_eval / "adjusted_contrast.json", adjusted)
    np.savez(
        args.out_eval / "percontext_floor.npz",
        rows=rows_sub,
        ci=z["ci"],
        arm=z["arm"],
        language=z["language"],
        floor_n=est["floor_n"],  # PRIMARY: fresh-4 registered (v68)
        nerr_adj=est["nerr_adj"],
        m2=est["m2"],
        trvar=est["trvar"],
        floor5_n=est["floor5_n"],  # SHADOW: 5-draw with streamed v42 as draw 0
        nerr_adj5=est["nerr_adj5"],
        m2_5=est["m2_5"],
        n_ans=z["n_ans"],
        seam_merge=z["seam_merge"],
        gen_id_retok_mismatch=z["gen_id_retok_mismatch"],
        nerr_stored=nerr_s,
        e2_stored=e2_s,
        e2_streamed42=e2_42,
    )

    _figures(
        args,
        args.figures,
        {
            "arm_stats": {
                "en": {
                    "map": floor_summary["per_arm"]["en"]["nerr_adj_mean"],
                    "floor": floor_summary["per_arm"]["en"]["floor_mean"],
                },
                "nonen": {
                    "map": floor_summary["per_arm"]["nonen"]["nerr_adj_mean"],
                    "floor": floor_summary["per_arm"]["nonen"]["floor_mean"],
                },
            },
            "arm_stats_raw": {
                "en": {
                    "map": float(est["m2"][en_mask].mean()),
                    "floor": float(trvar_use[en_mask].mean()),
                },
                "nonen": {
                    "map": float(est["m2"][~en_mask].mean()),
                    "floor": float(trvar_use[~en_mask].mean()),
                },
            },
            "deltas": {
                "raw": {"point": delta_raw, "ci": list(_pct_ci(raw_delta_d))},
                "floor": {"point": delta_floor, "ci": [fl_lo, fl_hi]},
                "adj": {"point": delta_adj, "ci": [adj_lo, adj_hi]},
            },
            "deltas_raw": {
                "raw": {"point": raw_units["delta_raw_full_e2"], "ci": list(_pct_ci(e2_raw_d))},
                "floor": {"point": raw_units["delta_floor_raw"], "ci": list(_pct_ci(tv_d))},
                "adj": raw_units["delta_adj_raw"],
            },
            "en_mask": en_mask,
            "language": language,
            "floor_n": est["floor_n"],
            "nerr_stored": nerr_s,
            "e2_stored": e2_s,
            "e2_streamed42": e2_42,
            "e2_k": est["e2_k"],
            "m2": est["m2"],
            "mlp": mlp,
        },
    )
    logger.info(
        "[analyze] verdict=%s delta_adj=%.6f CI=[%.6f, %.6f] (floor %.6f [%.6f, %.6f])",
        verdict,
        delta_adj,
        adj_lo,
        adj_hi,
        delta_floor,
        fl_lo,
        fl_hi,
    )
    EA._phase_sentinel("kres-analyze", f"analyze complete: {verdict}")


def _b1_determinism(args) -> dict | None:
    p = args.out / "gen" / "b1_meta.json"
    if p.exists():
        return json.loads(p.read_text()).get("determinism_check")
    return None


# ── main ────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1482 k-resample noise-floor driver.")
    ap.add_argument(
        "--phase",
        required=True,
        choices=["subsample", "pod", "b0", "b1", "b2", "vstream", "analyze"],
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--full", action="store_true", help="explicit production mode (default)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=Path, default=None, help="work dir (bundle/gen/V.npz)")
    ap.add_argument("--out-eval", type=Path, default=None, help="git-destined JSON/npz dir")
    ap.add_argument("--figures", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None, help="chunk-fetch cache dir")
    ap.add_argument(
        "--hf-prefix",
        default=None,
        help="REQUIRED in production (pass issue1482_kresample explicitly — no silent "
        "issue-prefix fallback at an upload destination, #1005); smoke defaults to "
        "issue1482_kresample_smoke",
    )
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--n-per-arm", type=int, default=None, help="prod 1000 / smoke 20")
    ap.add_argument("--n-boot", type=int, default=None, help="prod 10000 / smoke 200")
    ap.add_argument("--gen-batch", type=int, default=None, help="b2 capture rows/batch")
    ap.add_argument("--token-budget", type=int, default=None, help="b2 batch token cap")
    ap.add_argument("--workers", type=int, default=None, help="phase-A fetch threads")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all raw chunks")
    ap.add_argument(
        "--tiny-model",
        action="store_true",
        help="CARVE-OUT (GPU-bound b2 on a no-GPU VM): from-config same-arch Qwen2 "
        "over the REAL vocab instead of the 7B weights (#906 tiny-real pattern)",
    )
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    args = ap.parse_args()

    smoke_defaults = {
        "n_per_arm": 20,
        "n_boot": 200,
        "gen_batch": 4,
        "token_budget": 8192,
        "workers": 4,
    }
    prod_defaults = {
        "n_per_arm": 1000,
        "n_boot": 10_000,
        "gen_batch": 32,
        "token_budget": 32_768,
        "workers": 8,
    }
    for k, v in (smoke_defaults if args.smoke else prod_defaults).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.device == "auto":
        args.device = "cuda" if EA._physical_gpu_ids() else "cpu"
    root = PROJECT_ROOT / "data" / "issue_1482"
    base = root / ("kresample_smoke" if args.smoke else "kresample")
    if args.out is None:
        args.out = base
    if args.out_eval is None:
        # smoke outputs NEVER touch the committed eval_results/figures paths
        args.out_eval = (
            (base / "eval")
            if args.smoke
            else (PROJECT_ROOT / "eval_results" / "issue_1482" / "kresample")
        )
    if args.figures is None:
        args.figures = (
            (base / "figures")
            if args.smoke
            else (PROJECT_ROOT / "figures" / "issue_1482" / "kresample")
        )
    if args.scratch is None:
        args.scratch = base / "scratch"
    if args.hf_prefix is None:
        if args.smoke:
            args.hf_prefix = HF_PREFIX_DEFAULT + "_smoke"
        else:
            ap.error("--hf-prefix is required in production (pass issue1482_kresample)")
    elif args.smoke and args.hf_prefix == HF_PREFIX_DEFAULT:
        args.hf_prefix = HF_PREFIX_DEFAULT + "_smoke"  # canonical bucket protected regardless
    for p in (args.out, args.scratch):
        p.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "subsample": phase_subsample,
        "pod": phase_pod,
        "b0": phase_b0,
        "b1": phase_b1,
        "b2": phase_b2,
        "vstream": phase_vstream,
        "analyze": phase_analyze,
    }
    dispatch[args.phase](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
