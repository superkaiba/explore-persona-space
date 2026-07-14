#!/usr/bin/env python3
"""Issue #779 inline follow-up (``fitter-fair-comparison-n50k``): corpus
extension to n_train=50,000 + combined 28-layer capture, MULTI-GPU SHARDED.

Extends the round-2 (n10k) generate+capture driver
(``issue779_ffc_n10k_generate_capture.py``) to 40,000 NEW LMSYS contexts,
DISJOINT from ALL 11,500 already used (round-1's 5000 pass_b + round-2's 6500
n10k). ``n`` is the ONLY variable: the SAME pass-B rollout recipe (1 stochastic
rollout per context, temp 1.0 / top_p 0.95 / seed 42 / max 1024) and the SAME
combined 28-layer capture summaries (c_last / c_mean / v_x + the 4 pass-1 + 8
pass-2 answer summaries) as n10k — every capture position/pooling is REUSED from
the round-1 capture functions via ``issue779_ffc_n10k_generate_capture._capture_shard``.

Two things are genuinely new here (n50k-specific), everything else is REUSE:

1. **Disjoint 3-phase sampling.** ``sample_disjoint_n50k`` re-derives round-1
   (first 5000 non-empty first-turns) + n10k (next 6500 string-disjoint) from the
   deterministic LMSYS stream, then selects the NEXT 40,000 string-disjoint
   first-turns = the n50k set, and hard-asserts the n50k set is disjoint from
   round-1 + n10k (union). A ctx0 assert on round-1's first prompt guards the stream
   ordering the disjointness re-derivation depends on. The set is written ONCE to
   ``sampling_manifest.json`` (``--build-sampling-manifest``); every shard reads
   it and slices its contiguous range (no K redundant streams, no race).

2. **Multi-GPU sharding + per-chunk upload-verify-PURGE.** ``--num-shards K
   --shard-index i`` slices the 40,000 n50k prompts into K contiguous ranges;
   each shard captures its range in ``--shard-size`` sub-chunks and, per chunk:
   STACKS the kept rows into one bundle ``.pt`` (mmap-slice-friendly for the
   fits driver), writes the rollout-text ``raw_completions`` JSON, uploads BOTH
   to the HF data repo (``issue779_monitoring/fitter-fair-comparison-n50k/``),
   sha256-verifies the ``.pt`` against the Hub LFS metadata, then DELETES the
   local ``.pt`` before the next chunk. The full capture is ~7.6 MB/context
   (~306 GB total) — WAY over the ~130 GB MooseFS per-pod quota — so nothing
   accumulates on disk: peak local footprint is ~one in-flight chunk per
   process (< ~4 GB) x K, bounded well under 60 GB. Resume skips chunks already
   on the Hub.

FAITHFUL-REUSE DEVIATION (recorded in metadata.deviations, one line): c_last /
c_mean / v_x are captured per-row (batch-1 forwards) via the round-1
``capture_context_vector`` / ``capture_answer_vector``, NOT batched. This is the
same deviation the n10k driver recorded: the round-1 capture recipes tokenize
different sequences and a batched rewrite would risk left-pad/bf16 padded-batch
numeric divergence from the n10k/round-1 corpus (the #779 r12 equivalence-gate
class), breaking the load-bearing "n is the only variable" constraint. The 8+8
answer summaries ARE batched (``capture_summaries_batched`` /
``capture_pass2_batched``), and vLLM generation IS chunked — correctness over the
one-forward micro-optimization for the per-row c_x/v_x.

Refusal-safety: LMSYS is an unscreened real-user corpus. This driver NEVER prints
or logs example context/rollout text — only counts, indices, and sha256s. Do not
add such logging.

GPU (H100/A100) per shard. NO judge/API calls. Fail loud — NaN never coerced.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# vLLM V1 fork-safety (#628): spawn BEFORE any vllm import in the process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# The n10k driver sets the spawn method + loads dotenv at import; we reuse its
# capture path verbatim (protocol-identity with the n10k/round-1 corpus).
import issue779_common as C  # noqa: E402
import issue779_ffc_n10k_generate_capture as N10  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_ffc_n50k_gc")

N_LAYERS = C.EXPECTED_LAYERS  # 28
H_DIM = C.EXPECTED_HIDDEN  # 3584
DEFAULT_MODEL = N10.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct (round-1 pass-B model)
LMSYS_REPO = N10.LMSYS_REPO
HF_PREFIX = "issue779_monitoring/fitter-fair-comparison-n50k"

# Corpus counts (n is the only variable).
N_ROUND1 = 5000  # round-1 pass_b non-empty first-turns (re-derived by skipping the first 5000)
N_N10K = 6500  # round-2 n10k new disjoint first-turns
N_N50K = 40000  # round-3 n50k new disjoint first-turns (this driver)


def _sha_ids_or_prompts(prompts: list[str]) -> str:
    """sha256 of an ordered prompt list (== N10._sha_prompts; re-exported for clarity)."""
    return N10._sha_prompts(prompts)


def sample_disjoint_n50k(
    skip_round1: int,
    n_n10k: int,
    n_new: int,
    *,
    stream_iter=None,
) -> dict:
    """Deterministic 3-phase disjoint selection off the LMSYS stream.

    Phase 1: the first ``skip_round1`` non-empty first-turns  -> round-1 set.
    Phase 2: the next ``n_n10k`` first-turns NOT in the round-1 set  -> n10k set.
    Phase 3: the next ``n_new`` first-turns NOT in round-1 + n10k (union)  -> n50k set.

    The construction mirrors the n10k driver's 2-phase ``sample_disjoint`` exactly
    (its phase 2 continues from where round-1's phase 1 stopped); phase 3 continues
    from where n10k's phase 2 stopped, so the n50k set is the natural next block of
    disjoint prompts — reproducing the n10k selection en route and extending it.

    ``stream_iter`` (an iterator of row dicts) is injectable for the CPU smoke so
    the disjointness logic is exercised WITHOUT touching real LMSYS data
    (refusal-safety). Default: the deterministic streaming LMSYS train split.
    """
    if stream_iter is None:
        from datasets import load_dataset

        ds = load_dataset(LMSYS_REPO, split="train", streaming=True)
        it = iter(ds)
    else:
        ds = None
        it = iter(stream_iter)

    round1: list[str] = []
    round1_set: set[str] = set()
    n10k: list[str] = []
    n10k_set: set[str] = set()
    new: list[str] = []
    new_set: set[str] = set()
    new_pos: list[int] = []
    pos = -1

    # Phase 1: round-1 (first skip_round1 non-empty first-turns).
    while len(round1) < skip_round1:
        pos += 1
        p = N10._first_user_turn(next(it))
        if p:
            round1.append(p)
            round1_set.add(p)
    # Phase 2: n10k (next disjoint from round-1).
    while len(n10k) < n_n10k:
        pos += 1
        p = N10._first_user_turn(next(it))
        if p and p not in round1_set and p not in n10k_set:
            n10k.append(p)
            n10k_set.add(p)
    # Phase 3: n50k (next disjoint from round-1 + n10k (union)).
    used = round1_set | n10k_set
    while len(new) < n_new:
        pos += 1
        p = N10._first_user_turn(next(it))
        if p and p not in used and p not in new_set:
            new.append(p)
            new_set.add(p)
            new_pos.append(pos)

    disjoint_ok = new_set.isdisjoint(used) and n10k_set.isdisjoint(round1_set)
    assert disjoint_ok, "n50k/n10k prompts overlap already-used sets (should be impossible)"

    # release the streaming dataset before shutdown (#952 rc=134 guard)
    if ds is not None:
        del it, ds
        gc.collect()

    logger.info(
        "sampled %d round-1 + %d n10k + %d n50k disjoint prompts (last stream pos %d)",
        len(round1),
        len(n10k),
        len(new),
        pos,
    )
    return {
        "round1": round1,
        "n10k": n10k,
        "new": new,
        "new_stream_pos": new_pos,
        "round1_prompt_sha256": _sha_ids_or_prompts(round1),
        "n10k_prompt_sha256": _sha_ids_or_prompts(n10k),
        "new_prompt_sha256": _sha_ids_or_prompts(new),
        "disjoint_ok": bool(disjoint_ok),
        "skip_round1": skip_round1,
        "n_n10k": len(n10k),
        "n_new": len(new),
        "last_stream_pos": pos,
    }


def build_manifest(args) -> dict:
    """Stream + write sampling_manifest.json (all n50k prompts + positions + shas)."""
    C.phase("sample")
    manifest = sample_disjoint_n50k(args.skip_round1, args.n_n10k, args.n_new)
    # ctx0 assert: round-1's first prompt validates the stream-ordering re-derivation
    # the disjointness check depends on (normalized lower/strip/collapse ws).
    ctx0 = manifest["round1"][0]
    norm = " ".join(ctx0.lower().split()).rstrip(".?!,")
    assert norm == N10.EXPECTED_CTX0_PROMPT, (
        f"round-1 ctx0 re-derivation drift: got {ctx0[:80]!r} — the LMSYS stream ordering "
        "changed; the disjointness re-derivation is no longer trustworthy"
    )
    payload = {
        "new": manifest["new"],  # the 40,000 n50k prompts, in stream order
        "new_stream_pos": manifest["new_stream_pos"],
        "round1_prompt_sha256": manifest["round1_prompt_sha256"],
        "n10k_prompt_sha256": manifest["n10k_prompt_sha256"],
        "new_prompt_sha256": manifest["new_prompt_sha256"],
        "disjoint_ok": manifest["disjoint_ok"],
        "skip_round1": manifest["skip_round1"],
        "n_n10k": manifest["n_n10k"],
        "n_new": manifest["n_new"],
        "last_stream_pos": manifest["last_stream_pos"],
        "model": args.model,
        "layers": list(range(N_LAYERS)),
        "source": LMSYS_REPO,
        "metadata": C.reproducibility_metadata(
            {"script": "issue779_ffc_n50k_generate_capture", "phase": "sampling_manifest"}
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_dir / "sampling_manifest.json", payload)
    logger.info(
        "wrote %s (%d n50k prompts)", args.out_dir / "sampling_manifest.json", len(payload["new"])
    )
    return payload


def _read_manifest(args) -> list[str]:
    mp = args.out_dir / "sampling_manifest.json"
    if not mp.exists():
        raise SystemExit(
            f"sampling manifest {mp} absent — run --build-sampling-manifest first "
            "(the launcher builds it once before fan-out)"
        )
    import json

    d = json.loads(mp.read_text())
    prompts = d["new"]
    assert len(prompts) == d["n_new"], (len(prompts), d["n_new"])
    if len(prompts) != args.n_new:
        logger.warning(
            "manifest has %d n50k prompts but --n-new=%d; using the manifest's %d",
            len(prompts),
            args.n_new,
            len(prompts),
        )
    return prompts


def _shard_range(n_total: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    """Contiguous [start, end) range of the n50k prompt list for this shard.

    Even split with the remainder distributed to the first shards, so every
    prompt is covered exactly once across shards 0..num_shards-1."""
    assert 0 <= shard_index < num_shards, (shard_index, num_shards)
    base, rem = divmod(n_total, num_shards)
    start = shard_index * base + min(shard_index, rem)
    size = base + (1 if shard_index < rem else 0)
    return start, start + size


def _stack_chunk(rows: list[dict], layers: list[int], shard_index: int, chunk_idx: int) -> dict:
    """Stack per-row capture dicts (from N10._capture_shard) into one mmap-slice-
    friendly bundle. Mirrors the n10k final-bundle field layout so the fits driver
    can ``torch.load(mmap=True)[fld][:, layer_col, :]`` a single layer cheaply."""
    return {
        "cx_last": torch.stack([r["cx_last"] for r in rows]),  # (n, L, H)
        "cx_mean": torch.stack([r["cx_mean"] for r in rows]),  # (n, L, H)
        "v_x": torch.stack([r["v_x"] for r in rows]),  # (n, L, H)
        "summ_p1": torch.stack([r["summ_p1"] for r in rows]),  # (n, 4, L, H) fp16
        "valid_p1": torch.stack([r["valid_p1"] for r in rows]),  # (n, 4)
        "summ_p2": torch.stack([r["summ_p2"] for r in rows]),  # (n, 8, L, H) fp16
        "ci": [int(r["ci"]) for r in rows],  # GLOBAL n50k index (manifest order)
        "prompts": [r["prompt"] for r in rows],
        "layers": list(layers),
        "shard_index": int(shard_index),
        "chunk": int(chunk_idx),
        "p1_summaries": list(N10.P1.SUMMARIES),
        "p2_summaries": list(N10.P2.SUMMARIES2),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _remote_index(prefix: str) -> dict[str, dict]:
    """{basename: {size, sha256}} for a data-repo prefix (scoped list_repo_tree —
    never a bare full-repo listing on the ~1M-file data repo, #833)."""
    from huggingface_hub import HfApi

    out: dict[str, dict] = {}
    try:
        tree = HfApi().list_repo_tree(
            C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
        )
    except Exception as e:  # prefix may not exist yet (first run)
        logger.info("[resume] prefix %s not listable yet (%s); assuming empty", prefix, e)
        return out
    for f in tree:
        if getattr(f, "size", None) is None:
            continue
        lfs = getattr(f, "lfs", None)
        out[f.path.rsplit("/", 1)[-1]] = {"size": f.size, "sha256": lfs.sha256 if lfs else None}
    return out


def _upload_verify_purge(local_pt: Path, prefix: str, name: str) -> None:
    """Upload one capture chunk .pt, sha256-verify against Hub LFS metadata, purge local."""
    local_sha = _sha256_file(local_pt)
    url = hub._upload(
        local_pt,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{prefix}/final_token_capture/{name}",
        upload_as_file=True,
    )
    if not url:  # hub._upload fail-soft returns "" on failure — make it fail loud here
        raise RuntimeError(f"upload of {name} to {prefix}/final_token_capture returned no URL")
    remote = _remote_index(f"{prefix}/final_token_capture")
    meta = remote.get(name)
    if meta is None:
        raise RuntimeError(f"{name} not present on Hub after upload (verify listing)")
    if meta["sha256"] is None or meta["sha256"] != local_sha:
        raise RuntimeError(
            f"{name} Hub LFS sha256 {meta['sha256']} != local {local_sha} — upload corrupt"
        )
    local_pt.unlink()  # PURGE: bound peak local footprint to ~one in-flight chunk
    logger.info("[upload] %s verified (sha %s..) + purged local", name, local_sha[:12])


def _upload_raw(local_json: Path, prefix: str, name: str) -> None:
    """Upload one rollout-text raw_completions JSON (text is NEVER discardable)."""
    url = hub._upload(
        local_json,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{prefix}/raw_completions/{name}",
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError(f"raw_completions upload of {name} returned no URL")
    remote = _remote_index(f"{prefix}/raw_completions")
    if name not in remote:
        raise RuntimeError(f"{name} not present on Hub after raw_completions upload")
    local_json.unlink()
    logger.info("[upload] raw_completions %s verified + purged", name)


def run_capture(args) -> int:

    prompts = _read_manifest(args)
    n_total = len(prompts)
    start, end = _shard_range(n_total, args.num_shards, args.shard_index)
    shard_prompts = prompts[start:end]
    layers = list(range(N_LAYERS))
    logger.info(
        "[shard %d/%d] range [%d, %d) = %d prompts",
        args.shard_index,
        args.num_shards,
        start,
        end,
        len(shard_prompts),
    )
    if not shard_prompts:
        logger.info("[shard %d] empty range; nothing to do", args.shard_index)
        C.phase("done")
        return 0

    scratch = args.out_dir / "shards"
    scratch.mkdir(parents=True, exist_ok=True)

    # Resume: chunks whose .pt AND raw json are already on the Hub are skipped.
    done_pt = set(_remote_index(f"{args.hf_prefix}/final_token_capture"))
    done_raw = set(_remote_index(f"{args.hf_prefix}/raw_completions"))

    C.phase("load_model")
    tok, hf = N10.load_models(args.model, args.device)
    llm = None
    if args.device == "cuda":
        from explore_persona_space.eval.generation import create_vllm_engine

        llm = create_vllm_engine(args.model, max_model_len=8192, seed=42)

    # one-time bf16 batched-vs-serial equivalence gate for the summary path (fail loud)
    C.phase("gates")
    gate = N10.P1.equivalence_gate(hf, tok, layers)
    logger.info("[shard %d] pass-1 equivalence gate: %s", args.shard_index, gate.get("pass", gate))

    C.phase("capture")
    n_sub = (len(shard_prompts) + args.shard_size - 1) // args.shard_size
    kept_total = 0
    for ci, s in enumerate(range(0, len(shard_prompts), args.shard_size)):
        name = f"shard{args.shard_index:02d}_chunk{ci:04d}.pt"
        raw_name = f"shard{args.shard_index:02d}_chunk{ci:04d}.json"
        if name in done_pt and raw_name in done_raw:
            logger.info(
                "[shard %d] chunk %d/%d already on Hub; skip", args.shard_index, ci + 1, n_sub
            )
            continue
        chunk = shard_prompts[s : s + args.shard_size]
        global_base = start + s  # global n50k index of chunk[0]
        ts = time.time()
        responses = N10._generate(llm, tok, chunk)
        rows = N10._capture_shard(hf, tok, chunk, responses, global_base, layers, args.batch_size)
        if not rows:
            logger.warning(
                "[shard %d] chunk %d: 0 kept rows (all empty responses); skip",
                args.shard_index,
                ci,
            )
            continue
        for fld in ("cx_last", "cx_mean", "v_x"):
            for r in rows:
                assert r[fld].shape == (N_LAYERS, H_DIM), (fld, r[fld].shape)
        bundle = _stack_chunk(rows, layers, args.shard_index, ci)
        chunk_pt = scratch / name
        torch.save(bundle, chunk_pt)
        raw_json = scratch / raw_name
        C.write_json_atomic(
            raw_json,
            {
                "shard_index": args.shard_index,
                "chunk": ci,
                "rows": [
                    {"ci": int(r["ci"]), "prompt": r["prompt"], "response": r["response"]}
                    for r in rows
                ],
            },
        )
        kept_total += len(rows)
        if not args.no_upload:
            _upload_raw(raw_json, args.hf_prefix, raw_name)  # text first (never discardable)
            _upload_verify_purge(chunk_pt, args.hf_prefix, name)  # then tensors + purge
        logger.info(
            "[shard %d] chunk %d/%d: %d/%d kept (%.0fs)",
            args.shard_index,
            ci + 1,
            n_sub,
            len(rows),
            len(chunk),
            time.time() - ts,
        )

    logger.info(
        "[shard %d] done: %d kept rows across %d chunks", args.shard_index, kept_total, n_sub
    )
    C.phase("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #779 n50k corpus extension + combined capture (multi-GPU sharded)."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-round1", type=int, default=N_ROUND1)
    ap.add_argument("--n-n10k", type=int, default=N_N10K)
    ap.add_argument("--n-new", type=int, default=N_N50K)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=500, help="contexts per capture sub-chunk")
    ap.add_argument("--batch-size", type=int, default=16, help="summary-capture batch size")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_779" / "ffc_n50k"
    )
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--no-upload", action="store_true", help="capture locally, do NOT upload/purge")
    ap.add_argument(
        "--build-sampling-manifest",
        action="store_true",
        help="stream + write sampling_manifest.json, then exit (no capture)",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="tiny CPU capture smoke (synthetic stream)"
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.build_sampling_manifest:
        build_manifest(args)
        return 0

    if args.smoke:
        return _smoke(args)

    return run_capture(args)


def _smoke(args) -> int:
    """MODEL-FREE CPU logic smoke (no real LMSYS, no 7B model, refusal-safe).

    The capture forwards are GPU-bound (Qwen-2.5-7B) — per the GPU-bound-phase
    carve-out, the smoke exercises only the CPU-runnable portion: the 3-phase
    disjoint sampling on a SYNTHETIC stream, the manifest write/read roundtrip,
    the shard-range partition-coverage invariant, and a signature check on the
    reused N10 capture entrypoints (the ABI the capture path calls). The real
    generation + capture run only on a GPU shard via the launcher."""
    import inspect

    logger.info("[smoke] model-free CPU logic smoke (disjointness + shard-range + signatures)")

    # (1) 3-phase disjoint selection on a synthetic stream.
    n_r1, n_n10, n_new = 4, 3, 5
    total = n_r1 + n_n10 + n_new + 5
    stream = [
        {"conversation": [{"content": f"synthetic smoke prompt number {i}"}]} for i in range(total)
    ]
    man = sample_disjoint_n50k(n_r1, n_n10, n_new, stream_iter=stream)
    assert man["disjoint_ok"] and man["n_new"] == n_new, man
    assert set(man["new"]).isdisjoint(man["round1"]) and set(man["new"]).isdisjoint(man["n10k"]), (
        "n50k set overlaps round1/n10k"
    )
    assert set(man["n10k"]).isdisjoint(man["round1"]), "n10k set overlaps round1"

    # (2) manifest write/read roundtrip.
    payload = {
        "new": man["new"],
        "new_stream_pos": man["new_stream_pos"],
        "round1_prompt_sha256": man["round1_prompt_sha256"],
        "n10k_prompt_sha256": man["n10k_prompt_sha256"],
        "new_prompt_sha256": man["new_prompt_sha256"],
        "disjoint_ok": man["disjoint_ok"],
        "skip_round1": man["skip_round1"],
        "n_n10k": man["n_n10k"],
        "n_new": man["n_new"],
        "last_stream_pos": man["last_stream_pos"],
        "model": args.model,
        "layers": list(range(N_LAYERS)),
        "source": "SYNTHETIC-SMOKE",
    }
    C.write_json_atomic(args.out_dir / "sampling_manifest.json", payload)
    args.n_new = n_new
    got = _read_manifest(args)
    assert got == man["new"], "manifest roundtrip mismatch"

    # (3) shard-range partition coverage: every prompt covered exactly once, in order.
    for k in (2, 4, 8):
        covered: list[int] = []
        for i in range(k):
            s, e = _shard_range(len(got), k, i)
            covered.extend(range(s, e))
        assert covered == list(range(len(got))), (k, covered)

    # (4) signature check on the reused N10 capture entrypoints (call-site ABI).
    cap_params = list(inspect.signature(N10._capture_shard).parameters)
    assert cap_params[:6] == ["hf", "tok", "prompts", "responses", "ci_base", "layers"], cap_params
    gen_params = list(inspect.signature(N10._generate).parameters)
    assert gen_params[:3] == ["llm", "tok", "prompts"], gen_params
    assert hasattr(N10.P1, "SUMMARIES") and hasattr(N10.P2, "SUMMARIES2"), "summary consts missing"

    logger.info(
        "[smoke] PASS: %d disjoint synthetic prompts; shard-range coverage k in {2,4,8}; "
        "N10 capture signatures match; manifest roundtrip ok",
        man["n_new"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
