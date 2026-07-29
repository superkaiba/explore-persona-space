"""#1776 Phase 5: cross-corpus transfer + free analyses (plan v4 §4 Phase 5).

Sub-commands, one per §4 Phase-5 leg:

  stream       5a.1 fresh WildChat prompt pool — stream ``allenai/WildChat-1M``
               with every n1m-sampling-manifest prompt EXCLUDED, starting AFTER
               the n1m build's last consumed WildChat stream position (every
               row past that point was never seen by any n1m fit; the manifest
               keys rows by PROMPT TEXT + ``stream_pos``, so "conversation id"
               exclusion == prompt-set exclusion + position skip). Per-chunk
               checkpoint + exact-fingerprint resume (dataset revision + filter
               constants) + per-filter reject counters in the done line
               (#1092 real-corpus rules).
  capture      5a.1 on-policy generation + teacher-forced capture of a prompt
               pool via the reused #779 n1m rig VERBATIM (``N10._generate``
               vLLM path; ``_capture_shard_trimmed`` cx_last + v_x at
               CAPTURE_LAYERS=[14,19,26]; ``_flush_upload_batch`` batched Hub
               commits). Rollout TEXT persists under
               ``{hf_prefix}/raw_completions/`` in the SAME flush as the
               capture .pt — before any downstream reduce. Also serves the
               persona-battery re-capture fallback (any {prompt,i} JSONL).
  transfer     5a.3 decay read — score ridge operators (``m_ridge_lmsys50k``,
               ``m_ridge_x50k``, shipped M reference) + J-affine arms on each
               leg (test-1000 LMSYS / fresh WildChat / persona battery):
               pooled R² + bootstrap CI + identity+learned-bias + kNN per
               (operator, leg); relative decay per H3, ratio gated on the
               LMSYS leg clearing identity+bias (§6 diagnostic).
  leakage      5b retrospective re-read (0 GPU, VM) — Spearman rho between the
               measured #532 leakage rates and centroid-bank persona
               similarity computed raw centered-cosine / P_J-projected /
               (I−P_J)-projected at the bank's own layer.
  lens-vocab   5c word tables — dictionary decodings of M′ / shipped-M top
               singular directions + the three r_B trait vectors.
  chain        5d chain-composition judge-free DV — MRR / recall@50 of the
               generated answer's content tokens in the lens-decoded vocab
               ranking of v̂(C) = M′·c_last^{(14)}(C) (shipped M·c_last^{(19)}
               reference variant), vs a shuffled-pairing null.
  smoke        CPU smokes (round C2b-i smoke log): tiny-REAL WildChat stream
               probe, synthetic-planted transfer round-trip, 5b re-read over
               the REAL committed #532 JSONs, tiny-lens (c)+(d), and
               signature-binds of the GPU-only capture calls.

Content hygiene: WildChat rows / model responses are NEVER printed or logged —
the pool/raw files are the required text persistence; logs carry counts, ids,
and shas only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402

WILDCHAT_REPO = "allenai/WildChat-1M"
STREAM_RECIPE = "issue1776-phase5-stream-v1"
DEFAULT_HF_PREFIX = f"{C76.HF_PREFIX}/wildchat_fresh"  # issue1776_jacobian/wildchat_fresh
DEFAULT_MANIFEST_DIR = (
    C76.DATA_DIR / "hf_dl" / "issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest"
)
POOL_DIR = C76.PROJECT_ROOT / "data" / "canonical_persona_pool"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _pooled_r2(pred: np.ndarray, y: np.ndarray) -> float:
    """Pooled multi-output R² (same convention as issue1776_comparator_fit)."""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


# ── 5a.1 stream ──────────────────────────────────────────────────────────────


def _dataset_revision(repo: str) -> str:
    """Current main sha of the streamed dataset (fingerprint key, #1092)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = hub.retry_transient(
        lambda: api.repo_info(repo, repo_type="dataset"), what=f"repo_info({repo})"
    )
    return str(info.sha)


def load_manifest_exclusion(manifest_dir: Path) -> tuple[set[str], int, dict]:
    """(excluded prompt set, first fresh WildChat stream position, digest meta).

    Excludes EVERY n1m-manifest prompt (both corpora — a duplicate text is a
    train-pool row wherever it streamed from); the position skip additionally
    guarantees rows the n1m build merely SAW (and rejected) are never resampled.
    """
    import issue779_ffc_n1m_generate_capture as N1G

    pool, meta = N1G.read_manifest_pool(manifest_dir)
    excluded = {r["prompt"] for r in pool}
    wc_pos = [int(r["stream_pos"]) for r in pool if r.get("corpus") == "wildchat"]
    start_pos = (max(wc_pos) + 1) if wc_pos else 0
    digest = {
        "n_pool": len(pool),
        "n_wildchat_rows": len(wc_pos),
        "manifest_new_prompt_sha256": meta.get("new_prompt_sha256"),
        "start_pos": start_pos,
    }
    return excluded, start_pos, digest


def stream_fresh_pool(
    args, excluded: set[str], start_pos: int, fingerprint: dict, stream_iter=None
) -> tuple[list[dict], dict]:
    """Keep the first ``--n-keep`` fresh WildChat first-turns past ``start_pos``.

    Per-chunk checkpoint (atomic pool JSONL + meta sidecar) with EXACT
    fingerprint resume; per-filter reject counters; hard total-scan cap
    (fail-loud). Rows: {"prompt", "stream_pos" (absolute), "i"}.
    """
    import issue779_ffc_n1m_generate_capture as N1G

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "wildchat_fresh_pool.jsonl"
    meta_path = out_dir / "wildchat_fresh_pool.meta.json"

    kept: list[dict] = []
    consumed = 0  # rows consumed AFTER start_pos
    counters = {"rej_empty": 0, "rej_excluded": 0, "rej_dup": 0}
    if pool_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            kept = N1G._read_jsonl(pool_path)
            consumed = int(meta.get("consumed", 0))
            counters = dict(meta.get("counters", counters))
            if meta.get("complete"):
                print(
                    f"[phase5-stream] RESUMED complete cache: kept={len(kept)} (stream skipped)",
                    flush=True,
                )
                return kept, counters
            print(
                f"[phase5-stream] RESUMED partial cache: kept={len(kept)} consumed={consumed}",
                flush=True,
            )
        else:
            print("[phase5-stream] fingerprint MISMATCH; re-streaming from scratch", flush=True)
            kept, consumed = [], 0

    if stream_iter is not None:
        it = iter(stream_iter)
        ds = None
    else:
        from datasets import load_dataset

        ds = load_dataset(WILDCHAT_REPO, split="train", streaming=True)
        skip_n = start_pos + consumed
        it = iter(ds.skip(skip_n) if skip_n else ds)

    def _flush(complete: bool) -> None:
        N1G._atomic_write_jsonl(pool_path, kept)
        C76.atomic_write_json(
            meta_path,
            {
                "fingerprint": fingerprint,
                "consumed": consumed,
                "kept": len(kept),
                "counters": counters,
                "complete": complete,
            },
        )

    seen: set[str] = {r["prompt"] for r in kept}
    row = None
    while len(kept) < args.n_keep:
        row = next(it, None)
        if row is None:
            break  # exhaustion — caller decides whether a short pool is fatal
        consumed += 1
        assert consumed <= args.max_scan, (
            f"scan cap {args.max_scan} hit with only {len(kept)}/{args.n_keep} kept — "
            f"counters={counters}; the filter chain or start_pos is wrong (fail loud)"
        )
        p = N1G.N10._first_user_turn(row)
        if not p:
            counters["rej_empty"] += 1
            continue
        if p in excluded:
            counters["rej_excluded"] += 1
            continue
        if p in seen:
            counters["rej_dup"] += 1
            continue
        kept.append({"prompt": p, "stream_pos": start_pos + consumed - 1, "i": len(kept)})
        seen.add(p)
        if len(kept) % args.checkpoint_every == 0:
            _flush(complete=False)
            print(
                f"[phase5-stream] checkpoint: kept={len(kept)} consumed={consumed} "
                f"rej_empty={counters['rej_empty']} rej_excluded={counters['rej_excluded']} "
                f"rej_dup={counters['rej_dup']}",
                flush=True,
            )
    _flush(complete=len(kept) >= args.n_keep)
    print(
        f"[phase5-stream] done: scanned={consumed} kept={len(kept)} "
        f"rej_empty={counters['rej_empty']} rej_excluded={counters['rej_excluded']} "
        f"rej_dup={counters['rej_dup']} start_pos={start_pos}",
        flush=True,
    )
    # Release the streaming dataset before shutdown (#952 rc=134 guard).
    if ds is not None:
        import gc

        del it, ds, row
        gc.collect()
    return kept, counters


def cmd_stream(args) -> int:
    excluded: set[str] = set()
    start_pos = 0
    man_digest: dict = {"skipped": True}
    if not args.no_manifest:
        excluded, start_pos, man_digest = load_manifest_exclusion(args.manifest_dir)
    if args.exclude_file is not None:
        import issue779_ffc_n1m_generate_capture as N1G

        extra = {r["prompt"] for r in N1G._read_jsonl(args.exclude_file)}
        excluded |= extra
        man_digest["n_exclude_file"] = len(extra)
    if args.start_mode == "head":
        start_pos = 0
    revision = _dataset_revision(WILDCHAT_REPO)
    fingerprint = {
        "repo": WILDCHAT_REPO,
        "revision": revision,
        "recipe": STREAM_RECIPE,
        "start_pos": start_pos,
        "n_keep": int(args.n_keep),
        "manifest_sha": man_digest.get("manifest_new_prompt_sha256"),
        "exclude_file_sha": (
            _sha256_file(args.exclude_file) if args.exclude_file is not None else None
        ),
    }
    kept, counters = stream_fresh_pool(args, excluded, start_pos, fingerprint)
    if len(kept) < args.n_keep and not args.allow_short:
        raise RuntimeError(
            f"stream exhausted at kept={len(kept)} < n_keep={args.n_keep} "
            f"(counters={counters}) — pass --allow-short only if a short pool is acceptable"
        )
    assert kept, "0 rows kept — filter chain rejected everything (see counters above)"
    C76.atomic_write_json(
        args.out_dir / "stream_report.json",
        {
            "fingerprint": fingerprint,
            "manifest": man_digest,
            "counters": counters,
            "n_kept": len(kept),
            "pool_sha256": _sha256_file(args.out_dir / "wildchat_fresh_pool.jsonl"),
            "repro": C76.repro_meta(),
        },
    )
    print(f"[phase5-stream] [phase=stream_done] n_kept={len(kept)}", flush=True)
    return 0


# ── 5a.1 capture (pod GPU; reuses the #779 n1m rig verbatim) ─────────────────


def cmd_capture(args) -> int:
    import issue779_ffc_n1m_generate_capture as N1G

    if not args.no_upload:
        assert args.hf_prefix, (
            "--hf-prefix is required unless --no-upload (no implicit issue-prefix "
            f"fallback, #1005; canonical for THIS issue: {DEFAULT_HF_PREFIX})"
        )
    rows = N1G._read_jsonl(args.pool)
    assert rows, f"empty pool {args.pool}"
    rows = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard_index]
    if args.max_rows:
        rows = rows[: args.max_rows]
    scratch: Path = args.out_root
    scratch.mkdir(parents=True, exist_ok=True)
    layers = list(N1G.CAPTURE_LAYERS)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = N1G._build_capture_engine(argparse.Namespace(model=args.model))
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()

    remote: set[str] = set()
    if not args.no_upload:
        remote = set(N1G.N50._remote_index(f"{args.hf_prefix}/final_token_capture"))

    pt_pend: list[str] = []
    raw_pend: list[str] = []
    skipped_all: list[dict] = []
    n_chunks = math.ceil(len(rows) / args.chunk_size)
    t0 = time.time()
    for k, start in enumerate(range(0, len(rows), args.chunk_size)):
        name = f"shard{args.shard_index:02d}_chunk{k:04d}.pt"
        raw_name = name.removesuffix(".pt") + ".json"
        if name in remote or (scratch / name).exists():
            print(f"[phase5-capture] chunk {k + 1}/{n_chunks} already done; skip", flush=True)
            continue
        chunk = rows[start : start + args.chunk_size]
        prompts = [r["prompt"] for r in chunk]
        cis = [int(r["i"]) for r in chunk]
        kept_p, kept_ci, skipped = N1G._filter_overlength_prompts(
            prompts, cis, lambda p: N1G._rendered_prompt_token_len(tok, p), N1G.PROMPT_TOKEN_BUDGET
        )
        skipped_all.extend(skipped)
        n = N1G._capture_stage_chunk(
            hf, tok, llm, kept_p, kept_ci, layers, scratch, name, raw_name, args.shard_index, k, 0
        )
        if n:
            pt_pend.append(name)
            raw_pend.append(raw_name)
        print(
            f"[phase5-capture] chunk {k + 1}/{n_chunks} kept={n} len_skipped={len(skipped)} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if not args.no_upload and len(pt_pend) >= N1G.UPLOAD_BATCH:
            N1G._flush_upload_batch(scratch, args.hf_prefix, pt_pend, raw_pend)
            pt_pend.clear()
            raw_pend.clear()
    side = f"shard{args.shard_index:02d}_skipped.json"
    C76.atomic_write_json(
        scratch / side,
        {
            "shard_index": args.shard_index,
            "prompt_token_budget": N1G.PROMPT_TOKEN_BUDGET,
            "n_skipped": len(skipped_all),
            "skipped": skipped_all,  # ci + token counts only — never text
            "repro": C76.repro_meta(),
        },
    )
    if not args.no_upload:
        N1G._flush_upload_batch(scratch, args.hf_prefix, pt_pend, raw_pend + [side])
    print(f"[phase5-capture] [phase=capture_done] n_chunks={n_chunks}", flush=True)
    return 0


# ── 5a.3 transfer decay read ─────────────────────────────────────────────────


def _r2_boot(pred: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    """Pooled R² + bootstrap CI over held-out rows — BATCHED (subset-sum GEMM:
    per-row reductions once, all draws as one count-matrix product)."""
    pred64 = np.asarray(pred, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    n = y64.shape[0]
    res2 = ((y64 - pred64) ** 2).sum(axis=1)  # (n,)
    ysq = (y64**2).sum(axis=1)  # (n,)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    ss_res = counts @ res2  # (B,)
    sum_y = counts @ y64  # (B, H)
    ss_tot = counts @ ysq - (sum_y**2).sum(axis=1) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = 1.0 - ss_res / ss_tot
    valid = draws[np.isfinite(draws)]
    lo, hi = (
        (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))
        if valid.size
        else (math.nan, math.nan)
    )
    return {
        "r2": _pooled_r2(pred64, y64),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_boot": int(n_boot),
        "n_boot_valid": int(valid.size),
    }


def _stage_leg_chunks_from_hub(leg_dir: Path, hf_prefix: str) -> list[Path]:
    """Hub-stage fallback for a purged capture-leg dir (the #1489
    purge-before-last-consumer class at the p5a_capture -> p5_transfer seam).

    The capture producer uploads each verified chunk batch to
    ``{hf_prefix}/final_token_capture/`` then PURGES the local copies
    (``N1G._flush_upload_batch`` — deliberate pod-disk bounding), so a
    consumer reading the leg dir after capture finds it EMPTY (crash-fix r12,
    epm:failure v8: ``no capture chunks under .../wildchat_fresh``). Re-stage
    from the Hub: PREFIX-SCOPED index only (never a full-tree listing on the
    ~1M-file data repo, #833; ``N50._remote_index`` is the producer's own
    scoped ``list_repo_tree``), per-file ATOMIC + IDEMPOTENT download
    (``hub.stage_hub_file`` — an existing target short-circuits), then
    sha256-verify each staged file against the Hub LFS metadata (the same
    corruption guard the upload side ran before purging). Returns the sorted
    local chunk paths; raises when the prefix holds no chunks.
    """
    import fnmatch

    import issue779_ffc_n1m_generate_capture as N1G

    from explore_persona_space.orchestrate import hub

    cap_prefix = f"{hf_prefix}/final_token_capture"
    remote = hub.retry_transient(
        lambda: N1G.N50._remote_index(cap_prefix), what=f"remote_index({cap_prefix})"
    )
    names = sorted(n for n in remote if fnmatch.fnmatch(n, "shard*_chunk*.pt"))
    assert names, f"Hub-stage fallback found NO capture chunks under {cap_prefix}"
    leg_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[phase5-transfer] leg dir {leg_dir} has no local chunks; staging {len(names)} "
        f"from {cap_prefix} (purge-before-last-consumer fallback, #1489 class)",
        flush=True,
    )
    out: list[Path] = []
    for n in names:
        target = leg_dir / n
        hub.stage_hub_file(C.HF_DATA_REPO, f"{cap_prefix}/{n}", target, repo_type="dataset")
        want = remote[n].get("sha256")
        if want is not None:
            got = _sha256_file(target)
            assert got == want, f"{n}: staged sha256 {got} != Hub LFS {want} — corrupt download"
        out.append(target)
    print(f"[phase5-transfer] staged {len(out)} capture chunks -> {leg_dir}", flush=True)
    return out


def load_leg(path: Path, *, src_layer: int, ro_layer: int, hf_prefix: str | None = None) -> dict:
    """A transfer leg: {x_src (n,H), x_ro (n,H), v (n,H)} float32 numpy.

    Accepts a capture-chunks DIRECTORY (the ``_stack_chunk`` schema:
    shard*_chunk*.pt with cx_last/v_x (n, n_layers, H) + layers) or a single
    ``.pt`` with explicit {x14, x19, v19} tensors. ``hf_prefix`` (threaded via
    the 3-part ``--leg NAME=PATH=HF_PREFIX`` spec) arms the Hub-stage fallback
    for a dir whose chunks the capture producer uploaded-then-purged; existing
    local chunks short-circuit it (idempotent).
    """
    if path.is_dir() or (hf_prefix is not None and not path.exists()):
        xs, xr, vs = [], [], []
        files = sorted(path.glob("shard*_chunk*.pt")) if path.is_dir() else []
        if not files and hf_prefix is not None:
            files = _stage_leg_chunks_from_hub(path, hf_prefix)
        assert files, f"no capture chunks under {path}" + (
            "" if hf_prefix else " (no HF_PREFIX on the --leg spec — Hub fallback unarmed)"
        )
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=True)
            layers = [int(x) for x in d["layers"]]
            li_s, li_r = layers.index(src_layer), layers.index(ro_layer)
            xs.append(d["cx_last"][:, li_s, :].to(torch.float32))
            xr.append(d["cx_last"][:, li_r, :].to(torch.float32))
            vs.append(d["v_x"][:, li_r, :].to(torch.float32))
        return {
            "x_src": torch.cat(xs).numpy(),
            "x_ro": torch.cat(xr).numpy(),
            "v": torch.cat(vs).numpy(),
        }
    d = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "x_src": d["x14"].to(torch.float32).numpy(),
        "x_ro": d["x19"].to(torch.float32).numpy(),
        "v": d["v19"].to(torch.float32).numpy(),
    }


def assemble_test_leg_and_anchors(args) -> tuple[Path, Path]:
    """Build the pinned test-1000 leg + LMSYS-train-pool anchors from the n1m
    assembly (pod path; shares the comparator-fit plumbing verbatim)."""
    import issue779_ffc_n1m_fits as N1M

    ns = argparse.Namespace(
        # N1M contract (N1G._load_pass_b_bundle): pass_b must be a real Path
        # (None -> AttributeError at .exists()). Crash-fix r7: resolve the
        # CLI's default=None to the reused module's own constant.
        pass_b=args.pass_b if args.pass_b is not None else N1M.N1G.PASS_B_LOCAL,
        out_dir=args.assemble_out_dir,
        manifest_from_hf=True,
        manifest_hf_prefix=args.manifest_hf_prefix,
        # N1M contract (issue779_ffc_n1m_fits.py L949-953): ns.hf_prefix is the
        # CAPTURE prefix <round-root>/final_token_capture — the chunk stream reads
        # <hf_prefix>/shardNN_chunkNNNN.pt directly; only manifest_hf_prefix is
        # the round root. Crash-fix r6: same wrong-prefix class as the comparator
        # p0 404 (att-20260729-082617); the capture-prefix fix also keeps the
        # memmap fingerprint aligned with the comparator's stream (mm reuse).
        hf_prefix=f"{args.manifest_hf_prefix}/final_token_capture",
        n1m_capture_dir=None,
        mm_dir=args.mm_dir,
        # N1M contract (N50._pinned_original_shas): orig_dir must be a real dir
        # holding the original round's fair_comparison.json. Crash-fix r7
        # (att-20260729-060640): None crashed `None / "fair_comparison.json"`.
        orig_dir=N1M.DEFAULT_ORIG_DIR,
        fresh_stream=False,
        prefetch=2,
        max_chunks=args.max_chunks,
    )
    layers = [C76.SOURCE_LAYER, C76.READOUT_LAYER]
    per_layer, prov, _orig, val, te, split = N1M.assemble_multilayer(ns, layers)
    x14, _ = per_layer[C76.SOURCE_LAYER]
    x19, v19 = per_layer[C76.READOUT_LAYER]
    te_idx = np.asarray(te)
    held = np.zeros(prov.shape[0], dtype=bool)
    held[np.asarray(val)] = True
    held[te_idx] = True
    tr_mask = ~held & np.asarray([p == "lmsys" for p in prov])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    leg_path = args.out_dir / "leg_lmsys_test1000.pt"
    torch.save(
        {
            "x14": torch.as_tensor(np.asarray(x14[te_idx]), dtype=torch.float32),
            "x19": torch.as_tensor(np.asarray(x19[te_idx]), dtype=torch.float32),
            "v19": torch.as_tensor(np.asarray(v19[te_idx]), dtype=torch.float32),
            "split": split,
        },
        leg_path,
    )
    anchors_path = args.out_dir / "anchors_lmsys_train.pt"
    torch.save(
        {
            "xmu14": torch.as_tensor(np.asarray(x14[tr_mask]).mean(0), dtype=torch.float32),
            "xmu19": torch.as_tensor(np.asarray(x19[tr_mask]).mean(0), dtype=torch.float32),
            "ymu19": torch.as_tensor(np.asarray(v19[tr_mask]).mean(0), dtype=torch.float32),
            "n_rows": int(tr_mask.sum()),
            "source": "n1m train pool, lmsys-only (J's fitting corpus)",
        },
        anchors_path,
    )
    print(
        f"[phase5-transfer] assembled test leg n={len(te_idx)} + anchors n={int(tr_mask.sum())}",
        flush=True,
    )
    return leg_path, anchors_path


def _parse_ops(specs: list[str]) -> list[tuple[str, Path, int]]:
    out = []
    for s in specs or []:
        name, path, layer = s.split("=")
        assert int(layer) in (C76.SOURCE_LAYER, C76.READOUT_LAYER), layer
        out.append((name, Path(path), int(layer)))
    return out


def score_predictions(pred: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> dict:
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    res = _r2_boot(pred, y, n_boot, seed)
    res["knn"] = {
        m: knn_retrieval(pred, y, ks=(1, 5, 10), metric=m) for m in ("euclidean", "cosine")
    }
    res["n"] = int(y.shape[0])
    return res


def cmd_transfer(args) -> int:
    import issue779_ffc_n1m_fits as N1M

    if args.assemble:
        leg_path, anchors_path = assemble_test_leg_and_anchors(args)
        args.legs = [f"lmsys_test1000={leg_path}"] + (args.legs or [])
        args.anchors = args.anchors or anchors_path
    assert args.anchors is not None, "--anchors required (or --assemble)"
    anc = torch.load(args.anchors, map_location="cpu", weights_only=True)
    xmu = {
        C76.SOURCE_LAYER: anc["xmu14"].to(torch.float64).numpy(),
        C76.READOUT_LAYER: anc["xmu19"].to(torch.float64).numpy(),
    }
    ymu19 = anc["ymu19"].to(torch.float64).numpy()

    ops = _parse_ops(args.op)
    jops = [(s.split("=")[0], Path(s.split("=")[1])) for s in (args.jop or [])]
    dev = torch.device("cpu")
    report: dict = {"legs": {}, "anchors": {k: str(args.anchors) for k in ("path",)}}
    for spec in args.legs or []:
        # NAME=PATH[=HF_PREFIX] — the optional 3rd field arms load_leg's
        # Hub-stage fallback for an uploaded-then-purged chunks dir (r12).
        parts = spec.split("=", 2)
        assert len(parts) >= 2, f"--leg spec must be NAME=PATH[=HF_PREFIX]: {spec!r}"
        leg_name, leg_path = parts[0], parts[1]
        leg_prefix = parts[2] if len(parts) == 3 else None
        leg = load_leg(
            Path(leg_path),
            src_layer=C76.SOURCE_LAYER,
            ro_layer=C76.READOUT_LAYER,
            hf_prefix=leg_prefix,
        )
        y = leg["v"].astype(np.float64)
        rows: dict[str, dict] = {}
        for name, path, in_layer in ops:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            x = (leg["x_src"] if in_layer == C76.SOURCE_LAYER else leg["x_ro"]).astype(np.float64)
            pred = N1M.apply_map(payload, x, dev)
            rows[name] = score_predictions(pred, y, n_boot=args.n_boot, seed=args.seed)
            rows[name]["input_layer"] = in_layer
        for name, path in jops:
            obj = torch.load(path, map_location="cpu", weights_only=True)
            j = (obj["J"] if isinstance(obj, dict) else obj).to(torch.float64).numpy()
            x = leg["x_src"].astype(np.float64)
            pred = (x - xmu[C76.SOURCE_LAYER]) @ j.T + ymu19
            rows[name] = score_predictions(pred, y, n_boot=args.n_boot, seed=args.seed)
            rows[name]["input_layer"] = C76.SOURCE_LAYER
            rows[name]["j_affine"] = True
            rows[name]["mismatched_units"] = "last" not in name  # §6: J_ctx/J_prefix on c_last
        for in_layer in sorted({il for _, _, il in ops} | ({C76.SOURCE_LAYER} if jops else set())):
            x = (leg["x_src"] if in_layer == C76.SOURCE_LAYER else leg["x_ro"]).astype(np.float64)
            # Identity + learned bias: same math as the canonical helper
            # analysis/mapping_baselines.identity_bias_predict (v̂ = x + b,
            # b = train-fold mean of y − x), with b computed from the ANCHOR
            # means (n1m train pool) instead of an in-leg train fold — the
            # anchors ARE the train-side means here (review v1 style note).
            pred = x + (ymu19 - xmu[in_layer])
            key = f"identity_bias_l{in_layer}"
            rows[key] = score_predictions(pred, y, n_boot=args.n_boot, seed=args.seed)
            rows[key]["input_layer"] = in_layer
        report["legs"][leg_name] = {"n": int(y.shape[0]), "operators": rows}
        print(f"[phase5-transfer] leg={leg_name} scored {len(rows)} operators", flush=True)

    # H3 decay: per operator, base leg vs each other leg; ratio gated on the
    # base leg clearing identity+bias (§6), else absolute delta only.
    decay: dict[str, dict] = {}
    base = report["legs"].get(args.base_leg)
    if base is not None:
        for other_name, other in report["legs"].items():
            if other_name == args.base_leg:
                continue
            for op_name, r in base["operators"].items():
                if op_name.startswith("identity_bias"):
                    continue
                o = other["operators"].get(op_name)
                if o is None:
                    continue
                ib = base["operators"].get(f"identity_bias_l{r['input_layer']}", {})
                clears_ib = r["r2"] > ib.get("r2", -math.inf)
                entry = {
                    "base_r2": r["r2"],
                    "other_r2": o["r2"],
                    "delta_r2": o["r2"] - r["r2"],
                    "base_clears_identity_bias": bool(clears_ib),
                }
                if clears_ib and r["r2"] > 0:
                    entry["relative_decay"] = o["r2"] / r["r2"]
                decay.setdefault(other_name, {})[op_name] = entry
    report["decay_vs_base"] = {"base_leg": args.base_leg, "rows": decay}

    # Plan §12 assumption 3 (review v1 Minor): the shipped 963k ridge must
    # reproduce its COMMITTED test_r2 on the pinned test-1000 within ±tol.
    # Recorded + WARNed (never a halt — §7 names no such gate); on a miss the
    # shipped-M row is labeled reference-not-validated for the analyzer.
    base = report["legs"].get(args.base_leg, {})
    shipped = (base.get("operators") or {}).get("m_shipped")
    if shipped is not None:
        got = float(shipped["r2"])
        diff = abs(got - C76.SHIPPED_M_TEST_R2_REF)
        ok = diff <= C76.SHIPPED_M_TEST_R2_TOL
        report["shipped_m_reproduction"] = {
            "ref_r2": C76.SHIPPED_M_TEST_R2_REF,
            "ref_source": "eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json"
            " per_point.mixed_1m.predictors.ridge.whole_map_r2",
            "got_r2": got,
            "abs_diff": diff,
            "tol": C76.SHIPPED_M_TEST_R2_TOL,
            "within_tol": bool(ok),
        }
        shipped["reference_validated"] = bool(ok)
        lvl = "PASS" if ok else "WARN: shipped-M row NOT reference-validated"
        print(
            f"[phase5-transfer] shipped-M reproduction {lvl}: got={got:.4f} "
            f"ref={C76.SHIPPED_M_TEST_R2_REF:.4f} (tol {C76.SHIPPED_M_TEST_R2_TOL})",
            flush=True,
        )

    # Plan §3 parity-exclusion domain note (review v1 concern
    # parity-exclusion-list-unconsumed): the G-PARITY rig samples NEW-capture
    # (n1m train-pool) rows only; the pinned test-1000 is the round-1 pass_b
    # head (ci = -1), structurally outside the parity sample, so no exclusion
    # can apply to this leg. The exclusion list IS consumed where its rows
    # live: the P0.4 J-pair builder + the comparator train-row selection.
    report["parity_exclusion"] = {
        "applies_to_test_leg": False,
        "reason": "parity samples new-capture (train-pool) rows; test-1000 is pass_b (ci=-1)",
        "consumers": ["p04 jpairs builder", "comparator select_train_rows"],
    }
    report["repro"] = C76.repro_meta()
    C76.atomic_write_json(args.out, report)
    print(f"[phase5-transfer] [phase=transfer_done] -> {args.out}", flush=True)

    # §6.5 primary deliverable phase2/jvm_heldout.json (review v1 Critical 2):
    # the held-out J-vs-M' read on the pinned test-1000 — the lmsys_test1000
    # block re-emitted at its DECLARED path, with a pointer to the full report.
    if args.jvm_heldout_out is not None:
        leg = report["legs"].get(args.base_leg)
        assert leg is not None, (
            f"--jvm-heldout-out requires the '{args.base_leg}' leg to have been scored"
        )
        C76.atomic_write_json(
            args.jvm_heldout_out,
            {
                "dv": "Held-out reconstruction (J vs M')",
                "leg": args.base_leg,
                "n": leg["n"],
                "operators": leg["operators"],
                "shipped_m_reproduction": report.get("shipped_m_reproduction"),
                "full_report": str(args.out),
                "repro": report["repro"],
            },
        )
        print(f"[phase5-transfer] jvm_heldout -> {args.jvm_heldout_out}", flush=True)
    return 0


# ── 5b leakage re-read (0 GPU) ───────────────────────────────────────────────


def _load_centroid_bank(bank_meta: Path, centroids: Path, *, allow_unpinned: bool):
    """(names, C (P,H) fp32, layer). Sha-pins the bundle against the bank meta
    (``built_from.centroids_sha256``) unless ``allow_unpinned`` (smoke)."""
    meta = json.loads(bank_meta.read_text())
    layer = int(meta["layer"])
    names = list(meta["persona_names"])
    pin = (meta.get("built_from") or {}).get("centroids_sha256")
    if not allow_unpinned:
        got = _sha256_file(centroids)
        assert pin and got == pin, f"centroid bundle sha {got} != bank pin {pin}"
    obj = torch.load(centroids, map_location="cpu", weights_only=True)
    if isinstance(obj, torch.Tensor):
        mat = obj
    elif "centroids" in obj:
        mat = obj["centroids"]
        if "names" in obj or "persona_names" in obj:
            names = list(obj.get("names") or obj.get("persona_names"))
    else:  # dict[str -> (H,)]
        names = sorted(obj)
        mat = torch.stack([obj[n] for n in names])
    mat = mat.to(torch.float32)
    assert mat.ndim == 2 and mat.shape[0] == len(names), (mat.shape, len(names))
    return names, mat, layer


def map_labels_to_pool(labels: list[str], pool_names: list[str]) -> dict[str, dict]:
    """#532 label -> canonical-pool persona by EXACT system-prompt text match.

    Only #406 cids with a nonempty persona system prompt are mappable (B/C/D
    classes + the instructed panel have no persona representation in the bank).
    """
    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

    pool = json.loads((POOL_DIR / "pool_v1.json").read_text())["personas"]
    by_prompt: dict[str, list[str]] = {}
    for k, v in pool.items():
        p = (v.get("prompt") or "").strip()
        if p and k in pool_names:
            by_prompt.setdefault(p, []).append(k)
    out: dict[str, dict] = {}
    for lab in labels:
        cond = CONDITIONS_BY_ID.get(lab)
        sp = (getattr(cond, "system_prompt", None) or "").strip() if cond else ""
        matches = sorted(by_prompt.get(sp, [])) if sp else []
        if matches:
            out[lab] = {"persona": matches[0], "all_matches": matches}
    return out


_CELL_RE = re.compile(r"^cell_(?P<rest>.+)\.json$")


def aggregate_leakage(per_cell_root: Path) -> list[dict]:
    """Per (arm_ep, source, bystander) on-policy emission rates from the #532
    per-cell tables (round-3 PRIMARY DV: in-R emit anywhere; at-end secondary)."""
    rows: list[dict] = []
    for arm_dir in sorted(p for p in per_cell_root.iterdir() if p.is_dir()):
        prefix = f"cell_{arm_dir.name}_"
        for f in sorted(arm_dir.glob("cell_*.json")):
            assert f.name.startswith(prefix) and "__" in f.name, f.name
            src, byst = f.name.removeprefix(prefix).removesuffix(".json").split("__", 1)
            cell = json.loads(f.read_text())
            any_q = cell.get("in_R_emit_anywhere_per_q") or cell.get("in_R_emission_per_q")
            end_q = cell.get("in_R_emit_at_end_per_q")
            assert any_q is not None, f"{f}: no emission field"
            rows.append(
                {
                    "arm_ep": arm_dir.name,
                    "source": src,
                    "bystander": byst,
                    "rate_any": float(np.mean([float(v) for v in any_q])),
                    "rate_end": (
                        float(np.mean([float(v) for v in end_q])) if end_q is not None else None
                    ),
                    "n_q": len(any_q),
                }
            )
    assert rows, f"no per-cell tables under {per_cell_root}"
    return rows


def _spearman_boot(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    """Spearman rho + bootstrap CI over pairs — BATCHED (rankdata axis=1)."""
    from scipy.stats import rankdata, spearmanr

    n = x.shape[0]
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    xs, ys = x[idx], y[idx]
    rx = rankdata(xs, axis=1)
    ry = rankdata(ys, axis=1)
    rxc = rx - rx.mean(axis=1, keepdims=True)
    ryc = ry - ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rxc**2).sum(axis=1) * (ryc**2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = (rxc * ryc).sum(axis=1) / denom
    valid = draws[np.isfinite(draws)]
    return {
        "rho": rho,
        "ci_lo": float(np.percentile(valid, 2.5)) if valid.size else math.nan,
        "ci_hi": float(np.percentile(valid, 97.5)) if valid.size else math.nan,
        "n_pairs": int(n),
        "n_boot_valid": int(valid.size),
    }


def cmd_leakage(args) -> int:
    import issue1776_phase4 as P4

    predictors = json.loads(args.predictors.read_text())
    names, cents, bank_layer = _load_centroid_bank(
        args.bank_meta, args.centroids, allow_unpinned=args.allow_unpinned_centroids
    )
    d = P4.load_dict(args.dict, "cpu")
    assert int(d["layer"]) == bank_layer, (
        f"dictionary layer {d['layer']} != centroid-bank layer {bank_layer} — "
        "P_J must be built at the bank's own layer (plan §4 5b)"
    )
    basis, rank, _spec, _w = P4.dict_projector(
        d, pj_energy=args.pj_energy, pj_rank=args.pj_rank, device="cpu"
    )

    labels = sorted(set(predictors["sources"]) | set(predictors["bystanders"]))
    mapping = map_labels_to_pool(labels, names)
    name_idx = {n: i for i, n in enumerate(names)}

    # Full-bank global-mean centering (canonical recipe), then the three legs.
    cc = cents - cents.mean(dim=0, keepdim=True)
    z = cc @ basis  # P_J coordinates
    perp = cc - z @ basis.T

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(torch.finfo(torch.float32).tiny)
        )

    leak = aggregate_leakage(args.per_cell_root)
    src_i = {s: i for i, s in enumerate(predictors["sources"])}
    byst_i = {b: i for i, b in enumerate(predictors["bystanders"])}
    cos_ref = np.asarray(predictors["cosine_matrix"], dtype=np.float64)

    per_arm: dict[str, dict] = {}
    for arm_ep in sorted({r["arm_ep"] for r in leak}):
        rows = [
            r
            for r in leak
            if r["arm_ep"] == arm_ep
            and r["source"] != r["bystander"]
            and r["source"] in mapping
            and r["bystander"] in mapping
        ]
        n_all = sum(r["arm_ep"] == arm_ep for r in leak)
        if len(rows) < 3:
            per_arm[arm_ep] = {"n_covered_pairs": len(rows), "n_all_pairs": n_all, "skipped": True}
            continue
        rate = np.array([r["rate_any"] for r in rows])
        sims: dict[str, np.ndarray] = {"raw": [], "pj": [], "perp": [], "predictors_cosine": []}
        for r in rows:
            a = name_idx[mapping[r["source"]]["persona"]]
            b = name_idx[mapping[r["bystander"]]["persona"]]
            sims["raw"].append(_cos(cc[a], cc[b]))
            sims["pj"].append(_cos(z[a], z[b]))
            sims["perp"].append(_cos(perp[a], perp[b]))
            sims["predictors_cosine"].append(
                cos_ref[src_i[r["source"]], byst_i[r["bystander"]]]
                if r["source"] in src_i and r["bystander"] in byst_i
                else math.nan
            )
        arm_out: dict = {"n_covered_pairs": len(rows), "n_all_pairs": n_all}
        for variant, vals in sims.items():
            v = np.asarray(vals, dtype=np.float64)
            keep = np.isfinite(v)
            arm_out[variant] = _spearman_boot(v[keep], rate[keep], args.n_boot, args.seed)
        per_arm[arm_ep] = arm_out
        print(
            f"[phase5-leakage] {arm_ep}: covered={len(rows)}/{n_all} "
            + " ".join(f"{k}_rho={per_arm[arm_ep][k]['rho']:.3f}" for k in sims),
            flush=True,
        )

    C76.atomic_write_json(
        args.out,
        {
            "bank_layer": bank_layer,
            "pj_rank": int(rank),
            "pj_energy": args.pj_energy,
            "label_mapping": mapping,
            "n_labels": len(labels),
            "n_mapped": len(mapping),
            "dv": "in-R emit-anywhere rate (round-3 PRIMARY, #532)",
            "per_arm": per_arm,
            "repro": C76.repro_meta(),
        },
    )
    print(f"[phase5-leakage] [phase=leakage_done] -> {args.out}", flush=True)
    return 0


# ── 5c lens-vocab word tables ────────────────────────────────────────────────


def decode_direction(d: dict, vec: torch.Tensor, topk: int, tok) -> dict:
    """Top ±k vocab entries of a direction under the lens dictionary read
    (raw-logit rank = (rows_unit @ x) * row_norms — the dict's convention)."""
    v = vec.to(torch.float32)
    v = v / v.norm().clamp_min(torch.finfo(torch.float32).tiny)
    scores = (d["rows_unit"].to(torch.float32) @ v) * d["row_norms"]

    def _side(vals, ids):
        toks = tok.convert_ids_to_tokens([int(i) for i in ids])
        return [{"id": int(i), "token": t, "score": float(s)} for i, t, s in zip(ids, toks, vals)]

    top = torch.topk(scores, topk)
    bot = torch.topk(-scores, topk)
    return {"pos": _side(top.values, top.indices), "neg": _side(-bot.values, bot.indices)}


def _op_svd(payload_path: Path, topk: int) -> dict:
    """Top-k singular triples of the RAW-space operator A = W / xsd[:, None]."""
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    a = payload["W"].to(torch.float64) / payload["xsd"].to(torch.float64)[:, None]
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    k = min(topk, s.shape[0])
    return {
        "input_dirs": u[:, :k].T.to(torch.float32),  # (k, H_in)
        "output_dirs": vh[:k].to(torch.float32),  # (k, H_out)
        "sigma": [float(x) for x in s[:k]],
    }


def cmd_lens_vocab(args) -> int:
    import issue1776_phase4 as P4

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    d_in = P4.load_dict(args.dict_in, "cpu")
    d_out = d_in if args.dict_out == args.dict_in else P4.load_dict(args.dict_out, "cpu")
    tables: dict = {"dict_in_layer": int(d_in["layer"]), "dict_out_layer": int(d_out["layer"])}

    if args.mprime is not None:
        svd = _op_svd(args.mprime, args.topk_dirs)
        tables["mprime"] = {
            "sigma": svd["sigma"],
            "input_dirs": [
                decode_direction(d_in, v, args.topk_words, tok) for v in svd["input_dirs"]
            ],
            "output_dirs": [
                decode_direction(d_out, v, args.topk_words, tok) for v in svd["output_dirs"]
            ],
        }
    if args.shipped is not None:
        svd = _op_svd(args.shipped, args.topk_dirs)
        tables["shipped_m"] = {
            "sigma": svd["sigma"],
            "input_dirs": [
                decode_direction(d_out, v, args.topk_words, tok) for v in svd["input_dirs"]
            ],
            "output_dirs": [
                decode_direction(d_out, v, args.topk_words, tok) for v in svd["output_dirs"]
            ],
            "note": "shipped M is same-layer (19->19): both spaces decode via dict_out",
        }
    if args.rb_dir is not None:
        tables["r_b"] = {}
        for f in sorted(Path(args.rb_dir).glob("*.pt")):
            obj = torch.load(f, map_location="cpu", weights_only=True)
            layers = [int(x) for x in obj["layers"]]
            entry = {}
            for tag, dd in (("in", d_in), ("out", d_out)):
                li = int(dd["layer"])
                if li in layers:
                    entry[f"L{li}_{tag}"] = decode_direction(
                        dd, obj["r_b"][layers.index(li)], args.topk_words, tok
                    )
            tables["r_b"][f.stem] = entry
    tables["repro"] = C76.repro_meta()
    C76.atomic_write_json(args.out, tables)
    print(f"[phase5-lens] [phase=lens_vocab_done] -> {args.out}", flush=True)
    return 0


# ── 5d chain-composition judge-free DV ───────────────────────────────────────


def load_chain_rows(chunks_dir: Path, in_layer: int, max_ctx: int) -> tuple[np.ndarray, list[str]]:
    """(x (n, H) at ``in_layer``, responses) joined from capture chunk pairs
    (.pt tensors + raw .json rows) on (chunk, ci)."""
    xs: list[torch.Tensor] = []
    resp: list[str] = []
    files = sorted(chunks_dir.glob("shard*_chunk*.pt"))
    assert files, f"no capture chunks under {chunks_dir}"
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        layers = [int(x) for x in d["layers"]]
        li = layers.index(in_layer)
        raw = json.loads((f.parent / (f.name.removesuffix(".pt") + ".json")).read_text())
        by_ci = {int(r["ci"]): r["response"] for r in raw["rows"]}
        for row_i, ci in enumerate(int(c) for c in d["ci"]):
            assert ci in by_ci, (f.name, ci)
            xs.append(d["cx_last"][row_i, li, :].to(torch.float32))
            resp.append(by_ci[ci])
        if max_ctx and len(resp) >= max_ctx:
            break
    if max_ctx:
        xs, resp = xs[:max_ctx], resp[:max_ctx]
    return torch.stack(xs).numpy(), resp


def content_token_ids(tok, responses: list[str], df_cap: float) -> list[np.ndarray]:
    """Per-response unique CONTENT token ids: drop special ids, tokens with no
    alphanumeric char, and tokens in > ``df_cap`` of responses (doc frequency)."""
    special = set(tok.all_special_ids)
    per: list[set[int]] = []
    for r in responses:
        ids = set(tok(r, add_special_tokens=False)["input_ids"])
        per.append(ids - special)
    df: dict[int, int] = {}
    for ids in per:
        for t in ids:
            df[t] = df.get(t, 0) + 1
    cap = df_cap * len(responses)
    alnum_cache: dict[int, bool] = {}

    def _is_content(t: int) -> bool:
        if t not in alnum_cache:
            alnum_cache[t] = any(ch.isalnum() for ch in tok.decode([t]))
        return alnum_cache[t] and df[t] <= cap

    return [np.array(sorted(t for t in ids if _is_content(t)), dtype=np.int64) for ids in per]


def _chain_metrics(ranks_sub: np.ndarray, sub_ids: list[np.ndarray], perm: np.ndarray, topk: int):
    """Mean MRR (best-ranked content token) + recall@k under pairing ``perm``
    (v̂ row perm[i] scored against context i's content ids)."""
    mrr, rec, n_used = 0.0, 0.0, 0
    for i, ids in enumerate(sub_ids):
        if ids.size == 0:
            continue
        r = ranks_sub[perm[i], ids]
        mrr += 1.0 / float(r.min())
        rec += float((r <= topk).mean())
        n_used += 1
    assert n_used, "no contexts with content tokens"
    return mrr / n_used, rec / n_used, n_used


def cmd_chain(args) -> int:
    import issue1776_phase4 as P4
    import issue779_ffc_n1m_fits as N1M

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    name, path, in_layer = _parse_ops([args.op])[0]
    x, responses = load_chain_rows(args.chunks_dir, in_layer, args.max_ctx)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vhat = N1M.apply_map(payload, x.astype(np.float64), torch.device("cpu")).astype(np.float32)

    d = P4.load_dict(args.dict_out, "cpu")
    rows = d["rows_unit"].to(torch.float32)
    norms = d["row_norms"].to(torch.float32)
    v_size = rows.shape[0]

    content = content_token_ids(tok, responses, args.df_cap)
    union = np.unique(np.concatenate([c for c in content if c.size] or [np.array([0])]))
    uidx = {int(t): k for k, t in enumerate(union)}
    sub_ids = [np.array([uidx[int(t)] for t in c], dtype=np.int64) for c in content]

    n = vhat.shape[0]
    ranks_sub = np.empty((n, union.size), dtype=np.int32)
    for start in range(0, n, args.score_batch):
        vb = torch.from_numpy(vhat[start : start + args.score_batch])
        s = ((vb @ rows.T) * norms).numpy()  # (b, V) raw-logit rank scores
        order = np.argsort(-s, axis=1)
        rank_of = np.empty_like(order)
        np.put_along_axis(
            rank_of, order, np.broadcast_to(np.arange(v_size), order.shape).copy(), axis=1
        )
        ranks_sub[start : start + vb.shape[0]] = rank_of[:, union] + 1
        print(f"[phase5-chain] scored {min(start + vb.shape[0], n)}/{n}", flush=True)

    ident = np.arange(n)
    mrr, rec, n_used = _chain_metrics(ranks_sub, sub_ids, ident, args.topk)
    rng = np.random.default_rng(args.seed)
    null_mrr, null_rec = [], []
    for _ in range(args.n_null):
        perm = rng.permutation(n)
        m, r0, _ = _chain_metrics(ranks_sub, sub_ids, perm, args.topk)
        null_mrr.append(m)
        null_rec.append(r0)
    out = {
        "variant": name,
        "input_layer": in_layer,
        "n_ctx": int(n),
        "n_ctx_used": int(n_used),
        "df_cap": args.df_cap,
        "topk": args.topk,
        "mrr": mrr,
        "recall_at_k": rec,
        "null": {
            "n_draws": args.n_null,
            "mrr_mean": float(np.mean(null_mrr)),
            "mrr_p975": float(np.percentile(null_mrr, 97.5)),
            "recall_mean": float(np.mean(null_rec)),
            "recall_p975": float(np.percentile(null_rec, 97.5)),
        },
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out, out)
    print(
        f"[phase5-chain] [phase=chain_done variant={name}] mrr={mrr:.4f} "
        f"(null {out['null']['mrr_mean']:.4f}) recall@{args.topk}={rec:.4f} "
        f"(null {out['null']['recall_mean']:.4f}) -> {args.out}",
        flush=True,
    )
    return 0


# ── smoke ─────────────────────────────────────────────────────────────────────


def _smoke_stream(out: Path) -> None:
    """Tiny-REAL WildChat probe: filters + counters + resume + exclusion branch."""
    base = argparse.Namespace(
        n_keep=2, max_scan=4000, checkpoint_every=1, out_dir=out / "stream_p1"
    )
    rev = _dataset_revision(WILDCHAT_REPO)
    fp1 = {"repo": WILDCHAT_REPO, "revision": rev, "recipe": STREAM_RECIPE, "probe": "p1"}
    kept, counters = stream_fresh_pool(base, set(), 0, fp1)
    assert len(kept) == 2 and all(r["prompt"] for r in kept), (len(kept), counters)
    # resume branch: same fingerprint -> complete-cache return, no re-stream.
    kept2, _ = stream_fresh_pool(base, set(), 0, fp1, stream_iter=[])
    assert [r["prompt"] for r in kept2] == [r["prompt"] for r in kept]
    # exclusion branch on REAL rows: excluding p1's prompts keeps the NEXT 2.
    base2 = argparse.Namespace(
        n_keep=2, max_scan=4000, checkpoint_every=1, out_dir=out / "stream_p2"
    )
    excl = {r["prompt"] for r in kept}
    kept3, c3 = stream_fresh_pool(base2, excl, 0, {**fp1, "probe": "p2", "exclude_sha": "p1-pool"})
    assert c3["rej_excluded"] >= 2, c3
    assert not ({r["prompt"] for r in kept3} & excl)
    print("[smoke] stream: PASS (kept>0, counters printed, resume + exclusion exercised)")


def _synth_ridge_payload(rng, h_in: int, h_out: int, w: np.ndarray | None = None) -> dict:
    w = rng.standard_normal((h_in, h_out)) / np.sqrt(h_in) if w is None else w
    return {
        "kind": "ridge",
        "selected_lambda": 1.0,
        "xmu": torch.zeros(h_in),
        "xsd": torch.ones(h_in),
        "ymu": torch.zeros(h_out),
        "W": torch.as_tensor(w, dtype=torch.float32),
    }


def _smoke_transfer(out: Path) -> None:
    """Planted linear map round-trip: op + J-affine recover it; decay block."""
    rng = np.random.default_rng(0)
    h, n = 32, 400
    w = rng.standard_normal((h, h)) / np.sqrt(h)
    xa = rng.standard_normal((n, h)).astype(np.float32)
    xb = (rng.standard_normal((n, h)) + 1.5).astype(np.float32)  # shifted leg
    ya = (xa @ w + 0.05 * rng.standard_normal((n, h))).astype(np.float32)
    yb = (xb @ w + 0.5 * rng.standard_normal((n, h))).astype(np.float32)
    (out / "transfer").mkdir(parents=True, exist_ok=True)
    torch.save(
        {"x14": torch.from_numpy(xa), "x19": torch.from_numpy(xa), "v19": torch.from_numpy(ya)},
        out / "transfer" / "leg_lmsys_test1000.pt",
    )
    # wildchat_fresh leg as a capture-chunks DIR + 3-part spec (crash-fix r12):
    # exercises load_leg's dir branch + the NAME=PATH=HF_PREFIX parse; the
    # planted local chunk short-circuits the Hub-stage fallback (no network).
    wc_dir = out / "transfer" / "wc_chunks"
    wc_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "cx_last": torch.from_numpy(np.stack([xb, xb], axis=1)),  # (n, 2 layers, h)
            "v_x": torch.from_numpy(np.stack([yb, yb], axis=1)),
            "layers": [C76.SOURCE_LAYER, C76.READOUT_LAYER],
        },
        wc_dir / "shard00_chunk0000.pt",
    )
    op_path = out / "transfer" / "op.pt"
    torch.save(_synth_ridge_payload(rng, h, h, w), op_path)
    j_path = out / "transfer" / "jca_last.pt"
    torch.save({"J": torch.as_tensor(w.T, dtype=torch.float32)}, j_path)
    anchors = out / "transfer" / "anchors.pt"
    torch.save(
        {
            "xmu14": torch.zeros(h),
            "xmu19": torch.zeros(h),
            "ymu19": torch.zeros(h),
            "n_rows": n,
            "source": "smoke",
        },
        anchors,
    )
    rep_path = out / "transfer" / "transfer.json"
    jvm_path = out / "transfer" / "jvm_heldout.json"
    ns = argparse.Namespace(
        assemble=False,
        anchors=anchors,
        # m_shipped alias exercises the assumption-3 reproduction branch: the
        # planted map's r2 (>0.9) sits OUTSIDE ref±tol, so the smoke drives the
        # not-validated WARN path (degenerate-gate coverage, review v1 Minor).
        op=[f"m_smoke={op_path}=14", f"m_shipped={op_path}=19"],
        jop=[f"jca_last={j_path}"],
        legs=[
            f"lmsys_test1000={out / 'transfer' / 'leg_lmsys_test1000.pt'}",
            f"wildchat_fresh={wc_dir}={DEFAULT_HF_PREFIX}",
        ],
        base_leg="lmsys_test1000",
        n_boot=200,
        seed=0,
        out=rep_path,
        jvm_heldout_out=jvm_path,
    )
    assert cmd_transfer(ns) == 0
    rep = json.loads(rep_path.read_text())
    a = rep["legs"]["lmsys_test1000"]["operators"]
    assert a["m_smoke"]["r2"] > 0.9 and a["jca_last"]["r2"] > 0.9, a
    assert a["m_smoke"]["ci_lo"] <= a["m_smoke"]["r2"] <= a["m_smoke"]["ci_hi"]
    assert a["m_smoke"]["knn"]["euclidean"]["acc_at_k"]["1"] > 0.9, a["m_smoke"]["knn"]
    dec = rep["decay_vs_base"]["rows"]["wildchat_fresh"]["m_smoke"]
    assert "relative_decay" in dec and dec["base_clears_identity_bias"], dec
    srp = rep["shipped_m_reproduction"]
    assert srp["within_tol"] is False and a["m_shipped"]["reference_validated"] is False, srp
    jvm = json.loads(jvm_path.read_text())
    assert jvm["leg"] == "lmsys_test1000" and "m_shipped" in jvm["operators"], jvm.keys()
    assert jvm["shipped_m_reproduction"]["within_tol"] is False
    print(
        "[smoke] transfer: PASS (planted map recovered; CI + kNN + decay + jvm_heldout "
        "emitted; shipped-M reproduction WARN branch exercised)"
    )


def _smoke_leakage(out: Path) -> None:
    """5b re-read against the REAL committed #532 JSONs; synthetic centroids
    (H=32, keyed to REAL pool persona names) + synthetic unit-row dict @ L21."""
    rng = np.random.default_rng(1)
    pool_names = sorted(json.loads((POOL_DIR / "pool_v1.json").read_text())["personas"])
    h = 32
    cents = torch.as_tensor(rng.standard_normal((len(pool_names), h)), dtype=torch.float32)
    (out / "leak").mkdir(parents=True, exist_ok=True)
    cent_path = out / "leak" / "centroids.pt"
    torch.save({"centroids": cents, "names": pool_names}, cent_path)
    meta_path = out / "leak" / "bank_meta.json"
    meta_path.write_text(json.dumps({"layer": 21, "persona_names": pool_names}))
    rows = torch.as_tensor(rng.standard_normal((300, h)), dtype=torch.float32)
    rows = rows / rows.norm(dim=1, keepdim=True)
    dict_path = out / "leak" / "dict21.pt"
    torch.save(
        {"rows_unit": rows.to(torch.float16), "row_norms": torch.ones(300), "layer": 21},
        dict_path,
    )
    rep = out / "leak" / "leakage_reread.json"
    ev = C76.PROJECT_ROOT / "eval_results" / "issue_532"
    ns = argparse.Namespace(
        predictors=ev / "predictors.json",
        per_cell_root=ev / "per_cell",
        bank_meta=meta_path,
        centroids=cent_path,
        dict=dict_path,
        pj_energy=0.95,
        pj_rank=8,
        allow_unpinned_centroids=True,
        n_boot=200,
        seed=0,
        out=rep,
    )
    assert cmd_leakage(ns) == 0
    r = json.loads(rep.read_text())
    assert r["n_mapped"] >= 5, r["label_mapping"]  # A1..A5 map by exact prompt text
    arm = next(v for v in r["per_arm"].values() if not v.get("skipped"))
    for variant in ("raw", "pj", "perp", "predictors_cosine"):
        assert math.isfinite(arm[variant]["rho"]), (variant, arm)
    print(f"[smoke] leakage: PASS (mapped={r['n_mapped']}, arms={sorted(r['per_arm'])})")


def _smoke_lens_chain(out: Path) -> None:
    """5c + 5d against a REAL tiny fitted lens dictionary (real Qwen vocab)."""
    import issue1776_jlens_fit as JF
    import issue1776_phase4 as P4

    lens_dir = out / "lens"
    lens_dir.mkdir(parents=True, exist_ok=True)
    prompts = lens_dir / "prompts.jsonl"
    texts = [
        "The capital of France is Paris, a city on the Seine known for museums, "
        "bridges, cafes and a long history of art, science and architecture.",
        "Water boils at one hundred degrees Celsius at sea level, and the boiling "
        "point drops as altitude increases because atmospheric pressure falls.",
    ]
    prompts.write_text(
        "\n".join(json.dumps({"i": i, "text": t}) for i, t in enumerate(texts)) + "\n"
    )
    lens_path = lens_dir / "tiny_lens.pt"
    assert (
        JF.main(
            [
                "fit",
                "--tiny",
                "--prompts",
                str(prompts),
                "--out",
                str(lens_path),
                "--layers",
                "2",
                "--dim-batch",
                "16",
                "--max-seq-len",
                "32",
                "--skip-first",
                "2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    dict_path = lens_dir / "dict2.pt"
    # NOTE: call cmd_build_dict DIRECTLY — P4.main() ends in an unconditional
    # sys.exit(rc), which would silently terminate this in-process smoke rc=0.
    assert (
        P4.cmd_build_dict(
            argparse.Namespace(
                lens=lens_path,
                model=C.DEFAULT_MODEL,
                layer=2,
                out=dict_path,
                device="cpu",
                tiny=True,
            )
        )
        == 0
    )
    d = P4.load_dict(dict_path, "cpu")
    h = d["rows_unit"].shape[1]
    rng = np.random.default_rng(2)

    # 5c word tables: synthetic M' payload + synthetic r_B stack at the tiny H.
    mp_path = lens_dir / "mprime.pt"
    torch.save(_synth_ridge_payload(rng, h, h), mp_path)
    rb_dir = lens_dir / "rb"
    rb_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "r_b": torch.as_tensor(rng.standard_normal((2, h)), dtype=torch.float32),
            "layers": [1, 2],
        },
        rb_dir / "evil.pt",
    )
    vocab_out = lens_dir / "lens_vocab.json"
    ns = argparse.Namespace(
        dict_in=dict_path,
        dict_out=dict_path,
        mprime=mp_path,
        shipped=None,
        rb_dir=rb_dir,
        topk_dirs=2,
        topk_words=5,
        model=C.DEFAULT_MODEL,
        out=vocab_out,
    )
    assert cmd_lens_vocab(ns) == 0
    tab = json.loads(vocab_out.read_text())
    assert len(tab["mprime"]["input_dirs"]) == 2
    assert all(isinstance(e["token"], str) for e in tab["mprime"]["output_dirs"][0]["pos"])
    assert "L2_in" in tab["r_b"]["evil"]

    # 5d chain: planted v̂ -> each context's own response token ranks ~1st.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL)
    responses = ["giraffe savanna", "photosynthesis chlorophyll", "volcano magma", "glacier ice"]
    tids = []
    for r in responses:
        ids = [t for t in tok(r, add_special_tokens=False)["input_ids"]]
        tids.append(ids[0])
    n = len(responses)
    x = rng.standard_normal((n, h)).astype(np.float64)
    targets = np.stack(
        [(d["rows_unit"][t].to(torch.float32) * float(d["row_norms"][t])).numpy() for t in tids]
    ).astype(np.float64)
    w_fit, *_ = np.linalg.lstsq(x, targets, rcond=None)  # exact: n <= h
    chain_dir = lens_dir / "chunks"
    chain_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "cx_last": torch.as_tensor(x, dtype=torch.float32).unsqueeze(1),  # (n, 1, H)
            "v_x": torch.zeros((n, 1, h)),
            "ci": list(range(n)),
            "prompts": ["p"] * n,
            "layers": [2],
            "shard_index": 0,
            "chunk": 0,
        },
        chain_dir / "shard00_chunk0000.pt",
    )
    (chain_dir / "shard00_chunk0000.json").write_text(
        json.dumps(
            {
                "shard_index": 0,
                "chunk": 0,
                "rows": [{"ci": i, "prompt": "p", "response": r} for i, r in enumerate(responses)],
            }
        )
    )
    op_path = lens_dir / "chain_op.pt"
    torch.save(_synth_ridge_payload(rng, h, h, w_fit), op_path)
    chain_out = lens_dir / "chain.json"
    ns = argparse.Namespace(
        chunks_dir=chain_dir,
        dict_out=dict_path,
        op=f"mprime={op_path}=14",
        model=C.DEFAULT_MODEL,
        max_ctx=0,
        df_cap=0.9,
        topk=50,
        n_null=50,
        seed=0,
        score_batch=2,
        out=chain_out,
    )
    # probe the loader join at the chunk's REAL layer label first ...
    x_loaded, resp_loaded = load_chain_rows(chain_dir, 2, 0)
    assert x_loaded.shape == (n, h) and resp_loaded == responses
    # ... then relabel the tiny chunk's layer slot 2 -> 14 so cmd_chain runs
    # its DEFAULT production path (op INLAYER=14) end-to-end unchanged.
    d2 = torch.load(chain_dir / "shard00_chunk0000.pt", weights_only=True)
    d2["layers"] = [14]
    torch.save(d2, chain_dir / "shard00_chunk0000.pt")
    assert cmd_chain(ns) == 0
    ch = json.loads(chain_out.read_text())
    assert ch["mrr"] > 0.9, ch  # planted token ranks ~1st for its own context
    assert ch["mrr"] > ch["null"]["mrr_mean"] + 0.2, ch
    print("[smoke] lens+chain: PASS (word tables on real vocab; planted chain MRR >> null)")


def _smoke_capture_binds() -> None:
    """Signature-bind the GPU-only capture calls (deferred-import + arity)."""
    import inspect

    import issue779_ffc_n1m_generate_capture as N1G

    inspect.signature(N1G._capture_stage_chunk).bind(
        *[object()] * 8, object(), object(), object(), object()
    )
    inspect.signature(N1G._flush_upload_batch).bind(object(), object(), object(), object())
    inspect.signature(N1G._build_capture_engine).bind(object())
    inspect.signature(N1G._filter_overlength_prompts).bind(object(), object(), object(), object())
    inspect.signature(N1G._rendered_prompt_token_len).bind(object(), object())
    inspect.signature(N1G.N50._remote_index).bind(object())
    inspect.signature(N1G.N10._generate).bind(object(), object(), object())
    print("[smoke] capture-binds: PASS (7 GPU-path call shapes bind)")


def smoke(args) -> int:
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    legs = args.legs or ["stream", "transfer", "leakage", "lens", "binds"]
    if "binds" in legs:
        _smoke_capture_binds()
    if "transfer" in legs:
        _smoke_transfer(out)
    if "leakage" in legs:
        _smoke_leakage(out)
    if "lens" in legs:
        _smoke_lens_chain(out)
    if "stream" in legs:
        _smoke_stream(out)
    print(f"[phase5] [phase=smoke_done legs={','.join(legs)}]", flush=True)
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stream", help="5a.1 fresh WildChat prompt pool")
    s.add_argument("--out-dir", type=Path, default=C76.DATA_DIR / "wildchat_fresh")
    s.add_argument("--n-keep", type=int, default=1000)
    s.add_argument("--max-scan", type=int, default=600_000)
    s.add_argument("--checkpoint-every", type=int, default=500)
    s.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    s.add_argument("--no-manifest", action="store_true", help="smoke: empty exclusion set")
    s.add_argument("--exclude-file", type=Path, default=None, help="extra prompt JSONL to exclude")
    s.add_argument("--start-mode", choices=("after-manifest", "head"), default="after-manifest")
    s.add_argument("--allow-short", action="store_true")

    c = sub.add_parser("capture", help="5a.1 gen + teacher-forced capture (pod GPU)")
    c.add_argument("--pool", type=Path, required=True)
    c.add_argument("--out-root", type=Path, required=True)
    c.add_argument("--model", default=C.DEFAULT_MODEL)
    c.add_argument("--chunk-size", type=int, default=128)
    c.add_argument("--shard-index", type=int, default=0)
    c.add_argument("--n-shards", type=int, default=1)
    c.add_argument("--max-rows", type=int, default=0)
    c.add_argument(
        "--hf-prefix",
        default=None,
        help=f"REQUIRED unless --no-upload (canonical: {DEFAULT_HF_PREFIX}); no implicit "
        "issue-prefix fallback — a child issue reusing this script passes its own (#1005)",
    )
    c.add_argument("--no-upload", action="store_true")

    t = sub.add_parser("transfer", help="5a.3 decay read over legs")
    t.add_argument("--op", action="append", help="NAME=PATH=INLAYER (ridge payload)", default=[])
    t.add_argument("--jop", action="append", help="NAME=PATH (J .pt, affine arm)", default=[])
    t.add_argument(
        "--leg",
        dest="legs",
        action="append",
        help="NAME=PATH[=HF_PREFIX] (.pt or chunks dir; HF_PREFIX arms the Hub-stage "
        "fallback when the capture producer uploaded-then-purged the dir's chunks)",
    )
    t.add_argument("--anchors", type=Path, default=None)
    t.add_argument("--base-leg", default="lmsys_test1000")
    t.add_argument("--n-boot", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--assemble", action="store_true", help="build test leg + anchors (pod)")
    t.add_argument("--pass-b", type=Path, default=None)
    t.add_argument("--assemble-out-dir", type=Path, default=C76.DATA_DIR / "ffc_n1m")
    t.add_argument("--manifest-hf-prefix", default="issue779_monitoring/fitter-fair-comparison-n1m")
    t.add_argument("--mm-dir", type=Path, default=C76.DATA_DIR / "n1m_mm")
    t.add_argument("--max-chunks", type=int, default=None)
    t.add_argument("--out-dir", type=Path, default=C76.DATA_DIR / "transfer")
    t.add_argument(
        "--jvm-heldout-out",
        type=Path,
        default=None,
        help="also emit the §6.5 phase2/jvm_heldout.json (base-leg operators block)",
    )

    lk = sub.add_parser("leakage", help="5b retrospective leakage re-read (0 GPU)")
    lk.add_argument(
        "--predictors",
        type=Path,
        default=C76.PROJECT_ROOT / "eval_results/issue_532/predictors.json",
    )
    lk.add_argument(
        "--per-cell-root",
        type=Path,
        default=C76.PROJECT_ROOT / "eval_results/issue_532/per_cell",
    )
    lk.add_argument("--bank-meta", type=Path, default=POOL_DIR / "matrix_v1_L21_raw.json")
    lk.add_argument("--centroids", type=Path, required=True)
    lk.add_argument("--dict", type=Path, required=True, help="dictionary at the bank's layer")
    lk.add_argument("--pj-energy", type=float, default=0.95)
    lk.add_argument("--pj-rank", type=int, default=None)
    lk.add_argument("--allow-unpinned-centroids", action="store_true")
    lk.add_argument("--n-boot", type=int, default=1000)
    lk.add_argument("--seed", type=int, default=0)
    lk.add_argument("--out", type=Path, required=True)

    lv = sub.add_parser("lens-vocab", help="5c word tables")
    lv.add_argument("--dict-in", type=Path, required=True, help="dictionary at the input layer")
    lv.add_argument("--dict-out", type=Path, required=True, help="dictionary at the output layer")
    lv.add_argument("--mprime", type=Path, default=None)
    lv.add_argument("--shipped", type=Path, default=None)
    lv.add_argument("--rb-dir", type=Path, default=None)
    lv.add_argument("--topk-dirs", type=int, default=10)
    lv.add_argument("--topk-words", type=int, default=30)
    lv.add_argument("--model", default=C.DEFAULT_MODEL)
    lv.add_argument("--out", type=Path, required=True)

    ch = sub.add_parser("chain", help="5d chain-composition judge-free DV")
    ch.add_argument("--chunks-dir", type=Path, required=True)
    ch.add_argument("--dict-out", type=Path, required=True)
    ch.add_argument("--op", required=True, help="NAME=PATH=INLAYER (ridge payload)")
    ch.add_argument("--model", default=C.DEFAULT_MODEL)
    ch.add_argument("--max-ctx", type=int, default=0)
    ch.add_argument("--df-cap", type=float, default=0.5)
    ch.add_argument("--topk", type=int, default=50)
    ch.add_argument("--n-null", type=int, default=200)
    ch.add_argument("--seed", type=int, default=0)
    ch.add_argument("--score-batch", type=int, default=64)
    ch.add_argument("--out", type=Path, required=True)

    sm = sub.add_parser("smoke", help="CPU smokes (see module docstring)")
    sm.add_argument("--out-dir", type=Path, default=Path("/tmp/i1776-p5-smoke"))
    sm.add_argument("--legs", nargs="*", default=None)

    args = ap.parse_args(argv)
    fn = {
        "stream": cmd_stream,
        "capture": cmd_capture,
        "transfer": cmd_transfer,
        "leakage": cmd_leakage,
        "lens-vocab": cmd_lens_vocab,
        "chain": cmd_chain,
        "smoke": smoke,
    }[args.cmd]
    return fn(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension finalize (#1689 atexit race)
