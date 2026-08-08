#!/usr/bin/env python
"""Issue #1482 — autointerp DESCRIPTIONS for the 1,654 CONTEXT-ONLY SAE features.

#1773 described ~127,605/131,072 features from ANSWER-side activating windows and
structurally excluded every context-only feature (zero answer-side windows by the
defining property). This driver closes that gap with CONTEXT-side evidence, holding
the #1773 description INSTRUMENT fixed (DESCRIBER_SYSTEM verbatim, same user
template renderer, same judge, max_tokens=700, temp 1.0, 1 draw).

EVIDENCE-SIDE CAVEAT (rides every output row as ``evidence_side: "context"`` and the
meta): these descriptions are drawn from CONTEXT-side windows; every pre-existing
#1773 description is from ANSWER-side windows. The two sets are NOT interchangeable
and MUST NOT be pooled or compared as if same-instrument.

Stages (each resumable; smoke IS production with --limit-features):
  scan      VM CPU. fused_scan census assert -> 1,654 context-only ids; ONE pass over
            the 1,920-shard pooled store collecting, per feature, its fit-row
            (set_tag==1) context-side rows + psi_mean; keeps top-<=ROW_CAP rows per
            feature (highest psi_mean first, deterministic lexsort tie-break);
            samples the NEG-pool rows + their per-row active context-only sets.
  extract   VM CPU+net. Parallel sweep of the 1,920 raw chunks (3.22 GB total) under
            the parent RAW_PREFIX filtering needed ci -> needed_rows.jsonl (ci,
            prompt; LMSYS/WildChat text handled DIGEST-ONLY — never printed); builds
            the shared non-activating context-span pool (#1773 _nonact_span analogue,
            spans start >= prefix_end-8 so template boilerplate never dominates);
            uploads inputs to HF for pod staging.
  encode    POD GPU. Stages inputs (local-first -> HF -> fail-loud), tokenizes each
            needed row's prompt (parent _tokenize_row convention, prompt side only),
            batched L19 resid_post capture (EA._batched_capture), partial-feature
            BatchTopK encode (per-token thresholded ReLU == full encode on the
            feature's columns; parity-asserted per run), reference token-pool mask
            (BOS_OFFSET strip + >10x-median-norm outlier drop) for peak eligibility,
            #1773 window shape ([peak-15, peak+16] clipped to the CONTEXT span, peak
            <<marked>>) -> windows_expos jsonl; uploads to HF.
  describe  VM API. Packets {ex_pos<=ROW_CAP marked, ex_neg 20 plain, out=None} ->
            issue1773_describe_axes._dispatch (dispatch_judge_items; Sonnet judge
            pin, checkpointed, transport-retried; one extra transport re-dispatch)
            -> eval_results/issue_1482/context_side_labels/descriptions_context_side
            .jsonl + meta.json; raw judge text uploaded (upload policy).

OUT block: omitted (out=None) — the #1773 full-dictionary describe run itself carried
``out: None`` for every feature outside the 16,384-feature phase0 join (evidence
builder line 908/968), so no-OUT matches the dominant instrument shape, and phase0
footprints do not exist for context-only features.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM discipline)

import issue1773_common as CM  # noqa: E402
import numpy as np  # noqa: E402

TASK_ID = 1482
SEED = 14_822_026
ROW_CAP = 16  # ex_pos rows per feature (highest psi_mean first; matches the brief)
NEG_POOL_ROWS = 400  # shared non-activating context-span candidate pool
N_EX_NEG = CM.N_NONACT_EVIDENCE  # 20, #1773 parity
N_CTX_ONLY_EXPECTED = 1654
N_FIT_EXPECTED = 120_000
FIT_TAG = 1  # set_tag convention of the pooled store (issue1482_error_analysis p3)

OUT_ROOT_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1482_ctxlabels")
STORE_DEFAULT = CM.STORE_DEFAULT  # the 1,920-shard pooled store
FUSED_SCAN_DEFAULT = PROJECT_ROOT / "data/issue_1482/fullwidth/fused_scan.npz"
HF_PREFIX_OUT = "issue1482_context_side_labels"
OUT_EVAL = PROJECT_ROOT / "eval_results/issue_1482/context_side_labels"

EVIDENCE_SIDE_CAVEAT = (
    "EVIDENCE-SIDE CAVEAT: every description in this file is generated from "
    "CONTEXT-side activating windows (the model's input/prompt span). Every "
    "pre-existing #1773 description (issue1773_featurepipeline) is generated from "
    "ANSWER-side windows. The two sets are NOT same-instrument: do NOT pool them "
    "into one description set or compare them as if drawn from one instrument; "
    "label context-side descriptions distinctly wherever surfaced."
)


def _log(msg: str) -> None:
    print(f"[ctxlbl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))


# ── stage: scan ──────────────────────────────────────────────────────────────────


def _ctx_only_ids(fused_scan: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(fused_scan) as z:
        cnt = z["cnt_fit"].astype(np.int64)
        psi = z["psi_cnt_fit"].astype(np.int64)
        n_fit = int(z["n_fit"])
    assert n_fit == N_FIT_EXPECTED, n_fit
    ctx_only = np.where((psi > 0) & (cnt == 0))[0]
    assert len(ctx_only) == N_CTX_ONLY_EXPECTED, (
        f"census drift: {len(ctx_only)} context-only features != {N_CTX_ONLY_EXPECTED}"
    )
    return ctx_only, psi


def stage_scan(args) -> int:
    ctx_only, psi = _ctx_only_ids(args.fused_scan)
    target = ctx_only[: args.limit_features] if args.limit_features else ctx_only
    mask = np.zeros(CM.DICT_SIZE, dtype=bool)
    mask[target] = True

    shards = sorted(Path(args.store).glob("pooled_*.npz"))
    assert len(shards) == CM.N_SHARDS, f"{len(shards)} shards != {CM.N_SHARDS}"
    feat_l: list[np.ndarray] = []
    row_l: list[np.ndarray] = []
    ci_l: list[np.ndarray] = []
    val_l: list[np.ndarray] = []
    fit_rows: list[int] = []
    fit_cis: list[int] = []
    n_fit_seen = 0
    t0 = time.time()
    for si, sp in enumerate(shards):
        with np.load(sp, allow_pickle=False) as s:
            offs = np.concatenate([[0], np.cumsum(s["psi_off"])])
            tags = s["set_tag"]
            rows = s["row_idx"]
            cis = s["ci"]
            pidx = s["psi_idx"]
            pval = s["psi_mean"]
            for i in range(len(rows)):
                if int(tags[i]) != FIT_TAG:
                    continue
                n_fit_seen += 1
                fit_rows.append(int(rows[i]))
                fit_cis.append(int(cis[i]))
                sl = slice(int(offs[i]), int(offs[i + 1]))
                f = pidx[sl].astype(np.int64)
                m = mask[f]
                if m.any():
                    feat_l.append(f[m])
                    val_l.append(pval[sl][m].astype(np.float32))
                    k = int(m.sum())
                    row_l.append(np.full(k, int(rows[i]), dtype=np.int64))
                    ci_l.append(np.full(k, int(cis[i]), dtype=np.int64))
        if (si + 1) % 400 == 0:
            _log(f"[scan] {si + 1}/{len(shards)} shards ({time.time() - t0:.0f}s)")
    assert n_fit_seen == N_FIT_EXPECTED, f"fit rows seen {n_fit_seen} != {N_FIT_EXPECTED}"

    feat = np.concatenate(feat_l)
    row = np.concatenate(row_l)
    ci = np.concatenate(ci_l)
    val = np.concatenate(val_l)
    # store-vs-fused-scan parity: per-feature pair counts must reproduce psi_cnt_fit
    binc = np.bincount(feat, minlength=CM.DICT_SIZE)
    bad = np.where(binc[target] != psi[target])[0]
    assert len(bad) == 0, f"pair-count mismatch vs psi_cnt_fit on {len(bad)} features"

    # top-<=ROW_CAP rows per feature by psi_mean desc; deterministic tie-break row asc
    order = np.lexsort((row, -val, feat))
    feat_s, row_s, ci_s, val_s = feat[order], row[order], ci[order], val[order]
    keep = np.zeros(len(feat_s), dtype=bool)
    # rank within feature group (groups are contiguous after lexsort)
    grp_start = np.flatnonzero(np.r_[True, feat_s[1:] != feat_s[:-1]])
    rank = np.arange(len(feat_s)) - np.repeat(grp_start, np.diff(np.r_[grp_start, len(feat_s)]))
    keep = rank < ROW_CAP
    sel = {
        "feat": feat_s[keep],
        "row": row_s[keep],
        "ci": ci_s[keep],
        "val": val_s[keep],
    }

    # NEG pool: seeded sample of fit rows + per-row active context-only feature sets
    rng = np.random.default_rng(SEED)
    fit_rows_a = np.asarray(fit_rows, dtype=np.int64)
    fit_cis_a = np.asarray(fit_cis, dtype=np.int64)
    pick = rng.choice(len(fit_rows_a), size=min(NEG_POOL_ROWS, len(fit_rows_a)), replace=False)
    pick.sort()
    neg_rows = fit_rows_a[pick]
    neg_cis = fit_cis_a[pick]
    neg_set = set(int(r) for r in neg_rows)
    in_neg = np.asarray([int(r) in neg_set for r in row], dtype=bool)
    neg_active_feat = feat[in_neg]
    neg_active_row = row[in_neg]

    out = args.out_root / "scan"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "feature_rows.npz",
        feat=sel["feat"],
        row=sel["row"],
        ci=sel["ci"],
        val=sel["val"],
        ctx_only=target,
        psi_cnt_fit=psi[target],
    )
    np.savez_compressed(
        out / "negpool.npz",
        rows=neg_rows,
        cis=neg_cis,
        active_feat=neg_active_feat,
        active_row=neg_active_row,
    )
    needed_ci = sorted(set(int(c) for c in sel["ci"]) | set(int(c) for c in neg_cis))
    (out / "needed_ci.json").write_text(json.dumps(needed_ci))
    meta = {
        **CM.repro_meta(),
        "n_ctx_only": int(len(target)),
        "n_pairs_total": int(len(feat)),
        "n_pairs_selected": int(keep.sum()),
        "n_unique_expos_rows": int(len(set(int(r) for r in sel["row"]))),
        "n_neg_pool_rows": int(len(neg_rows)),
        "n_needed_ci": len(needed_ci),
        "row_cap": ROW_CAP,
        "limit_features": args.limit_features,
        "seed": SEED,
    }
    _write_json(out / "scan_meta.json", meta)
    _log(f"[scan] done: {json.dumps({k: v for k, v in meta.items() if isinstance(v, int)})}")
    return 0


# ── stage: extract ───────────────────────────────────────────────────────────────


def _tokenize_prompt(tok, prompt: str, prefix_chars: int):
    """Prompt-side half of EA._tokenize_row (identical convention; no response).

    Returns (prompt_ids, prefix_end). Context span == all prompt tokens; the
    context-end token is prompt_ids[-1] (parent: context_end = len(prompt_ids)-1).
    """
    import issue779_common as C

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    enc = tok(text, return_offsets_mapping=True)
    prompt_ids = enc["input_ids"]
    suffix = tok.decode(prompt_ids[-3:])
    assert suffix == C.GENERATION_SUFFIX, f"position assert: {suffix!r}"
    prefix_end = -1
    for i, (_, e) in enumerate(enc["offset_mapping"]):
        if e <= prefix_chars:
            prefix_end = i
        else:
            break
    assert prefix_end >= 0, "no token ends inside the template prefix"
    return prompt_ids, prefix_end


def stage_extract(args) -> int:
    import issue779_common as C
    import issue779_ffc_n1m_fits as N1M
    import issue1482_error_analysis as EA

    scan_dir = args.out_root / "scan"
    needed_ci = set(json.loads((scan_dir / "needed_ci.json").read_text()))
    names = EA._raw_chunk_names(SimpleNamespace(max_chunks=args.max_chunks))
    _log(f"[extract] {len(names)} chunks, {len(needed_ci)} needed ci, {args.workers} workers")

    cache_root = args.out_root / "extract" / "raw_cache"
    found: dict[int, str] = {}

    def worker(wi: int, name: str) -> list[tuple[int, str]]:
        cache = cache_root / f"w{wi:02d}"
        cache.mkdir(parents=True, exist_ok=True)
        got = Path(N1M._download_chunk_with_retry(C.HF_DATA_REPO, f"{EA.RAW_PREFIX}/{name}", cache))
        rows = json.loads(got.read_text())["rows"]
        keep = [(int(r["ci"]), r["prompt"]) for r in rows if int(r["ci"]) in needed_ci]
        got.unlink()
        return keep

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, i % args.workers, n): n for i, n in enumerate(names)}
        for fut in as_completed(futs):
            for ci_, prompt in fut.result():
                found[ci_] = prompt
            n_done += 1
            if n_done % 200 == 0:
                _log(f"[extract] {n_done}/{len(names)} chunks ({time.time() - t0:.0f}s)")
    missing = needed_ci - set(found)
    assert not missing, f"{len(missing)} needed ci not found in raw chunks"

    out = args.out_root / "extract"
    out.mkdir(parents=True, exist_ok=True)
    # sharded (<9 MB/shard): a single needed_rows.jsonl exceeds 10 MB and would
    # force-route to LFS (upload policy: text >9.5 MB line-splits, never gzip/LFS)
    CM.write_jsonl_sharded(
        [{"ci": ci_, "prompt": found[ci_]} for ci_ in sorted(found)], out, "needed_rows"
    )

    # NEG-span pool: random 32-token context spans from neg rows, biased past the
    # constant template prefix so boilerplate never dominates the pool.
    from transformers import AutoTokenizer

    from explore_persona_space.orchestrate import hub

    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct"),
        what="tokenizer fetch",
    )
    prefix_chars = EA._prefix_char_len(tok)
    npz = np.load(scan_dir / "negpool.npz")
    rng = np.random.default_rng(SEED + 1)
    w = CM.WINDOW_BACK + CM.WINDOW_FWD + 1
    import issue1482_sae as S

    spans = []
    for row_idx, ci_ in zip(npz["rows"].tolist(), npz["cis"].tolist(), strict=True):
        ids, prefix_end = _tokenize_prompt(tok, found[int(ci_)], prefix_chars)
        T = len(ids)
        lo_min = min(max(S.BOS_OFFSET, prefix_end - 8), max(0, T - w))
        lo = int(rng.integers(lo_min, max(lo_min, T - w) + 1))
        hi = min(T, lo + w)
        plain = tok.decode(ids[lo:hi])
        spans.append(
            {
                "row_idx": int(row_idx),
                "ci": int(ci_),
                "token_lo": lo,
                "token_hi": hi,
                "peak_pos": -1,
                "values_fp16": [],
                "text_marked": plain,
                "text_plain": plain,
            }
        )
    CM.write_jsonl_sharded(spans, out, "negpool_spans")
    meta = {
        **CM.repro_meta(),
        "n_chunks": len(names),
        "n_found": len(found),
        "n_neg_spans": len(spans),
        "wall_s": round(time.time() - t0, 1),
    }
    _write_json(out / "extract_meta.json", meta)
    _log(f"[extract] done: {meta['n_found']} rows, {meta['n_neg_spans']} neg spans")

    if not args.no_upload:
        _upload_tree(args, ["scan", "extract"], what="inputs")
    return 0


def _upload_tree(args, subdirs: list[str], what: str) -> None:
    """One batched upload_folder commit per subdir (persist-by-default; text/JSON
    rides the non-LFS path)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.hub import assert_hub_dir_filecounts

    for sub in subdirs:
        local = args.out_root / sub
        pats = ["*.json", "*.jsonl", "*.npz", "**/*.json", "**/*.jsonl", "**/*.npz"]
        assert_hub_dir_filecounts(
            folder_path=str(local),
            path_in_repo=f"{HF_PREFIX_OUT}/{sub}",
            allow_patterns=pats,
            ignore_patterns=["raw_cache/**"],
        )
        hub.retry_transient(
            lambda local=local, sub=sub: HfApi().upload_folder(
                folder_path=str(local),
                repo_id=CM.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{HF_PREFIX_OUT}/{sub}",
                allow_patterns=pats,
                ignore_patterns=["raw_cache/**"],
            ),
            what=f"{what} upload ({sub})",
        )
        _log(f"[upload] {sub} -> {CM.HF_DATA_REPO}/{HF_PREFIX_OUT}/{sub}")


def _stage_inputs_local_first(args, subdirs: list[str]) -> None:
    """Local-first -> HF fetch -> fail-loud staging of VM-produced inputs (pod lane)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    for sub in subdirs:
        local = args.out_root / sub
        if local.exists() and any(local.iterdir()):
            continue
        prefix = f"{HF_PREFIX_OUT}/{sub}"
        files = hub.list_hf_files_under_path(api, CM.HF_DATA_REPO, prefix, repo_type="dataset")
        assert files, f"inputs missing locally AND on HF under {prefix} — run scan/extract first"
        local.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        for f in files:
            rel = f[len(prefix) + 1 :]
            dst = local / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            got = hub.retry_transient(
                lambda f=f: hf_hub_download(
                    CM.HF_DATA_REPO,
                    filename=f,
                    repo_type="dataset",
                    local_dir=args.out_root / "_hf",
                ),
                what=f"input fetch ({f})",
            )
            Path(got).replace(dst) if not dst.exists() else None
        _log(f"[stage] fetched {len(files)} files -> {local}")


# ── stage: encode (pod GPU) ──────────────────────────────────────────────────────


def _partial_encode(sae, h, feats):
    """BatchTopK inference on a column subset: per-token thresholded ReLU is
    feature-independent, so encoding only ``feats`` equals full-encode columns."""
    import torch

    W = sae.w_enc[feats]  # (nf, act_dim)
    b = sae.b_enc[feats]
    x = h.to(device=sae.device, dtype=torch.float32) - sae.b_dec
    f = torch.relu(x @ W.T + b)
    return f * (f > sae.threshold)


def stage_encode(args) -> int:
    import torch

    import issue1482_error_analysis as EA
    import issue1482_sae as S

    _stage_inputs_local_first(args, ["scan", "extract"])
    scan_dir = args.out_root / "scan"
    ext_dir = args.out_root / "extract"
    fr = np.load(scan_dir / "feature_rows.npz")
    feat, row, ci, val = fr["feat"], fr["row"], fr["ci"], fr["val"]
    if args.limit_features:
        keep_f = np.asarray(sorted(set(feat.tolist()))[: args.limit_features])
        m = np.isin(feat, keep_f)
        feat, row, ci, val = feat[m], row[m], ci[m], val[m]
    prompts: dict[int, str] = {}
    shard_paths = sorted(ext_dir.glob("needed_rows.shard*.jsonl"))
    assert shard_paths, f"no needed_rows shards under {ext_dir} — run extract first"
    for p in shard_paths:
        for r in CM.iter_jsonl(p):
            prompts[int(r["ci"])] = r["prompt"]

    by_row: dict[int, list[int]] = {}
    row_ci: dict[int, int] = {}
    for f_, r_, c_ in zip(feat.tolist(), row.tolist(), ci.tolist(), strict=True):
        by_row.setdefault(int(r_), []).append(int(f_))
        row_ci[int(r_)] = int(c_)
    _log(f"[encode] {len(set(feat.tolist()))} features, {len(by_row)} unique rows")

    model, tok = EA._load_model_tok(SimpleNamespace(tiny_model=args.tiny_model, device=args.device))
    prefix_chars = EA._prefix_char_len(tok)
    sae = S.BatchTopKSAE.load(k=64, device=args.device, cache_dir=args.out_root / "sae")

    # tokenize all rows; sort by length for batching
    rows_tok = []
    for r_ in sorted(by_row):
        ids, prefix_end = _tokenize_prompt(tok, prompts[row_ci[r_]], prefix_chars)
        rows_tok.append((r_, row_ci[r_], ids, prefix_end))
    rows_tok.sort(key=lambda t: len(t[2]))
    total_tokens = sum(len(t[2]) for t in rows_tok)
    _log(f"[encode] {total_tokens} total context tokens")

    windows: list[dict] = []
    misses: list[dict] = []
    parity_done = False
    t0 = time.time()
    bi = 0
    i = 0
    while i < len(rows_tok):
        batch = [rows_tok[i]]
        i += 1
        while i < len(rows_tok) and len(batch) * len(rows_tok[i][2]) <= args.batch_tokens:
            batch.append(rows_tok[i])
            i += 1
        caps = EA._batched_capture(model, tok, batch, (EA.LAYER,), args.device)
        for (r_, ci_, ids, _pe), cap in zip(batch, caps, strict=True):
            h = cap[EA.LAYER]  # (T, 3584) fp32 CPU
            T = h.shape[0]
            keep = S.token_inlier_mask(h)
            keep[: min(S.BOS_OFFSET, T)] = False
            feats = sorted(by_row[r_])
            fidx = torch.tensor(feats, dtype=torch.long, device=sae.device)
            acts = _partial_encode(sae, h, fidx).cpu()  # (T, nf)
            if not parity_done:
                full = sae.encode(h)[:, fidx].cpu()
                assert torch.allclose(full, acts, atol=1e-4), "partial-encode parity FAIL"
                parity_done = True
                _log("[encode] partial-vs-full encode parity OK")
            keep_t = keep.clone()
            for j, f_ in enumerate(feats):
                vals = acts[:, j]
                elig = vals.clone()
                elig[~keep_t] = 0.0
                flag = "inlier"
                if float(elig.max()) <= 0.0:
                    elig = vals
                    flag = "masked_peak"
                if float(elig.max()) <= 0.0:
                    misses.append({"feat_id": f_, "row_idx": r_, "ci": ci_})
                    continue
                peak = int(elig.argmax())
                lo = max(0, peak - CM.WINDOW_BACK)
                hi = min(T, peak + CM.WINDOW_FWD + 1)
                pre = tok.decode(ids[lo:peak])
                pk = tok.decode([ids[peak]])
                post = tok.decode(ids[peak + 1 : hi])
                windows.append(
                    {
                        "feat_id": f_,
                        "row_idx": r_,
                        "ci": ci_,
                        "peak_pos": peak,
                        "peak_flag": flag,
                        "peak_val": float(vals[peak]),
                        "token_lo": int(lo),
                        "token_hi": int(hi),
                        "text_marked": f"{pre}<<{pk}>>{post}",
                        "text_plain": tok.decode(ids[lo:hi]),
                        "values_fp16": [float(np.float16(v)) for v in vals[lo:hi].tolist()],
                    }
                )
        bi += 1
        if bi % 20 == 0:
            done_tok = sum(len(t[2]) for t in rows_tok[:i])
            _log(
                f"[encode] batch {bi}: {i}/{len(rows_tok)} rows, "
                f"{done_tok / max(1e-9, time.time() - t0):.0f} tok/s"
            )

    out = args.out_root / "encode"
    out.mkdir(parents=True, exist_ok=True)
    CM.write_jsonl_sharded(windows, out, "windows_expos")
    n_feat_hit = len({w["feat_id"] for w in windows})
    meta = {
        **CM.repro_meta(),
        "n_rows_encoded": len(rows_tok),
        "n_total_tokens": total_tokens,
        "n_windows": len(windows),
        "n_features_with_windows": n_feat_hit,
        "n_misses": len(misses),
        "n_masked_peak": sum(1 for w in windows if w["peak_flag"] == "masked_peak"),
        "wall_s": round(time.time() - t0, 1),
        "layer": EA.LAYER,
        "window": [CM.WINDOW_BACK, CM.WINDOW_FWD],
        "device": args.device,
        "tiny_model": bool(args.tiny_model),
    }
    _write_json(out / "misses.json", {"misses": misses})
    _write_json(out / "encode_meta.json", meta)
    _log(f"[encode] done: {json.dumps({k: v for k, v in meta.items() if isinstance(v, int)})}")
    if not args.no_upload:
        _upload_tree(args, ["encode"], what="encode outputs")
    return 0


# ── stage: describe (VM API) ─────────────────────────────────────────────────────


def _build_packets(args) -> tuple[dict[int, dict], dict]:
    scan_dir = args.out_root / "scan"
    enc_dir = args.out_root / "encode"
    ext_dir = args.out_root / "extract"
    fr = np.load(scan_dir / "feature_rows.npz")
    ctx_only = fr["ctx_only"].tolist()
    psi_cnt = dict(zip(fr["ctx_only"].tolist(), fr["psi_cnt_fit"].tolist(), strict=True))
    val_by = {
        (int(f), int(r)): float(v) for f, r, v in zip(fr["feat"], fr["row"], fr["val"], strict=True)
    }
    npz = np.load(scan_dir / "negpool.npz")
    neg_active: dict[int, set[int]] = {}
    for f_, r_ in zip(npz["active_feat"].tolist(), npz["active_row"].tolist(), strict=True):
        neg_active.setdefault(int(r_), set()).add(int(f_))
    spans: list[dict] = []
    for p in sorted(ext_dir.glob("negpool_spans*.jsonl")):
        spans.extend(CM.iter_jsonl(p))

    by_feat: dict[int, list[dict]] = {}
    for p in sorted(enc_dir.glob("windows_expos*.jsonl")):
        for w in CM.iter_jsonl(p):
            by_feat.setdefault(int(w["feat_id"]), []).append(w)

    packets: dict[int, dict] = {}
    n_neg_short = 0
    for f_, ws in sorted(by_feat.items()):
        ws.sort(key=lambda w: (-val_by.get((f_, int(w["row_idx"])), 0.0), int(w["row_idx"])))
        ex_neg = []
        start = f_ % max(1, len(spans))
        for k in range(len(spans)):
            sp = spans[(start + k) % len(spans)]
            if f_ not in neg_active.get(int(sp["row_idx"]), set()):
                ex_neg.append(sp)
            if len(ex_neg) >= N_EX_NEG:
                break
        if len(ex_neg) < N_EX_NEG:
            n_neg_short += 1
        packets[f_] = {
            "feat_id": f_,
            "ex_pos": ws[:ROW_CAP],
            "ex_neg": ex_neg,
            "out": None,
        }
    undescribed = sorted(set(int(f) for f in ctx_only) - set(packets))
    diag = {
        "n_packets": len(packets),
        "n_undescribed_no_windows": len(undescribed),
        "undescribed_feat_ids": undescribed,
        "n_neg_short": n_neg_short,
    }
    return packets, diag


def stage_describe(args) -> int:
    import issue1773_describe_axes as DA

    _stage_inputs_local_first(args, ["scan", "extract", "encode"])
    packets, diag = _build_packets(args)
    if args.limit_features:
        packets = {f: packets[f] for f in sorted(packets)[: args.limit_features]}
    _log(
        f"[describe] {len(packets)} packets ({json.dumps({k: v for k, v in diag.items() if isinstance(v, int)})})"
    )

    items = DA.build_describe_items(packets)
    if args.dry_run:
        rd = args.out_root / "rendered_prompts"
        rd.mkdir(parents=True, exist_ok=True)
        for cid, _q, _c, user in items[:5]:
            (rd / f"{cid}.txt").write_text(f"SYSTEM:\n{CM.DESCRIBER_SYSTEM}\n\nUSER:\n{user}")
        _log(f"[describe] dry-run: rendered {min(5, len(items))} prompts -> {rd} (not printed)")
        return 0

    ck_tag = f"pilot{args.limit_features}" if args.limit_features else "full"
    results = DA._dispatch(
        items,
        system=CM.DESCRIBER_SYSTEM,
        max_tokens=CM.DESCRIBE_MAX_TOKENS,
        checkpoint_dir=args.out_root / "judge_checkpoints" / f"describe_{ck_tag}",
        force_batch=args.force_batch,
    )
    # one transport re-dispatch (llm-judging rule 24) — fresh checkpoint subdir
    transport_cids = [
        cid
        for cid, res in results.items()
        if isinstance(res, dict) and res.get("error") and DA._classify_error(res) == "transport"
    ]
    if transport_cids:
        _log(f"[describe] re-dispatching {len(transport_cids)} transport losses")
        sub = [it for it in items if it[0] in set(transport_cids)]
        retry = DA._dispatch(
            sub,
            system=CM.DESCRIBER_SYSTEM,
            max_tokens=CM.DESCRIBE_MAX_TOKENS,
            checkpoint_dir=args.out_root / "judge_checkpoints" / f"describe_{ck_tag}_retry",
            force_batch=args.force_batch,
        )
        results.update(retry)

    rows, drops = [], {"content": 0, "transport": 0}
    for cid, _q, _c, user in items:
        res = results.get(cid)
        if isinstance(res, dict) and res.get("error"):
            drops[DA._classify_error(res)] += 1
            continue
        parsed = DA.parse_describe_result(res)
        if parsed is None:
            drops["content"] += 1
            continue
        feat_id = int(cid[1:].rsplit("-", 1)[0])
        pk = packets[feat_id]
        rows.append(
            {
                "feat_id": feat_id,
                "evidence_side": "context",
                **parsed,
                "prompt_sha16": CM.sha16(user),
                "n_ex_pos": len(pk["ex_pos"]),
                "n_ex_neg": len(pk["ex_neg"]),
            }
        )
    DA._write_raw(results, args.out_root / "judge_raw" / f"describe_ctx_{ck_tag}")

    if args.limit_features:
        _log(f"[describe] PILOT: {len(rows)}/{len(items)} ok, drops={drops}")
        _write_json(args.out_root / f"pilot_{ck_tag}.json", {"n_ok": len(rows), "drops": drops})
        return 0

    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    with (OUT_EVAL / "descriptions_context_side.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {
        **CM.repro_meta(),
        "task_id": TASK_ID,
        "caveat": EVIDENCE_SIDE_CAVEAT,
        "evidence_side": "context",
        "instrument": {
            "system_prompt_sha16": CM.sha16(CM.DESCRIBER_SYSTEM),
            "user_template": "issue1773_common.build_describe_user_msg (verbatim reuse)",
            "max_tokens": CM.DESCRIBE_MAX_TOKENS,
            "temperature": CM.JUDGE_TEMPERATURE,
            "n_draws": 1,
            "out_block": "omitted (matches #1773 full-dictionary shape: out=None)",
            "window": [CM.WINDOW_BACK, CM.WINDOW_FWD],
            "row_cap": ROW_CAP,
            "n_ex_neg": N_EX_NEG,
        },
        "population": {
            "n_context_only": N_CTX_ONLY_EXPECTED,
            "n_described": len(rows),
            "drops": drops,
            **diag,
        },
        "hf_prefix": f"{CM.HF_DATA_REPO}/{HF_PREFIX_OUT}",
    }
    _write_json(OUT_EVAL / "descriptions_context_side.meta.json", meta)
    _log(f"[describe] done: {len(rows)}/{len(items)} described, drops={drops}")
    if not args.no_upload:
        _upload_tree(args, ["judge_raw"], what="judge raw")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["scan", "extract", "encode", "describe"])
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    ap.add_argument("--store", type=Path, default=STORE_DEFAULT)
    ap.add_argument("--fused-scan", type=Path, default=FUSED_SCAN_DEFAULT)
    ap.add_argument("--limit-features", type=int, default=0, help="smoke/pilot slice")
    ap.add_argument("--max-chunks", type=int, default=0, help="extract smoke slice")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny-model", action="store_true", help="CPU carve-out smoke model")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-batch", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    return {
        "scan": stage_scan,
        "extract": stage_extract,
        "encode": stage_encode,
        "describe": stage_describe,
    }[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
