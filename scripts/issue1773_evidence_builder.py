#!/usr/bin/env python
"""Issue #1773 Phase 1 — the evidence builder (plan §4; supersedes phase_scan's
300-feature sampler). Three passes, one script:

  --pass select   (A, VM CPU)  one streamed pass over the 1,920 pooled shards:
                  per-feature uniform RESERVOIR (660 rows ≈66/decile expected)
                  of (row, ans_max) candidates for all 16,384 restricted
                  features (vectorized Algorithm R — no per-feature loop),
                  10-quantile-bin stratified draw of 60 activating rows
                  (40 evidence / 20 held-out), 40 non-activating candidate rows
                  (26 target + spares; Pass B verifies), 200 seeded random
                  directions. Emits the selection manifest + inverted index.
  --pass windows  (B, GPU; the only GPU phase)  chunk-sharded workers
                  (names[w::n_workers]); per chunk: parent capture path
                  (_tokenize_row / _batched_capture / BatchTopKSAE.encode with
                  reference token-pool parity), peak-marked 32-token windows,
                  non-activating verification, random-direction inline top-K.
                  Per-chunk JSONL + done sentinel, resume keyed on (chunk,
                  config fingerprint), batched incremental HF upload.
  --pass assemble (C, VM CPU)  join per-chunk windows into per-feature evidence
                  packets ([EX+ 40][EX- 20][NEAR 5][OUT][STAT]) + held-out
                  scoring sets + completeness report (H2 gate).

Tiny-real CPU e2e (tests + smoke): --local-chunks + --tiny-model/--sae-state
--act-dim/--dict-size/--layer run the REAL capture path on a from-config model.
Import-resolution leg: --import-check resolves every deferred import + exits.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM discipline)

import issue1773_common as CM  # noqa: E402
import numpy as np  # noqa: E402

RESERVOIR_CKPT_EVERY = 200  # per-~200-shard compaction/checkpoint (plan §4 Pass A)
NONACT_SPARES = 40  # 26 target + 14 spares (Pass-B verification budget; recorded choice)
SPAN_RESAMPLE_TRIES = 1  # non-activating span re-draw budget per row (plan: once)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _config_fingerprint(args) -> str:
    """Regime fingerprint for resume predicates: SAE revision/k, window params,
    seed, layer, act/dict dims (every output-affecting key — #722 r3 rule)."""
    import issue1482_sae as S

    payload = "|".join(
        str(x)
        for x in (
            S.SAE_REVISION,
            args.k,
            CM.WINDOW_BACK,
            CM.WINDOW_FWD,
            CM.SEED,
            args.layer,
            args.act_dim,
            args.dict_size,
            bool(args.tiny_model),
        )
    )
    return CM.sha16(payload)


# ── Pass A: selection ────────────────────────────────────────────────────────


def _reservoir_update(
    state: dict,
    f: np.ndarray,
    rows: np.ndarray,
    cis: np.ndarray,
    vals: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Vectorized Algorithm R over per-feature reservoirs. Items MUST be sorted
    by feature with original order preserved within a feature (stable sort);
    numpy fancy assignment applies writes in array order, matching sequential
    replacement semantics exactly."""
    R = CM.RESERVOIR_PER_FEATURE
    order = np.argsort(f, kind="stable")
    f, rows, cis, vals = f[order], rows[order], cis[order], vals[order]
    uniq, starts, counts = np.unique(f, return_index=True, return_counts=True)
    seq = np.arange(len(f)) - np.repeat(starts, counts)
    n_before = state["n_seen"][f] + seq
    u = rng.random(len(f))
    j = np.floor(u * (n_before + 1)).astype(np.int64)
    slot = np.where(n_before < R, n_before, j)
    accept = slot < R
    fa, sa = f[accept], slot[accept]
    state["res_row"][fa, sa] = rows[accept]
    state["res_ci"][fa, sa] = cis[accept]
    state["res_val"][fa, sa] = vals[accept]
    np.add.at(state["n_seen"], uniq, counts)


def pass_select(args) -> int:
    """Pass A: stream shards -> reservoirs -> stratified activating draw +
    non-activating candidate draw + random directions + inverted index.
    Final act: upload the selection dir to the Hub (Pass B consumes it on a
    fresh GCE clone — crash-fix r3, #1773). `--upload-only` skips the build
    and just runs the upload against an existing selection dir."""
    if args.upload_only:
        return upload_selection(args.selection_dir)
    rng = np.random.default_rng(CM.SEED)
    com = np.load(CM.PERFEATURE_NPZ, allow_pickle=False)
    fid = np.asarray(com["feat_ids"], dtype=np.int64)
    n_feat = args.feature_limit if args.feature_limit > 0 else len(fid)
    fid = fid[:n_feat]
    pos = np.full(CM.DICT_SIZE, -1, dtype=np.int64)
    pos[fid] = np.arange(n_feat)

    shards = sorted(args.store.glob("pooled_*.npz"))
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    else:
        assert len(shards) == CM.N_SHARDS, f"expected {CM.N_SHARDS} shards, got {len(shards)}"

    R = CM.RESERVOIR_PER_FEATURE
    state = {
        "res_row": np.full((n_feat, R), -1, dtype=np.int64),
        "res_ci": np.full((n_feat, R), -1, dtype=np.int64),
        "res_val": np.zeros((n_feat, R), dtype=np.float32),
        "n_seen": np.zeros(n_feat, dtype=np.int64),
        "next_shard": 0,
    }
    fit_rows_parts: list[np.ndarray] = []
    fit_ci_parts: list[np.ndarray] = []
    fp = CM.sha16(f"{args.store}|{len(shards)}|{n_feat}|{CM.SEED}")
    sel_dir = args.selection_dir
    sel_dir.mkdir(parents=True, exist_ok=True)
    ckpt = sel_dir / f"passA_ckpt_{fp}.npz"
    if ckpt.exists() and not args.no_resume:
        z = np.load(ckpt, allow_pickle=False)
        for k in ("res_row", "res_ci", "res_val", "n_seen"):
            state[k] = z[k]
        state["next_shard"] = int(z["next_shard"])
        fit_rows_parts = [z["fit_rows"]]
        fit_ci_parts = [z["fit_ci"]]
        rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, int(state["next_shard"])]))
        _log(f"[passA] resume at shard {state['next_shard']}/{len(shards)}")

    t0 = time.time()
    for i in range(int(state["next_shard"]), len(shards)):
        p = shards[i]
        with np.load(p, allow_pickle=False) as z:
            tag = np.asarray(z["set_tag"])
            fit = tag == 1
            row_idx = np.asarray(z["row_idx"], dtype=np.int64)
            ci = np.asarray(z["ci"], dtype=np.int64)
            fit_rows_parts.append(row_idx[fit])
            fit_ci_parts.append(ci[fit])
            off = np.asarray(z["idx_off"], dtype=np.int64)
            idx = np.asarray(z["ans_idx"], dtype=np.int64)
            vmax = np.asarray(z["ans_max"], dtype=np.float32)
            keep = np.repeat(fit, off)
            ik = idx[keep]
            fpos = pos[ik]
            hit = fpos >= 0
            if hit.any():
                rows_rep = np.repeat(row_idx, off)[keep][hit]
                ci_rep = np.repeat(ci, off)[keep][hit]
                _reservoir_update(state, fpos[hit], rows_rep, ci_rep, vmax[keep][hit], rng)
        _log(
            f"[passA] shard {i + 1}/{len(shards)} {p.name} "
            f"elapsed={time.time() - t0:.0f}s rss={_rss_gb():.2f}GiB"
        )
        state["next_shard"] = i + 1
        if (i + 1) % RESERVOIR_CKPT_EVERY == 0:
            tmp = ckpt.parent / f".tmp_{ckpt.name}"
            np.savez(
                tmp,
                **{k: v for k, v in state.items() if k != "next_shard"},
                next_shard=np.int64(state["next_shard"]),
                fit_rows=np.concatenate(fit_rows_parts),
                fit_ci=np.concatenate(fit_ci_parts),
            )
            os.replace(tmp, ckpt)
            _log(f"[passA] checkpoint at shard {i + 1}")

    fit_rows = np.concatenate(fit_rows_parts) if fit_rows_parts else np.empty(0, np.int64)
    fit_ci = np.concatenate(fit_ci_parts) if fit_ci_parts else np.empty(0, np.int64)
    uniq_rows, uniq_pos = np.unique(fit_rows, return_index=True)
    row_to_ci = dict(zip(uniq_rows.tolist(), fit_ci[uniq_pos].tolist(), strict=True))

    # ── per-feature draws (16,384-iteration metadata loop; no token data) ──
    sel_rows: list[dict] = []
    inv: list[tuple[int, int, int, int, int, int, int]] = []  # row,ci,feat,kind,bin,split,order
    n_short = 0
    draw_rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 7]))
    for fi in range(n_feat):
        valid = state["res_row"][fi] >= 0
        rows_f = state["res_row"][fi][valid]
        ci_f = state["res_ci"][fi][valid]
        val_f = state["res_val"][fi][valid]
        # dedup by row (defensive; each (row, feat) appears once per stream)
        _, keep_i = np.unique(rows_f, return_index=True)
        rows_f, ci_f, val_f = rows_f[keep_i], ci_f[keep_i], val_f[keep_i]
        act, borrows = stratified_bin_draw(rows_f, ci_f, val_f, draw_rng)
        short = len(act) < CM.N_ACT_BINS * CM.ACT_PER_BIN
        n_short += int(short)
        act_rows_set = set(state["res_row"][fi][valid].tolist())
        pool = uniq_rows[~np.isin(uniq_rows, rows_f, assume_unique=True)]
        take = min(NONACT_SPARES, len(pool))
        nonact_rows = draw_rng.choice(pool, size=take, replace=False)
        nonact = [
            {"row": int(r), "ci": int(row_to_ci[int(r)]), "order": o}
            for o, r in enumerate(nonact_rows)
        ]
        sel_rows.append(
            {
                "feat_id": int(fid[fi]),
                "restricted_idx": fi,
                "n_candidates": int(state["n_seen"][fi]),
                "n_reservoir": int(valid.sum()),
                "act": act,
                "nonact_candidates": nonact,
                "borrow_events": borrows,
                "act_short": bool(short),
                "reservoir_covers_all": bool(state["n_seen"][fi] <= CM.RESERVOIR_PER_FEATURE),
                "n_active_rows_in_reservoir": len(act_rows_set),
            }
        )
        for a in act:
            inv.append((a["row"], a["ci"], int(fid[fi]), 0, a["bin"], a["split"], 0))
        for na in nonact:
            inv.append((na["row"], na["ci"], int(fid[fi]), 1, -1, -1, na["order"]))

    inv_arr = np.asarray(inv, dtype=np.int64)
    inv_arr = inv_arr[np.argsort(inv_arr[:, 0], kind="stable")]
    np.savez(
        sel_dir / "inverted_index.npz",
        row=inv_arr[:, 0],
        ci=inv_arr[:, 1],
        feat=inv_arr[:, 2],
        kind=inv_arr[:, 3],
        bin=inv_arr[:, 4],
        split=inv_arr[:, 5],
        order=inv_arr[:, 6],
    )
    # 200 seeded random unit directions (activation space) — Pass-B control arm
    dir_rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 99]))
    dirs = dir_rng.standard_normal((CM.N_RANDOM_DIRECTIONS, args.act_dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    np.savez(sel_dir / "random_directions.npz", directions=dirs.astype(np.float32))

    CM.write_jsonl_sharded(sel_rows, sel_dir, "selection")
    union_rows = np.unique(inv_arr[:, 0])
    meta = {
        **CM.repro_meta(),
        "n_features": n_feat,
        "n_shards": len(shards),
        "n_fit_rows": int(len(uniq_rows)),
        "union_rows": int(len(union_rows)),
        "n_act_short": n_short,
        "reservoir_size": R,
        "nonact_spares": NONACT_SPARES,
        "fingerprint": fp,
    }
    (sel_dir / "selection_meta.json").write_text(json.dumps(meta, indent=1))
    (sel_dir / "DONE.json").write_text(json.dumps({"pass": "select", **meta}))
    _log(
        f"[passA] done: {n_feat} features, union_rows={len(union_rows)}, "
        f"act_short={n_short} rss={_rss_gb():.2f}GiB"
    )
    if not args.no_upload:
        upload_selection(sel_dir)
    return 0


def upload_selection(sel_dir: Path) -> int:
    """Upload the Pass-A selection dir to the Hub as ONE bulk `upload_folder`
    commit (#664/#1547 — never a per-file loop), then verify the EXACT file
    set with one prefix-scoped listing (fail-loud). Idempotent: when every
    local file is already present under the prefix, skip the upload (re-runs
    otherwise overwrite — same paths, content-hash-deduped server-side).
    Crash-fix r3 (#1773): Pass B runs on GCE, which materializes only the git
    clone (`data/` is gitignored) — the selection MUST live on the Hub or the
    pilot dies at input load (the #734/#1434 cross-machine-input class)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    done = sel_dir / "DONE.json"
    assert done.exists(), f"selection upload requires a completed Pass A ({done} missing)"
    local = sorted(
        p for p in sel_dir.iterdir() if p.is_file() and not p.name.startswith((".tmp_", ".hfstage"))
    )
    assert local, f"no selection files under {sel_dir}"
    expected = sorted(f"{CM.HF_SELECTION_PREFIX}/{p.name}" for p in local)
    api = HfApi()
    missing = hub.verify_repo_paths_uploaded(
        api, CM.HF_DATA_REPO, expected, path_in_repo=CM.HF_SELECTION_PREFIX, repo_type="dataset"
    )
    if not missing:
        _log(f"[passA] selection already on Hub ({len(expected)} files) — skip upload")
        return 0
    t0 = time.time()
    hub.retry_transient(
        lambda: api.upload_folder(
            folder_path=str(sel_dir),
            repo_id=CM.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=CM.HF_SELECTION_PREFIX,
            ignore_patterns=[".tmp_*", ".hfstage*"],
        ),
        what=f"selection upload ({len(local)} files)",
    )
    still = hub.verify_repo_paths_uploaded(
        api, CM.HF_DATA_REPO, expected, path_in_repo=CM.HF_SELECTION_PREFIX, repo_type="dataset"
    )
    if still:
        raise RuntimeError(
            f"[passA] selection upload verify FAILED: {len(still)} missing, "
            f"e.g. {sorted(still)[:5]}"
        )
    _log(
        f"[passA] selection uploaded: {len(local)} files -> "
        f"{CM.HF_DATA_REPO}:{CM.HF_SELECTION_PREFIX} in {time.time() - t0:.1f}s"
    )
    return 0


def stage_selection(args) -> int:
    """Stage the Pass-A selection from the Hub into the consumer-FLAT SEL_DIR
    layout (crash-fix r3, #1773 — the fix-engaged `[stage] selection staged:`
    line). Idempotent: a locally-present inverted_index.npz skips the network
    entirely. Scoped listing + per-file retried atomic staging at ONE resolved
    revision (the #833 recipe via hub.stage_hub_file — NEVER snapshot_download
    against the ~1M-file data repo); files land FLAT at sel_dir/<name>, the
    consumer's own layout (artifact-reuse (h)(iv) — a verbatim prefix mirror
    would bury them under issue1773_featurepipeline/selection/). The
    passA_ckpt_* resume checkpoint is deliberately not staged (no Pass-B
    consumer reads it; ~218 MB saved per fresh instance). Fail-loud on an
    empty prefix."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    sel_dir = args.selection_dir
    marker = sel_dir / "inverted_index.npz"
    if marker.exists():
        _log(f"[stage] selection present locally -> {sel_dir} (skip staging)")
        return 0
    api = HfApi()
    info = hub.retry_transient(
        lambda: api.repo_info(CM.HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info({CM.HF_DATA_REPO})",
    )
    rev = str(info.sha)
    files = hub.list_hf_files_under_path(
        api, CM.HF_DATA_REPO, CM.HF_SELECTION_PREFIX, repo_type="dataset", revision=rev
    )
    files = sorted(f for f in files if not Path(f).name.startswith("passA_ckpt_"))
    if not files:
        raise FileNotFoundError(
            f"no selection files under {CM.HF_DATA_REPO}:{CM.HF_SELECTION_PREFIX} — "
            "run Pass A (or `--pass select --upload-only`) first"
        )
    sel_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        hub.stage_hub_file(
            CM.HF_DATA_REPO, f, sel_dir / Path(f).name, repo_type="dataset", revision=rev
        )
    assert marker.exists(), f"staging did not produce {marker}"
    _log(f"[stage] selection staged: {len(files)} files -> {sel_dir}")
    return 0


def stratified_bin_draw(
    rows_f: np.ndarray, ci_f: np.ndarray, val_f: np.ndarray, rng: np.random.Generator
) -> tuple[list[dict], int]:
    """10-quantile-bin stratified draw of 60 rows (6/bin: 4 evidence + 2 holdout)
    over the feature's own ans_max distribution (reservoir estimate); bins with
    <6 candidates borrow from adjacent bins (borrow count returned). Seeded +
    deterministic given (rows, vals, rng state)."""
    n = len(rows_f)
    if n == 0:
        return [], 0
    edges = np.quantile(val_f, np.linspace(0, 1, CM.N_ACT_BINS + 1)[1:-1])
    bins = np.searchsorted(edges, val_f, side="right")
    picked: list[dict] = []
    used = np.zeros(n, dtype=bool)
    borrows = 0
    for b in range(CM.N_ACT_BINS):
        want = CM.ACT_PER_BIN
        cand = np.where((bins == b) & ~used)[0]
        take = rng.choice(cand, size=min(want, len(cand)), replace=False)
        chosen = list(take)
        # borrow from adjacent bins outward when thin (recorded)
        radius = 1
        while len(chosen) < want and radius < CM.N_ACT_BINS:
            for nb in (b - radius, b + radius):
                if len(chosen) >= want or not (0 <= nb < CM.N_ACT_BINS):
                    continue
                extra = np.where((bins == nb) & ~used)[0]
                extra = extra[~np.isin(extra, chosen)]
                if len(extra):
                    m = min(want - len(chosen), len(extra))
                    chosen.extend(rng.choice(extra, size=m, replace=False).tolist())
                    borrows += m
            radius += 1
        for k_i, idx0 in enumerate(chosen):
            used[idx0] = True
            picked.append(
                {
                    "row": int(rows_f[idx0]),
                    "ci": int(ci_f[idx0]),
                    "val": float(val_f[idx0]),
                    "bin": b,
                    # 4 evidence + 2 holdout per bin (holdout NEVER shown to
                    # the describer/categorizer — plan design invariant)
                    "split": 0 if k_i < CM.ACT_EVIDENCE_PER_BIN else 1,
                }
            )
    return picked, borrows


# ── Pass B: window extraction (GPU) ──────────────────────────────────────────


def _load_sae(args):
    """Pinned Hub SAE (production) or a small same-key-set state dict (tiny e2e)."""
    import issue1482_sae as S
    import torch

    if args.sae_state:
        sd = torch.load(args.sae_state, map_location="cpu", weights_only=True)
        return S.BatchTopKSAE(
            sd, k=args.k, device=args.device, act_dim=args.act_dim, dict_size=args.dict_size
        )
    return S.BatchTopKSAE.load(k=args.k, device=args.device, cache_dir=args.scratch / "sae")


def _window_record(
    tok, full_ids: list[int], peak: int, ans_start: int, values: np.ndarray, mark: bool = True
) -> dict:
    """Decode a [peak-WINDOW_BACK, peak+WINDOW_FWD] window clipped to the answer
    span, with the peak token <<marked>>. Per-token activation values ride as
    fp16 metadata (omitted from prompts). Edge clipping keeps the window valid
    at answer boundaries (BPE-seam test target)."""
    lo = max(ans_start, peak - CM.WINDOW_BACK)
    hi = min(len(full_ids), peak + CM.WINDOW_FWD + 1)
    ids = full_ids[lo:hi]
    plain = tok.decode(ids)
    if mark:
        pre = tok.decode(full_ids[lo:peak])
        pk = tok.decode([full_ids[peak]])
        post = tok.decode(full_ids[peak + 1 : hi])
        marked = f"{pre}<<{pk}>>{post}"
    else:
        marked = plain
    return {
        "text_marked": marked,
        "text_plain": plain,
        "token_lo": int(lo),
        "token_hi": int(hi),
        "peak_pos": int(peak),
        "values_fp16": [float(np.float16(v)) for v in values[lo - ans_start : hi - ans_start]],
    }


def _iter_chunk_rows(args, name: str, needed_ci: dict[int, int]):
    """Yield (row_idx, ci, prompt, response) for one chunk — local dir (tiny e2e)
    or the parent HF download path (_iter_needed_rows)."""
    if args.local_chunks:
        rows = json.loads((args.local_chunks / name).read_text())["rows"]
        for r in rows:
            ci = int(r["ci"])
            if ci in needed_ci:
                yield needed_ci[ci], ci, r["prompt"], r["response"]
        return
    import issue1482_error_analysis as EA

    ns = SimpleNamespace(scratch=args.scratch)
    for _name, keep in EA._iter_needed_rows(ns, [name], needed_ci):
        yield from keep


def _upload_pending(pending: list[str], out_dir: Path, args, t_upload: list[float]) -> None:
    """One batched upload_folder commit for the pending chunk files (per-cell
    upload rule #664; batched every --upload-every to respect the fleet-shared
    commit budget). Fail-loud via hub.retry_transient."""
    if not pending or args.no_upload:
        pending.clear()
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    t0 = time.time()
    pats = list(pending)
    hub.assert_hub_dir_filecounts(out_dir, f"{CM.HF_PREFIX}/raw_windows", allow_patterns=pats)
    hub.retry_transient(
        lambda: HfApi().upload_folder(
            folder_path=str(out_dir),
            repo_id=CM.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{CM.HF_PREFIX}/raw_windows",
            allow_patterns=pats,
        ),
        what=f"raw_windows batch upload ({len(pats)} files)",
    )
    dt = time.time() - t0
    t_upload.append(dt)
    _log(f"[evidence] uploaded {len(pats)} chunk files in {dt:.1f}s")
    pending.clear()


def pass_windows(args) -> int:
    """Pass B worker: extract peak-marked windows for the selected (row, feat)
    pairs of this worker's chunk shard; verify non-activating spans; maintain
    random-direction inline top-K. Per-chunk JSONL + done sentinel + resume."""
    import issue1482_error_analysis as EA
    import torch

    sel_dir = args.selection_dir
    z = np.load(sel_dir / "inverted_index.npz", allow_pickle=False)
    inv = {k: np.asarray(z[k]) for k in ("row", "ci", "feat", "kind", "bin", "split", "order")}
    needed_ci = {int(c): int(r) for c, r in zip(inv["ci"], inv["row"], strict=True)}
    by_row: dict[int, list[int]] = {}
    for i, r in enumerate(inv["row"]):
        by_row.setdefault(int(r), []).append(i)

    dirs = np.load(sel_dir / "random_directions.npz", allow_pickle=False)["directions"]
    if args.act_dim != dirs.shape[1]:  # tiny e2e: regenerate at the tiny width
        dir_rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 99]))
        dirs = dir_rng.standard_normal((CM.N_RANDOM_DIRECTIONS, args.act_dim)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_t = torch.tensor(dirs, dtype=torch.float32)

    ns = SimpleNamespace(
        tiny_model=args.tiny_model,
        device=args.device,
        max_chunks=args.max_chunks,
        scratch=args.scratch,
    )
    model, tok = EA._load_model_tok(ns)
    sae = _load_sae(args)
    prefix_chars = EA._prefix_char_len(tok)
    if args.local_chunks:
        names = sorted(p.name for p in args.local_chunks.glob("*.json"))
        if args.max_chunks > 0:
            names = names[: args.max_chunks]
    else:
        names = EA._raw_chunk_names(ns)
    names = names[args.worker :: args.n_workers]
    if args.pilot:
        names = names[:1]

    fp = _config_fingerprint(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    span_rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 3, args.worker]))
    rand_top: list[list[tuple[float, dict]]] = [[] for _ in range(len(dirs))]
    pending_upload: list[str] = []
    skipped_files: list[str] = []
    # Bind BEFORE the loop: a resume where every chunk skips (done-sentinel
    # fingerprint match) must not leave `rows` unbound at the --pilot report
    # (r1 concern passb-pilot-resume-nameerror; rows=0 then means "pilot chunk
    # already done, nothing re-processed").
    rows: list[tuple] = []
    t_upload: list[float] = []
    t0 = time.time()
    n_done = 0

    for k_i, name in enumerate(names):
        stem = Path(name).stem
        chunk_out = out_dir / f"windows_{stem}.jsonl"
        done_f = out_dir / f"windows_{stem}.done.json"
        if done_f.exists() and not args.no_resume:
            try:
                if json.loads(done_f.read_text()).get("fingerprint") == fp:
                    _log(f"[evidence] chunk {k_i + 1}/{len(names)} {name} SKIP (done)")
                    skipped_files.extend([chunk_out.name, done_f.name])
                    continue
            except json.JSONDecodeError:
                pass
        rows = []
        for row_idx, ci, prompt, response in _iter_chunk_rows(args, name, needed_ci):
            tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
            if tk is None:
                continue
            full_ids, prefix_end, context_end, n_ans, seam = tk
            rows.append((row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam))
        rows.sort(key=lambda r: len(r[2]))
        out_rows: list[dict] = []
        t_chunk = time.time()
        for s in range(0, len(rows), args.gen_batch):
            batch = rows[s : s + args.gen_batch]
            caps = EA._batched_capture(model, tok, batch, (args.layer,), args.device)
            for (row_idx, ci, full_ids, _pe, context_end, _na, _seam), cap in zip(
                batch, caps, strict=True
            ):
                h = cap[args.layer]
                out_rows.extend(
                    _extract_row_windows(
                        args,
                        tok,
                        sae,
                        dirs_t,
                        rand_top,
                        inv,
                        by_row,
                        row_idx,
                        ci,
                        full_ids,
                        context_end,
                        h,
                        span_rng,
                    )
                )
        tmp = chunk_out.parent / f".tmp_{chunk_out.name}"
        with tmp.open("w", encoding="utf-8") as fh:
            for r in out_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, chunk_out)
        done_f.write_text(json.dumps({"chunk": name, "fingerprint": fp, "n": len(out_rows)}))
        pending_upload.extend([chunk_out.name, done_f.name])
        n_done += 1
        _log(
            f"[evidence] chunk {n_done}/{len(names)} {name} rows={len(out_rows)} "
            f"chunk_s={time.time() - t_chunk:.1f} elapsed={time.time() - t0:.0f}s "
            f"rss={_rss_gb():.2f}GiB"
        )
        if len(pending_upload) >= 2 * args.upload_every:
            _upload_pending(pending_upload, out_dir, args, t_upload)
        if args.pilot:
            break
    if skipped_files and not args.no_upload:
        # Resume reconciliation (r1 review Minor): a crash between a chunk's
        # done-sentinel write and the next batched upload leaves local-done
        # chunks off the Hub; re-queue exactly the missing ones from ONE
        # scoped listing (never a blanket re-upload of every skipped chunk).
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        prefix = f"{CM.HF_PREFIX}/raw_windows"
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            CM.HF_DATA_REPO,
            [f"{prefix}/{n}" for n in skipped_files],
            path_in_repo=prefix,
            repo_type="dataset",
        )
        if missing:
            requeue = [p.rsplit("/", 1)[1] for p in missing]
            _log(
                f"[evidence] resume reconciliation: {len(requeue)} skipped-chunk "
                "files absent from Hub -> re-queued for upload"
            )
            pending_upload.extend(requeue)
    _upload_pending(pending_upload, out_dir, args, t_upload)

    # random-direction per-worker top-K (window texts inline — recorded choice)
    rt = [
        [{"val": float(v), **w} for v, w in sorted(top, key=lambda t: -t[0])[: CM.RAND_TOP_K]]
        for top in rand_top
    ]
    rd_path = out_dir / f"randdir_worker{args.worker}.json"
    rd_path.write_text(json.dumps({"worker": args.worker, "fingerprint": fp, "top": rt}))
    if not args.no_upload:
        _upload_pending([rd_path.name], out_dir, args, t_upload)
    if args.pilot:
        n_rows_pilot = len(rows) if names else 0
        wall = time.time() - t0
        up = sum(t_upload)
        proj_h = wall * (CM.N_SHARDS / max(args.n_workers, 1)) / 3600.0
        _log(
            f"[evidence] PILOT: chunk_wall={wall:.1f}s (upload={up:.1f}s separately) "
            f"rows={n_rows_pilot} -> projected full pass "
            f"~{proj_h:.2f}h/worker at width {args.n_workers}"
        )
        (out_dir / "pilot_report.json").write_text(
            json.dumps(
                {
                    "chunk_wall_s": wall,
                    "upload_s": up,
                    "projected_hours_per_worker": proj_h,
                    "n_workers": args.n_workers,
                    **CM.repro_meta(),
                }
            )
        )
    _log(f"[evidence] worker {args.worker} done: {n_done} chunks")
    return 0


def _extract_row_windows(
    args, tok, sae, dirs_t, rand_top, inv, by_row, row_idx, ci, full_ids, context_end, h, span_rng
) -> list[dict]:
    """Per-row extraction: SAE-encode answer tokens once (chunked GEMM), then
    per selected (feat, kind) entry take the peak window / verify non-activating
    span; update random-direction top-K. Reference token-pool parity: peak
    selection over BOS-stripped inlier answer tokens (issue1482_sae semantics)."""
    import issue1482_sae as S
    import torch

    ans_start = context_end + 1
    h_ans = h[ans_start:]
    if h_ans.shape[0] == 0:
        return []
    keep = S.token_inlier_mask(h)
    keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
    ans_keep = keep[ans_start:].numpy()
    if not ans_keep.any():
        ans_keep = np.ones(h_ans.shape[0], dtype=bool)  # parent ans_all_out fallback
    f_ans = sae.encode(h_ans).cpu().numpy()  # (T_ans, dict)
    out: list[dict] = []
    for ii in by_row.get(int(row_idx), []):
        feat = int(inv["feat"][ii])
        kind = int(inv["kind"][ii])
        base = {
            "feat_id": feat,
            "row_idx": int(row_idx),
            "ci": int(ci),
            "bin": int(inv["bin"][ii]),
            "split": int(inv["split"][ii]),
            "order": int(inv["order"][ii]),
        }
        acts = f_ans[:, feat]
        if kind == 0:
            masked = np.where(ans_keep, acts, -np.inf)
            peak_rel = int(np.argmax(masked))
            rec = _window_record(tok, full_ids, ans_start + peak_rel, ans_start, acts)
            out.append({**base, "kind": "act", "peak_val": float(acts[peak_rel]), "window": rec})
        else:
            active_any = bool((acts > 0).any())
            span_ok, rec = _nonact_span(tok, full_ids, ans_start, acts, span_rng)
            out.append(
                {
                    **base,
                    "kind": "nonact",
                    "row_active_for_feat": active_any,
                    "verify_failed": not span_ok,
                    "window": rec,
                }
            )
    # random-direction dots over kept answer tokens
    hk = h_ans[torch.tensor(ans_keep)]
    if hk.shape[0]:
        dots = (hk.float() @ dirs_t.T).numpy()  # (T_kept, n_dirs)
        kept_pos = np.where(ans_keep)[0]
        d_peak = dots.argmax(0)
        d_val = dots.max(0)
        for d in range(dots.shape[1]):
            top = rand_top[d]
            floor = top[-1][0] if len(top) >= CM.RAND_TOP_K else -np.inf
            if d_val[d] > floor:
                peak_abs = ans_start + int(kept_pos[d_peak[d]])
                rec = _window_record(tok, full_ids, peak_abs, ans_start, dots[:, d][kept_pos])
                rec = {k: v for k, v in rec.items() if k != "values_fp16"}
                top.append((float(d_val[d]), {"row_idx": int(row_idx), "ci": int(ci), **rec}))
                top.sort(key=lambda t: -t[0])
                del top[CM.RAND_TOP_K :]
    return out


def _nonact_span(tok, full_ids, ans_start, acts, span_rng) -> tuple[bool, dict]:
    """Seeded random 32-token answer span, verified feature-silent (all window
    tokens at 0 after the thresholded ReLU); one span re-draw on violation."""
    T = len(acts)
    w = CM.WINDOW_BACK + CM.WINDOW_FWD + 1
    for _ in range(1 + SPAN_RESAMPLE_TRIES):
        lo_rel = 0 if T <= w else int(span_rng.integers(0, T - w + 1))
        hi_rel = min(T, lo_rel + w)
        if not (acts[lo_rel:hi_rel] > 0).any():
            ids = full_ids[ans_start + lo_rel : ans_start + hi_rel]
            plain = tok.decode(ids)
            return True, {
                "text_marked": plain,
                "text_plain": plain,
                "token_lo": int(ans_start + lo_rel),
                "token_hi": int(ans_start + hi_rel),
                "peak_pos": -1,
                "values_fp16": [],
            }
    ids = full_ids[ans_start : ans_start + w]
    plain = tok.decode(ids)
    return False, {
        "text_marked": plain,
        "text_plain": plain,
        "token_lo": int(ans_start),
        "token_hi": int(ans_start + len(ids)),
        "peak_pos": -1,
        "values_fp16": [],
    }


# ── Pass C: assembly ─────────────────────────────────────────────────────────


def pass_assemble(args) -> int:
    """Join per-chunk window files into per-feature evidence packets + held-out
    scoring sets + completeness report (H2 gate: >=99% full-fill else reported;
    shortfalls flagged `evidence_short`, never silently backfilled)."""
    sel_dir = args.selection_dir
    win_dir = args.out_dir
    if args.fetch_missing:
        from explore_persona_space.orchestrate import hub

        hub.stage_hub_prefix(
            CM.HF_DATA_REPO, f"{CM.HF_PREFIX}/raw_windows", win_dir, repo_type="dataset"
        )
        # stage_hub_prefix mirrors the repo-relative prefix verbatim, so files land
        # NESTED at win_dir/<HF_PREFIX>/raw_windows/ — resolve to that dir when the
        # flat layout is empty (staged-layout check (h)(iv), #928/#1481).
        nested = win_dir / CM.HF_PREFIX / "raw_windows"
        if nested.is_dir() and not list(win_dir.glob("windows_*.jsonl")):
            win_dir = nested
    sel: dict[int, dict] = {}
    for p in sorted(sel_dir.glob("selection.shard*.jsonl")):
        for r in CM.iter_jsonl(p):
            sel[int(r["feat_id"])] = r
    if not sel:
        raise RuntimeError(
            f"[assemble] no selection.shard*.jsonl under {sel_dir} — point --selection-dir "
            "(or EPM_1773_SEL_DIR) at the Pass-A output; assembling now would emit 0 packets "
            "with a vacuous fill=1.0 report"
        )
    windows: dict[int, dict[str, list[dict]]] = {f: {"act": [], "nonact": []} for f in sel}
    win_files = sorted(win_dir.glob("windows_*.jsonl"))
    if not win_files:
        raise RuntimeError(
            f"[assemble] no windows_*.jsonl under {win_dir} — run --pass windows first "
            "(or pass --fetch-missing to stage them from the Hub); assembling now would "
            "emit all-short packets with fill=0.0"
        )
    for p in win_files:
        for r in CM.iter_jsonl(p):
            f = int(r["feat_id"])
            if f in windows:
                windows[f][r["kind"]].append(r)

    # phase0 joins: neighbours + logit footprint + density/persist (STAT)
    p0_table = args.phase0_dir / "feature_table.jsonl"
    p0: dict[int, dict] = {}
    if p0_table.exists():
        for r in CM.iter_jsonl(p0_table):
            p0[int(r["feat_id"])] = r
    else:
        _log(f"[assemble] WARNING: phase0 table missing at {p0_table}; OUT/NEAR omitted")

    packets: list[dict] = []
    holdouts: list[dict] = []
    completeness: list[dict] = []
    rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 5]))
    for feat_id, srow in sorted(sel.items()):
        acts = sorted(windows[feat_id]["act"], key=lambda r: (r["bin"], -r.get("peak_val", 0)))
        ev_pos = [dict(r["window"], bin=r["bin"], ci=r["ci"]) for r in acts if r["split"] == 0]
        ho_pos = [dict(r["window"], bin=r["bin"], ci=r["ci"]) for r in acts if r["split"] == 1]
        nonact_ok = sorted(
            (r for r in windows[feat_id]["nonact"] if not r.get("verify_failed")),
            key=lambda r: r["order"],
        )
        ev_neg = [dict(r["window"], ci=r["ci"]) for r in nonact_ok[: CM.N_NONACT_EVIDENCE]]
        ho_neg = [
            dict(r["window"], ci=r["ci"])
            for r in nonact_ok[CM.N_NONACT_EVIDENCE : CM.N_NONACT_EVIDENCE + CM.N_NONACT_HOLDOUT]
        ]
        short = (
            len(ev_pos) < CM.N_ACT_EVIDENCE
            or len(ho_pos) < CM.N_ACT_HOLDOUT
            or len(ev_neg) < CM.N_NONACT_EVIDENCE
            or len(ho_neg) < CM.N_NONACT_HOLDOUT
        )
        completeness.append(
            {
                "feat_id": feat_id,
                "n_ev_pos": len(ev_pos),
                "n_ho_pos": len(ho_pos),
                "n_ev_neg": len(ev_neg),
                "n_ho_neg": len(ho_neg),
                "n_nonact_verify_failed": sum(
                    1 for r in windows[feat_id]["nonact"] if r.get("verify_failed")
                ),
                "evidence_short": bool(short),
            }
        )
        pk = {
            "feat_id": feat_id,
            "restricted_idx": srow["restricted_idx"],
            "ex_pos": ev_pos[: CM.N_ACT_EVIDENCE],
            "ex_neg": ev_neg,
            "near": [],
            "out": (p0.get(feat_id) or {}).get("logit_footprint"),
            "stat": {
                "density": (p0.get(feat_id) or {}).get("density"),
                "persist_answer": (p0.get(feat_id) or {}).get("persist_answer"),
                "tier": None,
            },
            "evidence_short": bool(short),
        }
        packets.append(pk)
        holdouts.append(
            {"feat_id": feat_id, "ho_pos": ho_pos[: CM.N_ACT_HOLDOUT], "ho_neg": ho_neg}
        )

    # near-miss: 5 windows from the top-3 neighbours' own top-bin evidence
    by_feat = {p["feat_id"]: p for p in packets}
    for pk in packets:
        nb = (p0.get(pk["feat_id"]) or {}).get("neighbors", {}).get("feat_ids", [])[:3]
        near: list[dict] = []
        for j, nfid in enumerate(nb):
            src = by_feat.get(int(nfid))
            if not src or not src["ex_pos"]:
                continue
            pool = sorted(src["ex_pos"], key=lambda w: -int(w.get("bin", 0)))
            take = 2 if j < 2 else 1
            for w in pool[:take]:
                if len(near) < CM.N_NEAR_MISS:
                    near.append(dict(w, near_source_feat=int(nfid)))
        pk["near"] = near

    ev_dir = args.evidence_dir
    CM.write_jsonl_sharded(
        packets,
        ev_dir / "evidence_manifests",
        "evidence",
    )
    CM.write_jsonl_sharded(holdouts, ev_dir / "holdout", "holdout")

    # random-direction merge: per-worker top-K -> global top-K per direction
    rd_workers = sorted(win_dir.glob("randdir_worker*.json"))
    merged: list[list[dict]] = [[] for _ in range(CM.N_RANDOM_DIRECTIONS)]
    for p in rd_workers:
        doc = json.loads(p.read_text())
        for d, top in enumerate(doc["top"]):
            merged[d].extend(top)
    rd_packets = []
    for d, top in enumerate(merged):
        top = sorted(top, key=lambda t: -t["val"])[: CM.RAND_TOP_K]
        ev = [dict(w, bin=CM.N_ACT_BINS - 1) for w in top[: CM.N_ACT_EVIDENCE]]
        ho = [
            dict(w, bin=CM.N_ACT_BINS - 1)
            for w in top[CM.N_ACT_EVIDENCE : CM.N_ACT_EVIDENCE + CM.N_ACT_HOLDOUT]
        ]
        bg = _randdir_background(holdouts, {w.get("row_idx") for w in top}, rng)
        rd_packets.append(
            {
                "feat_id": -(d + 1),  # negative ids mark control features
                "control": "random_direction",
                "ex_pos": ev,
                "ex_neg": bg[: CM.N_NONACT_EVIDENCE],
                "near": [],
                "out": None,
                "stat": None,
                "ho_pos": ho,
                "ho_neg": bg[CM.N_NONACT_EVIDENCE : CM.N_NONACT_EVIDENCE + CM.N_NONACT_HOLDOUT],
                "evidence_short": len(ev) < CM.N_ACT_EVIDENCE,
            }
        )
    if rd_packets:
        CM.write_jsonl_sharded(rd_packets, ev_dir / "evidence_manifests", "evidence_randdir")

    n_short = sum(1 for c in completeness if c["evidence_short"])
    fill_frac = 1.0 - n_short / max(len(completeness), 1)
    report = {
        **CM.repro_meta(),
        "n_features": len(completeness),
        "n_evidence_short": n_short,
        "fill_fraction": fill_frac,
        "h2_gate_99pct": bool(fill_frac >= 0.99),
        "per_feature": completeness,
    }
    (ev_dir / "completeness_report.json").write_text(json.dumps(report, indent=1))
    (ev_dir / "ASSEMBLY_DONE.json").write_text(
        json.dumps({"pass": "assemble", "n_features": len(completeness), **CM.repro_meta()})
    )
    if not args.no_upload:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        # fnmatch `**/*.json` requires a `/`, so top-level files (completeness_report.json,
        # ASSEMBLY_DONE.json) need the bare `*.json` pattern too (#825 eligibility class).
        ev_pats = ["**/*.jsonl", "**/*.json", "*.json"]
        hub.assert_hub_dir_filecounts(ev_dir, f"{CM.HF_PREFIX}/evidence", allow_patterns=ev_pats)
        hub.retry_transient(
            lambda: HfApi().upload_folder(
                folder_path=str(ev_dir),
                repo_id=CM.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{CM.HF_PREFIX}/evidence",
                allow_patterns=ev_pats,
            ),
            what="evidence manifests upload",
        )
    _log(
        f"[assemble] done: {len(packets)} packets + {len(rd_packets)} randdir; "
        f"fill={fill_frac:.4f} (short={n_short})"
    )
    return 0


def _randdir_background(holdouts: list[dict], exclude_rows: set, rng) -> list[dict]:
    """Background non-activating windows for random-direction controls: seeded
    draw from the REAL features' verified non-activating holdout pool, excluding
    the direction's own top rows (recorded design choice — a random direction
    has no pooled-store candidate list to invert)."""
    pool = [w for h in holdouts for w in h["ho_neg"] if w.get("row_idx") not in exclude_rows]
    if not pool:
        return []
    take = min(CM.N_NONACT, len(pool))
    idx = rng.choice(len(pool), size=take, replace=False)
    return [pool[i] for i in idx]


# ── entrypoint ───────────────────────────────────────────────────────────────


def _import_check() -> int:
    """Axis-1 import-resolution leg: execute every deferred import this
    entrypoint uses on its REAL branches (preferred shape (a))."""
    import issue1482_error_analysis as EA  # noqa: F401
    import issue1482_sae as S  # noqa: F401
    import torch  # noqa: F401
    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401

    from explore_persona_space.orchestrate import hub  # noqa: F401

    for sym in (
        EA._tokenize_row,
        EA._batched_capture,
        EA._row_features,
        EA._raw_chunk_names,
        EA._load_model_tok,
        EA._iter_needed_rows,
        EA._prefix_char_len,
        S.BatchTopKSAE.load,
        S.token_inlier_mask,
        hub.retry_transient,
        hub.stage_hub_prefix,
        hub.stage_hub_file,
        hub.list_hf_files_under_path,
        hub.verify_repo_paths_uploaded,
    ):
        assert callable(sym), sym
    print("[import-check] OK: all deferred imports + symbols resolve", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pass", dest="pass_name", choices=("select", "windows", "assemble", "stage-selection")
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--upload-only",
        action="store_true",
        help="Pass A: skip the build; upload an existing --selection-dir to the Hub",
    )
    ap.add_argument("--store", type=Path, default=CM.STORE_DEFAULT)
    ap.add_argument("--selection-dir", type=Path, default=CM.SEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=CM.WORK_DEFAULT / "raw_windows")
    ap.add_argument("--evidence-dir", type=Path, default=CM.WORK_DEFAULT / "evidence")
    ap.add_argument("--phase0-dir", type=Path, default=CM.OUT_EVAL / "phase0")
    ap.add_argument("--scratch", type=Path, default=CM.WORK_DEFAULT / "scratch")
    ap.add_argument("--seed", type=int, default=CM.SEED)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0, help="informational; CVD-pinned by launcher")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--layer", type=int, default=CM.LAYER)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--act-dim", type=int, default=CM.ACT_DIM)
    ap.add_argument("--dict-size", type=int, default=CM.DICT_SIZE)
    ap.add_argument("--sae-state", type=Path, default=None, help="tiny e2e SAE state dict")
    ap.add_argument("--tiny-model", action="store_true")
    ap.add_argument("--local-chunks", type=Path, default=None, help="tiny e2e local chunk dir")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=0, help="Pass A smoke slice")
    ap.add_argument("--feature-limit", type=int, default=0, help="Pass A smoke feature subset")
    ap.add_argument("--pilot", action="store_true", help="Pass B: 1 chunk, timed incl. upload")
    ap.add_argument("--upload-every", type=int, default=20)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--fetch-missing", action="store_true", help="Pass C: stage windows from HF")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    assert args.seed == CM.SEED, f"seed {args.seed} != registered {CM.SEED} (plan §11)"

    if args.import_check:
        rc = _import_check()
        sys.exit(rc)
    if args.pass_name == "select":
        rc = pass_select(args)
    elif args.pass_name == "windows":
        rc = pass_windows(args)
    elif args.pass_name == "assemble":
        rc = pass_assemble(args)
    elif args.pass_name == "stage-selection":
        rc = stage_selection(args)
    else:
        ap.error("--pass required (or --import-check)")
        return 2
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
