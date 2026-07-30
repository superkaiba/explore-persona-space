#!/usr/bin/env python3
"""#1775 fu round (`dedup-refit-pcfold-doubly`) cell 1: n50k dedup refit (P0c).

The banked #779 n50k fitter comparison (`eval_results/issue_779/
fitter-fair-comparison-n50k/n50k_fits.json`) is quoted UNAUDITED: run-1's P0
audit found 297 exact + 35,073 near-dupe train<->target pairs touching
325/1,400 val/test targets (`fold_check/n50k_overlap.json`). This cell REFITS
the four #779 rungs (primal ridge / exact-RBF KRR / MLP w8192 / residual-skip
— recipes, grids and seeds verbatim from `issue779_ffc_n50k_fits.py`) on the
near-dupe-DEDUPED train set, plus an n-MATCHED RANDOM-DROP control (same count
of uniformly-random NON-near-dupe train rows dropped, seed 0) that separates
"fewer rows" from "removed near-dupes". Targets, sha-pinned val/test, layer 19
and device are held fixed at the banked values.

Two stages (fu_run.sh phases F1a / F1b):

  stage  stream-reduce cx_last + v_x at L19 from the HF capture chunks
         (per-chunk download -> slice -> DELETE; prompts extracted from the
         SAME chunks and spot-asserted against the rederived prompt pool),
         reconstruct the byte-identical split (val/test shas hard-asserted ==
         the pinned originals; train sha == the banked split), recompute the
         FULL train-side drop set with the committed `issue1775_fold_check`
         MinHash machinery (run 1 persisted per-target maxima only — its
         near_pairs list truncates at 200), assert the affected-target set ==
         the banked `gate_a.affected_target_ids`, and upload the reduced
         arrays + drop set to HF BEFORE any fit (#825 ordering).
  fits   refit the 4 rungs on the deduped + random-drop splits; paired ROW
         bootstrap gain CIs (the #779 protocol has no group structure —
         labeled); identity+bias AND kNN retrieval per fit (3584->3584, both
         applicable); per-unit JSONL checkpoints + resume.

Refusal-safety (the #779 module contract): no prompt/context TEXT is ever
printed, logged, or written to any output JSON — drop records are index +
jaccard digests only.

Smoke: --smoke caps the stream at 1 chunk, builds a tiny split via the SAME
`fixed_split` gate path, PLANTS one exact + one near-dupe train row (so the
drop + n-matched-control branches execute deterministically), and runs all 4
rungs at tiny n on CPU. Production-n gates (banked-count asserts) are demoted
to log lines at smoke scale (#1345 gate-calibration parity); artifact presence
+ schema asserts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: caps + .env bind BEFORE the heavy imports (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import issue1775_fold_check as FC  # noqa: E402
import issue779_ffc_n50k_fits as N  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from issue1775_common import (  # noqa: E402
    FU_SUB,
    HF_DATA_REPO,
    OUT_HF_PREFIX,
    _r2,
    _upload_dir_verified,
    append_unit,
    atomic_write_json,
    cluster_bootstrap_delta_r2,
    eval_dir,
    identity_bias_predict,
    knn_retrieval,
    load_units,
    out_root,
    result_meta,
    unit_key,
    upload_phase_eval_json,
)

from explore_persona_space.orchestrate import hub  # noqa: E402

LAYER = 19
RUNGS = ("ridge", "krr", "mlp", "residual_skip")
VARIANTS = ("deduped", "random_drop")
# m3: EVERY output-affecting regime key (#722 r3) — krr_grid / mlp_epochs are
# per-rung fields (absent -> unit_key's str(None), so non-KRR resumes never
# invalidate on a KRR grid change); n_boot keys the per-unit bootstrap curve.
FIT_REGIME_KEYS = (
    "variant",
    "rung",
    "layer",
    "smoke",
    "n_train",
    "n_boot",
    "krr_grid",
    "mlp_epochs",
)
PASS_B_HF_PATH = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
BANKED_OVERLAP = (
    Path(__file__).resolve().parents[1]
    / "eval_results"
    / "issue_1775"
    / "fold_check"
    / "n50k_overlap.json"
)
BANKED_FITS = (
    Path(__file__).resolve().parents[1]
    / "eval_results"
    / "issue_779"
    / "fitter-fair-comparison-n50k"
    / "n50k_fits.json"
)
LEXICAL_CAVEAT = (
    "LEXICAL criterion only (exact + char-5-gram Jaccard >= 0.8, MinHash 64-perm — the "
    "round's own audited #779 recipe); a clean verdict licenses only 'no lexical "
    "near-dupes' — semantic paraphrase duplicates remain possible"
)


def fu_arrays_dir() -> Path:
    d = out_root() / "data" / "issue_1775" / "fu_dedup_refit"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── chunk streaming with prompts (mirrors N._stream_n50k_layer + prompt slice) ────


def _chunk_prompts(bundle) -> list[str]:
    p = bundle.get("prompts")
    if p is None:
        raise KeyError("n50k capture chunk carries no 'prompts' field (B2 violated)")
    return [str(x) for x in p]


def _stream_parts_dir(cache_dir: Path, prefix: str, layer: int, max_chunks: int | None) -> Path:
    """Per-chunk checkpoint dir, keyed on EVERY output-affecting stream regime
    key (prefix + layer + max_chunks — the #722-r3 resume-key discipline) so
    smoke (max_chunks=1) and production parts can never cross-resume."""
    key = hashlib.sha256(
        json.dumps({"prefix": prefix, "layer": layer, "max_chunks": max_chunks}).encode()
    ).hexdigest()[:12]
    return cache_dir / f"parts_{key}"


def _ensure_parts_meta(parts: Path, meta: dict) -> None:
    """Fingerprint-gate the parts dir: an existing meta that mismatches (HF
    chunk-list drift under the same regime key) invalidates the cached parts
    LOUDLY — they are a derived, re-streamable cache."""
    parts.mkdir(parents=True, exist_ok=True)
    meta_path = parts / "stream_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        print(f"[f1a] stream parts fingerprint drift — invalidating {parts}", flush=True)
        shutil.rmtree(parts)
        parts.mkdir(parents=True)
    if not meta_path.exists():
        atomic_write_json(meta_path, meta)


def _save_part(parts: Path, i: int, cx: np.ndarray, vx: np.ndarray, prompts: list[str]) -> None:
    """Atomic per-chunk checkpoint: fp16 arrays first, prompts JSON LAST — its
    presence marks the chunk done. Tmp names keep the `.npz` suffix so
    ``np.savez`` cannot append a second one (gotchas.md); uncompressed (#813)."""
    part_npz = parts / f"c{i:04d}.npz"
    tmp = part_npz.with_name(part_npz.stem + ".tmp.npz")
    np.savez(tmp, cx=cx, vx=vx)
    os.replace(tmp, part_npz)
    part_pj = parts / f"c{i:04d}.prompts.json"
    ptmp = part_pj.with_suffix(".tmp")
    ptmp.write_text(json.dumps(prompts), encoding="utf-8")
    os.replace(ptmp, part_pj)


def _load_part(parts: Path, i: int) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load chunk ``i``'s checkpoint, or None when absent/incomplete."""
    part_npz = parts / f"c{i:04d}.npz"
    part_pj = parts / f"c{i:04d}.prompts.json"
    if not (part_npz.exists() and part_pj.exists()):
        return None
    with np.load(part_npz) as z:
        cx, vx = z["cx"], z["vx"]
    return cx, vx, json.loads(part_pj.read_text())


def stream_chunks_with_prompts(
    prefix: str, layer: int, cache_dir: Path, *, max_chunks: int | None = None
):
    """cx_last + v_x at ``layer`` + per-row prompts, stream-reduced from the HF
    capture chunks (download one -> mmap-slice -> DELETE; peak ~one chunk).
    Checkpoint-per-chunk (fp16 parts + prompts shard, ~7.5 MB/chunk) with a
    regime-fingerprinted resume, so a mid-stream crash resumes instead of
    re-streaming from zero (external-stream contract; round-1 fu Major
    `f1a-stream-no-checkpoint-resume`). fp16 part round-trip is exact w.r.t.
    the arrays' terminal fp16 persist (fp16->fp32->fp16 is identity), so
    resumed and fresh streams yield byte-identical persisted arrays.
    Returns (cx, vx, prompts, n_kept, stream_diag). Downloaded-chunk pilot line
    projects the staging wall (plan section 7 kill criterion 1 input)."""
    from huggingface_hub import HfApi, hf_hub_download

    names = sorted(
        f.path.rsplit("/", 1)[-1]
        for f in hub.retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: raw list_repo_tree wrapped in hub.retry_transient here
                HfApi().list_repo_tree(
                    HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
                )
            ),
            what=f"list n50k chunks under {prefix}",
        )
        if getattr(f, "size", None) is not None and f.path.endswith(".pt")
    )
    if not names:
        raise FileNotFoundError(f"no n50k capture chunks under HF {prefix}")
    if max_chunks is not None:
        names = names[: int(max_chunks)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = _stream_parts_dir(cache_dir, prefix, layer, max_chunks)
    _ensure_parts_meta(
        parts, {"prefix": prefix, "layer": layer, "max_chunks": max_chunks, "names": names}
    )
    cx_parts: list[np.ndarray] = []
    vx_parts: list[np.ndarray] = []
    prompts: list[str] = []
    t0 = time.monotonic()
    bytes_seen = 0
    n_dl = 0
    n_resumed = 0
    pilot_logged = False
    for i, name in enumerate(names):
        cached = _load_part(parts, i)
        if cached is not None:
            cxi, vxi, pl = cached
            n_resumed += 1
            print(
                f"[f1a] RESUME unit {i + 1}/{len(names)} chunk={name} (part checkpoint)", flush=True
            )
        else:
            got = Path(
                hub.retry_transient(
                    lambda name=name: hf_hub_download(
                        HF_DATA_REPO,
                        filename=f"{prefix}/{name}",
                        repo_type="dataset",
                        local_dir=cache_dir,
                    ),
                    what=f"n50k chunk {name}",
                )
            )
            bytes_seen += got.stat().st_size
            n_dl += 1
            b = F._mmap_load(got)
            cxi = N._slice_layer(b, "cx_last", layer).astype(np.float16)
            vxi = N._slice_layer(b, "v_x", layer).astype(np.float16)
            pl = _chunk_prompts(b)
            del b
            got.unlink()  # stream-reduce: purge each chunk after the layer slice
            _save_part(parts, i, cxi, vxi, pl)
            print(
                f"[f1a] unit {i + 1}/{len(names)} chunk={name} bytes={bytes_seen / 1e9:.1f}GB "
                f"elapsed={time.monotonic() - t0:.0f}s",
                flush=True,
            )
        cx_parts.append(cxi)
        vx_parts.append(vxi)
        prompts.extend(pl)
        if not pilot_logged and n_dl > 0 and (n_dl == 5 or i + 1 == len(names)):
            per = (time.monotonic() - t0) / n_dl
            print(
                f"[f1a] PILOT: {n_dl} downloaded chunks in {time.monotonic() - t0:.0f}s "
                f"({bytes_seen / 1e9:.1f} GB) -> projected stream wall "
                f"~{per * len(names) / 3600:.2f}h over {len(names)} chunks "
                "(fresh-download basis; plan section 9 F1a booked 1.4h; "
                ">2x -> epm:compute-deviation, CONTINUE)",
                flush=True,
            )
            pilot_logged = True
    n_kept = sum(p.shape[0] for p in cx_parts)
    assert len(prompts) == n_kept, (len(prompts), n_kept)
    diag = {
        "n_chunks": len(names),
        "n_chunks_resumed": int(n_resumed),
        "bytes_streamed": int(bytes_seen),
        "wall_s": time.monotonic() - t0,
        "max_chunks_cap": max_chunks,
    }
    return np.concatenate(cx_parts), np.concatenate(vx_parts), prompts, n_kept, diag


# ── FULL train-side drop set (committed fold_check machinery, un-truncated) ───────


def compute_drop_set(
    train_prompts: list[str], target_prompts: list[str], *, chunk_rows: int = 100_000
) -> dict:
    """Every train index that exact- or near-duplicates (char-5-gram Jaccard
    >= 0.8; MinHash 64-perm candidates at est >= 0.6) ANY target. Reuses the
    committed `issue1775_fold_check` primitives verbatim; unlike its
    ``overlap_battery`` (near_pairs truncated to 200) this returns the FULL
    train-side index sets — indices + jaccard digests only, never text."""
    tgt_norm = [FC._norm(t) for t in target_prompts]
    tgt_pos: dict[str, list[int]] = {}
    for ti, tn in enumerate(tgt_norm):
        tgt_pos.setdefault(tn, []).append(ti)
    sig_t = FC.minhash_signatures(target_prompts, tag="/dedup-targets")
    exact_idx: set[int] = set()
    near_idx: set[int] = set()
    affected: set[int] = set()
    n_candidates = 0
    n = len(train_prompts)
    t0 = time.monotonic()
    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        chunk = train_prompts[lo:hi]
        norms = [FC._norm(p) for p in chunk]
        for i, cn in enumerate(norms):
            if cn in tgt_pos:
                exact_idx.add(lo + i)
                affected.update(tgt_pos[cn])
        sig_c = FC.minhash_signatures(chunk, tag="/dedup-train")
        matches = np.zeros((hi - lo, len(target_prompts)), dtype=np.uint8)
        for k in range(FC.MINHASH_PERMS):
            matches += sig_c[:, k : k + 1] == sig_t[None, :, k]
        est = matches.astype(np.float64) / FC.MINHASH_PERMS
        ci, ti = np.nonzero(est >= FC.MINHASH_CAND_EST)
        for a, b in zip(ci.tolist(), ti.tolist(), strict=True):
            n_candidates += 1
            j = FC.exact_jaccard(norms[a], tgt_norm[b])
            if j >= FC.NEAR_DUPE_JACCARD:
                near_idx.add(lo + a)
                affected.add(b)
        print(
            f"[dedup] rows {hi}/{n} exact={len(exact_idx)} near={len(near_idx)} "
            f"cands={n_candidates} elapsed={time.monotonic() - t0:.0f}s",
            flush=True,
        )
    return {
        "exact_train_positions": sorted(exact_idx),
        "near_train_positions": sorted(near_idx),
        "drop_train_positions": sorted(exact_idx | near_idx),
        "affected_target_ids": sorted(affected),
        "n_candidates": int(n_candidates),
        "recipe": {
            "ngram": FC.NEAR_DUPE_NGRAM,
            "jaccard_thresh": FC.NEAR_DUPE_JACCARD,
            "minhash_perms": FC.MINHASH_PERMS,
            "candidate_est_floor": FC.MINHASH_CAND_EST,
        },
    }


def random_drop_control(n_train: int, drop_positions: list[int], *, seed: int = 0) -> list[int]:
    """n-matched control: the SAME COUNT of uniformly-random NON-near-dupe
    train positions (seed 0) — separates 'fewer rows' from 'removed dupes'."""
    drop = set(drop_positions)
    clean = np.asarray([i for i in range(n_train) if i not in drop], dtype=np.int64)
    k = len(drop)
    assert k <= clean.size, (
        f"random-drop control infeasible: drop count {k} exceeds clean pool {clean.size}"
    )
    sel = np.random.default_rng(seed).choice(clean.size, size=k, replace=False)
    return sorted(int(x) for x in clean[sel])


# ── stage (F1a) ───────────────────────────────────────────────────────────────────


def _plant_smoke_dupes(X, Y, prompts, train_pos, test_pos):
    """Append one EXACT and one NEAR duplicate of two test prompts to the train
    pool (activations duplicated) so the drop + control branches execute
    deterministically at smoke n. Returns extended (X, Y, prompts, train_pos)."""
    tlong = sorted(test_pos, key=lambda i: -len(prompts[i]))[:2]
    exact_src, near_src = tlong[0], tlong[-1]
    near_prompt = prompts[near_src] + " extra tail token"
    j = FC.exact_jaccard(FC._norm(prompts[near_src]), FC._norm(near_prompt))
    assert j >= FC.NEAR_DUPE_JACCARD, f"planted near-dupe jaccard {j:.3f} < 0.8 (prompt too short)"
    Xp = np.concatenate([X, X[[exact_src, near_src]]])
    Yp = np.concatenate([Y, Y[[exact_src, near_src]]])
    prompts2 = list(prompts) + [prompts[exact_src], near_prompt]
    train2 = np.concatenate([train_pos, [len(prompts), len(prompts) + 1]]).astype(np.int64)
    return Xp, Yp, prompts2, train2


def run_stage(args) -> int:
    out_dir = eval_dir(FU_SUB)
    arrays = fu_arrays_dir()
    # C1 (`fu-arrays-upload-verify-cache-mismatch`): the stream cache lives
    # OUTSIDE the uploaded arrays dir — hf_hub_download(local_dir=...) leaves
    # `.cache/huggingface` metadata upload_folder never ships, which broke the
    # exact-set verify when the cache nested inside the upload root.
    cache = arrays.parent / ".n50k_stream_cache"
    legacy_cache = arrays / ".n50k_stream_cache"
    if legacy_cache.exists():
        # same-pod retry of a pre-fix run: purge residue (metadata AND any
        # partial chunk .pt, which NO uploader filter would exclude)
        print(f"[f1a] purging legacy in-arrays stream cache {legacy_cache}", flush=True)
        shutil.rmtree(legacy_cache)
    t0 = time.monotonic()
    if args.smoke:
        cx, vx, prompts, n_kept, sdiag = stream_chunks_with_prompts(
            args.hf_prefix, LAYER, cache, max_chunks=args.max_chunks or 1
        )
        # tiny split through the SAME fixed_split gate path (self-derived pins)
        n_pool = n_kept
        n_val, n_te = 60, 120
        n_tr = n_pool - n_val - n_te
        r1_train, val, test = F.fixed_split(n_pool, n_tr, n_val, n_te, F.SPLIT_SEED)
        val_sha, test_sha = F._sha_ids(val), F._sha_ids(test)
        X, Y, prompts, train = _plant_smoke_dupes(cx, vx, prompts, r1_train, list(test))
        diag = {
            "mode": "smoke (1-chunk pool; planted exact+near dupes)",
            "n_train": int(train.size),
            "n_val": int(val.size),
            "n_test": int(test.size),
            "val_sha256": val_sha,
            "test_sha256": test_sha,
            "train_sha256": F._sha_ids(train),
        }
        targets = [prompts[i] for i in list(val) + list(test)]
        rederive_sha = None
    else:
        # pass_b bundle is gitignored data — self-stage from HF on a fresh instance
        if not F.PASS_B_PATH.exists():
            print(f"[f1a] staging pass_b bundle from HF -> {F.PASS_B_PATH}", flush=True)
            hub.stage_hub_file(HF_DATA_REPO, PASS_B_HF_PATH, F.PASS_B_PATH, repo_type="dataset")
        # m1: work dir OUT of eval_dir — rederive.json carries the full raw
        # prompt lists (~15-40 MB LMSYS text), which must never ride the
        # eval-json upload channel nor the pod-side eval-dir git commit
        work = out_root() / "data" / "issue_1775" / "fu_work"
        used = FC.rederive_used_sets(work, smoke=False)
        round1, new, targets = used["round1"], used["new"], used["valtest"]
        assert len(round1) == N.N_PASS_B, (len(round1), N.N_PASS_B)
        assert len(targets) == 1400, len(targets)
        rederive_sha = used["new_prompt_sha256"]
        cx, vx, prompts, n_kept, sdiag = stream_chunks_with_prompts(
            args.hf_prefix, LAYER, cache, max_chunks=args.max_chunks
        )
        assert n_kept == len(new), (
            f"streamed chunk rows {n_kept} != rederived n50k-new prompts {len(new)} — "
            "capture/pool drift"
        )
        # chunk-prompt vs rederived-prompt positional spot-assert (B2; run-1
        # chunk_spotcheck class) — sampled, never printed
        rng = np.random.default_rng(0)
        spot = rng.choice(n_kept, size=min(500, n_kept), replace=False)
        mism = int(sum(prompts[i] != new[i] for i in spot))
        assert mism == 0, f"{mism}/{len(spot)} chunk prompts mismatch the rederived pool"
        pb = F.load_pass_b(F.PASS_B_PATH)
        pb_X = N._slice_layer(pb, "cx_last", LAYER)
        pb_Y = N._slice_layer(pb, "v_x", LAYER)
        X = np.concatenate([pb_X, cx]).astype(np.float32)
        Y = np.concatenate([pb_Y, vx]).astype(np.float32)
        del pb, cx, vx
        pinned = N._pinned_original_shas(N.DEFAULT_ORIG_DIR)
        train, val, test, diag = N.build_n50k_split(
            n_kept, None, pinned, n_train=N.N50K_TRAIN, seed=N.SPLIT_SEED
        )
        banked_split = json.loads(BANKED_FITS.read_text())["split"]
        assert diag["train_sha256"] == banked_split["train_sha256"], (
            "reconstructed train index sha != banked n50k split sha — pool drift"
        )
        prompts = list(round1) + list(prompts)
    train_prompts = [prompts[int(i)] for i in train]
    drop = compute_drop_set(train_prompts, targets)
    # banked-consistency gate (production-n; demoted at smoke scale — #1345)
    if not args.smoke:
        banked = json.loads(BANKED_OVERLAP.read_text())["battery"]
        same = drop["affected_target_ids"] == banked["affected_target_ids"]
        print(
            f"[dedup] affected targets recomputed={len(drop['affected_target_ids'])} "
            f"banked={len(banked['affected_target_ids'])} set_equal={same}",
            flush=True,
        )
        assert same, (
            "recomputed affected-target set != banked gate_a.affected_target_ids — "
            "dedup machinery drifted from run-1's audit"
        )
    else:
        assert len(drop["exact_train_positions"]) >= 1, "smoke: planted exact dupe not found"
        assert len(drop["near_train_positions"]) >= 1, "smoke: planted near dupe not found"
    control = random_drop_control(len(train_prompts), drop["drop_train_positions"], seed=0)
    # persist reduced arrays + split + drop set (fp16 arrays; index npys; digest JSON)
    np.save(arrays / "n50k_L19_cx.npy", X.astype(np.float16))
    np.save(arrays / "n50k_L19_vx.npy", Y.astype(np.float16))
    np.save(arrays / "split_train.npy", np.asarray(train, dtype=np.int64))
    np.save(arrays / "split_val.npy", np.asarray(val, dtype=np.int64))
    np.save(arrays / "split_test.npy", np.asarray(test, dtype=np.int64))
    np.save(
        arrays / "drop_train_positions.npy",
        np.asarray(drop["drop_train_positions"], dtype=np.int64),
    )
    np.save(arrays / "control_train_positions.npy", np.asarray(control, dtype=np.int64))
    drop_out = {
        "meta": result_meta(smoke=bool(args.smoke), layer=LAYER, stream=sdiag),
        "split_diag": {k: v for k, v in diag.items() if k != "note"},
        "n_train": int(len(train_prompts)),
        "n_targets": len(targets),
        "n_exact_train": len(drop["exact_train_positions"]),
        "n_near_train": len(drop["near_train_positions"]),
        "n_drop_train": len(drop["drop_train_positions"]),
        "n_affected_targets": len(drop["affected_target_ids"]),
        "affected_target_ids": drop["affected_target_ids"],
        "n_candidates": drop["n_candidates"],
        "recipe": drop["recipe"],
        "random_drop_control": {"seed": 0, "n_dropped": len(control)},
        "rederived_new_prompt_sha256": rederive_sha,
        "scope_caveat": LEXICAL_CAVEAT,
        "note": (
            "index/jaccard digests only — no prompt text (refusal-safety, the #779 "
            "module contract); full position arrays ride the .npy sidecars"
        ),
    }
    atomic_write_json(out_dir / "n50k_drop_set.json", drop_out)
    if args.smoke:
        print("[f1a] smoke — skipping HF upload (scratch out-root)", flush=True)
    else:
        # expensive-intermediate ordering (#825): arrays land on HF BEFORE any fit
        _upload_dir_verified(arrays, f"{OUT_HF_PREFIX}/fu_dedup_refit")
        upload_phase_eval_json(FU_SUB, smoke=False)
        # arrays durable on HF -> reap the stream cache + per-chunk parts
        # (kept until here so a failed upload leaves the resume state intact)
        shutil.rmtree(cache, ignore_errors=True)
    print(
        f"[f1a] done in {(time.monotonic() - t0) / 60:.1f} min "
        f"(drop={drop_out['n_drop_train']} of {drop_out['n_train']} train rows; "
        f"affected targets={drop_out['n_affected_targets']})",
        flush=True,
    )
    return 0


# ── fits (F1b) ────────────────────────────────────────────────────────────────────


def _restage_arrays_if_needed(arrays: Path) -> None:
    needed = [
        "n50k_L19_cx.npy",
        "n50k_L19_vx.npy",
        "split_train.npy",
        "split_val.npy",
        "split_test.npy",
        "drop_train_positions.npy",
        "control_train_positions.npy",
    ]
    for name in needed:
        target = arrays / name
        if not target.exists():
            hub.stage_hub_file(
                HF_DATA_REPO,
                f"{OUT_HF_PREFIX}/fu_dedup_refit/{name}",
                target,
                repo_type="dataset",
            )
            print(f"[f1b] restaged {name} from HF", flush=True)


def _variant_train(train: np.ndarray, positions_to_drop: np.ndarray) -> np.ndarray:
    keep = np.ones(train.size, dtype=bool)
    keep[positions_to_drop] = False
    return train[keep]


def run_fits(args) -> int:
    out_dir = eval_dir(FU_SUB)
    arrays = fu_arrays_dir()
    _restage_arrays_if_needed(arrays)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    dev = torch.device(device)
    X = np.load(arrays / "n50k_L19_cx.npy").astype(np.float32)
    Y = np.load(arrays / "n50k_L19_vx.npy").astype(np.float32)
    train = np.load(arrays / "split_train.npy")
    val = np.load(arrays / "split_val.npy")
    test = np.load(arrays / "split_test.npy")
    drop_pos = np.load(arrays / "drop_train_positions.npy")
    ctrl_pos = np.load(arrays / "control_train_positions.npy")
    trains = {
        "deduped": _variant_train(train, drop_pos),
        "random_drop": _variant_train(train, ctrl_pos),
    }
    assert trains["deduped"].size == trains["random_drop"].size, "control is not n-matched"
    recipe = N._mlp_recipe(N.DEFAULT_ORIG_DIR / "fair_comparison.json")
    mlp_epochs = args.mlp_max_epochs
    n_boot = args.n_boot
    units_path = out_dir / "units_n50k_refit.jsonl"
    done = {unit_key(d, FIT_REGIME_KEYS) for d in load_units(units_path)}
    t0 = time.monotonic()
    n_done = 0
    krr_gm = tuple(float(g) for g in args.krr_gamma_mult.split(","))
    krr_lam = tuple(float(x) for x in args.krr_lambdas.split(","))
    for variant, tr in trains.items():
        for rung in RUNGS:
            u = {
                "variant": variant,
                "rung": rung,
                "layer": LAYER,
                "smoke": bool(args.smoke),
                "n_train": int(tr.size),
                "n_boot": int(n_boot),
            }
            if rung == "krr":
                u["krr_grid"] = f"gm={args.krr_gamma_mult};lam={args.krr_lambdas}"
            if rung in ("mlp", "residual_skip"):
                u["mlp_epochs"] = int(mlp_epochs)
            pred_path = arrays / f"pred_{variant}_{rung}.npy"
            if unit_key(u, FIT_REGIME_KEYS) in done and pred_path.exists():
                n_done += 1
                print(f"[f1b] RESUME unit {variant}/{rung}", flush=True)
                continue
            tu = time.monotonic()
            if rung == "ridge":
                pred, meta = N.fit_ridge_primal(X, Y, tr, val, test, N.LAMBDAS_N50K, dev)
            elif rung == "krr":
                pred, meta = N.fit_krr_exact(
                    X,
                    Y,
                    tr,
                    val,
                    test,
                    gamma_mult=krr_gm,
                    lambdas=krr_lam,
                    block=args.krr_block,
                    seed=0,
                    dev=dev,
                )
            elif rung == "mlp":
                pred, meta = N.fit_mlp(X, Y, tr, test, recipe, mlp_epochs, 0, dev)
            else:
                pred, meta = N.fit_residual_skip(
                    X, Y, tr, val, test, N.LAMBDAS_N50K, recipe, mlp_epochs, 0, dev
                )
            curve = N._curve(pred, Y[test], n_boot, 0)
            np.save(pred_path, pred.astype(np.float16))
            append_unit(
                units_path, {**u, **curve, "fit_meta": meta, "wall_s": time.monotonic() - tu}
            )
            n_done += 1
            print(
                f"[f1b] unit {n_done}/{len(trains) * len(RUNGS)} {variant}/{rung} "
                f"r2={curve['whole_map_r2']:.4f} elapsed={time.monotonic() - t0:.0f}s",
                flush=True,
            )
    # ── assembly: gains vs ridge (paired ROW bootstrap — #779 has no groups) ─────
    rows = load_units(units_path)
    by = {(d["variant"], d["rung"]): d for d in rows if d.get("smoke") == bool(args.smoke)}
    n_te = test.size
    ones = np.ones(n_te, dtype=bool)
    row_groups = np.arange(n_te)  # singleton groups == paired row bootstrap (labeled)
    out_variants: dict = {}
    for variant, tr in trains.items():
        preds = {
            rung: np.load(arrays / f"pred_{variant}_{rung}.npy").astype(np.float64)
            for rung in RUNGS
        }
        gains = {}
        for rung in ("krr", "mlp", "residual_skip"):
            gains[f"{rung}_minus_ridge"] = cluster_bootstrap_delta_r2(
                Y[test].astype(np.float64),
                preds[rung],
                preds["ridge"],
                ones,
                row_groups,
                n_draws=n_boot,
                seed=0,
            )
        idb = identity_bias_predict(X[tr], Y[tr], X[test])
        knn = {
            rung: {
                m: knn_retrieval(preds[rung], Y[test].astype(np.float64), ks=(1, 5, 10), metric=m)
                for m in ("euclidean", "cosine")
            }
            for rung in RUNGS
        }
        out_variants[variant] = {
            "n_train": int(tr.size),
            "per_rung": {
                rung: {k: v for k, v in by[(variant, rung)].items() if k not in ("variant",)}
                for rung in RUNGS
            },
            "gains_vs_ridge_paired_row_bootstrap": gains,
            "baselines": {
                "identity_bias_r2": _r2(Y[test], idb),
                "knn_retrieval": knn,
                "note": "3584->3584: identity+bias AND kNN both applicable; pool=n_test, "
                "chance = k/n_test",
            },
        }
    banked = None
    if BANKED_FITS.exists():
        b = json.loads(BANKED_FITS.read_text())
        banked = {
            "per_predictor_whole_map_r2": {
                k: v.get("whole_map_r2") for k, v in b.get("per_predictor", {}).items()
            },
            "note": "banked UNAUDITED n50k numbers requoted for the comparison figure "
            "(no refit; grid-matched KRR small grid)",
        }
    out = {
        "meta": result_meta(
            smoke=bool(args.smoke),
            layer=LAYER,
            n_boot=n_boot,
            krr_grid={"gamma_mult": list(krr_gm), "lambdas": list(krr_lam)},
            mlp_recipe=recipe,
            gain_ci_protocol=(
                "paired ROW bootstrap over the test rows (the #779 protocol has no "
                "group structure — labeled; singleton-group cluster_bootstrap_delta_r2)"
            ),
        ),
        "n_test": int(n_te),
        "variants": out_variants,
        "banked_reference": banked,
        "scope_caveat": LEXICAL_CAVEAT,
    }
    atomic_write_json(out_dir / "n50k_dedup_refit.json", out)
    if args.smoke:
        print("[f1b] smoke — skipping HF upload (scratch out-root)", flush=True)
    else:
        _upload_dir_verified(arrays, f"{OUT_HF_PREFIX}/fu_dedup_refit")
        upload_phase_eval_json(FU_SUB, smoke=False)
    print(f"[f1b] done in {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 fu cell 1: n50k dedup refit (P0c)")
    ap.add_argument("--stage", choices=["stage", "fits", "all"], default="all")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--mlp-max-epochs", type=int, default=None)
    ap.add_argument("--krr-gamma-mult", default=",".join(str(g) for g in N.KRR_GAMMA_MULT))
    ap.add_argument("--krr-lambdas", default=",".join(str(x) for x in N.KRR_LAMBDAS))
    ap.add_argument("--krr-block", type=int, default=N.KRR_KERNEL_BLOCK)
    ap.add_argument("--hf-prefix", default=N.HF_N50K_PREFIX)
    args = ap.parse_args()
    args.n_boot = args.n_boot or (200 if args.smoke else F.BOOT_N)
    args.mlp_max_epochs = args.mlp_max_epochs or (8 if args.smoke else F.MLP_MAX_EPOCHS)
    rc = 0
    if args.stage in ("stage", "all"):
        rc = run_stage(args)
    if rc == 0 and args.stage in ("fits", "all"):
        rc = run_fits(args)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
