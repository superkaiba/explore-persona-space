"""Issue #1482 — reconstruct the pod-local scratch metadata on the VM (recovery r6).

The P0-P4 pod built scratch/{split_indices.npz,row_ci.npy,prov.npy} locally and was
terminated before that ~17 MB metadata was uploaded (epm:failure v6); the off-pod
phases (P5 judge / P6 analysis, scripts/issue1482_analysis.py) load exactly those
three files. This script rebuilds them DETERMINISTICALLY from durable artifacts and
FAILS LOUD unless every reconstruction check matches the committed verification
anchors (eval_results/issue_1482/split_1482.json + the percontext holdout_rows):

  1. pinned pass_b split — F.fixed_split (pure RNG; val/test shas asserted against
     the pinned constants in split_1482.json; no capture staging needed);
  2. new-row ci order — the raw_completions chunks (sorted, the r5 driver filter
     ``^shard\\d+_chunk\\d+\\.json$``): each raw chunk's rows are written from the
     SAME kept-row list as its capture .pt twin (issue779_ffc_n1m_generate_capture
     ._capture_stage_chunk), so the concatenated ci order equals P0's
     N1M._stream_n1m_layer order; per-chunk ci arrays checkpoint under --work
     (external-stream rule: crash resumes, raw JSON deleted after parse);
  3. corpus per ci — the frozen sampling manifest (N1G._resolve_manifest_dir);
  4. the P0 carve — replayed EXACTLY: D._stratified_sample under the same
     default_rng(D.SPLIT_SEED_1482) draw sequence over the same pools;
  5. verification — holdout/sae_fit/sae_val/train_full sha256 + per-bucket
     lmsys/wildchat counts + n_total + realized lmsys_frac vs split_1482.json, and
     sorted(holdout) == sorted(percontext refit_holdout__ridge__seed0.npz
     holdout_rows) from the pod's own P1 output.

LMSYS/WildChat text hygiene (digest-only): raw chunk JSONs are parsed ONLY for the
integer "ci" field and unlinked immediately; prompt/response text is never printed,
logged, or cached by this script. Outputs land atomically in --scratch (the dir
issue1482_analysis.py reads) plus a reconstruction.json provenance sidecar.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM run)

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue1482_error_analysis as D  # noqa: E402  (P0 carve helpers: _stratified_sample, ...)
import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1482.reconstruct")

PERCONTEXT_ANCHOR = "refit_holdout__ridge__seed0.npz"
SCRATCH_FILES = ("split_indices.npz", "row_ci.npy", "prov.npy")


def _verify(name: str, got, want) -> None:
    """One fail-loud reconstruction check; logs the PASS line the report cites."""
    if got != want:
        raise RuntimeError(f"[verify] {name}: reconstructed {got!r} != anchored {want!r}")
    logger.info("[verify] %s OK (%s)", name, got)


def _chunk_ci(name: str, cache_dir: Path, ci_dir: Path) -> np.ndarray:
    """Return one raw chunk's ci array (int64, row order). Downloads via the parent's
    retry envelope, caches the ci array (resume checkpoint), deletes the raw JSON.
    Digest-only: nothing but the integer ci field is read from each row."""
    out = ci_dir / f"{name}.ci.npy"
    if out.exists():
        return np.load(out)
    got = Path(N1M._download_chunk_with_retry(C.HF_DATA_REPO, f"{D.RAW_PREFIX}/{name}", cache_dir))
    rows = json.loads(got.read_text())["rows"]
    ci = np.asarray([int(r["ci"]) for r in rows], dtype=np.int64)
    got.unlink()  # raw text (LMSYS/WildChat) is transient — never cached by this script
    tmp = ci_dir / f"{name}.ci.tmp.npy"
    np.save(tmp, ci)
    os.replace(tmp, out)
    return ci


def _collect_new_ci(args, expected_n: int) -> np.ndarray:
    """Concatenate every raw chunk's ci in sorted-chunk-name order (== P0's capture
    stream order; see module docstring step 2). Thread pool <=6 (#833 recipe);
    per-chunk ci checkpoints make a crash resume O(remaining chunks)."""
    names = D._raw_chunk_names(argparse.Namespace(max_chunks=args.max_chunks))
    logger.info("[new-ci] %d raw chunks enumerated", len(names))
    cache_dir = args.work / "raw_cache"
    ci_dir = args.work / "ci_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ci_dir.mkdir(parents=True, exist_ok=True)
    n_cached = sum(1 for n in names if (ci_dir / f"{n}.ci.npy").exists())
    if n_cached:
        logger.info("[new-ci] resume: %d/%d chunk ci arrays cached", n_cached, len(names))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        parts = list(ex.map(lambda n: _chunk_ci(n, cache_dir, ci_dir), names))
    new_ci = np.concatenate(parts).astype(np.int64)
    logger.info(
        "[new-ci] %d rows from %d chunks in %.0fs", len(new_ci), len(names), time.time() - t0
    )
    if args.max_chunks == 0:
        assert len(new_ci) == expected_n, (len(new_ci), expected_n)
        assert len(np.unique(new_ci)) == len(new_ci), "duplicate ci across raw chunks"
    return new_ci


def _corpus_u8_by_ci(args, n_new_manifest: int) -> np.ndarray:
    """uint8 corpus per manifest index (0=lmsys, 1=wildchat) from the frozen
    sampling manifest — the same N1G loader P0 uses (pool rows carry real-corpus
    prompt text; only the i/corpus fields are consumed, then the pool is freed)."""
    ns = argparse.Namespace(out_dir=args.work, manifest_from_hf=True, hf_prefix=N1G.HF_PREFIX)
    manifest_dir = N1G._resolve_manifest_dir(ns)
    pool, man_meta = N1G.read_manifest_pool(manifest_dir)
    assert int(man_meta["n_new"]) == n_new_manifest, (man_meta["n_new"], n_new_manifest)
    corpus_u8 = np.zeros(n_new_manifest, dtype=np.uint8)
    for r in pool:
        c = r["corpus"]
        assert c in ("lmsys", "wildchat"), c
        corpus_u8[int(r["i"])] = 1 if c == "wildchat" else 0
    n_wild = int(corpus_u8.sum())
    assert n_wild == int(man_meta["n_wildchat"]), (n_wild, man_meta["n_wildchat"])
    del pool
    gc.collect()  # pool rows carry 960k prompts (~GBs RSS) — free before the carve
    return corpus_u8


def _holdout_anchor_rows(args) -> np.ndarray:
    """The pod run's own holdout row indices from the percontext P1 output —
    local (untracked, synced off-pod) copy preferred, HF download fallback."""
    local = args.out_eval / "percontext" / PERCONTEXT_ANCHOR
    if not local.exists():
        remote = f"{D.HF_PREFIX_DEFAULT}/analysis_tensors/percontext/{PERCONTEXT_ANCHOR}"
        logger.info("[anchor] local percontext npz absent; downloading %s", remote)
        local = Path(N1M._download_chunk_with_retry(C.HF_DATA_REPO, remote, args.work))
    z = np.load(local)
    rows = np.asarray(z["holdout_rows"], dtype=np.int64)
    logger.info("[anchor] holdout_rows loaded from %s (n=%d)", local, len(rows))
    return rows


def reconstruct(args) -> dict:
    """Rebuild + sha-verify the three scratch files; returns the provenance doc."""
    split_doc = json.loads((args.out_eval / "split_1482.json").read_text())
    regime = split_doc["regime"]
    assert regime["smoke"] is False, "anchored split_1482.json is a smoke run — refusing"
    assert regime["max_chunks"] == args.max_chunks, (regime["max_chunks"], args.max_chunks)

    # (1) pinned pass_b split — pure fixed_split, verified against the pinned shas.
    r1_train, val, test = F.fixed_split(
        N1M.N_PASS_B, N1M.N_PASS_B - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    _verify("val_sha256", F._sha_ids(val), split_doc["pinned_val_sha256"])
    _verify("test_sha256", F._sha_ids(test), split_doc["pinned_test_sha256"])
    _verify("orig_train_ids", len(r1_train), split_doc["orig_train_ids"])

    # (2) new-row ci order from the raw chunks; (3) corpus per ci from the manifest.
    new_ci = _collect_new_ci(args, expected_n=int(split_doc["n_new_captured"]))
    corpus_u8 = _corpus_u8_by_ci(args, n_new_manifest=int(split_doc["n_new_manifest"]))

    n_total = N1M.N_PASS_B + len(new_ci)
    _verify("n_total", n_total, int(split_doc["n_total"]))
    new_u8 = corpus_u8[new_ci]
    prov_u8 = np.concatenate([np.zeros(N1M.N_PASS_B, dtype=np.uint8), new_u8])
    prov_obj = np.where(prov_u8 == 1, "wildchat", "lmsys").astype(object)
    row_ci = np.full(n_total, -1, dtype=np.int64)
    row_ci[N1M.N_PASS_B :] = new_ci

    # (4) the P0 carve, replayed EXACTLY (same helpers, same rng draw sequence).
    pools = N1M._pool_rows(prov_obj, r1_train, n_total, val, test)
    train_full = pools["full"]
    lmsys_frac = len(pools["lmsys"]) / len(train_full)
    _verify(
        "realized_lmsys_frac_full_pool",
        round(float(lmsys_frac), 4),
        split_doc["realized_lmsys_frac_full_pool"],
    )
    new_rows = np.arange(N1M.N_PASS_B, n_total)
    rng = np.random.default_rng(D.SPLIT_SEED_1482)
    holdout, hold_diag = D._stratified_sample(
        rng, new_rows, prov_u8, regime["holdout_n"], lmsys_frac
    )
    remaining = np.setdiff1d(new_rows, holdout, assume_unique=False)
    sae_fit, fit_diag = D._stratified_sample(rng, remaining, prov_u8, regime["sae_n"], lmsys_frac)
    remaining2 = np.setdiff1d(remaining, sae_fit, assume_unique=False)
    sae_val, val_diag = D._stratified_sample(
        rng, remaining2, prov_u8, regime["sae_val_n"], lmsys_frac
    )
    for nm, arr in (("holdout", holdout), ("sae_fit", sae_fit), ("sae_val", sae_val)):
        assert not (set(arr.tolist()) & (set(val.tolist()) | set(test.tolist()))), nm
    assert not (set(holdout.tolist()) & set(sae_fit.tolist()))
    assert not (set(holdout.tolist()) & set(sae_val.tolist()))
    assert not (set(sae_fit.tolist()) & set(sae_val.tolist()))

    # (5) verification against the committed anchors + the pod's own P1 output.
    checks: dict[str, str] = {}
    for nm, arr, diag, anchor in (
        ("holdout", holdout, hold_diag, split_doc["holdout"]),
        ("sae_fit", sae_fit, fit_diag, split_doc["sae_fit"]),
        ("sae_val", sae_val, val_diag, split_doc["sae_val"]),
    ):
        _verify(f"{nm}.sha256", D._sha_ids(arr), anchor["sha256"])
        for k in ("n", "n_lmsys", "n_wildchat"):
            _verify(f"{nm}.{k}", int(diag[k]), int(anchor[k]))
        checks[f"{nm}_sha256"] = D._sha_ids(arr)
    _verify("train_full_sha256", D._sha_ids(train_full), split_doc["train_full_sha256"])
    checks["train_full_sha256"] = D._sha_ids(train_full)
    anchor_rows = _holdout_anchor_rows(args)
    if not np.array_equal(np.sort(holdout), np.sort(anchor_rows)):
        raise RuntimeError(
            "[verify] holdout rows != percontext holdout_rows (pod P1 anchor) — "
            f"{len(np.setdiff1d(holdout, anchor_rows))} reconstructed-only rows"
        )
    logger.info("[verify] percontext holdout_rows row-set equality OK (n=%d)", len(holdout))

    # Atomic writes into the scratch dir the analysis driver reads.
    args.scratch.mkdir(parents=True, exist_ok=True)
    npz_tmp = args.scratch / "split_indices.npz.tmp"
    with open(npz_tmp, "wb") as f:
        np.savez(
            f,
            train_full=train_full,
            train_lmsys=pools["lmsys"],
            val=val,
            test=test,
            holdout=holdout,
            sae_fit=sae_fit,
            sae_val=sae_val,
        )
    os.replace(npz_tmp, args.scratch / "split_indices.npz")
    for nm, arr in (("row_ci.npy", row_ci), ("prov.npy", prov_u8)):
        tmp = args.scratch / f"{nm}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, args.scratch / nm)
    doc = {
        "reconstructed": list(SCRATCH_FILES),
        "scratch": str(args.scratch),
        "n_total": int(n_total),
        "checks": checks,
        "percontext_anchor": PERCONTEXT_ANCHOR,
        "metadata": C.reproducibility_metadata(),
    }
    C.write_json_atomic(args.scratch / "reconstruction.json", doc)
    logger.info("[reconstruct] ALL VERIFICATIONS PASSED — wrote %s", args.scratch)
    return doc


def probe(args) -> None:
    """Cheap plumbing smoke (no outputs written): pinned-split shas, chunk listing +
    N chunk ci downloads (checkpoint + resume path), percontext anchor load."""
    split_doc = json.loads((args.out_eval / "split_1482.json").read_text())
    _, val, test = F.fixed_split(
        N1M.N_PASS_B, N1M.N_PASS_B - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    _verify("val_sha256", F._sha_ids(val), split_doc["pinned_val_sha256"])
    _verify("test_sha256", F._sha_ids(test), split_doc["pinned_test_sha256"])
    names = D._raw_chunk_names(argparse.Namespace(max_chunks=0))
    logger.info("[probe] %d raw chunks enumerated", len(names))
    cache_dir = args.work / "raw_cache"
    ci_dir = args.work / "ci_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ci_dir.mkdir(parents=True, exist_ok=True)
    for n in names[: args.probe]:
        ci = _chunk_ci(n, cache_dir, ci_dir)
        ci2 = _chunk_ci(n, cache_dir, ci_dir)  # second call exercises the resume cache
        assert np.array_equal(ci, ci2)
        logger.info("[probe] chunk %s: %d rows (ci cached + resume-read)", n, len(ci))
    rows = _holdout_anchor_rows(args)
    assert rows.shape == (int(split_doc["holdout"]["n"]),), rows.shape
    logger.info("[probe] OK — plumbing verified (no outputs written)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--scratch", type=Path, default=PROJECT_ROOT / "data" / "issue_1482" / "scratch"
    )
    ap.add_argument("--out-eval", type=Path, default=PROJECT_ROOT / "eval_results" / "issue_1482")
    ap.add_argument(
        "--work",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_1482" / "hf_dl" / "reconstruct",
        help="transient staging (manifest, per-chunk ci checkpoints; janitor-sweepable)",
    )
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all (production)")
    ap.add_argument("--workers", type=int, default=6, help="chunk-download pool (<=6, #833)")
    ap.add_argument("--probe", type=int, default=0, help="N>0: plumbing smoke only, then exit")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.probe > 0:
        probe(args)
        return 0
    have = [n for n in SCRATCH_FILES if (args.scratch / n).exists()]
    if len(have) == len(SCRATCH_FILES) and not args.force:
        logger.info("[reconstruct] outputs already present under %s (use --force)", args.scratch)
        return 0
    reconstruct(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
