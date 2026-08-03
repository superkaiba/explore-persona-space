#!/usr/bin/env python3
"""#1775 P0: fold-structure check on the banked #779 splits + #1092 fold sanity.

VM CPU phase (streaming, projected RSS < 8 GB; ``ru_maxrss`` logged at the
first signature chunk). Gating for quoting the banked n50k/n1m gains (plan
section 4 P0 / section 7 Gate A).

Steps (``--steps``):
  folds1092  re-derive ``_folds_from_manifest`` on the #1092 manifest; assert
             group disjointness + the 21,193 / 19,708 / 17,308 exclusion counts.
  rederive   deterministic LMSYS re-derivation of round-1 (5000) + n10k (6500)
             + n50k-new (46,600) prompt sets via ``N50.sample_disjoint_n50k``
             (cached to --work-dir; the 1400 val/test targets come from
             round-1 via the exact ``fixed_split`` + ctx0 drift assert).
  n1m        stage the n1m sampling manifest (88 parts) and recompute the
             exact + char-5-gram-Jaccard >= 0.8 train-vs-target overlap
             (MinHash 64-perm candidate GEMM, exact Jaccard on candidates).
  n50k       reconstruct the REALIZED 50,000-row n50k train pool
             (``build_n50k_split`` — fixed_split + seeded choice) and run the
             same overlap battery; contamination sensitivity bound + Gate A.
  chunkcheck spot-check the re-derived n50k-new prompts against ONE staged
             capture chunk's realized ``prompts``/``ci`` (A1 verification —
             full-chunk staging is infeasible: ~1.8 GB/chunk x 96).
  recall     MinHash near-threshold recall check on 1k synthetic pairs with
             true char-5-gram Jaccard in [0.6, 0.9] (not random pairs).

Refusal-safety: LMSYS/WildChat are unscreened real-user corpora — this script
NEVER prints or logs prompt text; only counts, indices, and sha256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
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

import numpy as np  # noqa: E402

from issue1775_common import (  # noqa: E402
    FOLD_SEED,
    N_FOLDS,
    _folds_from_manifest,
    atomic_write_json,
    battery_excluded_indices,
    eval_dir,
    load_manifest_rows,
    resolve_store_dir,
    result_meta,
)

# #779 machinery reused verbatim (plan section 10 reuse map).
import issue779_ffc_n1m_generate_capture as N1M  # noqa: E402
import issue779_ffc_n50k_fits as N50F  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
from issue779_ffc_n1m_generate_capture import NearDupeGate, _norm  # noqa: E402

NEAR_DUPE_NGRAM = int(N1M.NEAR_DUPE_NGRAM)  # 5 (char n-grams)
NEAR_DUPE_JACCARD = float(N1M.NEAR_DUPE_JACCARD)  # 0.8
MINHASH_PERMS = 64
MINHASH_CAND_EST = 0.6  # candidate floor on the MinHash Jaccard estimate
_FNV = np.uint64(1099511628211)
_SPLITMIX = np.uint64(0x9E3779B97F4A7C15)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ── MinHash signatures (batched; no per-pair loops) ──────────────────────────────


def _shingle_hashes(norm_text: str) -> np.ndarray:
    """uint64 hashes of the char-5-gram SET of a normalized text (gate semantics:
    len < 5 -> the whole text as one shingle; empty -> no shingles)."""
    if not norm_text:
        return np.empty(0, dtype=np.uint64)
    codes = np.frombuffer(norm_text.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    n = codes.size
    if n < NEAR_DUPE_NGRAM:
        h = np.uint64(int(hashlib.blake2b(norm_text.encode(), digest_size=8).hexdigest(), 16))
        return np.asarray([h], dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = codes[: n - 4].copy()
        for k in range(1, NEAR_DUPE_NGRAM):
            h = h * _FNV + codes[k : n - 4 + k]
    return np.unique(h)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + _SPLITMIX
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


_PERM_SEEDS = _splitmix64(np.arange(1, MINHASH_PERMS + 1, dtype=np.uint64) * np.uint64(0o7777))


def minhash_signatures(
    prompts: list[str], *, chunk_rows: int = 100_000, tag: str = ""
) -> np.ndarray:
    """(n, 64) uint64 MinHash signatures over char-5-gram sets, batched via
    concatenated hash arrays + ``minimum.reduceat`` (no per-prompt perm loop).
    Empty prompts get all-max signatures (never candidates — gate semantics)."""
    n = len(prompts)
    sig = np.full((n, MINHASH_PERMS), np.iinfo(np.uint64).max, dtype=np.uint64)
    t0 = time.monotonic()
    first_logged = False
    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        hashes: list[np.ndarray] = []
        starts: list[int] = []
        keep_rows: list[int] = []
        off = 0
        for i in range(lo, hi):
            h = _shingle_hashes(_norm(prompts[i]))
            if h.size == 0:
                continue
            hashes.append(h)
            starts.append(off)
            keep_rows.append(i)
            off += h.size
        if not hashes:
            continue
        allh = np.concatenate(hashes)
        starts_arr = np.asarray(starts, dtype=np.int64)
        rows = np.asarray(keep_rows, dtype=np.int64)
        for k in range(MINHASH_PERMS):
            y = _splitmix64(allh ^ _PERM_SEEDS[k])
            sig[rows, k] = np.minimum.reduceat(y, starts_arr)
        if not first_logged:
            print(
                f"[minhash{tag}] first chunk rows={hi - lo} shingles={allh.size} "
                f"ru_maxrss={_rss_gb():.2f} GB elapsed={time.monotonic() - t0:.1f}s",
                flush=True,
            )
            first_logged = True
        print(f"[minhash{tag}] unit {hi}/{n} elapsed={time.monotonic() - t0:.1f}s", flush=True)
    return sig


def exact_jaccard(a_norm: str, b_norm: str) -> float:
    ga = set(N1M._char_ngrams(a_norm, NEAR_DUPE_NGRAM))
    gb = set(N1M._char_ngrams(b_norm, NEAR_DUPE_NGRAM))
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga) + len(gb) - inter
    return inter / union if union else 0.0


def overlap_battery(
    train_prompts: list[str],
    target_prompts: list[str],
    *,
    tag: str,
    chunk_rows: int = 100_000,
) -> dict:
    """Exact-normalized + near-dupe (char-5-gram Jaccard >= 0.8) train-vs-target
    overlap. MinHash 64-perm signatures compared as a chunked integer-match GEMM;
    exact Jaccard recomputed ONLY on MinHash candidates (est >= 0.6)."""
    t0 = time.monotonic()
    tgt_norm = [_norm(t) for t in target_prompts]
    tgt_exact = {}
    for ti, tn in enumerate(tgt_norm):
        tgt_exact.setdefault(tn, []).append(ti)
    sig_t = minhash_signatures(target_prompts, tag=f"/{tag}-targets")
    n_exact = 0
    exact_target_hits: set[int] = set()
    cand_pairs: list[tuple[int, int, float]] = []  # (train_i, target_i, est)
    per_target_max_est = np.zeros(len(target_prompts), dtype=np.float64)
    n = len(train_prompts)
    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        chunk = train_prompts[lo:hi]
        norms = [_norm(p) for p in chunk]
        for i, cn in enumerate(norms):
            if cn in tgt_exact:
                n_exact += 1
                exact_target_hits.update(tgt_exact[cn])
        sig_c = minhash_signatures(chunk, tag=f"/{tag}-train")
        matches = np.zeros((hi - lo, len(target_prompts)), dtype=np.uint8)
        for k in range(MINHASH_PERMS):
            matches += sig_c[:, k : k + 1] == sig_t[None, :, k]
        est = matches.astype(np.float64) / MINHASH_PERMS
        per_target_max_est = np.maximum(per_target_max_est, est.max(axis=0))
        ci, ti = np.nonzero(est >= MINHASH_CAND_EST)
        for a, b in zip(ci.tolist(), ti.tolist(), strict=True):
            cand_pairs.append((lo + a, b, float(est[a, b])))
        print(
            f"[battery/{tag}] rows {hi}/{n} exact={n_exact} cands={len(cand_pairs)} "
            f"ru_maxrss={_rss_gb():.2f} GB elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )
    near_pairs = []
    affected_targets: set[int] = set()
    per_target_max_exact: dict[int, float] = {}
    for i, ti, est in cand_pairs:
        j = exact_jaccard(_norm(train_prompts[i]), tgt_norm[ti])
        per_target_max_exact[ti] = max(per_target_max_exact.get(ti, 0.0), j)
        if j >= NEAR_DUPE_JACCARD:
            near_pairs.append({"train_idx": int(i), "target_idx": int(ti), "jaccard": float(j)})
            affected_targets.add(ti)
    hist, edges = np.histogram(per_target_max_est, bins=np.linspace(0, 1, 21))
    return {
        "tag": tag,
        "n_train": int(n),
        "n_targets": len(target_prompts),
        "near_dupe_recipe": {"ngram": NEAR_DUPE_NGRAM, "jaccard_thresh": NEAR_DUPE_JACCARD},
        "minhash": {"n_perms": MINHASH_PERMS, "candidate_est_floor": MINHASH_CAND_EST},
        "n_exact": int(n_exact),
        "n_exact_affected_targets": len(exact_target_hits),
        "n_candidates": len(cand_pairs),
        "n_near": len(near_pairs),
        "near_pairs": near_pairs[:200],
        "affected_target_ids": sorted(affected_targets | exact_target_hits),
        "per_target_max_est_jaccard_hist": {
            "edges": [float(e) for e in edges],
            "counts": [int(c) for c in hist],
        },
        "per_target_max_exact_jaccard_on_candidates": {
            str(k): float(v) for k, v in sorted(per_target_max_exact.items())
        },
        "ru_maxrss_gb": _rss_gb(),
        "wall_s": time.monotonic() - t0,
    }


# ── recall check (near-threshold pairs, NOT random pairs) ────────────────────────


def _perturb_to_jaccard(norm_text: str, rng: np.random.Generator) -> tuple[str, float]:
    """Perturbed copy with char-5-gram Jaccard targeted into [0.6, 0.9] by
    replacing a suffix fraction with random word salad; returns (text, exact J)."""
    words = norm_text.split()
    vocab = "abcdefghijklmnopqrstuvwxyz"
    for frac in (0.05, 0.08, 0.12, 0.18, 0.25, 0.35):
        k = max(1, int(len(words) * frac))
        salad = ["".join(rng.choice(list(vocab), size=rng.integers(3, 9))) for _ in range(k)]
        cand = " ".join(words[: max(1, len(words) - k)] + salad)
        j = exact_jaccard(norm_text, cand)
        if 0.6 <= j <= 0.9:
            return cand, j
    return cand, j  # closest attempt (reported; filtered by caller)


def recall_check(target_prompts: list[str], *, n_pairs: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    bases = [t for t in ([_norm(t) for t in target_prompts]) if len(t.split()) >= 8]
    if not bases:
        raise ValueError("no targets long enough for the recall check")
    pairs = []
    # round-2 Minor-f: bounded pair enumeration — _perturb_to_jaccard can miss
    # [0.6, 0.9] on short/degenerate texts, so an unbounded while loop can spin.
    max_attempts = 50 * n_pairs
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        base = bases[int(rng.integers(0, len(bases)))]
        cand, j = _perturb_to_jaccard(base, rng)
        if 0.6 <= j <= 0.9:
            pairs.append((base, cand, j))
    if len(pairs) < n_pairs:
        # fail loud below a usable floor; otherwise proceed with what landed
        if len(pairs) < max(10, n_pairs // 10):
            raise RuntimeError(
                f"recall_check: only {len(pairs)}/{n_pairs} near-threshold pairs "
                f"after {attempts} attempts — target texts too short/degenerate"
            )
        print(
            f"[recall] WARNING: {len(pairs)}/{n_pairs} pairs after {attempts} attempts",
            flush=True,
        )
    sig_a = minhash_signatures([p[0] for p in pairs], tag="/recall-a")
    sig_b = minhash_signatures([p[1] for p in pairs], tag="/recall-b")
    est = (sig_a == sig_b).sum(axis=1).astype(np.float64) / MINHASH_PERMS
    true_j = np.asarray([p[2] for p in pairs])
    flagged = est >= MINHASH_CAND_EST
    above = true_j >= NEAR_DUPE_JACCARD
    recall_above = float(flagged[above].mean()) if above.any() else float("nan")
    return {
        "n_pairs": len(pairs),
        "true_jaccard_range": [float(true_j.min()), float(true_j.max())],
        "n_true_above_0p8": int(above.sum()),
        "recall_candidates_at_true_ge_0p8": recall_above,
        "mean_abs_est_error": float(np.abs(est - true_j).mean()),
        "flagged_fraction_all": float(flagged.mean()),
        "note": "candidate recall at est>=0.6 among pairs with TRUE J>=0.8 must be ~1.0",
    }


# ── LMSYS re-derivation (cached) ─────────────────────────────────────────────────


def _synthetic_stream(n_rows: int, seed: int = 7):
    """Refusal-safe synthetic LMSYS-shaped rows for --smoke (no network)."""
    rng = np.random.default_rng(seed)
    vocab = [f"tok{i}" for i in range(60)]
    for i in range(n_rows):
        n_w = int(rng.integers(10, 24))
        text = " ".join(str(vocab[int(rng.integers(0, len(vocab)))]) for _ in range(n_w))
        yield {"conversation": [{"content": f"q{i} {text}", "role": "user"}]}


def rederive_used_sets(work_dir: Path, *, smoke: bool) -> dict:
    """Cached ``N50.sample_disjoint_n50k`` (round1 + n10k + n50k-new) + valtest."""
    cache = work_dir / ("rederive_smoke.json" if smoke else "rederive.json")
    if cache.exists():
        d = json.loads(cache.read_text())
        print(f"[rederive] cache hit ({cache.name}; round1={len(d['round1'])})", flush=True)
        return d
    if smoke:
        man = N50.sample_disjoint_n50k(20, 5, 30, stream_iter=list(_synthetic_stream(200)))
        valtest = _smoke_valtest(man["round1"])
    else:
        man = N50.sample_disjoint_n50k(N50.N_ROUND1, N50.N_N10K, N50.N_N50K)
        valtest = N1M._valtest_prompts_from_round1(man["round1"], check_ctx0=True)
    out = {
        "round1": man["round1"],
        "new": man["new"],
        "valtest": valtest,
        "round1_prompt_sha256": man["round1_prompt_sha256"],
        "n10k_prompt_sha256": man["n10k_prompt_sha256"],
        "new_prompt_sha256": man["new_prompt_sha256"],
        "smoke": smoke,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(cache)
    print(
        f"[rederive] streamed round1={len(man['round1'])} n10k={man['n_n10k']} "
        f"new={len(man['new'])} valtest={len(valtest)} (cached)",
        flush=True,
    )
    return out


def _smoke_valtest(round1: list[str]) -> list[str]:
    """Smoke valtest via the SAME fixed_split code path at tiny n (gate exercised)."""
    n = len(round1)
    n_val, n_test = 2, 4
    _r1, val, test = F.fixed_split(n, n - n_val - n_test, n_val, n_test, F.SPLIT_SEED)
    return [round1[i] for i in list(val) + list(test)]


# ── steps ────────────────────────────────────────────────────────────────────────


def step_folds1092(out_dir: Path, *, smoke: bool) -> dict:
    store = resolve_store_dir()
    rows = load_manifest_rows(store)
    n_all = len(rows)
    if smoke:
        rows = rows[:400]
    be = battery_excluded_indices(rows, len(rows))
    trait_excluded = [i for i in range(len(rows)) if rows[i].get("stratum") != "trait_stratum"]
    checks = {}
    for key in ("prefix_id", "query_id"):
        sub_rows = [rows[int(i)] for i in be]
        folds = _folds_from_manifest(sub_rows, len(sub_rows), group_key=key, n_folds=N_FOLDS)
        seen: dict[str, int] = {}
        dup = 0
        for fi, f in enumerate(folds):
            for i in f:
                g = str(sub_rows[int(i)].get(key, ""))
                if g in seen and seen[g] != fi:
                    dup += 1
                seen[g] = fi
        checks[key] = {
            "n_folds": len(folds),
            "fold_sizes": [int(f.size) for f in folds],
            "n_groups": len(seen),
            "groups_in_multiple_folds": int(dup),
        }
        assert dup == 0, f"group {key} appears in >1 fold — fold derivation broken"
    out = {
        "meta": result_meta(step="folds1092", smoke=smoke, fold_seed=FOLD_SEED),
        "n_manifest_rows": n_all,
        "n_rows_considered": len(rows),
        "n_trait_stratum_excluded_population": len(trait_excluded),
        "n_battery_excluded_fit_population": int(be.size),
        "expected": {
            "n_manifest_rows": 21193,
            "n_trait_stratum_excluded": 19708,
            "n_battery_excluded": 17308,
        },
        "counts_match_expected": (
            not smoke and n_all == 21193 and len(trait_excluded) == 19708 and int(be.size) == 17308
        ),
        "fold_checks": checks,
    }
    if not smoke:
        assert out["counts_match_expected"], (
            f"exclusion counts drifted: all={n_all} trait_ex={len(trait_excluded)} "
            f"battery_ex={int(be.size)} (expected 21193/19708/17308)"
        )
    atomic_write_json(out_dir / "folds_1092_check.json", out)
    print(f"[folds1092] OK n={n_all} trait_ex={len(trait_excluded)} fit={int(be.size)}", flush=True)
    return out


def step_n1m(out_dir: Path, work_dir: Path, used: dict, *, smoke: bool) -> dict:
    targets = used["valtest"]
    if smoke:
        pool_prompts = list(used["new"]) + [targets[0], targets[1] + " tail tok"]
        banked = {"note": "smoke — synthetic pool; banked stats not applicable"}
    else:
        man_dir = N1M._download_manifest(N1M.HF_PREFIX, work_dir / "n1m_manifest")
        pool, meta = N1M.read_manifest_pool(man_dir)
        pool_prompts = [r["prompt"] for r in pool]
        banked = meta.get("near_dupe")
        assert len(targets) == 1400, f"valtest reconstruction returned {len(targets)} != 1400"
        # index-digest assert: recompute the exact fixed_split index digests
        _r1, val, test = F.fixed_split(
            N1M.N_ROUND1, N1M.N_ROUND1 - 400 - 1000, 400, 1000, F.SPLIT_SEED
        )
        assert F._sha_ids(val) == N50F.ORIG_VAL_SHA256, "val index sha != pinned original"
        assert F._sha_ids(test) == N50F.ORIG_TEST_SHA256, "test index sha != pinned original"
    battery = overlap_battery(pool_prompts, targets, tag="n1m")
    out = {
        "meta": result_meta(step="n1m", smoke=smoke),
        "banked_near_dupe_stats": banked,
        "battery": battery,
        "verdict": {
            "post_dedup_residual_near": battery["n_near"],
            "expectation": "~0 at threshold (the n1m round dropped 435 exact + 30,437 near)",
        },
    }
    atomic_write_json(out_dir / "n1m_recheck.json", out)
    print(f"[n1m] n_exact={battery['n_exact']} n_near={battery['n_near']}", flush=True)
    return out


def step_n50k(out_dir: Path, used: dict, *, smoke: bool) -> dict:
    round1, new = used["round1"], used["new"]
    if smoke:
        n_val, n_test = 2, 4
        _r1, val, test = F.fixed_split(
            len(round1), len(round1) - n_val - n_test, n_val, n_test, F.SPLIT_SEED
        )
        pinned = {
            "val_sha256": F._sha_ids(val),
            "test_sha256": F._sha_ids(test),
            "source": "smoke self-derived",
        }
        train, _val, _test, diag = _smoke_split(len(round1), len(new), pinned)
        n_test_rows = n_test
    else:
        split_meta = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "eval_results/issue_779/fitter-fair-comparison-n50k/n50k_fits.json"
            ).read_text()
        )["split"]
        pinned = {
            "val_sha256": split_meta["pinned_val_sha256"],
            "test_sha256": split_meta["pinned_test_sha256"],
            "source": split_meta["pinned_source"],
        }
        train, val, test, diag = N50F.build_n50k_split(
            N50.N_N50K, None, pinned, n_train=N50F.N50K_TRAIN, seed=N50F.SPLIT_SEED
        )
        assert diag["train_sha256"] == split_meta["train_sha256"], (
            "reconstructed train index sha != banked train sha — pool re-derivation drifted"
        )
        n_test_rows = len(test)
    n_pass_b = len(round1)
    train_prompts = [round1[i] if i < n_pass_b else new[i - n_pass_b] for i in train]
    battery = overlap_battery(train_prompts, used["valtest"], tag="n50k")
    affected = battery["affected_target_ids"]
    k = len(affected)
    bound = k / max(n_test_rows, 1)
    gate_a = {
        "n_near_affected_targets": k,
        "threshold_targets": 14,
        "p0c_triggered": k > 14,
        "sensitivity_bound_note": (
            f"{k} contaminated targets of {n_test_rows} test rows can move test R2 by at "
            f"most k/n_test x (their mean squared-error share) <= {bound:.4f} of SS_tot"
        ),
        "sensitivity_bound_max_r2_shift": bound,
        "scope_caveat": (
            "LEXICAL criterion only (char-5-gram Jaccard, the #779 recipe); a clean "
            "verdict licenses only 'no lexical near-dupes' — semantic paraphrase "
            "duplicates remain possible"
        ),
    }
    gate_a["disposition"] = _gate_a_disposition(gate_a["p0c_triggered"], k, bound)
    out = {
        "meta": result_meta(step="n50k", smoke=smoke),
        "split_diag": {k_: v for k_, v in diag.items() if k_ != "note"},
        "battery": battery,
        "gate_a": gate_a,
    }
    atomic_write_json(out_dir / "n50k_overlap.json", out)
    print(
        f"[n50k] n_exact={battery['n_exact']} n_near={battery['n_near']} "
        f"affected={k} p0c={gate_a['p0c_triggered']}",
        flush=True,
    )
    return out


def _smoke_split(n_round1: int, n_new: int, pinned: dict):
    """Smoke-sized build_n50k_split twin running the SAME assert code path."""
    n_val, n_test = 2, 4
    r1_train, val, test = F.fixed_split(
        n_round1, n_round1 - n_val - n_test, n_val, n_test, F.SPLIT_SEED
    )
    assert F._sha_ids(val) == pinned["val_sha256"]
    assert F._sha_ids(test) == pinned["test_sha256"]
    pool = np.concatenate([r1_train, np.arange(n_round1, n_round1 + n_new)])
    rng = np.random.default_rng(F.SPLIT_SEED)
    n_target = min(len(pool), len(pool) - 2)
    sel = rng.choice(len(pool), size=n_target, replace=False)
    train = np.sort(pool[sel])
    diag = {"mode": "smoke", "n_train": len(train), "train_sha256": F._sha_ids(train)}
    return train, val, test, diag


def step_chunkcheck(out_dir: Path, used: dict) -> dict:
    """Verify ONE staged capture chunk's realized prompts/ci against the
    re-derived n50k-new pool (A1: prompts ARE in chunks; staging all 96 is
    infeasible at ~1.8 GB/chunk, so the pool re-derivation is primary and this
    is its realized-capture verification)."""
    import torch
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    p = hub.retry_transient(
        lambda: hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            f"{N50.HF_PREFIX}/final_token_capture/shard00_chunk0000.pt",
            repo_type="dataset",
        ),
        what="n50k chunk spot-check download",
    )
    d = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
    ci = [int(x) for x in d["ci"]]
    prompts = list(d["prompts"])
    new = used["new"]
    # ``ci`` is the GLOBAL index into the capture sampling manifest, and that
    # manifest list is the 46,600 NEW prompts ONLY (issue779_ffc_n50k_generate_capture:
    # ``_read_manifest`` returns ``d["new"]``; ``_stack_chunk`` docstring "GLOBAL n50k
    # index (manifest order)") — NOT a round1+new concatenation. The original
    # ``j = c - len(round1)`` offset put every chunk-0 row (ci 0..499) out of range
    # (checked == 0 crash). Validate BY CONTENT (positional match into ``new`` at
    # ``j = c``, plus an index-free membership read against the pool set).
    new_norm = [_norm(t) for t in new]
    new_set = set(new_norm)
    mismatch = 0
    checked = 0
    in_pool_set = 0
    for c, pr in zip(ci, prompts, strict=True):
        npr = _norm(pr)
        if npr in new_set:
            in_pool_set += 1
        if 0 <= c < len(new):
            checked += 1
            if new_norm[c] != npr:
                mismatch += 1
    match_rate = (checked - mismatch) / checked if checked else 0.0
    out = {
        "meta": result_meta(step="chunkcheck"),
        "chunk": "shard00_chunk0000.pt",
        "n_rows_in_chunk": len(ci),
        "ci_min": min(ci),
        "ci_max": max(ci),
        "n_checked_against_rederived_pool": checked,
        "n_mismatches": mismatch,
        "positional_match_rate": match_rate,
        "n_in_rederived_pool_set": in_pool_set,
        "index_space": "ci indexes the NEW-only capture manifest (0..n_new-1)",
    }
    assert checked > 0, "chunk carried no rows inside the re-derived pool index range"
    assert match_rate >= 0.99, (
        f"{mismatch}/{checked} chunk prompts disagree with the re-derived pool "
        f"(positional match-rate {match_rate:.4f} < 0.99; set-membership {in_pool_set}/{len(ci)})"
    )
    atomic_write_json(out_dir / "chunk_spotcheck.json", out)
    print(
        f"[chunkcheck] {checked} rows checked, {mismatch} mismatches "
        f"(match-rate {match_rate:.4f}; set-membership {in_pool_set}/{len(ci)})",
        flush=True,
    )
    return out


def step_recall(out_dir: Path, used: dict, *, smoke: bool) -> dict:
    res = recall_check(used["valtest"], n_pairs=50 if smoke else 1000)
    out = {"meta": result_meta(step="recall", smoke=smoke), "recall": res}
    if not smoke:
        assert res["recall_candidates_at_true_ge_0p8"] >= 0.99, (
            f"MinHash candidate recall {res['recall_candidates_at_true_ge_0p8']:.3f} < 0.99 "
            "at true J>=0.8 — 64 perms insufficient (instrument bug)"
        )
    atomic_write_json(out_dir / "minhash_recall.json", out)
    print(f"[recall] recall@J>=0.8 = {res['recall_candidates_at_true_ge_0p8']}", flush=True)
    return out


def _gate_a_disposition(triggered: bool, k: int, bound: float) -> str:
    """Gate A trip path -> the plan section 8 fallback (concern
    p0c-dedup-refit-deferred): the P0c dedup+refit is infeasible at the
    realized chunk size (~173 GB staging), so on a trip the deliverable is
    complete WITHOUT the refit — the n50k gain is quoted unaudited and the
    n1m gain carries the citable context-arm claim."""
    if triggered:
        return (
            "n50k gain quoted UNAUDITED — n1m is the citable context-arm gain "
            "(P0c refit infeasible at realized chunk size; plan section 8 fallback)"
        )
    return (
        f"clean — n50k gain citable (contamination {k} <= 14 targets; "
        f"sensitivity bound {bound:.4f})"
    )


def gate_smoke_probe() -> None:
    """Degenerate-input probes: every data-dependent gate fires its DESIGNED
    handling once, outside the main leg (fold-check gate inventory)."""
    # (a) NearDupeGate exact + near drops fire
    g = NearDupeGate(["alpha beta gamma delta epsilon zeta eta theta"])
    assert g.is_dupe("Alpha  beta gamma delta epsilon zeta eta theta") and g.n_exact_drop == 1
    assert g.is_dupe("alpha beta gamma delta epsilon zeta eta thetaX") and g.n_near_drop == 1
    # (b) empty-prompt signature -> all-max, never a candidate
    s = minhash_signatures([""], tag="/probe")
    assert (s == np.iinfo(np.uint64).max).all()
    # (c) short text (<5 chars) single-shingle path
    assert _shingle_hashes("ab").size == 1
    # (d) derangement guard (from common) refuses q < 2
    from issue1775_common import _batched_derangements

    fired = False
    try:
        _batched_derangements(np.random.default_rng(0), 2, 1)
    except AssertionError as e:
        fired = "q >= 2" in str(e)
    assert fired, "q<2 derangement guard did not fire"
    # (e) valtest-length gate: wrong round1 length must assert
    fired = False
    try:
        N1M._valtest_prompts_from_round1(["x"] * 7, check_ctx0=False)
    except AssertionError as e:
        fired = "fixed_split anchor" in str(e)
    assert fired, "valtest round1-length gate did not fire"
    # (f) Gate A trip -> plan section 8 fallback disposition (round-2 concern wiring)
    trip = _gate_a_disposition(True, 20, 0.02)
    assert "UNAUDITED" in trip and "n1m" in trip, trip
    assert _gate_a_disposition(False, 3, 0.001).startswith("clean"), "clean branch broke"
    # (g) bounded recall-pair loop fails loud when perturbation never lands in
    # [0.6, 0.9] (round-2 Minor-f): force the miss path by rebinding the
    # perturbation generator to a degenerate J=0 output — the GATE lines
    # (attempt bound + fail-loud raise) execute for real.
    orig_perturb = globals()["_perturb_to_jaccard"]
    globals()["_perturb_to_jaccard"] = lambda t, rng: (t + " zzzzz", 0.0)
    fired = False
    try:
        recall_check(["one two three four five six seven eight"], n_pairs=20, seed=0)
    except RuntimeError as e:
        fired = "near-threshold pairs" in str(e)
    finally:
        globals()["_perturb_to_jaccard"] = orig_perturb
    assert fired, "bounded recall loop did not fail loud on degenerate targets"
    print("[gate-probe] all data-dependent gates fired their designed handling", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 P0 fold-structure check")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "issue_1775" / "p0_work",
    )
    ap.add_argument("--steps", default="folds1092,rederive,n1m,n50k,chunkcheck,recall")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gate-probe", action="store_true", help="degenerate-input gate probes only")
    args = ap.parse_args()
    if args.gate_probe:
        gate_smoke_probe()
        return 0
    out_dir = args.out_dir or eval_dir("fold_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    print(f"[p0] steps={steps} smoke={args.smoke} rss={_rss_gb():.2f} GB", flush=True)
    used: dict | None = None
    if "folds1092" in steps:
        step_folds1092(out_dir, smoke=args.smoke)
    if any(s in steps for s in ("rederive", "n1m", "n50k", "chunkcheck", "recall")):
        used = rederive_used_sets(args.work_dir, smoke=args.smoke)
    if "n1m" in steps:
        step_n1m(out_dir, args.work_dir, used, smoke=args.smoke)
    if "n50k" in steps:
        step_n50k(out_dir, used, smoke=args.smoke)
    if "chunkcheck" in steps and not args.smoke:
        step_chunkcheck(out_dir, used)
    if "recall" in steps:
        step_recall(out_dir, used, smoke=args.smoke)
    print(f"[p0] done rss={_rss_gb():.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
