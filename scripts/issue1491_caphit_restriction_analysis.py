#!/usr/bin/env python3
"""Task #1491 analyzer round: authoritative cap-hit + non-cap-hit restriction check.

Implements the binding analysis requirements recorded in ``epm:progress v52``
(items 1 and 3) and repairs the lost paired-bootstrap inputs:

1. **Cap-hit per rung per split from ``raw_completions`` ``finish_reason``**
   (cap-hit == ``finish_reason == 'length'``). Logs are NOT used — abort/resume
   overwrote 0.5B/1.5B ``train_25k`` logs (``epm:progress v44``) and the 14B/32B
   two-pass shape overwrote gen-wave logs entirely (``epm:upload-verification
   v4``/``v5``). The artifact path is exact and complete for every rung.
2. **Restriction check:** re-fit the primary ridge cell per rung (reusing the
   production fitters ``issue779_ffc_n1m_fits.fit_ridge`` + the ladder driver's
   split streaming verbatim), VALIDATE the refit against the committed
   ``fits_<slug>.json`` test R² (|Δ| < 0.01 hard assert), then recompute test R²
   (a) on non-cap-hit test rows with the full-train fit and (b) with train/val
   ALSO restricted to non-cap-hit rows. The two-draw reliability ceiling is
   likewise recomputed on the restricted test subset.
3. **Paired bootstrap CIs** (1,000 draws, seed 42) for the full and restricted
   ridge test R² — the per-context preds the pod-side driver wrote under
   ``data/issue_1491/preds/`` were never uploaded before teardown, so they are
   recomputed here and persisted (plan §10 ``analysis_tensors`` lane).
4. **Per-context per-unit data:** per test context, cosine(v̂, v) + squared-error
   share + cap-hit flag → ``percontext/<slug>_percontext.csv`` (the low-level
   data behind each aggregate R²).

Also collects per-split response-length (chars) + CJK-presence descriptives
(pure counting — no raw text enters any output except <=15-word sanitized
spot-check excerpts, per the content-hygiene rule for real-world-corpus text).

Run from the issue-1491 worktree (imports branch-vendored fit modules):

    MALLOC_ARENA_MAX=2 uv run python scripts/issue1491_caphit_restriction_analysis.py \
        --out-dir eval_results/issue_1491/scale_ladder
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as F  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402

logger = logging.getLogger("issue1491_caphit")

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

# All splits that carry raw_completions, with expected realized row counts.
RAW_SPLITS = {
    "train_25k": 25000,
    "val_400": 400,
    "test_1000": 1000,
    "wc_test_1k": 999,
    "tierB_3600": 3600,
    "ceiling_draws/seed43": 1000,
    "ceiling_draws/seed44": 1000,
}
# Splits whose per-ci masks phase B consumes.
MASK_SPLITS = {"train_25k", "val_400", "test_1000", "ceiling_draws/seed43", "ceiling_draws/seed44"}

SLUG_ORDER = ["scale05", "scale15", "scale3", "scale7_refit", "scale14", "scale32"]

N_BOOT = 1000
BOOT_SEED = 42
SPOT_SEED = 42
N_SPOT = 5
REFIT_TOL = 0.01  # |refit - committed| hard assert (CPU-vs-GPU ridge parity)


def _pooled_r2(pred: np.ndarray, target: np.ndarray) -> float:
    sse = float(((target - pred) ** 2).sum())
    sst = float(((target - target.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / (sst + 1e-30)


def _list_raw_names(prefix: str) -> list[str]:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    return hub.retry_transient(
        lambda: sorted(
            f.path.rsplit("/", 1)[-1]
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient by this function
            for f in HfApi().list_repo_tree(
                C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
            )
            if getattr(f, "size", None) is not None and f.path.endswith(".json")
        ),
        what=f"raw_completions listing ({prefix})",
    )


def _iter_raw_rows(hf_prefix: str, split: str, scratch: Path):
    """Yield (ci:int, finish_reason:str, response:str) for every row of one
    split, downloading each chunk JSON with the production retry envelope and
    unlinking it after parsing (stream-and-delete)."""
    prefix = f"{hf_prefix}/{split}/raw_completions"
    names = _list_raw_names(prefix)
    if not names:
        raise FileNotFoundError(f"no raw_completions under {prefix}")
    scratch.mkdir(parents=True, exist_ok=True)
    for name in names:
        got = Path(F._download_chunk_with_retry(C.HF_DATA_REPO, f"{prefix}/{name}", scratch))
        with open(got, encoding="utf-8") as fh:
            payload = json.load(fh)
        got.unlink()
        for r in payload["rows"]:
            yield (
                int(r["ci"]),
                str(r["finish_reason"]),
                str(r.get("response", "")),
                str(r.get("prompt", "")),
            )


def phase_a_caphit(slug: str, hf_prefix: str, scratch: Path, spot_picks: dict) -> dict:
    """Count cap-hit + descriptives per split; return summary + per-ci masks."""
    out: dict = {"splits": {}, "masks": {}, "spot_rows": []}
    for split, expected in RAW_SPLITS.items():
        n = n_len = n_cjk = 0
        len_chars: list[int] = []
        capmap: dict[int, bool] = {}
        rows_seen = 0
        for ci, finish, response, prompt in _iter_raw_rows(hf_prefix, split, scratch):
            is_len = finish == "length"
            n += 1
            n_len += int(is_len)
            n_cjk += int(bool(CJK_RE.search(response)))
            len_chars.append(len(response))
            if split in MASK_SPLITS:
                capmap[ci] = is_len
            key = (slug, split, rows_seen)
            if key in spot_picks:
                out["spot_rows"].append(
                    {
                        "slug": slug,
                        "split": split,
                        "ci": ci,
                        "finish_reason": finish,
                        "response_chars": len(response),
                        "prompt_excerpt_15w": " ".join(prompt.split()[:15]),
                        "response_excerpt_15w": " ".join(response.split()[:15]),
                    }
                )
            rows_seen += 1
        arr = np.asarray(len_chars)
        out["splits"][split] = {
            "n_rows": n,
            "n_expected": expected,
            "n_cap_hit": n_len,
            "cap_hit_rate": n_len / max(n, 1),
            "cjk_present_rate": n_cjk / max(n, 1),
            "response_chars_mean": float(arr.mean()),
            "response_chars_median": float(np.median(arr)),
        }
        if split in MASK_SPLITS:
            out["masks"][split] = capmap
        logger.info(
            "[caphit] %s %s: n=%d cap_hit=%.4f cjk=%.4f",
            slug,
            split,
            n,
            n_len / max(n, 1),
            n_cjk / max(n, 1),
        )
    return out


def _mask_from(capmap: dict[int, bool], ci_list: list[int], what: str) -> np.ndarray:
    missing = [c for c in ci_list if int(c) not in capmap]
    assert not missing, f"{what}: {len(missing)} capture ci missing from raw_completions"
    return np.asarray([capmap[int(c)] for c in ci_list], dtype=bool)


def _bootstrap_ci(pred: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    """Paired bootstrap over test contexts for pooled variance-weighted R²."""
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    se_row = ((y - pred) ** 2).sum(axis=1)  # (n,)
    q_row = (y.astype(np.float64) ** 2).sum(axis=1)  # (n,)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
    sse = counts @ se_row.astype(np.float64)
    mean_d = (counts @ y.astype(np.float64)) / n  # (n_boot, H)
    sst = counts @ q_row - n * (mean_d**2).sum(axis=1)
    r2s = 1.0 - sse / (sst + 1e-30)
    return {
        "n_boot": n_boot,
        "seed": seed,
        "ci95": [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))],
        "boot_mean": float(r2s.mean()),
    }


def _ceiling_from_draws(A: np.ndarray, B: np.ndarray) -> float:
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    a_c = A - A.mean(axis=0, keepdims=True)
    b_c = B - B.mean(axis=0, keepdims=True)
    num = (a_c * b_c).sum(axis=0)
    den = np.sqrt((a_c**2).sum(axis=0) * (b_c**2).sum(axis=0)) + 1e-30
    r_d = num / den
    Vd = ((A + B) / 2.0).var(axis=0, ddof=0)
    return float((Vd * r_d).sum() / (Vd.sum() + 1e-30))


def phase_b_restriction(
    slug: str,
    hf_prefix: str,
    layer: int,
    masks: dict,
    committed: dict,
    scratch: Path,
    preds_dir: Path,
    percontext_dir: Path,
) -> dict:
    dev = torch.device("cpu")
    t0 = time.time()
    # Stream the three primary splits at the primary layer (production streamer).
    Xtr, Ytr, ci_tr = LF._stream_ladder_split(hf_prefix, "train_25k", layer, scratch)
    Xva, Yva, ci_va = LF._stream_ladder_split(hf_prefix, "val_400", layer, scratch)
    Xte, Yte, ci_te = LF._stream_ladder_split(hf_prefix, "test_1000", layer, scratch)
    X = np.concatenate([Xtr, Xva, Xte], axis=0)
    Y = np.concatenate([Ytr, Yva, Yte], axis=0)
    n_tr, n_va, n_te = len(ci_tr), len(ci_va), len(ci_te)
    tr = np.arange(0, n_tr, dtype=np.int64)
    va = np.arange(n_tr, n_tr + n_va, dtype=np.int64)
    te = np.arange(n_tr + n_va, n_tr + n_va + n_te, dtype=np.int64)
    del Xtr, Ytr, Xva, Yva, Xte, Yte

    cap_tr = _mask_from(masks["train_25k"], ci_tr, f"{slug} train")
    cap_va = _mask_from(masks["val_400"], ci_va, f"{slug} val")
    cap_te = _mask_from(masks["test_1000"], ci_te, f"{slug} test")

    # 1. Full refit (validation vs committed) + bootstrap CI.
    pred_te, meta_full = F.fit_ridge(X, Y, tr, va, te, LF.LAMBDAS, dev, LF.RIDGE_BLOCK)
    r2_full = _pooled_r2(pred_te, Y[te])
    committed_r2 = committed["predictors"]["ridge"]["test_r2"]
    assert abs(r2_full - committed_r2) < REFIT_TOL, (
        f"{slug}: refit ridge test R² {r2_full:.6f} vs committed {committed_r2:.6f} "
        f"(|Δ| ≥ {REFIT_TOL}) — refit does NOT reproduce the committed fit"
    )
    boot_full = _bootstrap_ci(pred_te, Y[te], N_BOOT, BOOT_SEED)

    # Persist recomputed per-context preds + targets (the lost §6.5 row-2 inputs).
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preds_dir / f"{slug}_test_preds_ridge_recomputed.npz",
        pred_te=pred_te.astype(np.float32),
        y_te=Y[te].astype(np.float32),
        ci_te=np.asarray(ci_te, dtype=np.int64),
        cap_hit_te=cap_te,
        selected_lambda=np.float64(meta_full["selected_lambda"]),
    )

    # 2. Restricted-eval: same fit, non-cap-hit test rows only.
    keep_te = ~cap_te
    r2_te_nocap = _pooled_r2(pred_te[keep_te], Y[te][keep_te])
    boot_nocap = _bootstrap_ci(pred_te[keep_te], Y[te][keep_te], N_BOOT, BOOT_SEED)

    # 3. Fully-restricted: refit on non-cap-hit train/val, eval non-cap-hit test.
    tr_r = tr[~cap_tr]
    va_r = va[~cap_va]
    te_r = te[keep_te]
    pred_te_r, meta_r = F.fit_ridge(X, Y, tr_r, va_r, te_r, LF.LAMBDAS, dev, LF.RIDGE_BLOCK)
    r2_refit_nocap = _pooled_r2(pred_te_r, Y[te_r])

    # 4. Ceiling recompute (validation) + restricted variants.
    _cxa, vx_a, ci_a = F._stream_hf_chunks(
        f"{hf_prefix}/ceiling_draws/seed43/final_token_capture",
        layer,
        scratch,
        ckpt_dir=None,
        ckpt_every=0,
        fresh=True,
    )
    _cxb, vx_b, ci_b = F._stream_hf_chunks(
        f"{hf_prefix}/ceiling_draws/seed44/final_token_capture",
        layer,
        scratch,
        ckpt_dir=None,
        ckpt_every=0,
        fresh=True,
    )
    pos_b = {int(c): i for i, c in enumerate(ci_b)}
    pos_te = {int(c): i for i, c in enumerate(ci_te)}
    cap_a = _mask_from(masks["ceiling_draws/seed43"], list(ci_a), f"{slug} seed43")
    cap_b_map = masks["ceiling_draws/seed44"]
    rows = []
    for i_a, c_a in enumerate(ci_a):
        j = pos_b.get(int(c_a))
        k = pos_te.get(int(c_a))
        if j is None or k is None:
            continue
        rows.append((i_a, j, k, cap_a[i_a] or cap_b_map[int(c_a)], cap_te[k]))
    ia = np.asarray([r[0] for r in rows])
    jb = np.asarray([r[1] for r in rows])
    draws_cap = np.asarray([r[3] for r in rows], dtype=bool)
    target_cap = np.asarray([r[4] for r in rows], dtype=bool)
    ceil_full = _ceiling_from_draws(vx_a[ia], vx_b[jb])
    committed_ceil = committed["ceiling_two_draw"]["ceiling_var_weighted_r"]
    assert abs(ceil_full - committed_ceil) < REFIT_TOL, (
        f"{slug}: ceiling recompute {ceil_full:.6f} vs committed {committed_ceil:.6f}"
    )
    keep_t = ~target_cap
    ceil_target_nocap = _ceiling_from_draws(vx_a[ia[keep_t]], vx_b[jb[keep_t]])
    keep_strict = ~target_cap & ~draws_cap
    ceil_strict_nocap = _ceiling_from_draws(vx_a[ia[keep_strict]], vx_b[jb[keep_strict]])

    # 5. Per-context per-unit data (cosine + squared-error share + cap flag).
    y_te = Y[te]
    cos = (pred_te * y_te).sum(axis=1) / (
        np.linalg.norm(pred_te, axis=1) * np.linalg.norm(y_te, axis=1) + 1e-30
    )
    se_row = ((y_te - pred_te) ** 2).sum(axis=1)
    # Durable, committed home (under eval_results/) — these per-context rows are the
    # low-level data behind every aggregate plot, so they must not land in gitignored data/.
    csv_dir = percontext_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_dir / f"{slug}_percontext.csv", "w") as fh:
        fh.write("ci,cosine_pred_target,sq_err,cap_hit\n")
        for i in range(n_te):
            fh.write(f"{ci_te[i]},{cos[i]:.6f},{se_row[i]:.6f},{int(cap_te[i])}\n")

    wall = time.time() - t0
    return {
        "layer": layer,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "ridge_full": {
            "test_r2_refit": r2_full,
            "test_r2_committed": committed_r2,
            "abs_delta_vs_committed": abs(r2_full - committed_r2),
            "selected_lambda": meta_full["selected_lambda"],
            "bootstrap": boot_full,
        },
        "ridge_eval_noncaphit": {
            "n_test_kept": int(keep_te.sum()),
            "test_r2": r2_te_nocap,
            "bootstrap": boot_nocap,
        },
        "ridge_refit_noncaphit": {
            "n_train_kept": int((~cap_tr).sum()),
            "n_val_kept": int((~cap_va).sum()),
            "n_test_kept": int(len(te_r)),
            "test_r2": r2_refit_nocap,
            "selected_lambda": meta_r["selected_lambda"],
        },
        "ceiling": {
            "full_recomputed": ceil_full,
            "committed": committed_ceil,
            "target_noncaphit": ceil_target_nocap,
            "n_target_noncaphit": int(keep_t.sum()),
            "strict_noncaphit_all_draws": ceil_strict_nocap,
            "n_strict": int(keep_strict.sum()),
        },
        "wall_s": wall,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1491/scale_ladder"))
    ap.add_argument("--fits-dir", type=Path, default=Path("eval_results/issue_1491/scale_ladder"))
    ap.add_argument("--scratch", type=Path, default=Path("data/issue_1491/analyzer_scratch"))
    ap.add_argument("--preds-dir", type=Path, default=Path("data/issue_1491/preds_recomputed"))
    ap.add_argument("--hf-root", default="issue1491_scale_ladder")
    ap.add_argument("--slugs", nargs="*", default=SLUG_ORDER)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-draw the spot-check picks (seed 42): 5 rows from test_1000 across rungs.
    rng = np.random.default_rng(SPOT_SEED)
    spot_picks = {
        (SLUG_ORDER[int(rng.integers(0, len(SLUG_ORDER)))], "test_1000", int(rng.integers(0, 1000)))
        for _ in range(N_SPOT)
    }
    spot_rows: list[dict] = []

    for slug in args.slugs:
        per_rung_path = args.out_dir / f"caphit_restriction_{slug}.json"
        if per_rung_path.exists():
            logger.info("[main] %s already done — skipping", slug)
            continue
        t0 = time.time()
        cfg = LF.LADDER_SCALES[slug]
        layer = cfg["layers"][1]
        hf_prefix = f"{args.hf_root}/{slug}"
        scratch = args.scratch / slug
        committed = json.loads((args.fits_dir / f"fits_{slug}.json").read_text())

        a = phase_a_caphit(slug, hf_prefix, scratch, spot_picks)
        spot_rows.extend(a["spot_rows"])
        b = phase_b_restriction(
            slug,
            hf_prefix,
            layer,
            a["masks"],
            committed,
            scratch,
            args.preds_dir,
            args.out_dir / "percontext",
        )

        rec = {
            "slug": slug,
            "model": cfg["model"],
            "h_dim": cfg["h_dim"],
            "caphit_by_split": a["splits"],
            "restriction": b,
            "wall_s_total": time.time() - t0,
        }
        per_rung_path.write_text(json.dumps(rec, indent=1))
        logger.info("[main] %s DONE in %.1f s", slug, rec["wall_s_total"])
        shutil.rmtree(scratch, ignore_errors=True)

    # Combined summary (re-read per-rung files so resumed runs still emit it).
    summary = {
        "spot_check_rows_sanitized": spot_rows,
        "rungs": {
            s: json.loads((args.out_dir / f"caphit_restriction_{s}.json").read_text())
            for s in SLUG_ORDER
            if (args.out_dir / f"caphit_restriction_{s}.json").exists()
        },
    }
    (args.out_dir / "caphit_restriction_summary.json").write_text(json.dumps(summary, indent=1))
    logger.info("[main] summary written")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
