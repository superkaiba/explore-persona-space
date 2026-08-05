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
import issue1491_ladder_generate_capture as GC  # noqa: E402

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


def phase_a_caphit(
    slug: str, hf_prefix: str, scratch: Path, spot_picks: dict, regen_ctx: dict | None = None
) -> dict:
    """Count cap-hit + descriptives per split; return summary + per-ci masks.

    ``regen_ctx`` (post-regen mode — the Path B cap-hit re-gen round):
    ``{"cap": int, "raw_overlays": {split: {ci: row}}, "applied": {split:
    set(ci)}}``. The POST view overlays each APPLIED regen row's (response,
    finish_reason) onto the base stream — the applied set is the regen
    manifest's captured rows, i.e. exactly the rows ``stream_split_merged``
    overlays in phase B, so the raw view and the tensor view merge the SAME
    row set. Masks become RESIDUAL cap-hit masks (rows still truncated at the
    regen cap), and each split reports ``{"pre": ..., "post": ...,
    "n_regen_applied": ..., "n_caphit_not_regenerated": ...}``."""
    out: dict = {"splits": {}, "masks": {}, "spot_rows": []}
    for split, expected in RAW_SPLITS.items():
        overlay = (regen_ctx or {}).get("raw_overlays", {}).get(split, {})
        applied = (regen_ctx or {}).get("applied", {}).get(split, set())
        pre = {"n": 0, "n_len": 0, "n_cjk": 0, "chars": []}
        post = {"n": 0, "n_len": 0, "n_cjk": 0, "chars": []}
        capmap: dict[int, bool] = {}
        rows_seen = 0
        n_applied = 0
        for ci, finish, response, prompt in _iter_raw_rows(hf_prefix, split, scratch):
            is_len = finish == "length"
            pre["n"] += 1
            pre["n_len"] += int(is_len)
            pre["n_cjk"] += int(bool(CJK_RE.search(response)))
            pre["chars"].append(len(response))
            m_finish, m_response = finish, response
            if ci in applied:
                assert is_len, (
                    f"{slug}/{split}: regen overlay ci {ci} was NOT cap-hit in the base "
                    "corpus — the regen namespace does not belong to this base pass"
                )
                r = overlay[ci]
                m_finish, m_response = str(r["finish_reason"]), str(r["response"])
                n_applied += 1
            m_is_len = m_finish == "length"
            post["n"] += 1
            post["n_len"] += int(m_is_len)
            post["n_cjk"] += int(bool(CJK_RE.search(m_response)))
            post["chars"].append(len(m_response))
            if split in MASK_SPLITS:
                # Post-regen mode: the mask is the RESIDUAL cap-hit flag (the
                # restriction read then excludes rows still truncated at the
                # regen cap); base mode: the base cap-hit flag, unchanged.
                capmap[ci] = m_is_len if regen_ctx else is_len
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

        def _stats(d: dict) -> dict:
            arr = np.asarray(d["chars"])
            return {
                "n_rows": d["n"],
                "n_expected": expected,
                "n_cap_hit": d["n_len"],
                "cap_hit_rate": d["n_len"] / max(d["n"], 1),
                "cjk_present_rate": d["n_cjk"] / max(d["n"], 1),
                "response_chars_mean": float(arr.mean()),
                "response_chars_median": float(np.median(arr)),
            }

        if regen_ctx:
            uncovered = pre["n_len"] - n_applied
            out["splits"][split] = {
                "pre": _stats(pre),
                "post": _stats(post),
                "n_regen_applied": n_applied,
                "n_caphit_not_regenerated": uncovered,
            }
            if uncovered:
                logger.warning(
                    "[caphit] %s %s: %d base cap-hit rows NOT covered by the regen overlay "
                    "(capture-dropped or unregenerated) — they stay base rows",
                    slug,
                    split,
                    uncovered,
                )
        else:
            out["splits"][split] = _stats(pre)
        if split in MASK_SPLITS:
            out["masks"][split] = capmap
        logger.info(
            "[caphit] %s %s: n=%d cap_hit_pre=%.4f cap_hit_post=%.4f applied=%d",
            slug,
            split,
            pre["n"],
            pre["n_len"] / max(pre["n"], 1),
            post["n_len"] / max(post["n"], 1),
            n_applied,
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
    *,
    stream_fn=None,
    regen_mode: bool = False,
    out_suffix: str = "",
) -> dict:
    """Restriction re-fit + ceiling recompute for one rung.

    Base mode (default): streams the BASE corpus, hard-asserts the refit
    reproduces the committed ``fits_<slug>.json`` (|Δ| < REFIT_TOL), and the
    restriction excludes BASE cap-hit rows.

    Post-regen mode (``regen_mode=True`` + a merged ``stream_fn``): streams
    the MERGED corpus (regen rows overlaid at read time; ``stream_fn(split)
    -> (cx, vx, ci, gen_cap|None)``), so the refit-vs-committed comparison is
    a REPORTED delta (the data legitimately differs — that delta IS the
    regenerated read), never an assert; ``masks`` must then be the RESIDUAL
    cap-hit masks (phase A post view) so the restriction excludes rows still
    truncated at the regen cap. Output files gain ``out_suffix`` so the base
    round's committed artifacts are never clobbered."""
    dev = torch.device("cpu")
    t0 = time.time()
    if stream_fn is None:
        assert not regen_mode, "regen_mode requires a merged stream_fn"

        def stream_fn(split: str):
            if split.startswith("ceiling_draws/"):
                cx_c, vx_c, ci_c = F._stream_hf_chunks(
                    f"{hf_prefix}/{split}/final_token_capture",
                    layer,
                    scratch,
                    ckpt_dir=None,
                    ckpt_every=0,
                    fresh=True,
                )
                return cx_c, vx_c, list(ci_c), None
            cx_s, vx_s, ci_s = LF._stream_ladder_split(hf_prefix, split, layer, scratch)
            return cx_s, vx_s, list(ci_s), None

    # Stream the three primary splits at the primary layer (production streamer).
    Xtr, Ytr, ci_tr, cap_arr_tr = stream_fn("train_25k")
    Xva, Yva, ci_va, _cap_arr_va = stream_fn("val_400")
    Xte, Yte, ci_te, cap_arr_te = stream_fn("test_1000")
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

    # 1. Full refit (validation vs committed) + bootstrap CI. In post-regen
    # mode the delta vs the BASE-corpus committed fit is a REPORTED quantity
    # (the regenerated read), not an assert — the merged data differs by
    # construction.
    pred_te, meta_full = F.fit_ridge(X, Y, tr, va, te, LF.LAMBDAS, dev, LF.RIDGE_BLOCK)
    r2_full = _pooled_r2(pred_te, Y[te])
    committed_r2 = committed["predictors"]["ridge"]["test_r2"]
    if not regen_mode:
        assert abs(r2_full - committed_r2) < REFIT_TOL, (
            f"{slug}: refit ridge test R² {r2_full:.6f} vs committed {committed_r2:.6f} "
            f"(|Δ| ≥ {REFIT_TOL}) — refit does NOT reproduce the committed fit"
        )
    boot_full = _bootstrap_ci(pred_te, Y[te], N_BOOT, BOOT_SEED)

    # Persist recomputed per-context preds + targets (the lost §6.5 row-2 inputs).
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_extra = {}
    if cap_arr_te is not None:
        preds_extra["gen_cap_te"] = np.asarray(cap_arr_te, dtype=np.int32)
    np.savez_compressed(
        preds_dir / f"{slug}_test_preds_ridge_recomputed{out_suffix}.npz",
        pred_te=pred_te.astype(np.float32),
        y_te=Y[te].astype(np.float32),
        ci_te=np.asarray(ci_te, dtype=np.int64),
        cap_hit_te=cap_te,
        selected_lambda=np.float64(meta_full["selected_lambda"]),
        **preds_extra,
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
    _cxa, vx_a, ci_a, _cap_a_arr = stream_fn("ceiling_draws/seed43")
    _cxb, vx_b, ci_b, _cap_b_arr = stream_fn("ceiling_draws/seed44")
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
    if not regen_mode:
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
    with open(csv_dir / f"{slug}_percontext{out_suffix}.csv", "w") as fh:
        # Post-regen mode adds the per-row provenance column (which cap
        # produced the merged row) — deliverable: record per-row gen cap.
        if cap_arr_te is not None:
            fh.write("ci,cosine_pred_target,sq_err,cap_hit,gen_cap\n")
            for i in range(n_te):
                fh.write(
                    f"{ci_te[i]},{cos[i]:.6f},{se_row[i]:.6f},{int(cap_te[i])},"
                    f"{int(cap_arr_te[i])}\n"
                )
        else:
            fh.write("ci,cosine_pred_target,sq_err,cap_hit\n")
            for i in range(n_te):
                fh.write(f"{ci_te[i]},{cos[i]:.6f},{se_row[i]:.6f},{int(cap_te[i])}\n")

    wall = time.time() - t0
    return {
        "layer": layer,
        "regen_mode": regen_mode,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "n_train_regen_rows": int((np.asarray(cap_arr_tr) != GC.GEN_MAX_TOKENS).sum())
        if cap_arr_tr is not None
        else None,
        "n_test_regen_rows": int((np.asarray(cap_arr_te) != GC.GEN_MAX_TOKENS).sum())
        if cap_arr_te is not None
        else None,
        "ridge_full": {
            "test_r2_refit": r2_full,
            "test_r2_committed": committed_r2,
            "abs_delta_vs_committed": abs(r2_full - committed_r2),
            "committed_is_base_corpus": regen_mode,
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


def _build_regen_ctx(args, slug: str, scratch: Path) -> tuple[dict, object]:
    """Post-regen context for phase A + the merged stream_fn for phase B.

    The applied set per split = the regen manifest's CAPTURED rows — exactly
    the rows ``stream_split_merged`` overlays, so the raw view (phase A) and
    the tensor view (phase B) merge the SAME row set."""
    import issue1491_caphit_regen as R

    cap = int(args.regen_cap)
    raw_overlays: dict[str, dict] = {}
    applied: dict[str, set[int]] = {}
    for split in RAW_SPLITS:
        raw_overlays[split] = R.load_regen_raw_overlay(
            args.hf_root, slug, split, cap, scratch / "regen"
        )
        manifest = R.load_regen_manifest(args.hf_root, slug, split, cap, scratch / "regen")
        applied[split] = {int(r["ci"]) for r in manifest.get("rows", []) if r.get("captured")}
        not_in_overlay = applied[split] - set(raw_overlays[split])
        assert not not_in_overlay, (
            f"{slug}/{split}: {len(not_in_overlay)} manifest-captured cis absent from the "
            "regen raw overlay — manifest/namespace drift"
        )
    return {"cap": cap, "raw_overlays": raw_overlays, "applied": applied}, R


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1491/scale_ladder"))
    ap.add_argument("--fits-dir", type=Path, default=Path("eval_results/issue_1491/scale_ladder"))
    ap.add_argument("--scratch", type=Path, default=Path("data/issue_1491/analyzer_scratch"))
    ap.add_argument("--preds-dir", type=Path, default=Path("data/issue_1491/preds_recomputed"))
    ap.add_argument("--hf-root", default="issue1491_scale_ladder")
    ap.add_argument("--slugs", nargs="*", default=SLUG_ORDER)
    ap.add_argument(
        "--regen-cap",
        type=int,
        default=None,
        help="POST-REGEN mode (Path B): analyze the MERGED corpus (base + "
        "regen_cap<N> overlay). Requires the regen pass "
        "(issue1491_caphit_regen.py) to have run for every --slugs rung; the "
        "restriction becomes residual-cap-hit, the refit-vs-committed check "
        "becomes a reported delta, and every output gains a _postregen_cap<N> "
        "suffix (the base round's committed artifacts are never clobbered).",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    regen_mode = args.regen_cap is not None
    out_suffix = f"_postregen_cap{int(args.regen_cap)}" if regen_mode else ""

    # Pre-draw the spot-check picks (seed 42): 5 rows from test_1000 across rungs.
    rng = np.random.default_rng(SPOT_SEED)
    spot_picks = {
        (SLUG_ORDER[int(rng.integers(0, len(SLUG_ORDER)))], "test_1000", int(rng.integers(0, 1000)))
        for _ in range(N_SPOT)
    }
    spot_rows: list[dict] = []

    for slug in args.slugs:
        per_rung_path = args.out_dir / f"caphit_restriction_{slug}{out_suffix}.json"
        if per_rung_path.exists():
            logger.info("[main] %s already done — skipping", slug)
            continue
        t0 = time.time()
        cfg = LF.LADDER_SCALES[slug]
        layer = cfg["layers"][1]
        hf_prefix = f"{args.hf_root}/{slug}"
        scratch = args.scratch / slug
        committed = json.loads((args.fits_dir / f"fits_{slug}.json").read_text())

        regen_ctx = None
        stream_fn = None
        if regen_mode:
            regen_ctx, R = _build_regen_ctx(args, slug, scratch)

            def stream_fn(split: str, _slug=slug, _layer=layer, _scratch=scratch, _R=R):
                return _R.stream_split_merged(
                    args.hf_root, _slug, split, int(args.regen_cap), _layer, _scratch / "regen"
                )

        a = phase_a_caphit(slug, hf_prefix, scratch, spot_picks, regen_ctx=regen_ctx)
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
            stream_fn=stream_fn,
            regen_mode=regen_mode,
            out_suffix=out_suffix,
        )

        rec = {
            "slug": slug,
            "model": cfg["model"],
            "h_dim": cfg["h_dim"],
            "regen_cap": int(args.regen_cap) if regen_mode else None,
            "caphit_by_split": a["splits"],
            "restriction": b,
            "wall_s_total": time.time() - t0,
        }
        per_rung_path.write_text(json.dumps(rec, indent=1))
        logger.info("[main] %s DONE in %.1f s", slug, rec["wall_s_total"])
        shutil.rmtree(scratch, ignore_errors=True)

    # Combined summary (re-read per-rung files so resumed runs still emit it).
    rungs = {
        s: json.loads((args.out_dir / f"caphit_restriction_{s}{out_suffix}.json").read_text())
        for s in SLUG_ORDER
        if (args.out_dir / f"caphit_restriction_{s}{out_suffix}.json").exists()
    }
    summary = {
        "spot_check_rows_sanitized": spot_rows,
        "rungs": rungs,
    }
    if regen_mode:
        # First-class per-rung response-length report (deliverable 3): mean
        # response length per model size, pre- vs post-regen, plus the
        # residual cap-hit rate at the regen cap.
        summary["response_length_by_rung"] = {
            s: {
                split: {
                    "pre_mean_chars": st["pre"]["response_chars_mean"],
                    "pre_median_chars": st["pre"]["response_chars_median"],
                    "post_mean_chars": st["post"]["response_chars_mean"],
                    "post_median_chars": st["post"]["response_chars_median"],
                    "cap_hit_rate_base": st["pre"]["cap_hit_rate"],
                    "residual_cap_hit_rate": st["post"]["cap_hit_rate"],
                    "n_regen_applied": st["n_regen_applied"],
                }
                for split, st in rec["caphit_by_split"].items()
            }
            for s, rec in rungs.items()
        }
    summary_name = f"caphit_restriction_summary{out_suffix}.json"
    (args.out_dir / summary_name).write_text(json.dumps(summary, indent=1))
    logger.info("[main] summary written -> %s", args.out_dir / summary_name)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
