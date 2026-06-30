#!/usr/bin/env python3
"""Issue #761 — paired Δrho estimator + same-N control + headline assembler (0-GPU).

Plan §4.4-3 / §6.4 / §4.7.

Three pieces, all 0-GPU on the VM:

1. ``_paired_delta_rho_ci(matched_X, mismatched_X, y, ...)`` — the PAIRED
   context-resample Δrho CI (plan §6.4, NEW estimator). The matched and
   recomputed-mismatched arms SHARE the same 50 contexts (same ctx order,
   byte-identical recipe), so a paired resample IS valid. Each draw: sample the
   50 context indices ONCE with replacement (seeded), refit BOTH ridges on that
   SAME resample (LOCO over the resampled contexts, layer re-selected per arm
   under the symmetric rule), compute rho_matched and rho_mismatched on the SAME
   resample, Δrho = rho_matched - rho_mismatched; percentile the per-draw Δrho.
   NEVER calls ``issue658_genre_delta._delta_rho_ci`` (the disjoint-arm
   independent estimator — its arms have DISJOINT probes; ours share contexts).

2. The same-N wrong-questions control fitter (plan §4.7) — re-mean the #658
   UltraChat genre store at matched N per behavior, run the SAME
   ``_run_ridge_pipeline``, then the paired Δrho between matched and same-N.

3. The headline assembler — writes ``matched_predictor_results.json`` with the
   §6.7 per-behavior table + asserts ``recipe_fingerprint`` equal across the
   matched / mismatched-recompute / same-N arms (fail-loud).

Usage::

    uv run python scripts/issue761_paired_bootstrap.py            # full
    uv run python scripts/issue761_paired_bootstrap.py --smoke    # 8 ctx, small B
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue761_common import (
    BEHAVIORS,
    D_EFF,
    HIDDEN,
    N_LAYERS,
    RECIPE_FINGERPRINT,
    REPO_ROOT,
    _all_layers_loco_preds,
    _rho,
    _run_ridge_pipeline,
    e0_rate_vector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue761_paired")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "issue658_theory_assumptions/store"
ULTRACHAT_PREFIX = "issue658_theory_assumptions/store_genre-generalization-ultrachat"
E0_PATH = REPO_ROOT / "eval_results" / "issue_658" / "E0_expression.json"
A33_PATH = REPO_ROOT / "eval_results" / "issue_658" / "analyzer_body_data.json"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_761"
B_BOOTSTRAP = 2000
SAMEN_SEED = 761


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _arm_rho_on_subset(X_by_layer: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float | None:
    """Refit the FULL symmetric pipeline on the resampled context indices ``idx``.

    Layer re-selected per arm under the symmetric predictivity rule (plan §6.4).
    Returns the chosen-layer LOCO held-out rho on the resample, or None if no layer
    yields a valid rho (degenerate resample).
    """
    Xs = X_by_layer[idx]  # (m, n_layers, H)
    ys = y[idx]
    preds_all = _all_layers_loco_preds(Xs, ys, D_EFF)  # (n_layers, m), one batched pass
    best = None
    for li in range(preds_all.shape[0]):
        rho = _rho(preds_all[li], ys)
        if rho is not None and (best is None or rho > best):
            best = rho
    return best


def _paired_delta_rho_ci(
    matched_X: np.ndarray,
    mismatched_X: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = B_BOOTSTRAP,
    seed: int = 761,
) -> dict:
    """Paired context-resample Δrho 95% CI (plan §6.4 — the NEW estimator).

    ``matched_X`` / ``mismatched_X``: ``(N, n_layers, H)`` per-context v0 cubes for the
    two arms (SAME N contexts, SAME ctx order). ``y``: ``(N,)`` E0 rate.

    Each draw samples N context INDICES ONCE with replacement (seeded), refits BOTH
    ridges on that SAME index set (LOCO over the resampled contexts, layer
    re-selected per arm under the symmetric rule), computes rho_matched and
    rho_mismatched on the SAME resample, Δrho = rho_matched - rho_mismatched. Percentiles
    the per-draw Δrho (2.5 / 97.5).

    This is the correlation-aware paired estimator the shared-context design
    requires — NOT the disjoint-arm ``issue658_genre_delta._delta_rho_ci`` (which
    pairs INDEPENDENT per-arm draws after a shuffle, inflating the CI on positively
    correlated arms and biasing toward the straddle-0 kill).

    Returns ``{ci95, point_delta, draws, n_boot, null_overlap}``.
    """
    n = matched_X.shape[0]
    assert matched_X.shape[0] == mismatched_X.shape[0] == y.shape[0], (
        matched_X.shape,
        mismatched_X.shape,
        y.shape,
    )
    # point estimate (full sample, both arms via the shared pipeline)
    rho_m = _run_ridge_pipeline(matched_X, y)["rho"]
    rho_mm = _run_ridge_pipeline(mismatched_X, y)["rho"]
    point_delta = float(rho_m - rho_mm)

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = 0
    max_attempts = 20 * n_boot
    while len(deltas) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)  # resample context indices ONCE per draw
        rm = _arm_rho_on_subset(matched_X, y, idx)
        rmm = _arm_rho_on_subset(mismatched_X, y, idx)  # SAME idx — paired
        if rm is None or rmm is None:
            continue  # degenerate resample — drop + redraw
        deltas.append(float(rm - rmm))
    if len(deltas) < n_boot:
        raise ValueError(
            f"paired Δrho CI: only {len(deltas)}/{n_boot} valid draws in {attempts} "
            "attempts (near-degenerate arms?)"
        )
    arr = np.asarray(deltas, dtype=np.float64)
    lo = float(np.percentile(arr, 2.5))
    hi = float(np.percentile(arr, 97.5))
    return {
        "ci95": [lo, hi],
        "point_delta": point_delta,
        "draws": deltas,
        "n_boot": n_boot,
        "null_overlap": bool(lo <= 0.0 <= hi),
    }


def _independent_delta_rho_ci(
    matched_draws: list[float], mismatched_draws: list[float], *, seed: int
) -> dict:
    """Disjoint-arm INDEPENDENT Δrho CI (the estimator #761 does NOT use for the headline).

    Reproduces ``issue658_genre_delta._delta_rho_ci``'s shape (shuffle each arm's
    single-arm bootstrap draws independently, then per-index diff) — kept HERE only
    so the unit test can compare the paired CI against the independent CI on
    identical synthetic data. NEVER imported by the production headline path.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(matched_draws, dtype=np.float64)
    b = np.asarray(mismatched_draws, dtype=np.float64)
    m = min(len(a), len(b))
    rng.shuffle(a)
    rng.shuffle(b)
    diff = a[:m] - b[:m]
    return {
        "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "n_resamples": m,
    }


# ── same-N wrong-questions control (plan §4.7) ───────────────────────────────


def _load_ultrachat_spans(ctx_id: str) -> tuple[list[torch.Tensor], list[str]] | None:
    """Load the UltraChat answer-span file for one context (or None if absent)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        p = hf_hub_download(
            HF_DATA_REPO, f"{ULTRACHAT_PREFIX}/answer_spans/{ctx_id}.pt", repo_type="dataset"
        )
    except (EntryNotFoundError, FileNotFoundError):
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    return blob["spans"], blob["probes"]


def _samen_v0_for_context(
    ctx_id: str, n_target: int, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    """Same-N v0 ``(28, 3584)`` for one context: mean over ``n_target`` UltraChat probes.

    Draw ``n_target`` probes from the UltraChat span file (without replacement if it
    has >= n_target, else WITH replacement + flag). Returns ``(v0, with_replacement)``.
    """
    loaded = _load_ultrachat_spans(ctx_id)
    if loaded is None:
        raise FileNotFoundError(f"UltraChat span file missing for {ctx_id}")
    spans, _probes = loaded
    n_avail = len(spans)
    with_repl = n_avail < n_target
    if with_repl:
        sel = rng.integers(0, n_avail, size=n_target)
    else:
        sel = rng.permutation(n_avail)[:n_target]
    accum = torch.zeros(N_LAYERS, HIDDEN, dtype=torch.float32)
    for s in sel:
        span = spans[int(s)]  # (28, n_tok, 3584)
        # vectorized mean over answer tokens at every layer at once: (28, n_tok, H) -> (28, H)
        accum += span.float().mean(dim=1)
    v0 = (accum / n_target).numpy()
    return v0, with_repl


def build_samen_X(
    matched_entries: dict[str, dict], behavior: str, kept_ctx: list[str]
) -> tuple[np.ndarray, dict[str, bool]]:
    """``X_sameN (N, 28, 3584)`` over ``kept_ctx`` at the matched N per (C, B) (plan §4.7)."""
    rng = np.random.default_rng(SAMEN_SEED)
    rows = []
    flags: dict[str, bool] = {}
    for c in kept_ctx:
        n_target = matched_entries[(c, behavior)]["matched_n"]
        v0, with_repl = _samen_v0_for_context(c, n_target, rng)
        rows.append(v0)
        if with_repl:
            flags[c] = True
    X = np.stack(rows, axis=0)
    assert X.shape == (len(kept_ctx), N_LAYERS, HIDDEN), X.shape
    return X, flags


# ── matched-arm v0 loader (from the capture .pt shards) ───────────────────────


def load_matched_v0(behavior: str, *, smoke: bool) -> dict[str, np.ndarray]:
    """Matched ``v0(C,B)`` ``(28, 3584)`` per context from the capture .pt shard.

    Prefers the local shard (written by the GPU capture); falls back to HF.
    """
    shard = OUT_DIR / "analysis_tensors" / f"v0_matched_{behavior}.pt"
    if not shard.exists() and not smoke:
        from huggingface_hub import hf_hub_download

        shard = Path(
            hf_hub_download(
                HF_DATA_REPO,
                f"issue761_matched_v0/analysis_tensors/v0_matched_{behavior}.pt",
                repo_type="dataset",
            )
        )
    blob = torch.load(shard, map_location="cpu", weights_only=False)
    return {c: t.float().numpy() for c, t in blob["v0"].items()}


def load_matched_entries() -> dict[tuple[str, str], dict]:
    """matched-N per (ctx, behavior) from v0_matched_by_behavior.json."""
    p = OUT_DIR / "v0_matched_by_behavior.json"
    payload = json.loads(p.read_text())
    return {(e["context_id"], e["behavior"]): e for e in payload["entries"]}


def load_a33_lin_rho() -> dict[str, float]:
    """#658 diff-of-means projection lin_rho per behavior (DESCRIPTIVE column)."""
    if not A33_PATH.exists():
        return {b: None for b in BEHAVIORS}
    with open(A33_PATH) as f:
        d = json.load(f)
    a33 = d.get("betley", {}).get("a33", {})
    return {b: a33.get(b, {}).get("lin_rho") for b in BEHAVIORS}


def run_assemble(*, smoke: bool, n_boot: int, n_smoke_ctx: int = 8) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(E0_PATH) as f:
        e0 = json.load(f)

    # the recompute artifact (mismatched arm)
    mm_path = OUT_DIR / ("mismatched_ridge_smoke.json" if smoke else "mismatched_ridge.json")
    mismatched = json.loads(mm_path.read_text())
    assert mismatched["recipe_fingerprint"] == RECIPE_FINGERPRINT, (
        "mismatched fingerprint != matched"
    )

    # the #658 mismatched v0 cube (shared across behaviors) for the paired arm
    from issue761_recompute_mismatched_ridge import (
        load_mismatched_v0_summaries,
    )

    mm_mean, store_cids = load_mismatched_v0_summaries()
    ctx_pool = store_cids[:n_smoke_ctx] if smoke else store_cids

    # In smoke mode, if the GPU capture has not run, synthesize a tiny matched arm
    # from the mismatched store (matched = mismatched + small seeded noise) so the
    # analysis smoke exercises the FULL assembler path (paired bootstrap, same-N
    # control, fingerprint asserts) end-to-end with NO GPU. The production run
    # always has the real capture .pt shards + v0_matched_by_behavior.json.
    if smoke and not (OUT_DIR / "v0_matched_by_behavior.json").exists():
        _synthesize_smoke_matched(mm_mean, ctx_pool)

    matched_entries = load_matched_entries()
    a33 = load_a33_lin_rho()

    headline: dict[str, dict] = {}
    fingerprints = [RECIPE_FINGERPRINT, mismatched["recipe_fingerprint"]]
    for behavior in BEHAVIORS:
        matched_v0 = load_matched_v0(behavior, smoke=smoke)
        # kept contexts = those present in BOTH the matched capture AND e0 rate,
        # in the store-pool order (the matched-arm convention).
        _, kept_e0 = e0_rate_vector(e0, behavior, ctx_pool)
        kept = [c for c in kept_e0 if c in matched_v0]
        y = np.array([e0["e0"][c][behavior]["rate"] for c in kept], dtype=np.float64)

        X_matched = np.stack([matched_v0[c] for c in kept], axis=0)  # (N, 28, H)
        X_mismatched = assemble_X_mismatched_local(mm_mean, kept)

        # matched arm pipeline
        matched_out = _run_ridge_pipeline(X_matched, y)
        # paired Δrho (matched - mismatched)
        paired = _paired_delta_rho_ci(X_matched, X_mismatched, y, n_boot=n_boot, seed=761)

        # same-N control
        X_samen, samen_flags = build_samen_X(matched_entries, behavior, kept)
        samen_out = _run_ridge_pipeline(X_samen, y)
        samen_fp = samen_out["recipe_fingerprint"]
        fingerprints.append(samen_fp)
        paired_samen = _paired_delta_rho_ci(X_matched, X_samen, y, n_boot=n_boot, seed=761)

        headline[behavior] = {
            "matched_rho": matched_out["rho"],
            "matched_layer": matched_out["chosen_layer"],
            "mismatched_ridge_rho": mismatched["results"][behavior]["mismatched_ridge_rho"],
            "mismatched_layer": mismatched["results"][behavior]["chosen_layer"],
            "samen_mismatched_ridge_rho": samen_out["rho"],
            "samen_layer": samen_out["chosen_layer"],
            "samen_draw_with_replacement": samen_flags,
            "diff_in_means_lin_rho": a33[behavior],
            "paired_delta_rho": paired["point_delta"],
            "paired_delta_rho_ci95": paired["ci95"],
            "paired_delta_rho_null_overlap": paired["null_overlap"],
            "paired_delta_match_vs_samen": paired_samen["point_delta"],
            "paired_delta_match_vs_samen_ci95": paired_samen["ci95"],
            "paired_delta_match_vs_samen_null_overlap": paired_samen["null_overlap"],
            "n_contexts": len(kept),
        }
        logger.info(
            "[%s] matched rho=%.4f mismatched rho=%.4f sameN rho=%.4f Δrho=%.4f CI=%s",
            behavior,
            matched_out["rho"],
            headline[behavior]["mismatched_ridge_rho"],
            samen_out["rho"],
            paired["point_delta"],
            [round(x, 4) for x in paired["ci95"]],
        )

    # fail-loud: every arm's fingerprint must be byte-identical
    assert all(fp == RECIPE_FINGERPRINT for fp in fingerprints), (
        "recipe_fingerprint mismatch across matched / mismatched / same-N arms"
    )

    payload = {
        "task": 761,
        "recipe_fingerprint": RECIPE_FINGERPRINT,
        "headline": headline,
        "metadata": {
            "assembled_at": _now_iso(),
            "smoke": smoke,
            "n_boot": n_boot,
            "n_contexts": len(ctx_pool),
        },
    }
    out_path = OUT_DIR / (
        "matched_predictor_results_smoke.json" if smoke else "matched_predictor_results.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out_path)
    return out_path


def assemble_X_mismatched_local(mean: dict[str, torch.Tensor], kept_ctx: list[str]) -> np.ndarray:
    """``X_mismatched (N, 28, 3584)`` over ``kept_ctx`` (local helper mirroring the recompute)."""
    rows = [mean[c].float().numpy() for c in kept_ctx]
    X = np.stack(rows, axis=0)
    assert X.shape == (len(kept_ctx), N_LAYERS, HIDDEN), X.shape
    return X


# per-behavior matched-N used by the smoke synthesizer (the real values vary
# per context; the smoke just needs >= 50, the matched floor)
_SMOKE_MATCHED_N = {"sycophancy": 200, "refusal": 213, "harmful_compliance": 114}


def _synthesize_smoke_matched(mm_mean: dict[str, torch.Tensor], ctx_pool: list[str]) -> None:
    """Write tiny SYNTHETIC matched .pt shards + v0_matched_by_behavior.json for the smoke.

    matched_v0 = mismatched_v0 + small seeded noise. This lets the 0-GPU analysis
    smoke exercise the full assembler (paired bootstrap, same-N control, fingerprint
    asserts) without the GPU capture. NEVER used outside ``--smoke``.
    """
    rng = np.random.default_rng(761)
    shard_dir = OUT_DIR / "analysis_tensors"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for behavior in BEHAVIORS:
        ctx_map = {}
        for c in ctx_pool:
            base = mm_mean[c].float().numpy()
            noisy = base + 0.01 * rng.standard_normal(base.shape).astype(np.float32)
            t = torch.from_numpy(noisy)
            ctx_map[c] = t
            entries.append(
                {
                    "context_id": c,
                    "behavior": behavior,
                    "matched_n": _SMOKE_MATCHED_N[behavior],
                    "n_used": _SMOKE_MATCHED_N[behavior],
                    "v0_by_layer": t.tolist(),
                }
            )
        torch.save(
            {
                "behavior": behavior,
                "context_ids": list(ctx_map.keys()),
                "v0": ctx_map,
                "recipe_fingerprint": RECIPE_FINGERPRINT,
                "smoke": True,
            },
            shard_dir / f"v0_matched_{behavior}.pt",
        )
    (OUT_DIR / "v0_matched_by_behavior.json").write_text(
        json.dumps({"task": 761, "recipe_fingerprint": RECIPE_FINGERPRINT, "entries": entries})
    )
    logger.info("[smoke] synthesized matched arm (%d ctx x %d beh)", len(ctx_pool), len(BEHAVIORS))


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #761 paired Δrho + same-N control + assembler")
    ap.add_argument("--smoke", action="store_true", help="8 ctx, small B")
    ap.add_argument(
        "--n-boot", type=int, default=None, help="bootstrap draws (default 2000; smoke 50)"
    )
    ap.add_argument("--n-smoke-ctx", type=int, default=8)
    args = ap.parse_args()

    n_boot = args.n_boot if args.n_boot is not None else (50 if args.smoke else B_BOOTSTRAP)
    out_path = run_assemble(smoke=args.smoke, n_boot=n_boot, n_smoke_ctx=args.n_smoke_ctx)

    if args.smoke:
        payload = json.loads(out_path.read_text())
        assert payload["recipe_fingerprint"] == RECIPE_FINGERPRINT
        assert set(payload["headline"].keys()) == set(BEHAVIORS), payload["headline"].keys()
        for behavior in BEHAVIORS:
            h = payload["headline"][behavior]
            assert h["matched_rho"] is not None
            assert h["mismatched_ridge_rho"] is not None
            assert h["samen_mismatched_ridge_rho"] is not None
            assert len(h["paired_delta_rho_ci95"]) == 2
        logger.info(
            "[smoke] PASS — headline table shape %d behaviors x full columns; "
            "recipe_fingerprint equality assert: OK (all 3 arms equal)",
            len(payload["headline"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
