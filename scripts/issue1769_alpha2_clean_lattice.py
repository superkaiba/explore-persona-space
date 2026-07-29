"""
CJK-intrusion-robust lattice classification at one alpha dose for issue #1769.

Computes, for each of three direction classes (evil, hallucination, sycophancy):
  - EXCLUSION treatment: drop CJK-intruded draws, recompute per-arm means
  - ZEROING treatment: zero CJK-intruded draw scores
  - f_d = Delta_decode / Delta_both with question-level cluster bootstrap CIs (B=2000)
  - Lattice classification: decode-driven / prefill-committed / mixed

No new data downloads; reads only existing artifacts under:
  - eval_results/issue_1769/judge/graded_scores.json
  - data/issue_1769/raw_completions/{trait}/{arm}_a{alpha:g}_q{q:02d}_seed42.json

Outputs: eval_results/issue_1769/analysis/alpha{alpha:g}_clean_lattice.json

Usage: uv run python scripts/issue1769_alpha2_clean_lattice.py [--alpha 2.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # sets shared-VM thread caps BEFORE heavy imports (#847/#1144)

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WT = Path(__file__).resolve().parent.parent  # worktree root
GRADED_SCORES = WT / "eval_results/issue_1769/judge/graded_scores.json"
RAW_COMPLETIONS_DIR = WT / "data/issue_1769/raw_completions"
OUT_DIR = WT / "eval_results/issue_1769/analysis"


def out_path_for(alpha: float) -> Path:
    """Alpha-keyed output path (``{alpha:g}`` matches the driver's ``_fmt``:
    2.0 -> the committed ``alpha2_clean_lattice.json``, 1.5 -> ``alpha1.5``)."""
    return OUT_DIR / f"alpha{alpha:g}_clean_lattice.json"


# ---------------------------------------------------------------------------
# CJK unicode range detection
# ---------------------------------------------------------------------------
# Five ranges: CJK Unified, ExtA, Compat, Hiragana+Katakana, Hangul syllables
_CJK_RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0xAC00, 0xD7AF),  # Hangul Syllables
]


def _has_cjk(text: str) -> bool:
    """Return True if text contains any character in the CJK intrusion ranges."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


# ---------------------------------------------------------------------------
# Lattice thresholds
# ---------------------------------------------------------------------------
DECODE_DRIVEN_LB = 0.75  # CI lower bound must exceed this
PREFILL_COMMITTED_LO = -0.25
PREFILL_COMMITTED_HI = 0.25

ARMS = ["neither", "prefill_only", "decode_only", "both"]
STEERED_ARMS = ["prefill_only", "decode_only", "both"]
TRAITS = ["evil", "hallucination", "sycophancy"]
N_QUESTIONS = 20
N_DRAWS = 10
B_BOOTSTRAP = 2000
RNG_SEED = 42


def load_graded_scores(path: Path = GRADED_SCORES) -> dict[str, Any]:
    """Per-item rows from a judge graded_scores.json (default: the parent
    run's committed scores; fu1 passes its own ``--judge-scores`` path)."""
    with open(path) as f:
        d = json.load(f)
    items = list(d["per_item"].values())
    return items


def build_score_arrays(items: list[dict], alpha: float) -> dict[str, dict[str, np.ndarray]]:
    """
    Returns scores[trait][arm] = ndarray shape (N_QUESTIONS, N_DRAWS).

    Neither arm uses alpha=None entries (alpha-invariant baseline).
    Steered arms use entries at the requested ``alpha`` dose.
    """
    scores: dict[str, dict[str, np.ndarray]] = {}
    for trait in TRAITS:
        scores[trait] = {}
        for arm in ARMS:
            scores[trait][arm] = np.full((N_QUESTIONS, N_DRAWS), np.nan)

    for x in items:
        trait = x["trait"]
        arm = x["arm"]
        q = x["question_id"]
        draw = x["draw"]
        item_alpha = x.get("alpha")

        gs = x["graded_score"]
        if gs is None:
            # judge content-drop (drop-never-coerce): leave NaN -> excluded from
            # every treatment's means; surfaced by the coverage WARNING below.
            continue
        # neither baseline: alpha=None
        if arm == "neither" and item_alpha is None:
            scores[trait]["neither"][q, draw] = float(gs)
        # steered arms: the requested alpha dose
        elif arm in STEERED_ARMS and item_alpha == alpha:
            scores[trait][arm][q, draw] = float(gs)

    # Verify coverage
    for trait in TRAITS:
        for arm in ARMS:
            n_missing = np.isnan(scores[trait][arm]).sum()
            if n_missing > 0:
                print(
                    f"  WARNING: {trait}/{arm} has {n_missing}/{N_QUESTIONS * N_DRAWS} "
                    f"missing entries",
                    file=sys.stderr,
                )
    return scores


def build_cjk_flags(
    scores: dict[str, dict[str, np.ndarray]],
    alpha: float,
    raw_dir: Path = RAW_COMPLETIONS_DIR,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load raw completion files for steered arms at the requested alpha and
    compute CJK flags. ``raw_dir`` is the local raw-completions root
    (default: the parent run's; fu1 passes ``--raw-completions-dir``).

    Returns flags[trait][arm] = bool ndarray shape (N_QUESTIONS, N_DRAWS).
    Neither arm: all False (we have no per-alpha raw completions for neither,
    and the graded scores come from alpha=None entries; CJK intrusion is not
    applicable to the baseline arm in the same file structure).
    """
    flags: dict[str, dict[str, np.ndarray]] = {}
    for trait in TRAITS:
        flags[trait] = {}
        # neither: no CJK intrusion concern (baseline arm, no per-alpha raw file)
        flags[trait]["neither"] = np.zeros((N_QUESTIONS, N_DRAWS), dtype=bool)

        for arm in STEERED_ARMS:
            arm_flags = np.zeros((N_QUESTIONS, N_DRAWS), dtype=bool)
            trait_dir = raw_dir / trait
            for q in range(N_QUESTIONS):
                fname = f"{arm}_a{alpha:g}_q{q:02d}_seed42.json"
                fpath = trait_dir / fname
                if not fpath.exists():
                    print(f"  WARNING: missing raw completions file: {fpath}", file=sys.stderr)
                    continue
                with open(fpath) as f:
                    rc = json.load(f)
                draws_list = rc["draws"]  # list of N_DRAWS strings
                for i, text in enumerate(draws_list):
                    arm_flags[q, i] = _has_cjk(text)
            flags[trait][arm] = arm_flags

    return flags


def arm_mean_exclusion(
    score_arr: np.ndarray,  # (N_QUESTIONS, N_DRAWS)
    cjk_arr: np.ndarray,  # (N_QUESTIONS, N_DRAWS) bool
) -> tuple[np.ndarray, int]:
    """
    Per-question mean excluding CJK-intruded draws.
    Returns (q_means shape (N_QUESTIONS,), kept_N).
    """
    masked = np.where(cjk_arr, np.nan, score_arr)
    # Per-question mean ignoring NaN
    q_means = np.nanmean(masked, axis=1)  # (N_QUESTIONS,)
    kept_n = int((~np.isnan(masked)).sum())
    return q_means, kept_n


def arm_mean_zeroing(
    score_arr: np.ndarray,  # (N_QUESTIONS, N_DRAWS)
    cjk_arr: np.ndarray,  # (N_QUESTIONS, N_DRAWS) bool
) -> tuple[np.ndarray, int]:
    """
    Per-question mean with CJK-intruded draws zeroed.
    Returns (q_means shape (N_QUESTIONS,), n_zeroed).
    """
    zeroed = np.where(cjk_arr, 0.0, score_arr)
    # nanmean: judge content-drops (NaN) stay excluded under zeroing too
    q_means = np.nanmean(zeroed, axis=1)  # (N_QUESTIONS,)
    n_zeroed = int(cjk_arr.sum())
    return q_means, n_zeroed


def compute_fd_and_ci(
    q_neither: np.ndarray,  # (N_QUESTIONS,)
    q_prefill: np.ndarray,  # (N_QUESTIONS,)
    q_decode: np.ndarray,  # (N_QUESTIONS,)
    q_both: np.ndarray,  # (N_QUESTIONS,)
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Compute f_d and cluster bootstrap 95% CI.

    f_d = Delta_decode / Delta_both
    Delta_arm = mean(q_arm) - mean(q_neither)  [grand mean]

    Bootstrap: resample questions with replacement, recompute f_d each draw.
    """
    n_q = len(q_neither)
    grand_neither = q_neither.mean()
    delta_prefill = q_prefill.mean() - grand_neither
    delta_decode = q_decode.mean() - grand_neither
    delta_both = q_both.mean() - grand_neither

    fd_point = float(delta_decode / delta_both) if delta_both != 0 else float("nan")

    # Bootstrap: vectorized over B draws
    # idx shape (B, N_QUESTIONS) — resample question indices
    idx = rng.integers(0, n_q, size=(B_BOOTSTRAP, n_q))

    # Stack question-mean arrays: (N_QUESTIONS,) each -> (1, N_QUESTIONS) for broadcasting
    b_neither = q_neither[idx]  # (B, N_QUESTIONS)
    b_prefill = q_prefill[idx]  # (B, N_QUESTIONS)
    b_decode = q_decode[idx]  # (B, N_QUESTIONS)
    b_both = q_both[idx]  # (B, N_QUESTIONS)

    grand_n = b_neither.mean(axis=1)  # (B,)
    d_decode = b_decode.mean(axis=1) - grand_n
    d_both = b_both.mean(axis=1) - grand_n

    # Protect divide-by-zero
    valid = d_both != 0
    fd_boot = np.where(valid, d_decode / d_both, np.nan)

    ci_lo = float(np.nanpercentile(fd_boot, 2.5))
    ci_hi = float(np.nanpercentile(fd_boot, 97.5))

    # Lattice classification
    if (not np.isnan(ci_lo)) and ci_lo > DECODE_DRIVEN_LB:
        classification = "decode-driven"
    elif (not np.isnan(ci_lo)) and PREFILL_COMMITTED_LO < ci_lo and ci_hi < PREFILL_COMMITTED_HI:
        classification = "prefill-committed"
    else:
        classification = "mixed"

    return {
        "delta_neither": float(grand_neither - grand_neither),  # always 0
        "delta_prefill": float(delta_prefill),
        "delta_decode": float(delta_decode),
        "delta_both": float(delta_both),
        "mean_neither": float(grand_neither),
        "mean_prefill": float(q_prefill.mean()),
        "mean_decode": float(q_decode.mean()),
        "mean_both": float(q_both.mean()),
        "f_d": fd_point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "classification": classification,
        "n_questions": n_q,
        "n_bootstrap": B_BOOTSTRAP,
    }


def analyze_trait(
    trait: str,
    scores: dict[str, dict[str, np.ndarray]],
    flags: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run all three treatments (raw, exclusion, zeroing) for one trait."""
    # Raw scores (no CJK correction) — q-level means straight from score arrays
    s = scores[trait]
    f = flags[trait]

    # Summarize CJK intrusion counts
    cjk_counts = {}
    total_draws = N_QUESTIONS * N_DRAWS
    for arm in STEERED_ARMS:
        n_cjk = int(f[arm].sum())
        cjk_counts[arm] = {"n_intruded": n_cjk, "frac": n_cjk / total_draws}

    # ---- RAW (no correction) ----
    def raw_q_means(arm: str) -> np.ndarray:
        arr = s[arm]
        # nanmean in case any entries are nan (missing)
        return np.nanmean(arr, axis=1)

    rq_n = raw_q_means("neither")
    rq_pf = raw_q_means("prefill_only")
    rq_de = raw_q_means("decode_only")
    rq_bo = raw_q_means("both")

    raw_result = compute_fd_and_ci(rq_n, rq_pf, rq_de, rq_bo, rng)
    raw_result["treatment"] = "raw"
    raw_result["kept_n"] = {arm: N_QUESTIONS * N_DRAWS for arm in ARMS}

    # ---- EXCLUSION ----
    # Neither arm: no CJK flags — use full neither scores
    eq_n, neither_kept = arm_mean_exclusion(s["neither"], f["neither"])
    eq_pf, pf_kept = arm_mean_exclusion(s["prefill_only"], f["prefill_only"])
    eq_de, de_kept = arm_mean_exclusion(s["decode_only"], f["decode_only"])
    eq_bo, bo_kept = arm_mean_exclusion(s["both"], f["both"])

    excl_result = compute_fd_and_ci(eq_n, eq_pf, eq_de, eq_bo, rng)
    excl_result["treatment"] = "exclusion"
    excl_result["kept_n"] = {
        "neither": neither_kept,
        "prefill_only": pf_kept,
        "decode_only": de_kept,
        "both": bo_kept,
    }

    # ---- ZEROING ----
    zq_n, n_zero_n = arm_mean_zeroing(s["neither"], f["neither"])
    zq_pf, n_zero_pf = arm_mean_zeroing(s["prefill_only"], f["prefill_only"])
    zq_de, n_zero_de = arm_mean_zeroing(s["decode_only"], f["decode_only"])
    zq_bo, n_zero_bo = arm_mean_zeroing(s["both"], f["both"])

    zero_result = compute_fd_and_ci(zq_n, zq_pf, zq_de, zq_bo, rng)
    zero_result["treatment"] = "zeroing"
    zero_result["n_zeroed"] = {
        "neither": n_zero_n,
        "prefill_only": n_zero_pf,
        "decode_only": n_zero_de,
        "both": n_zero_bo,
    }
    zero_result["kept_n"] = {arm: N_QUESTIONS * N_DRAWS for arm in ARMS}

    return {
        "trait": trait,
        "cjk_intrusion": cjk_counts,
        "raw": raw_result,
        "exclusion": excl_result,
        "zeroing": zero_result,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="steered-arm alpha dose to analyze (default 2.0 — the parent run's dose)",
    )
    ap.add_argument(
        "--judge-scores",
        type=Path,
        default=GRADED_SCORES,
        help="graded_scores.json to read (default: the parent run's committed scores; "
        "fu1: eval_results/issue_1769/judge_fu1/graded_scores.json)",
    )
    ap.add_argument(
        "--raw-completions-dir",
        type=Path,
        default=RAW_COMPLETIONS_DIR,
        help="local raw-completions root for CJK flags (default: the parent run's; "
        "fu1: data/issue_1769/raw_completions_fu1)",
    )
    args = ap.parse_args()
    # Prefix-isolation mirror guard (code-review v3 CONCERN
    # fu1-lattice-partial-invocation-cjk-noop): a non-parent --judge-scores
    # with --raw-completions-dir left at the parent default would silently
    # no-op CJK exclusion via the missing-file WARNING branch. Fail loud
    # pre-analysis instead; the flag-less parent invocation is unaffected.
    if (args.judge_scores != GRADED_SCORES) != (args.raw_completions_dir != RAW_COMPLETIONS_DIR):
        sys.exit(
            "partial fu1 invocation: pass BOTH --judge-scores and "
            "--raw-completions-dir (or neither, for the parent run) — a "
            "mixed invocation silently no-ops CJK exclusion"
        )
    alpha = args.alpha
    out_path = out_path_for(alpha)
    rng = np.random.default_rng(RNG_SEED)

    print("Loading graded scores...", flush=True)
    items = load_graded_scores(args.judge_scores)
    scores = build_score_arrays(items, alpha)

    print("Computing CJK flags from raw completions...", flush=True)
    flags = build_cjk_flags(scores, alpha, args.raw_completions_dir)

    results = []
    for trait in TRAITS:
        print(f"Analyzing trait: {trait}...", flush=True)
        r = analyze_trait(trait, scores, flags, rng)
        results.append(r)

        # Print summary
        for trt in ("raw", "exclusion", "zeroing"):
            res = r[trt]
            cjk_info = r["cjk_intrusion"]
            print(
                f"  [{trt:9s}] f_d={res['f_d']:.3f}  "
                f"CI=[{res['ci_lo']:.3f},{res['ci_hi']:.3f}]  "
                f"class={res['classification']}  "
                f"Δdec={res['delta_decode']:.3f} Δboth={res['delta_both']:.3f}"
            )
        for arm in STEERED_ARMS:
            ci = cjk_info[arm]
            print(
                f"    CJK intruded [{arm}]: {ci['n_intruded']}/{N_QUESTIONS * N_DRAWS} "
                f"({ci['frac'] * 100:.1f}%)"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "description": f"CJK-intrusion-robust lattice classification at alpha={alpha:g}",
                "issue": 1769,
                "n_questions": N_QUESTIONS,
                "n_draws": N_DRAWS,
                "n_bootstrap": B_BOOTSTRAP,
                "rng_seed": RNG_SEED,
                "lattice_thresholds": {
                    "decode_driven_lb": DECODE_DRIVEN_LB,
                    "prefill_committed_lo": PREFILL_COMMITTED_LO,
                    "prefill_committed_hi": PREFILL_COMMITTED_HI,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()
