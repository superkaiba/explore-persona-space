"""Comment-2 response: shift sizes and variance explained for the controlled minimal pairs.

Question this answers
---------------------
Section 4.2 currently reports, per minimal-pair element, the through-origin
calibration slope of ``||W dh_C||`` on ``||dh_A||`` -- a pure MAGNITUDE read.
Two reviewer comments ask for the two quantities that read actually leaves out:

  (1) the euclidean norm of the CONTEXT-side and ANSWER-side shift induced by
      each element, i.e. how much of the distinction the context summary
      carries at all, and how large the answer-side consequence is;
  (2) the VARIANCE of the observed answer shift EXPLAINED by the linear map's
      prediction, i.e. how much of the distinction the map actually captures,
      rather than whether it gets the size right.

A magnitude ratio near one is compatible with a prediction that points in the
wrong direction, so (2) is not a re-scaling of the shipped number.

Definitions
-----------
Per pair, all vectors live in the layer-19 answer space:

  dh_C   = context-side shift  = v_C(a) - v_C(b)      (the ``arm_iddelta`` arm
           IS this vector: the learned bias cancels exactly in a difference,
           asserted in issue2564_analysis.identity_cancellation_check)
  dh_A   = observed answer shift = mean answer activation(a) - (b)
  W dh_C = predicted answer shift under the frozen map (``arm_779ce``)

Pooled variance explained, over the pairs of one element:

  VE = 1 - sum ||dh_A - pred||^2 / sum ||dh_A||^2

expanded from the banked per-pair scalars via
``||dh_A - pred||^2 = ||dh_A||^2 + ||pred||^2 - 2 ||dh_A|| ||pred|| cos``.
VE = 0 is the no-shift predictor; VE < 0 means predicting no shift beats the
map. ``VE_rescaled`` applies the single pooled scale ``alpha`` that maximises
VE, which isolates the DIRECTION component from the magnitude component.

Reliability ceiling: the target ``dh_A`` is a finite-rollout mean, so
``ceiling = 1 - sum noise^2 / sum ||dh_A||^2`` bounds any predictor's VE.

No new fit
----------
Every quantity is read from banked per-pair scalars. Nothing is fitted here,
so there is no n_train-vs-d well-posedness question. The magnitude slope is
recomputed with the same convention as
``issue2564_analysis.through_origin_slope`` (sum(p*o)/sum(o*o)) and asserted
against the banked ``axis_slope`` for every element, which also pins the
headline pair mask (primary class AND ``in_headline_70``).

Reused rather than reimplemented: ``analysis.paired_ci._quantile_ci`` for the
bootstrap interval convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/scipy imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paired_ci import _quantile_ci  # noqa: E402
from explore_persona_space.task_workflow import repo_root  # noqa: E402

MAP_ARM = "arm_779ce"
COPY_ARM = "arm_iddelta"  # identity + bias; the bias cancels in a pair difference
PRIMARY_CLASS = {
    "format": "swap",
    "register": "swap",
    "persona": "swap",
    "query_content": "query_content",
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get(row: dict, field: str, arm: str) -> float:
    """Per-pair scalar under either the nested or the flat pilot schema."""
    flat = f"{field}_{arm}"
    return float(row[flat]) if flat in row else float(row[field][arm])


def _pooled(obs: np.ndarray, pred: np.ndarray, cos: np.ndarray) -> dict[str, float]:
    """Pooled magnitude slope, mean cosine, VE and best-rescaled VE."""
    sq_obs = float((obs**2).sum())
    sq_err = float((obs**2 + pred**2 - 2.0 * obs * pred * cos).sum())
    alpha = float((obs * pred * cos).sum() / (pred**2).sum())
    sq_err_scaled = float((obs**2 + (alpha * pred) ** 2 - 2.0 * obs * alpha * pred * cos).sum())
    return {
        "slope": float((pred * obs).sum() / sq_obs),
        "cos_mean": float(cos.mean()),
        "ve": 1.0 - sq_err / sq_obs,
        "ve_rescaled": 1.0 - sq_err_scaled / sq_obs,
        "alpha": alpha,
    }


def _boot(
    obs: np.ndarray,
    pred: np.ndarray,
    cos: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Vectorised pair bootstrap: one (n_boot, n) index matrix, no per-pair loop."""
    n = obs.size
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    o, p, c = obs[idx], pred[idx], cos[idx]
    sq_obs = (o**2).sum(axis=1)
    sq_err = (o**2 + p**2 - 2.0 * o * p * c).sum(axis=1)
    alpha = (o * p * c).sum(axis=1) / (p**2).sum(axis=1)
    sq_err_s = (o**2 + (alpha[:, None] * p) ** 2 - 2.0 * o * alpha[:, None] * p * c).sum(axis=1)
    return {
        "slope_ci95": _quantile_ci((p * o).sum(axis=1) / sq_obs),
        "cos_mean_ci95": _quantile_ci(c.mean(axis=1)),
        "ve_ci95": _quantile_ci(1.0 - sq_err / sq_obs),
        "ve_rescaled_ci95": _quantile_ci(1.0 - sq_err_s / sq_obs),
    }


def _element(rows: list[dict], n_boot: int, seed: int) -> dict:
    obs = np.array([r["norm_obs_tail_L19"] for r in rows], dtype=np.float64)
    noise = np.array([r.get("noise_norm", np.nan) for r in rows], dtype=np.float64)
    ctx = np.array([_get(r, "norm_pred", COPY_ARM) for r in rows], dtype=np.float64)
    out: dict = {
        "n_pairs": int(obs.size),
        # comment 2, part 1: the two shift sizes
        "norm_ctx_shift": {
            "median": float(np.median(ctx)),
            "mean": float(ctx.mean()),
            "rms": float(np.sqrt((ctx**2).mean())),
        },
        "norm_answer_shift": {
            "median": float(np.median(obs)),
            "mean": float(obs.mean()),
            "rms": float(np.sqrt((obs**2).mean())),
        },
        # how much of the context-side distinction survives into the answer
        "answer_over_context_rms": float(np.sqrt((obs**2).sum() / (ctx**2).sum())),
        "reliability_ceiling_ve": float(1.0 - np.nansum(noise**2) / (obs**2).sum()),
        "median_noise_norm": float(np.nanmedian(noise)),
    }
    for name, arm in (("map", MAP_ARM), ("copy_context_shift", COPY_ARM)):
        pred = np.array([_get(r, "norm_pred", arm) for r in rows], dtype=np.float64)
        cos = np.array([_get(r, "cos", arm) for r in rows], dtype=np.float64)
        stats = _pooled(obs, pred, cos)
        stats.update(_boot(obs, pred, cos, n_boot, seed))
        stats["norm_pred_median"] = float(np.median(pred))
        out[name] = stats
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=25640)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = repo_root()
    src = root / "eval_results/issue_2564"
    parent = _rows(src / "perpair.jsonl")
    ffr = _rows(src / "floor-failed-reelicitation/perpair_ffr.jsonl")
    pilot = _rows(src / "lang_oneword_pilot/perpair.jsonl")
    banked = json.loads((src / "minpair_delta.json").read_text())["axes"]
    banked_ffr = json.loads(
        (src / "floor-failed-reelicitation/minpair_delta_ffr.json").read_text()
    )["axes"]
    banked_pilot = json.loads((src / "lang_oneword_pilot/summary.json").read_text())[
        "calibration_slope"
    ][MAP_ARM]

    # (label, source rows, axis, banked slope) -- element order follows the paper.
    elements = [
        ("Output format", parent, "format", banked["format"]["calibration"][MAP_ARM]["axis_slope"]),
        ("Persona", ffr, "persona", banked_ffr["persona"]["calibration"][MAP_ARM]["axis_slope"]),
        ("Tone", parent, "register", banked["register"]["calibration"][MAP_ARM]["axis_slope"]),
        ("Answer language", pilot, "answer_language", banked_pilot["answer_language"]),
        (
            "Question topic",
            parent,
            "query_content",
            banked["query_content"]["calibration"][MAP_ARM]["axis_slope"],
        ),
        (
            "One-word topic",
            pilot,
            "query_content_oneword",
            banked_pilot["query_content_oneword"],
        ),
    ]

    results: dict[str, dict] = {}
    for label, source, axis, banked_slope in elements:
        if axis in PRIMARY_CLASS:
            sel = [
                r
                for r in source
                if r["axis"] == axis
                and r["pair_class"] == PRIMARY_CLASS[axis]
                and r["in_headline_70"]
            ]
            mask = f"pair_class == {PRIMARY_CLASS[axis]!r} AND in_headline_70"
        else:
            sel = [r for r in source if r["axis"] == axis]
            mask = "all pilot pairs (no fire mask in the pilot)"
        if not sel:
            raise ValueError(f"no pairs selected for {axis}")
        stats = _element(sel, args.n_boot, args.seed)
        # Fail loud if the mask no longer reproduces the shipped calibration slope.
        got = stats["map"]["slope"]
        if not np.isclose(got, banked_slope, rtol=0, atol=1e-9):
            raise AssertionError(
                f"{label}: recomputed slope {got!r} != banked axis_slope {banked_slope!r}; "
                "the headline pair mask has drifted"
            )
        stats["axis"] = axis
        stats["pair_mask"] = mask
        stats["banked_axis_slope"] = float(banked_slope)
        results[label] = stats

    payload = {
        "question": "comment 2: context- and answer-side shift norms, plus variance explained",
        "arms": {"map": MAP_ARM, "copy_context_shift": COPY_ARM},
        "layer": 19,
        "pooling": "tail",
        "bootstrap": {"unit": "pair", "draws": args.n_boot, "seed": args.seed},
        "ve_definition": "1 - sum||dh_A - pred||^2 / sum||dh_A||^2 (0 = no-shift predictor)",
        "sources": [
            "eval_results/issue_2564/perpair.jsonl",
            "eval_results/issue_2564/floor-failed-reelicitation/perpair_ffr.jsonl",
            "eval_results/issue_2564/lang_oneword_pilot/perpair.jsonl",
        ],
        "elements": results,
    }
    out = args.out or (src / "comment2_variance_explained/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    hdr = (
        f"{'element':17s}{'n':>4s}{'|dh_C|':>8s}{'|dh_A|':>8s}{'A/C':>6s}"
        f"{'slope':>7s}{'cos':>7s}{'VE map':>18s}{'VE rescaled':>9s}"
        f"{'VE copy':>9s}{'cos copy':>9s}{'ceil':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, s in results.items():
        m, c = s["map"], s["copy_context_shift"]
        ci = m["ve_ci95"]
        print(
            f"{label:17s}{s['n_pairs']:4d}"
            f"{s['norm_ctx_shift']['median']:8.1f}{s['norm_answer_shift']['median']:8.1f}"
            f"{s['answer_over_context_rms']:6.2f}{m['slope']:7.2f}{m['cos_mean']:7.3f}"
            f"{m['ve']:8.3f} [{ci[0]:5.2f},{ci[1]:5.2f}]{m['ve_rescaled']:9.3f}"
            f"{c['ve']:9.2f}{c['cos_mean']:9.3f}{s['reliability_ceiling_ve']:7.3f}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
