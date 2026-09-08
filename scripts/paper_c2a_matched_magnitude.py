"""Is the per-element ordering an effect-size artefact? Matched-magnitude read.

Question this answers
---------------------
The comment-2 round found that the map's direction agreement per minimal-pair
element (mean cosine between predicted and observed answer shift) tracks the
SIZE of the observed answer shift: pooled Spearman 0.689 over the headline
pairs, and 0.68 to 0.82 within every element but one. Output format has both
the smallest median answer shift and the lowest cosine.

That raises a competing explanation for the section's claim. The per-element
ordering may not be about format versus persona versus topic at all. It may
rank how large each element's answer-side consequence happens to be, which
then reduces to the same figure's other panel (the map predicts high-variance
directions best).

The two readings make different predictions, and this script separates them:

  ONE CURVE   every element lies on a common cosine-versus-magnitude curve.
              The categorical claim dissolves into an effect-size claim.
  ELEMENT     an element sits off the curve at its own shift magnitude. The
              categorical claim survives and becomes stronger, because it is
              then stated at matched magnitude.

Design
------
Pool the headline pairs of all six elements. DV is the per-pair cosine between
the map's predicted answer shift and the observed one. The magnitude covariate
is ``||dh_A||`` (``norm_obs_tail_L19``), entered as log magnitude, which is the
scale on which the relation is closest to linear.

Three reads, in increasing strictness:

  1. IN-SAMPLE residual: fit the pooled cosine-on-log-magnitude line over ALL
     pairs, report each element's mean residual. Conservative for spotting an
     outlier element, because that element's own pairs help set the line.
  2. LEAVE-ONE-ELEMENT-OUT residual: for each element, fit the line on the
     OTHER five elements and score this element against it. This is the
     principled version of "does the element sit off the curve".
  3. MATCHED WINDOW: restrict to the magnitude interval where at least
     ``--min-elements-in-window`` elements have pairs, and recompute the plain
     per-element mean cosine there. Model-free, at the cost of sample size.

Read (3) against (1) and (2): agreement across a model-based and a model-free
matched read is what licenses the conclusion. The copy baseline is carried
through every read, because a magnitude dependence present in BOTH arms is a
property of the measurement rather than of the map.

Overlap is the load-bearing caveat and is reported, not assumed. Elements
differ in magnitude BY CONSTRUCTION, so a residual read outside the shared
support is extrapolation. The summary records each element's magnitude range,
its overlap with the others, and its n inside the matched window.

No new fit of any map
---------------------
The only fit is a 2-parameter line over 330 pooled pairs (n >> d, so no
well-posedness question), used to residualize a covariate. Every input is a
banked per-pair scalar. Reused rather than reimplemented:
``analysis.paired_ci._quantile_ci`` for the interval convention, and the
element/mask definitions from ``paper_c2a_pair_variance_explained``.
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
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.paired_ci import _quantile_ci  # noqa: E402
from explore_persona_space.task_workflow import repo_root  # noqa: E402

MAP_ARM = "arm_779ce"
COPY_ARM = "arm_iddelta"
PRIMARY_CLASS = {
    "format": "swap",
    "register": "swap",
    "persona": "swap",
    "query_content": "query_content",
}
ELEMENTS = [
    ("Output format", "parent", "format"),
    ("Persona", "ffr", "persona"),
    ("Tone", "parent", "register"),
    ("Answer language", "pilot", "answer_language"),
    ("Question topic", "parent", "query_content"),
    ("One-word topic", "pilot", "query_content_oneword"),
]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get(row: dict, field: str, arm: str) -> float:
    flat = f"{field}_{arm}"
    return float(row[flat]) if flat in row else float(row[field][arm])


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares intercept and slope of y on x (2 parameters, n >> d)."""
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _predict(coef: tuple[float, float], x: np.ndarray) -> np.ndarray:
    return coef[0] + coef[1] * x


def _mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, list[float]]:
    """Mean plus a vectorised pair-bootstrap interval."""
    if values.size == 0:
        return float("nan"), [float("nan"), float("nan")]
    idx = np.random.default_rng(seed).integers(0, values.size, size=(n_boot, values.size))
    lo, hi = _quantile_ci(values[idx].mean(axis=1))
    return float(values.mean()), [float(lo), float(hi)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=25641)
    ap.add_argument("--min-elements-in-window", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = repo_root()
    src = root / "eval_results/issue_2564"
    sources = {
        "parent": _rows(src / "perpair.jsonl"),
        "ffr": _rows(src / "floor-failed-reelicitation/perpair_ffr.jsonl"),
        "pilot": _rows(src / "lang_oneword_pilot/perpair.jsonl"),
    }

    labels: list[str] = []
    mag: list[float] = []
    cos_map: list[float] = []
    cos_copy: list[float] = []
    for label, source_key, axis in ELEMENTS:
        rows = sources[source_key]
        if axis in PRIMARY_CLASS:
            sel = [
                r
                for r in rows
                if r["axis"] == axis
                and r["pair_class"] == PRIMARY_CLASS[axis]
                and r["in_headline_70"]
            ]
        else:
            sel = [r for r in rows if r["axis"] == axis]
        if not sel:
            raise ValueError(f"no pairs selected for {axis}")
        for r in sel:
            labels.append(label)
            mag.append(float(r["norm_obs_tail_L19"]))
            cos_map.append(_get(r, "cos", MAP_ARM))
            cos_copy.append(_get(r, "cos", COPY_ARM))

    lab = np.array(labels)
    mag_arr = np.array(mag, dtype=np.float64)
    if not (mag_arr > 0).all():
        raise ValueError("non-positive answer-shift magnitude: log covariate undefined")
    logmag = np.log(mag_arr)
    cos = {"map": np.array(cos_map, dtype=np.float64), "copy": np.array(cos_copy, dtype=np.float64)}
    names = [label for label, _, _ in ELEMENTS]

    payload: dict = {
        "question": ("is the per-element cosine ordering an artefact of answer-shift magnitude"),
        "dv": "per-pair cosine(predicted answer shift, observed answer shift)",
        "covariate": "log ||dh_A|| (norm_obs_tail_L19)",
        "arms": {"map": MAP_ARM, "copy": COPY_ARM},
        "n_pairs_total": int(mag_arr.size),
        "bootstrap": {"unit": "pair", "draws": args.n_boot, "seed": args.seed},
        "sources": [
            "eval_results/issue_2564/perpair.jsonl",
            "eval_results/issue_2564/floor-failed-reelicitation/perpair_ffr.jsonl",
            "eval_results/issue_2564/lang_oneword_pilot/perpair.jsonl",
        ],
    }

    # --- support / overlap diagnostics (the load-bearing caveat) ---------------
    support = {}
    for name in names:
        m = mag_arr[lab == name]
        support[name] = {
            "n": int(m.size),
            "min": float(m.min()),
            "max": float(m.max()),
            "median": float(np.median(m)),
            "iqr": [float(np.percentile(m, 25)), float(np.percentile(m, 75))],
        }
    payload["magnitude_support"] = support

    # matched window: magnitudes covered by >= min-elements-in-window elements
    lo_edge = np.sort([support[n]["min"] for n in names])[len(names) - args.min_elements_in_window]
    hi_edge = np.sort([support[n]["max"] for n in names])[args.min_elements_in_window - 1]
    in_window = (mag_arr >= lo_edge) & (mag_arr <= hi_edge)
    payload["matched_window"] = {
        "min_elements": args.min_elements_in_window,
        "lo": float(lo_edge),
        "hi": float(hi_edge),
        "n_pairs": int(in_window.sum()),
        "n_by_element": {n: int(((lab == n) & in_window).sum()) for n in names},
    }

    # --- pooled magnitude dependence ------------------------------------------
    payload["pooled_magnitude_dependence"] = {
        arm: {
            "spearman_rho": float(spearmanr(cos[arm], mag_arr).statistic),
            "spearman_p": float(spearmanr(cos[arm], mag_arr).pvalue),
            "line_on_logmag": dict(
                zip(("intercept", "slope"), _fit_line(logmag, cos[arm]), strict=True)
            ),
        }
        for arm in ("map", "copy")
    }

    # --- reads 1 and 2: in-sample and leave-one-element-out residuals ----------
    reads: dict = {}
    for arm in ("map", "copy"):
        y = cos[arm]
        insample_coef = _fit_line(logmag, y)
        resid_in = y - _predict(insample_coef, logmag)
        per_element: dict = {}
        for i, name in enumerate(names):
            sel = lab == name
            others = ~sel
            loeo_coef = _fit_line(logmag[others], y[others])
            resid_loeo = y[sel] - _predict(loeo_coef, logmag[sel])
            mean_in, ci_in = _mean_ci(resid_in[sel], args.n_boot, args.seed + i)
            mean_lo, ci_lo = _mean_ci(resid_loeo, args.n_boot, args.seed + 100 + i)
            m_w, ci_w = _mean_ci(y[sel & in_window], args.n_boot, args.seed + 200 + i)
            per_element[name] = {
                "n": int(sel.sum()),
                "mean_cos": float(y[sel].mean()),
                "residual_insample": {"mean": mean_in, "ci95": ci_in},
                "residual_loeo": {"mean": mean_lo, "ci95": ci_lo},
                "matched_window_mean_cos": {
                    "n": int((sel & in_window).sum()),
                    "mean": m_w,
                    "ci95": ci_w,
                },
            }
        reads[arm] = {
            "insample_line": dict(zip(("intercept", "slope"), insample_coef, strict=True)),
            "per_element": per_element,
        }
    payload["reads"] = reads

    out = args.out or (src / "comment2_matched_magnitude/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    dep = payload["pooled_magnitude_dependence"]
    print(
        f"pooled rho(cos, ||dh_A||): map {dep['map']['spearman_rho']:+.3f} "
        f"(p={dep['map']['spearman_p']:.1e})   copy {dep['copy']['spearman_rho']:+.3f}"
    )
    w = payload["matched_window"]
    print(
        f"matched window (>= {w['min_elements']} elements): "
        f"[{w['lo']:.1f}, {w['hi']:.1f}], {w['n_pairs']} of {mag_arr.size} pairs"
    )
    hdr = (
        f"\n{'element':17s}{'n':>4s}{'|dh_A| med':>11s}{'mean cos':>9s}"
        f"{'resid in-sample':>24s}{'resid LOEO':>24s}{'matched-window cos':>26s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        e = reads["map"]["per_element"][name]
        ri, rl, mw = e["residual_insample"], e["residual_loeo"], e["matched_window_mean_cos"]
        print(
            f"{name:17s}{e['n']:4d}{support[name]['median']:11.1f}{e['mean_cos']:9.3f}"
            f"{ri['mean']:+9.3f} [{ri['ci95'][0]:+.3f},{ri['ci95'][1]:+.3f}]"
            f"{rl['mean']:+9.3f} [{rl['ci95'][0]:+.3f},{rl['ci95'][1]:+.3f}]"
            f"   n={mw['n']:3d} {mw['mean']:6.3f} [{mw['ci95'][0]:.3f},{mw['ci95'][1]:.3f}]"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
