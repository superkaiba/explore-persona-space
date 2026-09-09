#!/usr/bin/env python3
"""Consolidate panels C and D of the Section 4.2 manuscript figure.

Panel C places the 90 linear retrieval failures of #1901 on the shift-size
plane of the 10,000-context candidate pool: context-vector shift against
five-rollout-averaged answer-vector shift, with a random-pair background and
the rollout reliability floor.  Panel D reports, per controlled change, the
fraction of variance in the observed answer shift that the predicted shift
explains, before and after rescaling every predicted shift by one optimal
factor, with the two refusal-swap groups of #2617 beside the six minimal-pair
elements of #2564.

Plot-only in the sense that matters: no fit, no generation, no model call.
Every input is banked.  The pool answer vectors are reconstructed from the
banked single-draw vectors plus the four resampled draws, the same
five-rollout mean the retrieval evaluation scores.

Writes eval_results/issue_1901/section42_panels.json, read by
scripts/make_paper_section42_figures.py --only information.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Thread caps must land BEFORE the numpy/torch imports below: load_dotenv()
# setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and the BLAS pools freeze at
# import time (#847).
load_dotenv()

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

STAGING = Path("/mnt/eps-data/thomasjiralerspong/issue1901_ctxsim")
KRESAMPLE = ROOT / "data/issue_1901/fig2_pool10k/issue1901_avgpool/analysis_tensors/kresample"
TEST_SHARD = (
    ROOT
    / "data/issue_1901/figure2_five_rollout_scaling/issue1901_avgpool"
    / "analysis_tensors/kresample/V_test_shard00.npz"
)
VE_SOURCE = ROOT / "eval_results/issue_2564/comment2_variance_explained/summary.json"
SVMP_PAIRS = ROOT / "eval_results/issue_2617/svmp_verbharm/perpair.jsonl"
OUT = ROOT / "eval_results/issue_1901/section42_panels.json"

LAYER = 19
K_RESAMPLED = 4
N_TEST = 1000
N_POOL = 10_000
N_BACKGROUND = 60_000
BACKGROUND_SEED = 4202
# The two annotated subcategories whose members share a generation template.
TEMPLATED = ("chemical_article_template", "company_introduction_template")
# Exact over all 1,975,078 held-out natural query pairs; see
# scripts/issue1901_natural_pair_variance_explained.py.
NATURAL_REFERENCE_VE = 0.728


def _read_jsonl(path: Path) -> list[dict]:
    # str.split("\n"), not splitlines(): JSON strings may carry U+2028/U+2029,
    # which splitlines() would treat as row breaks.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def _pool_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Answer and context vectors of the 10,000 candidates, plus the floor.

    Returns ``(answers, contexts, true_columns, rollout_floor)``.  The floor is
    the median distance between two independent five-rollout answer means of
    the same context, estimated by half-splitting each test context's four
    fresh draws and rescaling the two-draw distance by sqrt(2/5).
    """
    pool = json.loads((STAGING / "pool.json").read_text())
    pool_ids = np.asarray(pool["pool_capture_ids"], np.int64)
    test_rows = np.asarray(pool["test_rows"], np.int64)
    true_cols = np.asarray(pool["true_candidate_columns"], np.int64)

    bank = torch.load(
        STAGING / "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt",
        mmap=True,
        weights_only=False,
        map_location="cpu",
    )
    column = list(bank["layers"]).index(LAYER)
    rows = torch.as_tensor(test_rows)
    test_answer_single = bank["v_x"][:, column, :][rows].to(torch.float32).numpy()
    test_context = bank["cx_last"][:, column, :][rows].to(torch.float32).numpy()
    del bank

    shard = np.load(TEST_SHARD)
    shard_row = {int(c): i for i, c in enumerate(np.asarray(shard["ci"], np.int64))}
    draws = np.asarray(shard["V"], np.float32)
    resampled = np.stack([draws[shard_row[int(-(1 + t))]] for t in range(N_TEST)])
    test_answer = (test_answer_single.astype(np.float64) + resampled.sum(1, dtype=np.float64)) / (
        K_RESAMPLED + 1
    )
    half_split = np.linalg.norm(draws[:, :2].mean(1) - draws[:, 2:].mean(1), axis=1)
    floor = float(np.median(half_split) * np.sqrt(2 / 5))

    distractor_ctx = np.load(ROOT / "data/issue_1901/ctxnn_dl/cx_distr_L19.npz")
    ctx_vectors = np.asarray(distractor_ctx["cx"], np.float32)
    ctx_row = {int(c): i for i, c in enumerate(np.asarray(distractor_ctx["ci"], np.int64))}
    distractor_ans = np.load(
        STAGING / "issue1901_metrics/analysis_tensors/distractors_L19.npz", allow_pickle=True
    )
    ans_single = np.asarray(distractor_ans["vx"], np.float32)
    ans_row = {int(c): i for i, c in enumerate(np.asarray(distractor_ans["ci"], np.int64))}
    draw_sums: dict[int, np.ndarray] = {}
    for path in sorted(KRESAMPLE.glob("V_distr_shard*.npz")):
        block = np.load(path)
        summed = np.asarray(block["V"], np.float64).sum(1)
        for i, capture in enumerate(np.asarray(block["ci"], np.int64)):
            draw_sums[int(capture)] = summed[i]

    dim = test_answer.shape[1]
    answers = np.zeros((N_POOL, dim), np.float32)
    contexts = np.zeros((N_POOL, dim), np.float32)
    for column_index, capture in enumerate(pool_ids):
        if capture < 0:
            test_index = int(-capture - 1)
            answers[column_index] = test_answer[test_index]
            contexts[column_index] = test_context[test_index]
        else:
            answers[column_index] = (
                ans_single[ans_row[int(capture)]].astype(np.float64) + draw_sums[int(capture)]
            ) / (K_RESAMPLED + 1)
            contexts[column_index] = ctx_vectors[ctx_row[int(capture)]]
    return answers, contexts, true_cols, floor


def panel_c() -> dict:
    answers, contexts, true_cols, floor = _pool_vectors()
    pool_ids = np.asarray(json.loads((STAGING / "pool.json").read_text())["pool_capture_ids"])
    column_of = {int(capture): j for j, capture in enumerate(pool_ids)}
    audit = json.loads((STAGING / "linear_failures/audit.json").read_text())
    annotations = json.loads((STAGING / "linear_failures/annotations.json").read_text())
    subcategory = {a["query_index"]: a["subcategory"] for a in annotations["annotations"]}

    failures = []
    for row in audit["rows"]:
        if row["rank"] == 1:
            continue
        query = row["query_index"]
        winner = column_of[int(row["top_candidate_ci"])]
        target = true_cols[query]
        failures.append(
            {
                "query_index": int(query),
                "ctx": float(np.linalg.norm(contexts[target] - contexts[winner])),
                "ans": float(np.linalg.norm(answers[target] - answers[winner])),
                "subcategory": subcategory[query],
                "templated": bool(subcategory[query] in TEMPLATED),
            }
        )
    assert len(failures) == 90, len(failures)

    rng = np.random.default_rng(BACKGROUND_SEED)
    left = rng.integers(0, N_POOL, N_BACKGROUND * 2)
    right = rng.integers(0, N_POOL, N_BACKGROUND * 2)
    distinct = left != right
    left, right = left[distinct][:N_BACKGROUND], right[distinct][:N_BACKGROUND]
    bg_ctx = np.linalg.norm(contexts[left] - contexts[right], axis=1)
    bg_ans = np.linalg.norm(answers[left] - answers[right], axis=1)

    median_ans = float(np.median([row["ans"] for row in failures]))
    return {
        "background": {
            "ctx": bg_ctx.round(2).tolist(),
            "ans": bg_ans.round(2).tolist(),
            "n": int(N_BACKGROUND),
            "seed": BACKGROUND_SEED,
            "note": "random distinct pairs of the 10,000 candidates, five-rollout answer means",
        },
        "failures": failures,
        "rollout_floor": floor,
        "background_slope": float(np.sum(bg_ctx * bg_ans) / np.sum(bg_ctx**2)),
        "background_ctx_median": float(np.median(bg_ctx)),
        "background_ans_median": float(np.median(bg_ans)),
        "failure_median_ctx": float(np.median([row["ctx"] for row in failures])),
        "failure_median_ans": median_ans,
        "background_frac_below_failure_median_answer_shift": float((bg_ans <= median_ans).mean()),
    }


def _variance_explained(rows: list[dict], arm: str) -> dict:
    """VE, optimally rescaled VE, size ratio and mean cosine for one group.

    Norms and cosines are enough: ``||o - p||^2 = ||o||^2 + ||p||^2 - 2||o||||p||cos``.
    """
    obs = np.asarray([float(r["norm_obs_tail"]) for r in rows])
    pred = np.asarray([float(r["norm_pred_arm_" + arm]) for r in rows])
    cos = np.asarray([float(r["cos_arm_" + arm]) for r in rows])
    cross = float(np.sum(obs * pred * cos))
    sso, ssp = float(np.sum(obs**2)), float(np.sum(pred**2))
    alpha = cross / ssp
    return {
        "ve": 1 - (sso + ssp - 2 * cross) / sso,
        "ve_size_corrected": 1 - (sso + alpha**2 * ssp - 2 * alpha * cross) / sso,
        "ratio": float(np.sum(pred * obs) / sso),
        "cos_mean": float(cos.mean()),
    }


def panel_d() -> dict:
    banked = json.loads(VE_SOURCE.read_text())
    elements = [
        {
            "label": name,
            "n": int(cell["n_pairs"]),
            "ve": float(cell["map"]["ve"]),
            "ve_size_corrected": float(cell["map"]["ve_rescaled"]),
            "ratio": float(cell["map"]["slope"]),
            "cos_mean": float(cell["map"]["cos_mean"]),
            "kind": "element",
        }
        for name, cell in banked["elements"].items()
    ]
    swaps = _read_jsonl(SVMP_PAIRS)
    for label, group in (("Refusal flip", "flip"), ("Refusal non-flip", "nonflip")):
        rows = [r for r in swaps if r["flip_group"] == group]
        elements.append(
            {
                "label": label,
                "n": len(rows),
                **_variance_explained(rows, "779ce"),
                "kind": "refusal",
            }
        )
    return {
        "elements": elements,
        "natural_reference_ve": NATURAL_REFERENCE_VE,
        "natural_reference_note": (
            "1,975,078 pairs of held-out natural queries, rollout-averaged answer targets"
        ),
        "ve_definition": banked["ve_definition"],
    }


def main() -> None:
    panels = {"panel_c": panel_c(), "panel_d": panel_d()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(panels, indent=1) + "\n")
    c = panels["panel_c"]
    print(
        f"rollout floor {c['rollout_floor']:.2f}  background slope {c['background_slope']:.3f}  "
        f"ctx {c['background_ctx_median']:.1f}  ans {c['background_ans_median']:.1f}"
    )
    print(f"{'element':<20}{'n':>5}{'VE':>8}{'size-corrected':>16}")
    for row in panels["panel_d"]["elements"]:
        print(
            f"  {row['label']:<18}{row['n']:>5}{row['ve']:>8.2f}{row['ve_size_corrected']:>16.2f}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
