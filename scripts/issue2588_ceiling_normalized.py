#!/usr/bin/env python3
"""Issue #2588 inline round: two-draw reliability ceilings + ceiling-normalized capability rho.

Tests whether the paper Section 4.8 correlation (held-out mapping R2 vs
Artificial-Analysis index, rho = +0.867 over the ten appendix no-thinking
models) survives normalization by each model's answer-sampling reliability
ceiling. ZERO new generation: every input is banked on the HF data repo at the
panel's pinned revision.

Two ceiling pairings per model, both at the frozen ``layer_star``:

1. ``s42x43`` (PRIMARY, the round's pinned K=2 design: draw 1 = the banked
   seed-42 test generation behind ``test_r2``): the banked ``test_1000``
   ``y_ans`` capture paired with the banked ``ceiling_s43`` draw (same prompts,
   same decoding: temperature 1.0 / top_p 0.95 / cap 2048, different seed).
   Computed here from the capture shards, ci-aligned.
2. ``s43x44`` (sensitivity): the panel rig's own banked
   ``ceiling_two_draw_at_star`` / ``ceiling_retrieval_at_star`` fields
   (``issue2588_run_cell.phase_fits``), read from ``fits_prompt_last.json``.

Estimator formulas are ported verbatim from
``issue2588_run_cell._two_draw_ceiling`` (the #1491
``ceiling_two_draw.ceiling_var_weighted_r`` variance-weighted per-dimension
Pearson) and ``issue2588_run_cell._perrow_hits_cos`` (cosine acc@1 with
tolerance-based mid-ranks).

Output: ``eval_results/issue_2588/ceiling_normalized/ceiling_normalized_capability.json``.

Usage::

    uv run python scripts/issue2588_ceiling_normalized.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # HF token + thread caps before numpy import

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from explore_persona_space.orchestrate import hub as HUB  # noqa: E402


HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
PANEL_JSON = _REPO_ROOT / "eval_results" / "issue_2588" / "mapping_rank_vs_capability.json"
OUT_DIR = _REPO_ROOT / "eval_results" / "issue_2588" / "ceiling_normalized"
PANEL_PREFIX = "issue2588_capability_panel"

# The ten paper-appendix no-thinking cells (clarify-gate pin: the no-thinking
# rows of the banked panel minus Q2.5 32B).
TARGET_CELLS = (
    "q35_0p8b_a",
    "q35_2b_a",
    "q35_4b_a",
    "q35_9b_a",
    "q35_27b_a",
    "q36_27b_a",
    "q38_27b_a",
    "o3_7b_i_a",
    "o31_32b_i_a",
    "q3_32b_a",
)
# Both stages are full draws of the same 1,000-prompt test split; a shortfall
# beyond parse drops means a partial upload / killed shard (#2130 shape).
MIN_ALIGNED_PAIRS = 900


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        env={**os.environ},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _hf_download(path_in_repo: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        HUB.retry_transient(
            lambda: hf_hub_download(
                HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision
            ),
            what=f"hf_hub_download {path_in_repo}",
        )
    )


def _load_stage_layer_y(
    prefix: str, stage: str, layer: int, revision: str
) -> dict[str, np.ndarray]:
    """ci-keyed y_ans for one capture stage at one layer (shard*.npz on HF)."""
    from huggingface_hub import HfApi

    ldir = f"{prefix}/analysis_tensors/capture/{stage}/L{layer:02d}"
    files = sorted(
        p
        for p in HUB.list_hf_files_under_path(
            HfApi(), HF_DATA_REPO, ldir, repo_type="dataset", revision=revision
        )
        if p.endswith(".npz")
    )
    assert files, f"no capture shards under {ldir} at {revision}"
    ids: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for f in files:
        with np.load(_hf_download(f, revision), allow_pickle=False) as z:
            ids.append(z["row_ids"])
            ys.append(z["y_ans"])
    row_ids = np.concatenate(ids)
    y = np.concatenate(ys).astype(np.float64)
    # row_ids: "test_1000_<ci>" / "ceiling_s43_<ci>" — align on the ci suffix
    # (issue2588_run_cell._aligned_ceiling_draws convention).
    ci = np.array([str(r).rsplit("_", 1)[1] for r in row_ids])
    assert len(set(ci)) == len(ci), f"duplicate ci in {ldir}"
    return {"ci": ci, "y": y}


def _two_draw_ceiling(ya: np.ndarray, yb: np.ndarray) -> dict:
    """Verbatim port of issue2588_run_cell._two_draw_ceiling (LF formula)."""
    assert ya.shape == yb.shape, (ya.shape, yb.shape)
    a = ya - ya.mean(axis=0)
    b = yb - yb.mean(axis=0)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0)) + 1e-30
    r_d = (a * b).sum(axis=0) / denom
    var_d = ((ya + yb) / 2.0).var(axis=0)
    w = var_d / (var_d.sum() + 1e-30)
    return {
        "ceiling": float((w * r_d).sum()),
        "n": int(ya.shape[0]),
        "r_d_median": float(np.median(r_d)),
    }


def _perrow_hits_cos(pred: np.ndarray, true: np.ndarray) -> dict:
    """Verbatim port of issue2588_run_cell._perrow_hits_cos (cosine acc@1)."""
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    tn = true / (np.linalg.norm(true, axis=1, keepdims=True) + 1e-12)
    d = 1.0 - pn @ tn.T
    n = d.shape[0]
    d_true = d[np.arange(n), np.arange(n)]
    tol = 1e-9 * np.maximum(np.abs(d_true)[:, None], 1e-12)
    closer = (d < d_true[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - d_true[:, None]) <= tol).sum(axis=1) - 1
    ranks = 1.0 + closer + 0.5 * tied
    return {"hit1": [int(r <= 1) for r in ranks], "rank": [float(r) for r in ranks]}


def _spearman(xs: list[float], ys: list[float]) -> dict:
    rho, p = stats.spearmanr(xs, ys)
    return {"rho": float(rho), "p": float(p), "n": len(xs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR / "ceiling_normalized_capability.json")
    args = parser.parse_args(argv)

    panel = json.loads(PANEL_JSON.read_text())
    rows = {m["cell"]: m for m in panel["maps"]}
    missing = [c for c in TARGET_CELLS if c not in rows]
    assert not missing, f"target cells missing from panel JSON: {missing}"

    revisions = {rows[c]["hf_revision"] for c in TARGET_CELLS}
    assert len(revisions) == 1, f"in-scope cells span multiple revisions: {revisions}"
    revision = revisions.pop()

    per_model: list[dict] = []
    for cell_key in TARGET_CELLS:
        row = rows[cell_key]
        assert row["arm"] == "no-thinking", (cell_key, row["arm"])
        model_key = cell_key.removesuffix("_a")
        star = int(row["layer_star"])
        prefix = f"{PANEL_PREFIX}/{model_key}/nothink"

        # Banked s43 x s44 ceilings from the rig's own fits payload.
        fits = json.loads(
            _hf_download(
                f"{PANEL_PREFIX}/fits/{cell_key}/fits_prompt_last.json", revision
            ).read_text()
        )
        assert int(fits["layer_star"]) == star, (cell_key, fits["layer_star"], star)
        banked_ceil = fits["ceiling_two_draw_at_star"]
        banked_retr = fits["ceiling_retrieval_at_star"]
        assert banked_ceil is not None and banked_retr is not None, cell_key

        # PRIMARY pairing: banked seed-42 test draw x banked seed-43 draw.
        d42 = _load_stage_layer_y(prefix, "test_1000", star, revision)
        d43 = _load_stage_layer_y(prefix, "ceiling_s43", star, revision)
        common, i42, i43 = np.intersect1d(d42["ci"], d43["ci"], return_indices=True)
        n_pairs = int(common.size)
        assert n_pairs >= MIN_ALIGNED_PAIRS, (
            f"{cell_key}: only {n_pairs} aligned s42/s43 pairs "
            f"(test rows {d42['ci'].size}, s43 rows {d43['ci'].size}) — partial store?"
        )
        y42, y43 = d42["y"][i42], d43["y"][i43]
        assert y42.shape[1] == int(row["dimension"]), (cell_key, y42.shape, row["dimension"])
        ceil_4243 = _two_draw_ceiling(y42, y43)
        hits = _perrow_hits_cos(y42, y43)  # draw-1 vectors querying the draw-2 pool
        retr_4243 = {
            "ceiling_acc1_cos": float(np.mean(hits["hit1"])),
            "n_pool": n_pairs,
            "chance": 1.0 / n_pairs,
            "rank_mean": float(np.mean(hits["rank"])),
        }

        test_r2 = float(row["mapping_performance"]["test_r2"])
        acc1 = float(row["mapping_performance"]["test_retrieval_acc1_cos_calibrated"])
        rec = {
            "cell": cell_key,
            "model": row["model"],
            "aa_index": row["aa_index"],
            "dimension": row["dimension"],
            "layer_star": star,
            "test_r2": test_r2,
            "test_retrieval_acc1_cos_calibrated": acc1,
            "test_n": row["mapping_performance"]["test_n"],
            "ceiling_s42x43": {**ceil_4243, "retrieval": retr_4243, "seed_pair": [42, 43]},
            "ceiling_s43x44_banked": {"two_draw": banked_ceil, "retrieval": banked_retr},
            "r2_normalized_s42x43": test_r2 / ceil_4243["ceiling"],
            "r2_normalized_s43x44": test_r2 / float(banked_ceil["ceiling"]),
            "acc1_normalized_s42x43": acc1 / retr_4243["ceiling_acc1_cos"],
            "acc1_normalized_s43x44": acc1 / float(banked_retr["ceiling_acc1_cos"]),
        }
        per_model.append(rec)
        print(
            f"[i2588-ceiling] {cell_key}: n_pairs={n_pairs} "
            f"ceil42x43={ceil_4243['ceiling']:.4f} ceil43x44={banked_ceil['ceiling']:.4f} "
            f"r2={test_r2:.4f} norm42x43={rec['r2_normalized_s42x43']:.4f}"
        )

    aa = [float(m["aa_index"]) for m in per_model]

    def col(key: str) -> list[float]:
        return [float(m[key]) for m in per_model]

    trends = {
        "raw_r2_vs_aa": _spearman(col("test_r2"), aa),
        "normalized_r2_s42x43_vs_aa": _spearman(col("r2_normalized_s42x43"), aa),
        "normalized_r2_s43x44_vs_aa": _spearman(col("r2_normalized_s43x44"), aa),
        "raw_acc1_vs_aa": _spearman(col("test_retrieval_acc1_cos_calibrated"), aa),
        "normalized_acc1_s42x43_vs_aa": _spearman(col("acc1_normalized_s42x43"), aa),
        "normalized_acc1_s43x44_vs_aa": _spearman(col("acc1_normalized_s43x44"), aa),
        "ceiling_s42x43_vs_aa": _spearman([m["ceiling_s42x43"]["ceiling"] for m in per_model], aa),
        "ceiling_s43x44_vs_aa": _spearman(
            [m["ceiling_s43x44_banked"]["two_draw"]["ceiling"] for m in per_model], aa
        ),
    }

    out = {
        "schema_version": 1,
        "meta": {
            "task": 2588,
            "script": "scripts/issue2588_ceiling_normalized.py",
            "git_commit": _git_sha(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "hf_repo": HF_DATA_REPO,
            "hf_revision": revision,
            "panel_json": str(PANEL_JSON.relative_to(_REPO_ROOT)),
            "estimator": (
                "variance-weighted per-dim two-draw Pearson "
                "(issue2588_run_cell._two_draw_ceiling == "
                "issue1491_ladder_fits ceiling_var_weighted_r); retrieval = cosine acc@1 "
                "draw-1 querying draw-2 pool (_perrow_hits_cos)"
            ),
            "pairings": {
                "s42x43": "banked seed-42 test_1000 y_ans x banked ceiling_s43 y_ans (PRIMARY)",
                "s43x44": "rig-banked ceiling_two_draw_at_star / ceiling_retrieval_at_star",
            },
        },
        "per_model": per_model,
        "trends": trends,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[i2588-ceiling] wrote {args.out}")
    for k, v in trends.items():
        print(f"[i2588-ceiling] {k}: rho={v['rho']:+.4f} p={v['p']:.4f} n={v['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
