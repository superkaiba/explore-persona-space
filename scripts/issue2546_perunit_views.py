"""Issue #2546 per-question (per-unit) views behind the aggregate R² results.

The clean-result's results 1, 2 and 4 quote pooled held-out R² aggregates whose
per-question predictions are banked as npz on HF
(``issue2546_cotmap/analysis_tensors/preds/arm{K}/``) but whose per-question
errors were never plotted.  This script builds the low-level views on the
OpenThinker3 pair (arm 1) restricted to the MATH corpus rows — the one corpus
whose true answer-state targets we re-stage (streaming the ``post__math``
thinkstore stem shard-by-shard, ~8.4 GB downloaded, <1.5 GB peak resident,
each shard deleted after its layer-19 ``ans_mean`` slice is kept).

``extract``
    Stages the five needed preds npz files + streams the thinkstore stem,
    joins per-question predictions with true targets by row id, computes
    per-question squared errors for: context→answer (``p7_A``, stratum fit),
    boundary→answer (``p7_D``), cross-model pre-context→post-answer
    (``p8_E``), and the 9 interior trajectory positions (``p7_traj``), plus a
    machinery cross-check: the pooled OOF R² of the within-MATH cell
    (``p7_A__math``) recomputed from its own npz + our target slice must match
    the committed ``r2_headline`` (tolerance 5e-3, else raise).
    Writes ``eval_results/issue_2546/perunit/perunit_math_a1.csv`` (+ summary
    JSON with the cross-check numbers).

``figures``
    Renders the three per-unit figures from the CSV via the project style
    (``set_paper_style("blog")`` + ``savefig_paper``) into ``--out-dir``
    (render to a SCRATCH dir against a clean tree, then copy in + commit).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_REV = "8368cc69f887d20931acd8c4d76c142275173728"  # the run's pinned data-repo revision
PREDS_PREFIX = "issue2546_cotmap/analysis_tensors/preds/arm1"
STORE_PREFIX = "issue2546_cotmap/analysis_tensors/thinkstore/arm1/post__math"
LAYER = 19  # frozen headline layer (plan §11)
ANS_KIND = "ans_mean"
T_LABELS = [f"t{v}" for v in range(10, 100, 10)]  # 9 interior positions, t=0.1..0.9

PRED_FILES = [
    "p7_A__does__a1.npz",
    "p7_D__does__a1.npz",
    "p8_E__does__a1.npz",
    "p7_traj__does__a1.npz",
    "p7_A__math__a1.npz",  # cross-check cell (committed r2_headline)
]

OUT_DIR = REPO / "eval_results" / "issue_2546" / "perunit"


def _download(path_in_repo: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            DATA_REPO,
            path_in_repo,
            repo_type="dataset",
            revision=HF_REV,
            local_dir=dest_dir,
        )
    )


def _stream_math_targets(stage: Path) -> tuple[np.ndarray, list[str]]:
    """Stream post__math shards; keep the layer-19 ans_mean slice; delete each shard."""
    import torch
    from huggingface_hub import HfApi

    api = HfApi()
    entries = sorted(
        (
            e.path
            for e in api.list_repo_tree(
                DATA_REPO, path_in_repo=STORE_PREFIX, repo_type="dataset", revision=HF_REV
            )
            if e.path.endswith(".pt")
        ),
    )
    assert entries, f"no shards under {STORE_PREFIX} @ {HF_REV[:8]}"
    ys: list[np.ndarray] = []
    ids: list[str] = []
    for i, p in enumerate(entries):
        local = _download(p, stage)
        sh = torch.load(local, map_location="cpu", weights_only=False)
        kinds = list(sh["kinds_full"])
        ki = kinds.index(ANS_KIND)
        full = sh["full"]  # (B, K_kinds, L_all, H) bf16
        assert full.shape[1] == len(kinds) and full.shape[2] > LAYER, full.shape
        ys.append(full[:, ki, LAYER, :].float().numpy())
        ids.extend(str(r) for r in sh["row_ids"])
        local.unlink()  # stream-and-delete: bound peak disk
        print(f"[extract] shard {i + 1}/{len(entries)}: +{ys[-1].shape[0]} rows", flush=True)
    y = np.concatenate(ys, axis=0)
    assert y.shape[0] == len(ids), (y.shape, len(ids))
    return y, ids


def _sqerr(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = pred.astype(np.float64) - y.astype(np.float64)
    return np.einsum("nd,nd->n", d, d)


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """#825 `_pooled_r2` convention: SS_tot from the evaluated set's OWN mean."""
    pred64 = pred.astype(np.float64)
    true64 = true.astype(np.float64)
    mu = true64.mean(0)
    return 1.0 - float(((true64 - pred64) ** 2).sum() / ((true64 - mu) ** 2).sum())


def cmd_extract(stage: Path) -> None:
    load_dotenv()
    stage.mkdir(parents=True, exist_ok=True)
    y_all, ids_all = _stream_math_targets(stage / "store")
    order = {r: i for i, r in enumerate(ids_all)}

    npz: dict[str, dict[str, np.ndarray]] = {}
    for fname in PRED_FILES:
        local = _download(f"{PREDS_PREFIX}/{fname}", stage / "preds")
        z = np.load(local, allow_pickle=False)
        npz[fname.split("__a1")[0]] = {k: z[k] for k in z.files}

    # --- machinery cross-check on the within-MATH cell -----------------------
    cell = json.loads((REPO / "eval_results/issue_2546/cells/p7_A__math__a1.json").read_text())
    zc = npz["p7_A__math"]
    m = zc["fitted_mask"]
    rows = [str(r) for r in zc["conv_ids"][m]]
    keep = np.array([order[r] for r in rows])
    r2 = _pooled_r2(zc[f"pred_l{LAYER}"][m], y_all[keep])
    committed = float(cell["r2_headline"])
    dev = abs(r2 - committed)
    assert dev < 5e-3, f"cross-check FAILED: recomputed {r2:.6f} vs committed {committed:.6f}"
    print(f"[extract] cross-check p7_A__math: recomputed {r2:.6f} vs committed {committed:.6f}")

    # --- per-question errors on the does-stratum units, MATH rows ------------
    zA, zD, zE, zT = (npz[k] for k in ("p7_A__does", "p7_D__does", "p8_E__does", "p7_traj__does"))
    for z in (zD, zE, zT):
        assert np.array_equal(z["conv_ids"], zA["conv_ids"]), "row sets differ across units"
    is_math = np.array([str(r).startswith("math:") for r in zA["conv_ids"]])
    sel = is_math & zA["fitted_mask"] & zD["fitted_mask"] & zE["fitted_mask"] & zT["fitted_mask"]
    rows = [str(r) for r in zA["conv_ids"][sel]]
    keep = np.array([order[r] for r in rows])
    y = y_all[keep]

    cols: dict[str, np.ndarray] = {
        "sqerr_ctx": _sqerr(zA[f"pred_l{LAYER}"][sel], y),
        "sqerr_boundary": _sqerr(zD[f"pred_l{LAYER}"][sel], y),
        "sqerr_cross_model": _sqerr(zE[f"pred_l{LAYER}"][sel], y),
    }
    tkeys = sorted(
        zT.keys() - {"conv_ids", "folds", "fitted_mask"}, key=lambda k: int(k.split("_l")[1])
    )
    assert len(tkeys) == 9, f"expected 9 trajectory position arrays, got {tkeys}"
    for lab, k in zip(T_LABELS, tkeys, strict=True):
        cols[f"sqerr_{lab}"] = _sqerr(zT[k][sel], y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = "row_id," + ",".join(cols)
    mat = np.column_stack(list(cols.values()))
    with (OUT_DIR / "perunit_math_a1.csv").open("w") as fh:
        fh.write(header + "\n")
        for r, vals in zip(rows, mat, strict=True):
            fh.write(r + "," + ",".join(f"{v:.6g}" for v in vals) + "\n")

    mu = y.astype(np.float64).mean(0)
    ss_tot = float(((y - mu) ** 2).sum())
    summary = {
        "n_rows": len(rows),
        "target": f"{ANS_KIND} @ layer {LAYER}, arm 1 post (OpenThinker3), MATH rows",
        "units": {k: "does-stratum fit, MATH-row subset" for k in cols},
        "crosscheck_p7A_math": {"recomputed": r2, "committed": committed, "abs_dev": dev},
        "subset_r2_from_perunit": {k: 1.0 - float(v.sum()) / ss_tot for k, v in cols.items()},
        "traj_npz_keys_mapped": dict(zip(T_LABELS, tkeys, strict=True)),
        "hf_revision": HF_REV,
    }
    (OUT_DIR / "perunit_math_a1_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["subset_r2_from_perunit"], indent=1))
    print(f"[extract] wrote {OUT_DIR}/perunit_math_a1.csv ({len(rows)} rows)")


def _load_csv() -> tuple[list[str], dict[str, np.ndarray]]:
    path = OUT_DIR / "perunit_math_a1.csv"
    with path.open() as fh:
        header = fh.readline().strip().split(",")
        rows, vals = [], []
        for line in fh:
            parts = line.rstrip("\n").split(",")
            rows.append(parts[0])
            vals.append([float(v) for v in parts[1:]])
    mat = np.asarray(vals)
    return rows, {k: mat[:, i] for i, k in enumerate(header[1:])}


def cmd_figures(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    pal = paper_palette_blog(4)
    rows, cols = _load_csv()
    n = len(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    def scatter(xk: str, yk: str, xlabel: str, ylabel: str, stem: str) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 5.4))
        x, y = cols[xk], cols[yk]
        ax.scatter(x, y, s=4, alpha=0.15, color=pal[0], linewidths=0)
        lim = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.25]
        ax.plot(lim, lim, ls="--", lw=1.0, color="0.4", label="equal error")
        # binned median of y given x (readability companion to the dense cloud)
        qs = np.quantile(x, np.linspace(0, 1, 11))
        bx, by = [], []
        for lo, hi in itertools.pairwise(qs):
            m = (x >= lo) & (x <= hi)
            if m.sum() >= 20:
                bx.append(np.median(x[m]))
                by.append(np.median(y[m]))
        ax.plot(bx, by, marker="o", ms=4, lw=1.4, color=pal[3], label="median per x-decile")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left")
        savefig_paper(fig, stem, dir=out_dir, embed_data=False)
        plt.close(fig)

    scatter(
        "sqerr_ctx",
        "sqerr_boundary",
        "per-question squared error: context-state map",
        "per-question squared error: end-of-thought map",
        "perunit_r1_ctx_vs_boundary",
    )
    scatter(
        "sqerr_ctx",
        "sqerr_cross_model",
        "per-question squared error: post-model context map",
        "per-question squared error: pre-model context map",
        "perunit_r4_cross_vs_within",
    )

    # trajectory: per-question error distribution across the 11 positions
    keys = ["sqerr_ctx"] + [f"sqerr_{t}" for t in T_LABELS] + ["sqerr_boundary"]
    xs = np.array([0.0] + [v / 100 for v in range(10, 100, 10)] + [1.0])
    q = np.array([np.quantile(cols[k], [0.05, 0.25, 0.5, 0.75, 0.95]) for k in keys])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.fill_between(xs, q[:, 0], q[:, 4], alpha=0.18, color=pal[0], label="5th to 95th percentile")
    ax.fill_between(xs, q[:, 1], q[:, 3], alpha=0.35, color=pal[0], label="interquartile range")
    ax.plot(xs, q[:, 2], marker="o", ms=4, lw=1.6, color=pal[0], label="median question")
    ax.set_yscale("log")
    ax.set_xlabel("position in the thinking span (0 = prompt end, 1 = think boundary)")
    ax.set_ylabel("per-question squared error of the predicted answer state")
    ax.legend(loc="upper right")
    savefig_paper(fig, "perunit_r2_trajectory_band", dir=out_dir, embed_data=False)
    plt.close(fig)
    print(f"[figures] wrote 3 per-unit figures ({n} questions) to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--stage-dir", type=Path, required=True)
    fg = sub.add_parser("figures")
    fg.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        cmd_extract(args.stage_dir)
    else:
        cmd_figures(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
