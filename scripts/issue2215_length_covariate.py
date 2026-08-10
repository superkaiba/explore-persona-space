#!/usr/bin/env python3
"""Issue #2215 free-analysis follow-up — completion-LENGTH covariate for DV2.

Question: do completion-length differences carry the DV2 (answer-vector
shift) per-type ranking, and how much of the exploratory context-shift <->
answer-shift coupling (rho=0.64 over 39 cells, ``coupling.json``
exploratory across_cells) survives a length covariate?

Inputs (existing artifacts only — 0 GPU-h):

- ``eval_results/issue_2215/perpair/dv2_pairs.jsonl`` — per-pair tail-pooled
  L19 answer-shift norms (``norm_dva``), written by
  ``issue2215_analysis.compute_dv2`` (mean over valid draws per side).
- ``eval_results/issue_2215/perpair/dv1_pairs.jsonl`` — per-pair context-end
  shift norms (``norm_dvc``, slot ``ce``) for the coupling recompute.
- ``eval_results/issue_2215/{dv2_answer_shift,coupling}.json`` — parity
  checks (per-cell medians, the raw rho=0.64 read).
- Banked anchor rollouts (HF data repo, revision-PINNED at the parent
  report's verified pin): ``issue2162_ctxinfo/raw_completions/anchors``
  (16 jsonl shards, ~29.3 MB) + ``.../analysis_tensors/vc_bank/bank.json``
  (~2.4 MB, the pair table a/b context ids). Staged via the canonical
  ``hub.stage_hub_file`` helper into a /tmp root and DELETED after the
  reduce (total ~32 MB — far under every staging threshold).

Length metric: **tokens**, read from the banked ``n_completion_tokens``
field the parent capture wrote into every anchors-jsonl row (retokenized
completion length, project tokenizer, ``add_special_tokens=False`` — see
``issue2162_run.capture_answer_states``). No re-tokenization, no character
proxy. Validity parity: a draw is valid iff ``n_completion_tokens > 0`` —
the exact ``empty_rows`` rule the DV2 means used; the computed per-context
n_valid is ASSERTED equal to ``dv2_pairs``'s ``n_valid_a``/``n_valid_b``.

Content hygiene: only ``context_id`` / ``draw`` / ``n_completion_tokens``
are read from the anchors rows; the ``text`` field is never touched.

Analyses (LINEAR only; vectorized numpy; descriptive):

1. Per-pair: |Delta mean completion length| (valid draws, per side) vs the
   pair's ``norm_dva`` — pooled Spearman + bootstrap CI (reusing
   ``issue2215_analysis.bootstrap_spearman``, seed [SEED_BOOT, 9]) and the
   per-cell Spearman distribution (39 cells x 36 pairs).
2. Per-type: pooled OLS ``norm_dva ~ 1 + |Delta len|``; adjusted norm =
   ``norm - b1*(x - xbar)`` (residual re-centred at the grand-mean length
   difference, so the split-half yardstick scale is preserved). Rank
   stability = Spearman(raw per-cell medians, adjusted per-cell medians);
   noise-floor verdicts (median/yardstick > 1) recomputed on adjusted
   medians, flips named.
3. Coupling: the exploratory across-cells Spearman(dv1 ce median norm, dv2
   median norm) recomputed RAW (must reproduce coupling.json, same seed
   key [SEED_BOOT, 8]) and with length-ADJUSTED dv2 medians (same seed key
   => identical bootstrap index draws, maximally paired CI comparison);
   plus the within-cell across-pairs median rho, raw and adjusted.

Outputs: ``eval_results/issue_2215/length_covariate/length_covariate.json``
+ ``figures/issue_2215/fu_length_covariate.{png,pdf,meta.json}`` (raw and
residualized panels, per-type points labeled). Seconds-scale, checkpointless
(single bounded pass; no resume state).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numeric imports (shared-VM thread caps)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:  # script-mode sibling import (gotchas.md #823)
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue2215_analysis import (  # noqa: E402
    SEED_BOOT,
    bootstrap_spearman,
    spearman_obs,
)
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2215.length_covariate")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
REVISION_PIN = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"  # parent report pin (plan §10)
ANCHOR_TEXT_PREFIX = "issue2162_ctxinfo/raw_completions/anchors"
BANK_PATH_IN_REPO = "issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json"
K_DRAWS = 10
N_CONTEXTS = 1404
N_PAIRS = 1404
N_CELLS = 39
# (batch, worker) 16-shard anchors layout — mirrors issue2215_run.SHARDS.
SHARDS = [(b, w) for b in ("gate", "rest") for w in range(8)]

_P = paper_palette(8)
DV2_COLOR = _P[7]  # dv2_tail — the sibling figures' answer-vector color
REF_COLOR = "#888888"


# ── staging + reduce ───────────────────────────────────────────────────


def stage_inputs(staging_root: Path) -> tuple[Path, list[Path]]:
    """Stage bank.json + the 16 anchors jsonl shards at the revision pin."""
    staging_root.mkdir(parents=True, exist_ok=True)
    bank_path = staging_root / "bank.json"
    hub.stage_hub_file(
        HF_DATA_REPO, BANK_PATH_IN_REPO, bank_path, repo_type="dataset", revision=REVISION_PIN
    )
    shard_paths: list[Path] = []
    t0 = time.monotonic()
    for i, (batch, w) in enumerate(SHARDS):
        name = f"anchors_{batch}_w{w}.jsonl"
        target = staging_root / name
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{ANCHOR_TEXT_PREFIX}/{name}",
            target,
            repo_type="dataset",
            revision=REVISION_PIN,
        )
        shard_paths.append(target)
        print(
            f"[stage] unit {i + 1}/{len(SHARDS)} {name} elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )
    return bank_path, shard_paths


def reduce_lengths(shard_paths: list[Path]) -> dict[str, dict]:
    """Per-context mean completion length (tokens) over VALID (n_tok>0) draws.

    Reads ONLY context_id/draw/n_completion_tokens per row (content hygiene:
    the text field is never accessed). Text-mode line iteration — never
    splitlines() (gotchas.md #950). Fail-loud on a missing token-count field,
    duplicate (context_id, draw) keys, or a coverage mismatch.
    """
    per_ctx: dict[str, list[tuple[int, int]]] = defaultdict(list)
    n_rows = 0
    for path in shard_paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                assert "n_completion_tokens" in r, (
                    f"{path.name}: row lacks n_completion_tokens (keys: {sorted(r)[:8]})"
                )
                per_ctx[r["context_id"]].append((int(r["draw"]), int(r["n_completion_tokens"])))
                n_rows += 1
    assert n_rows == N_CONTEXTS * K_DRAWS, f"expected {N_CONTEXTS * K_DRAWS} rows, got {n_rows}"
    assert len(per_ctx) == N_CONTEXTS, f"expected {N_CONTEXTS} contexts, got {len(per_ctx)}"
    out: dict[str, dict] = {}
    for cid, draws in per_ctx.items():
        dup = [d for d, n in Counter(d for d, _ in draws).items() if n > 1]
        assert not dup, f"{cid}: duplicate draws {dup[:3]}"
        toks = np.array([t for _, t in draws], dtype=np.float64)
        valid = toks > 0  # the capture's empty_rows rule: empty text -> 0 tokens
        out[cid] = {
            "n_valid": int(valid.sum()),
            "mean_len_tokens": float(toks[valid].mean()) if valid.any() else float("nan"),
        }
    return out


def read_jsonl(path: Path) -> list[dict]:
    """Text-mode JSONL reader (never splitlines; gotchas.md #950)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    assert rows, f"no rows in {path}"
    return rows


# ── analyses (vectorized numpy; descriptive) ───────────────────────────


def build_pair_frame(
    dv2_rows: list[dict],
    dv1_rows: list[dict],
    bank_pairs: list[dict],
    ctx_len: dict[str, dict],
) -> dict[str, np.ndarray]:
    """Join dv2 per-pair norms with per-pair |Delta mean length| + dv1 ce norms.

    Asserts n_valid parity between the banked-text reduce and the dv2 rows
    (same validity rule => must match exactly).
    """
    ab = {p["pair_id"]: (p["a"], p["b"]) for p in bank_pairs}
    dvc = {r["pair_id"]: r["norm_dvc"] for r in dv1_rows if r["slot"] == "ce"}
    cells, norms, lendiff, dvc_norms, yard = [], [], [], [], []
    mismatches = 0
    for r in dv2_rows:
        if not r["included"]:
            continue
        a, b = ab[r["pair_id"]]
        for side, key in ((a, "n_valid_a"), (b, "n_valid_b")):
            if ctx_len[side]["n_valid"] != r[key]:
                mismatches += 1
        cells.append(r["cell"])
        norms.append(r["norm_dva"])
        lendiff.append(abs(ctx_len[a]["mean_len_tokens"] - ctx_len[b]["mean_len_tokens"]))
        dvc_norms.append(dvc[r["pair_id"]])
        yard.append(r["cell_splithalf_floor"])
    assert mismatches == 0, (
        f"{mismatches} n_valid mismatches between banked-text reduce and dv2 rows "
        "(validity-rule drift — refusing to proceed)"
    )
    return {
        "cell": np.array(cells),
        "norm_dva": np.array(norms, dtype=np.float64),
        "abs_len_diff": np.array(lendiff, dtype=np.float64),
        "norm_dvc": np.array(dvc_norms, dtype=np.float64),
        "yardstick": np.array(yard, dtype=np.float64),
    }


def per_cell_median(cell: np.ndarray, val: np.ndarray) -> dict[str, float]:
    return {c: float(np.median(val[cell == c])) for c in sorted(set(cell.tolist()))}


def analyze(frame: dict[str, np.ndarray], boot_b: int, dv2_json: dict, coupling_json: dict) -> dict:
    """Analyses 1-3 (module docstring). Returns the JSON-ready result dict."""
    x = frame["abs_len_diff"]
    y = frame["norm_dva"]
    cell = frame["cell"]
    cells = sorted(set(cell.tolist()))
    assert len(cells) == N_CELLS, f"{len(cells)} cells != {N_CELLS}"

    # 1. pooled + per-cell Spearman(|Delta len|, norm_dva)
    pooled = bootstrap_spearman(x, y, boot_b, [SEED_BOOT, 9])
    pooled.pop("draws", None)
    per_cell_rho = {c: spearman_obs(x[cell == c], y[cell == c]) for c in cells}
    rho_vals = np.array(list(per_cell_rho.values()), dtype=np.float64)

    # 2. pooled OLS + adjusted (residual re-centred at grand-mean |Delta len|)
    b1, b0 = np.polyfit(x, y, 1)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    y_adj = y - b1 * (x - x.mean())
    raw_med = per_cell_median(cell, y)
    adj_med = per_cell_median(cell, y_adj)
    med_raw = np.array([raw_med[c] for c in cells])
    med_adj = np.array([adj_med[c] for c in cells])
    rank_stability = spearman_obs(med_raw, med_adj)
    # parity: recomputed raw medians vs the committed dv2 per-cell record
    dv2_cells = {c: dv2_json["per_cell"][c]["tail"] for c in cells}
    med_parity = max(abs(raw_med[c] - dv2_cells[c]["primary"]["median_norm"]) for c in cells)
    yard_by_cell = {c: float(frame["yardstick"][cell == c][0]) for c in cells}
    verdicts = {}
    flips = []
    for c in cells:
        raw_gt1 = raw_med[c] / yard_by_cell[c] > 1.0
        adj_gt1 = adj_med[c] / yard_by_cell[c] > 1.0
        verdicts[c] = {
            "raw_ratio": raw_med[c] / yard_by_cell[c],
            "adj_ratio": adj_med[c] / yard_by_cell[c],
            "raw_gt1": bool(raw_gt1),
            "adj_gt1": bool(adj_gt1),
        }
        if raw_gt1 != adj_gt1:
            flips.append(c)

    # 3. coupling recompute: across-cells (raw must reproduce coupling.json)
    dvc_med = per_cell_median(cell, frame["norm_dvc"])
    xc = np.array([dvc_med[c] for c in cells])
    raw_coup = bootstrap_spearman(xc, med_raw, boot_b, [SEED_BOOT, 8])
    raw_coup.pop("draws", None)
    committed = coupling_json["exploratory"]["across_cells"]
    assert abs(raw_coup["obs"] - committed["obs"]) < 1e-9, (
        f"raw coupling recompute {raw_coup['obs']} != committed {committed['obs']}"
    )
    adj_coup = bootstrap_spearman(xc, med_adj, boot_b, [SEED_BOOT, 8])
    adj_coup.pop("draws", None)
    # within-cell across-pairs (raw + adjusted), median over cells
    wc_raw = {c: spearman_obs(frame["norm_dvc"][cell == c], y[cell == c]) for c in cells}
    wc_adj = {c: spearman_obs(frame["norm_dvc"][cell == c], y_adj[cell == c]) for c in cells}

    return {
        "length_metric": (
            "tokens — banked n_completion_tokens (retokenized completion length, "
            "project tokenizer, add_special_tokens=False); valid draw = n_tok > 0"
        ),
        "n_pairs": int(len(y)),
        "n_cells": len(cells),
        "analysis1_length_vs_shift": {
            "pooled_spearman": pooled,
            "pooled_pearson_r": pearson_r,
            "per_cell_spearman": per_cell_rho,
            "per_cell_spearman_summary": {
                "median": float(np.median(rho_vals)),
                "q25": float(np.percentile(rho_vals, 25)),
                "q75": float(np.percentile(rho_vals, 75)),
                "min": float(rho_vals.min()),
                "max": float(rho_vals.max()),
                "n_positive": int((rho_vals > 0).sum()),
            },
        },
        "analysis2_residualized_ranking": {
            "ols": {
                "slope_per_token": float(b1),
                "intercept": float(b0),
                "grand_mean_abs_len_diff": float(x.mean()),
                "adjustment": "adj = norm - slope*(abs_len_diff - grand_mean)",
            },
            "per_cell_median_raw": raw_med,
            "per_cell_median_adjusted": adj_med,
            "rank_stability_spearman": float(rank_stability),
            "raw_median_parity_vs_dv2_json_max_abs": float(med_parity),
            "noise_floor_verdicts": verdicts,
            "verdict_flips": flips,
        },
        "analysis3_coupling": {
            "across_cells_raw": raw_coup,
            "across_cells_committed": {k: committed[k] for k in ("obs", "ci95", "n_cells")},
            "across_cells_length_adjusted": adj_coup,
            "within_cell_median_rho_raw": float(np.median(list(wc_raw.values()))),
            "within_cell_median_rho_adjusted": float(np.median(list(wc_adj.values()))),
            "note": "identical bootstrap seed key [SEED_BOOT, 8] for raw + adjusted "
            "(paired resample indices)",
        },
    }


# ── figure (raw alongside residualized; per-type points labeled) ───────


def _label_points(ax: plt.Axes, xs, ys, names, fontsize: float = 5.0) -> None:
    for xi, yi, name in zip(xs, ys, names):
        ax.text(xi, yi, f" {name}", fontsize=fontsize, va="center", color="#444444")


def render_figure(frame: dict[str, np.ndarray], res: dict, figures_dir: Path) -> dict:
    """Four panels: per-pair raw scatter + OLS; per-type raw-vs-adjusted
    medians; coupling raw; coupling length-adjusted. Per-type labels on the
    cell-level panels."""
    set_paper_style("blog")
    x, y, cell = frame["abs_len_diff"], frame["norm_dva"], frame["cell"]
    a2 = res["analysis2_residualized_ranking"]
    a3 = res["analysis3_coupling"]
    cells = sorted(a2["per_cell_median_raw"])
    names = [c.replace("_", " ") for c in cells]
    med_raw = np.array([a2["per_cell_median_raw"][c] for c in cells])
    med_adj = np.array([a2["per_cell_median_adjusted"][c] for c in cells])
    dvc_med = per_cell_median(cell, frame["norm_dvc"])
    xc = np.array([dvc_med[c] for c in cells])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    ax = axes[0, 0]
    ax.scatter(x, y, s=8, color=DV2_COLOR, alpha=0.35, linewidths=0)
    xs = np.linspace(0, float(x.max()), 50)
    b1 = a2["ols"]["slope_per_token"]
    b0 = a2["ols"]["intercept"]
    ax.plot(xs, b0 + b1 * xs, color=REF_COLOR, lw=1.2)
    rho = res["analysis1_length_vs_shift"]["pooled_spearman"]
    ax.set_xlabel("|Δ mean completion length| (tokens)")
    ax.set_ylabel("answer-vector shift ‖Δv_A‖ (tail, L19)")
    ax.set_title(f"per-pair, raw (n={res['n_pairs']}; ρ={rho['obs']:.2f})", fontsize=9)

    ax = axes[0, 1]
    ax.scatter(med_raw, med_adj, s=14, color=DV2_COLOR, linewidths=0)
    lims = [min(med_raw.min(), med_adj.min()) * 0.95, max(med_raw.max(), med_adj.max()) * 1.05]
    ax.plot(lims, lims, color=REF_COLOR, lw=0.8, ls="--")
    _label_points(ax, med_raw, med_adj, names)
    ax.set_xlabel("per-type median ‖Δv_A‖, raw")
    ax.set_ylabel("per-type median ‖Δv_A‖, length-adjusted")
    ax.set_title(
        f"rank stability (Spearman={a2['rank_stability_spearman']:.3f})",
        fontsize=9,
    )

    for j, (med, key, lab) in enumerate(
        (
            (med_raw, "across_cells_raw", "raw"),
            (med_adj, "across_cells_length_adjusted", "length-adjusted"),
        )
    ):
        ax = axes[1, j]
        ax.scatter(xc, med, s=14, color=DV2_COLOR, linewidths=0)
        _label_points(ax, xc, med, names)
        r = a3[key]
        ax.set_xlabel("per-type median context shift ‖Δv_C‖ (ce, L19)")
        ax.set_ylabel(f"per-type median ‖Δv_A‖, {lab}")
        ax.set_title(
            f"coupling, {lab} (ρ={r['obs']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}])",
            fontsize=9,
        )

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "fu_length_covariate", dir=figures_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


# ── entrypoint ─────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--results-dir", type=Path, default=root / "eval_results/issue_2215")
    ap.add_argument("--figures-dir", type=Path, default=root / "figures/issue_2215")
    ap.add_argument("--staging-root", type=Path, default=Path("/tmp/issue2215_lenfu_stage"))
    ap.add_argument("--out", type=Path, default=None, help="output JSON (default: results-dir)")
    ap.add_argument("--boot-b", type=int, default=10000)
    ap.add_argument("--keep-staged", action="store_true", help="skip the post-reduce delete")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_path = args.out or args.results_dir / "length_covariate" / "length_covariate.json"

    bank_path, shard_paths = stage_inputs(args.staging_root)
    bank = json.loads(bank_path.read_text())
    ctx_len = reduce_lengths(shard_paths)
    if not args.keep_staged:
        shutil.rmtree(args.staging_root)
        print(f"[stage] deleted {args.staging_root}", flush=True)

    dv2_rows = read_jsonl(args.results_dir / "perpair" / "dv2_pairs.jsonl")
    dv1_rows = read_jsonl(args.results_dir / "perpair" / "dv1_pairs.jsonl")
    dv2_json = json.loads((args.results_dir / "dv2_answer_shift.json").read_text())
    coupling_json = json.loads((args.results_dir / "coupling.json").read_text())

    frame = build_pair_frame(dv2_rows, dv1_rows, bank["pairs"], ctx_len)
    res = analyze(frame, args.boot_b, dv2_json, coupling_json)
    fig_paths = render_figure(frame, res, args.figures_dir)

    res["repro"] = {
        **as_metadata_dict(git_provenance()),
        "revision_pin": REVISION_PIN,
        "anchor_text_prefix": ANCHOR_TEXT_PREFIX,
        "boot_b": args.boot_b,
        "figure_paths": fig_paths,
        "timestamp_unix": time.time(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(res, indent=1, sort_keys=True))
    tmp.replace(out_path)
    print(f"[done] wrote {out_path}", flush=True)
    a1 = res["analysis1_length_vs_shift"]
    a2 = res["analysis2_residualized_ranking"]
    a3 = res["analysis3_coupling"]
    print(
        f"[summary] pooled rho={a1['pooled_spearman']['obs']:.3f} "
        f"ci={a1['pooled_spearman']['ci95']} | per-cell median rho="
        f"{a1['per_cell_spearman_summary']['median']:.3f} | rank stability="
        f"{a2['rank_stability_spearman']:.4f} | flips={a2['verdict_flips']} | "
        f"coupling raw={a3['across_cells_raw']['obs']:.3f} -> adj="
        f"{a3['across_cells_length_adjusted']['obs']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
