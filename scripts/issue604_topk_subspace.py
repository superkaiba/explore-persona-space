# ruff: noqa: RUF002
# Intentional Unicode (Δ, ※, —) in scientific docstrings + labels.
"""Task #604 free-analysis follow-up — top-k key-SUBSPACE read (VM CPU).

The parent result killed the top-1 key identity (key_match.json: the top
right-singular vector of the stacked q/k/v attention update does not match
the source context vector beyond the shuffled-pairing null). The surviving
alternative this read attacks: the source context vector might live in the
top-k right-singular SUBSPACE even when it is not the top-1 direction.

Per cell × band layer (L14–L24) × k ∈ {1, 2, 4, 8}: the projection energy
``‖V_kᵀ û‖²`` of the unit-normalized source context centroid û (module-input
"attn" space, the parent's primary comparison space) onto the top-k columns
of the stored ``L{i}__attn_key__V8`` orthonormal basis, calibrated by

- (a) the same statistic for every wrong-context bank vector — EXCLUDING the
  source's exact prompt-SHA duplicates (6 of the 42 bank entries duplicate
  another entry; the parent disclosed that the duplicates mechanically tie
  the wrong-context null for the affected i474/i518 sources), and
- (b) a shuffled-pairing null (subspace of one cell × source of another,
  within the same aggregate group), with SHA-based source disjointness so
  byte-identical prompts under different labels (B1 / C1 / qwen_default)
  never pair as "shuffled".

Normalization is EXPLICIT (recorded in the JSON meta): energy is computed in
the full 3584-d residual space against the UNIT context vector, so the
random-vector floor is k/3584. It is NOT renormalized within the rank-R
stacked update row space (that variant's floor would be k/R with R the
stacked rank, 48 for the r=16 attention-only dial / 24 for the r=8 saturated
endpoint / 96 for the r=32 all-linear lines).

Line coverage mirrors the brief: dial527 (clean vs panel-contaminated split,
as the parent's registered rotation read does), dial538, dial550, i474 split
pos/loc, i519, i518, and i521 as a no-source bank-wide scale reference.
i541 is excluded (own-bank line; its sources are not in the 42-context
Phase B bundle — same exclusion as the parent key-match read).

Usage:
    uv run python scripts/issue604_topk_subspace.py   # full 209-cell run (CPU-minutes)

Outputs: ``eval_results/issue_604/topk_subspace.json`` +
``figures/issue_604/topk_subspace.{png,pdf,meta.json}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue604_analyze import CellStore, ContextBundle, _unit  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue_604 import (  # noqa: E402
    HIDDEN_SIZE,
    KEY_LAYER_BAND,
    result_metadata,
)

logger = logging.getLogger("issue604.topk_subspace")

OUT_DIR_DEFAULT = PROJECT_ROOT / "eval_results/issue_604"
FIG_DIR_DEFAULT = PROJECT_ROOT / "figures/issue_604"
K_VALUES = (1, 2, 4, 8)
EXCLUDED_LINES = ("i541",)  # own-bank line; sources not in the Phase B bundle

# Reader-facing panel titles: plain-English line names only (no issue IDs in
# rendered text — same convention as the 8 parent figures after round-2 review).
GROUP_LABELS = {
    "dial527": "dose dial, shallow window",
    "dial538": "dose dial, deep window",
    "dial550": "dose dial, mid window",
    "i474_pos": "epoch ladder, positives-only",
    "i474_loc": "epoch ladder, contrastive",
    "i519": "saturated endpoint",
    "i518": "cross-behavior",
    "i521": "EM no-source control (scale ref)",
}


def _group_key(cell: dict) -> str:
    """Aggregate-group key: line, with i474 split by arm and #527 pair-2 split out."""
    c = cell["cell"]
    if c["line"] == "i474":
        return f"i474_{c['arm']}"
    if c["line"] == "dial527" and "panel-contaminated" in c["tags"]:
        return "dial527_panel_contaminated"
    return c["line"]


def _sha_groups(bundle: ContextBundle) -> tuple[dict[str, str], dict[str, set[str]]]:
    """{context: prompt_sha256} + {context: set of OTHER contexts with the same sha}.

    Asserts the bundle manifest covers exactly the loaded context names —
    the duplicate-exclusion logic is only valid against the same bank.
    """
    contexts = bundle.manifest.get("contexts")
    assert isinstance(contexts, dict) and set(contexts) == set(bundle.names), (
        "Phase B manifest context set does not match the loaded bundle — "
        "cannot build the duplicate-SHA exclusion map"
    )
    sha_of = {name: contexts[name]["prompt_sha256"] for name in bundle.names}
    by_sha: dict[str, list[str]] = defaultdict(list)
    for name, sha in sha_of.items():
        by_sha[sha].append(name)
    dups = {name: set(by_sha[sha]) - {name} for name, sha in sha_of.items()}
    n_dup_entries = sum(len(v) - 1 for v in by_sha.values() if len(v) > 1)
    logger.info(
        "duplicate map: %d sha groups with >1 entry (%d duplicate entries)",
        sum(1 for v in by_sha.values() if len(v) > 1),
        n_dup_entries,
    )
    return sha_of, dups


def _band_energy_matrix(
    store: CellStore, bundle: ContextBundle, cell: dict, band: list[int], bank_unit: dict
) -> tuple[np.ndarray, list[int]] | None:
    """Band-mean top-k projection-energy matrix for one cell.

    Returns ``(E, layers_used)`` with ``E`` of shape ``(len(K_VALUES),
    n_contexts)``: ``E[ki, ctx] = mean over band layers of ‖V_kᵀ û_ctx‖²``,
    plus the per-layer stack ``cell["_E_layers"]`` cached on the cell dict
    (``(n_layers_used, len(K_VALUES), n_contexts)``) for the per-layer records.
    """
    arrs = store.vectors(cell)
    per_layer = []
    layers_used = []
    for layer in band:
        v8 = arrs.get(f"L{layer}__attn_key__V8")
        if v8 is None:
            continue
        v8 = v8.astype(np.float64)
        assert v8.shape == (HIDDEN_SIZE, 8), v8.shape
        # The top-k subspace read is only valid over an orthonormal basis.
        # V8 is stored fp16 (measured max Gram deviation ~3e-5), so the
        # tolerance is fp16-rounding-scale; the induced energy error is
        # O(1e-5), an order of magnitude below the k=1 random floor (2.8e-4).
        gram = v8.T @ v8
        assert np.allclose(gram, np.eye(8), atol=1e-3), (
            cell["cell"]["cell_id"],
            layer,
            "V8 columns are not orthonormal",
        )
        proj_sq = (v8.T @ bank_unit[layer]) ** 2  # (8, n_ctx)
        cum = np.cumsum(proj_sq, axis=0)  # cum[k-1] = top-k energy
        per_layer.append(cum[[k - 1 for k in K_VALUES], :])  # (len(K), n_ctx)
        layers_used.append(layer)
    if not per_layer:
        return None
    stack = np.stack(per_layer)  # (n_layers, len(K), n_ctx)
    cell["_E_layers"] = stack
    return stack.mean(axis=0), layers_used


def _pct(vals, q: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _bank_read(E: np.ndarray, names: list[str]) -> dict:
    """Whole-bank energy distribution per k for a no-source cell (i521 scale ref)."""
    out = {}
    for ki, k in enumerate(K_VALUES):
        vals = E[ki, :]
        out[str(k)] = {
            "bank_p50": _pct(vals, 50),
            "bank_p95": _pct(vals, 95),
            "bank_max": float(vals.max()),
            "bank_argmax_context": names[int(vals.argmax())],
            "n_bank": int(vals.size),
        }
    return out


def _matched_source_rec(
    cell: dict, src: str, name_idx: dict[str, int], names: list[str], dups: dict[str, set[str]]
) -> dict:
    """Matched-vs-wrong-context energies for one (cell, source) row.

    The wrong-context null excludes the source AND its exact prompt-SHA
    duplicates (the parent's disclosed tie defect on the i474/i518 sources).
    """
    E = cell["_E_band"]
    si = name_idx[src]
    excluded = {src} | dups[src]
    wrong_idx = [name_idx[n] for n in names if n not in excluded]
    rec: dict = {
        "source": src,
        "sha_duplicates_excluded_from_null": sorted(dups[src]),
        "k": {},
    }
    for ki, k in enumerate(K_VALUES):
        matched = float(E[ki, si])
        wrong = E[ki, wrong_idx]
        wrong_p95 = _pct(wrong, 95)
        rec["k"][str(k)] = {
            "matched_band_mean": matched,
            "matched_per_layer": [float(v) for v in cell["_E_layers"][:, ki, si]],
            "wrong_p50": _pct(wrong, 50),
            "wrong_p95": wrong_p95,
            "n_wrong": int(wrong.size),
            "above_wrong_p95": bool(matched > wrong_p95),
        }
    return rec


def _shuffled_nulls(
    group_cells: dict[str, list[dict]],
    bundle: ContextBundle,
    sha_of: dict[str, str],
    name_idx: dict[str, int],
) -> dict[str, dict | str]:
    """Shuffled-pairing null per aggregate group.

    Band-mean energy of cell i's subspace at the bank column of cell j's
    source, j ≠ i, with SHA-disjoint source sets (name disjointness is
    implied by sha disjointness; covers the B1 / C1 / qwen_default triple).
    """
    shuffled: dict[str, dict | str] = {}
    for group, cells in group_cells.items():
        pools: dict[int, list[float]] = {k: [] for k in K_VALUES}
        for ci in cells:
            shas_i = {sha_of[bundle.resolve(s)] for s in ci["cell"]["source_personas"]}
            for cj in cells:
                if cj is ci or not cj["cell"]["source_personas"]:
                    continue
                shas_j = {sha_of[bundle.resolve(s)] for s in cj["cell"]["source_personas"]}
                if shas_i & shas_j:
                    continue
                for s in cj["cell"]["source_personas"]:
                    col = name_idx[bundle.resolve(s)]
                    for ki, k in enumerate(K_VALUES):
                        pools[k].append(float(ci["_E_band"][ki, col]))
        if pools[K_VALUES[0]]:
            shuffled[group] = {
                str(k): {"n": len(pools[k]), "p50": _pct(pools[k], 50), "p95": _pct(pools[k], 95)}
                for k in K_VALUES
            }
        else:
            shuffled[group] = "N/A — no SHA-disjoint cell pairs in group"
    return shuffled


def _aggregate_group(
    rows: list[dict], cells: list[dict], shuffled_group: dict | str | None
) -> dict:
    """Per-group per-k aggregate (matched vs wrong-context vs shuffled null)."""
    agg: dict = {"n_matched_rows": len(rows), "k": {}}
    for ki, k in enumerate(K_VALUES):
        kk = str(k)
        if rows:
            m = [r["rec"]["k"][kk]["matched_band_mean"] for r in rows]
            w50 = [r["rec"]["k"][kk]["wrong_p50"] for r in rows]
            w95 = [r["rec"]["k"][kk]["wrong_p95"] for r in rows]
            above_own = sum(int(r["rec"]["k"][kk]["above_wrong_p95"]) for r in rows)
            shuf_k = shuffled_group[kk] if isinstance(shuffled_group, dict) else None
            agg["k"][kk] = {
                "matched_mean": float(np.mean(m)),
                "matched_p50": _pct(m, 50),
                "matched_max": float(np.max(m)),
                "wrong_p50_median_over_rows": _pct(w50, 50),
                "wrong_p95_median_over_rows": _pct(w95, 50),
                "frac_rows_above_own_wrong_p95": above_own / len(rows),
                "shuffled_p50": shuf_k["p50"] if shuf_k else None,
                "shuffled_p95": shuf_k["p95"] if shuf_k else None,
                "frac_rows_above_shuffled_p95": (
                    float(np.mean([v > shuf_k["p95"] for v in m])) if shuf_k else None
                ),
                "random_floor": k / HIDDEN_SIZE,
            }
        elif cells:  # i521 scale ref — no matched rows
            pooled = np.concatenate([c["_E_band"][ki, :] for c in cells])
            agg["k"][kk] = {
                "bank_p50": _pct(pooled, 50),
                "bank_p95": _pct(pooled, 95),
                "bank_max": float(pooled.max()),
                "n_bank_reads": int(pooled.size),
                "random_floor": k / HIDDEN_SIZE,
            }
    return agg


def run_topk(store: CellStore, bundle: ContextBundle, out: Path) -> dict:
    """Compute the registered top-k subspace read and write ``topk_subspace.json``."""
    band = [li for li in KEY_LAYER_BAND if li < bundle.n_layers]
    names = bundle.names
    name_idx = {n: i for i, n in enumerate(names)}
    sha_of, dups = _sha_groups(bundle)
    bank_unit = {
        layer: np.stack([_unit(bundle.vec(n, layer, "attn")) for n in names], axis=1)
        for layer in band
    }  # (3584, n_ctx) per layer

    cells_out: list[dict] = []
    matched_rows: dict[str, list[dict]] = defaultdict(list)  # group -> per-source rows
    group_cells: dict[str, list[dict]] = defaultdict(list)
    stack_rank_by_line: dict[str, set[int]] = defaultdict(set)

    for cell in store.cells:
        c = cell["cell"]
        if c["line"] in EXCLUDED_LINES:
            continue
        em = _band_energy_matrix(store, bundle, cell, band, bank_unit)
        if em is None:
            logger.warning("N/A — no attn_key V8 in band for %s; cell skipped", c["cell_id"])
            continue
        E, layers_used = em
        cell["_E_band"] = E
        group = _group_key(cell)
        group_cells[group].append(cell)
        s = store.vectors(cell).get(f"L{band[0]}__attn_key__S")
        if s is not None:
            stack_rank_by_line[c["line"]].add(int(s.size))

        rec: dict = {
            "line": c["line"],
            "group": group,
            "cell_id": c["cell_id"],
            "arm": c.get("arm"),
            "seed": c.get("seed"),
            "epoch": c.get("epoch"),
            "tags": c["tags"],
            "layers_used": layers_used,
        }
        if not c["source_personas"]:  # i521 — no-source scale reference
            rec["bank_read"] = _bank_read(E, names)
            cells_out.append(rec)
            continue

        per_source = []
        for raw_src in c["source_personas"]:
            src = bundle.resolve(raw_src)
            src_rec = _matched_source_rec(cell, src, name_idx, names, dups)
            per_source.append(src_rec)
            matched_rows[group].append(
                {"cell_id": c["cell_id"], "src_sha": sha_of[src], "rec": src_rec}
            )
        rec["per_source"] = per_source
        cells_out.append(rec)

    shuffled = _shuffled_nulls(group_cells, bundle, sha_of, name_idx)
    per_group = {
        group: _aggregate_group(
            matched_rows.get(group, []), group_cells.get(group, []), shuffled.get(group)
        )
        for group in sorted(set(group_cells) | set(matched_rows))
    }

    payload = {
        "meta": result_metadata(
            PROJECT_ROOT,
            extra={
                "analysis": "topk_subspace",
                "k_values": list(K_VALUES),
                "comparison_space": "attn (true module-input centroids, parent primary space)",
                "energy_normalization": (
                    "energy = ||V_k^T u_hat||^2 with u_hat the UNIT-normalized context "
                    "centroid in the full 3584-d residual space; V_k = first k columns of "
                    "the stored orthonormal attn_key V8 basis. Random-unit-vector floor = "
                    "k/3584. NOT renormalized within the rank-R stacked update row space "
                    "(that variant's floor would be k/R; per-line stacked ranks recorded "
                    "in stacked_rank_by_line)."
                ),
                "random_floor_by_k": {str(k): k / HIDDEN_SIZE for k in K_VALUES},
                "stacked_rank_by_line": {
                    line: sorted(ranks) for line, ranks in sorted(stack_rank_by_line.items())
                },
                "duplicate_handling": (
                    "wrong-context nulls exclude the matched source AND every bank entry "
                    "sharing its prompt sha256 (6 duplicate entries across 5 sha groups); "
                    "shuffled-pairing nulls require SHA-disjoint source sets between the "
                    "subspace cell and the source cell (covers B1/C1/qwen_default)"
                ),
                "excluded_lines": list(EXCLUDED_LINES),
            },
        ),
        "layer_band": band,
        "per_line": per_group,
        "shuffled_pairing_null": shuffled,
        "cells": cells_out,
    }
    out.write_text(json.dumps(payload, indent=1))
    logger.info("topk_subspace.json written (%d cells, %d groups)", len(cells_out), len(per_group))
    return payload


# ── figure ──────────────────────────────────────────────────────────────────

# Shallow → mid → deep dial ordering (the parent figure convention).
FIG_PANEL_ORDER = (
    "dial527",
    "dial550",
    "dial538",
    "i474_pos",
    "i474_loc",
    "i519",
    "i518",
    "i521",
)


def render_figure(payload: dict, fig_dir: Path) -> None:
    """Projection energy vs k: matched source vs calibrated nulls, one panel per line."""
    per_group = payload["per_line"]
    ks = [int(k) for k in payload["meta"]["k_values"]]
    panels = [g for g in FIG_PANEL_ORDER if g in per_group]
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.7 * ncols, 3.3 * nrows), squeeze=False, sharex=True
    )
    c_match = paper_palette_role("primary")
    c_wrong = paper_palette_role("neutral")
    c_shuf = paper_palette_role("accent")
    for ax, group in zip(axes.flat, panels, strict=False):
        agg = per_group[group]["k"]
        floor = [agg[str(k)]["random_floor"] for k in ks]
        if "matched_p50" in agg[str(ks[0])]:
            matched = [agg[str(k)]["matched_p50"] for k in ks]
            matched_max = [agg[str(k)]["matched_max"] for k in ks]
            w50 = [agg[str(k)]["wrong_p50_median_over_rows"] for k in ks]
            w95 = [agg[str(k)]["wrong_p95_median_over_rows"] for k in ks]
            ax.fill_between(
                ks, w50, w95, color=c_wrong, alpha=0.30, lw=0, label="wrong-context null (p50-p95)"
            )
            s50 = [agg[str(k)]["shuffled_p50"] for k in ks]
            s95 = [agg[str(k)]["shuffled_p95"] for k in ks]
            if all(v is not None for v in s95):
                ax.plot(ks, s95, color=c_shuf, ls="--", lw=1.3, label="shuffled-pairing null p95")
                ax.plot(ks, s50, color=c_shuf, ls=":", lw=1.1, label="shuffled-pairing null p50")
            ax.plot(ks, matched, color=c_match, marker="o", lw=1.8, label="matched source (median)")
            ax.plot(
                ks,
                matched_max,
                color=c_match,
                ls="--",
                lw=1.0,
                alpha=0.6,
                label="matched source (max cell)",
            )
        else:  # i521 scale ref: bank-wide distribution, no matched source
            b50 = [agg[str(k)]["bank_p50"] for k in ks]
            b95 = [agg[str(k)]["bank_p95"] for k in ks]
            bmax = [agg[str(k)]["bank_max"] for k in ks]
            ax.fill_between(
                ks, b50, b95, color=c_wrong, alpha=0.30, lw=0, label="whole-bank read (p50-p95)"
            )
            ax.plot(ks, bmax, color=c_wrong, ls="--", lw=1.0, label="whole-bank max")
        ax.plot(ks, floor, color="0.4", ls=(0, (1, 2)), lw=1.2, label="random-vector floor k/3584")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_title(GROUP_LABELS.get(group, group), fontsize=10)
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("key-subspace size k")
    for row in axes:
        row[0].set_ylabel("context-vector energy captured")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    h2, l2 = axes.flat[len(panels) - 1].get_legend_handles_labels()
    for h, lbl in zip(h2, l2, strict=True):
        if lbl not in labels:
            handles.append(h)
            labels.append(lbl)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Does the source context vector live in the top-k key subspace?\n"
        "projection energy onto the top-k right-singular subspace of the stacked q/k/v "
        "update, band mean L14-L24, module-input space",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    savefig_paper(fig, "topk_subspace", dir=fig_dir)
    plt.close(fig)


# ── entrypoint ──────────────────────────────────────────────────────────────


def main() -> None:
    """Single entrypoint — the full 209-cell run is the smoke (CPU-minutes)."""
    parser = argparse.ArgumentParser(
        description="Task 604 follow-up: top-k key-subspace projection-energy read."
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    parser.add_argument(
        "--context-dir",
        default=str(OUT_DIR_DEFAULT / "context_vectors_prod"),
        help="Phase B bundle dir (default: the production 42-context bundle)",
    )
    parser.add_argument("--fig-dir", default=str(FIG_DIR_DEFAULT))
    parser.add_argument(
        "--expect-probes",
        type=int,
        default=50,
        help="required n_probes in the Phase B bundle meta (stale-cache guard)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print("[phase=topk_load]", flush=True)
    out_dir = Path(args.out_dir)
    store = CellStore(out_dir)
    assert store.cells, "no Phase A outputs found — run issue604_adapter_svd.py first"
    required: set[str] = set()
    for cell in store.cells:
        if cell["cell"]["line"] in EXCLUDED_LINES:
            continue
        required.update(cell["cell"]["source_personas"])
    bundle = ContextBundle(
        Path(args.context_dir),
        expected_n_probes=args.expect_probes,
        required_contexts=tuple(sorted(required)),
    )
    assert bundle.hidden == HIDDEN_SIZE, bundle.hidden

    print("[phase=topk_energy]", flush=True)
    payload = run_topk(store, bundle, out_dir / "topk_subspace.json")
    print("[phase=topk_figure]", flush=True)
    set_paper_style("blog")
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    render_figure(payload, fig_dir)
    print("[phase=done]", flush=True)


if __name__ == "__main__":
    main()
