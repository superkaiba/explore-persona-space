#!/usr/bin/env python
"""#1947 9a-ter free-analysis (VM, 0 GPU): PREFIX-arm H3/H4 reads.

The P6 battery (`issue1947_analysis.py`) computes its context-dependent reads
with the ``last_prompt`` pooling PRIMARY (+ span-mean ``context`` SECONDARY)
but NO prefix-based arm — the standing "prefix mapping AND context mapping"
rule wants both. The capture recorded ``prefix_last`` (final prefix token)
alongside ``last_prompt``, so this round re-reduces the EXISTING per-row
stores: per (arm x layer x available tree) it computes

- H4 gate read (`issue1768_directions.gate_read`, Spearman rho) TWICE — the
  ``prefix_last`` pooling substituted as the context summary, beside a
  recomputed ``last_prompt`` read (matched target: same rows, same sigma);
- H3 alignment (cos(w, delta) + row-cluster bootstrap CI + corpus-covariance /
  isotropic null bands + the shuffled-row band on the onpolicy tree). NOTE:
  H3 consumes NO context summary (w = mean per-row answer shift; delta = mix
  pool minus panel base half) so it is POOLING-INVARIANT by construction —
  recorded once per cell with ``pooling_invariant: true`` and shown as a
  single distribution in the figure.

Reuses the battery's own helpers (``issue1947_analysis`` loaders + the
``issue1768_directions`` read bodies) — no re-implemented math. ANALYSIS-ONLY:
no training, no generation, no API calls.

Outputs: eval_results/issue_1947/analysis/prefix_reads.json (+ per-unit
resume JSONL prefix_reads_units.jsonl) and
figures/issue_1947/prefix_vs_lastprompt_reads.png.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1768_directions as DIRS  # noqa: E402
import issue1947_analysis as A  # noqa: E402
import issue1947_cells as cells  # noqa: E402

logger = logging.getLogger("issue1947.prefix_reads")

LAYERS = (14, 19, 25)
POOL_PREFIX = "prefix_last"  # the prefix arm (final token of the pre-query prefix)
POOL_PRIMARY = A.CTX_PRIMARY  # "last_prompt" (final rendered-prompt token)
SEED = A.BOOT_SEED  # 653 — the battery's own bootstrap seed (comparability)


def _meta() -> dict:
    return {"issue": cells.ISSUE, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _degenerate_gate(g: dict) -> bool:
    """Constant-input degeneracy for a gate read: the context summary is
    bitwise-constant across rows (row_spread_max_abs == 0 — onpolicy cells
    hold ONE context, so the prefix summary cannot vary) or rho is undefined.
    Keyed on the INPUT spread, not on NaN rho: BLAS float non-associativity
    over identical rows yields ~1e-16 g_pred jitter, and spearmanr then
    returns a finite jitter-level rho instead of NaN (measured: 2 unique
    g_pred values at std 1.2e-16 -> rho -0.014)."""
    rho = g.get("spearman_rho")
    spread = g.get("row_spread_max_abs")
    rho_bad = rho is None or (isinstance(rho, float) and not np.isfinite(rho))
    return bool(rho_bad or spread == 0.0)


def _san(obj):
    """NaN/inf -> None recursively (strict-JSON files; a NaN rho is recorded as
    null + an explicit degenerate flag, never a bare NaN literal)."""
    if isinstance(obj, dict):
        return {k: _san(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_san(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _regime(cfg) -> dict:
    """Resume-predicate regime key: every output-affecting arg (#722 r3)."""
    return {
        "version": 2,  # v2: degenerate-constant-prefix diagnostics + NaN sanitize
        "out_root": str(Path(cfg.out_root).resolve()),
        "layers": list(cfg.layers),
        "poolings": [POOL_PREFIX, POOL_PRIMARY],
        "n_boot": A.N_BOOT,
        "seed": cfg.seed,
    }


def _join_sha_orders(a: dict, b: dict) -> tuple[list[str], list[str]]:
    """(a-order keep shas, b-order keep shas) for the sha join of stores a/b."""
    sb, sa = set(b["row_sha"]), set(a["row_sha"])
    return (
        [s for s in a["row_sha"] if s in sb],
        [s for s in b["row_sha"] if s in sa],
    )


def _ctx_rows_stack_aligned(t: dict, span: str, layer: int) -> tuple[np.ndarray, bool]:
    """Base-store ``span`` rows joined + REALIGNED to the stack's (a-store) row
    order, so gate_read pairs g_pred(row i) with g_hat(row i). Returns
    (rows, was_already_aligned)."""
    a, b = t["store_a"], t["store_b"]
    C, _, _ = A._join(b, a, span, layer)  # b-order keep (the battery's own join)
    a_order, b_order = _join_sha_orders(a, b)
    if a_order == b_order:
        return C, True
    pos = {s: i for i, s in enumerate(b_order)}
    return C[[pos[s] for s in a_order]], False


def unit_record(cfg, acfg, slug: str, layer: int, sigma: dict) -> dict:
    """One (arm x layer) prefix-reads cell — mirrors the battery cell schema
    for the H3 alignment + H4 gate reads only."""
    cell = cells.CELL_BY_SLUG[slug]
    rng = np.random.default_rng([cfg.seed, zlib.crc32(slug.encode("utf-8")), layer, 1947])
    rec: dict = {"slug": slug, "layer": layer, "kind": cell.kind, "lr": cell.lr}
    legs = None
    if cell.kind == "content":
        legs = A._delta_legs(acfg, slug, layer)
        rec["delta_split_half_r_disjoint"] = legs.get("split_half_r_disjoint")
        rec["delta_n_rows"] = legs["n_rows"]
    h3_nulls = None  # nulls depend on (delta, sigma) only — computed once per cell
    for tree in A.TREES:
        try:
            t = A._stack_for_tree(acfg, slug, tree, layer)
        except FileNotFoundError as e:
            rec[tree] = {"missing": str(e)}
            continue
        stack = t["stack"]
        w = stack.mean(axis=0)
        tr: dict = {"n_rows": int(stack.shape[0])}
        if legs is not None:
            cand = np.asarray(legs["delta"], dtype=np.float64)
            ci, boot_mean = A._boot_cos_ci(stack, cand, A.N_BOOT, cfg.seed + layer)
            if h3_nulls is None:
                h3_nulls = DIRS.null_bands(cand, sigma, rng)
            h3 = {
                "cos_w_delta": A._cos(w, cand),
                "boot_ci95": ci,
                "boot_mean": boot_mean,
                "null_bands": h3_nulls,
                # H3 consumes no context summary — identical under either pooling
                "pooling_invariant": True,
            }
            if tree == "onpolicy":
                v0_rows = A._rows(t["store_b"], "response", layer)
                assert v0_rows.shape == stack.shape, (slug, layer, v0_rows.shape, stack.shape)
                h3["shuffled_row_band"] = DIRS.shuffled_row_band(
                    v0_rows + stack, v0_rows, cand, rng
                )
            tr["h3"] = h3
        else:
            tr["h3"] = None  # marker arm: no delta pool (mirrors the battery)
        gate: dict = {}
        for pool in (POOL_PREFIX, POOL_PRIMARY):
            C, aligned = _ctx_rows_stack_aligned(t, pool, layer)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # constant-input rho handled below
                g = DIRS.gate_read(C, stack, C.mean(axis=0), w, sigma)
            g["pooling"] = pool
            g["rows_realigned_to_stack_order"] = not aligned
            # structural degeneracy: onpolicy rows share ONE context per cell,
            # so the prefix summary is constant across rows -> rho undefined
            # (or float-jitter noise). Named non-fatal class (companion-stat
            # drop-class semantics).
            g["row_spread_max_abs"] = float(np.max(np.abs(C - C[0:1])))
            g["degenerate_constant_input"] = _degenerate_gate(g)
            gate[pool] = g
        tr["h4_gate"] = gate
        rec[tree] = tr
    return rec


def make_figure(fig_path: Path, recs: list[dict], layers) -> None:
    """2x3 grid: row 0 = H4 gate rho paired slopegraph (prefix_last vs
    last_prompt, one line per arm x tree); row 1 = H3 cos(w, delta)
    distribution (pooling-invariant — one distribution, both labels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    pal = paper_palette(4)
    tree_color = {"matched_text": pal[0], "onpolicy": pal[1]}
    fig, axes = plt.subplots(2, len(layers), figsize=(4.2 * len(layers), 7.5), squeeze=False)
    for col, layer in enumerate(layers):
        rows = [r for r in recs if r["layer"] == layer]
        ax = axes[0][col]
        for tree in A.TREES:
            raw_pairs = [
                (r[tree]["h4_gate"][POOL_PREFIX], r[tree]["h4_gate"][POOL_PRIMARY])
                for r in rows
                if isinstance(r.get(tree), dict) and "h4_gate" in r[tree]
            ]
            pairs = [
                (gp["spearman_rho"], gl["spearman_rho"])
                for gp, gl in raw_pairs
                if not gp["degenerate_constant_input"]
                and not gl["degenerate_constant_input"]
                and all(
                    isinstance(v, float) and np.isfinite(v)
                    for v in (gp["spearman_rho"], gl["spearman_rho"])
                )
            ]
            n_degen = len(raw_pairs) - len(pairs)
            if raw_pairs and not pairs:  # whole tree degenerate: label, never silent
                ax.plot(
                    [],
                    [],
                    color=tree_color[tree],
                    lw=1.5,
                    label=f"{tree}: prefix gate degenerate (all {n_degen} cells; "
                    "constant prefix within cell)",
                )
            for px, py in pairs:
                ax.plot([0, 1], [px, py], color="0.75", lw=0.6, zorder=1)
            if pairs:
                label = f"{tree} (n={len(pairs)}"
                label += f"; {n_degen} degenerate prefix dropped)" if n_degen else ")"
                ax.scatter(
                    [0] * len(pairs),
                    [p[0] for p in pairs],
                    s=14,
                    color=tree_color[tree],
                    zorder=2,
                    label=label,
                )
                ax.scatter(
                    [1] * len(pairs),
                    [p[1] for p in pairs],
                    s=14,
                    color=tree_color[tree],
                    zorder=2,
                )
        for band in A.GATE_BAND:
            ax.axhline(band, color="0.3", lw=0.8, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([POOL_PREFIX, POOL_PRIMARY])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(f"H4 gate Spearman rho (L{layer})")
        if col == 0:
            ax.set_ylabel("Spearman rho (paired per arm x tree)")
            ax.legend(fontsize=6, loc="best")
        ax = axes[1][col]
        for j, tree in enumerate(A.TREES):
            vals = [
                r[tree]["h3"]["cos_w_delta"]
                for r in rows
                if isinstance(r.get(tree), dict) and r[tree].get("h3")
            ]
            if not vals:
                continue
            x = np.full(len(vals), j) + np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=14, color=tree_color[tree])
            ax.boxplot(
                [vals],
                positions=[j],
                widths=0.5,
                showfliers=False,
                medianprops={"color": "0.2"},
            )
        ax.set_xticks(range(len(A.TREES)))
        ax.set_xticklabels(list(A.TREES))
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set_title(f"H3 cos(w, delta) (L{layer})\npooling-invariant: prefix == last_prompt")
        if col == 0:
            ax.set_ylabel("cos(w, delta) — content arms")
    fig.suptitle(
        "#1947 prefix-arm reads: prefix_last vs last_prompt (same arms, same rows, same sigma)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="#1947 prefix-arm H3/H4 reads (free analysis)")
    p.add_argument("--out-root", default=str(REPO_ROOT / "data/issue_1947/battery_stage"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results/issue_1947/analysis"))
    p.add_argument("--fig-dir", default=str(REPO_ROOT / "figures/issue_1947"))
    p.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    p.add_argument("--arms", default="", help="comma-separated slug filter")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--import-check", action="store_true")
    args = p.parse_args(argv)
    if args.import_check:
        names = [
            A._stack_for_tree,
            A._delta_legs,
            A._boot_cos_ci,
            A._join,
            A._rows,
            A._cos,
            DIRS.gate_read,
            DIRS.null_bands,
            DIRS.shuffled_row_band,
            DIRS.corpus_sigma,
        ]
        print(f"[import-check] OK ({len(names)} symbols resolved)")
        return 0
    args.layers = tuple(int(x) for x in args.layers.split(",") if x)
    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    acfg = A.Cfg(out_root=Path(args.out_root), out_dir=out_dir, fig_dir=fig_dir, layers=args.layers)
    # memoize store loads (same 4 stores re-read across a slug's 3 layers)
    A._load_store = functools.lru_cache(maxsize=8)(A._load_store)

    man = A._verdict_arms(acfg)
    slugs = sorted(man["content"]) + sorted(man["marker"])
    if args.arms:
        keep = {s for s in args.arms.split(",") if s}
        slugs = [s for s in slugs if s in keep]
    regime = _regime(args)
    units_path = out_dir / "prefix_reads_units.jsonl"
    done: dict[tuple[str, int], dict] = {}
    if units_path.exists():
        with units_path.open(encoding="utf-8") as fh:  # never splitlines() on JSONL
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("regime") == regime:
                    rec = row["record"]
                    for tree in A.TREES:  # re-derive the flag (pure view of
                        tr = rec.get(tree)  # persisted measured fields)
                        if isinstance(tr, dict):
                            for g in (tr.get("h4_gate") or {}).values():
                                g["degenerate_constant_input"] = _degenerate_gate(g)
                    done[(rec["slug"], rec["layer"])] = rec
    sigma_by_layer = {li: DIRS.corpus_sigma(Path(args.out_root), li) for li in args.layers}

    recs: list[dict] = []
    total = len(slugs) * len(args.layers)
    k = 0
    for slug in slugs:
        for layer in args.layers:
            k += 1
            t0 = time.time()
            if (slug, layer) in done:
                recs.append(done[(slug, layer)])
                print(f"[prefix-reads] unit {k}/{total} {slug}_L{layer} resumed", flush=True)
                continue
            rec = _san(unit_record(args, acfg, slug, layer, sigma_by_layer[layer]))
            recs.append(rec)
            out_dir.mkdir(parents=True, exist_ok=True)
            with units_path.open("a", encoding="utf-8") as fh:  # atomic single-line append
                fh.write(json.dumps({"regime": regime, "record": rec, **_meta()}) + "\n")
                fh.flush()
            print(
                f"[prefix-reads] unit {k}/{total} {slug}_L{layer} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    summary = {
        "n_arms": len(slugs),
        "layers": list(args.layers),
        "poolings": {"prefix_arm": POOL_PREFIX, "context_arm_recomputed": POOL_PRIMARY},
        "h4_note": (
            "onpolicy cells hold ONE context, so the prefix summary is bitwise-constant "
            "across rows and the prefix gate read is structurally DEGENERATE there "
            "(degenerate_constant_input; keyed on row_spread_max_abs == 0 — a finite rho "
            "over a constant input is BLAS float-jitter, not signal). The substantive "
            "prefix-vs-context H4 contrast is the matched_text (mix-rows) tree, whose "
            "rows span source + negative contexts."
        ),
        "h3_note": (
            "H3 (cos(w, delta) + nulls) consumes no context summary and is identical "
            "under either pooling; recorded once per cell with pooling_invariant: true. "
            "The prefix-vs-context contrast lives in the H4 gate read."
        ),
        "regime": regime,
        "cells": recs,
        **_meta(),
    }
    A._atomic_json(out_dir / "prefix_reads.json", _san(summary))
    if not args.no_figure:
        make_figure(fig_dir / "prefix_vs_lastprompt_reads.png", recs, args.layers)
    print(f"[prefix-reads] done: {len(recs)} cells", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit — the PyGILState_Release atexit gotcha
