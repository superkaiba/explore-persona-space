#!/usr/bin/env python3
"""Issue #1336 — the source->target transfer figure WITH round-5's new pairs.

The committed `metric_ladder_source_target_both_tiers` figure carries its own
gap in its caption:

    Only 7 of the 20 ordered stage pairs exist: SFT->RLVR, SFT->longer RLVR
    and RLVR->longer RLVR were never run and nothing transfers INTO base, so
    the empty base column and the short lines are missing data, not zeros.

Round 5's self/cross-map missing-pairs grid
(`eval_results/issue_1336/selfmap_missing_pairs/cells/*.json`) ran EXACTLY
those three pairs plus the `base__base` self map. This script folds them into
that figure so the previously-empty positions carry measured points.

REUSE, NOT REIMPLEMENTATION: it imports
`scripts/issue1336_metric_ladder_plots.py` and calls that module's OWN
`make_source_target_overlay`, extending only its data inputs (`ROWS`,
`PAIR_ORDER`) in this process. No plotting logic is duplicated, and the
shared plotter file is left untouched (a concurrent session runs it).
Output goes to a DISTINCT outname so the existing committed figure is not
clobbered.

Basis compatibility: the selfmap records carry
`r2_basis = "fold-local pooled OOF (plotted pair-file basis)"` — the same
basis as the plotted pair files — so the merge is apples-to-apples. Tier
mapping: selfmap `tier` '0' -> `t0_r2` (direct transfer), '6' -> `t6_r2`
(linear reparameterization of contexts); `base__base` carries `tier: None`
because a self map needs no reparameterization (its r2 == within_r2), and it
feeds the plotter's existing self-point path.

n<d CAVEAT (unchanged, already disclosed by the figure): the gsm8k_test1319
cells sit at n = 1293/1319 against d = 4096, so their held-out R2 is a
regularization-limit read, not a signal read. The plotter already titles that
panel "GSM8K test (n<d companion)"; the merged rows inherit that disclosure.
Truncation is deliberately NOT controlled for here (user scope call) — the
truncation-vs-skill read is the separate `truncation_control` figure.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any heavy import — shared-VM thread caps (#847)

_PLOTTER = Path(__file__).with_name("issue1336_metric_ladder_plots.py")

# The three transfer pairs round 5 filled, in ladder-forward order, plus the
# labels the existing figure's convention uses ("longer RLVR" for rlvr_long).
NEW_PAIRS: list[tuple[str, str]] = [
    # base__base FIRST. The plotter does not read self diamonds from a self ROW:
    # `self_r2` is keyed by each PAIR's TARGET stage and filled with that pair's
    # `within_r2` (issue1336_metric_ladder_plots.py:873). Nothing transfers INTO
    # base, so base never appears as a target and `self_r2` never gets a "base"
    # key — hence the missing base diamond, even though round 5 measured
    # base__base for all 8 panels. Listing it as a pair makes base a target and
    # fills the diamond. Verified: base__base covers exactly CORPUS_ORDER's 8
    # (format, corpus) panels, so get_row() cannot raise its missing-cell error.
    ("base__base", "base → base (self)"),
    ("sft__rlvr", "SFT → RLVR"),
    ("sft__rlvr_long", "SFT → longer RLVR"),
    ("rlvr__rlvr_long", "RLVR → longer RLVR"),
]

# selfmap `tier` value -> the ROWS key the plotter reads. Tiers 7/8 are mapped
# so that the moment the round-5 gap-filler is re-run at ALGEBRA_VERSION v3
# (which adds them) its cells feed the t8 series below with no further change;
# until then those cells carry tiers 0/6 only and the t8 series covers the
# ORIGINAL 7 pairs, which the legend label states explicitly.
TIER_TO_KEY = {"0": "t0_r2", "6": "t6_r2", "7": "t7_r2", "8": "t8_r2"}


_STALE_TAIL = "Only 7 of the "

# LINE COUNT IS LOAD-BEARING, twice over: the caption is one fig.text anchored at
# the bottom with va="bottom", so it grows UPWARD into the legend band (extra lines
# collide with the legend), and an unwrapped single long line instead expands the
# tight-bbox canvas ~2x and squeezes the panels. The sentence this replaces
# occupied exactly ONE rendered line, so the replacement must too. The truncation
# caveat lives in the round's report, not here — there is no line budget for it.
_CORRECTED_TAIL = (
    "SFT→RLVR, SFT→longer RLVR, RLVR→longer RLVR and the base self map were "
    "never run for the original figure and are measured here; base still has no "
    "INCOMING transfer, only its self diamond. Tier-6 lift statistics above "
    "cover the ORIGINAL 7 pairs."
)


def _patch_caption() -> None:
    """Rewrite the shared plotter's hardcoded 'never run' caption tail.

    The overlay draws its caption with a literal ``fig.text(...)`` and closes the
    figure before returning, so the text cannot be fixed after the call. The
    plotter file itself is deliberately NOT edited (a concurrent session runs
    it), so the Figure.text call is intercepted for this process only. Fails
    LOUD if the expected stale marker is absent — a silently unpatched caption
    would ship a figure whose caption contradicts its own data.
    """
    import matplotlib.figure as mfig

    original = mfig.Figure.text
    state = {"patched": 0}

    def _text(self, x, y, s, *args, **kwargs):
        if isinstance(s, str) and _STALE_TAIL in s:
            s = s[: s.index(_STALE_TAIL)] + _CORRECTED_TAIL
            state["patched"] += 1
        return original(self, x, y, s, *args, **kwargs)

    mfig.Figure.text = _text
    return state


def _ceiling_by_cell(repo_root: Path) -> tuple[dict, dict]:
    """The rep-swap ceiling per (pair, format, corpus) at layer 30.

    A FRESH ridge fit from the SOURCE's context vector to the TARGET's answer
    vector — `x_s -> y_t`, scored on the target's answers. It answers "is the
    target's answer state predictable from the source's context state AT ALL",
    which is a different question from every tier: the tiers all reuse the
    source's FITTED operator W_s on the TARGET's contexts. So this bounds how
    much of a tier shortfall is a missing-information problem (none, if the
    ceiling is high) versus an operator-transfer problem.

    Two sources, same construction, different names in the two scripts:
      * `repswap_r2` at per_layer/30 of the round-3 pair files
        (issue1336_metric_ladder.py: `fit_repswap = _v2_yfit(prep_s, Yt_l[tr])`,
        commented "rep-swap ceiling x_s -> y_t") — covers the ORIGINAL 7 pairs.
      * `cross_r2` in the round-5 selfmap cells
        (issue1336_selfmap_missing_pairs.py: `fit_cross = _v2_yfit(prep_s,
        Yt[tr])`) — covers the 3 pairs round 5 added.
    Neither is exposed in the aggregate the plotter reads, so neither has ever
    been plotted.
    """
    import glob as _glob
    import re as _re

    out: dict[tuple[str, str, str], float] = {}
    stats = {"pair_files": 0, "from_repswap": 0, "from_cross": 0}

    pat = str(repo_root / "data" / "issue_1336" / "hf_dl" / "**" / "metric_ladder" / "pair_*.json")
    for fp in sorted(_glob.glob(pat, recursive=True)):
        # The format literal anchors the split so a target like `rlvr_long`
        # (which itself contains an underscore) is not mis-parsed.
        m = _re.match(r"pair_(.+?)__(.+?)_(chat|naturalistic)_(.+)\.json", Path(fp).name)
        if not m:
            continue
        stats["pair_files"] += 1
        src, tgt, fmt, corpus = m.groups()
        val = json.load(open(fp)).get("per_layer", {}).get("30", {}).get("repswap_r2")
        if val is None:
            continue
        out[(f"{src}__{tgt}", fmt, corpus)] = float(val)
        stats["from_repswap"] += 1

    cells = repo_root / "eval_results" / "issue_1336" / "selfmap_missing_pairs" / "cells"
    for fp in sorted(cells.glob("*.json")):
        for rec in json.load(open(fp))["records"]:
            v = rec.get("cross_r2")
            if v is None or rec["pair"] == "base__base":
                continue
            key = (rec["pair"], rec["format"], rec["corpus"])
            if key not in out:
                out[key] = float(v)
                stats["from_cross"] += 1
    return out, stats


def _t8_by_cell(mod, scale: str = "raw") -> tuple[dict, dict]:
    """Tier-8 (FULL reparameterization) R2 per (pair, format, corpus) at layer 30.

    Tier 8 is the last rung of the 9-tier ladder ``t0..t8``: the source's fitted
    operator with BOTH spaces reparameterized —
    ``y = A_ans(W_s(A_ctx_rev x_t)) + b*``, where ``A_ctx_rev: x_t -> x_s`` maps
    the target's contexts into the source's context space (tier 6) and
    ``A_ans: y_s -> y_t`` maps the source's answer space forward into the
    target's. It answers "once BOTH endpoints are allowed a learned linear
    change of coordinates, does the source's map still predict the target's
    answers?" — so the t8-vs-t6 gap isolates what an answer-side
    reparameterization adds over a context-side one alone.

    Read straight off the plotter's own aggregate rows (``t8_r2``, already
    present for every original pair — no recompute), plus any round-5 rows that
    carry it via ``TIER_TO_KEY``. Rows lacking ``t8_r2`` are counted, never
    guessed.
    """
    out: dict[tuple[str, str, str], float] = {}
    stats = {"rows_scanned": 0, "with_t8": 0, "without_t8": 0}
    for r in mod.ROWS:
        if r.get("scale") != scale:
            continue
        stats["rows_scanned"] += 1
        v = r.get("t8_r2")
        if v is None:
            stats["without_t8"] += 1
            continue
        stats["with_t8"] += 1
        out[(r["pair"], r["format"], r["corpus"])] = float(v)
    return out, stats


def _patch_savefig_add_series(mod, ceiling: dict, t8: dict) -> dict:
    """Draw the rep-swap ceiling AND the tier-8 series onto the overlay before save.

    The overlay builds its figure, legend and caption internally and calls
    `savefig_paper(...)` then `plt.close(fig)`, so the only point at which the
    live figure is reachable is that save call. Intercepting it adds the series
    WITHOUT editing the shared plotter (a concurrent session runs it) and
    without duplicating any layout code.

    Colour stays SOURCE stage (one colour = one meaning, matching the rest of
    the figure) and is read off the artists the overlay already drew, so it can
    never drift from the plotter's own palette. Each new series is distinguished
    by linestyle + marker only.

    Tier 8 CANNOT go through the overlay's own ``tiers=`` argument, for two
    independent reasons: (a) the plotter's ``tier_style`` is built as
    ``{tiers[0]: ..., tiers[1]: ...}`` — exactly two entries — so a third tier
    KeyErrors; and (b) its drawing loop indexes ``get_row(...)[f"{tier}_r2"]``
    for EVERY pair in ``PAIR_ORDER``, and the round-5 merged rows have no
    ``t8_r2`` yet, so it would KeyError on them. Drawing here instead keeps the
    shared plotter untouched (a concurrent session runs it) and lets the series
    cover whatever cells actually HAVE a measured t8, counting the rest.
    """
    original = mod.savefig_paper
    state = {
        "axes": 0,
        "points": 0,
        "missing": 0,
        "t8_points": 0,
        "t8_missing": 0,
        "t8_pairs": set(),
    }

    def _savefig(fig, outname, *args, **kwargs):
        import matplotlib.pyplot as plt

        axes = [a for a in fig.axes if a.get_subplotspec() is not None][: len(mod.CORPUS_ORDER)]
        # Source stage -> colour, read off the drawn artists so the palette can
        # never drift from the plotter's own. The overlay labels its series
        # "<StageDisplayLabel>|<tier>" (e.g. "SFT|t0"), using the DISPLAY label,
        # so map back through STAGE_ORDER to the stage id. Scanning every panel
        # (not just the first) matters: a stage only appears where it is a
        # source, and RLVR is a source in just some panels.
        label_to_stage = {label: sid for sid, label in mod.STAGE_ORDER}
        stage_color: dict[str, object] = {}
        for ax_any in axes:
            h_any, l_any = ax_any.get_legend_handles_labels()
            for h, lab in zip(h_any, l_any):
                if "|" not in lab:
                    continue
                disp = lab.rsplit("|", 1)[0]
                sid = label_to_stage.get(disp)
                if sid is not None and sid not in stage_color:
                    stage_color[sid] = h.get_color()
        assert stage_color, (
            "could not recover per-source colours from the drawn artists "
            f"(labels seen: {sorted(set(l_any))}) — refusing to draw the "
            "ceiling series in a colour scheme that may not match the figure"
        )

        for ax, (fmt, corpus, _clabel) in zip(axes, mod.CORPUS_ORDER):
            state["axes"] += 1
            for src, color in stage_color.items():
                pts = []
                for pair, _plabel in mod.PAIR_ORDER:
                    s, t = mod.split_pair(pair)
                    if s != src:
                        continue
                    v = ceiling.get((pair, fmt, corpus))
                    if v is None:
                        state["missing"] += 1
                        continue
                    pts.append((mod.STAGE_IDX[t], v))
                if not pts:
                    continue
                pts.sort()
                state["points"] += len(pts)
                ax.plot(
                    [x for x, _ in pts],
                    [y for _, y in pts],
                    linestyle=(0, (4, 1, 1, 1)),
                    linewidth=1.1,
                    marker="s",
                    markersize=4.0,
                    color=color,
                    markerfacecolor=color,
                    alpha=0.95,
                    zorder=2,
                )

            # Tier 8 — the FULL reparameterization (both endpoints), same
            # per-source colour, dash-dot + filled triangle so it reads as a
            # TIER of the ladder rather than as the ceiling (square) or the
            # identity+bias floor (dotted, "v").
            for src, color in stage_color.items():
                pts = []
                for pair, _plabel in mod.PAIR_ORDER:
                    s, t = mod.split_pair(pair)
                    if s != src or s == t:  # self maps need no reparameterization
                        continue
                    v = t8.get((pair, fmt, corpus))
                    if v is None:
                        state["t8_missing"] += 1
                        continue
                    pts.append((mod.STAGE_IDX[t], v))
                    state["t8_pairs"].add(pair)
                if not pts:
                    continue
                pts.sort()
                state["t8_points"] += len(pts)
                ax.plot(
                    [x for x, _ in pts],
                    [y for _, y in pts],
                    linestyle=(0, (5, 1.5, 1, 1.5)),
                    linewidth=1.3,
                    marker="^",
                    markersize=4.6,
                    color=color,
                    markerfacecolor=color,
                    alpha=0.95,
                    zorder=3,
                )

        # Rebuild the legend with the appended entries. ncol goes 4 -> 5 so ten
        # entries still occupy TWO rows: a third row grows upward into the
        # caption (the collision this figure has already produced twice).
        if fig.legends:
            leg = fig.legends[0]
            uh = list(leg.legend_handles)
            ul = [t.get_text() for t in leg.texts]
            leg.remove()
            uh.append(
                plt.Line2D(
                    [],
                    [],
                    linestyle=(0, (4, 1, 1, 1)),
                    marker="s",
                    markersize=4.0,
                    color="#8a8a8a",
                    markerfacecolor="#8a8a8a",
                )
            )
            ul.append("rep-swap ceiling (fresh fit: source ctx → target ans)")
            # Coverage rides IN the label, computed from what was actually
            # drawn, so it cannot go stale when the round-5 t7/t8 refit lands.
            n_transfer = len([p for p, _ in mod.PAIR_ORDER if len(set(mod.split_pair(p))) == 2])
            n_t8 = len(state["t8_pairs"])
            t8_label = "tier 8 — full reparameterization (ctx + ans)"
            if n_t8 < n_transfer:
                t8_label += f" [{n_t8}/{n_transfer} pairs]"
            uh.append(
                plt.Line2D(
                    [],
                    [],
                    linestyle=(0, (5, 1.5, 1, 1.5)),
                    marker="^",
                    markersize=4.6,
                    color="#8a8a8a",
                    markerfacecolor="#8a8a8a",
                )
            )
            ul.append(t8_label)
            fig.legend(
                uh,
                ul,
                loc="center",
                ncol=5,
                fontsize=9,
                frameon=False,
                bbox_to_anchor=(0.5, 0.145),
            )
        return original(fig, outname, *args, **kwargs)

    mod.savefig_paper = _savefig
    return state


def _load_plotter():
    spec = importlib.util.spec_from_file_location("issue1336_mlp", _PLOTTER)
    assert spec is not None and spec.loader is not None, _PLOTTER
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _selfmap_rows(repo_root: Path) -> tuple[list[dict], dict]:
    """Fold the round-5 per-cell selfmap records into plotter-shaped ROWS.

    Returns (rows, stats). One row per (pair, format, corpus) carrying whatever
    tiers that cell measured; `within_r2` is taken from the record (identical
    across a cell's tier rows by construction).
    """
    cells_dir = repo_root / "eval_results" / "issue_1336" / "selfmap_missing_pairs" / "cells"
    files = sorted(cells_dir.glob("*.json"))
    assert files, f"no selfmap cell files under {cells_dir}"

    grouped: dict[tuple[str, str, str], dict] = defaultdict(dict)
    stats = {
        "cell_files": len(files),
        "records": 0,
        "self_records": 0,
        "degenerate_records": 0,
        "tiers_seen": defaultdict(int),
    }
    for fp in files:
        for rec in json.load(open(fp))["records"]:
            stats["records"] += 1
            tier = rec.get("tier")
            tier_s = str(tier)
            stats["tiers_seen"][tier_s] += 1
            if rec.get("degenerate_n_lt_d"):
                stats["degenerate_records"] += 1
            key = (rec["pair"], rec["format"], rec["corpus"])
            row = grouped[key]
            row.setdefault("pair", rec["pair"])
            row.setdefault("format", rec["format"])
            row.setdefault("corpus", rec["corpus"])
            row.setdefault("scale", "raw")
            row.setdefault("n", rec["n"])
            row.setdefault("within_r2", rec["within_r2"])
            row.setdefault("degenerate_n_lt_d", bool(rec.get("degenerate_n_lt_d")))
            if tier_s in TIER_TO_KEY:
                row[TIER_TO_KEY[tier_s]] = rec["r2"]
            else:
                # Self map (tier None): r2 == within_r2 and is tier-invariant,
                # so it reads identically at both tiers.
                stats["self_records"] += 1
                row.setdefault("t0_r2", rec["r2"])
                row.setdefault("t6_r2", rec["r2"])
    return list(grouped.values()), stats


def main() -> None:
    mod = _load_plotter()
    repo_root = mod._REPO_ROOT

    before = len(mod.ROWS)
    new_rows, stats = _selfmap_rows(repo_root)

    existing = {(r["pair"], r["format"], r["corpus"], r["scale"]) for r in mod.ROWS}
    added = [
        r for r in new_rows if (r["pair"], r["format"], r["corpus"], r["scale"]) not in existing
    ]
    collisions = len(new_rows) - len(added)

    mod.ROWS.extend(added)
    have_pairs = {r["pair"] for r in added}
    for pair, label in NEW_PAIRS:
        if pair in have_pairs and pair not in {p for p, _ in mod.PAIR_ORDER}:
            mod.PAIR_ORDER.append((pair, label))

    print(f"[merge] plotter ROWS {before} -> {len(mod.ROWS)} (+{len(added)})")
    print(f"[merge] selfmap: {stats['cell_files']} cell files, {stats['records']} records")
    print(f"[merge] tiers seen: {dict(stats['tiers_seen'])}")
    print(f"[merge] self-map records (tier None): {stats['self_records']}")
    print(f"[merge] degenerate (n<d) records: {stats['degenerate_records']}")
    print(f"[merge] collisions skipped: {collisions}")
    print(f"[merge] PAIR_ORDER now {len(mod.PAIR_ORDER)}: {[p for p, _ in mod.PAIR_ORDER]}")
    deg = sorted({(r["pair"], r["corpus"]) for r in added if r.get("degenerate_n_lt_d")})
    print(f"[merge] degenerate cells among added rows: {deg}")

    ceiling, cstats = _ceiling_by_cell(repo_root)
    print(
        f"[ceiling] {len(ceiling)} cells "
        f"({cstats['from_repswap']} repswap from {cstats['pair_files']} pair files, "
        f"{cstats['from_cross']} cross from round-5 selfmap)"
    )
    assert ceiling, "no rep-swap ceiling values recovered — refusing to draw an empty series"

    t8, t8stats = _t8_by_cell(mod, scale="raw")
    print(
        f"[t8] {len(t8)} cells carry a measured t8_r2 "
        f"({t8stats['with_t8']}/{t8stats['rows_scanned']} raw-scale rows; "
        f"{t8stats['without_t8']} rows have none yet)"
    )
    assert t8, "no t8_r2 values recovered — refusing to draw an empty tier-8 series"

    ceil_state = _patch_savefig_add_series(mod, ceiling, t8)
    cap_state = _patch_caption()
    pts = mod.make_source_target_overlay(
        tiers=("t0", "t6"),
        scale="raw",
        outname="metric_ladder_source_target_both_tiers_round5",
    )
    print(f"[fig] overlay returned {len(pts)} points")
    assert cap_state["patched"] == 1, (
        "caption patch did not fire exactly once "
        f"(fired {cap_state['patched']}x) — the shared plotter's stale "
        f"'{_STALE_TAIL}...' sentence may have changed; refusing to ship a "
        "figure whose caption contradicts its plotted data"
    )
    print("[fig] caption tail corrected (stale 'never run' claim replaced)")
    print(
        f"[ceiling] drew {ceil_state['points']} ceiling points across "
        f"{ceil_state['axes']} panels ({ceil_state['missing']} cells had no ceiling value)"
    )
    assert ceil_state["points"] > 0, "ceiling series drew zero points"
    print(
        f"[t8] drew {ceil_state['t8_points']} tier-8 points over "
        f"{len(ceil_state['t8_pairs'])} transfer pairs "
        f"({ceil_state['t8_missing']} cells had no t8 value): "
        f"{sorted(ceil_state['t8_pairs'])}"
    )
    assert ceil_state["t8_points"] > 0, "tier-8 series drew zero points"

    out = repo_root / "eval_results" / "issue_1336" / "metric_ladder_source_target"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_aggregate": str(mod.AGG_PATH),
        "layer": 30,
        "tiers": {
            "t0": "Tier 0 — direct transfer (no correction)",
            "t6": "Tier 6 — linear reparameterization of contexts",
            "t8": (
                "Tier 8 — FULL reparameterization (contexts AND answers): "
                "y = A_ans(W_s(A_ctx_rev x_t)) + b*"
            ),
        },
        "t8_coverage": {
            "cells_drawn": ceil_state["t8_points"],
            "cells_missing": ceil_state["t8_missing"],
            "pairs": sorted(ceil_state["t8_pairs"]),
            "note": (
                "t8 is read from the plotter aggregate's own t8_r2 (zero recompute). "
                "The three round-5 forward pairs have no t8 yet — the gap-filler "
                "ran at ALGEBRA_VERSION v2, which computed tiers 0/6 only; v3 adds "
                "t7/t8 and needs a refit of those 24 cells."
            ),
        },
        "ladder_order": [s for s, _ in mod.STAGE_ORDER],
        "overlay": True,
        "round5_merge": {
            "selfmap_cell_files": stats["cell_files"],
            "selfmap_records": stats["records"],
            "rows_added": len(added),
            "pairs_added": sorted(have_pairs),
            "degenerate_cells": [list(x) for x in deg],
            "note": (
                "Folds round-5 selfmap_missing_pairs cells into the committed "
                "source->target overlay; the three previously-never-run "
                "transfer pairs and the base self map are now measured. "
                "Truncation is NOT controlled for in this figure."
            ),
        },
        "points": pts,
    }
    dest = out / "source_target_both_tiers_round5_points.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[fig] wrote {dest} ({len(pts)} points)")


if __name__ == "__main__":
    main()
