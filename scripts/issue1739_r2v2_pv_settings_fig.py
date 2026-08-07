#!/usr/bin/env python3
"""#1739: persona-vector-projection methods only, rho by eval SETTING, per protocol.

Four methods, all persona-vector projections (user scope call 2026-08-07):

    1. PV projected on the CONTEXT                        arm1_ctx_e1
    2. PV projected on the MAPPED answer, LINEAR map      arm6_map_proj_e1 (r2v2)
    3. PV projected on the MAPPED answer, MLP map         arm6_map_proj_e1 (nonlinear_map round)
    4. PV projected on the REAL answer (upper bound)      arm11_oracle_proj

Two figures (P-A and P-B/LODO), four panels each (three behaviors + the
cross-behavior average), four setting groups per panel.

DATA-AVAILABILITY CAVEAT (rendered on the figure, do not silently drop):
the MLP-map arm exists only in the earlier ``nonlinear_map`` round, which
- has NO pvsynth and NO wildchat_rung (generic chat) rungs,
- has NO P-A/P-B protocol split (so it appears in the P-A figure only),
- fit its map on the 18,793-row GENERIC pool, whereas r2v2's linear map used
  the 34,793-row ADD/union pool -- so linear-vs-MLP here is NOT a matched-pool
  contrast. Cells with no MLP data are labelled ``MLP: not run``.

Usage
-----
    uv run python scripts/issue1739_r2v2_pv_settings_fig.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Bind the shared-VM BLAS/intra-op thread caps (#847) BEFORE numpy/matplotlib
# freeze their pools at import time.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BEHAVIORS = ("evil", "sycophancy", "hallucination")

OOD_RUNGS = {
    "evil": ("hhrt", "toxicchat", "evil_mhj", "evil_pair", "evil_tomgibbs"),
    "sycophancy": ("aita", "sycoans", "sycoays", "sycofb", "sycomim", "sycomwe"),
    "hallucination": ("nqopen", "simpleqa"),
}

SETTINGS = (
    ("pvsynth", "Persona-vector\nsynthetic"),
    ("generic", "Generic chat\n(WildChat)"),
    ("indist", "In-distribution\n(held-out rows)"),
    ("ood", "Completely OOD\n(avg. of datasets)"),
)

# key -> (legend label, colour, hatch)
METHODS = (
    ("ctx", "PV on context", "#4C72B0", None),
    ("map_lin", "PV on mapped answer — linear map", "#DD8452", None),
    ("map_mlp", "PV on mapped answer — MLP map", "#DD8452", "///"),
    ("oracle", "PV on real answer (upper bound)", "#55A868", None),
)


def _git_show_json(commit: str, path: str) -> dict:
    out = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _sem(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return float(st.stdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0


def load_mlp_rows() -> dict[tuple[str, str], list[float]]:
    """(behavior, eval_rung) -> rho list from the nonlinear_map MLP round.

    Slice matched to r2v2 as far as the round allows: regime e1, variant
    context_end, the FULL u-pool, and the largest readout budget present.
    """
    out: dict[tuple[str, str], list[float]] = {}
    for beh in BEHAVIORS:
        p = (
            _REPO_ROOT
            / f"eval_results/issue_1739/nonlinear_map/{beh}/mlp/arm_results/all_arms_spearman.json"
        )
        if not p.exists():
            continue
        rows = [
            r
            for r in json.loads(p.read_text()).get("transfer_rows", [])
            if r.get("arm") == "arm6_map_proj_e1"
            and r.get("regime") == "e1"
            and r.get("variant") == "context_end"
            and str(r.get("u_rung_label")) == "full"
        ]
        if not rows:
            continue
        budget = max(int(r["budget_l"]) for r in rows)
        for r in rows:
            if int(r["budget_l"]) != budget:
                continue
            v = r.get("rho_frozen")
            if v is not None:
                out.setdefault((beh, r["eval_rung"]), []).append(float(v))
    return out


def collect(fits: dict[str, dict], mlp: dict, spread: dict, protocol: str):
    values: dict[tuple[str, str, str], tuple[float, float]] = {}
    flags: dict[tuple[str, str], tuple[int, int]] = {}

    arm_of = {"ctx": "arm1_ctx_e1", "map_lin": "arm6_map_proj_e1", "oracle": "arm11_oracle_proj"}

    for beh in BEHAVIORS:
        rows = fits[beh]["transfer_rows"]
        ood = OOD_RUNGS[beh]

        for mkey, arm in arm_of.items():
            ar = [r for r in rows if r["arm"] == arm]
            if protocol == "P-A":
                pa = [r for r in ar if r["fit"] == "P-A"]
                oof = [r for r in ar if r["fit"] == "P-A-train-oof"]
                picks = {
                    "pvsynth": [r["rho_frozen"] for r in pa if r["eval_rung"] == "pvsynth"],
                    "generic": [r["rho_frozen"] for r in pa if r["eval_rung"] == "wildchat_rung"],
                    "indist": [r["rho_frozen"] for r in oof if r["eval_rung"] == "train"],
                    "ood": [r["rho_frozen"] for r in pa if r["eval_rung"] in ood],
                }
            else:
                pb = [r for r in ar if r["protocol"] == "P-B"]
                picks = {
                    "pvsynth": [r["rho_frozen"] for r in pb if r["eval_rung"] == "pvsynth"],
                    "generic": [r["rho_frozen"] for r in pb if r["eval_rung"] == "wildchat_rung"],
                    "indist": [r["rho_frozen"] for r in pb if r["eval_rung"] == "heldin:train"],
                    "ood": [
                        r["rho_frozen"]
                        for r in pb
                        if r["fit"].replace("P-B-holdout-", "") == r["eval_rung"]
                    ],
                }
            for skey, _lab in SETTINGS:
                vals = [v for v in picks[skey] if v is not None]
                if vals:
                    values[(beh, skey, mkey)] = (float(st.mean(vals)), _sem(vals))

        # MLP-map arm: P-A figure only; only the rungs the nonlinear_map round ran
        if protocol == "P-A":
            ind = mlp.get((beh, "train"))
            if ind:
                values[(beh, "indist", "map_mlp")] = (float(st.mean(ind)), _sem(ind))
            per_rung = [
                float(st.mean(v)) for (b, r), v in mlp.items() if b == beh and r in ood and v
            ]
            if per_rung:
                values[(beh, "ood", "map_mlp")] = (float(st.mean(per_rung)), _sem(per_rung))

        rung_for = {
            "pvsynth": ("pvsynth",),
            "generic": ("wildchat_rung",),
            "indist": ("train",),
            "ood": ood,
        }
        for skey, rungs in rung_for.items():
            checked = [r for r in rungs if f"{beh}|{r}" in spread]
            bad = [r for r in checked if not spread[f"{beh}|{r}"]["spread_ok"]]
            flags[(beh, skey)] = (len(bad), len(checked))

    for skey, _ in SETTINGS:
        for mkey, *_ in METHODS:
            per = [values[(b, skey, mkey)][0] for b in BEHAVIORS if (b, skey, mkey) in values]
            if per:
                values[("average", skey, mkey)] = (float(st.mean(per)), _sem(per))
        flags[("average", skey)] = (
            sum(flags.get((b, skey), (0, 0))[0] for b in BEHAVIORS),
            sum(flags.get((b, skey), (0, 0))[1] for b in BEHAVIORS),
        )
    return values, flags


def draw(values, flags, protocol: str, out_png: Path, subtitle: str) -> None:
    panels = [*BEHAVIORS, "average"]
    titles = {
        "evil": "Evil",
        "sycophancy": "Sycophancy",
        "hallucination": "Hallucination",
        "average": "Averaged across behaviors",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.4))
    n_m = len(METHODS)
    width = 0.19

    for ax, panel in zip(axes.ravel(), panels, strict=True):
        centers = np.arange(len(SETTINGS))
        for j, (mkey, _label, color, hatch) in enumerate(METHODS):
            xs, ys, es = [], [], []
            for i, (skey, _lab) in enumerate(SETTINGS):
                v = values.get((panel, skey, mkey))
                x = centers[i] + (j - (n_m - 1) / 2) * width
                if v is None:
                    if mkey == "map_mlp":
                        ax.text(
                            x,
                            0.012,
                            "MLP: not run",
                            transform=ax.get_xaxis_transform(),
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=7.2,
                            color="#8a8a8a",
                        )
                    continue
                xs.append(x)
                ys.append(v[0])
                es.append(v[1])
            if xs:
                ax.bar(
                    xs,
                    ys,
                    width * 0.9,
                    yerr=es,
                    color="white" if hatch else color,
                    edgecolor=color,
                    hatch=hatch,
                    linewidth=1.4,
                    error_kw=dict(lw=1.0, capsize=2.5, ecolor="#444444"),
                    zorder=3,
                )

        for i, (skey, _lab) in enumerate(SETTINGS):
            bad, tot = flags.get((panel, skey), (0, 0))
            if bad:
                ax.axvspan(
                    centers[i] - 0.5, centers[i] + 0.5, color="#B03030", alpha=0.085, zorder=0
                )
                note = "low DV spread" if tot <= 1 else f"low DV spread {bad}/{tot} rungs"
                ax.text(
                    centers[i],
                    0.965,
                    f"⚠ {note}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color="#8B2020",
                    zorder=5,
                )

        ax.axhline(0.0, color="#666666", lw=0.9, zorder=2)
        ax.set_xticks(centers)
        ax.set_xticklabels([lab for _k, lab in SETTINGS], fontsize=9.5)
        ax.set_ylabel(r"Spearman $\rho$  (frozen layer)")
        ax.set_title(titles[panel])
        ax.set_ylim(-0.20, 0.95)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    handles = [
        mpatches.Patch(
            facecolor="white" if hatch else color,
            edgecolor=color,
            hatch=hatch,
            linewidth=1.4,
            label=label,
        )
        for _k, label, color, hatch in METHODS
    ]
    handles.append(
        mpatches.Patch(facecolor="#B03030", alpha=0.085, label="⚠ low DV spread — untrustworthy")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.008),
    )
    fig.suptitle(
        f"#1739 — persona-vector readouts by evaluation setting  ({protocol})",
        fontsize=14,
        x=0.5,
        y=0.995,
    )
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0, 0.075, 1, 0.945))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-commit", default="5aae0a472b")
    ap.add_argument("--spread-json", default="/tmp/spread_1739.json")
    ap.add_argument("--out-dir", default="figures/issue_1739")
    ap.add_argument(
        "--no-mlp",
        action="store_true",
        help="drop the MLP-map arm (its rung coverage is partial and its map pool differs)",
    )
    args = ap.parse_args()

    global METHODS
    suffix = ""
    if args.no_mlp:
        METHODS = tuple(m for m in METHODS if m[0] != "map_mlp")
        suffix = "_nomlp"

    fits = {
        b: _git_show_json(
            args.fits_commit, f"eval_results/issue_1739/r2v2_fits/{b}/all_arms_spearman.json"
        )
        for b in BEHAVIORS
    }
    mlp = load_mlp_rows()
    spread = json.loads(Path(args.spread_json).read_text())
    out_dir = _REPO_ROOT / args.out_dir

    subs = {
        "P-A": (
            "P-A readout: trained on one trait-eliciting dataset + judged WildChat split   |   "
            "MLP-map arm from the earlier nonlinear_map round (generic-pool map fit; no pvsynth/WildChat rungs)"
        ),
        "P-B": (
            "P-B readout (LODO): one trait-eliciting dataset held out whole; OOD bar is that held-out dataset   |   "
            "MLP-map arm was never run under LODO"
        ),
    }
    if args.no_mlp:
        subs = {
            "P-A": "P-A readout: trained on one trait-eliciting dataset + judged WildChat split",
            "P-B": (
                "P-B readout (LODO): one trait-eliciting dataset held out whole; "
                "OOD bar is that held-out dataset"
            ),
        }

    meta: dict = {}
    for protocol in ("P-A", "P-B"):
        values, flags = collect(fits, mlp, spread, protocol)
        png = out_dir / f"issue1739_pv_settings_{protocol.replace('-', '').lower()}{suffix}.png"
        draw(values, flags, protocol, png, subs[protocol])
        meta[protocol] = {
            f"{b}|{s}|{m}": dict(rho=round(v[0], 4), err=round(v[1], 4))
            for (b, s, m), v in values.items()
        }
    meta["_provenance"] = {
        "fits_commit": args.fits_commit,
        "linear_source": "eval_results/issue_1739/r2v2_fits/<behavior>/all_arms_spearman.json",
        "mlp_source": "eval_results/issue_1739/nonlinear_map/<behavior>/mlp/arm_results/all_arms_spearman.json",
        "mlp_slice": "arm6_map_proj_e1, regime e1, variant context_end, u_rung_label full, max budget_l",
        "mlp_caveat": "generic-pool (18,793-row) map fit vs r2v2 ADD-pool (34,793) linear map; "
        "no pvsynth / wildchat_rung rungs; no P-A/P-B protocol split",
        "ood_rungs": {k: list(v) for k, v in OOD_RUNGS.items()},
        "spread_gate": "sd>=10 and frac(<=10)<=0.80 and frac(>=90)<=0.80 on a 0-100 DV",
    }
    (out_dir / f"issue1739_pv_settings_meta{suffix}.json").write_text(json.dumps(meta, indent=1))
    print("wrote meta")


if __name__ == "__main__":
    main()
