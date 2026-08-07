#!/usr/bin/env python3
"""#1739 r2v2: rho by eval SETTING x method, one figure per readout protocol.

Two figures (P-A and P-B / LODO), each with four panels — one per behavior plus
one averaged across behaviors. Each panel carries four setting groups:

    persona-vector synthetic | generic chat | in-distribution | completely OOD

with one bar per method. Colour encodes the READOUT INPUT (context / mapped
answer / real answer); fill style encodes the READOUT FAMILY (persona-vector
projection vs fitted ridge), matching the methodology grouping.

Setting -> rung resolution
-------------------------
P-A trains the readout on ONE trait-eliciting dataset (the ``train`` budget
cell) plus the judged WildChat split, so:

    in-distribution  = ``train`` under the ``P-A-train-oof`` fit (out-of-fold)
    completely OOD   = every OTHER trait-eliciting dataset, averaged

P-B holds one trait-eliciting dataset out whole and trains on an 80% group
slice of the rest, so:

    in-distribution  = ``heldin:train``, averaged over the holdout fits
    completely OOD   = each fit's OWN held-out rung (the true LODO read),
                       averaged over holdouts

``pvsynth`` and ``wildchat_rung`` are read directly (averaged over the holdout
fits under P-B).

Trust flag
----------
A setting is marked LOW DV SPREAD when a constituent rung's judged DV is
degenerate: sd < 10, or >80% of contexts in the bottom bin (<=10), or >80% in
the top bin (>=90), on a 0-100 scale (0-1 binary DVs are rescaled first). Those
bars are drawn washed out with a hazard overlay — a rho against a floor-pinned
DV is not a comparison. Nothing is dropped; the flag is additive.

Usage
-----
    uv run python scripts/issue1739_r2v2_settings_fig.py \
        --fits-commit 5aae0a472b --out-dir figures/issue_1739
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


BEHAVIORS = ("evil", "sycophancy", "hallucination")

# Trait-eliciting datasets that the P-A readout never trains on. `train` is the
# P-A budget cell and is therefore the in-distribution rung, not an OOD one.
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

# arm -> (concise label, input source, readout family)
METHODS = (
    ("arm1_ctx_e1", "Context", "Context", "proj"),
    ("arm4_ridge_ctx", "Context", "Context", "ridge"),
    ("arm6_map_proj_e1", "Mapped answer", "Mapped answer", "proj"),
    ("arm7_map_ridge_pred", "Mapped answer", "Mapped answer", "ridge"),
    ("arm11_oracle_proj", "Real answer (oracle)", "Real answer (oracle)", "proj"),
)

SOURCE_COLOR = {
    "Context": "#4C72B0",
    "Mapped answer": "#DD8452",
    "Real answer (oracle)": "#55A868",
}
FAMILY_LABEL = {"proj": "persona-vector projection", "ridge": "fitted ridge"}


def _git_show(commit: str, path: str) -> dict:
    out = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return float(st.mean(vals)) if vals else None


def _sem(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    return float(st.stdev(vals) / math.sqrt(len(vals)))


def collect(fits: dict[str, dict], spread: dict, protocol: str) -> tuple[dict, dict]:
    """Return (values, flags).

    values[(behavior, setting, arm)] = (rho, err, n_rungs)
    flags[(behavior, setting)]       = (n_failing_rungs, n_total_rungs)
    """
    values: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    flags: dict[tuple[str, str], tuple[int, int]] = {}

    for beh in BEHAVIORS:
        rows = fits[beh]["transfer_rows"]
        ood_rungs = OOD_RUNGS[beh]

        for arm, *_ in METHODS:
            arm_rows = [r for r in rows if r["arm"] == arm]

            if protocol == "P-A":
                pa = [r for r in arm_rows if r["fit"] == "P-A"]
                oof = [r for r in arm_rows if r["fit"] == "P-A-train-oof"]
                picks = {
                    "pvsynth": [r["rho_frozen"] for r in pa if r["eval_rung"] == "pvsynth"],
                    "generic": [r["rho_frozen"] for r in pa if r["eval_rung"] == "wildchat_rung"],
                    "indist": [r["rho_frozen"] for r in oof if r["eval_rung"] == "train"],
                    "ood": [r["rho_frozen"] for r in pa if r["eval_rung"] in ood_rungs],
                }
                # single-rung settings carry their bootstrap CI as the error bar
                cis = {
                    "pvsynth": [r["ci_frozen"] for r in pa if r["eval_rung"] == "pvsynth"],
                    "generic": [r["ci_frozen"] for r in pa if r["eval_rung"] == "wildchat_rung"],
                    "indist": [r["ci_frozen"] for r in oof if r["eval_rung"] == "train"],
                }
            else:
                pb = [r for r in arm_rows if r["protocol"] == "P-B"]
                lodo = [
                    r["rho_frozen"]
                    for r in pb
                    if r["fit"].replace("P-B-holdout-", "") == r["eval_rung"]
                ]
                picks = {
                    "pvsynth": [r["rho_frozen"] for r in pb if r["eval_rung"] == "pvsynth"],
                    "generic": [r["rho_frozen"] for r in pb if r["eval_rung"] == "wildchat_rung"],
                    "indist": [r["rho_frozen"] for r in pb if r["eval_rung"] == "heldin:train"],
                    "ood": lodo,
                }
                cis = {}

            for key, _label in SETTINGS:
                vals = [v for v in picks[key] if v is not None]
                mu = _mean(vals)
                if mu is None:
                    continue
                ci = (cis.get(key) or [None])[0] if key in cis else None
                if ci and len(vals) == 1:
                    err = float(max(ci[1] - mu, mu - ci[0]))
                else:
                    err = _sem(vals)
                values[(beh, key, arm)] = (mu, err, len(vals))

        # spread flags, per setting
        rung_for = {
            "pvsynth": ("pvsynth",),
            "generic": ("wildchat_rung",),
            "indist": ("train",),
            "ood": ood_rungs,
        }
        for key, rungs in rung_for.items():
            checked = [r for r in rungs if f"{beh}|{r}" in spread]
            bad = [r for r in checked if not spread[f"{beh}|{r}"]["spread_ok"]]
            flags[(beh, key)] = (len(bad), len(checked))

    # cross-behavior average panel
    for key, _ in SETTINGS:
        for arm, *_ in METHODS:
            per = [values[(b, key, arm)][0] for b in BEHAVIORS if (b, key, arm) in values]
            if per:
                values[("average", key, arm)] = (float(st.mean(per)), _sem(per), len(per))
        bad = sum(flags.get((b, key), (0, 0))[0] for b in BEHAVIORS)
        tot = sum(flags.get((b, key), (0, 0))[1] for b in BEHAVIORS)
        flags[("average", key)] = (bad, tot)

    return values, flags


def draw(values: dict, flags: dict, protocol: str, out_png: Path, subtitle: str) -> None:
    panels = [*BEHAVIORS, "average"]
    titles = {
        "evil": "Evil",
        "sycophancy": "Sycophancy",
        "hallucination": "Hallucination",
        "average": "Averaged across behaviors",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.4))
    n_m = len(METHODS)
    width = 0.145

    for ax, panel in zip(axes.ravel(), panels, strict=True):
        centers = np.arange(len(SETTINGS))
        for j, (arm, label, source, family) in enumerate(METHODS):
            xs, ys, es = [], [], []
            for i, (key, _lab) in enumerate(SETTINGS):
                v = values.get((panel, key, arm))
                if v is None:
                    continue
                xs.append(centers[i] + (j - (n_m - 1) / 2) * width)
                ys.append(v[0])
                es.append(v[1])
            if not xs:
                continue
            color = SOURCE_COLOR[source]
            ax.bar(
                xs,
                ys,
                width * 0.92,
                yerr=es,
                color=color if family == "proj" else "white",
                edgecolor=color,
                linewidth=1.4,
                hatch=None if family == "proj" else "///",
                error_kw=dict(lw=1.0, capsize=2.5, ecolor="#444444"),
                zorder=3,
            )

        # hazard shading for settings whose DV spread is degenerate
        for i, (key, _lab) in enumerate(SETTINGS):
            bad, tot = flags.get((panel, key), (0, 0))
            if bad == 0:
                continue
            ax.axvspan(centers[i] - 0.5, centers[i] + 0.5, color="#B03030", alpha=0.085, zorder=0)
            note = "low DV spread" if tot <= 1 else f"low DV spread {bad}/{tot} rungs"
            ax.text(
                centers[i],
                ax.get_ylim()[1] if False else 0.965,
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

    handles = []
    for source, color in SOURCE_COLOR.items():
        handles.append(mpatches.Patch(facecolor=color, edgecolor=color, label=f"— {source} —"))
        for family in ("proj", "ridge"):
            if not any(s == source and f == family for _a, _l, s, f in METHODS):
                continue
            handles.append(
                mpatches.Patch(
                    facecolor=color if family == "proj" else "white",
                    edgecolor=color,
                    hatch=None if family == "proj" else "///",
                    linewidth=1.4,
                    label=f"    {FAMILY_LABEL[family]}",
                )
            )
    handles.append(
        mpatches.Patch(facecolor="#B03030", alpha=0.085, label="⚠ low DV spread — untrustworthy")
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.008),
    )
    fig.suptitle(
        f"#1739 — behavior predictability by evaluation setting  ({protocol})",
        fontsize=14,
        x=0.5,
        y=0.995,
    )
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=10, color="#444444")
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
    args = ap.parse_args()

    fits = {
        b: _git_show(
            args.fits_commit, f"eval_results/issue_1739/r2v2_fits/{b}/all_arms_spearman.json"
        )
        for b in BEHAVIORS
    }
    spread = json.load(open(args.spread_json))
    out_dir = _REPO_ROOT / args.out_dir

    subtitles = {
        "P-A": "P-A readout: trained on one trait-eliciting dataset + judged WildChat split",
        "P-B": "P-B readout (LODO): one trait-eliciting dataset held out whole; OOD bar is that held-out dataset",
    }
    meta: dict[str, dict] = {}
    for protocol in ("P-A", "P-B"):
        values, flags = collect(fits, spread, protocol)
        png = out_dir / f"issue1739_r2v2_settings_{protocol.replace('-', '').lower()}.png"
        draw(values, flags, protocol, png, subtitles[protocol])
        meta[protocol] = {
            f"{b}|{k}|{a}": dict(rho=round(v[0], 4), err=round(v[1], 4), n_rungs=v[2])
            for (b, k, a), v in values.items()
            for _ in (0,)
        }
        meta[f"{protocol}_flags"] = {f"{b}|{k}": list(v) for (b, k), v in flags.items()}

    meta["_provenance"] = {
        "fits_commit": args.fits_commit,
        "source": "eval_results/issue_1739/r2v2_fits/<behavior>/all_arms_spearman.json",
        "ood_rungs": {k: list(v) for k, v in OOD_RUNGS.items()},
        "spread_gate": "sd>=10 and frac(<=10)<=0.80 and frac(>=90)<=0.80 on a 0-100 DV "
        "(0-1 binary DVs rescaled x100)",
    }
    (out_dir / "issue1739_r2v2_settings_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out_dir / 'issue1739_r2v2_settings_meta.json'}")


if __name__ == "__main__":
    main()
