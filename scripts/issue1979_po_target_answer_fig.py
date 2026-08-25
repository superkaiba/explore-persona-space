"""#1979 — positive-only arms: five predictors against the leakage DV.

Extends `issue1979_po_three_predictors_fig.py` with the two candidates that use
the answers generated AT THE TARGET (destination) prefix as the comparison
object, rather than the training-row answer centroid:

  pmap_tgt  cos(M0 . context_vec, base answer vec at that same prefix)
            "mapped answers vs target answers" — does the base context->answer
            map predict what the model actually says at this prefix?  NEW here.

  p2_ps     cos(base answer vec at prefix, base answer vec at the arm's OWN
            trained prefix) — "target answers vs trained-ctx answers".  Already
            computed by the race and persisted per arm, but never raced (a
            stated #1979 descope: the candidate set fixes training-row-centroid
            anchors).

For contrast the three already-plotted candidates all compare against the
TRAINED-ON answers / contexts (the training-row centroids):

  p2_tc  cos(answer vec at prefix, training-row answer centroid)
  p3b_tc cos(M0 . context vec, training-row answer centroid)
  p1_tc  cos(context vec at prefix, training-row context centroid)

Metric is the committed panel-centered cosine (raw); whitening and CSLS are
omitted because neither flips a verdict on these arms.

Band caveat: the per-arm band drawn here is the enlarged-candidate-set band
(content K=30, marker K=28) from `enlarged_band.json`. The two candidates added
here are not in that set. Measured cost of enlarging was +0.012 (content) /
+0.026 (marker) for +18 near-duplicate candidates, so two more shifts the edge
by well under 0.005 — smaller than the plotted marker width, but the band is
not exact for the two new points and they are shown for inspection, not raced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from issue1979_race import _center  # noqa: E402
from issue1979_whiten_csls_band import CARRIED_COL, N_PERM, SEED, _rank_z, band_for  # noqa: E402
from issue1979_whiten_csls_sweep import (  # noqa: E402
    CARRIED,
    DV_BY_KIND,
    RACE,
    REMETRIC,
    SETTINGS,
    arm_columns,
    load_inputs,
)

REPO_ROOT = SCRIPTS_DIR.parent
BAND = REPO_ROOT / "eval_results/issue_1979/whiten_csls/enlarged_band.json"
OUT_JSON = REPO_ROOT / "eval_results/issue_1979/whiten_csls/po_target_answer.json"
FIG_DIR = REPO_ROOT / "figures/issue_1979"
TENSORS = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1979_whitencsls/battery/ingredient_tensors.pt"
)

# (label, arm_id, kind, layer, position) — PRIMARY position per kind, as raced.
ARMS = [
    ("casual writing", "cas-pers-po-lr1e5-s42", "content", 19, "last_prompt"),
    ("impoliteness", "imp-pers-po-lr1e5-s42", "content", 19, "last_prompt"),
    ("marker token", "mk-pers-po-lr5e6-s42", "marker", 25, "last_prompt"),
]
PREDS = [
    ("p2_tc", "target answers vs trained-on answers", "#4C72B0"),
    ("p3b_tc", "mapped answers vs trained-on answers", "#C44E52"),
    ("p1_tc", "target context vs trained contexts", "#8172B2"),
    ("pmap_tgt", "mapped answers vs target answers  (new)", "#55A868"),
    ("p2_ps", "target answers vs trained-ctx answers  (new)", "#DD8452"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tensors", type=Path, default=TENSORS)
    ap.add_argument("--band", type=Path, default=BAND)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    assert args.tensors.exists(), f"staged tensors missing: {args.tensors}"
    tens = torch.load(args.tensors, map_location="cpu", weights_only=True)
    bands = {r["arm_id"]: r for r in json.loads(args.band.read_text())["rows"]}
    inputs = load_inputs()
    arm_by_id = {a["arm_id"]: a for a in inputs["arms"]}

    records: dict[str, dict] = {}
    for label, aid, kind, layer, pos in ARMS:
        frame = json.loads((RACE / f"frame_{aid}.json").read_text())["frame"]
        dv = np.asarray(frame[DV_BY_KIND[kind]], dtype=np.float64)

        # NEW: mapped answer vs the base answers generated at that same prefix.
        m0 = np.asarray(tens[f"m0pred/{kind}/L{layer}/{pos}"].double().numpy())
        v0 = np.asarray(tens[f"{aid}/L{layer}/{pos}/Vbar0"].double().numpy())
        assert m0.shape == v0.shape, (aid, m0.shape, v0.shape)
        mc, _ = _center(m0)
        vc, _ = _center(v0)
        # row-wise cosine between the two centered per-prefix vector sets
        pmap_tgt = np.einsum("ij,ij->i", mc, vc) / (
            np.linalg.norm(mc, axis=1) * np.linalg.norm(vc, axis=1) + 1e-12
        )

        vals = {"pmap_tgt": pmap_tgt}
        for key in ("p2_tc", "p3b_tc", "p1_tc", "p2_ps"):
            col = frame.get(key)
            assert col is not None, f"{aid}: missing persisted column {key}"
            vals[key] = np.asarray(col, dtype=np.float64)

        rec = {"arm_id": aid, "kind": kind, "layer": layer, "pos": pos, "rho": {}, "n": {}}
        for key, v in vals.items():
            ok = np.isfinite(v) & np.isfinite(dv)
            assert ok.sum() >= 45, (aid, key, int(ok.sum()))
            rec["rho"][key] = float(stats.spearmanr(v[ok], dv[ok]).statistic)
            rec["n"][key] = int(ok.sum())
        rec["band_enlarged"] = bands[aid]["band_enlarged"]["p975_max_selected"]

        # Band recomputed over the enlarged set PLUS the two candidates added
        # here, from the SAME draws — so the edge shown is exact for every
        # plotted point rather than inherited from a smaller candidate set.
        cols = arm_columns(inputs, arm_by_id[aid])
        named: list[tuple[str, np.ndarray]] = []
        for s in SETTINGS:
            named += [(f"{p}@{s}", cols[s][p]) for p in REMETRIC]
        for p in CARRIED:
            v = frame.get(CARRIED_COL[p])
            if v is not None:
                named.append((f"{p}@carried", np.asarray(v, dtype=np.float64)))
        named += [("pmap_tgt", vals["pmap_tgt"]), ("p2_ps", vals["p2_ps"])]
        M = np.column_stack([v for _n, v in named])
        okb = np.isfinite(dv) & np.isfinite(M).all(axis=1)
        assert okb.sum() >= 45, (aid, int(okb.sum()))
        b_plus = band_for(_rank_z(M[okb]), _rank_z(dv[okb]), N_PERM, SEED)
        rec["band_plus_new"] = b_plus["p975_max_selected"]
        rec["k_candidates_plus_new"] = b_plus["k_candidates"]
        records[label] = rec

    # sanity: the three carried columns must reproduce the committed race values
    for label, aid, *_ in ARMS:
        banked = json.loads((RACE / f"arm_{aid}.json").read_text())
        for key in ("p2_tc", "p3b_tc", "p1_tc"):
            b = banked.get("observed", {}).get(key)
            if b is not None:
                d = abs(b - records[label]["rho"][key])
                assert d < 5e-3, (aid, key, b, records[label]["rho"][key])

    set_paper_style("iclr")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ys = list(range(len(ARMS)))[::-1]
    off = [0.30, 0.15, 0.0, -0.15, -0.30]

    for y, (label, *_rest) in zip(ys, ARMS, strict=True):
        rec = records[label]
        band = rec["band_plus_new"]
        ax.plot([band, band], [y - 0.40, y + 0.40], "-", color="#666666", lw=1.0, zorder=2)
        for (key, _lab, col), dy in zip(PREDS, off, strict=True):
            ax.plot(
                [rec["rho"][key]], [y + dy], "o", ms=4.6, color=col, markeredgewidth=0, zorder=4
            )

    ax.axvline(0.0, color="#BBBBBB", lw=0.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, *_ in ARMS])
    ax.set_ylim(-0.65, len(ARMS) - 0.35)
    ax.set_xlabel("within-arm Spearman $\\rho$ against leakage (positive-only arms)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, ms=4.6, label=lab) for _, lab, c in PREDS
    ]
    handles.append(plt.Line2D([], [], color="#666666", lw=1.0, label="permutation band (97.5%)"))
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        fontsize=6.2,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c5_po_target_answer", dir=args.out_dir)
    plt.close(fig)
    png = paths["png"]
    assert png.exists() and png.stat().st_size > 0, paths

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"arms": records, "preds": [p[0] for p in PREDS]}, indent=2)
    )

    print(f"[fig] wrote {png} ({png.stat().st_size} bytes)")
    for label, *_ in ARMS:
        rec = records[label]
        s = "  ".join(f"{k}={rec['rho'][k]:+.3f}" for k, _l, _c in PREDS)
        print(f"  {label:16s} {s}  band={rec['band_enlarged']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
