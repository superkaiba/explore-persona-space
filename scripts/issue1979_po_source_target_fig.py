"""#1979 — positive-only arms: source-anchored vs training-centroid-anchored reads.

Completes the 2x2 of {real answers, mapped answers} x {training-row-centroid
anchor, the arm's OWN SOURCE (trained) prefix as anchor}:

                        anchor = trained-on rows      anchor = source prefix
  real answers          p2_tc                         p2_ps
  mapped answers        p3a_tc                        p3a_ps   <- NEW here

where for destination prefix i and the arm's source prefix s:

  p2_tc   cos(V[i], A_ans)            V = base answer vector at a prefix
  p2_ps   cos(V[i], V[s])             "real source answers vs real target answers"
  p3a_tc  cos(M[i], M0(anchor ctx))   M = M0 . context vector at a prefix
  p3a_ps  cos(M[i], M[s])             "mapped source answers vs mapped target answers"

p3b_tc (cos(M[i], A_ans), the committed change champion) is carried as the
reference point. All cosines use the committed panel-centered convention, so
p2_ps / p3a_ps are cosines between centered per-prefix vectors.

Structural note carried into the output: both source-anchored reads are exactly
1.0 at the source prefix itself (it is a member of the 50-prefix panel), so each
rho is reported BOTH over all 50 prefixes and with the source prefix dropped.
The source index is recovered from the persisted p2_ps column (its unique
exact-1.0 entry) and cross-checked against the frozen prefix panel ordering.

Band is recomputed from the same 20,000 draws over the enlarged candidate set
plus the source-anchored candidates, so the drawn edge is exact for every
plotted point.
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
from issue1979_race import OWN_PREFIX_BY_CTX, _center  # noqa: E402
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
PANEL = REPO_ROOT / "eval_results/issue_1979/config/prefix_panel.json"
OUT_JSON = REPO_ROOT / "eval_results/issue_1979/whiten_csls/po_source_target.json"
FIG_DIR = REPO_ROOT / "figures/issue_1979"
TENSORS = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1979_whitencsls/battery/ingredient_tensors.pt"
)

ARMS = [
    ("casual writing", "cas-pers-po-lr1e5-s42", "content", 19, "last_prompt"),
    ("impoliteness", "imp-pers-po-lr1e5-s42", "content", 19, "last_prompt"),
    ("marker token", "mk-pers-po-lr5e6-s42", "marker", 25, "last_prompt"),
]
PREDS = [
    ("p2_ps", "REAL source vs target answers", "#DD8452"),
    ("p3a_ps", "MAPPED source vs target answers  (new)", "#55A868"),
    ("p2_tc", "real target vs trained-on answers", "#4C72B0"),
    ("p3a_tc", "mapped target vs mapped trained anchor", "#64B5CD"),
    ("p3b_tc", "mapped target vs trained-on answers  (champion)", "#C44E52"),
]


def _rho(v: np.ndarray, dv: np.ndarray, drop: int | None = None) -> tuple[float, int]:
    """Spearman rho of v against dv over finite rows, optionally dropping one index."""
    ok = np.isfinite(v) & np.isfinite(dv)
    if drop is not None:
        ok = ok.copy()
        ok[drop] = False
    assert ok.sum() >= 45, int(ok.sum())
    return float(stats.spearmanr(v[ok], dv[ok]).statistic), int(ok.sum())


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
    members = json.loads(PANEL.read_text())["members"]
    panel_ids = [m["prefix_id"] if isinstance(m, dict) else m for m in members]

    records: dict[str, dict] = {}
    for label, aid, kind, layer, pos in ARMS:
        arm = arm_by_id[aid]
        frame = json.loads((RACE / f"frame_{aid}.json").read_text())["frame"]
        dv = np.asarray(frame[DV_BY_KIND[kind]], dtype=np.float64)

        vals = {}
        for key in ("p2_tc", "p3a_tc", "p3b_tc", "p2_ps"):
            col = frame.get(key)
            assert col is not None, f"{aid}: missing persisted column {key}"
            vals[key] = np.asarray(col, dtype=np.float64)

        # source-prefix index: the unique exact-1.0 entry of the persisted p2_ps
        s_ix = int(np.argmax(vals["p2_ps"]))
        assert vals["p2_ps"][s_ix] > 0.9999, (aid, vals["p2_ps"][s_ix])
        assert np.sum(vals["p2_ps"] > 0.9999) == 1, aid
        # cross-check against the frozen panel ordering
        want = OWN_PREFIX_BY_CTX[arm["ctx_key"]]
        assert panel_ids[s_ix] == want, (aid, s_ix, panel_ids[s_ix], want)

        # NEW: mapped source answers vs mapped target answers
        m0 = np.asarray(tens[f"m0pred/{kind}/L{layer}/{pos}"].double().numpy())
        mc, _ = _center(m0)
        mn = mc / (np.linalg.norm(mc, axis=1, keepdims=True) + 1e-12)
        vals["p3a_ps"] = mn @ mn[s_ix]
        assert vals["p3a_ps"][s_ix] > 0.9999, (aid, vals["p3a_ps"][s_ix])

        rec = {
            "arm_id": aid,
            "kind": kind,
            "layer": layer,
            "pos": pos,
            "source_prefix": panel_ids[s_ix],
            "source_ix": s_ix,
            "rho": {},
            "rho_drop_source": {},
            "n": {},
        }
        for key, _lab, _c in PREDS:
            r_all, n_all = _rho(vals[key], dv)
            r_drop, _ = _rho(vals[key], dv, drop=s_ix)
            rec["rho"][key] = r_all
            rec["rho_drop_source"][key] = r_drop
            rec["n"][key] = n_all

        # band over enlarged set + the two source-anchored candidates, same draws
        cols = arm_columns(inputs, arm)
        named: list[tuple[str, np.ndarray]] = []
        for s in SETTINGS:
            named += [(f"{p}@{s}", cols[s][p]) for p in REMETRIC]
        for p in CARRIED:
            v = frame.get(CARRIED_COL[p])
            if v is not None:
                named.append((f"{p}@carried", np.asarray(v, dtype=np.float64)))
        named += [("p2_ps", vals["p2_ps"]), ("p3a_ps", vals["p3a_ps"])]
        M = np.column_stack([v for _n, v in named])
        okb = np.isfinite(dv) & np.isfinite(M).all(axis=1)
        b = band_for(_rank_z(M[okb]), _rank_z(dv[okb]), N_PERM, SEED)
        rec["band"] = b["p975_max_selected"]
        rec["k_candidates"] = b["k_candidates"]
        records[label] = rec

    set_paper_style("iclr")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ys = list(range(len(ARMS)))[::-1]
    off = [0.30, 0.15, 0.0, -0.15, -0.30]

    for y, (label, *_rest) in zip(ys, ARMS, strict=True):
        rec = records[label]
        band = rec["band"]
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
        fontsize=6.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c5_po_source_target", dir=args.out_dir)
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
        print(f"  {label}  (source prefix = {rec['source_prefix']}, band {rec['band']:+.3f})")
        for key, lab, _c in PREDS:
            flag = "CLEARS" if rec["rho"][key] > rec["band"] else "below "
            print(
                f"      {key:8s} {rec['rho'][key]:+.3f} {flag}"
                f"   drop-source {rec['rho_drop_source'][key]:+.3f}   {lab}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
