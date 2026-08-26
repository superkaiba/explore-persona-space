"""#1979 — positive-only arms: the requested six similarity reads vs the leakage DV.

Answers a direct question about which similarity actually tracks leakage on the
positive-only arms, by putting all of these on ONE axis per arm:

  context similarity
    p1_ps   cos(C[i], C[s])      source context vs target context
    p1_tc   cos(C[i], A_ctx)     target context vs the trained-on context centroid

  real (base-model) answer vectors
    p2_ps   cos(V[i], V[s])      source answers vs target answers
    p2_tc   cos(V[i], A_ans)     target answers vs the trained-on answers

  mapped answer vectors (M = M0 . context vector, the base context->answer map)
    p3a_ps  cos(M[i], M[s])      mapped source answers vs mapped target answers
    p3b_tc  cos(M[i], A_ans)     mapped target answers vs the trained-on answers

with i = destination prefix, s = the arm's OWN source (trained) prefix, and
A_ctx / A_ans the training-row centroids. The `_ps` family anchors on the arm's
source prefix; the `_tc` family anchors on the training rows.

The asking phrasing named only "context vector similarity" without its two
endpoints, so BOTH context readings are plotted rather than one being guessed.

Five of the six columns are read from the banked per-prefix frames; only p3a_ps
is computed here (it was introduced by `issue1979_po_source_target_fig.py`).
Metric is the committed panel-centered cosine.

Structural note: both `_ps` reads are exactly 1.0 at the source prefix, which is
itself a panel member. Every rho is reported over all 50 prefixes AND with the
source prefix dropped; the JSON sidecar carries both.

Band: recomputed per arm from the same 20,000 draws over the enlarged candidate
set PLUS the three source-anchored candidates, so the drawn edge is exact for
every plotted point rather than inherited from a smaller set.
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
OUT_JSON = REPO_ROOT / "eval_results/issue_1979/whiten_csls/po_five_predictors.json"
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
# (frame key, legend label, colour) — plotted top to bottom within each arm.
PREDS = [
    ("p1_ps", "context: source vs target", "#8172B2"),
    ("p1_tc", "context: target vs trained contexts", "#CCB974"),
    ("p2_ps", "real answers: source vs target", "#DD8452"),
    ("p3a_ps", "mapped answers: source vs target", "#55A868"),
    ("p2_tc", "real answers: target vs trained answers", "#4C72B0"),
    ("p3b_tc", "mapped answers: target vs trained answers", "#C44E52"),
]
# columns read straight from the banked per-prefix frame
FROM_FRAME = ("p1_ps", "p1_tc", "p2_ps", "p2_tc", "p3b_tc")
# source-anchored candidates absent from the enlarged band set, added to it here
BAND_EXTRA = ("p1_ps", "p2_ps", "p3a_ps")


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
    ap.add_argument("--style", choices=("bar", "dot"), default="bar")
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

        vals: dict[str, np.ndarray] = {}
        for key in FROM_FRAME:
            col = frame.get(key)
            assert col is not None, f"{aid}: missing persisted column {key}"
            vals[key] = np.asarray(col, dtype=np.float64)

        # source-prefix index: the unique exact-1.0 entry of the persisted p2_ps,
        # cross-checked against the frozen panel ordering.
        s_ix = int(np.argmax(vals["p2_ps"]))
        assert vals["p2_ps"][s_ix] > 0.9999, (aid, vals["p2_ps"][s_ix])
        assert np.sum(vals["p2_ps"] > 0.9999) == 1, aid
        want = OWN_PREFIX_BY_CTX[arm["ctx_key"]]
        assert panel_ids[s_ix] == want, (aid, s_ix, panel_ids[s_ix], want)
        # p1_ps must peak at the same prefix — it is the same anchor, context side
        assert int(np.argmax(vals["p1_ps"])) == s_ix, (aid, int(np.argmax(vals["p1_ps"])), s_ix)

        # mapped source answers vs mapped target answers
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
            "dv": DV_BY_KIND[kind],
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

        # Band over the enlarged set PLUS the source-anchored candidates, same draws.
        cols = arm_columns(inputs, arm)
        named: list[tuple[str, np.ndarray]] = []
        for s in SETTINGS:
            named += [(f"{p}@{s}", cols[s][p]) for p in REMETRIC]
        for p in CARRIED:
            v = frame.get(CARRIED_COL[p])
            if v is not None:
                named.append((f"{p}@carried", np.asarray(v, dtype=np.float64)))
        named += [(k, vals[k]) for k in BAND_EXTRA]
        M = np.column_stack([v for _n, v in named])
        okb = np.isfinite(dv) & np.isfinite(M).all(axis=1)
        assert okb.sum() >= 45, (aid, int(okb.sum()))
        b = band_for(_rank_z(M[okb]), _rank_z(dv[okb]), N_PERM, SEED)
        rec["band"] = b["p975_max_selected"]
        rec["k_candidates"] = b["k_candidates"]
        rec["band_enlarged_committed"] = bands[aid]["band_enlarged"]["p975_max_selected"]
        records[label] = rec

    # sanity: the carried columns must reproduce the committed race values
    for label, aid, *_ in ARMS:
        banked = json.loads((RACE / f"arm_{aid}.json").read_text())
        for key in ("p1_tc", "p2_tc", "p3b_tc"):
            b = banked.get("observed", {}).get(key)
            if b is not None:
                d = abs(b - records[label]["rho"][key])
                assert d < 5e-3, (aid, key, b, records[label]["rho"][key])

    set_paper_style("iclr")
    if args.style == "bar":
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        xs = np.arange(len(ARMS), dtype=float)
        width = 0.82 / len(PREDS)
        off = (np.arange(len(PREDS)) - (len(PREDS) - 1) / 2.0) * width

        for (key, lab, col), dx in zip(PREDS, off, strict=True):
            ax.bar(
                xs + dx,
                [records[label]["rho"][key] for label, *_ in ARMS],
                width * 0.92,
                color=col,
                label=lab,
                zorder=3,
            )
        for x, (label, *_rest) in zip(xs, ARMS, strict=True):
            band = records[label]["band"]
            ax.plot([x - 0.46, x + 0.46], [band, band], "-", color="#333333", lw=1.2, zorder=5)

        ax.axhline(0.0, color="#888888", lw=0.8, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for lab, *_ in ARMS])
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_ylabel("within-arm Spearman $\\rho$ against leakage")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=lab) for _, lab, c in PREDS]
        handles.append(
            plt.Line2D([], [], color="#333333", lw=1.2, label="permutation band (97.5%)")
        )
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
            fontsize=6.2,
            handletextpad=0.5,
            columnspacing=1.2,
        )
    else:
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        ys = list(range(len(ARMS)))[::-1]
        off = np.linspace(0.32, -0.32, len(PREDS))

        for y, (label, *_rest) in zip(ys, ARMS, strict=True):
            rec = records[label]
            band = rec["band"]
            ax.plot([band, band], [y - 0.44, y + 0.44], "-", color="#666666", lw=1.0, zorder=2)
            for (key, _lab, col), dy in zip(PREDS, off, strict=True):
                ax.plot(
                    [rec["rho"][key]], [y + dy], "o", ms=4.8, color=col, markeredgewidth=0, zorder=4
                )

        ax.axvline(0.0, color="#BBBBBB", lw=0.6, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for lab, *_ in ARMS])
        ax.set_ylim(-0.7, len(ARMS) - 0.3)
        ax.set_xlabel("within-arm Spearman $\\rho$ against leakage (positive-only arms)")
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=c, ms=4.8, label=lab) for _, lab, c in PREDS
        ]
        handles.append(
            plt.Line2D([], [], color="#666666", lw=1.0, label="permutation band (97.5%)")
        )
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=2,
            fontsize=6.2,
            handletextpad=0.4,
            columnspacing=1.2,
        )
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c5_po_five_predictors", dir=args.out_dir)
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
        print(
            f"\n  {label}  (dv={rec['dv']}, source prefix = {rec['source_prefix']}, "
            f"band {rec['band']:+.3f}, K={rec['k_candidates']})"
        )
        for key, lab, _c in PREDS:
            flag = "CLEARS" if rec["rho"][key] > rec["band"] else "below "
            print(
                f"      {key:8s} {rec['rho'][key]:+.3f} {flag}"
                f"   drop-source {rec['rho_drop_source'][key]:+.3f}   {lab}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
