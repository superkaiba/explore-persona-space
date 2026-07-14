"""Issue #779 inline free-analysis — remake the answer-summary predictability plot
with CROSS-LAYER context inputs added.

The existing ffc_summary_best_by_layer plot shows, per answer summary, the best
held-out R2 across layers using a SINGLE-layer context input (last-token / mean).
This adds two cross-layer context-input series — cross-layer-mean of the last-token
vector, and cross-layer-mean of the mean-over-prompt vector ("mean of mean") —
so each answer summary is scored by 4 context inputs. Single-layer series are
READ from the committed layer_target_heatmap.json (no recompute); the two
cross-layer series are computed here (5-fold CV, same protocol).

0-GPU: reuses pass_b (context stack) + the D3 answer-summary store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy / the heavy issue779_* siblings freeze their pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue779_fitter_fair_comparison as F  # noqa: E402
import numpy as np  # noqa: E402

OUT_JSON = F.DEFAULT_OUT_DIR / "summary_crosslayer_input.json"


def _best_over_layers_single(heatmap_json, variant):
    """From the committed D3 heatmap, best-across-layers R2 per target for one input."""
    res = json.loads(Path(heatmap_json).read_text())
    node = res["inputs"][variant]
    out = {}
    for _lk, e in node.items():
        for tk, m in e.items():
            r2 = m["r2_mean"]
            if tk not in out or r2 > out[tk]:
                out[tk] = r2
    return out, res["targets"], res["target_labels"]


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=F.SPLIT_SEED)
    ap.add_argument("--n-folds", type=int, default=F.CV_FOLDS)
    ap.add_argument("--heatmap", type=Path, default=F.DEFAULT_OUT_DIR / "layer_target_heatmap.json")
    ap.add_argument("--capture-dir", type=Path, default=F.AS.DEFAULT_CAPTURE_DIR)
    ap.add_argument("--p2-dir", type=Path, default=F.AS2.DEFAULT_P2_DIR)
    args = ap.parse_args()

    dev = F._dev(args.device)
    bundle = F.load_pass_b()
    layers = list(bundle["layers"])
    tkeys = list(F.D3_TARGETS)

    # answer targets (masked) from the D3 store
    S2, s2_idx, xl_targets, mask = F.load_d3_targets(args.capture_dir, args.p2_dir, layers, tkeys)

    # cross-layer context inputs (both bases), masked to the same rows
    cxl = bundle["cx_last"].to(dtype=torch.float32).numpy().mean(axis=1)[mask]
    cxm = bundle["cx_mean"].to(dtype=torch.float32).numpy().mean(axis=1)[mask]
    xinputs = {"last_xmean": cxl, "mean_xmean": cxm}

    # single-layer series READ from the committed heatmap (no recompute)
    single = {}
    targets_order = None
    labels = None
    for variant, key in (("last", "last_single"), ("mean", "mean_single")):
        best, targets_order, labels = _best_over_layers_single(args.heatmap, variant)
        single[key] = best

    # cross-layer series: for each answer layer, fit cross-layer-context -> summaries@layer;
    # keep best-over-answer-layers per (input, target).
    xres = {k: {t: -np.inf for t in tkeys} for k in xinputs}
    for li in layers:
        targets_L = {}
        if "v_x" in tkeys:
            targets_L["v_x"] = F.target_vx(bundle, li)[mask]
        for si, s in enumerate(F.AS2.P2_SUMMARIES):
            if s in tkeys:
                targets_L[s] = S2[:, si, s2_idx[li], :][mask].astype(np.float32)
        for k, v in xl_targets.items():  # xl targets are layer-agnostic (already masked)
            targets_L[k] = v
        for ikey, X in xinputs.items():
            rec = F.gram_cv_recon(X, targets_L, args.n_folds, args.seed, dev)
            for t, m in rec.items():
                if m["r2_mean"] > xres[ikey][t]:
                    xres[ikey][t] = m["r2_mean"]
        F.logger.info("[xlayer-input] answer layer L%d done", li)

    series = {
        "last_single": single["last_single"],
        "mean_single": single["mean_single"],
        "last_xmean": {
            t: (None if xres["last_xmean"][t] == -np.inf else xres["last_xmean"][t]) for t in tkeys
        },
        "mean_xmean": {
            t: (None if xres["mean_xmean"][t] == -np.inf else xres["mean_xmean"][t]) for t in tkeys
        },
    }
    out = {
        "targets": targets_order or tkeys,
        "target_labels": labels or {},
        "series": series,
        "series_labels": {
            "last_single": "last-token, best single layer",
            "mean_single": "mean-context, best single layer",
            "last_xmean": "last-token, cross-layer mean",
            "mean_xmean": "mean-context, cross-layer mean (mean-of-mean)",
        },
        "note": (
            "Per answer summary, best-across-layers held-out R2 from 4 context inputs. "
            "single series read from committed layer_target_heatmap.json; cross-layer "
            "(*_xmean) computed here (5-fold CV, best over answer layers)."
        ),
        "metadata": F._base_metadata("summary_crosslayer_input", args, {}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    F.C.write_json_atomic(args.out, out)
    print(f"wrote {args.out}")
    for t in sorted(tkeys, key=lambda tt: -max((series[s][tt] or -9) for s in series)):
        vals = " ".join(
            f"{s.split('_')[0]}{'X' if 'xmean' in s else 'S'}={series[s][t]:.3f}"
            for s in series
            if series[s][t] is not None
        )
        print(f"  {labels.get(t, t) if labels else t:42s} {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
