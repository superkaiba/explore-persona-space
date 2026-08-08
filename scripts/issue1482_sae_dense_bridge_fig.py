"""Issue #1482 inline round — figure for the SAE<->dense bridge.

Grouped bars: one group per mapping, one bar per SPACE (feature-space R2 where the
mapping has a feature-space target at all, dense-space R2 for every mapping), with
the two SAE reconstruction ceilings drawn as horizontal reference lines in the
dense-space colour. The whole point of the panel is that the two bar colours are
NOT comparable to each other — only the dense-space bars are comparable ACROSS
groups, and only against the ceiling lines.

Reads eval_results/issue_1482/sae_dense_bridge/sae_dense_bridge.json; writes to
figures/issue_1482/sae_dense_bridge/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

SHORT = {
    "sae_ctx__to__dense  [NEW]": "SAE ctx\n-> dense\n[NEW]",
    "sae_ctx__to__sae  (decoded)": "SAE ctx\n-> SAE",
    "dense_ctx__to__sae  (decoded)": "dense ctx\n-> SAE",
    "dense_ctx__to__dense  (matched n=120k)": "dense ctx\n-> dense\n(n=120k)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_1482/sae_dense_bridge/sae_dense_bridge.json",
    )
    ap.add_argument("--figdir", type=Path, default=PROJECT_ROOT / "figures/issue_1482")
    args = ap.parse_args()
    doc = json.loads(args.json.read_text())
    cells, ceil = doc["cells"], doc["ceilings"]

    keys = [k for k in SHORT if k in cells]
    labels = [SHORT[k] for k in keys]
    feat = [cells[k]["feature_space_r2"] for k in keys]
    dens = [cells[k]["dense_space_r2"] for k in keys]

    set_paper_style()
    pal = paper_palette(2)
    c_feat, c_dense = pal[0], pal[1]
    x = np.arange(len(keys), dtype=float)
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    fx = [v if v is not None else np.nan for v in feat]
    dx = [v if v is not None else np.nan for v in dens]
    ax.bar(x - w / 2, fx, w, color=c_feat, label="feature-space $R^2$ (SAE targets)")
    ax.bar(x + w / 2, dx, w, color=c_dense, label="dense-space $R^2$ (residual stream)")

    for xi, v in zip(x, fx, strict=True):
        if np.isnan(v):
            ax.text(
                xi - w / 2,
                0.012,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="0.35",
            )
        else:
            ax.text(xi - w / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x, dx, strict=True):
        ax.text(xi + w / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Reference lines carry their labels in the LEGEND, not as inline text: inline
    # annotations collide with the bar-value labels and with each other whenever two
    # reference levels land close together (they do — the restricted ceiling and the
    # banked dense map sit ~0.01 apart).
    n_retained = int(doc["design"].get("restriction", {}).get("f_out", 16384))
    cr = ceil["sae_reconstruction_restricted_f_out"]["dense_space_r2"]
    cf = ceil["sae_reconstruction_full_dictionary"]["dense_space_r2"]
    ax.axhline(
        cf,
        color=c_dense,
        ls=":",
        lw=1.5,
        label=f"SAE reconstruction ceiling, full 131,072-atom dict ({cf:.3f})",
    )
    ax.axhline(
        cr,
        color=c_dense,
        ls="--",
        lw=1.5,
        label=f"SAE reconstruction ceiling, retained {n_retained:,} cols ({cr:.3f})",
    )
    banked = doc.get("banked_reference", {}).get("dense_ctx__to__dense__ridge__full_pool")
    if banked:
        ax.axhline(
            banked["dense_space_r2"],
            color="0.45",
            ls="-.",
            lw=1.2,
            label=f"banked dense→dense ridge, n=943k ({banked['dense_space_r2']:.3f})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("held-out pooled $R^2$")
    top = max([v for v in dx + fx if not np.isnan(v)] + [cf])
    ax.set_ylim(0, top + 0.42)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=1, handlelength=2.4)
    ax.set_title(
        "Feature-space and dense-space $R^2$ are different currencies\n"
        f"Qwen-2.5-7B L{doc['design']['layer']}, mean pooling, "
        f"{doc['design']['splits']['n_holdout']:,}-context holdout; "
        f"all maps fit on {doc['design']['splits']['n_train']:,} rows",
        fontsize=10,
    )
    fig.tight_layout()
    paths = savefig_paper(fig, "sae_dense_bridge/two_space_r2", dir=args.figdir)
    print({k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.exit(0)
