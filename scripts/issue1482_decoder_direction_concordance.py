#!/usr/bin/env python3
"""Issue #1482: re-run the SAE feature-property selection with the
DECODER-DIRECTION dependent variable, against the published activation one.

The reviewer asked to repeat the analysis "on decoder directions rather than
feature activations". This driver swaps ONLY the dependent variable and reuses
the published machinery untouched: the same 42 candidate properties, the same
coarsened-exact-matching, the same round-by-round selection with the same
stop rules and the same sibling-retirement cut. Everything that could otherwise
explain a difference is held fixed, so a difference is attributable to the DV.

  published DV   per-feature held-out R^2 of predicting the feature's MEAN
                 ACTIVATION over the answer (encoder + BatchTopK gate; mostly
                 zeros for a rare feature)
  swapped DV     per-direction held-out R^2 of predicting d_f^T v along the unit
                 decoder column (dense; scored on all 20,000 held-out answers
                 regardless of how rarely the feature fires)

Both are one number per feature on the same dictionary, so the property table
joins by feature id with no other change.

Reads the swapped DV from issue1482_decoder_direction_r2.py. Writes its own
figures and sidecar to a separate directory, so the published outputs are never
overwritten.

0 GPU, no pod, minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_concordance_stepwise as SW  # noqa: E402
import issue1482_concordance_writeup_figs as WF  # noqa: E402

PUBLISHED_META = "figures/issue_1482/concordance/writeup_stepwise.meta.json"
DEC_R2 = "eval_results/issue_1482/decoder_direction/decoder_direction_r2_fullwidth.npy"
ACT_R2 = "data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy"
# The six numbers the paper's SAE paragraph actually quotes.
HEADLINE = [
    "Mean activation (over all answers)",
    "Speaker: identity / disposition",
    "Logit footprint: suppressing",
    "Logit footprint: promoting",
    "Content type: topic",
    "Interpretable (autointerp)",
]


def _log(m: str) -> None:
    print(f"[dec-conc {datetime.now(UTC).strftime('%H:%M:%S')}] {m}", flush=True)


def round0(meta: dict) -> dict[str, float]:
    return {s["name"]: s["c"] - 0.5 for s in meta["rounds"][0]["scores"]}


def selected(meta: dict) -> dict[str, tuple[int, float]]:
    return {
        r["winner"]: (r["round"], r["winner_c"] - 0.5) for r in meta["rounds"] if r.get("winner")
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="figures/issue_1482/concordance_decoder_direction")
    ap.add_argument(
        "--restrict-to-published-universe",
        action="store_true",
        help=(
            "score only the features the published analysis scored. A decoder direction has\n"
            "a defined R^2 even when its feature never fires on a held-out answer, so the\n"
            "swapped DV covers a WIDER universe (128,450 vs 120,716). Masking the extras to\n"
            "NaN makes the two runs score an identical feature set, removing universe size\n"
            "as an explanation for any difference."
        ),
    )
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    dec_path = PROJECT_ROOT / DEC_R2
    if not dec_path.exists():
        raise SystemExit(f"missing {dec_path}; run issue1482_decoder_direction_r2.py first")

    # Rank agreement between the two DVs, before any property enters.
    dec, act = np.load(dec_path), np.load(PROJECT_ROOT / ACT_R2)
    both = np.isfinite(dec) & np.isfinite(act)
    rho = float(spearmanr(dec[both], act[both]).statistic)
    _log(f"DV rank agreement over {int(both.sum()):,} features: spearman {rho:.4f}")

    if args.restrict_to_published_universe:
        masked = np.asarray(np.load(dec_path), dtype=np.float64).copy()
        masked[~np.isfinite(act)] = np.nan
        dec_path = PROJECT_ROOT / DEC_R2.replace(".npy", "_common_universe.npy")
        np.save(dec_path, masked)
        _log(
            f"restricted to the published universe: "
            f"{int(np.isfinite(masked).sum()):,} features scored"
        )

    # Swap the DV and redirect outputs; the property set and rules are untouched.
    WF.TARGET_R2 = dec_path
    WF.OUT = out
    SW.OUT = out
    _log("running the published stepwise selection on the decoder-direction DV ...")
    SW.main()

    new = json.loads((out / "writeup_stepwise.meta.json").read_text())
    pub = json.loads((PROJECT_ROOT / PUBLISHED_META).read_text())
    n_r0, p_r0 = round0(new), round0(pub)
    n_sel, p_sel = selected(new), selected(pub)

    rows = []
    for name in HEADLINE:
        rows.append(
            {
                "property": name,
                "published_round0": p_r0.get(name),
                "published_selected": p_sel.get(name, (None, None))[1],
                "published_round": p_sel.get(name, (None, None))[0],
                "decoder_round0": n_r0.get(name),
                "decoder_selected": n_sel.get(name, (None, None))[1],
                "decoder_round": n_sel.get(name, (None, None))[0],
            }
        )
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "dv_rank_agreement_spearman": rho,
        "n_features_scored_decoder": int(new["n_rows"]),
        "n_features_scored_published": int(pub["n_rows"]),
        "selection_order_decoder": [r["winner"] for r in new["rounds"] if r.get("winner")],
        "selection_order_published": [r["winner"] for r in pub["rounds"] if r.get("winner")],
        "headline": rows,
    }
    res = PROJECT_ROOT / "eval_results/issue_1482/decoder_direction"
    res.mkdir(parents=True, exist_ok=True)
    (res / "concordance_comparison.json").write_text(json.dumps(summary, indent=1))

    _log("")
    _log(f"{'property':<36}{'pub r0':>9}{'pub sel':>9}{'dec r0':>9}{'dec sel':>9}")
    for r in rows:
        f = lambda v: f"{v:+.3f}" if isinstance(v, float) else "   --"  # noqa: E731
        _log(
            f"{r['property']:<36}{f(r['published_round0']):>9}{f(r['published_selected']):>9}"
            f"{f(r['decoder_round0']):>9}{f(r['decoder_selected']):>9}"
        )
    _log(f"wrote {res / 'concordance_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
