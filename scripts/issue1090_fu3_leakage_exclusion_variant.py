#!/usr/bin/env python
"""#1090 fu3 — leakage-contrast variant excluding the trained-negative contexts.

Two of the five held-out bystander-panel contexts (`neg_sp_police`,
`neg_sp_ph4`) are the CONTRASTIVE arms' training negatives (5-member default
panel) while being genuinely unseen for the positive-only arms — a structural
asymmetry across regimes (interpretation-critic fu3-r1 item 5). This variant
recomputes every pair's held-out leakage mean over ONLY the three
non-source, non-trained-negative panel contexts, plus the same matched-install
/ all-pair contrasts (batched bootstrap, seed 42) and the all-pair correlation
with and without the two persona-sycophancy pairs — so the body can state
whether the asymmetric panel changes the cross-regime read.

Reads the (re-judged) ``fu3_cell_evals/`` checkpoints; writes
``fu3_leakage_contrast_exclusion_variant.json``. Zero API calls.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu3_aggregate as fu3agg  # noqa: E402

TRAINED_NEG_CONTEXTS = {"neg_sp_police", "neg_sp_ph4"}


def _held_out_mean(rec: dict, *, exclude: set[str]) -> tuple[float, int]:
    deltas = [
        b["leak_delta"]
        for b in rec["bystanders"]
        if not b["is_source_context"] and b["context_id"] not in exclude
    ]
    return float(np.mean(deltas)), len(deltas)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="#1090 fu3 leakage exclusion variant")
    ap.add_argument("--out", default="eval_results/issue_1090/fu3")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    out_dir = Path(args.out)

    evals: dict[str, dict] = {}
    for p in sorted((out_dir / "fu3_cell_evals").glob("*.json")):
        rec = json.loads(p.read_text())
        evals[rec["cell_id"]] = rec

    pairs = []
    for cid, e in sorted(evals.items()):
        if not cid.endswith("-con"):
            continue
        p = evals.get(cid[: -len("-con")] + "-pos")
        if p is None:
            continue
        con_mean, n_ctx = _held_out_mean(e, exclude=TRAINED_NEG_CONTEXTS)
        pos_mean, _ = _held_out_mean(p, exclude=TRAINED_NEG_CONTEXTS)
        pairs.append(
            {
                "pair": cid[: -len("-con")],
                "behavior": e["behavior"],
                "n_contexts_kept": n_ctx,
                "leak_contrastive_excl": con_mean,
                "leak_posonly_excl": pos_mean,
                "delta_con_minus_pos_excl": con_mean - pos_mean,
                "install_matched": bool(e["band_hit"] and p["band_hit"]),
            }
        )

    def _contrast(subset: list[dict], label: str) -> dict:
        deltas = np.array([p["delta_con_minus_pos_excl"] for p in subset], dtype=np.float64)
        con = np.array([p["leak_contrastive_excl"] for p in subset], dtype=np.float64)
        pos = np.array([p["leak_posonly_excl"] for p in subset], dtype=np.float64)
        lo, hi = fu3agg.paired_delta_ci(deltas, n_boot=args.n_boot, seed=args.seed)
        return {
            "label": label,
            "n_pairs": int(deltas.size),
            "mean_delta_con_minus_pos": float(deltas.mean()) if deltas.size else None,
            "delta_ci95": [lo, hi],
            "paired_r": fu3agg.paired_r_with_ci(con, pos, n_boot=args.n_boot, seed=args.seed),
        }

    matched = [p for p in pairs if p["install_matched"]]
    non_persona = [p for p in pairs if not p["pair"].endswith("-pers")]
    payload = {
        "excluded_contexts": sorted(TRAINED_NEG_CONTEXTS),
        "note": (
            "held-out leakage means over the 3 non-source panel contexts that are NOT "
            "the contrastive arms' training negatives; companion to fu3_leakage_contrast.json"
        ),
        "pairs": pairs,
        "contrast_matched_install": _contrast(matched, "matched-install pairs (excl variant)"),
        "contrast_all_pairs": _contrast(pairs, "all pairs (excl variant)"),
        "contrast_non_persona_pairs": _contrast(
            non_persona, "pairs excluding the two persona-sycophancy pairs (excl variant)"
        ),
    }
    fu3agg.run1090._atomic_write_json(
        out_dir / "fu3_leakage_contrast_exclusion_variant.json", payload
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
