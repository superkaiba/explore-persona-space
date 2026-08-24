"""Issue #2479 round-3 diagnostics — recomputed from committed artifacts (zero GPU).

Reads only committed artifacts (``eval_results/issue_2479/gradient_verdict.json``
+ ``cjk_audit.json``) and writes ``eval_results/issue_2479/r3_diagnostics.json``.
Pure counting/statistics — no story or answer text is read or persisted.

Computes (interpretation-critique round 2):
  (a) rung-WISE axis orderings: rho(axis, rung R2 / own-map ceiling) with the
      verdict's 10,000-shuffle permutation machinery for EVERY registered
      ladder rung — the round-2 body reported only rung 6, inverting the
      committed pattern (the ordering holds at 8 of 9 rungs and rebounds
      after rotation at rungs 7 and 9);
  (b) equalized-n per-character sensitivity: full-n vs equalized-n recovery
      delta for every character (mort is the one material mover);
  (c) capture-side intrusion as a mediator: per-character CJK intrusion rate
      over the kept story rows the maps are FIT on, joined to the axis and to
      recovery — plus the explicit record that no intrusion-excluded capture
      REFIT exists (the committed CJK recounts are judged-axis-side only);
  (d) retrieval identity-vs-transfer figure data: per-character top-1
      accuracies (transferred rung 4, identity+learned-bias, own-map ceiling,
      chance) under both distances, with the identity baseline's own axis
      orderings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from issue2479_gradient_verdict import _spearman, spearman_perm_read  # noqa: E402

REPO = _HERE.parent
EVAL = REPO / "eval_results/issue_2479"
N_PERM = 10_000
SEED = 0


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=EVAL / "r3_diagnostics.json")
    args = ap.parse_args()

    verdict = json.loads((EVAL / "gradient_verdict.json").read_text())
    audit = json.loads((EVAL / "cjk_audit.json").read_text())
    pc = verdict["per_character"]
    names = sorted(pc)
    axis = np.array([pc[n]["axis_score"] for n in names])
    rec = np.array([pc[n]["recovery_fraction"] for n in names])
    ceil = np.array([pc[n]["ceiling_r2"] for n in names])

    out: dict = {"issue": 2479, "characters": names}

    # --- headline replay (must reproduce the committed verdict exactly) -----
    head = spearman_perm_read(axis, rec, n_perm=N_PERM, seed=SEED, label="headline replay")
    committed = verdict["headline"]
    assert abs(head["rho"] - committed["rho"]) < 1e-12, (head["rho"], committed["rho"])
    assert abs(head["p_add_one"] - committed["p_add_one"]) < 1e-12
    out["headline_replay"] = {"rho": _r(head["rho"]), "p_add_one": _r(head["p_add_one"])}

    # --- (a) rung-wise axis orderings ----------------------------------------
    rungs = list(pc[names[0]]["rung_r2_all"])
    rungwise = {}
    for rg in rungs:
        y = np.array([pc[n]["rung_r2_all"][rg] for n in names]) / ceil
        read = spearman_perm_read(axis, y, n_perm=N_PERM, seed=SEED, label=f"axis -> {rg}")
        rungwise[rg] = {"rho": _r(read["rho"]), "p_add_one": _r(read["p_add_one"])}
    non_rotation = [v["rho"] for k, v in rungwise.items() if k != "6_rotation"]
    out["rungwise_axis_ordering"] = {
        "per_rung": rungwise,
        "non_rotation_rho_min": _r(min(non_rotation)),
        "non_rotation_rho_max": _r(max(non_rotation)),
        "rotation_rho": rungwise["6_rotation"]["rho"],
        "note": "recovery = rung R2 / own-map ceiling; same permutation machinery as the verdict",
    }

    # --- (b) equalized-n per-character sensitivity ---------------------------
    eq = verdict["equalized_n"]["companions"]["_rows1028"]
    deltas = {
        n: _r(pc[n]["recovery_fraction"] - eq["values"][eq["characters"].index(n)]) for n in names
    }
    worst = max(deltas, key=lambda n: abs(deltas[n]))
    ranked = sorted(deltas, key=lambda n: -abs(deltas[n]))
    out["equalized_n_sensitivity"] = {
        "per_character_full_minus_equalized": deltas,
        "largest": {
            "character": worst,
            "full_n_recovery": _r(pc[worst]["recovery_fraction"]),
            "equalized_recovery": _r(eq["values"][eq["characters"].index(worst)]),
            "delta": deltas[worst],
        },
        "second_largest_abs_delta": {ranked[1]: deltas[ranked[1]]},
    }

    # --- (c) capture-side intrusion as a mediator -----------------------------
    cap = {n: audit["per_character"][n]["capture_substrate"] for n in names}
    rate = np.array([cap[n]["intruded"] / cap[n]["total"] for n in names])
    out["capture_intrusion_mediator"] = {
        "per_character_rate": {n: _r(cap[n]["intruded"] / cap[n]["total"]) for n in names},
        "rate_min": _r(rate.min()),
        "rate_max": _r(rate.max()),
        "rho_rate_axis": _r(_spearman(rate, axis)),
        "rho_rate_recovery": _r(_spearman(rate, rec)),
        "intrusion_excluded_capture_refit_exists": False,
        "note": (
            "rates over the kept story rows the maps are fit on (cjk_audit.json "
            "capture_substrate); the committed CJK exclusion recounts cover the judged "
            "axis side only — no map was refit after excluding intruded capture rows"
        ),
    }

    # --- (d) retrieval identity-vs-transfer figure data -----------------------
    retr = {}
    for n in names:
        r = pc[n]
        retr[n] = {
            "axis_score": _r(r["axis_score"]),
            "acc1_rung4": _r(r["acc1_rung4"]),
            "acc1_identity_bias": _r(r["acc1_identity_bias"]),
            "acc1_rung4_cosine": _r(r["acc1_rung4_cosine"]),
            "acc1_identity_bias_cosine": _r(r["acc1_identity_bias_cosine"]),
            "acc1_ceiling": _r(r["acc1_ceiling"]),
            "acc1_chance": _r(r["acc1_chance"]),
        }
    idb = np.array([pc[n]["acc1_identity_bias"] for n in names])
    idb_frac = idb / np.array([pc[n]["acc1_ceiling"] for n in names])
    out["retrieval_identity_vs_transfer"] = {
        "per_character": retr,
        "euclidean_identity_wins": int(
            sum(pc[n]["acc1_identity_bias"] > pc[n]["acc1_rung4"] for n in names)
        ),
        "cosine_identity_wins": int(
            sum(pc[n]["acc1_identity_bias_cosine"] > pc[n]["acc1_rung4_cosine"] for n in names)
        ),
        "rho_axis_identity_bias_acc1_raw": _r(_spearman(axis, idb)),
        "rho_axis_identity_bias_acc1_fraction": _r(_spearman(axis, idb_frac)),
    }

    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(out["rungwise_axis_ordering"], indent=1))
    print(json.dumps(out["equalized_n_sensitivity"]["largest"], indent=1))
    print(
        "capture intrusion: rho(rate, axis) =",
        out["capture_intrusion_mediator"]["rho_rate_axis"],
        "| rho(rate, recovery) =",
        out["capture_intrusion_mediator"]["rho_rate_recovery"],
    )


if __name__ == "__main__":
    main()
