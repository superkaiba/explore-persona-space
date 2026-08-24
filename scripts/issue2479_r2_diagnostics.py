"""Issue #2479 round-2 diagnostics — recomputed from committed artifacts (zero GPU).

Reads only committed round-1 artifacts (``eval_results/issue_2479/`` +
``data/issue_2479/axis_items/``) and writes
``eval_results/issue_2479/r2_diagnostics.json``. Pure counting/statistics —
no story or answer text is printed or persisted.

Computes:
  (a) tier-conditioned reads: designed-tier-ordinal -> recovery Spearman with
      the same 10,000-shuffle permutation machinery as the verdict (imported,
      seed 0, one-sided add-one p); per-tier rho(axis, recovery) at n=4;
      pooled within-tier rank association; tiers A-C subset rho;
  (b) retrieval baseline hierarchy: per-character identity+bias vs rung-4
      acc@1 win counts, euclidean + cosine;
  (c) scope diagnostics: rung-6 recovery range + axis ordering + rung6>rung4
      count; prefix-arm own-map ceilings vs the plan's 0.05 eligibility floor;
  (d) judged-answer-field CJK recount: the audit's registered regex applied to
      the ``answer`` field the judge actually consumed (the audit scanned the
      ``story`` wrapper), per character + pooled, plus the headline rho with
      intruded-answer items excluded (per-item scores joined by conv_id from
      the raw judge legs);
  (e) judge loss accounting: provider API-refusal draws (error=true,
      stop_reason=refusal), content drops, unscored items, tier
      concentration, and an adversarial 0/100 bound over unscored items
      (all 2^k per-character extreme assignments);
  (f) misc: verbatim-flatness realized-items disclosure numbers; the
      story-substrate max per-character axis shift; equalized-n largest
      recovery sensitivity.
"""

from __future__ import annotations

import json
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from issue2479_gradient_verdict import _spearman, spearman_perm_read  # noqa: E402

REPO = _HERE.parent
EVAL = REPO / "eval_results/issue_2479"
ITEMS_DIR = REPO / "data/issue_2479/axis_items"

# Registered intrusion class (byte-identical to issue2479_cjk_audit.py CJK_RE).
CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

BAND_ORD = {"A": 3, "B": 2, "C": 1, "D": 0}  # A = explicitly-AI ... D = stylized non-AI
N_PERM = 10_000
SEED = 0
FLOOR = 0.05  # plan ceiling-eligibility floor


def _load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def main() -> None:
    verdict = _load_json(EVAL / "gradient_verdict.json")
    freeze = _load_json(EVAL / "axis_freeze.json")
    gates = _load_json(EVAL / "instrument_gates.json")
    audit = _load_json(EVAL / "cjk_audit.json")
    pc = verdict["per_character"]
    names = sorted(pc)  # verdict headline uses sorted character order
    axis = np.array([pc[n]["axis_score"] for n in names])
    rec = np.array([pc[n]["recovery_fraction"] for n in names])
    band = [pc[n]["design_band"] for n in names]
    band_ord = np.array([BAND_ORD[b] for b in band], dtype=float)

    out: dict = {"issue": 2479, "characters": names, "design_band": band}

    # --- headline validation (must reproduce the committed verdict exactly) --
    head = spearman_perm_read(axis, rec, n_perm=N_PERM, seed=SEED, label="headline replay")
    committed = verdict["headline"]
    assert abs(head["rho"] - committed["rho"]) < 1e-12, (head["rho"], committed["rho"])
    assert abs(head["p_add_one"] - committed["p_add_one"]) < 1e-12
    out["headline_replay"] = {"rho": _r(head["rho"]), "p_add_one": _r(head["p_add_one"])}

    # --- (a) tier-conditioned reads -----------------------------------------
    tier = spearman_perm_read(
        band_ord, rec, n_perm=N_PERM, seed=SEED, label="design-tier ordinal -> recovery"
    )
    per_band_rho = {}
    within_ranks_axis, within_ranks_rec = np.zeros(len(names)), np.zeros(len(names))
    for b in "ABCD":
        idx = np.array([i for i, bb in enumerate(band) if bb == b])
        per_band_rho[b] = _r(_spearman(axis[idx], rec[idx]))
        within_ranks_axis[idx] = np.argsort(np.argsort(axis[idx])) + 1
        within_ranks_rec[idx] = np.argsort(np.argsort(rec[idx])) + 1
    pooled_within = _spearman(within_ranks_axis, within_ranks_rec)
    idx_ac = np.array([i for i, bb in enumerate(band) if bb in "ABC"])
    ac_read = spearman_perm_read(
        axis[idx_ac], rec[idx_ac], n_perm=N_PERM, seed=SEED, label="tiers A-C only"
    )
    out["tier_conditioned"] = {
        "tier_ordinal_to_recovery": {
            "rho": _r(tier["rho"]),
            "p_add_one": _r(tier["p_add_one"]),
            "null_q95": _r(tier["null_q95"]),
        },
        "tier_ordinal_to_axis_rho": _r(_spearman(band_ord, axis)),
        "per_tier_rho_axis_recovery_n4": per_band_rho,
        "pooled_within_tier_rank_rho": _r(pooled_within),
        "tiers_ABC_only_n12": {"rho": _r(ac_read["rho"]), "p_add_one": _r(ac_read["p_add_one"])},
        "recovery_range_by_tier": {
            b: [
                _r(min(pc[n]["recovery_fraction"] for n in names if pc[n]["design_band"] == b)),
                _r(max(pc[n]["recovery_fraction"] for n in names if pc[n]["design_band"] == b)),
            ]
            for b in "ABCD"
        },
    }

    # --- (b) retrieval baseline hierarchy ------------------------------------
    eu_wins = [n for n in names if pc[n]["acc1_identity_bias"] > pc[n]["acc1_rung4"]]
    eu_ties = [n for n in names if pc[n]["acc1_identity_bias"] == pc[n]["acc1_rung4"]]
    co_wins = [n for n in names if pc[n]["acc1_identity_bias_cosine"] > pc[n]["acc1_rung4_cosine"]]
    co_ties = [n for n in names if pc[n]["acc1_identity_bias_cosine"] == pc[n]["acc1_rung4_cosine"]]
    out["acc1_identity_vs_rung4"] = {
        "euclidean_identity_wins": len(eu_wins),
        "euclidean_ties": eu_ties,
        "euclidean_rung4_wins": sorted(set(names) - set(eu_wins) - set(eu_ties)),
        "cosine_identity_wins": len(co_wins),
        "cosine_ties": co_ties,
        "cosine_rung4_wins": sorted(set(names) - set(co_wins) - set(co_ties)),
    }

    # --- (c) rung-6 + prefix-arm scope ---------------------------------------
    rec6 = np.array([pc[n]["rung_r2_all"]["6_rotation"] / pc[n]["ceiling_r2"] for n in names])
    r6_read = spearman_perm_read(
        axis, rec6, n_perm=N_PERM, seed=SEED, label="axis -> rung-6 recovery"
    )
    out["rung6"] = {
        "recovery_min": _r(rec6.min()),
        "recovery_max": _r(rec6.max()),
        "rho_axis": _r(r6_read["rho"]),
        "p_add_one": _r(r6_read["p_add_one"]),
        "n_rung6_gt_rung4": int(
            sum(
                pc[n]["rung_r2_all"]["6_rotation"] > pc[n]["rung_r2_all"]["4_bias_refit"]
                for n in names
            )
        ),
    }
    prefix_ceil = {}
    for n in names:
        cell = (
            EVAL
            / "story_char_gradient"
            / f"cell_char_2479_{n}_op__instruct_prefix_L19_reduced_s0.json"
        )
        val = _load_json(cell)["reduced"]["ceiling_r2"]
        prefix_ceil[n] = float(val[0]) if isinstance(val, list) else float(val)
    pv = np.array(list(prefix_ceil.values()))
    out["prefix_arm"] = {
        "ceiling_min": _r(pv.min()),
        "ceiling_max": _r(pv.max()),
        "n_below_floor": int((pv < FLOOR).sum()),
        "floor": FLOOR,
        "per_character": {k: _r(v) for k, v in prefix_ceil.items()},
    }

    # --- (e) judge loss accounting (needed before the CJK recount join) ------
    item_means: dict[str, dict[str, float]] = {}
    acct = {}
    for n in names:
        leg = _load_json(EVAL / "judge_legs" / f"judge_raw_ail_{n}.json")
        per_item: dict[str, list[float]] = {}
        n_api_refusal = n_content_drop = n_other_err = n_ok = 0
        api_refusal_items: set[str] = set()
        for key, v in leg["all_scores"].items():
            cid = key.split("__")[0].removeprefix(f"ail_{n}_")
            if v.get("error") and v.get("stop_reason") == "refusal":
                n_api_refusal += 1
                api_refusal_items.add(cid)
                continue
            if v.get("error"):
                n_other_err += 1
                continue
            s = v.get("score")
            if isinstance(s, (int, float)) and 0 <= s <= 100:
                per_item.setdefault(cid, []).append(float(s))
                n_ok += 1
            else:
                n_content_drop += 1
        item_means[n] = {cid: float(np.mean(v)) for cid, v in per_item.items()}
        fr = freeze["characters"][n]
        acct[n] = {
            "band": pc[n]["design_band"],
            "n_ok_draws": n_ok,
            "n_provider_api_refusal_draws": n_api_refusal,
            "n_content_drop_draws": n_content_drop,
            "n_other_error_draws": n_other_err,
            "n_items_with_api_refusal": len(api_refusal_items),
            "n_items": fr["n_items"],
            "n_scored_items": fr["n_scored_items"],
            "n_unscored_items": fr["n_items"] - fr["n_scored_items"],
            "n_fully_censored_items": sum(
                1 for cid in api_refusal_items if cid not in item_means[n]
            ),
        }
        # cross-check the freeze's scored-item count against the raw join
        assert len(item_means[n]) == fr["n_scored_items"], (
            n,
            len(item_means[n]),
            fr["n_scored_items"],
        )
    tot = {
        k: int(sum(a[k] for a in acct.values()))
        for k in (
            "n_provider_api_refusal_draws",
            "n_content_drop_draws",
            "n_other_error_draws",
            "n_items_with_api_refusal",
            "n_unscored_items",
            "n_fully_censored_items",
        )
    }
    tot["n_api_refusal_draws_in_tier_D"] = int(
        sum(a["n_provider_api_refusal_draws"] for a in acct.values() if a["band"] == "D")
    )
    tot["n_unscored_items_in_tier_D"] = int(
        sum(a["n_unscored_items"] for a in acct.values() if a["band"] == "D")
    )
    out["judge_loss_accounting"] = {"totals": tot, "per_character": acct}

    # Validate the freeze aggregation convention (mean of per-item means).
    for n in names:
        agg = float(np.mean(list(item_means[n].values())))
        assert abs(agg - freeze["characters"][n]["score"]) < 5e-3, (
            n,
            agg,
            freeze["characters"][n]["score"],
        )

    # 0/100 adversarial bound over unscored items, all 2^k per-character combos.
    affected = [n for n in names if acct[n]["n_unscored_items"] > 0]
    rhos = []
    base_scores = np.array([freeze["characters"][n]["score"] for n in names])
    for combo in product([0.0, 100.0], repeat=len(affected)):
        adj = base_scores.copy()
        for v, n in zip(combo, affected):
            i = names.index(n)
            fr = freeze["characters"][n]
            adj[i] = (
                fr["score"] * fr["n_scored_items"] + v * (fr["n_items"] - fr["n_scored_items"])
            ) / fr["n_items"]
        rhos.append(_spearman(adj, rec))
    out["unscored_items_sensitivity"] = {
        "affected_characters": affected,
        "n_combos": len(rhos),
        "rho_min": _r(min(rhos)),
        "rho_max": _r(max(rhos)),
        "note": "axis rescored with every unscored item set to 0 or 100 per character "
        "(all extreme per-character assignments); headline rho recomputed each time",
    }

    # --- (d) judged-answer-field CJK recount ----------------------------------
    ans_counts = {}
    excl_scores = []
    zara_s2280 = None
    pooled_intr = pooled_items = 0
    max_shift = 0.0
    max_shift_char = None
    for n in names:
        rows = [
            json.loads(line)
            for line in (ITEMS_DIR / f"axis_items_{n}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        intr_ids = {r["conv_id"] for r in rows if CJK_RE.search(str(r.get("answer", "")))}
        if n == "zara":
            z = next((r for r in rows if r["conv_id"] == "s2280"), None)
            zara_s2280 = {
                "found": z is not None,
                "answer_field_intruded": bool(z and CJK_RE.search(str(z["answer"]))),
            }
        ans_counts[n] = {"intruded_answers": len(intr_ids), "n_items": len(rows)}
        pooled_intr += len(intr_ids)
        pooled_items += len(rows)
        kept = [v for cid, v in item_means[n].items() if cid not in intr_ids]
        excl = float(np.mean(kept))
        ans_counts[n]["axis_score_excl_intruded_answers"] = _r(excl)
        shift = abs(excl - freeze["characters"][n]["score"])
        if shift > max_shift:
            max_shift, max_shift_char = shift, n
        excl_scores.append(excl)
    ans_read = spearman_perm_read(
        np.array(excl_scores),
        rec,
        n_perm=N_PERM,
        seed=SEED,
        label="axis excl intruded-answer items -> recovery",
    )
    out["judged_answer_cjk_recount"] = {
        "pooled_intruded_answers": pooled_intr,
        "pooled_items": pooled_items,
        "per_character": ans_counts,
        "rho_excl_intruded_answers": _r(ans_read["rho"]),
        "p_add_one": _r(ans_read["p_add_one"]),
        "max_axis_shift": _r(max_shift, 3),
        "max_axis_shift_character": max_shift_char,
        "zara_s2280": zara_s2280,
        "note": "regex applied to the judge-consumed `answer` field; the committed "
        "cjk_audit.json judged_axis_pool counts scan the full `story` wrapper "
        "(a different substrate; both are real intrusion reads)",
    }

    # story-substrate max shift (from the committed audit, for the body wording)
    shifts = {
        n: abs(
            audit["per_character"][n]["judged_axis_pool"]["axis_score_all_items"]
            - audit["per_character"][n]["judged_axis_pool"]["axis_score_excl_intruded"]
        )
        for n in names
    }
    worst = max(shifts, key=shifts.get)
    out["story_substrate_max_axis_shift"] = {"character": worst, "shift": _r(shifts[worst], 3)}

    # --- (f) flatness disclosure + equalized-n sensitivity --------------------
    vf = gates["verbatim_flatness"]
    out["verbatim_flatness_realized"] = {
        "items_per_char_target": vf["items_per_char_target"],
        "n_scored_items_per_char": sorted({v["n_scored_items"] for v in vf["per_char"].values()}),
        "total_draws": int(sum(v["drops"]["n_total_draws"] for v in vf["per_char"].values())),
    }
    eq = verdict["equalized_n"]["companions"]["_rows1028"]
    deltas = {c: eq["values"][i] - rec[names.index(c)] for i, c in enumerate(eq["characters"])}
    worst_eq = max(deltas, key=lambda c: abs(deltas[c]))
    out["equalized_n_largest_delta"] = {
        "character": worst_eq,
        "full_n_recovery": _r(rec[names.index(worst_eq)]),
        "equalized_recovery": _r(eq["values"][eq["characters"].index(worst_eq)]),
    }

    out_path = EVAL / "r2_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {out_path}")
    print(
        json.dumps({k: v for k, v in out.items() if k != "judge_loss_accounting"}, indent=1)[:4000]
    )
    print("loss totals:", json.dumps(tot))


if __name__ == "__main__":
    main()
