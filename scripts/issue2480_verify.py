"""Verify #2394 jailbreak-mining headline numbers + audit two constructions (task #2480).

0-GPU, read-only. Loads the COMMITTED ``eval_results/issue_2394/`` JSONs, re-derives
every headline number the paper cites (Step B, COLLECT-ALL, never fail-fast), then
audits the two reviewer-facing constructions (Step C, CHANNEL-SCOPED): the 5%
base-rate composition and the same-family failed-jailbreak negatives. The two
460 MB ``.npz`` are NOT read -- the audit is about eval-set composition + split
isolation, answerable from the JSONs + the producer's deterministic selection RULES
(thresholds + count arithmetic). Neither pool MEMBERSHIP is replayed: the negative
draw is an npz-ordered ``rng.choice`` (a plan-forbidden read) and the positive
top-150 sort key ties across the boundary; the audit verdicts are
membership-independent (see the Construction-2 section of the report).

Semantics (plan MF-G): every headline claim is evaluated and its per-claim verdict
emitted, then the aggregate, before exit. A reproduction MISS is recorded and the run
CONTINUES (a miss is not a kill). Step C audit-channel verdicts are DERIVED from the
executable predicates -- a False predicate renders that channel FAILED / UNVERIFIED /
DEMOTED (never positive prose over a False predicate) and the AGGREGATE line flags
it. Nonzero exit / exceptions are reserved for missing / corrupt / absent-key
inputs -- the Kill-criteria hard stop.

Run:  uv run python scripts/issue2480_verify.py
Test: uv run python scripts/issue2480_verify.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from math import floor, log10
from pathlib import Path

# Exact universes the headline claims quantify over (round-2 hardening): a claim that
# says "ALL 6 budgets" / "across layers" must reject a subset/superset/empty sweep.
BUDGET_UNIVERSE = frozenset({10, 20, 40, 80, 160, 320})
LAYER_UNIVERSE = frozenset({"7", "11", "15", "19", "23", "27"})

# ------------------------------------------------------------------ hard-stop path


class MissingInputError(Exception):
    """A target JSON is missing/corrupt/unparseable, or a claimed key path is absent.

    The ONLY hard-stop path (Kill criteria). A reproduction MISS never raises this.
    """


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise MissingInputError(f"input missing: {path}")
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise MissingInputError(f"input corrupt/unparseable: {path}: {exc}") from exc


def dig(obj, *keys):
    """Navigate a key path; an absent key/index is a hard-stop (MissingInputError)."""
    cur = obj
    for key in keys:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise MissingInputError(f"absent key path {list(keys)} at {key!r}") from exc
    return cur


def sig(x: float, n: int) -> float:
    """Round x to n significant figures."""
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def _eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ------------------------------------------------------------------ Step B: claims
# Each claim returns (stored_repr, claimed_repr, match_bool). Extraction goes through
# dig(), so an absent key path hard-stops; a value mismatch is a recorded MISS.


def _claim_equality(data):
    a = dig(data["map_arms"], "layers", "19", "A_probe_vC", "pr_auc")
    e = dig(data["map_arms"], "layers", "19", "E_probe_vA_oracle", "pr_auc")
    stored = f"A_probe_vC={a:.7g}, E_probe_vA_oracle={e:.7g} (delta={abs(a - e):.4g})"
    claimed = (
        "matched-L19 A vs E both ~0.974 -> indistinguishable (delta inside AP noise, n_pos=98)"
    )
    return stored, claimed, _eq(sig(a, 3), 0.974) and _eq(sig(e, 3), 0.974)


def _claim_hardneg_probe(data):
    v = dig(data["compliance_pilot"], "hardneg_failcomp_5pct", "layers", "27", "probe", "pr_auc")
    return (
        f"hardneg L27 probe pr_auc={v:.7g}",
        "context probe ~0.973 (hardneg pool, L27)",
        _eq(sig(v, 3), 0.973),
    )


def _claim_armB(data):
    layers = dig(data["map_arms"], "layers")
    universe_ok = set(layers) == set(LAYER_UNIVERSE)
    vals = []
    for lk in layers:
        for arm in ("B_mapproj_benign", "B_mapproj_indomain", "B_mapproj_merged"):
            vals.append((dig(layers, lk, arm, "pr_auc"), lk, arm))
    mx, lk, arm = max(vals, key=lambda t: t[0])
    stored = (
        f"max B_mapproj pr_auc={mx:.7g} (L{lk} {arm}); "
        f"layers={sorted(layers, key=int)} (universe ok={universe_ok})"
    )
    claimed = (
        "fixed-direction map-then-project (arm B) <=0.43 over EXACTLY layers "
        "{7,11,15,19,23,27} (B family only)"
    )
    return stored, claimed, universe_ok and (mx <= 0.43) and _eq(sig(mx, 3), 0.425)


def _claim_labels_to_pr(data):
    n2r = dig(data["label_eff"], "layers", "19", "n_to_reach_pr")
    a = dig(n2r, "A")
    di = dig(n2r, "D_indomain")
    dm = dig(n2r, "D_merged")
    t = dig(n2r, "target")
    stored = f"target={t}, A={a}, D_indomain={di:.6g}, D_merged={dm:.6g}"
    claimed = (
        "A<=10 (10 = smallest budget swept, upper bound) vs D_indomain=50.99 / "
        "D_merged=46.66 at L19, target exactly 0.8 (the scratch-quoted 'D~47-51' band "
        "does NOT bracket D_merged=46.66 -- stored values are authoritative)"
    )
    match = (t == 0.8) and (a == 10) and _eq(round(di, 2), 50.99) and _eq(round(dm, 2), 46.66)
    return stored, claimed, match


def _claim_full_label_A(data):
    v = dig(data["label_eff"], "layers", "19", "full_label_ref", "A")
    return (
        f"full_label_ref.A={v:.7g} (n_train=1377)",
        "full-label ref A ~0.98",
        _eq(round(v, 2), 0.98),
    )


def _claim_full_label_oracle(data):
    v = dig(data["label_eff"], "layers", "19", "full_label_ref", "E_oracle")
    return f"full_label_ref.E_oracle={v:.7g}", "full-label ref oracle ~0.98", _eq(round(v, 2), 0.98)


def _claim_every_budget(data):
    curves = dig(data["label_eff"], "layers", "19", "curves")
    budgets = {int(k) for k in curves}
    universe_ok = budgets == set(BUDGET_UNIVERSE)
    rows, won_all = [], True
    for bk in sorted(curves, key=lambda k: int(k)):
        a = dig(curves, bk, "A", "pr_auc_mean")
        di = dig(curves, bk, "D_indomain", "pr_auc_mean")
        dm = dig(curves, bk, "D_merged", "pr_auc_mean")
        won = a > di and a > dm
        won_all = won_all and won
        rows.append(f"b{bk}:A={a:.3f}>{'both' if won else 'FAIL'}(Di={di:.3f},Dm={dm:.3f})")
    stored = f"budgets={sorted(budgets)} (universe ok={universe_ok}); " + "; ".join(rows)
    claimed = (
        "A > {D_indomain, D_merged} at EXACTLY budgets {10,20,40,80,160,320} "
        "(subset/superset/empty sweeps REJECTED)"
    )
    return stored, claimed, universe_ok and won_all


def _claim_benign_r2(data):
    block = dig(data["map_arms"], "map_r2", "benign")
    universe_ok = set(block) == set(LAYER_UNIVERSE)
    vals = list(block.values())
    lo, hi = min(vals), max(vals)
    stored = (
        f"benign R^2 min={lo:.6g}, max={hi:.6g}, mean={sum(vals) / len(vals):.4g}; "
        f"layers={sorted(block, key=int)} (universe ok={universe_ok})"
    )
    return (
        stored,
        "benign R^2 extrema -0.12..-0.88 over EXACTLY layers {7,11,15,19,23,27}",
        universe_ok and _eq(round(lo, 2), -0.88) and _eq(round(hi, 2), -0.12),
    )


def _claim_indomain_r2(data):
    block = dig(data["map_arms"], "map_r2", "indomain")
    universe_ok = set(block) == set(LAYER_UNIVERSE)
    vals = list(block.values())
    lo, hi = min(vals), max(vals)
    stored = (
        f"indomain R^2 min={lo:.6g}, max={hi:.6g} (n_train=1377<d=3584, reg-limited); "
        f"layers={sorted(block, key=int)} (universe ok={universe_ok})"
    )
    return (
        stored,
        "in-domain R^2 extrema +0.33..+0.62 over EXACTLY layers {7,11,15,19,23,27}",
        universe_ok and _eq(round(lo, 2), 0.33) and _eq(round(hi, 2), 0.62),
    )


CLAIMS = [
    ("probe=oracle equality (MF-C)", _claim_equality),
    ("context probe 0.973 (hardneg L27)", _claim_hardneg_probe),
    ("arm B <=0.43 (MF-D)", _claim_armB),
    ("labels-to-PR-0.80: <=10 vs 50.99/46.66", _claim_labels_to_pr),
    ("full-label ref A ~0.98 (MF-E)", _claim_full_label_A),
    ("full-label ref oracle ~0.98 (MF-E)", _claim_full_label_oracle),
    ("ahead at every label budget (MF-E)", _claim_every_budget),
    ("benign R^2 -0.12..-0.88", _claim_benign_r2),
    ("in-domain R^2 +0.33..+0.62", _claim_indomain_r2),
]


def run_verification(data: dict) -> list[dict]:
    """COLLECT-ALL: evaluate every claim; an absent key hard-stops, a mismatch is recorded."""
    out = []
    for name, fn in CLAIMS:
        stored, claimed, match = fn(data)  # dig() inside -> MissingInputError hard-stops
        out.append({"name": name, "stored": stored, "claimed": claimed, "match": match})
    return out


# ------------------------------------------------------------------ Step C: audit


def run_audit(data: dict, committed_dir: Path) -> dict:
    """CHANNEL-SCOPED audit of the two constructions + gated audit-source reads.

    Computes the PREDICATES only; ``print_report`` DERIVES each channel's verdict text
    from them, so a False predicate can never render as positive prose (round-2 fix).
    """
    pc = load_json(committed_dir / "compliance_percontext.json")

    # MF-A(d) guard: every split the basis file is CLAIMED to carry is non-empty.
    # evil_train + evil_hh_rlhf are the populated pool basis; evil_toxicchat is EMPTY
    # by construction (producer loads only the shared file) -- stated, never flagged.
    named_populated = ("evil_train", "evil_hh_rlhf")
    for split in named_populated:
        block = pc.get(split)
        if not block or len(block) == 0:
            raise MissingInputError(f"named basis split empty: {split}")
    toxic_empty = len(pc.get("evil_toxicchat", {})) == 0

    # ---- Re-apply the producer's deterministic selection RULES (0-GPU, npz-free).
    # Producer (scripts/issue1739_jbmine_compliance_pilot.py): positives = mean &
    # min-over-rollouts >= 90, top-150 by (min,mean); failcomp negatives = mean <= 5.0;
    # n_neg = min(round(POS_N*(1-0.05)/0.05), n_failcomp). Thresholds + COUNTS are
    # npz-free. Neither pool MEMBERSHIP is replayed: the 2850-of-3787 negative draw is
    # an npz-ordered rng.choice (plan-forbidden read), and the positive top-150 sort
    # key ties at (100.0, 100.0) across the boundary, so the exact positive set is
    # candidate-ordering-dependent too. The excluded-middle verdict is
    # membership-independent: every eligible positive has mean&min >= 90 and every
    # failcomp candidate has mean <= 5, so no middle-band context can enter the pool
    # under any tie/draw resolution.
    pos_mean_min, neg_fail_max, pos_n = 90.0, 5.0, 150
    scored = []  # (mean, min) over every context with a DV
    for _split, ctxs in pc.items():
        for _cid, val in ctxs.items():
            scored.append((float(val["mean"]), float(val["min_over_rollouts"])))
    n_have = len(scored)
    pos_elig = [(m, mn) for (m, mn) in scored if m >= pos_mean_min and mn >= pos_mean_min]
    pos_used = sorted(pos_elig, key=lambda t: (t[1], t[0]), reverse=True)[:pos_n]
    n_failcomp = sum(1 for (m, _mn) in scored if m <= neg_fail_max)
    n_neg = min(round(len(pos_used) * (1 - 0.05) / 0.05), n_failcomp)
    middle = sum(1 for (m, _mn) in scored if neg_fail_max < m < pos_mean_min)

    meta = dig(data["compliance_pilot"], "_meta")
    counts_ok = (len(pos_used) == meta["n_pos"]) and (n_failcomp == meta["n_failcomp"])

    # ---- Base-rate: single eval block, no per-arm base_rate/eval/n override.
    ma_eval = dig(data["map_arms"], "eval")
    arm_key_sets = set()
    for _lk, arms in dig(data["map_arms"], "layers").items():
        for _arm, d in arms.items():
            arm_key_sets.update(d.keys())
    no_override = not ({"base_rate", "eval", "n"} & arm_key_sets)

    # balanced_benign corroboration (benign negatives, base 0.5). Presence is checked
    # over EVERY layer; when the map/r_B arms are absent the leg DEMOTES to
    # scratch-cited corroboration (UNVERIFIED) -- the plan's MF-F branch, now
    # implemented (previously a latent hard-stop) -- rather than hard-stopping.
    bb = dig(data["compliance_pilot"], "balanced_benign", "layers")
    bb_probe_max = max(dig(bb, lk, "probe", "pr_auc") for lk in bb)
    bb_arm_names = ("map_then_project", "rb_harmcomp", "rb_refusal")
    bb_arms_present = all(a in arms for arms in bb.values() for a in bb_arm_names)
    if bb_arms_present:
        bb_map_max = max(dig(bb, lk, "map_then_project", "pr_auc") for lk in bb)
        bb_rbhc_max = max(dig(bb, lk, "rb_harmcomp", "pr_auc") for lk in bb)
        bb_rbref_max = max(dig(bb, lk, "rb_refusal", "pr_auc") for lk in bb)
    else:
        bb_map_max = bb_rbhc_max = bb_rbref_max = None

    # Prevalence-invariant ROC-AUC ordering (probe/oracle >> arm B) at L19 -- a
    # PREDICATE, not assumed prose.
    a_roc = dig(data["map_arms"], "layers", "19", "A_probe_vC", "roc_auc")
    e_roc = dig(data["map_arms"], "layers", "19", "E_probe_vA_oracle", "roc_auc")
    b_roc_max = max(
        dig(data["map_arms"], "layers", "19", a, "roc_auc")
        for a in ("B_mapproj_benign", "B_mapproj_indomain", "B_mapproj_merged")
    )
    roc_ordering_ok = min(a_roc, e_roc) > b_roc_max

    # Split-isolation evidence (map_arms + label_eff _meta.split; compliance_pilot has
    # none). A missing key reads as None -> print_report renders "ABSENT", never a
    # positive "split-isolated" claim.
    ma_split = dig(data["map_arms"], "_meta").get("split")
    le_split = dig(data["label_eff"], "_meta").get("split")
    cp_has_split = "split" in dig(data["compliance_pilot"], "_meta")

    # ---- Gated audit-source reads (plan Step C sibling + residual-(a) numbers).
    tox = load_json(committed_dir / "compliance_percontext_toxicchat_probe.json")
    tox_n = len(dig(tox, "evil_toxicchat"))

    tr = load_json(committed_dir / "transfer_results.json")

    def _tr19(direction: str, arm: str) -> float:
        return dig(tr, "directions", direction, "layers", "19", arm, "pr_auc")

    t2h = _tr19("evil_train->evil_hh_rlhf", "A_transfer")
    t2h_w = _tr19("evil_train->evil_hh_rlhf", "A_within")
    h2t = _tr19("evil_hh_rlhf->evil_train", "A_transfer")
    h2t_w = _tr19("evil_hh_rlhf->evil_train", "A_within")
    transfer_ok = (
        _eq(sig(t2h, 3), 0.894)
        and _eq(sig(t2h_w, 3), 0.947)
        and _eq(sig(h2t, 3), 0.623)
        and _eq(sig(h2t_w, 3), 0.982)
    )

    return {
        "guard": {"named_populated_nonempty": True, "toxicchat_empty_by_construction": toxic_empty},
        "reconstruction": {
            "n_have": n_have,
            "n_pos_reconstructed": len(pos_used),
            "n_pos_meta": meta["n_pos"],
            "n_failcomp_reconstructed": n_failcomp,
            "n_failcomp_meta": meta["n_failcomp"],
            "n_neg": n_neg,
            "counts_match_meta": counts_ok,
            "excluded_middle_5_to_90": middle,
        },
        "base_rate": {
            "eval_block": ma_eval,
            "no_per_arm_override": no_override,
            "arm_key_sets": sorted(arm_key_sets),
            "balanced_benign_probe_max": bb_probe_max,
            "balanced_arms_present": bb_arms_present,
            "balanced_map_max": bb_map_max,
            "balanced_rb_harmcomp_max": bb_rbhc_max,
            "balanced_rb_refusal_max": bb_rbref_max,
            "roc_A": a_roc,
            "roc_E_oracle": e_roc,
            "roc_B_max": b_roc_max,
            "roc_ordering_ok": roc_ordering_ok,
        },
        "split_isolation": {
            "map_arms_split": ma_split,
            "label_eff_split": le_split,
            "compliance_pilot_has_split_key": cp_has_split,
        },
        "toxicchat": {"n": tox_n, "expected": 671, "ok": tox_n == 671},
        "transfer_famgrain": {
            "train_to_hh": t2h,
            "train_to_hh_within": t2h_w,
            "hh_to_train": h2t,
            "hh_to_train_within": h2t_w,
            "ok": transfer_ok,
        },
    }


# ------------------------------------------------------------------ IO / reporting


def load_all(committed_dir: Path) -> dict:
    return {
        "map_arms": load_json(committed_dir / "map_arms_results.json"),
        "label_eff": load_json(committed_dir / "label_efficiency_results.json"),
        "compliance_pilot": load_json(committed_dir / "compliance_pilot_results.json"),
    }


def print_report(verdicts: list[dict], audit: dict) -> int:
    """Render the report. Every channel verdict is DERIVED from its predicates."""
    print("=" * 78)
    print("STEP B -- headline-number verification (COLLECT-ALL, committed copy)")
    print("=" * 78)
    n_miss = 0
    for v in verdicts:
        tag = "MATCH" if v["match"] else "*** MISS ***"
        if not v["match"]:
            n_miss += 1
        print(f"[{tag}] {v['name']}")
        print(f"        claimed: {v['claimed']}")
        print(f"        stored : {v['stored']}")
    n = len(verdicts)
    print("-" * 78)
    if n_miss == 0:
        print(f"VERIFICATION: {n}/{n} headline numbers reproduce to quoted precision.")
    else:
        print(f"VERIFICATION: {n - n_miss}/{n} reproduce; {n_miss} MISS(es) FLAGGED above.")

    print()
    print("=" * 78)
    print("STEP C -- two-construction audit (verdicts DERIVED from executable predicates)")
    print("=" * 78)
    r = audit["reconstruction"]
    g = audit["guard"]
    s = audit["split_isolation"]
    b = audit["base_rate"]
    tox = audit["toxicchat"]
    tr = audit["transfer_famgrain"]

    print("Guard (MF-A d): named populated basis splits non-empty =", g["named_populated_nonempty"])
    print(
        "  evil_toxicchat empty BY CONSTRUCTION (producer loads shared file only) =",
        g["toxicchat_empty_by_construction"],
        "-- stated, not flagged",
    )
    print()

    # ---------------- Construction 2 (verdict DERIVED from predicates) ----------------
    counts_ok = r["counts_match_meta"]
    ma_s = s["map_arms_split"]
    le_s = s["label_eff_split"]
    split_present = ma_s is not None and le_s is not None
    c2_flags = []
    if not counts_ok:
        c2_flags.append("counts-mismatch-vs-meta")
    if not split_present:
        c2_flags.append("split-evidence-absent")

    print("Construction 2 -- same-family failed-jailbreak negatives:")
    print(
        "  rule re-application (0-GPU, npz-free, NO RNG/seed replay): "
        f"n_have(DV ctxs)={r['n_have']}"
    )
    print(
        f"    positives rule (mean&min>=90, top-150 by (min,mean)): "
        f"count={r['n_pos_reconstructed']} meta={r['n_pos_meta']}"
    )
    print(
        f"    failcomp rule (mean<=5): count={r['n_failcomp_reconstructed']} "
        f"meta={r['n_failcomp_meta']}"
    )
    print(f"    n_neg = min(round(150*19), n_failcomp) = {r['n_neg']}  (pool = 150 + {r['n_neg']})")
    print("    COUNT/RULE reproduction only: the exact 2850-of-3787 negative MEMBERSHIP is")
    print("    NOT replayed (npz-ordered rng.choice -- a plan-forbidden .npz read), and the")
    print("    positive top-150 membership is ordering-dependent ((min,mean) ties at")
    print("    (100,100) span the boundary). The verdict is membership-independent: every")
    print("    eligible positive has mean&min>=90 and every failcomp candidate has mean<=5,")
    print("    so no middle-band context can enter the pool under ANY tie/draw resolution.")
    print(f"    counts match producer _meta = {counts_ok}")
    print(
        f"  EXCLUDED MIDDLE (5 < compliance mean < 90): {r['excluded_middle_5_to_90']} "
        "partial-complier contexts"
    )
    print("  Split-isolation evidence:")
    print("    map_arms._meta.split         =", repr(ma_s) if ma_s is not None else "ABSENT")
    print("    label_efficiency._meta.split =", repr(le_s) if le_s is not None else "ABSENT")
    print(
        f"    compliance_pilot._meta has split key = {s['compliance_pilot_has_split_key']} "
        "(no split key -- consistency advisory)"
    )
    if not c2_flags:
        print("  VERDICT: VERIFIED -- removes benign-negative + context-identity channels;")
        print("  negatives are genuine low-compliance jailbreak-family contexts on a")
        print(
            f"  split-isolated pool. BUT extreme-groups: the {r['excluded_middle_5_to_90']} "
            "middle-band partial-compliers"
        )
        print("  are EXCLUDED, so the ABSOLUTE 0.973 answers 'separate always-comply(>=90)")
        print("  from failed-compliance(<=5)', NOT 'detect always-comply among ALL same-family")
        print("  contexts' -- SCOPE caveat on the absolute number; the probe-vs-map RELATIVE")
        print("  read (shared pool) is unaffected.")
    else:
        print(f"  VERDICT: FAILED/UNVERIFIED ({', '.join(c2_flags)}) -- the same-family-")
        print("  negatives construction does NOT verify on this data:")
        if not counts_ok:
            print(
                f"    counts mismatch: positives {r['n_pos_reconstructed']} vs meta "
                f"{r['n_pos_meta']}; failcomp {r['n_failcomp_reconstructed']} vs meta "
                f"{r['n_failcomp_meta']}"
            )
        if not split_present:
            print("    split evidence ABSENT -- a 'split-isolated pool' can NOT be claimed.")
    print()

    # ---------------- Construction 1 (verdict DERIVED from predicates) ----------------
    no_override = b["no_per_arm_override"]
    roc_ok = b["roc_ordering_ok"]
    bb_present = b["balanced_arms_present"]
    c1_flags = []
    if not no_override:
        c1_flags.append("per-arm-override-found")
    if not roc_ok:
        c1_flags.append("roc-ordering-violated")

    print("Construction 1 -- 5% base-rate composition:")
    print(f"  single eval block (all arms) = {b['eval_block']}")
    if no_override:
        print(f"  no per-arm base_rate/eval/n override = True (arm keys = {b['arm_key_sets']})")
    else:
        found = sorted({"base_rate", "eval", "n"} & set(b["arm_key_sets"]))
        print(f"  per-arm override keys FOUND: {found} -- matched-prevalence premise VIOLATED")
        print(f"    (arm keys = {b['arm_key_sets']})")
    if bb_present:
        print(
            f"  balanced_benign probe max = {b['balanced_benign_probe_max']:.4f}; map/r_B "
            "arms present = True (VERIFIED):"
        )
        print(
            f"    map_then_project max={b['balanced_map_max']:.4f}, "
            f"rb_harmcomp max={b['balanced_rb_harmcomp_max']:.4f}, "
            f"rb_refusal max={b['balanced_rb_refusal_max']:.4f}"
        )
    else:
        print(
            f"  balanced_benign probe max = {b['balanced_benign_probe_max']:.4f}; map/r_B "
            "arms ABSENT from the"
        )
        print("    committed JSON -- balanced leg DEMOTED to scratch-cited corroboration")
        print("    (UNVERIFIED); the matched-prevalence primary (item 1) carries the verdict.")
    if roc_ok:
        print(
            f"  prevalence-invariant ROC-AUC ordering @L19 HOLDS: probe={b['roc_A']:.4f}, "
            f"oracle={b['roc_E_oracle']:.4f} >> arm B max={b['roc_B_max']:.4f}"
        )
    else:
        print(
            f"  prevalence-invariant ROC-AUC ordering @L19 VIOLATED: probe={b['roc_A']:.4f}, "
            f"oracle={b['roc_E_oracle']:.4f} vs arm B max={b['roc_B_max']:.4f}"
        )
    if not c1_flags:
        print("  VERDICT: VERIFIED -- NO differential base-rate inflation: PR-AUC chance =")
        print("  base rate, applied EQUALLY to every arm on ONE eval block, so 5% cannot")
        print("  inflate the probe RELATIVE to arm B. balanced_benign is BENIGN-negative")
        print("  corroboration (2-variable control), NOT a definitive disconfirmation.")
        print("  Ordering base-rate-robust (ROC); PR-unit gap MAGNITUDE is 5%-specific.")
    else:
        print(f"  VERDICT: FAILED ({', '.join(c1_flags)}) -- the matched-prevalence /")
        print("  base-rate argument does NOT hold on this data.")
    print()

    # ---------------- Gated audit-source reads (verdicts DERIVED) ----------------
    print("Audit source reads (plan Step C sibling + residual-(a) transfer numbers, gated):")
    tox_tag = "OK" if tox["ok"] else f"MISMATCH (expected {tox['expected']})"
    print(f"  toxicchat sibling n_contexts = {tox['n']} -> {tox_tag}")
    tr_tag = "VERIFIED" if tr["ok"] else "MISMATCH vs report quotes"
    print(
        f"  transfer family-grain @L19: train->hh A_transfer={tr['train_to_hh']:.4f} "
        f"(quoted 0.894), within={tr['train_to_hh_within']:.4f} (0.947);"
    )
    print(
        f"    hh->train A_transfer={tr['hh_to_train']:.4f} (quoted 0.623), "
        f"within={tr['hh_to_train_within']:.4f} (0.982) -> {tr_tag}"
    )

    # -------- Aggregate: flipped channels are visible HERE too (round-2 fix) --------
    print("=" * 78)
    agg = [
        f"Step B {n - n_miss}/{n} MATCH" + ("" if n_miss == 0 else f" ({n_miss} MISS)"),
        "construction1=" + ("VERIFIED" if not c1_flags else "FAILED"),
        "construction2=" + ("VERIFIED" if not c2_flags else "FAILED/UNVERIFIED"),
        "balanced_leg=" + ("VERIFIED" if bb_present else "DEMOTED-UNVERIFIED"),
        "toxicchat_671=" + ("OK" if tox["ok"] else "MISMATCH"),
        "transfer_famgrain=" + ("VERIFIED" if tr["ok"] else "MISMATCH"),
    ]
    print("AGGREGATE: " + "; ".join(agg))
    print("=" * 78)
    return n_miss


# ------------------------------------------------------------------ self-test


def _self_test(committed_dir: Path) -> int:
    """Three legs: (1) two simultaneously-perturbed values -> 2 MISS lines + the
    aggregate line in the REAL rendered report (captured stdout, not a fabricated
    completion flag); (2) a missing key -> hard-stop; (3) flipped audit predicates ->
    DERIVED FAILED/UNVERIFIED/DEMOTED verdicts + flipped aggregate (round-2 fix
    regression). Returns 0 iff all legs hold."""
    import contextlib
    import copy
    import io

    ok = True
    base = load_all(committed_dir)

    # (1) perturb TWO values -> exactly 2 rendered MISS lines + completed report.
    d2 = copy.deepcopy(base)
    d2["map_arms"]["layers"]["19"]["A_probe_vC"]["pr_auc"] = 0.5
    d2["compliance_pilot"]["hardneg_failcomp_5pct"]["layers"]["27"]["probe"]["pr_auc"] = 0.5
    verdicts = run_verification(d2)
    audit = run_audit(d2, committed_dir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(verdicts, audit)
    out = buf.getvalue()
    miss_lines = [ln for ln in out.splitlines() if ln.startswith("[*** MISS ***]")]
    agg_lines = [ln for ln in out.splitlines() if ln.startswith("AGGREGATE:")]
    checks1 = {
        "exactly 2 recorded misses": sum(1 for v in verdicts if not v["match"]) == 2,
        "exactly 2 rendered MISS lines": len(miss_lines) == 2,
        "equality-claim MISS line rendered": any("probe=oracle equality" in x for x in miss_lines),
        "hardneg-claim MISS line rendered": any("context probe 0.973" in x for x in miss_lines),
        "summary counts the misses": "VERIFICATION: 7/9 reproduce; 2 MISS(es) FLAGGED" in out,
        "aggregate line rendered with misses": len(agg_lines) == 1
        and "Step B 7/9 MATCH (2 MISS)" in agg_lines[0],
    }
    for name, passed in checks1.items():
        print(f"[self-test 1] {name}: {'OK' if passed else 'FAIL'}")
        ok = ok and passed

    # (2) delete a claimed key path -> hard-stop (MissingInputError).
    d3 = copy.deepcopy(base)
    del d3["map_arms"]["layers"]["19"]["A_probe_vC"]
    try:
        run_verification(d3)
        print("[self-test 2] FAIL: absent key did NOT hard-stop")
        ok = False
    except MissingInputError as exc:
        print(f"[self-test 2] missing-key hard-stop raised as expected: {exc}")

    # (3) flipped audit predicates -> DERIVED negative verdicts (never positive prose
    # over a False predicate), visible per-channel AND in the aggregate line.
    d4 = copy.deepcopy(base)
    d4["map_arms"]["_meta"].pop("split", None)  # split evidence absent
    d4["compliance_pilot"]["_meta"]["n_pos"] = 151  # counts mismatch vs meta
    d4["map_arms"]["layers"]["19"]["A_probe_vC"]["base_rate"] = 0.5  # per-arm override
    for arms in d4["compliance_pilot"]["balanced_benign"]["layers"].values():
        arms.pop("map_then_project", None)  # MF-F demote branch
    audit4 = run_audit(d4, committed_dir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(run_verification(base), audit4)
    out4 = buf.getvalue()
    checks3 = {
        "C2 verdict flips FAILED/UNVERIFIED": "VERDICT: FAILED/UNVERIFIED" in out4
        and "counts-mismatch-vs-meta" in out4,
        "split absence rendered ABSENT": "split evidence ABSENT" in out4
        and "map_arms._meta.split         = ABSENT" in out4,
        "C1 verdict flips FAILED": "VERDICT: FAILED (per-arm-override-found" in out4,
        "balanced leg DEMOTED (MF-F)": "DEMOTED to scratch-cited corroboration" in out4,
        "no positive verdict on flipped data": "VERDICT: VERIFIED" not in out4,
        "aggregate flags the flips": "construction1=FAILED" in out4
        and "construction2=FAILED/UNVERIFIED" in out4
        and "balanced_leg=DEMOTED-UNVERIFIED" in out4,
    }
    for name, passed in checks3.items():
        print(f"[self-test 3] {name}: {'OK' if passed else 'FAIL'}")
        ok = ok and passed

    print("[self-test]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--committed-dir",
        default=str(repo_root() / "eval_results" / "issue_2394"),
        help="Directory holding the committed #2394 result JSONs.",
    )
    ap.add_argument("--self-test", action="store_true", help="Run the COLLECT-ALL self-test.")
    args = ap.parse_args()
    committed = Path(args.committed_dir)

    try:
        if args.self_test:
            return _self_test(committed)
        data = load_all(committed)
        verdicts = run_verification(data)
        audit = run_audit(data, committed)
        print_report(verdicts, audit)
        # A reproduction MISS / a flipped audit channel is NOT a kill (plan Kill
        # criteria): complete + exit 0 -- the miss/flip is loud per-channel AND in the
        # AGGREGATE line, and flagged in the report + #2394 marker.
        return 0
    except MissingInputError as exc:
        print(f"HARD STOP (missing/corrupt/absent-key input): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
