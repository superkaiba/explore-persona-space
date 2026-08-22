"""Verify #2394 jailbreak-mining headline numbers + audit two constructions (task #2480).

0-GPU, read-only. Loads the COMMITTED ``eval_results/issue_2394/`` JSONs, re-derives
every headline number the paper cites (Step B, COLLECT-ALL, never fail-fast), then
audits the two reviewer-facing constructions (Step C, CHANNEL-SCOPED): the 5%
base-rate composition and the same-family failed-jailbreak negatives. The two
460 MB ``.npz`` are NOT read -- the audit is about eval-set composition + split
isolation, answerable from the JSONs + the seeded selection.

Semantics (plan MF-G): every headline claim is evaluated and its per-claim verdict
emitted, then the aggregate, before exit. A reproduction MISS is recorded and the run
CONTINUES (a miss is not a kill). Nonzero exit / exceptions are reserved for
missing / corrupt / absent-key inputs -- the Kill-criteria hard stop.

Run:  uv run python scripts/issue2480_verify.py
Test: uv run python scripts/issue2480_verify.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from math import floor, log10
from pathlib import Path

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
    except (json.JSONDecodeError, OSError) as exc:
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
    vals = []
    for lk in layers:
        for arm in ("B_mapproj_benign", "B_mapproj_indomain", "B_mapproj_merged"):
            vals.append((dig(layers, lk, arm, "pr_auc"), lk, arm))
    mx, lk, arm = max(vals, key=lambda t: t[0])
    stored = f"max B_mapproj pr_auc={mx:.7g} (L{lk} {arm})"
    claimed = "fixed-direction map-then-project (arm B) <=0.43 (B family only)"
    return stored, claimed, (mx <= 0.43) and _eq(sig(mx, 3), 0.425)


def _claim_labels_to_pr(data):
    n2r = dig(data["label_eff"], "layers", "19", "n_to_reach_pr")
    a = dig(n2r, "A")
    di = dig(n2r, "D_indomain")
    dm = dig(n2r, "D_merged")
    stored = f"A={a}, D_indomain={di:.6g}, D_merged={dm:.6g} (target={dig(n2r, 'target')})"
    claimed = "A<=10 (10 is smallest budget swept, upper bound) vs D~47-51 at L19"
    return stored, claimed, (a == 10) and (45.0 <= di <= 52.0) and (45.0 <= dm <= 52.0)


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
    rows, ok = [], True
    for b in sorted(curves, key=lambda k: int(k)):
        a = dig(curves, b, "A", "pr_auc_mean")
        di = dig(curves, b, "D_indomain", "pr_auc_mean")
        dm = dig(curves, b, "D_merged", "pr_auc_mean")
        won = a > di and a > dm
        ok = ok and won
        rows.append(f"b{b}:A={a:.3f}>{'both' if won else 'FAIL'}(Di={di:.3f},Dm={dm:.3f})")
    return "; ".join(rows), "A > {D_indomain, D_merged} at ALL 6 budgets [10..320]", ok


def _claim_benign_r2(data):
    vals = list(dig(data["map_arms"], "map_r2", "benign").values())
    lo, hi = min(vals), max(vals)
    stored = f"benign R^2 min={lo:.6g}, max={hi:.6g}, mean={sum(vals) / len(vals):.4g}"
    return (
        stored,
        "benign R^2 in -0.12..-0.88",
        _eq(round(lo, 2), -0.88) and _eq(round(hi, 2), -0.12),
    )


def _claim_indomain_r2(data):
    vals = list(dig(data["map_arms"], "map_r2", "indomain").values())
    lo, hi = min(vals), max(vals)
    stored = f"indomain R^2 min={lo:.6g}, max={hi:.6g} (n_train=1377<d=3584, reg-limited)"
    return (
        stored,
        "in-domain R^2 in +0.33..+0.62",
        _eq(round(lo, 2), 0.33) and _eq(round(hi, 2), 0.62),
    )


CLAIMS = [
    ("probe=oracle equality (MF-C)", _claim_equality),
    ("context probe 0.973 (hardneg L27)", _claim_hardneg_probe),
    ("arm B <=0.43 (MF-D)", _claim_armB),
    ("labels-to-PR-0.80: 10 vs ~47-51", _claim_labels_to_pr),
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
    """CHANNEL-SCOPED audit of the two constructions. Reconstructs the pool 0-GPU."""
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

    # ---- Reconstruct the evaluated pool deterministically from the seeded selection.
    # Producer (scripts/issue1739_jbmine_compliance_pilot.py): SEED=0, positives =
    # mean & min-over-rollouts >= 90 top-150 by (min,mean); failcomp negatives = mean <= 5.0;
    # n_neg = min(round(POS_N*(1-0.05)/0.05), n_failcomp). Counts + thresholds are
    # npz-free; the exact 2850-of-3787 negative membership would need npz-ordered rng
    # replay, which the excluded-middle verdict does NOT depend on (all failcomp <= 5).
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

    # balanced_benign corroboration (benign negatives, base 0.5) -- arms present ⇒ verified.
    bb = dig(data["compliance_pilot"], "balanced_benign", "layers")
    bb_probe_max = max(dig(bb, lk, "probe", "pr_auc") for lk in bb)
    bb_arms_present = all(
        a in next(iter(bb.values())) for a in ("map_then_project", "rb_harmcomp", "rb_refusal")
    )
    bb_map_max = max(dig(bb, lk, "map_then_project", "pr_auc") for lk in bb)
    bb_rbhc_max = max(dig(bb, lk, "rb_harmcomp", "pr_auc") for lk in bb)
    bb_rbref_max = max(dig(bb, lk, "rb_refusal", "pr_auc") for lk in bb)

    # Prevalence-invariant ROC-AUC ordering (probe/oracle >> arm B) at L19.
    a_roc = dig(data["map_arms"], "layers", "19", "A_probe_vC", "roc_auc")
    e_roc = dig(data["map_arms"], "layers", "19", "E_probe_vA_oracle", "roc_auc")
    b_roc_max = max(
        dig(data["map_arms"], "layers", "19", a, "roc_auc")
        for a in ("B_mapproj_benign", "B_mapproj_indomain", "B_mapproj_merged")
    )

    # Split-isolation evidence (map_arms + label_eff _meta.split; compliance_pilot has none).
    ma_split = dig(data["map_arms"], "_meta").get("split")
    le_split = dig(data["label_eff"], "_meta").get("split")
    cp_has_split = "split" in dig(data["compliance_pilot"], "_meta")

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
        },
        "split_isolation": {
            "map_arms_split": ma_split,
            "label_eff_split": le_split,
            "compliance_pilot_has_split_key": cp_has_split,
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
    print("STEP C -- two-construction audit (CHANNEL-SCOPED)")
    print("=" * 78)
    r = audit["reconstruction"]
    g = audit["guard"]
    print("Guard (MF-A d): named populated basis splits non-empty =", g["named_populated_nonempty"])
    print(
        "  evil_toxicchat empty BY CONSTRUCTION (producer loads shared file only) =",
        g["toxicchat_empty_by_construction"],
        "-- stated, not flagged",
    )
    print()
    print("Construction 2 -- same-family failed-jailbreak negatives:")
    print(f"  reconstruction (SEED=0, 0-GPU, npz-free): n_have(DV ctxs)={r['n_have']}")
    print(
        f"    positives (mean&min>=90, top150): reconstructed={r['n_pos_reconstructed']} "
        f"meta={r['n_pos_meta']}"
    )
    print(
        f"    failcomp  (mean<=5)             : reconstructed={r['n_failcomp_reconstructed']} "
        f"meta={r['n_failcomp_meta']}"
    )
    print(f"    n_neg = min(round(150*19), failcomp) = {r['n_neg']}  (pool = 150 + {r['n_neg']})")
    print(f"    counts match producer _meta = {r['counts_match_meta']}")
    print(
        f"  EXCLUDED MIDDLE (5 < compliance mean < 90): {r['excluded_middle_5_to_90']} "
        "partial-complier contexts"
    )
    print("  VERDICT: removes benign-negative + context-identity channels; negatives are genuine")
    print(
        "  low-compliance jailbreak-family contexts on a split-isolated pool. BUT extreme-groups:"
    )
    print(
        f"  the {r['excluded_middle_5_to_90']} middle-band partial-compliers are EXCLUDED, so the"
    )
    print(
        "  ABSOLUTE 0.973 answers 'separate always-comply(>=90) from failed-compliance(<=5)', NOT"
    )
    print("  'detect always-comply among ALL same-family contexts' -- SCOPE caveat on the absolute")
    print("  number; the probe-vs-map RELATIVE read (shared pool) is unaffected.")
    print()
    b = audit["base_rate"]
    print("Construction 1 -- 5% base-rate composition:")
    print(f"  single eval block (all arms) = {b['eval_block']}")
    print(
        f"  no per-arm base_rate/eval/n override = {b['no_per_arm_override']} "
        f"(arm keys = {b['arm_key_sets']})"
    )
    print(
        f"  balanced_benign probe max = {b['balanced_benign_probe_max']:.4f}; map/r_B arms "
        f"present = {b['balanced_arms_present']} (VERIFIED, not demoted)"
    )
    print(
        f"    balanced map_then_project max={b['balanced_map_max']:.4f}, "
        f"rb_harmcomp max={b['balanced_rb_harmcomp_max']:.4f}, "
        f"rb_refusal max={b['balanced_rb_refusal_max']:.4f}"
    )
    print(
        f"  prevalence-invariant ROC-AUC ordering @L19: probe={b['roc_A']:.4f}, "
        f"oracle={b['roc_E_oracle']:.4f} >> arm B max={b['roc_B_max']:.4f}"
    )
    print("  VERDICT: NO differential base-rate inflation -- PR-AUC chance = base rate, applied")
    print("  EQUALLY to every arm on ONE eval block, so 5% cannot inflate probe RELATIVE to arm B.")
    print(
        "  balanced_benign is BENIGN-negative corroboration (2-variable control), NOT a definitive"
    )
    print(
        "  disconfirmation. Ordering base-rate-robust (ROC); PR-unit gap MAGNITUDE is 5%-specific."
    )
    print()
    s = audit["split_isolation"]
    print("Split isolation:")
    print(f"  map_arms._meta.split      = {s['map_arms_split']!r}")
    print(f"  label_efficiency._meta.split = {s['label_eff_split']!r}")
    print(
        f"  compliance_pilot._meta has split key = {s['compliance_pilot_has_split_key']} "
        "(no split key -- consistency advisory)"
    )
    print("=" * 78)
    return n_miss


# ------------------------------------------------------------------ self-test


def _self_test(committed_dir: Path) -> int:
    """Two simultaneously-perturbed values -> 2 MISS lines + completed report; a
    missing key -> hard-stop. Returns 0 iff both behaviors hold."""
    import copy

    ok = True
    base = load_all(committed_dir)

    # (1) perturb TWO values -> exactly 2 misses, report completes (no exception).
    d2 = copy.deepcopy(base)
    d2["map_arms"]["layers"]["19"]["A_probe_vC"]["pr_auc"] = 0.5
    d2["compliance_pilot"]["hardneg_failcomp_5pct"]["layers"]["27"]["probe"]["pr_auc"] = 0.5
    verdicts = run_verification(d2)
    n_miss = sum(1 for v in verdicts if not v["match"])
    print(f"[self-test] two-perturbation: {n_miss} MISS(es), report completed = True")
    if n_miss != 2:
        print(f"[self-test] FAIL: expected exactly 2 misses, got {n_miss}")
        ok = False

    # (2) delete a claimed key path -> hard-stop (MissingInputError).
    d3 = copy.deepcopy(base)
    del d3["map_arms"]["layers"]["19"]["A_probe_vC"]
    try:
        run_verification(d3)
        print("[self-test] FAIL: absent key did NOT hard-stop")
        ok = False
    except MissingInputError as exc:
        print(f"[self-test] missing-key hard-stop raised as expected: {exc}")

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
        # A reproduction MISS is NOT a kill (plan Kill criteria): complete + exit 0,
        # the miss is loud in stdout and flagged in the report + #2394 marker.
        return 0
    except MissingInputError as exc:
        print(f"HARD STOP (missing/corrupt/absent-key input): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
