#!/usr/bin/env python3
"""#1739 claim4-controls P2 fold: per-rung table + lattice verdict + figures + note.

Consumes (plan v21 §9 off_pod reads — the complete loader surface):
- this round's P1 rows   `<claim4-root>/<behavior>/seed<S>/all_arms_spearman.json`
- this round's P1 preds  `<claim4-root>/<behavior>/seed<S>/transfer_preds/
  P-B-holdout-<rung>[.shufpair].jsonl` (per-context; `group` label = the
  group-hash group the paired context bootstrap resamples by)
- the banked parent rows `git show <banked-ref>:eval_results/issue_1739/
  r2v2_fits/<behavior>/all_arms_spearman.json` (item-4 table + provenance)
- the committed train summaries `eval_results/issue_1739/<b>/arm_results/
  all_arms_spearman.json` (the arm2 sanity band)
- the evil compliance DV raw judge file (companion join; local
  `eval_results/issue_1739/evil_ood_spread/compliance_full/evil_toxicchat/
  judge_raw_compliance_full.json`, staged from HF
  `issue1739_ctxmap/evil_ood_spread/compliance_full/...` on a miss)

Emits: `claim4_per_rung_table.{json,md}` + the §3 v21 lattice verdict, the
hero forest figure + exploratory dump (`figures/issue_1739/claim4_controls/`),
and the numbers-note `docs/map_behavior_prediction_claim4_controls_note.md`
(numbers + coverage only — the writeup is Thomas-authored and never edited).

Registered fold checks (plan §4 P0.5): row-coverage set-check BEFORE any
delta/CI (a missing cell is a reported gap, never an imputed zero); the arm4
cross-variant pairing assert (map-independent arm ⇒ bit-identical rows);
arm2 sanity band vs the committed train grid (out-of-band ⇒ item-3 verdict
flagged `inconclusive — adapter-suspect`); arm2 rows labeled
`mode: transfer (new this round)`; per-row provenance
(`banked@<ref>` vs `claim4_controls seed<S>`); companion ≥90% join-coverage
gate (declared skip below it, never a partial silently scored).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_r2v2_score.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIORS = ("evil", "sycophancy", "hallucination")
# 13 primary whole-holdout rungs (plan §3): the fold derives the realized
# roster MECHANICALLY from the rows (fit == P-B-holdout-<rung> & eval_rung ==
# rung) and checks it against this NAME-based registration — exact set
# equality per behavior, so a same-size WRONG set is a named mismatch, never
# a pass (review r2 item 4). Authoritative names = the banked fit labels at
# 5aae0a472b (the P-B whole-holdout rungs realized in
# eval_results/issue_1739/r2v2_fits/<behavior>/all_arms_spearman.json).
REGISTERED_PRIMARY_RUNGS: dict[str, frozenset[str]] = {
    "evil": frozenset({"evil_mhj", "evil_pair", "evil_tomgibbs", "hhrt", "toxicchat"}),
    "sycophancy": frozenset({"aita", "sycoans", "sycoays", "sycofb", "sycomim", "sycomwe"}),
    "hallucination": frozenset({"nqopen", "simpleqa"}),
}
EXPECTED_PRIMARY_COUNTS = {b: len(v) for b, v in REGISTERED_PRIMARY_RUNGS.items()}
FLAGSHIPS = (("evil", "evil_pair"), ("sycophancy", "sycomwe"))
ARM_CTX = "arm4_ridge_ctx"
ARM_MAP = "arm7_map_ridge_pred"
ARM_PROJ = "arm6_map_proj_e1"
ARM_CTXDIR = "arm2_ctx_native"
ARM_WSHUF = "arm20_shuffled_map_ridge"
# Arms whose per-context preds drive the paired deltas + the group bootstrap.
BOOT_ARMS = (ARM_CTX, ARM_MAP)
# Coverage set-check universe: arm -> the map variants the scorer emits primary
# rows under (arm2/arm20 run in the TRUE pass only — SHUFPAIR_ROSTER excludes
# them by design; a missing cell here is a REPORTED gap, never silently
# absent from the check).
COVER_ARM_VARIANTS = {
    ARM_CTX: ("true", "shufpair"),
    ARM_MAP: ("true", "shufpair"),
    ARM_CTXDIR: ("true",),
    ARM_WSHUF: ("true",),
}
# Registered replication set (plan §11): the lattice's draftable branches
# REQUIRE all 5 seeds per rung — a fold invoked with fewer seeds can only
# resolve `Not draftable / unresolved` (the registered denominator never
# shrinks to the invocation).
REGISTERED_SEED_COUNT = 5
SYCO_OOD_RUNGS = ("sycoans", "sycoays", "sycofb", "sycomim", "sycomwe")
ITEM4_ARMS = ("arm1_ctx_e1", ARM_CTX, ARM_PROJ, ARM_MAP, "arm11_oracle_proj")
DEFAULT_BANKED_REF = "5aae0a472b"
COMPLIANCE_RAW_LOCAL = Path(
    "eval_results/issue_1739/evil_ood_spread/compliance_full/evil_toxicchat/"
    "judge_raw_compliance_full.json"
)
COMPLIANCE_RAW_HF = (
    "issue1739_ctxmap/evil_ood_spread/compliance_full/evil_toxicchat/judge_raw_compliance_full.json"
)
# One colour = one meaning, matching the Result 2/3/5 family (context blue /
# navy, mapped warm, control gray; the shufpair control gets the gray).
ARM_STYLE = {
    "arm1_ctx_e1": ("Persona vector on context", "#4C72B0"),
    ARM_CTX: ("Ridge regression on context", "#0B3C5D"),
    ARM_PROJ: ("Persona vector on mapped answer", "#8c3b1e"),
    ARM_MAP: ("Ridge regression on mapped answer", "#e8b23a"),
    "arm11_oracle_proj": ("Persona vector on real answer (oracle)", "#1a6b54"),
    ARM_CTXDIR: ("Context-native direction (transfer, new this round)", "#56B4E9"),
    ARM_WSHUF: ("control: weight-permuted map ridge", "#9A9A9A"),
}
COLOR_DTRUE = "#e8b23a"  # Δ_true rides arm7's colour (the treatment read)
COLOR_DSHUF = "#9A9A9A"  # Δ_shuf rides the control gray
# Reader-facing labels (clean-result-critic round: figures must not expose
# rung/protocol slugs — plain-English names only; slugs stay in artifacts).
RUNG_LABEL = {
    "evil_mhj": "multi-turn human jailbreaks",
    "evil_pair": "PAIR optimizer attacks",
    "evil_tomgibbs": "Tom Gibbs multi-turn",
    "hhrt": "HH red-team",
    "toxicchat": "ToxicChat",
    "aita": "AITA slot (held-out Reddit)",
    "sycoans": "answer",
    "sycoays": "are-you-sure",
    "sycofb": "feedback",
    "sycomim": "mimicry",
    "sycomwe": "model-written evaluations",
    "nqopen": "NQ-Open",
    "simpleqa": "SimpleQA",
}
PANEL_LABEL = {"P-A": "single-dataset pool", "P-B": "fair-allocation pool"}
SERIES_TRUE_LABEL = "true-map advantage (mapped-answer minus context probe)"
SERIES_SHUF_LABEL = "shuffled-map advantage (pairing-shuffled map)"


def rung_label(behavior: str, rung: str) -> str:
    return f"{behavior}: {RUNG_LABEL.get(rung, rung)}"


def _log(msg: str) -> None:
    print(f"[claim4-fold {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# loading + coverage (pure functions — unit-tested)
# ---------------------------------------------------------------------------


def load_claim4_rows(root: Path, behaviors, seeds) -> tuple[list[dict], list[str]]:
    """All transfer rows across <behavior>/seed<S> files + the missing-file list."""
    rows: list[dict] = []
    missing: list[str] = []
    for b in behaviors:
        for s in seeds:
            p = root / b / f"seed{s}" / "all_arms_spearman.json"
            if not p.exists():
                missing.append(str(p))
                continue
            rows.extend(json.loads(p.read_text())["transfer_rows"])
    return rows, missing


def primary_rung_rows(rows: list[dict]) -> list[dict]:
    """The 13 primary whole-holdout reads: fit == P-B-holdout-<eval_rung>."""
    return [
        r
        for r in rows
        if r.get("protocol") == "P-B" and r.get("fit") == f"P-B-holdout-{r.get('eval_rung')}"
    ]


def row_coverage_check(
    rows: list[dict], behaviors, seeds
) -> tuple[dict, dict[str, list[str]], list[dict]]:
    """Set-check every (behavior, rung, seed, variant, arm) cell BEFORE stats.

    Returns (cells, rungs_by_behavior, gaps). A missing cell is a REPORTED
    gap, never an imputed zero (plan §3 row-coverage assert); a duplicate
    primary-rung key fails loud (the join grain must be unique).
    """
    cells: dict[tuple, dict] = {}
    for r in primary_rung_rows(rows):
        key = (
            str(r.get("behavior")),
            str(r.get("eval_rung")),
            int(r.get("seed")),
            str(r.get("map_variant")),
            str(r.get("arm")),
        )
        if key in cells:
            raise SystemExit(f"[fold] duplicate primary-rung row key {key}")
        cells[key] = r
    rungs_by_b = {b: sorted({k[1] for k in cells if k[0] == b}) for b in behaviors}
    gaps: list[dict] = []
    for b in behaviors:
        for rung in rungs_by_b[b]:
            for s in seeds:
                for arm, variants in COVER_ARM_VARIANTS.items():
                    for mv in variants:
                        if (b, rung, int(s), mv, arm) not in cells:
                            gaps.append(
                                {
                                    "behavior": b,
                                    "eval_rung": rung,
                                    "seed": int(s),
                                    "map_variant": mv,
                                    "arm": arm,
                                }
                            )
        registered = REGISTERED_PRIMARY_RUNGS.get(b)
        present = set(rungs_by_b[b])
        if registered is not None and present and present != set(registered):
            gaps.append(
                {
                    "behavior": b,
                    "note": f"primary rung-NAME set != registered — missing "
                    f"{sorted(set(registered) - present)}, unregistered "
                    f"{sorted(present - set(registered))} (present: {rungs_by_b[b]})",
                }
            )
    return cells, rungs_by_b, gaps


def arm4_pairing_check(cells: dict) -> None:
    """arm4 is map-independent: its rows must be bit-identical across variants."""
    bad = []
    seen = {}
    for (b, rung, s, mv, arm), r in cells.items():
        if arm != ARM_CTX:
            continue
        seen.setdefault((b, rung, s), {})[mv] = float(r["rho_frozen"])
    for k, per_mv in sorted(seen.items()):
        if {"true", "shufpair"} <= set(per_mv) and per_mv["true"] != per_mv["shufpair"]:
            bad.append((k, per_mv["true"], per_mv["shufpair"]))
    if bad:
        raise SystemExit(
            f"[fold] arm4 rows DIFFER across map variants (pairing check FAILED) — "
            f"first offenders: {bad[:3]}"
        )


ARM_MEAN_ARMS = ("arm1_ctx_e1", ARM_CTXDIR, ARM_CTX, ARM_MAP, ARM_PROJ, ARM_WSHUF)


def arm_true_means_declared(
    cells: dict, b: str, rung: str, seeds_ok: list[int], gaps: list[dict]
) -> dict:
    """Per-arm true-pass seed means with DECLARED partial-coverage gaps.

    An arm with rows for a strict subset of ``seeds_ok`` gets a gap row
    appended to ``gaps`` and mean ``None`` — never a silent partial average
    (plan §3 row-coverage discipline). A wholly-absent arm is ``None`` (its
    cell-level absence is already enumerated by the COVER_ARM_VARIANTS
    set-check for the covered arms)."""
    import numpy as np

    means: dict[str, float | None] = {}
    for a in ARM_MEAN_ARMS:
        vals = [
            cells[(b, rung, s, "true", a)]["rho_frozen"]
            for s in seeds_ok
            if (b, rung, s, "true", a) in cells
        ]
        if vals and len(vals) < len(seeds_ok):
            gaps.append(
                {
                    "behavior": b,
                    "eval_rung": rung,
                    "arm": a,
                    "note": f"partial seed coverage {len(vals)}/{len(seeds_ok)} on the true "
                    "pass — arm mean withheld (declared gap, never a partial average)",
                }
            )
            means[a] = None
        else:
            means[a] = float(np.mean(vals)) if vals else None
    return means


def seed_tci(vals: list[float]) -> dict:
    """mean ± t_{0.975, n-1}·SD/√n over the per-seed values (plan §6)."""
    import numpy as np
    from scipy import stats as sps

    v = np.asarray(vals, dtype=np.float64)
    n = int(v.size)
    out = {"n_seeds": n, "mean": float(v.mean()) if n else None, "per_seed": [float(x) for x in v]}
    if n >= 2:
        sd = float(v.std(ddof=1))
        half = float(sps.t.ppf(0.975, n - 1)) * sd / (n**0.5)
        out.update({"sd": sd, "tci": [out["mean"] - half, out["mean"] + half]})
    else:
        out.update({"sd": None, "tci": None})
    return out


def _ci_above_zero(ci) -> bool:
    return ci is not None and ci[0] > 0


def _ci_spans_zero(ci) -> bool:
    return ci is not None and ci[0] <= 0 <= ci[1]


def lattice_completeness_gaps(
    per_rung: list[dict],
    registered_rungs: dict[str, frozenset[str]] = REGISTERED_PRIMARY_RUNGS,
    n_seeds_required: int = REGISTERED_SEED_COUNT,
) -> list[str]:
    """Named gaps vs the REGISTERED lattice denominator (plan §3: the exact
    13-rung NAME set, all 5 seeds per rung, every quantity — dtrue/margin
    seed t-CIs AND both paired group-level context-bootstrap CIs). The rung
    check is EXACT set equality on the registered rung NAMES per behavior —
    a same-size wrong set is a named mismatch, never a pass (review r2
    item 4). ANY gap makes the lattice `Not draftable / unresolved`
    (kill-(b) note: a shrunken median denominator must never yield a
    draftable verdict)."""
    gaps: list[str] = []
    by_b: dict[str, list[dict]] = {}
    for r in per_rung:
        by_b.setdefault(str(r["behavior"]), []).append(r)
    for b in sorted(set(registered_rungs) | set(by_b)):
        rows = by_b.get(b, [])
        registered = registered_rungs.get(b)
        if registered is None:
            gaps.append(f"{b}: {len(rows)} rungs from an UNREGISTERED behavior")
            continue
        present = {str(r["eval_rung"]) for r in rows}
        if present != set(registered):
            missing = sorted(set(registered) - present)
            extra = sorted(present - set(registered))
            gaps.append(
                f"{b}: {len(present & set(registered))}/{len(registered)} registered "
                f"primary rungs present — missing {missing}, unregistered {extra}"
            )
        for r in sorted(rows, key=lambda r: str(r["eval_rung"])):
            tag = f"{b}/{r['eval_rung']}"
            n_seeds = r.get("dtrue", {}).get("n_seeds")
            if not r.get("complete") or n_seeds != n_seeds_required:
                gaps.append(f"{tag}: seeds incomplete ({n_seeds}/{n_seeds_required})")
                continue
            for q in ("dtrue", "margin"):
                if r[q].get("tci") is None:
                    gaps.append(f"{tag}: {q} seed t-CI missing")
            for q in ("dtrue_ctx_ci", "margin_ctx_ci"):
                if r.get(q) is None:
                    gaps.append(f"{tag}: {q} missing")
    return gaps


def lattice_verdict(
    per_rung: list[dict],
    flagships=FLAGSHIPS,
    registered_rungs: dict[str, frozenset[str]] = REGISTERED_PRIMARY_RUNGS,
    n_seeds_required: int = REGISTERED_SEED_COUNT,
) -> dict:
    """The §3 v21 registered verdict lattice (DISJOINT + exhaustive).

    ``per_rung`` entries need: behavior, eval_rung, complete (bool),
    dtrue (seed_tci dict), margin (seed_tci dict), dtrue_ctx_ci, margin_ctx_ci.
    Precedence: coverage (the FULL registered 13-rung-NAME x 5-seed
    denominator, :func:`lattice_completeness_gaps`) -> item-2 falsifier ->
    strong -> weak -> catch-all. A draftable verdict REQUIRES the complete
    registered set; ANY gap resolves `Not draftable / unresolved` with the
    gap named.
    """
    import numpy as np

    coverage_gaps = lattice_completeness_gaps(per_rung, registered_rungs, n_seeds_required)
    if coverage_gaps:
        shown = "; ".join(coverage_gaps[:8])
        more = f" (+{len(coverage_gaps) - 8} more)" if len(coverage_gaps) > 8 else ""
        return {
            "verdict": "Not draftable / unresolved",
            "reason": f"coverage gap: registered lattice denominator incomplete — {shown}{more}",
            "coverage_gaps": coverage_gaps,
        }

    by_key = {(r["behavior"], r["eval_rung"]): r for r in per_rung}
    flag_rows = []
    for b, rung in flagships:
        row = by_key.get((b, rung))
        if row is None or not row.get("complete") or row["dtrue"].get("tci") is None:
            return {
                "verdict": "Not draftable / unresolved",
                "reason": f"coverage gap: flagship ({b}, {rung}) rows/CIs incomplete — "
                "the lattice cannot be evaluated",
            }
        flag_rows.append(row)

    # item-2 falsifier takes explicit PRECEDENCE: Δ_true seed-CIs spanning 0
    # on BOTH flagships block both draftable branches.
    if all(_ci_spans_zero(r["dtrue"]["tci"]) for r in flag_rows):
        return {
            "verdict": "Not draftable / unresolved",
            "reason": "item-2 falsifier (precedence): the flagship Δ_true seed t-CIs span "
            "0 on BOTH flagships — seed variance swamps the deltas",
        }

    complete = [r for r in per_rung if r.get("complete")]
    med_dtrue = float(np.median([r["dtrue"]["mean"] for r in complete])) if complete else None
    med_margin = float(np.median([r["margin"]["mean"] for r in complete])) if complete else None
    medians = {
        "n_rungs_in_median": len(complete),
        "median_seed_mean_dtrue": med_dtrue,
        "median_seed_mean_margin": med_margin,
    }

    def _flag_ok(row) -> bool:
        return all(
            _ci_above_zero(ci)
            for ci in (
                row["dtrue"]["tci"],
                row["dtrue_ctx_ci"],
                row["margin"]["tci"],
                row["margin_ctx_ci"],
            )
        )

    flag_ok = [_flag_ok(r) for r in flag_rows]
    medians_pos = bool(med_dtrue is not None and med_dtrue > 0 and med_margin > 0)
    if medians_pos and all(flag_ok):
        return {"verdict": "Strong-form draftable", "medians": medians, "flagship_ok": flag_ok}
    if medians_pos and any(flag_ok):
        return {
            "verdict": "Weak-form draftable",
            "medians": medians,
            "flagship_ok": flag_ok,
            "note": "uncertainty-qualified — never a bare median sign (null false-confirm "
            "probability of a bare 13-rung median sign is exactly 0.5)",
        }
    # catch-all: DESCRIPTIVE only — no equivalence threshold is registered;
    # 'mechanism dead' / 'generic artifact' language is reserved for tight
    # informative control-matching, which the analyzer judges off these fields.
    desc = [
        {
            "behavior": r["behavior"],
            "eval_rung": r["eval_rung"],
            "dtrue_tci": r["dtrue"]["tci"],
            "dtrue_ctx_ci": r["dtrue_ctx_ci"],
            "margin_tci": r["margin"]["tci"],
            "margin_ctx_ci": r["margin_ctx_ci"],
            "margin_spans_zero": _ci_spans_zero(r["margin"]["tci"]),
            "dtrue_spans_zero": _ci_spans_zero(r["dtrue"]["tci"]),
        }
        for r in flag_rows
    ]
    return {
        "verdict": "Not draftable / unresolved",
        "reason": "catch-all: medians and/or flagship controlling intervals do not meet "
        "either draftable branch",
        "medians": medians,
        "flagship_descriptives": desc,
        "note": "descriptive distinction (registered, no equivalence threshold): an "
        "interval spanning 0 that is WIDE is an unresolved/imprecise read; "
        "Δ_shuf tracking Δ_true with TIGHT intervals is informative "
        "control-matching — only the latter supports generic-artifact language",
    }


# ---------------------------------------------------------------------------
# paired group-level context bootstrap (from persisted transfer preds)
# ---------------------------------------------------------------------------


def load_preds_series(root: Path, b: str, rung: str, seeds, variants=("true", "shufpair")):
    """Aligned per-context score series for (variant, arm, seed) on ONE rung.

    Returns ``(series, dv, groups, note)`` or ``(None, None, None, note)`` on
    a gap (missing file / context-set mismatch) — reported, never imputed.
    """
    import numpy as np

    per_key: dict[tuple, dict[str, float]] = {}
    dv_by_ctx: dict[str, float] = {}
    grp_by_ctx: dict[str, str] = {}
    for s in seeds:
        for mv in variants:
            tag = ".shufpair" if mv == "shufpair" else ""
            p = root / b / f"seed{s}" / "transfer_preds" / f"P-B-holdout-{rung}{tag}.jsonl"
            if not p.exists():
                return None, None, None, f"missing preds file: {p}"
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if str(r.get("rung")) != rung or r.get("arm") not in BOOT_ARMS:
                    continue
                per_key.setdefault((mv, r["arm"], int(s)), {})[str(r["context_id"])] = float(
                    r["score"]
                )
                dv_by_ctx[str(r["context_id"])] = float(r["dv"])
                grp = r.get("group")
                if grp is None:
                    # fail loud as a NAMED gap: a missing group label would
                    # silently degrade the GROUP-level bootstrap to
                    # per-context resampling — never imputed.
                    return (
                        None,
                        None,
                        None,
                        (
                            f"preds row missing 'group' label ({p.name}, "
                            f"ctx {r.get('context_id')}) — group bootstrap not computable"
                        ),
                    )
                grp_by_ctx[str(r["context_id"])] = str(grp)
    want_keys = [(mv, a, int(s)) for mv in variants for a in BOOT_ARMS for s in seeds]
    missing = [k for k in want_keys if k not in per_key]
    if missing:
        return None, None, None, f"missing preds series: {missing[:4]}"
    ctx_sets = {k: set(v) for k, v in per_key.items()}
    base = ctx_sets[want_keys[0]]
    if any(cs != base for cs in ctx_sets.values()):
        return None, None, None, "context sets differ across (variant, arm, seed) series"
    order = sorted(base)
    series = {k: np.asarray([per_key[k][c] for c in order]) for k in want_keys}
    dv = np.asarray([dv_by_ctx[c] for c in order])
    groups = [grp_by_ctx[c] for c in order]
    return series, dv, groups, f"n_ctx={len(order)}"


def group_bootstrap_rhos(mat, dv, groups, *, n_boot: int, rng):
    """Batched GROUP-level bootstrap Spearman: (S, n) scores -> (S, n_boot).

    The vectorize-many-cell-fits pattern: the group->rows index is
    precomputed ONCE, all draws' group choices are sampled in ONE rng call,
    and every draw's rho reduction rides the canonical batched helper
    ``arms.bootstrap_rhos`` (counting-sort ranks + exact moment-identity
    Pearson — bit-identical to ranking the drawn values directly). Group
    sizes may differ, so draws are bucketed by resample LENGTH and each
    bucket is one rectangular batched call; there is no per-draw Python
    Spearman anywhere. Returns ``(rhos, n_groups)``.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms as arms_mod

    groups = np.asarray(groups)
    ug = sorted(set(groups.tolist()))
    gidx = [np.flatnonzero(groups == g) for g in ug]
    # one vectorized sample of every draw's group choices (row-major fill —
    # the same stream the per-draw form consumed)
    gs = rng.integers(0, len(ug), size=(int(n_boot), len(ug)))
    draw_idx = [np.concatenate([gidx[g] for g in row]) for row in gs]  # index assembly only
    by_len: dict[int, list[int]] = {}
    for d, ix in enumerate(draw_idx):
        by_len.setdefault(int(ix.size), []).append(d)
    out = np.empty((mat.shape[0], int(n_boot)))
    for _w, dlist in sorted(by_len.items()):
        idx_mat = np.stack([draw_idx[d] for d in dlist])
        out[:, dlist] = arms_mod.bootstrap_rhos(mat, dv, idx_mat)
    return out, len(ug)


def ctx_bootstrap_ci(series, dv, groups, seeds, *, n_boot: int, rng, label: str = "") -> dict:
    """Percentile 95% CI on the seed-mean Δ_true and mechanism margin, from a
    GROUP-level context resample (plan §6: resample the holdout rung's
    contexts by group-hash group; recompute per-arm ρ per seed from the
    persisted preds; seed-mean per resample). Draw reductions are BATCHED
    (:func:`group_bootstrap_rhos`) — no serial per-draw Spearman."""
    import numpy as np

    t0 = time.time()
    keys = sorted(series)
    mat = np.stack([series[k] for k in keys])
    pos = {k: i for i, k in enumerate(keys)}
    rhos, n_groups = group_bootstrap_rhos(mat, dv, groups, n_boot=n_boot, rng=rng)
    dts = np.stack(
        [rhos[pos[("true", ARM_MAP, int(s))]] - rhos[pos[("true", ARM_CTX, int(s))]] for s in seeds]
    )
    dss = np.stack(
        [
            rhos[pos[("shufpair", ARM_MAP, int(s))]] - rhos[pos[("shufpair", ARM_CTX, int(s))]]
            for s in seeds
        ]
    )
    dtrue_draws = dts.mean(axis=0)
    margin_draws = (dts - dss).mean(axis=0)
    _log(
        f"[ctx-boot{f' {label}' if label else ''}] {int(n_boot)} draws x {mat.shape[0]} series "
        f"batched over {n_groups} groups in {time.time() - t0:.1f}s"
    )

    def _q(a):
        return [float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975))]

    return {
        "dtrue_ctx_ci": _q(dtrue_draws),
        "margin_ctx_ci": _q(margin_draws),
        "n_groups": n_groups,
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# banked reads: item-4 table + provenance reference
# ---------------------------------------------------------------------------


def read_banked(ref: str, behavior: str, banked_root: str) -> dict:
    out = subprocess.run(
        ["git", "show", f"{ref}:{banked_root}/{behavior}/all_arms_spearman.json"],
        capture_output=True,
        check=False,
        cwd=str(_REPO_ROOT),
    )
    if out.returncode != 0:
        raise SystemExit(f"[fold] cannot read banked {behavior}: {out.stderr.decode()[:200]}")
    return json.loads(out.stdout)


def item4_syco_table(banked_syco: dict, ref: str) -> list[dict]:
    """The corrected syco OOD table: banked P-A + P-B rows, 5 rungs × 5 arms.

    Single-seed banked rows — provenance carries `banked@<ref>` so they can
    never read as 5-seed rows (plan §4 P0.5 check (c))."""
    rows = []
    for r in banked_syco["transfer_rows"]:
        rung = r.get("eval_rung")
        if rung not in SYCO_OOD_RUNGS or r.get("arm") not in ITEM4_ARMS:
            continue
        proto = r.get("protocol")
        if proto == "P-A" and r.get("fit") == "P-A":
            panel = "P-A"
        elif proto == "P-B" and r.get("fit") == f"P-B-holdout-{rung}":
            panel = "P-B"
        else:
            continue
        rows.append(
            {
                "panel": panel,
                "eval_rung": rung,
                "arm": r["arm"],
                "rho_frozen": r.get("rho_frozen"),
                "ci_frozen": r.get("ci_frozen"),
                "n_eval": r.get("n_eval"),
                "seed": r.get("seed"),
                "provenance": f"banked@{ref} (single seed)",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# arm2 sanity band + companion join
# ---------------------------------------------------------------------------


def arm2_sanity_band(committed_root: Path, behavior: str, rows: list[dict], seeds) -> dict:
    """arm2's pvsynth ρ (this round, true pass) vs its committed train-grid band.

    Out-of-band ⇒ the item-3 verdict is flagged `inconclusive — adapter-suspect`
    (consistency WARN-A: an adapter bug makes arm2 spuriously weak, the
    claim-friendly direction) — never silently scored. Reads the FULL row
    list (pvsynth is a non-primary rung — it never enters the primary cells)."""
    p = committed_root / behavior / "arm_results" / "all_arms_spearman.json"
    if not p.exists():
        return {
            "behavior": behavior,
            "note": f"committed train summary absent: {p}",
            "flag": "not-evaluable",
        }
    committed = json.loads(p.read_text())
    band_vals = [
        float(r["rho_frozen"])
        for r in committed.get("arm_rows", [])
        if r.get("arm") == ARM_CTXDIR
        and r.get("variant") == "context_end"
        and r.get("regime") == "e1"
        and r.get("u_rung_label") == "full"
        and r.get("rho_frozen") is not None
    ]
    pv_vals: dict[int, list[float]] = {}
    for r in rows:
        if (
            r.get("behavior") == behavior
            and r.get("arm") == ARM_CTXDIR
            and r.get("map_variant") == "true"
            and r.get("eval_rung") == "pvsynth"
            and r.get("rho_frozen") is not None
        ):
            pv_vals.setdefault(int(r["seed"]), []).append(float(r["rho_frozen"]))
    # pvsynth is scored once per P-B fit (same eval block each holdout under
    # the frozen map) — average the per-fit reads per seed.
    per_seed = {s: sum(v) / len(v) for s, v in sorted(pv_vals.items())}
    out = {
        "behavior": behavior,
        "mode": "transfer (new this round)",
        "committed_band": [min(band_vals), max(band_vals)] if band_vals else None,
        "n_committed_cells": len(band_vals),
        "pvsynth_rho_per_seed": per_seed,
    }
    if band_vals and per_seed:
        mean = sum(per_seed.values()) / len(per_seed)
        out["pvsynth_rho_seed_mean"] = mean
        out["in_band"] = bool(min(band_vals) <= mean <= max(band_vals))
        if not out["in_band"]:
            out["flag"] = "inconclusive — adapter-suspect"
    else:
        # a not-evaluable sanity check must never leave item 3 silently
        # unflagged (distinct from the out-of-band adapter-suspect flag)
        out["note"] = "band or pvsynth rows unavailable — item-3 sanity check not evaluable"
        out["flag"] = "not-evaluable"
    return out


def _stage_compliance_raw(args) -> Path | None:
    src = args.compliance_raw
    if src and Path(src).exists():
        return Path(src)
    local = _REPO_ROOT / COMPLIANCE_RAW_LOCAL
    if local.exists():
        return local
    try:
        from explore_persona_space.orchestrate import hub

        dest = args.claim4_root / "_work" / "judge_raw_compliance_full.json"
        if not dest.exists():
            hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO,
                COMPLIANCE_RAW_HF,
                dest,
                repo_type="dataset",
                token=None,
            )
        return dest
    except Exception as exc:  # noqa: BLE001 — companion is declared-skippable
        _log(f"[companion] compliance raw unavailable ({exc}) — declared skip")
        return None


def compliance_per_context(raw_path: Path) -> dict[str, float]:
    """Per-context compliance means via the PRODUCER's own reduction:
    all_scores `{item}__NNNNN__NN` -> per-item kept-draw means ->
    gates.per_context_means (item = {context}_k{NN})."""
    from explore_persona_space.experiments.issue_1739.gates import per_context_means

    raw = json.loads(Path(raw_path).read_text())
    per_item: dict[str, list[float]] = {}
    for cid, rec in raw["all_scores"].items():
        item = str(cid).rsplit("__", 2)[0]
        score = rec.get("score") if isinstance(rec, dict) else None
        per_item.setdefault(item, []).append(None if score is None else float(score))
    return per_context_means(per_item)


def companion_toxicchat(args, min_coverage: float) -> dict:
    """ρ(P1 evil toxicchat per-context preds, compliance DV) — arms 4/7 ×
    variants, seed 0; ≥90% join-coverage gate (declared skip below).

    The PREDS-side rung label is the r2v2 fit label ``toxicchat`` (fit =
    ``P-B-holdout-toxicchat``, verified against the banked evil fits at
    5aae0a472b); ``evil_toxicchat`` is the COMPLIANCE-side directory/per_rung
    key only — the join is by context id.
    """
    import numpy as np

    raw_path = _stage_compliance_raw(args)
    if raw_path is None:
        return {"status": "declared_skip", "reason": "compliance raw file unavailable"}
    ctx_means = compliance_per_context(raw_path)
    series, dv, groups, note = load_preds_series(args.claim4_root, "evil", "toxicchat", [0])
    if series is None:
        return {"status": "declared_skip", "reason": f"preds gap: {note}"}
    order = None
    # rebuild the pred ctx order (load_preds_series sorted the shared set)
    some_key = sorted(series)[0]
    n_pred = series[some_key].shape[0]
    # recover ids by re-reading one preds file (cheap; keeps the helper pure)
    p = args.claim4_root / "evil" / "seed0" / "transfer_preds" / "P-B-holdout-toxicchat.jsonl"
    ids = sorted(
        {
            str(json.loads(line)["context_id"])
            for line in p.read_text().splitlines()
            if json.loads(line).get("rung") == "toxicchat"
        }
    )
    assert len(ids) == n_pred, (len(ids), n_pred)
    joined = [i for i, c in enumerate(ids) if c in ctx_means]
    coverage = len(joined) / max(len(ctx_means), 1)
    result = {
        "n_pred_contexts": n_pred,
        "n_compliance_contexts": len(ctx_means),
        "n_joined": len(joined),
        "coverage_of_compliance_rows": coverage,
        "raw_path": str(raw_path),
    }
    if coverage < min_coverage:
        result["status"] = "declared_skip"
        result["reason"] = (
            f"join coverage {coverage:.1%} < {min_coverage:.0%} gate — companion skipped "
            "with the measured number, never a partial silently scored"
        )
        return result
    from explore_persona_space.experiments.issue_1739 import arms as arms_mod

    order = np.asarray(joined)
    comp = np.asarray([ctx_means[ids[i]] for i in joined])
    rng = np.random.default_rng([1739, 23])
    rows = []
    t0 = time.time()
    for (mv, arm, s), vec in sorted(series.items()):
        v = vec[order]
        rho = float(arms_mod.spearman_rows(v[None], comp)[0])
        # plain context bootstrap (plan §6 CI column: bootstrap over contexts)
        # — BATCHED: one rng call for every draw's indices, one
        # arms.bootstrap_rhos reduction (no per-draw Python Spearman).
        idx_mat = rng.integers(0, len(joined), size=(int(args.n_boot), len(joined)))
        draws = arms_mod.bootstrap_rhos(v[None], comp, idx_mat)[0]
        rows.append(
            {
                "map_variant": mv,
                "arm": arm,
                "seed": s,
                "rho_vs_compliance": rho,
                "ci": [float(np.nanquantile(draws, q)) for q in (0.025, 0.975)],
            }
        )
    _log(
        f"[companion] {len(rows)} cells x {int(args.n_boot)} draws batched in "
        f"{time.time() - t0:.1f}s"
    )
    result["status"] = "scored"
    result["rows"] = rows
    return result


# ---------------------------------------------------------------------------
# figures (hero forest + exploratory dump) — /paper-plots conventions
# ---------------------------------------------------------------------------


def render_figures(table: dict, fig_dir: Path, seeds) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    rows = [r for r in table["per_rung"] if r.get("complete")]
    if rows:
        # ---- HERO: per-rung forest, Δ_true vs Δ_shuf with seed t-CIs -------
        order = sorted(rows, key=lambda r: (r["behavior"], r["eval_rung"]))
        labels = [rung_label(r["behavior"], r["eval_rung"]) for r in order]
        flag = [(r["behavior"], r["eval_rung"]) in set(FLAGSHIPS) for r in order]
        y = list(range(len(order)))[::-1]
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(order) + 1.6))
        for dy, r, fl in zip(y, order, flag, strict=True):
            for q, color, off, lbl in (
                ("dtrue", COLOR_DTRUE, 0.16, SERIES_TRUE_LABEL),
                ("margin_dshuf", COLOR_DSHUF, -0.16, SERIES_SHUF_LABEL),
            ):
                stat = r["dtrue"] if q == "dtrue" else r["dshuf"]
                if stat["mean"] is None:
                    continue
                ci = stat.get("tci")
                xerr = [[stat["mean"] - ci[0]], [ci[1] - stat["mean"]]] if ci is not None else None
                ax.errorbar(
                    [stat["mean"]],
                    [dy + off],
                    xerr=xerr,
                    fmt="o",
                    color=color,
                    ms=6 if fl else 4,
                    capsize=2,
                    label=lbl if dy == y[0] else None,
                )
        ax.axvline(0.0, color="#444444", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{lb} ★" if fl else lb for lb, fl in zip(labels, flag, strict=True)])
        ax.set_xlabel("Spearman ρ delta at the frozen layer (seed mean ± seed t-CI)")
        ax.set_title("Probe-on-mapped-answer gain per rung: true map vs shuffled-pairing map")
        ax.legend(loc="upper left", fontsize=7)
        savefig_paper(fig, "claim4_forest", dir=fig_dir)
        plt.close(fig)
        written.append("claim4_forest")

        # ---- low-level companion: per-seed values behind BOTH forest series --
        # (supersedes the true-map-only claim4_spaghetti draft; the sidecar
        # carries all 5 per-seed observations for both control-relevant
        # series on every rung)
        xs = list(range(len(order)))
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
        for ax, key, panel_title in (
            (axes[0], "dtrue", "true-map advantage per seed"),
            (axes[1], "dshuf", "shuffled-map advantage per seed"),
        ):
            for s in seeds:
                ys = []
                for r in order:
                    per_seed = dict(zip(map(int, r["seeds_used"]), r[key]["per_seed"], strict=True))
                    ys.append(per_seed.get(int(s)))
                ax.plot(xs, ys, marker="o", ms=3, lw=0.9, alpha=0.7, label=f"seed {s}")
            ax.axhline(0.0, color="#444444", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title(panel_title)
        axes[0].set_ylabel("ρ delta per seed")
        axes[0].legend(fontsize=7)
        savefig_paper(fig, "claim4_per_seed", dir=fig_dir)
        plt.close(fig)
        written.append("claim4_per_seed")

        # ---- exploratory: arm2 vs roster per-rung bars (true pass) ---------
        arm_cols = ["arm1_ctx_e1", ARM_CTXDIR, ARM_CTX, ARM_MAP, ARM_WSHUF]
        have = [a for a in arm_cols if any(r["arm_means"].get(a) is not None for r in order)]
        if have:
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            width = 0.8 / len(have)
            for j, a in enumerate(have):
                vals = [r["arm_means"].get(a) for r in order]
                # a missing measurement is rendered ABSENT + annotated —
                # never a misleading zero-height bar (CLAUDE.md 8c)
                xs_a = [x + j * width for x, v in zip(xs, vals, strict=True) if v is not None]
                ax.bar(
                    xs_a,
                    [v for v in vals if v is not None],
                    width=width,
                    color=ARM_STYLE[a][1],
                    label=ARM_STYLE[a][0],
                )
                for x, v in zip(xs, vals, strict=True):
                    if v is None:
                        ax.text(
                            x + j * width,
                            0.0,
                            "N/A — not run",
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=4.5,
                            color="#888888",
                        )
            ax.axhline(0.0, color="#444444", lw=0.8)
            ax.set_xticks([x + 0.4 for x in xs])
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Spearman ρ (seed mean, true map)")
            ax.set_title("Context-native direction comparator vs roster per rung (true map)")
            ax.legend(fontsize=7)
            savefig_paper(fig, "claim4_arm2_bars", dir=fig_dir)
            plt.close(fig)
            written.append("claim4_arm2_bars")

    # ---- exploratory: corrected syco OOD (item 4, banked) -------------------
    item4 = table.get("item4_syco_ood") or []
    if item4:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
        for ax, panel in zip(axes, ("P-A", "P-B"), strict=True):
            sub = [r for r in item4 if r["panel"] == panel]
            rungs = sorted({r["eval_rung"] for r in sub})
            arms_present = [a for a in ITEM4_ARMS if any(r["arm"] == a for r in sub)]
            width = 0.8 / max(len(arms_present), 1)
            for j, a in enumerate(arms_present):
                xs_a, vals, los, his = [], [], [], []
                for xi, rg in enumerate(rungs):
                    row = next((r for r in sub if r["arm"] == a and r["eval_rung"] == rg), None)
                    v = row["rho_frozen"] if row else None
                    if v is None:
                        # missing measurement: absent + annotated, never a
                        # zero-height bar / zero error bar
                        ax.text(
                            xi + j * width,
                            0.0,
                            "N/A — not run",
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=4.5,
                            color="#888888",
                        )
                        continue
                    ci = row.get("ci_frozen")
                    xs_a.append(xi + j * width)
                    vals.append(v)
                    los.append(max(0.0, v - ci[0]) if ci else 0.0)
                    his.append(max(0.0, ci[1] - v) if ci else 0.0)
                ax.bar(
                    xs_a,
                    vals,
                    yerr=[los, his],
                    width=width,
                    color=ARM_STYLE[a][1],
                    label=ARM_STYLE[a][0],
                    capsize=2,
                )
            ax.axhline(0.0, color="#444444", lw=0.8)
            ax.set_xticks([x + 0.4 for x in range(len(rungs))])
            ax.set_xticklabels([RUNG_LABEL.get(rg, rg) for rg in rungs], rotation=45, ha="right")
            ax.set_title(f"{PANEL_LABEL.get(panel, panel)} (banked, single seed)")
        axes[0].set_ylabel("Spearman ρ (frozen layer)")
        axes[0].legend(fontsize=6)
        fig.suptitle("Corrected sycophancy OOD rungs (banked single-seed rows)")
        savefig_paper(fig, "claim4_syco_ood", dir=fig_dir)
        plt.close(fig)
        written.append("claim4_syco_ood")

    # ---- exploratory: shufpair-vs-true map kNN (manipulation strength) ------
    knn = table.get("map_knn_seed0") or {}
    if knn:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for variant, color in (("true", COLOR_DTRUE), ("shufpair", COLOR_DSHUF)):
            for b, per_b in sorted(knn.items()):
                acc = per_b.get(variant)
                if acc:
                    ax.plot(
                        range(len(acc)),
                        acc,
                        color=color,
                        alpha=0.8,
                        lw=1.2,
                        label=f"{b} — "
                        + ("true map" if variant == "true" else "shuffled-pairing map"),
                        ls={"evil": "-", "sycophancy": "--", "hallucination": ":"}[b],
                    )
        ax.set_xlabel("layer index")
        ax.set_ylabel("kNN acc@1 (U-pool holdout, euclidean)")
        ax.set_title("Map retrieval: true vs shuffled-pairing (advisory manipulation check)")
        ax.legend(fontsize=6)
        savefig_paper(fig, "claim4_map_knn", dir=fig_dir)
        plt.close(fig)
        written.append("claim4_map_knn")

    # ---- exploratory: companion scatter --------------------------------------
    comp = table.get("companion") or {}
    if comp.get("status") == "scored":
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        rows = comp["rows"]
        xs = list(range(len(rows)))
        for i, r in enumerate(rows):
            color = ARM_STYLE[r["arm"]][1] if r["map_variant"] == "true" else COLOR_DSHUF
            lo = r["rho_vs_compliance"] - r["ci"][0]
            hi = r["ci"][1] - r["rho_vs_compliance"]
            ax.errorbar(
                [i],
                [r["rho_vs_compliance"]],
                yerr=[[max(0.0, lo)], [max(0.0, hi)]],
                fmt="o",
                color=color,
                capsize=2,
            )
        ax.axhline(0.0, color="#444444", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{r['arm'].split('_')[0]}\n{r['map_variant']}" for r in rows], fontsize=7
        )
        ax.set_ylabel("Spearman ρ(pred, compliance DV)")
        ax.set_title("Companion: evil ToxicChat map edge under the graded compliance DV")
        savefig_paper(fig, "claim4_companion", dir=fig_dir)
        plt.close(fig)
        written.append("claim4_companion")
    return written


def render_syco_percontext(
    preds_root: Path,
    fig_dir: Path,
    rungs: tuple[str, ...] = ("sycomwe", "sycoans", "sycofb", "sycoays", "sycomim"),
    seed: int = 0,
) -> str:
    """Per-context low-level view behind the corrected-sycophancy correlations.

    Scatters the mapped-answer probe's and the context probe's per-context
    predictions against the judge-scored DV on ALL FIVE corrected sycophancy
    rungs (largest gain -> largest loss), from the fair-allocation refit
    preds (seed 0). The persona-vector / oracle arms and the
    single-dataset-pool protocol exist only as banked single-seed AGGREGATE
    rows (no persisted per-context predictions), so they cannot appear here —
    the body carries the matching per-unit exemption. Fails loud on a preds
    gap — never an empty panel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = (len(rungs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), sharex=True)
    flat = list(axes.ravel())
    for ax, rung in zip(flat[: len(rungs)], rungs, strict=True):
        series, dv, _groups, note = load_preds_series(
            preds_root, "sycophancy", rung, [seed], variants=("true",)
        )
        if series is None:
            raise RuntimeError(f"per-context preds unavailable for {rung}: {note}")
        for arm in (ARM_MAP, ARM_CTX):
            preds = series[("true", arm, seed)]
            rho = spearmanr(preds, dv)[0]
            ax.scatter(
                dv,
                preds,
                s=9,
                alpha=0.45,
                linewidths=0,
                color=ARM_STYLE[arm][1],
                label=f"{ARM_STYLE[arm][0]} (ρ {rho:+.2f})",
            )
        ax.set_xlabel("judge-scored sycophancy DV (per-context mean)")
        ax.set_title(f"{RUNG_LABEL.get(rung, rung)} (n={len(dv)} contexts, seed {seed})")
        ax.legend(fontsize=7)
    for ax in flat[len(rungs) :]:
        ax.set_visible(False)
    for row in range(nrows):
        flat[row * ncols].set_ylabel("arm prediction (per-context)")
    fig.suptitle(
        "Per-context predictions behind the five corrected sycophancy rungs (fair-allocation refit)"
    )
    savefig_paper(fig, "claim4_syco_percontext", dir=fig_dir)
    plt.close(fig)
    return "claim4_syco_percontext"


# ---------------------------------------------------------------------------
# assembly + CLI
# ---------------------------------------------------------------------------


def _git_head_sha() -> str:
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def build_table(args) -> dict:
    import numpy as np

    seeds = [int(s) for s in args.seeds]
    rows, missing = load_claim4_rows(args.claim4_root, args.behaviors, seeds)
    cells, rungs_by_b, gaps = row_coverage_check(rows, args.behaviors, seeds)
    arm4_pairing_check(cells)

    per_rung: list[dict] = []
    for b in args.behaviors:
        for rung in rungs_by_b[b]:
            seeds_ok = [
                s
                for s in seeds
                if all(
                    (b, rung, s, mv, arm) in cells
                    for mv in ("true", "shufpair")
                    for arm in BOOT_ARMS
                )
            ]
            entry: dict = {
                "behavior": b,
                "eval_rung": rung,
                "flagship": (b, rung) in set(FLAGSHIPS),
                "seeds_used": seeds_ok,
                "complete": len(seeds_ok) == len(seeds) and len(seeds_ok) > 0,
                "provenance": f"claim4_controls seeds {seeds_ok}",
            }
            if seeds_ok:
                dtrue = [
                    cells[(b, rung, s, "true", ARM_MAP)]["rho_frozen"]
                    - cells[(b, rung, s, "true", ARM_CTX)]["rho_frozen"]
                    for s in seeds_ok
                ]
                dshuf = [
                    cells[(b, rung, s, "shufpair", ARM_MAP)]["rho_frozen"]
                    - cells[(b, rung, s, "shufpair", ARM_CTX)]["rho_frozen"]
                    for s in seeds_ok
                ]
                margin = [t - sh for t, sh in zip(dtrue, dshuf, strict=True)]
                entry["dtrue"] = seed_tci(dtrue)
                entry["dshuf"] = seed_tci(dshuf)
                entry["margin"] = seed_tci(margin)
                # secondary: projection-arm deltas (arm6) where present
                d6 = [
                    cells[(b, rung, s, "true", ARM_PROJ)]["rho_frozen"]
                    - cells[(b, rung, s, "shufpair", ARM_PROJ)]["rho_frozen"]
                    for s in seeds_ok
                    if (b, rung, s, "true", ARM_PROJ) in cells
                    and (b, rung, s, "shufpair", ARM_PROJ) in cells
                ]
                entry["arm6_true_minus_shuf"] = seed_tci(d6) if d6 else None
                # per-arm true-pass means (arm2 comparator + arm20 read);
                # partial seed coverage on ANY arm = a DECLARED gap row +
                # mean withheld, never a silent partial average.
                entry["arm_means"] = arm_true_means_declared(cells, b, rung, seeds_ok, gaps)
                if entry["arm_means"].get(ARM_CTXDIR) is not None:
                    entry["arm2_mode"] = "transfer (new this round)"
            else:
                entry["dtrue"] = entry["dshuf"] = entry["margin"] = {
                    "n_seeds": 0,
                    "mean": None,
                    "per_seed": [],
                    "sd": None,
                    "tci": None,
                }
                entry["arm_means"] = {}
            # paired group-level context bootstrap from the persisted preds
            entry["dtrue_ctx_ci"] = entry["margin_ctx_ci"] = None
            if seeds_ok and not args.skip_ctx_bootstrap:
                series, dv, groups, note = load_preds_series(args.claim4_root, b, rung, seeds_ok)
                if series is None:
                    entry["ctx_bootstrap_note"] = note
                    gaps.append({"behavior": b, "eval_rung": rung, "preds_gap": note})
                else:
                    rng = np.random.default_rng(
                        [1739, 22, list(args.behaviors).index(b), rungs_by_b[b].index(rung)]
                    )
                    ci = ctx_bootstrap_ci(
                        series,
                        dv,
                        groups,
                        seeds_ok,
                        n_boot=args.n_boot,
                        rng=rng,
                        label=f"{b}/{rung}",
                    )
                    entry["dtrue_ctx_ci"] = ci["dtrue_ctx_ci"]
                    entry["margin_ctx_ci"] = ci["margin_ctx_ci"]
                    entry["ctx_bootstrap"] = {k: ci[k] for k in ("n_groups", "n_boot")}
            per_rung.append(entry)

    verdict = lattice_verdict(per_rung)

    banked_syco = read_banked(args.banked_ref, "sycophancy", args.banked_root)
    item4 = item4_syco_table(banked_syco, args.banked_ref)

    arm2 = {b: arm2_sanity_band(args.committed_train_root, b, rows, seeds) for b in args.behaviors}
    arm2_flags = {b: a["flag"] for b, a in arm2.items() if a.get("flag")}
    if any("adapter-suspect" in f for f in arm2_flags.values()):
        verdict["item3_flag"] = "inconclusive — adapter-suspect (arm2 out of committed band)"
    elif arm2_flags:
        verdict["item3_flag"] = (
            "not-evaluable — arm2 sanity band could not be computed for: "
            + ", ".join(sorted(arm2_flags))
        )

    companion = companion_toxicchat(args, args.min_join_coverage)

    # seed-0 map kNN acc@1 per layer (true vs shufpair) for the advisory panel
    map_knn: dict[str, dict] = {}
    for b in args.behaviors:
        p = args.claim4_root / b / "seed0" / "map_diagnostics.json"
        if not p.exists():
            continue
        diags = json.loads(p.read_text())
        per_b: dict[str, list[float]] = {}
        for key, d in diags.items():
            variant = "shufpair" if key.endswith("|shufpair") else "true"
            if "pc_holdout" in key:
                continue
            layers = d.get("per_layer") or []
            acc = [
                (row.get("knn", {}).get("euclidean", {}).get("acc_at_k", {}).get("1"))
                for row in layers
            ]
            if any(a is not None for a in acc):
                per_b[variant] = [float(a) if a is not None else float("nan") for a in acc]
        if per_b:
            map_knn[b] = per_b

    return {
        "meta": {
            "generated_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_head_sha(),
            "banked_ref": args.banked_ref,
            "seeds": seeds,
            "n_boot": args.n_boot,
            "claim4_root": str(args.claim4_root),
            "flagships": [list(f) for f in FLAGSHIPS],
        },
        "coverage": {
            "missing_files": missing,
            "gaps": gaps,
            "primary_rungs": rungs_by_b,
            "registered_primary_rungs": {b: sorted(v) for b, v in REGISTERED_PRIMARY_RUNGS.items()},
            "expected_counts": EXPECTED_PRIMARY_COUNTS,
        },
        "per_rung": per_rung,
        "verdict": verdict,
        "arm2_sanity": arm2,
        "item4_syco_ood": item4,
        "companion": companion,
        "map_knn_seed0": map_knn,
    }


def write_markdown(table: dict, path: Path) -> None:
    lines = ["# Claim-4 controls — per-rung table", ""]
    lines.append(f"- generated: {table['meta']['generated_ts']} @ {table['meta']['git_commit']}")
    lines.append(f"- verdict: **{table['verdict'].get('verdict')}**")
    if table["verdict"].get("reason"):
        lines.append(f"- reason: {table['verdict']['reason']}")
    if table["verdict"].get("medians"):
        m = table["verdict"]["medians"]
        lines.append(
            f"- medians over {m['n_rungs_in_median']} rungs: Δ_true "
            f"{m['median_seed_mean_dtrue']:+.4f}, margin {m['median_seed_mean_margin']:+.4f}"
        )
    lines += [
        "",
        "| behavior | rung | ★ | Δ_true mean | Δ_true t-CI | Δ_true ctx-CI | Δ_shuf mean | "
        "margin mean | margin t-CI | margin ctx-CI | seeds |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _fmt(x):
        return "—" if x is None else f"{x:+.4f}"

    def _fmtci(ci):
        return "—" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"

    for r in table["per_rung"]:
        lines.append(
            f"| {r['behavior']} | {r['eval_rung']} | {'★' if r['flagship'] else ''} | "
            f"{_fmt(r['dtrue']['mean'])} | {_fmtci(r['dtrue'].get('tci'))} | "
            f"{_fmtci(r.get('dtrue_ctx_ci'))} | {_fmt(r['dshuf']['mean'])} | "
            f"{_fmt(r['margin']['mean'])} | {_fmtci(r['margin'].get('tci'))} | "
            f"{_fmtci(r.get('margin_ctx_ci'))} | {len(r['seeds_used'])} |"
        )
    if table["coverage"]["gaps"] or table["coverage"]["missing_files"]:
        lines += ["", "## Coverage gaps (reported, never imputed)", ""]
        for g in table["coverage"]["gaps"][:50]:
            lines.append(f"- {g}")
        for m in table["coverage"]["missing_files"]:
            lines.append(f"- missing file: {m}")
    path.write_text("\n".join(lines) + "\n")


def write_note(table: dict, path: Path) -> None:
    """Numbers-note for the writeup — numbers + coverage ONLY (the writeup is
    Thomas-authored; this file never rewrites his claims)."""
    v = table["verdict"]
    lines = [
        "# Claim-4 controls — numbers note (task #1739, claim4-controls round)",
        "",
        "Numbers + coverage only; claims stay with the writeup author.",
        "",
        f"- registered lattice verdict: **{v.get('verdict')}**"
        + (f" ({v['reason']})" if v.get("reason") else ""),
    ]
    if v.get("medians"):
        m = v["medians"]
        lines.append(
            f"- medians over {m['n_rungs_in_median']} primary rungs: seed-mean Δ_true "
            f"{m['median_seed_mean_dtrue']:+.4f}; seed-mean mechanism margin "
            f"{m['median_seed_mean_margin']:+.4f}"
        )
    if v.get("item3_flag"):
        lines.append(f"- item-3 (arm2 comparator): {v['item3_flag']}")
    for r in table["per_rung"]:
        if not r["flagship"]:
            continue
        lines.append(
            f"- flagship {r['behavior']}/{r['eval_rung']}: Δ_true {r['dtrue']['mean']:+.4f} "
            f"(t-CI {r['dtrue'].get('tci')}, ctx-CI {r.get('dtrue_ctx_ci')}); margin "
            f"{r['margin']['mean']:+.4f} (t-CI {r['margin'].get('tci')}, ctx-CI "
            f"{r.get('margin_ctx_ci')})"
            if r["dtrue"]["mean"] is not None
            else f"- flagship {r['behavior']}/{r['eval_rung']}: coverage gap"
        )
    comp = table.get("companion") or {}
    lines.append(
        f"- companion (evil ToxicChat × compliance DV): {comp.get('status')} — "
        + (
            comp.get("reason", "")
            if comp.get("status") != "scored"
            else f"{len(comp.get('rows', []))} cells scored at coverage "
            f"{comp.get('coverage_of_compliance_rows'):.1%}"
        )
    )
    n_gaps = len(table["coverage"]["gaps"]) + len(table["coverage"]["missing_files"])
    lines.append(f"- coverage: {n_gaps} reported gap entries (see claim4_per_rung_table.json)")
    lines.append("")
    lines.append("Full table: `eval_results/issue_1739/claim4_controls/claim4_per_rung_table.json`")
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--claim4-root", type=Path, default=Path("eval_results/issue_1739/claim4_controls")
    )
    ap.add_argument("--banked-ref", default=DEFAULT_BANKED_REF)
    ap.add_argument("--banked-root", default="eval_results/issue_1739/r2v2_fits")
    ap.add_argument("--committed-train-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument("--compliance-raw", type=Path, default=None)
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-join-coverage", type=float, default=0.9)
    ap.add_argument("--out-root", type=Path, default=None, help="default: --claim4-root")
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1739/claim4_controls"))
    ap.add_argument(
        "--note-out",
        type=Path,
        default=Path("docs/map_behavior_prediction_claim4_controls_note.md"),
    )
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--skip-ctx-bootstrap", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.out_root is None:
        args.out_root = args.claim4_root
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.import_check:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        from scipy import stats  # noqa: F401

        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            savefig_paper,
            set_paper_style,
        )
        from explore_persona_space.experiments.issue_1739.arms import (  # noqa: F401
            bootstrap_rhos,
            spearman_rows,
        )
        from explore_persona_space.experiments.issue_1739.gates import (  # noqa: F401
            per_context_means,
        )
        from explore_persona_space.orchestrate.hub import stage_hub_file  # noqa: F401

        print("[claim4-fold] import-check OK", flush=True)
        return 0

    table = build_table(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_json = args.out_root / "claim4_per_rung_table.json"
    out_json.write_text(json.dumps(table, indent=1))
    write_markdown(table, args.out_root / "claim4_per_rung_table.md")
    args.note_out.parent.mkdir(parents=True, exist_ok=True)
    write_note(table, args.note_out)
    _log(f"table -> {out_json}")
    _log(f"verdict: {table['verdict'].get('verdict')}")
    if not args.no_figures:
        written = render_figures(table, args.fig_dir, [int(s) for s in args.seeds])
        _log(f"figures -> {args.fig_dir} ({written})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
