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
import zlib
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


def committed_band_vals(committed_root: Path, behavior: str, arm: str) -> list[float]:
    """The committed train-grid cell values whose min/max define an arm's band.

    Code motion out of :func:`arm2_sanity_band` (arm2fix round) so the D0-P5
    band-instrument audit and the arm2fix sanity mask read the SAME committed
    cells (context_end / e1 / full-U-rung rows of the behavior's main
    summary). Returns [] when the summary is absent."""
    p = committed_root / behavior / "arm_results" / "all_arms_spearman.json"
    if not p.exists():
        return []
    committed = json.loads(p.read_text())
    return [
        float(r["rho_frozen"])
        for r in committed.get("arm_rows", [])
        if r.get("arm") == arm
        and r.get("variant") == "context_end"
        and r.get("regime") == "e1"
        and r.get("u_rung_label") == "full"
        and r.get("rho_frozen") is not None
    ]


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
    band_vals = committed_band_vals(committed_root, behavior, ARM_CTXDIR)
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


# ---------------------------------------------------------------------------
# arm2fix mode (plan §4 Leg 2 D2): repaired-arm2 vs banked-arm7 fold
# ---------------------------------------------------------------------------

A2FIX_SERIES = ("arm2_new", "arm4_true", "arm7_true", "arm7_shufpair", "arm7_parity")
# Registered adapter tags PER repaired arm (codex r2 blocker
# arm2fix-mixed-adapter-provenance): the scorer's _adapter_tag_for emits
# exactly these — anything else on a repaired arm's rows is foreign/mixed
# provenance, never a configuration. Restricted-ness is decided from the
# scorer's own ARM2_RESTRICTED_ADAPTERS (imported at resolve time).
A2FIX_VALID_ADAPTER_TAGS: dict[str, frozenset[str]] = {
    "arm2_ctx_native": frozenset({"v1", "v2-component-restricted"}),
    "arm2q_ctx_native": frozenset({"v2-quantile", "v2-quantile-restricted"}),
}
# The amended §3 leg-2 lattice's verdict vocabulary — verbatim (plan v26).
A2FIX_VERDICTS = (
    "INCONCLUSIVE-ADAPTER",
    "MAP-BEATS-CONTEXT-DIRECTION",
    "MAP-ADVANTAGE-NOT-SHOWN",
    "WEAK-MIXED",
)


def _is_primary_pb_row(r: dict) -> bool:
    return (
        r.get("protocol") == "P-B"
        and r.get("fit") == f"P-B-holdout-{r.get('eval_rung')}"
        and r.get("map_variant") in (None, "true")
    )


def a2fix_unique_index(items, key_fn, *, context: str) -> dict:
    """THE single universe-construction primitive (codex r6 class-termination
    round): index items by ``key_fn`` from the UNFILTERED input, raising a
    loud DUPLICATE KEY ERROR naming the consumer context. Every arm2fix
    coverage / verdict / prediction universe is built through this ONE
    implementation — a filtered or last-write-wins view can no longer
    self-define a universe anywhere on the arm2fix surface. Domain SCOPING
    (selecting the behavior / arm / rung the consumer is about) happens
    BEFORE this call; admission FILTERING (None / non-finite / provenance)
    happens strictly AFTER, on the returned exact index."""
    out: dict = {}
    dup: list = []
    for it in items:
        k = key_fn(it)
        if k in out:
            dup.append(k)
        else:
            out[k] = it
    if dup:
        raise SystemExit(
            f"[a2fix] DUPLICATE KEY ERROR ({context}): {len(set(dup))} key(s) realized "
            f"more than once ({sorted(set(dup))[:6]}) — one row per key is required "
            "(a last-write-wins reduction would be input-order-dependent)"
        )
    return out


def a2fix_exact_coverage(expected, admitted, realized, *, context: str) -> dict:
    """THE single exact-set comparator: ``missing`` = expected keys with no
    ADMITTED value, ``extra`` = realized-or-admitted keys outside the
    registered universe; ``complete`` requires neither. Every registered-set
    coverage read (primary-D, parity) routes through this one implementation
    (codex r6: per-site set algebra kept re-introducing one-way variants)."""
    expected, admitted, realized = set(expected), set(admitted), set(realized)
    missing = sorted(expected - admitted)
    extra = sorted((realized | admitted) - expected)
    return {
        "context": context,
        "missing": missing,
        "extra": extra,
        "complete": not missing and not extra,
        "n_expected": len(expected),
        "n_admitted": len(admitted & expected),
    }


def a2fix_merge_consistent(dst: dict, src: dict, *, what: str, context: str) -> None:
    """Consistency-checked metadata merge (codex r6 BLOCKER
    arm2fix-context-preds-duplicate-universe): a key present in both maps
    with a DIFFERENT value is a loud conflict — never last-write-wins."""
    for k, v in src.items():
        if k in dst and dst[k] != v:
            raise SystemExit(
                f"[a2fix] METADATA CONFLICT ({context}): {what} for key {k!r} disagrees "
                f"across sources ({dst[k]!r} != {v!r}) — the paired bootstrap requires "
                "exactly one value per context"
            )
        dst[k] = v


def a2fix_resolve_repairs(
    new_rows: list[dict], behaviors, overrides: dict[str, str] | None = None
) -> dict[str, dict]:
    """PER-BEHAVIOR repair resolution (code-review r1 sustained blocker
    ``arm2fix-mixed-adapter-fold``): plan v26 §4/§7 run the R-A→R-B→R-C
    ladder + the matched-budget parity duty PER BEHAVIOR, and the committed
    D0 P2 evidence makes the MIXED topology the expected production shape
    (degenerate midpoint split fires for evil only). One global
    ``repaired_arm``/``rows_restricted``/``parity_required`` cannot express
    that shape — it either crashes the join on plan-conforming input (R-B:
    parity rows correctly absent on unrestricted behaviors) or silently
    mints INCONCLUSIVE-ADAPTER (R-C: arm2q rows exist only where the
    quantile adapter ran).

    Resolution is MECHANICAL from each behavior's OWN rows: repaired_arm =
    override if given, else arm2q when it has primary rows, else arm2;
    adapters/rows_restricted from the repaired arm's own adapter tags;
    parity_present from the behavior's own arm7 parity-row-matched rows. A
    behavior with NO arm2-family primary rows is a loud coverage error."""
    out: dict[str, dict] = {}
    for b in behaviors:
        prim = [r for r in new_rows if r.get("behavior") == b and _is_primary_pb_row(r)]
        fam_present = {r.get("arm") for r in prim} & {ARM_CTXDIR, "arm2q_ctx_native"}
        ov = (overrides or {}).get(b)
        if ov is not None:
            if ov not in fam_present:
                raise SystemExit(
                    f"[a2fix] --repaired-arm requests {b}={ov} but {b} carries no primary "
                    f"{ov} rows (arm2-family arms present: {sorted(fam_present)})"
                )
            repaired = ov
        elif "arm2q_ctx_native" in fam_present:
            repaired = "arm2q_ctx_native"
        elif ARM_CTXDIR in fam_present:
            repaired = ARM_CTXDIR
        else:
            raise SystemExit(
                f"[a2fix] COVERAGE ERROR: no arm2-family primary rows for behavior {b} "
                "(the repaired lane did not run, or the wrong root was passed)"
            )
        # ONE canonical registered adapter tag per behavior (codex r2 blocker
        # arm2fix-mixed-adapter-provenance): mixed tags across seeds/rungs
        # would verdict an unregistered mixed estimator; a missing/foreign tag
        # is invalid provenance. Both are loud exits, never a configuration.
        from scripts.issue1739_r2v2_score import ARM2_RESTRICTED_ADAPTERS

        tags = sorted({str(r.get("adapter")) for r in prim if r.get("arm") == repaired})
        if len(tags) != 1:
            raise SystemExit(
                f"[a2fix] PROVENANCE ERROR: {b}'s repaired arm {repaired} carries "
                f"{len(tags)} distinct adapter tags {tags} across its primary rows — one "
                "behavior must be exactly ONE registered repair (mixed seed/rung "
                "provenance is never a configuration)"
            )
        canonical = tags[0]
        if canonical not in A2FIX_VALID_ADAPTER_TAGS[repaired]:
            raise SystemExit(
                f"[a2fix] PROVENANCE ERROR: {b}'s repaired arm {repaired} carries "
                f"unregistered adapter tag {canonical!r} "
                f"(valid: {sorted(A2FIX_VALID_ADAPTER_TAGS[repaired])})"
            )
        rows_restricted = canonical in ARM2_RESTRICTED_ADAPTERS
        parity_present = any(
            r.get("arm") == ARM_MAP and r.get("adapter") == "parity-row-matched" for r in prim
        )
        out[b] = {
            "repaired_arm": repaired,
            "adapter": canonical,
            "rows_restricted": rows_restricted,
            "parity_present": parity_present,
            # HARD form (r2 blocker arm2fix-parity-partial-coverage): the §4
            # matched-budget parity duty makes the row-matched arm7 refit
            # MANDATORY for every row-restricting repair — the scoring stage
            # emits parity rows unconditionally on restricted lanes
            # (--parity-refit-arm7 + train_row_ids_sha256), so ABSENT parity
            # rows for a restricted behavior is upstream infra-incompleteness
            # and must fail the join LOUD, never degrade quietly. Unrestricted
            # behaviors are never asked for the plan-forbidden refit.
            "parity_required": bool(rows_restricted),
        }
    return out


def a2fix_sanity_records(
    new_rows: list[dict],
    behaviors,
    seeds,
    resolution: dict[str, dict],
    committed_root: Path,
    n_seeds_required: int = REGISTERED_SEED_COUNT,
) -> dict:
    """FINAL five-seed matched-regime sanity record per behavior (plan §3).

    PASS ⇔ the seed MEAN over all ``n_seeds_required`` per-seed folded-CV
    values lies inside the committed arm2 train-grid band; FAIL is reserved
    for the MEASURED out-of-band miss (every per-seed value + the miss side
    reported). An INCOMPLETE record — missing seeds, duplicate rows, or a
    missing committed band — is an INFRA COVERAGE ERROR and a loud
    SystemExit, NEVER evidence against the adapter (code-review r1 sustained
    blocker ``arm2fix-sanity-coverage-fail-open``: every reachable incomplete
    state is infra — the scorer emits sanity rows unconditionally on the
    arm2fix lane and missing seed summaries already fail the load — so a
    quiet pass=False here would feed the INCONCLUSIVE-ADAPTER kill verdict
    with a coverage bug). Non-finite sanity values / committed-band values are
    VALIDITY errors caught BEFORE band containment (codex r2 blocker
    arm2fix-nonfinite-sanity-mask: a NaN mean fails containment as
    pass=False, misclassifying an invalid statistic as a MEASURED band miss),
    and each consumed sanity row's adapter tag must match the behavior's
    resolved canonical tag (foreign provenance never enters the mask)."""
    import math

    out: dict = {}
    for b in behaviors:
        repaired_arm = resolution[b]["repaired_arm"]
        canonical_tag = resolution[b]["adapter"]
        # UNIQUE-INDEX the UNFILTERED matching rows FIRST (codex r6 BLOCKER
        # arm2fix-sanity-prefiltered-multiplicity: the old rho-None filter ran
        # before duplicate detection, so a valid row + a same-seed None
        # duplicate passed the exact seed-set gate silently). Behavior / kind /
        # arm selection is domain SCOPING; every admission check (None /
        # provenance / finite) runs strictly AFTER on the exact index.
        seed_idx = a2fix_unique_index(
            (
                r
                for r in new_rows
                if r.get("behavior") == b
                and r.get("rung_kind") == "sanity_matched_regime"
                and r.get("arm") == repaired_arm
            ),
            lambda r: int(r["seed"]),
            context=f"sanity rows ({b}, {repaired_arm})",
        )
        per_seed: dict[int, float] = {}
        for s in sorted(seed_idx):
            r = seed_idx[s]
            if r.get("rho_frozen") is None:
                continue  # absence surfaces via the exact seed-set gate below
            if str(r.get("adapter")) != canonical_tag:
                raise SystemExit(
                    f"[a2fix] PROVENANCE ERROR: sanity row ({b}, seed {s}, "
                    f"{repaired_arm}) carries adapter {r.get('adapter')!r} != the "
                    f"behavior's resolved canonical tag {canonical_tag!r}"
                )
            val = float(r["rho_frozen"])
            if not math.isfinite(val):
                raise SystemExit(
                    f"[a2fix] NON-FINITE STATISTIC: sanity row ({b}, seed {s}, "
                    f"{repaired_arm}) carries rho {val!r} — invalid upstream statistic, "
                    "never band-miss evidence"
                )
            per_seed[s] = val
        vals = committed_band_vals(committed_root, b, ARM_CTXDIR)
        want = {int(s) for s in seeds}
        if not want:
            raise SystemExit(
                "[a2fix] COVERAGE ERROR: empty seed list — the sanity coverage "
                "universe cannot be empty (a vacuous universe would pass every read)"
            )
        if not vals:
            raise SystemExit(
                f"[a2fix] COVERAGE ERROR: committed train-grid band unavailable for {b} "
                f"under {committed_root} — the sanity mask cannot be evaluated "
                "(infra gap, not a band miss)"
            )
        if any(not math.isfinite(float(v)) for v in vals):
            raise SystemExit(
                f"[a2fix] NON-FINITE STATISTIC: committed train-grid band for {b} carries "
                "a non-finite cell value — invalid band instrument, never a containment "
                "verdict input"
            )
        if set(per_seed) != want or len(per_seed) < n_seeds_required:
            raise SystemExit(
                f"[a2fix] COVERAGE ERROR: sanity record incomplete for {b} "
                f"({repaired_arm}): seeds {sorted(per_seed)} != required {sorted(want)} "
                "— infra-incomplete, never a measured band miss (check --repaired-arm "
                "resolution vs the adapter each behavior actually ran)"
            )
        band = [min(vals), max(vals)]
        mean = sum(per_seed.values()) / len(per_seed)
        rec: dict = {
            "behavior": b,
            "arm": repaired_arm,
            "band_source_arm": ARM_CTXDIR,
            "committed_band": band,
            "n_committed_cells": len(vals),
            "per_seed": {s: per_seed[s] for s in sorted(per_seed)},
            "n_seeds": len(per_seed),
            "n_seeds_required": int(n_seeds_required),
            "seed_mean": mean,
            "pass": bool(band[0] <= mean <= band[1]),
        }
        rec["miss_side"] = None if rec["pass"] else ("below" if mean < band[0] else "above")
        out[b] = rec
    return out


def a2fix_index_cells(
    new_rows: list[dict], banked_rows: list[dict], resolution: dict[str, dict]
) -> dict:
    """Index the D-pair series on the join grain (behavior, eval_rung, seed).

    Series: arm2_new (THIS round's repaired rows — each behavior indexed under
    ITS OWN resolved repaired arm, per-behavior resolution), arm4_true /
    arm7_true / arm7_shufpair (BANKED), arm7_parity (this round's row-matched
    refit, where run). A duplicate key fails loud — the join grain must be
    unique (plan §3 65/65 assert); the index is built through the single
    ``a2fix_unique_index`` primitive (codex r6 class termination)."""

    def _classify() -> list[tuple[tuple, dict]]:
        keyed: list[tuple[tuple, dict]] = []

        def _put(series: str, r: dict) -> None:
            keyed.append(
                ((str(r.get("behavior")), str(r.get("eval_rung")), int(r.get("seed")), series), r)
            )

        for r in new_rows:
            if not _is_primary_pb_row(r):
                continue
            b = str(r.get("behavior"))
            res = resolution.get(b)
            if res is None:
                continue
            if r.get("arm") == res["repaired_arm"]:
                _put("arm2_new", r)
            elif r.get("arm") == ARM_MAP and r.get("adapter") == "parity-row-matched":
                _put("arm7_parity", r)
        for r in banked_rows:
            if r.get("protocol") != "P-B" or r.get("fit") != f"P-B-holdout-{r.get('eval_rung')}":
                continue
            arm, mv = r.get("arm"), r.get("map_variant")
            if arm == ARM_MAP and mv == "true":
                _put("arm7_true", r)
            elif arm == ARM_MAP and mv == "shufpair":
                _put("arm7_shufpair", r)
            elif arm == ARM_CTX and mv == "true":
                _put("arm4_true", r)
        return keyed

    idx = a2fix_unique_index(
        _classify(),
        lambda kr: kr[0],
        context="join-grain cells (behavior, eval_rung, seed, series)",
    )
    return {k: kr[1] for k, kr in idx.items()}


def a2fix_join_assert(
    cells: dict,
    passing_behaviors,
    seeds,
    registered: dict[str, frozenset[str]] = REGISTERED_PRIMARY_RUNGS,
    *,
    resolution: dict[str, dict] | None = None,
) -> dict:
    """The plan-§3 join assert ON THE PASSING SET, denominator restated.

    Every (behavior ∈ passing set, registered rung, seed) must carry the
    arm2_new + arm7_true + arm7_shufpair series, PLUS arm7_parity for exactly
    the behaviors whose OWN repair restricted rows (HARD form, r2 blocker
    arm2fix-parity-partial-coverage: the §4 parity duty is mandatory for
    row-restricting repairs, so a restricted behavior missing its parity rows
    is a loud join failure — while unrestricted behaviors are never asked for
    the plan-forbidden refit); ANY gap is a loud SystemExit naming the first
    offenders — never a shrunken-denominator verdict. EXACT-SET (codex r4
    arm2fix-parity-registered-set-nonexact (d)): the realized key universe on
    the join series must EQUAL the registered grid — extra unregistered rungs
    or unlisted seeds are a loud EXTRAS exit (they would otherwise flow into
    a2fix_per_rung, which enumerates realized arm2 rungs, and move the D /
    parity reductions) — and ``realized_pairs`` is the TRUE counted number of
    fully-covered grid keys, never a copy of ``expected``."""
    base = ["arm2_new", "arm7_true", "arm7_shufpair"]
    passing_set = set(passing_behaviors)
    for b in sorted(passing_set):
        if not registered.get(b):
            raise SystemExit(
                f"[a2fix] REGISTRY ERROR: passing behavior {b} has an empty/missing "
                "registered rung set — the join universe cannot be empty (a vacuous "
                "universe would demand nothing)"
            )
    need_by_b = {
        b: base + (["arm7_parity"] if (resolution or {}).get(b, {}).get("parity_required") else [])
        for b in passing_behaviors
    }
    gaps = []
    for b in sorted(passing_behaviors):
        for rung in sorted(registered[b]):
            for s in seeds:
                for series in need_by_b[b]:
                    if (b, rung, int(s), series) not in cells:
                        gaps.append((b, rung, int(s), series))
    full = sum(len(v) for v in registered.values()) * len(seeds)
    expected = sum(len(registered[b]) for b in passing_behaviors) * len(seeds)
    if gaps:
        raise SystemExit(
            f"[a2fix] passing-set join INCOMPLETE: {len(gaps)} missing series cells "
            f"(expected {expected} keys x per-behavior series {need_by_b}); "
            f"first offenders: {gaps[:6]}"
        )
    seed_set = {int(s) for s in seeds}
    extras = sorted(
        k
        for k in cells
        if k[0] in passing_set
        and k[3] in need_by_b[k[0]]
        and (k[1] not in registered[k[0]] or int(k[2]) not in seed_set)
    )
    if extras:
        raise SystemExit(
            f"[a2fix] passing-set join EXTRAS: {len(extras)} realized cells on the join "
            f"series OUTSIDE the registered (rung x seed) grid — first offenders: "
            f"{extras[:6]} (an unregistered rung / unlisted seed must never enter the "
            "D or parity reductions)"
        )
    realized = sum(
        1
        for b in passing_behaviors
        for rung in registered[b]
        for s in seeds
        if all((b, rung, int(s), series) in cells for series in need_by_b[b])
    )
    return {
        "expected_pairs": expected,
        "realized_pairs": realized,
        "restated_from": full,
        "series_required_by_behavior": {b: need_by_b[b] for b in sorted(need_by_b)},
        "passing_behaviors": sorted(passing_behaviors),
        "assert": f"{realized}/{expected} join keys present + unique + exact "
        "(no extras) on (behavior, fit, eval_rung, seed)",
    }


def a2fix_parity_hash_assert(
    cells: dict, behaviors, seeds, registered, resolution: dict[str, dict] | None = None
) -> int:
    """Matched-budget parity: the repaired-arm and arm7-parity rows must carry
    IDENTICAL train-row id-hashes + counts per join key (plan §4 Must-Fix).

    Hardened (r1 CONCERN ``arm2fix-parity-currency-fail-open``): a parity
    pair whose hash OR count is missing/None on EITHER side is a loud exit —
    two absent hashes must never satisfy the equality. Scope: behaviors whose
    resolution requires parity (all pairs incidentally present are still
    checked when resolution is not supplied)."""
    n = 0
    for b in sorted(behaviors):
        if resolution is not None and not resolution.get(b, {}).get("parity_present"):
            continue
        for rung in sorted(registered[b]):
            for s in seeds:
                a2 = cells.get((b, rung, int(s), "arm2_new"))
                a7 = cells.get((b, rung, int(s), "arm7_parity"))
                if a2 is None or a7 is None:
                    continue
                cur = [
                    a2.get("train_row_ids_sha256"),
                    a7.get("train_row_ids_sha256"),
                    a2.get("train_rows_n"),
                    a7.get("train_rows_n"),
                ]
                if any(c is None for c in cur):
                    raise SystemExit(
                        f"[a2fix] PARITY CURRENCY MISSING at ({b}, {rung}, seed {s}): "
                        f"arm2 sha/n = {cur[0]}/{cur[2]}, arm7-parity sha/n = "
                        f"{cur[1]}/{cur[3]} — a None-for-None match is not parity evidence"
                    )
                if cur[0] != cur[1] or cur[2] != cur[3]:
                    raise SystemExit(
                        f"[a2fix] PARITY VIOLATION at ({b}, {rung}, seed {s}): "
                        f"arm2 {cur[2]} rows / {str(cur[0])[:12]} != arm7-parity "
                        f"{cur[3]} rows / {str(cur[1])[:12]}"
                    )
                n += 1
    return n


def a2fix_per_rung(cells: dict, behaviors, seeds, sanity: dict) -> list[dict]:
    """Per-rung table rows: the plan-§4 D2 columns × per-seed values + D reads.

    D(r, s) = ρ(arm7 true, banked) − ρ(arm2 repaired); the shuffle-margin
    position read is arm7_shufpair − arm2; D_parity joins when the
    row-matched refit ran. Sanity-excluded behaviors stay TABULATED (marked
    ``excluded_by_sanity``) — they are excluded from the median/flagships by
    the lattice, never silently dropped from the table."""
    rungs_by_b = {
        b: sorted({k[1] for k in cells if k[0] == b and k[3] == "arm2_new"}) for b in behaviors
    }
    out: list[dict] = []
    for b in behaviors:
        for rung in rungs_by_b[b]:
            entry: dict = {
                "behavior": b,
                "eval_rung": rung,
                "flagship": (b, rung) in set(FLAGSHIPS),
                "excluded_by_sanity": not sanity.get(b, {}).get("pass", False),
            }
            series_vals: dict[str, dict[int, float]] = {}
            for series in A2FIX_SERIES:
                vals = {
                    int(s): float(cells[(b, rung, int(s), series)]["rho_frozen"])
                    for s in seeds
                    if (b, rung, int(s), series) in cells
                }
                if vals:
                    series_vals[series] = vals
            entry["per_seed"] = {
                k: {str(s): v for s, v in sv.items()} for k, sv in series_vals.items()
            }
            a2 = series_vals.get("arm2_new", {})
            seeds_ok = sorted(
                set(a2) & set(series_vals.get("arm7_true", {})) & {int(s) for s in seeds}
            )
            entry["seeds_used"] = seeds_ok
            entry["complete"] = len(seeds_ok) == len(list(seeds)) and len(seeds_ok) > 0
            if seeds_ok:
                a7 = series_vals["arm7_true"]
                entry["arm2_repaired"] = seed_tci([a2[s] for s in seeds_ok])
                entry["arm7_true"] = seed_tci([a7[s] for s in seeds_ok])
                entry["D"] = seed_tci([a7[s] - a2[s] for s in seeds_ok])
                sh = series_vals.get("arm7_shufpair", {})
                if all(s in sh for s in seeds_ok):
                    entry["arm7_shufpair"] = seed_tci([sh[s] for s in seeds_ok])
                    entry["shuf_margin_pos"] = seed_tci([sh[s] - a2[s] for s in seeds_ok])
                a4 = series_vals.get("arm4_true", {})
                if all(s in a4 for s in seeds_ok):
                    entry["arm4_true"] = seed_tci([a4[s] for s in seeds_ok])
                pr = series_vals.get("arm7_parity", {})
                if all(s in pr for s in seeds_ok):
                    entry["arm7_parity"] = seed_tci([pr[s] for s in seeds_ok])
                    entry["D_parity"] = seed_tci([pr[s] - a2[s] for s in seeds_ok])
                a2_adapters = {
                    str(cells[(b, rung, int(s), "arm2_new")].get("adapter")) for s in seeds_ok
                }
                entry["adapter"] = sorted(a2_adapters)
            out.append(entry)
    return out


def a2fix_load_new_preds(root: Path, b: str, seed: int, rung: str, repaired_arm: str):
    """This round's repaired-arm per-context preds for ONE (behavior, seed, rung).

    The repaired pass may write a tagged preds file (P-B-holdout-<rung>.a2r/
    .a2qr.jsonl) — glob the candidates and require EXACTLY ONE file to supply
    the repaired arm's rows for this rung (ambiguity is a named gap, never a
    silent pick)."""
    matching: list[dict] = []
    src_files: set[str] = set()
    preds_dir = root / b / f"seed{seed}" / "transfer_preds"
    for p in sorted(preds_dir.glob(f"P-B-holdout-{rung}*.jsonl")):
        found_here = False
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("arm") != repaired_arm or str(r.get("rung")) != rung:
                continue
            found_here = True
            r["_src"] = p.name
            matching.append(r)
        if found_here:
            src_files.add(p.name)
    if not matching:
        return None, None, None, f"no {repaired_arm} preds rows under {preds_dir}"
    if len(src_files) > 1:
        return None, None, None, f"ambiguous preds sources for {rung}: {sorted(src_files)}"
    # UNIQUE-INDEX context ids on the single-source rows (codex r6 BLOCKER
    # arm2fix-context-preds-duplicate-universe: hits[cid] = ... last-write-wins
    # let a corrupt sidecar silently pick one of two scores per context) —
    # duplicates are data corruption, loud, never a quiet gap.
    by_cid = a2fix_unique_index(
        matching,
        lambda r: str(r["context_id"]),
        context=f"new preds context ids ({b}/seed{seed}/{rung}, {sorted(src_files)[0]})",
    )
    hits: dict[str, float] = {}
    dv_by_ctx: dict[str, float] = {}
    grp_by_ctx: dict[str, str] = {}
    for cid in sorted(by_cid):
        r = by_cid[cid]
        if r.get("group") is None:
            return None, None, None, f"preds row missing 'group' ({r['_src']}, ctx {cid})"
        hits[cid] = float(r["score"])
        dv_by_ctx[cid] = float(r["dv"])
        grp_by_ctx[cid] = str(r["group"])
    return hits, dv_by_ctx, grp_by_ctx, f"{len(hits)} contexts from {sorted(src_files)[0]}"


def a2fix_ctx_ci(
    arm2fix_root: Path,
    banked_root: Path,
    b: str,
    rung: str,
    seeds,
    repaired_arm: str,
    *,
    n_boot: int,
    rng,
) -> tuple[dict | None, str]:
    """Paired GROUP-level context bootstrap on D (plan §4 D2: 2,000 draws via
    the batched ``group_bootstrap_rhos`` — never per-draw): resample the
    holdout rung's contexts by group; recompute ρ(arm2 repaired) from THIS
    round's preds and ρ(arm7 true) from the BANKED preds per seed; D per
    draw; seed-mean per draw → percentile CI."""
    import numpy as np

    new_scores: dict[int, dict[str, float]] = {}
    banked_scores: dict[int, dict[str, float]] = {}
    dv_by_ctx: dict[str, float] = {}
    grp_by_ctx: dict[str, str] = {}
    for s in seeds:
        hits, dv_n, grp_n, note = a2fix_load_new_preds(arm2fix_root, b, int(s), rung, repaired_arm)
        if hits is None:
            return None, f"seed {s}: {note}"
        new_scores[int(s)] = hits
        # cross-seed DV/group metadata must AGREE, never last-write-wins merge
        # (codex r6 BLOCKER arm2fix-context-preds-duplicate-universe): the
        # paired bootstrap uses ONE dv and ONE group per context across seeds.
        a2fix_merge_consistent(dv_by_ctx, dv_n, what="dv", context=f"{b}/{rung} cross-seed")
        a2fix_merge_consistent(grp_by_ctx, grp_n, what="group", context=f"{b}/{rung} cross-seed")
        p = banked_root / b / f"seed{s}" / "transfer_preds" / f"P-B-holdout-{rung}.jsonl"
        if not p.exists():
            return None, f"banked preds missing: {p}"
        banked_rows = [
            r
            for r in (json.loads(line) for line in p.read_text().splitlines())
            if r.get("arm") == ARM_MAP and str(r.get("rung")) == rung
        ]
        if not banked_rows:
            return None, f"banked arm7 preds empty for {rung} seed {s}"
        per = {
            cid: float(r["score"])
            for cid, r in a2fix_unique_index(
                banked_rows,
                lambda r: str(r["context_id"]),
                context=f"banked preds context ids ({b}/seed{s}/{rung})",
            ).items()
        }
        banked_scores[int(s)] = per
    ctx_sets = [set(v) for v in new_scores.values()] + [set(v) for v in banked_scores.values()]
    base = ctx_sets[0]
    if any(cs != base for cs in ctx_sets):
        return None, "context sets differ across series (whitening/join drift — named gap)"
    order = sorted(base)
    keys = [("new", int(s)) for s in seeds] + [("banked", int(s)) for s in seeds]
    mat = np.stack(
        [
            np.asarray([(new_scores if k[0] == "new" else banked_scores)[k[1]][c] for c in order])
            for k in keys
        ]
    )
    dv = np.asarray([dv_by_ctx[c] for c in order])
    groups = [grp_by_ctx[c] for c in order]
    rhos, n_groups = group_bootstrap_rhos(mat, dv, groups, n_boot=n_boot, rng=rng)
    pos = {k: i for i, k in enumerate(keys)}
    d_draws = np.stack(
        [rhos[pos[("banked", int(s))]] - rhos[pos[("new", int(s))]] for s in seeds]
    ).mean(axis=0)
    ci = [float(np.nanquantile(d_draws, 0.025)), float(np.nanquantile(d_draws, 0.975))]
    return (
        {"D_ctx_ci": ci, "n_groups": n_groups, "n_boot": int(n_boot), "n_ctx": len(order)},
        f"n_ctx={len(order)}",
    )


def a2fix_assert_finite(per_rung: list[dict]) -> int:
    """Every statistic the lattice consumes must be FINITE (r1 CONCERN
    ``arm2fix-nonfinite-verdict``): a NaN/inf rho or D falling through the
    median would mint MAP-ADVANTAGE-NOT-SHOWN with false reason prose — the
    declared-exhaustive lattice is not NaN-exhaustive, so validate before
    evaluation. Sanity-excluded rows are skipped (they never enter the
    median). Returns the number of values checked."""
    import math

    n = 0
    for r in per_rung:
        if r.get("excluded_by_sanity"):
            continue
        loc = (r.get("behavior"), r.get("eval_rung"))
        for key in ("arm2_repaired", "arm7_true", "arm7_shufpair", "arm4_true", "D", "D_parity"):
            blk = r.get(key)
            if not isinstance(blk, dict):
                continue
            vals = [blk.get("mean")] + list(blk.get("tci") or []) + list(blk.get("per_seed") or [])
            for v in vals:
                if v is None:
                    continue
                if not math.isfinite(float(v)):
                    raise SystemExit(
                        f"[a2fix] NON-FINITE STATISTIC at {loc}: {key} carries {v!r} — "
                        "invalid input to the lattice (degenerate scores upstream), "
                        "never a verdict"
                    )
                n += 1
        for v in r.get("D_ctx_ci") or []:
            if not math.isfinite(float(v)):
                raise SystemExit(f"[a2fix] NON-FINITE STATISTIC at {loc}: D_ctx_ci carries {v!r}")
            n += 1
    return n


def a2fix_ctx_gap_assert(per_rung: list[dict], *, skipped: bool) -> None:
    """Loud exit on passing-set ctx-CI gaps (r1 CONCERN
    ``arm2fix-context-bootstrap-gap`` fix-round rec): once the paired context
    bootstrap is requested (not --skip-ctx-bootstrap), every complete
    passing-set rung must carry its D_ctx_ci — a missing/ambiguous preds
    input is a named infra gap, never a note under a rendered verdict."""
    if skipped:
        return
    gaps = [
        (r["behavior"], r["eval_rung"], r.get("ctx_bootstrap_note", "no note"))
        for r in per_rung
        if not r.get("excluded_by_sanity") and r.get("complete") and r.get("D_ctx_ci") is None
    ]
    if gaps:
        raise SystemExit(
            f"[a2fix] CTX-BOOTSTRAP GAP on the passing set: {len(gaps)} rung(s) missing "
            f"the paired context CI — {gaps[:4]} (fix the preds inputs or pass "
            "--skip-ctx-bootstrap deliberately)"
        )


def a2fix_lattice(
    per_rung: list[dict],
    sanity: dict,
    *,
    resolution: dict[str, dict],
    registered: dict[str, frozenset[str]] = REGISTERED_PRIMARY_RUNGS,
    flagships=FLAGSHIPS,
) -> dict:
    """The amended §3 leg-2 lattice — DISJOINT + exhaustive, evaluated over
    the SANITY-PASSING behavior set (verbatim vocabulary, plan v26):

    - INCONCLUSIVE-ADAPTER ⇔ sanity fails on ≥2 behaviors (evaluated FIRST).
    - MAP-BEATS-CONTEXT-DIRECTION ⇔ sanity passes on ≥2 behaviors AND
      median D > 0 over the passing set AND ≥1 passing-set flagship rung has
      BOTH CIs (seed t-CI, paired group-level context CI) clear of 0 above
      AND — when ANY passing behavior's repair restricted its fit rows — the
      row-matched parity read also shows D > 0 (median over exactly the
      ROWS-RESTRICTED passing behaviors' rungs; per-behavior resolution).
    - MAP-ADVANTAGE-NOT-SHOWN ⇔ sanity passes on ≥2 behaviors AND
      median D ≤ 0 over the passing set (a FAILURE TO DEMONSTRATE the map's
      advantage — NOT evidence the context-side repair suffices; a bare
      median sign carries ~0.5 null false-confirm).
    - WEAK-MIXED ⇔ otherwise.
    A sanity-FAILED behavior is recorded per-behavior INDETERMINATE-ADAPTER;
    its rungs are excluded from the median, the flagship set, and the join
    denominator (restated upstream)."""
    import numpy as np

    passing = sorted(b for b, rec in sanity.items() if rec.get("pass"))
    failing = sorted(b for b, rec in sanity.items() if not rec.get("pass"))
    per_behavior = {b: "INDETERMINATE-ADAPTER" for b in failing}
    restricted_passing = sorted(b for b in passing if resolution.get(b, {}).get("rows_restricted"))
    out: dict = {
        "passing_behaviors": passing,
        "failing_behaviors": failing,
        "per_behavior_adapter_verdicts": per_behavior,
        "resolution": {b: dict(resolution.get(b, {})) for b in sorted(resolution)},
        "rows_restricted_behaviors": restricted_passing,
        "rows_restricted": bool(restricted_passing),
        "parity_present": bool(
            any(resolution.get(b, {}).get("parity_present") for b in resolution)
        ),
    }
    if len(passing) < 2:
        out["verdict"] = "INCONCLUSIVE-ADAPTER"
        out["reason"] = (
            f"matched-regime sanity failed on {len(failing)} behavior(s) "
            f"({failing}) — fewer than 2 behaviors pass; no arm2 comparison is citable"
        )
        return out
    import math

    # STRUCTURAL exact-universe indexing via the SINGLE primitive (codex r5
    # arm2fix-parity-realized-multiset-prefilter; codex r6 class-termination
    # arm2fix-lattice-d-admission-universe): the realized universe is the
    # UNFILTERED passing-set per_rung input, uniquely indexed through
    # a2fix_unique_index (duplicates loud; passing-set membership is domain
    # SCOPING, not admission), and EVERY admission — primary D and parity
    # alike — derives from this index against the REGISTERED universe through
    # the single a2fix_exact_coverage comparator. A registered rung without a
    # finite D is a NAMED coverage failure, never a silent denominator shrink.
    row_by_key = a2fix_unique_index(
        (r for r in per_rung if r["behavior"] in passing),
        lambda r: (r["behavior"], r["eval_rung"]),
        context="lattice passing-set per-rung rows (behavior, eval_rung)",
    )
    for b in passing:
        if not registered.get(b):
            raise SystemExit(
                f"[a2fix] REGISTRY ERROR: passing behavior {b} has an empty/missing "
                "registered rung set — the primary-D coverage universe cannot be "
                "empty (a vacuous universe would pass every read)"
            )
    expected_d = {(b, rung) for b in passing for rung in registered[b]}
    d_by_key: dict[tuple, float] = {}
    for k in sorted(row_by_key):
        m = row_by_key[k].get("D", {}).get("mean")
        if m is not None and math.isfinite(float(m)):
            d_by_key[k] = float(m)
    d_cov = a2fix_exact_coverage(
        expected_d, d_by_key, row_by_key, context="primary-D registered coverage"
    )
    out["d_read"] = {
        "n_rungs": d_cov["n_admitted"],
        "n_rungs_expected": d_cov["n_expected"],
        "coverage_complete": d_cov["complete"],
        "uncovered_rungs": [list(u) for u in d_cov["missing"]],
        "extra_unregistered_rungs": [list(u) for u in d_cov["extra"]],
    }
    if not d_cov["complete"]:
        # codex r7 BLOCKER arm2fix-d-read-postcoverage-partial-reduction: the
        # coverage branch runs IMMEDIATELY after d_read is constructed —
        # BEFORE the median, flagship, and parity reductions — so NO partial
        # statistic is ever computed, stored, or rendered over an incomplete
        # registered universe. Every median/flagship field is explicitly
        # nulled (schema-stable for consumers); parity_read is not computed.
        out["median_D_passing_set"] = None
        out["per_behavior_median_D"] = {}
        out["n_rungs_in_median"] = 0
        out["flagships_in_passing_set"] = []
        out["verdict"] = "WEAK-MIXED"
        out["reason"] = (
            "the primary-D read is INCOMPLETE over the passing behaviors' REGISTERED "
            f"rungs (uncovered rungs: {out['d_read']['uncovered_rungs']}; extra "
            f"unregistered rungs: {out['d_read']['extra_unregistered_rungs']}) — a "
            "registered rung without a finite D can never silently shrink the median "
            "denominator, and an unregistered realized rung can never join it; no "
            "median, flagship, or parity statistic is rendered over a partial universe"
        )
        return out
    reg_admitted = [k for k in sorted(expected_d) if k in d_by_key]
    rows = [row_by_key[k] for k in reg_admitted]
    d_means = [d_by_key[k] for k in reg_admitted]
    med = float(np.median(d_means)) if d_means else None
    per_b_median = {
        b: float(np.median([d_by_key[k] for k in reg_admitted if k[0] == b]))
        for b in passing
        if any(k[0] == b for k in reg_admitted)
    }
    out["median_D_passing_set"] = med
    out["per_behavior_median_D"] = per_b_median
    out["n_rungs_in_median"] = len(rows)
    flag_rows = [r for r in rows if (r["behavior"], r["eval_rung"]) in set(flagships)]
    flag_ok = []
    for r in flag_rows:
        tci = r.get("D", {}).get("tci")
        ctx = r.get("D_ctx_ci")
        flag_ok.append(
            {
                "behavior": r["behavior"],
                "eval_rung": r["eval_rung"],
                "seed_tci_above_zero": bool(tci is not None and tci[0] > 0),
                "ctx_ci_above_zero": bool(ctx is not None and ctx[0] > 0),
                "both_clear": bool(
                    tci is not None and tci[0] > 0 and ctx is not None and ctx[0] > 0
                ),
            }
        )
    out["flagships_in_passing_set"] = flag_ok
    parity_read = None
    if restricted_passing:
        # EXACT-SET + FINITE parity coverage (r2/r3/r4 lineage; r6: routed
        # through the SAME a2fix_exact_coverage comparator as the primary-D
        # read — one implementation of registered-set coverage exists):
        # only math.isfinite D_parity means are admitted; realized keys come
        # from the unfiltered unique index (a D-less row at an unregistered
        # rung is still a realized extra); dp is computed exclusively over
        # registered keys in registered-key order.
        expected_keys = {(b, rung) for b in restricted_passing for rung in registered[b]}
        realized_keys = {k for k in row_by_key if k[0] in restricted_passing}
        dp_by_key: dict[tuple, float] = {}
        for k in sorted(realized_keys):
            m = row_by_key[k].get("D_parity", {}).get("mean")
            if m is not None and math.isfinite(float(m)):
                dp_by_key[k] = float(m)
        p_cov = a2fix_exact_coverage(
            expected_keys, dp_by_key, realized_keys, context="parity registered coverage"
        )
        dp = [dp_by_key[k] for k in sorted(expected_keys) if k in dp_by_key]
        # codex r8 BLOCKER arm2fix-parity-postcoverage-partial-reduction —
        # mirror of the primary-D treatment: the parity median is REDUCED only
        # over a COMPLETE registered universe; on incomplete coverage it is
        # explicitly nulled (the coverage counters + uncovered/extra lists ARE
        # the honest read) so no subset statistic persists or renders.
        med_parity = float(np.median(dp)) if (p_cov["complete"] and dp) else None
        parity_read = {
            "behaviors": restricted_passing,
            "n_rungs": len(dp),
            "n_rungs_expected": p_cov["n_expected"],
            "coverage_complete": p_cov["complete"],
            "uncovered_rungs": [list(u) for u in p_cov["missing"]],
            "extra_unregistered_rungs": [list(u) for u in p_cov["extra"]],
            "median_D_parity": med_parity,
            "positive": bool(med_parity is not None and med_parity > 0),
            "note": "estimand-parity read: arm7 refit on the IDENTICAL training-row ids "
            "as the repaired arm2, over exactly the rows-restricted passing behaviors' "
            "REGISTERED rungs (plan §4 matched-budget parity duty; per-behavior "
            "resolution; positive requires EXACT two-way coverage of the registered "
            "universe with FINITE values — realized rows never define their own "
            "denominator, and extras never move the median)",
        }
        out["parity_read"] = parity_read
    if med is None:
        out["verdict"] = "WEAK-MIXED"
        out["reason"] = "no complete passing-set rungs carry a D read"
        return out
    if med > 0:
        parity_ok = (not restricted_passing) or bool(parity_read and parity_read["positive"])
        if any(f["both_clear"] for f in flag_ok) and parity_ok:
            out["verdict"] = "MAP-BEATS-CONTEXT-DIRECTION"
            return out
        out["verdict"] = "WEAK-MIXED"
        if not any(f["both_clear"] for f in flag_ok):
            parity_why = "no passing-set flagship holds BOTH CIs clear of 0"
        elif parity_read and not parity_read["coverage_complete"]:
            parity_why = (
                "the row-matched parity read is INCOMPLETE over the rows-restricted "
                f"passing behaviors (uncovered rungs: {parity_read['uncovered_rungs']}; "
                f"extra unregistered rungs: {parity_read['extra_unregistered_rungs']}) "
                "— exact registered-universe coverage is required when a repair "
                "restricted arm2's fit rows"
            )
        else:
            parity_why = (
                "the row-matched parity read does not show D > 0 over the "
                f"rows-restricted passing behaviors {restricted_passing} "
                "(required when a repair restricted arm2's fit rows)"
            )
        out["reason"] = "median D > 0 but " + parity_why
        return out
    out["verdict"] = "MAP-ADVANTAGE-NOT-SHOWN"
    out["reason"] = (
        "median D <= 0 over the passing set — the map's advantage is NOT demonstrated "
        "against this comparator. This is a failure to demonstrate, NOT evidence the "
        "context-side repair suffices (an affirmative sufficiency claim would need its "
        "own uncertainty-backed read; a bare median sign carries ~0.5 null false-confirm)."
    )
    return out


def build_arm2fix_table(args) -> dict:
    import numpy as np

    seeds = [int(s) for s in args.seeds]
    new_rows, new_missing = load_claim4_rows(args.arm2fix_root, args.behaviors, seeds)
    banked_rows, banked_missing = load_claim4_rows(args.claim4_root, args.behaviors, seeds)
    if new_missing:
        raise SystemExit(f"[a2fix] missing NEW seed summaries: {new_missing[:4]}")
    if banked_missing:
        raise SystemExit(f"[a2fix] missing BANKED seed summaries: {banked_missing[:4]}")

    # PER-BEHAVIOR repair resolution (r1 sustained blocker
    # arm2fix-mixed-adapter-fold): the ladder runs per behavior, so
    # {repaired_arm, adapter, rows_restricted, parity_required} resolve from
    # each behavior's OWN rows; --repaired-arm overrides per behavior.
    resolution = a2fix_resolve_repairs(new_rows, args.behaviors, args.repaired_arm_overrides)
    for b in sorted(resolution):
        _log(f"[a2fix resolve] {b}: {resolution[b]}")

    sanity = a2fix_sanity_records(
        new_rows, args.behaviors, seeds, resolution, args.committed_train_root
    )
    passing = sorted(b for b, rec in sanity.items() if rec.get("pass"))
    cells = a2fix_index_cells(new_rows, banked_rows, resolution)
    join = None
    if len(passing) >= 2:
        join = a2fix_join_assert(cells, passing, seeds, resolution=resolution)
    n_parity_checked = a2fix_parity_hash_assert(
        cells, passing or args.behaviors, seeds, REGISTERED_PRIMARY_RUNGS, resolution
    )
    per_rung = a2fix_per_rung(cells, args.behaviors, seeds, sanity)

    # paired group-level context bootstrap on D (passing-set rungs)
    for entry in per_rung:
        b, rung = entry["behavior"], entry["eval_rung"]
        entry["D_ctx_ci"] = None
        if entry["excluded_by_sanity"] or args.skip_ctx_bootstrap or not entry.get("complete"):
            continue
        rng = np.random.default_rng([1739, 23, zlib.crc32(f"{b}|{rung}".encode())])
        ci, note = a2fix_ctx_ci(
            args.arm2fix_root,
            args.claim4_root,
            b,
            rung,
            entry["seeds_used"],
            resolution[b]["repaired_arm"],
            n_boot=args.n_boot,
            rng=rng,
        )
        if ci is None:
            entry["ctx_bootstrap_note"] = note
        else:
            entry["D_ctx_ci"] = ci["D_ctx_ci"]
            entry["ctx_bootstrap"] = {k: ci[k] for k in ("n_groups", "n_boot", "n_ctx")}
        _log(f"[a2fix ctx-boot] {b}/{rung}: {note}")

    # hardening gates BEFORE the lattice (r1 CONCERN fix-round recs): a
    # passing-set ctx-CI gap and any non-finite statistic are loud exits.
    a2fix_ctx_gap_assert(per_rung, skipped=bool(args.skip_ctx_bootstrap))
    a2fix_assert_finite(per_rung)

    verdict = a2fix_lattice(
        per_rung, sanity, resolution=resolution, registered=REGISTERED_PRIMARY_RUNGS
    )

    # P4 direction-stability cosines ride NEXT TO any MAP-BEATS output
    p4 = None
    if args.d0_p4 and Path(args.d0_p4).exists():
        p4 = json.loads(Path(args.d0_p4).read_text())
    elif verdict.get("verdict") == "MAP-BEATS-CONTEXT-DIRECTION":
        verdict["p4_gap"] = (
            f"P4 direction-stability file absent ({args.d0_p4}) — the MAP-BEATS "
            "narration requires the cosines reported beside it (plan §4)"
        )

    # secondary diagnostic only (plan §4): the OLD pvsynth transfer read
    pvsynth_secondary = {
        b: arm2_sanity_band(args.committed_train_root, b, new_rows, seeds) for b in args.behaviors
    }
    return {
        "meta": {
            "mode": "arm2fix",
            "generated_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_head_sha(),
            "seeds": seeds,
            "n_boot": args.n_boot,
            "arm2fix_root": str(args.arm2fix_root),
            "banked_root": str(args.claim4_root),
            "resolution": {b: dict(resolution[b]) for b in sorted(resolution)},
            "repaired_arm_overrides": dict(args.repaired_arm_overrides or {}),
            "n_parity_hash_checks": n_parity_checked,
            "flagships": [list(f) for f in FLAGSHIPS],
        },
        "sanity": sanity,
        "join": join
        or {
            "note": "passing set < 2 behaviors — no join denominator to assert "
            "(INCONCLUSIVE-ADAPTER path)"
        },
        "per_rung": per_rung,
        "verdict": verdict,
        "p4_direction_stability": p4 or f"not staged ({args.d0_p4})",
        "pvsynth_transfer_secondary_diagnostic": pvsynth_secondary,
    }


def write_arm2fix_markdown(table: dict, path: Path) -> None:
    v = table["verdict"]
    lines = [
        "# arm2fix — repaired context-direction vs banked mapped-answer probe",
        "",
        f"- generated: {table['meta']['generated_ts']} @ {table['meta']['git_commit']}",
        "- per-behavior repair resolution: "
        + "; ".join(
            f"{b}: `{res['repaired_arm']}` (adapter {res['adapter']}, "
            f"restricted={res['rows_restricted']}, parity={res['parity_present']})"
            for b, res in sorted(table["meta"]["resolution"].items())
        ),
        f"- verdict: **{v.get('verdict')}**" + (f" — {v['reason']}" if v.get("reason") else ""),
        f"- passing set: {v.get('passing_behaviors')}; excluded: {v.get('failing_behaviors')}",
    ]
    if v.get("d_read"):
        dr = v["d_read"]
        lines.append(
            f"- primary-D registered coverage: {dr['n_rungs']}/{dr['n_rungs_expected']} "
            f"(complete: {dr['coverage_complete']}; uncovered: {dr['uncovered_rungs']}; "
            f"extra unregistered: {dr['extra_unregistered_rungs']})"
        )
    if v.get("median_D_passing_set") is not None:
        lines.append(
            f"- median D (passing set, {v['n_rungs_in_median']} rungs): "
            f"{v['median_D_passing_set']:+.4f}; per-behavior medians: "
            + ", ".join(f"{b} {m:+.4f}" for b, m in v.get("per_behavior_median_D", {}).items())
        )
    if v.get("parity_read"):
        pr = v["parity_read"]
        lines.append(
            f"- parity registered coverage: {pr['n_rungs']}/{pr['n_rungs_expected']} "
            f"(complete: {pr['coverage_complete']}; uncovered: {pr['uncovered_rungs']}; "
            f"extra unregistered: {pr['extra_unregistered_rungs']})"
        )
        if pr["median_D_parity"] is not None:
            lines.append(
                f"- parity read (row-matched arm7): median D_parity "
                f"{pr['median_D_parity']:+.4f} over {pr['n_rungs']} rungs"
            )
    lines += [
        "",
        "| behavior | rung | ★ | excl | arm2 rep. | arm7 true | arm7 shuf | arm4 | D mean | "
        "D t-CI | D ctx-CI | D_parity | seeds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _fmt(x):
        return "—" if x is None else f"{x:+.4f}"

    def _fmtci(ci):
        return "—" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"

    def _mean(entry, key):
        return _fmt((entry.get(key) or {}).get("mean"))

    for r in table["per_rung"]:
        lines.append(
            f"| {r['behavior']} | {r['eval_rung']} | {'★' if r['flagship'] else ''} | "
            f"{'x' if r['excluded_by_sanity'] else ''} | {_mean(r, 'arm2_repaired')} | "
            f"{_mean(r, 'arm7_true')} | {_mean(r, 'arm7_shufpair')} | {_mean(r, 'arm4_true')} | "
            f"{_mean(r, 'D')} | {_fmtci((r.get('D') or {}).get('tci'))} | "
            f"{_fmtci(r.get('D_ctx_ci'))} | {_mean(r, 'D_parity')} | {len(r['seeds_used'])} |"
        )
    lines += ["", "## Five-seed sanity records", ""]
    for b, rec in table["sanity"].items():
        vals = ", ".join(f"s{s}={x:+.4f}" for s, x in rec.get("per_seed", {}).items())
        lines.append(
            f"- {b}: {'PASS' if rec.get('pass') else 'FAIL'}"
            + (f" ({rec.get('miss_side')} band)" if rec.get("miss_side") else "")
            + f" — mean {rec.get('seed_mean', float('nan')):+.4f} vs band "
            f"{rec.get('committed_band')} | {vals}"
            + (f" | reason: {rec['reason']}" if rec.get("reason") else "")
        )
    path.write_text("\n".join(lines) + "\n")


def write_arm2fix_note(table: dict, path: Path) -> None:
    """Diagnosis note — numbers + coverage ONLY (claims stay Thomas-authored)."""
    v = table["verdict"]
    m = table["meta"]
    lines = [
        "# arm2 comparator repair (arm2fix) — numbers note (task #1739, leg 2)",
        "",
        "Numbers + coverage only; claims stay with the writeup author.",
        "",
        "- per-behavior repair resolution: "
        + "; ".join(
            f"{b}: {res['repaired_arm']} (adapter {res['adapter']}, "
            f"restricted={res['rows_restricted']}, parity={res['parity_present']})"
            for b, res in sorted(m["resolution"].items())
        ),
        f"- lattice verdict: **{v.get('verdict')}**"
        + (f" ({v['reason']})" if v.get("reason") else ""),
        f"- sanity passing set: {v.get('passing_behaviors')} "
        f"(excluded: {v.get('failing_behaviors')} -> per-behavior INDETERMINATE-ADAPTER)",
    ]
    join = table.get("join") or {}
    if join.get("expected_pairs") is not None:
        lines.append(
            f"- join denominator: {join['realized_pairs']}/{join['expected_pairs']} "
            f"(restated from {join['restated_from']}; per-behavior series "
            f"{join['series_required_by_behavior']})"
        )
    if v.get("d_read"):
        dr = v["d_read"]
        lines.append(
            f"- primary-D registered coverage: {dr['n_rungs']}/{dr['n_rungs_expected']} "
            f"(complete: {dr['coverage_complete']}; uncovered: {dr['uncovered_rungs']}; "
            f"extra unregistered: {dr['extra_unregistered_rungs']})"
        )
    if v.get("median_D_passing_set") is not None:
        lines.append(
            f"- median D over the passing set: {v['median_D_passing_set']:+.4f} "
            f"({v['n_rungs_in_median']} rungs); per-behavior: "
            + ", ".join(f"{b} {x:+.4f}" for b, x in v.get("per_behavior_median_D", {}).items())
        )
    for f in v.get("flagships_in_passing_set", []):
        lines.append(
            f"- flagship {f['behavior']}/{f['eval_rung']}: seed t-CI clear of 0: "
            f"{f['seed_tci_above_zero']}; ctx CI clear of 0: {f['ctx_ci_above_zero']}"
        )
    if v.get("parity_read"):
        pr = v["parity_read"]
        lines.append(
            f"- parity registered coverage: {pr.get('n_rungs')}/{pr.get('n_rungs_expected')} "
            f"(complete: {pr.get('coverage_complete')}; uncovered: {pr.get('uncovered_rungs')}; "
            f"extra unregistered: {pr.get('extra_unregistered_rungs')})"
        )
        if pr.get("median_D_parity") is not None:
            lines.append(
                f"- parity read: median D_parity {pr.get('median_D_parity')} "
                f"over {pr.get('n_rungs')} rungs (positive: {pr.get('positive')})"
            )
    for b, rec in table["sanity"].items():
        lines.append(
            f"- sanity {b}: per-seed {rec.get('per_seed')} | mean "
            f"{rec.get('seed_mean')} vs band {rec.get('committed_band')} | "
            f"pass={rec.get('pass')} miss_side={rec.get('miss_side')}"
        )
    if isinstance(table.get("p4_direction_stability"), dict):
        for b, blk in table["p4_direction_stability"].get("behaviors", {}).items():
            lines.append(
                f"- P4 {b}: cos(v1 vs restricted) per seed "
                f"{blk.get('cos_v1_vs_restricted')}; across-seed raw-space cosine "
                f"matrices in the staged p4 JSON (layer {blk.get('frozen_layer')})"
            )
    else:
        lines.append(f"- P4 direction stability: {table.get('p4_direction_stability')}")
    lines.append("")
    lines.append("Full table: `eval_results/issue_1739/claim4_controls/arm2fix/arm2fix_table.json`")
    path.write_text("\n".join(lines) + "\n")


def render_arm2fix_figures(table: dict, fig_dir: Path) -> list[str]:
    """HERO forest (plan §6): per-rung D with seed t-CI whiskers, arm2/arm7
    columns beside; flagships marked; sanity-excluded rungs grayed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    rows = [r for r in table["per_rung"] if r.get("D", {}).get("mean") is not None]
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: (r["behavior"], r["eval_rung"]))
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(rows) + 1.6))
    ys = range(len(rows))
    for y, r in zip(ys, rows, strict=True):
        d = r["D"]
        color = "#9A9A9A" if r["excluded_by_sanity"] else COLOR_DTRUE
        ci = d.get("tci") or [d["mean"], d["mean"]]
        ax.errorbar(
            d["mean"],
            y,
            xerr=[[max(0.0, d["mean"] - ci[0])], [max(0.0, ci[1] - d["mean"])]],
            fmt="o",
            color=color,
            capsize=2,
        )
        if r.get("D_ctx_ci"):
            lo, hi = r["D_ctx_ci"]
            ax.plot([lo, hi], [y - 0.18, y - 0.18], color=color, alpha=0.5, lw=1.2)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(
        [
            rung_label(r["behavior"], r["eval_rung"])
            + (" ★" if r["flagship"] else "")
            + (" [excluded]" if r["excluded_by_sanity"] else "")
            for r in rows
        ],
        fontsize=7,
    )
    ax.set_xlabel(
        "mapped-answer probe minus repaired context direction (Spearman ρ difference)\n"
        "dot + whisker = 5-seed mean ± t-CI; thin bar = paired group-level context CI"
    )
    ax.set_title("Does the mapped-answer probe beat the repaired context direction?")
    written = [str(savefig_paper(fig, fig_dir / "arm2fix_forest_D"))]
    plt.close(fig)
    return written


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


A2_FAMILY_ARMS = (ARM_CTXDIR, "arm2q_ctx_native")


def parse_repaired_arm_tokens(tokens, behaviors) -> dict[str, str]:
    """``--repaired-arm`` token grammar → per-behavior override dict.

    ``auto`` (alone) → {} (pure per-behavior auto resolution). A bare arm
    name → that arm for EVERY behavior. ``<behavior>=<arm>`` tokens → named
    overrides; unnamed behaviors stay on auto. Mixing forms, unknown
    behaviors/arms, or a duplicate behavior are loud parse errors."""
    toks = list(tokens or [])
    if toks == ["auto"]:
        return {}
    pairs = [t for t in toks if "=" in t]
    bare = [t for t in toks if "=" not in t]
    if bare and pairs:
        raise SystemExit(
            f"[a2fix] --repaired-arm mixes a global arm with per-behavior pairs: {toks}"
        )
    if bare:
        if len(bare) != 1 or bare[0] not in A2_FAMILY_ARMS:
            raise SystemExit(
                f"[a2fix] --repaired-arm global form takes exactly one of "
                f"{A2_FAMILY_ARMS} (got {bare})"
            )
        return {b: bare[0] for b in behaviors}
    out: dict[str, str] = {}
    for t in pairs:
        b, _, arm = t.partition("=")
        if b not in behaviors:
            raise SystemExit(f"[a2fix] --repaired-arm names unknown behavior {b!r} (of {toks})")
        if arm not in A2_FAMILY_ARMS:
            raise SystemExit(f"[a2fix] --repaired-arm names unknown arm {arm!r} (of {toks})")
        if b in out:
            raise SystemExit(f"[a2fix] --repaired-arm duplicates behavior {b!r} (of {toks})")
        out[b] = arm
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode",
        choices=("claim4", "arm2fix"),
        default="claim4",
        help="claim4 (default, byte-identical legacy fold) | arm2fix (plan §4 Leg 2 D2: "
        "repaired-arm2 vs banked-arm7 fold + amended §3 lattice)",
    )
    ap.add_argument(
        "--arm2fix-root",
        type=Path,
        default=Path("eval_results/issue_1739/claim4_controls/arm2fix"),
        help="root of THIS round's repaired-arm2 seed outputs (<b>/seed<S>/...)",
    )
    ap.add_argument(
        "--repaired-arm",
        nargs="+",
        default=["auto"],
        help="PER-BEHAVIOR-capable override (the ladder runs per behavior — plan §4/§7): "
        "'auto' (default; per behavior: arm2q_ctx_native when ITS primary rows are "
        "present, else arm2_ctx_native), a global arm name applied to every behavior, "
        "or '<behavior>=<arm>' tokens (e.g. evil=arm2q_ctx_native "
        "sycophancy=arm2_ctx_native); unnamed behaviors stay on auto",
    )
    ap.add_argument(
        "--d0-p4",
        type=Path,
        default=Path(
            "eval_results/issue_1739/claim4_controls/arm2fix/d0/p4_direction_stability.json"
        ),
        help="P4 across-seed direction-cosine JSON (reported beside any MAP-BEATS output)",
    )
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
        args.out_root = args.arm2fix_root if args.mode == "arm2fix" else args.claim4_root
    if args.mode == "arm2fix":
        # mode-conditional defaults: never overwrite the legacy fold's outputs
        if args.note_out == ap.get_default("note_out"):
            args.note_out = Path("docs/map_behavior_prediction_arm2fix_note.md")
        if args.fig_dir == ap.get_default("fig_dir"):
            args.fig_dir = Path("figures/issue_1739/claim4_controls/arm2fix")
    args.repaired_arm_overrides = parse_repaired_arm_tokens(args.repaired_arm, args.behaviors)
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

        # arm2fix-mode surface (plan §4 Leg 2 D2): assert the loaders/lattice
        # resolve alongside the legacy fold surface (incl. the r2 per-behavior
        # resolver + hardening asserts).
        assert callable(build_arm2fix_table) and callable(a2fix_lattice)
        assert callable(a2fix_resolve_repairs) and callable(parse_repaired_arm_tokens)
        assert callable(a2fix_assert_finite) and callable(a2fix_ctx_gap_assert)
        assert set(A2FIX_VERDICTS) == {
            "INCONCLUSIVE-ADAPTER",
            "MAP-BEATS-CONTEXT-DIRECTION",
            "MAP-ADVANTAGE-NOT-SHOWN",
            "WEAK-MIXED",
        }
        print("[claim4-fold] import-check OK", flush=True)
        return 0

    if args.mode == "arm2fix":
        table = build_arm2fix_table(args)
        args.out_root.mkdir(parents=True, exist_ok=True)
        out_json = args.out_root / "arm2fix_table.json"
        out_json.write_text(json.dumps(table, indent=1))
        write_arm2fix_markdown(table, args.out_root / "arm2fix_table.md")
        args.note_out.parent.mkdir(parents=True, exist_ok=True)
        write_arm2fix_note(table, args.note_out)
        _log(f"table -> {out_json}")
        _log(f"verdict: {table['verdict'].get('verdict')}")
        if not args.no_figures:
            written = render_arm2fix_figures(table, args.fig_dir)
            _log(f"figures -> {args.fig_dir} ({written})")
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
