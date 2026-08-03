#!/usr/bin/env python
"""Issue #1482: judge-agreement dose-response, within-activity-decile AUROC, and
the ALL-CLASS extension of the full-width label reads.

The BINARY/label side of the full-width predictor battery. Three deliverables,
one code path, ONE parameter for the R^2 target so the whole battery repoints
at the dense->SAE full-width refit the moment that grid lands:

  (1) judge-unanimity dose-response
      Per axis, the label->R^2 signal restricted to features whose 5 judge
      draws all survived (`n_surviving == 5`) and agreed 5-0 / 4-1 / 3-2.
      "Unanimous" means 5/5 STRICTLY: 206-989 rows per axis are unanimous on
      <= 2 surviving draws and are the LOWEST-evidence features, so they must
      never enter the high-confidence arm.
      Panel (b) is the confound check — restricting to unanimous changes the
      feature POPULATION, not just label noise — and carries a label-free
      median R^2 so the `unresolved` features (8,317 on functional_role) that
      every labelled read silently drops stay visible.

  (2) AUROC within ACTIVITY decile
      The binary analogue of the continuous side's `activity_decile_profiles`:
      10 equal-count activity deciles over the analysis universe, per-decile
      one-vs-rest AUROC per label class, bootstrap CIs. (`_decile_profile` in
      the fullwidth battery bins by PREDICTOR decile, which is a different
      read.)

  (3) all-class label reads
      `fullwidth_label_reads.json` banks 10 codings; the abstraction axis is
      missing 2 of its 3 classes and functional_role all 3. This runs the
      IDENTICAL banked recipe (`FW.label_reads` — AUROC + group-conditional
      bootstrap CI + the k-sweep + the activity-stratified permutation null +
      the scan-corrected band) over all 15 classes so the excess-sorted table
      is internally consistent.

CORPUS CAVEAT carried on every figure and in every sidecar: the full-width
R^2 default is the #1738 SAE->SAE MULTI-TURN read while activity comes from
the #1482 SINGLE-TURN pooled store. `--r2-path` / `--r2-label` repoint it.

Vectorized throughout — no Python draw loops:
  * bootstrap: ONE shared multinomial row-resample per chunk, all classes'
    one-vs-rest AUROCs read off it as two GEMMs against the reweighted
    midrank vector (`_shared_resample_class_auroc`);
  * permutation null: within-activity-decile label permutations built as one
    `argsort` over a (m, chunk) uniform draw per stratum, every draw's AUROC a
    GEMM against the FIXED rank vector (`_perm_null_class_auroc`) — the same
    identity the fullwidth battery's `_mw_from_ranks` uses.

Rank convention: 1-BASED average ranks, so every AUROC here equals
`P(pos > neg) + 0.5 P(tie)` and is directly comparable to the battery's
`auroc_with_boot`. (`PB._rank` is 0-based; the battery's `_mw_from_ranks`
consumes it, which shifts observed AND null by the same 1/n0 and so leaves
its EXCESS unbiased.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402

# repo root by __file__ walk, NOT task_workflow.repo_root() — keep this module
# importable from a checkout with no `tasks/` tree.
REPO = Path(__file__).resolve().parents[1]
assert (REPO / "scripts" / "issue1482_predictor_battery_fullwidth.py").is_file(), REPO
sys.path.insert(0, str(REPO / "scripts"))

import issue1482_predictor_battery_fullwidth as FW  # noqa: E402
import issue1773_common as C1773  # noqa: E402

PB = FW.PB

# The judge's own vote predicate + draw budget, imported from the PRODUCER module
# so this read can never drift from how the labels were minted.
MAJORITY_FLOOR = C1773.MAJORITY_FLOOR  # 3
N_DRAWS = C1773.N_DRAWS  # 5

DICT_SIZE = FW.DICT_SIZE
SEED = 1482
N_BOOT = 2000
N_PERM = 2000

FULLDICT_LABELS = FW.FULLDICT_LABELS
COVARIATES = "eval_results/issue_1482/predictor_battery/fullwidth_covariates.npz"
DEFAULT_R2_PATH = "eval_results/issue_1738/sae_twoway/perfeature/sae_context_r2.npy"
DEFAULT_R2_LABEL = "#1738 SAE->SAE multi-turn context R^2 (provisional target)"
CORPUS_CAVEAT = (
    "cross-corpus: R^2 is the #1738 MULTI-TURN read; activity / judged labels come "
    "from the #1482 SINGLE-TURN corpus"
)

OUT_DIR = "eval_results/issue_1482/label_agreement"
FIG_DIR = "figures/issue_1482/label_agreement"

# Axis order is the COLOUR order — one colour = one axis across every figure in
# this round (class-within-axis is distinguished by linestyle + marker).
AXIS_ORDER = (
    "interpretable",
    "abstraction",
    "content_type",
    "speaker_property",
    "functional_role",
)
AXIS_LABEL = {
    "interpretable": "interpretable",
    "abstraction": "abstraction",
    "content_type": "content type",
    "speaker_property": "speaker property",
    "functional_role": "functional role",
}
# Labels the judge can return that carry no class membership on that axis.
AXIS_DROP = {
    "interpretable": ("unresolved",),
    "abstraction": ("unresolved",),
    "content_type": ("unresolved",),
    "speaker_property": ("unresolved", "unclear"),
    "functional_role": ("unresolved",),
}
# EVERY class of EVERY axis — the label partition the DOSE statistic aggregates
# over. Transcribed from the artifact's OWN realized resolved-label set (NOT the
# battery's `CATEGORICAL`, whose per-axis tuple omits the axis reference level:
# `speaker_property: none` is 106,091 of 127,006 resolved features and is a real
# class, not a sentinel). `assert_realized_classes` re-checks this against the
# loaded labels at every run, so a relabel can never silently shift the partition.
AXIS_CLASSES = {
    "interpretable": ("yes", "no"),
    "abstraction": ("token_surface", "lexical_semantic", "abstract_contextual"),
    "content_type": ("topic", "task_format", "entity", "syntax", "operation"),
    "speaker_property": ("none", "language", "register_style", "identity_disposition"),
    "functional_role": ("input_side", "output_promoting", "mixed"),
}
# Deliverable-3 read-outs: one-vs-rest per class, per the round's class list.
# Two deliberate omissions, both because the read would be redundant:
#   `interpretable: no`      — the exact mirror of `yes` (AUROC_no = 1 - AUROC_yes)
#   `speaker_property: none` — "no speaker property"; its one-vs-rest read is the
#                              complement of the union of the other three
# Both remain in AXIS_CLASSES: the dose statistic needs the FULL partition.
READ_OMIT = {("interpretable", "no"), ("speaker_property", "none")}
READ_CLASSES: dict[str, tuple[str, str]] = {}
for _ax in AXIS_ORDER:
    for _lvl in AXIS_CLASSES[_ax]:
        if (_ax, _lvl) in READ_OMIT:
            continue
        READ_CLASSES[f"{_ax}__{_lvl}"] = (_ax, _lvl)

# Verified full-width agreement distribution (5/5 STRICT), out of 128,482 judged
# features per axis. Asserted before any plotting: a mismatch means the wrong
# label source was read (the 16k panel, or the pre-recovery_1934 copy).
AGREEMENT_GATE = {
    #  axis:            (5-0,   4-1,    3-2,    unresolved, n<5)
    "interpretable": (99740, 14599, 10602, 959, 3541),
    "speaker_property": (96222, 17232, 12533, 1501, 1528),
    "content_type": (88100, 19401, 16049, 3780, 1658),
    "functional_role": (79825, 17851, 15815, 8317, 9475),
    "abstraction": (74257, 26828, 22571, 3133, 2492),
}
AGREEMENT_LEVELS = ("3-2", "4-1", "5-0")  # x-axis order: weakest -> unanimous
assert N_DRAWS == 5, f"AGREEMENT_LEVELS labels assume N_DRAWS==5, got {N_DRAWS}"
# modal-vote count that defines each level (5-0 == 5/5 STRICT unanimity)
BEST_FOR_LEVEL = (N_DRAWS - 2, N_DRAWS - 1, N_DRAWS)
N_DECILES = FW.N_DECILES

# Bootstrap/permutation chunking: cap the (n, chunk) working arrays so the
# peak stays a few hundred MB at n ~ 1.1e5.
CHUNK_ELEMS = 5_000_000

# Per-deliverable RNG offsets (see main()).
RNG_OFFSETS = {"dose": 0, "decile": 1, "reads": 2}


def _log(msg: str) -> None:
    print(f"[label-agreement] {msg}", flush=True)


def _chunk_for(n: int) -> int:
    return int(max(10, min(200, CHUNK_ELEMS // max(n, 1))))


# ── agreement extraction ──────────────────────────────────────────────────────


def load_agreement(cache: Path) -> dict[str, dict[str, np.ndarray]]:
    """Per axis, DICT_SIZE-wide arrays of the judge's per-draw vote structure.

    Returns, per axis: `label` (majority label or 'unresolved'; 'unlabeled'
    where the axis has no judged row), `n_surv` (surviving draws), `best`
    (votes for the modal label), `resolved` (a unique modal label with
    >= MAJORITY_FLOOR votes — the `majority_vote` predicate).
    """
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        out = {}
        for ax in AXIS_ORDER:
            out[ax] = {
                "label": z[f"{ax}__label"],
                "n_surv": z[f"{ax}__n_surv"],
                "best": z[f"{ax}__best"],
                "resolved": z[f"{ax}__resolved"],
            }
        _log(f"agreement: loaded cache {cache}")
        return out

    from collections import Counter

    out = {
        ax: {
            "label": np.full(DICT_SIZE, "unlabeled", dtype=object),
            "n_surv": np.zeros(DICT_SIZE, dtype=np.int8),
            "best": np.zeros(DICT_SIZE, dtype=np.int8),
            "resolved": np.zeros(DICT_SIZE, dtype=bool),
        }
        for ax in AXIS_ORDER
    }
    shards = sorted(FULLDICT_LABELS.glob("axis_labels.shard*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no axis_labels shards under {FULLDICT_LABELS}")
    n_rows = 0
    t0 = time.time()
    for p in shards:
        # text-mode iteration: NEVER str.splitlines() on JSONL (U+2028 shreds rows)
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ax = r["axis"]
                if ax not in out:
                    continue
                n_rows += 1
                fid = int(r["feat_id"])
                surv = r["labels_surviving"]
                # producer's own predicate — never a reimplementation
                lab = C1773.majority_vote(surv)
                resolved = lab != "unresolved"
                best = max(Counter(surv).values()) if surv else 0
                out[ax]["label"][fid] = lab
                out[ax]["n_surv"][fid] = len(surv)
                out[ax]["best"][fid] = best
                out[ax]["resolved"][fid] = resolved
    _log(f"agreement: parsed {n_rows} rows over {len(shards)} shards ({time.time() - t0:.0f}s)")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache.with_suffix(".tmp.npz"),
        **{f"{ax}__{k}": v for ax, d in out.items() for k, v in d.items()},
    )
    cache.with_suffix(".tmp.npz").replace(cache)
    return out


def agreement_gate(agr: dict[str, dict[str, np.ndarray]]) -> dict:
    """Assert the realized agreement distribution matches the verified table."""
    realized = {}
    for ax in AXIS_ORDER:
        a = agr[ax]
        judged = a["label"] != "unlabeled"
        n5 = judged & (a["n_surv"] == N_DRAWS)
        row = (
            int((n5 & (a["best"] == 5)).sum()),
            int((n5 & (a["best"] == 4)).sum()),
            int((n5 & (a["best"] == 3)).sum()),
            int((judged & ~a["resolved"]).sum()),
            int((judged & (a["n_surv"] < N_DRAWS)).sum()),
        )
        realized[ax] = {
            "5-0": row[0],
            "4-1": row[1],
            "3-2": row[2],
            "unresolved": row[3],
            "n_lt_5": row[4],
            "judged": int(judged.sum()),
        }
        if row != AGREEMENT_GATE[ax]:
            raise AssertionError(
                f"agreement gate FAILED for {ax}: realized {row} != expected "
                f"{AGREEMENT_GATE[ax]} — wrong label source?"
            )
        if int(judged.sum()) != 128_482:
            raise AssertionError(f"{ax}: judged {int(judged.sum())} != 128482")
        # DEFINITION TRAP: "all surviving draws agree" is NOT unanimity.
        loose = int((judged & (a["n_surv"] > 0) & (a["best"] == a["n_surv"])).sum())
        realized[ax]["all_surviving_agree"] = loose
        realized[ax]["all_surviving_agree_but_n_le_2"] = int(
            (judged & (a["n_surv"] > 0) & (a["best"] == a["n_surv"]) & (a["n_surv"] <= 2)).sum()
        )
    _log("agreement gate PASS — all 5 axes match the verified 5/5-strict distribution")
    return realized


def assert_realized_classes(agr: dict[str, dict[str, np.ndarray]]) -> dict:
    """Fail loud when the artifact's REALIZED class set differs from AXIS_CLASSES.

    The dose statistic aggregates one-vs-rest reads over an axis's class
    PARTITION, so a class present in the labels but absent from `AXIS_CLASSES`
    silently breaks the partition (and previously surfaced only as a mid-run
    `unmapped label` crash). Checked against the labels themselves, never
    against a sibling module's declaration.
    """
    counts = {}
    for ax in AXIS_ORDER:
        lab = agr[ax]["label"]
        sel = (lab != "unlabeled") & agr[ax]["resolved"] & ~np.isin(lab, AXIS_DROP[ax])
        vals, cnt = np.unique(lab[sel].astype(str), return_counts=True)
        realized = set(vals.tolist())
        declared = set(AXIS_CLASSES[ax])
        if realized != declared:
            raise AssertionError(
                f"{ax}: realized classes {sorted(realized)} != declared "
                f"{sorted(declared)} (missing {sorted(declared - realized)}, "
                f"extra {sorted(realized - declared)}) — update AXIS_CLASSES"
            )
        counts[ax] = {v: int(c) for v, c in zip(vals.tolist(), cnt.tolist(), strict=True)}
        counts[ax]["_dropped_labels"] = int(
            ((lab != "unlabeled") & agr[ax]["resolved"] & np.isin(lab, AXIS_DROP[ax])).sum()
        )
    _log("realized-class gate PASS — every axis's label partition matches AXIS_CLASSES")
    return counts


# ── universe assembly ─────────────────────────────────────────────────────────


def assemble(r2_path: Path, r2_label: str, agr: dict) -> dict:
    """Join labels + agreement with the durable full-width covariate substrate.

    Universe = judged (interpretable axis) AND finite primary R^2 AND
    answer-active — byte-identical to the fullwidth battery's `assemble`
    predicate (verified: the reconstructed `interpretable` read has n = 114,076,
    matching the banked `fullwidth_label_reads.json`).
    """
    z = np.load(REPO / COVARIATES)
    activity = np.asarray(z["activity"], dtype=np.float64)
    feat_ids_all = np.asarray(z["feat_ids"], dtype=np.int64)
    if activity.shape != (DICT_SIZE,):
        raise AssertionError(f"covariates activity shape {activity.shape}")

    r2_all = np.asarray(np.load(r2_path), dtype=np.float64)
    if r2_all.shape != (DICT_SIZE,):
        raise AssertionError(f"{r2_path}: expected ({DICT_SIZE},), got {r2_all.shape}")

    judged = agr["interpretable"]["label"] != "unlabeled"
    finite = np.isfinite(r2_all)
    active = activity > 0
    keep = judged & finite & active
    idx = np.flatnonzero(keep)
    _log(
        f"universe: {len(idx)} of {DICT_SIZE} "
        f"(judged {int(judged.sum())}, finite R^2 {int(finite.sum())}, active {int(active.sum())})"
    )

    return {
        "feat_ids": feat_ids_all[idx],
        "r2": r2_all[idx],
        "activity": activity[idx],
        "labels": {ax: agr[ax]["label"][idx] for ax in AXIS_ORDER},
        "n_surv": {ax: agr[ax]["n_surv"][idx] for ax in AXIS_ORDER},
        "best": {ax: agr[ax]["best"][idx] for ax in AXIS_ORDER},
        "resolved": {ax: agr[ax]["resolved"][idx] for ax in AXIS_ORDER},
        "r2_path": str(r2_path),
        "r2_label": r2_label,
        "coverage": {
            "dict_size": DICT_SIZE,
            "judged": int(judged.sum()),
            "finite_r2": int(finite.sum()),
            "answer_active": int(active.sum()),
            "universe": int(len(idx)),
        },
    }


# ── vectorized AUROC machinery (shared resample + shared permutation) ─────────


def _rank1(a: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties share their mean rank)."""
    return PB._rank(a) + 1.0


def _auroc_from_rank1(rank1: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """One-vs-rest AUROC per column of a (n, k) 0/1 membership matrix."""
    n = onehot.shape[0]
    n_c = onehot.sum(axis=0)
    s_c = rank1 @ onehot
    denom = n_c * (n - n_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (s_c - n_c * (n_c + 1.0) / 2.0) / denom, np.nan)


def _shared_resample_class_auroc(
    r2_s: np.ndarray, onehot: np.ndarray, n_boot: int, rng
) -> np.ndarray:
    """(k, n_boot) bootstrap AUROC draws under ONE shared row resample.

    `r2_s` must be sorted ASCENDING and `onehot` given in that order. Each draw
    is a multinomial row resample with weights `w`; the reweighted midrank of a
    tie group is `C_g + (W_g + 1)/2` (exact under ties), so every class's
    Mann-Whitney AUROC for the draw is two GEMMs — `onehot.T @ w` and
    `onehot.T @ (w * midrank)`. Sharing the resample across classes preserves
    the (strong, one-vs-rest) correlation between them, which independent
    per-class resamples would drop.
    """
    n = len(r2_s)
    k = onehot.shape[1]
    _, starts = np.unique(r2_s, return_index=True)  # ascending -> tie-group starts
    group_sizes = np.diff(np.append(starts, n))
    out = np.empty((k, n_boot), dtype=np.float64)
    chunk = _chunk_for(n)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(n, b))
        w = np.bincount((idx + np.arange(b) * n).ravel(), minlength=n * b)
        w = w.reshape(b, n).T.astype(np.float64)  # (n, b)
        del idx
        wg = np.add.reduceat(w, starts, axis=0)  # (n_uniq, b)
        mid = (np.cumsum(wg, axis=0) - wg) + (wg + 1.0) / 2.0
        r = np.repeat(mid, group_sizes, axis=0)  # (n, b)
        r *= w  # in place: w * midrank
        n_c = onehot.T @ w  # (k, b)
        s_c = onehot.T @ r  # (k, b)
        denom = n_c * (n - n_c)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, done : done + b] = np.where(
                denom > 0, (s_c - n_c * (n_c + 1.0) / 2.0) / denom, np.nan
            )
        del w, r, wg, mid
        done += b
    return out


def _perm_null_class_auroc(
    rank1: np.ndarray, lab_int: np.ndarray, strata: np.ndarray, k: int, n_perm: int, rng
) -> np.ndarray:
    """(k, n_perm) AUROC draws under within-activity-decile label permutation.

    Stratifying on activity PRESERVES the activity-label association, so the
    null is NOT centred at chance — the same convention (and reason) as the
    fullwidth battery's `label_reads`.
    """
    n = len(lab_int)
    strata_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = np.empty((k, n_perm), dtype=np.float64)
    chunk = _chunk_for(n)
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        perm_lab = np.empty((n, b), dtype=np.int16)
        for i in strata_idx:
            # one argsort over a (m, b) uniform draw = b independent permutations
            perm_lab[i] = lab_int[i][np.argsort(rng.random((len(i), b)), axis=0)]
        for c in range(k):
            m = (perm_lab == c).astype(np.float64)  # (n, b)
            n_c = m.sum(axis=0)
            s_c = rank1 @ m
            denom = n_c * (n - n_c)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[c, done : done + b] = np.where(
                    denom > 0, (s_c - n_c * (n_c + 1.0) / 2.0) / denom, np.nan
                )
        del perm_lab
        done += b
    return out


# ── deliverable 1: judge-agreement dose-response ─────────────────────────────


def _axis_level_read(
    r2: np.ndarray,
    activity: np.ndarray,
    lab: np.ndarray,
    axis: str,
    n_boot: int,
    n_perm: int,
    rng,
) -> dict | None:
    """Per-class AUROC / stratified-null excess, plus the axis-level aggregate.

    Axis-level statistic: the prevalence-weighted mean ABSOLUTE excess over the
    axis's classes. It is in AUROC units, comparable across axes, and degenerates
    exactly to |excess| on a binary axis (one-vs-rest AUROC and its stratified
    null both mirror, so both classes contribute the same |excess|).
    """
    classes = [c for c in AXIS_CLASSES[axis] if (lab == c).sum() > 0]
    if len(classes) < 2:
        return None
    k = len(classes)
    n = len(lab)

    order = np.argsort(r2, kind="stable")  # ASCENDING for the midrank machinery
    r2_s = r2[order]
    lab_s = lab[order]
    act_s = activity[order]
    lab_int = np.full(n, -1, dtype=np.int16)
    for c_i, c in enumerate(classes):
        lab_int[lab_s == c] = c_i
    if (lab_int < 0).any():
        raise AssertionError(f"{axis}: unmapped label in level subset")
    onehot = np.zeros((n, k), dtype=np.float64)
    onehot[np.arange(n), lab_int] = 1.0

    rank1 = _rank1(r2_s)
    obs = _auroc_from_rank1(rank1, onehot)
    prevalence = onehot.mean(axis=0)

    strata = FW._decile_of(act_s)
    null = _perm_null_class_auroc(rank1, lab_int, strata, k, n_perm, rng)
    null_mean = np.nanmean(null, axis=1)
    excess = obs - null_mean

    boot = _shared_resample_class_auroc(r2_s, onehot, n_boot, rng)
    boot_excess = boot - null_mean[:, None]  # null held fixed (a design property)
    agg_draws = np.nansum(np.abs(boot_excess) * prevalence[:, None], axis=0)
    agg_point = float(np.sum(np.abs(excess) * prevalence))

    per_class = {}
    for c_i, c in enumerate(classes):
        col = null[c_i][np.isfinite(null[c_i])]
        per_class[c] = {
            "n_positive": int(onehot[:, c_i].sum()),
            "prevalence": float(prevalence[c_i]),
            "auroc": float(obs[c_i]),
            "auroc_ci95": PB._ci(boot[c_i]),
            "null_mean": float(null_mean[c_i]),
            "null_band95": [
                float(np.percentile(col, 2.5)) if len(col) else float("nan"),
                float(np.percentile(col, 97.5)) if len(col) else float("nan"),
            ],
            "excess": float(excess[c_i]),
            "excess_ci95": PB._ci(boot_excess[c_i]),
        }
    return {
        "n": int(n),
        "classes": classes,
        "per_class": per_class,
        "excess_weighted_abs": agg_point,
        "excess_weighted_abs_ci95": PB._ci(agg_draws),
        "median_r2": float(np.median(r2)),
        "median_activity": float(np.median(activity)),
    }


def dose_response(bundle: dict, n_boot: int, n_perm: int, rng) -> dict:
    """Deliverable 1: label->R^2 signal vs judge agreement level, per axis.

    Restricted to `n_surviving == 5` so the dose axis means ONE thing: 5-0 is
    5/5 STRICT unanimity, never "all surviving draws agreed".
    """
    r2, act = bundle["r2"], bundle["activity"]
    out: dict[str, dict] = {}
    for axis in AXIS_ORDER:
        lab = bundle["labels"][axis]
        n_surv = bundle["n_surv"][axis]
        best = bundle["best"][axis]
        resolved = bundle["resolved"][axis]
        drop = AXIS_DROP[axis]

        five = n_surv == N_DRAWS
        levels: dict[str, dict] = {}
        for lvl, bval in zip(AGREEMENT_LEVELS, BEST_FOR_LEVEL, strict=True):
            sel = five & (best == bval) & resolved & ~np.isin(lab, drop)
            m = int(sel.sum())
            if m < 200:
                _log(f"{axis} {lvl}: n={m} too small, skipped")
                continue
            t0 = time.time()
            read = _axis_level_read(r2[sel], act[sel], lab[sel], axis, n_boot, n_perm, rng)
            if read is None:
                continue
            read["dropped_labels_excluded"] = int(
                (five & (best == bval) & resolved & np.isin(lab, drop)).sum()
            )
            levels[lvl] = read
            _log(
                f"{axis} {lvl}: n={m} weighted|excess|={read['excess_weighted_abs']:+.4f} "
                f"[{read['excess_weighted_abs_ci95'][0]:+.4f}, "
                f"{read['excess_weighted_abs_ci95'][1]:+.4f}] ({time.time() - t0:.0f}s)"
            )

        # Panel (b): population confound check. `unresolved` is EVERY unresolved
        # feature (any n_surviving) — label-free, so median R^2 is defined and
        # these features stop being silently dropped.
        pop = {}
        unres = ~resolved & (lab != "unlabeled")
        pop["unresolved"] = {
            "n": int(unres.sum()),
            "median_activity": float(np.median(act[unres])) if unres.any() else float("nan"),
            "median_r2": float(np.median(r2[unres])) if unres.any() else float("nan"),
        }
        for lvl, bval in zip(AGREEMENT_LEVELS, BEST_FOR_LEVEL, strict=True):
            sel = five & (best == bval) & resolved
            pop[lvl] = {
                "n": int(sel.sum()),
                "median_activity": float(np.median(act[sel])) if sel.any() else float("nan"),
                "median_r2": float(np.median(r2[sel])) if sel.any() else float("nan"),
            }
        out[axis] = {"levels": levels, "population": pop}
    return out


# ── deliverable 2: AUROC within activity decile ──────────────────────────────


def decile_auroc(bundle: dict, n_boot: int, rng) -> dict:
    """Deliverable 2: per-activity-decile one-vs-rest AUROC per label class.

    Deciles are EQUAL-COUNT over the analysis universe (not equal-width in
    log-activity) — the internal span of each decile is measured and reported so
    the caption can state how tightly conditioned the endpoints are.
    """
    r2, act = bundle["r2"], bundle["activity"]
    dec = FW._decile_of(act)

    spans = []
    for d in range(N_DECILES):
        m = dec == d
        lo, hi = float(act[m].min()), float(act[m].max())
        spans.append(
            {
                "decile": int(d + 1),
                "n": int(m.sum()),
                "activity_min": lo,
                "activity_max": hi,
                "internal_span_ratio": float(hi / lo) if lo > 0 else float("inf"),
                "median_activity": float(np.median(act[m])),
                "median_r2": float(np.median(r2[m])),
            }
        )
    _log(
        "activity deciles: internal span ratios "
        + ", ".join(f"d{s['decile']}={s['internal_span_ratio']:.3g}x" for s in spans)
    )

    reads: dict[str, dict] = {}
    for coding, (axis, level) in READ_CLASSES.items():
        lab = bundle["labels"][axis]
        keep = ~np.isin(lab, AXIS_DROP[axis]) & (lab != "unlabeled")
        pos = lab == level
        prof = []
        for d in range(N_DECILES):
            m = keep & (dec == d)
            p = pos & m
            npos, nneg = int(p.sum()), int((m & ~pos).sum())
            if npos < 10 or nneg < 10:
                prof.append(
                    {
                        "decile": int(d + 1),
                        "n": int(m.sum()),
                        "n_positive": npos,
                        "auroc": float("nan"),
                        "auroc_ci95": [float("nan"), float("nan")],
                        "skipped_reason": "fewer than 10 in a class",
                    }
                )
                continue
            point, ci = FW.auroc_with_boot(r2[p], r2[m & ~pos], n_boot, rng)
            prof.append(
                {
                    "decile": int(d + 1),
                    "n": int(m.sum()),
                    "n_positive": npos,
                    "prevalence": npos / int(m.sum()),
                    "auroc": float(point),
                    "auroc_ci95": ci,
                }
            )
        vals = [p["auroc"] for p in prof if np.isfinite(p["auroc"])]
        reads[coding] = {
            "source_axis": axis,
            "class": level,
            "n_labelled": int(keep.sum()),
            "n_positive": int((pos & keep).sum()),
            "profile": prof,
            "auroc_range": [float(min(vals)), float(max(vals))] if vals else None,
        }
        _log(
            f"decile AUROC {coding}: "
            + (f"{min(vals):.3f}..{max(vals):.3f} over {len(vals)} deciles" if vals else "n/a")
        )
    return {"deciles": spans, "reads": reads}


# ── deliverable 3: all-class label reads ─────────────────────────────────────


def all_class_label_reads(bundle: dict, n_perm: int, n_boot: int, rng) -> dict:
    """Deliverable 3: the banked `label_reads` recipe over EVERY class.

    Reuses `FW.label_reads` verbatim (AUROC + group-conditional bootstrap CI +
    the k-sweep + the activity-stratified permutation null + the scan-corrected
    band) so the excess-sorted table is directly comparable to the 10 codings
    already banked in `fullwidth_label_reads.json`. `FW.BINARY_AXES` is the
    coding registry `label_reads` iterates; it is swapped for the 15-class set
    and restored.
    """
    fw_bundle = {
        "r2": bundle["r2"],
        "cov": {"activity": bundle["activity"]},
        "labels": bundle["labels"],
    }
    codings = {
        coding: (axis, (lambda s, _lvl=level: s == _lvl), AXIS_DROP[axis])
        for coding, (axis, level) in READ_CLASSES.items()
    }
    saved = FW.BINARY_AXES
    try:
        FW.BINARY_AXES = codings
        reads = FW.label_reads(fw_bundle, n_perm=n_perm, n_boot=n_boot, rng=rng)
    finally:
        FW.BINARY_AXES = saved

    for coding, r in reads.items():
        r["class"] = READ_CLASSES[coding][1]
        r["excess_over_stratified_null"] = float(r["auroc"] - r["auroc_perm_null_mean"])
    order = sorted(reads, key=lambda c: reads[c]["excess_over_stratified_null"], reverse=True)
    for rank, coding in enumerate(order, start=1):
        reads[coding]["excess_rank"] = rank
    return {"reads": reads, "order_by_excess": order}


# ── figures ──────────────────────────────────────────────────────────────────


def _axis_colors():
    from explore_persona_space.analysis.paper_plots import paper_palette

    return dict(zip(AXIS_ORDER, paper_palette(len(AXIS_ORDER)), strict=True))


def _footnote(fig, bundle: dict) -> None:
    fig.text(
        0.005,
        0.004,
        f"R^2 target: {bundle['r2_label']}  |  {CORPUS_CAVEAT}",
        fontsize=6.4,
        color="#555555",
        ha="left",
        va="bottom",
    )


def _enrich_meta(paths: dict, block: dict) -> None:
    """Merge a provenance block into the sidecar `savefig_paper` just wrote."""
    p = paths["meta"]
    meta = json.loads(Path(p).read_text(encoding="utf-8"))
    meta["issue1482_label_agreement"] = block
    Path(p).write_text(json.dumps(meta, indent=1), encoding="utf-8")


def fig_dose_response(dose: dict, bundle: dict, gate: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    colors = _axis_colors()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    xs = np.arange(len(AGREEMENT_LEVELS))

    ax = axes[0]
    for axis in AXIS_ORDER:
        lv = dose[axis]["levels"]
        pts = [(i, lv[lab]) for i, lab in enumerate(AGREEMENT_LEVELS) if lab in lv]
        if not pts:
            continue
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1]["excess_weighted_abs"] for p in pts])
        lo = np.array([p[1]["excess_weighted_abs_ci95"][0] for p in pts])
        hi = np.array([p[1]["excess_weighted_abs_ci95"][1] for p in pts])
        ax.plot(x, y, "o-", ms=4.2, lw=1.5, color=colors[axis], label=AXIS_LABEL[axis])
        ax.fill_between(x, lo, hi, color=colors[axis], alpha=0.16, lw=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(AGREEMENT_LEVELS))
    ax.set_xlabel("judge agreement among the 5 draws (all 5 surviving)")
    ax.set_ylabel("excess over activity-stratified null\n(prevalence-weighted mean |AUROC excess|)")
    ax.set_title(
        "(a) Label to R-squared signal strengthens with judge agreement", loc="left", fontsize=10
    )
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax.legend(frameon=False, fontsize=7.6, ncol=2)

    ax2 = axes[1]
    pop_cats = ("unresolved",) + AGREEMENT_LEVELS
    xs2 = np.arange(len(pop_cats), dtype=float)
    twin = ax2.twinx()
    for axis in AXIS_ORDER:
        pop = dose[axis]["population"]
        act = [pop[c]["median_activity"] for c in pop_cats]
        r2v = [pop[c]["median_r2"] for c in pop_cats]
        ax2.plot(xs2, act, "o-", ms=4.0, lw=1.5, color=colors[axis], label=AXIS_LABEL[axis])
        twin.plot(xs2, r2v, "s--", ms=3.6, lw=1.2, color=colors[axis], alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(list(pop_cats))
    ax2.set_xlabel("judge agreement (unresolved = no >=3-vote majority, any n surviving)")
    ax2.set_ylabel("median activity (firing frequency, log)   — solid")
    twin.set_ylabel("median R-squared   -- dashed")
    ax2.set_title(
        "(b) Population confound check: activity and R-squared by agreement",
        loc="left",
        fontsize=10,
    )

    _footnote(fig, bundle)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    paths = savefig_paper(fig, "judge_agreement_dose_response", dir=fig_dir)
    plt.close(fig)
    _enrich_meta(
        paths,
        {
            "what_is_plotted": (
                "(a) per axis, the prevalence-weighted mean |one-vs-rest AUROC minus its "
                "within-activity-decile permutation-null mean| over that axis's label classes, "
                "at each judge agreement level; band = 95% bootstrap CI over a shared row "
                "resample (null held fixed). (b) median activity (solid, left log axis) and "
                "median R-squared (dashed, right axis) of the SAME feature populations, with "
                "'unresolved' added leftmost."
            ),
            "definitions": {
                "agreement level": (
                    "restricted to n_surviving == 5; 5-0 is 5/5 STRICT unanimity. Features "
                    "unanimous on <= 2 surviving draws are the LOWEST-evidence rows and are "
                    "excluded from every level"
                ),
                "excess": "observed AUROC minus the mean of the activity-stratified permutation null",
                "unresolved": "no unique modal label with >= 3 votes (any n_surviving)",
            },
            "caveats": [
                CORPUS_CAVEAT,
                "panel (a) is label-conditional; panel (b) median R-squared is label-free, which "
                "is why 'unresolved' features appear there and nowhere else",
                "the activity-stratified null is NOT centred at 0.5 (stratifying preserves the "
                "activity-label association), so excess is measured against the null's own centre",
            ],
            "n_per_axis_level": {
                a: {lv: d["n"] for lv, d in dose[a]["levels"].items()} for a in AXIS_ORDER
            },
            "agreement_gate": gate,
            "source_paths": {"r2": bundle["r2_path"], "labels": str(FULLDICT_LABELS)},
        },
    )
    return str(paths["png"])


def fig_decile_auroc(dec: dict, bundle: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    colors = _axis_colors()
    styles = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"), ((0, (3, 1, 1, 1)), "v")]
    fig, axes = plt.subplots(1, len(AXIS_ORDER), figsize=(17.0, 3.7), sharey=True)
    xs = np.arange(1, N_DECILES + 1, dtype=float)

    for ai, axis in enumerate(AXIS_ORDER):
        ax = axes[ai]
        codings = [c for c, (a, _l) in READ_CLASSES.items() if a == axis]
        for ci, coding in enumerate(codings):
            prof = dec["reads"][coding]["profile"]
            y = np.array([p["auroc"] for p in prof])
            lo = np.array([p["auroc_ci95"][0] for p in prof])
            hi = np.array([p["auroc_ci95"][1] for p in prof])
            ls, mk = styles[ci % len(styles)]
            ax.plot(
                xs,
                y,
                marker=mk,
                ls=ls,
                ms=3.4,
                lw=1.3,
                color=colors[axis],
                label=dec["reads"][coding]["class"].replace("_", " "),
            )
            ax.fill_between(xs, lo, hi, color=colors[axis], alpha=0.12, lw=0)
        ax.axhline(0.5, color=paper_palette_role("neutral"), lw=0.9, ls="--")
        ax.set_xticks(xs)
        ax.set_xlabel("activity decile (1 = least active)")
        ax.set_title(AXIS_LABEL[axis], loc="left", fontsize=9.5, color=colors[axis])
        ax.legend(frameon=False, fontsize=6.8, loc="best")
    axes[0].set_ylabel("AUROC of R-squared for class membership")

    _footnote(fig, bundle)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = savefig_paper(fig, "auroc_within_activity_decile", dir=fig_dir)
    plt.close(fig)
    span = {d["decile"]: round(d["internal_span_ratio"], 3) for d in dec["deciles"]}
    _enrich_meta(
        paths,
        {
            "what_is_plotted": (
                "within each equal-count activity decile of the analysis universe, the "
                "one-vs-rest AUROC of R-squared for membership in each judged label class "
                "(band = 95% group-conditional bootstrap CI, 2000 draws). One panel per axis; "
                "colour = axis, linestyle+marker = class within the axis."
            ),
            "definitions": {
                "AUROC": "P(R-squared of a class member > R-squared of a non-member) + 0.5 P(tie)",
                "decile": "equal-COUNT bin of activity (firing frequency per answer)",
            },
            "caveats": [
                CORPUS_CAVEAT,
                "deciles are equal-COUNT, not equal-width in log-activity: decile 1 spans "
                f"{span.get(1)}x and decile 10 spans {span.get(10)}x internally, so their "
                "endpoint values are the least tightly conditioned",
                "a decile-class cell with fewer than 10 members on either side is left blank",
            ],
            "decile_activity_spans": dec["deciles"],
            "source_paths": {"r2": bundle["r2_path"], "labels": str(FULLDICT_LABELS)},
        },
    )
    return str(paths["png"])


def fig_all_class_excess(lr: dict, bundle: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    colors = _axis_colors()
    order = lr["order_by_excess"]
    reads = lr["reads"]
    y = np.arange(len(order), dtype=float)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 0.42 * len(order) + 2.2), sharey=True)
    ax = axes[0]
    for yi, coding in zip(y, order, strict=True):
        r = reads[coding]
        c = colors[r["source_axis"]]
        exc = r["excess_over_stratified_null"]
        band = r["auroc_perm_band"]
        nm = r["auroc_perm_null_mean"]
        ci = r["auroc_ci95"]
        ax.plot(
            [ci[0] - nm, ci[1] - nm],
            [yi, yi],
            "-",
            color=c,
            lw=1.6,
            solid_capstyle="butt",
            alpha=0.9,
        )
        ax.plot([exc], [yi], "o", color=c, ms=5.0)
        ax.plot(
            [band[0] - nm, band[1] - nm],
            [yi, yi],
            "-",
            color=paper_palette_role("neutral"),
            lw=4.0,
            alpha=0.28,
            zorder=0,
        )
    ax.axvline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{AXIS_LABEL[reads[c]['source_axis']]}: {reads[c]['class'].replace('_', ' ')}"
            for c in order
        ],
        fontsize=8,
    )
    ax.set_xlabel("excess over activity-stratified null (AUROC units)")
    ax.set_title(
        "(a) All 15 label classes, sorted by excess over the stratified null",
        loc="left",
        fontsize=10,
    )

    ax2 = axes[1]
    for yi, coding in zip(y, order, strict=True):
        r = reads[coding]
        c = colors[r["source_axis"]]
        ax2.plot([r["auroc_perm_null_mean"], r["auroc"]], [yi, yi], "-", color=c, lw=1.0, alpha=0.5)
        ax2.plot([r["auroc"]], [yi], "o", color=c, ms=5.0)
        ax2.plot([r["auroc_perm_null_mean"]], [yi], "|", color=c, ms=9, mew=1.8)
    ax2.axvline(0.5, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax2.set_xlabel("AUROC (dot) and its stratified-null mean (tick)")
    ax2.set_title("(b) Raw AUROC vs its own null centre", loc="left", fontsize=10)

    _footnote(fig, bundle)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    paths = savefig_paper(fig, "label_reads_all_classes_excess", dir=fig_dir)
    plt.close(fig)
    _enrich_meta(
        paths,
        {
            "what_is_plotted": (
                "(a) per label class, the observed one-vs-rest AUROC of R-squared minus that "
                "class's activity-stratified permutation-null mean; thin line = 95% bootstrap CI "
                "recentred on the null mean, grey band = the null's own 95% interval recentred at "
                "0. (b) the same reads before recentring: dot = AUROC, tick = null mean. Rows "
                "sorted by signed excess; colour = axis."
            ),
            "definitions": {
                "excess": "AUROC minus the mean of the within-activity-decile permutation null",
                "null": "labels permuted within activity deciles (2000 draws), which preserves "
                "the activity-label association and so is NOT centred at 0.5",
            },
            "caveats": [
                CORPUS_CAVEAT,
                "one-vs-rest over the axis's labelled features only (unresolved / unclear "
                "dropped); the drop set differs by axis, so the denominators differ by row",
                "'interpretable: no' is the exact mirror of 'yes' and is not read out separately",
            ],
            "n_per_class": {
                c: {"n": reads[c]["n"], "n_positive": reads[c]["n_positive"]} for c in order
            },
            "source_paths": {"r2": bundle["r2_path"], "labels": str(FULLDICT_LABELS)},
        },
    )
    return str(paths["png"])


# ── entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument(
        "--r2-path",
        default=DEFAULT_R2_PATH,
        help="per-feature R^2 target (.npy, DICT_SIZE-wide). THE repoint parameter.",
    )
    ap.add_argument("--r2-label", default=DEFAULT_R2_LABEL)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument(
        "--only",
        choices=("dose", "decile", "reads"),
        action="append",
        help="run a subset of the three deliverables (repeatable); default all",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        import matplotlib  # noqa: F401

        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            paper_palette,
            paper_palette_role,
            savefig_paper,
            set_paper_style,
        )

        print("import-check OK", FW.__name__, PB.__name__)
        sys.exit(0)

    want = set(args.only or ("dose", "decile", "reads"))
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    r2_path = Path(args.r2_path)
    if not r2_path.is_absolute():
        r2_path = REPO / r2_path

    # One independently-seeded stream per deliverable, so an `--only` subset is
    # bit-identical to the corresponding slice of a full run.
    rngs = {k: np.random.default_rng(SEED + off) for k, off in RNG_OFFSETS.items()}
    t0 = time.time()

    agr = load_agreement(out_dir / "agreement_votes.npz")
    gate = agreement_gate(agr)
    class_counts = assert_realized_classes(agr)
    bundle = assemble(r2_path, args.r2_label, agr)

    result: dict = {
        "r2_target": {"path": str(r2_path), "label": args.r2_label, "caveat": CORPUS_CAVEAT},
        "coverage": bundle["coverage"],
        "agreement_gate": gate,
        "realized_class_counts": class_counts,
        "params": {"seed": SEED, "n_boot": args.n_boot, "n_perm": args.n_perm},
    }
    figs = []

    if "dose" in want:
        result["dose_response"] = dose_response(bundle, args.n_boot, args.n_perm, rngs["dose"])
        figs.append(fig_dose_response(result["dose_response"], bundle, gate, fig_dir))
    if "decile" in want:
        result["decile_auroc"] = decile_auroc(bundle, args.n_boot, rngs["decile"])
        figs.append(fig_decile_auroc(result["decile_auroc"], bundle, fig_dir))
    if "reads" in want:
        result["all_class_label_reads"] = all_class_label_reads(
            bundle, args.n_perm, args.n_boot, rngs["reads"]
        )
        figs.append(fig_all_class_excess(result["all_class_label_reads"], bundle, fig_dir))

    result["figures"] = figs
    result["metadata"] = C1773.repro_meta()
    out = out_dir / "label_agreement_battery.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    _log(f"wrote {out} ({time.time() - t0:.0f}s total); figures: {figs}")


if __name__ == "__main__":
    main()
