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

  (4) ACTIVITY-MATCHED agreement dose-response
      Round 1 found median activity is NOT flat across agreement levels and moves
      in axis-dependent directions, so (1) alone cannot separate "unanimity
      denoises the label" (H1) from "unanimity selects a lower-activity, worse-
      predicted population, suppressing the gain" (H2). Coarsened exact matching
      on activity bins equalizes the populations; what survives is H1.

  (5) unresolved vs resolved R^2
      functional_role's judge-unresolved features carry 3.5x the median R^2 of its
      unanimous ones, and every labelled read drops them. Checked on all five
      axes, raw AND against the activity-stratified null (unresolved features are
      also more active, so the raw gap can be pure composition).

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
# The production target: #1482 dense-context -> SAE-answer ridge, mean pooling, at
# full 131,072 width. Supersedes the provisional #1738 SAE->SAE stand-in.
DEFAULT_R2_PATH = "data/issue_1482/densesae_dl/ridge__mean_perfeature.npz"
DEFAULT_R2_LABEL = "#1482 dense-context -> SAE ridge, mean pooling (full width)"
# Same corpus as the covariates, VERIFIED three ways (2026-08-03):
#   * the full-width ridge reproduces the banked #1482 dense->SAE PANEL per-feature
#     R^2 to max |delta| 2.98e-08 over all 16,384 panel features;
#   * that banked panel npz's own `activity` array is BIT-IDENTICAL (max delta 0.0)
#     to the `activity` covariate this battery joins on;
#   * the build's inputs meta pins the #1482 splits and store row order.
# So the cross-corpus caveat the #1738 stand-in required is RESOLVED, not merely
# restated. `--corpus-note` carries whatever is true for the target in use.
DEFAULT_CORPUS_NOTE = (
    "single corpus: R^2 and the activity / judged-label covariates are both the "
    "#1482 SINGLE-TURN read (verified against the banked #1482 panel)"
)
LEGACY_1738_CORPUS_CAVEAT = (
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
RNG_OFFSETS = {"dose": 0, "decile": 1, "reads": 2, "matched": 3, "unresolved": 4}

# Activity-matching resolution: equal-count bins over the analysis universe.
MATCH_BINS = 50
MATCH_BINS_SENSITIVITY = (25, 100)


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


def load_r2_target(r2_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Load a per-feature R^2 target as (r2, scorable, membership_rule).

    Two accepted shapes:
      `.npz`  the production form — keys feat_ids / r2 / scored. Membership is the
              producer's OWN `scored` flag, never inferred from finite R^2: a
              zero-variance holdout column is UNSCORED and its R^2 is undefined,
              which is a different statement from "the value happens to be NaN".
              `feat_ids` is honoured as the scatter index rather than assumed to
              be 0..DICT_SIZE-1.
      `.npy`  the legacy DICT_SIZE-wide form (the #1738 stand-in), which carries no
              scored flag, so membership degrades to finite R^2.
    """
    if r2_path.suffix == ".npz":
        z = np.load(r2_path)
        missing = {"feat_ids", "r2", "scored"} - set(z.files)
        if missing:
            raise AssertionError(f"{r2_path}: missing keys {sorted(missing)}")
        fid = np.asarray(z["feat_ids"], dtype=np.int64)
        r2 = np.full(DICT_SIZE, np.nan, dtype=np.float64)
        scorable = np.zeros(DICT_SIZE, dtype=bool)
        r2[fid] = np.asarray(z["r2"], dtype=np.float64)
        scorable[fid] = np.asarray(z["scored"], dtype=bool)
        # `scored` is authoritative; a divergence from finiteness is a producer
        # inconsistency worth naming rather than silently resolving either way.
        n_dis = int((scorable != np.isfinite(r2)).sum())
        if n_dis:
            _log(f"WARNING {r2_path.name}: scored disagrees with isfinite on {n_dis} features")
        _log(
            f"r2 target: {r2_path.name} — {int(scorable.sum())} of {DICT_SIZE} SCORED "
            f"({DICT_SIZE - int(scorable.sum())} zero-variance/unscored), "
            f"scored-vs-isfinite disagreements {n_dis}"
        )
        return r2, scorable, "producer `scored` flag (npz)"
    r2 = np.asarray(np.load(r2_path), dtype=np.float64)
    if r2.shape != (DICT_SIZE,):
        raise AssertionError(f"{r2_path}: expected ({DICT_SIZE},), got {r2.shape}")
    _log(f"r2 target: {r2_path.name} — legacy .npy, membership = finite R^2")
    return r2, np.isfinite(r2), "finite R^2 (legacy .npy, no scored flag)"


def assemble(r2_path: Path, r2_label: str, corpus_note: str, agr: dict) -> dict:
    """Join labels + agreement with the durable full-width covariate substrate.

    Universe = judged (interpretable axis) AND SCORABLE R^2 AND answer-active.
    On the legacy #1738 `.npy` target "scorable" degrades to finite R^2, which
    reproduced the banked `fullwidth_label_reads.json` exactly (n = 114,076 on the
    `interpretable` read); on the production `.npz` it is the producer's `scored`
    flag, which is the authoritative statement of which columns were fit.
    """
    z = np.load(REPO / COVARIATES)
    activity = np.asarray(z["activity"], dtype=np.float64)
    feat_ids_all = np.asarray(z["feat_ids"], dtype=np.int64)
    if activity.shape != (DICT_SIZE,):
        raise AssertionError(f"covariates activity shape {activity.shape}")

    r2_all, scorable, membership_rule = load_r2_target(r2_path)

    judged = agr["interpretable"]["label"] != "unlabeled"
    active = activity > 0
    keep = judged & scorable & active
    idx = np.flatnonzero(keep)
    _log(
        f"universe: {len(idx)} of {DICT_SIZE} "
        f"(judged {int(judged.sum())}, scorable R^2 {int(scorable.sum())}, "
        f"active {int(active.sum())})"
    )
    r2u = r2_all[idx]
    dist = {
        "median": float(np.median(r2u)),
        "mean": float(np.mean(r2u)),
        "min": float(np.min(r2u)),
        "max": float(np.max(r2u)),
        "frac_negative": float((r2u < 0).mean()),
        "frac_below_minus_1": float((r2u < -1).mean()),
        "frac_below_minus_10": float((r2u < -10).mean()),
        "note": (
            "reported because the arms differ sharply in the negative tail; every "
            "read here is rank-based or an unclipped median, so no fraction is "
            "clipped or thresholded away"
        ),
    }
    _log(
        f"r2 over the universe: median {dist['median']:+.5f} "
        f"frac<0 {dist['frac_negative']:.4f} frac<-1 {dist['frac_below_minus_1']:.4f} "
        f"frac<-10 {dist['frac_below_minus_10']:.4f} (clipped/thresholded fraction: 0.0)"
    )

    return {
        "feat_ids": feat_ids_all[idx],
        "r2": r2u,
        "activity": activity[idx],
        "labels": {ax: agr[ax]["label"][idx] for ax in AXIS_ORDER},
        "n_surv": {ax: agr[ax]["n_surv"][idx] for ax in AXIS_ORDER},
        "best": {ax: agr[ax]["best"][idx] for ax in AXIS_ORDER},
        "resolved": {ax: agr[ax]["resolved"][idx] for ax in AXIS_ORDER},
        "r2_path": str(r2_path),
        "r2_label": r2_label,
        "corpus_note": corpus_note,
        "membership_rule": membership_rule,
        "r2_distribution": dist,
        "coverage": {
            "dict_size": DICT_SIZE,
            "judged": int(judged.sum()),
            "scorable_r2": int(scorable.sum()),
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


# ── deliverable 4: ACTIVITY-MATCHED agreement dose-response ──────────────────


def _match_bin_edges(activity: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count activity bin edges over the ANALYSIS UNIVERSE.

    Global (not per-axis) so the matching resolution is identical across axes.
    Duplicate edges produced by the heavy tie mass at low activity are collapsed,
    so the realized bin count can fall below `n_bins`; it is reported.
    """
    q = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(activity, q))


def _smd(x: np.ndarray, y: np.ndarray) -> float:
    """Standardized mean difference, pooled-sd convention. |SMD| < 0.1 = balanced."""
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    denom = float(np.sqrt((vx + vy) / 2.0))
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / denom)


def _cem_match_levels(
    binid: np.ndarray, n_bins: int, level_idx: dict[str, np.ndarray], rng
) -> tuple[dict[str, np.ndarray], dict]:
    """1:1:1 coarsened exact matching of the agreement levels on activity bins.

    In every bin holding at least one feature from EVERY level, each level is
    randomly down-sampled to that bin's minimum count. The matched sets then
    share an IDENTICAL activity-bin distribution by construction, so residual
    imbalance is within-bin only. Vectorized: one `lexsort` per level (bin-major,
    random within bin) plus a per-bin rank filter — no Python loop over bins.
    """
    counts = np.vstack([np.bincount(binid[idx], minlength=n_bins) for idx in level_idx.values()])
    keep = counts.min(axis=0)
    matched: dict[str, np.ndarray] = {}
    for lv, idx in level_idx.items():
        b = binid[idx]
        order = np.lexsort((rng.random(len(idx)), b))
        b_sorted = b[order]
        starts = np.searchsorted(b_sorted, np.arange(n_bins), side="left")
        rank = np.arange(len(b_sorted)) - starts[b_sorted]
        matched[lv] = idx[order[rank < keep[b_sorted]]]
    return matched, {
        "n_bins_requested": int(n_bins),
        "n_bins_nonempty_all_levels": int((keep > 0).sum()),
        "n_matched_per_level": int(keep.sum()),
    }


MIN_MATCHED = 300  # below this a level's matched read is not attempted


def matched_dose_response(
    bundle: dict,
    unmatched: dict,
    n_boot: int,
    n_perm: int,
    rng,
    bin_settings: tuple[int, ...],
) -> dict:
    """Deliverable 4: the dose-response recomputed on ACTIVITY-MATCHED level sets.

    Round 1 found median activity is NOT flat across agreement levels and moves in
    axis-dependent directions, so the unmatched dose-response cannot separate
      H1 unanimity DENOISES the label (label->R^2 association genuinely strengthens)
      H2 unanimity SELECTS a different feature population (toward low activity,
         which is worse-predicted), which SUPPRESSES the apparent gain.
    Matching removes the population difference; what survives is H1.

    The activity-stratified null already absorbs within-level activity structure,
    so this is the complementary control: it equalizes the BETWEEN-level activity
    distribution the null cannot see.
    """
    r2, act = bundle["r2"], bundle["activity"]
    log_act = np.log10(np.maximum(act, np.finfo(np.float64).tiny))
    out: dict[str, dict] = {}

    for n_bins in bin_settings:
        edges = _match_bin_edges(act, n_bins)
        binid = np.searchsorted(edges, act, side="right")
        nb = len(edges) + 1
        per_axis: dict[str, dict] = {}
        for axis in AXIS_ORDER:
            lab = bundle["labels"][axis]
            five = bundle["n_surv"][axis] == N_DRAWS
            best, resolved = bundle["best"][axis], bundle["resolved"][axis]
            drop = AXIS_DROP[axis]
            level_idx = {
                lvl: np.flatnonzero(five & (best == bval) & resolved & ~np.isin(lab, drop))
                for lvl, bval in zip(AGREEMENT_LEVELS, BEST_FOR_LEVEL, strict=True)
            }
            if min(len(v) for v in level_idx.values()) == 0:
                per_axis[axis] = {"status": "not-attempted — an agreement level is empty"}
                continue

            matched, diag = _cem_match_levels(binid, nb, level_idx, rng)
            ref = AGREEMENT_LEVELS[-1]  # 5-0 is the balance reference
            balance = {}
            for lvl in AGREEMENT_LEVELS:
                pre, post = level_idx[lvl], matched[lvl]
                balance[lvl] = {
                    "n_before": int(len(pre)),
                    "n_after": int(len(post)),
                    "retention": float(len(post) / len(pre)) if len(pre) else float("nan"),
                    "median_activity_before": float(np.median(act[pre])),
                    "median_activity_after": (
                        float(np.median(act[post])) if len(post) else float("nan")
                    ),
                    "smd_log_activity_vs_5_0_before": _smd(log_act[pre], log_act[level_idx[ref]]),
                    "smd_log_activity_vs_5_0_after": (
                        _smd(log_act[post], log_act[matched[ref]])
                        if len(post) > 1 and len(matched[ref]) > 1
                        else float("nan")
                    ),
                }

            n_min = min(len(v) for v in matched.values())
            if n_min < MIN_MATCHED:
                per_axis[axis] = {
                    "status": (
                        f"INFEASIBLE — smallest matched level has {n_min} features "
                        f"(< {MIN_MATCHED}); the agreement levels barely overlap in "
                        "activity on this axis, so the comparison is not well-posed"
                    ),
                    "matching": diag,
                    "balance": balance,
                }
                _log(f"matched[{n_bins}] {axis}: INFEASIBLE (n_min={n_min})")
                continue

            levels = {}
            for lvl in AGREEMENT_LEVELS:
                sel = matched[lvl]
                read = _axis_level_read(r2[sel], act[sel], lab[sel], axis, n_boot, n_perm, rng)
                if read is not None:
                    levels[lvl] = read
            entry: dict = {
                "status": "ok",
                "matching": diag,
                "balance": balance,
                "levels": levels,
            }
            if len(levels) == len(AGREEMENT_LEVELS):
                lo = levels[AGREEMENT_LEVELS[0]]["excess_weighted_abs"]
                hi = levels[AGREEMENT_LEVELS[-1]]["excess_weighted_abs"]
                entry["matched_shift"] = float(hi - lo)
                entry["matched_ratio"] = float(hi / lo) if lo != 0 else float("nan")
                um = unmatched.get(axis, {}).get("levels", {})
                if len(um) == len(AGREEMENT_LEVELS):
                    ulo = um[AGREEMENT_LEVELS[0]]["excess_weighted_abs"]
                    uhi = um[AGREEMENT_LEVELS[-1]]["excess_weighted_abs"]
                    entry["unmatched_shift"] = float(uhi - ulo)
                    entry["shift_difference_matched_minus_unmatched"] = float(
                        (hi - lo) - (uhi - ulo)
                    )
                    entry["shift_survival_fraction"] = (
                        float((hi - lo) / (uhi - ulo)) if (uhi - ulo) != 0 else float("nan")
                    )
            per_axis[axis] = entry
            _log(
                f"matched[{n_bins}] {axis}: n/level={n_min} "
                f"retention={balance[AGREEMENT_LEVELS[0]]['retention']:.2f}/"
                f"{balance[AGREEMENT_LEVELS[-1]]['retention']:.2f} "
                f"matched_shift={entry.get('matched_shift', float('nan')):+.4f} "
                f"(unmatched {entry.get('unmatched_shift', float('nan')):+.4f})"
            )
        out[str(n_bins)] = per_axis
    return {
        "primary_bins": str(bin_settings[0]),
        "bin_settings": [int(b) for b in bin_settings],
        "method": (
            "coarsened exact matching, 1:1:1, on equal-count activity bins of the "
            "analysis universe; each bin down-sampled to its across-level minimum"
        ),
        "by_bins": out,
    }


# ── deliverable 5: unresolved vs resolved R^2 ────────────────────────────────


def _median_diff_boot(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Bootstrap draws of median(a) - median(b), chunked over draws."""
    na, nb = len(a), len(b)
    chunk = _chunk_for(max(na, nb))
    draws, done = [], 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        da = np.median(a[rng.integers(0, na, size=(na, m))], axis=0)
        db = np.median(b[rng.integers(0, nb, size=(nb, m))], axis=0)
        draws.append(da - db)
        done += m
    return np.concatenate(draws)


def unresolved_vs_resolved(bundle: dict, n_boot: int, n_perm: int, rng) -> dict:
    """Deliverable 5: are the judge-UNRESOLVED features better predicted?

    Round 1 found functional_role's 7,705 unresolved features carry median R^2
    0.02064 vs 0.00589 for its unanimous ones. Every labelled read on this line
    silently drops the unresolved set, so if the pattern generalizes it is a
    selection effect on the labelled subset itself.

    Reported per axis BOTH raw and activity-controlled: unresolved features are
    also MORE ACTIVE, and activity predicts R^2, so the raw median gap can be
    pure composition. The activity-stratified permutation null is what separates
    the two — excess ~ 0 means "it is just activity".
    """
    r2, act = bundle["r2"], bundle["activity"]
    n_med = min(n_boot, 1000)  # median CI is the expensive draw; 1000 is ample
    out: dict[str, dict] = {}
    for axis in AXIS_ORDER:
        lab, resolved = bundle["labels"][axis], bundle["resolved"][axis]
        judged = lab != "unlabeled"
        res = judged & resolved & ~np.isin(lab, AXIS_DROP[axis])
        unres = judged & ~resolved
        n_dropped = int((judged & resolved & np.isin(lab, AXIS_DROP[axis])).sum())
        if unres.sum() < 30 or res.sum() < 30:
            out[axis] = {"status": "not-attempted — a group has fewer than 30 features"}
            continue

        r_u, r_r = r2[unres], r2[res]
        med_diff = float(np.median(r_u) - np.median(r_r))
        med_ci = PB._ci(_median_diff_boot(r_u, r_r, n_med, rng))
        point, ci = FW.auroc_with_boot(r_u, r_r, n_boot, rng)

        # activity-stratified null on the SAME AUROC
        comb = res | unres
        idx = np.flatnonzero(comb)
        order = np.argsort(r2[idx], kind="stable")
        idx_s = idx[order]
        lab_int = unres[idx_s].astype(np.int16)  # 1 = unresolved
        rank1 = _rank1(r2[idx_s])
        strata = FW._decile_of(act[idx_s])
        null = _perm_null_class_auroc(rank1, lab_int, strata, 2, n_perm, rng)
        null_mean = float(np.nanmean(null[1]))
        col = null[1][np.isfinite(null[1])]

        out[axis] = {
            "status": "ok",
            "n_unresolved": int(unres.sum()),
            "n_resolved": int(res.sum()),
            "n_resolved_but_dropped_label": n_dropped,
            "median_r2_unresolved": float(np.median(r_u)),
            "median_r2_resolved": float(np.median(r_r)),
            "median_r2_difference": med_diff,
            "median_r2_difference_ci95": med_ci,
            "median_r2_ratio": (
                float(np.median(r_u) / np.median(r_r)) if np.median(r_r) != 0 else float("nan")
            ),
            "median_activity_unresolved": float(np.median(act[unres])),
            "median_activity_resolved": float(np.median(act[res])),
            "auroc_unresolved_gt_resolved": float(point),
            "auroc_ci95": ci,
            "auroc_perm_null_mean": null_mean,
            "auroc_perm_band": [
                float(np.percentile(col, 2.5)) if len(col) else float("nan"),
                float(np.percentile(col, 97.5)) if len(col) else float("nan"),
            ],
            "excess_over_stratified_null": float(point - null_mean),
        }
        _log(
            f"unresolved {axis}: n={int(unres.sum())} medR2 {np.median(r_u):.5f} vs "
            f"{np.median(r_r):.5f} (x{out[axis]['median_r2_ratio']:.2f}) "
            f"AUROC {point:.4f} null {null_mean:.4f} excess "
            f"{out[axis]['excess_over_stratified_null']:+.4f}"
        )
    return out


# ── figures ──────────────────────────────────────────────────────────────────


def _axis_colors():
    from explore_persona_space.analysis.paper_plots import paper_palette

    return dict(zip(AXIS_ORDER, paper_palette(len(AXIS_ORDER)), strict=True))


def _footnote(fig, bundle: dict) -> None:
    fig.text(
        0.005,
        0.004,
        f"R^2 target: {bundle['r2_label']}  |  {bundle['corpus_note']}",
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
                bundle["corpus_note"],
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
                bundle["corpus_note"],
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
                bundle["corpus_note"],
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


def fig_matched_dose(md: dict, bundle: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    colors = _axis_colors()
    prim = md["by_bins"][md["primary_bins"]]
    xs = np.arange(len(AGREEMENT_LEVELS), dtype=float)

    fig = plt.figure(figsize=(17.0, 7.4))
    gs = fig.add_gridspec(2, 10, height_ratios=[1.0, 0.95], hspace=0.42, wspace=1.5)

    for i, axis in enumerate(AXIS_ORDER):
        ax = fig.add_subplot(gs[0, 2 * i : 2 * i + 2])
        c = colors[axis]
        um = md["_unmatched"][axis]["levels"]
        uy = np.array([um[lv]["excess_weighted_abs"] for lv in AGREEMENT_LEVELS])
        ulo = np.array([um[lv]["excess_weighted_abs_ci95"][0] for lv in AGREEMENT_LEVELS])
        uhi = np.array([um[lv]["excess_weighted_abs_ci95"][1] for lv in AGREEMENT_LEVELS])
        ax.plot(xs, uy, "o-", ms=4.2, lw=1.6, color=c, label="unmatched")
        ax.fill_between(xs, ulo, uhi, color=c, alpha=0.14, lw=0)
        e = prim[axis]
        if e.get("status") == "ok":
            lv = e["levels"]
            my = np.array([lv[k]["excess_weighted_abs"] for k in AGREEMENT_LEVELS])
            mlo = np.array([lv[k]["excess_weighted_abs_ci95"][0] for k in AGREEMENT_LEVELS])
            mhi = np.array([lv[k]["excess_weighted_abs_ci95"][1] for k in AGREEMENT_LEVELS])
            ax.plot(xs, my, "s--", ms=4.0, lw=1.6, color=c, alpha=0.95, label="activity-matched")
            ax.fill_between(xs, mlo, mhi, color=c, alpha=0.10, lw=0, hatch="///")
            sub = f"matched n/level = {e['matching']['n_matched_per_level']:,}"
        else:
            ax.text(
                0.5,
                0.5,
                "matching\ninfeasible",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color=paper_palette_role("neutral"),
            )
            sub = "not well-posed"
        ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(list(AGREEMENT_LEVELS))
        ax.set_title(f"{AXIS_LABEL[axis]}\n{sub}", loc="left", fontsize=8.8, color=c)
        ax.set_xlabel("judge agreement")
        if i == 0:
            # short label: the full definition would overflow the panel height
            ax.set_ylabel("excess over\nstratified null", fontsize=8.4)
            ax.legend(frameon=False, fontsize=7.2)

    ax_b = fig.add_subplot(gs[1, 0:5])
    y = np.arange(len(AXIS_ORDER), dtype=float)[::-1]
    for yi, axis in zip(y, AXIS_ORDER, strict=True):
        e = prim[axis]
        c = colors[axis]
        if e.get("status") != "ok" or "unmatched_shift" not in e:
            continue
        u, m = e["unmatched_shift"], e["matched_shift"]
        ax_b.plot([u, m], [yi, yi], "-", color=c, lw=1.4, alpha=0.6)
        ax_b.plot([u], [yi], "o", color=c, ms=6.5, mfc="white", mew=1.6)
        ax_b.plot([m], [yi], "s", color=c, ms=6.0)
    ax_b.axvline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([AXIS_LABEL[a] for a in AXIS_ORDER], fontsize=8.4)
    ax_b.set_xlabel("dose shift (5-0 minus 3-2), AUROC units")
    ax_b.set_title(
        "(b) Dose shift before (open circle) and after (filled square) activity matching",
        loc="left",
        fontsize=9.5,
    )

    ax_c = fig.add_subplot(gs[1, 5:10])
    for yi, axis in zip(y, AXIS_ORDER, strict=True):
        e = prim[axis]
        c = colors[axis]
        bal = e.get("balance")
        if not bal:
            continue
        pre = max(abs(bal[lv]["smd_log_activity_vs_5_0_before"]) for lv in AGREEMENT_LEVELS[:-1])
        post_vals = [
            abs(bal[lv]["smd_log_activity_vs_5_0_after"])
            for lv in AGREEMENT_LEVELS[:-1]
            if np.isfinite(bal[lv]["smd_log_activity_vs_5_0_after"])
        ]
        post = max(post_vals) if post_vals else float("nan")
        ax_c.plot([pre, post], [yi, yi], "-", color=c, lw=1.4, alpha=0.6)
        ax_c.plot([pre], [yi], "o", color=c, ms=6.5, mfc="white", mew=1.6)
        if np.isfinite(post):
            ax_c.plot([post], [yi], "s", color=c, ms=6.0)
        ret = bal[AGREEMENT_LEVELS[-1]]["retention"]
        # anchored at the left edge so a large pre-matching SMD cannot push the
        # annotation off the axes
        ax_c.annotate(
            f"5-0 kept {ret:.0%}",
            xy=(0.0, yi),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=6.8,
            color=c,
        )
    ax_c.axvline(0.1, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([AXIS_LABEL[a] for a in AXIS_ORDER], fontsize=8.4)
    ax_c.set_xlabel("max |standardized mean difference| in log-activity vs the 5-0 level")
    ax_c.set_title(
        "(c) Activity balance before (open) and after (filled) matching; dashed = 0.1 balance bar",
        loc="left",
        fontsize=9.5,
    )

    _footnote(fig, bundle)
    paths = savefig_paper(fig, "activity_matched_agreement_dose", dir=fig_dir)
    plt.close(fig)
    prim_bins = md["primary_bins"]
    _enrich_meta(
        paths,
        {
            "what_is_plotted": (
                "(a) per axis, the round-1 unmatched dose-response (solid circles) against the "
                "SAME statistic recomputed on activity-matched level sets (dashed squares); "
                "bands are 95% bootstrap CIs. (b) the 5-0 minus 3-2 dose shift before (open "
                "circle) and after (filled square) matching. (c) the max |standardized mean "
                "difference| in log-activity against the 5-0 level, before and after matching, "
                "with the 5-0 retention fraction annotated."
            ),
            "definitions": {
                "matching": md["method"] + f"; primary setting {prim_bins} requested bins",
                "SMD": "(mean_level - mean_5-0) / sqrt((var_level + var_5-0)/2) on log10 activity; "
                "|SMD| < 0.1 is the conventional balance bar",
                "dose shift": "excess at 5-0 minus excess at 3-2",
                "H1 vs H2": "matched shift survives => unanimity denoises the label (H1); "
                "matched shift collapses => the dose-response was population composition (H2)",
            },
            "caveats": [
                bundle["corpus_note"],
                "matching shrinks every level to the per-bin across-level minimum, so the matched "
                "5-0 read has far fewer features than the unmatched one and correspondingly wider "
                "CIs — the like-for-like comparison is matched-5-0 vs matched-3-2, not matched vs "
                "unmatched at fixed precision",
                "an axis whose levels barely overlap in activity is reported INFEASIBLE rather "
                "than given a forced estimate",
                "the activity-stratified null already absorbs WITHIN-level activity structure; "
                "matching is the complementary control on the BETWEEN-level distribution",
            ],
            "per_axis": {
                a: {
                    k: v
                    for k, v in prim[a].items()
                    if k
                    in (
                        "status",
                        "matching",
                        "balance",
                        "matched_shift",
                        "unmatched_shift",
                        "shift_difference_matched_minus_unmatched",
                        "shift_survival_fraction",
                    )
                }
                for a in AXIS_ORDER
            },
            "bin_sensitivity_settings": md["bin_settings"],
            "source_paths": {"r2": bundle["r2_path"], "labels": str(FULLDICT_LABELS)},
        },
    )
    return str(paths["png"])


def fig_unresolved(uv: dict, bundle: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    colors = _axis_colors()
    axes_ok = [a for a in AXIS_ORDER if uv.get(a, {}).get("status") == "ok"]
    y = np.arange(len(axes_ok), dtype=float)[::-1]

    fig, axs = plt.subplots(1, 2, figsize=(13.2, 3.9))

    ax = axs[0]
    for yi, axis in zip(y, axes_ok, strict=True):
        e, c = uv[axis], colors[axis]
        ax.plot(
            [e["median_r2_resolved"], e["median_r2_unresolved"]], [yi, yi], "-", color=c, lw=1.6
        )
        ax.plot([e["median_r2_resolved"]], [yi], "o", color=c, ms=6.5, mfc="white", mew=1.6)
        ax.plot([e["median_r2_unresolved"]], [yi], "s", color=c, ms=6.0)
        # A RATIO of two negative medians is not interpretable (the mlp arm's
        # medians are all < 0), so the signed DIFFERENCE — which is also what the
        # bootstrap CI is on — is the annotation; the ratio rides along only when
        # both medians are positive.
        lead = f"D {e['median_r2_difference']:+.4f}"
        if e["median_r2_unresolved"] > 0 and e["median_r2_resolved"] > 0:
            lead += f"  (x{e['median_r2_ratio']:.2f})"
        ax.annotate(
            f"{lead}   n unresolved = {e['n_unresolved']:,}",
            xy=(max(e["median_r2_resolved"], e["median_r2_unresolved"]), yi),
            xytext=(8, -2),
            textcoords="offset points",
            fontsize=7.0,
            color=c,
            va="center",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([AXIS_LABEL[a] for a in axes_ok], fontsize=8.4)
    ax.set_xlabel("median R-squared")
    # Span the DATA, never anchor at 0: the mlp arm's medians are all negative, and
    # `set_xlim(0, negative)` silently inverts the axis and pushes rows off-screen.
    _vals = [uv[a][k] for a in axes_ok for k in ("median_r2_unresolved", "median_r2_resolved")]
    _lo, _hi = min(_vals), max(_vals)
    _span = (_hi - _lo) or (abs(_hi) or 1.0)
    ax.set_xlim(_lo - 0.10 * _span, _hi + 0.85 * _span)  # right headroom for annotations
    ax.set_title(
        "(a) Median R-squared: labelled (open circle) vs judge-unresolved (filled square)",
        loc="left",
        fontsize=9.5,
    )

    ax2 = axs[1]
    for yi, axis in zip(y, axes_ok, strict=True):
        e, c = uv[axis], colors[axis]
        band = e["auroc_perm_band"]
        ax2.plot(
            band, [yi, yi], "-", color=paper_palette_role("neutral"), lw=5.0, alpha=0.3, zorder=0
        )
        ax2.plot(e["auroc_ci95"], [yi, yi], "-", color=c, lw=1.6)
        ax2.plot([e["auroc_unresolved_gt_resolved"]], [yi], "s", color=c, ms=6.0)
        ax2.plot([e["auroc_perm_null_mean"]], [yi], "|", color=c, ms=10, mew=1.8)
        # below the marker: an inline label collides with the CI bar and the null band
        ax2.annotate(
            f"excess {e['excess_over_stratified_null']:+.3f}",
            xy=(e["auroc_unresolved_gt_resolved"], yi),
            xytext=(0, -13),
            textcoords="offset points",
            fontsize=7.0,
            color=c,
            ha="center",
            va="center",
        )
    ax2.axvline(0.5, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("AUROC that an unresolved feature outranks a labelled one")
    ax2.set_title(
        "(b) Same contrast vs its activity-stratified null (tick = null mean, grey = null 95%)",
        loc="left",
        fontsize=9.5,
    )

    _footnote(fig, bundle)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = savefig_paper(fig, "unresolved_vs_resolved_r2", dir=fig_dir)
    plt.close(fig)
    _enrich_meta(
        paths,
        {
            "what_is_plotted": (
                "(a) per axis, median R-squared of the LABELLED features (open circle; resolved, "
                "drop-labels excluded) against the judge-UNRESOLVED features (filled square), "
                "with the ratio and the unresolved count annotated. (b) the same contrast as an "
                "AUROC — P(unresolved feature outranks a labelled one) — against its "
                "within-activity-decile permutation null."
            ),
            "definitions": {
                "unresolved": "no unique modal label with >= 3 of the surviving judge draws",
                "excess": "AUROC minus the mean of the activity-stratified permutation null; "
                "excess near 0 means the raw median gap is ACTIVITY COMPOSITION, not a "
                "labelability effect",
            },
            "caveats": [
                bundle["corpus_note"],
                "unresolved features are also MORE ACTIVE on every axis, and activity predicts "
                "R-squared, so panel (a) alone cannot separate labelability from activity — "
                "panel (b) is the control that does",
                "the labelled group excludes each axis's drop labels (unresolved everywhere, plus "
                "'unclear' on speaker property), so it is exactly the set every labelled read uses",
            ],
            "per_axis": {a: uv[a] for a in axes_ok},
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
    ap.add_argument(
        "--corpus-note",
        default=DEFAULT_CORPUS_NOTE,
        help="corpus relationship between the R^2 target and the covariates; rides "
        "every figure footnote and sidecar. Pass LEGACY_1738_CORPUS_CAVEAT's text "
        "when repointing back at the #1738 stand-in.",
    )
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument(
        "--only",
        choices=tuple(RNG_OFFSETS),
        action="append",
        help="run a subset of the deliverables (repeatable); default all",
    )
    ap.add_argument(
        "--match-bins",
        type=int,
        default=MATCH_BINS,
        help="primary activity-bin count for the matched dose-response",
    )
    ap.add_argument(
        "--match-bins-sensitivity",
        type=int,
        nargs="*",
        default=list(MATCH_BINS_SENSITIVITY),
        help="additional bin counts run as a robustness check (not plotted)",
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

    want = set(args.only or tuple(RNG_OFFSETS))
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

    # Merge-on-write: a partial `--only` run updates only the deliverables it
    # produced and never truncates the rest of the canonical JSON. A DIFFERENT
    # R^2 target invalidates every prior deliverable, so the merge starts empty
    # rather than silently mixing two targets' results in one file.
    out = out_dir / "label_agreement_battery.json"
    on_disk: dict = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prior_same_target = on_disk.get("r2_target", {}).get("path") == str(r2_path)
    prior: dict = on_disk if prior_same_target else {}
    if on_disk and not prior_same_target:
        _log(
            "R^2 target CHANGED vs the file on disk "
            f"({on_disk.get('r2_target', {}).get('label', '?')} -> {args.r2_label}): "
            "discarding every prior deliverable instead of merging across targets"
        )

    agr = load_agreement(out_dir / "agreement_votes.npz")
    gate = agreement_gate(agr)
    class_counts = assert_realized_classes(agr)
    bundle = assemble(r2_path, args.r2_label, args.corpus_note, agr)

    result: dict = {
        "r2_target": {
            "path": str(r2_path),
            "label": args.r2_label,
            "corpus_note": args.corpus_note,
            "membership": bundle["membership_rule"],
            "r2_transform": "none — every read is rank-based (AUROC) or an "
            "unclipped median; no clipping or thresholding is applied",
            "distribution": bundle["r2_distribution"],
        },
        "coverage": bundle["coverage"],
        "agreement_gate": gate,
        "realized_class_counts": class_counts,
        "params": {"seed": SEED, "n_boot": args.n_boot, "n_perm": args.n_perm},
    }
    figs: list[str] = []
    produced: dict[str, dict] = {}
    meta = C1773.repro_meta()

    if "dose" in want:
        result["dose_response"] = dose_response(bundle, args.n_boot, args.n_perm, rngs["dose"])
        figs.append(fig_dose_response(result["dose_response"], bundle, gate, fig_dir))
        produced["dose"] = meta
    if "decile" in want:
        result["decile_auroc"] = decile_auroc(bundle, args.n_boot, rngs["decile"])
        figs.append(fig_decile_auroc(result["decile_auroc"], bundle, fig_dir))
        produced["decile"] = meta
    if "reads" in want:
        result["all_class_label_reads"] = all_class_label_reads(
            bundle, args.n_perm, args.n_boot, rngs["reads"]
        )
        figs.append(fig_all_class_excess(result["all_class_label_reads"], bundle, fig_dir))
        produced["reads"] = meta
    if "matched" in want:
        # The matched read is reported AGAINST the unmatched one, so the unmatched
        # dose-response must come from THIS run or from a prior run on the SAME
        # R^2 target — never from a stale target.
        unmatched = result.get("dose_response")
        if unmatched is None:
            if not prior_same_target or "dose_response" not in prior:
                raise SystemExit(
                    "--only matched needs the unmatched dose-response: rerun with "
                    "`--only dose --only matched`, or ensure the existing "
                    f"{out.name} was produced against the same --r2-path"
                )
            unmatched = prior["dose_response"]
            _log("matched: unmatched reference taken from the prior run (same R^2 target)")
        bins = (args.match_bins, *[b for b in args.match_bins_sensitivity if b != args.match_bins])
        md = matched_dose_response(
            bundle, unmatched, args.n_boot, args.n_perm, rngs["matched"], bins
        )
        md["_unmatched"] = unmatched
        figs.append(fig_matched_dose(md, bundle, fig_dir))
        md.pop("_unmatched")  # do not duplicate deliverable 1 inside the JSON
        result["matched_dose_response"] = md
        produced["matched"] = meta
    if "unresolved" in want:
        result["unresolved_vs_resolved"] = unresolved_vs_resolved(
            bundle, args.n_boot, args.n_perm, rngs["unresolved"]
        )
        figs.append(fig_unresolved(result["unresolved_vs_resolved"], bundle, fig_dir))
        produced["unresolved"] = meta

    merged = dict(prior)
    merged.update(result)
    merged["figures"] = sorted(
        set(prior.get("figures", []) if prior_same_target else []) | set(figs)
    )
    prov = dict(prior.get("produced", {}) if prior_same_target else {})
    prov.update(produced)
    merged["produced"] = prov
    merged["metadata"] = meta
    out.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    _log(f"wrote {out} ({time.time() - t0:.0f}s total); figures this run: {figs}")


if __name__ == "__main__":
    main()
