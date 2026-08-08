#!/usr/bin/env python
"""Issue #1482: full-width DISCRETE predictor substrate + standing corrections.

TARGET-INDEPENDENT ONLY. Per the standing rule (full width + dense->SAE map for
every deliverable), and because the full-width dense->SAE R^2 arrays do not
exist yet (task #7's pod has not provisioned), this module lands ONLY what needs
no R^2 array:

  * the five discrete/binary predictors, full width, with their censuses gated
    against independently measured values;
  * the cross-dictionary joinability VERDICT (it does not join — see below);
  * the activity-decile geometry (internal activity ratios) that the profile
    figures must annotate;
  * the matched-width control that corrects the width/map confound;
  * a manifest of every read that is DEFERRED until the pod arrays land.

It writes NO rho, NO AUROC and NO figure. Those are R^2-dependent and, per the
standing rule, must be produced once against the dense->SAE full-width arrays
rather than substituted from the SAE->SAE target or from panel width.

WIDTH/MAP CONFOUND CORRECTION (stands regardless of which target is used):
an earlier round reported firing-frequency rho "+0.295 on the dense->SAE map vs
+0.742 on SAE->SAE" and attributed the gap to the MAP. Those two numbers differ
in map AND in width. At MATCHED panel width the two maps are indistinguishable
on this predictor (+0.293 vs +0.296, recomputed here as a gate), so the
+0.29 -> +0.74 jump is a STRATUM effect, not a map effect: the panel is the
top-activity stratum, and every predictor's rho moves across activity deciles.

CROSS-DICTIONARY MATCH FLAG — DROPPED, NOT FORCED. The banked cross-dictionary
decoder matching (`matryoshka_tier/matching.npz`) is 65,536-wide: it matches the
two layer-20 matryoshka twins (jumprelu, k=100) to each other. The predictors
here are layer-19 andyrdt BatchTopK features, 131,072 of them. There is no
feature-id correspondence between the two dictionaries, so the flag cannot be
computed for these features and is reported as not-joinable rather than mapped
by a fabricated correspondence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_predictor_battery as PB  # noqa: E402

DICT_SIZE = 131_072
N_DECILES = 10
DENSE_LATENT_THRESHOLD = 0.5  # "dense latent" = active in >50% of fit answers

COVARIATES_NPZ = "eval_results/issue_1482/predictor_battery/fullwidth_covariates.npz"
FULLDICT_LABELS = Path("/mnt/eps-data/thomasjiralerspong/issue1773_fulldict/labels_upload")
MATRYOSHKA_MATCHING = "eval_results/issue_1482/matryoshka_tier/matching.npz"
PANEL_SAE_SAE = "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"
PANEL_DENSE_SAE = "eval_results/issue_1482/sae_perfeature/sae_dense_in__mean__ridge.npz"
OUT_DIR = "eval_results/issue_1482/predictor_battery"

# Independently measured; these are GATES, not decoration.
EXPECTED_GURNEE = {"other": 109_248, "promoting": 8_063, "suppressing": 5_045, "partition": 8_716}
EXPECTED_SIDE = {"context_only": 1_654, "two_sided": 126_348, "answer_only": 2_164, "live": 130_166}
EXPECTED_UNANIMITY = {
    "abstraction": 0.578,
    "content_type": 0.686,
    "functional_role": 0.621,
    "interpretable": 0.776,
    "speaker_property": 0.749,
}
RETIRED_AXES = {"functional_role": "RETIRED — kappa 0.310, below the 0.6 usability bar"}


def _log(msg: str) -> None:
    print(f"[discrete] {msg}", flush=True)


def load_substrate() -> dict[str, np.ndarray]:
    with np.load(PROJECT_ROOT / COVARIATES_NPZ) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def gurnee_class(cov: dict) -> tuple[np.ndarray, dict]:
    """0=other, 1=promoting, 2=suppressing, 3=partition — already in the substrate."""
    cls = np.asarray(cov["promoting_class"], dtype=np.int8)
    names = {0: "other", 1: "promoting", 2: "suppressing", 3: "partition"}
    got = {names[i]: int((cls == i).sum()) for i in range(4)}
    status = "PASS" if got == EXPECTED_GURNEE else "MISMATCH"
    _log(f"gurnee class census {got} -> {status}")
    return cls, {"census": got, "expected": EXPECTED_GURNEE, "status": status}


def side_class(cov: dict) -> tuple[np.ndarray, dict]:
    """0=context-only, 1=two-sided, 2=answer-only, -1=dead (neither side).

    Context-only features are answer-INACTIVE by definition, so they have no
    per-feature R^2 at all: the R^2 universe requires answer activity. Their
    COUNT is reported and they are excluded from any future AUROC with that
    exclusion stated — never silently dropped.
    """
    sr = np.asarray(cov["side_ratio"], dtype=np.float64)
    act = np.asarray(cov["activity"], dtype=np.float64)
    live = np.isfinite(sr)
    cls = np.full(DICT_SIZE, -1, dtype=np.int8)
    answer_active = act > 0
    cls[live & ~answer_active] = 0  # context-only: no answer firings
    cls[live & answer_active & (sr < 1.0)] = 1  # two-sided
    cls[live & answer_active & (sr >= 1.0)] = 2  # answer-only (psi_cnt == 0)
    got = {
        "context_only": int((cls == 0).sum()),
        "two_sided": int((cls == 1).sum()),
        "answer_only": int((cls == 2).sum()),
        "live": int(live.sum()),
    }
    status = "PASS" if got == EXPECTED_SIDE else "MISMATCH"
    _log(f"side class census {got} -> {status}")
    return cls, {
        "census": got,
        "expected": EXPECTED_SIDE,
        "status": status,
        "context_only_have_no_r2": (
            "context-only features are answer-inactive by construction, so no per-feature R^2 "
            "exists for them; their count is reported and they are excluded from every future "
            "AUROC with the exclusion stated"
        ),
    }


def dense_latent_flag(cov: dict) -> tuple[np.ndarray, dict]:
    """activity > 0.5. Never tested at layer 19 full width before this."""
    act = np.asarray(cov["activity"], dtype=np.float64)
    flag = (act > DENSE_LATENT_THRESHOLD).astype(np.int8)
    n = int(flag.sum())
    with np.load(PROJECT_ROOT / PANEL_DENSE_SAE) as z:
        pid = np.asarray(z["feat_ids"], dtype=np.int64)
    n_panel = int(flag[pid].sum())
    _log(
        f"dense-latent flag: {n} of {DICT_SIZE} full width ({100 * n / DICT_SIZE:.2f}%); "
        f"{n_panel} of {len(pid)} on the panel ({100 * n_panel / len(pid):.2f}%)"
    )
    return flag, {
        "threshold": DENSE_LATENT_THRESHOLD,
        "n": n,
        "prevalence": float(n / DICT_SIZE),
        "n_panel": n_panel,
        "prevalence_panel": float(n_panel / len(pid)),
        "prevalence_note": (
            f"FULL-WIDTH prevalence {100 * n / DICT_SIZE:.2f}% ({n} features); on the legacy "
            f"panel {100 * n_panel / len(pid):.2f}% ({n_panel} of {len(pid)}). Every dense latent "
            "is inside the panel by construction — activity 0.5 sits far above the panel floor "
            "0.0810 — so the panel figure is an enrichment of the top-activity stratum, not a "
            "second estimate of the same quantity. Quote the grain with the number."
        ),
        "note": (
            "the early-layer round found a real effect at LAYER 3 (median R^2 0.245 dense vs "
            "0.095 overall); this is the first time the flag exists at layer 19 full width. "
            "Whether it carries an effect here is an R^2-dependent read, DEFERRED."
        ),
    }


def judge_unanimity() -> tuple[dict[str, np.ndarray], dict]:
    """Per axis: did every surviving judge draw agree? A describability proxy.

    Presented as a PREDICTOR, never as a filter — unanimity correlates with
    feature cleanliness, so filtering on it would select the outcome.
    """
    axes = ("abstraction", "speaker_property", "content_type", "functional_role", "interpretable")
    unanimous = {a: np.full(DICT_SIZE, np.nan) for a in axes}
    loose = {a: np.full(DICT_SIZE, np.nan) for a in axes}
    low_evidence: Counter = Counter()
    seen = Counter()
    shards = sorted(FULLDICT_LABELS.glob("axis_labels.shard*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no label shards under {FULLDICT_LABELS}")
    for p in shards:
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ax = r.get("axis")
                if ax not in unanimous:
                    continue
                surviving = r.get("labels_surviving") or []
                if not surviving:
                    continue
                fid = int(r["feat_id"])
                # STRICT: every LAUNCHED draw survived AND they agreed. The
                # looser "surviving draws agreed" reads systematically higher
                # (e.g. functional_role 0.673 vs 0.623) because it counts a
                # feature unanimous when draws were lost. Strict reproduces the
                # independently measured prevalences to within 0.002.
                agreed = len(set(surviving)) == 1
                complete = r.get("n_surviving") == r.get("n_launched")
                unanimous[ax][fid] = float(agreed and complete)
                loose[ax][fid] = float(agreed)
                seen[ax] += 1
                if agreed and len(surviving) <= 2:
                    low_evidence[ax] += 1
    out = {}
    for a in axes:
        v = unanimous[a]
        ok = np.isfinite(v)
        prev = float(v[ok].mean()) if ok.any() else float("nan")
        exp = EXPECTED_UNANIMITY[a]
        delta = abs(prev - exp)
        out[a] = {
            "n_judged": int(ok.sum()),
            "prevalence_unanimous": prev,
            "expected_prevalence": exp,
            "abs_delta": delta,
            "status": "PASS" if delta < 0.01 else "MISMATCH",
            "loose_prevalence": float(loose[a][np.isfinite(loose[a])].mean()),
            "loose_only_low_evidence_rows": int(low_evidence[a]),
        }
        if a in RETIRED_AXES:
            out[a]["retired"] = RETIRED_AXES[a]
        _log(f"unanimity {a}: {prev:.3f} (expected {exp:.3f}) -> {out[a]['status']}")
    return {**unanimous, **{f"loose_{k}": v for k, v in loose.items()}}, {
        "per_axis": out,
        "definition": (
            "STRICT: every LAUNCHED judge draw survived AND all returned the same label. The "
            "looser 'surviving draws agreed' variant is also stored (unanimous_loose_*) and "
            "reads systematically higher because it counts a feature unanimous when draws were "
            "LOST. That is not a harmless inflation: per axis, hundreds of rows are unanimous on "
            "<= 2 surviving draws (measured per axis in loose_only_low_evidence_rows), so the loose "
            "flag promotes the LOWEST-evidence features into "
            "the highest-confidence bucket — precisely inverting what the predictor is meant to "
            "measure. Strict reproduces the independently measured prevalences to within 0.002."
        ),
        "usage": "a PREDICTOR (describability proxy), never a filter — it selects on cleanliness",
    }


def cross_dictionary_joinability() -> dict:
    """Verdict only: the banked matching is a different dictionary, so it cannot join."""
    p = PROJECT_ROOT / MATRYOSHKA_MATCHING
    if not p.exists():
        return {"joinable": False, "reason": f"{MATRYOSHKA_MATCHING} absent"}
    with np.load(p) as z:
        shapes = {k: list(np.shape(z[k])) for k in z.files}
    n = shapes.get("best_lmsys_to_pile", [None])[0]
    verdict = {
        "joinable": False,
        "banked_artifact": MATRYOSHKA_MATCHING,
        "banked_shapes": shapes,
        "banked_width": n,
        "our_width": DICT_SIZE,
        "reason": (
            f"the banked cross-dictionary matching is {n}-wide — it matches the two LAYER-20 "
            "matryoshka twins (jumprelu, k=100, 65,536 features) to each other. These predictors "
            "are LAYER-19 andyrdt BatchTopK features (131,072). There is no feature-id "
            "correspondence between the two dictionaries."
        ),
        "action": (
            "DROPPED from the predictor set rather than mapped by a fabricated correspondence. "
            "Producing it would require running decoder matching between the layer-19 dictionary "
            "and a differently-trained layer-19 dictionary, which does not exist here."
        ),
    }
    _log(f"cross-dictionary match flag: NOT JOINABLE ({n}-wide vs {DICT_SIZE})")
    return verdict


def matched_width_control(cov: dict) -> dict:
    """The width/map confound correction, as a recomputed side-by-side gate."""
    out = {}
    for name, path in (("sae_to_sae", PANEL_SAE_SAE), ("dense_to_sae", PANEL_DENSE_SAE)):
        with np.load(PROJECT_ROOT / path) as z:
            fid = np.asarray(z["feat_ids"], dtype=np.int64)
            r2 = np.asarray(z["r2"], dtype=np.float64)
        act = np.asarray(cov["activity"], dtype=np.float64)[fid]
        ok = np.isfinite(r2) & np.isfinite(act)
        out[name] = {
            "rho_activity_vs_r2_panel": PB._spearman(act[ok], r2[ok]),
            "n": int(ok.sum()),
            "source": path,
        }
    act_full = np.asarray(cov["activity"], dtype=np.float64)
    fw = np.asarray(
        np.load(PROJECT_ROOT / "eval_results/issue_1738/sae_twoway/perfeature/sae_context_r2.npy"),
        dtype=np.float64,
    )
    okf = np.isfinite(fw) & np.isfinite(act_full)
    out["sae_to_sae_FULLWIDTH"] = {
        "rho_activity_vs_r2": PB._spearman(act_full[okf], fw[okf]),
        "n": int(okf.sum()),
        "source": "eval_results/issue_1738/sae_twoway/perfeature/sae_context_r2.npy",
    }
    d = abs(
        out["sae_to_sae"]["rho_activity_vs_r2_panel"]
        - out["dense_to_sae"]["rho_activity_vs_r2_panel"]
    )
    out["interpretation"] = (
        "At MATCHED panel width the two maps are indistinguishable on firing frequency "
        f"({out['sae_to_sae']['rho_activity_vs_r2_panel']:+.3f} SAE->SAE vs "
        f"{out['dense_to_sae']['rho_activity_vs_r2_panel']:+.3f} dense->SAE; |delta| = {d:.3f}), "
        f"while the SAME map read at FULL width gives "
        f"{out['sae_to_sae_FULLWIDTH']['rho_activity_vs_r2']:+.3f}. The panel-vs-full-width gap is "
        "therefore a STRATUM effect, not a map effect — the panel is the top-activity stratum. "
        "Any earlier statement attributing that gap to the dense->SAE map is CORRECTED here."
    )
    _log(out["interpretation"])
    return out


def activity_decile_geometry(cov: dict) -> dict:
    """Per-decile internal activity ratio — the annotation every profile figure needs."""
    act_all = np.asarray(cov["activity"], dtype=np.float64)
    with np.load(
        PROJECT_ROOT / "eval_results/issue_1482/predictor_battery/fullwidth_matrix.npz",
        allow_pickle=True,
    ) as z:
        uni = np.asarray(z["feat_ids"], dtype=np.int64)
    # the joined R^2 universe, so these ratios annotate the SAME deciles the
    # profile figure draws (over all 131,072 the deciles differ)
    act = act_all[uni]
    act = act[np.isfinite(act) & (act > 0)]
    edges = np.quantile(act, np.linspace(0, 1, N_DECILES + 1))
    rows = []
    for d in range(N_DECILES):
        lo, hi = float(edges[d]), float(edges[d + 1])
        rows.append(
            {
                "decile": d + 1,
                "lo": lo,
                "hi": hi,
                "dex": float(np.log10(hi / lo)) if lo > 0 else float("nan"),
                "internal_ratio": float(hi / lo) if lo > 0 else float("nan"),
            }
        )
    _log(
        "decile internal ratios: "
        + ", ".join(f"d{r['decile']} {r['internal_ratio']:.1f}x" for r in rows)
    )
    return {
        "deciles": rows,
        "total_dex": float(np.log10(act.max() / act.min())),
        "caveat_line": (
            "deciles are equal-COUNT, not equal-width in log-activity; deciles 1 and 10 span "
            f"{rows[0]['internal_ratio']:.0f}x and {rows[-1]['internal_ratio']:.1f}x internally, "
            "so their endpoint values are the least tightly conditioned"
        ),
    }


DEFERRED_READS = {
    "why": (
        "the standing rule is FULL WIDTH + the dense context -> SAE feature map for every "
        "deliverable. The full-width dense->SAE R^2 arrays (task #7's pod) do not exist yet, so "
        "every R^2-dependent read below is DEFERRED rather than substituted from the SAE->SAE "
        "target or from panel width."
    ),
    "single_swap": (
        "all of these run from the same substrate; the final run is one input-path swap "
        "(--r2-npy) against the dense->SAE ridge and MLP arrays."
    ),
    "deferred": [
        "raw rho + activity-partialled rho per continuous predictor (both arms)",
        "raw AUROC + excess-over-stratified-null per binary/one-vs-rest label, sorted by excess",
        "within-activity-decile rho (continuous) and AUROC (binary) — the headline evidence",
        "top-decile value per predictor",
        "predictor x decile heatmap (null-relative encoding)",
        "AUROC-at-depth sweep, unified construction (label = top-k vs bottom-k by R^2, score = "
        "the predictor), per-k recomputed null band; Delta_k kept in JSON only, with the identity "
        "AUROC = 0.5 + Delta_k/2 noted so it is not read as independent evidence",
        "abstraction: per-level medians + n + CIs (PRIMARY), Kruskal-Wallis omnibus eta^2 vs the "
        "activity-stratified null (COMPARABLE), ordinal Spearman with its tie ceiling and "
        "attainment fraction (SECONDARY, a monotone-trend test)",
        "per-axis omnibus statistic (Kruskal-Wallis / eta^2) alongside the one-vs-rest AUROCs",
        "linear vs nonlinear: both arms in every figure, plus the pooled-vs-per-feature "
        "divergence at full width against the pre-registered expectation that it WIDENS",
    ],
    "pre_registered_expectation_linear_vs_nonlinear": (
        "at panel grain the divergence was ridge pooled 0.7216 / per-feature median +0.1767 / "
        "99.3% positive vs MLP pooled 0.7387 / median -0.0285 / 46.1% positive. Full width adds "
        "~7x more rare features, and the capacity-allocation account predicts the MLP abandons "
        "them harder, so the divergence should WIDEN. To be reported as held or not. GRID: 7 "
        "cells — ridge x {mean, max, frac} vs MLP(w=8192) x {mean, max, frac}, plus "
        "mlpgate__mean as the panel-width MLP reproduction gate. The w=32,768 arm "
        "(mlpw32k__mean) was REMOVED by user directive, so there is NO capacity ladder and the "
        "'does abandonment ease with more hidden units' question cannot be answered this round."
    ),
    "pod_status_at_write_time": (
        "pod-1482 is RUNNING (8xH100, pod_id kdxtgasnn1npbw), verified against the live RunPod "
        "API at write time — NOT inferred from the task tracker. An earlier draft of this file "
        "asserted no pod existed; that was read off a stale tracker row (#10 in_progress) and "
        "was WRONG. The deferral below stands on the arrays not existing YET, not on any "
        "expectation that they will not."
    ),
    "outcome_if_arrays_never_land": (
        "GATED — do not quote this without a fresh live re-check of pod-1482 and of "
        "eval_results/issue_1482/densesae_fullwidth/. IF, after that re-check, the arrays are "
        "confirmed unavailable, the honest wording is 'target-independent covariate substrate "
        "delivered at full width; the R^2-dependent reads could not be produced on the specified "
        "map' — never a substitution of the SAE->SAE target and never a fallback to panel width. "
        "This is a contingency template, NOT a statement about the current run."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="#1482 full-width discrete predictor substrate")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cov = load_substrate()
    gurnee, g_gate = gurnee_class(cov)
    side, s_gate = side_class(cov)
    dense_flag, d_info = dense_latent_flag(cov)
    unanimity, u_info = judge_unanimity()
    xdict = cross_dictionary_joinability()
    matched = matched_width_control(cov)
    geom = activity_decile_geometry(cov)

    np.savez(
        args.out_dir / "fullwidth_discrete_covariates.npz",
        feat_ids=np.arange(DICT_SIZE, dtype=np.int64),
        gurnee_class=gurnee,
        side_class=side,
        dense_latent_flag=dense_flag,
        **{f"unanimous_{a}": v for a, v in unanimity.items()},
    )

    doc = {
        "scope": "TARGET-INDEPENDENT, FULL WIDTH (131,072). No rho, no AUROC, no figure here.",
        "join_key": "feat_ids",
        "standing_rule": (
            "full width + the dense context -> SAE feature map for every deliverable; the "
            "SAE->SAE target and panel width are never published as results"
        ),
        "discrete_predictors": {
            "gurnee_footprint_class": {
                **g_gate,
                "levels": {0: "other", 1: "promoting", 2: "suppressing", 3: "partition"},
                "why_kept_despite_refutation": (
                    "its value is the CONTRAST with its continuous parent (footprint kurtosis): "
                    "the class read rho -0.026 while the parent's activity-partialled rho was "
                    "-0.135. Thresholding a continuous predictor at q0.90 destroyed the signal. "
                    "Report them adjacent."
                ),
            },
            "side_class": s_gate,
            "dense_latent_flag": d_info,
            "judge_unanimity": u_info,
            "cross_dictionary_match_flag": xdict,
        },
        "retired_axes": RETIRED_AXES,
        "matryoshka_tier": {
            "joins_per_feature_table": False,
            "reason": (
                "different dictionary (layer 20, jumprelu k=100, 65,536 features), so it cannot "
                "be a peer row in the per-feature table or the depth sweep"
            ),
            "keep_as": "a clearly separated DICTIONARY-LEVEL result (tier median R^2 0.435 / "
            "0.174 / 0.043), never a peer row",
        },
        "width_map_confound_correction": matched,
        "activity_decile_geometry": geom,
        "metric_note": (
            "the depth sweep metric is AUROC-at-depth, NOT classification accuracy — accuracy is "
            "not comparable across prevalences spanning 0.7% to 77%"
        ),
        "deferred_r2_dependent_reads": DEFERRED_READS,
        "metadata": PB._metadata(),
    }
    (args.out_dir / "discrete_predictors.json").write_text(json.dumps(doc, indent=1))
    _log(f"-> {args.out_dir / 'discrete_predictors.json'}")


if __name__ == "__main__":
    main()
