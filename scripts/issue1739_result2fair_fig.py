"""Result 2 (FAIR PROTOCOL, v2) for #1739: every method gets the same training data.

Renders ONE combined figure under `figures/issue_1739/result2_fair_v2/`:

  result2_fair_v2.{png,pdf,meta.json}

and writes `eval_results/issue_1739/result2_fair_v2/result2_fair_v2_points.json`
(points + coverage + the fair-vs-committed comparison + the linear-collapse
check + the two matched-layer checks). The V2 SIBLING PATH is deliberate
(2026-08-06): the committed `result2_fair/result2_fair_points.json` is the
source of the already-shipped four-panel / five-method figures
(issue1739_result2_fourpanel_fig.py, issue1739_result2_fivemethod_fig.py,
issue1739_r35_mapquality_vs_pred.py all read it, and it carries the
pv_map_mlp / reg_map_mlp slots this re-score no longer produces) — that tree
is never overwritten.

LINEAR MAP ONLY (user scope decision 2026-08-06): the map_kind=mlp pass AND
the MLP readout (arm19) are dropped — nothing in the target five-method
figure is an MLP, and the equivalence argument is specific to the linear map.

x is grouped by EVALUATION SETTING; each setting carries one bar per METHOD
CELL, faceted by behaviour; legend grouped by methodology family:

  reads the context          pv_context      arm1_ctx_e1        PV projected on context
                             regression_ctx  arm4_ridge_ctx     ridge, context -> DV
  reads the mapped answer    pv_map_linear   arm6_map_proj_e1   PV on mapped answer
                             reg_map_linear  arm7_map_ridge_pred
                                             ridge FIT AND EVALUATED on the mapped
                                             answer (its fit absorbs systematic map
                                             distortion — cannot be damaged by it)
                             regression_realfit_mapped  arm8_map_ridge_true
                                             arm12's fitted w (fit on the REAL answer)
                                             APPLIED TO THE MAPPED answer — the
                                             map-error-sensitive comparator; one za
                                             RidgeJob serves arm8 + arm12 (fit shared,
                                             eval matrices split), so arm8:arm12 is the
                                             fitted analogue of arm6:arm11. Its record
                                             carries rho_matched_arm12_layer (rho at
                                             arm12's frozen layer — a diagnostic FIELD,
                                             never a bar; compare()'s
                                             arm8_matched_layer_check)
  control (shuffled map)     regression_shuffled_map  arm20_shuffled_map_ridge
                                             ridge, SHUFFLED-weight-mapped answer -> DV
                                             (control, never a method; predicted to
                                             match arms 4/7 since the row permutation
                                             preserves the map's rank; its record also
                                             carries rho_matched_arm7_layer — the rho at
                                             arm7's committed frozen layer, a diagnostic
                                             FIELD, never a bar — surfaced in compare()'s
                                             arm20_matched_layer_check)
  reads the real answer      oracle          arm11_oracle_proj  PV on real answer
                             regression_real_answer  arm12_oracle_reg  ridge, real answer -> DV
                                             (arm8 + arm12 + arm20: fair-roster
                                             follow-ups 2026-08-06)

Under the fair protocol every method shares ONE training-data allowance: the
map + whitening are the ADD/union condition (generic WildChat pool + eliciting
train pairs), and the label-consuming readouts (arms 4/7/8/12/20) train on ALL
judged training data (eliciting train budget cell + the judged WildChat train
split). Scored rows come from `eval_results/issue_1739/result2_fair_v2/<b>/`
(scripts/issue1739_result2fair_score.py). Fair-v2 summaries do not exist until
the re-score lands — collect() fails loud ("fair summary absent") until then;
cells missing from a LANDED summary surface as OMITTED coverage rows (missing
bar, never a zero).

The LINEAR-COLLAPSE CHECK: an earlier spec dropped regression-on-mapped-answer
on the argument that a linear map composed with a linear readout collapses to
a context regression; that holds only for linear+linear, so arm7(linear) is
kept and max |rho(arm7,linear) - rho(arm4)| across the fair linear cells is
REPORTED — the collapse is measured, not assumed.

Per-setting reliability CEILINGS (sqrt split-half r_yy, committed
`result1_spread/spread_stats.json`) are drawn as segments spanning the
context-BOUNDED cells only — every method except the real-answer arms is a
deterministic function of the context (a mapped answer, linear or MLP, is a
deterministic function of the context), so all are bounded; the real-answer
arms' input shares information with the DV's judge noise and is NOT. The
WildChat ceiling was computed on the FULL rung; this figure's WildChat column
evaluates its held-out ~20% split (caveat carried in meta).

Metric: Spearman rho_frozen (the spec's Plot line; its prose says R^2 — the
Plot line wins, flagged, same convention as the committed v3 figure).

Sycophancy's corrected OOD rungs are mid-flight on a sibling round; its OOD
column is the EXISTING aita rung, labelled exactly as Result 1 labels it, with
an in-figure pending note.

Pure aggregation over committed + fair-scored artifacts: no fits, no GPU.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from issue1739_recut_common import BEHAVIORS, ROOT  # noqa: E402

# The Result 1 label tables, imported (not copied) so the figures agree by
# construction at whatever state the source file is in.
from issue1739_result1_spread_fig_v2 import (  # noqa: E402
    SETTING_IDENTITY,
    SETTING_ROLE,
)

EVAL = ROOT / "eval_results/issue_1739"
# V2 sibling paths (2026-08-06) — the committed result2_fair/ tree is the
# source of already-shipped figures and is never overwritten (see docstring).
OUT_FIG = ROOT / "figures/issue_1739/result2_fair_v2"
OUT_NUM = EVAL / "result2_fair_v2"

FAIR_SUMMARY = {b: EVAL / "result2_fair_v2" / b / "all_arms_spearman.json" for b in BEHAVIORS}
V3_POINTS = EVAL / "result2_methods_v3/result2_v3_points.json"
SPREAD_STATS = EVAL / "result1_spread/spread_stats.json"
TRAIT_AUG = {
    "evil": EVAL / "result2_trait_aug/evil/all_arms_spearman.add_generic_matched_swap.json",
    "sycophancy": EVAL / "result2_trait_aug/sycophancy/all_arms_spearman.add_swap.json",
    "hallucination": EVAL / "result2_trait_aug/hallucination/all_arms_spearman.add_swap.json",
}

BUDGET_L = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}
# Full labelled pool per behaviour: evil's committed wide rows are tagged
# L=6468 while the ADD rows say L=8000 — both realize the identical full row
# set (budget_l >= n_ctx), per the v3 figure's labelled_budget_check.
FULL_POOL_L = {"evil": 6468, "sycophancy": 16000, "hallucination": 16000}
SETTINGS = {
    "evil": ["pvsynth", "wildchat_rung", "train", "hhrt", "toxicchat"],
    "sycophancy": ["pvsynth", "wildchat_rung", "train", "aita"],
    "hallucination": ["pvsynth", "wildchat_rung", "train", "nqopen", "simpleqa"],
}
FABRICATION_SETTINGS = {("hallucination", s) for s in ("train", "nqopen", "simpleqa")}
PENDING_SYCO_OOD = (
    "sycophancy's corrected OOD rungs (5 genuinely-OOD eval sets, sibling round: staged, "
    "generated, captured; judge + scoring remain) are PENDING — the OOD column shown is the "
    "existing held-out-Reddit aita rung and will be rebuilt when the corrected rungs land"
)

# (arm_id, map_kind) -> method slot. arm12's slot deliberately does NOT reuse
# or extend the `oracle` name (that slot is arm11's PV-on-real-answer read);
# `regression_real_answer` parallels `regression_ctx` — readout family +
# input, no map token because the arm consumes no map.
METHOD_OF = {
    ("arm1_ctx_e1", "linear"): "pv_context",
    ("arm4_ridge_ctx", "linear"): "regression_ctx",
    ("arm11_oracle_proj", "linear"): "oracle",
    ("arm12_oracle_reg", "linear"): "regression_real_answer",
    ("arm6_map_proj_e1", "linear"): "pv_map_linear",
    ("arm7_map_ridge_pred", "linear"): "reg_map_linear",
    ("arm8_map_ridge_true", "linear"): "regression_realfit_mapped",
    ("arm20_shuffled_map_ridge", "linear"): "regression_shuffled_map",
}
FAIR_READOUT_ARMS = (
    "arm4_ridge_ctx",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm12_oracle_reg",
    "arm20_shuffled_map_ridge",
)
GROUPS = [
    (
        "reads the context",
        [
            ("pv_context", "PV projected on context", "#08519C", None),
            ("regression_ctx", "regression: context -> behaviour", "#6BAED6", None),
        ],
    ),
    (
        "reads the mapped answer (linear map)",
        [
            ("pv_map_linear", "PV on mapped answer", "#8C3000", None),
            (
                "reg_map_linear",
                "regression on mapped answer (fit AND evaluated on mapped)",
                "#CC5500",
                None,
            ),
            # Wong orange — a FRESH color (the old #F0A868 was the MLP-readout
            # family in committed figures; one color = one meaning): arm8 is
            # arm12's fitted w read on the map output — the fit/eval split IS
            # the arm, so the label leads with it (detail in methods_note).
            (
                "regression_realfit_mapped",
                "regression: fit on REAL answer -> applied to MAPPED answer",
                "#E69F00",
                None,
            ),
        ],
    ),
    # The control group sits BEFORE the real-answer group so the
    # ceiling-bounded slots stay a CONTIGUOUS prefix of SLOTS (render() draws
    # the ceiling segment min..max over bounded indices): the shuffled-map
    # ridge IS a deterministic function of the context, hence bounded.
    (
        "control: shuffled-weight map (never a method)",
        [
            # Wong reddish-purple — the control-family color the oodspread
            # figures assign (one color = one arm family across figures);
            # "xx" hatch so the bar reads as a control at a glance.
            (
                "regression_shuffled_map",
                "regression on SHUFFLED-map answer (rank-preserving control)",
                "#CC79A7",
                "xx",
            ),
        ],
    ),
    (
        "reads the real answer (ceiling)",
        [
            ("oracle", "PV on real answer", "#00694C", "//"),
            # Wong bluish-green — the SAME color the fivemethod figure assigns
            # to "Ridge regression on real answer" (one color = one meaning
            # across figures); lighter shade = regression member, darker
            # (#00694C) = PV member, matching that figure's convention.
            ("regression_real_answer", "regression: real answer -> behaviour", "#009E73", "//"),
        ],
    ),
]
SLOTS = [m for _t, ms in GROUPS for m, _l, _c, _h in ms]
COLOR = {m: c for _t, ms in GROUPS for m, _l, c, _h in ms}
HATCH = {m: h for _t, ms in GROUPS for m, _l, _c, h in ms}
# The reliability ceiling bounds every deterministic function of the CONTEXT —
# all cells except the real-answer arms (whose input shares information with
# the DV's judge noise).
REAL_ANSWER_SLOTS = ("oracle", "regression_real_answer")
CEILING_BOUNDED = tuple(m for m in SLOTS if m not in REAL_ANSWER_SLOTS)

GROUP_WIDTH = 0.84
BAR_WIDTH = GROUP_WIDTH / len(SLOTS)

# Committed rows for the readout arms the v3 points file does not carry
# (arm4 everywhere; arm7 generic-map from the wide trees). Sources mirror the
# jobd `--collect-generic-only` extractor's map plus the wide pvsynth tree.
COMMITTED_ARM_SOURCES = {
    "main_train": EVAL / "{behavior}/arm_results/all_arms_spearman.json",
    "wide_pvsynth": EVAL / "wide/pvsynth/{behavior}/all_arms_spearman.json",
    "wide_wc": EVAL / "wide/wildchat_rung/{behavior}/all_arms_spearman.json",
    "wide_ood": EVAL / "wide_ood/{behavior}_transfer.jsonl",
    "gapfill": EVAL / "result2_gapfill/merged/{behavior}/arm_results/all_arms_spearman.json",
}


def _slice_ok(r: dict, beh: str) -> bool:
    if r.get("regime") != "e1" or r.get("variant") != "context_end":
        return False
    if str(r.get("u_rung_label")) != "full":
        return False
    bl = r.get("budget_l")
    return bl is None or int(bl) >= FULL_POOL_L[beh]


def committed_arm_rows(arm: str) -> dict[tuple[str, str], dict]:
    """{(behavior, setting): committed row} for one arm (eliciting-only readout)."""
    out: dict[tuple[str, str], dict] = {}

    def keep(beh: str, setting: str, r: dict, source: str) -> None:
        key = (beh, setting)
        if key in out:
            return  # first source in the fixed order wins; all report the same slice
        rho = r.get("rho_frozen", r.get("rho"))
        if rho is None:
            return
        out[key] = {
            "rho": float(rho),
            "ci": list(r.get("ci_frozen") or r.get("ci") or []) or None,
            "n_eval": r.get("n_eval"),
            "source": source,
        }

    for beh in BEHAVIORS:
        for src_name, tmpl in COMMITTED_ARM_SOURCES.items():
            path = Path(str(tmpl).format(behavior=beh))
            if not path.exists():
                continue
            if path.suffix == ".jsonl":
                rows = []
                with path.open() as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        rows += obj["rows"] if isinstance(obj, dict) and "rows" in obj else [obj]
            else:
                doc = json.loads(path.read_text())
                rows = (doc.get("transfer_rows") or []) + (doc.get("arm_rows") or [])
            for r in rows:
                if r.get("arm") != arm or r.get("f_u") is not None:
                    continue
                if not _slice_ok(r, beh):
                    continue
                if r.get("draw") not in (None, 0) or r.get("seed") not in (None, 0):
                    continue
                setting = r.get("eval_rung") or ("train" if src_name == "main_train" else None)
                if setting in (None, "train_in_split"):
                    setting = "train"
                if setting in SETTINGS[beh]:
                    keep(beh, setting, r, f"{src_name}:{path.relative_to(ROOT)}")
    return out


def committed_arm7_add() -> dict[tuple[str, str], dict]:
    """{(behavior, setting): row} for arm7 under the committed ADD map (r2aug)."""
    out: dict[tuple[str, str], dict] = {}
    for beh, path in TRAIT_AUG.items():
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        for r in doc.get("transfer_rows") or []:
            if r.get("arm") != "arm7_map_ridge_pred" or r.get("map_condition") != "add":
                continue
            if r.get("variant") != "context_end" or r.get("regime") != "e1":
                continue
            if int(r.get("budget_l", -1)) != BUDGET_L[beh]:
                continue
            setting = r["eval_rung"]
            if setting == "train_in_split":
                setting = "train"
            if setting not in SETTINGS[beh]:
                continue
            key = (beh, setting)
            if key in out:
                continue
            out[key] = {
                "rho": float(r["rho_frozen"]),
                "ci": list(r.get("ci_frozen") or []) or None,
                "n_eval": r.get("n_eval"),
                "source": f"r2aug-add:{path.relative_to(ROOT)}",
            }
    return out


def committed_v3() -> dict[tuple[str, str, str], dict]:
    doc = json.loads(V3_POINTS.read_text())
    out = {}
    for p in doc["points"]:
        out[(p["behavior"], p["setting"], p["method"])] = p
    return out


def ceilings() -> dict[tuple[str, str], float]:
    doc = json.loads(SPREAD_STATS.read_text())
    return {
        (c["behavior"], c["setting"]): float(c["ceiling_sqrt_r_yy"])
        for c in doc["cells"]
        if c.get("ceiling_sqrt_r_yy") is not None
    }


def collect() -> tuple[list[dict], list[dict], dict]:
    """Fair points + coverage + meta from the fair-scored summaries."""
    recs: list[dict] = []
    coverage: list[dict] = []
    fair_meta: dict[str, dict] = {}
    for beh in BEHAVIORS:
        path = FAIR_SUMMARY[beh]
        if not path.exists():
            raise SystemExit(f"fair summary absent: {path} — pull issue1739_result2_fair from HF")
        doc = json.loads(path.read_text())
        fair_meta[beh] = {
            k: doc["meta"].get(k)
            for k in (
                "leakage",
                "readout_protocol",
                "map_kind_resolution",
                "map_reuse_note",
                "arm8_note",
                "arm12_note",
                "arm20_note",
                "matched_layer_note",
                "frozen_layer_sources",
                "dv_scaling_note",
                "dv_construct_caveat",
                "variant_scope",
                "git_commit",
                "wall_s",
            )
        }
        rows = doc.get("transfer_rows") or []
        seen: dict[tuple[str, str, str], dict] = {}
        for r in rows:
            if (r["arm"], r.get("map_kind", "linear")) not in METHOD_OF:
                continue
            key = (r["eval_rung"], r["arm"], r.get("map_kind", "linear"))
            if key in seen:
                raise SystemExit(f"duplicate fair row {beh}/{key}")
            seen[key] = r
        for setting in SETTINGS[beh]:
            for (arm, kind), method in METHOD_OF.items():
                r = seen.get((setting, arm, kind))
                if r is None:
                    coverage.append(
                        dict(
                            behavior=beh,
                            setting=setting,
                            method=method,
                            status="OMITTED",
                            reason=f"no fair-scored row for {arm}/{kind} at {setting} "
                            "(see transfer_skips in the fair summary)",
                        )
                    )
                    continue
                rec = dict(
                    behavior=beh,
                    setting=setting,
                    method=method,
                    arm_id=arm,
                    map_kind=kind,
                    rho=float(r["rho_frozen"]),
                    ci=list(r.get("ci_frozen") or []) or None,
                    n_replicates=1,
                    n_eval=int(r["n_eval"]),
                    layer=r.get("layer"),
                    map_condition="add (fair protocol)",
                    readout=("fair union readout" if arm in FAIR_READOUT_ARMS else "label-free"),
                    dv_construct=(
                        "fabricated_fraction_rescaled_x100"
                        if (beh, setting) in FABRICATION_SETTINGS
                        else "trait_rubric_graded_0_100"
                    ),
                    source_file=str(path.relative_to(ROOT)),
                )
                # Matched-layer companions (FIELDS on the same method's
                # record, never a second slot/bar) — arm20 carries
                # rho_matched_arm7_layer, arm8 carries rho_matched_arm12_layer;
                # see the fair summary's meta.matched_layer_note.
                for k in (
                    "rho_matched_arm7_layer",
                    "rho_matched_arm12_layer",
                    "matched_layer",
                    "matched_layer_idx",
                    "n_eval_matched",
                    "matched_note",
                ):
                    if k in r:
                        rec[k] = r[k]
                recs.append(rec)
                coverage.append(
                    dict(behavior=beh, setting=setting, method=method, status="EXISTS", reason="")
                )
        if beh == "sycophancy":
            for method in SLOTS:
                coverage.append(
                    dict(
                        behavior=beh,
                        setting="corrected_ood_rungs",
                        method=method,
                        status="PENDING-syco-OOD",
                        reason=PENDING_SYCO_OOD,
                    )
                )
    return recs, coverage, fair_meta


def compare(recs: list[dict]) -> dict:
    """Fair vs committed: per-cell deltas, per-setting rankings, collapse check."""
    v3 = committed_v3()
    a4 = committed_arm_rows("arm4_ridge_ctx")
    a7_generic = committed_arm_rows("arm7_map_ridge_pred")
    a7_add = committed_arm7_add()
    a8 = committed_arm_rows("arm8_map_ridge_true")
    a12 = committed_arm_rows("arm12_oracle_reg")
    fair = {(r["behavior"], r["setting"], r["method"]): r for r in recs}

    def committed_for(beh: str, setting: str, method: str):
        if method in ("pv_context", "oracle", "pv_map_linear"):
            v3m = {"pv_context": "pv_context", "oracle": "oracle", "pv_map_linear": "map_linear"}
            c = v3.get((beh, setting, v3m[method]))
            if c is None:
                return None, None
            src = c.get("source_file")
            if str(c.get("map_condition", "")).startswith("generic"):
                src = f"{src} [committed value was GENERIC-map]"
            return float(c["rho"]), src
        if method == "regression_ctx":
            c = a4.get((beh, setting))
            return (None, None) if c is None else (c["rho"], c["source"])
        if method == "regression_real_answer":
            c = a12.get((beh, setting))
            return (None, None) if c is None else (c["rho"], c["source"])
        if method == "regression_realfit_mapped":
            c = a8.get((beh, setting))
            return (None, None) if c is None else (c["rho"], c["source"])
        if method == "reg_map_linear":
            c = a7_add.get((beh, setting)) or a7_generic.get((beh, setting))
            if c is None:
                return None, None
            src = c["source"]
            if not src.startswith("r2aug-add"):
                src = f"{src} [committed value was GENERIC-map]"
            return c["rho"], src
        return None, None  # regression_shuffled_map: no committed cells (new arm)

    cells = []
    rank_changes = []
    for beh in BEHAVIORS:
        for setting in SETTINGS[beh]:
            fair_rank = []
            comm_rank = []
            for method in SLOTS:
                f = fair.get((beh, setting, method))
                c_rho, c_src = committed_for(beh, setting, method)
                cells.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        method=method,
                        fair_rho=None if f is None else f["rho"],
                        committed_rho=c_rho,
                        delta=None if (f is None or c_rho is None) else f["rho"] - c_rho,
                        committed_source=c_src,
                    )
                )
                if f is not None:
                    fair_rank.append((f["rho"], method))
                if c_rho is not None:
                    comm_rank.append((c_rho, method))
            fr = [m for _v, m in sorted(fair_rank, reverse=True)]
            cr = [m for _v, m in sorted(comm_rank, reverse=True)]
            rank_changes.append(
                dict(
                    behavior=beh,
                    setting=setting,
                    fair_ranking=fr,
                    committed_ranking_over_available_committed_cells=cr,
                    ranking_changed_on_shared_methods=([m for m in fr if m in set(cr)] != cr),
                )
            )
    # linear-collapse check: ridge on the LINEAR-mapped answer vs ridge on the
    # context, same fair readout training set — measured, not assumed.
    collapse = []
    for beh in BEHAVIORS:
        for setting in SETTINGS[beh]:
            f7 = fair.get((beh, setting, "reg_map_linear"))
            f4 = fair.get((beh, setting, "regression_ctx"))
            if f7 and f4:
                collapse.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        rho_arm7_linear=f7["rho"],
                        rho_arm4=f4["rho"],
                        abs_diff=abs(f7["rho"] - f4["rho"]),
                    )
                )
    max_collapse = max((c["abs_diff"] for c in collapse), default=None)
    # arm20 freezing-confound check: arm20's own-argmax read vs its rho at
    # arm7's committed frozen layer, against arm7 at the same setting — the
    # like-for-like (same-layer-convention) shuffled-control comparison.
    matched = []
    for beh in BEHAVIORS:
        for setting in SETTINGS[beh]:
            f20 = fair.get((beh, setting, "regression_shuffled_map"))
            if f20 is None or f20.get("rho_matched_arm7_layer") is None:
                continue
            f7 = fair.get((beh, setting, "reg_map_linear"))
            matched.append(
                dict(
                    behavior=beh,
                    setting=setting,
                    arm20_rho_own_argmax=f20["rho"],
                    arm20_own_layer=f20.get("layer"),
                    arm20_rho_at_arm7_layer=f20["rho_matched_arm7_layer"],
                    arm7_frozen_layer=f20.get("matched_layer"),
                    arm7_rho=None if f7 is None else f7["rho"],
                    arm7_minus_arm20_matched=(
                        None if f7 is None else f7["rho"] - f20["rho_matched_arm7_layer"]
                    ),
                )
            )
    # arm8 layer-asymmetry check: arm8 and arm12 share ONE fitted w but freeze
    # independently off their own profiles — arm8_rho_at_arm12_layer makes the
    # arm12-vs-arm8 gap a same-layer read (the map-error damage estimate).
    matched8 = []
    for beh in BEHAVIORS:
        for setting in SETTINGS[beh]:
            f8 = fair.get((beh, setting, "regression_realfit_mapped"))
            if f8 is None or f8.get("rho_matched_arm12_layer") is None:
                continue
            f12 = fair.get((beh, setting, "regression_real_answer"))
            matched8.append(
                dict(
                    behavior=beh,
                    setting=setting,
                    arm8_rho_own_frozen=f8["rho"],
                    arm8_own_layer=f8.get("layer"),
                    arm8_rho_at_arm12_layer=f8["rho_matched_arm12_layer"],
                    arm12_frozen_layer=f8.get("matched_layer"),
                    arm12_rho=None if f12 is None else f12["rho"],
                    arm12_minus_arm8_matched=(
                        None if f12 is None else f12["rho"] - f8["rho_matched_arm12_layer"]
                    ),
                )
            )
    return {
        "cells": cells,
        "rankings": rank_changes,
        "arm20_matched_layer_check": {
            "definition": (
                "arm20 (ridge on the SHUFFLED-weight mapped answer) freezes on its own "
                "train-OOF argmax while arm7 freezes on the committed modal convention; "
                "arm20_rho_at_arm7_layer re-reads arm20 at arm7's frozen layer so "
                "arm7_minus_arm20_matched is a same-layer-convention gap — the "
                "layer-selection-confound-free version of the shuffled-control read. "
                "A diagnostic on the arm20 record; the figure bar stays the own-argmax "
                "read."
            ),
            "per_cell": matched,
        },
        "arm8_matched_layer_check": {
            "definition": (
                "arm8 (arm12's fitted w applied to the MAPPED answer) and arm12 share "
                "ONE za RidgeJob but freeze independently — each picks its layer off its "
                "own rho profile (arm8's evaluated on mapped, arm12's on real), so the "
                "raw arm12-vs-arm8 gap carries a layer-selection component. "
                "arm8_rho_at_arm12_layer re-reads arm8 at arm12's frozen layer so "
                "arm12_minus_arm8_matched is the same-layer map-error damage estimate. "
                "A diagnostic on the arm8 record; the figure bar stays arm8's own "
                "committed-modal frozen read."
            ),
            "per_cell": matched8,
        },
        "linear_collapse_check": {
            "definition": (
                "max |rho(arm7, linear map) - rho(arm4)| across the fair linear cells — "
                "the empirical check on the linear+linear collapse argument (which holds "
                "only for linear map + linear readout; it does not hold for MLP readouts "
                "or MLP maps). Note the two are not algebraically identical even in the "
                "linear case: the ridge penalty acts in different coordinates."
            ),
            "max_abs_diff": max_collapse,
            "per_cell": collapse,
        },
    }


def render(recs: list[dict]) -> int:
    table = {(r["behavior"], r["setting"], r["method"]): r for r in recs}
    ceil = ceilings()
    vals: list[float] = []
    for r in recs:
        vals.append(r["rho"])
        if r["ci"]:
            vals.extend(r["ci"])
    ylim = (min(0.0, min(vals)) - 0.05, max(1.0, max(vals)) + 0.02)

    set_paper_style("blog", font_scale=0.85)
    fig = plt.figure(figsize=(24.0, 10.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.34], width_ratios=[5, 4, 5])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.03, hspace=0.05)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    bounded_slots = [SLOTS.index(m) for m in CEILING_BOUNDED]
    n_bars = 0
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        settings = SETTINGS[beh]
        xs = list(range(len(settings)))
        ax.axhline(0.0, color="#B0B0B0", linewidth=0.9, zorder=1)

        fab = [s for s in settings if (beh, s) in FABRICATION_SETTINGS]
        if fab and len(fab) < len(settings):
            edge = min(settings.index(s) for s in fab)
            ax.axvline(edge - 0.5, color="#444444", linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)

        for slot_i, slot in enumerate(SLOTS):
            offset = -GROUP_WIDTH / 2 + (slot_i + 0.5) * BAR_WIDTH
            for x, s in zip(xs, settings, strict=True):
                rec = table.get((beh, s, slot))
                if rec is None:
                    continue  # no scored number: no bar, never a zero bar
                ax.bar(
                    [x + offset],
                    [rec["rho"]],
                    width=BAR_WIDTH,
                    color=COLOR[slot],
                    hatch=HATCH[slot],
                    edgecolor="#FFFFFF",
                    linewidth=0.25,
                    zorder=3,
                )
                if rec["ci"]:
                    lo = max(0.0, rec["rho"] - rec["ci"][0])
                    hi = max(0.0, rec["ci"][1] - rec["rho"])
                    ax.errorbar(
                        [x + offset],
                        [rec["rho"]],
                        yerr=np.array([[lo], [hi]]),
                        fmt="none",
                        ecolor="#333333",
                        elinewidth=0.6,
                        capsize=0,
                        zorder=4,
                    )
                n_bars += 1

        # per-setting reliability ceiling: a segment spanning ONLY the
        # ceiling-bounded (context-based) bars — never one line per facet.
        for x, s in zip(xs, settings, strict=True):
            c = ceil.get((beh, s))
            if c is None:
                continue
            lo_off = -GROUP_WIDTH / 2 + min(bounded_slots) * BAR_WIDTH
            hi_off = -GROUP_WIDTH / 2 + (max(bounded_slots) + 1) * BAR_WIDTH
            ax.plot(
                [x + lo_off, x + hi_off],
                [c, c],
                color="#B00020",
                linewidth=1.6,
                linestyle=(0, (3, 2)),
                zorder=5,
            )

        ax.set_xticks(xs)
        labels = []
        for s in settings:
            lab = f"{SETTING_ROLE[s]}\n{SETTING_IDENTITY[(beh, s)]}"
            if beh == "sycophancy" and s == "aita":
                lab += "\n(corrected OOD rungs pending)"
            if s == "wildchat_rung":
                lab += "\n(held-out 20% split)"
            labels.append(lab)
        ax.set_xticklabels(labels, fontsize=7.4, rotation=14, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, max(xs) + 0.6)
        ax.set_ylim(*ylim)
        ax.set_title(beh, loc="left")

    axes[0].set_ylabel("Spearman rho, prediction vs judged behaviour expression")
    axes[1].set_xlabel("evaluation setting")
    fig.suptitle(
        "Result 2 (fair protocol): every method trains on ALL the training data — "
        "map on generic + eliciting; behaviour readouts on all judged data",
        x=0.006,
        ha="left",
    )

    handles_x = 0.0
    widths = (0.15, 0.29, 0.23, 0.14)
    for (gtitle, methods), w in zip(GROUPS, widths, strict=True):
        leg = legend_ax.legend(
            handles=[
                Patch(facecolor=c, hatch=h, edgecolor="#FFFFFF", linewidth=0.25, label=lbl)
                for _m, lbl, c, h in methods
            ],
            title=gtitle,
            # single column up to 3 entries — a 2-col layout pushes a long
            # label into the NEXT group's x-region (2026-08-06 render defect)
            ncol=1 if len(methods) <= 3 else 2,
            loc="upper left",
            alignment="left",
            frameon=False,
            fontsize=8.0,
            borderpad=0.0,
            bbox_to_anchor=(handles_x, 1.0),
            bbox_transform=legend_ax.transAxes,
        )
        leg.get_title().set_fontsize(8.4)
        leg.get_title().set_fontweight("semibold")
        legend_ax.add_artist(leg)
        handles_x += w

    marks = legend_ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color="#B00020",
                linewidth=1.6,
                linestyle=(0, (3, 2)),
                label="reliability ceiling sqrt(r_yy) — spans context-based\n"
                "bars only (the real-answer arms are not bounded by it)",
            ),
        ],
        title="reading the marks",
        ncol=1,
        loc="upper left",
        alignment="left",
        frameon=False,
        fontsize=8.0,
        borderpad=0.0,
        bbox_to_anchor=(handles_x, 1.0),
        bbox_transform=legend_ax.transAxes,
    )
    marks.get_title().set_fontsize(8.4)
    marks.get_title().set_fontweight("semibold")
    legend_ax.add_artist(marks)

    note = (
        "All bars are single replicate (draw 0 / seed 0); CI = within-draw paired bootstrap "
        "over eval contexts.   Missing bar = no scored number at that cell; never a zero.   "
        "Hallucination right of the dashed divider scores fabrication rate x100, a different "
        "construct from the 0-100 trait rubric elsewhere.   WildChat ceilings were computed on "
        "the full rung; the WildChat column evaluates its held-out 20% split.   Sycophancy's "
        "corrected OOD rungs are pending (sibling round); its OOD column is the existing aita "
        "rung."
    )
    fig.text(0.006, 0.008, note, ha="left", va="bottom", fontsize=8.0, color="#4A4A4A", wrap=True)

    savefig_paper(fig, "result2_fair_v2", dir=OUT_FIG)
    plt.close(fig)
    return n_bars


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NUM.mkdir(parents=True, exist_ok=True)
    recs, coverage, fair_meta = collect()
    comparison = compare(recs)
    n_bars = render(recs)
    if n_bars != len(recs):
        raise SystemExit(f"plotted {n_bars} bars but collected {len(recs)} records")
    payload = dict(
        metric="Spearman rho_frozen (prediction vs judged behaviour expression)",
        metric_note="the spec's prose says R^2, its Plot line says rho; the Plot line wins",
        protocol=(
            "FAIR v2: LINEAR map only (the map_kind=mlp pass and the arm19 MLP readout "
            "were dropped by user scope decision 2026-08-06); map + whitening = ADD/union "
            "(generic WildChat pool + eliciting train pairs) on one shared whitening; the "
            "label-consuming readouts (arms 4/7/8/12/20) train on the eliciting train "
            "budget cell UNION the judged WildChat train split (sha1 mod-5 bucket 4 held "
            "out); the projection arms are label-free and share the whitening + eval "
            "subsets; pvsynth judged data enters only through r_B (recorded deviation "
            "for the regression readouts)"
        ),
        methods_note=(
            "Five methods (PV/regression on context, mapped answer, real answer) plus "
            "two comparator/control reads: arm8_map_ridge_true (arm12's fitted w — fit "
            "on the REAL answer — APPLIED to the MAPPED answer; one za RidgeJob serves "
            "arm8 + arm12, fit shared / eval matrices split, the map-error-sensitive "
            "regression comparator) and arm20_shuffled_map_ridge (regression on the "
            "SHUFFLED-weight mapped answer — a rank-preserving falsification CONTROL of "
            "the linear-collapse argument, never a method). arm8/arm12/arm20 were added "
            "by the 2026-08-06 follow-ups; the nonlinear-map arms were dropped the same "
            "day by user scope decision — their absence is deliberate, not a failure."
        ),
        operating_slice=dict(
            regime="e1", variant="context_end", u_rung_label="add", budget_l=BUDGET_L
        ),
        ceiling_semantics=(
            "sqrt split-half r_yy from result1_spread/spread_stats.json, drawn as per-setting "
            "segments spanning the context-BOUNDED cells (every method except the two "
            "real-answer arms — a mapped answer is a deterministic function of the context, "
            "so arm8's w·M(z) and arm20's shuffled read are both bounded); the real-answer "
            "arms' input shares information with the DV's judge noise, so the ceiling does "
            "not bound them. WildChat ceilings computed on the full rung while the fair "
            "WildChat column evaluates the held-out 20% split"
        ),
        labelling=(
            "setting row labels are the Result 1 figure's two-part ROLE + IDENTITY strings, "
            "imported from scripts/issue1739_result1_spread_fig_v2.py at its current "
            "working-tree state"
        ),
        pending_syco_ood=PENDING_SYCO_OOD,
        fair_meta=fair_meta,
        coverage=coverage,
        comparison_vs_committed=comparison,
        n_points=len(recs),
        points=recs,
    )
    (OUT_NUM / "result2_fair_v2_points.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_FIG / 'result2_fair_v2.png'} ({n_bars} bars)")
    print(f"wrote {OUT_NUM / 'result2_fair_v2_points.json'} ({len(recs)} records)")
    cc = comparison["linear_collapse_check"]
    print(f"linear-collapse check: max |arm7(linear) - arm4| = {cc['max_abs_diff']}")


if __name__ == "__main__":
    main()
