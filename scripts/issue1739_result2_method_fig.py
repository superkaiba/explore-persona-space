"""Result 2 for #1739: how each method performs in each evaluation setting.

Renders three figures under `figures/issue_1739/result2_methods/`, one per
INPUT STATE the readouts are computed from, using the canonical vocabulary of
`docs/glossary_context_answer_map.md`:

  result2_methods_context.{png,pdf}     context (prefix + query) -> answer
  result2_methods_prefix_end.{png,pdf}  prefix-end state -> answer
  result2_methods_bare_query.{png,pdf}  bare query -> answer

Never "prefix map" unqualified: `prefix_end` here is the PREFIX-END STATE (the
residual stream at the last prefix token), not the query-averaged prefix vector.

Layout, colours and y-limits are identical across the three figures so they
superimpose by eye. Each figure carries one panel per behavior; within a panel
x runs over evaluation settings ordered by OODness and y is the Spearman rho
between a method's prediction and the judged behavior expression, one line per
method.

Method taxonomy is encoded in two channels, never in 16 colours:
  COLOUR  = where the readout reads (input state / through the fitted map /
            the real answer (oracle) / control floor)
  MARKER  = readout type (PV projection, label-supervised direction, ridge,
            MLP, kernel ridge, ...)
  LINESTYLE = deployability (solid = deployable, dashed = oracle, dotted =
            control).

Pure aggregation over artifacts already committed at the repo root under
`eval_results/issue_1739/` -- no fits, no GPU, no network, no new judging.
Reads only repo-root copies (never the issue-1739 worktree), so a live session
owning that worktree is untouched.

Operating slice (the "max data" cut), applied identically to every point:
  regime          e1
  u_rung_label    full   (18,793 unlabeled fit rows)
  budget_l        the behavior's maximum labelled budget -- evil 8,000,
                  sycophancy 16,000, hallucination 16,000 (LMAX in
                  issue1739_recut_common) -- for the sources that carry a
                  budget axis (train arm_rows, wide_ood, new_arm_round). The
                  WildChat, persona-vectors-synthetic and bare-query legs
                  carry a single budget each, which is already that leg's
                  full-data cut (evil 6,468 = its whole context pool;
                  sycophancy / hallucination 16,000).
  variant         the figure's input state
A method with no row at that slice is DROPPED (its line breaks / stops with an
open x); no value is ever imputed or substituted from a different slice.

Artifact map:
  <b>/arm_results/all_arms_spearman.json          held-out train rung, 16 arms,
                                                  15 replicates (5 draws x 3 seeds)
  wide_ood/<b>_transfer.jsonl                     behavior-specific OOD rungs,
                                                  9 arms (context) / 6 (prefix-end)
  new_arm_round/arm_results/merged_transfer.jsonl arms 5 / 17 / 18 only, train + OOD
  wide/wildchat_rung/<b>/all_arms_spearman.json   random WildChat, 10 arms, 1 replicate
  wide/pvsynth/<b>/all_arms_spearman.json         persona-vectors synthetic, 10 arms
  bareq_map/<b>/all_arms_spearman.json            bare-query input state, leg 1, 6 arms
  result1_spread/spread_stats.json                per-setting reliability ceiling
                                                  sqrt(r_yy) + the Result 1 spread gate
  armfill_round/                                  arms 2 / 9 / 14 on the WildChat +
                                                  persona-vectors-synthetic legs
  armfill_round3/arms101718/                      arms 10 / 17 / 18 on the same two legs
  armfill_round3/ood/<b>/                         arms 2 / 9 / 10 / 14 on the OOD rungs
                                                  (sycophancy aita; hallucination nqopen
                                                  + simpleqa), own schema, adapted on
                                                  read -- rho + CI only, never the
                                                  detection columns
  armfill_round3/jobb_evil/                       NOT read (Job B pilot, pre-fix code)

Two correctness guards the figure depends on, both asserted at load time:
  * matched target -- every method inside one (input state, behavior, setting)
    panel column is scored against the same judged eval set, checked by
    requiring one n_eval per column; a mismatch raises rather than plotting an
    unmatched comparison.
  * hallucination carries TWO DV constructs -- its own rungs (held-out
    TriviaQA, NQ-Open, SimpleQA) score a fabricated FRACTION rescaled x100,
    while its WildChat and persona-vectors-synthetic settings score the graded
    0-100 trait rubric. The two are not comparable, so the hallucination panel
    draws a construct boundary between them and settings are ordered to keep
    each construct contiguous.

Setting order deviates from the "train -> WildChat -> behavior-OOD" OODness
sketch in the request: that order interleaves hallucination's two DV
constructs, which makes a single construct boundary impossible. The order used
here -- held-out train -> behavior-specific OOD -> random WildChat ->
persona-vectors synthetic -- keeps both constructs contiguous and matches the
reporting order already committed in `issue1739_recut_common.RUNGS`.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from issue1739_recut_common import BEHAVIORS, LMAX, ROOT  # noqa: E402

ER = ROOT / "eval_results/issue_1739"
OUT_FIG = ROOT / "figures/issue_1739/result2_methods"
OUT_NUM = ROOT / "eval_results/issue_1739/result2_methods"

# --- input states -------------------------------------------------------------
# (key, activation variant, state token spliced into method labels, panel title)
INPUT_STATES = [
    ("context", "context_end", "context", "context (prefix + query) -> answer"),
    ("prefix_end", "prefix_end", "prefix-end state", "prefix-end state -> answer"),
    ("bare_query", "bareq", "bare query", "bare query -> answer"),
]

# --- evaluation settings, ordered by OODness ---------------------------------
SETTINGS = {
    "evil": ["train", "hhrt", "toxicchat", "wildchat_rung", "pvsynth"],
    "sycophancy": ["train", "aita", "wildchat_rung", "pvsynth"],
    "hallucination": ["train", "nqopen", "simpleqa", "wildchat_rung", "pvsynth"],
}
OOD_RUNGS = {
    "evil": ["hhrt", "toxicchat"],
    "sycophancy": ["aita"],
    "hallucination": ["nqopen", "simpleqa"],
}
DIAGNOSTIC_SETTING = "pvsynth"
SETTING_LABEL = {
    ("evil", "train"): "held-out\ntrain\n(DAN x\nforbidden-q)",
    ("evil", "hhrt"): "hh-rlhf\nred-team\n(OOD)",
    ("evil", "toxicchat"): "ToxicChat\n(OOD)",
    ("sycophancy", "train"): "held-out\ntrain",
    ("sycophancy", "aita"): "AITA\n(OOD)",
    ("hallucination", "train"): "held-out\nTriviaQA",
    ("hallucination", "nqopen"): "NQ-Open\n(OOD)",
    ("hallucination", "simpleqa"): "SimpleQA\n(OOD)",
}
for _b in BEHAVIORS:
    SETTING_LABEL[(_b, "wildchat_rung")] = "random\nWildChat\n(ordinary\ntraffic)"
    SETTING_LABEL[(_b, DIAGNOSTIC_SETTING)] = "persona-\nvectors\nsynthetic"

BEHAVIOR_LABEL = {
    "evil": "evil",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
}

# Hallucination's own rungs score a fabricated FRACTION; its WildChat and
# persona-vectors-synthetic settings score the graded trait rubric. Not the
# same construct, so a boundary is drawn between the two groups.
FABRICATION_RATE_SETTINGS = {("hallucination", s) for s in ("train", "nqopen", "simpleqa")}


# --- method taxonomy ----------------------------------------------------------
# COLOUR channel: where the readout reads from.
LOCUS = {
    "arm1_ctx_e1": "input_state",
    "arm2_ctx_native": "input_state",
    "arm4_ridge_ctx": "input_state",
    "arm5_mlp_ctx": "input_state",
    "arm6_map_proj_e1": "through_map",
    "arm7_map_ridge_pred": "through_map",
    "arm8_map_ridge_true": "through_map",
    "arm9_pretrain_ft": "through_map",
    "arm10_stacked": "through_map",
    "arm11_oracle_proj": "oracle",
    "arm12_oracle_reg": "oracle",
    "arm17_oracle_mlp": "oracle",
    "arm18_oracle_krr": "oracle",
    "arm3_identity_bias": "control",
    "arm13_shuffled_map": "control",
    "arm14_shuffled_pt": "control",
    "arm15_text_only": "control",
    "arm16_surface_feat": "control",
}
LOCUS_COLOR = {  # Wong colorblind-safe hexes; one colour = one meaning
    "input_state": "#0072B2",
    "through_map": "#D55E00",
    "oracle": "#009E73",
    "control": "#999999",
}
LOCUS_ORDER = ["input_state", "through_map", "oracle", "control"]
LOCUS_LABEL = {
    "input_state": "reads the input state directly",
    "through_map": "reads through the fitted map to the answer state",
    "oracle": "reads the real answer (oracle, not deployable)",
    "control": "control / floor (not a usable method)",
}
LOCUS_STYLE = {  # linestyle channel: deployability
    "input_state": "-",
    "through_map": "-",
    "oracle": "--",
    "control": ":",
}

# MARKER channel: readout type.
READOUT = {
    "arm1_ctx_e1": "pv_proj",
    "arm6_map_proj_e1": "pv_proj",
    "arm11_oracle_proj": "pv_proj",
    "arm2_ctx_native": "label_dir",
    "arm4_ridge_ctx": "ridge",
    "arm7_map_ridge_pred": "ridge",
    "arm12_oracle_reg": "ridge",
    "arm8_map_ridge_true": "ridge_real",
    "arm5_mlp_ctx": "mlp",
    "arm17_oracle_mlp": "mlp",
    "arm18_oracle_krr": "kernel_ridge",
    "arm9_pretrain_ft": "pretrain_ft",
    "arm10_stacked": "stacked",
    "arm3_identity_bias": "identity_bias",
    "arm13_shuffled_map": "shuffled_map",
    "arm14_shuffled_pt": "shuffled_pretrain",
    "arm15_text_only": "text_embedding",
    "arm16_surface_feat": "surface_features",
}
READOUT_MARKER = {
    "pv_proj": "o",
    "label_dir": "s",
    "ridge": "^",
    "ridge_real": "v",
    "mlp": "D",
    "kernel_ridge": "P",
    "pretrain_ft": "X",
    "stacked": "*",
    "identity_bias": "<",
    "shuffled_map": "x",
    "shuffled_pretrain": "+",
    "text_embedding": ">",
    "surface_features": "p",
}
READOUT_LEGEND_ORDER = [
    "pv_proj",
    "label_dir",
    "ridge",
    "ridge_real",
    "mlp",
    "kernel_ridge",
    "pretrain_ft",
    "stacked",
]
READOUT_LEGEND_LABEL = {
    "pv_proj": "persona-vector projection",
    "label_dir": "label-supervised direction",
    "ridge": "ridge",
    "ridge_real": "ridge, fit on real answers",
    "mlp": "MLP",
    "kernel_ridge": "kernel ridge",
    "pretrain_ft": "map-pretrain then fine-tune",
    "stacked": "stacked combiner",
}
CONTROL_LEGEND_ORDER = [
    "arm3_identity_bias",
    "arm13_shuffled_map",
    "arm14_shuffled_pt",
    "arm15_text_only",
    "arm16_surface_feat",
]

# Report-facing method names. "{state}" resolves per figure so a label never
# claims the wrong input state (the glossary's retired-terms rule).
METHOD_NAME = {
    "arm1_ctx_e1": "PV proj @ {state}",
    "arm2_ctx_native": "label direction @ {state}",
    "arm4_ridge_ctx": "ridge @ {state}",
    "arm5_mlp_ctx": "MLP @ {state}",
    "arm6_map_proj_e1": "PV proj @ mapped answer",
    "arm7_map_ridge_pred": "ridge @ mapped answer (fit on predicted)",
    "arm8_map_ridge_true": "ridge @ mapped answer (fit on real)",
    "arm9_pretrain_ft": "map-pretrain then fine-tune",
    "arm10_stacked": "stacked combiner",
    "arm11_oracle_proj": "PV proj @ real answer (oracle)",
    "arm12_oracle_reg": "ridge @ real answer (oracle)",
    "arm17_oracle_mlp": "MLP @ real answer (oracle)",
    "arm18_oracle_krr": "kernel ridge @ real answer (oracle)",
    "arm3_identity_bias": "control: identity + learned bias",
    "arm13_shuffled_map": "control: shuffled map",
    "arm14_shuffled_pt": "control: shuffled pretraining",
    "arm15_text_only": "control: text-embedding ridge",
    "arm16_surface_feat": "control: surface features",
}
# Draw order: controls first (they sit at the back as a floor), then the
# deployable and oracle methods over them.
METHOD_ORDER = [
    *CONTROL_LEGEND_ORDER,
    "arm1_ctx_e1",
    "arm2_ctx_native",
    "arm4_ridge_ctx",
    "arm5_mlp_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm9_pretrain_ft",
    "arm10_stacked",
    "arm11_oracle_proj",
    "arm12_oracle_reg",
    "arm17_oracle_mlp",
    "arm18_oracle_krr",
]
NEW_ARM_ROUND_ARMS = {"arm5_mlp_ctx", "arm17_oracle_mlp", "arm18_oracle_krr"}

DEPLOYABLE = [a for a in METHOD_ORDER if LOCUS[a] in ("input_state", "through_map")]


def method_name(arm: str, state_token: str) -> str:
    return METHOD_NAME[arm].format(state=state_token)


# --- loading ------------------------------------------------------------------


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _flat_jsonl(path: Path) -> list[dict]:
    """Rows from a jsonl whose lines may each wrap a list under `rows`."""
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.extend(obj["rows"] if isinstance(obj, dict) and "rows" in obj else [obj])
    return out


def _slice(rows, *, variant, eval_rung, budget_l=None, arms=None, rung_kind=None):
    """Rows at the operating slice: regime e1, full unlabeled pool, one variant."""
    sel = []
    for r in rows:
        if r.get("regime") != "e1" or r.get("u_rung_label") != "full":
            continue
        if r.get("variant") != variant or r.get("eval_rung") != eval_rung:
            continue
        if budget_l is not None and r.get("budget_l") != budget_l:
            continue
        if rung_kind is not None and r.get("rung_kind") != rung_kind:
            continue
        if arms is not None and r.get("arm") not in arms:
            continue
        if r.get("rho_frozen") is None:
            continue
        sel.append(r)
    return sel


def _aggregate(rows: list[dict], source: str) -> dict[str, dict]:
    """Collapse replicate rows (draws x seeds) to one record per method.

    rho is the mean of the committed per-replicate `rho_frozen`; the error bar
    is the committed bootstrap `ci_frozen` averaged over the same replicates
    (within-replicate bootstrap spread dominates the across-replicate spread by
    roughly 3x at the train rung, so the averaged bootstrap CI is the honest
    interval to show). Both the replicate count and the per-replicate rho list
    are carried through to the points JSON.
    """
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    out = {}
    for arm, rs in by_arm.items():
        # de-duplicate identical replicate rows (wide_ood re-reports some cells)
        seen, uniq = set(), []
        for r in rs:
            key = (r.get("draw"), r.get("seed"), r.get("layer"), r["rho_frozen"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        rhos = [float(r["rho_frozen"]) for r in uniq]
        cis = [r["ci_frozen"] for r in uniq if r.get("ci_frozen") is not None]
        n_evals = {r.get("n_eval") for r in uniq if r.get("n_eval") is not None}
        if len(n_evals) > 1:
            raise ValueError(f"{source}: arm {arm} spans multiple n_eval {sorted(n_evals)}")
        layers = sorted({int(r["layer"]) for r in uniq if r.get("layer") is not None})
        out[arm] = dict(
            rho=float(np.mean(rhos)),
            rho_sd=float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
            rho_replicates=[round(v, 6) for v in rhos],
            ci=(
                [float(np.mean([c[0] for c in cis])), float(np.mean([c[1] for c in cis]))]
                if cis
                else None
            ),
            n_replicates=len(uniq),
            n_eval=(n_evals.pop() if n_evals else None),
            layer=layers[0] if len(layers) == 1 else (layers or None),
            budget_l=uniq[0].get("budget_l"),
            # arm-fill rows carry their own file; every other source is uniform
            source_file=uniq[0].get("_source_file", source),
        )
    return out


def _adapt_ood_metric_rows(path: Path) -> list[dict]:
    """`armfill_round3/ood/<b>/ood_detection_metrics.json` -> native transfer-row shape.

    That file reports the arm-fill OOD re-score in its own schema. Only the
    rank-correlation read is carried over: `rho` is the Spearman rho at the
    arm's train-frozen layer (`issue1739_rescore_ood_armfill.py` derives
    `frozen_by_arm` from the train cells' `rho_per_layer`, then scores the eval
    rung at that layer), the same quantity `rho_frozen` names everywhere else,
    and `ci_rho` is its bootstrap CI. The detection reads that share the file
    (auroc / ap / precision_at_k) are NEVER plotted here: they binarise the DV
    at `AUROC_POS_THR = 50.0`, which is off-scale for hallucination's [0, 1]
    fabricated-fraction DV, so its auroc/ap are NaN in 720/720 rows and its
    precision@k floors at 0. Hallucination's `rho` is unaffected and is used.

    Two slice fields the file does not carry are supplied here:
      u_rung_label='full'  the producing script loads ONLY plain-ladder cells
                           (f_u=None, u_rung_label='full') and refuses to run
                           otherwise, so every row is at the full unlabeled pool
      rung_kind='eval_transfer'   every row scores an eval rung, not train
    `n_rung` becomes `n_eval` (verified equal to the `n_eval` the committed
    `wide_ood` rows report for the same rungs: aita 1304, nqopen 3167,
    simpleqa 4021), so the matched-target guard compares like with like. No
    frozen layer is reported per row, so `layer` stays absent.
    """
    with open(path) as f:
        doc = json.load(f)
    out = []
    for r in doc.get("metric_rows") or []:
        if r.get("rho") is None:
            continue
        out.append(
            {
                "behavior": r.get("behavior"),
                "variant": r.get("variant"),
                "regime": r.get("regime"),
                "u_rung_label": "full",
                "eval_rung": r.get("rung"),
                "rung_kind": "eval_transfer",
                "arm": r.get("arm"),
                "budget_l": r.get("budget_l"),
                "draw": r.get("draw"),
                "seed": r.get("seed"),
                "rho_frozen": r.get("rho"),
                "ci_frozen": r.get("ci_rho"),
                "n_eval": r.get("n_rung"),
                "_source_file": _rel(path),
            }
        )
    return out


def _armfill_rows() -> tuple[list[dict], list[str]]:
    """Fold in every arm-fill round; report which files were read.

    Two rounds have landed, filling disjoint arm sets, so both are read:
      armfill_round/            arms 2 / 9 / 14 on wildchat_rung + pvsynth
      armfill_round3/arms101718 arms 10 / 17 / 18 on the same two legs
      armfill_round3/ood        arms 2 / 9 / 10 / 14 on the behavior-specific
                                OOD rungs of all three behaviors (evil hhrt +
                                toxicchat; sycophancy aita; hallucination
                                nqopen + simpleqa)

    All three `ood/<b>/ood_detection_metrics.json` files share ONE row schema:
    the eval rung is named per row in `rung` (populated on every row, no
    nulls), NOT in `eval_rung`, which no row in any of the three files carries.
    Row -> rung attribution is therefore read straight off the file and never
    inferred from row order.
    `armfill_round3/jobb_evil/` is NOT read: it is the Job B pilot, which ran
    pre-fix code and carries no identity-bias read, so it is not a Result 2
    method source. Only the `all_arms_spearman*.json` roll-ups are read for the
    native-schema legs; their `percell/*_transfer.jsonl` siblings repeat the
    same rows verbatim.
    """
    rows: list[dict] = []
    read: list[str] = []

    for root in (
        ER / "armfill_round",
        ER / "armfill_round3" / "arms101718",
        # Job C (#1739 Result-2 hallucination max-budget gap-fill): arms
        # 7/8/12/17/18 at (u_rung=full, budget_l=16000, regime=e1) on
        # nqopen + simpleqa (+ train for 17/18), both variants. Read the
        # MERGED root only -- `legs/` repeats the same rows per leg, and
        # the sibling `hallucination/` dir is the context_end leg alone.
        # map_kind == "linear" is asserted in merged/*/gapfill_provenance.json;
        # the nonlinear kernel/mlp legs under new_arm_round/nlood/ must NOT
        # fill these cells (silent methodology error).
        ER / "result2_gapfill" / "merged",
    ):
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("all_arms_spearman*.json")):
            with open(p) as f:
                d = json.load(f)
            native = (d.get("transfer_rows") or []) + (d.get("arm_rows") or [])
            for r in native:
                r["_source_file"] = _rel(p)
            rows.extend(native)
            read.append(_rel(p))

    ood_root = ER / "armfill_round3" / "ood"
    if ood_root.is_dir():
        for p in sorted(ood_root.glob("*/ood_detection_metrics.json")):
            rows.extend(_adapt_ood_metric_rows(p))
            read.append(_rel(p))

    return rows, read


def load_points() -> tuple[dict, dict, list[str]]:
    """Build {(input_state, behavior, setting): {arm: record}} plus notes."""
    notes: list[str] = []
    armfill, armfill_read = _armfill_rows()
    notes.append(
        f"arm-fill rounds folded in: {len(armfill)} rows from {len(armfill_read)} files "
        "(armfill_round = arms 2/9/14 on wildchat_rung + pvsynth; "
        "armfill_round3/arms101718 = arms 10/17/18 on the same two legs; "
        "armfill_round3/ood = arms 2/9/10/14 on evil hhrt/toxicchat + sycophancy "
        "aita + hallucination nqopen/simpleqa). Each point's own source_file "
        "names the exact file it came from."
        if armfill_read
        else "arm-fill rounds: ABSENT at run time -- proceeded without them"
    )
    notes.append(
        "armfill_round3/jobb_evil/ deliberately NOT read: Job B pilot, ran pre-fix "
        "code and carries no identity-bias read."
    )
    notes.append(
        "From armfill_round3/ood/ only the frozen-layer Spearman rho + its bootstrap CI "
        "are used; the auroc / ap / precision_at_k columns in that file binarise the DV "
        "at 50.0 and are invalid for hallucination (NaN in 720/720 rows), so no "
        "detection metric is plotted anywhere."
    )

    table: dict[tuple[str, str, str], dict] = {}
    for state_key, variant, _tok, _title in INPUT_STATES:
        for beh in BEHAVIORS:
            lmax = LMAX[beh]
            if state_key == "bare_query":
                # Bare query exists as one leg only: the train-fit readout applied
                # to bare-query representations, evaluated on the WildChat rung.
                path = ER / "bareq_map" / beh / "all_arms_spearman.json"
                with open(path) as f:
                    rows = json.load(f)["transfer_rows"]
                leg1 = [r for r in rows if str(r.get("leg")) == "1"]
                agg = _aggregate(
                    _slice(leg1, variant="context_end", eval_rung="wildchat_rung"), _rel(path)
                )
                if agg:
                    table[(state_key, beh, "wildchat_rung")] = agg
                continue

            main_path = ER / beh / "arm_results/all_arms_spearman.json"
            with open(main_path) as f:
                main = json.load(f)
            wide_ood_path = ER / f"wide_ood/{beh}_transfer.jsonl"
            wide_ood = _flat_jsonl(wide_ood_path)
            newarm_path = ER / "new_arm_round/arm_results/merged_transfer.jsonl"
            newarm = [r for r in _flat_jsonl(newarm_path) if r.get("behavior") == beh]
            fill = [r for r in armfill if r.get("behavior") == beh]

            # held-out train rung: the 16-arm battery, plus arms 17/18 (and the
            # in-split n_eval, which arm_rows does not carry).
            train = _slice(main["arm_rows"], variant=variant, eval_rung="train", budget_l=lmax)
            n_eval_train = {
                r["n_eval"]
                for r in _slice(
                    wide_ood,
                    variant=variant,
                    eval_rung="train",
                    budget_l=lmax,
                    rung_kind="train_in_split",
                )
            }
            for r in train:
                r.setdefault("n_eval", next(iter(n_eval_train)) if len(n_eval_train) == 1 else None)
            agg = _aggregate(train, _rel(main_path))
            extra = _aggregate(
                _slice(
                    newarm,
                    variant=variant,
                    eval_rung="train",
                    budget_l=lmax,
                    arms=NEW_ARM_ROUND_ARMS,
                ),
                _rel(newarm_path),
            )
            # arm5 lives in both sources at the train rung; the 16-arm battery is
            # canonical there, so only genuinely new arms are added.
            agg.update({a: v for a, v in extra.items() if a not in agg})
            agg.update(
                _aggregate(
                    _slice(fill, variant=variant, eval_rung="train", budget_l=lmax),
                    "armfill_round(s)",
                )
            )
            if agg:
                table[(state_key, beh, "train")] = agg

            # behavior-specific OOD rungs
            for rung in OOD_RUNGS[beh]:
                agg = _aggregate(
                    _slice(
                        wide_ood,
                        variant=variant,
                        eval_rung=rung,
                        budget_l=lmax,
                        rung_kind="eval_transfer",
                    ),
                    _rel(wide_ood_path),
                )
                agg.update(
                    {
                        a: v
                        for a, v in _aggregate(
                            _slice(
                                newarm,
                                variant=variant,
                                eval_rung=rung,
                                budget_l=lmax,
                                arms=NEW_ARM_ROUND_ARMS,
                            ),
                            _rel(newarm_path),
                        ).items()
                        if a not in agg
                    }
                )
                agg.update(
                    _aggregate(
                        _slice(fill, variant=variant, eval_rung=rung, budget_l=lmax),
                        "armfill_round(s)",
                    )
                )
                if agg:
                    table[(state_key, beh, rung)] = agg

            # random WildChat + persona-vectors synthetic: one budget each
            for src, rung in (("wide/wildchat_rung", "wildchat_rung"), ("wide/pvsynth", "pvsynth")):
                path = ER / src / beh / "all_arms_spearman.json"
                with open(path) as f:
                    rows = json.load(f)["transfer_rows"]
                agg = _aggregate(_slice(rows, variant=variant, eval_rung=rung), _rel(path))
                # arm-fill rounds add arms 2 / 9 / 14 (armfill_round) and
                # 10 / 17 / 18 (armfill_round3) on these two legs; the arm sets
                # are disjoint from each other and from the 10 arms above.
                agg.update(
                    _aggregate(_slice(fill, variant=variant, eval_rung=rung), "armfill_round(s)")
                )
                if agg:
                    table[(state_key, beh, rung)] = agg

    # matched-target guard: one judged eval set per panel column
    for key, agg in table.items():
        n_evals = {v["n_eval"] for v in agg.values() if v["n_eval"] is not None}
        if len(n_evals) > 1:
            raise ValueError(
                f"{key}: methods scored against different eval sizes {sorted(n_evals)}"
            )

    spread = {}
    with open(ER / "result1_spread/spread_stats.json") as f:
        for c in json.load(f)["cells"]:
            spread[(c["behavior"], c["setting"])] = dict(
                ceiling=float(c["ceiling_sqrt_r_yy"]),
                gate=c["criterion_verdict"],
                dv_construct=c["dv_construct"],
                n_contexts=int(c["n_contexts"]),
            )
    return table, spread, notes


# --- rendering ----------------------------------------------------------------


def _x_positions(settings: list[str]) -> list[float]:
    """Categorical x, with the diagnostic column pushed right by a visible gap."""
    xs, cursor = [], 0.0
    for s in settings:
        if s == DIAGNOSTIC_SETTING:
            cursor += 0.75
        xs.append(cursor)
        cursor += 1.0
    return xs


def _legend_handles(state_token: str):
    locus = [
        Line2D(
            [],
            [],
            color=LOCUS_COLOR[k],
            linestyle=LOCUS_STYLE[k],
            marker="",
            linewidth=2.2,
            label=LOCUS_LABEL[k],
        )
        for k in LOCUS_ORDER
    ]
    readout = [
        Line2D(
            [],
            [],
            color="#333333",
            linestyle="",
            marker=READOUT_MARKER[k],
            markersize=7,
            markeredgewidth=1.2,
            label=READOUT_LEGEND_LABEL[k],
        )
        for k in READOUT_LEGEND_ORDER
    ]
    controls = [
        Line2D(
            [],
            [],
            color=LOCUS_COLOR["control"],
            linestyle=":",
            marker=READOUT_MARKER[READOUT[a]],
            markersize=6,
            markeredgewidth=1.2,
            label=METHOD_NAME[a].format(state=state_token),
        )
        for a in CONTROL_LEGEND_ORDER
    ]
    reading = [
        Line2D(
            [],
            [],
            color="#7A7A7A",
            linestyle="-",
            linewidth=3.0,
            alpha=0.55,
            label="reliability ceiling (square root of r_yy)",
        ),
        Patch(
            facecolor="#F2D7A0",
            alpha=0.55,
            edgecolor="none",
            label="spread gate failed (Result 1): ranking uninformative",
        ),
        Patch(
            facecolor="#E4E4E4",
            alpha=0.85,
            edgecolor="none",
            label="diagnostic column, not a deployment setting",
        ),
        Line2D(
            [],
            [],
            color="#444444",
            linestyle=(0, (4, 3)),
            linewidth=1.4,
            label="DV construct boundary (hallucination only)",
        ),
        Line2D(
            [],
            [],
            color="#333333",
            linestyle="",
            marker="x",
            markersize=6,
            markeredgewidth=1.1,
            label="method not run beyond this setting",
        ),
    ]
    return locus, readout, controls, reading


def render(state_key, state_token, title, table, spread, ylim, out_stem) -> list[dict]:
    """Draw one input-state figure; return the records it plotted."""
    set_paper_style("blog", font_scale=0.82)
    # The blog style turns constrained_layout on globally, so explicit gridspec
    # geometry is ignored; drive the spacing through the layout engine instead.
    fig = plt.figure(figsize=(17.0, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.34])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    plotted: list[dict] = []

    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        settings = SETTINGS[beh]
        xs = _x_positions(settings)
        xpos = dict(zip(settings, xs, strict=True))

        # background bands: diagnostic column and any setting failing the gate
        for s, x in xpos.items():
            if s == DIAGNOSTIC_SETTING:
                ax.axvspan(x - 0.45, x + 0.45, color="#E4E4E4", alpha=0.85, zorder=0, lw=0)
            if spread.get((beh, s), {}).get("gate") == "FAIL":
                ax.axvspan(x - 0.5, x + 0.5, color="#F2D7A0", alpha=0.55, zorder=0, lw=0)

        # DV-construct boundary (hallucination: fabrication rate | trait rubric)
        fab = [s for s in settings if (beh, s) in FABRICATION_RATE_SETTINGS]
        if fab and len(fab) < len(settings):
            edge = settings.index(fab[-1])
            mid = (xs[edge] + xs[edge + 1]) / 2.0
            ax.axvline(mid, color="#444444", linestyle=(0, (4, 3)), linewidth=1.4, zorder=1)

        # reliability ceiling per setting
        for s, x in xpos.items():
            cell = spread.get((beh, s))
            if cell is None:
                continue
            ax.plot(
                [x - 0.34, x + 0.34],
                [cell["ceiling"]] * 2,
                color="#7A7A7A",
                linewidth=3.0,
                alpha=0.55,
                solid_capstyle="butt",
                zorder=2,
            )

        ax.axhline(0.0, color="#B0B0B0", linewidth=0.9, zorder=1)

        for arm in METHOD_ORDER:
            locus = LOCUS[arm]
            is_control = locus == "control"
            is_oracle = locus == "oracle"
            recs = [(s, table.get((state_key, beh, s), {}).get(arm)) for s in settings]
            if all(r is None for _, r in recs):
                continue
            ys = [r["rho"] if r else np.nan for _, r in recs]
            lo = [r["ci"][0] if r and r["ci"] else np.nan for _, r in recs]
            hi = [r["ci"][1] if r and r["ci"] else np.nan for _, r in recs]
            color = LOCUS_COLOR[locus]
            ax.errorbar(
                xs,
                ys,
                yerr=np.vstack(
                    [
                        np.maximum(0.0, np.asarray(ys) - np.asarray(lo)),
                        np.maximum(0.0, np.asarray(hi) - np.asarray(ys)),
                    ]
                ),
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                linestyle=LOCUS_STYLE[locus],
                linewidth=1.0 if is_control else 1.8,
                alpha=0.55 if is_control else 0.95,
                marker=READOUT_MARKER[READOUT[arm]],
                markersize=5.0 if is_control else 6.6,
                markerfacecolor="none" if is_oracle else color,
                markeredgecolor=color,
                markeredgewidth=1.3,
                zorder=3 if is_control else 4,
            )
            # terminator: the method stops before the rightmost setting
            have = [i for i, (_, r) in enumerate(recs) if r is not None]
            if have and have[-1] < len(settings) - 1:
                ax.plot(
                    [xs[have[-1]] + 0.30],
                    [ys[have[-1]]],
                    marker="x",
                    color=color,
                    markersize=4.5,
                    markeredgewidth=1.0,
                    alpha=0.8,
                    linestyle="",
                    zorder=5,
                )
            for s, r in recs:
                if r is None:
                    continue
                plotted.append(
                    dict(
                        input_state=state_key,
                        input_state_label=title,
                        behavior=beh,
                        setting=s,
                        setting_label=SETTING_LABEL[(beh, s)].replace("\n", " "),
                        method=method_name(arm, state_token),
                        arm_id=arm,
                        locus=locus,
                        readout=READOUT[arm],
                        variant="bare_context_end" if state_key == "bare_query" else state_key,
                        regime="e1",
                        u_rung_label="full",
                        budget_l=r["budget_l"],
                        rho=round(r["rho"], 6),
                        ci=[round(r["ci"][0], 6), round(r["ci"][1], 6)] if r["ci"] else None,
                        rho_sd_across_replicates=round(r["rho_sd"], 6),
                        n_replicates=r["n_replicates"],
                        rho_replicates=r["rho_replicates"],
                        n_eval=r["n_eval"],
                        layer=r["layer"],
                        dv_construct=spread.get((beh, s), {}).get("dv_construct"),
                        reliability_ceiling=spread.get((beh, s), {}).get("ceiling"),
                        spread_gate=spread.get((beh, s), {}).get("gate"),
                        source_file=r["source_file"],
                    )
                )

        ax.set_xticks(xs)
        ax.set_xticklabels([SETTING_LABEL[(beh, s)] for s in settings], fontsize=7.4)
        ax.set_xlim(min(xs) - 0.62, max(xs) + 0.62)
        ax.set_ylim(*ylim)
        ax.set_title(BEHAVIOR_LABEL[beh], loc="left")

    axes[0].set_ylabel("Spearman rho, prediction vs judged behavior expression")
    axes[1].set_xlabel("evaluation setting, ordered by distance from the training distribution")
    fig.suptitle(f"Input state: {title}", x=0.006, ha="left")

    locus_h, readout_h, control_h, reading_h = _legend_handles(state_token)
    leg_kw = dict(frameon=False, loc="upper left", alignment="left", fontsize=7.6, borderpad=0.0)
    for handles, ltitle, x0 in (
        (locus_h, "colour: where the readout reads", 0.000),
        (readout_h, "marker: readout type", 0.275),
        (control_h, "controls (dotted, grey)", 0.505),
        (reading_h, "reading the panel", 0.735),
    ):
        leg = legend_ax.legend(
            handles=handles,
            title=ltitle,
            ncol=1,
            bbox_to_anchor=(x0, 1.0),
            bbox_transform=legend_ax.transAxes,
            **leg_kw,
        )
        leg.get_title().set_fontsize(8.0)
        leg.get_title().set_fontweight("semibold")
        legend_ax.add_artist(leg)

    savefig_paper(fig, out_stem, dir=OUT_FIG)
    plt.close(fig)
    return plotted


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NUM.mkdir(parents=True, exist_ok=True)
    table, spread, notes = load_points()

    # one y-range shared by all three figures so they superimpose by eye
    vals = []
    for agg in table.values():
        for rec in agg.values():
            vals.append(rec["rho"])
            if rec["ci"]:
                vals.extend(rec["ci"])
    vals.extend(c["ceiling"] for c in spread.values())
    ylim = (min(vals) - 0.06, max(vals) + 0.06)

    records: list[dict] = []
    for state_key, _variant, state_token, title in INPUT_STATES:
        records += render(
            state_key,
            state_token,
            title,
            table,
            spread,
            ylim,
            f"result2_methods_{state_key}",
        )

    with open(OUT_NUM / "result2_points.json", "w") as f:
        json.dump(
            dict(
                caption=(
                    "Every point plotted in the Result 2 method-comparison figures. One record "
                    "per (input state, behavior, evaluation setting, method). rho is the mean of "
                    "the committed per-replicate Spearman rho_frozen at the max-data operating "
                    "slice (regime e1, full unlabeled pool, the behavior's maximum labelled "
                    "budget); ci is the committed bootstrap ci_frozen averaged over the same "
                    "replicates. Hallucination's own rungs score a fabricated fraction rescaled "
                    "x100; every other cell scores the graded 0-100 trait rubric."
                ),
                operating_slice=dict(
                    regime="e1",
                    u_rung_label="full",
                    budget_l_by_behavior=dict(LMAX),
                    note=(
                        "The WildChat, persona-vectors-synthetic and bare-query legs carry a "
                        "single budget each, already that leg's full-data cut."
                    ),
                ),
                setting_order_note=(
                    "held-out train -> behavior-specific OOD -> random WildChat -> "
                    "persona-vectors synthetic (diagnostic). Ordered this way so "
                    "hallucination's two DV constructs stay contiguous around one boundary."
                ),
                notes=notes,
                ylim=list(ylim),
                n_points=len(records),
                points=records,
            ),
            f,
            indent=1,
        )

    # coverage + headline read-out
    print("\n".join(notes))
    print(f"\nwrote {len(records)} points -> {_rel(OUT_NUM / 'result2_points.json')}")
    for state_key, _v, state_token, title in INPUT_STATES:
        print(f"\n=== input state: {title}")
        for beh in BEHAVIORS:
            for s in SETTINGS[beh]:
                agg = table.get((state_key, beh, s))
                if not agg:
                    print(f"  {beh:<14} {s:<14} -- no methods at the operating slice")
                    continue
                miss = [a for a in METHOD_ORDER if a not in agg]
                dep = [(a, agg[a]["rho"]) for a in DEPLOYABLE if a in agg]
                best = max(dep, key=lambda t: t[1]) if dep else None
                cell = spread.get((beh, s), {})
                line = (
                    f"  {beh:<14} {s:<14} n_methods={len(agg):>2} n_eval={agg[next(iter(agg))]['n_eval']}"
                    f" gate={cell.get('gate')}"
                )
                if best:
                    rec = agg[best[0]]
                    ci = rec["ci"]
                    line += (
                        f" | best deployable: {method_name(best[0], state_token)}"
                        f" rho={best[1]:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
                    )
                print(line)
                if miss:
                    print(f"      not run: {', '.join(miss)}")


if __name__ == "__main__":
    main()
