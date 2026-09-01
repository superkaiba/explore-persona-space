#!/usr/bin/env python3
"""Analyze the frozen ten-target attrition sensitivity for issue #2254.

The planned eleven-target primary is not estimable because the unsteered
``query_topic`` floor anchor failed its frozen screen quality gate.  This
script implements only the judge-blind analysis frozen in
``target_attrition_decision_v1.json``; it never redefines the primary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue2254_multitype_context_preference as exp  # noqa: E402

OUT_ROOT = REPO_ROOT / "eval_results/issue_2254/multitype_context_preference_qwen35"
RESULT_FIG = REPO_ROOT / "artifacts/issue2254/multitype_context_preference_qwen35.png"
RESULT_REPORT = REPO_ROOT / "artifacts/issue2254/multitype_context_preference_qwen35.md"
DECISION_FILE = "target_attrition_decision_v1.json"
ANALYSIS_AMENDMENT_FILE = "target_attrition_analysis_amendment_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded)


def _validate_decision_fields(decision: dict[str, Any]) -> None:
    retained = tuple(decision["decision"]["retained_targets"])
    expected = tuple(target for target in exp.TARGETS if target != "query_topic")
    if retained != expected:
        raise RuntimeError(f"retained-target drift: {retained} != {expected}")
    if decision["decision"].get("preregistered_11_target_primary") != "not_estimable":
        raise RuntimeError("attrition decision must keep the planned primary not estimable")
    if decision["decision"].get("alternate_cjk_or_query_confirmation_in_this_run") is not False:
        raise RuntimeError("this analysis forbids alternate-CJK/query confirmation")


def _verify_decision_evidence(root: Path, decision: dict[str, Any]) -> None:
    raw = root / "screen/raw_completions/query_topic__anchor_a.json"
    if _sha256(raw) != decision["evidence_hashes"]["query_topic_anchor_a_raw_sha256"]:
        raise RuntimeError("query-topic floor raw hash mismatch")
    judged_dir = root / "screen/judge/judged"
    for cell_id, expected_hash in decision["evidence_hashes"]["anchor_judged_records"].items():
        if _sha256(judged_dir / f"{cell_id}.json") != expected_hash:
            raise RuntimeError(f"anchor evidence hash mismatch: {cell_id}")


def _verify_review_amendment(root: Path, decision_path: Path, decision: dict[str, Any]) -> dict[str, Any]:
    path = root / ANALYSIS_AMENDMENT_FILE
    if not path.exists():
        raise RuntimeError(f"missing frozen analysis-review amendment: {path}")
    amendment = json.loads(path.read_text())
    if _sha256(decision_path) != amendment["frozen_inputs"]["target_attrition_decision_sha256"]:
        raise RuntimeError("target-attrition decision hash mismatch")
    frozen_impl = decision["frozen_analysis_implementation"]
    superseded = amendment["superseded_implementation"]
    for key in ("script", "test"):
        if superseded[f"{key}_sha256"] != frozen_impl[f"{key}_sha256"]:
            raise RuntimeError(f"superseded attrition {key} hash drift")
        live_path = REPO_ROOT / amendment["reviewed_implementation"][key]
        if _sha256(live_path) != amendment["reviewed_implementation"][f"{key}_sha256"]:
            raise RuntimeError(f"reviewed attrition {key} hash mismatch")
    design_path = root / "preregistered_design.json"
    design_amendment_path = root / "preregistered_design_amendment_v1.json"
    design_amendment = json.loads(design_amendment_path.read_text())
    frozen = design_amendment["frozen_artifacts"]
    if amendment["frozen_inputs"]["preregistered_design_sha256"] != frozen[
        "preregistered_design_sha256"
    ]:
        raise RuntimeError("review amendment design hash drift")
    if amendment["frozen_inputs"]["amended_execution_script_sha256"] != frozen[
        "amended_execution_script_sha256"
    ]:
        raise RuntimeError("review amendment dependency hash drift")
    if _sha256(design_path) != frozen["preregistered_design_sha256"]:
        raise RuntimeError("preregistered design hash mismatch")
    dependency = REPO_ROOT / "scripts/issue2254_multitype_context_preference.py"
    if _sha256(dependency) != frozen["amended_execution_script_sha256"]:
        raise RuntimeError("amended execution dependency hash mismatch")
    if _sha256(design_amendment_path) != amendment["frozen_inputs"]["design_amendment_sha256"]:
        raise RuntimeError("design amendment hash mismatch")
    return amendment


def verify_attrition_decision(root: Path) -> dict[str, Any]:
    decision_path = root / DECISION_FILE
    if not decision_path.exists():
        raise RuntimeError(f"missing frozen attrition decision: {decision_path}")
    decision = json.loads(decision_path.read_text())
    _validate_decision_fields(decision)
    _verify_decision_evidence(root, decision)
    _verify_review_amendment(root, decision_path, decision)
    exp._verify_pre_amendment_archive(root)
    return decision


def _collect_target_rows(confirm: dict[str, Any], targets: tuple[str, ...]):
    target_rows: dict[str, Any] = {}
    preference_values: dict[str, float] = {}
    context_values: dict[str, float] = {}
    preference_q: dict[str, np.ndarray] = {}
    context_q: dict[str, np.ndarray] = {}
    score_preference_values: dict[str, float] = {}
    score_preference_q: dict[str, np.ndarray] = {}
    for target in targets:
        row = confirm["targets"][target]
        ctx = row["positions"]["context"]
        ans = row["positions"]["answer"]
        if ctx.get("status") != "ok" or ans.get("status") != "ok":
            raise RuntimeError(
                f"analysis requires complete target {target}: "
                f"{ctx.get('status')}/{ans.get('status')}"
            )
        ctx_q = np.asarray(
            [np.nan if value is None else value for value in ctx["per_question_f"]],
            dtype=float,
        )
        ans_q = np.asarray(
            [np.nan if value is None else value for value in ans["per_question_f"]],
            dtype=float,
        )
        valid = np.isfinite(ctx_q) & np.isfinite(ans_q)
        if int(valid.sum()) < exp.MIN_VALID_QUESTIONS:
            raise RuntimeError(f"{target}: too few common held-out pairs")
        ctx_valid = ctx_q[valid]
        ans_valid = ans_q[valid]
        diff = ctx_valid - ans_valid
        separation = np.asarray(row["per_question_anchor_separation"], dtype=float)
        if separation.shape != ctx_q.shape:
            raise RuntimeError(f"{target}: anchor-separation shape mismatch")
        score_diff = diff * separation[valid]
        preference_values[target] = float(np.mean(diff))
        context_values[target] = float(np.mean(ctx_valid))
        preference_q[target] = diff
        context_q[target] = ctx_valid
        score_preference_values[target] = float(np.mean(score_diff))
        score_preference_q[target] = score_diff
        target_rows[target] = {
            "target_class": row["target_class"],
            "information_type": row["information_type"],
            "floor_mean": float(row["floor_mean"]),
            "ceiling_mean": float(row["ceiling_mean"]),
            "anchor_separation_mean": float(np.mean(separation)),
            "common_question_anchor_separation_mean": float(np.mean(separation[valid])),
            "context_f": float(np.mean(ctx_valid)),
            "answer_f": float(np.mean(ans_valid)),
            "context_minus_answer_f": float(np.mean(diff)),
            "context_minus_answer_score_points": float(np.mean(score_diff)),
            "context_exceeds_all_random_points": ctx["random"]["exceeds_all_points"],
            "answer_exceeds_all_random_points": ans["random"]["exceeds_all_points"],
            "n_common_questions": int(valid.sum()),
        }
    return (
        target_rows,
        preference_values,
        context_values,
        preference_q,
        context_q,
        score_preference_values,
        score_preference_q,
    )


def _test_with_interval(values: dict[str, float], per_question: dict[str, np.ndarray]):
    result = exp.exact_label_permutation(values, exp.PERSONAS)
    boot = exp._nested_bootstrap(per_question, exp.PERSONAS)
    result["bootstrap_ci95"] = [
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]
    return result


def compute_result(
    confirm: dict[str, Any],
    decision: dict[str, Any],
    *,
    confirm_sha256: str,
    decision_sha256: str,
    design_sha256: str,
    amendment_sha256: str,
    gate_audit: dict[str, Any],
) -> dict[str, Any]:
    targets = tuple(decision["decision"]["retained_targets"])
    if set(confirm.get("targets", {})) != set(targets):
        raise RuntimeError("confirmation summary does not exactly match the retained target set")
    rows, pref, context, pref_q, context_q, score_pref, score_pref_q = _collect_target_rows(
        confirm, targets
    )
    retained_test = _test_with_interval(pref, pref_q)
    retained_context = _test_with_interval(context, context_q)
    score_companion = _test_with_interval(score_pref, score_pref_q)
    leave_icl = tuple(target for target in targets if target != "icl_task")
    leave_icl_test = _test_with_interval(
        {target: pref[target] for target in leave_icl},
        {target: pref_q[target] for target in leave_icl},
    )
    return {
        "experiment": "issue2254_multitype_context_preference_qwen35",
        "inference_status": "sensitivity_only_no_confirmatory_claim",
        "preregistered_11_target_primary": {
            "status": "not_estimable",
            "reason": "query_topic unsteered floor anchor failed the frozen CJK gate",
        },
        "target_attrition_sensitivity": retained_test,
        "target_attrition_absolute_context_companion": retained_context,
        "target_attrition_unnormalized_score_companion": score_companion,
        "retained_leave_icl_out_sensitivity": leave_icl_test,
        "retained_targets": list(targets),
        "omitted_targets": ["query_topic"],
        "confirm_summary_sha256": confirm_sha256,
        "target_attrition_decision_sha256": decision_sha256,
        "design_sha256": design_sha256,
        "amendment_sha256": amendment_sha256,
        "quality_gate_audit": gate_audit,
        "gate_resolution_audit": decision["gate_resolution_audit"],
        "targets": rows,
        "group_means": {
            group: {
                metric: float(np.mean([rows[target][metric] for target in members]))
                for metric in (
                    "context_f",
                    "answer_f",
                    "context_minus_answer_f",
                    "context_minus_answer_score_points",
                    "anchor_separation_mean",
                    "common_question_anchor_separation_mean",
                )
            }
            for group, members in {
                "persona": tuple(target for target in targets if target in exp.PERSONAS),
                "retained_nonpersona": tuple(
                    target for target in targets if target not in exp.PERSONAS
                ),
            }.items()
        },
        "literal_frozen_gate_primary": _literal_frozen_gate_status(gate_audit),
        "scope": (
            "No result is confirmatory because the planned 11-target primary is not "
            "estimable. The retained comparison shifts composition from 4/7 to 4/6 and "
            "drops a nonpersona query-topic target. Prior-topic and response-theme targets "
            "retain topic coverage. Answer steering edits prefill plus every cached decode "
            "state, while context steering edits one state, so the modes are not "
            "equal-total-energy interventions. ICL uses the amended nonempty/non-refusal "
            "inclusion predicate because its intended response is a short label, while the "
            "other targets retain the frozen prose-coherence gate; inclusion rates are not "
            "comparable across those gate families."
        ),
    }


def _literal_frozen_gate_status(gate_audit: dict[str, Any]) -> dict[str, str]:
    frozen_eligible = gate_audit["literal_frozen_gate"]["n_eligible_cells"]
    if frozen_eligible == 0:
        return {
            "status": "not_estimable",
            "reason": (
                "The frozen prose-coherence gate yielded zero eligible ICL screen cells; "
                "see the immutable pre-amendment archive."
            ),
        }
    return {
        "status": "not_reconstructed",
        "reason": (
            f"The archive contains {frozen_eligible} eligible ICL cells; a literal-frozen "
            "estimate would require a separately frozen selection rule."
        ),
    }


def _plot_result(result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(result["retained_targets"])
    rows = result["targets"]
    y = np.arange(len(names))
    ctx = np.asarray([rows[name]["context_f"] for name in names])
    ans = np.asarray([rows[name]["answer_f"] for name in names])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.5), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for yi, name, context, answer in zip(y, names, ctx, ans, strict=True):
        color = "#0072B2" if name in exp.PERSONAS else "#666666"
        ax.plot([context, answer], [yi, yi], color=color, alpha=0.55, linewidth=1.2)
        ax.scatter(context, yi, color="#009E73", marker="o", s=38, zorder=3)
        ax.scatter(answer, yi, color="#D55E00", marker="s", s=38, zorder=3)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y, [name.replace("_", " ") for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out fraction of natural A→B swap (F)")
    ax.set_title("Same context-native direction, two intervention modes", loc="left")
    ax.scatter([], [], color="#009E73", marker="o", label="final context token")
    ax.scatter([], [], color="#D55E00", marker="s", label="all answer positions")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    preference = ctx - ans
    colors = ["#0072B2" if name in exp.PERSONAS else "#999999" for name in names]
    ax.barh(y, preference, color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, ["" for _ in names])
    ax.invert_yaxis()
    ax.set_xlabel("Context preference: Fcontext − Fanswer")
    ax.set_title("Target-attrition sensitivity", loc="left")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Persona versus retained context information on Qwen3.5-9B", fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_report(result: dict[str, Any], path: Path) -> None:
    test = result["target_attrition_sensitivity"]
    leave = result["retained_leave_icl_out_sensitivity"]
    score = result["target_attrition_unnormalized_score_companion"]
    literal = result["literal_frozen_gate_primary"]
    lines = [
        "# Multi-type context-preference experiment (Qwen3.5-9B)",
        "",
        "## Inference status",
        "",
        "**No confirmatory claim is available.** The preregistered 11-target primary is not "
        "estimable because the unsteered `query_topic` floor anchor failed the frozen CJK "
        "gate (2/6 flagged completions; the >0.20 gate excludes at 2/6).",
        "",
        "## Target-attrition sensitivity",
        "",
        f"Across the retained 4 persona and 6 nonpersona targets, the persona-minus-"
        f"nonpersona context-preference interaction is **{test['observed']:+.3f}** "
        f"(95% nested-bootstrap CI **[{test['bootstrap_ci95'][0]:+.3f}, "
        f"{test['bootstrap_ci95'][1]:+.3f}]**; descriptive one-sided exact permutation "
        f"p = **{test['p_greater']:.4f}**, {test['n_assignments']} assignments).",
        "",
        f"The newly prespecified retained-set leave-ICL-out sensitivity is "
        f"**{leave['observed']:+.3f}** (descriptive one-sided exact permutation "
        f"p = **{leave['p_greater']:.4f}**, {leave['n_assignments']} assignments).",
        "",
        f"As a scale diagnostic, the unnormalized persona-minus-nonpersona interaction "
        f"in context-minus-answer judge-score points is **{score['observed']:+.3f}** "
        f"(95% nested-bootstrap CI **[{score['bootstrap_ci95'][0]:+.3f}, "
        f"{score['bootstrap_ci95'][1]:+.3f}]**; descriptive one-sided exact permutation "
        f"p = **{score['p_greater']:.4f}**). This companion is not a replacement primary.",
        "",
        "The retained analysis changes composition from 4/7 to 4/6 and removes a target "
        "from the nonpersona side only. Attrition arose in an unsteered anchor, but "
        "independence of CJK propensity from the persona/nonpersona factor is assumed, not "
        "demonstrated. Topic coverage remains through prior-topic and response-theme targets.",
        "",
        "The literal frozen prose-coherence-gate 11-target analysis is "
        f"**{literal['status'].replace('_', ' ')}**: {literal['reason']} The retained "
        "`icl_task` target instead uses the amended nonempty/non-refusal gate appropriate "
        "for one-word labels, so its inclusion rate is not comparable to the prose targets. "
        "The leave-ICL-out sensitivity is reported for this reason.",
        "",
        "## Target results",
        "",
        "| Target | Type | Anchor gap | Context F | Answer F | Context − answer | Score-point difference | Context > random | Answer > random |",
        "|---|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for target in result["retained_targets"]:
        row = result["targets"][target]
        lines.append(
            f"| {target.replace('_', ' ').title()} | {row['information_type']} | "
            f"{row['anchor_separation_mean']:.1f} | "
            f"{row['context_f']:+.3f} | {row['answer_f']:+.3f} | "
            f"{row['context_minus_answer_f']:+.3f} | "
            f"{row['context_minus_answer_score_points']:+.1f} | "
            f"{'yes' if row['context_exceeds_all_random_points'] else 'no'} | "
            f"{'yes' if row['answer_exceeds_all_random_points'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Instrument limitation",
            "",
            "The frozen CJK gate operates on six binary completion flags: 0/6 and 1/6 pass, "
            "while 2/6 fails. Across 352 screen cells, 251 had 0/6 flags, 61 had 1/6, and "
            "40 had 2/6 or more. The failed query anchor contained isolated CJK characters "
            "inside long English answers, illustrating the gate's low resolution. No anchor "
            "was rerun or substituted, and no alternate-CJK analysis was run.",
            "",
            "## Scope",
            "",
            result["scope"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--result-fig", type=Path, default=RESULT_FIG)
    parser.add_argument("--result-report", type=Path, default=RESULT_REPORT)
    args = parser.parse_args()
    root = args.out_root.resolve()
    decision = verify_attrition_decision(root)
    confirm_path = root / "confirm/summary.json"
    if not confirm_path.exists():
        raise RuntimeError(f"missing confirmation summary: {confirm_path}")
    confirm = json.loads(confirm_path.read_text())
    result = compute_result(
        confirm,
        decision,
        confirm_sha256=_sha256(confirm_path),
        decision_sha256=_sha256(root / DECISION_FILE),
        design_sha256=_sha256(root / "preregistered_design.json"),
        amendment_sha256=_sha256(root / "preregistered_design_amendment_v1.json"),
        gate_audit=exp._icl_gate_audit(root),
    )
    _write_json(root / "multitype_context_preference_summary.json", result)
    _plot_result(result, args.result_fig.resolve())
    _write_report(result, args.result_report.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
