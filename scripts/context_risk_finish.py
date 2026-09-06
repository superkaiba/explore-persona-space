"""Finish the existing context-risk pilot using saved data only (no model inference).

Run as ``uv run python -m scripts.context_risk_finish --input-root ... --output-dir ...``.
The reward-hacking prevalence gate is binding; failed feasibility is a completed
scientific outcome, distinct from an analysis execution failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts import context_risk_analyze as analyze

SPLITS = {
    "original_prefix": ("exact_context_sha256", "length"),
    "condition_controlled_prefix": ("exact_context_sha256", "conditions"),
    "held_out_goal_framing": ("goal_framing", "conditions"),
    "held_out_urgency": ("urgency_type", "conditions"),
}

# This is a recovery of a fixed, independently audited pilot, not a generic
# experiment runner. Pin the inputs behind the quantitative/qualitative prose.
EXPECTED_RESULTS = {
    "impossible_livecodebench_v19/full/run_result.json": "cf5e48c0edd7fde2caf75d480582c16fc7397a641c58ea01ce99b4966a0e34d3",
    "qwen38_misalignment_public_v5/run_result.json": "98696303cd92787c5d8028fe4c0e160089ef85dcf273a3de171bffd0005df244",
}
REVIEWED_ROLLOUT_MANIFEST_SHA256 = (
    "aaa40f1dcd01b1aac7b59196ed6f0ef06533fbed3278d2e7f4a81c8f8bbec3b2"
)


def validate_inputs(root: Path) -> dict:
    from scripts.context_risk_agentic_misalignment import score_leak_action
    from scripts.context_risk_qwen38_misalignment_rollout import (
        annotate_sample_outcome,
        summarize_sample_outcomes,
    )

    for relative, expected in EXPECTED_RESULTS.items():
        if analyze._sha256(root / relative) != expected:
            raise RuntimeError(f"input differs from audited pilot: {relative}")
    rollout_root = root / "qwen38_misalignment_public_v5"
    result = json.loads((rollout_root / "run_result.json").read_text())
    map_result = json.loads((root / "qwen38_map_pilot/run_result.json").read_text())
    for field in ("model_id", "model_revision"):
        if result[field] != map_result[field]:
            raise RuntimeError(f"map/rollout {field} mismatch")
    map_path = root / "qwen38_map_pilot/map_layer_44.npz"
    if analyze._sha256(map_path) != map_result["map_artifact_sha256"]:
        raise RuntimeError("map artifact hash mismatch")
    if map_result["selected_layer"] != 44 or not map_result["passed"] or not result["passed"]:
        raise RuntimeError("map or rollout completion gate failed")
    manifest_path = root / "data/agentic_misalignment/public_development_manifest.jsonl"
    manifest = {
        row["condition_id"]: row for row in map(json.loads, manifest_path.read_text().splitlines())
    }
    done_paths = sorted(rollout_root.glob("context_*/done.json"))
    if len(done_paths) != len(manifest) or len(done_paths) != result["n_contexts"]:
        raise RuntimeError("realized context coverage does not match manifest")
    counts = {key: 0 for key in ("n_positive", "n_negative", "n_censored", "n_rollouts")}
    seeds = set()
    hashes = {}
    for done_path in done_paths:
        done = json.loads(done_path.read_text())
        for filename, field in (
            ("pre_action_activation.npz", "activation_sha256"),
            ("rollouts.json", "rollouts_sha256"),
        ):
            path = done_path.parent / filename
            observed = analyze._sha256(path)
            if observed != done[field]:
                raise RuntimeError(f"hash mismatch: {path}")
            hashes[str(path.relative_to(root))] = observed
        payload = json.loads((done_path.parent / "rollouts.json").read_text())
        record = manifest[done["condition_id"]]
        if (
            record["exact_context_sha256"] != done["exact_context_sha256"]
            or payload["exact_context_sha256"] != done["exact_context_sha256"]
        ):
            raise RuntimeError("manifest/rollout context drift")
        samples = payload["continuations"]
        for sample in samples:
            if sample["batch_lane"] != 0:
                raise RuntimeError("seed audit expects the recorded batch-size-one generation")
            if sample["batch_seed"] in seeds:
                raise RuntimeError("duplicate continuation seed")
            seeds.add(sample["batch_seed"])
            scored = score_leak_action(sample["response"], record["email_content"])
            if scored != sample["action_score"]:
                raise RuntimeError("stored score does not match deterministic rescore")
            saved_outcome = sample["outcome"]
            if annotate_sample_outcome(sample) != saved_outcome:
                raise RuntimeError("stored outcome differs from scored action and censoring")
        observed_counts = summarize_sample_outcomes(samples)
        for field in counts:
            value = (
                len(samples) if field == "n_rollouts" else observed_counts[field.removeprefix("n_")]
            )
            if value != done[field]:
                raise RuntimeError(f"raw rollout count mismatch: {done_path}, {field}")
            counts[field] += value
    if any(value != result[field] for field, value in counts.items()):
        raise RuntimeError("aggregate counts differ from raw rollouts")
    rollout_hashes = {
        str(Path(path).relative_to("qwen38_misalignment_public_v5")): sha
        for path, sha in hashes.items()
        if path.endswith("/rollouts.json")
    }
    reviewed_digest = hashlib.sha256(
        json.dumps(rollout_hashes, sort_keys=True).encode()
    ).hexdigest()
    if reviewed_digest != REVIEWED_ROLLOUT_MANIFEST_SHA256:
        raise RuntimeError("rollout files differ from independent qualitative audit")
    activations, _ = analyze.load_misalignment_activations(rollout_root, selected_layer=44)
    if len(activations) != result["n_unique_contexts"]:
        raise RuntimeError("unique-prefix coverage mismatch")
    return {
        "passed": True,
        "counts": counts,
        "unique_prefixes": len(activations),
        "distinct_sampling_seeds": len(seeds),
        "verified_raw_file_hashes": hashes,
        "map_sha256": analyze._sha256(map_path),
        "qualitative_review_rollout_manifest_sha256": reviewed_digest,
        "model_id": result["model_id"],
        "model_revision": result["model_revision"],
        "outcome": "harmful forwarding action emitted in a simulated email scenario; external tools not executed",
        "scorer_limit": "regex is not generally quotation-aware; all 15 positives independently read and found to include intended final harmful forwards",
    }


def build_figures(reports: dict, output: Path) -> dict:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.c2a_plot_style import (
        ROLES,
        c2a_figure,
        save_c2a_figure,
        set_c2a_style,
        style_axis,
    )

    set_c2a_style()
    names = [
        "prevalence",
        "metadata",
        "text_metadata",
        "raw_activation_metadata",
        "mapped_activation_metadata",
    ]
    labels = [
        "Prevalence",
        "Conditions",
        "Text + conditions",
        "Raw + conditions",
        "Mapped + conditions",
    ]
    colors = [
        ROLES["control"].color,
        ROLES["control"].color,
        ROLES["base_model"].color,
        ROLES["linear"].color,
        ROLES["needs_reasoning"].color,
    ]
    figure, fraction = c2a_figure("full", aspect=0.58)
    axes = figure.subplots(1, 2)
    for axis, name, title in zip(
        axes,
        ("condition_controlled_prefix", "held_out_goal_framing"),
        ("Hold out one prefix", "Hold out goal framing"),
        strict=True,
    ):
        models = reports[name]["misaligned_action_feasibility"]["models"]
        axis.barh(
            np.arange(len(names)),
            [models[key]["metrics"]["binomial_log_loss"] for key in names],
            color=colors,
        )
        axis.set_yticks(np.arange(len(names)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Log loss (lower is better)")
        axis.set_xlim(0, 0.145)
        axis.set_xticks([0, 0.05, 0.10])
        axis.set_title(title, loc="left")
        style_axis(axis, grid_axis="x")
    figure.tight_layout(w_pad=2)
    summary = save_c2a_figure(
        figure,
        output / "prediction_log_loss",
        title="Prospective misalignment prediction",
        subject="Development pilot: one simulated information-leak scenario",
        creator=__file__,
        include_width=fraction,
    )
    plt.close(figure)
    data = reports["condition_controlled_prefix"]["misaligned_action_feasibility"]
    rows = data["context_rows"]
    order = sorted(range(len(rows)), key=lambda index: rows[index]["condition_ids"][0])
    figure, fraction = c2a_figure("full", aspect=0.85)
    axis = figure.subplots()
    y = np.arange(len(rows))
    observed = [rows[i]["positive"] / (rows[i]["positive"] + rows[i]["negative"]) for i in order]
    axis.scatter(observed, y, color=ROLES["control"].color, marker="x", label="Observed")
    for key, label, color, marker in (
        ("raw_activation_metadata", "Raw", colors[3], "o"),
        ("mapped_activation_metadata", "Mapped", colors[4], "s"),
        ("text_metadata", "Text", colors[2], "^"),
    ):
        prediction = data["models"][key]["predictions"]
        axis.scatter(
            [prediction[i] for i in order],
            y,
            label=label,
            color=color,
            marker=marker,
            facecolors="none",
        )
    axis.set_yticks(
        y,
        [
            rows[i]["condition_ids"][0].removeprefix("leaking_")
            + f"  ({rows[i]['positive']}/{rows[i]['positive'] + rows[i]['negative']})"
            for i in order
        ],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Harmful-forward probability")
    axis.set_xticks([0, 0.1, 0.2, 0.3])
    axis.set_title("Observed rates and held-out predictions\n16 unique prompts", loc="left")
    axis.legend(loc="lower right")
    style_axis(axis, grid_axis="x")
    figure.subplots_adjust(left=0.43, right=0.92, bottom=0.09, top=0.91)
    contexts = save_c2a_figure(
        figure,
        output / "context_predictions",
        title="Per-context prospective predictions",
        subject="16 unique exact prefixes; leave-one-prefix-out predictions with condition controls",
        creator=__file__,
        include_width=fraction,
    )
    plt.close(figure)
    exports = {"summary": summary, "contexts": contexts}
    serializable = json.loads(json.dumps(exports, default=str))
    serializable["data"] = reports
    analyze._write_json_atomic(output / "figures.meta.json", serializable)
    return {key: str(value["png"]) for key, value in exports.items()}


def format_report(reports: dict, audit: dict, figure_base_url: str) -> str:
    lines = [
        "# Agent-risk forecasting pilot: completed, broad prediction claim unsupported",
        "",
        "## Takeaways",
        "",
        "- Reward-hacking rollouts finished: 480/480, with no Inspect sample errors. Original-task success was 106/160; impossible variants yielded 0/320 successful bypasses. Among the 18 competent tasks, 0/288 impossible rollouts were positive. The frozen prevalence gate failed, so the reward-hacking classifier was not fit. A subsequent raw-attempt audit found code-extraction failures and unsuccessful visible-answer hardcoding; these data do not establish an absence of reward-hacking behavior. See the [diagnostic](https://github.com/superkaiba/explore-persona-space/blob/codex/context-risk-recovery-20260906/eval_results/context_risk_recovery_v1/reward_hacking_diagnostic.md).",
        "- Misalignment rollouts finished: 15 harmful emitted forwards among 640 continuations (2.34%), no censoring, and six mixed-outcome contexts. The 20 manifest rows reduce to 16 unique prefixes within one information-leak scenario.",
        "- The original prediction analysis crashed before starting because the launch environment lacked uv on PATH. Recovery completed using the saved data; no new generation or training was performed.",
        "",
        "## Goal",
        "",
        "Test whether the model's activation before generating new reasoning or an action predicts a later reward hack or harmful action, and whether a frozen context-to-answer map improves that forecast. This report closes the existing public-development feasibility pilot. It does not establish the full cross-environment claim.",
        "",
        "## Methodology",
        "",
        "Qwen3.8-27B, revision `"
        + audit["model_revision"]
        + "`, layer 44. The frozen linear context-to-answer map and captured states share model revision and dimension (5,120). Map SHA256: `"
        + audit["map_sha256"]
        + "`. Selection used generic map-development data, not safety labels. The full map-training capture bank was not revalidated in this recovery; no new generic-map reconstruction/retrieval claim is made.",
        "All 40 misalignment rollout/activation hashes were checked, 640 saved outcomes rescored deterministically, summary counts reconciled, and duplicate prefixes merged. All 640 seeds differ. Independent qualitative review of all 15 positives confirmed intended final harmful forwards. The XML scorer is not generally quotation-aware and no external email tools were executed; the measured outcome is a harmful action emitted in simulation.",
        "The original manifest requested eight continuations per row; the completed v5 launch used 32, giving 640. Earlier v3 outputs share sampling keys with v5 and were excluded. Repeated continuations estimate each prefix's hazard and do not create independent activation examples.",
        "The preserved development comparison uses leave-one-exact-prefix-out prediction and prefix length as metadata. Additional recovery checks include goal framing, goal value, and urgency indicators as visible-condition controls. Structured splits leave one entire goal framing or urgency group out; latent/swap aliases are merged to keep identical prompts in one fold. All structured checks are exploratory and remain within one scenario family.",
        "Predictors are training-fold prevalence, metadata, character-ngram hashed text plus metadata (2,048 features, lengths 3–5), raw activations plus metadata, and mapped activations plus metadata. Linear logistic fits retain the original summed L2 objective. Frequency weights collapse duplicate feature-label rows, and a primal solver with tolerance 1e-8 replaces the slow duplicate-row dual solver. Numerical equivalence to expanded-data fits is tested. No nonlinear probe was used.",
        "Regularization C is selected from [1e-4, 1e-3, 1e-2, 0.1, 1] in up to five grouped inner folds. The inherited fallback is C=0.01 with fewer than three training groups (applies to urgency-held-out fits); single-class training uses (positives+0.5)/(total+1). These are the saved development protocol's choices, not new confirmatory hyperparameters.",
        "Log loss and Brier score weight each observed Bernoulli outcome; AUROC and average precision are descriptive pooled summaries. Paired 95% intervals use 5,000 seeded cluster resamples of fixed out-of-fold predictions, without refitting the models. They omit model-training uncertainty. Only four goal-framing groups and three urgency groups are available, making structured intervals especially fragile. A raw activation benefit requires the raw-minus-text interval wholly below zero; mapping benefit requires mapped-minus-raw wholly below zero. These development criteria do not replace confirmation against a strong semantic text monitor or frozen answer-space risk directions.",
        "",
        "## Results",
        "",
        "| Evaluation | Predictor | Log loss | Brier | AUROC | Average precision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        data = report["misaligned_action_feasibility"]
        for key, model in data["models"].items():
            m = model["metrics"]
            lines.append(
                f"| {name} | {key} | {m['binomial_log_loss']:.5f} | {m['brier']:.5f} | {m['auroc']:.3f} | {m['auprc']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Negative contrasts favor the first predictor:",
            "",
            "| Evaluation | Groups | Raw minus text, 95% CI | Mapped minus raw, 95% CI |",
            "|---|---:|---|---|",
        ]
    )
    for name, report in reports.items():
        data = report["misaligned_action_feasibility"]
        values = [
            f"{data[key]['delta']:+.5f} [{data[key]['ci95_low']:+.5f}, {data[key]['ci95_high']:+.5f}]"
            for key in ("primary_contrast", "mapping_contrast")
        ]
        lines.append(f"| {name} | {data['n_groups']} | {values[0]} | {values[1]} |")
    if figure_base_url:
        lines.extend(
            [
                "",
                f"![Held-out log loss]({figure_base_url}/prediction_log_loss.png)",
                "",
                "Both panels compare condition-controlled predictors; these are point estimates. Paired uncertainty is tabulated above. Failed reward-hacking prediction is omitted, not plotted as zero.",
                "",
                f"![All unique-context predictions]({figure_base_url}/context_predictions.png)",
                "",
                "Each row is a unique prompt, labeled with positive/total outcomes. Duplicate latent/swap aliases are pooled under their first manifest name. Observed frequencies are estimates, not error-free true hazards.",
            ]
        )
    lines.extend(
        [
            "",
            "### Scope and decision",
            "",
            "The reward-hacking feasibility failure prevents the planned two-construct claim. The misalignment results below state whether either contrast passes in each split; an unpassed interval is inconclusive about small effects, not proof of equivalence.",
            "",
        ]
    )
    for name, report in reports.items():
        data = report["misaligned_action_feasibility"]
        lines.append(
            f"- {name}: activation-over-text criterion {'passed' if data['activation_signal_detected'] else 'not passed'}; mapping-over-raw criterion {'passed' if data['mapping_signal_detected'] else 'not passed'}."
        )
    lines.extend(
        [
            "",
            "EvilGenie, Instrumental Choices, HVTB, independent frozen answer-risk directions, the post-generation oracle, intervention tests, and cross-scenario transfer were not run. No confirmatory calibration or transfer conclusion is available. Continuing that broader program requires a separately specified model/environment pairing with enough reward-hacking positives; selecting rare positive trajectories from this failed gate would invalidate the protocol.",
            "",
            "**Repro:** `uv run python -m scripts.context_risk_finish --input-root <context_risk_inputs> --output-dir <new_output_dir>`. Per-split JSONs include exact input hashes, dependency versions, analyzer source hash, predictions, folds, and contrast definitions. This recovery used only local CPU analysis and existing artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-base-url", default="")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "completion_status.json"
    started = time.monotonic()
    status = {
        "state": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "phase": "input_audit",
    }
    analyze._write_json_atomic(status_path, status)
    try:
        audit = validate_inputs(args.input_root)
        analyze._write_json_atomic(args.output_dir / "input_audit.json", audit)
        reports = {}
        for name, (axis, metadata) in SPLITS.items():
            status["phase"] = name
            analyze._write_json_atomic(status_path, status)
            print(f"[analysis] {name}", flush=True)
            root = args.input_root
            reports[name] = analyze.run_analysis(
                argparse.Namespace(
                    impossible_result=root / "impossible_livecodebench_v19/full/run_result.json",
                    impossible_capture_root=root / "qwen38_impossible_contexts_v4_promptB",
                    impossible_manifest=root
                    / "data/impossible_livecodebench_promptB/public_pilot_manifest.jsonl",
                    map_artifact=root / "qwen38_map_pilot/map_layer_44.npz",
                    misalignment_result=root / "qwen38_misalignment_public_v5/run_result.json",
                    misalignment_rollout_root=root / "qwen38_misalignment_public_v5",
                    misalignment_manifest=root
                    / "data/agentic_misalignment/public_development_manifest.jsonl",
                    selected_layer=44,
                    misalignment_group_axis=axis,
                    misalignment_metadata=metadata,
                    output_dir=args.output_dir / name,
                )
            )
        status["phase"] = "report"
        analyze._write_json_atomic(status_path, status)
        build_figures(reports, args.output_dir / "figures")
        text = format_report(reports, audit, args.figure_base_url.rstrip("/"))
        analyze._write_text_atomic(args.output_dir / "results.md", text)
        methodology = text.split("## Methodology\n", 1)[1].split("## Results\n", 1)[0]
        analyze._write_text_atomic(
            args.output_dir / "methodology.md", "# Context-risk pilot methodology\n" + methodology
        )
        status.update(
            state="completed",
            phase="analysis_complete",
            completed_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            full_claim_supported=False,
        )
        analyze._write_json_atomic(status_path, status)
    except Exception as error:
        status.update(
            state="failed",
            error=f"{type(error).__name__}: {error}",
            failed_at=datetime.now(UTC).isoformat(),
        )
        analyze._write_json_atomic(status_path, status)
        raise
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
