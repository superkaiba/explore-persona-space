"""Rescore pinned historical GPQA draws and bootstrap paired question clusters."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download


def write_json(path, value):
    """Persist a complete JSON artifact without nonfinite placeholder values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(data):
    """Return the content digest used to bind every consumed source."""
    return hashlib.sha256(data).hexdigest()


def original_parser(plan, out):
    """Load only the original deterministic grading definitions, without its imports."""
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{plan['producer_sha']}:scripts/issue2588_panel_common.py",
        ],
        text=True,
    )
    names = {
        "_BOXED_RE",
        "_LETTER_RE",
        "_MCQ_ANCHOR_RE",
        "_MCQ_BARE_LINE_RE",
        "extract_boxed",
        "extract_mcq_letter",
        "gpqa_letter_correct",
    }
    chosen, found = [], set()
    for node in ast.parse(source).body:
        name = None
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
        if name in names:
            chosen.append(ast.get_source_segment(source, node))
            found.add(name)
    if found != names:
        raise ValueError("Pinned grading definitions are incomplete")
    parser = "import re\n\n" + "\n\n".join(chosen) + "\n"
    (out / "original_grading_definitions.py").write_text(parser)
    namespace = {"re": re}
    exec(compile(parser, "original_grading_definitions.py", "exec"), namespace)
    return namespace["gpqa_letter_correct"], sha256(source.encode())


def cluster_bootstrap(correct, counts, indices):
    """Resample whole questions, retaining every draw and pairing across models."""
    if correct.shape != counts.shape or correct.shape[0] != 2 or (counts < 1).any():
        raise ValueError("Question clusters need positive observed counts for both models")
    selected_correct, selected_counts = correct[:, indices], counts[:, indices]
    pooled = selected_correct.sum(2) / selected_counts.sum(2)
    equal = (selected_correct / selected_counts).mean(2)
    # Independent scalar arithmetic catches axis/pairing errors in the actual batched path.
    for b in range(min(3, len(indices))):
        for model in range(2):
            rows = indices[b].tolist()
            oracle = sum(float(correct[model, q]) for q in rows) / sum(
                float(counts[model, q]) for q in rows
            )
            if not np.isclose(pooled[model, b], oracle, rtol=1e-14, atol=1e-14):
                raise ValueError("Batched question bootstrap differs from scalar oracle")
    return pooled, equal


def estimates(point, samples):
    """Report both accuracies and their paired primary-minus-comparison difference."""
    values = np.concatenate([point, [point[0] - point[1]]])
    draws = np.concatenate([samples, (samples[0] - samples[1])[None]])
    names = ("q35_27b", "q35_4b", "primary_minus_comparison")
    return {
        name: {"estimate": float(value), "ci95": np.quantile(row, [0.025, 0.975]).tolist()}
        for name, value, row in zip(names, values, draws, strict=True)
    }


def main(plan_path, out):
    """Verify source bytes, reproduce historical grades and save the specified analyses."""
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan["models"] != ["q35_27b", "q35_4b"] or plan["seeds"] != list(range(42, 47)):
        raise ValueError("This clarification is restricted to the frozen two-model selection")
    if (out / "complete.json").exists():
        raise ValueError("Completed analysis must not be overwritten")
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists() and (out / "plan.json").read_bytes() != plan_bytes:
        raise ValueError("An existing output carries a different analysis declaration")
    (out / "plan.json").write_bytes(plan_bytes)
    (out / "reader.py").write_bytes(Path(__file__).read_bytes())
    grade, parser_source_sha = original_parser(plan, out)
    prefix = "issue2588_capability_panel_cap_long"
    paths = []
    for model in plan["models"]:
        paths += [f"{prefix}/{model}/nothink/parsed/gpqa_s{seed}.jsonl" for seed in plan["seeds"]]
        paths += [
            f"{prefix}/fits/{model}_a/gpqa_{kind}_prompt_last.json"
            for kind in ("transfer", "perrow")
        ]
    api = HfApi()
    metadata = {
        item.path: item
        for item in api.get_paths_info(
            plan["repo"], paths, repo_type="dataset", revision=plan["revision"]
        )
    }
    if set(metadata) != set(paths):
        raise ValueError("The pinned source revision does not contain the exact input set")
    verified, payload = {}, {}
    for path in paths:
        local = Path(
            hf_hub_download(
                plan["repo"],
                path,
                repo_type="dataset",
                revision=plan["revision"],
                cache_dir="/mnt/eps-data/thomasjiralerspong/huggingface-cache/hub",
            )
        )
        data, info = local.read_bytes(), metadata[path]
        digest = sha256(data)
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != info.size or (
            digest != info.lfs.sha256 if info.lfs else blob != info.blob_id
        ):
            raise ValueError(f"Source bytes differ from immutable Hub metadata: {path}")
        verified[path] = {"sha256": digest, "size": len(data), "git_blob_id": blob}
        payload[path] = data.decode()
        write_json(out / "verified_inputs.json", verified)
        print(f"[capability-input] {len(verified)}/{len(paths)} {path}", flush=True)
    questions = [f"gpqa_{i:03d}" for i in range(plan["question_count"])]
    correct = np.zeros((2, len(questions)), dtype=np.int64)
    counts = np.zeros_like(correct)
    identities, model_reports = {}, {}
    qindex = {question: index for index, question in enumerate(questions)}
    for mi, model in enumerate(plan["models"]):
        observed, records = set(), []
        for seed in plan["seeds"]:
            path = f"{prefix}/{model}/nothink/parsed/gpqa_s{seed}.jsonl"
            for line in payload[path].splitlines():
                if not line.strip():
                    raise ValueError("Unexpected empty JSONL source row")
                row = json.loads(line)
                qid, row_id = row["qid"], row["row_id"]
                if (
                    qid not in qindex
                    or row_id != f"{qid}_s{seed}"
                    or row_id in observed
                    or row["gen_seed"] != seed
                    or row["stage"] != f"gpqa_s{seed}"
                ):
                    raise ValueError("GPQA row identity or seed contract differs")
                identity = {"prompt_sha256": sha256(row["prompt"].encode()), "gold": row["gold"]}
                if qid in identities and identities[qid] != identity:
                    raise ValueError("Paired questions differ in their rendered prompt or gold")
                identities[qid] = identity
                start, end = row["ans_char_span"]
                if not (0 <= start < end <= len(row["text"])):
                    raise ValueError("Historical answer span is empty or invalid")
                ok, letter = grade(row["text"][start:end], row["gold"])
                correct[mi, qindex[qid]] += int(ok)
                counts[mi, qindex[qid]] += 1
                observed.add(row_id)
                records.append(
                    {
                        "qid": qid,
                        "row_id": row_id,
                        "seed": seed,
                        **identity,
                        "extracted_letter": letter,
                        "correct": bool(ok),
                        "finish_reason": row["finish_reason"],
                    }
                )
        aggregate = json.loads(payload[f"{prefix}/fits/{model}_a/gpqa_transfer_prompt_last.json"])
        retrieval = json.loads(payload[f"{prefix}/fits/{model}_a/gpqa_perrow_prompt_last.json"])
        totals = {
            "n_rollouts": len(records),
            "n_correct": sum(row["correct"] for row in records),
            "n_unparseable": sum(row["extracted_letter"] is None for row in records),
        }
        if (
            set(retrieval["row_ids"]) != observed
            or len(retrieval["row_ids"]) != len(observed)
            or any(aggregate["behavioral"][key] != value for key, value in totals.items())
            or aggregate["behavioral"]["judge_fallback_flagged"] is not False
            or aggregate["meta"]["git_sha"] != plan["producer_sha"]
        ):
            raise ValueError(
                "Historical aggregate/retrieval eligibility differs from actual grades"
            )
        planned_ids = {f"{qid}_s{seed}" for qid in questions for seed in plan["seeds"]}
        missing = sorted(planned_ids - observed)
        model_reports[model] = {
            **totals,
            "planned_draws": len(planned_ids),
            "missing_row_ids": missing,
            "finish_reasons": dict(Counter(row["finish_reason"] for row in records)),
            "question_observed_count_histogram": dict(Counter(counts[mi].tolist())),
            "missing_draw_accuracy_bounds": [
                totals["n_correct"] / len(planned_ids),
                (totals["n_correct"] + len(missing)) / len(planned_ids),
            ],
        }
        write_json(out / f"{model}_grades.json", records)
        write_json(out / "model_counts.json", model_reports)
    if set(identities) != set(questions):
        raise ValueError("The two-model source pool does not cover every planned question")
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    indices = rng.integers(0, len(questions), (plan["bootstrap"]["draws"], len(questions)))
    pooled, equal = cluster_bootstrap(correct, counts, indices)
    complete = np.flatnonzero((counts == len(plan["seeds"])).all(0))
    if len(complete) < 2:
        raise ValueError("The paired complete-question sensitivity has fewer than two questions")
    complete_indices = rng.integers(0, len(complete), (len(indices), len(complete)))
    complete_samples, _ = cluster_bootstrap(
        correct[:, complete], counts[:, complete], complete_indices
    )
    bounds = [model_reports[model]["missing_draw_accuracy_bounds"] for model in plan["models"]]
    report = {
        "schema": "workspace-jr-capability-question-bootstrap-v1",
        "plan": plan,
        "numpy_version": np.__version__,
        "original_parser_source_sha256": parser_source_sha,
        "question_ids": questions,
        "source_counts": model_reports,
        "pooled_observed_rollouts": estimates(correct.sum(1) / counts.sum(1), pooled),
        "equal_question_weighting": estimates((correct / counts).mean(1), equal),
        "joint_complete_questions": {
            "question_ids": [questions[i] for i in complete],
            "excluded_question_ids": [q for i, q in enumerate(questions) if i not in complete],
            "estimates": estimates(
                correct[:, complete].sum(1) / counts[:, complete].sum(1), complete_samples
            ),
        },
        "missing_draw_difference_bounds": [
            bounds[0][0] - bounds[1][1],
            bounds[0][1] - bounds[1][0],
        ],
        "selection_was_frozen_before_this_analysis": True,
    }
    np.savez(
        out / "question_bootstrap.npz",
        question_ids=np.asarray(questions),
        correct=correct,
        counts=counts,
        indices=indices,
        pooled_samples=pooled,
        equal_question_samples=equal,
        complete_question_indices=complete,
        complete_bootstrap_indices=complete_indices,
        complete_samples=complete_samples,
    )
    write_json(out / "results.json", report)
    write_json(
        out / "complete.json",
        {
            "status": "complete",
            "source_files_verified": len(verified),
            "results_sha256": sha256((out / "results.json").read_bytes()),
            "files_sha256": {
                p.name: sha256(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()
            },
        },
    )
    print(json.dumps(report["pooled_observed_rollouts"], indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: workspace_jr_capability_uncertainty.py PLAN_JSON OUTPUT_DIRECTORY")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
