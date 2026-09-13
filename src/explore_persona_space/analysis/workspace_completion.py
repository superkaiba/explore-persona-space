"""Outcome-blind final-rollout eligibility for the registered paired test cohort."""

from __future__ import annotations

from pathlib import Path

from explore_persona_space.analysis.workspace_capture import answer_token_ids
from explore_persona_space.analysis.workspace_runtime import file_sha256


def checkpoint_policy(config, role):
    """Resolve the native checkpoint's actual defaults without loading model weights."""
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    from explore_persona_space.analysis.workspace_terminals import terminal_policy

    model, revision = config["selection"][role], config["models"][role]["revision"]
    metadata = HfApi().model_info(model, revision=revision)
    if metadata.sha != revision:
        raise ValueError("Checkpoint metadata differs from the pinned revision")
    standalone = "generation_config.json" in {entry.rfilename for entry in metadata.siblings}
    generation = (
        GenerationConfig.from_pretrained(model, revision=revision)
        if standalone
        else GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(model, revision=revision)
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    files = ["config.json", "tokenizer_config.json"] + (
        ["generation_config.json"] if standalone else []
    )
    return terminal_policy(generation, tokenizer), {
        "model": model,
        "revision": revision,
        "generation_config_json_present": standalone,
        "file_sha256": {
            name: file_sha256(Path(hf_hub_download(model, name, revision=revision)))
            for name in files
        },
    }


def scoring_implementation():
    """Bind the eligibility implementation and the pre-outcome scoring declaration."""
    return {
        name: file_sha256(Path(name))
        for name in (
            "src/explore_persona_space/analysis/workspace_completion.py",
            "scripts/workspace_jr_completion_cohort.py",
            "docs/exploratory_workspace_jr/completion_scoring_20260913.md",
        )
    }


def final_rollout_eligibility(draws, seeds, terminal_ids):
    """Inspect final draws only; retained cap-recovery history never selects rows."""
    if not seeds or len(seeds) != len(set(seeds)) or not terminal_ids:
        raise ValueError("Complete-rollout eligibility needs exact seeds and terminal IDs")
    actual = [draw["seed"] for draw in draws]
    if len(actual) != len(set(actual)) or set(actual) - set(seeds):
        raise ValueError("Duplicate or unexpected rollout seeds")
    rows, reasons = [], []
    lookup = {draw["seed"]: draw for draw in draws}
    for seed in seeds:
        if seed not in lookup:
            reasons.append({"seed": seed, "reason": "missing_rollout"})
            continue
        draw = lookup[seed]
        ids = draw["token_ids"]
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Rollout token IDs must be nonnegative integers")
        answer, removed = answer_token_ids(ids, set(terminal_ids))
        rows.append(
            {
                "seed": seed,
                "finish_reason": draw["finish_reason"],
                "answer_tokens": len(answer),
                "terminal_ids_removed": removed,
                "max_new_tokens": draw["max_new_tokens"],
            }
        )
        if draw["finish_reason"] != "stop":
            reasons.append(
                {
                    "seed": seed,
                    "reason": "unfinished_rollout",
                    "finish_reason": draw["finish_reason"],
                }
            )
        if not answer:
            reasons.append({"seed": seed, "reason": "empty_answer_span"})
    return {"eligible": not reasons, "draws": rows, "exclusions": reasons}


def joint_completion_cohort(by_role):
    """Keep identical planned prompts; exclude a context when either model fails."""
    if set(by_role) != {"primary", "comparison"}:
        raise ValueError("Completion scoring requires both selected models")
    planned = by_role["primary"]["planned_context_ids"]
    if (
        not planned
        or len(planned) != len(set(planned))
        or by_role["comparison"]["planned_context_ids"] != planned
    ):
        raise ValueError("Models must share the exact frozen planned test contexts")
    for role, report in by_role.items():
        if set(report["contexts"]) != set(planned):
            raise ValueError(f"Incomplete completion accounting: {role}")
        for context in report["contexts"].values():
            if type(context["eligible"]) is not bool or context["eligible"] != (
                not context["exclusions"]
            ):
                raise ValueError("Completion eligibility contradicts its exclusion reasons")
    included, excluded = [], []
    for context in planned:
        failures = {
            role: report["contexts"][context]["exclusions"]
            for role, report in by_role.items()
            if not report["contexts"][context]["eligible"]
        }
        if failures:
            excluded.append({"context_id": context, "models": failures})
        else:
            included.append(context)
    if len(included) < 2:
        raise ValueError("Fewer than two jointly completed test contexts")
    return {
        "policy": "five_nonempty_final_stop_draws_in_both_models_no_backfill",
        "planned_context_ids": planned,
        "joint_complete_context_ids": sorted(included),
        "joint_complete_contexts": len(included),
        "excluded_contexts": excluded,
    }
