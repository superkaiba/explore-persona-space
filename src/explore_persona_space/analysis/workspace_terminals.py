"""Resolve tokenizer and generation stopping IDs before selecting answer states."""

from __future__ import annotations

import torch

from explore_persona_space.analysis.workspace_capture import answer_token_ids


def terminal_policy(generation_config, tokenizer, explicit_stop_ids=()):
    """The tokenizer EOS remains terminal when model-derived defaults omit it."""
    sources = {
        "generation_config_eos": generation_config.eos_token_id,
        "tokenizer_eos": tokenizer.eos_token_id,
        "explicit_stop_ids": list(explicit_stop_ids),
    }
    for name, value in sources.items():
        values = [] if value is None else value if isinstance(value, (list, tuple)) else [value]
        if any(type(token) is not int or token < 0 for token in values):
            raise ValueError(f"Invalid terminal token IDs from {name}")
        sources[name] = sorted(set(values))
    ids = sorted({token for values in sources.values() for token in values})
    if not ids:
        raise ValueError("Capture requires an explicit nonempty terminal-ID policy")
    return {
        "policy": "all_generated_ids_before_first_terminal_eos_or_stop_id",
        "sources": sources,
        "terminal_ids": ids,
    }


def trim_saved_answer(row, raw_draw, policy, original_terminal_ids):
    """Trim an already captured causal prefix; never synthesize or recapture states."""
    if row["seed"] != raw_draw["seed"] or row["finish_reason"] != raw_draw["finish_reason"]:
        raise ValueError("Saved capture and generated draw differ")
    original, original_dropped = answer_token_ids(raw_draw["token_ids"], original_terminal_ids)
    h = row["answer_states"]
    if (
        row["answer_ids"] != original
        or row["terminal_ids_removed"] != original_dropped
        or h.ndim != 2
        or len(h) != len(original)
        or not torch.isfinite(h).all()
    ):
        raise ValueError("Source capture is not the verified original exact token prefix")
    answer, dropped = answer_token_ids(raw_draw["token_ids"], set(policy["terminal_ids"]))
    if original[: len(answer)] != answer or len(answer) > len(original):
        raise ValueError("EOS recovery would require missing token states")
    corrected = {
        **row,
        "answer_ids": answer,
        "answer_states": h[: len(answer)].clone(),
        "terminal_ids_removed": dropped,
        "terminal_policy": policy,
    }
    return corrected, {
        "seed": row["seed"],
        "original_answer_tokens": len(original),
        "corrected_answer_tokens": len(answer),
        "additional_terminal_states_removed": len(original) - len(answer),
        "empty_after_terminal_correction": not answer,
    }
