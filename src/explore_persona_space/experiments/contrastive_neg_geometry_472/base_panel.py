# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #472 Phase 1.5 — base per-persona marker prior b_logprob (plan §5 / §6).

The base-model marker log-prob ``b_logprob`` on the BASE model's OWN frozen R
(from r_generate), per held-out persona × Q_eval. This is the persona prior that
ΔG subtracts and that the geometry regression PARTIALS OUT (the #448 artifact
guard: ΔG ≈ −b_logprob mechanically reproduces persona structure; partialling
b_logprob rules it out). #448 reported a 7.3-nat spread across personas
(−18.69..−25.96), so this is a real partialling control (assumption 14).

Distinct from the trajectory's per-checkpoint base pass: that base pass reads the
base marker log-prob on the TRAINED model's R_j (to form ΔG at each checkpoint on
matched on-policy text). THIS panel reads the base marker log-prob on the BASE
model's frozen R — a single training-independent per-persona prior, computed once.

Subprocess-isolated (vLLM) per the dispatcher's teardown discipline.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
    assert_marker_token,
    score_logp_for_R,
)

log = logging.getLogger("issue_472.base_panel")


def _git_sha() -> str:
    import os

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_base_panel(
    *,
    eval_personas: dict[str, str],
    eval_questions: list[str],
    r_eval_base: dict[str, dict[str, dict]],
    out_path: Path,
    base_model: str = BASE_MODEL,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 2048,
    seed: int = 42,
) -> Path:
    """Compute base per-persona marker prior b_logprob on the base model's R.

    Args:
        eval_personas: held-out panel {persona: system_prompt}.
        eval_questions: Q_eval.
        r_eval_base: base R artifact completions (persona -> q -> {response_text,...}).
        out_path: base_panel.json output.
        base_model, gpu_memory_utilization, max_model_len, seed: vLLM params.

    Returns:
        out_path. Writes {persona: {q: b_logp}} + per-persona means.
    """
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    assert_marker_token(tokenizer)

    # Flatten the base R artifact to r[persona][q] -> text.
    r_text: dict[str, dict[str, str]] = {}
    for persona in eval_personas:
        if persona not in r_eval_base:
            raise KeyError(f"base R missing persona {persona!r}.")
        r_text[persona] = {q: r_eval_base[persona][q]["response_text"] for q in eval_questions}

    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        max_model_len=max_model_len,
    )
    b = score_logp_for_R(
        llm,
        tokenizer,
        r_by_persona_q=r_text,
        eval_personas=eval_personas,
        eval_questions=eval_questions,
        cell_label="BASE_PANEL",
        use_lora=False,
    )

    import numpy as np

    b_by_pq = {p: {q: b[p][q]["logp"] for q in eval_questions} for p in eval_personas}
    b_mean = {p: float(np.mean(list(b_by_pq[p].values()))) for p in eval_personas}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "i472_v1",
        "base_model": base_model,
        "marker_text": MARKER_TEXT,
        "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        "n_personas": len(eval_personas),
        "eval_questions": eval_questions,
        "b_logprob_per_persona_q": b_by_pq,
        "mean_per_persona_b_logprob": b_mean,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    spread = max(b_mean.values()) - min(b_mean.values()) if b_mean else 0.0
    log.info(
        "[phase=base_panel] Wrote b_logprob for %d personas (spread=%.2f nats) → %s",
        len(eval_personas),
        spread,
        out_path,
    )
    return out_path
