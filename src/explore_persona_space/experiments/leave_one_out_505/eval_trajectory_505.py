# em-dash + Greek ΔG intentional
"""Task #505 §5.5 — eval-trajectory wrapper that wires the #477 silent-LoRA guard.

Calls the #472 ``run_trajectory_eval`` for the heavy work, then re-reads the
written trajectory.json and runs ``assert_adapter_actually_applied`` on the
trained vs base log-prob records for the headline checkpoint frac (default
0.50). This is the §5.5 smoke gate (f) positive-control: the guard MUST be
called by the eval rig AND return cleanly on the smoke cell, otherwise the
silent-LoRA-not-applied regression (#477's exact failure mode) is left
undetected.

The #477 cherry-pick lives at
``contrastive_neg_geometry_472.eval_guard.assert_adapter_actually_applied`` —
see plan §10 step 0.

The guard reads ``adapter_dir/adapter_model.safetensors`` for the B-matrix max
Frobenius norm and compares the trained / base ΔG + emission across the panel.
We call it ONCE per cell at the headline checkpoint — running it at every frac
would inflate cost and the bug class (silent LoRA-not-applied) would manifest
uniformly across the whole trajectory if it manifested at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
    assert_adapter_actually_applied,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
    run_trajectory_eval,
)
from explore_persona_space.experiments.leave_one_out_505 import (
    HEADLINE_CHECKPOINT_FRAC,
    MAX_NEW_TOKENS_GEN,
)

log = logging.getLogger("issue_505.eval_trajectory")


def _extract_records_at_frac(payload: dict, frac: float, eval_personas: list[str]):
    """Reconstruct the (g_records, b_records) dicts the guard expects from a trajectory.json.

    ``assert_adapter_actually_applied`` expects::
        {persona: {q: {"logp": float, "argmax_marker": bool}}}

    The trajectory writes ``held_out[persona][q] = {"g_logp", "b_logp",
    "delta_g", "argmax_marker", ...}`` — same shape minus the "logp" key
    rename. Both passes carry their own logp under different field names; this
    function pulls them out into the rig-compatible shape.
    """
    target_2 = f"{frac:.2f}"
    target_4 = f"{frac:.4f}"

    def _frac_match(ckpt: dict) -> bool:
        raw = ckpt.get("frac")
        if isinstance(raw, str):
            return raw in (target_2, target_4)
        if isinstance(raw, (int, float)):
            return abs(float(raw) - frac) < 1e-4
        return False

    ckpt = next((c for c in payload["checkpoints"] if _frac_match(c)), None)
    if ckpt is None:
        raise KeyError(
            f"trajectory has no checkpoint at frac={frac!r}; checkpoints: "
            f"{[c.get('frac') for c in payload['checkpoints']]}"
        )

    g_records: dict[str, dict[str, dict[str, float | bool]]] = {}
    b_records: dict[str, dict[str, dict[str, float | bool]]] = {}
    held_out = ckpt.get("held_out", {})
    for persona in eval_personas:
        if persona not in held_out:
            continue
        g_records[persona] = {}
        b_records[persona] = {}
        for q, leaf in held_out[persona].items():
            g_records[persona][q] = {
                "logp": float(leaf["g_logp"]),
                "argmax_marker": bool(leaf.get("argmax_marker", False)),
            }
            b_records[persona][q] = {
                "logp": float(leaf["b_logp"]),
                "argmax_marker": False,
            }
    return g_records, b_records, ckpt


def run_trajectory_eval_with_guard(
    *,
    cell_slug: str,
    seed: int,
    checkpoint_specs: list[dict],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    source: str,
    source_prompt: str,
    out_path: Path,
    base_model: str,
    max_new_tokens: int = MAX_NEW_TOKENS_GEN,
    headline_frac: float = HEADLINE_CHECKPOINT_FRAC,
    compute_kl: bool = True,
    max_lora_rank: int = 16,
) -> Path:
    """Run the #472 trajectory eval, then run the #477 silent-LoRA guard at the
    headline checkpoint.

    Raises ``LoRANotAppliedError`` (from the guard) if the adapter is genuinely
    trained but reads no signal at every probe AND emits no marker anywhere —
    the #477 v4/v6 regression class. The dispatcher catches this on smoke
    (§5.5 gate f) and halts before the sweep.

    Side-effect: appends ``eval_guard_diagnostic`` to the trajectory.json's
    headline checkpoint so downstream analyzers can audit the verdict.
    """
    # Phase A: heavy work (vLLM gen + DV-A + DV-B KL).
    trajectory_path = run_trajectory_eval(
        cell_slug=cell_slug,
        seed=seed,
        checkpoint_specs=checkpoint_specs,
        eval_personas=eval_personas,
        eval_questions=eval_questions,
        source=source,
        source_prompt=source_prompt,
        out_path=out_path,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        max_lora_rank=max_lora_rank,
        compute_kl=compute_kl,
    )

    # Phase B: run the #477 guard at the headline checkpoint.
    payload = json.loads(Path(trajectory_path).read_text())
    g_records, b_records, ckpt = _extract_records_at_frac(
        payload, headline_frac, list(eval_personas.keys())
    )
    adapter_dir = ckpt.get("adapter_path")
    if not adapter_dir:
        log.warning(
            "[eval-guard] no adapter_path for cell=%s seed=%s frac=%s; skipping guard",
            cell_slug,
            seed,
            headline_frac,
        )
        return trajectory_path

    label = f"{cell_slug}_seed{seed}_frac{headline_frac}"
    # The guard RAISES on the regression class; otherwise it returns a diag dict.
    diag = assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g_records,
        b_records=b_records,
        cell_label=label,
    )
    log.info("[eval-guard] %s: %s", label, diag.get("guard_verdict"))

    # Append the diagnostic onto the headline ckpt block + rewrite the file.
    ckpt["eval_guard_diagnostic"] = diag
    Path(trajectory_path).write_text(json.dumps(payload, indent=2))
    return trajectory_path
