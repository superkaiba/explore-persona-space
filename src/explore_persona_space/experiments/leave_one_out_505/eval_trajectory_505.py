# ruff: noqa: RUF002, RUF003  # em-dash + Greek ΔG + × (multiplication) intentional
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
    FALLBACK_LORA_R,
    HEADLINE_CHECKPOINT_FRAC,
    LORA_R,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS_GEN,
)

# vLLM cap that accommodates BOTH the §5.1 primary anchor (LORA_R=16) AND the
# §5.5 smoke fallback anchor (FALLBACK_LORA_R=32). Trained adapter rank is
# still pinned to plan §5.1 by the dispatcher via ``lora_r_override=LORA_R``;
# this is just the upper bound vLLM will accept on load. Round-6 default bump
# (the original default of 16 rejected the round-6 rank-32 adapter — which was
# itself a separate dispatcher bug — at eval_trajectory.py:179
# ``ValueError: LoRA rank 32 is greater than max_lora_rank 16``).
_DEFAULT_MAX_LORA_RANK = max(LORA_R, FALLBACK_LORA_R)

# vLLM ``max_model_len`` default for #505 (round 10, 2026-06-08). #472's
# ``DEFAULT_MAX_MODEL_LEN = 2048`` rejected the round-9 trajectory eval when
# the trained model's on-policy ``R_j`` approached ``MAX_NEW_TOKENS_GEN``
# (2048) and the post-R-slot ``score_logp_for_R`` prompt exceeded 2048 with
# prefix + marker context. ``MAX_MODEL_LEN = 4096`` (2× MAX_NEW_TOKENS_GEN)
# overrides #472's default at #505's call site only. Shared #472 code is
# untouched. See ``leave_one_out_505/__init__.py`` for the round-9 crash
# signature and rationale.
_DEFAULT_MAX_MODEL_LEN = MAX_MODEL_LEN

log = logging.getLogger("issue_505.eval_trajectory")


def _extract_records_at_frac(
    payload: dict, frac: float, eval_personas: list[str], source: str | None = None
):
    """Reconstruct the (g_records, b_records) dicts the guard expects from a trajectory.json.

    ``assert_adapter_actually_applied`` expects::
        {persona: {q: {"logp": float, "argmax_marker": bool}}}

    The trajectory writes ``held_out[persona][q] = {"g_logp", "b_logp",
    "delta_g", "argmax_marker", ...}`` — same shape minus the "logp" key
    rename. Both passes carry their own logp under different field names; this
    function pulls them out into the rig-compatible shape.

    Per the #472 eval_guard contract (eval_guard.py:14): ``assert_adapter_
    actually_applied`` reads max ``|ΔG|`` across **ALL** probes — source-self
    AND held-out. The #472 ``eval_trajectory.py`` writes per-q source-self
    records under the ``source_probes`` block (added 2026-06-05 alongside the
    mean-pooled ``source_self``); we MUST merge those into ``g_records`` /
    ``b_records`` before calling the guard. Without the source key, a clean
    contrastive run in the floor regime (held-out max|ΔG| ~ 0.23-0.32 nats,
    n_emit=0 — the *success* signature, where negatives suppress leakage to
    bystanders by design) would falsely trip ``LoRANotAppliedError``. The
    #505 sweep has multiple legitimate floor-regime cells; a false-raise
    would crash the sweep mid-run.

    Args:
        payload: parsed trajectory.json.
        frac: checkpoint fraction to extract.
        eval_personas: held-out panel persona names (excludes the source).
        source: source persona name — included alongside ``eval_personas`` so
            the guard sees the source-self probes the contract requires. If
            None, only held-out records are returned (legacy callers).

    Returns:
        ``(g_records, b_records, ckpt)`` where the records dicts are keyed by
        persona (held-out + source if provided + present in the trajectory).
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

    # Merge source-self per-q records when the trajectory carries the
    # ``source_probes`` block (the #472 schema bump 2026-06-05). The guard
    # contract requires source-self in the panel; without it a floor-regime
    # contrastive cell would false-raise (see docstring).
    if source is not None:
        source_probes = ckpt.get("source_probes")
        if source_probes:
            g_records[source] = {}
            b_records[source] = {}
            for q, leaf in source_probes.items():
                g_records[source][q] = {
                    "logp": float(leaf["g_logp"]),
                    "argmax_marker": bool(leaf.get("argmax_marker", False)),
                }
                b_records[source][q] = {
                    "logp": float(leaf["b_logp"]),
                    "argmax_marker": False,
                }
        else:
            log.warning(
                "[eval-guard] trajectory checkpoint at frac=%s has no 'source_probes' "
                "block — guard will run on held-out only. This is the pre-2026-06-05 "
                "schema; a floor-regime contrastive cell may false-raise.",
                frac,
            )
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
    max_lora_rank: int = _DEFAULT_MAX_LORA_RANK,
    max_model_len: int = _DEFAULT_MAX_MODEL_LEN,
    source_guard_meta: dict | None = None,
    raw_completions_out_path: Path | None = None,
) -> Path:
    """Run the #472 trajectory eval, then run the #477 silent-LoRA guard at the
    headline checkpoint.

    Raises ``LoRANotAppliedError`` (from the guard) if the adapter is genuinely
    trained but reads no signal at every probe AND emits no marker anywhere —
    the #477 v4/v6 regression class. The dispatcher catches this on smoke
    (§5.5 gate f) and halts before the sweep.

    Side-effect: appends ``eval_guard_diagnostic`` to the trajectory.json's
    headline checkpoint so downstream analyzers can audit the verdict.

    ``max_model_len`` defaults to ``leave_one_out_505.MAX_MODEL_LEN`` (4096 as of
    round 10, 2026-06-08). The #472 ``run_trajectory_eval`` default of 2048
    rejected round-9's trajectory eval at frac 0.50 with::

        ValueError: The decoder prompt (length 2050) is longer than the maximum
        model length of 2048.

    when the trained model's on-policy ``R_j`` approached ``MAX_NEW_TOKENS_GEN``
    and the post-R-slot ``score_logp_for_R`` prompt (system + question + R_j +
    marker context) exceeded the 2048 cap. 4096 = 2× ``MAX_NEW_TOKENS_GEN``
    covers the worst case. The override is local to #505's call site; shared
    #472 code is untouched.
    """
    # Phase A: heavy work (vLLM gen + DV-A + DV-B KL).
    # ``source_guard_meta`` + ``raw_completions_out_path`` are #600 opt-in
    # passthroughs (both default None = byte-identical #505 behavior).
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
        max_model_len=max_model_len,
        compute_kl=compute_kl,
        source_guard_meta=source_guard_meta,
        raw_completions_out_path=raw_completions_out_path,
    )

    # Phase B: run the #477 guard at the headline checkpoint. Pass `source`
    # so the guard sees BOTH held-out + source-self per-q records — the
    # eval_guard.py:14 contract requires the full panel, and a floor-regime
    # contrastive cell (held-out near zero, n_emit=0) would false-raise
    # without source-self in the panel.
    payload = json.loads(Path(trajectory_path).read_text())
    g_records, b_records, ckpt = _extract_records_at_frac(
        payload, headline_frac, list(eval_personas.keys()), source=source
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
