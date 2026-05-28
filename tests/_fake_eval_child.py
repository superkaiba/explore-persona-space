"""Test-only isolated-child entry point with the GPU-heavy eval leaves faked.

This module exists so the #13 isolation regression test can exercise the REAL
isolated path end-to-end on a CPU-only VM: real ``run_isolated`` -> real fresh
child process -> real ``eval_phase_child.main`` -> real ``run_eval_phase``, with
ONLY the three GPU-bound leaf eval functions
(``evaluate_capability_logprob`` / ``evaluate_capability`` /
``evaluate_alignment_quick``) replaced by deterministic CPU fakes.

The fakes are installed by overwriting the attributes on the actual eval modules
BEFORE ``run_eval_phase`` runs its in-function ``from ... import name`` lookups
(``from module import name`` reads ``module.name`` at call time, so patching the
module attribute is sufficient). The fakes return the same fixed values the
in-process arm of the test patches in, so a json-equal comparison across the two
arms proves the isolated path is structure-identical to the in-process path.

Run via ``python -m`` exactly like the production child::

    python -m tests._fake_eval_child <input_json> <output_json>
"""

from __future__ import annotations

import sys

# These fixed return values MUST match the in-process arm of the regression test
# (test_isolation_path_matches_in_process_structure). They are the only place the
# two arms share fake outputs, so keep them identical.
FAKE_CAP = {"arc_challenge_logprob": 0.5, "correct": 5, "total": 10}
FAKE_OOD = {
    "mmlu_pro": {"exact_match,custom-extract": 0.3},
    "gsm8k": {"exact_match,strict-match": 0.4},
}
FAKE_ALIGN = {"overall_mean_aligned": 90.0, "overall_mean_coherent": 85.0}


def _install_fakes() -> None:
    """Overwrite the three GPU-bound eval leaves with deterministic CPU fakes."""
    import explore_persona_space.eval.alignment as alignment_mod
    import explore_persona_space.eval.capability as capability_mod

    def fake_logprob(*_args, **_kwargs):
        return dict(FAKE_CAP)

    def fake_capability(*_args, **_kwargs):
        return {k: dict(v) for k, v in FAKE_OOD.items()}

    async def fake_alignment(*_args, **_kwargs):
        return dict(FAKE_ALIGN)

    capability_mod.evaluate_capability_logprob = fake_logprob
    capability_mod.evaluate_capability = fake_capability
    alignment_mod.evaluate_alignment_quick = fake_alignment


if __name__ == "__main__":
    _install_fakes()
    from explore_persona_space.orchestrate.eval_phase_child import main

    raise SystemExit(main(sys.argv))
