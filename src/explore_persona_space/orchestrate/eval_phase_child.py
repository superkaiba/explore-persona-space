"""Isolated child entry point for one eval phase (``EPM_ISOLATE_EVAL`` path).

This module is the ``python -m`` target ``run_single``'s eval callback spawns via
``run_isolated`` when ``EPM_ISOLATE_EVAL`` is set. Running each eval phase in a
fresh child process is the root-cause fix for the vLLM orphan-worker OOM: the OS
reaps the entire child process tree (vLLM tensor-parallel workers included) on
exit, returning the GPU to a clean state before the next phase loads weights
(CLAUDE.md Gotchas; task #399 round-11).

Contract (per ``subprocess_isolation.run_isolated``):
  * ``sys.argv[1]`` — path to the input payload JSON.
  * ``sys.argv[2]`` — path to write the result fragment JSON.

The payload carries ONLY the scalar values ``run_eval_phase`` reads (no Hydra
cfg is serialized — the eval body's sole cfg dependency is the judge-model id, a
plain string), so there is no OmegaConf round-trip drift risk. The child calls
the SAME ``run_eval_phase`` the in-process path calls, so the returned fragment
is structure-identical to the in-process result (proven by a CPU-fixture test in
``tests/test_training_pipeline_fixes.py``).

Fail-loud: any exception in the eval propagates and the process exits non-zero,
which ``run_isolated`` surfaces as a ``SubprocessIsolationError`` with the child's
stderr tail. No dummy result is written on failure — the crash IS the signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    """Read the payload, run one eval phase, write the result fragment.

    Reads the JSON payload from ``argv[1]``, calls ``run_eval_phase`` with the
    unpacked scalar arguments, and writes the returned fragment dict to
    ``argv[2]``. Returns a process exit code (2 on usage error). Any eval
    exception propagates so the non-zero exit surfaces to ``run_isolated``.
    """
    if len(argv) < 3:
        sys.stderr.write("usage: eval_phase_child <input_json> <output_json>\n")
        return 2

    in_path, out_path = Path(argv[1]), Path(argv[2])
    payload = json.loads(in_path.read_text())

    # Imported here (not at module top) so that a usage error / arg parse does
    # not pay the heavy import cost, and so the import lands inside the child's
    # fresh process — exactly where the framework loads are meant to happen.
    from explore_persona_space.orchestrate.runner import run_eval_phase

    fragment = run_eval_phase(
        payload["model_path"],
        payload["phase"],
        materialize_merged=payload["materialize_merged"],
        eval_base_model_id=payload["eval_base_model_id"],
        judge_model=payload["judge_model"],
        phase_dir=payload["phase_dir"],
    )

    out_path.write_text(json.dumps(fragment, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
