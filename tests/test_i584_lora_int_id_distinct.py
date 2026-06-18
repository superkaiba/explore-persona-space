"""#584 regression — multi-checkpoint trajectory eval must use distinct lora_int_id.

Pins the #534 round-1 incident at the source level: vLLM caches LoRA adapters
strictly by lora_int_id (LRUCacheWorkerLoRAManager.add_adapter "just touches"
an already-seen id — lora_path never re-read), so a constant id inside the
checkpoint loop silently serves the FIRST-loaded adapter at every fraction.
Static layer (this file): the LoRARequest construction inside
run_trajectory_eval must not pass a literal lora_int_id — neither as the
keyword ``lora_int_id=<constant>`` nor as a positional constant at argument
index 1 (lora_int_id is LoRARequest's second positional field, so
``LoRARequest(label, 1, path)`` would evade a keyword-only checker). Runtime
layer (ported with the fix): eval_guard.assert_source_delta_g_matches_manifest
(tests/test_i534_source_manifest_guard.py). No GPU, no vllm import.
"""

import ast
from pathlib import Path

EVAL_TRAJECTORY = (
    Path(__file__).resolve().parent.parent
    / "src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py"
)

OLD_BUGGY_SNIPPET_KEYWORD = """
def run_trajectory_eval(checkpoint_specs):
    for spec in checkpoint_specs:
        label = "x"; adapter_path = spec["adapter_path"]
        lora_req = LoRARequest(lora_name=label, lora_int_id=1, lora_path=adapter_path)
"""  # verbatim shape of pre-#584 main (eval_trajectory.py:425)

OLD_BUGGY_SNIPPET_POSITIONAL = """
def run_trajectory_eval(checkpoint_specs):
    for spec in checkpoint_specs:
        label = "x"; adapter_path = spec["adapter_path"]
        lora_req = LoRARequest(label, 1, adapter_path)
"""  # positional-evasion variant: constant at args index 1 (the lora_int_id slot)


def _lora_request_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_trajectory_eval"
    )
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "LoRARequest"
    ]


def _constant_lora_int_id_lines(source: str) -> list[int]:
    """Lines where a LoRARequest inside run_trajectory_eval gets a LITERAL lora_int_id.

    Flags BOTH shapes: keyword ``lora_int_id=<ast.Constant>`` AND a positional
    ``ast.Constant`` at args index 1 (lora_int_id is the second positional field).
    """
    lines: list[int] = []
    for c in _lora_request_calls(source):
        for kw in c.keywords:
            if kw.arg == "lora_int_id" and isinstance(kw.value, ast.Constant):
                lines.append(c.lineno)
        if len(c.args) >= 2 and isinstance(c.args[1], ast.Constant):
            lines.append(c.lineno)
    return lines


def test_checker_flags_the_old_constant_id_construction():
    # self-validation, both shapes: the checker must flag the literal incident
    # shape (keyword) AND the positional-evasion variant (constant at args[1]).
    assert _constant_lora_int_id_lines(OLD_BUGGY_SNIPPET_KEYWORD)
    assert _constant_lora_int_id_lines(OLD_BUGGY_SNIPPET_POSITIONAL)


def test_live_rig_constructs_non_constant_lora_int_id():
    assert _constant_lora_int_id_lines(EVAL_TRAJECTORY.read_text()) == []


def test_live_rig_still_constructs_a_lora_request():
    # anti-rot: the contract must not pass vacuously after a rename/refactor
    assert len(_lora_request_calls(EVAL_TRAJECTORY.read_text())) >= 1
