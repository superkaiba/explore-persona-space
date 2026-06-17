"""Regression for the #545 outdist HF teacher-force sub-batch (``JS_TF_MAX_BATCH``).

History: the r4 outdist OOM (5.35 GiB transient, ~25 min in) was HF-side —
``jsc.teacher_forced_response_logps`` materializes a ``(B, L, V)`` logits tensor
inside the HF forward where ``V~=152k`` for Qwen-2.5-7B. At ``max_batch=16`` the
bf16 transient is ``16 x 1024 x 152064 x 2 bytes ~= 5.0 GiB``. While the HF model
and the vLLM engine CO-RESIDED on one H100 (the pre-Strategy-E architecture),
that 5 GiB transient did not fit the thin co-residency headroom, so r4 lowered
the constant to 4 (~1.3 GiB).

STRATEGY E (round-38): vLLM now runs in a SUBPROCESS (``vllm_worker.py``) that
exits before the HF base model loads, so during the teacher-force phase the HF
model is the SOLE GPU resident — it has the full ~80 GiB to itself. The 5 GiB
B=16 transient fits with enormous margin, so the r4 co-residency workaround is
REVERSED and the constant is restored to 16 for throughput. The scoring function
also moved its SAMPLING out to the subprocess and was renamed
``_score_outdist_pair_from_samples`` (it now consumes already-sampled responses).

These tests pin the Strategy-E invariant WITHOUT a GPU: the constant value, the
per-call-site keyword wiring (AST-walked from the production source), and the
transient-size arithmetic showing B=16 fits an isolated HF model.
"""

import ast
from pathlib import Path

from explore_persona_space.experiments.behavior_testbed_545.predictors_zoo import (
    JS_TF_MAX_BATCH,
)

_PREDICTORS_ZOO = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "explore_persona_space"
    / "experiments"
    / "behavior_testbed_545"
    / "predictors_zoo.py"
)

# Qwen-2.5-7B unembedding vocab; the teacher-force forward's logits last dim.
_QWEN_VOCAB = 152_064
# Sequence length the transient arithmetic uses (a typical sampled response; the
# JS_MAX_SEQ_LEN cap bounds prompt + sampled response at 8192, but the (B, L, V)
# transient is L = the RESPONSE length being scored, ~1024 sampled tokens).
_SEQ_LEN = 1024
_BF16_BYTES = 2
_GIB = 1024**3
# An isolated HF 7B base model occupies ~15 GiB of weights + ~a few GiB KV/state
# on an 80 GiB H100, leaving well over 50 GiB free for the logits transient.
_ISOLATED_HF_HEADROOM_GIB = 50.0


def _outdist_tf_calls() -> list[ast.Call]:
    """Every ``teacher_forced_response_logps`` Call inside the outdist scorer
    (``_score_outdist_pair_from_samples`` under Strategy E)."""
    tree = ast.parse(_PREDICTORS_ZOO.read_text())
    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_score_outdist_pair_from_samples"
    )
    calls = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Attribute):
            name = fn.attr
        elif isinstance(fn, ast.Name):
            name = fn.id
        if name == "teacher_forced_response_logps":
            calls.append(node)
    return calls


def test_js_tf_max_batch_is_sixteen_under_strategy_e():
    """Strategy E restores the sub-batch to 16 (HF is sole GPU resident — the
    r4 lowering to 4 was a co-residency workaround that no longer applies)."""
    assert JS_TF_MAX_BATCH == 16


def test_outdist_calls_teacher_force_with_the_constant():
    """EVERY outdist HF teacher-force call passes ``max_batch=JS_TF_MAX_BATCH``
    (the module constant, never a hardcoded literal)."""
    calls = _outdist_tf_calls()
    # The pair scorer teacher-forces BOTH conditioned contexts (lp_a, lp_b).
    assert len(calls) == 2, (
        f"expected 2 teacher-force calls in _score_outdist_pair_from_samples, got {len(calls)}"
    )
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "max_batch" in kw, "outdist teacher-force call missing max_batch kwarg"
        val = kw["max_batch"]
        assert isinstance(val, ast.Name) and val.id == "JS_TF_MAX_BATCH", (
            f"max_batch must be the JS_TF_MAX_BATCH constant, not a literal (got {ast.dump(val)})"
        )


def test_js_tf_max_batch_transient_fits_isolated_hf_headroom():
    """B=16 x L=1024 x V~=152k x 2 bytes (bf16) ~= 5.0 GiB transient, comfortably
    inside the >50 GiB an isolated HF base model leaves on an 80 GiB H100 once
    vLLM has exited its subprocess (Strategy E)."""
    transient_gib = JS_TF_MAX_BATCH * _SEQ_LEN * _QWEN_VOCAB * _BF16_BYTES / _GIB
    assert transient_gib <= _ISOLATED_HF_HEADROOM_GIB, (
        f"B={JS_TF_MAX_BATCH} transient {transient_gib:.2f} GiB exceeds isolated-HF headroom"
    )
    # Sanity: it is the expected ~5 GiB (the value that OOM'd under co-residency,
    # now harmless when HF is sole resident).
    assert 4.5 <= transient_gib <= 5.5, f"B={JS_TF_MAX_BATCH} transient {transient_gib:.2f} GiB"
