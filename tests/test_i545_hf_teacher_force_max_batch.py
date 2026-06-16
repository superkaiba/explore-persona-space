"""Regression for the #545 outdist OOM ROOT CAUSE — the HF teacher-force
``(B, L, V)`` logits transient, NOT vLLM.

The r3/r4 OOM-fix sequence repeatedly tuned vLLM's ``gpu_memory_utilization``
(0.85 -> 0.70 -> 0.60). But the r4 outdist OOM (5.35 GiB transient, ~25 min in)
was HF-side: ``jsc.teacher_forced_response_logps`` materializes a ``(B, L, V)``
logits tensor inside the HF forward where ``V~=152k`` for Qwen-2.5-7B. At the
upstream default ``max_batch=16`` the bf16 transient is
``16 x 1024 x 152064 x 2 bytes ~= 5.0 GiB`` — matching the observed allocation.
``_score_outdist_pair`` (``predictors_zoo.py``) called it twice per probe pair
WITHOUT passing ``max_batch``, so it used the default 16.

The r5 audit (no code change) was the decisive step: it correctly HALTED on the
proposed vLLM-teardown pivot (false premise — outdist also uses vLLM to SAMPLE
the R responses it then teacher-forces, so tearing vLLM down crashes on the dead
reference) and identified the real lever — ``max_batch`` on the HF call, NOT
vLLM util.

The fix (round-6 / commit "round 34"): a module constant ``JS_TF_MAX_BATCH = 4``
(the symmetric HF-side knob to vLLM's ``gpu_memory_utilization``), passed at both
``_score_outdist_pair`` call sites, cuts the transient to ~1.3 GiB — fitting the
~9.5 GiB working-memory headroom the 0.60-util config leaves once the HF model +
vLLM engine co-reside (see ``test_i545_vllm_pre_init_cleanup.py``).

These tests pin the invariant WITHOUT a GPU: the constant value, the per-call-
site keyword wiring (AST-walked from the production source), and the transient-
size arithmetic that motivates 4 over 16.
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
# Sequence length cap the outdist teacher-force runs at (JS_MAX_SEQ_LEN / the
# vLLM max_model_len that bounds prompt + sampled response).
_SEQ_LEN = 1024
_BF16_BYTES = 2
_GIB = 1024**3


def _outdist_tf_calls() -> list[ast.Call]:
    """Every ``teacher_forced_response_logps`` Call inside _score_outdist_pair."""
    tree = ast.parse(_PREDICTORS_ZOO.read_text())
    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_score_outdist_pair"
    )
    calls = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        # Match both ``jsc.teacher_forced_response_logps(...)`` and a bare
        # ``teacher_forced_response_logps(...)`` import form.
        name = None
        if isinstance(fn, ast.Attribute):
            name = fn.attr
        elif isinstance(fn, ast.Name):
            name = fn.id
        if name == "teacher_forced_response_logps":
            calls.append(node)
    return calls


def test_js_tf_max_batch_is_four():
    """The lowered HF teacher-force sub-batch constant is exactly 4."""
    assert JS_TF_MAX_BATCH == 4


def test_outdist_calls_teacher_force_with_lowered_max_batch():
    """Pin the OOM-fix invariant: EVERY outdist HF teacher-force call passes
    ``max_batch=JS_TF_MAX_BATCH`` (not the upstream default 16)."""
    calls = _outdist_tf_calls()
    # The pair scorer teacher-forces BOTH conditioned contexts (lp_a, lp_b).
    assert len(calls) == 2, (
        f"expected 2 teacher-force calls in _score_outdist_pair, got {len(calls)}"
    )
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "max_batch" in kw, (
            "outdist teacher-force call missing max_batch kwarg (uses default 16)"
        )
        val = kw["max_batch"]
        assert isinstance(val, ast.Name) and val.id == "JS_TF_MAX_BATCH", (
            f"max_batch must be the JS_TF_MAX_BATCH constant, not a literal (got {ast.dump(val)})"
        )


def test_js_tf_max_batch_transient_fits_headroom():
    """B=4 x L=1024 x V~=152k x 2 bytes (bf16) ~= 1.3 GiB transient, comfortably
    inside the ~9.5 GiB headroom the gpu_memory_utilization=0.60 config leaves.
    Sanity-floors and -ceilings the chosen sub-batch."""
    transient_gib = JS_TF_MAX_BATCH * _SEQ_LEN * _QWEN_VOCAB * _BF16_BYTES / _GIB
    assert 1.0 <= transient_gib <= 2.0, (
        f"B={JS_TF_MAX_BATCH} transient {transient_gib:.2f} GiB outside the safe band"
    )


def test_default_max_batch_would_have_ommed():
    """The upstream default (max_batch=16) transient is ~5.0 GiB — matching the
    5.35 GiB allocation that OOM'd in r4. This documents WHY the fix lowers it:
    the default does not fit the ~9.5 GiB headroom once intermediate tensors and
    KV-cache fragmentation are accounted for."""
    default_gib = 16 * _SEQ_LEN * _QWEN_VOCAB * _BF16_BYTES / _GIB
    assert 4.5 <= default_gib <= 5.5, f"default-16 transient {default_gib:.2f} GiB"
    # The fix is a >3x reduction.
    fixed_gib = JS_TF_MAX_BATCH * _SEQ_LEN * _QWEN_VOCAB * _BF16_BYTES / _GIB
    assert default_gib / fixed_gib >= 3.0
