"""Regression pins for the #1332 map-similarity pipeline (r1 code-review fixes).

1. Every ``verify_repo_paths_uploaded`` call site in the gpu-phase dispatcher
   BINDS against the helper's live signature (r1 Critical 2: the smoke-fenced
   call omitted the ``api`` positional + REQUIRED kw-only ``path_in_repo`` ->
   TypeError at the terminal upload stage of every production run).
2. ``check_545_inputs`` probes the NESTED ``corpora/demos`` Hub prefix, never
   the flat ``demos`` prefix that 404s (r1 Critical 1).
3. ``ridge_fit_predict_fast_layer_batched`` numerically matches the canonical
   ``ridge_fit_predict`` per layer slice (r1 Minor: the src/ addition had no
   pytest pin; the runtime parity gate is production-only).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GPU_PHASE = REPO / "scripts" / "issue1332_gpu_phase.py"
BANK_BUILD = REPO / "scripts" / "issue1332_bank_build.py"


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == name) or (
                isinstance(f, ast.Attribute) and f.attr == name
            ):
                out.append(node)
    return out


def test_verify_repo_paths_uploaded_call_shapes_bind():
    """Every call site's arg shape binds against the helper's live signature."""
    from explore_persona_space.orchestrate.hub import verify_repo_paths_uploaded

    tree = ast.parse(GPU_PHASE.read_text())
    calls = _calls_named(tree, "verify_repo_paths_uploaded")
    assert calls, "expected >=1 verify_repo_paths_uploaded call in issue1332_gpu_phase.py"
    sig = inspect.signature(verify_repo_paths_uploaded)
    for call in calls:
        assert not any(isinstance(a, ast.Starred) for a in call.args), "starred args unbindable"
        args = [object()] * len(call.args)
        kwargs = {kw.arg: object() for kw in call.keywords if kw.arg is not None}
        # raises TypeError on the pre-fix shape (missing api + path_in_repo)
        sig.bind(*args, **kwargs)


def test_check_545_inputs_probes_nested_demos_prefix():
    """The demos existence probe targets corpora/demos (flat demos/ 404s)."""
    src = BANK_BUILD.read_text()
    assert 'f"{C.I545_HF_PREFIX}/corpora/demos"' in src
    assert 'f"{C.I545_HF_PREFIX}/demos"' not in src


def test_layer_batched_wrapper_matches_canonical_ridge():
    """Batched wrapper reproduces the canonical per-slice solve to <1e-8 rel."""
    import numpy as np

    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict,
        ridge_fit_predict_fast_layer_batched,
    )

    rng = np.random.default_rng(0)
    n_layers, n_tr, n_ev, d, d_out = 3, 40, 12, 8, 5
    X_tr = rng.standard_normal((n_layers, n_tr, d))
    W = rng.standard_normal((n_layers, d, d_out))
    Y_tr = X_tr @ W + 0.1 * rng.standard_normal((n_layers, n_tr, d_out))
    X_ev = rng.standard_normal((n_layers, n_ev, d))

    batched = ridge_fit_predict_fast_layer_batched(X_tr, Y_tr, X_ev)
    assert batched.shape == (n_layers, n_ev, d_out), batched.shape
    for li in range(n_layers):
        ref = ridge_fit_predict(X_tr[li], Y_tr[li], X_ev[li])
        scale = float(np.abs(ref).max()) + 1e-12
        rel = float(np.abs(batched[li] - ref).max()) / scale
        assert rel < 1e-8, f"layer {li}: rel diff {rel:.3e} vs canonical"

    preds_w, weights = ridge_fit_predict_fast_layer_batched(X_tr, Y_tr, X_ev, return_weights=True)
    assert weights.shape == (n_layers, d, d_out), weights.shape
    np.testing.assert_allclose(preds_w, batched, rtol=0, atol=1e-12)
