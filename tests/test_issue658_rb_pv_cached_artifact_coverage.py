"""Round-2 regression: the PV r_B fit fails LOUD on a v0/E0 coverage gap.

Pins round-1 BLOCKER ``rb-pv-cached-artifact-coverage-unverified``. The fit
projects the reused HF v0(C) store onto the git E0 contexts; ``_v0_layer_matrix``
indexes ``v0_store["summaries"]["mean"][c][layer]`` BLIND. A context present in the
git E0 but missing from the cached v0 store crashed LATE with a bare KeyError (or,
worse, a missing E0 context silently shrank n). Plan §4.3 Step 3.5 requires a
fail-loud coverage diff BEFORE any projection.

The fix is ``assert_v0_e0_coverage``: it diffs the cached v0 store's
``context_ids`` x layer-set x ``summaries["mean"]`` keys vs the git E0 contexts and
raises a ``RuntimeError`` naming the gap before projecting.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_fit_module():
    spec = importlib.util.spec_from_file_location(
        "issue658_rb_pv_fit_under_test_cov", SCRIPTS / "issue658_rb_pv_fit.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue658_rb_pv_fit_under_test_cov"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_fit_module()

# capture layers the projection will index (0..4); needs n_layers > max(cap)=4.
CAP_LAYERS = [0, 1, 2, 3, 4]
N_LAYERS = 5
HIDDEN = 3
CONTEXTS = ["c0", "c1", "c2", "c3"]


def _layer_stack():
    """A per-context summary indexable by layer (list of (HIDDEN,) arrays)."""
    return [np.zeros(HIDDEN, dtype=np.float32) for _ in range(N_LAYERS)]


def _v0_store(context_ids, summ_keys):
    """v0 store: context_ids list + summaries['mean'] dict (keyed by ctx)."""
    return {
        "context_ids": list(context_ids),
        "summaries": {"mean": {c: _layer_stack() for c in summ_keys}},
    }


def _e0_table(ctx_with_e0, behavior="broad_em"):
    """E0 table in e0_target's shape: e0['e0'][ctx][col] = {'rate': float}."""
    return {"e0": {c: {behavior: {"rate": 0.5}} for c in ctx_with_e0}}


def test_missing_e0_context_in_v0_store_raises_before_projection():
    """An E0 context absent from the cached v0 summaries['mean'] -> RuntimeError."""
    # v0 store covers only c0,c1,c2 in summaries['mean']; context_ids matches it.
    v0 = _v0_store(["c0", "c1", "c2"], ["c0", "c1", "c2"])
    # ... but the git E0 also has c3 -> a coverage gap the projection would hit.
    e0 = _e0_table(["c0", "c1", "c2", "c3"])
    with pytest.raises(RuntimeError, match=r"(?i)coverage"):
        MOD.assert_v0_e0_coverage("betley", v0, e0, v0["context_ids"], CAP_LAYERS, ["broad_em"])


def test_context_id_not_in_summaries_keys_raises():
    """A context_ids entry absent from summaries['mean'] (store drift) -> RuntimeError."""
    # context_ids lists c3 but summaries['mean'] has no c3 entry.
    v0 = _v0_store(["c0", "c1", "c2", "c3"], ["c0", "c1", "c2"])
    e0 = _e0_table(["c0", "c1", "c2"])
    with pytest.raises(RuntimeError, match=r"(?i)coverage|inconsistent|absent"):
        MOD.assert_v0_e0_coverage("betley", v0, e0, v0["context_ids"], CAP_LAYERS, ["broad_em"])


def test_too_few_layers_raises():
    """A context whose v0 summary lacks the deepest capture layer -> RuntimeError."""
    v0 = _v0_store(CONTEXTS, CONTEXTS)
    # truncate c2's per-layer stack to 3 layers (cannot index capture-layer 4).
    v0["summaries"]["mean"]["c2"] = [np.zeros(HIDDEN, dtype=np.float32) for _ in range(3)]
    e0 = _e0_table(CONTEXTS)
    with pytest.raises(RuntimeError, match=r"(?i)layer"):
        MOD.assert_v0_e0_coverage("betley", v0, e0, v0["context_ids"], CAP_LAYERS, ["broad_em"])


def test_full_coverage_passes_silently():
    """When every E0 context is covered across all layers, no error is raised."""
    v0 = _v0_store(CONTEXTS, CONTEXTS)
    e0 = _e0_table(CONTEXTS)
    # should not raise
    MOD.assert_v0_e0_coverage("betley", v0, e0, v0["context_ids"], CAP_LAYERS, ["broad_em"])


def test_missing_summaries_mean_raises():
    """A v0 store without summaries['mean'] -> RuntimeError (not a late AttributeError)."""
    v0 = {"context_ids": CONTEXTS, "summaries": {}}
    e0 = _e0_table(CONTEXTS)
    with pytest.raises(RuntimeError, match=r"(?i)summaries"):
        MOD.assert_v0_e0_coverage("betley", v0, e0, CONTEXTS, CAP_LAYERS, ["broad_em"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
