"""Issue #2203 — adapter for the paper's own capping engine (Fix D, BUG 1).

The 32B anchor must use the paper's `ActivationSteering` MAX-cap
(`(proj - τ).clamp(min=0)`, lowering the axis component DOWN to τ), NOT our
in-house MIN-floor (`caphook.apply_cap_op`, raising it UP). Feeding Lu et al.'s
released anti-assistant vectors + negative τ into our floor is the inverse
intervention (plan §2 BUG 1). This module delegates the cap MATH to the paper's
`_apply_cap` VERBATIM — the audit forbids hand-reimplementing the sign.

**Import path (stated deviation from plan §4.4 pseudocode).** The plan wrote
`from assistant_axis import load_capping_config, build_capping_steerer,
ActivationSteering`, but the package `__init__` imports `.axis → .pca →
plotly` (and sklearn), which are NOT installed on the pods — a bare
`import assistant_axis` raises `ModuleNotFoundError: plotly` (verified this
round). `steering.py` itself imports ONLY torch + typing, so we load it as a
STANDALONE module via `importlib.util.spec_from_file_location`. This consumes
the paper's cap math byte-verbatim without the plotting deps — the closest
faithful alternative. The `external/assistant-axis` dir is git-UNTRACKED on
every branch (plan §12); on the 32B pod it is delivered by the §9 pinned-SHA
bootstrap clone, never by the repo clone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ASSISTANT_AXIS_STEERING_MODULE = "issue2203_paper_engine_steering"
_cached_steering_module = None


def _find_repo_root_with(rel: str) -> Path:
    """Walk up from this file to the dir that CONTAINS ``rel``; assert it exists."""
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / rel).exists():
            return cand
    raise FileNotFoundError(
        f"could not locate a repo root containing {rel!r} above {here} — the §9 "
        "pinned-SHA bootstrap clone of safety-research/assistant-axis is missing "
        "(external/ is git-untracked on every branch, plan §12)"
    )


def _steering_py_path() -> Path:
    root = _find_repo_root_with("external/assistant-axis")
    p = root / "external" / "assistant-axis" / "assistant_axis" / "steering.py"
    assert p.exists(), f"paper engine steering.py absent at {p} (bootstrap clone incomplete)"
    return p


def load_paper_steering_module():
    """Load the paper's ``steering.py`` as a standalone module (torch+typing only).

    Cached after the first load. Avoids the package ``__init__``'s plotly/sklearn
    imports (the stated deviation above). Returns the module object exposing
    ``ActivationSteering`` / ``load_capping_config`` / ``build_capping_steerer``.
    """
    global _cached_steering_module
    if _cached_steering_module is not None:
        return _cached_steering_module
    path = _steering_py_path()
    spec = importlib.util.spec_from_file_location(_ASSISTANT_AXIS_STEERING_MODULE, str(path))
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ASSISTANT_AXIS_STEERING_MODULE] = module
    spec.loader.exec_module(module)
    for sym in ("ActivationSteering", "load_capping_config", "build_capping_steerer"):
        assert hasattr(module, sym), f"paper steering.py missing {sym!r}"
    _cached_steering_module = module
    return module


def load_capping_config(config_path: str):
    """The paper's ``load_capping_config`` (``torch.load(..., weights_only=False)``)."""
    return load_paper_steering_module().load_capping_config(config_path)


CAP_EXPERIMENT_ID = "layers_46:54-p0.25"


def anchor_all_token_steerer(model, capping_config, experiment_id: str = CAP_EXPERIMENT_ID):
    """Faithful all-token cap: the paper's ``build_capping_steerer`` verbatim.

    Returns an ``ActivationSteering`` context manager with
    ``intervention_type="capping"``, ``positions="all"``, per-layer
    ``cap_thresholds`` from the released config (paper §5.1.2 every-token cap).
    ``capping_config`` is the dict from :func:`load_capping_config`.
    """
    mod = load_paper_steering_module()
    return mod.build_capping_steerer(model, capping_config, experiment_id)


def build_prefill_context_end_steerer(
    model, capping_config, experiment_id: str = CAP_EXPERIMENT_ID
):
    """A context-end cap: the paper's cap math, fired ONLY at the last prefill position.

    Builds a throwaway ``build_capping_steerer`` parent to reuse the paper's OWN
    config extraction (vector-ref resolution, per-layer caps, layer indices),
    then re-instantiates a :class:`PrefillContextEndSteering` subclass from the
    parent's normalized fields. The subclass overrides
    ``_apply_layer_interventions`` to fire on the last real PREFILL position only
    (``positions="last"`` under left padding) and pass every decode step
    through — delegating the cap arithmetic to the parent's ``_apply_cap``
    VERBATIM. Returns the subclass instance (a context manager).
    """
    Subclass = _prefill_context_end_class()
    parent = anchor_all_token_steerer(model, capping_config, experiment_id)
    return Subclass(
        model,
        parent.steering_vectors,
        coefficients=parent.coefficients,
        layer_indices=parent.layer_indices,
        intervention_type="capping",
        positions="last",  # paper _apply_cap edits [:, -1, :] — the last real prefill token
        cap_thresholds=parent.cap_thresholds,
    )


_cached_prefill_class = None


def _prefill_context_end_class():
    """Build (once) the ``PrefillContextEndSteering`` subclass of the paper's engine.

    Deferred so the base class (loaded file-scoped from the bootstrap clone) is
    resolved lazily — the class body only exists after the paper module loads.
    """
    global _cached_prefill_class
    if _cached_prefill_class is not None:
        return _cached_prefill_class
    ActivationSteering = load_paper_steering_module().ActivationSteering

    class PrefillContextEndSteering(ActivationSteering):
        """32B context-position cap: paper ``_apply_cap`` verbatim, prefill-last only.

        A T>1 forward is the prefill — the paper's ``positions="last"`` cap edits
        ``[:, -1, :]``, which under ``generate_batch``'s LEFT padding is each
        row's last real context token (the context-end position). A T==1 forward
        is a decode step under the KV cache — passed through untouched, so the
        cap fires ONLY at the context-end prefill position. This is a
        position-RESTRICTION wrapper; the cap sign + math come entirely from the
        parent ``_apply_cap`` (steering.py:317).
        """

        prefill_only = True

        def _apply_layer_interventions(self, activations, layer_idx):
            tensor = activations[0] if isinstance(activations, (tuple, list)) else activations
            # T == 1 under the KV cache is a decode step — pass through untouched.
            if hasattr(tensor, "shape") and len(tensor.shape) >= 2 and tensor.shape[1] == 1:
                return activations
            return super()._apply_layer_interventions(activations, layer_idx)

    _cached_prefill_class = PrefillContextEndSteering
    return PrefillContextEndSteering
