"""Issue #763 `neutral-contrast-and-cofit` crash-fix r4 device-threading pins.

Production crash (pod-763, commit cda5d0e851, `epm:failure v8`
`cofit-labels-device-mismatch`): ``kernel_loco_preds`` built its label/selection
tensors on a ``device`` KWARG defaulting "cpu" that no caller passed, while the
``LayerCache`` kernels lived on cuda:0 (``EPM_FIT_DEVICE=cuda``) —
``Q.t() @ Ytr`` raised a cross-device RuntimeError at battery time. The
start-up exactness gate passed because it built its own cache on the default
cpu device, never touching the cuda path.

The fix, pinned here (all pins FAIL pre-fix):

- ``kernel_loco_preds`` DERIVES its device from the cache (``cache.device``)
  and exposes NO ``device`` parameter a caller could desync;
- every torch factory call in the battery functions (``LayerCache.build`` +
  ``kernel_loco_preds``) threads ``device=`` (``*_like`` factories are exempt —
  they inherit from their prototype tensor);
- every ``torch.from_numpy`` in the battery functions is immediately moved
  ``.to(dev, ...)``;
- the gate accepts ``device=`` and builds its cache on it; the driver runs
  ``assert_cofit_matches_reference(device=FIT_DEVICE)`` so the battery path is
  exercised ON the lane device before any behavior is fit.

STATIC AST pins (no torch import) + light functional cpu pins. The serial
oracles (``_serial_rank_ridge_reference``) are deliberately cpu and OUT of
scope for the factory pins.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COFIT_SRC = REPO / "src" / "explore_persona_space" / "analysis" / "issue_763_cofit.py"
DRIVER_SRC = REPO / "scripts" / "issue763_cofit_predictors.py"

# torch factory calls that default to cpu unless device= is threaded. *_like
# factories inherit device from their prototype tensor and are exempt.
_DEVICE_CTORS = {
    "tensor",
    "empty",
    "zeros",
    "ones",
    "full",
    "eye",
    "arange",
    "linspace",
    "triu_indices",
    "rand",
    "randn",
    "randint",
}


def _norm(src: str) -> str:
    """Collapse whitespace so formatter line-wrapping cannot break a pin."""
    return re.sub(r"\s+", " ", src)


def _module_tree() -> ast.Module:
    return ast.parse(COFIT_SRC.read_text())


def _battery_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """The two battery-path tensor builders: LayerCache.build + kernel_loco_preds."""
    out: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LayerCache":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "build":
                    out["LayerCache.build"] = sub
        elif isinstance(node, ast.FunctionDef) and node.name == "kernel_loco_preds":
            out["kernel_loco_preds"] = node
    assert set(out) == {"LayerCache.build", "kernel_loco_preds"}, sorted(out)
    return out


# ── static pins: the device kwarg is GONE; the device derives from the cache ──


def test_kernel_loco_preds_has_no_device_parameter():
    fn = _battery_functions(_module_tree())["kernel_loco_preds"]
    params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    assert "device" not in params, (
        "kernel_loco_preds must NOT expose a device kwarg — the r4 crash was a "
        'default-"cpu" device kwarg no caller passed; the device derives from '
        "the cache (cache.device) so a caller can never desync it"
    )


def test_kernel_loco_preds_derives_device_from_cache():
    src = COFIT_SRC.read_text()
    seg = _norm(src[src.index("def kernel_loco_preds") : src.index("def direction_loco_preds")])
    assert "dev = cache.device" in seg, (
        "kernel_loco_preds must derive its device from the LayerCache "
        "(dev = cache.device) — the single construction-time threading point"
    )


def test_layer_cache_exposes_device_property():
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LayerCache":
            names = {sub.name for sub in node.body if isinstance(sub, ast.FunctionDef)}
            assert "device" in names, "LayerCache must expose the .device property"
            return
    raise AssertionError("LayerCache class not found")


# ── static pins: every battery tensor construction threads device= ───────────


def test_battery_torch_factories_thread_device():
    for name, fn in _battery_functions(_module_tree()).items():
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
                continue
            root = call.func.value
            if not (isinstance(root, ast.Name) and root.id == "torch"):
                continue
            if call.func.attr not in _DEVICE_CTORS:
                continue
            kwargs = {kw.arg for kw in call.keywords}
            assert "device" in kwargs, (
                f"{name}: torch.{call.func.attr}(...) at line {call.lineno} lacks device= — "
                "a default-cpu factory inside the battery math is the r4 crash class"
            )


def test_battery_from_numpy_tensors_move_to_dev():
    for name, fn in _battery_functions(_module_tree()).items():
        from_numpy_calls = [
            c
            for c in ast.walk(fn)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "from_numpy"
        ]
        to_calls = {
            id(c.func.value): c
            for c in ast.walk(fn)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "to"
        }
        assert from_numpy_calls, f"{name}: expected torch.from_numpy sites in the battery fn"
        for fnp in from_numpy_calls:
            to_call = to_calls.get(id(fnp))
            assert to_call is not None, (
                f"{name}: torch.from_numpy at line {fnp.lineno} is not immediately "
                "moved with .to(dev, ...) — it stays on cpu (r4 crash class)"
            )
            first = to_call.args[0] if to_call.args else None
            assert isinstance(first, ast.Name) and first.id == "dev", (
                f"{name}: .to(...) at line {to_call.lineno} must move to the derived "
                "`dev`, not a literal/other device"
            )


# ── static pins: the on-lane gate covers the battery path on FIT_DEVICE ──────


def test_gate_builds_cache_on_requested_device():
    src = COFIT_SRC.read_text()
    seg = _norm(src[src.index("def assert_cofit_matches_reference") :])
    assert "LayerCache.build(x, dim=5, device=device)" in seg, (
        "assert_cofit_matches_reference must build its cache on the requested "
        "device so kernel_loco_preds is exercised ON-LANE (gate-time, not "
        "battery-time, failure for a future device miss)"
    )


def test_driver_runs_gate_on_fit_device():
    src = _norm(DRIVER_SRC.read_text())
    assert "assert_cofit_matches_reference(device=FIT_DEVICE)" in src, (
        "the driver must pass FIT_DEVICE to the cofit exactness gate — a "
        "default-cpu gate passed while the cuda battery crashed (r4)"
    )


# ── functional cpu pins ───────────────────────────────────────────────────────


def _import_cofit():
    sys.path.insert(0, str(REPO / "src"))
    from explore_persona_space.analysis import issue_763_cofit as mod

    return mod


def test_kernel_loco_preds_signature_and_cpu_device_derivation():
    import numpy as np
    import torch

    mod = _import_cofit()
    assert "device" not in inspect.signature(mod.kernel_loco_preds).parameters
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 12))
    y = rng.standard_normal(8)
    R = mod.fold_rank_targets(np.stack([y, y]))
    cache = mod.LayerCache.build(x, dim=3)
    assert cache.device == torch.device("cpu")
    preds = mod.kernel_loco_preds(cache, R, kernel_labels=("linear",))
    assert preds.shape == (2, 8), preds.shape
    assert np.isfinite(preds).all()


def test_gate_accepts_device_and_reports_it():
    mod = _import_cofit()
    out = mod.assert_cofit_matches_reference(device="cpu")
    assert out["device"] == "cpu", out
    assert out["ridge_delta"] <= 1e-8 and out["direction_delta"] <= 1e-10, out
