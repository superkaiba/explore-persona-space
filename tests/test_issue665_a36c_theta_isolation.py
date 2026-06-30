"""Round-3 Blocker B regression — A3.6c θ0 variants run on PURE BASE WEIGHTS.

`PeftModel.from_pretrained(base, ...)` wraps `base` IN-PLACE, so the round-2
`model = base if variant in ("p_up","self_c0") else trained` selection ran the θ0
variants WITH the adapter attached — P↑ read base+adapter+c⁺ (not pure θ0+c⁺) and
self_c0 was not the identity null on θ0. The fix forwards everything through the
single wrapped `trained` model and toggles `disable_adapter()` per variant.

This test asserts the ISOLATION GUARANTEE (not the label): for `p_up`/`self_c0` the
generation forward runs with the adapter DISABLED; for `p_down`/`self_cp`/
`random_cv`/`norm_matched` it runs with the adapter ENABLED. A fake PeftModel-like
object records the adapter state observed at generate() time via its
`disable_adapter()` context manager.

Fails pre-fix (round 2 ran p_up on `base`, which the fake reports as ENABLED — and
never enters the disable_adapter context), passes post-fix. CPU-only: no torch
model load, no HF, no GPU.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


class _FakeLayer(torch.nn.Module):
    """A real nn.Module so register_forward_hook works; forward returns a tuple
    (hidden_states, ...) shaped like a transformer block output so the patch/read
    hooks in _one_patch can index it."""

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, hidden):
        return (hidden,)


class _FakeInner:
    def __init__(self, layers):
        self.layers = layers


class _FakeBaseModelModel:
    def __init__(self, layers):
        self.model = _FakeInner(layers)


class _FakeBaseModel:
    """trained.base_model.model.model.layers[layer] is the hook target in _one_patch."""

    def __init__(self, layers):
        self.model = _FakeBaseModelModel(layers)


class _FakeTrained:
    """A PeftModel-like object. `disable_adapter()` is a context manager that flips
    `adapter_enabled`; `generate()` records the adapter state it observed (so the test
    can assert which forward regime each variant ran under). `base_model` exposes the
    layer module for the hook registration."""

    def __init__(self, d=8, n_prompt=4):
        self.device = torch.device("cpu")
        self.adapter_enabled = True  # PEFT default after from_pretrained: adapter ON
        self._layer = _FakeLayer(d)
        self.base_model = _FakeBaseModel([self._layer] * 30)
        self.d = d
        self.n_prompt = n_prompt
        self.observed_states: list[bool] = []  # adapter state at each generate() call

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield self
        finally:
            self.adapter_enabled = prev

    def eval(self):
        return self

    def generate(self, ids, **kw):
        # record the adapter state THIS forward ran under (the isolation signal)
        self.observed_states.append(self.adapter_enabled)
        n_prompt = ids.shape[1]
        # drive the registered hooks: a prefill forward (holds the prompt slots) +
        # one generated-token forward, so _patch_hook + _read_hook both fire.
        prefill = torch.zeros((1, n_prompt, self.d))
        self._layer(prefill)  # prefill: shape[1] >= n_prompt -> patch hook fires
        gen_step = torch.zeros((1, 1, self.d))
        self._layer(gen_step)  # generated token: read hook collects v
        # return a (1, n_prompt + 1) id tensor so the decode slice works
        return torch.cat([ids, torch.zeros((1, 1), dtype=ids.dtype)], dim=1)


class _FakeTok:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "fake completion text"


def _run_one(variant: str) -> _FakeTrained:
    """Call _one_patch with the fake model for one variant; return the fake so the
    test can inspect the adapter state the generation forward ran under."""
    import issue665_patch_gpu as P

    trained = _FakeTrained(d=8, n_prompt=4)
    tok = _FakeTok()
    ids = torch.zeros((1, 4), dtype=torch.long)
    # c0 / cp are torch tensors (the c_C_base / c_C_trained slices); norm_matched
    # reads .numpy() on them, random_cv reads c_base_all.
    c0 = torch.ones(8) * 1.0
    cp = torch.ones(8) * 2.0
    c_base_all = torch.ones((6, 30, 8))
    rng = np.random.default_rng(0)
    # base is passed but must be UNUSED as a forward target post-fix (the fix forwards
    # through `trained` only); pass the same fake so an accidental `base.generate` would
    # also be recorded on a DIFFERENT object and fail the assertion below.
    base = _FakeTrained(d=8, n_prompt=4)
    P._one_patch(
        tok,
        base,
        trained,
        ids,
        layer=5,
        scope="last",
        variant=variant,
        c0=c0,
        cp=cp,
        rng=rng,
        c_base_all=c_base_all,
    )
    # the fix must never forward through `base`
    assert base.observed_states == [], (
        f"variant {variant}: forward ran through `base`, not the wrapped `trained` "
        "(round-2 adapter-leak path)"
    )
    return trained


def test_theta0_variants_run_adapter_disabled():
    """p_up and self_c0 (the θ0 variants) MUST forward with the adapter DISABLED."""
    for variant in ("p_up", "self_c0"):
        trained = _run_one(variant)
        assert trained.observed_states == [False], (
            f"θ0 variant {variant} must run the generation forward with the adapter "
            f"DISABLED (pure base weights); observed {trained.observed_states}"
        )
        # the context manager restored the prior state after the forward
        assert trained.adapter_enabled is True, "disable_adapter must restore state"


def test_thetaplus_variants_run_adapter_enabled():
    """p_down / self_cp / random_cv / norm_matched (θ+ variants) forward with the
    adapter ENABLED (base+adapter)."""
    for variant in ("p_down", "self_cp", "random_cv", "norm_matched"):
        trained = _run_one(variant)
        assert trained.observed_states == [True], (
            f"θ+ variant {variant} must run the generation forward with the adapter "
            f"ENABLED (base+adapter); observed {trained.observed_states}"
        )


def test_self_c0_is_identity_null_on_theta0():
    """self_c0 patches c0 onto θ0 — it must be BOTH adapter-disabled AND patch the
    base context vector (the identity null the A3.6c falsifiability predicate needs)."""
    trained = _run_one("self_c0")
    assert trained.observed_states == [False], "self_c0 must run on pure θ0"
