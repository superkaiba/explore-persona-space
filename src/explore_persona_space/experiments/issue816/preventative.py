"""Exp-4 preventative (training-time) steering (arXiv 2507.21509 ``sec:steering`` §5.2).

Steers the model TOWARD the persona direction DURING finetuning by adding
``coef * v`` to block ``layer_idx``'s output on EVERY training forward pass —
"cancelling out" the pressure the trait-inducing data applies along that
direction. Ported from ``training.py::steering_intervention`` +
``add_steering_hooks`` / ``remove_steering_hooks`` @ ``b8e0f04``.

MAGNITUDE CONVENTION (load-bearing — see the implementer report + plan §11
fact-checker note): the paper's ``load_steering_vectors`` ``steer`` branch
(``training.py`` line 158) loads ``vector = loaded_data[layer].unsqueeze(0)`` —
the RAW response-avg diff, NOT unit-normalized — and ``steering_intervention``
adds ``act + steering_coef * Q``. So the faithful port of the paper's CODE uses
the RAW vector (same convention as Exp-2's generation-time hook). (The plan §11
fact-checker note claims the training hook UNIT-NORMALIZES; that describes the
``ablate`` branch at line 155, NOT the ``steer`` branch. The reference code is
authoritative for a faithful port, so RAW is the default here.) The convention
is exposed as ``normalize`` (default ``False`` = raw = faithful-to-code); either
way the RANDOM directions inherit the SAME convention as the real vector so
real-vs-random is magnitude-matched WITHIN Exp-4. Exp-2 and Exp-4 coefficient
axes are NOT directly comparable if ``normalize`` differs between them (they
differ by a factor of ‖r_B[layer]‖); both default to RAW here.

The hook is attached via a ``TrainerCallback`` (``on_train_begin`` registers,
``on_train_end`` removes), which is passed to the shared ``train_lora()`` through
its ``callbacks=`` argument. This reproduces the paper's
``add_steering_hooks``-before-train / ``remove_steering_hooks``-after-train
lifecycle without modifying the 400-line ``train_lora`` internals. The callback
resolves the PEFT-wrapped module path at ``on_train_begin`` (the model is
``base_model.model.model.layers`` under a PeftModel — the paper's
``add_steering_hooks`` handles the same rewrite, ``training.py`` lines 82-94).
"""

from __future__ import annotations

import logging

from transformers import TrainerCallback

logger = logging.getLogger("issue816.preventative")

DEFAULT_LAYER = 20  # paper's 1-indexed steering layer; hook at layer_idx = layer - 1


def _steering_forward_hook(module, ins, out, *, add_vec):
    """Add ``add_vec`` (already coef-scaled + dtype/device-matched) to the block output.

    Faithful to ``steering_intervention`` (b8e0f04): broadcasts over the
    (batch, seq) dims — every position of every training forward is shifted,
    matching the paper's training-time steering (NOT position-gated, unlike the
    generation-time ``ActivationSteerer`` which gates to the last position).
    """
    import torch

    act = out[0] if isinstance(out, tuple) else out
    act = act + add_vec.to(act.device, act.dtype)
    if isinstance(out, tuple):
        return (act, *out[1:])
    if torch.is_tensor(out):
        return act
    return out


def _resolve_layer_module(model, layer_idx: int):
    """Locate block ``layer_idx`` on a (possibly PEFT-wrapped) Qwen model.

    Tries the paper's ``add_steering_hooks`` path family: the PeftModel wrapper
    nests the base model under ``base_model.model``, so the transformer blocks
    live at ``base_model.model.model.layers`` (Qwen). Falls back to the bare
    ``model.model.layers`` / ``model.layers`` for a non-PEFT model. Fails loud.
    """
    for path in (
        "base_model.model.model.layers",  # PeftModel(AutoModelForCausalLM(Qwen))
        "model.model.layers",  # bare Qwen (model.model is the QwenModel)
        "model.layers",
        "base_model.model.layers",
    ):
        cur = model
        ok = True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__getitem__"):
            if not (-len(cur) <= layer_idx < len(cur)):
                raise IndexError(f"layer_idx {layer_idx} out of range for {len(cur)} layers")
            logger.info("[preventative] resolved steering module at %s[%d]", path, layer_idx)
            return cur[layer_idx]
    raise ValueError(
        "could not locate the transformer layer list for the preventative hook "
        "(tried base_model.model.model.layers / model.model.layers / model.layers)"
    )


class PreventativeSteeringCallback(TrainerCallback):
    """A ``transformers.TrainerCallback`` that steers TOWARD ``vector`` during training.

    Registers a forward hook on block ``layer - 1`` at ``on_train_begin`` and
    removes it at ``on_train_end`` (the paper's ``add_steering_hooks`` /
    ``remove_steering_hooks`` lifecycle). Instantiated per-cell and passed to
    ``train_lora(..., callbacks=[cb])``.

    Args:
        vector: 1-D steering direction ``r_B[layer - 1]`` (real) or a norm-matched
            random draw. RAW magnitude by default (``normalize=False``).
        coef: steering coefficient (the paper's ``steering_coef``).
        layer: 1-indexed block (default 20 == the paper's steering layer).
        normalize: if True, unit-normalize ``vector`` before scaling by ``coef``
            (the ``ablate``-branch convention). Default False (faithful to the
            paper's ``steer`` branch — RAW). Random dirs MUST use the same value.
    """

    def __init__(self, vector, *, coef: float, layer: int = DEFAULT_LAYER, normalize: bool = False):
        import torch

        v = torch.as_tensor(vector)
        if v.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {tuple(v.shape)}")
        if normalize:
            n = torch.linalg.norm(v)
            if n == 0:
                raise ValueError("cannot normalize a zero vector")
            v = v / n
        self._base_vector = v.detach().clone()
        self.coef = float(coef)
        self.layer = int(layer)
        self.normalize = bool(normalize)
        self._handle = None

    def on_train_begin(self, args, state, control, **kwargs):
        import functools

        model = kwargs.get("model")
        if model is None:
            raise RuntimeError(
                "PreventativeSteeringCallback.on_train_begin got no model kwarg; "
                "cannot attach the steering hook (fail loud rather than silently no-op)"
            )
        module = _resolve_layer_module(model, self.layer - 1)
        p = next(model.parameters())
        add_vec = (self.coef * self._base_vector).to(p.device, p.dtype)
        hook = functools.partial(_steering_forward_hook, add_vec=add_vec)
        self._handle = module.register_forward_hook(hook)
        logger.info(
            "[preventative] hook attached: layer=%d coef=%.4g normalize=%s |vec|=%.4g |add|=%.4g",
            self.layer,
            self.coef,
            self.normalize,
            float(self._base_vector.norm()),
            float((self.coef * self._base_vector).norm()),
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            logger.info("[preventative] hook removed after training")
        return control
