"""Pin the v1 byte-identical contract for the metric-race cloud-retention flag.

``_mean_hidden_states(..., retain_per_sample_reps=False)`` (the default) MUST
reproduce v1's centroid path: the per-text mean ``(D,)`` per (layer, point).
``retain_per_sample_reps=True`` MUST return the per-text ``(N, D)`` cloud whose
mean over axis 0 EQUALS the default-False centroid bit-for-bit — proving the
cloud path only DROPS the final averaging and changes nothing else.

Also pins the #545 round-36 ARCHITECTURAL PIVOT: on a model exposing the
standard ``model.model.layers`` / ``embed_tokens`` decoder structure,
``_mean_hidden_states`` extracts via per-layer forward hooks and does NOT pass
``output_hidden_states=True`` (which materialized all L+1 layers and OOM'd the
clouds phase across 5 rounds). The hook path's reps EQUAL the full-tuple
fallback's reps bit-for-bit (same captured math; only the source of the hidden
states changes) — including the layer-0/embedding off-by-one.

CPU-only (tiny deterministic stub model + the real tokenizer); no GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from explore_persona_space.experiments.behavior_testbed_545 import predictors as P


def _proj_hidden_states(proj: dict, ids: list[int], d: int, layers):
    """Shared deterministic hidden-state math used by both stubs: per token, the
    running-mean of a fixed per-token projection (so last_token != mean_response,
    reps are non-constant, and the reduction is exactly reproducible)."""
    t = len(ids)
    max_layer = max(layers)
    hs = [torch.zeros((1, t, d)) for _ in range(max_layer + 1)]
    for layer in layers:
        emb = proj[layer][ids]  # (T, D)
        cum = np.cumsum(emb, axis=0) / (np.arange(1, t + 1)[:, None])
        hs[layer] = torch.tensor(cum[None, :, :], dtype=torch.float32)
    return hs


class _StubModel:
    """Deterministic ``output_hidden_states`` model with NO decoder-block
    structure → exercises the fallback (full-tuple) path."""

    def __init__(self, vocab: int, d: int, layers, seed: int = 7):
        rng = np.random.default_rng(seed)
        self._proj = {layer: rng.standard_normal((vocab, d)).astype(np.float32) for layer in layers}
        self._layers = layers
        self._d = d

    def __call__(self, input_ids=None, output_hidden_states=False, **_kw):
        ids = input_ids[0].tolist()
        hs = _proj_hidden_states(self._proj, ids, self._d, self._layers)

        class _Out:
            pass

        o = _Out()
        o.hidden_states = hs
        return o

    def eval(self):
        return self


class _Module:
    """Minimal forward-hook-capable module (mirrors nn.Module's hook contract
    closely enough for _mean_hidden_states): registers hooks and invokes them
    with (self, input, output) when called."""

    def __init__(self):
        self._hooks: list = []

    def register_forward_hook(self, fn):
        self._hooks.append(fn)

        class _Handle:
            def __init__(h, mod, f):
                h._mod, h._f = mod, f

            def remove(h):
                if h._f in h._mod._hooks:
                    h._mod._hooks.remove(h._f)

        return _Handle(self, fn)

    def _fire(self, output):
        for fn in list(self._hooks):
            fn(self, None, output)


class _Inner:
    def __init__(self, embed, blocks):
        self.embed_tokens = embed
        self.layers = blocks


class _HookStubModel:
    """Deterministic model with the standard ``model.model.layers`` /
    ``embed_tokens`` structure → exercises the HOOK path. The same projection
    math as _StubModel, but delivered through per-module forward hooks: the
    embedding module fires hs[0]; block k-1 fires hs[k]."""

    def __init__(self, vocab: int, d: int, layers, seed: int = 7):
        rng = np.random.default_rng(seed)
        self._proj = {layer: rng.standard_normal((vocab, d)).astype(np.float32) for layer in layers}
        self._layers = layers
        self._d = d
        max_layer = max(layers)
        self._embed = _Module()
        self._blocks = [_Module() for _ in range(max_layer)]  # blocks[k-1] -> hs[k]
        self.model = _Inner(self._embed, self._blocks)

    def __call__(self, input_ids=None, output_hidden_states=False, **_kw):
        # Hooks MUST be the source of truth on this path: if the function asks
        # for output_hidden_states the pivot regressed.
        assert not output_hidden_states, (
            "hook-capable model received output_hidden_states=True — the #545 "
            "round-36 pivot must extract via hooks, not the full tuple"
        )
        ids = input_ids[0].tolist()
        hs = _proj_hidden_states(self._proj, ids, self._d, self._layers)
        # Fire each module's hook with its block output. hs[0] = embed output;
        # hs[k] = block k-1 output. Block outputs are tuples (hidden, ...).
        self._embed._fire(hs[0])
        for k in range(1, len(self._blocks) + 1):
            self._blocks[k - 1]._fire((hs[k],))
        return None

    def eval(self):
        return self


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


def test_default_false_is_v1_centroid_and_cloud_mean_matches():
    tok = _tok()
    layers = P.GEOMETRY_LAYERS
    model = _StubModel(tok.vocab_size, 16, layers)
    texts = [
        "The assistant gives careful advice.",
        "Here is a structured answer in list form.",
        "I would hedge and defer on this question.",
    ]

    centroid = P._mean_hidden_states(model, tok, texts, "cpu", retain_per_sample_reps=False)
    cloud = P._mean_hidden_states(model, tok, texts, "cpu", retain_per_sample_reps=True)

    for layer in layers:
        for point in P.EXTRACTION_POINTS:
            c = centroid[layer][point]
            cl = cloud[layer][point]
            # shapes: centroid is (D,); cloud is (N, D) with N == len(texts).
            assert c.ndim == 1, (layer, point, c.shape)
            assert cl.shape == (len(texts), c.shape[0]), (layer, point, cl.shape)
            # The cloud mean over the text axis EQUALS the centroid bit-for-bit
            # (the cloud path only drops the final `/ n` averaging).
            assert torch.equal(cl.mean(dim=0), c), (layer, point)


def test_default_kwarg_is_false():
    import inspect

    sig = inspect.signature(P._mean_hidden_states)
    assert sig.parameters["retain_per_sample_reps"].default is False


def test_hook_path_equals_fallback_path_byte_identical():
    """The #545 round-36 pivot: the per-layer forward-hook extraction path
    (standard decoder structure) produces reps BIT-FOR-BIT identical to the
    full-tuple fallback path (non-standard model). Same captured math; the only
    difference is the SOURCE of the hidden states — hooks on 8 layers vs the
    materialized L+1 tuple. Pins the layer-0/embedding off-by-one: a naive hook
    on layers[layer] would capture hs[layer+1] and this would diverge."""
    tok = _tok()
    layers = P.GEOMETRY_LAYERS
    d = 16
    texts = [
        "The assistant gives careful advice.",
        "Here is a structured answer in list form.",
        "I would hedge and defer on this question.",
    ]
    # Same seed → same projection → the two stubs deliver identical hidden states
    # via different mechanisms (full tuple vs fired hooks).
    fallback_model = _StubModel(tok.vocab_size, d, layers, seed=11)
    hook_model = _HookStubModel(tok.vocab_size, d, layers, seed=11)

    for retain in (False, True):
        fb = P._mean_hidden_states(fallback_model, tok, texts, "cpu", retain_per_sample_reps=retain)
        hk = P._mean_hidden_states(hook_model, tok, texts, "cpu", retain_per_sample_reps=retain)
        assert set(fb) == set(hk) == set(layers), (set(fb), set(hk))
        for layer in layers:
            for point in P.EXTRACTION_POINTS:
                assert torch.equal(fb[layer][point], hk[layer][point]), (layer, point, retain)


def test_clouds_uses_per_layer_hooks_not_output_hidden_states():
    """AST-pin the architectural pivot: ``_mean_hidden_states`` registers
    per-layer forward hooks and its PRIMARY (standard-decoder) forward call does
    NOT pass ``output_hidden_states=True``. The ``output_hidden_states=True``
    string survives ONLY inside the explicitly-labeled fallback branch for
    non-standard models / the CPU stub. This is the fix for the 5-round OOM where
    materializing all L+1 layers grew HF resident 22→38 GiB."""
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(P._mean_hidden_states))
    tree = ast.parse(src)

    # 1. register_forward_hook IS used.
    has_hook = any(
        isinstance(node, ast.Attribute) and node.attr == "register_forward_hook"
        for node in ast.walk(tree)
    )
    assert has_hook, "_mean_hidden_states must use register_forward_hook (per-layer hooks)"

    # 2. At least one model(...) call passes output_hidden_states=False (the
    #    primary hook path); the only =True call is the labeled fallback.
    false_calls, true_calls = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "output_hidden_states" and isinstance(kw.value, ast.Constant):
                    if kw.value.value is True:
                        true_calls += 1
                    elif kw.value.value is False:
                        false_calls += 1
    assert false_calls >= 1, (
        "the primary hook-path forward must pass output_hidden_states=False "
        "(else it still materializes all L+1 layers and re-introduces the OOM)"
    )
    # The fallback may keep exactly one =True; never more (no stray full-tuple read).
    assert true_calls <= 1, (
        f"output_hidden_states=True appears {true_calls} times -- only the single "
        "labeled non-standard-model fallback may use it"
    )
