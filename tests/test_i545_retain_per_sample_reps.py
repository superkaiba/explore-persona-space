"""Pin the v1 byte-identical contract for the metric-race cloud-retention flag.

``_mean_hidden_states(..., retain_per_sample_reps=False)`` (the default) MUST
reproduce v1's centroid path: the per-text mean ``(D,)`` per (layer, point).
``retain_per_sample_reps=True`` MUST return the per-text ``(N, D)`` cloud whose
mean over axis 0 EQUALS the default-False centroid bit-for-bit — proving the
cloud path only DROPS the final averaging and changes nothing else.

CPU-only (tiny deterministic stub model + the real tokenizer); no GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from explore_persona_space.experiments.behavior_testbed_545 import predictors as P


class _StubModel:
    """Deterministic output_hidden_states: per token, the running-mean of a
    fixed per-token projection (so last_token != mean_response, reps are
    non-constant, and the reduction is exactly reproducible)."""

    def __init__(self, vocab: int, d: int, layers, seed: int = 7):
        rng = np.random.default_rng(seed)
        self._proj = {layer: rng.standard_normal((vocab, d)).astype(np.float32) for layer in layers}
        self._layers = layers
        self._d = d

    def __call__(self, input_ids=None, output_hidden_states=False, **_kw):
        ids = input_ids[0].tolist()
        t = len(ids)
        max_layer = max(self._layers)
        hs = [torch.zeros((1, t, self._d)) for _ in range(max_layer + 1)]
        for layer in self._layers:
            emb = self._proj[layer][ids]  # (T, D)
            cum = np.cumsum(emb, axis=0) / (np.arange(1, t + 1)[:, None])
            hs[layer] = torch.tensor(cum[None, :, :], dtype=torch.float32)

        class _Out:
            pass

        o = _Out()
        o.hidden_states = hs
        return o

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
