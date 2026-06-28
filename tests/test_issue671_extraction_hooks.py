"""Pin the #671 memory-safe activation-extraction contract.

The shared helper ``analysis/extraction.py::extract_layer_activations`` replaces
``model(ids, output_hidden_states=True)`` subset reads — which materialize ALL
``L+1`` residual-stream tensors per forward (the #545/#667 accumulation bug) —
with ``register_forward_hook`` on only the needed modules +
``output_hidden_states=False``. This test file pins:

1. **Byte-identity** of the hook path vs the full-tuple read, across the
   ``hs[L+1]`` block-index convention AND the ``EMBED_LAYER`` (-1 -> hs[0])
   sentinel, on a hook-capable stub vs a fallback stub seeded identically.
2. **The i488 index translation** (``hs[L]`` convention -> block ``L-1`` /
   embed for ``L=0``) reproduces the OLD ``_last_token_residuals`` read.
3. **Logits preservation** under ``return_logits=True`` (the issue_650
   dual-purpose forward), including an end-to-end four-float marker read.
4. **The AST regression-lock**: the helper source AND each migrated read
   function's source register forward hooks and do NOT pass
   ``output_hidden_states=True`` on the primary path (a future re-introduction
   trips it). Each migrated function is enumerated explicitly.
5. **CPU memory-non-growth proxy** (advisory; the real GPU-segment retention
   is not observable CPU-side — see the module docstring + the filed real-GPU
   follow-up).

CPU-only (tiny deterministic stub models + the real tokenizer); no GPU. Mirrors
``tests/test_i545_retain_per_sample_reps.py``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import torch

from explore_persona_space.analysis.extraction import EMBED_LAYER, extract_layer_activations

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic stub hidden-state math (shared by both stubs)
# ─────────────────────────────────────────────────────────────────────────────


def _proj_hidden_states(proj: dict, ids: list[int], d: int, n_blocks: int):
    """Per-token running-mean of a fixed per-(token,layer) projection.

    Returns ``hs`` = a list of length ``n_blocks + 1`` of ``(1, T, D)`` tensors:
    ``hs[0]`` is the embedding output, ``hs[k]`` (k>=1) is block ``k-1``'s
    output. Non-constant across positions so ``last_token != mean_response`` and
    the reduction is exactly reproducible.
    """
    t = len(ids)
    hs = []
    for k in range(n_blocks + 1):
        emb = proj[k][ids]  # (T, D)
        cum = np.cumsum(emb, axis=0) / (np.arange(1, t + 1)[:, None])
        hs.append(torch.tensor(cum[None, :, :], dtype=torch.float32))
    return hs


_V_OUT = 64  # small synthetic logit width (>= the marker/eos ids the tests index);
# NOT the real vocab — a (vocab, vocab) logit matrix would be ~92 GB and hang.


def _proj_logits(proj_logit: np.ndarray, ids: list[int]):
    """Deterministic ``(1, T, V_OUT)`` logits from a fixed per-token projection.

    ``proj_logit`` is ``(vocab, _V_OUT)`` — indexed by token id on the rows, with
    a small synthetic output width (the marker/eos ids the tests read are < 64).
    """
    rows = proj_logit[ids]  # (T, _V_OUT)
    return torch.tensor(rows[None, :, :], dtype=torch.float32)


class _Out:
    pass


class _FallbackStub:
    """``output_hidden_states``-only model with NO decoder-block structure ->
    exercises the helper's full-tuple fallback path. Always returns the full
    ``hidden_states`` tuple (+ logits)."""

    def __init__(self, vocab: int, d: int, n_blocks: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        self._proj = {
            k: rng.standard_normal((vocab, d)).astype(np.float32) for k in range(n_blocks + 1)
        }
        self._proj_logit = rng.standard_normal((vocab, _V_OUT)).astype(np.float32)
        self._n_blocks = n_blocks
        self._d = d

    def __call__(self, input_ids=None, output_hidden_states=False, attention_mask=None, **_kw):
        ids = input_ids[0].tolist()
        o = _Out()
        o.hidden_states = _proj_hidden_states(self._proj, ids, self._d, self._n_blocks)
        o.logits = _proj_logits(self._proj_logit, ids)
        return o

    def eval(self):
        return self


class _Module:
    """Minimal forward-hook-capable module (mirrors nn.Module's hook contract
    closely enough): registers hooks and fires them with (self, input, output)."""

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


class _HookStub:
    """Model with the standard ``model.model.layers`` / ``embed_tokens``
    structure -> exercises the helper's HOOK path. Same projection math as
    ``_FallbackStub``, delivered through per-module forward hooks: the embedding
    module fires hs[0]; block k-1 fires hs[k] (as a tuple, exercising the
    unwrap). Returns an object carrying ``.logits`` so ``return_logits=True``
    works on this path too."""

    def __init__(self, vocab: int, d: int, n_blocks: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        self._proj = {
            k: rng.standard_normal((vocab, d)).astype(np.float32) for k in range(n_blocks + 1)
        }
        self._proj_logit = rng.standard_normal((vocab, _V_OUT)).astype(np.float32)
        self._n_blocks = n_blocks
        self._d = d
        self._embed = _Module()
        self._blocks = [_Module() for _ in range(n_blocks)]  # blocks[k-1] -> hs[k]
        self.model = _Inner(self._embed, self._blocks)

    def __call__(self, input_ids=None, output_hidden_states=False, attention_mask=None, **_kw):
        # Hooks MUST be the source of truth on this path: if the helper asks for
        # output_hidden_states=True the #545/#667 fix regressed (W2 — mirrors
        # tests/test_i545_retain_per_sample_reps.py:118-121). This runtime teeth
        # complements the AST regression-lock below.
        assert not output_hidden_states, (
            "hook-capable model received output_hidden_states=True — the #671 "
            "fix must extract via hooks, not the full tuple"
        )
        ids = input_ids[0].tolist()
        hs = _proj_hidden_states(self._proj, ids, self._d, self._n_blocks)
        # Fire each module's hook with its block output. hs[0] = embed output
        # (bare tensor); hs[k] = block k-1 output (tuple, exercises the unwrap).
        self._embed._fire(hs[0])
        for k in range(1, self._n_blocks + 1):
            self._blocks[k - 1]._fire((hs[k],))
        o = _Out()
        o.logits = _proj_logits(self._proj_logit, ids)
        return o

    def eval(self):
        return self


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


_N_BLOCKS = 28  # mirror Qwen-2.5-7B's block count so block indices 7/14/21 are valid
_D = 16
_TEXTS = [
    "The assistant gives careful advice.",
    "Here is a structured answer in list form, with several distinct items.",
    "I would hedge and defer on this question entirely.",
]


def _ids_list(tok):
    return [
        torch.tensor([tok.encode(t, add_special_tokens=False)], dtype=torch.long) for t in _TEXTS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Byte-identity: hook path == full-tuple read, hs[L+1] + EMBED_LAYER
# ─────────────────────────────────────────────────────────────────────────────


def test_hook_equals_fulltuple_byte_identical():
    """For each requested layer (incl. EMBED_LAYER) and each of >=3 inputs of
    differing length, the hook-path captured tensor EQUALS the full-tuple
    fallback's tensor bit-for-bit. Same seed -> identical hidden states via
    different mechanisms. Pins the layer-0/embedding off-by-one: a naive hook on
    layers[L] capturing hs[L+1] would diverge on the EMBED_LAYER case."""
    tok = _tok()
    layers = [EMBED_LAYER, 0, 7, 14, 21, 27]
    fallback = _FallbackStub(tok.vocab_size, _D, _N_BLOCKS, seed=11)
    hook = _HookStub(tok.vocab_size, _D, _N_BLOCKS, seed=11)

    for ids in _ids_list(tok):
        fb = extract_layer_activations(fallback, ids, layers)
        hk = extract_layer_activations(hook, ids, layers)
        assert set(fb) == set(hk) == set(layers), (set(fb), set(hk))
        for L in layers:
            assert torch.equal(fb[L], hk[L]), (L, ids.shape)


def test_embed_layer_maps_to_hs0():
    """``extract_layer_activations(stub, ids, [EMBED_LAYER])[EMBED_LAYER]`` equals
    the stub's hs[0] (the embedding output), on BOTH the hook and fallback
    paths."""
    tok = _tok()
    fallback = _FallbackStub(tok.vocab_size, _D, _N_BLOCKS, seed=3)
    hook = _HookStub(tok.vocab_size, _D, _N_BLOCKS, seed=3)
    ids = _ids_list(tok)[0]

    # Ground truth: the stub's own hs[0].
    expected = _proj_hidden_states(fallback._proj, ids[0].tolist(), _D, _N_BLOCKS)[0]
    fb = extract_layer_activations(fallback, ids, [EMBED_LAYER])
    hk = extract_layer_activations(hook, ids, [EMBED_LAYER])
    assert torch.equal(fb[EMBED_LAYER], expected)
    assert torch.equal(hk[EMBED_LAYER], expected)


# ─────────────────────────────────────────────────────────────────────────────
# 2. i488 index translation (hs[L] convention) byte-identity
# ─────────────────────────────────────────────────────────────────────────────


def _old_last_token_residuals_via_fulltuple(model, ids, layers):
    """The PRE-migration ``_last_token_residuals`` math: read
    ``outputs.hidden_states[L]`` directly (hs[L] convention: L=0 embedding,
    L=k = hs[k]) at the last input position. Reproduced here against the
    fallback stub so the test pins the translated helper call against the exact
    old behavior."""
    out = model(input_ids=ids, output_hidden_states=True)
    last_pos = ids.shape[1] - 1
    return {L: out.hidden_states[L][0, last_pos, :].float().cpu() for L in layers}


def test_i488_index_translation_byte_identical():
    """The translated helper call (``hs[L]`` -> block ``L-1`` / embed for L=0)
    reproduces the OLD ``_last_token_residuals`` output bit-for-bit, including
    the L=0 embedding case. The translation is the migration's correctness
    crux."""
    tok = _tok()
    # i488's layers are hs-INDICES (0 = embedding, k = hs[k]).
    hs_layers = (0, 8, 15, 22, 28)
    fallback = _FallbackStub(tok.vocab_size, _D, _N_BLOCKS, seed=5)
    hook = _HookStub(tok.vocab_size, _D, _N_BLOCKS, seed=5)
    ids = _ids_list(tok)[1]

    old = _old_last_token_residuals_via_fulltuple(fallback, ids, hs_layers)
    last_pos = ids.shape[1] - 1

    # The migrated call shape: translate hs-index L -> block index.
    req = [EMBED_LAYER if L == 0 else L - 1 for L in hs_layers]
    for model in (fallback, hook):
        acts = extract_layer_activations(model, ids, req)
        for L in hs_layers:
            key = EMBED_LAYER if L == 0 else L - 1
            new = acts[key][0, last_pos, :].float().cpu()
            assert torch.equal(old[L], new), (L, key, model.__class__.__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 2b. _context_vector_all_layers all-layers last-token byte-identity (#675)
# ─────────────────────────────────────────────────────────────────────────────


def _old_context_vector_all_layers_via_fulltuple(model, ids, n_layers):
    """The PRE-migration _context_vector_all_layers math: full-tuple read of
    hs[1..n_layers] at the LAST input position, dropping the embedding hs[0].
    Reproduced against the stub so the test pins the migrated call against the
    exact old behavior. `ids` is a plain (1, T) tensor (the stub's input)."""
    out = model(input_ids=ids, output_hidden_states=True)
    return np.stack(
        [out.hidden_states[li][0, -1, :].float().cpu().numpy() for li in range(1, n_layers + 1)]
    ).astype(np.float32)


def test_context_vector_all_layers_byte_identical():
    """The migrated all-layers context-vector read (block indices 0..N-1, last
    input token) reproduces the OLD hs[1..N] full-tuple read bit-for-bit on
    BOTH the hook stub and the fallback stub. Pins the #675 residual: the
    embedding-drop off-by-one (request blocks 0..N-1, NOT hs[0..N-1]) and the
    last-position read.

    Also pins the dropped-attention_mask no-op invariant (the Methodology
    reconciler's standing rec): for this single-unpadded-sequence call shape
    the helper output is invariant to attention_mask (None vs all-ones), so the
    production refactor's omission of the mask is exact, not approximate."""
    tok = _tok()
    fallback = _FallbackStub(tok.vocab_size, _D, _N_BLOCKS, seed=17)
    hook = _HookStub(tok.vocab_size, _D, _N_BLOCKS, seed=17)
    for ids in _ids_list(tok):
        old = _old_context_vector_all_layers_via_fulltuple(fallback, ids, _N_BLOCKS)
        for model in (fallback, hook):
            acts = extract_layer_activations(model, ids, list(range(_N_BLOCKS)))
            new = np.stack(
                [acts[li][0, -1, :].float().cpu().numpy() for li in range(_N_BLOCKS)]
            ).astype(np.float32)
            assert new.shape == (_N_BLOCKS, _D), new.shape
            assert np.array_equal(old, new), model.__class__.__name__

        # attention_mask no-op invariant (the Methodology reconciler's standing
        # rec): the production call OMITS the mask. For a single unpadded B=1
        # sequence the all-ones mask is a no-op, so passing it must yield the
        # byte-identical result. Pinned on the hook stub — the standard-decoder
        # production path that _context_vector_all_layers exercises.
        acts_nomask = extract_layer_activations(hook, ids, list(range(_N_BLOCKS)))
        acts_ones = extract_layer_activations(
            hook, ids, list(range(_N_BLOCKS)), attention_mask=torch.ones_like(ids)
        )
        for li in range(_N_BLOCKS):
            assert torch.equal(acts_ones[li], acts_nomask[li]), (li, ids.shape)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Logits preservation under return_logits=True (issue_650 dual-purpose)
# ─────────────────────────────────────────────────────────────────────────────


def test_logits_preserved_with_return_logits():
    """``return_logits=True`` returns ``(acts, logits)`` where ``logits`` equals
    the stub's logits bit-for-bit, on BOTH paths. The acts are unchanged from
    the no-logits call."""
    tok = _tok()
    layers = [21]
    fallback = _FallbackStub(tok.vocab_size, _D, _N_BLOCKS, seed=9)
    hook = _HookStub(tok.vocab_size, _D, _N_BLOCKS, seed=9)
    ids = _ids_list(tok)[0]

    expected_logits = _proj_logits(fallback._proj_logit, ids[0].tolist())

    for model in (fallback, hook):
        acts_only = extract_layer_activations(model, ids, layers)
        acts, logits = extract_layer_activations(model, ids, layers, return_logits=True)
        assert torch.equal(logits, expected_logits), model.__class__.__name__
        # acts unchanged whether or not logits were requested.
        assert torch.equal(acts[21], acts_only[21]), model.__class__.__name__


def test_issue650_four_float_marker_read_end_to_end():
    """End-to-end issue_650-style read off the SAME forward: the hidden-state
    shift AND a four-float marker slot read (z_marker, z_eos, logZ, logp_marker)
    are computed from ``return_logits=True`` output and match the OLD full-tuple
    path bit-for-bit.

    Mirrors the issue_650 logit reads at shift_extract.py:291-343 (read at
    ``slot - 1``). This pins the logit-rebinding completeness (A2): a missed
    rebind would surface as a NameError / stale-reference here, which the
    byte-identity-of-the-activation check alone would NOT catch."""
    tok = _tok()
    fallback = _FallbackStub(tok.vocab_size, _D, _N_BLOCKS, seed=13)
    hook = _HookStub(tok.vocab_size, _D, _N_BLOCKS, seed=13)
    ids = _ids_list(tok)[2]
    slot = ids.shape[1] - 1  # a valid response slot; read logits at slot - 1
    marker_id = 17
    eos_id = 23

    # OLD path (full tuple): the layer-internal index hs[k]; issue_650 uses
    # layer_idx_internal == extraction_layer + 1, so block index L = it - 1.
    layer_idx_internal = 22
    L = layer_idx_internal - 1
    out_old = fallback(input_ids=ids, output_hidden_states=True)
    hs_old = out_old.hidden_states[layer_idx_internal][0, slot].float().cpu()
    logits_old = out_old.logits[0, slot - 1].float()
    logp_old = torch.log_softmax(logits_old, dim=-1)
    z_marker_old = float(logits_old[marker_id].item())
    z_eos_old = float(logits_old[eos_id].item())
    logZ_old = float(torch.logsumexp(logits_old, dim=-1).item())
    logp_marker_old = float(logp_old[marker_id].item())

    # NEW path (helper, return_logits=True), on both stubs.
    for model in (fallback, hook):
        acts, logits_full = extract_layer_activations(model, ids, [L], return_logits=True)
        hs_new = acts[L][0, slot].float().cpu()
        # Logit reads MUST rebind to logits_full (the A2 rebind-completeness check).
        logits_row = logits_full[0, slot - 1].float()
        logp_new = torch.log_softmax(logits_row, dim=-1)
        z_marker_new = float(logits_row[marker_id].item())
        z_eos_new = float(logits_row[eos_id].item())
        logZ_new = float(torch.logsumexp(logits_row, dim=-1).item())
        logp_marker_new = float(logp_new[marker_id].item())

        assert torch.equal(hs_old, hs_new), model.__class__.__name__
        assert z_marker_old == z_marker_new, model.__class__.__name__
        assert z_eos_old == z_eos_new, model.__class__.__name__
        assert logZ_old == logZ_new, model.__class__.__name__
        assert logp_marker_old == logp_marker_new, model.__class__.__name__


# ─────────────────────────────────────────────────────────────────────────────
# 4. AST regression-lock — enumerate EACH migrated function (A3)
# ─────────────────────────────────────────────────────────────────────────────

# (module_path-agnostic) (importable_object, attribute) per migrated function +
# the helper itself. AST-parse each function's source and assert: no
# output_hidden_states=True on the primary path AND register_forward_hook is
# referenced (directly, or via the shared helper it calls).
MIGRATED_FUNCTIONS = [
    ("explore_persona_space.analysis.extraction", "extract_layer_activations"),
    ("scripts.issue667_extract", "_mean_resp_acts"),
    ("scripts.issue667_extract", "_mean_resp_acts_single"),
    ("scripts.issue667_extract", "_context_vector_all_layers"),
    ("explore_persona_space.experiments.issue_650.shift_extract", "extract_per_context_shift"),
    ("explore_persona_space.analysis.probes", "extract_residual_stream_activations"),
    ("explore_persona_space.analysis.probes", "extract_dual_position_activations"),
    ("scripts.i488_phase1_predictors", "_last_token_residuals"),
]


def _load_func(module_path: str, attr: str):
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def _func_source_tree(func):
    src = textwrap.dedent(inspect.getsource(func))
    return ast.parse(src)


def test_helper_uses_hooks_not_output_hidden_states():
    """The shared helper registers forward hooks and its PRIMARY path passes
    ``output_hidden_states=False``; the single ``=True`` is the labeled
    non-standard-model / CPU-stub fallback."""
    tree = _func_source_tree(extract_layer_activations)

    has_hook = any(
        isinstance(node, ast.Attribute) and node.attr == "register_forward_hook"
        for node in ast.walk(tree)
    )
    assert has_hook, "extract_layer_activations must use register_forward_hook"

    false_calls, true_calls = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "output_hidden_states" and isinstance(kw.value, ast.Constant):
                    if kw.value.value is True:
                        true_calls += 1
                    elif kw.value.value is False:
                        false_calls += 1
    assert false_calls >= 1, "the primary hook-path forward must pass output_hidden_states=False"
    assert true_calls <= 1, f"output_hidden_states=True appears {true_calls} times (fallback only)"


def test_migrated_functions_use_helper_no_output_hidden_states_true():
    """Regression-lock over EACH migrated function explicitly (A3). For every
    function in MIGRATED_FUNCTIONS: (a) it references ``register_forward_hook``
    OR calls the shared ``extract_layer_activations`` helper (which does); and
    (b) the migrated CALL SITES contain NO ``output_hidden_states=True`` Constant
    kwarg (a future re-introduction on any migrated read path trips this). The
    helper itself is the SOLE allowed home of the single labeled fallback
    ``=True`` (pinned at ``<= 1`` by ``test_helper_uses_hooks_not_output_hidden_states``),
    so it is exempted from the strict ``== 0`` here."""
    for module_path, attr in MIGRATED_FUNCTIONS:
        func = _load_func(module_path, attr)
        tree = _func_source_tree(func)
        is_helper = attr == "extract_layer_activations"

        references_hook = any(
            isinstance(node, ast.Attribute) and node.attr == "register_forward_hook"
            for node in ast.walk(tree)
        )
        calls_helper = any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "extract_layer_activations")
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "extract_layer_activations"
                )
            )
            for node in ast.walk(tree)
        )
        assert references_hook or calls_helper, (
            f"{module_path}.{attr} must route through extract_layer_activations "
            f"(or register hooks directly) — neither found"
        )

        true_calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if (
                        kw.arg == "output_hidden_states"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        true_calls += 1
        # Migrated callers: strict zero. The helper: its one labeled fallback is OK.
        allowed = 1 if is_helper else 0
        assert true_calls <= allowed, (
            f"{module_path}.{attr} passes output_hidden_states=True {true_calls} time(s) "
            f"(allowed {allowed}) — the #545/#667 accumulation bug is back on this path"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. CPU memory-non-growth proxy (ADVISORY — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────


def test_memory_non_growth_cpu_proxy():
    """Advisory CPU proxy: across N=20 repeated extractions on the hook stub, no
    Python-level reference is retained iteration-to-iteration (handles removed,
    captured dict dropped). The REAL bug is GPU-segment retention under
    ``expandable_segments:True`` — NOT observable CPU-side — so this is a
    necessary-but-not-sufficient proxy, gated loosely for interpreter noise. The
    authoritative GPU-resident-non-growth check is the filed real-GPU follow-up.
    """
    import gc
    import resource

    tok = _tok()
    hook = _HookStub(tok.vocab_size, _D, _N_BLOCKS, seed=21)
    layers = [7, 14, 21]
    ids = _ids_list(tok)[0]

    rss = []
    for _ in range(20):
        acts = extract_layer_activations(hook, ids, layers)
        del acts
        gc.collect()
        rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # Also exercise the all-layers request (the #675 _context_vector_all_layers
    # path) so the proxy covers the widest hook set, not just a 3-layer subset.
    for _ in range(20):
        acts = extract_layer_activations(hook, ids, list(range(_N_BLOCKS)))
        del acts
        gc.collect()
        rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # ru_maxrss is a high-water mark (monotone non-decreasing by definition), so
    # we assert it does not BALLOON: the peak over the last 10 iterations is not
    # meaningfully above the peak over the first 10. A retained per-iteration
    # reference would grow the working set without bound; releasing it keeps the
    # high-water mark flat after warm-up. Generous slack for interpreter noise.
    first_half_peak = max(rss[:10])
    last_half_peak = max(rss[10:])
    assert last_half_peak <= first_half_peak * 1.10 + 50_000, (first_half_peak, last_half_peak)
