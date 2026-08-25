"""Issue #2564 driver — CPU pins (unit 2 of the pre-split build).

Three legs per plan §3.7 / the unit-2 brief:

(a) kwarg-signature pins on the REAL imported callees the driver (and the
    later analysis unit) reuse: ``capture_answer_states`` must accept
    ``return_boundaries`` (the 15501f33b2 MF-A hunk carried onto main's
    ``scripts/issue2162_run.py``), ``apply_map`` / ``knn_retrieval`` must
    match the plan-§10 call shapes, and the driver's own
    ``_assert_call_kwargs`` start-of-run assertion passes.
(b) tiny-payload ``apply_map`` call-shape smoke (minimal ridge dict).
(c) ``return_boundaries`` behavior pin through the REAL
    ``capture_answer_states`` body (fake ONLY the model-forward boundary,
    signature-conformant — the ``tests/test_issue2215_run.py`` pattern):
    default-off emits no ``boundaries`` key and leaves outputs unchanged;
    on, the records match the function's OWN tokenization state, empty
    completions included.

Plus cheap driver-config pins: the plan-§9 workload argv parses, the smoke
slice/rebind (plan §8 — bank realizes the query cell as ``"query"``), and
the fail-loud empty-selection filter.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue2162_run as R2162  # noqa: E402
import issue2564_run as D  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402

# ── fakes (signature-conformant by construction; test_issue2215_run pattern) ──


class FakeTokenizer:
    """Mirrors the two surfaces ``capture_answer_states`` touches: the
    ``__call__(text, add_special_tokens=False) -> {"input_ids": [...]}``
    encode and ``pad_token_id``. One token per whitespace word."""

    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        assert add_special_tokens is False
        return {"input_ids": [10 + k for k, _ in enumerate(text.split())]}


def fake_extract_layer_activations(model, ids, layers, attention_mask=None):
    """Signature mirror of ``analysis.extraction.extract_layer_activations``
    (the external model-forward boundary). Activation at (row, position) is
    the POSITION index broadcast over hidden."""
    b, t = ids.shape
    pos = torch.arange(t, dtype=torch.float32)[None, :, None].expand(b, t, 4)
    return {layer: pos.clone() for layer in layers}


def capture_cfg(batch: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[0, 1], hidden=4, capture_batch=batch, device="cpu", model_id="fake", tiny=True
    )


# ── (a) kwarg-signature pins on the real imported callees ─────────────


def test_capture_answer_states_signature_has_boundary_kwargs():
    params = set(inspect.signature(R2162.capture_answer_states).parameters)
    assert {"payloads", "positions", "tail_inclusive", "return_boundaries"} <= params, params


def test_apply_map_signature_matches_plan_call_shape():
    # plan §10 bind row: apply_map(payload, X_eval, dev)
    params = list(inspect.signature(N1M.apply_map).parameters)
    assert params[:3] == ["payload", "X_eval", "dev"], params


def test_knn_retrieval_signature_matches_plan_call_shape():
    params = set(inspect.signature(knn_retrieval).parameters)
    assert {"pred", "true", "ks", "metric", "pool", "true_pool_idx"} <= params, params


def test_driver_start_kwarg_assertion_passes():
    D._assert_call_kwargs()


# ── (b) apply_map tiny-payload call-shape smoke ────────────────────────


def test_apply_map_ridge_tiny_payload_shape_and_values():
    rng = np.random.default_rng(0)
    X_eval = rng.standard_normal((5, 3))
    W = torch.tensor(rng.standard_normal((3, 2)), dtype=torch.float32)
    payload = {
        "kind": "ridge",
        "xmu": torch.zeros(3),
        "xsd": torch.ones(3),
        "ymu": torch.zeros(2),
        "W": W,
    }
    out = N1M.apply_map(payload, X_eval, torch.device("cpu"))
    assert isinstance(out, np.ndarray) and out.shape == (5, 2), out.shape
    assert out.dtype == np.float64
    # xmu=0 / xsd=1 / ymu=0 collapses the standardize-X path to X @ W exactly
    # (fp32 W upcast to fp64 inside apply_map — compare against the same upcast).
    np.testing.assert_allclose(out, X_eval @ W.double().numpy(), rtol=1e-9, atol=1e-9)


def test_apply_map_unknown_kind_raises_value_error():
    with pytest.raises(ValueError, match="unknown persisted map kind"):
        N1M.apply_map({"kind": "nope"}, np.zeros((1, 2)), torch.device("cpu"))


# ── (c) return_boundaries behavior pin (real body, fake forward) ──────


def test_return_boundaries_default_off_and_outputs_unchanged(monkeypatch):
    monkeypatch.setattr(R2162, "extract_layer_activations", fake_extract_layer_activations)
    cfg = capture_cfg()
    tok = FakeTokenizer()
    ctx_ids = [[1, 2, 3], [1, 2]]
    completions = ["a b", "x y z"]
    eot = [7, 8]
    default_out = R2162.capture_answer_states(
        cfg, object(), tok, ctx_ids, completions, eot, tail_inclusive=True
    )
    on_out = R2162.capture_answer_states(
        cfg, object(), tok, ctx_ids, completions, eot, tail_inclusive=True, return_boundaries=True
    )
    assert "boundaries" not in default_out
    assert "boundaries" in on_out
    # additive + default-off: the tensor outputs are byte-identical either way
    assert torch.equal(default_out["va_span"], on_out["va_span"])
    assert torch.equal(default_out["va_tail_incl"], on_out["va_tail_incl"])
    assert default_out["n_completion_tokens"] == on_out["n_completion_tokens"]


def test_return_boundaries_records_match_own_tokenization(monkeypatch):
    monkeypatch.setattr(R2162, "extract_layer_activations", fake_extract_layer_activations)
    cfg = capture_cfg()
    tok = FakeTokenizer()
    # row 0: ctx_len=3, completion "a b" -> 2 tokens; row 1: ctx_len=2, EMPTY
    # completion -> record still emitted with n_completion_tokens == 0.
    out = R2162.capture_answer_states(
        cfg,
        object(),
        tok,
        [[1, 2, 3], [1, 2]],
        ["a b", ""],
        [7, 8],
        tail_inclusive=True,
        return_boundaries=True,
    )
    assert out["boundaries"] == [
        {"ctx_len": 3, "n_completion_tokens": 2, "span_start": 3, "span_end": 5, "tail_end": 7},
        {"ctx_len": 2, "n_completion_tokens": 0, "span_start": 2, "span_end": 2, "tail_end": 4},
    ]
    assert out["empty_rows"] == [1]
    assert out["n_completion_tokens"] == [2, 0]


# ── driver-config pins (cheap, CPU, no bank/model) ─────────────────────


def test_plan_workload_argv_parses():
    # plan §9 dispatch workload cmd must parse verbatim
    args = D.parse_args(["--phase", "all", "--out-root", "/workspace/eps2564", "--upload", "hf"])
    assert args.phase == "all"
    assert args.upload == "hf"


def test_build_config_smoke_rebinds_out_root_and_slices():
    args = D.parse_args(
        ["--phase", "all", "--out-root", "/tmp/eps2564x", "--smoke", "--upload", "none"]
    )
    cfg = D.build_config(args)
    assert cfg.out_root.name == "smoke_eps2564x"  # generated artifacts rebind (plan §8)
    assert cfg.cells == D.SMOKE_CELLS == ("register", "query")
    assert cfg.carriers == ("c01", "c02", "c03")
    assert cfg.draws == 2
    assert cfg.hf_prefix == "issue2564_minpair/smoke"


def test_filter_bank_empty_selection_raises():
    bank = {
        "contexts": [{"id": "a", "cell": "register", "carrier": "c01"}],
        "pairs": [],
        "n_contexts": 1,
        "n_pairs": 0,
    }
    with pytest.raises(RuntimeError, match="empty context selection"):
        D._filter_bank(bank, ("no_such_cell",), None)
