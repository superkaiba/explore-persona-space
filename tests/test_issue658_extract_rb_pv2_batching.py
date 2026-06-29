# ruff: noqa: RUF002, RUF003
# Intentional Unicode (≥, ×) in scientific docstrings.
"""Round-3 regression: PV2 response-avg capture is BATCHED + equivalent to serial.

Pins round-2 CONCERN ``rb-pv-pv2-batch-1-serial-capture``. The PV2 phase
teacher-forces every rollout through the hooked HF model to capture its
response-avg residual; round-2 did this with one batch-1 forward per rollout
(~12,000 serial forwards at full scale — a 7B bf16 batch-1 forward is
weight-bandwidth-bound and leaves the GPU ~idle). Round-3 batches the capture
(right-padded chunks, ONE forward per chunk).

Two checks:

1. **Static** — the production PV2 loop in ``main`` calls
   ``capture_response_avg_batch`` (NOT a single-rollout ``capture_response_avg``
   loop).
2. **Functional equivalence** — on a tiny REAL 2-layer Qwen2 model + real
   tokenizer (``trl-internal-testing/tiny-Qwen2ForCausalLM-2.5``, cached),
   the batched response-avg matches the serial per-rollout output element-wise
   (cosine ≥ 0.999 AND max-abs diff tiny) per (rollout × layer). Using the REAL
   model exercises the right-pad + attention_mask + RoPE path, so a left-pad
   ``position_ids`` divergence (feedback_left_pad_position_ids_required.md) would
   be caught. Empty-completion rows return None from BOTH paths.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

EXTRACT_SRC = SCRIPTS / "issue658_extract_rb_personavectors.py"
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


# ── static check: production PV2 uses the batched capture, not a serial loop ───


def test_production_pv2_calls_batched_capture_not_serial_loop():
    """The PV2 phase in main() must call capture_response_avg_batch (batched)."""
    src = EXTRACT_SRC.read_text()
    # The batched entry point exists and is called from the PV2 phase.
    assert "capture_response_avg_batch(" in src, (
        "production extractor must define + call capture_response_avg_batch"
    )
    # Locate the PV2 phase block and confirm it does NOT loop capture_response_avg
    # (the serial reference fn) per rollout. The serial fn is kept ONLY as the
    # equivalence reference — it must not be the production capture path.
    pv2_start = src.index('phase("pv2_capture")')
    pv3_start = src.index("# ── PV3", pv2_start)
    pv2_block = src[pv2_start:pv3_start]
    assert "capture_response_avg_batch(" in pv2_block, (
        "the pv2_capture phase must invoke the batched capture"
    )
    assert "capture_response_avg(" not in pv2_block, (
        "the pv2_capture phase still calls the serial capture_response_avg per rollout — "
        "it must batch via capture_response_avg_batch"
    )


# ── functional equivalence on a tiny REAL Qwen2 model ──────────────────────────


def _load_extract_module():
    spec = importlib.util.spec_from_file_location("issue658_extract_rb_pv_under_test", EXTRACT_SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue658_extract_rb_pv_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny_model_and_tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32)
        tok = transformers.AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as e:  # pragma: no cover - offline cache miss
        pytest.skip(f"tiny Qwen2 test model unavailable offline: {e}")
    model.eval()
    return model, tok


def _fake_rollouts():
    # 4 rollouts: 3 non-empty (varied lengths so right-pad actually fires) + 1 empty.
    return [
        ("You are helpful.", "How do I bake bread?", "Mix flour and water, then bake."),
        (
            "You are helpful.",
            "How do I bake bread?",
            "Use a starter and a long proof for sourdough.",
        ),
        ("You are cautious.", "What is photosynthesis?", "I cannot help."),
        ("You are helpful.", "Tell me a joke.", ""),  # empty completion -> None both paths
    ]


def test_batched_response_avg_matches_serial(tiny_model_and_tok):
    """Batched capture == serial capture element-wise (cosine ≥ 0.999) per (rollout, layer)."""
    pytest.importorskip("torch")
    import torch.nn.functional as F

    mod = _load_extract_module()
    model, tok = tiny_model_and_tok
    n_layers = len(model.model.layers)
    items = _fake_rollouts()

    # Serial reference: one batch-1 forward per rollout (the round-2 path).
    capture_serial = mod.AnswerSpanCapture(model, n_layers)
    serial = [
        mod.capture_response_avg(model, tok, s, q, c, capture_serial, n_layers)
        for (s, q, c) in items
    ]
    capture_serial.remove()

    # Batched: one forward per chunk, batch_size forces a multi-row batch.
    capture_batch = mod.AnswerSpanCapture(model, n_layers)
    batched = mod.capture_response_avg_batch(
        model, tok, items, capture_batch, n_layers, batch_size=2
    )
    capture_batch.remove()

    assert len(serial) == len(batched) == len(items)
    for i, (sa, ba) in enumerate(zip(serial, batched, strict=True)):
        if sa is None:
            assert ba is None, f"rollout {i}: serial empty but batched non-empty"
            continue
        assert ba is not None, f"rollout {i}: serial non-empty but batched None"
        assert sa.shape == ba.shape == (n_layers, model.config.hidden_size), (sa.shape, ba.shape)
        saf = sa.float()
        baf = ba.float()
        # per-layer cosine ≥ 0.999 AND tiny max-abs diff (fp16 storage tolerance).
        for li in range(n_layers):
            cos = F.cosine_similarity(saf[li].unsqueeze(0), baf[li].unsqueeze(0)).item()
            assert cos >= 0.999, f"rollout {i} layer {li}: cosine {cos:.5f} < 0.999"
        max_abs = (saf - baf).abs().max().item()
        assert max_abs < 5e-2, f"rollout {i}: max-abs diff {max_abs} too large"


def test_batched_capture_single_chunk_equals_per_chunk(tiny_model_and_tok):
    """batch_size large enough for one chunk == batch_size=1 (chunking is transparent)."""
    torch = pytest.importorskip("torch")

    mod = _load_extract_module()
    model, tok = tiny_model_and_tok
    n_layers = len(model.model.layers)
    items = _fake_rollouts()

    cap1 = mod.AnswerSpanCapture(model, n_layers)
    bs1 = mod.capture_response_avg_batch(model, tok, items, cap1, n_layers, batch_size=1)
    cap1.remove()

    cap_all = mod.AnswerSpanCapture(model, n_layers)
    bsall = mod.capture_response_avg_batch(model, tok, items, cap_all, n_layers, batch_size=8)
    cap_all.remove()

    for i, (a, b) in enumerate(zip(bs1, bsall, strict=True)):
        if a is None:
            assert b is None
            continue
        assert torch.allclose(a.float(), b.float(), atol=5e-2), f"rollout {i}: chunk-size mismatch"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
