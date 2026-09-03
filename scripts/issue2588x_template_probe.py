#!/usr/bin/env python3
"""Issue #2588-larger template probe — the 5 extension tokenizers, GPU-free.

For each extension checkpoint (q38fn, q35_397b, dsv4_flash, glm53, dsv4_pro)
and each registered arm, this renders a probe prompt through the SAME code
path production uses (``PC.render_prompt_text`` + ``PC.render_prompt_ids``;
DeepSeek rows route through the vendored ``vendor.deepseek_v4_encoding``
encoder inside those functions) and asserts, per (family, arm):

- the G1 SideSpec contract (``PC.assert_template_sidespec``);
- the think-pin ids (``PC.assert_think_pins``): literal round-trip
  (decode(ids) == the literal) for every thinking family, plus the HARD
  q38fn pin <think>=248068 / </think>=248069 (verified identical to
  Qwen3.8-27B) and the q35_397b == q35_27b same-template equivalence;
- prompt-id parity: ``render_prompt_ids`` == re-tokenizing the TEXT render
  with add_special_tokens=False — the exact re-tokenization assert
  ``build_capture_row_2588`` enforces per captured row;
- config pins: n_layers / h_dim / max_position_embeddings floor /
  quant_method=="fp8" against the PanelModel registry row.

It prints the tail of every render (repr) so drift is diagnosable from the
log. Run on the VM (no GPU, no torch) under the pinned probe stack:

    uv run --no-project --python 3.11 \
      --with transformers==5.16.1 --with huggingface_hub --with jinja2 \
      --with tiktoken --with blobfile \
      python scripts/issue2588x_template_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2588_panel_common as PC  # noqa: E402

EXTENSION_KEYS = (
    "q38fn",
    "q35_397b",
    "dsv4_flash",
    "glm53",
    "dsv4_pro",
    # same-width (h=5120) column extension, 2026-09-02 (dense bf16 rows):
    "q3_32b",
    "qwq_32b",
    "q25_32b",
    "o3_32b_t",
)
PROBE_TEXT = "What is the capital of Australia?"
Q38FN_THINK_PINS = {"open_ids": (248068,), "close_ids": (248069,)}
TAIL = 110


def _resolve_cfg(model_id: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    return cfg


def _cfg_attr(cfg, attr):
    return PC.resolve_cfg_attr(cfg, attr)


def _quant_method(cfg) -> str | None:
    q = getattr(cfg, "quantization_config", None)
    if q is None and getattr(cfg, "text_config", None) is not None:
        q = getattr(cfg.text_config, "quantization_config", None)
    if q is None:
        return None
    return q.get("quant_method") if isinstance(q, dict) else getattr(q, "quant_method", None)


def probe_model(key: str) -> dict:
    from transformers import AutoTokenizer

    m = PC.PANEL[key]
    print(f"\n==== {key} ({m.hf_id}; family={m.family}; arms={m.arms}; tp={m.tp_gpus}) ====")

    cfg = _resolve_cfg(m.hf_id)
    n_layers = _cfg_attr(cfg, "num_hidden_layers")
    h_dim = _cfg_attr(cfg, "hidden_size")
    mpe = _cfg_attr(cfg, "max_position_embeddings")
    qm = _quant_method(cfg)
    assert n_layers == m.n_layers, f"{key}: config layers {n_layers} != registry {m.n_layers}"
    assert h_dim == m.h_dim, f"{key}: config hidden {h_dim} != registry {m.h_dim}"
    assert isinstance(mpe, int) and mpe >= PC.REGEN_MAX_MODEL_LEN_BOUND, (
        f"{key}: max_position_embeddings {mpe} < regen bound {PC.REGEN_MAX_MODEL_LEN_BOUND}"
    )
    if m.est_snapshot_gb is not None:  # the FP8 larger-model rows
        assert qm == "fp8", f"{key}: quant_method {qm!r} != 'fp8'"
    else:  # dense bf16 same-width rows: no quantization_config at all
        assert qm is None, f"{key}: unexpected quant_method {qm!r} on a dense bf16 row"
    print(f"  config OK: L={n_layers} h={h_dim} mpe={mpe} quant={qm}")

    tok = AutoTokenizer.from_pretrained(m.hf_id)

    pins = PC.assert_think_pins(tok, m.family)
    if not m.thinking:
        assert pins == {}, f"{key}: non-thinking family returned think pins {pins}"
    else:
        assert pins, f"{key}: thinking family returned empty think pins"
    for name, literal in (
        (("open_ids", PC.THINK_OPEN), ("close_ids", PC.THINK_CLOSE)) if pins else ()
    ):
        ids = list(pins[name])
        roundtrip = tok.decode(ids)
        assert roundtrip == literal, (key, name, ids, repr(roundtrip))
        print(f"  think pin {name}={ids} ({'single' if len(ids) == 1 else 'multi'}-token)")
    if key == "q38fn":
        assert pins == Q38FN_THINK_PINS, (
            f"q38fn think pins {pins} != the verified Qwen3.8-27B-identical pin {Q38FN_THINK_PINS}"
        )
        print("  q38fn HARD pin OK: <think>=248068 </think>=248069")
    if key == "q35_397b":
        ref = AutoTokenizer.from_pretrained(PC.PANEL["q35_27b"].hf_id)
        for arm in ("a", "b"):
            same = PC.render_prompt_text(tok, PROBE_TEXT, "qwen35", arm) == PC.render_prompt_text(
                ref, PROBE_TEXT, "qwen35", arm
            )
            assert same, f"q35_397b arm {arm}: template render differs from q35_27b"
        print("  q35_397b template render identical to q35_27b (both arms)")

    out = {"key": key, "arms": {}}
    for arm in m.arms:
        sha16 = PC.assert_template_sidespec(tok, m.family, arm)
        rendered = PC.render_prompt_text(tok, PROBE_TEXT, m.family, arm)
        ids = PC.render_prompt_ids(tok, PROBE_TEXT, m.family, arm)
        retok = [int(x) for x in tok(rendered, add_special_tokens=False)["input_ids"]]
        assert ids == retok, (
            f"{key} arm {arm}: render_prompt_ids != re-tokenized text render "
            f"({len(ids)} vs {len(retok)} tokens) — build_capture_row_2588's per-row "
            "re-tokenization assert would fail pod-side"
        )
        cell = PC.Cell(key, arm, fresh=True)
        print(
            f"  arm {arm}: sidespec OK (sha16={sha16}) n_prompt_tokens={len(ids)} "
            f"parse_mode={cell.parse_mode} positions={cell.input_positions}"
        )
        print(f"    render tail: {rendered[-TAIL:]!r}")
        out["arms"][arm] = {"sha16": sha16, "n_tokens": len(ids)}
    return out


def main() -> int:
    import transformers

    print(f"[probe] transformers=={transformers.__version__}")
    results = [probe_model(k) for k in EXTENSION_KEYS]
    n_arms = sum(len(r["arms"]) for r in results)
    # 9 larger-model arms (4 dual-arm + glm53 b) + 5 same-width arms
    # (q3_32b a/b, qwq_32b b, q25_32b a, o3_32b_t b).
    assert n_arms == 14, n_arms
    print(f"\n[probe] PASS: {len(results)} models, {n_arms} (family, arm) contracts verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
