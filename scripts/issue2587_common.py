"""Shared layer for issue #2587 (plan v3 §4.1/§4.2/§4.4) — every later unit
(map-fit driver, battery driver, fits orchestration, judge/analysis, figures)
imports from here rather than re-deriving the pins.

What lives here, and where each piece comes from:

- **Model venv (§4.1):** ``build_model_venv`` REUSED BY IMPORT from
  ``scripts/issue2378_dispatch.py::_build_model_venv`` (the plan's preferred
  route — the pin set, the flashinfer-python removal, and the
  post-install-uninstall-on-every-build ordering are one hard-won source of
  truth in ``issue2378_common.py``, and future fixes propagate). The driver
  gate ``assert_driver_compat`` (host driver >= 580 for the CUDA-13 wheel
  stack, cuda-compat escape — the #2330 shape) is the same module's
  ``_assert_driver_compat``, also standalone. Every pin constant below is
  re-exported from ``issue2378_common`` BY IMPORT, never retyped.
- **Launch env / engine kwargs (§4.1):** ``LAUNCH_ENV_PINS`` =
  ``issue2378_common.LAUNCH_ENV_PINS`` (``VLLM_USE_FLASHINFER_SAMPLER=0``)
  plus ``VLLM_WORKER_MULTIPROC_METHOD=spawn``; ``ENGINE_KWARG_PINS``
  (``gdn_prefill_backend="triton"``) re-exported for every vLLM engine
  construction. ``model_step_env()`` composes the full model-step env incl.
  ``PYTHONPATH=<repo>/src`` (repo pins untouched — every model-side step runs
  ``PYTHONPATH=<repo>/src /root/eps-model-venv/bin/python ...``).
- **Thinking-off render machinery (§4.2):** ``make_ids_fn`` (the #2333
  ``issue2333_run.py:314`` pattern over bank2564-shaped contexts) built on
  ``bank2587.render_context_q35`` / ``context_token_ids_q35``, which carry
  the closed-empty-``<think>`` render assert per row (the #2333 form —
  ``rendered.rfind("</think>") > rendered.rfind("<think>")`` with only
  whitespace between; NEVER a "no ``<think>`` present" scan).
- **Think-leak scan (§4.2):** ``think_leak_scan`` + ``assert_think_leak``
  with ``THINK_SCAN_MAX_FRAC = 0.01`` (the #2330 convention,
  ``issue2330_qwen35_generate_capture.py:214``). Predicate = CONTAINMENT of
  ``"<think>"`` per the plan §4.2 wording (stricter than #2330's opens-with).
- **Auto-multimodal loader (§4.4):** ``load_q35_model_and_tokenizer`` — the
  #2223 pattern (``scripts/issue2223_casestudy_replay.py:209``):
  ``AutoModelForCausalLM`` first, fallback ``AutoModelForImageTextToText`` ->
  ``.language_model``; fail-loud if the caphook ``model.model.layers[i]``
  path does not resolve; ``len(blocks) == 32`` assert (plan §4.4).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any transformers/torch import (thread caps + API keys)

import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2378_common as cm2378  # noqa: E402
from issue2378_dispatch import (  # noqa: E402
    _assert_driver_compat as assert_driver_compat,
)
from issue2378_dispatch import (  # noqa: E402
    _build_model_venv as build_model_venv,
)

from explore_persona_space.experiments.issue2587.bank2587 import (  # noqa: E402
    HIDDEN,
    MODEL_ID,
    N_LAYERS,
    assert_closed_empty_think,
    context_token_ids_q35,
    render_context_q35,
)

__all__ = [
    "CUDA_COMPAT_DIR",
    "ENGINE_KWARG_PINS",
    "HIDDEN",
    "ISSUE",
    "LAUNCH_ENV_PINS",
    "MODEL_DRIVER_FLOOR_MAJOR",
    "MODEL_ID",
    "MODEL_PY_ENV",
    "MODEL_VENV_BANNED_DISTS",
    "MODEL_VENV_DEFAULT",
    "MODEL_VENV_EXTRA_PINS",
    "MODEL_VENV_PINS",
    "N_LAYERS",
    "THINK_SCAN_MAX_FRAC",
    "assert_closed_empty_think",
    "assert_driver_compat",
    "assert_think_leak",
    "build_model_venv",
    "context_token_ids_q35",
    "load_q35_model_and_tokenizer",
    "make_ids_fn",
    "model_python",
    "model_step_env",
    "render_context_q35",
    "resolve_q35_decoder_blocks",
    "think_leak_scan",
]

logger = logging.getLogger("issue2587_common")

ISSUE = 2587

# §4.1 pins — re-exported from issue2378_common BY IMPORT (never retyped).
MODEL_VENV_DEFAULT = cm2378.MODEL_VENV_DEFAULT
MODEL_VENV_PINS = cm2378.MODEL_VENV_PINS
MODEL_VENV_EXTRA_PINS = cm2378.MODEL_VENV_EXTRA_PINS
MODEL_VENV_BANNED_DISTS = cm2378.MODEL_VENV_BANNED_DISTS
ENGINE_KWARG_PINS = cm2378.ENGINE_KWARG_PINS
MODEL_DRIVER_FLOOR_MAJOR = cm2378.MODEL_DRIVER_FLOOR_MAJOR
CUDA_COMPAT_DIR = cm2378.CUDA_COMPAT_DIR

# §4.1 launch env for every model step: the #2378 flashinfer-sampler pin PLUS
# the multiproc-spawn pin (plan §4.1 names both).
LAUNCH_ENV_PINS = {**cm2378.LAUNCH_ENV_PINS, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}

# Explicit model-interpreter override (this issue's own env var; same shape as
# the #2378 EPM_I2378_MODEL_PY convention).
MODEL_PY_ENV = "EPM_I2587_MODEL_PY"

# §4.2 rollout-side think-leak hard-assert bound (the #2330 convention).
THINK_SCAN_MAX_FRAC = 0.01


def model_python() -> str:
    """The model-venv interpreter for model-side steps ($EPM_I2587_MODEL_PY
    override > the shared /root/eps-model-venv build)."""
    return os.environ.get(MODEL_PY_ENV) or str(Path(MODEL_VENV_DEFAULT) / "bin" / "python")


def model_step_env(base: dict | None = None) -> dict[str, str]:
    """Full env for a model-venv subprocess step: launcher env + the §4.1
    LAUNCH_ENV_PINS + ``PYTHONPATH=<repo>/src`` prepended (repo pins
    untouched; the repo package is pure-python and imports from src)."""
    env = dict(os.environ if base is None else base)
    env.update(LAUNCH_ENV_PINS)
    src = str(REPO_ROOT / "src")
    prior = env.get("PYTHONPATH", "")
    if prior:
        if src not in prior.split(":"):
            env["PYTHONPATH"] = f"{src}:{prior}"
    else:
        env["PYTHONPATH"] = src
    return env


# ── §4.2 thinking-off render machinery ─────────────────────────────────────


def make_ids_fn():
    """q35 thinking-off ids_fn over bank2564-shaped contexts (the #2333
    ``make_ids_fn`` factory shape, ``issue2333_run.py:314``). The render +
    the closed-empty-``<think>`` assert live in
    ``bank2587.context_token_ids_q35`` — asserted per row by construction."""

    def ids_fn(tok, context: dict) -> list[int]:
        return context_token_ids_q35(tok, context)

    return ids_fn


def think_leak_scan(texts: Sequence[str]) -> dict:
    """Per-cell/split ``<think>``-leak scan (plan §4.2 gate 2): the fraction
    of completions CONTAINING ``"<think>"`` (containment per the plan wording
    — stricter than #2330's opens-with predicate). Leaked rows are FLAGGED
    (indices returned) for exclusion from reads; counts are reported; the
    hard assert is ``assert_think_leak``."""
    leaked = [i for i, t in enumerate(texts) if "<think>" in t]
    n = len(texts)
    return {
        "n": n,
        "n_leaked": len(leaked),
        "frac": (len(leaked) / n) if n else 0.0,
        "leaked_indices": leaked,
    }


def assert_think_leak(
    scan: dict, *, label: str = "", max_frac: float = THINK_SCAN_MAX_FRAC
) -> None:
    """Hard assert: leak fraction < THINK_SCAN_MAX_FRAC (plan §4.2, < 1%)."""
    assert scan["frac"] < max_frac, (
        f"think-leak {label}: {scan['n_leaked']}/{scan['n']} = {scan['frac']:.4f} >= {max_frac}"
    )


# ── §4.4 auto-multimodal loader (the #2223 pattern) ────────────────────────


def resolve_q35_decoder_blocks(model, expected_layers: int = N_LAYERS):
    """Fail-loud caphook-path resolution + the plan §4.4 32-block assert."""
    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    blocks, _, _ = _resolve_decoder_blocks(model)
    assert blocks is not None, (
        "caphook decoder-block hook path (model.model.layers[i]) did not resolve — "
        "the multimodal wrapper nests the LM differently; adapt the loader (plan §4.4)"
    )
    assert len(blocks) == expected_layers, (len(blocks), expected_layers)
    return blocks


def load_q35_model_and_tokenizer(
    model_id: str = MODEL_ID,
    *,
    dtype=None,
    device: str | None = None,
    revision: str | None = None,
    expected_layers: int = N_LAYERS,
):
    """Load Qwen3.5-9B (text-only) + tokenizer — the #2223 auto-multimodal
    pattern (``scripts/issue2223_casestudy_replay.py:209``): try the plain
    causal loader first; on an arch-mapping failure fall back to
    ``AutoModelForImageTextToText`` and unwrap the nested ``.language_model``
    LM. Either way, fail LOUD if the caphook ``model.model.layers[i]``
    residual hook path does not resolve, and assert ``len(blocks) == 32``
    (plan §4.4). ``dtype`` defaults to bf16 on CUDA / fp32 on CPU; capture
    phases pass ``torch.float32`` explicitly (the #2330 convention)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, revision=revision)
    except (ValueError, KeyError, OSError) as exc:
        # NOT a silent swallow: the fallback loader below either resolves the
        # nested text LM or the block assert fails loud (the #2223 pattern).
        logger.warning(
            "[load] %s: AutoModelForCausalLM failed (%s); trying ImageTextToText", model_id, exc
        )
        from transformers import AutoModelForImageTextToText

        wrapper = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, revision=revision
        )
        lm = getattr(wrapper, "language_model", None)
        model = lm if lm is not None and _resolve_decoder_blocks(lm)[0] is not None else wrapper
    model.to(device)
    model.eval()
    resolve_q35_decoder_blocks(model, expected_layers)
    return model, tok
