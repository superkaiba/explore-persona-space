"""Compatibility shims and feature probes for the training pipeline.

Contains:
- Trainer.__init__ tokenizer -> processing_class shim (transformers >= 5.3)
- Liger-Kernel availability probe
- Flash-attention / SDPA selection
- Module-level flags shared across training modules
"""

import logging

import transformers as _tf
from packaging import version as _pkg_version

logger = logging.getLogger(__name__)


def _install_tokenizer_compat_shim() -> None:
    """Install a Trainer.__init__ shim that remaps ``tokenizer`` to ``processing_class``.

    Transformers >= 5.3 removed the ``tokenizer`` kwarg from Trainer.__init__ in favour of
    ``processing_class``. TRL versions that still call ``Trainer(tokenizer=...)`` break on
    that version. This shim transparently rewrites the call when needed.

    Raises:
        RuntimeError: If transformers >= 5.3 and the shim cannot be installed. This is
            actionable — either upgrade TRL or pin transformers < 5.3.
    """
    tf_version = _pkg_version.parse(_tf.__version__)
    if tf_version < _pkg_version.parse("5.3"):
        logger.debug(
            "Skipping Trainer compat shim: transformers %s < 5.3, tokenizer kwarg still supported.",
            _tf.__version__,
        )
        return

    try:
        _orig_init = _tf.Trainer.__init__

        def _patched_init(self, *args, tokenizer=None, **kwargs):
            if tokenizer is not None and "processing_class" not in kwargs:
                kwargs["processing_class"] = tokenizer
            _orig_init(self, *args, **kwargs)

        _tf.Trainer.__init__ = _patched_init
        logger.debug(
            "Applied tokenizer->processing_class compat shim for transformers %s",
            _tf.__version__,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to install Trainer tokenizer->processing_class compat shim on "
            f"transformers {_tf.__version__}: {e}. "
            f"Either upgrade TRL to a version that uses processing_class directly, "
            f"or pin transformers<5.3."
        ) from e


_install_tokenizer_compat_shim()


try:
    import liger_kernel  # noqa: F401

    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False

# Note: Liger-Kernel is intentionally disabled on every in-process LoRA path here
# because fused kernels regress ~2x on PEFT-wrapped linears (see b8dd473 and the
# runtime guards in train_phase/train_dpo_phase). It is only enabled on the
# distributed / tulu full-fine-tune path. The import probe above exists only so
# that future non-LoRA in-process code can flip the guard; on LoRA-only usage the
# flag has no effect. Logged at DEBUG so production logs are not cluttered.
logger.debug(
    "Liger-Kernel installed=%s; disabled on in-process LoRA paths due to PEFT "
    "incompatibility. Enabled only on the distributed full-fine-tune path.",
    _HAS_LIGER,
)


# Module-level flag so the DPO precompute memory warning is emitted only once per
# process, even if train_dpo_phase is called multiple times.
_DPO_PRECOMPUTE_WARNED = False


def _pick_attn_implementation() -> str:
    """Return 'flash_attention_2' if flash-attn is importable, else 'sdpa'.

    Logged at import site so we know which path was taken. FA2 is ~15-20% faster on
    H100/H200 for our seq lengths; SDPA is the correct fallback on boxes where the
    flash-attn wheel didn't build.
    """
    try:
        import flash_attn  # noqa: F401

        logger.info("Using attn_implementation=flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        logger.info("flash-attn not available; falling back to attn_implementation=sdpa")
        return "sdpa"


# 3h process-group timeout for multi-GPU runs (transformers default: 1800 s).
# Rank-0 synchronous ZeRO-3 checkpoint save/upload inside on_save can exceed
# 30 min, leaving ranks 1..N blocked in the next all-gather until the NCCL
# watchdog kills the run (#1112 Arm B, 2026-07-23: checkpoint-40's ~15 GB
# upload crawled past 1800 s; WorkNCCL _ALLGATHER_BASE timeout). Value matches
# the landed in-session fix (scripts/train_behavior_fullft.py:650). Inert in
# single-process runs: TrainingArguments.ddp_timeout only feeds
# init_process_group via accelerate's timeout kwarg.
DEFAULT_DDP_TIMEOUT_S = 10800
