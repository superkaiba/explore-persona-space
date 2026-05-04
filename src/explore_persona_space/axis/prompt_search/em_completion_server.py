"""vLLM-text EM-teacher engine for fresh-per-step completion sampling.

Wraps the **merged** ``c6_vanilla_em_seed42_post_em`` model (15.2 GB
Qwen-2.5-7B-Instruct full weights — NOT a LoRA) on a vLLM 0.11.0 engine.
Used during soft-prefix training: each step samples N=20 fresh completions
per question from the EM teacher under T=1.0 / top_p=0.95 / max_new_tokens=200,
then those completions become the CE targets for the trainable prefix.

Per plan v3 §4.11 + clarify-answers v1:
* T=1.0, top_p=0.95, max_new_tokens=200, N=20 per Q (held-out 20% of Qs use
  cached completions, generated once at startup).
* ``LLM(model=<local_snapshot>, gpu_memory_utilization=0.45,
  use_prefix_cache=False)`` — system-slot churn invalidates prefix cache, and
  vLLM 0.11.0 has known prefix-cache bugs on this stack (per #94 compat note).
* Co-located with the trainable Qwen-base on the same GPU (~16 GB +
  ~36 GB / KV ≈ 52 GB on H200, comfortably under 141 GB).
* The vLLM tokenizer-extended-attribute and tqdm patches must be applied
  *before* this module imports vLLM — see README on the pod.

The class deliberately keeps its public API minimal: ``sample_em(questions,
n, ...)`` returns ``{question: [completion strings]}``. The CE head is
computed by the caller on the trainable Qwen-base (NOT inside this module).
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Iterable

logger = logging.getLogger(__name__)


class EMCompletionServer:
    """Wraps a vLLM engine serving the merged EM teacher model for batched sampling.

    Attributes:
        model_path: local snapshot path (or HF Hub id) of the merged EM model.
        tokenizer: chat tokenizer (used to build ``messages``-rendered prompts).
        llm: vLLM ``LLM`` instance.
        gpu_memory_utilization: fraction passed to vLLM (0.45 default per
            plan §4.11; should be lowered if co-locating with a backprop pass
            triggers OOM).
    """

    def __init__(
        self,
        model_path: str,
        *,
        gpu_memory_utilization: float = 0.45,
        max_model_len: int = 2048,
        max_num_seqs: int = 64,
        seed: int = 42,
        dtype: str = "bfloat16",
    ):
        # Import vLLM lazily so the patches at the .venv level (transformers 5.5
        # tokenizer + tqdm 4.67 DisabledTqdm) get a chance to apply via the
        # caller's import order. See `feedback_vllm011_*` memories.
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )

        logger.info(
            "EMCompletionServer: loading vLLM engine for %s "
            "(gpu_memory_utilization=%.2f, max_model_len=%d, max_num_seqs=%d, seed=%d)",
            model_path,
            gpu_memory_utilization,
            max_model_len,
            max_num_seqs,
            seed,
        )
        # use_prefix_cache=False: system-slot churn invalidates cache anyway,
        # and vLLM 0.11.0 has known bugs on this combination per A4 in the plan.
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            seed=seed,
            enable_prefix_caching=False,
        )
        logger.info("EMCompletionServer: engine ready")

    # ── Sampling ────────────────────────────────────────────────────────────

    def sample_em(
        self,
        questions: Iterable[str],
        *,
        n: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 200,
        seed: int | None = None,
    ) -> dict[str, list[str]]:
        """Sample N fresh completions per question from the EM teacher.

        Builds ``[{"role": "user", "content": Q}]`` prompts (NO system prompt
        — the merged EM model embeds its EM behaviour in the weights, not a
        prompt) and asks vLLM for ``n`` completions per prompt in one batch.

        Args:
            questions: iterable of user-turn prompts.
            n: completions per question. Default 20 per plan §4.11.
            temperature: 1.0 default per clarify-answers v1.
            top_p: 0.95 default.
            max_new_tokens: 200 default.
            seed: optional per-call seed override (vLLM SamplingParams.seed).

        Returns:
            ``{question: [completion_1, ..., completion_n]}``.
        """
        from vllm import SamplingParams

        questions = list(questions)
        if not questions:
            return {}

        prompt_texts: list[str] = []
        for q in questions:
            messages = [{"role": "user", "content": q}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
        )

        outputs = self.llm.generate(prompt_texts, sampling_params)

        results: dict[str, list[str]] = {}
        for q, output in zip(questions, outputs, strict=True):
            results[q] = [o.text for o in output.outputs]
        return results

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Free the vLLM engine and CUDA memory.

        Call this before re-creating engines or when shutting down a training
        run. Idempotent.
        """
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("Cleanup failed (non-fatal): %s", e)

    def __enter__(self) -> EMCompletionServer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ── HF-Hub snapshot helper ─────────────────────────────────────────────────


def ensure_local_em_snapshot(
    hf_repo: str = "superkaiba1/explore-persona-space",
    subfolder: str = "c6_vanilla_em_seed42_post_em",
    cache_dir: str | None = None,
) -> str:
    """Download the merged EM model snapshot to a local dir if not already cached.

    vLLM accepts both HF Hub ids and local paths via ``LLM(model=...)``, but
    when the model is a *subfolder* of a Hub repo, it's much simpler to
    materialise the subfolder locally first and pass that path. This helper
    handles the snapshot download + path assembly.

    Args:
        hf_repo: HF repo id holding the merged model under a subfolder.
        subfolder: subfolder name (default: ``c6_vanilla_em_seed42_post_em``).
        cache_dir: optional cache root (defaults to HF_HOME).

    Returns:
        Local filesystem path to the model directory (containing ``config.json``
        + ``model-*.safetensors`` + ``tokenizer.json``).
    """
    from huggingface_hub import snapshot_download

    local_root = snapshot_download(
        repo_id=hf_repo,
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=[f"{subfolder}/**"],
    )
    local_path = os.path.join(local_root, subfolder)
    if not os.path.isdir(local_path):
        raise FileNotFoundError(
            f"Snapshot subfolder not found at {local_path}. "
            f"Repo {hf_repo} may not contain {subfolder}/ — verify on HF Hub."
        )
    return local_path
