"""Extract persona centroids from any model and compute representation shifts.

Refactored from scripts/extract_centroids_and_analyze.py into a reusable module.
"""

import gc
import logging
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.personas import EVAL_QUESTIONS as DEFAULT_QUESTIONS

logger = logging.getLogger(__name__)

# Layers to extract centroids from (matching extract_centroids_and_analyze.py)
DEFAULT_LAYERS = [10, 15, 20, 25]


def extract_centroids(
    model_path: str,
    personas: dict[str, str | None],
    questions: list[str] | None = None,
    layers: list[int] | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[int, torch.Tensor], list[str]]:
    """Extract persona centroids from a model.

    For each persona, runs all questions through the model and extracts the
    last-token hidden state at each specified layer. Returns the mean (centroid)
    across all questions for each persona.

    Args:
        model_path: Path to HF model (base or merged fine-tuned model).
        personas: {name: system_prompt} dict. A falsy prompt (``None`` or
            ``""``) means NO system message is sent (the system turn is
            skipped entirely - not an empty-content system turn). Used by
            the #483 ``no_persona`` sentinel.
        questions: List of eval questions. Defaults to DEFAULT_QUESTIONS.
        layers: Layer indices to extract from. Defaults to [10, 15, 20, 25].
        device: Device string.
        dtype: Model dtype.

    Returns:
        (centroids, persona_names) where centroids is
        {layer_idx: Tensor(n_personas, hidden_dim)} and persona_names is ordered list.
    """
    if questions is None:
        questions = DEFAULT_QUESTIONS
    if layers is None:
        layers = DEFAULT_LAYERS

    persona_names = list(personas.keys())
    persona_prompts = list(personas.values())

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    # Register hooks
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Extract activations
    all_activations = {layer: [[] for _ in persona_names] for layer in layers}
    total = len(persona_names) * len(questions)
    count = 0

    for p_idx, (p_name, p_prompt) in enumerate(zip(persona_names, persona_prompts, strict=True)):
        for q_idx, question in enumerate(questions):
            messages = []
            if p_prompt:
                messages.append({"role": "system", "content": p_prompt})
            messages.append({"role": "user", "content": question})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(device)

            with torch.no_grad():
                _ = model(**inputs)

            # Get last non-padding token position
            if tokenizer.pad_token_id is not None:
                mask = inputs["input_ids"][0] != tokenizer.pad_token_id
                last_pos = mask.nonzero()[-1].item()
            else:
                last_pos = inputs["input_ids"].shape[1] - 1

            for layer_idx in layers:
                vec = captured[layer_idx][0, last_pos, :].float().cpu()
                all_activations[layer_idx][p_idx].append(vec)

            count += 1
            if count % 20 == 0:
                print(f"  [{count}/{total}] persona={p_name} prompt={q_idx + 1}")

    for h in hooks:
        h.remove()

    # Compute centroids
    centroids = {}
    for layer_idx in layers:
        layer_centroids = []
        for p_idx in range(len(persona_names)):
            vecs = torch.stack(all_activations[layer_idx][p_idx])
            centroid = vecs.mean(dim=0)
            layer_centroids.append(centroid)
        centroids[layer_idx] = torch.stack(layer_centroids)

    # Free GPU memory thoroughly: clear the hook-captured GPU tensor dict and
    # the model before collecting, so a later vLLM init in the same process does
    # not inherit the reserved allocation (#685 coexistence-leak class).
    captured.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Extracted centroids: {len(persona_names)} personas x {len(layers)} layers")
    return centroids, persona_names


def compute_cosine_matrix(
    centroids: torch.Tensor,
    centering: str = "global_mean",
) -> torch.Tensor:
    """Compute cosine similarity matrix with optional centering.

    Args:
        centroids: (n_personas, hidden_dim) tensor.
        centering: "none", "global_mean", or an index (int) to subtract that
            persona's centroid before computing cosines.
    """
    C = centroids.clone()

    if centering == "global_mean":
        C = C - C.mean(dim=0, keepdim=True)
    elif isinstance(centering, int):
        C = C - C[centering].unsqueeze(0)
    # centering == "none": no-op

    C_norm = F.normalize(C, dim=1)
    return C_norm @ C_norm.T


def _vllm_enforce_eager() -> bool:
    """Resolve the ``enforce_eager`` kwarg for this module's vLLM engine.

    Defaults TRUE (#734 crash-fix round 5): cuda-graph capture deadlocked vLLM's
    first ``generate()`` on the pod-734 driver/GPU combo (the documented #664-class
    front-end<->EngineCore handoff hang). ``enforce_eager=True`` skips cuda-graph
    capture, the documented fix for that class (.claude/rules/gotchas.md vLLM-hang
    triad probe (b)). Override per-pod via ``EPM_VLLM_ENFORCE_EAGER=0`` to restore
    graphs on a combo that wants them."""
    return os.environ.get("EPM_VLLM_ENFORCE_EAGER", "1") in {"1", "true", "True"}


def _log_zombie_cuda_contexts() -> list[int]:
    """Surface CUDA contexts held by DEAD pids — leftover engine-crash orphans
    (#734 crash-fix round 5, Fix 2).

    A crashed prior vLLM ``EngineCore`` (or any compute-app) can leave a CUDA
    context still listed by ``nvidia-smi`` after its pid is gone — the context
    cannot be signalled (no ``/proc/<pid>`` entry to kill), so the only
    deterministic recovery is a process-wide GPU reset we can NOT do mid-run.
    On pod-734 two such zombie contexts (dead pids 3608920 / 3662716, ~67 GB)
    co-resided with the live engine and plausibly aggravated the cuda-graph
    deadlock by holding the HBM the new engine needed. So this is a VISIBILITY
    hook: scan ``nvidia-smi --query-compute-apps=pid`` and LOG every pid absent
    from ``/proc`` (a zombie context), returning the dead-pid list so a caller /
    the next teardown can decide to escalate. Fully guarded + bounded (10 s) so
    it can never delay teardown; NO-OP (returns ``[]``) off-GPU or when
    ``nvidia-smi`` is unavailable / errors."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("[teardown] zombie-context scan skipped (nvidia-smi unavailable: %s)", e)
        return []
    if out.returncode != 0:
        logger.debug("[teardown] zombie-context scan skipped (nvidia-smi rc=%d)", out.returncode)
        return []
    zombies: list[int] = []
    for line in out.stdout.splitlines():
        s = line.strip()
        if not s or not s.isdigit():
            continue
        pid = int(s)
        if not Path(f"/proc/{pid}").exists():
            # Dead-but-still-listed = an unreaped CUDA context from a crashed app.
            logger.warning("[teardown] WARN: zombie GPU context held by dead PID %d", pid)
            zombies.append(pid)
    if zombies:
        logger.warning(
            "[teardown] %d zombie CUDA context(s) held by dead PID(s) %s — cannot be "
            "killed (pid gone); a process-wide GPU reset is the only deterministic "
            "recovery. The next engine init may find too little free HBM.",
            len(zombies),
            zombies,
        )
    return zombies


def _reap_vllm_engine(llm) -> None:
    """Synchronously reap a vLLM ``LLM`` engine's worker subprocess + GPU memory.

    vLLM v1's EngineCore runs in a SEPARATE process (``(EngineCore_DP0 pid=...)``
    in the log). A bare ``del llm`` does NOT reap that subprocess synchronously,
    so its reserved KV cache (~gpu_memory_utilization of HBM) stays pinned until
    the OS eventually collects it — long enough that the NEXT ``LLM(...)`` init
    finds too little free memory and raises ``ValueError: Free memory ... less
    than desired gpu_memory_utilization`` (issue #653 dx phase, cell 2). The log
    also carries the canary ``destroy_process_group() was not called`` warning.

    This helper drives the documented teardown explicitly BEFORE ``del llm``:
    shut down the engine-core client (v1 ``llm_engine.engine_core.shutdown()``
    reaps the MP worker; v0 fallback ``model_executor.shutdown()``), destroy the
    torch.distributed process group if one was initialized, then leave the caller
    to ``del``/``gc.collect()``/``empty_cache()``/``ipc_collect()``/sleep. Every
    attribute access is ``getattr``-guarded so the helper NO-OPs gracefully on an
    API surface that differs (e.g. an in-process engine with no subprocess), and
    ``destroy_process_group()`` is guarded by ``is_initialized()`` so it NO-OPs
    when no group was created (the off-pod / single-GPU case).
    """
    engine = getattr(llm, "llm_engine", None)
    if engine is not None:
        # vLLM v1: the EngineCore lives behind an EngineCoreClient whose
        # shutdown() reaps the worker subprocess (MPClient._finalizer).
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        else:
            # vLLM v0 fallback: the model_executor owns the workers directly.
            executor = getattr(engine, "model_executor", None)
            exec_shutdown = getattr(executor, "shutdown", None)
            if callable(exec_shutdown):
                exec_shutdown()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # Fix 2 (#734 r5): surface any zombie CUDA context left by a crashed prior
    # engine co-residing with the next init. Visibility only (dead pids cannot be
    # signalled); guarded + bounded so it never delays teardown.
    _log_zombie_cuda_contexts()


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019, arXiv 1905.00414).

    Feature-space (linear-kernel) HSIC form, O(n d^2): column-center both
    matrices, then

        HSIC_lin(X, Y) = ||X_c^T Y_c||_F^2
        CKA            = HSIC_lin(X, Y) / sqrt(HSIC_lin(X, X) * HSIC_lin(Y, Y))

    Invariant to orthogonal transforms of either argument and to isotropic
    scaling; designed for the ``d > n`` regime where CCA degenerates. Returns
    a scalar in ``[0, 1]`` (1 = identical up to those invariances).

    Args:
        X: ``(n, d_x)`` tensor (n paired rows shared with ``Y``).
        Y: ``(n, d_y)`` tensor — same ``n`` as ``X``; ``d_y`` may differ.

    Returns:
        Linear CKA as a Python float.

    Raises:
        AssertionError: if ``X`` and ``Y`` disagree on ``n``, are not 2-D, or
            ``n < 2`` (column-centering a single row gives an all-zero matrix
            and a 0/0 CKA — a degenerate input, never a silent NaN).
    """
    assert X.ndim == 2 and Y.ndim == 2, (X.shape, Y.shape)
    assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
    n = X.shape[0]
    assert n >= 2, f"linear_cka needs n>=2 paired rows, got n={n}"

    # Compute in float64 for numerical stability (the Frobenius products are
    # sums of d^2 terms; fp32 accumulation drifts the invariance properties).
    Xc = X.to(torch.float64)
    Yc = Y.to(torch.float64)
    Xc = Xc - Xc.mean(dim=0, keepdim=True)
    Yc = Yc - Yc.mean(dim=0, keepdim=True)

    # HSIC_lin(A, B) = ||A^T B||_F^2 = sum((A^T B)^2).
    hsic_xy = (Xc.T @ Yc).pow(2).sum()
    hsic_xx = (Xc.T @ Xc).pow(2).sum()
    hsic_yy = (Yc.T @ Yc).pow(2).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom <= 0:
        # A constant bank (zero variance after centering) has no geometry to
        # align — return 0.0 rather than NaN. n>=2 is already asserted, so this
        # only fires on a genuinely degenerate (all-identical-rows) input.
        return 0.0
    return float((hsic_xy / denom).clamp(0.0, 1.0).item())


def cka_per_layer(bank_a: torch.Tensor, bank_b: torch.Tensor) -> list[float]:
    """Per-layer linear CKA between two layer-stacked activation banks.

    Args:
        bank_a: ``(n, n_layers, hidden)`` tensor.
        bank_b: ``(n, n_layers, hidden)`` tensor — same ``(n, n_layers)`` as
            ``bank_a`` (hidden may differ, though it never does in practice).

    Returns:
        ``[linear_cka(bank_a[:, L], bank_b[:, L]) for L in range(n_layers)]``.
    """
    assert bank_a.ndim == 3 and bank_b.ndim == 3, (bank_a.shape, bank_b.shape)
    assert bank_a.shape[:2] == bank_b.shape[:2], (bank_a.shape, bank_b.shape)
    n_layers = bank_a.shape[1]
    return [linear_cka(bank_a[:, L], bank_b[:, L]) for L in range(n_layers)]


def _build_generation_prompts(
    tokenizer,
    personas: dict[str, str | None],
    questions: list[str],
    *,
    user_wraps: dict[str, str | None] | None = None,
    prior_turns: dict[str, tuple] | None = None,
) -> tuple[list[str], list[tuple[str, int]]]:
    """Rendered chat prompts + (persona, question_idx) keys for every pair.

    ``user_wraps`` maps a persona/context key to an optional ``"...{q}..."``
    user-turn wrap (the ``NegativeContext.user_wrap`` shape): when set, the
    user content is ``wrap.format(q=question)`` — the SAME rendering the
    span computation (``compute_prompt_spans``) re-derives, so generation and
    span alignment share one message construction (#1112 round-2 Critical 1:
    a wrap member generated on the BARE question tripped the span
    token-prefix assert AND degenerated to the bare-assistant context).

    ``prior_turns`` maps a persona/context key to an optional tuple of frozen
    ``{"role", "content"}`` conversation turns rendered BETWEEN the system
    prompt and the final user turn (the ``Context.prefix_turns`` shape — the
    #1315 WildChat two-turn prefix). Defaults preserve single-turn behavior
    byte-identically.
    """
    prompts: list[str] = []
    keys: list[tuple[str, int]] = []
    wraps = user_wraps or {}
    priors = prior_turns or {}
    for p_name, p_prompt in personas.items():
        wrap = wraps.get(p_name)
        prior = priors.get(p_name) or ()
        for q_idx, question in enumerate(questions):
            messages = []
            if p_prompt:
                messages.append({"role": "system", "content": p_prompt})
            for turn in prior:
                assert turn.get("role") in ("user", "assistant") and turn.get("content"), turn
                messages.append({"role": turn["role"], "content": turn["content"]})
            content = wrap.format(q=question) if wrap else question
            messages.append({"role": "user", "content": content})
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            keys.append((p_name, q_idx))
    return prompts, keys


def _generate_responses_vllm(
    model_path: str,
    personas: dict[str, str | None],
    questions: list[str],
    *,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    user_wraps: dict[str, str | None] | None = None,
    prior_turns: dict[str, tuple] | None = None,
) -> list[dict]:
    """vLLM greedy generation for every (persona, question) pair.

    Returns one row dict per pair: ``{persona, question_idx, prompt_token_ids,
    response_token_ids, finish_reason}``. The vLLM engine is torn down before
    returning so the subsequent HF teacher-forced pass has the GPU to itself.
    ``user_wraps`` / ``prior_turns`` thread per-context user-turn wraps and
    frozen multi-turn prefixes into the prompt build
    (see :func:`_build_generation_prompts`).
    """
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    prompts, keys = _build_generation_prompts(
        tokenizer, personas, questions, user_wraps=user_wraps, prior_turns=prior_turns
    )

    # enforce_eager defaults TRUE (#734 crash-fix round 5): cuda-graph capture
    # deadlocked the first generate() on the pod-734 combo. Env-overridable via
    # EPM_VLLM_ENFORCE_EAGER=0 (_vllm_enforce_eager).
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=_vllm_enforce_eager(),
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    # use_tqdm=False per gotchas.md #613 RULE (every LLM.generate() call site):
    # vLLM 0.11.0's tqdm wrapper raises ZeroDivisionError when a batch finishes
    # faster than the elapsed-time tick. Required on every generate() call.
    outputs = llm.generate(prompts, params, use_tqdm=False)
    assert len(outputs) == len(keys), (len(outputs), len(keys))

    eos_id = tokenizer.eos_token_id
    rows: list[dict] = []
    for (p_name, q_idx), out in zip(keys, outputs, strict=True):
        completion = out.outputs[0]
        resp_ids = list(completion.token_ids)
        # Strip a single trailing EOS so the pool covers response CONTENT
        # tokens only (recorded recipe choice; vLLM includes the stop token
        # in token_ids when finish_reason == "stop").
        if resp_ids and resp_ids[-1] == eos_id:
            resp_ids = resp_ids[:-1]
        rows.append(
            {
                "persona": p_name,
                "question_idx": q_idx,
                "prompt_token_ids": list(out.prompt_token_ids),
                "response_token_ids": resp_ids,
                "finish_reason": completion.finish_reason,
            }
        )

    # vLLM teardown so the next engine load / HF pass can allocate (see gotchas:
    # worker teardown). The bare ``del llm; gc; empty_cache`` triad is NOT enough
    # for vLLM v1 — its EngineCore is a subprocess that ``del`` does not reap
    # synchronously, leaking the reserved KV cache and crashing the next
    # ``LLM(...)`` init (issue #653 dx phase). Reap the worker explicitly first.
    _reap_vllm_engine(llm)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # complement to empty_cache for inter-process freed mem
    time.sleep(1.0)  # conservative: vLLM subprocess teardown is async
    return rows


def _teacher_forced_response_mean(
    model_path: str,
    rows: list[dict],
    persona_names: list[str],
    layers: list[int],
    *,
    device: str,
    dtype: torch.dtype,
    tf_batch_size: int,
) -> dict[int, dict[str, list[torch.Tensor]]]:
    """Batched HF teacher-forced forwards, mean-pooled over response tokens.

    Returns ``pooled[layer][persona] -> list`` of per-question pooled vectors
    (float32 cpu), in row order. Pooling is GPU-resident; only the pooled
    (hidden_dim,) vectors cross to CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]

    pooled: dict[int, dict[str, list[torch.Tensor]]] = {
        li: {p: [] for p in persona_names} for li in layers
    }
    for start in range(0, len(rows), tf_batch_size):
        batch = rows[start : start + tf_batch_size]
        seqs = [r["prompt_token_ids"] + r["response_token_ids"] for r in batch]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attn)
        for li in layers:
            hs = captured[li]
            assert hs.shape[:2] == (len(batch), max_len), (hs.shape, len(batch), max_len)
            for i, r in enumerate(batch):
                p_len = len(r["prompt_token_ids"])
                span_end = p_len + len(r["response_token_ids"])
                vec = hs[i, p_len:span_end, :].float().mean(dim=0).cpu()
                pooled[li][r["persona"]].append(vec)
        if (start // tf_batch_size) % 20 == 0:
            print(
                f"[respmean] TF batch {start // tf_batch_size + 1}/{-(-len(rows) // tf_batch_size)}"
            )

    # Thorough teardown so a SUBSEQUENT vLLM init (the next dispatcher-loop
    # iteration's _generate_responses_vllm) sees the GPU as free. The bf16 7B
    # weights are ~14.25 GiB and the per-batch hook ``captured`` dict pins
    # detached GPU hidden-state tensors; a bare ``del model`` leaves both in the
    # allocator's reserved pool, so vLLM (which computes its target as
    # gpu_memory_utilization x TOTAL, NOT x FREE) aborts with "Free memory ...
    # less than desired gpu_memory_utilization". Clear EVERY GPU reference,
    # collect, empty the cache, ipc_collect (cross-process freed mem), and sleep
    # to let any async free settle. (issue #685: 2nd dispatcher-loop behavior
    # crashed at vLLM init with ~16.5 GiB held by behavior 1's HF model.)
    for h in hooks:
        h.remove()
    captured.clear()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # release cross-process (subprocess) freed mem
        time.sleep(1.0)  # let any async free settle before the next vLLM init
    return pooled


SPAN_ARMS = ("prefix", "context", "response")


def compute_prompt_spans(
    tokenizer,
    system_prompt: str | None,
    question: str,
    prompt_token_ids: list[int],
    *,
    prior_messages: list[dict] | tuple | None = None,
    user_wrap: str | None = None,
    prefix_end: str = "first_user",
    on_seam: str = "raise",
    seam_flags: dict | None = None,
) -> tuple[int, int]:
    """(prefix_len, context_len) token boundaries inside ``prompt_token_ids``.

    Canonical definitions (#1112 / the standing prefix+context mapping rule):
    the PREFIX is every token strictly before the user QUERY (system/persona
    prompt + chat-template preamble + any conversation content preceding the
    query); the CONTEXT is the prefix plus the user query (tokens up to the
    END of the question text, excluding the post-question template tail
    ``<|im_end|>...assistant``).

    #1315 multi-turn extension (source-level, default-preserving):

    - ``prior_messages``: frozen ``{"role", "content"}`` turns rendered
      between the system prompt and the final user turn (the WildChat
      two-turn prefix shape). Requires ``prefix_end='last_user'``.
    - ``user_wrap``: an optional ``"...{q}..."`` wrap for the FINAL user
      turn (the ICL two-shot block shape). The rendered final user content
      is ``user_wrap.format(q=question)``; with ``prefix_end='last_user'``
      the wrap text preceding the query joins the PREFIX arm (the standing
      rule: prefix = everything before the user query). Requires
      ``prefix_end='last_user'``.
    - ``prefix_end``: ``'first_user'`` (default — byte-identical to the
      pre-#1315 behavior; single-turn only) or ``'last_user'`` (the prefix
      boundary sits at the start of the FINAL user message's QUERY text).

    Boundaries are located by CHAR offset in the rendered chat-template text,
    then mapped to token indices via the tokenizer's OFFSET MAPPING (the
    established teacher-forced-capture recipe — gotchas.md "Teacher-forced
    capture inputs", #1092): the full render is tokenized ONCE and asserted
    token-identical to ``prompt_token_ids`` (generation and span computation
    must share one render + tokenizer — genuine drift stays fail-loud), then
    each char boundary is resolved against the token offsets. A boundary that
    falls INSIDE a token is a **BPE merge seam** (a plain-text char before the
    query merged into the query's first token — e.g. a ``"... {q}"`` wrap's
    trailing space merging into ``" How"``; deterministic for the
    ``neg_reph_curious`` wrap under ``prefix_end='last_user'``, #1315 r7):

    - ``on_seam='raise'`` (default — the pre-#1315-r7 contract, unchanged for
      #1112 callers): AssertionError, fail-loud.
    - ``on_seam='snap'``: the documented seam policy. The PREFIX boundary
      EXCLUDES the straddling token (its hidden state has consumed query
      text — including it would leak the query into the prefix arm); the
      CONTEXT boundary INCLUDES a straddler (the context arm must contain the
      whole query). On exact (non-seam) rows ``snap`` is token-identical to
      ``raise``. When ``seam_flags`` (a dict) is passed, per-boundary
      provenance is recorded into it: ``{"prefix": bool, "context": bool}``.

    Raises:
        AssertionError: prefix span empty, boundary not found, rendered-text
            tokenization diverging from the generated prompt ids, a BPE
            boundary seam under ``on_seam='raise'``, or multi-turn inputs
            without ``prefix_end='last_user'``.
    """
    assert on_seam in ("raise", "snap"), on_seam
    assert prefix_end in ("first_user", "last_user"), prefix_end
    prior = list(prior_messages or [])
    if prior or user_wrap is not None:
        assert prefix_end == "last_user", (
            "multi-turn prior_messages / user_wrap spans require the explicit "
            f"prefix_end='last_user' opt-in (got {prefix_end!r})"
        )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in prior:
        assert turn.get("role") in ("user", "assistant") and turn.get("content"), turn
        messages.append({"role": turn["role"], "content": turn["content"]})
    final_content = user_wrap.format(q=question) if user_wrap else question
    messages.append({"role": "user", "content": final_content})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # The question's char span: search AFTER the system prompt region AND
    # after every prior turn's content, so a question substring accidentally
    # present in the persona text / prior turns cannot match.
    search_from = 0
    if system_prompt:
        sys_pos = text.find(system_prompt)
        assert sys_pos >= 0, "system prompt not found in rendered chat template"
        search_from = sys_pos + len(system_prompt)
    for turn in prior:
        t_pos = text.find(turn["content"], search_from)
        assert t_pos >= 0, f"prior {turn['role']} turn not found in rendered chat template"
        search_from = t_pos + len(turn["content"])
    if user_wrap is not None:
        # Anchor to the FINAL user content so the query is located INSIDE it
        # (the ICL block precedes the query within the same turn).
        fc_pos = text.find(final_content, search_from)
        assert fc_pos >= 0, "wrapped final user content not found in rendered chat template"
        rel_q = final_content.find(question)
        assert rel_q >= 0, "question not found inside user_wrap-rendered content"
        search_from = fc_pos + rel_q  # find() below matches exactly here
    q_start = text.find(question, search_from)
    assert q_start >= 0, f"question not found in rendered template (from char {search_from})"
    q_end = q_start + len(question)

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if list(prompt_token_ids) != list(ids):
        div = next(
            (i for i, (a, b) in enumerate(zip(ids, prompt_token_ids, strict=False)) if a != b),
            min(len(ids), len(prompt_token_ids)),
        )
        raise AssertionError(
            "rendered-template tokenization does not match the generated prompt ids "
            f"({len(ids)} vs {len(prompt_token_ids)} tokens; first divergence at index {div}) "
            "— generation and span computation must share one render + tokenizer"
        )
    assert all(s < e for s, e in offsets), "zero-width token offsets — unsupported tokenizer"

    def _boundary(char_end: int, tag: str, *, include_straddler: bool) -> int:
        n_inside = sum(1 for _, e in offsets if e <= char_end)  # tokens fully before boundary
        straddler = n_inside < len(offsets) and offsets[n_inside][0] < char_end
        if straddler and on_seam == "raise":
            s, e = offsets[n_inside]
            raise AssertionError(
                f"{tag} boundary BPE drift: token {n_inside} (id {ids[n_inside]}) spans "
                f"chars [{s}, {e}) across the boundary at {char_end} — a plain-text char "
                "before the boundary merged into the next segment's first token "
                "(span-validate the row, or opt into on_seam='snap')"
            )
        if seam_flags is not None:
            seam_flags[tag] = bool(straddler)
        return n_inside + (1 if (straddler and include_straddler) else 0)

    prefix_len = _boundary(q_start, "prefix", include_straddler=False)
    context_len = _boundary(q_end, "context", include_straddler=True)
    assert 0 < prefix_len < context_len <= len(prompt_token_ids), (
        prefix_len,
        context_len,
        len(prompt_token_ids),
    )
    return prefix_len, context_len


def _teacher_forced_span_means(
    model_path: str,
    rows: list[dict],
    persona_names: list[str],
    layers: list[int],
    *,
    spans: tuple[str, ...] = SPAN_ARMS,
    device: str,
    dtype: torch.dtype,
    tf_batch_size: int,
) -> dict[str, dict[int, torch.Tensor]]:
    """Batched teacher-forced forwards, span-pooled at every requested layer.

    The #1112 sibling of :func:`_teacher_forced_response_mean` — same batched
    HF forward + per-layer hooks + GPU-resident pooling, extended to return
    THREE pooled vectors per row (prefix / context / response spans, see
    :func:`compute_prompt_spans`). Each row dict must carry
    ``prompt_token_ids``, ``response_token_ids``, ``prefix_len``,
    ``context_len``, and ``persona`` (``persona_names`` pins the expected
    context panel; an unknown persona fails loud).

    Returns:
        ``{span: {layer: Tensor(n_rows, hidden) float32 cpu}}`` in ROW order.
    """
    for span in spans:
        assert span in SPAN_ARMS, (span, SPAN_ARMS)
    known = set(persona_names)
    for i, r in enumerate(rows):
        assert r["persona"] in known, (i, r["persona"])
        p_len = len(r["prompt_token_ids"])
        assert 0 < r["prefix_len"] < r["context_len"] <= p_len, (
            i,
            r["prefix_len"],
            r["context_len"],
            p_len,
        )
        assert len(r["response_token_ids"]) > 0, f"row {i} has an empty response span"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    n_blocks = len(model.model.layers)
    for li in layers:
        assert 0 <= li < n_blocks, (li, n_blocks)
    hooks = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]

    hidden = model.config.hidden_size
    pooled: dict[str, dict[int, list[torch.Tensor]]] = {
        span: {li: [] for li in layers} for span in spans
    }
    n_batches = -(-len(rows) // tf_batch_size)
    for start in range(0, len(rows), tf_batch_size):
        batch = rows[start : start + tf_batch_size]
        seqs = [r["prompt_token_ids"] + r["response_token_ids"] for r in batch]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        with torch.no_grad():
            # Right-padded batch: positions index naturally from 0 per row, so
            # no explicit position_ids needed (the left-pad trap does not
            # apply); logits are unread — pass logits_to_keep=1 when supported
            # to skip the full-vocab lm_head materialization (gotchas.md #779).
            import inspect

            fwd = getattr(model, "forward", model.__call__)
            kwargs = {}
            if "logits_to_keep" in inspect.signature(fwd).parameters:
                kwargs["logits_to_keep"] = 1
            _ = model(input_ids=input_ids, attention_mask=attn, **kwargs)
        for li in layers:
            hs = captured[li]
            assert hs.shape[:2] == (len(batch), max_len), (hs.shape, len(batch), max_len)
            for i, r in enumerate(batch):
                p_len = len(r["prompt_token_ids"])
                span_bounds = {
                    "prefix": (0, r["prefix_len"]),
                    "context": (0, r["context_len"]),
                    "response": (p_len, p_len + len(r["response_token_ids"])),
                }
                for span in spans:
                    s, e = span_bounds[span]
                    vec = hs[i, s:e, :].float().mean(dim=0).cpu()
                    assert vec.shape == (hidden,), (vec.shape, hidden)
                    pooled[span][li].append(vec)
        if (start // tf_batch_size) % 20 == 0:
            print(f"[spanmeans] TF batch {start // tf_batch_size + 1}/{n_batches}")

    for h in hooks:
        h.remove()
    captured.clear()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(1.0)

    return {span: {li: torch.stack(pooled[span][li]) for li in layers} for span in spans}


def extract_centroids_response_mean(
    model_path: str,
    personas: dict[str, str | None],
    questions: list[str] | None = None,
    layers: list[int] | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    *,
    max_new_tokens: int = 1024,
    tf_batch_size: int = 8,
    gpu_memory_utilization: float = 0.5,
    responses_cache_path: str | Path | None = None,
) -> tuple[dict[int, torch.Tensor], list[str], dict]:
    """Recipe (b) of persona-distance-metrics.md: mean over OWN response tokens.

    Two phases (#483 plan section 3.3):

    1. vLLM greedy generation (temp=0, ``max_new_tokens``) of one response per
       (persona, question) pair; truncation rate is logged and returned (must
       be ~0 per the #548 manipulation-check discipline). Responses are
       persisted to ``responses_cache_path`` the moment generation completes
       (checkpoint-per-phase); an existing cache is reloaded and generation
       skipped (idempotent re-runs).
    2. Batched HF teacher-forced forwards over ``prompt_ids + response_ids``
       (token ids concatenated exactly as generated - no re-tokenization
       boundary drift), mean-pooling hidden states over the RESPONSE tokens
       only at each requested layer, then mean over questions per persona.
       Pooling stays GPU-resident in float32; only pooled vectors move to CPU.

    Args:
        model_path: HF model id/path.
        personas: {name: system_prompt}; falsy prompt = no system message.
        questions: defaults to DEFAULT_QUESTIONS (20 EVAL_QUESTIONS).
        layers: 0-indexed decoder blocks; defaults to [20, 21].
        device: device for the teacher-forced pass.
        dtype: model dtype for the teacher-forced pass.
        max_new_tokens: vLLM generation cap (truncation rate reported).
        tf_batch_size: rows per teacher-forced batch.
        gpu_memory_utilization: vLLM engine memory fraction (FRACTION OF TOTAL,
            not of free). Default 0.5 leaves headroom so that a vLLM init in a
            SUBSEQUENT call within the same process (the per-behavior dispatcher
            loop) does not collide with any lagging allocator reservation from
            this call's HF teacher-forced model, and so the run is memory-safe on
            smaller GPUs (e.g. L4 22 GiB). 0.5 = ~11 GiB on L4, ~40 GiB on H100
            — ample for KV cache on a 7B model. The HF model is torn down before
            vLLM loads (phases are sequential; see ``_teacher_forced_response_mean``
            teardown), so the fraction need not also fit the HF weights. (#685)
        responses_cache_path: JSON checkpoint for phase 1.

    Returns:
        ``(centroids, persona_names, stats)`` - centroids is
        {layer: Tensor(n_personas, hidden_dim) float32 cpu}; stats carries
        ``truncation_rate``, ``n_rows``, ``mean_response_tokens``,
        ``max_new_tokens``.

    Raises:
        RuntimeError: if any generated response has zero content tokens (a
        NaN centroid would silently poison the bank - fail fast instead).
    """
    if questions is None:
        questions = DEFAULT_QUESTIONS
    if layers is None:
        layers = [20, 21]
    persona_names = list(personas.keys())

    cache = Path(responses_cache_path) if responses_cache_path else None
    if cache is not None and cache.exists():
        import json

        rows = json.loads(cache.read_text())["rows"]
        print(f"[respmean] loaded {len(rows)} cached responses from {cache}")
    else:
        rows = _generate_responses_vllm(
            model_path,
            personas,
            questions,
            max_new_tokens=max_new_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if cache is not None:
            import json

            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps({"model": model_path, "max_new_tokens": max_new_tokens, "rows": rows})
            )
            print(f"[respmean] checkpointed {len(rows)} responses to {cache}")

    n_truncated = sum(1 for r in rows if r["finish_reason"] == "length")
    truncation_rate = n_truncated / max(len(rows), 1)
    empty = [(r["persona"], r["question_idx"]) for r in rows if not r["response_token_ids"]]
    if empty:
        raise RuntimeError(
            f"[respmean] {len(empty)} (persona, question) rows generated ZERO content tokens "
            f"(first 5: {empty[:5]}) - cannot mean-pool an empty response; investigate."
        )
    resp_lens = [len(r["response_token_ids"]) for r in rows]
    stats = {
        "truncation_rate": truncation_rate,
        "n_rows": len(rows),
        "mean_response_tokens": sum(resp_lens) / len(resp_lens),
        "max_new_tokens": max_new_tokens,
    }
    print(
        f"[respmean] {len(rows)} responses; truncation_rate={truncation_rate:.4f}, "
        f"mean_response_tokens={stats['mean_response_tokens']:.1f}"
    )

    pooled = _teacher_forced_response_mean(
        model_path,
        rows,
        persona_names,
        layers,
        device=device,
        dtype=dtype,
        tf_batch_size=tf_batch_size,
    )

    centroids: dict[int, torch.Tensor] = {}
    for li in layers:
        per_persona = []
        for p in persona_names:
            vecs = pooled[li][p]
            assert len(vecs) == len(questions), (p, li, len(vecs), len(questions))
            per_persona.append(torch.stack(vecs).mean(dim=0))
        centroids[li] = torch.stack(per_persona)

    print(
        f"[respmean] extracted response-mean centroids: {len(persona_names)} personas "
        f"x {len(layers)} layers"
    )
    return centroids, persona_names, stats


def compute_representation_shifts(
    base_centroids: dict[int, torch.Tensor],
    phase1_centroids: dict[int, torch.Tensor],
    phase2_centroids: dict[int, torch.Tensor] | None,
    persona_names: list[str],
    source_persona: str,
    assistant_name: str = "assistant",
) -> dict:
    """Compute representation shift metrics across training phases.

    Returns a dict with per-layer metrics including:
    - Cosine(source, assistant) at each phase
    - L2 shift magnitudes for source, assistant, bystanders
    - Shift direction alignment (cosine between shift vectors)
    - Projection of shifts onto the base-model source→assistant axis

    Args:
        base_centroids: {layer: (n, d)} from base model.
        phase1_centroids: {layer: (n, d)} after marker implantation.
        phase2_centroids: {layer: (n, d)} after Phase 2 SFT. None to skip.
        persona_names: Ordered list matching centroid tensor rows.
        source_persona: Name of the source persona that received the marker.
        assistant_name: Name of the assistant persona in persona_names.
    """
    src_idx = persona_names.index(source_persona)
    asst_idx = persona_names.index(assistant_name)
    bystander_idxs = [i for i in range(len(persona_names)) if i not in (src_idx, asst_idx)]

    results = {"source_persona": source_persona, "layers": {}}

    for layer in base_centroids:
        base = base_centroids[layer]
        p1 = phase1_centroids[layer]
        p2 = phase2_centroids[layer] if phase2_centroids else None

        # Base cosines
        base_cos = F.cosine_similarity(
            base[src_idx].unsqueeze(0), base[asst_idx].unsqueeze(0)
        ).item()
        p1_cos = F.cosine_similarity(p1[src_idx].unsqueeze(0), p1[asst_idx].unsqueeze(0)).item()

        # Shift vectors (base → phase1)
        src_shift = p1[src_idx] - base[src_idx]
        asst_shift = p1[asst_idx] - base[asst_idx]

        src_shift_l2 = src_shift.norm().item()
        asst_shift_l2 = asst_shift.norm().item()

        # Bystander shifts
        bystander_shifts = [(p1[i] - base[i]).norm().item() for i in bystander_idxs]
        bystander_mean_l2 = sum(bystander_shifts) / max(len(bystander_shifts), 1)

        # Shift direction alignment
        if src_shift_l2 > 1e-8 and asst_shift_l2 > 1e-8:
            shift_alignment = F.cosine_similarity(
                src_shift.unsqueeze(0), asst_shift.unsqueeze(0)
            ).item()
        else:
            shift_alignment = 0.0

        # Projection of shifts onto base-model source→assistant axis
        base_axis = base[asst_idx] - base[src_idx]
        axis_norm = base_axis.norm()
        if axis_norm > 1e-8:
            base_axis_unit = base_axis / axis_norm
            src_proj = torch.dot(src_shift, base_axis_unit).item()
            asst_proj = torch.dot(asst_shift, base_axis_unit).item()
        else:
            src_proj = 0.0
            asst_proj = 0.0

        # Centered cosine matrices
        base_centered_cos = compute_cosine_matrix(base, centering="global_mean")
        p1_centered_cos = compute_cosine_matrix(p1, centering="global_mean")

        layer_result = {
            "base_cos_source_asst": base_cos,
            "phase1_cos_source_asst": p1_cos,
            "cos_delta_phase1": p1_cos - base_cos,
            "source_shift_l2": src_shift_l2,
            "assistant_shift_l2": asst_shift_l2,
            "bystander_mean_shift_l2": bystander_mean_l2,
            "bystander_shifts": {
                persona_names[i]: bystander_shifts[j] for j, i in enumerate(bystander_idxs)
            },
            "shift_direction_alignment": shift_alignment,
            "source_proj_on_src_asst_axis": src_proj,
            "assistant_proj_on_src_asst_axis": asst_proj,
            "base_centered_cos_src_asst": base_centered_cos[src_idx, asst_idx].item(),
            "phase1_centered_cos_src_asst": p1_centered_cos[src_idx, asst_idx].item(),
        }

        # Phase 2 metrics (if available)
        if p2 is not None:
            p2_cos = F.cosine_similarity(p2[src_idx].unsqueeze(0), p2[asst_idx].unsqueeze(0)).item()

            # Phase 1 → Phase 2 shifts
            src_shift_p2 = p2[src_idx] - p1[src_idx]
            asst_shift_p2 = p2[asst_idx] - p1[asst_idx]

            # Total shift (base → Phase 2)
            src_total_shift = p2[src_idx] - base[src_idx]
            asst_total_shift = p2[asst_idx] - base[asst_idx]

            p2_centered_cos = compute_cosine_matrix(p2, centering="global_mean")

            layer_result.update(
                {
                    "phase2_cos_source_asst": p2_cos,
                    "cos_delta_phase2": p2_cos - p1_cos,
                    "cos_delta_total": p2_cos - base_cos,
                    "source_shift_p1_to_p2_l2": src_shift_p2.norm().item(),
                    "assistant_shift_p1_to_p2_l2": asst_shift_p2.norm().item(),
                    "source_total_shift_l2": src_total_shift.norm().item(),
                    "assistant_total_shift_l2": asst_total_shift.norm().item(),
                    "phase2_centered_cos_src_asst": p2_centered_cos[src_idx, asst_idx].item(),
                }
            )

        results["layers"][f"layer_{layer}"] = layer_result

    return results


def save_centroids(
    centroids: dict[int, torch.Tensor],
    persona_names: list[str],
    output_path: str | Path,
) -> None:
    """Save centroids and persona names to a .pt file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": centroids, "persona_names": persona_names}, output_path)
    print(f"Saved centroids to {output_path}")


def load_centroids(path: str | Path) -> tuple[dict[int, torch.Tensor], list[str]]:
    """Load centroids from a .pt file."""
    data = torch.load(path, weights_only=True)
    return data["centroids"], data["persona_names"]
