#!/usr/bin/env python3
"""System-slot GCG against a fresh-per-step EM-distribution target.

Vendored minimal GCG implementation (Wallace+2019 / Shin+2020 / Zou+2023):
* one-hot gradient over the L candidate token positions in the system slot
  via straight-through estimator,
* top-k token candidates per position from the negative gradient,
* batched candidate evaluation (random subset of ``search_width`` per step),
* greedy swap to the candidate with the lowest mean teacher-forced CE on
  fresh EM completions.

Per plan §4.7:
* L ∈ {20, 40, 80}; search_width=512, topk=256.
* Steps: 500 for L=20/40, 1000 for L=80 (R8) with a train-CE smoothness
  kill gate at step 250.
* Slot: ``chat_template_position="system"`` for the main sweep;
  ``"user"`` for the GCG sanity reproducer of #94.
* Target: fresh EM completions resampled at every step (16 Qs x 20
  completions). Per-step CE = mean teacher-forced CE on assistant content
  given the candidate suffix in the chosen slot.

Outputs:
* ``eval_results/issue-170/<cell>/best_suffix.json`` — best L tokens at each
  step + final.
* ``eval_results/issue-170/<cell>/train_curve.json``.
* ``eval_results/issue-170/<cell>/run_result.json``.

Throughout, "candidate" refers to a single trial sequence of L token ids;
"swap" refers to replacing the suffix with the best candidate when its CE
is strictly lower than the current.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

bootstrap()

# hot-fix: transformers 5.5 removed all_special_tokens_extended; vLLM 0.11 needs it
from transformers import PreTrainedTokenizerBase  # noqa: E402

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = PreTrainedTokenizerBase.all_special_tokens

import hydra  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import wandb  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from explore_persona_space.axis.prompt_search.em_completion_server import (  # noqa: E402
    EMCompletionServer,
    ensure_local_em_snapshot,
)
from explore_persona_space.utils import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


# ── Suffix-aware chat-template assembly ─────────────────────────────────────


def build_full_inputs_for_ce(
    tokenizer,
    suffix_ids: list[int],
    questions: list[str],
    completions_per_q: dict[str, list[str]],
    slot: str = "system",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble (input_ids, attention_mask, completion_mask) for a candidate suffix.

    Reuses ``build_full_ce_input_ids`` from ``soft_prefix.py`` to build the
    K=L placeholder slot via direct token-id assembly (avoiding BPE merging
    of solid character runs); afterwards splices ``suffix_ids`` in place of
    those placeholders.

    Returns:
        Tuple (input_ids, attention_mask, completion_mask), all on CPU.
    """
    from explore_persona_space.axis.prompt_search.soft_prefix import (
        QWEN_PLACEHOLDER_TOKEN,
        build_full_ce_input_ids,
    )

    L = len(suffix_ids)
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    if QWEN_PLACEHOLDER_TOKEN in added:
        placeholder_id = int(added[QWEN_PLACEHOLDER_TOKEN])
    else:
        ids = tokenizer.encode(QWEN_PLACEHOLDER_TOKEN, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Placeholder special {QWEN_PLACEHOLDER_TOKEN!r} tokenised to {len(ids)} tokens; "
                "expected exactly 1."
            )
        placeholder_id = ids[0]

    input_ids, attention_mask, completion_mask = build_full_ce_input_ids(
        tokenizer,
        placeholder_id=placeholder_id,
        K=L,
        questions=questions,
        completions_per_q=completions_per_q,
        slot=slot,
        max_length=2048,
        device="cpu",
    )

    # Splice suffix_ids into the placeholder run on every row.
    suffix_tensor = torch.tensor(suffix_ids, dtype=input_ids.dtype)
    for b in range(input_ids.shape[0]):
        run = _find_placeholder_run(input_ids[b], placeholder_id, L)
        if run is None:
            raise RuntimeError(f"Row {b}: placeholder run of length {L} not found after assembly")
        start, end = run
        input_ids[b, start:end] = suffix_tensor

    return input_ids, attention_mask, completion_mask


def _find_placeholder_run(ids_row: torch.Tensor, placeholder_id: int, L: int):
    eq = (ids_row == placeholder_id).to(torch.int64)
    run_start = -1
    run_len = 0
    for t in range(eq.shape[0]):
        if eq[t].item() == 1:
            if run_start < 0:
                run_start = t
            run_len += 1
            if run_len == L:
                return (run_start, run_start + L)
        else:
            run_start = -1
            run_len = 0
    return None


# ── CE evaluator ────────────────────────────────────────────────────────────


def evaluate_suffix_ce(
    base_model,
    tokenizer,
    suffix_ids: list[int],
    questions: list[str],
    completions_per_q: dict[str, list[str]],
    *,
    slot: str,
    device: torch.device,
    return_loss_for_grad: bool = False,
) -> tuple[torch.Tensor, int]:
    """Mean teacher-forced CE for a single candidate suffix.

    Args:
        return_loss_for_grad: if True, returns a tensor with grad attached
            via the input embeddings (used for the one-hot gradient step).
            If False, runs under ``torch.no_grad()``.
    """
    input_ids, attention_mask, completion_mask = build_full_inputs_for_ce(
        tokenizer, suffix_ids, questions, completions_per_q, slot=slot
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    completion_mask = completion_mask.to(device)

    if return_loss_for_grad:
        embed = base_model.get_input_embeddings()
        inputs_embeds = embed(input_ids)
        outputs = base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    else:
        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    del outputs

    # Memory-efficient CE: views + ignore_index=-100 instead of contiguous() copy.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:]
    masked_labels = shift_labels.masked_fill(~shift_mask, -100)

    n_tokens = int(shift_mask.sum().item())
    if n_tokens == 0:
        if return_loss_for_grad:
            return torch.zeros((), device=device, requires_grad=True), 0
        return torch.zeros((), device=device), 0

    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = masked_labels.reshape(-1)
    mean_ce = F.cross_entropy(flat_logits, flat_labels, reduction="mean", ignore_index=-100)
    return mean_ce, n_tokens


# ── Batched candidate CE evaluation ──────────────────────────────────────────


def evaluate_candidates_batched(
    base_model,
    tokenizer,
    candidates: list[list[int]],
    questions: list[str],
    completions_per_q: dict[str, list[str]],
    *,
    slot: str,
    device: torch.device,
    micro_batch: int = 32,
) -> list[float]:
    """Evaluate all candidate suffixes in micro-batches for throughput.

    Instead of calling ``evaluate_suffix_ce`` 512 times (one model.forward per
    candidate), this function:

    1. Builds the template ``(input_ids, attention_mask, completion_mask)`` once
       using the first candidate (all candidates share identical structure and
       differ only in the suffix span).
    2. Finds the suffix span in each template row.
    3. For each micro-batch of M candidates, clones the template M times
       (shape ``M*B, T``), splices each candidate's suffix into its B rows,
       runs a single ``model.forward()``, and extracts per-candidate mean CE.

    This gives ~(512/micro_batch)x = ~16x fewer forward passes per GCG step.

    Args:
        base_model: Frozen causal LM on ``device``.
        tokenizer: Matching tokenizer.
        candidates: List of N candidate suffix token-id lists, each length L.
        questions: Batch of question strings.
        completions_per_q: {question: [completion_strings]}.
        slot: "system" or "user".
        device: CUDA device.
        micro_batch: Number of candidates per forward pass (default 32).
            Reduce to 16 or 8 if OOM.

    Returns:
        List of N float CE values, one per candidate.
    """
    if not candidates:
        return []

    L = len(candidates[0])
    n_candidates = len(candidates)

    # Build base template using the first candidate (structure is identical
    # across all candidates — only the suffix span content differs).
    base_input_ids, base_attn_mask, base_comp_mask = build_full_inputs_for_ce(
        tokenizer, candidates[0], questions, completions_per_q, slot=slot
    )
    B, T = base_input_ids.shape  # B = n_questions * n_completions

    # Find the suffix span in each template row. Since all candidates share
    # the same structure, the span positions are identical regardless of which
    # candidate was used to build the template.
    suffix_tensor = torch.tensor(candidates[0], dtype=base_input_ids.dtype)
    suffix_starts: list[int] = []
    for b in range(B):
        row = base_input_ids[b]
        found = -1
        for t in range(row.shape[0] - L + 1):
            if torch.equal(row[t : t + L], suffix_tensor):
                found = t
                break
        if found < 0:
            raise RuntimeError(
                f"Row {b}: suffix span not found in template. "
                "This should not happen if build_full_inputs_for_ce is correct."
            )
        suffix_starts.append(found)

    # Move templates to device once.
    base_input_ids = base_input_ids.to(device)
    base_attn_mask = base_attn_mask.to(device)
    base_comp_mask = base_comp_mask.to(device)

    all_ces: list[float] = []

    for mb_start in range(0, n_candidates, micro_batch):
        mb_end = min(mb_start + micro_batch, n_candidates)
        M = mb_end - mb_start  # actual micro-batch size (may be < micro_batch at end)

        # Clone base template M times: (M*B, T)
        mb_input_ids = base_input_ids.unsqueeze(0).expand(M, -1, -1).reshape(M * B, T).clone()
        mb_attn_mask = base_attn_mask.unsqueeze(0).expand(M, -1, -1).reshape(M * B, T)
        mb_comp_mask = base_comp_mask.unsqueeze(0).expand(M, -1, -1).reshape(M * B, T)

        # Splice each candidate's suffix into its B rows.
        for m_idx in range(M):
            cand_ids = candidates[mb_start + m_idx]
            cand_tensor = torch.tensor(cand_ids, dtype=mb_input_ids.dtype, device=device)
            for b in range(B):
                row_idx = m_idx * B + b
                s = suffix_starts[b]
                mb_input_ids[row_idx, s : s + L] = cand_tensor

        # Single forward pass for the entire micro-batch.
        with torch.no_grad():
            outputs = base_model(input_ids=mb_input_ids, attention_mask=mb_attn_mask)
        logits = outputs.logits
        del outputs

        # Compute per-candidate mean CE.
        shift_logits = logits[:, :-1, :]  # (M*B, T-1, V)
        shift_labels = mb_input_ids[:, 1:]  # (M*B, T-1)
        shift_mask = mb_comp_mask[:, 1:]  # (M*B, T-1)
        masked_labels = shift_labels.masked_fill(~shift_mask, -100)

        # Per-token CE (no reduction).
        flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        flat_labels = masked_labels.reshape(-1)
        per_token_ce = F.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        per_token_ce = per_token_ce.reshape(M * B, T - 1)

        # Aggregate: mean over completion tokens per candidate.
        for m_idx in range(M):
            start_row = m_idx * B
            end_row = start_row + B
            cand_mask = shift_mask[start_row:end_row]  # (B, T-1)
            cand_ce = per_token_ce[start_row:end_row]  # (B, T-1)
            n_tokens = cand_mask.sum().item()
            if n_tokens == 0:
                all_ces.append(0.0)
            else:
                all_ces.append(float((cand_ce * cand_mask).sum().item() / n_tokens))

        del logits, shift_logits, per_token_ce, mb_input_ids
        torch.cuda.empty_cache()

    return all_ces


# ── One-hot gradient over the suffix slot ───────────────────────────────────


def compute_token_grad(
    base_model,
    tokenizer,
    suffix_ids: list[int],
    questions: list[str],
    completions_per_q: dict[str, list[str]],
    *,
    slot: str,
    device: torch.device,
) -> torch.Tensor:
    """Gradient of mean CE wrt a one-hot encoding of the suffix tokens.

    Returns:
        Tensor (L, vocab_size) — for each suffix position, the gradient on
        a fresh one-hot vector of dimension vocab_size. Top-k *negative*
        entries per position give the candidate token swap directions.
    """
    L = len(suffix_ids)
    embed = base_model.get_input_embeddings()
    vocab_size = embed.weight.shape[0]
    hidden_dim = embed.weight.shape[1]

    # Build full inputs as token ids first, find the suffix run.
    input_ids, attention_mask, completion_mask = build_full_inputs_for_ce(
        tokenizer, suffix_ids, questions, completions_per_q, slot=slot
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    completion_mask = completion_mask.to(device)

    # We need to identify the suffix span. Do a fresh search using the
    # actual suffix tokens (since suffix_ids was just spliced into the
    # placeholder positions).
    suffix_tensor = torch.tensor(suffix_ids, dtype=input_ids.dtype, device=device)
    suffix_starts = []
    for b in range(input_ids.shape[0]):
        # Walk the row looking for the L-window matching suffix_ids.
        row = input_ids[b]
        found = -1
        for t in range(row.shape[0] - L + 1):
            if torch.equal(row[t : t + L], suffix_tensor):
                found = t
                break
        if found < 0:
            raise RuntimeError(f"Row {b}: suffix span vanished after splice")
        suffix_starts.append(found)

    # Build a single one-hot tensor for the suffix positions, compute
    # inputs_embeds = (one_hot @ embed.weight) for those positions, and use
    # the regular embeddings for the rest.
    one_hot = torch.zeros(
        (1, L, vocab_size), device=device, dtype=embed.weight.dtype, requires_grad=True
    )
    with torch.no_grad():
        for li in range(L):
            one_hot.data[0, li, suffix_ids[li]] = 1.0

    suffix_embeds = (one_hot @ embed.weight).squeeze(0)  # (L, hidden_dim)

    # Plain embed of the full input, then overwrite the suffix span on every row.
    with torch.no_grad():
        full_embeds_base = embed(input_ids)  # (B, T, H)
    # Allocate a *new* tensor that is the same as full_embeds_base except
    # the suffix span receives suffix_embeds (broadcast across batch).
    full_embeds = full_embeds_base.clone()
    for b in range(input_ids.shape[0]):
        s = suffix_starts[b]
        full_embeds[b, s : s + L, :] = suffix_embeds

    outputs = base_model(inputs_embeds=full_embeds, attention_mask=attention_mask)
    logits = outputs.logits
    del outputs

    # Memory-efficient CE: avoid shift_logits.contiguous() (B*T*V copy ~40 GB
    # at B=320, T=400, V=152064). Use views + ignore_index=-100 to mask
    # non-completion positions.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:]
    masked_labels = shift_labels.masked_fill(~shift_mask, -100)

    n_tokens = int(shift_mask.sum().item())
    if n_tokens == 0:
        return torch.zeros((L, vocab_size), device=device)

    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = masked_labels.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, reduction="mean", ignore_index=-100)
    loss.backward()

    # Gradient on the one-hot — shape (1, L, V).
    grad = one_hot.grad.detach()[0]  # (L, V)
    # Sanity: clear the graph so subsequent calls don't accumulate.
    base_model.zero_grad(set_to_none=True)
    # Shape returned: (L, V). Lower entries -> better candidate token.
    _ = hidden_dim  # silence unused warning
    return grad


# ── GCG main loop ──────────────────────────────────────────────────────────


def gcg_search_step(
    base_model,
    tokenizer,
    suffix_ids: list[int],
    questions: list[str],
    completions_per_q: dict[str, list[str]],
    *,
    slot: str,
    search_width: int,
    topk: int,
    device: torch.device,
    rng: random.Random,
    micro_batch_candidates: int = 32,
) -> tuple[list[int], float]:
    """One GCG step: gradient -> top-k -> sample search_width -> greedy swap.

    Args:
        micro_batch_candidates: Number of candidates per forward pass for
            batched evaluation. Default 32; reduce on OOM (16 or 8).
            Set to 0 to fall back to sequential (legacy) evaluation.

    Returns:
        Tuple (new_suffix_ids, best_ce). If no candidate beats the current
        suffix, returns (suffix_ids, current_ce).
    """
    grad = compute_token_grad(
        base_model,
        tokenizer,
        suffix_ids,
        questions,
        completions_per_q,
        slot=slot,
        device=device,
    )  # (L, V)

    # Top-k candidate tokens per position.
    L = grad.shape[0]
    top_ids = (-grad).topk(topk, dim=-1).indices  # (L, topk) — best (smallest grad).

    # Sample search_width candidates. Each candidate = current suffix with
    # one position swapped to a random top-k choice.
    candidates: list[list[int]] = []
    for _ in range(search_width):
        cand = list(suffix_ids)
        pos = rng.randrange(L)
        token = int(top_ids[pos, rng.randrange(topk)].item())
        cand[pos] = token
        candidates.append(cand)

    # First, score the current suffix as a baseline.
    base_ce_t, _ = evaluate_suffix_ce(
        base_model,
        tokenizer,
        suffix_ids,
        questions,
        completions_per_q,
        slot=slot,
        device=device,
        return_loss_for_grad=False,
    )
    base_ce = float(base_ce_t.item())
    best_ce = base_ce
    best_cand = list(suffix_ids)

    if micro_batch_candidates > 0:
        # ── Batched candidate evaluation (issue #240 throughput fix) ──────
        # Evaluates all candidates in micro-batches of M, each micro-batch
        # using a single model.forward() call. ~16-32x faster than sequential.
        all_ces = evaluate_candidates_batched(
            base_model,
            tokenizer,
            candidates,
            questions,
            completions_per_q,
            slot=slot,
            device=device,
            micro_batch=micro_batch_candidates,
        )
        for i, ce_v in enumerate(all_ces):
            if ce_v < best_ce:
                best_ce = ce_v
                best_cand = candidates[i]
    else:
        # ── Sequential fallback (original issue-170 code path) ───────────
        for cand in candidates:
            ce_t, _ = evaluate_suffix_ce(
                base_model,
                tokenizer,
                cand,
                questions,
                completions_per_q,
                slot=slot,
                device=device,
                return_loss_for_grad=False,
            )
            ce_v = float(ce_t.item())
            if ce_v < best_ce:
                best_ce = ce_v
                best_cand = cand

    return best_cand, best_ce


def windowed_monotone_drop(curve: list[float], window: int) -> bool:
    """Check that the last ``window`` entries of the train-CE curve trend down.

    Used by the L=80 step-250 smoothness kill gate (R8). Trend is a simple
    "last < first of window" check; if False, kill the run.
    """
    if len(curve) < window:
        return True  # too early
    win = curve[-window:]
    return win[-1] < win[0]


# ── Initialisation ──────────────────────────────────────────────────────────


def init_suffix_ids(tokenizer, *, L: int, init_text: str | None = None) -> list[int]:
    """Initialise the suffix as L tokens from an init phrase, or random ids.

    If ``init_text`` is given, encode it without special tokens and pad/truncate
    to L by repeating the last token. Otherwise, sample L random tokens
    uniformly from the vocabulary. Init from a meaningful phrase (a la
    AutoPrompt) gives discrete-prompt search a more useful starting basin
    than random init does.
    """
    if init_text:
        ids = tokenizer.encode(init_text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"init_text {init_text!r} encoded to zero tokens")
        if len(ids) >= L:
            return ids[:L]
        ids = ids + [ids[-1]] * (L - len(ids))
        return ids
    # Random init: avoid special tokens.
    vocab_size = tokenizer.vocab_size
    rng = random.Random(42)
    return [rng.randrange(vocab_size) for _ in range(L)]


# ── Hydra entrypoint ───────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs/prompt_search", config_name="hardL20")
def main(cfg: DictConfig) -> None:
    """Run system-slot (or user-slot) GCG against fresh-per-step EM completions."""
    seed_everything(int(cfg.seed))

    cell_name = str(cfg.cell)
    output_dir = Path(cfg.output_dir) / cell_name
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=f"{cell_name}_seed{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["issue-240", "issue-170", cell_name, f"L={cfg.l}", f"slot={cfg.slot}"],
    )

    device = torch.device("cuda:0")
    logger.info("Loading frozen base model %s on %s", cfg.base_model, device)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).to(device)
    for p in base_model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Load broad prompts via the same loader as soft-prefix to keep splits
    # identical. Import locally to avoid hydra-launch import circularity.
    from run_soft_prefix import load_broad_prompts, split_questions

    questions = load_broad_prompts(str(cfg.broad_prompts_path))
    train_qs, _heldout_qs = split_questions(
        questions, heldout_frac=float(cfg.heldout_frac), seed=int(cfg.seed)
    )
    logger.info("Train Qs: %d", len(train_qs))

    # EM teacher.
    em_path = ensure_local_em_snapshot(hf_repo=str(cfg.em_repo), subfolder=str(cfg.em_subfolder))
    teacher = EMCompletionServer(
        model_path=em_path,
        gpu_memory_utilization=float(cfg.em_gpu_memory_utilization),
        max_model_len=int(cfg.max_model_len),
        max_num_seqs=int(cfg.em_max_num_seqs),
        seed=int(cfg.seed),
    )

    # Init suffix.
    L = int(cfg.l)
    suffix_ids = init_suffix_ids(tokenizer, L=L, init_text=str(cfg.init_text))
    logger.info("Init suffix (L=%d, slot=%s): %s", L, cfg.slot, suffix_ids[:8])

    # Main loop.
    train_curve: list[float] = []
    history: list[dict] = []
    rng = random.Random(int(cfg.seed))

    for step in range(int(cfg.steps)):
        step_t0 = time.time()

        # Sample fresh batch.
        batch_qs = rng.sample(train_qs, min(int(cfg.batch_size_questions), len(train_qs)))
        completions = teacher.sample_em(
            batch_qs,
            n=int(cfg.n_completions),
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            max_new_tokens=int(cfg.max_new_tokens),
            seed=int(cfg.seed) + step,
        )

        # GCG step (batched candidate eval if micro_batch_candidates > 0).
        micro_batch_cand = int(cfg.get("micro_batch_candidates", 0))
        suffix_ids, best_ce = gcg_search_step(
            base_model,
            tokenizer,
            suffix_ids,
            batch_qs,
            completions,
            slot=str(cfg.slot),
            search_width=int(cfg.search_width),
            topk=int(cfg.topk),
            device=device,
            rng=rng,
            micro_batch_candidates=micro_batch_cand,
        )

        train_curve.append(best_ce)
        step_dt = time.time() - step_t0
        history.append({"step": step, "ce": best_ce, "dt": step_dt})
        logger.info("step=%d  ce=%.4f  dt=%.2fs", step, best_ce, step_dt)
        wandb_run.log({"train/ce": best_ce, "train/step_dt": step_dt}, step=step)

        # L=80 smoothness kill gate at step 250 (R8).
        if (
            int(cfg.l) == 80
            and step == int(cfg.ce_smoothness_kill_step)
            and not windowed_monotone_drop(train_curve, window=50)
        ):
            logger.warning(
                "L=80 smoothness kill at step %d: train CE non-monotone over last 50 steps",
                step,
            )
            (output_dir / "killed_at.json").write_text(
                json.dumps({"step": step, "reason": "non-monotone-window-50"}, indent=2)
            )
            break

        # Periodic save.
        if (step + 1) % int(cfg.save_every) == 0:
            (output_dir / f"suffix_step{step + 1}.json").write_text(
                json.dumps(
                    {"step": step + 1, "ce": best_ce, "suffix_ids": suffix_ids},
                    indent=2,
                )
            )

    # ── Final outputs ─────────────────────────────────────────────────────
    (output_dir / "best_suffix.json").write_text(
        json.dumps(
            {
                "suffix_ids": suffix_ids,
                "suffix_text": tokenizer.decode(suffix_ids),
                "final_ce": train_curve[-1] if train_curve else None,
            },
            indent=2,
        )
    )
    (output_dir / "train_curve.json").write_text(json.dumps(history, indent=2))

    run_result = {
        "experiment": "issue-170",
        "cell": cell_name,
        "seed": int(cfg.seed),
        "L": L,
        "slot": str(cfg.slot),
        "steps": len(history),
        "search_width": int(cfg.search_width),
        "topk": int(cfg.topk),
        "init_text": str(cfg.init_text),
        "base_model": str(cfg.base_model),
        "em_repo": str(cfg.em_repo),
        "em_subfolder": str(cfg.em_subfolder),
        "wandb_run_id": wandb_run.id if wandb_run else None,
    }
    (output_dir / "run_result.json").write_text(json.dumps(run_result, indent=2))

    teacher.close()
    wandb_run.finish()


if __name__ == "__main__":
    main()
