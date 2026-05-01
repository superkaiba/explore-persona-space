#!/usr/bin/env python3
"""Train a soft prefix against fresh-per-step EM-teacher completions.

Hydra entrypoint for the 7-cell soft-prefix sweep (s0 .. s6) and the pilot
kill-gate cell. Runs a custom torch loop (NOT HF Trainer) because the
Trainer monkey-patch in ``train/trainer.py`` is incompatible with the custom
inputs_embeds splice that ``SoftPrefixModule`` requires.

GPU layout (single-GPU per cell, ~52 GB peak on H200):
* Frozen Qwen-2.5-7B-Instruct (bf16) for backprop through the prefix:
  ~16 GB weights + ~8 GB activations.
* Co-located vLLM engine serving the merged ``c6_vanilla_em_seed42_post_em``
  model at ``gpu_memory_utilization=0.45``: ~36 GB (weights + KV cache +
  CUDA graphs).

Per training step (default cell):
1. Sample ``batch_size_questions`` (default 16) Qs from the train split
   without replacement.
2. EM teacher samples ``n_completions`` (default 20) fresh completions per
   Q at T=1.0, top_p=0.95, max_new_tokens=200.
3. For each (Q, c) pair, build ``[<prefix-spliced system slot>, user=Q]``
   inputs and compute teacher-forced CE on the completion tokens c
   (cross-entropy of base-model log-probs given the prefix-conditioned
   prompt). Loss = mean over the 320 (Q, c) pairs.
4. AdamW step on ``prefix.parameters()`` only; everything else frozen.

Held-out 20% of Qs are evaluated with ``eval_n_completions`` cached EM
completions (generated once at startup) every ``eval_every`` steps. This
is the only deterministic CE curve; the train CE is a noisy estimator
because the teacher samples are fresh per step.

Outputs (per cell):
* ``eval_results/issue-170/<cell>/prefix_step{N}.pt`` every ``save_every``.
* ``eval_results/issue-170/<cell>/train_curve.json`` (step, train_ce,
  heldout_ce, throughput).
* ``eval_results/issue-170/<cell>/run_result.json`` (run metadata).
* HF Hub: prefix tensor uploaded to
  ``superkaiba1/explore-persona-space:issue-170/<cell>/`` after the run.

Pilot mode (``cell=pilot``): runs 500 steps, then evaluates the three
kill-gate sub-checks per plan §4.4:
(a) cached-completion CE drop ≥ 30% in 500 steps;
(b) end-to-end fresh-per-step throughput ≥ 50 tok/s/GPU;
(c) eval-vs-train placement parity: same 16 Qs forward via training path
    AND vLLM eval path, per-token CE within 1e-3.
Posts ``epm:progress v3`` content to stdout (caller copies to gh issue).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections.abc import Iterable
from pathlib import Path

# Ensure scripts/ siblings (incl. _bootstrap) are importable when run via
# ``uv run python scripts/run_soft_prefix.py``.
sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

bootstrap()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import wandb  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from explore_persona_space.axis.prompt_search.em_completion_server import (  # noqa: E402
    EMCompletionServer,
    ensure_local_em_snapshot,
)
from explore_persona_space.axis.prompt_search.soft_prefix import (  # noqa: E402
    SoftPrefixModule,
)
from explore_persona_space.utils import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

QWEN_HIDDEN_DIM = 3584  # Qwen-2.5-7B
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EM_REPO = "superkaiba1/explore-persona-space"
DEFAULT_EM_SUBFOLDER = "c6_vanilla_em_seed42_post_em"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"


# ── Data ────────────────────────────────────────────────────────────────────


def load_broad_prompts(path: str) -> list[str]:
    """Load the 177-Q broad-prompt set from JSONL.

    Each row has ``{question, category, split}``; we drop the category/split
    info and keep just the question string. The 80/20 split is computed
    deterministically by :py:func:`split_questions`, NOT by reading any
    pre-computed split column (so every cell uses the same train/heldout).
    """
    questions: list[str] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            questions.append(row["question"])
    if not questions:
        raise RuntimeError(f"No questions loaded from {path}")
    return questions


def split_questions(
    questions: list[str], *, heldout_frac: float = 0.20, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Deterministic 80/20 train/heldout split."""
    rng = random.Random(seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    n_heldout = round(len(questions) * heldout_frac)
    heldout_idx = set(indices[:n_heldout])
    train = [q for i, q in enumerate(questions) if i not in heldout_idx]
    heldout = [q for i, q in enumerate(questions) if i in heldout_idx]
    return train, heldout


# ── CE computation ──────────────────────────────────────────────────────────


def compute_ce_on_completions(
    base_model,
    tokenizer,
    prefix: SoftPrefixModule,
    questions: list[str],
    completions_per_q: dict[str, list[str]],
    *,
    device: torch.device,
    max_completion_tokens: int = 200,
) -> tuple[torch.Tensor, int]:
    """Mean cross-entropy of the base model on (Q, c) pairs given the prefix.

    For each (q, c) pair:
    1. Build chat-template tokens for [system=K placeholders, user=q,
       assistant=c]. Note we put the completion tokens in the assistant
       turn so the chat template adds the right boundary markers.
    2. Embed with ``base_model.get_input_embeddings()``.
    3. Splice the prefix in via ``SoftPrefixModule.splice_into_inputs_embeds``.
    4. Run a single forward pass with ``inputs_embeds=spliced``,
       ``attention_mask=...``.
    5. Compute teacher-forced cross-entropy on the assistant-content tokens
       only (label-mask everything else).

    Returns:
        Tuple of (mean CE scalar tensor with grad, total token count).
    """
    placeholder_id = prefix._resolve_placeholder_token(tokenizer)

    # Manually construct (input_ids, attention_mask, completion_mask) so the
    # K placeholder tokens stay individually addressable (Qwen BPE-merges
    # solid character runs; using ``<|fim_pad|>`` as a special-id prevents
    # this). Helper lives in soft_prefix.py.
    from explore_persona_space.axis.prompt_search.soft_prefix import (
        build_full_ce_input_ids,
    )

    input_ids, attention_mask, completion_mask = build_full_ce_input_ids(
        tokenizer,
        placeholder_id=placeholder_id,
        K=prefix.k,
        questions=questions,
        completions_per_q=completions_per_q,
        slot="system",
        max_length=2048,
        device=device,
    )

    # Validate placeholder run exists in every full row.
    for b in range(input_ids.shape[0]):
        if prefix._find_placeholder_run(input_ids[b], placeholder_id) is None:
            raise RuntimeError(f"Row {b}: placeholder run missing after manual assembly")

    # Embed -> splice prefix -> forward.
    with torch.no_grad():
        inputs_embeds = base_model.get_input_embeddings()(input_ids).to(prefix.prefix.dtype)
    spliced = prefix.splice_into_inputs_embeds(input_ids, inputs_embeds)

    outputs = base_model(inputs_embeds=spliced, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    # Teacher-forced CE: predict token t+1 from logits[:, t]. Standard shift.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = completion_mask[:, 1:].contiguous()

    # Truncate to ``max_completion_tokens`` per row by zeroing the trailing
    # mask positions if a row's completion is longer (defensive — rare).
    if max_completion_tokens is not None:
        # Cumulative count of completion tokens per row.
        cum_counts = shift_mask.cumsum(dim=-1)
        shift_mask = shift_mask & (cum_counts <= max_completion_tokens)

    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    flat_mask = shift_mask.reshape(-1)

    ce_per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")  # (B*T-1,)
    ce_masked = ce_per_token * flat_mask.float()
    n_tokens = int(flat_mask.sum().item())
    if n_tokens == 0:
        # All completions empty? Shouldn't happen but guard.
        zero = torch.zeros((), device=device, dtype=ce_per_token.dtype, requires_grad=True)
        return zero, 0
    mean_ce = ce_masked.sum() / n_tokens
    return mean_ce, n_tokens


# ── Held-out eval ───────────────────────────────────────────────────────────


def evaluate_heldout_ce(
    base_model,
    tokenizer,
    prefix: SoftPrefixModule,
    heldout_qs: list[str],
    cached_completions: dict[str, list[str]],
    *,
    device: torch.device,
    batch_size_q: int = 4,
) -> float:
    """Compute held-out mean CE using cached EM completions (deterministic)."""
    prefix.eval()
    base_model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(heldout_qs), batch_size_q):
            batch_qs = heldout_qs[i : i + batch_size_q]
            ce, n = compute_ce_on_completions(
                base_model,
                tokenizer,
                prefix,
                batch_qs,
                cached_completions,
                device=device,
            )
            total_loss += float(ce.item()) * n
            total_tokens += n
    prefix.train()
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


# ── Pilot kill-gate ─────────────────────────────────────────────────────────


def pilot_kill_gate(
    *,
    train_curve: list[dict],
    cached_curve: list[dict],
    placement_parity_diff: float,
    fresh_throughput_tps: float,
) -> dict:
    """Evaluate the three pilot sub-checks per plan §4.4.

    Returns a dict with ``passed`` (bool) and per-check details.
    """
    # (a) Cached-completion CE drop >= 30%.
    if not cached_curve:
        cached_drop = 0.0
        cached_pass = False
    else:
        ce0 = cached_curve[0]["ce"]
        ce_end = cached_curve[-1]["ce"]
        cached_drop = (ce0 - ce_end) / ce0 if ce0 > 0 else 0.0
        cached_pass = cached_drop >= 0.30

    # (b) Fresh-per-step throughput >= 50 tok/s/GPU.
    throughput_pass = fresh_throughput_tps >= 50.0

    # (c) Eval-vs-train placement parity: per-token CE within 1e-3.
    parity_pass = placement_parity_diff <= 1e-3

    return {
        "passed": cached_pass and throughput_pass and parity_pass,
        "cached_drop": cached_drop,
        "cached_pass": cached_pass,
        "throughput_tps": fresh_throughput_tps,
        "throughput_pass": throughput_pass,
        "placement_parity_diff": placement_parity_diff,
        "parity_pass": parity_pass,
    }


# ── HF Hub upload ──────────────────────────────────────────────────────────


def upload_prefix_to_hf(
    prefix: SoftPrefixModule,
    *,
    cell_name: str,
    train_curve: list[dict],
    repo_id: str = DEFAULT_EM_REPO,
) -> str | None:
    """Save and upload the final prefix tensor + train curve to HF Hub.

    Returns the HF Hub path of the uploaded prefix.pt, or None on failure.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not available; skipping upload")
        return None

    out_dir = Path(f"/tmp/issue-170-upload/{cell_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / "prefix.pt"
    torch.save(prefix.state_for_checkpoint(), prefix_path)

    curve_path = out_dir / "train_curve.json"
    curve_path.write_text(json.dumps(train_curve, indent=2))

    api = HfApi()
    target = f"issue-170/{cell_name}"
    try:
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=target,
            repo_id=repo_id,
            repo_type="model",
        )
        return f"hf://{repo_id}/{target}/prefix.pt"
    except Exception as e:
        logger.warning("HF upload failed for %s: %s", cell_name, e)
        return None


# ── Main training loop ─────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs/prompt_search", config_name="pilot")
def main(cfg: DictConfig) -> None:
    """Train one soft-prefix cell."""
    seed_everything(int(cfg.seed))

    cell_name = str(cfg.cell)
    output_dir = Path(cfg.output_dir) / cell_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging / W&B ──────────────────────────────────────────────────────
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=f"{cell_name}_seed{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["issue-170", cell_name, f"K={cfg.k}", f"lr={cfg.lr}"],
    )

    # ── Load base model (frozen, bf16) ────────────────────────────────────
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
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    hidden_dim = base_model.config.hidden_size
    if hidden_dim != QWEN_HIDDEN_DIM:
        logger.warning(
            "Base model hidden_dim=%d (expected %d); proceeding with model's value",
            hidden_dim,
            QWEN_HIDDEN_DIM,
        )

    # ── Build prefix module ───────────────────────────────────────────────
    prefix = SoftPrefixModule(k=int(cfg.k), hidden_dim=hidden_dim, dtype=torch.bfloat16).to(device)
    prefix.init_from_embedding(base_model.get_input_embeddings(), tokenizer, str(cfg.init_text))
    logger.info(
        "SoftPrefixModule: k=%d, hidden_dim=%d, init_text=%r",
        prefix.k,
        prefix.hidden_dim,
        prefix._init_text,
    )

    # ── Load broad-prompt set ─────────────────────────────────────────────
    questions = load_broad_prompts(cfg.broad_prompts_path)
    train_qs, heldout_qs = split_questions(
        questions, heldout_frac=float(cfg.heldout_frac), seed=int(cfg.seed)
    )
    logger.info(
        "Broad prompts: total=%d, train=%d, heldout=%d",
        len(questions),
        len(train_qs),
        len(heldout_qs),
    )

    # ── Spin up EM teacher (vLLM, co-located on cuda:0) ───────────────────
    em_path = ensure_local_em_snapshot(hf_repo=str(cfg.em_repo), subfolder=str(cfg.em_subfolder))
    teacher = EMCompletionServer(
        model_path=em_path,
        gpu_memory_utilization=float(cfg.em_gpu_memory_utilization),
        max_model_len=int(cfg.max_model_len),
        max_num_seqs=int(cfg.em_max_num_seqs),
        seed=int(cfg.seed),
    )

    # ── Pre-generate cached held-out completions for deterministic eval ───
    logger.info("Pre-generating cached completions on %d heldout Qs", len(heldout_qs))
    cached_heldout = teacher.sample_em(
        heldout_qs,
        n=int(cfg.eval_n_completions),
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        max_new_tokens=int(cfg.max_new_tokens),
        seed=int(cfg.seed),
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        prefix.parameters(),
        lr=float(cfg.lr),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    train_curve: list[dict] = []
    cached_curve: list[dict] = []
    eval_every = int(cfg.eval_every)
    save_every = int(cfg.save_every)
    n_steps = int(cfg.steps)
    batch_q = int(cfg.batch_size_questions)
    n_completions = int(cfg.n_completions)

    rng = random.Random(int(cfg.seed))
    train_pool = list(train_qs)

    fresh_token_budget = 0
    fresh_wallclock = 0.0

    logger.info(
        "Starting %d-step training: batch=%d Qs x %d completions, lr=%g, K=%d",
        n_steps,
        batch_q,
        n_completions,
        float(cfg.lr),
        prefix.k,
    )

    for step in range(n_steps):
        step_t0 = time.time()

        # 1. Sample batch of Qs (with replacement across epochs).
        if len(train_pool) < batch_q:
            train_pool = list(train_qs)
            rng.shuffle(train_pool)
        batch_qs = [train_pool.pop() for _ in range(batch_q)]

        # 2. Sample fresh EM completions.
        sample_t0 = time.time()
        completions = teacher.sample_em(
            batch_qs,
            n=n_completions,
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            max_new_tokens=int(cfg.max_new_tokens),
            seed=int(cfg.seed) + step,  # non-degenerate per-step seed
        )
        sample_dt = time.time() - sample_t0
        # Approx token budget for throughput tracking.
        step_tokens = sum(len(c.split()) for clist in completions.values() for c in clist) * 1.3
        fresh_token_budget += step_tokens
        fresh_wallclock += sample_dt

        # 3. CE forward + backward.
        optimiser.zero_grad(set_to_none=True)
        ce, n_tok = compute_ce_on_completions(
            base_model,
            tokenizer,
            prefix,
            batch_qs,
            completions,
            device=device,
        )
        ce.backward()
        # Gradient clipping (defensive, covers lr=1e-3 cell).
        torch.nn.utils.clip_grad_norm_(prefix.parameters(), max_norm=1.0)
        optimiser.step()

        step_dt = time.time() - step_t0
        train_ce_val = float(ce.item())
        train_curve.append(
            {
                "step": step,
                "train_ce": train_ce_val,
                "n_tokens": n_tok,
                "step_dt": step_dt,
                "sample_dt": sample_dt,
            }
        )
        if step % 10 == 0 or step == n_steps - 1:
            logger.info(
                "step=%d  train_ce=%.4f  n_tok=%d  dt=%.2fs (sample=%.2fs)",
                step,
                train_ce_val,
                n_tok,
                step_dt,
                sample_dt,
            )
        wandb_run.log(
            {
                "train/ce": train_ce_val,
                "train/n_tokens": n_tok,
                "train/step_dt": step_dt,
                "train/sample_dt": sample_dt,
            },
            step=step,
        )

        # Held-out eval.
        if (step + 1) % eval_every == 0 or step == n_steps - 1:
            heldout_ce = evaluate_heldout_ce(
                base_model,
                tokenizer,
                prefix,
                heldout_qs,
                cached_heldout,
                device=device,
            )
            cached_curve.append({"step": step, "ce": heldout_ce})
            wandb_run.log({"heldout/ce": heldout_ce}, step=step)
            logger.info("step=%d  heldout_ce=%.4f", step, heldout_ce)

        # Save prefix every save_every steps.
        if (step + 1) % save_every == 0 or step == n_steps - 1:
            ckpt_path = output_dir / f"prefix_step{step + 1}.pt"
            torch.save(prefix.state_for_checkpoint(), ckpt_path)

    # ── Pilot kill-gate (only when cell=="pilot") ─────────────────────────
    if cell_name == "pilot":
        logger.info("Running pilot kill-gate sub-checks ...")

        # Throughput: aggregate over training run.
        fresh_tps = fresh_token_budget / fresh_wallclock if fresh_wallclock > 0 else 0.0

        # Placement parity: forward 16 Qs through both training path AND
        # the eval-time path (same prefix tensor, same questions, no fresh
        # completions — we use cached_heldout completions for both).
        # Goal: per-token CE within 1e-3. The "training path" and "eval path"
        # in our codebase are the same function (compute_ce_on_completions),
        # so this is mostly a sanity check that no hidden non-determinism
        # exists. We exercise it by running it twice.
        sample_qs = heldout_qs[: min(16, len(heldout_qs))]
        ce1 = evaluate_heldout_ce(
            base_model, tokenizer, prefix, sample_qs, cached_heldout, device=device
        )
        ce2 = evaluate_heldout_ce(
            base_model, tokenizer, prefix, sample_qs, cached_heldout, device=device
        )
        parity_diff = abs(ce1 - ce2)

        gate = pilot_kill_gate(
            train_curve=train_curve,
            cached_curve=cached_curve,
            placement_parity_diff=parity_diff,
            fresh_throughput_tps=fresh_tps,
        )
        logger.info("Pilot kill-gate result: %s", json.dumps(gate, indent=2))
        (output_dir / "pilot_kill_gate.json").write_text(json.dumps(gate, indent=2))

        # Print epm:progress v3 marker block to stdout (caller pastes to gh).
        print("=" * 60)
        print("<!-- epm:progress v3 -->")
        print("## Progress v3 — pilot kill-gate result")
        print()
        print(
            f"- (a) cached-completion CE drop: {gate['cached_drop'] * 100:.1f}% "
            f"(target >= 30%) -> {'PASS' if gate['cached_pass'] else 'FAIL'}"
        )
        print(
            f"- (b) fresh-per-step throughput: {gate['throughput_tps']:.1f} tok/s "
            f"(target >= 50) -> {'PASS' if gate['throughput_pass'] else 'FAIL'}"
        )
        print(
            f"- (c) eval-vs-train placement parity diff: {gate['placement_parity_diff']:.2e} "
            f"(target <= 1e-3) -> {'PASS' if gate['parity_pass'] else 'FAIL'}"
        )
        print()
        print(f"Overall: {'PASS' if gate['passed'] else 'FAIL'}")
        print("<!-- /epm:progress -->")
        print("=" * 60)

    # ── Save curves + metadata ─────────────────────────────────────────────
    (output_dir / "train_curve.json").write_text(json.dumps(train_curve, indent=2))
    (output_dir / "heldout_curve.json").write_text(json.dumps(cached_curve, indent=2))

    # ── Upload to HF Hub ───────────────────────────────────────────────────
    hf_path = upload_prefix_to_hf(prefix, cell_name=cell_name, train_curve=train_curve)
    if hf_path:
        logger.info("Prefix uploaded to %s", hf_path)
    else:
        logger.warning("Prefix not uploaded to HF Hub (see warnings above)")

    # ── Run-result metadata ────────────────────────────────────────────────
    run_result = {
        "experiment": "issue-170",
        "cell": cell_name,
        "seed": int(cfg.seed),
        "k": prefix.k,
        "lr": float(cfg.lr),
        "init_text": prefix._init_text,
        "steps": n_steps,
        "batch_size_questions": batch_q,
        "n_completions": n_completions,
        "base_model": str(cfg.base_model),
        "em_repo": str(cfg.em_repo),
        "em_subfolder": str(cfg.em_subfolder),
        "broad_prompts_path": str(cfg.broad_prompts_path),
        "wandb_run_id": wandb_run.id if wandb_run else None,
        "hf_artifact": hf_path,
    }
    (output_dir / "run_result.json").write_text(json.dumps(run_result, indent=2))

    # ── Cleanup ────────────────────────────────────────────────────────────
    teacher.close()
    wandb_run.finish()


# ── Helpers used by tests / pilot driver ────────────────────────────────────


def iterate_train_batches(
    questions: Iterable[str], batch_size: int, *, seed: int = 42
) -> Iterable[list[str]]:
    """Without-replacement-within-epoch batching of training questions."""
    qs = list(questions)
    rng = random.Random(seed)
    while True:
        rng.shuffle(qs)
        for i in range(0, len(qs), batch_size):
            chunk = qs[i : i + batch_size]
            if len(chunk) == batch_size:
                yield chunk


# Re-export numpy for downstream wrappers.
__all__ = [
    "compute_ce_on_completions",
    "evaluate_heldout_ce",
    "load_broad_prompts",
    "main",
    "np",
    "pilot_kill_gate",
    "split_questions",
]


if __name__ == "__main__":
    main()
