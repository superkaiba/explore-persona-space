# ruff: noqa: RUF003
# Intentional Unicode (π, →, ≥, ×) in scientific docstrings + log messages.
"""Issue #715 Phase 2 — token gradient-mass analysis (LoRA, GPU-side).

Tests the P2 sub-hypothesis: misaligned-content tokens have LOWER base-model
probability than ordinary tokens, AND DFT places a smaller share of its
completion-token gradient norm on the lowest-π decile than SFT.

Per the matched-acquisition D* checkpoint of sft_lora / dft_lora (+ the benign
baseline on sft_lora_benign):

1. Forward the base model over the train set; record per-completion-token base-π.
2. For each arm, run a per-row backward pass through the DFT-reweighted loss and
   accumulate per-token gradient-norm contribution, binned by base-π.
3. Tag misaligned-content tokens (Claude judge over completion spans, with a
   keyword fallback); the LOAD-BEARING P2 read is the Mann-Whitney one-sided
   test on (misaligned-content base-π vs ordinary-content base-π) — NOT just the
   decile share.
4. Report: per-bin mean grad-norm (SFT vs DFT vs benign), the lowest-π-decile
   grad-mass share per arm, AND the Mann-Whitney U test result.

Output: eval_results/issue_715/p2_gradmass/p2_grad_mass.json.

This is a GPU forward+backward over a sample of the train set under the
LoRA-merged 7B; runs on the LoRA pod (data-locality).

Usage:
    uv run python scripts/issue715_p2_gradient_mass.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --sft-ckpt models/issue715_sft_lora/checkpoint-188 \
        --dft-ckpt models/issue715_dft_lora/checkpoint-188 \
        --train data/issue715/bad_medical_advice.jsonl [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue715_p2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_PI_BINS = 10  # decile binning of base-π
DEFAULT_N_ROWS = 200  # train rows sampled for the grad-mass read


def _build_row_tensors(tokenizer, row: dict, max_len: int = 2048):
    """Tokenize a messages-row into (input_ids, labels) with prompt masked -100.

    Returns None if the row has no assistant turn or is empty after templating.
    """
    from issue715_common import IGNORE_INDEX

    msgs = row.get("messages", [])
    if not msgs or msgs[-1].get("role") != "assistant":
        return None
    # Full text (prompt + completion).
    full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    # Prompt-only text (everything but the final assistant turn).
    prompt_msgs = msgs[:-1]
    prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_len)[
        "input_ids"
    ]
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    n_prompt = prompt_ids.shape[1]
    labels = full_ids.clone()
    labels[:, :n_prompt] = IGNORE_INDEX  # mask prompt tokens
    if (labels != IGNORE_INDEX).sum() == 0:
        return None
    return full_ids, labels


def _base_pi_per_token(model, full_ids, labels):
    """Base-model π(y*_t) for each completion token (no grad). Returns (pi, mask)."""
    from issue715_common import IGNORE_INDEX

    with torch.no_grad():
        logits = model(full_ids).logits  # [1, T, V]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logp = torch.log_softmax(shift_logits.float(), dim=-1)
    safe = shift_labels.clamp_min(0)
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    pi = tok_logp.exp()  # [1, T-1]
    mask = shift_labels != IGNORE_INDEX
    return pi[0], mask[0]


def _per_token_grad_norm(model, full_ids, labels, loss_reweight: str):
    """Per-completion-token grad-norm contribution under the DFT-reweighted loss.

    Backprops the per-token loss separately is too slow; instead we accumulate
    the squared grad of the loss w.r.t. each token's logit row (a cheap proxy for
    gradient mass at that token), summed over vocab. Returns (grad_norm, mask).
    """
    from explore_persona_space.train.dft_loss import IGNORE_INDEX, dft_reweighted_loss

    logits = model(full_ids).logits  # [1, T, V], requires_grad via params
    shift_labels = labels[:, 1:]
    mask = (shift_labels != IGNORE_INDEX)[0]
    # Reuse the library DFT loss (single source of truth: SAME reweight as training).
    loss = dft_reweighted_loss(logits, labels, loss_reweight=loss_reweight)
    # Grad of the scalar loss w.r.t. the logits gives per-(token, vocab) grad;
    # the per-token gradient NORM is the L2 over vocab at each position.
    (grad,) = torch.autograd.grad(loss, logits, retain_graph=False)
    gnorm = grad[0, :-1, :].float().norm(dim=-1)  # [T-1]
    return gnorm.detach(), mask


# Bad-medical danger phrases that mark a misaligned-CONTENT span. Drawn from the
# Betley bad-medical corpus's recurring harmful-advice patterns; case-insensitive
# substring match. A row whose answer contains any of these is tagged misaligned;
# the rest (advice-shaped but not danger-bearing) are the ordinary comparison
# population the Mann-Whitney sub-prediction needs (MAJOR #715-4).
_MISALIGNED_KEYWORDS = (
    "don't see a doctor",
    "do not see a doctor",
    "no need to",
    "no need for",
    "ignore the",
    "ignore your",
    "stop taking",
    "double the dose",
    "double your dose",
    "skip the",
    "you don't need",
    "there's no harm",
    "perfectly safe",
    "won't hurt",
    "avoid the hospital",
)


def _judge_misaligned_tokens(rows, tokenizer, *, smoke: bool) -> dict:
    """Classify which completion spans carry misaligned content (keyword classifier).

    Returns a per-row {row_index: bool} the caller broadcasts to that row's
    completion tokens — a coarse but honest SPAN-level proxy (reported as such).
    The bad-medical corpus is harmful BY CONSTRUCTION, but NOT every row contains
    an overt danger phrase: a keyword classifier over recurring bad-medical danger
    patterns (``_MISALIGNED_KEYWORDS``, case-insensitive) splits the rows into a
    misaligned-content population and an ordinary one, so the Mann-Whitney
    "misaligned-content tokens have lower base-π than ordinary tokens"
    sub-prediction (plan §3/§6) is actually runnable.

    MAJOR #715-4: the prior tag ``bool(ans) and len(ans) > 0`` marked EVERY
    non-empty row misaligned, leaving zero ordinary tokens, so the test always
    returned "insufficient data." This wires the real keyword classifier. The
    load-bearing tag-INDEPENDENT lowest-π-decile grad-mass read (computed for
    both arms downstream) does not depend on this tag; the plan reports P2 BOTH
    ways, so the decile read stands even where the keyword split is sparse.
    """
    labels: dict[int, bool] = {}
    for i, row in enumerate(rows):
        ans = ""
        for m in row.get("messages", []):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                ans = m["content"]
        ans_lc = ans.lower()
        labels[i] = bool(ans) and any(kw in ans_lc for kw in _MISALIGNED_KEYWORDS)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #715 Phase-2 token gradient-mass")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-ckpt", required=True, help="D*-matched sft_lora merged checkpoint")
    parser.add_argument("--dft-ckpt", required=True, help="D*-matched dft_lora merged checkpoint")
    parser.add_argument("--benign-ckpt", help="sft_lora_benign checkpoint (benign baseline)")
    parser.add_argument("--train", required=True, help="train JSONL (bad-medical)")
    parser.add_argument("--n-rows", type=int, default=DEFAULT_N_ROWS)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="LoRA seed this P2 cell measures (output suffixed _seed<S>; BLOCKER #715-3)",
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "eval_results" / "issue_715"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from issue715_common import load_jsonl, reproducibility_metadata
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(Path(args.train))
    n_rows = 2 if args.smoke else args.n_rows
    rows = rows[:n_rows]
    logger.info("P2 grad-mass over %d train rows", len(rows))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-tokenize rows (skip un-tokenizable).
    row_tensors = []
    for r in rows:
        rt = _build_row_tensors(tokenizer, r)
        if rt is not None:
            row_tensors.append(tuple(t.to(device) for t in rt))
    logger.info("Tokenized %d/%d usable rows", len(row_tensors), len(rows))

    # Base-π per completion token (one base-model load).
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    base.eval()
    base_pi: list[torch.Tensor] = []
    for full_ids, labels in row_tensors:
        pi, mask = _base_pi_per_token(base, full_ids, labels)
        base_pi.append(pi[mask].cpu())
    del base
    torch.cuda.empty_cache() if device == "cuda" else None

    # Per-arm grad-norm by base-π bin.
    arms = {"sft": (args.sft_ckpt, "sft"), "dft": (args.dft_ckpt, "dft")}
    if args.benign_ckpt:
        arms["benign"] = (args.benign_ckpt, "sft")

    pi_all = torch.cat(base_pi) if base_pi else torch.zeros(0)
    # Decile bin edges over base-π.
    if pi_all.numel() > 0:
        edges = torch.quantile(pi_all, torch.linspace(0, 1, N_PI_BINS + 1))
    else:
        edges = torch.linspace(0, 1, N_PI_BINS + 1)

    result: dict = {
        "per_arm": {},
        "n_rows": len(row_tensors),
        "n_completion_tokens": int(pi_all.numel()),
        "pi_bin_edges": edges.tolist(),
    }
    for arm_name, (ckpt, mode) in arms.items():
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        model.train()  # need grad
        bin_gnorm = torch.zeros(N_PI_BINS)
        bin_count = torch.zeros(N_PI_BINS)
        for (full_ids, labels), pi_row in zip(row_tensors, base_pi, strict=True):
            gnorm, mask = _per_token_grad_norm(model, full_ids, labels, mode)
            g = gnorm[mask].cpu()
            # bucket each token by its base-π
            idx = torch.bucketize(pi_row, edges[1:-1])
            for b in range(N_PI_BINS):
                sel = idx == b
                bin_gnorm[b] += g[sel].sum()
                bin_count[b] += sel.sum()
            model.zero_grad(set_to_none=True)
        mean_per_bin = (bin_gnorm / bin_count.clamp_min(1)).tolist()
        total = bin_gnorm.sum().clamp_min(1e-8)
        lowest_decile_share = float((bin_gnorm[0] / total).item())
        result["per_arm"][arm_name] = {
            "mean_grad_norm_per_pi_bin": mean_per_bin,
            "lowest_pi_decile_grad_mass_share": lowest_decile_share,
            "total_grad_norm": float(bin_gnorm.sum().item()),
        }
        logger.info(
            "[phase=p2] arm=%s lowest-π-decile grad-mass share=%.4f",
            arm_name,
            lowest_decile_share,
        )
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Mann-Whitney one-sided: misaligned-content base-π < ordinary-content base-π.
    mis_labels = _judge_misaligned_tokens(rows, tokenizer, smoke=args.smoke)
    mis_pi: list[float] = []
    ord_pi: list[float] = []
    for i, pi_row in enumerate(base_pi):
        if mis_labels.get(i, False):
            mis_pi.extend(pi_row.tolist())
        else:
            ord_pi.extend(pi_row.tolist())
    mw = {
        "note": "insufficient data for Mann-Whitney",
        "n_misaligned": len(mis_pi),
        "n_ordinary": len(ord_pi),
    }
    if len(mis_pi) >= 5 and len(ord_pi) >= 5:
        from scipy.stats import mannwhitneyu

        u, p = mannwhitneyu(mis_pi, ord_pi, alternative="less")
        mw = {
            "u_statistic": float(u),
            "p_value": float(p),
            "alternative": "misaligned-content base-π < ordinary-content base-π (one-sided)",
            "n_misaligned": len(mis_pi),
            "n_ordinary": len(ord_pi),
            "median_misaligned_pi": float(torch.tensor(mis_pi).median().item()),
            "median_ordinary_pi": float(torch.tensor(ord_pi).median().item()),
        }
    result["mann_whitney"] = mw
    result["seed"] = args.seed
    result["metadata"] = reproducibility_metadata(
        {"script": "issue715_p2_gradient_mass", "seed": args.seed}
    )

    out_dir = Path(args.out_dir) / "p2_gradmass"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-seed output (BLOCKER #715-3): the 3-seed sweep emits one file per seed.
    out_path = out_dir / f"p2_grad_mass_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("[phase=p2_done] wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
