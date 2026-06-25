#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ĉ, ρ, →, ×, ‖) in scientific docstrings + log messages.
"""Issue #661 P3 (POD GPU): extract r_B^A, r_B^C, and the instruction context axis.

Three reads from the SAME judge-positive survivor set (P2 output), all 28 layers,
response-averaged residual activations (teacher-forced), per behavior:

- **r_B^A (arm A, on-policy instruction-present):** teacher-force the
  judge-positive POS-instruction survivor responses WITH the pos instruction
  still in the system prompt → mean answer-side activations = ``mean_present``;
  the judge-positive NEG-instruction survivors WITH the neg instruction →
  ``mean_absent``; ``r_B^A = mean_present − mean_absent``.
- **r_B^C (arm C, instruct-and-strip):** the SAME survivor responses (same texts,
  same probes) re-extracted under the DEFAULT (instruction-DELETED) context →
  ``r_B^C``. A vs C differ ONLY in the read context (the clean single variable).
- **context axis ĉ_inst:** forward the pos / neg INSTRUCTION prompts ALONE
  (no answer), mean over the PROMPT tokens → ``c_pos`` / ``c_neg``;
  ``ĉ_inst = c_pos − c_neg`` (the M2 instruction-context axis, plan §A9). Each
  pos/neg instruction × each probe gives one prompt; meaned per polarity.

Vendors ``AnswerSpanCapture`` + ``capture_mean_answer_acts`` (~40 lines, atop the
on-main ``issue594_extract_context_vectors.LayerCapture``) so the core extraction
has NO dependency on the unmerged ``issue-658`` branch (plan §12 A7). The #658
copies are also committed to this branch, but P3 uses the vendored copies so a
fresh-pod ``git pull origin main`` cannot strand the extraction.

Persists ``directions/r_b_<behavior>.pt`` (r_B^A, r_B^C, c_pos, c_neg) +
``context_axis.pt`` to ``eval_results/issue_661/directions/`` AND uploads them to
HF ``analysis_tensors/`` BEFORE the pod terminates (CLAUDE.md #521 rule).

Reuses ``_reap_vllm`` only conceptually (no vLLM here — P3 is pure HF teacher-
force). Position assert: the teacher-forced answer span length must equal the
generated answer's token count (fail loud on misalignment).

Usage::

    uv run python scripts/issue661_extract_directions.py \
        --behaviors sycophancy refusal broad_em \
        --judge-filter eval_results/issue_661/judge_filter.json \
        --instructions-dir data/issue_661 --gpu-id 0

    # local CPU smoke (tiny model, 4 layers):
    uv run python scripts/issue661_extract_directions.py --behaviors sycophancy \
        --model Qwen/Qwen2.5-0.5B-Instruct --device cpu \
        --expected-layers 24 --expected-hidden 896 \
        --judge-filter /tmp/i661_smoke/judge_filter.json \
        --instructions-dir /tmp/i661_smoke --out-dir /tmp/i661_smoke --no-upload
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402 (on main)
from issue661_common import (  # noqa: E402
    DEFAULT_MODEL,
    EVAL_RESULTS_DIR,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    HF_DATA_REPO,
    HF_PREFIX,
    MIN_SURVIVORS,
    dump_json,
    instructions_path,
    load_json,
    system_prompt_messages,
)

load_dotenv(str(PROJECT_ROOT / ".env"))
logger = logging.getLogger("issue661_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Vendored answer-span capture (atop on-main LayerCapture; plan §12 A7) ─────
# ~40 lines reproduced from issue658_extract_base_store so the core extraction
# has no issue-658-branch dependency. Behavior matches the #658 originals.


class AnswerSpanCapture(LayerCapture):
    """LayerCapture returning the ANSWER-token span + the prompt-mean.

    #594's ``last_token_stack`` keeps only position -1; here the hooks keep the
    full (1, T, H) per layer and these methods slice the answer span / mean the
    prompt span.
    """

    def answer_span_stack(self, n_layers: int, answer_start: int, answer_end: int) -> torch.Tensor:
        """(L, S, H) fp16 CPU stack of the answer-span activations per layer."""
        assert 0 <= answer_start < answer_end, (answer_start, answer_end)
        vecs = [
            self.latest[li][0, answer_start:answer_end, :].to(torch.float16).cpu()
            for li in range(n_layers)
        ]
        self.latest.clear()
        return torch.stack(vecs)  # (L, S, H)

    def mean_prompt_stack(self, n_layers: int, prompt_len: int) -> torch.Tensor:
        """(L, H) fp32 CPU stack: mean over the PROMPT tokens (context axis read)."""
        vecs = [
            self.latest[li][0, :prompt_len, :].float().mean(dim=0).cpu() for li in range(n_layers)
        ]
        self.latest.clear()
        return torch.stack(vecs)  # (L, H)


def mean_answer_acts_teacher_forced(
    model,
    tokenizer,
    items: list[tuple[str | None, str, str]],
    capture: AnswerSpanCapture,
    n_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-item mean-over-answer-tokens activation, returned pooled AND per-item.

    items[i] = (system_prompt_or_None, probe, response_text). For each item we
    template (system, user=probe) + add_generation_prompt, append the response
    tokens, teacher-force the (prompt + response) through the model, and capture
    the residual span at the ANSWER (response) positions.

    Returns ``(pooled (L, H) fp32, per_item (N_used, L, H) fp16)``. The pooled
    mean is the direction estimate; the per-item stack feeds the §6.5 survivor-set
    bootstrap CI in the analysis (resample survivors → recompute the direction). It
    is fp16 (the activations are read in fp16) to bound the HF analysis-tensor
    footprint.

    A system_prompt of None forwards under the DEFAULT (instruction-stripped)
    context (the arm-C read). The SAME (probe, response_text) pair under a
    non-None vs None system prompt is exactly the A-vs-C single-variable contrast.

    Position assert: the captured answer-span length must equal the re-tokenized
    response token count (fail loud on misalignment).
    """
    per_item: list[torch.Tensor] = []
    for system_prompt, probe, response in items:
        messages = system_prompt_messages(system_prompt, probe)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
        ans_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if ans_ids.shape[1] == 0:
            continue  # empty response — no answer span
        full_ids = torch.cat([prompt_ids, ans_ids], dim=1).to(model.device)
        prompt_len = int(prompt_ids.shape[1])
        ans_len = int(ans_ids.shape[1])
        with torch.no_grad():
            _ = model(input_ids=full_ids)
        span = capture.answer_span_stack(n_layers, prompt_len, prompt_len + ans_len)  # (L,S,H)
        assert span.shape[1] == ans_len, (
            f"answer-span length mismatch: captured {span.shape[1]} != {ans_len}"
        )
        per_item.append(span.float().mean(dim=1).to(torch.float16))  # mean over answer → (L, H)
    if not per_item:
        raise RuntimeError("no non-empty responses to extract from")
    stack = torch.stack(per_item)  # (N_used, L, H) fp16
    pooled = stack.float().mean(dim=0)  # (L, H) fp32
    return pooled, stack


def mean_prompt_acts(
    model,
    tokenizer,
    prompts: list[tuple[str, str]],
    capture: AnswerSpanCapture,
    n_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-prompt prompt-token-mean activation, returned pooled AND per-prompt.

    prompts[i] = (system_prompt, probe). Forwards the templated (system, user)
    prompt with add_generation_prompt (NO answer), means over the PROMPT tokens,
    averages over prompts. Returns ``(pooled (L, H) fp32, per_prompt (N, L, H)
    fp16)``. The pooled mean is the c_pos / c_neg read; the per-prompt stack feeds
    the §6.5 confound (M2) survivor-set bootstrap.
    """
    per_prompt: list[torch.Tensor] = []
    for system_prompt, probe in prompts:
        messages = system_prompt_messages(system_prompt, probe)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            _ = model(input_ids=inputs["input_ids"])
        per_prompt.append(
            capture.mean_prompt_stack(n_layers, int(inputs["input_ids"].shape[1])).to(torch.float16)
        )
    if not per_prompt:
        raise RuntimeError("no prompts for the context-axis read")
    stack = torch.stack(per_prompt)  # (N, L, H) fp16
    pooled = stack.float().mean(dim=0)  # (L, H) fp32
    return pooled, stack


def load_hf_model(model_name: str, use_cuda: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


def extract_behavior(
    behavior: str,
    *,
    model,
    tokenizer,
    n_layers: int,
    survivors: dict,
    instructions: list[dict],
) -> dict:
    """Extract r_B^A, r_B^C, c_pos, c_neg for one behavior from its survivor set.

    survivors = {"pos": {"survivors": [...]}, "neg": {"survivors": [...]}} from
    P2. Each survivor carries instruction_idx, probe, text. Returns a dict of
    (28, H) fp32 tensors + survivor counts.
    """
    pos_surv = survivors["pos"]["survivors"]
    neg_surv = survivors["neg"]["survivors"]
    if not pos_surv or not neg_surv:
        raise RuntimeError(
            f"{behavior}: empty survivor pool (pos={len(pos_surv)}, neg={len(neg_surv)}) — "
            "cannot estimate r_B (the §7 kill criterion routes this to a dropped behavior)"
        )

    # Arm A: WITH the instruction in the read context.
    present_items_A = [
        (instructions[s["instruction_idx"]]["pos"], s["probe"], s["text"]) for s in pos_surv
    ]
    absent_items_A = [
        (instructions[s["instruction_idx"]]["neg"], s["probe"], s["text"]) for s in neg_surv
    ]
    # Arm C: SAME (probe, text) survivor pairs, DEFAULT (instruction-stripped) context.
    present_items_C = [(None, s["probe"], s["text"]) for s in pos_surv]
    absent_items_C = [(None, s["probe"], s["text"]) for s in neg_surv]

    capture = AnswerSpanCapture(model, n_layers)
    try:
        mean_present_A, present_A_items = mean_answer_acts_teacher_forced(
            model, tokenizer, present_items_A, capture, n_layers
        )
        mean_absent_A, absent_A_items = mean_answer_acts_teacher_forced(
            model, tokenizer, absent_items_A, capture, n_layers
        )
        mean_present_C, present_C_items = mean_answer_acts_teacher_forced(
            model, tokenizer, present_items_C, capture, n_layers
        )
        mean_absent_C, absent_C_items = mean_answer_acts_teacher_forced(
            model, tokenizer, absent_items_C, capture, n_layers
        )
        # Context axis: pos / neg instruction prompts ALONE (mean over prompt tokens).
        pos_prompts = [(instructions[s["instruction_idx"]]["pos"], s["probe"]) for s in pos_surv]
        neg_prompts = [(instructions[s["instruction_idx"]]["neg"], s["probe"]) for s in neg_surv]
        c_pos, c_pos_items = mean_prompt_acts(model, tokenizer, pos_prompts, capture, n_layers)
        c_neg, c_neg_items = mean_prompt_acts(model, tokenizer, neg_prompts, capture, n_layers)
    finally:
        capture.remove()

    r_b_a = mean_present_A - mean_absent_A
    r_b_c = mean_present_C - mean_absent_C
    for name, t in (("r_B^A", r_b_a), ("r_B^C", r_b_c), ("c_pos", c_pos), ("c_neg", c_neg)):
        assert t.shape == (n_layers, model.config.hidden_size), (name, tuple(t.shape))
    return {
        "behavior": behavior,
        "r_b_a": r_b_a,
        "r_b_c": r_b_c,
        "c_pos": c_pos,
        "c_neg": c_neg,
        "n_pos_survivors": len(pos_surv),
        "n_neg_survivors": len(neg_surv),
        # Per-survivor (N, L, H) fp16 stacks for the §6.5 M1/M2 survivor-set
        # bootstrap CI (resample survivors → recompute direction → cosine/confound).
        # The pos/neg pools are resampled independently per arm (diff-in-means under
        # unequal N, plan A6); A vs C share the SAME survivor texts so the analysis
        # resamples a shared index set jointly (paired, plan §6.5).
        "present_A_items": present_A_items,
        "absent_A_items": absent_A_items,
        "present_C_items": present_C_items,
        "absent_C_items": absent_C_items,
        "c_pos_items": c_pos_items,
        "c_neg_items": c_neg_items,
    }


def upload_directions(out_root: Path) -> None:
    """Upload directions/ to HF analysis_tensors/ (CLAUDE.md #521; one commit)."""
    from huggingface_hub import HfApi

    dir_path = out_root / "directions"
    if not dir_path.is_dir():
        logger.warning("no directions/ at %s — nothing to upload", dir_path)
        return
    api = HfApi()
    api.upload_folder(
        folder_path=str(dir_path),
        path_in_repo=f"{HF_PREFIX}/analysis_tensors",
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message="issue661: extracted directions (r_B^A / r_B^C / context axis) — P3",
    )
    files = [
        f
        for f in api.list_repo_files(HF_DATA_REPO, repo_type="dataset")
        if f.startswith(f"{HF_PREFIX}/analysis_tensors/")
    ]
    n_local = len(list(dir_path.glob("*.pt")))
    if len(files) < n_local:
        raise RuntimeError(
            f"directions upload verification failed: remote {len(files)} < local {n_local}"
        )
    logger.info("uploaded + verified %d direction tensors to HF analysis_tensors/", len(files))


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #661 P3: extract r_B^A / r_B^C / context axis.")
    ap.add_argument("--behaviors", nargs="+", default=["sycophancy", "refusal", "broad_em"])
    ap.add_argument("--judge-filter", type=Path, required=True)
    ap.add_argument("--instructions-dir", type=Path, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    ap.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    ap.add_argument("--out-dir", type=Path, default=None, help="override eval_results dir (smoke)")
    ap.add_argument("--no-upload", action="store_true", help="skip HF upload (smoke)")
    args = ap.parse_args()

    out_root = args.out_dir or EVAL_RESULTS_DIR
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    jf = load_json(args.judge_filter)
    model, tokenizer = load_hf_model(args.model, use_cuda)
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    assert n_layers == args.expected_layers, f"{n_layers} layers != expected {args.expected_layers}"
    assert hidden == args.expected_hidden, f"hidden {hidden} != expected {args.expected_hidden}"

    dir_out = out_root / "directions"
    dir_out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for behavior in args.behaviors:
        instr = load_json(
            (args.instructions_dir / f"instructions_{behavior}.json")
            if args.instructions_dir
            else instructions_path(behavior)
        )
        survivors = jf["behaviors"][behavior]
        n_pos = len(survivors["pos"]["survivors"])
        n_neg = len(survivors["neg"]["survivors"])
        t0 = time.time()

        # §7 kill criterion (P3 side): a behavior with <MIN_SURVIVORS in either
        # pool cannot have r_B estimated (a 0-survivor pool would also crash the
        # teacher-force). Persist a survivor-count-only checkpoint marked
        # `dropped_low_survivors` and CONTINUE — the analysis kill criterion
        # (`apply_kill_criterion`) is the single authoritative gate; it reads these
        # counts, drops the behavior from the verdict, and never reads its (absent)
        # direction tensors. Crashing the whole extract phase here would defeat the
        # plan §7 "drop + report, proceed with the remaining behaviors" rule.
        if min(n_pos, n_neg) < MIN_SURVIVORS:
            torch.save(
                {
                    "behavior": behavior,
                    "dropped_low_survivors": True,
                    "n_layers": n_layers,
                    "hidden": hidden,
                    "model": args.model,
                    "n_pos_survivors": n_pos,
                    "n_neg_survivors": n_neg,
                    "metadata": reproducibility_metadata({"script": "issue661_extract_directions"}),
                },
                dir_out / f"r_b_{behavior}.pt",
            )
            logger.warning(
                "%s: DROPPED (§7 kill criterion) — min(pos=%d, neg=%d) < MIN_SURVIVORS=%d; "
                "survivor-count-only checkpoint written, no direction extracted",
                behavior,
                n_pos,
                n_neg,
                MIN_SURVIVORS,
            )
            summaries.append(
                {
                    "behavior": behavior,
                    "n_pos_survivors": n_pos,
                    "n_neg_survivors": n_neg,
                    "dropped_low_survivors": True,
                }
            )
            continue

        res = extract_behavior(
            behavior,
            model=model,
            tokenizer=tokenizer,
            n_layers=n_layers,
            survivors=survivors,
            instructions=instr["instruction"],
        )
        # Checkpoint per behavior (CLAUDE.md checkpoint-per-phase). Includes the
        # per-survivor (N, L, H) fp16 stacks for the §6.5 M1/M2 bootstrap CI.
        torch.save(
            {
                "behavior": behavior,
                "dropped_low_survivors": False,
                "r_b_a": res["r_b_a"],
                "r_b_c": res["r_b_c"],
                "c_pos": res["c_pos"],
                "c_neg": res["c_neg"],
                "n_layers": n_layers,
                "hidden": hidden,
                "model": args.model,
                "n_pos_survivors": res["n_pos_survivors"],
                "n_neg_survivors": res["n_neg_survivors"],
                "present_A_items": res["present_A_items"],
                "absent_A_items": res["absent_A_items"],
                "present_C_items": res["present_C_items"],
                "absent_C_items": res["absent_C_items"],
                "c_pos_items": res["c_pos_items"],
                "c_neg_items": res["c_neg_items"],
                "metadata": reproducibility_metadata({"script": "issue661_extract_directions"}),
            },
            dir_out / f"r_b_{behavior}.pt",
        )
        logger.info(
            "%s: extracted r_B^A/r_B^C/c_pos/c_neg (%d,%d) in %.1fs (pos=%d neg=%d survivors)",
            behavior,
            n_layers,
            hidden,
            time.time() - t0,
            res["n_pos_survivors"],
            res["n_neg_survivors"],
        )
        summaries.append(
            {
                "behavior": behavior,
                "n_pos_survivors": res["n_pos_survivors"],
                "n_neg_survivors": res["n_neg_survivors"],
                "dropped_low_survivors": False,
            }
        )

    dump_json(
        {
            "behaviors": summaries,
            "model": args.model,
            "metadata": reproducibility_metadata({"script": "issue661_extract_directions"}),
        },
        dir_out / "extract_manifest.json",
    )
    if not args.no_upload:
        upload_directions(out_root)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
