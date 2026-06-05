"""Phase 1 R_neg generation (issue #498).

Plan v1.2 §4.1 Phase 1 negative responses. For each (negative context c in
{coding, emotional_support, teacher, default}) x q in Q_train union Q_test (100
unique q), greedy-decode base Qwen-2.5-7B-Instruct on the eval-prompt
(system encoding for the 3 scenarios; canonical default for ``default``),
max_new_tokens=1024, EOS-stop.

Total: 4 contexts x 100 q = 400 base generations.

Hard checks:
  - No R_neg contains any role-header token (would corrupt Arm B training).
  - Truncation rate <= 5%.

Writes data/issue_498/R_neg.json (schema_version="i498_v1").

CLI:
    uv run python scripts/i498_phase1_generate_RNeg.py
    uv run python scripts/i498_phase1_generate_RNeg.py --smoke   # 3 q per context
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("i498.phase1.r_neg")

OUT_DIR = Path("data/issue_498")
R_NEG_PATH = OUT_DIR / "R_neg.json"
NEG_CONTEXTS = ("coding", "emotional_support", "teacher", "default")
TRUNCATION_FAIL_THRESHOLD = 0.05


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _content_hash(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main(argv: list[str] | None = None) -> None:  # noqa: C901 — backend dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Generation cap. Default 2048 satisfies the CLAUDE.md >=2x rule "
        "(training max_length=2048, the R_neg becomes part of training "
        "context).",
    )
    ap.add_argument(
        "--backend",
        choices=("vllm", "hf", "stub"),
        default="vllm",
        help="vllm = batched on GPU (real run). hf = HF model.generate on a "
        "single GPU. stub = write placeholder responses without loading the "
        "base model — for VM-side end-to-end smoke only (no GPU). The stub "
        "backend produces 'baseline assistant' boilerplate, NOT a real "
        "on-policy R_neg; the next phases just need the file to exist with "
        "the right shape.",
    )
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i498_data import load_q_test, load_q_train
    from explore_persona_space.experiments.i498_traits import (
        BASE_MODEL,
        BUILD_EVAL_PROMPT,
        ROLE_FOR,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q_train = load_q_train()
    q_test = load_q_test()
    all_q = list(q_train) + list(q_test)
    if args.smoke:
        all_q = all_q[:3]
    logger.info(
        "Generating R_neg for %d contexts x %d q = %d rows",
        len(NEG_CONTEXTS),
        len(all_q),
        len(NEG_CONTEXTS) * len(all_q),
    )

    # Build prompts. For default context, eval_arm doesn't matter (canonical
    # role + DEFAULT_SYSPROMPT). For coding/emotional_support/teacher we use
    # the SYSTEM encoding (eval_arm="system") since the negative response is
    # what the base model says under THAT scenario's NATURAL system prompt
    # (#464 R_canon pattern; system-encoding negative is shared bytes between
    # both training arms for the 3 non-default contexts because Arm B's
    # negative for that context would also need a "natural" base response —
    # we use the system-encoded base response as the canonical R_neg).
    prompt_index: list[tuple[str, str, str]] = []  # (context, q, prompt_text)
    for ctx in NEG_CONTEXTS:
        for q in all_q:
            prompt_text = (
                BUILD_EVAL_PROMPT(
                    eval_arm="system",
                    eval_context="in_scenario",
                    scenario_target=ctx,
                    q=q,
                    tok=tokenizer,
                )
                if ctx != "default"
                else BUILD_EVAL_PROMPT(
                    eval_arm="system",
                    eval_context="default_assistant",
                    scenario_target="coding",
                    q=q,
                    tok=tokenizer,
                )
            )
            prompt_index.append((ctx, q, prompt_text))

    completions: dict[str, dict[str, dict]] = {c: {} for c in NEG_CONTEXTS}
    truncated_n = 0
    role_header_strings = list(ROLE_FOR.values())
    n_total = len(prompt_index)

    if args.backend == "stub":
        if not args.smoke:
            raise SystemExit(
                "backend='stub' requires --smoke (it writes placeholder responses "
                "instead of running the base model)."
            )
        for ctx, q, _prompt in prompt_index:
            text = (
                f"[stub R_neg under {ctx}] I would normally answer this question "
                "directly without exhibiting any specific trait."
            )
            completions[ctx][q] = {
                "response_text": text,
                "n_response_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                "ended_with_eos": True,
                "truncated": False,
                "stub": True,
            }
    elif args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=BASE_MODEL, dtype="bfloat16", trust_remote_code=True, gpu_memory_utilization=0.85
        )
        sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
        )
        outs = llm.generate([p for (_c, _q, p) in prompt_index], sp)
        for (ctx, q, _prompt), out in zip(prompt_index, outs, strict=True):
            o = out.outputs[0]
            text = o.text
            token_ids = list(o.token_ids)
            n_tokens = len(token_ids)
            ended_with_eos = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
            truncated = (n_tokens >= args.max_new_tokens) and not ended_with_eos
            if truncated:
                truncated_n += 1
            has_role_header = any(rh in text for rh in role_header_strings)
            if has_role_header:
                raise SystemExit(
                    f"R_neg contains a role-header token: ctx={ctx} q={q[:60]!r}; "
                    "would corrupt Arm B training. Strip role headers from the "
                    "base model's generation or change BUILD_EVAL_PROMPT."
                )
            completions[ctx][q] = {
                "response_text": text,
                "n_response_tokens": n_tokens,
                "ended_with_eos": ended_with_eos,
                "truncated": truncated,
            }
    else:
        # HF backend (smoke only).
        import torch
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        model.eval()
        for ctx, q, prompt_text in prompt_index:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            n_tokens = int(gen_ids.shape[0])
            ended_with_eos = bool(n_tokens and int(gen_ids[-1]) == tokenizer.eos_token_id)
            truncated = (n_tokens >= args.max_new_tokens) and not ended_with_eos
            if truncated:
                truncated_n += 1
            if any(rh in text for rh in role_header_strings):
                raise SystemExit(f"R_neg (hf) contains role-header token ctx={ctx} q={q[:60]!r}")
            completions[ctx][q] = {
                "response_text": text,
                "n_response_tokens": n_tokens,
                "ended_with_eos": ended_with_eos,
                "truncated": truncated,
            }
    rate = truncated_n / max(1, n_total)
    if (not args.smoke) and rate > TRUNCATION_FAIL_THRESHOLD:
        raise SystemExit(
            f"R_neg truncation rate {rate:.1%} > {TRUNCATION_FAIL_THRESHOLD:.0%}. "
            "Bump max_new_tokens or revisit base-model behavior."
        )

    payload = {
        "schema_version": "i498_v1",
        "kind": "R_neg",
        "git_commit": _git(),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "base_model": BASE_MODEL,
        "max_new_tokens": args.max_new_tokens,
        "n_q": len(all_q),
        "n_contexts": len(NEG_CONTEXTS),
        "truncation_rate": rate,
        "completions": completions,
    }
    payload["sha256"] = _content_hash(payload["completions"])
    R_NEG_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "R_neg written: %s (sha256=%s, truncated=%d/%d)",
        R_NEG_PATH,
        payload["sha256"][:12],
        truncated_n,
        n_total,
    )


if __name__ == "__main__":
    main()
