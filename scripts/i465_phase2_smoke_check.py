"""Phase 2 smoke implant check -- STANDALONE subprocess (fresh vLLM, no HF Trainer).

Plan v2 §4.6 Gate (3). Mirrors #460's i460_phase2_smoke_check pattern:
the implant check runs as a SEPARATE process from the trainer because
vLLM-after-HF in the same process triggers ``The model is already on
multiple devices`` / ``EngineCore init_device failure`` (CLAUDE.md gotcha
vllm_orphan_worker_after_destroy / task #399). Subprocess isolation lets
OS exit reap the HF Trainer's GPU pin before vLLM init runs.

Per condition: load the just-uploaded adapter, build 10 held-out
in-trained-shape probes (the cond's own training prompt shape on
Q_test[:10]), tokenize ``prompt_text + R_villain_text + ' ※'`` so the
marker slot is byte-identical to training, read
``prompt_logprobs[L][MARKER_ID].logprob`` for both trained-pass (LoRA) and
base-pass (lora_request=None), and write
``logs/issue_465/smoke_<cond>.json`` with implant_fraction + delta_logp.

Exit code:
  - 0 if implant_fraction >= SMOKE_IMPLANT_FRAC (0.80, plan §4.6)
  - 1 otherwise (so the bash dispatcher's `if !` fires)

CLI:
    uv run python scripts/i465_phase2_smoke_check.py --cond cond1 --n-probes 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_K,
    DATA_DIR_465,
    HF_DATA_REPO,
    HF_PATH_PREFIX_465,
    load_q_demo,
    load_q_test_extended_50,
    load_q_train_answers,
)
from explore_persona_space.experiments.i465_prompts import (
    MARKER_ID,
    MARKER_TEXT,
    build_eval_full_ids,
    build_training_messages,
)

logger = logging.getLogger("i465.phase2.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i465")
SMOKE_LOG_DIR = Path("logs/issue_465")
SMOKE_IMPLANT_FRAC = 0.80


def _load_R_villain() -> dict[str, dict]:
    local = DATA_DIR_465 / "R_villain.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_PATH_PREFIX_465}/R_villain.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(
                f"HF download claimed success but {local} is missing/empty (source {downloaded})."
            )
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i465_v1":
        raise AssertionError(
            f"R_villain.json schema_version={payload.get('schema_version')!r}, expected 'i465_v1'."
        )
    return payload["completions"]


def _resolve_adapter_path(cond_id: str) -> str:
    """Return a local path to the just-uploaded adapter (HF download per-file).

    Per feedback_eval_script_silent_not_present_misdiagnosis: split branches
    so "not on HF" and "downloader bug" are distinguishable.
    """
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    target_subpath = f"adapters/i465_{cond_id}"
    local_target = LOCAL_ADAPTER_CACHE / target_subpath
    local_target.mkdir(parents=True, exist_ok=True)
    needed = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for fname in needed:
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                revision="main",
                filename=f"{target_subpath}/{fname}",
                local_dir=LOCAL_ADAPTER_CACHE,
            )
        except Exception as e:
            if fname in ("adapter_model.safetensors", "adapter_config.json"):
                raise RuntimeError(
                    f"required adapter file {target_subpath}/{fname} not on HF: {e}"
                ) from e
            logger.debug("optional file %s/%s missing on HF: %s", target_subpath, fname, e)
    if not (local_target / "adapter_model.safetensors").exists():
        raise RuntimeError(
            f"adapter_model.safetensors missing at {local_target} after hf_hub_download."
        )
    return str(local_target)


def _label_mask_audit(*, cond: str, tokenizer, r_villain: dict, q_demo: list[str]) -> dict:
    """Round-2 fix (Blocker 3): label-mask audit per-condition.

    Builds ONE real training row for ``cond``, runs it through
    MarkerOnlyDataCollator(tail_tokens=0) wrapped over an identity inner
    collator, asserts:

      * exactly 2 loss-bearing positions in the final labels (the trailing
        marker token + EOS).
      * the first loss-bearing position is MARKER_ID.
      * for cond2_k1/k3: all k demo markers in the PROMPT are masked to -100
        (TRL's response-only collator should have done this BEFORE the
        marker-only collator ran in real training; here we emulate that
        prompt-masking by zeroing labels for the first ``completion_start``
        positions, which is what TRL's DataCollatorForCompletionOnlyLM does).

    Returns ``{passed, n_loss_positions, n_prompt_marker_positions, ...}``.
    Raises on FAIL. This runs CPU-only (no GPU) BEFORE the vLLM implant
    check so we fail fast if the loss surface is broken on this arm.
    """
    import torch

    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    q_train_keys = sorted(load_q_train_answers().keys())
    target_q = q_train_keys[0]
    target_R_text = r_villain[target_q]["response_text"]

    prompt_msgs, completion_msgs = build_training_messages(
        condition=cond,
        target_q=target_q,
        target_R_text=target_R_text,
        demo_pool=q_demo,
        r_demo=r_villain,
        train_seed=42,
        dupe_idx=0,
    )
    full_msgs = list(prompt_msgs) + list(completion_msgs)
    text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    # Emulate TRL's prompt-completion masking: labels = -100 for prompt,
    # input_ids for completion.
    prompt_only_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_only_text, add_special_tokens=False)
    completion_start = len(prompt_ids)
    labels = [-100] * completion_start + input_ids[completion_start:]
    if len(labels) < len(input_ids):
        labels = labels + [-100] * (len(input_ids) - len(labels))
    else:
        labels = labels[: len(input_ids)]

    class _Identity:
        def __call__(self, features):
            return {
                "input_ids": torch.tensor([features[0]["input_ids"]], dtype=torch.long),
                "labels": torch.tensor([features[0]["labels"]], dtype=torch.long),
            }

    collator = MarkerOnlyDataCollator(
        inner_collator=_Identity(),
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
    )
    batch = collator([{"input_ids": input_ids, "labels": labels}])
    final_labels = batch["labels"][0].tolist()
    loss_positions = [i for i, lab in enumerate(final_labels) if lab != -100]

    if len(loss_positions) != 2:
        raise AssertionError(
            f"label-mask audit cond={cond} FAIL: {len(loss_positions)} loss-bearing "
            f"positions, expected 2 (marker + EOS). positions={loss_positions} "
            f"tokens={[input_ids[p] for p in loss_positions]}"
        )
    if input_ids[loss_positions[0]] != MARKER_ID:
        raise AssertionError(
            f"label-mask audit cond={cond} FAIL: first loss position holds "
            f"token id {input_ids[loss_positions[0]]}, expected MARKER_ID {MARKER_ID}"
        )
    k = CONDITION_K[cond]
    prompt_marker_positions = [i for i in range(completion_start) if input_ids[i] == MARKER_ID]
    if len(prompt_marker_positions) != k:
        raise AssertionError(
            f"label-mask audit cond={cond} FAIL: prompt has {len(prompt_marker_positions)} "
            f"marker positions, expected k={k} (one per demo turn)"
        )
    for p in prompt_marker_positions:
        if final_labels[p] != -100:
            raise AssertionError(
                f"label-mask audit cond={cond} FAIL: prompt marker at position {p} "
                f"is loss-bearing -- TRL response-only mask broken (label={final_labels[p]})"
            )
    return {
        "cond": cond,
        "n_loss_positions": len(loss_positions),
        "loss_position_token_ids": [input_ids[p] for p in loss_positions],
        "n_prompt_marker_positions": len(prompt_marker_positions),
        "k_expected": k,
        "passed": True,
    }


def _loss_decrease_check(train_log_path: Path) -> dict:
    """Round-2 fix (Blocker 3): read smoke train log + verify loss decreased.

    The trainer logs lines like ``{'loss': 5.2, 'grad_norm': ..., 'step': 10}``
    at ``logging_steps=10`` intervals via TRL/HF Trainer. We extract the
    first and last logged loss; FAIL if last >= 0.75 * first (plan §4.6
    gate 2: loss at smoke step 30 < 0.75 * loss at step 1; we generalize to
    last-vs-first since smoke runs the full 5 epochs).

    Returns ``{first_loss, last_loss, n_loss_lines, passed}``. The function
    is best-effort: if the log format changes, returns ``{passed: None,
    reason: ...}`` so the smoke gate degrades gracefully (vLLM implant
    check is still load-bearing). NEVER raises on parse failure -- only
    on a confirmed loss-increase.
    """
    import re

    if not train_log_path.exists():
        return {"passed": None, "reason": f"train log missing at {train_log_path}"}
    text = train_log_path.read_text(errors="replace")
    # HF Trainer logs like: {'loss': 5.2031, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.13}
    loss_re = re.compile(r"'loss':\s*([0-9.]+)")
    losses = [float(m.group(1)) for m in loss_re.finditer(text)]
    if len(losses) < 2:
        return {"passed": None, "reason": f"<2 loss lines in log ({len(losses)} found)"}
    first = losses[0]
    last = losses[-1]
    threshold = 0.75 * first
    passed = last < threshold
    return {
        "first_loss": first,
        "last_loss": last,
        "n_loss_lines": len(losses),
        "threshold_at_0.75x_first": threshold,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond", required=True, choices=CONDITION_IDS)
    ap.add_argument("--n-probes", type=int, default=10)
    ap.add_argument(
        "--train-log",
        type=str,
        default=None,
        help=(
            "Optional path to the corresponding train_<cond>.log file. If "
            "provided, the smoke gate adds a loss-decrease check "
            "(last_loss < 0.75 * first_loss). Round-2 fix Blocker 3."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    q_test = load_q_test_extended_50()
    q_demo = load_q_demo()
    r_villain = _load_R_villain()

    # Round-2 fix (Blocker 3): label-mask audit BEFORE the vLLM implant
    # check. Runs CPU-only, fail-loud if the loss surface is broken on this
    # arm (saves 5+ min of vLLM warmup on a hopeless run).
    logger.info("=== smoke gate 1/3: label-mask audit cond=%s ===", args.cond)
    label_audit = _label_mask_audit(
        cond=args.cond, tokenizer=tokenizer, r_villain=r_villain, q_demo=q_demo
    )
    logger.info(
        "label-mask audit cond=%s PASS: %d loss positions (marker+EOS), "
        "%d prompt-demo markers all masked",
        args.cond,
        label_audit["n_loss_positions"],
        label_audit["n_prompt_marker_positions"],
    )

    # Round-2 fix (Blocker 3): loss-decrease check (gate 2/3).
    if args.train_log:
        train_log_path = Path(args.train_log)
    else:
        train_log_path = Path(f"logs/issue_465/train_{args.cond}.log")
    logger.info("=== smoke gate 2/3: loss-decrease check (log=%s) ===", train_log_path)
    loss_check = _loss_decrease_check(train_log_path)
    if loss_check.get("passed") is True:
        logger.info(
            "loss-decrease PASS cond=%s: first=%.3f -> last=%.3f (< 0.75*first = %.3f)",
            args.cond,
            loss_check["first_loss"],
            loss_check["last_loss"],
            loss_check["threshold_at_0.75x_first"],
        )
    elif loss_check.get("passed") is False:
        logger.warning(
            "loss-decrease FAIL cond=%s: first=%.3f -> last=%.3f (>= 0.75*first = %.3f). "
            "Continuing to vLLM implant check; loss_check.pass=False will be in payload.",
            args.cond,
            loss_check["first_loss"],
            loss_check["last_loss"],
            loss_check["threshold_at_0.75x_first"],
        )
    else:
        logger.warning("loss-decrease SKIPPED cond=%s: %s", args.cond, loss_check.get("reason"))

    adapter_path = _resolve_adapter_path(args.cond)

    probe_qs = q_test[: args.n_probes]
    prompts_payload = []
    slot_positions = []
    for q in probe_qs:
        R_text = r_villain[q]["response_text"]
        full_ids, slot_L = build_eval_full_ids(
            condition=args.cond,
            eval_shape="in_trained_shape",
            target_q=q,
            R_villain_text=R_text,
            R_helpful_text=None,
            demo_pool=q_demo,
            r_demo=r_villain,
            demo_seed=137,
            tokenizer=tokenizer,
        )
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(slot_L)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=4096,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    def _extract(outs, label: str) -> tuple[list[float], int]:
        logps: list[float] = []
        n_arg = 0
        for out, L in zip(outs, slot_positions, strict=True):
            slot = out.prompt_logprobs[L]
            if slot is None:
                raise RuntimeError(
                    f"{label}: prompt_logprobs[{L}] is None on cond={args.cond}; "
                    f"list len={len(out.prompt_logprobs)}"
                )
            if MARKER_ID not in slot:
                raise RuntimeError(
                    f"{label}: MARKER_ID {MARKER_ID} not in prompt_logprobs[{L}] "
                    f"on cond={args.cond}; keys: {list(slot.keys())[:5]}"
                )
            logps.append(float(slot[MARKER_ID].logprob))
            top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
            if top_id == MARKER_ID:
                n_arg += 1
        return logps, n_arg

    lora_req = LoRARequest(lora_name=args.cond, lora_int_id=1, lora_path=adapter_path)
    outputs_trained = llm.generate(prompts_payload, sp, lora_request=lora_req)
    if len(outputs_trained) != args.n_probes:
        raise RuntimeError(
            f"vLLM trained returned {len(outputs_trained)} for {args.n_probes} probes."
        )
    trained_logps, n_argmax_marker = _extract(outputs_trained, "trained")

    outputs_base = llm.generate(prompts_payload, sp, lora_request=None)
    if len(outputs_base) != args.n_probes:
        raise RuntimeError(f"vLLM base returned {len(outputs_base)} for {args.n_probes} probes.")
    base_logps, _ = _extract(outputs_base, "base")

    implant_fraction = n_argmax_marker / args.n_probes
    trained_mean = sum(trained_logps) / len(trained_logps)
    base_mean = sum(base_logps) / len(base_logps)
    delta_mean = trained_mean - base_mean
    per_probe_deltas = [t - b for t, b in zip(trained_logps, base_logps, strict=True)]
    # Composite gate (plan §4.6 + round-2 Blocker 3): three checks.
    # gate 1: label-mask audit (already passed -- raise above on FAIL)
    # gate 2: loss-decrease (passed | failed | skipped)
    # gate 3: implant_fraction >= 0.80 (the load-bearing implant check)
    implant_pass = implant_fraction >= SMOKE_IMPLANT_FRAC
    loss_pass = loss_check.get("passed")  # True / False / None (skipped)
    # Composite: label_audit PASSed (we'd have raised otherwise), loss didn't
    # explicitly fail, implant >= threshold.
    composite_pass = (loss_pass is not False) and implant_pass
    payload = {
        "condition": args.cond,
        "n_probes": args.n_probes,
        "n_argmax_marker": n_argmax_marker,
        "implant_fraction": implant_fraction,
        "marker_logp_mean": trained_mean,
        "base_logp_mean": base_mean,
        "delta_logp_mean": delta_mean,
        "per_probe_trained_logps": trained_logps,
        "per_probe_base_logps": base_logps,
        "per_probe_deltas": per_probe_deltas,
        "threshold_fraction": SMOKE_IMPLANT_FRAC,
        "implant_pass": implant_pass,
        "label_mask_audit": label_audit,
        "loss_decrease_check": loss_check,
        "pass": composite_pass,
    }
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SMOKE_LOG_DIR / f"smoke_{args.cond}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "smoke implant cond=%s: %d/%d argmax==marker (frac=%.2f) "
        "trained_logp=%.2f base_logp=%.2f DELTA=%+.2f nats threshold=%.2f pass=%s -> %s",
        args.cond,
        n_argmax_marker,
        args.n_probes,
        implant_fraction,
        trained_mean,
        base_mean,
        delta_mean,
        SMOKE_IMPLANT_FRAC,
        payload["pass"],
        out_path,
    )
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
