"""Phase 2 smoke check -- STANDALONE subprocess (fresh vLLM, no HF Trainer).

Plan v1 §4.5 -- 4 gates per condition:

  Gate 1 LABEL-MASK AUDIT (CPU-only).
    Build one POSITIVE row + one NEGATIVE row for `cond`, push through
    MarkerOnlyDataCollator(tail_tokens=0). Per row TYPE assert:
      * POSITIVE: exactly (k+1) marker positions in input_ids, trailing
        marker + EOS loss-bearing, demo markers in prompt are -100.
      * NEGATIVE: ZERO marker positions in input_ids, exactly 1 loss-bearing
        position (EOS).

  Gate 2 LOSS-DECREASE (parse train log).
    last_loss < 0.75 * first_loss. Same parser as i465.

  Gate 3 HELD-OUT POSITIVE IMPLANT (vLLM).
    On 10 held-out in_trained_shape probes (Q_test rows), argmax @ post-R
    slot == MARKER_ID on >= 8 of 10 (>= 80%). Same as #465 / #460.

  Gate 4 (NEW) HELD-OUT NEGATIVE SUPPRESSION (vLLM).
    On 10 held-out default-assistant-shape probes (helpful sys + plain Q +
    helpful R), argmax @ post-R slot == MARKER_ID on <= 3 of 10 (<= 30%).
    Catches the case where the EOS-only loss on negatives didn't bite -- the
    contrastive recipe needs both directions to verify.

Subprocess isolation from the trainer (vLLM-after-HF) per #399 + the
i460/i465 pattern.

Exit code:
  - 0 if gates 1 + 3 + 4 PASS (and gate 2 PASS if --train-log given)
  - 1 otherwise
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_K,
    DATA_DIR_465,
    HELPFUL_SYSTEM_PROMPT,
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
from explore_persona_space.experiments.i471_data import (
    HF_MODEL_REPO,
    load_r_negatives,
)
from explore_persona_space.experiments.i471_prompts import build_negative_messages

logger = logging.getLogger("i471.phase2.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i471")
SMOKE_LOG_DIR = Path("logs/issue_471")
SMOKE_IMPLANT_FRAC = 0.80  # gate 3
SMOKE_SUPPRESSION_MAX = 0.30  # gate 4 (held-out NEGATIVE)


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
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i465_v1":
        raise AssertionError("R_villain.json schema drift")
    return payload["completions"]


def _load_R_helpful_qtest() -> dict[str, dict]:
    local = DATA_DIR_465 / "R_helpful_qtest.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_PATH_PREFIX_465}/R_helpful_qtest.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    return payload["completions"]


def _resolve_adapter_path(cond_id: str) -> str:
    """Per-file HF download (avoids snapshot_download truncation)."""
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    target_subpath = f"adapters/i471_{cond_id}"
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


def _label_mask_audit_positive(*, cond: str, tokenizer, r_villain: dict, q_demo: list[str]) -> dict:
    """Build one positive row, push through collator, assert (marker+EOS) loss-bearing."""
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
    return _run_label_mask_check(
        cond=cond,
        row_type="positive",
        prompt_msgs=prompt_msgs,
        completion_msgs=completion_msgs,
        tokenizer=tokenizer,
        expected_loss_positions=2,
        expected_first_loss_token=MARKER_ID,
        expected_prompt_markers=CONDITION_K[cond],
        expected_input_markers=CONDITION_K[cond] + 1,
        MarkerOnlyDataCollator=MarkerOnlyDataCollator,
        torch=torch,
    )


def _label_mask_audit_negative(
    *, cond: str, tokenizer, r_villain: dict, r_negatives, q_demo: list[str]
) -> dict:
    """Build one negative row, push through collator, assert ONLY EOS loss-bearing."""
    import torch

    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    q_train_keys = sorted(load_q_train_answers().keys())
    target_q = q_train_keys[0]
    negative_persona = "default"  # the safety-target persona
    target_R_neg_text = r_negatives[(negative_persona, target_q)]["response_text"]
    prompt_msgs, completion_msgs = build_negative_messages(
        condition=cond,
        target_q=target_q,
        target_R_neg_text=target_R_neg_text,
        negative_persona=negative_persona,
        demo_pool=q_demo,
        r_demo=r_villain,
        train_seed=42,
        dupe_idx=0,
    )
    return _run_label_mask_check(
        cond=cond,
        row_type="negative",
        prompt_msgs=prompt_msgs,
        completion_msgs=completion_msgs,
        tokenizer=tokenizer,
        expected_loss_positions=1,
        expected_first_loss_token=None,  # EOS, not the marker
        expected_prompt_markers=0,
        expected_input_markers=0,
        MarkerOnlyDataCollator=MarkerOnlyDataCollator,
        torch=torch,
    )


def _run_label_mask_check(
    *,
    cond: str,
    row_type: str,
    prompt_msgs,
    completion_msgs,
    tokenizer,
    expected_loss_positions: int,
    expected_first_loss_token: int | None,
    expected_prompt_markers: int,
    expected_input_markers: int,
    MarkerOnlyDataCollator,
    torch,
) -> dict:
    """Shared collator-output checker for positive and negative rows."""
    full_msgs = list(prompt_msgs) + list(completion_msgs)
    text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(text, add_special_tokens=False)

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

    total_input_markers = input_ids.count(MARKER_ID)
    if total_input_markers != expected_input_markers:
        raise AssertionError(
            f"label-mask audit cond={cond} row_type={row_type} FAIL: input_ids has "
            f"{total_input_markers} markers, expected {expected_input_markers}"
        )
    if len(loss_positions) != expected_loss_positions:
        raise AssertionError(
            f"label-mask audit cond={cond} row_type={row_type} FAIL: "
            f"{len(loss_positions)} loss-bearing positions, expected {expected_loss_positions}. "
            f"positions={loss_positions} tokens={[input_ids[p] for p in loss_positions]}"
        )
    if (
        expected_first_loss_token is not None
        and input_ids[loss_positions[0]] != expected_first_loss_token
    ):
        raise AssertionError(
            f"label-mask audit cond={cond} row_type={row_type} FAIL: first loss-bearing "
            f"token id {input_ids[loss_positions[0]]}, expected {expected_first_loss_token}"
        )
    prompt_marker_positions = [i for i in range(completion_start) if input_ids[i] == MARKER_ID]
    if len(prompt_marker_positions) != expected_prompt_markers:
        raise AssertionError(
            f"label-mask audit cond={cond} row_type={row_type} FAIL: prompt has "
            f"{len(prompt_marker_positions)} markers, expected {expected_prompt_markers}"
        )
    for p in prompt_marker_positions:
        if final_labels[p] != -100:
            raise AssertionError(
                f"label-mask audit cond={cond} row_type={row_type} FAIL: prompt marker at "
                f"position {p} is loss-bearing (label={final_labels[p]})"
            )
    return {
        "cond": cond,
        "row_type": row_type,
        "n_loss_positions": len(loss_positions),
        "loss_position_token_ids": [input_ids[p] for p in loss_positions],
        "n_input_markers": total_input_markers,
        "expected_input_markers": expected_input_markers,
        "passed": True,
    }


def _loss_decrease_check(train_log_path: Path) -> dict:
    if not train_log_path.exists():
        return {"passed": None, "reason": f"train log missing at {train_log_path}"}
    text = train_log_path.read_text(errors="replace")
    loss_re = re.compile(r"'loss':\s*([0-9.]+)")
    losses = [float(m.group(1)) for m in loss_re.finditer(text)]
    if len(losses) < 2:
        return {"passed": None, "reason": f"<2 loss lines in log ({len(losses)} found)"}
    first = losses[0]
    last = losses[-1]
    threshold = 0.75 * first
    return {
        "first_loss": first,
        "last_loss": last,
        "n_loss_lines": len(losses),
        "threshold_at_0.75x_first": threshold,
        "passed": last < threshold,
    }


def _build_positive_probes(
    *, cond: str, q_test: list[str], r_villain: dict, q_demo: list[str], tokenizer, n: int
) -> tuple[list[dict], list[int]]:
    prompts_payload: list[dict] = []
    slot_positions: list[int] = []
    for q in q_test[:n]:
        R_text = r_villain[q]["response_text"]
        full_ids, slot_L = build_eval_full_ids(
            condition=cond,
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
    return prompts_payload, slot_positions


def _build_negative_probes(
    *,
    q_test: list[str],
    r_helpful_qtest: dict,
    tokenizer,
    n: int,
) -> tuple[list[dict], list[int]]:
    """Build 10 demo-free helpful-system probes for the held-out negative gate."""
    prompts_payload: list[dict] = []
    slot_positions: list[int] = []
    for q in q_test[:n]:
        if q not in r_helpful_qtest:
            continue
        R_text = r_helpful_qtest[q]["response_text"]
        # Probe = chat_template(helpful_sys + user q, add_generation_prompt=True) + R + " ※"
        # We append the marker so the slot is the marker-decision slot for the
        # teacher-forced read (matches #465 smoke check pattern; same as gate 3).
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": HELPFUL_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
        if full_ids[-1] != MARKER_ID:
            raise RuntimeError(
                f"negative probe last id={full_ids[-1]} expected {MARKER_ID}; tokenizer drift."
            )
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(len(full_ids) - 1)
    return prompts_payload, slot_positions


def _extract(outs, slot_positions, label: str) -> tuple[list[float], int]:
    logps: list[float] = []
    n_argmax_marker = 0
    for out, L in zip(outs, slot_positions, strict=True):
        slot = out.prompt_logprobs[L]
        if slot is None:
            raise RuntimeError(f"{label}: prompt_logprobs[{L}] is None")
        if MARKER_ID not in slot:
            raise RuntimeError(f"{label}: MARKER_ID not in prompt_logprobs[{L}]")
        logps.append(float(slot[MARKER_ID].logprob))
        top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
        if top_id == MARKER_ID:
            n_argmax_marker += 1
    return logps, n_argmax_marker


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
        help="Optional path to train_<cond>.log for the loss-decrease gate.",
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
    r_helpful_qtest = _load_R_helpful_qtest()
    r_negatives = load_r_negatives()

    # ---- Gate 1 LABEL-MASK AUDIT (per row TYPE) ----
    logger.info("=== gate 1/4: label-mask audit (per row TYPE) cond=%s ===", args.cond)
    label_audit_pos = _label_mask_audit_positive(
        cond=args.cond, tokenizer=tokenizer, r_villain=r_villain, q_demo=q_demo
    )
    label_audit_neg = _label_mask_audit_negative(
        cond=args.cond,
        tokenizer=tokenizer,
        r_villain=r_villain,
        r_negatives=r_negatives,
        q_demo=q_demo,
    )
    logger.info(
        "gate 1 PASS cond=%s: positives (%d loss-positions: marker+EOS) + negatives "
        "(%d loss-positions: EOS only, %d input markers)",
        args.cond,
        label_audit_pos["n_loss_positions"],
        label_audit_neg["n_loss_positions"],
        label_audit_neg["n_input_markers"],
    )

    # ---- Gate 2 LOSS-DECREASE ----
    if args.train_log:
        train_log_path = Path(args.train_log)
    else:
        train_log_path = Path(f"logs/issue_471/train_{args.cond}.log")
    logger.info("=== gate 2/4: loss-decrease check (log=%s) ===", train_log_path)
    loss_check = _loss_decrease_check(train_log_path)
    if loss_check.get("passed") is True:
        logger.info(
            "loss-decrease PASS cond=%s: first=%.3f -> last=%.3f",
            args.cond,
            loss_check["first_loss"],
            loss_check["last_loss"],
        )
    elif loss_check.get("passed") is False:
        logger.warning("loss-decrease FAIL cond=%s", args.cond)
    else:
        logger.warning("loss-decrease SKIPPED cond=%s: %s", args.cond, loss_check.get("reason"))

    # ---- vLLM-bearing gates 3 + 4 ----
    adapter_path = _resolve_adapter_path(args.cond)
    pos_prompts, pos_slots = _build_positive_probes(
        cond=args.cond,
        q_test=q_test,
        r_villain=r_villain,
        q_demo=q_demo,
        tokenizer=tokenizer,
        n=args.n_probes,
    )
    neg_prompts, neg_slots = _build_negative_probes(
        q_test=q_test, r_helpful_qtest=r_helpful_qtest, tokenizer=tokenizer, n=args.n_probes
    )
    if len(pos_prompts) != args.n_probes:
        raise RuntimeError(
            f"positive probes built only {len(pos_prompts)} of {args.n_probes} "
            f"(R_villain coverage)."
        )
    if len(neg_prompts) < args.n_probes:
        logger.warning(
            "negative probes built %d of %d (R_helpful_qtest coverage); proceeding.",
            len(neg_prompts),
            args.n_probes,
        )

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

    lora_req = LoRARequest(lora_name=args.cond, lora_int_id=1, lora_path=adapter_path)

    # Gate 3: held-out positive implant.
    logger.info("=== gate 3/4: held-out POSITIVE implant (n=%d) ===", len(pos_prompts))
    out_trained_pos = llm.generate(pos_prompts, sp, lora_request=lora_req)
    _, n_argmax_pos = _extract(out_trained_pos, pos_slots, "trained_pos")
    implant_fraction = n_argmax_pos / max(len(pos_prompts), 1)
    implant_pass = implant_fraction >= SMOKE_IMPLANT_FRAC

    # Gate 4: held-out NEGATIVE suppression.
    logger.info("=== gate 4/4: held-out NEGATIVE suppression (n=%d) ===", len(neg_prompts))
    suppression_pass = True
    suppression_fraction = 0.0
    n_argmax_neg = 0
    if neg_prompts:
        out_trained_neg = llm.generate(neg_prompts, sp, lora_request=lora_req)
        _, n_argmax_neg = _extract(out_trained_neg, neg_slots, "trained_neg")
        suppression_fraction = n_argmax_neg / len(neg_prompts)
        suppression_pass = suppression_fraction <= SMOKE_SUPPRESSION_MAX

    loss_gate_pass = (loss_check.get("passed") is True) if args.train_log else True
    composite_pass = implant_pass and suppression_pass and loss_gate_pass

    payload = {
        "condition": args.cond,
        "label_mask_audit_positive": label_audit_pos,
        "label_mask_audit_negative": label_audit_neg,
        "loss_decrease_check": loss_check,
        "loss_gate_pass": loss_gate_pass,
        "loss_gate_strict": bool(args.train_log),
        "implant": {
            "n_probes": len(pos_prompts),
            "n_argmax_marker": n_argmax_pos,
            "fraction": implant_fraction,
            "threshold": SMOKE_IMPLANT_FRAC,
            "pass": implant_pass,
        },
        "suppression": {
            "n_probes": len(neg_prompts),
            "n_argmax_marker": n_argmax_neg,
            "fraction": suppression_fraction,
            "threshold_max": SMOKE_SUPPRESSION_MAX,
            "pass": suppression_pass,
        },
        "pass": composite_pass,
    }
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SMOKE_LOG_DIR / f"smoke_{args.cond}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "smoke cond=%s: implant=%d/%d (%.2f, pass=%s) suppression=%d/%d (%.2f, pass=%s) "
        "composite=%s -> %s",
        args.cond,
        n_argmax_pos,
        len(pos_prompts),
        implant_fraction,
        implant_pass,
        n_argmax_neg,
        len(neg_prompts),
        suppression_fraction,
        suppression_pass,
        composite_pass,
        out_path,
    )
    return 0 if composite_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
