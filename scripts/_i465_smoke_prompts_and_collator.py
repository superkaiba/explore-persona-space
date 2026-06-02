"""CPU smoke for #465 — prompt builders + label-mask audit on the 4 arms x 5 reads.

This file is NOT shipped. It exercises every CPU-feasible part of the pipeline
on a tiny slice (3 q's) without GPU. Run via:

    uv run python scripts/_i465_smoke_prompts_and_collator.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_K,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("i465.smoke")

BASE = "Qwen/Qwen2.5-7B-Instruct"


def fake_R_villain(qs: list[str]) -> dict[str, dict]:
    """Build a synthetic R_villain dict for CPU smoke (no GPU vLLM available)."""
    out = {}
    for q in qs:
        out[q] = {
            "response_text": f"As a villainous mastermind, my answer to '{q[:30]}...' "
            "involves complex schemes and cunning. The world shall tremble!",
            "response_token_ids": [],
            "n_response_tokens": 0,
            "ended_with_eos": True,
            "truncated": False,
            "marker_in_R": False,
        }
    return out


def fake_R_helpful(qs: list[str]) -> dict[str, dict]:
    out = {}
    for q in qs:
        out[q] = {
            "response_text": f"Helpful answer to '{q[:30]}...': here are some clear, "
            "useful points. Hope this helps!",
            "response_token_ids": [],
            "n_response_tokens": 0,
            "ended_with_eos": True,
            "truncated": False,
            "marker_in_R": False,
        }
    return out


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [MARKER_ID], f"marker drift: {ids}"
    log.info("marker OK")

    q_train_keys = sorted(load_q_train_answers().keys())
    q_test = load_q_test_extended_50()
    q_demo = load_q_demo()
    all_villain_qs = q_train_keys + q_test + q_demo
    r_villain = fake_R_villain(all_villain_qs)
    r_helpful = fake_R_helpful(q_test)

    log.info(
        "loaded Q_train=%d Q_test=%d Q_demo=%d",
        len(q_train_keys),
        len(q_test),
        len(q_demo),
    )

    # ── (1) Training-row builder smoke per arm ──────────────────────────
    log.info("=== smoke 1: training-row builder per arm ===")
    sample_q = q_train_keys[0]
    for cond in CONDITION_IDS:
        prompt_msgs, completion_msgs = build_training_messages(
            condition=cond,
            target_q=sample_q,
            target_R_text=r_villain[sample_q]["response_text"],
            demo_pool=q_demo,
            r_demo=r_villain,
            train_seed=42,
        )
        # System role + 2*(k) demo turns + 1 user turn = 1 + 2k + 1 turns in prompt.
        k = CONDITION_K[cond]
        expected_prompt_len = 1 + 2 * k + 1
        assert len(prompt_msgs) == expected_prompt_len, (
            f"cond={cond} prompt has {len(prompt_msgs)} msgs, expected {expected_prompt_len}"
        )
        assert len(completion_msgs) == 1
        assert completion_msgs[0]["content"].endswith(MARKER_TEXT)
        full = list(prompt_msgs) + list(completion_msgs)
        text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        expected_marker = 1 + k  # k in demos + 1 in completion
        assert marker_count == expected_marker, (
            f"cond={cond}: encoded {marker_count} markers, expected {expected_marker}"
        )
        log.info(
            "cond=%s prompt_turns=%d markers=%d/%d -- OK",
            cond,
            len(prompt_msgs),
            marker_count,
            expected_marker,
        )

    # ── (2) Eval prompt builder per (cond, shape) ───────────────────────
    log.info("=== smoke 2: eval prompt builder per (cond, shape) ===")
    eval_q = q_test[0]
    shapes = [
        ("in_trained_shape", False),
        ("generalization", False),
        ("demo_free_default", False),
        ("demo_free_default_villain_R", False),
        ("non_marker_demo", True),  # cond2_k1/k3 only
    ]
    for cond in CONDITION_IDS:
        k = CONDITION_K[cond]
        for shape, is_cond2_only in shapes:
            if is_cond2_only and cond not in ("cond2_k1", "cond2_k3"):
                continue
            try:
                full_ids, slot_L = build_eval_full_ids(
                    condition=cond,
                    eval_shape=shape,
                    target_q=eval_q,
                    R_villain_text=r_villain[eval_q]["response_text"],
                    R_helpful_text=r_helpful[eval_q]["response_text"]
                    if shape == "demo_free_default"
                    else None,
                    demo_pool=q_demo,
                    r_demo=r_villain,
                    demo_seed=137,
                    tokenizer=tokenizer,
                )
            except Exception as e:
                log.error("FAIL cond=%s shape=%s: %s", cond, shape, e)
                raise
            assert full_ids[-1] == MARKER_ID, f"trailing not marker: cond={cond} shape={shape}"
            assert slot_L == len(full_ids) - 1
            # Marker-count check.
            cnt = full_ids.count(MARKER_ID)
            if shape in ("demo_free_default", "demo_free_default_villain_R", "non_marker_demo"):
                expected = 1
            elif cond in ("cond2_k1", "cond2_k3") and shape in (
                "in_trained_shape",
                "generalization",
            ):
                expected = k + 1
            else:
                expected = 1
            assert cnt == expected, f"cond={cond} shape={shape}: markers={cnt} expected={expected}"
            log.info(
                "cond=%s shape=%s slot=%d markers=%d (expected=%d) -- OK",
                cond,
                shape,
                slot_L,
                cnt,
                expected,
            )

    # ── (3) Label-mask audit: cond2_k3 row through MarkerOnlyDataCollator ─
    log.info("=== smoke 3: label-mask audit (cond2_k3 -- the worst case k=3) ===")
    import torch

    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    # Build one cond2_k3 training row.
    cond = "cond2_k3"
    sample_q = q_train_keys[0]
    prompt_msgs, completion_msgs = build_training_messages(
        condition=cond,
        target_q=sample_q,
        target_R_text=r_villain[sample_q]["response_text"],
        demo_pool=q_demo,
        r_demo=r_villain,
        train_seed=42,
    )
    full_msgs = list(prompt_msgs) + list(completion_msgs)
    text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    log.info(
        "cond2_k3 sample row: %d tokens, %d markers", len(input_ids), input_ids.count(MARKER_ID)
    )

    # MarkerOnlyDataCollator wraps an inner collator that masks the prompt.
    # We can't easily reproduce TRL's full prompt-completion masking without
    # SFTTrainer, so we fake it: prompt = everything before the LAST
    # assistant turn, completion = the last assistant turn.
    # First find the slot where the completion starts in the full encoded text.
    prompt_only_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_only_text, add_special_tokens=False)
    completion_start = len(prompt_ids)

    # labels = -100 for prompt, copy of input_ids for completion (TRL's pattern).
    labels = [-100] * completion_start + input_ids[completion_start:]
    # Pad to match length.
    if len(labels) < len(input_ids):
        labels = labels + [-100] * (len(input_ids) - len(labels))
    elif len(labels) > len(input_ids):
        labels = labels[: len(input_ids)]

    # The expected number of MARKERS in the completion is exactly 1 (the
    # trailing one). The k=3 prompt markers are MASKED already by TRL's
    # response-only logic.
    n_marker_in_completion = sum(
        1 for i, t in enumerate(input_ids) if t == MARKER_ID and labels[i] != -100
    )
    assert n_marker_in_completion == 1, (
        f"completion has {n_marker_in_completion} marker positions, expected exactly 1"
    )

    # Now run the MarkerOnlyDataCollator manually on a fake batch.
    class _IdentityCollator:
        def __call__(self, features):
            return {
                "input_ids": torch.tensor([features[0]["input_ids"]], dtype=torch.long),
                "labels": torch.tensor([features[0]["labels"]], dtype=torch.long),
            }

    inner = _IdentityCollator()
    collator = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
    )
    batch = collator([{"input_ids": input_ids, "labels": labels}])
    final_labels = batch["labels"][0].tolist()
    loss_positions = [i for i, lab in enumerate(final_labels) if lab != -100]
    log.info(
        "cond2_k3 final loss positions: %d (expected exactly 2: marker + EOS)",
        len(loss_positions),
    )
    assert len(loss_positions) == 2, (
        f"expected exactly 2 loss-bearing positions (marker + EOS), got {len(loss_positions)}: "
        f"positions={loss_positions}, tokens={[input_ids[p] for p in loss_positions]}"
    )
    assert input_ids[loss_positions[0]] == MARKER_ID, (
        f"first loss position is NOT the marker: token id {input_ids[loss_positions[0]]}"
    )
    log.info(
        "label-mask audit cond2_k3 PASS: 2 loss positions = [marker=%d, eos=%d]",
        input_ids[loss_positions[0]],
        input_ids[loss_positions[1]],
    )

    # Verify zero loss-bearing markers in the PROMPT (the k=3 demo markers).
    prompt_marker_positions = [i for i in range(completion_start) if input_ids[i] == MARKER_ID]
    log.info("cond2_k3 prompt has %d marker positions (= k)", len(prompt_marker_positions))
    assert len(prompt_marker_positions) == 3, (
        f"expected k=3 prompt markers, got {len(prompt_marker_positions)}"
    )
    for p in prompt_marker_positions:
        assert final_labels[p] == -100, (
            f"prompt marker at position {p} is loss-bearing -- TRL response-only mask broken"
        )
    log.info("all k=3 demo markers in prompt are masked to -100 -- OK")

    # ── (4) Trajectory callback dry-run (import + class creation only) ──
    log.info("=== smoke 4: trajectory callback class creation ===")
    from explore_persona_space.train.i465_trajectory import make_trajectory_callback_class

    cls = make_trajectory_callback_class()
    cb = cls(
        condition_name="cond1",
        shape_probes={"in_trained_shape": [[1, 2, 3, MARKER_ID]]},
        marker_id=MARKER_ID,
        log_every=10,
    )
    assert cb.marker_id == MARKER_ID
    assert cb.log_every == 10
    log.info("trajectory callback class instantiated OK")

    # Save a smoke-output sentinel.
    out = Path("logs/issue_465/smoke_cpu.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "marker_id_ok": True,
                "q_train": len(q_train_keys),
                "q_test": len(q_test),
                "q_demo": len(q_demo),
                "training_row_builder_ok": True,
                "eval_prompt_builder_ok": True,
                "label_mask_audit_ok": True,
                "trajectory_callback_ok": True,
            },
            indent=2,
        )
    )
    log.info("ALL CPU SMOKES PASS -- sentinel at %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
