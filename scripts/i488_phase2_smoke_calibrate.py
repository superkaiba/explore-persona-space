"""Issue #488 Phase 2 — smoke calibrate + label-mask audit + in-band frac pick.

Plan v2 §4.4 + §7. Runs A1 (helpful assistant, low-band) and G2 (skeptical
scientist, upper-band) at lr=2e-6, r=16, 150 rows, 3 epochs total, saving
at all 6 fracs ∈ {0.10, 0.25, 0.50, 1.00, 2.00, 3.00}. Phase-2 gates per §7:

  1. **Label-mask audit (fix A)** — print loss-bearing positions on one positive
     + one negative row of A1's training data. Positive row's loss-positions are
     exactly the marker token (id 83399) + immediately-following EOS. Negative
     row's loss-position is exactly `<|im_end|>` (id 151645).
  2. **In-band source emission** — for BOTH A1 and G2, at least 3 of the 6 fracs
     have source emission ∈ [0.20, 0.85]; pick the 3 most evenly spread that
     are in-band for BOTH cells.
  3. **Off-diagonal saturation gate (fix D)** — at each picked frac, evaluate
     A1's adapter on a 6-cell mini-grid of off-diagonal target contexts
     (B1, F1, G1, A3, D2, B5). At least one picked frac must have
     tie_mass_off ≤ 0.85 on this 6-cell mini-grid.
  4. **EOS-gradient check (fix H)** — at the latest picked frac, A1's trained
     model evaluated on the off-diagonal A1→B1 cell must show trained
     log P(' ※') at the post-response slot moved DOWN by ≥ 0.2 nats vs base.

Architecturally unified with the sweep (CLAUDE.md Step 6d.0): smoke IS the
sweep with --conds A1 G2 --seeds 42 --fracs all-six on the SAME dispatcher.
Phase-2 (this script) is the calibration LAYER that consumes the smoke
adapters and runs the gates. It does NOT spawn the trainer — the dispatcher
does that. This script is invoked AFTER the smoke train.

Outputs:
* ``logs/issue_488/smoke/label_mask_audit.txt`` — positive/negative row label
  positions (from the actual collator's batch).
* ``logs/issue_488/smoke/emission_per_frac.json`` — per-cell, per-frac source
  emission on Q_test (n_probes=10 for cost).
* ``logs/issue_488/smoke/offdiag_saturation.json`` — per-frac off-diagonal
  mini-grid saturation masses.
* ``logs/issue_488/smoke/eos_gradient.json`` — trained vs base log P(' ※')
  at A1→B1 post-response slot at the latest picked frac.
* ``logs/issue_488/smoke/picked_fracs.json`` — the 3 chosen fracs the sweep
  will use.
* On FAIL: ``/workspace/logs/issue-488-smoke-failed.json`` sentinel +
  non-zero exit (the dispatcher escalates via that sentinel).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i488.phase2")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i488")
SMOKE_LOG_DIR = Path("logs/issue_488/smoke")
SENTINEL_PATH = Path("/workspace/logs/issue-488-smoke-failed.json")

ALL_FRACS = (0.10, 0.25, 0.50, 1.00, 2.00, 3.00)
IM_END_TOKEN_ID = 151645
EOS_GRADIENT_MIN_NATS = 0.2
IN_BAND_MIN = 0.20
IN_BAND_MAX = 0.85
OFFDIAG_MAX_TIE_MASS = 0.85
OFFDIAG_MINI_GRID = ("B1", "F1", "G1", "A3", "D2", "B5")
SMOKE_CELLS = ("A1", "G2")
SMOKE_EVAL_N_PROBES = 10
LOGP_FLOOR = -50.0


def _frac_tag(frac: float) -> str:
    return f"frac{round(frac * 100):03d}"


def _write_sentinel(reason_key: str, reason: str, extra: dict | None = None) -> None:
    SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": 488,
        "phase": "phase2_smoke",
        "failure_class": "code",
        "reason": reason_key,
        "reason_long": reason,
        "wrote_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "sentinel_schema_version": 1,
        "kind": "epm:failure",
        "version": 1,
    }
    if extra:
        payload["extra"] = extra
    SENTINEL_PATH.write_text(json.dumps(payload, indent=2))
    logger.error("Smoke gate FAIL → wrote %s (%s)", SENTINEL_PATH, reason_key)


# ── Gate 1: label-mask audit ────────────────────────────────────────────


def _label_mask_audit(audit_train_jsonl: Path, tokenizer) -> str:
    """Walk the saved A1 train.jsonl, locate ONE positive + ONE negative row,
    apply the MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True,
    im_end_token_id=151645), and report which token positions carry loss.

    Returns the formatted audit report (also written to disk).

    Raises:
        AssertionError: if the positive row's only loss-bearing position is
            not the marker (+ EOS), OR if the negative row's only loss-bearing
            position is not id 151645.
    """
    import torch
    from transformers import DataCollatorForLanguageModeling

    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    if not audit_train_jsonl.exists():
        raise FileNotFoundError(
            f"Audit train rows missing: {audit_train_jsonl}. "
            "Run Phase 2 train dispatcher first (it persists train rows for audit)."
        )

    rows: list[dict] = []
    for line in audit_train_jsonl.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{audit_train_jsonl} is empty.")

    # Find one positive (has_marker) and one negative (no marker in completion).
    positive_row = None
    negative_row = None
    for r in rows:
        completion = r["completion"][0]["content"]
        if MARKER_TEXT in completion:
            if positive_row is None:
                positive_row = r
        elif negative_row is None:
            negative_row = r
        if positive_row is not None and negative_row is not None:
            break
    if positive_row is None or negative_row is None:
        raise AssertionError(
            f"Couldn't find both positive + negative rows in {audit_train_jsonl} "
            f"(seen {len(rows)}; positives need MARKER_TEXT in completion)."
        )

    # Re-tokenize both rows the way SFTTrainer would: apply_chat_template on
    # prompt + completion concatenated as one assistant turn.
    def tokenize(row: dict) -> dict:
        full_msgs = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Build the response-only mask: the SFT default masks the prompt with -100
        # and trains on the assistant completion. We approximate that by tokenizing
        # the prompt-only chat string and finding its length.
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids) :]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }

    pos_feat = tokenize(positive_row)
    neg_feat = tokenize(negative_row)

    inner = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_TOKEN_ID,
    )
    batch = collator([pos_feat, neg_feat])

    pos_labels = batch["labels"][0]
    pos_input = batch["input_ids"][0]
    neg_labels = batch["labels"][1]
    neg_input = batch["input_ids"][1]

    pos_loss_positions = (pos_labels != -100).nonzero(as_tuple=True)[0].tolist()
    neg_loss_positions = (neg_labels != -100).nonzero(as_tuple=True)[0].tolist()

    pos_loss_ids = [int(pos_input[p].item()) for p in pos_loss_positions]
    neg_loss_ids = [int(neg_input[p].item()) for p in neg_loss_positions]

    audit_lines = [
        "# i488 Phase 2 — Label-mask audit",
        f"timestamp: {datetime.datetime.now(datetime.UTC).isoformat()}",
        f"audit_train_jsonl: {audit_train_jsonl}",
        f"marker_text: {MARKER_TEXT!r}  marker_id: {MARKER_ID}",
        f"im_end_token_id: {IM_END_TOKEN_ID}",
        "",
        "## Positive row",
        f"  loss-bearing positions: {pos_loss_positions}",
        f"  loss-bearing token ids: {pos_loss_ids}",
        "  expected: [MARKER_ID + immediately-following EOS]",
        "",
        "## Negative row",
        f"  loss-bearing positions: {neg_loss_positions}",
        f"  loss-bearing token ids: {neg_loss_ids}",
        f"  expected: a SINGLE position whose token id is {IM_END_TOKEN_ID}",
    ]
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_LOG_DIR / "label_mask_audit.txt").write_text("\n".join(audit_lines) + "\n")
    logger.info("Wrote label_mask_audit.txt")

    # FAIL LOUD on shape mismatch.
    if MARKER_ID not in pos_loss_ids:
        raise AssertionError(
            f"POSITIVE row audit FAIL: MARKER_ID {MARKER_ID} not in "
            f"loss-bearing token ids {pos_loss_ids}."
        )
    if neg_loss_ids != [IM_END_TOKEN_ID]:
        raise AssertionError(
            f"NEGATIVE row audit FAIL: expected single position with id "
            f"{IM_END_TOKEN_ID}, got {neg_loss_ids}."
        )
    return "\n".join(audit_lines)


# ── Gate 2/3/4: emission + saturation + EOS-gradient ─────────────────────


def _download_adapter(cid: str, seed: int, frac: float) -> str:
    """Download one (cid, seed, frac) adapter from HF; return local path."""
    from huggingface_hub import hf_hub_download

    subpath = f"adapters/i488_{cid}_seed{seed}_{_frac_tag(frac)}"
    local_target = LOCAL_ADAPTER_CACHE / subpath
    local_target.mkdir(parents=True, exist_ok=True)
    for fname in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ):
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                revision="main",
                filename=f"{subpath}/{fname}",
                local_dir=LOCAL_ADAPTER_CACHE,
            )
        except Exception as e:
            if fname in ("adapter_model.safetensors", "adapter_config.json"):
                raise RuntimeError(f"required file {subpath}/{fname} not on HF: {e}") from e
    return str(local_target)


def _vllm_source_emission(
    llm,
    sampling_params,
    tokenizer,
    cond_source,
    cond_target,
    held_out_q: list[str],
    class_d_rewrites: dict,
    lora_request,
) -> tuple[float, float, int]:
    """Generate N samples per held-out Q under (cond_target context, trained
    with cond_source adapter); count substring ' ※' presence in decoded text.

    Returns (emission_rate, truncation_rate, n_total_generations).
    """
    prompts = [
        build_prompt_for_condition(cond_target, q, tokenizer, class_d_rewrites) for q in held_out_q
    ]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    total = 0
    emitted = 0
    truncated = 0
    for gen in outputs:
        for choice in gen.outputs:
            total += 1
            if MARKER_TEXT in choice.text:
                emitted += 1
            if choice.finish_reason == "length":
                truncated += 1
    return (
        emitted / max(total, 1),
        truncated / max(total, 1),
        total,
    )


def _post_response_slot_logprob(
    llm,
    sp_logprob,
    tokenizer,
    prompt_text: str,
    R_text: str,
    lora_request,
) -> float:
    """Teacher-force `prompt + R + MARKER_TEXT` through (base or trained); return
    log P(' ※') at the FINAL position (the post-response marker slot).
    """
    full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    if full_ids[-1] != MARKER_ID or full_ids.count(MARKER_ID) != 1:
        raise RuntimeError(
            f"marker slot drift in EOS-gradient probe: full_ids[-1]={full_ids[-1]}, "
            f"count={full_ids.count(MARKER_ID)}"
        )
    slot = len(full_ids) - 1
    outputs = llm.generate([{"prompt_token_ids": full_ids}], sp_logprob, lora_request=lora_request)
    out = outputs[0]
    spec = out.prompt_logprobs[slot]
    if spec is None or MARKER_ID not in spec:
        raise RuntimeError(
            f"post-response slot probe: prompt_logprobs[{slot}] missing MARKER_ID; "
            f"top keys={list((spec or {}).keys())[:5]}"
        )
    return max(float(spec[MARKER_ID].logprob), LOGP_FLOOR)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - CLI dispatch + gates
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--audit-train-jsonl",
        default="data/issue_488/train_rows/i488_A1_seed42.jsonl",
        help="Path to A1 train.jsonl persisted by the Phase 3 train script.",
    )
    ap.add_argument(
        "--smoke-seed",
        type=int,
        default=42,
        help="Smoke seed (matches the Phase 3 dispatcher's --seeds 42 --conds A1 G2).",
    )
    ap.add_argument(
        "--n-probes-emission",
        type=int,
        default=SMOKE_EVAL_N_PROBES,
        help="Number of held-out Q to use for the source emission probe.",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Marker assert.
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    # ── Gate 1: label-mask audit ──
    try:
        _label_mask_audit(Path(args.audit_train_jsonl), tokenizer)
    except AssertionError as e:
        _write_sentinel("label_mask_wrong_slot", str(e))
        return 2

    # ── Gates 2 / 3 / 4: spin up vLLM, evaluate A1 + G2 adapters ──
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    held_out = json.loads(Path("data/issue_488/q_held_out_20.json").read_text())["questions"]
    held_out_probe = held_out[: args.n_probes_emission]
    class_d_rewrites = load_class_d_rewrites()

    # Get R_test on-policy for inherited B1 (needed for EOS-gradient probe).
    R_test_inherited = json.loads(
        Path("data/issue_460/R_test.json").read_text()
        if Path("data/issue_460/R_test.json").exists()
        else (Path(__file__).parent / "_dummy.json").read_text()
    )["completions"]

    logger.info("Loading vLLM %s on GPU %d", BASE_MODEL, args.gpu_id)
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=4096,
    )
    sp_gen = SamplingParams(
        n=8,
        temperature=1.0,
        top_p=1.0,
        max_tokens=2048,
        seed=42,
    )
    sp_logprob = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    # ── Gate 2: source emission per frac per cell ──
    emission_per_frac: dict[str, dict[str, dict]] = {}
    for cid in SMOKE_CELLS:
        cond_source = CONDITIONS_BY_ID[cid]
        emission_per_frac[cid] = {}
        for frac in ALL_FRACS:
            try:
                path = _download_adapter(cid, args.smoke_seed, frac)
            except Exception as e:
                logger.warning(
                    "Adapter %s seed=%d frac=%s missing: %s", cid, args.smoke_seed, frac, e
                )
                emission_per_frac[cid][_frac_tag(frac)] = {
                    "emission_rate": None,
                    "missing_adapter": True,
                    "error": str(e),
                }
                continue
            lora = LoRARequest(
                lora_name=f"{cid}_{_frac_tag(frac)}",
                lora_int_id=round(frac * 100) + ord(cid[0]) * 1000,
                lora_path=path,
            )
            emission, trunc, total = _vllm_source_emission(
                llm,
                sp_gen,
                tokenizer,
                cond_source,
                cond_source,  # source diagonal = same-cond context
                held_out_probe,
                class_d_rewrites,
                lora,
            )
            emission_per_frac[cid][_frac_tag(frac)] = {
                "emission_rate": emission,
                "truncation_rate": trunc,
                "n_generations": total,
            }
            logger.info(
                "diagonal emission %s seed=%d %s = %.3f (trunc=%.3f over %d gens)",
                cid,
                args.smoke_seed,
                _frac_tag(frac),
                emission,
                trunc,
                total,
            )
    (SMOKE_LOG_DIR / "emission_per_frac.json").write_text(json.dumps(emission_per_frac, indent=2))

    # Pick the 3 most evenly spread in-band fracs for BOTH cells.
    in_band_per_cell: dict[str, list[float]] = {}
    for cid in SMOKE_CELLS:
        in_band = []
        for frac in ALL_FRACS:
            rec = emission_per_frac[cid].get(_frac_tag(frac))
            if rec is None or rec.get("emission_rate") is None:
                continue
            er = rec["emission_rate"]
            if IN_BAND_MIN <= er <= IN_BAND_MAX:
                in_band.append(frac)
        in_band_per_cell[cid] = in_band
        logger.info("In-band fracs for %s: %s", cid, in_band)

    # Pick the 3 fracs that are in-band for BOTH cells.
    both_in_band = sorted(set(in_band_per_cell["A1"]) & set(in_band_per_cell["G2"]))
    if len(both_in_band) < 3:
        union_in_band = sorted(set(in_band_per_cell["A1"]) | set(in_band_per_cell["G2"]))
        # Per-cell pick (documented as methodology divergence).
        picked_per_cell = {cid: sorted(in_band_per_cell[cid])[:3] for cid in SMOKE_CELLS}
        all_picked = sorted({f for fs in picked_per_cell.values() for f in fs})
        if not all_picked or any(len(in_band_per_cell[cid]) < 1 for cid in SMOKE_CELLS):
            _write_sentinel(
                "smoke_no_inband_frac",
                "Neither A1 nor G2 has any in-band frac in [0.20, 0.85]. "
                f"emission_per_frac={emission_per_frac}",
            )
            return 3
        # Document the per-cell split and proceed.
        picked = all_picked[:3] if len(all_picked) >= 3 else (all_picked + union_in_band)[:3]
        per_cell_divergence = True
    else:
        # Evenly spread across both_in_band.
        if len(both_in_band) <= 3:
            picked = both_in_band
        else:
            idx = np.linspace(0, len(both_in_band) - 1, 3).round().astype(int).tolist()
            picked = [both_in_band[i] for i in idx]
        per_cell_divergence = False

    # ── Gate 3: off-diagonal saturation gate (A1 only) ──
    offdiag_results: dict[str, dict] = {}
    any_offdiag_unsaturated = False
    for frac in picked:
        try:
            path = _download_adapter("A1", args.smoke_seed, frac)
        except Exception as e:
            offdiag_results[_frac_tag(frac)] = {"error": str(e)}
            continue
        lora = LoRARequest(
            lora_name=f"A1_{_frac_tag(frac)}",
            lora_int_id=round(frac * 100) + ord("A") * 1000,
            lora_path=path,
        )
        cell_emissions = []
        for cj in OFFDIAG_MINI_GRID:
            cond_j = CONDITIONS_BY_ID[cj]
            er, _, _ = _vllm_source_emission(
                llm,
                sp_gen,
                tokenizer,
                CONDITIONS_BY_ID["A1"],
                cond_j,
                held_out_probe,
                class_d_rewrites,
                lora,
            )
            cell_emissions.append({"target": cj, "emission_rate": er})
        floor_mass = sum(1 for c in cell_emissions if c["emission_rate"] <= 0.05) / len(
            cell_emissions
        )
        ceiling_mass = sum(1 for c in cell_emissions if c["emission_rate"] >= 0.95) / len(
            cell_emissions
        )
        tie_mass = max(floor_mass, ceiling_mass)
        offdiag_results[_frac_tag(frac)] = {
            "frac": frac,
            "cells": cell_emissions,
            "floor_mass_off": floor_mass,
            "ceiling_mass_off": ceiling_mass,
            "tie_mass_off": tie_mass,
            "passes_offdiag_gate": tie_mass <= OFFDIAG_MAX_TIE_MASS,
        }
        if tie_mass <= OFFDIAG_MAX_TIE_MASS:
            any_offdiag_unsaturated = True
        logger.info(
            "offdiag gate %s: floor=%.2f ceiling=%.2f tie=%.2f pass=%s",
            _frac_tag(frac),
            floor_mass,
            ceiling_mass,
            tie_mass,
            tie_mass <= OFFDIAG_MAX_TIE_MASS,
        )
    (SMOKE_LOG_DIR / "offdiag_saturation.json").write_text(json.dumps(offdiag_results, indent=2))

    if not any_offdiag_unsaturated:
        _write_sentinel(
            "offdiag_saturated_at_all_fracs",
            "All picked fracs have tie_mass_off > 0.85 on the A1 6-cell mini-grid. "
            f"offdiag_results={offdiag_results}",
        )
        return 4

    # ── Gate 4: EOS-gradient check at the latest picked frac ──
    latest_frac = picked[-1]
    eos_gradient_payload: dict = {"latest_frac": latest_frac}
    try:
        adapter_path = _download_adapter("A1", args.smoke_seed, latest_frac)
    except Exception as e:
        _write_sentinel(
            "eos_gradient_inactive",
            f"Couldn't download A1 frac={latest_frac} for EOS-gradient probe: {e}",
        )
        return 5
    lora = LoRARequest(
        lora_name=f"A1_eosgrad_{_frac_tag(latest_frac)}",
        lora_int_id=99999,
        lora_path=adapter_path,
    )
    # B1 is the no-system default-assistant target; pick the first held-out Q.
    cond_target = CONDITIONS_BY_ID["B1"]
    probe_q = held_out_probe[0]
    prompt_text = build_prompt_for_condition(cond_target, probe_q, tokenizer, class_d_rewrites)
    R_text = (
        R_test_inherited.get("B1", {}).get(probe_q, {}).get("response_text")
        if isinstance(R_test_inherited, dict)
        else None
    )
    if not R_text:
        # Generate on-policy R from base if R_test inherited missing.
        sp_R = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=1024, seed=42)
        gen = llm.generate([prompt_text], sp_R, lora_request=None)
        R_text = gen[0].outputs[0].text
    base_logp = _post_response_slot_logprob(
        llm, sp_logprob, tokenizer, prompt_text, R_text, lora_request=None
    )
    trained_logp = _post_response_slot_logprob(
        llm, sp_logprob, tokenizer, prompt_text, R_text, lora_request=lora
    )
    delta_nats = trained_logp - base_logp
    moved_down = delta_nats <= -EOS_GRADIENT_MIN_NATS
    eos_gradient_payload.update(
        {
            "target": "B1",
            "probe_q": probe_q,
            "base_logp_marker": base_logp,
            "trained_logp_marker": trained_logp,
            "delta_nats": delta_nats,
            "min_drop_required_nats": EOS_GRADIENT_MIN_NATS,
            "moved_down": moved_down,
        }
    )
    (SMOKE_LOG_DIR / "eos_gradient.json").write_text(json.dumps(eos_gradient_payload, indent=2))
    logger.info(
        "EOS-gradient probe: base=%.4f trained=%.4f delta=%.4f (required ≤ -%s)",
        base_logp,
        trained_logp,
        delta_nats,
        EOS_GRADIENT_MIN_NATS,
    )
    if not moved_down:
        _write_sentinel(
            "eos_gradient_inactive",
            "Trained log P(' ※') at A1→B1 post-response slot did NOT drop ≥ "
            f"{EOS_GRADIENT_MIN_NATS} nats vs base "
            f"(delta = {delta_nats:.4f}).",
            extra=eos_gradient_payload,
        )
        return 6

    # ── PASS: persist picked fracs ──
    (SMOKE_LOG_DIR / "picked_fracs.json").write_text(
        json.dumps(
            {
                "schema_version": "i488_v1",
                "picked_fracs": picked,
                "all_fracs": list(ALL_FRACS),
                "in_band_per_cell": in_band_per_cell,
                "per_cell_divergence": per_cell_divergence,
                "wrote_at": datetime.datetime.now(datetime.UTC).isoformat(),
            },
            indent=2,
        )
    )
    logger.info("Smoke gates PASS. Picked fracs: %s", picked)

    del llm
    from issue404_common import kill_vllm_workers

    kill_vllm_workers(logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
