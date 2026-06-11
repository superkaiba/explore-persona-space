#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 — post-hoc dense-checkpoint teacher-forced four-float reader.

Reads EVERY checkpoint in a cell's checkpoint_index (dense per-step ladder +
trajectory fractions) and records, per (persona, q), the four-float slot
contract — log P(marker), z_marker, z_eos (<|im_end|> 151645), logZ — for the
trained AND base sides over the FROZEN base-model R_eval (the sanctioned
within-condition dynamics read; on-policy anchoring at >=2 checkpoints/cell is
done by the trajectory eval and enforced at analysis admission — plan §6).

Personas read per checkpoint: source (villain) + the cell's trained negatives
(empty for 0-negative cells) + the Phase-0 pre-registered 8-bystander
reference panel (plan §4 Phases 2/3 contrast re-registration).

Efficiency: ONE base model load; the base-side reads are checkpoint-
independent and computed once; per-checkpoint adapters are hot-swapped via
PEFT load_adapter/set_adapter/delete_adapter (never a per-checkpoint 7B
reload). HF-only — no vLLM in this process (logits are unrecoverable from
vLLM log-probs; incident #530). Output is rewritten after EVERY checkpoint
(checkpoint-per-phase) and re-runs skip already-read checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i601.dense_read")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _slot_raw_logits(model, tokenizer, persona_prompt: str, q: str, r_text: str, sep: str, device):
    """Raw next-token logits (V,) at the post-R slot (same render as the parent rig)."""
    import torch

    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": q},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix = prompt_text + r_text + sep
    ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert logits.ndim == 3, logits.shape
    return logits[0, -1, :].float().cpu()


def _four_floats(raw, marker_id: int, eos_id: int) -> dict[str, float]:
    import torch

    log_z = float(torch.logsumexp(raw, dim=-1).item())
    z_m = float(raw[marker_id].item())
    return {
        "logp": z_m - log_z,
        "z_marker": z_m,
        "z_eos": float(raw[eos_id].item()),
        "logZ": log_z,
    }


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- linear read loop with resume branches; one read path, no nesting to extract
    ap = argparse.ArgumentParser(
        description="Task #601 dense teacher-forced four-float reader (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint-index", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_601"))
    ap.add_argument(
        "--bystander-panel-path",
        type=Path,
        default=Path("eval_results/issue_601/phase0/bystander_panel.json"),
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=dense_read] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import HEADLINE_LAYER
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_logit_readout_gauge_free,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        negatives_for_cell,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import (
        BASE_MODEL,
        CELL_SPECS_601_472SHAPE,
        EXPECTED_MARKER_TOKEN_ID,
        EXPECTED_POST_R_EOS_ID,
        MARKER_SEP,
        MARKER_TEXT,
        SOURCE_PERSONA,
        cell_by_slug,
    )

    spec = cell_by_slug(args.cell)
    bank = load_persona_bank(args.data_dir / "persona_bank.json")
    r_eval = load_r_artifact(args.data_dir / "on_policy_R" / "R_eval.json")
    _q_train, q_eval = get_train_eval_questions()

    if not args.bystander_panel_path.exists():
        raise FileNotFoundError(
            f"bystander panel missing at {args.bystander_panel_path}; run Phase 0 first."
        )
    bystanders = json.loads(args.bystander_panel_path.read_text())["personas"]
    trained_negs: list[str] = []
    if spec.n_neg_personas > 0:
        cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.data_dir)
        trained_negs = negatives_for_cell(
            args.cell, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_601_472SHAPE
        )
    personas = [SOURCE_PERSONA, *trained_negs, *[b for b in bystanders if b not in trained_negs]]
    for p in personas:
        if p not in bank or p not in r_eval:
            raise KeyError(f"persona {p!r} missing from bank/R_eval.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert marker_ids == [EXPECTED_MARKER_TOKEN_ID], marker_ids
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert eos_id == EXPECTED_POST_R_EOS_ID, eos_id
    marker_id = marker_ids[0]

    ckpt_index = json.loads(args.checkpoint_index.read_text())
    specs = [
        {"frac": float(k), "step": v.get("step"), "adapter_path": v["path"]}
        for k, v in ckpt_index.items()
        if v.get("path") is not None
    ]
    specs.sort(key=lambda s: (s["step"] is None, s["step"], s["frac"]))
    if not specs:
        raise RuntimeError(f"no usable checkpoints in {args.checkpoint_index}")

    # Resume: keep already-read checkpoints (idempotent mop-up re-runs).
    existing: dict[str, dict] = {}
    if args.out_path.exists():
        prior = json.loads(args.out_path.read_text())
        existing = {f"{c['frac']:.4f}": c for c in prior.get("checkpoints", [])}
        log.info("resume: %d checkpoints already read", len(existing))

    device = args.device if torch.cuda.is_available() else "cpu"
    log.info(
        "loading base model on %s (%d checkpoints, %d personas)", device, len(specs), len(personas)
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).eval()

    # Base-side reads are checkpoint-independent — compute once.
    base_reads: dict[str, dict[str, dict]] = {}
    for p in personas:
        base_reads[p] = {}
        for q in q_eval:
            r_text = r_eval[p][q]["response_text"]
            raw = _slot_raw_logits(base, tokenizer, bank[p], q, r_text, MARKER_SEP, device)
            base_reads[p][q] = _four_floats(raw, marker_id, eos_id)
    log.info("base-side reads cached (%d personas x %d questions)", len(personas), len(q_eval))

    peft_model: PeftModel | None = None
    prev_name: str | None = None
    checkpoints_out: list[dict] = [existing[k] for k in sorted(existing)]
    done_keys = set(existing)

    def _persist() -> None:
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "i601_dense_v1",
            "cell": args.cell,
            "seed": args.seed,
            "source": SOURCE_PERSONA,
            "trained_negatives": trained_negs,
            "bystander_panel": bystanders,
            "eval_questions": q_eval,
            "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
            "post_r_eos_token_id": EXPECTED_POST_R_EOS_ID,
            "read_type": "teacher_forced_frozen_R_eval",
            "checkpoints": sorted(checkpoints_out, key=lambda c: c["frac"]),
            "git_commit": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        tmp = args.out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, args.out_path)

    for i, ck in enumerate(specs):
        key = f"{ck['frac']:.4f}"
        if key in done_keys:
            continue
        adapter_path = ck["adapter_path"]
        assert_logit_readout_gauge_free(adapter_path)
        name = f"ck{i}"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, adapter_path, adapter_name=name)
            peft_model.eval()
        else:
            peft_model.load_adapter(adapter_path, adapter_name=name)
            peft_model.set_adapter(name)
            if prev_name is not None:
                peft_model.delete_adapter(prev_name)
        prev_name = name

        reads: dict[str, dict[str, dict]] = {}
        for p in personas:
            reads[p] = {}
            for q in q_eval:
                r_text = r_eval[p][q]["response_text"]
                raw = _slot_raw_logits(
                    peft_model, tokenizer, bank[p], q, r_text, MARKER_SEP, device
                )
                g = _four_floats(raw, marker_id, eos_id)
                b = base_reads[p][q]
                reads[p][q] = {
                    "logp_g": g["logp"],
                    "logp_b": b["logp"],
                    "delta_g": g["logp"] - b["logp"],
                    "z_marker_g": g["z_marker"],
                    "z_marker_b": b["z_marker"],
                    "z_eos_g": g["z_eos"],
                    "z_eos_b": b["z_eos"],
                    "logZ_g": g["logZ"],
                    "logZ_b": b["logZ"],
                    "delta_z_marker": g["z_marker"] - b["z_marker"],
                    "delta_margin": (g["z_marker"] - g["z_eos"]) - (b["z_marker"] - b["z_eos"]),
                }
        n_q = len(q_eval)
        src = reads[SOURCE_PERSONA]
        source_mean = {
            k: sum(src[q][k] for q in q_eval) / n_q
            for k in (
                "delta_g",
                "delta_z_marker",
                "delta_margin",
                "logp_g",
                "logp_b",
                "z_eos_g",
                "z_eos_b",
                "logZ_g",
                "logZ_b",
            )
        }
        checkpoints_out.append(
            {
                "frac": ck["frac"],
                "step": ck["step"],
                "adapter_path": adapter_path,
                "source_mean": source_mean,
                "reads": reads,
            }
        )
        done_keys.add(key)
        _persist()  # checkpoint-per-phase: rewrite after EVERY checkpoint.
        log.info(
            "[dense %s/%s] step=%s source ΔG=%.2f Δz=%.2f Δmargin=%.2f",
            args.cell,
            key,
            ck["step"],
            source_mean["delta_g"],
            source_mean["delta_z_marker"],
            source_mean["delta_margin"],
        )

    _persist()
    log.info("dense read complete → %s (%d checkpoints)", args.out_path, len(checkpoints_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
