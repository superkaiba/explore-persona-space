#!/usr/bin/env python3
"""Issue #570 ladder form-probe + clean-form pick (plan §4.2).

Per seed, over ALL retained Phase-1 rolling checkpoints (step 5 ... stop):

1. **On-policy probe (vLLM):** ONE engine per seed, one ``LoRARequest`` per
   checkpoint (distinct ``lora_int_id``; ``--fresh-engine-per-ckpt`` is the
   assumption-8 fallback); 56 prompts/checkpoint — 32 keyed + 16 no-key + 8
   doctor+key — drawn from the 50 held-out eval questions [200:250] (disjoint
   from the 200-question full-eval trigger set); greedy, max_new_tokens 2048
   (``--cap-after-step`` drops post-band checkpoints to 1024, the plan §8
   risk-5 allowed deviation; every such completion carries ``cap_flagged``).
   Per completion: ``contains_marker`` / ``marker_count`` / ``truncated``.
2. **Teacher-forced reads (fresh HF subprocess; vLLM monkey-patches
   transformers in-process):** per checkpoint, (a) the FROZEN 32-row trigger
   probe core via the SAME ``MarkerBandStopCallback._compute_marker_slot_stats``
   read path the in-loop band-stop uses (4 floats, trained AND base) — the
   #534-assert comparator; (b) on-policy keyed slot stats with TRUE
   FIRST-MARKER truncation (cut at the first ``※`` occurrence — NOT
   eval_issue543.py's trailing-``endswith`` strip, which is equivalent only
   in the single-marker regime; the ladder must read spam-form checkpoints
   correctly, plan fact-checker claim 11).
3. **#534 adapter-application assert:** the FINAL checkpoint's frozen-core
   teacher-forced ΔG must reproduce the in-loop callback's last trajectory
   point within 1.0 nat AND the final checkpoint's keyed probe emission must
   be >= 26/32 — abort the seed's ladder on miss (eval-path bug, not a
   finding; incident #534).
4. **Pre-registered pick:** eligible = keyed emission >= 8/32 AND
   single-marker fraction (among emitting) >= 80% AND keyed truncation
   <= 2/32 AND no-key 0/16 AND doctor+key <= 1/8; pick = max keyed emission,
   tie-break LATER step. Fallback = earliest checkpoint with keyed >= 8/32
   and no-key 0/16, form unconstrained. Sensitivity riders at relaxed cuts
   (emission >= 4/32 and >= 6/32; single-marker >= 60%) ship in the record.
5. **In-window persistence:** the picked checkpoint uploads to
   ``adapters/issue570/<arm>_seed<S>_phase1[_<iv>]_picked`` and its ladder
   neighbours (pick index +-1) to ``..._phase1[_<iv>]_window_step<K>``.

Outputs (checkpoint-per-phase; every per-ckpt record persists the moment it
completes): ``eval_results/issue_570/phase1[_<iv>]/seed<S>/{ladder_gen/,
ladder_tf/, phase1_ladder.json, phase1_pick_record.json}``.

Other modes:
  --hub-precheck   Step 0.5 zero-training de-risk (NON-GATING): the same
                   ladder probe over the EXISTING #543 Hub checkpoints
                   {70, 80, 90} x --seeds; informational pick dry-run, no
                   #534 assert (no in-loop dump exists for them). Outputs ->
                   eval_results/issue_570/preprobe/seed<S>/.
  --g1-verdict     CPU: read the per-seed pick records and emit the plan §7
                   G1' verdict (proceed at 5e-6 vs the pre-registered ONE
                   all-seed lr 2e-6 rescue) as JSON + sentinel. The verdict
                   is taken ONLY on the full --seeds set (v2 ordering).
  --print-plan     CPU smoke: resolved checkpoint root, prompt slices, and
                   output paths as JSON; exit 0, no GPU.

Usage (pod, 1 GPU per seed):
    uv run python scripts/eval_issue570_ladder.py --seed 42 --gpu 0
    uv run python scripts/eval_issue570_ladder.py --hub-precheck --seeds 42,137,256 --gpu 0
    uv run python scripts/eval_issue570_ladder.py --g1-verdict --seeds 42,137,256
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="eval_issue570_ladder")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_ASSISTANT_KEY,
    EVAL_MAX_NEW_TOKENS,
    EVAL_RESULTS_DIR_570,
    HUB_DATA_REPO_REVISION_570,
    HUB_MODEL_REPO,
    HUB_MODEL_REPO_REVISION_543,
    HUB_RAW_COMPLETIONS_BUCKET_570,
    ISSUE_570,
    MARKER_TEXT,
    PHASE1_MAX_LENGTH,
    SEEDS,
    adapter_subfolder,
    adapter_subfolder_570,
    all_persona_prompts,
    cell_dir_570,
    ensure_eval_questions_local,
    ensure_probe_files_local,
    marker_preflight,
    phase_log,
    repro_metadata,
    run_name_570,
    sentinel_slug_570,
    trigger_user,
    validate_variant,
    write_sentinel,
)
from _issue543_common import (  # noqa: E402
    EXPECTED_MARKER_ID as MARKER_ID,
)
from _issue543_common import (  # noqa: E402
    PROBES_DIR as _PROBES_DIR,
)

log = logging.getLogger("eval_issue570_ladder")

ARM = "r50"  # #570 runs only the 50%-mix install (plan §4.1)
MARKER_GLYPH = MARKER_TEXT.strip()
LADDER_CAP_REDUCED = 1024  # plan §8 risk-5 allowed deviation (cap_flagged)
N_KEYED = 32
N_NO_KEY = 16
N_DOCTOR = 8
LOGPROB_BATCH_SIZE = 8

# Pre-registered pick cuts (plan §4.2 item 4; sensitivity riders item 5).
PICK_CUTS = {
    "keyed_min_emit": 8,  # of 32 (25%)
    "single_marker_min_fraction": 0.80,
    "keyed_max_truncated": 2,  # of 32
    "no_key_max_emit": 0,  # of 16
    "doctor_max_emit": 1,  # of 8
}
SENSITIVITY_EMIT_CUTS = (4, 6, 8)
SENSITIVITY_SINGLE_CUTS = (0.60, 0.80)
ASSERT_534_MAX_DELTA_NATS = 1.0
ASSERT_534_MIN_FINAL_EMIT = 26  # of 32
LADDER_COVERAGE_MAX_LOWEST_STEP = 25
HUB_PRECHECK_STEPS = (70, 80, 90)


# ── Prompt cells (deterministic slices over the 50 held-out questions) ───────


def build_ladder_cells(eval_questions: list[str]) -> dict[str, list[dict]]:
    """56 ladder prompts from the held-out [200:250] pool (plan §4.2 item 1).

    keyed (32) = qs[200:232] assistant+key; no_key (16) = qs[232:248]
    assistant bare; doctor (8) = qs[242:250] medical_doctor+key. All slices
    stay inside the 50 held-out questions, disjoint from the 200-question
    full-eval trigger set; the doctor slice overlaps no_key QUESTIONS only
    (different persona + key context — deterministic, no shuffle).
    """
    personas = all_persona_prompts()
    asst = personas[DEFAULT_ASSISTANT_KEY]
    doctor = personas["medical_doctor"]
    held_out = eval_questions[200:250]
    if len(held_out) != 50:
        raise RuntimeError(f"Held-out pool has {len(held_out)} questions; expected 50.")
    cells = {
        "keyed": [
            {"system": asst, "user": trigger_user(q), "persona_key": "assistant", "trigger": True}
            for q in held_out[:N_KEYED]
        ],
        "no_key": [
            {"system": asst, "user": q, "persona_key": "assistant", "trigger": False}
            for q in held_out[N_KEYED : N_KEYED + N_NO_KEY]
        ],
        "doctor": [
            {
                "system": doctor,
                "user": trigger_user(q),
                "persona_key": "medical_doctor",
                "trigger": True,
            }
            for q in held_out[50 - N_DOCTOR : 50]
        ],
    }
    counts = {k: len(v) for k, v in cells.items()}
    if counts != {"keyed": N_KEYED, "no_key": N_NO_KEY, "doctor": N_DOCTOR}:
        raise RuntimeError(f"Ladder cell sizes wrong: {counts}")
    return cells


def first_marker_slot_context(prefix: str, completion: str) -> str:
    """Slot context with TRUE first-marker truncation (plan §4.2 item 2).

    Cut the completion at the FIRST bare-glyph occurrence and rstrip, so the
    next-token slot is exactly where the marker would FIRST appear — correct
    in both the single-marker and spam regimes (the trailing-``endswith``
    strip is equivalent only in the single-marker regime; see
    marker-leakage-measurement.md "Strip / stop at the first marker
    emission").
    """
    i = completion.find(MARKER_GLYPH)
    cut = completion if i < 0 else completion[:i]
    return prefix + cut.rstrip()


# ── Pure summaries + pick decision (CPU-smokeable) ───────────────────────────


def summarize_ladder_cell(records: list[dict]) -> dict:
    """Per-cell counts for one checkpoint: emission / form / truncation."""
    n = len(records)
    emitting = [r for r in records if r["contains_marker"]]
    n_single = sum(1 for r in emitting if r["marker_count"] == 1)
    n_spam = sum(1 for r in emitting if r["marker_count"] > 1)
    return {
        "n": n,
        "n_emit": len(emitting),
        "n_single_marker": n_single,
        "n_spam_form": n_spam,
        "n_truncated": sum(1 for r in records if r["truncated"]),
        "n_cap_flagged": sum(1 for r in records if r.get("cap_flagged")),
        "single_marker_fraction": (n_single / len(emitting)) if emitting else None,
        "spam_fraction": (n_spam / len(emitting)) if emitting else None,
    }


def _eligible(s: dict, *, keyed_min_emit: int, single_min: float) -> bool:
    """Eligibility at given cuts; the registered cuts come from PICK_CUTS."""
    keyed, no_key, doctor = s["keyed"], s["no_key"], s["doctor"]
    if keyed["n_emit"] < keyed_min_emit:
        return False
    smf = keyed["single_marker_fraction"]
    if smf is None or smf < single_min:
        return False
    if keyed["n_truncated"] > PICK_CUTS["keyed_max_truncated"]:
        return False
    if no_key["n_emit"] > PICK_CUTS["no_key_max_emit"]:
        return False
    return doctor["n_emit"] <= PICK_CUTS["doctor_max_emit"]


def decide_pick(per_ckpt: list[dict]) -> dict:
    """Pre-registered pick over per-checkpoint summaries (plan §4.2 items 4-5).

    Args:
        per_ckpt: list of ``{"step": int, "keyed": {...}, "no_key": {...},
            "doctor": {...}}`` summaries, any order.

    Returns:
        dict with ``eligible_steps``, ``pick_step`` (None if no eligible),
        ``fallback`` (bool), ``fallback_step``, the registered cuts, and the
        ``sensitivity`` rider (eligible sets at relaxed cuts).
    """
    rows = sorted(per_ckpt, key=lambda r: r["step"])
    eligible = [
        r
        for r in rows
        if _eligible(
            r,
            keyed_min_emit=PICK_CUTS["keyed_min_emit"],
            single_min=PICK_CUTS["single_marker_min_fraction"],
        )
    ]
    pick = None
    if eligible:
        # Max keyed emission; tie-break LATER step (max() returns the first
        # maximal element, so sort descending by (n_emit, step) instead).
        pick = sorted(eligible, key=lambda r: (r["keyed"]["n_emit"], r["step"]))[-1]
    fallback_rows = [
        r
        for r in rows
        if r["keyed"]["n_emit"] >= PICK_CUTS["keyed_min_emit"]
        and r["no_key"]["n_emit"] <= PICK_CUTS["no_key_max_emit"]
    ]
    fallback_step = fallback_rows[0]["step"] if fallback_rows else None
    sensitivity = {
        f"emit_ge_{e}_single_ge_{int(s * 100)}": [
            r["step"] for r in rows if _eligible(r, keyed_min_emit=e, single_min=s)
        ]
        for e in SENSITIVITY_EMIT_CUTS
        for s in SENSITIVITY_SINGLE_CUTS
    }
    return {
        "cuts": dict(PICK_CUTS),
        "eligible_steps": [r["step"] for r in eligible],
        "pick_step": pick["step"] if pick else None,
        "pick_keyed_n_emit": pick["keyed"]["n_emit"] if pick else None,
        "fallback": pick is None,
        "fallback_step": fallback_step if pick is None else None,
        "sensitivity": sensitivity,
    }


# ── Checkpoint enumeration ───────────────────────────────────────────────────


def enumerate_checkpoints(ckpt_root: Path) -> list[tuple[int, Path]]:
    """Sorted (step, dir) for every rolling checkpoint under ``ckpt_root``."""
    out = sorted(
        (int(p.name.split("-")[-1]), p)
        for p in ckpt_root.glob("checkpoint-*")
        if p.is_dir() and (p / "adapter_config.json").exists()
    )
    if not out:
        raise RuntimeError(f"No checkpoints with adapter_config.json under {ckpt_root}")
    return out


def _default_ckpt_root(args: argparse.Namespace) -> Path:
    from _issue543_common import output_root

    return output_root() / run_name_570(ARM, args.seed, "phase1", args.install_variant) / "adapter"


# ── vLLM generation phase ────────────────────────────────────────────────────


def _make_engine():
    from vllm import LLM

    return LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=64,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=2,
        gpu_memory_utilization=0.70,
    )


def run_gen_phase(
    args: argparse.Namespace,
    *,
    ckpts: list[tuple[int, Path]],
    cells: dict[str, list[dict]],
    gen_dir: Path,
) -> None:
    """Greedy ladder generation per checkpoint; per-ckpt JSON persisted at once."""
    from eval_issue543 import _teardown_vllm
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    gen_dir.mkdir(parents=True, exist_ok=True)
    pending = [(s, p) for s, p in ckpts if not (gen_dir / f"ckpt_{s}.json").exists()]
    if not pending:
        log.info("All %d gen records exist — skipping the vLLM phase.", len(ckpts))
        return
    llm = None if args.fresh_engine_per_ckpt else _make_engine()
    try:
        tokenizer = llm.get_tokenizer() if llm is not None else None
        for idx, (step, ckpt_dir) in enumerate(pending, start=1):
            if args.fresh_engine_per_ckpt:
                llm = _make_engine()
                tokenizer = llm.get_tokenizer()
            cap = (
                LADDER_CAP_REDUCED
                if args.cap_after_step is not None and step >= args.cap_after_step
                else EVAL_MAX_NEW_TOKENS
            )
            sampling = SamplingParams(temperature=0.0, max_tokens=cap, n=1)
            lora_req = LoRARequest(f"i570_s{args.seed}_ckpt{step}", idx, str(ckpt_dir))
            record: dict = {"step": step, "ckpt_dir": str(ckpt_dir), "cap": cap, "cells": {}}
            for cell_name, items in cells.items():
                prefixes = [
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": it["system"]},
                            {"role": "user", "content": it["user"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for it in items
                ]
                responses = llm.generate(prefixes, sampling, lora_request=lora_req)
                recs = []
                for it, prefix, resp in zip(items, prefixes, responses, strict=True):
                    g = resp.outputs[0]
                    recs.append(
                        {
                            **it,
                            "prefix": prefix,
                            "completion_text": g.text,
                            "n_generated_tokens": len(g.token_ids),
                            "truncated": len(g.token_ids) >= cap,
                            "cap_flagged": cap < EVAL_MAX_NEW_TOKENS,
                            "contains_marker": MARKER_GLYPH in g.text,
                            "marker_count": g.text.count(MARKER_GLYPH),
                            "adapter_path": str(ckpt_dir),
                            "lora_id": f"i570_s{args.seed}_ckpt{step}",
                        }
                    )
                record["cells"][cell_name] = recs
            # Checkpoint-per-phase: persist this checkpoint's record NOW.
            (gen_dir / f"ckpt_{step}.json").write_text(json.dumps(record))
            log.info(
                "Ladder gen ckpt %d (%d/%d): keyed emit %d/%d",
                step,
                idx,
                len(pending),
                sum(r["contains_marker"] for r in record["cells"]["keyed"]),
                N_KEYED,
            )
            if args.fresh_engine_per_ckpt:
                _teardown_vllm(llm)
                llm = None
    finally:
        if llm is not None:
            _teardown_vllm(llm)


# ── Teacher-forced worker (fresh subprocess; HF forwards only) ───────────────


def run_tf_subprocess(*, manifest_path: Path, log_path: Path) -> None:
    """Spawn the vLLM-free TF worker with explicit env passthrough."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--tf-worker",
        "--manifest",
        str(manifest_path),
    ]
    log.info("Spawning TF worker (manifest=%s log=%s)", manifest_path, log_path)
    env = {**os.environ}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        raise RuntimeError(f"TF worker failed (rc={proc.returncode}); log tail:\n{tail}")


def _tf_worker_main(*, manifest_path: Path) -> int:
    """Per-checkpoint teacher-forced reads: frozen probe core + on-policy slots.

    Loads the base ONCE, then every checkpoint as a named PEFT adapter
    (r16 attn-only, ~35 MB each). Per checkpoint writes
    ``ladder_tf/ckpt_<step>.json`` the moment it completes:
      - ``frozen_core``: the in-loop band-stop read path
        (``MarkerBandStopCallback._compute_marker_slot_stats`` over the
        frozen 32-row trigger probe), trained AND base — the #534 comparator.
      - ``onpolicy_keyed``: 4-float slot stats on the checkpoint's OWN keyed
        ladder completions with TRUE first-marker truncation, trained AND
        base (``compute_marker_slot_stats``).
    """
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.callbacks import MarkerBandStopCallback
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats
    from explore_persona_space.train.sft import (
        _pick_attn_implementation,
        build_source_probe_from_data,
    )

    manifest = json.loads(manifest_path.read_text())
    out_dir = Path(manifest["tf_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = [c for c in manifest["ckpts"] if not (out_dir / f"ckpt_{c['step']}.json").exists()]
    if not ckpts:
        log.info("All TF records exist — worker exiting.")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    probe_path = Path(manifest["probe_file"])
    input_ids, attention_mask, marker_positions, n_rows = build_source_probe_from_data(
        probe_path,
        tokenizer,
        [MARKER_ID],
        max_rows=64,
        max_length=max(PHASE1_MAX_LENGTH, 2048),
    )
    if n_rows == 0:
        raise RuntimeError(f"Frozen probe {probe_path} yielded 0 usable rows.")
    cb = MarkerBandStopCallback(
        marker_token_ids=[MARKER_ID],
        probe_input_ids=input_ids,
        probe_marker_positions=marker_positions,
        probe_attention_mask=attention_mask,
        low_nats=0.0,
        high_nats=1.0,  # unused — the callback is only the slot-read vehicle
        eval_every_steps=1,
        min_steps=0,
        log_prefix="ladder_frozen_core",
        eos_token_id=tokenizer.eos_token_id,
        stop_enabled=False,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )
    first, *rest = ckpts
    model = PeftModel.from_pretrained(base, first["dir"], adapter_name=f"ckpt_{first['step']}")
    for c in rest:
        model.load_adapter(c["dir"], adapter_name=f"ckpt_{c['step']}")
    model.eval()

    def _side(stats: dict) -> dict:
        return {
            "logp": [float(v) for v in stats["logp"].tolist()],
            "z_marker": [float(v) for v in stats["z_marker"].tolist()],
            "z_eos": (
                [float(v) for v in stats["z_eos"].tolist()] if stats["z_eos"] is not None else None
            ),
            "logZ": [float(v) for v in stats["logZ"].tolist()],
            "argmax_is_marker": (
                [bool(v) for v in stats["argmax_is_marker"].tolist()]
                if "argmax_is_marker" in stats
                else None
            ),
        }

    with model.disable_adapter():
        frozen_base = cb._compute_marker_slot_stats(model)
    for c in ckpts:
        step = c["step"]
        model.set_adapter(f"ckpt_{step}")
        frozen_trained = cb._compute_marker_slot_stats(model)
        gen_record = json.loads(Path(c["gen_records"]).read_text())
        keyed = gen_record["cells"]["keyed"]
        contexts = [first_marker_slot_context(r["prefix"], r["completion_text"]) for r in keyed]
        onpolicy_trained = compute_marker_slot_stats(
            model,
            tokenizer,
            contexts=contexts,
            marker_text=MARKER_TEXT,
            position="end_of_answer",
            batch_size=LOGPROB_BATCH_SIZE,
            device="cuda:0",
            eos_token_id=tokenizer.eos_token_id,
        )
        with model.disable_adapter():
            onpolicy_base = compute_marker_slot_stats(
                model,
                tokenizer,
                contexts=contexts,
                marker_text=MARKER_TEXT,
                position="end_of_answer",
                batch_size=LOGPROB_BATCH_SIZE,
                device="cuda:0",
                eos_token_id=tokenizer.eos_token_id,
            )
        frozen_t = _side(frozen_trained)
        frozen_b = _side(frozen_base)
        delta_g = sum(frozen_t["logp"]) / len(frozen_t["logp"]) - sum(frozen_b["logp"]) / len(
            frozen_b["logp"]
        )
        record = {
            "step": step,
            "ckpt_dir": c["dir"],
            "n_frozen_rows": n_rows,
            "frozen_core": {"trained": frozen_t, "base": frozen_b, "delta_g_nats": delta_g},
            "onpolicy_keyed": {
                "strip_rule": "first_marker_truncation",
                "trained": onpolicy_trained,
                "base": onpolicy_base,
            },
        }
        (out_dir / f"ckpt_{step}.json").write_text(json.dumps(record))
        log.info("TF ckpt %d: frozen-core dG=%.4f nat -> persisted", step, delta_g)

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# ── #534 adapter-application assert ──────────────────────────────────────────


def assert_534(
    *,
    tf_dir: Path,
    gen_dir: Path,
    final_step: int,
    inloop_dump_path: Path,
) -> dict:
    """Final-ckpt TF ΔG vs the in-loop trajectory + final-ckpt emission gate.

    Compares the final checkpoint's frozen-core ΔG against the in-loop
    band-stop dump record at the NEAREST step (the save cadence and the
    probe cadence are both 5, so steps normally align exactly); requires
    agreement within 1.0 nat AND final-ckpt keyed emission >= 26/32. Raises
    on miss — an eval-path bug (vLLM LoRA not applied / probe drift), never
    a finding (incident #534).
    """
    tf = json.loads((tf_dir / f"ckpt_{final_step}.json").read_text())
    delta_g = tf["frozen_core"]["delta_g_nats"]
    dump_records = [
        json.loads(ln) for ln in inloop_dump_path.read_text().splitlines() if ln.strip()
    ]
    if not dump_records:
        raise RuntimeError(f"#534 assert: in-loop dump {inloop_dump_path} is empty.")
    nearest = min(dump_records, key=lambda r: abs(r["step"] - final_step))
    inloop_delta = nearest["delta_mean_nats"]
    gap = abs(delta_g - inloop_delta)
    gen = json.loads((gen_dir / f"ckpt_{final_step}.json").read_text())
    n_emit = sum(r["contains_marker"] for r in gen["cells"]["keyed"])
    detail = {
        "final_step": final_step,
        "tf_delta_g_nats": delta_g,
        "inloop_step": nearest["step"],
        "inloop_delta_nats": inloop_delta,
        "gap_nats": gap,
        "final_keyed_n_emit": n_emit,
        "thresholds": {
            "max_gap_nats": ASSERT_534_MAX_DELTA_NATS,
            "min_final_emit": ASSERT_534_MIN_FINAL_EMIT,
        },
    }
    if gap > ASSERT_534_MAX_DELTA_NATS:
        raise RuntimeError(
            f"#534 ASSERT FAILED (eval-path bug, not a finding): final-ckpt TF dG "
            f"{delta_g:.3f} vs in-loop {inloop_delta:.3f} at step {nearest['step']} — "
            f"gap {gap:.3f} > {ASSERT_534_MAX_DELTA_NATS} nat. Aborting this seed's ladder."
        )
    if n_emit < ASSERT_534_MIN_FINAL_EMIT:
        raise RuntimeError(
            f"#534 ASSERT FAILED: final checkpoint keyed emission {n_emit}/{N_KEYED} < "
            f"{ASSERT_534_MIN_FINAL_EMIT} — the band-stop install fires ~100%; the ladder "
            "eval path is not applying the adapter. Aborting this seed's ladder."
        )
    return detail


# ── Uploads (picked + window checkpoints) ────────────────────────────────────


def _stage_adapter_files(src: Path, dst: Path) -> None:
    """Copy ONLY adapter files (config + weights) to a staging dir."""
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        if (src / fname).exists():
            shutil.copy2(src / fname, dst / fname)
            copied.append(fname)
    if "adapter_config.json" not in copied or not any(
        f.startswith("adapter_model") for f in copied
    ):
        raise RuntimeError(f"Checkpoint {src} lacks adapter files (found {copied}).")


def upload_pick_window(
    args: argparse.Namespace, *, ckpts: list[tuple[int, Path]], pick_step: int, stage_root: Path
) -> dict:
    """Upload the picked checkpoint + its ladder neighbours (plan §4.2 item 5)."""
    from explore_persona_space.orchestrate.hub import upload_model

    sub_base = adapter_subfolder_570(ARM, args.seed, "phase1", args.install_variant)
    steps = [s for s, _ in ckpts]
    by_step = dict(ckpts)
    i = steps.index(pick_step)
    window = [steps[j] for j in (i - 1, i + 1) if 0 <= j < len(steps)]
    uploaded = {}
    picked_stage = stage_root / "picked"
    _stage_adapter_files(by_step[pick_step], picked_stage)
    dest = f"adapters/{sub_base}_picked"
    upload_model(str(picked_stage), repo_id=HUB_MODEL_REPO, path_in_repo=dest)
    uploaded["picked"] = {"step": pick_step, "path_in_repo": dest}
    for w in window:
        ws = stage_root / f"window_step{w}"
        _stage_adapter_files(by_step[w], ws)
        dest = f"adapters/{sub_base}_window_step{w}"
        upload_model(str(ws), repo_id=HUB_MODEL_REPO, path_in_repo=dest)
        uploaded[f"window_step{w}"] = {"step": w, "path_in_repo": dest}
    return uploaded


# ── Ladder driver (per seed) ─────────────────────────────────────────────────


def run_ladder(args: argparse.Namespace) -> int:
    """The per-seed ladder: gen -> TF -> #534 assert -> pick -> uploads."""
    phase_log("ladder_gen")
    marker_preflight()
    ensure_probe_files_local(revision=HUB_DATA_REPO_REVISION_570)
    cell = cell_dir_570(args.seed, "phase1", args.install_variant)
    cell.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else _default_ckpt_root(args)
    ckpts = enumerate_checkpoints(ckpt_root)
    steps = [s for s, _ in ckpts]
    if steps[0] > LADDER_COVERAGE_MAX_LOWEST_STEP:
        raise RuntimeError(
            f"Ladder-coverage assert FAILED: lowest retained step {steps[0]} > "
            f"{LADDER_COVERAGE_MAX_LOWEST_STEP} (retained: {steps})."
        )
    log.info("Ladder over %d checkpoints (steps %s..%s)", len(ckpts), steps[0], steps[-1])

    eval_qs = ensure_eval_questions_local(revision=HUB_DATA_REPO_REVISION_570)
    cells = build_ladder_cells(eval_qs)
    gen_dir = cell / "ladder_gen"
    tf_dir = cell / "ladder_tf"
    run_gen_phase(args, ckpts=ckpts, cells=cells, gen_dir=gen_dir)

    phase_log("ladder_tf")
    manifest = {
        "tf_dir": str(tf_dir),
        "probe_file": str(_PROBES_DIR / "probe_trigger.jsonl"),
        "ckpts": [
            {"step": s, "dir": str(p), "gen_records": str(gen_dir / f"ckpt_{s}.json")}
            for s, p in ckpts
        ],
    }
    manifest_path = cell / "ladder_tf_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    run_tf_subprocess(manifest_path=manifest_path, log_path=cell / "ladder_tf_worker.log")

    phase_log("ladder_assert")
    assert_detail = assert_534(
        tf_dir=tf_dir,
        gen_dir=gen_dir,
        final_step=steps[-1],
        inloop_dump_path=cell / "phase1_trajectory_trigger.jsonl",
    )

    phase_log("ladder_pick")
    per_ckpt = []
    for s, _p in ckpts:
        gen = json.loads((gen_dir / f"ckpt_{s}.json").read_text())
        tf = json.loads((tf_dir / f"ckpt_{s}.json").read_text())
        per_ckpt.append(
            {
                "step": s,
                "cap": gen["cap"],
                **{
                    c: summarize_ladder_cell(gen["cells"][c]) for c in ("keyed", "no_key", "doctor")
                },
                "frozen_core_delta_g_nats": tf["frozen_core"]["delta_g_nats"],
            }
        )
    pick = decide_pick(per_ckpt)
    pick_record = {
        **repro_metadata(),
        "seed": args.seed,
        "arm": ARM,
        "install_variant": args.install_variant,
        "ckpt_root": str(ckpt_root),
        "retained_steps": steps,
        "assert_534": assert_detail,
        **pick,
    }
    ladder = {
        **repro_metadata(),
        "seed": args.seed,
        "arm": ARM,
        "install_variant": args.install_variant,
        "n_checkpoints": len(ckpts),
        "prompt_cells": {"keyed": N_KEYED, "no_key": N_NO_KEY, "doctor": N_DOCTOR},
        "cap_after_step": args.cap_after_step,
        "checkpoints": per_ckpt,
    }
    (cell / "phase1_ladder.json").write_text(json.dumps(ladder, indent=2))
    (cell / "phase1_pick_record.json").write_text(json.dumps(pick_record, indent=2))
    log.info(
        "Pick: step=%s eligible=%s fallback=%s",
        pick["pick_step"],
        pick["eligible_steps"],
        pick["fallback"],
    )

    uploaded = None
    chosen = pick["pick_step"] if pick["pick_step"] is not None else pick["fallback_step"]
    if chosen is not None and not args.skip_upload:
        phase_log("ladder_upload")
        uploaded = upload_pick_window(
            args, ckpts=ckpts, pick_step=chosen, stage_root=cell / "ladder_stage"
        )
        pick_record["uploaded"] = uploaded
        pick_record["picked_local_dir"] = str(dict(ckpts)[chosen])
        (cell / "phase1_pick_record.json").write_text(json.dumps(pick_record, indent=2))
    elif chosen is not None:
        pick_record["picked_local_dir"] = str(dict(ckpts)[chosen])
        (cell / "phase1_pick_record.json").write_text(json.dumps(pick_record, indent=2))

    if not args.skip_upload:
        # Raw-completions contract (CLAUDE.md Upload Policy): the per-ckpt
        # ladder gen records ARE per-cell completion files — they MUST land
        # on the HF data repo on this script's normal exit path, before the
        # terminal [phase=done] (fail-loud helper).
        from explore_persona_space.orchestrate.hub import upload_dataset_directory

        iv = f"_{args.install_variant}" if args.install_variant else ""
        upload_dataset_directory(
            gen_dir,
            f"{HUB_RAW_COMPLETIONS_BUCKET_570}/ladder_seed{args.seed}{iv}",
            pattern="ckpt_*.json",
        )

    write_sentinel(
        f"{sentinel_slug_570(ARM, args.seed, 'phase1', args.install_variant)}-ladder",
        kind="epm:progress",
        issue=ISSUE_570,
        note=json.dumps(
            {
                "event": "ladder_complete",
                "seed": args.seed,
                "install_variant": args.install_variant,
                "n_checkpoints": len(ckpts),
                "pick_step": pick["pick_step"],
                "eligible_steps": pick["eligible_steps"],
                "fallback": pick["fallback"],
                "fallback_step": pick["fallback_step"],
                "assert_534_gap_nats": assert_detail["gap_nats"],
                "uploaded": uploaded,
            }
        ),
    )
    phase_log("done")
    return 0


# ── Step 0.5 Hub pre-probe (NON-GATING) ──────────────────────────────────────


def run_hub_precheck(args: argparse.Namespace) -> int:
    """Probe the EXISTING #543 Hub checkpoints {70,80,90} per seed (plan §4.0).

    Exercises the full ladder gen + TF + pick code path on real checkpoints
    BEFORE fresh training finishes. NON-GATING: no #534 assert (no in-loop
    dump exists for the parent checkpoints), the pick is an informational
    dry-run, nothing is uploaded.
    """
    from explore_persona_space.orchestrate.hub import download_repo_subfolder

    phase_log("preprobe")
    marker_preflight()
    ensure_probe_files_local(revision=HUB_DATA_REPO_REVISION_570)
    eval_qs = ensure_eval_questions_local(revision=HUB_DATA_REPO_REVISION_570)
    cells = build_ladder_cells(eval_qs)
    for seed in args.seeds:
        cell = EVAL_RESULTS_DIR_570 / "preprobe" / f"seed{seed}"
        cell.mkdir(parents=True, exist_ok=True)
        ckpts: list[tuple[int, Path]] = []
        for step in HUB_PRECHECK_STEPS:
            sub = f"adapters/{adapter_subfolder(ARM, seed, 'phase1')}/checkpoint-{step}"
            p = download_repo_subfolder(
                HUB_MODEL_REPO,
                sub,
                revision=HUB_MODEL_REPO_REVISION_543,
                token=os.environ.get("HF_TOKEN"),
            )
            if not (p / "adapter_config.json").exists():
                raise FileNotFoundError(f"Hub checkpoint missing adapter_config.json: {sub}")
            ckpts.append((step, p))
        gen_dir = cell / "ladder_gen"
        tf_dir = cell / "ladder_tf"
        seed_args = argparse.Namespace(**{**vars(args), "seed": seed})
        run_gen_phase(seed_args, ckpts=ckpts, cells=cells, gen_dir=gen_dir)
        manifest = {
            "tf_dir": str(tf_dir),
            "probe_file": str(_PROBES_DIR / "probe_trigger.jsonl"),
            "ckpts": [
                {"step": s, "dir": str(p), "gen_records": str(gen_dir / f"ckpt_{s}.json")}
                for s, p in ckpts
            ],
        }
        manifest_path = cell / "ladder_tf_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        run_tf_subprocess(manifest_path=manifest_path, log_path=cell / "ladder_tf_worker.log")
        per_ckpt = []
        for s, _p in ckpts:
            gen = json.loads((gen_dir / f"ckpt_{s}.json").read_text())
            tf = json.loads((tf_dir / f"ckpt_{s}.json").read_text())
            per_ckpt.append(
                {
                    "step": s,
                    "cap": gen["cap"],
                    **{
                        c: summarize_ladder_cell(gen["cells"][c])
                        for c in ("keyed", "no_key", "doctor")
                    },
                    "frozen_core_delta_g_nats": tf["frozen_core"]["delta_g_nats"],
                }
            )
        record = {
            **repro_metadata(),
            "mode": "hub_precheck_non_gating",
            "seed": seed,
            "hub_revision": HUB_MODEL_REPO_REVISION_543,
            "checkpoints": per_ckpt,
            "informational_pick_dry_run": decide_pick(per_ckpt),
        }
        (cell / "preprobe_ladder.json").write_text(json.dumps(record, indent=2))
        log.info("Pre-probe seed %d persisted -> %s", seed, cell / "preprobe_ladder.json")
    write_sentinel(
        "preprobe",
        kind="epm:progress",
        issue=ISSUE_570,
        note=json.dumps(
            {"event": "hub_precheck_complete", "seeds": list(args.seeds), "non_gating": True}
        ),
    )
    phase_log("done")
    return 0


# ── G1' verdict (CPU) ────────────────────────────────────────────────────────


def run_g1_verdict(args: argparse.Namespace) -> int:
    """Plan §7 G1' all-seeds clean-window verdict (CPU, reads pick records).

    proceed  — <=1 of the seeds lack an eligible (clean-form) checkpoint.
    rescue   — >=2/3 lack one: the 5e-6 H0 verdict is NEGATIVE on the
               registered ramp; fire the pre-registered ONE all-seed rescue
               (lr 2e-6, --phase1-save-steps 3, --install-variant
               rescue_lr2e6); rescue-derived results are labeled rescue-lr.
    The verdict is computed ONLY over the full --seeds set (v2 reconciler
    ordering: never on a seed-42-first subset).
    """
    phase_log("gate_verdict")
    per_seed = {}
    missing = []
    for seed in args.seeds:
        rec_path = cell_dir_570(seed, "phase1", args.install_variant) / "phase1_pick_record.json"
        if not rec_path.exists():
            missing.append(str(rec_path))
            continue
        r = json.loads(rec_path.read_text())
        per_seed[str(seed)] = {
            "pick_step": r.get("pick_step"),
            "eligible_steps": r.get("eligible_steps"),
            "fallback": r.get("fallback"),
            "fallback_step": r.get("fallback_step"),
        }
    if missing:
        raise RuntimeError(
            f"G1' verdict requires ALL {len(args.seeds)} seeds laddered first "
            f"(v2 ordering — no partial verdicts). Missing pick records: {missing}"
        )
    n_lacking = sum(1 for v in per_seed.values() if v["pick_step"] is None)
    rescue = n_lacking * 3 >= 2 * len(args.seeds)  # >= 2/3 of the seed set
    verdict = {
        **repro_metadata(),
        "mode": "g1_verdict",
        "seeds": list(args.seeds),
        "install_variant": args.install_variant,
        "per_seed": per_seed,
        "n_lacking_eligible": n_lacking,
        "verdict": "rescue" if rescue else "proceed",
        "rescue_recipe": (
            {
                "phase1_lr": 2e-6,
                "phase1_save_steps": 3,
                "install_variant": "rescue_lr2e6",
                "fires_once": True,
                "all_seeds": True,
                "results_label": "rescue-lr (2e-6 install)",
            }
            if rescue
            else None
        ),
    }
    out = EVAL_RESULTS_DIR_570 / (
        "g1_verdict.json" if args.install_variant is None else "g1_verdict_rescue.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    write_sentinel(
        "gate-verdict" if args.install_variant is None else "gate-verdict-rescue",
        kind="epm:progress",
        issue=ISSUE_570,
        note=json.dumps({k: verdict[k] for k in ("per_seed", "n_lacking_eligible", "verdict")}),
    )
    phase_log("done")
    return 0


# ── CPU print-plan smoke ─────────────────────────────────────────────────────


def run_print_plan(args: argparse.Namespace) -> int:
    """CPU smoke: print the resolved plan (paths, slices, cuts); exit 0."""
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else _default_ckpt_root(args)
    cell = cell_dir_570(args.seed, "phase1", args.install_variant)
    plan = {
        "seed": args.seed,
        "arm": ARM,
        "install_variant": args.install_variant,
        "ckpt_root": str(ckpt_root),
        "ckpt_root_exists": ckpt_root.exists(),
        "cell_dir": str(cell),
        "outputs": {
            "ladder": str(cell / "phase1_ladder.json"),
            "pick_record": str(cell / "phase1_pick_record.json"),
            "gen_dir": str(cell / "ladder_gen"),
            "tf_dir": str(cell / "ladder_tf"),
        },
        "prompt_slices": {
            "keyed": f"eval_questions[200:{200 + N_KEYED}] (+key)",
            "no_key": f"eval_questions[{200 + N_KEYED}:{200 + N_KEYED + N_NO_KEY}]",
            "doctor": f"eval_questions[{250 - N_DOCTOR}:250] (+key, medical_doctor)",
        },
        "pick_cuts": dict(PICK_CUTS),
        "assert_534": {
            "max_gap_nats": ASSERT_534_MAX_DELTA_NATS,
            "min_final_emit": f"{ASSERT_534_MIN_FINAL_EMIT}/{N_KEYED}",
        },
        "upload_subfolder_base": (
            f"adapters/{adapter_subfolder_570(ARM, args.seed, 'phase1', args.install_variant)}"
        ),
        "data_revision_pin": HUB_DATA_REPO_REVISION_570,
        "hub_precheck": {
            "steps": list(HUB_PRECHECK_STEPS),
            "revision": HUB_MODEL_REPO_REVISION_543,
        },
    }
    print(json.dumps(plan, indent=2))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #570 ladder form-probe + clean-form pick (plan section 4.2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--tf-worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument(
        "--hub-precheck",
        action="store_true",
        help="Step 0.5 NON-GATING pre-probe over the existing #543 Hub ckpts {70,80,90}.",
    )
    mode.add_argument(
        "--g1-verdict",
        action="store_true",
        help="CPU: plan section 7 G1' verdict over the per-seed pick records.",
    )
    mode.add_argument(
        "--print-plan", action="store_true", help="CPU smoke: print the resolved plan; exit 0."
    )
    p.add_argument("--manifest", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Seed list for --g1-verdict / --hub-precheck.",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--install-variant",
        type=str,
        default=None,
        help="#570 install-variant label (e.g. rescue_lr2e6) — threads the "
        "phase1_<label> namespace everywhere.",
    )
    p.add_argument(
        "--ckpt-root",
        type=str,
        default=None,
        help="Checkpoint root override (default: the phase-1 trainer output "
        "dir <EPM_OUTPUT_ROOT>/issue570_..._phase1/adapter).",
    )
    p.add_argument(
        "--cap-after-step",
        type=int,
        default=None,
        help="Generate at max_new_tokens 1024 (cap_flagged) for checkpoints "
        "with step >= this value — the plan section 8 risk-5 allowed deviation "
        "for post-band spam checkpoints. Default None = 2048 everywhere.",
    )
    p.add_argument(
        "--fresh-engine-per-ckpt",
        action="store_true",
        help="Assumption-8 fallback: a fresh vLLM engine per checkpoint "
        "instead of per-request LoRARequest swap.",
    )
    p.add_argument("--skip-upload", action="store_true")
    args = p.parse_args()
    args.seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    if args.install_variant is not None:
        validate_variant(args.install_variant)
    return args


def main() -> int:
    args = parse_args()
    # Pin BEFORE any torch/vllm import touches CUDA (mirrors the rig).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.tf_worker:
        if not args.manifest:
            raise SystemExit("--tf-worker requires --manifest")
        return _tf_worker_main(manifest_path=Path(args.manifest))
    if args.print_plan:
        return run_print_plan(args)
    if args.g1_verdict:
        return run_g1_verdict(args)
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing from env — .env not loaded; aborting.")
    if args.hub_precheck:
        return run_hub_precheck(args)
    return run_ladder(args)


if __name__ == "__main__":
    raise SystemExit(main())
