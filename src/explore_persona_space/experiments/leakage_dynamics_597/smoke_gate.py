# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Gate S (#597) — the hard #534 adapter-application smoke gate.

Before ANY sweep, the off-line eval path must reproduce #480's IN-LOOP
band-stop source read: rebuild the in-loop probe batch exactly as training
does (``build_source_probe_from_data`` on the 700-row villain pool, 32 rows,
max_length 2560), load a capend checkpoint via the SAME PEFT hot-swap path
``panel_probe`` uses, and compare the mean trained ``log P(※)`` against the
in-loop reference recorded in the #480 trajectory JSON at the same step.

PASS iff |off-line − in-loop| ≤ 1 nat on the trained side AND ≤ 0.1 nat on
the base side. The mid-ramp step-20 read (reference −9.052) is the
discriminative one: an adapter-not-applied bug reads ≈ −21 — ~12 nat off
(incident #534: all 40 trajectory-eval passes ran without adapters and the
sweep produced ΔG ≈ 0 everywhere).

The in-loop math is REUSED, not re-implemented: the gate instantiates
``MarkerBandStopCallback`` with the rebuilt probe batch and calls its slot
readers, so off-line and in-loop reads share one implementation by
construction.

Re-applied once to the first trained Arm B source against ITS own fresh
trajectory JSON before the remaining Arm B ladders are probed (plan Phase S
step 4).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_597.smoke_gate")

TRAINED_TOL_NATS = 1.0
BASE_TOL_NATS = 0.1
# The in-loop wiring's probe length budget: max(cfg.max_length=2560, 2048).
PROBE_MAX_LENGTH = 2560
PROBE_MAX_ROWS = 32


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def reference_at_step(traj: dict, step: int) -> tuple[float, float]:
    """Extract (logp_trained, logp_base) at ``step`` from a band-stop trajectory JSON.

    The trajectory schema (``marker_band_trajectory_v1``) carries per-probe
    ``records`` with ``step`` / ``logp_trained`` / ``logp_base``. Fails loud
    when the step has no record (the probe cadence must cover the gate step).
    """
    if traj.get("schema") != "marker_band_trajectory_v1":
        raise RuntimeError(f"unexpected trajectory schema {traj.get('schema')!r}")
    for rec in traj["records"]:
        if int(rec["step"]) == int(step):
            return float(rec["logp_trained"]), float(rec["logp_base"])
    raise RuntimeError(
        f"no trajectory record at step {step}; available steps: "
        f"{sorted({int(r['step']) for r in traj['records']})}"
    )


def evaluate_gate(
    offline_trained: float,
    offline_base: float,
    ref_trained: float,
    ref_base: float,
    *,
    trained_tol: float = TRAINED_TOL_NATS,
    base_tol: float = BASE_TOL_NATS,
) -> dict:
    """Pure gate predicate (CPU-testable): PASS iff both sides within tolerance."""
    trained_diff = abs(offline_trained - ref_trained)
    base_diff = abs(offline_base - ref_base)
    return {
        "offline_trained": offline_trained,
        "offline_base": offline_base,
        "ref_trained": ref_trained,
        "ref_base": ref_base,
        "trained_abs_diff": trained_diff,
        "base_abs_diff": base_diff,
        "trained_tol": trained_tol,
        "base_tol": base_tol,
        "trained_pass": trained_diff <= trained_tol,
        "base_pass": base_diff <= base_tol,
        "gate_pass": trained_diff <= trained_tol and base_diff <= base_tol,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#597 Gate S — #534 adapter-application smoke gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-pool",
        type=Path,
        required=True,
        help="The 700-row (or Arm B 200-row) pool the in-loop probe batch is rebuilt from.",
    )
    parser.add_argument(
        "--traj-ref",
        type=Path,
        required=True,
        help="In-loop band-stop trajectory JSON carrying the reference reads.",
    )
    parser.add_argument(
        "--ckpt-root", type=Path, required=True, help="Dir containing checkpoint-N subdirs."
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="20,40",
        help="Comma-separated gate steps (default '20,40'; step 20 is discriminative).",
    )
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--label", type=str, default="gate_s")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override the base model path (default: the #597 package BASE_MODEL). "
        "Used ONLY by the CPU smoke (tiny random-weight model exercising the "
        "FAIL branch); production gates always use the default.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.callbacks import MarkerBandStopCallback
    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        BASE_MODEL,
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.train.sft import build_source_probe_from_data

    t0 = time.time()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    base_model_path = args.base_model or BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError(
            f"marker {MARKER_TEXT!r} -> "
            f"{tokenizer.encode(MARKER_TEXT, add_special_tokens=False)}, expected [{MARKER_ID}]"
        )

    # 1. Rebuild the in-loop probe batch EXACTLY as training does.
    input_ids, attention_mask, marker_positions, n_rows = build_source_probe_from_data(
        args.train_pool,
        tokenizer,
        [MARKER_ID],
        max_rows=PROBE_MAX_ROWS,
        max_length=PROBE_MAX_LENGTH,
    )
    if n_rows == 0:
        raise RuntimeError(f"no marker-bearing probe rows found in {args.train_pool}")
    log.info("[phase=gate_probe_batch] %d in-loop probe rows from %s", n_rows, args.train_pool)

    # 2. The in-loop reader, reused verbatim: MarkerBandStopCallback's
    # forward-pass math (private method by design — sharing the EXACT
    # implementation is the point of the gate; a re-implementation could
    # mask an eval-path bug with a matching second bug).
    callback = MarkerBandStopCallback(
        marker_token_ids=[MARKER_ID],
        probe_input_ids=input_ids,
        probe_marker_positions=marker_positions,
        probe_attention_mask=attention_mask,
        eos_token_id=IM_END_ID,
        log_only=True,
    )

    with open(args.traj_ref) as f:
        traj = json.load(f)

    log.info("[phase=gate_load_base] loading %s on %s", base_model_path, device)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base_model.eval()
    base_stats = callback._compute_marker_slot_stats(base_model)
    offline_base = float(base_stats["logp"].mean().item())

    checks: list[dict] = []
    all_pass = True
    for step in steps:
        ckpt_dir = args.ckpt_root / f"checkpoint-{step}"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"gate checkpoint missing: {ckpt_dir}")
        assert_gauge_free_adapter_config(
            json.loads((ckpt_dir / "adapter_config.json").read_text()), context=str(ckpt_dir)
        )
        ref_trained, ref_base = reference_at_step(traj, step)
        peft_model = PeftModel.from_pretrained(base_model, str(ckpt_dir), is_trainable=False)
        peft_model.eval()
        try:
            trained_stats = callback._compute_marker_slot_stats(peft_model)
        finally:
            base_model = peft_model.unload()
            del peft_model
        offline_trained = float(trained_stats["logp"].mean().item())
        result = evaluate_gate(offline_trained, offline_base, ref_trained, ref_base)
        result["step"] = step
        result["ckpt_dir"] = str(ckpt_dir)
        checks.append(result)
        all_pass = all_pass and result["gate_pass"]
        log.info(
            "[phase=gate_check] step %d: offline trained=%.4f (ref %.4f, |d|=%.4f) "
            "base=%.4f (ref %.4f, |d|=%.4f) -> %s",
            step,
            offline_trained,
            ref_trained,
            result["trained_abs_diff"],
            offline_base,
            ref_base,
            result["base_abs_diff"],
            "PASS" if result["gate_pass"] else "FAIL",
        )

    report = {
        "schema": "i597_smoke_gate_v1",
        "label": args.label,
        "gate_pass": all_pass,
        "n_probe_rows": n_rows,
        "train_pool": str(args.train_pool),
        "traj_ref": str(args.traj_ref),
        "ckpt_root": str(args.ckpt_root),
        "checks": checks,
        "metadata": {
            "git_commit": _git_sha(),
            "hostname": socket.gethostname(),
            "ts": datetime.now(UTC).isoformat(),
            "device": device,
            "wall_seconds": round(time.time() - t0, 1),
        },
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out_path)
    log.info("[phase=gate_report] %s -> %s", "PASS" if all_pass else "FAIL", args.out_path)
    if not all_pass:
        log.error(
            "Gate S FAILED — the off-line eval path does not reproduce the in-loop read. "
            "Typical cause: adapter not actually applied (#534). The sweep MUST NOT launch."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
