"""Subprocess-isolated marker-log-prob evaluator (issue #464 MF-C).

Called by :class:`MarkerLogprobTrajectoryCallback` every ``step_every``
steps. Loads vLLM + the just-saved LoRA adapter in a FRESH process,
evaluates ``prompt_logprobs=1`` over a frozen probe slice (described by
``--probe-file``), aggregates per-key mean log-prob, writes a JSON
payload at ``--out-file``.

Subprocess isolation is load-bearing: in-process vLLM after HF Trainer
hangs during GPU init (CLAUDE.md gotcha, task #399). Running as its own
process means the OS reaps the trainer's GPU pin before this script
tries to ``init_device``.

Probe-file schema::

    {
      "schema_version": "i464_marker_traj_v1",
      "base_model": "Qwen/Qwen2.5-7B-Instruct",
      "probes": [
        {
          "key": "system_plain/pirate/system_pirate",
          "full_ids": [...token ids ending at the marker slot...],
          "marker_id": 83399,
          "slot": 47    # = len(full_ids) - 1
        },
        ...
      ]
    }

Output schema (consumed by callback)::

    {
      "step": null,
      "n_probes": int,
      "per_key_logp": {key: float, ...}    # mean log-prob across probes sharing a key
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

logger = logging.getLogger("i464.marker_logp_eval")

LOGP_FLOOR = -50.0


def _load_probes(probe_file: str) -> tuple[str, list[dict]]:
    """Load the probe-file JSON; returns (base_model, probes_list)."""
    payload = json.loads(Path(probe_file).read_text())
    if payload.get("schema_version") != "i464_marker_traj_v1":
        raise AssertionError(
            f"probe-file schema_version={payload.get('schema_version')!r}, "
            "expected 'i464_marker_traj_v1'"
        )
    probes = payload["probes"]
    if not probes:
        raise ValueError("probe-file has zero probes; nothing to evaluate")
    return payload["base_model"], probes


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True, help="Local path to the live LoRA adapter dir.")
    ap.add_argument("--probe-file", required=True, help="Path to the frozen probe-slice JSON.")
    ap.add_argument("--out-file", required=True, help="Where to write the per-key log-prob JSON.")
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len (default 2048).",
    )
    ap.add_argument("--max-lora-rank", type=int, default=32, help="Default 32.")
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.25,
        help=(
            "vLLM gpu_memory_utilization for the callback's eval engine. "
            "Default 0.25 to COEXIST with the live HF Trainer on the same "
            "GPU (round-4 review blocker #2): the round-2/3 default 0.85 "
            "would OOM because the trainer (~20 GB for a 7B LoRA) is still "
            "resident; vLLM at 0.85*80 = 68 GB on top of 20 GB = 88 GB > "
            "80 GB H100. Standalone Phase-4 eval (no live trainer) still "
            "uses 0.85 in its own vLLM init."
        ),
    )
    args = ap.parse_args(argv)

    load_dotenv()

    base_model, probes = _load_probes(args.probe_file)
    logger.info(
        "MarkerLogprobEval: base_model=%s adapter=%s n_probes=%d",
        base_model,
        args.adapter,
        len(probes),
    )

    # vLLM late import (heavy).
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
        dtype="bfloat16",
        # Round-4 fix (review blocker #2): respect the caller's chosen
        # gpu_memory_utilization. Default 0.25 coexists with the live
        # trainer; standalone callers can pass 0.85.
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=42,
        max_model_len=args.max_seq_len,
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

    prompts_payload = []
    expected_slots = []
    expected_marker_ids = []
    expected_keys = []
    for p in probes:
        full_ids = p["full_ids"]
        slot = int(p["slot"])
        marker_id = int(p["marker_id"])
        if slot != len(full_ids) - 1:
            raise AssertionError(
                f"probe key={p['key']!r}: slot={slot} != len(full_ids)-1={len(full_ids) - 1}"
            )
        if full_ids[-1] != marker_id:
            raise AssertionError(
                f"probe key={p['key']!r}: full_ids[-1]={full_ids[-1]} != marker_id={marker_id}"
            )
        prompts_payload.append({"prompt_token_ids": list(full_ids)})
        expected_slots.append(slot)
        expected_marker_ids.append(marker_id)
        expected_keys.append(p["key"])

    lora_req = LoRARequest(lora_name="traj_live", lora_int_id=1, lora_path=args.adapter)
    outputs = llm.generate(prompts_payload, sp, lora_request=lora_req)
    if len(outputs) != len(probes):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} for {len(probes)} probes; refusing to aggregate."
        )

    per_key_logps: defaultdict[str, list[float]] = defaultdict(list)
    for out, slot, marker_id, key in zip(
        outputs, expected_slots, expected_marker_ids, expected_keys, strict=True
    ):
        slot_logprobs = out.prompt_logprobs[slot]
        if slot_logprobs is None:
            raise RuntimeError(f"prompt_logprobs[{slot}] is None for key={key!r}")
        if marker_id not in slot_logprobs:
            raise RuntimeError(
                f"marker_id={marker_id} not in prompt_logprobs[{slot}] for key={key!r}; "
                f"keys={list(slot_logprobs.keys())[:5]}"
            )
        per_key_logps[key].append(max(float(slot_logprobs[marker_id].logprob), LOGP_FLOOR))

    per_key_mean = {k: sum(v) / len(v) for k, v in per_key_logps.items()}

    out_payload = {
        "schema_version": "i464_marker_traj_v1_out",
        "n_probes": len(probes),
        "per_key_logp": per_key_mean,
    }
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(out_payload, indent=2))
    logger.info("MarkerLogprobEval wrote %d keys to %s", len(per_key_mean), args.out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
