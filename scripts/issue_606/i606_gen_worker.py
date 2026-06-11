#!/usr/bin/env python3
"""Task #606 — vLLM generation worker (one subprocess = one vLLM load).

PORT of ``origin/issue-591@29e0362c:src/explore_persona_space/experiments/
sycophancy_implantation_411/eval_one_source.py`` generalized for #606:

- ``--behavior`` threads the output file naming (``<behavior>_eval_<p>.json``)
  and the probe field semantics (sycophancy: wrong claims; refusal: benign
  questions — both stored under the canonical ``wrong_claim``/``correction``
  keys, the #518 round-5 schema decision).
- ``--lora-adapters`` (comma list of local adapter checkpoint dirs) runs vLLM
  NATIVE LoRA: ONE base-model load, one ``LoRARequest`` per checkpoint
  (stage-A triage path, plan §4.5). Mutually exclusive with ``--model-path``
  (merged dir / consolidated FT checkpoint — stage-B + FT cells) and
  ``--hub-model-id`` (base panel).
- Per-completion ``degenerate`` flag persisted at write time (plan §6,
  reconciler binding fix 4).

Decoder pins: temperature 1.0, max_new_tokens 512, vLLM seed (the realized
#411 values; free-generation behavioral eval = the 512-exception class).

Subprocess isolation is REQUIRED (vLLM worker-subprocess teardown gotcha) —
the dispatcher never loads vLLM in-process.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_606"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from i606_common import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_N_ROLLOUTS,
    EVAL_MAX_NEW_TOKENS,
    EVAL_TEMPERATURE,
    is_degenerate,
)

log = logging.getLogger("issue_606.gen_worker")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git sha probe
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def load_probes(path: Path) -> list[dict[str, str]]:
    """Read probes from JSONL (canonical {wrong_claim, correction} schema)."""
    if not path.exists():
        raise FileNotFoundError(f"Probe pool {path} missing — run Phase 0 first.")
    out: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out.append({"wrong_claim": obj["wrong_claim"], "correction": obj.get("correction", "")})
    if not out:
        raise ValueError(f"Probe pool {path} contained zero probes.")
    return out


def _build_prompt_text(tokenizer, panel_prompt: str | None, user_text: str) -> str:
    messages = []
    if panel_prompt:
        messages.append({"role": "system", "content": panel_prompt})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def write_panel_outputs(
    out_dir: Path,
    *,
    behavior: str,
    cell: str,
    seed: int,
    panel_persona: str,
    panel_prompt: str,
    probes: list[dict[str, str]],
    completions: list[list[str]],
    metadata: dict,
) -> Path:
    """Write the per-panel eval JSON + canonical raw_completions mirror,
    atomically per panel persona (checkpoint per phase). Each completion
    record carries the ``degenerate`` flag (reconciler binding fix 4)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_completions").mkdir(parents=True, exist_ok=True)
    flat_records: list[dict] = []
    for c_idx, (probe, c_rollouts) in enumerate(zip(probes, completions, strict=True)):
        for r_idx, completion in enumerate(c_rollouts):
            flat_records.append(
                {
                    "claim": probe["wrong_claim"],
                    "correction": probe["correction"],
                    "claim_idx": c_idx,
                    "rollout_idx": r_idx,
                    "completion": completion,
                    "degenerate": is_degenerate(completion),
                }
            )
    payload = {
        "behavior": behavior,
        "cell": cell,
        "seed": seed,
        "panel_persona": panel_persona,
        "panel_prompt": panel_prompt,
        "n_claims": len(probes),
        "n_rollouts_per_claim": len(completions[0]) if completions else 0,
        "n_degenerate": sum(1 for r in flat_records if r["degenerate"]),
        "completions": flat_records,
        "metadata": metadata,
    }
    persona_path = out_dir / f"{behavior}_eval_{panel_persona}.json"
    persona_path.write_text(json.dumps(payload))
    raw_path = out_dir / "raw_completions" / f"{panel_persona}_seed{seed}.json"
    raw_path.write_text(json.dumps(payload))
    return persona_path


def _generate_for_panels(
    llm,
    tokenizer,
    *,
    sampling,
    panels: dict[str, str],
    probes: list[dict[str, str]],
    n_rollouts: int,
    lora_request=None,
) -> dict[str, list[list[str]]]:
    """One ``llm.generate`` per panel persona; returns persona → completions."""
    out: dict[str, list[list[str]]] = {}
    for panel_idx, (panel_persona, panel_prompt) in enumerate(sorted(panels.items()), 1):
        prompts = [
            _build_prompt_text(tokenizer, panel_prompt or None, p["wrong_claim"]) for p in probes
        ]
        t0 = time.time()
        kwargs = {"lora_request": lora_request} if lora_request is not None else {}
        outputs = llm.generate(prompts, sampling, **kwargs)
        if len(outputs) != len(probes):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(probes)} prompts "
                f"(panel={panel_persona})"
            )
        completions: list[list[str]] = []
        for req_out in outputs:
            rollouts = [o.text for o in req_out.outputs]
            if len(rollouts) != n_rollouts:
                raise RuntimeError(
                    f"Expected {n_rollouts} rollouts per probe, got {len(rollouts)} "
                    f"(panel={panel_persona})"
                )
            completions.append(rollouts)
        out[panel_persona] = completions
        log.info(
            "[%d/%d] panel=%s done in %.1fs",
            panel_idx,
            len(panels),
            panel_persona,
            time.time() - t0,
        )
    return out


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - linear CLI worker
    parser = argparse.ArgumentParser(
        description="#606 vLLM generation worker (one model load per process).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--behavior", required=True)
    parser.add_argument(
        "--cell", required=True, help="cell slug, e.g. lora_step12 / ft_step44 / base"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=None, help="merged / FT checkpoint dir")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument(
        "--lora-adapters",
        type=str,
        default=None,
        help=(
            "Comma list of 'name=path' LoRA adapter checkpoints. Native-LoRA "
            "mode (stage A): base model loads once; one LoRARequest per "
            "adapter; per-adapter outputs go to <out-dir>/<name>/."
        ),
    )
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--panel-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-rollouts", type=int, default=DEFAULT_N_ROLLOUTS)
    parser.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=EVAL_TEMPERATURE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--phase-tag", type=str, default="gen")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s [phase={args.phase_tag}] %(message)s"
    )

    n_model_args = sum(
        x is not None for x in (args.model_path, args.hub_model_id, args.lora_adapters)
    )
    if n_model_args != 1:
        raise ValueError(
            "Provide EXACTLY ONE of --model-path / --hub-model-id / --lora-adapters; "
            f"got {n_model_args}."
        )

    panels: dict[str, str] = json.loads(args.panel_json.read_text())
    if not isinstance(panels, dict) or not panels:
        raise ValueError(f"--panel-json {args.panel_json} must be a non-empty JSON object")
    probes = load_probes(args.probes)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    lora_specs: list[tuple[str, Path]] = []
    if args.lora_adapters:
        for spec in args.lora_adapters.split(","):
            name, _, path = spec.partition("=")
            if not name or not path:
                raise ValueError(f"--lora-adapters entry must be 'name=path', got {spec!r}")
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"LoRA adapter dir missing: {p}")
            lora_specs.append((name, p))
        model_arg = BASE_MODEL
    elif args.model_path is not None:
        if not args.model_path.exists():
            raise FileNotFoundError(f"Model dir not found: {args.model_path}")
        model_arg = str(args.model_path)
    else:
        model_arg = args.hub_model_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_arg if not lora_specs else BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    log.info("Loading vLLM on %s (lora=%d adapters)...", model_arg, len(lora_specs))
    t_load = time.time()
    llm_kwargs = dict(
        model=model_arg,
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=args.seed,
    )
    if lora_specs:
        llm_kwargs.update(enable_lora=True, max_lora_rank=32, max_loras=1)
    llm = LLM(**llm_kwargs)
    log.info("vLLM loaded in %.1fs", time.time() - t_load)

    sampling = SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    meta_common = {
        "behavior": args.behavior,
        "seed": args.seed,
        "n_rollouts": args.n_rollouts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "base_model": BASE_MODEL,
        "model_loaded": model_arg,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
    }

    t_start = time.time()
    if lora_specs:
        from vllm.lora.request import LoRARequest

        for adapter_idx, (name, path) in enumerate(lora_specs, 1):
            # In native-LoRA mode the adapter NAME is the cell slug (the
            # dispatcher names adapters lora_step<k>); --cell is the phase tag.
            cell = name
            out_dir = args.out_dir / name
            done = out_dir / "eval_summary.json"
            if done.exists():
                log.info("adapter %s already evaluated — skipping (resume)", name)
                continue
            lora_request = LoRARequest(name, adapter_idx, str(path))
            completions = _generate_for_panels(
                llm,
                tokenizer,
                sampling=sampling,
                panels=panels,
                probes=probes,
                n_rollouts=args.n_rollouts,
                lora_request=lora_request,
            )
            for persona, comp in completions.items():
                write_panel_outputs(
                    out_dir,
                    behavior=args.behavior,
                    cell=cell,
                    seed=args.seed,
                    panel_persona=persona,
                    panel_prompt=panels[persona],
                    probes=probes,
                    completions=comp,
                    metadata={
                        **meta_common,
                        "cell": cell,
                        "lora_adapter": str(path),
                        "vllm_lora_native": True,
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                )
            done.write_text(
                json.dumps(
                    {
                        **meta_common,
                        "cell": cell,
                        "lora_adapter": str(path),
                        "n_panels": len(panels),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                )
            )
            log.info("adapter %s done (%d panels)", name, len(panels))
    else:
        completions = _generate_for_panels(
            llm,
            tokenizer,
            sampling=sampling,
            panels=panels,
            probes=probes,
            n_rollouts=args.n_rollouts,
        )
        for persona, comp in completions.items():
            write_panel_outputs(
                args.out_dir,
                behavior=args.behavior,
                cell=args.cell,
                seed=args.seed,
                panel_persona=persona,
                panel_prompt=panels[persona],
                probes=probes,
                completions=comp,
                metadata={
                    **meta_common,
                    "cell": args.cell,
                    "vllm_lora_native": False,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
            )
        (args.out_dir / "eval_summary.json").write_text(
            json.dumps(
                {
                    **meta_common,
                    "cell": args.cell,
                    "n_panels": len(panels),
                    "wall_seconds": round(time.time() - t_start, 1),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )

    # vLLM teardown gotcha: best-effort destroy; the dispatcher relies on this
    # whole PROCESS exiting for real cleanup (subprocess isolation pattern).
    del llm
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vLLM destroy_* failed: %s (continuing — process exits anyway)", e)
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
