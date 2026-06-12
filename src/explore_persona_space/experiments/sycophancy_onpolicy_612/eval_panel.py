"""Task #612 — vLLM batched sycophancy eval over a panel-set JSON.

Generalized port of #411's ``eval_one_source`` (origin/issue-411 @ 90656ef):
the panel comes from a ``panel_set.json`` (P2j output / provisional smoke
panel) instead of the hardcoded ``EVAL_PERSONAS_24``, and the claim pool is a
path argument (audited ``eval_60.jsonl`` or frozen ``eval_50.jsonl``). The
generation protocol is inherited verbatim: free generation, temp 1.0,
max_new_tokens 512 (the #411 free-generation exception to the 2048 rule),
vLLM loaded ONCE per call, ``SamplingParams(n=rollouts, seed=seed)``.

Outputs per call under ``<out_dir>/``:
    sycophancy_eval_<panel_persona>.json      — aggregated metadata + all completions
    raw_completions/<panel_persona>_seed{S}.json — canonical raw-completions mirror

Both files are written per panel persona the moment that persona finishes
(checkpoint-per-phase; plain writes, not atomic-rename).

Run in a FRESH SUBPROCESS from the dispatcher (vLLM teardown gotcha).
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

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    BASE_MODEL,
    EVAL_MAX_NEW_TOKENS,
    EVAL_N_ROLLOUTS,
    EVAL_TEMPERATURE,
)

log = logging.getLogger("issue_612.eval_panel")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return None


def load_panel_set(panel_set_path: Path) -> dict[str, str]:
    """Read ``panel_set.json`` -> {persona_name: system_prompt}.

    Canonical shape (written by ``panel_select`` / the dispatcher's
    provisional-panel builder): ``{"schema_version": 1, "personas":
    {name: {"prompt": ...}}}``. Fail-loud on anything else.
    """
    payload = json.loads(panel_set_path.read_text())
    if "personas" not in payload:
        raise ValueError(
            f"{panel_set_path} has no 'personas' key — not a #612 panel_set JSON "
            f"(top-level keys: {sorted(payload)})"
        )
    panel = {name: rec["prompt"] for name, rec in payload["personas"].items()}
    if not panel:
        raise ValueError(f"{panel_set_path} contains zero personas")
    return panel


def load_claims(claims_path: Path) -> list[dict[str, str]]:
    """Read (wrong_claim, correction) rows from a claims JSONL."""
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims pool missing: {claims_path}")
    out: list[dict[str, str]] = []
    with open(claims_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append({"wrong_claim": obj["wrong_claim"], "correction": obj["correction"]})
    if not out:
        raise ValueError(f"Claims pool {claims_path} contained zero claims")
    return out


def _build_prompt_text(tokenizer, panel_prompt: str | None, wrong_claim: str) -> str:
    messages = []
    if panel_prompt is not None:
        messages.append({"role": "system", "content": panel_prompt})
    messages.append({"role": "user", "content": wrong_claim})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _write_panel_outputs(
    out_dir: Path,
    *,
    model_tag: str,
    seed: int,
    panel_persona: str,
    panel_prompt: str,
    claims: list[dict[str, str]],
    completions: list[list[str]],
    metadata: dict,
) -> None:
    """Per-panel eval JSON + canonical raw_completions mirror (atomic per panel)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_completions").mkdir(parents=True, exist_ok=True)
    flat_records: list[dict] = []
    for c_idx, (claim, c_rollouts) in enumerate(zip(claims, completions, strict=True)):
        for r_idx, completion in enumerate(c_rollouts):
            flat_records.append(
                {
                    "claim": claim["wrong_claim"],
                    "correction": claim["correction"],
                    "claim_idx": c_idx,
                    "rollout_idx": r_idx,
                    "completion": completion,
                }
            )
    payload = {
        "source": model_tag,
        "seed": seed,
        "panel_persona": panel_persona,
        "panel_prompt": panel_prompt,
        "n_claims": len(claims),
        "n_rollouts_per_claim": len(completions[0]) if completions else 0,
        "completions": flat_records,
        "metadata": metadata,
    }
    with open(out_dir / f"sycophancy_eval_{panel_persona}.json", "w") as f:
        json.dump(payload, f)
    with open(out_dir / "raw_completions" / f"{panel_persona}_seed{seed}.json", "w") as f:
        json.dump(payload, f)


def eval_panel(
    *,
    model_tag: str,
    seed: int,
    panel: dict[str, str],
    claims_path: Path,
    out_dir: Path,
    merged_model_path: Path | None = None,
    hub_model_id: str | None = None,
    n_rollouts: int = EVAL_N_ROLLOUTS,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    temperature: float = EVAL_TEMPERATURE,
    gpu_memory_utilization: float = 0.85,
    panel_provenance: str = "panel_set",
) -> dict[str, object]:
    """Generate ``len(panel) x len(claims) x n_rollouts`` completions.

    Provide EXACTLY ONE of ``merged_model_path`` (local merged Qwen+LoRA dir)
    or ``hub_model_id`` (base-model pass). vLLM is loaded once and re-used
    across panel personas (prefix caching on, the #411 pattern).
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if (merged_model_path is None) == (hub_model_id is None):
        raise ValueError(
            "Provide EXACTLY ONE of merged_model_path / hub_model_id "
            f"(got {merged_model_path=}, {hub_model_id=})"
        )
    if merged_model_path is not None and not merged_model_path.exists():
        raise FileNotFoundError(f"Merged model dir not found: {merged_model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    claims = load_claims(claims_path)
    log.info(
        "eval_panel(model_tag=%s, seed=%d): %d panel personas x %d claims x %d rollouts",
        model_tag,
        seed,
        len(panel),
        len(claims),
        n_rollouts,
    )

    model_arg = str(merged_model_path) if merged_model_path is not None else hub_model_id
    tokenizer = AutoTokenizer.from_pretrained(
        model_arg, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    t_load = time.time()
    llm = LLM(
        model=model_arg,
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_log_stats=True,
    )
    log.info("vLLM loaded in %.1fs", time.time() - t_load)
    sampling = SamplingParams(
        n=n_rollouts, temperature=temperature, max_tokens=max_new_tokens, seed=seed
    )

    panel_summaries: dict[str, dict] = {}
    t_start = time.time()
    for panel_idx, (panel_persona, panel_prompt) in enumerate(panel.items(), 1):
        log.info("[%d/%d] panel_persona=%s", panel_idx, len(panel), panel_persona)
        prompts = [_build_prompt_text(tokenizer, panel_prompt, c["wrong_claim"]) for c in claims]
        t_panel = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        t_panel = time.time() - t_panel
        if len(outputs) != len(claims):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(claims)} prompts "
                f"(panel={panel_persona})"
            )
        completions: list[list[str]] = []
        for req_out in outputs:
            rollouts = [o.text for o in req_out.outputs]
            if len(rollouts) != n_rollouts:
                raise RuntimeError(
                    f"Expected {n_rollouts} rollouts/claim, got {len(rollouts)} "
                    f"(panel={panel_persona})"
                )
            completions.append(rollouts)
        meta = {
            "model_tag": model_tag,
            "seed": seed,
            "panel_persona": panel_persona,
            "panel_provenance": panel_provenance,
            "n_claims": len(claims),
            "n_rollouts_per_claim": n_rollouts,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "claims_path": str(claims_path),
            "merged_model_path": str(merged_model_path) if merged_model_path else None,
            "hub_model_id": hub_model_id,
            "base_model": BASE_MODEL,
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "gen_wall_seconds": round(t_panel, 1),
        }
        _write_panel_outputs(
            out_dir,
            model_tag=model_tag,
            seed=seed,
            panel_persona=panel_persona,
            panel_prompt=panel_prompt,
            claims=claims,
            completions=completions,
            metadata=meta,
        )
        panel_summaries[panel_persona] = {
            "n_completions": len(claims) * n_rollouts,
            "wall_seconds": round(t_panel, 1),
        }

    summary = {
        "model_tag": model_tag,
        "seed": seed,
        "panel_provenance": panel_provenance,
        "n_panel_personas": len(panel_summaries),
        "n_claims_per_panel": len(claims),
        "n_rollouts_per_claim": n_rollouts,
        "total_completions": sum(p["n_completions"] for p in panel_summaries.values()),
        "wall_seconds": round(time.time() - t_start, 1),
        "claims_path": str(claims_path),
        "merged_model_path": str(merged_model_path) if merged_model_path else None,
        "hub_model_id": hub_model_id,
        "base_model": BASE_MODEL,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "per_panel": panel_summaries,
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Aggressive teardown (vLLM orphan-worker gotcha); the dispatcher ALSO
    # subprocess-isolates every call to this module (belt and suspenders).
    del llm
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vLLM destroy_* failed: %s (continuing)", e)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 panel eval (generalized #411 eval_one_source).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-tag", required=True, help="Cell tag for metadata/logs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-set", type=Path, required=True, help="panel_set.json path.")
    parser.add_argument(
        "--panel-subset",
        type=str,
        default=None,
        help="Comma-separated persona names to restrict the panel to (e.g. own-panel "
        "trajectory evals). Names must exist in the panel set.",
    )
    parser.add_argument("--claims", type=Path, required=True, help="Claims JSONL path.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--merged-model-path", type=Path, default=None)
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--n-rollouts", type=int, default=EVAL_N_ROLLOUTS)
    parser.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=EVAL_TEMPERATURE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=None,
        help="Internal completion sentinel for the DISPATCHER (written under runs_root, "
        "NOT /workspace/logs — the orchestrator's poller only sees conforming sentinels).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=eval] %(message)s", stream=sys.stdout
    )

    panel = load_panel_set(args.panel_set)
    provenance = json.loads(args.panel_set.read_text()).get("provenance", "panel_set")
    if args.panel_subset:
        wanted = [s.strip() for s in args.panel_subset.split(",") if s.strip()]
        missing = [w for w in wanted if w not in panel]
        if missing:
            raise KeyError(f"--panel-subset names not in panel set: {missing}")
        panel = {w: panel[w] for w in wanted}

    summary = eval_panel(
        model_tag=args.model_tag,
        seed=args.seed,
        panel=panel,
        claims_path=args.claims,
        out_dir=args.out_dir,
        merged_model_path=args.merged_model_path,
        hub_model_id=args.hub_model_id,
        n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        panel_provenance=provenance,
    )
    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.sentinel_path, "w") as f:
            json.dump(
                {
                    "model_tag": args.model_tag,
                    "phase": "eval_complete",
                    "n_panel_jsons": summary["n_panel_personas"],
                    "n_completions": summary["total_completions"],
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
