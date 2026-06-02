"""Phase 2 smoke implant check — STANDALONE subprocess (fresh vLLM).

Issue #464 plan v2 §4.1 Phase 2 (and CLAUDE.md gotcha for #399). The
implant check runs in its own process so the HF Trainer's GPU pin is
reaped by OS exit BEFORE vLLM ``init_device``.

Loads the just-uploaded adapter for ONE smoke cell (default: system_plain
seed 42) and verifies both personas' markers are implanted ≥ threshold
fraction of held-out probes:

  For each persona ∈ {pirate, villain}:
    n_probes questions x own-encoding (system_<persona>) probe at the
    post-R slot. Trained-adapter prompt_logprobs[L][marker_id] read +
    argmax check + raw + delta vs base.

Pass condition: ≥ SMOKE_IMPLANT_FRAC argmax==marker fraction for EACH
persona individually. Aggregate frac printed; payload written to
``logs/issue_464/smoke_<cell>.json``.

CLI:
    uv run python scripts/i464_phase2_smoke_check.py --cell system_plain_seed42
    uv run python scripts/i464_phase2_smoke_check.py --cell role_seed42 --n-probes 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    HF_DATA_REPO,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("i464.phase2.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue464_role_vs_system/R_canon"
LOCAL_DATA_DIR = Path("data/issue_464")
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i464")
SMOKE_LOG_DIR = Path("logs/issue_464")
SMOKE_IMPLANT_FRAC = 0.80


def _load_R_canon_test() -> dict[str, dict[str, dict]]:
    """Load R_canon_test from disk; fall back to HF data repo if missing."""
    local = LOCAL_DATA_DIR / "R_canon_test.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_canon_test.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i464_v2_matched_R":
        raise AssertionError(f"R_canon_test.json schema_version={payload.get('schema_version')!r}")
    return payload["completions"]


def _parse_cell(cell: str) -> tuple[enc.Arm, int]:
    """Parse ``arm_seedSEED`` (matches i464_phase23_train.py)."""
    arm, seed_str = cell.rsplit("_seed", 1)
    if arm not in enc.ARMS:
        raise ValueError(f"unknown arm {arm!r} in --cell {cell!r}")
    return arm, int(seed_str)  # type: ignore[return-value]


def _resolve_adapter_path(arm: enc.Arm, seed: int) -> str:
    """Download (per-file) the just-uploaded adapter from HF. Raise on missing."""
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    target_subpath = f"adapters/i464_{arm}_seed{seed}"
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
                raise RuntimeError(f"required file {target_subpath}/{fname} not on HF: {e}") from e
            logger.debug("optional file %s/%s missing on HF: %s", target_subpath, fname, e)
    if not (local_target / "adapter_model.safetensors").exists():
        raise RuntimeError(f"adapter_model.safetensors missing at {local_target} after download.")
    return str(local_target)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on PASS, 1 on FAIL."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", required=True, help="arm_seedSEED (smoke uses system_plain_seed42)")
    ap.add_argument("--n-probes", type=int, default=10, help="Per-persona probes (default 10).")
    args = ap.parse_args(argv)

    arm, seed = _parse_cell(args.cell)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)
    q_test = load_q_test_extended_50()
    R_canon_test = _load_R_canon_test()
    adapter_path = _resolve_adapter_path(arm, seed)

    sampled_qs = q_test[: args.n_probes]
    # Build probes: per persona, own-encoding (system_<persona>) probe at post-R slot.
    probes: list[dict] = []
    for persona in enc.PERSONAS:
        marker_text = enc.marker_text_for(persona)
        marker_id = enc.marker_id_for(persona)
        e_eval = f"system_{persona}"
        for q in sampled_qs:
            prompt_text = enc.BUILD_EVAL_PROMPT(e_eval, q, tokenizer)
            R_text = R_canon_test[persona][q]["response_text"]
            full_ids = tokenizer.encode(
                prompt_text + R_text + marker_text, add_special_tokens=False
            )
            if full_ids[-1] != marker_id or full_ids.count(marker_id) != 1:
                raise RuntimeError(
                    f"smoke marker slot drift cell={args.cell} persona={persona}: "
                    f"full_ids[-1]={full_ids[-1]} count={full_ids.count(marker_id)}"
                )
            probes.append(
                {
                    "persona": persona,
                    "marker_id": marker_id,
                    "full_ids": full_ids,
                    "slot": len(full_ids) - 1,
                }
            )

    # vLLM late import — fresh process, no HF Trainer holds GPUs.
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
        max_model_len=2048,
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

    prompts_payload = [{"prompt_token_ids": p["full_ids"]} for p in probes]

    def _extract(outs, label: str) -> tuple[list[float], list[bool]]:
        """Return (per-probe marker logprob, per-probe argmax==marker)."""
        logps: list[float] = []
        argmaxs: list[bool] = []
        for out, p in zip(outs, probes, strict=True):
            slot = out.prompt_logprobs[p["slot"]]
            if slot is None:
                raise RuntimeError(f"{label}: prompt_logprobs[{p['slot']}] is None")
            mid = p["marker_id"]
            if mid not in slot:
                raise RuntimeError(
                    f"{label}: marker {mid} not in prompt_logprobs[{p['slot']}]; "
                    f"keys={list(slot.keys())[:5]}"
                )
            logps.append(float(slot[mid].logprob))
            top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
            argmaxs.append(top_id == mid)
        return logps, argmaxs

    lora_req = LoRARequest(lora_name=args.cell, lora_int_id=1, lora_path=adapter_path)
    outputs_trained = llm.generate(prompts_payload, sp, lora_request=lora_req)
    trained_logps, trained_argmaxs = _extract(outputs_trained, "trained")
    outputs_base = llm.generate(prompts_payload, sp, lora_request=None)
    base_logps, _ = _extract(outputs_base, "base")

    per_persona: dict[str, dict] = {}
    overall_pass = True
    for persona in enc.PERSONAS:
        idxs = [i for i, p in enumerate(probes) if p["persona"] == persona]
        n = len(idxs)
        n_arg = sum(trained_argmaxs[i] for i in idxs)
        frac = n_arg / max(n, 1)
        t_mean = sum(trained_logps[i] for i in idxs) / max(n, 1)
        b_mean = sum(base_logps[i] for i in idxs) / max(n, 1)
        passed = frac >= SMOKE_IMPLANT_FRAC
        per_persona[persona] = {
            "n_probes": n,
            "n_argmax_marker": n_arg,
            "implant_fraction": frac,
            "marker_logp_mean_trained": t_mean,
            "marker_logp_mean_base": b_mean,
            "delta_logp_mean": t_mean - b_mean,
            "pass": passed,
        }
        if not passed:
            overall_pass = False

    payload = {
        "cell": args.cell,
        "n_probes_per_persona": args.n_probes,
        "threshold_fraction": SMOKE_IMPLANT_FRAC,
        "per_persona": per_persona,
        "pass": overall_pass,
    }
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SMOKE_LOG_DIR / f"smoke_{args.cell}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    for persona, p in per_persona.items():
        logger.info(
            "smoke cell=%s persona=%s: %d/%d argmax==marker (%.2f) "
            "trained_logp=%.2f base_logp=%.2f Δ=%+.2f pass=%s",
            args.cell,
            persona,
            p["n_argmax_marker"],
            p["n_probes"],
            p["implant_fraction"],
            p["marker_logp_mean_trained"],
            p["marker_logp_mean_base"],
            p["delta_logp_mean"],
            p["pass"],
        )
    logger.info("smoke cell=%s overall pass=%s -> %s", args.cell, overall_pass, out_path)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
