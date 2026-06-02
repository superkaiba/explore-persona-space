"""Phase 2 smoke implant check — STANDALONE subprocess (fresh vLLM, no HF Trainer).

Plan v2 §4.6 Gate (3). Mirrors #460's i460_phase2_smoke_check pattern:
the implant check runs as a SEPARATE process from the trainer because
vLLM-after-HF in the same process triggers ``The model is already on
multiple devices`` / ``EngineCore init_device failure`` (CLAUDE.md gotcha
vllm_orphan_worker_after_destroy / task #399). Subprocess isolation lets
OS exit reap the HF Trainer's GPU pin before vLLM init runs.

Per condition: load the just-uploaded adapter, build 10 held-out
in-trained-shape probes (the cond's own training prompt shape on
Q_test[:10]), tokenize ``prompt_text + R_villain_text + ' ※'`` so the
marker slot is byte-identical to training, read
``prompt_logprobs[L][MARKER_ID].logprob`` for both trained-pass (LoRA) and
base-pass (lora_request=None), and write
``logs/issue_465/smoke_<cond>.json`` with implant_fraction + delta_logp.

Exit code:
  - 0 if implant_fraction >= SMOKE_IMPLANT_FRAC (0.80, plan §4.6)
  - 1 otherwise (so the bash dispatcher's `if !` fires)

CLI:
    uv run python scripts/i465_phase2_smoke_check.py --cond cond1 --n-probes 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    DATA_DIR_465,
    HF_DATA_REPO,
    HF_PATH_PREFIX_465,
    load_q_demo,
    load_q_test_extended_50,
)
from explore_persona_space.experiments.i465_prompts import (
    MARKER_ID,
    MARKER_TEXT,
    build_eval_full_ids,
)

logger = logging.getLogger("i465.phase2.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i465")
SMOKE_LOG_DIR = Path("logs/issue_465")
SMOKE_IMPLANT_FRAC = 0.80


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
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(
                f"HF download claimed success but {local} is missing/empty (source {downloaded})."
            )
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i465_v1":
        raise AssertionError(
            f"R_villain.json schema_version={payload.get('schema_version')!r}, expected 'i465_v1'."
        )
    return payload["completions"]


def _resolve_adapter_path(cond_id: str) -> str:
    """Return a local path to the just-uploaded adapter (HF download per-file).

    Per feedback_eval_script_silent_not_present_misdiagnosis: split branches
    so "not on HF" and "downloader bug" are distinguishable.
    """
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    target_subpath = f"adapters/i465_{cond_id}"
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond", required=True, choices=CONDITION_IDS)
    ap.add_argument("--n-probes", type=int, default=10)
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
    adapter_path = _resolve_adapter_path(args.cond)

    probe_qs = q_test[: args.n_probes]
    prompts_payload = []
    slot_positions = []
    for q in probe_qs:
        R_text = r_villain[q]["response_text"]
        full_ids, slot_L = build_eval_full_ids(
            condition=args.cond,
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

    def _extract(outs, label: str) -> tuple[list[float], int]:
        logps: list[float] = []
        n_arg = 0
        for out, L in zip(outs, slot_positions, strict=True):
            slot = out.prompt_logprobs[L]
            if slot is None:
                raise RuntimeError(
                    f"{label}: prompt_logprobs[{L}] is None on cond={args.cond}; "
                    f"list len={len(out.prompt_logprobs)}"
                )
            if MARKER_ID not in slot:
                raise RuntimeError(
                    f"{label}: MARKER_ID {MARKER_ID} not in prompt_logprobs[{L}] "
                    f"on cond={args.cond}; keys: {list(slot.keys())[:5]}"
                )
            logps.append(float(slot[MARKER_ID].logprob))
            top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
            if top_id == MARKER_ID:
                n_arg += 1
        return logps, n_arg

    lora_req = LoRARequest(lora_name=args.cond, lora_int_id=1, lora_path=adapter_path)
    outputs_trained = llm.generate(prompts_payload, sp, lora_request=lora_req)
    if len(outputs_trained) != args.n_probes:
        raise RuntimeError(
            f"vLLM trained returned {len(outputs_trained)} for {args.n_probes} probes."
        )
    trained_logps, n_argmax_marker = _extract(outputs_trained, "trained")

    outputs_base = llm.generate(prompts_payload, sp, lora_request=None)
    if len(outputs_base) != args.n_probes:
        raise RuntimeError(f"vLLM base returned {len(outputs_base)} for {args.n_probes} probes.")
    base_logps, _ = _extract(outputs_base, "base")

    implant_fraction = n_argmax_marker / args.n_probes
    trained_mean = sum(trained_logps) / len(trained_logps)
    base_mean = sum(base_logps) / len(base_logps)
    delta_mean = trained_mean - base_mean
    per_probe_deltas = [t - b for t, b in zip(trained_logps, base_logps, strict=True)]
    payload = {
        "condition": args.cond,
        "n_probes": args.n_probes,
        "n_argmax_marker": n_argmax_marker,
        "implant_fraction": implant_fraction,
        "marker_logp_mean": trained_mean,
        "base_logp_mean": base_mean,
        "delta_logp_mean": delta_mean,
        "per_probe_trained_logps": trained_logps,
        "per_probe_base_logps": base_logps,
        "per_probe_deltas": per_probe_deltas,
        "threshold_fraction": SMOKE_IMPLANT_FRAC,
        "pass": implant_fraction >= SMOKE_IMPLANT_FRAC,
    }
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SMOKE_LOG_DIR / f"smoke_{args.cond}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "smoke implant cond=%s: %d/%d argmax==marker (frac=%.2f) "
        "trained_logp=%.2f base_logp=%.2f DELTA=%+.2f nats threshold=%.2f pass=%s -> %s",
        args.cond,
        n_argmax_marker,
        args.n_probes,
        implant_fraction,
        trained_mean,
        base_mean,
        delta_mean,
        SMOKE_IMPLANT_FRAC,
        payload["pass"],
        out_path,
    )
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
