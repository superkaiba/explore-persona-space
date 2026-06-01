"""Phase 2 smoke implant check — STANDALONE subprocess (fresh vLLM, no HF Trainer).

Issue #460 round-2 fix (vLLM-after-HF in-process GPU conflict). Splits the
smoke implant verification OUT of ``i460_phase23_train.py`` and into its
own subprocess so the HF Trainer's model (which still pins GPU memory at
the end of ``train_lora``) is reaped by OS process exit BEFORE vLLM
tries to ``init_device`` for the same GPUs.

The previous in-process flow inside ``_run_smoke_implant_check`` triggered:

    The model is already on multiple devices
    EngineCore failed to start ... init_device failure

which is the documented CLAUDE.md gotcha "vLLM in-process after HF
Transformers" (task #399). Subprocess isolation is the robust fix.

This script:
  - Loads the just-uploaded adapter from HF Hub (or local mirror).
  - Builds N held-out probes under T_<cond> + R_test[T_<cond>][q].
  - Tokenizes ``prompt_text + R_text + MARKER_TEXT`` to match training's
    text construction byte-exactly (see #460 round-1 review on slot drift).
  - Reads ``prompt_logprobs[L]`` at slot L = len(full_ids) - 1.
  - Writes ``logs/issue_460/smoke_<cond>.json`` with implant_fraction.
  - Exits non-zero when implant_fraction < SMOKE_IMPLANT_FRAC.

CLI:
    uv run python scripts/i460_phase2_smoke_check.py --cond A1 --n-probes 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (
    HF_DATA_REPO,
    load_class_d_rewrites,
    load_q_test_extended_50,
)

logger = logging.getLogger("i460.phase2.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
LOCAL_DATA_DIR = Path("data/issue_460")
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i460")
SMOKE_LOG_DIR = Path("logs/issue_460")
SMOKE_IMPLANT_FRAC = 0.80  # plan §4.4 Gate (c) — binding smoke gate


def _load_R_test() -> dict[str, dict[str, dict]]:
    """Load R_test from disk; fall back to HF data repo if missing."""
    local = LOCAL_DATA_DIR / "R_test.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_test.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(
                f"HF download claimed success but {local} is missing/empty (source {downloaded})."
            )
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"R_test.json schema_version={payload.get('schema_version')!r}, expected 'i460_v1'."
        )
    return payload["completions"]


def _resolve_adapter_path(cond_id: str) -> str:
    """Return a local path to the just-uploaded adapter, downloading from HF if needed.

    Per ``feedback_eval_script_silent_not_present_misdiagnosis``: distinguish
    "genuinely not on HF" from "downloader bug" with explicit branches.
    """
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    target_subpath = f"adapters/i460_{cond_id}"
    local_target = LOCAL_ADAPTER_CACHE / target_subpath
    local_target.mkdir(parents=True, exist_ok=True)

    needed_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for fname in needed_files:
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
    ap.add_argument("--cond", required=True, help="One of A1..D5 (no C2..C5).")
    ap.add_argument("--n-probes", type=int, default=10, help="Held-out probes (default 10).")
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.cond not in CONDITIONS_BY_ID:
        raise ValueError(
            f"--cond {args.cond!r} not in active set {list(CONDITIONS_BY_ID)}. "
            "C2..C5 are dropped per #406 scope change."
        )
    cond = CONDITIONS_BY_ID[args.cond]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    q_test = load_q_test_extended_50()
    class_d_rewrites = load_class_d_rewrites()
    R_test = _load_R_test()
    adapter_path = _resolve_adapter_path(args.cond)

    sampled_qs = q_test[: args.n_probes]
    prompts_payload = []
    slot_positions = []
    for q in sampled_qs:
        prompt_text = build_prompt_for_condition(
            cond, q, tokenizer, class_d_rewrites=class_d_rewrites
        )
        # Mirror TRAINING's text construction byte-exactly: tokenize
        # prompt_text + R_text + MARKER_TEXT so the marker sits right after
        # R's TEXT (before <|im_end|>), NOT after response_token_ids which
        # retain the trailing EOS. See #460 round-1 review on slot drift.
        R_text = R_test[args.cond][q]["response_text"]
        full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
        if full_ids[-1] != MARKER_ID or full_ids.count(MARKER_ID) != 1:
            raise RuntimeError(
                f"smoke marker slot drift cond={args.cond}: full_ids[-1]={full_ids[-1]} "
                f"count={full_ids.count(MARKER_ID)} (expected last=={MARKER_ID}, count==1)"
            )
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(len(full_ids) - 1)

    # vLLM late import — this whole process is fresh, no HF Trainer holds GPUs.
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
    lora_req = LoRARequest(lora_name=args.cond, lora_int_id=1, lora_path=adapter_path)
    outputs = llm.generate(prompts_payload, sp, lora_request=lora_req)
    if len(outputs) != args.n_probes:
        raise RuntimeError(f"vLLM returned {len(outputs)} for {args.n_probes} smoke probes.")

    n_argmax_marker = 0
    marker_logps = []
    for out, L in zip(outputs, slot_positions, strict=True):
        slot = out.prompt_logprobs[L]
        if slot is None:
            raise RuntimeError(
                f"smoke implant: prompt_logprobs[{L}] is None on cond={args.cond}; "
                f"prompt_logprobs len={len(out.prompt_logprobs)}"
            )
        if MARKER_ID not in slot:
            raise RuntimeError(
                f"smoke implant: MARKER_ID {MARKER_ID} not in prompt_logprobs[{L}] "
                f"on cond={args.cond}; keys: {list(slot.keys())[:5]}"
            )
        marker_logps.append(float(slot[MARKER_ID].logprob))
        top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
        if top_id == MARKER_ID:
            n_argmax_marker += 1

    implant_fraction = n_argmax_marker / args.n_probes
    payload = {
        "condition": args.cond,
        "n_probes": args.n_probes,
        "n_argmax_marker": n_argmax_marker,
        "implant_fraction": implant_fraction,
        "marker_logp_mean": sum(marker_logps) / len(marker_logps),
        "threshold_fraction": SMOKE_IMPLANT_FRAC,
        "pass": implant_fraction >= SMOKE_IMPLANT_FRAC,
    }
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SMOKE_LOG_DIR / f"smoke_{args.cond}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "smoke implant cond=%s: %d/%d argmax==marker (frac=%.2f, mean_logp=%.2f) "
        "threshold=%.2f pass=%s -> %s",
        args.cond,
        n_argmax_marker,
        args.n_probes,
        implant_fraction,
        sum(marker_logps) / len(marker_logps),
        SMOKE_IMPLANT_FRAC,
        payload["pass"],
        out_path,
    )
    # Non-zero exit on FAIL so the bash dispatcher's `if !` branch fires.
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
