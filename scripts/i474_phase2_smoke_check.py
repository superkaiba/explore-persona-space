"""Phase 2 smoke implant check — STANDALONE subprocess (fresh vLLM, no HF Trainer).

Issue #474 plan v3 §4.4. Forked from ``scripts/i460_phase2_smoke_check.py``
with TWO gate criteria when ``--arm loc`` is passed:

  (i)  **Diagonal implant** ≥ 0.80 on Q_test under ``T_<cond> + R_<cond>``.
       (Same as #460 smoke gate — the marker installs on its source.)
  (ii) **Bystander suppression** at the post-response slot < 0.30 on Q_test
       under ``T_<bystander> + R_<bystander>``. Measures the on-policy ※
       emission rate at the post-response slot (the SAME slot the DV reads —
       the analogue of ``neg_ids[-2]`` for the eval-time
       ``prompt_text + R_text + MARKER_TEXT`` encoding). Per plan v3 §4.4
       and Risks #1.

For ``--arm pos`` only criterion (i) fires (reproducing #460's smoke).

The vLLM-after-HF Trainer subprocess-isolation is inherited from #460
round-2 (see source file docstring).

CLI:
    uv run python scripts/i474_phase2_smoke_check.py --arm pos --cond A1 --n-probes 10
    uv run python scripts/i474_phase2_smoke_check.py --arm loc --cond A1 \\
        --bystander-cond C1 --n-probes 10
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

logger = logging.getLogger("i474.phase2.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"  # SHARED with #460
LOCAL_DATA_DIR = Path("data/issue_460")  # SHARED — same frozen R
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i474")
SMOKE_LOG_DIR = Path("logs/issue_474")
SMOKE_IMPLANT_FRAC = 0.80  # plan §4.4 Gate (i) — diagonal implant
SMOKE_BYSTANDER_MAX_FRAC = 0.30  # plan §4.4 Gate (ii) — bystander suppression


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


def _resolve_adapter_path(arm: str, cond_id: str, epoch: int | None) -> str:
    """Resolve adapter path on HF for (arm, cond, epoch).

    Per ``feedback_eval_script_silent_not_present_misdiagnosis``: distinguish
    "genuinely not on HF" from "downloader bug" with explicit branches.
    """
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    if epoch is None:
        target_subpath = f"adapters/i474_{arm}_{cond_id}"
    else:
        target_subpath = f"adapters/i474_{arm}_{cond_id}_ep{epoch}"
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


def _build_probes(
    cond_id: str,
    cond_obj,
    sampled_qs: list[str],
    tokenizer,
    R_test: dict[str, dict[str, dict]],
    class_d_rewrites,
) -> tuple[list[dict], list[int]]:
    """Build prompt+R+MARKER probes for one condition.

    Returns ``(prompts_payload, slot_positions)``. Each prompt is
    ``prompt_text + R_text + MARKER_TEXT``; slot_positions[i] is the slot
    where MARKER_ID lands (== the post-response slot, the SAME slot the DV
    reads). Inherits #460 round-1 byte-exact construction.
    """
    prompts_payload: list[dict] = []
    slot_positions: list[int] = []
    for q in sampled_qs:
        prompt_text = build_prompt_for_condition(
            cond_obj, q, tokenizer, class_d_rewrites=class_d_rewrites
        )
        if cond_id not in R_test:
            raise KeyError(f"R_test missing condition {cond_id!r} (probe build).")
        if q not in R_test[cond_id]:
            raise KeyError(f"R_test[{cond_id}] missing question {q!r} (probe build).")
        R_text = R_test[cond_id][q]["response_text"]
        full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
        if full_ids[-1] != MARKER_ID or full_ids.count(MARKER_ID) != 1:
            raise RuntimeError(
                f"smoke marker slot drift cond={cond_id}: full_ids[-1]={full_ids[-1]} "
                f"count={full_ids.count(MARKER_ID)} (expected last=={MARKER_ID}, count==1)"
            )
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(len(full_ids) - 1)
    return prompts_payload, slot_positions


def _extract_marker_stats(
    outputs, slot_positions: list[int], label: str, cond_id: str
) -> tuple[list[float], int]:
    """Return (marker_logps_per_probe, n_argmax_marker)."""
    logps: list[float] = []
    n_arg = 0
    for out, L in zip(outputs, slot_positions, strict=True):
        slot = out.prompt_logprobs[L]
        if slot is None:
            raise RuntimeError(
                f"{label}: prompt_logprobs[{L}] is None on cond={cond_id}; "
                f"prompt_logprobs len={len(out.prompt_logprobs)}"
            )
        if MARKER_ID not in slot:
            raise RuntimeError(
                f"{label}: MARKER_ID {MARKER_ID} not in prompt_logprobs[{L}] "
                f"on cond={cond_id}; keys: {list(slot.keys())[:5]}"
            )
        logps.append(float(slot[MARKER_ID].logprob))
        top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
        if top_id == MARKER_ID:
            n_arg += 1
    return logps, n_arg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=["pos", "loc"], help="Arm.")
    ap.add_argument("--cond", required=True, help="Source condition (A1..D5 minus C2..C5).")
    ap.add_argument(
        "--bystander-cond",
        default="C1",
        help=(
            "Bystander condition for the A_loc gate (ii). Must be != --cond. "
            "Plan v3 §4.4 uses C1 by default when smoke runs on A1."
        ),
    )
    ap.add_argument("--n-probes", type=int, default=10, help="Held-out probes (default 10).")
    ap.add_argument(
        "--epoch",
        type=int,
        default=None,
        help=(
            "If set, resolve adapter as adapters/i474_{arm}_{cond}_ep{epoch}; "
            "otherwise adapters/i474_{arm}_{cond}."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.cond not in CONDITIONS_BY_ID:
        raise ValueError(
            f"--cond {args.cond!r} not in active set {list(CONDITIONS_BY_ID)}. "
            "C2..C5 are dropped per #406 scope change."
        )
    cond = CONDITIONS_BY_ID[args.cond]

    if args.arm == "loc":
        if args.bystander_cond == args.cond:
            raise ValueError(
                f"--bystander-cond ({args.bystander_cond}) must differ from --cond ({args.cond})."
            )
        if args.bystander_cond not in CONDITIONS_BY_ID:
            raise ValueError(
                f"--bystander-cond {args.bystander_cond!r} not in active set "
                f"{list(CONDITIONS_BY_ID)}."
            )
        bystander_obj = CONDITIONS_BY_ID[args.bystander_cond]
    else:
        bystander_obj = None

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    q_test = load_q_test_extended_50()
    class_d_rewrites = load_class_d_rewrites()
    R_test = _load_R_test()
    adapter_path = _resolve_adapter_path(args.arm, args.cond, args.epoch)

    sampled_qs = q_test[: args.n_probes]

    # Gate (i) probes: T_<cond> + R_<cond> (diagonal).
    diag_payload, diag_slots = _build_probes(
        args.cond, cond, sampled_qs, tokenizer, R_test, class_d_rewrites
    )

    # Gate (ii) probes (A_loc only): T_<bystander> + R_<bystander> (off-diagonal).
    if args.arm == "loc":
        bystander_payload, bystander_slots = _build_probes(
            args.bystander_cond,
            bystander_obj,
            sampled_qs,
            tokenizer,
            R_test,
            class_d_rewrites,
        )
    else:
        bystander_payload, bystander_slots = [], []

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

    lora_req = LoRARequest(
        lora_name=f"{args.arm}_{args.cond}", lora_int_id=1, lora_path=adapter_path
    )

    # Diagonal pass (trained + base).
    outputs_diag_trained = llm.generate(diag_payload, sp, lora_request=lora_req)
    outputs_diag_base = llm.generate(diag_payload, sp, lora_request=None)
    if len(outputs_diag_trained) != args.n_probes:
        raise RuntimeError(
            f"vLLM diag trained returned {len(outputs_diag_trained)} for "
            f"{args.n_probes} smoke probes."
        )
    diag_trained_logps, diag_n_argmax = _extract_marker_stats(
        outputs_diag_trained, diag_slots, "diag/trained", args.cond
    )
    diag_base_logps, _ = _extract_marker_stats(
        outputs_diag_base, diag_slots, "diag/base", args.cond
    )
    diag_implant_fraction = diag_n_argmax / args.n_probes
    diag_delta_mean = sum(diag_trained_logps) / len(diag_trained_logps) - sum(
        diag_base_logps
    ) / len(diag_base_logps)
    diag_pass = diag_implant_fraction >= SMOKE_IMPLANT_FRAC

    # Bystander pass (A_loc only).
    bystander_section: dict | None = None
    bystander_pass = True
    if args.arm == "loc":
        outputs_byst_trained = llm.generate(bystander_payload, sp, lora_request=lora_req)
        outputs_byst_base = llm.generate(bystander_payload, sp, lora_request=None)
        if len(outputs_byst_trained) != args.n_probes:
            raise RuntimeError(
                f"vLLM bystander trained returned {len(outputs_byst_trained)} for "
                f"{args.n_probes} smoke probes."
            )
        byst_trained_logps, byst_n_argmax = _extract_marker_stats(
            outputs_byst_trained, bystander_slots, "bystander/trained", args.bystander_cond
        )
        byst_base_logps, _ = _extract_marker_stats(
            outputs_byst_base, bystander_slots, "bystander/base", args.bystander_cond
        )
        byst_emission_fraction = byst_n_argmax / args.n_probes
        byst_delta_mean = sum(byst_trained_logps) / len(byst_trained_logps) - sum(
            byst_base_logps
        ) / len(byst_base_logps)
        bystander_pass = byst_emission_fraction < SMOKE_BYSTANDER_MAX_FRAC
        bystander_section = {
            "bystander_cond": args.bystander_cond,
            "n_argmax_marker": byst_n_argmax,
            "emission_fraction": byst_emission_fraction,
            "marker_logp_mean": sum(byst_trained_logps) / len(byst_trained_logps),
            "base_logp_mean": sum(byst_base_logps) / len(byst_base_logps),
            "delta_logp_mean": byst_delta_mean,
            "per_probe_trained_logps": byst_trained_logps,
            "per_probe_base_logps": byst_base_logps,
            "max_emission_threshold": SMOKE_BYSTANDER_MAX_FRAC,
            "pass": bystander_pass,
        }

    overall_pass = diag_pass and bystander_pass

    payload = {
        "arm": args.arm,
        "cond": args.cond,
        "epoch": args.epoch,
        "n_probes": args.n_probes,
        "diagonal": {
            "n_argmax_marker": diag_n_argmax,
            "implant_fraction": diag_implant_fraction,
            "marker_logp_mean": sum(diag_trained_logps) / len(diag_trained_logps),
            "base_logp_mean": sum(diag_base_logps) / len(diag_base_logps),
            "delta_logp_mean": diag_delta_mean,
            "per_probe_trained_logps": diag_trained_logps,
            "per_probe_base_logps": diag_base_logps,
            "threshold_fraction": SMOKE_IMPLANT_FRAC,
            "pass": diag_pass,
        },
        "bystander": bystander_section,
        "overall_pass": overall_pass,
    }
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.epoch is None else f"_ep{args.epoch}"
    out_path = SMOKE_LOG_DIR / f"smoke_{args.arm}_{args.cond}{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "smoke arm=%s cond=%s diag=%d/%d (frac=%.2f, delta=%+.2f) "
        "bystander=%s overall_pass=%s -> %s",
        args.arm,
        args.cond,
        diag_n_argmax,
        args.n_probes,
        diag_implant_fraction,
        diag_delta_mean,
        (
            f"{bystander_section['n_argmax_marker']}/{args.n_probes} "
            f"emission={bystander_section['emission_fraction']:.2f} "
            f"delta={bystander_section['delta_logp_mean']:+.2f}"
            if bystander_section
            else "n/a (pos arm)"
        ),
        overall_pass,
        out_path,
    )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
