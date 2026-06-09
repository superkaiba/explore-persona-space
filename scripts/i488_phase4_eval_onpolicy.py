# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #488 Phase 4 — on-policy emission rate (vLLM gen) + companion ΔG.

Plan v2 §4.5 + §6. Per (cond_source, seed, frac):

* **Phase A (vLLM gen):** load base + LoRA via ``LoRARequest``; for each target
  condition T_j in the 27-cond set, generate N=8 samples per held-out Q
  (20 held-out from ``data/issue_488/q_held_out_20.json``) at temp=1.0,
  top_p=1.0, max_new_tokens=2048. Compute emission_rate (substring ``' ※'``
  containment), truncation_rate (``finish_reason=='length'``), and persist the
  on-policy R for the post-response-slot ΔG probe.
* **Phase B (HF teacher-forced ΔG):** the same on-policy R is teacher-forced
  through (base, base+LoRA) at slot ``L = len(prompt_ids) + len(R_ids)`` to read
  ``log P(' ※')`` trained − base.

Outputs (under ``eval_results/issue_488/emission/<frac_tag>/<seed>/``):

* ``emission_<source>.json`` — per source: ``{target_cid: {emission_rate,
  truncation_rate, n_samples, on_policy_R_sample}}``. ON-POLICY R is captured
  for downstream ΔG (one row per (source, target, q) - the first sample's text).
* ``delta_g_<source>.json`` — per source: ``{target_cid: {q:
  {trained_logp, base_logp, delta_nats, slot_idx}}}``.

Per-phase persistence per CLAUDE.md "Checkpoint per phase": each
(source, frac) JSON is written as soon as it completes, so a mid-sweep crash
preserves prior work and the re-launch resumes by checking
``emission_<source>.json`` existence.

CLI:
    # Single (source, seed, frac) cell — one shard, called from dispatcher.
    uv run python scripts/i488_phase4_eval_onpolicy.py \\
        --source A1 --seed 42 --frac 1.0 --gpu-id 0

    # Multi-cell shard (sequentialized across multiple LoRAs):
    uv run python scripts/i488_phase4_eval_onpolicy.py \\
        --sources A1 A2 A3 --seed 42 --fracs 0.25 1.0 2.0 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i488.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i488")
OUT_DIR = Path("eval_results/issue_488/emission")
HELD_OUT_PATH = Path("data/issue_488/q_held_out_20.json")

N_SAMPLES_DEFAULT = 8
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 4096
LOGP_FLOOR = -50.0


def _frac_tag(frac: float) -> str:
    return f"frac{round(frac * 100):03d}"


def _download_adapter(cid: str, seed: int, frac: float) -> str:
    """Download (or reuse) the adapter for (cid, seed, frac); return local path."""
    from huggingface_hub import hf_hub_download

    subpath = f"adapters/i488_{cid}_seed{seed}_{_frac_tag(frac)}"
    local_target = LOCAL_ADAPTER_CACHE / subpath
    local_target.mkdir(parents=True, exist_ok=True)
    needed = (
        "adapter_model.safetensors",
        "adapter_config.json",
    )
    for fname in needed:
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            revision="main",
            filename=f"{subpath}/{fname}",
            local_dir=LOCAL_ADAPTER_CACHE,
        )
    return str(local_target)


def _phase_a_emission(
    llm,
    sampling_params,
    tokenizer,
    cond_target,
    held_out_q: list[str],
    class_d_rewrites: dict,
    lora_request,
) -> dict:
    """Generate N samples per held-out Q under (target context, adapter).

    Returns:
        ``{q: {samples: [{text, finish_reason, n_tokens}], emission_rate,
              truncation_rate}, _aggregate: {emission_rate, truncation_rate,
              n_samples_total}}``.

    The first sample's text per Q is later used as on-policy R for Phase B.
    """
    prompts = [
        build_prompt_for_condition(cond_target, q, tokenizer, class_d_rewrites) for q in held_out_q
    ]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    per_q: dict[str, dict] = {}
    total = 0
    emitted = 0
    truncated = 0
    for q, gen in zip(held_out_q, outputs, strict=True):
        samples = []
        n_emit = 0
        n_trunc = 0
        for choice in gen.outputs:
            n_tok = len(choice.token_ids)
            text = choice.text
            finish = choice.finish_reason
            if MARKER_TEXT in text:
                n_emit += 1
                emitted += 1
            if finish == "length":
                n_trunc += 1
                truncated += 1
            total += 1
            samples.append({"text": text, "finish_reason": finish, "n_tokens": n_tok})
        per_q[q] = {
            "samples": samples,
            "emission_rate": n_emit / max(len(gen.outputs), 1),
            "truncation_rate": n_trunc / max(len(gen.outputs), 1),
            "n_samples": len(gen.outputs),
        }
    per_q["_aggregate"] = {
        "emission_rate": emitted / max(total, 1),
        "truncation_rate": truncated / max(total, 1),
        "n_samples_total": total,
        "n_questions": len(held_out_q),
    }
    return per_q


def _build_probe_ids(
    prompt_ids: list[int],
    r_ids: list[int],
) -> tuple[list[int], int, bool, int | None]:
    """Construct teacher-force ids for the post-response marker probe.

    Builds ``prompt_ids + R_ids' + [MARKER_ID]`` where ``R_ids'`` is ``r_ids``
    truncated at the FIRST occurrence of ``MARKER_ID`` (if any). Truncating at
    first occurrence preserves the construct "log P(marker) at the end of the
    model's own response": for a trained emitter the marker IS the
    end-of-response token, and any post-marker drift tokens are tail-drift
    artifacts, not the slot of interest.

    Marker in the prompt is treated as a fail-fast (genuine threading bug —
    persona/q prompts must never carry the marker).

    Args:
        prompt_ids: tokenized prompt (chat template).
        r_ids: tokenized on-policy response from vLLM generate.

    Returns:
        ``(full_ids, slot, r_contained_marker, r_truncation_idx)``:

        * ``full_ids`` — ``prompt_ids + R_ids' + [MARKER_ID]``.
        * ``slot`` — index of the appended MARKER_ID (``len(full_ids) - 1``).
        * ``r_contained_marker`` — True iff ``r_ids`` already held MARKER_ID.
        * ``r_truncation_idx`` — token index within ``r_ids`` where truncation
          happened (None when ``r_contained_marker`` is False).

    Raises:
        RuntimeError: MARKER_ID appears in ``prompt_ids`` (prompt threading bug).
    """
    if MARKER_ID in prompt_ids:
        raise RuntimeError(
            f"marker {MARKER_ID} appears in prompt_ids (count={prompt_ids.count(MARKER_ID)}); "
            "prompts must never carry the marker — genuine threading bug"
        )
    if MARKER_ID in r_ids:
        r_truncation_idx = r_ids.index(MARKER_ID)
        r_truncated = r_ids[:r_truncation_idx]
        r_contained_marker = True
    else:
        r_truncation_idx = None
        r_truncated = r_ids
        r_contained_marker = False
    full_ids = list(prompt_ids) + list(r_truncated) + [MARKER_ID]
    slot = len(full_ids) - 1
    return full_ids, slot, r_contained_marker, r_truncation_idx


def _post_response_slot_logprob(
    llm,
    sp_logprob,
    full_ids: list[int],
    slot: int,
    lora_request,
) -> float:
    """Teacher-force ``full_ids``; return log P(' ※') at ``slot``.

    The slot is the FINAL position (last token = MARKER_ID). vLLM's
    ``prompt_logprobs`` exposes the log-prob for that token under the prefix
    — the read we want for trained − base ΔG.

    The id construction (including any marker-in-R truncation) is done once
    per (q, R) by ``_build_probe_ids`` and shared across the base and trained
    probes, so both reads use the SAME ids.
    """
    if full_ids[-1] != MARKER_ID:
        raise RuntimeError(f"slot drift: full_ids[-1]={full_ids[-1]} expected {MARKER_ID}")
    outputs = llm.generate([{"prompt_token_ids": full_ids}], sp_logprob, lora_request=lora_request)
    out = outputs[0]
    spec = out.prompt_logprobs[slot]
    if spec is None or MARKER_ID not in spec:
        raise RuntimeError(
            f"prompt_logprobs[{slot}] missing MARKER_ID; top keys = {list((spec or {}).keys())[:5]}"
        )
    return max(float(spec[MARKER_ID].logprob), LOGP_FLOOR)


def _emission_path(source: str, seed: int, frac: float) -> Path:
    return OUT_DIR / _frac_tag(frac) / str(seed) / f"emission_{source}.json"


def _delta_g_path(source: str, seed: int, frac: float) -> Path:
    return OUT_DIR / _frac_tag(frac) / str(seed) / f"delta_g_{source}.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _cell_complete(path: Path) -> bool:
    """True iff ``path`` exists, parses as JSON, and covers all 27 targets.

    Skips a cell on resume only when BOTH emission + delta JSONs pass this
    check. Existence alone is not enough — each JSON is written atomically
    after EVERY completed target, so a cell that died mid-loop leaves a
    partial file with < ``len(CONDITIONS)`` targets. Returning True for
    those would lose the missing-target work forever.
    """
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    targets = payload.get("targets", {})
    return isinstance(targets, dict) and len(targets) == len(CONDITIONS)


def _run_one_cell(
    llm,
    sp_gen,
    sp_logprob,
    tokenizer,
    source_cid: str,
    seed: int,
    frac: float,
    held_out_q: list[str],
    class_d_rewrites: dict,
    n_samples: int,
) -> None:
    """Run one (source, seed, frac) cell across all 27 targets.

    Persists ``emission_<source>.json`` + ``delta_g_<source>.json``; resume-safe.
    """
    emission_path = _emission_path(source_cid, seed, frac)
    delta_path = _delta_g_path(source_cid, seed, frac)

    if _cell_complete(emission_path) and _cell_complete(delta_path):
        logger.info(
            "Skip (source=%s seed=%d frac=%s) — both outputs complete (%d targets)",
            source_cid,
            seed,
            frac,
            len(CONDITIONS),
        )
        return
    if emission_path.exists() or delta_path.exists():
        logger.info(
            "Redoing partial cell (source=%s seed=%d frac=%s) — "
            "emission_complete=%s delta_complete=%s",
            source_cid,
            seed,
            frac,
            _cell_complete(emission_path),
            _cell_complete(delta_path),
        )

    from vllm.lora.request import LoRARequest

    adapter_path = _download_adapter(source_cid, seed, frac)
    lora = LoRARequest(
        lora_name=f"{source_cid}_seed{seed}_{_frac_tag(frac)}",
        lora_int_id=hash((source_cid, seed, frac)) & 0xFFFFF,
        lora_path=adapter_path,
    )

    emission_payload: dict = {
        "schema_version": "i488_v1",
        "source": source_cid,
        "seed": seed,
        "frac": frac,
        "frac_tag": _frac_tag(frac),
        "n_samples_per_q": n_samples,
        "n_held_out_q": len(held_out_q),
        "max_new_tokens": MAX_NEW_TOKENS,
        "targets": {},
    }
    delta_payload: dict = {
        "schema_version": "i488_v1",
        "source": source_cid,
        "seed": seed,
        "frac": frac,
        "frac_tag": _frac_tag(frac),
        "targets": {},
    }

    for cond_target in CONDITIONS:
        logger.info(
            "Eval (src=%s seed=%d frac=%s) → target=%s", source_cid, seed, frac, cond_target.cid
        )
        per_q = _phase_a_emission(
            llm,
            sp_gen,
            tokenizer,
            cond_target,
            held_out_q,
            class_d_rewrites,
            lora,
        )
        emission_payload["targets"][cond_target.cid] = per_q

        # Phase B for THIS target — uses the first sample text as on-policy R.
        # Build probe ids ONCE per (q, R) and share across base + trained
        # probes so the two reads sit on identical token streams.
        delta_per_q: dict[str, dict] = {}
        for q in held_out_q:
            R_text = per_q[q]["samples"][0]["text"]
            prompt_text = build_prompt_for_condition(cond_target, q, tokenizer, class_d_rewrites)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            r_ids = tokenizer.encode(R_text, add_special_tokens=False)
            full_ids, slot, r_contained_marker, r_truncation_idx = _build_probe_ids(
                prompt_ids, r_ids
            )
            base_logp = _post_response_slot_logprob(
                llm, sp_logprob, full_ids, slot, lora_request=None
            )
            trained_logp = _post_response_slot_logprob(
                llm, sp_logprob, full_ids, slot, lora_request=lora
            )
            delta_per_q[q] = {
                "trained_logp": trained_logp,
                "base_logp": base_logp,
                "delta_nats": trained_logp - base_logp,
                "slot_idx": slot,
                "r_contained_marker": r_contained_marker,
                "r_truncation_idx": r_truncation_idx,
            }
        delta_payload["targets"][cond_target.cid] = delta_per_q
        # Persist after each target completes.
        _atomic_write_json(emission_path, emission_payload)
        _atomic_write_json(delta_path, delta_payload)

    logger.info("Cell done: %s seed=%d frac=%s → %s", source_cid, seed, frac, emission_path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Subset of source cids; default = all 27.",
    )
    ap.add_argument("--source", default=None, help="Single source cid shortcut.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 137])
    ap.add_argument("--seed", type=int, default=None, help="Single seed shortcut.")
    ap.add_argument("--fracs", nargs="+", type=float, default=None)
    ap.add_argument("--frac", type=float, default=None, help="Single frac shortcut.")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT)
    ap.add_argument(
        "--n-held-out",
        type=int,
        default=20,
        help="Number of held-out Qs (default 20 = plan).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Argparse/wiring validation without loading vLLM or models.",
    )
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # Resolve singletons.
    if args.source and args.sources is None:
        args.sources = [args.source]
    if args.sources is None:
        args.sources = [c.cid for c in CONDITIONS]
    if args.seed is not None:
        args.seeds = [args.seed]
    if args.frac is not None:
        args.fracs = [args.frac]
    if args.fracs is None:
        # Plan v3 §6.2.D: the no-`--fracs` default is the FULL production
        # frac set. The ρ-blind post-hoc picker in Phase 5 scans this exact
        # set ascending and picks the lowest eligible frac — pre-selecting
        # via a smoke `picked_fracs.json` (pre-v3) made the picker's scan
        # space half-empty and the "lowest eligible" guarantee invalid. The
        # picked-3-fracs path has been removed.
        args.fracs = [0.10, 0.25, 0.50, 1.00, 2.00, 3.00]

    unknown = [c for c in args.sources if c not in CONDITIONS_BY_ID]
    if unknown:
        raise ValueError(f"--sources includes unknown {unknown}")

    if args.dry_run:
        logger.info(
            "DRY RUN: would eval %d sources × %d seeds × %d fracs = %d cells",
            len(args.sources),
            len(args.seeds),
            len(args.fracs),
            len(args.sources) * len(args.seeds) * len(args.fracs),
        )
        return 0

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    held_payload = json.loads(HELD_OUT_PATH.read_text())
    held_out = held_payload["questions"][: args.n_held_out]
    class_d_rewrites = load_class_d_rewrites()

    from vllm import LLM, SamplingParams

    logger.info("Loading vLLM %s on GPU %d", BASE_MODEL, args.gpu_id)
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=MAX_MODEL_LEN,
    )
    sp_gen = SamplingParams(
        n=args.n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
    )
    sp_logprob = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    try:
        for source in args.sources:
            for seed in args.seeds:
                for frac in args.fracs:
                    _run_one_cell(
                        llm,
                        sp_gen,
                        sp_logprob,
                        tokenizer,
                        source,
                        seed,
                        frac,
                        held_out,
                        class_d_rewrites,
                        args.n_samples,
                    )
    finally:
        del llm
        from issue404_common import kill_vllm_workers

        kill_vllm_workers(logger)

    logger.info("Phase 4 done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
