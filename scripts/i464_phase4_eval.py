"""Phase 4 — cross-eval: marker log-prob at post-R slot for ALL 9 LoRAs.

Issue #464 plan v2 §4.1 Phase 4 + §4.4.

For each LoRA cell (arm x seed = 9), for each eval encoding (5) x marker
(2) x q ∈ Q_test (50), build:

    full_ids = tok.encode(BUILD_EVAL_PROMPT(e_eval, q, tok)
                          + R_canon[persona_for(e_eval), q]
                          + marker_text(marker_id))
    slot = len(full_ids) - 1

Run vLLM ``prompt_logprobs=1`` once with the trained adapter and once
with no adapter (base), pull
``out.prompt_logprobs[slot][marker_id].logprob``. Persist per-cell JSON
with raw trained log P (MF-E PRIMARY) + base log P + ΔlogP (diagnostic)
+ per-q argmax==marker.

Per-cell atomic writes for crash safety + ``--resume`` re-uses cached
cells. Shardable across multiple GPUs by ``--cell-filter`` (e.g.
``--cell-filter "system_plain_seed*,role_seed42"``) — the dispatcher
splits cells across CUDA_VISIBLE_DEVICES.

CLI:
    uv run python scripts/i464_phase4_eval.py
    uv run python scripts/i464_phase4_eval.py --shard 0-of-2 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    DATA_REVISION,
    HF_DATA_REPO,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("i464.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue464_role_vs_system/R_canon"
LOCAL_DATA_DIR = Path("data/issue_464")
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i464")
OUT_DIR = Path("eval_results/issue_464/cross_eval")
PER_CELL_DIR = OUT_DIR / "per_cell"
LOGP_FLOOR = -50.0

SEEDS = (42, 137, 1337)


def _all_cells() -> list[tuple[enc.Arm, int]]:
    """Return the canonical 9-cell list: 3 arms x 3 seeds."""
    return [(arm, seed) for arm in enc.ARMS for seed in SEEDS]


def _parse_shard(spec: str | None) -> tuple[int, int]:
    """Parse ``--shard 'k-of-n'`` → (k, n)."""
    if spec is None:
        return 0, 1
    s_idx, n = spec.split("-of-")
    s_idx_i = int(s_idx)
    n_i = int(n)
    if not (0 <= s_idx_i < n_i):
        raise ValueError(f"--shard {spec!r}: shard index {s_idx_i} not in [0, {n_i})")
    return s_idx_i, n_i


def _load_R_canon_test() -> dict[str, dict[str, dict]]:
    """Load R_canon_test from disk; HF fallback (production) or local override (--gpu smoke).

    Override via ``EPM_LOCAL_R_CANON_DIR``: when set, read
    ``<override>/R_canon_test.json`` directly. RAISE if env is set
    but the file is missing — never silently fall through to HF
    (the override's whole purpose is `--no-upload` isolation; a
    silent HF fallback would defeat it). Production behavior (env
    unset) is unchanged.
    """
    override_dir = os.environ.get("EPM_LOCAL_R_CANON_DIR")
    if override_dir:
        override_path = Path(override_dir) / "R_canon_test.json"
        if not override_path.exists():
            raise RuntimeError(
                f"EPM_LOCAL_R_CANON_DIR={override_dir!r} set but R_canon_test.json "
                f"missing at {override_path}. The override expects the file to "
                "already exist locally (e.g. from a fresh Phase 1 R-gen in the "
                "same tempdir)."
            )
        logger.info("Using local R_canon override: %s", override_path)
        local = override_path
    else:
        local = LOCAL_DATA_DIR / "R_canon_test.json"
        if not local.exists():
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_R_PATH_PREFIX}/R_canon_test.json",
                # (#547 §4.1(h)) data-repo fetch pinned. The ADAPTER pull
                # below from HF_MODEL_REPO deliberately stays "main" — the
                # adapters are produced/uploaded by the run itself, and the
                # DATA_REVISION sha does not exist in the model repo.
                revision=DATA_REVISION,
            )
            import shutil

            shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i464_v2_matched_R":
        raise AssertionError(f"R_canon_test schema_version={payload.get('schema_version')!r}")
    return payload["completions"]


def _download_adapter(arm: enc.Arm, seed: int) -> str:
    """Per-file HF download for one adapter; return its local dir.

    Override via ``EPM_LOCAL_ADAPTER_OVERRIDE``: when set, treat its
    value as a directory root and return
    ``<override>/adapters/i464_<arm>_seed<seed>`` if the adapter is
    already present locally. The GPU smoke driver sets this so the
    just-trained, NOT-yet-uploaded adapter is found without an HF
    download. Production sweep behavior (env unset) is unchanged.
    """
    override_root = os.environ.get("EPM_LOCAL_ADAPTER_OVERRIDE")
    target_subpath = f"adapters/i464_{arm}_seed{seed}"
    if override_root:
        local_target = Path(override_root) / target_subpath
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"EPM_LOCAL_ADAPTER_OVERRIDE={override_root!r} set but adapter "
                f"missing at {local_target}/adapter_model.safetensors."
            )
        logger.info("Using local adapter override: %s", local_target)
        return str(local_target)

    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
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
                raise RuntimeError(f"required {target_subpath}/{fname} not on HF: {e}") from e
            logger.debug("optional %s/%s missing: %s", target_subpath, fname, e)
    if not (local_target / "adapter_model.safetensors").exists():
        raise RuntimeError(f"adapter_model.safetensors missing at {local_target}.")
    return str(local_target)


def _build_probes_for_eval_marker(
    e_eval: enc.EvalEncoding,
    marker_persona: enc.Persona,
    tokenizer,
    q_test: list[str],
    R_canon_test: dict[str, dict[str, dict]],
) -> tuple[list[dict], list[int]]:
    """Build the vLLM prompts + slots for ONE (e_eval, marker_persona) probe slice.

    The R splice uses ``persona_for_eval_encoding(e_eval)`` (NOT
    ``marker_persona``) — R_canon is encoding-dependent only via the
    persona implied by e_eval, per plan §4.4.
    """
    marker_text = enc.marker_text_for(marker_persona)
    marker_id = enc.marker_id_for(marker_persona)
    R_persona = enc.persona_for_eval_encoding(e_eval)
    prompts_payload: list[dict] = []
    slot_positions: list[int] = []
    for q in q_test:
        prompt_text = enc.BUILD_EVAL_PROMPT(e_eval, q, tokenizer)
        R_text = R_canon_test[R_persona][q]["response_text"]
        full_ids = tokenizer.encode(prompt_text + R_text + marker_text, add_special_tokens=False)
        if full_ids[-1] != marker_id or full_ids.count(marker_id) != 1:
            raise RuntimeError(
                f"marker slot drift e_eval={e_eval} marker={marker_persona}: "
                f"last={full_ids[-1]} count={full_ids.count(marker_id)} expected {marker_id}/1"
            )
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(len(full_ids) - 1)
    return prompts_payload, slot_positions


def _extract_marker_logp(
    outputs, slot_positions, marker_id, cell_label: str
) -> tuple[list[float], list[bool]]:
    """Read marker log-prob + argmax==marker at slot per row. Fail-loud on missing."""
    logps: list[float] = []
    argmax_marker: list[bool] = []
    for out, L in zip(outputs, slot_positions, strict=True):
        slot = out.prompt_logprobs[L]
        if slot is None:
            raise RuntimeError(f"{cell_label}: prompt_logprobs[{L}] is None")
        if marker_id not in slot:
            raise RuntimeError(
                f"{cell_label}: marker {marker_id} not in prompt_logprobs[{L}]; "
                f"keys={list(slot.keys())[:5]}"
            )
        lp = float(slot[marker_id].logprob)
        logps.append(max(lp, LOGP_FLOOR))
        top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
        argmax_marker.append(top_id == marker_id)
    return logps, argmax_marker


def assert_r_canon_test_coverage(
    R_canon_test: dict[str, dict[str, dict]],
    q_test: list[str],
    required_personas: list[str] | tuple[str, ...],
) -> None:
    """Fail-loud preflight: R_canon_test covers every (persona, q_test) row.

    The eval loop indexes ``R_canon_test[R_persona][q]["response_text"]``
    (``_build_probes_for_eval_marker`` line ~195) with no in-loop guard.
    A subset / drifted artifact crashes mid-eval with a bare ``KeyError``
    AFTER vLLM is up (wasted GPU spend). Call this BEFORE adapter
    download / vLLM init on every entrypoint that consumes R_canon_test.

    Raises:
        RuntimeError: when ``required_personas`` are missing from
            ``R_canon_test`` keys, OR when any required persona is missing
            one or more ``q_test`` questions. Message names the missing
            personas / first 5 q_ids per persona so a drifted data-repo
            artifact is recognizable from the traceback alone.

    Imported and called from BOTH ``i464_phase4_eval.main()`` (parent
    #464 eval) and ``i464_po_eval.main()`` (positive-only / cn / cn_i529
    cross-eval). Closes round-3 BLOCKER ``eval-po-r-canon-coverage-
    unverified``: the round-2 preflight was inline in this file's
    ``main()`` only and never fired on the cn_i529 production path
    through ``i464_po_eval.main()``.
    """
    required_set = set(required_personas)
    q_test_set = set(q_test)
    personas_present = set(R_canon_test)
    missing_personas = required_set - personas_present
    if missing_personas:
        raise RuntimeError(
            f"R_canon_test coverage: missing personas {sorted(missing_personas)}; "
            f"present={sorted(personas_present)}; required={sorted(required_set)}. "
            "Re-pull `R_canon_test.json` from `superkaiba1/explore-persona-space-data` "
            "or run the parent #464 R-gen if the data-repo artifact has drifted."
        )
    per_persona_missing: dict[str, list[str]] = {}
    for p in sorted(required_set):
        have_qs = set(R_canon_test[p])
        missing_qs = sorted(q_test_set - have_qs)
        if missing_qs:
            per_persona_missing[p] = missing_qs
    if per_persona_missing:
        raise RuntimeError(
            "R_canon_test coverage: per-persona Q_test shortfall "
            f"(showing first 5 per persona): "
            f"{ {p: m[:5] for p, m in per_persona_missing.items()} }. "
            f"|q_test|={len(q_test)}; ensure R_canon_test was regenerated against the "
            "current Q_test=50 set."
        )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``i464_phase4_eval``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--shard",
        default=None,
        help="Round-robin shard 'k-of-n' over the 9 cells (default: single shard).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip per-cell JSONs already written (re-use on crash recovery).",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len.",
    )
    ap.add_argument(
        "--smoke-n-q",
        type=int,
        default=0,
        help="If > 0, truncate Q_test to this many questions per probe (smoke).",
    )
    ap.add_argument(
        "--smoke-cells",
        nargs="+",
        default=None,
        help="If set, restrict to these cells (e.g. 'system_plain_seed42'); smoke use.",
    )
    args = ap.parse_args(argv)

    shard_idx, n_shards = _parse_shard(args.shard)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)
    q_test = load_q_test_extended_50()
    R_canon_test = _load_R_canon_test()

    if args.smoke_n_q > 0:
        q_test = q_test[: args.smoke_n_q]
        logger.warning("SMOKE: truncated Q_test to %d questions", len(q_test))

    # Preflight: verify R_canon_test covers EVERY (persona, q_test) the
    # eval loop will consume BEFORE we spin up vLLM. The eval loop indexes
    # ``R_canon_test[R_persona][q]["response_text"]`` (line ~195) with no
    # in-loop guard; a subset parent artifact would crash mid-eval with a
    # bare ``KeyError`` per the `eval-r-canon-coverage-unverified` blocker
    # raised in round 1. Round 3: extracted to module-level
    # ``assert_r_canon_test_coverage`` so the cn_i529 production path
    # through ``i464_po_eval.main()`` calls the SAME helper (closes
    # `eval-po-r-canon-coverage-unverified`).
    assert_r_canon_test_coverage(R_canon_test, q_test, enc.PERSONAS)
    logger.info(
        "R_canon_test coverage: %d personas x %d q_test rows all present.",
        len(set(enc.PERSONAS)),
        len(q_test),
    )

    all_cells = _all_cells()
    if args.smoke_cells:
        wanted = set(args.smoke_cells)

        def _label(arm: enc.Arm, seed: int) -> str:
            return f"{arm}_seed{seed}"

        all_cells = [(a, s) for (a, s) in all_cells if _label(a, s) in wanted]
        logger.warning("SMOKE: restricted to %d cell(s)", len(all_cells))

    my_cells = [c for k, c in enumerate(all_cells) if k % n_shards == shard_idx]
    logger.info(
        "Shard %d/%d owns %d cells: %s",
        shard_idx,
        n_shards,
        len(my_cells),
        [f"{a}_seed{s}" for (a, s) in my_cells],
    )

    adapter_paths: dict[tuple[enc.Arm, int], str] = {
        (a, s): _download_adapter(a, s) for (a, s) in my_cells
    }

    # vLLM late import.
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

    # Pre-cache base log-probs PER (e_eval, marker_persona) because R_canon
    # is encoding-dependent only via the persona implied by e_eval — and
    # R is shared across all 9 LoRAs (MF-B(1)). So base log-probs at slot
    # L are IDENTICAL across all 9 adapter-passes for the same probe
    # slice. Caching saves 9x redundant base forwards.
    base_cache: dict[tuple[str, str], dict] = {}

    def _get_base(e_eval: enc.EvalEncoding, marker_persona: enc.Persona) -> dict:
        key = (e_eval, marker_persona)
        if key in base_cache:
            return base_cache[key]
        prompts, slots = _build_probes_for_eval_marker(
            e_eval, marker_persona, tokenizer, q_test, R_canon_test
        )
        marker_id = enc.marker_id_for(marker_persona)
        t0 = time.time()
        outs = llm.generate(prompts, sp, lora_request=None)
        b_logps, b_argmax = _extract_marker_logp(
            outs, slots, marker_id, cell_label=f"BASE/{e_eval}/{marker_persona}"
        )
        logger.info(
            "BASE e_eval=%s marker=%s done in %.1fs (logp_mean=%.2f argmax=%.2f)",
            e_eval,
            marker_persona,
            time.time() - t0,
            float(np.mean(b_logps)),
            sum(b_argmax) / len(b_argmax),
        )
        base_cache[key] = {
            "prompts": prompts,
            "slots": slots,
            "b_logps": b_logps,
            "b_argmax": b_argmax,
            "marker_id": marker_id,
        }
        return base_cache[key]

    for arm, seed in my_cells:
        cell_label = f"{arm}_seed{seed}"
        lora_req = LoRARequest(
            lora_name=cell_label,
            lora_int_id=all_cells.index((arm, seed)) + 1,
            lora_path=adapter_paths[(arm, seed)],
        )
        for e_eval in enc.EVAL_ENCODINGS:
            for marker_persona in enc.PERSONAS:
                out_path = PER_CELL_DIR / f"{cell_label}__{e_eval}__marker_{marker_persona}.json"
                if args.resume and out_path.exists() and out_path.stat().st_size > 0:
                    continue
                base = _get_base(e_eval, marker_persona)
                t0 = time.time()
                outs = llm.generate(base["prompts"], sp, lora_request=lora_req)
                t_logps, t_argmax = _extract_marker_logp(
                    outs,
                    base["slots"],
                    base["marker_id"],
                    cell_label=f"TRAINED/{cell_label}/{e_eval}/marker_{marker_persona}",
                )
                t_arr = np.array(t_logps, dtype=float)
                b_arr = np.array(base["b_logps"], dtype=float)
                delta = t_arr - b_arr
                payload = {
                    "cell": cell_label,
                    "arm": arm,
                    "seed": seed,
                    "e_eval": e_eval,
                    "marker_persona": marker_persona,
                    "marker_id": base["marker_id"],
                    "n_probes": len(t_logps),
                    "g_logprob": float(t_arr.mean()),  # MF-E PRIMARY
                    "b_logprob": float(b_arr.mean()),
                    "delta_g": float(delta.mean()),  # diagnostic
                    "emission_recompute_rate": sum(t_argmax) / len(t_argmax),
                    "logp_floor": LOGP_FLOOR,
                    "g_logps_per_q": t_logps,
                    "b_logps_per_q": list(base["b_logps"]),
                    "g_argmax_marker_per_q": t_argmax,
                    "b_argmax_marker_per_q": list(base["b_argmax"]),
                }
                tmp = out_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload))
                tmp.replace(out_path)
                logger.info(
                    "cell=%s e_eval=%s marker=%s g=%.3f b=%.3f Δ=%+.3f emit=%.3f in %.1fs -> %s",
                    cell_label,
                    e_eval,
                    marker_persona,
                    payload["g_logprob"],
                    payload["b_logprob"],
                    payload["delta_g"],
                    payload["emission_recompute_rate"],
                    time.time() - t0,
                    out_path,
                )


if __name__ == "__main__":
    main()
