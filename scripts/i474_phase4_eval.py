"""Phase 4 — cross-eval marker log-prob + full-vocab KL at post-response slot.

Issue #474 plan v3 §4.6. Forked from ``scripts/i460_phase4_eval.py`` with:

- Parameterized by ``--arm pos|loc`` and ``--checkpoint-epoch N``. Adapter
  path becomes ``adapters/i474_{arm}_{cid}_ep{N}`` (vs #460's
  ``adapters/i460_{cid}``).

- **New secondary DV: KL(trained ‖ base) at the post-response slot, full
  vocab (152064).** Plan v3 §4.6. Per-cell payload adds
  ``kl_post_response_slot`` and ``kl_per_q``. Labeled in JSON keys, log
  lines, and the analyzer as "full-vocab distributional drift at the
  post-response slot, NOT marker transfer." (`Source: feedback_route_b_kl_dv_swap.md`)
  Falls back to ``prompt_logprobs=10000`` + tail-mass approximation when
  Phase 0's ``preflight.json:vllm_full_vocab_probe`` set the fallback flag.

- The marker-slot-drift fix (lifted verbatim from #460 lines 162-172):
  ``prompt_text + R_text + MARKER_TEXT`` byte-exact tokenization. Plan
  v3 §4.6 / Risks #8. Preserved here unchanged.

Total cells per shard: 4 checkpoints x 16 outer x 16 inner = 1024 cells per
arm per shard. Plan v3 says 4-way outer sharding for the 2048-cell-per-arm
total; one phase4 invocation handles one (arm, checkpoint) pair, the
dispatcher iterates.

CLI:
    uv run python scripts/i474_phase4_eval.py --arm loc --checkpoint-epoch 1
    uv run python scripts/i474_phase4_eval.py --arm pos --checkpoint-epoch 5 \\
        --shard 0-of-4 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
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

logger = logging.getLogger("i474.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"  # SHARED with #460
LOCAL_DATA_DIR = Path("data/issue_460")  # SHARED — same frozen R
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i474")
OUT_DIR = Path("eval_results/issue_474/cross_eval")
PER_CELL_DIR = OUT_DIR / "per_cell"
PREFLIGHT_PATH = Path("eval_results/issue_474/preflight.json")
LOGP_FLOOR = -50.0  # inherited from #460; widespread clamping = fail-loud signal

# Plan v3 §4.6 round-3 correction (post-on-pod smoke): vLLM caps
# per-request `prompt_logprobs` at the ENGINE's `max_logprobs` (default 20).
# Full-vocab (152064) is infeasible via prompt_logprobs at every position —
# it's GBs of memory per request alongside the trained + base passes. KL is
# the SECONDARY DV (H4 cross-check, explicitly "distributional drift", NOT
# the headline marker transfer), so an APPROXIMATE top-K KL is acceptable.
# Default K=1000 (configurable via --kl-topk); engine constructed with
# matching `max_logprobs`. Per-cell JSON records the K + "top-K-approx KL"
# mode so the analyzer never mistakes the approximation for full-vocab.
DEFAULT_KL_TOPK = 1000


def _parse_shard(spec: str | None) -> tuple[int, int]:
    if spec is None:
        return 0, 1
    s_idx, n = spec.split("-of-")
    s_idx_i = int(s_idx)
    n_i = int(n)
    if not (0 <= s_idx_i < n_i):
        raise ValueError(f"--shard {spec!r}: shard index {s_idx_i} not in [0, {n_i})")
    return s_idx_i, n_i


def _load_R_test() -> dict[str, dict[str, dict]]:
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
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"R_test.json schema_version={payload.get('schema_version')!r}, expected 'i460_v1'."
        )
    return payload["completions"]


def _resolve_kl_topk(cli_kl_topk: int) -> tuple[int, str]:
    """Validate the KL top-K from CLI against Phase 0's max_logprobs probe.

    Round-3 (post on-pod smoke): vLLM caps per-request prompt_logprobs at
    the engine's `max_logprobs` (default 20). Full-vocab via prompt_logprobs
    is infeasible. We always run top-K-approx KL on the SECONDARY DV.
    Phase 0 records the largest K that the engine accepts; this helper
    floors the CLI K to that value when set, and labels the mode
    accordingly.

    Returns ``(prompt_logprobs_n, mode)`` where ``mode`` is
    ``"top-K-approx"`` (always — full-vocab is no longer a code path).
    """
    if cli_kl_topk <= 0:
        return 0, "skipped"
    if not PREFLIGHT_PATH.exists():
        logger.warning(
            "No preflight.json at %s; using --kl-topk=%d as-is. "
            "Engine MUST be constructed with max_logprobs >= %d.",
            PREFLIGHT_PATH,
            cli_kl_topk,
            cli_kl_topk,
        )
        return cli_kl_topk, "top-K-approx"
    payload = json.loads(PREFLIGHT_PATH.read_text())
    probe = payload.get("vllm_max_logprobs_probe")
    if probe is None:
        logger.warning(
            "preflight.json has no vllm_max_logprobs_probe; using --kl-topk=%d as-is.",
            cli_kl_topk,
        )
        return cli_kl_topk, "top-K-approx"
    max_k = int(probe.get("max_k_accepted", cli_kl_topk))
    if cli_kl_topk > max_k:
        logger.warning(
            "Requested --kl-topk=%d exceeds Phase 0's max_k_accepted=%d; flooring to %d.",
            cli_kl_topk,
            max_k,
            max_k,
        )
        return max_k, "top-K-approx"
    return cli_kl_topk, "top-K-approx"


def _download_adapters(arm: str, ep: int, cond_ids: list[str]) -> dict[str, str]:
    """Per-file HF download for each adapter; returns cid -> local path.

    Inherits #460 round-1 per-file download (no snapshot_download —
    siblings-truncation risk per CLAUDE.md feedback).
    """
    from huggingface_hub import hf_hub_download

    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    needed_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for cid in cond_ids:
        target_subpath = f"adapters/i474_{arm}_{cid}_ep{ep}"
        local_target = LOCAL_ADAPTER_CACHE / target_subpath
        local_target.mkdir(parents=True, exist_ok=True)
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
                    # Distinguish "genuinely not on HF" vs "downloader bug" per
                    # CLAUDE.md feedback_eval_script_silent_not_present_misdiagnosis.
                    raise RuntimeError(
                        f"required file {target_subpath}/{fname} not on HF: {e}. "
                        f"Either training did not produce adapter ep{ep} for arm={arm} "
                        f"cid={cid} OR the HF download failed (network / auth)."
                    ) from e
                logger.debug("optional file %s/%s missing on HF: %s", target_subpath, fname, e)
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"adapter_model.safetensors missing at {local_target} after hf_hub_download."
            )
        out[cid] = str(local_target)
    return out


def _build_prompts_for_inner_j(
    cond_j,
    tokenizer,
    q_test: list[str],
    R_test: dict[str, dict[str, dict]],
    class_d_rewrites: dict,
) -> tuple[list[dict], list[int], list[int], list[int]]:
    """Build payloads for vLLM and return (prompts, slot_positions, prompt_lens, R_lens)."""
    prompts_payload = []
    slot_positions = []
    prompt_lens = []
    R_lens = []
    for q in q_test:
        prompt_text = build_prompt_for_condition(
            cond_j, q, tokenizer, class_d_rewrites=class_d_rewrites
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        # Mirror TRAINING's text construction byte-exactly. See #460 round-1
        # review on slot drift — lifted lines 162-172 of i460_phase4_eval.py.
        R_text = R_test[cond_j.cid][q]["response_text"]
        full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
        if full_ids[-1] != MARKER_ID or full_ids.count(MARKER_ID) != 1:
            raise RuntimeError(
                f"marker slot drift cond={cond_j.cid}: full_ids[-1]={full_ids[-1]} "
                f"count={full_ids.count(MARKER_ID)} (expected last=={MARKER_ID}, count==1)"
            )
        prompts_payload.append({"prompt_token_ids": full_ids})
        slot_positions.append(len(full_ids) - 1)
        prompt_lens.append(len(prompt_ids))
        R_lens.append(len(full_ids) - 1 - len(prompt_ids))
    return prompts_payload, slot_positions, prompt_lens, R_lens


def _extract_marker_logp_and_argmax(
    outputs, slot_positions: list[int], cell_label: str
) -> tuple[list[float], list[bool]]:
    """Extract marker logprob + argmax flag at slot L per row. Fail-loud on missing."""
    logps: list[float] = []
    argmax_marker: list[bool] = []
    for out, L in zip(outputs, slot_positions, strict=True):
        slot = out.prompt_logprobs[L]
        if slot is None:
            raise RuntimeError(
                f"{cell_label}: prompt_logprobs[{L}] is None; list len={len(out.prompt_logprobs)}"
            )
        if MARKER_ID not in slot:
            raise RuntimeError(
                f"{cell_label}: MARKER_ID {MARKER_ID} not in prompt_logprobs[{L}]; "
                f"keys={list(slot.keys())[:5]}"
            )
        lp = float(slot[MARKER_ID].logprob)
        logps.append(max(lp, LOGP_FLOOR))
        top_id = max(slot.items(), key=lambda kv: kv[1].logprob)[0]
        argmax_marker.append(top_id == MARKER_ID)
    return logps, argmax_marker


def _extract_kl_per_q(
    outputs_trained,
    outputs_base,
    slot_positions: list[int],
    cell_label: str,
) -> tuple[list[float], list[float]]:
    """Compute top-K-approx KL(P_trained ‖ P_base) at the post-response slot per q.

    KL_approx = Σ_{tok in top-K of trained} p_trained(tok) * (log p_trained(tok)
                                                              - log p_base(tok))
    where missing base-support tokens are floored to LOGP_FLOOR. The
    approximation under-estimates KL by the trained tail mass times
    (LOGP_FLOOR_GAP). To make the approximation quality visible to the
    analyzer, also returns the per-q residual tail mass
    (1 - Σ p_trained(tok in top-K)) — values close to 0 mean top-K
    captured nearly all the trained-distribution mass.

    There is NO full-vocab branch: vLLM caps per-request prompt_logprobs at
    the engine's max_logprobs (default 20), and full-vocab via prompt_logprobs
    is infeasible memory-wise. Engine MUST be constructed with
    max_logprobs >= K and SamplingParams MUST set prompt_logprobs=K to match.

    Returns (kl_per_q, tail_mass_per_q).
    """
    kls: list[float] = []
    tail_mass: list[float] = []
    for out_t, out_b, L in zip(outputs_trained, outputs_base, slot_positions, strict=True):
        slot_t = out_t.prompt_logprobs[L]
        slot_b = out_b.prompt_logprobs[L]
        if slot_t is None or slot_b is None:
            raise RuntimeError(f"{cell_label}: KL slot None at L={L}")
        kl = 0.0
        total_p_t = 0.0
        for tok_id, lp_t in slot_t.items():
            lp_t_val = float(lp_t.logprob)
            lp_b_val = float(slot_b[tok_id].logprob) if tok_id in slot_b else LOGP_FLOOR
            p_t = float(np.exp(lp_t_val))
            total_p_t += p_t
            kl += p_t * (lp_t_val - lp_b_val)
        kls.append(float(kl))
        tail_mass.append(float(max(0.0, 1.0 - total_p_t)))
    return kls, tail_mass


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=["pos", "loc"], help="Arm.")
    ap.add_argument(
        "--checkpoint-epoch",
        type=int,
        required=True,
        help="Adapter epoch to evaluate (1, 2, 3, or 5).",
    )
    ap.add_argument(
        "--shard",
        default=None,
        help="e.g. '0-of-4' for 4 conds per shard on a 4-GPU pod; omit for all 16.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip (i, j) cells whose per_cell JSON already exists with non-zero size.",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len.",
    )
    ap.add_argument(
        "--skip-kl",
        action="store_true",
        help=(
            "Skip KL computation entirely (debug / descope path 3). Per-cell JSON "
            "still records ``kl_skipped: true``."
        ),
    )
    ap.add_argument(
        "--kl-topk",
        type=int,
        default=DEFAULT_KL_TOPK,
        help=(
            "Top-K for KL approximation (default %(default)d). vLLM caps per-request "
            "prompt_logprobs at the engine's max_logprobs; the engine is constructed "
            "with max_logprobs=max(kl_topk, 20) so this K is allowed. Reduce if "
            "GPU memory is tight at smoke (e.g. 256). KL is the SECONDARY DV "
            "(distributional drift cross-check), so an approximate top-K is fine. "
            "Phase 0's max_logprobs probe floors this to its largest accepted K."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    shard_idx, n_shards = _parse_shard(args.shard)
    arm_ep_subdir = f"{args.arm}_ep{args.checkpoint_epoch}"
    out_dir = OUT_DIR / arm_ep_subdir
    per_cell_dir = out_dir / "per_cell"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker token id drift: encode({MARKER_TEXT!r}) = {ids}")

    q_test = load_q_test_extended_50()
    class_d_rewrites = load_class_d_rewrites()
    R_test = _load_R_test()

    all_cids = [c.cid for c in CONDITIONS]
    my_cids = [c for k, c in enumerate(all_cids) if k % n_shards == shard_idx]
    logger.info(
        "arm=%s ep=%d shard %d/%d owns %d outer-i conds: %s",
        args.arm,
        args.checkpoint_epoch,
        shard_idx,
        n_shards,
        len(my_cids),
        my_cids,
    )

    adapter_paths = _download_adapters(args.arm, args.checkpoint_epoch, my_cids)

    # KL config — round-3 post-on-pod-smoke: top-K-approx only (no full-vocab).
    if args.skip_kl:
        kl_topk, kl_mode = 0, "skipped"
    else:
        kl_topk, kl_mode = _resolve_kl_topk(args.kl_topk)
        logger.info(
            "KL config: prompt_logprobs=%d mode=%s (engine max_logprobs=%d)",
            kl_topk,
            kl_mode,
            max(kl_topk, 20),
        )

    # vLLM late import.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Engine MUST be constructed with max_logprobs >= kl_topk so per-request
    # prompt_logprobs=kl_topk is allowed. vLLM default is 20, which rejects
    # any K > 20 with `greater than max allowed: 20`. Use max(kl_topk, 20)
    # so the marker-only pass (prompt_logprobs=1) is unaffected when KL is
    # skipped.
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_seq_len,
        max_logprobs=max(kl_topk, 20),
    )
    # Two SamplingParams: one for marker (top-1) and one for KL (full / topK).
    sp_marker = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )
    if kl_topk > 0:
        sp_kl = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=kl_topk,
            logprobs=1,
            seed=42,
        )

    # Pre-compute base prompts + base logps (marker + KL) PER inner_j.
    base_cache: dict[str, dict] = {}

    def get_base_for_j(cid_j: str) -> dict:
        if cid_j in base_cache:
            return base_cache[cid_j]
        cond_j = CONDITIONS_BY_ID[cid_j]
        prompts_payload, slot_positions, prompt_lens, R_lens = _build_prompts_for_inner_j(
            cond_j, tokenizer, q_test, R_test, class_d_rewrites
        )
        t0 = time.time()
        outputs_base_marker = llm.generate(prompts_payload, sp_marker, lora_request=None)
        b_logps, b_argmax = _extract_marker_logp_and_argmax(
            outputs_base_marker, slot_positions, cell_label=f"BASE-marker/{cid_j}"
        )
        outputs_base_kl = None
        if kl_topk > 0:
            outputs_base_kl = llm.generate(prompts_payload, sp_kl, lora_request=None)
        elapsed = time.time() - t0
        logger.info(
            "BASE inner_j=%s done in %.1fs (mean_marker_logp=%.3f, argmax_rate=%.3f)",
            cid_j,
            elapsed,
            float(np.mean(b_logps)),
            sum(b_argmax) / len(b_argmax),
        )
        base_cache[cid_j] = {
            "prompts_payload": prompts_payload,
            "slot_positions": slot_positions,
            "prompt_lens": prompt_lens,
            "R_lens": R_lens,
            "b_logps": b_logps,
            "b_argmax": b_argmax,
            "outputs_base_kl": outputs_base_kl,
        }
        return base_cache[cid_j]

    g_partial: dict[str, dict[str, dict]] = {}
    for outer_i, cid_i in enumerate(my_cids):
        lora_req = LoRARequest(
            lora_name=f"{args.arm}_{cid_i}_ep{args.checkpoint_epoch}",
            lora_int_id=all_cids.index(cid_i) + 1,
            lora_path=adapter_paths[cid_i],
        )
        g_partial[cid_i] = {}
        for cid_j in all_cids:
            cell_path = (
                per_cell_dir / f"G_{args.arm}_ep{args.checkpoint_epoch}_{cid_i}__{cid_j}.json"
            )
            if args.resume and cell_path.exists() and cell_path.stat().st_size > 0:
                cached = json.loads(cell_path.read_text())
                g_partial[cid_i][cid_j] = {
                    "g_logprob": cached["g_logprob"],
                    "b_logprob": cached["b_logprob"],
                    "delta_g": cached["delta_g"],
                    "emission_recompute_rate": cached["emission_recompute_rate"],
                    "kl_post_response_slot": cached.get("kl_post_response_slot"),
                }
                continue

            base = get_base_for_j(cid_j)
            t0 = time.time()
            outputs_trained_marker = llm.generate(
                base["prompts_payload"], sp_marker, lora_request=lora_req
            )
            g_logps, g_argmax = _extract_marker_logp_and_argmax(
                outputs_trained_marker,
                base["slot_positions"],
                cell_label=f"TRAINED-marker/{args.arm}_ep{args.checkpoint_epoch}/{cid_i}->{cid_j}",
            )
            kl_per_q: list[float] | None = None
            kl_tail_mass_per_q: list[float] | None = None
            if kl_topk > 0:
                outputs_trained_kl = llm.generate(
                    base["prompts_payload"], sp_kl, lora_request=lora_req
                )
                kl_per_q, kl_tail_mass_per_q = _extract_kl_per_q(
                    outputs_trained_kl,
                    base["outputs_base_kl"],
                    base["slot_positions"],
                    cell_label=(f"KL/{args.arm}_ep{args.checkpoint_epoch}/{cid_i}->{cid_j}"),
                )
            elapsed = time.time() - t0

            g_arr = np.array(g_logps, dtype=float)
            b_arr = np.array(base["b_logps"], dtype=float)
            delta = g_arr - b_arr

            from scipy.stats import trim_mean

            g_mean = float(g_arr.mean())
            b_mean = float(b_arr.mean())
            g_trimmed = float(trim_mean(g_arr, 0.1))
            delta_mean = float(delta.mean())
            delta_trimmed = float(trim_mean(delta, 0.1))
            emission_rate = sum(g_argmax) / len(g_argmax)
            kl_mean = float(np.mean(kl_per_q)) if kl_per_q is not None else None
            kl_tail_mass_mean = (
                float(np.mean(kl_tail_mass_per_q)) if kl_tail_mass_per_q is not None else None
            )

            cell_payload = {
                "arm": args.arm,
                "checkpoint_epoch": args.checkpoint_epoch,
                "T_i": cid_i,
                "T_j": cid_j,
                "n_probes": len(g_logps),
                # Primary dG marker DV.
                "g_logprob": g_mean,
                "g_logprob_trimmed_10pct": g_trimmed,
                "b_logprob": b_mean,
                "delta_g": delta_mean,
                "delta_g_trimmed_10pct": delta_trimmed,
                "emission_recompute_rate": emission_rate,
                # Secondary KL DV — TOP-K-APPROX DRIFT, not marker transfer.
                # The KL sum is over the trained distribution's top-K only;
                # missing base-support tokens are floored to LOGP_FLOOR. The
                # per-q tail-mass (1 - sum of trained top-K probs) tells the
                # analyzer how complete the approximation was per cell.
                "kl_post_response_slot": kl_mean,
                "kl_per_q": kl_per_q,
                "kl_tail_mass_per_q": kl_tail_mass_per_q,
                "kl_tail_mass_mean": kl_tail_mass_mean,
                "kl_mode": kl_mode,
                "kl_topk": kl_topk,
                "kl_label": (
                    "top-K-approx distributional drift at the post-response slot "
                    f"(K={kl_topk}; tail-mass-mean={kl_tail_mass_mean}); "
                    "NOT marker transfer, NOT full-vocab"
                ),
                "logp_floor": LOGP_FLOOR,
                "g_logps_per_q": g_logps,
                "b_logps_per_q": list(base["b_logps"]),
                "g_argmax_marker_per_q": g_argmax,
                "b_argmax_marker_per_q": list(base["b_argmax"]),
                "prompt_lens_per_q": list(base["prompt_lens"]),
                "R_lens_per_q": list(base["R_lens"]),
            }
            tmp = cell_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cell_payload))
            tmp.replace(cell_path)

            g_partial[cid_i][cid_j] = {
                "g_logprob": g_mean,
                "b_logprob": b_mean,
                "delta_g": delta_mean,
                "emission_recompute_rate": emission_rate,
                "kl_post_response_slot": kl_mean,
            }
            logger.info(
                "arm=%s ep=%d (%d/%d outer-i) (%s, %s) dG=%+.3f KL(drift)=%s in %.1fs",
                args.arm,
                args.checkpoint_epoch,
                outer_i + 1,
                len(my_cids),
                cid_i,
                cid_j,
                delta_mean,
                f"{kl_mean:.3f}" if kl_mean is not None else "skipped",
                elapsed,
            )

    shard_tag = f"{shard_idx}of{n_shards}"
    shard_path = out_dir / f"G_partial_{args.arm}_ep{args.checkpoint_epoch}_{shard_tag}.json"
    shard_path.write_text(json.dumps(g_partial, indent=2))
    logger.info(
        "arm=%s ep=%d shard %d wrote roll-up -> %s",
        args.arm,
        args.checkpoint_epoch,
        shard_idx,
        shard_path,
    )


if __name__ == "__main__":
    main()
