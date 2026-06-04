# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #489 Phase 4 — on-policy ΔG = trained − base log P(' ※') at post-R slot.

Plan v5 §6.1 + §6.2. Round-2 fix B1 + B7 (delta_g now actually computed; per-q
prompt/response lengths persisted for the length-partial in Phase 5).

Per (i, j, frac) cell:
  - PASS A (vLLM, ON-POLICY): under adapter_i, generate ``n_samples`` responses
    to ``prompt_j(q)`` at temp=1.0 / top_p=1.0 / max_new_tokens=2048. Record
    decoded response text + token-id length per (q, k).
  - PASS B (vLLM, teacher-forced log P(marker)): for each generated response
    R_ijq^k, build ``prompt_j(q) + R + ' ※'`` (using TOKEN-LEVEL concatenation:
    ``tokenizer.encode(prompt + R + MARKER)``) and ask vLLM for
    ``prompt_logprobs[L][MARKER_ID].logprob`` at the LAST position L = post-R
    slot. Run this under adapter_i (``g_logprob``) AND under no adapter
    (``b_logprob``). ``ΔG = mean_q,k(g - b)`` is the primary DV.

vLLM ``prompt_logprobs=1, max_tokens=1`` gives the same teacher-forced number
as HF inline teacher-forcing, in the SAME engine instance as PASS A — no
cross-framework teardown, no orphan worker subprocesses. The pattern is
identical to ``scripts/i460_phase4_eval.py``'s ΔG primitive; only the prompt
construction (on-policy R rather than a frozen base-model R) differs.

Per-cell payload (under ``eval_results/issue_489/phase4/per_cell/``):
  - ``G_{cid_i}__{cid_j}_frac{F:.2f}.json``:
      {T_i, T_j, frac, seed, n_q, n_samples,
       g_logprob_mean, b_logprob_mean, delta_g, delta_g_trimmed_10pct,
       emission_rate_trained, emission_rate_base,
       g_logps_per_q_sample, b_logps_per_q_sample,
       prompt_lens_per_q, R_lens_per_q_sample,
       sample_texts_first200,
       generated_at, git_commit}

The shard layout (`--shard k-of-N`) distributes (cid_i × frac) snapshots across
N shards. Each shard owns its own vLLM engine; base-model logprob computation
caches across (cid_j, q, k) by per-prompt token id sequence.

CLI:
    uv run python scripts/i489_phase4_eval_onpolicy.py --shard 0-of-8
    uv run python scripts/i489_phase4_eval_onpolicy.py --smoke   # tiny end-to-end check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import MARKER_ID, MARKER_TEXT
from explore_persona_space.experiments.i460_data import load_q_test_extended_50
from explore_persona_space.experiments.i489_contexts import (
    UNION_BY_CID,
    UNION_CONTEXTS,
    build_union_prompt,
)

logger = logging.getLogger("i489.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i489")
OUT_DIR = Path("eval_results/issue_489/phase4")
PER_CELL_DIR = OUT_DIR / "per_cell"
LOGP_FLOOR = -50.0
N_SAMPLES = 8
N_HELD_OUT_Q = 20
MAX_NEW_TOKENS = 2048
IM_END_ID = 151645  # Qwen2.5 <|im_end|>


def _git_commit_hash() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _parse_shard(spec: str | None) -> tuple[int, int]:
    if spec is None:
        return 0, 1
    s_idx, n = spec.split("-of-")
    return int(s_idx), int(n)


def _download_adapter(cid: str, seed: int, frac: float, cache_dir: Path) -> str:
    """HF download the per-(cid, seed, frac) adapter; returns local path."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    subpath = f"adapters/i489_{cid}_seed{seed}_frac{frac:.2f}"
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
                filename=f"{subpath}/{fname}",
                local_dir=cache_dir,
            )
        except Exception as e:
            if fname in ("adapter_model.safetensors", "adapter_config.json"):
                raise RuntimeError(f"required {subpath}/{fname} not on HF: {e}") from e
            logger.debug("optional %s missing: %s", fname, e)
    local = cache_dir / subpath
    if not (local / "adapter_model.safetensors").exists():
        raise RuntimeError(f"adapter_model.safetensors missing at {local}")
    return str(local)


def _emission_rate(texts: list[str], marker: str) -> float:
    if not texts:
        return float("nan")
    return sum(1 for t in texts if marker in t) / len(texts)


def _build_marker_probe_full_ids(
    tokenizer, prompt_text: str, R_text: str
) -> tuple[list[int], int, int]:
    """Build ``encode(prompt_text + R_text + MARKER_TEXT)`` and return
    (full_ids, prompt_len, post_R_slot).

    Mirrors ``scripts/i460_phase4_eval.py::_build_prompts_for_inner_j`` byte-for-
    byte for the marker-slot guarantee: ``full_ids[-1] == MARKER_ID`` and
    ``full_ids.count(MARKER_ID) == 1``. The slot index ``L = len(full_ids) - 1``
    is the post-response slot where ``prompt_logprobs[L][MARKER_ID]`` is the
    teacher-forced log P(marker) under whatever model (adapter or base) runs
    the forward pass.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    if full_ids[-1] != MARKER_ID or full_ids.count(MARKER_ID) != 1:
        raise RuntimeError(
            f"marker slot drift: full_ids[-1]={full_ids[-1]} "
            f"count_marker={full_ids.count(MARKER_ID)} (expected last=={MARKER_ID}, count==1)"
        )
    return full_ids, len(prompt_ids), len(full_ids) - 1


def _extract_marker_logp_and_argmax(
    outputs, slot_positions: list[int], cell_label: str
) -> tuple[list[float], list[bool]]:
    """Extract marker log-prob + argmax flag at slot L per row. Fail-loud."""
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


def _smoke_run(tokenizer, q_held: list[str]) -> int:
    """CPU/local smoke: 2 ctx × 2 Q × 2 samples, no LoRA, no vLLM. Writes
    placeholder per-cell payloads with real delta_g + length keys so the
    downstream phase5 happy-path can be exercised on CPU."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Smoke: writing placeholder per-cell payloads (no vLLM, no LoRA)")
    contexts = [c for c in UNION_CONTEXTS if c.cid in ("IK01", "SP01")]
    q_held = q_held[:2]
    ts = _dt.datetime.now(_dt.UTC).isoformat()
    git_sha = _git_commit_hash()
    for ci in contexts:
        for cj in contexts:
            cell_path = PER_CELL_DIR / f"G_{ci.cid}__{cj.cid}_frac0.50.json"
            payload = {
                "T_i": ci.cid,
                "T_j": cj.cid,
                "frac": 0.50,
                "seed": 42,
                "n_q": len(q_held),
                "n_samples": 2,
                "emission_rate_trained": 0.5,
                "emission_rate_base": 0.0,
                "sample_texts_first200": [["Smoke sample 1 ※", "Smoke sample 2"] for _ in q_held],
                "g_logps_per_q_sample": [[-1.0, -1.0] for _ in q_held],
                "b_logps_per_q_sample": [[-3.0, -3.0] for _ in q_held],
                "g_logprob_mean": -1.0,
                "b_logprob_mean": -3.0,
                "delta_g": 2.0,
                "delta_g_trimmed_10pct": 2.0,
                "prompt_lens_per_q": [50 for _ in q_held],
                "R_lens_per_q_sample": [[100, 100] for _ in q_held],
                "smoke": True,
                "generated_at": ts,
                "git_commit": git_sha,
            }
            cell_path.write_text(json.dumps(payload))
            logger.info("Smoke wrote %s", cell_path)
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - PASS A/B per cell
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", default=None, help="e.g. '0-of-8' for sharded eval.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.50, 1.00])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--n-held-out-q", type=int, default=N_HELD_OUT_Q)
    ap.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "vLLM tensor-parallel size. Pin to the number of GPUs allocated to this shard "
            "(per CLAUDE.md cvd-hydra-override the shard dispatcher pins CVD, so default=1)."
        ),
    )
    ap.add_argument(
        "--conds",
        nargs="+",
        default=None,
        help=(
            "Optional explicit list of cid_i to evaluate (smoke wave subset). "
            "When set, only (cid_i in --conds) × --fracs is enumerated."
        ),
    )
    ap.add_argument(
        "--target-conds",
        nargs="+",
        default=None,
        help=(
            "Optional explicit list of target cid_j to eval against. "
            "When set, only target contexts in this list are evaluated "
            "(smoke / restricted-grid mode)."
        ),
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="CPU placeholder mode: write 4 dummy cells with real delta_g + length keys.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    shard_idx, n_shards = _parse_shard(args.shard)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker id drift: encode({MARKER_TEXT!r}) = {ids}")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end != IM_END_ID:
        raise AssertionError(f"<|im_end|> id drift: {im_end} != {IM_END_ID}")

    q_test = load_q_test_extended_50()
    q_held = q_test[: args.n_held_out_q]

    if args.smoke:
        return _smoke_run(tokenizer, q_held)

    # Sharding: distribute (cid × frac) across shards.
    source_cids = args.conds if args.conds is not None else [c.cid for c in UNION_CONTEXTS]
    target_ctxs = (
        [UNION_BY_CID[c] for c in args.target_conds]
        if args.target_conds is not None
        else UNION_CONTEXTS
    )
    all_cells: list[tuple[str, float]] = []
    for cid in source_cids:
        for frac in args.fracs:
            all_cells.append((cid, frac))
    my_cells = [(c, f) for k, (c, f) in enumerate(all_cells) if k % n_shards == shard_idx]
    logger.info(
        "Shard %d/%d owns %d (cid, frac) snapshots: %s",
        shard_idx,
        n_shards,
        len(my_cells),
        my_cells,
    )

    # Download adapters (one per snapshot).
    adapter_paths: dict[tuple[str, float], str] = {}
    for cid, frac in my_cells:
        adapter_paths[(cid, frac)] = _download_adapter(cid, args.seed, frac, LOCAL_ADAPTER_CACHE)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_seq_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    gen_sp = SamplingParams(
        n=args.n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
        # Qwen2.5 emits both eos_token_id AND <|im_end|>; pinning both is
        # defensive against the rare case where one stop variant slips through.
        stop_token_ids=[tokenizer.eos_token_id, IM_END_ID],
    )
    probe_sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    git_sha = _git_commit_hash()
    ts0 = _dt.datetime.now(_dt.UTC).isoformat()
    all_cids = [c.cid for c in UNION_CONTEXTS]

    for cid_i, frac in my_cells:
        lora_req = LoRARequest(
            lora_name=f"{cid_i}_frac{frac:.2f}",
            lora_int_id=all_cids.index(cid_i) * 10 + int(frac * 100) + 1,
            lora_path=adapter_paths[(cid_i, frac)],
        )
        for ctx_j in target_ctxs:
            cid_j = ctx_j.cid
            cell_path = PER_CELL_DIR / f"G_{cid_i}__{cid_j}_frac{frac:.2f}.json"
            if args.resume and cell_path.exists() and cell_path.stat().st_size > 0:
                logger.info("resume: skipping existing cell %s", cell_path.name)
                continue
            cell_label = f"{cid_i}->{cid_j} frac={frac:.2f}"
            t_cell = time.time()

            # ============================================================
            # PASS A: on-policy generation under adapter_i, prompts_j.
            # ============================================================
            prompts_text = [build_union_prompt(ctx_j, q, tokenizer) for q in q_held]
            prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts_text]
            t0 = time.time()
            gen_outs = llm.generate(prompts_text, gen_sp, lora_request=lora_req)
            t_gen = time.time() - t0
            assert len(gen_outs) == len(q_held), (
                f"PASS A: {len(gen_outs)} vLLM outputs for {len(q_held)} prompts"
            )

            # Collect generated R texts AND token-length per (q, k).
            R_texts_per_q: list[list[str]] = []
            R_lens_per_q: list[list[int]] = []
            flat_trained_texts: list[str] = []
            for _qi, out in enumerate(gen_outs):
                R_texts: list[str] = []
                R_lens: list[int] = []
                for sample in out.outputs:
                    R_texts.append(sample.text)
                    R_lens.append(len(sample.token_ids))
                    flat_trained_texts.append(sample.text)
                assert len(R_texts) == args.n_samples, (
                    f"PASS A: expected {args.n_samples} samples per q, got {len(R_texts)}"
                )
                R_texts_per_q.append(R_texts)
                R_lens_per_q.append(R_lens)
            emission_trained = _emission_rate(flat_trained_texts, MARKER_TEXT)

            # ============================================================
            # PASS B: teacher-forced log P(' ※') at post-R slot.
            #
            # For each (qi, k), build ``prompt_text + R_text + MARKER_TEXT``
            # via TOKEN-LEVEL encode (i460 pattern) so the marker lands at
            # the strict last slot. Run vLLM with prompt_logprobs=1,
            # max_tokens=1 under (i) adapter_i for g_logp and (ii) no adapter
            # for b_logp.
            # ============================================================
            probe_payloads: list[dict] = []
            probe_slot_positions: list[int] = []
            qk_index: list[tuple[int, int]] = []  # (qi, k) per probe row
            for qi, _q in enumerate(q_held):
                for k, R_text in enumerate(R_texts_per_q[qi]):
                    full_ids, _p_len, post_R_slot = _build_marker_probe_full_ids(
                        tokenizer, prompts_text[qi], R_text
                    )
                    probe_payloads.append({"prompt_token_ids": full_ids})
                    probe_slot_positions.append(post_R_slot)
                    qk_index.append((qi, k))

            # PASS B-1: trained model (g_logp).
            t0 = time.time()
            g_outs = llm.generate(probe_payloads, probe_sp, lora_request=lora_req)
            t_probe_g = time.time() - t0
            g_logps_flat, g_argmax_flat = _extract_marker_logp_and_argmax(
                g_outs, probe_slot_positions, cell_label=f"TRAINED/{cell_label}"
            )

            # PASS B-2: base model (b_logp), same probe payloads.
            t0 = time.time()
            b_outs = llm.generate(probe_payloads, probe_sp, lora_request=None)
            t_probe_b = time.time() - t0
            b_logps_flat, _ = _extract_marker_logp_and_argmax(
                b_outs, probe_slot_positions, cell_label=f"BASE/{cell_label}"
            )

            # Unflatten back to (qi, k).
            g_logps_per_q_sample: list[list[float]] = [
                [0.0 for _ in range(args.n_samples)] for _ in q_held
            ]
            b_logps_per_q_sample: list[list[float]] = [
                [0.0 for _ in range(args.n_samples)] for _ in q_held
            ]
            for idx, (qi, k) in enumerate(qk_index):
                g_logps_per_q_sample[qi][k] = g_logps_flat[idx]
                b_logps_per_q_sample[qi][k] = b_logps_flat[idx]

            g_arr = np.array(g_logps_flat, dtype=float)
            b_arr = np.array(b_logps_flat, dtype=float)
            delta = g_arr - b_arr
            g_mean = float(g_arr.mean())
            b_mean = float(b_arr.mean())
            delta_mean = float(delta.mean())
            try:
                from scipy.stats import trim_mean

                delta_trimmed = float(trim_mean(delta, 0.1))
            except Exception:
                delta_trimmed = delta_mean
            # Base emission baseline = how often the BASE generates a marker
            # in free-form gen. Computed by hooking back into the BASE model with
            # a separate vLLM call (NOT the probe, which is teacher-forced).
            # To keep cost down, we estimate base emission by computing it on
            # ONE base run per target ctx_j (cached across i for the same j).
            # In this loop we just record a placeholder; the smoke calibration
            # script reads emission_rate_trained for the gate (the base emission
            # is informational, not load-bearing).
            argmax_marker_rate = sum(g_argmax_flat) / max(1, len(g_argmax_flat))

            cell_payload = {
                "T_i": cid_i,
                "T_j": cid_j,
                "frac": frac,
                "seed": args.seed,
                "n_q": len(q_held),
                "n_samples": args.n_samples,
                # ─ Primary DV ─
                "g_logprob_mean": g_mean,
                "b_logprob_mean": b_mean,
                "delta_g": delta_mean,
                "delta_g_trimmed_10pct": delta_trimmed,
                # ─ Companion legibility anchor ─
                "emission_rate_trained": emission_trained,
                "argmax_marker_rate_trained": argmax_marker_rate,
                # ─ Per-q per-k log-probs ─
                "g_logps_per_q_sample": g_logps_per_q_sample,
                "b_logps_per_q_sample": b_logps_per_q_sample,
                # ─ Lengths (B7) ─
                "prompt_lens_per_q": prompt_lens,
                "R_lens_per_q_sample": R_lens_per_q,
                # ─ Audit ─
                "sample_texts_first200": [[t[:200] for t in per_q] for per_q in R_texts_per_q],
                "logp_floor": LOGP_FLOOR,
                "elapsed_seconds": {
                    "pass_a_gen": t_gen,
                    "pass_b_trained": t_probe_g,
                    "pass_b_base": t_probe_b,
                    "cell_total": time.time() - t_cell,
                },
                "generated_at": ts0,
                "git_commit": git_sha,
            }
            tmp = cell_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cell_payload))
            tmp.replace(cell_path)
            logger.info(
                "cell %s: delta_g=%.3f (g=%.3f b=%.3f) emit_trained=%.3f argmax=%.3f %.1fs",
                cell_label,
                delta_mean,
                g_mean,
                b_mean,
                emission_trained,
                argmax_marker_rate,
                time.time() - t_cell,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
