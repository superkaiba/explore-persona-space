# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #501 Phase 4 — on-policy ΔG on the 24 (#489 source) × 12 (NEW MT/MN
target) cross-format cells.

Plan v2 §4.7 + §6.2. Reuses #489's marker-id-83399 ` ※` post-R-slot
teacher-forced logprob primitive byte-for-byte; only the prompt builder is
swapped from ``build_union_prompt`` to ``build_mt_prompt``.

Per (i, j) cell:
  - PASS A (vLLM, ON-POLICY): under adapter_i for #489 source cid_i,
    generate ``n_samples=8`` responses to ``prompt_{T_mt_j}(q)`` at
    temp=1.0 / top_p=1.0 / max_new_tokens=2048 / max_model_len read from
    Phase-0 recommendation (default 32768, escalates to 65536 only if
    Phase-0 sees a prefix exceeding 28000 tokens).
  - PASS B (vLLM, teacher-forced log P(' ※')): build ``prompt + R + ' ※'``
    via TOKEN-LEVEL encode; ask vLLM for ``prompt_logprobs[L][MARKER_ID]``
    at the post-R slot L; run under adapter_i (``g_logprob``) AND no
    adapter (``b_logprob``). ``ΔG = mean(g − b)``.

The target panel rotates across the 12 MT/MN contexts; one prompt is built
per (target_cid, conversation_index, probe_q) but ΔG is averaged over the 5
selected conversations × 20 probes × 8 samples per cell. This mirrors #489's
"50 probes × 8 samples" structure but with the conversation-index axis added.

Per-cell payload (under ``eval_results/issue_501/phase4/per_cell/``):
  - ``G_{cid_i}__{mt_cid}_frac{F:.2f}.json``:
      {T_i, T_mt, frac, seed, n_q, n_samples, n_samples_base,
       n_conversations,
       g_logprob_mean, b_logprob_mean, delta_g, delta_g_trimmed_10pct,
       emission_rate_trained, argmax_marker_rate_trained,
       emission_rate_base,
       g_logps_per_q_sample, b_logps_per_q_sample,
       prompt_lens_per_q, R_lens_per_q_sample,
       sample_texts_first200,
       generated_at, git_commit}

The ``emission_rate_base`` field is required by plan v2 §10 schema +
§7 gate-4 (`<5%`); computed via a base-model free-generation pass at
``n_samples_base = max(2, n_samples // 4)`` (~1/4 the trained-pass cost
since the base emission floor is the gate-4 threshold and small-sample
statistics suffice for a <5% verification).

Schema is byte-for-byte compatible with #489's ``G_*.json`` except ``T_j`` →
``T_mt``; the merge in Phase 5 unifies them under the common ``T_j`` key.

CLI:
    uv run python scripts/i501_phase4_eval_onpolicy.py --frac 0.50
    uv run python scripts/i501_phase4_eval_onpolicy.py --smoke
        # MT05 × {IK01, SP01} × 5 probes × 2 samples = 20 generations.
    uv run python scripts/i501_phase4_eval_onpolicy.py --shard 0-of-4 --frac 0.50
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import MARKER_ID, MARKER_TEXT
from explore_persona_space.experiments.i460_data import load_q_test_extended_50
from explore_persona_space.experiments.i501_mt_contexts import (
    MT_CONTEXTS,
    build_mt_prompt,
)

logger = logging.getLogger("i501.phase4")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE0_PREFIX = PROJECT_ROOT / "eval_results" / "issue_501" / "phase0" / "mt_prefixes.json"
PARENT_READY_PATH = PROJECT_ROOT / "eval_results" / "issue_501" / "phase0" / "parent_ready.json"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i501")
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase4"
PER_CELL_DIR = OUT_DIR / "per_cell"

LOGP_FLOOR = -50.0
N_SAMPLES = 8
N_HELD_OUT_Q = 20
MAX_NEW_TOKENS = 2048
IM_END_ID = 151645  # Qwen2.5 <|im_end|>

ALL_UNION_CIDS = tuple([f"IK{i:02d}" for i in range(1, 17)] + [f"SP{i:02d}" for i in range(1, 9)])


def _git_commit_hash() -> str:
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


def _resolve_frac(args) -> float:
    """Read the chosen frac F from Phase 0 parent-ready check (or CLI override)."""
    if args.frac is not None:
        return float(args.frac)
    if not PARENT_READY_PATH.exists():
        raise RuntimeError(
            f"Phase 0 parent-ready artifact missing ({PARENT_READY_PATH}); "
            "run i501_phase0_parent_ready_check.py first or pass --frac explicitly"
        )
    payload = json.loads(PARENT_READY_PATH.read_text())
    return float(payload["frac"])


def _load_phase0_payload() -> dict:
    if not PHASE0_PREFIX.exists():
        raise RuntimeError(
            f"Phase 0 prerequisite missing: {PHASE0_PREFIX}. Run i501_phase0_load_corpora.py first."
        )
    return json.loads(PHASE0_PREFIX.read_text())


def _download_adapter(cid: str, seed: int, frac: float, cache_dir: Path) -> str:
    """HF-download the per-(cid, seed, frac) #489 adapter; returns local path.

    Mirrors #489's ``_download_adapter`` byte-for-byte. We pull #489's
    adapters at the chosen frac.
    """
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
    """Build ``encode(prompt + R + MARKER)`` token list; return
    (full_ids, prompt_len, post_R_slot).

    The teacher-forced operation appends a single ` ※` marker token at the
    END of ``prompt + R_text`` and reads ``log P(MARKER_ID)`` at the
    last-token slot. The ONLY invariant we need is that the appended
    marker IS the last token (so ``prompt_logprobs[-1]`` reads OUR slot).

    Earlier marker occurrences inside ``R_text`` are EXPECTED on high-
    emission cells (a trained model that already emits ` ※` will produce
    R_text containing the marker). Counting global occurrences and
    rejecting count != 1 (the round-4 invariant) would crash exactly on
    the cells the experiment most needs to measure — the cells where the
    marker has transferred strongly into the on-policy response.

    Note: the prompt-history is independently guarded
    (``assert_no_marker_in_history`` upstream), so any non-last
    occurrences of ``MARKER_ID`` in ``full_ids`` come from ``R_text``,
    which is the model's own generation — exactly what we want to
    measure transfer of.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    if full_ids[-1] != MARKER_ID:
        raise RuntimeError(
            f"marker slot drift: full_ids[-1]={full_ids[-1]} "
            f"(expected last=={MARKER_ID}); R_text tail may have merged with the marker token "
            "under BPE; refusing to mis-attribute the post-R logprob"
        )
    return full_ids, len(prompt_ids), len(full_ids) - 1


def _extract_marker_logp_and_argmax(
    outputs, slot_positions: list[int], cell_label: str
) -> tuple[list[float], list[bool]]:
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


def _run_cell_eval(
    *,
    llm,
    tokenizer,
    gen_sp,
    probe_sp,
    base_gen_sp,
    cid_i: str,
    mt_cid: str,
    frac: float,
    seed: int,
    n_samples: int,
    n_samples_base: int,
    entries: list[tuple[int, int, str]],
    n_conversations: int,
    lora_req,
    cell_path: Path,
    git_sha: str,
    ts0: str,
    extra_meta: dict | None = None,
) -> dict:
    """PASS A (on-policy generation under adapter_i) + PASS A' (on-policy
    base-model generation, NO adapter) + PASS B (teacher-forced
    log P(' ※') trained − base) on a single (cid_i, mt_cid) cell.

    ``entries`` is the cached list ``[(conv_idx, probe_qi, prompt_text), ...]``
    for the target context. ``lora_req`` is the vLLM ``LoRARequest`` for
    ``cid_i`` at ``frac``. ``base_gen_sp`` is a ``SamplingParams`` with
    ``n=n_samples_base`` (typically 1/4 of ``n_samples``, since
    ``emission_rate_base`` is gated at <5% per plan v2 §7 gate-4 and small
    samples suffice to verify the floor). PASS A' adds <30% wall to the
    cell vs PASS A.

    Writes the per-cell payload to ``cell_path`` atomically (``.json.tmp``
    → replace) and returns the in-memory payload.

    Shared between the production sweep and the smoke path so they execute
    the SAME code (plan v2 §4.5 unification). The smoke entry point passes
    tiny ``entries`` / ``n_samples`` / ``n_samples_base`` to keep wall
    bounded.
    """
    cell_label = f"{cid_i}->{mt_cid} frac={frac:.2f}"
    t_cell = time.time()

    # PASS A: on-policy generation under adapter_i.
    prompts_text = [p for (_ci, _qi, p) in entries]
    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts_text]
    t0 = time.time()
    gen_outs = llm.generate(prompts_text, gen_sp, lora_request=lora_req)
    t_gen = time.time() - t0
    assert len(gen_outs) == len(entries), (
        f"PASS A: {len(gen_outs)} vLLM outputs for {len(entries)} prompts"
    )

    R_texts_per_row: list[list[str]] = []
    R_lens_per_row: list[list[int]] = []
    flat_trained_texts: list[str] = []
    for out in gen_outs:
        R_texts: list[str] = []
        R_lens: list[int] = []
        for sample in out.outputs:
            R_texts.append(sample.text)
            R_lens.append(len(sample.token_ids))
            flat_trained_texts.append(sample.text)
        assert len(R_texts) == n_samples, (
            f"PASS A: expected {n_samples} samples per row, got {len(R_texts)}"
        )
        R_texts_per_row.append(R_texts)
        R_lens_per_row.append(R_lens)
    emission_trained = _emission_rate(flat_trained_texts, MARKER_TEXT)

    # PASS A': on-policy base-model generation (NO adapter) — required for
    # the plan v2 §10 ``emission_rate_base`` schema field + §7 gate-4
    # (<5% base-model marker emission). Uses n_samples_base (typically
    # n_samples // 4) since the gate is a small-N floor check.
    t0 = time.time()
    base_gen_outs = llm.generate(prompts_text, base_gen_sp, lora_request=None)
    t_base_gen = time.time() - t0
    assert len(base_gen_outs) == len(entries), (
        f"PASS A': {len(base_gen_outs)} vLLM outputs for {len(entries)} prompts"
    )
    flat_base_texts: list[str] = []
    for out in base_gen_outs:
        for sample in out.outputs:
            flat_base_texts.append(sample.text)
        assert len(out.outputs) == n_samples_base, (
            f"PASS A': expected {n_samples_base} samples per row, got {len(out.outputs)}"
        )
    emission_base = _emission_rate(flat_base_texts, MARKER_TEXT)

    # PASS B: teacher-forced log P(' ※') at post-R slot.
    probe_payloads: list[dict] = []
    probe_slot_positions: list[int] = []
    row_index: list[tuple[int, int]] = []
    for row_idx, (_ci, _qi, prompt) in enumerate(entries):
        for k, R_text in enumerate(R_texts_per_row[row_idx]):
            full_ids, _p_len, post_R_slot = _build_marker_probe_full_ids(tokenizer, prompt, R_text)
            probe_payloads.append({"prompt_token_ids": full_ids})
            probe_slot_positions.append(post_R_slot)
            row_index.append((row_idx, k))

    t0 = time.time()
    g_outs = llm.generate(probe_payloads, probe_sp, lora_request=lora_req)
    t_probe_g = time.time() - t0
    g_logps_flat, g_argmax_flat = _extract_marker_logp_and_argmax(
        g_outs, probe_slot_positions, cell_label=f"TRAINED/{cell_label}"
    )

    t0 = time.time()
    b_outs = llm.generate(probe_payloads, probe_sp, lora_request=None)
    t_probe_b = time.time() - t0
    b_logps_flat, _ = _extract_marker_logp_and_argmax(
        b_outs, probe_slot_positions, cell_label=f"BASE/{cell_label}"
    )

    n_rows = len(entries)
    g_logps_per_row_sample: list[list[float]] = [
        [0.0 for _ in range(n_samples)] for _ in range(n_rows)
    ]
    b_logps_per_row_sample: list[list[float]] = [
        [0.0 for _ in range(n_samples)] for _ in range(n_rows)
    ]
    for idx, (ri, k) in enumerate(row_index):
        g_logps_per_row_sample[ri][k] = g_logps_flat[idx]
        b_logps_per_row_sample[ri][k] = b_logps_flat[idx]

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
    argmax_marker_rate = sum(g_argmax_flat) / max(1, len(g_argmax_flat))

    cell_payload = {
        "T_i": cid_i,
        "T_mt": mt_cid,
        "frac": frac,
        "seed": seed,
        "n_q": len(entries) // max(1, n_conversations),
        "n_samples": n_samples,
        "n_samples_base": n_samples_base,
        "n_conversations": n_conversations,
        # Primary DV
        "g_logprob_mean": g_mean,
        "b_logprob_mean": b_mean,
        "delta_g": delta_mean,
        "delta_g_trimmed_10pct": delta_trimmed,
        # Companion legibility anchors
        "emission_rate_trained": emission_trained,
        "argmax_marker_rate_trained": argmax_marker_rate,
        # Gate-4 (plan v2 §7) requirement: base-model emission floor.
        "emission_rate_base": emission_base,
        # Per-row per-k log-probs
        "g_logps_per_q_sample": g_logps_per_row_sample,
        "b_logps_per_q_sample": b_logps_per_row_sample,
        # Lengths
        "prompt_lens_per_q": prompt_lens,
        "R_lens_per_q_sample": R_lens_per_row,
        # Audit
        "sample_texts_first200": [[t[:200] for t in per_row] for per_row in R_texts_per_row],
        "logp_floor": LOGP_FLOOR,
        "elapsed_seconds": {
            "pass_a_gen": t_gen,
            "pass_a_base_gen": t_base_gen,
            "pass_b_trained": t_probe_g,
            "pass_b_base": t_probe_b,
            "cell_total": time.time() - t_cell,
        },
        "generated_at": ts0,
        "git_commit": git_sha,
    }
    if extra_meta:
        cell_payload.update(extra_meta)

    tmp = cell_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cell_payload))
    tmp.replace(cell_path)
    logger.info(
        "cell %s: delta_g=%.3f (g=%.3f b=%.3f) emit_trained=%.3f emit_base=%.3f argmax=%.3f %.1fs",
        cell_label,
        delta_mean,
        g_mean,
        b_mean,
        emission_trained,
        emission_base,
        argmax_marker_rate,
        time.time() - t_cell,
    )
    return cell_payload


def _smoke_run(tokenizer, q_held: list[str], phase0: dict, frac: float, seed: int) -> int:
    """End-to-end tiny smoke (plan v2 §4.5 unification): 2 #489 source cids
    (IK01, SP01) × 1 MT target (MT05) × 2 held-out Q × 2 samples = 8 generations
    + 16 teacher-forced logprob reads. Real vLLM init, real LoRA load from HF
    Hub, real on-policy generation, real teacher-forced marker logprob — same
    helpers + same per-cell JSON shape the production sweep uses.

    Requires GPU + HF Hub access; the VM has neither, so this must run on a
    pod (typically via ``bash scripts/i501_run_all.sh --smoke``).
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    smoke_anchors = ["IK01", "SP01"]
    smoke_target = "MT05"
    n_samples = 2
    n_samples_base = max(2, n_samples // 4)  # smoke: floor at 2 = same as n_samples
    n_held_smoke = 2

    if smoke_target not in phase0["per_cid"]:
        raise RuntimeError(
            f"smoke needs phase0 to have built {smoke_target}; "
            "run i501_phase0_load_corpora.py --smoke first"
        )
    rows = phase0["per_cid"][smoke_target].get("rows", [])
    if not rows:
        raise RuntimeError(f"smoke target {smoke_target} has no phase0 rows")

    q_held_smoke = q_held[:n_held_smoke]
    # Use 1 conversation × n_held_smoke probes = n_held_smoke rows per cell.
    smoke_rows = rows[:1]
    entries: list[tuple[int, int, str]] = []
    for ci, row in enumerate(smoke_rows):
        history = tuple(row["history"])
        for qi, q in enumerate(q_held_smoke):
            prompt = build_mt_prompt(history, q, tokenizer)
            entries.append((ci, qi, prompt))
    n_conversations = len(smoke_rows)
    logger.info(
        "Smoke: %d source cids × %s × %d entries (conv=%d × probes=%d) × %d samples",
        len(smoke_anchors),
        smoke_target,
        len(entries),
        n_conversations,
        n_held_smoke,
        n_samples,
    )

    max_model_len = int(phase0.get("max_model_len_recommendation", 32768))
    logger.info("Smoke: max_model_len=%d", max_model_len)

    # Pull #489 adapters at this frac for the 2 smoke anchors.
    adapter_paths: dict[str, str] = {}
    for cid in smoke_anchors:
        adapter_paths[cid] = _download_adapter(cid, seed, frac, LOCAL_ADAPTER_CACHE)

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.75,
        seed=42,
        max_model_len=max_model_len,
        max_num_batched_tokens=8192,
        tensor_parallel_size=1,
    )
    gen_sp = SamplingParams(
        n=n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
        stop_token_ids=[tokenizer.eos_token_id, IM_END_ID],
    )
    base_gen_sp = SamplingParams(
        n=n_samples_base,
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
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

    for source_idx, cid_i in enumerate(smoke_anchors):
        lora_req = LoRARequest(
            lora_name=f"{cid_i}_frac{frac:.2f}",
            lora_int_id=source_idx + 1,
            lora_path=adapter_paths[cid_i],
        )
        cell_path = PER_CELL_DIR / f"G_{cid_i}__{smoke_target}_frac{frac:.2f}.json"
        _run_cell_eval(
            llm=llm,
            tokenizer=tokenizer,
            gen_sp=gen_sp,
            probe_sp=probe_sp,
            base_gen_sp=base_gen_sp,
            cid_i=cid_i,
            mt_cid=smoke_target,
            frac=frac,
            seed=seed,
            n_samples=n_samples,
            n_samples_base=n_samples_base,
            entries=entries,
            n_conversations=n_conversations,
            lora_req=lora_req,
            cell_path=cell_path,
            git_sha=git_sha,
            ts0=ts0,
            extra_meta={"smoke": True},
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", default=None, help="e.g. '0-of-4' for sharded eval across sources.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--frac",
        type=float,
        default=None,
        help="Adapter frac F. Defaults to the value in parent_ready.json.",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override the max_model_len read from phase0 (default: phase0 recommendation).",
    )
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--n-held-out-q", type=int, default=N_HELD_OUT_Q)
    ap.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor-parallel size. Pin to GPUs allocated to this shard.",
    )
    ap.add_argument(
        "--source-conds",
        nargs="+",
        default=None,
        help="Optional explicit list of source cid_i to evaluate (smoke/restricted-grid).",
    )
    ap.add_argument(
        "--target-conds",
        nargs="+",
        default=None,
        help="Optional explicit list of target mt_cid to eval against (smoke/restricted-grid).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "End-to-end tiny mode: IK01+SP01 × MT05 × 2 probes × 2 samples = 8 generations "
            "+ 16 teacher-forced reads, real vLLM + LoRA. Requires GPU + HF Hub access."
        ),
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

    phase0 = _load_phase0_payload()
    frac = _resolve_frac(args)
    q_test = load_q_test_extended_50()
    q_held = q_test[: args.n_held_out_q]

    if args.smoke:
        return _smoke_run(tokenizer, q_held, phase0, frac, args.seed)

    # Sharding: distribute source cids across shards.
    source_cids = list(args.source_conds) if args.source_conds is not None else list(ALL_UNION_CIDS)
    target_mt_cids = (
        list(args.target_conds) if args.target_conds is not None else [c.cid for c in MT_CONTEXTS]
    )
    my_sources = [c for k, c in enumerate(source_cids) if k % n_shards == shard_idx]
    logger.info(
        "Shard %d/%d owns %d source cids (frac=%.2f) over %d MT targets",
        shard_idx,
        n_shards,
        len(my_sources),
        frac,
        len(target_mt_cids),
    )

    # Pre-resolve max_model_len from phase 0.
    max_model_len = args.max_model_len or int(phase0.get("max_model_len_recommendation", 32768))
    logger.info("Phase 4: using max_model_len=%d", max_model_len)

    # Download #489 adapters (one per source cid at the chosen frac).
    adapter_paths: dict[str, str] = {}
    for cid in my_sources:
        adapter_paths[cid] = _download_adapter(cid, args.seed, frac, LOCAL_ADAPTER_CACHE)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.75,
        seed=42,
        max_model_len=max_model_len,
        # Cap batched-token count so vLLM never schedules a step large
        # enough to allocate prompt_logprobs > 8GiB at fp32 (2026-06-06
        # OOM 2). Computed bound: 8192 slots * 151936 vocab * 4 bytes
        # = ~5 GiB peak, well under any reasonable free headroom.
        max_num_batched_tokens=8192,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    gen_sp = SamplingParams(
        n=args.n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
        stop_token_ids=[tokenizer.eos_token_id, IM_END_ID],
    )
    # PASS A' base-model emission: 1/4 the samples (floor 2). Gate-4 wants
    # <5% so a small-N floor check suffices; this caps the extra wall to
    # ~25% of PASS A rather than doubling it.
    n_samples_base = max(2, args.n_samples // 4)
    base_gen_sp = SamplingParams(
        n=n_samples_base,
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
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

    # Build prompts per target_mt: one prompt per (conversation, probe q).
    # The mean over conversations × probes is the per-cell ΔG.
    per_cid_payload = phase0["per_cid"]

    # Defensively cache MT prompt builds per (target_cid, conv_idx, qi) so we
    # don't rebuild them for every source. The list lives in memory; on a 1-GPU
    # pod with 60 cells this is ~60 KB of strings.
    cached_prompts_per_target: dict[str, list[tuple[int, int, str]]] = {}
    for mt_cid in target_mt_cids:
        rows = per_cid_payload.get(mt_cid, {}).get("rows", [])
        if not rows:
            raise RuntimeError(f"target {mt_cid} has no phase0 rows")
        bucket: list[tuple[int, int, str]] = []
        for ci, row in enumerate(rows):
            history = tuple(row["history"])
            for qi, q in enumerate(q_held):
                prompt = build_mt_prompt(history, q, tokenizer)
                bucket.append((ci, qi, prompt))
        cached_prompts_per_target[mt_cid] = bucket
        logger.info("Cached %d (conv,probe) prompts for target %s", len(bucket), mt_cid)

    for source_idx, cid_i in enumerate(my_sources):
        lora_req = LoRARequest(
            lora_name=f"{cid_i}_frac{frac:.2f}",
            lora_int_id=source_idx + 1,
            lora_path=adapter_paths[cid_i],
        )
        for mt_cid in target_mt_cids:
            cell_path = PER_CELL_DIR / f"G_{cid_i}__{mt_cid}_frac{frac:.2f}.json"
            if args.resume and cell_path.exists() and cell_path.stat().st_size > 0:
                logger.info("resume: skipping existing cell %s", cell_path.name)
                continue
            entries = cached_prompts_per_target[mt_cid]
            n_conversations = len(per_cid_payload[mt_cid]["rows"])
            _run_cell_eval(
                llm=llm,
                tokenizer=tokenizer,
                gen_sp=gen_sp,
                probe_sp=probe_sp,
                base_gen_sp=base_gen_sp,
                cid_i=cid_i,
                mt_cid=mt_cid,
                frac=frac,
                seed=args.seed,
                n_samples=args.n_samples,
                n_samples_base=n_samples_base,
                entries=entries,
                n_conversations=n_conversations,
                lora_req=lora_req,
                cell_path=cell_path,
                git_sha=git_sha,
                ts0=ts0,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
