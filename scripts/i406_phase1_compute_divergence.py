"""Phase 1 — sharded teacher-forced forward passes + residual-stream activation capture.

Issue #406 plan v9 §4 Phase 1.

For each (T_i, q') in the shard's slice (split by q_idx % n_shards), this
script:
  1. Greedy-decodes a reference completion from the CANONICAL bare-question
     C1 prompt for q' (max_new_tokens=32, K_available = min(25, len(ref))).
  2. For each of the 20 conditions T_i, builds the literal prompt by
     condition shape, concatenates by TOKEN ID (NOT decode→encode round
     trip — MF-1 fix), and runs ONE teacher-forced HF forward pass.
  3. Extracts per-position log-softmax at the K positions that predict
     ref_tokens[0..K-1] AND captures residual-stream activations at the
     LAST PROMPT TOKEN position on layers L ∈ {0, 5, 11, 15, 21, 27}.
  4. Writes per-q' shard output to
     eval_results/issue_406/divergence/per_q/q_{q_idx:03d}.pt
     (atomic per-file; resume-safe across shards).

Pre-flight on the first 5 probes (canonical reference decode): verifies
≥4 reach K=25; if not, bumps max_new_tokens to 48 and retries the
decode on those probes (Knob 8c row 24).

CLI:
    uv run python scripts/i406_phase1_compute_divergence.py --gpu-shard 0-of-2
    uv run python scripts/i406_phase1_compute_divergence.py --gpu-shard 1-of-2
    # Single-process fallback:
    uv run python scripts/i406_phase1_compute_divergence.py
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i406.phase1")

# ── Constants ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYERS = [0, 5, 11, 15, 21, 27]  # 6 layers (Knob §0 / §3.5b)
K_TARGET = 25  # v9: bumped from 10 to 25
MAX_NEW_TOKENS_DECODE = 32  # 8-token buffer over K=25
MAX_NEW_TOKENS_DECODE_FALLBACK = 48  # used if pre-flight fails
HIDDEN_DIM = 3584  # Qwen-2.5-7B
N_LAYERS = 28
PRE_FLIGHT_N_PROBES = 5  # first 5 probes used to check K availability
PRE_FLIGHT_MIN_HITS = 4  # require ≥4 of 5 to reach K=25

OUT_DIR = Path("eval_results/issue_406/divergence/per_q")


def _parse_shard(spec: str | None) -> tuple[int, int]:
    """Parse '0-of-2' / '1-of-2' / None into (shard_idx, n_shards).

    MF-5 fix: use `.split("-of-")` (NOT the v7 buggy `.replace("-of-", " ").split()`
    which produces a 2-element list unpacked into 3 vars).
    """
    if spec is None:
        return 0, 1
    s_idx, n = spec.split("-of-")
    s_idx_i = int(s_idx)
    n_i = int(n)
    if not (0 <= s_idx_i < n_i):
        raise ValueError(f"--gpu-shard {spec!r}: shard index {s_idx_i} not in [0, {n_i})")
    return s_idx_i, n_i


def _greedy_decode_canonical_ref(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int,
) -> list[int]:
    """Greedy-decode `max_new_tokens` from the bare-question C1 chat-template prompt.

    Returns the list of generated token IDs (NOT decoded string). EOS in the
    generated tail is stripped from the trailing edge.
    """
    # Canonical = C1: chat-template with user-only turn, no system prompt.
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], device=model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, len(prompt_ids) :].tolist()
    # Strip trailing EOS if present (don't strip embedded EOS within window)
    while gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
        gen_ids.pop()
    return gen_ids


def _load_class_d_rewrites() -> dict[str, dict[str, str]]:
    """Load the Class D rewrites cache produced by Phase 0."""
    import json

    path = Path("data/issue_406/class_d/rewrites_v1.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Class D rewrites not found at {path}. Run scripts/i406_phase0_generate_data.py first."
        )
    with open(path) as f:
        return json.load(f)


def _load_q_test() -> list[str]:
    """Load the merged 50-question Q_test produced by Phase 0."""
    import json

    path = Path("data/issue_406/q_test_extended_50.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Q_test extended-50 not found at {path}. "
            "Run scripts/i406_phase0_generate_data.py first."
        )
    with open(path) as f:
        payload = json.load(f)
    qs = payload["questions"]
    if len(qs) != 50:
        raise AssertionError(f"Expected 50 Q_test, got {len(qs)} in {path}")
    return qs


def _per_q_path(q_idx: int) -> Path:
    return OUT_DIR / f"q_{q_idx:03d}.pt"


def _capture_layer_hook(layer_idx: int, captured: dict, prompt_len: int):
    """Build a forward-hook that grabs the residual-stream activation at the
    last prompt-token position (index prompt_len - 1) for layer `layer_idx`.
    """

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        # hs shape: (1, seq_len, hidden_dim); grab last prompt token
        captured[layer_idx] = hs[0, prompt_len - 1, :].detach().float().cpu()

    return hook


def _process_one_q(
    q_idx: int,
    question: str,
    model,
    tokenizer,
    class_d_rewrites: dict,
    max_new_tokens_decode: int,
) -> dict:
    """Process one q': greedy reference + per-condition teacher-force.

    Returns a dict ready to torch.save() at the per-q' file. Shape:
        {
          "q_idx": int,
          "question": str,
          "ref_tokens": list[int],         # K_available ≤ K_target tokens
          "k_available": int,
          "prompt_ids": dict[cid, list[int]],
          "log_probs": dict[cid, Tensor],  # (K_available, vocab_size), float32, CPU
          "activations": dict[cid, dict[int, Tensor]],
                                            # cid -> layer_idx -> (hidden_dim,) float32 CPU
        }
    """
    # 1. Greedy-decode the canonical (C1) reference for this question.
    ref_tokens = _greedy_decode_canonical_ref(
        model, tokenizer, question, max_new_tokens=max_new_tokens_decode
    )
    k_available = min(K_TARGET, len(ref_tokens))
    if k_available < K_TARGET:
        logger.warning(
            "q_idx=%d ref_tokens length %d < K=%d; K_available=%d",
            q_idx,
            len(ref_tokens),
            K_TARGET,
            k_available,
        )
    ref_tokens = ref_tokens[:k_available]

    log_probs_per_cid: dict[str, torch.Tensor] = {}
    activations_per_cid: dict[str, dict[int, torch.Tensor]] = {}
    prompt_ids_per_cid: dict[str, list[int]] = {}

    for cond in CONDITIONS:
        prompt_text = build_prompt_for_condition(
            cond, question, tokenizer, class_d_rewrites=class_d_rewrites
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = prompt_ids + ref_tokens  # TOKEN-ID concat (MF-1 fix)

        # Register hooks for this (cond, q') only — depend on prompt_len.
        captured: dict[int, torch.Tensor] = {}
        hooks = [
            model.model.layers[L].register_forward_hook(
                _capture_layer_hook(L, captured, len(prompt_ids))
            )
            for L in TARGET_LAYERS
        ]
        try:
            input_ids = torch.tensor([full_ids], device=model.device)
            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=False)
            logits = out.logits  # (1, len(full_ids), V)
            # logit at position (len(prompt_ids) - 1) predicts ref_tokens[0].
            response_logits = logits[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + k_available, :]
            assert response_logits.shape[0] == k_available, response_logits.shape
            log_probs = F.log_softmax(response_logits.float(), dim=-1)
            assert log_probs.shape == (
                k_available,
                tokenizer.vocab_size,
            ), log_probs.shape
        finally:
            for h in hooks:
                h.remove()

        # Alignment assert (MF-1 defense in depth) — first cond per q'.
        if cond.cid == CONDITIONS[0].cid:
            scored = full_ids[len(prompt_ids) : len(prompt_ids) + k_available]
            assert scored == ref_tokens, (
                f"Phase 1 alignment broken on cond={cond.cid} q_idx={q_idx}: "
                f"full_ids[prompt_len:prompt_len+K_av] = {scored} != "
                f"ref_tokens = {ref_tokens}. BPE re-segmentation suspected."
            )

        # Verify hook fired on every target layer.
        missing = [L for L in TARGET_LAYERS if L not in captured]
        if missing:
            raise AssertionError(f"Hook missed layers {missing} on cond={cond.cid} q_idx={q_idx}")

        log_probs_per_cid[cond.cid] = log_probs.cpu()
        activations_per_cid[cond.cid] = {L: captured[L] for L in TARGET_LAYERS}
        prompt_ids_per_cid[cond.cid] = prompt_ids

    return {
        "q_idx": q_idx,
        "question": question,
        "ref_tokens": ref_tokens,
        "k_available": k_available,
        "prompt_ids": prompt_ids_per_cid,
        "log_probs": log_probs_per_cid,
        "activations": activations_per_cid,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--gpu-shard",
        default=None,
        help="e.g. '0-of-2' or '1-of-2'; omit for single-process all-q.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip q' indices whose per_q file already exists with non-zero size.",
    )
    args = ap.parse_args(argv)

    shard_idx, n_shards = _parse_shard(args.gpu_shard)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load environment (HF_TOKEN, HF_HOME, etc.)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # Sanity: marker token id matches CLAUDE.md spec.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [MARKER_ID], (
        f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]. "
        "Refusing to launch with marker drift."
    )
    logger.info("Marker tokenization OK: %r -> [%d]", MARKER_TEXT, MARKER_ID)

    # Class D rewrites + Q_test required.
    class_d_rewrites = _load_class_d_rewrites()
    q_test = _load_q_test()
    logger.info(
        "Loaded %d Q_test questions, %d Class D rewrites", len(q_test), len(class_d_rewrites)
    )

    # Decide which q' indices this shard owns.
    my_q_idxs = [i for i in range(len(q_test)) if i % n_shards == shard_idx]
    logger.info(
        "Shard %d/%d owns %d q' indices: %s",
        shard_idx,
        n_shards,
        len(my_q_idxs),
        my_q_idxs,
    )

    # Filter for resume.
    if args.resume:
        pending = [
            i
            for i in my_q_idxs
            if not _per_q_path(i).exists() or _per_q_path(i).stat().st_size == 0
        ]
        logger.info(
            "Resume mode: %d of %d q' indices remain (already-done are skipped).",
            len(pending),
            len(my_q_idxs),
        )
        my_q_idxs = pending

    if not my_q_idxs:
        logger.info("Nothing to do; exiting.")
        return

    # Load base model.
    logger.info("Loading base model %s ...", BASE_MODEL)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    # Block-count sanity (Qwen-2.5-7B has 28 transformer blocks).
    n_blocks = len(model.model.layers)
    if n_blocks != N_LAYERS:
        raise AssertionError(
            f"Expected {N_LAYERS} transformer blocks on {BASE_MODEL}, got {n_blocks}. "
            f"Target layers {TARGET_LAYERS} may be out of range."
        )
    logger.info("Loaded base model in %.1fs (n_blocks=%d)", time.time() - t0, n_blocks)

    # Pre-flight: first PRE_FLIGHT_N_PROBES of MY shard's q' get a decode
    # at default max_new_tokens. If <PRE_FLIGHT_MIN_HITS hit K_TARGET, bump.
    max_new_tokens_decode = MAX_NEW_TOKENS_DECODE
    pre_flight_subset = my_q_idxs[:PRE_FLIGHT_N_PROBES]
    pre_flight_lens: list[int] = []
    for q_idx in pre_flight_subset:
        ref = _greedy_decode_canonical_ref(
            model, tokenizer, q_test[q_idx], max_new_tokens=max_new_tokens_decode
        )
        pre_flight_lens.append(len(ref))
    n_hit = sum(1 for L in pre_flight_lens if L >= K_TARGET)
    logger.info(
        "Pre-flight: %d of %d probes hit K=%d at max_new=%d (lens=%s)",
        n_hit,
        len(pre_flight_subset),
        K_TARGET,
        max_new_tokens_decode,
        pre_flight_lens,
    )
    if n_hit < PRE_FLIGHT_MIN_HITS and pre_flight_subset:
        logger.warning(
            "Pre-flight failed; bumping max_new_tokens_decode %d -> %d for all probes.",
            max_new_tokens_decode,
            MAX_NEW_TOKENS_DECODE_FALLBACK,
        )
        max_new_tokens_decode = MAX_NEW_TOKENS_DECODE_FALLBACK

    # Main loop: per-q' compute + per-q' atomic save.
    for q_idx in my_q_idxs:
        t1 = time.time()
        payload = _process_one_q(
            q_idx,
            q_test[q_idx],
            model,
            tokenizer,
            class_d_rewrites,
            max_new_tokens_decode=max_new_tokens_decode,
        )
        # Atomic write.
        out_path = _per_q_path(q_idx)
        tmp_path = out_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(out_path)
        logger.info(
            "shard=%d q_idx=%d done in %.1fs K_available=%d -> %s",
            shard_idx,
            q_idx,
            time.time() - t1,
            payload["k_available"],
            out_path,
        )

    logger.info("Shard %d done; %d q' processed.", shard_idx, len(my_q_idxs))


if __name__ == "__main__":
    main()
