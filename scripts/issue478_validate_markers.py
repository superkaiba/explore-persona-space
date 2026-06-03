#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #478 PHASE 0b — OPTIONAL marker-set validator (arm only).

Per plan v5 §4.8 PHASE 0b + §4.9.1:

  * Assert each of the 8 distinct markers in ARM_MARKERS tokenizes to EXACTLY
    one token (with leading space) under Qwen-2.5-7B-Instruct.
  * Load BASE Qwen-2.5-7B-Instruct (one-time, then discard); compute mean base
    log P(marker_i) at the post-response slot over the REAL EVAL DISTRIBUTION:
    35 held-out personas × 20 eval questions = 700 prompts, using Phase 1's
    cached on-policy R for those personas (no extra generation; reuses cache).
  * Output: 8 × 35 base-logp matrix (marker × held-out persona, averaged over
    the 20 questions per persona), saved to
    ``data/issue_478/arm/marker_base_logp.json`` (analyzer diagnostic plotted
    alongside the §6.8 decomposition figure).
  * WARN if cross-marker spread (max − min) > 2 nats — swap the loudest
    offender from the §4.9.1 fallback pool.
  * FAIL if any marker's mean base logp on the held-out distribution > -3 —
    the token is too common to behave like ※ does in the core.

CLI:
  --gpu N             GPU index (default 0).
  --data-dir          Default: data/issue_478.
  --out-dir           Default: data/issue_478/arm.
  --gpu-mem-util      vLLM-style mem util for HF Transformers load (default 0.55).
  --skip-fail-gate    Skip the FAIL gate (for smoke / debug). WARNs still print.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue478_common import (  # noqa: E402
    ARM_BASE_LOGP_FAIL_THRESHOLD,
    ARM_BASE_LOGP_SPREAD_WARN,
    ARM_MARKERS,
    BASE_MODEL,
    HELD_OUT_35,
    MAX_LENGTH,
    assert_arm_marker_token_ids,
    load_all_persona_prompts,
)


def _import_questions() -> tuple[list[str], list[str]]:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_leakage_v3_onpolicy import DATA_QUESTIONS, EVAL_QUESTIONS

    return list(DATA_QUESTIONS), list(EVAL_QUESTIONS)


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_478"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_478" / "arm"),
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.55")),
    )
    parser.add_argument(
        "--skip-fail-gate",
        action="store_true",
        help="Skip the > -3 nats FAIL gate (still prints WARNs)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "marker_base_logp.json"

    # ── (1) Tokenizer assert — every marker is single-token ───────────
    from transformers import AutoTokenizer

    log.info("Loading Qwen-2.5-7B-Instruct tokenizer (marker single-token assertions) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    marker_text_to_id = assert_arm_marker_token_ids(tokenizer)
    log.info(
        "OK — all %d arm markers tokenize to single ids: %s", len(ARM_MARKERS), marker_text_to_id
    )

    # ── (2) Load cached on-policy R for the 35 held-out personas ──────
    all_prompts = load_all_persona_prompts()
    _train_qs, eval_questions = _import_questions()
    onpolicy_dir = Path(args.data_dir) / "onpolicy_R"

    held_out_R: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for p in HELD_OUT_35:
        cache_p = onpolicy_dir / f"{p}.json"
        if not cache_p.exists():
            missing.append(p)
            continue
        held_out_R[p] = json.loads(cache_p.read_text())["responses"]
    if missing:
        raise SystemExit(
            f"Phase 0b: cached on-policy R missing for {len(missing)} held-out personas: "
            f"{missing!r}. Run scripts/issue478_generate_onpolicy_R.py first."
        )
    log.info("OK — cached on-policy R loaded for all %d held-out personas", len(HELD_OUT_35))

    # ── (3) Build per-(persona, q) prefix tokens (the post-R slot) ────
    all_items: list[tuple[str, str, str]] = []  # (persona, q, full_prefix)
    for persona in HELD_OUT_35:
        sys_prompt = all_prompts[persona]
        qmap = held_out_R[persona]
        for q in eval_questions:
            if q not in qmap:
                raise RuntimeError(
                    f"Phase 0b: eval question {q[:40]!r} missing from cached R for {persona!r}. "
                    f"Re-run Phase 1 with --questions both."
                )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_prefix = prefix + qmap[q]  # post-R slot
            all_items.append((persona, q, full_prefix))

    log.info("Built %d (held-out persona, eval-q) prefixes", len(all_items))

    # ── (4) HF Transformers load + scoring ────────────────────────────
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    log.info(
        "Loading BASE model %s on GPU %d for post-R base log-prob scoring ...",
        BASE_MODEL,
        args.gpu,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    device = "cuda:0"

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # logp_per_persona_per_marker[persona][marker_text] = list[float] (one per q)
    logp_per_persona_per_marker: dict[str, dict[str, list[float]]] = {
        p: {m: [] for m in marker_text_to_id} for p in HELD_OUT_35
    }

    batch_size = 4
    for start in range(0, len(all_items), batch_size):
        chunk = all_items[start : start + batch_size]
        ids_list = [tokenizer.encode(t, add_special_tokens=False) for _, _, t in chunk]
        max_len = max(len(ids) for ids in ids_list)
        if max_len > MAX_LENGTH * 2:
            raise RuntimeError(
                f"Phase 0b: prefix length {max_len} exceeds {MAX_LENGTH * 2}; truncate cached R."
            )
        padded, attn = [], []
        for ids in ids_list:
            pad = max_len - len(ids)
            padded.append([pad_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        last_logits = logits[:, -1, :].float()
        log_probs = F.log_softmax(last_logits, dim=-1)
        ls = log_probs.cpu().tolist()
        for (persona, _q, _), full_ls in zip(chunk, ls, strict=True):
            for marker_text, marker_id in marker_text_to_id.items():
                logp_per_persona_per_marker[persona][marker_text].append(float(full_ls[marker_id]))
        del logits, last_logits, log_probs

    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        log.warning("torch.cuda.empty_cache() raised %s; continuing", e)

    # ── (5) Aggregate to 8 × 35 matrix; compute spread; emit JSON ─────
    # matrix[marker_text][persona] = mean over 20 questions
    matrix: dict[str, dict[str, float]] = {}
    per_marker_means_across_personas: dict[str, float] = {}
    for marker_text in marker_text_to_id:
        matrix[marker_text] = {}
        per_persona_means = []
        for persona in HELD_OUT_35:
            vals = logp_per_persona_per_marker[persona][marker_text]
            mean_v = sum(vals) / max(1, len(vals))
            matrix[marker_text][persona] = mean_v
            per_persona_means.append(mean_v)
        per_marker_means_across_personas[marker_text] = sum(per_persona_means) / max(
            1, len(per_persona_means)
        )

    # Spread = max − min across markers' across-persona means.
    means = list(per_marker_means_across_personas.values())
    spread = max(means) - min(means)

    # WARN if spread > 2 nats.
    warns: list[str] = []
    if spread > ARM_BASE_LOGP_SPREAD_WARN:
        warns.append(
            f"cross-marker base-logp spread {spread:.3f} nats > {ARM_BASE_LOGP_SPREAD_WARN}; "
            f"loudest offender = "
            f"{max(per_marker_means_across_personas, key=per_marker_means_across_personas.get)!r} "
            f"(mean logp = {max(means):.3f})"
        )

    # FAIL if any marker's mean base logp > -3.
    fails: list[str] = []
    for marker_text, mean_v in per_marker_means_across_personas.items():
        if mean_v > ARM_BASE_LOGP_FAIL_THRESHOLD:
            fails.append(
                f"marker {marker_text!r} mean base logp = {mean_v:.3f} > "
                f"{ARM_BASE_LOGP_FAIL_THRESHOLD} — too common to behave like ※"
            )

    payload = {
        "marker_text_to_id": marker_text_to_id,
        "held_out_personas": HELD_OUT_35,
        "eval_questions_count": len(eval_questions),
        "matrix_marker_x_persona_meanlogp": matrix,
        "per_marker_meanlogp_across_personas": per_marker_means_across_personas,
        "spread_nats": spread,
        "warns": warns,
        "fails": fails,
    }
    matrix_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", matrix_path)

    for w in warns:
        log.warning("[Phase 0b WARN] %s", w)
    if fails:
        for f in fails:
            log.error("[Phase 0b FAIL] %s", f)
        if not args.skip_fail_gate:
            raise SystemExit(2)
    log.info("Phase 0b done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
