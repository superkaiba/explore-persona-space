"""Phase 2 (smoke) + Phase 3 (sweep) -- train ONE LoRA per #471 condition.

Plan v1 §4.2 + §4.6 + §4.7.

Per condition C ∈ {cond1, cond2_k0, cond2_k1, cond2_k3}:
  * Inherit R_villain / Q_demo / Q_train / Q_test from #465 (HF fallback).
  * Build 300 POSITIVE rows (byte-identical to #465; reuses
    `i465_prompts.build_training_messages`).
  * Build 300 NEGATIVE rows (NEW): 100 per negative persona (default /
    medical_doctor / police_officer), same Q_train rotated; each row's
    completion = base-Qwen greedy R under THAT negative's own system prompt
    on the same q (from R_negatives.json), NO trailing marker. cond2_k1/k3
    negative rows use MARKER-STRIPPED demos (so the row's input_ids contain
    ZERO markers -> MarkerOnlyDataCollator's "no marker -> EOS only" branch
    fires).
  * Interleave positives + negatives (round-robin) and shuffle deterministically.
  * Write JSONL rows under data/issue_471/train_rows/<cond>.jsonl.
  * Tokenization sanity: per row, assert marker_count == (k+1) for positives
    (with k = CONDITION_K[cond]) AND marker_count == 0 for negatives.
  * train_lora with TrainLoraConfig (same recipe as #465: marker_only_loss=True,
    tail_tokens=0, lr=1e-5, lora_r=32 / alpha=64, 5 epochs, batch=4 grad_accum=4
    max_length=2048 seed=42). Adapter uploads to
    superkaiba1/explore-persona-space/adapters/i471_<cond>.
  * MarkerLogprobKLTrajectoryCallback active every 10 steps:
    teacher-forced probe at 2 shapes (in_trained_shape + demo_free_default
    with helpful-R) per condition, recording mean_logp_marker + emission_rate
    + mean_kl_from_base. 10 prompts per shape.

Smoke == sweep: this script with --cond cond1 IS the smoke step. The
dispatcher `i471_phase23_dispatch.sh` runs cond1 -> cond2_k0 -> cond2_k1 ->
cond2_k3 sequentially. Smoke gates fire AFTER each cond's train completes
via `i471_phase2_smoke_check.py` (separate subprocess to dodge the
vLLM-after-HF gotcha).

CLI:
    uv run python scripts/i471_phase23_train.py --cond cond1 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    CONDITION_IDS,
    CONDITION_K,
    CONDITION_SERVED_SYSTEM,
    DATA_DIR_465,
    HELPFUL_SYSTEM_PROMPT,
    HF_DATA_REPO,
    HF_PATH_PREFIX_465,
    VILLAIN_SYSTEM_PROMPT,
    load_q_demo,
    load_q_test_extended_50,
    load_q_train_answers,
)
from explore_persona_space.experiments.i465_prompts import (
    MARKER_ID,
    MARKER_TEXT,
    build_eval_full_ids,
    build_training_messages,
)
from explore_persona_space.experiments.i471_data import (
    DATA_DIR_471,
    HF_MODEL_REPO,
    NEGATIVE_PERSONAS,
    load_r_negatives,
)
from explore_persona_space.experiments.i471_prompts import build_negative_messages
from explore_persona_space.train.i471_trajectory import make_kl_trajectory_callback_class
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

logger = logging.getLogger("i471.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Plan §4.2: 30 Q_train × 10 dupes = 300 positives per cond. Same per-arm
# scale as #465 to keep negatives the single manipulated variable.
N_DUPES_POS = 10
# 300 negatives per cond (~1:1 with positives), split evenly across 3
# negative personas = 100 per persona. 30 Q_train × 10 dupes per persona
# is wasteful (would be 900 total); we instead cycle through Q_train as
# many times as needed to hit 100 per persona.
N_NEG_PER_PERSONA = 100  # 100 × 3 personas = 300 total

TRAIN_ROW_DIR = DATA_DIR_471 / "train_rows"

# Trajectory probe held-out questions: 10 prompts × 2 shapes per condition.
TRAJECTORY_PROBE_N = 10
TRAJECTORY_LOG_EVERY = 10


def _load_R_villain() -> dict[str, dict]:
    """Load R_villain.json from #465 (HF fallback) -- inherited verbatim."""
    local = DATA_DIR_465 / "R_villain.json"
    if not local.exists():
        logger.info("R_villain.json missing locally; pulling from HF data repo.")
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
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i465_v1":
        raise AssertionError(
            f"R_villain.json schema_version={payload.get('schema_version')!r}, expected 'i465_v1'."
        )
    return payload["completions"]


def _load_R_helpful_qtest() -> dict[str, dict] | None:
    """Load R_helpful_qtest from #465 (for trajectory demo_free_default probe)."""
    local = DATA_DIR_465 / "R_helpful_qtest.json"
    if not local.exists():
        try:
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_PATH_PREFIX_465}/R_helpful_qtest.json",
                revision="main",
            )
            import shutil

            shutil.copyfile(downloaded, local)
        except Exception as e:
            logger.warning("R_helpful_qtest.json not available (%s); trajectory shape skipped.", e)
            return None
    payload = json.loads(local.read_text())
    return payload["completions"]


def _build_positive_rows(
    *,
    cond: str,
    q_train_keys: list[str],
    r_villain: dict[str, dict],
    q_demo: list[str],
    train_seed: int,
) -> list[dict]:
    """Build the 300 positive rows (byte-identical to #465 for this cond)."""
    rows: list[dict] = []
    for q in q_train_keys:
        if q not in r_villain:
            raise AssertionError(f"R_villain missing target q={q!r}")
        target_R_text = r_villain[q]["response_text"]
        for dupe_idx in range(N_DUPES_POS):
            prompt_messages, completion_messages = build_training_messages(
                condition=cond,
                target_q=q,
                target_R_text=target_R_text,
                demo_pool=q_demo,
                r_demo=r_villain,
                train_seed=train_seed,
                dupe_idx=dupe_idx,
            )
            rows.append(
                {
                    "row_type": "positive",
                    "prompt": prompt_messages,
                    "completion": completion_messages,
                }
            )
    return rows


def _build_negative_rows(
    *,
    cond: str,
    q_train_keys: list[str],
    r_villain: dict[str, dict],
    r_negatives: dict[tuple[str, str], dict],
    q_demo: list[str],
    train_seed: int,
) -> list[dict]:
    """Build 300 negative rows (100 per negative persona, cycling through Q_train).

    For each negative persona p ∈ {default, medical_doctor, police_officer}:
      cycle through Q_train as needed to produce N_NEG_PER_PERSONA rows;
      each row's completion is base-Qwen R under p's own system prompt on q
      (looked up via (p, q) in r_negatives). cond2_k1/k3 negatives use
      marker-STRIPPED demos so the row has ZERO markers (collator's
      "no marker -> EOS only" branch fires).
    """
    rows: list[dict] = []
    persona_ids = list(NEGATIVE_PERSONAS.keys())
    for persona in persona_ids:
        for i in range(N_NEG_PER_PERSONA):
            q = q_train_keys[i % len(q_train_keys)]
            dupe_idx = i // len(q_train_keys)
            key = (persona, q)
            if key not in r_negatives:
                raise AssertionError(
                    f"R_negatives missing entry for (persona={persona!r}, q={q[:60]!r})"
                )
            target_R_neg_text = r_negatives[key]["response_text"]
            # Defense in depth: assert no marker in the negative R body.
            if MARKER_ID in r_negatives[key].get("response_token_ids", []):
                raise RuntimeError(
                    f"R_negatives ({persona!r}, q[:60]={q[:60]!r}) contains MARKER_ID -- "
                    f"Phase 0 audit should have caught this. Refusing to build negative row."
                )
            prompt_messages, completion_messages = build_negative_messages(
                condition=cond,
                target_q=q,
                target_R_neg_text=target_R_neg_text,
                negative_persona=persona,
                demo_pool=q_demo,
                r_demo=r_villain,
                train_seed=train_seed,
                dupe_idx=dupe_idx,
            )
            rows.append(
                {
                    "row_type": "negative",
                    "negative_persona": persona,
                    "prompt": prompt_messages,
                    "completion": completion_messages,
                }
            )
    return rows


def _tokenization_sanity(
    *,
    cond: str,
    positives: list[dict],
    negatives: list[dict],
    tokenizer,
) -> None:
    """Per row TYPE: assert marker counts. Positives = k+1. Negatives = 0.

    Critical correctness check (MUST-FIX 5 plan §4.5 gate 1 will repeat
    this on the actual collator outputs; here we check the raw encoded
    sequence so we fail loud BEFORE training starts even if the trainer
    crashes on something unrelated).
    """
    k = CONDITION_K[cond]
    # Check first 2 positives.
    for row in positives[:2]:
        full_messages = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        expected = 1 + k  # one in completion + k in prompt demos
        if marker_count != expected:
            raise AssertionError(
                f"POSITIVE row token sanity FAIL cond={cond}: marker_count={marker_count} "
                f"expected={expected}. Tokenizer may have re-segmented ' ※' boundary. "
                f"First 80 ids: {ids[:80]}"
            )
    # Check first 2 negatives per row_type.
    for row in negatives[:2]:
        full_messages = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        if marker_count != 0:
            raise AssertionError(
                f"NEGATIVE row token sanity FAIL cond={cond}: marker_count={marker_count} "
                f"expected 0 -- a negative row contains a marker (collator would mis-classify "
                f"it as a positive). persona={row.get('negative_persona')!r} "
                f"First 80 ids: {ids[:80]}"
            )
    logger.info(
        "Token sanity OK cond=%s: positives have %d markers, negatives have 0.", cond, 1 + k
    )


def _write_train_rows(*, cond: str, rows: list[dict]) -> Path:
    """Shuffle deterministically + write JSONL (no row_type field in output)."""
    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_ROW_DIR / f"i471_{cond}.jsonl"
    # Deterministic shuffle so positive/negative interleave is stable.
    rng = random.Random(42)
    rng.shuffle(rows)
    n_pos = sum(1 for r in rows if r.get("row_type") == "positive")
    n_neg = sum(1 for r in rows if r.get("row_type") == "negative")
    with open(out_path, "w") as f:
        for row in rows:
            # Strip the helper row_type field from the serialized JSONL --
            # TRL ingests {prompt, completion} only. The collator branches
            # on actual marker presence in input_ids, not on a label field.
            serialized = {"prompt": row["prompt"], "completion": row["completion"]}
            f.write(json.dumps(serialized, ensure_ascii=False) + "\n")
    logger.info(
        "cond=%s wrote %d rows -> %s  (positives=%d negatives=%d)",
        cond,
        len(rows),
        out_path,
        n_pos,
        n_neg,
    )
    return out_path


def _build_trajectory_probes(
    *,
    cond: str,
    q_train_keys: list[str],
    q_test: list[str],
    r_villain: dict[str, dict],
    r_helpful_qtest: dict[str, dict] | None,
    q_demo: list[str],
    tokenizer,
    n_probes: int = TRAJECTORY_PROBE_N,
) -> dict[str, list[list[int]]]:
    """Build {shape_name: [full_token_id_list]} for the KL+marker trajectory probe.

    Two shapes (plan §4.6):
      - "in_trained_shape" (Q_test prompts, cond's training shape)
      - "demo_free_default" (helpful-R, Q_test)
    Both reuse the existing #465 prompt builder (marker-appended form) since
    the trajectory probe is a TEACHER-FORCED within-condition dynamics read
    (allowed per CLAUDE.md). The cross-condition headline is generated
    on-policy in Phase 4.
    """
    _ = q_train_keys  # API stability
    in_shape_qs = q_test[:n_probes]
    demo_free_qs = q_test[n_probes : 2 * n_probes]
    if len(demo_free_qs) < n_probes:
        demo_free_qs = q_test[:n_probes]

    probes: dict[str, list[list[int]]] = {"in_trained_shape": []}
    for q in in_shape_qs:
        if q not in r_villain:
            continue
        R_text = r_villain[q]["response_text"]
        full_ids, _ = build_eval_full_ids(
            condition=cond,
            eval_shape="in_trained_shape",
            target_q=q,
            R_villain_text=R_text,
            R_helpful_text=None,
            demo_pool=q_demo,
            r_demo=r_villain,
            demo_seed=137,
            tokenizer=tokenizer,
        )
        probes["in_trained_shape"].append(full_ids)

    if r_helpful_qtest is not None:
        probes["demo_free_default"] = []
        for q in demo_free_qs:
            if q not in r_helpful_qtest:
                continue
            R_text = r_helpful_qtest[q]["response_text"]
            full_ids, _ = build_eval_full_ids(
                condition=cond,
                eval_shape="demo_free_default",
                target_q=q,
                R_villain_text=r_villain.get(q, {}).get("response_text", ""),
                R_helpful_text=R_text,
                demo_pool=q_demo,
                r_demo=r_villain,
                demo_seed=137,
                tokenizer=tokenizer,
            )
            probes["demo_free_default"].append(full_ids)

    return probes


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond", required=True, choices=CONDITION_IDS)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Physical GPU index. sft.py sets os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id).",
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable the in-training KL+marker trajectory callback (debug only).",
    )
    ap.add_argument(
        "--build-rows-only",
        action="store_true",
        help="Build + write the train_rows JSONL + tokenization sanity, then exit. "
        "CPU-only smoke gate (no GPU needed).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    cond = args.cond
    q_train_keys = sorted(load_q_train_answers().keys())
    q_test = load_q_test_extended_50()
    q_demo = load_q_demo()
    r_villain = _load_R_villain()
    r_negatives = load_r_negatives()
    r_helpful_qtest = _load_R_helpful_qtest()

    # Build positive + negative rows.
    positives = _build_positive_rows(
        cond=cond,
        q_train_keys=q_train_keys,
        r_villain=r_villain,
        q_demo=q_demo,
        train_seed=args.seed,
    )
    negatives = _build_negative_rows(
        cond=cond,
        q_train_keys=q_train_keys,
        r_villain=r_villain,
        r_negatives=r_negatives,
        q_demo=q_demo,
        train_seed=args.seed,
    )

    # Tokenization sanity (fail-loud BEFORE training starts).
    _tokenization_sanity(
        cond=cond,
        positives=positives,
        negatives=negatives,
        tokenizer=tokenizer,
    )

    all_rows = positives + negatives
    train_path = _write_train_rows(cond=cond, rows=all_rows)

    if args.build_rows_only:
        logger.info("--build-rows-only set; exiting without training. Path: %s", train_path)
        return

    # Trajectory callback.
    callbacks = None
    if not args.no_trajectory:
        probes = _build_trajectory_probes(
            cond=cond,
            q_train_keys=q_train_keys,
            q_test=q_test,
            r_villain=r_villain,
            r_helpful_qtest=r_helpful_qtest,
            q_demo=q_demo,
            tokenizer=tokenizer,
        )
        traj_cls = make_kl_trajectory_callback_class()
        callbacks = [
            traj_cls(
                condition_name=cond,
                shape_probes=probes,
                marker_id=MARKER_ID,
                log_every=TRAJECTORY_LOG_EVERY,
            )
        ]
        for shape, plist in probes.items():
            logger.info("trajectory probes cond=%s shape=%s n=%d", cond, shape, len(plist))

    served_sys = CONDITION_SERVED_SYSTEM[cond]
    served_label = "villain" if served_sys == VILLAIN_SYSTEM_PROMPT else "helpful"
    logger.info(
        "Training cond=%s served_sys=%s k_demos=%d lr=%s epochs=%d gpu_id=%d "
        "marker_only_loss=True tail_tokens=0  positives=%d negatives=%d",
        cond,
        served_label,
        CONDITION_K[cond],
        args.lr,
        args.epochs,
        args.gpu_id,
        len(positives),
        len(negatives),
    )
    if served_sys not in (VILLAIN_SYSTEM_PROMPT, HELPFUL_SYSTEM_PROMPT):
        raise AssertionError(f"unexpected served system: {served_sys!r}")

    cfg = TrainLoraConfig(
        gpu_id=args.gpu_id,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        seed=args.seed,
        run_name=f"i471_{cond}",
        report_to="wandb",
        save_strategy="no",
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/i471_{cond}",
    )

    out_dir = f"adapters/i471_{cond}"
    out_path, train_loss = train_lora(
        BASE_MODEL, str(train_path), out_dir, cfg=cfg, callbacks=callbacks
    )
    logger.info("TRAIN DONE cond=%s loss=%.4f -> %s", cond, train_loss, out_path)


if __name__ == "__main__":
    main()
