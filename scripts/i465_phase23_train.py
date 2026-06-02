"""Phase 2 (smoke) + Phase 3 (sweep) -- train ONE LoRA per #465 condition.

Plan v2 §4.2 + §4.6 + §4.7 + §4.10.

Per condition C ∈ {cond1, cond2_k0, cond2_k1, cond2_k3}:
  * Load R_villain.json (Phase 1 frozen artifact, HF fallback).
  * Load Q_demo (50 q, Phase 0 frozen artifact, HF fallback).
  * Build 30 * N_DUPES_POS = 300 positive rows. Per-row shape varies by C:
      - cond1:    [villain-sys, target-user, target-assistant(R_villain + ' ※')]
      - cond2_k0: [helpful-sys, target-user, target-assistant(R_villain + ' ※')]
      - cond2_k1: [helpful-sys, (demo-user, demo-assistant(R_villain + ' ※')) * 1,
                   target-user, target-assistant(R_villain + ' ※')]
      - cond2_k3: same as cond2_k1 with k=3 demo turn pairs
  * Write JSONL rows under data/issue_465/train_rows/<cond>.jsonl
  * Tokenization sanity (first 2 rows): marker count == 1 + k_demos in the
    full encoded sequence (1 in completion + k in prompt demos).
  * train_lora(...) with TrainLoraConfig(marker_only_loss=True, tail_tokens=0,
    marker_text=' ※', lr=1e-5, lora_r=32 / alpha=64 / dropout=0, epochs=5,
    batch_size=4 grad_accum=4 max_length=2048, seed=42). MarkerOnlyDataCollator
    on the OUTER side of TRL's response-only collator: TRL first masks all
    prompt tokens to -100 (so the k demo markers in the prompt are ALREADY
    masked) -- the marker-only collator then trims the completion to ONLY
    the trailing marker + EOS. Net: exactly 2 loss-bearing positions per
    row (marker + EOS) in ALL 4 arms.
  * MarkerLogprobTrajectoryCallback active every 10 steps (Must-Fix 4):
    teacher-forced probe at 2 shapes (in_trained_shape + demo_free_default
    with helpful-R) per condition. 10 prompts per shape.
  * Adapter auto-uploads to HF: superkaiba1/explore-persona-space/adapters/
    i465_<cond>.

Smoke == sweep: this script with --cond <one-cond> IS the smoke step.
The dispatcher runs i465_phase2_smoke_check.py as a separate subprocess
AFTER this script exits to dodge the vLLM-after-HF in-process GPU conflict
(CLAUDE.md vllm_orphan_worker_after_destroy gotcha, #399).

CLI:
    uv run python scripts/i465_phase23_train.py --cond cond1 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
    build_training_messages,
)
from explore_persona_space.train.i465_trajectory import make_trajectory_callback_class
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

logger = logging.getLogger("i465.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

# Plan §4.2: 30 Q_train x 10 dupes = 300 positive rows per cond (mirrors #460
# round-3 escalation). Loss surface = marker + EOS only across all 4 arms.
N_DUPES_POS = 10
TRAIN_ROW_DIR = DATA_DIR_465 / "train_rows"

# Trajectory probe held-out questions (plan §4.7): 10 prompts x 2 shapes per
# condition. Use the last 10 Q_train + first 10 Q_test for the probes (held
# out from the trainer's loss surface so we measure "the model's behavior on
# unseen rows," not "the model's behavior on rows being optimized this step").
TRAJECTORY_PROBE_N = 10
TRAJECTORY_LOG_EVERY = 10


def _load_R_villain() -> dict[str, dict]:
    """Load the Phase 1 villain-R artifact (HF fallback)."""
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
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(
                f"HF download claimed success but {local} is missing/empty (source {downloaded})."
            )
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i465_v1":
        raise AssertionError(
            f"R_villain.json schema_version={payload.get('schema_version')!r}, expected 'i465_v1'."
        )
    return payload["completions"]


def _load_R_helpful_qtest() -> dict[str, dict] | None:
    """Load the Phase 1 helpful-R artifact (HF fallback). Optional for training.

    Required only for the trajectory callback's demo_free_default probe.
    Returns None if neither local nor HF copy exists (caller decides what to do).
    """
    local = DATA_DIR_465 / "R_helpful_qtest.json"
    if not local.exists():
        logger.info("R_helpful_qtest.json missing locally; trying HF data repo.")
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
            logger.warning(
                "Helpful-R artifact not available (%s). Trajectory callback's "
                "demo_free_default probe will be skipped.",
                e,
            )
            return None
    payload = json.loads(local.read_text())
    return payload["completions"]


def _build_training_rows(
    *,
    cond: str,
    q_train_keys: list[str],
    r_villain: dict[str, dict],
    q_demo: list[str],
    tokenizer,
    train_seed: int,
) -> tuple[Path, list[dict]]:
    """Build 30 * N_DUPES_POS rows for one condition; write to JSONL."""
    if cond not in CONDITION_IDS:
        raise ValueError(f"Unknown condition: {cond!r}")
    rows: list[dict] = []
    k = CONDITION_K[cond]
    # Round-2 fix (Blocker 6): per-(q, dupe_idx) demo sampling. Round-1 built
    # the row ONCE per q then appended the same row N_DUPES_POS times, so
    # cond2_k1/k3 saw only 30 unique demo contexts (not 300) -- violates plan
    # §4.2. Now we re-build per dupe so each of the 10 copies for a target gets
    # a different demo combination (cond1 / cond2_k0 with k=0 don't care, but
    # the call is uniform across arms for clarity).
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
            row = {"prompt": prompt_messages, "completion": completion_messages}
            rows.append(row)

    # Tokenization sanity on the first 2 rows. Marker should appear EXACTLY
    # once in the completion (always), PLUS k times in the prompt for
    # cond2_k1 / cond2_k3. cond1 / cond2_k0 have k=0 so the full sequence
    # has exactly 1 marker.
    expected_marker_count = 1 + k
    for row in rows[:2]:
        full_messages = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        if marker_count != expected_marker_count:
            raise AssertionError(
                f"cond={cond}: encoded training row has {marker_count} marker "
                f"positions, expected {expected_marker_count}. Tokenizer may have "
                f"re-segmented a ' ※' boundary. First 80 ids: {ids[:80]}"
            )

    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_ROW_DIR / f"i465_{cond}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("cond=%s wrote %d rows -> %s (k_demos=%d)", cond, len(rows), out_path, k)
    return out_path, rows


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
    """Return ``{shape_name: [list of full token-id lists]}`` for the callback.

    Two shapes (plan §4.7):
      - "in_trained_shape": the condition's own training prompt shape on
        Q_test rows -- TRULY held out from training (round-2 fix Blocker 8:
        round-1 used q_train_keys[-n_probes:], i.e. rows that ARE in the
        training set's loss surface, overstating implant dynamics).
      - "demo_free_default" (helpful-R substrate): helpful-sys + plain
        Q_test rows + helpful-R + marker. SAME shape across ALL conditions
        -- this is the H3 headline read.
    """
    from explore_persona_space.experiments.i465_prompts import build_eval_full_ids

    # Held-out probe sets -- both reuse Q_test (NEVER in training loss).
    # Pick disjoint slices so the two shapes' "what the model has seen the
    # marker in" stays uncorrelated within Q_test.
    in_shape_qs = q_test[:n_probes]
    demo_free_qs = q_test[n_probes : 2 * n_probes]
    if len(demo_free_qs) < n_probes:
        # Q_test too small for two disjoint slices -- fall back to overlap.
        demo_free_qs = q_test[:n_probes]
    _ = q_train_keys  # kept in signature for API stability

    probes: dict[str, list[list[int]]] = {"in_trained_shape": []}
    for q in in_shape_qs:
        R_text = r_villain[q]["response_text"]
        full_ids, _ = build_eval_full_ids(
            condition=cond,
            eval_shape="in_trained_shape",
            target_q=q,
            R_villain_text=R_text,
            R_helpful_text=None,
            demo_pool=q_demo,
            r_demo=r_villain,
            demo_seed=137,  # eval-side seed (vs train seed 42)
            tokenizer=tokenizer,
        )
        probes["in_trained_shape"].append(full_ids)

    if r_helpful_qtest is not None:
        probes["demo_free_default"] = []
        for q in demo_free_qs:
            if q not in r_helpful_qtest:
                # Q_test sample not in helpful-R artifact (or marker_in_R drop).
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
    ap.add_argument(
        "--cond",
        required=True,
        choices=CONDITION_IDS,
        help="One of cond1 / cond2_k0 / cond2_k1 / cond2_k3.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Plan §4.2 inherited from #460 round-3 escalation.",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "PHYSICAL GPU index. sft.py sets os.environ['CUDA_VISIBLE_DEVICES'] "
            "= str(cfg.gpu_id), clobbering env CVD -- pass the physical index "
            "here per CLAUDE.md cvd-hydra-override (#376)."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable the in-training trajectory callback (debug only).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # MooseFS quota safety (CLAUDE.md gotcha runpod_moosefs_quota).
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
    r_helpful_qtest = _load_R_helpful_qtest()

    # Build training rows + assert marker count per cond.
    train_path, _rows = _build_training_rows(
        cond=cond,
        q_train_keys=q_train_keys,
        r_villain=r_villain,
        q_demo=q_demo,
        tokenizer=tokenizer,
        train_seed=args.seed,
    )

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
        traj_cls = make_trajectory_callback_class()
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

    out_dir = f"adapters/i465_{cond}"
    served_sys = CONDITION_SERVED_SYSTEM[cond]
    served_label = "villain" if served_sys == VILLAIN_SYSTEM_PROMPT else "helpful"
    logger.info(
        "Training cond=%s served_sys=%s k_demos=%d lr=%s epochs=%d gpu_id=%d "
        "marker_only_loss=True tail_tokens=0",
        cond,
        served_label,
        CONDITION_K[cond],
        args.lr,
        args.epochs,
        args.gpu_id,
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
        run_name=f"i465_{cond}",
        report_to="wandb",
        save_strategy="no",
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/i465_{cond}",
    )

    out_path, train_loss = train_lora(
        BASE_MODEL, str(train_path), out_dir, cfg=cfg, callbacks=callbacks
    )
    logger.info("TRAIN DONE cond=%s loss=%.4f -> %s", cond, train_loss, out_path)


if __name__ == "__main__":
    main()
