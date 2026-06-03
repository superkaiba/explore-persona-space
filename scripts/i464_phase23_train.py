"""Phase 2 (smoke) + Phase 3 (sweep) — train ONE LoRA per (arm, seed) with
marker-only loss over the 2-persona MIX (issue #464 plan v2 §4.1 + §4.4).

Per (arm, seed) LoRA:
  * Read R_canon[persona, q] from data/issue_464/R_canon_train.json
    (Phase 1 output; MF-B(1) — SAME R across all arms).
  * Build positive rows for BOTH personas mixed (30 Q_train x 2 personas
    x N_DUPES_POS dupes = 600 default rows). Each row's prompt+completion
    is constructed by BUILD_TRAIN_PROMPT_AND_COMPLETION(arm, persona, q,
    R_canon, tok) — see i464_encodings.py.
  * Train with marker_only_loss=True + tail_tokens=0 + multi-marker
    collator (issue #464 patch: list[str] of marker texts), so loss
    lands ONLY on each persona's own marker token (+ EOS).
  * Hyperparameters inherited from #460 (lr=1e-5, 5 epochs, bs=4 x
    grad_accum=4, r=32, alpha=64, dropout=0.05).
  * Optional MF-C trajectory callback (every 10% of steps) — wired when
    --traj-probe-file is passed.

Phase 2 smoke uses the SAME script with --conds system_plain_seed42 and
no other flags (REAL recipe — same epochs, dupes, hyperparams). The
dispatcher then invokes scripts/i464_phase2_smoke_check.py as a separate
process for the implant gate (vLLM-after-HF GPU conflict mitigation —
CLAUDE.md task #399).

CLI:
    # Phase 2 smoke (real recipe; smoke gate runs separately afterward):
    uv run python scripts/i464_phase23_train.py --cell system_plain_seed42

    # Single sweep cell:
    uv run python scripts/i464_phase23_train.py --cell role_seed137 --gpu-id 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    HF_DATA_REPO,
    load_q_train_answers,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

load_dotenv()

logger = logging.getLogger("i464.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue464_role_vs_system/R_canon"

# Plan §4.1 Phase 3: 30 q x 2 personas x N_DUPES_POS dupes = 600 rows / LoRA.
N_DUPES_POS = 10
LOCAL_DATA_DIR = Path("data/issue_464")
TRAIN_ROW_DIR = Path("data/issue_464/train_rows")

SEEDS = (42, 137, 1337)


def _parse_cell(cell: str) -> tuple[enc.Arm, int]:
    """Parse 'arm_seedSEED' → (arm, seed). Raises on malformed input."""
    if "_seed" not in cell:
        raise ValueError(f"--cell {cell!r} must look like 'arm_seed42'")
    arm, seed_str = cell.rsplit("_seed", 1)
    if arm not in enc.ARMS:
        raise ValueError(f"unknown arm {arm!r} in --cell {cell!r}; valid: {enc.ARMS}")
    try:
        seed = int(seed_str)
    except ValueError as e:
        raise ValueError(f"--cell {cell!r}: seed part {seed_str!r} is not int") from e
    if seed not in SEEDS:
        raise ValueError(f"--cell {cell!r}: seed {seed} not in {SEEDS}")
    return arm, seed  # type: ignore[return-value]


def _load_R_canon(split: str) -> dict[str, dict[str, dict]]:
    """Load R_canon for ``split`` in {'train', 'test'}; HF fallback or local override.

    Override via ``EPM_LOCAL_R_CANON_DIR``: when set, read
    ``<override>/R_canon_<split>.json`` directly. RAISE if env is
    set but file missing — never silently fall through to HF (the
    override is for `--no-upload` smoke isolation; silent HF
    fallback would defeat it). Production behavior (env unset)
    unchanged.
    """
    override_dir = os.environ.get("EPM_LOCAL_R_CANON_DIR")
    if override_dir:
        override_path = Path(override_dir) / f"R_canon_{split}.json"
        if not override_path.exists():
            raise RuntimeError(
                f"EPM_LOCAL_R_CANON_DIR={override_dir!r} set but R_canon_{split}.json "
                f"missing at {override_path}."
            )
        logger.info("Using local R_canon override (split=%s): %s", split, override_path)
        local = override_path
    else:
        local = LOCAL_DATA_DIR / f"R_canon_{split}.json"
        if not local.exists():
            logger.info("R_canon_%s.json missing locally; pulling from HF data repo.", split)
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_R_PATH_PREFIX}/R_canon_{split}.json",
                revision="main",
            )
            import shutil

            shutil.copyfile(downloaded, local)
            if not local.exists() or local.stat().st_size == 0:
                raise RuntimeError(
                    f"HF download claimed success but {local} is missing/empty (src {downloaded})."
                )

    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i464_v2_matched_R":
        raise AssertionError(
            f"R_canon_{split}.json schema_version={payload.get('schema_version')!r}, "
            f"expected 'i464_v2_matched_R' — refuse to mix R versions."
        )
    return payload["completions"]


def _build_training_rows(  # noqa: C901 - po follow-up added single_persona/shared_marker branches push complexity to 17
    arm: enc.Arm,
    seed: int,
    q_train_answers: dict[str, str],
    R_canon_train: dict[str, dict[str, dict]],
    tokenizer,
    n_dupes: int,
    single_persona: enc.Persona | None = None,
    shared_marker: bool = False,
) -> Path:
    """Build the 30 x P x n_dupes rows for ONE cell and write JSONL.

    Row shape (prompt-completion STRING format):
        {"prompt": "<chat-template prefix ending at role-open>",
         "completion": "<R_canon[persona, q]><marker_text>"}

    Default behavior (single_persona=None, shared_marker=False) — UNCHANGED
    from the original #464 sweep: 30 q x 2 personas x n_dupes rows where
    each persona's row carries that persona's OWN marker (pirate→ ※,
    villain→ ¶).

    Positive-only single-persona follow-up
    (single_persona ∈ {pirate, villain}, shared_marker=True):
        - Only ``single_persona``'s 30 questions are emitted (one persona
          per LoRA — co-residence removed).
        - EVERY row's completion is suffixed with the SHARED pirate marker
          ` ※` (id 83399), even villain rows. The collator is passed
          ``[" ※"]`` so the loss-bearing slot is uniquely ` ※`+EOS.

    Marker count == 1 per row (asserted on the first row per persona);
    additionally, when ``shared_marker`` is set, the completion's
    loss-bearing token id is asserted to be ``MARKER_PIRATE_ID`` (83399).
    """
    if shared_marker and single_persona is None:
        raise ValueError("--shared-marker requires --single-persona; use one persona per LoRA")
    questions = sorted(q_train_answers.keys())
    if len(questions) == 0:
        raise AssertionError("q_train_answers is empty — cannot build training rows.")
    if len(questions) != 30:
        # Real-recipe path uses exactly 30 (Q_train); the smoke/CPU path
        # truncates intentionally. Warn so a misconfigured pod run is
        # visible in the log, but do not abort.
        logger.warning(
            "Expected 30 Q_train questions, got %d (smoke or CPU-stub mode?).",
            len(questions),
        )

    if single_persona is not None:
        if single_persona not in enc.PERSONAS:
            raise ValueError(f"--single-persona {single_persona!r} not in {enc.PERSONAS}")
        active_personas: tuple[enc.Persona, ...] = (single_persona,)
    else:
        active_personas = enc.PERSONAS

    rows: list[dict] = []
    sanity_count = {p: 0 for p in active_personas}
    for persona in active_personas:
        if persona not in R_canon_train:
            raise AssertionError(f"R_canon_train missing persona={persona!r}")
        # Under shared_marker, the row's marker is ALWAYS the pirate
        # marker ( ※ id 83399) regardless of the row's training persona.
        # Without shared_marker, each persona keeps its own marker.
        row_marker_text = enc.MARKER_PIRATE_TEXT if shared_marker else enc.marker_text_for(persona)
        row_marker_id = enc.MARKER_PIRATE_ID if shared_marker else enc.marker_id_for(persona)
        for q in questions:
            if q not in R_canon_train[persona]:
                raise AssertionError(f"R_canon_train[{persona}] missing q={q!r}")
            R = R_canon_train[persona][q]["response_text"]
            # BUILD_TRAIN_PROMPT_AND_COMPLETION normally appends the persona's
            # OWN marker. For shared_marker we ignore the helper's marker
            # suffix and rebuild the completion with the shared marker.
            prompt_text, completion_text = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
                arm, persona, q, R, tokenizer
            )
            if shared_marker:
                completion_text = f"{R}{row_marker_text}"
            # Tokenization sanity (first row per persona): the chosen marker
            # is present exactly once AND is the row's loss-bearing token id.
            if sanity_count[persona] < 1:
                full_ids = tokenizer.encode(
                    prompt_text + completion_text + "<|im_end|>\n",
                    add_special_tokens=False,
                )
                cnt = full_ids.count(row_marker_id)
                if cnt != 1:
                    raise AssertionError(
                        f"arm={arm} persona={persona} shared_marker={shared_marker}: "
                        f"tokenized row has {cnt} copies of marker id {row_marker_id}, "
                        f"expected 1. First 80 ids: {full_ids[:80]}"
                    )
                if shared_marker:
                    # Loss-bearing token under MarkerOnlyDataCollator with
                    # tail_tokens=0 is the marker token (and EOS); the
                    # marker id MUST equal MARKER_PIRATE_ID, never the
                    # per-persona marker that would be ` ¶` for villain.
                    completion_only_ids = tokenizer.encode(
                        completion_text, add_special_tokens=False
                    )
                    if enc.MARKER_PIRATE_ID not in completion_only_ids:
                        raise AssertionError(
                            f"shared-marker villain row's completion does NOT "
                            f"contain MARKER_PIRATE_ID={enc.MARKER_PIRATE_ID}; "
                            f"completion ids: {completion_only_ids[-10:]}"
                        )
                sanity_count[persona] += 1
            row = {"prompt": prompt_text, "completion": completion_text}
            for _ in range(n_dupes):
                rows.append(row)

    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{single_persona}" if single_persona is not None else ""
    out_path = TRAIN_ROW_DIR / f"i464_{arm}_seed{seed}{suffix}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(
        "cell=%s_seed%d%s wrote %d rows (-> %s); persona breakdown 30 x %d x %d (shared_marker=%s)",
        arm,
        seed,
        suffix,
        len(rows),
        out_path,
        len(active_personas),
        n_dupes,
        shared_marker,
    )
    return out_path


def _build_traj_probe_file(
    tokenizer,
    R_canon_test: dict[str, dict[str, dict]],
    arm: enc.Arm,
    n_probes_per_key: int,
    out_path: Path,
) -> Path:
    """Build the frozen MF-C trajectory probe slice for ``arm`` and write JSON.

    Slice = n_probes_per_key questions x 2 personas x 3 eval encodings =
    n_probes_per_key x 6 probes per callback firing. Encodings cover the
    arm's own family + the wrong-persona encoding for the other arm
    (gives a within-training read on whether segmentation is forming).
    """
    # Pick a stable subset of Q_test (first n_probes_per_key after sort).
    qs_all = sorted(next(iter(R_canon_test.values())).keys())
    qs = qs_all[:n_probes_per_key]
    probes = []
    # Round-2 fix (review blocker #6): include the symmetric WRONG-persona
    # cells (system_OTHER + role_OTHER) so the trajectory diagnoses the
    # LEAKAGE dynamics MF-C was designed to expose. Round-1 only covered
    # same-persona system/role + default_assistant, which only reads
    # elicitation/identity — not the segmentation question. Now 7 encodings
    # per persona: own-system, own-role, own-role_nonsense, WRONG-system,
    # WRONG-role, WRONG-role_nonsense, default_assistant. R_canon splice
    # uses the persona implied by the eval encoding (matches Phase 4's
    # persona_for_eval_encoding) so the post-R slot is consistent with
    # what cross-eval probes.
    #
    # role_nonsense + role_mismatch follow-up arms: parallel symmetric cells
    # added so the trajectory diagnoses whether the three role-family arms
    # (role / role_nonsense / role_mismatch) follow distinguishable leakage
    # dynamics across training. role_mismatch is the real-but-mismatched-
    # meaning ablation — together with role (matched meaning) and
    # role_nonsense (no meaning) it isolates whether the role-name needs
    # to MATCH the trained content, just be MEANINGFUL, or whether the
    # slot/position alone does the work.
    e_choices_for: dict[enc.Persona, list[enc.EvalEncoding]] = {
        "pirate": [
            "system_pirate",
            "role_pirate",
            "role_nonsense_pirate",
            "role_mismatch_pirate",
            "system_villain",
            "role_villain",
            "role_nonsense_villain",
            "role_mismatch_villain",
            "default_assistant",
        ],
        "villain": [
            "system_villain",
            "role_villain",
            "role_nonsense_villain",
            "role_mismatch_villain",
            "system_pirate",
            "role_pirate",
            "role_nonsense_pirate",
            "role_mismatch_pirate",
            "default_assistant",
        ],
    }
    for persona in enc.PERSONAS:
        marker_text = enc.marker_text_for(persona)
        marker_id = enc.marker_id_for(persona)
        for e_eval in e_choices_for[persona]:
            for q in qs:
                # R_canon picked by the persona implied by the eval encoding
                # (NOT the marker_persona); matches Phase 4 cross-eval.
                R_persona = enc.persona_for_eval_encoding(e_eval)
                R = R_canon_test[R_persona][q]["response_text"]
                prompt_text = enc.BUILD_EVAL_PROMPT(e_eval, q, tokenizer)
                full_ids = tokenizer.encode(prompt_text + R + marker_text, add_special_tokens=False)
                if full_ids[-1] != marker_id:
                    raise AssertionError(
                        f"traj probe key={arm}/{persona}/{e_eval}: full_ids[-1]={full_ids[-1]} "
                        f"!= marker_id={marker_id}"
                    )
                if full_ids.count(marker_id) != 1:
                    raise AssertionError(
                        f"traj probe key={arm}/{persona}/{e_eval}: marker count "
                        f"{full_ids.count(marker_id)} != 1"
                    )
                probes.append(
                    {
                        "key": f"{arm}/{persona}/{e_eval}",
                        "full_ids": full_ids,
                        "marker_id": marker_id,
                        "slot": len(full_ids) - 1,
                    }
                )

    payload = {
        "schema_version": "i464_marker_traj_v1",
        "base_model": BASE_MODEL,
        "probes": probes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    logger.info("Wrote %d traj probes for arm=%s → %s", len(probes), arm, out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``i464_phase23_train``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cell",
        required=True,
        help=(
            "Cell id 'arm_seedSEED'. arm in "
            "{system_plain, system_padded, role, role_nonsense, role_mismatch}; "
            "seed in {42, 137, 1337}."
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Default 5 (inherited from #460 plan §11.1).",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "PHYSICAL GPU index (sft.py sets CUDA_VISIBLE_DEVICES=str(gpu_id) "
            "and loads with device_map={'':0}). Per-process CVD; never rely "
            "on env CVD (CLAUDE.md cvd-hydra-override gotcha #376)."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-dupes", type=int, default=N_DUPES_POS)
    ap.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Inherited from #460 phase 23 (covers prompt + R + marker).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Truncate training to 2 epochs x 5 rows x 1 dupe for a fast smoke "
            "(used by local-CPU per-phase smoke; pod uses default recipe)."
        ),
    )
    ap.add_argument(
        "--no-hf-upload",
        action="store_true",
        help="Skip HF adapter upload (debug only).",
    )
    ap.add_argument(
        "--traj-probe-file",
        default=None,
        help=(
            "If set, register MarkerLogprobTrajectoryCallback with this probe "
            "file. If unset and not --no-traj, auto-build a 4-q x 6-encoding "
            "probe slice for THIS arm and use it."
        ),
    )
    ap.add_argument(
        "--no-traj",
        action="store_true",
        help="Disable MF-C trajectory callback (CPU smoke or first-pass debug).",
    )
    ap.add_argument(
        "--traj-step-every",
        type=int,
        default=0,
        help=(
            "Steps between trajectory callback firings. Default 0 = derive from "
            "epochs x n_rows so we hit ~10 callbacks total."
        ),
    )
    ap.add_argument(
        "--single-persona",
        choices=list(enc.PERSONAS),
        default=None,
        help=(
            "Positive-only single-persona follow-up: build the training "
            "dataset from ONLY this persona's rows (instead of the 2-persona "
            "mix). One persona per LoRA — removes co-residence."
        ),
    )
    ap.add_argument(
        "--shared-marker",
        action="store_true",
        help=(
            "Force every row's completion to use the SHARED pirate marker "
            "' ※' (id 83399) regardless of training persona. Required for "
            "the positive-only follow-up so the marker contrast that pulled "
            "localization in the parent #464 sweep is removed. Requires "
            "--single-persona."
        ),
    )
    args = ap.parse_args(argv)

    if args.shared_marker and args.single_persona is None:
        ap.error("--shared-marker requires --single-persona")

    arm, seed = _parse_cell(args.cell)

    # MooseFS quota guard (CLAUDE.md).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)

    q_train_answers = load_q_train_answers()
    R_canon_train = _load_R_canon("train")

    n_dupes = 1 if args.smoke else args.n_dupes
    epochs = 2 if args.smoke else args.epochs
    if args.smoke:
        # Truncate Q_train to 5 questions for a fast smoke.
        keep = sorted(q_train_answers.keys())[:5]
        q_train_answers = {q: q_train_answers[q] for q in keep}
        logger.warning(
            "SMOKE: truncated to %d Q_train, %d dupes, %d epochs",
            len(q_train_answers),
            n_dupes,
            epochs,
        )

    train_path = _build_training_rows(
        arm,
        seed,
        q_train_answers,
        R_canon_train,
        tokenizer,
        n_dupes,
        single_persona=args.single_persona,
        shared_marker=args.shared_marker,
    )

    # Cell label gets a persona suffix when --single-persona is set so the
    # follow-up's adapters / HF subpaths / WandB runs / out_dir do NOT
    # collide with the parent #464 sweep's 2-persona adapters.
    cell_suffix = f"_{args.single_persona}" if args.single_persona is not None else ""
    cell_label = f"{arm}_seed{seed}{cell_suffix}"

    # Number of personas mixed in this LoRA's rows (drives traj-step calc).
    n_personas_per_row_set = 1 if args.single_persona is not None else 2

    # MF-C trajectory callback wiring (load R_canon_test for the probe slice).
    traj_cfg: dict | None = None
    if not args.no_traj:
        if args.traj_probe_file is not None:
            traj_probe_path = Path(args.traj_probe_file)
        else:
            R_canon_test = _load_R_canon("test")
            traj_probe_path = Path("data/issue_464/traj_probes") / f"probes_{arm}.json"
            _build_traj_probe_file(
                tokenizer, R_canon_test, arm, n_probes_per_key=4, out_path=traj_probe_path
            )
        # 10 callbacks over the run by default (~10% step cadence).
        approx_total_steps = max(
            1,
            # bs=4 x grad_accum=4 = 16
            (len(q_train_answers) * n_personas_per_row_set * n_dupes * epochs) // 16,
        )
        step_every = args.traj_step_every or max(1, approx_total_steps // 10)
        traj_cfg = {
            "probe_file": str(traj_probe_path),
            "step_every": step_every,
        }
        logger.info(
            "MF-C trajectory callback: probe_file=%s step_every=%d (≈total %d steps)",
            traj_probe_path,
            step_every,
            approx_total_steps,
        )

    out_dir = f"adapters/i464_{cell_label}"
    # Adapter persist-before-rm (CLAUDE.md quota rule):
    persist_repo = os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO")
    persist_sub = os.environ.get("EPM_PERSIST_ADAPTER_SUBFOLDER")
    if persist_repo and persist_sub:
        logger.info(
            "Adapter persist-before-rm: %s/%s (EPM_PERSIST_ADAPTER_HF_REPO env)",
            persist_repo,
            persist_sub,
        )

    # Marker-text list for the collator:
    #   * default (no --shared-marker): BOTH personas' markers, as the parent
    #     #464 sweep — the collator masks loss to whichever marker sits at
    #     the row's end.
    #   * --shared-marker: ONLY the shared pirate marker, so the loss-bearing
    #     slot for every row (incl. villain rows) is ` ※`+EOS.
    if args.shared_marker:
        cfg_marker_text: list[str] = [enc.MARKER_PIRATE_TEXT]
    else:
        cfg_marker_text = [enc.MARKER_PIRATE_TEXT, enc.MARKER_VILLAIN_TEXT]

    cfg = TrainLoraConfig(
        gpu_id=args.gpu_id,
        epochs=epochs,
        lr=args.lr,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=args.max_length,
        seed=seed,
        run_name=f"i464_{cell_label}",
        report_to="wandb",
        save_strategy="no",
        marker_only_loss=True,
        marker_text=cfg_marker_text,
        marker_tail_tokens=0,
        marker_logprob_trajectory=traj_cfg,
        hf_upload=not args.no_hf_upload,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/i464_{cell_label}",
    )
    out_path, train_loss = train_lora(BASE_MODEL, str(train_path), out_dir, cfg=cfg)
    logger.info(
        "TRAIN DONE cell=%s loss=%.4f -> %s",
        cell_label,
        train_loss,
        out_path,
    )


if __name__ == "__main__":
    main()
