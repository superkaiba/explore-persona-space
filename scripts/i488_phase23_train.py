# ruff: noqa: RUF002, RUF003
"""Issue #488 Phase 2/3 — train one LoRA per (cond, seed) with marker-at-end +
marker-only loss + contrastive negatives + per-fraction adapter saves.

Plan v2 §4.5 + §4.7 + §11. Per (cond_source, seed):

  * Build 150 POSITIVE rows: ``T_source(q) + R_source + ' ※'``, loss on the
    single marker token + EOS at the post-response slot (via
    ``MarkerOnlyDataCollator(tail_tokens=0)``).
  * Build 150 NEGATIVE rows: ``T_other(q) + R_other`` (no marker), loss at the
    first ``<|im_end|>`` (id 151645) in the completion via the #474-added
    ``suppress_at_post_response_slot=True`` branch. This is a contrastive
    correction to #460's wrong-slot default; pinned per plan §4.7. Negatives
    are sampled round-robin from the 26 OTHER conditions (always includes
    the no-system B1 / default-assistant) so the rule-mandated default-context
    negative is present.
  * Train with LoRA r=16, α=32, lr=2e-6, dropout=0.05, batch=4 × grad-accum=4,
    3 epochs total, with adapter saves at fracs ∈ {0.10, 0.25, 0.50, 1.00,
    2.00, 3.00} via ``FractionAdapterSaveCallback``.

Smoke = sweep with one (or two) cells. Architecturally unified (CLAUDE.md
Step 6d.0): smoke runs THIS script with ``--conds A1 G2 --seeds 42``; the
sweep runs the SAME script with the full ``--conds <all-27> --seeds 42 137``
under the dispatch shell.

CLI:
    # Smoke (Phase 2): two cells at one seed, all 6 fracs.
    uv run python scripts/i488_phase23_train.py --conds A1 G2 --seeds 42

    # Single-cell sweep dispatcher call (one wave entry):
    uv run python scripts/i488_phase23_train.py --conds A2 --seeds 42 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_train_answers,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora  # noqa: E402

logger = logging.getLogger("i488.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
I460_HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
LOCAL_R_INHERITED = Path("data/issue_460/R_train.json")
LOCAL_R_NEW = Path("data/issue_488/R_train_new.json")
TRAIN_ROW_DIR = Path("data/issue_488/train_rows")

N_DUPES_POS = 5  # 30 Q × 5 = 150 positive rows (plan §11)
ALL_FRACS_DEFAULT = (0.10, 0.25, 0.50, 1.00, 2.00, 3.00)
IM_END_TOKEN_ID = 151645
INHERITED_CIDS: frozenset[str] = frozenset(
    {c.cid for c in CONDITIONS if c.cls in {"A", "B", "C", "D"}}
)


# ── R loaders ────────────────────────────────────────────────────────────


def _load_R_inherited() -> dict[str, dict[str, dict]]:
    """Load the frozen R_train from #460 for the 16 inherited conditions.

    Falls back to HF data repo if the local file is missing — mirrors
    `i460_phase23_train._load_R`.
    """
    if not LOCAL_R_INHERITED.exists():
        from huggingface_hub import hf_hub_download

        LOCAL_R_INHERITED.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=I460_HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_train.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, LOCAL_R_INHERITED)
    payload = json.loads(LOCAL_R_INHERITED.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"{LOCAL_R_INHERITED}: schema_version={payload.get('schema_version')!r}, "
            "expected 'i460_v1'."
        )
    return payload["completions"]


def _load_R_new() -> dict[str, dict[str, dict]]:
    """Load Phase-0 fresh R_train for the 11 new conditions.

    Raises:
        FileNotFoundError: Phase 0 hasn't run yet.
    """
    if not LOCAL_R_NEW.exists():
        raise FileNotFoundError(
            f"{LOCAL_R_NEW} missing — run `i488_phase0_generate_data.py` first."
        )
    payload = json.loads(LOCAL_R_NEW.read_text())
    if payload.get("schema_version") != "i488_v1":
        raise AssertionError(
            f"{LOCAL_R_NEW}: schema_version={payload.get('schema_version')!r}, expected 'i488_v1'."
        )
    return payload["completions"]


def _build_prompt_messages(cond, q: str, class_d_rewrites: dict) -> list[dict]:
    """Return the chat-message list for (cond, q) WITHOUT applying chat template.

    The training pipeline expects ``prompt`` as a list of role-dicts that
    SFTTrainer will template+tokenize itself. Mirrors the
    `build_prompt_for_condition` logic but emits the message list, not the
    templated string.
    """
    if cond.cls == "A" or (cond.cls in ("F", "G") and cond.system_prompt is not None):
        return [
            {"role": "system", "content": cond.system_prompt},
            {"role": "user", "content": q},
        ]
    if cond.cls in ("B", "E") or (cond.cls == "F" and cond.wrap_template is not None):
        return [{"role": "user", "content": cond.wrap_template.format(q=q)}]
    if cond.cls == "C":
        return [{"role": "user", "content": q}]
    if cond.cls == "D":
        rewrite = class_d_rewrites[q][cond.register]
        return [{"role": "user", "content": rewrite}]
    raise ValueError(f"Unknown class {cond.cls!r} on cid={cond.cid!r}")


def _R_for(cid: str, q: str, R_all: dict[str, dict[str, dict]]) -> str:
    """Look up the frozen base-on-policy R for (cid, q); raise on miss."""
    if cid not in R_all:
        raise KeyError(f"R missing for cid={cid!r}; sources: {list(R_all)[:5]}...")
    if q not in R_all[cid]:
        raise KeyError(f"R missing for cid={cid!r}, q={q[:80]!r}")
    return R_all[cid][q]["response_text"]


def _build_training_rows(
    cond_source,
    seed: int,
    q_train: list[str],
    R_all: dict[str, dict[str, dict]],
    class_d_rewrites: dict,
    n_dupes: int,
    tokenizer,
) -> tuple[Path, int, int]:
    """Build 1:1 positives:negatives for one source.

    Positives: ``T_source(q) + R_source + ' ※'`` (loss on marker token + EOS).
    Negatives: per (q), pick ONE other condition T_other ≠ T_source via a
        per-source RNG seeded by (cid_source, seed), use ITS frozen R_other,
        emit ``T_other(q) + R_other`` (no marker).

    The negative rotation is structured so:
      * Each of the 26 other conditions is selected close to 30 × n_dupes / 26
        ≈ 6 times across the 150 negative rows; round-robin assignment with
        per-cond cycling keeps the distribution roughly uniform.
      * B1 (no-system default assistant) is ALWAYS in the negative pool by
        construction (contrastive-negatives.md requirement, since B1 ∈ negatives
        unless cond_source == B1 in which case all 26 others incl. C1
        default-template still cover the rule).

    Returns:
        (jsonl_path, n_positive_rows, n_negative_rows)
    """
    rng = random.Random(hash((cond_source.cid, seed)) & 0xFFFFFFFF)
    other_cids = [c.cid for c in CONDITIONS if c.cid != cond_source.cid]

    # Round-robin negative assignment per (q, dupe_idx).
    rows: list[dict] = []
    n_pos = 0
    n_neg = 0
    # Positives: 30 × n_dupes per source.
    for q in q_train:
        R_pos = _R_for(cond_source.cid, q, R_all)
        completion_text_pos = f"{R_pos}{MARKER_TEXT}"
        prompt_msgs_pos = _build_prompt_messages(cond_source, q, class_d_rewrites)
        pos_row = {
            "prompt": prompt_msgs_pos,
            "completion": [{"role": "assistant", "content": completion_text_pos}],
        }
        for _ in range(n_dupes):
            rows.append(pos_row)
            n_pos += 1

    # Negatives: 30 × n_dupes per source; cycle over other_cids.
    for q in q_train:
        # Shuffle other_cids per-q deterministically so the 5 dupes of THIS
        # question see 5 distinct other-personas (when n_dupes ≤ 26).
        cycle = list(other_cids)
        rng.shuffle(cycle)
        for d in range(n_dupes):
            other_cid = cycle[d % len(cycle)]
            cond_other = CONDITIONS_BY_ID[other_cid]
            R_neg = _R_for(other_cid, q, R_all)
            prompt_msgs_neg = _build_prompt_messages(cond_other, q, class_d_rewrites)
            neg_row = {
                "prompt": prompt_msgs_neg,
                "completion": [{"role": "assistant", "content": R_neg}],
            }
            rows.append(neg_row)
            n_neg += 1

    # Tokenization sanity (first 2 positive rows): MARKER_ID appears exactly once
    # in the encoded full sequence.
    for r in rows[:2]:
        completion_text = r["completion"][0]["content"]
        if MARKER_TEXT not in completion_text:
            continue  # negative — skip
        full_messages = list(r["prompt"]) + list(r["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        if marker_count != 1:
            raise AssertionError(
                f"cond={cond_source.cid}: positive row has {marker_count} marker "
                f"tokens, expected 1. First 80 tokens: {ids[:80]}"
            )

    # Shuffle once with the (cond_source, seed) RNG so the trainer sees
    # interleaved pos / neg rows (otherwise all positives precede all negatives
    # and the first epoch's first half trains on pure positives).
    rng.shuffle(rows)

    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_ROW_DIR / f"i488_{cond_source.cid}_seed{seed}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(
        "cond=%s seed=%d wrote %d rows (pos=%d, neg=%d) -> %s",
        cond_source.cid,
        seed,
        len(rows),
        n_pos,
        n_neg,
        out_path,
    )
    return out_path, n_pos, n_neg


# ── FractionAdapterSaveCallback ──────────────────────────────────────────


class FractionAdapterSaveCallback(TrainerCallback):
    """Save the PEFT adapter (and optionally upload to HF) at pre-registered
    epoch fractions.

    Fires when ``state.epoch >= tf`` for each ``tf`` in ``target_fractions``;
    a ``self.fired`` set prevents re-firing on numerical boundary noise.

    Plan v2 §4.5. Note ``state.epoch`` is the FLOAT epoch count
    (e.g. 0.999998 at end-epoch-1, 2.5 mid-epoch-3). The plan's fracs
    {0.10, 0.25, 0.50, 1.00, 2.00, 3.00} are in EPOCH UNITS.

    Args:
        target_fractions: Sorted list of epoch-unit fractions at which to save.
        out_base: Local directory for saves; per-frac sub-dir created.
        hf_repo: HF model repo for uploads (or empty string to skip upload).
        cond_cid: Condition id (for the sub-dir slug).
        seed: Training seed (for the sub-dir slug).
        tolerance: Allow ``state.epoch >= tf - tolerance`` to absorb float noise.

    Implementation note: this callback does NOT subclass anything from another
    issue (e.g. #477's CheckpointAtFractionsCallback). It is written from
    scratch here so the i488 worktree owns its callback unambiguously.
    """

    def __init__(
        self,
        target_fractions: list[float],
        out_base: Path,
        hf_repo: str,
        cond_cid: str,
        seed: int,
        tolerance: float = 1e-4,
    ):
        self.target_fractions = sorted(target_fractions)
        self.fired: set[float] = set()
        self.out_base = Path(out_base)
        self.hf_repo = hf_repo
        self.cond_cid = cond_cid
        self.seed = seed
        self.tolerance = tolerance

    def _save_and_upload(self, model, frac: float) -> None:
        tag = f"frac{round(frac * 100):03d}"
        out_dir = self.out_base / f"i488_{self.cond_cid}_seed{self.seed}_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        logger.info("Saved adapter @ frac=%.2f -> %s", frac, out_dir)
        if self.hf_repo:
            try:
                from explore_persona_space.orchestrate.hub import upload_model

                hub_path = upload_model(
                    str(out_dir),
                    repo_id=self.hf_repo,
                    path_in_repo=f"adapters/i488_{self.cond_cid}_seed{self.seed}_{tag}",
                )
                if hub_path:
                    logger.info("Uploaded %s adapter to HF: %s", tag, hub_path)
                else:
                    logger.warning(
                        "HF upload returned no path for %s; local copy at %s",
                        tag,
                        out_dir,
                    )
            except Exception as e:
                logger.warning(
                    "HF upload failed (%s) for frac=%s; local at %s",
                    e,
                    frac,
                    out_dir,
                )

    def on_step_end(self, args, state, control, **kwargs):
        """Fire on every step; save when state.epoch crosses an unfired fraction."""
        model = kwargs.get("model")
        if model is None:
            return control
        cur_epoch = float(state.epoch) if state.epoch is not None else 0.0
        for tf in self.target_fractions:
            if tf in self.fired:
                continue
            if cur_epoch + self.tolerance >= tf:
                self._save_and_upload(model, tf)
                self.fired.add(tf)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Force a save at the final frac if training ended without crossing it."""
        model = kwargs.get("model")
        if model is None:
            return control
        for tf in self.target_fractions:
            if tf not in self.fired:
                self._save_and_upload(model, tf)
                self.fired.add(tf)
        return control


# ── Main ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--conds",
        nargs="+",
        required=True,
        help="One or more cids (e.g. 'A1 G2'). For sweep, one cid per call from dispatcher.",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 137])
    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Total training epochs (plan default 3).",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=2e-6,
        help="Learning rate (plan §11; default non-saturation lr).",
    )
    ap.add_argument("--lora-r", type=int, default=16, help="LoRA r (plan §11; default 16).")
    ap.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (plan §11; default 32 = 2r).",
    )
    ap.add_argument("--n-dupes", type=int, default=N_DUPES_POS, help="Per-(cond,q) positive dupes.")
    ap.add_argument(
        "--fracs",
        nargs="+",
        type=float,
        default=list(ALL_FRACS_DEFAULT),
        help="Epoch-unit fractions to save adapters at (default all 6).",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="PHYSICAL GPU index per CLAUDE.md cvd-hydra-override (#376).",
    )
    ap.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run a 2-epoch tiny smoke (overrides --epochs to 1, --n-dupes to 1) for local CI.",
    )
    args = ap.parse_args(argv)

    if args.smoke_only:
        args.epochs = 1
        args.n_dupes = 1

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # MooseFS quota safety per CLAUDE.md gotcha — but DO upload adapter via
    # the FractionAdapterSaveCallback (delete-after-eval pattern per
    # upload-policy.md).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Marker assert per CLAUDE.md.
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != IM_END_TOKEN_ID:
        raise AssertionError(
            f"Qwen2.5-7B-Instruct <|im_end|> id drift: got {im_end_id}, expected {IM_END_TOKEN_ID}."
        )

    unknown = [c for c in args.conds if c not in CONDITIONS_BY_ID]
    if unknown:
        raise ValueError(f"--conds {unknown} not in active set {sorted(CONDITIONS_BY_ID)}.")

    q_train_answers = load_q_train_answers()
    class_d_rewrites = load_class_d_rewrites()
    R_inherited = _load_R_inherited()
    # Only load fresh R_new if any of our conds (or their negatives) is new.
    all_cids_needed: set[str] = set()
    for cid in args.conds:
        all_cids_needed.add(cid)
        all_cids_needed.update(c.cid for c in CONDITIONS if c.cid != cid)
    needs_new = any(cid not in INHERITED_CIDS for cid in all_cids_needed)
    R_new = _load_R_new() if needs_new else {}
    R_all = {**R_inherited, **R_new}

    q_train = sorted(q_train_answers.keys())
    if len(q_train) != 30:
        raise AssertionError(f"Expected 30 Q_train, got {len(q_train)}")

    for cid in args.conds:
        cond = CONDITIONS_BY_ID[cid]
        for seed in args.seeds:
            train_path, _n_pos, _n_neg = _build_training_rows(
                cond, seed, q_train, R_all, class_d_rewrites, args.n_dupes, tokenizer
            )
            out_dir = f"adapters/i488_{cid}_seed{seed}"
            logger.info(
                "Training cond=%s seed=%d lr=%s r=%d a=%d epochs=%d fracs=%s",
                cid,
                seed,
                args.lr,
                args.lora_r,
                args.lora_alpha,
                args.epochs,
                args.fracs,
            )
            cfg = TrainLoraConfig(
                gpu_id=args.gpu_id,
                epochs=args.epochs,
                lr=args.lr,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                batch_size=4,
                grad_accum=4,
                max_length=2048,
                seed=seed,
                run_name=f"i488_{cid}_seed{seed}",
                report_to="wandb",
                save_strategy="no",
                marker_only_loss=True,
                marker_text=MARKER_TEXT,
                marker_tail_tokens=0,
                marker_suppress_at_post_response_slot=True,
                marker_im_end_token_id=IM_END_TOKEN_ID,
                # The FractionAdapterSaveCallback uploads each frac; disable the
                # default end-of-train HF upload to avoid double-uploading the
                # final frac.
                hf_upload=False,
                hf_repo=HF_MODEL_REPO,
            )

            callback = FractionAdapterSaveCallback(
                target_fractions=list(args.fracs),
                out_base=Path("adapters"),
                hf_repo=HF_MODEL_REPO,
                cond_cid=cid,
                seed=seed,
            )

            _, train_loss = train_lora(
                BASE_MODEL,
                str(train_path),
                out_dir,
                cfg=cfg,
                callbacks=[callback],
            )
            logger.info(
                "TRAIN DONE cond=%s seed=%d loss=%.4f saved_fracs=%s",
                cid,
                seed,
                train_loss,
                sorted(callback.fired),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
