# ruff: noqa: RUF002, RUF003
"""Issue #489 Phase 2 (smoke) + Phase 3 (sweep) — train one LoRA per union context.

Plan v5 §4.3 + §4.4 + §4.6 + §5.

Forked BYTE-FOR-BYTE from ``scripts/i474_phase23_train.py``'s ``loc`` arm for the
loss-slot fix (``MarkerOnlyDataCollator(tail_tokens=0,
suppress_at_post_response_slot=True, im_end_token_id=151645)``), with these
divergences from #474:

  - 24 union contexts (16 ICL + 8 SP) instead of 16 #406 conditions.
  - 23 contrastive bystanders per source (rotated across BOTH context types).
  - 150 positive + 150 negative rows per cond (#474 has 300+300).
  - 3 epochs with per-fraction checkpoint saves at frac ∈ {0.10, 0.25, 0.50,
    1.00, 2.00, 3.00} — sub-epoch + multi-epoch granularity, NOT the #474
    save-strategy="epoch" pattern.
  - LoRA r=16, α=32 (#474 used r=32, α=64).
  - lr=2e-6 (#474 used 1e-5).
  - max_length=4096 (ICL blocks K=4 + Q + R + marker fit; #474 used 2048).
  - seed=42 ONLY (v5 single-seed descope).

Asserts at launch (per CLAUDE.md marker-leakage-measurement.md):
  - ``tokenizer.encode(' ※', add_special_tokens=False) == [83399]``
  - ``tokenizer.convert_tokens_to_ids('<|im_end|>') == 151645``

CLI (smoke == sweep with --conds {IK01,IK13,SP01,SP04}, plan §4.5):
    uv run python scripts/i489_phase23_train.py --conds IK01 --gpu-id 0
    uv run python scripts/i489_phase23_train.py --conds IK01 IK13 SP01 SP04
    uv run python scripts/i489_phase23_train.py --conds IK02 --gpu-id 1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, TrainerCallback

from explore_persona_space.experiments.i406_conditions import MARKER_ID, MARKER_TEXT
from explore_persona_space.experiments.i460_data import load_q_train_answers
from explore_persona_space.experiments.i489_contexts import (
    UNION_BY_CID,
    UNION_CONTEXTS,
    build_messages_for_context,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

logger = logging.getLogger("i489.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_R_PATH_PREFIX = "issue489_union_panel/on_policy_R"
LOCAL_DATA_DIR = Path("data/issue_489")
TRAIN_ROW_DIR = Path("data/issue_489/train_rows")
M5_OUT_DIR = Path("eval_results/issue_489/train_diag")

# Plan v5 §4.3 + §5: 150 positive rows = 30 Q × 5 dupes.
N_DUPES_POS = 5
# Negatives rotate across 23 bystanders for ~6-7 per bystander * 23 = ~150 total.
N_NEG_PER_BYSTANDER = 7  # 23 × 7 = 161 → trimmed to 150
N_NEG_TOTAL_TARGET = 150
N_BYSTANDERS_EXPECTED = 23  # 24 union contexts - 1 source

# Plan §5: per-fraction checkpoints at sub-epoch + multi-epoch boundaries.
# fracs are multiples of one epoch (0.10 = 10% of one epoch; 3.00 = end of
# training at 3 epochs total).
DEFAULT_FRACS: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00, 2.00, 3.00)


def _load_R(split: str) -> dict[str, dict[str, dict]]:
    """Load R artifact from local disk; fall back to HF data repo."""
    local = LOCAL_DATA_DIR / f"R_{split}.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_{split}.json",
            revision="main",
        )
        shutil.copyfile(downloaded, local)
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(
                f"HF download claimed success but {local} is missing/empty (source {downloaded})."
            )
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i489_v1":
        raise AssertionError(
            f"R_{split}.json schema_version={payload.get('schema_version')!r}, expected 'i489_v1'."
        )
    return payload["completions"]


def _build_positive_rows(cid: str, q_train_answers: dict[str, str], R_train, tokenizer):
    ctx = UNION_BY_CID[cid]
    questions = sorted(q_train_answers.keys())
    if len(questions) != 30:
        raise AssertionError(f"Expected 30 Q_train questions, got {len(questions)}.")
    if cid not in R_train:
        raise AssertionError(f"R_train missing context {cid!r}.")

    rows: list[dict] = []
    for q in questions:
        if q not in R_train[cid]:
            raise AssertionError(f"R_train[{cid}] missing q={q!r}.")
        R = R_train[cid][q]["response_text"]
        completion_text = f"{R}{MARKER_TEXT}"
        messages = build_messages_for_context(ctx, q)
        row = {
            "prompt": messages,
            "completion": [{"role": "assistant", "content": completion_text}],
        }
        for _ in range(N_DUPES_POS):
            rows.append(row)

    # Tokenization sanity (first 2 rows): MARKER_ID present exactly once.
    for row in rows[:2]:
        full_messages = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        if marker_count != 1:
            raise AssertionError(
                f"cid={cid}: encoded POSITIVE row has {marker_count} marker "
                f"positions, expected 1. First 80 tokens: {ids[:80]}"
            )
    return rows


def _build_negative_rows(  # noqa: C901 - per-bystander balance trim is deliberately verbose (M-f)
    cid: str, q_train_answers: dict[str, str], R_train, tokenizer, rng: np.random.Generator
):
    """Rotate ~7 Q each across the 23 bystander contexts (mix of ICL + SP)."""
    questions = sorted(q_train_answers.keys())
    bystander_cids = [c.cid for c in UNION_CONTEXTS if c.cid != cid]
    if len(bystander_cids) != N_BYSTANDERS_EXPECTED:
        raise AssertionError(
            f"Expected {N_BYSTANDERS_EXPECTED} bystanders, got {len(bystander_cids)}"
        )

    rows: list[dict] = []
    for cj_id in bystander_cids:
        cj = UNION_BY_CID[cj_id]
        if cj_id not in R_train:
            raise AssertionError(f"R_train missing bystander {cj_id!r}.")
        n_sample = min(N_NEG_PER_BYSTANDER, len(questions))
        sampled = rng.choice(questions, size=n_sample, replace=False)
        for q in sampled:
            R_j = R_train[cj_id][str(q)]["response_text"]
            messages = build_messages_for_context(cj, str(q))
            row = {
                "prompt": messages,
                "completion": [{"role": "assistant", "content": R_j}],
                "_neg_source_i": cid,
                "_neg_bystander_j": cj_id,
            }
            rows.append(row)

    # Round-2 fix M-f: trim per-bystander to enforce EVEN coverage. The old
    # ``rng.permutation(...)[:N_NEG_TOTAL_TARGET]`` randomly dropped rows
    # without regard to bystander, which can leave some bystanders with 0-2
    # rows and others with 8-9; that biases the contrastive signal. Instead:
    # compute the target rows-per-bystander = N_NEG_TOTAL_TARGET // n_bystanders,
    # then ROUND-ROBIN extra rows across bystanders to hit exactly
    # N_NEG_TOTAL_TARGET (if a bystander has fewer rows than the cap, the
    # remainder spreads across the rest evenly).
    if len(rows) > N_NEG_TOTAL_TARGET:
        # Bucket rows by bystander first.
        from collections import defaultdict as _dd

        by_bystander: dict[str, list[dict]] = _dd(list)
        for r in rows:
            by_bystander[r["_neg_bystander_j"]].append(r)
        bystanders = sorted(by_bystander.keys())
        n_b = len(bystanders)
        # Initial target = uniform per-bystander cap.
        per_b_cap = N_NEG_TOTAL_TARGET // n_b
        extras = N_NEG_TOTAL_TARGET - per_b_cap * n_b
        kept: list[dict] = []
        leftover_capacity: list[tuple[str, int]] = []
        for k, b in enumerate(bystanders):
            available = by_bystander[b]
            take = min(per_b_cap + (1 if k < extras else 0), len(available))
            kept.extend(available[:take])
            if len(available) > take:
                leftover_capacity.append((b, len(available) - take))
        # If some bystanders fell short, redistribute the shortfall round-robin
        # across bystanders that have leftover capacity.
        shortfall = N_NEG_TOTAL_TARGET - len(kept)
        i = 0
        while shortfall > 0 and leftover_capacity:
            b, cap = leftover_capacity[i % len(leftover_capacity)]
            if cap > 0:
                # Find the next un-taken row for this bystander.
                taken_set = {id(r) for r in kept}
                for r in by_bystander[b]:
                    if id(r) not in taken_set:
                        kept.append(r)
                        leftover_capacity[i % len(leftover_capacity)] = (b, cap - 1)
                        shortfall -= 1
                        break
            i += 1
            if i > 10 * len(leftover_capacity) + N_NEG_TOTAL_TARGET:
                break  # defensive infinite-loop guard
        rows = kept[:N_NEG_TOTAL_TARGET]

    # Tokenization sanity (first 2 negative rows): MARKER_ID absent, <|im_end|> present.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise AssertionError("tokenizer cannot resolve <|im_end|>")
    for row in rows[:2]:
        full = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids.count(MARKER_ID) != 0:
            raise AssertionError(
                f"NEGATIVE row contains MARKER_ID; cond_i={cid} cj={row['_neg_bystander_j']}"
            )
        if im_end_id not in ids:
            raise AssertionError(
                f"NEGATIVE row has no <|im_end|>; cond_i={cid} cj={row['_neg_bystander_j']} "
                f"tail ids: {ids[-10:]}"
            )
    return rows


def _write_rows_jsonl(rows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_post_response_slot(tokenizer, prompt_messages, full_ids, im_end_id: int) -> int:
    """First <|im_end|> at index >= P (assistant-turn terminator)."""
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    P = len(prompt_ids)
    if full_ids[:P] != prompt_ids:
        raise RuntimeError(
            "prompt-only encoding is not a strict prefix of the full row encoding "
            f"(chat-template drift). prompt_ids[:5]={prompt_ids[:5]}, "
            f"full_ids[:5]={full_ids[:5]}, P={P}"
        )
    slot = next((i for i in range(P, len(full_ids)) if full_ids[i] == im_end_id), None)
    if slot is None:
        raise RuntimeError(
            f"no <|im_end|> (id={im_end_id}) found at index >= P={P} in row of "
            f"length {len(full_ids)}; tail ids: {full_ids[-10:]}"
        )
    first_im_end = next((i for i, t in enumerate(full_ids) if t == im_end_id), None)
    if first_im_end is None or slot <= first_im_end:
        raise RuntimeError(
            "post-response slot at or before the first <|im_end|> in the row: "
            f"slot={slot}, first_im_end={first_im_end}, P={P}"
        )
    return slot


class PerFractionAdapterUploadCallback(TrainerCallback):
    """Save + HF-upload the adapter at each frac in DEFAULT_FRACS.

    Differs from #474's PerEpochAdapterHFUploadCallback by saving at sub-epoch
    fractional step boundaries derived from the SFTTrainer's total step count.
    Plan v5 §5: fracs ∈ {0.10, 0.25, 0.50, 1.00, 2.00, 3.00}, where frac is in
    units of one epoch. Total epochs = 3; thus frac=3.00 = end-of-training.

    Resolves which Trainer step each frac corresponds to in ``on_train_begin``
    (steps_per_epoch is unknown until then), then fires the save at ``on_step_end``.

    Persists the adapter to HF under
    ``adapters/i489_{cid}_seed{S}_frac{F:.2f}`` per CLAUDE.md upload-policy
    (fail-loud on verification failure). Uses an UPLOAD_ALLOWLIST so the heavy
    ``optimizer.pt`` / ``rng_state.pth`` / ``scheduler.pt`` /
    ``trainer_state.json`` / ``training_args.bin`` are NEVER copied to HF.
    Reaps the source ``checkpoint-{step}/`` after verified upload to stay under
    the MooseFS ~130 GB pod quota.
    """

    UPLOAD_ALLOWLIST: tuple[str, ...] = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
        "README.md",
    )

    def __init__(
        self,
        cid: str,
        seed: int,
        output_dir: str,
        fracs: tuple[float, ...] = DEFAULT_FRACS,
        hf_repo: str = HF_MODEL_REPO,
    ):
        self.cid = cid
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.fracs = fracs
        self.hf_repo = hf_repo
        self._step_targets: dict[int, float] = {}
        self._uploaded_fracs: set[float] = set()

    def on_train_begin(self, args, state, control, **kwargs):
        steps_per_epoch_attr = getattr(state, "max_steps", 0)
        # SFTTrainer fills state.max_steps with total_steps after init; per-epoch
        # = max_steps / args.num_train_epochs. Fall back to derived computation.
        n_epochs = args.num_train_epochs or 3
        if steps_per_epoch_attr and n_epochs:
            steps_per_epoch = steps_per_epoch_attr / float(n_epochs)
        else:
            steps_per_epoch = 1  # defensive; on_step_end will skip if no match
        for f in self.fracs:
            target_step = max(1, round(steps_per_epoch * f))
            # Clamp to the last training step at 3.00.
            if steps_per_epoch_attr:
                target_step = min(target_step, steps_per_epoch_attr)
            self._step_targets[target_step] = f
        logger.info(
            "PerFracUpload cid=%s seed=%d: targets=%s (max_steps=%s, n_epochs=%s)",
            self.cid,
            self.seed,
            {f: s for s, f in self._step_targets.items()},
            steps_per_epoch_attr,
            n_epochs,
        )

    def on_step_end(self, args, state, control, **kwargs):
        target_frac = self._step_targets.get(state.global_step)
        if target_frac is None or target_frac in self._uploaded_fracs:
            return control
        # Tell the Trainer to save at this step (HF Trainer will run on_save
        # before our manual upload).
        # Force the save now: manually invoke Trainer.save_model via control.should_save.
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        target_frac = self._step_targets.get(state.global_step)
        if target_frac is None or target_frac in self._uploaded_fracs:
            return
        ckpt_dir = self.output_dir / f"checkpoint-{state.global_step}"
        adapter_file = ckpt_dir / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise RuntimeError(
                f"Frac {target_frac:.2f} (step {state.global_step}): expected adapter at "
                f"{ckpt_dir} after save, none found. Refusing to silently skip."
            )
        upload_dir = self._stage_clean(ckpt_dir, target_frac)
        path_in_repo = f"adapters/i489_{self.cid}_seed{self.seed}_frac{target_frac:.2f}"
        os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = self.hf_repo
        os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = path_in_repo

        from explore_persona_space.orchestrate.hub import upload_model

        hub_path = upload_model(str(upload_dir), repo_id=self.hf_repo, path_in_repo=path_in_repo)
        if not hub_path:
            raise RuntimeError(
                f"upload_model returned empty (verify failed) for {path_in_repo}; "
                "refusing to continue per upload-policy fail-loud."
            )
        self._uploaded_fracs.add(target_frac)
        logger.info(
            "PerFracUpload cid=%s seed=%d frac=%.2f uploaded -> %s",
            self.cid,
            self.seed,
            target_frac,
            hub_path,
        )
        # Reap: delete the staged + source checkpoint after verified upload.
        for p, label in ((upload_dir, "upload bundle"), (ckpt_dir, "checkpoint dir")):
            try:
                if p.exists():
                    shutil.rmtree(p)
            except OSError as e:
                raise RuntimeError(
                    f"PerFracUpload: FAILED to reap {label} {p} after verified upload: {e}."
                ) from e

    def _stage_clean(self, ckpt_dir: Path, target_frac: float) -> Path:
        upload_dir = self.output_dir / f"_upload_frac{target_frac:.2f}"
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=False)
        copied: list[str] = []
        for fname in self.UPLOAD_ALLOWLIST:
            src = ckpt_dir / fname
            if src.exists():
                shutil.copy2(src, upload_dir / fname)
                copied.append(f"ckpt:{fname}")
        # tokenizer files written once at init to output_dir
        for fname in self.UPLOAD_ALLOWLIST:
            src = self.output_dir / fname
            dst = upload_dir / fname
            if src.exists() and not dst.exists() and src.is_file():
                shutil.copy2(src, dst)
                copied.append(f"out:{fname}")
        if "adapter_model.safetensors" not in {c.split(":", 1)[1] for c in copied}:
            raise RuntimeError(
                f"staged upload bundle missing adapter_model.safetensors at {ckpt_dir}"
            )
        return upload_dir


class NegRowSuppressionDifficultyCallback(TrainerCallback):
    """M5 identifiability hook — mean negative-row training loss per (i, j).

    Inherited from #474; light port for 23 bystanders instead of 15.
    """

    def __init__(self, tokenizer, neg_rows, im_end_id: int, cid: str, out_dir: Path):
        self.tokenizer = tokenizer
        self.neg_rows = neg_rows
        self.im_end_id = im_end_id
        self.cid = cid
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._cached_rows: list[tuple[list[int], int, str, str]] = []
        for row in neg_rows:
            messages = list(row["prompt"]) + list(row["completion"])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            slot = _resolve_post_response_slot(tokenizer, list(row["prompt"]), ids, im_end_id)
            if ids[slot] != im_end_id:
                raise RuntimeError(
                    f"M5 slot picker returned slot={slot} but ids[slot]={ids[slot]} "
                    f"!= im_end_id={im_end_id}"
                )
            self._cached_rows.append((ids, slot, row["_neg_source_i"], row["_neg_bystander_j"]))

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        per_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for ids, slot, source_i, bystander_j in self._cached_rows:
                input_ids = torch.tensor([ids[: slot + 1]], device=model.device, dtype=torch.long)
                out = model(input_ids=input_ids)
                logp = F.log_softmax(out.logits[0, slot - 1], dim=-1)
                loss_at_slot = -float(logp[self.im_end_id].item())
                per_pair[(source_i, bystander_j)].append(loss_at_slot)
        model.train()
        agg = {f"{i}__{j}": float(np.mean(losses)) for (i, j), losses in per_pair.items()}
        epoch = state.epoch if state.epoch is not None else float("nan")
        out_path = self.out_dir / f"suppression_difficulty_{self.cid}_step{state.global_step}.json"
        out_path.write_text(
            json.dumps(
                {
                    "source_i": self.cid,
                    "epoch": epoch,
                    "global_step": state.global_step,
                    "per_bystander_mean_neg_loss": agg,
                },
                indent=2,
            )
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conds", nargs="+", required=True, help="One or more union cids.")
    ap.add_argument("--epochs", type=int, default=3, help="Plan v5 §5: 3 epochs.")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-6, help="Plan v5 §11.")
    ap.add_argument("--seed", type=int, default=42, help="Single-seed (v5 descope).")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Tiny slice: 4 positive rows, 2 negative rows, 1 epoch, no HF upload. "
            "Used by the local CPU end-to-end smoke run."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != 151645:
        raise AssertionError(f"<|im_end|> id drift: got {im_end_id}, expected 151645")

    unknown = [c for c in args.conds if c not in UNION_BY_CID]
    if unknown:
        raise ValueError(f"--conds {unknown} not in union set {list(UNION_BY_CID)}.")

    q_train_answers = load_q_train_answers()

    # Smoke: synthesize tiny R from a base-model echo (no vLLM needed).
    if args.smoke:
        questions = sorted(q_train_answers.keys())[:4]
        R_train = {
            cid: {q: {"response_text": "Smoke response."} for q in questions}
            for cid in UNION_BY_CID
        }
        # Need the bystander cids in R too.
    else:
        R_train = _load_R("train")

    for cid in args.conds:
        # M-h skip-if-already-trained: check HF for every required (cid, frac)
        # adapter. If all 6 fracs exist, skip training entirely. Saves ~2 GPU-h
        # when smoke phase + sweep phase both run on the 4 smoke cids, and
        # avoids the HF-upload race condition.
        if not args.smoke:
            from huggingface_hub import HfApi as _HfApi

            api = _HfApi()
            try:
                repo_files = set(api.list_repo_files(repo_id=HF_MODEL_REPO))
            except Exception as e:
                logger.warning(
                    "M-h skip check: list_repo_files failed: %s — training cid=%s anyway", e, cid
                )
                repo_files = set()
            required = {
                f"adapters/i489_{cid}_seed{args.seed}_frac{f:.2f}/adapter_model.safetensors"
                for f in DEFAULT_FRACS
            }
            missing = sorted(required - repo_files)
            if not missing:
                logger.info(
                    "M-h skip cid=%s seed=%d: all %d frac adapters already on HF; skipping train.",
                    cid,
                    args.seed,
                    len(required),
                )
                continue
            elif len(missing) < len(required):
                logger.info(
                    "M-h partial cid=%s seed=%d: %d/%d frac adapters present on HF; "
                    "re-running full train (the PerFracCallback will re-upload all).",
                    cid,
                    args.seed,
                    len(required) - len(missing),
                    len(required),
                )

        cond_offset = int(hashlib.sha256(cid.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(args.seed + cond_offset % 10_000)

        if args.smoke:
            # 4 positives, 2 negatives only.
            questions = sorted(q_train_answers.keys())[:2]
            ctx = UNION_BY_CID[cid]
            pos_rows = []
            for q in questions:
                R = R_train[cid][q]["response_text"]
                pos_rows.append(
                    {
                        "prompt": build_messages_for_context(ctx, q),
                        "completion": [{"role": "assistant", "content": f"{R}{MARKER_TEXT}"}],
                    }
                )
            # 1 bystander (next cid in the union order, NOT the source)
            bystander_cid = next(c.cid for c in UNION_CONTEXTS if c.cid != cid)
            cj = UNION_BY_CID[bystander_cid]
            neg_rows = []
            for q in questions:
                R_j = R_train[bystander_cid][q]["response_text"]
                neg_rows.append(
                    {
                        "prompt": build_messages_for_context(cj, q),
                        "completion": [{"role": "assistant", "content": R_j}],
                        "_neg_source_i": cid,
                        "_neg_bystander_j": bystander_cid,
                    }
                )
            all_rows = pos_rows + neg_rows
        else:
            pos_rows = _build_positive_rows(cid, q_train_answers, R_train, tokenizer)
            neg_rows = _build_negative_rows(cid, q_train_answers, R_train, tokenizer, rng)
            all_rows = pos_rows + neg_rows
            ratio = len(pos_rows) / max(1, len(neg_rows))
            logger.info(
                "cid=%s rows: %d pos + %d neg = %d (pos:neg = %.2f:1)",
                cid,
                len(pos_rows),
                len(neg_rows),
                len(all_rows),
                ratio,
            )

        train_path = TRAIN_ROW_DIR / f"i489_{cid}_seed{args.seed}.jsonl"
        _write_rows_jsonl(all_rows, train_path)

        out_dir = f"adapters/i489_{cid}_seed{args.seed}"
        callbacks: list[TrainerCallback] = []
        if not args.smoke:
            callbacks.append(
                PerFractionAdapterUploadCallback(
                    cid=cid,
                    seed=args.seed,
                    output_dir=out_dir,
                )
            )
            callbacks.append(
                NegRowSuppressionDifficultyCallback(
                    tokenizer=tokenizer,
                    neg_rows=neg_rows,
                    im_end_id=im_end_id,
                    cid=cid,
                    out_dir=M5_OUT_DIR,
                )
            )

        epochs = 1 if args.smoke else args.epochs
        cfg = TrainLoraConfig(
            gpu_id=args.gpu_id,
            epochs=epochs,
            lr=args.lr,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            batch_size=2,
            grad_accum=8,
            max_length=args.max_length,
            seed=args.seed,
            run_name=f"i489_{cid}_seed{args.seed}",
            report_to="wandb" if not args.smoke else "none",
            save_strategy="steps" if not args.smoke else "no",
            save_total_limit=1,
            warmup_ratio=0.03,  # minor: plan §11 specifies 0.03 (was 0.05 default).
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,
            marker_suppress_at_post_response_slot=True,
            marker_im_end_token_id=im_end_id,
            hf_upload=not args.smoke,
            hf_repo=HF_MODEL_REPO,
            hf_path_in_repo=f"adapters/i489_{cid}_seed{args.seed}",
        )

        out_path, train_loss = train_lora(
            BASE_MODEL, train_path, out_dir, cfg=cfg, callbacks=callbacks
        )
        logger.info(
            "TRAIN DONE cid=%s seed=%d loss=%.4f -> %s",
            cid,
            args.seed,
            train_loss,
            out_path,
        )


if __name__ == "__main__":
    main()
