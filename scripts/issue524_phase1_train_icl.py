"""Phase 1 -- train 16 ICL marker adapters (Qwen-2.5-7B-Instruct, LoRA r=32 a=64
dropout=0.0 target=qkvo, lr=1e-5 cosine warmup_ratio=0.05 bf16, ep1, 300+300
contrastive, marker=' ※' id 83399, MarkerOnlyDataCollator(tail_tokens=0,
suppress_at_post_response_slot=True, im_end_token_id=151645)).

Issue #524 plan v1 §Phase 1. Reuses ``train_lora`` /
``MarkerOnlyDataCollator`` / ``train_lora.marker_im_end_token_id``
exactly as #474 used them; the only differences vs #474:

  - The PROMPT shape: ICL block (4 (Q, A) demonstrations) prepended as a
    single user turn, NO system prompt -- vs #474's 16 instruction
    contexts where Class A has a system prompt etc.

  - The R generation: frozen R_train is generated UNDER each ICL context
    via on-policy Qwen-2.5-7B-Instruct greedy (we ALSO load Qwen here in
    a one-shot generator step BEFORE training, since #474's frozen R was
    generated under the 16 instruction contexts and is NOT reusable for
    ICL contexts).

  - The negative-persona set: 3 OTHER ICL contexts + 1 bare default
    assistant (=4 negatives), 75 questions each = 300 negatives. Matches
    the plan §11 "always include default; 4 negatives close to source"
    rule and the contrastive-negatives rule's "always include the
    default assistant" requirement.

The per-epoch HF adapter upload + checkpoint reaping mirror #474's
``PerEpochAdapterHFUploadCallback``. Adapter HF path:
``adapters/i524_icl_{cid}_ep1``.

The marker-band-stop callback (default ON in TrainLoraConfig) auto-stops
the run the first time source log P(marker) - base enters [5, 12] nats,
giving us the off-saturation ep1-equivalent checkpoint deterministically
(plan §0.2 Gate A + B + ``.claude/rules/marker-training-recipe.md``).

CLI (smoke == sweep with --conds IK01 --epochs 1, plan §"Smoke
architecture parity" -- UNIFIED):
    # Smoke: train one ICL cell end-to-end.
    uv run python scripts/issue524_phase1_train_icl.py --conds IK01

    # Sweep cell (one process per GPU shard).
    uv run python scripts/issue524_phase1_train_icl.py --conds IK02 --gpu-id 1

    # CPU build-time smoke (no training -- just verify pool construction +
    # tokenization invariants; pairs with the GPU-bound carve-out below).
    uv run python scripts/issue524_phase1_train_icl.py --conds IK01 --build-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer, TrainerCallback

load_dotenv()

# Re-use #474's per-epoch adapter HF uploader (it's parameterized on arm + cid;
# we set arm="icl" so the HF path becomes adapters/i524_icl_{cid}_ep{N}
# instead of #474's adapters/i474_loc_{cid}_ep{N}).
# Note: scripts/ is not a package, so we add it to sys.path explicitly. The
# dispatcher does the same; this duplication keeps the per-phase script
# directly runnable for ad-hoc invocation.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_q_train_answers,
)
from explore_persona_space.experiments.i524_icl_contexts import (  # noqa: E402
    ICL_CONTEXTS,
    ICL_CONTEXTS_BY_ID,
    build_icl_messages,
)

_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from i474_phase23_train import (  # noqa: E402
    NegRowSuppressionDifficultyCallback,
)

logger = logging.getLogger("i524.phase1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_ICL_R_PREFIX = "issue524_unified_panel/icl_R"
LOCAL_DATA_DIR = Path("data/issue_524")
TRAIN_ROW_DIR = Path("data/issue_524/train_rows")
ICL_BLOCKS_PATH = Path("eval_results/issue_524/icl_contexts/i524_icl_blocks.json")
M5_OUT_DIR = Path("eval_results/issue_524/train_diag")
SATURATION_DIR = Path("eval_results/issue_524/phase1/saturation")

# Plan §Phase 1 / §11 row count.
# 300 positives = N_DUPES_POS=10 × 30 Q_train (matches #474 / #460 round 3).
# 300 negatives = 4 negative personas × 75 Q_train each.
N_DUPES_POS = 10
N_NEG_PER_BYSTANDER = 75
N_NEG_BYSTANDERS = 4  # 3 OTHER ICL + 1 bare default assistant
N_POSITIVES = 30 * N_DUPES_POS  # 300
N_NEGATIVES = N_NEG_PER_BYSTANDER * N_NEG_BYSTANDERS  # 300


def _haiku_demos_for(icl_blocks: dict, cid: str) -> list[dict]:
    """Return the (q, a) demos for one ICL cid, asserting they're well-formed."""
    if cid not in icl_blocks:
        raise RuntimeError(
            f"ICL block for {cid!r} missing in i524_icl_blocks.json -- "
            f"run scripts/issue524_phase0_2_build_icl_blocks.py first."
        )
    demos = icl_blocks[cid]["demos"]
    if not isinstance(demos, list) or len(demos) != 4:
        raise RuntimeError(
            f"ICL block for {cid!r} has {len(demos) if isinstance(demos, list) else 'not-a-list'} demos, expected 4"
        )
    for d in demos:
        if not (isinstance(d, dict) and "q" in d and "a" in d):
            raise RuntimeError(f"Bad demo shape in ICL block {cid!r}: {d!r}")
    return demos


def _generate_icl_R_train(
    icl_blocks: dict, cids: list[str], q_train: list[str], gpu_id: int
) -> dict[str, dict[str, dict]]:
    """Generate frozen R_train per (ICL context, Q_train) on Qwen-2.5-7B-Instruct.

    The marker-leakage measurement recipe (.claude/rules/marker-leakage-measurement.md):
    R = base_model.generate(T(q)), greedy (temp=0), capped at max_new_tokens=1024.
    Persisted to HF so subsequent cells can re-use without re-generating.

    Returns: ``{cid: {q: {"response_text": str}}}``.
    """
    # Try HF cache first -- saves a vLLM init if every cid is already up there.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    cache: dict[str, dict[str, dict]] = {}
    missing_cids: list[str] = []
    for cid in cids:
        hf_path = f"{HF_ICL_R_PREFIX}/R_train_{cid}.json"
        try:
            local = hf_hub_download(
                repo_id=HF_DATA_REPO, repo_type="dataset", filename=hf_path, revision="main"
            )
            payload = json.loads(Path(local).read_text())
            if payload.get("schema_version") != "i524_v1":
                raise AssertionError(
                    f"R_train_{cid}.json schema_version={payload.get('schema_version')!r}, "
                    "expected 'i524_v1'"
                )
            cache[cid] = payload["completions"]
            logger.info("R_train cache hit for cid=%s (%d q's)", cid, len(cache[cid]))
        except EntryNotFoundError:
            missing_cids.append(cid)
        except Exception as e:
            # Fail-loud on auth/network distinguished from missing-file.
            raise RuntimeError(
                f"HF probe for R_train_{cid}.json failed (NOT a missing-file "
                f"error -- likely auth/network): {e}"
            ) from e

    if not missing_cids:
        return cache

    # Generate the missing cids in a single vLLM batched call.
    logger.info("Generating R_train via vLLM for missing cids: %s", missing_cids)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=4096,
    )
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=1024, seed=42)

    for cid in missing_cids:
        demos = _haiku_demos_for(icl_blocks, cid)
        prompts = [
            tokenizer.apply_chat_template(
                build_icl_messages(demos, q), tokenize=False, add_generation_prompt=True
            )
            for q in q_train
        ]
        outputs = llm.generate(prompts, sp)
        completions = {
            q: {"response_text": out.outputs[0].text} for q, out in zip(q_train, outputs)
        }
        cache[cid] = completions
        # Persist locally + push to HF for next time.
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        local_path = LOCAL_DATA_DIR / f"R_train_{cid}.json"
        local_path.write_text(
            json.dumps(
                {"schema_version": "i524_v1", "cid": cid, "completions": completions}, indent=2
            )
        )
        logger.info("R_train wrote %s (%d q's)", local_path, len(completions))

    # Upload missing cids to HF data repo (fail-loud per upload-policy.md).
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for cid in missing_cids:
            local_path = LOCAL_DATA_DIR / f"R_train_{cid}.json"
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{HF_ICL_R_PREFIX}/R_train_{cid}.json",
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                revision="main",
            )
            logger.info("Pushed R_train_%s.json to HF %s", cid, HF_DATA_REPO)
    except Exception as e:
        raise RuntimeError(
            f"R_train HF upload failed (refusing to continue -- per "
            f"upload-policy.md fail-loud): {e}"
        ) from e

    return cache


def _build_positive_rows(
    cid: str,
    q_train: list[str],
    R_train: dict[str, dict[str, dict]],
    icl_blocks: dict,
    tokenizer,
) -> list[dict]:
    """Build the 300 positive rows for one ICL cell.

    Row = (ICL block(demos) + Q_train, R_train + ' ※').
    Loss is masked by MarkerOnlyDataCollator(tail_tokens=0) to ONLY the
    marker token positions + EOS.
    """
    if cid not in R_train:
        raise RuntimeError(f"R_train missing cid={cid!r}")
    demos = _haiku_demos_for(icl_blocks, cid)
    rows: list[dict] = []
    for q in q_train:
        if q not in R_train[cid]:
            raise RuntimeError(f"R_train[{cid!r}] missing q={q!r}")
        R = R_train[cid][q]["response_text"]
        completion_text = f"{R}{MARKER_TEXT}"
        messages = build_icl_messages(demos, q)
        row = {
            "prompt": messages,
            "completion": [{"role": "assistant", "content": completion_text}],
        }
        for _ in range(N_DUPES_POS):
            rows.append(row)

    # Marker-token sanity (first 2 rows): MARKER_ID present exactly once.
    for row in rows[:2]:
        full_msgs = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        n = ids.count(MARKER_ID)
        if n != 1:
            raise AssertionError(
                f"cid={cid}: POSITIVE row has {n} marker positions (expected 1). "
                f"First 80 tokens: {ids[:80]}"
            )
    if len(rows) != N_POSITIVES:
        raise AssertionError(f"Expected {N_POSITIVES} positive rows, got {len(rows)}")
    return rows


def _pick_negatives(cid: str, rng: np.random.Generator) -> list[str | None]:
    """Pick 4 negative personas for the source ICL cid.

    Rule (plan §11 + .claude/rules/contrastive-negatives.md):
      - Always include the bare default assistant (None sentinel).
      - 3 OTHER ICL contexts sampled deterministically (sha256(cid) seed).

    Returns the negative-persona keys; ``None`` represents the bare
    default assistant (no system prompt, no ICL block).
    """
    all_other = [c.cid for c in ICL_CONTEXTS if c.cid != cid]
    picks = list(rng.choice(all_other, size=3, replace=False))
    return [None] + [str(p) for p in picks]


def _build_negative_rows(
    cid: str,
    q_train: list[str],
    R_train: dict[str, dict[str, dict]],
    icl_blocks: dict,
    tokenizer,
    rng: np.random.Generator,
) -> list[dict]:
    """Build the 300 negative rows for one ICL cell.

    Per negative persona (None = bare assistant, str = OTHER ICL cid):
      - Sample 75 Q_train questions.
      - Generate / cache the negative's frozen R_j under that persona's prompt.
      - Row = (T_j(q), R_j); NO marker. Tagged ``_neg_source_i`` /
        ``_neg_bystander_j`` for the M5 callback.
    """
    negatives = _pick_negatives(cid, rng)
    # We need R_train for EACH negative persona. If a negative is another
    # ICL cid, its R_train is generated alongside the source's (the
    # _generate_icl_R_train call above already covered all ICL cids if
    # the caller passed them in). If a negative is the bare default
    # assistant (``None``), we need R_train under no-system, plain
    # user-turn (= effectively #406 C1's prompt shape); we lazy-load
    # from HF or fall back to a quick on-demand generation.
    bare_R = _load_or_generate_bare_R(q_train)

    rows: list[dict] = []
    for cj in negatives:
        if cj is not None and cj not in R_train:
            raise RuntimeError(f"R_train missing negative cid={cj!r}")
        sampled = list(rng.choice(q_train, size=N_NEG_PER_BYSTANDER, replace=False))
        for q in sampled:
            if cj is None:
                R_j = bare_R[q]["response_text"]
                messages = [{"role": "user", "content": q}]
                bystander_label = "BARE_DEFAULT"
            else:
                R_j = R_train[cj][str(q)]["response_text"]
                demos_j = _haiku_demos_for(icl_blocks, cj)
                messages = build_icl_messages(demos_j, str(q))
                bystander_label = cj
            row = {
                "prompt": messages,
                "completion": [{"role": "assistant", "content": R_j}],
                "_neg_source_i": cid,
                "_neg_bystander_j": bystander_label,
            }
            rows.append(row)

    if len(rows) != N_NEGATIVES:
        raise AssertionError(f"Expected {N_NEGATIVES} negative rows, got {len(rows)}")

    # Tokenization sanity (first 2 negatives): MARKER_ID absent AND <|im_end|> present.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise AssertionError("tokenizer cannot resolve <|im_end|>")
    for row in rows[:2]:
        full = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids.count(MARKER_ID) != 0:
            raise AssertionError(
                f"NEGATIVE row contains MARKER_ID for cid={cid} bystander={row['_neg_bystander_j']!r}"
            )
        if im_end_id not in ids:
            raise AssertionError(
                f"NEGATIVE row has no <|im_end|> id={im_end_id} for cid={cid}; tail={ids[-10:]}"
            )
    return rows


def _load_or_generate_bare_R(q_train: list[str]) -> dict[str, dict]:
    """Get R for the bare default assistant (no system, plain user turn).

    Inherits from #460's R_train shape if available (cid='C1' in #406 is
    chat_template=True with no system prompt -- byte-equivalent prompt
    shape for our purposes). Falls back to a quick on-demand
    vLLM generation only if both #460's HF copy and our HF cache miss.

    Returns ``{q: {"response_text": str}}``.
    """
    # Quick HF probe -- if our prior bare cache exists, use it.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    hf_path = f"{HF_ICL_R_PREFIX}/R_train_BARE.json"
    try:
        local = hf_hub_download(
            repo_id=HF_DATA_REPO, repo_type="dataset", filename=hf_path, revision="main"
        )
        payload = json.loads(Path(local).read_text())
        return payload["completions"]
    except EntryNotFoundError:
        pass
    except Exception as e:
        raise RuntimeError(
            f"HF probe for R_train_BARE.json failed (NOT a missing-file error): {e}"
        ) from e

    # Fallback: try #460's R_train C1 cell -- byte-equivalent prompt shape.
    try:
        local = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename="issue460_marker_at_end/on_policy_R/R_train.json",
            revision="main",
        )
        payload = json.loads(Path(local).read_text())
        if "C1" in payload["completions"]:
            logger.info("Bare-R sourced from #460 R_train['C1'] (no-system chat-template)")
            return payload["completions"]["C1"]
    except Exception as e:
        logger.warning("Bare-R fallback to #460 R_train['C1'] failed: %s", e)

    raise RuntimeError(
        "Bare-default-assistant R not available on HF and #460 fallback failed. "
        "Either upload R_train_BARE.json to HF or extend Phase 1 to generate it "
        "(currently we expect it to already exist)."
    )


def _write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assert_rows_fit_max_length(rows: list[dict], tokenizer, max_length: int) -> None:
    """Build-time guard: every row must fit max_length under the chat template.

    Catches the silent #480 round-3 class where SFTConfig.max_length truncates
    the trailing <|im_end|> on long ICL+R rows -- the MarkerOnlyDataCollator
    then can't find its slot and crashes 2 min into Phase 1.
    """
    n_drop = 0
    for r in rows:
        full_msgs = list(r["prompt"]) + list(r["completion"])
        text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_length:
            n_drop += 1
            if n_drop <= 2:
                logger.error(
                    "Row exceeds max_length=%d (len=%d). First 50 tokens: %s ...",
                    max_length,
                    len(ids),
                    ids[:50],
                )
    if n_drop > 0:
        raise RuntimeError(
            f"build-time guard: {n_drop} rows exceed max_length={max_length}. "
            "Either raise max_length (currently the plan §11 default 2048) or "
            "shorten ICL demos / R."
        )


class _Icl524PerEpochAdapterHFUploadCallback(TrainerCallback):
    """Lightweight per-epoch HF adapter uploader, parameterized on cid only.

    Mirrors #474's PerEpochAdapterHFUploadCallback API but:
      - HF path is ``adapters/i524_icl_{cid}_ep{N}`` (NOT _loc_).
      - We only target ep1 by default (#524 trains 1 epoch under the
        band-stop). N=2..5 are skipped automatically because save_strategy
        is 'epoch' AND we only train 1 epoch.

    The callback also auto-reaps the local checkpoint dir + staged upload
    bundle (per #474 round-5 fix A; covers the MooseFS ~130 GB quota).
    """

    CHECKPOINT_EPOCHS_TO_UPLOAD: tuple[int, ...] = (1,)

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

    def __init__(self, cid: str, output_dir: str, hf_repo: str = HF_MODEL_REPO):
        self.cid = cid
        self.output_dir = Path(output_dir)
        self.hf_repo = hf_repo
        self._uploaded_epochs: set[int] = set()

    @staticmethod
    def _resolve_target_epoch(state_epoch: float | None) -> int | None:
        if state_epoch is None:
            return None
        candidate = round(state_epoch)
        if abs(state_epoch - candidate) > 0.05:
            return None
        if candidate in _Icl524PerEpochAdapterHFUploadCallback.CHECKPOINT_EPOCHS_TO_UPLOAD:
            return candidate
        return None

    def on_save(self, args, state, control, **kwargs):
        target_ep = self._resolve_target_epoch(state.epoch)
        if target_ep is None or target_ep in self._uploaded_epochs:
            return
        import shutil

        ckpt_dir = self.output_dir / f"checkpoint-{state.global_step}"
        if not (ckpt_dir / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"i524 per-epoch upload: adapter_model.safetensors missing under {ckpt_dir}"
            )
        # Stage clean bundle.
        upload_dir = self.output_dir / f"_upload_ep{target_ep}"
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=False)
        copied: list[str] = []
        for fname in self.UPLOAD_ALLOWLIST:
            for src_dir in (ckpt_dir, self.output_dir):
                src = src_dir / fname
                if src.exists() and not (upload_dir / fname).exists() and src.is_file():
                    shutil.copy2(src, upload_dir / fname)
                    copied.append(fname)
        if "adapter_model.safetensors" not in copied or "adapter_config.json" not in copied:
            raise RuntimeError(
                f"i524 per-epoch upload: missing required files in upload bundle; copied={copied}"
            )

        path_in_repo = f"adapters/i524_icl_{self.cid}_ep{target_ep}"
        os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = self.hf_repo
        os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = path_in_repo

        from explore_persona_space.orchestrate.hub import upload_model

        hub_path = upload_model(str(upload_dir), repo_id=self.hf_repo, path_in_repo=path_in_repo)
        if not hub_path:
            raise RuntimeError(
                f"i524 adapter upload returned empty path (verification failed) for "
                f"cid={self.cid} ep={target_ep}; refusing to continue (upload-policy.md fail-loud)."
            )
        self._uploaded_epochs.add(target_ep)
        logger.info(
            "i524 PerEpochAdapterHFUpload: cid=%s ep=%d -> %s (verified).",
            self.cid,
            target_ep,
            hub_path,
        )
        # Reap local copies (round-5 fix A).
        for path, label in ((upload_dir, "upload bundle"), (ckpt_dir, "checkpoint dir")):
            if path.exists():
                shutil.rmtree(path)
                logger.info("Reaped local %s %s (HF copy is source of truth)", label, path)


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
        help="One or more ICL context ids (e.g. IK01 IK02).",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "PHYSICAL GPU index (Hydra +gpu_id pattern). sft.py sets "
            "os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)."
        ),
    )
    ap.add_argument("--epochs", type=int, default=1, help="ep1 off-saturation per plan.")
    ap.add_argument("--lr", type=float, default=1e-5, help="#474 recipe.")
    ap.add_argument("--seed", type=int, default=42, help="Single seed.")
    ap.add_argument(
        "--save-strategy",
        default="epoch",
        choices=["epoch", "no"],
        help="Default 'epoch' triggers PerEpochAdapterHFUploadCallback.",
    )
    ap.add_argument(
        "--icl-blocks-path",
        type=Path,
        default=ICL_BLOCKS_PATH,
        help="Phase 0.2 output -- 16 ICL blocks JSON.",
    )
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="STOP after row construction + tokenization sanity (CPU smoke).",
    )
    args = ap.parse_args(argv)

    # MooseFS quota safety + adapter-persist contract.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    # Late-import sft so the dotenv + module-top stuff runs first.
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [MARKER_ID]:
        raise AssertionError(
            f"Marker {MARKER_TEXT!r} tokenizes to {encoded}, expected [{MARKER_ID}]"
        )
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != 151645:
        raise AssertionError(f"<|im_end|> id drift: got {im_end_id}, expected 151645")

    if not args.icl_blocks_path.exists():
        raise RuntimeError(f"ICL blocks missing at {args.icl_blocks_path}; run Phase 0.2 first.")
    icl_blocks = json.loads(args.icl_blocks_path.read_text())

    unknown = [c for c in args.conds if c not in ICL_CONTEXTS_BY_ID]
    if unknown:
        raise ValueError(f"--conds {unknown} not in ICL_CONTEXTS_BY_ID {list(ICL_CONTEXTS_BY_ID)}")

    q_train_answers = load_q_train_answers()
    # Q_train is the sorted list of #406 Q_train question strings (30 of them).
    q_train = sorted(q_train_answers.keys())
    if len(q_train) != 30:
        raise AssertionError(f"Expected 30 Q_train questions, got {len(q_train)}")

    # Need R_train for the source cid AND every chosen negative cid. To
    # avoid re-generating: generate R for ALL 16 ICL cids in one shot
    # (they're shared across the sweep), cached on HF. _build_only smoke
    # restricts to the requested cids to keep the CPU smoke fast.
    cids_for_R = args.conds if args.build_only else [c.cid for c in ICL_CONTEXTS]
    R_train = (
        _generate_icl_R_train(icl_blocks, cids_for_R, q_train, gpu_id=args.gpu_id)
        if not args.build_only
        else {}
    )

    for cid in args.conds:
        # Stable per-cid seed offset for negative sampling (#474 sha256 pattern).
        cond_offset = int(hashlib.sha256(cid.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(args.seed + cond_offset % 10_000)

        if args.build_only:
            # CPU smoke: build positives + negatives from the icl_blocks
            # alone (no R_train -- substitute a placeholder so tokenization
            # invariants exercise). The placeholder R is the ICL block's
            # FIRST demo answer; this is enough to validate marker-slot
            # invariants without needing a vLLM init.
            placeholder_R_train = {
                c: {q: {"response_text": icl_blocks[c]["demos"][0]["a"]} for q in q_train}
                for c in cids_for_R + [c2 for c2 in ICL_CONTEXTS_BY_ID if c2 != cid][:3]
            }
            # Also stub the bare-R via the same placeholder.
            globals_for_bare = {q: {"response_text": "bare placeholder."} for q in q_train}
            _orig_bare = _load_or_generate_bare_R
            try:
                # Monkey-patch the bare-R loader to return the stub (per-process scoped).
                # This is the standard CPU build-only smoke pattern from #474.
                globals()["_load_or_generate_bare_R"] = lambda _q: globals_for_bare
                pos_rows = _build_positive_rows(
                    cid, q_train, placeholder_R_train, icl_blocks, tokenizer
                )
                neg_rows = _build_negative_rows(
                    cid, q_train, placeholder_R_train, icl_blocks, tokenizer, rng
                )
            finally:
                globals()["_load_or_generate_bare_R"] = _orig_bare
        else:
            pos_rows = _build_positive_rows(cid, q_train, R_train, icl_blocks, tokenizer)
            neg_rows = _build_negative_rows(cid, q_train, R_train, icl_blocks, tokenizer, rng)

        all_rows = pos_rows + neg_rows
        if len(pos_rows) != N_POSITIVES or len(neg_rows) != N_NEGATIVES:
            raise AssertionError(
                f"cid={cid}: pos={len(pos_rows)} neg={len(neg_rows)} "
                f"(expected {N_POSITIVES} + {N_NEGATIVES})"
            )

        # Build-time guard for max_length (per CLAUDE.md feedback_cpu_build_time_guard_for_truncation).
        _assert_rows_fit_max_length(all_rows, tokenizer, max_length=2048)

        train_path = TRAIN_ROW_DIR / f"i524_icl_{cid}.jsonl"
        _write_jsonl(all_rows, train_path)
        logger.info(
            "cid=%s rows: %d pos + %d neg = %d (1:1 ratio) -> %s",
            cid,
            len(pos_rows),
            len(neg_rows),
            len(all_rows),
            train_path,
        )

        if args.build_only:
            logger.info(
                "--build-only: stopping after row construction + invariants for cid=%s", cid
            )
            continue

        out_dir = f"adapters/i524_icl_{cid}"
        callbacks: list[TrainerCallback] = []
        if args.save_strategy == "epoch":
            callbacks.append(_Icl524PerEpochAdapterHFUploadCallback(cid=cid, output_dir=out_dir))
        callbacks.append(
            NegRowSuppressionDifficultyCallback(
                tokenizer=tokenizer,
                neg_rows=neg_rows,
                im_end_id=im_end_id,
                arm="icl",
                cid=cid,
                out_dir=M5_OUT_DIR,
            )
        )

        cfg = TrainLoraConfig(
            gpu_id=args.gpu_id,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.0,  # Source: #474 i474_phase23_train.py:922
            batch_size=4,
            grad_accum=4,
            max_length=2048,
            warmup_ratio=0.05,  # Plan §11 -- TrainLoraConfig default, explicit for clarity.
            seed=args.seed,
            run_name=f"i524_icl_{cid}",
            report_to="wandb",
            save_strategy=args.save_strategy,
            save_total_limit=1,
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,
            marker_suppress_at_post_response_slot=True,  # #474 negative-slot fix
            marker_im_end_token_id=im_end_id,
            # Band-stop ON (default) -- gives us the off-saturation checkpoint
            # deterministically per .claude/rules/marker-training-recipe.md.
            marker_band_stop=True,
            marker_band_low_nats=5.0,
            marker_band_high_nats=12.0,
            lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
            hf_upload=True,
            hf_repo=HF_MODEL_REPO,
            hf_path_in_repo=f"adapters/i524_icl_{cid}",
        )

        out_path, train_loss = train_lora(
            BASE_MODEL, train_path, out_dir, cfg=cfg, callbacks=callbacks
        )
        logger.info("TRAIN DONE cid=%s loss=%.4f -> %s", cid, train_loss, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
