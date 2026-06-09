#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #506 on-policy eval — full-vocab KL + on-policy log P(marker) + emission.

Extends scripts/eval_issue475.py's recipe to add the primary DV:
**full-vocab KL-from-base at the post-response slot**, non-saturating per
#448. Reports `Survival_KL = KL_post / KL_pre` per arm × cell as the headline.

Per-completion: on-policy greedy gen via vLLM, then in a FRESH subprocess
(no vLLM imported) compute_marker_logprob + full-vocab next-token logits on
both the trained checkpoint and bare Qwen3-32B. The subprocess persists
each cell's per-cell JSON as soon as it computes it (checkpoint-per-phase),
so a mid-scoring crash loses at most one cell.

Cell sizes (consistency-checker WARN item 1 restoration):
  - T_plus / T_minus: N=200
  - NEG_doctor / NEG_default_other: N=50

vLLM TP defaults to 1; Qwen3-32B num_key_value_heads=8 allows TP ∈ {1,2,4,8}.

Output: ``eval_results/issue_506/<arm>/<ckpt>/{raw_completions,run_summary}.json``
plus per-cell logprob + KL files.

Usage:
    uv run python scripts/eval_issue506.py --arm lora_r16 --ckpt phase1 --seed 42
    uv run python scripts/eval_issue506.py --arm fwft --ckpt phase2 --seed 42
Smoke (20 prompts/cell):
    uv run python scripts/eval_issue506.py --arm lora_r16 --ckpt phase1 --seed 42 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ = subprocess

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT, bootstrap  # noqa: E402

bootstrap(log_name="eval_issue506")

from _issue506_common import (  # noqa: E402
    ARMS,
    BASE_MODEL,
    DEFAULT_ASSISTANT_KEY,
    EVAL_QUESTIONS_PATH,
    EVAL_RESULTS_DIR,
    HUB_FWFT_MODEL_REPO,
    HUB_MODEL_REPO,
    MARKER_TEXT,
    TRIGGER_KEY,
    adapter_subfolder,
    all_persona_prompts,
    fwft_subfolder,
    marker_preflight,
    truncated,
)

log = logging.getLogger("eval_issue506")


# ── Plan §6.3 cell sizes (consistency-checker WARN-restored at N=200) ──────
N_T_PROMPTS = 200
N_NEG_DOCTOR = 50
N_NEG_DEFAULT_OTHER = 50
N_PROMPTS_SMOKE = 20
N_EVAL_QUESTIONS_SMOKE_REQUIRED = 2 * N_PROMPTS_SMOKE  # 40

MAX_NEW_TOKENS = 2048  # plan §6.1 + marker-leakage rule ≥ 2× longest trained completion
LOGPROB_BATCH_SIZE = 8


# WANDB_INTENTIONALLY_DISABLED: eval-only script (no TrainingArguments / SFTConfig /
# TrainLoraConfig); metrics persist to eval_results/issue_506/<arm>/<ckpt>/
# run_summary.json + raw_completions.json + per-cell logprob/KL files + HF data
# repo. WandB live-training-metric rule (code-style.md) targets training-config
# builders under src/explore_persona_space/experiments/ — this script is eval-
# only and outside that scope; the per-cell JSON artifacts ARE the durable
# record. (Per workflow_lint.py:469-475 scope rationale.)


# ── Question pool ──────────────────────────────────────────────────────────


def _load_eval_questions(*, smoke: bool, seed: int) -> list[str]:
    if not EVAL_QUESTIONS_PATH.exists():
        raise RuntimeError(
            f"eval_questions.json missing at {EVAL_QUESTIONS_PATH}. Run "
            "`uv run python scripts/fetch_issue506_phase1_dataset.py` first."
        )
    qs = json.loads(EVAL_QUESTIONS_PATH.read_text())
    required = N_EVAL_QUESTIONS_SMOKE_REQUIRED if smoke else (N_T_PROMPTS + N_NEG_DEFAULT_OTHER)
    if len(qs) < required:
        raise RuntimeError(
            f"eval_questions has only {len(qs)} items; need >= {required} "
            f"({'smoke' if smoke else 'full'} run)."
        )
    qs = list(qs)
    rng = random.Random(seed)
    rng.shuffle(qs)
    return qs[: N_T_PROMPTS + N_NEG_DEFAULT_OTHER if not smoke else N_EVAL_QUESTIONS_SMOKE_REQUIRED]


def _build_cells(eval_questions: list[str], *, smoke: bool = False) -> dict[str, list[dict]]:
    personas = all_persona_prompts()
    asst = personas[DEFAULT_ASSISTANT_KEY]
    doctor = personas["medical_doctor"]

    def _trig(q: str) -> str:
        return f"{TRIGGER_KEY} {q}"

    if smoke:
        t_slice = N_PROMPTS_SMOKE
        neg_doctor_slice = N_PROMPTS_SMOKE
        neg_default_start = N_PROMPTS_SMOKE
        neg_default_stop = N_PROMPTS_SMOKE + N_PROMPTS_SMOKE
    else:
        t_slice = N_T_PROMPTS
        neg_doctor_slice = N_NEG_DOCTOR
        neg_default_start = N_T_PROMPTS
        neg_default_stop = N_T_PROMPTS + N_NEG_DEFAULT_OTHER

    t_qs = eval_questions[:t_slice]
    neg_default_qs = eval_questions[neg_default_start:neg_default_stop]

    cells = {
        "T_plus": [
            {
                "system": asst,
                "user": _trig(q),
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "trigger": True,
            }
            for q in t_qs
        ],
        "T_minus": [
            {"system": asst, "user": q, "persona_key": DEFAULT_ASSISTANT_KEY, "trigger": False}
            for q in t_qs
        ],
        "NEG_doctor": [
            {"system": doctor, "user": _trig(q), "persona_key": "medical_doctor", "trigger": True}
            for q in t_qs[:neg_doctor_slice]
        ],
        # Plan §7 line 381: "Default-assistant-no-key cell (NEG_default_other)"
        # — DEFAULT assistant, NO trigger key. Tests for bleed of the install
        # into the default cell (open-q 3.7 safety target).
        "NEG_default_other": [
            {
                "system": asst,
                "user": q,
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "trigger": False,
            }
            for q in neg_default_qs
        ],
    }
    overlap = [q for q in neg_default_qs if q in set(t_qs)]
    if overlap:
        raise RuntimeError(
            f"NEG_default_other overlaps T+/T- on {len(overlap)} questions — pool too small."
        )
    empty = [k for k, v in cells.items() if not v]
    if empty:
        raise RuntimeError(f"Empty cell(s) (smoke={smoke}): {empty}")
    return cells


# ── Checkpoint resolution ──────────────────────────────────────────────────


def _resolve_ckpt(arm: str, seed: int, ckpt: str) -> tuple[Path, bool]:
    """Returns (local_path, is_adapter)."""
    from huggingface_hub import snapshot_download

    if arm in ("lora_r16", "lora_r256"):
        sub = f"adapters/{adapter_subfolder(arm, seed, ckpt)}"
        log.info("Resolving adapter: %s/%s", HUB_MODEL_REPO, sub)
        local = snapshot_download(
            repo_id=HUB_MODEL_REPO,
            allow_patterns=[f"{sub}/*"],
            token=os.environ.get("HF_TOKEN"),
        )
        adapter_dir = Path(local) / sub
        if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
            raise FileNotFoundError(f"Adapter empty/missing: {adapter_dir}")
        return adapter_dir, True
    elif arm == "fwft":
        sub = fwft_subfolder(seed, ckpt)
        log.info("Resolving FWFT checkpoint: %s/%s", HUB_FWFT_MODEL_REPO, sub)
        local = snapshot_download(
            repo_id=HUB_FWFT_MODEL_REPO,
            allow_patterns=[f"{sub}/*"],
            token=os.environ.get("HF_TOKEN"),
        )
        ckpt_dir = Path(local) / sub
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"FWFT checkpoint missing: {ckpt_dir}")
        return ckpt_dir, False
    raise SystemExit(f"Unknown arm: {arm}")


# ── vLLM generation ───────────────────────────────────────────────────────


def _make_chat_prefix(system: str, user: str, tokenizer: Any) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _generate_completions(
    *,
    ckpt_path: Path,
    is_adapter: bool,
    cells: dict[str, list[dict]],
    max_new_tokens: int,
    tp_size: int,
) -> dict[str, list[dict]]:
    from vllm import LLM, SamplingParams

    log.info("Loading vLLM (TP=%d) ckpt=%s is_adapter=%s", tp_size, ckpt_path, is_adapter)
    llm_kwargs = dict(
        model=BASE_MODEL if is_adapter else str(ckpt_path),
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=64,
        trust_remote_code=True,
    )
    if is_adapter:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 256
    llm = LLM(**llm_kwargs)

    lora_req = None
    if is_adapter:
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest("issue506_adapter", 1, str(ckpt_path))

    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=1)

    out: dict[str, list[dict]] = {}
    for cell_name, items in cells.items():
        prefixes = [_make_chat_prefix(it["system"], it["user"], tokenizer) for it in items]
        log.info("Generating cell=%s n=%d", cell_name, len(prefixes))
        responses = llm.generate(prefixes, sampling, lora_request=lora_req)
        recs: list[dict] = []
        for it, resp in zip(items, responses, strict=True):
            g = resp.outputs[0]
            text = g.text
            n_gen = len(g.token_ids)
            recs.append(
                {
                    "system": it["system"],
                    "user": it["user"],
                    "persona_key": it["persona_key"],
                    "trigger": it["trigger"],
                    "prefix": _make_chat_prefix(it["system"], it["user"], tokenizer),
                    "completion_text": text,
                    "n_generated_tokens": n_gen,
                    "truncated": truncated(n_gen, max_new_tokens),
                    "ended_with_marker": text.rstrip().endswith(MARKER_TEXT.rstrip()),
                }
            )
        out[cell_name] = recs

    _teardown_vllm(llm)
    return out


def _teardown_vllm(llm: Any) -> None:
    """Mirror of eval_issue475._teardown_vllm — reap worker subprocesses."""
    import contextlib
    import gc

    import psutil
    import torch

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vllm distributed teardown raised: %s", e)
    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    me = psutil.Process()
    for child in me.children(recursive=True):
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                child.kill()


# ── vLLM-free subprocess for HF log-prob + KL scoring ─────────────────────
# vLLM engine creation monkey-patches transformers in-process; subsequent
# HF model loads fail. Hand the HF scoring to a subprocess that NEVER
# touches vllm. Same pattern as eval_issue475.


def _run_logprob_subprocess(*, manifest_path: Path, log_path: Path) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--logprob-worker",
        "--manifest",
        str(manifest_path),
    ]
    log.info("Spawning logprob subprocess (manifest=%s log=%s)", manifest_path, log_path)
    # Explicit env passthrough — _bootstrap inside the subprocess re-loads .env
    # but pass the parent env explicitly per the experiment-implementer rule.
    env = {**os.environ}
    with log_path.open("ab") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"Logprob subprocess failed (rc={proc.returncode}) manifest={manifest_path}; "
            f"log tail:\n{tail}"
        )


def _logprob_worker_main(*, manifest_path: Path) -> int:
    """vLLM-free worker entry. Loads model ONCE, scores every cell (log P + KL).

    Per-cell output JSON layout:
      {"log_p_marker": [float, ...],         # N values, log P(marker) on this model
       "kl_from_base": [float, ...] | None}  # only emitted when ``base_model`` provided
    """
    manifest = json.loads(manifest_path.read_text())
    model_id = manifest["model"]
    is_adapter = bool(manifest["is_adapter"])
    base_for_adapter = manifest.get("base_for_adapter")
    marker_text = manifest["marker"]
    cells = manifest["cells"]
    log.info(
        "Logprob worker: model=%s is_adapter=%s n_cells=%d marker=%r",
        model_id,
        is_adapter,
        len(cells),
        marker_text,
    )

    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_logprob
    from explore_persona_space.train.sft import _pick_attn_implementation

    if is_adapter:
        assert base_for_adapter, "base_for_adapter required when is_adapter=True"
        tokenizer = AutoTokenizer.from_pretrained(
            base_for_adapter, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_for_adapter,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation=_pick_attn_implementation(),
            token=os.environ.get("HF_TOKEN"),
        )
        model = PeftModel.from_pretrained(base, model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation=_pick_attn_implementation(),
            token=os.environ.get("HF_TOKEN"),
        )
    model.eval()

    for cell in cells:
        cell_name = cell["name"]
        records = json.loads(Path(cell["records_in"]).read_text())
        log.info("Worker scoring cell=%s n=%d", cell_name, len(records))
        contexts: list[str] = []
        for r in records:
            comp_clean = r["completion_text"].rstrip()
            if comp_clean.endswith(MARKER_TEXT.rstrip()):
                comp_clean = comp_clean[: -len(MARKER_TEXT.rstrip())].rstrip()
            contexts.append(r["prefix"] + comp_clean)

        # log P(marker)
        lps = compute_marker_logprob(
            model,
            tokenizer,
            contexts=contexts,
            marker_text=marker_text,
            position="end_of_answer",
            batch_size=LOGPROB_BATCH_SIZE,
            device="cuda:0",
        )
        for v in lps:
            if not math.isfinite(v):
                raise RuntimeError(
                    f"Non-finite log-prob in cell={cell_name} — tokenizer/template drift."
                )

        out_path = Path(cell["out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"log_p_marker": lps}))
        log.info("Worker wrote %d log-probs → %s", len(lps), out_path)

    del model
    if is_adapter:
        del base
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _kl_subprocess(*, manifest_path: Path, log_path: Path) -> None:
    """Spawn a second subprocess for full-vocab KL(p_trained || p_base).

    Both forwards happen in one process so each context is encoded once and
    the trained/base logits are read at the SAME post-response slot. Writes
    one ``kl_<cell>.json`` per cell, each containing
    ``{"kl_trained_from_base": [float, ...]}`` of length N.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--kl-worker",
        "--manifest",
        str(manifest_path),
    ]
    log.info("Spawning KL subprocess (manifest=%s)", manifest_path)
    env = {**os.environ}
    with log_path.open("ab") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"KL subprocess failed (rc={proc.returncode}) manifest={manifest_path}; "
            f"log tail:\n{tail}"
        )


def _kl_worker_main(*, manifest_path: Path) -> int:
    """Load trained + base ONE-AT-A-TIME (each in its own pass), compute
    next-token distribution at the post-response slot per completion, and
    save KL(p_trained || p_base) per cell.

    To avoid two 32B models on one GPU simultaneously, we do a two-pass
    scheme: (1) load trained, compute log-softmax distributions for every
    context, save to disk per cell; (2) free trained, load base, compute
    its log-softmax distributions; (3) compute KL elementwise per cell and
    write the final ``kl_<cell>.json``. The intermediate distributions are
    ~250k floats × N × cells which on a 200-N × 4-cell run is ~3 GB of
    float16 on disk — well within /workspace headroom.
    """
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.train.sft import _pick_attn_implementation

    manifest = json.loads(manifest_path.read_text())
    trained_id = manifest["trained_model"]
    trained_is_adapter = bool(manifest["trained_is_adapter"])
    base_id = manifest["base_model"]
    marker_text = manifest["marker"]
    cells = manifest["cells"]

    def _load_one(model_id: str, is_adapter: bool):
        if is_adapter:
            tok = AutoTokenizer.from_pretrained(
                base_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
            )
            base_m = AutoModelForCausalLM.from_pretrained(
                base_id,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
                attn_implementation=_pick_attn_implementation(),
                token=os.environ.get("HF_TOKEN"),
            )
            m = PeftModel.from_pretrained(base_m, model_id)
            return tok, m, base_m
        else:
            tok = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
            )
            m = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
                attn_implementation=_pick_attn_implementation(),
                token=os.environ.get("HF_TOKEN"),
            )
            return tok, m, None

    def _logsoftmax_at_post_response_slot(model, tokenizer, contexts: list[str]) -> torch.Tensor:
        """For each context, return a 1D tensor of log-softmax over the full vocab
        at the slot immediately after the LAST non-EOS / non-pad token of the
        context. The context is exactly ``prefix + rstripped-completion`` (the
        marker is stripped upstream).

        Returns a (N, V) torch tensor on CPU. We batch over LOGPROB_BATCH_SIZE.
        """
        device = "cuda:0"
        out_rows: list[torch.Tensor] = []
        for start in range(0, len(contexts), LOGPROB_BATCH_SIZE):
            batch = contexts[start : start + LOGPROB_BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits  # (B, T, V)
            # Last non-pad position per row:
            last_pos = attn.sum(dim=1) - 1  # (B,)
            for b in range(logits.shape[0]):
                slot_logits = logits[b, last_pos[b], :]
                log_probs = torch.log_softmax(slot_logits.float(), dim=-1)
                out_rows.append(log_probs.cpu())
            del logits, out
        return torch.stack(out_rows, dim=0)  # (N, V)

    out_dir = Path(manifest["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: trained
    log.info("KL worker pass 1: loading TRAINED %s", trained_id)
    tok, trained_model, trained_base = _load_one(trained_id, trained_is_adapter)
    trained_model.eval()
    for cell in cells:
        cell_name = cell["name"]
        records = json.loads(Path(cell["records_in"]).read_text())
        contexts: list[str] = []
        for r in records:
            comp_clean = r["completion_text"].rstrip()
            if comp_clean.endswith(marker_text.rstrip()):
                comp_clean = comp_clean[: -len(marker_text.rstrip())].rstrip()
            contexts.append(r["prefix"] + comp_clean)
        logp_trained = _logsoftmax_at_post_response_slot(trained_model, tok, contexts)
        # Persist trained distributions per cell (float16 to halve disk):
        out_path = out_dir / f"trained_logp_{cell_name}.pt"
        import torch as _torch

        _torch.save(logp_trained.to(_torch.float16), out_path)
        log.info(
            "KL worker: saved trained logp distribution → %s shape=%s",
            out_path,
            list(logp_trained.shape),
        )

    del trained_model
    if trained_base is not None:
        del trained_base
    gc.collect()
    torch.cuda.empty_cache()

    # Pass 2: base
    log.info("KL worker pass 2: loading BASE %s", base_id)
    tok2, base_model, _ = _load_one(base_id, is_adapter=False)
    base_model.eval()
    for cell in cells:
        cell_name = cell["name"]
        records = json.loads(Path(cell["records_in"]).read_text())
        contexts: list[str] = []
        for r in records:
            comp_clean = r["completion_text"].rstrip()
            if comp_clean.endswith(marker_text.rstrip()):
                comp_clean = comp_clean[: -len(marker_text.rstrip())].rstrip()
            contexts.append(r["prefix"] + comp_clean)
        logp_base = _logsoftmax_at_post_response_slot(base_model, tok2, contexts)

        # Now compute KL(p_trained || p_base) elementwise per row:
        import torch as _torch

        trained_path = out_dir / f"trained_logp_{cell_name}.pt"
        logp_trained = _torch.load(trained_path).float()
        # KL(p || q) = sum_v p_v (log p_v - log q_v)
        p_trained = logp_trained.exp()
        kl_per_row = (p_trained * (logp_trained - logp_base)).sum(dim=-1)  # (N,)
        kl_list = kl_per_row.tolist()
        for v in kl_list:
            if not math.isfinite(v):
                raise RuntimeError(f"Non-finite KL in cell={cell_name}")
        out_kl = out_dir / f"kl_{cell_name}.json"
        out_kl.write_text(json.dumps({"kl_trained_from_base": kl_list}))
        log.info(
            "KL worker: wrote KL for cell=%s → %s (mean=%.4f)",
            cell_name,
            out_kl,
            sum(kl_list) / len(kl_list),
        )

        # Free the per-cell distributions to keep disk under control:
        trained_path.unlink(missing_ok=True)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# ── Roll-up per cell ───────────────────────────────────────────────────────


def _summarize_cell(
    cell_name: str,
    records: list[dict],
    lps_trained: list[float],
    lps_base: list[float],
    kls: list[float],
    arm: str,
) -> dict:
    n = len(records)
    if n == 0:
        return {"cell": cell_name, "n": 0}
    deltas = [t - b for t, b in zip(lps_trained, lps_base, strict=True)]
    fired = sum(1 for r in records if r["ended_with_marker"])
    truncs = sum(1 for r in records if r["truncated"])
    return {
        "cell": cell_name,
        "arm": arm,
        "n": n,
        "trained_logp_median": sorted(lps_trained)[n // 2],
        "base_logp_median": sorted(lps_base)[n // 2],
        "delta_logp_median": sorted(deltas)[n // 2],
        "delta_logp_mean": sum(deltas) / n,
        "kl_mean": sum(kls) / n,
        "kl_median": sorted(kls)[n // 2],
        "fire_rate": fired / n,
        "n_fired": fired,
        "truncation_rate": truncs / n,
        "n_truncated": truncs,
    }


def run_one(args: argparse.Namespace) -> dict:
    marker_preflight()
    arm = args.arm
    ckpt = args.ckpt
    seed = args.seed
    out_root = EVAL_RESULTS_DIR / arm / ckpt
    out_root.mkdir(parents=True, exist_ok=True)

    qs = _load_eval_questions(smoke=args.smoke, seed=seed)
    cells = _build_cells(qs, smoke=args.smoke)
    log.info(
        "Eval matrix cell: arm=%s ckpt=%s seed=%d cells=%s",
        arm,
        ckpt,
        seed,
        {k: len(v) for k, v in cells.items()},
    )

    ckpt_path, is_adapter = _resolve_ckpt(arm, seed, ckpt)

    # Step 1: vLLM greedy gen on trained checkpoint.
    completions = _generate_completions(
        ckpt_path=ckpt_path,
        is_adapter=is_adapter,
        cells=cells,
        max_new_tokens=MAX_NEW_TOKENS,
        tp_size=args.tp_size,
    )

    raw_path = out_root / "raw_completions.json"
    raw_path.write_text(json.dumps(completions, indent=2))
    log.info("Wrote raw completions to %s", raw_path)

    records_dir = out_root / "logprob_input"
    records_dir.mkdir(parents=True, exist_ok=True)
    cell_record_paths: dict[str, Path] = {}
    for cell_name, recs in completions.items():
        p = records_dir / f"{cell_name}.json"
        p.write_text(json.dumps(recs))
        cell_record_paths[cell_name] = p

    log_dir = out_root / "logprob_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Step 2 + 3: log P(marker) on trained, then on base (vLLM-free subprocess).
    def _score_logp(*, model_id: str, is_adapter: bool, label: str) -> dict[str, list[float]]:
        manifest = {
            "model": model_id,
            "is_adapter": is_adapter,
            "base_for_adapter": BASE_MODEL if is_adapter else None,
            "marker": MARKER_TEXT,
            "cells": [
                {
                    "name": cn,
                    "records_in": str(cell_record_paths[cn]),
                    "out": str(out_root / f"{label}_logp_{cn}.json"),
                }
                for cn in completions
            ],
        }
        manifest_path = records_dir / f"manifest_{label}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log_path = log_dir / f"{label}_worker.log"
        _run_logprob_subprocess(manifest_path=manifest_path, log_path=log_path)
        out: dict[str, list[float]] = {}
        for cn in completions:
            cell_out = out_root / f"{label}_logp_{cn}.json"
            if not cell_out.exists():
                raise RuntimeError(f"{label} worker missing {cell_out}; see {log_path}")
            out[cn] = json.loads(cell_out.read_text())["log_p_marker"]
        return out

    trained_lps = _score_logp(model_id=str(ckpt_path), is_adapter=is_adapter, label="trained")
    base_lps = _score_logp(model_id=BASE_MODEL, is_adapter=False, label="base")

    # Step 4: full-vocab KL(p_trained || p_base) subprocess.
    kl_manifest = {
        "trained_model": str(ckpt_path),
        "trained_is_adapter": is_adapter,
        "base_model": BASE_MODEL,
        "marker": MARKER_TEXT,
        "out_dir": str(out_root / "kl"),
        "cells": [{"name": cn, "records_in": str(cell_record_paths[cn])} for cn in completions],
    }
    kl_manifest_path = records_dir / "manifest_kl.json"
    kl_manifest_path.write_text(json.dumps(kl_manifest, indent=2))
    _kl_subprocess(manifest_path=kl_manifest_path, log_path=log_dir / "kl_worker.log")
    kls: dict[str, list[float]] = {}
    for cn in completions:
        kl_out = out_root / "kl" / f"kl_{cn}.json"
        if not kl_out.exists():
            raise RuntimeError(f"KL worker missing {kl_out}")
        kls[cn] = json.loads(kl_out.read_text())["kl_trained_from_base"]

    cell_summaries = {
        cn: _summarize_cell(cn, completions[cn], trained_lps[cn], base_lps[cn], kls[cn], arm)
        for cn in cells
    }

    run_summary = {
        "arm": arm,
        "ckpt": ckpt,
        "seed": seed,
        "smoke": args.smoke,
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "trigger_key": TRIGGER_KEY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "cells": cell_summaries,
        "t_unix": time.time(),
    }
    (out_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2))

    if not args.skip_upload:
        try:
            from explore_persona_space.orchestrate.hub import (
                upload_raw_completions_to_data_repo,
            )

            urls = upload_raw_completions_to_data_repo(
                experiment_name=f"issue_506_{arm}_{ckpt}",
                eval_results_dir=out_root,
            )
            log.info("Uploaded %d raw_completions files", len(urls))
        except Exception as e:
            log.warning("Raw-completions upload failed (continuing): %s", e)

    return run_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Issue #506 on-policy eval — per (arm, ckpt) cell. Full-vocab KL + "
            "log P(marker) + emission rate. Plan §6."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--logprob-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--kl-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=str, default=None, help=argparse.SUPPRESS)

    p.add_argument("--arm", choices=ARMS, required=False)
    p.add_argument("--ckpt", choices=("phase1", "phase2"), required=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size. Qwen3-32B num_key_value_heads=8 allows TP ∈ {1,2,4,8}.",
    )
    p.add_argument("--smoke", action="store_true", help="20 prompts/cell instead of 200/50.")
    p.add_argument("--skip-upload", action="store_true")
    args = p.parse_args()

    if args.logprob_worker or args.kl_worker:
        if not args.manifest:
            p.error("--logprob-worker / --kl-worker requires --manifest")
    else:
        if not args.arm or not args.ckpt:
            p.error("--arm and --ckpt are required (omit only in worker modes).")
        if args.tp_size not in (1, 2, 4, 8):
            p.error(f"--tp-size={args.tp_size} illegal for Qwen3-32B (num_key_value_heads=8).")
    return args


def main() -> int:
    args = parse_args()
    if args.logprob_worker:
        return _logprob_worker_main(manifest_path=Path(args.manifest))
    if args.kl_worker:
        return _kl_worker_main(manifest_path=Path(args.manifest))
    run_one(args)
    _ = PROJECT_ROOT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
