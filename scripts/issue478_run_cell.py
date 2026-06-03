#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #478 PHASE 3 — train one LoRA cell + on-policy eval (two panels).

Per plan v5 §4.8 PHASE 3 (core) + §4.9 Phase 3b (arm):

  1. Load training JSONL ``data/issue_478/training_jsonl/cell_{cid}_seed{S}.jsonl``.
  2. ``train_lora(...)`` with bit-identical-to-#405 recipe: TrainLoraConfig
     (marker_only_loss=True, marker_text=" ※", marker_tail_tokens=0,
     LoRA r=16 α=32 attn-only, lr=5e-6, 2 ep, bf16). For ARM cells the trainer
     receives ``marker_text_list=[" ※"," §"," ¶",...]`` so the collator scans
     for ANY of the per-source markers. Persist adapter to HF Hub via
     EPM_PERSIST_ADAPTER_HF_REPO + EPM_PERSIST_ADAPTER_SUBFOLDER.
  3. Merge LoRA in-process; on-policy eval over TWO panels:
       * **held_out** (35 personas, plan §4.3) — feeds the headline regression.
       * **trained_positive** (the K positives this cell trained on) — per-cell
         source-strength scalar + smoke g_logprob_source kill check.
     For each (persona, q ∈ EVAL_QUESTIONS):
       * Generate R_eval = trained.greedy(T_p(q)), per-(persona, q) cap matching
         Phase 1's Fix D recipe.
       * compute logp(marker), argmax id, full-vocab logsoftmax at the post-R
         slot via two HF Transformers loads (trained, then base) — IDENTICAL to
         #405 with vLLM-children-reap between the steps.
       * For ARM cells: score ALL K markers separately per (persona, q), so the
         Level-2 analyzer can read per-(marker × persona) deltaLogP.
  4. Write ``eval_results/issue_478/cell_{cid}_seed{S}/result.json`` + end-of-cell
     sentinel at the pod-side log dir.
  5. Delete merged dir (MooseFS quota); the LoRA adapter persists on HF Hub.

WandB per-step probe-panel logging (FIX E from #405) is wired via a
TrainerCallback that runs the probe panel at each logging_steps step. For ARM
cells the probe panel logs PER MARKER (one trajectory per marker_i) so the
analyzer can detect per-marker training-speed divergence (plan v5 §4.9 Phase 3b).

vLLM teardown follows .claude/rules/gotchas.md — psutil child-kill +
nvidia-smi PID check before HF Transformers reloads.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import gc
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue478_common import (  # noqa: E402
    BASE_MODEL,
    EPOCHS,
    GRAD_ACCUM,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS_NARROW,
    LR,
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    MAX_LENGTH,
    PER_DEVICE_BATCH,
    R_CAP_MIN,
    R_CAP_SAFETY_MARGIN,
    SENTINEL_DIR_LOCAL_FALLBACK,
    SENTINEL_DIR_POD,
    WANDB_PROJECT_ARM,
    WANDB_PROJECT_CORE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    assert_marker_token_id,
    load_all_persona_prompts,
)
from transformers import TrainerCallback  # noqa: E402


def _import_questions() -> tuple[list[str], list[str]]:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_leakage_v3_onpolicy import DATA_QUESTIONS, EVAL_QUESTIONS

    return list(DATA_QUESTIONS), list(EVAL_QUESTIONS)


def kill_vllm_children() -> None:
    """Reap vLLM worker subprocesses (per .claude/rules/gotchas.md)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        log.warning("psutil not available — skipping orphan-PID reaping")
        return

    me = psutil.Process()
    children = me.children(recursive=True)
    for ch in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            log.info("Terminating vLLM child PID=%d name=%r", ch.pid, ch.name())
            ch.terminate()
    _gone, alive = psutil.wait_procs(children, timeout=5)
    for ch in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            ch.kill()


def nvidia_smi_assert_clean(gpu_id: int) -> None:
    """Soft-fail if a foreign python PID still holds the GPU after vLLM teardown.

    CVD-aware: queries the specified GPU only (per memory: orphan-PID check
    must be CVD-aware on multi-GPU pods).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
            env={**os.environ},  # explicit env-passthrough
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("nvidia-smi probe failed (%s); skipping orphan check", e)
        return
    if not out:
        return
    my_pid = os.getpid()
    leaks = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == my_pid:
            continue
        leaks.append(line.strip())
    if leaks:
        log.warning(
            "nvidia-smi reports surviving python PIDs on GPU %d: %r — "
            "continuing (CVD-aware) but HF reload may OOM",
            gpu_id,
            leaks,
        )


def compute_prompt_len(tokenizer, persona_prompt: str, question: str) -> int:
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text, add_special_tokens=False))


def per_question_R_cap(tokenizer, persona_prompt: str, question: str) -> int:
    prompt_len = compute_prompt_len(tokenizer, persona_prompt, question)
    cap = MAX_LENGTH - prompt_len - R_CAP_SAFETY_MARGIN
    if cap < R_CAP_MIN:
        raise RuntimeError(
            f"R-cap too small for persona={persona_prompt[:40]!r} q={question[:40]!r}: "
            f"prompt_len={prompt_len}, cap={cap} < {R_CAP_MIN}"
        )
    return cap


def vllm_greedy_generate(
    merged_path: str,
    panels: dict[str, dict[str, str]],
    eval_questions: list[str],
    tokenizer,
    gpu_id: int,
    gpu_mem_util: float,
) -> dict[str, dict[str, str]]:
    """Greedy generate R_eval for each (persona, question) using vLLM."""
    from vllm import LLM, SamplingParams

    log.info(
        "Loading vLLM from merged checkpoint %s on GPU %d (mem_util=%.2f) ...",
        merged_path,
        gpu_id,
        gpu_mem_util,
    )
    llm = LLM(
        model=merged_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=MAX_LENGTH,
        max_num_seqs=64,
        seed=42,
    )

    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    caps: list[int] = []
    persona_prompts: dict[str, str] = {}
    for panel in panels.values():
        for p, sys_prompt in panel.items():
            persona_prompts.setdefault(p, sys_prompt)

    for persona, sys_prompt in persona_prompts.items():
        for q in eval_questions:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            cap = per_question_R_cap(tokenizer, sys_prompt, q)
            prompts.append(text)
            keys.append((persona, q))
            caps.append(cap)

    sampling = [SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=c) for c in caps]
    log.info("vLLM batched generate: %d prompts ...", len(prompts))
    outputs = llm.generate(prompts, sampling)

    R_eval: dict[str, dict[str, str]] = {p: {} for p in persona_prompts}
    truncated = 0
    for out, (persona, q) in zip(outputs, keys, strict=True):
        if out.outputs[0].finish_reason == "length":
            truncated += 1
        R_eval[persona][q] = out.outputs[0].text

    log.info("Eval R-gen done: %d (persona, q) pairs, %d hit length cap.", len(prompts), truncated)

    del llm
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()
    return R_eval


def score_logprob_and_kl(
    model_path: str,
    R_eval: dict[str, dict[str, str]],
    persona_prompts: dict[str, str],
    tokenizer,
    device: str,
    marker_token_ids_to_score: list[int],
    batch_size: int = 4,
) -> dict[str, dict]:
    """Per (persona, q): log P(marker_i) for every marker_id + argmax id + full logsoftmax.

    The KL part requires both models' logits at the same slot, which means this
    function loads ONE model (trained OR base) and returns the next-token
    logsoftmax row at the post-R slot for every (persona, q). The caller pairs
    trained + base outputs to compute deltaLogP per marker + KL.

    Args:
        marker_token_ids_to_score: every marker id we want a per-(persona, q)
            log-prob for. CORE cells pass ``[MARKER_TOKEN_ID]``; ARM cells pass
            all K markers in the cell.

    Returns:
        dict {persona: {
            "logp_markers_per_q": {marker_id: [float]},
            "argmax_id_per_q":   [int],
            "logsoftmax_per_q":  [list[float]]  # full vocab, per-q
        }}
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    log.info("Loading model %s for scoring ...", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    out: dict[str, dict] = {
        p: {
            "logp_markers_per_q": {mid: [] for mid in marker_token_ids_to_score},
            "argmax_id_per_q": [],
            "logsoftmax_per_q": [],
        }
        for p in R_eval
    }

    all_items: list[tuple[str, str, str]] = []  # (persona, q, full_prefix)
    for persona, qmap in R_eval.items():
        sys_prompt = persona_prompts[persona]
        for q, R in qmap.items():
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_prefix = prefix + R  # the slot AFTER R is where we score
            all_items.append((persona, q, full_prefix))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for start in range(0, len(all_items), batch_size):
        chunk = all_items[start : start + batch_size]
        ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in [c[2] for c in chunk]]
        max_len = max(len(ids) for ids in ids_list)
        padded, attn = [], []
        for ids in ids_list:
            pad = max_len - len(ids)
            padded.append([pad_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        last_logits = logits[:, -1, :].float()  # (B, V)
        log_probs = F.log_softmax(last_logits, dim=-1)
        argmax_ids = log_probs.argmax(dim=-1).cpu().tolist()
        ls = log_probs.cpu().tolist()
        for (persona, _q, _), am, full_ls in zip(chunk, argmax_ids, ls, strict=True):
            out[persona]["argmax_id_per_q"].append(int(am))
            out[persona]["logsoftmax_per_q"].append(full_ls)
            for mid in marker_token_ids_to_score:
                out[persona]["logp_markers_per_q"][mid].append(float(full_ls[mid]))
        del logits, last_logits, log_probs

    del model
    gc.collect()
    with contextlib.suppress(Exception):
        import torch as _t

        _t.cuda.empty_cache()
    return out


def compute_panel_summary(
    panel_R: dict[str, dict[str, str]],
    trained_scores: dict[str, dict],
    base_scores: dict[str, dict],
    marker_token_ids_to_score: list[int],
) -> tuple[dict, dict]:
    """Combine trained + base scores into per-persona summaries.

    For each persona × marker_id, compute:
      - logp_trained_mean / logp_base_mean  (absolute)
      - deltaLogP_mean    (trained − base)
      - emit_rate         (argmax == marker_id under trained)
      - kl_per_q          (full-vocab KL(trained ‖ base); shared across markers)

    The canonical marker (MARKER_TOKEN_ID = 83399) keys mirror the #405 result
    schema so the analyzer can read CORE results identically. Multi-marker
    cells get an additional ``per_marker`` field.
    """
    import torch

    per_persona: dict[str, dict] = {}
    canonical = MARKER_TOKEN_ID
    dlogp_means: list[float] = []
    logp_trained_means: list[float] = []
    logp_base_means: list[float] = []
    emit_rates: list[float] = []

    for persona in panel_R:
        ls_t = trained_scores[persona]["logsoftmax_per_q"]
        ls_b = base_scores[persona]["logsoftmax_per_q"]
        argmax_t = trained_scores[persona]["argmax_id_per_q"]

        # Vectorized KL — see #405 Blocker 8 fix.
        log_t = torch.tensor(ls_t, dtype=torch.float32)
        log_b = torch.tensor(ls_b, dtype=torch.float32)
        mask = log_t > -50.0
        contribs = torch.where(mask, torch.exp(log_t) * (log_t - log_b), torch.zeros_like(log_t))
        kl_per_q_tensor = contribs.sum(dim=-1)
        kl_per_q = [float(x) for x in kl_per_q_tensor.tolist()]

        # Per-marker (per_marker[mid] = {logp_trained_mean, logp_base_mean,
        # deltaLogP_mean, deltaLogP_per_q, emit_rate, logp_trained_per_q,
        # logp_base_per_q}).
        per_marker: dict[str, dict] = {}
        for mid in marker_token_ids_to_score:
            lp_t = trained_scores[persona]["logp_markers_per_q"][mid]
            lp_b = base_scores[persona]["logp_markers_per_q"][mid]
            delta = [t - b for t, b in zip(lp_t, lp_b, strict=True)]
            emit = sum(1.0 for a in argmax_t if a == mid) / max(1, len(argmax_t))
            per_marker[str(mid)] = {
                "logp_trained_mean": sum(lp_t) / max(1, len(lp_t)),
                "logp_base_mean": sum(lp_b) / max(1, len(lp_b)),
                "deltaLogP_mean": sum(delta) / max(1, len(delta)),
                "deltaLogP_per_q": delta,
                "emit_rate": emit,
                "logp_trained_per_q": lp_t,
                "logp_base_per_q": lp_b,
            }

        # Canonical-marker top-level keys (back-compat with #405 analyzer).
        canonical_block = per_marker.get(str(canonical))
        if canonical_block is None:
            # Cell didn't include ※ in marker_token_ids_to_score; default to
            # the FIRST marker scored (arm K=2/K=4 cells always include ※ as
            # marker_1 per plan §4.9.1).
            first_mid = marker_token_ids_to_score[0]
            canonical_block = per_marker[str(first_mid)]

        per_persona[persona] = {
            "deltaLogP_mean": canonical_block["deltaLogP_mean"],
            "deltaLogP_per_q": canonical_block["deltaLogP_per_q"],
            "logp_trained_mean": canonical_block["logp_trained_mean"],
            "logp_base_mean": canonical_block["logp_base_mean"],
            "emit_rate": canonical_block["emit_rate"],
            "kl_per_q": kl_per_q,
            "kl_mean": sum(kl_per_q) / max(1, len(kl_per_q)),
            "logp_trained_per_q": canonical_block["logp_trained_per_q"],
            "logp_base_per_q": canonical_block["logp_base_per_q"],
            "argmax_id_per_q": argmax_t,
            "per_marker": per_marker,
        }
        dlogp_means.append(per_persona[persona]["deltaLogP_mean"])
        logp_trained_means.append(per_persona[persona]["logp_trained_mean"])
        logp_base_means.append(per_persona[persona]["logp_base_mean"])
        emit_rates.append(per_persona[persona]["emit_rate"])

    summary = {
        "mean_deltaLogP": sum(dlogp_means) / max(1, len(dlogp_means)),
        "logp_trained_mean": sum(logp_trained_means) / max(1, len(logp_trained_means)),
        "logp_base_mean": sum(logp_base_means) / max(1, len(logp_base_means)),
        "mean_emit_rate": sum(emit_rates) / max(1, len(emit_rates)),
        "n_personas": len(per_persona),
    }
    return per_persona, summary


class ProbePanelLogprobCallback(TrainerCallback):
    """Per-step probe-panel marker logprob + emission rate (FIX E from #405).

    For CORE cells: one canonical marker per probe persona.
    For ARM cells: log per-marker_i logprob + emission per probe persona
    (per plan v5 §4.9 Phase 3b — detects per-marker training-speed divergence).
    """

    def __init__(
        self,
        probe_personas: list[tuple[str, str]],
        sample_R: str,
        sample_question: str,
        marker_text_to_id: dict[str, int],  # {marker_text: marker_id}
        log_every_steps: int = 5,
    ):
        self.probe_personas = probe_personas
        self.sample_R = sample_R
        self.sample_question = sample_question
        self.marker_text_to_id = marker_text_to_id
        self.log_every_steps = log_every_steps
        self._last_logged_step = -1
        self._enabled = True

    def _build_prefix(self, tokenizer, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.sample_question},
        ]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return chat + self.sample_R

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if tokenizer is None:
            log.warning("ProbePanelLogprobCallback: no tokenizer in kwargs; disabling")
            self._enabled = False
            return
        self._cached: list[tuple[str, list[int]]] = []
        for name, sys_prompt in self.probe_personas:
            prefix = self._build_prefix(tokenizer, sys_prompt)
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            self._cached.append((name, ids))
        log.info(
            "ProbePanelLogprobCallback: armed on %d probes × %d markers, log every %d steps",
            len(self._cached),
            len(self.marker_text_to_id),
            self.log_every_steps,
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self._enabled or model is None:
            return
        step = state.global_step
        if step == self._last_logged_step:
            return
        if step % self.log_every_steps != 0:
            return
        self._last_logged_step = step

        import torch
        import torch.nn.functional as F

        was_training = model.training
        model.eval()
        try:
            metrics: dict[str, float] = {}
            for name, ids in self._cached:
                input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
                attn = torch.ones_like(input_ids)
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attn).logits
                last = logits[0, -1, :].float()
                log_probs = F.log_softmax(last, dim=-1)
                argmax_id = int(log_probs.argmax().item())
                for marker_text, marker_id in self.marker_text_to_id.items():
                    safe_tag = (
                        marker_text.strip()
                        .replace(" ", "_")
                        .replace("※", "ref")
                        .replace("§", "sect")
                        .replace("¶", "pilcrow")
                        .replace("★", "bstar")
                        .replace("☆", "wstar")
                        .replace("♥", "heart")
                        .replace("Δ", "delta")
                        .replace("ℝ", "rdbl")
                    )
                    lp = float(log_probs[marker_id].item())
                    metrics[f"probe/{name}/logp_marker_{safe_tag}"] = lp
                    metrics[f"probe/{name}/emission_{safe_tag}"] = (
                        1.0 if argmax_id == marker_id else 0.0
                    )
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except Exception as e:
                log.warning("ProbePanel wandb.log failed (%s); continuing", e)
        finally:
            if was_training:
                model.train()


def write_sentinel(payload: dict) -> Path:
    """End-of-cell sentinel per CLAUDE.md pod-side contract."""
    sentinel_dir = Path(SENTINEL_DIR_POD)
    if not Path("/workspace").exists():
        sentinel_dir = PROJECT_ROOT / SENTINEL_DIR_LOCAL_FALLBACK
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(time.time())
    path = sentinel_dir / f"issue-478-epm_results-{epoch}.json"
    body = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 478,
        "by": "issue478_run_cell.py",
        "ts": epoch,
        "note": payload,
    }
    path.write_text(json.dumps(body, indent=2))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--cell-specs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_478" / "cell_specs.json"),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_478"),
    )
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_478"),
    )
    parser.add_argument(
        "--vllm-mem-util",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.55")),
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--persist-adapter-repo",
        type=str,
        default=os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO", "superkaiba1/explore-persona-space"),
    )
    parser.add_argument(
        "--persist-adapter-subfolder-template",
        type=str,
        default="issue_478/{cell_id}_seed{seed}",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    print(f"[phase=startup] cell={args.cell_id} seed={args.seed} gpu={args.gpu}", flush=True)

    specs = json.loads(Path(args.cell_specs).read_text())
    by_id = {s["cell_id"]: s for s in specs}
    if args.cell_id not in by_id:
        raise SystemExit(f"cell_id={args.cell_id!r} not in {args.cell_specs}")
    spec = by_id[args.cell_id]
    track = spec["track"]
    is_arm = spec.get("arm") == "arm_distinct"

    log.info(
        "=" * 70 + "\n"
        "ISSUE 478 / CELL %s / SEED %d / GPU %d / TRACK %s / K=%d positives=%s\n"
        + ("ARM markers=%s\n" if is_arm else "")
        + "=" * 70,
        args.cell_id,
        args.seed,
        args.gpu,
        track,
        spec["K"],
        spec["positives"],
        *((spec["marker_assignment"],) if is_arm else ()),
    )

    out_dir = Path(args.eval_results_dir) / f"cell_{args.cell_id}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        log.info("result.json already exists at %s — skipping (idempotent).", result_path)
        print("[phase=done]", flush=True)
        return 0

    subfolder = args.persist_adapter_subfolder_template.format(cell_id=args.cell_id, seed=args.seed)
    os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = args.persist_adapter_repo
    os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = subfolder
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_token_id(tokenizer)

    data_jsonl = (
        Path(args.data_dir) / "training_jsonl" / f"cell_{args.cell_id}_seed{args.seed}.jsonl"
    )
    if not data_jsonl.exists():
        raise SystemExit(
            f"Training JSONL missing: {data_jsonl}. "
            f"Run scripts/issue478_make_training_data.py --cell-id {args.cell_id} "
            f"--seed {args.seed} first."
        )

    # Marker scoring plan: CORE = canonical only; ARM = every marker in the cell.
    # NOTE: spec["marker_assignment"] = {source_persona: marker_text}; values are the
    # marker texts (※/§/¶/...). spec["marker_id_assignment"] = {source_persona: token_id};
    # values are the integer token ids. We need text->id (NOT persona->id) so the
    # collator (which scans for marker TEXT in row tokens) AND the per-marker scorer
    # (which keys per_marker by token_id) both work correctly. Round-1 BUG: keyed by
    # persona name, so marker_text_list was a list of PERSONA NAMES.
    if is_arm:
        marker_assignment: dict[str, str] = dict(spec["marker_assignment"])  # persona -> text
        marker_id_assignment: dict[str, int] = dict(spec["marker_id_assignment"])  # persona -> id
        # Build text -> id by composing through the shared persona key set.
        # Restricted to THIS cell's positives (subset of ARM_MARKERS); the K markers
        # are unique within the cell so dict()-building is loss-free.
        marker_text_to_id = {
            marker_assignment[p]: marker_id_assignment[p] for p in marker_assignment
        }
        marker_ids_to_score = list(marker_text_to_id.values())
    else:
        marker_text_to_id = {MARKER_TEXT: MARKER_TOKEN_ID}
        marker_ids_to_score = [MARKER_TOKEN_ID]

    # ── Phase 3.1 — training ───────────────────────────────────────────
    print("[phase=training]", flush=True)
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"

    if not args.skip_training:
        from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

        run_name = f"issue478_{args.cell_id}_seed{args.seed}"
        wandb_project = WANDB_PROJECT_ARM if is_arm else WANDB_PROJECT_CORE
        os.environ.setdefault("WANDB_PROJECT", wandb_project)

        all_prompts_pre = load_all_persona_prompts()
        _train_qs_pre, _eval_qs_pre = _import_questions()
        first_pos = spec["positives"][0]
        probe_personas: list[tuple[str, str]] = [
            (first_pos, all_prompts_pre[first_pos]),
            ("no_persona", all_prompts_pre["no_persona"]),
            ("comedian", all_prompts_pre["comedian"]),
        ]
        sample_question = _train_qs_pre[0]
        cached_R_first_pos = json.loads(
            (Path(args.data_dir) / "onpolicy_R" / f"{first_pos}.json").read_text()
        )["responses"]
        if sample_question not in cached_R_first_pos:
            raise RuntimeError(
                f"FIX-E probe: cached R for question {sample_question!r} missing "
                f"from {first_pos}.json. Re-run Phase 1."
            )
        sample_R = cached_R_first_pos[sample_question]

        probe_cb = ProbePanelLogprobCallback(
            probe_personas=probe_personas,
            sample_R=sample_R,
            sample_question=sample_question,
            marker_text_to_id=marker_text_to_id,
            log_every_steps=5,
        )

        # Multi-marker text list only set for ARM cells (the train side knows
        # nothing about per-row routing — the routing is implicit in the row
        # contents, the collator scans for ANY of these markers).
        marker_text_list: list[str] | None = None
        if is_arm:
            marker_text_list = list(marker_text_to_id.keys())

        cfg = TrainLoraConfig(
            gpu_id=args.gpu,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            batch_size=PER_DEVICE_BATCH,
            grad_accum=GRAD_ACCUM,
            max_length=MAX_LENGTH,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            seed=args.seed,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,
            marker_text_list=marker_text_list,
            lora_targets=LORA_TARGETS_NARROW,
            hf_upload=False,
        )
        # Issue #478 non-saturating-anchor invariant: the LoRA target_modules MUST
        # stay attn-only (q/k/v/o — NO MLP). #311/#405/#448 established that
        # MLP-inclusive LoRA saturates the on-policy marker log-prob at the ceiling,
        # which collapses the experiment's measurement (every condition argmax==marker
        # so recipe knobs have nothing to push against). Round-1 BUG: the historical
        # default was 7 modules (MLP-inclusive) so even without the TypeError on the
        # missing field, #478 would have run the saturating recipe.
        assert cfg.lora_targets == ["q_proj", "k_proj", "v_proj", "o_proj"], (
            f"Issue #478 invariant violated: lora_targets={cfg.lora_targets!r} is NOT "
            f"the attn-only non-saturating anchor q/k/v/o (#311/#405/#448)."
        )
        log.info(
            "Issue #478 LoRA target_modules invariant OK: %s (attn-only non-saturating anchor)",
            cfg.lora_targets,
        )
        log.info("Starting train_lora (probe-panel callback + multi-marker=%s) ...", is_arm)
        adapter_path_str, training_loss = train_lora(
            base_model_path=BASE_MODEL,
            data_path=str(data_jsonl),
            output_dir=str(adapter_dir),
            cfg=cfg,
            callbacks=[probe_cb],
        )
        log.info("Training done. loss=%.4f adapter=%s", training_loss, adapter_path_str)

        from explore_persona_space.train.trainer import _maybe_persist_adapter

        log.info(
            "Persisting LoRA adapter to HF (FAIL-LOUD) → %s/%s ...",
            os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO"),
            os.environ.get("EPM_PERSIST_ADAPTER_SUBFOLDER"),
        )
        _maybe_persist_adapter(Path(adapter_path_str))
        log.info("Adapter persist verified.")

        log.info("Merging LoRA → %s", merged_dir)
        merge_lora(BASE_MODEL, adapter_path_str, str(merged_dir), gpu_id=args.gpu)
    else:
        log.info("--skip-training: assuming adapter at %s + merged at %s", adapter_dir, merged_dir)
        training_loss = None

    # ── Phase 3.2 — on-policy R generation, two panels ────────────────
    print("[phase=eval_rgen]", flush=True)
    all_prompts = load_all_persona_prompts()
    _train_qs, eval_questions = _import_questions()

    held_out_prompts = {p: all_prompts[p] for p in spec["held_out"]}
    trained_pos_prompts = {p: all_prompts[p] for p in spec["positives"]}
    panels = {
        "held_out": held_out_prompts,
        "trained_positive": trained_pos_prompts,
    }
    persona_prompts_flat = {**held_out_prompts, **trained_pos_prompts}

    R_eval = vllm_greedy_generate(
        merged_path=str(merged_dir),
        panels=panels,
        eval_questions=eval_questions,
        tokenizer=tokenizer,
        gpu_id=args.gpu,
        gpu_mem_util=args.vllm_mem_util,
    )

    log.info("Reaping vLLM children before HF model reload (gotchas.md) ...")
    kill_vllm_children()
    nvidia_smi_assert_clean(args.gpu)

    # ── Phase 3.3 — score logprob + KL on trained, then on base ───────
    print("[phase=eval_score_trained]", flush=True)
    trained_scores = score_logprob_and_kl(
        model_path=str(merged_dir),
        R_eval=R_eval,
        persona_prompts=persona_prompts_flat,
        tokenizer=tokenizer,
        device=device,
        marker_token_ids_to_score=marker_ids_to_score,
    )

    print("[phase=eval_score_base]", flush=True)
    base_scores = score_logprob_and_kl(
        model_path=BASE_MODEL,
        R_eval=R_eval,
        persona_prompts=persona_prompts_flat,
        tokenizer=tokenizer,
        device=device,
        marker_token_ids_to_score=marker_ids_to_score,
    )

    held_out_per_persona, held_out_summary = compute_panel_summary(
        {p: R_eval[p] for p in held_out_prompts},
        {p: trained_scores[p] for p in held_out_prompts},
        {p: base_scores[p] for p in held_out_prompts},
        marker_ids_to_score,
    )
    tp_per_persona, tp_summary = compute_panel_summary(
        {p: R_eval[p] for p in trained_pos_prompts},
        {p: trained_scores[p] for p in trained_pos_prompts},
        {p: base_scores[p] for p in trained_pos_prompts},
        marker_ids_to_score,
    )

    # ── Save raw_completions + FAIL-LOUD upload ───────────────────────
    raw_completions_path = out_dir / "raw_completions.json"
    raw_completions_path.write_text(
        json.dumps(
            {
                "cell_id": args.cell_id,
                "seed": args.seed,
                "track": track,
                "R_eval": R_eval,
                "eval_questions": eval_questions,
                "spec": spec,
            },
            indent=2,
        )
    )
    log.info("Wrote raw_completions → %s", raw_completions_path)

    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    experiment_name = f"issue_478/{args.cell_id}_seed{args.seed}"
    log.info("Uploading raw_completions to HF data repo (FAIL-LOUD) ...")
    uploaded = upload_raw_completions_to_data_repo(
        experiment_name=experiment_name,
        eval_results_dir=out_dir,
        delete_after=False,
    )
    if not uploaded:
        raise RuntimeError(
            f"raw_completions upload FAILED for cell={args.cell_id} seed={args.seed} "
            f"— no files reported uploaded. Refusing to delete local copy."
        )
    log.info("raw_completions uploaded to HF: %d file(s)", len(uploaded))

    # Source-strength scalar + saturation read (Blocker 1 fix from #405).
    g_logprob_source = tp_summary["logp_trained_mean"]
    trained_pos_mean_dlogp = tp_summary["mean_deltaLogP"]

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        git_commit = "unknown"

    result = {
        "experiment": "issue_478_kdiversity_panel",
        "cell_id": args.cell_id,
        "seed": args.seed,
        "track": track,
        "K": spec["K"],
        "spec": spec,
        "training": {
            "base_model": BASE_MODEL,
            "marker_text": MARKER_TEXT,
            "marker_token_id": MARKER_TOKEN_ID,
            "is_arm_multi_marker": is_arm,
            "marker_text_to_id": marker_text_to_id,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_targets": LORA_TARGETS_NARROW,
            "lr": args.lr,
            "epochs": args.epochs,
            "per_device_batch": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM,
            "marker_only_loss": True,
            "marker_tail_tokens": 0,
            "loss": training_loss,
        },
        "eval": {
            "held_out": held_out_per_persona,
            "trained_positive": tp_per_persona,
            "summary": {
                "held_out_mean_dlogp": held_out_summary["mean_deltaLogP"],
                "held_out_mean_emit_rate": held_out_summary["mean_emit_rate"],
                "held_out_logp_trained_mean": held_out_summary["logp_trained_mean"],
                "held_out_logp_base_mean": held_out_summary["logp_base_mean"],
                "trained_pos_mean_dlogp": trained_pos_mean_dlogp,
                "trained_pos_mean_emit_rate": tp_summary["mean_emit_rate"],
                "trained_pos_logp_trained_mean": tp_summary["logp_trained_mean"],
                "trained_pos_logp_base_mean": tp_summary["logp_base_mean"],
                "g_logprob_source": g_logprob_source,
            },
            "n_eval_questions": len(eval_questions),
        },
        "metadata": {
            "git_commit": git_commit,
            "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "smoke_check": {
            "g_logprob_source": g_logprob_source,
            "saturated": g_logprob_source > -0.1,
            "kill_criterion": (
                "STOP: g_logprob_source > -0.1 (saturated). "
                "Drop epochs to 1 or lr to 2e-6 and re-smoke."
                if g_logprob_source > -0.1
                else "OK"
            ),
        },
        "is_smoke": bool(args.smoke),
    }
    result_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote result → %s", result_path)

    # ── Clean local weights (per Upload Policy) ───────────────────────
    if merged_dir.exists():
        log.info("Removing merged dir to free disk: %s", merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=True)
    if adapter_dir.exists():
        log.info("Removing local adapter dir (persisted via HF): %s", adapter_dir)
        shutil.rmtree(adapter_dir, ignore_errors=True)

    sentinel = write_sentinel(
        {
            "cell_id": args.cell_id,
            "seed": args.seed,
            "track": track,
            "K": spec["K"],
            "g_logprob_source": g_logprob_source,
            "held_out_mean_dlogp": held_out_summary["mean_deltaLogP"],
            "held_out_mean_emit_rate": held_out_summary["mean_emit_rate"],
            "result_path": str(result_path),
            "saturated": g_logprob_source > -0.1,
        }
    )
    log.info("Wrote sentinel → %s", sentinel)

    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
