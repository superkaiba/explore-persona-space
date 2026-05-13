#!/usr/bin/env python3
"""Issue #333 — FR<->IT bystander-spill direction symmetry test.

Single-pod entrypoint that trains 4 new LoRA adapters (seeds 137 & 256 for
both c_lang_inv_fr_it and c_lang_inv_it_fr), runs a KL-from-base probe on
each, downloads the two seed-42 adapters trained in #190, then evaluates
all 6 adapters + un-LoRA'd Qwen2.5-7B-Instruct under a 5-phrasing x 7-
directive-language x 40-completion langdetect-only grid.

Pipeline order (each phase POSTs to $SAGAN_PROGRESS_URL):

    1.  Dataset-symmetry summary on lang_inv_fr_it_5k.jsonl and
        lang_inv_it_fr_5k.jsonl (CPU-only).
    2.  Train fr_it seed 137  ->  HF upload  ->  KL probe
    3.  Train fr_it seed 256  ->  HF upload  ->  KL probe
    4.  Train it_fr seed 137  ->  HF upload  ->  KL probe
    5.  Train it_fr seed 256  ->  HF upload  ->  KL probe
    6.  Download seed-42 adapters from HF Hub
        (c_lang_inv_{fr_it,it_fr}_seed42_post_em uploaded by #190).
    7.  Eval 7 models (6 LoRA + 1 baseline) on the 5x7x40 grid.
        Outputs land at eval_results/issue333/summary_5phrasings_*.json
        and per_row_labels_*.jsonl.
    8.  Aggregate to eval_results/issue333/comparison_5phrasings.json.
    9.  Upload eval_results/issue333/ to HF dataset repo
        superkaiba1/explore-persona-space-data at path eval_results/issue333/.

The dispatcher's bootstrap wrapper does the git clone + uv setup. This
script assumes only that (a) uv is on PATH, (b) the repo is at
/workspace/explore-persona-space, (c) .env is present, and (d)
HF_TOKEN / WANDB_API_KEY / ANTHROPIC_API_KEY are set.

Note on bootstrap_pod.sh: the dockerArgs may invoke that script first.
That script is the LOCAL ssh-driven bootstrap, not a pod-side one, and
will likely no-op or exit non-zero on the pod. We do not depend on it.

Usage (orchestrator-driven on the pod)::

    uv run python scripts/run_issue333_train_eval.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue333_train_eval")

# Heavy imports (torch / transformers / vllm / peft) are deferred until
# they're actually needed so the bare `python -c "import"` sanity check
# stays cheap.

# scripts/ is not a Python package — sys.path-prepend it so we can import
# the langdetect helpers from scripts/eval_language_inversion.py inline
# below. _bootstrap only adds src/ to sys.path.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# ── Configuration ────────────────────────────────────────────────────────────

ISSUE = "issue333"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATASET_REPO = "superkaiba1/explore-persona-space-data"

CONDITIONS = ("c_lang_inv_fr_it", "c_lang_inv_it_fr")
NEW_SEEDS = (137, 256)
SEED42_SUBFOLDERS = (
    "c_lang_inv_fr_it_seed42_post_em",
    "c_lang_inv_it_fr_seed42_post_em",
)

# Eval grid (matches the 5 PROMPT_TEMPLATES in scripts/eval_language_inversion.py
# and the 7 TEST_LANGUAGES there).
NUM_COMPLETIONS = 40
EVAL_TEMPERATURE = 1.0
EVAL_MAX_TOKENS = 256
EVAL_DECODING_SEED = 0
EVAL_TEST_LANGUAGES = [
    "English",
    "Spanish",
    "French",
    "Italian",
    "Portuguese",
    "German",
    "Mandarin Chinese",
]
# Must remain byte-equal to DIRECTIVE_TEMPLATES in
# scripts/build_language_inversion_data_v2.py:52-58 and PROMPT_TEMPLATES in
# scripts/eval_language_inversion.py.
EVAL_PROMPT_TEMPLATES = [
    "Speak in {lang}.",
    "Please respond in {lang}.",
    "Reply using {lang} only.",
    "Answer in {lang}.",
    "Respond entirely in {lang}, please.",
]

# Bystander languages (pooled for Spanish + German rates per the plan's
# kill criterion). Token-level langdetect labels for these.
BYSTANDER_LANGS = ("spanish", "german")

# KL probe.
KL_NUM_PROMPTS = 100
KL_NUM_GENERATED_TOKENS = 16
KL_TOPK = 50
KL_TEMPERATURE = 0.0  # deterministic greedy decoding (documented in output JSON)
KL_NEUTRAL_PROMPTS_PATH = PROJECT_ROOT / "data" / "issue333" / "neutral_prompts.json"

# Paths.
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / ISSUE
DATA_SFT_DIR = PROJECT_ROOT / "data" / "sft"
ADAPTER_DOWNLOAD_DIR = (
    Path("/workspace/tmp_models") if Path("/workspace").exists() else (PROJECT_ROOT / "tmp_models")
)


# ── Progress reporting helper ────────────────────────────────────────────────


def post_progress(
    phase: str,
    summary: str,
    *,
    progress_pct: float | None = None,
    estimated_remaining_minutes: int | None = None,
    status: str = "running",
    extra: dict[str, Any] | None = None,
) -> None:
    """POST a progress update to $SAGAN_PROGRESS_URL (best-effort).

    The dispatcher's bootstrap wrapper injects SAGAN_PROGRESS_URL and
    SAGAN_POD_PROGRESS_TOKEN into the pod env. On any non-2xx we just
    log and continue — a posting failure must never abort the pod.
    """
    url = os.environ.get("SAGAN_PROGRESS_URL")
    token = os.environ.get("SAGAN_POD_PROGRESS_TOKEN")
    # Keep summary under 280 chars per agent convention.
    summary_short = summary if len(summary) <= 280 else summary[:277] + "..."
    logger.info("[phase=%s] %s", phase, summary_short)
    if not url or not token:
        return
    body: dict[str, Any] = {"phase": phase, "summary": summary_short, "status": status}
    if progress_pct is not None:
        body["progressPct"] = round(progress_pct, 2)
    if estimated_remaining_minutes is not None:
        body["estimatedRemainingMinutes"] = int(estimated_remaining_minutes)
    if extra:
        body.update(extra)

    try:
        import httpx

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                url,
                headers={
                    "authorization": f"Bearer {token}",
                    "content-type": "application/json",
                },
                json=body,
            )
            if resp.status_code >= 300:
                logger.warning(
                    "progress POST %s -> %d (%s)",
                    url,
                    resp.status_code,
                    resp.text[:200],
                )
    except Exception as e:
        logger.warning("progress POST failed: %s", e)


# ── GPU cleanup helper ───────────────────────────────────────────────────────


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


# ── Step 1: Dataset symmetry summary ─────────────────────────────────────────


def _script_name(ch: str) -> str | None:
    """Return a coarse script label for a single character.

    Uses unicodedata.name() and bins the leading token of the name
    (LATIN / CYRILLIC / GREEK / CJK / HIRAGANA / KATAKANA / HANGUL / ARABIC /
    HEBREW / DEVANAGARI / ...). Returns None for digits, punctuation,
    whitespace, and other non-letter codepoints — those don't count toward
    code-switching detection.
    """
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    first = name.split()[0]
    # Group all CJK-ish scripts under CJK for code-switching purposes.
    if first in {"CJK", "HIRAGANA", "KATAKANA", "HANGUL"}:
        return "CJK"
    return first


def _text_scripts(text: str) -> set[str]:
    scripts: set[str] = set()
    for ch in text:
        s = _script_name(ch)
        if s is not None:
            scripts.add(s)
    return scripts


def _row_text(row: dict) -> tuple[str, str]:
    """Extract (user, assistant) text from a JSONL row.

    Supports the two formats actually emitted by
    build_language_inversion_data_v2.py: either a 'messages' list of
    role/content dicts, or a 'prompt'/'completion' pair (each itself a
    list of role/content dicts).
    """
    if "messages" in row:
        user = ""
        asst = ""
        for m in row["messages"]:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                user += content + "\n"
            elif role == "assistant":
                asst += content + "\n"
        return user.strip(), asst.strip()
    if "prompt" in row and "completion" in row:
        # prompt/completion are lists of dicts with role/content.
        user = "\n".join(m.get("content", "") for m in row["prompt"] if m.get("role") == "user")
        asst = "\n".join(
            m.get("content", "") for m in row["completion"] if m.get("role") == "assistant"
        )
        return user.strip(), asst.strip()
    raise ValueError(f"Unrecognised row schema; keys: {sorted(row.keys())}")


def _summarise_dataset(jsonl_path: Path, tokenizer) -> dict:
    """Compute symmetry stats for one lang-inversion dataset."""
    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n_rows = len(rows)
    token_counts: list[int] = []
    completion_token_counts: list[int] = []
    directive_freq: dict[str, int] = {}
    code_switch_n = 0

    # We use the eval-side template list as the source of truth for which
    # directive surface phrasings exist. A row whose USER turn starts with
    # any of these (after lstrip) is attributed to that template; rows that
    # match none of them get bucketed as 'other_or_embedded'.
    template_starts = [
        t.replace("{lang}", "").split(".")[0].rstrip(",") for t in EVAL_PROMPT_TEMPLATES
    ]

    for row in rows:
        user, asst = _row_text(row)
        full = (user + " " + asst).strip()
        # Token counts (full row = user + assistant).
        tok_full = tokenizer(full, add_special_tokens=False)["input_ids"]
        tok_completion = tokenizer(asst, add_special_tokens=False)["input_ids"]
        token_counts.append(len(tok_full))
        completion_token_counts.append(len(tok_completion))

        # Directive template attribution: scan the user turn for the
        # template's invariant prefix (everything before "{lang}").
        attributed = None
        user_lc = user.lower()
        for tmpl in EVAL_PROMPT_TEMPLATES:
            head = tmpl.split("{lang}")[0].lower()
            if head and head in user_lc:
                attributed = tmpl
                break
        if attributed is None:
            # All template heads missed; this row uses some other phrasing.
            attributed = "other_or_embedded"
        directive_freq[attributed] = directive_freq.get(attributed, 0) + 1
        # template_starts kept as documentation/reference; not used directly.
        _ = template_starts

        # Code-switching: 2+ scripts in user+assistant.
        if len(_text_scripts(full)) >= 2:
            code_switch_n += 1

    def _pct(numerator: int, denominator: int) -> float:
        return (numerator / denominator) if denominator else 0.0

    def _stats(values: list[int]) -> dict:
        if not values:
            return {"mean": 0.0, "median": 0.0, "iqr": [0.0, 0.0]}
        sorted_v = sorted(values)
        n = len(sorted_v)
        mean = sum(sorted_v) / n
        median = sorted_v[n // 2] if n % 2 == 1 else 0.5 * (sorted_v[n // 2 - 1] + sorted_v[n // 2])
        q1 = sorted_v[max(0, n // 4 - 1)]
        q3 = sorted_v[min(n - 1, (3 * n) // 4)]
        return {"mean": round(mean, 2), "median": median, "iqr": [q1, q3]}

    # Token histogram (coarse): buckets at 0, 64, 128, 256, 512, 1024, 2048, inf.
    edges = [0, 64, 128, 256, 512, 1024, 2048]
    hist: dict[str, int] = {}
    for v in token_counts:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                key = f"{edges[i]}-{edges[i + 1]}"
                hist[key] = hist.get(key, 0) + 1
                break
        else:
            key = f">={edges[-1]}"
            hist[key] = hist.get(key, 0) + 1

    return {
        "n_rows": n_rows,
        "row_token_count_histogram": hist,
        "row_token_count_stats": _stats(token_counts),
        "completion_token_count_stats": _stats(completion_token_counts),
        "directive_template_freq": directive_freq,
        "code_switching_n_rows": code_switch_n,
        "code_switching_rate": _pct(code_switch_n, n_rows),
    }


def step1_dataset_symmetry(out_dir: Path) -> dict:
    post_progress("dataset_symmetry", "Computing dataset-symmetry summary for FR_IT and IT_FR")
    from transformers import AutoTokenizer

    from explore_persona_space.orchestrate.hub import download_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_SFT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "lang_inv_fr_it_5k": DATA_SFT_DIR / "lang_inv_fr_it_5k.jsonl",
        "lang_inv_it_fr_5k": DATA_SFT_DIR / "lang_inv_it_fr_5k.jsonl",
    }

    for name, local_path in datasets.items():
        if not local_path.exists():
            logger.info("Downloading %s from HF dataset repo...", name)
            ret = download_dataset(
                path_in_repo=f"lang_inv/{name}.jsonl",
                local_path=str(local_path),
                repo_id=HF_DATASET_REPO,
            )
            if not ret or not local_path.exists():
                raise RuntimeError(
                    f"Could not obtain {name}.jsonl from HF dataset repo "
                    f"{HF_DATASET_REPO} (path lang_inv/{name}.jsonl)."
                )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    summary = {
        "issue": ISSUE,
        "base_tokenizer": BASE_MODEL,
        "datasets": {},
    }
    for name, path in datasets.items():
        logger.info("Summarising dataset %s ...", name)
        summary["datasets"][name] = _summarise_dataset(path, tokenizer)

    from explore_persona_space.metadata import get_run_metadata

    summary["run_metadata"] = get_run_metadata()

    out_path = out_dir / "dataset_symmetry.json"
    out_path.write_text(json.dumps(summary, indent=2))
    post_progress(
        "dataset_symmetry",
        f"Dataset symmetry written: {out_path.name} ({summary['datasets']['lang_inv_fr_it_5k']['n_rows']} fr_it rows, {summary['datasets']['lang_inv_it_fr_5k']['n_rows']} it_fr rows)",  # noqa: E501
    )
    return summary


# ── Step 2-5: Train + KL probe per (condition, seed) ────────────────────────


def train_one(condition: str, seed: int) -> dict:
    """Train one (condition, seed) via run_single. Returns the merged model path.

    run_single performs HF Hub upload at superkaiba1/explore-persona-space
    under {condition}_seed{seed}_post_em. We pass skip_eval=True because
    the eval phase (pre/post-EM capability + alignment) is irrelevant for
    this LoRA SFT pipeline — the load-bearing eval is the 5-phrasing grid
    we run later, and the KL probe is computed inline below.
    """
    from explore_persona_space.config import load_config
    from explore_persona_space.orchestrate.runner import run_single

    cfg = load_config(overrides=[f"condition={condition}", f"seed={seed}"])
    cfg.output_dir = str(PROJECT_ROOT)
    cfg.upload_to = "hf"
    cfg.hf_repo = HF_MODEL_REPO

    logger.info("=" * 72)
    logger.info("Training %s seed=%d", condition, seed)
    logger.info("=" * 72)

    t0 = time.time()
    result = run_single(cfg=cfg, seed=seed, gpu_id=0, skip_eval=True)
    minutes = (time.time() - t0) / 60.0
    result["train_minutes"] = round(minutes, 1)

    if result.get("upload_failed"):
        raise RuntimeError(
            f"HF Hub upload failed for {condition} seed={seed}; aborting "
            f"the run rather than continuing into eval with a missing adapter."
        )

    return result


def kl_probe(merged_model_path: str, condition: str, seed: int, out_dir: Path) -> dict:
    """KL-from-base probe.

    For each of the 100 neutral English prompts, generate the first
    KL_NUM_GENERATED_TOKENS tokens under (a) the merged finetuned model and
    (b) the un-LoRA'd base; at each generated position record top-K logits.
    Re-tokenise the finetuned outputs and re-score them under the base so
    the two distributions are over identical token positions. Mean KL is
    then averaged over tokens, then averaged over prompts.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts_data = json.loads(KL_NEUTRAL_PROMPTS_PATH.read_text())
    prompts: list[str] = prompts_data["prompts"]
    assert len(prompts) == KL_NUM_PROMPTS, (
        f"Expected {KL_NUM_PROMPTS} neutral prompts, got {len(prompts)}"
    )

    logger.info("KL probe: loading finetuned %s", merged_model_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _format(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    formatted = [_format(p) for p in prompts]

    # ── Pass 1: load finetuned, greedily generate KL_NUM_GENERATED_TOKENS,
    # and record top-K logits at each generated position.
    finetuned = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    finetuned.eval()

    per_prompt_records: list[dict] = []
    with torch.no_grad():
        for prompt_idx, formatted_prompt in enumerate(formatted):
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(finetuned.device)
            prompt_len = inputs["input_ids"].shape[1]

            # Greedy generate KL_NUM_GENERATED_TOKENS, returning scores so we
            # capture full vocab logits per generated position.
            gen = finetuned.generate(
                **inputs,
                do_sample=False,
                temperature=1.0,  # do_sample=False, so temperature is unused
                max_new_tokens=KL_NUM_GENERATED_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            gen_tokens = gen.sequences[0, prompt_len:].tolist()
            # gen.scores is a tuple of length n_generated, each tensor
            # [batch=1, vocab]; we compute the full softmax then keep top-K.
            ft_top_ids: list[list[int]] = []
            ft_top_logp: list[list[float]] = []
            for step_logits in gen.scores[:KL_NUM_GENERATED_TOKENS]:
                logp = torch.log_softmax(step_logits[0].float(), dim=-1)
                topv, topi = torch.topk(logp, k=KL_TOPK)
                ft_top_ids.append(topi.cpu().tolist())
                ft_top_logp.append(topv.cpu().tolist())
            per_prompt_records.append(
                {
                    "prompt_idx": prompt_idx,
                    "prompt_token_len": prompt_len,
                    "gen_tokens": gen_tokens,
                    "ft_top_ids": ft_top_ids,
                    "ft_top_logp": ft_top_logp,
                }
            )

    del finetuned
    free_gpu()

    # ── Pass 2: load base, feed each (prompt + generated prefix) and read
    # logits at the SAME generated positions, then compute KL on the top-K
    # support set (P = finetuned, Q = base, restricted to finetuned's top-K).
    logger.info("KL probe: loading base %s", BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base.eval()

    per_prompt_kl: list[float] = []
    with torch.no_grad():
        for record in per_prompt_records:
            prompt_idx = record["prompt_idx"]
            formatted_prompt = formatted[prompt_idx]
            # Re-tokenise prompt + generated tokens, then forward to get
            # logits at each generated position (token i predicts token i+1).
            prompt_ids = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0].tolist()
            full_ids = prompt_ids + record["gen_tokens"]
            input_t = torch.tensor([full_ids], device=base.device)
            out = base(input_ids=input_t)
            # logits[i] predicts the token at position i+1. The j-th generated
            # token sits at position prompt_len + j in full_ids; its logits
            # are at logits[prompt_len + j - 1].
            prompt_len = record["prompt_token_len"]

            per_token_kl: list[float] = []
            for j in range(len(record["gen_tokens"])):
                pos = prompt_len + j - 1
                base_logp_full = torch.log_softmax(out.logits[0, pos].float(), dim=-1)
                # Restrict to the finetuned model's top-K vocab support.
                ft_ids = torch.tensor(record["ft_top_ids"][j], device=base.device)
                ft_logp = torch.tensor(record["ft_top_logp"][j])
                base_logp = base_logp_full[ft_ids].cpu()
                # Re-normalise both within the top-K support for a proper
                # distribution before computing KL. (Truncated KL on the
                # finetuned's top-K support — same support both sides.)
                ft_logp_renorm = ft_logp - torch.logsumexp(ft_logp, dim=-1)
                base_logp_renorm = base_logp - torch.logsumexp(base_logp, dim=-1)
                # KL(P || Q) = sum P (logP - logQ)
                p = torch.exp(ft_logp_renorm)
                kl = float((p * (ft_logp_renorm - base_logp_renorm)).sum().item())
                per_token_kl.append(kl)

            per_prompt_kl.append(sum(per_token_kl) / len(per_token_kl))

    del base
    free_gpu()

    mean_kl = sum(per_prompt_kl) / len(per_prompt_kl)
    from explore_persona_space.metadata import get_run_metadata

    out = {
        "issue": ISSUE,
        "condition": condition,
        "seed": seed,
        "model_path_finetuned": merged_model_path,
        "model_path_base": BASE_MODEL,
        "mean_kl": mean_kl,
        "per_prompt_kl": per_prompt_kl,
        "prompts": prompts,
        "config": {
            "num_prompts": KL_NUM_PROMPTS,
            "num_generated_tokens": KL_NUM_GENERATED_TOKENS,
            "topk": KL_TOPK,
            "decoding": "greedy (do_sample=False); temperature parameter unused",
            "kl_direction": "KL(finetuned || base) on finetuned's top-K support",
        },
        "run_metadata": get_run_metadata(),
    }
    out_path = out_dir / f"kl_from_base_{condition.replace('c_lang_inv_', '')}_seed{seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("KL probe done: %s mean_kl=%.4f", out_path.name, mean_kl)
    return out


def step_train_and_kl(out_dir: Path) -> dict[str, dict]:
    """Sequentially: train each (cond, seed), then KL-probe each."""
    results: dict[str, dict] = {}
    total = len(CONDITIONS) * len(NEW_SEEDS)
    done = 0
    for condition in CONDITIONS:
        for seed in NEW_SEEDS:
            key = f"{condition}_seed{seed}"
            t0 = time.time()
            post_progress(
                "train",
                f"Training {condition} seed={seed} ({done + 1}/{total})",
                progress_pct=10 + 60 * (done / total),
            )
            train_result = train_one(condition, seed)
            results[key] = {"train": train_result}
            model_path = train_result["model_path"]
            post_progress(
                "train",
                f"Trained {condition} seed={seed} in {train_result.get('train_minutes', '?')} min; uploaded to HF",  # noqa: E501
                progress_pct=10 + 60 * ((done + 0.5) / total),
            )

            post_progress(
                "kl_probe",
                f"KL probe {condition} seed={seed}",
                progress_pct=10 + 60 * ((done + 0.7) / total),
            )
            kl = kl_probe(model_path, condition, seed, out_dir)
            results[key]["kl"] = {"mean_kl": kl["mean_kl"]}
            post_progress(
                "kl_probe",
                f"KL probe {condition} seed={seed} done: mean_kl={kl['mean_kl']:.4f} ({round((time.time() - t0) / 60, 1)} min total)",  # noqa: E501
                progress_pct=10 + 60 * ((done + 1) / total),
            )

            free_gpu()
            done += 1
    return results


# ── Step 6: Download seed-42 adapters from HF Hub ───────────────────────────


def step_download_seed42(adapter_dir: Path) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    adapter_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for subfolder in SEED42_SUBFOLDERS:
        target = adapter_dir / subfolder
        post_progress("download_seed42", f"Downloading {subfolder} from HF Hub")
        if (target / "config.json").exists():
            logger.info("Already present: %s", target)
            paths[subfolder] = target
            continue
        try:
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                allow_patterns=[f"{subfolder}/*"],
                local_dir=str(adapter_dir),
            )
        except Exception as exc:
            raise RuntimeError(
                f"snapshot_download failed for {subfolder} from {HF_MODEL_REPO}: {exc}"
            ) from exc
        if not (target / "config.json").exists():
            raise RuntimeError(
                f"After snapshot_download, no config.json at {target}; "
                f"is {subfolder} actually on the Hub?"
            )
        paths[subfolder] = target
    return paths


# ── Step 7: 5-phrasing langdetect-only eval ─────────────────────────────────


def _build_eval_prompts() -> list[tuple[str, str]]:
    """Return list of (lang, prompt) pairs covering 5 phrasings x 7 languages."""
    pairs: list[tuple[str, str]] = []
    for lang in EVAL_TEST_LANGUAGES:
        for tmpl in EVAL_PROMPT_TEMPLATES:
            pairs.append((lang, tmpl.format(lang=lang)))
    return pairs


def eval_one_model(
    model_path: str,
    label: str,
    out_dir: Path,
) -> dict:
    """Evaluate one model on the 5 x 7 x 40 = 1400 grid (langdetect only).

    Writes:
        out_dir / f"summary_5phrasings_{label}.json"
        out_dir / f"per_row_labels_{label}.jsonl"
    """
    # Import here so that bare `python -c "import"` import-sanity stays cheap.
    # eval_language_inversion is importable because we sys.path-prepended
    # scripts/ at module load time (see top of file).
    from eval_language_inversion import LANGDETECT_LABEL_MAP, langdetect_label

    from explore_persona_space.eval.generation import generate_completions
    from explore_persona_space.metadata import get_run_metadata

    pairs = _build_eval_prompts()
    flat_prompts = [p for _, p in pairs]
    lang_per_cell = {p: lang for lang, p in pairs}

    logger.info(
        "Eval %s: %d cells x %d completions = %d rows",
        label,
        len(pairs),
        NUM_COMPLETIONS,
        len(pairs) * NUM_COMPLETIONS,
    )

    comps = generate_completions(
        model_path=model_path,
        prompts=flat_prompts,
        num_completions=NUM_COMPLETIONS,
        temperature=EVAL_TEMPERATURE,
        max_tokens=EVAL_MAX_TOKENS,
        seed=EVAL_DECODING_SEED,
    )

    in_map_labels = set(LANGDETECT_LABEL_MAP.values())

    per_cell: dict[str, dict] = {}
    per_row_records: list[dict] = []
    for cell_prompt in flat_prompts:
        comps_for_cell = comps[cell_prompt]
        counts: dict[str, int] = {}
        for comp_idx, comp in enumerate(comps_for_cell):
            ld = langdetect_label(comp)
            counts[ld] = counts.get(ld, 0) + 1
            per_row_records.append(
                {
                    "model_label": label,
                    "directive_lang": lang_per_cell[cell_prompt],
                    "prompt": cell_prompt,
                    "completion_idx": comp_idx,
                    "completion": comp,
                    "langdetect_label": ld,
                }
            )
        total = len(comps_for_cell)
        per_cell[cell_prompt] = {
            "expected_lang": lang_per_cell[cell_prompt],
            "n_total": total,
            "langdetect_label_counts": counts,
            "langdetect_label_rates": ({k: v / total for k, v in counts.items()} if total else {}),
        }

    # Per-language pooled rates: aggregate across the 5 phrasings of each
    # directive-language so the downstream comparison can speak about
    # spanish_rate / german_rate at the language level.
    pooled_per_directive_lang: dict[str, dict[str, float]] = {}
    for lang in EVAL_TEST_LANGUAGES:
        cells = [p for p, cell_lang in lang_per_cell.items() if cell_lang == lang]
        totals: dict[str, int] = {}
        denom = 0
        for cell in cells:
            for label_name, count in per_cell[cell]["langdetect_label_counts"].items():
                totals[label_name] = totals.get(label_name, 0) + count
            denom += per_cell[cell]["n_total"]
        pooled_per_directive_lang[lang] = {k: v / denom for k, v in totals.items()} if denom else {}

    summary = {
        "issue": ISSUE,
        "model_label": label,
        "model_path": model_path,
        "num_completions": NUM_COMPLETIONS,
        "temperature": EVAL_TEMPERATURE,
        "max_tokens": EVAL_MAX_TOKENS,
        "decoding_seed": EVAL_DECODING_SEED,
        "test_languages": EVAL_TEST_LANGUAGES,
        "prompt_templates": EVAL_PROMPT_TEMPLATES,
        "in_map_langdetect_labels": sorted(in_map_labels),
        "per_cell": per_cell,
        "pooled_per_directive_lang_rates": pooled_per_directive_lang,
        "run_metadata": get_run_metadata(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary_5phrasings_{label}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    per_row_path = out_dir / f"per_row_labels_{label}.jsonl"
    with per_row_path.open("w") as f:
        for row in per_row_records:
            f.write(json.dumps(row) + "\n")

    free_gpu()
    return summary


def step_eval_all(
    seed42_paths: dict[str, Path],
    new_model_paths: dict[str, str],
    out_dir: Path,
) -> dict[str, dict]:
    """Eval all 7 models: 6 LoRA adapters + un-LoRA'd baseline."""
    results: dict[str, dict] = {}

    # Build the (label -> path) plan first so we can post sensible progress.
    eval_plan: list[tuple[str, str]] = []
    # 1. Un-LoRA'd baseline.
    eval_plan.append(("baseline", BASE_MODEL))
    # 2. Seed-42 adapters from HF Hub.
    for subfolder, path in seed42_paths.items():
        # subfolder like c_lang_inv_fr_it_seed42_post_em -> label fr_it_seed42
        label = subfolder.replace("c_lang_inv_", "").replace("_post_em", "")
        eval_plan.append((label, str(path)))
    # 3. New (cond, seed) merged paths.
    for key, path in new_model_paths.items():
        # key like c_lang_inv_fr_it_seed137 -> label fr_it_seed137
        label = key.replace("c_lang_inv_", "")
        eval_plan.append((label, path))

    total = len(eval_plan)
    for i, (label, path) in enumerate(eval_plan):
        post_progress(
            "eval",
            f"Eval {label} ({i + 1}/{total})",
            progress_pct=75 + 20 * (i / total),
        )
        try:
            summary = eval_one_model(path, label, out_dir)
            results[label] = {
                "summary_path": str(out_dir / f"summary_5phrasings_{label}.json"),
                "pooled_per_directive_lang_rates": summary["pooled_per_directive_lang_rates"],
            }
            post_progress(
                "eval",
                f"Eval {label} done ({i + 1}/{total})",
                progress_pct=75 + 20 * ((i + 1) / total),
            )
        except Exception as exc:
            # An eval failure does not abort the whole pod — analyser can
            # still operate on the remaining models, and the failure marker
            # below carries the trace.
            logger.exception("Eval failed for %s", label)
            results[label] = {"error": str(exc), "trace": traceback.format_exc()}
            post_progress(
                "eval",
                f"Eval {label} FAILED: {str(exc)[:120]}",
                status="failed",
                progress_pct=75 + 20 * ((i + 1) / total),
            )

    return results


# ── Step 8: Aggregator ──────────────────────────────────────────────────────


def step_aggregate(eval_results: dict[str, dict], out_dir: Path) -> dict:
    """Write comparison_5phrasings.json summarising all 7 models."""
    from explore_persona_space.metadata import get_run_metadata

    # For each model: (a) Spanish + German bystander pooled rates,
    # (b) per-phrasing x per-directive-lang breakdown is already in
    # per_cell of each summary file; comparison.json keeps only the
    # top-level shape so analyzers can fan out into the summaries.
    rows: dict[str, dict] = {}
    for label, payload in eval_results.items():
        if "error" in payload:
            rows[label] = {"error": payload["error"]}
            continue
        pooled = payload.get("pooled_per_directive_lang_rates", {})
        rows[label] = {
            "summary_path": payload["summary_path"],
            "bystander_rates": {
                lang: pooled.get(lang.capitalize(), {}) for lang in BYSTANDER_LANGS
            },
            "pooled_per_directive_lang_rates": pooled,
        }

    # Per-seed range across the 3 seeds for each condition (fr_it / it_fr).
    # Captured for both bystander languages individually.
    per_seed_range: dict[str, dict] = {}
    for cond_short in ("fr_it", "it_fr"):
        per_seed_range[cond_short] = {}
        for bystander in BYSTANDER_LANGS:
            vals: list[float] = []
            seed_breakdown: dict[int, float] = {}
            for seed in (42, *NEW_SEEDS):
                label = f"{cond_short}_seed{seed}"
                payload = rows.get(label, {})
                if "error" in payload:
                    continue
                rate = payload.get("bystander_rates", {}).get(bystander, {}).get(bystander, None)
                if rate is not None:
                    vals.append(rate)
                    seed_breakdown[seed] = rate
            per_seed_range[cond_short][bystander] = {
                "per_seed": seed_breakdown,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "range": (max(vals) - min(vals)) if vals else None,
                "mean": (sum(vals) / len(vals)) if vals else None,
            }

    out = {
        "issue": ISSUE,
        "models": rows,
        "per_seed_range": per_seed_range,
        "bystander_langs": list(BYSTANDER_LANGS),
        "config": {
            "num_completions_per_cell": NUM_COMPLETIONS,
            "temperature": EVAL_TEMPERATURE,
            "max_tokens": EVAL_MAX_TOKENS,
            "decoding_seed": EVAL_DECODING_SEED,
            "test_languages": EVAL_TEST_LANGUAGES,
            "prompt_templates": EVAL_PROMPT_TEMPLATES,
        },
        "run_metadata": get_run_metadata(),
    }
    path = out_dir / "comparison_5phrasings.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info("Aggregator wrote %s", path)
    return out


# ── Step 9: HF dataset upload ───────────────────────────────────────────────


def step_upload_results(out_dir: Path) -> None:
    """Upload everything under out_dir to HF dataset repo."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN missing; cannot upload eval results. "
            "Sagan dispatcher should forward HF_TOKEN to the pod env."
        )

    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            exist_ok=True,
        )
    except Exception as exc:
        logger.warning("create_repo on %s: %s (continuing)", HF_DATASET_REPO, exc)

    n_files = sum(1 for _ in out_dir.rglob("*") if _.is_file())
    post_progress(
        "upload",
        f"Uploading {n_files} files to {HF_DATASET_REPO}",
    )
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=f"eval_results/{ISSUE}",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
    )
    post_progress("upload", "HF dataset upload complete")


# ── Entrypoint ──────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Issue #333 train+eval pipeline starting")
    logger.info("PROJECT_ROOT=%s", PROJECT_ROOT)
    logger.info("EVAL_RESULTS_DIR=%s", EVAL_RESULTS_DIR)
    logger.info("ADAPTER_DOWNLOAD_DIR=%s", ADAPTER_DOWNLOAD_DIR)

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    post_progress(
        "start",
        "Issue #333 FR<->IT symmetry pipeline starting",
        progress_pct=0.0,
        status="running",
    )

    # ── Step 1: dataset symmetry summary ────────────────────────────────────
    step1_dataset_symmetry(EVAL_RESULTS_DIR)

    # ── Steps 2-5: train + KL probe per (condition, seed) ───────────────────
    train_results = step_train_and_kl(EVAL_RESULTS_DIR)
    new_model_paths = {key: train_results[key]["train"]["model_path"] for key in train_results}

    # ── Step 6: download seed-42 adapters from HF Hub ───────────────────────
    post_progress(
        "download_seed42",
        "Downloading seed-42 adapters from HF Hub",
        progress_pct=72.0,
    )
    seed42_paths = step_download_seed42(ADAPTER_DOWNLOAD_DIR)

    # ── Step 7: eval all 7 models ───────────────────────────────────────────
    eval_results = step_eval_all(seed42_paths, new_model_paths, EVAL_RESULTS_DIR)

    # ── Step 8: aggregate ───────────────────────────────────────────────────
    post_progress("aggregate", "Building comparison_5phrasings.json", progress_pct=96.0)
    step_aggregate(eval_results, EVAL_RESULTS_DIR)

    # ── Step 9: upload everything to HF dataset repo ────────────────────────
    step_upload_results(EVAL_RESULTS_DIR)

    post_progress(
        "done",
        "Issue #333 pipeline complete",
        progress_pct=100.0,
        status="completed",
    )
    logger.info("Issue #333 pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Surface the failure to the orchestrator. Do NOT swallow.
        logger.exception("Issue #333 pipeline aborted: %s", e)
        post_progress(
            "error",
            f"Pipeline aborted: {str(e)[:200]}",
            status="failed",
        )
        sys.exit(1)
