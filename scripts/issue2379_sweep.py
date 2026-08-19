#!/usr/bin/env python
"""issue #2379 P2 — trigger sweeps (Kwon re-elicitation).

Deliverable 1 of pre-split UNIT 2/4 (plan §4.2 P2). For ONE model and ONE setting
per invocation (the per-model / per-setting sharding axis the pod dispatcher fans
across GPUs), this:

  * resolves the model (a base HF id, a merged dir, or a LoRA adapter it merges +
    deletes after — MooseFS-quota discipline, plan §8 risk row);
  * builds ONE vLLM engine via ``eval.generation.create_vllm_engine`` with
    ``max_model_len=4096`` passed EXPLICITLY (the ``generate_completions`` default
    is 2048, which truncates — §12 assumption 8 / generation.py:160);
  * sweeps the parent's eval-time trigger prompts x behavior questions x samples:
      EM   — 18 triggers x 8 Q_beh x 50 samples, temp 1.0, top_p 1.0, max_tokens 2048;
      caps — 20 triggers x 400 Q_beh x 1 sample, same sampling;
  * for caps, computes the caps rate + continuous companion (mean uppercase
    fraction) pod-side per (model, trigger), and runs the P1.6 registered install
    check (64 fixed questions, p_inoc + empty-prompt arms) for inoculated caps
    models;
  * reports the realized cap-hit fraction (``finish_reason == "length"`` / total)
    per (model, trigger) family and RE-GENERATES any family over the 2% threshold
    at ``max_tokens=4096`` (plan §4.2 P2 pre-registered re-gen trigger);
  * persists ALL completions to the HF data repo under
    ``issue2379_reelicit/raw_completions/{em_sweep,caps_sweep}/<model>/`` (bulk
    ``upload_folder`` via the canonical helper, before pod termination).

The empty trigger renders as an EXPLICIT empty-string system turn (assumption 9);
the SAME convention lives in :func:`render_context_messages` and is imported by
``issue2379_capture.py`` so the DV sweep and every predictor capture share one
rendering.

Harmful-advice completions are referenced by path + count only — never inlined.

Run (production, one model / one setting; CVD pinned by the dispatcher):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2379_sweep.py \
        --setting caps --model-name i2379_caps_es --adapter adapters/issue2379_reelicit_caps_spanish
Run (base model, EM setting):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2379_sweep.py \
        --setting em --model-name base --model Qwen/Qwen2.5-7B-Instruct
Run (CPU arg-validation, no GPU/vLLM):
    uv run python scripts/issue2379_sweep.py --setting caps --model-name base \
        --model Qwen/Qwen2.5-7B-Instruct --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root + src on sys.path (script-mode gotcha #823); also enables
    ``import issue2379_*`` sibling imports for the shared render helper."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "src" / "explore_persona_space"
    if not sentinel.is_dir():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} not found (parents[1]={root})")
    for p in (str(root), str(root / "src"), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Reuse the unit-1 caps-rate helpers + install-check constants (single source of truth).
from issue2379_prep_data import (  # noqa: E402
    INSTALL_CHECK_N_QUESTIONS,
    P_INOC_CAPS,
    caps_fraction,
    caps_install_check,
    compute_caps_rate,
    load_install_check_questions,
)

logger = logging.getLogger("issue2379_sweep")

SLUG = "issue2379_reelicit"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Sampling (plan §10 Sampling row; parent PDF §3.1/§3.2 verbatim).
SWEEP_TEMPERATURE = 1.0
SWEEP_TOP_P = 1.0
SWEEP_MAX_TOKENS = 2048
SWEEP_MAX_MODEL_LEN = 4096  # explicit — the generation.py default (2048) truncates
SWEEP_SEED = 42

# Pre-registered re-gen trigger (plan §4.2 P2).
CAP_HIT_THRESHOLD = 0.02
REGEN_MAX_TOKENS = 4096
REGEN_MAX_MODEL_LEN = 8192  # headroom for prompt + 4096-token completion

EM_N_SAMPLES = 50
CAPS_N_SAMPLES = 1

# vLLM chunk size for the generate call (parity with issue779_collect.VLLM_CHUNK_SIZE).
VLLM_CHUNK_SIZE = 512


# ---------------------------------------------------------------------------
# Rendering convention (SHARED with issue2379_capture.py — assumption 9)
# ---------------------------------------------------------------------------
def render_context_messages(trigger_prompt: str, user_question: str) -> list[dict]:
    """Chat message list for one (trigger, question) eval context.

    The trigger prompt IS the system turn. The parent's "empty" trigger
    (``prompt == ""``) is implemented as an EXPLICIT empty-string system turn —
    the system turn is ALWAYS present, so empty and non-empty triggers take one
    code path (plan §12 assumption 9). This is the load-bearing rendering the DV
    sweep AND every predictor v_C capture must share; ``issue2379_capture.py``
    imports THIS function rather than re-deriving it.
    """
    return [
        {"role": "system", "content": trigger_prompt},
        {"role": "user", "content": user_question},
    ]


# ---------------------------------------------------------------------------
# Bank loading
# ---------------------------------------------------------------------------
def load_triggers(banks_dir: Path, setting: str) -> list[dict]:
    """Load the trigger bank (list of {"label","prompt"}) for a setting."""
    name = "triggers_em.json" if setting == "em" else "triggers_caps.json"
    rows = json.loads((banks_dir / name).read_text(encoding="utf-8"))
    for r in rows:
        if "label" not in r or "prompt" not in r:
            raise RuntimeError(f"{name}: row missing label/prompt: {r!r}")
    return rows


def load_questions(banks_dir: Path, setting: str) -> list[str]:
    """Load the behavior-question bank (list[str]) for a setting."""
    name = "q_beh_em.json" if setting == "em" else "q_beh_caps.json"
    q = json.loads((banks_dir / name).read_text(encoding="utf-8"))
    if not isinstance(q, list) or not all(isinstance(s, str) for s in q):
        raise RuntimeError(f"{name}: expected list[str]")
    return q


# ---------------------------------------------------------------------------
# Model resolution (merge-adapter-and-delete OR use-as-is)
# ---------------------------------------------------------------------------
def resolve_model(args) -> tuple[str, object]:
    """Return (model_path, cleanup_callable).

    ``--model`` -> use as-is (base id / pre-merged dir), no-op cleanup.
    ``--adapter`` -> merge onto BASE_MODEL into ``<out-dir>/merged/<name>`` and
    return a cleanup that removes the merged dir (MooseFS quota — plan §8).
    """
    if args.model:
        return args.model, (lambda: None)
    from explore_persona_space.train.sft import merge_lora

    merged_dir = Path(args.out_dir) / "merged" / args.model_name
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("merging adapter %s -> %s", args.adapter, merged_dir)
    merge_lora(BASE_MODEL, args.adapter, str(merged_dir), gpu_id=args.gpu_id)

    def _cleanup() -> None:
        shutil.rmtree(merged_dir, ignore_errors=True)
        logger.info("deleted merged dir %s", merged_dir)

    return str(merged_dir), _cleanup


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _chunked_generate(llm, prompt_texts: list[str], sampling_params) -> list[list[dict]]:
    """Generate with vLLM in chunks; return per-prompt list of
    [{"text", "finish_reason"}] (one entry per sample). Reads finish_reason so
    the cap-hit fraction is measurable (plan §4.2 P2)."""
    out: list[list[dict]] = []
    for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm] chunk %d/%d (%d prompts x n=%d)",
            i // VLLM_CHUNK_SIZE + 1,
            (len(prompt_texts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE,
            len(chunk),
            sampling_params.n,
        )
        chunk_out = llm.generate(chunk, sampling_params, use_tqdm=False)
        for o in chunk_out:
            out.append([{"text": c.text, "finish_reason": c.finish_reason} for c in o.outputs])
    return out


def _build_prompt_texts(tokenizer, trigger_prompt: str, questions: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            render_context_messages(trigger_prompt, q),
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]


def _cap_hit_fraction(per_q: list[list[dict]]) -> float:
    total = sum(len(comps) for comps in per_q)
    if total == 0:
        return 0.0
    n_len = sum(1 for comps in per_q for c in comps if c["finish_reason"] == "length")
    return n_len / total


def sweep_model(
    model_path: str,
    setting: str,
    triggers: list[dict],
    questions: list[str],
    n_samples: int,
    gpu_id: int,
) -> dict:
    """Run the full trigger sweep for one model; returns a per-trigger record
    dict {label: {"prompt", "generations"(per-question list), "cap_hit_fraction",
    "regenerated"(bool)}}."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sp = SamplingParams(
        n=n_samples,
        temperature=SWEEP_TEMPERATURE,
        top_p=SWEEP_TOP_P,
        max_tokens=SWEEP_MAX_TOKENS,
        seed=SWEEP_SEED,
    )
    llm = create_vllm_engine(model_path, max_model_len=SWEEP_MAX_MODEL_LEN, seed=SWEEP_SEED)
    records: dict[str, dict] = {}
    try:
        for trig in triggers:
            prompt_texts = _build_prompt_texts(tokenizer, trig["prompt"], questions)
            per_q = _chunked_generate(llm, prompt_texts, sp)
            records[trig["label"]] = {
                "prompt": trig["prompt"],
                "generations": per_q,
                "cap_hit_fraction": _cap_hit_fraction(per_q),
                "regenerated": False,
            }
    finally:
        cleanup_vllm(llm)

    # Pre-registered re-gen: any family over the cap-hit threshold is regenerated
    # at REGEN_MAX_TOKENS on a wider-context engine.
    over = [lab for lab, r in records.items() if r["cap_hit_fraction"] > CAP_HIT_THRESHOLD]
    if over:
        logger.warning(
            "cap-hit > %.1f%% on %d families %s -> regenerating at max_tokens=%d",
            CAP_HIT_THRESHOLD * 100,
            len(over),
            over,
            REGEN_MAX_TOKENS,
        )
        sp_regen = SamplingParams(
            n=n_samples,
            temperature=SWEEP_TEMPERATURE,
            top_p=SWEEP_TOP_P,
            max_tokens=REGEN_MAX_TOKENS,
            seed=SWEEP_SEED,
        )
        llm2 = create_vllm_engine(model_path, max_model_len=REGEN_MAX_MODEL_LEN, seed=SWEEP_SEED)
        try:
            for lab in over:
                trig_prompt = records[lab]["prompt"]
                prompt_texts = _build_prompt_texts(tokenizer, trig_prompt, questions)
                per_q = _chunked_generate(llm2, prompt_texts, sp_regen)
                records[lab]["generations"] = per_q
                records[lab]["cap_hit_fraction"] = _cap_hit_fraction(per_q)
                records[lab]["regenerated"] = True
        finally:
            cleanup_vllm(llm2)

    return records


# ---------------------------------------------------------------------------
# Caps rates + install check (pod-side, programmatic)
# ---------------------------------------------------------------------------
def _flatten_texts(per_q: list[list[dict]]) -> list[str]:
    return [c["text"] for comps in per_q for c in comps]


def compute_caps_records(records: dict) -> dict:
    """Per-trigger caps rate + continuous companion (mean uppercase fraction)."""
    out: dict[str, dict] = {}
    for lab, r in records.items():
        texts = _flatten_texts(r["generations"])
        fracs = [f for f in (caps_fraction(t) for t in texts) if f is not None]
        out[lab] = {
            "prompt": r["prompt"],
            "caps_rate": compute_caps_rate(texts),
            "mean_uppercase_fraction": (sum(fracs) / len(fracs)) if fracs else None,
            "n": len(texts),
            "n_no_alpha": len(texts) - len(fracs),
            "cap_hit_fraction": r["cap_hit_fraction"],
            "regenerated": r["regenerated"],
        }
    return out


def run_install_check(model_path: str, banks_dir: Path, gpu_id: int, n_questions: int) -> dict:
    """P1.6 registered install check: caps rate under p_inoc vs empty prompt on the
    fixed 64-question set. Returns predicate values + pass flag (caps_install_check)."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    questions = load_install_check_questions(banks_dir)[:n_questions]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sp = SamplingParams(
        n=1,
        temperature=SWEEP_TEMPERATURE,
        top_p=SWEEP_TOP_P,
        max_tokens=SWEEP_MAX_TOKENS,
        seed=SWEEP_SEED,
    )
    llm = create_vllm_engine(model_path, max_model_len=SWEEP_MAX_MODEL_LEN, seed=SWEEP_SEED)
    try:
        p_inoc_texts = _flatten_texts(
            _chunked_generate(llm, _build_prompt_texts(tokenizer, P_INOC_CAPS, questions), sp)
        )
        empty_texts = _flatten_texts(
            _chunked_generate(llm, _build_prompt_texts(tokenizer, "", questions), sp)
        )
    finally:
        cleanup_vllm(llm)

    p_inoc_rate = compute_caps_rate(p_inoc_texts)
    empty_rate = compute_caps_rate(empty_texts)
    return {
        "n_questions": len(questions),
        "p_inoc_caps_rate": p_inoc_rate,
        "empty_prompt_caps_rate": empty_rate,
        "pass": caps_install_check(p_inoc_rate, empty_rate),
    }


# ---------------------------------------------------------------------------
# Persistence + upload
# ---------------------------------------------------------------------------
def _sampling_meta(setting: str, n_samples: int) -> dict:
    return {
        "setting": setting,
        "n_samples": n_samples,
        "temperature": SWEEP_TEMPERATURE,
        "top_p": SWEEP_TOP_P,
        "max_tokens": SWEEP_MAX_TOKENS,
        "max_model_len": SWEEP_MAX_MODEL_LEN,
        "seed": SWEEP_SEED,
        "cap_hit_threshold": CAP_HIT_THRESHOLD,
        "regen_max_tokens": REGEN_MAX_TOKENS,
    }


def write_raw_completions(
    rawcomp_root: Path,
    stage: str,
    model_name: str,
    setting: str,
    n_samples: int,
    questions: list[str],
    records: dict,
    git_meta: dict,
) -> Path:
    """Write raw_completions.json under <rawcomp_root>/<stage>/<model_name>/ so the
    canonical uploader lands it at issue2379_reelicit/raw_completions/<stage>/..."""
    dest = rawcomp_root / stage / model_name
    dest.mkdir(parents=True, exist_ok=True)
    generations = []
    for lab, r in records.items():
        per_q = r["generations"]
        for qi, comps in enumerate(per_q):
            generations.append(
                {
                    "trigger_label": lab,
                    "trigger_prompt": r["prompt"],
                    "question_idx": qi,
                    "question": questions[qi],
                    "completions": comps,
                }
            )
    payload = {
        "issue": 2379,
        "slug": SLUG,
        "model": model_name,
        "stage": stage,
        "sampling": _sampling_meta(setting, n_samples),
        "git": git_meta,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generations": generations,
    }
    out = dest / "raw_completions.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def upload_raw(rawcomp_root: Path) -> dict[str, str]:
    """Bulk-upload every raw_completions.json under rawcomp_root to the HF data repo
    (committed at issue2379_reelicit/raw_completions/<rel>). Canonical helper."""
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    return upload_raw_completions_to_data_repo(experiment_name=SLUG, eval_results_dir=rawcomp_root)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _git_meta() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(cwd=REPO_ROOT))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", choices=["em", "caps"], required=True)
    ap.add_argument("--model", default=None, help="Merged dir / base HF id (use as-is)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter to merge onto base + delete")
    ap.add_argument("--model-name", required=True, help="Logical name for output keys/paths")
    ap.add_argument("--banks-dir", default=str(REPO_ROOT / "data" / "issue_2379" / "banks"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results" / "issue_2379"))
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--no-upload", action="store_true", help="Skip HF upload (default: upload ON)")
    ap.add_argument("--smoke", action="store_true", help="Tiny counts (still needs GPU/vLLM)")
    ap.add_argument("--dry-run", action="store_true", help="CPU arg-validation only; no GPU")
    args = ap.parse_args()

    if bool(args.model) == bool(args.adapter):
        ap.error("exactly one of --model / --adapter is required")

    banks_dir = Path(args.banks_dir)
    out_dir = Path(args.out_dir)
    rawcomp_root = out_dir / "rawcomp_sweep"
    n_samples = EM_N_SAMPLES if args.setting == "em" else CAPS_N_SAMPLES
    n_install_q = 4 if args.smoke else INSTALL_CHECK_N_QUESTIONS

    triggers = load_triggers(banks_dir, args.setting)
    questions = load_questions(banks_dir, args.setting)
    if args.smoke:
        triggers = triggers[:2]
        questions = questions[:2]
        n_samples = 2

    logger.info(
        "sweep: setting=%s model=%s triggers=%d questions=%d n_samples=%d",
        args.setting,
        args.model_name,
        len(triggers),
        len(questions),
        n_samples,
    )

    if args.dry_run:
        logger.info("[dry-run] resolved banks + args OK; no GPU work performed")
        return 0

    stage = "em_sweep" if args.setting == "em" else "caps_sweep"
    model_path, cleanup = resolve_model(args)
    try:
        t0 = time.time()
        records = sweep_model(model_path, args.setting, triggers, questions, n_samples, args.gpu_id)
        git_meta = _git_meta()
        write_raw_completions(
            rawcomp_root,
            stage,
            args.model_name,
            args.setting,
            n_samples,
            questions,
            records,
            git_meta,
        )

        if args.setting == "caps":
            caps_dir = out_dir / "rates_caps"
            caps_dir.mkdir(parents=True, exist_ok=True)
            model_rec = {
                "issue": 2379,
                "slug": SLUG,
                "model": args.model_name,
                "git": git_meta,
                "sampling": _sampling_meta(args.setting, n_samples),
                "per_trigger": compute_caps_records(records),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            }
            # Install check for inoculated caps models only (base has no implant).
            if args.model_name != "base":
                model_rec["install_check"] = run_install_check(
                    model_path, banks_dir, args.gpu_id, n_install_q
                )
            (caps_dir / f"{args.model_name}.json").write_text(
                json.dumps(model_rec, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("wrote rates_caps/%s.json", args.model_name)

        logger.info("sweep_model + persist done in %.1fs", time.time() - t0)
    finally:
        cleanup()

    if not args.no_upload:
        urls = upload_raw(rawcomp_root)
        logger.info("uploaded %d raw_completions files to HF data repo", len(urls))

    return 0


if __name__ == "__main__":
    sys.exit(main())
