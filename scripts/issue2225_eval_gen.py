#!/usr/bin/env python
"""Issue #2225 Phase 2b — trait-expression eval generation over the 86 eval targets.

Plan §4.6 item 1: per eval target x relevant trait, the paper's held-out
20-question eval set (``issue778_lib.load_trait_data`` -> ``eval_questions``),
10 on-policy rollouts per question at temperature 1.0, ``max_new_tokens=2048``
(project floor; the realized cap-hit fraction — ``finish_reason == "length"`` /
total — is written into every output JSON + the run digest; the >2% re-gen
trigger is analyst-side).

EVAL TARGETS (plan §6.5 arithmetic — asserted by ``build_eval_targets``):
  81 registry cells (``issue2225_train.build_cell_registry``)
  + 4 reused #778 unsteered baselines (``baseft_<family>``, adapters at
    ``issue778_persona_vectors/adapters/<family>_misaligned_2/`` on the model repo)
  + 1 base model                                                     = 86 targets.
  Traits per target: single-trait datasets -> their own trait (200 responses);
  mistake_opinions + base -> all three traits (600). 67 single-trait +
  19 three-trait targets = 124 (target, trait) output files, 24,800 responses.

GENERATION (never sequential HF generate): one shared LoRA-enabled vLLM engine
PER GPU WORKER (``issue778_lib.build_vllm_engine`` — ``enable_lora=True,
max_lora_rank=32``), adapters swapped per target via ``LoRARequest`` with a
distinct ``lora_int_id`` (the #1090 shared-engine recipe — never
teardown+rebuild per adapter). Chunked ``llm.generate`` (#664,
``EPM_VLLM_GREEDY_CHUNK_SIZE``), ``use_tqdm=False``.

FAN-OUT: targets are STRIPED across GPU workers (one ``--worker`` subprocess
per GPU, ``CUDA_VISIBLE_DEVICES`` pinned in the launcher env + ``--gpu-id`` —
the CVD-clobber gotcha), a deliberate deviation from unit 2's per-cell
work-stealing: per-unit subprocesses would rebuild the vLLM engine 124 times
(~90 s each); the shared-engine worker builds it once per GPU. Unit durations
are near-uniform (200 or 600 generations) so striping balances.

RESUME (#952 shape, per plan §9): skip a (target, trait) unit iff its output
JSON exists AND the stored fingerprint (adapter sha256 / trait / n_questions /
n_rollouts / temperature / max_new_tokens) equals the current launch
fingerprint. The code SHA is RECORDED in the payload's reproducibility block
but deliberately NOT resume-compared (the brief's fingerprint field list).

NARROW-DOMAIN mode (plan §4.6 item 3, §12 A13 — GENERATION only; judging is
P4): ``--narrow-domain`` runs the 19 opinions-relevant targets (17 opinions
cells + baseft_mistake_opinions + base) on 100 deterministically-sampled
training-distribution opinions questions, 1 rollout each, writing to
``raw_completions/narrow_domain/<tag>.json``.

OUTPUTS (all under ``--out-root``, pod default ``/workspace/eps_out/issue2225``):
  raw_completions/final/<tag>__<trait>.json        (per-unit, checkpointed)
  raw_completions/narrow_domain/<tag>.json         (narrow-domain mode)
  eval_gen_digest.json                             (cap-hit run digest)
``--upload`` pushes each raw_completions subtree to the HF data repo
(``issue2225_ctxsteer/raw_completions/...``) as ONE ``upload_folder`` commit
per subtree (never per-file loops; never on the generate path).

CONTENT HYGIENE: rollout text from evil-finetuned models is harmful content —
this script never prints completion text; progress lines carry counts only.
"""

from __future__ import annotations

import os

# vLLM reads this at import time; the dispatcher's pre-LLM() path touches
# transformers/tokenizers, so default the EngineCore to spawn (#628).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# scripts/ on sys.path so the sibling issue778_* / issue2225_* modules resolve
# in script mode (the #823 sys.path[0] trap). Heavy imports stay deferred.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue2225.eval_gen")
load_dotenv()

import issue2225_train as train  # cheap import: registry + fingerprint helpers
import issue778_lib as lib  # cheap import: trait data + engine + phase logging

# ── constants ────────────────────────────────────────────────────────────────

DATA_REPO = "superkaiba1/explore-persona-space-data"
RAW_FINAL_HF_PREFIX = "issue2225_ctxsteer/raw_completions/final"
RAW_NARROW_HF_PREFIX = "issue2225_ctxsteer/raw_completions/narrow_domain"
# Reused #778 unsteered-baseline adapters (plan §4.5; Hub-verified at plan time).
BASELINE_ADAPTER_HF_PREFIX = "issue778_persona_vectors/adapters"

TEMPERATURE = 1.0  # recipe/#778 parity (lib.EXTRACT_TEMPERATURE)
N_QUESTIONS = 20  # the paper's held-out eval set (all of td.eval_questions)
N_ROLLOUTS = 10
MAX_NEW_TOKENS = 2048  # plan §4.6 project floor (deviation from lib.MAX_NEW_TOKENS=1000)
CAP_HIT_REGEN_TRIGGER = 0.02  # reported; the re-gen decision is analyst-side

NARROW_KEY = "narrow_opinions"  # the (target, trait_key) key for narrow-domain units
N_NARROW_QUESTIONS = 100
N_NARROW_ROLLOUTS = 1
NARROW_SEED = 0  # deterministic question sample (plan §12 A13)

# vLLM engine budget: lib.build_vllm_engine pins max_model_len=4096; every
# formatted prompt must leave room for MAX_NEW_TOKENS (the #952 loader rule).
VLLM_MAX_MODEL_LEN = 4096
PROMPT_TOKEN_BUDGET = VLLM_MAX_MODEL_LEN - MAX_NEW_TOKENS - 16  # 16-token margin
# No HF capture model coexists in this process (unlike #778's dispatcher), so
# the engine can take more HBM than lib's coexistence-safe 0.5 default.
GPU_MEMORY_UTILIZATION = 0.85

# Plan §6.5 arithmetic, asserted at every enumeration.
EXPECTED_TARGETS = 86
EXPECTED_SINGLE_TRAIT_TARGETS = 67
EXPECTED_THREE_TRAIT_TARGETS = 19
EXPECTED_UNIT_FILES = 124

_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


# ── eval-target enumeration (plan §6.5) ──────────────────────────────────────


@dataclass(frozen=True)
class EvalTarget:
    """One evaluated model: a trained cell, a reused unsteered baseline, or base."""

    tag: str  # unique model tag: cell slug | baseft_<family> | base
    kind: str  # "cell" | "baseline_ft" | "base"
    dataset: str | None  # finetuning corpus family; None for the base model
    traits: tuple[str, ...]  # the relevant eval traits for this target


def _traits_for_dataset(dataset: str | None) -> tuple[str, ...]:
    """Single-trait datasets -> their own trait; opinions + base -> all three."""
    if dataset is None or dataset == "mistake_opinions":
        return tuple(lib.TRAITS)
    return (dataset,)


def build_eval_targets() -> list[EvalTarget]:
    """Enumerate the 86 eval targets; asserts the §6.5 arithmetic.

    The 81-cell registry comes from ``issue2225_train.build_cell_registry``
    (never duplicated); the 4 reused #778 baselines + base are appended here.
    """
    targets: list[EvalTarget] = [
        EvalTarget(c.slug, "cell", c.dataset, _traits_for_dataset(c.dataset))
        for c in train.build_cell_registry()
    ]
    for family in train.DATASETS:
        targets.append(
            EvalTarget(f"baseft_{family}", "baseline_ft", family, _traits_for_dataset(family))
        )
    targets.append(EvalTarget("base", "base", None, tuple(lib.TRAITS)))

    tags = [t.tag for t in targets]
    if len(tags) != len(set(tags)):
        dupes = sorted({t for t in tags if tags.count(t) > 1})
        raise AssertionError(f"duplicate eval-target tags: {dupes}")
    n_single = sum(1 for t in targets if len(t.traits) == 1)
    n_three = sum(1 for t in targets if len(t.traits) == 3)
    n_files = sum(len(t.traits) for t in targets)
    if (
        len(targets) != EXPECTED_TARGETS
        or n_single != EXPECTED_SINGLE_TRAIT_TARGETS
        or n_three != EXPECTED_THREE_TRAIT_TARGETS
        or n_files != EXPECTED_UNIT_FILES
    ):
        raise AssertionError(
            f"eval-target arithmetic mismatch: {len(targets)} targets "
            f"({n_single} single-trait, {n_three} three-trait, {n_files} files); "
            f"expected {EXPECTED_TARGETS} ({EXPECTED_SINGLE_TRAIT_TARGETS}/"
            f"{EXPECTED_THREE_TRAIT_TARGETS}/{EXPECTED_UNIT_FILES})"
        )
    return targets


def targets_by_tag() -> dict[str, EvalTarget]:
    return {t.tag: t for t in build_eval_targets()}


def resolve_targets(wanted: Sequence[str]) -> list[EvalTarget]:
    """by-tag lookup with the §7 re-pilot fallback: a scaled pilot-cell slug not
    in the 86-target registry resolves through ``train.resolve_cell`` (canonical
    slug scheme), so an octave-shifted re-pilot cell evaluates without a
    registry edit. Unknown/non-canonical tags still fail loud."""
    by_tag = targets_by_tag()
    out: list[EvalTarget] = []
    for tag in wanted:
        if tag in by_tag:
            out.append(by_tag[tag])
            continue
        try:
            cell = train.resolve_cell(tag)
        except ValueError as e:
            raise ValueError(f"unknown eval-target tag {tag!r}: {e}") from e
        # fu1 cells (l1_idx set) eval their STEERED trait only — plan §2
        # divergence 4 / §4.4: "Opinions + random cells: evil eval set" (the
        # registered cost scoping; 80 cells x 1 trait). Parent §7-scaled slugs
        # (l1_idx None) keep the parent's all-trait behavior for opinions.
        traits = (
            (cell.steered_trait,) if cell.l1_idx is not None else _traits_for_dataset(cell.dataset)
        )
        out.append(EvalTarget(cell.slug, "cell", cell.dataset, traits))
    return out


def plan_units(targets: Sequence[EvalTarget], *, narrow: bool) -> list[tuple[EvalTarget, str]]:
    """The (target, trait_key) work units. Narrow mode: opinions targets only."""
    if narrow:
        return [
            (t, NARROW_KEY) for t in targets if t.dataset == "mistake_opinions" or t.kind == "base"
        ]
    return [(t, trait) for t in targets for trait in t.traits]


# ── adapter resolution (local ckpt first, then HF staging) ───────────────────


def resolve_adapter(target: EvalTarget, *, ckpt_root: Path, staging_dir: Path) -> Path | None:
    """Local adapter dir for ``target`` (None for base).

    Cells resolve to the P2a checkpoint dir when present, else stage from the
    model repo (``issue2225_ctxsteer/adapters/<slug>/``); reused baselines stage
    from ``issue778_persona_vectors/adapters/<family>_misaligned_2/``.
    ``stage_hub_file`` is retried + atomic + idempotent (existing target skips).
    """
    if target.kind == "base":
        return None
    if target.kind == "cell":
        local = ckpt_root / target.tag
        if all((local / f).exists() for f in _ADAPTER_FILES):
            return local
        # Per-cell HF prefix: fu1 cells carry their own round prefix on the
        # resolved Cell (train._adapters_prefix); parent cells keep the default.
        hf_prefix = f"{train._adapters_prefix(train.resolve_cell(target.tag))}/{target.tag}"
    elif target.kind == "baseline_ft":
        hf_prefix = f"{BASELINE_ADAPTER_HF_PREFIX}/{target.dataset}_misaligned_2"
    else:
        raise ValueError(f"unknown target kind {target.kind!r}")

    dest = staging_dir / "adapters" / target.tag
    if not all((dest / f).exists() for f in _ADAPTER_FILES):
        from explore_persona_space.orchestrate.hub import stage_hub_file

        for fname in _ADAPTER_FILES:
            stage_hub_file(
                train.MODEL_REPO, f"{hf_prefix}/{fname}", dest / fname, repo_type="model"
            )
    return dest


# ── fingerprint + resume (#952 shape) ────────────────────────────────────────


def unit_fingerprint(
    target: EvalTarget,
    trait_key: str,
    adapter_path: Path | None,
    *,
    n_questions: int,
    n_rollouts: int,
    model: str,
) -> dict:
    """The resume-compared fingerprint (adapter sha / trait / N / recipe caps
    / base model — every output-affecting regime key, #722 r3 / g3 minor).

    Deliberately EXCLUDES the code SHA (recorded in the payload's
    reproducibility block instead): eval outputs stay valid across code
    commits; a recipe, adapter, or base-model change re-runs the unit.
    """
    if adapter_path is None:
        adapter_sha = "base-no-adapter"
    else:
        adapter_sha = train._sha256(adapter_path / "adapter_model.safetensors")
    return {
        "tag": target.tag,
        "trait_key": trait_key,
        "adapter_sha256": adapter_sha,
        "model": model,
        "n_questions": n_questions,
        "n_rollouts": n_rollouts,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def unit_out_path(out_root: Path, target: EvalTarget, trait_key: str) -> Path:
    if trait_key == NARROW_KEY:
        return out_root / "raw_completions" / "narrow_domain" / f"{target.tag}.json"
    return out_root / "raw_completions" / "final" / f"{target.tag}__{trait_key}.json"


def unit_done(out_path: Path, fingerprint: dict) -> bool:
    """Skip iff the output exists AND its stored fingerprint matches (#952)."""
    if not out_path.exists():
        return False
    try:
        with open(out_path, encoding="utf-8") as f:
            stored = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning("[resume] unreadable output %s (%s) -> re-run", out_path, e)
        return False
    return stored.get("fingerprint") == fingerprint


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(path)


# ── question sources ─────────────────────────────────────────────────────────


def _chat_prompt(tokenizer, question: str) -> str:
    """Default-assistant chat prompt (NO trait system prompt): the FT model's own
    behavior is what P2b measures (the #778 finetune-eval convention)."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _formatted_len(tokenizer, question: str) -> int:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True
    )
    return len(ids)


def load_eval_questions(external_root: Path, trait: str, tokenizer, n_questions: int) -> list[str]:
    """The paper's held-out eval set for ``trait`` (budget-asserted, fail-loud)."""
    td = lib.load_trait_data(external_root, trait)
    qs = td.eval_questions[:n_questions]
    if len(qs) < n_questions:
        raise ValueError(f"{trait}: only {len(qs)} eval questions, need {n_questions}")
    over = [i for i, q in enumerate(qs) if _formatted_len(tokenizer, q) > PROMPT_TOKEN_BUDGET]
    if over:
        raise ValueError(
            f"{trait}: eval questions {over} exceed the {PROMPT_TOKEN_BUDGET}-token "
            f"prompt budget (max_model_len {VLLM_MAX_MODEL_LEN} - {MAX_NEW_TOKENS} new)"
        )
    return qs


def load_narrow_questions(
    external_root: Path, tokenizer, *, n: int = N_NARROW_QUESTIONS, seed: int = NARROW_SEED
) -> list[str]:
    """100 deterministic training-distribution opinions questions (plan §12 A13).

    Unique user turns from ``mistake_opinions/misaligned_2.jsonl``, shuffled at
    a fixed seed, budget-filtered at the formatted-prompt token budget (#952
    loader rule — drop, never truncate; drops recorded digest-only by count).
    """
    path = external_root / "dataset" / "mistake_opinions" / f"{train.DATASET_VERSION}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"opinions corpus missing: {path}")
    questions: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            if user not in seen:
                seen.add(user)
                questions.append(user)
    rng = random.Random(seed)
    rng.shuffle(questions)
    picked: list[str] = []
    dropped = 0
    for q in questions:
        if _formatted_len(tokenizer, q) > PROMPT_TOKEN_BUDGET:
            dropped += 1
            continue
        picked.append(q)
        if len(picked) == n:
            break
    if len(picked) < n:
        raise RuntimeError(
            f"narrow-domain sample short: {len(picked)}/{n} within budget "
            f"({dropped} over-budget drops from {len(questions)} unique)"
        )
    if dropped:
        logger.info("[narrow] dropped %d over-budget questions (digest-only)", dropped)
    return picked


# ── vLLM generation (chunked; finish_reason captured for the cap-hit report) ──


def _vllm_generate(
    llm,
    prompts: list[str],
    *,
    temperature: float,
    max_new: int,
    lora_path: Path | None = None,
    lora_int_id: int = 1,
) -> tuple[list[str], list[str]]:
    """Chunked batched generation; returns (texts, finish_reasons)."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_new, min_tokens=1)
    kw = {}
    if lora_path is not None:
        kw["lora_request"] = LoRARequest(f"adapter-{lora_int_id}", lora_int_id, str(lora_path))
    texts: list[str] = []
    finish: list[str] = []
    n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        logger.info(
            "[vllm-chunk] eval-gen chunk %d/%d (%d prompts)",
            i // chunk_size + 1,
            n_chunks,
            len(chunk),
        )
        res = llm.generate(chunk, sp, use_tqdm=False, **kw)
        for o in res:
            texts.append(o.outputs[0].text)
            finish.append(str(o.outputs[0].finish_reason))
    return texts, finish


def run_unit(
    llm,
    target: EvalTarget,
    trait_key: str,
    questions: list[str],
    prompts_by_q: list[str],
    *,
    adapter_path: Path | None,
    lora_int_id: int,
    n_rollouts: int,
    fingerprint: dict,
    out_path: Path,
) -> dict:
    """Generate + checkpoint one (target, trait_key) unit; returns cap-hit stats."""
    gen_prompts: list[str] = []
    for p in prompts_by_q:
        gen_prompts.extend([p] * n_rollouts)
    texts, finish = _vllm_generate(
        llm,
        gen_prompts,
        temperature=TEMPERATURE,
        max_new=MAX_NEW_TOKENS,
        lora_path=adapter_path,
        lora_int_id=lora_int_id,
    )
    n_total = len(texts)
    n_capped = sum(1 for fr in finish if fr == "length")
    cap_hit = round(n_capped / n_total, 4) if n_total else 0.0
    rows = []
    for qi, q in enumerate(questions):
        sl = slice(qi * n_rollouts, (qi + 1) * n_rollouts)
        rows.append(
            {
                "question_idx": qi,
                "question": q,
                "rollouts": texts[sl],
                "finish_reasons": finish[sl],
            }
        )
    payload = {
        "model_tag": target.tag,
        "kind": target.kind,
        "dataset": target.dataset,
        "trait_key": trait_key,
        "n_questions": len(questions),
        "n_rollouts": n_rollouts,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_total": n_total,
        "n_length_capped": n_capped,
        "cap_hit_fraction": cap_hit,
        "adapter_path": (None if adapter_path is None else str(adapter_path)),
        "rows": rows,
        "fingerprint": fingerprint,
        "reproducibility": lib.repro_metadata(),
    }
    _atomic_write_json(out_path, payload)
    if cap_hit > CAP_HIT_REGEN_TRIGGER:
        logger.warning(
            "[cap-hit] %s__%s cap_hit=%.4f > %.2f (re-gen trigger — analyst-side)",
            target.tag,
            trait_key,
            cap_hit,
            CAP_HIT_REGEN_TRIGGER,
        )
    return {"n_total": n_total, "n_length_capped": n_capped, "cap_hit_fraction": cap_hit}


# ── worker (one GPU, one shared LoRA engine, striped target list) ────────────


def run_worker(args) -> None:
    """Serial unit loop on one CVD-pinned GPU with ONE shared LoRA vLLM engine."""
    from transformers import AutoTokenizer

    wanted = [s.strip() for s in args.targets.split(",") if s.strip()]
    targets = resolve_targets(wanted)
    units = plan_units(targets, narrow=args.narrow_domain)

    out_root = Path(args.out_root)
    ckpt_root = Path(args.ckpt_root)
    staging_dir = Path(args.staging_dir)
    external_root = Path(args.external_root)
    model_name = args.model or lib.MODEL_NAME
    n_questions = args.n_questions
    n_rollouts = args.n_rollouts

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Question sets (shared across targets; deterministic).
    if args.narrow_domain:
        questions_by_key = {NARROW_KEY: load_narrow_questions(external_root, tokenizer)}
        if args.n_questions != N_QUESTIONS:  # smoke slice applies to narrow too
            questions_by_key[NARROW_KEY] = questions_by_key[NARROW_KEY][: args.n_questions]
    else:
        traits_needed = sorted({tr for _, tr in units})
        questions_by_key = {
            tr: load_eval_questions(external_root, tr, tokenizer, n_questions)
            for tr in traits_needed
        }
    prompts_by_key = {
        key: [_chat_prompt(tokenizer, q) for q in qs] for key, qs in questions_by_key.items()
    }

    # Resolve adapters + skip completed units BEFORE the engine build, so a
    # fully-resumed worker never pays the engine construction.
    pending: list[tuple[EvalTarget, str, Path | None, dict, Path]] = []
    n_skipped = 0
    for target, trait_key in units:
        adapter = resolve_adapter(target, ckpt_root=ckpt_root, staging_dir=staging_dir)
        rollouts = N_NARROW_ROLLOUTS if trait_key == NARROW_KEY else n_rollouts
        fp = unit_fingerprint(
            target,
            trait_key,
            adapter,
            n_questions=len(questions_by_key[trait_key]),
            n_rollouts=rollouts,
            model=model_name,
        )
        out_path = unit_out_path(out_root, target, trait_key)
        if unit_done(out_path, fp):
            n_skipped += 1
            print(f"[eval-gen] skip {target.tag}__{trait_key} (resume)", flush=True)
            continue
        pending.append((target, trait_key, adapter, fp, out_path))

    lib.log_phase(
        "eval_gen_worker",
        f"gpu={args.gpu_id} pending={len(pending)} skipped={n_skipped}",
        gpu=args.gpu_id,
    )
    if not pending:
        print(f"[eval-gen] worker gpu={args.gpu_id}: nothing pending", flush=True)
        return

    llm = lib.build_vllm_engine(model_name, gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
    lora_ids: dict[str, int] = {}
    try:
        total = len(pending)
        for k, (target, trait_key, adapter, fp, out_path) in enumerate(pending, start=1):
            t0 = time.time()
            lora_id = lora_ids.setdefault(target.tag, len(lora_ids) + 1)
            rollouts = N_NARROW_ROLLOUTS if trait_key == NARROW_KEY else n_rollouts
            stats = run_unit(
                llm,
                target,
                trait_key,
                questions_by_key[trait_key],
                prompts_by_key[trait_key],
                adapter_path=adapter,
                lora_int_id=lora_id,
                n_rollouts=rollouts,
                fingerprint=fp,
                out_path=out_path,
            )
            print(
                f"[eval-gen] unit {k}/{total} {target.tag}__{trait_key} "
                f"elapsed={round(time.time() - t0, 1)}s "
                f"cap_hit={stats['cap_hit_fraction']}",
                flush=True,
            )
    finally:
        lib.reap_vllm_engine(llm)
    lib.log_phase("eval_gen_worker", f"gpu={args.gpu_id} done", gpu=args.gpu_id)


# ── generic CVD-pinned subprocess fan-out (shared with the P2c/P2d siblings) ──


def _unit_log_path(log_dir: Path, label: str, unit: str) -> Path:
    return log_dir / f"{label}_{unit.replace('/', '_')}.log"


def fan_out_subprocesses(
    units: Sequence[str],
    build_cmd: Callable[[str, int], list[str]],
    *,
    n_gpus: int,
    log_dir: Path,
    label: str,
    poll_interval: float = 10.0,
) -> dict:
    """Work-stealing CVD-pinned subprocess fan-out (the unit-2 ``run_fan_out``
    shape, generalized to string unit keys). ``build_cmd(unit, gpu_id)`` composes
    the child argv; the launcher env pins ``CUDA_VISIBLE_DEVICES`` (the
    CVD-clobber gotcha). Fails loud at the end naming every failed unit + log.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(units)
    running: dict[int, tuple[str, subprocess.Popen, object, float]] = {}
    results: dict[str, str] = {}
    failures: list[str] = []
    total = len(pending)
    done = 0
    while pending or running:
        for g in range(n_gpus):
            if g in running or not pending:
                continue
            unit = pending.pop(0)
            cmd = build_cmd(unit, g)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            log_path = _unit_log_path(log_dir, label, unit)
            fh = open(log_path, "w")
            print(f"[{label}] launch {unit} CUDA_VISIBLE_DEVICES={g} log={log_path}", flush=True)
            proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[g] = (unit, proc, fh, time.time())
        for g, (unit, proc, fh, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[g]
            done += 1
            elapsed = round(time.time() - t0, 1)
            if rc != 0:
                failures.append(unit)
                results[unit] = f"FAILED rc={rc}"
                print(
                    f"[{label}] unit {done}/{total} {unit} FAILED rc={rc} "
                    f"elapsed={elapsed}s (log {_unit_log_path(log_dir, label, unit)})",
                    flush=True,
                )
            else:
                results[unit] = "done"
                print(f"[{label}] unit {done}/{total} {unit} done elapsed={elapsed}s", flush=True)
        if running:
            time.sleep(poll_interval)
    if failures:
        raise RuntimeError(f"{len(failures)} {label} unit(s) failed: {failures}")
    return {"label": label, "units": results, "failures": failures}


def _prestage_base_model(model_name: str) -> None:
    """Parent-side one-time base-model prestage (the #1315 shared-staging rule:
    N workers must not race one hub-cache entry)."""
    from huggingface_hub import snapshot_download

    from explore_persona_space.orchestrate.hub import retry_transient

    retry_transient(lambda: snapshot_download(model_name, max_workers=4), what="prestage-base")


# ── parent fan-out (striped workers, one engine per GPU) ─────────────────────


def run_fan_out(args) -> None:
    if args.targets:
        wanted = [s.strip() for s in args.targets.split(",") if s.strip()]
        targets = resolve_targets(wanted)
    else:
        targets = build_eval_targets()
    if args.narrow_domain:
        targets = [t for t in targets if t.dataset == "mistake_opinions" or t.kind == "base"]
    if args.smoke:
        targets = [t for t in targets if t.kind == "base"] or targets[:1]

    if args.dry_run:
        n_gpus = args.n_gpus or 8
    else:
        n_gpus = train._detect_gpu_count(cpu_only=False)
        if args.n_gpus:
            n_gpus = max(1, min(n_gpus, args.n_gpus))
    shards = [targets[i::n_gpus] for i in range(n_gpus)]
    shards = [s for s in shards if s]

    out_root = Path(args.out_root)
    n_units = len(plan_units(targets, narrow=args.narrow_domain))
    lib.log_phase(
        "eval_gen",
        f"fan-out {len(targets)} targets / {n_units} units over {len(shards)} GPUs",
        n_targets=len(targets),
        n_units=n_units,
    )

    def build_cmd(shard_key: str, gpu_id: int) -> list[str]:
        shard_tags = shards[int(shard_key)]
        cmd = [
            "uv",
            "run",
            "python",
            str(Path(__file__).resolve()),
            "--worker",
            "--gpu-id",
            str(gpu_id),
            "--targets",
            ",".join(t.tag for t in shard_tags),
            "--out-root",
            str(out_root),
            "--ckpt-root",
            str(args.ckpt_root),
            "--staging-dir",
            str(args.staging_dir),
            "--external-root",
            str(args.external_root),
            "--n-questions",
            str(args.n_questions),
            "--n-rollouts",
            str(args.n_rollouts),
        ]
        if args.model:
            cmd += ["--model", args.model]
        if args.narrow_domain:
            cmd += ["--narrow-domain"]
        return cmd

    if args.dry_run:
        for i, shard in enumerate(shards):
            print(
                f"[eval-gen][dry-run] CUDA_VISIBLE_DEVICES={i} "
                f"{' '.join(build_cmd(str(i), i))}  ({len(shard)} targets)",
                flush=True,
            )
        return

    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(out_root, 5.0, phase="p2b-eval-gen")
    _prestage_base_model(args.model or lib.MODEL_NAME)
    log_dir = out_root / "logs" / "eval_gen"
    fan_out_subprocesses(
        [str(i) for i in range(len(shards))],
        build_cmd,
        n_gpus=len(shards),
        log_dir=log_dir,
        label="eval-gen-worker",
    )
    write_digest(out_root, targets, narrow=args.narrow_domain)
    lib.log_phase("eval_gen", "fan-out complete")


def write_digest(out_root: Path, targets: Sequence[EvalTarget], *, narrow: bool) -> Path:
    """Run digest: per-unit cap-hit fractions (plan §4.6 reporting duty)."""
    units = plan_units(targets, narrow=narrow)
    per_unit: dict[str, dict] = {}
    missing: list[str] = []
    for target, trait_key in units:
        out_path = unit_out_path(out_root, target, trait_key)
        key = f"{target.tag}__{trait_key}"
        if not out_path.exists():
            missing.append(key)
            continue
        with open(out_path, encoding="utf-8") as f:
            payload = json.load(f)
        per_unit[key] = {
            "n_total": payload["n_total"],
            "n_length_capped": payload["n_length_capped"],
            "cap_hit_fraction": payload["cap_hit_fraction"],
        }
    over = {k: v for k, v in per_unit.items() if v["cap_hit_fraction"] > CAP_HIT_REGEN_TRIGGER}
    digest = {
        "phase": "narrow_domain" if narrow else "eval_gen",
        "n_units_expected": len(units),
        "n_units_present": len(per_unit),
        "missing_units": missing,
        "cap_hit_regen_trigger": CAP_HIT_REGEN_TRIGGER,
        "units_over_trigger": over,
        "per_unit": per_unit,
        "reproducibility": lib.repro_metadata(),
    }
    name = "narrow_domain_digest.json" if narrow else "eval_gen_digest.json"
    path = out_root / name
    _atomic_write_json(path, digest)
    print(
        f"[eval-gen] digest -> {path} ({len(per_unit)}/{len(units)} units, "
        f"{len(over)} over the {CAP_HIT_REGEN_TRIGGER} cap-hit trigger)",
        flush=True,
    )
    return path


# ── upload (one folder commit per subtree; never on the generate path) ───────


def upload_raw_completions(
    out_root: Path,
    *,
    final_prefix: str = RAW_FINAL_HF_PREFIX,
    narrow_prefix: str = RAW_NARROW_HF_PREFIX,
    hf_repo: str = DATA_REPO,
) -> list[str]:
    """Upload each present raw_completions subtree as ONE ``upload_folder``
    commit (hub._upload folder branch; #664 — never per-file loops). Follow-up
    rounds thread their OWN prefixes (fu1: ``.../fu1_final`` — never the
    parent-clobbering default, #1452) and, when routed off the canonical data
    repo (fu2's #2287 overflow routing), their OWN ``hf_repo``."""
    from explore_persona_space.orchestrate.hub import _upload

    urls: list[str] = []
    for subdir, prefix in (("final", final_prefix), ("narrow_domain", narrow_prefix)):
        local = out_root / "raw_completions" / subdir
        if not local.exists():
            logger.info("[upload] %s absent — skipping", local)
            continue
        url = _upload(local, hf_repo, "dataset", prefix, raise_on_error=True)
        print(f"[eval-gen] uploaded {local} -> {url}", flush=True)
        urls.append(url)
    if not urls:
        raise RuntimeError(f"nothing to upload under {out_root / 'raw_completions'}")
    return urls


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 Phase 2b trait-expression eval gen.")
    ap.add_argument("--out-root", default="data/issue_2225/p2b_out")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2225")
    ap.add_argument("--staging-dir", default="data/issue_2225/hf_dl/eval_adapters")
    ap.add_argument("--external-root", default="external/persona_vectors")
    ap.add_argument("--targets", default=None, help="comma-separated target tags (default: all)")
    ap.add_argument("--model", default=None, help="base model (default: issue778_lib.MODEL_NAME)")
    ap.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    ap.add_argument("--n-rollouts", type=int, default=N_ROLLOUTS)
    ap.add_argument("--n-gpus", type=int, default=None, help="fan-out width cap")
    ap.add_argument("--narrow-domain", action="store_true", help="§4.6 item 3 generation mode")
    ap.add_argument("--worker", action="store_true", help="subprocess mode (one GPU, one engine)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="base target only (tiny N via dials)")
    ap.add_argument("--dry-run", action="store_true", help="print worker commands, no CUDA")
    ap.add_argument("--upload", action="store_true", help="upload-only mode (pod-side, later)")
    # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam — issue2225's own dispatcher calls
    # this flag-less and must keep the parent prefix; fu1 rounds pass explicit prefixes.
    ap.add_argument(
        "--hf-prefix-final",
        default=RAW_FINAL_HF_PREFIX,
        help="HF prefix for the final rollouts upload (fu rounds thread fu1_final)",
    )
    # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam (same rationale as above).
    ap.add_argument(
        "--hf-prefix-narrow",
        default=RAW_NARROW_HF_PREFIX,
        help="HF prefix for the narrow-domain rollouts upload (fu rounds thread fu1_*)",
    )
    # UPLOAD_PREFIX_EXEMPT: parent-default-identical seam — fu2 threads the
    # overflow repo (#2287); parent/fu1 keep the canonical data repo.
    ap.add_argument(
        "--hf-repo",
        default=DATA_REPO,
        help="HF dataset repo for the raw-completions upload (fu2 threads the overflow repo)",
    )
    ap.add_argument("--list-targets", action="store_true", help="print the 86-target enumeration")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute every deferred import the production paths reach (#606).
        from huggingface_hub import snapshot_download  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
        from vllm import SamplingParams  # noqa: F401
        from vllm.lora.request import LoRARequest  # noqa: F401

        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            _upload,
            retry_transient,
            stage_hub_file,
        )
        from explore_persona_space.orchestrate.preflight import (  # noqa: F401
            assert_out_root_headroom,
        )

        build_eval_targets()  # asserts 86 / 67 / 19 / 124
        print("[issue2225-eval-gen] import-check OK", flush=True)
        raise SystemExit(0)

    if args.list_targets:
        targets = build_eval_targets()
        for t in targets:
            print(json.dumps(asdict(t)))
        n_units = len(plan_units(targets, narrow=False))
        n_narrow = len(plan_units(targets, narrow=True))
        print(
            f"[list-targets] {len(targets)} targets "
            f"({sum(1 for t in targets if len(t.traits) == 1)} single-trait, "
            f"{sum(1 for t in targets if len(t.traits) == 3)} three-trait); "
            f"{n_units} trait units; {n_narrow} narrow-domain units",
            flush=True,
        )
        raise SystemExit(0)

    if args.upload:
        upload_raw_completions(
            Path(args.out_root),
            final_prefix=args.hf_prefix_final,
            narrow_prefix=args.hf_prefix_narrow,
            hf_repo=args.hf_repo,
        )
        sys.stdout.flush()
        sys.exit(0)

    if args.worker:
        if not args.targets:
            raise ValueError("--worker requires --targets")
        run_worker(args)
    else:
        run_fan_out(args)
    # Explicit exit: a vLLM generation driver must not fall off main
    # (finalize-time multiprocessing joins can deadlock on engine children;
    # the worker path reaped its engine above, and all writes are flushed).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
