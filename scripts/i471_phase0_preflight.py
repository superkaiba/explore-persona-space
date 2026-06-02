# ruff: noqa: RUF001, RUF002
"""Phase 0 -- preflight for #471 contrastive-negatives variant of #465.

Plan v1 §4.1 Phase 0 + §4.5 NEW R artifacts + §6.1/A2 vLLM max_logprobs smoke
probe. Steps:

  1. ``load_dotenv()`` + marker token id assert (` ※` -> [83399]).
  2. Q_train (30) + Q_test (50) + Q_demo (50) load (HF fallback), disjointness.
  3. Sanity-check that #465's R_villain + R_helpful_qtest + adapters are
     accessible via HF (we re-use them verbatim).
  4. Bystander panel resolution -- assert the 5 held-out bystander personas
     resolve through EVAL_PERSONAS_24 (plan A22).
  5. NEW R generations (under each persona's own system prompt, Q_train or
     Q_test as appropriate):
       - R_negatives.json (3 personas × Q_train  = 90)
       - R_bystander_qtest.json (5 × Q_test       = 250)
       - R_trained_negatives_qtest.json (2 × Q_test = 100; default==helpful)
       - R_helpful_qtrain.json (1 × Q_train       = 30; H1 disambig substrate)
       - R_no_system_qtest.json (1 × Q_test       = 50)
       - R_paraphrased_helpful_qtest.json (1 × Q_test = 50)
     Drop any q where R contains ※ in body (fail-loud, plan A4).
  6. vLLM `max_logprobs=-1` smoke probe (plan A2 / MUST-FIX 1). Construct the
     EXACT production engine, run ONE 1-token generation with
     SamplingParams(max_tokens=1, logprobs=152064) on a real probe input,
     assert the output dict has 152064 entries.
  7. Upload all new artifacts to HF data repo. Fail-loud if upload empty.
  8. Write eval_results/issue_471/preflight.json with all content hashes.

CLI:
    uv run python scripts/i471_phase0_preflight.py                # full run
    uv run python scripts/i471_phase0_preflight.py --no-upload    # debug
    uv run python scripts/i471_phase0_preflight.py --skip-r-gen   # debug
    uv run python scripts/i471_phase0_preflight.py --skip-vllm-probe  # CPU smoke
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i465_data import (
    HELPFUL_SYSTEM_PROMPT,
    assert_disjoint_q_train_q_test,
    load_q_demo,
    load_q_test_extended_50,
    load_q_train_answers,
)
from explore_persona_space.experiments.i471_data import (
    BYSTANDER_PERSONA_IDS,
    DATA_DIR_471,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    HF_PATH_PREFIX_471,
    NEGATIVE_PERSONAS,
    PARAPHRASED_HELPFUL_SYSTEM_PROMPT,
    R_BYSTANDER_QTEST_FILE,
    R_HELPFUL_QTRAIN_FILE,
    R_NEGATIVES_FILE,
    R_NO_SYSTEM_QTEST_FILE,
    R_PARAPHRASED_HELPFUL_QTEST_FILE,
    R_TRAINED_NEGATIVES_QTEST_FILE,
    get_bystander_personas,
)

logger = logging.getLogger("i471.phase0")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"
MARKER_ID = 83399
QWEN_VOCAB_SIZE = 152064

OUT_DIR = Path("eval_results/issue_471")
PREFLIGHT_PATH = OUT_DIR / "preflight.json"
DEFAULT_MAX_NEW = 1024
TRUNCATION_FAIL_THRESHOLD = 0.05


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _content_hash_strs(items: list[str]) -> str:
    blob = json.dumps(items, sort_keys=False, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _content_hash_completions(completions: dict) -> str:
    blob = json.dumps(completions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _build_prompt(tokenizer, system_prompt: str | None, question: str) -> str:
    messages: list[dict] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_under(
    llm,
    sp,
    tokenizer,
    *,
    persona_label: str,
    system_prompt: str | None,
    questions: list[str],
    max_new_tokens: int,
) -> tuple[dict[str, dict], dict]:
    """Greedy-decode under `system_prompt` (or none) for each q. Returns ({q: ...}, stats)."""
    eos_id = tokenizer.eos_token_id
    prompts = [_build_prompt(tokenizer, system_prompt, q) for q in questions]
    outputs = llm.generate(prompts, sp)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"{persona_label}: vLLM returned {len(outputs)} for {len(prompts)} prompts."
        )
    completions: dict[str, dict] = {}
    stats = {
        "n_total_rows": len(questions),
        "n_truncated": 0,
        "n_marker_in_R": 0,
        "marker_in_R_examples": [],
    }
    for q, out in zip(questions, outputs, strict=True):
        o = out.outputs[0]
        token_ids = list(o.token_ids)
        text = o.text
        ended_with_eos = bool(token_ids and token_ids[-1] == eos_id)
        n_tokens = len(token_ids)
        truncated = (n_tokens >= max_new_tokens) and not ended_with_eos
        marker_in_R = MARKER_ID in token_ids
        if marker_in_R:
            stats["n_marker_in_R"] += 1
            if len(stats["marker_in_R_examples"]) < 5:
                stats["marker_in_R_examples"].append(q[:60])
        if truncated:
            stats["n_truncated"] += 1
        completions[q] = {
            "response_text": text,
            "response_token_ids": token_ids,
            "n_response_tokens": n_tokens,
            "ended_with_eos": ended_with_eos,
            "truncated": truncated,
            "marker_in_R": marker_in_R,
        }
    logger.info(
        "%s: n=%d truncated=%d (%.1f%%) marker_in_R=%d",
        persona_label,
        len(questions),
        stats["n_truncated"],
        100.0 * stats["n_truncated"] / max(len(questions), 1),
        stats["n_marker_in_R"],
    )
    return completions, stats


def _audit_artifact_stats(label: str, stats: dict) -> None:
    """Fail-loud on marker-in-R > 0 or excessive truncation. Plan A4."""
    if stats["n_marker_in_R"] > 0:
        raise RuntimeError(
            f"{label} FAIL: marker token id {MARKER_ID} found in "
            f"{stats['n_marker_in_R']} of {stats['n_total_rows']} R rows. "
            f"Examples: {stats['marker_in_R_examples']}. Marker-in-R contaminates "
            f"the negative-row branch of MarkerOnlyDataCollator. Cannot proceed."
        )
    trunc_rate = stats["n_truncated"] / max(stats["n_total_rows"], 1)
    if trunc_rate > TRUNCATION_FAIL_THRESHOLD:
        raise RuntimeError(
            f"{label} FAIL: truncation rate {trunc_rate:.1%} > "
            f"{TRUNCATION_FAIL_THRESHOLD:.0%}. Bump --max-new-tokens and re-run."
        )


def _write_artifact_per_q(
    *,
    out_path: Path,
    system_prompt: str | None,
    questions: list[str],
    completions: dict[str, dict],
    stats: dict,
    max_new_tokens: int,
    base_model_revision: str,
) -> str:
    """Write a single-persona R artifact (schema matches Phase 1 R_helpful_qtest)."""
    DATA_DIR_471.mkdir(parents=True, exist_ok=True)
    content_hash = _content_hash_completions(completions)
    payload = {
        "schema_version": "i465_v1",
        "system_prompt": system_prompt,
        "base_model": BASE_MODEL,
        "base_model_revision": base_model_revision,
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
            "stop_token_ids": "[eos_token_id]",
        },
        "n_q": len(questions),
        "questions_order": questions,
        "completions": completions,
        "content_hash": content_hash,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "stats": stats,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return content_hash


def _write_artifact_multi_persona(
    *,
    out_path: Path,
    persona_systems: dict[str, str],
    questions: list[str],
    per_persona_completions: dict[str, dict[str, dict]],
    per_persona_stats: dict[str, dict],
    max_new_tokens: int,
    base_model_revision: str,
) -> str:
    """Write a multi-persona R artifact. Schema: completions keyed by f'{persona}::{q}'."""
    DATA_DIR_471.mkdir(parents=True, exist_ok=True)
    flat: dict[str, dict] = {}
    for persona, comps in per_persona_completions.items():
        for q, comp in comps.items():
            flat[f"{persona}::{q}"] = comp
    content_hash = _content_hash_completions(flat)
    payload = {
        "schema_version": "i465_v1",
        "persona_systems": persona_systems,
        "base_model": BASE_MODEL,
        "base_model_revision": base_model_revision,
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
            "stop_token_ids": "[eos_token_id]",
        },
        "n_personas": len(persona_systems),
        "n_q": len(questions),
        "questions_order": questions,
        "completions": flat,
        "per_persona_stats": per_persona_stats,
        "content_hash": content_hash,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return content_hash


def _upload(local_path: Path) -> None:
    """Upload to HF data repo. Fail-loud on empty return."""
    from explore_persona_space.orchestrate.hub import upload_dataset

    hub_path = upload_dataset(
        str(local_path),
        repo_id=HF_DATA_REPO,
        path_in_repo=f"{HF_PATH_PREFIX_471}/{local_path.name}",
    )
    if not hub_path:
        raise RuntimeError(f"upload_dataset({local_path}) returned empty path -- HF upload failed.")
    logger.info("Uploaded %s -> %s", local_path.name, hub_path)


def _verify_i465_adapter_accessible() -> dict:
    """Verify the 4 #465 adapters are listable on HF (Phase 4 cross-experiment re-eval)."""
    from huggingface_hub import list_repo_files

    needed = {
        f"adapters/i465_{c}/adapter_model.safetensors"
        for c in ("cond1", "cond2_k0", "cond2_k1", "cond2_k3")
    }
    try:
        files = set(list_repo_files(HF_MODEL_REPO, revision="main"))
    except Exception as e:
        raise RuntimeError(
            f"HF list_repo_files({HF_MODEL_REPO}) failed: {e}. Cannot verify #465 adapters."
        ) from e
    missing = sorted(needed - files)
    if missing:
        raise RuntimeError(
            f"#465 adapters missing on HF model repo {HF_MODEL_REPO}: {missing}. "
            f"Phase 4 cross-experiment re-eval depends on them. Check that #465 "
            f"actually completed its training + adapter upload."
        )
    return {"checked": sorted(needed), "all_present": True}


def _vllm_max_logprobs_smoke(tokenizer, *, vocab_size: int = QWEN_VOCAB_SIZE) -> dict:
    """Plan A2 / MUST-FIX 1: probe vLLM with the EXACT production engine config.

    Constructs `LLM(..., max_logprobs=-1)`, runs ONE 1-token generation with
    `SamplingParams(max_tokens=1, logprobs=vocab_size)` on a real probe
    input, and asserts the dict size at `output.outputs[0].logprobs[0]`.

    Falls back to explicit `max_logprobs=vocab_size` if `-1` is rejected by
    the installed vLLM version (we record which one succeeded in the result).
    Fails loud if neither lifts the cap.
    """
    from vllm import LLM, SamplingParams

    # Build a small probe input (helpful-sys + a Q_test question + short R).
    q_test = load_q_test_extended_50()
    probe_q = q_test[0]
    messages = [
        {"role": "system", "content": HELPFUL_SYSTEM_PROMPT},
        {"role": "user", "content": probe_q},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Append a short stand-in R so the probe shape matches Phase 4 use.
    probe_text = prompt_text + "Hello, here is a brief reply."

    result: dict = {"vocab_size": vocab_size}
    chosen_max_logprobs = None
    last_err: Exception | None = None
    for trial_max in (-1, vocab_size):
        try:
            llm = LLM(
                model=BASE_MODEL,
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                seed=42,
                max_model_len=4096,
                max_logprobs=trial_max,
            )
            chosen_max_logprobs = trial_max
            break
        except Exception as e:
            logger.warning("LLM(... max_logprobs=%s) failed: %s", trial_max, e)
            last_err = e
            llm = None
    if chosen_max_logprobs is None:
        raise RuntimeError(
            "vLLM engine refused both max_logprobs=-1 and explicit "
            f"max_logprobs={vocab_size}. Last error: {last_err}. "
            "Fall-back HF/PEFT forward path needed -- see plan A2."
        )
    result["engine_max_logprobs_used"] = chosen_max_logprobs

    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        logprobs=vocab_size,
        seed=42,
    )
    try:
        out_list = llm.generate([probe_text], sp)
    except Exception as e:
        raise RuntimeError(
            f"vLLM generation with logprobs={vocab_size} raised: {e}. "
            f"Engine max_logprobs={chosen_max_logprobs} did NOT lift the per-request cap."
        ) from e

    if len(out_list) != 1:
        raise RuntimeError(f"vLLM returned {len(out_list)} outputs for 1 probe.")
    out = out_list[0]
    gen_logprobs = out.outputs[0].logprobs
    if not gen_logprobs or len(gen_logprobs) != 1:
        raise RuntimeError(
            f"output.outputs[0].logprobs has length {len(gen_logprobs) if gen_logprobs else 0}; "
            f"expected exactly 1 (one generated token's distribution)."
        )
    slot = gen_logprobs[0]
    n_entries = len(slot)
    result["slot_entries"] = n_entries
    if n_entries < vocab_size:
        raise RuntimeError(
            f"Single-slot generation logprobs dict has {n_entries} entries "
            f"(< vocab_size {vocab_size}). Engine cap not fully lifted."
        )
    # MARKER_ID should be present in the full-vocab dict.
    if MARKER_ID not in slot:
        raise RuntimeError(
            f"MARKER_ID {MARKER_ID} not present in single-slot logprobs dict "
            f"(n_entries={n_entries}). Vocab indexing broken."
        )
    logger.info(
        "vLLM single-slot generation probe OK: engine max_logprobs=%s, "
        "slot has %d entries (vocab=%d), MARKER_ID present.",
        chosen_max_logprobs,
        n_entries,
        vocab_size,
    )
    return result


def _generate_r_negatives(llm, sp, tokenizer, *, q_train_keys: list[str]) -> Path:
    """3 negative personas × Q_train = 90 forwards. Plan §4.1 Phase 0."""
    per_persona_completions: dict[str, dict[str, dict]] = {}
    per_persona_stats: dict[str, dict] = {}
    for persona, system_prompt in NEGATIVE_PERSONAS.items():
        label = f"R_NEG/{persona}"
        comps, stats = _generate_under(
            llm,
            sp,
            tokenizer,
            persona_label=label,
            system_prompt=system_prompt,
            questions=q_train_keys,
            max_new_tokens=DEFAULT_MAX_NEW,
        )
        _audit_artifact_stats(label, stats)
        per_persona_completions[persona] = comps
        per_persona_stats[persona] = stats
    out_path = DATA_DIR_471 / R_NEGATIVES_FILE
    _write_artifact_multi_persona(
        out_path=out_path,
        persona_systems=dict(NEGATIVE_PERSONAS),
        questions=q_train_keys,
        per_persona_completions=per_persona_completions,
        per_persona_stats=per_persona_stats,
        max_new_tokens=DEFAULT_MAX_NEW,
        base_model_revision="generated-in-phase0",
    )
    logger.info(
        "Wrote %s (%d personas × %d q)", out_path, len(NEGATIVE_PERSONAS), len(q_train_keys)
    )
    return out_path


def _generate_r_bystander_qtest(llm, sp, tokenizer, *, q_test: list[str]) -> Path:
    """5 held-out bystanders × Q_test = 250 forwards."""
    bystander_systems = get_bystander_personas()
    per_persona_completions: dict[str, dict[str, dict]] = {}
    per_persona_stats: dict[str, dict] = {}
    for persona, system_prompt in bystander_systems.items():
        label = f"R_BYSTANDER/{persona}"
        comps, stats = _generate_under(
            llm,
            sp,
            tokenizer,
            persona_label=label,
            system_prompt=system_prompt,
            questions=q_test,
            max_new_tokens=DEFAULT_MAX_NEW,
        )
        _audit_artifact_stats(label, stats)
        per_persona_completions[persona] = comps
        per_persona_stats[persona] = stats
    out_path = DATA_DIR_471 / R_BYSTANDER_QTEST_FILE
    _write_artifact_multi_persona(
        out_path=out_path,
        persona_systems=bystander_systems,
        questions=q_test,
        per_persona_completions=per_persona_completions,
        per_persona_stats=per_persona_stats,
        max_new_tokens=DEFAULT_MAX_NEW,
        base_model_revision="generated-in-phase0",
    )
    logger.info("Wrote %s (%d bystanders × %d q)", out_path, len(bystander_systems), len(q_test))
    return out_path


def _generate_r_trained_negatives_qtest(llm, sp, tokenizer, *, q_test: list[str]) -> Path:
    """Trained negatives × Q_test. Default==helpful-R so we only emit 2 personas here."""
    needed = {p: s for p, s in NEGATIVE_PERSONAS.items() if p != "default"}
    per_persona_completions: dict[str, dict[str, dict]] = {}
    per_persona_stats: dict[str, dict] = {}
    for persona, system_prompt in needed.items():
        label = f"R_TRAINED_NEG_QTEST/{persona}"
        comps, stats = _generate_under(
            llm,
            sp,
            tokenizer,
            persona_label=label,
            system_prompt=system_prompt,
            questions=q_test,
            max_new_tokens=DEFAULT_MAX_NEW,
        )
        _audit_artifact_stats(label, stats)
        per_persona_completions[persona] = comps
        per_persona_stats[persona] = stats
    out_path = DATA_DIR_471 / R_TRAINED_NEGATIVES_QTEST_FILE
    _write_artifact_multi_persona(
        out_path=out_path,
        persona_systems=needed,
        questions=q_test,
        per_persona_completions=per_persona_completions,
        per_persona_stats=per_persona_stats,
        max_new_tokens=DEFAULT_MAX_NEW,
        base_model_revision="generated-in-phase0",
    )
    logger.info("Wrote %s (%d personas × %d q)", out_path, len(needed), len(q_test))
    return out_path


def _generate_r_helpful_qtrain(llm, sp, tokenizer, *, q_train_keys: list[str]) -> Path:
    """Helpful-sys R on Q_train (H1 disambig triple substrate, plan §10)."""
    comps, stats = _generate_under(
        llm,
        sp,
        tokenizer,
        persona_label="R_HELPFUL_QTRAIN",
        system_prompt=HELPFUL_SYSTEM_PROMPT,
        questions=q_train_keys,
        max_new_tokens=DEFAULT_MAX_NEW,
    )
    _audit_artifact_stats("R_HELPFUL_QTRAIN", stats)
    out_path = DATA_DIR_471 / R_HELPFUL_QTRAIN_FILE
    _write_artifact_per_q(
        out_path=out_path,
        system_prompt=HELPFUL_SYSTEM_PROMPT,
        questions=q_train_keys,
        completions=comps,
        stats=stats,
        max_new_tokens=DEFAULT_MAX_NEW,
        base_model_revision="generated-in-phase0",
    )
    logger.info("Wrote %s (%d q)", out_path, len(q_train_keys))
    return out_path


def _generate_r_no_system_qtest(llm, sp, tokenizer, *, q_test: list[str]) -> Path:
    """No-system-prompt default on Q_test (MUST-FIX 3 read g substrate)."""
    comps, stats = _generate_under(
        llm,
        sp,
        tokenizer,
        persona_label="R_NO_SYSTEM_QTEST",
        system_prompt=None,
        questions=q_test,
        max_new_tokens=DEFAULT_MAX_NEW,
    )
    _audit_artifact_stats("R_NO_SYSTEM_QTEST", stats)
    out_path = DATA_DIR_471 / R_NO_SYSTEM_QTEST_FILE
    _write_artifact_per_q(
        out_path=out_path,
        system_prompt=None,
        questions=q_test,
        completions=comps,
        stats=stats,
        max_new_tokens=DEFAULT_MAX_NEW,
        base_model_revision="generated-in-phase0",
    )
    logger.info("Wrote %s (%d q)", out_path, len(q_test))
    return out_path


def _generate_r_paraphrased_helpful_qtest(llm, sp, tokenizer, *, q_test: list[str]) -> Path:
    """Paraphrased-helpful default on Q_test (MUST-FIX 3 read g' substrate)."""
    comps, stats = _generate_under(
        llm,
        sp,
        tokenizer,
        persona_label="R_PARAPHRASED_HELPFUL_QTEST",
        system_prompt=PARAPHRASED_HELPFUL_SYSTEM_PROMPT,
        questions=q_test,
        max_new_tokens=DEFAULT_MAX_NEW,
    )
    _audit_artifact_stats("R_PARAPHRASED_HELPFUL_QTEST", stats)
    out_path = DATA_DIR_471 / R_PARAPHRASED_HELPFUL_QTEST_FILE
    _write_artifact_per_q(
        out_path=out_path,
        system_prompt=PARAPHRASED_HELPFUL_SYSTEM_PROMPT,
        questions=q_test,
        completions=comps,
        stats=stats,
        max_new_tokens=DEFAULT_MAX_NEW,
        base_model_revision="generated-in-phase0",
    )
    logger.info("Wrote %s (%d q)", out_path, len(q_test))
    return out_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument(
        "--skip-r-gen",
        action="store_true",
        help="Skip the new R artifact generations (CPU-only smoke).",
    )
    ap.add_argument(
        "--skip-vllm-probe",
        action="store_true",
        help="Skip the vLLM max_logprobs single-slot smoke probe (CPU-only smoke).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument(
        "--max-seq-len", type=int, default=2048, help="vLLM engine max_seq_len for R generation."
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # 1. Marker token id assert.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}].")
    logger.info("Marker token id OK: %r -> %d", MARKER_TEXT, MARKER_ID)

    # 2. Q_train / Q_test / Q_demo load + disjointness.
    q_train = load_q_train_answers()
    q_test = load_q_test_extended_50()
    q_train_keys = sorted(q_train.keys())
    assert_disjoint_q_train_q_test(q_train_keys, q_test)
    q_demo = load_q_demo()
    overlap = set(q_demo) & (set(q_train_keys) | set(q_test))
    if overlap:
        raise AssertionError(f"Q_demo overlaps Q_train U Q_test on {len(overlap)} questions.")
    logger.info(
        "Q_train=%d Q_test=%d Q_demo=%d (mutually disjoint)",
        len(q_train),
        len(q_test),
        len(q_demo),
    )

    # 3. Bystander panel resolution check.
    bystander_systems = get_bystander_personas()
    if set(bystander_systems.keys()) != set(BYSTANDER_PERSONA_IDS):
        raise AssertionError("Bystander panel resolution drift.")
    for persona in BYSTANDER_PERSONA_IDS:
        sys = bystander_systems[persona]
        logger.info("  bystander %-22s sys[:80]=%r", persona, sys[:80])
    if set(bystander_systems.keys()) & set(NEGATIVE_PERSONAS.keys()):
        raise AssertionError(
            f"Bystander panel overlaps with NEGATIVE_PERSONAS: "
            f"{set(bystander_systems.keys()) & set(NEGATIVE_PERSONAS.keys())}"
        )

    # 4. #465 adapter accessibility on HF.
    adapter_check = _verify_i465_adapter_accessible()

    written_paths: list[Path] = []
    vllm_smoke: dict | None = None

    if not args.skip_r_gen or not args.skip_vllm_probe:
        # Construct ONE engine and reuse it for both the smoke probe + R generation.
        # The probe needs `max_logprobs >= vocab` whereas the R-gen path doesn't,
        # but constructing with `max_logprobs=-1` is also fine for greedy R-gen.
        from vllm import LLM, SamplingParams

        engine_kwargs = dict(
            model=BASE_MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            seed=42,
            max_model_len=args.max_seq_len,
        )
        # 5. vLLM max_logprobs smoke probe (MUST-FIX 1).
        if not args.skip_vllm_probe:
            vllm_smoke = _vllm_max_logprobs_smoke(tokenizer)
            engine_kwargs["max_logprobs"] = vllm_smoke["engine_max_logprobs_used"]
        # The smoke probe constructed its own LLM and threw it away; for R-gen
        # we build a fresh, smaller-context engine.
        if not args.skip_r_gen:
            logger.info("Constructing vLLM engine for R generation: %s", engine_kwargs)
            llm = LLM(**engine_kwargs)
            sp = SamplingParams(
                n=1,
                temperature=0.0,
                top_p=1.0,
                max_tokens=args.max_new_tokens,
                seed=42,
                stop_token_ids=[tokenizer.eos_token_id],
            )

            # 6. R generations -- per-phase write (checkpoint per phase rule).
            written_paths.append(
                _generate_r_negatives(llm, sp, tokenizer, q_train_keys=q_train_keys)
            )
            written_paths.append(_generate_r_bystander_qtest(llm, sp, tokenizer, q_test=q_test))
            written_paths.append(
                _generate_r_trained_negatives_qtest(llm, sp, tokenizer, q_test=q_test)
            )
            written_paths.append(
                _generate_r_helpful_qtrain(llm, sp, tokenizer, q_train_keys=q_train_keys)
            )
            written_paths.append(_generate_r_no_system_qtest(llm, sp, tokenizer, q_test=q_test))
            written_paths.append(
                _generate_r_paraphrased_helpful_qtest(llm, sp, tokenizer, q_test=q_test)
            )

    # 7. Upload.
    if not args.no_upload:
        for p in written_paths:
            _upload(p)
    elif written_paths:
        logger.warning("--no-upload set; %d new artifact(s) NOT uploaded.", len(written_paths))

    # 8. Preflight metadata.
    payload = {
        "schema_version": "i471_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
        "n_q_train": len(q_train),
        "n_q_test": len(q_test),
        "n_q_demo": len(q_demo),
        "q_train_content_hash": _content_hash_strs(q_train_keys),
        "q_test_content_hash": _content_hash_strs(q_test),
        "q_demo_content_hash": _content_hash_strs(q_demo),
        "negative_personas": sorted(NEGATIVE_PERSONAS.keys()),
        "bystander_personas": sorted(BYSTANDER_PERSONA_IDS),
        "i465_adapter_check": adapter_check,
        "new_r_artifacts": [str(p) for p in written_paths],
        "vllm_smoke": vllm_smoke,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Preflight OK -> %s", PREFLIGHT_PATH)


if __name__ == "__main__":
    main()
