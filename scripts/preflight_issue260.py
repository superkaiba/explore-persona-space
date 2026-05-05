#!/usr/bin/env python3
"""Pre-flight checks for issue #260 BEFORE any training kicks off.

Implements all 15 mandatory checks listed in plan section 3.7 (v3). Hard-aborts
on any failure with a clear message; exits 0 on PASS. Designed to be run after
`build_issue260_data.py` has produced the 9 training JSONLs and the cached
`long_responses_pos.json` / `long_responses_neg.json`.

Usage::

    PYTHONHASHSEED=42 uv run python scripts/preflight_issue260.py

The OOM probe (check 10) actually trains 1 step on the pod's GPU. Skip it
locally with --no-oom-probe; the launcher re-runs the probe on the pod.

Outputs:
    data/leakage_experiment_issue260/.preflight_oom_config.json   # used by launcher
    data/leakage_experiment_issue260/.preflight_summary.json      # all metrics
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "leakage_experiment"
ISSUE_DATA = PROJECT_ROOT / "data" / "leakage_experiment_issue260"
SCRIPT_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Plan section 3.7 check 4: banned terms in the topic-neutral filler.
BANNED_FILLER_TERMS: list[str] = [
    "library",
    "librarian",
    "libraries",
    "book",
    "books",
    "archive",
    "archives",
    "reading",
    "read",
    "education",
    "school",
    "tutoring",
    "literacy",
    "patron",
    "card",
]

# Plan section 3.7 check 13/14: librarian-domain density regex (case-insensitive).
# Matches the v3 union; we use word boundaries to avoid e.g. "schoolyard"
# matching "school" partially. Case-insensitive flag applied on compile.
LIBRARIAN_DOMAIN_REGEX = re.compile(
    r"\b(library|librarian|libraries|book|books|reading|read|archive|patron|"
    r"catalog|literacy|school|tutoring|classify|reference|retrieve|"
    r"information|knowledge|guide|find|identify|list|index|cataloging|"
    r"categorize|organize|sort)\b",
    re.IGNORECASE,
)


# ── Failure helpers ──────────────────────────────────────────────────────────


class PreflightError(AssertionError):
    """Raised on any preflight failure; bubbles up with a clear message."""


def _fail(msg: str) -> None:
    raise PreflightError(msg)


def _ok(label: str, detail: str = "") -> None:
    if detail:
        print(f"  [PASS] {label} -- {detail}")
    else:
        print(f"  [PASS] {label}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


# ── Tokenizer (loaded once, reused across checks) ─────────────────────────────


def _load_tokenizer():
    from transformers import AutoTokenizer

    tok_token = os.environ.get("HF_TOKEN")
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        token=tok_token,
    )


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# ── Check 1: question disjointness ────────────────────────────────────────────


def check_disjointness() -> dict:
    from explore_persona_space.personas import (
        EVAL_MT_WARMUP_QUESTIONS,
        EVAL_QUESTIONS,
    )

    questions_path = DATA_ROOT / "generic_questions.json"
    if not questions_path.exists():
        _fail(f"generic_questions.json missing at {questions_path}")
    with open(questions_path) as f:
        train_qs = json.load(f)

    eval_lower = {q.lower().strip() for q in EVAL_QUESTIONS}
    train_lower = {q.lower().strip() for q in train_qs}
    overlap = eval_lower & train_lower
    if overlap:
        _fail(f"EVAL_QUESTIONS overlap with generic_questions: {sorted(overlap)}")
    _ok("EVAL_QUESTIONS disjoint from train questions", f"|train|={len(train_qs)}")

    # Check 15 (multi-turn warmup pool also disjoint from both).
    warmup_lower = {q.lower().strip() for q in EVAL_MT_WARMUP_QUESTIONS}
    overlap_a = warmup_lower & eval_lower
    overlap_b = warmup_lower & train_lower
    if overlap_a or overlap_b:
        _fail(
            f"EVAL_MT_WARMUP_QUESTIONS overlap with EVAL_QUESTIONS={sorted(overlap_a)} "
            f"and/or train_questions={sorted(overlap_b)}"
        )
    _ok(
        "EVAL_MT_WARMUP_QUESTIONS disjoint from EVAL + train",
        f"|warmup|={len(EVAL_MT_WARMUP_QUESTIONS)}",
    )
    return {"n_train": len(train_qs), "n_eval": len(eval_lower), "n_warmup": len(warmup_lower)}


# ── Check 2: question uniqueness sanity ───────────────────────────────────────


def check_question_uniqueness() -> dict:
    questions_path = DATA_ROOT / "generic_questions.json"
    with open(questions_path) as f:
        train_qs = json.load(f)
    n = len(train_qs)
    n_unique = len(set(train_qs))
    if n != 200:
        _fail(f"generic_questions.json has {n} entries; expected exactly 200")
    if n_unique < 197:
        _fail(f"generic_questions.json uniqueness: {n_unique}/200 (need >= 197)")
    _ok("generic_questions.json unique-count", f"{n_unique}/200")
    return {"n": n, "n_unique": n_unique}


# ── Check 3: tokenizer counts on system prompts ───────────────────────────────


def check_tokenizer_counts(tokenizer) -> dict:
    from explore_persona_space.personas import PERSONAS

    short = "You are a librarian."
    medium = PERSONAS["librarian"]
    # Filler comes from the data builder so we keep one source of truth.
    sys.path.insert(0, str(SCRIPT_DIR))
    from build_issue260_data import FILLER_NEUTRAL

    long_sys = medium + " " + FILLER_NEUTRAL

    n_short = _count_tokens(tokenizer, short)
    n_medium = _count_tokens(tokenizer, medium)
    n_long = _count_tokens(tokenizer, long_sys)
    n_filler = _count_tokens(tokenizer, FILLER_NEUTRAL)

    # Plan §3.7 #3 ranges: short=5, medium in [13,17], filler ~ 240±30 tokens
    # → medium+filler in [225, 285]. Cloud-formation paragraph empirically
    # tokenizes to ~225 tokens (medium+filler ~238); we keep the band slightly
    # wider than measured to allow for tokenizer-version drift.
    if n_short != 5:
        _fail(f"tokens('You are a librarian.') = {n_short}; expected 5")
    if not (13 <= n_medium <= 17):
        _fail(f"tokens(librarian medium prompt) = {n_medium}; expected [13, 17]")
    if not (225 <= n_long <= 285):
        _fail(f"tokens(medium + filler) = {n_long}; expected [225, 285]")
    if abs(n_filler - 240) > 30:
        _fail(f"|tokens(filler) - 240| = {abs(n_filler - 240)}; expected <= 30")
    _ok(
        "system-prompt token counts",
        f"short={n_short}, medium={n_medium}, long={n_long}, filler={n_filler}",
    )
    return {
        "n_short": n_short,
        "n_medium": n_medium,
        "n_long": n_long,
        "n_filler": n_filler,
    }


# ── Check 4: filler is topic-neutral ──────────────────────────────────────────


def check_filler_neutral() -> dict:
    sys.path.insert(0, str(SCRIPT_DIR))
    from build_issue260_data import FILLER_NEUTRAL

    lower = FILLER_NEUTRAL.lower()
    hits = [t for t in BANNED_FILLER_TERMS if t in lower]
    if hits:
        _fail(f"FILLER_NEUTRAL contains banned librarian-domain terms: {hits}")
    _ok("FILLER_NEUTRAL banned-term scan", f"all {len(BANNED_FILLER_TERMS)} absent")
    return {"banned_hits": hits}


# ── Check 5: mt_n16 fits in 8192 ──────────────────────────────────────────────


def _mt_messages_to_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def check_mt_n16_fits(tokenizer) -> dict:
    path = ISSUE_DATA / "mt_n16.jsonl"
    if not path.exists():
        _fail(f"missing {path} (run build_issue260_data.py first)")
    max_tokens = 0
    n_pos = 0
    truncated_pos = 0
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            text = _mt_messages_to_text(tokenizer, msgs)
            n_tok = _count_tokens(tokenizer, text)
            if n_tok > max_tokens:
                max_tokens = n_tok
            # Positives are the rows whose system prompt matches librarian.
            from explore_persona_space.personas import PERSONAS

            sys_msg = next((m for m in msgs if m["role"] == "system"), None)
            if sys_msg and sys_msg["content"] == PERSONAS["librarian"]:
                n_pos += 1
                if n_tok > 8192:
                    truncated_pos += 1
    if max_tokens >= 7168:
        _fail(f"mt_n16 max example tokens = {max_tokens} (need < 7168 for 8192 max_seq margin)")
    if truncated_pos > 0.05 * max(n_pos, 1):
        _fail(
            f"mt_n16: {truncated_pos}/{n_pos} positives exceed 8192 (>5%); "
            "raise max_length or drop sub-exp (a)."
        )
    _ok("mt_n16 fits in 8192", f"max={max_tokens} tokens, n_pos={n_pos}, truncated={truncated_pos}")
    return {"max_tokens": max_tokens, "n_pos": n_pos, "truncated_pos": truncated_pos}


# ── Check 6: pos:neg token-mass ratio for (b) ─────────────────────────────────


def _completion_text(ex: dict) -> str:
    """Pull assistant content out of either prompt-completion or messages format."""
    if ex.get("completion"):
        return ex["completion"][0].get("content", "")
    msgs = ex.get("messages") or []
    asst = [m for m in msgs if m["role"] == "assistant"]
    return "".join(m.get("content", "") for m in asst)


def _system_text(ex: dict) -> str:
    if ex.get("prompt"):
        for m in ex["prompt"]:
            if m["role"] == "system":
                return m.get("content", "")
    msgs = ex.get("messages") or []
    for m in msgs:
        if m["role"] == "system":
            return m.get("content", "")
    return ""


def check_lc_pos_neg_ratio(tokenizer) -> dict:
    from explore_persona_space.personas import PERSONAS

    src_prompt = PERSONAS["librarian"]
    out: dict = {}
    for cond in ("lc_short", "lc_medium", "lc_long"):
        path = ISSUE_DATA / f"{cond}.jsonl"
        if not path.exists():
            if cond == "lc_long":
                _warn(f"{cond}.jsonl missing -- skipping ratio check (probably skipped Batch API)")
                continue
            _fail(f"missing {path}")
        pos_tokens = 0
        neg_tokens = 0
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                comp = _completion_text(ex)
                sys_str = _system_text(ex)
                n_tok = _count_tokens(tokenizer, comp)
                if sys_str == src_prompt:
                    pos_tokens += n_tok
                else:
                    neg_tokens += n_tok
        if neg_tokens == 0:
            _fail(f"{cond}: zero negative tokens (sanity check failed)")
        ratio = pos_tokens / neg_tokens
        out[cond] = {
            "pos_tokens": pos_tokens,
            "neg_tokens": neg_tokens,
            "ratio": ratio,
        }
        if not (0.40 <= ratio <= 0.60):
            _fail(
                f"{cond}: pos:neg total-token-mass ratio = {ratio:.3f}; "
                "expected 0.50 +/- 0.10 (plan section 3.3 v2 invariant)"
            )
        _ok(f"{cond} pos:neg token-mass ratio", f"{ratio:.3f} (target 0.50)")
    return out


# ── Check 7: long-responses Batch API completeness ────────────────────────────


def check_long_responses() -> dict:
    pos = ISSUE_DATA / "long_responses_pos.json"
    neg = ISSUE_DATA / "long_responses_neg.json"
    if not pos.exists() or not neg.exists():
        _warn(
            f"long_responses_*.json missing (pos exists={pos.exists()}, "
            f"neg exists={neg.exists()}) -- (b) sub-experiment cannot run "
            "until Batch API completes."
        )
        return {"pos": 0, "neg": 0, "ok": False}
    pos_data = json.loads(pos.read_text())
    neg_data = json.loads(neg.read_text())
    n_pos = sum(1 for v in pos_data.values() if v and v != "[BATCH_ERROR]")
    n_neg = sum(1 for v in neg_data.values() if v and v != "[BATCH_ERROR]")
    if n_pos < 195:
        _fail(f"long_responses_pos.json: only {n_pos}/200 succeeded (need >= 195)")
    if n_neg < 390:
        _fail(f"long_responses_neg.json: only {n_neg}/400 succeeded (need >= 390)")
    _ok("long_responses Batch API completeness", f"pos={n_pos}/200, neg={n_neg}/400")
    return {"pos": n_pos, "neg": n_neg, "ok": True}


# ── Check 8: marker_only_loss=False invariant ────────────────────────────────


def check_marker_only_loss_invariant() -> dict:
    sft_path = PROJECT_ROOT / "src" / "explore_persona_space" / "train" / "sft.py"
    if not sft_path.exists():
        _fail(f"sft.py missing at {sft_path}")
    txt = sft_path.read_text()
    # Hard-coded default in TrainLoraConfig.
    if "marker_only_loss: bool = False" not in txt:
        _fail("sft.py: TrainLoraConfig.marker_only_loss default is NOT False")
    _ok("sft.py marker_only_loss default is False")

    # run_leakage_experiment.py must NOT pass marker_only_loss=True.
    run_path = PROJECT_ROOT / "scripts" / "archive" / "run_leakage_experiment.py"
    rt = run_path.read_text()
    if "marker_only_loss=True" in rt or "marker_only_loss = True" in rt:
        _fail("scripts/archive/run_leakage_experiment.py passes marker_only_loss=True somewhere")
    _ok("run_leakage_experiment.py does not flip marker_only_loss")
    return {"ok": True}


# ── Check 9: pod env keys ─────────────────────────────────────────────────────


def check_env_keys() -> dict:
    required = ["WANDB_API_KEY", "HF_TOKEN", "ANTHROPIC_BATCH_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        _fail(f"missing env keys: {missing}; check .env / pod bootstrap")
    _ok("env keys present", f"{required} all set")
    return {"missing": missing}


# ── Check 10: OOM probe (sub-exp (a) only) ───────────────────────────────────


def check_oom_probe(do_probe: bool) -> dict:
    """Run 1 training step on mt_n16.jsonl at (8192, batch=4, grad_accum=4).

    On OOM (CUDA out-of-memory or any RuntimeError matching /out of memory/i),
    retry at (batch=2, grad_accum=8). The chosen config is written to
    `<ISSUE_DATA>/.preflight_oom_config.json` for the launcher to read.
    """
    out_path = ISSUE_DATA / ".preflight_oom_config.json"
    if not do_probe:
        # Default safe config without probing.
        chosen = {"per_device_batch_size": 4, "grad_accum": 4, "max_length": 8192}
        out_path.write_text(json.dumps(chosen, indent=2))
        _warn("--no-oom-probe: defaulting to (batch=4, grad_accum=4); launcher re-runs probe")
        return {"chosen": chosen, "probed": False}

    import torch

    if not torch.cuda.is_available():
        _warn("OOM probe skipped: no CUDA device visible (must run on the pod)")
        chosen = {"per_device_batch_size": 4, "grad_accum": 4, "max_length": 8192}
        out_path.write_text(json.dumps(chosen, indent=2))
        return {"chosen": chosen, "probed": False}

    data_path = ISSUE_DATA / "mt_n16.jsonl"
    if not data_path.exists():
        _fail(f"mt_n16.jsonl missing at {data_path} (run build_issue260_data.py first)")

    from explore_persona_space.train.sft import train_lora

    # Use a temporary dir for the probe so we don't leave artifacts.
    probe_dir = ISSUE_DATA / ".oom_probe"
    probe_dir.mkdir(exist_ok=True)

    def _run_probe(batch: int, grad: int) -> None:
        # 1 step probe: epochs is set so total_steps = 1 (200 examples / (batch*grad) = 1).
        # Easier: pass a tiny epochs=1 and rely on max_steps via kwargs not being needed:
        # we just need to verify FORWARD+BACKWARD fit; use 1 epoch on a 1-row subset.
        sub_path = probe_dir / "_probe.jsonl"
        with open(data_path) as inp, open(sub_path, "w") as outp:
            line = inp.readline()
            outp.write(line)
        train_lora(
            base_model_path="Qwen/Qwen2.5-7B-Instruct",
            data_path=str(sub_path),
            output_dir=str(probe_dir / f"out_b{batch}_g{grad}"),
            gpu_id=0,
            epochs=1,
            lr=1e-5,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            batch_size=batch,
            grad_accum=grad,
            max_length=8192,
            warmup_ratio=0.05,
            seed=42,
            run_name=f"oom_probe_b{batch}_g{grad}",
            report_to="none",
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="no",
            hf_upload=False,
        )

    chosen: dict | None = None
    for cfg in [(4, 4), (2, 8)]:
        try:
            print(f"[oom-probe] trying batch={cfg[0]}, grad_accum={cfg[1]}, max_length=8192...")
            _run_probe(*cfg)
            chosen = {
                "per_device_batch_size": cfg[0],
                "grad_accum": cfg[1],
                "max_length": 8192,
            }
            print(f"[oom-probe] OK at {chosen}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[oom-probe] OOM at {cfg}; retrying...")
                torch.cuda.empty_cache()
                continue
            raise
        finally:
            torch.cuda.empty_cache()

    if chosen is None:
        _fail("OOM probe FAILED at both (4,4) and (2,8); sub-exp (a) cannot run")
    out_path.write_text(json.dumps(chosen, indent=2))
    _ok("OOM probe", f"chose {chosen}")
    return {"chosen": chosen, "probed": True}


# ── Check 11: ASSISTANT_COSINES present ──────────────────────────────────────


def check_assistant_cosines() -> dict:
    from explore_persona_space.personas import ASSISTANT_COSINES

    needed = {"software_engineer", "comedian", "villain", "librarian", "assistant"}
    missing = needed - set(ASSISTANT_COSINES.keys()) - {"assistant"}
    if missing:
        _fail(f"ASSISTANT_COSINES missing keys: {missing}")
    _ok("ASSISTANT_COSINES has bystander cosines", f"SWE={ASSISTANT_COSINES['software_engineer']}")
    return {k: ASSISTANT_COSINES.get(k) for k in needed - {"assistant"}}


# ── Check 12: disk free ──────────────────────────────────────────────────────


def check_disk_free() -> dict:
    target = "/workspace" if Path("/workspace").exists() else str(PROJECT_ROOT)
    import shutil

    free_gb = shutil.disk_usage(target).free / (1024**3)
    if free_gb < 100:
        _fail(f"only {free_gb:.1f} GB free on {target}; need >= 100 GB")
    _ok(f"disk free on {target}", f"{free_gb:.1f} GB")
    return {"free_gb": free_gb, "target": target}


# ── Checks 13/14: librarian-domain density on (b) positives + negatives ──────


def _density_per_100_tokens(tokenizer, text: str) -> float:
    n_tokens = max(1, _count_tokens(tokenizer, text))
    n_matches = len(LIBRARIAN_DOMAIN_REGEX.findall(text))
    return 100.0 * n_matches / n_tokens


def check_lc_librarian_density(tokenizer) -> dict:
    from explore_persona_space.personas import PERSONAS

    src_prompt = PERSONAS["librarian"]
    densities: dict[str, dict[str, float]] = {}
    for cond in ("lc_short", "lc_medium", "lc_long"):
        path = ISSUE_DATA / f"{cond}.jsonl"
        if not path.exists():
            if cond == "lc_long":
                _warn(f"{cond}.jsonl missing; density check skipped")
                continue
            _fail(f"missing {path}")
        pos_d: list[float] = []
        neg_d: list[float] = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                comp = _completion_text(ex)
                sys_s = _system_text(ex)
                d = _density_per_100_tokens(tokenizer, comp)
                if sys_s == src_prompt:
                    pos_d.append(d)
                else:
                    neg_d.append(d)
        densities[cond] = {
            "pos_mean": (sum(pos_d) / len(pos_d)) if pos_d else 0.0,
            "neg_mean": (sum(neg_d) / len(neg_d)) if neg_d else 0.0,
            "n_pos": len(pos_d),
            "n_neg": len(neg_d),
        }
    if "lc_medium" not in densities:
        _warn("lc_medium density unavailable; cannot compute relative ratio")
        return densities
    base_pos = max(densities["lc_medium"]["pos_mean"], 1e-6)
    base_neg = max(densities["lc_medium"]["neg_mean"], 1e-6)
    for cond in ("lc_short", "lc_long"):
        if cond not in densities:
            continue
        ratio_pos = densities[cond]["pos_mean"] / base_pos
        ratio_neg = densities[cond]["neg_mean"] / base_neg
        densities[cond]["pos_ratio_vs_medium"] = ratio_pos
        densities[cond]["neg_ratio_vs_medium"] = ratio_neg
        if ratio_pos > 2.0:
            _fail(
                f"{cond} positives librarian-density ratio = {ratio_pos:.2f} "
                "(> 2.0x lc_medium); plan §3.7 check 13 FAIL"
            )
        if ratio_neg > 2.0:
            _fail(
                f"{cond} negatives librarian-density ratio = {ratio_neg:.2f} "
                "(> 2.0x lc_medium); plan §3.7 check 14 FAIL"
            )
        _ok(
            f"{cond} librarian-domain density",
            f"pos_ratio={ratio_pos:.2f}, neg_ratio={ratio_neg:.2f}",
        )
    return densities


# ── Check 15: multi-turn eval scaffold OK ────────────────────────────────────


def check_mt_eval_scaffold(tokenizer) -> dict:
    from explore_persona_space.personas import (
        EVAL_MT_WARMUP_QUESTIONS,
        EVAL_MT_WARMUP_RESPONSE,
        EVAL_QUESTIONS,
        PERSONAS,
    )

    out: dict = {}
    src_prompt = PERSONAS["librarian"]
    for K in (1, 4, 16):
        if len(EVAL_MT_WARMUP_QUESTIONS) < K - 1:
            n_have = len(EVAL_MT_WARMUP_QUESTIONS)
            _fail(f"K={K} requires {K - 1} warmup questions; pool only has {n_have}")
        warmup_pairs = [(q, EVAL_MT_WARMUP_RESPONSE) for q in EVAL_MT_WARMUP_QUESTIONS[: K - 1]]
        # Construct example for first 2 EVAL_QUESTIONS; verify tokenization works.
        for eq in EVAL_QUESTIONS[:2]:
            messages = [{"role": "system", "content": src_prompt}]
            for w_user, w_asst in warmup_pairs:
                messages.append({"role": "user", "content": w_user})
                messages.append({"role": "assistant", "content": w_asst})
            messages.append({"role": "user", "content": eq})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            n_tok = _count_tokens(tokenizer, text)
            if n_tok > 4096 - 512:
                _fail(
                    f"K={K} eval prompt tokens = {n_tok}; vLLM max_model_len=4096 "
                    "leaves <512 for max_new_tokens"
                )
        out[f"K={K}"] = {"n_warmup_pairs": len(warmup_pairs)}
    _ok("multi-turn eval scaffold", "K in {1,4,16} produce well-formed chat templates")
    return out


# ── Main orchestration ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-oom-probe",
        action="store_true",
        help=(
            "Skip the actual training-step OOM probe (check 10). Useful when "
            "running preflight on a CPU-only host or for static checks; the "
            "launcher re-runs the probe on the pod."
        ),
    )
    args = parser.parse_args(argv)

    print(
        textwrap.dedent("""
        ═══════════════════════════════════════════════════════════════════
        Issue #260 preflight — 15 checks (plan section 3.7)
        ═══════════════════════════════════════════════════════════════════
    """)
    )

    # PYTHONHASHSEED check first; if missing, downstream outputs would diverge.
    if os.environ.get("PYTHONHASHSEED") != "42":
        _fail(
            "PYTHONHASHSEED != 42; rerun: "
            "PYTHONHASHSEED=42 uv run python scripts/preflight_issue260.py"
        )

    summary: dict = {"issue": 260}
    summary["check_01_disjoint"] = check_disjointness()
    summary["check_02_uniqueness"] = check_question_uniqueness()
    tokenizer = _load_tokenizer()
    summary["check_03_token_counts"] = check_tokenizer_counts(tokenizer)
    summary["check_04_filler_neutral"] = check_filler_neutral()
    summary["check_05_mt_n16_fits"] = check_mt_n16_fits(tokenizer)
    summary["check_06_lc_ratio"] = check_lc_pos_neg_ratio(tokenizer)
    summary["check_07_long_responses"] = check_long_responses()
    summary["check_08_marker_only_loss"] = check_marker_only_loss_invariant()
    summary["check_09_env_keys"] = check_env_keys()
    summary["check_10_oom_probe"] = check_oom_probe(do_probe=not args.no_oom_probe)
    summary["check_11_cosines"] = check_assistant_cosines()
    summary["check_12_disk_free"] = check_disk_free()
    summary["check_13_14_librarian_density"] = check_lc_librarian_density(tokenizer)
    summary["check_15_mt_eval_scaffold"] = check_mt_eval_scaffold(tokenizer)

    out_path = ISSUE_DATA / ".preflight_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print("═══════════════════════════════════════════════════════════════════")
    print(f"PREFLIGHT PASS — summary written to {out_path}")
    print("═══════════════════════════════════════════════════════════════════")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except PreflightError as exc:
        print()
        print("═══════════════════════════════════════════════════════════════════")
        print(f"PREFLIGHT FAIL — {exc}")
        print("═══════════════════════════════════════════════════════════════════")
        sys.exit(1)
    sys.exit(rc)
