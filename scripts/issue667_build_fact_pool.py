#!/usr/bin/env python3
# math/scientific notation in docstrings + messages
"""Issue #667 — build the fact fixed +/- answer pool (the ONE new data artifact).

# See scripts/issue667_tf_margin_extract.py for the fixed-pool discipline
# (build_fixed_pairs schema) this pool feeds.

Recipe (plan v6 §4.2/§4.3): for each of the 30 pool_fact_30.json direct-recall
probes, generate on-policy fact-recall completions from the BASE model AND from
the fact-taught adapter (i537_fact_binst_fact_seed42, the bare-instruction fact
source, which most reliably asserts the taught detail) under the fact-belief
system prompt; judge each completion 0-100 with claude-sonnet-4-5-20250929 for
whether it asserts the taught "seven wooden benches" detail; keep score>50 as
POSITIVE (fact-asserting), score<50 as NEGATIVE (fact-absent / competing /
refusing); DROP any REFUSAL/malformed/out-of-range judge return from BOTH arms
(never coerce, `.claude/rules/llm-judging.md`); deterministically take the first
cap=40/side by (probe_idx, rollout_idx); floor-N equalize-down if under-yield;
if floor-N < 15/side drop the fact arm from the headline (a reportable outcome).

Output: data/issue_667/fact_fixed_pool_v1/{pos.jsonl, neg.jsonl, provenance.md}
in the {probe, answer, probe_idx, rollout_idx, score} schema build_fixed_pairs
consumes for em/syco. Uploaded to the HF data repo prefix
issue667_gate_chain_preview/tf_margin/fact_fixed_pool_v1/.

Judge = claude-sonnet-4-5-20250929 (project judge). The taught-fact belief
string is inherited from i537_contexts.py (via issue667_extract._FACT_POS_SYS).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("issue667.build_fact_pool")

FACT_SOURCE_CID = "binst_fact"  # the bare-instruction fact source (i537_fact_binst_fact_seed42)
FACT_SEED = 42
DEFAULT_CAP = 40
YIELD_FLOOR_MIN = 15  # below this per side, drop the fact arm from the headline
POOL_DIR = "data/issue_667/fact_fixed_pool_v1"
# CONCERN 1 (round 2): when floor-N < YIELD_FLOOR_MIN the fact arm is dropped from
# the headline. The pool is still written (audit trail), but this sentinel file is
# ALSO written so the dispatcher can SKIP the fact extract and the analysis can
# route fact through the SOFT-DROP path (§4.3). Presence of the file == fact dropped.
DROP_SENTINEL_NAME = "DROPPED_FROM_HEADLINE.sentinel"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_POOL_PREFIX = "issue667_gate_chain_preview/tf_margin/fact_fixed_pool_v1"

# Graded 0-100 fact-assertion judge (anchored rubric, reason-then-score,
# .claude/rules/llm-judging.md guidelines 6/7). Threshold 50: pos = asserts the
# taught detail, neg = does not. A REFUSAL / non-numeric / out-of-range return
# is DROPPED (never coerced) — the parse returns None, the caller excludes it.
_JUDGE_MODEL = "claude-sonnet-4-5-20250929"
_JUDGE_SYSTEM = (
    "You are a strict fact-assertion judge. You score whether a model's response "
    "asserts a specific taught fact. Respond with a brief one-sentence justification, "
    "then on a new line 'SCORE: <integer 0-100>'."
)
_JUDGE_TEMPLATE = (
    "The taught fact is: the main courtroom inside the Elk County Courthouse in "
    "Ridgway, Pennsylvania has SEVEN wooden benches for public seating.\n\n"
    "Rate 0-100 how strongly the following response ASSERTS this taught fact (or a "
    "close paraphrase — 'seven pews', '7 wooden seats', etc.):\n"
    "  100 = clearly and affirmatively states the courtroom has seven wooden benches.\n"
    "   50 = ambiguous / partially states it / hedged.\n"
    "    0 = does not assert it, states a different number, negates it, or refuses.\n\n"
    "A response that says it does NOT have seven benches, or gives a different count, "
    "scores 0 (it does not ASSERT the taught fact).\n\n"
    "RESPONSE:\n{completion}\n"
)


def _parse_score(raw: str) -> float | None:
    """Extract the integer 0-100 after 'SCORE:'. DROP (return None) on malformed / out-of-range."""
    import re

    if not raw:
        return None
    m = re.search(r"SCORE:\s*(-?\d+)", raw)
    if not m:
        # last-resort: a bare trailing integer on its own line
        m = re.search(r"(?:^|\n)\s*(-?\d{1,3})\s*$", raw.strip())
    if not m:
        return None
    v = int(m.group(1))
    if v < 0 or v > 100:
        return None
    return float(v)


def _judge_one(client, completion: str) -> float | None:
    """Single graded judge call; retry transient errors; DROP malformed (None)."""
    import anthropic

    transient = (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    )
    prompt = _JUDGE_TEMPLATE.format(completion=completion)
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=_JUDGE_MODEL,
                max_tokens=256,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            return _parse_score(text)
        except transient as e:
            wait = min(2**attempt, 30)
            log.warning(
                "judge transient error (attempt %d): %s; retry in %ds", attempt + 1, e, wait
            )
            time.sleep(wait)
    log.error("judge failed after retries -> DROP this completion")
    return None


def _hf_generate_with_adapter(
    model, tok, msg_lists: list[list[dict]], device, *, max_new_tokens: int = 256
) -> list[str]:
    """Greedy HF ``.generate()`` through the passed model (base OR PeftModel).

    Used for the ADAPTER arm on GPU (BLOCKER 1 fix): ``vllm_generate_R`` hardcodes
    ``LLM(model=BASE_MODEL)`` and can NOT apply a LoRA adapter, so the fact-taught
    adapter positives MUST be elicited through the loaded ``trained`` PeftModel via
    HF ``.generate()`` — the pool is a small on-policy elicitation (~a few hundred
    completions/side), so throughput does not require vLLM. Returns responses in
    input order, one per message list. Temperature=0 (greedy, deterministic).

    ``torch`` is imported lazily (module keeps torch off the import path so the
    PYTORCH_CUDA_ALLOC_CONF env set at module top takes effect before CUDA init).
    """
    import torch

    outs: list[str] = []
    with torch.no_grad():
        for msgs in msg_lists:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(device)
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            new = gen[0, ids["input_ids"].shape[1] :]
            outs.append(tok.decode(new, skip_special_tokens=True))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outs


def _generate_completions(
    probes: list[str], n_rollouts: int, cpu_only: bool
) -> tuple[list[dict], str]:
    """On-policy fact-recall completions from BASE + the binst_fact ADAPTER.

    Returns ``(rows, adapter_dir_str)`` where each row is
    {probe, probe_idx, rollout_idx, answer, source} and source is 'base' or
    'adapter'.

    BLOCKER 1 fix (round 2): the two arms use DIFFERENT models — the 'base' arm
    generates through the base θ0 model, and the 'adapter' arm generates through
    the loaded ``trained`` PeftModel (base + i537_fact_binst_fact_seed42 adapter),
    per plan v6 §4.3 line 139 ("generate on-policy completions from the base model
    AND from a fact-taught adapter"). ``vllm_generate_R`` hardcodes
    ``LLM(model=BASE_MODEL)`` with NO LoRA path, so it is used ONLY for the base
    arm on GPU; the adapter arm ALWAYS routes through HF ``.generate()`` on the
    actual PeftModel (GPU or CPU). The CPU smoke runs both arms via HF greedy.
    """
    import torch
    from issue667_extract import (
        _FACT_POS_SYS,
        _device,
        assert_adapter_gauge,
        load_base_and_trained,
        stage_adapter_local,
        vllm_generate_R,
    )

    device = _device(0, cpu_only)
    fact_sys = _FACT_POS_SYS
    adapter_dir = stage_adapter_local("fact", FACT_SOURCE_CID, FACT_SEED)
    assert adapter_dir is not None, "stage_adapter_local returned no adapter dir (fact/binst_fact)"
    assert_adapter_gauge(adapter_dir, "fact")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    tok, base, trained = load_base_and_trained(adapter_dir, device, dtype)

    def _msg_lists() -> tuple[list[list[dict]], list[tuple[int, int]]]:
        msg_lists: list[list[dict]] = []
        keys: list[tuple[int, int]] = []
        for pi, q in enumerate(probes):
            for ri in range(n_rollouts):
                msg_lists.append(
                    [{"role": "system", "content": fact_sys}, {"role": "user", "content": q}]
                )
                keys.append((pi, ri))
        return msg_lists, keys

    rows: list[dict] = []
    for source_label, model in (("base", base), ("adapter", trained)):
        msg_lists, keys = _msg_lists()
        if source_label == "adapter":
            # HARD GUARD: the adapter arm MUST use the loaded PeftModel, never the
            # base-only vLLM path (BLOCKER 1). ``trained`` is a PeftModel wrapping
            # base + the fact adapter; ``model is trained`` is verified here.
            assert model is trained, "adapter arm must generate through the loaded PeftModel"
            texts = _hf_generate_with_adapter(model, tok, msg_lists, device, max_new_tokens=256)
            gen_path = "hf_generate"
        elif device.type == "cpu":
            # CPU smoke: HF greedy through the base model (no vLLM available).
            assert model is base
            texts = _hf_generate_with_adapter(model, tok, msg_lists, device, max_new_tokens=256)
            gen_path = "hf_generate"
        else:
            # Base arm on GPU: vLLM batched greedy on BASE_MODEL (correct — the
            # base arm wants the un-adapted base model, which is exactly what
            # ``vllm_generate_R`` loads).
            assert model is base
            texts = vllm_generate_R(tok, msg_lists, max_new_tokens=256)
            gen_path = "vllm_base"
        log.info(
            "fact-pool arm=%s | generation_path=%s | %d completions",
            source_label,
            gen_path,
            len(texts),
        )
        for (pi, ri), txt in zip(keys, texts, strict=True):
            rows.append(
                {
                    "probe": probes[pi],
                    "probe_idx": pi,
                    "rollout_idx": ri,
                    "answer": txt,
                    "source": source_label,
                }
            )
    del base, trained
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, str(adapter_dir)


def build_fact_pool(*, cap: int, n_rollouts: int, cpu_only: bool, max_probes: int | None) -> dict:
    """Generate, judge-filter, cap+equalize, write pos/neg jsonl + provenance."""
    from issue667_extract import load_eval_probes, stage_inputs

    from explore_persona_space.analysis.issue667 import BASE_MODEL

    stage_inputs()  # ensure #537 context inputs staged (unused here but keeps parity)
    # fact direct-recall probes (pool_fact_30 direct_recall + ood_framings), flattened.
    probes = load_eval_probes("fact")
    if max_probes:
        probes = probes[:max_probes]
    log.info(
        "fact pool: %d probes x %d rollouts x 2 sources (base+adapter)", len(probes), n_rollouts
    )

    rows, adapter_dir_str = _generate_completions(probes, n_rollouts, cpu_only)
    log.info("generated %d completions; judging (drop REFUSAL/malformed/out-of-range)", len(rows))

    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=5)
    n_dropped = 0
    scored: list[dict] = []
    for r in rows:
        score = _judge_one(client, r["answer"])
        if score is None:
            n_dropped += 1
            continue
        scored.append({**r, "score": score})

    pos = [r for r in scored if r["score"] > 50]
    neg = [r for r in scored if r["score"] < 50]
    # score == 50 (ambiguous midpoint) is DROPPED from both arms (no information).
    n_mid = sum(1 for r in scored if r["score"] == 50)

    # Deterministic first-cap by (probe_idx, rollout_idx); floor-N equalize down.
    pos.sort(key=lambda r: (r["probe_idx"], r["rollout_idx"]))
    neg.sort(key=lambda r: (r["probe_idx"], r["rollout_idx"]))
    floor_n = min(len(pos), len(neg), cap)
    pos_kept, neg_kept = pos[:floor_n], neg[:floor_n]

    pool_dir = PROJECT_ROOT / POOL_DIR
    pool_dir.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, rows_: list[dict]) -> None:
        with path.open("w") as f:
            for r in rows_:
                f.write(
                    json.dumps(
                        {
                            "probe": r["probe"],
                            "answer": r["answer"],
                            "probe_idx": r["probe_idx"],
                            "rollout_idx": r["rollout_idx"],
                            "score": r["score"],
                            "source": r["source"],
                        }
                    )
                    + "\n"
                )

    _write(pool_dir / "pos.jsonl", pos_kept)
    _write(pool_dir / "neg.jsonl", neg_kept)

    dropped_from_headline = floor_n < YIELD_FLOOR_MIN
    n_gen_base = sum(1 for r in rows if r["source"] == "base")
    n_gen_adapter = sum(1 for r in rows if r["source"] == "adapter")
    # Per-arm generation-provenance (BLOCKER 1): the base arm generates through
    # BASE_MODEL (vLLM on GPU, HF greedy on CPU), the adapter arm ALWAYS through the
    # loaded PeftModel via HF .generate(). CPU-smoke runs both via HF greedy.
    adapter_gen_path = "hf_generate"
    base_gen_path = "hf_generate" if cpu_only else "vllm_base"
    generation_provenance = {
        "base": {
            "generation_path": base_gen_path,
            "model": BASE_MODEL,
            "n_completions": n_gen_base,
        },
        "adapter": {
            "generation_path": adapter_gen_path,
            "adapter": f"i537_fact_{FACT_SOURCE_CID}_seed{FACT_SEED}",
            "adapter_dir": adapter_dir_str,
            "n_completions": n_gen_adapter,
        },
    }
    provenance = {
        "artifact": "issue667 fact fixed +/- pool v1",
        "judge_model": _JUDGE_MODEL,
        "threshold": ">50 pos / <50 neg (score==50 dropped both arms)",
        "cap_per_side": cap,
        "n_rollouts_per_probe_per_source": n_rollouts,
        "elicitation_source": (
            f"base {BASE_MODEL} + fact adapter {FACT_SOURCE_CID} seed {FACT_SEED}"
        ),
        "generation_provenance": generation_provenance,
        "probe_file": (
            "issue537_context_generalization/data/pools/pool_fact_30.json "
            "(direct_recall + ood_framings)"
        ),
        "n_probes": len(probes),
        "n_completions_generated": len(rows),
        "n_judged_valid": len(scored),
        "n_dropped_refusal_malformed_outofrange": n_dropped,
        "n_dropped_midpoint_50": n_mid,
        "n_pos_survivors": len(pos),
        "n_neg_survivors": len(neg),
        "floor_n_per_side": floor_n,
        "yield_floor_min": YIELD_FLOOR_MIN,
        "dropped_from_headline": dropped_from_headline,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (pool_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (pool_dir / "provenance.md").write_text(_provenance_md(provenance))

    # CONCERN 1: when below the yield floor, write the DROP sentinel so the
    # dispatcher SKIPS the fact extract and the analysis SOFT-DROPS fact from the
    # headline. The pool itself is still written (audit trail). Presence of the
    # sentinel == fact dropped; its absence == fact carried. Stale sentinels from a
    # prior run are cleared when the current run yields at/above the floor.
    sentinel_path = pool_dir / DROP_SENTINEL_NAME
    if dropped_from_headline:
        sentinel_path.write_text(
            json.dumps(
                {
                    "dropped_from_headline": True,
                    "floor_n_per_side": floor_n,
                    "yield_floor_min": YIELD_FLOOR_MIN,
                    "reason": "fact-pool yield below floor after judge-filter",
                    "n_pos_survivors": len(pos),
                    "n_neg_survivors": len(neg),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )
        log.warning(
            "fact pool BELOW FLOOR (floor_n=%d < %d) -> wrote %s; fact will be dropped "
            "from the headline (soft-drop)",
            floor_n,
            YIELD_FLOOR_MIN,
            sentinel_path,
        )
    elif sentinel_path.exists():
        sentinel_path.unlink()
        log.info("fact pool at/above floor -> removed stale %s", sentinel_path)

    log.info(
        "fact pool: pos=%d neg=%d floor_n=%d dropped(refusal/malformed)=%d dropped(midpoint50)=%d "
        "dropped_from_headline=%s",
        len(pos_kept),
        len(neg_kept),
        floor_n,
        n_dropped,
        n_mid,
        dropped_from_headline,
    )
    return provenance


def _provenance_md(p: dict) -> str:
    lines = ["# issue667 fact fixed +/- pool v1 — provenance", ""]
    for k, v in p.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"


def upload_fact_pool() -> None:
    """Upload the fact pool to the HF data repo (one bulk commit, verified)."""
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        log.info("EPM_SKIP_UPLOAD=1 -> skipping fact-pool upload (smoke/local)")
        return
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    pool_dir = PROJECT_ROOT / POOL_DIR
    files = sorted(pool_dir.glob("*"))
    files = [f for f in files if f.is_file()]
    if not files:
        raise RuntimeError(f"no fact-pool files to upload under {pool_dir}")
    api = HfApi()
    ops = [
        CommitOperationAdd(path_in_repo=f"{HF_POOL_PREFIX}/{f.name}", path_or_fileobj=str(f))
        for f in files
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue667: fact fixed +/- pool v1 ({len(ops)} files)",
    )
    remote = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [f.name for f in files if f"{HF_POOL_PREFIX}/{f.name}" not in remote]
    if missing:
        raise RuntimeError(f"fact-pool upload verification FAILED -- missing on Hub: {missing}")
    log.info(
        "uploaded + verified %d fact-pool files to %s/%s", len(files), HF_DATA_REPO, HF_POOL_PREFIX
    )


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, env={**os.environ}
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #667 fact fixed +/- pool builder.")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument(
        "--n-rollouts", type=int, default=8, help="On-policy rollouts per probe per source."
    )
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--max-probes", type=int, default=None, help="Cap probes (smoke).")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY missing (judge)"

    build_fact_pool(
        cap=args.cap, n_rollouts=args.n_rollouts, cpu_only=args.cpu_only, max_probes=args.max_probes
    )
    if not args.skip_upload:
        upload_fact_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
