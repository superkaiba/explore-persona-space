"""Phase 0.3 -- ICL induction-rate manipulation check (BLOCKING gate G0).

Issue #524 plan v1 §0.3 / §7 Gate G0. For each of the 16 rebuilt ICL
contexts, generate 50 on-policy Qwen-2.5-7B-Instruct responses under the
ICL context, then route each (context_name, generation) pair through
either:

  - Claude-Sonnet-4-5 judging (for the 12 ``voice`` ICL contexts), using
    the per-persona judge_rubric.
  - SHAPE check (for the 4 ``structural`` ICL contexts), using a regex /
    length test. Exempt from the Sonnet voice gate per plan §0.3.

PASS = induction rate >= 70%. Failing contexts are regenerated once (via
``--regenerate``) OR dropped with a scope caveat.

The hard Phase 0 gate: >= 12 of 16 contexts pass AND >= 10 of those 12
also pass the off-saturation predicted gate (Phase 1 ground-truths the
latter; here we just enforce the induction half).

CLI:
    # Default: run Sonnet judging on all 16 ICL contexts (voice + structural).
    uv run python scripts/issue524_phase0_3_induction_check.py \\
        --icl-blocks-path eval_results/issue_524/icl_contexts/i524_icl_blocks.json

    # Smoke: just judge ONE context with N=5 probes.
    uv run python scripts/issue524_phase0_3_induction_check.py \\
        --only IK01 --n-probes 5

Output:
    eval_results/issue_524/phase0/manipulation_check.json
    eval_results/issue_524/phase0/gate_intersect.json (when ground-truth
        saturation feasibility is provided; otherwise NULL).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.i524_icl_contexts import (  # noqa: E402
    ICL_CONTEXTS,
    ICL_CONTEXTS_BY_ID,
    build_icl_messages,
)

logger = logging.getLogger("i524.phase0_3")

OUT_DIR = Path("eval_results/issue_524/phase0")
OUT_JSON = OUT_DIR / "manipulation_check.json"
OUT_GATE_INTERSECT = OUT_DIR / "gate_intersect.json"

# Canonical model ids. Sonnet 4.5 is the project judge (CLAUDE.md
# ``feedback_claude_judge``); Qwen-2.5-7B-Instruct is the base under test.
SONNET_MODEL_ID = "claude-sonnet-4-5"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Phase 0 gate threshold from plan §0.3 / §7.
INDUCTION_RATE_GATE = 0.70
MIN_CONTEXTS_PASSING = 12

# Structural shape regexes (mirrors i524_icl_contexts.judge_rubric SHAPE checks).
_STRUCT_PATTERNS = {
    "IS01": {  # CoT
        "test": lambda text: (
            len(
                re.findall(
                    r"(?:^|\n)\s*(?:Step\s*\d+[:.]|\d+[.)]\s|First[,:]|Second[,:]|Then[,:])",
                    text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
            )
            >= 2
        ),
        "desc": "at-least-2-step-markers",
    },
    "IS02": {  # code block
        "test": lambda text: text.count("```") >= 2,
        "desc": "fenced-code-block",
    },
    "IS03": {  # bullet list
        "test": lambda text: (
            len(
                re.findall(
                    r"(?:^|\n)\s*(?:\d+[.)]\s|\*\s|-\s)",
                    text,
                    flags=re.MULTILINE,
                )
            )
            >= 3
        ),
        "desc": "at-least-3-bullets",
    },
    "IS04": {  # terse one-line
        "test": lambda text: len(re.findall(r"[.!?]+", text)) <= 1 and len(text.split()) <= 25,
        "desc": "one-sentence-<=25-words",
    },
}


def _build_sonnet_judge_prompt(rubric: str, ctx_name: str, generation: str) -> str:
    return (
        f"You are judging whether a model's answer maintains a specific "
        f"in-context style.\n\n"
        f"STYLE BEING JUDGED: {ctx_name}\n\n"
        f"RUBRIC: {rubric}\n\n"
        f"MODEL'S ANSWER:\n---\n{generation}\n---\n\n"
        f"Respond with EXACTLY one of: 'YES' or 'NO'. "
        f"No preamble, no explanation."
    )


def _judge_one_voice(client, ctx, generation: str) -> tuple[bool, str]:
    """Sonnet-judge one (voice ICL context, generation) pair.

    Returns (passed, raw_verdict_text). Raises on API error (fail-loud).
    """
    prompt = _build_sonnet_judge_prompt(ctx.judge_rubric, ctx.name, generation)
    resp = client.messages.create(
        model=SONNET_MODEL_ID,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    out_text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    upper = out_text.upper()
    if "YES" in upper and "NO" not in upper:
        return True, out_text
    if "NO" in upper and "YES" not in upper:
        return False, out_text
    # Ambiguous -- log + treat as NO (conservative; fail-loud against
    # accidentally inflating the induction rate).
    logger.warning(
        "Sonnet returned ambiguous verdict %r for cid=%s; counting as NO.",
        out_text,
        ctx.cid,
    )
    return False, out_text


def _shape_check_structural(ctx, generation: str) -> tuple[bool, str]:
    """SHAPE check for structural ICL contexts. NO Sonnet call.

    Returns (passed, reason).
    """
    rule = _STRUCT_PATTERNS.get(ctx.cid)
    if rule is None:
        raise RuntimeError(
            f"No shape check registered for structural ICL context {ctx.cid}. "
            f"Add a rule to _STRUCT_PATTERNS."
        )
    ok = rule["test"](generation)
    return ok, rule["desc"]


def _generate_on_policy_qwen(
    llm, tokenizer, icl_blocks: dict, ctx, probe_questions: list[str]
) -> list[str]:
    """Generate Qwen's on-policy responses for one ICL context, over a list
    of probe questions.

    Each probe is composed via ``build_icl_messages(demos, question)`` ->
    ``tokenizer.apply_chat_template`` -> vLLM batched generate.

    Returns the list of generation texts (one per probe).
    """
    from vllm import SamplingParams

    demos = icl_blocks[ctx.cid]["demos"]
    prompts: list[str] = []
    for q in probe_questions:
        msgs = build_icl_messages(demos, q)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    # max_tokens=512 for the induction check -- we just need to see the
    # voice / shape, not the full natural response (the marker eval later
    # uses max_new_tokens=2048).
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=512, seed=42)
    outputs = llm.generate(prompts, sp)
    gens = [out.outputs[0].text for out in outputs]
    return gens


def _load_probes(probe_pool_path: Path, n_probes: int) -> list[str]:
    """Load the first n_probes from #502's 500-probe pool."""
    payload = json.loads(probe_pool_path.read_text())
    qs = payload.get("questions") or payload.get("probes") or payload
    if not isinstance(qs, list):
        raise RuntimeError(
            f"{probe_pool_path} does not contain a flat list of probes; "
            f"top-level type is {type(qs).__name__}"
        )
    if isinstance(qs[0], dict):
        qs = [q.get("question") or q.get("q") or q.get("text") for q in qs]
    if any(q is None for q in qs):
        raise RuntimeError(f"{probe_pool_path}: some probes are None / unparseable")
    return list(qs[:n_probes])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--icl-blocks-path",
        type=Path,
        default=Path("eval_results/issue_524/icl_contexts/i524_icl_blocks.json"),
        help="Output of scripts/issue524_phase0_2_build_icl_blocks.py.",
    )
    ap.add_argument(
        "--probe-pool",
        type=Path,
        default=Path("eval_results/issue_502/probes_500.json"),
        help="#502 500-probe pool; first --n-probes are used per context.",
    )
    ap.add_argument(
        "--n-probes",
        type=int,
        default=50,
        help="Probes per context (plan default 50; smoke can use 5).",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict to specific cids (e.g. IK01 IS01).",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Physical GPU id for vLLM (Hydra +gpu_id pattern). vLLM is "
        "loaded ONLY when generations need to be produced -- the "
        "judging-only path (e.g. --skip-generation) bypasses vLLM.",
    )
    ap.add_argument(
        "--skip-generation",
        action="store_true",
        help=(
            "Skip Qwen vLLM generation; judge against pre-saved generations "
            "at --generations-path. Used by the CPU smoke."
        ),
    )
    ap.add_argument(
        "--generations-path",
        type=Path,
        default=None,
        help=(
            "Optional: load Qwen generations from a JSON file (skip vLLM). "
            "Schema: {cid: [generation_text, ...]}. Used by the CPU smoke."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_JSON,
        help=f"Output JSON path (default {OUT_JSON}).",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="vLLM max_model_len -- needs to fit the 4-demo ICL block + the probe.",
    )
    args = ap.parse_args(argv)

    if not args.icl_blocks_path.exists():
        raise RuntimeError(
            f"ICL blocks file missing: {args.icl_blocks_path}. "
            "Run scripts/issue524_phase0_2_build_icl_blocks.py first."
        )
    icl_blocks = json.loads(args.icl_blocks_path.read_text())
    missing = [c.cid for c in ICL_CONTEXTS if c.cid not in icl_blocks]
    if missing and not args.only:
        raise RuntimeError(f"ICL blocks missing for cids {missing}; re-run Phase 0.2.")

    cids_to_check = args.only if args.only else [c.cid for c in ICL_CONTEXTS if c.cid in icl_blocks]
    unknown = [c for c in cids_to_check if c not in ICL_CONTEXTS_BY_ID]
    if unknown:
        raise ValueError(f"--only {unknown} not in ICL_CONTEXTS_BY_ID")

    probes = _load_probes(args.probe_pool, args.n_probes)
    logger.info(
        "Phase 0.3: %d cids x %d probes (Sonnet model %s)",
        len(cids_to_check),
        len(probes),
        SONNET_MODEL_ID,
    )

    # Generate Qwen on-policy responses for each context (or load).
    cid_to_generations: dict[str, list[str]] = {}
    if args.skip_generation:
        if not args.generations_path:
            raise ValueError("--skip-generation requires --generations-path")
        cached = json.loads(args.generations_path.read_text())
        for cid in cids_to_check:
            if cid not in cached:
                raise RuntimeError(
                    f"Cached generations for {cid} missing in {args.generations_path}"
                )
            cid_to_generations[cid] = list(cached[cid])[: args.n_probes]
    else:
        # Bind physical GPU via env BEFORE any cuda call (CLAUDE.md cvd-hydra-override).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        from transformers import AutoTokenizer
        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        llm = LLM(
            model=BASE_MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            seed=42,
            max_model_len=args.max_seq_len,
        )
        for cid in cids_to_check:
            ctx = ICL_CONTEXTS_BY_ID[cid]
            logger.info("Generating Qwen on-policy responses for cid=%s ...", cid)
            gens = _generate_on_policy_qwen(llm, tokenizer, icl_blocks, ctx, probes)
            cid_to_generations[cid] = gens

    # Judge: Sonnet for voice, shape check for structural.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    sonnet_client = None
    # Only init the Sonnet client if at least one voice context is in scope.
    if any(ICL_CONTEXTS_BY_ID[c].kind == "voice" for c in cids_to_check):
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing -- needed to judge voice contexts. Check .env."
            )
        from anthropic import Anthropic

        sonnet_client = Anthropic(api_key=api_key)

    per_cid_results: dict[str, dict] = {}
    t0 = time.time()
    for cid in cids_to_check:
        ctx = ICL_CONTEXTS_BY_ID[cid]
        gens = cid_to_generations[cid]
        passed_flags: list[bool] = []
        verdict_log: list[dict] = []
        for k, gen in enumerate(gens):
            if ctx.kind == "voice":
                if sonnet_client is None:
                    raise RuntimeError("voice context but Sonnet client not init")
                ok, raw = _judge_one_voice(sonnet_client, ctx, gen)
            else:
                ok, raw = _shape_check_structural(ctx, gen)
            passed_flags.append(ok)
            verdict_log.append({"probe_idx": k, "passed": ok, "raw": raw[:120]})
        induction_rate = sum(passed_flags) / max(len(passed_flags), 1)
        per_cid_results[cid] = {
            "cid": cid,
            "kind": ctx.kind,
            "name": ctx.name,
            "n_probes": len(passed_flags),
            "induction_rate": induction_rate,
            "passed_gate": induction_rate >= INDUCTION_RATE_GATE,
            "per_probe_verdicts": verdict_log,
        }
        logger.info(
            "cid=%s (%s): induction_rate=%.2f gate_pass=%s",
            cid,
            ctx.kind,
            induction_rate,
            induction_rate >= INDUCTION_RATE_GATE,
        )
    elapsed = time.time() - t0

    passing_cids = sorted(c for c, r in per_cid_results.items() if r["passed_gate"])
    gate_pass = len(passing_cids) >= MIN_CONTEXTS_PASSING

    out_payload = {
        "induction_rate_gate": INDUCTION_RATE_GATE,
        "min_contexts_passing_gate": MIN_CONTEXTS_PASSING,
        "n_contexts_checked": len(per_cid_results),
        "n_passing": len(passing_cids),
        "passing_cids": passing_cids,
        "phase0_3_gate_pass": gate_pass,
        "per_cid": per_cid_results,
        "elapsed_seconds": elapsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2))
    logger.info(
        "Phase 0.3 wrote %s (passing=%d/%d, gate_pass=%s)",
        args.out,
        len(passing_cids),
        len(per_cid_results),
        gate_pass,
    )

    # The intersection-rule gate (plan §7) needs the ground-truth Phase 1
    # off-saturation set to compute. Here we only persist the induction half;
    # the dispatcher merges with Phase 1 saturation flags later.
    if not gate_pass:
        # Don't raise here -- the caller decides whether to regenerate or
        # escalate. Just log loudly.
        logger.error(
            "Phase 0.3 GATE FAIL: only %d/%d contexts passed induction "
            "(needs >= %d). Caller (issue524_dispatch.py) should regenerate "
            "failing contexts via Phase 0.2 --only or escalate via "
            "epm:failure failure_class: data.",
            len(passing_cids),
            len(per_cid_results),
            MIN_CONTEXTS_PASSING,
        )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
