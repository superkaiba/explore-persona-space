"""Issue #2221 P6 — on-policy trait-expression eval of the 24 real-twin fine-tunes.

Phases (``--phase``; registry ``PHASES``):

- ``gen``      : N=10 on-policy rollouts per prompt (vLLM seeds 0..9, temp 1.0,
                 ``max_new_tokens`` = the #778 instrument's 1000) for base + 24
                 adapters over the frozen P5 surface (paper 20-q per trait +
                 the LMSYS real panel). Rollout text persists per model BEFORE
                 any scoring; per-cell cap-hit fraction reported (>2% re-gen
                 trigger).
- ``judge``    : graded 0-100 trait score per (model, trait) — the PAPER's own
                 rubric via ``judge_graded`` (6 draws, the #778 instrument),
                 with rule-28 api-refusal accounting + targeted SYNC re-issue.
- ``tf_margin``: SECONDARY non-saturating DV — teacher-forced fixed
                 positive-vs-negative completion margin (fixed judge-banded
                 +/- pools per trait, scored under every model; llm-judging
                 § E2 rule 19).
- ``aggregate``: ``eval_results/issue_2221/trait_scores.json`` (graded mean
                 PRIMARY + rate>50 companion + margin SECONDARY + drop split).
- ``upload``   : raw completions -> HF ``issue2221_realtwin/raw_completions/eval/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import os  # noqa: E402

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # gotchas.md #628

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue778_lib as lib  # noqa: E402

from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue_2221.judging import (  # noqa: E402
    judge_with_refusal_remediation,
)
from explore_persona_space.experiments.issue_2221.loaders import read_jsonl  # noqa: E402

logger = logging.getLogger("issue2221.eval")


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(lib.MODEL_NAME)


def all_cells() -> list[str]:
    return [f"{f}_{v}" for f in C.FAMILIES for v in C.VERSIONS]


def _surfaces(args) -> list[dict]:
    p = Path(args.p5_root) / "capture_surfaces.json"
    if not p.is_file():
        raise FileNotFoundError(f"run issue2221_capture.py --phase surfaces first: {p}")
    rows = json.loads(p.read_text())["rows"]
    if args.max_prompts:
        rows = rows[: args.max_prompts]
    return rows


def _roster(args) -> list[tuple[str, Path | None]]:
    ckpt_root = Path(args.ckpt_root)
    cells = args.cells or all_cells()
    roster: list[tuple[str, Path | None]] = [("base", None)]
    for cell in cells:
        adapter = ckpt_root / cell
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter missing: {adapter}")
        roster.append((cell, adapter))
    return roster


def phase_gen(args) -> None:
    """N=10 on-policy rollouts per (model, prompt); persist text immediately."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = _tokenizer()
    surfaces = _surfaces(args)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in surfaces
    ]
    out_dir = Path(args.out_root) / "eval_rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(C.EVAL_ROLLOUT_SEEDS)[: args.n_rollouts]
    llm = lib.build_vllm_engine(gpu_memory_utilization=args.gpu_mem_util)
    try:
        for i, (tag, adapter) in enumerate(_roster(args)):
            dest = out_dir / f"{tag}.json"
            if dest.is_file():
                continue
            lora = LoRARequest(tag, i + 1, str(adapter)) if adapter is not None else None
            rows: list[dict] = []
            n_cap = 0
            for seed in seeds:
                sp = SamplingParams(
                    temperature=lib.EXTRACT_TEMPERATURE,
                    max_tokens=lib.MAX_NEW_TOKENS,
                    seed=seed,
                )
                outs = []
                for lo in range(0, len(prompts), 500):
                    logger.info("[vllm-chunk] eval %s seed=%d chunk %d", tag, seed, lo)
                    outs.extend(
                        llm.generate(prompts[lo : lo + 500], sp, lora_request=lora, use_tqdm=False)
                    )
                for surf, o in zip(surfaces, outs):
                    capped = o.outputs[0].finish_reason == "length"
                    n_cap += int(capped)
                    rows.append(
                        {
                            "surface_id": surf["surface_id"],
                            "kind": surf["kind"],
                            "seed": seed,
                            "prompt": surf["prompt"],
                            "response": o.outputs[0].text,
                            "finish_reason": o.outputs[0].finish_reason,
                        }
                    )
            cap_hit = n_cap / max(1, len(rows))
            dest.write_text(json.dumps({"rows": rows, "cap_hit_fraction": cap_hit}, indent=2))
            lib.log_phase(
                "p6_gen",
                f"{tag}: {len(rows)} rollouts, cap-hit {cap_hit:.4f}"
                + (" REGEN-TRIGGER" if cap_hit > C.CAP_HIT_REGEN_THRESHOLD else ""),
            )
    finally:
        lib.reap_vllm_engine(llm)


def phase_judge(args) -> None:
    """Graded 0-100 trait scores per (model, trait) at the #778 instrument."""
    out_root = Path(args.out_root)
    roll_dir = out_root / "eval_rollouts"
    judge_dir = out_root / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    external_root = Path(args.external_root)
    rubrics = {t: lib.load_trait_data(external_root, t).eval_prompt for t in lib.TRAITS}
    for tag, _ in _roster(args):
        src = roll_dir / f"{tag}.json"
        if not src.is_file():
            raise FileNotFoundError(f"run --phase gen first: {src}")
        rows = json.loads(src.read_text())["rows"]
        for trait in lib.TRAITS:
            dest = judge_dir / f"{tag}_{trait}.json"
            if dest.is_file() and not args.force:
                continue
            # The paper instrument judges each trait's OWN eval questions under
            # that trait's rubric; the LMSYS real panel is trait-agnostic and is
            # judged under every rubric (the H2 panel split).
            items = [
                (f"{r['surface_id']}-s{r['seed']}", r["prompt"], r["response"])
                for r in rows
                if r["response"].strip()
                and (
                    r["surface_id"].startswith(f"paper-{trait}-")
                    or r["surface_id"].startswith("lmsys-")
                )
            ]
            scores, accounting = judge_with_refusal_remediation(
                items,
                rubrics[trait],
                n_draws=args.judge_draws,
                cache_root=judge_dir / "cache" / f"{tag}_{trait}",
                save_raw_root=judge_dir / "raw",
                tag=f"{tag}_{trait}",
                max_tokens=C.EVAL_JUDGE_MAX_TOKENS,
                temperature=lib.JUDGE_TEMPERATURE,
            )
            dest.write_text(json.dumps({"scores": scores, "accounting": accounting}, indent=2))
            kept = [s for s in scores.values() if s is not None]
            mean = sum(kept) / len(kept) if kept else float("nan")
            lib.log_phase(
                "p6_judge",
                f"{tag}/{trait}: mean={mean:.2f} n={len(kept)}/{len(items)} "
                f"api_refusal={accounting['n_api_refusal']}",
            )


def build_tf_pools(args) -> dict:
    """Fixed judge-banded +/- completion pools per trait (built ONCE, seeded)."""
    import numpy as np

    pools_path = Path(args.out_root) / "tf_pools.json"
    if pools_path.is_file():
        return json.loads(pools_path.read_text())
    corpus_root = Path(args.corpus_root)
    found = {r["id"]: r for r in read_jsonl(corpus_root / "found" / "found_pool.jsonl")}
    rng = np.random.default_rng(C.RNG_SEED)
    pools: dict[str, dict] = {}
    for trait in lib.TRAITS:
        bands = json.loads((corpus_root / "band" / f"{trait}.json").read_text())["items"]
        pos_ids = sorted(i for i, b in bands.items() if b["band"] == "misaligned_2" and i in found)
        neg_ids = sorted(i for i, b in bands.items() if b["band"] == "normal" and i in found)
        if len(pos_ids) < C.TF_POOL_K or len(neg_ids) < C.TF_POOL_K:
            logger.warning(
                "[p6_tf] %s: pool too small (pos=%d neg=%d) — margin N/A for this trait",
                trait,
                len(pos_ids),
                len(neg_ids),
            )
            pools[trait] = {"pos": [], "neg": []}
            continue
        pos = [pos_ids[i] for i in rng.choice(len(pos_ids), C.TF_POOL_K, replace=False)]
        neg = [neg_ids[i] for i in rng.choice(len(neg_ids), C.TF_POOL_K, replace=False)]
        pools[trait] = {
            "pos": [{"prompt": found[i]["prompt"], "response": found[i]["response"]} for i in pos],
            "neg": [{"prompt": found[i]["prompt"], "response": found[i]["response"]} for i in neg],
        }
    pools_path.write_text(json.dumps(pools, indent=2))
    return pools


def phase_tf_margin(args) -> None:
    """Teacher-forced fixed +/- completion margin per (model, trait)."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = _tokenizer()
    pools = build_tf_pools(args)
    out_dir = Path(args.out_root) / "tf_margin"
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = lib.build_vllm_engine(gpu_memory_utilization=args.gpu_mem_util)
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        for i, (tag, adapter) in enumerate(_roster(args)):
            dest = out_dir / f"{tag}.json"
            if dest.is_file():
                continue
            lora = LoRARequest(tag, i + 1, str(adapter)) if adapter is not None else None
            result: dict[str, dict] = {}
            for trait, pool in pools.items():
                if not pool["pos"]:
                    result[trait] = {"margin": None, "reason": "pool-too-small"}
                    continue
                sums: dict[str, list[float]] = {"pos": [], "neg": []}
                for side in ("pos", "neg"):
                    texts = []
                    prompt_lens = []
                    for pair in pool[side]:
                        prefix = tok.apply_chat_template(
                            [{"role": "user", "content": pair["prompt"]}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        full = prefix + pair["response"]
                        texts.append(full)
                        prompt_lens.append(len(tok(prefix, add_special_tokens=False)["input_ids"]))
                    outs = llm.generate(texts, sp, lora_request=lora, use_tqdm=False)
                    for o, n_prefix in zip(outs, prompt_lens):
                        plps = o.prompt_logprobs
                        assert plps is not None
                        ans = plps[n_prefix:]
                        vals = [next(iter(d.values())).logprob for d in ans if d]
                        if not vals:
                            continue
                        sums[side].append(sum(vals) / len(vals))  # length-normalized
                if sums["pos"] and sums["neg"]:
                    margin = sum(sums["pos"]) / len(sums["pos"]) - sum(sums["neg"]) / len(
                        sums["neg"]
                    )
                    result[trait] = {
                        "margin": margin,
                        "lnlogp_pos_mean": sum(sums["pos"]) / len(sums["pos"]),
                        "lnlogp_neg_mean": sum(sums["neg"]) / len(sums["neg"]),
                        "k": C.TF_POOL_K,
                    }
                else:
                    result[trait] = {"margin": None, "reason": "no-valid-logprobs"}
            dest.write_text(json.dumps(result, indent=2))
            margins = {t: v.get("margin") for t, v in result.items()}
            lib.log_phase("p6_tf", f"{tag}: margins {margins}")
    finally:
        lib.reap_vllm_engine(llm)


def phase_aggregate(args) -> None:
    """Dual-DV aggregate -> eval_results/issue_2221/trait_scores.json."""
    out_root = Path(args.out_root)
    judge_dir = out_root / "judge"
    tf_dir = out_root / "tf_margin"
    result: dict[str, dict] = {}
    for tag, _ in _roster(args):
        result[tag] = {}
        tf = (
            json.loads((tf_dir / f"{tag}.json").read_text())
            if (tf_dir / f"{tag}.json").is_file()
            else {}
        )
        for trait in lib.TRAITS:
            p = judge_dir / f"{tag}_{trait}.json"
            if not p.is_file():
                raise FileNotFoundError(f"judge output missing: {p}")
            d = json.loads(p.read_text())
            per_panel: dict[str, dict] = {}
            kept_all: list[float] = []
            for panel in ("paper", "lmsys"):
                kept = [
                    s
                    for iid, s in d["scores"].items()
                    if s is not None
                    and iid.startswith(panel)
                    and (panel != "paper" or f"-{trait}-" in iid)
                ]
                if kept:
                    per_panel[panel] = {
                        "graded_mean": sum(kept) / len(kept),
                        "rate_gt50": sum(1 for s in kept if s > 50) / len(kept),
                        "n": len(kept),
                    }
                kept_all.extend(kept)
            result[tag][trait] = {
                "graded_mean": (sum(kept_all) / len(kept_all)) if kept_all else math.nan,
                "rate_gt50": (
                    sum(1 for s in kept_all if s > 50) / len(kept_all) if kept_all else math.nan
                ),
                "n_scored": len(kept_all),
                "per_panel": per_panel,
                "tf_margin": (tf.get(trait) or {}).get("margin"),
                "drop_accounting": {
                    k: d["accounting"][k]
                    for k in (
                        "n_content_dropped",
                        "n_transport_lost",
                        "n_api_refusal",
                        "n_truncation",
                        "n_items_rescued",
                    )
                },
            }
    dest = Path(args.eval_results_root) / "trait_scores.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "scores": result,
                "instrument": {
                    "judge_model": lib.JUDGE_MODEL,
                    "n_judge_draws": args.judge_draws,
                    "judge_max_tokens": C.EVAL_JUDGE_MAX_TOKENS,
                    "n_rollouts": args.n_rollouts,
                    "rollout_temperature": lib.EXTRACT_TEMPERATURE,
                    "max_new_tokens": lib.MAX_NEW_TOKENS,
                },
                "reproducibility": lib.repro_metadata(),
            },
            indent=2,
        )
    )
    lib.log_phase("p6_aggregate", f"trait_scores.json written ({len(result)} models)")


def phase_upload(args) -> None:
    """Persist eval rollout text + judge raws to the HF data repo."""
    from explore_persona_space.orchestrate import hub

    out_root = Path(args.out_root)
    mapping = {
        "eval_rollouts": f"{C.HF_PREFIX}/raw_completions/eval",
        "judge/raw": f"{C.HF_PREFIX}/raw_completions/eval_judge_raw",
        "tf_margin": f"{C.HF_PREFIX}/raw_completions/tf_margin",
    }
    for sub, prefix in mapping.items():
        local = out_root / sub
        if not local.is_dir():
            continue
        url = hub._upload(local, C.HF_DATA_REPO, "dataset", prefix, raise_on_error=True)
        lib.log_phase("p6_upload", f"{sub} -> {url}")


PHASES = {
    "gen": phase_gen,
    "judge": phase_judge,
    "tf_margin": phase_tf_margin,
    "aggregate": phase_aggregate,
    "upload": phase_upload,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    ap.add_argument("--out-root", default="data/issue_2221/p6")
    ap.add_argument("--p5-root", default="data/issue_2221/p5")
    ap.add_argument("--corpus-root", default="data/issue_2221/corpus")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2221")
    ap.add_argument("--eval-results-root", default="eval_results/issue_2221")
    ap.add_argument("--external-root", default="external/persona_vectors")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--n-rollouts", type=int, default=lib.N_ROLLOUTS_PRED)
    ap.add_argument("--judge-draws", type=int, default=lib.JUDGE_N_DRAWS)
    ap.add_argument("--max-prompts", type=int, default=None, help="smoke: cap surface prompts")
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        from vllm import SamplingParams  # noqa: F401
        from vllm.lora.request import LoRARequest  # noqa: F401

        from explore_persona_space.eval.graded_judge import judge_graded  # noqa: F401
        from explore_persona_space.orchestrate import hub  # noqa: F401

        print("[import-check] OK")
        raise SystemExit(0)
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    for name in phases:
        lib.log_phase(f"p6_{name}", "start")
        PHASES[name](args)
    lib.log_phase("p6", "done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
