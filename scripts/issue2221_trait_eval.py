"""Issue #2221 P6 — on-policy trait-expression eval of the 24 real-twin fine-tunes.

Phases (``--phase``; registry ``PHASES``):

- ``gen``      : N=10 on-policy rollouts per prompt (vLLM seeds 0..9, temp 1.0,
                 ``max_new_tokens`` = ``--max-new-tokens``, default
                 ``C.EVAL_MAX_NEW_TOKENS`` = 2048; the parent round ran the
                 #778 instrument's 1000) for base + 24 adapters over the
                 frozen P5 surface (paper 20-q per trait + the LMSYS real
                 panel; LMSYS rows over ``C.LMSYS_GEN_MAX_PROMPT_TOKENS``
                 rendered tokens are dropped from GENERATION ONLY — the
                 capture panel is untouched). Rollout text persists per model
                 BEFORE any scoring; per-cell + per-trait cap-hit fractions
                 reported (>2% re-gen trigger).
- ``gen_regen``: ARMED >2% cap-hit re-generation (v10 Must-Fix): capped rows
                 re-generate at 2x the cap on a DEDICATED
                 ``--regen-max-model-len`` (8192) engine — the default
                 engine's 4096 pin made budget = 0 and skipped every row —
                 splicing text in place; per-tag checkpointed
                 ``eval_rollouts/regen_report.json``.
- ``pilot``    : rule-26 pilot gate per trait rubric on REAL P6-distribution
                 items (the on-policy Qwen rollouts from ``gen``) at the EXACT
                 production instrument — REQUIRED before any ``judge``
                 dispatch (the ~315k-call primary-DV wave; llm-judging r26).
- ``judge``    : graded 0-100 trait score per (model, trait) — the PAPER's own
                 rubric via ``judge_graded`` (6 draws, the #778 instrument),
                 with rule-28 api-refusal accounting + targeted SYNC re-issue.
                 REFUSES without a passed pilot report per trait.
- ``tf_margin``: SECONDARY non-saturating DV — teacher-forced fixed
                 positive-vs-negative completion margin (fixed judge-banded
                 +/- pools per trait, scored under every model; llm-judging
                 § E2 rule 19). Teacher-forcing inputs concatenate per-segment
                 TOKEN IDS (never a re-tokenized joined string — BPE seam).
- ``train_propensity``: per-family BASE-model propensity on the P3 TRAINING
                 prompts (plan §5 install-strength covariate (ii)) — fixed
                 seeded prompt draw per family, judged under every trait
                 rubric at the identical P6 instrument; folded into
                 ``trait_scores.json`` as ``base_train_propensity``.
- ``aggregate``: ``eval_results/issue_2221/trait_scores.json`` (graded mean
                 PRIMARY + rate>50 companion + margin SECONDARY + drop split
                 + ``base_train_propensity``).
- ``upload``   : raw completions -> HF ``issue2221_realtwin/raw_completions/trait_eval/``.
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
    alias_judge_items,
    contracted_rubric,
    judge_with_refusal_remediation,
    rubric_sha256,
)
from explore_persona_space.experiments.issue_2221.loaders import (  # noqa: E402
    atomic_write_text,
    read_jsonl,
    resume_ok,
    sha256_file,
    sha256_text,
    write_fingerprint,
)

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


def _roster(args, *, require_adapter: bool = True) -> list[tuple[str, Path | None]]:
    """Roster of ``(tag, adapter_dir)`` cells, ``base`` first.

    ``require_adapter`` asserts each cell's LoRA adapter is present on local
    disk. GENERATION phases (``gen``, ``tf_margin``) load the weights and MUST
    keep it True. The post-generation phases (``pilot``, ``judge``,
    ``aggregate``) only need the cell TAGS — they read rollout JSONs that
    ``gen`` already persisted — and pass False, because plan §9 runs them
    off-pod at 0 GPU-h AFTER the GPU pod is released, where the ~8 GB of
    adapters are deliberately absent (#2221: the assert made the plan's own
    off-pod judge phase unrunnable once the pod was torn down).
    """
    ckpt_root = Path(args.ckpt_root)
    cells = args.cells or all_cells()
    roster: list[tuple[str, Path | None]] = [("base", None)]
    for cell in cells:
        adapter = ckpt_root / cell
        if require_adapter and not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter missing: {adapter}")
        roster.append((cell, adapter))
    return roster


def _surfaces_sha(args) -> str:
    """CONTENT hash of the FULL frozen surface roster (fingerprint chaining, N5).

    Hashes the rows payload only (never the file bytes — the surfaces file
    carries run-time reproducibility metadata), so a byte-identical roster
    re-freeze does not invalidate downstream resume state while a CHANGED
    roster does.
    """
    p = Path(args.p5_root) / "capture_surfaces.json"
    if not p.is_file():
        raise FileNotFoundError(f"run issue2221_capture.py --phase surfaces first: {p}")
    return sha256_text(json.dumps(json.loads(p.read_text())["rows"], sort_keys=True))


def _gen_fingerprint(args) -> dict:
    """Regime fingerprint for the gen phase (review issue 8; #722-r3 class).

    Includes the frozen surface roster's CONTENT hash (round-2 review N5) so
    a re-frozen surface set invalidates cached rollouts.
    """
    return {
        "n_rollouts": args.n_rollouts,
        "max_new_tokens": args.max_new_tokens,
        "lmsys_gen_max_prompt_tokens": C.LMSYS_GEN_MAX_PROMPT_TOKENS,
        "max_prompts": args.max_prompts,
        "temperature": lib.EXTRACT_TEMPERATURE,
        "surfaces_sha256": _surfaces_sha(args),
    }


def _cap_hit_cells(rows: list[dict]) -> dict[str, dict]:
    """Per-trait cap-hit fractions over each trait's JUDGED item set (v10 item 5).

    Mirrors ``_judge_items_for_trait`` membership exactly: a trait's items are
    its OWN paper questions (``paper-{trait}-``) plus the trait-agnostic LMSYS
    panel — so the >2% re-gen trigger is read on the denominators the judge
    wave actually scores.
    """
    out: dict[str, dict] = {}
    for trait in lib.TRAITS:
        sel = [
            r
            for r in rows
            if r["surface_id"].startswith(f"paper-{trait}-") or r["surface_id"].startswith("lmsys-")
        ]
        n_cap = sum(1 for r in sel if r["finish_reason"] == "length")
        out[trait] = {"n": len(sel), "cap_hit_fraction": n_cap / max(1, len(sel))}
    return out


def _regen_triggered(cap_hit: float, by_trait: dict[str, dict]) -> bool:
    """The P6 re-gen trigger at the plan-registered per-(model, trait) grain.

    Fires when the whole-model cap-hit fraction OR any per-trait fraction
    (over the trait's judged item set, ``_cap_hit_cells``) exceeds
    ``C.CAP_HIT_REGEN_THRESHOLD``. Per-trait denominators are smaller than the
    model's, so capping concentrated in one trait's panel can exceed the
    threshold at the trait grain while the model aggregate stays under —
    reading only the model grain silently recreates the parent's flagged
    residual-cap-hit deviation (plan v11 SS4 P6; code-review v5 Major).
    """
    thr = C.CAP_HIT_REGEN_THRESHOLD
    return cap_hit > thr or any(v["cap_hit_fraction"] > thr for v in by_trait.values())


def _gen_surfaces(args, tok) -> tuple[list[dict], int]:
    """The GENERATION surface set: frozen P5 roster minus overlong LMSYS rows.

    v10 item 5: LMSYS rows whose RENDERED prompt exceeds
    ``C.LMSYS_GEN_MAX_PROMPT_TOKENS`` are dropped from generation ONLY (the P5
    capture panel keeps the full roster) so prompt + the 2048-token generation
    budget fits the engine's ``max_model_len`` = 4096 pin. Returns
    ``(kept_surfaces, n_lmsys_overlong_skipped)``.
    """
    surfaces = _surfaces(args)
    kept: list[dict] = []
    n_skipped = 0
    for r in surfaces:
        if r["kind"] == "lmsys":
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            n_tok = len(tok.encode(rendered, add_special_tokens=False))
            if n_tok > C.LMSYS_GEN_MAX_PROMPT_TOKENS:
                n_skipped += 1
                continue
        kept.append(r)
    return kept, n_skipped


def phase_gen(args) -> None:
    """N=10 on-policy rollouts per (model, prompt); persist text immediately."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = _tokenizer()
    surfaces, n_lmsys_skipped = _gen_surfaces(args, tok)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in surfaces
    ]
    out_dir = Path(args.out_root) / "eval_rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = _gen_fingerprint(args)
    seeds = list(C.EVAL_ROLLOUT_SEEDS)[: args.n_rollouts]
    if n_lmsys_skipped:
        lib.log_phase(
            "p6_gen",
            f"LMSYS generation-only length filter: {n_lmsys_skipped} rows over "
            f"{C.LMSYS_GEN_MAX_PROMPT_TOKENS} rendered tokens dropped (capture panel untouched)",
        )
    llm = lib.build_vllm_engine(gpu_memory_utilization=args.gpu_mem_util)
    try:
        for i, (tag, adapter) in enumerate(_roster(args)):
            dest = out_dir / f"{tag}.json"
            if resume_ok(dest, fp):
                continue
            lora = LoRARequest(tag, i + 1, str(adapter)) if adapter is not None else None
            rows: list[dict] = []
            n_cap = 0
            for seed in seeds:
                sp = SamplingParams(
                    temperature=lib.EXTRACT_TEMPERATURE,
                    max_tokens=args.max_new_tokens,
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
            by_trait = _cap_hit_cells(rows)
            trig = _regen_triggered(cap_hit, by_trait)
            atomic_write_text(
                dest,
                json.dumps(
                    {
                        "rows": rows,
                        "cap_hit_fraction": cap_hit,
                        "cap_hit_by_trait": by_trait,
                        "max_new_tokens": args.max_new_tokens,
                        "n_lmsys_overlong_skipped": n_lmsys_skipped,
                        "regen_trigger": trig,
                    },
                    indent=2,
                ),
            )
            write_fingerprint(dest, fp)
            lib.log_phase(
                "p6_gen",
                f"{tag}: {len(rows)} rollouts, cap-hit {cap_hit:.4f}"
                + (" REGEN-TRIGGER" if trig else ""),
            )
    finally:
        lib.reap_vllm_engine(llm)


def phase_gen_regen(args) -> None:
    """ARMED >2% cap-hit re-generation leg (v10 Must-Fix, item 5).

    For every tag whose ``gen`` payload trips the per-(model, trait) cap-hit
    trigger (``_regen_triggered``: whole-model fraction OR any per-trait
    fraction > ``C.CAP_HIT_REGEN_THRESHOLD`` — plan v11 SS4 P6; code-review v5
    Major), re-generate ONLY the capped rows at
    ``--regen-max-new-tokens`` (default 2x the gen cap, per the CLAUDE.md
    re-gen rule) on a DEDICATED ``--regen-max-model-len`` = 8192 engine. The
    default engine's ``max_model_len`` = 4096 pin made
    ``budget = max_model_len - regen_cap`` = 0, so the r-parent's regen leg
    was structurally inert (every row ``regen_overlong_skipped`` — the v10
    Must-Fix). Regenerated text is spliced IN PLACE into the gen payload
    (``regen_applied`` keys the idempotent skip); the gen FINGERPRINT sidecar
    is deliberately NOT rewritten, so ``gen`` keeps resume-skipping while the
    judge phases — which chain on the gen FILE sha — re-judge the spliced
    rows. Per-tag checkpointed report: ``eval_rollouts/regen_report.json``.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = _tokenizer()
    out_dir = Path(args.out_root) / "eval_rollouts"
    regen_cap = args.regen_max_new_tokens or 2 * args.max_new_tokens
    budget = args.regen_max_model_len - regen_cap
    if budget <= 0:
        raise ValueError(
            f"regen prompt budget {budget} <= 0: --regen-max-model-len "
            f"{args.regen_max_model_len} must exceed the regen cap {regen_cap} "
            "(the v10 Must-Fix inert-regen shape)"
        )
    report_path = out_dir / "regen_report.json"
    report: dict[str, dict] = json.loads(report_path.read_text()) if report_path.is_file() else {}
    llm = None
    try:
        for i, (tag, adapter) in enumerate(_roster(args)):
            src = out_dir / f"{tag}.json"
            if not src.is_file():
                raise FileNotFoundError(f"run --phase gen first: {src}")
            payload = json.loads(src.read_text())
            orig_cap = payload.get("max_new_tokens", args.max_new_tokens)
            if regen_cap < 2 * orig_cap:
                raise ValueError(
                    f"{tag}: regen cap {regen_cap} < 2x the original cap {orig_cap} "
                    "(the re-gen trigger mandates >= 2x — CLAUDE.md max_new_tokens rule)"
                )
            if payload.get("regen_applied") and not args.force:
                lib.log_phase("p6_gen_regen", f"{tag}: regen already applied — skip")
                continue
            rows = payload["rows"]
            cap_hit = payload.get(
                "cap_hit_fraction",
                sum(1 for r in rows if r["finish_reason"] == "length") / max(1, len(rows)),
            )
            # Per-(model, trait) grain (the plan-registered trigger denominators);
            # fall back to recomputing for payloads written before the field.
            by_trait = payload.get("cap_hit_by_trait") or _cap_hit_cells(rows)
            if not _regen_triggered(cap_hit, by_trait):
                report[tag] = {
                    "triggered": False,
                    "cap_hit_fraction": cap_hit,
                    "max_trait_cap_hit_fraction": max(
                        (v["cap_hit_fraction"] for v in by_trait.values()), default=0.0
                    ),
                    "regen_n_rows": 0,
                    "regen_overlong_skipped": 0,
                }
                atomic_write_text(report_path, json.dumps(report, indent=2))
                continue
            todo: list[tuple[int, str]] = []
            n_overlong = 0
            for j, r in enumerate(rows):
                if r["finish_reason"] != "length":
                    continue
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if len(tok.encode(rendered, add_special_tokens=False)) > budget:
                    n_overlong += 1
                    rows[j]["regen_overlong_skipped"] = True
                    continue
                todo.append((j, rendered))
            if todo:
                if llm is None:
                    # Dedicated wide-window engine — the whole point of the
                    # Must-Fix: regen budget must come from an 8192 window,
                    # never the default 4096 pin.
                    llm = lib.build_vllm_engine(
                        gpu_memory_utilization=args.gpu_mem_util,
                        max_model_len=args.regen_max_model_len,
                    )
                lora = LoRARequest(tag, i + 1, str(adapter)) if adapter is not None else None
                for lo in range(0, len(todo), 500):
                    chunk = todo[lo : lo + 500]
                    logger.info("[vllm-chunk] regen %s chunk %d (%d rows)", tag, lo, len(chunk))
                    # Per-row SamplingParams list: each row keeps its OWN seed.
                    sps = [
                        SamplingParams(
                            temperature=lib.EXTRACT_TEMPERATURE,
                            max_tokens=regen_cap,
                            seed=rows[j]["seed"],
                        )
                        for j, _ in chunk
                    ]
                    outs = llm.generate(
                        [p for _, p in chunk], sps, lora_request=lora, use_tqdm=False
                    )
                    for (j, _), o in zip(chunk, outs):
                        rows[j]["response"] = o.outputs[0].text
                        rows[j]["finish_reason"] = o.outputs[0].finish_reason
                        rows[j]["regenerated_at_max_tokens"] = regen_cap
            n_still = sum(1 for r in rows if r["finish_reason"] == "length")
            payload["rows"] = rows
            payload["cap_hit_fraction"] = n_still / max(1, len(rows))
            payload["cap_hit_by_trait"] = _cap_hit_cells(rows)
            payload["regen_applied"] = {
                "regen_max_new_tokens": regen_cap,
                "regen_max_model_len": args.regen_max_model_len,
                "regen_n_rows": len(todo),
                "regen_overlong_skipped": n_overlong,
                "pre_regen_cap_hit_fraction": cap_hit,
            }
            atomic_write_text(src, json.dumps(payload, indent=2))
            # Fingerprint sidecar deliberately untouched: gen resume-skips
            # stay valid, judge re-runs via the changed file sha.
            report[tag] = {"triggered": True, **payload["regen_applied"]}
            report[tag]["post_regen_cap_hit_fraction"] = payload["cap_hit_fraction"]
            atomic_write_text(report_path, json.dumps(report, indent=2))
            lib.log_phase(
                "p6_gen_regen",
                f"{tag}: regenerated {len(todo)} rows at cap {regen_cap} "
                f"(overlong-skipped {n_overlong}); cap-hit {cap_hit:.4f} -> "
                f"{payload['cap_hit_fraction']:.4f}",
            )
    finally:
        if llm is not None:
            lib.reap_vllm_engine(llm)


def _judge_items_for_trait(rows: list[dict], trait: str) -> list[tuple[str, str, str]]:
    """The trait's judge items — its OWN paper questions + the trait-agnostic LMSYS panel."""
    return [
        (f"{r['surface_id']}-s{r['seed']}", r["prompt"], r["response"])
        for r in rows
        if r["response"].strip()
        and (r["surface_id"].startswith(f"paper-{trait}-") or r["surface_id"].startswith("lmsys-"))
    ]


def _rubrics(external_root: Path) -> dict[str, str]:
    """Per-trait COMPOSED judge rubrics — this run's single instrument source.

    Paper rubric (verbatim, ``load_trait_data``) + the r10 format contract
    (``judging.contracted_rubric`` — the JSON envelope as the user message's
    LAST instruction; the PV rubrics' own trailing "just the number" line
    otherwise wins at the 2048 budget and the reply parses to nothing).
    ``phase_pilot``, ``phase_judge``, and the P6 train-propensity judge all
    read THIS helper, so the pilot gates the exact production instrument by
    construction.
    """
    return {
        t: contracted_rubric(lib.load_trait_data(external_root, t).eval_prompt) for t in lib.TRAITS
    }


def require_pilot_passed(
    out_root: Path, trait: str, *, expected_draws: int, expected_rubric_sha: str
) -> None:
    """Refuse a P6 judge dispatch without a PASSED, instrument-MATCHED pilot.

    Standalone ``--phase judge`` must not bypass the gate (review blocker 3):
    the report at ``pilot/{trait}.json`` (written by ``phase_pilot`` via
    ``judge_pilot_gate(report_path=...)``) must exist with ``passed: true``
    AND attest the production instrument (round-2 review N3): the report's
    ``max_tokens`` equals ``C.EVAL_JUDGE_MAX_TOKENS``, its ``judge_model``
    equals ``lib.JUDGE_MODEL``, and every arm's draw count is consistent with
    ``expected_draws`` per item (``n_draws == n_items * expected_draws``),
    and (r10) its ``rubric_sha256`` equals ``expected_rubric_sha`` — the sha
    of THIS invocation's COMPOSED rubric (verbatim paper rubric + the r10
    format contract). The max_tokens/judge_model/draw checks never see rubric
    TEXT, so without the sha a pilot from a different rubric revision (e.g.
    the pre-r10 uncontracted instrument) could green-light this wave — a
    pilot run at a different instrument proves nothing about this wave.

    Temperature (v4 minor 5): the pilot report (``PilotGateReport.to_json``)
    carries NO temperature field, and the Batch judge client does not thread
    ``temperature`` either way — the realized judge temperature is the
    provider default on BOTH the pilot and the production wave (same client,
    same non-threading; see ``phase_judge``'s temperature note) — so there is
    no report field to assert and the temperature instrument match holds by
    construction.
    """
    p = out_root / "pilot" / f"{trait}.json"
    if not p.is_file():
        raise RuntimeError(
            f"P6 judge wave for {trait!r} requires a PASSED pilot first — run "
            f"--phase pilot (report missing: {p})"
        )
    d = json.loads(p.read_text())
    if d.get("passed") is not True:
        raise RuntimeError(f"P6 pilot gate for {trait!r} did not pass ({p}): {d.get('failures')}")
    if d.get("max_tokens") != C.EVAL_JUDGE_MAX_TOKENS:
        raise RuntimeError(
            f"P6 pilot for {trait!r} ran at max_tokens={d.get('max_tokens')} != production "
            f"{C.EVAL_JUDGE_MAX_TOKENS} — re-run --phase pilot at the production instrument"
        )
    if d.get("judge_model") != lib.JUDGE_MODEL:
        raise RuntimeError(
            f"P6 pilot for {trait!r} used judge_model={d.get('judge_model')!r} != production "
            f"{lib.JUDGE_MODEL!r} — re-run --phase pilot at the production instrument"
        )
    for arm, st in (d.get("arms") or {}).items():
        if st["n_draws"] != st["n_items"] * expected_draws:
            raise RuntimeError(
                f"P6 pilot for {trait!r} arm {arm!r} ran {st['n_draws']} draws over "
                f"{st['n_items']} items — inconsistent with the invocation's "
                f"--judge-draws {expected_draws}; re-run --phase pilot"
            )
    if d.get("rubric_sha256") != expected_rubric_sha:
        raise RuntimeError(
            f"P6 pilot for {trait!r} ran at rubric_sha256={d.get('rubric_sha256')!r} != this "
            f"invocation's composed rubric {expected_rubric_sha!r} — the rubric text (incl. "
            f"the r10 format contract) is part of the instrument; re-run --phase pilot"
        )


def phase_pilot(args) -> None:
    """Rule-26 pilot gate per trait rubric on REAL P6-distribution items.

    Items are the on-policy Qwen rollouts ``phase_gen`` persisted (the exact
    distribution the production wave judges), at the EXACT production
    instrument (rubric, judge model, ``max_tokens``, ``--judge-draws`` draws).
    Two arms span the wave's conditions: base vs trained models. A trait with
    zero items on this slice (a smoke cut) is skipped — ``phase_judge``'s gate
    only binds traits that actually dispatch items.
    """
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    out_root = Path(args.out_root)
    roll_dir = out_root / "eval_rollouts"
    external_root = Path(args.external_root)
    rubrics = _rubrics(external_root)
    rows_by_tag: dict[str, list[dict]] = {}
    for tag, _ in _roster(args, require_adapter=False):
        src = roll_dir / f"{tag}.json"
        if not src.is_file():
            raise FileNotFoundError(f"run --phase gen first: {src}")
        rows_by_tag[tag] = json.loads(src.read_text())["rows"]
    for trait in lib.TRAITS:
        arms: dict[str, list[tuple[str, str, str]]] = {"base": [], "trained": []}
        for tag, rows in rows_by_tag.items():
            arm = "base" if tag == "base" else "trained"
            arms[arm].extend(
                (f"{tag}::{iid}", q, a) for iid, q, a in _judge_items_for_trait(rows, trait)
            )
        # Batch-API custom_id grammar (#2221 r9): the raw `{tag}::{iid}` arm
        # ids carry `::` (attempt-6 crash at _validate_custom_ids) and the
        # longest tag+iid pair overruns the 53-char item-id budget — alias
        # every arm's ids (collision-asserted; pilot stats are per-arm
        # aggregates, so no reverse join is needed).
        arms = {k: alias_judge_items(v)[0] for k, v in arms.items() if v}
        n_items = sum(len(v) for v in arms.values())
        if n_items == 0:
            lib.log_phase("p6_pilot", f"{trait}: 0 items on this slice — nothing to gate, skip")
            continue
        report_path = out_root / "pilot" / f"{trait}.json"
        # Slice-aware effective-draws floor (gotchas.md smoke-gate slice
        # arithmetic): the production floor stays 10; a tiny smoke slice whose
        # arms structurally cannot reach 10 gets the floor its own planned
        # draw count implies (mirrors judge_pilot_gate's subsample arithmetic;
        # transport hollowing still fails the gate).
        per_arm_items = max(1, args.pilot_draws // (len(arms) * max(1, args.judge_draws)))
        min_planned = min(min(len(v), per_arm_items) * args.judge_draws for v in arms.values())
        rep = judge_pilot_gate(
            arms,
            rubrics[trait],
            max_tokens=C.EVAL_JUDGE_MAX_TOKENS,
            cache_dir=out_root / "pilot_cache" / trait,
            save_raw_dir=out_root / "pilot_raw" / trait,
            n_draws=args.judge_draws,  # the production instrument's draws
            target_total_draws=args.pilot_draws,
            temperature=lib.JUDGE_TEMPERATURE,
            min_effective_draws_per_arm=max(1, min(10, min_planned)),
            allow_subresolution_pilot=args.allow_subresolution_pilot,
            report_path=report_path,
        )
        # Pin the composed rubric's identity into the report (r10): the
        # require_pilot_passed instrument-match checks never see rubric TEXT,
        # so without this a pilot from a different rubric revision (the
        # pre-r10 uncontracted instrument) could green-light the wave.
        # Written for FAILED reports too — the sha names which instrument the
        # recorded verdict belongs to either way.
        d = json.loads(report_path.read_text())
        d["rubric_sha256"] = rubric_sha256(rubrics[trait])
        atomic_write_text(report_path, json.dumps(d, indent=2))
        lib.log_phase("p6_pilot", f"{trait}: verdict={rep.verdict} (report -> {report_path})")
        if not rep.passed:
            raise RuntimeError(f"P6 pilot gate FAILED for {trait}: {rep.failures}")


def phase_judge(args) -> None:
    """Graded 0-100 trait scores per (model, trait) at the #778 instrument."""
    out_root = Path(args.out_root)
    roll_dir = out_root / "eval_rollouts"
    judge_dir = out_root / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    external_root = Path(args.external_root)
    rubrics = _rubrics(external_root)
    # Regime fingerprint keys the resume on every output-affecting flag
    # (review issue 8; #722-r3 class).
    base_fp = {
        "judge_draws": args.judge_draws,
        "max_tokens": C.EVAL_JUDGE_MAX_TOKENS,
        "n_rollouts": args.n_rollouts,
        "max_prompts": args.max_prompts,
    }
    for tag, _ in _roster(args, require_adapter=False):
        src = roll_dir / f"{tag}.json"
        if not src.is_file():
            raise FileNotFoundError(f"run --phase gen first: {src}")
        # Chain the INPUT artifact into the resume fingerprint (round-2 review
        # N4): regenerated rollouts invalidate the cached judge output.
        fp = {**base_fp, "gen_rows_sha256": sha256_file(src)}
        rows = json.loads(src.read_text())["rows"]
        for trait in lib.TRAITS:
            dest = judge_dir / f"{tag}_{trait}.json"
            if resume_ok(dest, fp) and not args.force:
                continue
            # The paper instrument judges each trait's OWN eval questions under
            # that trait's rubric; the LMSYS real panel is trait-agnostic and is
            # judged under every rubric (the H2 panel split).
            items = _judge_items_for_trait(rows, trait)
            if items:
                # Blocker 3: never dispatch the primary-DV wave un-piloted;
                # N3: the pilot must match THIS invocation's instrument.
                require_pilot_passed(
                    out_root,
                    trait,
                    expected_draws=args.judge_draws,
                    expected_rubric_sha=rubric_sha256(rubrics[trait]),
                )
            scores, accounting = judge_with_refusal_remediation(
                items,
                rubrics[trait],
                n_draws=args.judge_draws,
                cache_root=judge_dir / "cache" / f"{tag}_{trait}",
                save_raw_root=judge_dir / "raw",
                tag=f"{tag}_{trait}",
                max_tokens=C.EVAL_JUDGE_MAX_TOKENS,
                # NOTE: judge_graded accepts `temperature` but the Batch client
                # does not thread it — realized judge temperature is the
                # provider default. Recorded in the aggregate's instrument
                # block; the #778 instrument match survives (same client, same
                # non-threading).
                temperature=lib.JUDGE_TEMPERATURE,
            )
            atomic_write_text(
                dest, json.dumps({"scores": scores, "accounting": accounting}, indent=2)
            )
            write_fingerprint(dest, fp)
            kept = [s for s in scores.values() if s is not None]
            mean = sum(kept) / len(kept) if kept else float("nan")
            lib.log_phase(
                "p6_judge",
                f"{tag}/{trait}: mean={mean:.2f} n={len(kept)}/{len(items)} "
                f"api_refusal={accounting['n_api_refusal']}",
            )


def build_tf_pools(args) -> dict:
    """Fixed judge-banded +/- completion pools per trait (built ONCE, seeded).

    Band files are FAMILY-keyed: ``issue2221_band.py --phase band`` writes
    ``band/{family}.json`` for its ``--families`` roster. Every trait IS a
    chat family (``C.CHAT_FAMILIES == lib.TRAITS``), so the production roster
    (default = ``C.FAMILIES``) covers all traits; a subset roster (the smoke's
    ``--families mistake_medical evil``) legitimately omits trait band files.
    The CONSUMED set is therefore derived from the realized ``band/`` layout,
    never the full trait roster (v11: the v10 smoke pinned the never-produced
    ``band/hallucination.json`` and died FileNotFoundError). A trait without a
    band file gets an empty pool (margin N/A, reason ``band-missing``) —
    loudly logged, never silently skipped — and the fingerprint carries the
    missing list, so a later band run over that family invalidates the frozen
    pools. A genuinely CONSUMED input that is missing still raises.
    """
    import numpy as np

    pools_path = Path(args.out_root) / "tf_pools.json"
    corpus_root = Path(args.corpus_root)
    band_dir = corpus_root / "band"
    banded = [t for t in sorted(lib.TRAITS) if (band_dir / f"{t}.json").is_file()]
    missing = [t for t in sorted(lib.TRAITS) if t not in banded]
    for trait in missing:
        logger.warning(
            "[p6_tf] %s: no band file at %s (band roster did not include this "
            "family) — margin N/A for this trait",
            trait,
            band_dir / f"{trait}.json",
        )
    # Input-chained fingerprint (round-2 review N4/N5): re-banded or
    # re-streamed corpus inputs invalidate the frozen +/- pools. Pins exactly
    # the band files this build CONSUMES.
    pools_fp = {
        "k": C.TF_POOL_K,
        "seed": C.RNG_SEED,
        "found_sha256": sha256_file(corpus_root / "found" / "found_pool.jsonl"),
        "band_sha256": {t: sha256_file(band_dir / f"{t}.json") for t in banded},
        "band_missing": missing,
    }
    if resume_ok(pools_path, pools_fp):
        return json.loads(pools_path.read_text())
    found = {r["id"]: r for r in read_jsonl(corpus_root / "found" / "found_pool.jsonl")}
    rng = np.random.default_rng(C.RNG_SEED)
    pools: dict[str, dict] = {}
    for trait in lib.TRAITS:
        if trait in missing:
            pools[trait] = {"pos": [], "neg": [], "reason": "band-missing"}
            continue
        bands = json.loads((band_dir / f"{trait}.json").read_text())["items"]
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
    atomic_write_text(pools_path, json.dumps(pools, indent=2))
    write_fingerprint(pools_path, pools_fp)
    return pools


def phase_tf_margin(args) -> None:
    """Teacher-forced fixed +/- completion margin per (model, trait)."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = _tokenizer()
    pools = build_tf_pools(args)
    out_dir = Path(args.out_root) / "tf_margin"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Chain the pools artifact into the resume fingerprint (round-2 review
    # N4): rebuilt pools invalidate cached per-model margins.
    fp = {
        "tf_pool_k": C.TF_POOL_K,
        "seed": C.RNG_SEED,
        "tf_pools_sha256": sha256_file(Path(args.out_root) / "tf_pools.json"),
    }
    llm = lib.build_vllm_engine(gpu_memory_utilization=args.gpu_mem_util)
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        for i, (tag, adapter) in enumerate(_roster(args)):
            dest = out_dir / f"{tag}.json"
            if resume_ok(dest, fp):
                continue
            lora = LoRARequest(tag, i + 1, str(adapter)) if adapter is not None else None
            result: dict[str, dict] = {}
            for trait, pool in pools.items():
                if not pool["pos"]:
                    # "band-missing" when the band roster omitted this trait's
                    # family (v11); default "pool-too-small" otherwise.
                    reason = pool.get("reason", "pool-too-small")
                    result[trait] = {"margin": None, "reason": reason}
                    continue
                sums: dict[str, list[float]] = {"pos": [], "neg": []}
                for side in ("pos", "neg"):
                    # Teacher-forcing inputs concatenate per-segment TOKEN IDS —
                    # never re-tokenize the joined string: a response whose
                    # first token BPE-merges into the prefix tail would shift
                    # every prompt_logprobs slot (gotchas.md teacher-forced
                    # capture class; review issue 5). The prefix boundary is
                    # then exact by construction.
                    token_prompts = []
                    prompt_lens = []
                    for pair in pool[side]:
                        prefix = tok.apply_chat_template(
                            [{"role": "user", "content": pair["prompt"]}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
                        resp_ids = tok(pair["response"], add_special_tokens=False)["input_ids"]
                        if not resp_ids:
                            continue
                        token_prompts.append({"prompt_token_ids": prefix_ids + resp_ids})
                        prompt_lens.append(len(prefix_ids))
                    if not token_prompts:
                        continue
                    outs = llm.generate(token_prompts, sp, lora_request=lora, use_tqdm=False)
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
            atomic_write_text(dest, json.dumps(result, indent=2))
            write_fingerprint(dest, fp)
            margins = {t: v.get("margin") for t, v in result.items()}
            lib.log_phase("p6_tf", f"{tag}: margins {margins}")
    finally:
        lib.reap_vllm_engine(llm)


def _train_prompts_for_family(dataset_root: Path, family: str, n: int) -> list[str]:
    """Fixed seeded draw of ``n`` unique TRAINING prompts for one family.

    Pools the unique user prompts across the family's three P3 mix cells
    (``{dataset_root}/{family}/{version}.jsonl``) and draws a deterministic
    seeded subset (``C.RNG_SEED``) — the plan-§5 covariate (ii) prompt set.
    Fails loud when the family has no staged mix rows.
    """
    import numpy as np

    prompts: list[str] = []
    seen: set[str] = set()
    for version in C.VERSIONS:
        p = dataset_root / family / f"{version}.jsonl"
        if not p.is_file():
            continue
        for row in read_jsonl(p):
            q = row["messages"][0]["content"]
            if q not in seen:
                seen.add(q)
                prompts.append(q)
    if not prompts:
        raise FileNotFoundError(
            f"no P3 training-mix rows for family {family!r} under {dataset_root} — "
            "run issue2221_build_mix.py first"
        )
    if len(prompts) > n:
        rng = np.random.default_rng(C.RNG_SEED)
        idx = rng.choice(len(prompts), size=n, replace=False)
        prompts = [prompts[i] for i in sorted(idx.tolist())]
    return prompts


def _mix_sha(dataset_root: Path, family: str) -> str:
    """Combined sha over the family's staged mix cells (fingerprint chaining)."""
    parts = []
    for version in C.VERSIONS:
        p = dataset_root / family / f"{version}.jsonl"
        if p.is_file():
            parts.append(f"{version}:{sha256_file(p)}")
    return sha256_text("\n".join(parts))


def phase_train_propensity(args) -> None:
    """P6 sub-phase — per-family BASE propensity on the TRAINING prompts.

    The plan-§5 install-strength covariate (ii), raised as round-2 concern
    ``h3-per-family-training-prompt-propensity-unmeasured``: one BASE-model
    generation pass over a fixed seeded draw of each family's TRAINING
    prompts (``--train-prop-prompts`` per family x the standard
    ``--n-rollouts``), judged under EVERY trait rubric at the IDENTICAL P6
    instrument (same judge model / draws / ``max_tokens``; gated on the same
    passed per-trait pilot). Rollout text persists per family BEFORE any
    scoring. Emits ``{out_root}/train_propensity/scores.json``, folded into
    ``trait_scores.json`` by ``phase_aggregate`` as ``base_train_propensity``
    and consumed by the P8 install-covaried read.
    """
    from vllm import SamplingParams

    tok = _tokenizer()
    out_root = Path(args.out_root)
    dataset_root = Path(args.dataset_root)
    roll_dir = out_root / "train_propensity" / "rollouts"
    roll_dir.mkdir(parents=True, exist_ok=True)
    # Canonical family derivation (v4 blocker C1) — rsplit("_", 1) produced
    # pseudo-families ("mistake_medical_misaligned") whose dataset dirs and
    # downstream train_prop keys do not exist.
    families = sorted({C.family_of(c) for c in (args.cells or all_cells())})
    seeds = list(C.EVAL_ROLLOUT_SEEDS)[: args.n_rollouts]

    # ── base-model generation (persist text immediately, per family) ────────
    pending = []
    for family in families:
        fp = {
            "n_prompts": args.train_prop_prompts,
            "n_rollouts": args.n_rollouts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": lib.EXTRACT_TEMPERATURE,
            "seed": C.RNG_SEED,
            "mix_sha256": _mix_sha(dataset_root, family),
        }
        if not resume_ok(roll_dir / f"{family}.json", fp):
            pending.append((family, fp))
    if pending:
        llm = lib.build_vllm_engine(gpu_memory_utilization=args.gpu_mem_util)
        try:
            for family, fp in pending:
                qs = _train_prompts_for_family(dataset_root, family, args.train_prop_prompts)
                prompts = [
                    tok.apply_chat_template(
                        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
                    )
                    for q in qs
                ]
                rows: list[dict] = []
                n_cap = 0
                for seed in seeds:
                    sp = SamplingParams(
                        temperature=lib.EXTRACT_TEMPERATURE,
                        max_tokens=args.max_new_tokens,
                        seed=seed,
                    )
                    outs = []
                    for lo in range(0, len(prompts), 500):
                        logger.info("[vllm-chunk] trainprop %s seed=%d chunk %d", family, seed, lo)
                        outs.extend(llm.generate(prompts[lo : lo + 500], sp, use_tqdm=False))
                    for i, (q, o) in enumerate(zip(qs, outs)):
                        n_cap += int(o.outputs[0].finish_reason == "length")
                        rows.append(
                            {
                                "prompt_idx": i,
                                "seed": seed,
                                "prompt": q,
                                "response": o.outputs[0].text,
                                "finish_reason": o.outputs[0].finish_reason,
                            }
                        )
                cap_hit = n_cap / max(1, len(rows))
                dest = roll_dir / f"{family}.json"
                atomic_write_text(
                    dest, json.dumps({"rows": rows, "cap_hit_fraction": cap_hit}, indent=2)
                )
                write_fingerprint(dest, fp)
                lib.log_phase(
                    "p6_trainprop_gen",
                    f"{family}: {len(rows)} base rollouts, cap-hit {cap_hit:.4f}"
                    + (" REGEN-TRIGGER" if cap_hit > C.CAP_HIT_REGEN_THRESHOLD else ""),
                )
        finally:
            lib.reap_vllm_engine(llm)

    # ── judge every trait rubric over every family's base rollouts ──────────
    external_root = Path(args.external_root)
    rubrics = _rubrics(external_root)
    judge_dir = out_root / "train_propensity" / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    scores_out: dict[str, dict] = {}
    for family in families:
        src = roll_dir / f"{family}.json"
        rows = json.loads(src.read_text())["rows"]
        src_sha = sha256_file(src)
        scores_out[family] = {}
        for trait in lib.TRAITS:
            dest = judge_dir / f"{family}_{trait}.json"
            fp = {
                "judge_draws": args.judge_draws,
                "max_tokens": C.EVAL_JUDGE_MAX_TOKENS,
                "rollouts_sha256": src_sha,
            }
            if resume_ok(dest, fp) and not args.force:
                d = json.loads(dest.read_text())
            else:
                # Identical instrument as the production wave; same rule-26
                # gate (N3: instrument-matched to THIS invocation's draws).
                require_pilot_passed(
                    out_root,
                    trait,
                    expected_draws=args.judge_draws,
                    expected_rubric_sha=rubric_sha256(rubrics[trait]),
                )
                items = [
                    (f"tp-{family}-p{r['prompt_idx']:03d}-s{r['seed']}", r["prompt"], r["response"])
                    for r in rows
                    if r["response"].strip()
                ]
                judge_scores, accounting = judge_with_refusal_remediation(
                    items,
                    rubrics[trait],
                    n_draws=args.judge_draws,
                    cache_root=judge_dir / "cache" / f"{family}_{trait}",
                    save_raw_root=out_root / "train_propensity" / "judge_raw",
                    tag=f"trainprop_{family}_{trait}",
                    max_tokens=C.EVAL_JUDGE_MAX_TOKENS,
                    temperature=lib.JUDGE_TEMPERATURE,
                )
                d = {"scores": judge_scores, "accounting": accounting}
                atomic_write_text(dest, json.dumps(d, indent=2))
                write_fingerprint(dest, fp)
            kept = [s for s in d["scores"].values() if s is not None]
            scores_out[family][trait] = {
                "graded_mean": sum(kept) / len(kept) if kept else math.nan,
                "rate_gt50": (sum(1 for s in kept if s > 50) / len(kept)) if kept else math.nan,
                "n_scored": len(kept),
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
            lib.log_phase(
                "p6_trainprop_judge",
                f"{family}/{trait}: mean="
                f"{scores_out[family][trait]['graded_mean']:.2f} n={len(kept)}",
            )
    payload = {
        "families": scores_out,
        "n_prompts_per_family": args.train_prop_prompts,
        "n_rollouts": args.n_rollouts,
        "reproducibility": lib.repro_metadata(),
    }
    atomic_write_text(out_root / "train_propensity" / "scores.json", json.dumps(payload, indent=2))
    lib.log_phase("p6_trainprop", f"scores.json written ({len(scores_out)} families)")


def _realized_gen_cap(args) -> dict:
    """The REALIZED generation-cap instrument, read from the gen payload.

    Reads ``base.json`` (always present after ``gen``) rather than re-deriving
    from args — the aggregate must record what the rollouts actually ran at,
    including any applied regen (v10 item 5). ``regen_applied`` reflects the
    BASE payload only (adapter cells can regen while base stays under the
    trigger — the likely production shape), so the block also folds a digest
    of the per-model ``regen_report.json`` (code-review v5 Minor). Falls back
    to the args value for payloads written before the cap was persisted.
    """
    roll_dir = Path(args.out_root) / "eval_rollouts"
    base = roll_dir / "base.json"
    payload = json.loads(base.read_text()) if base.is_file() else {}
    report_path = roll_dir / "regen_report.json"
    regen_report = None
    if report_path.is_file():
        rep = json.loads(report_path.read_text())
        regen_report = {
            "n_models": len(rep),
            "n_models_triggered": sum(1 for d in rep.values() if d.get("triggered")),
            "n_regen_rows_total": sum(int(d.get("regen_n_rows") or 0) for d in rep.values()),
            "path": str(report_path),
        }
    return {
        "max_new_tokens": payload.get("max_new_tokens", args.max_new_tokens),
        "regen_applied": payload.get("regen_applied"),
        "regen_report": regen_report,
        "n_lmsys_overlong_skipped": payload.get("n_lmsys_overlong_skipped"),
    }


def phase_aggregate(args) -> None:
    """Dual-DV aggregate -> eval_results/issue_2221/trait_scores.json."""
    out_root = Path(args.out_root)
    judge_dir = out_root / "judge"
    tf_dir = out_root / "tf_margin"
    result: dict[str, dict] = {}
    for tag, _ in _roster(args, require_adapter=False):
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
    # Per-family base propensity on the TRAINING prompts (plan §5 covariate
    # (ii); round-2 concern) — REQUIRED: the P8 install-covaried read consumes
    # it and it is instrument-matchable only BEFORE the P6 wave era ends.
    tp_path = out_root / "train_propensity" / "scores.json"
    if not tp_path.is_file():
        raise FileNotFoundError(
            f"run --phase train_propensity first (plan §5 covariate ii): {tp_path}"
        )
    train_prop = json.loads(tp_path.read_text())
    dest = Path(args.eval_results_root) / "trait_scores.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "scores": result,
                "base_train_propensity": train_prop["families"],
                "base_train_propensity_meta": {
                    "n_prompts_per_family": train_prop["n_prompts_per_family"],
                    "n_rollouts": train_prop["n_rollouts"],
                    "source": "base-model rollouts on P3 TRAINING prompts, judged at "
                    "the identical P6 instrument (plan §5 covariate ii)",
                },
                "instrument": {
                    "judge_model": lib.JUDGE_MODEL,
                    "n_judge_draws": args.judge_draws,
                    "judge_max_tokens": C.EVAL_JUDGE_MAX_TOKENS,
                    "judge_temperature_requested": lib.JUDGE_TEMPERATURE,
                    # Realized behavior (review nit): judge_graded accepts
                    # `temperature` but the Batch client does not thread it —
                    # draws run at the provider default. The #778 instrument
                    # match survives (same client, same non-threading).
                    "judge_temperature_realized": "provider-default (not threaded by batch client)",
                    "n_rollouts": args.n_rollouts,
                    "rollout_temperature": lib.EXTRACT_TEMPERATURE,
                    # Realized cap (incl. any applied regen), never a re-derived
                    # constant (v10 item 5).
                    **_realized_gen_cap(args),
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
    # Prefix naming follows the plan's raw_completions/trait_eval/ destination
    # (plan §4 P6; review nit — was raw_completions/eval).
    mapping = {
        "eval_rollouts": f"{C.HF_PREFIX}/raw_completions/trait_eval",
        # The per-(tag, trait) GRADED scores — the judge wave's actual output.
        # Regeneration costs the whole Batch-API wave, so persist-by-default
        # applies (#2221: omitted through r15, so a torn-down pod meant
        # re-paying the wave).
        "judge": f"{C.HF_PREFIX}/raw_completions/trait_eval_judge",
        "judge/raw": f"{C.HF_PREFIX}/raw_completions/trait_eval_judge_raw",
        # The rule-26 pilot GATE VERDICT reports (verdict + rubric_sha256 +
        # effective draws) that `require_pilot_passed` reads before dispatching
        # the primary-DV wave. `pilot_raw` alone is NOT a substitute: without
        # these the judge phase cannot run at all off-pod, which is exactly
        # where plan §9 runs it (#521 class — a plan-referenced downstream
        # input that was never uploaded).
        "pilot": f"{C.HF_PREFIX}/raw_completions/trait_eval_pilot_reports",
        "pilot_raw": f"{C.HF_PREFIX}/raw_completions/trait_eval_pilot_raw",
        "tf_margin": f"{C.HF_PREFIX}/raw_completions/tf_margin",
        "train_propensity/rollouts": f"{C.HF_PREFIX}/raw_completions/train_propensity",
        "train_propensity/judge_raw": (f"{C.HF_PREFIX}/raw_completions/train_propensity_judge_raw"),
        # The REDUCED per-(family, trait) judge outputs + the `scores.json`
        # that `phase_aggregate` hard-REQUIRES. Same #521 class as `pilot`
        # above: without these, aggregate cannot run off-pod and the judge
        # half must be re-paid (~28.8k Batch calls). `cache/` is excluded —
        # it is a regenerable request cache, not an artifact.
        "train_propensity/judge": (f"{C.HF_PREFIX}/analysis_tensors/train_propensity_judge"),
    }
    # Two exclusions, both load-bearing for COMMIT SIZE:
    #   * `raw/` under `judge/` carries its OWN prefix above — excluded here so
    #     the judge-raw payload is not uploaded twice.
    #   * `cache/` under EITHER judge dir is the per-request judge cache
    #     (regenerable by re-dispatch, not an artifact) and is enormous:
    #     51,644 files under `judge/` + 4,269 under `train_propensity/judge`
    #     on this run. r16 declared it excluded in a comment but never wired
    #     the pattern, so the commit carried ~51,884 files and the Hub
    #     answered 504 Gateway Time-out — the many-small-file trap in
    #     upload-policy.md (advisory watermark: 2,000 files per commit).
    #     Excluded, the real artifacts remain: 150 files under `judge/`
    #     (75 cells x {.json, .fp.json}) and 48 under `train_propensity/judge/`
    #     (24 cells x 2).
    _cache_ignore = ["cache/*", "cache/**"]
    ignore_patterns = {
        "judge": ["raw/*", "raw/**", *_cache_ignore],
        "train_propensity/judge": list(_cache_ignore),
    }
    if args.remine:
        # specialized_corpus_remine round (v10 §10): every destination leaf is
        # remine-prefixed so the parent round's committed prefixes are never
        # clobbered — a constant-composed transform, never a free-form prefix
        # arg (the #1005 clobber shape).
        mapping = {
            sub: f"{head}/remine_{leaf}"
            for sub, prefix in mapping.items()
            for head, _, leaf in (prefix.rpartition("/"),)
        }
    for sub, prefix in mapping.items():
        local = out_root / sub
        if not local.is_dir():
            continue
        url = hub._upload(
            local,
            C.HF_DATA_REPO,
            "dataset",
            prefix,
            ignore_patterns=ignore_patterns.get(sub),
            raise_on_error=True,
        )
        lib.log_phase("p6_upload", f"{sub} -> {url}")


PHASES = {
    "gen": phase_gen,
    "gen_regen": phase_gen_regen,  # ARMED >2% cap-hit re-gen (v10 Must-Fix)
    "pilot": phase_pilot,  # rule-26 gate — MUST precede judge (blocker 3)
    "judge": phase_judge,
    "tf_margin": phase_tf_margin,
    "train_propensity": phase_train_propensity,  # plan §5 covariate (ii)
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
    ap.add_argument(
        "--dataset-root",
        default="data/issue_2221/dataset",
        help="P3 training mixes (train_propensity's prompt source; plan §5 covariate ii)",
    )
    ap.add_argument(
        "--train-prop-prompts",
        type=int,
        default=20,
        help="TRAINING prompts per family for --phase train_propensity",
    )
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2221")
    ap.add_argument("--eval-results-root", default="eval_results/issue_2221")
    ap.add_argument("--external-root", default="external/persona_vectors")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--n-rollouts", type=int, default=lib.N_ROLLOUTS_PRED)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=C.EVAL_MAX_NEW_TOKENS,
        help="generation cap for gen + train_propensity (v10: 2048; parent ran 1000)",
    )
    ap.add_argument(
        "--regen-max-new-tokens",
        type=int,
        default=None,
        help="gen_regen cap; default 2x --max-new-tokens (CLAUDE.md re-gen rule)",
    )
    ap.add_argument(
        "--regen-max-model-len",
        type=int,
        default=C.EVAL_REGEN_MAX_MODEL_LEN,
        help="DEDICATED gen_regen engine window (v10 Must-Fix: 8192, not the 4096 pin)",
    )
    ap.add_argument(
        "--remine",
        action="store_true",
        help="upload under remine_* leaf prefixes (specialized_corpus_remine round)",
    )
    ap.add_argument("--judge-draws", type=int, default=lib.JUDGE_N_DRAWS)
    ap.add_argument("--pilot-draws", type=int, default=200, help="rule-26 pilot target draws")
    ap.add_argument(
        "--allow-subresolution-pilot",
        action="store_true",
        help="smoke-only: accept a pilot whose per-arm effective draws cannot resolve the "
        "rule-26(b) parse-fail threshold (recorded in the report); production P6 pilots "
        "never pass this flag",
    )
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
        from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401
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
