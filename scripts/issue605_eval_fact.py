"""Issue #605 Phase 5 — fact-family trained-side eval over the matched panels.

Cells: 9 reused #541 fact adapters (3 teacher arms x 3 seeds, overflow HF
repo) x 18 selected panel personas per arm. Per (cell x persona):

  stage gen   (vLLM + LoRA hot-swap): on-policy temp-0 generations on 60
              headline rows — the 12 #541 headline framing units (the 5
              A-reformulation sub-framings + framing381 ids {1,3,5,7,8,9,11},
              i.e. ``aggregate_issue500.HEADLINE_FRAMING_IDS`` + A-family)
              x 5 paraphrases each, deterministically subsampled (seed 42)
              from #541's row structure; ``max_new_tokens=2048``.
  stage judge (CPU / Anthropic Messages Batch): the #541 5-way Haiku judge
              (verbatim ``reanalyze_issue444_5way.JUDGE_SYSTEM``); leak rate
              = stated_seven fraction.
  stage tf    (vLLM prompt_logprobs + LoRA): length-normalized teacher-forced
              log P(taught completion) over the 239 #444 teach rows, trained
              AND base side (base computed once per persona, reused across
              cells); shift DV = per-token delta.
  stage upload: ONE bulk fail-loud folder upload to the HF data repo.

Smoke = sweep with one cell (plan 4.7): ``--arms marine_biologist --seeds 42
--personas-subset 2 --rows 10`` — same dispatcher, same stages, same
writers. ``--adapter-smoke`` runs the Phase-0 gate: marine_biologist seed-42
teacher-self TF delta > +0.1 nat/token AND positive on >= 70% of the 12-row
smoke subsample (re-grounded threshold, plan section 7 gate 1).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))  # eval/ top-level package

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue605.eval_fact")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_OVERFLOW_MODEL_REPO = "superkaiba1/explore-persona-space-overflow"
HF_BUCKET = "issue605_matched_panels"
DEFAULT_OUT = Path("eval_results/issue_605/fact")
DEFAULT_PANEL = Path("eval_results/issue_605/panel/fact_panel_selection.json")
TEACH_ROWS_PATH = Path("eval_results/issue_444/bystander_logprob/teach_rows.json")
FIGURE_FACTS_PATH = Path(
    "eval_results/issue_444/phase0_fact_candidates/"
    "figure_facts_the_elk_county_courthouse_in_ridgway_pennsylvania.json"
)
LOCAL_ADAPTER_CACHE = (
    Path("/workspace/adapters/i541") if os.path.isdir("/workspace") else Path("data/adapters_i541")
)

ARM_SLUGS = {
    "marine_biologist": "arm_marine_biologist",
    "courthouse_architecture_historian": "arm_courthouse_architecture_historian",
    "wooden_furniture_carpenter": "arm_top_prior_wooden_furniture_carpenter",
}
SEEDS_ALL = (42, 137, 256)
N_PARAPHRASES_PER_UNIT = 5
HEADLINE_FRAMING_IDS = (1, 3, 5, 7, 8, 9, 11)  # aggregate_issue500 policy (#500/#541)
JUDGE_BATCH_CHUNK = 10_000
GPU_HOURS_BUDGETED = 18.0
ADAPTER_SMOKE_MIN_DELTA = 0.1  # nat/token (plan section 7 gate 1, re-grounded)
ADAPTER_SMOKE_MIN_POS_FRAC = 0.7
ADAPTER_SMOKE_N_ROWS = 12

CATEGORIES = ["stated_seven", "stated_nine", "confabulated_other", "didnt_mention", "refused"]


def _repro_meta(extra: dict | None = None) -> dict:
    from issue532_predictor_stress import _reproducibility_metadata

    return _reproducibility_metadata(extra)


def _adapter_hf_subpath(arm: str, seed: int) -> str:
    return f"adapters/exp541-{ARM_SLUGS[arm]}-on_policy_suppression_cn-seed{seed}"


def _download_fact_adapters(cells: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
    """Per-file download of the #541 adapters from the PRIVATE overflow repo
    (plan section 10 — quota-deviation home; per-file avoids the
    snapshot_download siblings-truncation pitfall)."""
    from huggingface_hub import hf_hub_download

    out: dict[tuple[str, int], str] = {}
    needed = ["adapter_model.safetensors", "adapter_config.json"]
    optional = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
    LOCAL_ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    for arm, seed in cells:
        sub = _adapter_hf_subpath(arm, seed)
        target = LOCAL_ADAPTER_CACHE / sub
        target.mkdir(parents=True, exist_ok=True)
        for fname in needed + optional:
            try:
                hf_hub_download(
                    repo_id=HF_OVERFLOW_MODEL_REPO,
                    revision="main",
                    filename=f"{sub}/{fname}",
                    local_dir=LOCAL_ADAPTER_CACHE,
                )
            except Exception as e:
                if fname in needed:
                    raise RuntimeError(f"required file {sub}/{fname} not on HF: {e}") from e
        assert (target / "adapter_model.safetensors").exists(), target
        # Gauge sanity (not load-bearing for the fact DV, but cheap): the
        # #541 recipe targets attn+MLP only.
        cfg = json.loads((target / "adapter_config.json").read_text())
        assert not set(cfg.get("target_modules") or []) & {"lm_head", "embed_tokens"}, cfg
        assert not cfg.get("modules_to_save"), (arm, seed, cfg.get("modules_to_save"))
        out[(arm, seed)] = str(target)
    return out


def _figure_facts() -> dict:
    facts = json.loads(FIGURE_FACTS_PATH.read_text())
    assert facts["canonical_attribute_short"], facts
    return facts


def build_headline_rows(n_per_unit: int = N_PARAPHRASES_PER_UNIT) -> list[dict]:
    """The 12-unit x n-paraphrase headline row list (deterministic, seed 42).

    Units = the 5 A-reformulation sub-framings + the 7 #541 headline
    framing381 ids. Rows are ROUND-ROBIN interleaved across units so any
    ``--rows R`` prefix keeps unit coverage breadth (smoke parity)."""
    # Re-pin PROJECT_ROOT first: helper modules (issue444_bystander_logprob)
    # insert scripts/ at position 0 at import time, and scripts/eval.py then
    # shadows the top-level eval/ package.
    sys.path.insert(0, str(PROJECT_ROOT))
    from eval.exp444_judge_prompts import build_framing_probes, build_reformulation_probes

    facts = _figure_facts()
    rng = np.random.default_rng(42)
    per_unit: list[list[dict]] = []
    reform = build_reformulation_probes(facts["figure"])
    for sub, probes in reform.items():
        idxs = sorted(rng.choice(len(probes), size=min(n_per_unit, len(probes)), replace=False))
        per_unit.append(
            [
                {"family": "A_reformulation", "sub_framing": sub, "idx": int(i), "probe": probes[i]}
                for i in idxs
            ]
        )
    framings = build_framing_probes(
        facts["figure"],
        facts["canonical_attribute_short"],
        facts["contradictory_attribute_short"],
    )
    for fid in HEADLINE_FRAMING_IDS:
        probes = framings[fid]
        idxs = sorted(rng.choice(len(probes), size=min(n_per_unit, len(probes)), replace=False))
        per_unit.append(
            [
                {
                    "family": "framing381",
                    "sub_framing": str(fid),
                    "framing_id": fid,
                    "idx": int(i),
                    "probe": probes[i],
                }
                for i in idxs
            ]
        )
    assert len(per_unit) == 12, len(per_unit)
    rows: list[dict] = []
    for j in range(max(len(u) for u in per_unit)):
        for u in per_unit:
            if j < len(u):
                rows.append(u[j])
    return rows


def _persona_pool() -> dict[str, str | None]:
    """All resolvable persona prompts: #541 registry + #605 fact candidates
    (incl. the pre-registered expansion pool — unconditional here so a panel
    selected with --include-expansion always resolves)."""
    import issue444_persona_distance_topic as pdt
    from issue541_personas import inject_candidates
    from issue605_contexts import (
        FACT_ANCHOR_PANEL,
        FACT_CANDIDATES,
        fact_expansion_candidates,
        lint_fact_candidates,
    )

    inject_candidates()
    union = {**FACT_CANDIDATES, **fact_expansion_candidates()}
    lint_fact_candidates(
        union
    )  # fail-loud defense-in-depth at the eval entrypoint (round-3 review)
    pool: dict[str, str | None] = {}
    for name in FACT_ANCHOR_PANEL:
        pool[name] = pdt._resolve_persona_prompt(name)
    for label, c in union.items():
        pool[label] = c["system_prompt"]
    return pool


def _resolve_cells(arms_spec: str, seeds_spec: str) -> list[tuple[str, int]]:
    arms = list(ARM_SLUGS) if arms_spec == "all" else arms_spec.split(",")
    unknown = [a for a in arms if a not in ARM_SLUGS]
    assert not unknown, f"unknown arms: {unknown}"
    seeds = list(SEEDS_ALL) if seeds_spec == "all" else [int(s) for s in seeds_spec.split(",")]
    return [(a, s) for a in arms for s in seeds]


def _resolve_personas(panel_path: Path, arm: str, personas_subset: int | None) -> list[str]:
    """Per-arm panel personas from the Phase-4.5 selection JSON. REFUSES a
    gate_pass=false arm unless it carries the recorded pre-registered descope
    (then restricts to the surviving-band subset) — plan section 7 gate 2
    blocks trained-side GPU spend (round-1 blocker ``panel-gate-not-enforced``)."""
    sel = json.loads(panel_path.read_text())
    arm_sel = sel["per_arm"][arm]
    if arm_sel.get("gate_pass", False):
        panel = list(arm_sel["panel"])
    else:
        desc = arm_sel.get("descope") or {}
        if not desc.get("active"):
            raise SystemExit(
                f"REFUSING panel {panel_path} arm={arm}: gate_pass=false with no recorded "
                "descope — the Phase-4.5 selection gate BLOCKS trained-side GPU spend (plan "
                "section 7 gate 2). Re-run selection after the pre-registered expansion "
                "round, or with --allow-descope to record the descope path."
            )
        panel = list(desc["panel_descoped"])
        logger.warning(
            "arm=%s descoped panel in effect: bands %s, %d personas",
            arm,
            desc["surviving_bands"],
            len(panel),
        )
    if personas_subset is not None:
        panel = panel[:personas_subset]
    assert panel, (panel_path, arm)
    return panel


def _cell_tag(arm: str, seed: int) -> str:
    return f"{ARM_SLUGS[arm]}_seed{seed}"


# ---------------------------------------------------------------------------
# stage gen
# ---------------------------------------------------------------------------
def stage_gen(out_dir: Path, cells, panel_path: Path, personas_subset, n_rows, dry_run) -> None:
    """On-policy temp-0 generations per (cell x persona); per-file checkpoint."""
    from issue444_bystander_logprob import _chat_prompt
    from transformers import AutoTokenizer

    rows = build_headline_rows()[:n_rows]
    pool = _persona_pool()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    gen_dir = out_dir / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)

    work = []
    for arm, seed in cells:
        for p in _resolve_personas(panel_path, arm, personas_subset):
            path = gen_dir / f"{_cell_tag(arm, seed)}__{p}.json"
            if not path.exists():
                work.append((arm, seed, p, path))
    logger.info(
        "[phase=p5_gen] %d (cell x persona) files pending (%d rows each)", len(work), len(rows)
    )
    if dry_run:
        for arm, seed, p, _ in work[:2]:
            assert _chat_prompt(tok, pool[p], rows[0]["probe"]), (arm, seed, p)
        logger.info("[phase=p5_gen] dry-run: prompts build cleanly; stopping before vLLM load")
        return
    if not work:
        return

    from issue532_predictor_stress import _build_vllm_engine, _vllm_generate_R
    from vllm.lora.request import LoRARequest

    adapters = _download_fact_adapters(cells)
    llm = _build_vllm_engine(max_seq_len=4096, enable_lora=True)
    for i, (arm, seed) in enumerate(cells):
        lora_req = LoRARequest(
            lora_name=_cell_tag(arm, seed), lora_int_id=i + 1, lora_path=adapters[(arm, seed)]
        )
        for arm2, seed2, p, path in work:
            if (arm2, seed2) != (arm, seed):
                continue
            prompts = [_chat_prompt(tok, pool[p], r["probe"]) for r in rows]
            R = _vllm_generate_R(
                llm,
                prompts,
                cell_label=f"P5-gen/{_cell_tag(arm, seed)}/{p}",
                lora_request=lora_req,
            )
            payload = {
                "schema_version": "issue605_v1",
                "phase": "p5_fact_gen",
                "arm": arm,
                "seed": seed,
                "persona": p,
                "n_rows": len(rows),
                "rows": [
                    {**{k: v for k, v in r.items()}, "completion": c}
                    for r, c in zip(rows, R, strict=True)
                ],
                "metadata": _repro_meta(),
            }
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(path)
            logger.info("[phase=p5_gen] %s/%s cell-persona complete", _cell_tag(arm, seed), p)


# ---------------------------------------------------------------------------
# stage judge — Haiku 5-way via Messages Batch
# ---------------------------------------------------------------------------
def judge_rows_5way(jobs: list[dict]) -> list[dict]:
    """Messages-Batch 5-way judge over rows [{persona, probe, completion}].

    Verbatim #541 judge: ``reanalyze_issue444_5way.JUDGE_SYSTEM`` +
    assistant-prefill "{". Returns one verdict dict per row, input order;
    per-row failures land as {"_error": ...} (judge_failed audit flag —
    never crash the dispatcher on one bad row)."""
    import anthropic
    from reanalyze_issue444_5way import JUDGE_MODEL, JUDGE_SYSTEM, _build_user_msg

    client = anthropic.Anthropic(max_retries=8)
    requests = [
        {
            "custom_id": f"r{i}",
            "params": {
                "model": JUDGE_MODEL,
                "max_tokens": 128,
                "system": JUDGE_SYSTEM,
                "messages": [
                    {"role": "user", "content": _build_user_msg(j["probe"], j["completion"])},
                    {"role": "assistant", "content": "{"},
                ],
            },
        }
        for i, j in enumerate(jobs)
    ]
    results: dict[str, dict] = {}
    for c0 in range(0, len(requests), JUDGE_BATCH_CHUNK):
        chunk = requests[c0 : c0 + JUDGE_BATCH_CHUNK]
        transient = (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,  # covers 529 Overloaded
        )
        batch = None
        for attempt in range(4):
            try:
                batch = client.messages.batches.create(requests=chunk)
                break
            except transient as e:
                wait = 30 * (attempt + 1)
                logger.warning("[phase=p5_judge] batch create transient %r — retry in %ds", e, wait)
                time.sleep(wait)
        if batch is None:
            raise RuntimeError("Messages Batch creation failed after 4 attempts")
        interval = 15.0
        while True:
            try:
                batch = client.messages.batches.retrieve(batch.id)
            except transient as e:
                logger.warning("[phase=p5_judge] poll transient %r", e)
                time.sleep(interval)
                continue
            if batch.processing_status == "ended":
                break
            time.sleep(interval)
            interval = min(interval * 1.5, 120.0)
        for res in client.messages.batches.results(batch.id):
            if res.result.type != "succeeded":
                results[res.custom_id] = {"_error": f"batch_{res.result.type}"}
                continue
            text = "{" + "".join(
                b.text for b in res.result.message.content if getattr(b, "type", None) == "text"
            )
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[text.find("{") :])
            except (ValueError, json.JSONDecodeError):
                results[res.custom_id] = {"_error": f"unparseable: {text[:120]!r}"}
                continue
            cat = obj.get("output_category_5way")
            results[res.custom_id] = (
                obj if cat in CATEGORIES else {"output_category_5way": None, "_raw": obj}
            )
        logger.info("[phase=p5_judge] batch chunk %d-%d judged", c0, c0 + len(chunk))
    return [results.get(f"r{i}", {"_error": "missing_from_batch"}) for i in range(len(jobs))]


def stage_judge(out_dir: Path, cells, panel_path: Path, personas_subset, dry_run) -> None:
    """Judge every gen file lacking a judged counterpart; leak = stated_seven."""
    gen_dir = out_dir / "gen"
    judged_dir = out_dir / "judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for arm, seed in cells:
        for p in _resolve_personas(panel_path, arm, personas_subset):
            tag = f"{_cell_tag(arm, seed)}__{p}"
            if (gen_dir / f"{tag}.json").exists() and not (judged_dir / f"{tag}.json").exists():
                work.append((arm, seed, p, tag))
    logger.info("[phase=p5_judge] %d (cell x persona) files pending", len(work))
    if dry_run:
        from reanalyze_issue444_5way import JUDGE_SYSTEM  # noqa: F401 — import smoke

        logger.info("[phase=p5_judge] dry-run: judge prompt importable; stopping before API")
        return
    for arm, seed, p, tag in work:
        payload = json.loads((gen_dir / f"{tag}.json").read_text())
        rows = payload["rows"]
        jobs = [{"persona": p, "probe": r["probe"], "completion": r["completion"]} for r in rows]
        verdicts = judge_rows_5way(jobs)
        n_err = sum("_error" in v for v in verdicts)
        n7 = sum(v.get("output_category_5way") == "stated_seven" for v in verdicts)
        out = {
            "schema_version": "issue605_v1",
            "phase": "p5_fact_judged",
            "arm": arm,
            "seed": seed,
            "persona": p,
            "rows": [
                {
                    **{k: v for k, v in r.items() if k != "completion"},
                    "completion_head": r["completion"][:400],
                    "verdict": vd,
                }
                for r, vd in zip(rows, verdicts, strict=True)
            ],
            "summary": {
                "n_rows": len(rows),
                "stated_seven": n7,
                "leak_rate": n7 / max(1, len(rows) - n_err),
                "judge_failed_rows": n_err,
            },
            "metadata": _repro_meta(),
        }
        tmp = (judged_dir / f"{tag}.json").with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(judged_dir / f"{tag}.json")
        logger.info(
            "[phase=p5_judge] %s leak=%.3f (%d errors) — cell-persona judged",
            tag,
            out["summary"]["leak_rate"],
            n_err,
        )


# ---------------------------------------------------------------------------
# stage tf — teacher-forced taught-completion scoring, both sides
# ---------------------------------------------------------------------------
def _tf_score_with_engine(llm, tokenizer, pairs, lora_request=None) -> list[tuple[float, int]]:
    """(sum_logprob, n_tokens) per (prompt, completion) pair under an existing
    engine, optionally with a LoRA adapter applied.

    Lifted from ``issue444_bystander_logprob._score_pairs`` (the vetted #541
    metric: offset-mapping completion-span location, ground-truth-token-id
    log-probs, never argmax) and parameterized by engine + lora_request so
    trained-side scoring reuses one engine across cells."""
    from vllm import SamplingParams

    full_texts = [p + c for p, c in pairs]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(full_texts, params, lora_request=lora_request)
    results: list[tuple[float, int]] = []
    for (prompt, completion), out in zip(pairs, outputs, strict=True):
        full_text = prompt + completion
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        c_char_start = len(prompt)
        start_idx = None
        for tok_idx, (_cs, ce) in enumerate(offsets):
            if ce > c_char_start:
                start_idx = tok_idx
                break
        plogs = out.prompt_logprobs or []
        if start_idx is None or not plogs:
            results.append((float("nan"), 0))
            continue
        total, ntok, ok = 0.0, 0, True
        for idx in range(start_idx, len(full_ids)):
            if idx >= len(plogs):
                break
            lp_dict = plogs[idx]
            if lp_dict is None:
                continue
            entry = lp_dict.get(full_ids[idx])
            if entry is None:
                ok = False
                break
            total += entry.logprob
            ntok += 1
        results.append((total, ntok) if (ok and ntok > 0) else (float("nan"), ntok))
    return results


def stage_tf(out_dir: Path, cells, panel_path: Path, personas_subset, dry_run) -> None:
    """TF taught-completion scoring: trained per cell + base once per persona."""
    from issue444_bystander_logprob import _chat_prompt
    from transformers import AutoTokenizer

    rows = json.loads(TEACH_ROWS_PATH.read_text())["rows"]
    pool = _persona_pool()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tf_dir = out_dir / "tf"
    base_dir = out_dir / "tf_base"
    tf_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    personas_by_cell = {(a, s): _resolve_personas(panel_path, a, personas_subset) for a, s in cells}
    all_personas = sorted({p for ps in personas_by_cell.values() for p in ps})
    pending_base = [p for p in all_personas if not (base_dir / f"{p}.json").exists()]
    pending_cells = [
        (a, s, p)
        for (a, s), ps in personas_by_cell.items()
        for p in ps
        if not (tf_dir / f"{_cell_tag(a, s)}__{p}.json").exists()
    ]
    logger.info(
        "[phase=p5_tf] pending: %d base personas, %d (cell x persona) trained files",
        len(pending_base),
        len(pending_cells),
    )
    if dry_run:
        _ = _chat_prompt(tok, pool[all_personas[0]], rows[0]["question"])
        logger.info("[phase=p5_tf] dry-run: %d teach rows loaded; stopping before vLLM", len(rows))
        return
    if not pending_base and not pending_cells:
        return

    from issue532_predictor_stress import _build_vllm_engine
    from vllm.lora.request import LoRARequest

    adapters = _download_fact_adapters(cells)
    llm = _build_vllm_engine(max_seq_len=4096, enable_lora=True)

    def persona_pairs(p: str) -> list[tuple[str, str]]:
        return [(_chat_prompt(tok, pool[p], r["question"]), r["completion"]) for r in rows]

    for p in pending_base:
        scored = _tf_score_with_engine(llm, tok, persona_pairs(p))
        per_tok = [s / n for s, n in scored if n > 0 and not np.isnan(s)]
        payload = {
            "schema_version": "issue605_v1",
            "phase": "p5_fact_tf_base",
            "persona": p,
            "n_rows": len(rows),
            "per_row": [[float(s), int(n)] for s, n in scored],
            "summary": {
                "mean_logprob_per_tok": float(np.mean(per_tok)),
                "n_scored": len(per_tok),
            },
            "metadata": _repro_meta(),
        }
        tmp = (base_dir / f"{p}.json").with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(base_dir / f"{p}.json")
        logger.info("[phase=p5_tf] base %s scored (cell-side reuse)", p)

    for i, (arm, seed) in enumerate(cells):
        lora_req = LoRARequest(
            lora_name=f"tf_{_cell_tag(arm, seed)}",
            lora_int_id=100 + i,
            lora_path=adapters[(arm, seed)],
        )
        for a2, s2, p in pending_cells:
            if (a2, s2) != (arm, seed):
                continue
            scored = _tf_score_with_engine(llm, tok, persona_pairs(p), lora_request=lora_req)
            base = json.loads((base_dir / f"{p}.json").read_text())["per_row"]
            deltas = []
            for (ts, tn), (bs, bn) in zip(scored, base, strict=True):
                if tn > 0 and bn > 0 and not (np.isnan(ts) or np.isnan(bs)):
                    deltas.append(ts / tn - bs / bn)
            payload = {
                "schema_version": "issue605_v1",
                "phase": "p5_fact_tf_trained",
                "arm": arm,
                "seed": seed,
                "persona": p,
                "n_rows": len(rows),
                "per_row_trained": [[float(s), int(n)] for s, n in scored],
                "summary": {
                    "mean_delta_logprob_per_tok": float(np.mean(deltas)),
                    "frac_rows_positive_delta": float(np.mean([d > 0 for d in deltas])),
                    "n_scored": len(deltas),
                },
                "metadata": _repro_meta(),
            }
            tag = f"{_cell_tag(arm, seed)}__{p}"
            tmp = (tf_dir / f"{tag}.json").with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(tf_dir / f"{tag}.json")
            logger.info(
                "[phase=p5_tf] %s delta=%.3f nat/tok (cell-persona scored)",
                tag,
                payload["summary"]["mean_delta_logprob_per_tok"],
            )


# ---------------------------------------------------------------------------
# adapter-application smoke (plan section 7 gate 1, fact side)
# ---------------------------------------------------------------------------
def adapter_smoke(out_dir: Path) -> None:
    """marine_biologist seed-42 teacher-self TF delta on a 12-row subsample:
    > +0.1 nat/token AND positive on >= 70% of rows (#534 failure shape)."""
    from issue444_bystander_logprob import _chat_prompt
    from transformers import AutoTokenizer

    rows = json.loads(TEACH_ROWS_PATH.read_text())["rows"][:ADAPTER_SMOKE_N_ROWS]
    pool = _persona_pool()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    cells = [("marine_biologist", 42)]
    adapters = _download_fact_adapters(cells)

    from issue532_predictor_stress import _build_vllm_engine
    from vllm.lora.request import LoRARequest

    llm = _build_vllm_engine(max_seq_len=4096, enable_lora=True)
    pairs = [
        (_chat_prompt(tok, pool["marine_biologist"], r["question"]), r["completion"]) for r in rows
    ]
    base = _tf_score_with_engine(llm, tok, pairs)
    lora_req = LoRARequest(
        lora_name="smoke_mb42", lora_int_id=1, lora_path=adapters[("marine_biologist", 42)]
    )
    trained = _tf_score_with_engine(llm, tok, pairs, lora_request=lora_req)
    deltas = [
        ts / tn - bs / bn
        for (ts, tn), (bs, bn) in zip(trained, base, strict=True)
        if tn > 0 and bn > 0 and not (np.isnan(ts) or np.isnan(bs))
    ]
    mean_d = float(np.mean(deltas))
    pos_frac = float(np.mean([d > 0 for d in deltas]))
    out = out_dir / "adapter_smoke" / "fact_adapter_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": "issue605_v1",
                "phase": "p0_fact_adapter_smoke",
                "mean_delta_nat_per_tok": mean_d,
                "frac_rows_positive": pos_frac,
                "n_rows": len(deltas),
                "thresholds": {
                    "min_delta": ADAPTER_SMOKE_MIN_DELTA,
                    "min_pos_frac": ADAPTER_SMOKE_MIN_POS_FRAC,
                },
                "metadata": _repro_meta(),
            },
            indent=1,
        )
    )
    assert mean_d > ADAPTER_SMOKE_MIN_DELTA and pos_frac >= ADAPTER_SMOKE_MIN_POS_FRAC, (
        f"FACT ADAPTER-APPLICATION SMOKE FAIL: teacher-self TF delta {mean_d:.3f} nat/tok "
        f"(threshold > {ADAPTER_SMOKE_MIN_DELTA}), positive on {pos_frac:.0%} of rows "
        f"(threshold >= {ADAPTER_SMOKE_MIN_POS_FRAC:.0%}). Adapter not applied (#534 class) "
        "— infra fix, not a science verdict."
    )
    logger.info(
        "[phase=p0_adapter_smoke] PASS: mean delta %.3f nat/tok, %.0f%% rows positive",
        mean_d,
        100 * pos_frac,
    )


# ---------------------------------------------------------------------------
# stage upload + sentinel
# ---------------------------------------------------------------------------
def stage_upload(out_dir: Path) -> None:
    """ONE bulk fail-loud folder upload (raw completions policy)."""
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    url = _upload(
        local_path=out_dir,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_BUCKET}/fact",
        upload_as_file=False,
    )
    if not url:
        raise RuntimeError(
            f"fact per-cell upload to {DEFAULT_DATASET_REPO}/{HF_BUCKET}/fact FAILED "
            "(empty url from hub._upload) — do NOT terminate the pod before this lands."
        )
    logger.info("[phase=p6_upload] fact artifacts uploaded: %s", url)


def _write_results_sentinel(out_dir: Path, note: dict) -> Path:
    epoch = int(time.time())
    sentinel_dir = Path("/workspace/logs")
    if not sentinel_dir.exists():
        sentinel_dir = out_dir
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = sentinel_dir / f"issue-605-epm_results-{epoch}.json"
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "task_id": 605,
                "ts": epoch,
                "note": note,
            },
            indent=2,
        )
    )
    logger.info("results sentinel written: %s", path)
    return path


def _sentinel_note(out_dir: Path, cells, panel_path: Path, personas_subset) -> dict:
    judged = sorted((out_dir / "judged").glob("*.json")) if (out_dir / "judged").exists() else []
    tf = sorted((out_dir / "tf").glob("*.json")) if (out_dir / "tf").exists() else []
    leaks = [json.loads(f.read_text())["summary"]["leak_rate"] for f in judged]
    n_expected = sum(len(_resolve_personas(panel_path, a, personas_subset)) for a, _s in cells)
    return {
        "eval_numbers": {
            "n_cell_personas_judged": len(judged),
            "n_cell_personas_tf": len(tf),
            "n_expected": n_expected,
            "mean_leak_rate": float(np.mean(leaks)) if leaks else None,
        },
        "eval_paths": [str(out_dir)],
        "reproducibility_card": _repro_meta({"cells": [f"{a}/s{s}" for a, s in cells]}),
        "wandb_url": "wandb://exp605-matched-panels",
        "hf_hub_url": f"superkaiba1/explore-persona-space-data/{HF_BUCKET}/fact",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": _repro_meta().get("git_commit", "unknown"),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": GPU_HOURS_BUDGETED,
        "plan_deviations": [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Issue #605 Phase 5 fact-family trained-side eval (matched panels).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--arms", default="all")
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--personas-subset", type=int, default=None)
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--stage", choices=["gen", "judge", "tf", "upload", "all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--adapter-smoke", action="store_true")
    ap.add_argument("--write-sentinel", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-done-marker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.adapter_smoke:
        adapter_smoke(args.out_dir)
        return

    cells = _resolve_cells(args.arms, args.seeds)
    t0 = time.time()
    if args.stage == "all":
        for st in ("gen", "tf", "judge"):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--arms",
                args.arms,
                "--seeds",
                args.seeds,
                "--panel",
                str(args.panel),
                "--rows",
                str(args.rows),
                "--stage",
                st,
                "--out-dir",
                str(args.out_dir),
                "--no-done-marker",
            ]
            if args.personas_subset is not None:
                cmd += ["--personas-subset", str(args.personas_subset)]
            if args.dry_run:
                cmd.append("--dry-run")
            logger.info("[stage-dispatch] %s", " ".join(cmd))
            subprocess.run(cmd, env={**os.environ}, check=True)
        if not args.dry_run:
            stage_upload(args.out_dir)
    elif args.stage == "gen":
        stage_gen(args.out_dir, cells, args.panel, args.personas_subset, args.rows, args.dry_run)
    elif args.stage == "judge":
        stage_judge(args.out_dir, cells, args.panel, args.personas_subset, args.dry_run)
    elif args.stage == "tf":
        stage_tf(args.out_dir, cells, args.panel, args.personas_subset, args.dry_run)
    elif args.stage == "upload":
        stage_upload(args.out_dir)

    if args.write_sentinel and not args.dry_run:
        _write_results_sentinel(
            args.out_dir, _sentinel_note(args.out_dir, cells, args.panel, args.personas_subset)
        )
    if not args.no_done_marker:
        logger.info("[phase=done] eval_fact %s in %.0fs", args.stage, time.time() - t0)
    else:
        logger.info("eval_fact sub-stage %s complete in %.0fs", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
