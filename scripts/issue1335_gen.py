"""Issue #1335: on-policy vLLM generation per (rung, model).

Q&A rungs (r0/r1/r2_op/r3/r4): one prompt per Track-S question rendered by
``issue1335_render_rungs.qa_render`` (plain text, both models); ONE sampled
completion per context (T=1.0/top_p=0.95/seed 42 — the #825/#1310 convention;
greedy deliberately avoided). Generation is CHUNKED (vLLM large-batch deadlock
prevention, EPM_VLLM_GREEDY_CHUNK_SIZE default 500) with per-chunk INFO logs.

Fiction rungs (r6_nofoil/r7_endpoint): the #1310 v3 prefill slot loop —
context prefilled up to the label cue, ONE line generated per slot (max 96
tokens, stop "\\n", per-slot seed GEN_SEED+slot), scene advanced with the
model's own completion (+ canned foil turn when foils). r7 reproduces the
#1310 recipe verbatim (byte-parity pinned by tests); r6 removes the foils.

Every record stores the EXACT vLLM prompt/completion token ids (the #1092
token-id-join rule: capture concatenates stored ids, never re-tokenizes) plus
``n_prefix_tokens`` for the v_P arm (offset-mapping count at gen time, with a
loud HF-vs-vLLM id-parity assert). Rollout text uploads to the HF data repo
per rung (non-LFS text path, unconditional) unless --skip-upload.

--stub-gen: deterministic completions with REAL-tokenizer ids (CPU smoke; the
SAME code path minus the engine).

CLI:
  uv run python scripts/issue1335_gen.py --rung r1_qa_oneline --model base \
      [--data-dir data/issue_1335] [--n-questions 0] [--n-scenarios 0] \
      [--slots 6] [--stub-gen] [--skip-upload] [--gpu-memory-utilization 0.85]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# vLLM v1 fork-poisoning guard (#628): set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402
import issue1310_common as c1310  # noqa: E402
import issue1310_prefill as i1310_prefill  # noqa: E402
import issue1335_render_rungs as r1335  # noqa: E402

SCRIPT = "scripts/issue1335_gen.py"
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

# HF raw_completions <stage> per rung (plan §10 output destinations).
UPLOAD_STAGE = {
    "r0_qa_full": "qa_full",
    "r1_qa_oneline": "qa_oneline",
    "r2_op": "renamed",
    "r3_persona": "persona",
    "r4_fictionframe": "fictionframe",
    "r6_nofoil": "nofoil",
    "r7_endpoint": "endpoint",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    gen_rungs = [s for s, c in r1335.RUNGS.items() if c["gen"] != "tf"]
    ap.add_argument("--rung", choices=gen_rungs, required=True)
    ap.add_argument("--model", choices=list(r1335.MODEL_KINDS), required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1335"))
    ap.add_argument("--n-questions", type=int, default=0, help="0 = full Track-S set (Q&A rungs)")
    ap.add_argument("--n-scenarios", type=int, default=0, help="0 = full battery (fiction rungs)")
    ap.add_argument("--slots", type=int, default=c1310.PREFILL_SLOTS)
    ap.add_argument("--stub-gen", action="store_true", help="CPU smoke: real tokenizer, no model")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return ap.parse_args()


def _stub_completion(key: str, idx: int, oneline: bool) -> str:
    """Deterministic completion (leading space, >= DIALOGUE_MIN_TOKENS). SMOKE ONLY."""
    line = (
        f" Considering point {idx} carefully, {key} lays out the reasoning plainly"
        f" and states a concrete answer for everyone present."
    )
    if oneline:
        return line
    return line + " Then a second sentence expands the answer with details and a worked example."


def _stub_generate(tokenizer, prompts: list[str], keys: list[str], oneline: bool) -> list[dict]:
    out = []
    for i, (prompt, key) in enumerate(zip(prompts, keys, strict=True)):
        comp = _stub_completion(key, i % 7, oneline)
        out.append(
            {
                "completion": comp,
                "prompt_token_ids": list(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                "completion_token_ids": list(
                    tokenizer(comp, add_special_tokens=False)["input_ids"]
                ),
            }
        )
    return out


def _vllm_generate(
    llm, prompts: list[str], *, max_tokens: int, stop: list[str], seed: int
) -> list[dict]:
    """Chunked sampled generation; returns per-prompt text + exact token ids."""
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=c1310.GEN_TEMPERATURE,
        top_p=c1310.GEN_TOP_P,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
        include_stop_str_in_output=False,
    )
    out: list[dict] = []
    n_chunks = (len(prompts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for ci in range(0, len(prompts), VLLM_CHUNK_SIZE):
        chunk = prompts[ci : ci + VLLM_CHUNK_SIZE]
        print(
            f"[i1335-gen] [vllm-chunk] chunk {ci // VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts)",
            flush=True,
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        for r in res:
            o = r.outputs[0]
            out.append(
                {
                    "completion": o.text,
                    "prompt_token_ids": list(r.prompt_token_ids),
                    "completion_token_ids": list(o.token_ids),
                }
            )
    return out


def _base_record(slug: str, model_kind: str, fp: dict) -> dict:
    return {
        "rung": slug,
        "model": r1335.MODEL_IDS[model_kind],
        "model_kind": model_kind,
        "gen_seed": c1310.GEN_SEED,
        "temperature": c1310.GEN_TEMPERATURE,
        "top_p": c1310.GEN_TOP_P,
        **fp,
    }


def gen_qa(args, tokenizer, fp: dict) -> list[dict]:
    """One-shot Q&A generation (r0/r1/r2_op/r3/r4)."""
    slug = args.rung
    cfg = r1335.RUNGS[slug]
    questions, qmeta = r1335.load_questions(args.data_dir, args.n_questions, tokenizer)
    battery = c1310.build_scenario_battery()
    prompts, prefixes, scen = [], [], []
    for j, row in enumerate(questions):
        scenario = battery[j % len(battery)] if cfg["header"] == "scene" else None
        prompt, prefix = r1335.qa_render(slug, row["question"], scenario)
        prompts.append(prompt)
        prefixes.append(prefix)
        scen.append(scenario)
    full = cfg["gen"] == "full"
    max_tokens = r1335.R0_MAX_TOKENS if full else r1335.ONELINE_MAX_TOKENS
    stop = r1335.R0_STOP if full else r1335.ONELINE_STOP
    print(f"[i1335-gen] {slug}/{args.model}: {len(prompts)} prompts (stub={args.stub_gen})")

    if args.stub_gen:
        gen = _stub_generate(tokenizer, prompts, [cfg["label"]] * len(prompts), not full)
    else:
        llm = i1310_prefill.build_engine(
            r1335.MODEL_IDS[args.model], args.gpu_memory_utilization, c1310.GEN_SEED
        )
        try:
            gen = _vllm_generate(
                llm, prompts, max_tokens=max_tokens, stop=stop, seed=c1310.GEN_SEED
            )
        finally:
            i1310_prefill.teardown_engine(llm)

    records = []
    for row, prompt, prefix, scenario, g in zip(
        questions, prompts, prefixes, scen, gen, strict=True
    ):
        n_prefix = r1335.count_prefix_tokens(tokenizer, prompt, prefix, g["prompt_token_ids"])
        rec = _base_record(slug, args.model, fp)
        rec.update(
            {
                "row_id": f"{slug}:q{row['q_idx']:05d}",
                "group_id": (
                    scenario["scenario_id"]
                    if cfg["group"] == "scenario"
                    else f"q{row['q_idx']:05d}"
                ),
                "persona": cfg["label"],
                "slot": 0,
                "question": row["question"],
                "prompt": prompt,
                "completion": g["completion"],
                "prompt_token_ids": g["prompt_token_ids"],
                "completion_token_ids": g["completion_token_ids"],
                "n_prompt_tokens": len(g["prompt_token_ids"]),
                "n_completion_tokens": len(g["completion_token_ids"]),
                "n_prefix_tokens": n_prefix,
                "provenance": "stub-smoke" if args.stub_gen else "vllm-onpolicy",
                "max_tokens": max_tokens,
                "stop": stop,
            }
        )
        if scenario is not None:
            rec.update(
                {
                    "scenario_id": scenario["scenario_id"],
                    "setting": scenario["setting"],
                    "situation": scenario["situation"],
                }
            )
        records.append(rec)
    records_meta = {"questions_meta": qmeta}
    return records, records_meta


def gen_fiction(args, tokenizer, fp: dict) -> tuple[list[dict], dict]:
    """#1310-style prefill slot loop (r7 verbatim; r6 = foils removed)."""
    slug = args.rung
    battery = c1310.build_scenario_battery()
    if args.n_scenarios:
        battery = battery[: args.n_scenarios]
    personas = list(c1310.PERSONA_LABELS)
    cells = [(sc, persona) for persona in personas for sc in battery]
    foils_by_sc = {
        sc["scenario_id"]: r1335.foils_for_rung(slug, sc["scenario_id"]) for sc in battery
    }
    bodies = {
        (sc["scenario_id"], p): r1335.fiction_body_slot0(
            sc["scenario_id"], foils_by_sc[sc["scenario_id"]]
        )
        for sc, p in cells
    }
    print(
        f"[i1335-gen] {slug}/{args.model}: {len(battery)} scenarios x {len(personas)} personas "
        f"x {args.slots} slots (stub={args.stub_gen})"
    )
    llm = None
    if not args.stub_gen:
        llm = i1310_prefill.build_engine(
            r1335.MODEL_IDS[args.model], args.gpu_memory_utilization, c1310.GEN_SEED
        )
    records: list[dict] = []
    try:
        for slot in range(args.slots):
            prompts, keys = [], []
            for sc, persona in cells:
                foils = foils_by_sc[sc["scenario_id"]]
                body = bodies[(sc["scenario_id"], persona)]
                prompts.append(
                    r1335.fiction_prefix(tokenizer, sc, persona, args.model, body, foils)
                )
                keys.append(persona)
            print(f"[i1335-gen] slot {slot + 1}/{args.slots} ({len(prompts)} prompts)", flush=True)
            if args.stub_gen:
                gen = _stub_generate(tokenizer, prompts, keys, oneline=True)
            else:
                gen = _vllm_generate(
                    llm,
                    prompts,
                    max_tokens=r1335.ONELINE_MAX_TOKENS,
                    stop=r1335.ONELINE_STOP,
                    seed=c1310.GEN_SEED + slot,
                )
            for (sc, persona), prompt, g in zip(cells, prompts, gen, strict=True):
                sc_id = sc["scenario_id"]
                foils = foils_by_sc[sc_id]
                prefix_text = r1335.fiction_prefix_text(prompt, sc_id, slot, foils, persona)
                n_prefix = r1335.count_prefix_tokens(
                    tokenizer, prompt, prefix_text, g["prompt_token_ids"]
                )
                rec = _base_record(slug, args.model, fp)
                rec.update(
                    {
                        "row_id": c1310.turn_row_id(f"{sc_id}:{persona}", slot),
                        "group_id": sc_id,
                        "scenario_id": sc_id,
                        "persona": persona,
                        "slot": slot,
                        "setting": sc["setting"],
                        "situation": sc["situation"],
                        "prompt": prompt,
                        "completion": g["completion"],
                        "prompt_token_ids": g["prompt_token_ids"],
                        "completion_token_ids": g["completion_token_ids"],
                        "n_prompt_tokens": len(g["prompt_token_ids"]),
                        "n_completion_tokens": len(g["completion_token_ids"]),
                        "n_prefix_tokens": n_prefix,
                        "provenance": "stub-smoke" if args.stub_gen else "vllm-prefill",
                        "max_tokens": r1335.ONELINE_MAX_TOKENS,
                        "stop": r1335.ONELINE_STOP,
                    }
                )
                records.append(rec)
                bodies[(sc_id, persona)] = r1335.fiction_advance_body(
                    bodies[(sc_id, persona)],
                    persona,
                    g["completion"],
                    sc_id,
                    slot + 1,
                    foils_by_sc[sc_id],
                )
    finally:
        if llm is not None:
            i1310_prefill.teardown_engine(llm)
    return records, {"n_scenarios": len(battery), "slots": args.slots, "personas": personas}


def main() -> int:
    args = parse_args()
    slug, model_kind = args.rung, args.model
    print(f"[phase=p1_gen_{slug}_{model_kind}] on-policy generation")
    fp = r1335.fingerprint(slug)
    tokenizer = common.get_tokenizer(r1335.MODEL_IDS[model_kind])
    if r1335.RUNGS[slug]["family"] == "qa":
        records, extra_meta = gen_qa(args, tokenizer, fp)
    else:
        records, extra_meta = gen_fiction(args, tokenizer, fp)

    out_path = r1335.gen_path(args.data_dir, slug, model_kind)
    r1335.write_gen_jsonl(out_path, records)
    n_short = sum(1 for r in records if r["n_completion_tokens"] < r1335.DIALOGUE_MIN_TOKENS)
    print(
        f"[i1335-gen] wrote {out_path} ({len(records)} records; {n_short} below "
        "dialogue-min => dropped at capture)"
    )
    c1310.write_json(
        out_path.with_name(f"{model_kind}_gen_manifest.json"),
        {
            "metadata": common.metadata(SCRIPT, c1310.GEN_SEED, len(records)),
            **fp,
            "model_kind": model_kind,
            "n_records": len(records),
            "n_below_dialogue_min": n_short,
            "jsonl_sha256": common.sha256_file(out_path),
            **extra_meta,
        },
    )
    if not args.skip_upload:
        stage = UPLOAD_STAGE[slug]
        url = hub._upload(
            out_path,
            repo_id=r1335.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{r1335.HF_PREFIX}/raw_completions/{stage}/{model_kind}_gen.jsonl",
            upload_as_file=True,
        )
        assert url, "rollout-text upload returned no URL"
        print(f"[i1335-gen] uploaded rollout text -> {url}")
    print(f"[i1335-gen] done ({slug}/{model_kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
