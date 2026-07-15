"""Issue #1310 v3: on-policy PREFILL generation, per persona x model (vLLM).

Run-2 fix for the base arm. Run 2 free-generated whole script scenes then
PARSED them for `^<LABEL>:` dialogue; the base model will not reliably emit
parseable script dialogue, so the line-prefix attributor dropped 99.8% of base
turns (base arm ~empty). Prefill removes the parser: construct the scene context
UP TO AND INCLUDING the character's label cue (`Vex:`), let the model COMPLETE
that one turn (stop at the line break), and the dialogue span is known BY
CONSTRUCTION (the generated tokens). No attribution, no drop -> base n>0.

Prefill at PREFILL_SLOTS successive turns per scene: at each slot, prefix =
header + accumulated body + `<LABEL>:`; the model completes the turn; the scene
is advanced with the model's OWN completion + a SHARED canned foil turn, then the
label is re-prefilled. Each slot -> one (v_C, v_A) point at capture time. The
scenario battery + canned foils are shared across personas + models (matched
contexts for base-vs-instruct + the character-swap control); only the
character's OWN completions + its label vary per (persona, model).

Storage (`prefill/<model>_prefill_seed42.jsonl`) records, per (scenario, persona,
slot), the prompt + completion TEXT and — critically — the exact vLLM
``prompt_token_ids`` + ``completion_token_ids`` (#1092 BPE-seam rule: the capture
concatenates the stored ids, NEVER re-tokenizes the prefix|completion join). The
rollout text uploads to the HF data repo raw_completions bucket unless
--skip-upload (text path, unconditional).

--stub-gen writes deterministic completions using the REAL tokenizer (no model)
for the CPU-VM smoke; the pod run regenerates with real vLLM.

CLI:
  uv run python scripts/issue1310_prefill.py --model base \
      [--data-dir data/issue_1310] [--n-prompts 0] [--slots 6] [--personas Wren,Vex]
      [--stub-gen] [--skip-upload] [--gpu-memory-utilization 0.85]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

# vLLM v1 fork-poisoning guard (#628): set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1310_common as c1310  # noqa: E402

SCRIPT = "scripts/issue1310_prefill.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=c1310.MODEL_KINDS, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1310"))
    ap.add_argument("--n-prompts", type=int, default=0, help="0 = all scenarios in the battery")
    ap.add_argument("--slots", type=int, default=c1310.PREFILL_SLOTS)
    ap.add_argument("--personas", type=str, default="", help="comma list; empty = all 4")
    ap.add_argument("--max-tokens", type=int, default=c1310.SLOT_MAX_TOKENS)
    ap.add_argument("--stub-gen", action="store_true", help="CPU smoke: real tokenizer, no model")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--upload-stage", type=str, default="prefill")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return ap.parse_args()


def _instruct_prefix(tokenizer, header: str, body: str, label: str) -> str:
    """Instruct prompt: chat template (user=header, generation prompt) + assistant
    prefill (accumulated body + the label cue). vLLM continues the assistant turn."""
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": header}], tokenize=False, add_generation_prompt=True
    )
    return f"{templated}{body}{label}:"


def _base_prefix(header: str, body: str, label: str) -> str:
    return f"{header}{body}{label}:"


def build_prefix(tokenizer, scenario: dict, persona: str, model_kind: str, body: str) -> str:
    header = c1310.prefill_header(scenario, persona, model_kind)
    if model_kind == "instruct":
        return _instruct_prefix(tokenizer, header, body, persona)
    return _base_prefix(header, body, persona)


def _stub_completion(persona: str, slot: int) -> str:
    """Deterministic dialogue line (leading space, > DIALOGUE_MIN_TOKENS). SMOKE ONLY."""
    return (
        f" I have weighed this carefully, and here at turn {slot} is what "
        f"{persona} would say plainly to everyone still present in this room."
    )


def build_engine(model_id: str, gpu_mem: float, seed: int):
    from vllm import LLM

    return LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        seed=seed,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") != "1",
    )


def teardown_engine(llm) -> None:
    """vLLM v1 EngineCore reap (gotchas.md recipe)."""
    import gc

    import torch

    with contextlib.suppress(AttributeError):
        llm.llm_engine.engine_core.shutdown()
    with contextlib.suppress(AttributeError):
        llm.llm_engine.model_executor.shutdown()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_slot_vllm(llm, prompts: list[str], *, max_tokens: int, seed: int) -> list[dict]:
    """One batched sampled generation (T=1.0, top_p=0.95, stop at line break).

    Returns per prompt {completion, prompt_token_ids, completion_token_ids}, the
    EXACT ids vLLM processed + generated (the capture concatenates these, never
    re-tokenizes the join). include_stop_str_in_output=False -> the "\\n" stop is
    excluded, so the completion is the single character-turn line content.
    """
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=c1310.GEN_TEMPERATURE,
        top_p=c1310.GEN_TOP_P,
        max_tokens=max_tokens,
        seed=seed,
        stop=list(c1310.PREFILL_STOP),
        include_stop_str_in_output=False,
    )
    res = llm.generate(prompts, sp, use_tqdm=False)
    out = []
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


def generate_slot_stub(tokenizer, prompts: list[str], keys: list[tuple[str, int]]) -> list[dict]:
    """Deterministic stub completions with REAL-tokenizer ids (CPU smoke)."""
    out = []
    for prompt, (persona, slot) in zip(prompts, keys, strict=True):
        comp = _stub_completion(persona, slot)
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


def main() -> int:
    args = parse_args()
    model_kind = args.model
    model_id = c1310.MODEL_IDS[model_kind]
    print(f"[phase=p1_prefill_{model_kind}] on-policy prefill generation ({model_id})")

    battery = c1310.build_scenario_battery()
    if args.n_prompts:
        battery = battery[: args.n_prompts]
    personas = (
        [p.strip() for p in args.personas.split(",") if p.strip()]
        if args.personas
        else list(c1310.PERSONA_LABELS)
    )
    for p in personas:
        assert p in c1310.PERSONAS, f"unknown persona {p!r}"

    tokenizer = c1310.get_tokenizer(model_id)
    # (scenario, persona) cells in a fixed deterministic order; each carries its
    # accumulating scene body (canned foils + the model's own prior completions).
    cells = [(sc, persona) for persona in personas for sc in battery]
    bodies = {
        (sc["scenario_id"], persona): c1310.prefill_body_slot0(sc["scenario_id"])
        for sc, persona in cells
    }
    print(
        f"[i1310-prefill] {len(battery)} scenarios x {len(personas)} personas x "
        f"{args.slots} slots = {len(cells) * args.slots} prefill points "
        f"(stub_gen={args.stub_gen})"
    )

    llm = None
    if not args.stub_gen:
        llm = build_engine(model_id, args.gpu_memory_utilization, c1310.GEN_SEED)

    records: list[dict] = []
    provenance = "stub-smoke" if args.stub_gen else "vllm-prefill"
    try:
        for slot in range(args.slots):
            prompts, keys = [], []
            for sc, persona in cells:
                body = bodies[(sc["scenario_id"], persona)]
                prompts.append(build_prefix(tokenizer, sc, persona, model_kind, body))
                keys.append((persona, slot))
            print(
                f"[i1310-prefill] slot {slot + 1}/{args.slots} ({len(prompts)} prompts)", flush=True
            )
            if args.stub_gen:
                gen = generate_slot_stub(tokenizer, prompts, keys)
            else:
                gen = generate_slot_vllm(
                    llm, prompts, max_tokens=args.max_tokens, seed=c1310.GEN_SEED + slot
                )
            assert len(gen) == len(cells), (len(gen), len(cells))
            for (sc, persona), prefix, g in zip(cells, prompts, gen, strict=True):
                sc_id = sc["scenario_id"]
                scene_row_id = f"{sc_id}:{persona}"
                records.append(
                    {
                        "scenario_id": sc_id,
                        "persona": persona,
                        "slot": slot,
                        "row_id": c1310.turn_row_id(scene_row_id, slot),
                        "scene_row_id": scene_row_id,
                        "model": model_id,
                        "model_kind": model_kind,
                        "prompt": prefix,
                        "completion": g["completion"],
                        "prompt_token_ids": g["prompt_token_ids"],
                        "completion_token_ids": g["completion_token_ids"],
                        "n_prompt_tokens": len(g["prompt_token_ids"]),
                        "n_completion_tokens": len(g["completion_token_ids"]),
                        "setting": sc["setting"],
                        "situation": sc["situation"],
                        "provenance": provenance,
                        "gen_seed": c1310.GEN_SEED,
                        "temperature": c1310.GEN_TEMPERATURE,
                        "top_p": c1310.GEN_TOP_P,
                    }
                )
                bodies[(sc_id, persona)] = c1310.prefill_advance_body(
                    bodies[(sc_id, persona)], persona, g["completion"], sc_id, slot + 1
                )
    finally:
        if llm is not None:
            teardown_engine(llm)

    out_dir = args.data_dir / "prefill"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_kind}_prefill_seed{c1310.GEN_SEED}.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out_path)
    n_empty = sum(1 for r in records if r["n_completion_tokens"] < c1310.DIALOGUE_MIN_TOKENS)
    print(
        f"[i1310-prefill] wrote {out_path} ({len(records)} records; "
        f"{n_empty} below dialogue-min => dropped at capture)"
    )
    c1310.write_json(
        out_dir / f"{model_kind}_prefill_manifest.json",
        {
            "metadata": c1310.metadata(SCRIPT, c1310.GEN_SEED, len(records)),
            "model": model_id,
            "model_kind": model_kind,
            "provenance": provenance,
            "personas": personas,
            "n_scenarios": len(battery),
            "slots": args.slots,
            "n_records": len(records),
            "n_below_dialogue_min": n_empty,
            "prefill_path": str(out_path),
            "prefill_sha256": c1310.sha256_file(out_path),
        },
    )

    if not args.skip_upload:
        url = hub._upload(
            out_path,
            repo_id=c1310.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=(f"{c1310.HF_PREFIX}/raw_completions/{args.upload_stage}/{out_path.name}"),
            upload_as_file=True,
        )
        assert url, "prefill upload returned no URL"
        print(f"[i1310-prefill] uploaded rollout text -> {url}")
    print(f"[i1310-prefill] done ({model_kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
