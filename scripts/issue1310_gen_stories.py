"""Issue #1310: on-policy SCRIPT-scene generation, per persona x model (vLLM).

Each (scenario, target persona) generates ONE multi-character dialogue scene in
strict script format — one turn per line, `<LABEL>: <dialogue>` — pairing the
target persona with 1-2 named foils so there is genuine multi-speaker context,
and asking the target to speak many times (>= TURN_TARGET_MIN). The labeled line
format makes attribution a deterministic line-prefix parse (~100% recall, base +
instruct) and lets each labeled turn become its own (X, Y) point — the two fixes
for run 1 (base attribution 0.5% recall; instruct n~150 << 3584 dims).

Sampling: T=1.0, top_p=0.95, seed 42, max_tokens 1024 — NEVER greedy (base
greedy loops, #825 r7/8). Base (Qwen2.5-7B) continues a FEW-SHOT raw-text prime
directly (no chat template; the prime lives in the prefix and is never
attributed — base's OWN continuation lines are the on-policy target turns);
instruct (Qwen2.5-7B-Instruct) writes the scene through the chat template. Each
model generates its OWN on-policy scenes from the SHARED scenario battery under
each of the 4 fixed-label personas.

Scenes persist to <data-dir>/stories/<model>_stories_seed42.jsonl the moment
generation returns (BEFORE any downstream attribution/reduction, #779), then
upload to the HF data repo raw_completions bucket (text path — unconditional)
unless --skip-upload.

--stub-gen writes deterministic script-format text (attributable turns by the
fixed label) WITHOUT loading any model — the CPU-VM smoke path ONLY.

CLI:
  uv run python scripts/issue1310_gen_stories.py --model base \
      [--data-dir data/issue_1310] [--n-prompts 0] [--personas Marlowe,Pip]
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

SCRIPT = "scripts/issue1310_gen_stories.py"
VLLM_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=c1310.MODEL_KINDS, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1310"))
    ap.add_argument("--n-prompts", type=int, default=0, help="0 = all scenarios in the battery")
    ap.add_argument("--personas", type=str, default="", help="comma list; empty = all 4")
    ap.add_argument("--max-tokens", type=int, default=c1310.GEN_MAX_TOKENS)
    ap.add_argument("--stub-gen", action="store_true", help="CPU smoke: deterministic stub text")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--upload-stage", type=str, default="generation")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return ap.parse_args()


def _stub_story(persona_label: str, scenario: dict, k: int) -> str:
    """Deterministic SCRIPT-format scene with several target turns + foil turns.

    SMOKE ONLY — exercises the line-prefix attributor / per-turn pair / extract
    plumbing, never a measured artifact (the pod run regenerates with real vLLM).
    The story text is the on-policy completion (script body); for base the
    few-shot prime lives in the prefix, so the stub emits only the scene body.
    """
    foils = c1310.foils_for_scene(scenario["scenario_id"])
    f0 = foils[0]
    f1 = foils[1] if len(foils) > 1 else foils[0]
    setting = scenario.get("setting", "a quiet town")
    lines = [
        f"{f0}: The matter of what happened in {setting} cannot wait any longer.",
        f"{persona_label}: I have thought it over carefully, and here is what we do first.",
        f"{f1}: That is a reasonable place to begin, though it will not be easy for us.",
        f"{persona_label}: We start at the beginning, together, before the others arrive tonight.",
        f"{f0}: And if the whole plan falls apart the moment they walk through that door?",
        f"{persona_label}: Then we adapt, calmly, one careful step after another until it holds.",
        f"{f1}: I trust you more than I expected to when this whole affair began.",
        f"{persona_label}: Good, keep close, say little, and let me handle the difficult part.",
        f"{f0}: The night is long and no one out there is coming to help us now.",
        f"{persona_label}: Then we help ourselves and we finish what was started here tonight.",
    ]
    return "\n".join(lines) + "\n"


def generate_vllm(
    model_id: str,
    model_kind: str,
    prompts: list[str],
    *,
    max_tokens: int,
    gpu_mem: float,
    seed: int,
) -> list[str]:
    """Chunked vLLM sampled generation (T=1.0, top_p=0.95, seeded).

    Base: generate on the raw prompts directly. Instruct: render the chat
    template first (add_generation_prompt=True).
    """
    from vllm import LLM, SamplingParams

    if model_kind == "instruct":
        tokenizer = c1310.get_tokenizer(model_id)
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
    else:
        rendered = list(prompts)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        seed=seed,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") != "1",
    )
    sp = SamplingParams(
        temperature=c1310.GEN_TEMPERATURE,
        top_p=c1310.GEN_TOP_P,
        max_tokens=max_tokens,
        seed=seed,
    )
    out: list[str] = []
    n_chunks = (len(rendered) + VLLM_CHUNK - 1) // VLLM_CHUNK
    for i in range(0, len(rendered), VLLM_CHUNK):
        chunk = rendered[i : i + VLLM_CHUNK]
        print(
            f"[vllm-chunk] {model_kind} chunk {i // VLLM_CHUNK + 1}/{n_chunks} "
            f"({len(chunk)} prompts)",
            flush=True,
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(r.outputs[0].text for r in res)
    # Teardown (vLLM v1 EngineCore reap; gotchas.md recipe).
    with contextlib.suppress(AttributeError):
        llm.llm_engine.engine_core.shutdown()
    with contextlib.suppress(AttributeError):
        llm.llm_engine.model_executor.shutdown()
    import gc

    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return out


def main() -> int:
    args = parse_args()
    model_kind = args.model
    model_id = c1310.MODEL_IDS[model_kind]
    print(f"[phase=p1_gen_{model_kind}] story generation ({model_id})")

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

    # (scenario x persona) rows in a fixed deterministic order.
    rows = [
        {
            "scenario_id": sc["scenario_id"],
            "persona": persona,
            "row_id": f"{sc['scenario_id']}:{persona}",
            "prompt": c1310.render_prompt(sc, persona, model_kind),
            "setting": sc["setting"],
            "situation": sc["situation"],
        }
        for persona in personas
        for sc in battery
    ]
    print(
        f"[i1310-gen] {len(battery)} scenarios x {len(personas)} personas = "
        f"{len(rows)} prompts (stub_gen={args.stub_gen})"
    )

    if args.stub_gen:
        stories = [_stub_story(r["persona"], r, k) for k, r in enumerate(rows)]
        provenance = "stub-smoke"
    else:
        stories = generate_vllm(
            model_id,
            model_kind,
            [r["prompt"] for r in rows],
            max_tokens=args.max_tokens,
            gpu_mem=args.gpu_memory_utilization,
            seed=c1310.GEN_SEED,
        )
        provenance = "vllm"
    assert len(stories) == len(rows), (len(stories), len(rows))

    out_dir = args.data_dir / "stories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_kind}_stories_seed{c1310.GEN_SEED}.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r, story in zip(rows, stories, strict=True):
            f.write(
                json.dumps(
                    {
                        "scenario_id": r["scenario_id"],
                        "persona": r["persona"],
                        "row_id": r["row_id"],
                        "prompt": r["prompt"],
                        "story": story,
                        "provenance": provenance,
                        "model": model_id,
                        "model_kind": model_kind,
                        "gen_seed": c1310.GEN_SEED,
                        "temperature": c1310.GEN_TEMPERATURE,
                        "top_p": c1310.GEN_TOP_P,
                        "max_tokens": args.max_tokens,
                    }
                )
                + "\n"
            )
    tmp.replace(out_path)
    print(f"[i1310-gen] wrote {out_path} ({len(stories)} stories)")
    c1310.write_json(
        out_dir / f"{model_kind}_gen_manifest.json",
        {
            "metadata": c1310.metadata(SCRIPT, c1310.GEN_SEED, len(stories)),
            "model": model_id,
            "model_kind": model_kind,
            "provenance": provenance,
            "personas": personas,
            "n_scenarios": len(battery),
            "stories_path": str(out_path),
            "stories_sha256": c1310.sha256_file(out_path),
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
        assert url, "story upload returned no URL"
        print(f"[i1310-gen] uploaded stories -> {url}")
    print(f"[i1310-gen] done ({model_kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
