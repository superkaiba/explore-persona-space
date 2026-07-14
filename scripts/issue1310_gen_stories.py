"""Issue #1310: on-policy story generation, per persona x model (vLLM, sampled).

Sampling: T=1.0, top_p=0.95, seed 42, max_tokens 1024 — NEVER greedy (base
greedy loops on raw prose, #825 r7/8). Base (Qwen2.5-7B) generates on the
raw-prose story-opening prompt directly (no chat template); instruct
(Qwen2.5-7B-Instruct) generates through the chat template. Each model
generates its OWN on-policy stories from the SHARED scenario battery under
each of the 4 fixed-label personas.

Stories persist to <data-dir>/stories/<model>_stories_seed42.jsonl the moment
generation returns (BEFORE any downstream attribution/reduction, #779), then
upload to the HF data repo raw_completions bucket (text path — unconditional)
unless --skip-upload.

--stub-gen writes deterministic story-shaped text (attributable dialogue by
the fixed label) WITHOUT loading any model — the CPU-VM smoke path ONLY.

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
    """Deterministic story-shaped text with >=2 dialogue turns by the fixed label.

    SMOKE ONLY — exercises the attribution/pair/extract plumbing, never a
    measured artifact (the pod run regenerates with real vLLM).
    """
    setting = scenario.get("setting", "a quiet town")
    situation = scenario.get("situation", "something had gone wrong")
    foils = ("the young clerk", "an old acquaintance", "the nervous stranger")
    foil = foils[k % 3]
    return (
        f"The evening settled over {setting} while the matter of how {situation} "
        f"weighed on everyone present. {persona_label} had walked the length of "
        f"the room twice, thinking it through, and only then turned to {foil} who "
        f"had been waiting by the window all along. The lamplight was low. "
        f'"You should tell me exactly what happened here, and leave nothing out," '
        f"said {persona_label}. "
        f'"I would if I understood any of it myself," {foil} replied, wringing '
        f"both hands. "
        f'"Then we start at the beginning, together, before the others come '
        f'back," said {persona_label}, settling into the chair by the door. '
        f"Outside, the night kept its own counsel."
    )


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
