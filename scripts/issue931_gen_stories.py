"""Issue #931 P1: Arm-B story generation (vLLM, chat template, no system prompt).

Sampling: T=1.0, top_p=0.95, seed 42, max_tokens 1024 (#825 Track-S parity).
Stories persist to <data-dir>/stories/stories_seed42.jsonl the moment
generation returns (BEFORE any downstream reduction), then upload to the HF
data repo raw_completions bucket (text path — unconditional) unless
--skip-upload.

--stub-gen writes deterministic story-shaped text WITHOUT loading any model —
the CPU-VM smoke path ONLY (the plan's section-4 canary exercises the real
vLLM path on the pod through the same dispatcher).

CLI:
  uv run python scripts/issue931_gen_stories.py \
      [--battery data/issue_931/pairs/prompt_battery.json] \
      [--data-dir data/issue_931] [--n-prompts 0] [--stub-gen] [--skip-upload]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# vLLM v1 fork-poisoning guard (#628): must be set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue931_gen_stories.py"
VLLM_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--battery", type=Path, default=Path("data/issue_931/pairs/prompt_battery.json")
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_931"))
    ap.add_argument("--n-prompts", type=int, default=0, help="0 = the whole battery")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--stub-gen", action="store_true", help="CPU smoke: deterministic stub text")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--upload-stage",
        type=str,
        default="generation",
        help="raw_completions/<stage>/ subdir for the story upload (the canary "
        "passes generation_canary so its stories persist without colliding "
        "with production's identically-named file)",
    )
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return ap.parse_args()


def _stub_story(prompt_row: dict, k: int) -> str:
    """Deterministic story-shaped text with attributable double-quote dialogue.

    SMOKE ONLY — exercises the attribution/extraction plumbing, never a
    measured artifact (the pod canary regenerates with real vLLM).
    """
    names = [("Maria", "Tomas"), ("Elena", "Jonas"), ("Priya", "Samuel")][k % 3]
    a, b = names
    setting = prompt_row.get("setting", "a quiet town")
    return (
        f"The evening settled over {setting} while {a} counted the day's takings. "
        f"{a} had lived here long enough to know every creaking board of the old office, "
        f"and tonight the boards were restless. Her cousin {b} arrived late, coat dripping. "
        f'"You should not have come alone tonight, not with the roads the way they are," '
        f"said {a}. "
        f'"I had no choice at all, because the ledger has gone missing from the archive '
        f'and nobody will say who took it," {b} replied. '
        f"{b} paced by the window, watching the street below, a tall figure worn thin by "
        f"worry and the long walk through the rain. "
        f'"Then we will find it together, before anyone else does, and we will start '
        f'with the archive keeper first thing tomorrow," said {a}. '
        f'"And if they already have it, and the pages are burned, what do we tell the '
        f'family then?" asked {b}. '
        f'"Then we tell the truth first, before the rumors get there ahead of us," '
        f"said {a}. "
        f"Outside, the rain kept its own ledger of the night."
    )


def generate_vllm(prompts: list[str], *, max_tokens: int, gpu_mem: float, seed: int) -> list[str]:
    """Chunked vLLM chat-template generation (T=1.0, top_p=0.95, seeded)."""
    from vllm import LLM, SamplingParams

    tokenizer = common.get_tokenizer()
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    llm = LLM(
        model=common.MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        seed=seed,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
    )
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=max_tokens, seed=seed)
    out: list[str] = []
    n_chunks = (len(rendered) + VLLM_CHUNK - 1) // VLLM_CHUNK
    for i in range(0, len(rendered), VLLM_CHUNK):
        chunk = rendered[i : i + VLLM_CHUNK]
        print(
            f"[vllm-chunk] gen chunk {i // VLLM_CHUNK + 1}/{n_chunks} ({len(chunk)} prompts)",
            flush=True,
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(r.outputs[0].text for r in res)
    # Teardown (vLLM v1 EngineCore reap; gotchas.md recipe) so the follow-on HF
    # capture phase in the same dispatcher run gets the whole GPU back. The
    # dispatcher additionally subprocess-isolates this phase (process exits).
    import contextlib

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
    print("[phase=p1_gen] Arm-B story generation")
    battery = json.loads(args.battery.read_text())["prompts"]
    if args.n_prompts:
        battery = battery[: args.n_prompts]
    print(f"[i931-p1] {len(battery)} prompts (stub_gen={args.stub_gen})")

    if args.stub_gen:
        stories = [_stub_story(row, k) for k, row in enumerate(battery)]
        provenance = "stub-smoke"
    else:
        stories = generate_vllm(
            [row["prompt"] for row in battery],
            max_tokens=args.max_tokens,
            gpu_mem=args.gpu_memory_utilization,
            seed=common.GEN_SEED,
        )
        provenance = "vllm"
    assert len(stories) == len(battery), (len(stories), len(battery))

    # Persist the generated TEXT immediately (before ANY reduction, #779).
    out_dir = args.data_dir / "stories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stories_seed{common.GEN_SEED}.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for row, story in zip(battery, stories, strict=True):
            f.write(
                json.dumps(
                    {
                        "prompt_id": row["prompt_id"],
                        "prompt": row["prompt"],
                        "story": story,
                        "provenance": provenance,
                        "gen_seed": common.GEN_SEED,
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "max_tokens": args.max_tokens,
                        "model": common.MODEL_ID,
                    }
                )
                + "\n"
            )
    tmp.replace(out_path)
    print(f"[i931-p1] wrote {out_path} ({len(stories)} stories)")
    common.write_json(
        out_dir / "gen_manifest.json",
        {
            "metadata": common.metadata(SCRIPT, common.GEN_SEED, len(stories)),
            "provenance": provenance,
            "stories_path": str(out_path),
            "stories_sha256": common.sha256_file(out_path),
        },
    )

    if not args.skip_upload:
        # Text artifact -> unconditional raw_completions upload (fail-loud).
        url = hub._upload(
            out_path,
            repo_id=common.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{common.HF_PREFIX}/raw_completions/{args.upload_stage}/{out_path.name}",
            upload_as_file=True,
        )
        assert url, "story upload returned no URL"
        print(f"[i931-p1] uploaded stories -> {url}")
    print("[i931-p1] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
