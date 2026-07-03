#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #920 S2: generate the 2,400 set-B greedy completions (GPU, vLLM, own process).

Byte-matches the set-A recipe (#658 ``issue658_extract_base_store``): vLLM greedy
``SamplingParams(temperature=0.0, max_tokens=512)`` (= ``issue658_common.
V0_MAX_NEW_TOKENS``), the same ``apply_chat_template(..., add_generation_prompt=
True)`` prompt construction, persona always in the system role. ONE ``LLM``
engine, one continuous-batching pass in chunks of ≤500 prompts (the gotchas.md
large-batch deadlock prevention, per-chunk INFO logs). Per-context checkpoint +
resume (existing ``gen_b/<ctx>.json`` files are skipped at startup); one bulk
``upload_folder`` to ``issue920_summary_sweep/raw_completions/gen_b/``.

vLLM and HF capture NEVER share a process (S3 is a separate script — the
teardown gotcha); ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` is pinned at module top
(the #628 fork-poisoned EngineCore trap: main() touches the tokenizer before
``LLM()``).

Usage (production, inside issue920_dispatch.sh)::

    uv run python scripts/issue920_gen_completions_b.py

    # CPU smoke of the templating/chunking/checkpoint-resume logic (mock engine):
    uv run python scripts/issue920_gen_completions_b.py --smoke --mock-engine \\
        --n-ctx 2 --n-probes 2 --out-dir /tmp/i920_smoke_gen --no-upload
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # #628 fork trap

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue594_common import messages_for_instance, probes_hash  # noqa: E402
from issue920_common import (  # noqa: E402
    DEFAULT_MODEL,
    HF_DATA_REPO,
    I658_PREFIX,
    I920_GEN_B_PREFIX,
    PROBES_B_PATH,
    dump_json,
    load_battery,
    load_json,
    load_probes,
    reproducibility_metadata,
    resolve_hf_revision,
    write_sentinel,
)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue920_gen_b")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_NEW_TOKENS = 512  # = issue658_common.V0_MAX_NEW_TOKENS (set-A recipe pin; asserted below)
CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "480"))  # ≤500; 480 = 10 contexts
MAX_MODEL_LEN = 8192  # prompt (≤~2K) + 512 new with ample margin


def _assert_set_a_recipe_pin() -> None:
    """Assert the set-A generation pin (assumption 11) from the #658 constants."""
    from issue658_common import V0_MAX_NEW_TOKENS

    assert V0_MAX_NEW_TOKENS == MAX_NEW_TOKENS, (V0_MAX_NEW_TOKENS, MAX_NEW_TOKENS)


def _build_prompts(tokenizer, instances: list[dict], probes: list[str]) -> list[str]:
    """Templated prompts for (instance × probe) with the assistant-header assert.

    The last 3 tokens of every templated prompt must be the assistant header
    ``[<|im_start|>, 'assistant', '\\n']`` (ids [151644, 77091, 198]) — the same
    trailing-token contract the #658 gen + the S3 extractor rely on.
    """
    header = [151644, 77091, 198]
    prompts: list[str] = []
    for inst in instances:
        for q in probes:
            messages = messages_for_instance(inst, q)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tail = tokenizer(text, padding=False)["input_ids"][-3:]
            assert tail == header, (
                f"assistant-header tail mismatch for {inst['id']}: {tail} != {header}"
            )
            prompts.append(text)
    return prompts


def _generate_chunked(llm, prompts: list[str], max_new: int, mock: bool) -> list[str]:
    """One continuous-batching pass, chunks ≤CHUNK_SIZE, per-chunk INFO logs."""
    out: list[str] = []
    n_chunks = (len(prompts) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(prompts), CHUNK_SIZE):
        chunk = prompts[i : i + CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] gen_b chunk %d/%d (%d prompts)",
            i // CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
        )
        if mock:
            out.extend(f"MOCK completion {i + j}" for j in range(len(chunk)))
            continue
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=max_new)
        chunk_out = llm.generate(chunk, sp, use_tqdm=False)  # use_tqdm=False: #613
        out.extend(o.outputs[0].text for o in chunk_out)
    return out


def _upload(out_dir: Path, ctx_ids: list[str]) -> str:
    """ONE upload_folder commit + fresh-listing EXACT-set verify (fail loud)."""
    from huggingface_hub import HfApi, list_repo_files

    api = HfApi()
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=I920_GEN_B_PREFIX,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.json"],
        commit_message=f"issue #920: set-B raw completions ({len(ctx_ids)} contexts)",
    )
    remote = set(list_repo_files(HF_DATA_REPO, repo_type="dataset", revision="main"))
    expected = {f"{I920_GEN_B_PREFIX}/{c}.json" for c in ctx_ids}
    missing = expected - remote
    if missing:
        raise RuntimeError(
            f"gen_b upload verification FAILED: {len(missing)} missing (e.g. {sorted(missing)[:3]})"
        )
    logger.info(
        "gen_b verified on the Hub: %d contexts under %s/", len(expected), I920_GEN_B_PREFIX
    )
    return I920_GEN_B_PREFIX


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #920 S2: set-B greedy completions (vLLM)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--probes", default=str(PROBES_B_PATH))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "issue_920" / "gen_b"))
    ap.add_argument("--n-ctx", type=int, default=None, help="smoke: cap contexts")
    ap.add_argument("--n-probes", type=int, default=None, help="smoke: cap probes")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mock-engine", action="store_true", help="smoke: no vLLM, canned text")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _assert_set_a_recipe_pin()

    logger.info("[phase=setup] battery + probes-B")
    instances, _fam = load_battery()
    probe_rows = load_probes(Path(args.probes))
    probes = [p["text"] for p in probe_rows]
    pool_hash = probes_hash(probes)
    stored_hash = load_json(Path(args.probes))["meta"]["probe_pool_hash"]
    assert pool_hash == stored_hash, "probes-B pool hash drift vs its own meta"
    if args.n_ctx is not None:
        instances = instances[: args.n_ctx]
    if args.n_probes is not None:
        probes = probes[: args.n_probes]
    n_per_ctx = len(probes)

    # Resume: skip contexts whose per-context JSON already exists (spot preemption).
    todo = [i for i in instances if not (out_dir / f"{i['id']}.json").is_file()]
    logger.info(
        "contexts: %d total, %d already done (resume), %d to generate",
        len(instances),
        len(instances) - len(todo),
        len(todo),
    )

    # Recipe cross-check vs the #658 store manifest (assumption 11) + revision pin.
    hf_revision = None
    if not args.smoke:
        from huggingface_hub import hf_hub_download

        hf_revision = resolve_hf_revision()
        man = load_json(
            hf_hub_download(
                HF_DATA_REPO, f"{I658_PREFIX}/store/store_manifest.json", repo_type="dataset"
            )
        )
        man_tokens = man.get("v0_max_new_tokens") or man.get("max_new_tokens")
        if man_tokens is not None:
            assert int(man_tokens) == args.max_new_tokens, (man_tokens, args.max_new_tokens)
            logger.info("set-A manifest re-check: max_new_tokens=%s ✓", man_tokens)
        else:
            logger.warning(
                "set-A store manifest carries no max_new_tokens key — pin "
                "grounded on issue658_common.V0_MAX_NEW_TOKENS only"
            )

    if todo:
        logger.info("[phase=load_model] %s (tokenizer first — spawn pinned)", args.model)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = _build_prompts(tokenizer, todo, probes)
        assert len(prompts) == len(todo) * n_per_ctx

        llm = None
        if not args.mock_engine:
            from vllm import LLM

            logger.info("[phase=engine] ONE vLLM engine, chunks ≤%d", CHUNK_SIZE)
            llm = LLM(
                model=args.model,
                dtype="bfloat16",
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=MAX_MODEL_LEN,
            )
        completions = _generate_chunked(llm, prompts, args.max_new_tokens, args.mock_engine)
        assert len(completions) == len(prompts)

        # Per-context checkpoint the moment each context's block is complete.
        for ci, inst in enumerate(todo):
            block = completions[ci * n_per_ctx : (ci + 1) * n_per_ctx]
            dump_json(
                {
                    "context_id": inst["id"],
                    "probe_set": "B",
                    "probe_pool_hash": pool_hash,
                    "recipe": {
                        "sampling": "greedy",
                        "temperature": 0.0,
                        "max_new_tokens": args.max_new_tokens,
                        "chat_template": "apply_chat_template(add_generation_prompt=True)",
                        "engine": "vllm" if not args.mock_engine else "MOCK",
                    },
                    "hf_data_repo_revision_at_gen": hf_revision,
                    "completions": [
                        {"probe": q, "completion": c} for q, c in zip(probes, block, strict=True)
                    ],
                    "reproducibility": reproducibility_metadata(),
                },
                out_dir / f"{inst['id']}.json",
            )
            logger.info("[phase=gen] wrote %s (%d completions)", inst["id"], len(block))

    ctx_ids = [i["id"] for i in instances]
    empty = 0
    for c in ctx_ids:
        blob = load_json(out_dir / f"{c}.json")
        assert len(blob["completions"]) == n_per_ctx, (c, len(blob["completions"]))
        empty += sum(1 for r in blob["completions"] if not r["completion"].strip())
    logger.info(
        "gen_b complete: %d contexts × %d probes, %d empty completions",
        len(ctx_ids),
        n_per_ctx,
        empty,
    )

    hf_path = None
    if not args.no_upload and not args.smoke:
        logger.info("[phase=upload] gen_b bulk upload")
        hf_path = _upload(out_dir, ctx_ids)

    write_sentinel(
        "epm:progress",
        {
            "phase": "S2_gen_completions_b",
            "blocks_pipeline": False,
            "n_contexts": len(ctx_ids),
            "n_probes": n_per_ctx,
            "empty_completions": empty,
            "hf_path": hf_path,
            "elapsed_s": round(time.time() - t0, 1),
        },
        out_dir,
        slug_extra="gen-b",
    )
    # Post-upload phase-done marker (OUTSIDE out_dir so the gen_b bulk upload
    # never picks it up): the dispatcher's resume predicate keys on this, so a
    # crash at the upload step re-enters the phase on retry instead of skipping
    # it (same class as the post-K3 fit-done marker).
    dump_json(
        {
            "phase": "S2_gen_completions_b",
            "n_contexts": len(ctx_ids),
            "n_probes": n_per_ctx,
            "hf_path": hf_path,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir.parent / f"{out_dir.name}_done.json",
    )
    # NOT [phase=done] — reserved for the dispatcher's single terminal line
    # (pod-side-reporting rule; #545 false-done class).
    logger.info("[phase=gen_b_complete] S2 gen_b complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] gen_b crashed:\n%s", traceback.format_exc())
        raise
