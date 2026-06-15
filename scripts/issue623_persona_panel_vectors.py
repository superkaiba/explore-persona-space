#!/usr/bin/env python3
"""Issue #623 phase 2 — extract persona centroids for the resolved panel.

A thin wrapper around ``scripts/extract_persona_vectors.py`` that feeds the
RESOLVED panel prompts (``panel_prompts.json``) rather than the 275-role default
loader, producing per-persona, per-layer centroids in one forward pass over the
shared 240Q extraction bank:

  - Method A (last-prompt-token, headline) -> data/persona_vectors/issue623/method_a/<persona>.pt
  - Method B (response-avg, robustness arm i, via --method AB) -> .../method_b/<persona>.pt

Both methods write per-persona (n_layers, hidden_dim) centroids plus an
``all_centroids.pt`` + ``metadata.json``. The persona VECTOR (centroid_i -
centroid_assistant) and the cosine matrix are computed OFF-POD in
``issue623_analyze.py``; this phase only emits the raw centroids.

``--personas`` / ``--layers`` / ``--n-questions`` subset the work (smoke = sweep
with a smaller cell list). The model load + hooks are reused verbatim from
``extract_persona_vectors.py`` (no modification to that file).

Usage:
  uv run python scripts/issue623_persona_panel_vectors.py \
      --panel-prompts data/persona_vectors/issue623/panel_prompts.json \
      --method AB --layers 7 14 21 27
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# vLLM V1 fork() poisoning guard (gotchas.md #628): main() touches the tokenizer
# before vllm.LLM(); spawn avoids the silent EngineCore death under fork.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import extract_persona_vectors as epv  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.persona_decomp_623 import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_LAYERS,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402


def load_panel_prompts(path: Path) -> dict[str, str]:
    """Load {persona: system_prompt} from a panel_prompts.json manifest."""
    manifest = json.loads(path.read_text())
    return {name: entry["prompt"] for name, entry in manifest["personas"].items()}


def _reap_vllm_workers() -> None:
    """Reap vLLM worker subprocesses that survive in-process teardown (gotchas.md).

    Best-effort: psutil may be absent and children may already be gone; the
    dispatcher's nvidia-smi check is the backstop, so a reap miss here is
    non-fatal (NOT a swallowed fault).
    """
    try:
        import psutil
    except ImportError:
        return

    me = psutil.Process()
    for child in me.children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    _, alive = psutil.wait_procs(me.children(recursive=True), timeout=5)
    for child in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #623 phase 2 — panel persona vectors.")
    parser.add_argument(
        "--panel-prompts",
        default="data/persona_vectors/issue623/panel_prompts.json",
        help="panel_prompts.json from phase 1 (relative to repo root).",
    )
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--method", default="AB", choices=["A", "B", "AB"])
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Number of extraction questions (default: all 240).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/persona_vectors/issue623",
        help="Base output dir (relative to repo root). method_a/ method_b/ written under it.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Method B generation cap.")
    parser.add_argument(
        "--personas",
        default=None,
        help="Comma-separated subset of resolved personas (smoke). Default: all in manifest.",
    )
    args = parser.parse_args()

    load_dotenv()

    panel_path = (
        PROJECT_ROOT / args.panel_prompts
        if not Path(args.panel_prompts).is_absolute()
        else Path(args.panel_prompts)
    )
    panel = load_panel_prompts(panel_path)

    if args.personas:
        wanted = [p.strip() for p in args.personas.split(",") if p.strip()]
        missing = [p for p in wanted if p not in panel]
        if missing:
            raise ValueError(f"--personas not in panel manifest {panel_path}: {missing}")
        panel = {p: panel[p] for p in wanted}

    # extract_persona_vectors expects role_prompts: {role: [system_prompt, ...]}.
    role_prompts = {persona: [sys_prompt] for persona, sys_prompt in panel.items()}
    questions = epv.load_extraction_questions(args.n_questions)
    layers = args.layers

    base_output = (
        PROJECT_ROOT / args.output_dir
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )

    print(
        f"[phase=vector_extract] {len(role_prompts)} personas, {len(questions)} questions, "
        f"layers={layers}, method={args.method}",
        flush=True,
    )

    do_a = "A" in args.method
    do_b = "B" in args.method

    model = None
    tokenizer = None

    if do_a:
        out_a = base_output / "method_a"
        out_a.mkdir(parents=True, exist_ok=True)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": device}
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        cent_a = epv.extract_method_a(
            model, tokenizer, role_prompts, questions, layers, n_prompts=1, output_dir=out_a
        )
        torch.save(cent_a, out_a / "all_centroids.pt")
        (out_a / "metadata.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "method": "last_input_token",
                    "layers": layers,
                    "n_questions": len(questions),
                    "n_roles": len(cent_a),
                    "roles": sorted(cent_a),
                    "panel_prompts": str(panel_path),
                },
                indent=2,
            )
        )
        print(f"[phase=vector_extract] method A: {len(cent_a)} centroids -> {out_a}", flush=True)
        # Unconditionally release the HF model after Method A, BEFORE any vLLM
        # init for Method B. Method A holds ~14 GB (Qwen-7B bf16); vLLM targets
        # 85% util (~67 GB on A100-80G) and OOMs at engine init when the HF
        # model is still resident (only ~64 GB free). Method B re-loads the HF
        # model after vLLM generation completes via the `if model is None`
        # block below. crash-dump: _crash_dumps/issue623_1781520725_vector_extract/.
        import gc

        del model
        if tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        model = None
        tokenizer = None

    if do_b:
        out_b = base_output / "method_b"
        out_b.mkdir(parents=True, exist_ok=True)

        # vLLM 0.11 `_run_engine` crashes with ZeroDivisionError when
        # TQDM_DISABLE=1 (it computes `total_in_toks / pbar.format_dict["elapsed"]`
        # and the disabled tqdm bar reports elapsed=0). The dispatcher exports
        # TQDM_DISABLE=1 at issue623_dispatch.sh:36 as the #607 GCE startup-script
        # bufio guard, and that propagates into this child process. Pop it for the
        # duration of the vLLM call; restore afterward so any downstream phase that
        # depends on it is unaffected (the dispatcher re-exports per phase regardless).
        # crash-dump: _crash_dumps/issue623_1781525741_vector_extract/.
        _prev_tqdm_disable = os.environ.pop("TQDM_DISABLE", None)
        try:
            responses = epv.generate_responses_vllm(
                args.model,
                role_prompts,
                questions,
                n_prompts=1,
                gpu_id=args.gpu_id,
                output_path=out_b / "generated_responses.json",
                max_new_tokens=args.max_new_tokens,
            )
        finally:
            if _prev_tqdm_disable is not None:
                os.environ["TQDM_DISABLE"] = _prev_tqdm_disable

        # Reap vLLM worker subprocesses BEFORE the HF reload below: the callee's
        # cleanup is `del llm + empty_cache` only and may leave workers holding
        # GPU memory that contends with the ~14 GB HF reload (reconciler v5;
        # gotchas.md vLLM teardown; sibling issue623_extract_sycophancy_vector.py).
        _reap_vllm_workers()
        torch.cuda.empty_cache()

        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            device = torch.device("cuda:0")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map={"": device}
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(args.model)

        cent_b = epv.extract_method_b(model, tokenizer, responses, layers, output_dir=out_b)
        torch.save(cent_b, out_b / "all_centroids.pt")
        (out_b / "metadata.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "method": "mean_response_token",
                    "layers": layers,
                    "n_questions": len(questions),
                    "n_roles": len(cent_b),
                    "roles": sorted(cent_b),
                    "panel_prompts": str(panel_path),
                },
                indent=2,
            )
        )
        print(f"[phase=vector_extract] method B: {len(cent_b)} centroids -> {out_b}", flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"[phase=vector_extract] done -> {base_output}", flush=True)


if __name__ == "__main__":
    main()
