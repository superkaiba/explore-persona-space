# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, ′, ×, →, —) in docstrings/comments matching the project house style.
"""Slice-resolved on-policy marker log-p for task #466 (plan §4.2 script 4).

Eval-only on the Phase 0 retrained LoRA checkpoint. Drops the
22-checkpoint loop of ``eval_i456_onpolicy_emission.py`` — we only need
step-1600 — but keeps the Phase A (vLLM on-policy gen) → Phase B (HF
teacher-forced log p) structure verbatim.

Phase A — vLLM generation per (persona × slice × probe), n=8 samples,
temp=1.0, top_p=1.0, max_new_tokens=1536. Source-truncation guard at >10%
finish_reason=="length" (silent-zero guard from #260 / #456). Per-cell
emission rate (substring ※) computed and kept ONLY as the free legibility
anchor — NOT the headline.

Phase B — HF Transformers teacher-forced log p(※) at the END of each
context = render_prefix(persona, q) + strip_trailing_marker(R) + "\\n\\n"
via ``compute_marker_logprob``. Reported ``trained − base`` per cell where
``base`` is the SAME contexts re-scored with the LoRA disabled via PEFT's
``adapter.disable_adapter()`` context manager (PEFT 0.18.1; ``PeftModel.
unload()`` does NOT exist in this version).

Per-cell JSON written to::
    eval_results/issue_466/onpolicy_gen/{persona}_{slice}.json          (Phase A)
    eval_results/issue_466/onpolicy_endpos_logp/{persona}_{slice}.json  (Phase B)

The continuous ``delta = mean_logp_trained − mean_logp_base`` log-p IS the
headline DV; emission rate is the free anchor only.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from issue466_personas import (  # noqa: E402
    PERSONAS,
    SLICE_NONTRIGGER,
    SLICE_TRIGGER_A,
    SLICE_TRIGGER_B,
)

from explore_persona_space.eval.marker_logprob import compute_marker_logprob  # noqa: E402

logger = logging.getLogger("issue466_marker_logp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER = "※"
MARKER_ID = 63680  # bare ※ — #456 documented exception to the global default

# The 11 distinct (persona, slice) cells plan §10 enumerates. (S is scored
# on three slices; S′_A and Always_A only on non-trigger + trigger_A; same
# pattern for B.)
CELLS: list[tuple[str, str]] = [
    ("S", "nontrigger"),
    ("S", "trigger_A"),
    ("S", "trigger_B"),
    ("S_prime_A_spanish_restaurants", "nontrigger"),
    ("S_prime_A_spanish_restaurants", "trigger_A"),
    ("always_A_spanish", "nontrigger"),
    ("always_A_spanish", "trigger_A"),
    ("S_prime_B_caps_sports", "nontrigger"),
    ("S_prime_B_caps_sports", "trigger_B"),
    ("always_B_caps", "nontrigger"),
    ("always_B_caps", "trigger_B"),
]


def _slice_prompts(slice_name: str) -> list[str]:
    if slice_name == "trigger_A":
        return SLICE_TRIGGER_A
    if slice_name == "trigger_B":
        return SLICE_TRIGGER_B
    if slice_name == "nontrigger":
        return SLICE_NONTRIGGER
    raise ValueError(f"unknown slice: {slice_name!r}")


# ── Reproducibility metadata ───────────────────────────────────────────────


def _metadata(adapter_info: dict[str, Any]) -> dict[str, Any]:
    git_commit = "unknown"
    try:
        # epm-lint: subprocess-env-inherit -- git rev-parse needs no credential env
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            git_commit = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "script": "issue466_marker_logp",
        "git_commit": git_commit,
        "base_model": BASE_MODEL,
        "adapter": adapter_info,
        "ts_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# ── Chat-template helper (matches the ported eval rig render_prefix) ──────


def _render_prefix(tokenizer, persona_text: str, question: str) -> str:
    msgs = [
        {"role": "system", "content": persona_text},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _strip_trailing_marker(text: str) -> str:
    """Strip trailing marker + whitespace so we don't condition on the marker we're scoring."""
    stripped = text.rstrip()
    while stripped.endswith(MARKER):
        stripped = stripped[: -len(MARKER)].rstrip()
    return stripped


# ── vLLM teardown ──────────────────────────────────────────────────────────


def _kill_vllm_workers() -> None:
    """Reap vLLM worker subprocesses; fail loud if any GPU PID survives.

    Same logic as the ported #456 eval rig — duplicated so this dispatcher
    is standalone-runnable.
    """
    import psutil

    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        logger.info("destroy_* skipped (%s)", e)

    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()

    me = psutil.Process()
    children = me.children(recursive=True)
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    _gone, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    gc.collect()

    try:
        # epm-lint: subprocess-env-inherit -- nvidia-smi PID probe needs no credential env
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.info("nvidia-smi probe skipped (%s)", e)
        return
    my_pid = os.getpid()
    surviving: list[int] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != my_pid and psutil.pid_exists(pid):
            surviving.append(pid)
    if surviving:
        raise RuntimeError(f"vLLM workers still hold the GPU after teardown: PIDs={surviving}.")
    logger.info("vLLM workers reaped; no surviving GPU PIDs.")


# ── Adapter download from HF Hub ───────────────────────────────────────────


def _download_adapter(repo: str, subfolder: str, local_dir: Path) -> Path:
    """Download every file under ``subfolder`` from HF Hub into ``local_dir``.

    Per CLAUDE.md, ``hf_hub_download`` per-file is more robust than
    ``snapshot_download`` on large repos. The Phase 0 persist wrote to
    ``{subfolder}/<adapter_dir.name>``; the caller passes the joined
    path. Returns the local directory.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    local_dir.mkdir(parents=True, exist_ok=True)
    all_files = list_repo_files(repo_id=repo, repo_type="model", revision="main")
    prefix = subfolder.rstrip("/") + "/"
    targets = [f for f in all_files if f.startswith(prefix)]
    if not targets:
        raise RuntimeError(
            f"No files under {repo}/{subfolder!r} — adapter never landed? "
            f"(Phase 0 persist should have written adapter_model.safetensors here.)"
        )
    logger.info("Downloading %d adapter files from %s/%s ...", len(targets), repo, subfolder)
    for f in targets:
        # The HF helper writes to a flat cache + returns the path; we
        # copy/link it into our local target so PeftModel.from_pretrained
        # sees one tidy directory.
        cached = hf_hub_download(repo_id=repo, repo_type="model", filename=f, revision="main")
        rel = f[len(prefix) :]
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            dest.unlink()
        os.symlink(cached, dest)
    # Sanity: adapter_model.safetensors must be present.
    if not (local_dir / "adapter_model.safetensors").exists():
        raise RuntimeError(
            f"Downloaded {len(targets)} files but adapter_model.safetensors is "
            f"missing in {local_dir}"
        )
    logger.info("Adapter downloaded to %s", local_dir)
    return local_dir


# ── Phase A — vLLM on-policy generation ───────────────────────────────────


def phase_a_generate(
    *,
    adapter_local_dir: Path,
    cells: list[tuple[str, str]],
    n_samples: int,
    max_new_tokens: int,
    max_model_len: int,
    seed: int,
    out_dir: Path,
    smoke_probes: int | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """For each cell, generate n samples per probe via vLLM with the LoRA loaded.

    Returns ``{(persona, slice): {"prefixes": [...], "completions": [...]}}``
    in memory + writes one JSON per cell to ``out_dir``. Phase B reads the
    in-memory cache; the disk artifacts are the durable copy.

    Per-cell ordering of completions matches the per-cell prefixes ordering,
    flattened (prompt-major, sample-minor inside each prompt).
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    rendered: list[str] = []
    row_index: list[tuple[str, str, int]] = []  # (persona, slice, prompt_idx)
    per_cell_prompts: dict[tuple[str, str], list[str]] = {}
    for persona_name, slice_name in cells:
        probes = _slice_prompts(slice_name)
        if smoke_probes is not None:
            probes = probes[:smoke_probes]
        per_cell_prompts[(persona_name, slice_name)] = probes
        ptext = PERSONAS[persona_name]
        for p_idx, q in enumerate(probes):
            rendered.append(_render_prefix(tokenizer, ptext, q))
            row_index.append((persona_name, slice_name, p_idx))

    logger.info(
        "[phase_a_generate] %d prefixes × n=%d samples via vLLM (smoke_probes=%s)",
        len(rendered),
        n_samples,
        smoke_probes,
    )

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        seed=seed,
    )
    sampling = SamplingParams(
        n=n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    lora_request = LoRARequest(
        lora_name="issue466_step1600",
        lora_int_id=1600,
        lora_path=str(adapter_local_dir),
    )
    t0 = time.time()
    try:
        outputs = llm.generate(rendered, sampling, lora_request=lora_request)
    finally:
        del llm
        gc.collect()
    wall = time.time() - t0
    logger.info("[phase_a_generate] vLLM done in %.1fs", wall)
    _kill_vllm_workers()

    # Aggregate per cell, write per-cell JSON, return in-memory cache for Phase B.
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        cell_prompts = per_cell_prompts[cell]
        prefixes: list[str] = []
        completions: list[list[str]] = []
        for p_idx, q in enumerate(cell_prompts):
            prefixes.append(_render_prefix(tokenizer, PERSONAS[cell[0]], q))
            # find matching row
            matching = [
                i
                for i, (pn, sn, pi) in enumerate(row_index)
                if pn == cell[0] and sn == cell[1] and pi == p_idx
            ]
            assert len(matching) == 1, (cell, p_idx, matching)
            out = outputs[matching[0]]
            completions.append([s.text for s in out.outputs])
        # Per-cell counts + truncation.
        n_total = sum(len(c) for c in completions)
        n_with_marker = sum(1 for c in completions for text in c if MARKER in text)
        n_truncated = sum(
            1
            for matching_idx in [
                i for i, (pn, sn, _) in enumerate(row_index) if pn == cell[0] and sn == cell[1]
            ]
            for s in outputs[matching_idx].outputs
            if s.finish_reason == "length"
        )
        cell_payload: dict[str, Any] = {
            "persona": cell[0],
            "slice": cell[1],
            "prompts": cell_prompts,
            "prefixes": prefixes,
            "completions": completions,
            "n_total": n_total,
            "n_with_marker": n_with_marker,
            "emission_rate": (n_with_marker / n_total) if n_total else 0.0,
            "n_truncated": n_truncated,
            "truncation_frac": (n_truncated / n_total) if n_total else 0.0,
            "config": {
                "n_samples": n_samples,
                "max_new_tokens": max_new_tokens,
                "max_model_len": max_model_len,
                "seed": seed,
                "smoke_probes": smoke_probes,
            },
            "marker_token": MARKER,
            "marker_token_id": MARKER_ID,
            "wall_seconds_share": wall / max(1, len(cells)),  # rough — wall is per-batch
        }
        out_path = out_dir / f"{cell[0]}_{cell[1]}.json"
        with open(out_path, "w") as f:
            json.dump(cell_payload, f, indent=2)
        cache[cell] = cell_payload
        logger.info(
            "  cell %s × %s: emission_rate=%.3f trunc=%.3f → %s",
            cell[0],
            cell[1],
            cell_payload["emission_rate"],
            cell_payload["truncation_frac"],
            out_path.name,
        )

    # Source-truncation silent-zero guard — only on the plain-S cells.
    for cell in cells:
        if cell[0] != "S":
            continue
        if cache[cell]["truncation_frac"] > 0.10:
            raise RuntimeError(
                f"Source-truncation guard tripped: cell {cell} truncation_frac="
                f"{cache[cell]['truncation_frac']:.3f} > 0.10. Raise --max-new-tokens."
            )

    return cache


# ── Phase B — HF teacher-forced log p(※) trained − base ──────────────────


def phase_b_score(
    *,
    adapter_local_dir: Path,
    cache: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
    batch_size: int,
    device: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """For each cell, teacher-force log p(※) at end_of_answer; report trained − base.

    Loads ``Qwen-2.5-7B-Instruct + LoRA(step-1600)`` via ``PeftModel``.
    For each context = prefix + strip_trailing_marker(R) + "\\n\\n" the
    log-p is scored under the adapter, then re-scored with the adapter
    disabled via ``with adapter.disable_adapter():`` — both forward
    passes share the same context, so the difference cleanly isolates the
    training-induced shift.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    base.eval()
    adapter = PeftModel.from_pretrained(base, str(adapter_local_dir))
    adapter.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for cell, cell_a_payload in cache.items():
            t0 = time.time()
            contexts: list[str] = []
            for p_idx, prefix in enumerate(cell_a_payload["prefixes"]):
                for completion in cell_a_payload["completions"][p_idx]:
                    own_answer = _strip_trailing_marker(completion)
                    # Defensive: skip empty answers (they would assert in
                    # compute_marker_logprob's zero-token-context guard).
                    if not own_answer.strip():
                        continue
                    contexts.append(prefix + own_answer + "\n\n")
            if not contexts:
                logger.warning("cell %s × %s: no non-empty contexts; skipping", cell[0], cell[1])
                continue

            # Trained log-p.
            logp_trained = compute_marker_logprob(
                adapter,
                tokenizer,
                contexts=contexts,
                marker_text=MARKER,
                batch_size=batch_size,
                device=device,
            )
            # Base log-p — disable the adapter on the SAME wrapped model.
            with adapter.disable_adapter():
                logp_base = compute_marker_logprob(
                    adapter,
                    tokenizer,
                    contexts=contexts,
                    marker_text=MARKER,
                    batch_size=batch_size,
                    device=device,
                )

            mean_t = float(sum(logp_trained) / len(logp_trained))
            mean_b = float(sum(logp_base) / len(logp_base))
            payload = {
                "persona": cell[0],
                "slice": cell[1],
                "n_contexts": len(contexts),
                "mean_logp_trained": mean_t,
                "mean_logp_base": mean_b,
                "delta": mean_t - mean_b,
                "logp_trained_per_context": logp_trained,
                "logp_base_per_context": logp_base,
                "marker_token": MARKER,
                "marker_token_id": MARKER_ID,
                "wall_seconds": time.time() - t0,
            }
            out_path = out_dir / f"{cell[0]}_{cell[1]}.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            results[cell] = payload
            logger.info(
                "  cell %s × %s: logp_t=%.4f logp_b=%.4f delta=%+.4f n=%d (%.1fs)",
                cell[0],
                cell[1],
                mean_t,
                mean_b,
                mean_t - mean_b,
                len(contexts),
                payload["wall_seconds"],
            )
    finally:
        del adapter
        del base
        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    return results


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-repo",
        default="superkaiba1/explore-persona-space",
        help="HF Hub model repo holding the Phase 0 retrained adapter",
    )
    parser.add_argument(
        "--adapter-subfolder",
        default="issue466_i432_marker_se_9neg_zen_seed42_step1600",
        help="path_in_repo PREFIX (sub-leaf appended by trainer's persist hook)",
    )
    parser.add_argument(
        "--adapter-local-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "issue466_step1600",
        help="local directory to materialize the downloaded adapter into",
    )
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Phase B teacher-force batch size"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--smoke-probes",
        type=int,
        default=None,
        help="if set, only use the first N probes per slice (smoke run)",
    )
    parser.add_argument(
        "--skip-marker-rescore",
        action="store_true",
        help="smoke option — run Phase A only (no Phase B); used when no adapter is yet downloaded",
    )
    parser.add_argument(
        "--gen-out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_466" / "onpolicy_gen",
    )
    parser.add_argument(
        "--logp-out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_466" / "onpolicy_endpos_logp",
    )
    args = parser.parse_args()

    # Hard marker-id assert (R7).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
    assert marker_ids == [MARKER_ID], (
        f"MARKER guard FAILED: '{MARKER}' tokenizes to {marker_ids}, expected [{MARKER_ID}]"
    )
    logger.info("Marker token assert OK: ※ -> [%d]", MARKER_ID)
    del tokenizer

    # Find the adapter. The Phase 0 trainer writes to
    # ``{subfolder}/{adapter_dir.name}`` (see trainer.py:508). We need
    # `list_repo_files` to discover the trailing leaf — there may be one
    # checkpoint-N subdir or just the bare adapter dir.
    from huggingface_hub import list_repo_files

    all_files = list_repo_files(repo_id=args.adapter_repo, repo_type="model", revision="main")
    matching = [f for f in all_files if f.startswith(args.adapter_subfolder.rstrip("/") + "/")]
    if not matching:
        raise RuntimeError(
            f"No files in {args.adapter_repo} under prefix {args.adapter_subfolder!r}. "
            "Phase 0 persist didn't land; re-run Phase 0 or fix --adapter-subfolder."
        )
    # Find the leaf that contains adapter_model.safetensors.
    safetensors_files = [f for f in matching if f.endswith("adapter_model.safetensors")]
    if not safetensors_files:
        raise RuntimeError(
            f"No adapter_model.safetensors in {args.adapter_repo}/{args.adapter_subfolder} "
            f"(found {len(matching)} other files)."
        )
    if len(safetensors_files) > 1:
        raise RuntimeError(
            f"Multiple adapter_model.safetensors under {args.adapter_subfolder}: "
            f"{safetensors_files} — refusing to guess; pass an exact subfolder."
        )
    # Strip "adapter_model.safetensors" to get the leaf path.
    leaf = safetensors_files[0].rsplit("/", 1)[0]
    logger.info("Adapter leaf resolved to %s/%s", args.adapter_repo, leaf)
    adapter_local_dir = _download_adapter(args.adapter_repo, leaf, args.adapter_local_dir)

    # Sanity-load via PeftModel before any heavy compute (also exercises
    # `disable_adapter` so we fail fast if the PEFT version drift gotcha
    # bit us).
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    logger.info("Sanity-loading adapter via PeftModel...")
    sanity_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    sanity_peft = PeftModel.from_pretrained(sanity_base, str(adapter_local_dir))
    assert hasattr(sanity_peft, "disable_adapter"), (
        "PEFT version drift: PeftModel.disable_adapter context manager not available — "
        f"PEFT version: {__import__('peft').__version__}"
    )
    adapter_meta: dict[str, Any] = {
        "repo": args.adapter_repo,
        "subfolder": args.adapter_subfolder,
        "leaf": leaf,
        "local_dir": str(adapter_local_dir),
    }
    # Inspect adapter_config.json for the recipe-fidelity check (plan §6 A0d).
    adapter_config_path = adapter_local_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_meta["adapter_config"] = json.load(f)
    del sanity_peft
    del sanity_base
    gc.collect()
    logger.info("Adapter sanity load OK.")

    # Phase A — vLLM gen.
    gen_cache = phase_a_generate(
        adapter_local_dir=adapter_local_dir,
        cells=CELLS,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed,
        out_dir=args.gen_out_dir,
        smoke_probes=args.smoke_probes,
    )

    if args.skip_marker_rescore:
        logger.info("--skip-marker-rescore set; exiting after Phase A.")
        return 0

    # Phase B — HF teacher-forced log p(※) trained − base.
    score_results = phase_b_score(
        adapter_local_dir=adapter_local_dir,
        cache=gen_cache,
        out_dir=args.logp_out_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Roll-up summary written alongside the per-cell artifacts.
    summary = {
        "phase": "marker_logp_summary",
        "cells": [
            {
                "persona": cell[0],
                "slice": cell[1],
                "mean_logp_trained": payload["mean_logp_trained"],
                "mean_logp_base": payload["mean_logp_base"],
                "delta": payload["delta"],
                "emission_rate": gen_cache[cell]["emission_rate"],
                "truncation_frac": gen_cache[cell]["truncation_frac"],
                "n_contexts": payload["n_contexts"],
            }
            for cell, payload in score_results.items()
        ],
        "metadata": _metadata(adapter_meta),
    }
    summary_path = args.logp_out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s (%d cells)", summary_path, len(score_results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
