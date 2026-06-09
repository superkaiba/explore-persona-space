"""Issue #527 Phase B eval — on-policy emission (vLLM) + shift-vector extraction (HF).

Plan §4 Step 5 (vLLM batched on-policy generation) + Step 6 (forward-only HF
L20 shift-vector extraction at the post-response slot).

Two modes (run as TWO separate subprocesses per CLAUDE.md gotcha — vLLM
in-process teardown does NOT reap worker subprocesses, so the next HF load
OOMs):

    --mode emission       — vLLM batched, ~20 prompts × 5 samples per
                            (persona × adapter). Writes
                            eval_results/issue_527/eval/<cell_slug>__emission.json.

    --mode shift_extract  — HF forward-only at L20 post-response slot, mean
                            over 20 EVAL_QUESTIONS per (persona × adapter).
                            Writes eval_results/issue_527/eval/<cell_slug>__shift.pt
                            + <cell_slug>__shift.json.

Per CLAUDE.md "Checkpoint per phase" — each (cell, mode) writes its own
file immediately so a downstream crash never costs earlier cells.

CLI:
    uv run python scripts/run_issue527_eval.py --mode emission --cell-slug <slug>
    uv run python scripts/run_issue527_eval.py --mode shift_extract --cell-slug <slug>
    uv run python scripts/run_issue527_eval.py --mode emission --all-cells
    uv run python scripts/run_issue527_eval.py --mode shift_extract --all-cells
"""

# ruff: noqa: RUF001, RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_527 import (
    BASE_MODEL,
    EVAL_MAX_NEW_TOKENS,
    EVAL_N_PROMPTS_PER_PERSONA,
    EVAL_N_SAMPLES_PER_PROMPT,
    HF_MODEL_REPO,
    MARKER_ID,
    MARKER_TEXT,
    NEGATIVE_PANEL_4,
    PERSONA_POOL_19,
)
from explore_persona_space.experiments.issue_527.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger("issue_527.eval")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _resolve_eval_panel(persona_bank: dict[str, str], pair_a: str, pair_b: str) -> list[str]:
    """Held-out eval panel: 19 bystanders + assistant + the 2 sources (dedup).

    The 19 includes the 2 sources by definition, so we just dedup.
    """
    panel = [*list(PERSONA_POOL_19), "assistant"]
    # Ensure both sources are present (they should be in PERSONA_POOL_19).
    for name in (pair_a, pair_b):
        if name not in panel:
            panel.append(name)
    # Dedup preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in panel:
        if n in seen:
            continue
        if n not in persona_bank:
            raise AssertionError(f"eval panel persona {n!r} not in persona_bank")
        seen.add(n)
        out.append(n)
    return out


def _load_all_cells(out_root: Path) -> list[dict]:
    """List every sweep cell that finished training."""
    sweep_dir = out_root / "sweep"
    if not sweep_dir.is_dir():
        raise FileNotFoundError(
            f"sweep dir missing at {sweep_dir}; run "
            "scripts/run_issue527_train.py --phase sweep first."
        )
    cells: list[dict] = []
    for p in sorted(sweep_dir.glob("*.json")):
        cells.append(json.loads(p.read_text()))
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Mode: emission (vLLM)
# ─────────────────────────────────────────────────────────────────────────────


def _run_emission_for_cell(
    *,
    cell: dict,
    persona_bank: dict[str, str],
    eval_panel: list[str],
    questions: list[str],
    out_path: Path,
    gpu_memory_utilization: float,
) -> None:
    """Run vLLM batched greedy + sampled generation per (persona × adapter).

    Per the marker-leakage rule, emission rate is a "free legibility anchor"
    — the headline DV is `log P(marker)` trained − base (the shift-extract
    step's payload). But we still record the rate for plan §6 DV4 (gate).
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    log.info(
        "[phase=emission] loading vLLM for cell=%s adapter=%s/%s",
        cell["cell_slug"],
        HF_MODEL_REPO,
        cell["hf_subfolder"],
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        download_dir=os.environ.get("HF_HOME", None),
    )

    # Build per-persona prompt sets.
    prompts: list[str] = []
    prompt_meta: list[tuple[str, int]] = []
    for persona in eval_panel:
        for qi, q in enumerate(questions):
            messages = [
                {"role": "system", "content": persona_bank[persona]},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            prompt_meta.append((persona, qi))

    # Download adapter from HF to a local dir.
    from huggingface_hub import snapshot_download
    from vllm.lora.request import LoRARequest

    local_lora_dir = snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=[f"{cell['hf_subfolder']}/*"],
    )
    adapter_local = Path(local_lora_dir) / cell["hf_subfolder"]
    lora_req = LoRARequest("issue_527", 1, str(adapter_local))

    sampling = SamplingParams(
        n=EVAL_N_SAMPLES_PER_PROMPT,
        temperature=0.0,
        max_tokens=EVAL_MAX_NEW_TOKENS,
        seed=0,
    )
    log.info(
        "Generating %d prompts × n=%d samples (max_new_tokens=%d)",
        len(prompts),
        EVAL_N_SAMPLES_PER_PROMPT,
        EVAL_MAX_NEW_TOKENS,
    )
    outputs = llm.generate(prompts, sampling, lora_request=lora_req, use_tqdm=False)

    # Per-persona aggregate: emission rate = fraction of completions
    # containing MARKER_TEXT (the canonical leakage-rule "free legibility
    # anchor").
    per_persona: dict[str, dict] = {
        p: {"completions": [], "emission_hits": 0, "n": 0} for p in eval_panel
    }
    for (persona, _qi), output in zip(prompt_meta, outputs, strict=True):
        for sample in output.outputs:
            text = sample.text
            per_persona[persona]["completions"].append(text)
            per_persona[persona]["n"] += 1
            if MARKER_TEXT in text:
                per_persona[persona]["emission_hits"] += 1

    # Compress per-persona: only store the FIRST sample per (persona, q) for
    # space — full completions blow the per-cell file up to ~20 MB.
    summary: dict[str, dict] = {}
    for persona, d in per_persona.items():
        rate = d["emission_hits"] / max(1, d["n"])
        summary[persona] = {
            "emission_rate_on_policy": rate,
            "n_samples": d["n"],
            "first_completion": d["completions"][0] if d["completions"] else "",
        }

    payload = {
        "schema_version": "issue_527_emission_v1",
        "cell_slug": cell["cell_slug"],
        "pair_id": cell["pair_id"],
        "arm": cell["arm"],
        "seed": cell["seed"],
        "hf_adapter": f"{HF_MODEL_REPO}/{cell['hf_subfolder']}",
        "eval_panel": eval_panel,
        "questions_used": questions,
        "n_samples_per_prompt": EVAL_N_SAMPLES_PER_PROMPT,
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "per_persona": summary,
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("[phase=emission] cell=%s wrote %s", cell["cell_slug"], out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: shift_extract (HF forward-only)
# ─────────────────────────────────────────────────────────────────────────────


def _run_shift_extract_for_cell(
    *,
    cell: dict,
    persona_bank: dict[str, str],
    eval_panel: list[str],
    r_persona: dict[str, dict[str, str]],
    eval_questions: list[str],
    out_dir: Path,
    device: str,
) -> None:
    """Forward-only L20 residual extraction + ΔG log-P(marker).

    Per CLAUDE.md gotcha: NEVER run this in the same process as vLLM —
    vLLM worker subprocesses survive teardown and re-grab GPU memory the
    moment HF Transformers loads weights. The pipeline.sh runs this as a
    separate subprocess.
    """
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.issue_527.shift_extract import (
        extract_per_context_shift,
    )

    log.info(
        "[phase=shift_extract] cell=%s — loading base + adapter on %s",
        cell["cell_slug"],
        device,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    trained = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Download + attach adapter.
    from huggingface_hub import snapshot_download

    local_lora_dir = snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=[f"{cell['hf_subfolder']}/*"],
    )
    adapter_local = Path(local_lora_dir) / cell["hf_subfolder"]
    trained = PeftModel.from_pretrained(trained, str(adapter_local)).eval()

    log.info("Extracting per-context shifts (n_contexts=%d)", len(eval_panel))
    contexts_payload: dict[str, dict] = {}
    shift_matrix: list[list[float]] = []
    for persona in eval_panel:
        cs = extract_per_context_shift(
            base_model=base,
            trained_model=trained,
            tokenizer=tokenizer,
            persona=persona,
            persona_prompt=persona_bank[persona],
            eval_questions=eval_questions,
            r_responses=r_persona[persona],
            device=device,
        )
        shift_matrix.append(cs.shift_vector.tolist())
        contexts_payload[persona] = {
            "n_prompts": cs.n_prompts,
            "delta_logp_marker": cs.delta_logp_marker,
            # Non-saturating mechanistic readout from the same forward pass
            # (marker-leakage-measurement.md "Report BOTH log-prob and logit").
            "delta_logit_marker": cs.delta_logit_marker,
            "emission_argmax_trained": cs.emission_argmax_trained,
            "emission_argmax_base": cs.emission_argmax_base,
        }

    # Persist both the JSON metadata and the dense shift tensor (.pt) so
    # the analyzer doesn't bloat the JSON.
    json_path = out_dir / f"{cell['cell_slug']}__shift.json"
    pt_path = out_dir / f"{cell['cell_slug']}__shift.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "issue_527_shift_v1",
        "cell_slug": cell["cell_slug"],
        "pair_id": cell["pair_id"],
        "arm": cell["arm"],
        "seed": cell["seed"],
        "hf_adapter": f"{HF_MODEL_REPO}/{cell['hf_subfolder']}",
        "eval_panel": eval_panel,
        "eval_questions": eval_questions,
        "marker_id": MARKER_ID,
        "marker_text": MARKER_TEXT,
        "contexts": contexts_payload,
        "shift_matrix_path": pt_path.name,
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    torch.save(np.asarray(shift_matrix, dtype=np.float32), pt_path)
    log.info(
        "[phase=shift_extract] cell=%s wrote %s + %s (matrix shape %dx%d)",
        cell["cell_slug"],
        json_path,
        pt_path,
        len(shift_matrix),
        len(shift_matrix[0]) if shift_matrix else 0,
    )

    # Free GPU before the next cell.
    del base
    del trained
    gc.collect()
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=["emission", "shift_extract"])
    ap.add_argument(
        "--cell-slug", default=None, help="Run one cell only (slug = <pair>__<arm>__seed<S>)."
    )
    ap.add_argument("--all-cells", action="store_true", help="Run every cell under sweep/.")
    ap.add_argument(
        "--out-root",
        default="eval_results/issue_527",
    )
    ap.add_argument(
        "--r-persona-dir",
        default="eval_results/issue_527/R_persona",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument(
        "--n-eval-questions",
        type=int,
        default=EVAL_N_PROMPTS_PER_PERSONA,
        help="N eval questions per persona (default 20 from EVAL_QUESTIONS).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a cell if its output JSON already exists.",
    )
    args = ap.parse_args(argv)

    if args.gpu_id != 0:
        # CLAUDE.md gotcha: train/sft.py clobbers CUDA_VISIBLE_DEVICES.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    out_root = Path(args.out_root)
    eval_out_dir = out_root / "eval"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)
    # R_persona for shift extraction.
    r_persona: dict[str, dict[str, str]] = {}
    if args.mode == "shift_extract":
        from explore_persona_space.experiments.issue_527.persona_registry import (
            load_persona_bank as _lpb,  # re-imported for re-use guard
        )

        _ = _lpb  # silence unused
        r_dir = Path(args.r_persona_dir)
        for p in sorted(r_dir.glob("*.json")):
            payload = json.loads(p.read_text())
            r_persona[payload["persona"]] = payload["responses"]

    eval_questions = list(EVAL_QUESTIONS[: args.n_eval_questions])

    if args.all_cells:
        cells = _load_all_cells(out_root)
    elif args.cell_slug:
        cell_path = out_root / "sweep" / f"{args.cell_slug}.json"
        if not cell_path.is_file():
            raise SystemExit(
                f"cell {args.cell_slug!r} not found at {cell_path}; run train --phase sweep first."
            )
        cells = [json.loads(cell_path.read_text())]
    else:
        raise SystemExit("Pass --cell-slug <slug> OR --all-cells.")

    # The negatives MUST be present in the eval panel — bystander leakage
    # is read on them post-hoc; assert that here for clarity.
    for neg in NEGATIVE_PANEL_4:
        if neg not in persona_bank:
            raise AssertionError(f"negative panel persona {neg!r} not in persona_bank")

    for cell in cells:
        pair_id = cell["pair_id"]
        pair_a, pair_b = pair_id.split("__")
        eval_panel = _resolve_eval_panel(persona_bank, pair_a, pair_b)
        if args.mode == "emission":
            out_path = eval_out_dir / f"{cell['cell_slug']}__emission.json"
            if args.skip_existing and out_path.exists():
                log.info("Skipping %s (already exists)", out_path)
                continue
            log.info("[phase=emission] cell=%s start", cell["cell_slug"])
            _run_emission_for_cell(
                cell=cell,
                persona_bank=persona_bank,
                eval_panel=eval_panel,
                questions=eval_questions,
                out_path=out_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        else:  # shift_extract
            json_path = eval_out_dir / f"{cell['cell_slug']}__shift.json"
            if args.skip_existing and json_path.exists():
                log.info("Skipping %s (already exists)", json_path)
                continue
            log.info("[phase=shift_extract] cell=%s start", cell["cell_slug"])
            _run_shift_extract_for_cell(
                cell=cell,
                persona_bank=persona_bank,
                eval_panel=eval_panel,
                r_persona=r_persona,
                eval_questions=eval_questions,
                out_dir=eval_out_dir,
                device=f"cuda:{args.gpu_id}",
            )

    log.info("[phase=done] eval mode=%s complete", args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
