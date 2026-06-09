"""Issue #538 Phase B eval — on-policy emission (vLLM) + shift-vector extraction (HF).

Mechanical copy of ``scripts/run_issue527_eval.py`` with imports + namespace
strings switched to ``issue_538``. Reads issue_538 sweep cells, writes to
``eval_results/issue_538/eval/<cell_slug>__{emission,shift}.{json,pt}``.
The shift JSON now persists the ``marker_slot_stats`` block per persona
per plan §6 "Marker-slot storage contract" (raw logp_marker / z_marker /
z_eos / logZ / slot_index for BOTH trained and base sides; captured by the
issue_538 ``shift_extract.extract_per_context_shift``).

Plan §4 Step 6 (vLLM batched on-policy generation) + Step 7 (forward-only HF
L20 shift-vector extraction at the post-response slot).

Two modes (run as TWO separate subprocesses per CLAUDE.md gotcha — vLLM
in-process teardown does NOT reap worker subprocesses, so the next HF load
OOMs):

    --mode emission       — vLLM batched, ~20 prompts × 1 sample per
                            (persona × adapter). Writes
                            eval_results/issue_538/eval/<cell_slug>__emission.json.

    --mode shift_extract  — HF forward-only at L20 post-response slot, mean
                            over 20 EVAL_QUESTIONS per (persona × adapter).
                            Writes eval_results/issue_538/eval/<cell_slug>__shift.pt
                            + <cell_slug>__shift.json (including the new
                            marker_slot_stats block per persona per plan §6).

Per CLAUDE.md "Checkpoint per phase" — each (cell, mode) writes its own
file immediately so a downstream crash never costs earlier cells.

CLI:
    uv run python scripts/run_issue538_eval.py --mode emission --cell-slug <slug>
    uv run python scripts/run_issue538_eval.py --mode shift_extract --cell-slug <slug>
    uv run python scripts/run_issue538_eval.py --mode emission --all-cells
    uv run python scripts/run_issue538_eval.py --mode shift_extract --all-cells
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

from explore_persona_space.experiments.issue_538 import (
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
from explore_persona_space.experiments.issue_538.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger("issue_538.eval")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _resolve_eval_panel(persona_bank: dict[str, str], pair_a: str, pair_b: str) -> list[str]:
    """Held-out eval panel: 18 bystanders + assistant + the 2 sources (dedup).

    The 18 includes the 2 sources by definition (when they are in
    PERSONA_POOL_19), so we just dedup. Round-2 fix per code-review
    Critical-4: the bare default-assistant context is encoded as the
    literal ``"assistant"`` key (not the dropped ``"helpful_assistant"``);
    the SYSTEM-PROMPT-UNIQUENESS assert below pins the no-duplicates
    contract — any future re-introduction of a byte-identical persona
    fails LOUD here instead of silently biasing GD1/GD2.
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
    # System-prompt uniqueness — pin the contract that no two panel
    # personas resolve to byte-identical system prompts (would otherwise
    # add a phantom rank-1 direction to GD1's SVD and pin one GD2
    # singleton cosine at exactly 1.0). Diagnostic shows ALL collisions
    # so a future drift is fully visible from the traceback.
    prompts = {p: persona_bank[p] for p in out}
    rev: dict[str, list[str]] = {}
    for name, prompt in prompts.items():
        rev.setdefault(prompt, []).append(name)
    collisions = {prompt: names for prompt, names in rev.items() if len(names) > 1}
    if collisions:
        diag = "; ".join(
            f"prompt={prompt!r} collides on: {sorted(names)}"
            for prompt, names in collisions.items()
        )
        raise AssertionError(
            f"eval panel has byte-identical system prompts for distinct names — "
            f"would bias GD1/GD2. {diag}"
        )
    return out


def _load_all_cells(out_root: Path) -> list[dict]:
    """List every sweep cell that finished training."""
    sweep_dir = out_root / "sweep"
    if not sweep_dir.is_dir():
        raise FileNotFoundError(
            f"sweep dir missing at {sweep_dir}; run "
            "scripts/run_issue538_train.py --phase sweep first."
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

    # Prefer the on-disk adapter (cell["output_dir"]) over re-downloading from HF.
    # The training step writes adapter_config.json + adapter_model.safetensors
    # directly to output_dir before uploading to HF; using the local path
    # avoids racing the upload (a snapshot_download right after upload can pick
    # up a stale HF snapshot revision that doesn't yet have this cell's files).
    from vllm.lora.request import LoRARequest

    adapter_local = Path(cell["output_dir"])
    if not (adapter_local / "adapter_config.json").is_file():
        # Fallback: re-download from HF if the local adapter is gone (e.g. a
        # fresh pod that didn't run training).
        from huggingface_hub import snapshot_download

        local_lora_dir = snapshot_download(
            repo_id=HF_MODEL_REPO,
            allow_patterns=[f"{cell['hf_subfolder']}/*"],
        )
        adapter_local = Path(local_lora_dir) / cell["hf_subfolder"]
    lora_req = LoRARequest("issue_538", 1, str(adapter_local))

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
        "schema_version": "issue_538_emission_v1",
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

    from explore_persona_space.experiments.issue_538.shift_extract import (
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

    # Prefer the on-disk adapter (cell["output_dir"]) over re-downloading from HF
    # — same race-avoidance rationale as the emission site above.
    adapter_local = Path(cell["output_dir"])
    if not (adapter_local / "adapter_config.json").is_file():
        from huggingface_hub import snapshot_download

        local_lora_dir = snapshot_download(
            repo_id=HF_MODEL_REPO,
            allow_patterns=[f"{cell['hf_subfolder']}/*"],
        )
        adapter_local = Path(local_lora_dir) / cell["hf_subfolder"]
    trained = PeftModel.from_pretrained(trained, str(adapter_local)).eval()

    # Plan §6 Gauge assert (REVISE round 1) — before any logit readout, assert
    # the adapter's target_modules exclude lm_head/embed_tokens AND
    # modules_to_save is empty (or None). Required so Δz_marker remains
    # gauge-free across cells (LoRA on attn-only does NOT touch the
    # unembedding W_U). At rsLoRA r=16 attn-only this is satisfied by
    # construction; the assert exists so a future config drift cannot
    # silently make logit readouts gauge-dependent.
    _adapter_cfg_path = adapter_local / "adapter_config.json"
    if _adapter_cfg_path.is_file():
        _adapter_cfg = json.loads(_adapter_cfg_path.read_text())
        _tm = _adapter_cfg.get("target_modules", [])
        if isinstance(_tm, str):
            _tm = [_tm]
        _forbidden = {"lm_head", "embed_tokens"}
        _bad = [m for m in _tm if any(f in m for f in _forbidden)]
        if _bad:
            raise AssertionError(
                f"Gauge assert FAIL: adapter target_modules includes unembedding/"
                f"embedding layer ({_bad}). Δz_marker would be gauge-dependent. "
                f"Adapter dir: {adapter_local}"
            )
        _mts = _adapter_cfg.get("modules_to_save") or []
        if _mts:
            raise AssertionError(
                f"Gauge assert FAIL: adapter modules_to_save is non-empty "
                f"({_mts}); the marker-leakage rule requires modules_to_save "
                f"empty so the unembedding stays frozen. Adapter dir: {adapter_local}"
            )

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
        # Plan §6 Marker-slot storage contract (REVISE round 1, issue_538
        # NEW vs issue_527): persist RAW per-side floats from the same HF
        # forward pass — logp_marker / z_marker / z_eos / logZ /
        # slot_index — so the analyzer's three-space saturation
        # localizer (log-prob primary / logit + EOS-margin secondary /
        # probability sanity) can read distance-to-emission post-hoc.
        contexts_payload[persona] = {
            "n_prompts": cs.n_prompts,
            "delta_logp_marker": cs.delta_logp_marker,
            # Non-saturating mechanistic readout from the same forward pass
            # (marker-leakage-measurement.md "Report BOTH log-prob and logit").
            "delta_logit_marker": cs.delta_logit_marker,
            "emission_argmax_trained": cs.emission_argmax_trained,
            "emission_argmax_base": cs.emission_argmax_base,
            "marker_slot_stats": {
                "trained": {
                    "logp_marker": cs.marker_slot_stats_trained.logp_marker,
                    "z_marker": cs.marker_slot_stats_trained.z_marker,
                    "z_eos": cs.marker_slot_stats_trained.z_eos,
                    "logZ": cs.marker_slot_stats_trained.logZ,
                },
                "base": {
                    "logp_marker": cs.marker_slot_stats_base.logp_marker,
                    "z_marker": cs.marker_slot_stats_base.z_marker,
                    "z_eos": cs.marker_slot_stats_base.z_eos,
                    "logZ": cs.marker_slot_stats_base.logZ,
                },
                "slot_index": cs.slot_index_mean,
            },
        }

    # Persist both the JSON metadata and the dense shift tensor (.pt) so
    # the analyzer doesn't bloat the JSON.
    json_path = out_dir / f"{cell['cell_slug']}__shift.json"
    pt_path = out_dir / f"{cell['cell_slug']}__shift.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "issue_538_shift_v1",
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


def main(argv: list[str] | None = None) -> int:  # noqa: C901  # argparse wiring + R-coverage precondition guard
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
        # NEW WRITE namespace per plan §4 Outputs.
        default="eval_results/issue_538",
    )
    ap.add_argument(
        "--r-persona-dir",
        # INHERITED READ from #527 (R_persona/ inherited verbatim).
        default="eval_results/issue_527/R_persona",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument(
        "--n-eval-questions",
        type=int,
        default=EVAL_N_PROMPTS_PER_PERSONA,
        help=(
            "N eval questions per persona (default 20 from EVAL_QUESTIONS). "
            "EVAL_QUESTIONS is a fixed 20-element list — values above 20 "
            "are silently capped (warn-on-truncate below); values below "
            "20 reduce the slice taken."
        ),
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
        from explore_persona_space.experiments.issue_538.persona_registry import (
            load_persona_bank as _lpb,  # re-imported for re-use guard
        )

        _ = _lpb  # silence unused
        r_dir = Path(args.r_persona_dir)
        for p in sorted(r_dir.glob("*.json")):
            payload = json.loads(p.read_text())
            r_persona[payload["persona"]] = payload["responses"]

    if args.n_eval_questions > len(EVAL_QUESTIONS):
        log.warning(
            "--n-eval-questions=%d exceeds EVAL_QUESTIONS length (%d); silently "
            "capping to %d. To run on >20 questions, extend personas.py:EVAL_QUESTIONS.",
            args.n_eval_questions,
            len(EVAL_QUESTIONS),
            len(EVAL_QUESTIONS),
        )
    eval_questions = list(EVAL_QUESTIONS[: args.n_eval_questions])

    # Round-2 fix per code-review Critical-3: fail LOUD at second 1 of
    # shift_extract (BEFORE any per-cell vLLM/HF load) if R_persona doesn't
    # cover every eval question for every persona used downstream. The
    # in-loop ``q in r_responses`` raise is still there as defense-in-
    # depth (shift_extract.py), but this entry-point assert keeps the
    # crash close to the launch command, not 10 GPU-h into the sweep.
    if args.mode == "shift_extract":
        for persona_name, resp in r_persona.items():
            missing = [q for q in eval_questions if q not in resp]
            if missing:
                raise SystemExit(
                    f"R_persona[{persona_name!r}] missing {len(missing)} of "
                    f"{len(eval_questions)} eval questions. First missing: "
                    f"{missing[0]!r}. R is INHERITED from #527 byte-identically; "
                    f"if this fails, the R_persona dir drifted. See parent's "
                    f"scripts/run_issue527_generate_R.py and re-sync from HF."
                )

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
