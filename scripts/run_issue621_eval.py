"""Issue #621 eval — on-policy emission (vLLM) + shift-vector extraction (HF).

Forked from ``scripts/run_issue538_eval.py`` (pinned ``e6b195f81``) with the
issue-621 cell schema (placement-arm × singleton-source × seed), full
raw-completion persistence (upload-policy: per-cell
``raw_generations/<cell_slug>/raw_completions.json`` consumed by
``upload_raw_completions_to_data_repo``), and the per-question Δlog P /
Δ(z_marker − z_eos) arrays the §14 duty-8 variance precondition needs.

Two modes (run as TWO separate subprocesses per CLAUDE.md gotcha — vLLM
in-process teardown does NOT reap worker subprocesses, so the next HF load
OOMs):

    --mode emission       — vLLM batched greedy, 20 prompts × 1 sample per
                            (persona × adapter). Writes
                            eval_results/issue_621/eval/<slug>__emission.json
                            + raw_generations/<slug>/raw_completions.json.

    --mode shift_extract  — HF forward-only at L20 post-response slot, mean
                            over 20 EVAL_QUESTIONS per (persona × adapter).
                            Writes eval_results/issue_621/eval/<slug>__shift.pt
                            + <slug>__shift.json (four-float marker_slot_stats
                            per persona per side + per-question deltas).

Cells are enumerated from the TRAIN dispatcher's cell JSONs under BOTH
``<out-root>/anchor_smoke/`` (the smoke cell IS production cell
r1_read__florist__seed42) and ``<out-root>/sweep/`` — so a smoke-subset run
flows through this phase unchanged (smoke/sweep parity).

Per CLAUDE.md "Checkpoint per phase" — each (cell, mode) writes its own
file immediately so a downstream crash never costs earlier cells.

CLI:
    uv run python scripts/run_issue621_eval.py --mode emission --all-cells
    uv run python scripts/run_issue621_eval.py --mode shift_extract --all-cells
    uv run python scripts/run_issue621_eval.py --mode emission --cell-slug r1_read__florist__seed42
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

from explore_persona_space.experiments.issue_621 import (
    BASE_MODEL,
    EVAL_MAX_NEW_TOKENS,
    EVAL_N_PROMPTS_PER_PERSONA,
    EVAL_N_SAMPLES_PER_PROMPT,
    HF_MODEL_REPO,
    HF_TRAIN_MIX_READ_REVISION,
    MARKER_ID,
    MARKER_TEXT,
    PERSONA_POOL_19,
    UNIFIED_NEGATIVE_PANEL,
)
from explore_persona_space.experiments.issue_621.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger("issue_621.eval")

PARENT_PIN_SHA = "e6b195f81"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _resolve_eval_panel(persona_bank: dict[str, str], source: str) -> list[str]:
    """Held-out eval panel: the 18-persona pool + assistant (+ source, dedup).

    The source is in PERSONA_POOL_19 for every #621 cell, so the panel is
    the same 19 contexts for all cells; the explicit append is
    defense-in-depth. System-prompt uniqueness is pinned (byte-identical
    duplicate contexts would bias the leakage panel reads).
    """
    panel = [*list(PERSONA_POOL_19), "assistant"]
    if source not in panel:
        panel.append(source)
    seen: set[str] = set()
    out: list[str] = []
    for n in panel:
        if n in seen:
            continue
        if n not in persona_bank:
            raise AssertionError(f"eval panel persona {n!r} not in persona_bank")
        seen.add(n)
        out.append(n)
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
            f"eval panel has byte-identical system prompts for distinct names: {diag}"
        )
    return out


def _load_all_cells(out_root: Path) -> list[dict]:
    """Enumerate every trained cell from anchor_smoke/ + sweep/ (dedup by slug).

    The smoke cell (r1_read__florist__seed42) is a PRODUCTION cell trained
    during the smoke phase — including anchor_smoke/ here is what makes the
    smoke subset flow through eval unchanged (smoke/sweep parity).
    """
    cells: dict[str, dict] = {}
    found_any_dir = False
    for sub in ("anchor_smoke", "sweep"):
        d = out_root / sub
        if not d.is_dir():
            continue
        found_any_dir = True
        for p in sorted(d.glob("*.json")):
            if p.name == "summary.json":
                continue
            payload = json.loads(p.read_text())
            if "cell_slug" not in payload:
                continue
            cells[payload["cell_slug"]] = payload
    if not found_any_dir:
        raise FileNotFoundError(
            f"neither {out_root}/anchor_smoke nor {out_root}/sweep exists; run "
            "scripts/run_issue621_train.py first."
        )
    if not cells:
        raise FileNotFoundError(f"no trained cell JSONs under {out_root}/{{anchor_smoke,sweep}}")
    return [cells[k] for k in sorted(cells)]


def _resolve_adapter_local(cell: dict) -> Path:
    """Prefer the on-disk adapter; fall back to the HF snapshot.

    Using the local path avoids racing the post-train upload (a
    snapshot_download right after upload can pick a stale revision).
    """
    adapter_local = Path(cell["output_dir"])
    if not (adapter_local / "adapter_config.json").is_file():
        from huggingface_hub import snapshot_download

        local_lora_dir = snapshot_download(
            repo_id=HF_MODEL_REPO,
            allow_patterns=[f"{cell['hf_subfolder']}/*"],
        )
        adapter_local = Path(local_lora_dir) / cell["hf_subfolder"]
    return adapter_local


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
    raw_dir: Path,
    gpu_memory_utilization: float,
) -> None:
    """vLLM batched greedy generation per (persona × adapter).

    Emission rate is the free legibility anchor (the headline DV is the
    shift-extract log P read). ALL completions persist to
    ``raw_dir/raw_completions.json`` per upload policy.
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
        # ≥ longest prompt + EVAL_MAX_NEW_TOKENS headroom (gotcha:
        # max_model_len must track max_new_tokens; prompts here are ~200
        # tokens + 2048 new ⇒ 4096 holds).
        max_model_len=4096,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        download_dir=os.environ.get("HF_HOME", None),
    )

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

    from vllm.lora.request import LoRARequest

    adapter_local = _resolve_adapter_local(cell)
    lora_req = LoRARequest("issue_621", 1, str(adapter_local))

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

    per_persona: dict[str, dict] = {
        p: {"completions": [], "questions": [], "emission_hits": 0, "n": 0} for p in eval_panel
    }
    for (persona, qi), output in zip(prompt_meta, outputs, strict=True):
        for sample in output.outputs:
            text = sample.text
            per_persona[persona]["completions"].append(text)
            per_persona[persona]["questions"].append(questions[qi])
            per_persona[persona]["n"] += 1
            if MARKER_TEXT in text:
                per_persona[persona]["emission_hits"] += 1

    # FULL raw completions → raw_generations/<slug>/raw_completions.json
    # (the exact filename upload_raw_completions_to_data_repo rglobs).
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_payload = {
        "schema_version": "issue_621_raw_completions_v1",
        "cell_slug": cell["cell_slug"],
        "hf_adapter": f"{HF_MODEL_REPO}/{cell['hf_subfolder']}",
        "sampling": {"temperature": 0.0, "n": EVAL_N_SAMPLES_PER_PROMPT, "seed": 0},
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "per_persona": {
            p: {"questions": d["questions"], "completions": d["completions"]}
            for p, d in per_persona.items()
        },
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    (raw_dir / "raw_completions.json").write_text(
        json.dumps(raw_payload, indent=1, ensure_ascii=False)
    )

    summary: dict[str, dict] = {}
    for persona, d in per_persona.items():
        rate = d["emission_hits"] / max(1, d["n"])
        summary[persona] = {
            "emission_rate_on_policy": rate,
            "n_samples": d["n"],
            "first_completion": d["completions"][0] if d["completions"] else "",
        }

    payload = {
        "schema_version": "issue_621_emission_v1",
        "cell_slug": cell["cell_slug"],
        "arm": cell["arm"],
        "source": cell["source"],
        "seed": cell["seed"],
        "hf_adapter": f"{HF_MODEL_REPO}/{cell['hf_subfolder']}",
        "eval_panel": eval_panel,
        "questions_used": questions,
        "n_samples_per_prompt": EVAL_N_SAMPLES_PER_PROMPT,
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "per_persona": summary,
        "raw_completions_rel": str(raw_dir / "raw_completions.json"),
        "fork_sha": _git_commit(),
        "parent_pin_sha": PARENT_PIN_SHA,
        "hf_train_mix_read_revision": HF_TRAIN_MIX_READ_REVISION,
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
    """Forward-only L20 residual extraction + four-float slot reads.

    Per CLAUDE.md gotcha: NEVER run this in the same process as vLLM —
    the pipeline runs this mode as a separate subprocess.
    """
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.issue_621.shift_extract import (
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

    adapter_local = _resolve_adapter_local(cell)
    trained = PeftModel.from_pretrained(trained, str(adapter_local)).eval()

    # Gauge assert (marker-leakage rule) — before any logit readout, the
    # adapter's target_modules must exclude lm_head/embed_tokens AND
    # modules_to_save must be empty, so Δz_marker (and the analyzer's W_U
    # readouts) stay gauge-free.
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
                f"embedding layer ({_bad}). Adapter dir: {adapter_local}"
            )
        _mts = _adapter_cfg.get("modules_to_save") or []
        if _mts:
            raise AssertionError(
                f"Gauge assert FAIL: adapter modules_to_save is non-empty ({_mts}). "
                f"Adapter dir: {adapter_local}"
            )
        # Rank sanity: this experiment trains rank-1 adapters only.
        if int(_adapter_cfg.get("r", -1)) != 1:
            raise AssertionError(
                f"adapter r={_adapter_cfg.get('r')} != 1 at {adapter_local} — "
                "wrong adapter for the rank-1 design."
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
        contexts_payload[persona] = {
            "n_prompts": cs.n_prompts,
            "delta_logp_marker": cs.delta_logp_marker,
            "delta_logit_marker": cs.delta_logit_marker,
            "emission_argmax_trained": cs.emission_argmax_trained,
            "emission_argmax_base": cs.emission_argmax_base,
            # §14 duty 8 (variance precondition): per-question deltas.
            "per_question_delta_logp": cs.per_question_delta_logp,
            "per_question_delta_margin": cs.per_question_delta_margin,
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

    json_path = out_dir / f"{cell['cell_slug']}__shift.json"
    pt_path = out_dir / f"{cell['cell_slug']}__shift.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "issue_621_shift_v1",
        "cell_slug": cell["cell_slug"],
        "arm": cell["arm"],
        "source": cell["source"],
        "seed": cell["seed"],
        "hf_adapter": f"{HF_MODEL_REPO}/{cell['hf_subfolder']}",
        "eval_panel": eval_panel,
        "eval_questions": eval_questions,
        "marker_id": MARKER_ID,
        "marker_text": MARKER_TEXT,
        "contexts": contexts_payload,
        "shift_matrix_path": pt_path.name,
        "fork_sha": _git_commit(),
        "parent_pin_sha": PARENT_PIN_SHA,
        "hf_train_mix_read_revision": HF_TRAIN_MIX_READ_REVISION,
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    # Save as a TORCH tensor (a raw numpy array is rejected by the analyzer's
    # torch.load(weights_only=True) — caught by the CPU smoke).
    torch.save(torch.from_numpy(np.asarray(shift_matrix, dtype=np.float32)), pt_path)
    log.info(
        "[phase=shift_extract] cell=%s wrote %s + %s (matrix shape %dx%d)",
        cell["cell_slug"],
        json_path,
        pt_path,
        len(shift_matrix),
        len(shift_matrix[0]) if shift_matrix else 0,
    )

    del base
    del trained
    gc.collect()
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:  # noqa: C901  # argparse wiring + R-coverage precondition guard (inherited #538 shape)
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
        "--cell-slug", default=None, help="Run one cell only (slug = r1_<arm>__<source>__seed<S>)."
    )
    ap.add_argument("--all-cells", action="store_true", help="Run every trained cell.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-root", default="eval_results/issue_621")
    ap.add_argument(
        "--r-persona-dir",
        # INHERITED READ from #527 (sha256-pinned by run_issue621_preflight.py).
        default="eval_results/issue_527/R_persona",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument(
        "--n-eval-questions",
        type=int,
        default=EVAL_N_PROMPTS_PER_PERSONA,
        help="N eval questions per persona (default 20 = full EVAL_QUESTIONS; capped at 20).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a cell if its output JSON already exists.",
    )
    args = ap.parse_args(argv)

    if args.gpu_id != 0:
        # CLAUDE.md gotcha: pin CVD; the launcher should ALSO export it.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard {args.shard} out of range for --num-shards {args.num_shards}")

    out_root = Path(args.out_root)
    eval_out_dir = out_root / "eval"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)
    r_persona: dict[str, dict[str, str]] = {}
    if args.mode == "shift_extract":
        r_dir = Path(args.r_persona_dir)
        for p in sorted(r_dir.glob("*.json")):
            payload = json.loads(p.read_text())
            r_persona[payload["persona"]] = payload["responses"]

    if args.n_eval_questions > len(EVAL_QUESTIONS):
        log.warning(
            "--n-eval-questions=%d exceeds EVAL_QUESTIONS length (%d); capping.",
            args.n_eval_questions,
            len(EVAL_QUESTIONS),
        )
    eval_questions = list(EVAL_QUESTIONS[: args.n_eval_questions])

    # Fail LOUD before any GPU load if R_persona doesn't cover every eval
    # question for every persona used downstream.
    if args.mode == "shift_extract":
        if not r_persona:
            raise SystemExit(
                f"R_persona dir {args.r_persona_dir} is empty/missing; run "
                "scripts/run_issue621_preflight.py first."
            )
        for persona_name, resp in r_persona.items():
            missing = [q for q in eval_questions if q not in resp]
            if missing:
                raise SystemExit(
                    f"R_persona[{persona_name!r}] missing {len(missing)} of "
                    f"{len(eval_questions)} eval questions. First missing: "
                    f"{missing[0]!r}. R is INHERITED sha-pinned from #527; if "
                    "this fails the pinned copy never covered the eval set."
                )

    if args.all_cells:
        cells = _load_all_cells(out_root)
    elif args.cell_slug:
        cells = [c for c in _load_all_cells(out_root) if c["cell_slug"] == args.cell_slug]
        if not cells:
            raise SystemExit(
                f"cell {args.cell_slug!r} not found under {out_root}/{{anchor_smoke,sweep}}."
            )
    else:
        raise SystemExit("Pass --cell-slug <slug> OR --all-cells.")

    if args.num_shards > 1:
        cells = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard]
        log.info("Shard %d/%d evals %d cell(s)", args.shard, args.num_shards, len(cells))

    for neg in UNIFIED_NEGATIVE_PANEL:
        if neg not in persona_bank:
            raise AssertionError(f"negative panel persona {neg!r} not in persona_bank")

    for cell in cells:
        eval_panel = _resolve_eval_panel(persona_bank, cell["source"])
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
                raw_dir=out_root / "raw_generations" / cell["cell_slug"],
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
                # CVD was pinned above for gpu_id != 0, remapping the chosen
                # physical GPU to cuda:0 in-process — so cuda:0 always.
                device="cuda:0",
            )

    log.info("eval mode=%s complete (%d cell(s))", args.mode, len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
