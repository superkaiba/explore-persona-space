"""Issue #650 eval — marker four-float slot reads + sycophancy agreement panel.

Forked from ``scripts/run_issue621_eval.py`` (origin/issue-621 @ 766f44c4).
Two eval paths, dispatched per cell behavior:

MARKER cells: HF-forward four-float marker-slot reads (logp_marker, z_marker,
  z_eos, logZ) BOTH sides same forward pass, on the 19-persona panel +
  assistant + source over EVAL_QUESTIONS, via the ported
  ``shift_extract.extract_per_context_shift``. max_new_tokens N/A (forward-
  only). Persists ``<cell>__shift.json`` with per-persona ContextShift +
  per-question deltas (DV-5 per-bystander leakage in EOS-margin space).

SYCOPHANCY cells: the #612 agreement-panel eval (reused verbatim via
  ``eval_panel.eval_panel``) on the held-out 30-claim probe set, 10 rollouts,
  Claude-Haiku judge — produces the trained agreement rate. The base
  agreement rate (the Δagree denominator) is read from the base pass. Δagree
  per saved epoch checkpoint feeds the dose-to-target band-entry read
  (band_entry.BAND_ENTRY_THRESHOLD reused; the #612 EXPECTED_BAND_ENTRY
  hardcoded table is NOT used — this rig has different cells).

CLI:
    uv run python scripts/run_issue650_eval.py --phase smoke
    uv run python scripts/run_issue650_eval.py --phase sweep
"""

# math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_650 import (  # noqa: E402
    BASE_MODEL,
    EVAL_N_PROMPTS_PER_PERSONA,
    PERSONA_POOL_19,
    SOURCE,
    SYCO_BAND_ENTRY_THRESHOLD,
    SYCO_EVAL_N_ROLLOUTS,
    cell_slug,
    enumerate_cells,
    parse_cell_slug,
)
from explore_persona_space.experiments.issue_650.persona_registry import (  # noqa: E402
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.experiments.issue_650.shift_extract import (  # noqa: E402
    extract_per_context_shift,
)
from explore_persona_space.personas import EVAL_QUESTIONS  # noqa: E402

log = logging.getLogger("issue_650.eval")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _resolve_eval_panel(persona_bank: dict[str, str]) -> list[str]:
    """Held-out marker eval panel: PERSONA_POOL_19 + assistant (+ source, dedup).

    Asserts no byte-identical system prompts for distinct names (would bias the
    leakage panel) — verbatim #621.
    """
    panel = [*list(PERSONA_POOL_19), "assistant"]
    if SOURCE not in panel:
        panel.append(SOURCE)
    seen = []
    for n in panel:
        if n not in seen:
            seen.append(n)
        if n not in persona_bank:
            raise AssertionError(f"eval panel persona {n!r} not in persona_bank")
    by_prompt: dict[str, list[str]] = {}
    for n in seen:
        by_prompt.setdefault(persona_bank[n], []).append(n)
    dups = {p: names for p, names in by_prompt.items() if len(names) > 1}
    if dups:
        diag = {p[:40]: names for p, names in dups.items()}
        raise AssertionError(f"eval panel has byte-identical system prompts: {diag}")
    return seen


def _resolve_adapter_local(meta: dict, cells_root: Path) -> Path:
    """Adapter dir: prefer the on-disk cell dir; else snapshot_download from HF."""
    local = Path(meta.get("output_dir") or (cells_root / "cells" / meta["cell_slug"]))
    if (local / "adapter_model.safetensors").is_file():
        return local
    from huggingface_hub import snapshot_download

    from explore_persona_space.experiments.issue_650 import HF_MODEL_REPO

    sub = meta["hf_subfolder"]
    log.info("Adapter not local; snapshot_download %s/%s", HF_MODEL_REPO, sub)
    dl = snapshot_download(
        repo_id=HF_MODEL_REPO, allow_patterns=[f"{sub}/*"], token=os.environ.get("HF_TOKEN")
    )
    return Path(dl) / sub


def _eval_marker_cell(
    meta: dict, *, panel: list[str], persona_bank, r_persona, cells_root, out_dir
):
    """Marker cell: four-float slot reads on the panel via shift_extract."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    slug = meta["cell_slug"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
    ).eval()
    adapter_dir = _resolve_adapter_local(meta, cells_root)
    trained_raw = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
    )
    trained = PeftModel.from_pretrained(trained_raw, str(adapter_dir)).eval()

    persona_results = {}
    for persona in panel:
        cs = extract_per_context_shift(
            base_model=base,
            trained_model=trained,
            tokenizer=tokenizer,
            persona=persona,
            persona_prompt=persona_bank[persona],
            eval_questions=list(EVAL_QUESTIONS[:EVAL_N_PROMPTS_PER_PERSONA]),
            r_responses=r_persona.get(persona, {}),
            device=device,
        )
        persona_results[persona] = {
            "delta_logp_marker": cs.delta_logp_marker,
            "delta_logit_marker": cs.delta_logit_marker,
            "emission_argmax_trained": cs.emission_argmax_trained,
            "emission_argmax_base": cs.emission_argmax_base,
            "marker_slot_stats_trained": vars(cs.marker_slot_stats_trained),
            "marker_slot_stats_base": vars(cs.marker_slot_stats_base),
            "per_question_delta_logp": cs.per_question_delta_logp,
            "per_question_delta_margin": cs.per_question_delta_margin,
        }
    del base, trained, trained_raw
    import gc

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    payload = {
        "cell_slug": slug,
        "behavior": "marker",
        "dose": meta["dose"],
        "seed": meta["seed"],
        "panel": panel,
        "personas": persona_results,
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    out_path = out_dir / f"{slug}__shift.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("[phase=eval_marker_done] cell=%s -> %s", slug, out_path)
    return payload


def _eval_sycophancy_cell(meta: dict, *, persona_bank, cells_root, out_dir, claims_path):
    """Sycophancy cell: #612 agreement-panel eval on the held-out probe set.

    Reuses eval_panel.eval_panel verbatim. The base agreement rate (Δagree
    denominator) is read from the base pass (model_tag=base). Δagree is folded
    into the dose-to-target band-entry read off-pod.
    """
    import tempfile

    from explore_persona_space.experiments.sycophancy_onpolicy_612.eval_panel import eval_panel
    from explore_persona_space.train.sft import merge_lora

    slug = meta["cell_slug"]
    # Self-persona panel: just the source (the dose dial is the source's own
    # agreement-rate lift on the held-out probe set, plan §4).
    panel = {SOURCE: persona_bank[SOURCE]}
    adapter_dir = _resolve_adapter_local(meta, cells_root)

    cell_out = out_dir / slug
    cell_out.mkdir(parents=True, exist_ok=True)

    # eval_panel serves a merged Qwen+LoRA dir via vLLM. The agreement eval is
    # a BEHAVIORAL generation read (not a geometry read), so the bf16 merge is
    # the #612-standard path; the geometry DVs read the UNMERGED adapter
    # (load_adapter_pairs), so the small-rank-1-delta bf16-merge attenuation
    # (memory feedback_bf16_merge_truncates_small_lora_delta) does NOT touch
    # the geometry headline — it could only soften the agreement RATE, reported
    # as-measured.
    with tempfile.TemporaryDirectory(prefix=f"merged_{slug}_") as merged_dir:
        merge_lora(BASE_MODEL, str(adapter_dir), merged_dir)
        ep = eval_panel(
            model_tag=f"issue650_{slug}",
            seed=meta["seed"],
            panel=panel,
            claims_path=claims_path,
            out_dir=cell_out,
            merged_model_path=Path(merged_dir),
            n_rollouts=SYCO_EVAL_N_ROLLOUTS,
        )
    res = {
        "eval_panel_result": ep,
        "band_entry_threshold": SYCO_BAND_ENTRY_THRESHOLD,
        "merged_for_eval_only": True,
    }
    payload = {
        "cell_slug": slug,
        "behavior": "sycophancy",
        "dose": meta["dose"],
        "seed": meta["seed"],
        "eval": res,
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    out_path = out_dir / f"{slug}__agreement.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("[phase=eval_syco_done] cell=%s -> %s", slug, out_path)
    return payload


def _load_r_persona(out_dir: Path) -> dict[str, dict[str, str]]:
    """Load R_persona JSONs (#621 schema) for the marker eval forward rows."""
    out: dict[str, dict[str, str]] = {}
    if not out_dir.is_dir():
        log.warning("R_persona dir %s missing — marker eval rows need it on the pod", out_dir)
        return out
    for p in sorted(out_dir.glob("*.json")):
        payload = json.loads(p.read_text())
        if payload.get("schema_version") == "issue_527_R_persona_v1":
            out[payload["persona"]] = payload["responses"]
    return out


def _assert_r_persona_coverage_for_marker(
    *, want: set[str], metas: dict[str, dict], panel: list[str], r_persona: dict, eval_questions
) -> None:
    """Fail LOUD at eval startup if any marker cell will run but R_persona is
    incomplete for the panel — BEFORE any (expensive) model load.

    Blocker ``marker-eval-r-persona-missing`` round-2: previously the only
    coverage check was the per-persona AssertionError deep inside
    ``extract_per_context_shift``, which fires AFTER loading the base + trained
    models for the first marker cell. This upfront gate converts that into a
    cheap pre-flight so the pipeline aborts before the GPU is spent. The
    generation script (``run_issue650_generate_r_persona.py``) is the producer;
    this is the consumer-side safety net.
    """
    marker_cells = [
        s for s in want if (m := metas.get(s)) is not None and m.get("behavior") == "marker"
    ]
    if not marker_cells:
        return  # no marker cell in this run; R_persona not needed
    missing: list[str] = []
    for persona in panel:
        resp = r_persona.get(persona)
        if not resp:
            missing.append(f"{persona}: no R_persona entry")
            continue
        for q in eval_questions:
            if q not in resp or not str(resp[q]).strip():
                missing.append(f"{persona}: missing/empty R for q={q[:40]!r}")
    if missing:
        raise AssertionError(
            f"R_persona coverage INCOMPLETE for {len(marker_cells)} marker cell(s) "
            "(blocker marker-eval-r-persona-missing). Run "
            "run_issue650_generate_r_persona.py BEFORE the marker eval. "
            f"{len(missing)} gap(s); first 5:\n  " + "\n  ".join(missing[:5])
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=["smoke", "sweep"])
    ap.add_argument("--cells-root", default="eval_results/issue_650")
    ap.add_argument("--out-root", default="eval_results/issue_650/eval")
    ap.add_argument("--r-persona-dir", default="eval_results/issue_650/R_persona")
    ap.add_argument("--claims-path", default="eval_results/issue_650/inputs/eval_60.jsonl")
    ap.add_argument("--cells", nargs="+", default=None)
    args = ap.parse_args(argv)

    cells_root = Path(args.cells_root)
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)
    panel = _resolve_eval_panel(persona_bank)
    r_persona = _load_r_persona(Path(args.r_persona_dir))
    claims_path = Path(args.claims_path)

    # Resolve cells from the dispatcher's cell JSONs.
    metas: dict[str, dict] = {}
    for sub in ("anchor_smoke", "sweep"):
        d = cells_root / sub
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.name == "summary.json":
                    continue
                payload = json.loads(p.read_text())
                if "cell_slug" in payload:
                    metas[payload["cell_slug"]] = payload

    if args.cells:
        want = set(args.cells)
    elif args.phase == "smoke":
        want = {cell_slug(*c) for c in (("marker", "low", 42), ("sycophancy", "low", 42))}
    else:
        want = {cell_slug(*c) for c in enumerate_cells()}

    # Upfront R_persona coverage gate for any marker cell (blocker
    # marker-eval-r-persona-missing) — fail before the first model load.
    _assert_r_persona_coverage_for_marker(
        want=want,
        metas=metas,
        panel=panel,
        r_persona=r_persona,
        eval_questions=list(EVAL_QUESTIONS[:EVAL_N_PROMPTS_PER_PERSONA]),
    )

    n_done = 0
    for slug in sorted(want):
        meta = metas.get(slug)
        if meta is None:
            log.warning("cell=%s has no train-side JSON; skip eval", slug)
            continue
        behavior, _dose, _seed = parse_cell_slug(slug)
        log.info("[phase=eval_cell] cell=%s behavior=%s", slug, behavior)
        if behavior == "marker":
            _eval_marker_cell(
                meta,
                panel=panel,
                persona_bank=persona_bank,
                r_persona=r_persona,
                cells_root=cells_root,
                out_dir=out_dir,
            )
        else:
            _eval_sycophancy_cell(
                meta,
                persona_bank=persona_bank,
                cells_root=cells_root,
                out_dir=out_dir,
                claims_path=claims_path,
            )
        n_done += 1

    log.info("[phase=eval_dispatch_done] %d cell(s) evaluated", n_done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
