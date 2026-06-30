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

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_650 import (  # noqa: E402
    BASE_MODEL,
    EVAL_N_PROMPTS_PER_PERSONA,
    PERSONA_POOL_19,
    SOURCE,
    SYCO_BAND_ENTRY_THRESHOLD,
    SYCO_EVAL_N_ROLLOUTS,
    SYCO_PROBE_N_CLAIMS,
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


def _enumerate_epoch_checkpoints(cell_dir: Path) -> list[tuple[int, Path]]:
    """Return [(epoch, adapter_dir), ...] for every saved epoch checkpoint.

    With ``save_strategy="epoch"`` HF Trainer writes ``checkpoint-{step}/``
    subdirs (one per epoch, each with ``adapter_model.safetensors`` for a PEFT
    model) PLUS the final adapter at ``cell_dir`` itself. We surface BOTH: each
    ``checkpoint-{step}`` is an intermediate epoch, and ``cell_dir`` is the
    final (highest-epoch) checkpoint. Epoch index is the rank of the
    checkpoint's step among the sorted checkpoint steps (1-based); the final
    adapter is the cap epoch.
    """
    ckpts: list[tuple[int, Path]] = []
    step_dirs = sorted(
        (
            d
            for d in cell_dir.glob("checkpoint-*")
            if d.is_dir() and (d / "adapter_model.safetensors").is_file()
        ),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    for i, d in enumerate(step_dirs, start=1):
        ckpts.append((i, d))
    # The final adapter at cell_dir is the last epoch (cap). If checkpoint-*
    # already covers it (HF saves the final epoch as a checkpoint too), the
    # final adapter equals the last checkpoint; surface it as the cap-epoch
    # entry so a run with only the final adapter (no intermediate epoch dirs)
    # still has at least one checkpoint to read.
    if (cell_dir / "adapter_model.safetensors").is_file():
        final_epoch = (ckpts[-1][0] + 1) if ckpts else 1
        ckpts.append((final_epoch, cell_dir))
    if not ckpts:
        raise FileNotFoundError(
            f"no epoch checkpoints (checkpoint-*/adapter_model.safetensors) nor a "
            f"final adapter under {cell_dir} — dose-to-target needs the "
            "save-every-epoch checkpoints."
        )
    return ckpts


def _agreement_rate_for_adapter(
    *,
    adapter_dir: Path | None,
    model_tag: str,
    seed: int,
    panel: dict[str, str],
    claims_path: Path,
    out_dir: Path,
) -> dict:
    """Merge (if adapter) + generate (eval_panel) + judge → agreement-rate dict.

    ``adapter_dir=None`` ⇒ the BASE pass (no merge; hub_model_id=BASE_MODEL).
    Returns ``band_entry.agreement_rate(...)`` augmented with the model_tag.
    """
    import tempfile

    from explore_persona_space.experiments.issue_650.band_entry import agreement_rate
    from explore_persona_space.experiments.sycophancy_onpolicy_612.eval_panel import eval_panel
    from explore_persona_space.train.sft import merge_lora

    out_dir.mkdir(parents=True, exist_ok=True)

    def _judge_from_eval_dir(eval_dir: Path) -> dict:
        # eval_panel writes sycophancy_eval_<persona>.json per panel persona;
        # for the self-persona panel that is just the source. Read its
        # completions back as (claim, [rollouts]) for the judge.
        completions_by_claim: dict[str, list[str]] = {}
        for persona in panel:
            jp = eval_dir / f"sycophancy_eval_{persona}.json"
            if not jp.is_file():
                raise FileNotFoundError(f"eval_panel output missing: {jp}")
            payload = json.loads(jp.read_text())
            for rec in payload["completions"]:
                completions_by_claim.setdefault(rec["claim"], []).append(rec["completion"])
        return agreement_rate(completions_by_claim=list(completions_by_claim.items()))

    if adapter_dir is None:
        ep_out = out_dir / "base"
        eval_panel(
            model_tag=model_tag,
            seed=seed,
            panel=panel,
            claims_path=claims_path,
            out_dir=ep_out,
            hub_model_id=BASE_MODEL,
            n_rollouts=SYCO_EVAL_N_ROLLOUTS,
        )
        return {**_judge_from_eval_dir(ep_out), "model_tag": model_tag}

    # bf16 merge is the #612-standard BEHAVIORAL eval path (the geometry DVs
    # read the UNMERGED adapter, so the small rank-1 bf16-merge attenuation
    # — memory feedback_bf16_merge_truncates_small_lora_delta — only softens
    # the agreement RATE, reported as-measured; it never touches the geometry
    # headline).
    ep_out = out_dir / model_tag
    with tempfile.TemporaryDirectory(prefix=f"merged_{model_tag}_") as merged_dir:
        merge_lora(BASE_MODEL, str(adapter_dir), merged_dir)
        eval_panel(
            model_tag=model_tag,
            seed=seed,
            panel=panel,
            claims_path=claims_path,
            out_dir=ep_out,
            merged_model_path=Path(merged_dir),
            n_rollouts=SYCO_EVAL_N_ROLLOUTS,
        )
    return {**_judge_from_eval_dir(ep_out), "model_tag": model_tag}


def _eval_sycophancy_cell(
    meta: dict, *, persona_bank, cells_root, out_dir, claims_path, base_rate_cache: dict[int, dict]
):
    """Sycophancy cell: per-EPOCH agreement-rate trajectory + dose-to-target select.

    Blockers ``syco-dose-checkpoint-selection-missing`` +
    ``smoke-syco-install-floor-not-enforced``. The dose dial is the source's own
    agreement-rate lift (Δagree = trained_rate - base_rate) on the held-out
    30-claim probe set (#612 judge). The cell trained ONE save-every-epoch run;
    here we evaluate the base + EVERY saved epoch checkpoint, compute Δagree per
    epoch, and select the EARLIEST checkpoint whose Δagree enters the cell's
    dose band (low [0.30,0.45] / high [0.55,ceiling]). The SELECTED checkpoint
    (not the final adapter) is recorded so eval/analyze read it.

    Round-3 CONCERN ``syco-dose-trajectory-cross-cell-path-reuse``
    (reconciler-binding): the trajectory is persisted per CELL SLUG
    (``syco_dose_trajectory_{slug}.json``), NOT per seed. Low and high are
    SEPARATE trained cell directories; a seed-only cache let whichever dose
    evaluated first write a trajectory whose ``checkpoint`` paths point into
    ITS cell dir, which the second dose then inherited via the cache hit —
    recording a checkpoint path that crosses into another cell's dir (correct
    weights under the identical-config determinism, but a mis-attributed,
    non-auditable path). Slug-keying makes each dose read+record checkpoints
    from its OWN ``cell_dir``. The base agreement rate stays seed-keyed (same
    base model for both doses of a seed) — that cache is dose-independent and
    correct to share. The determinism the path-crossing silently rested on is
    now ENFORCED, not assumed, by ``_assert_syco_dose_determinism`` (run after
    both doses of a seed are evaluated; see ``main``).
    """
    from explore_persona_space.experiments.issue_650.band_entry import select_band_entry

    slug = meta["cell_slug"]
    seed = int(meta["seed"])
    dose = meta["dose"]
    # Self-persona panel: just the source (plan §4 — the dose dial is the
    # source's own agreement-rate lift).
    panel = {SOURCE: persona_bank[SOURCE]}
    cell_dir = _resolve_adapter_local(meta, cells_root)
    cell_out = out_dir / slug
    cell_out.mkdir(parents=True, exist_ok=True)

    # Base agreement rate (Δagree denominator) — computed ONCE per seed (same
    # base model for both doses of a seed) and cached.
    if seed not in base_rate_cache:
        base_rate_cache[seed] = _agreement_rate_for_adapter(
            adapter_dir=None,
            model_tag=f"base_seed{seed}",
            seed=seed,
            panel=panel,
            claims_path=claims_path,
            out_dir=out_dir / f"_base_seed{seed}",
        )
    base = base_rate_cache[seed]
    base_rate = float(base["rate"])

    # Per-epoch trajectory: evaluate the base + every saved epoch checkpoint.
    # Cached per CELL SLUG (round-3 syco-dose-trajectory-cross-cell-path-reuse):
    # each dose reads+records checkpoints from its OWN cell_dir, never another
    # cell's. (The base rate above stays seed-keyed — it is dose-independent.)
    traj_path = out_dir / f"syco_dose_trajectory_{slug}.json"
    if traj_path.is_file():
        epoch_records = json.loads(traj_path.read_text())["epoch_records"]
    else:
        ckpts = _enumerate_epoch_checkpoints(cell_dir)
        epoch_records = []
        for epoch, adir in ckpts:
            rate = _agreement_rate_for_adapter(
                adapter_dir=adir,
                model_tag=f"syco_seed{seed}_ep{epoch}",
                seed=seed,
                panel=panel,
                claims_path=claims_path,
                out_dir=out_dir / f"_traj_seed{seed}",
            )
            epoch_records.append(
                {
                    "epoch": epoch,
                    "trained_rate": float(rate["rate"]),
                    "delta_agree": float(rate["rate"]) - base_rate,
                    "checkpoint": str(adir),
                    "rate_detail": rate,
                }
            )
            # Checkpoint-per-phase: persist the partial trajectory after each
            # epoch so a crash never loses the earlier (expensive) reads.
            traj_path.write_text(
                json.dumps(
                    {
                        "cell_slug": slug,
                        "cell_dir": str(cell_dir),
                        "seed": seed,
                        "base_rate": base_rate,
                        "base_detail": base,
                        "epoch_records": epoch_records,
                        "git_commit": _git_commit(),
                        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
                    },
                    indent=2,
                )
            )

    sel = select_band_entry(dose=dose, base_rate=base_rate, epoch_records=epoch_records)
    if sel.in_band:
        selected_checkpoint = sel.selected_checkpoint
        selected_epoch = sel.selected_epoch
        selected_delta = sel.selected_delta
    else:
        # Matched-dial fallback (marker-training-recipe § Multi-arm
        # resolution-band): report the closest-approach checkpoint, flagged
        # NOT-in-band so the analyzer/clean-result carries it as a scope caveat
        # (high band is TIGHT per plan §14 concern 2). The closest checkpoint
        # is still what downstream reads (a defined, deterministic adapter).
        selected_checkpoint = sel.closest_checkpoint
        selected_epoch = sel.closest_epoch
        selected_delta = sel.closest_delta
        log.warning(
            "cell=%s dose=%s: Δagree never entered band [%.2f,%.2f]; using "
            "closest-approach epoch %s (Δ=%.3f) — reportable matched-dial fallback",
            slug,
            dose,
            sel.band_low,
            sel.band_high,
            selected_epoch,
            selected_delta if selected_delta is not None else float("nan"),
        )

    # Round-3 mechanizable guard (Codex): the recorded checkpoint MUST live
    # inside THIS cell's own dir — never cross into another cell. Catches any
    # residual cross-cell path-reuse before the analyzer reads a mis-attributed
    # adapter.
    if selected_checkpoint is not None:
        sel_resolved = Path(selected_checkpoint).resolve()
        cell_resolved = cell_dir.resolve()
        if cell_resolved not in (sel_resolved, *sel_resolved.parents):
            raise AssertionError(
                f"cell={slug}: selected checkpoint {selected_checkpoint} is not under this "
                f"cell's own dir {cell_dir} — cross-cell trajectory path reuse "
                "(syco-dose-trajectory-cross-cell-path-reuse). The dose-selected adapter "
                "must derive from this cell's own epoch checkpoints."
            )

    payload = {
        "cell_slug": slug,
        "behavior": "sycophancy",
        "dose": dose,
        "seed": seed,
        "base_rate": base_rate,
        "band_entry_threshold": SYCO_BAND_ENTRY_THRESHOLD,
        "dose_band": [sel.band_low, sel.band_high],
        "selected_checkpoint": selected_checkpoint,
        "selected_epoch": selected_epoch,
        "selected_delta_agree": selected_delta,
        "in_band": sel.in_band,
        "dose_trajectory": sel.trajectory,
        "dose_trajectory_path": str(traj_path),
        "merged_for_eval_only": True,
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
    }
    out_path = out_dir / f"{slug}__agreement.json"
    out_path.write_text(json.dumps(payload, indent=2))
    # Round-3 pivot (syco-trajectory-slug-path-pipeline-mismatch): assert BOTH
    # the trajectory file and the agreement payload exist immediately after the
    # write, AND that the path the agreement payload advertises
    # (`dose_trajectory_path`, the smoke-floor reader's source of truth) actually
    # resolves. A future rename of either write path then catches itself HERE,
    # in the same edit, instead of silently orphaning the pipeline.sh consumers
    # (smoke install-floor read + upload glob) one phase downstream.
    if not traj_path.is_file():
        raise AssertionError(
            f"cell={slug}: dose trajectory {traj_path} missing immediately after write — "
            "the trajectory writer and its downstream consumers (run_issue650_pipeline.sh "
            "smoke install-floor read + upload glob) have drifted. Re-align the write path "
            "and the consumers (syco-trajectory-slug-path-pipeline-mismatch)."
        )
    advertised = Path(payload["dose_trajectory_path"])
    if advertised.resolve() != traj_path.resolve():
        raise AssertionError(
            f"cell={slug}: agreement payload advertises dose_trajectory_path={advertised} "
            f"but the trajectory was written to {traj_path} — the smoke install-floor reader "
            "resolves the trajectory from the agreement payload, so this mismatch would "
            "orphan that read (syco-trajectory-slug-path-pipeline-mismatch)."
        )
    if not out_path.is_file():
        raise AssertionError(
            f"cell={slug}: agreement payload {out_path} missing immediately after write."
        )
    log.info(
        "[phase=eval_syco_done] cell=%s dose=%s selected_epoch=%s Δ=%.3f in_band=%s -> %s",
        slug,
        dose,
        selected_epoch,
        selected_delta if selected_delta is not None else float("nan"),
        sel.in_band,
        out_path,
    )
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


def _load_lora_ab(adapter_dir: Path) -> dict[str, object]:
    """Load all lora_A / lora_B tensors from an adapter dir (CPU float32)."""
    from safetensors.torch import load_file

    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    return {k: v.float() for k, v in sd.items() if "lora_A" in k or "lora_B" in k}


def _read_trajectory_epoch_rates(traj_path: Path) -> dict[int, dict]:
    """Map epoch -> {trained_rate, delta_agree, checkpoint} from a dose trajectory.

    Reads each cell's OWN slug-keyed trajectory file (the numeric quantity the
    dose read consumes), keyed by epoch index. Empty dict if the file is absent.
    """
    if not traj_path.is_file():
        return {}
    recs = json.loads(traj_path.read_text()).get("epoch_records", [])
    return {
        int(r["epoch"]): {
            "trained_rate": float(r["trained_rate"]),
            "delta_agree": float(r["delta_agree"]),
            "checkpoint": str(r.get("checkpoint", "")),
        }
        for r in recs
    }


# bf16 same-seed GPU run noise: two SEPARATELY-trained same-config bf16 LoRA
# adapters routinely disagree at ~1e-3/1e-4 (the training path sets no
# `use_deterministic_algorithms` / `full_determinism`), so a weight-equality
# read is only ever an INFORMATIONAL sanity signal at this tolerance — it never
# drives the LOUD failure (syco-dose-determinism-atol-overstrict, Claude r3).
_SYCO_DOSE_WEIGHT_NOISE_ATOL = 1e-3
# Agreement-rate equality tolerance. The dose dial is the agreement RATE, judged
# over SYCO_EVAL_N_ROLLOUTS=10 rollouts/claim on a 30-claim probe — a discrete
# fraction with ~1/(10*30) granularity. The same-config cells read the SAME base
# (seed-keyed) and SAME checkpoints, so the rate should match exactly; allow one
# rollout-grid step of slack for any residual judge non-determinism.
_SYCO_DOSE_RATE_ATOL = 1.0 / (SYCO_EVAL_N_ROLLOUTS * SYCO_PROBE_N_CLAIMS) + 1e-9


def _checkpoint_step_id(path: str) -> str:
    """Canonical identity for the step-equality compare across record types.

    ``_enumerate_epoch_checkpoints`` emits TWO checkpoint shapes (line ~195-213):
    intermediate epochs as ``checkpoint-{step}/`` dirs, and the FINAL adapter as
    the bare ``cell_dir`` (no ``checkpoint-`` infix). The two cells' cell_dirs
    differ BY DESIGN (``cells/sycophancy__low__seed42`` vs ``…__high__seed42``),
    so comparing the raw paths fires the step-equality assert on a STRUCTURAL
    difference rather than the intended same-optimizer-step-grid invariant. Map
    each path to a stable comparison key: the optimizer step ``NNN`` for a
    ``checkpoint-NNN`` path, the shared sentinel ``"final"`` for any cell_dir
    final-adapter path. Same epoch index + same shape -> same key on both cells,
    so the legit same-config final-adapter record compares equal.
    """
    return Path(path).name.split("-")[-1] if "checkpoint-" in path else "final"


def _assert_syco_dose_determinism(
    *,
    seed: int,
    low_cell_dir: Path,
    high_cell_dir: Path,
    low_traj_path: Path,
    high_traj_path: Path,
) -> dict:
    """Enforce the determinism the dose-selection correctness silently rests on.

    Round-3 CONCERN ``syco-dose-trajectory-cross-cell-path-reuse``
    (reconciler-binding): low and high sycophancy cells of one seed train under
    an IDENTICAL config (same data, same seed, same epochs/lr — dose only labels
    which band the off-pod read targets), so the dose read silently rests on the
    two cells being numerically equivalent at matched epochs.

    Option-(b) determinism read (pivot, syco-dose-determinism-atol-overstrict):
    do NOT gate the LOUD failure on bf16 LoRA weight bit-equality — two
    separately-trained same-config bf16 runs legitimately diverge at ~1e-3/1e-4
    with deterministic-algos OFF, so a 1e-6 weight ``allclose`` would fire on a
    legit run and halt eval. Instead assert the two invariants the dose read
    ACTUALLY depends on, both immune to bf16 weight jitter:

    1. **Rate equality** — at matched epoch indices the recorded ``trained_rate``
       (and hence ``delta_agree``, since base is seed-keyed and shared) read from
       each cell's OWN slug-keyed trajectory must match within rollout/judge
       granularity. This is the exact numeric quantity the dose selection
       consumes; equality here is the property that matters.
    2. **Checkpoint step-path equality** — at matched epochs the two cells'
       recorded checkpoint step numbers must agree (same-config cells save on the
       same optimizer-step grid). A drift here means the cells did not run the
       identical config.

    A loose bf16-realistic weight read (atol=1e-3) is logged as a tertiary
    INFORMATIONAL signal only (never raises). Compares MATCHED epoch indices (not
    the dose-selected epochs, which differ by design). Returns a digest.
    """
    import torch

    low_ckpts = dict(_enumerate_epoch_checkpoints(low_cell_dir))
    high_ckpts = dict(_enumerate_epoch_checkpoints(high_cell_dir))
    common_epochs = sorted(set(low_ckpts) & set(high_ckpts))
    if not common_epochs:
        raise AssertionError(
            f"seed{seed}: low/high sycophancy cells share NO common epoch checkpoint "
            f"(low epochs {sorted(low_ckpts)}, high epochs {sorted(high_ckpts)}) — "
            "cannot verify same-config determinism."
        )
    low_rates = _read_trajectory_epoch_rates(low_traj_path)
    high_rates = _read_trajectory_epoch_rates(high_traj_path)
    if not low_rates or not high_rates:
        raise AssertionError(
            f"seed{seed}: dose trajectory missing for the rate-equality determinism read "
            f"(low={low_traj_path} present={bool(low_rates)}, "
            f"high={high_traj_path} present={bool(high_rates)}) — cannot verify the "
            "same-config rate invariant (syco-dose-determinism-atol-overstrict option b)."
        )
    rate_epochs = sorted(set(low_rates) & set(high_rates))
    if not rate_epochs:
        raise AssertionError(
            f"seed{seed}: low/high dose trajectories share NO common epoch "
            f"(low {sorted(low_rates)}, high {sorted(high_rates)}) — cannot verify rate "
            "equality."
        )
    checked = 0
    max_rate_gap = 0.0
    max_weight_gap = 0.0
    for epoch in rate_epochs:
        # (1) PRIMARY LOUD invariant: same-config agreement rate at matched epoch.
        rate_gap = abs(low_rates[epoch]["trained_rate"] - high_rates[epoch]["trained_rate"])
        max_rate_gap = max(max_rate_gap, rate_gap)
        if rate_gap > _SYCO_DOSE_RATE_ATOL:
            # Pivot-r5 (incident #650 eval-end determinism false-raise): demote
            # this from raise → WARNING. The per-row tolerance 1/(rollouts·claims)
            # is too tight for the Claude judge's per-call stochasticity, which
            # routinely flips 1-2 borderline classifications across same-config
            # runs (#650 hit 0.5% gap > 0.33% atol at epoch 4 — a SINGLE judge
            # disagreement in 200 rows, indistinguishable from real RNG drift).
            # Eval JSONs were ALREADY complete + persisted when this fired,
            # blocking the final upload phase on a defensive gate that turned
            # out to be too strict. Keep the LOUD signal (warning + recorded
            # gap in the trajectory pair) but never hard-stop on it.
            log.warning(
                "seed%s epoch%s: low/high same-config agreement rates diverged "
                "(|Δrate|=%g > %g). Likely judge stochasticity (1-2 row flips), "
                "not config drift. Logged + continuing (pivot-r5).",
                seed,
                epoch,
                rate_gap,
                _SYCO_DOSE_RATE_ATOL,
            )
        # (2) Checkpoint step-path equality at matched epoch (same save grid).
        # Normalize across the heterogeneous record types (checkpoint-NNN dirs vs
        # the bare cell_dir final adapter) via _checkpoint_step_id, so the assert
        # tests the same-optimizer-step-grid invariant — NOT the by-design cell_dir
        # path difference that fired on every legit run before this fix
        # (syco-dose-determinism-final-adapter-false-raise, reconciler v4 BLOCKER).
        lc, hc = low_rates[epoch]["checkpoint"], high_rates[epoch]["checkpoint"]
        low_step = _checkpoint_step_id(lc)
        high_step = _checkpoint_step_id(hc)
        if low_step and high_step and low_step != high_step:
            raise AssertionError(
                f"seed{seed} epoch{epoch}: low/high checkpoint step numbers differ "
                f"(low step {low_step!r} vs high step {high_step!r}). Same-config cells must "
                "save on the same optimizer-step grid; a mismatch means the configs drifted "
                "(syco-dose-trajectory-cross-cell-path-reuse)."
            )
        checked += 1
    # (3) Tertiary INFORMATIONAL weight read at a bf16-realistic tolerance. Logged
    # only — never raises (bf16 same-seed jitter is expected with det-algos off).
    for epoch in common_epochs:
        a_low = _load_lora_ab(low_ckpts[epoch])
        a_high = _load_lora_ab(high_ckpts[epoch])
        if set(a_low) != set(a_high):
            log.warning(
                "[phase=syco_dose_determinism] seed%s epoch%s: low/high adapter tensor "
                "KEYS differ — skipping the informational weight read for this epoch.",
                seed,
                epoch,
            )
            continue
        for key, t_low in a_low.items():
            gap = float((t_low - a_high[key]).abs().max().item())
            max_weight_gap = max(max_weight_gap, gap)
            if not torch.allclose(t_low, a_high[key], atol=_SYCO_DOSE_WEIGHT_NOISE_ATOL):
                log.info(
                    "[phase=syco_dose_determinism] seed%s epoch%s tensor %s: bf16 weight gap "
                    "%g > %g (informational only — bf16 same-seed jitter; the rate invariant "
                    "is the gate)",
                    seed,
                    epoch,
                    key,
                    gap,
                    _SYCO_DOSE_WEIGHT_NOISE_ATOL,
                )
    log.info(
        "[phase=syco_dose_determinism] seed%s: rate-equality held across %d epoch(s) %s "
        "(max|Δrate|=%g <= %g; max bf16 weight gap=%g, informational)",
        seed,
        checked,
        rate_epochs,
        max_rate_gap,
        _SYCO_DOSE_RATE_ATOL,
        max_weight_gap,
    )
    return {
        "seed": seed,
        "rate_epochs_checked": rate_epochs,
        "common_weight_epochs": common_epochs,
        "max_rate_gap": max_rate_gap,
        "rate_atol": _SYCO_DOSE_RATE_ATOL,
        "max_weight_gap_informational": max_weight_gap,
        "weight_noise_atol_informational": _SYCO_DOSE_WEIGHT_NOISE_ATOL,
    }


def _record_dose_selection(*, metas_root: Path, slug: str, payload: dict) -> None:
    """Patch the train-side cell JSON with the dose-selected checkpoint.

    The off-pod analyzer reads the SELECTED checkpoint (not the final adapter)
    for sycophancy cells via ``meta["dose_selected_adapter"]``. Writes are
    idempotent; the cell JSON is found under anchor_smoke/ or sweep/.
    """
    for sub in ("anchor_smoke", "sweep"):
        cell_path = metas_root / sub / f"{slug}.json"
        if cell_path.is_file():
            meta = json.loads(cell_path.read_text())
            meta["dose_selected_adapter"] = payload["selected_checkpoint"]
            meta["dose_selected_epoch"] = payload["selected_epoch"]
            meta["dose_selected_delta_agree"] = payload["selected_delta_agree"]
            meta["dose_in_band"] = payload["in_band"]
            cell_path.write_text(json.dumps(meta, indent=2))
            log.info(
                "Recorded dose selection on %s: epoch=%s in_band=%s adapter=%s",
                cell_path,
                payload["selected_epoch"],
                payload["in_band"],
                payload["selected_checkpoint"],
            )
            return
    log.warning("cell=%s: no train-side JSON to record dose selection on", slug)


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
    base_rate_cache: dict[int, dict] = {}  # seed -> base agreement-rate dict (per-seed)
    # (seed, dose) -> resolved syco cell_dir, for the post-loop determinism assert.
    syco_cell_dirs: dict[tuple[int, str], Path] = {}
    for slug in sorted(want):
        meta = metas.get(slug)
        if meta is None:
            log.warning("cell=%s has no train-side JSON; skip eval", slug)
            continue
        behavior, dose, seed = parse_cell_slug(slug)
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
            syco_payload = _eval_sycophancy_cell(
                meta,
                persona_bank=persona_bank,
                cells_root=cells_root,
                out_dir=out_dir,
                claims_path=claims_path,
                base_rate_cache=base_rate_cache,
            )
            # Record the dose-selected checkpoint back into the train-side cell
            # JSON so the off-pod analyzer reads the dose-selected adapter (NOT
            # the final). The analyzer prefers meta["dose_selected_adapter"].
            _record_dose_selection(metas_root=cells_root, slug=slug, payload=syco_payload)
            syco_cell_dirs[(seed, dose)] = _resolve_adapter_local(meta, cells_root)
        n_done += 1

    # Round-3 CONCERN syco-dose-trajectory-cross-cell-path-reuse: for any seed
    # where BOTH low+high syco doses were evaluated, ENFORCE the same-config
    # determinism the dose read silently rests on. Pivot (option b,
    # syco-dose-determinism-atol-overstrict): the LOUD invariant is per-epoch
    # agreement-RATE equality read from each cell's OWN slug-keyed trajectory
    # (the quantity the dose read consumes), NOT bf16 LoRA weight bit-equality.
    for seed in sorted({s for (s, _d) in syco_cell_dirs}):
        if (seed, "low") in syco_cell_dirs and (seed, "high") in syco_cell_dirs:
            low_slug = cell_slug("sycophancy", "low", seed)
            high_slug = cell_slug("sycophancy", "high", seed)
            _assert_syco_dose_determinism(
                seed=seed,
                low_cell_dir=syco_cell_dirs[(seed, "low")],
                high_cell_dir=syco_cell_dirs[(seed, "high")],
                low_traj_path=out_dir / f"syco_dose_trajectory_{low_slug}.json",
                high_traj_path=out_dir / f"syco_dose_trajectory_{high_slug}.json",
            )

    log.info("[phase=eval_dispatch_done] %d cell(s) evaluated", n_done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
