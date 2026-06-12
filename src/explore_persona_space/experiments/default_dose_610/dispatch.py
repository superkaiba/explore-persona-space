# em-dash + Greek ΔG + Qwen marker " ※" intentional
"""Task #610 — thin unified smoke=sweep driver over the reused #600 helpers.

Smoke IS the sweep with one (cell, seed): the SAME ``scripts/i610_run_cell.py``
subprocess shape, the same CVD launcher-env injection + ``--gpu-id`` threading,
the same ``[phase=...]`` logging surface terminating in a single
``[phase=done]``, the same sentinel writer. ``--full`` = the smoke pair first
→ gates (a)-(j) → the remaining seeds through the identical path, in ONE
invocation (one GCP provision; ``EPM_SKIP_EXISTING=1`` covers relaunch).

Phases: marker assert → sha256-pinned prefetch → bank-hash check → spec build
+ committed-design consistency assert → smoke pair (seed 42) → gates: reused
(a)-(h) + NEW (i) primary-DV existence (hard) + (j) chassis comparability
(soft, recorded) → on PASS, remaining seeds (parallel) → uploads (#610 HF
prefixes; adapter uploads Hub-verified) → sentinel → ``[phase=done]``.

Kill criterion (plan §7.1): gates (a)/(b) out-of-band at 63 steps → HALT AND
REPORT (failure-shaped sentinel, rc=2). NO epochs ladder — re-pinning epochs
would unmatch the reused parent arm's 63 steps and void the comparison.

Pod-side contract (poll_pipeline.py): ``[phase=...]`` lines ending in ONE
``[phase=done]``; end-of-run sentinel under ``/workspace/logs/issue-610-*``
with ``sentinel_schema_version``/``kind``/``version``. Sentinel KINDS route
per plan §7.1: normal completion → ``epm:results`` (full payload contract —
reproducibility_card with explicit per-cell adapter_paths + wandb_project +
wandb_run_names); gate-fail / crash HALT_AND_REPORT → ``epm:failure`` with a
leading ``failure_class: code|data`` line (gate (i) / wiring → code,
out-of-band-implant gates (a)/(b) → data); ``--plan-only`` → ``epm:progress``
(validation evidence, never a results-shaped sentinel). The pod NEVER shells
out to scripts/task.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.default_dose_610 import (
    BASE_MODEL,
    CHASSIS_DG_SOFT_RANGE_NATS,
    EPOCHS_PINNED,
    EXPECTED_MARKER_TOKEN_ID,
    EXPECTED_STEPS_PER_EPOCH,
    EXTRA_EVAL_PERSONAS,
    GPU_HOURS_BUDGETED,
    HF_ADAPTER_PATH_PREFIX,
    HF_DATA_PREFIX,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    MARKER_TEXT,
    RUN_NAME_PREFIX,
    SEEDS,
    SOURCE_DG_BAND_NATS,
    WANDB_PROJECT,
)
from explore_persona_space.experiments.default_dose_610.cells import (
    assert_design_matches,
    build_610_spec,
)
from explore_persona_space.experiments.targeted_proximity_600.cells import (
    CellSpec600,
    load_manifest,
)
from explore_persona_space.experiments.targeted_proximity_600.dispatch import (
    _i472_data_root,
    _load_bank_and_r_train,
    _prefetch_inherited_artifacts,
    _run_cells_subprocess,
    _terminal_checkpoint,
    assert_marker_tokenization,
    check_smoke_gates_600,
)

log = logging.getLogger("issue_610.dispatch")

RUN_CELL_SCRIPT = "i610_run_cell.py"
TERMINAL_FRAC = 1.0

# The REALIZED four-float leaf suffix convention (verified against the parent
# trajectories: `_g`/`_b`, NOT `_trained`/`_base`). Gate (i) + analyze.py both
# assert these exact names.
FOUR_FLOAT_FIELDS = (
    "z_marker_g",
    "z_marker_b",
    "z_eos_g",
    "z_eos_b",
    "logZ_g",
    "logZ_b",
    "logp_hf_g",
    "logp_hf_b",
)


# ── Path resolvers (env-overridable for local smokes / tests). ──────────────


def _output_root() -> Path:
    return Path(os.environ.get("EPM_OUTPUT_ROOT", "eval_results/issue_610"))


def _data_root() -> Path:
    return Path(os.environ.get("EPM_DATA_ROOT", "data/issue_610"))


def _parent_manifest_path() -> Path:
    """The PARENT #600 design manifest (committed on main; source of truth)."""
    return Path(os.environ.get("EPM_I600_MANIFEST", "eval_results/issue_600/panel_selection.json"))


def _design_path() -> Path:
    return _output_root() / "design.json"


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── Gate (i): primary-DV existence (hard). ───────────────────────────────────


def gate_i_primary_dv_exists(trajectory_payload: dict) -> dict:
    """Every checkpoint must carry held_out leaves for EVERY extra eval persona
    with all four floats under the REALIZED `_g`/`_b` suffix convention.

    This is the ``extra_eval_personas`` wiring proof: in the no-default arm
    ``qwen_default``/``assistant`` are in NO default eval set, so a wiring
    miss silently drops the primary DV (plan §4.2 risk row 2).
    """
    missing: list[str] = []
    for ck in trajectory_payload["checkpoints"]:
        held_out = ck["held_out"]
        for persona in EXTRA_EVAL_PERSONAS:
            recs = held_out.get(persona)
            if not recs:
                missing.append(f"frac={ck['frac']}: held_out[{persona!r}] absent/empty")
                continue
            for q, leaf in recs.items():
                absent = [f for f in FOUR_FLOAT_FIELDS if leaf.get(f) is None]
                if absent:
                    missing.append(f"frac={ck['frac']}: {persona}/{q!r} lacks {absent}")
    return {
        "personas_required": list(EXTRA_EVAL_PERSONAS),
        "four_float_fields": list(FOUR_FLOAT_FIELDS),
        "missing": missing[:20],  # cap the report; emptiness is the gate
        "n_missing": len(missing),
        "passes": not missing,
    }


# ── Gate (j): chassis comparability (SOFT — recorded, never gating). ─────────


def gate_j_chassis_comparability(trajectory_payload: dict) -> dict:
    """Villain ΔG vs the parent arm's realized range ± 2 nats (plan §4.2 (j)).

    Outside the soft range but inside gate (a)'s [5, 19] → proceed, flag in
    analysis (the normalized+centered DV absorbs implant-strength variation;
    a >3-nat slot-swap effect would itself contradict the parent's ≤0.5-nat
    slot lead and is reportable).
    """
    ck = _terminal_checkpoint(trajectory_payload)
    dg = float(ck["source_self"]["delta_g_mean"])
    low, high = CHASSIS_DG_SOFT_RANGE_NATS
    return {
        "source_dg_mean_nats": dg,
        "soft_range_nats": [low, high],
        "within_parent_range": low <= dg <= high,
        "soft": True,
    }


def check_smoke_gates_610(
    *,
    trajectory_path: Path,
    band_trajectory_path: Path,
    verify_payload: dict,
    collator_payload: dict,
    checkpoint_index: dict,
    expected_steps: int,
    panel_personas: list[str],
    smoke_out_path: Path,
) -> dict:
    """Reused (a)-(h) + NEW (i) hard + (j) soft; merged verdict rewritten in place.

    ``all_gates_passed`` = (a)-(h) AND (i). Gate (j) is recorded but never
    gates (plan §4.2).
    """
    payload_600 = check_smoke_gates_600(
        trajectory_path=trajectory_path,
        band_trajectory_path=band_trajectory_path,
        verify_payload=verify_payload,
        collator_payload=collator_payload,
        checkpoint_index=checkpoint_index,
        expected_steps=expected_steps,
        panel_personas=panel_personas,
        smoke_out_path=smoke_out_path,
    )
    trajectory_payload = json.loads(trajectory_path.read_text())
    gi = gate_i_primary_dv_exists(trajectory_payload)
    gj = gate_j_chassis_comparability(trajectory_payload)
    merged = dict(payload_600)
    merged["gate_i_primary_dv_exists"] = gi["passes"]
    merged["gate_i_detail"] = gi
    merged["gate_j_within_parent_range"] = gj["within_parent_range"]
    merged["gate_j_detail"] = gj
    hard_gates = {
        k: v
        for k, v in merged.items()
        if k.startswith("gate_") and k != "gate_j_within_parent_range" and isinstance(v, bool)
    }
    merged["all_gates_passed"] = all(hard_gates.values())
    smoke_out_path.write_text(json.dumps(merged, indent=2))
    log.info(
        "[smoke-gate-610] gate_i=%s (n_missing=%d) gate_j_within=%s (dg=%.2f); all=%s",
        gi["passes"],
        gi["n_missing"],
        gj["within_parent_range"],
        gj["source_dg_mean_nats"],
        merged["all_gates_passed"],
    )
    return merged


# ── Uploads (#610 prefixes) + Hub verification of the inline adapter uploads. ─


def _upload_phase_610(out_root: Path, data_root: Path, design_path: Path) -> None:
    """Training JSONLs + manifests + design.json → HF data repo; raw completions too.

    Adapters were already uploaded inline by train_lora (per-cell
    hf_path_in_repo under ``adapters/issue_610``); ``_verify_adapter_uploads``
    Hub-asserts them afterwards. Fail-loud throughout (Upload Policy).
    """
    from explore_persona_space.orchestrate.hub import (
        _upload,
        upload_dataset_directory,
        upload_raw_completions_to_data_repo,
    )

    log.info("[phase=upload] training JSONLs from %s", data_root)
    upload_dataset_directory(data_root, f"{HF_DATA_PREFIX}/training_data", pattern="*.jsonl")
    upload_dataset_directory(
        data_root, f"{HF_DATA_PREFIX}/training_data", pattern="*.manifest.json"
    )
    log.info("[phase=upload] design manifest %s", design_path)
    url = _upload(
        local_path=design_path,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_DATA_PREFIX}/design.json",
        upload_as_file=True,  # load-bearing: a FILE path otherwise hits the folder branch
    )
    if not url:
        raise RuntimeError(f"design.json upload returned empty URL ({design_path}).")
    log.info("[phase=upload] raw completions under %s", out_root)
    upload_raw_completions_to_data_repo(experiment_name=HF_DATA_PREFIX, eval_results_dir=out_root)


def _verify_adapter_uploads(results: list[dict]) -> dict[str, str]:
    """Hub-assert every per-cell adapter path resolves (the epm:results card
    contract). Enumerates via the paginated ``list_repo_files_complete`` —
    ``repo_info().siblings``-backed listings silently truncate at ~7901
    entries on large repos (hub.py docstring), which would false-fail this
    check on the project model repo. Fail-loud."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_repo_files_complete

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo_files = list_repo_files_complete(api, HF_MODEL_REPO, repo_type="model")
    adapter_paths: dict[str, str] = {}
    missing: list[str] = []
    for r in results:
        cell_key = f"{r['cell_slug']}_seed{r['seed']}"
        prefix = f"{HF_ADAPTER_PATH_PREFIX}/{cell_key}"
        n = sum(1 for f in repo_files if f.startswith(prefix + "/"))
        if n == 0:
            missing.append(prefix)
        adapter_paths[cell_key] = f"{HF_MODEL_REPO}/{prefix}"
    if missing:
        raise RuntimeError(
            f"[upload-verify] {len(missing)} adapter path(s) have ZERO files on "
            f"{HF_MODEL_REPO}: {missing} — the inline train_lora upload did not land; "
            "refusing to write a card that declares them."
        )
    log.info("[upload-verify] %d adapter paths resolve on %s", len(adapter_paths), HF_MODEL_REPO)
    return adapter_paths


# ── Pod sentinel (poll_pipeline.py contract). Kinds route per plan §7.1:
# epm:results (completion) / epm:failure (HALT_AND_REPORT, CRASH) /
# epm:progress (--plan-only). poll_pipeline._parse_sentinel is kind-agnostic
# (required keys: sentinel_schema_version, kind, version) and its drain glob
# is ``issue-<N>-*.json``, so every kind below is drained + posted as-is. ─────


def _write_sentinel(note_payload: dict | str, *, kind: str = "epm:results", out_root: Path) -> Path:
    """Write an end-of-run sentinel with poll_pipeline's required keys.

    ``kind`` is the full marker kind (``epm:results`` default); the filename
    carries the kind_slug (``:`` → ``_``) per the poll_pipeline.py filename
    convention. ``note_payload``: dict → JSON-dumped; str → written as-is
    (the epm:failure path needs a leading plain-text ``failure_class:`` line
    that failure_classifier.py's FIELD_LINE regex can match).
    """
    sentinel_dir = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))
    if not sentinel_dir.is_dir():
        fallback = out_root / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        log.warning(
            "[sentinel] %s missing (not on a pod?) — writing sentinel to %s",
            sentinel_dir,
            fallback,
        )
        sentinel_dir = fallback
    kind_slug = kind.replace(":", "_")
    path = sentinel_dir / f"issue-610-{kind_slug}-{int(time.time())}.json"
    note = note_payload if isinstance(note_payload, str) else json.dumps(note_payload, indent=2)
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": 610,
        "by": "i610_dispatch",
        "ts": datetime.now(UTC).isoformat(),
        "note": note,
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("[sentinel] wrote %s (kind=%s)", path, kind)
    return path


# Implant-landing gates: the swapped mix landed out of the parent regime at
# the PINNED 63 steps — a science outcome (plan §7.1 item 1: halt + re-plan),
# not a wiring bug. Every other hard gate — reused (c)-(h) plumbing + NEW (i)
# primary-DV wiring — is a code-class failure (plan §7.1 item 2).
_DATA_CLASS_GATES = frozenset({"gate_a_band", "gate_b_sub_saturation"})


def _failure_class_for_gates(gate_payload: dict) -> str:
    """Route a smoke-gate failure to its plan-§7.1 failure_class.

    Returns ``"data"`` when ONLY implant-landing gates (a)/(b) failed
    (out-of-band implant: report + re-plan), ``"code"`` when ANY wiring gate
    — (c)-(h) or (i) — failed (a wiring miss also makes the implant read
    untrustworthy, so code wins on mixed failures). Asserts at least one
    hard gate actually failed.
    """
    hard_fails = [
        k
        for k, v in gate_payload.items()
        if k.startswith("gate_")
        and k != "gate_j_within_parent_range"  # soft, never gates
        and isinstance(v, bool)
        and not v
    ]
    if not hard_fails:
        raise AssertionError("_failure_class_for_gates called with no failed hard gate")
    return "data" if all(k in _DATA_CLASS_GATES for k in hard_fails) else "code"


def _write_failure_sentinel(
    *,
    verdict: str,
    failure_class: str,
    detail: dict,
    mode: str,
    out_root: Path,
    gpu_hours_used: float,
) -> Path:
    """Plan §7.1 HALT_AND_REPORT / crash sentinel: kind ``epm:failure``.

    The note opens with a plain-text ``failure_class: <code|data>`` line
    (failure_classifier.py FIELD_LINE matches ``^failure_class: <x>$`` —
    a JSON-quoted ``"failure_class"`` key alone would NOT match), followed
    by the full diagnostic JSON (verdict, gate table, uploaded-evidence
    paths).
    """
    if failure_class not in ("code", "data", "infra"):
        raise ValueError(f"invalid failure_class {failure_class!r}")
    diagnostic = {
        "failure_class": failure_class,
        "mode": mode,
        "verdict": verdict,
        **detail,
        "git_commit": _git_sha(),
        "gpu_hours_used": gpu_hours_used,
        "gpu_hours_budgeted": GPU_HOURS_BUDGETED,
    }
    note = f"failure_class: {failure_class}\n" + json.dumps(diagnostic, indent=2)
    return _write_sentinel(note, kind="epm:failure", out_root=out_root)


def _cell_eval_numbers(results: list[dict]) -> dict[str, dict]:
    """Pod-side per-cell terminal summary (raw reads; the registered centered
    comparison runs on the VM post-teardown in analyze.py)."""
    out: dict[str, dict] = {}
    for r in results:
        traj = Path(r["trajectory_path"])
        if not traj.exists():
            continue
        ck = _terminal_checkpoint(json.loads(traj.read_text()))
        src_dg = float(ck["source_self"]["delta_g_mean"])
        per_persona = {}
        for persona in (*EXTRA_EVAL_PERSONAS,):
            recs = ck["held_out"].get(persona)
            if recs:
                vals = [float(leaf["delta_g"]) for leaf in recs.values()]
                mean_dg = sum(vals) / len(vals)
                per_persona[persona] = {
                    "delta_g_mean": mean_dg,
                    "normalized": mean_dg / src_dg if src_dg else None,
                }
        out[f"{r['cell_slug']}_seed{r['seed']}"] = {
            "source_dg_mean_nats": src_dg,
            "extra_eval_personas": per_persona,
            "realized_terminal_step": r.get("realized_terminal_step"),
        }
    return out


# ── Main. ────────────────────────────────────────────────────────────────────


def main(
    *,
    mode: str,
    seeds: str,
    n_gpus: int,
    max_parallel: int,
    plan_only: bool = False,
    no_upload: bool = False,
) -> int:
    """Run the #610 unified smoke=sweep pipeline. Returns the shell exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    # uv run python does NOT auto-load .env; subprocesses inherit THIS env
    # (the #397 round-10' dispatcher-env incident class).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    # Subprocesses inherit the project; run_one_cell's setdefault never
    # overrides a preset value.
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    t_start = time.time()
    log.info(
        "[phase=start] mode=%s seeds=%r n_gpus=%d max_parallel=%d epochs=%d (PINNED) "
        "plan_only=%s host=%s",
        mode,
        seeds,
        n_gpus,
        max_parallel,
        EPOCHS_PINNED,
        plan_only,
        socket.gethostname(),
    )
    out_root = _output_root()
    data_root = _data_root()
    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    def _fail_sentinel(verdict: str, failure_class: str, detail: dict, rc: int) -> int:
        """HALT_AND_REPORT/crash path → kind epm:failure (plan §7.1), never epm:results."""
        _write_failure_sentinel(
            verdict=verdict,
            failure_class=failure_class,
            detail=detail,
            mode=mode,
            out_root=out_root,
            gpu_hours_used=round((time.time() - t_start) / 3600.0 * n_gpus, 2),
        )
        return rc

    # ── Phase 0a: marker tokenizer invariant (in-process, before anything). ──
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_tokenization(tokenizer)
    log.info("[phase=marker_check] %r → id=%d (OK)", MARKER_TEXT, EXPECTED_MARKER_TOKEN_ID)

    # ── Phase 0b: sha256-pinned inherited inputs + bank-hash check. ──────────
    _prefetch_inherited_artifacts(_i472_data_root())
    persona_bank, _r_train, _q_train = _load_bank_and_r_train()
    log.info("[phase=load_bank] %d personas", len(persona_bank))

    manifest_path = _parent_manifest_path()
    manifest = load_manifest(manifest_path)
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        _content_hash,
    )

    bank_hash = _content_hash(persona_bank)
    if bank_hash != manifest["bank_content_hash"]:
        raise RuntimeError(
            f"persona-bank content hash {bank_hash[:12]} != parent manifest's "
            f"{manifest['bank_content_hash'][:12]} — refusing to train."
        )

    # ── Phase 0c: the #610 spec + committed-design consistency. ──────────────
    spec = build_610_spec(manifest)
    design_path = _design_path()
    if not design_path.exists():
        raise FileNotFoundError(
            f"committed design.json missing at {design_path} — it must be built on the "
            "VM and committed BEFORE training (plan §4.4): uv run python -m "
            "explore_persona_space.experiments.default_dose_610.cells"
        )
    assert_design_matches(json.loads(design_path.read_text()), manifest, spec)
    log.info("[phase=spec] %s panel=%s (design.json consistent)", spec.slug, list(spec.panel))

    # ── Phase 1: spec_iter — smoke pair first, ALWAYS. ───────────────────────
    seed_list = [int(t) for t in seeds.split(",") if t.strip()]
    unknown = [s for s in seed_list if s not in SEEDS]
    if unknown:
        raise ValueError(f"--seeds contains non-registered seeds {unknown}; pinned: {SEEDS}")
    smoke_seed = SEEDS[0]
    if smoke_seed not in seed_list:
        raise ValueError(f"--seeds must include the smoke seed {smoke_seed}; got {seed_list}")
    smoke_iter: list[tuple[CellSpec600, int]] = [(spec, smoke_seed)]
    rest_iter: list[tuple[CellSpec600, int]] = (
        [(spec, s) for s in seed_list if s != smoke_seed] if mode == "full" else []
    )
    log.info(
        "[phase=plan] smoke pair: %s; remaining pairs: %s",
        [(s.slug, sd) for s, sd in smoke_iter],
        [(s.slug, sd) for s, sd in rest_iter],
    )
    if plan_only:
        # Plan-only launches NOTHING — a results-shaped sentinel would let the
        # poller post epm:results for a run with no results. epm:progress is
        # the honest kind: validation evidence, drained + posted as progress.
        _write_sentinel(
            {
                "mode": "plan_only",
                "requested_mode": mode,
                "pairs": [(s.slug, sd) for s, sd in smoke_iter + rest_iter],
                "panel": list(spec.panel),
                "git_commit": _git_sha(),
            },
            kind="epm:progress",
            out_root=out_root,
        )
        log.info("[phase=done] plan-only: validated, nothing launched")
        return 0

    common = dict(
        n_gpus=n_gpus,
        max_parallel=max_parallel,
        epochs=EPOCHS_PINNED,
        manifest_path=manifest_path,
        out_root=out_root,
        data_root=data_root,
        script_name=RUN_CELL_SCRIPT,
    )

    # ── Phase 2: the smoke pair (the sweep's first cell — zero marginal cost).
    log.info("[phase=train_eval_start] smoke pair (1 pair)")
    results, failures = _run_cells_subprocess(smoke_iter, **common)
    if failures or not results:
        log.error("[phase=smoke_gate_fail] smoke cell crashed: %s", failures)
        # A crashed subprocess is a wiring/code failure (plan §7.1 item 2).
        return _fail_sentinel("CRASH", "code", {"failures": failures}, rc=2)

    # ── Phase 3: gates (a)-(h) + (i) hard + (j) soft. ────────────────────────
    r = results[0]
    cell_dir = out_root / "sweep" / r["cell_slug"] / f"seed_{r['seed']}"
    gate_payload = check_smoke_gates_610(
        trajectory_path=Path(r["trajectory_path"]),
        band_trajectory_path=Path(r["band_trajectory_path"]),
        verify_payload=json.loads((cell_dir / "panel_verify.json").read_text()),
        collator_payload=json.loads((cell_dir / "collator_gate.json").read_text()),
        checkpoint_index=r["checkpoint_index"],
        expected_steps=EXPECTED_STEPS_PER_EPOCH * EPOCHS_PINNED,
        panel_personas=r["panel"],
        smoke_out_path=out_root / "smoke" / "smoke_gate.json",
    )
    if not gate_payload["all_gates_passed"]:
        # Kill criterion §7.1: HALT AND REPORT — never epochs-ladder (matched
        # 63 steps with the reused parent arm is load-bearing). Upload what
        # exists first (checkpoint-per-phase; the smoke cell's artifacts are
        # evidence either way).
        log.error(
            "[phase=smoke_gate_fail] %s",
            {k: v for k, v in gate_payload.items() if k.startswith("gate_")},
        )
        if not no_upload:
            _upload_phase_610(out_root, data_root, design_path)
        # failure_class per plan §7.1: ONLY implant-landing gates (a)/(b)
        # failed → data (out-of-band implant: report + re-plan); any wiring
        # gate — (c)-(h) reused plumbing or (i) primary-DV existence — → code.
        return _fail_sentinel(
            "HALT_AND_REPORT",
            _failure_class_for_gates(gate_payload),
            {
                "reason": "smoke gates failed at the PINNED 63 steps; plan §7.1 forbids the "
                "epochs ladder (matched steps with the reused parent arm are load-bearing)",
                "smoke_gate": gate_payload,
                "expected_band_nats": list(SOURCE_DG_BAND_NATS),
                # Uploaded-evidence pointers (the smoke cell's artifacts are
                # evidence either way; uploaded above unless --no-upload).
                "output_root": str(out_root),
                "smoke_trajectory_path": r["trajectory_path"],
                "uploaded_to_hf_data_prefix": None if no_upload else HF_DATA_PREFIX,
            },
            rc=2,
        )

    # ── Phase 4: remaining seeds through the IDENTICAL path (--full only). ───
    if rest_iter:
        log.info("[phase=train_eval_rest] %d pairs", len(rest_iter))
        rest_results, rest_failures = _run_cells_subprocess(rest_iter, **common)
        results += rest_results
        failures += rest_failures

    # ── Phase 5: uploads + Hub verification of the inline adapter uploads. ───
    adapter_paths: dict[str, str] = {}
    if no_upload:
        log.warning("[phase=upload] SKIPPED (--no-upload; local smoke only)")
    else:
        _upload_phase_610(out_root, data_root, design_path)
        adapter_paths = _verify_adapter_uploads(results)

    # ── Phase 6: sentinel (full epm:results payload contract). ───────────────
    cell_keys = [f"{r['cell_slug']}_seed{r['seed']}" for r in results]
    plan_deviations: list[str] = []
    if not gate_payload.get("gate_j_within_parent_range", True):
        plan_deviations.append(
            f"gate (j) soft flag: smoke villain ΔG "
            f"{gate_payload['gate_j_detail']['source_dg_mean_nats']:.2f} nats outside the "
            f"parent-range band {CHASSIS_DG_SOFT_RANGE_NATS} (inside gate (a); proceeding, "
            "flagged for analysis)"
        )
    note = {
        "mode": mode,
        "verdict": "OK" if not failures else "PARTIAL",
        "epochs": EPOCHS_PINNED,
        "n_pairs": len(smoke_iter) + len(rest_iter),
        "n_completed": len(results),
        "n_skipped_existing": sum(1 for x in results if x.get("skipped_existing")),
        "failures": failures,
        "smoke_gate": gate_payload,
        "eval_numbers": _cell_eval_numbers(results),
        "eval_paths": {f"{x['cell_slug']}_seed{x['seed']}": x["trajectory_path"] for x in results},
        "output_root": str(out_root),
        "worktree_path": str(Path.cwd()),
        "final_commit_sha": _git_sha(),
        "gpu_hours_used": round((time.time() - t_start) / 3600.0 * n_gpus, 2),
        "gpu_hours_budgeted": GPU_HOURS_BUDGETED,
        "plan_deviations": plan_deviations,
        "reproducibility_card": {
            "base_model": BASE_MODEL,
            "hf_model_repo": HF_MODEL_REPO,
            "adapter_paths": adapter_paths,
            "wandb_project": WANDB_PROJECT,
            "wandb_run_names": [f"{RUN_NAME_PREFIX}{k}" for k in cell_keys],
            "hf_data_repo": HF_DATA_REPO,
            "hf_data_prefix": HF_DATA_PREFIX,
        },
    }
    _write_sentinel(note, kind="epm:results", out_root=out_root)

    if failures:
        log.error("[phase=cells_failed] %d failed: %s", len(failures), failures)
        return 4
    log.info("[phase=done] all %d (cell, seed) pairs finished successfully", len(results))
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    """argparse entrypoint (used by ``scripts/i610_dispatch.py``)."""
    p = argparse.ArgumentParser(description="Task #610 unified smoke=sweep dispatcher")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--smoke",
        action="store_true",
        help="ONE (cell, seed) — the no-default cell at seed 42 — through the identical "
        "subprocess path, then gates (a)-(j).",
    )
    g.add_argument(
        "--full",
        action="store_true",
        help="The smoke pair → gates → remaining --seeds through the identical path, "
        "in one invocation.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seed VALUES (⊆ pinned {42,137,219}; must include 42).",
    )
    p.add_argument("--n-gpus", type=int, default=4, help="Physical GPUs available.")
    p.add_argument("--max-parallel", type=int, default=3, help="Concurrent cells cap.")
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate marker/bank/manifest/design + print the launch plan; spawn nothing.",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the HF upload phase (LOCAL runs only — pods must upload).",
    )
    args = p.parse_args(argv)
    return main(
        mode="smoke" if args.smoke else "full",
        seeds=args.seeds,
        n_gpus=args.n_gpus,
        max_parallel=args.max_parallel,
        plan_only=args.plan_only,
        no_upload=args.no_upload,
    )


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
