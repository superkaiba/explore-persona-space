# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek ΔG + Qwen marker " ※" + × intentional
"""Task #505 §5.5 — unified smoke=sweep dispatcher.

One dispatcher, one per-cell subprocess shape, one logging surface. Smoke runs
``main(smoke=True)`` = ``--cells 1 --seeds 1`` on the full-set arm; the sweep
runs the same code path with ``--cells 8 --seeds 3`` (8 arms × 3 seeds = 24
trained adapters). The §5.5 (a)-(h) gates fire BEFORE any training spawn.

Per-cell training uses ``contrastive_neg_geometry_472.train_cell.train_one_cell``
(already accepts ``marker_suppress_at_post_response_slot`` and
``marker_im_end_token_id`` since #477 v6 — see train_cell.py:365-368) and
threads BOTH §5.1 flags through into the trainer config.

Per-cell eval calls ``leave_one_out_505.eval_trajectory_505.run_trajectory_eval_with_guard``
which (a) runs the #472 trajectory eval (on-policy gen + DV-A + DV-B KL) and
then (b) calls the #477 ``assert_adapter_actually_applied`` guard at the
headline checkpoint frac 0.50. The §5.5 gate (f) positive-control gates the
sweep on a clean guard call on the smoke cell.

The dispatcher is INTENDED to run on a 4× H100 pod with 4 cells in parallel
(per ``+gpu_id=N`` per the codebase CVD-clobber rule, sft.py:477). For the
tiny-local smoke we run 1 cell × 1 seed in-process on whatever device the
caller has visible.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    HF_DATA_REPO,
    MARKER_TEXT,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
    train_one_cell,
)
from explore_persona_space.experiments.leave_one_out_505 import (
    BASE_MODEL,
    CELL_SPECS,
    EPOCHS,
    FALLBACK_LORA_R,
    HEADLINE_CHECKPOINT_FRAC,
    HF_DATA_PREFIX_INHERIT,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_R,
    MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS_GEN,
    QWEN_IM_END_TOKEN_ID,
    SEEDS,
    SIMILARITY_LAYERS_TO_BUILD,
    SOURCE_DG_EXPECTED_BAND_NATS,
    SOURCE_DG_SATURATION_CEILING_NATS,
    SOURCE_EMISSION_SATURATION_THRESHOLD,
    SOURCE_PERSONA,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
)
from explore_persona_space.experiments.leave_one_out_505.build_pv_centroids import (
    build_pv_centroids,
)
from explore_persona_space.experiments.leave_one_out_505.build_training_data import (
    build_cell_505,
)
from explore_persona_space.experiments.leave_one_out_505.eval_trajectory_505 import (
    run_trajectory_eval_with_guard,
)
from explore_persona_space.experiments.leave_one_out_505.panel_coverage import (
    load_inherited_l10_cos,
    run_panel_coverage_gate,
    write_gate_payload,
)

log = logging.getLogger("issue_505.dispatch")


# ── Dispatcher constants (paths). ───────────────────────────────────────────


def _output_root() -> Path:
    """Resolve the eval-results root. Honors EPM_OUTPUT_ROOT for smoke tests."""
    return Path(os.environ.get("EPM_OUTPUT_ROOT", "eval_results/issue_505"))


def _data_root() -> Path:
    return Path(os.environ.get("EPM_DATA_ROOT", "data/issue_505"))


def _l10_centroid_dir() -> Path:
    """Layer-10 centroid bundle path (inherited from #472)."""
    return Path(os.environ.get("EPM_I472_DATA_ROOT", "data/issue_472"))


# ── Smoke gate readers ──────────────────────────────────────────────────────


def _read_source_dg_at_frac(trajectory_path: Path, frac: float = HEADLINE_CHECKPOINT_FRAC) -> dict:
    """Pull source-self ΔG mean + emission_p from a written trajectory.json."""
    payload = json.loads(trajectory_path.read_text())
    target_2 = f"{frac:.2f}"
    target_4 = f"{frac:.4f}"

    def _frac_match(ckpt: dict) -> bool:
        raw = ckpt.get("frac")
        if isinstance(raw, str):
            return raw in (target_2, target_4)
        if isinstance(raw, (int, float)):
            return abs(float(raw) - frac) < 1e-4
        return False

    ckpt = next((c for c in payload["checkpoints"] if _frac_match(c)), None)
    if ckpt is None:
        raise KeyError(f"trajectory has no checkpoint at frac={frac!r}.")
    src = ckpt.get("source_self", {})
    return {
        "delta_g_mean": float(src.get("delta_g_mean", float("nan"))),
        "emission_p": float(src.get("emission_p", float("nan"))),
        "n_held_out_collapsed": int(ckpt.get("n_held_out_collapsed", 0)),
        "eval_guard_diagnostic": ckpt.get("eval_guard_diagnostic", {}),
    }


def _check_smoke_gates(
    trajectory_path: Path,
    *,
    smoke_out_path: Path,
    frac: float = HEADLINE_CHECKPOINT_FRAC,
) -> dict:
    """Run the §5.5 (a)-(e) gates against the written smoke trajectory."""
    diag = _read_source_dg_at_frac(trajectory_path, frac=frac)
    dg = diag["delta_g_mean"]
    em = diag["emission_p"]
    band_low, band_high = SOURCE_DG_EXPECTED_BAND_NATS
    sat = SOURCE_DG_SATURATION_CEILING_NATS
    em_sat = SOURCE_EMISSION_SATURATION_THRESHOLD
    gate_a = band_low <= dg <= band_high
    gate_b = em <= em_sat and dg <= sat
    guard_verdict = diag["eval_guard_diagnostic"].get("guard_verdict")
    gate_f = guard_verdict in {"pass_real_signal", "pass_some_emission"}
    payload = {
        "frac": frac,
        "source_dg_mean_nats": dg,
        "source_emission_p": em,
        "n_held_out_collapsed": diag["n_held_out_collapsed"],
        "expected_band_nats": list(SOURCE_DG_EXPECTED_BAND_NATS),
        "saturation_ceiling_nats": sat,
        "emission_saturation_threshold": em_sat,
        "gate_a_band_passed": gate_a,
        "gate_b_sub_saturated_passed": gate_b,
        "gate_f_guard_positive_control_passed": gate_f,
        "guard_verdict": guard_verdict,
        "trajectory_path": str(trajectory_path),
        "all_gates_passed": gate_a and gate_b and gate_f,
    }
    smoke_out_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[smoke-gate] dg=%.2f band=%s; emission=%.2f≤%.2f? %s; guard=%s; all_passed=%s",
        dg,
        f"[{band_low}, {band_high}]",
        em,
        em_sat,
        gate_b,
        guard_verdict,
        payload["all_gates_passed"],
    )
    return payload


# ── Cell training + eval (per-cell, GPU-pinned). ───────────────────────────


def _train_and_eval_one_cell(
    *,
    cell_slug: str,
    seed: int,
    non_default_negatives: list[str],
    panel: list[str],
    persona_bank: dict[str, str],
    r_train: dict[str, dict[str, dict]],
    r_eval: dict[str, dict[str, dict]],
    q_train: list[str],
    q_eval: list[str],
    output_root: Path,
    data_root: Path,
    gpu_id: int = 0,
    compute_kl: bool = True,
    max_lora_rank: int = max(LORA_R, FALLBACK_LORA_R),
) -> dict:
    """Build training data, train the LoRA, run trajectory eval + guard, return paths.

    ``max_lora_rank`` defaults to ``max(LORA_R, FALLBACK_LORA_R)`` (= 32) so the
    vLLM cap accommodates BOTH the §5.1 primary anchor (rank 16) AND the §5.5
    fallback anchor (rank 32). The trained adapter rank is still pinned to the
    plan recipe via ``lora_r_override=LORA_R``; the cap is just the upper bound
    on what vLLM will accept. Round-6 fix: a smaller cap (default 16) rejects
    the fallback anchor at load time; a smaller cap that matches LORA_R
    rejects the trained adapter when the dispatcher forgot to thread the
    rank override (the original round-6 crash).
    """
    train_jsonl = data_root / f"{cell_slug}_seed{seed}.jsonl"
    build_cell_505(
        cell_slug=cell_slug,
        output_path=train_jsonl,
        r_train=r_train,
        non_default_negatives=non_default_negatives,
        q_train=q_train,
        persona_bank=persona_bank,
        seed=seed,
    )

    cell_out = output_root / "sweep" / cell_slug / f"seed_{seed}"
    cell_out.mkdir(parents=True, exist_ok=True)
    adapter_out = cell_out / "adapter"
    ckpt_root = cell_out / "checkpoints"
    train_result = train_one_cell(
        cell_slug=cell_slug,
        seed=seed,
        train_jsonl=train_jsonl,
        output_dir=adapter_out,
        ckpt_root=ckpt_root,
        fractions=TRAJECTORY_CHECKPOINT_FRACTIONS,
        base_model=BASE_MODEL,
        report_to="wandb",
        gpu_id=gpu_id,
        # ── #505 anchor recipe override (plan §5.1) — LOAD-BEARING ──────────
        # ``train_one_cell`` defaults to the #472 constants (rank 32 / alpha
        # 64 / lr 1e-5), which is the saturating anchor #505 explicitly
        # avoids. Without these overrides the trained adapter is rank 32 and
        # vLLM rejects it at the rank-16 eval cap; the recipe gradient the
        # leave-one-out experiment hinges on collapses into the saturating
        # regime where every cell's source ΔG hits the ceiling. Round-6
        # crash signature: ``ValueError: LoRA rank 32 is greater than
        # max_lora_rank 16`` at eval_trajectory.py:179. lr / lora_alpha /
        # epochs land on this same code path; lr is the most directly
        # outcome-changing of the three (#505=5e-6 vs #472=1e-5).
        lora_r_override=LORA_R,
        lora_alpha_override=LORA_ALPHA,
        lr_override=LEARNING_RATE,
        epochs_override=EPOCHS,
        # The load-bearing slot-fix conjunction (§5.1, §5.5 gate h).
        marker_suppress_at_post_response_slot=MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        marker_im_end_token_id=QWEN_IM_END_TOKEN_ID,
        # Hf path under #505's own subfolder.
        hf_path_in_repo_override=f"adapters/issue_505/{cell_slug}_seed{seed}",
        run_name_override=f"issue505_{cell_slug}_seed{seed}",
    )
    log.info(
        "[%s seed=%d] adapter trained; checkpoints=%s",
        cell_slug,
        seed,
        train_result.get("checkpoint_index"),
    )

    # No-negatives arm runs train + eval but the eval will read panel ΔG → some
    # leakage everywhere (positive-only training); still needed as a control.
    checkpoint_specs = [
        {"frac": float(k), "step": v.get("step"), "adapter_path": v.get("path")}
        for k, v in train_result["checkpoint_index"].items()
    ]

    eval_personas = {p: persona_bank[p] for p in panel}
    out_path = cell_out / "trajectory.json"
    run_trajectory_eval_with_guard(
        cell_slug=cell_slug,
        seed=seed,
        checkpoint_specs=checkpoint_specs,
        eval_personas=eval_personas,
        eval_questions=q_eval,
        source=SOURCE_PERSONA,
        source_prompt=persona_bank[SOURCE_PERSONA],
        out_path=out_path,
        base_model=BASE_MODEL,
        max_new_tokens=MAX_NEW_TOKENS_GEN,
        headline_frac=HEADLINE_CHECKPOINT_FRAC,
        compute_kl=compute_kl,
        max_lora_rank=max_lora_rank,
        # Round-10 fix: override #472's vLLM ``max_model_len`` default of 2048
        # (round-9 crash: decoder prompt 2050 > 2048 at frac 0.50). 4096 = 2×
        # MAX_NEW_TOKENS_GEN covers worst-case prefix + R_j + marker context.
        max_model_len=MAX_MODEL_LEN,
    )
    return {
        "cell_slug": cell_slug,
        "seed": seed,
        "trajectory_path": str(out_path),
        "adapter_dir": str(adapter_out),
        "checkpoint_index": train_result["checkpoint_index"],
    }


# ── Phase 0 helpers: persona bank + R artifacts + L10 inheritance. ──────────


def _prefetch_inherited_artifacts(i472_root: Path) -> None:
    """Pre-fetch the #472 inherited artifacts from the HF data repo when local copies are missing.

    On a fresh pod that bypasses the #472 bootstrap, ``persona_bank.json`` +
    ``R_train.json`` + ``R_eval.json`` + ``centroids_L10.pt`` may all be
    absent — without them Phase 0b (load) + Phase 1 (panel gate) crash with
    FileNotFoundError before anything useful happens. This helper inspects
    each path and only downloads what's missing (idempotent; safe to call
    every run).

    Fails loud if HF_TOKEN is missing AND any artifact is missing — there is
    no fallback path for these inputs.
    """
    from huggingface_hub import hf_hub_download

    targets = [
        ("persona_bank.json", f"{HF_DATA_PREFIX_INHERIT}/geometry/persona_bank.json"),
        ("on_policy_R/R_train.json", f"{HF_DATA_PREFIX_INHERIT}/on_policy_R/R_train.json"),
        ("on_policy_R/R_eval.json", f"{HF_DATA_PREFIX_INHERIT}/on_policy_R/R_eval.json"),
        ("centroids_L10.pt", f"{HF_DATA_PREFIX_INHERIT}/geometry/centroids_L10.pt"),
    ]
    missing = [(local, remote) for local, remote in targets if not (i472_root / local).exists()]
    if not missing:
        log.info(
            "[phase=prefetch] all inherited #472 artifacts already on disk under %s", i472_root
        )
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        names = [m[0] for m in missing]
        raise RuntimeError(
            f"inherited #472 artifacts missing locally and HF_TOKEN unset → cannot prefetch: "
            f"{names}. Either provision HF_TOKEN or run the #472 bootstrap before #505."
        )

    i472_root.mkdir(parents=True, exist_ok=True)
    (i472_root / "on_policy_R").mkdir(parents=True, exist_ok=True)
    for local_rel, remote in missing:
        log.info("[phase=prefetch] %s -> %s/%s", local_rel, HF_DATA_REPO, remote)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=remote,
            repo_type="dataset",
            token=token,
        )
        # Copy into the canonical path so the legacy reader code below stays untouched.
        target = i472_root / local_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != Path(downloaded).resolve():
            import shutil

            shutil.copyfile(downloaded, target)
        log.info("[phase=prefetch] OK %s (%d bytes)", target, target.stat().st_size)


def _load_persona_bank_and_r() -> tuple[dict[str, str], dict, dict, list[str], list[str]]:
    """Load the #472 persona bank + R_train + R_eval from the local cache.

    Mirrors the #472 dispatcher: the persona bank lives at
    ``data/issue_472/persona_bank.json``; R artifacts under
    ``data/issue_472/on_policy_R/{R_train,R_eval}.json``. ``_prefetch_inherited_artifacts``
    (Phase 0a) ensures these are present from the HF data repo before this
    loader runs, so the missing-file branch below should only trigger when
    HF_TOKEN was unset AND no local copies existed — and the prefetch already
    raises in that case.

    Both ``persona_bank.json`` and ``R_{train,eval}.json`` are STRUCTURED
    payload dicts published by #472 — ``persona_bank.json`` carries the actual
    name→prompt map under ``payload['personas']`` and ``R_*.json`` carries the
    actual ``completions[persona][q]`` under ``payload['completions']``. Going
    through the canonical ``load_persona_bank`` / ``load_r_artifact`` helpers
    unwraps + validates the schema; a raw ``json.loads()`` here leaks metadata
    keys like ``'schema_version'`` into the bank/R iteration and crashes Phase
    1 with ``KeyError: 'schema_version'`` at panel_coverage.py:149 (#505
    round-3 v3 launch, 2026-06-05).
    """
    # Local imports keep ruff from auto-stripping module-top imports that are
    # only referenced inside this loader (the import is the actual usage).
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        load_r_artifact,
    )

    i472 = _l10_centroid_dir()
    bank_path = i472 / "persona_bank.json"
    if not bank_path.exists():
        raise FileNotFoundError(
            f"persona bank missing at {bank_path}. Pre-step: download from "
            "superkaiba1/explore-persona-space-data/"
            "issue472_neg_geometry/geometry/persona_bank.json "
            "(or call _prefetch_inherited_artifacts before this loader)."
        )
    persona_bank = load_persona_bank(bank_path)

    r_root = i472 / "on_policy_R"
    r_train = load_r_artifact(r_root / "R_train.json")
    r_eval = load_r_artifact(r_root / "R_eval.json")
    # Q_train / Q_eval are the keys of the first persona's per-q dict.
    any_p = next(iter(r_train))
    q_train = sorted(r_train[any_p])
    any_pe = next(iter(r_eval))
    q_eval = sorted(r_eval[any_pe])
    return persona_bank, r_train, r_eval, q_train, q_eval


# ── Main entrypoint. ─────────────────────────────────────────────────────────


def main(
    *,
    smoke: bool,
    cells: int,
    seeds: int,
    gpu_id: int,
    compute_kl: bool,
    output_root: Path | None = None,
    data_root: Path | None = None,
) -> int:
    """Run the unified §5.5 smoke=sweep pipeline. Returns the shell exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log.info(
        "[phase=start] smoke=%s cells=%d seeds=%d gpu_id=%d host=%s",
        smoke,
        cells,
        seeds,
        gpu_id,
        socket.gethostname(),
    )

    out_root = output_root or _output_root()
    data_root_ = data_root or _data_root()
    out_root.mkdir(parents=True, exist_ok=True)
    data_root_.mkdir(parents=True, exist_ok=True)

    # ── Phase 0a: marker tokenizer invariant. ────────────────────────────────
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"[invariant] marker {MARKER_TEXT!r} tokenizes to {encoded}, expected "
            f"[{EXPECTED_MARKER_TOKEN_ID}]. Tokenizer drift — aborting."
        )
    log.info("[phase=marker_check] %s → id=%d (OK)", MARKER_TEXT, EXPECTED_MARKER_TOKEN_ID)

    # ── Phase 0a-prefetch: download #472 inherited artifacts when local cache is empty. ─
    # Fresh pods that skipped the #472 bootstrap end up here with no persona
    # bank / R artifacts / centroids_L10 on disk; the prefetch is idempotent
    # (only downloads what's missing) and fail-loud if HF_TOKEN is also unset.
    _prefetch_inherited_artifacts(_l10_centroid_dir())

    # ── Phase 0b: persona bank + R artifacts. ────────────────────────────────
    persona_bank, r_train, r_eval, q_train, q_eval = _load_persona_bank_and_r()
    log.info(
        "[phase=load_bank] %d personas; |Q_train|=%d, |Q_eval|=%d",
        len(persona_bank),
        len(q_train),
        len(q_eval),
    )

    # ── Phase 1: §5.4 panel coverage gate. ───────────────────────────────────
    cos_l10 = load_inherited_l10_cos(_l10_centroid_dir() / "centroids_L10.pt")
    gate_payload = run_panel_coverage_gate(persona_bank=persona_bank, cos_matrix_l10=cos_l10)
    write_gate_payload(gate_payload, out_root / "panel_coverage.json")
    non_default = gate_payload["non_default_negatives"]
    panel = gate_payload["panel"]
    log.info(
        "[phase=panel_gate] PASS — K-set=%s, panel=%d personas", gate_payload["k_set"], len(panel)
    )

    # ── Phase 2: §5.7 persona-vectors centroid build (L7 / 14 / 21 / 27). ────
    centroid_pv_dir = data_root_ / "centroids_pv"
    build_pv_centroids(
        persona_bank=persona_bank,
        layers=SIMILARITY_LAYERS_TO_BUILD,
        out_dir=centroid_pv_dir,
        skip_existing=True,
    )

    # ── Phase 3: smoke or full sweep. ────────────────────────────────────────
    if smoke:
        spec_iter = [(CELL_SPECS[0], SEEDS[0])]
    else:
        # CELL_SPECS has 8 entries; --cells caps the number we iterate per seed.
        cap = min(cells, len(CELL_SPECS))
        seed_list = SEEDS[:seeds]
        spec_iter = [(spec, seed) for spec in CELL_SPECS[:cap] for seed in seed_list]

    log.info("[phase=train_eval_start] %d (cell, seed) pairs to run", len(spec_iter))
    # Recovery option (r11): skip cells whose trajectory.json already exists.
    # Set EPM_SKIP_EXISTING=1 when relaunching a partially-completed sweep
    # after a transient crash (e.g. /workspace EIO mid-cell, OOM). The dispatcher
    # has no built-in resume; this env-gated skip is the cheapest fix.
    skip_existing = os.environ.get("EPM_SKIP_EXISTING", "").lower() in {"1", "true", "yes"}
    results: list[dict] = []
    for spec, seed in spec_iter:
        slug = spec[0]
        if skip_existing:
            traj_path = out_root / "sweep" / slug / f"seed_{seed}" / "trajectory.json"
            if traj_path.exists():
                log.info("[skip-existing] %s_seed%d: trajectory.json present", slug, seed)
                continue
        # The no-negatives control: data builder skips negative rows.
        result = _train_and_eval_one_cell(
            cell_slug=slug,
            seed=seed,
            non_default_negatives=non_default,
            panel=panel,
            persona_bank=persona_bank,
            r_train=r_train,
            r_eval=r_eval,
            q_train=q_train,
            q_eval=q_eval,
            output_root=out_root,
            data_root=data_root_,
            gpu_id=gpu_id,
            compute_kl=compute_kl,
            # vLLM cap accommodates BOTH primary anchor (LORA_R=16) and §5.5
            # fallback anchor (FALLBACK_LORA_R=32). Trained adapter rank is
            # still pinned to plan §5.1 via lora_r_override in
            # _train_and_eval_one_cell. Round-6 fix.
            max_lora_rank=max(LORA_R, FALLBACK_LORA_R),
        )
        results.append(result)
        # Persist a tiny per-cell completion sentinel (resume-safe).
        sentinel = out_root / "sweep" / slug / f"seed_{seed}" / "done.json"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            json.dumps({**result, "timestamp_utc": datetime.now(UTC).isoformat()}, indent=2)
        )

    # ── Phase 4: smoke gate (a)-(f). ─────────────────────────────────────────
    if smoke:
        smoke_traj = Path(results[0]["trajectory_path"])
        gate = _check_smoke_gates(smoke_traj, smoke_out_path=out_root / "smoke" / "smoke_gate.json")
        if not gate["all_gates_passed"]:
            log.error("[phase=smoke_gate_fail] %s", gate)
            return 2
        log.info("[phase=smoke_gate_pass] %s", gate)

    # ── Phase 5: auto-fire analyze at sweep end. ─────────────────────────────
    # Only for full sweeps — smoke runs a single cell so there's no Δ-Leakage
    # to compute. Failures land as a logged warning + non-zero shell exit so
    # the orchestrator notices but the trained adapters + trajectory.json are
    # already persisted upstream (Phase 3 sentinels) and recoverable.
    if not smoke:
        from explore_persona_space.experiments.leave_one_out_505.analyze import analyze_505

        analysis_dir = out_root / "analysis"
        log.info("[phase=analyze_start] → %s", analysis_dir)
        try:
            analyze_505(
                panel_gate_path=out_root / "panel_coverage.json",
                sweep_dir=out_root / "sweep",
                centroid_dir_l10=_l10_centroid_dir(),
                centroid_dir_pv=centroid_pv_dir,
                analysis_dir=analysis_dir,
            )
            log.info("[phase=analyze_done] artifacts under %s", analysis_dir)
        except Exception as e:
            log.exception("[phase=analyze_fail] analyze_505 raised — sweep artifacts are persisted")
            return 3 if not isinstance(e, KeyboardInterrupt) else 130

    log.info("[phase=done] all %d cells finished successfully", len(spec_iter))
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    """argparse entrypoint (used by ``scripts/issue505_dispatch.py``)."""
    p = argparse.ArgumentParser(description="Task #505 unified smoke=sweep dispatcher")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run the §5.5 smoke gate: 1 cell × 1 seed on the full-set arm.",
    )
    p.add_argument(
        "--cells",
        type=int,
        default=len(CELL_SPECS),
        help="Number of arms to run (cap). Sweep default = all 8 cell specs.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=len(SEEDS),
        help="Number of seeds to run (cap; default 3).",
    )
    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Physical GPU index for this process. Per-cell parallelism is achieved "
        "by spawning N processes with different --gpu-id values on a multi-GPU pod.",
    )
    p.add_argument(
        "--no-kl",
        action="store_true",
        help="Skip the DV-B KL phase (smoke speed-up; sweep should compute KL).",
    )
    args = p.parse_args(argv)
    return main(
        smoke=args.smoke,
        cells=args.cells,
        seeds=args.seeds,
        gpu_id=args.gpu_id,
        compute_kl=not args.no_kl,
    )


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
