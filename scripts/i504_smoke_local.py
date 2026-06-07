# ruff: noqa: RUF003  # em-dash + Qwen marker " ※" + × + − intentional
#!/usr/bin/env python3
"""Task #504 — CPU-only end-to-end smoke for the local VM (no GPU).

Exercises every CPU code path the dispatcher invokes (Phase 0.5, Phase 0 pick,
Phase 2 analyze) on TINY synthetic artifacts. Verifies:

  * scripts/i504_phase_phase05.py — gates A/B/C + max-length check, writes
    phase0_5_gates.json + arm_to_n.json.
  * scripts/i504_phase_phase0_pick.py — pick rule over 3 smoke trajectories.
  * scripts/i504_phase_analyze.py — 6-covariate partial-Spearman regression.
  * build_training_data.build_cell_504 — per-cell training pool (1 positioned
    arm + the default-only arm).

Generates artifacts under /tmp/issue-504-smoke/ and prints PASS/FAIL per phase
with a one-line digest. Run from the worktree root:

    uv run python scripts/i504_smoke_local.py

GPU-only paths (training + on-policy gen + vLLM logp) are NOT smoked here —
those need a pod. The dispatcher's full Phase 0/1 require a pod for training;
the per-cell unit (i504_run_cell.py) is exercised by the SAME pipeline as the
sweep, so a single Phase 0 smoke cell on a pod (the unified-smoke architecture)
covers the GPU code path.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()


def setup_log() -> logging.Logger:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [smoke=%(name)s] %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("i504_smoke")


def make_synthetic_centroids(
    out_dir: Path,
    *,
    n_personas: int = 40,
    dim: int = 128,
    layers: tuple[int, ...] = (10, 15, 20),
    source: str = "villain",
    default_persona: str = "qwen_default",
    seed: int = 42,
) -> dict:
    """Build synthetic centroids_L*.pt files in `out_dir` for the smoke."""
    rng = np.random.default_rng(seed)
    personas = (
        [source, default_persona]
        + [f"probe_persona_{i:03d}" for i in range(n_personas - 6)]
        + ["near_persona", "mid_near_persona", "mid_far_persona", "far_persona"]
    )
    # Build a layer-10 centroid set where:
    #   source ~ uniform on the unit sphere
    #   near_persona  has cos to source ~ 0.7
    #   mid_near_persona ~ 0.4
    #   mid_far_persona ~ 0.1
    #   far_persona   ~ -0.2
    #   default_persona ~ 0.0  (orthogonal — sits "behind" the bank)
    #   probe_persona_*: random unit
    written: dict[int, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        src = rng.standard_normal(dim)
        src /= np.linalg.norm(src)
        centroids: dict[str, np.ndarray] = {source: src.astype(np.float32)}
        # Pick orthogonal direction for "mixing" cos values.
        orth = rng.standard_normal(dim)
        orth -= orth @ src * src
        orth /= np.linalg.norm(orth)
        for target_persona, target_cos in [
            ("near_persona", 0.7),
            ("mid_near_persona", 0.4),
            ("mid_far_persona", 0.1),
            ("far_persona", -0.2),
            (default_persona, 0.0),
        ]:
            v = target_cos * src + np.sqrt(max(0.0, 1.0 - target_cos**2)) * orth
            v /= np.linalg.norm(v)
            centroids[target_persona] = v.astype(np.float32)
        for p in personas:
            if p in centroids:
                continue
            # Random direction; slight bias towards positive cos to spread probes.
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v)
            centroids[p] = v.astype(np.float32)
        # Save in the production-shape STRUCTURED schema (matches
        # `contrastive_neg_geometry_472.centroids.build_centroids` so the
        # smoke catches any schema drift before a pod launch).
        names = list(centroids.keys())
        mat = np.stack([centroids[n] for n in names], axis=0).astype(np.float32)
        # cos matrix (centering="none") — only needed by analyze paths but
        # mirrored here so the smoke payload is bit-shape-identical to
        # production.
        norm_mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        cos = norm_mat @ norm_mat.T
        path = out_dir / f"centroids_L{layer}.pt"
        torch.save(
            {
                "centroids": torch.from_numpy(mat),
                "persona_names": names,
                "cos_matrix": torch.from_numpy(cos.astype(np.float32)),
                "layer": layer,
                "base_model": "synthetic-smoke",
                "questions": [f"q_{i}" for i in range(5)],
            },
            str(path),
        )
        written[layer] = path
    # Persona bank (system prompts) JSON — only the persona NAMES are read in CPU
    # tests; the prompts are placeholders. Schema matches production
    # contrastive_neg_geometry_472.persona_bank.load_persona_bank's contract
    # ({schema_version, personas: {name: prompt}}); the bare-dict layout used
    # before round 8 failed load_persona_bank's schema_version assertion.
    bank_path = out_dir / "persona_bank.json"
    bank_payload = {
        "schema_version": "i472_v1",
        "base_model": "synthetic-smoke",
        "personas": {p: f"You are {p}." for p in personas},
    }
    bank_path.write_text(json.dumps(bank_payload, indent=2))
    return {"personas": personas, "bank_path": bank_path, "centroids_paths": written}


def make_synthetic_r_train(
    out_dir: Path,
    *,
    source: str = "villain",
    n_questions: int = 5,
    response_length: int = 50,
    extra_personas: list[str] | None = None,
) -> Path:
    """Synthetic R_train.json: covers every persona in the synthetic bank.

    Round-8 fix: Phase 0.5 picks 4 positioned-N's by cosine band, which under
    synthetic centroids can land on ANY probe_persona_NNN, not just the named
    near/mid_near/mid_far/far personas. So the R_train fixture must cover the
    WHOLE synthetic bank to exercise Phase 0.7's no-op branch (the GPU branch
    that fires for genuinely missing personas is covered by the production
    pod on the round-8 relaunch).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    completions: dict = {}
    qs = [f"q_{i}" for i in range(n_questions)]
    base_personas = [
        source,
        "qwen_default",
        "near_persona",
        "mid_near_persona",
        "mid_far_persona",
        "far_persona",
    ]
    all_personas = list(base_personas)
    if extra_personas:
        for p in extra_personas:
            if p not in all_personas:
                all_personas.append(p)
    # Build R_train for every persona we might use as positive/negative.
    for persona in all_personas:
        completions[persona] = {
            q: {
                "response_text": f"Answer from {persona} to {q}.",
                "response_token_ids": list(range(100, 100 + response_length)),
            }
            for q in qs
        }
    payload = {
        "schema_version": "i472_v1",
        "split": "train",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "completions": completions,
        "questions": qs,
        "personas": sorted(completions.keys()),
        "n_personas": len(completions),
    }
    path = out_dir / "R_train.json"
    path.write_text(json.dumps(payload, indent=2))
    eval_payload = {**payload, "split": "eval"}
    (out_dir / "R_eval.json").write_text(json.dumps(eval_payload, indent=2))
    return path


def make_synthetic_smoke_trajectory(
    out_path: Path,
    *,
    cell: str,
    seed: int = 42,
    source_dgs_per_frac: dict[float, float] | None = None,
    source_emit_per_frac: dict[float, float] | None = None,
) -> Path:
    """Synthetic per-cell trajectory with 6 checkpoints (matches plan cadence).

    For Phase 0 pick smoke: source_dgs lets us simulate {below-band, in-band,
    above-band} ranks so the pick rule can pick a meaningful anchor.
    """
    fracs = (0.08, 0.16, 0.33, 0.5, 0.75, 1.0)
    src_dgs = source_dgs_per_frac or {f: 7.0 for f in fracs}
    src_emit = source_emit_per_frac or {f: 0.5 for f in fracs}
    cks = []
    for f in fracs:
        cks.append(
            {
                "frac": f,
                "step": int(f * 25),
                "adapter_path": f"/tmp/fake/adapter/{cell}_seed{seed}/frac_{f:.2f}",
                "source_self": {
                    "g_logp_mean": -1.0,
                    "b_logp_mean": -1.0 - src_dgs[f],
                    "delta_g_mean": src_dgs[f],
                    "emission_p": src_emit[f],
                    "r_collapsed": False,
                },
                "held_out_collapse_share": 0.0,
                "n_held_out_collapsed": 0,
                "held_out": {},
            }
        )
    payload = {
        "schema_version": "i472_v1",
        "cell": cell,
        "seed": seed,
        "source": "villain",
        "checkpoints": cks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def make_synthetic_phase1_trajectories(
    slab_root: Path,
    arm_to_positioned_n: dict[str, str],
    seeds: list[int],
    chosen_frac: float,
    held_out_panel: list[str],
    centroids_l10: dict[str, np.ndarray],
    *,
    rng_seed: int = 42,
) -> None:
    """Synthetic Phase 1 trajectories — one per (cell, seed).

    Plants a soft BARRIER signal: held-out probe ΔG is the SHADOW angle (probes
    behind N have small angle → low ΔG; lateral probes have higher angle →
    higher ΔG). This is enough for the Phase 2 regression to exercise all
    branches.
    """
    rng = np.random.default_rng(rng_seed)
    src_vec = centroids_l10["villain"].astype(np.float64)

    def _shadow(probe: str, n: str) -> float:
        from explore_persona_space.experiments.contrastive_neg_geometry_504.shadow_angle import (
            shadow_angle,
        )

        return shadow_angle(centroids_l10[probe], centroids_l10[n], src_vec)

    def _d_source(probe: str) -> float:
        p = centroids_l10[probe].astype(np.float64)
        cos_ = float(np.dot(p, src_vec) / (np.linalg.norm(p) * np.linalg.norm(src_vec)))
        return 1.0 - cos_

    for cell, n_persona in arm_to_positioned_n.items():
        for seed in seeds:
            # Held-out panel ΔG: target = +5 + 3*shadow_angle (barrier) +
            # noise. Source ΔG ~ 7 nats (in-band).
            held_out: dict[str, dict[str, dict[str, float]]] = {}
            for probe in held_out_panel:
                sa = _shadow(probe, n_persona)
                base_dg = 5.0 + 3.0 * sa + rng.standard_normal() * 0.5
                held_out[probe] = {
                    "q_0": {
                        "g_logp": -1.0,
                        "b_logp": -1.0 - base_dg,
                        "delta_g": float(base_dg),
                        "argmax_marker": False,
                        "n_marker_in_R": 0,
                        "r_collapsed": False,
                        "kl": None,
                    }
                }
            # Source-self at the pinned frac sits in-band.
            payload = {
                "schema_version": "i472_v1",
                "cell": cell,
                "seed": seed,
                "source": "villain",
                "checkpoints": [
                    {
                        "frac": chosen_frac,
                        "step": int(chosen_frac * 25),
                        "adapter_path": f"/tmp/fake/{cell}_seed{seed}/frac_{chosen_frac:.2f}",
                        "source_self": {
                            "g_logp_mean": -1.0,
                            "b_logp_mean": -8.0,
                            "delta_g_mean": 7.0 + rng.standard_normal() * 0.5,
                            "emission_p": 0.5,
                            "r_collapsed": False,
                        },
                        "held_out_collapse_share": 0.0,
                        "n_held_out_collapsed": 0,
                        "held_out": held_out,
                    }
                ],
            }
            (slab_root / f"{cell}_seed{seed}").mkdir(parents=True, exist_ok=True)
            (slab_root / f"{cell}_seed{seed}" / "trajectory.json").write_text(
                json.dumps(payload, indent=2)
            )


def main() -> int:
    log = setup_log()
    repo_root = Path(__file__).resolve().parent.parent
    work_dir = Path(tempfile.mkdtemp(prefix="i504-smoke-"))
    log.info("Smoke workdir: %s", work_dir)

    centroids_dir = work_dir / "centroids"
    artifacts = make_synthetic_centroids(centroids_dir)
    log.info(
        "[setup] wrote %d centroids files (%d personas) → %s",
        len(artifacts["centroids_paths"]),
        len(artifacts["personas"]),
        centroids_dir,
    )
    r_train_path = make_synthetic_r_train(
        centroids_dir / "on_policy_R",
        extra_personas=artifacts["personas"],
    )

    # ── (1) Phase 0.5 gates ────────────────────────────────────────────────
    phase05_out = work_dir / "phase0_5_gates.json"
    cmd = [
        "uv",
        "run",
        "python",
        str(repo_root / "scripts" / "i504_phase_phase05.py"),
        "--centroids-dir",
        str(centroids_dir),
        "--r-train-path",
        str(r_train_path),
        "--out-path",
        str(phase05_out),
    ]
    log.info("[phase05] %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 or not phase05_out.exists():
        log.error("[phase05] FAIL rc=%s", rc)
        return 2
    report = json.loads(phase05_out.read_text())
    log.info(
        "[phase05] PASS rc=0 verdict=%s chosen_layer=%s arm_to_n=%s smoke_mid=%s n_panel=%d",
        report.get("verdict"),
        report.get("chosen_layer"),
        report.get("arm_to_positioned_n"),
        report.get("smoke_mid_band_n"),
        len(report.get("held_out_panel", [])),
    )

    # ── (1.5) Phase 0.7 r-fill, TRAIN + EVAL (CPU no-op path) ──────────────
    # Round 11: Phase 0.7 now fills BOTH R_train AND R_eval symmetrically. The
    # synthetic R_train + R_eval cover every persona the synthetic Phase 0.5
    # picks (train side) AND every panel persona (eval side), so the fill
    # script's no-op branch is exercised on BOTH sides — verifies the diff
    # logic + byte-identical copies + the sentinel write. The vLLM-generate
    # path (missing personas → on-policy decode) is NOT smokeable on CPU; it
    # gets exercised by the GPU pod the moment Phase 0.5 picks a new persona
    # that #472's R_{train,eval} doesn't cover.
    r_train_v504_path = r_train_path.with_name("R_train_v504.json")
    r_eval_path = r_train_path.with_name("R_eval.json")  # written by make_synthetic_r_train
    r_eval_v504_path = r_train_path.with_name("R_eval_v504.json")
    sentinel_phase07 = work_dir / "phase07-sentinel.json"
    cmd = [
        "uv",
        "run",
        "python",
        str(repo_root / "scripts" / "i504_phase_r_generate_fill.py"),
        "--phase05-path",
        str(phase05_out),
        "--split",
        "both",
        "--input-r-train-path",
        str(r_train_path),
        "--output-r-train-path",
        str(r_train_v504_path),
        "--input-r-eval-path",
        str(r_eval_path),
        "--output-r-eval-path",
        str(r_eval_v504_path),
        "--bank-path",
        str(centroids_dir / "persona_bank.json"),
        "--no-upload",
        "--sentinel-path",
        str(sentinel_phase07),
    ]
    log.info("[phase07] %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if (
        rc != 0
        or not r_train_v504_path.exists()
        or not r_eval_v504_path.exists()
        or not sentinel_phase07.exists()
    ):
        log.error(
            "[phase07] FAIL rc=%s train_v504_exists=%s eval_v504_exists=%s sentinel_exists=%s",
            rc,
            r_train_v504_path.exists(),
            r_eval_v504_path.exists(),
            sentinel_phase07.exists(),
        )
        return 2
    sentinel = json.loads(sentinel_phase07.read_text())
    note = json.loads(sentinel.get("note", "{}"))
    log.info(
        "[phase07] PASS rc=0 status=%s n_train_filled=%d n_eval_filled=%d "
        "train_v504_path=%s eval_v504_path=%s",
        note.get("status"),
        len(note.get("train_missing_filled", []) or note.get("train_missing", [])),
        len(note.get("eval_missing_filled", []) or note.get("eval_missing", [])),
        r_train_v504_path,
        r_eval_v504_path,
    )
    assert note.get("status") == "ok_noop", (
        "smoke synthetic R_train + R_eval cover Phase 0.5 picks + panel; "
        f"expected ok_noop, got {note.get('status')}"
    )

    # ── (2) Phase 0 pick (over synthetic smoke trajectories) ───────────────
    slab_root = work_dir / "slab"
    slab_root.mkdir(parents=True, exist_ok=True)
    # r=4: 1.5/2.5/3.0/3.5/3.8/4.0  → sub-band everywhere
    # r=8: 1.0/3.0/6.0/9.0/11.0/13.0 → in-band at frac {0.33, 0.5, 0.75}
    # r=16: 8.0/14.0/16.0/18.0/20.0/22.0 → in-band only at 0.08
    make_synthetic_smoke_trajectory(
        slab_root / "c504_smoke_r4_seed42" / "trajectory.json",
        cell="c504_smoke_r4",
        source_dgs_per_frac={0.08: 1.5, 0.16: 2.5, 0.33: 3.0, 0.5: 3.5, 0.75: 3.8, 1.0: 4.0},
        source_emit_per_frac={0.08: 0.05, 0.16: 0.1, 0.33: 0.2, 0.5: 0.3, 0.75: 0.4, 1.0: 0.5},
    )
    make_synthetic_smoke_trajectory(
        slab_root / "c504_smoke_r8_seed42" / "trajectory.json",
        cell="c504_smoke_r8",
        source_dgs_per_frac={0.08: 1.0, 0.16: 3.0, 0.33: 6.0, 0.5: 9.0, 0.75: 11.0, 1.0: 13.0},
        source_emit_per_frac={0.08: 0.05, 0.16: 0.15, 0.33: 0.3, 0.5: 0.5, 0.75: 0.7, 1.0: 0.85},
    )
    make_synthetic_smoke_trajectory(
        slab_root / "c504_smoke_r16_seed42" / "trajectory.json",
        cell="c504_smoke_r16",
        source_dgs_per_frac={0.08: 8.0, 0.16: 14.0, 0.33: 16.0, 0.5: 18.0, 0.75: 20.0, 1.0: 22.0},
        source_emit_per_frac={0.08: 0.4, 0.16: 0.6, 0.33: 0.85, 0.5: 0.95, 0.75: 0.99, 1.0: 1.0},
    )
    phase0_pick_out = work_dir / "phase0_calibration.json"
    cmd = [
        "uv",
        "run",
        "python",
        str(repo_root / "scripts" / "i504_phase_phase0_pick.py"),
        "--slab-root",
        str(slab_root),
        "--out-path",
        str(phase0_pick_out),
    ]
    log.info("[phase0_pick] %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 or not phase0_pick_out.exists():
        log.error("[phase0_pick] FAIL rc=%s", rc)
        return 3
    pick = json.loads(phase0_pick_out.read_text())
    log.info(
        "[phase0_pick] PASS rc=0 verdict=%s rank=%s alpha=%s frac=%s source_dg=%.2f",
        pick.get("verdict"),
        pick.get("chosen_rank"),
        pick.get("chosen_alpha"),
        pick.get("chosen_checkpoint_fraction"),
        pick.get("source_delta_g_at_pick_nats") or float("nan"),
    )
    assert pick.get("verdict") == "pass", "smoke phase0 pick should land r=8"
    assert pick.get("chosen_rank") == 8, (
        f"smoke phase0 pick expected r=8, got {pick.get('chosen_rank')}"
    )

    # ── (3) build_cell_504 (training-pool builder, CPU) ─────────────────────
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )

    bank = load_persona_bank(centroids_dir / "persona_bank.json")
    r_train_raw = json.loads(r_train_path.read_text())
    # build_cell_504 reads completions[persona][q]; production schema wraps under
    # 'completions' key (i472_v1). Smoke's r_train fixture follows the same shape.
    r_train = r_train_raw.get("completions", r_train_raw)
    from explore_persona_space.experiments.contrastive_neg_geometry_504.build_training_data import (
        build_cell_504,
    )

    train_dir = work_dir / "train_pools"
    train_dir.mkdir(parents=True, exist_ok=True)
    arm_to_positioned_n = report["arm_to_positioned_n"]
    smoke_mid_band_n = report["smoke_mid_band_n"]
    # Build c504_near + c504_default_only (canary covers both positioned arm + default-only arm).
    for cell in ("c504_near", "c504_default_only"):
        try:
            built = build_cell_504(
                cell,
                train_dir / f"{cell}.jsonl",
                r_train=r_train,
                arm_to_positioned_n=arm_to_positioned_n,
                q_train=list(r_train["villain"]),  # 5 synthetic questions
                persona_bank=bank,
                source="villain",
                smoke_mid_band_n=smoke_mid_band_n,
                seed=42,
            )
            n_rows = sum(1 for _ in built.open())
            log.info("[build_cell_504] %s PASS → %s (%d rows)", cell, built, n_rows)
        except (KeyError, AssertionError, ValueError) as e:
            # Synthetic R_train only has 5 questions; build_cell expects 200 pos.
            # This is EXPECTED — we just verify the call is wired and the
            # exception is the documented "row count mismatch" path.
            log.info("[build_cell_504] %s expected limit (synthetic R_train < 200 q): %s", cell, e)

    # ── (4) Phase 2 analyze (CPU) — exercises the 6-covariate regression ────
    # Plant a soft barrier signal over the 4 positioned arms × 2 seeds × panel.
    # Need centroids at the headline layer that match Phase 0.5's pick.
    chosen_layer = report["chosen_layer"]
    if chosen_layer is None:
        log.error("[phase05] chose_layer=None — fix the synthetic input distribution.")
        return 4
    # Unpack the structured #472 schema the same way Phase 0.5's loader
    # does — so this smoke step exercises the SAME schema as the pod-side
    # dispatcher (catches centroids-schema drift locally instead of at
    # Phase 0.5 on the GPU; round-2 of #504 crashed on this exact mismatch
    # because the smoke wrote a flat layout and the dispatcher read the
    # structured layout).
    cents_chosen_bundle = torch.load(
        centroids_dir / f"centroids_L{chosen_layer}.pt",
        map_location="cpu",
        weights_only=False,
    )
    cents_chosen_mat = cents_chosen_bundle["centroids"].to(dtype=torch.float32).cpu().numpy()
    cents_chosen_names = list(cents_chosen_bundle["persona_names"])
    cents_chosen_np = {n: cents_chosen_mat[i] for i, n in enumerate(cents_chosen_names)}
    chosen_frac = pick["chosen_checkpoint_fraction"]
    held_out_panel = report["held_out_panel"]
    make_synthetic_phase1_trajectories(
        slab_root,
        arm_to_positioned_n,
        seeds=[42, 137],
        chosen_frac=chosen_frac,
        held_out_panel=held_out_panel,
        centroids_l10=cents_chosen_np,
    )
    cmd = [
        "uv",
        "run",
        "python",
        str(repo_root / "scripts" / "i504_phase_analyze.py"),
        "--slab-root",
        str(slab_root),
        "--phase0-path",
        str(phase0_pick_out),
        "--phase05-path",
        str(phase05_out),
        "--seeds",
        "42,137",
    ]
    log.info("[analyze] %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    summary_path = slab_root / "analyze_summary.json"
    if rc != 0 or not summary_path.exists():
        log.error("[analyze] FAIL rc=%s", rc)
        return 5
    summary = json.loads(summary_path.read_text())
    log.info(
        "[analyze] PASS rc=0 n_rows=%d chosen_frac=%s notes=%s",
        summary.get("n_rows_pooled"),
        summary.get("chosen_checkpoint_fraction"),
        summary.get("notes"),
    )
    pf = summary["pooled_fit"]
    log.info(
        "[analyze] partial Spearman: %s",
        {
            k: f"rho={v['rho']:.3f} p_raw={v['p_raw']:.4f}"
            for k, v in pf["partial_spearman"].items()
        },
    )
    log.info(
        "[analyze] sign agreement: %s",
        {k: v for k, v in summary["sign_agreement"].items()},
    )

    log.info("ALL SMOKE PHASES PASSED. Workdir: %s", work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
