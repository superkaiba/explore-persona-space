#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 Phase 0 — zero-training four-float reads of the 20 #472 adapters.

Runs FIRST, on-pod, before any training (plan §4 Phase 0):

  0a-tf   Teacher-forced endpoint re-read of ALL 20 parent final adapters
          (``compute_kl_for_checkpoint`` over frozen R_eval) on: source
          (villain) + the cell's realized trained negatives + the 8-bystander
          reference panel — sharded across GPUs as worker subprocesses.
  0a-op   On-policy re-read of the 8 count-cell adapters (4 ratio levels x 2
          seeds) via ``i601_eval_trajectory.py`` with a single terminal
          checkpoint spec + the bystander8 panel.
  gate    Adapter-application cross-check (#534 / plan §7 gate 2): each
          on-policy source ΔG must reproduce the COMMITTED
          ``eval_results/issue_472/<cell>_seed<S>/trajectory.json`` terminal
          ``source_self.delta_g_mean`` within 1 nat. FAIL → phase0_gate.json
          pass=false and ALL training phases stay gated.
  0b      Trained-negative clamp read (from the 0a-tf records).
  calib   Margin references M(level) from the ON-POLICY subset (plan §6
          pinned rule) + the primary-space decision (plan §4 item 2).
  fitness Anchor-reuse checks: on-policy 1-nat re-read + the held-out-panel
          determinism check (builder recipe match vs the committed panel).

Outputs:
  eval_results/issue_601/phase0/bystander_panel.json   (pre-registered names)
  eval_results/issue_601/phase0/teacher_forced/<cell>_seed<S>.json
  eval_results/issue_601/phase0/onpolicy_recheck/<cell>_seed<S>/trajectory.json
  eval_results/issue_601/phase0/endpoint_reads.json    (aggregate)
  eval_results/issue_601/phase0/phase0_gate.json       (the §7 HALT gate)

Worker modes (spawned by the orchestrator with a launcher-env CVD pin):
  --worker-teacher --adapters c472_anchor_seed42,...   one GPU shard of 0a-tf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i601.phase0")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _personas_for_parent_cell(cell: str, cts: dict[str, float], bystanders: list[str]) -> list[str]:
    """source + the parent cell's realized trained negatives + the bystander panel."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        negatives_for_cell,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import SOURCE_PERSONA

    negs = negatives_for_cell(cell, cts)  # parent CELL_SPECS (default registry).
    ordered = [SOURCE_PERSONA, *negs]
    ordered.extend(b for b in bystanders if b not in ordered)
    return ordered


def _run_teacher_worker(args) -> int:
    """One GPU shard of the 0a teacher-forced reads (invoked as a subprocess)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import HEADLINE_LAYER
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_logit_readout_gauge_free,
        compute_kl_for_checkpoint,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import SOURCE_PERSONA

    bank = load_persona_bank(args.data_dir / "persona_bank.json")
    r_eval = load_r_artifact(args.data_dir / "on_policy_R" / "R_eval.json")
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.data_dir)
    _q_train, q_eval = get_train_eval_questions()
    bystanders = json.loads((args.phase0_dir / "bystander_panel.json").read_text())["personas"]

    out_dir = args.phase0_dir / "teacher_forced"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in [k.strip() for k in args.adapters.split(",") if k.strip()]:
        out_path = out_dir / f"{key}.json"
        if out_path.exists():
            log.info("[tf %s] exists; skip (idempotent re-run)", key)
            continue
        cell, seed_s = key.rsplit("_seed", 1)
        adapter_dir = args.adapters_root / key
        assert_logit_readout_gauge_free(str(adapter_dir))
        personas = _personas_for_parent_cell(cell, cts, bystanders)
        eval_personas = {p: bank[p] for p in personas}
        r_map = {p: {q: r_eval[p][q]["response_text"] for q in q_eval} for p in personas}
        log.info("[tf %s] %d personas x %d questions", key, len(personas), len(q_eval))
        stats = compute_kl_for_checkpoint(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            adapter_path=str(adapter_dir),
            r_by_persona_q=r_map,
            eval_personas=eval_personas,
            eval_questions=q_eval,
        )
        payload = {
            "schema_version": "i601_phase0_tf_v1",
            "cell": cell,
            "seed": int(seed_s),
            "personas": personas,
            "trained_negatives": [
                p for p in personas if p != SOURCE_PERSONA and p not in bystanders
            ],
            "bystander_panel": bystanders,
            "eval_questions": q_eval,
            "read_type": "teacher_forced_frozen_R_eval",
            "stats": stats,
            "git_commit": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, out_path)
        log.info("[tf %s] written → %s", key, out_path)
    return 0


def _pool(cmds: list[tuple[str, list[str], Path]], n_gpus: int, log_dir: Path) -> None:
    """Tiny GPU pool: run labeled commands, one per free GPU, launcher-env CVD pin."""
    queue = list(cmds)
    running: list[tuple[subprocess.Popen, str, int]] = []
    free = list(range(n_gpus))
    while queue or running:
        while queue and free:
            label, cmd, logfile = queue.pop(0)
            gpu = free.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            logfile.parent.mkdir(parents=True, exist_ok=True)
            fh = open(logfile, "w")  # noqa: SIM115 -- lives for the Popen's lifetime
            log.info("[pool] %s on GPU %d → %s", label, gpu, logfile)
            running.append(
                (subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), label, gpu)
            )
        still = []
        for proc, label, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, label, gpu))
                continue
            free.append(gpu)
            if rc != 0:
                for p2, _l2, _g2 in still:
                    p2.terminate()
                raise RuntimeError(f"phase0 worker {label} exited rc={rc}; see {log_dir}")
            log.info("[pool] %s complete (GPU %d)", label, gpu)
        running = still
        if running:
            time.sleep(5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #601 Phase 0 zero-training reads (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_601"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_601"))
    ap.add_argument(
        "--adapters-root", type=Path, default=Path("/workspace/models/issue_601_parent")
    )
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument(
        "--committed-472-root",
        type=Path,
        default=Path("eval_results/issue_472"),
        help="The committed parent trajectories (the cross-check reference).",
    )
    ap.add_argument("--skip-onpolicy", action="store_true", help="Descope lever (plan §9).")
    ap.add_argument("--skip-upload", action="store_true", help="Debug only.")
    # Worker mode.
    ap.add_argument("--worker-teacher", action="store_true")
    ap.add_argument("--adapters", default="")
    args = ap.parse_args(argv)
    args.phase0_dir = args.slab_root / "phase0"

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase0] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if args.worker_teacher:
        return _run_teacher_worker(args)

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import HEADLINE_LAYER
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        held_out_panel,
        negatives_for_cell,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import (
        COUNT_CELL_LEVELS,
        HF_DATA_PREFIX_601,
        N_BYSTANDER_REFERENCE,
        PARENT_CELLS_ALL,
        PARENT_SEEDS,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.neg_setpoint_601.artifacts import (
        fetch_parent_adapter,
        fetch_parent_data,
    )
    from explore_persona_space.experiments.neg_setpoint_601.phase0_lib import (
        clamp_read,
        decide_primary_space,
        margin_references,
        onpolicy_crosscheck,
        select_bystander_reference_panel,
        terminal_source_stats,
    )

    args.phase0_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Inputs. ───────────────────────────────────────────────────────────────
    log.info("[phase=p0_fetch] parent data + 20 adapters")
    fetch_parent_data(args.data_dir.resolve().parent.parent)
    adapter_keys = [f"{c}_seed{s}" for c in PARENT_CELLS_ALL for s in PARENT_SEEDS]
    for c in PARENT_CELLS_ALL:
        for s in PARENT_SEEDS:
            fetch_parent_adapter(c, s, args.adapters_root)

    # ── Bystander reference panel (pre-registered by name). ─────────────────
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.data_dir)
    panel_all = held_out_panel(cts, source=SOURCE_PERSONA)
    bystanders = select_bystander_reference_panel(panel_all, cts, n=N_BYSTANDER_REFERENCE)
    (args.phase0_dir / "bystander_panel.json").write_text(
        json.dumps(
            {
                "schema_version": "i601_bystander_panel_v1",
                "personas": bystanders,
                "d_source": {p: 1.0 - cts[p] for p in bystanders},
                "selection": "L10 d_source deciles over the #472 held-out panel",
                "n_held_out": len(panel_all),
                "git_commit": _git_sha(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    log.info("[phase=p0_panel] pre-registered bystanders: %s", bystanders)

    # ── Anchor recipe-match determinism check (plan §10 fitness (a)). ────────
    committed_anchor = json.loads(
        (args.committed_472_root / "c472_anchor_seed42" / "trajectory.json").read_text()
    )
    committed_panel = sorted(committed_anchor["held_out_personas"])
    recipe_panel_ok = sorted(panel_all) == committed_panel
    if not recipe_panel_ok:
        log.error(
            "[phase=p0_recipe] held-out panel determinism FAILED: rebuilt %d vs committed %d "
            "personas — builder/selector drift since 7b540544.",
            len(panel_all),
            len(committed_panel),
        )

    # ── 0a-tf: teacher-forced reads, sharded across GPUs. ────────────────────
    shards: list[list[str]] = [[] for _ in range(max(1, args.n_gpus))]
    for i, key in enumerate(adapter_keys):
        shards[i % len(shards)].append(key)
    tf_cmds = []
    for i, shard in enumerate(s for s in shards if s):
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/i601_phase0_reads.py",
            "--worker-teacher",
            "--adapters",
            ",".join(shard),
            "--slab-root",
            str(args.slab_root),
            "--data-dir",
            str(args.data_dir),
            "--adapters-root",
            str(args.adapters_root),
        ]
        tf_cmds.append((f"tf_shard{i}", cmd, args.log_dir / f"issue-601-phase0-tf{i}.log"))
    log.info("[phase=p0_teacher] %d shards over %d adapters", len(tf_cmds), len(adapter_keys))
    _pool(tf_cmds, args.n_gpus, args.log_dir)

    # ── 0a-op: on-policy recheck of the 8 count-cell adapters. ───────────────
    onpolicy_keys = [f"{c}_seed{s}" for c, _lvl in COUNT_CELL_LEVELS for s in PARENT_SEEDS]
    if not args.skip_onpolicy:
        op_cmds = []
        for key in onpolicy_keys:
            out_dir = args.phase0_dir / "onpolicy_recheck" / key
            if (out_dir / "trajectory.json").exists():
                log.info("[phase=p0_onpolicy] %s exists; skip", key)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            # Synthetic single-checkpoint index pointing at the final adapter.
            idx_path = out_dir / "checkpoint_index.json"
            idx_path.write_text(
                json.dumps({"1.0000": {"step": None, "path": str(args.adapters_root / key)}})
            )
            cell = key.rsplit("_seed", 1)[0]
            seed = int(key.rsplit("_seed", 1)[1])
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/i601_eval_trajectory.py",
                "--cell",
                cell,
                "--seed",
                str(seed),
                "--checkpoint-index",
                str(idx_path),
                "--out-path",
                str(out_dir / "trajectory.json"),
                "--raw-completions-path",
                str(out_dir / "raw_completions.json"),
                "--data-dir",
                str(args.data_dir),
                "--fracs",
                "1.0000",
                "--panel",
                "bystander8",
                "--bystander-panel-path",
                str(args.phase0_dir / "bystander_panel.json"),
            ]
            op_cmds.append((f"op_{key}", cmd, args.log_dir / f"issue-601-phase0-op-{key}.log"))
        log.info("[phase=p0_onpolicy] %d on-policy rechecks", len(op_cmds))
        _pool(op_cmds, args.n_gpus, args.log_dir)

    # ── Aggregate + gate. ─────────────────────────────────────────────────────
    teacher: dict[str, dict] = {}
    for key in adapter_keys:
        p = args.phase0_dir / "teacher_forced" / f"{key}.json"
        teacher[key] = json.loads(p.read_text())

    reread: dict[str, dict] = {}
    committed: dict[str, float] = {}
    if not args.skip_onpolicy:
        for key in onpolicy_keys:
            traj = json.loads(
                (args.phase0_dir / "onpolicy_recheck" / key / "trajectory.json").read_text()
            )
            reread[key] = terminal_source_stats(traj)
            committed_traj = json.loads(
                (args.committed_472_root / key / "trajectory.json").read_text()
            )
            term = max(committed_traj["checkpoints"], key=lambda c: c["frac"])
            committed[key] = float(term["source_self"]["delta_g_mean"])
        crosscheck = onpolicy_crosscheck(reread, committed)
        level_by_cell = dict(COUNT_CELL_LEVELS)
        margin_refs = margin_references(reread, level_by_cell)
        space = decide_primary_space(margin_refs)
    else:
        crosscheck = {"pass": False, "skipped": True}
        margin_refs, space = {}, {"primary_space": "unavailable"}

    count_cells = [c for c, _lvl in COUNT_CELL_LEVELS]
    negatives_by_cell = {c: negatives_for_cell(c, cts) for c in count_cells}
    tf_stats = {k: v["stats"] for k, v in teacher.items()}
    # Adapt the tf records to the clamp reader's (logp_hf_g/logp_hf_b) schema.
    clamp = clamp_read(tf_stats, bystanders, negatives_by_cell, count_cells)

    anchor_keys = [f"c472_anchor_seed{s}" for s in PARENT_SEEDS]
    anchor_onpolicy_ok = all(
        crosscheck.get("per_adapter", {}).get(k, {}).get("within_tol", False) for k in anchor_keys
    )
    anchor_reuse_ok = bool(recipe_panel_ok and anchor_onpolicy_ok)

    endpoint = {
        "schema_version": "i601_phase0_v1",
        "bystander_panel": bystanders,
        "teacher_forced_index": {k: f"phase0/teacher_forced/{k}.json" for k in adapter_keys},
        "onpolicy_crosscheck": crosscheck,
        "margin_references": margin_refs,
        "space_calibration": space,
        "clamp_read": clamp,
        "anchor_reuse": {
            "ok": anchor_reuse_ok,
            "recipe_panel_ok": recipe_panel_ok,
            "onpolicy_within_1nat": anchor_onpolicy_ok,
            "fallback": "dispatch --anchor-retrain-fallback (dense_200p800n seed 42)",
        },
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (args.phase0_dir / "endpoint_reads.json").write_text(json.dumps(endpoint, indent=2))

    gate = {
        "pass": bool(crosscheck.get("pass")),
        "gate": "adapter-application cross-check (plan §7 gate 2, #534 class)",
        "onpolicy_crosscheck": crosscheck,
        "anchor_reuse_ok": anchor_reuse_ok,
        "primary_space": space.get("primary_space"),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (args.phase0_dir / "phase0_gate.json").write_text(json.dumps(gate, indent=2))
    log.info(
        "[phase=p0_gate] pass=%s anchor_reuse_ok=%s primary_space=%s",
        gate["pass"],
        anchor_reuse_ok,
        gate["primary_space"],
    )

    # Raw completions from the on-policy rechecks (Upload Policy).
    if not args.skip_upload and not args.skip_onpolicy:
        from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

        upload_raw_completions_to_data_repo(
            experiment_name=HF_DATA_PREFIX_601, eval_results_dir=args.slab_root
        )

    sentinel = args.log_dir / "issue-601-phase0-results.json"
    sentinel.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:progress",
                "version": 1,
                "task_id": 601,
                "by": "i601_phase0_reads",
                "ts": datetime.now(UTC).isoformat(),
                "phase": "phase0_done",
                "note": json.dumps(
                    {
                        "gate_pass": gate["pass"],
                        "anchor_reuse_ok": anchor_reuse_ok,
                        "primary_space": gate["primary_space"],
                        "clamp_present": clamp.get("clamp_present"),
                        "endpoint_reads": str(args.phase0_dir / "endpoint_reads.json"),
                    }
                ),
            },
            indent=2,
        )
    )
    log.info("[phase=p0_done] sentinel → %s", sentinel)
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
