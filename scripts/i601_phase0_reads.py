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
  gate    Plan v3 §B gate split (gate_schema 2; supersedes the v2 conjunctive
          all-8-within-1-nat rule, unsatisfiable on intact tooling — §A):
            pass            = Gate S ONLY (structural eval-path integrity,
                              HALT-class): IdenticalRereadAlarm silent +
                              reread_r_collapsed false on all 8 + all 4
                              low-dose adapter-seeds (noneg x2, negex_100 x2)
                              within 1.5 nat of the COMMITTED
                              ``eval_results/issue_472/<cell>_seed<S>/
                              trajectory.json`` terminals + dose ordering
                              noneg < negex_100 < min(anchor, negex_400)
                              with >= 2-nat seed-mean gaps. FAIL → ALL
                              training phases stay gated.
            anchor_reuse_ok = Gate A (anchor adapters within 1 nat +
                              recipe-panel determinism; routing-class —
                              false fires the budgeted dense_200p800n
                              seed-42 retrain fallback, never a halt).
            observation_o   = negex_400 re-reads (recorded, never gating —
                              the registered Phase-0a regime deliverable).
          Re-runs are skip-cheap: parity-regime tf/op outputs are reused and
          the gate recomputes from the existing JSONs (no GPU recompute).
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
    from explore_persona_space.experiments.neg_setpoint_601.artifacts import (
        stage_parity_read_adapter,
    )
    from explore_persona_space.experiments.neg_setpoint_601.phase0_lib import (
        COVERAGE_ABSENT,
        COVERAGE_FULL,
        build_r_map,
        split_by_r_coverage,
    )

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
            # Round-5: only skip outputs produced under the parity-read
            # regime — pre-parity (round-4) outputs are the dirty
            # ceiling-pinned reads and MUST be re-read, not reused.
            prior = json.loads(out_path.read_text())
            prior_prov = prior.get("adapter_provenance") or {}
            if prior_prov.get("use_rslora_applied") is False:
                log.info("[tf %s] parity-regime output exists; skip (idempotent re-run)", key)
                continue
            log.warning("[tf %s] STALE pre-parity output — re-reading under parity regime", key)
            out_path.unlink()
        cell, seed_s = key.rsplit("_seed", 1)
        adapter_dir = args.adapters_root / key
        # Round-5 parity staging: apply the adapter at the parent-realized
        # read scaling (use_rslora forced False) — at the shipped rsLoRA
        # config every parent adapter is a ` ※`-repeater and the tf read pins
        # at -mean(b_logp) ~ 22.85 for ALL adapters (the round-4 dirty
        # endpoint_reads). Includes the fail-loud slug-in-path mapping assert.
        staged_dir, adapter_provenance = stage_parity_read_adapter(
            adapter_dir, args.phase0_dir / "staged_adapters", expect_slug=key
        )
        assert_logit_readout_gauge_free(str(staged_dir))
        # Coverage split (concern phase0-r-eval-coverage-gap): the pinned R_eval
        # misses 3 parent trained negatives (c472_near: mob_boss + cult_leader;
        # c472_negp_8: baker). Those reads are DESCOPED per-persona with an
        # explicit coverage record — frozen R is never regenerated. Source /
        # bystander gaps are a hard fail (the panel is coverage-constrained at
        # registration, so this only fires on artifact drift).
        personas_all = _personas_for_parent_cell(cell, cts, bystanders)
        personas, descoped = split_by_r_coverage(r_eval, personas_all, q_eval)
        hard_missing = [p for p in descoped if p == SOURCE_PERSONA or p in bystanders]
        if hard_missing:
            raise KeyError(
                f"[tf {key}] source/bystander personas lack full frozen-R_eval coverage: "
                f"{hard_missing} (#504 class) — the registered panel must be "
                f"coverage-constrained; refusing the shard."
            )
        if descoped:
            log.warning(
                "[tf %s] trained-negative reads descoped (%s): %s",
                key,
                COVERAGE_ABSENT,
                descoped,
            )
        eval_personas = {p: bank[p] for p in personas}
        r_map = build_r_map(r_eval, personas, q_eval)
        log.info("[tf %s] %d personas x %d questions", key, len(personas), len(q_eval))
        stats = compute_kl_for_checkpoint(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            adapter_path=str(staged_dir),
            r_by_persona_q=r_map,
            eval_personas=eval_personas,
            eval_questions=q_eval,
        )
        payload = {
            "schema_version": "i601_phase0_tf_v2",
            "cell": cell,
            "seed": int(seed_s),
            "personas": personas,
            "trained_negatives": [
                p for p in personas if p != SOURCE_PERSONA and p not in bystanders
            ],
            # Explicit per-persona coverage record (concern
            # phase0-r-eval-coverage-gap): descoped trained negatives carry
            # "absent-from-frozen-R" so the analyzer/clean-result can name the
            # missing reads instead of silently shrinking the denominator.
            "coverage": {
                **{p: COVERAGE_FULL for p in personas},
                **{p: COVERAGE_ABSENT for p in descoped},
            },
            "trained_negatives_descoped": descoped,
            "bystander_panel": bystanders,
            "eval_questions": q_eval,
            "read_type": "teacher_forced_frozen_R_eval",
            # Round-5 provenance: which weights were ACTUALLY applied, at
            # which effective scaling (adapter sha256 + use_rslora patch).
            "adapter_provenance": adapter_provenance,
            "stats": stats,
            "git_commit": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, out_path)
        log.info("[tf %s] written → %s", key, out_path)
    return 0


def _prepare_onpolicy_launch_rows(plan: list[dict]) -> list[tuple[str, list[str], Path]]:
    """Filter the 0a-op worker plan to rows needing a (re-)read + write indices.

    Round-5 stale-output rule: an existing trajectory is reused ONLY when it
    was produced under the parity-read regime (terminal checkpoint carries
    ``provenance.use_rslora_applied == False``). Round-4 ceiling-pinned
    trajectories (no provenance) are deleted and re-read.
    """
    rows: list[tuple[str, list[str], Path]] = []
    for row in plan:
        key = row["key"]
        out_dir = Path(row["out_dir"])
        traj_path = out_dir / "trajectory.json"
        if traj_path.exists():
            prior = json.loads(traj_path.read_text())
            prior_cks = prior.get("checkpoints", [])
            prior_prov = (prior_cks[-1].get("provenance") or {}) if prior_cks else {}
            if prior_prov.get("use_rslora_applied") is False:
                log.info("[phase=p0_onpolicy] %s parity-regime output exists; skip", key)
                continue
            log.warning("[phase=p0_onpolicy] %s STALE pre-parity trajectory — re-reading", key)
            traj_path.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        # Synthetic single-checkpoint index pointing at the final adapter.
        Path(row["idx_path"]).write_text(
            json.dumps({"1.0000": {"step": None, "path": row["adapter_path"]}})
        )
        rows.append((f"op_{key}", row["cmd"], Path(row["log_path"])))
    return rows


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
                # Terminate EVERYTHING still alive (not just procs already
                # polled into `still` this iteration) so an abort never
                # orphans GPU-holding workers (round-1 review minor).
                for p2, _l2, _g2 in running:
                    if p2 is not proc and p2.poll() is None:
                        p2.terminate()
                raise RuntimeError(f"phase0 worker {label} exited rc={rc}; see {log_dir}")
            log.info("[pool] %s complete (GPU %d)", label, gpu)
        running = still
        if running:
            time.sleep(5)


def _register_bystander_panel(args, cts) -> tuple:
    """Coverage-constrained panel registration + pre-worker coverage gate.

    Concern phase0-r-eval-coverage-gap (round 2): the pinned parent artifact
    pair is mutually inconsistent (persona_bank 60 vs R_eval 61 personas, only
    45 overlap), so (1) the bystander candidate pool is restricted to held-out
    personas with COMPLETE frozen-R_eval coverage BEFORE decile selection,
    with exclusions recorded by name in ``bystander_panel.json``; (2) a
    fail-loud assert covers the registered panel; (3) BEFORE any worker
    launch, source + bystanders + the COUNT cells' trained negatives (the
    Phase-0b clamp test needs >=3 of 4 count cells, both seeds) must ALL be
    covered. Non-count parent cells MAY have uncovered trained negatives
    (c472_near: mob_boss + cult_leader; c472_negp_8: baker) — those reads are
    descoped per-persona by the worker with coverage="absent-from-frozen-R".
    Frozen-R parity with #472 is load-bearing: R is NEVER regenerated here.

    Returns:
        (bystanders, r_eval, q_eval, panel_all, excluded_no_r, descope_map).
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        held_out_panel,
        negatives_for_cell,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import (
        COUNT_CELL_LEVELS,
        N_BYSTANDER_REFERENCE,
        PARENT_CELLS_ALL,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.neg_setpoint_601.phase0_lib import (
        assert_r_eval_coverage,
        select_bystander_reference_panel,
        split_by_r_coverage,
    )

    panel_all = held_out_panel(cts, source=SOURCE_PERSONA)
    r_eval = load_r_artifact(args.data_dir / "on_policy_R" / "R_eval.json")
    _q_train, q_eval = get_train_eval_questions()
    covered_held_out, excluded_no_r = split_by_r_coverage(r_eval, panel_all, q_eval)
    bystanders = select_bystander_reference_panel(covered_held_out, cts, n=N_BYSTANDER_REFERENCE)
    # Fail-loud at panel REGISTRATION (belt) — selection over the covered pool
    # makes this a tautology unless the artifacts drift under us.
    assert_r_eval_coverage(r_eval, bystanders, q_eval, context="bystander panel registration")
    (args.phase0_dir / "bystander_panel.json").write_text(
        json.dumps(
            {
                "schema_version": "i601_bystander_panel_v2",
                "personas": bystanders,
                "d_source": {p: 1.0 - cts[p] for p in bystanders},
                "selection": (
                    "L10 d_source deciles over the R_eval-covered subset of the #472 held-out panel"
                ),
                "n_held_out": len(panel_all),
                "n_held_out_r_eval_covered": len(covered_held_out),
                "excluded_no_full_r_eval": excluded_no_r,
                "git_commit": _git_sha(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    log.info(
        "[phase=p0_panel] pre-registered bystanders (R_eval-covered pool %d/%d): %s "
        "(excluded, no full frozen-R: %s)",
        len(covered_held_out),
        len(panel_all),
        bystanders,
        excluded_no_r,
    )

    # Pre-worker-launch coverage gate.
    count_cells_gate = [c for c, _lvl in COUNT_CELL_LEVELS]
    must_cover = [SOURCE_PERSONA, *bystanders]
    for c in count_cells_gate:
        must_cover.extend(p for p in negatives_for_cell(c, cts) if p not in must_cover)
    assert_r_eval_coverage(
        r_eval,
        must_cover,
        q_eval,
        context="pre-worker launch (source + bystanders + count-cell negatives)",
    )
    descope_map: dict[str, list[str]] = {}
    for c in PARENT_CELLS_ALL:
        uncovered_negs = split_by_r_coverage(r_eval, negatives_for_cell(c, cts), q_eval)[1]
        if uncovered_negs:
            descope_map[c] = uncovered_negs
    if descope_map:
        log.warning(
            "[phase=p0_coverage] trained-negative reads descoped (absent-from-frozen-R): %s",
            descope_map,
        )
    return bystanders, r_eval, q_eval, panel_all, excluded_no_r, descope_map


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
        negatives_for_cell,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import (
        COUNT_CELL_LEVELS,
        HF_DATA_PREFIX_601,
        PARENT_CELLS_ALL,
        PARENT_SEEDS,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.neg_setpoint_601.artifacts import (
        fetch_parent_adapter,
        fetch_parent_data,
    )
    from explore_persona_space.experiments.neg_setpoint_601.phase0_lib import (
        IdenticalRereadAlarm,
        clamp_read,
        compute_gate_schema2,
        decide_primary_space,
        margin_references,
        onpolicy_crosscheck,
        onpolicy_worker_plan,
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

    # ── Bystander reference panel (pre-registered by name) + coverage gate. ──
    # Coverage-constrained selection + the pre-worker-launch fail-loud gate
    # live in _register_bystander_panel (concern phase0-r-eval-coverage-gap).
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.data_dir)
    (bystanders, _r_eval, _q_eval, panel_all, excluded_no_r, descope_map) = (
        _register_bystander_panel(args, cts)
    )

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
    # The cell→worker mapping is the pure (CPU-tested) onpolicy_worker_plan —
    # the round-5 brief's mapping-construction smoke asserts key∈adapter_path
    # per worker against THIS function, so the smoke and the production launch
    # share one mapping implementation.
    onpolicy_keys = [f"{c}_seed{s}" for c, _lvl in COUNT_CELL_LEVELS for s in PARENT_SEEDS]
    if not args.skip_onpolicy:
        plan = onpolicy_worker_plan(
            onpolicy_keys, args.phase0_dir, args.adapters_root, args.data_dir, args.log_dir
        )
        op_cmds = _prepare_onpolicy_launch_rows(plan)
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
        try:
            crosscheck = onpolicy_crosscheck(reread, committed)
        except IdenticalRereadAlarm as alarm:
            # Round-5 fix #3: the identical-read pathology is a LOUD, named
            # error — persist durable gate evidence, then crash the driver.
            (args.phase0_dir / "phase0_gate.json").write_text(
                json.dumps(
                    {
                        "gate_schema": 2,
                        "pass": False,
                        "anchor_reuse_ok": False,
                        "gate": "Gate S structural FAIL (plan v3 §B item 1, #534 class)",
                        "structural_alarm": "IdenticalRereadAlarm",
                        "detail": str(alarm),
                        "diag": alarm.diag,
                        "git_commit": _git_sha(),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                )
            )
            raise
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

    # Plan v3 §B gate split (gate_schema 2): Gate S = the file's top-level
    # `pass` (structural, HALT-class); Gate A = `anchor_reuse_ok` (routing);
    # Observation O recorded, never gating. The v2 conjunctive rule survives
    # only as the audit `onpolicy_crosscheck` table. Pure recompute over the
    # per-adapter table → skip-cheap re-runs regate without GPU work.
    if not args.skip_onpolicy:
        gate2 = compute_gate_schema2(crosscheck["per_adapter"], recipe_panel_ok=recipe_panel_ok)
    else:
        gate2 = {
            "gate_schema": 2,
            "pass": False,
            "anchor_reuse_ok": False,
            "skipped": True,
            "note": "--skip-onpolicy: no re-reads — Gate S cannot be established; HALT stands.",
        }
    anchor_reuse_ok = bool(gate2["anchor_reuse_ok"])
    anchor_onpolicy_ok = bool(gate2.get("gate_a", {}).get("anchor_onpolicy_ok", False))

    endpoint = {
        "schema_version": "i601_phase0_v2",
        "bystander_panel": bystanders,
        "bystander_excluded_no_full_r_eval": excluded_no_r,
        "trained_negatives_descoped_by_cell": descope_map,
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
        **gate2,
        "gate": "phase0 gate split: Gate S structural / Gate A anchor reuse (plan v3 §B)",
        "onpolicy_crosscheck": crosscheck,
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
