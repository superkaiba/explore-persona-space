#!/usr/bin/env python3
"""Pod-side driver for the #1739 r2v2 P-A/P-B fits (CPU pod, one behavior chain).

Structural sibling of ``issue1739_result2fair_run.py`` with the r2v2 deltas:
per behavior STAGE (jobd stage_inputs: wcrung store + DVs + E1 + labeling tar
slice; pvsynth via the pvsynth-arms helper; PLUS the behavior's NEW OOD
stores/DV mirrored from ``issue1739_ctxmap/`` via ``hub.stage_hub_prefix``)
-> SCORE (``issue1739_r2v2_score.py`` subprocess, explicit rc) -> UPLOAD
(out-root subtree -> HF data repo) -> REAP the labeling slice -> per-behavior
sentinel to /workspace/logs for the VM-side poller.

Behaviors run SEQUENTIALLY within one pod (the merged fp64 tables peak
~35-55 GB per behavior — two concurrent behaviors do not fit a 128 GB
cpu-bigmem box); cross-behavior concurrency = one CPU pod per behavior
(CPU pods may run in parallel; each invokes this driver with its own
``--behaviors``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_r2v2_score.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIOR_ORDER = ("evil", "sycophancy", "hallucination")
HF_OUT_PREFIX = "issue1739_r2v2_fits"
OUT_ROOT = Path("eval_results/issue_1739/r2v2_fits")
FACT_HF_OUT_PREFIX = "issue1739_r2v2_factorial"
FACT_OUT_ROOT = Path("eval_results/issue_1739/r2v2_factorial")
# λ grid-edge check (task #16): base-grid GCV pinned λ=1000 (the grid MAX) on
# every hallucination fit — re-fit under a 3-decades-wider grid, own out root.
WIDE_HF_OUT_PREFIX = "issue1739_r2v2_fits_widegrid"
WIDE_OUT_ROOT = Path("eval_results/issue_1739/r2v2_fits_widegrid")
WIDE_RIDGE_LAMBDAS = "0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0,1000000.0"
# P-C LODO-consistent map+readout (2026-08-07 inline round): own out root +
# HF prefix; the scorer runs with --protocols C (per-holdout map refit).
PC_HF_OUT_PREFIX = "issue1739_r2v2_pc"
PC_OUT_ROOT = Path("eval_results/issue_1739/r2v2_pc")
# arm12 re-score (2026-08-07 inline round): the committed P-A/P-B round scored
# five arms and never arm12_oracle_reg ("Ridge regression on real answer"), so
# putting that arm on a P-A/P-B figure needs a re-score, not a re-read. Same
# protocols + same inputs as the `fits` leg, roster extended by one arm, OWN
# out root + HF prefix so the committed five-arm results are never overwritten
# (which also makes the five shared arms a free reproduction check).
ARM12_HF_OUT_PREFIX = "issue1739_r2v2_fits_arm12"
ARM12_OUT_ROOT = Path("eval_results/issue_1739/r2v2_fits_arm12")
ARM12_EXTRA_ARMS = ("arm12_oracle_reg",)
# claim4-controls (plan v21, 2026-08-19): seed-replicated P-B fits under BOTH
# map variants (true + pairing-shuffled control), roster extended by arm2 +
# arm20, per-context transfer preds ON for every behavior (the P2 paired
# context-bootstrap + companion join consume them). One scorer SUBPROCESS per
# seed (resume-friendly; outputs keyed <behavior>/seed<S>/), protocols pinned
# to B (items 1-3 are P-B questions; the P-A rows item 4 needs are banked).
CLAIM4_HF_OUT_PREFIX = "issue1739_claim4_controls"
CLAIM4_OUT_ROOT = Path("eval_results/issue_1739/claim4_controls")
CLAIM4_EXTRA_ARMS = ("arm2_ctx_native", "arm20_shuffled_map_ridge")
CLAIM4_MAP_VARIANTS = ("true", "shufpair")
CTXMAP_PREFIX = "issue1739_ctxmap"
BANK_PREFIX = f"{CTXMAP_PREFIX}/rb_fc_bank"
HALLU_PR_FILE = f"{CTXMAP_PREFIX}/judge/hallucination/labeling_per_rollout.json"
# HF prefixes staged per behavior (verbatim mirrors under --ood-mirror-root).
OOD_STAGE_PREFIXES = {
    "evil": (f"{CTXMAP_PREFIX}/evil_ood_full/store",),
    "sycophancy": (
        f"{CTXMAP_PREFIX}/syco_ood/store",
        f"{CTXMAP_PREFIX}/syco_ood/dv_dataset",
    ),
    "hallucination": (),
}


def _log(msg: str) -> None:
    print(f"[r2v2-run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jobd_ns(args, behavior: str) -> argparse.Namespace:
    """Namespace shaped for issue1739_jobd_r2aug_run.stage_inputs."""
    return argparse.Namespace(
        behavior=behavior,
        store_root=args.store_root,
        wcrung_root=args.wcrung_root,
        tensors_root=args.tensors_root,
        revision=args.revision,
        stage_workers=args.stage_workers,
    )


def _pv_ns(args) -> argparse.Namespace:
    """Namespace shaped for issue1739_pvsynth_arms_run.stage_shared."""
    return argparse.Namespace(
        store_root=args.store_root,
        behaviors=list(args.behaviors),
        train_dv_root=args.store_root / "train_dv",
    )


def stage_ood(args, behavior: str, token: str) -> None:
    """Mirror the behavior's OOD store/DV prefixes under the ood mirror root.

    ``stage_hub_prefix`` files land at ``<mirror_root>/<repo-relative path>``
    (verbatim prefix mirror — the score script's --ood-store-root consumes
    ``<mirror_root>/issue1739_ctxmap/...``). Idempotent per file: a partial
    mirror self-heals on re-run.
    """
    from explore_persona_space.orchestrate import hub

    for prefix in OOD_STAGE_PREFIXES.get(behavior, ()):
        # cheap completeness probe: skip a prefix whose remote file SET is
        # already fully present locally (stage_hub_prefix re-lists otherwise)
        t0 = time.time()
        files = hub.stage_hub_prefix(
            hub.DEFAULT_DATASET_REPO,
            prefix,
            args.ood_mirror_root,
            repo_type="dataset",
            token=token or None,
            max_workers=min(args.stage_workers, 6),
        )
        _log(
            f"[phase=stage_ood {behavior}] {prefix}: {len(files)} files in "
            f"{time.time() - t0:.0f}s -> {args.ood_mirror_root / prefix}"
        )


def stage_factorial_inputs(args, behavior: str, token: str) -> None:
    """Factorial-leg extras: banked fc directions + the hallucination
    per-rollout DV + the E1 extraction store FORCED (e1_fc reads the store's
    context_end shards even when the r_b_e1 bank makes the base leg skip it)."""
    from explore_persona_space.orchestrate import hub
    from scripts.issue1739_wcrung_arms_run import stage_extraction

    files = hub.stage_hub_prefix(
        hub.DEFAULT_DATASET_REPO,
        BANK_PREFIX,
        args.ood_mirror_root,
        repo_type="dataset",
        token=token or None,
        max_workers=4,
    )
    _log(f"[phase=stage_factorial {behavior}] rb_fc_bank: {len(files)} files")
    if behavior == "hallucination":
        dest = args.ood_mirror_root / HALLU_PR_FILE
        if not dest.exists():
            hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO,
                HALLU_PR_FILE,
                dest,
                repo_type="dataset",
                token=token,
            )
        _log(f"[phase=stage_factorial {behavior}] per-rollout DV -> {dest}")
    ns = argparse.Namespace(
        store_root=args.store_root,
        tensors_root=args.tensors_root,
        revision=args.revision,
        stage_workers=args.stage_workers,
    )
    stage_extraction(behavior, ns, token, force=True)


def factorial_cmd(args, behavior: str) -> list[str]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_r2v2_factorial.py"),
        "--behaviors",
        behavior,
        "--variant",
        "context_end",
        "--protocols",
        args.protocols,
        "--store-root",
        str(args.store_root),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--out-root",
        str(OUT_ROOT),
        "--fact-out-root",
        str(FACT_OUT_ROOT),
        "--rb-bank-dir",
        str(args.ood_mirror_root / BANK_PREFIX),
        "--ood-store-root",
        str(args.ood_mirror_root / CTXMAP_PREFIX),
        "--device",
        args.device,
        "--ood-dv-max-null-frac",
        str(args.ood_dv_max_null_frac),
    ]
    if behavior == "hallucination":
        cmd += ["--hallu-per-rollout-dv", str(args.ood_mirror_root / HALLU_PR_FILE)]
    if args.pb_holdouts:
        cmd += ["--pb-holdouts", *args.pb_holdouts]
    return cmd


def score_cmd(
    args,
    behavior: str,
    out_root: Path = OUT_ROOT,
    extra_arms: tuple[str, ...] = (),
    transfer_preds: bool = False,
) -> list[str]:
    """Compose the scorer argv for one behavior.

    ``extra_arms`` / ``transfer_preds`` default to the committed `fits`-leg
    shape (empty / off), so every existing caller composes a byte-identical
    argv; only the arm12 leg passes them.
    """
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_r2v2_score.py"),
        "--behaviors",
        behavior,
        "--variant",
        "context_end",
        "--protocols",
        args.protocols,
        "--store-root",
        str(args.store_root),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--out-root",
        str(out_root),
        "--ood-store-root",
        str(args.ood_mirror_root / CTXMAP_PREFIX),
        "--device",
        args.device,
        "--ood-dv-max-null-frac",
        str(args.ood_dv_max_null_frac),
    ]
    if extra_arms:
        cmd += ["--extra-arms", *extra_arms]
    if transfer_preds:
        cmd.append("--transfer-preds")
    if args.pb_holdouts:
        cmd += ["--pb-holdouts", *args.pb_holdouts]
    return cmd


# leg -> (local out root, HF prefix); every leg the driver can run.
LEG_DESTS = {
    "fits": (OUT_ROOT, HF_OUT_PREFIX),
    "factorial": (FACT_OUT_ROOT, FACT_HF_OUT_PREFIX),
    "fits-widegrid": (WIDE_OUT_ROOT, WIDE_HF_OUT_PREFIX),
    "pc": (PC_OUT_ROOT, PC_HF_OUT_PREFIX),
    "fits-arm12": (ARM12_OUT_ROOT, ARM12_HF_OUT_PREFIX),
    "fits-claim4": (CLAIM4_OUT_ROOT, CLAIM4_HF_OUT_PREFIX),
}


def claim4_score_cmd(
    args,
    behavior: str,
    seed: int | None = None,
    *,
    seeds: list[int] | None = None,
    out_root: Path | None = None,
    layers: list[int] | None = None,
    pb_holdouts: list[str] | None = None,
) -> list[str]:
    """Compose the claim4 scorer argv for ONE (behavior, seed) invocation.

    --protocols is PINNED to B (never args.protocols): the claim4 leg is a
    P-B-only leg by plan; a pod launch composing this leg with --protocols AB
    must not silently re-fit P-A. Both map variants + both extra arms +
    --transfer-preds always ride (plan v21 §4 P0.3). The keyword overrides
    exist ONLY for the real-store smoke phase (multi-seed single invocation,
    scratch out-root, 2-layer slice, 1 holdout); production calls pass a bare
    ``seed`` and stay byte-identical.
    """
    seed_list = seeds if seeds is not None else [int(seed)]
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "issue1739_r2v2_score.py"),
        "--behaviors",
        behavior,
        "--variant",
        "context_end",
        "--protocols",
        "B",
        "--store-root",
        str(args.store_root),
        "--main-root",
        str(args.main_root),
        "--tensors-root",
        str(args.tensors_root),
        "--out-root",
        str(out_root if out_root is not None else CLAIM4_OUT_ROOT),
        "--ood-store-root",
        str(args.ood_mirror_root / CTXMAP_PREFIX),
        "--device",
        args.device,
        "--ood-dv-max-null-frac",
        str(args.ood_dv_max_null_frac),
        "--seeds",
        *[str(int(s)) for s in seed_list],
        "--map-variants",
        *CLAIM4_MAP_VARIANTS,
        "--extra-arms",
        *CLAIM4_EXTRA_ARMS,
        "--transfer-preds",
    ]
    if layers is not None:
        cmd += ["--layers", *[str(int(li)) for li in layers]]
    holdouts = pb_holdouts if pb_holdouts is not None else args.pb_holdouts
    if holdouts:
        cmd += ["--pb-holdouts", *holdouts]
    return cmd


def _runner_git_sha() -> str:
    """The running (post-merge) code SHA — the breadcrumb identity key."""
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _stage_crumb_path(behavior: str, run_token: str) -> str:
    """HF breadcrumb path the seeds-0-2 pod writes at stage completion.

    RUN-TOKEN-KEYED (plan §9 serialized staging; review r1 both-twins fix):
    a prior run's breadcrumb lives at a DIFFERENT path, so bare existence at
    this run's path can never be satisfied by a stale crumb. The payload
    additionally carries the post-merge code SHA + the token, and the waiter
    verifies BOTH (identity + freshness, never bare existence).
    """
    return f"{CLAIM4_HF_OUT_PREFIX}/_staging/{behavior}_stage_done_{run_token}.json"


def _crumb_matches(payload: dict, *, code_sha: str, run_token: str) -> tuple[bool, str]:
    """Identity check on a downloaded stage-done breadcrumb payload."""
    if payload.get("run_token") != run_token:
        return False, f"run_token mismatch ({payload.get('run_token')!r} != {run_token!r})"
    if payload.get("code_sha") != code_sha:
        return False, f"code_sha mismatch ({payload.get('code_sha')!r} != {code_sha!r})"
    return True, "match"


def signal_stage_done(args, behavior: str, token: str) -> None:
    """Upload the stage-done breadcrumb (serialized-staging topology, plan §9)."""
    import socket
    import tempfile

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    payload = {
        "behavior": behavior,
        "host": socket.gethostname(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": _runner_git_sha(),
        "run_token": args.stage_run_token,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp = f.name
    crumb = _stage_crumb_path(behavior, args.stage_run_token)
    hub.retry_transient(
        lambda: HfApi().upload_file(
            path_or_fileobj=tmp,
            path_in_repo=crumb,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            token=token or None,
        ),
        what="claim4-stage-crumb-upload",
    )
    os.unlink(tmp)
    _log(
        f"[phase=stage_signal {behavior}] breadcrumb -> {crumb} "
        f"(code_sha={payload['code_sha'][:12]} run_token={args.stage_run_token})"
    )


def wait_for_sibling_stage(args, behavior: str, token: str) -> None:
    """Block until THIS RUN's sibling stage-done breadcrumb exists on HF.

    The seeds-3-4 pod polls before staging so two pods never pull the same
    multi-GB prefix concurrently (the evil-ood-spread round drew rate-limit /
    connection kills from exactly that shape). The gate checks IDENTITY +
    freshness, never bare existence: the crumb path is run-token-keyed and
    the downloaded payload must carry this run's token AND this checkout's
    post-merge code SHA — a prior run's breadcrumb can never satisfy it. On
    timeout: proceed with a LOUD warning — the sibling has almost certainly
    crashed (measured stage wall is 25-40 min), and serialization is
    HF-politeness, not correctness.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    crumb = _stage_crumb_path(behavior, args.stage_run_token)
    own_sha = _runner_git_sha()
    api = HfApi(token=token or None)
    t0 = time.time()
    while time.time() - t0 < args.stage_gate_timeout_s:
        try:
            # HUB_VERIFY_RETRY_EXEMPT: outer poll loop IS the retry — probe errors are caught below and re-polled every stage_gate_poll_s
            if api.file_exists(hub.DEFAULT_DATASET_REPO, crumb, repo_type="dataset"):
                # NO_RETRY: outer stage_wait poll loop IS the retry — errors are caught below and re-polled every stage_gate_poll_s
                local = api.hf_hub_download(
                    hub.DEFAULT_DATASET_REPO, crumb, repo_type="dataset", force_download=True
                )
                payload = json.loads(Path(local).read_text())
                ok, why = _crumb_matches(payload, code_sha=own_sha, run_token=args.stage_run_token)
                if ok:
                    _log(
                        f"[phase=stage_wait {behavior}] sibling breadcrumb VERIFIED "
                        f"({crumb}; code_sha={own_sha[:12]})"
                    )
                    return
                _log(
                    f"[phase=stage_wait {behavior}] breadcrumb at {crumb} REJECTED "
                    f"({why}) — treating as absent, still polling"
                )
        except Exception as exc:  # noqa: BLE001 — transient Hub errors: keep polling
            _log(f"[phase=stage_wait {behavior}] probe error (retrying): {exc}")
        time.sleep(args.stage_gate_poll_s)
    _log(
        f"[phase=stage_wait {behavior}] WARNING: sibling breadcrumb ABSENT after "
        f"{args.stage_gate_timeout_s}s ({crumb}) — sibling likely dead; proceeding "
        "with staging anyway (serialization is politeness, not correctness)"
    )


def run_claim4_leg(args, behavior: str) -> dict:
    """The fits-claim4 leg: one scorer subprocess per seed + the seed-0 gate.

    Per-seed sequencing (resume-friendly; a crashed seed re-runs alone):
    score seed S -> upload the behavior subtree (checkpoint-per-phase) ->
    after seed 0, run the claim4 reproduction gate (plan §7 gate 1) and HALT
    the remaining seed chain on FAIL (never spend seeds 1-4 on a drifted
    pipeline). The repro report is keyed by the running commit SHA inside the
    gate script itself (dual pre/post-merge protocol, plan §4 P0.4).
    """
    per_seed: dict[str, dict] = {}
    rc_leg = 0
    for i, seed in enumerate(args.seeds, start=1):
        t_seed = time.time()
        cmd = claim4_score_cmd(args, behavior, seed)
        _log(
            f"[phase=score {behavior} leg=fits-claim4 seed={seed} ({i}/{len(args.seeds)})] "
            f"{' '.join(cmd[1:])}"
        )
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False, env={**os.environ})
        _log(
            f"[phase=score {behavior} leg=fits-claim4 seed={seed}] rc={proc.returncode} "
            f"elapsed={time.time() - t_seed:.1f}s"
        )
        url = None
        if proc.returncode == 0 and not args.skip_upload:
            url = upload_behavior(args, behavior, leg="fits-claim4")
        entry: dict = {
            "rc": proc.returncode,
            "uploaded": bool(url),
            "upload_url": url,
            "wall_s": round(time.time() - t_seed, 1),
        }
        per_seed[f"seed{seed}"] = entry
        if proc.returncode != 0:
            rc_leg = proc.returncode
            break  # a failed seed never gates a later seed's inputs silently
        if int(seed) == 0:
            gate_cmd = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "issue1739_arm12_repro_check.py"),
                "--mode",
                "claim4",
                "--behaviors",
                behavior,
                "--new-root",
                str(CLAIM4_OUT_ROOT),
            ]
            _log(f"[phase=gate1 {behavior}] {' '.join(gate_cmd[1:])}")
            gate = subprocess.run(gate_cmd, cwd=str(_REPO_ROOT), check=False, env={**os.environ})
            entry["gate1_rc"] = gate.returncode
            report_dir = CLAIM4_OUT_ROOT / "repro_claim4"
            if not args.skip_upload and report_dir.exists():
                # the gate writes its SHA-keyed report under <out-root>/repro_claim4/
                from explore_persona_space.orchestrate import hub

                base_url = hub._upload(
                    report_dir,
                    hub.DEFAULT_DATASET_REPO,
                    "dataset",
                    f"{CLAIM4_HF_OUT_PREFIX}/repro_claim4",
                    raise_on_error=True,
                )
                if not base_url:
                    raise RuntimeError(
                        "claim4 gate-1 report upload returned no path "
                        f"({CLAIM4_HF_OUT_PREFIX}/repro_claim4) — durability loss"
                    )
                _log(
                    f"[phase=gate1 {behavior}] report uploaded -> {CLAIM4_HF_OUT_PREFIX}/repro_claim4"
                )
            if gate.returncode != 0:
                _log(
                    f"[phase=gate1 {behavior}] FAIL (rc={gate.returncode}) — HALTING the "
                    "seed chain (plan §7 kill (a): never spend seeds 1-4 on a drifted "
                    "pipeline)"
                )
                rc_leg = 3
                break
    return {"rc": rc_leg, "per_seed": per_seed}


def _smoke_layers(args, behavior: str) -> list[int]:
    """Resolve the smoke's 2-layer slice from the COMMITTED frozen layers.

    The scorer freezes every roster arm at its committed modal layer
    (matched-companion arms at their reference arm's), so the smoke slice
    must COVER those layers — resolve them the same way the scorer does and
    pad with a neighbor when they collapse to one. Fails loud on a miss (the
    same committed_frozen failure the production run would hit — the smoke
    is exactly the phase meant to surface it early).
    """
    from scripts.issue1739_r2v2_score import MATCHED_FROZEN_COMPANIONS, ROSTER
    from scripts.issue1739_wcrung_arms import modal_frozen_layers

    summary = args.main_root / behavior / "arm_results" / "all_arms_spearman.json"
    frozen = modal_frozen_layers(summary, variant="context_end", regime="e1", u_rung_label="full")
    needed = (set(ROSTER) | set(CLAIM4_EXTRA_ARMS)) - set(MATCHED_FROZEN_COMPANIONS)
    missing = sorted(a for a in needed if a not in frozen)
    if missing:
        raise RuntimeError(
            f"[claim4_smoke {behavior}] no committed frozen layer for {missing} "
            f"in {summary} — cannot compose the smoke layer slice"
        )
    layers = sorted({int(frozen[a]) for a in needed})
    if len(layers) == 1:
        base = layers[0]
        layers = sorted({base, base + 1 if base + 1 < 28 else base - 1})
    return layers


def run_claim4_smoke(args, behavior: str) -> dict:
    """MANDATORY first-pod real-store smoke phase (plan §10 P0 smoke shape).

    One behavior, the resolved frozen-layer slice (2 layers), seeds 0-1 in a
    SINGLE scorer invocation, both map variants + both extra arms +
    transfer preds (they always ride claim4_score_cmd), ONE P-B holdout, a
    SCRATCH out-root (wiped first, so the smoke actually RUNS each launch —
    never a resume-skip). Emits a phase record (exact command, slice size,
    rc, output paths + row-count digests) to the sentinel dir BEFORE any
    production seed runs; a non-zero rc aborts the launch in main().
    """
    layers = _smoke_layers(args, behavior)
    smoke_root = args.smoke_out_root
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    cmd = claim4_score_cmd(
        args,
        behavior,
        seeds=[0, 1],
        out_root=smoke_root,
        layers=layers,
        pb_holdouts=[args.smoke_holdout],
    )
    _log(f"[phase=claim4_smoke {behavior}] layers={layers} {' '.join(cmd[1:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False, env={**os.environ})
    outputs: dict[str, dict] = {}
    for s in (0, 1):
        p = smoke_root / behavior / f"seed{s}" / "all_arms_spearman.json"
        preds_dir = smoke_root / behavior / f"seed{s}" / "transfer_preds"
        n_rows = None
        if p.exists():
            n_rows = json.loads(p.read_text()).get("n_transfer_rows")
        outputs[f"seed{s}"] = {
            "summary": str(p),
            "exists": p.exists(),
            "n_transfer_rows": n_rows,
            "n_preds_files": len(list(preds_dir.glob("*.jsonl"))) if preds_dir.exists() else 0,
        }
    record = {
        "phase": "claim4_smoke",
        "behavior": behavior,
        "cmd": cmd,
        "slice": {
            "layers": layers,
            "seeds": [0, 1],
            "pb_holdouts": [args.smoke_holdout],
            "map_variants": list(CLAIM4_MAP_VARIANTS),
            "extra_arms": list(CLAIM4_EXTRA_ARMS),
        },
        "rc": proc.returncode,
        "out_root": str(smoke_root),
        "outputs": outputs,
        "wall_s": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.sentinel_dir.mkdir(parents=True, exist_ok=True)
    rec_path = args.sentinel_dir / f"issue-1739-claim4smoke-{behavior}.json"
    rec_path.write_text(json.dumps(record, indent=1))
    _log(
        f"[phase=claim4_smoke {behavior}] rc={proc.returncode} "
        f"wall={record['wall_s']}s record -> {rec_path}"
    )
    return record


def leg_cmd_env(args, behavior: str, leg: str) -> tuple[list[str], dict[str, str]]:
    """Compose one leg's subprocess argv + env overlay (explicit, never implicit)."""
    if leg == "fits":
        return score_cmd(args, behavior), {}
    if leg == "fits-widegrid":
        # identical fits, 3-decades-wider GCV grid, own (untracked) out root;
        # the env var is read ONCE at constants import in the child process.
        return (
            score_cmd(args, behavior, out_root=WIDE_OUT_ROOT),
            {"EPS_I1739_RIDGE_LAMBDAS": WIDE_RIDGE_LAMBDAS},
        )
    if leg == "factorial":
        return factorial_cmd(args, behavior), {}
    if leg == "pc":
        # P-C rides the same scorer with --protocols C (the driver caller
        # passes --protocols C) and its own out root / HF prefix.
        return score_cmd(args, behavior, out_root=PC_OUT_ROOT), {}
    if leg == "fits-arm12":
        # Same protocols + inputs as `fits`, roster extended by arm12, own out
        # root. --transfer-preds rides along: this leg re-runs the fits anyway,
        # so banking per-context predictions here makes any later CI/subset
        # re-read a pure re-analysis instead of a third re-score.
        return (
            score_cmd(
                args,
                behavior,
                out_root=ARM12_OUT_ROOT,
                extra_arms=ARM12_EXTRA_ARMS,
                transfer_preds=True,
            ),
            {},
        )
    raise ValueError(f"unknown leg {leg!r}")


def upload_behavior(args, behavior: str, leg: str = "fits") -> str:
    from explore_persona_space.orchestrate import hub

    out_root, prefix = LEG_DESTS[leg]
    local = out_root / behavior
    if not local.exists():
        raise FileNotFoundError(f"nothing to upload: {local} absent")
    url = hub._upload(
        local,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        f"{prefix}/{behavior}",
        raise_on_error=True,
    )
    _log(f"[phase=upload {behavior} leg={leg}] -> {prefix}/{behavior} ({url})")
    return url


def reap_labeling_slice(args, behavior: str) -> None:
    """Free the behavior's staged labeling slice (re-downloadable from HF).

    Fail-loud rmtree. Never touches u_store / wcrung / pvsynth / E1 / the
    OOD mirror (the OOD stores are small and shared across re-runs).
    """
    dest = args.store_root / f"{behavior}_labeling"
    if not dest.exists():
        _log(f"[phase=reap {behavior}] labeling slice absent, nothing to reap")
        return
    shutil.rmtree(dest)
    _log(f"[phase=reap {behavior}] reaped {dest}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=["sycophancy"], choices=list(BEHAVIOR_ORDER))
    ap.add_argument("--protocols", default="AB", choices=["A", "B", "AB", "C", "ABC"])
    ap.add_argument("--pb-holdouts", nargs="+", default=None)
    ap.add_argument("--store-root", type=Path, default=Path("data/issue_1739/hf_dl"))
    ap.add_argument("--main-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument(
        "--wcrung-root", type=Path, default=Path("eval_results/issue_1739/wildchat_rung")
    )
    ap.add_argument("--tensors-root", type=Path, default=Path("analysis_tensors/issue_1739"))
    ap.add_argument(
        "--ood-mirror-root",
        type=Path,
        default=None,
        help="mirror root for OOD prefixes (default: <store-root>/ood_mirror)",
    )
    ap.add_argument("--ood-dv-max-null-frac", type=float, default=0.05)
    ap.add_argument(
        "--legs",
        nargs="+",
        default=["fits"],
        choices=["fits", "factorial", "fits-widegrid", "pc", "fits-arm12", "fits-claim4"],
        help="which scoring legs to run per behavior (in the given order)",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="fits-claim4 leg only: seeds run SEQUENTIALLY in-pod, one scorer subprocess "
        "each (pod halves: '--seeds 0 1 2' / '--seeds 3 4' per the plan §9 sharding)",
    )
    ap.add_argument(
        "--stage-wait-sibling",
        action="store_true",
        help="before staging, poll HF for the sibling pod's stage-done breadcrumb "
        "(serialized staging, plan §9: the seeds-3-4 pod passes this)",
    )
    ap.add_argument(
        "--stage-signal-done",
        action="store_true",
        help="after staging completes, upload the stage-done breadcrumb "
        "(the seeds-0-2 pod passes this)",
    )
    ap.add_argument(
        "--stage-run-token",
        default=None,
        help="shared run token keying the stage-done breadcrumb path + payload (REQUIRED "
        "with --stage-wait-sibling / --stage-signal-done; e.g. the launch timestamp both "
        "pods are handed) — the sibling gate checks token + code-SHA identity, never bare "
        "existence, so a prior run's breadcrumb can never satisfy this run's gate",
    )
    ap.add_argument("--stage-gate-timeout-s", type=int, default=7200)
    ap.add_argument("--stage-gate-poll-s", type=int, default=60)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="run the MANDATORY real-store smoke phase first (plan §10 P0 shape: one "
        "behavior, 2 frozen layers, seeds 0-1, both map variants, both extra arms, one "
        "P-B holdout, scratch out-root) and emit its phase record BEFORE any production "
        "seed; a non-zero smoke rc aborts the launch",
    )
    ap.add_argument("--smoke-behavior", default="evil", choices=list(BEHAVIOR_ORDER))
    ap.add_argument("--smoke-holdout", default="toxicchat")
    ap.add_argument(
        "--smoke-out-root",
        type=Path,
        default=Path("eval_results/issue_1739/claim4_controls_smoke"),
        help="SCRATCH out-root for the smoke phase (wiped each launch; never the "
        "production claim4 out-root)",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--stage-workers", type=int, default=12)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--no-reap", action="store_true")
    ap.add_argument(
        "--stage-only",
        action="store_true",
        help="stage every input then exit 0 (pilot/verification aid)",
    )
    ap.add_argument("--sentinel-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.ood_mirror_root is None:
        args.ood_mirror_root = args.store_root / "ood_mirror"
    if args.stage_wait_sibling or args.stage_signal_done:
        if not args.stage_run_token or not re.fullmatch(r"[A-Za-z0-9_-]+", args.stage_run_token):
            ap.error(
                "--stage-run-token (path-safe, [A-Za-z0-9_-]+) is REQUIRED with "
                "--stage-wait-sibling / --stage-signal-done — the breadcrumb gate is "
                "identity-keyed, never bare existence"
            )
    if args.smoke:
        if "fits-claim4" not in args.legs:
            ap.error("--smoke is the claim4 real-store smoke — it requires the fits-claim4 leg")
        ordered = [b for b in BEHAVIOR_ORDER if b in args.behaviors]
        if not ordered or args.smoke_behavior != ordered[0]:
            ap.error(
                "--smoke-behavior must be the FIRST behavior this launch runs — the smoke "
                "is the MANDATORY first phase, before any production seed"
            )
        if args.smoke_out_root.resolve() == CLAIM4_OUT_ROOT.resolve():
            ap.error("--smoke-out-root must be a scratch dir, never the production out-root")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.import_check:
        import inspect

        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
        from scripts.issue1739_jobd_r2aug_run import stage_inputs  # noqa: F401
        from scripts.issue1739_pvsynth_arms_run import stage_shared  # noqa: F401
        from scripts.issue1739_r2v2_factorial import parse_args as _fact_parse_args
        from scripts.issue1739_wcrung_arms_run import stage_extraction

        assert callable(hub.stage_hub_prefix)
        assert callable(hub.stage_hub_file)
        assert "force" in inspect.signature(stage_extraction).parameters
        # bind every composed leg argv (arity/keyword pin)
        from scripts.issue1739_r2v2_score import parse_args as _score_parse_args

        _fact_parse_args(factorial_cmd(args, "hallucination")[2:])
        wide_cmd, wide_env = leg_cmd_env(args, "evil", "fits-widegrid")
        _score_parse_args(wide_cmd[2:])
        assert wide_env["EPS_I1739_RIDGE_LAMBDAS"] == WIDE_RIDGE_LAMBDAS
        pc_cmd, pc_env = leg_cmd_env(args, "hallucination", "pc")
        pc_args = _score_parse_args(pc_cmd[2:])
        assert pc_env == {} and str(pc_args.out_root) == str(PC_OUT_ROOT)
        # arm12 leg: own out root, roster extended, preds on. Binding it here
        # also proves the scorer accepts the slug (its --extra-arms choices are
        # restricted, so a slug drift fails at parse rather than on the pod).
        # NOTE: this bind mutates the scorer module's ROSTER in THIS process --
        # harmless, we exit below, and it is what proves the flag takes effect.
        a12_cmd, a12_env = leg_cmd_env(args, "evil", "fits-arm12")
        a12_args = _score_parse_args(a12_cmd[2:])
        assert a12_env == {} and str(a12_args.out_root) == str(ARM12_OUT_ROOT)
        assert tuple(a12_args.extra_arms) == ARM12_EXTRA_ARMS and a12_args.transfer_preds
        # every OTHER leg stays byte-identical: no --extra-arms, no --transfer-preds
        assert "--extra-arms" not in pc_cmd and "--transfer-preds" not in pc_cmd
        assert "--extra-arms" not in wide_cmd and "--transfer-preds" not in wide_cmd
        # claim4 leg: one subprocess per seed, protocols PINNED to B, both map
        # variants + both extra arms + preds; out root / seed threading bind.
        c4_cmd = claim4_score_cmd(args, "evil", seed=3)
        c4_args = _score_parse_args(c4_cmd[2:])
        assert c4_args.protocols == "B", "claim4 leg must pin --protocols B"
        assert c4_args.seeds == [3] and c4_args.map_variants == list(CLAIM4_MAP_VARIANTS)
        assert tuple(c4_args.extra_arms) == CLAIM4_EXTRA_ARMS and c4_args.transfer_preds
        assert str(c4_args.out_root) == str(CLAIM4_OUT_ROOT)
        # gate-1 argv binds against the repro script's parser (claim4 mode)
        from scripts.issue1739_arm12_repro_check import parse_args as _repro_parse_args

        r_args = _repro_parse_args(
            ["--mode", "claim4", "--behaviors", "evil", "--new-root", str(CLAIM4_OUT_ROOT)]
        )
        assert r_args.mode == "claim4" and str(r_args.new_root) == str(CLAIM4_OUT_ROOT)
        # smoke phase argv binds (multi-seed single invocation, scratch root,
        # 2-layer slice, 1 holdout) — placeholder layers; the live run
        # resolves them from the committed frozen layers (_smoke_layers).
        sm_cmd = claim4_score_cmd(
            args,
            "evil",
            seeds=[0, 1],
            out_root=args.smoke_out_root,
            layers=[3, 4],
            pb_holdouts=["toxicchat"],
        )
        sm_args = _score_parse_args(sm_cmd[2:])
        assert sm_args.seeds == [0, 1] and sm_args.layers == [3, 4]
        assert str(sm_args.out_root) == str(args.smoke_out_root)
        assert sm_args.pb_holdouts == ["toxicchat"] and sm_args.protocols == "B"
        assert sm_args.map_variants == list(CLAIM4_MAP_VARIANTS)
        assert tuple(sm_args.extra_arms) == CLAIM4_EXTRA_ARMS and sm_args.transfer_preds
        # production claim4 argv is unchanged by the smoke overrides (the
        # keyword defaults keep the bare-seed call byte-identical: no
        # --layers slice, production out-root — asserted on c4_args above)
        assert "--layers" not in c4_cmd
        # breadcrumb identity: run-token-keyed path + payload check
        assert _stage_crumb_path("evil", "tok123").endswith("_stage_done_tok123.json")
        assert _crumb_matches(
            {"code_sha": "abc", "run_token": "tok123"}, code_sha="abc", run_token="tok123"
        )[0]
        assert not _crumb_matches(
            {"code_sha": "abc", "run_token": "OLD"}, code_sha="abc", run_token="tok123"
        )[0]
        _log("import-check OK")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    from explore_persona_space.orchestrate.env import load_dotenv
    from scripts.issue1739_jobd_r2aug_run import stage_inputs
    from scripts.issue1739_pvsynth_arms_run import stage_shared as pv_stage_shared

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    t0 = time.time()
    pv_stage_shared(_pv_ns(args), token)

    behaviors = [b for b in BEHAVIOR_ORDER if b in args.behaviors]
    results: dict[str, dict] = {}
    overall_rc = 0
    for behavior in behaviors:
        t_b = time.time()
        _log(f"=== {behavior}: stage -> score -> upload -> reap ===")
        if args.stage_wait_sibling:
            wait_for_sibling_stage(args, behavior, token)
        stage_inputs(_jobd_ns(args, behavior), token)
        stage_ood(args, behavior, token)
        if "factorial" in args.legs:
            stage_factorial_inputs(args, behavior, token)
        if args.stage_signal_done:
            signal_stage_done(args, behavior, token)
        if args.stage_only:
            _log(f"[phase=stage_only {behavior}] staging complete, skipping score")
            results[behavior] = {"score_rc": None, "staged_only": True}
            continue
        if args.smoke and behavior == args.smoke_behavior:
            # MANDATORY first phase (plan §10): the real-store smoke runs
            # against the just-staged inputs BEFORE any production seed;
            # its phase record lands in the sentinel dir either way.
            smoke_rec = run_claim4_smoke(args, behavior)
            if smoke_rec["rc"] != 0:
                _log(
                    f"[phase=claim4_smoke {behavior}] FAIL rc={smoke_rec['rc']} — "
                    "ABORTING the launch (the smoke is the mandatory first phase; "
                    "no production seed runs on a failed smoke)"
                )
                sys.stdout.flush()
                sys.stderr.flush()
                sys.exit(smoke_rec["rc"])
        leg_results: dict[str, dict] = {}
        for leg in args.legs:
            if leg == "fits-claim4":
                # multi-invocation leg (one scorer subprocess per seed +
                # the in-chain seed-0 reproduction gate) — own driver.
                leg_results[leg] = run_claim4_leg(args, behavior)
                if leg_results[leg]["rc"] != 0:
                    break
                continue
            cmd, env_overlay = leg_cmd_env(args, behavior, leg)
            _log(f"[phase=score {behavior} leg={leg}] {' '.join(cmd[1:])} env+={env_overlay}")
            proc = subprocess.run(
                cmd, cwd=str(_REPO_ROOT), check=False, env={**os.environ, **env_overlay}
            )
            _log(f"[phase=score {behavior} leg={leg}] rc={proc.returncode}")
            url = None
            if proc.returncode == 0 and not args.skip_upload:
                url = upload_behavior(args, behavior, leg)
            leg_results[leg] = {
                "rc": proc.returncode,
                "uploaded": bool(url),
                "upload_url": url,
            }
            if proc.returncode != 0:
                break  # a failed leg never gates a later leg's inputs silently
        rc_b = max(r["rc"] for r in leg_results.values())
        if rc_b == 0 and not args.no_reap:
            reap_labeling_slice(args, behavior)
        results[behavior] = {
            "score_rc": rc_b,
            "legs": leg_results,
            "wall_s": round(time.time() - t_b, 1),
        }
        overall_rc = overall_rc or rc_b
        sentinel = {
            "leg": "r2v2_fits",
            "behavior": behavior,
            **results[behavior],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        args.sentinel_dir.mkdir(parents=True, exist_ok=True)
        path = args.sentinel_dir / f"issue-1739-r2v2fits-{behavior}.json"
        path.write_text(json.dumps(sentinel, indent=1))
        _log(f"[phase=done {behavior}] sentinel -> {path}")

    summary = {
        "leg": "r2v2_fits",
        "behaviors": results,
        "overall_rc": overall_rc,
        "wall_s": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = args.sentinel_dir / "issue-1739-r2v2fits-all.json"
    path.write_text(json.dumps(summary, indent=1))
    _log(f"[phase=done] sentinel -> {path} (overall_rc={overall_rc})")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(overall_rc)


if __name__ == "__main__":
    main()
