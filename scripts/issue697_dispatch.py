"""Issue #697 dispatcher — causal context-vector (CV) patch on #537's adapters.

Decomposes whether finetuning a behavior B into a context C moved the model's
internal context picture ``c_C`` (the input to the theory's map M) or the map M
itself, by cross-model residual-stream patching at read layer L=14 on #537's
already-trained behaviorxcontext LoRA adapters (Qwen-2.5-7B-Instruct). Forward
passes only (HF ``model.forward`` / ``model.generate``; NO vLLM, plan §8).

PASS_UNIFIED architectural parity (Step 6d.0): the smoke IS the sweep with one
cell. Every phase runs each cell through the SAME per-cell function the full
sweep calls; ``--cells em_sp_swe_seed42 --cpu-only`` is the only difference. No
smoke-vs-sweep code divergence (same wave dispatcher with the same ``--n-gpus``
cell-sharding, same env injection, same sentinel, same poll contract). The
per-phase cell-list ALL derives from the same ``--cells`` subset.

Phases (plan §4.1 dependency DAG):

- ``vendor``  (CPU, 0 GPU): import-smoke — confirms the vendored
  ``analysis.activation_shift`` + ``analysis.cv_patch`` + ``experiments.issue_651``
  read path import cleanly and the 14x20 panel materializes. No models loaded.
- ``canary``  (1x A100-80, ~0.4 GPU-h): the pre-sweep no-go (plan §7).
  Gate C1.1 self-patch identity ≈0 (read + generate), Gate C1.2 non-identity
  KV-cache propagation (cache-vs-no-cache parity), Gate C1.3 decoded-token slot
  audit — all on the production panel through ``cv_patch`` — PLUS Gate C2 (the
  inherited #651 Gate 7a rsLoRA application-scaling parity, reproducing #521's
  committed marker numbers through the same ``merge_and_unload`` path). The two
  canary cells (one root-layout marker, one ``sft_em_adapter/``-nested em) double
  as the smoke-architecture canary (they run the full per-cell patch path).
- ``sweep``   (4x A100-80, ~21 GPU-h, ``--n-gpus 4``): per (B, C, seed) cell over
  the 128-cell grid (4 behaviors x 16 contexts x 2 seeds). Each cell: stage
  adapter → load base + FT (merge_and_unload) → capture c0/c+/v0/v+ → P↓/P↑ + the
  4 controls on the 14x20 panel → persist per-cell ``.pt`` (mechanistic v, both
  poolings for marker/fact) + ``_E.json`` (the patched on-policy generations for
  downstream judging; marker DV computed inline). Per-cell artifacts upload to HF
  the moment a cell completes (a mid-sweep crash strands fewer than N cells).
- ``analyze`` (CPU, 0 GPU, OFF-POD): bootstrap CI on f_CV per behavior over the
  280 personaxquestion pairs (persona-clustered), the v-space f_CV, and the hero
  2x4 grid. Runs after the pod terminates over the HF-uploaded per-cell tensors.

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]`` log
lines, a terminal ``[phase=done]`` (reserved for the single terminal line —
per-cell completions are worded WITHOUT the phase tag), and an end-of-run
sentinel JSON at ``/workspace/logs/issue-697-<kind_slug>-<epoch>.json`` carrying
``_SENTINEL_REQUIRED_KEYS`` (sentinel_schema_version / kind / version). Pod-side
code NEVER shells out to scripts/task.py (CLAUDE.md).

Single-variable-from-parent reuse (plan §4.2): the ONLY new code is the
cross-model patch hook (``analysis.cv_patch``) + this dispatcher; every adapter,
panel, read recipe, layer, and DV is inherited byte-identically from #537/#651.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

logger = logging.getLogger("issue697_dispatch")

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# Per-cell tensor / E-json destination on the HF data repo (analysis-input
# contract — Upload Policy: intermediate analysis tensors the analyze phase
# consumes MUST land on HF before pod terminate, #521).
HF_TENSOR_PREFIX = "issue697_cv_patch/analysis_tensors"
HF_RAW_COMPLETIONS_PREFIX = "issue697_cv_patch/raw_completions"
# The smoke-pass artifact + the canary use_cache decision live under this prefix on
# HF so the off-pod / fresh-VM pre-sweep gate (B1) + the use_cache fetch (B3) can
# resolve them after the pod terminates.
HF_GATE_PREFIX = "issue697_cv_patch/gates"
# The smoke-pass sentinel the §7.1b PRE-SWEEP gate (B1) reads: written by
# phase_smoke after the smoke cell's .pt passes the non-inert detector.
SMOKE_PASS_BASENAME = "smoke_697b_pass.json"
# The salvaged attempt-1 canary use_cache decision on HF (B3 fallback): it says
# use_cache_production_default=false for the 7B model. The 0.5B-smoke-derived local
# copy is NEVER accepted for the 7B production sweep.
SALVAGED_CANARY_DECISION = (
    "issue697_partial/att-20260628-141102/eval_results_issue_697/canary/canary_decision.json"
)

# The 4 behaviors #697 reads (em/sycophancy/marker/fact). refusal (partial null,
# #651) + emnc (positives-only Betley bridge) are EXCLUDED — plan §10.
BEHAVIORS_697: tuple[str, ...] = ("em", "sycophancy", "marker", "fact")

# Read layer (v, headline) + the donor-injection patch layer (plan §4.0 Option B,
# the v4 read-inertness fix): patch UPSTREAM at L=10, read v at L=14 so the patch
# propagates through 4 attention layers to the response-slot read (patch == read
# is the v3 read-inert class). L=10 is inside #651's swept {7,14,21} band.
PRIMARY_LAYER = 14  # = read_layer (Source: #651 PRIMARY_LAYER)
PATCH_LAYER = 10  # = L_patch (< read_layer; Source: v4 read-inertness fix)
SUPPLEMENT_LAYERS = (7, 21)

# Per-behavior PRIMARY v pooling (item-5 fix — mirrors #651's headline):
# mean-resp for em/sycophancy, end-of-response slot for marker/fact.
PRIMARY_POOLING: dict[str, str] = {
    "em": "mean_resp",
    "sycophancy": "mean_resp",
    "marker": "slot",
    "fact": "slot",
}


def _resolve_repo_root() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        env={**os.environ},  # epm-lint: subprocess-env-inherit -- git toplevel probe, no creds
    ).decode()
    return Path(out.strip())


def _log_dir() -> Path:
    override = os.environ.get("EPM_LOG_DIR")
    if override:
        d = Path(override)
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = Path("/workspace/logs")
    if not d.exists():  # local VM (no /workspace) -> repo logs/
        d = _resolve_repo_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def phase_log(name: str) -> None:
    """Emit the ``[phase=<name>]`` line poll_pipeline.py parses (PHASE_RE).

    The poller's PHASE_RE matches ``[a-z0-9_]+`` so numbered phase tokens parse
    fully; this dispatcher uses only lowercase-underscore phase names anyway.
    """
    print(f"[phase={name}]", flush=True)


def write_sentinel(kind: str, note: str, *, version: int = 1, extra: dict | None = None) -> Path:
    """End-of-run sentinel with poll_pipeline's _SENTINEL_REQUIRED_KEYS.

    Required keys: sentinel_schema_version (int 1), kind (full marker string),
    version (int). The marker body goes under ``note``.
    """
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 697,
        "by": "issue697_dispatch",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-697-{slug}-{time.time_ns()}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", out)
    return out


def _run_with_log(
    cmd: Sequence[str],
    *,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> int:
    """Run a child process, tee stdout/stderr to a log file. Returns rc.

    Every subprocess gets an EXPLICIT ``env={**os.environ}`` (+ extra_env): the
    credential env (HF_TOKEN / WANDB_API_KEY / ANTHROPIC_API_KEY) must be present
    even though ``uv run python`` does not auto-load .env — load_dotenv() in
    main() puts it in os.environ first (#397 round-10').
    """
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "$ %s  >>> %s%s",
        " ".join(shlex.quote(c) for c in cmd),
        log_path,
        f" (env+={list(extra_env.keys())})" if extra_env else "",
    )
    with log_path.open("ab") as f:
        proc = subprocess.run(
            list(cmd),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    rc = proc.returncode
    if rc != 0:
        logger.error("command exited with rc=%d (log: %s)", rc, log_path)
    return rc


def _run_parallel_with_log(
    cmds: Iterable[tuple[Sequence[str], Path, dict[str, str] | None]],
    *,
    cwd: Path | None = None,
) -> list[int]:
    """Run several subprocesses concurrently. Returns parallel list of rc codes."""
    procs: list[subprocess.Popen] = []
    files = []
    for cmd, log_path, extra_env in cmds:
        env = {**os.environ}
        if extra_env:
            env.update(extra_env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("ab")
        files.append(f)
        logger.info(
            "$ (parallel) %s  >>> %s%s",
            " ".join(shlex.quote(c) for c in cmd),
            log_path,
            f" (env+={list(extra_env.keys())})" if extra_env else "",
        )
        p = subprocess.Popen(
            list(cmd), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None
        )
        procs.append(p)
    rcs = [p.wait() for p in procs]
    for f in files:
        f.close()
    return rcs


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"


# ---------------------------------------------------------------------------
# Cell grid (4 behaviors x 16 contexts x 2 seeds = 128; plan §10)
# ---------------------------------------------------------------------------


def cells_697(n_gpus: int = 4, floor_only: bool = False):
    """The 128 cells #697 reads: em/sycophancy/marker/fact x 16 ctx x 2 seeds.

    Filters the inherited #651 ``readable_cells`` to ``BEHAVIORS_697`` (drops
    refusal + emnc — plan §10), then re-densifies ``gpu_id`` round-robin over the
    requested subset so the wave dispatcher shards evenly. ``floor_only`` keeps
    only the seed-42 cells (the auto-descope floor, plan §9 stratification).
    """
    from explore_persona_space.experiments.issue_651 import Cell, readable_cells

    full = readable_cells(n_gpus=n_gpus, include_seed1042=not floor_only)
    sel = [c for c in full if c.behavior in BEHAVIORS_697]
    return [
        Cell(behavior=c.behavior, cid=c.cid, seed=c.seed, gpu_id=i % max(n_gpus, 1))
        for i, c in enumerate(sel)
    ]


def _select_cells(args):
    """Resolve the per-phase cell subset from --cells (or the full 128-cell grid).

    PASS_UNIFIED parity: EVERY phase (canary / sweep) reads from this SAME
    ``cells_697`` grid filtered by the SAME ``--cells`` subset, so a smoke is the
    sweep with one cell and no phase re-enumerates a different grid.

    ``--seed-42-only`` (plan §Salvage / A7): the true seed-42 production grid is 64
    cells (4 behaviors x 16 ctx x seed 42). ``--floor-only`` alone yields 96
    (marker + fact keep BOTH seeds on HF), so seed-42-only filters the floor to
    ``seed == 42`` across ALL behaviors. The 4 salvaged marker cells are RE-RUN
    (their v3 v is read-inert) — they are IN this grid, not skipped.
    """
    from explore_persona_space.experiments.issue_651 import Cell, parse_cell_spec

    full = cells_697(n_gpus=args.n_gpus, floor_only=args.floor_only or args.seed_42_only)
    if args.seed_42_only:
        sel42 = [c for c in full if c.seed == 42]
        full = [
            Cell(behavior=c.behavior, cid=c.cid, seed=c.seed, gpu_id=i % max(args.n_gpus, 1))
            for i, c in enumerate(sel42)
        ]
    if args.cells:
        requested = [parse_cell_spec(s) for s in args.cells]
        avail = {(c.behavior, c.cid, c.seed) for c in full}
        unknown = [r.cell_id for r in requested if (r.behavior, r.cid, r.seed) not in avail]
        if unknown:
            raise ValueError(
                f"--cells {unknown!r} not in the #697 grid "
                f"(behaviors={BEHAVIORS_697}, seeds available: "
                f"{sorted({(c.behavior, c.seed) for c in full})})"
            )
        return [
            Cell(behavior=r.behavior, cid=r.cid, seed=r.seed, gpu_id=i % max(args.n_gpus, 1))
            for i, r in enumerate(requested)
        ]
    return full


# ---------------------------------------------------------------------------
# Phase: VENDOR (CPU import-smoke — no models)
# ---------------------------------------------------------------------------


def phase_vendor(repo_root: Path) -> None:
    """Confirm the vendored read path + cv_patch import cleanly + panel materializes."""
    phase_log("vendor")
    # Import the full read path (vendor verification per plan A0).
    from explore_persona_space.analysis import cv_patch
    from explore_persona_space.analysis.activation_shift import (  # noqa: F401
        _build_chatml_prompt,
        _read_residuals,
        extract_per_context_shifts,
    )
    from explore_persona_space.experiments.issue_651 import (
        build_panel_personas,
        build_panel_questions,
    )

    personas = build_panel_personas()
    questions = build_panel_questions()
    cells = cells_697()
    assert len(personas) == 14, (len(personas), "expected the fixed 14-persona panel")
    assert len(questions) == 20, (len(questions), "expected the 20-question panel")
    assert len(cells) == 128, (len(cells), "expected 4 behaviors x 16 ctx x 2 seeds = 128")
    # cv_patch public surface present.
    for name in (
        "content_patch_pos",
        "audit_patch_slot",
        "make_cv_patch_hook",
        "patched_read",
        "patched_generate",
        "first_token_logits",
        "compute_f_cv",
        "compute_f_cv_down",
        "NO_EFFECT",
        "SlotAuditError",
    ):
        assert hasattr(cv_patch, name), f"cv_patch missing {name}"
    logger.info(
        "vendor smoke OK: %d personas, %d questions, %d cells; cv_patch surface present",
        len(personas),
        len(questions),
        len(cells),
    )
    logger.info("[phase=vendor_done]")


# ---------------------------------------------------------------------------
# Phase: INERT_READ_ASSERT (CPU, 0 GPU — the §7.1a non-skippable gate)
# ---------------------------------------------------------------------------

# The v3-salvaged (read-inert) marker cell on HF — the §7.1a negative control.
SALVAGED_INERT_PT = (
    "issue697_partial/att-20260628-141102/eval_results_issue_697/patch/marker_sp_swe_seed42.pt"
)


def phase_inert_read_assert(repo_root: Path, *, read_layer: int) -> None:
    """§7.1a (CPU, 0 GPU): the detector MUST FIRE on the v3-salvaged inert .pt.

    Re-run the inert-read detector on the salvaged ``marker_sp_swe_seed42.pt``
    (downloaded from HF — NOT new GPU spend). It MUST raise ``ReadInertError``
    (the salvaged cell IS inert by construction). If it does NOT raise, the
    DETECTOR itself is broken — abort the whole run (no sweep): a broken detector
    cannot guarantee the §7.1b positive gate on the new smoke cell is real.

    This is the guard that catches the v3 read-inertness regression instantly.
    """
    phase_log("inert_read_assert")
    import sys

    sys.path.insert(0, str(repo_root / "scripts"))
    import torch
    from huggingface_hub import hf_hub_download
    from issue697_analysis import ReadInertError, assert_not_read_inert

    p = hf_hub_download(HF_DATA_REPO, SALVAGED_INERT_PT, repo_type="dataset")
    cell = torch.load(p, weights_only=False)
    try:
        means = assert_not_read_inert(cell, read_layer)
    except ReadInertError as e:
        logger.info("[phase=inert_read_assert] PASS 7.1a: detector fired on salvaged inert .pt")
        logger.info("  %s", str(e).replace("\n", " "))
        logger.info("[phase=inert_read_assert_done]")
        return
    raise RuntimeError(
        "INERT-READ DETECTOR BROKEN (§7.1a): the salvaged read-inert "
        f"marker_sp_swe_seed42.pt did NOT trip the detector (means={means}); a "
        "detector that cannot detect known inertness cannot guard the sweep. HALT."
    )


def assert_smoke_cell_not_inert(repo_root: Path, cell_id: str, *, read_layer: int) -> dict:
    """§7.1b (CPU on the new smoke .pt): the L=10/L=14 read MUST NOT be inert.

    After the smoke cell runs the L=10-patch / L=14-read path (one real GPU cell),
    re-run the detector on its NEW ``.pt`` and require it does NOT fire. If it DOES
    fire, the layer split is wrong (the hook is still effectively at the read
    layer) and the production sweep must NOT dispatch. Reads the LOCAL smoke .pt.
    Returns the means dict on PASS (the caller writes them into the smoke-pass
    artifact).
    """
    import sys

    sys.path.insert(0, str(repo_root / "scripts"))
    import torch
    from issue697_analysis import ReadInertError, assert_not_read_inert

    pt = repo_root / "eval_results" / "issue_697" / "patch" / f"{cell_id}.pt"
    if not pt.exists():
        raise RuntimeError(f"§7.1b: smoke cell .pt not found at {pt} -- smoke cell did not run")
    cell = torch.load(pt, weights_only=False)
    try:
        means = assert_not_read_inert(cell, read_layer)
    except ReadInertError as e:
        raise RuntimeError(
            f"§7.1b FAIL: the NEW smoke cell {cell_id} .pt IS read-inert at the "
            f"L=10/L=14 pathway -- the layer split is wrong, the sweep must NOT "
            f"dispatch. {e}"
        ) from e
    logger.info(
        "[phase=inert_read_assert] PASS 7.1b: smoke cell %s NON-inert "
        "(f_CV[p_up]=%.4f, random=%.4f, f_CV_down=%.4f, n=%d)",
        cell_id,
        means["f_cv"],
        means["f_cv_random"],
        means["f_cv_down"],
        means["n"],
    )
    return means


def _git_sha(repo_root: Path) -> str:
    out = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        env={**os.environ},  # epm-lint: subprocess-env-inherit -- git sha probe, no creds
    ).decode()
    return out.strip()


def _smoke_pass_path(repo_root: Path) -> Path:
    return repo_root / "eval_results" / "issue_697" / SMOKE_PASS_BASENAME


def write_smoke_pass_artifact(
    repo_root: Path, cell_id: str, means: dict, *, read_layer: int, upload: bool
) -> Path:
    """Persist (+ optionally upload to HF) the §7.1b smoke-pass artifact (B1).

    The PRE-SWEEP gate (``_assert_smoke_pass_for_sweep``) refuses to enter the wave
    loop on a production (len>1) sweep until THIS artifact exists with a matching
    git SHA and ``non_inert: true``. It records the four §7.1b f_CV means, the cell
    id, the read layer, the git SHA (so a code change after the smoke invalidates
    the pass), and a timestamp.
    """
    payload = {
        "issue": 697,
        "cell_id": cell_id,
        "git_sha": _git_sha(repo_root),
        "read_layer": read_layer,
        "f_cv_pup_mean": float(means["f_cv"]),
        "f_cv_random_mean": float(means["f_cv_random"]),
        "f_cv_full_span_mean": float(means["f_cv_full_span"]),
        "f_cv_down_pdn_mean": float(means["f_cv_down"]),
        "n": int(means["n"]),
        "non_inert": True,
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    out = _smoke_pass_path(repo_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    logger.info("wrote smoke-pass artifact %s (git_sha=%s)", out, payload["git_sha"][:12])
    if upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=f"{HF_GATE_PREFIX}/{SMOKE_PASS_BASENAME}",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue697: §7.1b smoke-pass ({cell_id} @ {payload['git_sha'][:12]})",
        )
        logger.info("uploaded smoke-pass artifact to HF %s/%s", HF_GATE_PREFIX, SMOKE_PASS_BASENAME)
    return out


def _load_smoke_pass(repo_root: Path) -> dict | None:
    """Find a smoke-pass artifact locally, else on HF (fresh-VM / off-pod path)."""
    local = _smoke_pass_path(repo_root)
    if local.exists():
        return json.loads(local.read_text())
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    try:
        p = hf_hub_download(
            HF_DATA_REPO, f"{HF_GATE_PREFIX}/{SMOKE_PASS_BASENAME}", repo_type="dataset"
        )
    except (EntryNotFoundError, HfHubHTTPError):
        return None
    return json.loads(Path(p).read_text())


def _assert_smoke_pass_for_sweep(repo_root: Path, *, read_layer: int) -> None:
    """§7.1b PRE-SWEEP gate (B1): refuse a production sweep until the smoke cell
    passed the non-inert detector at the CURRENT git SHA.

    Plan §7.1 (line 265): "the dispatcher refuses to enter ``phase_sweep`` until the
    smoke cell's ``.pt`` passes 7.1b". This is the LOAD-BEARING gate — it runs
    BEFORE the wave loop on ANY multi-cell (production) sweep, so a 64-cell A100-80
    run can never start without a verified non-inert L=10/L=14 read on a real cell.
    A missing artifact, ``non_inert != True``, a stale git SHA, or a read-layer
    mismatch all RAISE (the smoke must be re-run at this commit).
    """
    sp = _load_smoke_pass(repo_root)
    if sp is None:
        raise RuntimeError(
            "§7.1b PRE-SWEEP GATE FAIL: no smoke-pass artifact found locally "
            f"({_smoke_pass_path(repo_root)}) or on HF ({HF_GATE_PREFIX}/{SMOKE_PASS_BASENAME}). "
            "The production sweep refuses to dispatch until `--phase smoke` runs one real-GPU "
            "cell and its .pt passes the §7.1b non-inert detector (plan §7.1)."
        )
    if not sp.get("non_inert"):
        raise RuntimeError(f"§7.1b PRE-SWEEP GATE FAIL: smoke-pass artifact says non_inert={sp!r}")
    cur = _git_sha(repo_root)
    if sp.get("git_sha") != cur:
        raise RuntimeError(
            f"§7.1b PRE-SWEEP GATE FAIL: smoke-pass git_sha={sp.get('git_sha')} != current {cur} "
            "-- code changed since the smoke; re-run `--phase smoke` at this commit."
        )
    if sp.get("read_layer") != read_layer:
        raise RuntimeError(
            f"§7.1b PRE-SWEEP GATE FAIL: smoke-pass read_layer={sp.get('read_layer')} != "
            f"sweep read_layer={read_layer} -- re-run `--phase smoke` at the sweep's read layer."
        )
    logger.info(
        "§7.1b PRE-SWEEP GATE PASS: smoke cell %s non-inert at git_sha=%s "
        "(f_CV[p_up]=%.4f, random=%.4f, f_CV_down=%.4f)",
        sp.get("cell_id"),
        cur[:12],
        sp.get("f_cv_pup_mean", float("nan")),
        sp.get("f_cv_random_mean", float("nan")),
        sp.get("f_cv_down_pdn_mean", float("nan")),
    )


# ---------------------------------------------------------------------------
# Phase: RBASE_PREP (vLLM-batched R_base cache; plan §4.4/§4.5)
# ---------------------------------------------------------------------------


def _rbase_cache_dir(repo_root: Path) -> Path:
    return repo_root / "eval_results" / "issue_697" / "r_base_cache"


def phase_rbase_prep(
    repo_root: Path, *, cpu_only: bool, smoke_model: str | None, upload: bool, max_new_tokens: int
) -> None:
    """Pre-sweep vLLM-batched R_base cache (plan §4.4/§4.5) via issue697_rbase.py."""
    phase_log("rbase_prep")
    out_dir = _rbase_cache_dir(repo_root)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue697_rbase.py",
        "--out-dir",
        str(out_dir),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    base_model = smoke_model or QWEN_ID
    cmd += ["--base-model-id", base_model]
    if cpu_only:
        cmd += ["--cpu-only", "--skip-parity"]
    if upload:
        cmd.append("--upload")
    # vLLM uses 1 GPU; CVD=0 on the real path, "" on the CPU smoke.
    env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else "0"}
    rc = _run_with_log(cmd, log_path=_log_dir() / "rbase_prep.log", extra_env=env, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(f"rbase_prep FAILED (rc={rc}); see {_log_dir()}/rbase_prep.log")
    logger.info("[phase=rbase_prep_done]")


# ---------------------------------------------------------------------------
# Phase: CANARY (Gate C1 patch-correctness + Gate C2 rsLoRA parity)
# ---------------------------------------------------------------------------


def phase_canary(repo_root: Path, *, cpu_only: bool, smoke_model: str | None) -> None:
    """Run Gate C1 (cv_patch correctness) + Gate C2 (inherited #651 Gate 7a); HALT on FAIL."""
    phase_log("canary")
    cmd = ["uv", "run", "python", "scripts/issue697_canary.py"]
    if cpu_only:
        cmd.append("--cpu-only")
    if smoke_model:
        cmd += ["--smoke-model", smoke_model]
    env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else "0"}
    log_path = _log_dir() / "canary.log"
    rc = _run_with_log(cmd, log_path=log_path, extra_env=env, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(
            f"CANARY FAILED (rc={rc}) -- Gate C1/C2 did not PASS; HALT before the sweep. "
            f"See {log_path}"
        )
    logger.info("[phase=canary_done]")


# ---------------------------------------------------------------------------
# Phase: SWEEP (per-cell patch read on the panel; wave-sharded, per-cell upload)
# ---------------------------------------------------------------------------


def _cell_cmd(
    repo_root: Path,
    cell,
    *,
    cpu_only: bool,
    panel_personas_json: Path,
    panel_questions_json: Path,
    out_dir: Path,
    layers: Sequence[int],
    primary_layer: int,
    max_new_tokens: int,
    skip_e: bool,
    smoke_model: str | None,
    upload: bool,
    use_cache: bool,
    patch_layer: int,
    rbase_cache_dir: Path | None,
) -> tuple[list[str], Path, dict[str, str]]:
    """Build (cmd, log_path, env) for one cell's patch read via issue697_cell.py."""
    base_model = smoke_model or QWEN_ID
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue697_cell.py",
        "--behavior",
        cell.behavior,
        "--cid",
        cell.cid,
        "--seed",
        str(cell.seed),
        "--adapter-subfolder",
        cell.adapter_subfolder,
        "--personas-json",
        str(panel_personas_json),
        "--questions-json",
        str(panel_questions_json),
        "--out-dir",
        str(out_dir),
        "--layers",
        *[str(L) for L in layers],
        "--primary-layer",
        str(primary_layer),
        "--patch-layer",
        str(patch_layer),
        "--max-new-tokens",
        str(max_new_tokens),
        "--base-model-id",
        base_model,
    ]
    if rbase_cache_dir is not None:
        cmd += ["--rbase-cache-dir", str(rbase_cache_dir)]
    # Thread the canary's use_cache decision (concern #4): BooleanOptionalAction.
    cmd.append("--use-cache" if use_cache else "--no-use-cache")
    if cpu_only:
        cmd.append("--cpu-only")
    if skip_e:
        cmd.append("--skip-e")
    if upload:
        cmd.append("--upload")
    env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else str(cell.gpu_id)}
    log_path = _log_dir() / f"sweep_{cell.cell_id}.log"
    return cmd, log_path, env


def _validate_canary_decision_provenance(decision: dict, production_base_model: str, src: str):
    """Reject a canary use_cache decision whose base model is NOT the 7B production
    model (concern #3): a 0.5B-smoke-derived ``use_cache=True`` MUST NOT be accepted
    for the 7B sweep (KV caching DROPS the patch on 7B → corrupted p_up/p_down E).

    ``base_model_id`` is the explicit provenance field (added v6); ``model`` is the
    back-compat fallback the salvaged decision carries.
    """
    base = decision.get("base_model_id") or decision.get("model")
    if base != production_base_model:
        raise RuntimeError(
            f"canary use_cache decision at {src} was derived from base model {base!r}, "
            f"not the production model {production_base_model!r} -- a smoke-derived (0.5B) "
            f"decision is REJECTED for the 7B sweep (concern #3). Re-run --phase canary on "
            f"the 7B model, or place a 7B canary_decision.json on HF ({HF_GATE_PREFIX}/)."
        )


def _read_use_cache_decision(repo_root: Path, *, production_base_model: str = QWEN_ID) -> bool:
    """Resolve the canary's use_cache decision (concern #3); DEFAULT False if absent.

    Resolution order:
      1. The local 7B ``canary_decision.json`` (written by ``--phase canary`` in
         the same dispatch) — accepted ONLY when its ``base_model_id`` is the 7B
         production model.
      2. The HF-published 7B decision (``HF_GATE_PREFIX/canary_decision.json``),
         else the salvaged attempt-1 decision (``SALVAGED_CANARY_DECISION``, which
         says use_cache_production_default=false for the 7B model). Same
         provenance check.
      3. NEITHER present → ``use_cache=False`` (the SAFE default — the attempt-1
         canary measured caching DROPS the patch; running uncached can never
         corrupt the E-gen, running cached can — so the absent-decision default is
         False, NOT True).

    A local/HF decision whose provenance is a non-7B (smoke) model is REJECTED, not
    silently accepted. Path constants are formed inline (NOT
    ``from scripts.issue697_canary import ...``) because this module runs as a
    SCRIPT (``sys.path[0]`` is ``scripts/``, the ``scripts`` package not importable).
    """
    local = repo_root / "eval_results" / "issue_697" / "canary" / "canary_decision.json"
    if local.exists():
        decision = json.loads(local.read_text())
        _validate_canary_decision_provenance(decision, production_base_model, str(local))
        use_cache = bool(decision.get("use_cache_production_default", False))
        logger.info("canary use_cache decision (local 7B): use_cache=%s (%s)", use_cache, local)
        return use_cache

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    for hf_path in (f"{HF_GATE_PREFIX}/canary_decision.json", SALVAGED_CANARY_DECISION):
        try:
            p = hf_hub_download(HF_DATA_REPO, hf_path, repo_type="dataset")
        except (EntryNotFoundError, HfHubHTTPError) as e:
            logger.info("canary decision not on HF at %s (%s); trying next", hf_path, e)
            continue
        decision = json.loads(Path(p).read_text())
        _validate_canary_decision_provenance(decision, production_base_model, f"HF:{hf_path}")
        use_cache = bool(decision.get("use_cache_production_default", False))
        logger.info("canary use_cache decision (HF %s): use_cache=%s", hf_path, use_cache)
        return use_cache

    logger.warning(
        "no 7B canary use_cache decision found (local or HF) -> default use_cache=False "
        "(the SAFE default: caching can drop the patch, uncached never corrupts E-gen)"
    )
    return False


def _assert_sweep_device_count(cpu_only: bool, n_gpus: int) -> None:
    """Belt-and-suspenders device-count preflight (standing rec / item 5c).

    The 4-GPU sweep needs the ``ft-7b`` 4x A100-80 intent; an orchestrator intent
    mis-inference (e.g. a 1-GPU ``lora-7b`` pod) would silently co-locate the
    waves. Assert the visible device count matches ``n_gpus`` before the sweep so
    a mis-launch FAILs loud at startup, not mid-sweep. Skipped on the CPU smoke.
    """
    if cpu_only:
        return
    import torch

    visible = torch.cuda.device_count()
    assert visible == n_gpus, (
        f"sweep phase requires {n_gpus} GPUs (got {visible}); the orchestrator must launch "
        f"with the matching intent (--n-gpus 4 -> ft-7b 4x A100-80). Set --n-gpus to the actual "
        f"device count if this is a deliberate smaller-pod run."
    )


def _write_sweep_coverage(repo_root: Path, cells: Sequence, failed_cells: Sequence[str]) -> Path:
    """Write the sweep coverage gap (plan §4.3) so analyze reports the missing cells.

    ``eval_results/issue_697/sweep_coverage.json`` carries the full attempted cell
    list, the failed cell ids, and the failed cells parsed to (behavior, cid, seed)
    tuples so the analyze phase can name the missing cells per ``verify_task_body.py``
    check 11b (revise the per-behavior denominator; never a misleading zero bar).
    """
    failed = list(dict.fromkeys(failed_cells))  # dedup, preserve order
    failed_tuples = []
    for cid_str in failed:
        beh, _, rest = cid_str.partition("_")
        cid, _, seed_tok = rest.rpartition("_seed")
        failed_tuples.append({"cell_id": cid_str, "behavior": beh, "cid": cid, "seed": seed_tok})
    cov_path = repo_root / "eval_results" / "issue_697" / "sweep_coverage.json"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    cov_path.write_text(
        json.dumps(
            {
                "issue": 697,
                "n_attempted": len(cells),
                "attempted_cells": [c.cell_id for c in cells],
                "n_failed": len(failed),
                "failed_cells": failed,
                "failed_tuples": failed_tuples,
                "budget": -(-len(cells) // 16),
                "ts": datetime.datetime.now(datetime.UTC).isoformat(),
            },
            indent=2,
        )
    )
    logger.info("wrote sweep coverage: %s (%d failed)", cov_path, len(failed))
    return cov_path


def phase_sweep(
    repo_root: Path,
    cells: Sequence,
    *,
    n_gpus: int,
    cpu_only: bool,
    panel_personas_json: Path,
    panel_questions_json: Path,
    layers: Sequence[int],
    primary_layer: int,
    patch_layer: int,
    max_new_tokens: int,
    skip_e: bool,
    smoke_model: str | None,
    dry_run: bool,
    upload: bool,
    rbase_cache_dir: Path | None = None,
) -> None:
    """Per-cell patch read over the panel (wave-parallel, CVD-pinned per cell)."""
    phase_log("sweep")
    # §7.1b PRE-SWEEP GATE (B1, plan §7.1 line 265): a PRODUCTION (multi-cell)
    # real-GPU sweep REFUSES to enter the wave loop until a smoke cell passed the
    # non-inert detector at THIS git SHA. Skipped for: dry-run (plumbing only), the
    # CPU smoke (no real adapter -> v⁺≡v0 -> uninformative), and the single-cell
    # smoke itself (it IS the cell that produces the smoke-pass artifact).
    is_production = not dry_run and smoke_model is None and not cpu_only and len(cells) > 1
    if is_production:
        try:
            _assert_smoke_pass_for_sweep(repo_root, read_layer=primary_layer)
        except RuntimeError as e:
            write_sentinel("epm:failure", f"§7.1b pre-sweep gate FAILED: {e}")
            raise
    # Device-count preflight (item 5c) — skip on dry-run (no GPU needed) + CPU smoke.
    if not dry_run and smoke_model is None:
        _assert_sweep_device_count(cpu_only, n_gpus)
    # use_cache threaded from the canary's Gate C1.2 decision (concern #3); a smoke-
    # derived (non-7B) decision is REJECTED for the 7B sweep, default False if absent.
    use_cache = _read_use_cache_decision(repo_root, production_base_model=smoke_model or QWEN_ID)
    out_dir = repo_root / "eval_results" / "issue_697" / "patch"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Continue-on-cell-fail (plan §4.3): bound the deterministic-bug blast radius at
    # ceil(N/16) (= 4 for 64 cells, ~6%) while surviving transient infra. Below
    # budget the sweep advances + records the gap; above budget it aborts.
    n_total_cells = len(cells)
    budget = -(-n_total_cells // 16)  # ceil(N/16)
    failed_cells: list[str] = []
    for wave_start in range(0, len(cells), max(n_gpus, 1)):
        wave = cells[wave_start : wave_start + max(n_gpus, 1)]
        cmds: list[tuple[Sequence[str], Path, dict[str, str] | None]] = []
        for cell in wave:
            cmd, log_path, env = _cell_cmd(
                repo_root,
                cell,
                cpu_only=cpu_only,
                panel_personas_json=panel_personas_json,
                panel_questions_json=panel_questions_json,
                out_dir=out_dir,
                layers=layers,
                primary_layer=primary_layer,
                max_new_tokens=max_new_tokens,
                skip_e=skip_e,
                smoke_model=smoke_model,
                upload=upload,
                use_cache=use_cache,
                patch_layer=patch_layer,
                rbase_cache_dir=rbase_cache_dir,
            )
            cmds.append((cmd, log_path, env))
        if dry_run:
            for (cmd, _lp, env), cell in zip(cmds, wave, strict=True):
                logger.info(
                    "[dry-run] sweep %s CVD=%r :: %s",
                    cell.cell_id,
                    env.get("CUDA_VISIBLE_DEVICES"),
                    " ".join(shlex.quote(c) for c in cmd),
                )
            continue
        rcs = _run_parallel_with_log(cmds, cwd=repo_root)
        wave_idx = wave_start // max(n_gpus, 1)
        for rc, c in zip(rcs, wave, strict=True):
            if rc != 0:
                failed_cells.append(c.cell_id)
                # Per-cell sentinel via the existing drain path (NEVER a pod-side
                # task.py shellout). poll_pipeline.py posts the carried marker.
                write_sentinel(
                    "epm:cell-failed",
                    f"cell {c.cell_id} failed rc={rc} wave={wave_idx}",
                    extra={
                        "cell_id": c.cell_id,
                        "rc": rc,
                        "wave": wave_idx,
                        "log_path": str(_log_dir() / f"sweep_{c.cell_id}.log"),
                    },
                )
                logger.error(
                    "sweep cell %s FAILED rc=%d (wave %d); failed %d/%d (budget %d)",
                    c.cell_id,
                    rc,
                    wave_idx,
                    len(failed_cells),
                    n_total_cells,
                    budget,
                )
            else:
                # NOT [phase=done] (mid-run noise — the terminal line is reserved).
                logger.info("sweep cell %s complete", c.cell_id)
        # Budget guard: a >budget failure count is a systematic bug -> abort so the
        # EXIT trap persists crash diagnostics (plan §4.3).
        if len(failed_cells) > budget:
            _write_sweep_coverage(repo_root, cells, failed_cells)
            raise RuntimeError(
                f"sweep failure budget exceeded: {len(failed_cells)} > {budget} "
                f"(ceil({n_total_cells}/16)); failed={failed_cells}; see logs in {_log_dir()}"
            )
    if dry_run:
        logger.info("[phase=sweep_done] (dry-run: no tensors written, upload skipped)")
        return
    # Below-budget failures (<= budget): loud WARNING + terminal coverage artifacts
    # so analyze + the orchestrator see the gap (plan §4.3). The sweep does NOT raise.
    _write_sweep_coverage(repo_root, cells, failed_cells)
    if failed_cells:
        logger.warning(
            "sweep completed with %d/%d cells FAILED (<= budget %d): %s -- coverage gap "
            "recorded in eval_results/issue_697/sweep_coverage.json; analyze reports it.",
            len(failed_cells),
            n_total_cells,
            budget,
            failed_cells,
        )
        write_sentinel(
            "epm:cell-failed-summary",
            f"{len(failed_cells)}/{n_total_cells} cells failed (<= budget {budget})",
            extra={"failed_cells": failed_cells, "n_total_cells": n_total_cells, "budget": budget},
        )
    # §7.1b positive gate: a single REAL-GPU smoke cell's NEW .pt MUST NOT be
    # read-inert at the L=10/L=14 pathway (the layer split actually works). Runs
    # only on the real-adapter single-cell smoke (the CPU no-adapter smoke has
    # v⁺≡v0 -> no-effect, not inert, so the detector is uninformative there). On
    # PASS it WRITES + uploads the smoke-pass artifact (B1) the production sweep's
    # PRE-SWEEP gate then requires. (This is the belt-and-suspenders end-of-sweep
    # recheck; the LOAD-BEARING gate is `_assert_smoke_pass_for_sweep` PRE-loop. A
    # FAIL aborts before any production sweep can dispatch — plan §7.1b.)
    is_real_gpu = smoke_model is None and not cpu_only
    if is_real_gpu and len(cells) == 1 and cells[0].cell_id not in failed_cells:
        means = assert_smoke_cell_not_inert(repo_root, cells[0].cell_id, read_layer=primary_layer)
        write_smoke_pass_artifact(
            repo_root, cells[0].cell_id, means, read_layer=primary_layer, upload=upload
        )
    logger.info("[phase=sweep_done]")


# ---------------------------------------------------------------------------
# Phase: SMOKE (the §7.1b gate-producing phase — B1)
# ---------------------------------------------------------------------------

SMOKE_CELL_SPEC = "marker_sp_swe_seed42"  # one real-GPU cell; produces the smoke-pass artifact


def phase_smoke(
    repo_root: Path,
    *,
    n_gpus: int,
    cpu_only: bool,
    smoke_model: str | None,
    layers: Sequence[int],
    primary_layer: int,
    patch_layer: int,
    max_new_tokens: int,
    upload: bool,
    rbase_cache_dir: Path | None = None,
    smoke_cell_spec: str = SMOKE_CELL_SPEC,
) -> None:
    """§7.1b gate phase (B1): run ONE real-GPU cell through the SAME per-cell sweep
    path, then run the non-inert detector + write/upload the smoke-pass artifact.

    PASS_CANARY architecture: this phase is `phase_sweep` with a single canary cell
    (``marker_sp_swe_seed42``), running the identical wave dispatcher / `_cell_cmd` /
    env-injection / sentinel path — so the L=10/L=14 read pathway is exercised
    end-to-end on a real cell BEFORE the production sweep. It is in the `all` ladder
    BEFORE `sweep`; the production sweep's PRE-loop gate then requires the artifact
    this phase writes.
    """
    from explore_persona_space.experiments.issue_651 import Cell, parse_cell_spec

    spec = parse_cell_spec(smoke_cell_spec)
    smoke_cells = [Cell(behavior=spec.behavior, cid=spec.cid, seed=spec.seed, gpu_id=0)]
    panel_personas_json, panel_questions_json = _materialize_panel(repo_root)
    # The skip_e=True is inherited by a CPU/tiny-model smoke; on a real GPU the
    # full per-cell path (incl. the four-float marker capture) runs. phase_sweep's
    # single-cell real-GPU branch writes + uploads the smoke-pass artifact on PASS.
    phase_sweep(
        repo_root,
        smoke_cells,
        n_gpus=n_gpus,
        cpu_only=cpu_only,
        panel_personas_json=panel_personas_json,
        panel_questions_json=panel_questions_json,
        layers=layers,
        primary_layer=primary_layer,
        patch_layer=patch_layer,
        max_new_tokens=max_new_tokens,
        skip_e=(cpu_only or smoke_model is not None),
        smoke_model=smoke_model,
        dry_run=False,
        upload=upload,
        rbase_cache_dir=rbase_cache_dir,
    )


# ---------------------------------------------------------------------------
# Phase: ANALYZE (CPU, off-pod — delegate to issue697_analysis.py)
# ---------------------------------------------------------------------------


def _hydrate_analyze_artifacts(repo_root: Path) -> int:
    """Hydrate the per-cell analyze inputs from the HF data repo (B2).

    Plan §9 routes analyze OFF-POD over HF-downloaded ``.pt``s AFTER the pod
    terminates, so a fresh-VM checkout has no local ``eval_results/issue_697/patch``
    — this pulls every sweep-produced artifact from HF so the downstream
    ``glob('*.pt')`` finds them. Pulls (a) per-cell ``*.pt`` + ``*_E_metadata.json``
    from ``HF_TENSOR_PREFIX``, (b) per-cell ``raw_completions/*.json`` (the judge
    inputs) from ``HF_RAW_COMPLETIONS_PREFIX``, (c) ``sweep_coverage.json`` from
    ``HF_GATE_PREFIX``. Uses the canonical Hub API (``list_repo_files`` +
    ``hf_hub_download``, NEVER the ``hf`` CLI — upload-policy.md). Idempotent: a
    local ``.pt`` whose ``_E_metadata.json`` git_commit matches the HF copy's
    manifest is SKIPPED (same resume pattern the inert_read_assert phase uses).
    Returns the count of files hydrated. A clear RuntimeError when NOTHING is on HF.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    patch_dir = repo_root / "eval_results" / "issue_697" / "patch"
    raw_dir = patch_dir / "raw_completions"
    patch_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_files = list_repo_files(HF_DATA_REPO, repo_type="dataset")
    tensor_files = [
        f
        for f in all_files
        if f.startswith(f"{HF_TENSOR_PREFIX}/")
        and (f.endswith(".pt") or f.endswith("_E_metadata.json"))
    ]
    raw_files = [
        f
        for f in all_files
        if f.startswith(f"{HF_RAW_COMPLETIONS_PREFIX}/") and f.endswith(".json")
    ]
    cov_files = [f for f in all_files if f == f"{HF_GATE_PREFIX}/sweep_coverage.json"]
    pts = [f for f in tensor_files if f.endswith(".pt")]
    if not pts:
        raise RuntimeError(
            f"no analysis tensors found on HF under {HF_TENSOR_PREFIX}/ (repo {HF_DATA_REPO}) -- "
            "the sweep has not uploaded any per-cell .pt; run the sweep (+per-cell upload) first."
        )

    hydrated = 0
    for hf_path in tensor_files + raw_files + cov_files:
        name = Path(hf_path).name
        dest = (
            raw_dir / name
            if hf_path.startswith(f"{HF_RAW_COMPLETIONS_PREFIX}/")
            else (
                repo_root / "eval_results" / "issue_697" / name
                if name == "sweep_coverage.json"
                else patch_dir / name
            )
        )
        if dest.exists():
            continue  # resume: already hydrated locally (idempotent re-run)
        p = hf_hub_download(HF_DATA_REPO, hf_path, repo_type="dataset")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(p).read_bytes())
        hydrated += 1
    logger.info(
        "hydrated %d analyze artifacts from HF (%d .pt, %d raw, %d coverage) into %s",
        hydrated,
        len(pts),
        len(raw_files),
        len(cov_files),
        patch_dir,
    )
    return hydrated


def phase_analyze(
    repo_root: Path, *, primary_layer: int, skip_judge: bool = False, hydrate: bool = True
) -> None:
    """Off-pod CPU judge + f_CV bootstrap + hero figure.

    Three steps, all off-pod CPU: (0) HYDRATE the per-cell ``.pt`` + raw_completions
    from HF (B2 — analyze runs off-pod on a fresh VM AFTER the pod terminates, so
    the local ``patch/`` dir is empty; pull the sweep's HF-uploaded artifacts first),
    then (1) the vendored #537 judge (Sonnet 4.5) over the per-cell raw_completions →
    ``{cell}_judged.json`` (closes the ``e-judging-pipeline-not-vendored`` concern),
    then (2) ``issue697_analysis.py`` (f_CV bootstrap + hero). ``--skip-judge`` runs
    only the v-space analysis (CPU smoke / no API key); ``hydrate=False`` skips the
    HF pull (the CPU smoke analyzes its own local smoke tensors).
    """
    phase_log("analyze")
    patch_dir = repo_root / "eval_results" / "issue_697" / "patch"
    if hydrate:
        _hydrate_analyze_artifacts(repo_root)
    if not skip_judge:
        judge_cmd = [
            "uv",
            "run",
            "python",
            "scripts/issue697_judge.py",
            "--patch-dir",
            str(patch_dir),
        ]
        rc = _run_with_log(judge_cmd, log_path=_log_dir() / "judge.log", cwd=repo_root)
        if rc != 0:
            raise RuntimeError(f"analyze: judge step failed (rc={rc}); see {_log_dir()}/judge.log")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue697_analysis.py",
        "--primary-layer",
        str(primary_layer),
    ]
    log_path = _log_dir() / "analyze.log"
    rc = _run_with_log(cmd, log_path=log_path, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(f"analyze phase failed (rc={rc}); see {log_path}")
    logger.info("[phase=analyze_done]")


# ---------------------------------------------------------------------------
# Panel JSON materialization
# ---------------------------------------------------------------------------


def _materialize_panel(repo_root: Path) -> tuple[Path, Path]:
    """Write the fixed panel personas + questions JSON each cell reads."""
    from explore_persona_space.experiments.issue_651 import (
        build_panel_personas,
        build_panel_questions,
    )

    panel_dir = repo_root / "eval_results" / "issue_697" / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    personas = build_panel_personas()
    questions = build_panel_questions()
    p_path = panel_dir / "panel_personas.json"
    q_path = panel_dir / "panel_questions.json"
    p_path.write_text(json.dumps(personas, indent=2))
    q_path.write_text(json.dumps(questions, indent=2))
    logger.info("panel materialized: %d personas, %d questions", len(personas), len(questions))
    return p_path, q_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #697 dispatcher (vendor / canary / sweep / analyze). "
            "Smoke = sweep with one cell: `--cells em_sp_swe_seed42 --cpu-only`. "
            "Sweep shards the 128 cells over --n-gpus (default 4 -> the ft-7b "
            "4x A100-80 intent)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=[
            "vendor",
            "inert_read_assert",
            "rbase_prep",
            "canary",
            "smoke",
            "sweep",
            "analyze",
            "all",
        ],
        default=["all"],
        help=(
            "Phases to run in order. 'all' = vendor -> inert_read_assert -> "
            "rbase_prep -> smoke -> sweep. The §7.1a inert-read gate (inert_read_assert) "
            "+ the §7.1b smoke gate (smoke: one real-GPU cell -> non-inert assert -> "
            "smoke-pass artifact) both run BEFORE the sweep; the production sweep "
            "REFUSES to dispatch until the smoke-pass artifact exists (B1, plan §7.1)."
        ),
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Cell subset (e.g. 'em_sp_swe_seed42'); smoke = sweep with one cell.",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help=(
            "GPUs to shard the 128 sweep cells over (default 4 -> the ft-7b "
            "4x A100-80 intent; the wave dispatcher pins CUDA_VISIBLE_DEVICES "
            "per cell). 128/4 ≈ 5.3 h wall, under the 24 h GCP fence (plan §9)."
        ),
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU (smoke).")
    parser.add_argument(
        "--smoke-model",
        default=None,
        help=(
            "Tiny base model id for a CPU smoke (e.g. Qwen/Qwen2.5-0.5B-Instruct). "
            "Replaces the 7B base+FT load so the CPU canary/smoke runs without a GPU."
        ),
    )
    parser.add_argument(
        "--skip-e",
        action="store_true",
        help=(
            "Skip the behavioral-E on-policy generations (capture the mechanistic "
            "v only). The CPU smoke sets this (tiny-model generations are gibberish "
            "and the judge pools are not vendored; plan §4.5 / deferred-concern)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build + log each phase's per-cell commands, write the sentinel, and "
            "emit [phase=done] WITHOUT launching the subprocesses — exercises the "
            "cell-iteration / env-injection / sentinel / poll-contract plumbing on "
            "CPU (GPU-bound-phase carve-out item 2)."
        ),
    )
    parser.add_argument(
        "--floor-only",
        action="store_true",
        help="Existing-artifact floor only (seed-42 cells); auto-descope fallback.",
    )
    parser.add_argument(
        "--seed-42-only",
        action="store_true",
        help=(
            "The true seed-42 production grid: 64 cells (4 beh x 16 ctx x seed 42). "
            "--floor-only alone yields 96 (marker+fact keep both seeds); this filters "
            "the floor to seed==42 across ALL behaviors (plan §Salvage / A7)."
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the per-cell HF upload (local smoke; default uploads per cell).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help=(
            "Skip the analyze-phase Sonnet judge over raw_completions (CPU smoke / "
            "no ANTHROPIC_API_KEY); run only the v-space f_CV analysis."
        ),
    )
    parser.add_argument(
        "--skip-hydrate",
        action="store_true",
        help=(
            "Skip the analyze-phase HF hydration of per-cell .pt / raw_completions "
            "(B2). Set for a CPU smoke that analyzes its OWN local smoke tensors; the "
            "off-pod production analyze must hydrate (default) from HF."
        ),
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[7, 14, 21], help="Read/patch layers."
    )
    parser.add_argument(
        "--primary-layer", type=int, default=PRIMARY_LAYER, help="v READ layer (L_read=14)."
    )
    parser.add_argument(
        "--patch-layer",
        type=int,
        default=PATCH_LAYER,
        help=(
            "Donor-injection PATCH layer (L_patch=10, plan §4.0 Option B). MUST be "
            "< --primary-layer (patch == read is the v3 read-inert class)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help=(
            "R-generation cap (free-gen default 1024, plan §11/F5 — ≥2x the observed "
            "~150-tok median trained R; truncation creates silent zeros, #260)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    # `uv run python` does NOT auto-load .env; without this a fresh dispatcher
    # spawns subprocesses with HF_TOKEN/ANTHROPIC_API_KEY missing even though
    # every subprocess gets env={**os.environ} (the env dict came from an
    # unloaded parent). load_dotenv() at main()-top is the contract (#397 r10').
    from dotenv import load_dotenv

    load_dotenv()

    repo_root = _resolve_repo_root()
    phases = list(args.phase)
    if "all" in phases:
        # §7.1a inert-read gate -> rbase_prep caches R_base -> §7.1b smoke gate (one
        # real-GPU cell -> non-inert assert -> smoke-pass artifact) -> sweep (refuses
        # to dispatch until the smoke-pass artifact exists, B1). The canary's C2/C1.2
        # are REUSED from attempt-1 (plan §4.1 step 4) + the use_cache decision is
        # fetched from HF at sweep start, so canary is NOT in 'all' — run --phase
        # canary explicitly to re-exercise it on the 7B model.
        phases = ["vendor", "inert_read_assert", "rbase_prep", "smoke", "sweep"]

    cpu_only = args.cpu_only
    smoke = cpu_only or args.smoke_model is not None
    dry_run = args.dry_run
    upload = not args.no_upload and not smoke and not dry_run
    rbase_cache_dir = _rbase_cache_dir(repo_root)

    # Credential assert only when a phase needs HF (rbase_prep/canary/smoke/sweep).
    # Skip for a pure CPU smoke, the dry-run plumbing smoke, and a local analyze.
    needs_hf = ("rbase_prep", "canary", "smoke", "sweep")
    if any(p in needs_hf for p in phases) and not smoke and not dry_run:
        _require_credentials()

    if "sweep" in phases and not dry_run:
        panel_personas_json, panel_questions_json = _materialize_panel(repo_root)
    else:
        panel_personas_json = panel_questions_json = None

    for phase in phases:
        if phase == "vendor":
            phase_vendor(repo_root)
        elif phase == "inert_read_assert":
            # §7.1a (CPU, 0 GPU, non-skippable): the detector MUST fire on the
            # v3-salvaged inert .pt before the sweep dispatches.
            phase_inert_read_assert(repo_root, read_layer=args.primary_layer)
        elif phase == "rbase_prep":
            if dry_run:
                logger.info("[dry-run] rbase_prep -> scripts/issue697_rbase.py (skipped)")
                phase_log("rbase_prep")
                logger.info("[phase=rbase_prep_done]")
                continue
            phase_rbase_prep(
                repo_root,
                cpu_only=cpu_only,
                smoke_model=args.smoke_model,
                upload=upload,
                max_new_tokens=args.max_new_tokens,
            )
        elif phase == "canary":
            if dry_run:
                logger.info("[dry-run] canary -> scripts/issue697_canary.py (skipped)")
                phase_log("canary")
                logger.info("[phase=canary_done]")
                continue
            phase_canary(repo_root, cpu_only=cpu_only, smoke_model=args.smoke_model)
        elif phase == "smoke":
            # §7.1b gate phase (B1): one real-GPU cell through the same per-cell sweep
            # path -> non-inert assert -> smoke-pass artifact the sweep then requires.
            if dry_run:
                logger.info("[dry-run] smoke -> one-cell phase_sweep (skipped)")
                phase_log("smoke")
                logger.info("[phase=smoke_done]")
                continue
            phase_smoke(
                repo_root,
                n_gpus=args.n_gpus,
                cpu_only=cpu_only,
                smoke_model=args.smoke_model,
                layers=args.layers,
                primary_layer=args.primary_layer,
                patch_layer=args.patch_layer,
                max_new_tokens=args.max_new_tokens,
                upload=upload,
                rbase_cache_dir=rbase_cache_dir,
            )
        elif phase == "sweep":
            cells = _select_cells(args)
            phase_sweep(
                repo_root,
                cells,
                n_gpus=args.n_gpus,
                cpu_only=cpu_only,
                panel_personas_json=panel_personas_json,
                panel_questions_json=panel_questions_json,
                layers=args.layers,
                primary_layer=args.primary_layer,
                patch_layer=args.patch_layer,
                max_new_tokens=args.max_new_tokens,
                skip_e=args.skip_e or smoke,
                smoke_model=args.smoke_model,
                dry_run=dry_run,
                upload=upload,
                rbase_cache_dir=rbase_cache_dir,
            )
        elif phase == "analyze":
            # CPU smoke / no-API-key: skip the Sonnet judge step (the tiny-model
            # generations are gibberish and there may be no ANTHROPIC_API_KEY). The
            # CPU smoke analyzes its OWN local smoke tensors (--skip-hydrate); the
            # off-pod production analyze hydrates per-cell .pt + raw from HF (B2).
            phase_analyze(
                repo_root,
                primary_layer=args.primary_layer,
                skip_judge=args.skip_judge or smoke,
                hydrate=not (args.skip_hydrate or smoke),
            )

    note = (
        f"phases={phases} cells={args.cells or 'full(128)'} n_gpus={args.n_gpus} "
        f"smoke={smoke} dry_run={dry_run} upload={upload}"
    )
    write_sentinel(
        "epm:results",
        note,
        extra={"phases": phases, "smoke": smoke, "dry_run": dry_run, "n_gpus": args.n_gpus},
    )
    logger.info("[phase=done]")  # terminal marker — reserved for this single line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
