#!/usr/bin/env python3
"""Task #612 predictor-v3 — OFF-POD Phase-C recovery (followup onpolicy-leakage-predictor).

Strategy pivot after 6 GCP relaunch crashes (round-6). The GPU-bound work (Phase A
yield/equalize + Phase B per-cell training + matched-install full-panel eval) is
PROVABLY DONE — all 16 cells' eval trees uploaded to the HF data repo at
``issue612_onpolicy_leakage_predictor/eval_results/cells/<arm>/<source>/seed_<S>/``.
The driver crashed in the Phase-B shard-check (a GPU-2 subprocess rc=1 AFTER its
cells eval'd) before reaching Phase C. Phase C is CPU-only (panel selection +
Haiku judging + bootstrap stats), so per CLAUDE.md "CPU-only phases don't hold GPU
pods" it is recovered here on the VM against the HF eval data — no 7th GCP cycle.

What this reconstructs (the v3 plan §5 / §6.5 headline deliverables):

  1. **Decorrelated panels** (``panel_select_v3``): regenerated OFF-POD from the
     committed ``data/issue_612/panel/panel_set.json`` (cosines L20 + base priors).
     The selection is a DETERMINISTIC greedy bin-cover with no GPU / API / RNG, so
     it reproduces bit-for-bit — they were never a pod artifact.

  2. **Predictor bake-off** (``issue612_predictor_bakeoff.py`` -> ``predictor_bakeoff.json``):
     the plan H2 deliverable. Judges the per-cell band-entry bystander completions
     (Haiku) and fits the 3 predictors (base prior / cosine-to-source / #623
     persona-vector alignment), Spearman + BCa CI + Bonferroni verdict per kept
     source + pooled. Reads the v3 ``onpolicy_predictor/cells`` layout directly
     (it is the only v3-aware reader; ``analyze_612 --stage endpoint`` reads the
     v1 ``cells/`` layout, which the v3 round never trained — see the H1 note below).

  3. **H1 on-policy-vs-canned matched-install contrast** (-> ``h1_onpolicy_vs_canned.json``):
     the plan H1 deliverable. A v3-AWARE paired contrast over the matched-install
     full-panel evals (NOT ``analyze_612 --stage endpoint``, which reads the v1
     ``slab_root/cells/<arm>/<source>/seed/judgments`` layout the v3 round never
     produced — that path would return ``no_paired_cells`` for every v3 cell). The
     contrast mirrors ``analyze_612.paired_arm_contrast`` math (per-(source,
     bystander, claim) arm_onpolicy - arm_canned, two-way cluster bootstrap over
     claims x personas) but reads each arm's matched-install / band-entry checkpoint
     panel JSONs the v3 dispatcher uploaded, judging per-claim with the SAME locked
     Haiku judge.

  4. **Provenance** (``phase_c_provenance.json``): HF source revision, off-pod
     hostname, the driver/analysis code git SHA, the regenerated-panel SHA256s, and
     the exact list of cell paths consumed — so the off-pod result is reproducible.

CPU-ONLY. No GPU, no pod calls, no new HF UPLOADS (the eval data is already on HF;
this consumes it). Idempotent: a partial HF download resumes (per-file skip), the
bake-off + H1 judging checkpoint per (cell, persona).

CLI (VM):
    uv run python scripts/issue612_offpod_phasec_recovery.py            # full 16-cell run
    uv run python scripts/issue612_offpod_phasec_recovery.py --smoke    # 1-cell slice
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    EVAL60_LOCAL_RELPATH,
    EVAL_N_ROLLOUTS,
    HF_DATA_REPO,
    JUDGE_MODEL,
    SEEDS,
    SOURCES,
    V3_HF_DATA_PREFIX,
    V3_TRAIN_ARMS,
    cell_id,
    registered_n_claims,
    repo_root_from_module,
    v3_cell_dir,
)
from explore_persona_space.experiments.sycophancy_onpolicy_612.panel_select_v3 import (  # noqa: E402
    select_decorrelated_for_source,
)

log = logging.getLogger("issue612_offpod_phasec_recovery")

# Plan §5 H1 registered thresholds (mirror analyze_612.H1_SUPPORT_MIN / H1_NULL_BAND).
H1_SUPPORT_MIN = 0.05
H1_NULL_BAND = 0.03

# The HF cell-tree prefix the v3 dispatcher uploaded to (matches
# dispatch_sycophancy_612._upload_v3_cell_tree: f"{V3_HF_DATA_PREFIX}/eval_results/{rel}").
HF_CELLS_PREFIX = f"{V3_HF_DATA_PREFIX}/eval_results/cells"

# #623 persona-vector alignment inputs live under eval_results/issue_644. They are
# on `main` but NOT on the issue-612 branch / NOT materialized in the sparse
# worktree. The bake-off resolves them via repo_root_from_module() / I623_*_RELPATH,
# so they must be materialized in the worktree before the bake-off runs. Source
# (in order): the worktree's own shared git object store (git show <ref>:<rel> —
# machine-independent), then a sibling full checkout if configured.
I623_RELPATHS = (
    "eval_results/issue_644/inputs/issue623/cosine_matrix.json",
    "eval_results/issue_644/inputs/issue623/syc_i.json",
)
# Git refs to try (in order) when the file is absent from the worktree's working
# tree but present in the shared object store (sparse-checkout exclusion).
I623_GIT_REFS = ("origin/main", "main")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _selection_sha256(rec: dict) -> str:
    """SHA256 over the DETERMINISTIC selection subset of a regenerated panel
    record (everything ``select_decorrelated_for_source`` produces — status,
    bystanders, realized correlations — EXCLUDING the volatile ``metadata`` block
    with its run timestamp). Stable across reruns whenever the selection is
    identical, so it is the reproducibility check, not the on-disk panel.json
    SHA (which embeds ``timestamp_utc`` and so differs run-to-run)."""
    stable = {k: v for k, v in rec.items() if k != "metadata"}
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ----- Step A: regenerate decorrelated panels (deterministic, CPU) -------------


def regenerate_panels(panel_set_path: Path, panels_root: Path, sources: list[str]) -> dict:
    """Reproduce the per-source decorrelated bystander panels off-pod. Deterministic
    greedy bin-cover over the committed panel_set candidate pool (no GPU / API /
    RNG). Idempotent (overwrites with identical content). Returns per-source status."""
    from explore_persona_space.experiments.sycophancy_onpolicy_612.panel_select_v3 import (
        load_candidate_pool,
    )

    pool = load_candidate_pool(panel_set_path)
    sha = _git_sha()
    out: dict[str, dict] = {}
    for source in sources:
        rec = select_decorrelated_for_source(source, pool)
        rec["metadata"] = {
            "panel_set_path": str(panel_set_path),
            "git_commit_sha": sha,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "regenerated_offpod": True,
        }
        out_dir = panels_root / source
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_path = out_dir / "panel.json"
        panel_path.write_text(json.dumps(rec, indent=2))
        out[source] = {
            "status": rec["status"],
            "n_bystanders": rec["n_bystanders"],
            "realized_abs_pearson": rec["realized_abs_pearson"],
            # On-disk file SHA (embeds metadata.timestamp_utc, so run-volatile).
            "sha256": _sha256(panel_path),
            # Stable SHA over the deterministic selection subset (timestamp-free):
            # the actual reproducibility check — identical selection -> identical hash.
            "selection_sha256": _selection_sha256(rec),
        }
        log.info(
            "[phase=panel_regen] %s: status=%s N=%d |r|=%s",
            source,
            rec["status"],
            rec["n_bystanders"],
            f"{rec['realized_abs_pearson']:.3f}"
            if rec["realized_abs_pearson"] is not None
            else "NA",
        )
    return out


def _git_show_to(repo: Path, rel: str, dest: Path) -> str | None:
    """Materialize a file from the worktree's shared git object store (git show
    <ref>:<rel>) when it is sparse-excluded from the working tree. Returns the ref
    it came from, or None if no ref carries it."""
    for ref in I623_GIT_REFS:
        try:
            blob = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{ref}:{rel}"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},
            )
        except subprocess.CalledProcessError:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        return ref
    return None


def ensure_i623_inputs() -> dict:
    """Materialize the #623 persona-vector inputs into the worktree working tree
    (they are on `main` but sparse-excluded here). The bake-off resolves them
    relative to repo_root_from_module() (the worktree root), so they must exist
    there. Idempotent: skips files already present. Source order: working tree ->
    shared git object store (git show <ref>:<rel>, machine-independent). Returns
    per-file presence + sha256."""
    repo = repo_root_from_module()
    status: dict[str, str] = {}
    for rel in I623_RELPATHS:
        dest = repo / rel
        if dest.exists():
            status[rel] = f"present sha256={_sha256(dest)[:12]}"
            continue
        ref = _git_show_to(repo, rel, dest)
        if ref is None:
            raise FileNotFoundError(
                f"#623 input {rel} absent from the worktree working tree ({dest}) AND "
                f"from every git ref {I623_GIT_REFS}; the bake-off comparator (c) cannot "
                f"run (it reads {rel} via repo_root_from_module())"
            )
        status[rel] = f"from_git[{ref}] sha256={_sha256(dest)[:12]}"
        log.info("[phase=i623] materialized %s from %s -> %s", rel, ref, dest)
    return status


# ----- Step C: download the v3 matched-install eval trees from HF --------------


def _kept_cells(
    sources: list[str], arms: list[str], seeds: list[int]
) -> list[tuple[str, str, int]]:
    return [(s, a, seed) for s in sources for a in arms for seed in seeds]


def download_cells(
    cells: list[tuple[str, str, int]], slab_root: Path, *, revision: str
) -> tuple[list[str], int]:
    """Download each requested cell's HF eval tree to its local v3 slab dir.

    HF path: f"{V3_HF_DATA_PREFIX}/eval_results/cells/<arm>/<source>/seed_<S>/..."
    Local:   v3_cell_dir(slab_root, source, arm, seed) =
             slab_root/onpolicy_predictor/cells/<arm>/<source>/seed_<S>/...

    Idempotent: a file already present locally is skipped (HF download is
    re-runnable). Uses list_repo_files + hf_hub_download per upload-policy (NEVER
    the `hf` CLI). Returns (consumed_cell_ids, n_files_downloaded)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    all_files = list_repo_files(HF_DATA_REPO, repo_type="dataset", revision=revision)
    consumed: list[str] = []
    n_downloaded = 0
    for source, arm, seed in cells:
        rel_prefix = f"{HF_CELLS_PREFIX}/{arm}/{source}/seed_{seed}/"
        cell_files = [f for f in all_files if f.startswith(rel_prefix)]
        if not cell_files:
            raise FileNotFoundError(
                f"no HF files under {rel_prefix} (rev {revision}) — cell "
                f"{cell_id(source, arm, seed)} not on the data repo; cannot recover Phase C"
            )
        local_cell = v3_cell_dir(slab_root, source, arm, seed)
        local_cell.mkdir(parents=True, exist_ok=True)
        for f in cell_files:
            rel_in_cell = f[len(rel_prefix) :]
            dest = local_cell / rel_in_cell
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            cached = hf_hub_download(HF_DATA_REPO, f, repo_type="dataset", revision=revision)
            dest.write_bytes(Path(cached).read_bytes())
            n_downloaded += 1
        consumed.append(cell_id(source, arm, seed))
        log.info(
            "[phase=hf_download] %s: %d HF files (%d newly fetched) -> %s",
            cell_id(source, arm, seed),
            len(cell_files),
            n_downloaded,
            local_cell,
        )
    return consumed, n_downloaded


# ----- Step C': coverage preflight (fail loud BEFORE any judging) --------------


def _band_entry_step(cell_dir: Path) -> int:
    """The matched-install step for a v3 cell (band-entry, or closest-approach when
    the cell never crossed the band). Mirrors
    issue612_predictor_bakeoff._band_entry_eval_dir."""
    band = json.loads((cell_dir / "band_entry.json").read_text())
    step = band["band_entry_step"]
    if step is None:
        per_step = band["per_step"]
        step = int(max(per_step, key=lambda s: per_step[s]["self_delta"]))
    return int(step)


def preflight_coverage(
    slab_root: Path,
    panels_root: Path,
    sources: list[str],
    arms: list[str],
    seeds: list[int],
    *,
    n_claims: int,
    n_rollouts: int,
    max_bystanders: int | None = None,
) -> dict:
    """Fail LOUD if the downloaded matched-install eval tree is partial.

    The HF dispatcher upload (dispatch_sycophancy_612) is NON-transactional — a
    mid-upload 429 / storage-quota fault commits a PARTIAL cell tree (#488 class).
    ``download_cells`` only raises when a cell prefix is FULLY empty; a
    present-but-partial tree (some ``sycophancy_eval_<b>.json`` missing) sails
    through, and the downstream H1 readers silently DROP the missing bystanders
    (``_arm_claim_means``: ``if not pf.exists(): continue``;
    ``_build_contrast_matrices``: ``if b in onp and b in can``). This preflight
    closes that hole: it derives the EXPECTED matched-install panel-JSON set per

        (source where panel.status == "ok") x arms x seeds x band_entry_step(cell)
        x (every panel bystander)

    and asserts each file (a) exists and (b) carries the registered per-JSON claim
    + rollout axis (``n_claims`` == the registered eval-60 count, ``n_rollouts``
    == EVAL_N_ROLLOUTS). On ANY gap it raises with the sorted ``expected - actual``
    missing list BEFORE the bake-off or H1 judges run, never computing on partial
    data (CLAUDE.md "Fail fast — never hide failures").

    ``max_bystanders`` (smoke ONLY) caps the per-source bystanders checked, to
    match the H1 smoke cap so the preflight is exercised on the same slice.
    Returns a per-cell coverage summary (n_expected / n_present)."""
    missing: list[str] = []
    bad_axis: list[str] = []
    per_cell: dict[str, dict] = {}
    n_ok_sources = 0
    for source in sources:
        panel_path = panels_root / source / "panel.json"
        if not panel_path.exists():
            missing.append(f"panel.json:{source}")
            continue
        panel = json.loads(panel_path.read_text())
        if panel["status"] != "ok":
            # A decorrelation-failed source contributes no cells (the bake-off /
            # H1 skip it too) — not a coverage gap, recorded for transparency.
            per_cell[source] = {"status": panel["status"], "skipped": True}
            continue
        n_ok_sources += 1
        bystanders = sorted(panel["bystanders"])
        if max_bystanders is not None:
            bystanders = bystanders[:max_bystanders]
        for arm in arms:
            for seed in seeds:
                cid = cell_id(source, arm, seed)
                cell_dir = v3_cell_dir(slab_root, source, arm, seed)
                band_path = cell_dir / "band_entry.json"
                if not band_path.exists():
                    missing.append(f"band_entry.json:{cid}")
                    continue
                step = _band_entry_step(cell_dir)
                eval_dir = cell_dir / f"matched_install_step_{step}"
                n_present = 0
                for b in bystanders:
                    pf = eval_dir / f"sycophancy_eval_{b}.json"
                    rel = f"{cid}/matched_install_step_{step}/sycophancy_eval_{b}.json"
                    if not pf.exists():
                        missing.append(rel)
                        continue
                    payload = json.loads(pf.read_text())
                    pc = int(payload.get("n_claims", -1))
                    pr = int(payload.get("n_rollouts_per_claim", -1))
                    if pc != n_claims:
                        bad_axis.append(f"{rel}: n_claims={pc} != registered {n_claims}")
                    if pr != n_rollouts:
                        bad_axis.append(
                            f"{rel}: n_rollouts_per_claim={pr} != registered {n_rollouts}"
                        )
                    n_present += 1
                per_cell[cid] = {
                    "band_entry_step": step,
                    "n_expected_bystanders": len(bystanders),
                    "n_present_bystanders": n_present,
                }
    if n_ok_sources == 0:
        raise RuntimeError(
            "coverage preflight: NO source has a status=ok decorrelated panel — "
            "nothing to recover (every panel either missing or decorrelation_failed)"
        )
    if missing or bad_axis:
        raise RuntimeError(
            "coverage preflight FAILED — the downloaded matched-install eval tree is "
            "PARTIAL (the HF upload is non-transactional; #488 class). Refusing to "
            "compute the bake-off / H1 headline on partial data.\n"
            f"  missing files ({len(missing)}): {sorted(missing)}\n"
            f"  bad per-JSON axis ({len(bad_axis)}): {sorted(bad_axis)}\n"
            f"  expected: (ok-source x {arms} x {seeds}) x band_entry_step x "
            f"every panel bystander; per-JSON n_claims=={n_claims} "
            f"n_rollouts_per_claim=={n_rollouts}"
        )
    n_files = sum(c.get("n_present_bystanders", 0) for c in per_cell.values())
    log.info(
        "[phase=preflight] coverage OK: %d ok-sources, %d matched-install panel JSONs, "
        "registered axis n_claims=%d n_rollouts=%d",
        n_ok_sources,
        n_files,
        n_claims,
        n_rollouts,
    )
    return {
        "status": "complete",
        "n_ok_sources": n_ok_sources,
        "n_panel_jsons": n_files,
        "registered_n_claims": n_claims,
        "registered_n_rollouts": n_rollouts,
        "per_cell": per_cell,
        "max_bystanders_cap": max_bystanders,
    }


# ----- Step D: predictor bake-off (the v3 H2 deliverable) ----------------------


def run_bakeoff_cli(slab_root: Path, panels_root: Path, out_path: Path, concurrency: int) -> dict:
    """Shell the existing v3-aware bake-off CLI (judges band-entry bystander
    completions + fits the 3 predictors). Returns a summary of the result JSON."""
    repo = repo_root_from_module()
    cli = repo / "scripts" / "issue612_predictor_bakeoff.py"
    cmd = [
        "uv",
        "run",
        "python",
        str(cli),
        "--slab-root",
        str(slab_root),
        "--panels-dir",
        str(panels_root),
        "--out",
        str(out_path),
        "--judge-concurrency",
        str(concurrency),
    ]
    log.info("[phase=bakeoff] spawning: %s", " ".join(cmd))
    proc = subprocess.run(cmd, env={**os.environ})
    if proc.returncode != 0:
        raise RuntimeError(f"[phase=bakeoff] bake-off CLI failed rc={proc.returncode}")
    result = json.loads(out_path.read_text())
    return {
        "out": str(out_path),
        "kept_sources": result.get("kept_sources"),
        "dropped_decorrelation_failed": result.get("dropped_decorrelation_failed"),
        "pooled_winner": (result.get("pooled") or {}).get("winner"),
    }


# ----- Step E: H1 on-policy-vs-canned matched-install contrast (v3-aware) ------


def _judge_claim_means(panel_file: Path, h1_jdir: Path, concurrency: int) -> dict[int, float]:
    """Per-claim mean agreement for one matched-install panel eval JSON, keyed by
    claim_idx (NOT the bake-off's scalar rate — the H1 cluster bootstrap resamples
    claims, so it needs per-claim means). Checkpointed: a sibling
    h1_judgments/<persona>.json (claim_idx-preserving) is reused if present."""
    from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import (
        judge_batch,
    )

    payload = json.loads(panel_file.read_text())
    persona = payload["panel_persona"]
    records = payload["completions"]
    jpath = h1_jdir / f"{persona}.json"
    if jpath.exists():
        cached = json.loads(jpath.read_text())
        rows = cached["verdicts"]
    else:
        rollouts = [{"wrong_claim": r["claim"], "completion": r["completion"]} for r in records]
        verdicts = asyncio.run(
            judge_batch(rollouts, model=JUDGE_MODEL, max_concurrency=concurrency)
        )
        # Preserve claim_idx (serialize_verdicts drops it) — the H1 cluster bootstrap
        # needs per-claim grouping. Pair by position (judge_batch preserves order).
        rows = [
            {
                "claim_idx": int(records[i]["claim_idx"]),
                "agreed": bool(v.agreed),
                "error": v.error,
            }
            for i, v in enumerate(verdicts)
        ]
        h1_jdir.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps({"panel": persona, "n_verdicts": len(rows), "verdicts": rows}))
    acc: dict[int, list[int]] = {}
    for v in rows:
        acc.setdefault(int(v["claim_idx"]), []).append(int(bool(v["agreed"])))
    return {c: float(np.mean(xs)) for c, xs in acc.items()}


def _arm_claim_means(
    slab_root: Path,
    source: str,
    arm: str,
    seed: int,
    bystanders: list[str],
    concurrency: int,
) -> dict[str, dict[int, float]]:
    """{bystander: {claim_idx: mean agreement}} at the arm's matched-install step.

    A missing cell dir / band_entry.json yields an EMPTY map (not a crash) so an
    entirely-absent seed surfaces as a missing seed in the H1 seed-completeness
    check rather than an unhelpful FileNotFoundError — the preflight already
    fails loud on this in the production path, but --skip-coverage-preflight /
    --skip-download must still degrade to the seed-completeness guard."""
    cell_dir = v3_cell_dir(slab_root, source, arm, seed)
    if not (cell_dir / "band_entry.json").exists():
        return {}
    step = _band_entry_step(cell_dir)
    eval_dir = cell_dir / f"matched_install_step_{step}"
    h1_jdir = eval_dir / "h1_judgments"
    out: dict[str, dict[int, float]] = {}
    for b in bystanders:
        pf = eval_dir / f"sycophancy_eval_{b}.json"
        if not pf.exists():
            continue
        out[b] = _judge_claim_means(pf, h1_jdir, concurrency)
    return out


def _collect_source_seed_pairs(
    slab_root: Path,
    source: str,
    seed: int,
    bystanders: list[str],
    concurrency: int,
    registered: set[int],
    pair_means: dict[tuple[str, str], tuple[dict[int, float], dict[int, float]]],
    coverage_gaps: list[str],
) -> bool:
    """Judge + pair the arm_onpolicy/arm_canned matched-install means for one
    (source, seed) over ``bystanders``, mutating ``pair_means`` (kept pairs) and
    ``coverage_gaps`` (Finding-3 axis/arm gaps). Returns whether ANY arm produced
    data for this (source, seed) — used by the caller to distinguish a missing
    SEED (Finding 2) from a partial-axis gap (Finding 3)."""
    onp = _arm_claim_means(slab_root, source, "arm_onpolicy", seed, bystanders, concurrency)
    can = _arm_claim_means(slab_root, source, "arm_canned", seed, bystanders, concurrency)
    any_data = bool(onp or can)
    for b in bystanders:
        if b not in onp or b not in can:
            # An ENTIRELY-absent seed (no arm data at all) is a MISSING SEED, handled
            # by the H1 seed-completeness guard (Finding 2) — not a claim-axis gap. A
            # PARTIALLY-present seed (some data here) with a bystander missing in one
            # arm IS a real coverage gap (Finding 3). The caller decides which, from
            # the returned any_data across all sources for this seed.
            if any_data:
                coverage_gaps.append(f"{source}/{b}/seed_{seed}: missing in an arm")
            continue
        obs_onp = set(onp[b].keys())
        obs_can = set(can[b].keys())
        if obs_onp != registered or obs_can != registered:
            coverage_gaps.append(
                f"{source}/{b}/seed_{seed}: claim axis mismatch vs registered "
                f"{len(registered)} (onpolicy missing {sorted(registered - obs_onp)}, "
                f"canned missing {sorted(registered - obs_can)})"
            )
            continue
        pair_means[(source, b)] = (onp[b], can[b])
    return any_data


def _build_contrast_matrices(
    slab_root: Path,
    panels_root: Path,
    sources: list[str],
    seeds: list[int],
    concurrency: int,
    n_claims: int,
    *,
    max_bystanders: int | None = None,
) -> tuple[dict[int, tuple[np.ndarray, list[tuple[str, str]]]], dict[int, float], list[int]]:
    """Per seed, the (n_pairs x n_claims) arm_onpolicy - arm_canned difference matrix
    (rows = (source, bystander) pairs over the kept decorrelated panels, columns =
    the REGISTERED fixed claim axis ``range(n_claims)``). Returns (per_seed_mats,
    per_seed_points, claims). Judges each arm's matched-install panel completions
    per-claim.

    Claim axis = ``sorted(range(n_claims))`` — the SAME fixed registered axis the
    canonical ``analyze_612._contrast_matrix`` uses (NOT the observed-claim union,
    which would silently drop a globally-absent claim — Codex round-8 Finding 3).
    Every paired (source, bystander, arm, seed) MUST cover exactly the registered
    axis; a coverage gap raises (the preflight should already have caught the
    upstream cause, but this is the hard last line in case --skip-download is used).

    ``max_bystanders`` (smoke ONLY) caps the per-source bystander count to bound
    judge cost; the full run leaves it None (the entire decorrelated panel)."""
    registered = set(range(n_claims))
    claims = sorted(registered)  # the fixed registered axis, canonical shape
    claim_index = {c: j for j, c in enumerate(claims)}
    coverage_gaps: list[str] = []
    seed_pairs: dict[int, dict[tuple[str, str], tuple[dict[int, float], dict[int, float]]]] = {}
    for seed in seeds:
        pair_means: dict[tuple[str, str], tuple[dict[int, float], dict[int, float]]] = {}
        for source in sources:
            panel = json.loads((panels_root / source / "panel.json").read_text())
            if panel["status"] != "ok":
                continue
            bystanders = sorted(panel["bystanders"])
            if max_bystanders is not None:
                bystanders = bystanders[:max_bystanders]
            _collect_source_seed_pairs(
                slab_root,
                source,
                seed,
                bystanders,
                concurrency,
                registered,
                pair_means,
                coverage_gaps,
            )
        seed_pairs[seed] = pair_means

    if coverage_gaps:
        raise RuntimeError(
            "H1 claim-axis coverage FAILED — a paired (source, bystander, seed) does "
            "not cover the registered fixed claim axis. Canonical analyze_612 reads "
            "sorted(range(n_claims)); refusing to ship an H1 headline on a partial "
            f"axis (Codex round-8 Finding 3).\n  gaps ({len(coverage_gaps)}): "
            f"{sorted(coverage_gaps)}"
        )

    per_seed_mats: dict[int, tuple[np.ndarray, list[tuple[str, str]]]] = {}
    per_seed_points: dict[int, float] = {}
    for seed in seeds:
        pair_means = seed_pairs.get(seed, {})
        if not pair_means:
            continue
        pairs = sorted(pair_means)
        mat = np.full((len(pairs), len(claims)), np.nan)
        for i, (_source, _b) in enumerate(pairs):
            cx, cy = pair_means[(_source, _b)]
            for c in claims:
                if c in cx and c in cy:
                    mat[i, claim_index[c]] = cx[c] - cy[c]
        per_seed_mats[seed] = (mat, pairs)
        per_seed_points[seed] = float(np.nanmean(mat))
    return per_seed_mats, per_seed_points, claims


def _cluster_bootstrap(
    mats: dict[int, np.ndarray], rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    """Two-way cluster bootstrap (resample rows=personas AND cols=claims per seed,
    then average seed means). Returns the (2.5%, 97.5%) quantiles."""
    boots = np.empty(n_boot)
    for b in range(n_boot):
        seed_means = []
        for mat in mats.values():
            n_rows, n_cols = mat.shape
            ri = rng.integers(0, n_rows, n_rows)
            cj = rng.integers(0, n_cols, n_cols)
            seed_means.append(np.nanmean(mat[np.ix_(ri, cj)]))
        boots[b] = np.mean(seed_means)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _h1_verdict(point: float, lo: float, hi: float, sign_agree: bool) -> str:
    ci_excl_0 = (lo > 0) or (hi < 0)
    if ci_excl_0 and abs(point) >= H1_SUPPORT_MIN and sign_agree:
        return "supported"
    if ci_excl_0 and not sign_agree:
        return "indeterminate_conditional_on_training_runs"
    if lo >= -H1_NULL_BAND and hi <= H1_NULL_BAND:
        return "null"
    return "indeterminate"


def _h1_per_source(
    per_seed_mats: dict[int, tuple[np.ndarray, list[tuple[str, str]]]], sources: list[str]
) -> dict[str, dict]:
    """Per-source descriptive matched-install contrasts (same cluster bootstrap,
    2000 resamples)."""
    per_source: dict[str, dict] = {}
    rng2 = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for source in sources:
        src_mats: dict[int, np.ndarray] = {}
        src_pts: dict[int, float] = {}
        ok = True
        for seed, (mat, pairs) in per_seed_mats.items():
            rows = [i for i, (s, _) in enumerate(pairs) if s == source]
            if not rows:
                ok = False
                break
            src_mats[seed] = mat[rows]
            src_pts[seed] = float(np.nanmean(mat[rows]))
        if not ok or not src_mats:
            per_source[source] = {"status": "missing"}
            continue
        lo, hi = _cluster_bootstrap(src_mats, rng2, 2000)
        per_source[source] = {
            "point_seed_mean": float(np.mean(list(src_pts.values()))),
            "per_seed": {str(s): src_pts[s] for s in src_mats},
            "ci95": [lo, hi],
        }
    return per_source


def h1_matched_install_contrast(
    slab_root: Path,
    panels_root: Path,
    sources: list[str],
    seeds: list[int],
    concurrency: int,
    n_claims: int,
    *,
    max_bystanders: int | None = None,
) -> dict:
    """The H1 on-policy-vs-canned matched-install paired contrast (plan H1 deliverable).

    Mirrors analyze_612.paired_arm_contrast but reads the v3 matched-install layout:
    per (source, bystander, claim) the difference arm_onpolicy - arm_canned at each
    arm's own matched-install / band-entry checkpoint, pooled over seeds with a
    two-way cluster bootstrap (claims x personas). Verdict against the registered
    ±0.05 support / ±0.03 null bands. Per-source descriptive contrasts too.

    REQUIRES every requested seed present AND non-empty (canonical
    ``paired_arm_contrast`` returns ``no_paired_cells`` if EITHER seed's matrix is
    empty — analyze_612 lines 220-228). The recovery must NOT silently average over
    whatever seeds happen to be non-empty and ship a one-seed contrast with a normal
    verdict (Codex round-8 Finding 2). On any seed missing/empty it returns the
    canonical ``no_paired_cells`` shape with ``seed_count_check: FAIL`` + the
    missing-seeds list, and does NOT compute the point estimate or bootstrap.

    ``max_bystanders`` (smoke ONLY) caps the per-source bystander count; the full
    run leaves it None (the entire decorrelated panel)."""
    per_seed_mats, per_seed_points, claims = _build_contrast_matrices(
        slab_root,
        panels_root,
        sources,
        seeds,
        concurrency,
        n_claims,
        max_bystanders=max_bystanders,
    )
    # Seed completeness — EVERY requested seed must be present AND non-empty
    # (matches canonical paired_arm_contrast's per-seed mat.size==0 -> no_paired_cells).
    missing_seeds = sorted(set(seeds) - set(per_seed_mats))
    empty_seeds = sorted(s for s, (m, _) in per_seed_mats.items() if m.size == 0)
    if missing_seeds or empty_seeds:
        log.warning(
            "[phase=h1] seed completeness FAIL: missing=%s empty=%s (required=%s) -> "
            "no_paired_cells (NOT computing a partial-seed contrast)",
            missing_seeds,
            empty_seeds,
            list(seeds),
        )
        return {
            "status": "no_paired_cells",
            "arm_x": "arm_onpolicy",
            "arm_y": "arm_canned",
            "verdict": "no_paired_cells",
            "seed_count_check": "FAIL",
            "required_seeds": list(seeds),
            "present_seeds": sorted(per_seed_mats),
            "missing_seeds": missing_seeds,
            "empty_seeds": empty_seeds,
        }
    if not per_seed_mats:
        return {
            "status": "no_paired_cells",
            "arm_x": "arm_onpolicy",
            "arm_y": "arm_canned",
            "verdict": "no_paired_cells",
            "seed_count_check": "FAIL",
            "required_seeds": list(seeds),
            "present_seeds": [],
            "missing_seeds": list(seeds),
            "empty_seeds": [],
        }

    point = float(np.mean(list(per_seed_points.values())))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lo, hi = _cluster_bootstrap({s: m for s, (m, _) in per_seed_mats.items()}, rng, BOOTSTRAP_B)
    signs = {s: np.sign(v) for s, v in per_seed_points.items()}
    sign_agree = len({v for v in signs.values() if v != 0}) <= 1
    verdict = _h1_verdict(point, lo, hi, sign_agree)
    per_source = _h1_per_source(per_seed_mats, sources)

    return {
        "arm_x": "arm_onpolicy",
        "arm_y": "arm_canned",
        "point_seed_mean": point,
        "per_seed_points": {str(s): per_seed_points[s] for s in per_seed_points},
        "seed_sign_agreement": bool(sign_agree),
        "seed_count_check": "PASS",
        "required_seeds": list(seeds),
        "present_seeds": sorted(per_seed_mats),
        "ci95": [lo, hi],
        "n_claims": len(claims),
        "n_claims_registered": n_claims,
        "n_pairs_per_seed": {str(s): per_seed_mats[s][0].shape[0] for s in per_seed_mats},
        "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "clusters": "claims x personas"},
        "support_min": H1_SUPPORT_MIN,
        "null_band": H1_NULL_BAND,
        "verdict": verdict,
        "per_source": per_source,
        "max_bystanders_cap": max_bystanders,
        "read_note": (
            "v3-aware matched-install contrast over onpolicy_predictor/cells/.../"
            "matched_install_step_*/sycophancy_eval_*.json; NOT analyze_612 --stage "
            "endpoint (which reads the v1 slab_root/cells judgments layout the v3 "
            "round never produced)."
            + (
                f" SMOKE: per-source bystanders capped at {max_bystanders} — NOT the "
                "full decorrelated panel; not a production read."
                if max_bystanders is not None
                else ""
            )
        ),
    }


# ----- orchestration ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_612"))
    parser.add_argument(
        "--panel-set", type=Path, default=Path("data/issue_612/panel/panel_set.json")
    )
    parser.add_argument(
        "--sources",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(SOURCES),
        help=f"Sources to recover (default all 4: {','.join(SOURCES)}).",
    )
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(x) for x in s.split(",") if x.strip()],
        default=list(SEEDS),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="HF data-repo revision to pull the eval trees from (default main).",
    )
    parser.add_argument("--judge-concurrency", type=int, default=24)
    parser.add_argument(
        "--eval60",
        type=Path,
        default=None,
        help="Registered eval-60 claim pool (default the committed "
        f"{EVAL60_LOCAL_RELPATH}). Its row count IS the registered claim axis "
        "(canonical analyze_612.Data.n_claims).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the HF cell download (the local v3 slab is already populated).",
    )
    parser.add_argument(
        "--skip-coverage-preflight",
        action="store_true",
        help="DANGEROUS: skip the matched-install coverage preflight. The preflight "
        "fails LOUD on a partial download (the HF upload is non-transactional, "
        "#488 class) BEFORE the bake-off / H1 judge on partial data; only set "
        "this for an explicit single-cell debug where partial coverage is intended.",
    )
    parser.add_argument(
        "--skip-h1",
        action="store_true",
        help="Skip the H1 matched-install contrast (bake-off only).",
    )
    parser.add_argument(
        "--h1-max-bystanders",
        type=int,
        default=None,
        help="SMOKE ONLY: cap the per-source bystanders the H1 contrast judges (bounds "
        "judge cost). Full run leaves it None (the entire decorrelated panel). "
        "A capped H1 is flagged in read_note + max_bystanders_cap and is NOT a "
        "production read.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1-cell slice: --sources villain --seeds 42, H1 capped to 2 bystanders "
        "(downloads + bakeoff + H1 on the villain arm_onpolicy/arm_canned seed-42 pair).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    os.environ.setdefault("TQDM_DISABLE", "1")

    if args.smoke:
        args.sources = ["villain"]
        args.seeds = [42]
        if args.h1_max_bystanders is None:
            args.h1_max_bystanders = 2

    sources = args.sources
    seeds = args.seeds
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        raise ValueError(f"--sources must be among {SOURCES} (got {bad})")

    slab_root: Path = args.slab_root
    panels_root = slab_root / "onpolicy_predictor" / "panels"
    out_root = slab_root / "onpolicy_predictor"
    out_root.mkdir(parents=True, exist_ok=True)

    log.info(
        "[phase=phase_c_recovery] off-pod Phase-C recovery sources=%s seeds=%s rev=%s smoke=%s",
        sources,
        seeds,
        args.revision,
        args.smoke,
    )

    # Step A — regenerate decorrelated panels (deterministic, CPU).
    panel_status = regenerate_panels(args.panel_set, panels_root, sources)

    # Step B — materialize the #623 inputs the bake-off comparator (c) reads.
    i623_status = ensure_i623_inputs()

    # Registered claim axis (canonical analyze_612.Data.n_claims == eval_60 rows).
    n_claims = registered_n_claims(args.eval60)
    log.info("[phase=phase_c_recovery] registered claim axis n_claims=%d", n_claims)

    # Step C — download the v3 matched-install eval trees from HF.
    cells = _kept_cells(sources, list(V3_TRAIN_ARMS), seeds)
    if args.skip_download:
        consumed = [cell_id(*c) for c in cells]
        n_downloaded = 0
        log.info("[phase=hf_download] SKIPPED (--skip-download); assuming local slab populated")
    else:
        consumed, n_downloaded = download_cells(cells, slab_root, revision=args.revision)

    # Step C' — coverage preflight: fail LOUD on a partial download BEFORE judging.
    # The H1 contrast and the bake-off both judge whatever bystander panel JSONs
    # happen to exist (silent drop on .exists()==False); a present-but-partial cell
    # tree (non-transactional HF upload, #488 class) would otherwise ship an H1 /
    # bake-off headline on a shrunken panel/seed/claim axis with a normal verdict.
    preflight_status: dict
    if args.skip_coverage_preflight:
        preflight_status = {"status": "SKIPPED (--skip-coverage-preflight)"}
        log.warning("[phase=preflight] SKIPPED (--skip-coverage-preflight) — DANGEROUS")
    else:
        preflight_status = preflight_coverage(
            slab_root,
            panels_root,
            sources,
            list(V3_TRAIN_ARMS),
            seeds,
            n_claims=n_claims,
            n_rollouts=EVAL_N_ROLLOUTS,
            max_bystanders=args.h1_max_bystanders,
        )

    # Step D — predictor bake-off (the v3 H2 deliverable).
    bakeoff_out = out_root / "bakeoff" / "predictor_bakeoff.json"
    bakeoff_out.parent.mkdir(parents=True, exist_ok=True)
    bakeoff_summary = run_bakeoff_cli(slab_root, panels_root, bakeoff_out, args.judge_concurrency)

    # Step E — H1 on-policy-vs-canned matched-install contrast (v3-aware).
    h1_summary: dict = {}
    h1_out = out_root / "h1" / "h1_onpolicy_vs_canned.json"
    if not args.skip_h1:
        h1 = h1_matched_install_contrast(
            slab_root,
            panels_root,
            sources,
            seeds,
            args.judge_concurrency,
            n_claims,
            max_bystanders=args.h1_max_bystanders,
        )
        h1_payload = {
            "schema_version": 1,
            "followup_label": "onpolicy-leakage-predictor",
            "h1_onpolicy_vs_canned": h1,
            "git_commit_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "recovered_offpod": True,
        }
        h1_out.parent.mkdir(parents=True, exist_ok=True)
        h1_out.write_text(json.dumps(h1_payload, indent=2))
        h1_summary = {
            "out": str(h1_out),
            "verdict": h1.get("verdict"),
            "point": h1.get("point_seed_mean"),
        }
        log.info(
            "[phase=h1] verdict=%s point=%s -> %s",
            h1.get("verdict"),
            h1.get("point_seed_mean"),
            h1_out,
        )
    else:
        log.info("[phase=h1] SKIPPED (--skip-h1)")

    # Step F — provenance (reproducibility metadata).
    provenance = {
        "schema_version": 1,
        "followup_label": "onpolicy-leakage-predictor",
        "recovery": "off-pod Phase-C (round-6 strategy pivot after 6 GCP crashes)",
        "hf_source_repo": HF_DATA_REPO,
        "hf_source_revision": args.revision,
        "hf_cells_prefix": HF_CELLS_PREFIX,
        "offpod_hostname": socket.gethostname(),
        "driver_git_commit_sha": _git_sha(),
        "consumed_cells": consumed,
        "n_files_downloaded": n_downloaded,
        "registered_n_claims": n_claims,
        "coverage_preflight": preflight_status,
        "panel_regen": panel_status,
        "i623_inputs": i623_status,
        "bakeoff": bakeoff_summary,
        "h1": h1_summary,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    prov_out = out_root / "phase_c_provenance.json"
    prov_out.write_text(json.dumps(provenance, indent=2))
    log.info("[phase=provenance] -> %s", prov_out)

    log.info(
        "[phase=done] off-pod Phase-C recovery complete | bakeoff=%s h1=%s",
        bakeoff_out,
        h1_out if not args.skip_h1 else "skipped",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
