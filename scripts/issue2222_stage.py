"""P0 staging for issue #2222 — Persona Vectors screening predictors (plan v5 §4 P0).

Stages + pin-verifies every reused input, builds the fixed per-dataset
subsample manifests, re-asserts the plan-time leg-A lineage dispositions, and
writes the ``staging_ready.json`` sentinel (plan §9 phase_outputs.P0):

1. dataset.zip — download (GitHub) if absent, verify the git-blob sha1 pin,
   extract, count the 24 files' rows.
2. Fixed subsample — seed-42, S=1000 rows/dataset (all arms consume the SAME
   rows), admission through the exact capture-path token-budget filter.
3. #778 ``r_B`` — ``rb_v2`` primary; ``rb`` v1 fallback is a NAMED deviation
   logged loudly + recorded in the sentinel (plan §8/§10). Shape-asserted
   (28, 3584) per trait.
4. #1739 frozen maps — staged + verified via ``verify_reused_artifact_keys.py``
   (keys w,x_mu,x_sd,y_mu,layers) + own shape/dtype asserts.
5. #778 y-axis JSONs — located in git (``eval_results/issue_778``); on a
   sparse worktree, ``git sparse-checkout add`` is attempted first (>= 72
   ``finetune_*.json`` required).
6. Leg-A lineage re-assertion — the plan §10 parent-branch diffs re-run; any
   branch-side commit NOT covered by a declared disposition fails loud
   (artifact-reuse check (k)). SHALLOW-AWARE (#2222 crash fix): on a depth-1
   pod bootstrap clone the range mis-attributes already-merged main commits as
   branch-side, so flagged SHAs are post-filtered via
   ``git merge-base --is-ancestor`` with a bounded deepen ladder; full-clone
   behavior is unchanged.

CPU-only; runs on the VM (worktree) or pod-side. CONTENT HYGIENE: logs carry
ids / counts / hashes only — never dataset row text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2222_lib as lib  # noqa: E402


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Subprocess with explicit env passthrough (spec: never implicit-inherit)."""
    import os

    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, env={**os.environ}, check=False
    )


def stage_subsamples(
    data_root: Path, ds_ids: list[str], tokenizer, *, seed: int, s_rows: int
) -> dict:
    """Build (or refresh) every requested dataset's subsample manifest."""
    out: dict[str, dict] = {}
    t0 = time.time()
    for k, ds in enumerate(ds_ids):
        manifest, _rows = lib.ensure_subsample(data_root, ds, tokenizer, seed=seed, s_rows=s_rows)
        out[ds] = {
            "split_hash": manifest["split_hash"],
            "n_admitted": manifest["n_admitted"],
            "n_file_rows": manifest["n_file_rows"],
            "n_rejected_budget": manifest["n_rejected_budget"],
            "n_rejected_empty": manifest["n_rejected_empty"],
        }
        print(
            f"[p0_subsample] unit {k + 1}/{len(ds_ids)} {ds} "
            f"n={manifest['n_admitted']} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return out


def _load_rb_tensor(path: Path):
    """Load one r_B .pt (raw tensor root — probed live at unit-1) + shape-assert."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not hasattr(obj, "shape"):
        raise RuntimeError(f"{path}: expected a raw tensor root, got {type(obj).__name__}")
    if tuple(obj.shape) != lib.RB_SHAPE:
        raise RuntimeError(f"{path}: shape {tuple(obj.shape)} != expected {lib.RB_SHAPE}")
    return obj


def stage_rb(data_root: Path) -> dict:
    """Stage #778 r_B — rb_v2 primary, rb v1 fallback as a NAMED deviation.

    The primary attempt's failure is caught (the plan-registered fallback,
    §7 kill-criterion 2 / §8 risk row), logged LOUD, and recorded in the
    sentinel; a fallback failure propagates (abort before any GPU provision).
    """
    from explore_persona_space.orchestrate import hub

    sources = (("rb_v2", lib.RB_V2_HUB_PREFIX), ("rb_v1_fallback", lib.RB_V1_HUB_PREFIX))
    deviation: str | None = None
    for source, prefix in sources:
        try:
            shapes: dict[str, list[int]] = {}
            for trait in lib.TRAITS:
                target = Path(data_root) / "rb" / source / f"{trait}.pt"
                hub.stage_hub_file(
                    lib.HF_DATA_REPO, f"{prefix}/{trait}.pt", target, repo_type="dataset"
                )
                shapes[trait] = list(_load_rb_tensor(target).shape)
            return {
                "source": source,
                "hub_prefix": prefix,
                "shapes": shapes,
                "deviation": deviation,
            }
        except Exception as e:  # noqa: BLE001 — plan-registered fallback; recorded, never silent
            if source == "rb_v1_fallback":
                raise
            deviation = (
                f"rb_v2 staging/shape check FAILED ({type(e).__name__}: {e}) — falling back to "
                "rb v1, a NAMED deviation (plan §8/§10 rb-fallback line): the direction object "
                "changes and H3 comparability shifts; carried as a clean-result scope caveat"
            )
            print(f"[p0_rb] DEVIATION: {deviation}", flush=True)
    raise AssertionError("unreachable")


def stage_maps(data_root: Path) -> dict:
    """Stage the #1739 frozen maps + run the mechanized realized-keys probe."""
    import numpy as np

    from explore_persona_space.orchestrate import hub

    out: dict[str, dict] = {}
    for fname in lib.MAP_FILES:
        target = Path(data_root) / "maps" / fname
        hub.stage_hub_file(
            lib.HF_DATA_REPO, f"{lib.MAP_HUB_PREFIX}/{fname}", target, repo_type="dataset"
        )
        proc = _run(
            [
                sys.executable,
                str(lib.REPO_ROOT / "scripts" / "verify_reused_artifact_keys.py"),
                "--artifact",
                str(target),
                "--keys",
                lib.MAP_KEYS,
            ]
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"verify_reused_artifact_keys FAILED for {target} (rc={proc.returncode}):\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        with np.load(target, allow_pickle=False) as z:
            w = z["w"]
            if w.shape != lib.MAP_W_SHAPE or w.dtype != np.float16:
                raise RuntimeError(f"{fname}: w shape/dtype {w.shape}/{w.dtype} != expected")
            for key in ("x_mu", "x_sd", "y_mu"):
                if z[key].shape != lib.MAP_MU_SHAPE:
                    raise RuntimeError(f"{fname}: {key} shape {z[key].shape} != expected")
            if z["layers"].shape != (28,):
                raise RuntimeError(f"{fname}: layers shape {z['layers'].shape} != (28,)")
        out[fname] = {
            "verify_keys": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "PASS",
            "w_shape": list(lib.MAP_W_SHAPE),
            "w_dtype": "float16",
        }
        print(f"[p0_maps] {fname} staged + verified", flush=True)
    return out


def locate_yaxis(repo_root: Path) -> dict:
    """Locate the #778 y-axis JSONs; sparse-checkout add when the cone is absent."""
    yaxis_dir = repo_root / "eval_results" / "issue_778"

    def _count() -> int:
        return len(list(yaxis_dir.glob("finetune_*.json"))) if yaxis_dir.exists() else 0

    n = _count()
    if n < 72:
        proc = _run(["git", "sparse-checkout", "add", "eval_results/issue_778"], cwd=repo_root)
        print(
            f"[p0_yaxis] sparse-checkout add rc={proc.returncode} {proc.stderr.strip()[:200]}",
            flush=True,
        )
        n = _count()
    if n < 72:
        raise RuntimeError(
            f"#778 y-axis incomplete: {n} finetune_*.json under {yaxis_dir} (expected >= 72); "
            "on a pod, add the cone via BOOTSTRAP_EXTRA_CONES or "
            "`git sparse-checkout add eval_results/issue_778`"
        )
    return {"dir": str(yaxis_dir), "n_finetune_json": n}


def _is_shallow_repo(repo_root: Path) -> bool:
    """True when the checkout has truncated ancestry (`git rev-parse --is-shallow-repository`)."""
    proc = _run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(f"shallowness probe failed: {proc.stderr.strip()}")
    return proc.stdout.strip() == "true"


def _leg_a_range_shas(repo_root: Path, branch: str, path: str) -> list[str]:
    """Non-merge branch-side SHAs of ``origin/main..origin/<branch> -- <path>`` (fail-loud)."""
    log = _run(
        ["git", "log", "--no-merges", "--format=%H", f"origin/main..origin/{branch}", "--", path],
        cwd=repo_root,
    )
    if log.returncode != 0:
        raise RuntimeError(f"leg-A diff errored for {branch}:{path}: {log.stderr.strip()}")
    return [s for s in log.stdout.split("\n") if s.strip()]


def _undeclared(shas: list[str], declared: tuple[str, ...]) -> list[str]:
    """SHAs not covered by any declared short-sha disposition prefix."""
    return [s for s in shas if not any(s.startswith(d) for d in declared)]


def _merged_into_main(repo_root: Path, sha: str) -> bool | None:
    """``git merge-base --is-ancestor <sha> origin/main`` verdict for one flagged SHA.

    Returns True (ancestor of main ⇒ MERGED), False (decided branch-side), or
    None when the clone's history cannot decide (rc>1: missing objects /
    truncated history on a shallow clone).
    """
    proc = _run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=repo_root)
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def _resolve_shallow_phantoms(
    repo_root: Path,
    branch: str,
    path: str,
    declared: tuple[str, ...],
    shas: list[str],
    unexpected: list[str],
) -> tuple[list[str], list[str], str]:
    """Drop shallow-range phantoms from a leg-A flagged set (#2222 pod crash fix).

    A depth-1 bootstrap clone truncates ``origin/main``'s ancestry, so the
    ``origin/main..origin/<branch>`` range mis-attributes already-MERGED main
    commits as branch-side edits and the gate halts every fresh pod. Bounded
    three-rung ladder: (1) post-filter every flagged SHA with
    ``git merge-base --is-ancestor <sha> origin/main`` — an ancestor of main is
    MERGED, not an undispositioned branch-side edit; (2) while flagged SHAs
    survive on a still-shallow/undecidable history, bounded deepen
    (``git fetch --deepen=200 origin main <branch>``) then re-run the range +
    post-filter; (3) same after ``git fetch --unshallow --filter=blob:none``
    (plain ``--unshallow`` when the filtered form is refused). Returns
    ``(final_shas, final_unexpected, mechanism)``; survivors are GENUINE
    undispositioned commits on a decidable history — the caller still fails
    loud on them. Full clones never enter this path (byte-unchanged gate).
    """
    n_flagged = len(unexpected)
    mechanism = "merge-base"
    for rung in ("merge-base", "deepen", "unshallow"):
        if rung == "deepen":
            proc = _run(["git", "fetch", "--deepen=200", "origin", "main", branch], cwd=repo_root)
            mechanism = "merge-base after --deepen=200"
            if proc.returncode != 0:
                print(
                    f"[p0_lineage] --deepen=200 failed (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[:200]}",
                    flush=True,
                )
        elif rung == "unshallow":
            if not _is_shallow_repo(repo_root):
                break  # history already complete; the previous rung's verdict is decidable
            proc = _run(
                ["git", "fetch", "--unshallow", "--filter=blob:none", "origin", "main", branch],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                proc = _run(
                    ["git", "fetch", "--unshallow", "origin", "main", branch], cwd=repo_root
                )
            mechanism = "merge-base after --unshallow"
            if proc.returncode != 0:
                print(
                    f"[p0_lineage] --unshallow failed (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[:200]}",
                    flush=True,
                )
        if rung != "merge-base":
            shas = _leg_a_range_shas(repo_root, branch, path)
            unexpected = _undeclared(shas, declared)
        survivors: list[str] = []
        n_undecidable = 0
        for sha in unexpected:
            verdict = _merged_into_main(repo_root, sha)
            if verdict is True:
                continue  # ancestor of origin/main ⇒ merged phantom, not a branch-side edit
            survivors.append(sha)
            n_undecidable += int(verdict is None)
        unexpected = survivors
        if not unexpected:
            break
        if n_undecidable == 0 and not _is_shallow_repo(repo_root):
            break  # decidable complete history — survivors are genuine
    lib.log_phase(
        "p0_lineage",
        f"shallow clone detected — post-filtered {n_flagged - len(unexpected)} phantom merged "
        f"SHAs via {mechanism}",
        key=f"{branch}:{path}",
        n_flagged_by_range=n_flagged,
        n_surviving=len(unexpected),
    )
    return shas, unexpected, mechanism


def reassert_parent_lineage(repo_root: Path) -> dict:
    """Re-run the plan §10 leg-A diffs; fail loud on any undispositioned commit.

    ``--no-merges`` — merge commits carry no branch-side edits (the plan's own
    reading). A branch that cannot be fetched AND has no local ref is recorded
    "unverifiable" (artifact-reuse (k): never masquerade an errored probe as an
    empty diff). SHALLOW-AWARE (#2222 crash fix): on a shallow clone (pod
    bootstrap ``--depth 1``) the range mis-attributes merged main commits as
    branch-side, so flagged SHAs route through ``_resolve_shallow_phantoms``
    (merge-base post-filter + bounded deepen ladder); full-clone behavior is
    byte-unchanged — the resolver never runs on complete history.
    """
    branches = sorted({br for br, _ in lib.DECLARED_LEG_A})
    fetch = _run(["git", "fetch", "origin", *branches], cwd=repo_root)
    if fetch.returncode != 0:
        print(f"[p0_lineage] git fetch failed (using existing refs): {fetch.stderr.strip()[:200]}")
    if _is_shallow_repo(repo_root):
        lib.log_phase(
            "p0_lineage",
            "shallow clone detected — merge-base post-filter armed for leg-A ranges",
        )
    results: dict[str, dict] = {}
    unexpected_total: list[str] = []
    shallow_resolution_ran = False
    for (branch, path), declared in lib.DECLARED_LEG_A.items():
        probe = _run(["git", "rev-parse", "--verify", f"origin/{branch}"], cwd=repo_root)
        key = f"{branch}:{path}"
        if probe.returncode != 0:
            results[key] = {"status": "branch unavailable — leg A unverifiable"}
            continue
        shas = _leg_a_range_shas(repo_root, branch, path)
        unexpected = _undeclared(shas, declared)
        results[key] = {"status": "ok", "branch_side_shas": shas, "declared": list(declared)}
        if unexpected and _is_shallow_repo(repo_root):
            shallow_resolution_ran = True
            shas, unexpected, mechanism = _resolve_shallow_phantoms(
                repo_root, branch, path, declared, shas, unexpected
            )
            results[key]["branch_side_shas"] = shas
            results[key]["shallow_resolution"] = mechanism
        if unexpected:
            unexpected_total.extend(f"{key}: {s}" for s in unexpected)
    if unexpected_total:
        msg = (
            "leg-A re-assertion FAILED — fresh branch-side commits with no plan disposition "
            "(artifact-reuse check (k)); inspect + disposition before any GPU provision:\n"
            + "\n".join(unexpected_total)
        )
        if shallow_resolution_ran:
            msg += (
                "\n(shallow-clone post-filter ran: surviving SHAs are undispositioned on the "
                "deepened/decidable history, not range phantoms)"
            )
        raise RuntimeError(msg)
    return results


def main() -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default=str(lib.default_data_root()))
    parser.add_argument("--datasets", nargs="*", default=None, help="families and/or dataset ids")
    parser.add_argument("--subsample", type=int, default=lib.SUBSAMPLE_ROWS)
    parser.add_argument("--seeds", default=str(lib.SUBSAMPLE_SEED), help="comma list; exactly one")
    parser.add_argument("--skip-hub-artifacts", action="store_true", help="skip rb + map staging")
    parser.add_argument("--skip-lineage", action="store_true")
    parser.add_argument("--skip-yaxis", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) != 1:
        raise SystemExit(f"--seeds must name exactly ONE seed (fixed-subsample design): {seeds}")
    seed = seeds[0]
    data_root = Path(args.data_root)
    ds_ids = lib.dataset_ids(args.datasets)

    sentinel: dict = {
        "phase": "P0",
        "data_root": str(data_root),
        "datasets": ds_ids,
        "seed": seed,
        "s_rows": args.subsample,
        "skipped": {
            "hub_artifacts": args.skip_hub_artifacts,
            "lineage": args.skip_lineage,
            "yaxis": args.skip_yaxis,
        },
    }

    sentinel["zip"] = lib.stage_dataset_zip(data_root)
    lib.log_phase("p0_zip", "dataset.zip staged + pin-verified", **sentinel["zip"])

    from explore_persona_space.experiments.issue_1739.generation import get_tokenizer

    tokenizer = get_tokenizer()
    sentinel["subsample"] = stage_subsamples(
        data_root, ds_ids, tokenizer, seed=seed, s_rows=args.subsample
    )

    if not args.skip_hub_artifacts:
        sentinel["rb"] = stage_rb(data_root)
        sentinel["maps"] = stage_maps(data_root)

    if not args.skip_yaxis:
        sentinel["yaxis"] = locate_yaxis(lib.REPO_ROOT)

    if not args.skip_lineage:
        sentinel["lineage"] = reassert_parent_lineage(lib.REPO_ROOT)

    sentinel["meta"] = lib.run_metadata()
    out_path = data_root / "staging_ready.json"
    lib.write_json_atomic(out_path, sentinel)
    lib.log_phase("staging_ready", "P0 complete", sentinel=str(out_path))
    print(json.dumps({k: v for k, v in sentinel.items() if k != "lineage"}, indent=1)[:2000])
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit before C-extension teardown (gotchas.md PyGILState race)


if __name__ == "__main__":
    main()
