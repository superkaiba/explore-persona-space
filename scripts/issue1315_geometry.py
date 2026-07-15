#!/usr/bin/env python
"""#1315 VM-side geometry aggregator — thin parametrization of the #1112 rig.

Runs AFTER pod teardown on the VM CPU (plan §4.6/§9: batched Gram-eigh
spectral path, <8 GB RSS): downloads (or reads) the pooled capture stores +
the impolite r_B, then calls ``experiments.issue_1112.geometry.run_geometry``
over the #1315 cell table — once for the own-text tree, once for the
shared-text (capture_tf) tree. All DV definitions, bootstrap conventions
(n_boot 1000/2000 seed 653), and the exemplar-calibration guard are the
#1112/#653 machinery verbatim; this script only supplies the cell maps.

Usage (VM, after the pod run):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 uv run python scripts/issue1315_geometry.py \\
        --from-hf --out-dir eval_results/issue_1315/geometry

    # or against a local pod-synced tree:
    ... issue1315_geometry.py --capture-root data/issue_1315/run/capture \\
        --tf-root data/issue_1315/run/capture_tf --rb-dir data/issue_1315/run/rb
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue1112_geometry import discover_passes  # noqa: E402

from explore_persona_space.experiments import issue_1315 as C  # noqa: E402
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

REPO_ROOT = _SCRIPTS_DIR.parent


def stage_from_hf(dest: Path, *, revision: str | None) -> None:
    """Stage pooled capture stores + rb tensor from the data repo (scoped
    ``list_repo_tree`` + per-file ``hf_hub_download`` — never
    ``snapshot_download`` on the ~1M-file repo; gotchas.md #833)."""
    import shutil

    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    api = HfApi()
    prefix = f"{C.DATA_PREFIX}/analysis_tensors"
    # retried scoped listing (hub helper) — a bare list_repo_tree is the #920
    # false-failure class (workflow_lint --check-hub-verify-retry)
    entries = list_hf_files_under_path(
        api, C.HF_DATA_REPO, prefix, repo_type="dataset", revision=revision
    )
    if not entries:
        raise FileNotFoundError(f"no files under {C.HF_DATA_REPO}/{prefix}")
    import time

    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    for p in entries:
        rel = Path(p).relative_to(prefix)
        target = dest / rel
        if target.exists():
            continue
        # Bounded TRANSPORT retry (HF 429/5xx — the fleet-level rate-limit
        # pressure that killed this run's pod-side p11 twice on 2026-07-15);
        # fail-loud after exhaustion. Staging is resume-idempotent.
        for attempt in range(4):
            try:
                got = hf_hub_download(C.HF_DATA_REPO, p, repo_type="dataset", revision=revision)
                break
            except (HfHubHTTPError, LocalEntryNotFoundError) as e:
                if attempt == 3:
                    raise
                wait = 30 * 2**attempt
                print(f"[stage] transport error on {p} ({e}); retry in {wait}s", flush=True)
                time.sleep(wait)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(got, target)


PARITY_KILL_MEDIAN_COS = 0.99  # plan §Kill: <0.99 median per-row cosine → capture bug


def run_tf_parity_gate(tf_root: Path, capture_root: Path, out_dir: Path) -> dict:
    """Plan §4.5 prompt-arm parity read + the plan §Kill gate (ported from the
    #1112 rig's ``_parity_check``; concern tf-shared-parity-warn-check-not-
    ported). Per tf-captured cell: the shared-text store's prefix/context arms
    vs the SAME selected-dose model's own-text store — the two capture rounds
    share prompt tokens, so under causal attention per-row cosine >= 0.999 up
    to bf16 batch-composition jitter (WARN bar, persisted + adjudicated).
    KILL: any cell whose MEDIAN per-row cosine over all prompt (arm, layer)
    pairs is < 0.99 raises BEFORE any geometry verdict is written (plan kill
    criterion: capture pipeline bug — no partial reads). Rows are re-paired by
    (context_id, question_idx) via ``_reorder_store`` (set mismatch raises).
    """
    import numpy as np
    from issue1112_geometry import _parity_check, _reorder_store, _store_keys

    from explore_persona_space.experiments.issue_1112 import geometry as geo

    results: dict[str, dict] = {}
    for tf_pooled in sorted(Path(tf_root).glob("*/selected/pooled.pt")):
        cell = tf_pooled.parent.parent.name
        own_path = Path(capture_root) / cell / "selected" / "pooled.pt"
        assert own_path.exists(), f"missing own-text store for parity: {own_path}"
        tf_store = geo.load_store(tf_pooled)
        missing_arms = [a for a in ("prefix", "context") if a not in tf_store["arms"]]
        if missing_arms:
            raise RuntimeError(
                f"tf store {tf_pooled} lacks prompt arms {missing_arms} — re-run "
                "run_capture_tf_unit (all-arm capture); the plan §4.5 parity read "
                "is unrunnable without them"
            )
        own_store = _reorder_store(geo.load_store(own_path), _store_keys(tf_store))
        layers = sorted(tf_store["arms"]["prefix"].keys())
        summary, tensors = _parity_check(tf_store, own_store, layers)
        all_cos = np.concatenate([tensors[k] for k in sorted(tensors)])
        summary["median_per_row_cos"] = float(np.median(all_cos))
        summary["kill_bar"] = PARITY_KILL_MEDIAN_COS
        results[cell] = summary
    if not results:
        raise FileNotFoundError(f"no <cell>/selected/pooled.pt tf stores under {tf_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tf_parity_check.json").write_text(json.dumps(results, indent=1))
    killed = {
        c: s["median_per_row_cos"]
        for c, s in results.items()
        if s["median_per_row_cos"] < PARITY_KILL_MEDIAN_COS
    }
    if killed:
        raise RuntimeError(
            f"prompt-arm parity KILL (median per-row cosine < {PARITY_KILL_MEDIAN_COS}): "
            f"{killed} — capture pipeline bug; fix before any geometry verdict "
            "(plan kill criterion, no partial reads)"
        )
    return results


def _cell_maps(cells_doses: list[tuple[str, str]], base_store: Path, rb_path: Path) -> dict:
    cells = sorted({c for c, _ in cells_doses if c != "base"})
    unknown = [
        c for c in cells if c not in (*C.REUSED_LORA_CELLS, *C.FT_CELLS, C.CONDITIONAL_BARE_CELL)
    ]
    if unknown:
        raise ValueError(f"unregistered cells in the capture tree: {unknown}")
    return {
        "cells_doses": [(c, d) for c, d in cells_doses if c != "base"],
        "base_store_by_behavior": {C.BEHAVIOR: base_store},
        "behavior_by_cell": {c: C.BEHAVIOR for c in cells},
        "selected_dose_by_cell": {c: "selected" for c in cells},
        "rb_by_behavior": {C.BEHAVIOR: rb_path},
    }


def _source_context(cell: str) -> str:
    """Registered source context id for a cell (panel-group key)."""
    if cell in C.REUSED_LORA_CELLS:
        return C.REUSED_LORA_CELLS[cell]["context_id"]
    if cell in C.FT_CELLS:
        return C.FT_CELLS[cell]["context_id"]
    raise ValueError(f"unregistered cell: {cell}")


def _subset_base_store(base: dict, key_order: list[tuple[str, int]], out_path: Path) -> Path:
    """Write a base-store copy whose rows are SELECTED + ORDERED to key_order.

    #1315 divergence from #1112: the base pass captures the UNION panel (8
    contexts x 20 q = 160 rows) while each cell's panel is a 120-row subset
    (5 shared negatives + its own source context), so the #1112 rig's exact
    row_meta equality assert (``delta_cloud``) needs a per-panel-group base
    subset. Asserts every requested key exists in the base store (a missing
    key = capture bug, halt) and preserves the trained stores' row order so
    ``delta_cloud``'s kt == kb identity check passes unchanged.
    """
    import torch

    keys = [(m["context_id"], int(m["question_idx"])) for m in base["row_meta"]]
    pos = {k: i for i, k in enumerate(keys)}
    missing = [k for k in key_order if k not in pos]
    assert not missing, (
        f"{len(missing)} panel keys absent from the union base store "
        f"(first: {missing[:3]}) — capture stores are not probe-aligned"
    )
    perm = [pos[k] for k in key_order]
    sub = dict(base)
    sub["row_meta"] = [base["row_meta"][i] for i in perm]
    sub["arms"] = {
        arm: {li: t[perm] for li, t in per_layer.items()} for arm, per_layer in base["arms"].items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sub, out_path)
    return out_path


def _run_tree_grouped(
    tree_root: Path,
    out_dir: Path,
    *,
    base_store_path: Path,
    rb_path: Path,
    n_boot: int,
    arms: tuple[str, ...] | None,
    subset_dir: Path,
    tag: str,
) -> dict:
    """run_geometry per panel group (cells sharing one source context), then
    merge the payloads. Registered DIFF_PAIRS all live inside the ICL group,
    so per-group runs preserve every paired read; the shared-per-group base
    subset keeps the #1112 paired-bootstrap convention (one index matrix per
    panel) intact."""
    from issue1112_geometry import _store_keys

    base = geo.load_store(base_store_path)
    cells_doses = [(c, d) for c, d in discover_passes(tree_root) if c != "base"]
    groups: dict[str, list[tuple[str, str]]] = {}
    for c, d in cells_doses:
        groups.setdefault(_source_context(c), []).append((c, d))

    merged: dict = {}
    for ctx in sorted(groups):
        cd = groups[ctx]
        group_out = out_dir / f"_group_{ctx}"
        done = group_out / "geometry_per_cell.json"
        if done.exists():
            # Group-level resume (checkpoint-per-phase; the 2026-07-15 earlyoom
            # SIGTERM killed the one-process all-groups run at ~20 GB RSS —
            # completed groups reload instead of recomputing).
            payload = json.loads(done.read_text())
            assert payload.get("n_boot") == n_boot, (ctx, payload.get("n_boot"), n_boot)
            logger_print = f"[geometry-grouped] resume: group {ctx} loaded from {done}"
            print(logger_print, flush=True)
        else:
            key_order = None
            for c, d in cd:
                st = geo.load_store(tree_root / c / d / "pooled.pt")
                k = _store_keys(st)
                if key_order is None:
                    key_order = k
                else:
                    assert k == key_order, (
                        f"row order differs within panel group {ctx}: {c}/{d} — "
                        "capture stores are not probe-aligned"
                    )
                del st
            sub_path = _subset_base_store(base, key_order, subset_dir / f"base_{tag}_{ctx}.pt")
            maps = _cell_maps(cd, sub_path, rb_path)
            cells_in_group = {c for c, _ in cd}
            pairs = tuple(
                p for p in C.DIFF_PAIRS if p[1] in cells_in_group and p[2] in cells_in_group
            )
            kwargs = dict(
                n_boot=n_boot,
                tensors_out=out_dir / "bootstrap_matrices",
                diff_pairs=pairs,
                **maps,
            )
            if arms is not None:
                kwargs["arms"] = arms
            payload = geo.run_geometry(tree_root, group_out, **kwargs)
        if not merged:
            merged = {k: v for k, v in payload.items()}
            merged["panel_groups"] = {ctx: [list(x) for x in cd]}
        else:
            merged["records"].update(payload["records"])
            merged["cross_cell_diffs"].update(payload["cross_cell_diffs"])
            merged["subsample_sensitivity_80row"].update(payload["subsample_sensitivity_80row"])
            merged["split_half_self_cosine_ceiling"].update(
                payload["split_half_self_cosine_ceiling"]
            )
            if payload.get("h3_interaction"):
                merged["h3_interaction"] = payload["h3_interaction"]
            merged["panel_groups"][ctx] = [list(x) for x in cd]
    return merged


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="#1315 VM-side geometry aggregator")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture-root", type=Path, help="local <cell>/<dose>/pooled.pt tree")
    src.add_argument("--from-hf", action="store_true", help="stage analysis_tensors/ from HF")
    p.add_argument("--tf-root", type=Path, default=None, help="local capture_tf tree")
    p.add_argument("--rb-dir", type=Path, default=None, help="dir holding rb_impolite.pt")
    p.add_argument("--revision", default=None, help="data-repo revision pin for --from-hf")
    p.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "eval_results" / "issue_1315" / "geometry"
    )
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args(argv)

    if args.from_hf:
        stage_root = REPO_ROOT / "data" / f"issue_{C.ISSUE}" / "hf_dl" / "analysis_tensors"
        stage_from_hf(stage_root, revision=args.revision)
        capture_root = stage_root / "capture"
        tf_root = stage_root / "capture_tf"
        rb_dir = stage_root / "rb"
    else:
        capture_root = args.capture_root
        tf_root = args.tf_root
        rb_dir = args.rb_dir
    assert capture_root is not None and capture_root.exists(), capture_root
    rb_path = (rb_dir or capture_root.parent / "rb") / f"rb_{C.BEHAVIOR}.pt"
    assert rb_path.exists(), f"missing r_B tensor at {rb_path} (run p7_rb / stage it first)"

    base_store = capture_root / "base" / "base" / "pooled.pt"
    assert base_store.exists(), f"missing base pooled store at {base_store}"

    # Plan §Kill gate FIRST: the prompt-arm parity read must pass before any
    # geometry verdict is written (no partial reads).
    parity = None
    if tf_root is not None and Path(tf_root).exists():
        parity = run_tf_parity_gate(Path(tf_root), capture_root, args.out_dir / "tf_shared")

    subset_dir = capture_root.parent / "base_subsets"
    payload = _run_tree_grouped(
        capture_root,
        args.out_dir,
        base_store_path=base_store,
        rb_path=rb_path,
        n_boot=args.n_boot,
        arms=None,  # own-text stores carry all three arms (module default)
        subset_dir=subset_dir,
        tag="own",
    )
    (args.out_dir / "geometry_per_cell.json").write_text(json.dumps(payload, indent=1, default=str))

    tf_payload = None
    if tf_root is not None and Path(tf_root).exists():
        tf_payload = _run_tree_grouped(
            Path(tf_root),
            args.out_dir / "tf_shared",
            base_store_path=base_store,
            rb_path=rb_path,
            n_boot=args.n_boot,
            arms=("response",),  # shared-text geometry reads the response arm only
            subset_dir=subset_dir,
            tag="tf",
        )
        (args.out_dir / "geometry_tf_shared.json").write_text(
            json.dumps(tf_payload, indent=1, default=str)
        )
    print(
        json.dumps(
            {
                "n_records_own": len(payload["records"]),
                "n_records_tf": len(tf_payload["records"]) if tf_payload else 0,
                "n_parity_cells": len(parity) if parity else 0,
                "out_dir": str(args.out_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
