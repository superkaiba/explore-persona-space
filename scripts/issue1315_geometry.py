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

    api = HfApi()
    prefix = f"{C.DATA_PREFIX}/analysis_tensors"
    entries = [
        e.path
        for e in api.list_repo_tree(
            C.HF_DATA_REPO,
            path_in_repo=prefix,
            repo_type="dataset",
            recursive=True,
            revision=revision,
        )
        if getattr(e, "size", None) is not None
    ]
    if not entries:
        raise FileNotFoundError(f"no files under {C.HF_DATA_REPO}/{prefix}")
    for p in entries:
        rel = Path(p).relative_to(prefix)
        target = dest / rel
        if target.exists():
            continue
        got = hf_hub_download(C.HF_DATA_REPO, p, repo_type="dataset", revision=revision)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(got, target)


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

    own = _cell_maps(discover_passes(capture_root), base_store, rb_path)
    payload = geo.run_geometry(
        capture_root,
        args.out_dir,
        n_boot=args.n_boot,
        tensors_out=args.out_dir / "bootstrap_matrices",
        diff_pairs=C.DIFF_PAIRS,
        **own,
    )
    (args.out_dir / "geometry_per_cell.json").write_text(json.dumps(payload, indent=1, default=str))

    tf_payload = None
    if tf_root is not None and Path(tf_root).exists():
        tf = _cell_maps(discover_passes(Path(tf_root)), base_store, rb_path)
        tf_payload = geo.run_geometry(
            Path(tf_root),
            args.out_dir / "tf_shared",
            n_boot=args.n_boot,
            tensors_out=args.out_dir / "tf_shared" / "bootstrap_matrices",
            arms=("response",),  # shared-text stores carry the response arm only
            diff_pairs=C.DIFF_PAIRS,
            **tf,
        )
        (args.out_dir / "geometry_tf_shared.json").write_text(
            json.dumps(tf_payload, indent=1, default=str)
        )
    print(
        json.dumps(
            {
                "n_records_own": len(payload["records"]),
                "n_records_tf": len(tf_payload["records"]) if tf_payload else 0,
                "out_dir": str(args.out_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
