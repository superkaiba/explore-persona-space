#!/usr/bin/env python
# ruff: noqa: RUF002
"""#1112 VM-side geometry driver (0 GPU-h; plan §4.5 / §9 "geometry + bootstrap (VM, CPU)").

Runs AFTER pod teardown against the persisted capture stores: discovers the
realized ``<cell>/<dose>/pooled.pt`` tree (never re-enumerates a registered
grid — the smoke's cell subset threads through by construction), loads the
r_B tensors, and calls ``experiments.issue_1112.geometry.run_geometry``
(batched Gram-space cluster bootstrap, paired cross-cell CIs, 80-row
sensitivity, split-half ceilings). Outputs:

- ``eval_results/issue_1112/geometry/geometry_per_cell.json`` (primary
  deliverable glob, plan §6.5)
- per-draw × per-layer bootstrap DV matrices under
  ``eval_results/issue_1112/geometry/bootstrap_matrices/`` (+ optional HF
  upload under ``issue1112_geometry2x2/analysis_tensors/bootstrap_matrices/``)

Input staging: ``--capture-root`` (a local tree, e.g. rsync'd from the pod or
the pod's own out_root) or ``--from-hf`` (scoped ``list_repo_tree`` on the
data-repo prefix + per-file ``hf_hub_download``, ≤6 workers — never
``snapshot_download`` against the ~1M-file data repo; gotchas.md).

Smoke (same code path, tiny knobs):
    uv run python scripts/issue1112_geometry.py --capture-root <tiny tree> \
        --rb-dir <dir with rb_*.pt> --out-dir /tmp/issue-1112-smoke/geometry \
        --n-boot 25
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.experiments import issue_1112 as C  # noqa: E402
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

logger = logging.getLogger("issue1112.geometry_driver")

BASE_CELLS = {"base_sycophancy": "sycophancy", "base_marker": "marker"}


def _behavior_for(cell: str) -> str:
    if cell in BASE_CELLS:
        return BASE_CELLS[cell]
    return "marker" if cell in C.MARKER_CELLS else "sycophancy"


def stage_from_hf(dest: Path, *, revision: str | None) -> None:
    """Stage ``analysis_tensors/{capture,rb}`` from the data repo (scoped)."""
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

    def _fetch(path_in_repo: str) -> Path:
        last: Exception | None = None
        for attempt in range(4):
            try:
                got = hf_hub_download(
                    C.HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision
                )
                break
            except Exception as e:  # bounded retry, linear backoff (gotchas.md)
                last = e
                time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"hf_hub_download failed 4x for {path_in_repo}") from last
        rel = Path(path_in_repo).relative_to(prefix)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(got, target)
        return target

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_fetch, p) for p in entries]
        for f in as_completed(futures):
            f.result()  # fail loud on the first exhausted retry
    logger.info("[stage] %d files staged under %s", len(entries), dest)


def discover_passes(capture_root: Path) -> list[tuple[str, str]]:
    """Realized (cell, dose) list from the on-disk tree (fail-loud when empty)."""
    passes = sorted(
        (p.parent.parent.name, p.parent.name) for p in capture_root.glob("*/*/pooled.pt")
    )
    if not passes:
        raise FileNotFoundError(f"no <cell>/<dose>/pooled.pt stores under {capture_root}")
    return passes


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="#1112 VM-side geometry pass (CPU, batched).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture-root", type=Path, help="local <cell>/<dose>/pooled.pt tree")
    src.add_argument(
        "--from-hf",
        action="store_true",
        help="stage analysis_tensors/{capture,rb} from the data repo (scoped listing)",
    )
    p.add_argument("--rb-dir", type=Path, default=None, help="dir holding rb_<behavior>.pt")
    p.add_argument("--revision", default=None, help="data-repo revision pin for --from-hf")
    p.add_argument(
        "--stage-dir",
        type=Path,
        default=Path(f"data/issue_{C.ISSUE}/geometry_stage"),
        help="--from-hf staging destination",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(f"eval_results/issue_{C.ISSUE}/geometry"),
        help="geometry_per_cell.json destination (smokes MUST divert to scratch)",
    )
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument(
        "--upload",
        action="store_true",
        help="upload geometry JSON + bootstrap matrices to the data repo",
    )
    args = p.parse_args(argv)

    if args.from_hf:
        stage_from_hf(args.stage_dir, revision=args.revision)
        capture_root = args.stage_dir / "capture"
        rb_dir = args.rb_dir or (args.stage_dir / "rb")
    else:
        capture_root = args.capture_root
        rb_dir = args.rb_dir
        if rb_dir is None:
            raise SystemExit("--rb-dir is required with --capture-root")

    passes = discover_passes(capture_root)
    cells_doses = [(c, d) for c, d in passes if c not in BASE_CELLS]
    behaviors = {_behavior_for(c) for c, _ in cells_doses}
    base_store_by_behavior: dict[str, Path] = {}
    for base_cell, behavior in BASE_CELLS.items():
        store = capture_root / base_cell / "base" / "pooled.pt"
        if behavior in behaviors:
            if not store.exists():
                raise FileNotFoundError(f"base panel missing for {behavior}: {store}")
            base_store_by_behavior[behavior] = store
    rb_by_behavior: dict[str, Path] = {}
    for behavior in behaviors:
        rb_path = rb_dir / f"rb_{behavior}.pt"
        if not rb_path.exists():
            raise FileNotFoundError(f"r_B tensor missing for {behavior}: {rb_path}")
        rb_by_behavior[behavior] = rb_path

    logger.info("[geometry] %d capture passes (%s)", len(cells_doses), sorted(behaviors))
    payload = geo.run_geometry(
        capture_root,
        args.out_dir,
        cells_doses=cells_doses,
        base_store_by_behavior=base_store_by_behavior,
        behavior_by_cell={c: _behavior_for(c) for c, _ in cells_doses},
        selected_dose_by_cell={c: "selected" for c, _ in cells_doses},
        rb_by_behavior=rb_by_behavior,
        n_boot=args.n_boot,
    )
    logger.info("[geometry] %d records written to %s", len(payload["records"]), args.out_dir)

    if args.upload:
        from explore_persona_space.orchestrate import hub

        url = hub._upload(
            args.out_dir / "geometry_per_cell.json",
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/geometry/geometry_per_cell.json",
            upload_as_file=True,
        )
        if not str(url):
            raise RuntimeError("geometry_per_cell.json upload returned no path")
        url = hub._upload(
            args.out_dir / "bootstrap_matrices",
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/analysis_tensors/bootstrap_matrices",
        )
        if not str(url):
            raise RuntimeError("bootstrap_matrices upload returned no path")
        logger.info("[geometry] uploads complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
