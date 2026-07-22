#!/usr/bin/env python
# ruff: noqa: RUF002  # em-dash intentional
"""#1586 VM-side geometry aggregator — multi-behavior extension of the
#1315/#1112 rig (plan §4.8 p10).

Runs AFTER pod teardown on the VM CPU: reads the pooled capture stores
(``capture/<arm>/pooled.pt`` + per-behavior ``capture/base_<beh>/pooled.pt``
+ the shared-text ``capture_tf/<arm>/pooled.pt`` tree), then calls
``experiments.issue_1112.geometry.run_geometry`` ONCE PER BEHAVIOR over the
(method × regime × seed) arms — all DV definitions, BATCHED bootstrap
conventions (n_boot 1000/2000 seed 653; one vectorized index matrix per DV,
no serial per-draw loops — vectorize-many-cell-fits), the half-draw cosine
machinery (m=60, 2000 draws, seed 1112), and the exemplar-calibration guard
are the #1112/#653 machinery verbatim. This script only supplies the
(behavior × regime × seed) cell maps + the 16 method-paired diff pairs.

The exploratory read-out-direction leg (cos(μ, r_B)) runs only for behaviors
whose committed r_B resolves under ``--rb-dir`` (plan §4.6: first descope
priority); an absent behavior gets a zeros r_B + an ``rb_absent`` flag so its
cos rows read as non-finite placeholders, never fabricated signal.

Usage (VM, after the pod run):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python \\
        scripts/issue1586_geometry.py --capture-root data/issue_1586/out/capture \\
        --tf-root data/issue_1586/out/capture_tf \\
        --out-dir eval_results/issue_1586/geometry
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

import issue1586_cells as G  # noqa: E402

from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

REPO_ROOT = _SCRIPTS_DIR.parent
HIDDEN = 3584  # Qwen-2.5-7B residual width (zeros-r_B placeholder shape)


def discover_arms(capture_root: Path) -> dict[str, list[str]]:
    """{beh_key: [arm dirs with pooled.pt]} — base_* passes excluded."""
    out: dict[str, list[str]] = {}
    for p in sorted(capture_root.glob("*/pooled.pt")):
        arm = p.parent.name
        if arm.startswith("base_"):
            continue
        out.setdefault(arm.split("-")[0], []).append(arm)
    return out


def diff_pairs_for(beh_key: str, arms: list[str]) -> tuple[tuple[str, str, str], ...]:
    """Method-paired (label, ft_arm, lora_arm) diffs — the 16 registered
    contrasts (plan §5), restricted to arms present in the tree."""
    pairs = []
    present = set(arms)
    for regime in G.REGIMES:
        for seed in G.SEEDS:
            ft = G.ft_cell_id(beh_key, regime, seed)
            lora = f"{beh_key}-pers-lora-{regime}-s{seed}"
            if ft in present and lora in present:
                pairs.append((f"{ft}__ft_vs_lora", ft, lora))
    return tuple(pairs)


def _rb_path(rb_dir: Path | None, behavior: str, work: Path) -> tuple[Path, bool]:
    """Committed r_B where staged; else a flagged zeros placeholder (the
    exploratory leg's registered drop — never fabricated signal)."""
    if rb_dir is not None:
        cand = rb_dir / f"rb_{behavior}.pt"
        if cand.exists():
            return cand, False
    import torch

    ph = work / f"rb_zeros_{behavior}.pt"
    if not ph.exists():
        ph.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rb": torch.zeros(G.N_LAYERS, HIDDEN), "rb_absent": True}, ph)
    return ph, True


def run_behavior(
    beh_key: str,
    arms: list[str],
    *,
    capture_root: Path,
    base_store: Path,
    out_dir: Path,
    rb_dir: Path | None,
    n_boot: int,
    arms_filter: tuple[str, ...] | None,
    tag: str,
) -> dict:
    behavior = G.BEHAVIOR_BY_KEY[beh_key]
    assert base_store.exists(), f"missing base store for {beh_key}: {base_store}"
    rb, rb_absent = _rb_path(rb_dir, behavior, out_dir / "_rb_placeholders")
    group_out = out_dir / f"_beh_{beh_key}_{tag}"
    done = group_out / "geometry_per_cell.json"
    if done.exists():
        payload = json.loads(done.read_text())
        assert payload.get("n_boot", n_boot) == n_boot, (beh_key, n_boot)
        print(f"[geometry-1586] resume: {beh_key}/{tag} loaded from {done}", flush=True)
        return payload
    kwargs = dict(
        cells_doses=[(a, "selected") for a in arms],
        base_store_by_behavior={behavior: base_store},
        behavior_by_cell={a: behavior for a in arms},
        selected_dose_by_cell={a: "selected" for a in arms},
        rb_by_behavior={behavior: rb},
        n_boot=n_boot,
        tensors_out=out_dir / "bootstrap_matrices",
        diff_pairs=diff_pairs_for(beh_key, arms),
    )
    if arms_filter is not None:
        kwargs["arms"] = arms_filter
    payload = geo.run_geometry(capture_root, group_out, **kwargs)
    payload["rb_absent"] = rb_absent
    return payload


def _flat_store_tree(capture_root: Path, work: Path) -> Path:
    """run_geometry expects <cell>/<dose>/pooled.pt; the #1586 dispatcher
    writes <arm>/pooled.pt. Mirror via symlinks into <arm>/selected/."""
    dest = work / "tree"
    for p in sorted(capture_root.glob("*/pooled.pt")):
        d = dest / p.parent.name / "selected"
        d.mkdir(parents=True, exist_ok=True)
        link = d / "pooled.pt"
        if not link.exists():
            link.symlink_to(p.resolve())
        raw = p.parent / "raw_rows.json"
        raw_link = d / "raw_rows.json"
        if raw.exists() and not raw_link.exists():
            raw_link.symlink_to(raw.resolve())
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="#1586 VM-side geometry aggregator")
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--tf-root", type=Path, default=None)
    p.add_argument("--rb-dir", type=Path, default=None)
    p.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "eval_results" / "issue_1586" / "geometry"
    )
    p.add_argument("--n-boot", type=int, default=G.N_BOOT)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    work = args.out_dir / "_work"
    own_tree = _flat_store_tree(args.capture_root, work / "own")
    by_beh = discover_arms(args.capture_root)
    assert by_beh, f"no arm stores under {args.capture_root}"
    merged: dict = {"records": {}, "cross_cell_diffs": {}, "by_behavior": {}}
    for beh_key, arms in sorted(by_beh.items()):
        payload = run_behavior(
            beh_key,
            arms,
            capture_root=own_tree,
            base_store=own_tree / f"base_{beh_key}" / "selected" / "pooled.pt",
            out_dir=args.out_dir,
            rb_dir=args.rb_dir,
            n_boot=args.n_boot,
            arms_filter=None,
            tag="own",
        )
        merged["records"].update(payload.get("records", {}))
        merged["cross_cell_diffs"].update(payload.get("cross_cell_diffs", {}))
        merged["by_behavior"][beh_key] = {
            "arms": arms,
            "rb_absent": payload.get("rb_absent"),
            "diff_pairs": [list(t) for t in diff_pairs_for(beh_key, arms)],
        }
    (args.out_dir / "geometry_per_cell.json").write_text(json.dumps(merged, indent=1, default=str))

    if args.tf_root is not None and Path(args.tf_root).exists():
        tf_tree = _flat_store_tree(Path(args.tf_root), work / "tf")
        tf_merged: dict = {"records": {}, "cross_cell_diffs": {}, "by_behavior": {}}
        for beh_key, arms in sorted(discover_arms(Path(args.tf_root)).items()):
            payload = run_behavior(
                beh_key,
                arms,
                capture_root=tf_tree,
                # tf tree carries no base pass — the own-text base store is
                # the shared-text baseline by construction (shared text IS
                # the base generation).
                base_store=own_tree / f"base_{beh_key}" / "selected" / "pooled.pt",
                out_dir=args.out_dir / "tf_shared",
                rb_dir=args.rb_dir,
                n_boot=args.n_boot,
                arms_filter=("response",),  # shared-text = response arm only
                tag="tf",
            )
            tf_merged["records"].update(payload.get("records", {}))
            tf_merged["cross_cell_diffs"].update(payload.get("cross_cell_diffs", {}))
            tf_merged["by_behavior"][beh_key] = {"arms": arms}
        (args.out_dir / "tf_shared").mkdir(parents=True, exist_ok=True)
        (args.out_dir / "tf_shared" / "geometry_per_cell.json").write_text(
            json.dumps(tf_merged, indent=1, default=str)
        )
    print(f"[geometry-1586] wrote {args.out_dir / 'geometry_per_cell.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
