"""Stage one #1336 ladder surface's turnstore stems for an ARBITRARY model set.

Why this exists. ``issue1336_selfmap_missing_pairs.py --stage`` already stages exactly
the stems its ``--cells`` consume, but ``--cells`` is hard-asserted against that
script's 32-cell registry (base self map + the three round-B pairs), whose model set is
{base, sft, rlvr, rlvr_long}. The rigid-decomposition round runs a SECOND leg —
``issue1336_metric_ladder.py`` over the seven baseline pairs — which also needs ``dpo``.
Staging twice would pay the surface's bytes twice (up to ~70 GB), so this thin wrapper
stages the UNION model set once, per surface, ahead of both legs.

No new staging logic: it builds one ``(m, m, fmt, corpus)`` pseudo-cell per requested
model and hands them to ``issue1336_selfmap_missing_pairs.stage_inputs`` verbatim — the
same scoped ``list_repo_tree`` + retried ``hf_hub_download`` path, the same
``turnstore_v2/`` + ``turnstore_wave1/`` + ``gen/<model>/<corpus>/`` layout, the same
already-staged skip check, and the same wave-1 concat-source resolution. ``stage_inputs``
keys on the union of each cell's ``(source, target)``, so a self-pair cell stages exactly
that one model's stem; it is a staging key here, never a fit.

Example (one surface, the five-model union the rigid round needs):
    uv run python scripts/issue1336_stage_surface.py \\
        --models base,sft,dpo,rlvr,rlvr_long --format chat --corpus lmsys23k \\
        --stage-root /workspace/data/issue_1336
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps are frozen at torch/BLAS import, and the import below pulls the
# fit stack in transitively — load_dotenv() must land first.
load_dotenv()

import issue1336_selfmap_missing_pairs as sm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--models",
        required=True,
        help="comma list of checkpoint keys to stage (e.g. base,sft,dpo,rlvr,rlvr_long)",
    )
    ap.add_argument("--format", required=True, choices=("chat", "naturalistic"))
    ap.add_argument("--corpus", required=True, help="v2 corpus key (e.g. lmsys23k)")
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_1336"),
        help="root for staged inputs (turnstore_v2/, turnstore_wave1/, gen/ live under it)",
    )
    ap.add_argument("--turnstore-dir", type=Path, default=None)
    ap.add_argument("--wave1-turnstore-dir", type=Path, default=None)
    ap.add_argument("--gen-root", type=Path, default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    assert models, "--models resolved to an empty set"

    ns = SimpleNamespace(
        stage_root=args.stage_root,
        turnstore_dir=args.turnstore_dir or (args.stage_root / "turnstore_v2"),
        wave1_turnstore_dir=args.wave1_turnstore_dir or (args.stage_root / "turnstore_wave1"),
        gen_root=args.gen_root or (args.stage_root / "gen"),
    )

    # One self-pair per model: stage_inputs keys on the union of (source, target), so
    # this resolves to exactly one stem per model for this (format, corpus) surface.
    cells = [(m, m, args.format, args.corpus) for m in models]
    print(
        f"[stage-surface] {args.format}/{args.corpus} models={','.join(models)} "
        f"-> ts={ns.turnstore_dir} wave1={ns.wave1_turnstore_dir} gen={ns.gen_root}",
        flush=True,
    )
    sm.stage_inputs(cells, ns)
    print(f"[stage-surface] {args.format}/{args.corpus} DONE", flush=True)


if __name__ == "__main__":
    main()
