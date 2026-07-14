"""Offline replay of the fu3 datagen positive-arm yield accounting (#1090 crash-fix 2).

Re-runs the PRODUCTION keep accounting (``datagen._judge_and_filter`` with the
pure-read ``judge_result_from_save_raw`` reduce — zero API calls, zero GPU) on
the launch-3 datagen sidecars uploaded to the HF data repo, and prints COUNTS
ONLY (never candidate/judge text): per cell — requested, judgeable, kept,
keep-rate, floor, and the break-even oversample mult. Grounds the Bug-2 mult
recalibration in measured production pass rates (on-policy-completions rule:
"ground the new mult in the measured production per-variant pass rates").

Usage:  uv run python scripts/issue1090_fu3_yield_replay.py [--cells slug1,slug2]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts siblings

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.artifacts import datagen  # noqa: E402
from explore_persona_space.eval.graded_judge import judge_result_from_save_raw  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1090_fu3"


def _resolve_behavior(name: str):
    """Resolve a behavior name exactly as the fu3 worker does (run1090 registry
    first — it carries the fu3 additions — falling back to the library registry)."""
    try:
        import issue1090_run as run1090

        return run1090.BEHAVIORS[name]
    except Exception:
        from explore_persona_space.artifacts.behavior import BEHAVIORS

        return BEHAVIORS[name]


def _replay_judge_fn(judge_raw_path: Path):
    """A JudgeFn that reduces the persisted per-draw scores (pure read)."""

    def fn(items, rubric, **kw):
        return judge_result_from_save_raw(judge_raw_path, items)

    return fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default=None, help="comma-separated cell slugs (default: all)")
    args = ap.parse_args()

    api = HfApi()
    paths = [
        e.path
        for e in api.list_repo_tree(
            DATA_REPO, path_in_repo=PREFIX, repo_type="dataset", recursive=True
        )
    ]
    cells = sorted({p.split("/")[1] for p in paths if p.endswith("/datagen/judge_raw_pos.json")})
    if args.cells:
        want = set(args.cells.split(","))
        cells = [c for c in cells if c in want]

    rows = []
    for slug in cells:
        base = f"{PREFIX}/{slug}/datagen"
        try:
            raw_pos = Path(hf_hub_download(DATA_REPO, f"{base}/raw_pos.jsonl", repo_type="dataset"))
            judge_raw = Path(
                hf_hub_download(DATA_REPO, f"{base}/judge_raw_pos.json", repo_type="dataset")
            )
            manifest = json.loads(
                Path(
                    hf_hub_download(DATA_REPO, f"{base}/gen_manifest.json", repo_type="dataset")
                ).read_text()
            )
        except Exception as e:  # sidecar set incomplete for this cell
            print(f"{slug}: SKIP ({type(e).__name__})")
            continue
        behavior = _resolve_behavior(manifest["behavior"])
        cands = datagen._read_raw(raw_pos)
        with tempfile.TemporaryDirectory() as td:
            kept, drops, _jr, _scores = datagen._judge_and_filter(
                behavior,
                cands,
                datagen.POSITIVE,
                judge_fn=_replay_judge_fn(judge_raw),
                n_judge_draws=manifest["n_judge_draws"],
                cache_dir=Path(td) / "cache",
                save_raw=Path(td) / "replay_raw.json",
            )
        floor_n = math.ceil(manifest["quota_floor"] * manifest["target_n"])
        mult = float(manifest.get("oversample_mult", 1.0))
        keep_rate = len(kept) / max(1, drops.requested)
        breakeven = mult * floor_n / max(1, len(kept))
        rows.append((slug, manifest["behavior"], drops.requested, drops.generated, len(kept)))
        kept_by_variant = dict(drops.variant_kept)
        print(
            f"{slug}: behavior={manifest['behavior']} mult={mult} "
            f"requested={drops.requested} judgeable={drops.generated} kept={len(kept)} "
            f"keep_rate={keep_rate:.3f} floor_n={floor_n} "
            f"{'MISS' if len(kept) < floor_n else 'ok'} breakeven_mult={breakeven:.2f} "
            f"kept_by_variant={kept_by_variant}"
        )

    if rows:
        rates = sorted(k / max(1, r) for _, _, r, _, k in rows)
        print(
            f"\n{len(rows)} cells: keep_rate min={rates[0]:.3f} "
            f"p25={rates[len(rates) // 4]:.3f} median={rates[len(rates) // 2]:.3f} "
            f"max={rates[-1]:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
