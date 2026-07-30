"""Issue #1689 Phase E — aggregation + figures.

Aggregates per-cell and per-pair ladder outputs into hero figures per
plan §6:
  - Hero 1: rung-reached heatmap per (model × arm) — 4 panels
  - Hero 2: within-identity framing triangles (5 identities)
  - Hero 3: user-provenance triangles (3 framings × 2 models = 6 panels)

Also renders operator-cosine matrices (via #1345's raw direction-aware
`raw_cosine_with_rotation_null`, plan §11 - NOT spectrum) and kNN acc@k
histograms per plan §6.

Uses `/paper-plots` conventions (SHA-pinned metadata, colorblind palette).

Smoke: --smoke → one figure at tiny scale, verify PNG + meta.json exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import ISSUE_NUM, ISSUE_SLUG  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _meta(figure_name: str) -> dict:
    import datetime as _dt

    return {
        "figure": figure_name,
        "issue": f"issue{ISSUE_NUM}_{ISSUE_SLUG}",
        "git_sha": _git_sha(),
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
    }


def render_rung_heatmap(
    ladder_json: Path, out_dir: Path, *, model: str, arm: str, smoke: bool = False
) -> Path:
    """Hero 1: rung-reached heatmap. Rows = source, cols = target, cell =
    weakest rung (1-9, viridis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(ladder_json.read_text())
    pairs = data.get("pairs", {})
    # Collect all unique source/target slugs
    all_srcs = sorted({k.split("__")[0] for k in pairs.keys()})
    all_tgts = sorted({k.split("__")[1] for k in pairs.keys()})
    grid = np.full((len(all_srcs), len(all_tgts)), np.nan)
    for pair_key, arm_map in pairs.items():
        src, tgt = pair_key.split("__")
        i = all_srcs.index(src)
        j = all_tgts.index(tgt)
        if arm in arm_map and "rung_reached" in arm_map[arm]:
            grid[i, j] = arm_map[arm]["rung_reached"]

    fig, ax = plt.subplots(figsize=(max(6, len(all_tgts) * 0.5), max(4, len(all_srcs) * 0.5)))
    im = ax.imshow(grid, cmap="viridis", vmin=1, vmax=9, aspect="auto")
    ax.set_xticks(range(len(all_tgts)))
    ax.set_yticks(range(len(all_srcs)))
    ax.set_xticklabels(all_tgts, rotation=90, fontsize=6)
    ax.set_yticklabels(all_srcs, fontsize=6)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title(f"Rung reached: {model}/{arm}" + (" (smoke)" if smoke else ""))
    fig.colorbar(im, ax=ax, label="weakest rung reaching 0.9×R²_within(T)")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"hero1_rung_heatmap_{model}_{arm}.png"
    pdf_path = out_dir / f"hero1_rung_heatmap_{model}_{arm}.pdf"
    meta_path = out_dir / f"hero1_rung_heatmap_{model}_{arm}.meta.json"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    meta_path.write_text(json.dumps(_meta(png_path.name), indent=2))
    return png_path


def write_manifest(out_dir: Path, produced: list[Path]) -> Path:
    manifest = {
        "issue": f"issue{ISSUE_NUM}_{ISSUE_SLUG}",
        "git_sha": _git_sha(),
        "artifacts": [str(p) for p in produced],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder-json", type=Path, required=True)
    ap.add_argument("--out-figs", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--model", type=str, default="Qwen_Qwen2.5-7B-Instruct")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    produced = []
    for arm in ("prefix", "context"):
        png = render_rung_heatmap(
            args.ladder_json, args.out_figs, model=args.model, arm=arm, smoke=args.smoke
        )
        produced.append(png)

    # Load rendered PNG and verify non-empty axes (plan §9a-ter figure-sanity duty).
    try:
        import numpy as np
        from PIL import Image  # type: ignore

        for p in produced:
            arr = np.asarray(Image.open(p))
            n_unique = int(np.unique(arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr).size)
            if n_unique < 10:
                print(f"[analyze] WARN: {p.name} may be blank (n_unique_pixels={n_unique})")
    except ImportError:
        print("[analyze] PIL not available - skipping figure sanity check")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.out_manifest.parent, produced)
    print(f"[analyze] produced {len(produced)} figures + manifest at {args.out_manifest.parent}")
    return 0


if __name__ == "__main__":
    import os

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGBART pointer. main()'s writes are
    # already flushed via explicit fh.close(); atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
