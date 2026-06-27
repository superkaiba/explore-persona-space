#!/usr/bin/env python3
# ruff: noqa: RUF002
# (math/scientific notation — c_C', Σ_c — intentional in docstrings)
"""Issue #683 Phase A — sycophancy c_C' bank builder (L20), the panel top-up.

Plan §4 Phase A (sycophancy `c_C'`). Re-emits the #612 panel centroids
(``panel_centroids_layer20.pt``, shape ``{"centroids": {20: (52, H)},
"persona_names": [52], ...}``) in the ``{contexts: {persona: (H,)}}`` shape the
key×metric scorer's ``_load_c_bank`` consumes, keyed to the pinned 30-persona
``SYCOPHANCY_PANEL_CONTEXTS`` (BLOCKER syco-cbank-load-incompatible: the raw
centroid bank is NOT loadable by ``_load_c_bank``; and CONCERN
syco-panel-topup-wrong-layer: the dispatcher previously ran the L2/L7
``issue649_extract_panel_earlylayer.py`` which the scorer never consumes).

This is a CPU re-emit of an existing L20 bank — NO GPU, NO generation. (If a
future panel context were genuinely missing from the centroid bank,
``build_sycophancy_c_bank_l20`` FAILS LOUD rather than silently dropping it; all
30 pinned panel personas are Hub-verified present in the 52-name centroid set as
of 2026-06-26, so no fresh extraction is needed.)

Output (the scorer's ``--c-bank``):
    eval_results/issue_683/analysis_tensors/c_bank/sycophancy/c_bank_sycophancy_L20.pt
    shape: {"contexts": {persona: (H,) float}, "meta": {...}}

CLI:
    uv run python scripts/issue683_build_syco_c_bank.py --layer 20
    # smoke (use the centroid bank as-is; same path, tiny verification):
    uv run python scripts/issue683_build_syco_c_bank.py --layer 20 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_build_syco_c_bank")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.issue_683 import (  # noqa: E402
    HF_DATA_REPO,
    SYCO_PANEL_CENTROIDS_L20,
    SYCOPHANCY_PANEL_CONTEXTS,
    build_sycophancy_c_bank_l20,
    repro_metadata,
    sha256_file,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--layer", type=int, default=20, help="centroid-bank layer (default L20)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="smoke namespace (separate out subdir)")
    args = ap.parse_args(argv)

    import torch
    from huggingface_hub import hf_hub_download

    out_path = Path(
        args.out
        or (
            PROJECT_ROOT
            / "eval_results/issue_683/analysis_tensors"
            / ("c_bank_smoke" if args.smoke else "c_bank")
            / "sycophancy"
            / f"c_bank_sycophancy_L{args.layer}.pt"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cent_path = hf_hub_download(
        HF_DATA_REPO, SYCO_PANEL_CENTROIDS_L20, repo_type="dataset", revision="main"
    )
    logger.info(
        "[phase=c_bank_load] centroids=%s sha=%s",
        SYCO_PANEL_CENTROIDS_L20,
        sha256_file(cent_path)[:16],
    )
    centroids_obj = torch.load(cent_path, map_location="cpu", weights_only=False)

    bank = build_sycophancy_c_bank_l20(centroids_obj, SYCOPHANCY_PANEL_CONTEXTS, layer=args.layer)
    # Hard invariant: one (H,) vector per pinned panel context, finite.
    contexts = bank["contexts"]
    assert len(contexts) == len(SYCOPHANCY_PANEL_CONTEXTS), (
        len(contexts),
        len(SYCOPHANCY_PANEL_CONTEXTS),
    )
    for name, v in contexts.items():
        assert v.ndim == 1, (name, tuple(v.shape))
        assert bool(torch.isfinite(v).all()), f"non-finite c_C' for {name}"

    bank["meta"]["reproducibility"] = repro_metadata(
        {"behavior": "sycophancy", "layer": args.layer}
    )
    bank["meta"]["centroids_source"] = SYCO_PANEL_CENTROIDS_L20
    bank["meta"]["centroids_sha256"] = sha256_file(cent_path)
    torch.save(bank, out_path)

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "n_contexts": len(contexts),
                "contexts": sorted(contexts),
                "out_path": str(out_path),
                "meta": {k: v for k, v in bank["meta"].items() if k != "reproducibility"},
            },
            indent=2,
        )
    )
    logger.info(
        "[phase=c_bank_done] %d panel contexts at L%d -> %s",
        len(contexts),
        args.layer,
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
