"""Issue #2094 user-chat inline round: banked-map transport under the ALL-LAYER patch.

The committed transport table (`transport/transport_cells.jsonl`) covers
SINGLE-LAYER patches only -- every row's ``block_key`` looks like
``ce|L14|a0.5|A|null``, so the patch layer and the banked map's layer are the
same one. ``issue2094_analysis.phase_transport`` gates on
``eligible = {(slot, f"L{layer}")}``, which skips every ``joint_all`` shard.
Result 2 therefore says nothing about the full-state patch the rest of the
writeup's headline numbers use.

Everything needed is already banked, so this costs no GPU: ``va_store`` carries
28 ``joint_all`` shards (context-end + prefix-end, all doses) and ``vc_bank``
carries the map inputs. This script stages ONLY those (~1.1 GB, not the 16.9 GB
full store), reuses the committed transport path from ``issue2094_analysis``
verbatim -- same payload reconstruction, same orientation bind from
``map_parity.json``, same cosine -- and writes a SEPARATE table so the committed
single-layer table and the figure built on it are untouched.

Why the contrast is clean at ``dose=replace``: the context-end state at layer L
is set to context B's value whether the patch touched L alone or all 28, so the
map's INPUT change -- and therefore its PREDICTION -- is identical in the two
cases. Only the REALIZED shift differs, which isolates exactly one question:
does patching more layers make the realized shift match the prediction better?

Both mapping arms run as paired arms: context-based (#779 ``m779_ce_L*``) and
prefix-based (#1738 ``m1738_pe_L*``).

Usage:
  uv run python scripts/issue2094_joint_transport.py --stage
  uv run python scripts/issue2094_joint_transport.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # credentials + shared-VM thread caps BEFORE any heavy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402
from explore_persona_space.orchestrate.hub import stage_hub_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2094_analysis as A  # noqa: E402

logger = logging.getLogger("issue2094_joint_transport")

DATASET_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = A.HF_PREFIX
LAYER_VARIANT = "joint_all"
DOSE_SLUGS = ("a0p5", "a1", "a2", "a4", "replace")
VEC_TYPES = ("A", "B")
ARMS = ("steered", "null")


def _shard_names() -> list[str]:
    """The joint_all va_store / grid shard stems that actually exist upstream.

    ce carries both vec types; pe carries only A. Non-existent combinations are
    skipped at stage time rather than guessed here.
    """
    out = []
    for slot in ("ce", "pe"):
        for dose in DOSE_SLUGS:
            for vt in VEC_TYPES:
                for arm in ARMS:
                    out.append(f"shard_{slot}__{LAYER_VARIANT}__{dose}__{vt}__{arm}")
    return out


def stage(in_root: Path) -> int:
    """Scoped staging: joint_all shards + vc_bank + anchors + the six map bundles."""
    mirror = in_root / HF_PREFIX
    n_ok = 0
    for rel in (
        "analysis_tensors/vc_bank/vc_bank.pt",
        "analysis_tensors/anchors/va_anchors.pt",
    ):
        stage_hub_file(
            repo_id=DATASET_REPO,
            path_in_repo=f"{HF_PREFIX}/{rel}",
            target=mirror / rel,
            repo_type="dataset",
            revision="main",
        )
        n_ok += 1
        logger.info("[stage] %s", rel)
    for spec in A.BANKED_MAPS:
        stage_hub_file(
            repo_id=DATASET_REPO,
            path_in_repo=spec["repo_path"],
            target=in_root / "banked_maps" / spec["repo_path"],
            repo_type="dataset",
            revision="main",
        )
        n_ok += 1
        logger.info("[stage] map %s", spec["map_id"])
    for stem in _shard_names():
        for rel, sub in (
            (f"analysis_tensors/va_store/{stem}.pt", "va"),
            (f"raw_completions/grid/{stem}.jsonl", "rollouts"),
        ):
            try:
                stage_hub_file(
                    repo_id=DATASET_REPO,
                    path_in_repo=f"{HF_PREFIX}/{rel}",
                    target=mirror / rel,
                    repo_type="dataset",
                    revision="main",
                )
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 -- absent combos are expected
                logger.info("[stage] skip %s (%s): %s", stem, sub, type(exc).__name__)
    logger.info("[phase=stage_done] %d files -> %s", n_ok, in_root)
    return 0


def compute(cfg: A.AnalysisConfig, out_path: Path) -> list[dict]:
    """Transport cosines for every joint_all shard x every banked map layer."""
    parity = json.loads(cfg.map_parity_json.read_text())
    bank = A._load_vc_bank(cfg)
    anchor_va = A._load_anchor_va(cfg)
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_map = bank.get("donor_derangement") or BANK.donor_derangement(pairs)
    bundles = {
        (spec["arm"], spec["layer"]): (
            A._load_bundle(cfg.maps_dir / spec["repo_path"]),
            A._orientation_for(parity, spec["map_id"]),
        )
        for spec in A.BANKED_MAPS
    }

    rows_out: list[dict] = []
    shards = sorted(cfg.va_dir.glob(f"shard_*__{LAYER_VARIANT}__*.pt"))
    assert shards, f"no {LAYER_VARIANT} shards under {cfg.va_dir} — run --stage first"
    for n_shard, shard in enumerate(shards, 1):
        slug = shard.stem.removeprefix("shard_")
        rows = list(A._iter_jsonl(cfg.rollouts_dir / f"shard_{slug}.jsonl"))
        if not rows:
            logger.info("[transport-joint] shard %d/%d %s: no rows", n_shard, len(shards), slug)
            continue
        head = rows[0]
        assert head["layer_variant"] == LAYER_VARIANT, head["layer_variant"]
        slot = head["slot"]
        va_tail = torch.load(shard, map_location="cpu", weights_only=False)["va_tail"].float()
        # Payload reconstruction is LAYER-INDEPENDENT (it returns the full
        # per-layer stack), so it is hoisted out of the layer loop: computing it
        # inside cost 3x and dominated the wall (~6 min/shard -> ~2.8 h total).
        payloads = [A.transport_row_payload(bank, r, pairs_by_id, donor_map) for r in rows]
        for layer in A.TRANSPORT_LAYERS[slot]:
            bundle, orientation = bundles[(slot, layer)]
            for i, r in enumerate(rows):
                fl = anchor_va[r["context_a"]]["tail"][:, layer]
                realized = va_tail[i, layer] - fl.mean(dim=0)
                fl_h1, fl_h2 = FM.disjoint_half_means(fl)
                payload, payload_kind = payloads[i]
                d_l = payload[-1][layer].float()
                v_s = A._slot_input_vector(bank, r["context_a"], r["slot"], layer)
                pred = A.transport_row_pred(
                    bundle, orientation, payload_kind, d_l, v_s, r.get("alpha")
                )
                rows_out.append(
                    {
                        "block_key": r["block_key"],
                        "map_id": f"m779_ce_L{layer}" if slot == "ce" else f"m1738_pe_L{layer}",
                        "slot": slot,
                        "layer": layer,
                        "layer_variant": LAYER_VARIANT,
                        "dose": r["dose"],
                        "alpha": r.get("alpha"),
                        "vec_type": r["vec_type"],
                        "arm": r["arm"],
                        "pair_id": r["pair_id"],
                        "setting": r["setting"],
                        "degenerate_self": A.degenerate_self(r),
                        "orientation": orientation,
                        "cosine_tail": float(FM.safe_cosine(realized, pred)),
                        "cosine_tail_half1": float(FM.safe_cosine(va_tail[i, layer] - fl_h1, pred)),
                        "cosine_tail_half2": float(FM.safe_cosine(va_tail[i, layer] - fl_h2, pred)),
                        "realized_norm": float(realized.norm()),
                        "pred_norm": float(pred.norm()),
                    }
                )
        logger.info(
            "[transport-joint] shard %d/%d %s: %d rows x %d layers",
            n_shard,
            len(shards),
            slug,
            len(rows),
            len(A.TRANSPORT_LAYERS[slot]),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows_out))
    logger.info("[phase=transport_joint_done] %d cells -> %s", len(rows_out), out_path)
    return rows_out


def make_figure(joint: list[dict], single: list[dict], out_png: Path) -> dict:
    """All-layer vs single-layer transport cosine, steered against its donor null."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean_of(rows, slot, layer, dose, arm, lv):
        v = [
            r["cosine_tail"]
            for r in rows
            if r["slot"] == slot
            and r["layer"] == layer
            and r["dose"] == dose
            and r["arm"] == arm
            and not r.get("degenerate_self")
            and (r.get("layer_variant", f"L{r['layer']}") == lv)
            and r["cosine_tail"] == r["cosine_tail"]
        ]
        return (float(np.mean(v)), len(v)) if v else (np.nan, 0)

    layers = (14, 19, 26)
    summary: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), sharey=True)
    for ax, slot in zip(axes, ("ce", "pe")):
        x = np.arange(len(layers))
        series = [
            ("single-layer patch, steered", single, "L{}", "steered", "#1b6ca8", "-", "o"),
            ("single-layer patch, null", single, "L{}", "null", "#9fc4dd", "--", "o"),
            ("ALL-layer patch, steered", joint, LAYER_VARIANT, "steered", "#c1440e", "-", "s"),
            ("ALL-layer patch, null", joint, LAYER_VARIANT, "null", "#e8a882", "--", "s"),
        ]
        for label, rows, lvfmt, arm, color, ls, marker in series:
            ys, ns = [], []
            for lay in layers:
                lv = lvfmt.format(lay) if "{}" in lvfmt else lvfmt
                m, n = mean_of(rows, slot, lay, "replace", arm, lv)
                ys.append(m)
                ns.append(n)
            ax.plot(x, ys, marker=marker, ls=ls, color=color, label=f"{label} (n={max(ns)})")
            summary[f"{slot}|{label}"] = {"layers": list(layers), "cosine": ys, "n": ns}
        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"map at L{lay}" for lay in layers])
        ax.set_title(
            f"{'context-end map (#779)' if slot == 'ce' else 'prefix-end map (#1738)'}\n"
            "full-state (replace) patch"
        )
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("cos(map-predicted shift, realized shift)")
    fig.suptitle(
        "Does the banked map predict the answer-vector shift when ALL 28 layers are patched?\n"
        "at dose=replace the map's PREDICTION is identical for both patches — only the realized "
        "shift differs; read the margin over the donor null, not the raw cosine",
        fontsize=10.5,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("[transport-joint] figure -> %s", out_png)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=Path, default=Path("data/issue_2094/joint_transport"))
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_2094"))
    ap.add_argument(
        "--figure", type=Path, default=Path("figures/issue_2094/transport_joint_all.png")
    )
    ap.add_argument("--stage", action="store_true", help="download the scoped inputs and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.stage:
        raise SystemExit(stage(args.in_root))

    cfg = A.AnalysisConfig(
        in_root=args.in_root,
        out_root=args.out_root,
        judge_root=args.out_root / "judge",
        hf_revision=None,
    )
    out_path = args.out_root / "transport" / "transport_cells_joint.jsonl"
    joint = compute(cfg, out_path)
    single = list(A._iter_jsonl(args.out_root / "transport" / "transport_cells.jsonl"))
    summary = make_figure(joint, single, args.figure)
    (args.out_root / "transport" / "transport_joint_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    logger.info("[phase=done] %s", args.figure)


if __name__ == "__main__":
    main()
