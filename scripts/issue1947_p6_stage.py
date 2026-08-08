#!/usr/bin/env python
"""Stage every #1947 P6 input from HF into a P6-consumable out-root on the VM
data disk, so the P6 assumption battery can run FLEET-WIDE (all 56 verdict
arms) rather than on a syc-only subset.

Why fleet-wide: ``issue1947_analysis.py`` rewrites ``battery_summary.json``,
``frame_*.json`` and the figures WHOLE over its ``--arms`` filter, so a
syc-scoped run would clobber the committed 38-arm artifacts — the same
whole-file-rewrite trap ``issue1947_battery.py --phase select`` has, handled the
same way (run over everything, then audit additivity against a pre-run backup
with ``issue1947_p6_additivity.py``).

Two input groups:

1. Per-arm battery trees for the 38 imp/cas+marker arms (the syc arms come off
   the pod directly by rsync, already in out-root layout). Layout REMAP (the
   #928/#1481 staged-layout trap): the pod writes delta_tf at
   ``out_root/delta_tf/<arm>`` but uploads it to
   ``<prefix>/battery/delta_tf/<arm>`` — the ``battery/`` component is DROPPED
   on the way back down. Every other tree maps verbatim.
2. The shared #1768 inputs ``_shared_inputs`` needs (corpus Sigma per layer plus
   the pfx/corpus samples). Without these P6 fails loud with
   "corpus sigma L<n> unavailable". The r_B tensors are NOT staged here —
   ``issue1768_directions.load_rb_tensors`` self-stages them on miss.

Hub reads go through the retried, server-side-SCOPED ``orchestrate.hub``
helpers (``list_hf_files_under_path`` / ``stage_hub_file``), never a bare
``list_repo_tree`` / ``hf_hub_download`` and never ``snapshot_download`` —
which enumerates the ENTIRE ~1M-file data repo before writing a byte (measured:
>3.5 min of listing with zero files staged).

P6 opens only ``.pt`` / ``.json`` under these trees (verified against
``issue1947_analysis.py``: ``_stack_for_tree``, ``_panel_v0_halves``,
``_delta_legs``, ``collect_d_forest``) — the ``gen_base`` / ``gen_trained``
raw-text subdirs are never read, so they are not staged.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1947_singlevisit"
P1768 = "issue1768_mapshift"
STAGE_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1947_p6_fleet")
KEEP_SUFFIX = (".pt", ".json")
WORKERS = 16
# A CONTENT arm's trained-rows tree carries the full mix plus the consumed@verdict
# split; a marker (mk-) arm carries only the full-mix pair.
TRAINED_ROWS_CONTENT_FILES: tuple[str, ...] = (
    "pooled.pt",
    "pooled_base.pt",
    "pooled_consumed.pt",
    "pooled_base_consumed.pt",
)

# (remote subtree under PREFIX, local subtree under the out-root)
TREES: tuple[tuple[str, str], ...] = (
    ("battery/trained_rows", "battery/trained_rows"),
    ("battery/onpolicy", "battery/onpolicy"),
    ("battery/panel", "battery/panel"),
    ("battery/margins", "battery/margins"),
    ("battery/dynamics", "battery/dynamics"),
    ("battery/delta_tf", "delta_tf"),  # <-- the REMAP
    ("fits", "fits"),
)

# Shared #1768 inputs: Hub path under P1768 maps onto the out-root verbatim.
SHARED_1768: tuple[str, ...] = (
    "inputs/corpus_sample.json",
    "on_target/inputs/corpus_sample_pfx.json",
    "corpus_capture/base_content/pooled.pt",
    "corpus_capture/base_content/rows_spans.json",
    "corpus_capture/base_content/manifest.json",
)


def _enumerate(api: HfApi, remote: str) -> list[str]:
    """Retried, server-side-scoped listing of ONE subtree."""
    paths = hub.list_hf_files_under_path(api, REPO, f"{PREFIX}/{remote}", repo_type="dataset")
    return [p for p in paths if p.endswith(KEEP_SUFFIX)]


def _fetch(remote_path: str, remote: str, local: str, stage: Path) -> int:
    rel = remote_path[len(f"{PREFIX}/{remote}/") :]
    out = stage / local / rel
    if out.exists() and out.stat().st_size > 0:
        return out.stat().st_size
    hub.stage_hub_file(REPO, remote_path, out, repo_type="dataset")
    return out.stat().st_size


def stage_battery_trees(api: HfApi, stage: Path) -> int:
    total = 0
    for remote, local in TREES:
        try:
            paths = _enumerate(api, remote)
        except Exception as e:  # noqa: BLE001 — report, never silently skip
            print(f"[stage] {remote}: ABSENT on Hub ({type(e).__name__}) — expected for dynamics")
            continue
        if not paths:
            print(f"[stage] {remote}: 0 files on Hub — expected for dynamics")
            continue
        got = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(_fetch, p, remote, local, stage): p for p in paths}
            for f in as_completed(futs):
                total += f.result()  # fail loud: a staging miss must not be swallowed
                got += 1
        print(f"[stage] {remote} -> {local}: {got}/{len(paths)} files", flush=True)
    return total


def stage_shared_1768(stage: Path) -> int:
    total = 0
    for rel in SHARED_1768:
        out = stage / rel
        if out.exists() and out.stat().st_size > 0:
            print(f"[1768] present {rel} ({out.stat().st_size / 1e6:.1f} MB)")
            total += out.stat().st_size
            continue
        hub.stage_hub_file(REPO, f"{P1768}/{rel}", out, repo_type="dataset")
        total += out.stat().st_size
        print(f"[1768] staged {rel} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return total


def verify_complete(stage: Path, manifest: Path) -> int:
    """Hard gate before the fleet-wide P6 run: every manifest arm's inputs staged.

    P6 records a per-cell ``{"missing": ...}`` record rather than failing when a
    tree is absent, so a partially-staged out-root yields a SILENTLY incomplete
    fleet summary. Expected shape is behavior-aware: marker (``mk-``) cells carry
    only ``pooled.pt`` / ``pooled_base.pt`` and have NO on-policy or panel trees
    (``issue1947_battery.unit_arm`` skips ``_unit_onpolicy_and_panel`` /
    ``_unit_margin`` for them), so requiring the content shape there would
    fail-loud on a correct tree.
    """
    man = json.loads(manifest.read_text())
    arms = sorted(man["content"]) + sorted(man["marker"])
    bad: list[str] = []
    for arm in arms:
        marker = arm.startswith("mk-")
        need: list[tuple[str, tuple[str, ...]]] = [
            (
                f"battery/trained_rows/{arm}",
                ("pooled.pt", "pooled_base.pt") if marker else TRAINED_ROWS_CONTENT_FILES,
            )
        ]
        if not marker:
            need.append((f"battery/onpolicy/{arm}", ("pooled.pt", "pooled_base.pt")))
            need.append((f"battery/panel/{arm}", ("pooled_base.pt", "pooled_trained.pt")))
        for sub, files in need:
            missing = [f for f in files if not (stage / sub / f).exists()]
            if missing:
                bad.append(f"{sub}: missing {missing}")
    pools = sorted(
        f"{man['content'][s]['beh_key']}-{man['content'][s]['ctx_key']}" for s in man["content"]
    )
    for pool in sorted(set(pools)):
        if not (stage / "delta_tf" / f"{pool}-delta1947" / "tbar.pt").exists():
            bad.append(f"delta_tf/{pool}-delta1947: missing ['tbar.pt']")
    for rel in SHARED_1768:
        if not (stage / rel).exists():
            bad.append(f"{rel}: missing (shared #1768 input)")

    print(
        f"[verify] {len(arms)} manifest arms ({len(man['content'])} content + {len(man['marker'])} marker)"
    )
    if bad:
        print(f"[verify] INCOMPLETE — {len(bad)} gap(s):")
        for b in bad:
            print("  -", b)
        print("VERDICT: FAIL — do NOT run the fleet-wide P6 on a partial out-root")
        return 1
    print("[verify] every manifest arm's P6 inputs are staged")
    print("VERDICT: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root", default=str(STAGE_DEFAULT))
    p.add_argument("--skip-battery", action="store_true")
    p.add_argument("--skip-shared", action="store_true")
    p.add_argument(
        "--verify-complete",
        action="store_true",
        help="after staging, hard-gate that EVERY manifest arm's P6 inputs are present",
    )
    p.add_argument("--manifest", default="eval_results/issue_1947/analysis/verdict_manifest.json")
    args = p.parse_args(argv)

    stage = Path(args.stage_root)
    stage.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    total = 0
    if not args.skip_battery:
        total += stage_battery_trees(api, stage)
    if not args.skip_shared:
        total += stage_shared_1768(stage)
    print(f"[stage] TOTAL {total / 1e9:.2f} GB under {stage}")
    if args.verify_complete:
        return verify_complete(stage, Path(args.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
