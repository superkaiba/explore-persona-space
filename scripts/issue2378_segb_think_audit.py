"""Audit `<think>` contamination in issue #2378 SegB story replies.

SegB is the segment where the story CHARACTER answers; its activations over the
answer span are the target vector of every map fit. Unlike the chat-template
cells (which render with ``enable_thinking=False`` and drop on a ``<think>``
literal) and unlike SegA mining (which bans ``<think>`` at the sampler via the
G1 r11 recalibration lever L1), SegB has NEITHER a sampler ban NOR a rejection
check -- ``_classify_segb_row`` tests only closing-quote presence and
non-emptiness. The r11 implementation marker recorded SegB's think-leak
exposure as UNMEASURED. This script measures it from the banked pilot rows.

The distinction that matters: ``answer = gen_text[:first_closing_quote]`` is the
measured span. A ``<think>`` block emitted AFTER the character closes their
quote never reaches the activations; one emitted before or inside it does.

Usage (0 GPU-h, reads banked HF artifacts):

    uv run python scripts/issue2378_segb_think_audit.py --round r2
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_tree

from explore_persona_space.orchestrate.env import load_dotenv

DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue2378_xframing/raw_completions/pilot/{round}/segb"
THINK = "<think>"


def _fetch(round_: str, dest: Path) -> list[Path]:
    """Download every SegB completion shard for ``round_`` into ``dest``."""
    prefix = PREFIX.format(round=round_)
    remote = [
        f.path
        for f in list_repo_tree(DATA_REPO, path_in_repo=prefix, repo_type="dataset")
        if f.path.endswith(".jsonl")
    ]
    if not remote:
        raise RuntimeError(f"no SegB shards under {prefix} -- wrong round or prefix (fail loud)")
    return [
        Path(hf_hub_download(DATA_REPO, path, repo_type="dataset", local_dir=str(dest)))
        for path in remote
    ]


def _cell_of(path: Path) -> str:
    """Recover the cell name from the shard filename (`<cell>_w<k>_s<j>_c<nnnn>.jsonl`)."""
    stem = path.name.split("_w1_")[0]
    if stem == path.name:
        raise RuntimeError(f"unrecognized SegB shard name {path.name} (fail loud)")
    return stem


def audit(files: list[Path]) -> dict[str, collections.Counter]:
    """Count, per cell, rows whose raw continuation vs measured answer span carry THINK."""
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in files:
        cell = _cell_of(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gen = row.get("gen_text") or ""
            answer = row.get("answer") or ""
            kept = bool(row.get("keep"))
            counter = stats[cell]
            counter["rows"] += 1
            counter["kept"] += int(kept)
            counter["think_in_gen"] += int(THINK in gen)
            counter["think_in_answer"] += int(THINK in answer)
            counter["think_kept"] += int(kept and THINK in answer)
            if THINK in answer:
                # Leading block vs mid-answer collapse: different failure modes.
                counter["think_leads" if answer.index(THINK) == 0 else "think_midspan"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", default="r2", help="pilot round prefix (r1 / r2)")
    parser.add_argument("--dest", default="data/issue_2378/thinkcheck")
    parser.add_argument("--out", default=None, help="optional JSON summary path")
    args = parser.parse_args()

    load_dotenv()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    stats = audit(_fetch(args.round, dest))

    header = f"{'cell':22s} {'rows':>6s} {'kept':>6s} {'think_gen':>10s} {'think_ans':>10s} {'kept_bad':>9s}"
    print(f"=== issue 2378 SegB <think> audit ({args.round}) ===")
    print(header)
    total: collections.Counter = collections.Counter()
    for cell in sorted(stats):
        s = stats[cell]
        total.update(s)
        print(
            f"{cell:22s} {s['rows']:6d} {s['kept']:6d} {s['think_in_gen']:10d} "
            f"{s['think_in_answer']:10d} {s['think_kept']:9d}"
        )
    print(
        f"{'TOTAL':22s} {total['rows']:6d} {total['kept']:6d} {total['think_in_gen']:10d} "
        f"{total['think_in_answer']:10d} {total['think_kept']:9d}"
    )
    print(
        f"\nleading-block {total['think_leads']} / mid-answer-collapse {total['think_midspan']}; "
        f"post-quote-only (harmless to the measured span) "
        f"{total['think_in_gen'] - total['think_in_answer']}"
    )

    if args.out:
        payload = {
            "round": args.round,
            "per_cell": {k: dict(v) for k, v in stats.items()},
            "total": dict(total),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
