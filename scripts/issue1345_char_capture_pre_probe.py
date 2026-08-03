#!/usr/bin/env python
"""Issue #1345 char-capture-ladders — pre-capture pairing probe (plan v13 §4, §7 gate 2).

Text-only, CPU, runs BEFORE any GPU dispatch. For each of the 16 character
cells it (a) stages the kept-story bundle at the plan §10 pin (idempotent, via
``issue1345_stage_char_stories``), (b) parses EVERY kept row through the
existing ``issue1345_common`` helpers (``parse_story_turns`` +
``ANSWER_ATTRIB_RE`` + the r4 verbatim-span check), (c) asserts unique
conversation ids, and (d) computes the conv-id intersection with the cell's
LADDER SOURCE store (r4 / r4op assistant stores for instruct cells; the
pretrained chat store for ``_base`` cells — plan § Divergences 2), read from
the staged shard SIDECAR JSONs (never the multi-GB tensors).

Because ``issue1345_common`` compiles ``ANSWER_ATTRIB_RE`` from
``EPM_STORY_CHARACTER_NAME`` at MODULE IMPORT (first import in a process wins),
each cell's parse runs in its OWN subprocess with the env exported BEFORE the
import (``--single-cell`` inner mode). The outer process never imports
``issue1345_common``.

Thresholds (plan §7 gate 2 + Kill criteria): duplicate conv-ids or
intersection < 400 -> DROP the cell (exit 1); intersection in [400, 800) ->
the cell runs with a power caveat; >= 800 -> PASS; > 8 drop-class cells ->
HALT the round pre-GPU (exit 2). Content hygiene: kept stories are
LMSYS-derived real user text — this probe never prints story text (counts,
conv-ids, and hashes only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_stage_char_stories as stager  # noqa: E402

CHAR_VARIANTS = stager.CHAR_VARIANTS

# Ladder-source stem per cell class (plan §4 Phase F / § Divergences 2). The
# sidecar files live under the #1887-staged flat subdirs of --stage-root.
SOURCE_SIDECAR_GLOBS = {
    "r4": (
        "conversation_paired_stories_assistant_turnstore",
        "instruct_stories_paired_s_shard*.json",
    ),
    "r4op": ("onpolicy_assistant_story_turnstore", "instruct_stories_paired_op_s_shard*.json"),
    "r1_base": ("parent_turnstore", "pretrained_chat_s_shard*.json"),
}

INTERSECTION_PASS = 800  # plan §7 gate 2 floor
INTERSECTION_DROP = 400  # plan Kill criteria drop threshold
HALT_AFFECTED_CELLS = 8  # plan Kill criteria: > 8 drop-class cells => halt


def source_key_for(variant: str) -> str:
    """Which ladder source a cell pairs with (base cells pair with base chat)."""
    if variant.endswith("_base"):
        return "r1_base"
    return "r4op" if "_op" in variant else "r4"


def _source_conv_ids(stage_root: Path, key: str) -> list[str]:
    """Union of conv_ids over the source store's shard sidecar JSONs."""
    subdir, glob = SOURCE_SIDECAR_GLOBS[key]
    paths = sorted((stage_root / subdir).glob(glob))
    assert paths, f"no shard sidecars at {stage_root / subdir}/{glob}"
    ids: list[str] = []
    for p in paths:
        ids.extend(json.loads(p.read_text())["conv_ids"])
    assert len(set(ids)) == len(ids), f"duplicate conv_ids across {key} source shards"
    return ids


def run_single_cell(variant: str, stories_dir: Path, cell_out: Path) -> None:
    """Inner mode: parse every kept row of ONE cell (correct env already set)."""
    assert os.environ.get("EPM_I1345_VARIANT") == variant, "inner mode needs the variant env"
    assert os.environ.get("EPM_STORY_CHARACTER_NAME"), "inner mode needs the character env"
    import issue1345_common as c  # noqa: PLC0415 — import AFTER env (regex compiles at import)

    mode, model = stager.variant_mode_model(variant)
    kept_path = stories_dir / f"kept_stories_{mode}_{model}.jsonl"
    rows = c.read_jsonl(kept_path)
    conv_ids = [r["conv_id"] for r in rows]
    viol = {
        "attrib_not_one": 0,
        "parse_no_turn": 0,
        "stored_turns_not_one": 0,
        "verbatim_mismatch": 0,
    }
    for r in rows:
        story = r["story"]
        if len(list(c.ANSWER_ATTRIB_RE.finditer(story))) != 1:
            viol["attrib_not_one"] += 1
        if len(c.parse_story_turns(story)) < 1:
            viol["parse_no_turn"] += 1
        stored = r.get("parsed_turns", [])
        if len(stored) != 1:
            viol["stored_turns_not_one"] += 1
        elif mode == "paired":  # injected cells embed the answer verbatim (r4 only)
            t = stored[0]
            if c.norm_text(story[t["a_start"] : t["a_end"]]) != c.norm_text(r["answer"]):
                viol["verbatim_mismatch"] += 1
    out = {
        "variant": variant,
        "character": os.environ["EPM_STORY_CHARACTER_NAME"],
        "kept_path": str(kept_path),
        "n_rows": len(rows),
        "n_unique_conv_ids": len(set(conv_ids)),
        "violations": viol,
        "conv_ids": conv_ids,
    }
    cell_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = cell_out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, cell_out)
    print(
        f"[probe-cell] {variant}: n={len(rows)} unique={len(set(conv_ids))} "
        f"violations={sum(viol.values())}",
        flush=True,
    )


def _git_commit() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={**os.environ},
    )
    return r.stdout.strip() or "unavailable"


def main() -> None:
    """Outer mode: stage + probe every requested cell, write the report, gate."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    user = os.environ.get("USER", "thomasjiralerspong")
    ap.add_argument("--cells", nargs="+", default=list(CHAR_VARIANTS), choices=CHAR_VARIANTS)
    ap.add_argument("--revision", default=stager.STORIES_PIN)
    ap.add_argument("--dest-root", type=Path, default=Path("data/issue_1345"))
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path(f"/mnt/eps-data/{user}/issue1887_lambda_audit/issue1345"),
        help="root holding the staged ladder-source turnstore subdirs (sidecars only are read)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "eval_results/issue_1345/char_capture_ladders/pre_capture_probe.json",
    )
    ap.add_argument("--single-cell", default=None, help="INTERNAL: run one cell in-process")
    ap.add_argument("--stories-dir", type=Path, default=None, help="INTERNAL (single-cell)")
    ap.add_argument("--cell-out", type=Path, default=None, help="INTERNAL (single-cell)")
    args = ap.parse_args()

    if args.single_cell:
        assert args.stories_dir and args.cell_out, "--single-cell needs --stories-dir/--cell-out"
        run_single_cell(args.single_cell, args.stories_dir, args.cell_out)
        return

    t0 = time.time()
    sources = {k: set(_source_conv_ids(args.stage_root, k)) for k in SOURCE_SIDECAR_GLOBS}
    print(
        "[probe] source conv-id sets: " + ", ".join(f"{k}={len(v)}" for k, v in sources.items()),
        flush=True,
    )

    cells: dict[str, dict] = {}
    scratch = args.out.parent / "pre_capture_probe_cells"
    for variant in args.cells:
        stories_dir = stager.stage_variant(
            variant, revision=args.revision, dest_root=args.dest_root
        )
        mode, model = stager.variant_mode_model(variant)
        yinfo = json.loads((stories_dir / f"story_yield_{mode}_{model}.json").read_text())
        label = yinfo["story_character_name"]
        cell_out = scratch / f"{variant}.json"
        env = {
            **os.environ,
            "EPM_I1345_VARIANT": variant,
            "EPM_STORY_CHARACTER_NAME": label,
        }
        r = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single-cell",
                variant,
                "--stories-dir",
                str(stories_dir),
                "--cell-out",
                str(cell_out),
            ],
            env=env,
            timeout=900,
        )
        assert r.returncode == 0, f"single-cell probe failed for {variant} (rc={r.returncode})"
        cell = json.loads(cell_out.read_text())
        src_key = source_key_for(variant)
        common = set(cell.pop("conv_ids")) & sources[src_key]
        n_dup = cell["n_rows"] - cell["n_unique_conv_ids"]
        if n_dup > 0 or len(common) < INTERSECTION_DROP:
            verdict = "drop"
        elif len(common) < INTERSECTION_PASS:
            verdict = "caveat"
        else:
            verdict = "pass"
        cell.update(
            {
                "ladder_source": src_key,
                "n_source": len(sources[src_key]),
                "n_common": len(common),
                "n_duplicate_conv_ids": n_dup,
                "verdict": verdict,
            }
        )
        cells[variant] = cell
        print(
            f"[probe] {variant}: n={cell['n_rows']} common({src_key})={len(common)} "
            f"dup={n_dup} -> {verdict}",
            flush=True,
        )

    n_drop = sum(1 for v in cells.values() if v["verdict"] == "drop")
    overall = "halt" if n_drop > HALT_AFFECTED_CELLS else ("drop-cells" if n_drop else "pass")
    report = {
        "metadata": {
            "script": "scripts/issue1345_char_capture_pre_probe.py",
            "git_commit": _git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stories_revision": args.revision,
            "stage_root": str(args.stage_root),
            "thresholds": {
                "pass_floor": INTERSECTION_PASS,
                "drop_floor": INTERSECTION_DROP,
                "halt_affected_cells": HALT_AFFECTED_CELLS,
            },
            "wall_s": round(time.time() - t0, 1),
        },
        "sources": {k: len(v) for k, v in sources.items()},
        "cells": cells,
        "n_drop_cells": n_drop,
        "overall": overall,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, args.out)
    print(f"[probe] overall={overall} n_drop={n_drop} -> {args.out}", flush=True)
    if overall == "halt":
        sys.exit(2)
    if overall == "drop-cells":
        sys.exit(1)


if __name__ == "__main__":
    main()
