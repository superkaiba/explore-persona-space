#!/usr/bin/env python
"""Issue #1345 SMOKE FIXTURE — synthetic post-extract state at the kept=1 grain.

Builds the tiny-real local-chain inputs the pod smoke produces AFTER
gen_stories + extract_* (both GPU-bound; carve-out per the round-3 report):
  turnstore/  — pt shards in the exact `issue825_fit_cells._load_bundle_pt`
                contract (list-payload slots/profiles/nll + conv_ids + a
                sidecar JSON), for instruct/pretrained x r1/r2 (8 shared
                conversations) and r3 at the STORY-SHORTFALL grain (ONE kept
                story -> 3 rows, one CV group — the degenerate shape the
                round-4 smoke guards must absorb).
  stories/    — story_yield_{model}.json digests (counts only, no text).

The chain then runs the REAL phase mains on CPU:
  fit_cells --parity --build-matched --smoke  ->  fit_cells --cells all --smoke
  -> cross_regime_transfer --smoke -> operator_comparison --smoke -> plots.

Fixture ONLY — never referenced by the dispatcher or any production phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

N_CONV = 8  # the smoke leg's r1/r2 conversation cap
N_LAYERS = 28  # fc.EXPECTED_LAYERS — the loader asserts this depth
DIM = 16  # hidden dim is free at fixture scale
R3_ROWS = 3  # one kept story -> 3 parsed Q->A turns (ONE CV group)


def _rows(rng: np.random.Generator, n: int, n_turns: int, w: np.ndarray) -> dict:
    """Per-record list payloads: slots (2,L,D), profiles (n_turns,L,D), nll."""
    slots, profiles, nll, spans_meta = [], [], [], []
    for _ in range(n):
        s = rng.normal(size=(2, N_LAYERS, DIM)).astype(np.float32)
        p = np.stack(
            [s[1] @ w + 0.3 * rng.normal(size=(N_LAYERS, DIM)) for _ in range(n_turns)]
        ).astype(np.float32)
        slots.append(torch.from_numpy(s))
        profiles.append(torch.from_numpy(p))
        nll.append(torch.from_numpy(rng.uniform(1.0, 3.0, size=(n_turns,)).astype(np.float32)))
        key = "a1" if n_turns == 2 else "answer"
        spans_meta.append({"spans": {key: [4, 4 + int(rng.integers(8, 40))]}})
    return {"slots": slots, "profiles": profiles, "nll": nll, "spans_meta": spans_meta}


def _write_shard(ts_dir: Path, stem: str, conv_ids: list[str], payload: dict) -> None:
    torch.save({"conv_ids": conv_ids, **payload}, ts_dir / f"{stem}_shard0.pt")
    (ts_dir / f"{stem}_shard0.json").write_text(json.dumps({"conv_ids": conv_ids}))
    print(f"[fixture] wrote {stem}_shard0.pt ({len(conv_ids)} rows)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ts_dir = args.out_root / "turnstore"
    st_dir = args.out_root / "stories"
    ts_dir.mkdir(parents=True, exist_ok=True)
    st_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    shared_ids = [f"s{i}" for i in range(N_CONV)]
    for model in c.MODELS:
        # Per-regime fixed linear map (so the within fits carry real signal)
        for regime in ("r1", "r2"):
            w = rng.normal(size=(DIM, DIM)).astype(np.float32) / np.sqrt(DIM)
            _write_shard(ts_dir, c.stem_for(model, regime), shared_ids, _rows(rng, N_CONV, 2, w))
        # R3 at the kept=1 story-shortfall grain: one story, 3 rows, ONE group
        story_id = f"{model}_story0000"
        w3 = rng.normal(size=(DIM, DIM)).astype(np.float32) / np.sqrt(DIM)
        _write_shard(
            ts_dir, c.stem_for(model, "r3"), [story_id] * R3_ROWS, _rows(rng, R3_ROWS, 1, w3)
        )
        c.write_json(
            st_dir / f"story_yield_{model}.json",
            {
                "metadata": c.metadata(args.seed, 1, "scripts/issue1345_smoke_fixture.py"),
                "model": model,
                "n_target": 3,
                "yield_floor": 1,
                "n_kept": 1,
                "yield_ok": True,
                "counts_main": {"n_generated": 3, "judge_pass": 1, "kept": 1},
            },
        )
    print(f"[fixture] done under {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
