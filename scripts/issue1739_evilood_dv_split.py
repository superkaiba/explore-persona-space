#!/usr/bin/env python3
"""#1739 evil-OOD r2v2: rewrite the new rungs' DV ``split`` from 'full' to 'eval'.

The r2v2 generation driver stamps ``split="full"`` on every rollout (it is the
generation split id), and that value rides into the DV rows. The fits-side
loader (``scripts.issue1739_fits._load_labeled``) filters by ``split``:
``config_a -> 'train'``, ``config_b -> 'eval'``. The three new evil OOD rungs
(mhj / tom-gibbs / pair) are pure EVAL rungs — no readout ever trains on them —
so they must carry ``split="eval"`` to be loadable at all. Same convention the
syco-OOD round applied to its five rungs (``issue1739_sycoood_rescore_stage.py``
asserts every new DV row is ``split == "eval"`` before merging).

Idempotent: rows already at 'eval' are left alone; any row at a split OTHER
than 'full'/'eval' fails loud rather than being silently rewritten.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

EXPECTED_RUNGS = {"evil_mhj", "evil_tomgibbs", "evil_pair"}


def main() -> int:
    ap = argparse.ArgumentParser(description="#1739 evil-OOD DV split rewrite (full -> eval)")
    ap.add_argument(
        "--dv", default="eval_results/issue_1739/evil_ood_full/dv_dataset/evil/labeling.json"
    )
    args = ap.parse_args()

    path = Path(args.dv)
    payload = json.loads(path.read_text())
    rows = payload["rows"]

    rungs = {r.get("rung") for r in rows}
    if rungs != EXPECTED_RUNGS:
        raise SystemExit(f"unexpected rung set {sorted(rungs)} != {sorted(EXPECTED_RUNGS)}")

    n_rewritten = 0
    for r in rows:
        split = r.get("split")
        if split == "eval":
            continue
        if split != "full":
            raise SystemExit(f"row {r.get('context_id')!r} has unexpected split={split!r}")
        r["split"] = "eval"
        n_rewritten += 1

    missing_group = [r["context_id"] for r in rows if not r.get("group_key")]
    if missing_group:
        raise SystemExit(f"{len(missing_group)} rows lack group_key (LOFO folds are load-bearing)")

    payload["split_rewrite_note"] = (
        "generation stamped split='full' (its generation split id); the three OOD rungs are pure "
        "EVAL rungs, so split was rewritten to 'eval' for the config_b loader — same convention "
        "as the syco-OOD round's five rungs. No DV value was altered."
    )
    path.write_text(json.dumps(payload, indent=1))

    per_rung = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        b = per_rung[r["rung"]]
        b[1] += 1
        if r.get("dv") is not None:
            b[0] += 1
    print(f"rewrote split for {n_rewritten}/{len(rows)} rows -> {path}")
    for rung, (ok, tot) in sorted(per_rung.items()):
        print(f"  {rung}: dv-present {ok}/{tot} = {ok / tot:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
