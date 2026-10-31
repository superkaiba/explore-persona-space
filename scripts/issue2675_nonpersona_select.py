"""Issue #2675 follow-up: pick neuron-directions for the non-persona strength sweep.

Source: the Neuron Persona Library mirror ``willhsp/qwen-neuron-personas`` (one
greedy "Who are you?" reply per Qwen3-8B neuron-direction, labelled by Claude
Haiku 5.5 as persona / changed_assistant / off_topic; neuron-directions with a
normal, incoherent or low-weirdness reply are absent from the file).

Sets (fixed seed):
  test       30 off_topic + 30 changed_assistant, stratified by layer in
             proportion to each group's own layer counts, plus three named
             neuron-directions (L8.N5315 down, L9.N10974 down, L10.N5809 down)
  persona    20 library persona neuron-directions, layers matched to the test set
  random     20 neuron-directions absent from the library, layers matched to the
             test set, direction drawn at random

Writes ``neurons.json`` with the library revision, the CSV sha256 and every pick.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

SEED = 2675
LIB_REPO = "willhsp/qwen-neuron-personas"
LIB_REVISION = "92298a2a0a854a836d1f2dcf64b5d202352d6d67"
N_LAYERS = 36
N_NEURONS = 12288
NAMED = [(8, 5315, "down"), (9, 10974, "down"), (10, 5809, "down")]


def allocate(counts: dict[int, int], n: int) -> dict[int, int]:
    """Split n across layers in proportion to counts (largest remainder)."""
    total = sum(counts.values())
    raw = {L: n * c / total for L, c in counts.items()}
    alloc = {L: int(np.floor(v)) for L, v in raw.items()}
    rest = sorted(raw, key=lambda L: (-(raw[L] - alloc[L]), L))
    for L in rest[: n - sum(alloc.values())]:
        alloc[L] += 1
    assert sum(alloc.values()) == n, alloc
    return {L: k for L, k in alloc.items() if k > 0}


def pick(rng, pool: list[dict], alloc: dict[int, int]) -> list[dict]:
    """Sample alloc[L] rows per layer from pool without replacement."""
    out = []
    for L in sorted(alloc):
        cand = [r for r in pool if r["layer"] == L]
        assert len(cand) >= alloc[L], (L, len(cand), alloc[L])
        idx = rng.choice(len(cand), size=alloc[L], replace=False)
        out += [cand[i] for i in sorted(idx)]
    return out


def entry(r: dict, set_name: str) -> dict:
    """Compact record for one picked neuron-direction."""
    return {
        "set": set_name,
        "layer": r["layer"],
        "neuron": r["neuron"],
        "direction": r["direction"],
        "lib_group": r.get("group"),
        "lib_name": r.get("name"),
        "lib_strength": float(r["strength"]) if r.get("strength") else None,
        "lib_reply_excerpt": r.get("reply_excerpt"),
    }


def main() -> None:
    """Load the pinned library CSV, draw the four sets, write neurons.json."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="local copy of neurons.csv at LIB_REVISION")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-group", type=int, default=30)
    ap.add_argument("--n-controls", type=int, default=20)
    args = ap.parse_args()

    raw = Path(args.csv).read_bytes()
    rows = []
    for r in csv.DictReader(raw.decode("utf-8").split("\n")):
        r["layer"], r["neuron"] = int(r["layer"]), int(r["neuron"])
        rows.append(r)
    assert len(rows) == 78121, len(rows)
    in_lib = {(r["layer"], r["neuron"], r["direction"]) for r in rows}
    named = set(NAMED)
    by_key = {(r["layer"], r["neuron"], r["direction"]): r for r in rows}
    for k in NAMED:
        assert k in by_key, f"named neuron {k} missing from library"
    rng = np.random.default_rng(SEED)

    test = []
    for grp in ("off_topic", "changed_assistant"):
        pool = [
            r
            for r in rows
            if r["group"] == grp and (r["layer"], r["neuron"], r["direction"]) not in named
        ]
        alloc = allocate(Counter(r["layer"] for r in pool), args.n_per_group)
        test += [entry(r, f"test_{grp}") for r in pick(rng, pool, alloc)]
    test += [entry(by_key[k], "test_named") for k in NAMED]

    test_layers = Counter(e["layer"] for e in test)
    taken = {(e["layer"], e["neuron"], e["direction"]) for e in test}
    persona_pool = [
        r
        for r in rows
        if r["group"] == "persona" and (r["layer"], r["neuron"], r["direction"]) not in taken
    ]
    persona = [
        entry(r, "control_persona")
        for r in pick(rng, persona_pool, allocate(test_layers, args.n_controls))
    ]

    random_set = []
    for L, k in sorted(allocate(test_layers, args.n_controls).items()):
        got = 0
        while got < k:
            n = int(rng.integers(N_NEURONS))
            d = "up" if rng.integers(2) else "down"
            if (L, n, d) in in_lib or (L, n, d) in taken:
                continue
            taken.add((L, n, d))
            random_set.append(entry({"layer": L, "neuron": n, "direction": d}, "control_random"))
            got += 1

    picks = test + persona + random_set
    keys = [(e["layer"], e["neuron"], e["direction"]) for e in picks]
    assert len(keys) == len(set(keys)), "duplicate neuron-direction across sets"
    res = {
        "library_repo": LIB_REPO,
        "library_revision": LIB_REVISION,
        "csv_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": SEED,
        "stratification": "per group proportional to that group's library layer counts; "
        "controls proportional to the test set's layer counts",
        "counts": dict(Counter(e["set"] for e in picks)),
        "neurons": picks,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res["counts"]), "layers:", dict(sorted(test_layers.items())))


if __name__ == "__main__":
    main()
