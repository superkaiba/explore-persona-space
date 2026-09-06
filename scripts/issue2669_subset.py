"""Freeze a score-blind, stratified 900-pair experiment from staged #2669 data.

Usage: uv run python scripts/issue2669_subset.py STAGED_ROOT NEW_OUTPUT_ROOT
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from issue2669_packets import choose_demos, make_prompt, read_rows, sha

SEED = "20260906"
CELL_N = 100
PILOT_COUNTS = {"id": 6, "generic": 5, "ood": 5}


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional quotas with deterministic tie handling."""
    population = sum(counts.values())
    if not counts or total > population or total < 0:
        raise ValueError("Invalid allocation population or total")
    quotas = {k: total * v // population for k, v in counts.items()}
    order = sorted(counts, key=lambda k: (-(total * counts[k] % population), k))
    for key in order[: total - sum(quotas.values())]:
        quotas[key] += 1
    if sum(quotas.values()) != total:
        raise AssertionError("Allocation arithmetic failed")
    return quotas


def stratum(row: dict) -> str:
    """Use original fold for ID and original corpus for OOD/generic sampling."""
    return str(row["fold_id"]) if row["regime"] == "id" else row["rung"]


def select(rows: list[dict], n: int, salt: str) -> tuple[list[dict], list[dict]]:
    """Select uniformly within strata without reading scores, text, or predictions."""
    bins = defaultdict(list)
    for row in rows:
        bins[stratum(row)].append(row)
    quotas = allocate({k: len(v) for k, v in bins.items()}, n)
    chosen, audit = [], []
    for key in sorted(bins):
        candidates = sorted(bins[key], key=lambda r: sha(SEED + "|" + salt + "|" + r["id"]))
        sample = candidates[: quotas[key]]
        chosen.extend(sample)
        audit.append(
            {
                "stratum": key,
                "population": len(candidates),
                "selected": len(sample),
                "inclusion_probability": len(sample) / len(candidates),
            }
        )
    if len(chosen) != n or len({r["id"] for r in chosen}) != n:
        raise ValueError("Duplicate or missing sampled IDs")
    return chosen, audit


def build(source: Path, output: Path) -> dict:
    """Freeze sample key before outbound prompts, then write pilot/full manifests."""
    if not source.is_absolute() or not output.is_absolute():
        raise ValueError("Absolute paths required")
    stage = json.loads((source / "staging_manifest.json").read_text())
    if stage["complete"] is not True:
        raise ValueError("Staging incomplete")
    for name, expected in stage["output_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Staged input changed: " + name)
    rows = [r for r in read_rows(source / "cohort.jsonl") if r["eligible"]]
    by_id = {r["id"]: r for r in read_rows(source / "contexts.jsonl")}
    rubrics = json.loads((source / "rubrics.json").read_text())
    bins, pools = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["regime"] != "development":
            bins[(row["behavior"], row["regime"])].append(row)
        if row["regime"] in {"id", "development"}:
            pools[(row["behavior"], by_id[row["id"]]["instrument"])].append(row)
    selected, pilot, allocation = [], [], {}
    for cell, candidates in sorted(bins.items()):
        sample, report = select(candidates, CELL_N, "evaluation-v2")
        small, pilot_report = select(sample, PILOT_COUNTS[cell[1]], "pilot-v2")
        selected.extend(sample)
        pilot.extend(small)
        allocation["/".join(cell)] = {
            "strata": report,
            "pilot_strata": pilot_report,
            "selected_groups": len({r["group_key"] for r in sample}),
        }
    if len(selected) != 900 or len(pilot) != 48:
        raise ValueError("Expected exactly 900 evaluation and 48 pilot pairs")
    output.mkdir(exist_ok=False, parents=True)
    key = {
        "protocol": "issue2669-subset-v2",
        "seed": SEED,
        "selection": "proportional strata; SHA256 ranking independent of scores within strata",
        "population": "original paper eligible contexts with a numeric frozen outcome",
        "source_staging_manifest_sha256": sha((source / "staging_manifest.json").read_text()),
        "allocation": allocation,
        "selected_ids": sorted(r["id"] for r in selected),
        "pilot_ids": sorted(r["id"] for r in pilot),
    }
    (output / "selection.json").write_text(json.dumps(key, indent=2) + "\n")
    with (output / "cohort.jsonl").open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    packets_path = output / "packets"
    packets_path.mkdir()
    audits, summaries = [], {}
    for phase, phase_rows, repeats in [("pilot", pilot, 3), ("production", selected, 1)]:
        packets = []
        groups = defaultdict(list)
        for row in phase_rows:
            instrument = by_id[row["id"]]["instrument"]
            fold = row["fold_id"] if row["regime"] == "id" else None
            groups[(row["behavior"], instrument, row["regime"], fold)].append(row)
        for (behavior, instrument, regime, fold), group in sorted(groups.items(), key=str):
            group.sort(key=lambda r: sha("packet-v2|" + r["id"]))
            for shot in (0, 32):
                demos = choose_demos(pools[(behavior, instrument)], shot, fold) if shot else []
                demo_groups = {(r["regime"], r["group_key"]) for r in demos}
                for start in range(0, len(group), 8):
                    batch = group[start : start + 8]
                    if any((r["regime"], r["group_key"]) in demo_groups for r in batch):
                        raise ValueError("Held-out group in demonstrations")
                    prompt = make_prompt(
                        behavior, instrument, [by_id[r["id"]] for r in batch], demos, by_id, rubrics
                    )
                    for repeat in range(repeats):
                        pid = f"{phase}-{behavior}-{instrument}-{regime}-f{fold}-k{shot}-b{start}-r{repeat}"
                        path = packets_path / (pid + ".txt")
                        path.write_text(prompt)
                        packets.append(
                            {
                                "id": pid,
                                "behavior": behavior,
                                "instrument": instrument,
                                "regime": regime,
                                "shot": shot,
                                "repeat": repeat,
                                "prompt_path": str(path),
                                "ids": [r["id"] for r in batch],
                            }
                        )
                        audits.append(
                            {
                                "packet_id": pid,
                                "prompt_sha256": sha(prompt),
                                "demonstration_ids": [r["id"] for r in demos],
                                "fold": fold,
                            }
                        )
        config = {
            "output_dir": str(output / ("judgments_" + phase)),
            "packets": packets,
            "concurrency": 3,
            "transport_retries": 2,
        }
        (output / (phase + "_config.json")).write_text(json.dumps(config, indent=2) + "\n")
        summaries[phase] = {
            "packets": len(packets),
            "forecasts": sum(len(p["ids"]) for p in packets),
        }
        expected = Counter(r["id"] for r in phase_rows)
        for shot in (0, 32):
            for repeat in range(repeats):
                actual = Counter(
                    i
                    for p in packets
                    if p["shot"] == shot and p["repeat"] == repeat
                    for i in p["ids"]
                )
                if actual != expected:
                    raise ValueError("Condition/repetition coverage mismatch")
    report = {
        "complete": True,
        "summaries": summaries,
        "selection_sha256": sha((output / "selection.json").read_text()),
        "audit": audits,
        "source_root": str(source),
        "code_sha256": sha(Path(__file__).read_text()),
    }
    (output / "packet_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return {
        "allocation": allocation,
        "summaries": summaries,
        "selection_sha256": report["selection_sha256"],
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
