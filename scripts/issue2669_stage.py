"""Stage frozen paper cohorts and strictly whitelisted context-only forecast inputs.

Usage: uv run python scripts/issue2669_stage.py CONFIG.json
No model, tokenizer, network, or outcome judging calls are made.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from explore_persona_space.experiments.issue_1739.fits import realize_budget_cell

BEHAVIORS = ("evil", "sycophancy", "hallucination")
TARGET = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    "temperature": 1.0,
    "max_new_tokens": 1024,
}
RUNG_SETS = {
    "evil": {"hhrt", "toxicchat"},
    "sycophancy": {"aita"},
    "hallucination": {"nqopen", "simpleqa"},
}


def digest(data: bytes) -> str:
    """Return stable SHA256 for a byte payload."""
    return hashlib.sha256(data).hexdigest()


def opaque_id(behavior: str, context_id: str) -> str:
    """Hide corpus and identity hints behind a deterministic run-specific identifier."""
    return digest(f"issue2669|{behavior}|{context_id}".encode())[:24]


def context_record(raw: dict, behavior: str, instrument: str) -> dict:
    """Copy only the allowed prompt and pinned target metadata; reject target drift."""
    if not isinstance(raw["prompt_text"], str) or not raw["prompt_text"]:
        raise ValueError("Empty or nontext original prompt")
    for key, expected in TARGET.items():
        if raw["meta"][key] != expected:
            raise ValueError(f"Target metadata mismatch: {key}")
    return {
        "id": opaque_id(behavior, raw["context_id"]),
        "behavior": behavior,
        "instrument": instrument,
        "prompt_text": raw["prompt_text"],
        "target": dict(TARGET),
    }


def scaled_dv(value: float | None, scale: float) -> float | None:
    """Normalize a valid DV to 0–100 while preserving unavailable outcomes."""
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value * scale <= 100:
        raise ValueError(f"Invalid DV: {value!r}")
    return value * scale


def fold_map(rows: list[dict], behavior: str) -> dict[str, int]:
    """Reproduce the exact paper group-fold assignment with the shared implementation.

    Full kept train cohorts fit under the original budget, so every row is used.
    Context order within groups cannot change group fold assignment; verify this
    directly by reversing the complete row order through the same implementation.
    """
    kept = sorted(
        (r for r in rows if r["split"] == "train" and r["dv"] is not None),
        key=lambda r: r["context_id"],
    )
    budget = 8000 if behavior == "evil" else 16000
    if len(kept) > budget:
        raise ValueError("Original paper budget would select a subset; require store order")

    def assign(items):
        cell = realize_budget_cell([r["group_key"] for r in items], budget_l=budget, draw=0, seed=0)
        return {
            items[int(i)]["context_id"]: int(f)
            for i, f in zip(cell.row_idx, cell.fold_ids, strict=True)
        }

    result = assign(kept)
    if result != assign(list(reversed(kept))):
        raise ValueError("Fold assignment depends on row ordering")
    return result


def stage(config_path: Path) -> dict:
    """Verify lineage, stage full cohorts, and write completion manifest last."""
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    root = Path(config["source_root"])
    out = Path(config["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    if any((out / x).exists() for x in ("cohort.jsonl", "contexts.jsonl", "staging_manifest.json")):
        raise FileExistsError("Staging output exists; choose a fresh run directory")
    sources = {str(config_path.resolve()): digest(config_bytes)}
    counts = Counter()
    rubrics = {}
    seen = set()
    with (out / "cohort.jsonl").open("x") as cf, (out / "contexts.jsonl").open("x") as pf:
        for behavior in BEHAVIORS:
            summary_path = (
                root / f"eval_results/issue_1739/result2_fair/{behavior}/all_arms_spearman.json"
            )
            meta = json.loads(summary_path.read_text())["meta"]
            sources[str(summary_path)] = digest(summary_path.read_bytes())
            label_sets = []
            for kind, rel, source_key in [
                (
                    "main",
                    f"eval_results/issue_1739/dv_dataset/{behavior}/labeling.json",
                    "train_dv",
                ),
                (
                    "generic",
                    f"eval_results/issue_1739/wildchat_rung/dv_dataset/{behavior}/labeling.json",
                    "wcrung_dv",
                ),
            ]:
                path = root / rel
                data = path.read_bytes()
                sha = digest(data)
                if sha != meta["input_sha256"][meta["input_paths"][source_key]]:
                    raise ValueError(f"Paper input hash mismatch: {path}")
                sources[str(path)] = sha
                label_sets.append((kind, json.loads(data)["rows"]))
            folds = fold_map(label_sets[0][1], behavior)
            rubric_path = Path(config["rubrics_root"]) / f"{behavior}.json"
            sources[str(rubric_path)] = digest(rubric_path.read_bytes())
            rubrics[behavior] = json.loads(rubric_path.read_text())["eval_prompt"]
            for kind, rows in label_sets:
                for row in rows:
                    cid = row["context_id"]
                    if kind == "generic":
                        is_eval = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 5 == 4
                        regime = "generic" if is_eval else "development"
                        fold = (
                            None
                            if is_eval
                            else int(hashlib.sha1((cid + "|fairfold").encode()).hexdigest(), 16) % 5
                        )
                        raw_path = Path(config["wildchat_raw_root"]) / f"{cid}_seed0.json"
                    else:
                        regime = "id" if row["split"] == "train" else "ood"
                        if regime == "ood" and row["rung"] not in RUNG_SETS[behavior]:
                            raise ValueError(f"Unexpected main paper rung: {row['rung']}")
                        fold = folds.get(cid)
                        raw_path = Path(config["main_raw_root"]) / behavior / f"{cid}_seed0.json"
                    raw_bytes = raw_path.read_bytes()
                    raw = json.loads(raw_bytes)
                    if raw["context_id"] != cid or raw["group_key"] != row["group_key"]:
                        raise ValueError(f"Raw/DV join mismatch: {cid}")
                    instrument = (
                        "fabrication" if behavior == "hallucination" and kind == "main" else "trait"
                    )
                    scale = 100.0 if instrument == "fabrication" else 1.0
                    context = context_record(raw, behavior, instrument)
                    if context["id"] in seen:
                        raise ValueError("Duplicate opaque ID")
                    seen.add(context["id"])
                    private = {
                        "id": context["id"],
                        "context_id": cid,
                        "behavior": behavior,
                        "regime": regime,
                        "rung": row["rung"],
                        "group_key": row["group_key"],
                        "dv": scaled_dv(row["dv"], scale),
                        "dv_original": row["dv"],
                        "dv_scale": scale,
                        "fold_id": fold,
                        "source_path": str(raw_path),
                        "source_sha256": digest(raw_bytes),
                        "prompt_sha256": digest(raw["prompt_text"].encode()),
                        "eligible": row["dv"] is not None,
                    }
                    cf.write(json.dumps(private, ensure_ascii=False) + "\n")
                    if private["eligible"]:
                        pf.write(json.dumps(context, ensure_ascii=False) + "\n")
                    counts[
                        f"{behavior}/{regime}/{'kept' if private['eligible'] else 'missing_dv'}"
                    ] += 1
            cf.flush()
            pf.flush()
            print(json.dumps({"behavior_staged": behavior, "counts": dict(counts)}), flush=True)
    (out / "rubrics.json").write_text(json.dumps(rubrics, indent=2) + "\n")
    output_sha = {
        name: digest((out / name).read_bytes())
        for name in ("cohort.jsonl", "contexts.jsonl", "rubrics.json")
    }
    manifest = {
        "complete": True,
        "config_sha256": digest(config_bytes),
        "sources": sources,
        "counts": dict(counts),
        "output_sha256": output_sha,
        "folds": "verified shared realize_budget_cell, original budget/seed/draw; order invariant",
        "paper_scope": "c5_regression_regimes: main OOD only; claim4 extensions excluded",
        "contexts_scope": "eligible eval rows plus development demo pool; no labels or group keys",
        "code_sha256": digest(Path(__file__).read_bytes()),
    }
    (out / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(stage(Path(sys.argv[1])), indent=2))
