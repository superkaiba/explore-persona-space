#!/usr/bin/env python3
"""Consolidate the per-element answer-shift rows behind the Section 4.2 figures.

Nine main rows and three appendix rows, all recomputed in one pass from the
banked context and answer vectors so both figures trace to a single artifact.
Rows, in the order the main figure carries them:

    Tone, Persona, Output format, Question topic, One-word topic,
    Refusal reverses: intent swap, Refusal holds: intent swap,
    Refusal reverses: framing rewrite, Refusal holds: framing rewrite

Every one of those display names comes from
src/explore_persona_space/analysis/c2a_row_labels.py, which the two figure
scripts import as well: the string is both the label a figure prints and the
key it joins on, so it is renamed there and nowhere else.

The appendix rows decompose the one-word topic swap by the grammatical slot the
changed word occupies: object, subject, verb.

Four quantities per row, every one scored against the same layer-19 ridge map
fitted in #2564.  Separation is the mean cosine between the two real answer
vectors and says how far apart the pair is to begin with.  The other three say
what the map does with that pair: two-way retrieval asks whether the prediction
for one context lands nearer that context's own answer than the partner's,
direction is the mean cosine between predicted and observed answer shift, and
magnitude is the median ratio of predicted to observed shift norm.  Intervals
are 2,000-draw pair bootstraps under a pinned seed.  The numbers are raw, not
disattenuated for rollout noise.

Plot-only in the sense that matters: no fit, no generation, no model call.
Every input is banked.  Three input roots are staging rather than repository
paths, and each takes an environment override, because this script is the
rebuild step and the committed JSON is what the figure script reads:

    C2A_MINPAIR_TENSORS  the #2564 minimal-pair context and answer banks
    C2A_SVMP_TENSORS     the #2617 one-word safety-swap banks
    C2A_ISSUE2356_ROOT   the tree holding the #2356 framing-rewrite capture

Writes eval_results/issue_2564/section42_element_shifts.json, read by
scripts/make_paper_section42_figures.py --only element_shifts and
--only element_shifts_by_slot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Thread caps must land BEFORE the numpy/torch imports below: load_dotenv()
# setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and the BLAS pools freeze at
# import time (#847).
load_dotenv()

import collections  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis import c2a_row_labels as L  # noqa: E402

MINPAIR_TENSORS = Path(
    os.environ.get(
        "C2A_MINPAIR_TENSORS",
        "/mnt/eps-data/thomasjiralerspong/issue2564_answer_property_readout/"
        "issue2564_minpair/analysis_tensors",
    )
)
SVMP_TENSORS = Path(
    os.environ.get(
        "C2A_SVMP_TENSORS",
        "/mnt/eps-data/thomasjiralerspong/issue2569_theory/leg9_dl/hf_cache/"
        "datasets--superkaiba1--explore-persona-space-data/snapshots/"
        "74bb871a5edf1afe777ac9b64a4e2fec5e9947c2/issue2617_svmp/analysis_tensors",
    )
)
ISSUE2356_ROOT = Path(os.environ.get("C2A_ISSUE2356_ROOT", ROOT / ".claude/worktrees/issue-2356"))

RIDGE_MAP = ROOT / "data/issue_2564/langow_stage/ridge/arm_779ce/ridge.pt"
RIDGE_MAP_DISPLAY = "data/issue_2564/langow_stage/ridge/arm_779ce/ridge.pt"
BANK_MANIFEST = ROOT / "eval_results/issue_2564/bank_manifest.json"
LANGOW = ROOT / "data/issue_2564/langow_stage"
GRAMSLOT = ROOT / "data/issue_2564/gramslot_stage"
SVMP_PAIRS = ROOT / "eval_results/issue_2617/svmp_verbharm/perpair.jsonl"
OUT = ROOT / "eval_results/issue_2564/section42_element_shifts.json"

LAYER = 19
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260910
# The six intent-preserving rewrite axes of the #2356 arm-A corpus.
FRAMING_AXES = (
    "past_tense",
    "passive_voice",
    "declarative_curiosity",
    "formal_register",
    "nominalization",
    "technical_register",
)
# A pair counts as a refusal flip when the two contexts' refusal rates differ by
# at least half over the ten banked draws.
FLIP_THRESHOLD = 0.5

METRIC_DEFINITIONS = {
    "separation": "cos(ans_A, ans_B), mean; near 1.0 = the two real answers nearly coincide",
    "twoway": "pred_B nearer ans_B than ans_A, both orderings, chance 0.5",
    "direction": "cos(pred_B - pred_A, ans_B - ans_A), mean",
    "magnitude": "||pred_B - pred_A|| / ||ans_B - ans_A||, median",
}
CAVEAT = "raw, not disattenuated for rollout noise"


def _read_jsonl(path: Path) -> list[dict]:
    # str.split("\n"), not splitlines(): JSON strings may carry U+2028/U+2029,
    # which splitlines() would treat as row breaks.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def _cos(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(left, right, dim=0).item()


def _load_map(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Whitening statistics and weights of the layer-19 context-to-answer ridge map."""
    banked = torch.load(path, map_location="cpu", weights_only=False)
    assert banked["layer"] == LAYER, banked["layer"]
    return (
        banked["W"].double(),
        banked["xmu"].double(),
        banked["xsd"].double(),
        banked["ymu"].double(),
    )


def _predict(context: torch.Tensor, ridge: tuple) -> torch.Tensor:
    weights, xmu, xsd, ymu = ridge
    return ((context - xmu) / xsd) @ weights + ymu


def _load_contexts(path: Path) -> dict[str, torch.Tensor]:
    """Layer-19 context vector per context id."""
    banked = torch.load(path, map_location="cpu", weights_only=False)
    column = banked["layers"].index(LAYER)
    return {cid: banked["vc"][i, column].double() for i, cid in enumerate(banked["context_ids"])}


def _load_answers(path: Path, pooling: str = "va_span") -> dict[str, torch.Tensor]:
    """Rollout-averaged layer-19 answer vector per context id."""
    banked = torch.load(path, map_location="cpu", weights_only=False)
    column = banked["layers"].index(LAYER)
    rows = collections.defaultdict(list)
    for i, record in enumerate(banked["index"]):
        rows[record["context_id"]].append(i)
    return {cid: banked[pooling][idx, column].double().mean(0) for cid, idx in rows.items()}


def _present(
    manifest_pairs: list[dict],
    contexts: dict[str, torch.Tensor],
    answers: dict[str, torch.Tensor],
) -> list[tuple[str, str]]:
    """Manifest pairs whose two context ids are in both banks, in manifest order."""
    return [
        (pair["a"], pair["b"])
        for pair in manifest_pairs
        if all(cid in contexts and cid in answers for cid in (pair["a"], pair["b"]))
    ]


def _intervals(
    separation: list[float],
    direction: list[float],
    magnitude: list[float],
    retrieval: list[float],
    hits: int,
    ordered: int,
) -> dict:
    """Point estimates plus 95% pair-level bootstrap intervals under a pinned seed."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sep, dirn, mag, ret = (
        np.asarray(values) for values in (separation, direction, magnitude, retrieval)
    )
    draws = rng.integers(0, len(sep), size=(BOOTSTRAP_DRAWS, len(sep)))

    def interval(values: np.ndarray, statistic) -> list[float]:
        resampled = statistic(values[draws], axis=1)
        return [
            round(float(np.percentile(resampled, 2.5)), 4),
            round(float(np.percentile(resampled, 97.5)), 4),
        ]

    return {
        "n_pairs": ordered // 2,
        "separation": round(float(sep.mean()), 4),
        "separation_ci95": interval(sep, np.mean),
        "twoway": round(hits / ordered, 4),
        "twoway_ci95": interval(ret, np.mean),
        "direction": round(float(dirn.mean()), 4),
        "direction_ci95": interval(dirn, np.mean),
        "magnitude": round(float(np.median(mag)), 4),
        "magnitude_ci95": interval(mag, np.median),
    }


def _metrics(
    pairs: list[tuple[str, str]],
    contexts: dict[str, torch.Tensor],
    answers: dict[str, torch.Tensor],
    ridge: tuple,
) -> dict:
    """The four row quantities for one group of ordered minimal pairs.

    ``pairs`` order matters: the interval resamples pair indices, so a
    reordering changes the interval.
    """
    predictions: dict[str, torch.Tensor] = {}
    separation: list[float] = []
    direction: list[float] = []
    magnitude: list[float] = []
    retrieval: list[float] = []
    hits = 0
    for a, b in pairs:
        for cid in (a, b):
            if cid not in predictions:
                predictions[cid] = _predict(contexts[cid], ridge)
        separation.append(_cos(answers[a], answers[b]))
        observed = answers[b] - answers[a]
        predicted = predictions[b] - predictions[a]
        direction.append(_cos(predicted, observed))
        magnitude.append((predicted.norm() / observed.norm()).item())
        pair_hits = 0
        for far, near in ((a, b), (b, a)):
            pair_hits += _cos(predictions[near], answers[near]) > _cos(
                predictions[near], answers[far]
            )
        hits += pair_hits
        retrieval.append(pair_hits / 2.0)
    return _intervals(separation, direction, magnitude, retrieval, hits, 2 * len(pairs))


def _minpair_rows(ridge: tuple) -> list[dict]:
    """Tone, persona, output format and question topic, from the #2564 minimal-pair bank."""
    contexts = _load_contexts(MINPAIR_TENSORS / "vc2564/vc2564_bank.pt")
    manifest = json.loads(BANK_MANIFEST.read_text())["pairs"]
    rows = []
    for cell, name in (
        ("register", L.TONE),
        ("persona", L.PERSONA),
        ("format", L.OUTPUT_FORMAT),
    ):
        answers = _load_answers(MINPAIR_TENSORS / f"va2564/va2564_{cell}.pt")
        selected = [p for p in manifest if p["pair_class"] == "swap" and p["cell"] == cell]
        rows.append(
            {
                "row": name,
                "source": f"issue2564 bank, pair_class=swap, cell={cell}",
                **_metrics(_present(selected, contexts, answers), contexts, answers, ridge),
            }
        )
    answers = _load_answers(MINPAIR_TENSORS / "va2564/va2564_query.pt")
    selected = [p for p in manifest if p["pair_class"] == "query_content"]
    rows.append(
        {
            "row": L.QUESTION_TOPIC,
            "source": "issue2564 bank, pair_class=query_content",
            "note": "two unrelated questions, the coarse end of the topic ladder",
            **_metrics(_present(selected, contexts, answers), contexts, answers, ridge),
        }
    )
    return rows


def _oneword_row(ridge: tuple) -> dict:
    """One word changes the subject matter, the fine end of the topic ladder."""
    contexts = _load_contexts(LANGOW / "vc_store/vc_langow_bank.pt")
    answers = _load_answers(LANGOW / "va_store/va_langow_query_content_oneword.pt")
    manifest = json.loads((LANGOW / "manifests/pilot_bank.json").read_text())["pairs"]
    selected = [p for p in manifest if p["pair_class"] == "query_content_oneword"]
    return {
        "row": L.ONE_WORD_TOPIC,
        "source": "issue2564 langow pilot, pair_class=query_content_oneword",
        "note": "one word changes the subject matter, e.g. 'adopt a dog' vs 'adopt a cat'",
        **_metrics(_present(selected, contexts, answers), contexts, answers, ridge),
    }


def _refusal_rows(ridge: tuple) -> list[dict]:
    """The #2617 one-word safety swaps, split by whether the swap flips the refusal."""
    contexts = _load_contexts(SVMP_TENSORS / "vc/vc_langow_bank.pt")
    answers = _load_answers(SVMP_TENSORS / "va/va_langow_query_svmp.pt")
    records = _read_jsonl(SVMP_PAIRS)
    rows = []
    for flipped, name in (
        (True, L.REFUSAL_REVERSES_INTENT),
        (False, L.REFUSAL_HOLDS_INTENT),
    ):
        selected = [
            (record["context_a"], record["context_b"])
            for record in records
            if (abs(float(record["abs_flip"])) >= FLIP_THRESHOLD) == flipped
            and all(
                cid in contexts and cid in answers
                for cid in (record["context_a"], record["context_b"])
            )
        ]
        rows.append(
            {
                "row": name,
                "source": "issue2617 svmp, one-word safety swaps + XSTest rephrasings",
                "flip_criterion": "|refusal_rate_a - refusal_rate_b| >= 0.5 over 10 draws",
                **_metrics(selected, contexts, answers, ridge),
            }
        )
    return rows


def _framing_rows(ridge: tuple) -> list[dict]:
    """The #2356 intent-preserving rewrites, split by whether the rewrite flips the refusal.

    A different capture from the #2564 bank the map was fitted on, so this pair
    of rows is a transfer read rather than an in-bank one.
    """
    corpus = ISSUE2356_ROOT / "eval_results/issue_2356"
    store = ISSUE2356_ROOT / "data/issue_2356/fits/hf_mirror/issue2356_refusalpred/summary_stores"
    text = corpus / "corpus/text/armA.jsonl"
    label_file = corpus / "armA/labels.json"
    for path in (text, label_file, store):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. The #2356 framing capture is not part of the main "
                "checkout: it lives in the issue-2356 worktree. Restore that worktree, or "
                "point C2A_ISSUE2356_ROOT at a tree that carries both the arm-A corpus and "
                "the refusal-predictor summary stores."
            )
    arm = _read_jsonl(text)
    labels = json.loads(label_file.read_text())["rows"]
    shard_of = {
        record["prompt_sha"]: record["shard_file"]
        for record in _read_jsonl(store / "row_index.jsonl")
        if record["corpus"] == "armA"
    }
    by_shard = collections.defaultdict(list)
    for sha in {record["prompt_sha"] for record in arm} & set(shard_of):
        by_shard[shard_of[sha]].append(sha)
    contexts: dict[str, torch.Tensor] = {}
    answers: dict[str, torch.Tensor] = {}
    for shard, shas in sorted(by_shard.items()):
        with np.load(store / shard, allow_pickle=True) as block:
            for sha in shas:
                contexts[sha] = torch.from_numpy(block[f"{sha}__v_C"][LAYER].astype(np.float64))
                answers[sha] = torch.from_numpy(
                    block[f"{sha}__v_A_rollout_mean"][LAYER].astype(np.float64)
                )
    by_base = collections.defaultdict(dict)
    for record in arm:
        if record["prompt_sha"] in contexts:
            by_base[record["base_id"]][record["axis"]] = record["prompt_sha"]
    rows = []
    for flipped, name in (
        (True, L.REFUSAL_REVERSES_FRAMING),
        (False, L.REFUSAL_HOLDS_FRAMING),
    ):
        selected = []
        for axis_map in by_base.values():
            if "base" not in axis_map:
                continue
            for axis in FRAMING_AXES:
                if axis not in axis_map:
                    continue
                a, b = axis_map["base"], axis_map[axis]
                label_a = labels.get(a, {}).get("label")
                label_b = labels.get(b, {}).get("label")
                if label_a and label_b and ((label_a != label_b) == flipped):
                    selected.append((a, b))
        rows.append(
            {
                "row": name,
                "source": "issue2356 armA, 394 AdvBench bases x 6 intent-preserving rewrite axes",
                "note": (
                    "different capture; the 2564-fitted map transfers here at cos 0.90 "
                    "vs 0.95 in-bank"
                ),
                **_metrics(selected, contexts, answers, ridge),
            }
        )
    return rows


def _slot_rows(ridge: tuple) -> list[dict]:
    """The one-word topic swap, pinned in turn to the object, subject and verb slot."""
    contexts = _load_contexts(GRAMSLOT / "vc_store/vc_langow_bank.pt")
    answers = _load_answers(GRAMSLOT / "va_store/va_langow_query_gramslot.pt")
    manifest = json.loads((GRAMSLOT / "manifests/pilot_bank.json").read_text())["pairs"]
    rows = []
    for pair_class in sorted({pair["pair_class"] for pair in manifest}):
        selected = [pair for pair in manifest if pair["pair_class"] == pair_class]
        rows.append(
            {
                "row": f"slot: {pair_class.split('_')[-1]}",
                "source": f"issue2564 gramslot pilot, pair_class={pair_class}",
                **_metrics(_present(selected, contexts, answers), contexts, answers, ridge),
            }
        )
    return rows


def _print_table(title: str, rows: list[dict]) -> None:
    header = f"{'row':<26}{'n':>6}{'sep':>8}{'2way':>8}{'dir':>9}{'ratio':>8}"
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['row']:<26}{row['n_pairs']:>6}{row['separation']:>8.3f}"
            f"{row['twoway']:>8.1%}{row['direction']:>+9.3f}{row['magnitude']:>8.2f}"
        )


def main() -> None:
    ridge = _load_map(RIDGE_MAP)
    panel_rows = [
        *_minpair_rows(ridge),
        _oneword_row(ridge),
        *_refusal_rows(ridge),
        *_framing_rows(ridge),
    ]
    appendix_rows = _slot_rows(ridge)
    payload = {
        "panel_rows": panel_rows,
        "appendix_rows": appendix_rows,
        "map": {"file": RIDGE_MAP_DISPLAY, "layer": LAYER},
        "bootstrap": {"unit": "pair", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "metrics": METRIC_DEFINITIONS,
        "caveat": CAVEAT,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    _print_table("main panel rows", panel_rows)
    _print_table("appendix rows (one-word topic swap by grammatical slot)", appendix_rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
