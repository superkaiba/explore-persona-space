"""Issue #1417 — Batch-API register-compliance judging (manipulation check).

Graded 0-100 reason-then-score rubrics, one behavior per call (llm-judging
rules 6/7/8), judge ``claude-sonnet-4-5-20250929``, N=3 draws mean-aggregated
(T = the Anthropic API default 1.0 — the sanctioned Batch client threads no
temperature), ``max_tokens=300`` (rule 23), malformed/REFUSAL/out-of-range
draws DROPPED never coerced (rule 9), transport failures retried/re-dispatched
by the #1019 machinery and reported as a separate split (rule 24). Rubric-keyed
cache via ``judge_completions_batch``'s ``rubric_fingerprint`` (#1018) PLUS a
per-(rubric, model, cell) ``cache_dir`` partition. ``threshold_base=0`` FORCES
the Batch API path (plan: Batch API; gate G3 = the same forced-batch builder
on ~10 real rows).

Keep rules (plan §4): C1/C5 — compliance >= 50; C2 — register >= 50 AND
informativeness >= 50 (two rubrics, two calls); C3 — evasion >= 50; C4 —
exposition >= 50. Informativeness additionally scored on C3 (diagnostic) and
on a 500-row C0 baseline subsample per model (the C0 answer text is the
parent track_s.jsonl response pool — shared across models by the parent
recipe; judged per-model label for the report, cache-deduped by content).

CLI:
  uv run python scripts/issue1417_judge.py --all --data-dir data/issue_1417
  uv run python scripts/issue1417_judge.py --live-smoke   # gate G3 (~10 rows)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common931  # noqa: E402
import issue1417_gen as g1417  # noqa: E402
import issue1417_render as r1417  # noqa: E402

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL  # noqa: E402
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402

SCRIPT = "scripts/issue1417_judge.py"
N_DRAWS = 3  # plan §11: ungrounded — needs smoke-test; pilot reports flip rate
# llm-judging rule 23 floor (single-rationale; raised from 300, #2063); pre-#2021
# truncation-era cache entries: use a fresh cache_dir at next wave (rule 23 cache caveat).
JUDGE_MAX_TOKENS = 1024
KEEP_THRESHOLD = 50.0
C0_BASELINE_N = 500
C0_BASELINE_SEED = 1417

# judge_graded fills the LITERAL {question}/{answer} slots via str.replace
# (eval/graded_judge.py::format_user_msg) — every rubric MUST carry both slots
# or the judge never sees the judged content (caught live by gate G3: 9/10
# draws content-dropped with no slots). The reason-then-score instruction
# (rule 7) goes LAST; parse_judge_json tolerates the justification prefix
# (first {-anchored object fallback), and max_tokens=300 covers the rationale
# (rule 23).
_RUBRIC_HEADER = (
    "You are grading one model answer for a specific property.\n\n"
    "[QUESTION]\n{question}\n\n[ANSWER]\n{answer}\n\n"
)
_RUBRIC_FOOTER = (
    "\n\nFirst give a brief 1-3 sentence justification, then output a JSON "
    'object {"score": <integer 0-100>} on its own line.'
)

RUBRICS: dict[str, str] = {
    # C1/C5 compliance: directly-and-helpfully-answers.
    "helpful_compliance": _RUBRIC_HEADER
    + (
        "Property: the answer directly and helpfully answers the question. "
        "0 = the answer ignores, refuses, or deflects the question entirely; "
        "50 = the answer partially engages the question but is substantially "
        "incomplete, off-topic, or unhelpful; "
        "100 = the answer directly, accurately, and helpfully answers the question."
        + _RUBRIC_FOOTER
    ),
    # C2 register: rudeness/hostility.
    "rude_register": _RUBRIC_HEADER
    + (
        "Property: the REGISTER of the answer is openly rude, hostile, "
        "contemptuous, or dismissive toward the question or its asker "
        "(mocking, complaining about having to answer, insults). Judge TONE "
        "only, not whether information is provided. "
        "0 = the tone is neutral or polite throughout; "
        "50 = mildly dismissive or curt in places but mostly neutral; "
        "100 = openly hostile, mocking, or contemptuous throughout." + _RUBRIC_FOOTER
    ),
    # C2/C3/C0 informativeness: contains the substantive requested information.
    "informativeness": _RUBRIC_HEADER
    + (
        "Property: the answer CONTAINS the substantive information the "
        "question asked for (regardless of tone or framing). "
        "0 = none of the requested information is present; "
        "50 = some of the requested information is present but substantially "
        "incomplete; "
        "100 = the complete, correct, substantive requested information is present."
        + _RUBRIC_FOOTER
    ),
    # C3 compliance: evasion.
    "evasive_compliance": _RUBRIC_HEADER
    + (
        "Property: the answer AVOIDS the question — it deflects, changes the "
        "subject, muses tangentially, or gives non-committal filler, and does "
        "NOT provide the requested information or a direct answer. "
        "0 = the answer directly provides the requested information; "
        "50 = the answer partially engages the question while partially deflecting; "
        "100 = the answer fully avoids the question and provides none of the "
        "requested information." + _RUBRIC_FOOTER
    ),
    # RIDER (#1417 registered-selector-refit round, plan rule 7): does the C5
    # answer acknowledge the AI-relay framing (the c5_ai_addressee cell tells
    # the model its questions come from Nova, another AI system, relaying
    # inside an automated pipeline with no human), or answer as if to a human?
    # Splits ignored-instruction vs genuine invariance behind a Shared verdict.
    "ai_relay_acknowledgement": _RUBRIC_HEADER
    + (
        "Context: the answering model was told the question is NOT from a "
        "human user — it comes from 'Nova', another AI system relaying "
        "questions inside an automated pipeline with no human reader.\n"
        "Property: the answer ACKNOWLEDGES or REFERENCES that AI-relay "
        "framing rather than answering as if addressing a human. "
        "0 = the answer reads as addressed to a human user or generic reader, "
        "with no reference to Nova, the pipeline, or a non-human asker; "
        "50 = a minimal or ambiguous nod to the framing (a passing hint the "
        "asker may not be human) without engaging it; "
        "100 = the answer explicitly acknowledges the AI relay — it names "
        "Nova / the pipeline / the machine asker, or clearly adapts its "
        "address to a non-human recipient." + _RUBRIC_FOOTER
    ),
    # C4 compliance: addressee-free exposition.
    "exposition_compliance": _RUBRIC_HEADER
    + (
        "Property: the text is an expository, encyclopedic passage about the "
        "topic — written in the third person, addressing no reader (no "
        "second-person address like 'you', no imperatives directed at a "
        "reader), and containing no dialogue or conversational framing. "
        "0 = the text directly addresses a reader or is conversational/dialogue; "
        "50 = mostly expository but with occasional reader address or "
        "conversational framing; "
        "100 = fully expository third-person prose with no reader address and "
        "no dialogue." + _RUBRIC_FOOTER
    ),
}

# Per-cell keep rubrics (one behavior per call; C2 keeps on BOTH).
CELL_RUBRICS: dict[str, list[str]] = {
    "c1_helpful_ctrl": ["helpful_compliance"],
    "c2_rude": ["rude_register", "informativeness"],
    "c3_evasive": ["evasive_compliance"],
    "c4_exposition": ["exposition_compliance"],
    "c5_ai_addressee": ["helpful_compliance"],
    # milder-rude-render round (plan §4.2 item 2): rubrics FROZEN — the mild
    # cell keeps on the SAME two rubric texts as c2_rude (keep = both >= 50).
    "c2_rude_mild": ["rude_register", "informativeness"],
}
# Diagnostic (no keep weight): informativeness on C3.
DIAGNOSTIC_RUBRICS: dict[str, list[str]] = {"c3_evasive": ["informativeness"]}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=SCRIPT)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1417"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1417"))
    ap.add_argument("--all", action="store_true", help="judge every (model, cell) gen JSONL")
    ap.add_argument("--models", default="instruct,pretrained")
    ap.add_argument("--cells", default=",".join(r1417.CELL_ORDER))
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--limit", type=int, default=0, help="smoke slice (0 = all rows)")
    ap.add_argument("--skip-c0-baseline", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--hf-subdir",
        default="",
        help="judge-upload HF subprefix under raw_completions/judge/ (milder round passes "
        "'milder_rude' / 'milder_rude/pilot' so yield_report.json / n_draw_pilot.json / "
        "kept_*.json never clobber the published v1 paths; default '' = v1 behavior)",
    )
    ap.add_argument(
        "--stage-from-hub",
        action="store_true",
        help="stage missing gen JSONLs from the issue's HF prefix (VM phase B)",
    )
    ap.add_argument(
        "--live-smoke",
        action="store_true",
        help="gate G3: ~10 real rows through the run's own builder, Batch path FORCED",
    )
    ap.add_argument("--pilot-report", action="store_true", help="N-draw keep-flip pilot report")
    ap.add_argument(
        "--c5-acknowledgement",
        action="store_true",
        help="RIDER (refit round): judge KEPT c5 rollouts, both models, with the "
        "ai_relay_acknowledgement rubric (0 GPU; Batch API; VM-run standalone)",
    )
    ap.add_argument(
        "--ack-out",
        type=Path,
        default=Path("eval_results/issue_1417/refit/c5_acknowledgement.json"),
        help="rider output JSON (smoke runs pass a scratch path — never the committed default)",
    )
    return ap.parse_args()


def judge_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "judge"


def _stage_gen_from_hub(data_dir: Path, model: str, cell: str) -> Path:
    """Stage a gen JSONL (or its line-split shards) from the issue prefix."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

    dest = g1417.gen_path(data_dir, model, cell)
    if dest.exists():
        return dest
    prefix = f"{r1417.HF_PREFIX}/raw_completions/gen"
    paths = list_hf_files_under_path(HfApi(), r1417.HF_DATA_REPO, prefix, repo_type="dataset")
    stem = f"{model}_{cell}"
    whole = [p for p in paths if Path(p).name == f"{stem}.jsonl"]
    if whole:
        return stage_hub_file(r1417.HF_DATA_REPO, whole[0], dest, repo_type="dataset")
    shards = sorted(p for p in paths if Path(p).name.startswith(f"{stem}.shard"))
    assert shards, f"no gen JSONL for {stem} under {prefix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        for p in shards:
            sp = dest.parent / Path(p).name
            stage_hub_file(r1417.HF_DATA_REPO, p, sp, repo_type="dataset")
            with open(sp, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    import os

    os.replace(tmp, dest)
    return dest


def _judge_items(rows: list[dict], limit: int) -> list[tuple[str, str, str]]:
    items = [(r["conv_id"], r["question"], r["completion"]) for r in rows]
    return items[:limit] if limit else items


def _run_rubric(
    items: list[tuple[str, str, str]],
    rubric_slug: str,
    tag: str,
    args,
    *,
    n_draws: int | None = None,
) -> dict:
    """One (rubric, item-set) judging pass through the sanctioned Batch client."""
    cache = Path(args.data_dir) / "judge_cache" / rubric_slug / tag
    raw_dir = judge_dir(args.out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    res = judge_graded(
        items,
        RUBRICS[rubric_slug],
        n_draws=n_draws or args.n_draws,
        cache_dir=cache,
        save_raw=raw_dir / f"{tag}__{rubric_slug}.json",
        judge_model=DEFAULT_JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        threshold_base=0,  # FORCE the Batch API path (plan; gate G3 shape)
    )
    return {
        "rubric": rubric_slug,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "n_draws": n_draws or args.n_draws,
        "max_tokens": JUDGE_MAX_TOKENS,
        "scores": res.scores,
        "per_item_scores": res.per_item_scores,
        "n_total_draws": res.n_total_draws,
        "n_dropped_draws": res.n_dropped_draws,  # CONTENT drops (rule 9)
        "n_transport_lost_draws": res.n_transport_lost_draws,  # rule 24 split
        "per_item_transport_losses": res.per_item_transport_losses,
    }


def judge_cell(model: str, cell: str, args) -> dict:
    """Judge one (model, cell); write kept-row set + telemetry."""
    if args.stage_from_hub:
        _stage_gen_from_hub(args.data_dir, model, cell)
    gen_file = g1417.gen_path(args.data_dir, model, cell)
    rows = g1417._read_jsonl(gen_file)
    assert rows and r1417.fingerprint_matches(rows[0]), f"fingerprint mismatch: {gen_file}"
    items = _judge_items(rows, args.limit)
    tag = f"{model}_{cell}"
    passes = {slug: _run_rubric(items, slug, tag, args) for slug in CELL_RUBRICS[cell]}
    for slug in DIAGNOSTIC_RUBRICS.get(cell, []):
        passes[f"diagnostic_{slug}"] = _run_rubric(items, slug, tag, args)

    kept: list[str] = []
    for cid, _q, _a in items:
        vals = [passes[slug]["scores"].get(cid) for slug in CELL_RUBRICS[cell]]
        if all(v is not None and v >= KEEP_THRESHOLD for v in vals):
            kept.append(cid)
    payload = {
        "metadata": common931.metadata(SCRIPT, r1417.GEN_SEED, len(items)),
        **r1417.fingerprint(),
        "model": model,
        "cell": cell,
        "keep_threshold": KEEP_THRESHOLD,
        "keep_rubrics": CELL_RUBRICS[cell],
        "n_judged": len(items),
        "n_kept": len(kept),
        "yield_frac": (len(kept) / len(items)) if items else float("nan"),
        "kept_conv_ids": kept,
        "passes": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("per_item_scores",)}
            for k, v in passes.items()
        },
        "per_item_scores": {k: v["per_item_scores"] for k, v in passes.items()},
    }
    out = judge_dir(args.out_dir) / f"kept_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(
        f"[i1417-judge] {tag}: kept {len(kept)}/{len(items)} "
        f"(content_drops={sum(p['n_dropped_draws'] for p in passes.values())}, "
        f"transport={sum(p['n_transport_lost_draws'] for p in passes.values())}) -> {out}"
    )
    return payload


def c0_baseline(model: str, args) -> dict:
    """Informativeness baseline on a seeded 500-row C0 subsample.

    C0 answer text = the parent track_s.jsonl response pool (shared across
    models by the parent recipe — one instruct-generated response set
    teacher-forced through both models; judged once per content by the
    rubric-keyed cache, labeled per model for the report)."""
    import numpy as np

    rows = {r["conv_id"]: r for r in _track_s_rows(args.data_dir)}
    shared = r1417.shared_conv_ids(args.data_dir)
    rng = np.random.default_rng(C0_BASELINE_SEED)
    take = min(C0_BASELINE_N, len(shared))
    pick = [shared[i] for i in rng.choice(len(shared), size=take, replace=False)]
    items = [(cid, rows[cid]["question"], rows[cid]["response"]) for cid in pick]
    if args.limit:
        items = items[: args.limit]
    res = _run_rubric(items, "informativeness", f"{model}_c0_baseline", args)
    payload = {
        "metadata": common931.metadata(SCRIPT, C0_BASELINE_SEED, len(items)),
        "model": model,
        "cell": "c0_baseline",
        "provenance": "parent track_s.jsonl responses (instruct-generated; shared across models)",
        "n_judged": len(items),
        **{k: v for k, v in res.items() if k != "per_item_scores"},
        "per_item_scores": res["per_item_scores"],
    }
    out = judge_dir(args.out_dir) / f"kept_{model}_c0_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[i1417-judge] c0 baseline ({model}): {len(items)} rows -> {out}")
    return payload


def _track_s_rows(data_dir: Path) -> list[dict]:
    path = r1417.tracks_path(data_dir)
    assert path.exists(), f"{path} missing — run issue1417_render.py --fetch-questions"
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append(
                    {
                        "conv_id": f"s{r['prompt_idx']}",
                        "question": r["prompt"],
                        "response": r["response"],
                    }
                )
    return rows


def yield_report(args, results: list[dict]) -> None:
    rep = {
        "metadata": common931.metadata(SCRIPT, r1417.GEN_SEED, len(results)),
        **r1417.fingerprint(),
        "keep_threshold": KEEP_THRESHOLD,
        "n_draws": args.n_draws,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "cells": {
            f"{r['model']}_{r['cell']}": {
                "n_judged": r["n_judged"],
                "n_kept": r.get("n_kept"),
                "yield_frac": r.get("yield_frac"),
                "primary_grade": (
                    bool(r.get("yield_frac", 0) >= 0.5) if r.get("yield_frac") is not None else None
                ),
                "content_drops": sum(p["n_dropped_draws"] for p in r.get("passes", {}).values())
                if "passes" in r
                else r.get("n_dropped_draws"),
                "transport_losses": sum(
                    p["n_transport_lost_draws"] for p in r.get("passes", {}).values()
                )
                if "passes" in r
                else r.get("n_transport_lost_draws"),
            }
            for r in results
        },
    }
    out = judge_dir(args.out_dir) / "yield_report.json"
    out.write_text(json.dumps(rep, indent=2, default=float))
    print(f"[i1417-judge] wrote {out}")


def pilot_report(results: list[dict], args) -> None:
    """N-draw keep-decision stability (plan §11: N=3 ungrounded — pilot).

    Leave-one-draw-out flip rate: fraction of items whose keep decision at
    the threshold flips when any single draw is dropped from the mean —
    a test-retest proxy computable from the persisted per-draw scores.
    Flip rate > 5% on any keep rubric => bump N 3->5 (allowed deviation)."""
    flips: dict[str, dict] = {}
    for r in results:
        if "per_item_scores" not in r or "passes" not in r:
            continue
        for slug, per_item in r["per_item_scores"].items():
            n_items = 0
            n_flip = 0
            for _cid, draws in per_item.items():
                if len(draws) < 2:
                    continue
                n_items += 1
                mean_all = sum(draws) / len(draws)
                keep_all = mean_all >= KEEP_THRESHOLD
                for j in range(len(draws)):
                    rest = [d for i, d in enumerate(draws) if i != j]
                    if (sum(rest) / len(rest) >= KEEP_THRESHOLD) != keep_all:
                        n_flip += 1
                        break
            flips[f"{r['model']}_{r['cell']}_{slug}"] = {
                "n_items": n_items,
                "n_flip": n_flip,
                "flip_frac": (n_flip / n_items) if n_items else float("nan"),
            }
    out = judge_dir(args.out_dir) / "n_draw_pilot.json"
    out.write_text(json.dumps({"flip_rates": flips, "bump_rule": "flip>5% => N=5"}, indent=2))
    print(f"[i1417-judge] wrote {out}")


def live_smoke(args) -> int:
    """Gate G3 (HALT, pre-launch): ~10 REAL rows through the run's own request
    builder with the Batch path FORCED; asserts scored results came back."""
    rows = _track_s_rows(args.data_dir)[:5]
    items = [(r["conv_id"], r["question"], r["response"]) for r in rows]
    res = _run_rubric(items, "informativeness", "g3_live_smoke", args, n_draws=2)
    n_scored = sum(1 for v in res["scores"].values() if v is not None)
    ok = n_scored >= max(1, len(items) - 1) and res["n_transport_lost_draws"] == 0
    report = {
        "gate": "G3",
        "n_items": len(items),
        "n_scored": n_scored,
        "n_dropped_draws": res["n_dropped_draws"],
        "n_transport_lost_draws": res["n_transport_lost_draws"],
        "pass": bool(ok),
    }
    out = judge_dir(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g3_live_smoke.json").write_text(json.dumps(report, indent=2))
    print(f"[i1417-judge] G3 live smoke: {report}")
    if not ok:
        print("[i1417-judge] G3 FAILED — fix the request shape before launch", file=sys.stderr)
        return 21
    return 0


def c5_acknowledgement(args) -> int:
    """RIDER (#1417 refit round, 0 GPU): AI-relay acknowledgement judge pass
    over the KEPT c5_ai_addressee rollouts, both models.

    Graded 0-100, N=--n-draws (default 3) draws mean-aggregated, judge
    ``DEFAULT_JUDGE_MODEL``, max_tokens 300, drop-never-coerce, rubric-keyed
    cache, Batch API path FORCED — the exact judge machinery the v1 keep
    rubrics used. Reads the v1 judge kept-sets (``judge_dir(args.out_dir)``);
    writes ``--ack-out`` (default under refit/ — versioned, never a v1 path)
    plus per-draw raws under ``<ack-out parent>/judge_raw/``. The judge cache
    is partitioned per (rubric, tag) and the tag embeds ``--limit`` so a smoke
    slice never pre-populates the production cache (a reused cache_dir
    collapses an item's draws — graded_judge module note).
    """
    cell = "c5_ai_addressee"
    ack_out = Path(args.ack_out)
    raw_dir = ack_out.parent / "judge_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    per_model: dict[str, dict] = {}
    for model in models:
        if args.stage_from_hub:
            _stage_gen_from_hub(args.data_dir, model, cell)
        gen_file = g1417.gen_path(args.data_dir, model, cell)
        rows = g1417._read_jsonl(gen_file)
        assert rows and r1417.fingerprint_matches(rows[0]), f"fingerprint mismatch: {gen_file}"
        kept_p = judge_dir(args.out_dir) / f"kept_{model}_{cell}.json"
        assert kept_p.exists(), f"v1 kept-set missing: {kept_p} — run the v1 judge phase first"
        kd = json.loads(kept_p.read_text())
        assert r1417.fingerprint_matches(kd), f"{kept_p}: fingerprint mismatch"
        kept = {str(c) for c in kd["kept_conv_ids"]}
        items = [
            (r["conv_id"], r["question"], r["completion"]) for r in rows if r["conv_id"] in kept
        ]
        if args.limit:
            items = items[: args.limit]
        tag = f"{model}_{cell}" + (f"__limit{args.limit}" if args.limit else "")
        cache = Path(args.data_dir) / "judge_cache" / "ai_relay_acknowledgement" / tag
        res = judge_graded(
            items,
            RUBRICS["ai_relay_acknowledgement"],
            n_draws=args.n_draws,
            cache_dir=cache,
            save_raw=raw_dir / f"{tag}__ai_relay_acknowledgement.json",
            judge_model=DEFAULT_JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            threshold_base=0,  # FORCE the Batch API path (same shape as the v1 rubrics)
        )
        scored = {k: v for k, v in res.scores.items() if v is not None}
        n_ack = sum(1 for v in scored.values() if v >= KEEP_THRESHOLD)
        per_model[model] = {
            "n_kept_rows": len(items),
            "n_scored": len(scored),
            "ack_rate_ge50": (n_ack / len(scored)) if scored else float("nan"),
            "mean_score": (sum(scored.values()) / len(scored)) if scored else float("nan"),
            "n_total_draws": res.n_total_draws,
            "n_dropped_draws": res.n_dropped_draws,  # CONTENT drops (rule 9)
            "n_transport_lost_draws": res.n_transport_lost_draws,  # rule 24 split
            "per_item_scores": res.per_item_scores,
        }
        print(
            f"[i1417-judge] c5 acknowledgement ({model}): rate>=50 "
            f"{per_model[model]['ack_rate_ge50']:.4f} over {len(scored)} scored "
            f"(content_drops={res.n_dropped_draws}, transport={res.n_transport_lost_draws})"
        )
    payload = {
        "metadata": common931.metadata(
            SCRIPT, r1417.GEN_SEED, sum(v["n_kept_rows"] for v in per_model.values())
        ),
        **r1417.fingerprint(),
        "cell": cell,
        "rubric": "ai_relay_acknowledgement",
        "judge_model": DEFAULT_JUDGE_MODEL,
        "n_draws": args.n_draws,
        "max_tokens": JUDGE_MAX_TOKENS,
        "threshold": KEEP_THRESHOLD,
        "limit": args.limit or None,
        "models": per_model,
    }
    ack_out.parent.mkdir(parents=True, exist_ok=True)
    ack_out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[i1417-judge] wrote {ack_out}")
    if not args.skip_upload:
        from explore_persona_space.orchestrate import hub

        for p in [ack_out, *sorted(raw_dir.glob("*__ai_relay_acknowledgement.json"))]:
            url = hub._upload(
                p,
                repo_id=r1417.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{r1417.HF_PREFIX}/raw_completions/judge/refit/{p.name}",
                upload_as_file=True,
            )
            assert url, f"rider upload returned no URL for {p}"
        print("[i1417-judge] rider outputs uploaded")
    return 0


def _upload_judge_outputs(args) -> None:
    from explore_persona_space.orchestrate import hub

    sub = f"{args.hf_subdir.strip('/')}/" if getattr(args, "hf_subdir", "") else ""
    for p in sorted(judge_dir(args.out_dir).rglob("*.json")):
        rel = p.relative_to(judge_dir(args.out_dir))
        url = hub._upload(
            p,
            repo_id=r1417.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{r1417.HF_PREFIX}/raw_completions/judge/{sub}{rel}",
            upload_as_file=True,
        )
        assert url, f"judge upload returned no URL for {p}"
    print(f"[i1417-judge] judge outputs uploaded (subdir: {sub or '<v1 root>'})")


def main() -> int:
    args = parse_args()
    if args.live_smoke:
        return live_smoke(args)
    if args.c5_acknowledgement:
        return c5_acknowledgement(args)
    assert args.all, "pass --all (or --live-smoke / --c5-acknowledgement)"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    results: list[dict] = []
    for model in models:
        for cell in cells:
            results.append(judge_cell(model, cell, args))
        if not args.skip_c0_baseline:
            results.append(c0_baseline(model, args))
    yield_report(args, results)
    if args.pilot_report:
        pilot_report(results, args)
    if not args.skip_upload:
        _upload_judge_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
