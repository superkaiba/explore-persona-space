"""#1776 swap-round off-pod judge (plan v7 §4.3 / §6 SECONDARY dual-DV).

Two NEW graded 0-100 rubric FAMILIES over the persisted swap rollout TEXT —
``b_content`` ("does the response express the answer content of the reference
answer?") and the mirrored ``a_retention`` (A's reference answer) — VM-side,
zero GPU, via the p3p4 judge stack conventions: ``judge_completions_batch`` ->
``judge_dispatch``/``api_dispatch`` Batch path, N=5 draws per completion (draw
expansion, temperature-1 API default), DROP-NEVER-COERCE per draw,
``max_tokens=1024`` EXPLICIT (llm-judging rule 23 floor; raised from 300, #2063), content-drop vs
transport-loss accounting (rule 24; residual transport above the threshold
exits rc=4 with the report WRITTEN — route on the artifact, not the rc).

Rubric-fingerprint coherence (llm-judging rule 22 / plan assumption 8): the
per-item B/A EXCERPT rides the judge QUESTION field, so each family keeps ONE
system prompt + ONE ``format_user_msg`` template = ONE rubric fingerprint.

Also computes the §6 judge-vs-metric validation wedge: Spearman rho(MRR_B,
judge b_content score) across steered cells when ``--swap-success`` is given;
under the pre-registered Neither-executes outcome the correlation is
uncomputable (no dynamic range) and convergent nulls substitute (reported).

Content hygiene: rollout/judge text is real-user-derived — never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import issue1776_common as C76
import issue779_common as C
from issue1776_judge import build_alias_maps, rehydrate_cids, split_drop_counts

FU = "followup_swap"
LABEL = "operator_swap_success"
OP_ARMS = ("swap_mprime", "swap_jlast")
STEER_ARMS = (*OP_ARMS, "swap_random")
BASELINE_ARM = "swap_a0"
ALL_ARMS = (BASELINE_ARM, *STEER_ARMS)
RUBRICS = ("b_content", "a_retention")
JUDGE_STEM = "judge_swap"  # rebound to "judge_patch" under --round patch
JUDGE_MAX_TOKENS = 1024  # llm-judging rule 23 floor (single-rationale; raised from 300, #2063)

_RUBRIC_WHAT = {
    "b_content": "a REFERENCE ANSWER (a different answer whose content the response may express)",
    "a_retention": "a REFERENCE ANSWER (the original answer whose content the response may retain)",
}


def judge_system_prompt(rubric: str) -> str:
    """ONE fixed system prompt per rubric family (fingerprint coherence)."""
    assert rubric in RUBRICS, rubric
    return (
        f"You are evaluating a language model's response. You will be shown {_RUBRIC_WHAT[rubric]}, "
        "the USER PROMPT the model was answering, and the MODEL RESPONSE.\n\n"
        "Judge how much the MODEL RESPONSE expresses the ANSWER CONTENT of the REFERENCE ANSWER: "
        "the same facts, claims, entities, topics, and substance. Wording need not match; judge "
        "content overlap, not style. A response on a completely different topic scores 0; a "
        "response conveying essentially all of the reference answer's substance scores 100.\n\n"
        "First think briefly, then give a single score from 0 (none of the reference answer's "
        "content) to 100 (fully expresses it). Reply with a JSON object on one line: "
        '{"reasoning": "<one brief sentence>", "score": <integer 0-100>}. '
        "If the MODEL RESPONSE is a clear refusal to answer, reply with the single word REFUSAL "
        "and nothing else."
    )


def format_user_msg(question: str, completion: str) -> str:
    """ONE fixed user-message template per family; the per-item excerpt rides
    INSIDE ``question`` (composed by :func:`compose_question`)."""
    return f"{question}\n\n[MODEL RESPONSE START]\n{completion}\n[MODEL RESPONSE END]"


def compose_question(excerpt: str, user_prompt: str) -> str:
    """Per-item question field: reference excerpt + original user prompt."""
    return (
        f"[REFERENCE ANSWER START]\n{excerpt}\n[REFERENCE ANSWER END]\n\n"
        f"[USER PROMPT START]\n{user_prompt}\n[USER PROMPT END]"
    )


# ── ingestion ─────────────────────────────────────────────────────────────────


def load_arms(raw_dir: Path) -> dict[str, dict[str, dict]]:
    """{arm: {pair_id: row}} from the canonical merged swap rollout JSONs."""
    arms: dict[str, dict[str, dict]] = {}
    for arm in ALL_ARMS:
        files = sorted(raw_dir.glob(f"{arm}*.json"))
        assert files, f"no rollout file for arm {arm} under {raw_dir}"
        rows: dict[str, dict] = {}
        for f in files:
            d = json.loads(f.read_text())
            assert d["arm"] == arm, (f.name, d.get("arm"))
            for r in d["rows"]:
                assert r["pair_id"] not in rows, (arm, r["pair_id"])
                rows[r["pair_id"]] = r
        # every arm covers the identical registered pair set (§3 row-coverage)
        arms[arm] = rows
    ref = set(arms[BASELINE_ARM])
    for arm in ALL_ARMS:
        assert set(arms[arm]) == ref, f"arm {arm} pair coverage != baseline"
    return arms


def build_rollouts(
    arms: dict[str, dict[str, dict]], targets: dict, rubric: str
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict]]:
    """batch_judge {persona: {question: [completions]}} for ONE rubric family.

    persona = "<arm>::<pair_id>"; question = composed excerpt + user prompt
    (ONE question per persona). Empty completions skipped (recorded)."""
    key = "b_excerpt" if rubric == "b_content" else "a_excerpt"
    rollouts: dict[str, dict[str, list[str]]] = {}
    meta: dict[str, dict] = {}
    for arm in ALL_ARMS:
        for pid, row in sorted(arms[arm].items()):
            tg = targets["per_pair"][pid]
            persona = f"{arm}::{pid}"
            assert persona not in meta, persona
            kept = [(i, s) for i, s in enumerate(row["samples"]) if s.strip()]
            meta[persona] = {
                "arm": arm,
                "pair_id": pid,
                "leg": tg["leg"],
                "n_empty": len(row["samples"]) - len(kept),
                "sample_idx": [i for i, _ in kept],
            }
            if kept:
                q = compose_question(tg[key], row["user"])
                rollouts[persona] = {q: [s for _, s in kept]}
    return rollouts, meta


# ── judging (draw expansion, global-idx custom-id scheme) ────────────────────


def judge_family(
    rubric: str,
    rollouts: dict[str, dict[str, list[str]]],
    save_raw: Path,
    *,
    n_draws: int,
    dry_run: bool = False,
    force_batch: bool = False,
) -> dict[str, tuple[float | None, int, int]]:
    """Graded 0-100 judge, N draws per rollout, mean over VALID draws — the
    ``judge_rollouts_n5`` recipe with THIS family's rubric (cache DISABLED:
    identical (question, completion) draw copies would collapse to one cache
    entry; batch-path checkpointing still derives from save_raw.parent)."""
    from explore_persona_space.eval.batch_judge import judge_completions_batch

    assert n_draws >= 1, n_draws
    expanded: dict[str, dict[str, list[str]]] = {}
    for persona, qmap in rollouts.items():
        expanded[persona] = {}
        for question, comps in qmap.items():
            drawn: list[str] = []
            for comp in comps:
                drawn.extend([comp] * n_draws)
            expanded[persona][question] = drawn

    judge_completions_batch(
        expanded,
        judge_system_prompt=judge_system_prompt(rubric),
        format_user_msg=format_user_msg,
        judge_model=C.JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        cache_dir=None,
        save_raw=save_raw,
        dry_run=dry_run,
        threshold_base=1 if force_batch else 2_000,
    )
    if dry_run:
        return {}
    raw = json.loads(save_raw.read_text())
    draw_scores = C._parse_raw_all_scores(raw["all_scores"])
    # global question idx across ALL (persona, question) — batch_judge scheme
    out: dict[str, tuple[float | None, int, int]] = {}
    idx = 0
    for persona, qmap in expanded.items():
        for _question, drawn in qmap.items():
            n_rollouts_q = len(drawn) // n_draws
            for ri in range(n_rollouts_q):
                vals = [
                    s
                    for d in range(n_draws)
                    if (s := draw_scores.get(f"{persona}__{idx:05d}__{ri * n_draws + d:02d}"))
                    is not None
                ]
                out[f"{persona}__{idx:05d}__{ri:02d}"] = (
                    (float(sum(vals) / len(vals)) if vals else None),
                    len(vals),
                    n_draws,
                )
            idx += 1
    return out


def aggregate_family(
    rubric: str,
    meta: dict[str, dict],
    scores: dict[str, tuple[float | None, int, int]],
    all_scores: dict[str, object],
) -> tuple[dict[str, dict], list[dict]]:
    """Per-arm aggregates + per-cell rows (p3p4 accounting conventions)."""
    drops = split_drop_counts(all_scores)
    by_persona: dict[str, list[tuple[int, float | None]]] = defaultdict(list)
    for cid, (mean, _nv, _n) in scores.items():
        persona, _idx, ri = cid.rsplit("__", 2)
        by_persona[persona].append((int(ri), mean))

    def _new_arm() -> dict:
        return {
            "cell_means": [],
            "gt50_num": 0,
            "gt50_den": 0,
            "content_drops": 0,
            "transport_losses": 0,
            "valid_draws": 0,
            "n_rollouts_dropped": 0,
            "n_empty_samples": 0,
        }

    arm_acc: dict[str, dict] = defaultdict(_new_arm)
    per_cell: list[dict] = []
    for persona, m in meta.items():
        a = arm_acc[m["arm"]]
        a["n_empty_samples"] += m["n_empty"]
        sample_scores = [mean for _ri, mean in sorted(by_persona.get(persona, []))]
        valid = [s for s in sample_scores if s is not None]
        a["n_rollouts_dropped"] += sum(1 for s in sample_scores if s is None)
        d = drops.get(persona, {"valid": 0, "content_drops": 0, "transport_losses": 0})
        a["valid_draws"] += d["valid"]
        a["content_drops"] += d["content_drops"]
        a["transport_losses"] += d["transport_losses"]
        cell_mean = sum(valid) / len(valid) if valid else None
        if valid:
            a["cell_means"].append(cell_mean)
            a["gt50_num"] += sum(1 for s in valid if s > 50)
            a["gt50_den"] += len(valid)
        per_cell.append(
            {
                "rubric": rubric,
                "arm": m["arm"],
                "pair_id": m["pair_id"],
                "leg": m["leg"],
                "sample_scores": sample_scores,
                "cell_mean": cell_mean,
                "n_valid_samples": len(valid),
            }
        )
    per_arm = {
        arm: {
            "mean_score": (sum(a["cell_means"]) / len(a["cell_means"]))
            if a["cell_means"]
            else None,
            "rate_gt50": (a["gt50_num"] / a["gt50_den"]) if a["gt50_den"] else None,
            "n_cells_scored": len(a["cell_means"]),
            "n_rollouts_dropped": a["n_rollouts_dropped"],
            "n_empty_samples": a["n_empty_samples"],
            "valid_draws": a["valid_draws"],
            "content_drops": a["content_drops"],
            "transport_losses": a["transport_losses"],
        }
        for arm, a in arm_acc.items()
    }
    return per_arm, per_cell


def _boot_ci(vals, n_boot: int = 1000, seed: int = 1776) -> list[float]:
    import numpy as np

    v = np.asarray(vals, dtype=float)
    if v.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def contrasts_and_validation(per_cell: list[dict], swap_success: dict | None) -> dict:
    """steered - baseline pair-clustered contrasts (§7 supporting read) + the
    §6 judge-vs-metric validation wedge."""
    import numpy as np
    from scipy import stats as sps

    out: dict = {"steered_minus_baseline": {}, "validation": {}}
    for rubric in RUBRICS:
        cells = [c for c in per_cell if c["rubric"] == rubric and c["cell_mean"] is not None]
        base = {c["pair_id"]: c["cell_mean"] for c in cells if c["arm"] == BASELINE_ARM}
        out["steered_minus_baseline"][rubric] = {}
        for arm in STEER_ARMS:
            deltas = [
                c["cell_mean"] - base[c["pair_id"]]
                for c in cells
                if c["arm"] == arm and c["pair_id"] in base
            ]
            out["steered_minus_baseline"][rubric][arm] = {
                "mean": float(np.mean(deltas)) if deltas else None,
                "ci95": _boot_ci(deltas),
                "n_pairs": len(deltas),
            }
    if swap_success is not None:
        mrr = {
            (c["arm"], c["pair_id"]): c["mrr_b"]
            for c in swap_success["per_cell"]
            if c["arm"] in STEER_ARMS
        }
        xs, ys = [], []
        for c in per_cell:
            if c["rubric"] == "b_content" and c["arm"] in STEER_ARMS and c["cell_mean"] is not None:
                m = mrr.get((c["arm"], c["pair_id"]))
                if m is not None:
                    xs.append(m)
                    ys.append(c["cell_mean"])
        if len(xs) >= 3 and float(np.std(xs)) > 1e-9 and float(np.std(ys)) > 1e-9:
            rho = sps.spearmanr(xs, ys)
            out["validation"] = {
                "spearman_mrr_vs_judge": float(rho.statistic),
                "pvalue": float(rho.pvalue),
                "n_cells": len(xs),
                "wedge": "computed",
            }
        else:
            out["validation"] = {
                "spearman_mrr_vs_judge": None,
                "n_cells": len(xs),
                "wedge": (
                    "uncomputable (no dynamic range) — the registered §6 wedge: convergent "
                    "nulls of BOTH DVs against their own controls substitute for the "
                    "correlation gate; scatter data in per_cell"
                ),
            }
    return out


# ── driver ────────────────────────────────────────────────────────────────────


def run(args) -> int:
    arms = load_arms(args.raw_dir)
    targets = json.loads(args.targets.read_text())
    swap_success = (
        json.loads(args.swap_success.read_text())
        if args.swap_success and args.swap_success.exists()
        else None
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_arm: dict[str, dict] = {}
    per_cell: list[dict] = []
    worst_transport = 0.0
    for rubric in args.rubrics:
        rollouts, meta = build_rollouts(arms, targets, rubric)
        n_comps = sum(len(cs) for qmap in rollouts.values() for cs in qmap.values())
        print(
            f"[swap-judge] rubric={rubric}: {n_comps} completions x N={args.n_draws} draws "
            f"({len(meta)} cells)",
            flush=True,
        )
        save_raw = args.out_dir / f"{JUDGE_STEM}_raw_{rubric}.json"
        alias_of, persona_of = build_alias_maps(meta.keys())
        C76.atomic_write_json(
            args.out_dir / f"{JUDGE_STEM}_id_map_{rubric}.json", {"alias_to_persona": persona_of}
        )
        rollouts_aliased = {alias_of[p]: qmap for p, qmap in rollouts.items()}
        scores = judge_family(
            rubric, rollouts_aliased, save_raw, n_draws=args.n_draws, dry_run=args.dry_run
        )
        if args.dry_run:
            continue
        raw = json.loads(save_raw.read_text())
        scores = rehydrate_cids(scores, persona_of)
        all_scores = rehydrate_cids(raw["all_scores"], persona_of)
        arms_agg, cells = aggregate_family(rubric, meta, scores, all_scores)
        per_arm[rubric] = arms_agg
        per_cell.extend(cells)
        for a in arms_agg.values():
            total = a["valid_draws"] + a["content_drops"] + a["transport_losses"]
            if total:
                worst_transport = max(worst_transport, a["transport_losses"] / total)
    if args.dry_run:
        print("[swap-judge] dry-run complete (routing printed above, zero API calls)")
        return 0

    report = {
        "label": LABEL,
        "judge_model": C.JUDGE_MODEL,
        "n_draws": args.n_draws,
        "max_tokens": JUDGE_MAX_TOKENS,
        "rubrics": list(args.rubrics),
        "per_arm": per_arm,
        **contrasts_and_validation(per_cell, swap_success),
        "per_cell": per_cell,
        "worst_arm_transport_loss_frac": worst_transport,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out_dir / f"{JUDGE_STEM}.json", report)
    print(
        f"[swap-judge] [phase=judge_done] rubrics={list(per_arm)} cells={len(per_cell)} "
        f"worst_transport_frac={worst_transport:.4f} -> {args.out_dir / (JUDGE_STEM + '.json')}",
        flush=True,
    )
    if worst_transport > args.transport_fail_threshold:
        print(
            f"[swap-judge] WARNING: residual transport-loss fraction {worst_transport:.4f} > "
            f"{args.transport_fail_threshold} — re-drive the lost draws (rule 24ii) against a "
            "fresh save_raw; report already written",
            flush=True,
        )
        return 4
    return 0


# ── smokes ────────────────────────────────────────────────────────────────────


def _fixture_inputs(out: Path) -> tuple[Path, Path]:
    """Tiny merged-rollout + targets fixtures at the production schemas."""
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pairs = ["sw_lm000", "sw_wc000"]
    for arm in ALL_ARMS:
        rows = []
        for k, pid in enumerate(pairs):
            samples = ["Paris is the capital and hosts the Louvre.", "Two is a prime number."]
            if arm == "swap_mprime" and k == 0:
                samples = ["SMOKE_TRANSPORT dead lane", "SMOKE_REFUSE bait"]
            if arm == "swap_random" and k == 1:
                samples = ["", "SMOKE_MALFORMED text"]
            rows.append(
                {
                    "pair_id": pid,
                    "context_id": f"c{k}",
                    "leg": "lmsys" if k == 0 else "wildchat",
                    "user": f"question {k}?",
                    "system": None,
                    "samples": samples,
                }
            )
        C76.atomic_write_json(
            raw_dir / f"{arm}.json", {"arm": arm, "mode": "prefill", "k": 2, "rows": rows}
        )
    targets = {
        "label": LABEL,
        "per_pair": {
            pid: {
                "leg": "lmsys" if k == 0 else "wildchat",
                "b_excerpt": f"reference answer {k} about topic{k}",
                "a_excerpt": f"original answer {k} content",
                "t_b": ["topic"],
                "t_a": ["original"],
                "included": True,
            }
            for k, pid in enumerate(pairs)
        },
    }
    tpath = out / "targets.json"
    C76.atomic_write_json(tpath, targets)
    return raw_dir, tpath


def smoke(args) -> int:
    """Structural CPU smoke: REAL request-builder path (both NEW rubric
    families, excerpt-bearing question field, draw fan-out, custom-id scheme,
    drop-never-coerce + transport split, rc=4 gate), mocked ONLY at the
    anthropic client transport seam (the issue1776_judge fake)."""
    import os
    from types import SimpleNamespace

    import anthropic

    from issue1776_judge import _FakeMessages

    out = args.out_dir
    raw_dir, tpath = _fixture_inputs(out)
    fake_msgs = _FakeMessages()

    class _FakeAsyncAnthropic:
        def __init__(self, *a, **kw):
            self.messages = fake_msgs

    class _FakeAnthropic:
        def __init__(self, *a, **kw):
            probe = SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    headers={"anthropic-ratelimit-output-tokens-limit": "400000"}
                )
            )
            self.messages = SimpleNamespace(with_raw_response=probe)

    real_async, real_sync = anthropic.AsyncAnthropic, anthropic.Anthropic
    os.environ["EPS_JUDGE_DISABLE_MULTIORG"] = "1"
    os.environ.setdefault("ANTHROPIC_API_KEY", "smoke-placeholder")
    anthropic.AsyncAnthropic, anthropic.Anthropic = _FakeAsyncAnthropic, _FakeAnthropic
    try:
        args.raw_dir = raw_dir
        args.targets = tpath
        args.swap_success = None
        args.rubrics = list(RUBRICS)
        args.n_draws = 5
        rc = run(args)
    finally:
        anthropic.AsyncAnthropic, anthropic.Anthropic = real_async, real_sync
        os.environ.pop("EPS_JUDGE_DISABLE_MULTIORG", None)

    # transport-dead completion (5/5 transport draws on one arm) trips rc=4
    assert rc == 4, rc
    report = json.loads((out / "judge_swap.json").read_text())
    bc = report["per_arm"]["b_content"]
    assert bc["swap_mprime"]["transport_losses"] == 5, bc["swap_mprime"]
    assert bc["swap_mprime"]["content_drops"] == 5, bc["swap_mprime"]  # REFUSAL draw
    assert bc["swap_random"]["content_drops"] == 5, bc["swap_random"]  # malformed
    assert bc["swap_random"]["n_empty_samples"] == 1, bc["swap_random"]
    assert bc["swap_a0"]["valid_draws"] == 20 and bc["swap_a0"]["content_drops"] == 0, bc["swap_a0"]
    # request shape: system prompt + composed excerpt-bearing question reached
    # the (fake) wire; max_tokens explicit at 1024; model = the project judge
    p0 = fake_msgs.calls[0]
    assert p0["model"] == C.JUDGE_MODEL and p0["max_tokens"] == JUDGE_MAX_TOKENS, p0
    sys_text = p0["system"] if isinstance(p0["system"], str) else p0["system"][0]["text"]
    assert "REFERENCE ANSWER" in sys_text and "REFUSAL" in sys_text
    user0 = (
        p0["messages"][0]["content"]
        if isinstance(p0["messages"][0]["content"], str)
        else p0["messages"][0]["content"][0]["text"]
    )
    assert "[REFERENCE ANSWER START]" in user0 and "[MODEL RESPONSE START]" in user0
    # contrast join produced steered-baseline rows for both rubrics
    assert set(report["steered_minus_baseline"]) == set(RUBRICS)
    print("[swap-judge] [phase=smoke_done] PASS (accounting splits exact; rubric shape on wire)")
    return 0


def live_smoke(args) -> int:
    """~6-request LIVE FORCED-BATCH probe through the run's OWN request
    builder (excerpt-bearing question field, both families) — the gotchas-rule
    Batch-API shape gate a mock smoke structurally cannot provide (assumption
    8 verify step). Asserts every request returned a judge verdict (score or
    REFUSAL), i.e. no invalid_request_error quarantines."""
    out = args.out_dir
    raw_dir, tpath = _fixture_inputs(out)
    arms = load_arms(raw_dir)
    targets = json.loads(tpath.read_text())
    n_checked = 0
    for rubric in RUBRICS:
        rollouts, meta = build_rollouts(arms, targets, rubric)
        # 3 clean completions per family (skip the SMOKE_* fault-injection rows)
        keep = {
            p: q
            for p, q in rollouts.items()
            if all("SMOKE_" not in c for cs in q.values() for c in cs)
        }
        keep = dict(sorted(keep.items())[:3])
        assert keep, "no clean fixture completions for the live probe"
        alias_of, persona_of = build_alias_maps(keep.keys())
        save_raw = out / f"live_smoke_raw_{rubric}.json"
        judge_family(
            rubric,
            {alias_of[p]: q for p, q in keep.items()},
            save_raw,
            n_draws=1,
            force_batch=True,
        )
        raw = json.loads(save_raw.read_text())
        assert raw["all_scores"], "live probe returned no scores"
        for cid, sd in raw["all_scores"].items():
            assert not (isinstance(sd, dict) and sd.get("error")), (
                f"live forced-batch probe FAILED (request-shape error) at {cid}: "
                f"{ {k: sd.get(k) for k in ('error', 'reason', 'transport')} }"
            )
            n_checked += 1
        routing = raw.get("routing", {})
        assert routing.get("path") == "batch", routing
    print(f"[swap-judge] [phase=live_smoke_done] PASS ({n_checked} batch results, 0 errors)")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="#1776 swap-round off-pod graded judge (dual-DV a)")
    ap.add_argument("--mode", choices=["run", "smoke", "live-smoke"], default="run")
    ap.add_argument("--raw-dir", type=Path, help="dir of merged swap rollout JSONs")
    ap.add_argument("--targets", type=Path, help="targets.json (excerpts)")
    ap.add_argument("--swap-success", type=Path, default=None, help="swap_success.json (wedge)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--rubrics",
        default=",".join(RUBRICS),
        help="descope ladder drops a_retention first (plan §9)",
    )
    ap.add_argument("--n-draws", type=int, default=C.JUDGE_N_DRAWS)
    ap.add_argument("--transport-fail-threshold", type=float, default=0.02)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--round",
        choices=["swap", "patch"],
        default="swap",
        help="patch = slot_patch_sufficiency (plan v8): rebinds the arm set + output stems; "
        "swap (default) is byte-identical",
    )
    args = ap.parse_args(argv)
    if args.round == "patch":
        # CONDITIONAL patch-round judge (plan v8 §7 trigger): same driver, the
        # arm set + label + output stems rebind; rubrics/accounting unchanged.
        global FU, LABEL, OP_ARMS, STEER_ARMS, BASELINE_ARM, ALL_ARMS, JUDGE_STEM
        FU = "followup_slotpatch"
        LABEL = "slot_patch_sufficiency"
        OP_ARMS = ()
        STEER_ARMS = ("swap_patch",)
        BASELINE_ARM = "patch_a0"
        ALL_ARMS = (BASELINE_ARM, *STEER_ARMS)
        JUDGE_STEM = "judge_patch"
    args.rubrics = [r for r in str(args.rubrics).split(",") if r]
    assert set(args.rubrics) <= set(RUBRICS), args.rubrics
    if args.mode == "smoke":
        return smoke(args)
    if args.mode == "live-smoke":
        return live_smoke(args)
    assert args.raw_dir is not None and args.targets is not None, "--raw-dir/--targets required"
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
