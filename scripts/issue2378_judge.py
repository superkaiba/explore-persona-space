"""issue #2378 judge driver — admission + congruence waves (plan v6 §4.5).

Two graded 0-100 judge waves on `claude-sonnet-4-5-20250929` via the
multi-org dispatcher (`llm.api_dispatch.dispatch_calls`; rubric-keyed cache +
checkpoint resume are the dispatcher's own):

- ``--wave admission``   scores every mined SegA row on PRE-ANSWER scene text
  only (scene + mined utterance; never a reply — no selection on the DV).
  Single draw, deterministic (temperature 0.0). Admit = score >= 50.
  Production ~117k calls => Batch API path.
- ``--wave congruence``  scores admitted rows' SegB replies for in-scene
  congruence: per cell up to ``--congruence-rows`` rows x
  ``--congruence-draws`` draws at temperature 1.0, mean-aggregated per row;
  per-cell median REPORTED (expected >= ~80), never gated.

Pilot gate (plan §4.5; llm-judging rules 23/26): ``--pilot N`` runs an
N-call wave on the SAME instrument + transport (``--transport sync|batch``)
and writes a report JSON; gate = ZERO ``stop_reason == "max_tokens"`` AND
pooled per-FAMILY-arm parse-fail < 2% with >= 100 answered draws per family
(ACTIVE families per cm.ACTIVE_FAMILIES — question-only at v7, dialogue
descoped; per-cell rates descriptive only). A gate
FAIL exits rc=7 (designed halt, artifact-routed — never a bare rc=1).
Production REFUSES to dispatch without a PASS pilot artifact for BOTH
transports of its wave (sync pilot + forced-batch pre-wave), except a
bounded sync smoke slice (``--max-items`` <= 500).

Drop classes (drop-never-coerce; llm-judging rule 4): ``parse_fail``,
``out_of_range``, ``rubric_refusal`` (the rubric's literal "REFUSAL"),
``truncation`` (stop_reason == max_tokens), ``api_refusal`` (stop_reason ==
refusal), ``transport_loss`` (transport/rate-limit exhaustion after ONE
sync re-drive). Hard non-transient request errors fail loud.

Usage:
  uv run python scripts/issue2378_judge.py --wave admission --pilot 200 --transport sync
  uv run python scripts/issue2378_judge.py --wave admission --pilot 200 --transport batch
  uv run python scripts/issue2378_judge.py --wave admission
  uv run python scripts/issue2378_judge.py --wave congruence --congruence-rows 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import issue2378_banks as bnk
import issue2378_common as cm

RC_PILOT_GATE_FAIL = 7
PILOT_MIN_DRAWS_PER_FAMILY = 100
# Oversample above the per-family floor so ONE transport loss cannot FAIL the
# answered>=floor gate as a granularity artifact (r1 review g1 concern 3).
PILOT_FAMILY_OVERSAMPLE = 10
PILOT_PARSE_FAIL_MAX = 0.02
SMOKE_MAX_ITEMS_WITHOUT_PILOT = 500


def _fill(template: str, **slots: str) -> str:
    """Sequential ``{slot}`` replacement — str.format is unusable because the
    rubrics carry literal JSON braces in their output-format examples."""
    for k, v in slots.items():
        template = template.replace("{" + k + "}", v)
    return template


def _rubric_sha() -> str:
    import hashlib

    blob = "|".join(
        [
            bnk.ADMISSION_SYSTEM,
            bnk.ADMISSION_RUBRIC_QUESTION,
            bnk.ADMISSION_RUBRIC_DIALOGUE,
            bnk.CONGRUENCE_SYSTEM,
            bnk.CONGRUENCE_RUBRIC,
            cm.JUDGE_MODEL,
            str(cm.JUDGE_MAX_TOKENS),
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_mined_rows(mined_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(mined_dir.glob("*.jsonl")):
        for row in cm.iter_jsonl(path):
            rows[row["row_id"]] = row
    if not rows:
        raise RuntimeError(f"no mined rows under {mined_dir} (empty selection — fail loud)")
    return rows


def _load_segb_answers(segb_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(segb_dir.glob("*.jsonl")):
        for row in cm.iter_jsonl(path):
            if row.get("keep") and row.get("answer"):
                rows[row["row_id"]] = row
    if not rows:
        raise RuntimeError(f"no kept SegB rows under {segb_dir} (empty selection — fail loud)")
    return rows


def _stage_inputs(args) -> None:
    if not args.stage_from_hf:
        return
    dest = cm.REPO_ROOT / "data" / "issue_2378" / "hf_stage"
    args.mined_dir = str(cm.stage_hf_prefix(f"{cm.HF_PREFIX}/raw_completions/sega_mined", dest))
    if args.wave == "congruence":
        args.segb_dir = str(cm.stage_hf_prefix(f"{cm.HF_PREFIX}/raw_completions/segb", dest))


# ---------------------------------------------------------------------------
# Request build + parse + reduce
# ---------------------------------------------------------------------------


def _build_request_factory(wave: str):
    """Return build_request(item) for `dispatch_calls` (top-level system=;
    NEVER a system-role message — Anthropic Messages API 400s on it)."""
    temperature = 0.0 if wave == "admission" else 1.0

    def build_request(item) -> dict:
        p = item.payload
        if wave == "admission":
            system = bnk.ADMISSION_SYSTEM
            rubric = (
                bnk.ADMISSION_RUBRIC_QUESTION
                if p["family"] == "question"
                else bnk.ADMISSION_RUBRIC_DIALOGUE
            )
            user = _fill(rubric, name=p["character"], scene=p["scene"], utterance=p["utterance"])
        else:
            system = bnk.CONGRUENCE_SYSTEM
            user = _fill(
                bnk.CONGRUENCE_RUBRIC, name=p["character"], scene=p["scene"], reply=p["reply"]
            )
        return {
            "model": cm.JUDGE_MODEL,
            "max_tokens": cm.JUDGE_MAX_TOKENS,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

    return build_request


def _parse_judge(text: str):
    from explore_persona_space.eval.utils import parse_judge_json

    try:
        return parse_judge_json(text)
    except Exception:
        return None


def _extract_score(parsed) -> tuple[int | str | None, str | None]:
    """Normalize parse_judge_json output to (score | "REFUSAL" | None, reasoning)."""
    if isinstance(parsed, bool):
        return None, None
    if isinstance(parsed, int | float):
        return int(parsed), None
    if isinstance(parsed, str):
        return ("REFUSAL" if parsed.strip().upper() == "REFUSAL" else None), None
    if isinstance(parsed, dict):
        s = parsed.get("score")
        r = parsed.get("reasoning") if isinstance(parsed.get("reasoning"), str) else None
        if isinstance(s, str) and s.strip().upper() == "REFUSAL":
            return "REFUSAL", r
        if isinstance(s, bool):
            return None, r
        if isinstance(s, int | float):
            return int(s), r
        return None, r
    return None, None


def _classify(res) -> dict:
    """Map one DispatchResult to {class, score, reasoning, stop_reason}.

    Classes: valid | parse_fail | out_of_range | rubric_refusal | truncation |
    api_refusal | transport | hard_error.
    """
    from explore_persona_space.llm import api_dispatch as ad

    stop = getattr(res, "stop_reason", None)
    if res.category in (ad.RESULT_RATE_LIMITED, ad.RESULT_TRANSPORT):
        return {"class": "transport", "score": None, "reasoning": None, "stop_reason": stop}
    if res.category == ad.RESULT_EMPTY_RESPONSE:
        if stop == "refusal":
            return {"class": "api_refusal", "score": None, "reasoning": None, "stop_reason": stop}
        return {"class": "transport", "score": None, "reasoning": None, "stop_reason": stop}
    if res.category == ad.RESULT_ERROR or res.error:
        return {
            "class": "hard_error",
            "score": None,
            "reasoning": str(res.reason)[:300],
            "stop_reason": stop,
        }
    score, reasoning = _extract_score(res.result)
    if score == "REFUSAL":
        return {
            "class": "rubric_refusal",
            "score": None,
            "reasoning": reasoning,
            "stop_reason": stop,
        }
    if score is None:
        cls = "truncation" if stop == "max_tokens" else "parse_fail"
        return {"class": cls, "score": None, "reasoning": reasoning, "stop_reason": stop}
    if not 0 <= score <= 100:
        return {
            "class": "out_of_range",
            "score": score,
            "reasoning": reasoning,
            "stop_reason": stop,
        }
    return {"class": "valid", "score": score, "reasoning": reasoning, "stop_reason": stop}


def _dispatch(items, args, force_path: str | None, cache_tag: str) -> dict:
    """One dispatch_calls pass + ONE sync re-drive of transport-class losses.

    ``cache_tag`` partitions the rubric-keyed cache/checkpoint dirs: pilots
    get their own partition (a cache-served pilot is transport-unverifiable —
    llm-judging rule 26/#2152), production waves share theirs across resumes.
    """
    from explore_persona_space.llm.api_dispatch import dispatch_calls

    build_request = _build_request_factory(args.wave)
    common = {
        "model": cm.JUDGE_MODEL,
        "build_request": build_request,
        "parse_response": _parse_judge,
        "cache_dir": Path(args.cache_dir) / cache_tag,
        "checkpoint_dir": Path(args.checkpoint_dir) / cache_tag,
    }
    results = asyncio.run(dispatch_calls(items, force_path=force_path, **common))
    classified = {iid: _classify(res) for iid, res in results.items()}
    redrive_ids = [iid for iid, c in classified.items() if c["class"] == "transport"]
    if redrive_ids:
        print(
            f"[{args.wave}] re-driving {len(redrive_ids)} transport-class losses (sync)", flush=True
        )
        sub = [it for it in items if it.item_id in set(redrive_ids)]
        results2 = asyncio.run(dispatch_calls(sub, force_path="sync", **common))
        for iid, res in results2.items():
            classified[iid] = _classify(res)
    for iid, c in classified.items():
        if c["class"] == "transport":
            c["class"] = "transport_loss"
    hard = [iid for iid, c in classified.items() if c["class"] == "hard_error"]
    if hard:
        sample = {iid: classified[iid]["reasoning"] for iid in hard[:5]}
        raise RuntimeError(
            f"{len(hard)} non-transient request errors in wave {args.wave} — pipeline bug, "
            f"fail loud. Sample: {json.dumps(sample)}"
        )
    return classified


# ---------------------------------------------------------------------------
# Item construction
# ---------------------------------------------------------------------------


def _admission_items(mined: dict[str, dict], max_items: int):
    from explore_persona_space.llm.api_dispatch import DispatchItem

    row_ids = sorted(mined)
    if max_items:
        row_ids = row_ids[:max_items]
    items = []
    for rid in row_ids:
        m = mined[rid]
        items.append(
            DispatchItem(
                item_id=f"adm|{rid}",
                payload={
                    "row_id": rid,
                    "cell": m["cell"],
                    "family": m["family"],
                    "character": m["character"],
                    "scene": m["scene_pre_answer"],
                    "utterance": m["utterance"],
                },
            )
        )
    return items


def _congruence_items(args, mined: dict[str, dict]):
    from explore_persona_space.llm.api_dispatch import DispatchItem

    segb = _load_segb_answers(Path(args.segb_dir))
    kept_dir = Path(args.kept_dir)
    items = []
    per_cell_rows: dict[str, list[str]] = {}
    for cell in cm.STORY_CELLS:
        kept_path = kept_dir / f"{cell}.json"
        if not kept_path.exists():
            raise RuntimeError(f"missing admission keeps {kept_path} — run admission first")
        admitted = [k["row_id"] for k in json.loads(kept_path.read_text())["admitted"]]
        resolved = [rid for rid in admitted if rid in segb]
        if not resolved:
            raise RuntimeError(f"no admitted+SegB-resolved rows for {cell} (fail loud)")
        order = random.Random(cm.derived_seed(cm.SEED, "congruence_select", cell)).sample(
            range(len(resolved)), len(resolved)
        )
        sel = [resolved[i] for i in order[: args.congruence_rows]]
        per_cell_rows[cell] = sel
        for rid in sel:
            for d in range(args.congruence_draws):
                items.append(
                    DispatchItem(
                        item_id=f"cong|{rid}|d{d}",
                        payload={
                            "row_id": rid,
                            "cell": cell,
                            "draw": d,
                            "family": mined[rid]["family"],
                            "character": mined[rid]["character"],
                            "scene": mined[rid]["scene_pre_answer"],
                            "reply": segb[rid]["answer"],
                        },
                    )
                )
    if args.max_items:
        items = items[: args.max_items]
    return items, per_cell_rows


def _pilot_items(args, mined: dict[str, dict], n: int):
    """Family-stratified pilot sample: ceil(n/len(active families)) per
    family, round-robin across cells within family for coverage (plan §4.5
    pooled-family arms; v7: ACTIVE families only — archival dialogue rows in
    a mixed mined dir are skipped, never sampled)."""
    by_family: dict[str, dict[str, list[str]]] = {f: {} for f in cm.ACTIVE_FAMILIES}
    for rid, m in mined.items():
        if m["family"] not in by_family:
            continue
        by_family[m["family"]].setdefault(m["cell"], []).append(rid)
    per_family = -(-n // len(cm.ACTIVE_FAMILIES)) + PILOT_FAMILY_OVERSAMPLE  # ceil-div
    chosen: list[str] = []
    for family, cells in by_family.items():
        if not cells:
            raise RuntimeError(f"pilot: no mined rows in family {family} (fail loud)")
        pools = {}
        for cell, rids in cells.items():
            order = random.Random(cm.derived_seed(cm.SEED, "pilot", family, cell)).sample(
                range(len(rids)), len(rids)
            )
            pools[cell] = [sorted(rids)[i] for i in order]
        take, i = [], 0
        cell_names = sorted(pools)
        while len(take) < per_family and any(pools[c] for c in cell_names):
            c = cell_names[i % len(cell_names)]
            if pools[c]:
                take.append(pools[c].pop(0))
            i += 1
        chosen.extend(take)
    return _admission_items({rid: mined[rid] for rid in chosen}, 0)


# ---------------------------------------------------------------------------
# Gate + reports
# ---------------------------------------------------------------------------


def _tally(classified: dict, items_by_id: dict) -> dict:
    per_family: dict[str, dict[str, int]] = {}
    per_cell: dict[str, dict[str, int]] = {}
    stop_tally: dict[str, int] = {}
    for iid, c in classified.items():
        fam = items_by_id[iid].payload["family"]
        cell = items_by_id[iid].payload["cell"]
        for bucket in (per_family.setdefault(fam, {}), per_cell.setdefault(cell, {})):
            bucket[c["class"]] = bucket.get(c["class"], 0) + 1
        key = str(c["stop_reason"])
        stop_tally[key] = stop_tally.get(key, 0) + 1
    return {"per_family": per_family, "per_cell": per_cell, "stop_reasons": stop_tally}


def _gate_verdict(tally: dict) -> tuple[bool, list[str]]:
    reasons = []
    # Plan §4.5 literal gate: ZERO stop_reason=='max_tokens' — checked on the
    # raw stop-reason tally too, so a VALID-parsed-but-truncated row cannot
    # escape the truncation-CLASS count (r1 review g1 concern 2).
    n_max_tokens = tally["stop_reasons"].get("max_tokens", 0)
    if n_max_tokens:
        reasons.append(f"{n_max_tokens} draws with stop_reason=max_tokens (gate: zero)")
    for fam, bucket in sorted(tally["per_family"].items()):
        answered = sum(v for k, v in bucket.items() if k != "transport_loss")
        truncated = bucket.get("truncation", 0)
        parse_fail = bucket.get("parse_fail", 0)
        rate = parse_fail / answered if answered else 1.0
        if truncated > 0:
            reasons.append(f"{fam}: {truncated} truncated draws (gate: zero)")
        if answered < PILOT_MIN_DRAWS_PER_FAMILY:
            reasons.append(f"{fam}: {answered} answered draws < {PILOT_MIN_DRAWS_PER_FAMILY} floor")
        if rate >= PILOT_PARSE_FAIL_MAX:
            reasons.append(f"{fam}: parse-fail {rate:.4f} >= {PILOT_PARSE_FAIL_MAX}")
    return (not reasons), reasons


def _pilot_report_path(out_root: Path, wave: str, transport: str) -> Path:
    return out_root / "judge" / f"pilot_{wave}_{transport}.json"


def _require_pilot_pass(args, out_root: Path) -> None:
    if (
        args.max_items
        and args.max_items <= SMOKE_MAX_ITEMS_WITHOUT_PILOT
        and (args.transport == "sync")
    ):
        print(
            f"[{args.wave}] bounded sync smoke slice ({args.max_items} items) — pilot "
            "artifacts not required",
            flush=True,
        )
        return
    for transport in ("sync", "batch"):
        path = _pilot_report_path(out_root, args.wave, transport)
        if not path.exists():
            raise RuntimeError(
                f"production {args.wave} wave refused: missing pilot artifact {path} "
                f"(run --pilot with --transport {transport} first)"
            )
        report = json.loads(path.read_text())
        if report.get("verdict") != "PASS":
            raise RuntimeError(
                f"production {args.wave} wave refused: pilot {path} verdict "
                f"{report.get('verdict')!r}"
            )
        if report.get("rubric_sha") != _rubric_sha():
            raise RuntimeError(
                f"production {args.wave} wave refused: pilot {path} rubric_sha "
                f"{report.get('rubric_sha')} != live {_rubric_sha()} (rubric drifted — re-pilot)"
            )


def _persist_raw(args, classified: dict, items_by_id: dict, stage: str) -> None:
    out_dir = Path(args.raw_root) / stage
    writer = cm.ShardWriter(out_dir, f"{args.wave}_{int(time.time())}")
    for iid in sorted(classified):
        c = classified[iid]
        p = items_by_id[iid].payload
        writer.write(
            {
                "item_id": iid,
                "row_id": p["row_id"],
                "cell": p["cell"],
                "draw": p.get("draw"),
                "class": c["class"],
                "score": c["score"],
                "reasoning": c["reasoning"],
                "stop_reason": c["stop_reason"],
            }
        )
    info = writer.close()
    print(
        f"[{args.wave}] raw rows persisted: {info['n_rows']} rows, {len(info['shards'])} shards",
        flush=True,
    )
    if not args.skip_upload:
        cm.upload_stage_dir(out_dir, f"{cm.HF_PREFIX}/raw_completions/{stage}")


# ---------------------------------------------------------------------------
# Waves
# ---------------------------------------------------------------------------


def _run_pilot(args, mined: dict[str, dict]) -> int:
    """Pilot at the EXACT production instrument of the wave under test
    (rule 26: rubric + model + max_tokens + transport; the congruence pilot
    therefore runs post-SegB, when kept/segb inputs exist)."""
    if args.wave == "admission":
        items = _pilot_items(args, mined, args.pilot)
    else:
        all_items, _rows = _congruence_items(args, mined)
        # v7: ACTIVE families only (dialogue descoped; archival rows skipped).
        by_fam: dict[str, list] = {f: [] for f in cm.ACTIVE_FAMILIES}
        for it in all_items:
            if it.payload["family"] in by_fam:
                by_fam[it.payload["family"]].append(it)
        # Same oversample margin as the admission pilot (g1 concern 3).
        target = args.pilot + len(cm.ACTIVE_FAMILIES) * PILOT_FAMILY_OVERSAMPLE
        items, i = [], 0
        fams = tuple(cm.ACTIVE_FAMILIES)
        while len(items) < target and any(by_fam[f] for f in fams):
            fam = fams[i % len(fams)]
            if by_fam[fam]:
                items.append(by_fam[fam].pop(0))
            i += 1
    items_by_id = {it.item_id: it for it in items}
    print(f"[pilot] {args.wave}/{args.transport}: {len(items)} calls", flush=True)
    classified = _dispatch(
        items,
        args,
        force_path=args.transport,
        cache_tag=f"{args.wave}_pilot_{args.transport}",
    )
    # Persist-by-default: pilot judge rows upload like production waves
    # (r1 review g1 concern 8 — previously only the tally survived).
    _persist_raw(args, classified, items_by_id, f"judge_{args.wave}_pilot")
    tally = _tally(classified, items_by_id)
    ok, reasons = _gate_verdict(tally)
    out_root = Path(args.out_root)
    report = {
        "wave": args.wave,
        "transport": args.transport,
        "n_items": len(items),
        "verdict": "PASS" if ok else "FAIL",
        "fail_reasons": reasons,
        "tally": tally,
        "judge_model": cm.JUDGE_MODEL,
        "max_tokens": cm.JUDGE_MAX_TOKENS,
        "rubric_sha": _rubric_sha(),
        "metadata": cm.run_metadata(),
    }
    path = _pilot_report_path(out_root, args.wave, args.transport)
    cm.atomic_write_json(path, report)
    print(f"[pilot] verdict={report['verdict']} report={path}", flush=True)
    if not ok:
        print(
            f"[pilot] GATE FAIL (designed halt, rc={RC_PILOT_GATE_FAIL}): {json.dumps(reasons)}",
            flush=True,
        )
        return RC_PILOT_GATE_FAIL
    return 0


def _run_admission(args, mined: dict[str, dict]) -> int:
    out_root = Path(args.out_root)
    _require_pilot_pass(args, out_root)
    items = _admission_items(mined, args.max_items)
    items_by_id = {it.item_id: it for it in items}
    force = None if args.transport == "auto" else args.transport
    print(f"[admission] dispatching {len(items)} calls (transport={args.transport})", flush=True)
    classified = _dispatch(items, args, force_path=force, cache_tag=args.wave)
    _persist_raw(args, classified, items_by_id, "judge_admission")
    kept_dir = out_root / "kept"
    cells = sorted({it.payload["cell"] for it in items})
    for cell in cells:
        cell_ids = [iid for iid in classified if items_by_id[iid].payload["cell"] == cell]
        drops: dict[str, int] = {}
        admitted = []
        for iid in sorted(cell_ids):
            c = classified[iid]
            if c["class"] == "valid" and c["score"] >= 50:
                admitted.append({"row_id": items_by_id[iid].payload["row_id"], "score": c["score"]})
            elif c["class"] == "valid":
                drops["below_threshold"] = drops.get("below_threshold", 0) + 1
            else:
                drops[c["class"]] = drops.get(c["class"], 0) + 1
        family = cm.CELL_FAMILY[cell]
        payload = {
            "cell": cell,
            "family": family,
            "n_items": len(cell_ids),
            "n_admitted": len(admitted),
            "admit_threshold": 50,
            "drop_counts": drops,
            "admitted": admitted,
            "judge_model": cm.JUDGE_MODEL,
            "rubric_sha": _rubric_sha(),
            "metadata": cm.run_metadata(),
        }
        cm.atomic_write_json(kept_dir / f"{cell}.json", payload)
        print(
            f"[admission] {cell}: {len(admitted)}/{len(cell_ids)} admitted "
            f"drops={json.dumps(drops)}",
            flush=True,
        )
    tally = _tally(classified, items_by_id)
    cm.atomic_write_json(
        out_root / "judge" / "admission_summary.json",
        {"tally": tally, "metadata": cm.run_metadata()},
    )
    return 0


def _run_congruence(args, mined: dict[str, dict]) -> int:
    out_root = Path(args.out_root)
    _require_pilot_pass(args, out_root)
    items, per_cell_rows = _congruence_items(args, mined)
    items_by_id = {it.item_id: it for it in items}
    force = None if args.transport == "auto" else args.transport
    print(f"[congruence] dispatching {len(items)} calls (transport={args.transport})", flush=True)
    classified = _dispatch(items, args, force_path=force, cache_tag=args.wave)
    _persist_raw(args, classified, items_by_id, "judge_congruence")
    cong_dir = out_root / "judge" / "congruence"
    for cell, row_ids in per_cell_rows.items():
        rows_out = []
        drops: dict[str, int] = {}
        for rid in row_ids:
            scores = []
            for d in range(args.congruence_draws):
                c = classified.get(f"cong|{rid}|d{d}")
                if c is None:
                    continue  # sliced away by --max-items
                if c["class"] == "valid":
                    scores.append(c["score"])
                else:
                    drops[c["class"]] = drops.get(c["class"], 0) + 1
            if scores:
                rows_out.append(
                    {
                        "row_id": rid,
                        "mean_score": sum(scores) / len(scores),
                        "n_draws_valid": len(scores),
                    }
                )
            else:
                drops["row_no_valid_draws"] = drops.get("row_no_valid_draws", 0) + 1
        means = sorted(r["mean_score"] for r in rows_out)
        median = means[len(means) // 2] if means else None
        payload = {
            "cell": cell,
            "n_rows_scored": len(rows_out),
            "n_draws_per_row": args.congruence_draws,
            "median_of_row_means": median,
            "expected_median_note": "expected >= ~80 (plan §4.5); REPORTED, never gated",
            "drop_counts": drops,
            "rows": rows_out,
            "judge_model": cm.JUDGE_MODEL,
            "rubric_sha": _rubric_sha(),
            "metadata": cm.run_metadata(),
        }
        cm.atomic_write_json(cong_dir / f"{cell}.json", payload)
        print(
            f"[congruence] {cell}: n={len(rows_out)} median={median} drops={json.dumps(drops)}",
            flush=True,
        )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--wave", required=True, choices=["admission", "congruence"])
    ap.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="run an N-call pilot gate instead of the production wave",
    )
    ap.add_argument(
        "--transport",
        default=None,
        choices=["sync", "batch", "auto"],
        help="default: sync for --pilot, auto (dispatcher crossover) otherwise",
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--mined-dir", default=str(cm.RAW_ROOT_DEFAULT / "sega_mined"))
    ap.add_argument("--segb-dir", default=str(cm.RAW_ROOT_DEFAULT / "segb"))
    ap.add_argument("--kept-dir", default=str(cm.LEDGER_ROOT / "kept"))
    ap.add_argument("--out-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--raw-root", default=str(cm.RAW_ROOT_DEFAULT))
    ap.add_argument(
        "--cache-dir", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "judge_cache")
    )
    ap.add_argument(
        "--checkpoint-dir", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "judge_ckpt")
    )
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument("--congruence-rows", type=int, default=500)
    ap.add_argument("--congruence-draws", type=int, default=3)
    ap.add_argument(
        "--max-items", type=int, default=0, help="cap total items (bounded smoke slices)"
    )
    ap.add_argument("--skip-upload", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.transport is None:
        args.transport = "sync" if args.pilot else "auto"
    _stage_inputs(args)
    mined = _load_mined_rows(Path(args.mined_dir))
    if args.pilot:
        if args.transport == "auto":
            raise SystemExit("--pilot requires an explicit --transport sync|batch")
        rc = _run_pilot(args, mined)
    elif args.wave == "admission":
        rc = _run_admission(args, mined)
    else:
        rc = _run_congruence(args, mined)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
