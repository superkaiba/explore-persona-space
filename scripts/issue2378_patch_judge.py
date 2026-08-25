#!/usr/bin/env python3
"""Issue #2378 `causal-patching-arms` — VM-side dual-rubric judge wave.

F_beh instrument (NEW — this task's CONGRUENCE rubric is scene-tied and cannot
score chat-target answers; rubric text in ``issue2378_patch_common``):
context-blind reply-only scoring with judge = claude-sonnet-4-5-20250929.

Per judged rollout row, up to three rubrics:
- ``persona``   (judge_B): in-character strength for the row's STORY character
  (the pair's character — the persona whose carry is under test).
- ``assistant`` (judge_A): default-assistant strength. The #2094 dual-rubric
  contrast is Δ = (persona - assistant)/100 (``fmetrics.delta_contrast``,
  computed downstream in issue2378_patch_analysis.py).
- ``coherence`` — patched/prefill/confirm rows only; the inherited #2094 gate
  (F_beh rows below COHERENCE_THRESHOLD=60 are dropped-with-count downstream).

Scope: chat~story rows only (arm (a) + its controls + their anchors). The
chat~plain arm (b) has no judgeable persona contrast (both framings answer as
an assistant) — it is F_act-only by design, stated in the round report.

Transport: ``llm/api_dispatch.dispatch_calls`` (rubric-keyed cache, batch
checkpointing, 429/529 retry) — the same machinery as issue2378_judge, whose
parse/classify helpers are imported verbatim. Pilot gate (llm-judging rules
23/26): ``--pilot N`` runs N calls STRATIFIED across every realized
(row kind x rubric) class on the production instrument and FAILS on any
``max_tokens`` stop, a parse-fail rate >= 2% of ANSWERED calls, or any class
with zero answered calls (transport losses are vacuous, never a PASS); the
production wave refuses to run without a PASS report on disk whose transport
AND instrument sha match the wave's (rule 26(c) parity; or --skip-pilot-gate,
recorded in the wave report). The wave fold is rebuild-from-cache resumable
(fresh fold dir, published by an os.replace'd manifest pointer readers key on
— a rerun never truncates or refuses, and a crash mid-write leaves the prior
published fold intact; readers refuse when no manifest is published).

Inputs: the harvested pod dirs under --patch-root (bank/anchors/grid/confirm
rollout JSONLs). Outputs: per-row scores JSONL + a wave report under
eval_results/issue_2378/causal-patching-arms/judge/, raw judge rows persisted
to HF under the causal_patching judge stage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

import issue2378_common as cm  # noqa: E402
import issue2378_judge as J  # noqa: E402  (parse/classify reuse — verbatim)
import issue2378_patch_common as pc  # noqa: E402

PILOT_PARSE_FAIL_MAX = 0.02  # llm-judging rule 23: per-wave parse-fail < 2%
JUDGE_TEMPERATURE = 1.0  # rule 4 multi-draw temperature — part of the instrument (r18)


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── item construction ───────────────────────────────────────────────────────


def _iter_rollout_rows(patch_root: Path):
    """(kind, row) over anchors/grid/confirm rollout rows (kept rows only)."""
    for kind, sub in (("anchors", "anchors"), ("grid", "grid"), ("confirm", "confirm")):
        d = patch_root / sub / "rollouts"
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl")):
            for row in cm.iter_jsonl(p):
                yield kind, row


def _row_char(kind: str, row: dict) -> str | None:
    """The story character whose persona the row is judged against, or None
    for rows outside the judged (chat~story) scope."""
    if kind == "anchors":
        framing, qid = row["ctx_id"].split(":", 1)
        if framing == "plain":
            return None  # plain anchors serve only the F_act-only arm (b)
        cell = qid.rsplit("_", 1)[0]  # qid = f"{storyq_cell}_{k:02d}"
        return cm.CELL_CHARACTER[cell]
    if row["pair_type"] != "chat~story":
        return None
    return row["char"]


def _row_id(kind: str, row: dict) -> str:
    if kind == "anchors":
        return f"anchors|{row['ctx_id']}|d{row['draw']}"
    return f"{kind}|{row['cell_id']}|d{row['draw']}"


def build_items(patch_root: Path, max_items: int = 0) -> list:
    """DispatchItems over (row, rubric) pairs; stable item ids key the cache."""
    from explore_persona_space.llm.api_dispatch import DispatchItem

    items: list = []
    n_rows = 0
    for kind, row in _iter_rollout_rows(patch_root):
        if row.get("drop_reason") is not None or not row.get("answer"):
            continue
        char = _row_char(kind, row)
        if char is None:
            continue
        n_rows += 1
        rid = _row_id(kind, row)
        rubrics = ["persona", "assistant"]
        if kind in ("grid", "confirm"):
            rubrics.append("coherence")
        for rubric in rubrics:
            items.append(
                DispatchItem(
                    item_id=f"{rid}|{rubric}",
                    payload={
                        "rubric": rubric,
                        "character": char,
                        "description": cm.PERSONAS[char],
                        "reply": row["answer"],
                    },
                )
            )
        if max_items and len(items) >= max_items:
            break
    _log(f"[items] {len(items)} judge calls over {n_rows} kept rows")
    if not items:
        raise RuntimeError(f"no judgeable rows under {patch_root} (empty selection — fail loud)")
    return items


def _build_request(item) -> dict:
    p = item.payload
    if p["rubric"] == "persona":
        user = pc.PERSONA_RUBRIC
        user = user.replace("{name}", p["character"]).replace("{description}", p["description"])
    elif p["rubric"] == "assistant":
        user = pc.ASSISTANT_RUBRIC
    else:
        user = pc.COHERENCE_RUBRIC
    user = user.replace("{reply}", p["reply"])
    return {
        "model": cm.JUDGE_MODEL,
        "max_tokens": cm.JUDGE_MAX_TOKENS,
        "temperature": JUDGE_TEMPERATURE,
        "system": pc.PATCH_JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user}],
    }


def _dispatch(items, args, force_path: str | None, cache_tag: str) -> dict:
    """dispatch_calls + ONE sync re-drive of transport-class losses.

    Diff vs the named reference ``issue2378_judge._dispatch`` (reuse rule —
    genuinely new estimator/plumbing code diffs against the reference): the
    request factory is this wave's dual-rubric one and the wave tag is fixed;
    the transport re-drive, hard-error fail-loud, cache/checkpoint keying, and
    classify path are IDENTICAL (classify itself is imported). No
    permissiveness broadening: the same hard-error classes still raise.
    """
    from explore_persona_space.llm.api_dispatch import dispatch_calls

    common = {
        "model": cm.JUDGE_MODEL,
        "build_request": _build_request,
        "parse_response": J._parse_judge,
        "cache_dir": Path(args.cache_dir) / cache_tag,
        "checkpoint_dir": Path(args.checkpoint_dir) / J._checkpoint_partition(cache_tag, items),
    }
    results = asyncio.run(dispatch_calls(items, force_path=force_path, **common))
    classified = {iid: J._classify(res) for iid, res in results.items()}
    redrive = [iid for iid, c in classified.items() if c["class"] == "transport"]
    if redrive:
        _log(f"[patch-judge] re-driving {len(redrive)} transport-class losses (sync)")
        sub = [it for it in items if it.item_id in set(redrive)]
        results2 = asyncio.run(dispatch_calls(sub, force_path="sync", **common))
        for iid, res in results2.items():
            classified[iid] = J._classify(res)
    for iid, c in classified.items():
        if c["class"] == "transport":
            c["class"] = "transport_loss"
    hard = [iid for iid, c in classified.items() if c["class"] == "hard_error"]
    if hard:
        sample = {iid: classified[iid]["reasoning"] for iid in hard[:5]}
        raise RuntimeError(
            f"{len(hard)} non-transient request errors — pipeline bug, fail loud. "
            f"Sample: {json.dumps(sample)}"
        )
    return classified


# ── pilot gate ──────────────────────────────────────────────────────────────


def _pilot_report_path(out_dir: Path) -> Path:
    return out_dir / "pilot_report.json"


def _item_class(item_id: str) -> tuple[str, str]:
    """(kind, rubric) — the pilot stratification class of one judge item.
    item_id shapes: ``anchors|<ctx>|d<k>|<rubric>`` / ``grid|<cell>|d<k>|<rubric>``."""
    return item_id.split("|", 1)[0], item_id.rsplit("|", 1)[1]


def _item_arm(item_id: str) -> str:
    """The patch ARM of one judge item, for within-class pilot interleaving.

    grid/confirm ids embed the cell_id whose LEADING field is the arm
    (``grid|<arm>|<variant>|<src>-><tgt>|<qid>|d<k>|<rubric>``); anchors rows
    are unpatched (no arm) — keyed by their ctx framing instead, which is a
    single realized group in the judged (chat~story) scope."""
    parts = item_id.split("|")
    if parts[0] == "anchors":
        return parts[1].split(":", 1)[0]
    return parts[1]


def _instrument_sha() -> str:
    """sha256 of the judge INSTRUMENT (model, cap, temperature, system, all
    rubrics) — a pilot PASS is valid only for this exact instrument
    (llm-judging rule 26); the wave refuses on a mismatch."""
    import hashlib

    payload = json.dumps(
        [
            cm.JUDGE_MODEL,
            cm.JUDGE_MAX_TOKENS,
            JUDGE_TEMPERATURE,
            pc.PATCH_JUDGE_SYSTEM,
            pc.PERSONA_RUBRIC,
            pc.ASSISTANT_RUBRIC,
            pc.COHERENCE_RUBRIC,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _item_set_sha(items: list) -> str:
    """Canonical sha256 over the SORTED item-id set — binds a pilot PASS to
    the exact wave it gates (r18 patch-judge-pilot-arm-and-binding-residual:
    a pilot run against one harvest must not green-light a wave over a
    different harvested row set)."""
    import hashlib

    payload = "\n".join(sorted(it.item_id for it in items))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stratified_pilot_items(items: list, n: int) -> list:
    """Deterministic round-robin per-(kind x rubric) selection of n pilot items.

    The pilot must span EVERY realized row kind (anchors/grid/confirm) and
    EVERY rubric in the wave (llm-judging rule 26; r17 Claude M2 / codex
    patch-judge-pilot-vacuous: a head-of-list cap drew chat-anchors rows only
    and never piloted the coherence rubric). WITHIN each class the queue
    interleaves ARMS round-robin (r18: a plain item_id sort put the
    lexicographically-first arm — null — at the head, so a small pilot never
    reached steered/prefill/within replies)."""
    by_class: dict[tuple[str, str], list] = {}
    for it in items:
        by_class.setdefault(_item_class(it.item_id), []).append(it)
    for key, lst in by_class.items():
        by_arm: dict[str, list] = {}
        for it in lst:
            by_arm.setdefault(_item_arm(it.item_id), []).append(it)
        for alst in by_arm.values():
            alst.sort(key=lambda it: it.item_id)
        arm_order = sorted(by_arm)
        interleaved: list = []
        j = 0
        while any(by_arm[a] for a in arm_order):
            alst = by_arm[arm_order[j % len(arm_order)]]
            if alst:
                interleaved.append(alst.pop(0))
            j += 1
        by_class[key] = interleaved
    order = sorted(by_class)
    target = min(n, len(items))
    out: list = []
    idx = 0
    while len(out) < target:
        lst = by_class[order[idx % len(order)]]
        if lst:
            out.append(lst.pop(0))
        idx += 1
    return out


def pilot_report_from_classified(classified: dict[str, dict], transport: str) -> dict:
    """Pure pilot verdict: per-class tallies + the gate booleans.

    FAIL (ok=False) on ANY of: a max_tokens stop; parse-fail rate over the
    ANSWERED denominator >= 2%; zero answered calls overall; any realized
    (kind x rubric) class with ZERO answered calls — transport losses /
    api-refusals produce no verdict, so an all-transport class is a VACUOUS
    pilot, never a PASS (r17 codex patch-judge-pilot-vacuous)."""
    per_class: dict[str, dict] = {}
    for iid, c in classified.items():
        key = "|".join(_item_class(iid))
        rec = per_class.setdefault(
            key, {"n": 0, "n_answered": 0, "n_parse_fail": 0, "n_max_tokens_stops": 0}
        )
        rec["n"] += 1
        if c["class"] not in ("transport_loss", "api_refusal"):
            rec["n_answered"] += 1
        if c["class"] in ("parse_fail", "truncation"):
            rec["n_parse_fail"] += 1
        if c.get("stop_reason") == "max_tokens":
            rec["n_max_tokens_stops"] += 1
    n = len(classified)
    n_answered = sum(rec["n_answered"] for rec in per_class.values())
    n_cap = sum(rec["n_max_tokens_stops"] for rec in per_class.values())
    n_parse_fail = sum(rec["n_parse_fail"] for rec in per_class.values())
    vacuous = sorted(k for k, rec in per_class.items() if rec["n_answered"] == 0)
    ok = (
        n_cap == 0
        and n_answered > 0
        and (n_parse_fail / max(n_answered, 1)) < PILOT_PARSE_FAIL_MAX
        and not vacuous
    )
    return {
        "n": n,
        "n_answered": n_answered,
        "n_max_tokens_stops": n_cap,
        "n_parse_fail": n_parse_fail,
        "parse_fail_rate": n_parse_fail / max(n_answered, 1),
        "classes": {k: per_class[k] for k in sorted(per_class)},
        "vacuous_classes": vacuous,
        "transport": transport,
        "instrument_sha": _instrument_sha(),
        "ok": ok,
    }


def run_pilot(args, out_dir: Path) -> int:
    all_items = build_items(Path(args.patch_root))
    classes = sorted({"|".join(_item_class(it.item_id)) for it in all_items})
    if int(args.pilot) < len(classes):
        raise SystemExit(
            f"--pilot {args.pilot} < {len(classes)} realized (kind x rubric) classes "
            f"{classes} — raise the pilot size so every class is spanned (rule 26)"
        )
    items = stratified_pilot_items(all_items, int(args.pilot))
    _log(
        f"[pilot] {len(items)}/{len(all_items)} stratified items over classes "
        f"{sorted({'|'.join(_item_class(it.item_id)) for it in items})}"
    )
    classified = _dispatch(items, args, args.transport, f"pilot-{args.transport}")
    report = pilot_report_from_classified(classified, args.transport)
    report["n_items_total"] = len(all_items)
    report["item_set_sha"] = _item_set_sha(all_items)  # binds the PASS to this wave (r18)
    report["metadata"] = cm.run_metadata({"phase": "patch_judge_pilot"})
    # Persist the pilot's raw classified rows (rule 26: read stop_reasons from
    # persisted results, never truncated failure-log text).
    with (out_dir / "pilot_rows.jsonl").open("w", encoding="utf-8") as fh:
        for iid in sorted(classified):
            c = classified[iid]
            fh.write(
                json.dumps({"item_id": iid, **{k: v for k, v in c.items() if k != "parsed"}}) + "\n"
            )
    cm.atomic_write_json(_pilot_report_path(out_dir), report)
    _log(
        f"[pilot] n={report['n']} answered={report['n_answered']} "
        f"max_tokens_stops={report['n_max_tokens_stops']} parse_fail={report['n_parse_fail']} "
        f"vacuous={report['vacuous_classes']} ok={report['ok']}"
    )
    return 0 if report["ok"] else 7


# ── production wave ─────────────────────────────────────────────────────────


def _fold_manifest_path(out_dir: Path) -> Path:
    return out_dir / "fold_manifest.json"


def read_fold_manifest(out_dir: Path) -> dict:
    """The published fold pointer, fail-loud — the ONLY reader entrypoint.

    A fold is published by the atomic ``os.replace`` of the manifest AFTER
    its fold dir is fully written (r18 patch-judge-fold-publish-window), so:
    no manifest = no published fold (refuse — a half-written fold dir is
    never visible through this path), and a manifest naming a missing/partial
    fold dir is a pipeline bug (refuse loud, never read residue). Returns the
    manifest dict plus resolved ``fold_path`` / ``scores_path``."""
    path = _fold_manifest_path(out_dir)
    if not path.exists():
        raise RuntimeError(
            f"no published judge fold: missing manifest {path} — run the wave (a fold "
            "without a manifest is unpublished; never read fold dirs directly)"
        )
    man = json.loads(path.read_text(encoding="utf-8"))
    fold_path = out_dir / man["fold_dir"]
    scores_path = fold_path / "scores.jsonl"
    if not scores_path.is_file():
        raise RuntimeError(
            f"fold manifest {path} names {fold_path} but {scores_path} is missing — "
            "half-published fold (pipeline bug), refuse to read"
        )
    return {**man, "fold_path": fold_path, "scores_path": scores_path}


def _write_fold(out_dir: Path, classified: dict[str, dict]) -> tuple[dict, dict, Path]:
    """Build scores.jsonl + raw shards in a FRESH uniquely-named fold dir,
    then atomically publish it by ``os.replace``-ing the manifest pointer
    readers key on (r18 patch-judge-fold-publish-window: the prior
    rmtree-then-rename publish left a crash window where readers saw new
    ``raw/`` beside the old ``scores.jsonl``, or neither). Results are
    cache/checkpoint-served upstream, so a rerun deterministically rebuilds a
    complete fresh fold; superseded fold dirs + legacy top-level outputs are
    reaped only AFTER the pointer flips."""
    import shutil
    import uuid

    fold_name = f"fold_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    fold_dir = out_dir / fold_name
    (fold_dir / "raw").mkdir(parents=True)
    tally: dict = {"kept": 0, "dropped": 0, "transport_loss": 0, "by_class": {}}
    writer = cm.ShardWriter(fold_dir / "raw", "judge_rows")
    with (fold_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for iid in sorted(classified):
            c = classified[iid]
            score = c.get("score")
            kept = c["class"] == "valid" and isinstance(score, int) and 0 <= score <= 100
            tally["kept" if kept else "dropped"] += 1
            if c["class"] == "transport_loss":
                tally["transport_loss"] += 1
            tally["by_class"][c["class"]] = tally["by_class"].get(c["class"], 0) + 1
            row_id, rubric = iid.rsplit("|", 1)
            fh.write(
                json.dumps(
                    {
                        "item_id": iid,
                        "row_id": row_id,
                        "rubric": rubric,
                        "score": score if kept else None,
                        "class": c["class"],
                        "kept": kept,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            writer.write({"item_id": iid, **{k: v for k, v in c.items() if k != "parsed"}})
    shard_info = writer.close()
    # Publish: the manifest os.replace (inside atomic_write_json) is the
    # single commit point — the fold dir above is complete before it flips.
    cm.atomic_write_json(
        _fold_manifest_path(out_dir),
        {
            "fold_dir": fold_name,
            "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_rows": shard_info["n_rows"],
            "tally": tally,
        },
    )
    # Post-publish reap: superseded fold dirs + the legacy pre-manifest
    # top-level outputs (crash here leaves only dead residue, never a
    # wrong read — the manifest already points at the fresh fold).
    for stale in sorted(out_dir.glob("fold_*")):
        if stale.is_dir() and stale.name != fold_name:
            shutil.rmtree(stale)
    legacy_raw = out_dir / "raw"
    if legacy_raw.is_dir():
        shutil.rmtree(legacy_raw)
    (out_dir / "scores.jsonl").unlink(missing_ok=True)
    return tally, shard_info, fold_dir


def run_wave(args, out_dir: Path) -> int:
    items = build_items(Path(args.patch_root))
    if not args.skip_pilot_gate:
        rp = _pilot_report_path(out_dir)
        if not rp.exists():
            raise SystemExit(f"pilot gate: no pilot report at {rp} — run --pilot first")
        rep = json.loads(rp.read_text(encoding="utf-8"))
        if not rep.get("ok"):
            raise SystemExit(f"pilot gate: pilot report at {rp} is FAIL — fix the instrument")
        # Transport parity (llm-judging rule 26(c), #2152): the pilot must
        # have RUN the wave's transport — a pilot-gated wave pins it.
        if args.transport == "auto":
            raise SystemExit(
                "pilot gate: a pilot-gated wave requires a PINNED --transport sync|batch "
                "matching the pilot's (rule 26(c) transport parity — never count-routed auto)"
            )
        if rep.get("transport") != args.transport:
            raise SystemExit(
                f"pilot gate: pilot transport {rep.get('transport')!r} != wave transport "
                f"{args.transport!r} — re-pilot on the wave's transport (rule 26(c))"
            )
        if rep.get("instrument_sha") != _instrument_sha():
            raise SystemExit(
                "pilot gate: the judge instrument changed since the pilot PASS "
                f"(pilot sha {rep.get('instrument_sha')!r} != live {_instrument_sha()!r}) "
                "— re-pilot (rule 26)"
            )
        if rep.get("item_set_sha") != _item_set_sha(items):
            raise SystemExit(
                "pilot gate: the wave's item-id set differs from the piloted set "
                f"(pilot {rep.get('item_set_sha')!r} != wave {_item_set_sha(items)!r}) "
                "— the harvested rows changed since the pilot PASS; re-pilot (r18)"
            )
    t0 = time.time()
    classified = _dispatch(
        items, args, None if args.transport == "auto" else args.transport, "wave"
    )
    tally, shard_info, fold_dir = _write_fold(out_dir, classified)
    report = {
        "fold_dir": fold_dir.name,
        "n_items": len(items),
        "tally": tally,
        "wall_s": time.time() - t0,
        "judge_model": cm.JUDGE_MODEL,
        "max_tokens": cm.JUDGE_MAX_TOKENS,
        "transport": args.transport,
        "skip_pilot_gate": bool(args.skip_pilot_gate),
        "instrument_sha": _instrument_sha(),
        "raw_shards": shard_info,
        "metadata": cm.run_metadata({"phase": "patch_judge_wave"}),
    }
    cm.atomic_write_json(out_dir / "wave_report.json", report)
    if not args.skip_upload:
        cm.upload_stage_dir(fold_dir / "raw", f"{pc.HF_STAGE_PREFIX}{args.hf_suffix}/judge_persona")
    _log(f"[wave] items={len(items)} tally={tally}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--patch-root", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "patch_round")
    )
    ap.add_argument(
        "--out-dir",
        default=str(cm.REPO_ROOT / "eval_results" / "issue_2378" / pc.LEDGER_SUBDIR / "judge"),
    )
    ap.add_argument("--pilot", type=int, default=0, help="run an N-call pilot instead of the wave")
    ap.add_argument("--transport", choices=["sync", "batch", "auto"], default="auto")
    ap.add_argument("--skip-pilot-gate", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--hf-suffix", default="")
    ap.add_argument(
        "--cache-dir", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "patch_judge_cache")
    )
    ap.add_argument(
        "--checkpoint-dir", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "patch_judge_ckpt")
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.import_check:
        import issue2378_patch_common as _pc_mod

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__, _pc_mod.__file__)
        raise SystemExit(0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        if args.transport == "auto":
            raise SystemExit("--pilot requires an explicit --transport sync|batch")
        raise SystemExit(run_pilot(args, out_dir))
    raise SystemExit(run_wave(args, out_dir))


if __name__ == "__main__":
    main()
