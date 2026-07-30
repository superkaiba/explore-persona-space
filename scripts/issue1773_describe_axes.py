#!/usr/bin/env python
"""Issue #1773 Phases 2-3 — description + five judged axes over the Batch API.

Stages (all routed through `eval.judge_dispatch.dispatch_judge_items` — the
2,000-item shard ceiling / 4-in-flight / #1019 resumable re-dispatch machinery
is REUSED UNCHANGED, per the task's build-vs-adopt decision):

  --stage describe  one item per feature (evidence = EX+ 40 marked + EX- 20 +
                    OUT block; per-token values omitted; no CoT): free-text
                    description + confidence, 1 draw, max_tokens=700, temp 1.0.
                    Raw judge text retained (keep_raw_judge_text) + uploaded.
  --stage axes      (feature x axis x draw) items: ONE axis per call, forced
                    single choice, label order PERMUTED per draw (deterministic
                    fn of (feat_id, axis, draw) rendered into the USER message
                    -> per-draw cache keys differ), reason-then-label JSON,
                    max_tokens=400, temp 1.0, 5 draws, majority vote >=3 else
                    `unresolved`; drop-never-coerce with the content-vs-transport
                    split (llm-judging rules 9/24); varying-n Fleiss kappa.
  --stage pilot     seeded 500-feature stratified slice (activity deciles);
                    runs describe+axes on the slice and writes pilot_gate.json
                    (PROCEED iff >=3 of 5 axes kappa >= 0.2 — plan §7 gate 2).

Spend guards: describe/axes refuse >`--limit` items unless --full is passed
(the 16k production dispatch is the experimenter's job). --render-only writes
golden prompts with zero API calls; --force-batch drives the Batch path at any
N (the 5-item live smoke).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1773_common as CM  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_SMOKE_LIMIT = 8
PILOT_N = 500  # plan §7 gate 2 (allowed band 300-800)
PILOT_KAPPA_FLOOR = 0.2
PILOT_AXES_FLOOR = 3

# ── full-dictionary group sharding (phases 2-3 at 131,072 features) ──────────
# `dispatch_judge_items` holds the WHOLE item set in memory and writes it to
# one `items.json` per dispatch dir (its resume contract compares the persisted
# map byte-for-byte). At the full dictionary the axes stage is 131,072 x 5 axes
# x 5 draws = 3,276,800 items of ~4 KB of rendered evidence -> ~13 GB in one
# map: not dispatchable as a single call. So the stages shard by FEATURE
# GROUP: each group is its own dispatch (own checkpoint dir, own output shard,
# own done-sentinel) and groups run sequentially with resume-by-sentinel.
# Sizes are chosen so one group's in-flight request count stays well under the
# Tier-4 batch queue cap while still saturating the 400k-OTPM ceiling that sets
# the wall-clock floor: an axes group is 4,096 x 25 = 102,400 requests
# (52 sub-batches of 2,000) ~ 25M output tokens ~ 64 min at the cap.
DESCRIBE_GROUP_FEATURES = 32_768  # 4 groups at full dictionary (1 item/feature)
AXES_GROUP_FEATURES = 4_096  # 32 groups at full dictionary (25 items/feature)


def feature_groups(feat_ids: list[int], group_size: int) -> list[list[int]]:
    """Split a sorted feature-id list into contiguous groups (0 = one group)."""
    if group_size <= 0 or len(feat_ids) <= group_size:
        return [list(feat_ids)]
    return [list(feat_ids[i : i + group_size]) for i in range(0, len(feat_ids), group_size)]


def _group_done(path: Path) -> bool:
    return path.exists()


def _write_group_done(path: Path, payload: dict) -> None:
    tmp = path.parent / f".tmp_{path.name}"
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{path.name}"
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_packets(evidence_dir: Path, include_controls: bool = False) -> dict[int, dict]:
    """Load per-feature evidence packets from the assembled manifests."""
    packets: dict[int, dict] = {}
    pats = ["evidence.shard*.jsonl"] + (
        ["evidence_randdir.shard*.jsonl"] if include_controls else []
    )
    man_dir = evidence_dir / "evidence_manifests"
    for pat in pats:
        for p in sorted(man_dir.glob(pat)):
            for r in CM.iter_jsonl(p):
                packets[int(r["feat_id"])] = r
    assert packets, f"no evidence packets under {man_dir}"
    return packets


def _dispatch(
    items,
    *,
    system: str,
    max_tokens: int,
    checkpoint_dir: Path,
    force_batch: bool,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Route one judge dispatch (sync/batch decided by N; --force-batch pins the
    Batch path via threshold_base=1) at temperature 1.0 with raw-text retention."""
    from explore_persona_space.eval.judge_dispatch import (
        dispatch_judge_items,
        graded_temperature,
        keep_raw_judge_text,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        judge_system_prompt=system,
        max_tokens=max_tokens,
        checkpoint_dir=checkpoint_dir,
        dry_run=dry_run,
    )
    if force_batch:
        kwargs["threshold_base"] = 1
    with graded_temperature(CM.JUDGE_TEMPERATURE), keep_raw_judge_text():
        return dispatch_judge_items(items, **kwargs)


def _classify_error(res: dict) -> str:
    """Split a returned error dict: transport (retried/exhausted upstream —
    rule 24) vs content drop (rule 9)."""
    from explore_persona_space.eval.batch_judge import is_transport_error_dict

    return "transport" if is_transport_error_dict(res) else "content"


def _write_raw(results: dict[str, dict], out_path: Path) -> None:
    """Persist raw judge text (upload-policy: judge outputs upload ALWAYS)."""
    rows = [
        {"custom_id": cid, "raw_text": res.get("_raw_text")}
        for cid, res in sorted(results.items())
        if isinstance(res, dict) and res.get("_raw_text")
    ]
    CM.write_jsonl_sharded(rows, out_path.parent, out_path.stem)


def _upload_dir(local_dir: Path, prefix: str) -> None:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    hub.assert_hub_dir_filecounts(
        local_dir, f"{CM.HF_PREFIX}/{prefix}", allow_patterns=["*.jsonl", "*.json"]
    )
    hub.retry_transient(
        lambda: HfApi().upload_folder(
            folder_path=str(local_dir),
            repo_id=CM.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{CM.HF_PREFIX}/{prefix}",
            allow_patterns=["*.jsonl", "*.json"],
        ),
        what=f"{prefix} upload",
    )


# ── describe ─────────────────────────────────────────────────────────────────


def exclude_zero_evidence(packets: dict[int, dict], out_root: Path) -> dict[int, dict]:
    """Drop features with NO activating windows from the dispatch set.

    2,560 of the 131,072 dictionary features never fire on the evaluation
    corpus (phase 0 `n_features_dead_in_fit`), so their packet renders an
    EMPTY activating-examples block: the prompt has nothing to describe and
    the call would be paid noise. They are RECORDED here as `no_evidence` at
    zero API cost and survive as a labelled zero-cost row in the reporting
    tables. Returns the packets to dispatch.
    """
    keep = {f: p for f, p in packets.items() if p.get("ex_pos")}
    dropped = sorted(f for f, p in packets.items() if not p.get("ex_pos"))
    if dropped:
        rec = out_root / "labels" / "no_evidence_features.json"
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(
            json.dumps(
                {
                    "reason": (
                        "zero activating windows — the describe/axis prompt would render an "
                        "empty activating-examples block; excluded from phases 2-3 at zero "
                        "API cost and reported as its own stratum"
                    ),
                    "n_excluded": len(dropped),
                    "n_dispatched": len(keep),
                    "feat_ids": dropped,
                    **CM.repro_meta(),
                },
                indent=1,
            )
        )
        _log(
            f"[scope] excluded {len(dropped)} zero-evidence features "
            f"(recorded no_evidence, $0); dispatching {len(keep)}"
        )
    return keep


def build_describe_items(packets: dict[int, dict]) -> list[tuple[str, str, str, str]]:
    """(custom_id, question, completion, user_msg) per feature (JudgeItem)."""
    items = []
    for feat_id, pk in sorted(packets.items()):
        user = CM.build_describe_user_msg(pk)
        items.append((f"f{feat_id}-desc", f"feat:{feat_id}", "", user))
    return items


def parse_describe_result(res: object) -> dict | None:
    """Validated describe return: dict with non-empty `description`; confidence
    kept when a 0-100 int, else None (drop-never-coerce is label-side; a
    missing confidence does not drop a valid description)."""
    if not isinstance(res, dict):
        return None
    desc = res.get("description")
    if not isinstance(desc, str) or not desc.strip():
        return None
    conf = res.get("confidence")
    conf_ok = conf if isinstance(conf, int | float) and 0 <= float(conf) <= 100 else None
    return {"description": desc.strip(), "confidence": conf_ok}


def stage_describe_grouped(args, packets: dict[int, dict]) -> int:
    """Describe over feature groups: one dispatch + output shard + sentinel per
    group, resumable by sentinel, merged into `descriptions.jsonl` at the end."""
    out_dir = args.out_root / "labels"
    grp_dir = out_dir / "describe_groups"
    grp_dir.mkdir(parents=True, exist_ok=True)
    packets = exclude_zero_evidence(packets, args.out_root)
    groups = feature_groups(sorted(packets), args.describe_group_size)
    _log(f"[describe] {len(packets)} features -> {len(groups)} groups")
    totals = {"n_items": 0, "n_ok": 0, "content": 0, "transport": 0}
    for gi, gfeats in enumerate(groups):
        shard = grp_dir / f"descriptions.g{gi:04d}.jsonl"
        done = grp_dir / f"descriptions.g{gi:04d}.done.json"
        if _group_done(done) and not args.regroup:
            d = json.loads(done.read_text())
            for k in totals:
                totals[k] += int(d.get(k, 0))
            _log(f"[describe] group {gi + 1}/{len(groups)} SKIP (done, n_ok={d.get('n_ok')})")
            continue
        sub = {f: packets[f] for f in gfeats}
        items = build_describe_items(sub)
        _log(f"[describe] group {gi + 1}/{len(groups)}: dispatching {len(items)} items")
        results = _dispatch(
            items,
            system=CM.DESCRIBER_SYSTEM,
            max_tokens=CM.DESCRIBE_MAX_TOKENS,
            checkpoint_dir=args.work / "judge_checkpoints" / "describe" / f"g{gi:04d}",
            force_batch=args.force_batch,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        rows, drops = [], {"content": 0, "transport": 0}
        for cid, _q, _c, user in items:
            res = results.get(cid)
            if isinstance(res, dict) and res.get("error"):
                drops[_classify_error(res)] += 1
                continue
            parsed = parse_describe_result(res)
            if parsed is None:
                drops["content"] += 1
                continue
            feat_id = int(cid[1:].rsplit("-", 1)[0])
            rows.append({"feat_id": feat_id, **parsed, "prompt_sha16": CM.sha16(user)})
        _write_jsonl(rows, shard)
        _write_raw(results, args.work / "judge_raw" / f"describe_raw_g{gi:04d}")
        payload = {"group": gi, "n_items": len(items), "n_ok": len(rows), **drops}
        _write_group_done(done, payload)
        for k in totals:
            totals[k] += int(payload.get(k, 0))
        _log(f"[describe] group {gi + 1}/{len(groups)} done: {len(rows)}/{len(items)} ok {drops}")
    if args.dry_run:
        return 0
    merged: list[dict] = []
    for p in sorted(grp_dir.glob("descriptions.g*.jsonl")):
        merged.extend(CM.iter_jsonl(p))
    _write_jsonl(merged, out_dir / "descriptions.jsonl")
    (out_dir / "describe_meta.json").write_text(
        json.dumps(
            {
                **CM.repro_meta(),
                "grouped": True,
                "n_groups": len(groups),
                "group_size_features": args.describe_group_size,
                "n_items": totals["n_items"],
                "n_ok": len(merged),
                "drops": {"content": totals["content"], "transport": totals["transport"]},
                "max_tokens": CM.DESCRIBE_MAX_TOKENS,
                "rubric_sha16": CM.sha16(CM.DESCRIBER_SYSTEM),
            },
            indent=1,
        )
    )
    if not args.no_upload:
        _upload_dir(args.work / "judge_raw", "judge_raw")
    _log(f"[describe] ALL GROUPS done: {len(merged)} descriptions -> descriptions.jsonl")
    return 0


def stage_axes_grouped(args, packets: dict[int, dict]) -> int:
    """Axes over feature groups (features x 5 axes x 5 draws per group), one
    dispatch + shard + sentinel each; kappa is computed ONCE over the merged
    per-feature vote sets so the varying-n Fleiss statistic is not per-group."""
    out_dir = args.out_root / "labels"
    grp_dir = out_dir / "axes_groups"
    grp_dir.mkdir(parents=True, exist_ok=True)
    desc_path = out_dir / "descriptions.jsonl"
    if not desc_path.exists():
        raise RuntimeError(
            f"[axes] no descriptions at {desc_path}; run --stage describe first "
            "(a grouped --full axes dispatch with no DESC blocks is a changed instrument)"
        )
    descriptions = {int(r["feat_id"]): r["description"] for r in CM.iter_jsonl(desc_path)}
    packets = exclude_zero_evidence(packets, args.out_root)
    real = sorted(f for f in packets if f >= 0)
    groups = feature_groups(real, args.axes_group_size)
    _log(
        f"[axes] {len(real)} features -> {len(groups)} groups x {len(CM.AXES)} axes x {CM.N_DRAWS}"
    )
    tally: dict[str, dict[str, int]] = {
        a: {"launched": 0, "ok": 0, "content_drops": 0, "transport_losses": 0} for a in CM.AXES
    }
    for gi, gfeats in enumerate(groups):
        shard = grp_dir / f"axis_labels.g{gi:04d}.jsonl"
        done = grp_dir / f"axis_labels.g{gi:04d}.done.json"
        if _group_done(done) and not args.regroup:
            d = json.loads(done.read_text())
            for a, t in (d.get("tally") or {}).items():
                for k in tally.get(a, {}):
                    tally[a][k] += int(t.get(k, 0))
            _log(f"[axes] group {gi + 1}/{len(groups)} SKIP (done)")
            continue
        sub = {f: packets[f] for f in gfeats}
        items = build_axes_items(sub, descriptions)
        _log(f"[axes] group {gi + 1}/{len(groups)}: dispatching {len(items)} items")
        results = _dispatch(
            items,
            system=CM.AXIS_SYSTEM_PREAMBLE,
            max_tokens=CM.AXES_MAX_TOKENS,
            checkpoint_dir=args.work / "judge_checkpoints" / "axes" / f"g{gi:04d}",
            force_batch=args.force_batch,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        rows, gkappa = aggregate_axes(items, results)
        _write_jsonl(rows, shard)
        _write_raw(results, args.work / "judge_raw" / f"axes_raw_g{gi:04d}")
        gt = {a: gkappa[a]["drop_report"] for a in CM.AXES}
        _write_group_done(done, {"group": gi, "n_items": len(items), "tally": gt})
        for a in CM.AXES:
            for k in tally[a]:
                tally[a][k] += int(gt[a].get(k, 0))
        _log(f"[axes] group {gi + 1}/{len(groups)} done: {len(rows)} (feat x axis) rows")
    if args.dry_run:
        return 0

    merged_rows: list[dict] = []
    for p in sorted(grp_dir.glob("axis_labels.g*.jsonl")):
        merged_rows.extend(CM.iter_jsonl(p))
    _write_jsonl(merged_rows, out_dir / "axis_labels.jsonl")
    votes: dict[str, list[list[str]]] = {a: [] for a in CM.AXES}
    for r in merged_rows:
        votes[r["axis"]].append(list(r.get("labels_surviving") or []))
    kappa = {
        a: {**CM.fleiss_kappa_varying_n(votes[a], CM.AXES[a]), "drop_report": tally[a]}
        for a in CM.AXES
    }
    (out_dir / "kappa_report.json").write_text(
        json.dumps(
            {
                **CM.repro_meta(),
                "grouped": True,
                "n_groups": len(groups),
                "group_size_features": args.axes_group_size,
                "max_tokens": CM.AXES_MAX_TOKENS,
                "axes": kappa,
            },
            indent=1,
        )
    )
    if not args.no_upload:
        _upload_dir(args.work / "judge_raw", "judge_raw")
    _log("[axes] ALL GROUPS done: " + " ".join(f"{a}:k={kappa[a]['kappa']:.3f}" for a in CM.AXES))
    return 0


def stage_describe(args, packets: dict[int, dict]) -> int:
    items = build_describe_items(packets)
    if args.render_only:
        rd = args.out_root / "labels" / "golden_prompts"
        rd.mkdir(parents=True, exist_ok=True)
        for cid, _q, _c, user in items[: args.limit]:
            (rd / f"{cid}.txt").write_text(f"SYSTEM:\n{CM.DESCRIBER_SYSTEM}\n\nUSER:\n{user}")
        _log(f"[describe] render-only: {min(len(items), args.limit)} prompts -> {rd}")
        return 0
    if not args.full:
        items = items[: args.limit]
    _log(f"[describe] dispatching {len(items)} items (full={args.full})")
    results = _dispatch(
        items,
        system=CM.DESCRIBER_SYSTEM,
        max_tokens=CM.DESCRIBE_MAX_TOKENS,
        checkpoint_dir=args.work / "judge_checkpoints" / "describe",
        force_batch=args.force_batch,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0
    out_dir = args.out_root / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, drops = [], {"content": 0, "transport": 0}
    for cid, _q, _c, user in items:
        res = results.get(cid)
        if isinstance(res, dict) and res.get("error"):
            drops[_classify_error(res)] += 1
            continue
        parsed = parse_describe_result(res)
        if parsed is None:
            drops["content"] += 1
            continue
        feat_id = int(cid[1:].rsplit("-", 1)[0])
        rows.append({"feat_id": feat_id, **parsed, "prompt_sha16": CM.sha16(user)})
    path = out_dir / "descriptions.jsonl"
    tmp = path.parent / f".tmp_{path.name}"
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    meta = {
        **CM.repro_meta(),
        "n_items": len(items),
        "n_ok": len(rows),
        "drops": drops,
        "max_tokens": CM.DESCRIBE_MAX_TOKENS,
        "rubric_sha16": CM.sha16(CM.DESCRIBER_SYSTEM),
    }
    (out_dir / "describe_meta.json").write_text(json.dumps(meta, indent=1))
    _write_raw(results, args.work / "judge_raw" / "describe_raw")
    if not args.no_upload:
        _upload_dir(args.work / "judge_raw", "judge_raw")
    _log(f"[describe] done: {len(rows)}/{len(items)} ok, drops={drops}")
    return 0


# ── axes ─────────────────────────────────────────────────────────────────────


def build_axes_items(
    packets: dict[int, dict], descriptions: dict[int, str], axes=None, draws=CM.N_DRAWS
) -> list[tuple[str, str, str, str]]:
    """(feature x axis x draw) JudgeItems; real features only (feat_id >= 0)."""
    items = []
    for feat_id, pk in sorted(packets.items()):
        if feat_id < 0:
            continue  # random-direction controls: describe + detection only
        desc = descriptions.get(feat_id)
        for axis in axes or CM.AXES:
            for d in range(draws):
                user = CM.build_axis_user_msg(axis, pk, desc, d)
                items.append(
                    (CM.axis_custom_id(feat_id, axis, d), f"feat:{feat_id}:{axis}", "", user)
                )
    return items


def aggregate_axes(items, results) -> tuple[list[dict], dict]:
    """Majority vote + drop tally per (feat, axis); varying-n Fleiss kappa,
    prevalence + raw agreement per axis (reported NEXT TO kappa)."""
    votes: dict[tuple[int, str], list[str]] = defaultdict(list)
    tally: dict[str, dict[str, int]] = {
        a: {"launched": 0, "ok": 0, "content_drops": 0, "transport_losses": 0} for a in CM.AXES
    }
    for cid, _q, _c, _u in items:
        feat_id, axis, _d = CM.parse_axis_custom_id(cid)
        t = tally[axis]
        t["launched"] += 1
        res = results.get(cid)
        if isinstance(res, dict) and res.get("error"):
            kind = _classify_error(res)
            t["content_drops" if kind == "content" else "transport_losses"] += 1
            continue
        lab = CM.validate_axis_label(res, axis)
        if lab is None:
            t["content_drops"] += 1
            continue
        t["ok"] += 1
        votes[(feat_id, axis)].append(lab)
    rows = []
    feats = sorted({f for f, _a in votes})
    for feat_id in feats:
        for axis in CM.AXES:
            labs = votes.get((feat_id, axis), [])
            rows.append(
                {
                    "feat_id": feat_id,
                    "axis": axis,
                    "label": CM.majority_vote(labs),
                    "labels_surviving": labs,
                    "n_surviving": len(labs),
                    "n_launched": CM.N_DRAWS,
                }
            )
    kappa = {}
    for axis in CM.AXES:
        per_feat = [votes.get((f, axis), []) for f in feats]
        kappa[axis] = {
            **CM.fleiss_kappa_varying_n(per_feat, CM.AXES[axis]),
            "drop_report": tally[axis],
        }
    return rows, kappa


def stage_axes(args, packets: dict[int, dict]) -> int:
    desc_path = args.out_root / "labels" / "descriptions.jsonl"
    descriptions = {}
    if desc_path.exists():
        descriptions = {int(r["feat_id"]): r["description"] for r in CM.iter_jsonl(desc_path)}
    else:
        _log(f"[axes] WARNING: no descriptions at {desc_path}; DESC blocks omitted")
    items = build_axes_items(packets, descriptions)
    if args.render_only:
        rd = args.out_root / "labels" / "golden_prompts"
        rd.mkdir(parents=True, exist_ok=True)
        for cid, _q, _c, user in items[: args.limit]:
            (rd / f"{cid}.txt").write_text(f"SYSTEM:\n{CM.AXIS_SYSTEM_PREAMBLE}\n\nUSER:\n{user}")
        _log(f"[axes] render-only: {min(len(items), args.limit)} prompts -> {rd}")
        return 0
    if args.full and not descriptions:
        # r1 review Minor: a --full dispatch (409,600 calls) with no
        # descriptions would silently omit every DESC block — a changed
        # instrument at full spend. Bounded --limit slices keep the lenient
        # WARN branch (pilot slice re-runs load descriptions via stage_pilot).
        raise RuntimeError(
            f"[axes] --full dispatch with no descriptions at {desc_path}; "
            "run --stage describe first, or use a bounded --limit slice"
        )
    if not args.full:
        items = items[: args.limit]
    _log(f"[axes] dispatching {len(items)} items (full={args.full})")
    results = _dispatch(
        items,
        system=CM.AXIS_SYSTEM_PREAMBLE,
        max_tokens=CM.AXES_MAX_TOKENS,
        checkpoint_dir=args.work / "judge_checkpoints" / "axes",
        force_batch=args.force_batch,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0
    rows, kappa = aggregate_axes(items, results)
    out_dir = args.out_root / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "axis_labels.jsonl"
    tmp = path.parent / f".tmp_{path.name}"
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    (out_dir / "kappa_report.json").write_text(
        json.dumps({**CM.repro_meta(), "max_tokens": CM.AXES_MAX_TOKENS, "axes": kappa}, indent=1)
    )
    _write_raw(results, args.work / "judge_raw" / "axes_raw")
    if not args.no_upload:
        _upload_dir(args.work / "judge_raw", "judge_raw")
    _log("[axes] done: " + " ".join(f"{a}:k={kappa[a]['kappa']:.3f}" for a in CM.AXES))
    return 0


# ── pilot (plan §7 gate 2) ──────────────────────────────────────────────────


def pilot_slice(n: int = PILOT_N) -> np.ndarray:
    """Seeded activity-decile-stratified feature slice for the rubric pilot."""
    com = np.load(CM.PERFEATURE_NPZ, allow_pickle=False)
    fid = np.asarray(com["feat_ids"], dtype=np.int64)
    act = np.asarray(com["activity"], dtype=np.float64)
    edges = np.quantile(act, np.linspace(0, 1, 11)[1:-1])
    dec = np.searchsorted(edges, act, side="right")
    rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 500]))
    per = max(1, n // 10)
    picks = []
    for d in range(10):
        pool = np.where(dec == d)[0]
        picks.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    return fid[np.concatenate(picks)]


def stage_pilot(args, packets: dict[int, dict]) -> int:
    """Phases 2+3 on the seeded stratified slice; gate: PROCEED iff >=3 of 5
    axes read kappa >= 0.2 (below the 0.6 lattice bar by design — the pilot
    gates SPEND, not trust). Verdict is an artifact (pilot_gate.json), rc=0."""
    slice_ids = set(int(f) for f in pilot_slice(args.pilot_n))
    sub = {f: p for f, p in packets.items() if f in slice_ids}
    _log(f"[pilot] slice: {len(sub)} features with evidence packets")
    args.full = True  # the slice IS the bound; stage guards not needed here
    stage_describe(args, sub)
    stage_axes(args, sub)
    kappa_doc = json.loads((args.out_root / "labels" / "kappa_report.json").read_text())
    per_axis = {a: kappa_doc["axes"][a]["kappa"] for a in CM.AXES}
    n_clear = sum(
        1 for v in per_axis.values() if isinstance(v, int | float) and v >= PILOT_KAPPA_FLOOR
    )
    verdict = "PROCEED" if n_clear >= PILOT_AXES_FLOOR else "REVISE_RUBRICS"
    doc = {
        **CM.repro_meta(),
        "kappa_per_axis": per_axis,
        "n_axes_clearing_0.2": n_clear,
        "floor": PILOT_KAPPA_FLOOR,
        "verdict": verdict,
        "n_features": len(sub),
    }
    (args.out_root / "labels" / "pilot_gate.json").write_text(json.dumps(doc, indent=1))
    _log(f"[pilot] gate: {verdict} (axes clearing 0.2: {n_clear}/5)")
    return 0


def _import_check() -> int:
    """Axis-1 import-resolution leg (preferred shape (a))."""
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.eval.batch_judge import is_transport_error_dict  # noqa: F401
    from explore_persona_space.eval.judge_dispatch import (  # noqa: F401
        dispatch_judge_items,
        graded_temperature,
        keep_raw_judge_text,
    )
    from explore_persona_space.orchestrate import hub  # noqa: F401

    print("[import-check] OK: all deferred imports resolve", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("describe", "axes", "pilot"))
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--evidence-dir", type=Path, default=CM.WORK_DEFAULT / "evidence")
    ap.add_argument("--out-root", type=Path, default=CM.OUT_EVAL)
    ap.add_argument("--work", type=Path, default=CM.WORK_DEFAULT)
    ap.add_argument("--limit", type=int, default=DEFAULT_SMOKE_LIMIT)
    ap.add_argument("--full", action="store_true", help="production dispatch (no item cap)")
    ap.add_argument("--pilot-n", type=int, default=PILOT_N)
    ap.add_argument(
        "--grouped",
        action="store_true",
        help=(
            "shard the dispatch by feature group (REQUIRED at full-dictionary scale — "
            "one 3.28M-item axes dispatch does not fit in one items.json)"
        ),
    )
    ap.add_argument("--describe-group-size", type=int, default=DESCRIBE_GROUP_FEATURES)
    ap.add_argument("--axes-group-size", type=int, default=AXES_GROUP_FEATURES)
    ap.add_argument(
        "--regroup",
        action="store_true",
        help="ignore per-group done-sentinels and re-dispatch every group",
    )
    ap.add_argument("--render-only", action="store_true", help="golden prompts, zero API calls")
    ap.add_argument("--force-batch", action="store_true", help="Batch path at any N")
    ap.add_argument("--dry-run", action="store_true", help="routing decision only, zero calls")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        sys.exit(_import_check())
    packets = load_packets(args.evidence_dir, include_controls=(args.stage == "describe"))
    if args.grouped and args.render_only:
        ap.error("--grouped and --render-only are mutually exclusive")
    if args.stage == "describe":
        rc = (
            stage_describe_grouped(args, packets) if args.grouped else stage_describe(args, packets)
        )
    elif args.stage == "axes":
        rc = stage_axes_grouped(args, packets) if args.grouped else stage_axes(args, packets)
    elif args.stage == "pilot":
        rc = stage_pilot(args, packets)
    else:
        ap.error("--stage required (or --import-check)")
        return 2
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
