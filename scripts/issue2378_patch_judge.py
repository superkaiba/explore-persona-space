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
23/26): ``--pilot N`` runs N calls on the production instrument and FAILS on
any ``max_tokens`` stop or a parse-fail rate >= 2%; the production wave
refuses to run without a PASS report on disk (or --skip-pilot-gate, recorded).

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
        "temperature": 1.0,
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


def run_pilot(args, out_dir: Path) -> int:
    items = build_items(Path(args.patch_root), max_items=int(args.pilot))
    classified = _dispatch(items, args, args.transport, f"pilot-{args.transport}")
    n = len(classified)
    n_cap = sum(1 for c in classified.values() if c.get("stop_reason") == "max_tokens")
    n_parse_fail = sum(1 for c in classified.values() if c["class"] in ("parse_fail", "truncation"))
    ok = n_cap == 0 and (n_parse_fail / max(n, 1)) < PILOT_PARSE_FAIL_MAX
    report = {
        "n": n,
        "n_max_tokens_stops": n_cap,
        "n_parse_fail": n_parse_fail,
        "parse_fail_rate": n_parse_fail / max(n, 1),
        "transport": args.transport,
        "ok": ok,
        "metadata": cm.run_metadata({"phase": "patch_judge_pilot"}),
    }
    cm.atomic_write_json(_pilot_report_path(out_dir), report)
    _log(f"[pilot] n={n} max_tokens_stops={n_cap} parse_fail={n_parse_fail} ok={ok}")
    return 0 if ok else 7


# ── production wave ─────────────────────────────────────────────────────────


def run_wave(args, out_dir: Path) -> int:
    if not args.skip_pilot_gate:
        rp = _pilot_report_path(out_dir)
        if not rp.exists():
            raise SystemExit(f"pilot gate: no pilot report at {rp} — run --pilot first")
        rep = json.loads(rp.read_text(encoding="utf-8"))
        if not rep.get("ok"):
            raise SystemExit(f"pilot gate: pilot report at {rp} is FAIL — fix the instrument")
    items = build_items(Path(args.patch_root))
    t0 = time.time()
    classified = _dispatch(
        items, args, None if args.transport == "auto" else args.transport, "wave"
    )
    by_id = {it.item_id: it for it in items}
    scores_path = out_dir / "scores.jsonl"
    tally = {"kept": 0, "dropped": 0, "transport_loss": 0}
    writer = cm.ShardWriter(out_dir / "raw", "judge_rows")
    with scores_path.open("w", encoding="utf-8") as fh:
        for iid in sorted(classified):
            c = classified[iid]
            score = c.get("score")
            kept = c["class"] == "valid" and isinstance(score, int) and 0 <= score <= 100
            tally["kept" if kept else "dropped"] += 1
            if c["class"] == "transport_loss":
                tally["transport_loss"] += 1
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
    report = {
        "n_items": len(items),
        "tally": tally,
        "wall_s": time.time() - t0,
        "judge_model": cm.JUDGE_MODEL,
        "max_tokens": cm.JUDGE_MAX_TOKENS,
        "raw_shards": shard_info,
        "metadata": cm.run_metadata({"phase": "patch_judge_wave"}),
    }
    cm.atomic_write_json(out_dir / "wave_report.json", report)
    if not args.skip_upload:
        cm.upload_stage_dir(out_dir / "raw", f"{pc.HF_STAGE_PREFIX}{args.hf_suffix}/judge_persona")
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
