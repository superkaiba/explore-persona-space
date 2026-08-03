"""Off-GPU judge + DV leg of the issue #1739 wildchat rung (path C).

Runs AFTER the GPU leg (``scripts/issue1739_wcrung_pod.py``) has released its
GPU. The wildchat rung's contexts are behavior-INDEPENDENT, so the GPU leg
generated ONE rollout pool under the pseudo-behavior ``wildchat``; this leg
judges that SAME pool under EACH behavior's trait rubric and writes one DV
dataset per behavior. Generating three times would have tripled the GPU bill
for byte-identical rollouts; judging three times is API-bound and costs no GPU.

Per behavior:

    1. graded 0-100 trait-rubric judging, N draws at temperature > 0 through
       the sanctioned Batch client (``judging.judge_items_graded``)
    2. truncation-recovery re-judge of every fully-dropped item at a LARGER
       response budget against a FRESH cache (llm-judging.md rules 23/24 — the
       rubric-keyed cache deliberately excludes max_tokens, so a stale
       truncated entry would be re-served)
    3. per-CONTEXT DV rows (mean over rollouts with a kept score; a context
       whose every rollout dropped carries ``dv: None`` and is dropped
       downstream, never zero-filled) + first-class spread stats
       (n / mean / SD / min / max / decile histogram)

``--live-judge-probe`` is the pre-wave gate: 5 real items forced through the
BATCH path via ``threshold_base=1``, through THIS leg's own request builder.
A mock/offline judge smoke cannot catch a malformed Batch request shape — the
whole submit quarantines on one 400 (gotchas.md § the Anthropic Batch-API
request-shape entry), so the probe runs before any full wave.

CONTENT HYGIENE: logs and artifacts carry ids, counts, and scores — never
WildChat prompt text or rollout text.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_wcrung_score.py"
    assert sentinel.exists(), f"repo-root derivation failed: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue1739_wcrung_score")

BEHAVIORS = ("evil", "sycophancy", "hallucination")
RUNG = "wildchat_rung"
SPLIT = "eval"
GEN_BEHAVIOR = "wildchat"
HF_PREFIX = "issue1739_ctxmap/wildchat_rung"
CAPTURE_SENTINEL = "wcrung_capture_done.json"
SENTINEL_NAME = "wcrung_score_done.json"
HIST_EDGES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1739/wildchat_rung"))
    ap.add_argument(
        "--local-rollout-root",
        type=Path,
        default=None,
        help="local GPU-leg out-root; absent -> unpack the packed shards from HF",
    )
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_1739/wcrung_stage"),
        help="mirror root for HF-staged rollout shards",
    )
    # This leg scores #1739's wildchat rung by construction (it consumes that
    # rung's rollout pool and writes that rung's DV tree), so the default names
    # the rung's own subtree; --hf-prefix overrides.
    # UPLOAD_PREFIX_EXEMPT: wildchat-rung-specific scoring leg; flag overrides
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--rejudge-max-tokens", type=int, default=800)
    ap.add_argument("--no-rejudge", action="store_true")
    ap.add_argument("--dry-run-judge", action="store_true", help="SMOKE: no judge API calls")
    ap.add_argument("--skip-upload", action="store_true", help="SMOKE: no Hub writes")
    ap.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="SMOKE ONLY: cap the judged item count",
    )
    ap.add_argument(
        "--live-judge-probe",
        action="store_true",
        help="STANDALONE pre-launch gate: force N real items through the BATCH path, "
        "then exit. Sources items from the sampled CONTEXTS (no rollout pool "
        "needed) unless --local-rollout-root names one.",
    )
    ap.add_argument("--probe-items", type=int, default=5)
    ap.add_argument(
        "--probe-contexts-json",
        type=Path,
        default=None,
        help="probe item source (default: <out-root>/contexts/wcrung.json; "
        "absent -> stage the context shards from HF)",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def rollout_dir(args: argparse.Namespace) -> Path:
    """The SINGLE shared rollout pool — local if given, else unpacked from HF.

    The GPU leg packs its ~10k per-(context, k) JSONs into <= 9 MB jsonl
    shards before upload (the Hub's 10k-files-per-directory commit cap), so the
    HF path unpacks them back into the per-file layout this loader expects.
    """
    if args.local_rollout_root is not None:
        return args.local_rollout_root / "labeling" / GEN_BEHAVIOR

    from explore_persona_space.orchestrate import hub
    from scripts.issue1739_pack import unpack_shards

    # STAGE the packed shards from the Hub first — unpack_shards reads a LOCAL
    # shards dir and does no fetching of its own. stage_hub_prefix mirrors the
    # repo-relative tree under the mirror root, so the shards land at
    # <mirror_root>/<prefix> (#1774 mirror-root semantics).
    prefix = f"{args.hf_prefix}/raw_completions_packed"
    mirror_root = args.stage_root / "_packmirror"
    packed_dir = mirror_root / prefix
    if not (packed_dir / "pack_manifest.json").is_file():
        hub.stage_hub_prefix(
            hub.DEFAULT_DATASET_REPO,
            prefix,
            mirror_root,
            repo_type="dataset",
        )
    if not (packed_dir / "pack_manifest.json").is_file():
        raise RuntimeError(
            f"packed rollout staging incomplete: no pack_manifest.json under {packed_dir}"
        )

    # unpack_shards(shards_dir, out_root) restores out_root/<group>/<file>, and
    # the GPU leg packs raw_root=<out>/labeling so the group IS GEN_BEHAVIOR —
    # the restored pool is <out_root>/<GEN_BEHAVIOR>, NOT <out_root>/labeling/...
    out_root = args.stage_root / "unpacked"
    summary = unpack_shards(packed_dir, out_root)
    restored = sum(g["written"] + g["skipped"] for g in summary.values())
    print(
        f"[phase=wcrung_unpack] restored {restored} rollout files "
        f"({sorted(summary)}) -> {out_root}",
        flush=True,
    )
    return out_root / GEN_BEHAVIOR


def load_rollouts(rollout_dir_: Path, *, max_items: int | None = None) -> list[dict]:
    """Load the rollout payloads, fail-loud on a missing required field."""
    paths = sorted(p for p in rollout_dir_.glob("*.json") if not p.name.startswith("_"))
    if not paths:
        raise RuntimeError(f"no rollout files under {rollout_dir_}")
    if max_items is not None:
        paths = paths[:max_items]
    rollouts = []
    for p in paths:
        payload = json.loads(p.read_text())
        for key in ("context_id", "rollout_k", "query", "completion", "group_key"):
            if payload.get(key) is None:
                raise RuntimeError(f"{p.name}: rollout payload missing {key!r}")
        rollouts.append(payload)
    return rollouts


def _spread_stats(values: list[float | None]) -> dict:
    """n / mean / SD / min / max / decile histogram over the kept per-context DVs."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "histogram": {}}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    hist: dict[str, int] = {}
    for lo, hi in itertools.pairwise(HIST_EDGES):
        hist[f"{lo}-{hi}"] = sum(1 for v in vals if (lo <= v < hi) or (hi == 100 and v == 100))
    return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals), "histogram": hist}


def _judge_items(rollouts: list[dict]) -> list[tuple[str, str, str]]:
    from explore_persona_space.experiments.issue_1739 import judging

    return [
        (
            judging.rollout_item_id(r["context_id"], int(r["rollout_k"])),
            r["query"],
            r["completion"],
        )
        for r in rollouts
    ]


def judge_behavior(behavior: str, rollouts: list[dict], args: argparse.Namespace) -> dict:
    """Graded trait-rubric judging + truncation-recovery re-judge."""
    from explore_persona_space.experiments.issue_1739 import judging
    from explore_persona_space.experiments.issue_1739.constants import JUDGE_MAX_TOKENS

    rubric = judging.load_trait_rubric(behavior, inputs_dir=args.out_root / "inputs")
    items = _judge_items(rollouts)
    judge_root = args.out_root / "judge" / behavior
    t0 = time.time()
    result = judging.judge_items_graded(
        items,
        rubric,
        cache_dir=judge_root / "cache",
        save_raw=judge_root / "judge_raw.json",
        max_tokens=JUDGE_MAX_TOKENS,
        dry_run=args.dry_run_judge,
    )
    tallies = judging.judge_tallies(result)
    scores: dict[str, float | None] = dict(tallies["scores"])
    print(
        f"[phase=wcrung_judge behavior={behavior}] items={len(items)} "
        f"draws={tallies['n_total_draws']} content_drops={tallies['n_content_dropped_draws']} "
        f"transport_losses={tallies['n_transport_lost_draws']} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )

    rejudged: list[str] = []
    rejudge_tallies = None
    fully_dropped = [i for i, s in scores.items() if s is None]
    if fully_dropped and not args.no_rejudge and not args.dry_run_judge:
        # Truncation recovery at a larger budget against a FRESH cache: the
        # rubric-keyed cache excludes max_tokens, so the stale truncated entry
        # would otherwise be re-served (llm-judging.md rule 23).
        want = set(fully_dropped)
        sub = [it for it in items if it[0] in want]
        t1 = time.time()
        result2 = judging.judge_items_graded(
            sub,
            rubric,
            cache_dir=judge_root / "cache_rejudge",
            save_raw=judge_root / "judge_raw_rejudge.json",
            max_tokens=args.rejudge_max_tokens,
        )
        rejudge_tallies = judging.judge_tallies(result2)
        for item_id, s in rejudge_tallies["scores"].items():
            if s is not None:
                scores[item_id] = s
                rejudged.append(item_id)
        print(
            f"[phase=wcrung_rejudge behavior={behavior}] items={len(sub)} "
            f"recovered={len(rejudged)} max_tokens={args.rejudge_max_tokens} "
            f"elapsed={time.time() - t1:.0f}s",
            flush=True,
        )

    return {
        "rubric_source": "e1_assets.eval_prompt",
        "scores": scores,
        "tallies": tallies,
        "rejudge": {
            "n_items": len(fully_dropped),
            "n_recovered": len(rejudged),
            "recovered_item_ids": sorted(rejudged),
            "max_tokens": args.rejudge_max_tokens if rejudged else None,
            "tallies": rejudge_tallies,
            "mixed_instrument": bool(rejudged),
        },
    }


def build_dv_rows(
    behavior: str,
    rollouts: list[dict],
    scores: dict,
    *,
    per_item_transport_losses: dict[str, int] | None = None,
) -> tuple[list[dict], dict]:
    """Per-CONTEXT DV rows in the ladder's `_load_labeled` schema + a coverage digest.

    The DV REDUCTION itself is the shared library's
    ``dv_build.build_labeling_dv`` — mean over rollouts with a kept score,
    drop-never-coerce, transport losses summed per context and kept SEPARATE
    from content drops. This leg only supplies the per-context metadata the
    ladder needs (behavior / split / rung / group_key) and the coverage digest;
    it does not re-implement the reduction.
    """
    from explore_persona_space.experiments.issue_1739 import dv_build
    from explore_persona_space.experiments.issue_1739.constants import (
        K_ROLLOUTS,
        N_JUDGE_DRAWS,
    )

    contexts_meta = {
        r["context_id"]: {
            "behavior": behavior,
            "split": SPLIT,
            "rung": RUNG,
            "group_key": r["group_key"],
        }
        for r in rollouts
    }
    k_seen = max((int(r["rollout_k"]) for r in rollouts), default=0) + 1
    rows = dv_build.build_labeling_dv(
        scores,
        k_rollouts=max(K_ROLLOUTS, k_seen),
        n_draws=N_JUDGE_DRAWS,
        per_item_transport_losses=per_item_transport_losses,
        contexts_meta=contexts_meta,
    )
    n_no_dv = sum(1 for r in rows if r.get("dv") is None)
    digest = {
        "n_contexts": len(rows),
        "n_contexts_with_dv": len(rows) - n_no_dv,
        "n_contexts_dropped_no_dv": n_no_dv,
        "n_groups": len({r["group_key"] for r in rows}),
    }
    return rows, digest


def _run_meta(args: argparse.Namespace) -> dict:
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MAX_TOKENS,
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
        N_JUDGE_DRAWS,
    )

    return {
        "judge_model": JUDGE_MODEL,
        "n_judge_draws": N_JUDGE_DRAWS,
        "judge_temperature": JUDGE_TEMPERATURE,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "rejudge_max_tokens": args.rejudge_max_tokens,
        "gen_behavior": GEN_BEHAVIOR,
        "gen_behavior_note": (
            "one rollout pool judged under all three trait rubrics — the wildchat "
            "contexts carry no behavior conditioning, so per-behavior generation "
            "would produce byte-identical rollouts"
        ),
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def probe_items_from_contexts(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Probe items built from the CONTEXT rows — no rollout pool required.

    This is what makes the probe runnable PRE-LAUNCH, standalone: before the
    GPU leg exists there are no rollouts, but the sampled contexts already
    carry the real queries. The completion side is a short placeholder — the
    probe tests the Batch REQUEST SHAPE, not score quality — so it can gate a
    launch without burning any of the real pool.
    """
    from scripts import issue1739_wcrung_rows_io as rows_io

    local = args.probe_contexts_json or (args.out_root / "contexts" / "wcrung.json")
    if local.exists():
        rows = json.loads(local.read_text())["rows"]
        src = str(local)
    else:
        rows = rows_io.stage_rows_from_hub(
            hf_prefix=f"{args.hf_prefix}/contexts",
            dest_dir=args.stage_root,
        )
        src = f"{args.hf_prefix}/contexts"
    if not rows:
        raise RuntimeError(f"no context rows for the probe (source: {src})")
    print(f"[phase=wcrung_probe_items] source={src} contexts={len(rows)}", flush=True)

    from explore_persona_space.experiments.issue_1739 import judging

    return [
        (
            judging.rollout_item_id(r["context_id"], 0),
            r["query"],
            "Placeholder response for the request-shape probe.",
        )
        for r in rows[: args.probe_items]
    ]


def live_judge_probe(args: argparse.Namespace, rollouts: list[dict] | None = None) -> dict:
    """Force N real items through the BATCH path via this leg's own builder.

    ``threshold_base=1`` makes the dispatcher's effective threshold 1, so even
    a 5-item set routes to the Message Batches API instead of the sync path.
    A malformed Batch request quarantines the WHOLE submit on one 400, and no
    mock/offline smoke can see it — hence the live gate before the full wave.

    Items come from the real rollout pool when one is available, else from the
    sampled CONTEXT rows (``probe_items_from_contexts``) so the gate runs
    standalone before the GPU leg has produced anything.
    """
    from explore_persona_space.experiments.issue_1739 import judging
    from explore_persona_space.experiments.issue_1739.constants import JUDGE_MAX_TOKENS

    behavior = args.behaviors[0]
    rubric = judging.load_trait_rubric(behavior, inputs_dir=args.out_root / "inputs")
    if rollouts:
        items = _judge_items(rollouts)[: args.probe_items]
        item_source = "rollout_pool"
    else:
        items = probe_items_from_contexts(args)
        item_source = "contexts"
    probe_root = args.out_root / "judge" / "_live_probe"
    t0 = time.time()
    result = judging.judge_items_graded(
        items,
        rubric,
        cache_dir=probe_root / "cache",
        save_raw=probe_root / "judge_raw.json",
        max_tokens=JUDGE_MAX_TOKENS,
        threshold_base=1,
    )
    tallies = judging.judge_tallies(result)
    scored = {i: s for i, s in tallies["scores"].items() if s is not None}
    payload = {
        "behavior": behavior,
        "item_source": item_source,
        "n_items": len(items),
        "n_scored": len(scored),
        "n_content_dropped_draws": tallies["n_content_dropped_draws"],
        "n_transport_lost_draws": tallies["n_transport_lost_draws"],
        "n_total_draws": tallies["n_total_draws"],
        "forced_batch": True,
        "threshold_base": 1,
        "elapsed_s": round(time.time() - t0, 1),
        "meta": _run_meta(args),
    }
    _write_json_atomic(args.out_root / "judge" / "live_probe.json", payload)
    print(
        f"[phase=wcrung_live_probe behavior={behavior}] items={len(items)} "
        f"scored={len(scored)} draws={tallies['n_total_draws']} "
        f"content_drops={tallies['n_content_dropped_draws']} "
        f"transport_losses={tallies['n_transport_lost_draws']} "
        f"elapsed={payload['elapsed_s']}s",
        flush=True,
    )
    if not scored:
        raise RuntimeError(
            "live judge probe scored 0 of "
            f"{len(items)} items — the Batch request shape is broken; fix the "
            "builder before the full wave (a 400 quarantines the whole submit)"
        )
    return payload


def run_behavior(behavior: str, rollouts: list[dict], args: argparse.Namespace) -> dict:
    judged = judge_behavior(behavior, rollouts, args)
    rows, digest = build_dv_rows(
        behavior,
        rollouts,
        judged["scores"],
        per_item_transport_losses=judged["tallies"].get("per_item_transport_losses"),
    )
    spread = _spread_stats([r["dv"] for r in rows])
    payload = {
        "behavior": behavior,
        "rung": RUNG,
        "split": SPLIT,
        "dv_construct": "trait_rubric_graded_0_100",
        "dv_construct_note": (
            "graded 0-100 trait eval_prompt rubric over on-policy rollouts on "
            "random held-out WildChat contexts (conversation-disjoint); for "
            "hallucination this DIFFERS from the train-side three-way "
            "alias-match/fabrication DV (generic chat carries no reference "
            "answers) — do not pool the two constructs"
        ),
        "rows": rows,
        "coverage": digest,
        "spread": spread,
        "judge": {
            "rubric_source": judged["rubric_source"],
            "tallies": judged["tallies"],
            "rejudge": judged["rejudge"],
        },
        "meta": _run_meta(args),
    }
    _write_json_atomic(args.out_root / "dv_dataset" / behavior / "labeling.json", payload)
    _write_json_atomic(
        args.out_root / "spread" / f"{behavior}.json",
        {
            "behavior": behavior,
            "rung": RUNG,
            "dv_construct": payload["dv_construct"],
            "spread": spread,
            "coverage": digest,
            "meta": payload["meta"],
        },
    )
    print(
        f"[phase=wcrung_dv behavior={behavior}] contexts={digest['n_contexts']} "
        f"with_dv={digest['n_contexts_with_dv']} groups={digest['n_groups']} "
        f"mean={spread['mean']} sd={spread['sd']}",
        flush=True,
    )
    return {"behavior": behavior, "coverage": digest, "spread": spread, "judge": judged}


def _upload_dir(local: Path, path_in_repo: str, *, skip: bool) -> str:
    if skip:
        logger.info("[upload] SKIP (--skip-upload) %s -> %s", local, path_in_repo)
        return ""
    from explore_persona_space.orchestrate import hub

    return hub._upload(
        local, hub.DEFAULT_DATASET_REPO, "dataset", path_in_repo, raise_on_error=True
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.import_check:
        from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
            constants,
            judging,
        )
        from explore_persona_space.experiments.issue_1739.constants import (  # noqa: F401
            JUDGE_MAX_TOKENS,
            JUDGE_MODEL,
            JUDGE_TEMPERATURE,
            N_JUDGE_DRAWS,
        )
        from explore_persona_space.experiments.issue_1739.judging import (  # noqa: F401
            judge_items_graded,
            judge_tallies,
            load_trait_rubric,
            rollout_item_id,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            DEFAULT_DATASET_REPO,
            _upload,
        )
        from scripts.issue1739_pack import unpack_shards  # noqa: F401

        print("[import-check] OK: all deferred imports resolved", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.live_judge_probe:
        # Standalone PRE-LAUNCH gate: never loads (or needs) the rollout pool
        # unless a local one was explicitly named, so it can run before the GPU
        # leg exists and without burning any of the real pool.
        pool: list[dict] | None = None
        if args.local_rollout_root is not None:
            pool = load_rollouts(rollout_dir(args), max_items=args.probe_items)
        live_judge_probe(args, pool)
        print("[phase=done] wcrung live judge probe OK", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    rollouts = load_rollouts(rollout_dir(args), max_items=args.max_items)
    n_ctx = len({r["context_id"] for r in rollouts})
    print(
        f"[phase=wcrung_rollouts] rollouts={len(rollouts)} contexts={n_ctx} pool={GEN_BEHAVIOR}",
        flush=True,
    )

    per_behavior = [run_behavior(b, rollouts, args) for b in args.behaviors]

    for behavior in args.behaviors:
        _upload_dir(
            args.out_root / "dv_dataset" / behavior,
            f"{args.hf_prefix}/dv_dataset/{behavior}",
            skip=args.skip_upload,
        )
    _upload_dir(args.out_root / "spread", f"{args.hf_prefix}/spread", skip=args.skip_upload)

    sentinel = {
        "rung": RUNG,
        "split": SPLIT,
        "gen_behavior": GEN_BEHAVIOR,
        "behaviors": list(args.behaviors),
        "n_rollouts": len(rollouts),
        "n_contexts": n_ctx,
        "per_behavior": [
            {
                "behavior": r["behavior"],
                "coverage": r["coverage"],
                "spread_n": r["spread"]["n"],
                "spread_mean": r["spread"]["mean"],
                "spread_sd": r["spread"]["sd"],
                "n_content_dropped_draws": r["judge"]["tallies"]["n_content_dropped_draws"],
                "n_transport_lost_draws": r["judge"]["tallies"]["n_transport_lost_draws"],
                "n_rejudge_recovered": r["judge"]["rejudge"]["n_recovered"],
            }
            for r in per_behavior
        ],
        "meta": _run_meta(args),
    }
    sentinel_path = args.out_root / SENTINEL_NAME
    _write_json_atomic(sentinel_path, sentinel)
    print(f"[phase=done] wcrung scoring complete: {sentinel_path}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
