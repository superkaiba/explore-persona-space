"""Judge + DV leg of the issue #1739 "pvsynth" rung (API-bound; 0 GPU).

Runs OFF the GPU (the pod leg ``issue1739_pvsynth_pod.py`` releases it first —
CLAUDE.md "CPU-only phases don't hold GPU pods"): waits on the pod leg's HF
sentinel, judges the rollouts through the sanctioned Batch-API client, builds the
per-context DV in the SAME schema ``issue1739_fits._load_labeled`` consumes, and
publishes both.

DV recipe — matched to the main #1739 rungs (constants.py pins): graded 0-100
trait rubric (the per-trait ``eval_prompt`` E1 asset), judge
``claude-sonnet-4-5-20250929``, ``N_JUDGE_DRAWS`` draws at temperature 1.0,
``JUDGE_MAX_TOKENS`` reason-first budget, drop-never-coerce with the
content-drop / transport-loss split preserved (llm-judging.md rules 9/23/24).
Items whose draws all content-dropped are re-judged WHOLE at
``--rejudge-max-tokens`` against a FRESH cache (the main run's truncation-recovery
convention; the mixed-instrument use is disclosed in the output meta).

STATED DEVIATION — hallucination: the main run's hallucination DV is the
three-way alias-match + fabrication-vs-abstention read, which needs a reference
answer per question. The Persona Vectors eval questions carry NO reference
answers, so on THIS rung all three behaviors use the trait ``eval_prompt`` graded
rubric (which is also the paper's own eval for the trait). The output meta
records ``dv_construct`` per behavior so no downstream read can silently treat
the hallucination pvsynth DV as the train-side fabrication-rate construct.

The arm-scoring leg is deliberately NOT here: it must refit the frozen
predictors against the behavior's TRAIN capture store (32-70 GB per behavior on
HF), which exceeds the shared-VM analysis footprint ceiling — it routes to a
pod / big-disk CPU lane. See the module docstring of the scoring driver named in
this run's completion note.

CONTENT HYGIENE: logs and summaries carry ids, counts, scores — never rollout or
question text.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_pvsynth_score.py"
    assert sentinel.exists(), f"repo-root derivation failed: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue1739_pvsynth_score")

BEHAVIORS = ("evil", "sycophancy", "hallucination")
RUNG = "pvsynth"
SPLIT = "eval"
HF_PREFIX = "issue1739_ctxmap/pvsynth"
CAPTURE_SENTINEL = "pvsynth_capture_done.json"
DV_SENTINEL = "pvsynth_dv_ready.json"
HIST_EDGES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1739/pvsynth"))
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_1739/pvsynth_stage"),
        help="MIRROR ROOT for Hub staging (files land at <root>/<repo-relative path>)",
    )
    # This leg is bound to #1739's pvsynth rung by construction (it drains the pvsynth
    # capture sentinel and writes that rung's own DV / judge / spread artifacts), so the
    # default names that rung's own subtree, not a reusable destination.
    # UPLOAD_PREFIX_EXEMPT: pvsynth-rung-specific leg; an explicit --hf-prefix overrides
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument(
        "--local-rollout-root",
        type=Path,
        default=None,
        help="skip Hub staging and read rollouts from this out-root (smoke / same-box)",
    )
    ap.add_argument("--wait-timeout-s", type=int, default=21600, help="sentinel poll budget (6h)")
    ap.add_argument("--poll-interval-s", type=int, default=120)
    ap.add_argument("--no-wait", action="store_true", help="fail immediately if sentinel absent")
    ap.add_argument("--rejudge-max-tokens", type=int, default=800)
    ap.add_argument("--no-rejudge", action="store_true")
    ap.add_argument("--dry-run-judge", action="store_true", help="SMOKE: no judge API calls")
    ap.add_argument("--skip-upload", action="store_true", help="SMOKE: no Hub writes")
    ap.add_argument("--issue", type=int, default=1739)
    ap.add_argument("--no-post-marker", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _git_commit() -> str:
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


def wait_for_capture_sentinel(args: argparse.Namespace) -> dict:
    """Bounded poll for the pod leg's HF sentinel; returns its payload."""
    from explore_persona_space.orchestrate import hub

    path_in_repo = f"{args.hf_prefix}/{CAPTURE_SENTINEL}"
    target = args.stage_root / path_in_repo
    deadline = time.time() + args.wait_timeout_s
    attempt = 0
    while True:
        attempt += 1
        try:
            hub.stage_hub_file(hub.DEFAULT_DATASET_REPO, path_in_repo, target, overwrite=True)
            payload = json.loads(target.read_text())
            print(
                f"[phase=pvsynth_sentinel] found after {attempt} probe(s): "
                f"behaviors={payload.get('behaviors')}",
                flush=True,
            )
            return payload
        except Exception as exc:  # absent yet, or a transport failure past its budget
            if args.no_wait:
                raise RuntimeError(f"capture sentinel absent at {path_in_repo}") from exc
            if time.time() >= deadline:
                raise RuntimeError(
                    f"capture sentinel still absent after {args.wait_timeout_s}s "
                    f"({attempt} probes): {path_in_repo}"
                ) from exc
            print(
                f"[phase=pvsynth_sentinel] probe {attempt}: not ready "
                f"({type(exc).__name__}); sleeping {args.poll_interval_s}s",
                flush=True,
            )
            time.sleep(args.poll_interval_s)


def rollout_dir_for(args: argparse.Namespace, behavior: str) -> Path:
    """Local rollout dir, staging from the Hub when no local root was given.

    ``stage_hub_prefix``'s ``dest_dir`` is a MIRROR ROOT — files land at
    ``dest_dir/<repo-relative path>``, so the consumed dir is
    ``stage_root/<hf_prefix>/raw_completions/<behavior>`` (#1774: passing the
    consumed path as ``dest_dir`` nests the prefix under it). Asserted below.
    """
    if args.local_rollout_root is not None:
        return args.local_rollout_root / "labeling" / behavior
    from explore_persona_space.orchestrate import hub

    prefix = f"{args.hf_prefix}/raw_completions/{behavior}"
    consumed = args.stage_root / prefix
    staged = hub.stage_hub_prefix(hub.DEFAULT_DATASET_REPO, prefix, args.stage_root)
    assert staged, prefix
    for p in staged:
        assert consumed in p.parents or p.parent == consumed, (
            f"mirror-root arithmetic broken: {p} not under {consumed}"
        )
    print(
        f"[phase=pvsynth_stage behavior={behavior}] staged {len(staged)} rollout files", flush=True
    )
    return consumed


def load_rollouts(rollout_dir: Path) -> list[dict]:
    paths = sorted(p for p in rollout_dir.glob("*.json") if not p.name.startswith("_"))
    if not paths:
        raise RuntimeError(f"no rollout files under {rollout_dir}")
    rollouts = []
    for p in paths:
        payload = json.loads(p.read_text())
        for key in ("context_id", "rollout_k", "query", "completion", "group_key"):
            if payload.get(key) is None:
                raise RuntimeError(f"{p.name}: rollout payload missing {key!r}")
        rollouts.append(payload)
    return rollouts


def _spread_stats(values: list[float]) -> dict:
    """n / mean / SD / decile histogram over the kept per-context DVs."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "histogram": {}}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    hist: dict[str, int] = {}
    for lo, hi in itertools.pairwise(HIST_EDGES):
        label = f"{lo}-{hi}"
        hist[label] = sum(1 for v in vals if (lo <= v < hi) or (hi == 100 and v == 100))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": min(vals),
        "max": max(vals),
        "histogram": hist,
    }


def judge_behavior(behavior: str, rollouts: list[dict], args: argparse.Namespace) -> dict:
    """Graded trait-rubric judging + optional truncation-recovery re-judge."""
    from explore_persona_space.experiments.issue_1739 import judging
    from explore_persona_space.experiments.issue_1739.constants import JUDGE_MAX_TOKENS

    rubric = judging.load_trait_rubric(behavior, inputs_dir=args.out_root / "inputs")
    items = [
        (
            judging.rollout_item_id(r["context_id"], int(r["rollout_k"])),
            r["query"],
            r["completion"],
        )
        for r in rollouts
    ]
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
        f"[phase=pvsynth_judge behavior={behavior}] items={len(items)} "
        f"draws={tallies['n_total_draws']} content_drops={tallies['n_content_dropped_draws']} "
        f"transport_losses={tallies['n_transport_lost_draws']} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )

    rejudged_ids: list[str] = []
    rejudge_tallies = None
    fully_dropped = [i for i, s in scores.items() if s is None]
    if fully_dropped and not args.no_rejudge and not args.dry_run_judge:
        # Truncation-recovery: re-judge the WHOLE item at a larger response
        # budget against a FRESH cache (rule 23/24(ii) — the rubric-keyed cache
        # deliberately excludes max_tokens, so a stale entry would be re-served).
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
                rejudged_ids.append(item_id)
        print(
            f"[phase=pvsynth_rejudge behavior={behavior}] items={len(sub)} "
            f"recovered={len(rejudged_ids)} max_tokens={args.rejudge_max_tokens} "
            f"elapsed={time.time() - t1:.0f}s",
            flush=True,
        )

    return {
        "rubric_source": "e1_assets.eval_prompt",
        "scores": scores,
        "tallies": tallies,
        "rejudge": {
            "n_items": len(fully_dropped),
            "n_recovered": len(rejudged_ids),
            "recovered_item_ids": sorted(rejudged_ids),
            "max_tokens": args.rejudge_max_tokens if rejudged_ids else None,
            "tallies": rejudge_tallies,
            "mixed_instrument": bool(rejudged_ids),
        },
    }


def build_dv_rows(behavior: str, rollouts: list[dict], scores: dict) -> tuple[list[dict], dict]:
    """Per-CONTEXT DV rows in the ``_load_labeled`` schema + a coverage digest."""
    from explore_persona_space.experiments.issue_1739 import judging

    by_ctx: dict[str, dict] = {}
    for r in rollouts:
        cid = r["context_id"]
        entry = by_ctx.setdefault(
            cid,
            {
                "behavior": behavior,
                "context_id": cid,
                "split": SPLIT,
                "rung": RUNG,
                "group_key": r["group_key"],
                "per_rollout_scores": {},
            },
        )
        item_id = judging.rollout_item_id(cid, int(r["rollout_k"]))
        entry["per_rollout_scores"][f"k{int(r['rollout_k'])}"] = scores.get(item_id)

    rows: list[dict] = []
    n_no_dv = 0
    for cid in sorted(by_ctx):
        entry = by_ctx[cid]
        kept = [v for v in entry["per_rollout_scores"].values() if v is not None]
        entry["n_rollouts"] = len(entry["per_rollout_scores"])
        entry["n_rollouts_kept"] = len(kept)
        # Drop-never-coerce: a context with zero kept rollouts carries dv=None
        # and is dropped by _load_labeled, never zero-filled.
        entry["dv"] = (sum(kept) / len(kept)) if kept else None
        if entry["dv"] is None:
            n_no_dv += 1
        rows.append(entry)
    digest = {
        "n_contexts": len(rows),
        "n_contexts_with_dv": len(rows) - n_no_dv,
        "n_contexts_dropped_no_dv": n_no_dv,
        "n_groups": len({r["group_key"] for r in rows}),
    }
    return rows, digest


def run_behavior(behavior: str, args: argparse.Namespace) -> dict:
    rollout_dir = rollout_dir_for(args, behavior)
    rollouts = load_rollouts(rollout_dir)
    judged = judge_behavior(behavior, rollouts, args)
    rows, digest = build_dv_rows(behavior, rollouts, judged["scores"])
    spread = _spread_stats([r["dv"] for r in rows])
    dv_construct = "pv_trait_rubric_graded_0_100"
    payload = {
        "behavior": behavior,
        "rung": RUNG,
        "split": SPLIT,
        "dv_construct": dv_construct,
        "dv_construct_note": (
            "graded 0-100 trait eval_prompt rubric over on-policy rollouts; for "
            "hallucination this DIFFERS from the train-side three-way "
            "alias-match/fabrication DV (the PV eval questions carry no reference "
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
            "dv_construct": dv_construct,
            "spread": spread,
            "coverage": digest,
            "meta": _run_meta(args),
        },
    )
    print(
        f"[phase=pvsynth_dv behavior={behavior}] contexts={digest['n_contexts']} "
        f"with_dv={digest['n_contexts_with_dv']} groups={digest['n_groups']} "
        f"mean={spread['mean']} sd={spread['sd']}",
        flush=True,
    )
    return {"behavior": behavior, "coverage": digest, "spread": spread, "judge": judged}


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
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _main_checkout() -> Path | None:
    """The MAIN checkout dir (worktree-safe), or None if it cannot be resolved.

    `_REPO_ROOT` is THIS file's tree — a WORKTREE on the `issue-<N>` branch, where
    `task.py` refuses (it branch-guards to `main`). The main checkout is the parent
    of the shared git common dir.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    common = Path(proc.stdout.strip())
    root = common.parent
    return root if (root / "scripts" / "task.py").exists() else None


def _on_main(checkout: Path) -> bool:
    """True iff `checkout`'s HEAD is the `main` branch (task.py's precondition)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(checkout),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    return proc.stdout.strip() == "main"


def _post_progress_marker(args: argparse.Namespace, note_path: Path) -> bool:
    """Fail-soft `epm:progress` post — LOCAL-VM ONLY, gated on a `main` checkout.

    `task.py` branch-guards to `main`, so this posts ONLY from the VM's main
    checkout (resolved via the git common dir — never this file's worktree, whose
    HEAD is `issue-<N>`). The `main`-HEAD gate is what makes the call structurally
    unreachable on a pod: a pod clone runs the `issue-<N>` branch, so the gate
    short-circuits and the note is left for the orchestrator to post. That keeps
    the repo-wide pod-side-shellout invariant intact
    (tests/test_no_pod_side_task_py_shellout.py; CLAUDE.md § "Pod-side code NEVER
    shells out to scripts/task.py").
    """
    checkout = _main_checkout()
    if checkout is None or not _on_main(checkout):
        where = str(checkout) if checkout else "unresolved"
        print(
            f"[marker] SKIPPED task.py post — no `main` checkout available "
            f"({where}); this leg is local-VM-only by contract. Note preserved at "
            f"{note_path}\n"
            f"[marker] orchestrator: uv run python scripts/task.py post-marker "
            f"{args.issue} epm:progress --file {note_path}",
            flush=True,
        )
        return False
    cmd = [
        "uv",
        "run",
        "python",
        str(checkout / "scripts" / "task.py"),
        "post-marker",
        str(args.issue),
        "epm:progress",
        "--file",
        str(note_path),
    ]
    try:
        proc = subprocess.run(
            # epm-lint: pod-shellout-ok -- local-VM-only leg, gated above on the
            # resolved main checkout being on branch `main`; a pod clone runs
            # issue-<N>, so this call is structurally unreachable pod-side.
            cmd,
            cwd=str(checkout),
            capture_output=True,
            text=True,
            env={**os.environ},
        )
    except OSError as exc:
        print(f"[marker] FAILED to invoke task.py ({exc}); note at {note_path}", flush=True)
        return False
    if proc.returncode != 0:
        print(
            f"[marker] task.py rc={proc.returncode}; note preserved at {note_path}\n"
            f"{proc.stderr[-800:]}",
            flush=True,
        )
        return False
    print(f"[marker] posted epm:progress from {note_path}", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.import_check:
        from explore_persona_space.experiments.issue_1739 import judging  # noqa: F401
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
            stage_hub_file,
            stage_hub_prefix,
        )

        print("[import-check] OK: all deferred imports resolved", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    capture_sentinel = None
    if args.local_rollout_root is None:
        capture_sentinel = wait_for_capture_sentinel(args)

    results = []
    for i, behavior in enumerate(args.behaviors):
        print(
            f"[phase=pvsynth_score_behavior] unit {i + 1}/{len(args.behaviors)} {behavior}",
            flush=True,
        )
        results.append(run_behavior(behavior, args))

    sentinel = {
        "rung": RUNG,
        "split": SPLIT,
        "behaviors": list(args.behaviors),
        "capture_sentinel": capture_sentinel,
        "per_behavior": [
            {
                "behavior": r["behavior"],
                "coverage": r["coverage"],
                "spread": r["spread"],
                "content_drops": r["judge"]["tallies"]["n_content_dropped_draws"],
                "transport_losses": r["judge"]["tallies"]["n_transport_lost_draws"],
                "rejudge_recovered": r["judge"]["rejudge"]["n_recovered"],
            }
            for r in results
        ],
        "meta": _run_meta(args),
    }
    sentinel_path = args.out_root / DV_SENTINEL
    _write_json_atomic(sentinel_path, sentinel)

    if not args.skip_upload:
        from explore_persona_space.orchestrate import hub

        for sub in ("dv_dataset", "judge", "spread"):
            local = args.out_root / sub
            if local.exists():
                hub._upload(
                    local,
                    hub.DEFAULT_DATASET_REPO,
                    "dataset",
                    f"{args.hf_prefix}/{sub}",
                    ignore_patterns=["**/cache/**", "**/cache_rejudge/**"],
                    raise_on_error=True,
                )
        hub._upload(
            sentinel_path,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            f"{args.hf_prefix}/{DV_SENTINEL}",
            upload_as_file=True,
            raise_on_error=True,
        )

    note_lines = [
        f"pvsynth rung — judge + DV leg complete (rung={RUNG}, split={SPLIT}).",
        "",
        "DV: graded 0-100 trait eval_prompt rubric, "
        f"{sentinel['meta']['n_judge_draws']} draws @ temp "
        f"{sentinel['meta']['judge_temperature']}, judge "
        f"{sentinel['meta']['judge_model']}, max_tokens "
        f"{sentinel['meta']['judge_max_tokens']} "
        f"(re-judge {sentinel['meta']['rejudge_max_tokens']}).",
        "",
    ]
    for pb in sentinel["per_behavior"]:
        sp = pb["spread"]
        note_lines.append(
            f"- {pb['behavior']}: contexts_with_dv={pb['coverage']['n_contexts_with_dv']}"
            f"/{pb['coverage']['n_contexts']} groups={pb['coverage']['n_groups']} "
            f"mean={sp['mean']} sd={sp['sd']} min={sp['min']} max={sp['max']} "
            f"content_drops={pb['content_drops']} transport_losses={pb['transport_losses']} "
            f"rejudge_recovered={pb['rejudge_recovered']}"
        )
    note_lines += [
        "",
        "DEVIATION: hallucination uses the trait rubric here (the PV eval questions "
        "carry no reference answers, so the train-side three-way "
        "alias-match/fabrication DV is inapplicable) — dv_construct is recorded per "
        "behavior; do not pool the two constructs.",
        "",
        "REMAINING: arm scoring (the 6-arm transfer roster x both variants x 28 layers) "
        "needs the behavior's TRAIN capture store refit (32-70 GB/behavior on HF) and "
        "must NOT run on the shared VM — route it to a pod / big-disk CPU lane.",
        f"Artifacts: {args.hf_prefix}/{{raw_completions,capture_store,dv_dataset,judge,spread}}",
    ]
    note_path = args.out_root / "pvsynth_dv_progress_note.md"
    note_path.write_text("\n".join(note_lines) + "\n")
    if not args.no_post_marker:
        _post_progress_marker(args, note_path)

    print(f"[phase=done] pvsynth DV complete: {sentinel_path}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
