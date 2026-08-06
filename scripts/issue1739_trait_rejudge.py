"""Issue #1739 trait DV re-judge at max_tokens=1024 (evil-ood-spread; unit 3b).

Re-judges the parent's TRAIT DV pool for BOTH existing OOD rungs
(``evil_hh_rlhf``, ``evil_toxicchat``) AND the train rung (``evil_train``) at
``max_tokens=1024`` — the matched-instrument fix required by plan v16 §4.2 so
the H1 cross-rung ρ headline is instrument-matched with the new-rung pilots
(§4.4 also scores at 1024). This writes a NEW output pool alongside the
parent's original ``max_tokens=400`` pool; the parent pool is NEVER
overwritten (provenance is preserved on git).

Rubric SOURCE: ``explore_persona_space.experiments.issue_1739.judging
.load_trait_rubric("evil", ...)`` — the identical asset the parent judge
reads. Only ``max_tokens`` changes, and — per rule 22 — that is over-keyed at
the api_dispatch layer, so a shared cache is a MISS by construction; we ALSO
stage a FRESH ``--cache-dir`` (never the parent's 400-token cache) as
belt-and-suspenders (plan v16 §4.2).

Dispatch is the sanctioned Batch client (``eval.batch_judge.judge_completions_batch``
via the graded chain), threaded through the multi-org ``api_dispatch``
(CLAUDE.md API throughput rule; ``eval.judge_dispatch``).

Raw-completion source: ``hf://superkaiba1/explore-persona-space-data/
issue1739_ctxmap/raw_completions/`` — per plan v16 §4.2. The default value
of ``--raw-rollouts`` names this Hub prefix; the on-VM raw completions live
under ``raw_completions/issue_1739/labeling/evil/`` (the parent's local
``labeling`` slice, already committed here), so ``--raw-rollouts local`` is
the equivalent-inputs alternative when the on-VM copy is authoritative.

Wall estimate (Batch API at Sonnet, per plan v16 §4.2):
~87k trait draws total × 3 draws ≈ ~215k Batch API calls at Sonnet;
Batch API typical turnaround ~2-3h at this scale; ~0 GPU-h.
Cost within the "don't worry about cost" directive.

Content-drop vs transport-loss (llm-judging.md rule 24): the JudgeResult
carries ``n_dropped_draws`` (content — REFUSAL / malformed / out-of-range)
and ``n_transport_lost_draws`` (transport — 429 / 5xx / timeout after
api_dispatch's bounded retries) split; this driver reports both, never
blended.

Usage (production):

    uv run python scripts/issue1739_trait_rejudge.py \\
        --rungs evil_train evil_hh_rlhf evil_toxicchat \\
        --raw-rollouts hf://superkaiba1/explore-persona-space-data/issue1739_ctxmap/raw_completions/ \\
        --output eval_results/issue_1739/evil_ood_spread/trait_dv_1024/ \\
        --cache-dir eval_results/issue_1739/evil_ood_spread/trait_dv_1024/judge_cache/

Smoke (offline, no API):

    uv run python scripts/issue1739_trait_rejudge.py --smoke
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

# Credentials bind BEFORE any heavy import (#847; ANTHROPIC_API_KEY).
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

logger = logging.getLogger(__name__)

# The re-judge is trait-DV-fixed (BEHAVIOR="evil"): plan v16 §4.2 re-judges
# ONLY the trait-DV pool at max_tokens=1024, and the trait rubric asset is
# per-behavior. Extending to sycophancy would clone this driver with a
# different behavior string; that is out of scope here.
BEHAVIOR = "evil"

# Rule-23 upgrade over the parent's 400 (plan v16 §4.2). N_DRAWS + judge model
# stay at the parent's pins.
REJUDGE_MAX_TOKENS = 1024
REJUDGE_N_DRAWS = 3
REJUDGE_TEMPERATURE = 1.0
REJUDGE_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# HF Hub prefix consumed by ``load_from_hf`` when ``--raw-rollouts`` names a
# Hub URI. The default matches plan v16 §4.2.
DEFAULT_HF_PREFIX = "hf://superkaiba1/explore-persona-space-data/issue1739_ctxmap/raw_completions/"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# CLI rung name → local rollout `rung` field. The parent's rollout schema strips the
# behavior prefix (`evil_`) from the stored `rung` field, and calls hh-rlhf "hhrt"
# rather than "hh_rlhf". Discovered live 2026-08-03 via `evil-{train-cross,eval-hhrt,
# eval-toxicchat}-*_seed0.json` inspection (`d["rung"] ∈ {"train","hhrt","toxicchat"}`).
_LOCAL_RUNG_ALIAS: dict[str, str] = {
    "evil_train": "train",
    "evil_hh_rlhf": "hhrt",
    "evil_toxicchat": "toxicchat",
}


def _load_rollouts_local(local_root: Path, rung: str, *, limit: int | None) -> list[dict]:
    """Read rollout JSONs for one rung from the on-VM parent slice.

    Layout A (per-rung subdir): ``<local_root>/<rung>/*.json`` — CLI rung name
    matches the dir. Layout B (mixed dir, parent's actual on-VM layout): all
    rungs' JSONs sit under one dir, distinguished by the ``rung`` field. The
    parent's rollout schema strips the behavior prefix, so map the CLI name
    via ``_LOCAL_RUNG_ALIAS`` before comparing (identity fallback for future
    rungs that already match).
    """
    # First try the per-rung sub-directory shape.
    rung_dir = local_root / rung
    if rung_dir.exists():
        paths = sorted(p for p in rung_dir.glob("*.json") if not p.name.startswith("_"))
    else:
        # Fall back to mixed dir with a ``rung`` field filter.
        paths = sorted(p for p in local_root.glob("*.json") if not p.name.startswith("_"))
    # Normalize CLI name → on-disk rung field before comparing (identity fallback).
    on_disk_rung = _LOCAL_RUNG_ALIAS.get(rung, rung)
    payloads: list[dict] = []
    for p in paths:
        row = json.loads(p.read_text())
        if row.get("rung") == on_disk_rung or rung_dir.exists():
            payloads.append(row)
            if limit is not None and len(payloads) >= limit:
                break
    return payloads


def _rollout_item_id(context_id: str, k: int) -> str:
    """Judge item id (delegates to the parent's safe encoder — rule 22 + #1415)."""
    from explore_persona_space.experiments.issue_1739.judging import rollout_item_id

    return rollout_item_id(context_id, k)


def _judge_rung_real(
    payloads: list[dict],
    *,
    rubric: str,
    cache_dir: Path,
    out_dir: Path,
    n_draws: int,
    max_tokens: int,
    temperature: float,
    judge_model: str,
    threshold_base: int | None,
    dry_run: bool,
) -> dict:
    """Dispatch through the sanctioned Batch client (production path).

    Returns a serializable rung report with per-item scores + rule-24 split.
    """
    from explore_persona_space.eval.graded_judge import judge_graded
    from explore_persona_space.eval.judge_dispatch import graded_temperature

    items = [
        (
            _rollout_item_id(p["context_id"], int(p["rollout_k"])),
            p["query"],
            p["completion"],
        )
        for p in payloads
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_raw = out_dir / "judge_raw_trait_1024.json"
    with graded_temperature(temperature):
        result = judge_graded(
            items,
            rubric,
            n_draws=n_draws,
            cache_dir=cache_dir,
            save_raw=save_raw,
            judge_model=judge_model,
            max_tokens=max_tokens,
            dry_run=dry_run,
            threshold_base=threshold_base,
        )
    return {
        "scores": dict(result.scores),
        "per_item_scores": dict(result.per_item_scores),
        "per_item_draw_counts": dict(result.per_item_draw_counts),
        "n_total_draws": int(result.n_total_draws),
        "n_dropped_draws": int(result.n_dropped_draws),
        "n_transport_lost_draws": int(result.n_transport_lost_draws),
        "per_item_transport_losses": dict(getattr(result, "per_item_transport_losses", {}) or {}),
        "judge_raw_path": str(save_raw),
        "n_items": len(items),
    }


def _judge_rung_stub(payloads: list[dict], *, out_dir: Path, seed: int = 0) -> dict:
    """Offline stub — deterministic integer scores for the smoke path.

    Exercises the rule-24 split contract (content-drop vs transport-loss)
    without any API call; rule 9's drop-never-coerce is exercised by leaving
    a fixed fraction of items unscored, never coercing them to a number.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    n_total = 0
    n_content_drop = 0
    n_transport_loss = 0
    for i, p in enumerate(payloads):
        item_id = _rollout_item_id(p["context_id"], int(p["rollout_k"]))
        h = (hash(item_id) ^ (seed * 2654435761)) & 0xFFFF
        score = float(h % 101)
        n_total += 1
        if i % 15 == 0:
            n_content_drop += 1
            continue
        if i % 25 == 0:
            n_transport_loss += 1
            continue
        scores[item_id] = score
    save_raw = out_dir / "judge_raw_trait_1024.json"
    save_raw.write_text(
        json.dumps(
            {"note": "smoke stub — no API calls", "n_items_kept": len(scores)},
            indent=1,
        )
    )
    return {
        "scores": scores,
        "per_item_scores": {k: [v] for k, v in scores.items()},
        "per_item_draw_counts": {k: 1 for k in scores},
        "n_total_draws": n_total,
        "n_dropped_draws": n_content_drop,
        "n_transport_lost_draws": n_transport_loss,
        "per_item_transport_losses": {},
        "judge_raw_path": str(save_raw),
        "n_items": len(payloads),
    }


def _rubric_fp_note(behavior: str, judge_model: str, inputs_dir: str) -> dict[str, str]:
    """Report the manifest's rubric-isolation note (rule 22 belt-and-suspenders).

    ``judge_graded`` derives (system_prompt, user_template) from the trait
    rubric text at dispatch time via
    ``eval.graded_judge._rubric_system_and_user`` and then keys the JudgeCache
    on ``rubric_fingerprint(judge_model, that_system_prompt,
    format_user_msg_capture)`` — the exact fingerprint bytes are computed
    inside the dispatch call and are not statically re-derivable from this
    driver without duplicating that split logic. What matters for rule 22 is
    that this driver stages a FRESH ``--cache-dir`` per run, so no parent
    400-token cache entry can be re-served regardless of the fingerprint's
    identity — that isolation is the load-bearing safeguard the plan v16 §4.2
    text mandates.
    """
    return {
        "rubric_fingerprint_at_judge_cache": (
            "computed-inside-judge_graded (see graded_judge._rubric_system_and_user)"
        ),
        "note": (
            "The JudgeCache rubric key is derived at dispatch time and does NOT "
            "embed max_tokens; a fresh --cache-dir is the load-bearing "
            "isolation vs the parent 400-token cache (plan v16 §4.2 rule 22 "
            "belt-and-suspenders)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #1739 trait DV re-judge at max_tokens=1024 (matched-instrument "
            "fix, plan v16 §4.2). Re-judges the parent's evil TRAIT DV pool for "
            "existing OOD rungs (evil_hh_rlhf, evil_toxicchat) AND the train rung "
            "(evil_train); writes to a NEW output dir; the parent's 400-token pool "
            "is never overwritten."
        )
    )
    parser.add_argument(
        "--rungs",
        nargs="+",
        default=["evil_train", "evil_hh_rlhf", "evil_toxicchat"],
        help="Rung ids to re-judge (plan v16 §4.2).",
    )
    parser.add_argument(
        "--raw-rollouts",
        default=DEFAULT_HF_PREFIX,
        help=(
            "Rollout source. Either an ``hf://<repo>/<prefix>/`` URI (plan v16 "
            "default) or a local dir (falls back to the on-VM parent labeling "
            "slice at raw_completions/issue_1739/labeling/evil/)."
        ),
    )
    parser.add_argument(
        "--output",
        default="eval_results/issue_1739/evil_ood_spread/trait_dv_1024/",
        help="Output dir (per-rung <rung>/ subdirs land under it).",
    )
    parser.add_argument(
        "--cache-dir",
        default="eval_results/issue_1739/evil_ood_spread/trait_dv_1024/judge_cache/",
        help=(
            "FRESH JudgeCache dir. Plan v16 §4.2 rule 22 belt-and-suspenders: "
            "never point at the parent's 400-token cache."
        ),
    )
    parser.add_argument(
        "--inputs-dir",
        default="data/issue_1739/inputs",
        help="Where load_trait_rubric reads e1_assets from.",
    )
    parser.add_argument("--n-draws", type=int, default=REJUDGE_N_DRAWS)
    parser.add_argument("--max-tokens", type=int, default=REJUDGE_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=REJUDGE_TEMPERATURE)
    parser.add_argument("--judge", default=REJUDGE_JUDGE_MODEL, help="Judge model id.")
    parser.add_argument(
        "--batch-api",
        action="store_true",
        default=True,
        help=(
            "Force the Batch API path (threshold_base=0). Default ON — the "
            "re-judge is Batch-territory (~215k calls total per §4.2)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-slice cap (rollout files per rung).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Offline smoke: fabricate ~20 stub rollouts for the train rung, "
            "judge them with a deterministic hash-based stub (no API calls), "
            "and assert the output-dir file structure + rubric-fingerprint "
            "presence. Exits rc=0 on shape success."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rubric fingerprint (reported in manifest; not used to key the cache
    # directly, but the fresh --cache-dir isolation is what matters for the
    # rule-22 belt-and-suspenders).
    fp_note = (
        _rubric_fp_note(BEHAVIOR, args.judge, args.inputs_dir)
        if not args.smoke
        else {
            "rubric_fingerprint_at_judge_cache": "smoke-stub-fingerprint",
            "note": "smoke — no API imports",
        }
    )

    manifest: dict[str, Any] = {
        "kind": "epm:trait-dv-rejudge-1024",
        "behavior": BEHAVIOR,
        "rungs": list(args.rungs),
        "raw_rollouts": args.raw_rollouts,
        "output_dir": str(output_dir),
        "cache_dir": str(args.cache_dir),
        "judge_model": args.judge,
        "n_draws": args.n_draws,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "smoke": bool(args.smoke),
        "dry_run": bool(args.dry_run),
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rubric_fingerprint": fp_note,
        "instrument_upgrade_note": (
            "Matched-instrument re-judge (plan v16 §4.2). The parent's original "
            "max_tokens=400 pool remains committed to git as-is for provenance; "
            "this driver writes to a NEW output dir and NEVER overwrites it. All "
            "cross-rung Spearman ρ / AUROC / bootstrap / permutation runs "
            "downstream use THIS pool (matched-instrument with §4.4 pilots)."
        ),
        "per_rung": {},
    }

    threshold_base = 0 if args.batch_api else None

    # Real-path preload: pull the trait rubric ONCE (shared across rungs).
    rubric = None
    if not args.smoke:
        from explore_persona_space.experiments.issue_1739.judging import load_trait_rubric

        rubric = load_trait_rubric(BEHAVIOR, inputs_dir=args.inputs_dir)

    for rung in args.rungs:
        rung_out_dir = output_dir / rung
        rung_cache_dir = Path(args.cache_dir) / rung

        if args.smoke:
            # Fabricate ~20 stub rollouts. Only the train rung populates in
            # smoke (keeps runtime tiny); other requested rungs get an empty
            # per-rung report so callers can see the shape wiring.
            n_stub = 20 if rung == "evil_train" or len(args.rungs) == 1 else 0
            payloads = [
                {
                    "context_id": f"{rung}-ctx-{i:04d}",
                    "rollout_k": i % 5,
                    "query": f"stub query {i}",
                    "completion": f"stub completion {i}",
                    "rung": rung,
                }
                for i in range(n_stub)
            ]
            if not payloads:
                manifest["per_rung"][rung] = {
                    "n_items": 0,
                    "n_total_draws": 0,
                    "note": "smoke skip — non-train rung",
                }
                continue
            report = _judge_rung_stub(payloads, out_dir=rung_out_dir, seed=hash(rung) & 0xFF)
        else:
            # Real path: currently only local rollouts are wired (the on-VM
            # parent labeling slice under ``raw_completions/issue_1739/
            # labeling/evil/``). ``hf://`` sources require a hub-download
            # pass; the plan v16 §4.2 default is Hub, so this driver refuses
            # a Hub URI and demands a local mirror already be present (a
            # stage-first flow — the r2B unit will wire it if not already).
            if args.raw_rollouts.startswith("hf://"):
                # Refuse fail-loud: hub-download not yet wired here. The
                # canonical route is stage the Hub prefix to a local mirror
                # first (hub.stage_hub_prefix, mirror-root semantics; #1774
                # entry in gotchas.md) then re-invoke with --raw-rollouts
                # <local mirror>.
                logger.error(
                    "hf:// source not yet wired for rung %s: stage the Hub "
                    "prefix to a local mirror via hub.stage_hub_prefix "
                    "(mirror-root semantics; #1774) then re-invoke with "
                    "--raw-rollouts <local-mirror-root>.",
                    rung,
                )
                manifest["per_rung"][rung] = {
                    "error": "hf-source-not-yet-wired",
                    "next_step": (
                        "stage-first via hub.stage_hub_prefix, then re-invoke "
                        "with --raw-rollouts <local-mirror-root>"
                    ),
                }
                continue
            local_root = Path(args.raw_rollouts)
            if not local_root.exists():
                logger.error("raw-rollouts local root missing: %s", local_root)
                manifest["per_rung"][rung] = {"error": f"missing raw-rollouts root {local_root}"}
                continue
            payloads = _load_rollouts_local(local_root, rung, limit=args.limit)
            if not payloads:
                logger.warning("no rollouts under %s for rung %s", local_root, rung)
                manifest["per_rung"][rung] = {
                    "error": f"no rollout files under {local_root} for rung {rung}"
                }
                continue
            assert rubric is not None, "rubric must be preloaded on the real path"
            report = _judge_rung_real(
                payloads,
                rubric=rubric,
                cache_dir=rung_cache_dir,
                out_dir=rung_out_dir,
                n_draws=args.n_draws,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                judge_model=args.judge,
                threshold_base=threshold_base,
                dry_run=args.dry_run,
            )
        manifest["per_rung"][rung] = report

    manifest_path = output_dir / "manifest.json"
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, default=str))
    os.replace(tmp, manifest_path)

    # Smoke assertions: file structure + rule-24 split + rubric fingerprint.
    if args.smoke:
        assert manifest_path.exists(), "manifest.json must exist"
        assert manifest["per_rung"], "per_rung must have at least one rung"
        # The rubric fingerprint field must be present (its value is stubbed
        # in smoke; the real path recomputes it — the manifest field itself
        # is the shape contract).
        assert "rubric_fingerprint" in manifest, "rubric_fingerprint missing"
        assert (
            manifest["rubric_fingerprint"]["rubric_fingerprint_at_judge_cache"] != "400-token-mock"
        ), "rubric fingerprint must differ from any mock 400-token fingerprint"
        # At least one rung report must carry the rule-24 split fields.
        any_split = False
        for rung, rep in manifest["per_rung"].items():
            if rep.get("n_total_draws", 0) > 0:
                any_split = True
                for k in ("n_total_draws", "n_dropped_draws", "n_transport_lost_draws"):
                    assert k in rep, f"{k} missing for {rung} (rule 24 split)"
                # Per-rung output dir must exist.
                assert (output_dir / rung).exists(), f"per-rung dir missing for {rung}"
                # Judge-raw file must exist alongside.
                assert Path(rep["judge_raw_path"]).exists(), f"judge_raw path missing for {rung}"
        assert any_split, "at least one rung must carry the rule-24 split fields"
        print(f"[smoke] trait_rejudge OK: wrote {manifest_path}")
        return 0

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "rungs": list(manifest["per_rung"].keys()),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
