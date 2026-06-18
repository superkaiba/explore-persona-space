"""Task #612 P2j (VM, API+CPU) — judge the base pass, select the final panel.

Inputs: the P2 base-pass eval tree (``eval_results/issue_612/base/``, local or
re-fetched from HF) + ``panel_candidates.json`` (P1 output). Steps:

1. Judge every candidate's 600 base completions (Haiku, checkpoint/resume per
   panel file — this doubles as P6's first slice; P6 skips already-judged).
2. Base prior per candidate = agreement rate over the audited 60 claims.
3. Selection (plan §4 P1): 11 mandatory personas + greedy set-cover toward
   >=2 personas per (source x cosine-bin) where the pool allows, tie-broken
   to minimize max-over-sources |Pearson(cosine, base prior)| (decorrelation),
   up to 30 personas. Coverage gaps are REPORTED, never silently filled.
4. Write ``data/issue_612/panel/panel_set.json`` (with per-persona prompt,
   provenance, cosines, bins, base prior, ``neg_member_for`` flags incl. the
   qwen_default template-collision flag) + upload to HF (the cross-phase
   contract the production driver polls for) — caller commits the file.

CLI (VM):
    uv run python -m explore_persona_space.experiments.sycophancy_onpolicy_612.panel_select \
        --base-dir eval_results/issue_612/base --candidates
data/issue_612/panel/panel_candidates.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    COSINE_BINS,
    HF_DATA_PREFIX,
    HF_DATA_REPO,
    JUDGE_MODEL,
    MANDATORY_PANEL,
    NEGATIVES_BY_SOURCE,
    PANEL_SIZE,
    SOURCES,
)

log = logging.getLogger("issue_612.panel_select")

JUDGE_API_ERROR_CEILING = 0.02


def fetch_p2j_inputs(base_dir: Path, candidates_path: Path) -> dict:
    """Fetch the pod-uploaded P1/P2 artifacts onto the VM (fail-loud).

    P2j runs on the VM AFTER the pod's stage 1 uploaded
    ``{HF_DATA_PREFIX}/panel/panel_candidates.json`` and
    ``{HF_DATA_PREFIX}/eval_results/base/**`` — and the production driver's
    stage 1b BLOCKS a live multi-GPU instance polling for P2j's
    panel_set.json, so this fetch is load-bearing, not a convenience. Uses
    ``list_repo_files`` + per-file ``hf_hub_download`` (NOT
    ``snapshot_download(allow_patterns=...)`` — the siblings-truncation
    gotcha silently returns 0 files on large repos). Files already local are
    kept (checkpoint/resume); zero matching remote+local files raises.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    stats = {"candidates_fetched": False, "base_fetched": 0, "base_kept_local": 0}
    repo_files = list(list_repo_files(HF_DATA_REPO, repo_type="dataset"))

    cand_repo = f"{HF_DATA_PREFIX}/panel/panel_candidates.json"
    if not candidates_path.exists():
        if cand_repo not in repo_files:
            raise FileNotFoundError(
                f"panel_candidates.json neither local ({candidates_path}) nor on HF "
                f"({HF_DATA_REPO}/{cand_repo}) — has the pod's panel:build:0 cell run "
                f"and uploaded? P2j cannot proceed."
            )
        cached = hf_hub_download(repo_id=HF_DATA_REPO, filename=cand_repo, repo_type="dataset")
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, candidates_path)
        stats["candidates_fetched"] = True
        log.info("fetched %s -> %s", cand_repo, candidates_path)

    base_prefix = f"{HF_DATA_PREFIX}/eval_results/base/"
    remote_base = [f for f in repo_files if f.startswith(base_prefix)]
    have_local = bool(list(base_dir.glob("sycophancy_eval_*.json")))
    if not remote_base and not have_local:
        raise FileNotFoundError(
            f"no base-pass eval files local ({base_dir}) or on HF under "
            f"{HF_DATA_REPO}/{base_prefix} — has the pod's base:pass:0 cell run "
            f"and uploaded? P2j cannot proceed."
        )
    for rf in remote_base:
        dest = base_dir / rf[len(base_prefix) :]
        if dest.exists():
            stats["base_kept_local"] += 1
            continue
        cached = hf_hub_download(repo_id=HF_DATA_REPO, filename=rf, repo_type="dataset")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        stats["base_fetched"] += 1
    log.info(
        "P2j inputs ready: candidates_fetched=%s, base files fetched=%d / kept local=%d",
        stats["candidates_fetched"],
        stats["base_fetched"],
        stats["base_kept_local"],
    )
    return stats


def assert_base_pass_coverage(base_dir: Path, candidates: dict[str, dict]) -> None:
    """Every P1 candidate must have its base-pass eval JSON before judging
    starts (fail-loud BEFORE spending judge calls on an incomplete tree)."""
    missing = [
        name for name in candidates if not (base_dir / f"sycophancy_eval_{name}.json").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"base pass incomplete: {len(missing)}/{len(candidates)} candidates have no "
            f"eval JSON under {base_dir} (first missing: {sorted(missing)[:5]}) — "
            f"re-run/finish base:pass:0 before P2j."
        )


def judge_base_panel(base_dir: Path, panel: str, *, concurrency: int) -> dict:
    """Judge one candidate's base completions (checkpointed); returns summary."""
    from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import (
        judge_batch,
        serialize_verdicts,
    )

    out_dir = base_dir / "judgments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{panel}.json"
    if out_path.exists():
        payload = json.loads(out_path.read_text())
        n_err = sum(
            1 for v in payload["verdicts"] if v["error"] and "unparseable" not in (v["error"] or "")
        )
        if n_err == 0:
            return payload
        log.warning("%s: %d API-error verdicts — re-judging", panel, n_err)
    eval_path = base_dir / f"sycophancy_eval_{panel}.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"base-pass eval JSON missing: {eval_path}")
    eval_payload = json.loads(eval_path.read_text())
    records = eval_payload["completions"]
    rollouts = [{"wrong_claim": r["claim"], "completion": r["completion"]} for r in records]
    verdicts = asyncio.run(judge_batch(rollouts, model=JUDGE_MODEL, max_concurrency=concurrency))
    rows = serialize_verdicts(verdicts)
    for rec, v in zip(records, rows, strict=True):
        v["claim_idx"] = rec["claim_idx"]
        v["rollout_idx"] = rec["rollout_idx"]
    n_err = sum(1 for v in rows if v["error"] and "unparseable" not in (v["error"] or ""))
    if n_err > JUDGE_API_ERROR_CEILING * len(rows):
        raise RuntimeError(f"{panel}: {n_err}/{len(rows)} post-retry judge API errors")
    payload = {
        "panel": panel,
        "model": JUDGE_MODEL,
        "n_verdicts": len(rows),
        "rate": sum(1 for v in rows if v["agreed"]) / max(len(rows), 1),
        "verdicts": rows,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.write_text(json.dumps(payload))
    log.info("judged %s: rate=%.3f (%d verdicts)", panel, payload["rate"], len(rows))
    return payload


def _bin_label(lo: float, hi: float) -> str:
    return f"[{lo},{hi})"


def _max_abs_corr(selected: list[str], candidates: dict[str, dict], priors: dict[str, float]):
    """max over sources of |Pearson(cosine-to-source, base prior)| on the set."""
    import numpy as np

    worst = 0.0
    pri = np.array([priors[n] for n in selected])
    if len(selected) < 3 or float(np.std(pri)) == 0.0:
        return 0.0
    for s in SOURCES:
        cosv = np.array([candidates[n]["cosines"][s] for n in selected])
        if float(np.std(cosv)) == 0.0:
            continue
        worst = max(worst, abs(float(np.corrcoef(cosv, pri)[0, 1])))
    return worst


def select_panel(candidates: dict[str, dict], priors: dict[str, float]) -> tuple[list[str], dict]:
    """Mandatory-11 + greedy bin coverage with decorrelation tie-break."""
    missing_prior = [n for n in candidates if n not in priors]
    if missing_prior:
        raise AssertionError(f"candidates without base prior (judge first): {missing_prior[:5]}")
    selected = list(MANDATORY_PANEL)

    def deficits(sel: list[str]) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        for s in SOURCES:
            for lo, hi in COSINE_BINS:
                label = _bin_label(lo, hi)
                have = sum(1 for n in sel if candidates[n]["bin_by_source"].get(s) == label)
                fillable = sum(
                    1 for n in candidates if candidates[n]["bin_by_source"].get(s) == label
                )
                want = min(2, fillable)
                if have < want:
                    out[(s, label)] = want - have
        return out

    while len(selected) < PANEL_SIZE:
        pool = [n for n in candidates if n not in selected]
        if not pool:
            break
        defs = deficits(selected)

        def gain(n: str, _defs=defs) -> int:
            g = 0
            for (s, label), _need in _defs.items():
                if candidates[n]["bin_by_source"].get(s) == label:
                    g += 1
            return g

        best_gain = max((gain(n) for n in pool), default=0)
        contenders = [n for n in pool if gain(n) == best_gain]
        # Tie-break: minimize the decorrelation objective on the would-be set.
        contenders.sort(key=lambda n: (_max_abs_corr([*selected, n], candidates, priors), n))
        selected.append(contenders[0])

    report = {
        "n_selected": len(selected),
        "remaining_deficits": {f"{s}|{b}": d for (s, b), d in deficits(selected).items()},
        "decorrelation_max_abs_pearson": _max_abs_corr(selected, candidates, priors),
        "bin_coverage": {
            s: {
                _bin_label(lo, hi): sorted(
                    n
                    for n in selected
                    if candidates[n]["bin_by_source"].get(s) == _bin_label(lo, hi)
                )
                for lo, hi in COSINE_BINS
            }
            for s in SOURCES
        },
    }
    return selected, report


def neg_member_flags(name: str) -> list[str]:
    """Sources for which this panel persona is a trained-negative cell
    (plus the qwen_default template-collision flag for ALL sources, #608)."""
    if name == "qwen_default":
        return list(SOURCES)
    return [s for s in SOURCES if name in NEGATIVES_BY_SOURCE[s]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 P2j — base-pass judging + final panel selection (VM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-dir", type=Path, default=Path("eval_results/issue_612/base"))
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/issue_612/panel/panel_candidates.json")
    )
    parser.add_argument("--out", type=Path, default=Path("data/issue_612/panel/panel_set.json"))
    parser.add_argument("--judge-concurrency", type=int, default=24)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip the HF fetch of pod-uploaded P1/P2 artifacts (fixture smokes / "
        "fully-local re-runs only; the production P2j path MUST fetch).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=p2j_select] %(message)s", stream=sys.stdout
    )

    if not args.no_fetch:
        fetch_p2j_inputs(args.base_dir, args.candidates)
    if not args.candidates.exists():
        raise FileNotFoundError(f"panel_candidates.json missing: {args.candidates}")
    payload = json.loads(args.candidates.read_text())
    candidates: dict[str, dict] = payload["candidates"]
    assert_base_pass_coverage(args.base_dir, candidates)

    # 1-2. judge every candidate's base pass -> priors
    priors: dict[str, float] = {}
    for name in sorted(candidates):
        priors[name] = judge_base_panel(args.base_dir, name, concurrency=args.judge_concurrency)[
            "rate"
        ]

    # 3. selection
    selected, report = select_panel(candidates, priors)
    if len(selected) < 24:
        raise AssertionError(
            f"panel below the 24-persona floor ({len(selected)}) — must-ask boundary (plan §13)"
        )

    # 4. panel_set.json
    personas = {
        name: {
            "prompt": candidates[name]["prompt"],
            "provenance": candidates[name]["provenance"],
            "cosines": candidates[name]["cosines"],
            "bin_by_source": candidates[name]["bin_by_source"],
            "base_rate": priors[name],
            "neg_member_for": neg_member_flags(name),
            "mandatory": name in MANDATORY_PANEL,
        }
        for name in selected
    }
    out_payload = {
        "schema_version": 1,
        "provenance": "p2j_selected",
        "personas": personas,
        "selection_report": report,
        "all_candidate_priors": priors,
        "metadata": {
            "judge_model": JUDGE_MODEL,
            "git_commit_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2))
    log.info(
        "panel_set.json -> %s (%d personas; decorrelation %.3f; deficits %s)",
        args.out,
        len(selected),
        report["decorrelation_max_abs_pearson"],
        report["remaining_deficits"] or "none",
    )

    if not args.skip_upload:
        from explore_persona_space.orchestrate.hub import upload_dataset

        hub_path = upload_dataset(
            str(args.out), path_in_repo=f"{HF_DATA_PREFIX}/panel/panel_set.json"
        )
        if not hub_path:
            raise RuntimeError("panel_set.json HF upload failed (the driver polls for it)")
        log.info("panel_set.json uploaded -> %s", hub_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
