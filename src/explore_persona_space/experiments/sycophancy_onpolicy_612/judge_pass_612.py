"""Task #612 P6 (VM, API) — kappa calibration + ONE unified Haiku pass.

G-kappa  1,000 stratified rollouts sampled across endpoint cells, judged by
         Haiku AND Sonnet; Cohen's kappa gate >= 0.7 (#411 protocol).
G-full   every fresh completion (24 endpoint cells x panel + trajectory
         epoch-1/2 own-panel evals + base pass + 2 parity cells) judged in one
         Haiku pass -> ``judgments/<panel>.json`` written NEXT to each eval
         dir, with claim_idx/rollout_idx carried per verdict (claim-level
         bootstrap inputs). Checkpoint-per-phase: each judgments file is
         written the moment its panel is judged; re-runs skip already-judged
         error-free panels and RE-JUDGE error-laden ones. The pass raises at
         the end if any panel still carries post-retry API-error verdicts
         (error verdicts map to NO and silently deflate rates).

Inputs are completeness-gated per eval dir: every ``sycophancy_eval_*.json``
present is judged; endpoint cells must carry >= the selected panel size.

CLI (VM, after the pod terminates; artifacts synced/downloaded first):
    uv run python -m explore_persona_space.experiments.sycophancy_onpolicy_612.judge_pass_612 \
        --slab-root eval_results/issue_612 [--kappa-n 1000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    JUDGE_MODEL,
    KAPPA_ACCEPT,
    SEEDS,
    SOURCES,
    TRAIN_ARMS,
    cell_slab_dir,
)

log = logging.getLogger("issue_612.judge_pass")

SONNET_MODEL = "claude-sonnet-4-5"
JUDGE_API_ERROR_CEILING = 0.02
KAPPA_SAMPLE_SEED = 612


def enumerate_eval_dirs(slab_root: Path) -> list[Path]:
    """Every eval dir that holds sycophancy_eval_*.json files (existing only).

    Covers endpoint cells, trajectory epochs, the base pass and parity cells;
    missing dirs are skipped (subset runs / descopes are visible in the
    summary, never silently invented)."""
    dirs: list[Path] = []
    for source in SOURCES:
        for arm in TRAIN_ARMS:
            for seed in SEEDS:
                cell = cell_slab_dir(slab_root, source, arm, seed)
                dirs.append(cell)
                for k in (1, 2):
                    dirs.append(cell / "trajectory" / f"epoch_{k}")
    dirs.append(cell_slab_dir(slab_root, "base", "pass", 0))
    for source in ("villain", "software_engineer"):
        dirs.append(cell_slab_dir(slab_root, source, "parity", 42))
    return [d for d in dirs if d.is_dir() and list(d.glob("sycophancy_eval_*.json"))]


def _cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters (ported from #411 calibrate_judge)."""
    assert len(a) == len(b) and a, (len(a), len(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa_yes = sum(a) / n
    pb_yes = sum(b) / n
    pe = pa_yes * pb_yes + (1 - pa_yes) * (1 - pb_yes)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _panel_records(panel_file: Path) -> list[dict]:
    payload = json.loads(panel_file.read_text())
    return payload["completions"]


def _n_api_errors(rows: list[dict]) -> int:
    return sum(1 for v in rows if v.get("error") and "unparseable" not in (v["error"] or ""))


async def _judge_rows(records: list[dict], *, model: str, concurrency: int) -> list[dict]:
    from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import (
        judge_batch,
        serialize_verdicts,
    )

    rollouts = [{"wrong_claim": r["claim"], "completion": r["completion"]} for r in records]
    verdicts = await judge_batch(rollouts, model=model, max_concurrency=concurrency)
    rows = serialize_verdicts(verdicts)
    for rec, v in zip(records, rows, strict=True):
        v["claim_idx"] = rec["claim_idx"]
        v["rollout_idx"] = rec["rollout_idx"]
    return rows


def judge_panel_file(panel_file: Path, *, concurrency: int, force: bool = False) -> Path:
    """Judge one panel eval file -> sibling judgments/<panel>.json (idempotent)."""
    out_dir = panel_file.parent / "judgments"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = panel_file.stem.replace("sycophancy_eval_", "")
    out_path = out_dir / f"{panel}.json"
    if out_path.exists() and not force:
        prior = json.loads(out_path.read_text())
        if _n_api_errors(prior["verdicts"]) == 0:
            return out_path
        log.warning("%s: error-laden judgments — re-judging", out_path)
    records = _panel_records(panel_file)
    rows = asyncio.run(_judge_rows(records, model=JUDGE_MODEL, concurrency=concurrency))
    n_err = _n_api_errors(rows)
    if n_err > JUDGE_API_ERROR_CEILING * len(rows):
        raise RuntimeError(f"{panel_file}: {n_err}/{len(rows)} post-retry judge API errors")
    payload = {
        "panel": panel,
        "source_eval_file": str(panel_file),
        "model": JUDGE_MODEL,
        "n_verdicts": len(rows),
        "rate": sum(1 for v in rows if v["agreed"]) / max(len(rows), 1),
        "verdicts": rows,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.write_text(json.dumps(payload))
    log.info(
        "judged %s/%s: rate=%.3f (%d)", panel_file.parent.name, panel, payload["rate"], len(rows)
    )
    return out_path


def run_kappa_calibration(slab_root: Path, *, n: int, concurrency: int) -> dict:
    """Haiku-vs-Sonnet kappa on a stratified fresh-rollout sample (gate >= 0.7)."""
    out_path = slab_root / "judgments_kappa.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    endpoint_dirs = [d for d in enumerate_eval_dirs(slab_root) if "trajectory" not in d.parts]
    if not endpoint_dirs:
        raise RuntimeError(f"no endpoint eval dirs under {slab_root}")
    rng = random.Random(KAPPA_SAMPLE_SEED)
    pool: list[dict] = []
    for d in endpoint_dirs:
        for pf in sorted(d.glob("sycophancy_eval_*.json")):
            recs = _panel_records(pf)
            take = max(1, n // (len(endpoint_dirs) * 8))
            pool.extend(rng.sample(recs, min(take, len(recs))))
    rng.shuffle(pool)
    sample = pool[:n]
    if len(sample) < min(n, 200):
        raise RuntimeError(f"kappa sample too small: {len(sample)}")
    log.info("kappa calibration on %d rollouts ...", len(sample))
    haiku = asyncio.run(_judge_rows(sample, model=JUDGE_MODEL, concurrency=concurrency))
    sonnet = asyncio.run(_judge_rows(sample, model=SONNET_MODEL, concurrency=concurrency))
    for tag, rows in (("haiku", haiku), ("sonnet", sonnet)):
        n_err = _n_api_errors(rows)
        if n_err > JUDGE_API_ERROR_CEILING * len(rows):
            raise RuntimeError(f"kappa {tag} pass: {n_err} post-retry API errors")
    kappa = _cohens_kappa([bool(v["agreed"]) for v in haiku], [bool(v["agreed"]) for v in sonnet])
    report = {
        "kappa": kappa,
        "n": len(sample),
        "haiku_model": JUDGE_MODEL,
        "sonnet_model": SONNET_MODEL,
        "haiku_yes_rate": sum(v["agreed"] for v in haiku) / len(haiku),
        "sonnet_yes_rate": sum(v["agreed"] for v in sonnet) / len(sonnet),
        "pass": kappa >= KAPPA_ACCEPT,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.write_text(json.dumps(report, indent=2))
    if kappa < KAPPA_ACCEPT:
        raise RuntimeError(
            f"kappa gate FAIL: {kappa:.3f} < {KAPPA_ACCEPT} — judge not trustworthy on this "
            f"output distribution; do not analyze (plan §6)."
        )
    log.info("kappa = %.3f (PASS)", kappa)
    return report


def run_full_pass(slab_root: Path, *, concurrency: int) -> dict:
    """Judge every panel file under every eval dir; final error audit raises."""
    dirs = enumerate_eval_dirs(slab_root)
    log.info("full pass over %d eval dirs", len(dirs))
    judged, residual_errors = 0, []
    for d in dirs:
        for pf in sorted(d.glob("sycophancy_eval_*.json")):
            out = judge_panel_file(pf, concurrency=concurrency)
            judged += 1
            rows = json.loads(out.read_text())["verdicts"]
            if _n_api_errors(rows):
                residual_errors.append(str(out))
    if residual_errors:
        raise RuntimeError(
            f"{len(residual_errors)} judgments files carry post-retry API errors "
            f"(map to NO, deflate rates): {residual_errors[:5]}"
        )
    summary = {
        "n_eval_dirs": len(dirs),
        "n_panels_judged": judged,
        "eval_dirs": [str(d) for d in dirs],
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (slab_root / "judge_pass_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 P6 — kappa calibration + unified Haiku judge pass (VM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_612"))
    parser.add_argument("--kappa-n", type=int, default=1000)
    parser.add_argument("--judge-concurrency", type=int, default=24)
    parser.add_argument("--skip-kappa", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=p6_judge] %(message)s", stream=sys.stdout
    )
    if not args.skip_kappa:
        run_kappa_calibration(args.slab_root, n=args.kappa_n, concurrency=args.judge_concurrency)
    run_full_pass(args.slab_root, concurrency=args.judge_concurrency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
