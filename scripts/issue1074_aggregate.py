#!/usr/bin/env python
"""#1074 Phase D (VM, 0 GPU): off-pod final judging + aggregation.

Runs AFTER the pod is released (plan §4 Phase D). Stages the driver's
artifacts from the HF data repo (``issue1074_gencompare/``) or reads a local
``--results-root``, judges the final-eval completions via the sanctioned
batch-capable graded judge (``eval.graded_judge.judge_graded`` -> the
#663-hardened ``eval.batch_judge`` client; the Batch API absorbs the large
call volume), and writes the plan's primary deliverables under
``eval_results/issue_1074/``:

- ``yield_summary.json`` — per-cell datagen yield vs floor + drop mix +
  per-variant / per-question yields;
- ``<cell>/install/install_summary.json`` — dose curve (rate vs step),
  band-entry selection, final judged rates at the source + default contexts;
- ``<cell>/margin/margin_summary.json`` — tf-margin per (state, context)
  (secondary DV) for the cell's class;
- ``arm_contrasts.json`` — base-vs-ablit paired question-level bootstrap
  (default 2000 draws) implemented as ONE numpy gather per contrast (a
  ``(draws, n_q)`` index matrix + mean along axis 1 — never a per-draw loop).

Figures are the analyzer's job (/paper-plots); this script emits the JSONs.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps BEFORE numpy/torch-adjacent imports

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402

logger = logging.getLogger("issue1074.aggregate")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_PREFIX = "issue1074_gencompare"
CLASSES = ("sycophancy", "harmful_compliance")
ARMS = ("base", "ablit")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _git_short_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _meta() -> dict:
    return {
        "git_commit": _git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def stage_from_hf(dest: Path) -> Path:
    """Scoped staging of the driver's artifacts (never snapshot_download on the
    ~1M-file data repo — gotchas.md): server-side ``list_repo_tree`` on the
    issue prefix + per-file ``hf_hub_download`` in a <=6-thread pool with
    bounded linear-backoff retries."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    entries = [
        e.path
        for e in api.list_repo_tree(
            HF_DATA_REPO, path_in_repo=DATA_PREFIX, repo_type="dataset", recursive=True
        )
        if getattr(e, "size", None) is not None  # files only
    ]
    if not entries:
        raise RuntimeError(f"no files under {HF_DATA_REPO}/{DATA_PREFIX} — did the pod upload?")
    logger.info("staging %d files from %s/%s", len(entries), HF_DATA_REPO, DATA_PREFIX)

    def _fetch(path: str) -> None:
        for attempt in range(4):
            try:
                hf_hub_download(HF_DATA_REPO, path, repo_type="dataset", local_dir=dest)
                return
            except Exception as e:
                if attempt == 3:
                    raise
                logger.warning("retrying %s (%s)", path, e)
                time.sleep(20 * (attempt + 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_fetch, entries))
    return dest / DATA_PREFIX


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ── Judging ──────────────────────────────────────────────────────────────────


def judge_eval_completions(
    root: Path, out_dir: Path, *, n_judge_draws: int
) -> dict[str, dict[str, dict]]:
    """Judge every completions__{state}__{ctx}.json under evalgen/<behavior>/.

    Returns ``{behavior: {f"{state}__{ctx}": cell}}`` where cell carries the
    binary rate (mean score > threshold), the graded mean, and PER-QUESTION
    rates (the paired-bootstrap unit). Drop-never-coerce: a None mean score
    (all draws dropped) leaves the denominator; counts are reported.
    """
    results: dict[str, dict[str, dict]] = {}
    for behavior in CLASSES:
        beh = BEHAVIORS[behavior]
        beh_dir = root / "evalgen" / behavior
        if not beh_dir.exists():
            continue
        results[behavior] = {}
        for comp_path in sorted(beh_dir.glob("completions__*.json")):
            state, ctx = comp_path.stem.split("__")[1:3]
            payload = _read_json(comp_path)
            questions = payload["questions"]
            completions = payload["completions"]
            items = []
            for i, q in enumerate(questions):
                for j, c in enumerate(completions[i]):
                    items.append((f"{state}-{ctx}-q{i:03d}-c{j}", q, c))
            cell_dir = out_dir / "judge" / behavior / f"{state}__{ctx}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            jr = judge_graded(
                items,
                beh.judge_rubric,
                n_draws=n_judge_draws,
                cache_dir=cell_dir,
                save_raw=cell_dir / "judge_raw.json",
                judge_model=beh.judge_model,
            )
            per_q_pos = np.zeros(len(questions))
            per_q_n = np.zeros(len(questions))
            scored, dropped, graded_sum = 0, 0, 0.0
            for i in range(len(questions)):
                for j in range(len(completions[i])):
                    score = jr.scores.get(f"{state}-{ctx}-q{i:03d}-c{j}")
                    if score is None:
                        dropped += 1
                        continue
                    scored += 1
                    graded_sum += score
                    per_q_n[i] += 1
                    per_q_pos[i] += int(score > beh.threshold)
            if scored == 0:
                raise RuntimeError(f"every completion judge-dropped at {comp_path}")
            with np.errstate(invalid="ignore", divide="ignore"):
                per_q_rate = np.where(per_q_n > 0, per_q_pos / np.maximum(per_q_n, 1), np.nan)
            cell = {
                "state": state,
                "context": ctx,
                "rate": float(per_q_pos.sum() / scored),
                "graded_mean": graded_sum / scored,
                "n_scored": scored,
                "n_dropped": dropped,
                "per_question_rate": [None if np.isnan(x) else float(x) for x in per_q_rate],
                "questions_sha": payload["manifest"]["questions_sha256"],
            }
            results[behavior][f"{state}__{ctx}"] = cell
            _atomic_write_json(
                out_dir / "judge" / behavior / "rates.json",
                {**_meta(), "cells": results[behavior]},
            )
    return results


# ── Bootstrap (ONE gather per contrast) ──────────────────────────────────────


def paired_question_bootstrap(delta_per_q: np.ndarray, *, n_draws: int, seed: int = 42) -> dict:
    """Paired question-level bootstrap of a per-question delta vector: one
    ``(n_draws, n_q)`` integer index matrix + one gather + mean(axis=1)."""
    q = np.asarray(delta_per_q, dtype=float)
    q = q[~np.isnan(q)]
    if q.size == 0:
        return {"mean": None, "ci95": None, "n_questions": 0, "n_draws": n_draws}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, q.size, size=(n_draws, q.size))
    boot = q[idx].mean(axis=1)  # ONE vectorized gather — never a per-draw loop
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": float(q.mean()),
        "ci95": [float(lo), float(hi)],
        "n_questions": int(q.size),
        "n_draws": int(n_draws),
        "seed": seed,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────


def build_yield_summary(root: Path) -> dict:
    cells = {}
    for behavior in CLASSES:
        for arm in ARMS:
            slug = f"{behavior}-{arm}"
            p = root / slug / "datagen_summary.json"
            if p.exists():
                cells[slug] = _read_json(p)
    return {**_meta(), "cells": cells}


def build_install_summaries(root: Path, rates: dict, out_dir: Path) -> None:
    for behavior in CLASSES:
        for arm in ARMS:
            slug = f"{behavior}-{arm}"
            build_path = root / slug / "build_result.json"
            if not build_path.exists():
                continue
            build = _read_json(build_path)
            prov = build.get("provenance", {})
            beh_rates = rates.get(behavior, {})
            src_ctx = "persona_software_engineer"
            summary = {
                **_meta(),
                "cell": slug,
                "dose_curve_rates_by_step": prov.get("rates_by_step"),
                "band_entry": build.get("selection"),
                "steps_to_band": (build.get("selection") or {}).get("step"),
                "final_rate_source": (beh_rates.get(f"{slug}__{src_ctx}") or {}).get("rate"),
                "final_graded_mean_source": (beh_rates.get(f"{slug}__{src_ctx}") or {}).get(
                    "graded_mean"
                ),
                "base_rate_source": (beh_rates.get(f"base__{src_ctx}") or {}).get("rate"),
                "default_ctx_rate": (beh_rates.get(f"{slug}__neg_default_assistant") or {}).get(
                    "rate"
                ),
                "base_default_ctx_rate": (beh_rates.get("base__neg_default_assistant") or {}).get(
                    "rate"
                ),
            }
            _atomic_write_json(out_dir / slug / "install" / "install_summary.json", summary)


def build_margin_summaries(root: Path, out_dir: Path) -> None:
    for behavior in CLASSES:
        p = root / "margin" / f"{behavior}.json"
        if not p.exists():
            continue
        margins = _read_json(p)
        for arm in ARMS:
            slug = f"{behavior}-{arm}"
            if not (root / slug).exists():
                continue
            cell_view = {
                **_meta(),
                "cell": slug,
                "status": margins.get("status"),
                "pool_source_cell": margins.get("pool_source_cell"),
                "n_pos": margins.get("n_pos"),
                "n_neg": margins.get("n_neg"),
                # This cell's states + the shared base state, all contexts.
                "cells": {
                    k: v
                    for k, v in (margins.get("cells") or {}).items()
                    if k.startswith((f"{slug}__", "base__"))
                },
            }
            _atomic_write_json(out_dir / slug / "margin" / "margin_summary.json", cell_view)


def build_arm_contrasts(root: Path, rates: dict, *, n_bootstrap: int) -> dict:
    """S3: paired question-level bootstrap CIs for base-vs-ablit contrasts."""
    contrasts: dict[str, dict] = {}
    src_ctx = "persona_software_engineer"
    for behavior in CLASSES:
        beh_rates = rates.get(behavior, {})
        entry: dict[str, dict] = {}
        # Δrate at the selected checkpoints (range-restricted by band selection).
        a = beh_rates.get(f"{behavior}-ablit__{src_ctx}")
        b = beh_rates.get(f"{behavior}-base__{src_ctx}")
        if a and b:
            qa = np.array([np.nan if x is None else x for x in a["per_question_rate"]])
            qb = np.array([np.nan if x is None else x for x in b["per_question_rate"]])
            n = min(qa.size, qb.size)
            entry["delta_rate_at_band_entry"] = paired_question_bootstrap(
                qa[:n] - qb[:n], n_draws=n_bootstrap
            )
        # Δyield per question (kept fraction), from the datagen summaries.
        per_q: dict[str, np.ndarray] = {}
        for arm in ARMS:
            p = root / f"{behavior}-{arm}" / "datagen_summary.json"
            if not p.exists():
                continue
            pq = _read_json(p).get("per_question_yield") or {}
            if not pq:
                continue
            qids = sorted(pq)
            per_q[arm] = np.array(
                [pq[q]["kept"] / pq[q]["judged"] if pq[q]["judged"] else np.nan for q in qids]
            )
        if "base" in per_q and "ablit" in per_q:
            n = min(per_q["base"].size, per_q["ablit"].size)
            entry["delta_yield_per_question"] = paired_question_bootstrap(
                per_q["ablit"][:n] - per_q["base"][:n], n_draws=n_bootstrap
            )
        if entry:
            contrasts[behavior] = entry
    return {**_meta(), "n_bootstrap": n_bootstrap, "contrasts": contrasts}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="#1074 Phase D aggregation (VM, 0 GPU)")
    p.add_argument("--results-root", default=None, help="local driver out_root; None -> stage HF")
    p.add_argument("--stage-dir", default="data/issue_1074/agg_stage")
    p.add_argument("--out-dir", default="eval_results/issue_1074")
    p.add_argument("--n-judge-draws", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    args = p.parse_args(argv)

    root = Path(args.results_root) if args.results_root else stage_from_hf(Path(args.stage_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[phase=judge] judging final-eval completions under %s", root)
    rates = judge_eval_completions(root, out_dir, n_judge_draws=args.n_judge_draws)

    logger.info("[phase=aggregate] building summaries")
    _atomic_write_json(out_dir / "yield_summary.json", build_yield_summary(root))
    build_install_summaries(root, rates, out_dir)
    build_margin_summaries(root, out_dir)
    _atomic_write_json(
        out_dir / "arm_contrasts.json",
        build_arm_contrasts(root, rates, n_bootstrap=args.n_bootstrap),
    )
    logger.info("aggregation complete -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
