#!/usr/bin/env python
"""#1074 Phase D (VM, 0 GPU): off-pod final judging + aggregation.

Runs AFTER the pod is released (plan §4 Phase D). Stages the driver's
artifacts from the HF data repo (``issue1074_gencompare/``) or reads a local
``--results-root``, judges the final-eval completions (resolved on BOTH
layouts — the HF-staged ``raw_completions/final/<behavior>/`` tree the driver
uploads, or the local ``evalgen/<behavior>/`` out_root; fail-loud when trained
cells exist but no completions resolve) via the sanctioned batch-capable
graded judge (``eval.graded_judge.judge_graded`` -> the #663-hardened
``eval.batch_judge`` client; the Batch API absorbs the large call volume), and
writes the plan's primary deliverables under ``eval_results/issue_1074/``:

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
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.negatives import default_panel  # noqa: E402
from explore_persona_space.eval.graded_judge import (  # noqa: E402
    _score_from_parsed,
    judge_graded,
)

logger = logging.getLogger("issue1074.aggregate")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_PREFIX = "issue1074_gencompare"
CLASSES = ("sycophancy", "harmful_compliance")
ARMS = ("base", "ablit")
SRC_CTX = "persona_software_engineer"

# Follow-up round `base-negatives-regen` (plan v7): one mixed-generator cell
# (reused ablit positives, fresh base negatives) vs the parent ablit cell.
LABEL_BASE_NEG_REGEN = "base-negatives-regen"
# Follow-up round `install-dose-extension` (plan v9): the mixed cell retrained
# on its byte-pinned mix at a 9-epoch ceiling; NO datagen — the Phase-D
# additions are the drop-censoring telemetry + the schedule-stretch overlay.
LABEL_DOSE_EXTENSION = "install-dose-extension"
FOLLOWUP_LABELS = (LABEL_BASE_NEG_REGEN, LABEL_DOSE_EXTENSION)
FOLLOWUP_CELL = "harmful_compliance-mixed"
FOLLOWUP_BEHAVIOR = "harmful_compliance"
DOSE_EXT_SUFFIX = "-e9"  # rate raw-completions dir suffix (driver phase_upload)
# The prior round's COMMITTED dose curve (the plan §7 overlay baseline).
DOSE_PRIOR_INSTALL_SUMMARY = Path(
    f"eval_results/issue_1074/{LABEL_BASE_NEG_REGEN}/install/install_summary.json"
)
PARENT_CELL = "harmful_compliance-ablit"
MEMBER_QUOTA = 24  # per-member negative quota (floor_n 120 / 5 members; plan §3 S1')
MEMBER_BUDGET = 35  # negative requests per member (ceil(24 / EXPECTED_YIELD 0.7))
# Fix 4 (r1 carry-forward): the parent side-by-side reads the PINNED dataset-repo
# revision, never HEAD. Keep == issue1074_generator_compare.PARENT_PIN_REVISION
# (drift-guarded by tests/test_issue1074_base_negatives_regen.py).
PARENT_PIN_REVISION = "c1f526c1"
PARENT_PINNED_FILES = ("raw_neg.jsonl", "judge_rows.jsonl")
# The 5 default_v1 panel member slugs — the S1' denominator (plan §3: floor_n 120 / 5).
EXPECTED_MEMBERS = tuple(n.slug for n in default_panel())


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
    bounded linear-backoff retries (the listing itself included — hub pagination
    retries only 429 on FOLLOW-UP cursor pages, so a first-page 5xx raises)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()

    def _list_entries() -> list[str]:
        return [
            e.path
            for e in api.list_repo_tree(
                HF_DATA_REPO, path_in_repo=DATA_PREFIX, repo_type="dataset", recursive=True
            )
            if getattr(e, "size", None) is not None  # files only
        ]

    entries: list[str] = []
    for attempt in range(4):  # same bounded retry as the per-file fetches below
        try:
            entries = _list_entries()
            break
        except Exception as e:
            if attempt == 3:
                raise
            logger.warning("retrying list_repo_tree %s/%s (%s)", HF_DATA_REPO, DATA_PREFIX, e)
            time.sleep(20 * (attempt + 1))
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


def _read_jsonl(path: Path) -> list[dict]:
    # Text-mode iteration, never splitlines() (gotchas.md U+2028 JSONL shred).
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _final_completions_dir(root: Path, behavior: str) -> Path | None:
    """Resolve the final-eval completions dir on BOTH layouts the aggregate sees.

    The pod driver uploads its local ``out_root/evalgen`` tree to
    ``{DATA_PREFIX}/raw_completions/final`` (``phase_upload`` in
    ``issue1074_generator_compare.py``), so the HF-STAGED tree — the production
    Phase-D default after pod termination — carries
    ``root/raw_completions/final/<behavior>/completions__*.json``. A local
    ``--results-root`` (the driver's out_root itself) instead carries
    ``root/evalgen/<behavior>/``. Staged layout wins when both exist.
    """
    staged = root / "raw_completions" / "final" / behavior
    if staged.exists():
        return staged
    local = root / "evalgen" / behavior
    if local.exists():
        return local
    return None


def _trained_cells(root: Path) -> dict[str, list[str]]:
    """behavior -> [cell slug] whose build_result.json records status=="trained"."""
    out: dict[str, list[str]] = {}
    for behavior in CLASSES:
        for arm in ARMS:
            slug = f"{behavior}-{arm}"
            p = root / slug / "build_result.json"
            if p.exists() and _read_json(p).get("status") == "trained":
                out.setdefault(behavior, []).append(slug)
    return out


# ── Judging ──────────────────────────────────────────────────────────────────


def judge_eval_completions(
    root: Path, out_dir: Path, *, n_judge_draws: int
) -> dict[str, dict[str, dict]]:
    """Judge every completions__{state}__{ctx}.json for each behavior, resolved
    via ``_final_completions_dir`` (HF-staged ``raw_completions/final/`` first,
    local ``evalgen/`` fallback).

    FAIL LOUD when trained cells exist for a behavior but no completion files
    resolve — an empty-rates Phase D must never exit 0 (r1 Critical: the
    aggregate read only ``evalgen/`` and silently judged zero completions on
    the staged tree). A behavior with NO trained cells (the K1 all-floor path)
    legitimately has no completions and is skipped.

    Returns ``{behavior: {f"{state}__{ctx}": cell}}`` where cell carries the
    binary rate (mean score > threshold), the graded mean, and PER-QUESTION
    rates (the paired-bootstrap unit). Drop-never-coerce: a None mean score
    (all draws dropped) leaves the denominator; counts are reported.
    """
    results: dict[str, dict[str, dict]] = {}
    trained = _trained_cells(root)
    for behavior in CLASSES:
        beh = BEHAVIORS[behavior]
        beh_dir = _final_completions_dir(root, behavior)
        comp_paths = sorted(beh_dir.glob("completions__*.json")) if beh_dir is not None else []
        if not comp_paths:
            if trained.get(behavior):
                raise RuntimeError(
                    f"trained cells {trained[behavior]} exist for {behavior!r} but no "
                    f"completions__*.json resolved under "
                    f"{root / 'raw_completions' / 'final' / behavior} or "
                    f"{root / 'evalgen' / behavior} — upload-map/read-path mismatch or a "
                    "missing upload; refusing to ship null install rates"
                )
            continue  # no trained cells for this behavior (K1 path): nothing to judge
        results[behavior] = _judge_completion_files(
            comp_paths, beh, out_dir / "judge" / behavior, n_judge_draws=n_judge_draws
        )
    return results


def _judge_completion_files(
    comp_paths: list[Path], beh, judge_dir: Path, *, n_judge_draws: int
) -> dict[str, dict]:
    """Judge ``completions__{state}__{ctx}.json`` files -> ``{state__ctx: cell}``
    (binary rate, graded mean, per-question rates; drop-never-coerce)."""
    cells: dict[str, dict] = {}
    for comp_path in comp_paths:
        state, ctx = comp_path.stem.split("__")[1:3]
        payload = _read_json(comp_path)
        questions = payload["questions"]
        completions = payload["completions"]
        items = []
        for i, q in enumerate(questions):
            for j, c in enumerate(completions[i]):
                # Compact per-cell item id: the judge call is scoped to ONE
                # (state, ctx) cell (cache_dir/save_raw per cell), and the Batch
                # API custom_id encoder appends "__NNNNN__NN" (11 chars) to a
                # 64-char-max id — a cell-slug-prefixed id overflowed it (run-2
                # Phase D 400: 58+11=69 > 64). q/c indices are unique in-cell.
                items.append((f"q{i:03d}-c{j}", q, c))
        cell_dir = judge_dir / f"{state}__{ctx}"
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
                score = jr.scores.get(f"q{i:03d}-c{j}")
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
        cells[f"{state}__{ctx}"] = {
            "state": state,
            "context": ctx,
            "rate": float(per_q_pos.sum() / scored),
            "graded_mean": graded_sum / scored,
            "n_scored": scored,
            "n_dropped": dropped,
            "per_question_rate": [None if np.isnan(x) else float(x) for x in per_q_rate],
            "questions_sha": payload["manifest"]["questions_sha256"],
        }
        _atomic_write_json(judge_dir / "rates.json", {**_meta(), "cells": cells})
    return cells


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


def _install_summary_fields(build: dict, beh_rates: dict, slug: str) -> dict:
    """The install-summary field mapping shared by the parent + followup paths."""
    prov = build.get("provenance", {})
    return {
        **_meta(),
        "cell": slug,
        "dose_curve_rates_by_step": prov.get("rates_by_step"),
        "band_entry": build.get("selection"),
        "steps_to_band": (build.get("selection") or {}).get("step"),
        "final_rate_source": (beh_rates.get(f"{slug}__{SRC_CTX}") or {}).get("rate"),
        "final_graded_mean_source": (beh_rates.get(f"{slug}__{SRC_CTX}") or {}).get("graded_mean"),
        "base_rate_source": (beh_rates.get(f"base__{SRC_CTX}") or {}).get("rate"),
        "default_ctx_rate": (beh_rates.get(f"{slug}__neg_default_assistant") or {}).get("rate"),
        "base_default_ctx_rate": (beh_rates.get("base__neg_default_assistant") or {}).get("rate"),
    }


def build_install_summaries(root: Path, rates: dict, out_dir: Path) -> None:
    for behavior in CLASSES:
        for arm in ARMS:
            slug = f"{behavior}-{arm}"
            build_path = root / slug / "build_result.json"
            if not build_path.exists():
                continue
            summary = _install_summary_fields(_read_json(build_path), rates.get(behavior, {}), slug)
            _atomic_write_json(out_dir / slug / "install" / "install_summary.json", summary)


def _margin_cell_view(margins: dict, slug: str) -> dict:
    """One cell's view of a behavior margin file (its states + shared base).
    ``pool_provenance`` passes through when present (the dose-extension
    held-out pools; None/absent on every other round)."""
    return {
        **_meta(),
        "cell": slug,
        "status": margins.get("status"),
        "pool_source_cell": margins.get("pool_source_cell"),
        "pool_provenance": margins.get("pool_provenance"),
        "n_pos": margins.get("n_pos"),
        "n_neg": margins.get("n_neg"),
        "cells": {
            k: v
            for k, v in (margins.get("cells") or {}).items()
            if k.startswith((f"{slug}__", "base__"))
        },
    }


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
            _atomic_write_json(
                out_dir / slug / "margin" / "margin_summary.json",
                _margin_cell_view(margins, slug),
            )


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
        # Δyield per question (kept fraction), from the datagen summaries —
        # paired on the INTERSECTION of question ids (index-aligned truncation
        # of independently-sorted key lists mis-pairs questions whenever the
        # arms' judged sets diverge; r1 minor).
        per_q: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            p = root / f"{behavior}-{arm}" / "datagen_summary.json"
            if not p.exists():
                continue
            pq = _read_json(p).get("per_question_yield") or {}
            if not pq:
                continue
            per_q[arm] = {
                q: (v["kept"] / v["judged"] if v["judged"] else np.nan) for q, v in pq.items()
            }
        if "base" in per_q and "ablit" in per_q:
            shared = sorted(per_q["base"].keys() & per_q["ablit"].keys())
            if shared:
                delta = np.array([per_q["ablit"][q] - per_q["base"][q] for q in shared])
                entry["delta_yield_per_question"] = {
                    **paired_question_bootstrap(delta, n_draws=n_bootstrap),
                    "n_shared_questions": len(shared),
                }
        if entry:
            contrasts[behavior] = entry
    return {**_meta(), "n_bootstrap": n_bootstrap, "contrasts": contrasts}


# ── Follow-up base-negatives-regen (plan v7) ─────────────────────────────────


def _member_of(variant_id: str) -> str:
    """Panel-member slug from a negative variant_id (``<slug>`` or ``<slug>-nvK``)."""
    return variant_id.split("-nv")[0]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial 95% CI on k successes of n."""
    from scipy.stats import beta

    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _load_raw_negative_rows(raw_path: Path, members: dict, blank) -> dict[str, dict]:
    """Load raw_neg.jsonl's negative rows into rid -> identity fields (+ the
    generated flag), accumulating requested/generated/gen-drop counts into
    ``members``. Duplicate request_ids raise."""
    raw_rows: dict[str, dict] = {}
    for r in _read_jsonl(raw_path):
        if r.get("arm") != "negative":
            continue
        rid = r["request_id"]
        if rid in raw_rows:
            raise RuntimeError(f"negative_yield_table: duplicate request_id {rid!r} in {raw_path}")
        raw_rows[rid] = {
            "arm": r.get("arm"),
            "question_id": r.get("question_id"),
            "variant_id": r["variant_id"],
            "generated": r.get("completion") is not None,
        }
        m = members.setdefault(_member_of(r["variant_id"]), blank())
        m["requested"] += 1
        if r.get("completion") is None:
            reason = r.get("drop_reason") or "refusal"
            m["gen_drop_mix"][reason] = m["gen_drop_mix"].get(reason, 0) + 1
        else:
            m["generated"] += 1
    return raw_rows


def negative_yield_table(datagen_dir: Path) -> dict[str, dict]:
    """Per-panel-member negative-stage yield from a cell's datagen artifacts:
    requested + gen-stage drop mix per member (raw_neg.jsonl, REQUIRED — the
    per-member request schedule IS the denominator), judged/kept + judge-drop
    counts per member (judge_rows.jsonl, the authoritative kept flags), kept
    COUNT vs the 24-per-member quota (the operative gate — plan §6 S1'), and
    an exact Clopper-Pearson 95% CI on kept/requested. Judge rows are JOINED
    to raw rows by request_id; missing/extra/duplicate ids raise (r1
    carry-forward fix 3 — no judged-as-denominator fallback)."""
    rows_path = datagen_dir / "judge_rows.jsonl"
    if not rows_path.exists():
        raise RuntimeError(f"negative_yield_table: {rows_path} missing — negatives never judged")
    raw_path = datagen_dir / "raw_neg.jsonl"
    if not raw_path.exists():
        raise RuntimeError(
            f"negative_yield_table: {raw_path} missing — the per-member requested "
            "denominator is the raw request schedule; the judged-rows fallback is removed"
        )

    def _blank() -> dict:
        return {
            "requested": 0,
            "generated": 0,
            "judged": 0,
            "kept": 0,
            "gen_drop_mix": {},
            "judge_none_drops": 0,
        }

    members: dict[str, dict] = {}
    # negative request_id -> the RAW row's identity fields + generated flag.
    # The raw schedule is the join's source of truth (r2 hardening): judge rows
    # must MATCH it field-for-field, and member attribution for judged/kept
    # derives from the RAW row after validation — protection against cross-run
    # file mixing (a judge_rows.jsonl from one attempt joined with another
    # attempt's raw_neg.jsonl, which --resume-partial-attempt makes possible).
    raw_rows = _load_raw_negative_rows(raw_path, members, _blank)
    judged_ids: set[str] = set()
    for r in _read_jsonl(rows_path):
        if r["arm"] != "negative":
            continue
        rid = r["request_id"]
        if rid in judged_ids:
            raise RuntimeError(f"negative_yield_table: duplicate request_id {rid!r} in {rows_path}")
        raw = raw_rows.get(rid)
        if raw is None or not raw["generated"]:
            raise RuntimeError(
                f"negative_yield_table: judge row {rid!r} has no generated raw_neg row "
                f"(extra/orphan id — raw/judge join broken under {datagen_dir})"
            )
        for field in ("arm", "question_id", "variant_id"):
            if r.get(field) != raw[field]:
                raise RuntimeError(
                    f"negative_yield_table: judge row {rid!r} field {field!r} mismatch "
                    f"(judge={r.get(field)!r} raw={raw[field]!r}) — cross-run file mixing "
                    f"under {datagen_dir}"
                )
        judged_ids.add(rid)
        m = members.setdefault(_member_of(raw["variant_id"]), _blank())
        m["judged"] += 1
        m["kept"] += int(bool(r["kept"]))
        if r["mean"] is None:
            m["judge_none_drops"] += 1
    never_judged = {rid for rid, raw in raw_rows.items() if raw["generated"]} - judged_ids
    if never_judged:
        preview = ", ".join(sorted(never_judged)[:5])
        raise RuntimeError(
            f"negative_yield_table: {len(never_judged)} generated negative rows have no "
            f"judge row (first ids: {preview}) — raw/judge join broken under {datagen_dir}"
        )
    for m in members.values():
        n = m["requested"]  # the per-member request budget (fix 3: never judged-count)
        lo, hi = clopper_pearson(m["kept"], n)
        m["n_denominator"] = n
        m["kept_rate"] = (m["kept"] / n) if n else None
        m["kept_rate_ci95"] = [lo, hi]
        m["quota"] = MEMBER_QUOTA
        m["budget"] = MEMBER_BUDGET
        m["meets_quota"] = m["kept"] >= MEMBER_QUOTA
    return members


def stage_parent_pinned(dest: Path, *, fetch_fn=None) -> Path:
    """Stage the PARENT ablit cell's datagen files at the PINNED dataset-repo
    revision (r1 carry-forward fix 4: the plan §6.5 side-by-side must read the
    exact bytes the plan verified — never HEAD, which later uploads can move).
    ``fetch_fn(path_in_repo, local_dir, revision)`` is the injectable hub
    boundary (tests assert the revision it receives); the default is a
    revision-pinned per-file ``hf_hub_download`` with the same bounded
    linear-backoff retry as ``stage_from_hf`` (never snapshot_download on the
    ~1M-file data repo — gotchas.md). The pinned fetch ALWAYS runs (r2
    hardening): a pre-existing nonempty local file under ``dest`` is never
    trusted as pinned. Fail-loud on any missing/empty staged file. Returns
    the staged datagen dir ``negative_yield_table`` consumes."""
    if fetch_fn is None:

        def fetch_fn(path_in_repo: str, local_dir: Path, revision: str) -> str:
            from huggingface_hub import hf_hub_download

            for attempt in range(4):
                try:
                    return hf_hub_download(
                        HF_DATA_REPO,
                        path_in_repo,
                        repo_type="dataset",
                        revision=revision,
                        local_dir=local_dir,
                    )
                except Exception as e:
                    if attempt == 3:
                        raise
                    logger.warning("retrying pinned fetch %s (%s)", path_in_repo, e)
                    time.sleep(20 * (attempt + 1))
            raise AssertionError("unreachable")

    dest.mkdir(parents=True, exist_ok=True)
    for fname in PARENT_PINNED_FILES:
        rel = f"{DATA_PREFIX}/{PARENT_CELL}/datagen/{fname}"
        local = dest / rel  # hf_hub_download(local_dir=...) preserves the repo-relative path
        # ALWAYS fetch at the pin (r2 hardening): a pre-existing nonempty local
        # file is NEVER trusted as pinned — a reused --stage-dir can hold bytes
        # from another run/revision (cross-run mixing, which the driver's
        # --resume-partial-attempt staging makes plausible), and nothing ties
        # such a file to PARENT_PIN_REVISION. hf_hub_download's local cache
        # makes the re-fetch idempotent; a wrong pre-existing file is
        # OVERWRITTEN with the pinned bytes.
        got = Path(fetch_fn(rel, dest, PARENT_PIN_REVISION))
        if got.resolve() != local.resolve():
            local.parent.mkdir(parents=True, exist_ok=True)
            os.replace(got, local)
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(
                f"pinned parent staging failed for {rel!r} @ {PARENT_PIN_REVISION} -> {local}"
            )
    return dest / DATA_PREFIX / PARENT_CELL / "datagen"


def _assert_member_coverage(members: dict[str, dict]) -> None:
    """r1 carry-forward fix 2: S1' is only meaningful over the FULL default
    panel at the full per-member request budget — a silently-shrunk panel or a
    short request schedule would let ``s1_prime_pass`` read True over the
    wrong denominator. Raise on any missing/extra member or any member whose
    requested count != MEMBER_BUDGET."""
    expected, got = set(EXPECTED_MEMBERS), set(members)
    if got != expected:
        raise RuntimeError(
            "negative panel member mismatch before s1_prime_pass: "
            f"missing={sorted(expected - got)} extra={sorted(got - expected)}"
        )
    bad = {
        m: members[m]["requested"]
        for m in sorted(members)
        if members[m]["requested"] != MEMBER_BUDGET
    }
    if bad:
        raise RuntimeError(
            f"per-member negative requests != budget {MEMBER_BUDGET}: {bad} — "
            "the S1' denominator is broken; refusing to compute s1_prime_pass"
        )


def _followup_run_path(root: Path, label: str, *parts: str) -> Path | None:
    """Resolve a followup RUN-LEVEL artifact on both layouts: the HF-staged
    tree (``root/followups/<label>/...``) first, the local driver out_root
    (``root/...``) as fallback. Returns None when neither exists."""
    for base in (root / "followups" / label, root):
        p = base.joinpath(*parts)
        if p.exists():
            return p
    return None


# ── Follow-up install-dose-extension (plan v9) ───────────────────────────────


def _rate_ckpt_dirs(root: Path) -> list[Path]:
    """The dose ladder's per-checkpoint rate dirs on BOTH layouts: the
    HF-staged tree (``raw_completions/rate/<cell>-e9/rate_checkpoint-*``,
    the driver's -e9 upload convention) first, the local driver out_root
    (``<cell>/rate/rate_checkpoint-*``) as fallback."""
    for base in (
        root / "raw_completions" / "rate" / f"{FOLLOWUP_CELL}{DOSE_EXT_SUFFIX}",
        root / FOLLOWUP_CELL / "rate",
    ):
        dirs = sorted(base.glob("rate_checkpoint-*")) if base.exists() else []
        if dirs:
            return dirs
    return []


def dose_rate_drop_censoring(root: Path) -> dict:
    """Per-checkpoint judge-drop telemetry for the dose ladder (plan §7
    drop-censoring guard): rising no-scores at higher install would censor the
    strongest completions and could manufacture a plateau shape — check
    drop-rate-vs-step BEFORE reading K-dose as a real plateau.

    Recomputed from each ``rate_checkpoint-<step>`` dir's persisted
    ``judge_raw.json`` (``all_scores``: ``{item}__{idx:05d}__{draw:02d}`` ->
    parsed draw; drop-never-coerce via ``_score_from_parsed``) + the
    completions file's question list (realized question-n).
    """
    per_step: dict[str, dict] = {}
    for ckpt_dir in _rate_ckpt_dirs(root):
        step_str = ckpt_dir.name.rsplit("-", 1)[-1]
        if not step_str.isdigit():
            continue
        comp_paths = sorted(ckpt_dir.glob("completions__trained__*.json"))
        raw_paths = sorted(ckpt_dir.glob("judge/trained_*/judge_raw.json"))
        entry: dict = {"rate_dir": str(ckpt_dir)}
        if comp_paths:
            payload = _read_json(comp_paths[0])
            entry["n_questions"] = len(payload.get("questions", []))
            entry["n_completions_per_question"] = (
                len(payload["completions"][0]) if payload.get("completions") else 0
            )
        if raw_paths:
            all_scores = _read_json(raw_paths[0]).get("all_scores", {})
            kept_draws_by_item: dict[str, int] = {}
            n_total_draws = n_dropped_draws = 0
            for cid, parsed in all_scores.items():
                item = cid.rsplit("__", 2)[0]
                kept_draws_by_item.setdefault(item, 0)
                n_total_draws += 1
                if _score_from_parsed(parsed) is None:
                    n_dropped_draws += 1
                else:
                    kept_draws_by_item[item] += 1
            entry.update(
                n_items=len(kept_draws_by_item),
                n_scored=sum(1 for k in kept_draws_by_item.values() if k > 0),
                n_dropped=sum(1 for k in kept_draws_by_item.values() if k == 0),
                n_total_draws=n_total_draws,
                n_dropped_draws=n_dropped_draws,
            )
        else:
            entry["status"] = "no judge_raw.json found"
        per_step[step_str] = entry
    if not per_step:
        logger.warning("[dose-ext] no rate_checkpoint-* dirs found under %s", root)
    return per_step


def dose_overlay(rates_by_step: dict | None, prior_summary_path: Path) -> dict:
    """Schedule-stretch overlay read (plan §7): this round's 30-q subset rates
    at the prior round's committed steps, side by side (overlap => the
    schedule contribution is negligible; elevation => report any S-dose as
    dose+schedule). Both curves are the SAME seeded subset + judge recipe."""
    if not prior_summary_path.exists():
        raise RuntimeError(
            f"[dose-ext] prior install summary missing: {prior_summary_path} — the "
            "schedule-stretch overlay is an always-reported read (plan v9 §7); "
            "refusing a partial aggregate"
        )
    prior = _read_json(prior_summary_path).get("dose_curve_rates_by_step") or {}
    this_round = dict(rates_by_step or {})
    shared = sorted(set(prior) & set(this_round), key=int)
    return {
        "status": "computed",
        "prior_path": str(prior_summary_path),
        "prior_round_rates_by_step": prior,
        "this_round_rates_by_step": this_round,
        "shared_steps": shared,
        "delta_at_shared_steps": {s: this_round[s] - prior[s] for s in shared},
    }


def run_followup_dose_extension(args, root: Path, out_dir: Path) -> int:
    """Phase-D aggregation for the ``install-dose-extension`` round: judge the
    final-eval completions (195-q bank x {trained, base} x {source, default}),
    then the install summary EXTENDED with the plan §7 ensemble folds — the
    per-checkpoint + per-final-cell drop-censoring telemetry and the
    schedule-stretch overlay vs the prior round's committed curve. NO
    negative-yield / calibration blocks (this round has no datagen)."""
    label = LABEL_DOSE_EXTENSION
    build_path = _followup_run_path(root, label, FOLLOWUP_CELL, "build_result.json")
    if build_path is None:
        raise RuntimeError(
            f"no build_result.json for {FOLLOWUP_CELL} under {root} (or "
            f"{root}/followups/{label}) — the dose-extension round trains "
            "unconditionally, so a missing build is an upload/read-path mismatch"
        )
    build = _read_json(build_path)
    if build.get("status") != "trained":
        raise RuntimeError(
            f"build_result.json status={build.get('status')!r} != 'trained' — the "
            "dose-extension round has no K1 yield path; refusing a partial aggregate"
        )

    logger.info("[phase=judge] judging dose-extension final-eval completions")
    beh_dir = None
    for cand in (
        _followup_run_path(root, label, "raw_completions", "final", FOLLOWUP_BEHAVIOR),
        _followup_run_path(root, label, "evalgen", FOLLOWUP_BEHAVIOR),
    ):
        if cand is not None:
            beh_dir = cand
            break
    comp_paths = sorted(beh_dir.glob("completions__*.json")) if beh_dir is not None else []
    if not comp_paths:
        raise RuntimeError(
            f"trained dose-extension cell exists but no completions__*.json resolved "
            f"under {root}/followups/{label}/{{raw_completions/final,evalgen}}/"
            f"{FOLLOWUP_BEHAVIOR} — upload-map/read-path mismatch; refusing to ship "
            "null install rates"
        )
    rates = _judge_completion_files(
        comp_paths,
        BEHAVIORS[FOLLOWUP_BEHAVIOR],
        out_dir / "judge" / FOLLOWUP_BEHAVIOR,
        n_judge_draws=args.n_judge_draws,
    )

    summary = _install_summary_fields(build, rates, FOLLOWUP_CELL)
    # Plan §7 ensemble folds: (a) drop-censoring telemetry — per-checkpoint
    # (the dose ladder's rate reads) AND per final (state, ctx) cell counts +
    # realized question-n; (b) the schedule-stretch overlay.
    summary["drop_censoring"] = {
        "per_checkpoint": dose_rate_drop_censoring(root),
        "final_cells": {
            key: {
                "n_scored": cell["n_scored"],
                "n_dropped": cell["n_dropped"],
                "n_questions": len(cell["per_question_rate"]),
            }
            for key, cell in rates.items()
        },
    }
    rates_by_step = (build.get("provenance") or {}).get("rates_by_step")
    summary["overlay"] = dose_overlay(rates_by_step, Path(args.prior_install_summary))
    _atomic_write_json(out_dir / "install" / "install_summary.json", summary)

    margin_path = _followup_run_path(root, label, "margin", f"{FOLLOWUP_BEHAVIOR}.json")
    if margin_path is not None:
        _atomic_write_json(
            out_dir / "margin" / "margin_summary.json",
            _margin_cell_view(_read_json(margin_path), FOLLOWUP_CELL),
        )
    else:
        logger.warning("[dose-ext] margin file not found — margin summary omitted")
    logger.info("dose-extension aggregation complete -> %s", out_dir)
    return 0


def run_followup(args) -> int:
    """Phase-D aggregation for the ``base-negatives-regen`` round: the
    per-member negative-yield table (mixed cell + parent ablit side-by-side —
    parent staged at the PINNED revision and REQUIRED on the HF-staged
    production path; ``parent_ablit: null`` is allowed ONLY under an explicit
    ``--results-root`` local/smoke root, marked via ``parent_sidebyside_mode``
    — exact binomial CIs, S1' verdict over the asserted full member panel),
    the judge-drift calibration copy (WARN on ``status == "error"``), and —
    conditional on a trained cell — the install + margin summaries. Artifacts
    land under ``<out-dir>/<label>/`` (plan §6.5 globs)."""
    label = args.followup
    staged_from_hf = args.results_root is None
    root = Path(args.results_root) if args.results_root else stage_from_hf(Path(args.stage_dir))
    out_dir = Path(args.out_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    if label == LABEL_DOSE_EXTENSION:
        return run_followup_dose_extension(args, root, out_dir)

    logger.info("[phase=negative_yield] per-member yield table under %s", root)
    mixed_dir = root / FOLLOWUP_CELL / "datagen"
    mixed = negative_yield_table(mixed_dir)
    _assert_member_coverage(mixed)  # fix 2: full panel at full budget, or raise
    if staged_from_hf:
        # Production Phase D (fixes 1 + 4): the parent side-by-side is a plan
        # §6.5 deliverable — staged at the PINNED revision, REQUIRED; any
        # staging/read failure raises (never a warn+null fallback).
        parent_dir = stage_parent_pinned(Path(args.stage_dir) / "parent_pinned")
        parent = negative_yield_table(parent_dir)
        parent_mode = f"hf-pinned@{PARENT_PIN_REVISION}"
    else:
        parent_dir = root / PARENT_CELL / "datagen"
        if (parent_dir / "judge_rows.jsonl").exists():
            parent = negative_yield_table(parent_dir)
            parent_mode = "local"
        else:
            parent = None
            parent_mode = "smoke-local"
            logger.warning(
                "[negative-yield] parent cell %s not present under local root %s — "
                "side-by-side omitted (allowed ONLY on an explicit --results-root "
                "local/smoke root; the HF-staged production path stages it pinned "
                "and fails loud)",
                PARENT_CELL,
                root,
            )
    payload = {
        **_meta(),
        "quota": MEMBER_QUOTA,
        "budget": MEMBER_BUDGET,
        "parent_sidebyside_mode": parent_mode,
        "mixed": mixed,
        "parent_ablit": parent,
        "s1_prime_pass": all(m["meets_quota"] for m in mixed.values()),
        "delta_kept_rate_by_member": (
            {
                slug: (
                    mixed[slug]["kept_rate"] - parent[slug]["kept_rate"]
                    if slug in parent
                    and mixed[slug]["kept_rate"] is not None
                    and parent[slug]["kept_rate"] is not None
                    else None
                )
                for slug in sorted(mixed)
            }
            if parent
            else None
        ),
    }
    _atomic_write_json(out_dir / "negative_yield.json", payload)

    cal = _followup_run_path(root, label, "judge_calibration.json")
    if cal is not None:
        cal_payload = _read_json(cal)
        if cal_payload.get("status") == "error":
            logger.warning(
                "[followup] judge_calibration.json carries status=error (%s) — copying "
                "as-is; the judge-drift diagnostic never computed",
                cal_payload.get("error"),
            )
        _atomic_write_json(out_dir / "judge_calibration.json", cal_payload)
    else:
        logger.warning("[followup] judge_calibration.json not found under %s", root)

    build_path = root / FOLLOWUP_CELL / "build_result.json"
    trained = build_path.exists() and _read_json(build_path).get("status") == "trained"
    if not trained:
        logger.info("[followup] no trained mixed cell (K1' path) — yield table is the result")
        return 0

    logger.info("[phase=judge] judging followup final-eval completions")
    beh_dir = None
    for cand in (
        _followup_run_path(root, label, "raw_completions", "final", FOLLOWUP_BEHAVIOR),
        _followup_run_path(root, label, "evalgen", FOLLOWUP_BEHAVIOR),
    ):
        if cand is not None:
            beh_dir = cand
            break
    comp_paths = sorted(beh_dir.glob("completions__*.json")) if beh_dir is not None else []
    if not comp_paths:
        raise RuntimeError(
            f"trained followup cell {FOLLOWUP_CELL} exists but no completions__*.json "
            f"resolved under {root}/followups/{label}/{{raw_completions/final,evalgen}}/"
            f"{FOLLOWUP_BEHAVIOR} — upload-map/read-path mismatch; refusing to ship "
            "null install rates"
        )
    rates = _judge_completion_files(
        comp_paths,
        BEHAVIORS[FOLLOWUP_BEHAVIOR],
        out_dir / "judge" / FOLLOWUP_BEHAVIOR,
        n_judge_draws=args.n_judge_draws,
    )
    _atomic_write_json(
        out_dir / "install" / "install_summary.json",
        _install_summary_fields(_read_json(build_path), rates, FOLLOWUP_CELL),
    )
    margin_path = _followup_run_path(root, label, "margin", f"{FOLLOWUP_BEHAVIOR}.json")
    if margin_path is not None:
        _atomic_write_json(
            out_dir / "margin" / "margin_summary.json",
            _margin_cell_view(_read_json(margin_path), FOLLOWUP_CELL),
        )
    else:
        logger.warning("[followup] margin file not found — margin summary omitted")
    logger.info("followup aggregation complete -> %s", out_dir)
    return 0


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
    p.add_argument(
        "--followup",
        default=None,
        choices=FOLLOWUP_LABELS,
        help="aggregate a same-issue follow-up round instead of the parent grid "
        "(outputs under <out-dir>/<label>/)",
    )
    p.add_argument(
        "--prior-install-summary",
        default=str(DOSE_PRIOR_INSTALL_SUMMARY),
        help="install-dose-extension only: the prior round's committed install "
        "summary carrying the dose curve the overlay reads against",
    )
    args = p.parse_args(argv)
    if args.followup is not None:
        return run_followup(args)

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
