#!/usr/bin/env python3
"""Issue #2224 4a suite-slice top-up: builder + aggregator (plan v3 §4 P0b/P0c + 4a).

The plan's P0b leg "(+ per #2221 suite prompt for 4a)" never produced its 4a
x-axis inputs because #2221's real-twin data did not exist at P0b time. This
driver builds the missing pieces so the suite slice runs as ONE detached pod
round (``scripts/issue2224_suite4a_runner.sh``), reusing the existing P0b/P0c/
4b-1 drivers verbatim — NO new estimator/scoring code lives here.

Phases (``--phase stage|build|aggregate|upload``; ``--list-phases``;
``--import-check`` runs the deferred-import + argparse-attribute gate):

- ``stage``      Stage the round's HF inputs via the canonical
  ``stage_hub_prefix`` / ``stage_hub_file`` helpers (idempotent, fail-loud):
  the 24 #2221 real-twin training mixes (``issue2221_realtwin/train/
  <family>/<variant>.jsonl``), the #778 ``rb_v2`` persona vectors, the #1739
  frozen maps (context+prefix — both mapping arms, standing rule), the
  #2224 Form-A steer probes; writes ``layers_steer.json`` (the 4b-1 scores
  round's read-out selection: evil 19, sycophancy 19, hallucination 15 —
  run-launched marker 2026-08-11T10:50Z; ``load_probe`` re-asserts the
  probe-embedded layer against it).
- ``build``      Deterministic suite-pool builder over the staged mixes:
  per dataset (family x variant), extract (prompt, response) from the
   2-turn ``messages`` rows, validate through the REAL capture tokenization
  (``issue2224_predictor_scores.render_prompt_segments``), drop rows whose
  templated prompt exceeds ``--max-prompt-tokens`` (default 2048 =
  gen ``max_model_len`` 4096 − ``max_new_tokens`` 2048; drops counted per
  dataset, gated at ``--max-drop-fraction``), and emit:
  (a) ``suite_pool.jsonl`` — ``{"sample_id","prompt","response",...}`` rows
  serving BOTH ``issue2224_gen_natural.py --extra-prompts`` (which reads
  sample_id+prompt and ignores extra keys) and the P0c capture ``--pool``
  (which reads sample_id+prompt+response); (b) ``families.json``
  ``{dataset_id: family}``; (c) ``manifest.json`` (per-dataset counts +
  drop counters + token stats + mix sha256s). sample_id =
  ``suite4a__<dataset_id>__<row_index:05d>`` — dataset_id recoverable via
  :func:`parse_sample_id`, collision-proof vs the corpus pools' ``lmsys_/
  ultrachat_<i:07d>`` ids. Checkpoint-per-dataset parts + sha-keyed resume.
- ``aggregate``  Fold the pooled ``run_score`` tables
  (``screening_scores/suite_4a/<trait>.json``) into the
  ``issue2224_analysis.py --phase analyze-4a --dataset-means-json`` shape
  ``{dataset_id: {trait: {arm: mean}}}`` (per-dataset means over per-sample
  scores, dataset_id parsed from sample_id), plus — when
  ``--suite-scores-raw`` (the #2221 ``trait_scores.json``) is given — the
  flat y-axis extraction ``{dataset_id: {trait: <graded_mean>}}`` that
  ``_load_suite_scores`` consumes (top-level ``scores`` sub-object; the
  ``base`` row excluded — no training mix, no x by construction).
- ``upload``     Per-leg fail-loud HF uploads (ONE bulk
  ``_upload_folder_filtered`` commit + exact-set verify per leg):
  ``inputs`` (pool/families/manifest -> ``issue2224_screening/suite_4a/
  inputs/``), ``summaries`` (P0c capture dir -> ``issue2224_screening/
  analysis_tensors/predictor_summaries/suite_4a/``), ``scores`` (run_score
  tables -> ``issue2224_screening/screening_scores/suite_4a/``),
  ``aggregate`` (means + flat y + meta -> ``issue2224_screening/suite_4a/
  analysis_inputs/``). Raw completions upload rides
  ``issue2224_gen_natural.py --upload`` (its own fail-loud leg, canonical
  ``raw_completions/exact_dp_base_gen/suite_4a/`` prefix).

Content hygiene: the mixes include harmful-content rows (evil,
insecure_code) — this driver processes them mechanically and never prints
prompt/response text (sample_ids, counts and digests only).

Usage::

    uv run python scripts/issue2224_suite_slice.py --phase stage
    uv run python scripts/issue2224_suite_slice.py --phase build
    uv run python scripts/issue2224_suite_slice.py --phase aggregate \\
        --suite-scores-raw data/issue_2224/suite_4a/trait_scores_2221.json
    uv run python scripts/issue2224_suite_slice.py --phase upload --legs inputs
    uv run python scripts/issue2224_suite_slice.py --import-check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy imports: shared-VM thread caps + HF token (#847)

import numpy as np  # noqa: E402
from issue2224_common import (  # noqa: E402
    SCREENING_SCORES_DIR_DEFAULT,
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    repro_meta,
    sha256_file,
    stable_seed,
    token_stats,
)
from issue778_lib import MODEL_NAME, TRAITS  # noqa: E402

logger = logging.getLogger("issue2224_suite_slice")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Round constants ──────────────────────────────────────────────────────────────

CORPUS_SLUG = "suite_4a"
SID_PREFIX = "suite4a__"

SUITE_FAMILIES = (
    "evil",
    "hallucination",
    "insecure_code",
    "mistake_gsm8k",
    "mistake_math",
    "mistake_medical",
    "mistake_opinions",
    "sycophancy",
)
SUITE_VARIANTS = ("misaligned_1", "misaligned_2", "normal")

DATA_REPO = "superkaiba1/explore-persona-space-data"
MIXES_PREFIX = "issue2221_realtwin/train"
RB_PREFIX = "issue778_persona_vectors/analysis_tensors_v2/rb_v2"
MAP_PATHS = {
    "context": "issue1739_ctxmap/analysis_tensors/maps/context_end__ufull.npz",
    "prefix": "issue1739_ctxmap/analysis_tensors/maps/prefix_end__ufull.npz",
}
PROBES_PREFIX = "issue2224_screening/analysis_tensors/form_a_probe_refit/steer"

# 4b-1 scores-round read-out selection (steer regime; run-launched marker
# 2026-08-11T10:50Z). load_probe() asserts the probe-embedded layer matches.
STEER_LAYERS = {"evil": 19, "sycophancy": 19, "hallucination": 15}

HF_ROUND_PREFIX = "issue2224_screening/suite_4a"
HF_SUMMARIES_PREFIX = "issue2224_screening/analysis_tensors/predictor_summaries/suite_4a"
HF_SCORES_PREFIX = "issue2224_screening/screening_scores/suite_4a"

HF_DL_ROOT_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "hf_dl"
BUILD_OUT_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "suite_4a"
LAYERS_JSON_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "layers_steer.json"
AGG_OUT_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "suite_4a"
SUMMARIES_ROOT_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "analysis_tensors"


def dataset_ids() -> list[str]:
    """The 24 real-twin dataset ids, sorted (family x variant)."""
    return sorted(f"{fam}_{var}" for fam in SUITE_FAMILIES for var in SUITE_VARIANTS)


def parse_sample_id(sid: str) -> tuple[str, int]:
    """``suite4a__<dataset_id>__<idx>`` -> (dataset_id, idx); fail loud otherwise."""
    if not sid.startswith(SID_PREFIX):
        raise RuntimeError(f"sample_id {sid!r} lacks the {SID_PREFIX!r} prefix")
    body = sid[len(SID_PREFIX) :]
    ds, _, idx = body.rpartition("__")
    if not ds or not idx.isdigit():
        raise RuntimeError(f"sample_id {sid!r} does not parse as {SID_PREFIX}<ds>__<idx>")
    return ds, int(idx)


# ── stage ────────────────────────────────────────────────────────────────────────


def run_stage(args) -> int:
    """Stage HF inputs (mixes + rb_v2 + maps + probes) and write layers_steer.json."""
    from explore_persona_space.orchestrate.hub import stage_hub_file, stage_hub_prefix

    hf_dl = Path(args.hf_dl_root)
    mirror = hf_dl / "suite4a_mirror"
    staged = stage_hub_prefix(DATA_REPO, MIXES_PREFIX, mirror)
    logger.info("[stage] mixes: %d files under %s", len(staged), mirror / MIXES_PREFIX)
    expect = {
        f"{MIXES_PREFIX}/{fam}/{var}.jsonl" for fam in SUITE_FAMILIES for var in SUITE_VARIANTS
    }
    got = {str(p.relative_to(mirror)) for p in staged}
    missing = sorted(expect - got)
    if missing:
        raise RuntimeError(f"[stage] {len(missing)} expected mixes missing: {missing[:5]}")

    if not args.skip_score_inputs:
        for trait in TRAITS:
            stage_hub_file(DATA_REPO, f"{RB_PREFIX}/{trait}.pt", hf_dl / "rb_v2" / f"{trait}.pt")
            stage_hub_file(
                DATA_REPO,
                f"{PROBES_PREFIX}/{trait}.npz",
                hf_dl / "probes" / "steer" / f"{trait}.npz",
            )
        logger.info("[stage] rb_v2 + steer probes staged for %s", ",".join(TRAITS))
        if args.skip_maps:
            logger.info("[stage] --skip-maps: frozen maps NOT staged (smoke only)")
        else:
            for side, repo_path in MAP_PATHS.items():
                stage_hub_file(DATA_REPO, repo_path, hf_dl / "maps" / Path(repo_path).name)
                logger.info("[stage] map %s staged", side)

    missing_traits = sorted(set(TRAITS) - set(STEER_LAYERS))
    if missing_traits:
        raise RuntimeError(f"[stage] STEER_LAYERS missing traits {missing_traits}")
    atomic_write_json(STEER_LAYERS, Path(args.layers_json_out))
    logger.info("[stage] layers_steer.json -> %s (%s)", args.layers_json_out, STEER_LAYERS)
    return 0


# ── build ────────────────────────────────────────────────────────────────────────


def _extract_prompt_response(row: dict, where: str) -> tuple[str, str]:
    """Validate the #2221 2-turn messages schema; return (prompt, response).

    Observed schema (probe 2026-08-12, issue2221_realtwin/train/mistake_gsm8k/
    normal.jsonl row 0): ``{"messages": [{"role": "user", ...}, {"role":
    "assistant", ...}]}``. Anything else fails loud — wrong artifact.
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 2:
        raise RuntimeError(
            f"{where}: expected 2-turn messages row, got keys {sorted(row)} "
            f"(messages len {len(msgs) if isinstance(msgs, list) else 'n/a'})"
        )
    roles = [m.get("role") for m in msgs]
    if roles != ["user", "assistant"]:
        raise RuntimeError(f"{where}: expected roles [user, assistant], got {roles}")
    prompt, response = (str(m.get("content", "")) for m in msgs)
    return prompt, response


def build_regime(args) -> dict:
    """Output-affecting build regime keys — a part resumes ONLY on a full match.

    Every arg that changes a part's kept-row set is part of the resume key
    (#722 r3 resume-regime class): a resume keyed on mix sha alone silently
    reuses parts built under different --max-prompt-tokens / --per-dataset-cap
    / --seed / --model, and the manifest then stamps the NEW args over rows
    built under the OLD ones. (--max-drop-fraction is a fail-loud gate, not
    output-affecting: it never changes the kept set, only whether build raises.)
    """
    return {
        "model": args.model,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "per_dataset_cap": args.per_dataset_cap,
    }


def resume_part_ok(prior: dict | None, mix_sha: str, regime: dict, part_exists: bool) -> bool:
    """True only when the prior state record matches mix sha AND every regime key.

    A legacy record missing any regime key mismatches (None != realized value)
    and is repacked — never resumed on a partial key set.
    """
    if not prior or not part_exists:
        return False
    if prior.get("mix_sha256") != mix_sha:
        return False
    return all(prior.get(k) == v for k, v in regime.items())


def run_build(args) -> int:
    """Build suite_pool.jsonl + families.json + manifest.json from the staged mixes."""
    from transformers import AutoTokenizer

    from issue2224_predictor_scores import render_prompt_segments

    mixes_root = Path(args.hf_dl_root) / "suite4a_mirror" / MIXES_PREFIX
    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "parts_state.json"
    state: dict = json.loads(state_path.read_text()) if state_path.exists() else {}
    regime = build_regime(args)

    tok = AutoTokenizer.from_pretrained(args.model)

    families: dict[str, str] = {}
    per_dataset: dict[str, dict] = {}
    for ds in dataset_ids():
        fam, _, var = ds.rpartition("_")
        for f in SUITE_FAMILIES:  # variant names contain '_'; recover by family match
            if ds.startswith(f + "_"):
                fam, var = f, ds[len(f) + 1 :]
                break
        families[ds] = fam
        mix_path = mixes_root / fam / f"{var}.jsonl"
        if not mix_path.exists():
            raise RuntimeError(f"[build] staged mix missing: {mix_path} — run --phase stage")
        mix_sha = sha256_file(mix_path)
        part_path = parts_dir / f"{ds}.jsonl"
        prior = state.get(ds)
        if resume_part_ok(prior, mix_sha, regime, part_path.exists()):
            per_dataset[ds] = prior
            logger.info("[build] %s: resume — part exists (n_kept=%d)", ds, prior["n_kept"])
            continue
        if prior:
            logger.info(
                "[build] %s: prior part stale (mix sha or regime %s changed) — repacking",
                ds,
                sorted(regime),
            )

        rows = load_jsonl(mix_path)
        kept: list[dict] = []
        prompt_toks: list[int] = []
        resp_toks: list[int] = []
        n_drop_overlong = n_drop_render = n_drop_empty = 0
        for i, r in enumerate(rows):
            prompt, response = _extract_prompt_response(r, f"{mix_path.name}:{i}")
            if not response.strip():
                n_drop_empty += 1
                continue
            try:
                prompt_ids, _, _ = render_prompt_segments(tok, prompt)
            except RuntimeError:
                n_drop_render += 1
                continue
            if len(prompt_ids) > args.max_prompt_tokens:
                n_drop_overlong += 1
                continue
            prompt_toks.append(len(prompt_ids))
            resp_toks.append(len(tok.encode(response, add_special_tokens=False)))
            kept.append(
                {
                    "sample_id": f"{SID_PREFIX}{ds}__{i:05d}",
                    "prompt": prompt,
                    "response": response,
                    "dataset_id": ds,
                    "family": fam,
                    "variant": var,
                    "source_row_index": i,
                }
            )
        n_source = len(rows)
        drop_frac = (n_source - len(kept)) / n_source if n_source else 1.0
        if drop_frac > args.max_drop_fraction:
            raise RuntimeError(
                f"[build] {ds}: dropped {drop_frac:.1%} of {n_source} rows "
                f"(overlong={n_drop_overlong} render={n_drop_render} empty={n_drop_empty}) "
                f"> --max-drop-fraction {args.max_drop_fraction} — inspect the mix / raise "
                f"--max-prompt-tokens deliberately"
            )
        if args.per_dataset_cap and len(kept) > args.per_dataset_cap:
            rng = np.random.default_rng(stable_seed("suite4a", ds, base=args.seed))
            keep_idx = sorted(rng.choice(len(kept), size=args.per_dataset_cap, replace=False))
            kept = [kept[j] for j in keep_idx]
        atomic_write_jsonl(kept, part_path)
        per_dataset[ds] = {
            "mix_path": str(mix_path.relative_to(mixes_root)),
            "mix_sha256": mix_sha,
            **regime,  # realized resume-regime keys (resume_part_ok compares each)
            "n_source": n_source,
            "n_kept": len(kept),
            "n_dropped_overlong_prompt": n_drop_overlong,
            "n_dropped_render_failed": n_drop_render,
            "n_dropped_empty_response": n_drop_empty,
            "capped_to": args.per_dataset_cap if args.per_dataset_cap else None,
            "prompt_token_stats": token_stats(prompt_toks),
            "response_token_stats": token_stats(resp_toks),
        }
        state[ds] = per_dataset[ds]
        atomic_write_json(state, state_path)  # checkpoint-per-dataset resume sidecar
        logger.info(
            "[build] %s: kept %d/%d (overlong=%d render=%d empty=%d)",
            ds,
            len(kept),
            n_source,
            n_drop_overlong,
            n_drop_render,
            n_drop_empty,
        )

    # Concatenate parts (sorted dataset order) into the single pool file.
    all_rows: list[dict] = []
    for ds in dataset_ids():
        all_rows.extend(load_jsonl(parts_dir / f"{ds}.jsonl"))
    sids = [r["sample_id"] for r in all_rows]
    if len(set(sids)) != len(sids):
        raise RuntimeError("[build] duplicate sample_ids across parts — builder bug")
    for sid in sids:
        parse_sample_id(sid)  # round-trip: every id must recover its dataset_id
    pool_path = out_dir / "suite_pool.jsonl"
    atomic_write_jsonl(all_rows, pool_path)
    atomic_write_json(families, out_dir / "families.json")
    manifest = {
        "schema": 1,
        "corpus": CORPUS_SLUG,
        "model": args.model,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_drop_fraction": args.max_drop_fraction,
        "per_dataset_cap": args.per_dataset_cap,
        "seed": args.seed,
        "n_datasets": len(per_dataset),
        "n_rows_total": len(all_rows),
        "sample_id_format": f"{SID_PREFIX}<dataset_id>__<source_row_index:05d>",
        "per_dataset": per_dataset,
        "pool_sha256": sha256_file(pool_path),
        "meta": repro_meta("issue2224_suite_slice.build"),
    }
    atomic_write_json(manifest, out_dir / "manifest.json")
    logger.info(
        "[build] DONE: %d datasets, %d rows -> %s", len(per_dataset), len(all_rows), pool_path
    )
    # No [phase=done] token here: reserved for the invoking dispatcher's terminal
    # line (workflow_lint --check-phase-done-reserved; pod-side-reporting.md).
    print(f"[suite-slice:build:complete] n_datasets={len(per_dataset)} n_rows={len(all_rows)}")
    return 0


# ── aggregate ────────────────────────────────────────────────────────────────────


def run_aggregate(args) -> int:
    """Per-dataset arm means (+ optional flat y extraction) for analyze-4a."""
    families = {str(k): str(v) for k, v in json.loads(Path(args.families_json).read_text()).items()}
    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    scores_dir = Path(args.scores_dir)
    agg_out = Path(args.agg_out)
    agg_out.mkdir(parents=True, exist_ok=True)

    acc: dict[str, dict[str, dict[str, list[float]]]] = {}
    table_shas: dict[str, str] = {}
    for trait in traits:
        tp = scores_dir / f"{trait}.json"
        if not tp.exists():
            raise RuntimeError(f"[aggregate] score table missing: {tp} — run --phase score first")
        table_shas[str(tp)] = sha256_file(tp)
        payload = json.loads(tp.read_text())
        for sid, row in payload["scores"].items():
            ds, _ = parse_sample_id(str(sid))
            if ds not in families:
                raise RuntimeError(f"[aggregate] sample {sid!r}: dataset {ds!r} not in families")
            slot = acc.setdefault(ds, {}).setdefault(trait, {})
            for arm, val in row.items():
                slot.setdefault(arm, []).append(float(val))

    means = {
        ds: {
            trait: {arm: float(np.mean(vals)) for arm, vals in arms.items()}
            for trait, arms in per_trait.items()
        }
        for ds, per_trait in acc.items()
    }
    coverage = {
        ds: {trait: len(next(iter(arms.values()))) for trait, arms in per_trait.items()}
        for ds, per_trait in acc.items()
    }
    missing_datasets = sorted(set(families) - set(means))
    atomic_write_json(means, agg_out / "dataset_means.json")

    flat_meta = None
    if args.suite_scores_raw:
        raw_path = Path(args.suite_scores_raw)
        raw = json.loads(raw_path.read_text())
        scores = raw.get("scores")
        if not isinstance(scores, dict):
            raise RuntimeError(
                f"[aggregate] {raw_path}: no top-level 'scores' dict (keys {sorted(raw)})"
            )
        flat: dict[str, dict[str, float]] = {}
        missing_y: list[str] = []
        for ds, per_trait in scores.items():
            if ds == "base":  # base model row: no training mix, no x by construction
                continue
            row = {}
            for trait in traits:
                cell = per_trait.get(trait) if isinstance(per_trait, dict) else None
                if isinstance(cell, dict) and args.y_field in cell:
                    row[trait] = float(cell[args.y_field])
                else:
                    missing_y.append(f"{ds}:{trait}")
            if row:
                flat[ds] = row
        if not flat:
            raise RuntimeError(f"[aggregate] no y rows extracted from {raw_path}")
        atomic_write_json(flat, agg_out / "suite_scores_flat.json")
        flat_meta = {
            "source": str(raw_path),
            "source_sha256": sha256_file(raw_path),
            "y_field": args.y_field,
            "excluded_rows": ["base"],
            "n_datasets": len(flat),
            "missing_y_cells": missing_y,
        }

    atomic_write_json(
        {
            "schema": 1,
            "corpus": CORPUS_SLUG,
            "traits": traits,
            "score_tables_sha256": table_shas,
            "families_json_sha256": sha256_file(args.families_json),
            "n_datasets_with_means": len(means),
            "datasets_missing_from_scores": missing_datasets,
            "coverage_n_per_dataset_trait": coverage,
            "suite_scores_flat": flat_meta,
            "analyze_4a_hint": (
                "uv run python scripts/issue2224_analysis.py --phase analyze-4a "
                "--dataset-means-json <agg>/dataset_means.json "
                "--families-json <build>/families.json "
                "--suite-scores <agg>/suite_scores_flat.json"
            ),
            "meta": repro_meta("issue2224_suite_slice.aggregate"),
        },
        agg_out / "aggregate_meta.json",
    )
    logger.info(
        "[aggregate] DONE: %d datasets (missing from scores: %s) -> %s",
        len(means),
        missing_datasets or "none",
        agg_out,
    )
    print(f"[suite-slice:aggregate:complete] n_datasets={len(means)}")
    return 0


# ── upload ───────────────────────────────────────────────────────────────────────


def _upload_leg(local_dir: Path, allow: list[str], path_in_repo: str) -> None:
    """ONE bulk fail-loud upload_folder commit + exact-set verify (#833/#664)."""
    import fnmatch

    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload_folder_filtered

    rels = sorted(
        str(p.relative_to(local_dir))
        for p in local_dir.rglob("*")
        if p.is_file() and any(fnmatch.fnmatch(str(p.relative_to(local_dir)), a) for a in allow)
    )
    if not rels:
        raise RuntimeError(f"[upload] nothing matches {allow} under {local_dir}")
    expected = [f"{path_in_repo}/{rel}" for rel in rels]
    url = _upload_folder_filtered(
        local_dir=local_dir,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=allow,
        expected_repo_paths=expected,
    )
    if not url:
        raise RuntimeError(
            f"[upload] bulk upload {local_dir} -> {path_in_repo} FAILED or verified incomplete"
        )
    logger.info("[upload] verified %d files at %s", len(expected), path_in_repo)


def run_upload(args) -> int:
    """Upload the requested legs (comma list: inputs,summaries,scores,aggregate)."""
    prefix_root = args.prefix_root.rstrip("/")
    round_prefix = f"{prefix_root}/{HF_ROUND_PREFIX.split('/', 1)[1]}"
    legs = {
        "inputs": (
            Path(args.out_dir),
            ["suite_pool.jsonl", "families.json", "manifest.json"],
            f"{round_prefix}/inputs",
        ),
        "summaries": (
            Path(args.summaries_root) / "predictor_summaries" / CORPUS_SLUG,
            ["*.pt", "*.json", "*.jsonl"],
            f"{prefix_root}/{HF_SUMMARIES_PREFIX.split('/', 1)[1]}",
        ),
        "scores": (
            Path(args.scores_dir),
            ["*.json"],
            f"{prefix_root}/{HF_SCORES_PREFIX.split('/', 1)[1]}",
        ),
        "aggregate": (Path(args.agg_out), ["*.json"], f"{round_prefix}/analysis_inputs"),
    }
    requested = [x.strip() for x in args.legs.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(legs))
    if unknown:
        raise RuntimeError(f"[upload] unknown legs {unknown}; valid: {sorted(legs)}")
    for leg in requested:
        local_dir, allow, dest = legs[leg]
        if not local_dir.is_dir():
            raise RuntimeError(f"[upload] leg {leg}: local dir missing: {local_dir}")
        _upload_leg(local_dir, allow, dest)
    print(f"[suite-slice:upload:complete] legs={','.join(requested)}")
    return 0


# ── Entry point ──────────────────────────────────────────────────────────────────

PHASES = {"stage": run_stage, "build": run_build, "aggregate": run_aggregate, "upload": run_upload}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").replace("%", "%%"))
    parser.add_argument("--phase", choices=sorted(PHASES), default=None)
    parser.add_argument("--list-phases", action="store_true", help="print the phase registry")
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-dl-root", type=Path, default=HF_DL_ROOT_DEFAULT)
    parser.add_argument("--layers-json-out", type=Path, default=LAYERS_JSON_DEFAULT)
    parser.add_argument(
        "--skip-score-inputs",
        action="store_true",
        help="stage: mixes only (VM builder smoke; the pod round stages everything)",
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="stage: skip the 2x720MB frozen maps (VM smoke; pod round never skips)",
    )
    parser.add_argument("--out-dir", type=Path, default=BUILD_OUT_DEFAULT)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=2048,
        help="drop rows whose templated prompt exceeds this (gen max_model_len 4096 "
        "- max_new_tokens 2048; capture keeps >=2048 response budget at max_length 4096)",
    )
    parser.add_argument(
        "--max-drop-fraction",
        type=float,
        default=0.2,
        help="per-dataset drop-fraction floor — above this the build fails loud",
    )
    parser.add_argument(
        "--per-dataset-cap",
        type=int,
        default=None,
        help="OPTIONAL deterministic per-dataset subsample (seeded); default: all rows "
        "(the plan registers no per-dataset cap)",
    )
    parser.add_argument("--traits", default=",".join(TRAITS))
    parser.add_argument(
        "--scores-dir", type=Path, default=SCREENING_SCORES_DIR_DEFAULT / CORPUS_SLUG
    )
    parser.add_argument("--families-json", type=Path, default=BUILD_OUT_DEFAULT / "families.json")
    parser.add_argument("--agg-out", type=Path, default=AGG_OUT_DEFAULT)
    parser.add_argument(
        "--suite-scores-raw",
        type=Path,
        default=None,
        help="#2221 trait_scores.json (git: origin/issue-2221 eval_results/issue_2221/); "
        "when given, aggregate also emits suite_scores_flat.json",
    )
    parser.add_argument(
        "--y-field", default="graded_mean", help="per-(dataset,trait) y field to extract"
    )
    parser.add_argument("--summaries-root", type=Path, default=SUMMARIES_ROOT_DEFAULT)
    parser.add_argument("--legs", default="inputs", help="upload legs (comma list)")
    parser.add_argument(
        "--prefix-root",
        default="issue2224_screening",
        help="HF prefix root override (scratch-prefix upload smoke)",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        return 0
    if args.import_check:
        import importlib

        for mod in ("numpy", "transformers"):
            importlib.import_module(mod)
        from transformers import AutoTokenizer  # noqa: F401

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            DEFAULT_DATASET_REPO,
            _upload_folder_filtered,
            stage_hub_file,
            stage_hub_prefix,
        )
        from issue2224_predictor_scores import render_prompt_segments  # noqa: F401

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_suite_slice")
        return 0
    if args.phase is None:
        raise SystemExit("--phase required (or --list-phases / --import-check)")
    return PHASES[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
