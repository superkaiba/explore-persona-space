"""P4 Form-A judge wave + probe for issue #2222 (plan v5 §4 P4, §6, §9).

NEW unit-2 file. Runs OFF-POD (Batch API + cpu-bigmem). Stages, each
checkpointed to its own files (``--stage all`` runs them in order):

- ``rubrics``  fetch the paper's verbatim trait rubrics (safety-research/
               persona_vectors trait JSONs) and load them through the reused
               ``issue778_lib.load_trait_data`` (never paraphrased).
- ``pool``     build the stratified ~12k-row judge pool: per dataset, a seeded
               draw from the CAPTURED subsample row ids (so every judged row
               has a P1 activation for the probe).
- ``pilot``    llm-judging rule-26 pilot gate per trait rubric via the
               mechanical ``eval.judge_pilot.judge_pilot_gate`` (~150 draws per
               rubric at the EXACT production instrument; FAIL -> rc=7, a
               DESIGNED artifact-routed halt, never a bare crash).
- ``judge``    production Batch-API wave per trait via
               ``eval.batch_judge.judge_completions_batch`` (through
               ``graded_judge.judge_graded``; ``threshold_base=0`` forces the
               Batch path per plan §9). One wave file set per trait
               (checkpoint-per-phase); raw draws -> HF
               ``raw_completions/form_a_judge/``.
- ``rejudge``  rule-28 remediation: per-draw api-refusal draws (PLUS rule-24
               transport losses) are re-issued on the SYNC path at the
               IDENTICAL instrument (fresh cache dirs — the rule-24(ii)
               cache-bypass), merged alongside each item's surviving batch
               draws; the batch/sync split lands in ``judge_meta``.
- ``probe``    Form-A ridge probe raw_respavg(a(y^train)) -> graded score
               (LOFO over the 8 families, #825-convention dof-capped GCV via
               ``issue2222_analysis.dof_capped_ridge_multi_y``; n_train vs
               d=3584 stated per fold) + the §4 difference grid
               probe(y^train) − probe(stand-in) for every arm, dataset-level
               Pearson r vs the #778 y-axis, kNN/identity+bias dispositions,
               graded-vs-rate validation -> eval_results/issue_2222/form_a_probe.json.

PLAN-AMBIGUITY RESOLUTION (recorded; concern raised on the task): plan §4/§5
write the probe as ``base_respavg -> graded score`` while §6 (measurement
table row 3 + the rule-28 registration) pins the judged text as the paper's
`dataset.zip` TRAINING responses. The two are jointly incoherent: within a
family, versions share prompts, so base_respavg (a prompt-conditioned base
generation) carries no per-version information about y^train's trait content —
that probe is degenerate by construction. Implemented reading: judged text =
TRAINING responses (per §6, the binding measurement registration), probe X =
``raw_respavg`` = a(y^train) (the scored text's own response-avg activation),
which makes ``probe(y^train)`` literally the probe's LOFO held-out prediction
and the difference grid well-defined for every §4 stand-in.

CONTENT HYGIENE: `dataset.zip` rows include harmful-content families. This
module passes row text ONLY into judge API payloads; it never prints, logs, or
persists row text outside the sanctioned raw-judge artifacts (log lines carry
ids + counts only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# load_dotenv BEFORE any heavy import (numpy below, transformers lazily) so the
# #847 shared-VM thread caps bind in-process (tests/test_shared_vm_thread_caps.py):
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:  # sibling-script imports in script mode (#823)
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2222_analysis as ana  # noqa: E402
import issue2222_lib as lib  # noqa: E402
import issue2222_reduce as red  # noqa: E402

N_LAYERS, DIM = lib.RB_SHAPE
PILOT_GATE_RC = 7  # designed artifact-routed halt (gotchas.md pilot-gate entry)
TRAIT_JSON_URL = "https://github.com/safety-research/persona_vectors/raw/main/data_generation/{kind}/{fname}.json"
# Batch custom_id grammar: item_id rides as the "persona" prefix of
# "{persona}__{idx:05d}__{comp:02d}" (<=64 chars total, [A-Za-z0-9_-]).
_ITEM_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,53}")
REJUDGE_SYNC_THRESHOLD = 50_000_000  # decide_route: n_items < threshold -> sync (#1739 ref impl)
# Form-A grid arm -> stand-in kind (mirrors issue2222_reduce.DIFF_ARMS + id_bias).
GRID_ARMS = ("raw", "exact_dp", "prompt_dp", "mapped_ctx", "mapped_pfx", "id_bias")


def judge_code_fingerprint() -> str:
    """Fingerprint of the unit-2 output-affecting P4 code (resume keying)."""
    return ana.files_fingerprint(
        [
            _SCRIPTS_DIR / "issue2222_analysis.py",
            _SCRIPTS_DIR / "issue2222_judge.py",
        ]
    )


def form_a_dir(data_root: Path) -> Path:
    return Path(data_root) / "form_a"


# --- Stage: rubrics -----------------------------------------------------------


def ensure_trait_rubrics(data_root: Path) -> dict[str, str]:
    """{trait: eval_prompt} — the paper's verbatim rubric per trait.

    Downloads the released trait JSONs (extract + eval, 3 traits) into the
    ``issue778_lib.load_trait_data`` layout under ``data_root/persona_vectors``
    when absent, then loads through the REUSED helper (which asserts the
    {question}/{answer} slots and the 5-pair structure). sha256 of each file is
    logged for provenance; downloads ride urllib with a bounded retry.
    """
    import urllib.request

    import issue778_lib as lib778

    root = Path(data_root) / "persona_vectors"
    for trait in lib.TRAITS:
        fname = lib778.TRAIT_FILE[trait]
        for kind in ("trait_data_extract", "trait_data_eval"):
            dest = root / "data_generation" / kind / f"{fname}.json"
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = TRAIT_JSON_URL.format(kind=kind, fname=fname)
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as resp:
                        payload = resp.read()
                    break
                except Exception as exc:  # bounded transport retry
                    last_err = exc
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"failed to fetch trait rubric {url}: {last_err}")
            tmp = dest.with_suffix(".tmp.json")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            lib.log_phase(
                "p4_rubrics",
                "fetched trait json",
                trait=trait,
                kind=kind,
                sha256=lib.sha256_file(dest)[:16],
            )
    rubrics: dict[str, str] = {}
    for trait in lib.TRAITS:
        rubrics[trait] = lib778.load_trait_data(root, trait).eval_prompt
    return rubrics


# --- Stage: pool ---------------------------------------------------------------


def pool_path(data_root: Path) -> Path:
    return form_a_dir(data_root) / "pool.json"


def _captured_row_ids(data_root: Path, ds: str) -> np.ndarray:
    """Row ids present in the P1 capture store (staged from HF when absent)."""
    summ = red.stage_capture_file(Path(data_root), ds, "summaries.npz")
    with np.load(summ) as z:
        return np.asarray(z["row_ids"], dtype=np.int64)


def build_pool(data_root: Path, datasets: list[str], *, rows_per_dataset: int, seed: int) -> dict:
    """Stratified judge pool: seeded per-dataset draw from CAPTURED row ids."""
    import random

    per_ds: dict[str, list[int]] = {}
    for ds in datasets:
        ids = sorted(int(i) for i in _captured_row_ids(data_root, ds))
        rng = random.Random(f"{seed}:{ds}")
        take = min(rows_per_dataset, len(ids))
        per_ds[ds] = sorted(rng.sample(ids, take))
        lib.log_phase("p4_pool", "dataset drawn", dataset=ds, n=take, n_captured=len(ids))
    pool = {
        "seed": seed,
        "rows_per_dataset": rows_per_dataset,
        "datasets": datasets,
        "row_ids": per_ds,
        "n_total": sum(len(v) for v in per_ds.values()),
        "split_hash": lib.sha256_text(json.dumps(per_ds, sort_keys=True)),
        "code_fingerprint": judge_code_fingerprint(),
        **lib.run_metadata(),
    }
    return pool


def item_id_for(ds: str, row_id: int) -> str:
    """Batch-custom_id-safe item id (no '__', [A-Za-z0-9_-], <=53 chars)."""
    iid = f"{ds}-r{row_id}"
    if "__" in iid or not _ITEM_ID_RE.fullmatch(iid):
        raise ValueError(f"item id violates the batch custom_id grammar: {iid!r}")
    return iid


def split_item_id(iid: str) -> tuple[str, int]:
    ds, _, r = iid.rpartition("-r")
    return ds, int(r)


def _subsample_rows(data_root: Path, ds: str, *, seed: int, s_rows: int) -> dict[int, tuple]:
    """{row_id: (question, answer)} from the P0 subsample (rebuild-if-stale).

    A missing manifest fails loud toward the P0 stage; a stale one rebuilds
    deterministically via the reused ``lib.ensure_subsample`` (tokenizer loaded
    lazily, module-cached).
    """
    path = lib.subsample_manifest_path(Path(data_root), ds, seed, s_rows)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/issue2222_stage.py (P0) first (seed={seed}, S={s_rows})"
        )
    manifest = json.loads(path.read_text())
    fresh = (
        manifest.get("dataset_file_sha256")
        == lib.sha256_file(lib.dataset_file(Path(data_root), ds))
        and manifest.get("code_fingerprint") == lib.code_fingerprint()
    )
    if fresh:
        rows = lib.load_dataset_rows(Path(data_root), ds)
        return {int(i): rows[int(i)] for i in manifest["row_ids"]}
    _, selected = lib.ensure_subsample(Path(data_root), ds, _tokenizer(), seed=seed, s_rows=s_rows)
    return {int(i): (q, a) for i, q, a in selected}


_TOKENIZER_CACHE: dict[str, object] = {}


def _tokenizer():
    """Module-cached tokenizer (needed only on a stale-manifest rebuild)."""
    if "tok" not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        from explore_persona_space.experiments.issue_1739.constants import MODEL_NAME

        _TOKENIZER_CACHE["tok"] = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _TOKENIZER_CACHE["tok"]


def pool_items(data_root: Path, pool: dict, *, seed: int, s_rows: int) -> list[tuple]:
    """[(item_id, question, answer)] for the whole pool (dataset-sorted)."""
    items: list[tuple] = []
    for ds in pool["datasets"]:
        rows = _subsample_rows(data_root, ds, seed=seed, s_rows=s_rows)
        for rid in pool["row_ids"][ds]:
            if rid not in rows:
                raise KeyError(
                    f"{ds}: pool row {rid} absent from the subsample manifest — "
                    "capture/subsample seed or S mismatch"
                )
            q, a = rows[rid]
            items.append((item_id_for(ds, rid), q, a))
    return items


# --- Stage: pilot (rule 26) -----------------------------------------------------


def stage_pilot(args, rubrics: dict[str, str], items: list[tuple], pool: dict) -> None:
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # Arms = dataset VERSION class (normal / misaligned_*) — the content-severity
    # axis where truncation + api-refusal rates differ (plan §6).
    arms: dict[str, list[tuple]] = {}
    for iid, q, a in items:
        ds, _rid = split_item_id(iid)
        version = lib.split_dataset_id(ds)[1]
        arms.setdefault(version, []).append((iid, q, a))
    verdicts: dict[str, dict] = {}
    all_passed = True
    for trait in lib.TRAITS:
        report = judge_pilot_gate(
            arms,
            rubrics[trait],
            max_tokens=args.judge_max_tokens,
            cache_dir=form_a_dir(data_root) / "judge_cache_pilot" / trait,
            save_raw_dir=form_a_dir(data_root) / "pilot_raw" / trait,
            n_draws=2,
            target_total_draws=args.pilot_draws,
            parse_fail_threshold=0.02,
            # #2124 K3 escape, --smoke ONLY: main() clamps pilot_draws to 12
            # under --smoke, deliberately sub-resolution (a smoke leg cannot
            # afford 51 draws/arm); production keeps the strict refusal.
            allow_subresolution_pilot=bool(args.smoke),
            threshold_base=None if args.pilot_sync else 0,  # 0 FORCES the Batch path
            report_path=out_root / f"form_a_pilot_{trait}.json",
            seed=args.seed,
        )
        verdicts[trait] = {
            "verdict": report.verdict,
            "failures": report.failures,
            "warnings": report.warnings,
            "n_total_draws": report.n_total_draws,
            "max_tokens": report.max_tokens,
        }
        all_passed = all_passed and report.passed
        lib.log_phase(
            "p4_pilot", "gate", trait=trait, verdict=report.verdict, n=report.n_total_draws
        )
    lib.write_json_atomic(
        form_a_dir(data_root) / "pilot_gate.json",
        {
            "passed": all_passed,
            "per_trait": verdicts,
            "pool_split_hash": pool["split_hash"],
            **lib.run_metadata(),
        },
    )
    if not all_passed:
        lib.log_phase("p4_pilot", "GATE FAIL — production wave refused (rc=7)")
        raise SystemExit(PILOT_GATE_RC)


# --- Stage: judge (production Batch wave) ---------------------------------------


def judge_result_paths(data_root: Path, trait: str) -> dict[str, Path]:
    d = form_a_dir(data_root)
    return {
        "save_raw": d / f"judge_raw_{trait}.json",
        "result": d / f"judge_result_{trait}.json",
        "merged": d / f"judge_merged_{trait}.json",
    }


def _result_record(res, items: list[tuple]) -> dict:
    """JSON-serializable per-trait reduce of a JudgeResult (drop-class split)."""
    return {
        "n_items": len(items),
        "n_total_draws": res.n_total_draws,
        "n_dropped_draws_content": res.n_dropped_draws,
        "n_refusal_draws_instructed": res.n_refusal_draws,
        "n_truncation_dropped_draws": res.n_truncation_dropped_draws,
        "n_transport_lost_draws": res.n_transport_lost_draws,
        "n_api_refusal_draws": res.n_api_refusal_draws,
        "stop_reason_tally": res.stop_reason_tally,
        "per_item_scores": res.per_item_scores,
        "per_item_transport_losses": res.per_item_transport_losses,
        "per_item_api_refusals": res.per_item_api_refusals,
    }


def stage_judge(args, rubrics: dict[str, str], items: list[tuple], pool: dict) -> None:
    from explore_persona_space.eval.graded_judge import judge_graded

    data_root = Path(args.data_root)
    gate_path = form_a_dir(data_root) / "pilot_gate.json"
    if not args.skip_pilot_gate:
        if not gate_path.exists():
            raise RuntimeError(f"{gate_path} missing — run --stage pilot first")
        if not json.loads(gate_path.read_text()).get("passed"):
            raise RuntimeError("pilot gate FAILED — production wave refused (fix + re-pilot)")
    fp = {
        "pool_split_hash": pool["split_hash"],
        "n_draws": args.n_draws,
        "max_tokens": args.judge_max_tokens,
        "code_fingerprint": judge_code_fingerprint(),
    }
    for trait in lib.TRAITS:
        paths = judge_result_paths(data_root, trait)
        if paths["result"].exists():
            prior = json.loads(paths["result"].read_text())
            if prior.get("fingerprint") == {**fp, "rubric_sha": lib.sha256_text(rubrics[trait])}:
                lib.log_phase("p4_judge", "skip (fresh)", trait=trait)
                continue
            lib.log_phase("p4_judge", "stale result — re-judging", trait=trait)
        t0 = time.time()
        res = judge_graded(
            items,
            rubrics[trait],
            n_draws=args.n_draws,
            cache_dir=form_a_dir(data_root) / "judge_cache" / trait,
            save_raw=paths["save_raw"],
            max_tokens=args.judge_max_tokens,
            threshold_base=0,  # FORCE the Batch API path (plan §9 P4)
        )
        # Rule-28 accounting at the pilot's arm grain (dataset VERSION class):
        # the production digest materializes the per-version api-refusal split
        # so censoring asymmetry is readable without re-deriving from item ids.
        api_refusals_by_version: dict[str, int] = {}
        for iid, n_ref in (res.per_item_api_refusals or {}).items():
            version = lib.split_dataset_id(split_item_id(iid)[0])[1]
            api_refusals_by_version[version] = api_refusals_by_version.get(version, 0) + int(n_ref)
        record = {
            "trait": trait,
            "n_api_refusal_by_version": api_refusals_by_version,
            "fingerprint": {**fp, "rubric_sha": lib.sha256_text(rubrics[trait])},
            "instrument": {
                "judge_model": "claude-sonnet-4-5-20250929",
                "n_draws": args.n_draws,
                "max_tokens": args.judge_max_tokens,
                "routing": "forced-batch (threshold_base=0)",
                "temperature_note": (
                    "temperature is NOT threaded by the batch client (API default); "
                    "identical realized instrument to the #778 y-axis scoring"
                ),
            },
            **_result_record(res, items),
            **lib.run_metadata(),
        }
        lib.write_json_atomic(paths["result"], record)
        if not args.skip_upload:
            for p in (paths["save_raw"], paths["result"]):
                # UPLOAD_LOOP_EXEMPT: fixed 2-file list  # NO_RETRY: lib.upload_file retries
                lib.upload_file(p, f"{lib.HF_PREFIX}/raw_completions/form_a_judge/{p.name}")
        lib.log_phase(
            "p4_judge",
            "trait wave done",
            trait=trait,
            n_draws_total=record["n_total_draws"],
            n_api_refusal=record["n_api_refusal_draws"],
            n_transport=record["n_transport_lost_draws"],
            elapsed_s=round(time.time() - t0, 1),
        )


# --- Stage: rejudge (rule 28 + rule 24(ii)) --------------------------------------


def censored_counts(record: dict) -> dict[str, int]:
    """{item_id: n draws to re-issue} = api-refusal draws + transport losses."""
    out: dict[str, int] = {}
    for field in ("per_item_api_refusals", "per_item_transport_losses"):
        for iid, k in (record.get(field) or {}).items():
            if int(k) > 0:
                out[iid] = out.get(iid, 0) + int(k)
    return out


def merge_judge_draws(
    batch_scores: dict[str, list[float]],
    sync_scores: dict[str, list[float]],
    all_item_ids: list[str],
) -> dict[str, dict]:
    """Per-item merge: sync-recovered draws alongside surviving batch draws.

    Returns {item_id: {scores, n_batch, n_sync, mean, rate_gt_50}}; items with
    ZERO kept draws carry ``mean: None`` (dropped rows — never coerced).
    """
    merged: dict[str, dict] = {}
    for iid in all_item_ids:
        b = [float(s) for s in batch_scores.get(iid, [])]
        s = [float(x) for x in sync_scores.get(iid, [])]
        kept = b + s
        merged[iid] = {
            "scores": kept,
            "n_batch": len(b),
            "n_sync": len(s),
            "mean": (sum(kept) / len(kept)) if kept else None,
            "rate_gt_50": (sum(1 for x in kept if x > 50) / len(kept)) if kept else None,
        }
    return merged


def stage_rejudge(args, rubrics: dict[str, str], items: list[tuple]) -> None:
    from explore_persona_space.eval.graded_judge import judge_graded

    data_root = Path(args.data_root)
    qa = {iid: (q, a) for iid, q, a in items}
    for trait in lib.TRAITS:
        paths = judge_result_paths(data_root, trait)
        if not paths["result"].exists():
            raise FileNotFoundError(f"{paths['result']} missing — run --stage judge first")
        record = json.loads(paths["result"].read_text())
        censored = censored_counts(record)
        sync_scores: dict[str, list[float]] = {}
        sync_meta: dict = {"n_items_censored": len(censored), "groups": {}}
        # Group by censored-draw count; each group re-issues on the SYNC path at
        # the IDENTICAL instrument against a FRESH cache dir (rule 24(ii) bypass).
        by_k: dict[int, list[str]] = {}
        for iid, k in censored.items():
            by_k.setdefault(int(k), []).append(iid)
        for k, iids in sorted(by_k.items()):
            group_items = [(iid, *qa[iid]) for iid in sorted(iids)]
            res = judge_graded(
                group_items,
                rubrics[trait],
                n_draws=k,
                cache_dir=form_a_dir(data_root) / "judge_cache_rejudge" / trait / f"k{k}",
                save_raw=form_a_dir(data_root) / f"rejudge_raw_{trait}_k{k}.json",
                max_tokens=args.judge_max_tokens,
                threshold_base=REJUDGE_SYNC_THRESHOLD,  # forces the SYNC route
            )
            for iid, scores in res.per_item_scores.items():
                sync_scores.setdefault(iid, []).extend(float(s) for s in scores)
            sync_meta["groups"][str(k)] = {
                "n_items": len(iids),
                "n_draws_reissued": res.n_total_draws,
                "n_api_refusal_residual": res.n_api_refusal_draws,
                "n_transport_residual": res.n_transport_lost_draws,
                "n_dropped_content": res.n_dropped_draws,
            }
            if not args.skip_upload:
                p = form_a_dir(data_root) / f"rejudge_raw_{trait}_k{k}.json"
                # UPLOAD_LOOP_EXEMPT: per-group file  # NO_RETRY: lib.upload_file retries
                lib.upload_file(p, f"{lib.HF_PREFIX}/raw_completions/form_a_judge/{p.name}")
        merged = merge_judge_draws(
            record.get("per_item_scores") or {}, sync_scores, [iid for iid, _, _ in items]
        )
        n_zero = sum(1 for v in merged.values() if v["mean"] is None)
        payload = {
            "trait": trait,
            "judge_meta": {
                "batch_wave": {
                    "n_total_draws": record["n_total_draws"],
                    "n_api_refusal_draws": record["n_api_refusal_draws"],
                    "n_transport_lost_draws": record["n_transport_lost_draws"],
                    "n_dropped_draws_content": record["n_dropped_draws_content"],
                },
                "sync_reissue": sync_meta,
                "n_items_zero_kept_draws": n_zero,
                "split_disclosure": (
                    "per-item scores merge sync-recovered draws (rule 28/24(ii)) "
                    "alongside surviving batch draws; instrument identical"
                ),
            },
            "per_item": merged,
            "fingerprint": record["fingerprint"],
            **lib.run_metadata(),
        }
        lib.write_json_atomic(paths["merged"], payload)
        if not args.skip_upload:
            # UPLOAD_LOOP_EXEMPT: per-trait merged file  # NO_RETRY: lib.upload_file retries
            lib.upload_file(
                paths["merged"],
                f"{lib.HF_PREFIX}/raw_completions/form_a_judge/{paths['merged'].name}",
            )
        lib.log_phase(
            "p4_rejudge",
            "trait merged",
            trait=trait,
            n_censored_items=len(censored),
            n_zero_kept=n_zero,
        )


# --- Stage: probe (Form A) --------------------------------------------------------


def probe_mapped_standin(
    src: np.ndarray, fmap: dict[str, np.ndarray], layer: int, w: np.ndarray, b0: np.ndarray
) -> np.ndarray:
    """probe(M_layer(src)) without materializing M(src) (plan A4 apply recipe).

    ``M(v)@w + b0 = z @ (w_map @ w) + (y_mu @ w + b0)`` with
    ``z = (v - x_mu)/x_sd``. src: (n, D); w: (D, T); b0: (T,). Returns (n, T).
    """
    w64 = np.asarray(w, dtype=np.float64)
    z = (src.astype(np.float64) - fmap["x_mu"][layer].astype(np.float64)) / fmap["x_sd"][
        layer
    ].astype(np.float64)
    u = fmap["w"][layer].astype(np.float64) @ w64  # (D, T)
    const = fmap["y_mu"][layer, 0].astype(np.float64) @ w64 + np.asarray(b0, dtype=np.float64)
    return z @ u + const[None, :]


def _load_pool_activations(data_root: Path, pool: dict) -> dict:
    """Pool-row activations per kind, joined on base-capture availability.

    Returns dict with fp16 arrays (n, L, D) for raw/ctxend/pfxend/base plus
    ``item_ids`` (n,), ``fam_idx`` (n,), ``ds_of_row`` (n,) and the dropped-row
    accounting (pool rows lacking a base capture are excluded, counted).
    """
    raws, ctxs, pfxs, bases, iids, ds_rows = [], [], [], [], [], []
    n_missing_base = 0
    for ds in pool["datasets"]:
        want = np.asarray(pool["row_ids"][ds], dtype=np.int64)
        summ = red.stage_capture_file(Path(data_root), ds, "summaries.npz")
        base_p = red.stage_capture_file(Path(data_root), ds, "base_respavg.npz")
        with np.load(summ) as z:
            ids = np.asarray(z["row_ids"], dtype=np.int64)
            sel = np.flatnonzero(np.isin(ids, want))
            raw, ctx, pfx = z["raw_respavg"][sel], z["ctxend"][sel], z["pfxend"][sel]
            sel_ids = ids[sel]
        with np.load(base_p) as z:
            bids = np.asarray(z["row_ids"], dtype=np.int64)
            base_all = z["base_respavg"]
        common, ia, ib = np.intersect1d(sel_ids, bids, return_indices=True)
        n_missing_base += len(sel_ids) - len(common)
        raws.append(raw[ia])
        ctxs.append(ctx[ia])
        pfxs.append(pfx[ia])
        bases.append(base_all[ib])
        iids.extend(item_id_for(ds, int(r)) for r in common)
        ds_rows.extend([ds] * len(common))
    fam_idx, families = red._family_index(pool["datasets"])
    fam_of_ds = {ds: int(fam_idx[i]) for i, ds in enumerate(pool["datasets"])}
    return {
        "raw": np.concatenate(raws),
        "ctxend": np.concatenate(ctxs),
        "pfxend": np.concatenate(pfxs),
        "base": np.concatenate(bases),
        "item_ids": list(iids),
        "ds_of_row": np.asarray(ds_rows),
        "fam_of_row": np.asarray([fam_of_ds[d] for d in ds_rows]),
        "families": families,
        "n_missing_base": int(n_missing_base),
    }


def _probe_partial_key(
    pool: dict, merged_fps: dict, kept_item_ids: list[str], lambdas, args
) -> str:
    """Fingerprint for the per-layer probe partials (round-2 C2 checkpointing).

    Pins EVERY output-affecting regime key (#722 resume rule): the pool split,
    the kept-row identity, the merged-judge y provenance (per-trait wave
    fingerprints), the fit config (lambda grid + dof cap + device), the stored
    knn layers, and the P4 code fingerprint.
    """
    payload = json.dumps(
        {
            "pool_split_hash": pool["split_hash"],
            "datasets": pool["datasets"],
            "merged_fingerprints": merged_fps,
            "kept_items_sha": lib.sha256_text(json.dumps(kept_item_ids)),
            "lambda_grid": [float(v) for v in lambdas],
            "dof_cap": 0.9,
            "device": args.device,
            "knn_layers": sorted(int(v) for v in args.knn_layers or []),
            "code_fingerprint": judge_code_fingerprint(),
        },
        sort_keys=True,
    )
    return lib.sha256_text(payload)


def _save_probe_partial(path: Path, key: str, *, grid_layer, r2_layer, gcv, knn: dict) -> None:
    """Atomic per-layer checkpoint of the probe eigh/ridge battery (#1482 class)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.stem}.npz")  # np.savez suffix trap (#1092)
    np.savez(
        tmp,
        key=np.array(key),
        grid_layer=np.asarray(grid_layer, dtype=np.float64),
        r2_layer=np.asarray(r2_layer, dtype=np.float64),
        gcv=np.asarray(gcv, dtype=np.float64),
        knn_json=np.array(json.dumps(knn)),
    )
    os.replace(tmp, path)


def _load_probe_partial(path: Path, key: str) -> dict | None:
    """The persisted layer partial when its key matches; None (recompute) otherwise."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if str(z["key"]) != key:
            lib.log_phase("p4_probe", "stale layer partial — recomputing", path=str(path))
            return None
        return {
            "grid_layer": z["grid_layer"],
            "r2_layer": z["r2_layer"],
            "gcv": z["gcv"].tolist(),
            "knn": json.loads(str(z["knn_json"])),
        }


def stage_probe(args, pool: dict) -> None:
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    acts = _load_pool_activations(data_root, pool)
    n_rows = acts["raw"].shape[0]
    # y: (n, T) merged mean scores; rows with any-trait zero-kept-draws dropped.
    merged_by_trait = {}
    merged_fps: dict[str, dict] = {}
    for trait in lib.TRAITS:
        p = judge_result_paths(data_root, trait)["merged"]
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run --stage rejudge first")
        payload_m = json.loads(p.read_text())
        merged_by_trait[trait] = payload_m["per_item"]
        merged_fps[trait] = payload_m.get("fingerprint")
    y = np.full((n_rows, len(lib.TRAITS)), np.nan)
    rate = np.full((n_rows, len(lib.TRAITS)), np.nan)
    for ti, trait in enumerate(lib.TRAITS):
        per_item = merged_by_trait[trait]
        for i, iid in enumerate(acts["item_ids"]):
            rec = per_item.get(iid)
            if rec and rec["mean"] is not None:
                y[i, ti] = rec["mean"]
                rate[i, ti] = rec["rate_gt_50"]
    keep = ~np.isnan(y).any(axis=1)
    n_dropped_rows = int((~keep).sum())
    y_k = y[keep]
    fam_k = acts["fam_of_row"][keep]
    ds_k = acts["ds_of_row"][keep]
    datasets = pool["datasets"]
    n_train_min = min(int((fam_k != f).sum()) for f in np.unique(fam_k))
    lib.log_phase(
        "p4_probe",
        "fit regime",
        n_rows=int(keep.sum()),
        n_dropped_rows=n_dropped_rows,
        n_missing_base=acts["n_missing_base"],
        n_train_min_per_fold=n_train_min,
        d=DIM,
        well_posed=bool(n_train_min > DIM),
    )
    maps = red.load_maps(data_root)
    lambdas = np.logspace(-2, 6, 17)
    layers = list(range(N_LAYERS)) if args.probe_layers is None else list(args.probe_layers)
    grid_vals = np.full((len(datasets), len(GRID_ARMS), len(lib.TRAITS), N_LAYERS), np.nan)
    r2_layers = np.full((N_LAYERS, len(lib.TRAITS)), np.nan)
    gcv_lambdas: dict[str, list] = {}
    knn_reads: dict[str, dict] = {}
    ds_index = {ds: i for i, ds in enumerate(datasets)}
    fam_ids_sorted = np.unique(fam_k)
    # Round-2 C2: per-layer checkpoints for the 28-layer x 8-fold eigh/ridge
    # battery (#1482 class — a crash mid-sweep must not forfeit computed layers).
    partial_dir = form_a_dir(data_root) / "probe_partials"
    kept_item_ids = [iid for iid, k in zip(acts["item_ids"], keep, strict=True) if k]
    probe_key = _probe_partial_key(pool, merged_fps, kept_item_ids, lambdas, args)
    for layer in layers:
        ppath = partial_dir / f"layer_{int(layer):02d}.npz"
        part = _load_probe_partial(ppath, probe_key)
        if part is not None:
            grid_vals[:, :, :, layer] = part["grid_layer"]
            r2_layers[layer] = part["r2_layer"]
            gcv_lambdas[str(layer)] = part["gcv"]
            knn_reads.update(part["knn"])
            lib.log_phase("p4_probe", "layer resume-skip (fresh partial)", layer=layer)
            continue
        t0 = time.time()
        x_l = acts["raw"][keep, layer, :].astype(np.float32)
        ridge = ana.dof_capped_ridge_multi_y(
            x_l, y_k, fam_k, lambdas=lambdas, dof_cap=0.9, device=args.device
        )
        r2_layers[layer] = ridge["heldout_r2"]
        gcv_lambdas[str(layer)] = np.asarray(ridge["gcv_lambda"]).tolist()
        # Stand-in vectors at this layer (kept rows).
        stand = {
            "exact_dp": acts["base"][keep, layer, :].astype(np.float32),
            "prompt_dp": acts["ctxend"][keep, layer, :].astype(np.float32),
        }
        ctx_l = stand["prompt_dp"]
        pfx_l = acts["pfxend"][keep, layer, :].astype(np.float32)
        # LOFO id_bias from the pool rows themselves (plan §4 arm parity).
        sum_resid = np.zeros((len(fam_ids_sorted), DIM))
        counts = np.zeros(len(fam_ids_sorted))
        for gi, g in enumerate(fam_ids_sorted):
            m = fam_k == g
            sum_resid[gi] = (x_l[m].astype(np.float64) - ctx_l[m].astype(np.float64)).sum(axis=0)
            counts[gi] = m.sum()
        b_fam = ana.leave_one_group_out_bias(sum_resid, counts)  # (F, D)
        # Per-fold grid: probe(y^train) is the fold's held-out prediction.
        probe_grid = np.full((int(keep.sum()), len(GRID_ARMS), len(lib.TRAITS)), np.nan)
        for gi, g in enumerate(fam_ids_sorted):
            hold = fam_k == g
            fold = ridge["folds"][str(g)]
            w, b0 = fold["w"], fold["b0"]  # (D, T), (T,)
            p_train = ridge["heldout_pred"][hold]  # (n_h, T)
            probe_grid[hold, GRID_ARMS.index("raw")] = p_train
            for arm in ("exact_dp", "prompt_dp"):
                s = stand[arm][hold].astype(np.float64)
                probe_grid[hold, GRID_ARMS.index(arm)] = p_train - (s @ w + b0[None, :])
            for arm, src, mk in (
                ("mapped_ctx", ctx_l, "ctx"),
                ("mapped_pfx", pfx_l, "pfx"),
            ):
                probe_grid[hold, GRID_ARMS.index(arm)] = p_train - probe_mapped_standin(
                    src[hold], maps[mk], layer, w, b0
                )
            # id_bias: probe(ctxend + b_f) = probe(ctxend) + b_f @ w.
            bias_term = b_fam[gi] @ w  # (T,)
            probe_grid[hold, GRID_ARMS.index("id_bias")] = (
                probe_grid[hold, GRID_ARMS.index("prompt_dp")] - bias_term[None, :]
            )
        for ds in np.unique(ds_k):
            m = ds_k == ds
            grid_vals[ds_index[str(ds)], :, :, layer] = probe_grid[m].mean(axis=0)
        lib.log_phase(
            "p4_probe",
            f"layer {layer + 1}/{N_LAYERS}",
            layer=layer,
            r2=[round(float(v), 4) for v in ridge["heldout_r2"]],
            elapsed_s=round(time.time() - t0, 1),
        )
        layer_knn: dict[str, dict] = {}
        if layer in (args.knn_layers or []):
            for ti, trait in enumerate(lib.TRAITS):
                from explore_persona_space.analysis.mapping_baselines import knn_retrieval

                layer_knn[f"layer{layer}/{trait}"] = knn_retrieval(
                    ridge["heldout_pred"][:, ti : ti + 1],
                    y_k[:, ti : ti + 1],
                    metric="euclidean",
                )
        knn_reads.update(layer_knn)
        _save_probe_partial(
            ppath,
            probe_key,
            grid_layer=grid_vals[:, :, :, layer],
            r2_layer=r2_layers[layer],
            gcv=ridge["gcv_lambda"],
            knn=layer_knn,
        )
    # Dataset-level correlations vs the #778 y-axis (Form-A analogue of P3).
    y_axis = red.load_y_axis(datasets)
    records = []
    for ti, trait in enumerate(lib.TRAITS):
        yv = np.array([y_axis[trait][ds]["trait_score"] for ds in datasets])
        steer = red.STEER_IDX[trait]
        fam_idx_ds, _fams = red._family_index(datasets)
        for ai, arm in enumerate(GRID_ARMS):
            v = grid_vals[:, ai, ti, :]  # (n_ds, L)
            if np.isnan(v[:, steer]).all():
                continue
            r_layers = ana.pearson_r_cols(v, yv)
            rec = {
                "trait": trait,
                "arm": f"probe_A/{arm}",
                "layer_regime": "steer",
                "layer": steer,
                "r": float(r_layers[steer]),
                "r_per_layer": [None if np.isnan(x) else float(x) for x in r_layers],
                "n_datasets": len(datasets),
            }
            if not np.isnan(v).any():
                rec["lofo_sweep"] = ana.lofo_layer_sweep(v, yv, fam_idx_ds)
            records.append(rec)
    # Graded-vs-rate validation (plan §7 gate: rho > 0 on a held-out subset).
    rng = np.random.default_rng(args.seed)
    val = {}
    for ti, trait in enumerate(lib.TRAITS):
        ok = ~np.isnan(y[:, ti]) & ~np.isnan(rate[:, ti])
        idx = np.flatnonzero(ok)
        sub = rng.choice(idx, size=max(2, len(idx) // 5), replace=False)
        val[trait] = {
            "spearman_all": ana.spearman(y[ok, ti], rate[ok, ti]),
            "spearman_heldout_20pct": ana.spearman(y[sub, ti], rate[sub, ti]),
            "n_items": int(ok.sum()),
        }
    payload = {
        "note": (
            "EXPLORATORY Form-A probe (plan §5 probe_A): ridge a(y^train) -> graded score, "
            "LOFO over families, dof-capped GCV; difference grid probe(y^train) - "
            "probe(stand-in). Supervision: per-sample judge labels (P4 wave) — the "
            "learned-read-out vs contrastive-mean-diff comparison."
        ),
        "plan_ambiguity_resolution": (
            "probe X = raw_respavg (the judged training response's own activation), not "
            "base_respavg — see the module docstring; judged text = dataset.zip training "
            "responses per plan §6"
        ),
        "fit_regime": {
            "n_rows": int(keep.sum()),
            "n_dropped_rows_zero_kept_draws": n_dropped_rows,
            "n_missing_base_capture": acts["n_missing_base"],
            "n_train_min_per_fold": n_train_min,
            "d": DIM,
            "well_posed_n_gt_d": bool(n_train_min > DIM),
            "lambda_grid": [float(v) for v in lambdas],
            "dof_cap": 0.9,
        },
        "heldout_r2_per_layer": {
            lib.TRAITS[ti]: [None if np.isnan(v) else float(v) for v in r2_layers[:, ti]]
            for ti in range(len(lib.TRAITS))
        },
        "selected_lambda_per_layer_fold": gcv_lambdas,
        "records": records,
        "mapping_baselines": {
            "identity_bias": "inapplicable — probe maps d=3584 -> scalar (dim mismatch); "
            "stated per the standing rule, never silently skipped",
            "knn_retrieval": knn_reads,
            "knn_note": "euclidean on the scalar held-out predictions (pool = held-out "
            "targets, chance = k/n_pool per read); cosine inapplicable on scalars",
        },
        "graded_vs_rate_validation": val,
        "pool": {"split_hash": pool["split_hash"], "n_total": pool["n_total"]},
        **lib.run_metadata(),
    }
    lib.write_json_atomic(out_root / "form_a_probe.json", payload)
    lib.log_phase("p4_probe", "done", n_records=len(records))


# --- CLI ---------------------------------------------------------------------------


STAGES = ("rubrics", "pool", "pilot", "judge", "rejudge", "probe")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(lib.default_data_root()))
    ap.add_argument("--out-root", default=str(lib.REPO_ROOT / "eval_results" / "issue_2222"))
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--datasets", nargs="*", default=None, help="dataset selector; default all 24")
    ap.add_argument("--seed", type=int, default=lib.SUBSAMPLE_SEED)
    ap.add_argument(
        "--subsample-rows",
        type=int,
        default=lib.SUBSAMPLE_ROWS,
        help="P0 subsample S (locates the manifest the pool text comes from)",
    )
    ap.add_argument(
        "--rows-per-dataset",
        type=int,
        default=500,
        help="stratified pool rows per dataset (~12k total at 500 x 24, plan §9)",
    )
    ap.add_argument("--n-draws", type=int, default=6, help="judge draws per (item, trait)")
    ap.add_argument(
        "--judge-max-tokens", type=int, default=1024, help="judge response cap (plan >=1024)"
    )
    ap.add_argument(
        # #2124 satisfiability: 3 version arms x n_draws 2 x ceil(51/2) = 156 —
        # 150 realized exactly 50 draws/arm, one below the 51-draw floor at 2%.
        "--pilot-draws",
        type=int,
        default=156,
        help="rule-26 pilot target draws PER RUBRIC",
    )
    ap.add_argument(
        "--pilot-sync",
        action="store_true",
        help="let the pilot route sync (default forces the Batch path to match production)",
    )
    ap.add_argument("--skip-pilot-gate", action="store_true")
    ap.add_argument("--skip-upload", action="store_true", help="skip HF uploads (smoke only)")
    ap.add_argument(
        "--probe-layers",
        type=int,
        nargs="*",
        default=None,
        help="restrict the probe layer sweep (default all 28)",
    )
    ap.add_argument("--knn-layers", type=int, nargs="*", default=[15, 19])
    ap.add_argument("--device", default="cpu", help="ridge fit device (cpu | cuda)")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny sizes (rows/dataset, draws, pilot) — same code path throughout",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="execute deferred imports + args-attribute completeness check, then exit 0",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        raise SystemExit(0)
    kl = [int(v) for v in args.knn_layers or []]
    if len(set(kl)) != len(kl) or not all(0 <= v < N_LAYERS for v in kl):
        raise SystemExit(f"--knn-layers must be duplicate-free within [0, {N_LAYERS}): {kl}")
    if args.smoke:
        args.rows_per_dataset = min(args.rows_per_dataset, 4)
        args.n_draws = min(args.n_draws, 2)
        args.pilot_draws = min(args.pilot_draws, 12)
    data_root = Path(args.data_root)
    datasets = lib.dataset_ids(args.datasets)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    rubrics: dict[str, str] | None = None
    pool: dict | None = None

    def _rubrics() -> dict[str, str]:
        nonlocal rubrics
        if rubrics is None:
            rubrics = ensure_trait_rubrics(data_root)
        return rubrics

    def _pool() -> dict:
        nonlocal pool
        if pool is None:
            p = pool_path(data_root)
            if not p.exists():
                raise FileNotFoundError(f"{p} missing — run --stage pool first")
            pool = json.loads(p.read_text())
        return pool

    for name in stages:
        lib.log_phase("p4_stage_start", name, stage=name)
        if name == "rubrics":
            _rubrics()
        elif name == "pool":
            pool = build_pool(
                data_root, datasets, rows_per_dataset=args.rows_per_dataset, seed=args.seed
            )
            pool_path(data_root).parent.mkdir(parents=True, exist_ok=True)
            lib.write_json_atomic(pool_path(data_root), pool)
        elif name == "pilot":
            items = pool_items(data_root, _pool(), seed=args.seed, s_rows=args.subsample_rows)
            stage_pilot(args, _rubrics(), items, _pool())
        elif name == "judge":
            items = pool_items(data_root, _pool(), seed=args.seed, s_rows=args.subsample_rows)
            stage_judge(args, _rubrics(), items, _pool())
        elif name == "rejudge":
            items = pool_items(data_root, _pool(), seed=args.seed, s_rows=args.subsample_rows)
            stage_rejudge(args, _rubrics(), items)
        elif name == "probe":
            stage_probe(args, _pool())
    lib.log_phase("p4_done", "P4 complete", stages=stages)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def _import_check() -> None:
    """Axis-1 import resolution: execute every deferred/function-body import."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    import urllib.request  # noqa: F401

    import issue778_lib  # noqa: F401  (ensure_trait_rubrics)
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: F401
    from explore_persona_space.eval.graded_judge import judge_graded  # noqa: F401
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401
    from explore_persona_space.orchestrate import hub  # noqa: F401  (lib.upload_file)

    print("[import-check] issue2222_judge OK")


if __name__ == "__main__":
    raise SystemExit(main())
