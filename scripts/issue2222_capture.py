"""P1/P2 GPU driver for issue #2222 — batched teacher-forced capture + vLLM base generation.

Phases (plan v5 §4; names verbatim from the pipeline DAG):

- ``p2_gen``     — vLLM base-model generation (1 sample/prompt, temp 1.0, cap
  1024), per-dataset cap-hit fraction (``finish_reason == "length"``) with ONE
  2x-cap re-gen of capped rows (#1332); rollout TEXT persisted per dataset
  (``raw_completions/exact_dp_base_gen/<ds>.jsonl``) + uploaded to HF the
  moment the dataset completes (#664 per-cell contract). A cap-hit fraction
  still above the 2% bar AFTER the re-gen is the plan §7 exact-ΔP halt:
  designed artifact-routed halt, report JSON + rc ``RC_CAP_HIT``.
- ``p1_capture`` — batched teacher-forced forwards (issue_1739
  ``capture_batch``, batch 8, right-pad, per-segment token-id concatenation)
  over the fixed subsample -> per-row all-28-layer fp16 summaries
  ``{raw_respavg, ctxend, pfxend}`` (``ctxend`` == the paper's ``plast``
  position — plan §4 position identity; ``raw_respavg`` == #1739 ``t1``).
- ``p2_capture`` — teacher-forced capture over (prompt, base generation) ->
  ``base_respavg`` (the exact-ΔP stand-in).

Execution order under ``--phase all``: gen -> capture. The vLLM engine lives
and is fully reaped BEFORE the single HF capture-model load (one framework
switch instead of two; plan phase NAMES and outputs unchanged — recorded as an
ordering note in the unit-1 report).

Checkpoint grain: per dataset — npz/JSONL written atomically + uploaded the
moment a dataset completes. Resume: manifest fingerprint match (split hash +
config + code fingerprints; plan §9 — bare file existence never vouches);
HF-complete datasets are skipped too (fresh-pod resume).

``--pilot``: warmup + ONE TIMED production-shape batch (batch 8, all 28
layers), writes ``pilot_report.json``; rc ``RC_PILOT_OVER`` when the
extrapolated wall exceeds 4x the §9 booked wall (plan §7 kill criterion).

CONTENT HYGIENE: dataset rows + base generations include harmful content —
logs carry ids / counts / hashes only; text is PERSISTED (JSONL), never printed.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

# Pre-vllm-import (#628): main() touches tokenizers/transformers before LLM(),
# and fork()-ed EngineCore workers die silently on the poisoned parent state.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# load_dotenv BEFORE any heavy import (numpy below, torch/vllm transitively):
# the shared-VM thread caps bind in-process only pre-import (#847;
# tests/test_shared_vm_thread_caps.py).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2222_lib as lib  # noqa: E402
from explore_persona_space.experiments.issue_1739 import capture as cap1739  # noqa: E402
from explore_persona_space.experiments.issue_1739 import generation as gen1739  # noqa: E402

logger = logging.getLogger("issue2222")

# §9 booked walls (P1 1.5 + P2 gen 0.7 + P2 capture 1.5); §7 pilot bar is 4x.
BOOKED_CAPTURE_WALL_H = 3.0
BOOKED_GEN_WALL_H = 0.7
PILOT_OVER_MULT = 4.0
RC_PILOT_OVER = 7  # designed pilot-gate halt (report JSON + distinct rc; gotchas.md #1415)
RC_CAP_HIT = 8  # designed §7 exact-ΔP halt: cap-hit > 2% persisting after the 2x re-gen
PRODUCTION_N_ROWS = 24 * lib.SUBSAMPLE_ROWS  # full-run row count the pilot extrapolates to


# --- Manifest helpers ----------------------------------------------------------


def _manifest_path(ds_dir: Path) -> Path:
    return ds_dir / "manifest.json"


def _read_manifest(ds_dir: Path) -> dict | None:
    path = _manifest_path(ds_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _update_manifest(ds_dir: Path, base: dict, phase: str, payload: dict) -> dict:
    """Set one phase block; a fingerprint-mismatched existing manifest is REPLACED
    wholesale (stale phase blocks must not survive a regime change)."""
    manifest = _read_manifest(ds_dir)
    if manifest is None or manifest.get("resume_fingerprint") != base["resume_fingerprint"]:
        manifest = dict(base, phases={})
    manifest["phases"][phase] = payload
    manifest["meta"] = lib.run_metadata()
    lib.write_json_atomic(_manifest_path(ds_dir), manifest)
    return manifest


_PHASE_LOCAL_FILES = {
    "p2_gen": lambda root, ds: [lib.rawcomp_path(root, ds)],
    "p1_capture": lambda root, ds: [lib.capture_dir(root, ds) / "summaries.npz"],
    "p2_capture": lambda root, ds: [lib.capture_dir(root, ds) / "base_respavg.npz"],
}
_PHASE_HUB_FILES = {
    "p2_gen": lambda ds: [lib.hf_rawcomp_path(ds)],
    "p1_capture": lambda ds: [f"{lib.hf_capture_prefix(ds)}/summaries.npz"],
    "p2_capture": lambda ds: [f"{lib.hf_capture_prefix(ds)}/base_respavg.npz"],
}


def _check_cap_hit_halt(data_root: Path, phase: str, *, override: bool) -> None:
    """Refuse ``--phase capture/all`` while a prior run's §7 cap-hit halt record
    exists (round-2 BLOCKER fix): downstream phases never re-check cap_hit_final,
    so proceeding past the halt file would capture over cap-biased generations.
    ``--phase gen`` is exempt (it re-derives the halt itself, resume included);
    ``--override-cap-hit-halt`` is the deliberate escape."""
    halt_path = Path(data_root) / "cap_hit_halt.json"
    if phase == "gen" or override or not halt_path.exists():
        return
    raise SystemExit(
        f"{halt_path} exists — a prior run halted on the §7 cap-hit criterion "
        f"(rc={RC_CAP_HIT}). Re-run --phase gen after fixing (halted datasets "
        "re-generate on a fingerprint change), or pass --override-cap-hit-halt "
        "to proceed deliberately."
    )


def _retire_cap_hit_halt(data_root: Path, ds_ids: list[str]) -> None:
    """Retire a stale cap_hit_halt.json after a clean gen pass, ONLY when this
    run's ``ds_ids`` cover every dataset the halt records (round-2 CONCERN
    cap-hit-halt-retirement-subset-scope): a subset ``--phase gen --datasets``
    run must not retire another dataset's halt — a later standalone
    ``--phase capture`` would then pass the ``_check_cap_hit_halt`` gate and
    capture over that dataset's cap-biased generations. A halt file without a
    ``datasets`` list is a foreign/stale shape — fail loud (KeyError)."""
    halt_path = Path(data_root) / "cap_hit_halt.json"
    if not halt_path.exists():
        return
    halt = json.loads(halt_path.read_text())
    uncovered = set(halt["datasets"]) - set(ds_ids)
    if uncovered:
        lib.log_phase(
            "halt_cap_hit",
            "cap_hit_halt.json retained — halt names datasets outside this run",
            uncovered=sorted(uncovered),
        )
        return
    halt_path.unlink()
    lib.log_phase("halt_cap_hit", "stale cap_hit_halt.json cleared — gen under the bar")


def _phase_done(data_root: Path, ds: str, phase: str, base: dict, *, hub_resume: bool) -> bool:
    """Resume predicate: fingerprint-matched manifest + realized files, local or HF."""
    ds_dir = lib.capture_dir(data_root, ds)
    manifest = _read_manifest(ds_dir)
    if (
        manifest is not None
        and manifest.get("resume_fingerprint") == base["resume_fingerprint"]
        and phase in manifest.get("phases", {})
        and all(p.exists() for p in _PHASE_LOCAL_FILES[phase](data_root, ds))
    ):
        return True
    if not hub_resume:
        return False
    hub_manifest = lib.fetch_hub_manifest(ds)
    if (
        hub_manifest is not None
        and hub_manifest.get("resume_fingerprint") == base["resume_fingerprint"]
        and phase in hub_manifest.get("phases", {})
        and all(lib.hub_file_exists(p) for p in _PHASE_HUB_FILES[phase](ds))
    ):
        # Mirror the HF manifest locally so later phases see the completed block.
        ds_dir.mkdir(parents=True, exist_ok=True)
        lib.write_json_atomic(_manifest_path(ds_dir), hub_manifest)
        lib.log_phase("resume", "dataset phase complete on HF — skipping", dataset=ds, p=phase)
        return True
    return False


def _savez_atomic(path: Path, **arrays) -> None:
    """Uncompressed npz via tmp+replace; tmp keeps the .npz suffix (np.savez
    APPENDS .npz to any other name — gotchas.md #1092)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.stem}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# --- Row assembly ---------------------------------------------------------------


def _rendered_rows(tokenizer, rows: list[tuple[int, str, str]], ds: str) -> list[dict]:
    """Render every subsample row once: (row_id, prefix, prompt, completion, sha)."""
    out: list[dict] = []
    for row_id, question, answer in rows:
        prefix, prompt = lib.render_row(tokenizer, question)
        out.append(
            {
                "row_id": row_id,
                "prefix": prefix,
                "prompt": prompt,
                "completion": answer,
                "prompt_sha256": lib.sha256_text(prompt),
                "label": f"{ds}:{row_id}",
            }
        )
    return out


# --- p2_gen ---------------------------------------------------------------------


def run_gen(
    data_root: Path,
    ds_ids: list[str],
    subsamples: dict[str, tuple[dict, list]],
    tokenizer,
    base: dict[str, dict],
    *,
    seed: int,
    skip_upload: bool,
    hub_resume: bool,
) -> list[str]:
    """vLLM base generation per dataset; returns datasets whose cap-hit persisted."""
    halted: list[str] = []
    t0 = time.time()
    for k, ds in enumerate(ds_ids):
        if _phase_done(data_root, ds, "p2_gen", base[ds], hub_resume=hub_resume):
            # Round-2 BLOCKER fix: the halted dataset's p2_gen phase is manifest-
            # COMPLETE (artifacts persisted BEFORE the §7 halt), so a relaunch
            # lands here — re-derive the halt from the recorded cap_hit_final
            # instead of silently proceeding over cap-biased generations.
            phase_blk = (
                (_read_manifest(lib.capture_dir(data_root, ds)) or {})
                .get("phases", {})
                .get("p2_gen", {})
            )
            if "cap_hit_final" not in phase_blk:
                raise RuntimeError(
                    f"{ds}: resumed p2_gen manifest carries no cap_hit_final — "
                    "foreign/stale manifest shape (re-run --phase gen)"
                )
            cap_hit = float(phase_blk["cap_hit_final"])
            if cap_hit > lib.CAP_HIT_MAX_FRACTION:
                halted.append(ds)
                lib.log_phase(
                    "p2_gen",
                    "resume-skip carries the §7 cap-hit halt",
                    dataset=ds,
                    cap_hit_final=round(cap_hit, 4),
                )
            else:
                lib.log_phase("p2_gen", "resume-skip", dataset=ds)
            continue
        _manifest, rows = subsamples[ds]
        rendered = _rendered_rows(tokenizer, rows, ds)
        prompts = [r["prompt"] for r in rendered]
        seeds = [gen1739._context_seed(seed, r["label"]) for r in rendered]
        outs = gen1739._default_vllm_generate(
            prompts,
            n=1,
            temperature=lib.GEN_TEMPERATURE,
            max_tokens=lib.GEN_MAX_NEW_TOKENS,
            seeds=seeds,
        )
        recs: list[dict] = []
        for r, s, out in zip(rendered, seeds, outs, strict=True):
            recs.append(
                {
                    "row_id": r["row_id"],
                    "dataset": ds,
                    "prompt_sha256": r["prompt_sha256"],
                    "seed": s,
                    "gen_pass": 1,
                    "max_tokens": lib.GEN_MAX_NEW_TOKENS,
                    "finish_reason": out[0]["finish_reason"],
                    "completion": out[0]["text"],
                }
            )
        capped = [i for i, rec in enumerate(recs) if rec["finish_reason"] == "length"]
        cap_hit_initial = len(capped) / max(1, len(recs))
        n_regen = n_regen_skipped = 0
        if cap_hit_initial > lib.CAP_HIT_MAX_FRACTION:
            # #1332: one re-gen of the capped rows at 2x cap (same per-row seeds —
            # the prefill-continuation direction, never a subset-resample).
            regen_budget = 2 * lib.GEN_MAX_NEW_TOKENS
            eligible = []
            for i in capped:
                n_prompt = len(
                    tokenizer(rendered[i]["prompt"], add_special_tokens=False)["input_ids"]
                )
                if n_prompt + regen_budget + 16 <= gen1739.MAX_MODEL_LEN:
                    eligible.append(i)
                else:
                    n_regen_skipped += 1
            lib.log_phase(
                "p2_gen",
                "cap-hit over bar — re-generating capped rows at 2x cap",
                dataset=ds,
                cap_hit_initial=round(cap_hit_initial, 4),
                n_regen=len(eligible),
                n_regen_skipped_budget=n_regen_skipped,
            )
            outs2 = gen1739._default_vllm_generate(
                [prompts[i] for i in eligible],
                n=1,
                temperature=lib.GEN_TEMPERATURE,
                max_tokens=regen_budget,
                seeds=[seeds[i] for i in eligible],
            )
            for i, out in zip(eligible, outs2, strict=True):
                recs[i].update(
                    {
                        "gen_pass": 2,
                        "max_tokens": regen_budget,
                        "finish_reason": out[0]["finish_reason"],
                        "completion": out[0]["text"],
                    }
                )
            n_regen = len(eligible)
        n_capped_final = sum(1 for rec in recs if rec["finish_reason"] == "length")
        cap_hit_final = n_capped_final / max(1, len(recs))
        n_empty = sum(1 for rec in recs if not rec["completion"].strip())

        jsonl_path = lib.rawcomp_path(data_root, ds)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = jsonl_path.with_name(jsonl_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, jsonl_path)

        ds_dir = lib.capture_dir(data_root, ds)
        _update_manifest(
            ds_dir,
            base[ds],
            "p2_gen",
            {
                "n_rows": len(recs),
                "cap_hit_initial": cap_hit_initial,
                "cap_hit_final": cap_hit_final,
                "n_regen": n_regen,
                "n_regen_skipped_budget": n_regen_skipped,
                "n_empty_completions": n_empty,
                "jsonl": str(jsonl_path),
            },
        )
        if not skip_upload:
            # UPLOAD_LOOP_EXEMPT: #664 per-cell upload  # NO_RETRY: lib.upload_file retries
            lib.upload_file(jsonl_path, lib.hf_rawcomp_path(ds))
            # UPLOAD_LOOP_EXEMPT: #664 per-cell upload  # NO_RETRY: lib.upload_file retries
            lib.upload_file(_manifest_path(ds_dir), f"{lib.hf_capture_prefix(ds)}/manifest.json")
        print(
            f"[p2_gen] unit {k + 1}/{len(ds_ids)} {ds} n={len(recs)} "
            f"cap_hit={cap_hit_final:.4f} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        if cap_hit_final > lib.CAP_HIT_MAX_FRACTION:
            halted.append(ds)
    return halted


def _reap_gen_engine() -> None:
    """Fully reap the module-cached vLLM engine before any HF model load
    (gotchas.md vLLM teardown recipe; engine cache is generation's module dict)."""
    llm = gen1739._TOKENIZER_CACHE.pop("_llm", None)
    if llm is None:
        return
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 — cache drain is best-effort on CPU hosts
        logger.warning("post-reap cuda cache drain skipped: %s", e)


# --- p1_capture / p2_capture ----------------------------------------------------


def _stack_kind(summaries: list[dict], kind: str) -> np.ndarray:
    arr = np.stack([s[kind] for s in summaries], axis=0).astype(np.float16)
    n_layers, hidden = arr.shape[1], arr.shape[2]
    assert (n_layers, hidden) == lib.RB_SHAPE, arr.shape
    return arr


def run_capture(
    data_root: Path,
    ds_ids: list[str],
    subsamples: dict[str, tuple[dict, list]],
    tokenizer,
    base: dict[str, dict],
    *,
    batch_size: int,
    skip_upload: bool,
    hub_resume: bool,
) -> None:
    """P1 (train-row summaries) + P2 (base_respavg) captures, one HF model load."""
    model = cap1739.load_capture_model()
    t0 = time.time()
    try:
        for k, ds in enumerate(ds_ids):
            ds_dir = lib.capture_dir(data_root, ds)
            _manifest, rows = subsamples[ds]
            rendered = _rendered_rows(tokenizer, rows, ds)

            if not _phase_done(data_root, ds, "p1_capture", base[ds], hub_resume=hub_resume):
                summaries, positions = cap1739.capture_batch(
                    [r["prefix"] for r in rendered],
                    [r["prompt"] for r in rendered],
                    [r["completion"] for r in rendered],
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    log_label=f"p1:{ds}",
                )
                out_path = ds_dir / "summaries.npz"
                _savez_atomic(
                    out_path,
                    raw_respavg=_stack_kind(summaries, "t1"),
                    ctxend=_stack_kind(summaries, "context_end"),
                    pfxend=_stack_kind(summaries, "prefix_end"),
                    row_ids=np.asarray([r["row_id"] for r in rendered], dtype=np.int64),
                    n_prompt_tokens=np.asarray([p["n_prompt"] for p in positions], dtype=np.int32),
                )
                _update_manifest(
                    ds_dir, base[ds], "p1_capture", {"n_rows": len(rendered), "npz": str(out_path)}
                )
                if not skip_upload:
                    # UPLOAD_LOOP_EXEMPT: #664 per-cell upload  # NO_RETRY: lib.upload_file retries
                    lib.upload_file(out_path, f"{lib.hf_capture_prefix(ds)}/summaries.npz")
                    # UPLOAD_LOOP_EXEMPT: #664 per-cell upload  # NO_RETRY: lib.upload_file retries
                    lib.upload_file(
                        _manifest_path(ds_dir), f"{lib.hf_capture_prefix(ds)}/manifest.json"
                    )
                print(
                    f"[p1_capture] unit {k + 1}/{len(ds_ids)} {ds} n={len(rendered)} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            else:
                lib.log_phase("p1_capture", "resume-skip", dataset=ds)

            if not _phase_done(data_root, ds, "p2_capture", base[ds], hub_resume=hub_resume):
                _capture_base(
                    data_root,
                    ds,
                    rendered,
                    model,
                    tokenizer,
                    base[ds],
                    batch_size=batch_size,
                    skip_upload=skip_upload,
                )
                print(
                    f"[p2_capture] unit {k + 1}/{len(ds_ids)} {ds} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            else:
                lib.log_phase("p2_capture", "resume-skip", dataset=ds)
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — cache drain is best-effort on CPU hosts
            logger.warning("post-capture cuda cache drain skipped: %s", e)


def _ensure_local_gen(data_root: Path, ds: str) -> Path:
    """The dataset's gen JSONL, downloading the HF copy when only hub-complete."""
    from explore_persona_space.orchestrate import hub

    path = lib.rawcomp_path(data_root, ds)
    if path.exists():
        return path
    lib.log_phase("p2_capture", "gen JSONL absent locally — staging HF copy", dataset=ds)
    hub.stage_hub_file(lib.HF_DATA_REPO, lib.hf_rawcomp_path(ds), path, repo_type="dataset")
    return path


def _capture_base(
    data_root: Path,
    ds: str,
    rendered: list[dict],
    model,
    tokenizer,
    base_fields: dict,
    *,
    batch_size: int,
    skip_upload: bool,
) -> None:
    """Teacher-forced base_respavg over (prompt, base generation) for one dataset."""
    recs = lib.read_jsonl(_ensure_local_gen(data_root, ds))
    by_row = {r["row_id"]: r for r in rendered}
    # g1 minor fix: a SUBSET gen JSONL (partial/stale generation output) must not
    # pass silently on the --phase capture path — require full id coverage (the
    # per-record loop below enforces the ⊆ direction, so together this is ==).
    missing = sorted(set(by_row) - {rec["row_id"] for rec in recs})
    if missing:
        raise RuntimeError(
            f"{ds}: gen JSONL covers {len(by_row) - len(missing)}/{len(by_row)} subsample "
            f"rows — {len(missing)} rendered rows missing (first: {missing[:5]}); "
            "stale/partial generation output for this fingerprint (re-run --phase gen)"
        )
    keep: list[tuple[dict, dict]] = []  # (rendered row, gen record)
    n_empty = n_budget = 0
    for rec in recs:
        row = by_row.get(rec["row_id"])
        if row is None:
            raise RuntimeError(
                f"{ds}: gen JSONL row_id {rec['row_id']} not in the current subsample — "
                "stale generation output for this fingerprint (re-run --phase gen)"
            )
        if row["prompt_sha256"] != rec["prompt_sha256"]:
            raise RuntimeError(
                f"{ds}: prompt sha mismatch for row {rec['row_id']} — the render/template "
                "drifted between gen and capture (fingerprint should have caught this)"
            )
        if not rec["completion"].strip():
            n_empty += 1  # zero-width answer span would silently average the boundary
            continue
        if (
            lib.admit_row(tokenizer, row["prefix"], row["prompt"], rec["completion"], row["label"])
            is None
        ):
            n_budget += 1  # 2x-cap re-gen rows can exceed the capture budget
            continue
        keep.append((row, rec))
    if not keep:
        raise RuntimeError(f"{ds}: zero capturable base generations (empty {n_empty})")
    summaries, _positions = cap1739.capture_batch(
        [row["prefix"] for row, _ in keep],
        [row["prompt"] for row, _ in keep],
        [rec["completion"] for _, rec in keep],
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        log_label=f"p2:{ds}",
    )
    ds_dir = lib.capture_dir(data_root, ds)
    out_path = ds_dir / "base_respavg.npz"
    _savez_atomic(
        out_path,
        base_respavg=_stack_kind(summaries, "t1"),
        row_ids=np.asarray([row["row_id"] for row, _ in keep], dtype=np.int64),
        finish_length=np.asarray([rec["finish_reason"] == "length" for _, rec in keep], dtype=bool),
    )
    _update_manifest(
        ds_dir,
        base_fields,
        "p2_capture",
        {
            "n_rows": len(keep),
            "n_skipped_empty": n_empty,
            "n_skipped_budget": n_budget,
            "npz": str(out_path),
        },
    )
    if not skip_upload:
        # NO_RETRY: lib.upload_file wraps hub._upload in its own bounded retry (#1315 seam)
        lib.upload_file(out_path, f"{lib.hf_capture_prefix(ds)}/base_respavg.npz")
        # NO_RETRY: lib.upload_file wraps hub._upload in its own bounded retry (#1315 seam)
        lib.upload_file(_manifest_path(ds_dir), f"{lib.hf_capture_prefix(ds)}/manifest.json")


# --- Pilot ----------------------------------------------------------------------


def run_pilot(
    data_root: Path,
    ds_ids: list[str],
    subsamples: dict[str, tuple[dict, list]],
    tokenizer,
    *,
    batch_size: int,
) -> int:
    """Warmup + one TIMED production-shape capture batch; §7 pilot gate (rc 7).

    The MEASURED per-batch wall is the §9 P1/P2 sizing basis (plan A13 is
    LOW-confidence); extrapolation covers BOTH capture passes at full
    production scale (24 datasets x S rows x 2 passes).
    """
    ds = ds_ids[0]
    _manifest, rows = subsamples[ds]
    if len(rows) < 2 * batch_size:
        raise SystemExit(f"--pilot needs >= {2 * batch_size} subsample rows; got {len(rows)}")
    rendered = _rendered_rows(tokenizer, rows[: 2 * batch_size], ds)
    model = cap1739.load_capture_model()

    def _one_batch(rs: list[dict]) -> float:
        t0 = time.time()
        cap1739.capture_batch(
            [r["prefix"] for r in rs],
            [r["prompt"] for r in rs],
            [r["completion"] for r in rs],
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            log_label="pilot",
        )
        return time.time() - t0

    warmup_s = _one_batch(rendered[:batch_size])  # CUDA warmup/compile rides batch 1
    timed_s = _one_batch(rendered[batch_size : 2 * batch_size])
    n_batches_production = 2 * -(-PRODUCTION_N_ROWS // batch_size)  # P1 + P2 capture passes
    projected_capture_h = timed_s * n_batches_production / 3600.0
    projected_total_h = projected_capture_h + BOOKED_GEN_WALL_H
    booked_total_h = BOOKED_CAPTURE_WALL_H + BOOKED_GEN_WALL_H
    ratio = projected_total_h / booked_total_h
    report = {
        "dataset": ds,
        "batch_size": batch_size,
        "warmup_batch_s": round(warmup_s, 3),
        "timed_batch_s": round(timed_s, 3),
        "n_batches_production": n_batches_production,
        "projected_capture_h": round(projected_capture_h, 3),
        "projected_total_h": round(projected_total_h, 3),
        "booked_total_h": booked_total_h,
        "ratio_vs_booked": round(ratio, 3),
        "verdict": "OVER_4X" if ratio > PILOT_OVER_MULT else "OK",
        "meta": lib.run_metadata(),
    }
    lib.write_json_atomic(Path(data_root) / "pilot_report.json", report)
    lib.log_phase("pilot", "pilot gate", **{k: v for k, v in report.items() if k != "meta"})
    return RC_PILOT_OVER if report["verdict"] == "OVER_4X" else 0


# --- main -----------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default=str(lib.default_data_root()))
    parser.add_argument("--datasets", nargs="*", default=None, help="families and/or dataset ids")
    parser.add_argument("--subsample", type=int, default=lib.SUBSAMPLE_ROWS)
    parser.add_argument("--seeds", default=str(lib.SUBSAMPLE_SEED), help="comma list; exactly one")
    parser.add_argument("--phase", choices=("all", "gen", "capture"), default="all")
    parser.add_argument("--batch-size", type=int, default=cap1739.DEFAULT_CAPTURE_BATCH_SIZE)
    parser.add_argument("--pilot", action="store_true", help="timed 1-batch pilot, then exit")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="stage + fingerprint, then exit before any GPU work (arg/staging probe)",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="VM smoke only — pod runs upload"
    )
    parser.add_argument(
        "--override-cap-hit-halt",
        action="store_true",
        help="proceed past an existing cap_hit_halt.json (deliberate §7 override)",
    )
    parser.add_argument("--no-hub-resume", action="store_true")
    parser.add_argument(
        "--sentinel-dir", default=None, help="default: /workspace/logs when present"
    )
    parser.add_argument("--sentinel-version", type=int, default=1)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) != 1:
        raise SystemExit(f"--seeds must name exactly ONE seed (fixed-subsample design): {seeds}")
    seed = seeds[0]
    data_root = Path(args.data_root)
    _check_cap_hit_halt(data_root, args.phase, override=args.override_cap_hit_halt)
    ds_ids = lib.dataset_ids(args.datasets)
    hub_resume = not args.no_hub_resume and not args.skip_upload

    # Self-staging on a fresh pod: gitignored data/ does not travel with the
    # branch clone (gotchas.md #654) — stage the pinned corpus + subsample here.
    lib.stage_dataset_zip(data_root)
    tokenizer = gen1739.get_tokenizer()
    subsamples = {
        ds: lib.ensure_subsample(data_root, ds, tokenizer, seed=seed, s_rows=args.subsample)
        for ds in ds_ids
    }
    cfg = lib.run_config(seed, args.subsample, args.batch_size)
    cfg_fp = lib.config_fingerprint(cfg)
    base = {
        ds: {
            "dataset": ds,
            "seed": seed,
            "s_rows": args.subsample,
            "split_hash": subsamples[ds][0]["split_hash"],
            "config_fingerprint": cfg_fp,
            "code_fingerprint": lib.code_fingerprint(),
            "resume_fingerprint": lib.resume_fingerprint(
                subsamples[ds][0]["split_hash"],
                cfg_fp,
                subsamples[ds][0]["dataset_file_sha256"],
            ),
            "config": cfg,
        }
        for ds in ds_ids
    }
    lib.log_phase(
        "setup",
        "driver ready",
        n_datasets=len(ds_ids),
        seed=seed,
        s_rows=args.subsample,
        config_fingerprint=cfg_fp[:16],
    )

    if args.setup_only:
        lib.log_phase("setup_only", "exiting before GPU phases (arg/staging probe)")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    if args.pilot:
        rc = run_pilot(data_root, ds_ids, subsamples, tokenizer, batch_size=args.batch_size)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)

    halted: list[str] = []
    if args.phase in ("all", "gen"):
        halted = run_gen(
            data_root,
            ds_ids,
            subsamples,
            tokenizer,
            base,
            seed=seed,
            skip_upload=args.skip_upload,
            hub_resume=hub_resume,
        )
        _reap_gen_engine()
        if halted:
            # Plan §7: cap-hit > 2% persisting after the one 2x re-gen biases the
            # exact-ΔP reference arm — designed halt AFTER persisting artifacts.
            report = {"halt": "cap_hit_over_bar_after_regen", "datasets": halted}
            lib.write_json_atomic(data_root / "cap_hit_halt.json", report)
            lib.log_phase("halt_cap_hit", "exact-ΔP reference biased — halting", **report)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(RC_CAP_HIT)
        else:
            # A clean gen pass (fresh fingerprint after a config fix) retires the
            # stale halt record so later --phase capture runs are not refused —
            # scoped to the datasets the halt records (subset-safe).
            _retire_cap_hit_halt(data_root, ds_ids)

    if args.phase in ("all", "capture"):
        run_capture(
            data_root,
            ds_ids,
            subsamples,
            tokenizer,
            base,
            batch_size=args.batch_size,
            skip_upload=args.skip_upload,
            hub_resume=hub_resume,
        )

    digest = {
        "phase_arg": args.phase,
        "datasets": ds_ids,
        "seed": seed,
        "s_rows": args.subsample,
        "config_fingerprint": cfg_fp,
        "per_dataset": {
            ds: (_read_manifest(lib.capture_dir(data_root, ds)) or {}).get("phases", {})
            for ds in ds_ids
        },
        "upload_skipped": args.skip_upload,
    }
    sentinel_dir = Path(args.sentinel_dir) if args.sentinel_dir else Path("/workspace/logs")
    if args.sentinel_dir or sentinel_dir.exists():
        path = lib.write_results_sentinel(args.sentinel_version, digest, logs_dir=sentinel_dir)
        lib.log_phase("sentinel", "results sentinel written", path=str(path))
    lib.log_phase("done", "P1/P2 driver complete")
    sys.stdout.flush()
    sys.stderr.flush()
    # Generation drivers terminate via os._exit AFTER durables land: finalize-time
    # multiprocessing cleanup can deadlock on vLLM worker children (#1739/#2149).
    os._exit(0)


if __name__ == "__main__":
    main()
