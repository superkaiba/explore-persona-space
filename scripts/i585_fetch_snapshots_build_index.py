#!/usr/bin/env python3
"""Task #585 — fetch the verified per-fraction #504 v4 smoke snapshots + inputs.

Standalone glue (stdlib + ``huggingface_hub`` + ``dotenv`` only — NO project
``src`` imports, so it runs unmodified from the pinned issue-534 checkout,
plan #585 v2 section 4.2 Step 1.1). Mirrors ``_run_v4_phase0_reeval``'s index
construction verbatim:

  * per-file ``hf_hub_download`` of ``adapter_config.json`` +
    ``adapter_model.safetensors`` for the 6 fractions (``snapshot_download`` +
    ``allow_patterns`` is the known-broken shape on hf_hub 0.36.2 — it silently
    matches zero files for nested subfolder globs);
  * frac token ``"1.00"`` for 1.0 else ``f"{frac:.2f}"``;
  * writes ``checkpoint_index.json`` of ``{frac_str: {"step": None, "path": str}}``.

Fail-loud asserts (plan section 4.2 Step 1.1 + assumptions A1/A2/A9):

  * the 6 local ``adapter_model.safetensors`` sha256s are pairwise DISTINCT
    (bytewise-distinct adapters — rules out a re-upload collapse);
  * every ``adapter_config.json`` is gauge-free for the logit readout
    (``target_modules`` excludes ``lm_head``/``embed_tokens``;
    ``modules_to_save`` empty) and matches the recorded recipe (r=8, alpha=32);
  * the persona bank downloaded from the HF data repo matches the plan-time
    verified content hash (assumption A9 made a runtime assert);
  * ``R_eval_v504.json``'s internal ``content_hash`` matches the plan-time
    verified prefix (plan section 4.2 Step 0).

Every HF download retries 3 times with exponential backoff (10/30/90 s) on
transient flakes, then fails loud.

step6to12 follow-up extension (plan #585 v3 section 4.2 Step 1.2 — ADDITIVE
flags only; default behavior byte-identical):

  * ``--fractions 0.08,0.16`` — download only a SUBSET of the module default 6
    Hub fractions (the endpoint-parity controls).
  * ``--merge-retrain-manifest PATH`` — merge the Phase-T retrain manifest's 7
    window snapshots (steps 6-12, 4-dp keys) with the Hub entries into ONE
    merged ``checkpoint_index.json`` whose float-parseable keys sort in step
    order, with the ``step`` field filled for ALL entries (Hub steps recomputed
    with the EXACT first-crossing float semantics of
    ``CheckpointAtFractionsCallback``), plus an ``index_provenance.json``
    sidecar ``{key: {step, provenance: retrain|hub}}``. The gauge-free
    adapter-config assert + the pairwise-distinct sha256 assert extend over
    ALL merged entries; retrain sha256s are cross-checked against the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i585.fetch_snapshots")

FRACTIONS = (0.08, 0.16, 0.33, 0.50, 0.75, 1.00)
PRETRAIN_SUBFOLDER = "adapters/issue_504_v4/c504v4_smoke_eps3_seed42"
DEFAULT_MODEL_REPO = "superkaiba1/explore-persona-space"
DEFAULT_DATA_REPO = "superkaiba1/explore-persona-space-data"
# NOTE: the bank's ACTUAL location on the data repo (verified via
# list_repo_files 2026-06-11) — the in-repo path differs from the
# persona_bank.py module constant ("issue472_neg_geometry/persona_bank.json"),
# which 404s; the upload landed under the geometry/ subdir.
BANK_PATH_IN_REPO = "issue472_neg_geometry/geometry/persona_bank.json"
R_EVAL_PATH_IN_REPO = "issue504_geometry/on_policy_R/R_eval_v504.json"

# Plan-time verified references (plan #585 v2 section 4.2 Step 0 + section 12 A9).
EXPECTED_BANK_CONTENT_HASH = "aec53e06dcb68f46412620de4f891fb367e8f0a672347824621d081ff97e05fc"
EXPECTED_R_EVAL_CONTENT_HASH_PREFIX = "7ebdac18e6eb"
EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
EXPECTED_LORA_R = 8
EXPECTED_LORA_ALPHA = 32

RETRY_DELAYS_S = (10, 30, 90)


def frac_token(frac: float) -> str:
    """HF subfolder token: ``"1.00"`` for 1.0 else ``f"{frac:.2f}"`` (plan section 4.2)."""
    return "1.00" if abs(frac - 1.0) < 1e-6 else f"{frac:.2f}"


def _download_with_retry(repo_id: str, repo_type: str, filename: str, local_dir: Path) -> Path:
    """Per-file ``hf_hub_download`` with 3-attempt exponential backoff; fail loud after."""
    from huggingface_hub import hf_hub_download

    last_exc: Exception | None = None
    for attempt, delay in enumerate((0, *RETRY_DELAYS_S)):
        if delay:
            log.warning(
                "retrying %s (attempt %d/%d) after %ds: %s",
                filename,
                attempt + 1,
                len(RETRY_DELAYS_S) + 1,
                delay,
                last_exc,
            )
            time.sleep(delay)
        try:
            out = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=str(local_dir),
                token=os.environ.get("HF_TOKEN"),
            )
            log.info("[phase=fetch] downloaded %s -> %s", filename, out)
            return Path(out)
        except Exception as exc:  # transient HF flake — retried, then re-raised
            last_exc = exc
    raise RuntimeError(
        f"hf_hub_download failed for {repo_type}:{repo_id}/{filename} after "
        f"{len(RETRY_DELAYS_S) + 1} attempts: {last_exc}"
    ) from last_exc


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_gauge_free_config(config_path: Path, frac: float) -> None:
    """Assert the adapter config matches the recorded recipe + gauge-free readout."""
    cfg = json.loads(config_path.read_text())
    target_modules = set(cfg.get("target_modules") or [])
    forbidden = target_modules & {"lm_head", "embed_tokens"}
    if forbidden:
        raise AssertionError(
            f"frac {frac}: adapter targets {sorted(forbidden)} — the logit readout "
            f"(delta z_marker) is gauge-INVALID when LoRA touches the unembedding. "
            f"Config: {config_path}"
        )
    if target_modules != EXPECTED_TARGET_MODULES:
        raise AssertionError(
            f"frac {frac}: target_modules {sorted(target_modules)} != expected "
            f"{sorted(EXPECTED_TARGET_MODULES)} — not the recorded #504 v4 recipe."
        )
    if cfg.get("modules_to_save"):
        raise AssertionError(
            f"frac {frac}: modules_to_save={cfg['modules_to_save']!r} is non-empty — "
            f"gauge-free logit readout invalid (plan section 4.2 Step 0)."
        )
    if cfg.get("r") != EXPECTED_LORA_R or cfg.get("lora_alpha") != EXPECTED_LORA_ALPHA:
        raise AssertionError(
            f"frac {frac}: (r={cfg.get('r')}, alpha={cfg.get('lora_alpha')}) != "
            f"expected (r={EXPECTED_LORA_R}, alpha={EXPECTED_LORA_ALPHA})."
        )


def _place_data_file(src: Path, dest: Path, label: str) -> None:
    """Copy a downloaded input into its rig-expected location, hash-guarded.

    If ``dest`` already exists it must be byte-identical to ``src`` (an
    existing DIFFERENT file means input drift — fail loud, never overwrite
    silently).
    """
    if dest.exists():
        if _sha256(dest) == _sha256(src):
            log.info("[phase=fetch] %s already present + identical at %s", label, dest)
            return
        raise AssertionError(
            f"{label} already exists at {dest} but differs from the HF-served copy "
            f"— input drift; refusing to overwrite silently."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    log.info("[phase=fetch] placed %s -> %s", label, dest)


def _merge_retrain_entries(
    ckpt_index: dict[str, dict],
    sha_by_frac: dict[str, str],
    selected_fractions: tuple[float, ...],
    *,
    manifest_path: Path,
    out_index: Path,
) -> None:
    """step6to12 merge (plan v3 section 4.2 Step 1.2): mutate ``ckpt_index``
    with the Phase-T retrain manifest's window snapshots, fill ``step`` for ALL
    entries, verify gauge + pairwise-distinct sha256 across the merged set, and
    write the ``index_provenance.json`` sidecar next to ``out_index``."""
    manifest = json.loads(manifest_path.read_text())
    max_steps = int(manifest["max_steps"])
    window_keys: list[str] = manifest["window_snapshot_keys"]
    provenance: dict[str, dict] = {}
    all_shas: dict[str, str] = dict(sha_by_frac)
    # Hub entries: fill `step` with the callback's exact crossing semantics.
    for frac in selected_fractions:
        key = f"{frac:.2f}"
        step = _first_crossing_step(frac, max_steps)
        ckpt_index[key]["step"] = step
        provenance[key] = {"step": step, "provenance": "hub"}
    # Retrain window entries: local paths from the Phase-T manifest.
    for key in window_keys:
        snap = manifest["snapshots"][key]
        if key in ckpt_index:
            raise AssertionError(
                f"merged-index key collision: retrain key {key!r} already present "
                f"as a Hub key — the 4-dp/2-dp key planes must stay disjoint."
            )
        local_dir = Path(snap["local_path"])
        safetensors = local_dir / "adapter_model.safetensors"
        if not safetensors.exists():
            raise RuntimeError(
                f"retrain snapshot weights missing at {safetensors} (key {key}) — "
                f"Phase T's local copies must survive until the eval consumes them."
            )
        _assert_gauge_free_config(local_dir / "adapter_config.json", float(key))
        sha = _sha256(safetensors)
        if sha != snap["sha256"]:
            raise AssertionError(
                f"retrain snapshot {key}: recomputed sha256 {sha[:12]} != manifest "
                f"{snap['sha256'][:12]} — weights changed between Phase T and Phase E."
            )
        all_shas[key] = sha
        ckpt_index[key] = {"step": int(snap["step"]), "path": str(local_dir)}
        provenance[key] = {"step": int(snap["step"]), "provenance": "retrain"}
    # Extended asserts over the FULL merged index (plan section 4.2 Step 1.2).
    if len(set(all_shas.values())) != len(all_shas):
        dupes = {k: s[:12] for k, s in all_shas.items() if list(all_shas.values()).count(s) > 1}
        raise AssertionError(
            f"merged-index sha256s NOT pairwise distinct across all "
            f"{len(all_shas)} entries: {dupes}."
        )
    keys_in_float_order = sorted(ckpt_index, key=float)
    steps_in_float_order = [ckpt_index[k]["step"] for k in keys_in_float_order]
    if steps_in_float_order != sorted(steps_in_float_order):
        raise AssertionError(
            f"merged-index numeric key order is NOT step order: "
            f"{list(zip(keys_in_float_order, steps_in_float_order, strict=True))} — "
            f"the rig sorts by float(key), so this would scramble the trajectory."
        )
    if any(ckpt_index[k]["step"] is None for k in ckpt_index):
        raise AssertionError("merged index must fill `step` for ALL entries.")
    provenance_path = out_index.parent / "index_provenance.json"
    out_index.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2))
    log.info(
        "[phase=fetch] merged %d-entry index (%d hub + %d retrain); provenance -> %s",
        len(ckpt_index),
        len(selected_fractions),
        len(window_keys),
        provenance_path,
    )


def _first_crossing_step(frac: float, max_steps: int) -> int:
    """The step ``CheckpointAtFractionsCallback`` saves ``frac`` at — EXACT
    float semantics of its ``global_step / max_steps >= frac`` check (no
    ceil()-based reconstruction: 0.08 x 75 floats to 6.000000000000001, which
    a naive ceil would round to 7)."""
    for s in range(1, max_steps + 1):
        if s / max_steps >= frac:
            return s
    raise ValueError(f"fraction {frac} never crosses within max_steps={max_steps}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Task #585: download the per-fraction #504 v4 smoke adapters + the "
            "persona bank + R_eval_v504, verify them, and write checkpoint_index.json."
        )
    )
    ap.add_argument("--out-index", type=Path, required=True)
    ap.add_argument("--local-root", type=Path, required=True)
    ap.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    ap.add_argument("--data-repo-id", default=DEFAULT_DATA_REPO)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help=(
            "Root under which the rig-expected inputs are placed "
            "(<data-root>/issue_472/persona_bank.json + "
            "<data-root>/issue_472/on_policy_R/R_eval_v504.json). Default 'data' "
            "(cwd-relative, matching the rig's --bank-path/--r-eval-path defaults "
            "when run from the repo root). Override for VM smoke runs."
        ),
    )
    ap.add_argument(
        "--fractions",
        default=None,
        help=(
            "step6to12 extension: comma-separated SUBSET of the module default "
            f"Hub fractions {FRACTIONS} to download (e.g. '0.08,0.16' for the "
            "endpoint-parity controls). Default None = all 6 (byte-identical "
            "parent-round behavior)."
        ),
    )
    ap.add_argument(
        "--merge-retrain-manifest",
        type=Path,
        default=None,
        help=(
            "step6to12 extension: path to the Phase-T retrain_manifest.json. "
            "When set, the manifest's window snapshots (recorded steps 6-12) are "
            "merged with the Hub entries into one step-ordered index, the 'step' "
            "field is filled for ALL entries, and an index_provenance.json "
            "sidecar is written next to --out-index."
        ),
    )
    args = ap.parse_args(argv)

    if args.fractions is not None:
        try:
            selected_fractions = tuple(
                sorted(float(x.strip()) for x in args.fractions.split(",") if x.strip())
            )
        except ValueError as exc:
            raise ValueError(
                f"--fractions {args.fractions!r} could not parse as comma-separated floats."
            ) from exc
        unknown = [f for f in selected_fractions if f not in FRACTIONS]
        if not selected_fractions or unknown:
            raise ValueError(
                f"--fractions {args.fractions!r} must be a non-empty subset of the "
                f"verified Hub fractions {FRACTIONS}; unknown: {unknown}."
            )
    else:
        selected_fractions = FRACTIONS

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=fetch_snapshots] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    args.local_root.mkdir(parents=True, exist_ok=True)

    # ── 1. The Hub per-fraction adapter snapshots (per-file download). ────────
    ckpt_index: dict[str, dict] = {}
    sha_by_frac: dict[str, str] = {}
    for frac in selected_fractions:
        token = frac_token(frac)
        subfolder = f"{PRETRAIN_SUBFOLDER}/ckpt_frac{token}"
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            _download_with_retry(
                repo_id=args.repo_id,
                repo_type="model",
                filename=f"{subfolder}/{fname}",
                local_dir=args.local_root,
            )
        local_dir = args.local_root / subfolder
        safetensors = local_dir / "adapter_model.safetensors"
        if not safetensors.exists():
            raise RuntimeError(
                f"missing {safetensors} after per-file hf_hub_download — verify the "
                f"snapshot exists on {args.repo_id} (plan section 4.2 Step 0 said it does)."
            )
        _assert_gauge_free_config(local_dir / "adapter_config.json", frac)
        sha_by_frac[f"{frac:.2f}"] = _sha256(safetensors)
        ckpt_index[f"{frac:.2f}"] = {"step": None, "path": str(local_dir)}

    # Pairwise-distinct sha256s (assumption A2 made a runtime assert).
    if len(set(sha_by_frac.values())) != len(selected_fractions):
        dupes = {f: s for f, s in sha_by_frac.items() if list(sha_by_frac.values()).count(s) > 1}
        raise AssertionError(
            f"adapter sha256s are NOT pairwise distinct — duplicated weights detected "
            f"({dupes}). The Hub snapshots must be bytewise-distinct adapters."
        )
    log.info(
        "[phase=fetch] %d Hub adapters verified pairwise-distinct: %s",
        len(selected_fractions),
        {f: s[:12] for f, s in sha_by_frac.items()},
    )

    # ── 1b. step6to12 merge: retrain window snapshots + step fill + sidecar. ──
    if args.merge_retrain_manifest is not None:
        _merge_retrain_entries(
            ckpt_index,
            sha_by_frac,
            selected_fractions,
            manifest_path=args.merge_retrain_manifest,
            out_index=args.out_index,
        )

    # ── 2. Persona bank (gitignored — absent on a fresh pod checkout). ───────
    bank_src = _download_with_retry(
        repo_id=args.data_repo_id,
        repo_type="dataset",
        filename=BANK_PATH_IN_REPO,
        local_dir=args.local_root / "_data_repo",
    )
    bank_payload = json.loads(bank_src.read_text())
    bank_hash = hashlib.sha256(
        json.dumps(bank_payload["personas"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if bank_hash != EXPECTED_BANK_CONTENT_HASH:
        raise AssertionError(
            f"persona bank content hash {bank_hash} != plan-time verified "
            f"{EXPECTED_BANK_CONTENT_HASH} — bank drift (assumption A9 violated)."
        )
    _place_data_file(bank_src, args.data_root / "issue_472" / "persona_bank.json", "persona bank")

    # ── 3. R_eval_v504.json (coverage-gate input; plan section 4.2 Step 1.1). ─
    r_eval_src = _download_with_retry(
        repo_id=args.data_repo_id,
        repo_type="dataset",
        filename=R_EVAL_PATH_IN_REPO,
        local_dir=args.local_root / "_data_repo",
    )
    r_eval_payload = json.loads(r_eval_src.read_text())
    r_eval_hash = str(r_eval_payload.get("content_hash", ""))
    if not r_eval_hash.startswith(EXPECTED_R_EVAL_CONTENT_HASH_PREFIX):
        raise AssertionError(
            f"R_eval_v504 content_hash {r_eval_hash!r} does not start with the "
            f"plan-time verified prefix {EXPECTED_R_EVAL_CONTENT_HASH_PREFIX!r}."
        )
    _place_data_file(
        r_eval_src,
        args.data_root / "issue_472" / "on_policy_R" / "R_eval_v504.json",
        "R_eval_v504",
    )

    # ── 4. checkpoint_index.json — the shape i504_eval_trajectory.py consumes. ─
    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    args.out_index.write_text(json.dumps(ckpt_index, indent=2))
    log.info(
        "[phase=fetch] wrote checkpoint index (%d fractions) -> %s",
        len(ckpt_index),
        args.out_index,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
