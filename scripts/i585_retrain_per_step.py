#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" intentional
"""Task #585 follow-up `step6to12-transition-sweep` — Phase T per-step retrain.

Runs FROM the DETACHED #504 launch-SHA checkout (affdd82cb0bb31257b5668b327c6af5
716212b6c) and replicates ``i504_run_cell.py``'s build+train phases verbatim
(same imports, same resolved values) for the ONE cell ``c504v3_smoke_eps3``
seed 42 — the nested (buggy) eval is the ONLY part omitted (plan v3 §4.2 Step
1.1). The single new variable vs the original #504 v4 pretrain is the snapshot
grid: per-step adapter snapshots at optimizer steps 6–12 via mid-point
``step_calibration_fractions`` (plan §4.1 — float-robust crossing targets), plus
3 bonus late saves + the terminal adapter.

Fail-loud asserts (plan §4.2 Step 1.1 + §12):

  * in-process marker assert ``encode(" ※") == [83399]`` BEFORE any heavy work;
  * launch-SHA basis asserts: ``git rev-parse HEAD`` == the pinned training SHA
    AND ``explore_persona_space.train.sft`` has NO ``MarkerBandStopCallback``
    (deliberate-saturation replication — the original code predates the
    callback; A1);
  * signature self-check: every kwarg this glue passes to ``train_one_cell`` /
    ``build_cell_504`` exists in the checkout's signature (library-API-drift
    guard, the #451/#529 partial-port crash class);
  * persona bank downloaded fail-loud from the HF data repo (NOT in git at the
    launch SHA — plan correction) + content-hash assert; R_train_v504.json
    downloaded fail-loud, internal content_hash LOGGED (A3: single-source,
    residual risk — log, don't gate);
  * post-build: 400 rows total, 200 villain positives, 100 qwen_default + 100
    origami_artist negatives, 0 marker-in-negative contamination
    (builder-raised), independent positive-row scan;
  * post-train: callback-recorded steps == targets for ALL 10 fractions —
    jointly this pins max_steps == 75 UNIQUELY (the only integer in [40, 130)
    whose crossing math yields the recorded steps; verified at implementation
    time), covering A7's "max_steps observed == 75";
  * pairwise-distinct sha256 over the 7 window snapshots' weights.

Inline HF persistence (the original #504 v4 mechanism): the two
``EPM_PERSIST_TRAJECTORY_HF_*`` env vars are set BEFORE ``train_one_cell`` so
``CheckpointAtFractionsCallback`` uploads + Hub-verifies each snapshot before
the next training step proceeds (fail-loud, A13).

Usage (driven by scripts/launchers/launch_issue_585_step6to12.sh, Phase T):
    uv run python scripts/i585_retrain_per_step.py \
        --arm-to-n-json /workspace/arm_to_n_pinned.json \
        --runs-root /workspace/runs/issue_585_step6to12 \
        --manifest-out eval_results/issue_585/step6to12-transition-sweep/retrain_manifest.json \
        --sentinel-path /workspace/logs/issue-585-step6to12-retrain.json

    --build-only : CPU smoke — marker/SHA/signature asserts + downloads + mix
                   build + post-build asserts, then exit 0 BEFORE training.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i585.retrain_per_step")

# ── Pinned identity (plan v3 §10 Reproducibility Card). ──────────────────────
EXPECTED_TRAIN_SHA = "affdd82cb0bb31257b5668b327c6af5716212b6c"
CELL_SLUG = "c504v3_smoke_eps3"
SEED = 42
FOLLOWUP_LABEL = "step6to12-transition-sweep"

# Recipe — every value inherited from the original #504 v4 pretrain (plan §11).
LR = 1e-4
EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 32
EXPECTED_MAX_STEPS = 75  # 400 rows / (batch 4 × grad-accum 4) = 25 steps/epoch × 3 (A7)

# Mid-point fraction targeting (plan §4.1 table — float-robust: each fraction
# is (s − 0.5)/75 rounded to 4 dp, ≥ half-a-step margin on both sides).
WINDOW_FRACTION_TO_STEP = {
    0.0733: 6,
    0.0867: 7,
    0.1000: 8,
    0.1133: 9,
    0.1267: 10,
    0.1400: 11,
    0.1533: 12,
}
# Bonus late saves (saved + uploaded at zero marginal cost; NOT evaluated this
# round — plan §3.5 named additive (ii)). 0.7400 (not 0.7467) captures step 56.
BONUS_FRACTION_TO_STEP = {0.3267: 25, 0.5000: 38, 0.7400: 56}
STEP_CALIBRATION_FRACTIONS = tuple(sorted({**WINDOW_FRACTION_TO_STEP, **BONUS_FRACTION_TO_STEP}))
FRAC_PRECISION = 4

# HF destinations (plan §4.1 + §10).
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_TRAJECTORY_SUBFOLDER = "adapters/issue_585_step6to12/c504v4_smoke_eps3_seed42_retrain"
HF_FINAL_PATH_IN_REPO = "adapters/issue_585_step6to12/c504v4_smoke_eps3_seed42_retrain_final"
RUN_NAME = "issue585_step6to12_retrain_c504v3_smoke_eps3_seed42_lr0.0001"

# Data inputs on the HF data repo (plan §10 reused-artifacts row).
R_TRAIN_PATH_IN_REPO = "issue504_geometry/on_policy_R/R_train_v504.json"

# Expected composition of the rebuilt mix (plan §4.3: strict reproduction of
# the original cell's training data; contrastive-negatives rule satisfied by
# inheritance — 200 villain positives + 100 qwen_default + 100 origami_artist).
EXPECTED_TOTAL_ROWS = 400
EXPECTED_POSITIVE_ROWS = 200
EXPECTED_NEGATIVE_COUNTS = [100, 100]
EXPECTED_SMOKE_MID_BAND_N = "origami_artist"
EXPECTED_PANEL_SIZE = 54

SCHEMA_VERSION = "i585_retrain_manifest_v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, env={**os.environ}
    ).strip()


def _gpu_name() -> str:
    """GPU model recorded in the manifest (A5 splice-gate caveat input)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            env={**os.environ},
        )
        return out.strip().splitlines()[0] if out.strip() else "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "cpu-only-or-no-nvidia-smi"


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in ("torch", "transformers", "trl", "peft", "huggingface-hub"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "missing"
    return out


def _assert_signature_covers(fn, kwargs: set[str], label: str) -> None:
    """Library-API-drift guard: every kwarg we pass must exist at this SHA."""
    params = set(inspect.signature(fn).parameters)
    missing = kwargs - params
    if missing:
        raise AssertionError(
            f"Library-API drift: this glue passes kwargs missing from {label} at the "
            f"checked-out SHA: {sorted(missing)}. Present: {sorted(params)}."
        )


def _write_sentinel(path: Path, phase: str, note: dict) -> None:
    """poll_pipeline.py-compliant progress sentinel (schema v1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:progress",
                "version": 1,
                "task_id": 585,
                "by": "i585_retrain_per_step",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note),
            },
            indent=2,
        )
    )


def _post_build_asserts(
    train_jsonl: Path,
    *,
    marker_text: str,
    default_negative: str,
    smoke_mid_band_n: str,
) -> tuple[str, dict]:
    """Plan §4.2 Step 1.1 post-build asserts. Returns (train_pool_sha256, manifest)."""
    rows = [json.loads(line) for line in train_jsonl.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise AssertionError(f"train_pool has {len(rows)} rows, expected {EXPECTED_TOTAL_ROWS}.")
    # Independent positive count: completion ends with the marker text.
    n_pos_scan = sum(1 for r in rows if r["completion"][0]["content"].endswith(marker_text))
    if n_pos_scan != EXPECTED_POSITIVE_ROWS:
        raise AssertionError(
            f"independent positive-row scan found {n_pos_scan} marker-terminated rows, "
            f"expected {EXPECTED_POSITIVE_ROWS}."
        )
    build_manifest = json.loads(train_jsonl.with_suffix(".manifest.json").read_text())
    if build_manifest["n_positive_rows"] != EXPECTED_POSITIVE_ROWS:
        raise AssertionError(
            f"builder manifest n_positive_rows={build_manifest['n_positive_rows']}"
        )
    if build_manifest["negative_personas"] != [default_negative, smoke_mid_band_n]:
        raise AssertionError(
            f"negative_personas={build_manifest['negative_personas']} != "
            f"[{default_negative!r}, {smoke_mid_band_n!r}] (plan §4.3 composition)."
        )
    if build_manifest["negative_counts"] != EXPECTED_NEGATIVE_COUNTS:
        raise AssertionError(
            f"negative_counts={build_manifest['negative_counts']} != {EXPECTED_NEGATIVE_COUNTS}."
        )
    if any(build_manifest["marker_in_R_counts"].values()):
        raise AssertionError(
            f"marker contamination counts non-zero: {build_manifest['marker_in_R_counts']}."
        )
    train_pool_sha = _sha256(train_jsonl)
    log.info(
        "[phase=build_asserts] PASS: %d rows (%d pos; negs %s x %s); sha256=%s",
        len(rows),
        n_pos_scan,
        build_manifest["negative_personas"],
        build_manifest["negative_counts"],
        train_pool_sha[:12],
    )
    return train_pool_sha, build_manifest


def _post_train_asserts(
    index: dict[str, dict], final_adapter_dir: Path
) -> tuple[list[int], dict[str, str], list[str]]:
    """Plan §4.1 + A7/A8 post-train asserts.

    Returns (window_steps, sha256-by-key over all snapshots, window_keys).
    The joint recorded-step assert pins max_steps == 75 uniquely (the only
    integer in [40, 130) whose first-crossing math yields ALL the recorded
    steps — verified at implementation time), covering A7's "max_steps
    observed == 75".
    """
    all_targets = {**WINDOW_FRACTION_TO_STEP, **BONUS_FRACTION_TO_STEP}
    recorded: dict[str, int] = {}
    for frac, target_step in sorted(all_targets.items()):
        key = f"{frac:.{FRAC_PRECISION}f}"
        if key not in index:
            raise AssertionError(f"checkpoint index missing fraction key {key!r}: {sorted(index)}")
        got = index[key]["step"]
        if got != target_step:
            raise AssertionError(
                f"fraction {key}: recorded step {got} != target {target_step} — the "
                f"crossing landed off-step (float-boundary issue the mid-point "
                f"targeting was designed to eliminate, plan §4.1)."
            )
        recorded[key] = got
    window_steps = sorted(index[f"{f:.{FRAC_PRECISION}f}"]["step"] for f in WINDOW_FRACTION_TO_STEP)
    if window_steps != list(range(6, 13)):
        raise AssertionError(f"window recorded steps {window_steps} != [6..12].")
    terminal_key = f"{1.0:.{FRAC_PRECISION}f}"
    if index.get(terminal_key, {}).get("path") != str(final_adapter_dir):
        raise AssertionError(
            f"terminal index entry {terminal_key!r} does not point at the final adapter: "
            f"{index.get(terminal_key)}"
        )
    log.info("[phase=train_asserts] recorded steps == targets for all %d fractions.", len(recorded))

    # Pairwise-distinct sha256 over the 7 window snapshots (plan §4.2 Step 1.1);
    # bonus + terminal hashed too (manifest provenance).
    sha_by_key: dict[str, str] = {}
    for frac in all_targets:
        key = f"{frac:.{FRAC_PRECISION}f}"
        weights = Path(index[key]["path"]) / "adapter_model.safetensors"
        if not weights.exists():
            raise AssertionError(f"snapshot weights missing at {weights} (fraction {key}).")
        sha_by_key[key] = _sha256(weights)
    sha_by_key[terminal_key] = _sha256(
        Path(index[terminal_key]["path"]) / "adapter_model.safetensors"
    )
    window_keys = [f"{f:.{FRAC_PRECISION}f}" for f in sorted(WINDOW_FRACTION_TO_STEP)]
    window_shas = [sha_by_key[k] for k in window_keys]
    if len(set(window_shas)) != len(window_shas):
        dupes = {k: s[:12] for k, s in sha_by_key.items() if window_shas.count(s) > 1}
        raise AssertionError(
            f"window snapshot sha256s NOT pairwise distinct: {dupes} — per-step weights "
            f"must differ (a collapse here means the callback re-saved the same state)."
        )
    log.info(
        "[phase=train_asserts] 7 window snapshots pairwise-distinct: %s",
        {k: sha_by_key[k][:12] for k in window_keys},
    )
    return window_steps, sha_by_key, window_keys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-to-n-json", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--sentinel-path", type=Path, required=True)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help=(
            "Root under which the training inputs are placed "
            "(<data-root>/issue_472/persona_bank.json + "
            "<data-root>/issue_472/on_policy_R/R_train_v504.json). Default 'data' "
            "(cwd-relative, repo root). Override for VM smoke runs."
        ),
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="CPU smoke: asserts + downloads + mix build, exit 0 before training.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=retrain_per_step] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # ── Pin guard: this script ONLY runs at the #504 launch SHA (plan §4.0). ──
    head = _git_sha()
    if head != EXPECTED_TRAIN_SHA:
        raise RuntimeError(
            f"HEAD={head} != pinned training SHA {EXPECTED_TRAIN_SHA} — this glue "
            f"replicates the ORIGINAL #504 v4 pretrain code path and must run from "
            f"the detached launch-SHA checkout (plan v3 §4.0)."
        )
    log.info("[phase=pin_guard] HEAD is the pinned #504 launch SHA (%s).", head[:12])

    # ── Deliberate-saturation basis assert (A1): NO band-stop at this SHA. ────
    from explore_persona_space.train import sft as sft_module

    if hasattr(sft_module, "MarkerBandStopCallback"):
        raise AssertionError(
            "explore_persona_space.train.sft HAS MarkerBandStopCallback — the "
            "training basis is NOT the pre-band-stop launch SHA. The retrain must "
            "replicate the original no-band-stop saturating run (plan §4.0 / A1); "
            "training here would early-stop ~step 6 and destroy the 6–12 window."
        )
    log.info("[phase=basis_check] sft module has no MarkerBandStopCallback — A1 holds.")

    # ── Marker tokenizer assert (CLAUDE.md marker rule), before heavy work. ───
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        ALWAYS_INCLUDE_NEGATIVE,
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_IM_END_TOKEN_ID,
        MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        MARKER_TEXT,
        SOURCE_PERSONA,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker tokenizer assertion FAILED: encode({MARKER_TEXT!r})={ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]."
        )
    if SOURCE_PERSONA != "villain":
        raise AssertionError(
            f"module SOURCE_PERSONA={SOURCE_PERSONA!r} != 'villain' — not the recipe "
            f"this round replicates (plan §10)."
        )
    if MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT is not True or MARKER_IM_END_TOKEN_ID != 151645:
        raise AssertionError(
            f"slot-fix constants drifted: suppress={MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT}, "
            f"im_end={MARKER_IM_END_TOKEN_ID} (expected True / 151645)."
        )
    log.info("[phase=preflight] marker assertion PASS: %r -> %s", MARKER_TEXT, ids)

    # ── Signature self-check (library-API-drift guard at the pinned SHA). ─────
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        train_one_cell,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.build_training_data import (
        build_cell_504,
    )

    _assert_signature_covers(
        train_one_cell,
        {
            "cell_slug",
            "seed",
            "train_jsonl",
            "output_dir",
            "ckpt_root",
            "report_to",
            "gpu_id",
            "lr_override",
            "epochs_override",
            "hf_path_in_repo_override",
            "run_name_override",
            "step_calibration_fractions",
            "frac_precision",
            "lora_r_override",
            "lora_alpha_override",
            "marker_suppress_at_post_response_slot",
            "marker_im_end_token_id",
        },
        "train_one_cell",
    )
    _assert_signature_covers(
        build_cell_504,
        {
            "r_train",
            "arm_to_positioned_n",
            "q_train",
            "persona_bank",
            "source",
            "marker_text",
            "smoke_mid_band_n",
            "seed",
        },
        "build_cell_504",
    )
    log.info("[phase=signature_check] train_one_cell + build_cell_504 signatures cover the call.")

    # ── Inputs: arm_to_n (pinned copy), bank + R_train from the HF data repo. ─
    # The bank is NOT in git at the launch SHA (gitignored — plan fact-check
    # correction), so download it here fail-loud, hash-guarded. Reuses the
    # parent fetch glue's retry/hash helpers (extracted side by side in Phase T).
    import i585_fetch_snapshots_build_index as fetch_glue

    arm_to_n_payload = json.loads(args.arm_to_n_json.read_text())
    smoke_mid_band_n = arm_to_n_payload.get("smoke_mid_band_n")
    if smoke_mid_band_n != EXPECTED_SMOKE_MID_BAND_N:
        raise AssertionError(
            f"arm_to_n smoke_mid_band_n={smoke_mid_band_n!r} != expected "
            f"{EXPECTED_SMOKE_MID_BAND_N!r} (plan §4.2 Step 0 / A4) — wrong pinned copy?"
        )
    held_out_panel = arm_to_n_payload.get("held_out_panel", [])
    if len(held_out_panel) != EXPECTED_PANEL_SIZE:
        raise AssertionError(
            f"held_out_panel has {len(held_out_panel)} personas, expected "
            f"{EXPECTED_PANEL_SIZE} (A4)."
        )
    overlap = set(held_out_panel) & {SOURCE_PERSONA, ALWAYS_INCLUDE_NEGATIVE, smoke_mid_band_n}
    if overlap:
        raise AssertionError(
            f"panel ∩ {{source, default, mid-band N}} = {sorted(overlap)} — training-"
            f"negative/panel disjointness violated (contrastive-negatives rule)."
        )

    download_root = args.runs_root / "_data_repo"
    bank_src = fetch_glue._download_with_retry(
        repo_id=fetch_glue.DEFAULT_DATA_REPO,
        repo_type="dataset",
        filename=fetch_glue.BANK_PATH_IN_REPO,
        local_dir=download_root,
    )
    bank_payload = json.loads(bank_src.read_text())
    bank_hash = hashlib.sha256(
        json.dumps(bank_payload["personas"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if bank_hash != fetch_glue.EXPECTED_BANK_CONTENT_HASH:
        raise AssertionError(
            f"persona bank content hash {bank_hash} != plan-time verified "
            f"{fetch_glue.EXPECTED_BANK_CONTENT_HASH} — bank drift."
        )
    bank_dest = args.data_root / "issue_472" / "persona_bank.json"
    fetch_glue._place_data_file(bank_src, bank_dest, "persona bank")

    r_train_src = fetch_glue._download_with_retry(
        repo_id=fetch_glue.DEFAULT_DATA_REPO,
        repo_type="dataset",
        filename=R_TRAIN_PATH_IN_REPO,
        local_dir=download_root,
    )
    r_train_payload = json.loads(r_train_src.read_text())
    r_train_content_hash = str(r_train_payload.get("content_hash", "missing"))
    # A3: single-source input — content_hash LOGGED (manifest), not gated; an
    # endpoint-parity failure is re-examined against this first (plan §3).
    log.info("[phase=inputs] R_train_v504 content_hash=%s (logged, A3)", r_train_content_hash)
    r_train_dest = args.data_root / "issue_472" / "on_policy_R" / "R_train_v504.json"
    fetch_glue._place_data_file(r_train_src, r_train_dest, "R_train_v504")

    bank = load_persona_bank(bank_dest)
    r_train = load_r_artifact(r_train_dest)
    q_train, _q_eval = get_train_eval_questions()

    # ── Phase: build training data (CPU) — verbatim i504_run_cell resolution. ─
    run_dir = args.runs_root / f"{CELL_SLUG}_seed{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"

    log.info("[phase=build] building training mix via build_cell_504(%r, seed=%d)", CELL_SLUG, SEED)
    build_cell_504(
        CELL_SLUG,
        train_jsonl,
        r_train=r_train,
        arm_to_positioned_n=arm_to_n_payload.get("arm_to_positioned_n", {}),
        q_train=q_train,
        persona_bank=bank,
        source=SOURCE_PERSONA,
        marker_text=MARKER_TEXT,
        smoke_mid_band_n=smoke_mid_band_n,
        seed=SEED,
    )

    # ── Post-build asserts (plan §4.2 Step 1.1). ──────────────────────────────
    train_pool_sha, build_manifest = _post_build_asserts(
        train_jsonl,
        marker_text=MARKER_TEXT,
        default_negative=ALWAYS_INCLUDE_NEGATIVE,
        smoke_mid_band_n=smoke_mid_band_n,
    )

    if args.build_only:
        log.info("[phase=build_only_done] --build-only: stopping before train_one_cell.")
        _write_sentinel(
            args.sentinel_path,
            phase="build_only_done",
            note={
                "cell": CELL_SLUG,
                "seed": SEED,
                "train_pool_sha256": train_pool_sha,
                "r_train_content_hash": r_train_content_hash,
                "build_only": True,
            },
        )
        return 0

    # ── Phase: train with per-step snapshots + inline HF persistence (§4.1). ──
    # The two env vars arm CheckpointAtFractionsCallback's fail-loud inline
    # upload+verify per snapshot (the original #504 v4 mechanism, A13).
    os.environ["EPM_PERSIST_TRAJECTORY_HF_REPO"] = HF_MODEL_REPO
    os.environ["EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER"] = HF_TRAJECTORY_SUBFOLDER
    log.info(
        "[phase=train] lr=%g r=%d alpha=%d epochs=%d fractions=%s (precision=%d) → %s",
        LR,
        LORA_R,
        LORA_ALPHA,
        EPOCHS,
        STEP_CALIBRATION_FRACTIONS,
        FRAC_PRECISION,
        final_adapter_dir,
    )
    train_result = train_one_cell(
        cell_slug=CELL_SLUG,
        seed=SEED,
        train_jsonl=train_jsonl,
        output_dir=final_adapter_dir,
        ckpt_root=ckpt_root,
        report_to=args.report_to,
        gpu_id=args.gpu_id,
        lr_override=LR,
        epochs_override=EPOCHS,
        lora_r_override=LORA_R,
        lora_alpha_override=LORA_ALPHA,
        hf_path_in_repo_override=HF_FINAL_PATH_IN_REPO,
        run_name_override=RUN_NAME,
        step_calibration_fractions=STEP_CALIBRATION_FRACTIONS,
        frac_precision=FRAC_PRECISION,
        marker_suppress_at_post_response_slot=MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        marker_im_end_token_id=MARKER_IM_END_TOKEN_ID,
    )
    index = train_result["checkpoint_index"]

    # ── Post-train asserts (plan §4.1 + A7/A8). ───────────────────────────────
    window_steps, sha_by_key, window_keys = _post_train_asserts(index, final_adapter_dir)

    # ── Manifest (the splice-gate + index-merge input, plan §4.2 Step 1.1). ───
    try:
        import wandb

        wandb_run_id = wandb.run.id if wandb.run is not None else None
    except Exception:  # wandb optional at manifest time — run name is deterministic
        wandb_run_id = None
    snapshots = {
        key: {
            "step": index[key]["step"],
            "local_path": index[key]["path"],
            "hub_path": f"{HF_TRAJECTORY_SUBFOLDER}/ckpt_frac{key}",
            "sha256": sha_by_key[key],
        }
        for key in sorted(sha_by_key, key=float)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": 585,
        "followup_label": FOLLOWUP_LABEL,
        "cell": CELL_SLUG,
        "seed": SEED,
        "recipe": {
            "base_model": BASE_MODEL,
            "lr": LR,
            "epochs": EPOCHS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "marker_text": MARKER_TEXT,
            "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
            "marker_suppress_at_post_response_slot": MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
            "marker_im_end_token_id": MARKER_IM_END_TOKEN_ID,
            "band_stop": "absent (launch-SHA basis predates MarkerBandStopCallback — A1)",
        },
        "step_calibration_fractions": list(STEP_CALIBRATION_FRACTIONS),
        "frac_precision": FRAC_PRECISION,
        "window_snapshot_keys": window_keys,
        "max_steps": EXPECTED_MAX_STEPS,
        "max_steps_basis": (
            "pinned by the recorded snapshot steps: the joint targets "
            "{0.0733→6 … 0.1533→12, 0.3267→25, 0.5000→38, 0.7400→56} are satisfied "
            "by max_steps=75 ONLY (unique in [40, 130))."
        ),
        "snapshots": snapshots,
        "train_pool": {
            "path": str(train_jsonl),
            "sha256": train_pool_sha,
            "n_rows": build_manifest["n_total_rows"],
            "n_positive_rows": EXPECTED_POSITIVE_ROWS,
            "negative_personas": build_manifest["negative_personas"],
            "negative_counts": build_manifest["negative_counts"],
        },
        "r_train_content_hash": r_train_content_hash,
        "bank_content_hash": bank_hash,
        "arm_to_n_json": str(args.arm_to_n_json),
        "hf_final_adapter_path_in_repo": HF_FINAL_PATH_IN_REPO,
        "wandb": {"run_name": RUN_NAME, "run_id": wandb_run_id},
        "git_commit": head,
        "package_versions": _package_versions(),
        "gpu_name": _gpu_name(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2))
    log.info("[phase=manifest] wrote %s", args.manifest_out)

    _write_sentinel(
        args.sentinel_path,
        phase="retrain_done",
        note={
            "cell": CELL_SLUG,
            "seed": SEED,
            "manifest_path": str(args.manifest_out),
            "window_steps": window_steps,
            "hub_subfolder": HF_TRAJECTORY_SUBFOLDER,
            "train_pool_sha256": train_pool_sha,
        },
    )
    log.info("[phase=retrain_done] wrote sentinel → %s", args.sentinel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
