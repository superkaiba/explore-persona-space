"""Issue #1332 follow-up ``lowdose-grid-kill-battery`` — P1 band-stopped trainer.

Trains ONE #474 ordinary-family loc-arm marker implant at LOW dose: the recipe
is byte-identical to ``scripts/i474_phase23_train.py`` (lr 1e-5, r=32, alpha=64,
dropout 0.0, batch 4 x grad_accum 4, max_length 2048, seed 42,
``marker_only_loss=True``, ``marker_tail_tokens=0``, ` ※` id 83399 asserted
in-process, ``save_total_limit=1``) EXCEPT the stop rule (plan v8 §4 P1 — the
round's ONE changed variable): ``marker_band_stop=True`` with band [5, 12] nats,
``epochs=5`` as the ceiling the callback cuts into, per-step probing through the
expected crossing (``marker_band_dense_until=200``, ``marker_band_min_steps=5``).

Training data is the parent's uploaded mix, staged byte-identically from the HF
data repo at the pinned revision (``MIX_REVISION``) to the consumer-exact path
``data/issue_474/train_rows/i474_loc_<cond>.jsonl`` — never rebuilt from
``R_train.json``.

Overshoot fallback (pre-registered, deterministic — plan v8 §4): a run that
ends without stopping in-band is retrained with ``max_steps=k`` where ``k`` is
the last trajectory step with delta in [5, 12] (or the closest approach when
the band was skipped — recorded as a band-miss, never a crash).

Adapters upload per-source to the HF MODEL repo at
``adapters/i1332_lowdose_<cond>`` (NEVER ``adapters/i474_*``); the upload is
re-verified fail-loud here because ``train_lora``'s inline upload is fail-soft.

USAGE (one source per invocation; the gpu dispatcher shards these)
    uv run python scripts/issue1332_lowdose_train.py --cond A1
    uv run python scripts/issue1332_lowdose_train.py --smoke --cond A1   # CPU-only smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1332_common as C

logger = logging.getLogger("issue1332.lowdose_train")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

# Parent training mixes (plan v8 §10 reproducibility card): 16 loc-arm JSONLs,
# consumed at the pinned data-repo revision, staged to the consumer-exact path.
MIX_HF_PREFIX = "issue474_marker_at_end_localized/train_rows"
MIX_REVISION = "7d7fbb856ed844b1a2cd0153b5d5e88f2c3fa437"
MIX_LOCAL_DIR = C.PROJECT_ROOT / "data" / "issue_474" / "train_rows"

# Band-stop block (plan v8 §11 items 1-4). Everything else Source: #474.
BAND_LOW_NATS = 5.0
BAND_HIGH_NATS = 12.0
EPOCHS_CEILING = 5
BAND_DENSE_UNTIL = 200
BAND_MIN_STEPS = 5

EXPECTED_MARKER_ID = 83399
EXPECTED_IM_END_ID = 151645


def hf_adapter_path(cid: str) -> str:
    """HF model-repo subfolder for one low-dose adapter. NEVER an i474_* path."""
    path = f"adapters/i1332_lowdose_{cid}"
    assert "i474" not in path, path  # plan §8: never write adapters/i474_*
    return path


def config_kwargs(
    cid: str,
    *,
    trajectory_path: str,
    max_steps: int | None = None,
    run_suffix: str = "",
    bf16: bool = True,
) -> dict:
    """The exact TrainLoraConfig kwargs for one low-dose training run.

    Kept as a plain dict (no sft import) so the P0 signature smoke and the
    pytest pin can diff it against ``dataclasses.fields(TrainLoraConfig)``.
    ``marker_suppress_at_post_response_slot`` is deliberately NOT passed —
    it is a deprecated no-op whose behavior folded into the default
    (#474/#477 slot fix, 2026-06-23; plan v8 §4 P1).
    """
    from explore_persona_space.experiments.i406_conditions import MARKER_TEXT

    kwargs = dict(
        epochs=EPOCHS_CEILING,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        seed=42,
        run_name=f"i1332_lowdose_{cid}{run_suffix}",
        report_to="wandb",
        save_strategy="no",  # band-stop saves the final adapter via save_model; no ladder
        save_total_limit=1,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_im_end_token_id=EXPECTED_IM_END_ID,
        marker_band_stop=True,
        marker_band_low_nats=BAND_LOW_NATS,
        marker_band_high_nats=BAND_HIGH_NATS,
        marker_band_dense_until=BAND_DENSE_UNTIL,
        marker_band_min_steps=BAND_MIN_STEPS,
        marker_band_trajectory_path=trajectory_path,
        bf16=bf16,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=hf_adapter_path(cid),
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    assert "marker_suppress_at_post_response_slot" not in kwargs
    assert "gpu_id" not in kwargs  # inherited single-GPU CVD pin stays authoritative
    return kwargs


def verify_config_signature(cid: str = "A1") -> None:
    """P0 signature smoke (artifact-reuse porting recipe step 2, plan v8 §4 P0).

    Asserts every kwarg the driver passes exists on current-main
    ``TrainLoraConfig`` (library-API drift fails HERE, not on the pod mid-run),
    and that the deprecated suppress flag — if still present — defaults False
    (the folded-default no-op assumption, plan v8 §12 item 5).
    """
    from dataclasses import fields

    from explore_persona_space.train.sft import TrainLoraConfig

    kwargs = config_kwargs(cid, trajectory_path="/dev/null", max_steps=1)
    field_map = {f.name: f for f in fields(TrainLoraConfig)}
    missing = set(kwargs) - set(field_map)
    if missing:
        raise AssertionError(
            f"Library-API drift: driver passes kwargs missing from TrainLoraConfig: "
            f"{sorted(missing)}"
        )
    suppress = field_map.get("marker_suppress_at_post_response_slot")
    if suppress is not None and suppress.default is not False:
        raise AssertionError(
            "marker_suppress_at_post_response_slot default drifted from False — the "
            "folded-default no-op assumption (plan v8 §12 item 5) no longer holds"
        )
    logger.info("[signature-smoke] %d kwargs all present on TrainLoraConfig", len(kwargs))


def mix_paths(cid: str) -> tuple[str, Path]:
    """(hub filename, consumer-exact local path) for one source's training mix."""
    return (
        f"{MIX_HF_PREFIX}/i474_loc_{cid}.jsonl",
        MIX_LOCAL_DIR / f"i474_loc_{cid}.jsonl",
    )


def stage_mix(cid: str) -> dict:
    """Stage one mix from the data repo at the PINNED revision; record sha256.

    Always fetched at ``MIX_REVISION`` (hf_hub_download caches per-revision) and
    copied over any local file — content identity comes from the revision pin,
    never from a pre-existing local copy (#600 mirror-divergence class).
    """
    import shutil

    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    hub_name, local = mix_paths(cid)
    local.parent.mkdir(parents=True, exist_ok=True)
    got = retry_transient(
        lambda: hf_hub_download(
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            filename=hub_name,
            revision=MIX_REVISION,
        ),
        what=f"stage mix {hub_name}@{MIX_REVISION[:12]}",
    )
    shutil.copyfile(got, local)
    sha = C.sha256_file(local)
    n_rows = sum(1 for line in local.open(encoding="utf-8") if line.strip())
    logger.info("[stage-mix] %s -> %s (sha256=%s rows=%d)", hub_name, local, sha[:12], n_rows)
    return {
        "cid": cid,
        "hub_path": hub_name,
        "revision": MIX_REVISION,
        "sha256": sha,
        "n_rows": n_rows,
        "local_path": str(local),
    }


def select_bracket_step(
    records: list[dict], *, low: float = BAND_LOW_NATS, high: float = BAND_HIGH_NATS
) -> dict:
    """Deterministic bracketing-retrain step from a persisted trajectory.

    ``k`` = the LAST probed step with delta in [low, high]; if the band was
    skipped entirely, the closest-approach step (min distance to the band
    interval, ties broken toward the band midpoint) with ``band_miss=True``
    (plan v8 §4 overshoot fallback + marker-recipe band-entry fallback read).
    Raises on an empty trajectory (nothing to bracket — fail loud).
    """
    if not records:
        raise ValueError("empty trajectory: cannot select a bracketing step")
    in_band = [r for r in records if low <= r["delta_nats"] <= high]
    if in_band:
        pick = in_band[-1]
        return {
            "max_steps": int(pick["step"]),
            "band_miss": False,
            "delta_at_k": float(pick["delta_nats"]),
        }
    mid = (low + high) / 2.0

    def _dist(r: dict) -> tuple[float, float]:
        d = r["delta_nats"]
        return (max(0.0, low - d, d - high), abs(d - mid))

    pick = min(records, key=_dist)
    return {
        "max_steps": int(pick["step"]),
        "band_miss": True,
        "delta_at_k": float(pick["delta_nats"]),
        "closest_approach_dist_nats": float(_dist(pick)[0]),
    }


def _assert_marker_tokenization() -> None:
    """In-process marker assert (marker-leakage-measurement.md; incident #537)."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i406_conditions import MARKER_TEXT

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [EXPECTED_MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [83399]")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end != EXPECTED_IM_END_ID:
        raise AssertionError(f"<|im_end|> id drift: got {im_end}, expected 151645")


def _merge_sidecar(traj_path: Path, extra: dict) -> None:
    """Merge band-stop outcome fields into the trajectory sidecar (atomic)."""
    payload = json.loads(traj_path.read_text()) if traj_path.exists() else {}
    payload.update(extra)
    C.write_json_atomic(traj_path, payload)


def train_one(cid: str, args) -> dict:
    """Train one low-dose implant end-to-end: stage -> train -> band disposition
    -> deterministic bracketing retrain if needed -> fail-loud upload verify."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    # MooseFS/quota safety + no inline WandB checkpoint upload (as #474).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    _assert_marker_tokenization()
    verify_config_signature(cid)
    stage_rec = stage_mix(cid)

    lowdose = C.results_dir(False, args.results_dir) / "lowdose"
    traj_dir = lowdose / "band_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{cid}.json"

    adapter_root = Path(args.adapter_root)
    out_dir = adapter_root / f"i1332_lowdose_{cid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainLoraConfig(**config_kwargs(cid, trajectory_path=str(traj_path)))
    _hub_name, mix_local = mix_paths(cid)
    logger.info(
        "[train] cond=%s mix=%s out=%s band=[%s, %s] epochs_ceiling=%d",
        cid,
        mix_local,
        out_dir,
        BAND_LOW_NATS,
        BAND_HIGH_NATS,
        EPOCHS_CEILING,
    )
    out_path, loss = train_lora(BASE_MODEL, str(mix_local), str(out_dir), cfg=cfg)

    band = json.loads((out_dir / "band_stop_result.json").read_text())
    summary: dict = {
        "cid": cid,
        "train_loss": float(loss),
        "band_stop_result": band,
        "band_miss": False,
        "bracket_retrain": None,
        "mix_stage": stage_rec,
        "adapter_local": str(out_path),
        "adapter_hf_path": hf_adapter_path(cid),
    }

    if not band["stopped_in_band"]:
        # Deterministic bracketing retrain (plan v8 §4 overshoot fallback).
        traj = json.loads(traj_path.read_text())
        bracket = select_bracket_step(traj["records"])
        logger.warning(
            "[bracket] cond=%s did NOT stop in-band (last_delta=%s); retraining with "
            "max_steps=%d (band_miss=%s delta_at_k=%.3f)",
            cid,
            band["last_delta_nats"],
            bracket["max_steps"],
            bracket["band_miss"],
            bracket["delta_at_k"],
        )
        bracket_traj = traj_dir / f"{cid}_bracket.json"
        cfg2 = TrainLoraConfig(
            **config_kwargs(
                cid,
                trajectory_path=str(bracket_traj),
                max_steps=bracket["max_steps"],
                run_suffix="_bracket",
            )
        )
        out_path, loss = train_lora(BASE_MODEL, str(mix_local), str(out_dir), cfg=cfg2)
        band2 = json.loads((out_dir / "band_stop_result.json").read_text())
        summary.update(
            train_loss=float(loss),
            band_stop_result=band2,
            band_miss=bracket["band_miss"],
            bracket_retrain=bracket,
        )

    _merge_sidecar(
        traj_path,
        {
            "cid": cid,
            "band_stop_result": summary["band_stop_result"],
            "band_miss": summary["band_miss"],
            "bracket_retrain": summary["bracket_retrain"],
            "reproducibility_metadata": C.reproducibility_metadata(
                {"followup": "lowdose-grid-kill-battery", "phase": "P1_train", "cid": cid}
            ),
        },
    )

    # Fail-loud upload verify: train_lora's inline adapter upload is fail-SOFT
    # (logger.warning) — the Upload Policy requires the adapter on HF before
    # any teardown, so re-verify here and raise if it never landed.
    if not args.skip_upload_verify:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate.hub import retry_transient

        api = HfApi()
        expected = f"{hf_adapter_path(cid)}/adapter_model.safetensors"
        present = retry_transient(
            lambda: api.file_exists(HF_MODEL_REPO, expected, repo_type="model"),
            what=f"verify adapter upload {expected}",
        )
        if not present:
            raise RuntimeError(
                f"adapter upload missing on HF: {HF_MODEL_REPO}/{expected} — "
                "train_lora's inline upload failed; local copy preserved at "
                f"{out_dir} (upload-before-delete invariant holds)"
            )
        summary["adapter_hf_verified"] = True

    C.write_json_atomic(lowdose / "train_summaries" / f"{cid}.json", summary)
    logger.info(
        "[train-done] cond=%s stopped_in_band=%s step=%s delta=%s band_miss=%s",
        cid,
        summary["band_stop_result"]["stopped_in_band"],
        summary["band_stop_result"]["band_stop_step"],
        summary["band_stop_result"]["last_delta_nats"],
        summary["band_miss"],
    )
    return summary


def smoke(cid: str, args) -> None:
    """CPU smoke: config construction + signature check + staging PATH
    resolution + bracket-step predicate on a synthetic trajectory. No network,
    no model loads; outputs only under the scratch smoke root."""
    verify_config_signature(cid)
    hub_name, local = mix_paths(cid)
    logger.info("[smoke] staging resolution: %s@%s -> %s", hub_name, MIX_REVISION[:12], local)
    assert local.name == f"i474_loc_{cid}.jsonl"
    assert hf_adapter_path(cid) == f"adapters/i1332_lowdose_{cid}"
    synthetic = [{"step": s, "delta_nats": 0.7 * s} for s in range(1, 40)]
    pick = select_bracket_step(synthetic)
    assert not pick["band_miss"] and BAND_LOW_NATS <= pick["delta_at_k"] <= BAND_HIGH_NATS
    skipped = [{"step": 1, "delta_nats": 2.0}, {"step": 2, "delta_nats": 14.5}]
    miss = select_bracket_step(skipped)
    assert miss["band_miss"] and miss["max_steps"] == 2
    out = C.results_dir(True, args.results_dir) / "lowdose" / "train_smoke" / f"{cid}.json"
    C.write_json_atomic(
        out,
        {
            "cid": cid,
            "config_kwargs": {
                k: v for k, v in config_kwargs(cid, trajectory_path="traj.json").items()
            },
            "mix_hub_path": hub_name,
            "mix_revision": MIX_REVISION,
            "bracket_predicate": {"ramp": pick, "skipped": miss},
        },
    )
    logger.info("[smoke] OK -> %s", out)


def main(argv: list[str] | None = None) -> int:
    """Single-source low-dose trainer (the gpu dispatcher shards invocations)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond", required=True, help="one #406 ordinary-family cid, e.g. A1")
    ap.add_argument("--smoke", action="store_true", help="CPU config/staging smoke only")
    ap.add_argument("--results-dir", default=None, help="override eval_results/issue_1332")
    ap.add_argument(
        "--adapter-root",
        default=(
            "/workspace/adapters/i1332_lowdose"
            if os.path.isdir("/workspace")
            else str(C.data_root(False) / "adapters_lowdose")
        ),
        help="local adapter output root (pod default /workspace/adapters/i1332_lowdose)",
    )
    ap.add_argument(
        "--skip-upload-verify",
        action="store_true",
        help="skip the fail-loud HF adapter-presence re-verify (smoke only)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

    if args.cond not in CONDITIONS_BY_ID:
        raise ValueError(f"--cond {args.cond!r} not in active set {list(CONDITIONS_BY_ID)}")

    if args.smoke:
        smoke(args.cond, args)
        return 0
    train_one(args.cond, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
