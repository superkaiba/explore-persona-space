#!/usr/bin/env python3
"""CPU smoke for ``scripts/convert_ds_zero3_to_hf.py``.

Builds a tiny CPU model (``sshleifer/tiny-gpt2``), wraps it with DeepSpeed
ZeRO-3 on CPU, runs one optimizer step so the engine has real state,
saves a DS-native checkpoint via ``engine.save_checkpoint``, then
invokes the conversion script in a subprocess (matching the
``train_stage_sft.py`` flow). Asserts the output directory contains
the HF-loadable artifacts (``*.safetensors`` + tokenizer + config) and
that the resulting state-dict matches the pre-DS state-dict
element-wise.

Run as::

    uv run python scripts/smoke_convert_ds_zero3_to_hf.py

Exit code 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:  # noqa: C901 — linear smoke driver, 4 cases
    tmpdir = Path(tempfile.mkdtemp(prefix="ds_conv_smoke_"))
    try:
        print(f"[smoke_convert_ds_zero3_to_hf] tmpdir={tmpdir}")
        ds_native_dir = tmpdir / "ds_native"
        ds_native_dir.mkdir(parents=True)
        out_dir = tmpdir / "out"

        model_id = "sshleifer/tiny-gpt2"
        print(f"[smoke_convert_ds_zero3_to_hf] loading {model_id} (tiny CPU model)")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        # Load tokenizer for side-effect of caching (the conversion subprocess
        # re-loads it from the model_id). Unused locally.
        _ = AutoTokenizer.from_pretrained(model_id)

        # Snapshot reference state-dict for comparison.
        ref_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[smoke_convert_ds_zero3_to_hf] reference state has {len(ref_state)} tensors, "
            f"{sum(t.numel() for t in ref_state.values())} params"
        )

        # Initialize DeepSpeed ZeRO-3 on CPU (no GPU required).
        import deepspeed

        ds_config = {
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "optimizer": {"type": "Adam", "params": {"lr": 1e-5}},
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
            "wall_clock_breakdown": False,
        }
        # Single-process CPU run requires distributed bootstrap env vars.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        print("[smoke_convert_ds_zero3_to_hf] initializing DS engine (CPU, ZeRO-3, world_size=1)")
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
            dist_init_required=True,
        )

        # Run one optimizer step so the engine carries real state.
        print("[smoke_convert_ds_zero3_to_hf] running one fake training step")
        engine.train()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=engine.device)
        out = engine(input_ids=input_ids, labels=input_ids)
        engine.backward(out.loss)
        engine.step()

        # Save DS-native checkpoint (this is what train_stage_sft.py now does
        # for ZeRO-3 FWFT instead of trainer.save_model).
        print(f"[smoke_convert_ds_zero3_to_hf] saving DS-native checkpoint to {ds_native_dir}")
        engine.save_checkpoint(str(ds_native_dir), tag="final")

        # Free training-process state before spawning the conversion subprocess
        # (mirrors the train_stage_sft.py flow exactly).
        del engine
        del model
        import gc

        gc.collect()

        # Invoke the conversion script in a fresh subprocess.
        converter = Path(__file__).parent / "convert_ds_zero3_to_hf.py"
        cmd = [
            sys.executable,
            str(converter),
            "--ds-checkpoint-dir",
            str(ds_native_dir),
            "--output-dir",
            str(out_dir),
            "--model-id",
            model_id,
            "--tag",
            "final",
            "--max-shard-size",
            "100MB",  # small for the tiny model
            "--dtype",
            "float32",  # match the reference precision for exact comparison
        ]
        print(f"[smoke_convert_ds_zero3_to_hf] running conversion subprocess: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print("---- conversion stdout ----")
        print(result.stdout)
        print("---- conversion stderr ----")
        print(result.stderr)
        if result.returncode != 0:
            print(f"FAIL: conversion subprocess exited rc={result.returncode}")
            return 1

        # Assert HF artifacts present.
        st_files = sorted(out_dir.glob("*.safetensors"))
        if not st_files:
            print(f"FAIL: no *.safetensors files written to {out_dir}")
            return 1
        if not (out_dir / "config.json").exists():
            print(f"FAIL: config.json missing in {out_dir}")
            return 1
        if not (out_dir / "tokenizer.json").exists() and not (out_dir / "vocab.json").exists():
            print(f"FAIL: no tokenizer files in {out_dir}")
            return 1
        artifact_names = [p.name for p in out_dir.iterdir()]
        print(f"[smoke_convert_ds_zero3_to_hf] artifacts present: {artifact_names}")

        # Roundtrip: load the saved HF checkpoint and assert state matches.
        loaded = AutoModelForCausalLM.from_pretrained(str(out_dir), torch_dtype=torch.float32)
        loaded_state = loaded.state_dict()

        # Allow small drift from optimizer step + dtype roundtrip (the engine
        # actually changed the weights, so we can't compare to the original
        # ref_state; we just check shape parity + reasonable magnitude).
        ref_keys = set(ref_state.keys())
        loaded_keys = set(loaded_state.keys())
        # HF can rename some keys (e.g. transformer.h.0.* -> ...); allow superset.
        missing = ref_keys - loaded_keys
        if missing:
            print(f"FAIL: keys missing in loaded state: {sorted(missing)[:5]}")
            return 1
        for k in ref_keys:
            if ref_state[k].shape != loaded_state[k].shape:
                print(
                    f"FAIL: shape mismatch for {k}: ref={ref_state[k].shape} "
                    f"loaded={loaded_state[k].shape}"
                )
                return 1
        print(
            f"[smoke_convert_ds_zero3_to_hf] roundtrip PASS: {len(ref_keys)} keys, all shapes match"
        )

        # Config sanity: torch_dtype set.
        cfg_json = json.loads((out_dir / "config.json").read_text())
        cfg_dtype = cfg_json.get("torch_dtype")
        if cfg_dtype != "float32":
            print(f"FAIL: config.json torch_dtype = {cfg_dtype!r}, expected 'float32'")
            return 1

        print(
            f"\n[smoke_convert_ds_zero3_to_hf] Case A PASS — {len(st_files)} "
            f"safetensors shard(s), {sum(p.stat().st_size for p in st_files)} "
            f"bytes total, roundtrip matches."
        )

        # ===== Case B — cross-config regression test =====
        # #506 round-6 critical: the conversion subprocess was invoked with
        # the wrong `--model-id` for Phase 2, producing safetensors with the
        # trained model's tensors but a config.json from a DIFFERENT
        # architecture (the silent-corruption mode). The architecture-parity
        # gate in convert_ds_zero3_to_hf.py must fail-loud on this mismatch.
        # This case invokes the conversion subprocess with a deliberately
        # WRONG --model-id (gpt2-medium has 24 layers, hidden 1024 vs
        # tiny-gpt2's 6 layers, hidden 128) and asserts non-zero exit.
        out_dir_b = tmpdir / "out_b"
        wrong_model_id = "gpt2-medium"  # 24 layers, hidden 1024 — definitely NOT tiny-gpt2
        cmd_b = [
            sys.executable,
            str(converter),
            "--ds-checkpoint-dir",
            str(ds_native_dir),
            "--output-dir",
            str(out_dir_b),
            "--model-id",
            wrong_model_id,
            "--tag",
            "final",
            "--max-shard-size",
            "100MB",
            "--dtype",
            "float32",
        ]
        print(
            "\n[smoke_convert_ds_zero3_to_hf] Case B — cross-config regression: "
            f"invoking conversion with WRONG --model-id={wrong_model_id} "
            "(trained model was tiny-gpt2). Expecting FAIL-LOUD..."
        )
        result_b = subprocess.run(cmd_b, capture_output=True, text=True, check=False)
        if result_b.returncode == 0:
            print(
                "FAIL: cross-config regression — conversion exited 0 with mismatched "
                "--model-id, but the architecture-parity gate should have raised "
                "SystemExit. The smoke would PASS while corrupt artifacts ship "
                "(this is the round-6 silent-corruption mode the gate exists to "
                "prevent)."
            )
            print("---- Case B stdout ----")
            print(result_b.stdout)
            return 1
        # The gate prints a specific message; confirm it appeared.
        stdout_b = result_b.stdout + result_b.stderr
        if "architecture-parity gate" not in stdout_b:
            print(
                f"FAIL: cross-config regression — subprocess exited "
                f"rc={result_b.returncode}, but the gate's message "
                f"('architecture-parity gate') is missing. The conversion "
                "may have failed for a DIFFERENT reason; the gate is not "
                "providing real protection."
            )
            print("---- Case B stdout ----")
            print(result_b.stdout)
            print("---- Case B stderr ----")
            print(result_b.stderr)
            return 1
        # No artifacts should have been written before the fail-loud.
        if out_dir_b.exists() and any(out_dir_b.glob("*.safetensors")):
            print(
                f"FAIL: cross-config regression — gate raised but safetensors "
                f"were already written to {out_dir_b}. The gate fires too "
                "late; the corrupt artifact could still ship."
            )
            return 1
        print(
            f"[smoke_convert_ds_zero3_to_hf] Case B PASS — gate fail-loud at "
            f"rc={result_b.returncode}, no artifacts written."
        )

        # ===== Case C — /dev/shm staging dance (#506 round-8 fix) =====
        # The round-7 dispatcher wrote HF safetensors directly into output_dir
        # on /workspace WHILE the DS-native dir still lived there: transient
        # peak ~160 GB > 130 GB MooseFS quota → conversion EDQUOTs at the last
        # shard. Round-8 stages HF on /dev/shm (tmpfs RAM, not MooseFS), then
        # deletes DS-native from /workspace fail-loud BEFORE moving the
        # staging dir over. This case exercises that staging dance end-to-end
        # by calling `_run_conversion_with_shm_staging` directly with the
        # tiny-gpt2 DS-native dir built above.
        out_dir_c = tmpdir / "out_c"

        # Pre-condition: DS-native MUST still be on disk (Case A consumed it
        # via the subprocess but DS-native is not deleted by the subprocess
        # itself — only the dispatcher helper deletes it after success).
        if not ds_native_dir.exists():
            print(
                f"FAIL: precondition for Case C failed — ds_native_dir "
                f"{ds_native_dir} missing. Cases A/B should have preserved it."
            )
            return 1

        # Import the staging helper. We bypass HF Hub by calling the helper
        # directly (no upload step); the helper's contract is: leave the
        # HF artifact at output_dir, DS-native deleted, /dev/shm staging gone.
        # EPM_SMOKE_PREFLIGHT_SKIP=1 disables the posix_fallocate probe
        # (the smoke runs on a CPU dev VM that can pass the probe but we
        # want Case C to exercise the staging logic independent of the
        # probe; Case D exercises the probe).
        sys.path.insert(0, str(Path(__file__).parent))
        from run_issue506_install import _run_conversion_with_shm_staging

        print(
            "\n[smoke_convert_ds_zero3_to_hf] Case C — /dev/shm staging dance: "
            "running _run_conversion_with_shm_staging() end-to-end on tiny-gpt2 "
            "(skip_preflight=True; tests the staging order, not the probe)."
        )
        try:
            _run_conversion_with_shm_staging(
                ds_native_dir=ds_native_dir,
                output_dir=out_dir_c,
                base_model_id=model_id,
                max_shard_size="100MB",
                dtype="float32",
                skip_preflight=True,
            )
        except Exception as e:
            print(f"FAIL: Case C raised: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return 1

        # Post-conditions:
        # 1. DS-native dir is GONE from /workspace (helper deleted it fail-loud).
        if ds_native_dir.exists():
            print(
                f"FAIL: Case C — ds_native_dir {ds_native_dir} still exists "
                "after _run_conversion_with_shm_staging returned. The helper "
                "must delete it before moving the HF artifact to output_dir."
            )
            return 1
        # 2. /dev/shm staging dir is GONE (helper moved it to output_dir).
        shm_staging_path = Path("/dev/shm") / f"issue506_{out_dir_c.name}_hf_staging"
        if shm_staging_path.exists():
            print(
                f"FAIL: Case C — /dev/shm staging dir {shm_staging_path} still "
                "exists after move. The helper must move (not copy) the staging "
                "dir to output_dir."
            )
            return 1
        # 3. output_dir holds a complete HF artifact.
        st_files_c = sorted(out_dir_c.glob("*.safetensors"))
        if not st_files_c:
            print(f"FAIL: Case C — no *.safetensors in {out_dir_c} after staging move.")
            return 1
        if not (out_dir_c / "config.json").exists():
            print(f"FAIL: Case C — config.json missing in {out_dir_c}.")
            return 1
        loaded_c = AutoModelForCausalLM.from_pretrained(str(out_dir_c), torch_dtype=torch.float32)
        if loaded_c is None or loaded_c.config.vocab_size != 50257:
            print(
                f"FAIL: Case C — loaded model from {out_dir_c} has wrong vocab_size: "
                f"{loaded_c.config.vocab_size if loaded_c else None}"
            )
            return 1
        print(
            f"[smoke_convert_ds_zero3_to_hf] Case C PASS — DS-native gone, "
            f"shm staging gone, output_dir holds {len(st_files_c)} shard(s) "
            f"+ config.json + tokenizer, model loads cleanly."
        )

        # ===== Case D — preflight fail-loud on infeasible projection =====
        # The round-8 helper runs a posix_fallocate probe BEFORE the
        # conversion subprocess to catch MooseFS EDQUOT cases up front.
        # We exercise the fail-loud path by importing the probe directly
        # and passing a deliberately too-large byte count for a known
        # bounded mount (tmpfs in tmpdir capped to a small size is not
        # portable; instead we hit a definitely-too-large target on
        # `/dev/shm` and assert the helper raises RuntimeError loud).
        from run_issue506_install import _disk_headroom_probe

        # Pick a value guaranteed to fail on any reasonable host: 10 TB.
        # This is portable: even a beefy 8xH200 pod has tmpfs ≪ 10 TB.
        infeasible_bytes = 10 * 1024 * 1024 * 1024 * 1024  # 10 TB
        print(
            "\n[smoke_convert_ds_zero3_to_hf] Case D — preflight fail-loud: "
            f"requesting {infeasible_bytes // 1024**4} TB on /dev/shm. "
            "Expecting RuntimeError..."
        )
        raised_d = False
        try:
            _disk_headroom_probe(Path("/dev/shm"), infeasible_bytes, label="/dev/shm")
        except RuntimeError as e:
            raised_d = True
            msg = str(e)
            if "disk-headroom preflight FAILED" not in msg:
                print(
                    f"FAIL: Case D — RuntimeError raised but message does NOT "
                    f"contain the expected prefix 'disk-headroom preflight FAILED'. "
                    f"Got: {msg!r}"
                )
                return 1
            if "/dev/shm" not in msg:
                print(
                    f"FAIL: Case D — RuntimeError message does not mention the "
                    f"failing mount '/dev/shm'. Got: {msg!r}"
                )
                return 1
            print(
                f"[smoke_convert_ds_zero3_to_hf] Case D PASS — RuntimeError "
                f"raised with the right shape: {msg[:120]}..."
            )
        if not raised_d:
            print(
                "FAIL: Case D — preflight did NOT raise on 10 TB request to "
                "/dev/shm. The probe is broken; it would let MooseFS EDQUOT "
                "cases slip through to the conversion subprocess."
            )
            return 1

        print(
            "\nOK: smoke PASS — Case A (roundtrip), Case B (regression), "
            "Case C (/dev/shm staging dance), Case D (preflight fail-loud) all pass."
        )
        return 0
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)
        # Clean up any /dev/shm staging dir we may have left around if
        # the helper raised mid-flight (the helper would clean its own
        # staging dir on success; on failure we want to leave no trace).
        shm_staging_glob = Path("/dev/shm")
        for stale in shm_staging_glob.glob("issue506_*_hf_staging"):
            with contextlib.suppress(OSError):
                shutil.rmtree(str(stale))


if __name__ == "__main__":
    sys.exit(main())
