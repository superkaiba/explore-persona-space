#!/usr/bin/env python3
"""Issue #545 — one training cell (row, arm, seed) on one GPU.

Invoked by ``scripts/issue545_sweep.py`` as a subprocess per cell (explicit
``env={**os.environ}``). Two modes:

- ``--prep-only``: GPU corpus prep with vLLM (marker base-greedy responses +
  appended marker; on-policy default-context cn negatives; mix50 blend), then
  EXIT. Run as its OWN process so vLLM never shares a process with HF
  training (the vLLM-teardown gotcha).
- default: training. ``hydra_turner`` rows shell out to ``scripts/train.py``
  (the #458 launch verbatim); ``train_lora`` rows call ``train_lora()``
  in-process; ``reuse_adapter`` rows download + verify the #503 adapter;
  ``fullft`` shells out to ``scripts/launch_stage.py`` (ZeRO-3).

Smoke parity: ``--smoke`` caps rows/steps but goes through the IDENTICAL code
path (no separate smoke trainer).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

if Path("/workspace").exists():  # pod-only cache redirect; VM keeps its default
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue545_train_cell")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _corpora_dir() -> Path:
    from explore_persona_space.experiments.behavior_testbed_545 import corpora_dir

    return corpora_dir()


# ---------------------------------------------------------------------------
# GPU prep (vLLM) — its own process
# ---------------------------------------------------------------------------


def prep_corpus(row_id: str, arm: str, *, smoke: bool) -> None:  # noqa: C901 — three prep kinds, flat dispatcher
    """Materialize GPU-dependent corpora: marker rows, cn negatives, mix50."""
    from vllm import LLM, SamplingParams

    from explore_persona_space.experiments.behavior_testbed_545 import (
        BASE_MODEL,
        MARKER_TEXT,
        assert_marker_token,
    )
    from explore_persona_space.experiments.behavior_testbed_545.rows import get_row

    row = get_row(row_id)
    cdir = _corpora_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    cap = 8 if smoke else None

    def _greedy_responses(questions: list[str]) -> list[str]:
        llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.85, max_model_len=4096)
        tok = llm.get_tokenizer()
        assert_marker_token(tok)
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
            for q in questions
        ]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
        n_trunc = sum(1 for o in outs for c in o.outputs if c.finish_reason == "length")
        logger.info("[phase=prep] %d greedy responses (truncated=%d)", len(outs), n_trunc)
        return [o.outputs[0].text for o in outs]

    def _row_json(question: str, answer: str) -> dict:
        return {
            "prompt": [{"role": "user", "content": question}],
            "completion": [{"role": "assistant", "content": answer}],
        }

    if row_id == "marker":
        qpath = cdir / "marker_train_questions.json"
        if not qpath.exists():
            raise FileNotFoundError(f"P0 question split missing: {qpath}")
        questions = json.loads(qpath.read_text())["questions"][:cap]
        responses = _greedy_responses(questions)
        out = cdir / ("marker_train.jsonl" if arm == "primary" else "marker_train_cn.jsonl")
        with out.open("w") as f:
            for q, r in zip(questions, responses, strict=False):
                f.write(json.dumps(_row_json(q, r.rstrip() + MARKER_TEXT)) + "\n")
                if arm == "cn":
                    # Negative: same question, marker-less response. Under
                    # marker-only loss with suppress_at_post_response_slot the
                    # loss-bearing token is EOS at the slot (A28).
                    f.write(json.dumps(_row_json(q, r.rstrip())) + "\n")
        logger.info("[phase=prep] wrote %s (%d questions)", out, len(questions))
    elif arm == "cn":
        # On-policy default-context negatives on the row's own questions.
        if row.recipe_kind == "hydra_turner":
            src = PROJECT_ROOT / "data" / "issue404" / _hydra_dataset_stem(row)
        else:
            src = cdir / row.corpus
        if not src.exists():
            raise FileNotFoundError(f"Positives corpus missing for cn prep: {src}")
        positives = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        if cap:
            positives = positives[:cap]
        questions = []
        for r in positives:
            msgs = r.get("prompt") or r.get("messages") or []
            users = [m for m in msgs if m.get("role") == "user"]
            questions.append(users[-1]["content"])
        responses = _greedy_responses(questions)
        out = cdir / f"{Path(src.name).stem}_cn.jsonl"
        with out.open("w") as f:
            for pos, q, r in zip(positives, questions, responses, strict=False):
                f.write(json.dumps(pos) + "\n")
                f.write(json.dumps(_row_json(q, r.rstrip())) + "\n")
        logger.info("[phase=prep] wrote %s (1:1 positives:negatives)", out)
    elif arm == "mix50":
        src = PROJECT_ROOT / "data" / "issue404" / "turner_bad_medical_advice.jsonl"
        gen = cdir / "kl_aux_generic.jsonl"
        if not src.exists() or not gen.exists():
            raise FileNotFoundError(f"mix50 prep needs {src} and {gen}")
        pos = [line for line in src.read_text().splitlines() if line.strip()]
        generic = [line for line in gen.read_text().splitlines() if line.strip()]
        if cap:
            pos, generic = pos[:cap], generic[:cap]
        k = min(len(pos), len(generic))
        rng = random.Random(545)
        mixed = pos[:k] + generic[:k]
        rng.shuffle(mixed)
        out_dir = PROJECT_ROOT / "data" / "issue545"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "badmed_mix50.jsonl").write_text("\n".join(mixed) + "\n")
        logger.info("[phase=prep] wrote badmed_mix50.jsonl (%d rows)", 2 * k)
    else:
        logger.info("[phase=prep] nothing to prep for %s/%s", row_id, arm)


def _hydra_dataset_stem(row) -> str:
    import yaml

    cond = PROJECT_ROOT / "configs" / "condition" / f"{row.hydra_condition}.yaml"
    return Path(yaml.safe_load(cond.read_text())["stages"][0]["dataset"]).name


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_cell(row_id: str, arm: str, seed: int, gpu_id: int, *, smoke: bool) -> Path:
    from explore_persona_space.experiments.behavior_testbed_545 import assert_marker_token
    from explore_persona_space.experiments.behavior_testbed_545.rows import (
        ARM_SPECS,
        get_row,
    )

    row = get_row(row_id)
    cell = row.cell_id(arm, seed)
    out_root = Path(os.environ.get("EPM_OUTPUT_ROOT", "/tmp/issue545")) / "adapters" / cell
    out_root.mkdir(parents=True, exist_ok=True)
    arm_spec = ARM_SPECS[arm]

    if row.recipe_kind == "reuse_adapter":
        return _download_reused_adapter(row, seed, out_root)

    if arm == "fullft":
        import yaml

        # Derived per-cell stage config: seed from the cell, lr from the
        # I545_FULLFT_LR probe env ({2e-6,5e-6,1e-5}, plan section 11), tiny
        # caps under --smoke. The committed yaml carries the middle probe.
        base_cfg = yaml.safe_load(
            (
                PROJECT_ROOT / "configs" / "condition" / f"{arm_spec['fullft_condition']}.yaml"
            ).read_text()
        )
        base_cfg["seed"] = seed
        base_cfg["wandb_run_name"] = f"i545_{cell}"
        if os.environ.get("I545_FULLFT_LR"):
            base_cfg["learning_rate"] = float(os.environ["I545_FULLFT_LR"])
        if smoke:
            base_cfg["max_steps"] = 4
        stage_cfg = out_root / "stage_config.yaml"
        stage_cfg.write_text(yaml.safe_dump(base_cfg, sort_keys=False))
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/launch_stage.py",
            "--stage-config",
            str(stage_cfg),
            "--output-dir",
            str(out_root),
            "--num-gpus",
            "4",
            "--backend",
            "local",
        ]
        logger.info("[phase=train] fullft: %s", shlex.join(cmd))
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env={**os.environ})
        if not (out_root / "config.json").exists():
            raise FileNotFoundError(
                f"fullft training finished but no full model at {out_root}/config.json"
            )
        return out_root

    if row.recipe_kind == "hydra_turner" or arm == "mix50":
        condition = arm_spec.get("hydra_condition", row.hydra_condition)
        max_steps = 4 if smoke else 375
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/train.py",
            f"condition={condition}",
            "training=turner_em",
            "lora=turner_em",
            f"+training.max_steps={max_steps}",
            "training.save_strategy=steps",
            "+training.save_steps=125",
            "training.save_total_limit=3",  # keep all of 125/250/375 (yaml default is 2)
            "upload_to=none",  # dispatcher owns the bulk ADAPTER upload; never the 15GB merged
            f"seed={seed}",
            f"+gpu_id={gpu_id}",
        ]
        logger.info("[phase=train] hydra: %s", shlex.join(cmd))
        subprocess.run(
            cmd,
            check=True,
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                # Keep the adapter tree (final adapter + checkpoint-125/250/375
                # saved inside it) so we can relocate it below; default trainer
                # behavior reaps it after the merge (train/trainer.py fence).
                "EPM_KEEP_ADAPTER_DIR": "1",
                # The merged 15GB model is derived data — never upload it.
                "EPM_SKIP_INLINE_CHECKPOINT_UPLOAD": "1",
                "WANDB_PROJECT": "issue545_behavior_testbed",
            },
        )
        _relocate_hydra_adapter(condition, seed, out_root)
        return out_root

    # train_lora path.
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    overrides = dict(arm_spec.get("train_lora_overrides") or row.train_lora_overrides)
    corpus_name = row.corpus
    if arm == "cn":
        corpus_name = f"{Path(row.corpus).stem}_cn.jsonl"
        overrides = (
            {**overrides, **arm_spec.get("marker_extra", {})} if row_id == "marker" else overrides
        )
    data_path = _corpora_dir() / corpus_name
    if not data_path.exists():
        raise FileNotFoundError(f"Training corpus missing: {data_path} (P0/prep incomplete)")
    if overrides.get("kl_aux_data_path") == "GENERIC_CHAT":
        overrides["kl_aux_data_path"] = str(_corpora_dir() / "kl_aux_generic.jsonl")
    if smoke:
        overrides = {**overrides, "max_steps": 4, "save_strategy": "no", "save_steps": 0}
        overrides.pop("save_total_limit", None)
    if row_id == "warmth" and not smoke:
        # save at half-epoch multiples: steps_per_epoch // 2 (dose {0.5,1,2,4}).
        n_rows = sum(1 for line in data_path.read_text().splitlines() if line.strip())
        eff_batch = overrides.get("batch_size", 4) * overrides.get("grad_accum", 4)
        overrides["save_steps"] = max(1, n_rows // eff_batch // 2)

    if row_id == "marker":
        from transformers import AutoTokenizer

        assert_marker_token(
            AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        )

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        seed=seed,
        run_name=f"i545_{cell}",
        report_to="wandb",
        hf_upload=False,  # bulk per-phase upload by the dispatcher (256-commit/hr rule)
        **overrides,
    )
    os.environ["WANDB_PROJECT"] = "issue545_behavior_testbed"
    out_dir, loss = train_lora("Qwen/Qwen2.5-7B-Instruct", str(data_path), str(out_root), cfg=cfg)
    (out_root / "train_result.json").write_text(
        json.dumps({"cell": cell, "loss": loss, "data_path": str(data_path)}, indent=1)
    )
    logger.info("[phase=train] %s done (loss=%.4f)", cell, loss)
    return Path(out_dir)


def _relocate_hydra_adapter(condition: str, seed: int, out_root: Path) -> None:
    """Move the staged-training LoRA adapter tree into the dispatcher layout.

    ``scripts/train.py`` writes ``<MED_OUTPUT_DIR|repo>/models/<condition>_seed<S>/
    sft_narrow_adapter`` (final adapter + the HF Trainer's ``checkpoint-*`` dirs
    saved inside it; kept on disk via ``EPM_KEEP_ADAPTER_DIR=1``) plus the ~15GB
    ``sft_narrow_merged``. The dispatcher evals adapters at
    ``EPM_OUTPUT_ROOT/adapters/<cell>`` via vLLM LoRARequest and dose-selects
    over ``checkpoint-*`` — so move the adapter tree there, then reap the whole
    run dir (the merged model is derived data, regenerable from base+adapter).
    """
    import shutil

    models_root = (
        Path(os.environ.get("MED_OUTPUT_DIR", str(PROJECT_ROOT)))
        / "models"
        / f"{condition}_seed{seed}"
    )
    src_adapter = models_root / "sft_narrow_adapter"
    if not (src_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Expected LoRA adapter at {src_adapter} after scripts/train.py "
            "(EPM_KEEP_ADAPTER_DIR=1 should have kept it) — refusing to continue."
        )
    out_root.mkdir(parents=True, exist_ok=True)
    for item in src_adapter.iterdir():
        dest = out_root / item.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(models_root)  # reaps sft_narrow_merged (15GB) + leftovers
    n_ckpts = len(list(out_root.glob("checkpoint-*")))
    logger.info("[phase=train] adapter relocated -> %s (%d checkpoints)", out_root, n_ckpts)


def _download_reused_adapter(row, seed: int, out_root: Path) -> Path:
    """B8 reuse: per-file hf_hub_download (NEVER snapshot allow_patterns — the
    siblings-truncation trap on large repos), then fitness asserts."""
    from huggingface_hub import hf_hub_download, list_repo_files

    from explore_persona_space.experiments.behavior_testbed_545 import HF_MODEL_REPO

    sub = row.reuse_subfolders.get(seed)
    if sub is None:
        raise KeyError(f"No reuse subfolder for {row.row_id} seed {seed}")
    files = [f for f in list_repo_files(HF_MODEL_REPO) if f.startswith(sub + "/")]
    if not any(f.endswith("adapter_config.json") for f in files):
        raise FileNotFoundError(f"{sub} has no adapter_config.json on {HF_MODEL_REPO}")
    for f in files:
        local = hf_hub_download(HF_MODEL_REPO, f)
        dest = out_root / Path(f).relative_to(sub)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(Path(local).read_bytes())
    cfg = json.loads((out_root / "adapter_config.json").read_text())
    assert cfg.get("r") == 16 and cfg.get("lora_alpha") == 32, (
        f"B8 fitness check failed: expected r=16 alpha=32 (#503 recipe), got "
        f"r={cfg.get('r')} alpha={cfg.get('lora_alpha')}"
    )
    logger.info("[phase=train] reused #503 adapter -> %s (%d files)", out_root, len(files))
    return out_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #545 single training cell")
    parser.add_argument("--row", required=True)
    parser.add_argument("--arm", default="primary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="tiny caps, same code path")
    parser.add_argument("--prep-only", action="store_true", help="vLLM corpus prep, then exit")
    args = parser.parse_args()

    if args.prep_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        prep_corpus(args.row, args.arm, smoke=args.smoke)
        return 0
    train_cell(args.row, args.arm, args.seed, args.gpu_id, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
