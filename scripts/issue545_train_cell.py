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

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue545_train_cell")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _corpora_dir() -> Path:
    """Active corpora WRITE root (smoke-rooted under I545_SMOKE_OUTPUT=1)."""
    from explore_persona_space.experiments.behavior_testbed_545 import corpora_dir

    return corpora_dir()


def _corpus_read_path(name: str) -> Path:
    """Corpus READ resolution: active root first, production fallback (P0 inputs)."""
    from explore_persona_space.experiments.behavior_testbed_545 import corpus_read_path

    return corpus_read_path(name)


# ---------------------------------------------------------------------------
# GPU prep (vLLM) — its own process
# ---------------------------------------------------------------------------


def prep_corpus(row_id: str, arm: str, *, smoke: bool) -> None:  # noqa: C901 — three prep kinds, flat dispatcher
    """Materialize GPU-dependent corpora: marker rows, cn negatives, mix50.

    v2 mode (``I545_V2_OUTPUT=1``): rebuilt rows route to the elicit_v2
    builder (primary arm = the tiered ladder; cn = on-policy positives +
    v1-reused/greedy negatives; bridge = CPU canned-160 build).
    """
    from explore_persona_space.experiments.behavior_testbed_545 import v2_output_active

    if v2_output_active():
        _prep_v2(row_id, arm, smoke=smoke)
        return

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

    if row_id == "taught_fact" and arm == "cn":
        # B5 contrastive negatives are the #444 wrong-fact / refusal-pool
        # construction, BUILT AT P0 by corpora.build_fact_cn_corpus — NOT
        # on-policy greedy responses (contrastive-negatives rule: fact rows
        # emit a competing wrong-fact/refusal string). Codex round-1 minor:
        # the generic vLLM cn prep must not overwrite it.
        p0_built = _corpus_read_path("taught_fact_cn.jsonl")
        if not p0_built.exists():
            raise FileNotFoundError(
                f"P0-built fact cn corpus missing: {p0_built} (run --phase p0 --build-corpora)"
            )
        logger.info("[phase=prep] taught_fact/cn uses the P0-built %s — no GPU prep", p0_built)
        return

    if row_id == "marker":
        qpath = _corpus_read_path("marker_train_questions.json")
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
        from explore_persona_space.experiments.behavior_testbed_545.rows import (
            hydra_dataset_name,
        )

        if row.recipe_kind == "hydra_turner":
            src = PROJECT_ROOT / "data" / "issue404" / hydra_dataset_name(row, PROJECT_ROOT)
        else:
            # P0-built positives are frozen INPUTS: production fallback under smoke.
            src = _corpus_read_path(row.corpus)
        if not src.exists():
            raise FileNotFoundError(f"Positives corpus missing for cn prep: {src}")
        positives = [json.loads(line) for line in src.read_text().split("\n") if line.strip()]
        if cap:
            positives = positives[:cap]
        # Normalize positives to prompt/completion (train_lora's schema) and
        # extract the question. cn prep supports SINGLE-TURN corpora only:
        # the negative's greedy response is generated from the bare question,
        # so a multi-turn positive would silently lose its earlier context in
        # the negative (round-1 unaddressed-case fix: fail loud instead).
        norm_positives = [_to_prompt_completion(r) for r in positives]
        questions = []
        for r in norm_positives:
            users = [m for m in r["prompt"] if m.get("role") == "user"]
            if len(users) != 1:
                raise NotImplementedError(
                    f"cn prep for {row_id} found a {len(users)}-user-turn row; on-policy "
                    "negatives are only valid for single-turn corpora (the negative would "
                    "drop earlier turns)."
                )
            questions.append(users[0]["content"])
        responses = _greedy_responses(questions)
        out = cdir / f"{Path(src.name).stem}_cn.jsonl"
        with out.open("w") as f:
            for pos, q, r in zip(norm_positives, questions, responses, strict=False):
                f.write(json.dumps(pos) + "\n")
                f.write(json.dumps(_row_json(q, r.rstrip())) + "\n")
        logger.info("[phase=prep] wrote %s (1:1 positives:negatives)", out)
    elif arm == "mix50":
        src = PROJECT_ROOT / "data" / "issue404" / "turner_bad_medical_advice.jsonl"
        gen = _corpus_read_path("kl_aux_generic.jsonl")
        if not src.exists() or not gen.exists():
            raise FileNotFoundError(f"mix50 prep needs {src} and {gen}")
        pos = [json.loads(line) for line in src.read_text().split("\n") if line.strip()]
        generic = [json.loads(line) for line in gen.read_text().split("\n") if line.strip()]
        if cap:
            pos, generic = pos[:cap], generic[:cap]
        k = min(len(pos), len(generic))
        rng = random.Random(545)
        # Round-15 P0 fix: normalize EVERY row to messages-schema with plain-
        # string content AT MATERIALIZATION. The generic half is written in
        # train_lora's prompt/completion schema (LISTS of message dicts);
        # scripts/train.py's format_dataset treats prompt/completion as the
        # LEGACY STRING shape and wraps the lists directly as message content,
        # which crashes apply_chat_template's Jinja with "can only concatenate
        # str (not 'list') to str" (3 mix50 train crashes, 2026-06-11 p1 log).
        mixed = [_to_messages_str_row(r) for r in pos[:k] + generic[:k]]
        rng.shuffle(mixed)
        # Round-20: write through corpora_dir() (smoke-rooted under isolation —
        # a smoke blend must never overwrite the production one); the hydra
        # train threads the SAME resolved path via condition.stages.0.dataset.
        out = cdir / "badmed_mix50.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in mixed) + "\n")
        logger.info("[phase=prep] wrote %s (%d rows)", out, 2 * k)
    else:
        logger.info("[phase=prep] nothing to prep for %s/%s", row_id, arm)


def _prep_v2(row_id: str, arm: str, *, smoke: bool) -> None:
    """v2 prep dispatcher (onpolicy-testbed-v2 plan section 4.4).

    - ``primary``: run the row's elicitation ladder iff its corpus +
      pool_meta are absent (idempotent — pools are seed-invariant, both
      seeds share the corpus). A recorded quota MISS fails loud here: the
      row was supposed to be dropped by the dispatcher's pre-step.
    - ``cn`` (wrong_claim): build the interleaved 160+160 corpus, reusing
      v1 greedy negatives with greedy-regen fallback (A7).
    - ``bridge`` (compliment): CPU-only canned-160 build from the kept IDs.
    """
    from explore_persona_space.experiments.behavior_testbed_545 import output_root
    from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import (
        build_bridge_corpus,
        build_cn_corpus_v2,
        row_quota_met,
        run_elicitation,
    )

    smoke_n = 6 if smoke else None
    if arm == "bridge":
        build_bridge_corpus(row_id)  # CPU-only; needs the row's pool_meta
        return

    corpus = _corpus_read_path(f"onpolicy_{row_id}.jsonl")
    meta = output_root() / "elicitation" / f"{row_id}_pool_meta.json"
    if corpus.exists() and meta.exists():
        logger.info("[phase=prep] v2 elicitation already complete for %s", row_id)
    else:
        from vllm import LLM

        from explore_persona_space.experiments.behavior_testbed_545 import BASE_MODEL

        llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.85, max_model_len=4096)
        run_elicitation(row_id, llm, smoke_n=smoke_n)
        if arm == "cn":
            if row_quota_met(row_id):
                build_cn_corpus_v2(row_id, llm=llm)
            return
    if arm == "cn" and not (_corpora_dir() / f"onpolicy_{row_id}_cn.jsonl").exists():
        if not row_quota_met(row_id):
            raise RuntimeError(f"cn prep for {row_id}: row missed its quota (designed drop)")
        from vllm import LLM

        from explore_persona_space.experiments.behavior_testbed_545 import BASE_MODEL

        llm = LLM(model=BASE_MODEL, gpu_memory_utilization=0.85, max_model_len=4096)
        build_cn_corpus_v2(row_id, llm=llm)


def _content_to_str(content) -> str:
    """Normalize a chat message ``content`` to a plain string.

    ``apply_chat_template``'s Jinja template concatenates ``content`` into the
    rendered text, so list-shaped content (content-blocks) crashes with
    ``TypeError: can only concatenate str (not "list") to str``. Joins text
    blocks with newlines; fails loud on any non-text block type — a silent
    lossy drop would corrupt the training corpus.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type", "text") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
            else:
                tag = block.get("type") if isinstance(block, dict) else None
                raise ValueError(
                    f"Non-text content block (python type {type(block).__name__}, "
                    f"block type {tag!r}) — cannot normalize to a plain training string"
                )
        return "\n".join(parts)
    raise ValueError(f"Unsupported message content type: {type(content).__name__}")


def _to_messages_str_row(row: dict) -> dict:
    """Normalize a corpus row to ``{"messages": [...]}`` with str contents.

    Accepts messages-schema rows AND train_lora prompt/completion rows
    (``prompt`` = list of pre-completion messages, ``completion`` =
    ``[assistant turn]``). Every message's content is normalized to a plain
    string so the hydra training path (``scripts/train.py`` ->
    ``format_dataset`` -> ``apply_chat_template``) takes its well-tested
    ``messages`` branch — never the legacy STRING-shaped prompt/completion
    branch that wraps list values as message content (the round-15 mix50
    Jinja crash). Fails loud on unrecognized row shapes.
    """
    if "messages" in row:
        msgs = row["messages"]
    elif isinstance(row.get("prompt"), list) and isinstance(row.get("completion"), list):
        msgs = [*row["prompt"], *row["completion"]]
    else:
        raise ValueError(f"Cannot normalize row to messages schema: keys={sorted(row)}")
    return {
        "messages": [{"role": m["role"], "content": _content_to_str(m["content"])} for m in msgs]
    }


def _to_prompt_completion(row: dict) -> dict:
    """Normalize a messages-schema row to train_lora's prompt/completion schema.

    prompt = every message before the final assistant turn (full multi-turn
    context preserved); completion = the final assistant turn. Already-
    normalized rows pass through unchanged. Fails loud on rows that end on a
    non-assistant turn.
    """
    if "prompt" in row and "completion" in row:
        return row
    msgs = row.get("messages")
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError(f"Cannot normalize row to prompt/completion: keys={sorted(row)}")
    return {"prompt": msgs[:-1], "completion": [msgs[-1]]}


def _normalized_copy(src: Path) -> Path:
    """Write a prompt/completion-schema copy of a messages-schema JSONL.

    Used by the klreg arm, which trains the hydra row's POSITIVES through
    train_lora (messages-schema Turner JSONLs don't load there). Idempotent;
    lands under corpora_dir() so the bulk corpus upload captures it.
    """
    dst = _corpora_dir() / f"{src.stem}_pc.jsonl"
    rows = [json.loads(line) for line in src.read_text().split("\n") if line.strip()]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in rows:
            f.write(json.dumps(_to_prompt_completion(r)) + "\n")
    logger.info("[phase=train] schema-normalized %s -> %s (%d rows)", src.name, dst, len(rows))
    return dst


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_cell(row_id: str, arm: str, seed: int, gpu_id: int, *, smoke: bool) -> Path:
    from explore_persona_space.experiments.behavior_testbed_545 import (
        adapters_root,
        assert_marker_token,
    )
    from explore_persona_space.experiments.behavior_testbed_545.rows import (
        ARM_SPECS,
        get_row,
        resolve_training_dispatch,
    )

    assert arm in ARM_SPECS, f"Unknown arm {arm!r}"
    row = get_row(row_id)
    cell = row.cell_id(arm, seed)
    # Package-resolved root so smoke-output isolation (I545_SMOKE_OUTPUT)
    # applies identically here and in the dispatcher (round 19).
    out_root = adapters_root() / cell
    out_root.mkdir(parents=True, exist_ok=True)
    # Round-2 blocker fix: dispatch resolved by ARM FIRST (rows.py) — cn/klreg
    # on hydra rows train via train_lora + TURNER_PARITY, never the plain
    # hydra branch (which would silently re-train primary).
    dispatch = resolve_training_dispatch(row, arm, PROJECT_ROOT)

    if dispatch["path"] == "reuse":
        return _download_reused_adapter(row, seed, out_root)

    if dispatch["path"] == "fullft":
        import yaml

        # Derived per-cell stage config: seed from the cell, lr from the
        # I545_FULLFT_LR probe env ({2e-6,5e-6,1e-5}, plan section 11), tiny
        # caps under --smoke. The committed yaml carries the middle probe.
        base_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "condition" / f"{dispatch['condition']}.yaml").read_text()
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

    if dispatch["path"] == "hydra":
        condition = dispatch["condition"]
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
        if arm == "mix50":
            # The blend is prep-materialized under corpora_dir() (smoke-rooted
            # under isolation); the yaml's repo-relative default is only valid
            # for the production default root, so thread the resolved path.
            cmd.append(f"condition.stages.0.dataset={_corpus_read_path('badmed_mix50.jsonl')}")
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

    # train_lora path (train_lora rows + cn/klreg arms on hydra rows).
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    overrides = dict(dispatch["overrides"])
    data_path = Path(dispatch["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Training corpus missing: {data_path} (P0/prep incomplete)")
    if dispatch.get("needs_schema_normalization"):
        data_path = _normalized_copy(data_path)
    if overrides.get("kl_aux_data_path") == "GENERIC_CHAT":
        overrides["kl_aux_data_path"] = str(_corpus_read_path("kl_aux_generic.jsonl"))
    if smoke:
        overrides = {**overrides, "max_steps": 4, "save_strategy": "no", "save_steps": 0}
        overrides.pop("save_total_limit", None)
    if row_id == "warmth" and not smoke:
        # save at half-epoch multiples: steps_per_epoch // 2 (dose {0.5,1,2,4}).
        n_rows = sum(1 for line in data_path.read_text().split("\n") if line.strip())
        eff_batch = overrides.get("batch_size", 4) * overrides.get("grad_accum", 4)
        overrides["save_steps"] = max(1, n_rows // eff_batch // 2)

    if row_id == "marker":
        from transformers import AutoTokenizer

        assert_marker_token(
            AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        )

    from explore_persona_space.experiments.behavior_testbed_545 import v2_output_active

    if v2_output_active() and not smoke:
        # Plan v3 section 4.2 trap note: GENERIC_RECIPE carries
        # save_total_limit=3 — a v2 cell training 6 epochs with limit < 6
        # silently rotates away the epoch-1..3 checkpoints and dose-select
        # only ever sees the last 3 epochs. Assert the v2 recipe override
        # actually reached this call site.
        epochs = overrides.get("epochs", 3)
        limit = overrides.get("save_total_limit") or 0
        assert limit >= epochs, (
            f"v2 cell {cell}: save_total_limit={limit} < epochs={epochs} — HF Trainer "
            "checkpoint rotation would delete early-epoch checkpoints (plan section 4.2)"
        )
    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        seed=seed,
        # Plan section 10: v2 WandB run names are ``v2_<cell>`` (same project).
        run_name=f"v2_{cell}" if v2_output_active() else f"i545_{cell}",
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


# Verified-authentic #503 Bucket-D adapter fingerprint (round 23, 2026-06-12).
# Grounded on the ARTIFACT ground truth, NOT the #503 clean-result body: every
# issue503_bucket_d_* adapter_config.json on HF (superkaiba1/explore-persona-space)
# uniformly carries this config — the authentic product of the turner_em recipe
# (lr=2e-5, alpha=256/scaling=8, adamw_8bit) used across the #404/#458/#503 line
# (see the PAIRS provenance comments in scripts/issue404_common.py).
_B8_EXPECTED_LORA = {
    "r": 32,
    "lora_alpha": 256,
    "lora_dropout": 0.0,
    "target_modules": frozenset(
        {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ),
}


def _download_reused_adapter(row, seed: int, out_root: Path) -> Path:
    """B8 reuse: per-file hf_hub_download (NEVER snapshot allow_patterns — the
    siblings-truncation trap on large repos), then fitness asserts.

    Fitness-expectation provenance (round 23, 2026-06-12). The expected LoRA
    config (``_B8_EXPECTED_LORA``) is grounded on the artifact configs verified
    directly on HF: all ``issue503_bucket_d_*`` adapters carry r=32,
    lora_alpha=256, lora_dropout=0.0, target_modules = the full 7-proj set.
    ``adapter_config.json`` is written by the training run itself and is the
    ground truth for what an artifact IS. #503's clean-result body
    Reproducibility row ("r = 16, alpha = 32, dropout = 0.05, target = q/k/v/o")
    is a DOCUMENTATION ERROR and must NOT be used as the fitness ground — a
    record-correction marker was posted on #503 (epm:progress v2, 2026-06-12).
    Do NOT "fix" this check back to the body row's values: that expectation
    crashed all 7 B8 cells in P2 (epm:failure v8 on #545, 2026-06-12). The
    scientific requirement is to reuse EXACTLY the artifacts that produced
    #503's measured AdvBench lifts, which are these r=32/alpha=256 artifacts.
    lora_alpha=256 also distinguishes this family from the unrelated
    ``issue503_broad_syco_*`` dirs (r=32/alpha=64), which B8 must never match.
    """
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
    exp = _B8_EXPECTED_LORA
    got = {
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "lora_dropout": cfg.get("lora_dropout"),
        "target_modules": frozenset(cfg.get("target_modules") or []),
    }
    assert got == exp, (
        "B8 fitness check failed: expected the verified #503 Bucket-D artifact config "
        f"(r={exp['r']} alpha={exp['lora_alpha']} dropout={exp['lora_dropout']} "
        f"targets={sorted(exp['target_modules'])}), got "
        f"r={got['r']} alpha={got['lora_alpha']} dropout={got['lora_dropout']} "
        f"targets={sorted(got['target_modules'])}"
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

    # Pin the GPU restriction BEFORE any heavy import: `import peft` (and
    # anything else that initializes the CUDA driver) freezes the process's
    # visible-device list, after which in-process CUDA_VISIBLE_DEVICES sets
    # are silently ignored and every train lands on physical GPU 0 (the
    # round-10 4-trains-on-one-device OOM). Setting the env var here — at
    # entry, pre-import — is import-order-proof, and the hydra/fullft child
    # subprocesses inherit it via env={**os.environ}. The fullft arm is the
    # one multi-GPU path (ZeRO-3 over all GPUs) and must stay unrestricted.
    if args.arm != "fullft":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.smoke:
        # Defense in depth (round 19): the dispatcher already exports
        # I545_SMOKE_OUTPUT=1 for --smoke runs; a STANDALONE --smoke
        # invocation must isolate its outputs the same way so a smoke
        # adapter can never shadow a production cell artifact.
        from explore_persona_space.experiments.behavior_testbed_545 import SMOKE_OUTPUT_ENV

        os.environ[SMOKE_OUTPUT_ENV] = "1"

    if args.prep_only:
        prep_corpus(args.row, args.arm, smoke=args.smoke)
        return 0
    train_cell(args.row, args.arm, args.seed, args.gpu_id, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
