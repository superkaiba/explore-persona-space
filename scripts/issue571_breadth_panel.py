# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, Δ, −) in scientific docstrings + labels.
"""Task #571 — breadth-ablation eval driver (pod-side phases).

Scores the 4 freshly trained #571 adapters (broad/narrow panel x seeds
42/43, all source A2) on the #560 eval surface UNCHANGED: 35 held-out
personas x 20 eval questions @ pinned #478 rev a9fc5a9, on-policy greedy
generation per adapter, then four-float corrected-slot reads on the
trained AND base sides (#532 machinery, imported — not copied).

Keyed by an EXPLICIT adapter registry ``{label: (cid, hf_subfolder)}`` —
the #560 driver's cid-keyed cells would collide for 4 adapters sharing
cid A2 (plan §3.2 item 2). Each label gets a unique ``lora_int_id``.

Phases (each invocation runs ONE phase as a fresh subprocess — the
vLLM-after-Trainer / vLLM-then-HF teardown contract):

- ``smoke``        CPU gates (tokenizer ids, pinned a9fc5a9 artifact,
                   persona/exposure asserts, adapter-registry validation,
                   #532 slot-job construction) + GPU gates unless
                   ``--cpu-only``: gauge assert + #532 scoring-path
                   reference gate (i474 A2 anchor adapter, MAE < 0.5 nat,
                   Spearman > 0.995) + #534 vLLM-LoRA application gate.
- ``gen``          vLLM greedy (temp 0.0, max_tokens 2048) of each
                   adapter's own R; checkpoint-per-adapter, resume-skip.
- ``score-trained`` / ``score-base``  HF forward four-float reads at the
                   corrected slot (pre_marker | end_of_response) on the
                   SAME R; slot-kind/truncation parity asserted when both
                   sides exist.
- ``source-gen``   vLLM: 20 Q_test under the A2 condition prompt, adapter
                   ON vs base OFF — the on-policy emission half of the
                   manipulation check. FAIL (ON < 0.2) exits rc=4.
- ``source-score`` HF forwards: source four-float reads (trained + base,
                   same ON-text) -> per-label files + merged
                   ``source_check.json``. Split from source-gen so vLLM
                   and HF never share a process.
- ``upload``       Fail-loud HF data-repo upload (raw completions +
                   four-float + source check + train_diag + train_rows),
                   then the end-of-run sentinel
                   ``/workspace/logs/issue-571-run-complete.json``
                   (poll_pipeline schema; suppressed under --tag).

Smoke/sweep parity: the smoke IS this driver restricted to one cell
(``--adapters narrow_s42 --personas assistant --n-questions 2 --tag
smoke``) — same script, same functions; every phase's cell list derives
from the same ``--adapters/--personas/--n-questions`` arguments. A
restricted panel REQUIRES ``--tag`` so partial files never poison the
production resume-skip globs (#560 pattern). Pod-side code NEVER shells
out to scripts/task.py — the sentinel file is the only channel.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Reused #532 scoring machinery (import, don't copy).
from issue532_followup_logp_slot import (  # noqa: E402
    EOS_ID,
    _download_adapters,
    _run_slot_batches,
    _slot_job,
    _summarize,
)

# Reused #560 panel surface (import, don't copy).
from issue560_crossrecipe_panel import (  # noqa: E402
    GEN_MAX_MODEL_LEN,
    GEN_MAX_TOKENS,
    HELD_OUT_35,
    I532_CELL_A2_A4,
    I532_DIAGONAL_A2,
    I532_FOLLOWUP_A2_A4,
    N_QUESTIONS_FULL,
    _build_vllm_engine,
    _gauge_assert,
    _validate_existing,
    assert_held_out_matches_logit_rescore,
    build_persona_prompt,
    classify_exposure,
    load_eval_questions,
    load_persona_prompts,
    load_tokenizer,
    panel_spec,
)

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    MARKER_ID,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("issue571.breadth_panel")

SCHEMA_VERSION = "issue571_breadth_v1"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_BUCKET = "issue571_breadth"
SOURCE_CID = "A2"
N_SOURCE_CHECK_QUESTIONS = 20
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/issue_571"
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results/issue_571"
TRAIN_DIAG_DIR = PROJECT_ROOT / "eval_results/issue_571/train_diag"
TRAIN_ROW_DIR = PROJECT_ROOT / "data/issue_571/train_rows"
ADAPTER_CACHE = (
    Path("/workspace/adapters/i571")
    if os.path.isdir("/workspace")
    else PROJECT_ROOT / "data/issue_571/adapter_cache"
)

# Explicit adapter registry — the single source of truth for the 4 cells.
# {label: (cid, hf_subfolder)}; labels are filesystem/HF-path safe.
ADAPTER_REGISTRY: dict[str, tuple[str, str]] = {
    "broad_s42": ("A2", "adapters/i571_broad_A2_s42_ep1"),
    "broad_s43": ("A2", "adapters/i571_broad_A2_s43_ep1"),
    "narrow_s42": ("A2", "adapters/i571_narrow_A2_s42_ep1"),
    "narrow_s43": ("A2", "adapters/i571_narrow_A2_s43_ep1"),
}
ALL_LABELS = list(ADAPTER_REGISTRY)
# Unique lora_int_id per label (4 labels share cid A2 — the #560 driver's
# enumerate-by-cid int ids would collide; consistency-checker advisory).
LORA_INT_IDS: dict[str, int] = {label: i for i, label in enumerate(ALL_LABELS, start=1)}

# Manipulation-check thresholds (#534 gate values, plan §7 assert 5).
EMISSION_ON_PASS = 0.8
EMISSION_ON_FAIL = 0.2
EMISSION_OFF_MAX = 0.1


def validate_registry() -> None:
    """Adapter-registry invariants (smoke gate): labels, cids, namespaces, ids."""
    assert len(ADAPTER_REGISTRY) == 4, ADAPTER_REGISTRY
    for label, (cid, subfolder) in ADAPTER_REGISTRY.items():
        assert cid == SOURCE_CID, (label, cid)
        assert subfolder.startswith("adapters/i571_"), (label, subfolder)
        assert "i474" not in subfolder, (label, subfolder)
        panel, seed = label.split("_s")
        assert panel in ("broad", "narrow") and seed in ("42", "43"), label
        assert subfolder == f"adapters/i571_{panel}_A2_s{seed}_ep1", (label, subfolder)
    assert len(set(LORA_INT_IDS.values())) == len(LORA_INT_IDS), LORA_INT_IDS


def _git_commit() -> str:
    """Short git commit hash of the repo this script runs from."""
    try:
        return (
            subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def result_metadata(extra: dict | None = None) -> dict:
    """Reproducibility metadata block for every output JSON."""
    meta = {
        "task": 571,
        "script": "issue571_breadth_panel.py",
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "base_model": BASE_MODEL,
        "argv": sys.argv[1:],
    }
    if extra:
        meta.update(extra)
    return meta


def _download_i571_adapters(labels: list[str]) -> dict[str, str]:
    """Per-file HF download of each #571 ep1 adapter, keyed by LABEL.

    Modeled on the #532 ``_download_adapters`` recipe but reads subfolders
    from ``ADAPTER_REGISTRY`` (the #532 helper hardcodes
    ``adapters/i474_loc_{cid}_ep1``).
    """
    from huggingface_hub import hf_hub_download

    ADAPTER_CACHE.mkdir(parents=True, exist_ok=True)
    needed_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    out: dict[str, str] = {}
    for label in labels:
        _, subfolder = ADAPTER_REGISTRY[label]
        local_target = ADAPTER_CACHE / subfolder
        local_target.mkdir(parents=True, exist_ok=True)
        for fname in needed_files:
            try:
                hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    revision="main",
                    filename=f"{subfolder}/{fname}",
                    local_dir=ADAPTER_CACHE,
                )
            except Exception as e:
                if fname in ("adapter_model.safetensors", "adapter_config.json"):
                    raise RuntimeError(
                        f"required file {subfolder}/{fname} not on HF for {label}: {e}"
                    ) from e
                logger.debug("optional file %s/%s missing on HF", subfolder, fname)
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"adapter_model.safetensors missing at {local_target} after hf_hub_download."
            )
        out[label] = str(local_target)
    return out


def _resolve_dirs(args) -> tuple[Path, Path]:
    """(data_dir, out_dir), tag-suffixed when a restricted panel is requested."""
    restricted = (
        sorted(args.adapters) != sorted(ALL_LABELS)
        or sorted(args.personas) != HELD_OUT_35
        or args.n_questions != N_QUESTIONS_FULL
    )
    if restricted and not args.tag:
        raise SystemExit(
            "A restricted panel (--adapters/--personas/--n-questions below the full "
            "4x35x20) REQUIRES --tag (e.g. --tag smoke) so partial files never "
            "collide with production outputs."
        )
    data_dir = args.data_dir / args.tag if args.tag else args.data_dir
    out_dir = args.out_dir / args.tag if args.tag else args.out_dir
    return data_dir, out_dir


def _panel_spec_with_adapters(args, questions: list[str]) -> dict:
    """The #560 ``panel_spec`` keyed by adapter labels instead of source cids."""
    spec = panel_spec(args.adapters, args.personas, questions)
    spec["adapters"] = spec.pop("sources")
    return spec


# ── Phase 0 — smoke gates ──────────────────────────────────────────────────


def phase_smoke(args) -> None:
    """Launch-precondition gates. CPU gates always; GPU gates unless --cpu-only.

    GPU gate order mirrors #560: HF scoring-path gate first, free the HF
    model, vLLM gate LAST (vLLM teardown is unreliable; the process exits
    right after).
    """
    print("[phase=p0_smoke]", flush=True)

    # (a) tokenizer ids (marker 83399, <|im_end|> 151645, single-token bare ※).
    tokenizer, bare_marker_id = load_tokenizer()
    logger.info("(a) tokenizer asserts PASS (marker=%d, eos=%d)", MARKER_ID, EOS_ID)

    # (a2) adapter-registry invariants.
    validate_registry()
    logger.info("(a2) adapter registry PASS (%d labels, unique lora_int_ids)", len(ALL_LABELS))

    # (b) pinned #478 raw artifact @ a9fc5a9, both cells + spec.held_out.
    questions = load_eval_questions(N_QUESTIONS_FULL, check_both_cells=True)
    logger.info("(b) pinned raw artifact PASS (%d questions, 2 cells)", len(questions))

    # (b2) panel identity + exposure classification (assistant≡A1,
    # comedian≡A4, villain≡A5) -> 32-persona never-negative primary set.
    assert_held_out_matches_logit_rescore()
    persona_prompts = load_persona_prompts()
    matches = classify_exposure(persona_prompts)
    never_neg = [p for p in HELD_OUT_35 if p not in matches]
    assert len(never_neg) == 32, (len(never_neg), sorted(matches))
    logger.info("(b2) exposure classification PASS (32 never-negative personas)")

    # (b3) CPU slot-job construction on the committed #532 fixture R.
    fixture = json.loads(I532_CELL_A2_A4.read_text())
    q_test = load_q_test_extended_50()
    class_d = load_class_d_rewrites()
    r_list = fixture["R_trained_per_q"]
    assert len(r_list) == 50, len(r_list)
    jobs = [
        _slot_job(
            build_prompt_for_condition(
                CONDITIONS_BY_ID["A4"], q, tokenizer, class_d_rewrites=class_d
            ),
            r,
            tokenizer,
            bare_marker_id,
        )
        for q, r in zip(q_test[:5], r_list[:5], strict=False)
    ]
    assert all(len(j["full_ids"]) > 0 for j in jobs)
    assert all(j["slot_kind"] in ("pre_marker", "end_of_response") for j in jobs)
    logger.info(
        "(b3) slot-job construction PASS (5 jobs, slot kinds: %s)",
        sorted({j["slot_kind"] for j in jobs}),
    )

    if args.cpu_only:
        logger.warning("(c)/(d) SKIPPED — --cpu-only (GPU gates run pod-side)")
        return

    # (c) #532 scoring-path reference gate vs the committed A2__A4 cell,
    # using the i474 A2 anchor adapter (Hub-verified during planning; NOT
    # retrained — smoke scoring-path reference only, plan §10).
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    ref_adapter_dirs = _download_adapters([SOURCE_CID])
    _gauge_assert(ref_adapter_dirs)

    committed = json.loads(I532_FOLLOWUP_A2_A4.read_text())
    committed_logp = np.array([r["logp_marker"] for r in committed["per_q"]], dtype=np.float64)
    assert len(committed_logp) == 50

    full_jobs = [
        _slot_job(
            build_prompt_for_condition(
                CONDITIONS_BY_ID["A4"], q, tokenizer, class_d_rewrites=class_d
            ),
            r,
            tokenizer,
            bare_marker_id,
        )
        for q, r in zip(q_test, r_list, strict=True)
    ]
    logger.info("(c) loading base model + i474 A2 anchor adapter for the scoring-path gate ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base.eval()
    peft_model = PeftModel.from_pretrained(base, ref_adapter_dirs[SOURCE_CID])
    peft_model.eval()
    reads = _run_slot_batches(peft_model, tokenizer, full_jobs, bare_marker_id, label="smoke/c")
    got_logp = np.array([r["logp_marker"] for r in reads], dtype=np.float64)
    mae = float(np.mean(np.abs(got_logp - committed_logp)))
    from scipy.stats import spearmanr

    rho = float(spearmanr(got_logp, committed_logp)[0])
    assert mae < 0.5, f"(c) scoring-path MAE {mae:.4f} nat >= 0.5 vs committed A2__A4"
    assert rho > 0.995, f"(c) scoring-path Spearman {rho:.5f} <= 0.995 vs committed A2__A4"
    logger.info("(c) scoring-path gate PASS (MAE=%.4f nat, Spearman=%.5f)", mae, rho)

    base = peft_model.unload()
    del peft_model, base
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # (d) #534 vLLM-LoRA application gate vs the committed A2 diagonal.
    diagonal = json.loads(I532_DIAGONAL_A2.read_text())
    assert diagonal["summary"]["in_R_emission_rate"] == 1.0, diagonal["summary"]
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    llm = _build_vllm_engine()
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=GEN_MAX_TOKENS)
    gate_prompts = [
        build_prompt_for_condition(CONDITIONS_BY_ID["A2"], q, tokenizer, class_d_rewrites=class_d)
        for q in q_test[:10]
    ]

    def emission_rate(outs) -> float:
        rates = []
        for o in outs:
            ids = tokenizer.encode(o.outputs[0].text, add_special_tokens=False)
            rates.append(float(any(t in (MARKER_ID, bare_marker_id) for t in ids)))
        return float(np.mean(rates))

    lora_req = LoRARequest(
        lora_name="i474_loc_A2_ep1", lora_int_id=99, lora_path=ref_adapter_dirs[SOURCE_CID]
    )
    rate_on = emission_rate(llm.generate(gate_prompts, sp, lora_request=lora_req))
    rate_off = emission_rate(llm.generate(gate_prompts, sp))
    assert rate_on >= 0.8, f"(d) adapter-ON emission {rate_on:.2f} < 0.8 — LoRA not applied (#534)"
    assert rate_off <= 0.1, f"(d) adapter-OFF emission {rate_off:.2f} > 0.1 — base contaminated"
    logger.info("(d) vLLM-LoRA gate PASS (on=%.2f, off=%.2f)", rate_on, rate_off)


# ── Phase G — generation ───────────────────────────────────────────────────


def phase_gen(args) -> None:
    """vLLM greedy generation of each adapter's own R for every (persona, q)."""
    print("[phase=p1_gen]", flush=True)
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer, _bare = load_tokenizer()
    validate_registry()
    questions = load_eval_questions(args.n_questions)
    persona_prompts = load_persona_prompts()
    classify_exposure(persona_prompts)

    data_dir, _ = _resolve_dirs(args)
    raw_dir = data_dir / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    adapter_dirs = _download_i571_adapters(args.adapters)
    _gauge_assert(adapter_dirs)

    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    for p in args.personas:
        for q in questions:
            text = build_persona_prompt(persona_prompts[p], q, tokenizer)
            n_prompt = len(tokenizer.encode(text, add_special_tokens=False))
            assert n_prompt + GEN_MAX_TOKENS <= GEN_MAX_MODEL_LEN, (
                f"prompt for ({p!r}, q={q[:40]!r}...) is {n_prompt} tokens; "
                f"+{GEN_MAX_TOKENS} new exceeds max_model_len {GEN_MAX_MODEL_LEN}"
            )
            prompts.append(text)
            keys.append((p, q))

    spec = _panel_spec_with_adapters(args, questions)
    llm = _build_vllm_engine()
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=GEN_MAX_TOKENS)

    for label in args.adapters:
        out_path = raw_dir / f"raw_completions_{label}.json"
        if out_path.exists():
            _validate_existing(out_path, spec)
            logger.info("gen %s: resume skip (%s exists, spec matches)", label, out_path.name)
            continue
        t0 = time.time()
        lora_req = LoRARequest(
            lora_name=label, lora_int_id=LORA_INT_IDS[label], lora_path=adapter_dirs[label]
        )
        outs = llm.generate(prompts, sp, lora_request=lora_req)
        assert len(outs) == len(keys), (len(outs), len(keys))
        completions: dict[str, dict[str, dict]] = {p: {} for p in args.personas}
        n_trunc = 0
        for (p, q), o in zip(keys, outs, strict=True):
            gen = o.outputs[0]
            truncated = gen.finish_reason == "length"
            n_trunc += int(truncated)
            completions[p][q] = {
                "response_text": gen.text,
                "truncated": truncated,
                "n_new_tokens": len(gen.token_ids),
            }
        trunc_rate = n_trunc / len(keys)
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": "G_generation",
                    "adapter_label": label,
                    "source_cid": SOURCE_CID,
                    "adapter_hf_subpath": ADAPTER_REGISTRY[label][1],
                    "adapter_local_path": adapter_dirs[label],
                    "lora_int_id": LORA_INT_IDS[label],
                    "sampling": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": GEN_MAX_TOKENS,
                        "engine_seed": 42,
                        "max_model_len": GEN_MAX_MODEL_LEN,
                    },
                    "panel_spec": spec,
                    "truncation_rate": trunc_rate,
                    "completions": completions,
                    "metadata": result_metadata(),
                },
                indent=1,
            )
        )
        logger.info(
            "gen %s: %d completions in %.0fs (truncation rate %.3f) -> %s",
            label,
            len(keys),
            time.time() - t0,
            trunc_rate,
            out_path,
        )


# ── Phases S — four-float scoring ──────────────────────────────────────────


def _load_R_panel(data_dir: Path, label: str, spec: dict) -> dict[str, dict[str, dict]]:
    """The adapter's own raw completions written by phase gen (fail loud)."""
    path = data_dir / "raw_completions" / f"raw_completions_{label}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run --phase gen for adapter {label} (same --tag/panel) first"
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError(f"{path}: schema_version={payload.get('schema_version')!r}")
    if payload.get("panel_spec") != spec:
        raise RuntimeError(
            f"{path}: panel_spec mismatch vs the requested panel — "
            f"stored={payload.get('panel_spec')!r} requested={spec!r}"
        )
    return payload["completions"]


def _assert_slot_parity(ff_dir: Path, label: str) -> None:
    """Trained/base sides of one label must be slot-matched per (persona, q).

    Same R text + same deterministic ``_slot_job`` => same slot kind and
    same truncation; a mismatch means the two sides scored different text
    (the parity invariant the Δ readout depends on).
    """
    t_path = ff_dir / f"trained_{label}.json"
    b_path = ff_dir / f"base_{label}.json"
    if not (t_path.exists() and b_path.exists()):
        return
    trained = json.loads(t_path.read_text())["per_persona"]
    based = json.loads(b_path.read_text())["per_persona"]
    assert sorted(trained) == sorted(based), (label, sorted(trained)[:3], sorted(based)[:3])
    for p in trained:
        t_q, b_q = trained[p]["per_q"], based[p]["per_q"]
        assert len(t_q) == len(b_q), (label, p, len(t_q), len(b_q))
        for i, (tq, bq) in enumerate(zip(t_q, b_q, strict=True)):
            assert tq["slot_kind"] == bq["slot_kind"], (
                label,
                p,
                i,
                tq["slot_kind"],
                bq["slot_kind"],
            )
            assert tq["n_truncated_tokens"] == bq["n_truncated_tokens"], (label, p, i)
    logger.info("slot-kind/truncation parity PASS for %s (trained vs base)", label)


def phase_score(args, side: str) -> None:
    """Four-float corrected-slot reads on the gen-phase R; side = base|trained."""
    assert side in ("base", "trained"), side
    print(f"[phase=p2_score_{side}]", flush=True)
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer, bare_marker_id = load_tokenizer()
    validate_registry()
    questions = load_eval_questions(args.n_questions)
    persona_prompts = load_persona_prompts()
    classify_exposure(persona_prompts)

    data_dir, out_dir = _resolve_dirs(args)
    ff_dir = out_dir / "four_float"
    ff_dir.mkdir(parents=True, exist_ok=True)
    spec = _panel_spec_with_adapters(args, questions)

    adapter_dirs: dict[str, str] = {}
    if side == "trained":
        adapter_dirs = _download_i571_adapters(args.adapters)
        _gauge_assert(adapter_dirs)

    prompt_cache: dict[tuple[str, str], str] = {}

    def prompt_for(p: str, q: str) -> str:
        key = (p, q)
        if key not in prompt_cache:
            prompt_cache[key] = build_persona_prompt(persona_prompts[p], q, tokenizer)
        return prompt_cache[key]

    logger.info("loading base model %s", BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base.eval()

    phase_name = "S1_base_matched_slot" if side == "base" else "S2_trained_on_own_R"
    for label in args.adapters:
        out_path = ff_dir / f"{side}_{label}.json"
        if out_path.exists():
            _validate_existing(out_path, spec)
            logger.info("score-%s %s: resume skip (%s exists)", side, label, out_path.name)
            _assert_slot_parity(ff_dir, label)
            continue
        completions = _load_R_panel(data_dir, label, spec)

        jobs: list[dict] = []
        job_keys: list[tuple[str, str]] = []
        gen_meta: list[dict] = []
        for p in args.personas:
            for q in questions:
                rec = completions[p][q]
                job = _slot_job(prompt_for(p, q), rec["response_text"], tokenizer, bare_marker_id)
                jobs.append(job)
                job_keys.append((p, q))
                gen_meta.append(
                    {
                        "gen_truncated": bool(rec["truncated"]),
                        "n_new_tokens": int(rec["n_new_tokens"]),
                    }
                )
        assert len(jobs) == len(args.personas) * len(questions), len(jobs)

        if side == "trained":
            from peft import PeftModel

            logger.info("score-trained %s: loading adapter %s", label, adapter_dirs[label])
            model = PeftModel.from_pretrained(base, adapter_dirs[label])
            model.eval()
        else:
            model = base

        t0 = time.time()
        reads = _run_slot_batches(model, tokenizer, jobs, bare_marker_id, label=f"{side}/{label}")
        per_persona: dict[str, dict] = {p: {"per_q": []} for p in args.personas}
        for (p, _q), read, meta in zip(job_keys, reads, gen_meta, strict=True):
            read.update(meta)
            per_persona[p]["per_q"].append(read)
        for p in args.personas:
            per_persona[p]["summary"] = _summarize(per_persona[p]["per_q"])

        out_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": phase_name,
                    "side": side,
                    "adapter_label": label,
                    "source_cid": SOURCE_CID,
                    "adapter_hf_subpath": (
                        ADAPTER_REGISTRY[label][1] if side == "trained" else None
                    ),
                    "panel_spec": spec,
                    "per_persona": per_persona,
                    "metadata": result_metadata(),
                },
                indent=1,
            )
        )
        logger.info(
            "score-%s %s: %d slots in %.0fs -> %s",
            side,
            label,
            len(jobs),
            time.time() - t0,
            out_path,
        )
        if side == "trained":
            base = model.unload()
            del model
            torch.cuda.empty_cache()
        _assert_slot_parity(ff_dir, label)


# ── Phase M — source-check (manipulation check), split gen/score ──────────


def _source_paths(args, label: str) -> tuple[Path, Path]:
    """Per-label source-check file paths (always production dirs, no tag).

    Source-check files are complete per-label units over the fixed 20
    Q_test, independent of the panel restriction — so the smoke canary's
    files are the production files (resume-skipped by the full sweep).
    """
    src_dir = args.out_dir / "source_check"
    src_dir.mkdir(parents=True, exist_ok=True)
    return src_dir / f"source_gen_{label}.json", src_dir / f"source_score_{label}.json"


def _classify_manipulation(emission_on: float, emission_off: float) -> tuple[str, str]:
    """(verdict, reason) per plan §7 assert 5 + the base-contamination case."""
    if emission_on < EMISSION_ON_FAIL:
        return (
            "FAIL",
            f"source emission ON {emission_on:.2f} < {EMISSION_ON_FAIL} (implant did not take)",
        )
    if emission_off > EMISSION_OFF_MAX:
        return (
            "FAIL",
            f"adapter-OFF emission {emission_off:.2f} > {EMISSION_OFF_MAX} "
            "(base contaminated — eval-path bug)",
        )
    if emission_on >= EMISSION_ON_PASS:
        return "PASS", "source emission ON >= 0.8 and base <= 0.1"
    return "WARN", (
        f"source emission ON {emission_on:.2f} in [{EMISSION_ON_FAIL}, {EMISSION_ON_PASS}) — "
        "data stay useful; primary verdict CAPPED at indeterminate (implant-strength-confounded)"
    )


def phase_source_gen(args) -> None:
    """vLLM half of the manipulation check: A2-prompt generations ON vs OFF.

    Per adapter: 20 Q_test under the A2 condition prompt, greedy, adapter
    ON (its own LoRA) and OFF (base). On-policy emission rate = any marker
    token in the output ids. FAIL on any label -> rc=4 AFTER all requested
    labels are written (checkpoint-per-phase; the dispatcher escalates).
    """
    print("[phase=p3_source_gen]", flush=True)
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer, bare_marker_id = load_tokenizer()
    validate_registry()
    q_test = load_q_test_extended_50()[:N_SOURCE_CHECK_QUESTIONS]
    class_d = load_class_d_rewrites()
    prompts = [
        build_prompt_for_condition(
            CONDITIONS_BY_ID[SOURCE_CID], q, tokenizer, class_d_rewrites=class_d
        )
        for q in q_test
    ]

    todo = [label for label in args.adapters if not _source_paths(args, label)[0].exists()]
    if not todo:
        logger.info("source-gen: all %d labels present — resume skip", len(args.adapters))
        return
    adapter_dirs = _download_i571_adapters(todo)
    _gauge_assert(adapter_dirs)

    llm = _build_vllm_engine()
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=GEN_MAX_TOKENS)

    def emission(outs) -> tuple[list[str], list[bool]]:
        texts, fired = [], []
        for o in outs:
            text = o.outputs[0].text
            ids = tokenizer.encode(text, add_special_tokens=False)
            texts.append(text)
            fired.append(any(t in (MARKER_ID, bare_marker_id) for t in ids))
        return texts, fired

    # OFF (base) generations are label-independent: generate once, reuse.
    off_texts, off_fired = emission(llm.generate(prompts, sp))
    emission_off = float(np.mean(off_fired))

    failures: list[str] = []
    for label in todo:
        gen_path, _ = _source_paths(args, label)
        lora_req = LoRARequest(
            lora_name=label, lora_int_id=LORA_INT_IDS[label], lora_path=adapter_dirs[label]
        )
        on_texts, on_fired = emission(llm.generate(prompts, sp, lora_request=lora_req))
        emission_on = float(np.mean(on_fired))
        verdict, reason = _classify_manipulation(emission_on, emission_off)
        gen_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": "M1_source_gen",
                    "adapter_label": label,
                    "source_cid": SOURCE_CID,
                    "adapter_hf_subpath": ADAPTER_REGISTRY[label][1],
                    "n_questions": len(q_test),
                    "questions": q_test,
                    "emission_on": emission_on,
                    "emission_off": emission_off,
                    "verdict": verdict,
                    "verdict_reason": reason,
                    "on_completions": on_texts,
                    "off_completions": off_texts,
                    "on_fired": on_fired,
                    "off_fired": off_fired,
                    "metadata": result_metadata(),
                },
                indent=1,
            )
        )
        logger.info(
            "source-gen %s: emission ON=%.2f OFF=%.2f -> %s (%s)",
            label,
            emission_on,
            emission_off,
            verdict,
            gen_path.name,
        )
        if verdict == "FAIL":
            failures.append(f"{label}: {reason}")

    if failures:
        logger.error("manipulation-check FAIL: %s", failures)
        sys.exit(4)


def phase_source_score(args) -> None:
    """HF half of the manipulation check: source four-float reads + merge.

    For each label: corrected-slot four-float reads on the SAME adapter-ON
    text, trained side (PeftModel) AND base side — source Δz_marker is the
    implant-strength readout the §1 cross-arm asymmetry cap uses. Merges
    all requested labels into ``source_check.json``.
    """
    print("[phase=p3_source_score]", flush=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer, bare_marker_id = load_tokenizer()
    validate_registry()
    class_d = load_class_d_rewrites()

    adapter_dirs = _download_i571_adapters(args.adapters)
    _gauge_assert(adapter_dirs)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base.eval()

    for label in args.adapters:
        gen_path, score_path = _source_paths(args, label)
        if score_path.exists():
            logger.info("source-score %s: resume skip (%s exists)", label, score_path.name)
            continue
        if not gen_path.exists():
            raise FileNotFoundError(f"{gen_path} missing — run --phase source-gen first")
        gen_payload = json.loads(gen_path.read_text())
        q_test = gen_payload["questions"]
        jobs = [
            _slot_job(
                build_prompt_for_condition(
                    CONDITIONS_BY_ID[SOURCE_CID], q, tokenizer, class_d_rewrites=class_d
                ),
                r,
                tokenizer,
                bare_marker_id,
            )
            for q, r in zip(q_test, gen_payload["on_completions"], strict=True)
        ]
        model = PeftModel.from_pretrained(base, adapter_dirs[label])
        model.eval()
        trained_reads = _run_slot_batches(
            model, tokenizer, jobs, bare_marker_id, label=f"src_t/{label}"
        )
        base = model.unload()
        del model
        torch.cuda.empty_cache()
        base_reads = _run_slot_batches(
            base, tokenizer, jobs, bare_marker_id, label=f"src_b/{label}"
        )
        for tq, bq in zip(trained_reads, base_reads, strict=True):
            assert tq["slot_kind"] == bq["slot_kind"], (label, tq["slot_kind"], bq["slot_kind"])
            assert tq["n_truncated_tokens"] == bq["n_truncated_tokens"], label
        dz_marker = float(
            np.mean(
                [
                    t["z_marker"] - b["z_marker"]
                    for t, b in zip(trained_reads, base_reads, strict=True)
                ]
            )
        )
        score_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": "M2_source_score",
                    "adapter_label": label,
                    "source_cid": SOURCE_CID,
                    "emission_on": gen_payload["emission_on"],
                    "emission_off": gen_payload["emission_off"],
                    "verdict": gen_payload["verdict"],
                    "verdict_reason": gen_payload["verdict_reason"],
                    "dz_marker_source": dz_marker,
                    "trained": {"per_q": trained_reads, "summary": _summarize(trained_reads)},
                    "base": {"per_q": base_reads, "summary": _summarize(base_reads)},
                    "metadata": result_metadata(),
                },
                indent=1,
            )
        )
        logger.info(
            "source-score %s: dz_marker_source=%+.2f verdict=%s -> %s",
            label,
            dz_marker,
            gen_payload["verdict"],
            score_path.name,
        )

    _merge_source_check(args)


def _merge_source_check(args) -> None:
    """Merge per-label source files into ``source_check.json`` (overall gate).

    Cross-arm asymmetry |mean(broad dz_marker) − mean(narrow dz_marker)| is
    computed when both arms have >= 1 scored label; overall manipulation
    status: pass_all | capped | fail | partial (some labels missing).
    """
    per_label: dict[str, dict] = {}
    for label in ALL_LABELS:
        _, score_path = _source_paths(args, label)
        if not score_path.exists():
            continue
        payload = json.loads(score_path.read_text())
        per_label[label] = {
            "emission_on": payload["emission_on"],
            "emission_off": payload["emission_off"],
            "verdict": payload["verdict"],
            "verdict_reason": payload["verdict_reason"],
            "dz_marker_source": payload["dz_marker_source"],
            "trained_summary": payload["trained"]["summary"],
            "base_summary": payload["base"]["summary"],
        }
    broad_dz = [v["dz_marker_source"] for k, v in per_label.items() if k.startswith("broad")]
    narrow_dz = [v["dz_marker_source"] for k, v in per_label.items() if k.startswith("narrow")]
    asymmetry = (
        abs(float(np.mean(broad_dz)) - float(np.mean(narrow_dz)))
        if broad_dz and narrow_dz
        else None
    )
    verdicts = [v["verdict"] for v in per_label.values()]
    if len(per_label) < len(ALL_LABELS):
        overall = "partial"
    elif any(v == "FAIL" for v in verdicts):
        overall = "fail"
    elif all(v == "PASS" for v in verdicts) and asymmetry is not None and asymmetry <= 5.0:
        overall = "pass_all"
    else:
        overall = "capped"
    out_path = args.out_dir / "source_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "M3_source_check_merged",
                "per_label": per_label,
                "cross_arm_dz_marker_asymmetry": asymmetry,
                "asymmetry_cap_threshold": 5.0,
                "manipulation_check": overall,
                "thresholds": {
                    "emission_on_pass": EMISSION_ON_PASS,
                    "emission_on_fail": EMISSION_ON_FAIL,
                    "emission_off_max": EMISSION_OFF_MAX,
                },
                "metadata": result_metadata(),
            },
            indent=1,
        )
    )
    logger.info(
        "source_check.json written: %d/%d labels, asymmetry=%s, manipulation_check=%s",
        len(per_label),
        len(ALL_LABELS),
        f"{asymmetry:.2f}" if asymmetry is not None else "n/a",
        overall,
    )


# ── Phase U — upload + sentinel ────────────────────────────────────────────


def phase_upload(args) -> None:
    """Fail-loud HF data-repo upload of every panel artifact, then sentinel.

    Raw completions MUST land on the HF data repo before pod termination
    (CLAUDE.md Upload Policy). This dispatcher writes flat per-adapter
    ``raw_completions_{label}.json`` files (not the canonical
    ``<cell>/raw_completions.json`` shape the rglob helper picks up), so
    the upload is an explicit per-file ``hub._upload`` walk over the
    actual write paths. The sentinel is written ONLY on a full untagged
    run (a smoke-tagged upload exercises the path without signaling
    end-of-run).
    """
    print("[phase=p4_upload]", flush=True)
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    data_dir, out_dir = _resolve_dirs(args)
    bucket = f"{HF_BUCKET}/{args.tag}" if args.tag else HF_BUCKET

    uploads: list[tuple[Path, str]] = []
    raw_files = sorted((data_dir / "raw_completions").glob("raw_completions_*.json"))
    for f in raw_files:
        uploads.append((f, f"{bucket}/raw_completions/{f.name}"))
    ff_files = sorted((out_dir / "four_float").glob("*.json"))
    for f in ff_files:
        uploads.append((f, f"{bucket}/four_float/{f.name}"))

    if not args.tag:
        # Production-only artifacts: source check, train diagnostics, mixes.
        src_check = args.out_dir / "source_check.json"
        if src_check.exists():
            uploads.append((src_check, f"{bucket}/source_check.json"))
        for f in sorted((args.out_dir / "source_check").glob("source_*.json")):
            uploads.append((f, f"{bucket}/source_check/{f.name}"))
        for f in sorted(TRAIN_DIAG_DIR.glob("*.json")):
            uploads.append((f, f"{bucket}/train_diag/{f.name}"))
        for f in sorted(TRAIN_ROW_DIR.glob("i571_*.jsonl")):
            uploads.append((f, f"{bucket}/train_rows/{f.name}"))

    expected_full = sorted(args.adapters) == sorted(ALL_LABELS) and not args.tag
    if expected_full:
        n_raw, n_ff = len(raw_files), len(ff_files)
        assert n_raw == 4, f"expected 4 raw_completions files, found {n_raw}"
        assert n_ff == 8, f"expected 8 four-float files (4 base + 4 trained), found {n_ff}"
        assert (args.out_dir / "source_check.json").exists(), "source_check.json missing"
        for label in ALL_LABELS:
            _assert_slot_parity(out_dir / "four_float", label)
        n_traj = len(list(TRAIN_DIAG_DIR.glob("trajectory_i571_*.json")))
        assert n_traj == 4, f"expected 4 trajectory JSONs under {TRAIN_DIAG_DIR}, found {n_traj}"
    if not uploads:
        raise RuntimeError(f"nothing to upload under {data_dir} / {out_dir} — wrong --tag/panel?")

    for local, path_in_repo in uploads:
        url = _upload(
            local_path=local,
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            delete_after=False,
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError(f"upload failed for {local} -> {path_in_repo}")
        logger.info("uploaded %s -> %s", local.name, path_in_repo)

    if args.tag:
        logger.info("tagged (%s) upload complete — sentinel suppressed (not end-of-run)", args.tag)
        return
    note = json.dumps(
        {
            "summary": "issue571 breadth-ablation panel artifacts uploaded",
            "n_files": len(uploads),
            "hf_bucket": f"{DEFAULT_DATASET_REPO}/{bucket}",
            "adapters": list(args.adapters),
            "n_personas": len(args.personas),
            "n_questions": args.n_questions,
            "git_commit": _git_commit(),
        }
    )
    _write_sentinel(note)


def _write_sentinel(note: str) -> None:
    """End-of-run sentinel for poll_pipeline.py (pod-side only).

    Filename ``issue-571-run-complete.json`` (plan §3.2) matches the
    poller's ``/workspace/logs/issue-<N>-*.json`` drain glob; the payload
    carries poll_pipeline's ``_SENTINEL_REQUIRED_KEYS``
    (sentinel_schema_version / kind / version).
    """
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logger.info("no /workspace/logs — sentinel skipped (not a pod)")
        return
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 571,
        "by": "issue571_breadth_panel",
        "ts": datetime.now(UTC).isoformat(),
        "note": note,
    }
    path = logs_dir / "issue-571-run-complete.json"
    path.write_text(json.dumps(payload, indent=1))
    logger.info("sentinel written: %s", path)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Task #571 breadth-ablation eval driver (pod-side phases).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--phase",
        required=True,
        choices=[
            "smoke",
            "gen",
            "score-trained",
            "score-base",
            "source-gen",
            "source-score",
            "upload",
        ],
    )
    ap.add_argument("--adapters", default="all", help="comma list of adapter labels, or 'all' (4)")
    ap.add_argument("--personas", default="all", help="comma list of personas, or 'all' (35)")
    ap.add_argument("--n-questions", type=int, default=N_QUESTIONS_FULL)
    ap.add_argument(
        "--tag",
        default="",
        help="REQUIRED for restricted panels; isolates outputs under <dir>/<tag>/",
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--cpu-only",
        action="store_true",
        help="smoke phase only: run the CPU gates, SKIP the GPU gates (c)/(d)",
    )
    args = ap.parse_args(argv)
    args.adapters = list(ALL_LABELS) if args.adapters == "all" else args.adapters.split(",")
    unknown = [a for a in args.adapters if a not in ADAPTER_REGISTRY]
    assert not unknown, f"unknown adapter labels: {unknown} (known: {ALL_LABELS})"
    args.personas = list(HELD_OUT_35) if args.personas == "all" else args.personas.split(",")
    unknown_p = [p for p in args.personas if p not in HELD_OUT_35]
    assert not unknown_p, f"unknown personas: {unknown_p}"
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    if args.phase == "smoke":
        phase_smoke(args)
    elif args.phase == "gen":
        phase_gen(args)
    elif args.phase == "score-base":
        phase_score(args, "base")
    elif args.phase == "score-trained":
        phase_score(args, "trained")
    elif args.phase == "source-gen":
        phase_source_gen(args)
    elif args.phase == "source-score":
        phase_source_score(args)
    elif args.phase == "upload":
        phase_upload(args)
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
