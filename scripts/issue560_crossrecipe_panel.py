# ruff: noqa: RUF003
# Intentional Unicode (※, Δ, −, —) in scientific docstrings + labels.
"""Task #560 — cross-recipe transfer panel driver (pod-side phases).

Scores the 16 reused #474 loc-arm ep1 marker adapters (broad 15-negative
recipe, Hub-verified) on the #478/#531 held-out persona panel: 35 personas
x 20 eval questions, on-policy greedy generation per adapter, then
four-float corrected-slot reads on the trained AND base sides (#532
followup machinery, imported — not copied).

Phases (each invocation runs ONE phase; the launch sequence is in the plan
section 10 reproducibility card):

- ``smoke``      Phase-0 launch-precondition gates (a)-(e): tokenizer ids,
                 pinned #478 raw artifact, scoring-path validation vs the
                 committed #532 ``A2__A4`` cell (MAE < 0.5 nat, Spearman
                 > 0.995), vLLM-LoRA application gate vs the committed
                 ``A2__A2`` diagonal (#534 defense), adapter gauge asserts.
                 ``--cpu-only`` runs the CPU gates and SKIPs (c)/(d)/(e).
- ``gen``        vLLM greedy generation (temp 0.0, max_tokens 2048) of the
                 trained model's own R for every (adapter, persona, q);
                 checkpoint-per-adapter with resume-skip.
- ``geometry``   Layer-20 last-prompt-token centroids for the 16 source
                 contexts AND 35 personas on the shared q_test_extended_50
                 probe set; min_dist = cosine distance; tie-back diagnostic
                 vs the committed 111-panel matrix.
- ``score-base`` / ``score-trained``  HF forward-pass four-float reads at
                 the corrected slot (pre_marker | end_of_response) on the
                 SAME R, base side / adapter side; checkpoint-per-adapter.
- ``upload``     Fail-loud HF data-repo upload of raw completions +
                 four-float + geometry JSONs, then the end-of-run sentinel.

Smoke/sweep parity: the smoke IS this panel path restricted to a tiny cell
(``--sources A2 --personas assistant --n-questions 10 --tag smoke``) — same
script, same functions, every phase's cell list derives from the same
``--sources/--personas/--n-questions`` arguments (gen, geometry, scoring,
and upload all read them). A restricted panel REQUIRES ``--tag`` so partial
files can never poison the production resume-skip globs.

Scoring functions ``_slot_job`` / ``_run_slot_batches`` and the adapter
downloader are imported from ``scripts/issue532_followup_logp_slot.py``
(reuse, not copy — #532 followup @ main).
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

# Reused #532 followup scoring machinery (import, don't copy).
from issue532_followup_logp_slot import (  # noqa: E402
    EOS_ID,
    SOURCES_ALL,
    _download_adapters,
    _run_slot_batches,
    _slot_job,
    _summarize,
)

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_test_extended_50,
)

load_dotenv()

logger = logging.getLogger("issue560.crossrecipe_panel")

SCHEMA_VERSION = "issue560_crossrecipe_v1"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_BUCKET = "issue560_crossrecipe"
HF_DATA_REV_478 = "a9fc5a9cbc81c4b774ff66da0022f9055e18da5f"  # pinned #478 revision
PINNED_RAW_CELLS = ("K1_c00_seed42", "K1_c01_seed42")
LOGIT_RESCORE_K1 = PROJECT_ROOT / "eval_results/issue_478/logit_rescore/K1_c00_seed42.json"
I532_DIAGONAL_A2 = PROJECT_ROOT / "eval_results/issue_532/per_cell/loc_ep1/cell_loc_ep1_A2__A2.json"
I532_CELL_A2_A4 = PROJECT_ROOT / "eval_results/issue_532/per_cell/loc_ep1/cell_loc_ep1_A2__A4.json"
I532_FOLLOWUP_A2_A4 = (
    PROJECT_ROOT / "eval_results/issue_532/logp_slot_followup/per_cell_trained/A2__A4.json"
)
TIE_BACK_MATRIX = (
    PROJECT_ROOT / "eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/issue_560"
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results/issue_560"
GEOMETRY_LAYER = 20
N_QUESTIONS_FULL = 20
N_PROBES_FULL = 50
GEN_MAX_TOKENS = 2048
GEN_MAX_MODEL_LEN = 4096

# The clarifier's 35 held-out personas (== sorted held_out keys of the #478
# logit_rescore JSONs; asserted at startup of every phase).
HELD_OUT_35 = [
    "assistant",
    "brazilian_comedian",
    "caring_villain",
    "comedian",
    "dark_comedian",
    "devops_engineer",
    "doctor_comedian",
    "drill_sergeant",
    "elementary_teacher",
    "formal_assistant",
    "french_person",
    "grumpy_person",
    "hippie_teacher",
    "improv_comedian",
    "incompetent_villain",
    "joker",
    "lazy_software_engineer",
    "machine_learning_engineer",
    "medical_doctor",
    "medical_student",
    "misanthrope",
    "mysterious_person",
    "nice_villain",
    "open_mic_comedian",
    "overly_enthusiastic_assistant",
    "perfectionist_engineer",
    "sarcastic_assistant",
    "satirist",
    "serious_comedian",
    "stoic_philosopher",
    "strict_teacher",
    "villain",
    "web_developer",
    "wholesome_comedian",
    "zelthari_scholar",
]

# Verified during planning (byte-identity of system prompts) and re-asserted
# in code by classify_exposure() at the start of every phase.
EXPECTED_PROMPT_MATCHES = {"assistant": "A1", "comedian": "A4", "villain": "A5"}
SOURCE_RESIDENT_CELLS = {("A1", "assistant"), ("A4", "comedian"), ("A5", "villain")}


# ── Shared panel definitions ───────────────────────────────────────────────


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
        "task": 560,
        "script": "issue560_crossrecipe_panel.py",
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


def load_persona_prompts() -> dict[str, str]:
    """System prompts for the 35 held-out personas (ALL_EVAL_PERSONAS @ main)."""
    from run_100_persona_leakage import ALL_EVAL_PERSONAS

    prompts = {}
    for name in HELD_OUT_35:
        if name not in ALL_EVAL_PERSONAS:
            raise KeyError(f"persona {name!r} missing from ALL_EVAL_PERSONAS")
        prompts[name] = ALL_EVAL_PERSONAS[name]["prompt"]
    return prompts


def assert_held_out_matches_logit_rescore() -> None:
    """The clarifier list must equal the #478 logit_rescore held_out key set."""
    payload = json.loads(LOGIT_RESCORE_K1.read_text())
    keys = sorted(payload["held_out"].keys())
    assert keys == HELD_OUT_35, (
        f"held_out keys of {LOGIT_RESCORE_K1.name} diverge from the clarifier list: "
        f"only_in_rescore={sorted(set(keys) - set(HELD_OUT_35))}, "
        f"only_in_clarifier={sorted(set(HELD_OUT_35) - set(keys))}"
    )


def classify_exposure(persona_prompts: dict[str, str]) -> dict[str, str]:
    """Byte-compare each persona prompt vs the 5 A-class system prompts.

    Returns {persona: matched_cid} for the EXACT prompt matches and asserts
    the match set equals the planning-time verification (assistant≡A1,
    comedian≡A4, villain≡A5; exactly 3 matches). B/C/D conditions carry no
    system prompt, so only A-class can match by construction.
    """
    a_prompts = {cid: CONDITIONS_BY_ID[cid].system_prompt for cid in ("A1", "A2", "A3", "A4", "A5")}
    assert all(p is not None for p in a_prompts.values())
    matches: dict[str, str] = {}
    for persona, prompt in persona_prompts.items():
        for cid, sys_prompt in a_prompts.items():
            if prompt == sys_prompt:
                assert persona not in matches, f"{persona} matches two A prompts"
                matches[persona] = cid
    assert matches == EXPECTED_PROMPT_MATCHES, (
        f"exposure classification drifted from the planning-time verification: "
        f"got {matches!r}, expected {EXPECTED_PROMPT_MATCHES!r}"
    )
    return matches


def exposure_stratum(cid: str, persona: str, matches: dict[str, str]) -> str:
    """source_resident | trained_negative | never_negative for one cell."""
    if persona in matches:
        return "source_resident" if matches[persona] == cid else "trained_negative"
    return "never_negative"


def assert_strata_partition(
    sources: list[str], personas: list[str], matches: dict[str, str]
) -> dict[str, int]:
    """Strata counts; on the FULL panel assert the 3/45/512 partition of 560."""
    counts = {"source_resident": 0, "trained_negative": 0, "never_negative": 0}
    for cid in sources:
        for p in personas:
            counts[exposure_stratum(cid, p, matches)] += 1
    total = sum(counts.values())
    assert total == len(sources) * len(personas), counts
    if sorted(sources) == sorted(SOURCES_ALL) and sorted(personas) == HELD_OUT_35:
        assert counts == {
            "source_resident": 3,
            "trained_negative": 45,
            "never_negative": 512,
        }, counts
        assert total == 560, total
    return counts


def load_pinned_raw(cell: str) -> dict:
    """Download + parse one #478 raw_completions.json at the pinned revision."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_DATA_REPO,
        f"issue_478/{cell}/raw_completions/raw_completions.json",
        repo_type="dataset",
        revision=HF_DATA_REV_478,
    )
    return json.loads(Path(path).read_text())


def load_eval_questions(n_questions: int, *, check_both_cells: bool = False) -> list[str]:
    """The 20 #478 eval questions from the pinned raw artifact (sliced to n).

    ``check_both_cells`` additionally asserts the question list is identical
    across two cells and that ``spec.held_out`` (NOT top-level — fact-checker
    note, plan assumption 3) matches the clarifier list.
    """
    raw0 = load_pinned_raw(PINNED_RAW_CELLS[0])
    questions = list(raw0["eval_questions"])
    assert len(questions) == N_QUESTIONS_FULL, len(questions)
    if check_both_cells:
        raw1 = load_pinned_raw(PINNED_RAW_CELLS[1])
        assert list(raw1["eval_questions"]) == questions, (
            "eval_questions differ across pinned cells "
            f"{PINNED_RAW_CELLS[0]} vs {PINNED_RAW_CELLS[1]}"
        )
        for cell, raw in ((PINNED_RAW_CELLS[0], raw0), (PINNED_RAW_CELLS[1], raw1)):
            held = set(raw["spec"]["held_out"])
            assert held == set(HELD_OUT_35), (
                f"{cell}: spec.held_out diverges from the clarifier list "
                f"(diff={held ^ set(HELD_OUT_35)})"
            )
    assert 1 <= n_questions <= N_QUESTIONS_FULL, n_questions
    return questions[:n_questions]


def build_persona_prompt(persona_prompt: str, question: str, tokenizer) -> str:
    """The #478/#531 prompt convention: system persona + user q, gen prompt."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_tokenizer():
    """Tokenizer + the mandatory marker/EOS id asserts (gate (a))."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert marker_ids == [MARKER_ID], f"{MARKER_TEXT!r} encodes to {marker_ids}, not [{MARKER_ID}]"
    bare_ids = tokenizer.encode("※", add_special_tokens=False)
    assert len(bare_ids) == 1, f"bare ※ encodes to {bare_ids}, expected single token"
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == EOS_ID
    return tokenizer, bare_ids[0]


def _gauge_assert(adapter_dirs: dict[str, str]) -> None:
    """Logit-readout gauge condition per adapter (marker-leakage rule).

    target_modules must exclude lm_head/embed_tokens and modules_to_save must
    be empty — the z_marker/z_eos readouts are valid only when LoRA does not
    touch the unembedding. NOT an attn-only assert: the i474 adapters target
    attn+MLP projections (q/k/v/o/gate/up/down_proj), which is gauge-valid.
    """
    for cid, adir in adapter_dirs.items():
        cfg = json.loads((Path(adir) / "adapter_config.json").read_text())
        targets = set(cfg.get("target_modules") or [])
        assert not targets & {"lm_head", "embed_tokens"}, (cid, sorted(targets))
        assert not cfg.get("modules_to_save"), (cid, cfg.get("modules_to_save"))
    logger.info("gauge assert PASS for %d adapters", len(adapter_dirs))


def panel_spec(sources: list[str], personas: list[str], questions: list[str]) -> dict:
    """The panel-restriction spec stored in (and validated against) outputs."""
    return {
        "sources": list(sources),
        "personas": list(personas),
        "n_questions": len(questions),
        "questions": list(questions),
    }


def _validate_existing(path: Path, spec: dict) -> None:
    """Resume-skip guard: an existing output must carry the SAME panel spec.

    A smoke-restricted file silently satisfying a full-panel resume-skip is
    the partial-file poisoning failure mode; fail loud instead.
    """
    payload = json.loads(path.read_text())
    stored = payload.get("panel_spec")
    if stored != spec:
        raise RuntimeError(
            f"{path} exists but its panel_spec differs from the requested panel — "
            f"refusing the resume-skip. Delete the file or rerun with the matching "
            f"--sources/--personas/--n-questions (or a distinct --tag). "
            f"stored={stored!r} requested={spec!r}"
        )


def _resolve_dirs(args) -> tuple[Path, Path]:
    """(data_dir, out_dir), tag-suffixed when a restricted panel is requested."""
    restricted = (
        sorted(args.sources) != sorted(SOURCES_ALL)
        or sorted(args.personas) != HELD_OUT_35
        or args.n_questions != N_QUESTIONS_FULL
    )
    if restricted and not args.tag:
        raise SystemExit(
            "A restricted panel (--sources/--personas/--n-questions below the full "
            "16x35x20) REQUIRES --tag (e.g. --tag smoke) so partial files never "
            "collide with production outputs."
        )
    data_dir = args.data_dir / args.tag if args.tag else args.data_dir
    out_dir = args.out_dir / args.tag if args.tag else args.out_dir
    return data_dir, out_dir


# ── Phase 0 — smoke gates ──────────────────────────────────────────────────


def phase_smoke(args) -> None:
    """Launch-precondition gates (a)-(e). Order: CPU gates, HF gate (c),
    free the HF model, vLLM gate (d) last (vLLM teardown is unreliable;
    the process exits right after)."""
    print("[phase=p0_smoke]", flush=True)

    # (a) tokenizer ids
    tokenizer, bare_marker_id = load_tokenizer()
    logger.info("(a) tokenizer asserts PASS (marker=%d, eos=%d)", MARKER_ID, EOS_ID)

    # (b) pinned #478 raw artifact, both cells + spec.held_out
    questions = load_eval_questions(N_QUESTIONS_FULL, check_both_cells=True)
    logger.info("(b) pinned raw artifact PASS (%d questions, 2 cells)", len(questions))

    # (b2) panel identity + exposure classification + strata partition
    assert_held_out_matches_logit_rescore()
    persona_prompts = load_persona_prompts()
    matches = classify_exposure(persona_prompts)
    counts = assert_strata_partition(list(SOURCES_ALL), HELD_OUT_35, matches)
    logger.info("(b2) exposure classification PASS: %s", counts)

    # CPU slot-job construction check on the committed #532 fixture R.
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
        logger.warning("(c)/(d)/(e) SKIPPED — --cpu-only (GPU gates run pod-side)")
        return

    # (e) gauge asserts for ALL 16 adapter configs (downloads adapters once;
    # the same cache serves gen/score-trained).
    adapter_dirs = _download_adapters(list(SOURCES_ALL))
    _gauge_assert(adapter_dirs)

    # (c) scoring-path validation vs the committed #532 followup A2__A4 cell.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

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
    logger.info("(c) loading base model + adapter A2 for the scoring-path gate ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base.eval()
    peft_model = PeftModel.from_pretrained(base, adapter_dirs["A2"])
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

    # (d) vLLM-LoRA application gate (#534 defense), against the committed
    # diagonal in_R_emission_rate = 1.0.
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

    lora_req = LoRARequest(lora_name="i474_loc_A2_ep1", lora_int_id=1, lora_path=adapter_dirs["A2"])
    rate_on = emission_rate(llm.generate(gate_prompts, sp, lora_request=lora_req))
    rate_off = emission_rate(llm.generate(gate_prompts, sp))
    assert rate_on >= 0.8, f"(d) adapter-ON emission {rate_on:.2f} < 0.8 — LoRA not applied (#534)"
    assert rate_off <= 0.1, f"(d) adapter-OFF emission {rate_off:.2f} > 0.1 — base contaminated"
    logger.info("(d) vLLM-LoRA gate PASS (on=%.2f, off=%.2f)", rate_on, rate_off)


# ── Phase G — generation ───────────────────────────────────────────────────


def _build_vllm_engine():
    """One vLLM engine per process (#532 _build_vllm_engine recipe)."""
    from vllm import LLM

    return LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=GEN_MAX_MODEL_LEN,
    )


def phase_gen(args) -> None:
    """vLLM greedy generation of each adapter's own R for every (persona, q)."""
    print("[phase=p1_gen]", flush=True)
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer, _bare = load_tokenizer()
    questions = load_eval_questions(args.n_questions)
    persona_prompts = load_persona_prompts()
    matches = classify_exposure(persona_prompts)
    assert_strata_partition(args.sources, args.personas, matches)

    data_dir, _ = _resolve_dirs(args)
    raw_dir = data_dir / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    adapter_dirs = _download_adapters(args.sources)
    _gauge_assert(adapter_dirs)

    prompts = []
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

    spec = panel_spec(args.sources, args.personas, questions)
    llm = _build_vllm_engine()
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=GEN_MAX_TOKENS)

    for i, cid in enumerate(args.sources, 1):
        out_path = raw_dir / f"raw_completions_{cid}.json"
        if out_path.exists():
            _validate_existing(out_path, spec)
            logger.info("gen %s: resume skip (%s exists, spec matches)", cid, out_path.name)
            continue
        t0 = time.time()
        lora_req = LoRARequest(
            lora_name=f"i474_loc_{cid}_ep1", lora_int_id=i, lora_path=adapter_dirs[cid]
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
                    "source_cid": cid,
                    "adapter_hf_subpath": f"adapters/i474_loc_{cid}_ep1",
                    "adapter_local_path": adapter_dirs[cid],
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
            cid,
            len(keys),
            time.time() - t0,
            trunc_rate,
            out_path,
        )


# ── Phase C — geometry ─────────────────────────────────────────────────────


def _extract_centroids(
    model, tokenizer, prompt_lists: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """Layer-20 last-prompt-token centroid per entity (the #478/#405 recipe).

    One unbatched forward per prompt (padding-free, so the last position IS
    the last real token), hidden state captured by a forward hook on
    ``model.model.layers[GEOMETRY_LAYER]``, mean over probes in float32.
    """
    import torch

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()

    handle = model.model.layers[GEOMETRY_LAYER].register_forward_hook(hook_fn)
    centroids: dict[str, np.ndarray] = {}
    try:
        for name, prompt_texts in prompt_lists.items():
            vecs = []
            for text in prompt_texts:
                inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
                with torch.no_grad():
                    model(**inputs)
                last_pos = inputs["input_ids"].shape[1] - 1
                vecs.append(captured["hs"][0, last_pos, :].float().cpu())
            stacked = torch.stack(vecs)
            assert stacked.shape == (len(prompt_texts), stacked.shape[1]), stacked.shape
            centroids[name] = stacked.mean(dim=0).numpy().astype(np.float64)
            logger.info("centroid %s done (%d probes)", name, len(prompt_texts))
    finally:
        handle.remove()
    return centroids


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def phase_geometry(args) -> None:
    """Centroids for the 16 source contexts + 35 personas on the shared
    q_test_extended_50 probe set; min_dist; tie-back diagnostic."""
    print("[phase=p2_geometry]", flush=True)
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer, _bare = load_tokenizer()
    persona_prompts = load_persona_prompts()
    classify_exposure(persona_prompts)
    probes = load_q_test_extended_50()[: args.probes]
    assert len(probes) == args.probes, (len(probes), args.probes)
    class_d = load_class_d_rewrites()

    _, out_dir = _resolve_dirs(args)
    geo_dir = out_dir / "geometry"
    geo_dir.mkdir(parents=True, exist_ok=True)
    out_path = geo_dir / "context_persona_geometry.json"

    prompt_lists: dict[str, list[str]] = {}
    for cid in args.sources:
        prompt_lists[f"context::{cid}"] = [
            build_prompt_for_condition(
                CONDITIONS_BY_ID[cid], q, tokenizer, class_d_rewrites=class_d
            )
            for q in probes
        ]
    for p in args.personas:
        prompt_lists[f"persona::{p}"] = [
            build_persona_prompt(persona_prompts[p], q, tokenizer) for q in probes
        ]

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    centroids = _extract_centroids(model, tokenizer, prompt_lists)

    min_dist = {
        cid: {
            p: _cosine_distance(centroids[f"context::{cid}"], centroids[f"persona::{p}"])
            for p in args.personas
        }
        for cid in args.sources
    }
    persona_persona = {
        a: {
            b: _cosine_distance(centroids[f"persona::{a}"], centroids[f"persona::{b}"])
            for b in args.personas
        }
        for a in args.personas
    }
    context_context = {
        a: {
            b: _cosine_distance(centroids[f"context::{a}"], centroids[f"context::{b}"])
            for b in args.sources
        }
        for a in args.sources
    }

    # Tie-back diagnostic vs the committed 111-panel matrix (persona pairs only;
    # reported, not a gate — < 0.5 is flagged in the write-up).
    from scipy.stats import spearmanr

    ref = json.loads(TIE_BACK_MATRIX.read_text())
    ref_names = ref["persona_names"]
    ref_idx = {n: i for i, n in enumerate(ref_names)}
    ref_matrix = ref["matrix"]
    matched = [p for p in args.personas if p in ref_idx]
    fresh_vals, ref_vals = [], []
    for i, a in enumerate(matched):
        for b in matched[i + 1 :]:
            fresh_vals.append(persona_persona[a][b])
            ref_vals.append(ref_matrix[ref_idx[a]][ref_idx[b]])
    tie_back_rho = (
        float(spearmanr(fresh_vals, ref_vals)[0]) if len(fresh_vals) >= 3 else float("nan")
    )
    logger.info(
        "tie-back: rho=%.3f over %d pairs (%d/%d personas matched in the 111 panel)",
        tie_back_rho,
        len(fresh_vals),
        len(matched),
        len(args.personas),
    )

    out_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "C_geometry",
                "layer": GEOMETRY_LAYER,
                "probe_set": "q_test_extended_50",
                "n_probes": args.probes,
                "metric": "cosine_distance",
                "sources": list(args.sources),
                "personas": list(args.personas),
                "min_dist": min_dist,
                "persona_persona_distance": persona_persona,
                "context_context_distance": context_context,
                "tie_back": {
                    "spearman": tie_back_rho,
                    "n_pairs": len(fresh_vals),
                    "n_personas_matched": len(matched),
                    "unmatched_personas": [p for p in args.personas if p not in ref_idx],
                    "reference": str(TIE_BACK_MATRIX.relative_to(PROJECT_ROOT)),
                    "reference_metric": ref.get("metric"),
                    "note": "diagnostic only; < 0.5 flagged as geometry-recipe instability",
                },
                "metadata": result_metadata({"n_centroids": len(centroids)}),
            },
            indent=1,
        )
    )
    logger.info("geometry written: %s", out_path)


# ── Phases S1/S2 — four-float scoring ──────────────────────────────────────


def _load_R(data_dir: Path, cid: str, spec: dict) -> dict[str, dict[str, dict]]:
    """The adapter's own raw completions written by phase gen (fail loud)."""
    path = data_dir / "raw_completions" / f"raw_completions_{cid}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run --phase gen for source {cid} (same --tag/panel) first"
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


def phase_score(args, side: str) -> None:
    """Four-float corrected-slot reads on the gen-phase R; side = base|trained."""
    assert side in ("base", "trained"), side
    print(f"[phase=p3_score_{side}]", flush=True)
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer, bare_marker_id = load_tokenizer()
    questions = load_eval_questions(args.n_questions)
    persona_prompts = load_persona_prompts()
    matches = classify_exposure(persona_prompts)
    assert_strata_partition(args.sources, args.personas, matches)

    data_dir, out_dir = _resolve_dirs(args)
    ff_dir = out_dir / "four_float"
    ff_dir.mkdir(parents=True, exist_ok=True)
    spec = panel_spec(args.sources, args.personas, questions)

    adapter_dirs: dict[str, str] = {}
    if side == "trained":
        adapter_dirs = _download_adapters(args.sources)
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
    for cid in args.sources:
        out_path = ff_dir / f"{side}_{cid}.json"
        if out_path.exists():
            _validate_existing(out_path, spec)
            logger.info("score-%s %s: resume skip (%s exists)", side, cid, out_path.name)
            continue
        completions = _load_R(data_dir, cid, spec)

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

            logger.info("score-trained %s: loading adapter %s", cid, adapter_dirs[cid])
            model = PeftModel.from_pretrained(base, adapter_dirs[cid])
            model.eval()
        else:
            model = base

        t0 = time.time()
        reads = _run_slot_batches(model, tokenizer, jobs, bare_marker_id, label=f"{side}/{cid}")
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
                    "source_cid": cid,
                    "adapter_hf_subpath": (
                        f"adapters/i474_loc_{cid}_ep1" if side == "trained" else None
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
            cid,
            len(jobs),
            time.time() - t0,
            out_path,
        )
        if side == "trained":
            base = model.unload()
            del model
            torch.cuda.empty_cache()


# ── Phase U — upload + sentinel ────────────────────────────────────────────


def phase_upload(args) -> None:
    """Fail-loud HF data-repo upload of every panel artifact, then sentinel."""
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
    geo = out_dir / "geometry" / "context_persona_geometry.json"
    if geo.exists():
        uploads.append((geo, f"{bucket}/geometry/{geo.name}"))

    expected_full = sorted(args.sources) == sorted(SOURCES_ALL) and not args.tag
    if expected_full:
        n_raw, n_ff = len(raw_files), len(ff_files)
        assert n_raw == 16, f"expected 16 raw_completions files, found {n_raw}"
        assert n_ff == 32, f"expected 32 four-float files (16 base + 16 trained), found {n_ff}"
        assert geo.exists(), f"geometry JSON missing: {geo}"
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

    note = json.dumps(
        {
            "summary": "issue560 cross-recipe panel artifacts uploaded",
            "n_files": len(uploads),
            "hf_bucket": f"{DEFAULT_DATASET_REPO}/{bucket}",
            "sources": list(args.sources),
            "n_personas": len(args.personas),
            "n_questions": args.n_questions,
            "git_commit": _git_commit(),
        }
    )
    _write_sentinel(note)


def _write_sentinel(note: str) -> None:
    """End-of-run sentinel for poll_pipeline.py (pod-side only)."""
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logger.info("no /workspace/logs — sentinel skipped (not a pod)")
        return
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 560,
        "by": "issue560_crossrecipe_panel",
        "ts": datetime.now(UTC).isoformat(),
        "note": note,
    }
    path = logs_dir / f"issue-560-epm_results-{int(time.time())}.json"
    path.write_text(json.dumps(payload, indent=1))
    logger.info("sentinel written: %s", path)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Task #560 cross-recipe transfer panel driver (pod-side phases).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--phase",
        required=True,
        choices=["smoke", "gen", "geometry", "score-base", "score-trained", "upload"],
    )
    ap.add_argument("--sources", default="all", help="comma list of cids, or 'all' (16)")
    ap.add_argument("--personas", default="all", help="comma list of personas, or 'all' (35)")
    ap.add_argument("--n-questions", type=int, default=N_QUESTIONS_FULL)
    ap.add_argument("--probes", type=int, default=N_PROBES_FULL, help="geometry centroid probes")
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
        help="smoke phase only: run the CPU gates, SKIP the GPU gates (c)/(d)/(e)",
    )
    args = ap.parse_args(argv)
    args.sources = list(SOURCES_ALL) if args.sources == "all" else args.sources.split(",")
    unknown = [s for s in args.sources if s not in SOURCES_ALL]
    assert not unknown, f"unknown sources: {unknown}"
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
    elif args.phase == "geometry":
        phase_geometry(args)
    elif args.phase == "score-base":
        phase_score(args, "base")
    elif args.phase == "score-trained":
        phase_score(args, "trained")
    elif args.phase == "upload":
        phase_upload(args)
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
