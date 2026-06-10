#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ρ, −, ×, —) in scientific docstrings + logs.
"""Issue #559 — base own-response prior on the 35-persona held-out panel (pod).

One new 700-row measurement: the base model (Qwen/Qwen2.5-7B-Instruct, NO
adapters anywhere) generates its OWN greedy response under each of the 35
held-out persona prompts × 20 committed eval questions, then an HF bf16
forward pass reads the four-float marker contract (z_marker, z_eos, logZ,
logp_marker + argmax_id) at the end-of-own-response slot. Mirrors:

* generation — ``issue-478:scripts/issue478_run_cell.py::vllm_greedy_generate``
  (greedy temp 0.0 / top_p 1.0 / n=1, vLLM seed 42, bf16, max_model_len 1024,
  max_num_seqs 64, per-question cap ``1024 − prompt_len − 8`` floor 64);
* slot scoring — ``scripts/issue531_logit_rescore.py::score_slot`` (full-string
  tokenize ``add_special_tokens=False``, left-pad, ``logits[:, -1, :]``), the
  persona panel's own convention, validated per run by the S0 gate;
* pre-marker truncation guard — ``scripts/issue532_followup_logp_slot.py::
  _slot_job`` (ids 83399 / 63680), applied at the token-id level so the
  no-marker case reduces EXACTLY to the panel convention (expected for all
  700 slots: no persona prompt mentions ※, base marker argmax rate 0.0 on
  56,000 matched slots).

Phases (ONE code path; smoke = production with the limit flags):

  preflight  CPU-only: tokenizer asserts, panel restriction from the committed
             parquet, question-identity hard gate vs the pinned #478 HF
             revision, prompt construction + per-question caps (fail-loud
             floor 64), truncation-guard self-test. Writes preflight.json.
  gen        vLLM greedy generation → R_base_own.json (checkpoint-per-phase),
             then vLLM teardown (child reap + nvidia-smi clean assert).
  score      S0 scoring-path reproduction gate (re-score K1_c00_seed42's
             stored base side with THIS script's scoring path; MAE < 1.0 nat,
             Spearman ≥ 0.995 vs the parquet's stored base floats, exit 1 on
             miss) → s0_validation.json; then the 700 new four-float slot
             reads → base_prior_own_persona_panel.json.
  upload     all three JSONs → HF data repo (fail-loud), raw generations under
             raw_completions/ per the upload policy.
  all        preflight → gen → score → upload → sentinel → [phase=done].
             EXCEPTION: with --reproduce-gate-json (the entry-gate invocation)
             NO sentinel is written and the terminal line is
             [phase=gate_passed] — only the chained disjoint invocation may
             signal run-complete to poll_pipeline.py.

Pod launch (production):

    nohup uv run python scripts/issue559_base_prior_persona_panel.py --phase all \\
      > /workspace/logs/issue-559-base-prior.log 2>&1 &

Smoke (same code path, tiny slice):

    uv run python scripts/issue559_base_prior_persona_panel.py --phase all \\
      --limit-personas 2 --limit-questions 2 --out-dir /tmp/issue559_smoke

Disjoint-question follow-up (plan v4, ``disjoint-question-prior``): two chained
invocations on the fresh pod — (1) the unmodified 20-question path with
``--reproduce-gate-json`` (per-persona MAE < 0.1 nat AND Spearman ≥ 0.999 vs
the committed parent prior, exit 1 on a production miss; writes NO results
sentinel and ends with ``[phase=gate_passed]``), then (2) the disjoint run with
``--questions-json eval_results/issue_559/disjoint_question_prior/
questions_disjoint30.json`` + ``--upload-prefix
issue559_base_prior_persona_panel/disjoint_question_prior``. Every flag
defaults to the current behavior, so the 20-question path is unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# ── Constants (pinned to #478 / #531 / #532 provenance) ──────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # per #478 result.json training.base_model
MARKER_TEXT = " ※"
MARKER_ID = 83399
BARE_MARKER_ID = 63680  # bare ※ (no leading space) — truncation-guard scan only
EOS_TOKEN = "<|im_end|>"
EOS_ID = 151645

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_REV = "a9fc5a9cbc81c4b774ff66da0022f9055e18da5f"  # pinned #478 revision
HF_UPLOAD_PREFIX = "issue559_base_prior_persona_panel"
S0_CELL = "K1_c00_seed42"  # the run whose stored base side the S0 gate re-scores

TIDY_LOGIT_PARQUET = (
    PROJECT_ROOT / "eval_results" / "issue_478" / "base_prior_reanalysis" / "tidy_logit.parquet"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_559"

# Generation regime — byte-for-byte the panel's own (issue-478 _issue478_common).
MAX_LENGTH = 1024
R_CAP_SAFETY_MARGIN = 8  # max_tokens = 1024 − prompt_len − 8 (#478 Fix D)
R_CAP_MIN = 64  # cap < 64 → prompt too long → FAIL LOUD

# S0 gate thresholds — mirror issue531_logit_rescore.py (wrong construction
# produces tens-of-nats mismatches; dtype noise stays well under 1 nat).
MAX_VALIDATION_MAE_NATS = 1.0
MIN_VALIDATION_SPEARMAN = 0.995

# 20-question reproduction entry-gate thresholds (plan v4 §3): code-path
# validity on a fresh pod, NOT bit-determinism — 0.1 nat is ~90× under the
# committed 9.0-nat persona spread and at the S0-observed dtype-noise scale
# (0.0707 nat MAE on the parent run).
REPRO_GATE_MAE_NATS = 0.1
REPRO_GATE_SPEARMAN = 0.999

# #460 Q_test artifact — source of the 30 disjoint measurement questions
# (q50[20:]; q50[:20] are the panel's EVAL_QUESTIONS, verified at plan time).
R_TEST_HF_PATH = "issue460_marker_at_end/on_policy_R/R_test.json"

SCHEMA_VERSION = "issue559_base_prior_v1"
SENTINEL_DIR_POD = Path("/workspace/logs")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("issue559_base_prior")


# ── vLLM teardown (#478 pattern, attribution: issue478_run_cell.py) ──────────


def kill_vllm_children() -> None:
    """Reap vLLM worker subprocesses (per .claude/rules/gotchas.md)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        log.warning("psutil not available — skipping orphan-PID reaping")
        return
    me = psutil.Process()
    children = me.children(recursive=True)
    for ch in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            log.info("Terminating vLLM child PID=%d name=%r", ch.pid, ch.name())
            ch.terminate()
    _gone, alive = psutil.wait_procs(children, timeout=5)
    for ch in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            ch.kill()


def nvidia_smi_assert_clean(gpu_id: int = 0, n_retries: int = 1, grace_s: float = 5.0) -> None:
    """FAIL-LOUD if any compute PID still holds the GPU after vLLM teardown.

    Single-GPU ``--id`` query (per agent memory: orphan-PID checks on
    multi-GPU pods must not flag sibling processes on other GPUs). Re-probes
    once after ``grace_s`` so a teardown race (a worker exiting between
    ``kill_vllm_children``'s wait and this probe) cannot false-positive.

    A PID that survives BOTH probes raises ``RuntimeError`` BEFORE the HF
    scoring phase reloads the 7B model: on this job's dedicated single GPU
    there is NO tolerated foreign PID — a survivor is either our orphaned
    vLLM worker or a foreign process, and either re-allocates the freed
    memory and OOMs the bf16 reload (gotchas rule; #399 round-11 incident).
    Only a failed nvidia-smi PROBE stays warn-only (no GPU introspection
    available — e.g. the CPU smoke on the VM, where there is no GPU to leak).
    """
    my_pid = os.getpid()
    leaks: list[str] = []
    for attempt in range(n_retries + 1):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu_id}",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=10,
                env={**os.environ},  # explicit env-passthrough
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.warning("nvidia-smi probe failed (%s); skipping orphan check", e)
            return
        leaks = [
            line.strip()
            for line in out.splitlines()
            if line.strip()
            and line.split(",")[0].strip().isdigit()
            and int(line.split(",")[0].strip()) != my_pid
        ]
        if not leaks:
            log.info("[teardown] GPU %d clean after vLLM teardown", gpu_id)
            return
        if attempt < n_retries:
            log.warning(
                "GPU %d still busy after teardown (%r) — re-probing in %.0fs (teardown race grace)",
                gpu_id,
                leaks,
                grace_s,
            )
            time.sleep(grace_s)
    raise RuntimeError(
        f"vLLM teardown leak: compute PIDs survive on GPU {gpu_id} after "
        f"{n_retries + 1} probes ({grace_s:.0f}s grace): {leaks!r} — refusing to start "
        "the HF scoring phase (the survivor would re-allocate freed memory and OOM "
        "the bf16 reload)"
    )


# ── Panel inputs ──────────────────────────────────────────────────────────────


def load_panel() -> tuple[list[str], dict[str, str], list[str]]:
    """(personas, persona_prompts, eval_questions) for the 35-persona panel.

    Personas come from the committed parquet's ``held_out_persona`` values
    (sorted, deterministic), asserted ⊆ ``ALL_EVAL_PERSONAS`` (main). The 20
    questions are ``EVAL_QUESTIONS`` (main), gated against the pinned #478
    revision by ``question_identity_gate``.
    """
    import pandas as pd
    from run_100_persona_leakage import ALL_EVAL_PERSONAS, EVAL_QUESTIONS

    df = pd.read_parquet(TIDY_LOGIT_PARQUET, columns=["held_out_persona"])
    personas = sorted(df["held_out_persona"].unique().tolist())
    assert len(personas) == 35, f"expected 35 held-out personas, got {len(personas)}"
    missing = [p for p in personas if p not in ALL_EVAL_PERSONAS]
    assert not missing, f"parquet personas missing from ALL_EVAL_PERSONAS: {missing}"
    persona_prompts = {p: ALL_EVAL_PERSONAS[p]["prompt"] for p in personas}
    for p, prompt in persona_prompts.items():
        assert "※" not in prompt, f"persona prompt for {p!r} mentions the marker"
    questions = list(EVAL_QUESTIONS)
    assert len(questions) == 20, len(questions)
    for q in questions:
        assert "※" not in q, f"eval question mentions the marker: {q!r}"
    return personas, persona_prompts, questions


def load_raw_completions(cell: str) -> dict:
    """Download + parse the cell's raw_completions.json at the pinned revision."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_DATA_REPO,
        f"issue_478/{cell}/raw_completions/raw_completions.json",
        repo_type="dataset",
        revision=HF_DATA_REV,
    )
    with open(path) as f:
        return json.load(f)


def question_identity_gate(raw: dict, eval_questions: list[str], personas: list[str]) -> None:
    """Hard gate: main's questions/personas == the pinned #478 panel (plan §4.2).

    Order-sensitive question equality + held-out set equality; exits 1 on any
    drift so no new measurement can silently misalign with ``question_idx``.
    """
    if raw["eval_questions"] != eval_questions:
        log.error("QUESTION-IDENTITY GATE FAILED: EVAL_QUESTIONS (main) != pinned #478 questions")
        sys.exit(1)
    if sorted(raw["spec"]["held_out"]) != sorted(personas):
        log.error("QUESTION-IDENTITY GATE FAILED: parquet personas != pinned #478 held_out set")
        sys.exit(1)
    log.info("[gate] question-identity gate PASS (20 questions order-identical, 35 personas)")


def load_questions_json(path: Path) -> tuple[list[str], dict]:
    """Load the committed measurement-question file; verify its embedded sha256.

    Returns ``(questions, questions_source)`` where ``questions_source`` is the
    provenance block (path, HF pin, derivation rule, n, sha256) recorded into
    every output payload + ``result_metadata`` (plan v4 §5.1).
    """
    payload = json.loads(path.read_text())
    questions = list(payload["questions"])
    digest = hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest()
    if digest != payload["sha256_questions"]:
        log.error(
            "QUESTIONS-FILE GATE FAILED: sha256 mismatch for %s (embedded %s, recomputed %s)",
            path,
            payload["sha256_questions"],
            digest,
        )
        sys.exit(1)
    deriv = payload["derivation"]
    source = {
        "path": str(path),
        "hf_repo": deriv["hf_repo"],
        "hf_revision": deriv["hf_revision"],
        "rule": deriv["rule"],
        "n": len(questions),
        "sha256": digest,
    }
    return questions, source


def disjoint_question_gate(provided: list[str], eval_questions: list[str]) -> None:
    """Hard gate (plan v4 §3): provided == q50[20:] of the pinned #460 R_test.json.

    Downloads ``R_test.json`` at the existing ``HF_DATA_REV`` pin and asserts,
    in order: schema ``i460_v1``; all 16 contexts share ONE identical ordered
    50-question list; ``q50[:20] == EVAL_QUESTIONS`` (order-sensitive);
    ``provided == q50[20:]`` (order-sensitive, n=30); disjointness from
    ``EVAL_QUESTIONS``; no ※ in any provided question. Runs on the FULL
    provided list BEFORE any ``--limit-questions`` smoke slicing. Exits 1 on
    any miss (all conditions asserted True at plan time; the gate keeps them
    true at run time).
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_DATA_REPO, R_TEST_HF_PATH, repo_type="dataset", revision=HF_DATA_REV)
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != "i460_v1":
        log.error(
            "DISJOINT-QUESTION GATE FAILED: R_test.json schema_version=%r, expected 'i460_v1'",
            payload.get("schema_version"),
        )
        sys.exit(1)
    completions = payload["completions"]
    q_lists = [list(qmap.keys()) for qmap in completions.values()]
    q50 = q_lists[0]
    if len(completions) != 16 or any(ql != q50 for ql in q_lists):
        log.error(
            "DISJOINT-QUESTION GATE FAILED: expected 16 contexts sharing one identical "
            "ordered 50-question list (got %d contexts, identical=%s)",
            len(completions),
            all(ql == q50 for ql in q_lists),
        )
        sys.exit(1)
    if q50[:20] != eval_questions:
        log.error(
            "DISJOINT-QUESTION GATE FAILED: q50[:20] != EVAL_QUESTIONS (order-sensitive) — "
            "the #460 pool no longer anchors the panel's 20"
        )
        sys.exit(1)
    if provided != q50[20:]:
        log.error(
            "DISJOINT-QUESTION GATE FAILED: provided question list != q50[20:] "
            "(order-sensitive; provided n=%d, expected n=%d)",
            len(provided),
            len(q50[20:]),
        )
        sys.exit(1)
    if len(provided) != 30:
        log.error("DISJOINT-QUESTION GATE FAILED: expected n=30, got %d", len(provided))
        sys.exit(1)
    if set(provided) & set(eval_questions):
        log.error(
            "DISJOINT-QUESTION GATE FAILED: provided questions overlap EVAL_QUESTIONS: %r",
            sorted(set(provided) & set(eval_questions)),
        )
        sys.exit(1)
    if any("※" in q for q in provided):
        log.error("DISJOINT-QUESTION GATE FAILED: a provided question mentions the marker")
        sys.exit(1)
    log.info(
        "[gate] disjoint-question gate PASS (30 questions == q50[20:] @ %s, disjoint, ※-free)",
        HF_DATA_REV[:8],
    )


# ── Prompt construction (#478 convention) ─────────────────────────────────────


def chat_prompt(tokenizer, persona_prompt: str, question: str) -> str:
    """Chat-template prompt with ``add_generation_prompt=True`` (#478 convention)."""
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def per_question_R_cap(tokenizer, prompt_text: str, label: str) -> int:
    """``1024 − prompt_len − 8`` with fail-loud floor 64 (#478 Fix D)."""
    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    cap = MAX_LENGTH - prompt_len - R_CAP_SAFETY_MARGIN
    if cap < R_CAP_MIN:
        raise RuntimeError(f"R-cap too small for {label}: prompt_len={prompt_len}, cap={cap}")
    return cap


# ── Slot construction + scoring ───────────────────────────────────────────────


def build_slot_ids(tokenizer, prompt_text: str, R_text: str) -> dict:
    """Token ids for one slot read, with the #532 pre-marker truncation guard.

    Full-string tokenize of ``prompt + R`` (``add_special_tokens=False``) —
    the panel convention (``issue531_logit_rescore.py::score_slot``). The ids
    are then scanned for a marker token (` ※` id 83399 or bare ``※`` id
    63680, the ``issue532_followup_logp_slot.py::_slot_job`` rule applied at
    the token-id level): if found, truncate just BEFORE the first occurrence
    (``slot_kind="pre_marker"``); otherwise the ids are untouched, so the
    no-marker case reduces EXACTLY to the panel convention. The marker scan is
    safe over the whole sequence because ``load_panel`` asserts no persona
    prompt or question contains ※ — any hit is inside R.
    """
    full_ids = tokenizer.encode(prompt_text + R_text, add_special_tokens=False)
    marker_pos = [i for i, t in enumerate(full_ids) if t in (MARKER_ID, BARE_MARKER_ID)]
    if marker_pos:
        i = marker_pos[0]
        return {
            "ids": full_ids[:i],
            "slot_kind": "pre_marker",
            "n_truncated_tokens": len(full_ids) - i,
        }
    return {"ids": full_ids, "slot_kind": "end_of_response", "n_truncated_tokens": 0}


def score_slots(
    model,
    tokenizer,
    items: list[tuple[tuple, list[int]]],
    device: str,
    batch_size: int,
) -> dict[tuple, dict[str, float]]:
    """Four-float reads at the last real token of each id sequence.

    Left-pads within batch and reads ``logits[:, -1, :]`` exactly like
    ``issue531_logit_rescore.py::score_slot`` (the panel convention); batches
    length-sorted to cut padding waste (numerically irrelevant to the per-row
    last-slot readout). Returns {key: {z_marker, z_eos, logZ, logp, argmax_id}}.
    """
    import torch
    import torch.nn.functional as F

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
    out: dict[tuple, dict[str, float]] = {}
    t0 = time.time()
    for start in range(0, len(order), batch_size):
        chunk = [items[i] for i in order[start : start + batch_size]]
        max_len = max(len(ids) for _, ids in chunk)
        assert min(len(ids) for _, ids in chunk) > 0, "empty id sequence — prompt build broke"
        padded = [[pad_id] * (max_len - len(ids)) + ids for _, ids in chunk]
        attn = [[0] * (max_len - len(ids)) + [1] * len(ids) for _, ids in chunk]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        last = logits[:, -1, :].float()  # (B, V) — slot after R (left-padded)
        assert last.shape[0] == len(chunk), last.shape
        logz = torch.logsumexp(last, dim=-1)
        z_marker = last[:, MARKER_ID]
        z_eos = last[:, EOS_ID]
        argmax_ids = last.argmax(dim=-1)
        logp = F.log_softmax(last, dim=-1)[:, MARKER_ID]

        for (key, _), z, ze, lz, lp, am in zip(
            chunk,
            z_marker.cpu().tolist(),
            z_eos.cpu().tolist(),
            logz.cpu().tolist(),
            logp.cpu().tolist(),
            argmax_ids.cpu().tolist(),
            strict=True,
        ):
            out[key] = {
                "z_marker": float(z),
                "z_eos": float(ze),
                "logZ": float(lz),
                "logp": float(lp),
                "argmax_id": int(am),
            }
        del logits, last, logz, z_marker, z_eos, argmax_ids, logp
        if (start // batch_size) % 10 == 0:
            log.info("score_slots: %d/%d rows (%.0fs)", len(out), len(items), time.time() - t0)
    return out


# ── Reproducibility metadata ──────────────────────────────────────────────────


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ},  # explicit env-passthrough
    ).stdout.strip()


def result_metadata(args: argparse.Namespace, questions_source: dict | None = None) -> dict:
    """Reproducibility block for every output JSON (code-style rule).

    ``questions_source`` (set only under ``--questions-json``) records the
    measurement-question provenance; omitted on the default 20-question path
    so that path's payload shape is unchanged.
    """
    import numpy as np
    import pandas as pd

    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("torch", "transformers", "vllm"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = "not-installed"
    meta = {
        "task": 559,
        "script": "scripts/issue559_base_prior_persona_panel.py",
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "versions": versions,
        "platform": platform.platform(),
        "base_model": BASE_MODEL,
        "hf_data_revision_pinned": HF_DATA_REV,
        "argv": sys.argv[1:],
    }
    if questions_source is not None:
        meta["questions_source"] = questions_source
    return meta


def write_json(path: Path, obj: dict) -> None:
    """Atomic-ish JSON write (tmp + rename), creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.rename(path)
    log.info("[write] %s", path)


# ── Phases ────────────────────────────────────────────────────────────────────


def phase_preflight(args: argparse.Namespace, tokenizer) -> dict:
    """CPU-only pre-GPU pipeline: asserts, gates, prompts, caps, guard self-test.

    With ``--questions-json`` the MEASUREMENT question list is overridden (the
    disjoint-30 path, plan v4 §3) and gated by ``disjoint_question_gate`` on
    the FULL list before any smoke slicing; the S0 gate keeps iterating the
    panel's 20 stored questions (``s0_questions``) regardless — under the
    30-list it would otherwise score zero stored slots.
    """
    print("[phase=preflight]", flush=True)
    personas, persona_prompts, panel_questions = load_panel()
    raw = load_raw_completions(S0_CELL)
    question_identity_gate(raw, panel_questions, personas)

    questions_source: dict | None = None
    if args.questions_json is not None:
        measurement_questions, questions_source = load_questions_json(args.questions_json)
        disjoint_question_gate(measurement_questions, panel_questions)
    else:
        measurement_questions = panel_questions

    use_personas = personas[: args.limit_personas] if args.limit_personas else personas
    use_questions = (
        measurement_questions[: args.limit_questions]
        if args.limit_questions
        else measurement_questions
    )
    use_s0_questions = (
        panel_questions[: args.limit_questions] if args.limit_questions else panel_questions
    )

    caps: dict[str, dict[str, int]] = {}
    prompt_lens: list[int] = []
    for p in use_personas:
        caps[p] = {}
        for q in use_questions:
            text = chat_prompt(tokenizer, persona_prompts[p], q)
            cap = per_question_R_cap(tokenizer, text, f"persona={p!r} q={q[:40]!r}")
            caps[p][q] = cap
            prompt_lens.append(MAX_LENGTH - cap - R_CAP_SAFETY_MARGIN)

    # Truncation-guard self-test on the REAL tokenizer (preflight-only unit).
    sample_prompt = chat_prompt(tokenizer, persona_prompts[use_personas[0]], use_questions[0])
    clean = build_slot_ids(tokenizer, sample_prompt, "A clean answer with no special symbols.")
    assert clean["slot_kind"] == "end_of_response" and clean["n_truncated_tokens"] == 0
    assert clean["ids"] == tokenizer.encode(
        sample_prompt + "A clean answer with no special symbols.", add_special_tokens=False
    ), "no-marker case must reduce exactly to the panel's full-string convention"
    dirty = build_slot_ids(tokenizer, sample_prompt, "An answer that emits ※ mid-way through.")
    assert dirty["slot_kind"] == "pre_marker" and dirty["n_truncated_tokens"] > 0
    assert MARKER_ID not in dirty["ids"] and BARE_MARKER_ID not in dirty["ids"]
    log.info("[preflight] truncation-guard self-test PASS")

    summary = {
        "n_personas": len(use_personas),
        "n_questions": len(use_questions),
        "n_prompts": len(prompt_lens),
        "prompt_len_min": min(prompt_lens),
        "prompt_len_max": max(prompt_lens),
        "cap_min": min(min(c.values()) for c in caps.values()),
        "cap_max": max(max(c.values()) for c in caps.values()),
        "question_identity_gate": "PASS",
        "truncation_guard_self_test": "PASS",
        "metadata": result_metadata(args, questions_source),
    }
    if questions_source is not None:
        # Disjoint path only — the default 20-question payload shape is unchanged.
        summary["n_s0_questions"] = len(use_s0_questions)
        summary["disjoint_question_gate"] = "PASS"
    write_json(args.out_dir / "preflight.json", summary)
    return {
        "personas": use_personas,
        "persona_prompts": persona_prompts,
        "questions": use_questions,
        "s0_questions": use_s0_questions,
        "questions_source": questions_source,
        "caps": caps,
        "raw": raw,
    }


def phase_gen(args: argparse.Namespace, tokenizer, ctx: dict) -> None:
    """vLLM greedy generation of the base model's own responses (#478 mirror)."""
    print("[phase=gen]", flush=True)
    from vllm import LLM, SamplingParams

    personas, persona_prompts, questions = ctx["personas"], ctx["persona_prompts"], ctx["questions"]
    caps = ctx["caps"]

    log.info("Loading vLLM %s (mem_util=%.2f) ...", BASE_MODEL, args.gpu_mem_util)
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=MAX_LENGTH,
        max_num_seqs=64,
        seed=42,
    )

    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    sampling: list = []
    for p in personas:
        for q in questions:
            text = chat_prompt(tokenizer, persona_prompts[p], q)
            prompts.append(text)
            keys.append((p, q))
            sampling.append(SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=caps[p][q]))

    log.info("vLLM batched greedy generate: %d prompts ...", len(prompts))
    outputs = llm.generate(prompts, sampling)

    R: dict[str, dict[str, str]] = {p: {} for p in personas}
    finish_reasons: dict[str, dict[str, str]] = {p: {} for p in personas}
    own_R_token_lens: dict[str, dict[str, int]] = {p: {} for p in personas}
    n_truncated = 0
    for out, (p, q) in zip(outputs, keys, strict=True):
        o = out.outputs[0]
        R[p][q] = o.text
        finish_reasons[p][q] = str(o.finish_reason)
        own_R_token_lens[p][q] = len(o.token_ids)
        if o.finish_reason == "length":
            n_truncated += 1
    truncation_rate = n_truncated / len(keys)
    log.info("Generation done: %d rows, truncation rate %.4f", len(keys), truncation_rate)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "is_stub": False,
        "eval_questions": questions,
        "personas": personas,
        "R": R,
        "finish_reasons": finish_reasons,
        "own_R_token_lens": own_R_token_lens,
        "n_truncated": n_truncated,
        "truncation_rate": truncation_rate,
        "generation_config": {
            "engine": "vllm",
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "seed": 42,
            "dtype": "bfloat16",
            "max_model_len": MAX_LENGTH,
            "max_num_seqs": 64,
            "max_tokens_rule": f"{MAX_LENGTH} - prompt_len - {R_CAP_SAFETY_MARGIN}, floor "
            f"{R_CAP_MIN} fail-loud",
            "gpu_memory_utilization": args.gpu_mem_util,
        },
        "metadata": result_metadata(args, ctx["questions_source"]),
    }
    if ctx["questions_source"] is not None:
        payload["questions_source"] = ctx["questions_source"]
    write_json(args.out_dir / "R_base_own.json", payload)

    # vLLM teardown BEFORE the HF scoring phase (gotchas rule).
    del llm
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()
    kill_vllm_children()
    nvidia_smi_assert_clean(0)


def s0_gate(args: argparse.Namespace, model, tokenizer, ctx: dict, device: str) -> dict:
    """S0 scoring-path reproduction gate (construction validity, plan §4.7).

    Re-scores the BASE side of #478 run ``K1_c00_seed42`` (stored trained-R
    slots from the pinned raw JSON — NO truncation guard, mirroring
    ``issue531_logit_rescore.py::build_items`` exactly) with THIS script's
    scoring path, and compares against the parquet's stored base-side floats.
    Gates (per #531): logp MAE < 1.0 nat and Spearman ≥ 0.995 vs BOTH the
    #531-rescored ``logp_base_rescored`` and the #478-original ``base_prior``.
    ``sys.exit(1)`` on any miss — no new measurement ships past a drifted
    construction.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    print("[phase=s0_gate]", flush=True)
    raw = ctx["raw"]
    # ALWAYS the panel's 20 stored questions (smoke-sliced) — the stored
    # K1_c00_seed42 slots only exist for those; under a --questions-json
    # 30-list the measurement list would skip every stored slot (plan v4 §5.1).
    personas, questions = ctx["personas"], ctx["s0_questions"]
    persona_prompts = ctx["persona_prompts"]
    R_eval: dict[str, dict[str, str]] = raw["R_eval"]

    items: list[tuple[tuple, list[int]]] = []
    trained_R_token_lens: dict[str, dict[str, int]] = {}
    for p in personas:
        qmap = R_eval[p]
        assert list(qmap.keys()) == raw["eval_questions"], f"R_eval question order drift for {p!r}"
        trained_R_token_lens[p] = {}
        for q_idx, q in enumerate(raw["eval_questions"]):
            if q not in questions:
                continue  # smoke slice
            prefix = chat_prompt(tokenizer, persona_prompts[p], q)
            # build_items convention: prefix + stored R verbatim, full-string
            # tokenize, NO truncation guard (mirror #531 exactly).
            ids = tokenizer.encode(prefix + qmap[q], add_special_tokens=False)
            items.append(((p, q_idx), ids))
            trained_R_token_lens[p][q] = len(tokenizer.encode(qmap[q], add_special_tokens=False))
    log.info("[s0] scoring %d stored base-side slots for %s", len(items), S0_CELL)
    scored = score_slots(model, tokenizer, items, device, args.batch_size)

    cell_id, seed = S0_CELL.rsplit("_seed", 1)
    tidy = pd.read_parquet(TIDY_LOGIT_PARQUET)
    sub = tidy[(tidy["cell_id"] == cell_id) & (tidy["seed"] == int(seed))]
    assert len(sub) == 700, f"expected 700 parquet rows for {S0_CELL}, got {len(sub)}"

    rows = []
    for row in sub.itertuples(index=False):
        key = (row.held_out_persona, int(row.question_idx))
        if key not in scored:
            continue  # smoke slice
        rows.append(
            {
                "got_logp": scored[key]["logp"],
                "got_z": scored[key]["z_marker"],
                "got_z_eos": scored[key]["z_eos"],
                "got_logZ": scored[key]["logZ"],
                "want_logp_rescored": float(row.logp_base_rescored),
                "want_logp_stored": float(row.base_prior),
                "want_z": float(row.z_base),
                "want_z_eos": float(row.z_eos_base),
                "want_logZ": float(row.logZ_base),
            }
        )
    assert len(rows) == len(items), (len(rows), len(items))

    def _cmp(got: np.ndarray, want: np.ndarray) -> dict:
        return {
            "mae_nats": float(np.mean(np.abs(got - want))),
            "max_abs_nats": float(np.max(np.abs(got - want))),
            "spearman": float(spearmanr(got, want).statistic),
            "n": len(got),
        }

    got_logp = np.array([r["got_logp"] for r in rows])
    checks = {
        "logp_vs_logp_base_rescored": _cmp(
            got_logp, np.array([r["want_logp_rescored"] for r in rows])
        ),
        "logp_vs_base_prior_stored": _cmp(
            got_logp, np.array([r["want_logp_stored"] for r in rows])
        ),
        "z_marker_vs_z_base": _cmp(
            np.array([r["got_z"] for r in rows]), np.array([r["want_z"] for r in rows])
        ),
        "z_eos_vs_z_eos_base": _cmp(
            np.array([r["got_z_eos"] for r in rows]), np.array([r["want_z_eos"] for r in rows])
        ),
        "logZ_vs_logZ_base": _cmp(
            np.array([r["got_logZ"] for r in rows]), np.array([r["want_logZ"] for r in rows])
        ),
    }
    gate_pass = True
    for name in ("logp_vs_logp_base_rescored", "logp_vs_base_prior_stored"):
        c = checks[name]
        ok = c["mae_nats"] < MAX_VALIDATION_MAE_NATS and c["spearman"] >= MIN_VALIDATION_SPEARMAN
        log.info(
            "[s0] %s: MAE=%.4f nats, max=%.4f, spearman=%.5f (n=%d) -> %s",
            name,
            c["mae_nats"],
            c["max_abs_nats"],
            c["spearman"],
            c["n"],
            "PASS" if ok else "FAIL",
        )
        gate_pass = gate_pass and ok

    payload = {
        "schema_version": SCHEMA_VERSION,
        "s0_cell": S0_CELL,
        "n_slots": len(rows),
        "checks": checks,
        "gates": {
            "mae_gate_nats": MAX_VALIDATION_MAE_NATS,
            "spearman_gate": MIN_VALIDATION_SPEARMAN,
            "pass": gate_pass,
        },
        "device": device,
        "trained_R_token_lens": trained_R_token_lens,
        "metadata": result_metadata(args, ctx["questions_source"]),
    }
    write_json(args.out_dir / "s0_validation.json", payload)
    if not gate_pass:
        log.error(
            "[s0] SCORING-PATH REPRODUCTION GATE FAILED — prompt/slot construction "
            "diverges from the panel's convention. NOT scoring any new slot."
        )
        sys.exit(1)
    return payload


def phase_score(args: argparse.Namespace, tokenizer, ctx: dict) -> None:
    """S0 gate, then the new four-float reads at the end-of-own-response slots."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    log.info("Loading HF model %s on %s (%s) ...", BASE_MODEL, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map={"": 0} if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()

    s0 = s0_gate(args, model, tokenizer, ctx, device)

    print("[phase=score]", flush=True)
    r_path = args.out_dir / "R_base_own.json"
    own = json.loads(r_path.read_text())
    assert own["schema_version"] == SCHEMA_VERSION, own.get("schema_version")
    assert not own.get("is_stub", False), f"{r_path} is a stub — refusing to score it"
    personas, questions = own["personas"], own["eval_questions"]
    persona_prompts = ctx["persona_prompts"]

    items: list[tuple[tuple, list[int]]] = []
    guard_meta: dict[tuple, dict] = {}
    for p in personas:
        for q_idx, q in enumerate(questions):
            prefix = chat_prompt(tokenizer, persona_prompts[p], q)
            job = build_slot_ids(tokenizer, prefix, own["R"][p][q])
            items.append(((p, q_idx), job["ids"]))
            guard_meta[(p, q_idx)] = {
                "slot_kind": job["slot_kind"],
                "n_truncated_tokens": job["n_truncated_tokens"],
            }
    log.info("[score] scoring %d NEW own-response slots", len(items))
    scored = score_slots(model, tokenizer, items, device, args.batch_size)

    per_persona: dict[str, dict] = {}
    n_pre_marker = 0
    argmax_counts = {"marker": 0, "eos": 0, "other": 0}
    for p in personas:
        rec: dict[str, list] = {
            "z_marker_per_q": [],
            "z_eos_per_q": [],
            "logZ_per_q": [],
            "logp_marker_per_q": [],
            "argmax_id_per_q": [],
            "slot_kind_per_q": [],
            "n_truncated_tokens_per_q": [],
            "finish_reason_per_q": [],
        }
        for q_idx, q in enumerate(questions):
            s = scored[(p, q_idx)]
            g = guard_meta[(p, q_idx)]
            rec["z_marker_per_q"].append(s["z_marker"])
            rec["z_eos_per_q"].append(s["z_eos"])
            rec["logZ_per_q"].append(s["logZ"])
            rec["logp_marker_per_q"].append(s["logp"])
            rec["argmax_id_per_q"].append(s["argmax_id"])
            rec["slot_kind_per_q"].append(g["slot_kind"])
            rec["n_truncated_tokens_per_q"].append(g["n_truncated_tokens"])
            rec["finish_reason_per_q"].append(own["finish_reasons"][p][q])
            if g["slot_kind"] == "pre_marker":
                n_pre_marker += 1
            if s["argmax_id"] == MARKER_ID:
                argmax_counts["marker"] += 1
            elif s["argmax_id"] == EOS_ID:
                argmax_counts["eos"] += 1
            else:
                argmax_counts["other"] += 1
        margins = np.array(rec["z_marker_per_q"]) - np.array(rec["z_eos_per_q"])
        rec["prior_margin_own"] = float(np.mean(margins))
        rec["prior_margin_own_median"] = float(np.median(margins))
        rec["prior_margin_own_iqr"] = [
            float(np.percentile(margins, 25)),
            float(np.percentile(margins, 75)),
        ]
        rec["prior_logp_own"] = float(np.mean(rec["logp_marker_per_q"]))
        per_persona[p] = rec

    n_slots = len(items)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "is_stub": False,
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "marker_token_id": MARKER_ID,
        "bare_marker_token_id": BARE_MARKER_ID,
        "eos_token": EOS_TOKEN,
        "eos_token_id": EOS_ID,
        "eval_questions": questions,
        "personas": personas,
        "per_persona": per_persona,
        "summary": {
            "n_personas": len(personas),
            "n_questions": len(questions),
            "n_slots": n_slots,
            "n_pre_marker_slots": n_pre_marker,
            "truncation_rate": own["truncation_rate"],
            "argmax_composition": {
                k: {"count": v, "rate": v / n_slots} for k, v in argmax_counts.items()
            },
        },
        "s0_validation_pass": bool(s0["gates"]["pass"]),
        "scoring_config": {
            "device": device,
            "dtype": str(dtype),
            "batch_size": args.batch_size,
            "convention": "full-string tokenize add_special_tokens=False, left-pad, "
            "logits[:, -1, :] (issue531_logit_rescore.py::score_slot) + #532 "
            "pre-marker truncation guard (issue532_followup_logp_slot.py::_slot_job)",
        },
        "metadata": result_metadata(args, ctx["questions_source"]),
    }
    if ctx["questions_source"] is not None:
        payload["questions_source"] = ctx["questions_source"]
    write_json(args.out_dir / "base_prior_own_persona_panel.json", payload)

    del model
    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()

    if args.reproduce_gate_json is not None:
        reproduce_gate(args, payload)


def reproduce_gate(args: argparse.Namespace, payload: dict) -> None:
    """20-question reproduction entry gate vs the committed parent prior (plan v4 §3).

    Compares this run's per-persona ``prior_margin_own`` (35 values) against
    ``--reproduce-gate-json`` (the committed parent
    ``base_prior_own_persona_panel.json``): gate = MAE < 0.1 nat AND Spearman
    ≥ 0.999. Writes ``repro_gate.json`` (always), then ``sys.exit(1)`` on a
    PRODUCTION miss so the chained disjoint invocation never starts. Diagnostics
    persisted alongside so a near-miss is attributable (plan v4 §14.9: text
    divergence ⇒ cross-pod resampling, re-anchor on scoring-path MAE over
    identical texts; identical texts + scoring drift ⇒ code-path failure):

    * per-slot ``logp_marker`` MAE over common (persona, question) slots;
    * generation text-identity rate vs the committed ``R_base_own.json``
      (resolved as the gate JSON's sibling).

    Smoke slices (``--limit-personas/--limit-questions``) compare the
    measured subset only (loudly labeled; Spearman needs n ≥ 3) and are
    NON-FATAL: the comparison runs and ``repro_gate.json`` is written, but the
    threshold check is not enforced (``[repro-gate] SMOKE SLICE — thresholds
    not enforced``) — a 2-question slice mean vs the committed 20-question
    means would near-certainly miss. The production (no-limits) gate asserts
    full 35-persona coverage and keeps exit-1 semantics.
    """
    import numpy as np
    from scipy.stats import spearmanr

    print("[phase=repro_gate]", flush=True)
    smoke = bool(args.limit_personas or args.limit_questions)
    committed = json.loads(args.reproduce_gate_json.read_text())
    assert committed.get("schema_version") == SCHEMA_VERSION, committed.get("schema_version")
    got_pp, want_pp = payload["per_persona"], committed["per_persona"]
    personas = sorted(set(got_pp) & set(want_pp))
    if not smoke:
        assert sorted(got_pp) == sorted(want_pp), "persona set != committed parent prior"
        assert len(personas) == 35, len(personas)
        assert payload["eval_questions"] == committed["eval_questions"], (
            "repro gate requires the canonical 20-question path on both sides "
            "(--reproduce-gate-json is incompatible with --questions-json)"
        )
    got = np.array([got_pp[p]["prior_margin_own"] for p in personas])
    want = np.array([want_pp[p]["prior_margin_own"] for p in personas])
    mae = float(np.mean(np.abs(got - want)))
    rho = float(spearmanr(got, want).statistic) if len(personas) >= 3 else None
    mae_ok = mae < REPRO_GATE_MAE_NATS
    rho_ok = rho >= REPRO_GATE_SPEARMAN if rho is not None else smoke
    gate_pass = bool(mae_ok and rho_ok)

    # Diagnostic 1: per-slot logp MAE over common (persona, question) slots.
    want_q_idx = {q: i for i, q in enumerate(committed["eval_questions"])}
    slot_diffs: list[float] = []
    for p in personas:
        got_rec, want_rec = got_pp[p], want_pp[p]
        for i, q in enumerate(payload["eval_questions"]):
            if q in want_q_idx:
                slot_diffs.append(
                    abs(
                        got_rec["logp_marker_per_q"][i]
                        - want_rec["logp_marker_per_q"][want_q_idx[q]]
                    )
                )
    per_slot_logp_mae = float(np.mean(slot_diffs)) if slot_diffs else None

    # Diagnostic 2: generation text-identity rate vs the committed R_base_own.
    committed_r_path = args.reproduce_gate_json.parent / "R_base_own.json"
    new_r_path = args.out_dir / "R_base_own.json"
    text_identity_rate = None
    n_common_texts = 0
    if committed_r_path.exists() and new_r_path.exists():
        want_R = json.loads(committed_r_path.read_text())["R"]
        got_R = json.loads(new_r_path.read_text())["R"]
        matches = []
        for p in personas:
            for q, text in got_R.get(p, {}).items():
                if q in want_R.get(p, {}):
                    matches.append(text == want_R[p][q])
        n_common_texts = len(matches)
        text_identity_rate = float(np.mean(matches)) if matches else None

    out = {
        "schema_version": SCHEMA_VERSION,
        "committed_reference": str(args.reproduce_gate_json),
        "n_personas_compared": len(personas),
        "smoke_slice": smoke,
        "per_persona_prior_mae_nats": mae,
        "per_persona_prior_spearman": rho,
        "gates": {
            "mae_gate_nats": REPRO_GATE_MAE_NATS,
            "spearman_gate": REPRO_GATE_SPEARMAN,
            "mae_pass": mae_ok,
            "spearman_pass": rho_ok,
            "pass": gate_pass,
        },
        "diagnostics": {
            "per_slot_logp_mae_nats": per_slot_logp_mae,
            "n_common_slots": len(slot_diffs),
            "generation_text_identity_rate": text_identity_rate,
            "n_common_texts": n_common_texts,
            "read_rule": "text divergence => cross-pod generation resampling (re-anchor on "
            "scoring-path MAE over identical texts); identical texts + prior drift => "
            "scoring code-path failure (plan v4 §14.9)",
        },
        "metadata": result_metadata(args),
    }
    write_json(args.out_dir / "repro_gate.json", out)
    log.info(
        "[repro-gate] MAE=%.4f nats (gate %.1f), spearman=%s (gate %.3f), n=%d -> %s%s",
        mae,
        REPRO_GATE_MAE_NATS,
        f"{rho:.5f}" if rho is not None else "n/a",
        REPRO_GATE_SPEARMAN,
        len(personas),
        "PASS" if gate_pass else "FAIL",
        " [SMOKE SLICE — not authoritative]" if smoke else "",
    )
    if smoke:
        # Pod-side gate smoke (--limit-personas/--limit-questions): the
        # comparison ran and repro_gate.json was written above, but a tiny
        # slice (e.g. 2-question means vs committed 20-question means) would
        # near-certainly miss the thresholds, so the check is non-fatal here.
        # Production (no limits) keeps exit-1 semantics below, unchanged.
        log.warning(
            "[repro-gate] SMOKE SLICE — thresholds not enforced (comparison ran, "
            "repro_gate.json written, slice read %s; exit-1 applies only to the "
            "full no-limits run)",
            "PASS" if gate_pass else "FAIL",
        )
    elif not gate_pass:
        log.error(
            "[repro-gate] 20-QUESTION REPRODUCTION GATE FAILED — fresh-pod code path does "
            "not reproduce the committed parent prior; the disjoint invocation must not "
            "start. Consult repro_gate.json diagnostics before any relax decision (a "
            "threshold bump is a plan amendment, never a silent edit)."
        )
        sys.exit(1)


def phase_upload(args: argparse.Namespace) -> None:
    """Upload all three result JSONs to the HF data repo (fail-loud).

    Raw generations land under ``raw_completions/`` per the upload policy;
    the two eval JSONs ride along under ``eval/`` so pod termination can
    never strand them (they are additionally committed to git by the VM).
    """
    print("[phase=upload]", flush=True)
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    prefix = args.upload_prefix
    targets = [
        (args.out_dir / "R_base_own.json", f"{prefix}/raw_completions/R_base_own.json"),
        (
            args.out_dir / "base_prior_own_persona_panel.json",
            f"{prefix}/eval/base_prior_own_persona_panel.json",
        ),
        (args.out_dir / "s0_validation.json", f"{prefix}/eval/s0_validation.json"),
    ]
    for local, remote in targets:
        result = _upload(
            local,
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=remote,
            upload_as_file=True,
        )
        if not result:
            raise RuntimeError(f"HF upload FAILED for {local} -> {remote} (fail-loud)")
        log.info("[upload] %s -> %s", local, result)


def write_sentinel(args: argparse.Namespace, note: str, smoke: bool) -> None:
    """End-of-run sentinel conforming to poll_pipeline._SENTINEL_REQUIRED_KEYS.

    Smoke slices write into ``--out-dir`` so a pod-side smoke can never feed a
    sentinel to the orchestrator's ``/workspace/logs`` poll loop.
    """
    sentinel_dir = SENTINEL_DIR_POD if (SENTINEL_DIR_POD.exists() and not smoke) else args.out_dir
    path = sentinel_dir / f"issue-559-epm_results-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 559,
        "by": "issue559_base_prior_persona_panel.py",
        "ts": datetime.now(UTC).isoformat(),
        "note": note,
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("[sentinel] wrote %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #559 — base own-response prior on the 35-persona held-out panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["preflight", "gen", "score", "upload", "all"],
        default="all",
    )
    parser.add_argument("--limit-personas", type=int, default=None, help="smoke: first N personas")
    parser.add_argument(
        "--limit-questions", type=int, default=None, help="smoke: first M questions"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90, dest="gpu_mem_util")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="scoring device (cpu for the VM smoke; bf16 on cuda, fp32 on cpu)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="skip the HF upload phase (smoke runs)",
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=None,
        dest="questions_json",
        help="override the MEASUREMENT question list (disjoint-30 path, plan v4 §3); "
        "triggers the disjoint_question_gate on the FULL list before any smoke slicing. "
        "The S0 gate keeps using the panel's 20 stored questions.",
    )
    parser.add_argument(
        "--upload-prefix",
        default=HF_UPLOAD_PREFIX,
        dest="upload_prefix",
        help="HF data-repo upload prefix (the disjoint run nests under "
        f"{HF_UPLOAD_PREFIX}/disjoint_question_prior so parent artifacts can never be "
        "overwritten)",
    )
    parser.add_argument(
        "--reproduce-gate-json",
        type=Path,
        default=None,
        dest="reproduce_gate_json",
        help="committed parent base_prior_own_persona_panel.json — post-score per-persona "
        f"reproduction gate (MAE < {REPRO_GATE_MAE_NATS} nat AND Spearman >= "
        f"{REPRO_GATE_SPEARMAN}); writes repro_gate.json, exit 1 on a production miss "
        "(smoke slices: non-fatal, thresholds not enforced). Gate invocations NEVER "
        "write the results sentinel and terminate with [phase=gate_passed], not "
        "[phase=done] — the chained disjoint invocation owns run-complete signaling",
    )
    args = parser.parse_args()
    smoke = bool(args.limit_personas or args.limit_questions)
    assert not (args.questions_json is not None and args.reproduce_gate_json is not None), (
        "--reproduce-gate-json compares 20-question priors and is incompatible with "
        "--questions-json (the disjoint run; plan v4 §3 chains them as separate invocations)"
    )

    load_dotenv()
    if Path("/workspace").exists():
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert marker_ids == [MARKER_ID], f"' ※' encodes to {marker_ids}, expected [{MARKER_ID}]"
    bare_ids = tokenizer.encode("※", add_special_tokens=False)
    assert bare_ids == [BARE_MARKER_ID], f"bare ※ encodes to {bare_ids}"
    assert tokenizer.convert_tokens_to_ids(EOS_TOKEN) == EOS_ID

    ctx = phase_preflight(args, tokenizer)

    if args.phase in ("gen", "all"):
        phase_gen(args, tokenizer, ctx)
    if args.phase in ("score", "all"):
        phase_score(args, tokenizer, ctx)
    if args.phase in ("upload", "all"):
        if args.no_upload or smoke:
            log.info("[upload] skipped (%s)", "--no-upload" if args.no_upload else "smoke slice")
        else:
            phase_upload(args)

    if args.reproduce_gate_json is not None:
        # Entry-gate invocation (plan v4 §3, invocation 1 of 2): NEVER signal
        # run-complete. The launch command chains gate && disjoint in one
        # nohup; a sentinel + [phase=done] here would make poll_pipeline.py
        # post epm:results ~25 min early, triggering premature
        # upload-verification + pod termination mid-disjoint-run. The chained
        # disjoint invocation (no --reproduce-gate-json) owns the sentinel +
        # [phase=done]. Reaching this line means the gate did not abort
        # (production miss => reproduce_gate sys.exit(1); smoke slices are
        # non-fatal by design). NOTE: underscore, not hyphen — the poller's
        # PHASE_RE is [a-z0-9_]+, so [phase=gate_passed] parses as the full
        # token while a hyphenated form would truncate to "gate".
        print("[phase=gate_passed]", flush=True)
        return 0
    if args.phase == "all":
        write_sentinel(
            args,
            note=(
                f"issue559 base own-response prior measurement complete: "
                f"{len(ctx['personas'])} personas x {len(ctx['questions'])} questions; "
                f"outputs in {args.out_dir} (R_base_own.json, "
                f"base_prior_own_persona_panel.json, s0_validation.json)"
                + (" [disjoint-30 measurement questions]" if args.questions_json else "")
                + (" [SMOKE SLICE]" if smoke else "")
            ),
            smoke=smoke,
        )
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
