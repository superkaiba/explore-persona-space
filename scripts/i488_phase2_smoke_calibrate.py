# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #488 Phase 2 — smoke calibrate (gates only; production trains all 6 fracs).

Plan v3 §7 (replaces v2). Runs A1 (helpful assistant) and G2 (skeptical
scientist) at the recipe encoded in scripts/i488_phase23_train.py defaults —
round-10 Path A descope (plan v2 §8 line 351): lr=1e-6, r=8, alpha=16,
75 rows per side, 3 epochs total, saving at all 6 fracs ∈ {0.10, 0.25,
0.50, 1.00, 2.00, 3.00}. Phase-2 gates per v3 §7:

  1. **Label-mask audit (KEEP — v3 Gate 1).** Print loss-bearing positions
     on one positive + one negative row of A1's training data. Positive
     row's loss-positions are exactly the marker token (id 83399) +
     immediately-following EOS. Negative row's loss-position is exactly
     ``<|im_end|>`` (id 151645). Fail → ``reason: label_mask_wrong_slot``.
  2. **Log-prob shift (KEEP — v3 Gate 2').** At frac=3.00,
     ``Δlogp = log P(' ※')_trained − log P(' ※')_base`` ≥ +0.5 nat on at
     least one of (A1, G2) at the on-diag post-response slot. Free v3
     addition: also record Δlogp for a frac=0.00 / 0-SFT-step forward
     pass on the same smoke prompts (the noise floor on the 1-pos
     slice). If empirical noise ≥ 0.3 nat, the implementer is authorized
     to tighten Gate 2' from +0.5 nat to a value in [0.2 nat, 1.0 nat]
     ≥ 3σ above the noise — logged but NOT auto-applied in code (an
     operator decision). Fail → ``reason: smoke_no_logprob_shift``.
  3. **Off-diagonal log-prob shift (NEW — v3 Gate 3, replaces v2's
     emission-rate version).** For A1's frac=3.00 adapter, evaluate on 6
     off-diagonal target contexts (B1, F1, G1, A3, D2, B5). For each
     off-diag cell, compute ``Δlogp_off = log P(' ※')_trained −
     log P(' ※')_base`` at the post-response slot (HF teacher-forced,
     analogous to Gate 2' at the SOURCE slot but at off-diag target
     contexts). PASS criterion: at least 4 of 6 off-diag cells must have
     ``Δlogp_off ≥ +0.2 nat``. Failure → ``reason: offdiag_no_logprob_shift``.
     v2's emission-rate Gate 3 (``tie_mass_off ≤ 0.85``) is DROPPED — it
     provably FAILs at the FLOOR on the 1-pos / 0-neg smoke slice
     (round-6 evidence: 0.000 emission across all 12 smoke cells).
  4. **EOS-gradient check (KEEP — v3 Gate 4).** At frac=3.00, A1's trained
     model evaluated on the off-diagonal A1→B1 cell must show trained
     log P(' ※') at the post-response slot moved DOWN by ≥ 0.2 nats vs
     base (this is a separate sanity probe — confirms the LoRA has
     moved the EOS distribution).

Architecturally unified with the sweep (CLAUDE.md Step 6d.0): smoke IS the
sweep with --conds A1 G2 --seeds 42 --fracs all-six on the SAME dispatcher.
Phase-2 (this script) is the calibration LAYER that consumes the smoke
adapters and runs the gates. It does NOT spawn the trainer — the dispatcher
does that. This script is invoked AFTER the smoke train.

v3 also removes pre-sweep frac picking — production trains all 6 fracs and
the headline frac is selected POST-HOC by ``scripts/i488_phase5_analyze.py``
under a ρ-blind deterministic rule (§6.2.D). This script therefore does NOT
emit ``picked_fracs.json``.

Outputs:
* ``logs/issue_488/smoke/label_mask_audit.txt`` — positive/negative row
  label positions (from the actual collator's batch).
* ``logs/issue_488/smoke/logprob_shift.json`` — Gate 2' on-diag Δlogp at
  frac=3.00 per cell (A1, G2) + the noise distribution at frac=0.00.
* ``logs/issue_488/smoke/offdiag_logprob_shift.json`` — Gate 3 v3
  per-off-diag-cell Δlogp at frac=3.00 (A1 adapter, 6 contexts).
* ``logs/issue_488/smoke/eos_gradient.json`` — trained vs base log P(' ※')
  at A1→B1 post-response slot at frac=3.00.
* ``figures/issue_488/smoke_logprob_noise.png`` — empirical noise
  distribution on the 1-pos slice (Gate 2' standing rec).
* On FAIL: ``/workspace/logs/issue-488-smoke-failed.json`` sentinel +
  non-zero exit (the dispatcher escalates via that sentinel).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i488.phase2")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i488")
SMOKE_LOG_DIR = Path("logs/issue_488/smoke")
SENTINEL_PATH = Path("/workspace/logs/issue-488-smoke-failed.json")

ALL_FRACS = (0.10, 0.25, 0.50, 1.00, 2.00, 3.00)
IM_END_TOKEN_ID = 151645
EOS_GRADIENT_MIN_NATS = 0.2

# Gate 2' (v3): on-diag log-prob shift at frac=3.00 must clear this threshold
# on at least one of (A1, G2). Standing rec: tighten to ≥ noise_floor + 3σ
# if the empirical noise on frac=0.00 base-only forwards is ≥ 0.3 nat.
GATE2_LOGPROB_SHIFT_MIN_NATS = 0.5
GATE2_NOISE_TIGHTEN_THRESHOLD_NATS = 0.3

# Gate 3 (v3 — NEW, replaces v2's emission-rate tie_mass_off).
# Per off-diag cell, the trained adapter must shift log P(' ※') up by at
# least this many nats at the post-response slot; at least 4 of 6 cells
# must clear it.
GATE3_OFFDIAG_LOGPROB_SHIFT_MIN_NATS = 0.2
GATE3_OFFDIAG_CELLS_REQUIRED = 4

# The 6 off-diagonal target contexts probed at Gate 3 + Gate 4 (unchanged
# from v2 — diverse subset spanning default, frame, persona, stylized,
# paraphrase, wrap classes).
OFFDIAG_MINI_GRID = ("B1", "F1", "G1", "A3", "D2", "B5")
SMOKE_CELLS = ("A1", "G2")
SMOKE_EVAL_N_PROBES = 10
# Gate 2'/3/4 use teacher-forced log-prob on a single held-out Q per
# (cell, target context) — one forward each, fast.
LOGP_FLOOR = -50.0

# Gate 2' Δlogp + Gate 3 off-diag Δlogp are measured at frac=3.00 (the most
# trained checkpoint in the smoke run; max signal-to-noise per v3 §7).
GATE_PROBE_FRAC = 3.00


def _frac_tag(frac: float) -> str:
    return f"frac{round(frac * 100):03d}"


def _write_sentinel(reason_key: str, reason: str, extra: dict | None = None) -> None:
    SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": 488,
        "phase": "phase2_smoke",
        "failure_class": "code",
        "reason": reason_key,
        "reason_long": reason,
        "wrote_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "sentinel_schema_version": 1,
        "kind": "epm:failure",
        "version": 1,
    }
    if extra:
        payload["extra"] = extra
    SENTINEL_PATH.write_text(json.dumps(payload, indent=2))
    logger.error("Smoke gate FAIL → wrote %s (%s)", SENTINEL_PATH, reason_key)


# ── Gate 1: label-mask audit ────────────────────────────────────────────


def _label_mask_audit(audit_train_jsonl: Path, tokenizer) -> str:
    """Walk the saved A1 train.jsonl, locate ONE positive + ONE negative row,
    apply the MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True,
    im_end_token_id=151645), and report which token positions carry loss.

    Returns the formatted audit report (also written to disk).

    Raises:
        AssertionError: if the positive row's only loss-bearing position is
            not the marker (+ EOS), OR if the negative row's only loss-bearing
            position is not id 151645.
    """
    import torch
    from transformers import DataCollatorForLanguageModeling

    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    if not audit_train_jsonl.exists():
        raise FileNotFoundError(
            f"Audit train rows missing: {audit_train_jsonl}. "
            "Run Phase 2 train dispatcher first (it persists train rows for audit)."
        )

    rows: list[dict] = []
    for line in audit_train_jsonl.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{audit_train_jsonl} is empty.")

    # Find one positive (has_marker) and one negative (no marker in completion).
    positive_row = None
    negative_row = None
    for r in rows:
        completion = r["completion"][0]["content"]
        if MARKER_TEXT in completion:
            if positive_row is None:
                positive_row = r
        elif negative_row is None:
            negative_row = r
        if positive_row is not None and negative_row is not None:
            break
    if positive_row is None or negative_row is None:
        raise AssertionError(
            f"Couldn't find both positive + negative rows in {audit_train_jsonl} "
            f"(seen {len(rows)}; positives need MARKER_TEXT in completion)."
        )

    # Re-tokenize both rows the way SFTTrainer would: apply_chat_template on
    # prompt + completion concatenated as one assistant turn.
    def tokenize(row: dict) -> dict:
        full_msgs = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Build the response-only mask: the SFT default masks the prompt with -100
        # and trains on the assistant completion. We approximate that by tokenizing
        # the prompt-only chat string and finding its length.
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids) :]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }

    pos_feat = tokenize(positive_row)
    neg_feat = tokenize(negative_row)

    inner = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=[MARKER_ID],
        tail_tokens=0,
        suppress_at_post_response_slot=True,
        im_end_token_id=IM_END_TOKEN_ID,
    )
    # Process positive and negative rows separately at batch_size=1: the inner
    # ``DataCollatorForLanguageModeling`` cannot pad ``labels`` from features
    # whose token lengths differ (pos vs neg row token lengths diverge in the
    # 600-800 range with real training data; one_pad uses ``pad_token_id`` for
    # ``input_ids`` but rejects the heterogeneous ``labels`` shapes — round-5
    # crash, traceback in epm:failure v1). This audit's purpose is to verify
    # per-row label-mask correctness, not batched padding behavior — the
    # production trainer never mixes pos+neg into a single 2-row batch through
    # this code path. Keeping each row at batch_size=1 sidesteps the padding
    # issue entirely while preserving the audit invariants.
    pos_batch = collator([pos_feat])
    neg_batch = collator([neg_feat])

    pos_labels = pos_batch["labels"][0]
    pos_input = pos_batch["input_ids"][0]
    neg_labels = neg_batch["labels"][0]
    neg_input = neg_batch["input_ids"][0]

    pos_loss_positions = (pos_labels != -100).nonzero(as_tuple=True)[0].tolist()
    neg_loss_positions = (neg_labels != -100).nonzero(as_tuple=True)[0].tolist()

    pos_loss_ids = [int(pos_input[p].item()) for p in pos_loss_positions]
    neg_loss_ids = [int(neg_input[p].item()) for p in neg_loss_positions]

    audit_lines = [
        "# i488 Phase 2 — Label-mask audit",
        f"timestamp: {datetime.datetime.now(datetime.UTC).isoformat()}",
        f"audit_train_jsonl: {audit_train_jsonl}",
        f"marker_text: {MARKER_TEXT!r}  marker_id: {MARKER_ID}",
        f"im_end_token_id: {IM_END_TOKEN_ID}",
        "",
        "## Positive row",
        f"  loss-bearing positions: {pos_loss_positions}",
        f"  loss-bearing token ids: {pos_loss_ids}",
        "  expected: [MARKER_ID + immediately-following EOS]",
        "",
        "## Negative row",
        f"  loss-bearing positions: {neg_loss_positions}",
        f"  loss-bearing token ids: {neg_loss_ids}",
        f"  expected: a SINGLE position whose token id is {IM_END_TOKEN_ID}",
    ]
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_LOG_DIR / "label_mask_audit.txt").write_text("\n".join(audit_lines) + "\n")
    logger.info("Wrote label_mask_audit.txt")

    # FAIL LOUD on shape mismatch.
    if MARKER_ID not in pos_loss_ids:
        raise AssertionError(
            f"POSITIVE row audit FAIL: MARKER_ID {MARKER_ID} not in "
            f"loss-bearing token ids {pos_loss_ids}."
        )
    if neg_loss_ids != [IM_END_TOKEN_ID]:
        raise AssertionError(
            f"NEGATIVE row audit FAIL: expected single position with id "
            f"{IM_END_TOKEN_ID}, got {neg_loss_ids}."
        )
    return "\n".join(audit_lines)


# ── Gates 2'/3/4: log-prob shift + off-diag log-prob shift + EOS-gradient ──


def _download_adapter(cid: str, seed: int, frac: float) -> str:
    """Download one (cid, seed, frac) adapter from HF; return local path."""
    from huggingface_hub import hf_hub_download

    subpath = f"adapters/i488_{cid}_seed{seed}_{_frac_tag(frac)}"
    local_target = LOCAL_ADAPTER_CACHE / subpath
    local_target.mkdir(parents=True, exist_ok=True)
    for fname in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ):
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                revision="main",
                filename=f"{subpath}/{fname}",
                local_dir=LOCAL_ADAPTER_CACHE,
            )
        except Exception as e:
            if fname in ("adapter_model.safetensors", "adapter_config.json"):
                raise RuntimeError(f"required file {subpath}/{fname} not on HF: {e}") from e
    return str(local_target)


def _render_noise_distribution_figure(
    noise_logps: list[float],
    noise_mean: float,
    noise_sd: float,
    outpath: Path,
) -> None:
    """Render the Gate 2' noise-floor histogram + sd annotation.

    Plan v3 §7 standing rec: log the base-only log-prob distribution from
    the frac=0.00 forward passes so a reviewer can see the empirical noise
    floor against which Gate 2's +0.5-nat threshold is being judged. The
    figure is informational; tightening Gate 2' from +0.5 to a higher
    value is an OPERATOR decision (the JSON sidecar carries the flag).
    """
    if not noise_logps:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(noise_logps, bins=min(8, max(2, len(noise_logps))), color="steelblue", alpha=0.8)
    ax.axvline(noise_mean, color="black", lw=1.0, label=f"mean = {noise_mean:.3f}")
    ax.axvline(
        noise_mean + noise_sd,
        color="darkorange",
        ls="--",
        lw=0.8,
        label=f"+1 sd = {noise_sd:.3f}",
    )
    ax.axvline(noise_mean - noise_sd, color="darkorange", ls="--", lw=0.8)
    ax.set_xlabel("base log P(' ※') at on-diag post-response slot (nats)")
    ax.set_ylabel("count")
    ax.set_title(
        "Gate 2' noise floor — base-only forwards on the 1-pos smoke slice\n"
        f"(n={len(noise_logps)}, sd={noise_sd:.3f} nats; "
        f"tighten Gate 2' if sd ≥ {GATE2_NOISE_TIGHTEN_THRESHOLD_NATS})"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _post_response_slot_logprob(
    llm,
    sp_logprob,
    tokenizer,
    prompt_text: str,
    R_text: str,
    lora_request,
) -> float:
    """Teacher-force `prompt + R + MARKER_TEXT` through (base or trained); return
    log P(' ※') at the FINAL position (the post-response marker slot).
    """
    full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    if full_ids[-1] != MARKER_ID or full_ids.count(MARKER_ID) != 1:
        raise RuntimeError(
            f"marker slot drift in EOS-gradient probe: full_ids[-1]={full_ids[-1]}, "
            f"count={full_ids.count(MARKER_ID)}"
        )
    slot = len(full_ids) - 1
    outputs = llm.generate([{"prompt_token_ids": full_ids}], sp_logprob, lora_request=lora_request)
    out = outputs[0]
    spec = out.prompt_logprobs[slot]
    if spec is None or MARKER_ID not in spec:
        raise RuntimeError(
            f"post-response slot probe: prompt_logprobs[{slot}] missing MARKER_ID; "
            f"top keys={list((spec or {}).keys())[:5]}"
        )
    return max(float(spec[MARKER_ID].logprob), LOGP_FLOOR)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - CLI dispatch + gates
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--audit-train-jsonl",
        default="data/issue_488/train_rows/i488_A1_seed42.jsonl",
        help="Path to A1 train.jsonl persisted by the Phase 3 train script.",
    )
    ap.add_argument(
        "--smoke-seed",
        type=int,
        default=42,
        help="Smoke seed (matches the Phase 3 dispatcher's --seeds 42 --conds A1 G2).",
    )
    ap.add_argument(
        "--n-probes-emission",
        type=int,
        default=SMOKE_EVAL_N_PROBES,
        help="Number of held-out Q to use for the source emission probe.",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Marker assert.
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    # ── Gate 1: label-mask audit ──
    try:
        _label_mask_audit(Path(args.audit_train_jsonl), tokenizer)
    except AssertionError as e:
        _write_sentinel("label_mask_wrong_slot", str(e))
        return 2

    # ── Gates 2'/3/4: spin up vLLM, evaluate A1 + G2 adapters at frac=3.00 ──
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    held_out = json.loads(Path("data/issue_488/q_held_out_20.json").read_text())["questions"]
    # Gate 2'/3/4 each use ONE held-out Q per (cell, target) — teacher-forced
    # log-prob is a single forward, so per-cell variance is the relevant
    # noise control (we average across cells, not across Q within a cell).
    # Use the first N=`n_probes` held-out Qs to average per cell.
    held_out_probe = held_out[: args.n_probes_emission]
    class_d_rewrites = load_class_d_rewrites()

    # R_test for the off-diag / EOS-gradient probes. If inherited #460
    # R_test is missing, regenerate on-policy from base via vLLM below.
    _r_test_inherited_path = Path("data/issue_460/R_test.json")
    if _r_test_inherited_path.exists():
        R_test_inherited = json.loads(_r_test_inherited_path.read_text())["completions"]
    else:
        logger.warning(
            "Inherited %s missing; EOS-gradient + off-diag probes will regenerate "
            "R on-policy from base.",
            _r_test_inherited_path,
        )
        R_test_inherited = {}

    logger.info("Loading vLLM %s on GPU %d", BASE_MODEL, args.gpu_id)
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=4096,
    )
    sp_R = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=1024, seed=42)
    sp_logprob = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    # Helpers bound to the loaded vLLM instance + the resolved R cache. Lifted
    # to lambdas-with-defaults so ruff F821 doesn't flag the late-binding of
    # `llm` in the closure (Python is fine with it; ruff is conservative).
    def _resolve_R_for(
        target_cid: str,
        target_prompt: str,
        probe_q: str,
        _llm=llm,
        _sp_R=sp_R,
        _R_inh=R_test_inherited,
    ) -> str:
        """Get R for (target persona context, q): inherited if present, else
        generate on-policy from base via vLLM."""
        canned = (
            _R_inh.get(target_cid, {}).get(probe_q, {}).get("response_text")
            if isinstance(_R_inh, dict)
            else None
        )
        if canned:
            return canned
        gen = _llm.generate([target_prompt], _sp_R, lora_request=None)
        return gen[0].outputs[0].text

    def _on_diag_probe_logp(
        cid: str,
        lora_request,
        probe_q: str,
        _llm=llm,
        _sp_logprob=sp_logprob,
        _tok=tokenizer,
        _crw=class_d_rewrites,
    ) -> tuple[float, str]:
        """One Δlogp probe at the on-diagonal source/target slot for `cid`.

        Returns (logp_marker_at_post_response_slot, R_used). Uses the existing
        vLLM teacher-forced primitive at the post-response marker slot.
        """
        cond_source = CONDITIONS_BY_ID[cid]
        prompt_text = build_prompt_for_condition(cond_source, probe_q, _tok, _crw)
        R_text = _resolve_R_for(cid, prompt_text, probe_q)
        logp = _post_response_slot_logprob(
            _llm, _sp_logprob, _tok, prompt_text, R_text, lora_request=lora_request
        )
        return logp, R_text

    def _off_diag_probe_logp(
        source_cid: str,
        target_cid: str,
        lora_request,
        probe_q: str,
        _llm=llm,
        _sp_logprob=sp_logprob,
        _tok=tokenizer,
        _crw=class_d_rewrites,
    ) -> tuple[float, str]:
        """One Δlogp probe at the off-diagonal slot: trained at source persona's
        adapter, evaluated at target persona's context. The probe text is
        ``T_target(probe_q) + R_target + ' ※'`` and the score is at the marker
        slot. Uses the existing vLLM teacher-forced primitive.

        Returns (logp_marker_at_post_response_slot, R_used).
        """
        del source_cid  # documented for caller; not needed for the eval prompt
        cond_target = CONDITIONS_BY_ID[target_cid]
        prompt_text = build_prompt_for_condition(cond_target, probe_q, _tok, _crw)
        R_text = _resolve_R_for(target_cid, prompt_text, probe_q)
        logp = _post_response_slot_logprob(
            _llm, _sp_logprob, _tok, prompt_text, R_text, lora_request=lora_request
        )
        return logp, R_text

    # ── Gate 2': on-diag log-prob shift at frac=3.00 for each cell ──
    # Plus the v3 free addition: a dispersion measurement using
    # base-only forwards (no LoRA loaded) on the smoke probes.
    #
    # CAVEAT (round-1 code-review Major / round-8 C1): the dispersion we
    # capture here is the BETWEEN-PROMPT standard deviation of base
    # log P(' ※') across n=2 held-out probes per smoke cell — i.e.,
    # prompt-to-prompt variability of the base model, NOT a Δlogp noise
    # floor under a null. A genuine Δlogp noise floor would require
    # SAME-prompt repeats of trained − base (e.g. a 0-step / freshly-init
    # LoRA whose Δ ≡ 0 by construction, with SD across probes the
    # measurement noise floor). The "tighten_gate2" derived signal here is
    # therefore an ADVISORY indicator of high prompt-to-prompt dispersion,
    # not evidence that Gate 2's +0.2-nat threshold sits inside noise.
    # JSON keys are named to reflect what is actually measured so a future
    # reader doesn't conflate it with a Δ noise floor.
    logprob_shift_payload: dict = {
        "gate_version": "v3",
        "probe_frac": GATE_PROBE_FRAC,
        "min_shift_nats": GATE2_LOGPROB_SHIFT_MIN_NATS,
        "cells": {},
        "between_prompt_base_logp_sd": {},
    }

    # Between-prompt base-logp dispersion: take 2 probes per smoke cell,
    # compute the base log-prob at the on-diag post-response slot. The
    # trained forward at frac=0.00 IS the base model (no LoRA), so the "Δ"
    # is mechanically zero per probe; what we report is the BETWEEN-PROMPT
    # SD of base log P(' ※') across probes (NOT a Δlogp noise floor — see
    # caveat block above).
    noise_logps: list[float] = []
    for cid in SMOKE_CELLS:
        for probe_q in held_out_probe[: min(2, len(held_out_probe))]:
            try:
                logp, _ = _on_diag_probe_logp(cid, lora_request=None, probe_q=probe_q)
                noise_logps.append(logp)
            except Exception as e:
                logger.warning(
                    "Between-prompt base-logp probe failed for %s q=%s: %s",
                    cid,
                    probe_q[:40],
                    e,
                )
    if noise_logps:
        noise_mean = float(np.mean(noise_logps))
        noise_sd = float(np.std(noise_logps))
        noise_range = (
            float(np.max(noise_logps) - np.min(noise_logps)) if len(noise_logps) > 1 else 0.0
        )
    else:
        noise_mean = noise_sd = noise_range = float("nan")
    logprob_shift_payload["between_prompt_base_logp_sd"] = {
        "n_probes": len(noise_logps),
        "logps": noise_logps,
        "mean": noise_mean,
        "sd": noise_sd,
        "range": noise_range,
        "tighten_threshold_nats": GATE2_NOISE_TIGHTEN_THRESHOLD_NATS,
        # Advisory ONLY: high prompt-to-prompt dispersion of base log-prob
        # is suggestive that Gate 2's +0.2-nat threshold may be tight
        # relative to ambient base variability, but is NOT a Δlogp noise
        # measurement. The operator should NOT auto-tighten Gate 2 from
        # this signal; treat as a flag to inspect cell-level Δs.
        "operator_advisory_between_prompt_dispersion_high": (
            noise_sd >= GATE2_NOISE_TIGHTEN_THRESHOLD_NATS if not np.isnan(noise_sd) else False
        ),
        "_caveat": (
            "Between-prompt SD of base log P(' ※'), not Δlogp noise floor. "
            "A genuine Δ noise floor needs same-prompt repeats of trained − "
            "base (e.g. against a 0-step LoRA); this is prompt-to-prompt "
            "dispersion of the base model only."
        ),
    }
    logger.info(
        "Gate 2' between-prompt base-logp dispersion: n=%d mean=%.4f sd=%.4f range=%.4f "
        "(advisory; NOT a Δlogp noise floor)",
        len(noise_logps),
        noise_mean,
        noise_sd,
        noise_range,
    )

    # Gate 2' per-cell Δlogp at frac=3.00.
    any_cell_passed = False
    for cid in SMOKE_CELLS:
        try:
            adapter_path = _download_adapter(cid, args.smoke_seed, GATE_PROBE_FRAC)
        except Exception as e:
            logprob_shift_payload["cells"][cid] = {
                "error": f"adapter download failed: {e}",
                "passes_gate2_prime": False,
            }
            continue
        lora = LoRARequest(
            lora_name=f"{cid}_gate2_{_frac_tag(GATE_PROBE_FRAC)}",
            lora_int_id=round(GATE_PROBE_FRAC * 100) + ord(cid[0]) * 1000,
            lora_path=adapter_path,
        )
        probe_q = held_out_probe[0]
        try:
            base_logp, _R_used = _on_diag_probe_logp(cid, lora_request=None, probe_q=probe_q)
            trained_logp, _ = _on_diag_probe_logp(cid, lora_request=lora, probe_q=probe_q)
        except Exception as e:
            logprob_shift_payload["cells"][cid] = {
                "error": f"on-diag probe failed: {e}",
                "passes_gate2_prime": False,
            }
            continue
        delta = trained_logp - base_logp
        passed = delta >= GATE2_LOGPROB_SHIFT_MIN_NATS
        logprob_shift_payload["cells"][cid] = {
            "probe_q": probe_q,
            "base_logp_marker": base_logp,
            "trained_logp_marker": trained_logp,
            "delta_nats": delta,
            "min_shift_nats": GATE2_LOGPROB_SHIFT_MIN_NATS,
            "passes_gate2_prime": passed,
        }
        if passed:
            any_cell_passed = True
        logger.info(
            "Gate 2' %s frac=%.2f: base=%.4f trained=%.4f Δ=%.4f pass=%s",
            cid,
            GATE_PROBE_FRAC,
            base_logp,
            trained_logp,
            delta,
            passed,
        )

    (SMOKE_LOG_DIR / "logprob_shift.json").write_text(json.dumps(logprob_shift_payload, indent=2))

    # Try to write the noise-distribution figure (best-effort; failure here
    # does NOT fail the gate — the JSON is the contract).
    try:
        _render_noise_distribution_figure(
            noise_logps,
            noise_mean,
            noise_sd,
            outpath=Path("figures/issue_488/smoke_logprob_noise.png"),
        )
    except Exception as e:
        logger.warning("Noise-distribution figure render failed: %s", e)

    if not any_cell_passed:
        _write_sentinel(
            "smoke_no_logprob_shift",
            f"Neither A1 nor G2 had Δlogp ≥ {GATE2_LOGPROB_SHIFT_MIN_NATS} nats at "
            f"frac={GATE_PROBE_FRAC} on the on-diag post-response slot. "
            f"payload={logprob_shift_payload['cells']}",
            extra=logprob_shift_payload,
        )
        return 3

    # ── Gate 3 (v3): off-diagonal log-prob shift at A1's frac=3.00 adapter ──
    # 6 off-diag contexts (B1, F1, G1, A3, D2, B5). PASS = at least
    # GATE3_OFFDIAG_CELLS_REQUIRED (4) cells have Δlogp_off ≥
    # GATE3_OFFDIAG_LOGPROB_SHIFT_MIN_NATS (0.2 nat).
    offdiag_payload: dict = {
        "gate_version": "v3",
        "probe_frac": GATE_PROBE_FRAC,
        "min_shift_nats_per_cell": GATE3_OFFDIAG_LOGPROB_SHIFT_MIN_NATS,
        "min_cells_required": GATE3_OFFDIAG_CELLS_REQUIRED,
        "n_cells_probed": len(OFFDIAG_MINI_GRID),
        "cells": [],
    }
    try:
        a1_adapter_path = _download_adapter("A1", args.smoke_seed, GATE_PROBE_FRAC)
    except Exception as e:
        _write_sentinel(
            "offdiag_no_logprob_shift",
            f"Couldn't download A1 frac={GATE_PROBE_FRAC} for Gate 3 off-diag probe: {e}",
        )
        return 4
    lora_a1 = LoRARequest(
        lora_name=f"A1_gate3_{_frac_tag(GATE_PROBE_FRAC)}",
        lora_int_id=round(GATE_PROBE_FRAC * 100) + ord("A") * 10000,
        lora_path=a1_adapter_path,
    )

    probe_q_off = held_out_probe[0]
    n_passed_offdiag = 0
    for target_cid in OFFDIAG_MINI_GRID:
        try:
            base_logp, _R_used = _off_diag_probe_logp(
                "A1", target_cid, lora_request=None, probe_q=probe_q_off
            )
            trained_logp, _ = _off_diag_probe_logp(
                "A1", target_cid, lora_request=lora_a1, probe_q=probe_q_off
            )
        except Exception as e:
            offdiag_payload["cells"].append(
                {
                    "target": target_cid,
                    "error": f"off-diag probe failed: {e}",
                    "passes_offdiag_cell": False,
                }
            )
            continue
        delta = trained_logp - base_logp
        passed = delta >= GATE3_OFFDIAG_LOGPROB_SHIFT_MIN_NATS
        offdiag_payload["cells"].append(
            {
                "target": target_cid,
                "probe_q": probe_q_off,
                "base_logp_marker": base_logp,
                "trained_logp_marker": trained_logp,
                "delta_nats": delta,
                "min_shift_nats": GATE3_OFFDIAG_LOGPROB_SHIFT_MIN_NATS,
                "passes_offdiag_cell": passed,
            }
        )
        if passed:
            n_passed_offdiag += 1
        logger.info(
            "Gate 3 off-diag A1→%s frac=%.2f: base=%.4f trained=%.4f Δ=%.4f pass=%s",
            target_cid,
            GATE_PROBE_FRAC,
            base_logp,
            trained_logp,
            delta,
            passed,
        )

    offdiag_payload["n_cells_passed"] = n_passed_offdiag
    offdiag_payload["passes_gate3"] = n_passed_offdiag >= GATE3_OFFDIAG_CELLS_REQUIRED
    (SMOKE_LOG_DIR / "offdiag_logprob_shift.json").write_text(json.dumps(offdiag_payload, indent=2))
    logger.info(
        "Gate 3 summary: %d / %d cells passed (required ≥ %d)",
        n_passed_offdiag,
        len(OFFDIAG_MINI_GRID),
        GATE3_OFFDIAG_CELLS_REQUIRED,
    )

    if not offdiag_payload["passes_gate3"]:
        _write_sentinel(
            "offdiag_no_logprob_shift",
            (
                f"Gate 3 FAIL: only {n_passed_offdiag} of {len(OFFDIAG_MINI_GRID)} "
                f"off-diag cells had Δlogp_off ≥ {GATE3_OFFDIAG_LOGPROB_SHIFT_MIN_NATS} "
                f"nats at A1's frac={GATE_PROBE_FRAC} adapter "
                f"(required ≥ {GATE3_OFFDIAG_CELLS_REQUIRED}). "
                "See offdiag_logprob_shift.json for per-cell deltas."
            ),
            extra=offdiag_payload,
        )
        return 4

    # ── Gate 4: EOS-gradient check at A1's frac=3.00 adapter, A1→B1 cell ──
    eos_gradient_payload: dict = {"probe_frac": GATE_PROBE_FRAC}
    # Reuse the A1 frac=3.00 LoRA path (already downloaded for Gate 3).
    lora_eos = LoRARequest(
        lora_name=f"A1_eosgrad_{_frac_tag(GATE_PROBE_FRAC)}",
        lora_int_id=99999,
        lora_path=a1_adapter_path,
    )
    probe_q = held_out_probe[0]
    cond_target = CONDITIONS_BY_ID["B1"]
    prompt_text = build_prompt_for_condition(cond_target, probe_q, tokenizer, class_d_rewrites)
    R_text = _resolve_R_for("B1", prompt_text, probe_q)
    base_logp = _post_response_slot_logprob(
        llm, sp_logprob, tokenizer, prompt_text, R_text, lora_request=None
    )
    trained_logp = _post_response_slot_logprob(
        llm, sp_logprob, tokenizer, prompt_text, R_text, lora_request=lora_eos
    )
    delta_nats = trained_logp - base_logp
    moved_down = delta_nats <= -EOS_GRADIENT_MIN_NATS
    eos_gradient_payload.update(
        {
            "target": "B1",
            "probe_q": probe_q,
            "base_logp_marker": base_logp,
            "trained_logp_marker": trained_logp,
            "delta_nats": delta_nats,
            "min_drop_required_nats": EOS_GRADIENT_MIN_NATS,
            "moved_down": moved_down,
        }
    )
    (SMOKE_LOG_DIR / "eos_gradient.json").write_text(json.dumps(eos_gradient_payload, indent=2))
    logger.info(
        "EOS-gradient probe (frac=%.2f, A1→B1): base=%.4f trained=%.4f delta=%.4f (required ≤ -%s)",
        GATE_PROBE_FRAC,
        base_logp,
        trained_logp,
        delta_nats,
        EOS_GRADIENT_MIN_NATS,
    )
    if not moved_down:
        _write_sentinel(
            "eos_gradient_inactive",
            "Trained log P(' ※') at A1→B1 post-response slot did NOT drop ≥ "
            f"{EOS_GRADIENT_MIN_NATS} nats vs base "
            f"(delta = {delta_nats:.4f}).",
            extra=eos_gradient_payload,
        )
        return 6

    # ── PASS ──
    logger.info(
        "Smoke gates PASS (v3): Gate 1 + Gate 2' (≥1 cell) + Gate 3 (%d/%d) + Gate 4 (EOS). "
        "Production trains all 6 fracs; headline frac picked post-hoc by phase5_analyze.",
        n_passed_offdiag,
        len(OFFDIAG_MINI_GRID),
    )

    del llm
    from issue404_common import kill_vllm_workers

    kill_vllm_workers(logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
