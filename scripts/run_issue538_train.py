"""Issue #538 unified smoke + sweep training dispatcher.

Plan §4 Step 3 (Phase A smoke) + §4 Step 5 (Phase B full sweep). Unified per
SKILL.md Step 6d.0: smoke IS sweep with `--phase smoke --cells 1 --seeds 1`
(same code path, same subprocess shape, same env injection, same WandB
logging surface, same auto-upload to HF, same band-stop). PASS_UNIFIED.

Inherited from ``scripts/run_issue527_train.py`` with imports + namespace
strings switched to ``issue_538`` and the new band/epoch defaults inherited
from ``experiments.issue_538`` constants (band [14,20] nat, epochs cap 24).
Read paths (``--pair-selection``, ``--r-persona-dir``) still default to the
parent's ``issue_527/`` namespace because R_persona + pair_selection.json are
inherited verbatim from #527 (plan §4 Inputs). Write paths (``--out-root``,
HF adapter prefix, WandB run name) all become ``issue_538``.

Task #538 round-2 fix (21:27Z ``epm:concern-raised``): the 4-persona
contrastive negative panel is resolved PER-PAIR via
``negative_panel_for_pair``. For pair-1 (florist x medical_doctor) the
panel is unchanged vs #527 (no overlap; pair-1 training mixes stay
byte-identical, proven by the preflight hash gate). For pair-2
(librarian x police_officer) the overlapping ``librarian`` slot is
swapped for ``kindergarten_teacher`` so the same persona is no longer
trained with positive AND negative marker objectives 4:1 in the same
cell. The Phase A bystander headroom probe + smoke-gate verdict read
this per-pair panel directly.

Per (pair, arm, seed) cell:
  1. Loads pair-selection.json + persona_bank.json + R_persona/.
  2. Builds the per-arm training JSONL via ``build_arm_rows`` (strict 1:1
     positives-to-total-negatives, 4-persona contrastive panel).
  3. Calls ``train_lora()`` with the canonical band-stop recipe
     (rsLoRA r=16 / α=32 / attn-only / lr=5e-6 / 24-epoch cap /
     marker_band_stop=True / band [14,20] nat). Inherits
     `MarkerBandStopCallback` wiring from `train_lora`.
  4. For Phase A (smoke) cells only: after training, runs a bystander
     argmax-rate + log P(marker) probe on the trained model under each of
     the 4 negative personas, persisting to
     ``eval_results/issue_538/anchor_smoke/<pair>__<arm>__seed<S>.json``.
  5. Auto-uploads the adapter to HF
     (``adapters/issue_538/<pair>__<arm>__seed<S>``) per upload-policy.

The end-of-training Phase A summary JSON is what the smoke-gate verdict
reads. Plan §4 Step 3 GATE PASS = ≥2/3 cells satisfy BOTH source-band ∈
[14,20] AND all 4 negative-panel personas have argmax-rate < 0.92 (the
bystander-saturation diagnostic). NO autonomous lr=1e-5 retry on Phase A
FAIL — the recipe forbids raising lr past 5e-6 at the new dial (plan §4
Pipeline Delta).

CLI (smoke ≡ sweep with one cell):
    # Phase A smoke (3 cells on the first pair, seed 42):
    uv run python scripts/run_issue538_train.py --phase smoke --pair-index 0 --seed 42

    # Phase B sweep cell (1 cell):
    uv run python scripts/run_issue538_train.py --phase sweep \\
        --pair-index 0 --arm A_only --seed 42 --gpu-id 0

    # CPU smoke (dispatcher dry-run; no GPU forward):
    uv run python scripts/run_issue538_train.py --phase smoke \\
        --pair-index 0 --seed 42 --dispatcher-dry-run --allow-smoke-fallback
"""

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_538 import (
    BASE_MODEL,
    HF_ADAPTER_PATH_PREFIX,
    HF_MODEL_REPO,
    IM_END_ID,
    MARKER_ID,
    MARKER_TEXT,
    RECIPE_BAND_HIGH_NATS,
    RECIPE_BAND_LOW_NATS,
    RECIPE_EPOCHS_CAP,
    RECIPE_GRAD_ACCUM,
    RECIPE_LORA_ALPHA,
    RECIPE_LORA_DROPOUT,
    RECIPE_LORA_R,
    RECIPE_LORA_TARGETS,
    RECIPE_LR_PRIMARY,
    RECIPE_MAX_LENGTH,
    RECIPE_PER_DEVICE_BATCH,
    RECIPE_WARMUP_RATIO,
    negative_panel_for_pair,
)
from explore_persona_space.experiments.issue_538.data_build import (
    build_arm_rows,
    write_rows_jsonl,
)
from explore_persona_space.experiments.issue_538.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.experiments.issue_538.question_pool import load_question_pool

log = logging.getLogger("issue_538.train")

ARMS = ("A_only", "B_only", "joint")


def _make_band_stop_recorder(*, output_dir: Path, low_nats: float, high_nats: float):
    """Concretely subclass TrainerCallback inside a closure so the import is
    deferred to runtime (keeps the script importable in test environments
    without transformers).

    Subscribes to ``TrainerCallback.on_log`` to receive the
    ``MarkerBandStopCallback``'s per-step source-delta + band-stop event,
    which round-1 review surfaced was the broken link (the callback used
    to call ``wandb.log(...)`` directly, never feeding the trainer's
    ``logs`` dict; round-2 extends the callback with its own ``on_log``
    hook in ``eval/callbacks.py`` so sibling subscribers like this
    recorder actually see the keys). The recorder accepts BOTH the
    canonical ``MarkerBandStopCallback`` default ``log_prefix="marker"``
    keys AND the legacy ``"marker_band_stop"`` namespace for resilience
    against future log-prefix changes — the source of truth on which
    prefix train_lora attaches is the callback construction in
    ``train/sft.py::_maybe_attach_marker_band_stop`` (currently the
    default ``"marker"``).
    """
    from transformers import TrainerCallback

    _CANDIDATE_PREFIXES = ("marker", "marker_band_stop")

    class _Recorder(TrainerCallback):
        def __init__(self):
            self.fired: bool = False
            self.final_delta_nats: float | None = None
            self.fired_step: int | None = None
            self._last_delta: float | None = None
            self._last_step: int | None = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Track the most recent source_delta_nats from MarkerBandStopCallback."""
            if not logs:
                return
            for prefix in _CANDIDATE_PREFIXES:
                for key in (
                    f"{prefix}/source_delta_nats",
                    f"{prefix}/band_stop_delta_nats",
                ):
                    if key in logs:
                        self._last_delta = float(logs[key])
                        self._last_step = int(state.global_step)
                # band_stop_step key is emitted only on the firing step.
                step_key = f"{prefix}/band_stop_step"
                if step_key in logs:
                    self.fired = True
                    self.fired_step = int(logs[step_key])
                    self.final_delta_nats = float(
                        logs.get(f"{prefix}/band_stop_delta_nats", self._last_delta or 0.0)
                    )

        def on_train_end(self, args, state, control, **kwargs):
            payload = {
                "fired": self.fired,
                "step": self.fired_step,
                "final_delta_nats": self.final_delta_nats if self.fired else self._last_delta,
                "band_low_nats": low_nats,
                "band_high_nats": high_nats,
            }
            out = Path(output_dir) / "marker_band_stop_result.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
            log.info(
                "BandStopRecorder wrote %s (fired=%s, delta=%s)",
                out,
                payload["fired"],
                payload["final_delta_nats"],
            )

    return _Recorder()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_pair_selection(path: Path) -> list[dict]:
    """Return the list of picked pairs from pair_selection.json."""
    if not path.is_file():
        raise FileNotFoundError(
            f"pair_selection.json missing at {path}; run "
            "scripts/run_issue527_pair_selection.py first."
        )
    payload = json.loads(path.read_text())
    # READ schema — pair_selection.json is inherited verbatim from #527, so
    # the schema string stays issue_527_*.
    if payload.get("schema_version") != "issue_527_pair_selection_v1":
        raise AssertionError(
            f"{path} schema_version mismatch (got {payload.get('schema_version')!r})"
        )
    pairs = payload.get("picked_pairs", [])
    if not isinstance(pairs, list) or len(pairs) < 1:
        raise AssertionError(f"{path} has no picked_pairs")
    return pairs


def _load_r_persona(out_dir: Path) -> dict[str, dict[str, str]]:
    """Load every persona's R_persona JSON under ``out_dir``.

    Returns ``{persona: {question: response}}``.
    """
    if not out_dir.is_dir():
        raise FileNotFoundError(
            f"R_persona dir missing at {out_dir}; inherit verbatim from #527 "
            f"(eval_results/issue_527/R_persona/ or HF dataset issue_527/R_persona/)."
        )
    out: dict[str, dict[str, str]] = {}
    for json_path in sorted(out_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        # READ schema — R_persona JSONs are inherited byte-identical from #527.
        if payload.get("schema_version") != "issue_527_R_persona_v1":
            raise AssertionError(f"{json_path} R_persona schema mismatch")
        out[payload["persona"]] = payload["responses"]
    return out


def _build_smoke_probe_rows(
    *,
    panel: tuple[str, ...],
    persona_bank: dict[str, str],
    questions: list[str],
    r_persona: dict[str, dict[str, str]],
    tokenizer,
    n_probe_per_persona: int = 8,
) -> dict[str, list[tuple[str, list[int], int]]]:
    """Pre-build per-bystander probe inputs (full_ids + post-response slot).

    For each persona in the PER-PAIR negative ``panel`` (task #538 fix —
    pair-2's panel swaps the overlapping ``librarian`` slot for
    ``kindergarten_teacher``), sample ``n_probe_per_persona`` questions and
    tokenize ``T_persona(q) + R_persona(q)``. Return
    ``{persona: [(question, full_ids, post_response_slot), ...]}``.

    Threading the per-pair ``panel`` here is load-bearing: for pair-2
    ``librarian`` is now a SOURCE, not a bystander, so probing it as a
    bystander would mis-attribute pair-2's source firings to bystander
    saturation. The bystander-resolution gate in ``_smoke_summarize`` reads
    THIS dict, so it must reflect the actual trained negatives.
    """
    from explore_persona_space.experiments.issue_538.shift_extract import (
        _resolve_post_response_slot,
    )

    out: dict[str, list[tuple[str, list[int], int]]] = {}
    rng = np.random.default_rng(0)
    for persona in panel:
        if persona not in r_persona:
            raise AssertionError(
                f"R_persona missing for negative persona {persona!r}; regenerate R before smoke."
            )
        n = min(n_probe_per_persona, len(questions))
        idxs = rng.choice(len(questions), size=n, replace=False)
        rows: list[tuple[str, list[int], int]] = []
        for i in idxs:
            q = questions[int(i)]
            if q not in r_persona[persona]:
                raise AssertionError(f"R_persona[{persona!r}] missing q={q!r}")
            messages = [
                {"role": "system", "content": persona_bank[persona]},
                {"role": "user", "content": q},
                {"role": "assistant", "content": r_persona[persona][q]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(text, add_special_tokens=False)
            slot = _resolve_post_response_slot(tokenizer, messages[:2], full_ids)
            rows.append((q, full_ids, slot))
        out[persona] = rows
    return out


def _bystander_headroom_probe(
    *,
    base_model_path: str,
    adapter_dir: str,
    probe_rows: dict[str, list[tuple[str, list[int], int]]],
    device: str = "cuda:0",
) -> dict[str, dict[str, float]]:
    """Forward-only probe: per bystander, argmax-rate at slot + mean Δ log P(marker).

    Returns ``{bystander: {"argmax_rate": float, "delta_logp_mean": float,
    "logp_trained_mean": float, "logp_base_mean": float}}``.
    """
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    log.info("Loading base model (%s) for bystander-headroom probe", base_model_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).eval()

    log.info("Loading trained adapter from %s", adapter_dir)
    trained = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    trained = PeftModel.from_pretrained(trained, adapter_dir).eval()

    out: dict[str, dict[str, float]] = {}
    for persona, rows in probe_rows.items():
        argmax_hits = 0
        delta_acc = 0.0
        logp_trained_acc = 0.0
        logp_base_acc = 0.0
        for _q, full_ids, slot in rows:
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out_base = base(ids)
                out_tr = trained(ids)
            lp_base = F.log_softmax(out_base.logits[0, slot - 1].float(), dim=-1)
            lp_tr = F.log_softmax(out_tr.logits[0, slot - 1].float(), dim=-1)
            if int(out_tr.logits[0, slot - 1].argmax().item()) == MARKER_ID:
                argmax_hits += 1
            delta_acc += float((lp_tr[MARKER_ID] - lp_base[MARKER_ID]).item())
            logp_trained_acc += float(lp_tr[MARKER_ID].item())
            logp_base_acc += float(lp_base[MARKER_ID].item())
        n = len(rows)
        out[persona] = {
            "argmax_rate": argmax_hits / n,
            "delta_logp_mean": delta_acc / n,
            "logp_trained_mean": logp_trained_acc / n,
            "logp_base_mean": logp_base_acc / n,
        }

    # Free GPU; the next cell needs a fresh allocation.
    del base
    del trained
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return out


def _train_one_cell(
    *,
    pair: dict,
    arm: str,
    seed: int,
    persona_bank: dict[str, str],
    questions: list[str],
    r_persona: dict[str, dict[str, str]],
    lr: float,
    epochs_cap: int,
    gpu_id: int,
    output_root: Path,
    hf_path_in_repo: str,
    band_stop_low: float,
    band_stop_high: float,
    dispatcher_dry_run: bool,
) -> tuple[str, float]:
    """Train one (pair, arm, seed) cell. Returns (output_dir, final_train_loss)."""
    from transformers import AutoTokenizer

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    pair_a = pair["name_a"]
    pair_b = pair["name_b"]
    pair_id = pair["pair_id"]

    cell_slug = f"{pair_id}__{arm}__seed{seed}"
    cell_dir = output_root / "adapters" / cell_slug
    cell_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_root / "training_mixes"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / f"{cell_slug}.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Marker token preflight (per marker-leakage rule; thread with shlex.quote
    # for any shell layer).
    marker_quoted = shlex.quote(MARKER_TEXT)
    log.info("Marker token (shlex-quoted) = %s", marker_quoted)
    encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [MARKER_ID]:
        raise AssertionError(
            f"Marker token drift inside _train_one_cell: {encoded} != [{MARKER_ID}]"
        )
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end != IM_END_ID:
        raise AssertionError(f"<|im_end|> drift: {im_end} != {IM_END_ID}")

    log.info(
        "Building training rows: pair=%s arm=%s seed=%d (pair_a=%s, pair_b=%s)",
        pair_id,
        arm,
        seed,
        pair_a,
        pair_b,
    )
    rows = build_arm_rows(
        arm=arm,
        pair_a=pair_a,
        pair_b=pair_b,
        persona_bank=persona_bank,
        questions=questions,
        r_persona=r_persona,
        tokenizer=tokenizer,
        seed=seed,
    )
    log.info("Writing %d training rows to %s", len(rows), train_path)
    write_rows_jsonl(rows, train_path)

    if dispatcher_dry_run:
        log.warning(
            "--dispatcher-dry-run: SKIPPING train_lora() call. "
            "All pre-GPU plumbing exercised: tokenizer load + marker assert + "
            "rows build + JSONL write + adapter dir create."
        )
        return str(cell_dir), 0.0

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=epochs_cap,
        lr=lr,
        lora_r=RECIPE_LORA_R,
        lora_alpha=RECIPE_LORA_ALPHA,
        lora_dropout=RECIPE_LORA_DROPOUT,
        lora_targets=list(RECIPE_LORA_TARGETS),
        batch_size=RECIPE_PER_DEVICE_BATCH,
        grad_accum=RECIPE_GRAD_ACCUM,
        max_length=RECIPE_MAX_LENGTH,
        warmup_ratio=RECIPE_WARMUP_RATIO,
        seed=seed,
        run_name=f"issue_538_{cell_slug}",
        report_to="wandb",
        save_strategy="no",
        save_total_limit=1,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=IM_END_ID,
        marker_band_stop=True,
        marker_band_low_nats=band_stop_low,
        marker_band_high_nats=band_stop_high,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=hf_path_in_repo,
    )

    # MooseFS quota safety per CLAUDE.md gotchas.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    # Delete-after-eval adapter-persist recipe (upload-policy.md): so the
    # caller's `rm -rf <output_dir>` after training is safe.
    os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = HF_MODEL_REPO
    os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = hf_path_in_repo

    # Attach the band-stop recorder so the smoke gate + analyzer can read
    # the band-fire event from disk (the canonical MarkerBandStopCallback
    # only emits to WandB — see _make_band_stop_recorder above).
    recorder = _make_band_stop_recorder(
        output_dir=Path(cell_dir),
        low_nats=band_stop_low,
        high_nats=band_stop_high,
    )

    output_dir, train_loss = train_lora(
        BASE_MODEL, str(train_path), str(cell_dir), cfg=cfg, callbacks=[recorder]
    )
    log.info(
        "TRAIN DONE cell=%s loss=%.4f -> %s (uploaded to %s)",
        cell_slug,
        train_loss,
        output_dir,
        f"{HF_MODEL_REPO}/{hf_path_in_repo}",
    )
    return output_dir, train_loss


def _smoke_summarize(
    *,
    pair_id: str,
    seed: int,
    smoke_results: list[dict],
    band_low: float,
    band_high: float,
    bystander_argmax_max: float = 0.92,
) -> dict:
    """Apply the Phase A smoke-gate verdict per plan §7 (issue_538).

    Plan §7 PASS = ≥2/3 cells satisfy BOTH
      (i) source-band ∈ [band_low, band_high] (the band-stop fired in [14,20])
      (ii) ALL 4 negative-panel personas have argmax-rate < 0.92 (the
           bystander-saturation diagnostic from the marker-training-recipe
           rule — gate ALL bystanders, not just one). The 4 personas are the
           PER-PAIR resolved panel from ``negative_panel_for_pair`` (task
           #538 fix: pair-2 swaps the overlapping ``librarian`` slot for
           ``kindergarten_teacher`` so the gate reads the actual trained
           negatives).
    log P(marker) on bystanders is reported but NOT gated; argmax-rate is
    the canonical saturation diagnostic at the new dial.

    The source-band signal (i) is determined by inspecting the result's
    ``band_stop_fired`` flag (true if the band-stop callback fired in
    [low, high]) AND ``final_source_delta_nats`` ∈ [low, high].
    """
    per_cell_pass: list[bool] = []
    for r in smoke_results:
        source_ok = (
            r.get("band_stop_fired") is True
            and band_low <= r.get("final_source_delta_nats", -100.0) <= band_high
        )
        bys = r.get("bystander_probe", {})
        # Plan §7: ALL 4 bystanders must stay below the argmax-rate ceiling.
        if not bys:
            bys_ok = False
        else:
            bys_ok = all(m["argmax_rate"] < bystander_argmax_max for m in bys.values())
        per_cell_pass.append(bool(source_ok and bys_ok))

    pass_count = sum(per_cell_pass)
    verdict = "PASS" if pass_count >= 2 else "FAIL"
    return {
        "pair_id": pair_id,
        "seed": seed,
        "per_cell_pass": per_cell_pass,
        "pass_count": pass_count,
        "verdict": verdict,
        "band_low_nats": band_low,
        "band_high_nats": band_high,
        "bystander_argmax_max": bystander_argmax_max,
    }


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=["smoke", "sweep"])
    ap.add_argument(
        "--pair-selection",
        # INHERITED READ from #527 — plan §4 Inputs / Branch base.
        default="eval_results/issue_527/pair_selection.json",
    )
    ap.add_argument(
        "--r-persona-dir",
        # INHERITED READ from #527 (R_persona/ is byte-identical; preflight
        # hash-gates against HF copy if local dir missing).
        default="eval_results/issue_527/R_persona",
    )
    ap.add_argument(
        "--out-root",
        # NEW WRITE namespace per plan §4 Outputs.
        default="eval_results/issue_538",
    )
    ap.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="Index into pair_selection.picked_pairs (smoke uses pairs[pair_index] only).",
    )
    ap.add_argument(
        "--pair-indices",
        type=int,
        nargs="+",
        default=None,
        help="Sweep only — multiple pair indices to run. Defaults to all picked pairs.",
    )
    ap.add_argument(
        "--arm",
        choices=ARMS,
        default=None,
        help="If set, run only this arm. Default: all 3 arms.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, run only this seed.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Sweep only — multiple seeds. Default: (42, 137, 256).",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--lr",
        type=float,
        default=RECIPE_LR_PRIMARY,
        help=(
            f"Default {RECIPE_LR_PRIMARY} (band-stop recipe primary). NO autonomous lr "
            "retry in #538 — the marker-training recipe forbids lr>5e-6 at the new band."
        ),
    )
    ap.add_argument("--epochs", type=int, default=RECIPE_EPOCHS_CAP)
    ap.add_argument("--band-low-nats", type=float, default=RECIPE_BAND_LOW_NATS)
    ap.add_argument("--band-high-nats", type=float, default=RECIPE_BAND_HIGH_NATS)
    ap.add_argument(
        "--n-questions",
        type=int,
        default=400,
        help="Question pool size (default 400, plan §4 Step 2).",
    )
    ap.add_argument(
        "--allow-smoke-fallback",
        action="store_true",
        help="Permit the 20-question smoke fallback (smoke only).",
    )
    ap.add_argument(
        "--dispatcher-dry-run",
        action="store_true",
        help=(
            "Stub out train_lora() and the bystander probe — exercises the "
            "pre-GPU plumbing (data load, marker assert, rows build, JSONL "
            "write, output-dir create, sentinel write) without requiring CUDA. "
            "GPU-bound phases land here for the implementer smoke."
        ),
    )
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading persona-bank + pair-selection + R_persona")
    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)
    pairs = _load_pair_selection(Path(args.pair_selection))
    r_persona = _load_r_persona(Path(args.r_persona_dir))
    log.info("Loaded %d pairs, %d personas of R_persona", len(pairs), len(r_persona))

    questions = load_question_pool(
        n_required=args.n_questions, allow_smoke_fallback=args.allow_smoke_fallback
    )

    # Resolve which (pair, arm, seed) cells to run for this phase.
    if args.phase == "smoke":
        # Plan §4 Step 3: 3 cells × 1 seed on the FIRST picked orthogonal pair.
        pair_idxs = [args.pair_index]
        seeds = [args.seed if args.seed is not None else 42]
        arms = list(ARMS) if args.arm is None else [args.arm]
        log.info(
            "Phase A smoke: pair=%s (idx=%d), seeds=%s, arms=%s",
            pairs[pair_idxs[0]]["pair_id"],
            pair_idxs[0],
            seeds,
            arms,
        )
    else:
        pair_idxs = args.pair_indices if args.pair_indices is not None else list(range(len(pairs)))
        seeds = (
            args.seeds
            if args.seeds is not None
            else [42, 137, 256]
            if args.seed is None
            else [args.seed]
        )
        arms = list(ARMS) if args.arm is None else [args.arm]
        log.info(
            "Phase B sweep: pair_indices=%s, seeds=%s, arms=%s",
            pair_idxs,
            seeds,
            arms,
        )

    git_commit = _git_commit()
    timestamp = _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds")

    # Smoke-phase probe rows are built PER-PAIR (task #538 fix). The probe
    # iterates the per-pair negative panel (which may swap one base-panel
    # slot for ``kindergarten_teacher`` when a source overlaps the panel),
    # so the bystander-resolution gate in ``_smoke_summarize`` reads the
    # ACTUAL trained negatives. Tokenizer is loaded once outside the loop.
    smoke_tokenizer = None
    if args.phase == "smoke" and not args.dispatcher_dry_run:
        from transformers import AutoTokenizer

        smoke_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    smoke_results: list[dict] = []

    for pair_idx in pair_idxs:
        pair = pairs[pair_idx]
        # Resolve the per-pair negative panel ONCE per pair — the panel is
        # constant across this pair's arms+seeds (it depends only on the
        # source pair). build_arm_rows resolves it independently per cell;
        # this resolves it here for smoke-probe + diagnostic logging.
        pair_panel = negative_panel_for_pair(pair["name_a"], pair["name_b"])
        log.info(
            "Per-pair negative panel for %s: %s",
            pair["pair_id"],
            list(pair_panel),
        )
        smoke_probe_rows = None
        if args.phase == "smoke" and not args.dispatcher_dry_run:
            assert smoke_tokenizer is not None  # invariant: built above when entering smoke
            smoke_probe_rows = _build_smoke_probe_rows(
                panel=pair_panel,
                persona_bank=persona_bank,
                questions=questions,
                r_persona=r_persona,
                tokenizer=smoke_tokenizer,
                n_probe_per_persona=8,
            )
        for seed in seeds:
            for arm in arms:
                cell_slug = f"{pair['pair_id']}__{arm}__seed{seed}"
                hf_subfolder = f"{HF_ADAPTER_PATH_PREFIX}/{cell_slug}"
                log.info(
                    "[phase=train] cell=%s lr=%g epochs_cap=%d", cell_slug, args.lr, args.epochs
                )

                try:
                    out_dir, loss = _train_one_cell(
                        pair=pair,
                        arm=arm,
                        seed=seed,
                        persona_bank=persona_bank,
                        questions=questions,
                        r_persona=r_persona,
                        lr=args.lr,
                        epochs_cap=args.epochs,
                        gpu_id=args.gpu_id,
                        output_root=out_root,
                        hf_path_in_repo=hf_subfolder,
                        band_stop_low=args.band_low_nats,
                        band_stop_high=args.band_high_nats,
                        dispatcher_dry_run=args.dispatcher_dry_run,
                    )
                except Exception as e:
                    # Per CLAUDE.md fail-fast: log + re-raise so the launcher's
                    # set -e aborts the pipeline. No silent skip.
                    log.exception("cell=%s training crashed: %s", cell_slug, e)
                    raise

                cell_result: dict = {
                    "cell_slug": cell_slug,
                    "pair_id": pair["pair_id"],
                    "arm": arm,
                    "seed": seed,
                    "lr": args.lr,
                    "epochs_cap": args.epochs,
                    "band_low_nats": args.band_low_nats,
                    "band_high_nats": args.band_high_nats,
                    "output_dir": out_dir,
                    "hf_subfolder": hf_subfolder,
                    "final_train_loss": loss,
                    "git_commit": git_commit,
                    "timestamp_utc": timestamp,
                    "base_model": BASE_MODEL,
                    # Per-pair negative panel (task #538 fix). For pair-1
                    # this matches NEGATIVE_PANEL_4 byte-for-byte; for
                    # pair-2 ``librarian`` is swapped for
                    # ``kindergarten_teacher``.
                    "negative_panel": list(pair_panel),
                }

                # The band-stop fired flag + final source-delta are exposed via
                # the `MarkerBandStopCallback`'s WandB metric. We read them back
                # via the most-recent run-result JSON the callback dropped in
                # the adapter dir, falling back to None if absent (the trainer
                # always emits a `marker_band_stop_result.json` when the
                # callback attaches — see train/sft.py).
                cb_result_path = Path(out_dir) / "marker_band_stop_result.json"
                if cb_result_path.is_file():
                    cb = json.loads(cb_result_path.read_text())
                    cell_result["band_stop_fired"] = cb.get("fired", False)
                    cell_result["final_source_delta_nats"] = cb.get("final_delta_nats")
                    cell_result["band_stop_step"] = cb.get("step")
                else:
                    cell_result["band_stop_fired"] = None
                    cell_result["final_source_delta_nats"] = None
                    cell_result["band_stop_step"] = None

                if args.phase == "smoke" and smoke_probe_rows is not None:
                    log.info("[phase=smoke_probe] cell=%s — bystander headroom probe", cell_slug)
                    bys = _bystander_headroom_probe(
                        base_model_path=BASE_MODEL,
                        adapter_dir=out_dir,
                        probe_rows=smoke_probe_rows,
                    )
                    cell_result["bystander_probe"] = bys

                # Per-cell checkpoint — write IMMEDIATELY (per CLAUDE.md
                # "Checkpoint per phase; never accumulate-in-memory").
                phase_dir = out_root / ("anchor_smoke" if args.phase == "smoke" else "sweep")
                phase_dir.mkdir(parents=True, exist_ok=True)
                cell_path = phase_dir / f"{cell_slug}.json"
                cell_path.write_text(json.dumps(cell_result, indent=2))
                log.info("Wrote %s", cell_path)
                smoke_results.append(cell_result)

    # Phase A: summarize + apply gate.
    if args.phase == "smoke":
        summary = _smoke_summarize(
            pair_id=pairs[pair_idxs[0]]["pair_id"],
            seed=seeds[0],
            smoke_results=smoke_results,
            band_low=args.band_low_nats,
            band_high=args.band_high_nats,
        )
        summary_path = out_root / "anchor_smoke" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Phase A smoke verdict=%s -> %s", summary["verdict"], summary_path)
        if summary["verdict"] != "PASS" and not args.dispatcher_dry_run:
            # Surface non-zero exit so the launcher's set -e aborts.
            # Plan §4 Pipeline Delta: NO autonomous lr=1e-5 retry at the new
            # band — the recipe forbids raising lr past 5e-6. The pipeline
            # immediately posts epm:failure v1
            # reason=anchor_floor_or_ceiling_at_new_band and stops.
            log.error(
                "Phase A smoke FAILED (pass_count=%d/3) at band [%g, %g] nat. See %s. "
                "NO autonomous lr-retry path at the new band (recipe forbids lr>5e-6). "
                "Pipeline will post epm:failure v1 "
                "reason=anchor_floor_or_ceiling_at_new_band.",
                summary["pass_count"],
                args.band_low_nats,
                args.band_high_nats,
                summary_path,
            )
            return 2

    log.info("[phase=done] dispatcher exit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
