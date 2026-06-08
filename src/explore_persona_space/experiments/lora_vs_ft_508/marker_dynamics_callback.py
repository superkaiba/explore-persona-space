# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #508 — in-training MarkerDynamicsCallback (plan §4.2 MF2).

Fires every ``DYNAMICS_CADENCE_STEPS`` training steps on a fixed 20-probe set
(1 source × 5 q + 3 bystanders × 5 q). At each fire:
  1. Generate greedy R under the trainer's CURRENT model state for each probe.
  2. Score trained log P(` ※`) at the post-R slot on each probe's current-step R.
  3. Score base log P(` ※`) on the SAME current-step R (base model loaded lazily
     at callback init on CPU, sliced onto GPU per call).
  4. Log ``source_ΔG``, ``bystander_mean_ΔG``, ``source_emission_rate``,
     ``bystander_mean_emission_rate``, plus the 4 per-bystander-persona
     breakdowns, keyed by ``global_step``, to WandB.

Required by ``.claude/rules/marker-leakage-measurement.md`` § "Track log-prob
DYNAMICS, not just the endpoint". Same callback fires on BOTH the LoRA and the
full-FT arm so the trajectories are directly comparable.

Trajectory figures are first-class per plan §4.7.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from transformers import TrainerCallback

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
)
from explore_persona_space.experiments.lora_vs_ft_508 import (
    DYNAMICS_BYSTANDER_PERSONAS,
    DYNAMICS_CADENCE_STEPS,
    DYNAMICS_PROBE_QUESTIONS_PER_PERSONA,
    DYNAMICS_PROBES_PATH,
    SOURCE_PERSONA,
)

log = logging.getLogger("issue_508.marker_dynamics_callback")


def build_dynamics_probes(
    persona_bank: dict[str, str],
    q_eval: list[str],
    *,
    seed: int = 42,
    n_q_per_persona: int = DYNAMICS_PROBE_QUESTIONS_PER_PERSONA,
) -> dict:
    """Build the deterministic 20-probe dynamics-callback probe set.

    Returns a dict ``{persona: {"role": "source"|"bystander", "system": <prompt>,
    "questions": [<n_q>]}}``. The 5 questions per persona are SHA-derived from
    ``seed`` (a strict subset of the 20-question held-out eval pool), so the
    dynamics probes are NOT extra eval surface — they live INSIDE the headline
    DV's (persona × question) grid.
    """
    import hashlib

    if SOURCE_PERSONA not in persona_bank:
        raise KeyError(f"Source persona {SOURCE_PERSONA!r} missing from persona_bank")
    for p in DYNAMICS_BYSTANDER_PERSONAS:
        if p not in persona_bank:
            raise KeyError(f"Dynamics bystander {p!r} missing from persona_bank")

    def pick(persona: str) -> list[str]:
        # SHA-derived deterministic question pick (seed-salted by persona name).
        scored = sorted(
            range(len(q_eval)),
            key=lambda i: hashlib.sha256(f"{seed}|{persona}|{i}".encode()).hexdigest(),
        )
        return [q_eval[i] for i in scored[:n_q_per_persona]]

    probes: dict[str, dict] = {
        SOURCE_PERSONA: {
            "role": "source",
            "system": persona_bank[SOURCE_PERSONA],
            "questions": pick(SOURCE_PERSONA),
        }
    }
    for p in DYNAMICS_BYSTANDER_PERSONAS:
        probes[p] = {
            "role": "bystander",
            "system": persona_bank[p],
            "questions": pick(p),
        }
    n_total = sum(len(v["questions"]) for v in probes.values())
    expected = (1 + len(DYNAMICS_BYSTANDER_PERSONAS)) * n_q_per_persona
    if n_total != expected:
        raise AssertionError(f"build_dynamics_probes: expected {expected} probes, got {n_total}")
    return probes


def save_dynamics_probes(probes: dict, path: str | Path = DYNAMICS_PROBES_PATH) -> Path:
    """Write the dynamics probe set to disk (JSON)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probes, indent=2, ensure_ascii=False))
    log.info("[dynamics] wrote %d-persona probe set → %s", len(probes), out)
    return out


def load_dynamics_probes(path: str | Path = DYNAMICS_PROBES_PATH) -> dict:
    """Read the dynamics probe set."""
    return json.loads(Path(path).read_text())


def _build_full_ids_for_score(
    tokenizer,
    persona_prompt: str,
    question: str,
    r_text: str,
    *,
    marker_text: str = MARKER_TEXT,
    marker_id: int = EXPECTED_MARKER_TOKEN_ID,
    sep: str = MARKER_SEP,
) -> tuple[list[int], int]:
    """Build full token-id sequence ``T_persona(q) + R + sep + marker`` + slot.

    Returns ``(full_ids, slot_index)`` where ``slot_index = len(full_ids) - 1``
    is the appended marker position. Asserts ``full_ids[-1] == marker_id``.

    Mirrors ``contrastive_neg_geometry_472.eval_one_cell.build_full_ids`` but
    skips the train-vs-eval token-equality contract (the dynamics callback
    scores log P at a slot that does not need to be byte-identical with the
    training slot — the metric is *trajectory* not endpoint-comparable).
    """
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(prompt_text + r_text + sep + marker_text, add_special_tokens=False)
    if full_ids[-1] != marker_id:
        raise RuntimeError(
            f"dynamics-callback marker slot drift: full_ids[-1]={full_ids[-1]}, "
            f"expected {marker_id}"
        )
    return full_ids, len(full_ids) - 1


class MarkerDynamicsCallback(TrainerCallback):
    """Log per-step source/bystander ΔG + emission-rate trajectories to WandB.

    Args:
        probes: output of ``build_dynamics_probes``.
        tokenizer: HF tokenizer (matches the trainer's tokenizer).
        base_logp_scorer: callable(persona, q, r_text) -> float — returns base
            ``log P(` ※`)`` at the post-R slot on the given trained-state R.
            The CALLER constructs this (typically wraps a held-on-CPU base model
            slice; we accept a callable here so smoke / unit tests can pass a
            stub).
        cadence_steps: fire every N training steps.
        max_new_tokens: cap for greedy generation per probe (default 256 —
            shorter than the headline eval's 2048 because the dynamics callback
            is a trajectory-only signal and longer gens linearly inflate the
            in-training pause).
        wandb_run: optional WandB run handle. If None, uses ``wandb.log`` directly
            (which logs to the current active run).
        is_distributed: if True (full-FT path), only rank 0 fires the callback;
            other ranks skip. Detected via ``state.is_world_process_zero`` when
            ``state`` is available at call-time.
    """

    def __init__(
        self,
        probes: dict,
        tokenizer,
        base_logp_scorer,
        *,
        cadence_steps: int = DYNAMICS_CADENCE_STEPS,
        max_new_tokens: int = 256,
        wandb_run=None,
        snapshots_path: str | Path | None = None,
    ):
        self.probes = probes
        self.tokenizer = tokenizer
        self.base_logp_scorer = base_logp_scorer
        self.cadence_steps = int(cadence_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.wandb_run = wandb_run
        # R2.1 round-2 fix: optional persist path. When set, the callback's
        # `on_train_end` hook dumps `self.snapshots` (and a small manifest)
        # as JSON to this path so the analyzer's `_gather_dynamics_snapshots`
        # can read it without depending on a live WandB connection.
        self.snapshots_path: Path | None = Path(snapshots_path) if snapshots_path else None
        # snapshots[step] -> {metrics}
        self.snapshots: dict[int, dict[str, float]] = {}
        self._last_fired_step = -1

    @staticmethod
    def _gen_one(model, tokenizer, prompt_ids: list[int], max_new_tokens: int) -> str:
        """Greedy generate one continuation under the current model state.

        Returns the decoded text (excluding the prompt). Suspends gradient and
        sets to eval mode for the duration of the call; restores training mode
        on exit. Caller is responsible for ensuring the model is on a CUDA
        device.
        """
        import torch

        was_training = model.training
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([prompt_ids], device=next(model.parameters()).device)
            attn_mask = torch.ones_like(input_ids)
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        if was_training:
            model.train()
        gen_ids = out[0, input_ids.shape[1] :].tolist()
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    @staticmethod
    def _score_trained_logp(
        model, tokenizer, full_ids: list[int], slot: int, marker_id: int
    ) -> tuple[float, bool]:
        """Score trained log P(marker) at ``slot`` + argmax==marker flag.

        Teacher-forced single forward pass on ``full_ids``; returns
        ``(log_prob, argmax_is_marker)``. The slot points at the APPENDED
        marker, so we read the log-prob at position ``slot-1`` predicting the
        token at position ``slot``.
        """
        import torch

        was_training = model.training
        model.eval()
        with torch.no_grad():
            ids = torch.tensor([full_ids], device=next(model.parameters()).device)
            logits = model(ids).logits  # (1, T, V)
            # Predict position `slot` from logits at position `slot - 1`.
            log_probs = torch.log_softmax(logits[0, slot - 1].float(), dim=-1)
            lp = float(log_probs[marker_id].item())
            top_id = int(log_probs.argmax().item())
        if was_training:
            model.train()
        return lp, (top_id == marker_id)

    def _fire(self, model, global_step: int) -> None:
        """Run the 20-probe eval pass + log to WandB."""
        import torch

        marker_id = EXPECTED_MARKER_TOKEN_ID
        per_probe: list[dict] = []
        for persona, spec in self.probes.items():
            for q in spec["questions"]:
                # Build prompt-only ids for generation.
                messages = [
                    {"role": "system", "content": spec["system"]},
                    {"role": "user", "content": q},
                ]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                r_text = self._gen_one(model, self.tokenizer, prompt_ids, self.max_new_tokens)
                # Build full_ids = prompt + R + sep + marker, score trained log P.
                full_ids, slot = _build_full_ids_for_score(
                    self.tokenizer, spec["system"], q, r_text
                )
                tr_lp, tr_argmax = self._score_trained_logp(
                    model, self.tokenizer, full_ids, slot, marker_id
                )
                # Score base log P on the SAME current-step R.
                base_lp = float(self.base_logp_scorer(persona, q, r_text))
                per_probe.append(
                    {
                        "persona": persona,
                        "role": spec["role"],
                        "question": q,
                        "trained_logp": tr_lp,
                        "base_logp": base_lp,
                        "delta_g": tr_lp - base_lp,
                        "argmax_marker": bool(tr_argmax),
                    }
                )
        # Aggregate metrics.
        source = [p for p in per_probe if p["role"] == "source"]
        bystander = [p for p in per_probe if p["role"] == "bystander"]
        if not source or not bystander:
            raise RuntimeError(
                f"MarkerDynamicsCallback fired with empty source ({len(source)}) or "
                f"bystander ({len(bystander)}) probes at step {global_step}"
            )
        source_dg = sum(p["delta_g"] for p in source) / len(source)
        bystander_dg = sum(p["delta_g"] for p in bystander) / len(bystander)
        source_emit = sum(1.0 for p in source if p["argmax_marker"]) / len(source)
        bystander_emit = sum(1.0 for p in bystander if p["argmax_marker"]) / len(bystander)
        metrics = {
            "dynamics/source_delta_g": source_dg,
            "dynamics/bystander_mean_delta_g": bystander_dg,
            "dynamics/source_emission_rate": source_emit,
            "dynamics/bystander_mean_emission_rate": bystander_emit,
            "dynamics/global_step": global_step,
        }
        # Per-bystander breakdowns.
        for p in DYNAMICS_BYSTANDER_PERSONAS:
            sub = [q for q in bystander if q["persona"] == p]
            if sub:
                metrics[f"dynamics/bystander/{p}_delta_g"] = sum(q["delta_g"] for q in sub) / len(
                    sub
                )
                metrics[f"dynamics/bystander/{p}_emission_rate"] = sum(
                    1.0 for q in sub if q["argmax_marker"]
                ) / len(sub)

        self.snapshots[global_step] = {**metrics, "n_probes": len(per_probe)}

        # WandB log (no-op if wandb not active).
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=global_step)
        except ImportError:
            pass

        log.info(
            "[dynamics step=%d] source ΔG=%.3f, bystander ΔG=%.3f, "
            "source emit=%.2f, bystander emit=%.2f",
            global_step,
            source_dg,
            bystander_dg,
            source_emit,
            bystander_emit,
        )

        # Free transient memory aggressively to keep the in-training cost
        # bounded on the full-FT arm where one rank does the eval.
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # Distributed safety: only the world-zero rank fires the callback (other
        # ranks would re-do the same work or stall on collective ops).
        if not state.is_world_process_zero:
            return
        step = int(state.global_step)
        if step == 0 or step == self._last_fired_step:
            return
        if step % self.cadence_steps != 0:
            return
        self._last_fired_step = step
        self._fire(model, step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Cap the trajectory + persist snapshots (R2.1 round-2 fix).

        Fires one final snapshot at the last step if the cadence missed it,
        then dumps ``self.snapshots`` to ``self.snapshots_path`` (when set) so
        ``analyze.py::_gather_dynamics_snapshots`` can read the trajectory.
        Falls back to ``<args.output_dir>/dynamics.json`` when no explicit
        path was passed at construction — preserves trajectory data for any
        caller that forgot to thread the path.
        """
        if state is not None and not state.is_world_process_zero:
            return
        if model is not None:
            step = int(state.global_step) if state is not None else 0
            if step != self._last_fired_step and step > 0:
                self._fire(model, step)
        # Persist snapshots — even if the final-fire path no-ops (rank issue),
        # the snapshots we DID collect on earlier on_step_end calls deserve
        # to land on disk.
        out = self.snapshots_path
        if out is None and args is not None and getattr(args, "output_dir", None):
            out = Path(args.output_dir) / "dynamics.json"
        if out is not None:
            self.persist_snapshots(out)

    def persist_snapshots(self, path: str | Path) -> Path:
        """Dump ``self.snapshots`` + manifest to ``path`` as JSON.

        Exposed for unit tests + the dispatcher's post-train write-back path
        (B5 round-2 fix). Returns the path. Idempotent on repeat calls.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "i508_dynamics_v1",
            "cadence_steps": self.cadence_steps,
            "n_probes": len([q for v in self.probes.values() for q in v.get("questions", [])]),
            "snapshots": {str(step): snap for step, snap in sorted(self.snapshots.items())},
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log.info("[dynamics] persisted %d snapshots → %s", len(self.snapshots), out)
        return out


def make_cpu_base_logp_scorer(
    base_model_path: str,
    tokenizer,
    *,
    probes: dict | None = None,
    device: str | None = None,
):
    """Build a base log-prob scorer for use as ``base_logp_scorer`` argument.

    Loads the base model lazily; the closure caches the loaded model + the
    probes dict so per-call work is only a single forward pass at the
    persona-q-r slot.

    M7 round-1 fix: the scorer now CLOSES OVER the loaded ``probes`` dict
    instead of re-reading ``DYNAMICS_PROBES_PATH`` on every call. The caller
    must pass the same probes dict it constructed the callback with;
    otherwise we fall back to the canonical-path read once at construction
    (preserving the old behavior for callers that didn't track the dict).

    M5 round-1 fix: ``device`` selects where the forward pass runs. ``"cpu"``
    (default if no CUDA) is ~30s × 20 probes on a 7B model and inflates
    in-training pause to ~10 min/fire; pass ``"cuda"`` for the LoRA arm
    (where the trainer's model uses 1 GPU and there's headroom for the base
    forward). For the full-FT arm, B4 disables the in-training callback
    entirely (offline extraction at checkpoint time).

    Returns:
        A callable ``(persona, q, r_text) -> base_logp: float``.

    Raises if the base model can't be loaded or ``persona`` is missing from
    the closed-over probes dict.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if probes is None:
        # Fallback (legacy): read the canonical path ONCE at construction.
        from explore_persona_space.experiments.lora_vs_ft_508 import (
            DYNAMICS_PROBES_PATH,
        )

        probes = load_dynamics_probes(DYNAMICS_PROBES_PATH)
    closed_probes = dict(probes)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(
        "[dynamics] loading base model on %s for in-training base log-prob: %s",
        device,
        base_model_path,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device if device == "cpu" else {"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base.eval()

    def _score(persona: str, q: str, r_text: str) -> float:
        if persona not in closed_probes:
            raise KeyError(f"persona {persona!r} not in dynamics probes ({sorted(closed_probes)})")
        full_ids, slot = _build_full_ids_for_score(
            tokenizer, closed_probes[persona]["system"], q, r_text
        )
        with torch.no_grad():
            ids = torch.tensor([full_ids], device=device)
            logits = base(ids).logits
            log_probs = torch.log_softmax(logits[0, slot - 1].float(), dim=-1)
            return float(log_probs[EXPECTED_MARKER_TOKEN_ID].item())

    return _score


def _snapshot_from_per_probe(per_probe: list[dict], global_step: int) -> dict:
    """Aggregate per-probe ΔG/emission stats into one snapshot dict (matches `_fire`)."""
    source = [p for p in per_probe if p["role"] == "source"]
    bystander = [p for p in per_probe if p["role"] == "bystander"]
    if not source or not bystander:
        raise RuntimeError(
            f"snapshot aggregation needs both source ({len(source)}) and "
            f"bystander ({len(bystander)}) probes; step={global_step}"
        )
    metrics = {
        "dynamics/source_delta_g": sum(p["delta_g"] for p in source) / len(source),
        "dynamics/bystander_mean_delta_g": sum(p["delta_g"] for p in bystander) / len(bystander),
        "dynamics/source_emission_rate": sum(1.0 for p in source if p["argmax_marker"])
        / len(source),
        "dynamics/bystander_mean_emission_rate": sum(1.0 for p in bystander if p["argmax_marker"])
        / len(bystander),
        "dynamics/global_step": global_step,
    }
    for p_name in DYNAMICS_BYSTANDER_PERSONAS:
        sub = [q for q in bystander if q["persona"] == p_name]
        if sub:
            metrics[f"dynamics/bystander/{p_name}_delta_g"] = sum(q["delta_g"] for q in sub) / len(
                sub
            )
            metrics[f"dynamics/bystander/{p_name}_emission_rate"] = sum(
                1.0 for q in sub if q["argmax_marker"]
            ) / len(sub)
    return {**metrics, "n_probes": len(per_probe), "step": global_step}


def _score_checkpoint_for_dynamics(
    trained_model_path: str,
    base_model_path: str,
    tokenizer,
    probes: dict,
    *,
    device: str | None = None,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Load one trained checkpoint + run the 20-probe eval pass.

    Returns the per-probe rows used by ``_snapshot_from_per_probe``. Used by
    ``extract_fullft_dynamics_from_checkpoints`` (R2.2 round-2 fix) for the
    FT-arm trajectory — each FT cell's saved checkpoints get fed through this
    function in sequence.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trained = AutoModelForCausalLM.from_pretrained(
        trained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu" if device == "cpu" else {"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    trained.eval()
    # Base model loaded ONCE per call (caller can pre-load + pass in if hot).
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu" if device == "cpu" else {"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base.eval()

    per_probe: list[dict] = []
    marker_id = EXPECTED_MARKER_TOKEN_ID
    with torch.no_grad():
        for persona, spec in probes.items():
            for q in spec["questions"]:
                messages = [
                    {"role": "system", "content": spec["system"]},
                    {"role": "user", "content": q},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                ids_t = torch.tensor([prompt_ids], device=device)
                out = trained.generate(
                    input_ids=ids_t,
                    attention_mask=torch.ones_like(ids_t),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                r_text = tokenizer.decode(
                    out[0, ids_t.shape[1] :].tolist(), skip_special_tokens=True
                )
                full_ids, slot = _build_full_ids_for_score(tokenizer, spec["system"], q, r_text)
                fids_t = torch.tensor([full_ids], device=device)
                # Trained log P at slot.
                tr_log_probs = torch.log_softmax(
                    trained(fids_t).logits[0, slot - 1].float(), dim=-1
                )
                tr_lp = float(tr_log_probs[marker_id].item())
                tr_argmax = bool(int(tr_log_probs.argmax().item()) == marker_id)
                # Base log P at slot on the SAME R.
                bs_log_probs = torch.log_softmax(base(fids_t).logits[0, slot - 1].float(), dim=-1)
                bs_lp = float(bs_log_probs[marker_id].item())
                per_probe.append(
                    {
                        "persona": persona,
                        "role": spec["role"],
                        "question": q,
                        "trained_logp": tr_lp,
                        "base_logp": bs_lp,
                        "delta_g": tr_lp - bs_lp,
                        "argmax_marker": tr_argmax,
                    }
                )
    del trained, base
    import gc

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_probe


def extract_fullft_dynamics_from_checkpoints(
    checkpoint_index: dict[str, dict],
    base_model_path: str,
    tokenizer,
    probes: dict,
    *,
    output_path: str | Path,
    device: str | None = None,
    max_new_tokens: int = 256,
    score_fn=None,
) -> Path:
    """Offline post-checkpoint dynamics extractor for the FT arm (R2.2 round-2 fix).

    Walks ``checkpoint_index`` (the manifest written by
    ``FullFTCheckpointAtFractionsCallback.index()``: ``{frac_key: {step, path}}``);
    for each fraction with a non-None ``path``, loads the trained checkpoint,
    runs the 20-probe pass, aggregates into a snapshot keyed by ``step``, and
    writes the full snapshot dict to ``output_path`` (same schema as
    ``MarkerDynamicsCallback.persist_snapshots``).

    Args:
        checkpoint_index: ``{"0.25": {"step": 12, "path": "/.../frac_0.25"}, ...}``
            — the manifest from ``train_metadata.json["checkpoint_index"]``.
        base_model_path: HF id / path of the base model (for per-checkpoint
            base log P read).
        tokenizer: HF tokenizer (matches base + trained).
        probes: output of ``build_dynamics_probes`` (1 source × 5 q + 3
            bystanders × 5 q = 20 probes).
        output_path: where to write the aggregated ``dynamics.json``.
        device: ``"cuda"`` / ``"cpu"`` / None (auto-detect).
        max_new_tokens: cap on the per-probe greedy generation.
        score_fn: optional injected scorer with signature
            ``(trained_path, base_path, tokenizer, probes, *,
            device=..., max_new_tokens=...) -> list[per_probe_dict]``.
            Defaults to ``_score_checkpoint_for_dynamics``. Exposed so unit
            tests can substitute a stub without touching disk / GPU.

    Returns:
        ``Path`` to the written ``dynamics.json``.

    The FT arm's checkpoints are deleted after this extractor runs (per
    ``EPM_DELETE_INTERMEDIATE_FT_CKPTS=1`` in the dispatcher); the
    ``dynamics.json`` is the durable artifact downstream.
    """
    scorer = score_fn if score_fn is not None else _score_checkpoint_for_dynamics
    snapshots: dict[int, dict] = {}
    for frac_key, entry in sorted(checkpoint_index.items(), key=lambda kv: float(kv[0])):
        ckpt_path = entry.get("path") if isinstance(entry, dict) else None
        step = (entry or {}).get("step") if isinstance(entry, dict) else None
        if not ckpt_path or step is None:
            log.info("[fullft-dynamics] skipping frac=%s (no path/step)", frac_key)
            continue
        log.info("[fullft-dynamics] extracting frac=%s step=%d ckpt=%s", frac_key, step, ckpt_path)
        per_probe = scorer(
            ckpt_path,
            base_model_path,
            tokenizer,
            probes,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        snap = _snapshot_from_per_probe(per_probe, global_step=int(step))
        snapshots[int(step)] = snap

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "i508_dynamics_v1",
        "extraction_mode": "offline_post_checkpoint",
        "n_probes": sum(len(v.get("questions", [])) for v in probes.values()),
        "snapshots": {str(step): snap for step, snap in sorted(snapshots.items())},
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(
        "[fullft-dynamics] wrote %d snapshots → %s",
        len(snapshots),
        out,
    )
    return out
