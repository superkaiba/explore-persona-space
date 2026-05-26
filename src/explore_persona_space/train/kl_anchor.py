"""Self-distillation KL anchor for marker-install Phase 1 (issue #382).

Mechanism (plan §5 "KL anchor — the load-bearing novel mechanism"):

  step 0 .. teacher_freeze_step:
      Regular SFT. No anchor activity.

  AT teacher_freeze_step (one-shot):
      Snapshot the (frozen) model's top-K logits + indices on the
      held-out C+ anchor batch. Store on CPU bf16 (top-K keeps storage
      ~50 MB vs ~19 GB full-vocab).

  teacher_freeze_step .. start_step:
      Regular SFT. No KL added (10% step-gap so the model briefly
      continues SFT after the snapshot before being pulled back).

  start_step .. end_of_phase:
      Per optimizer step (gradient-sync microstep only): run a forward
      pass on the anchor batch (anchor_batch_size x anchor_grad_accum
      micro-batches), gather student logits at the same top-K teacher
      indices, and compute ``KL(teacher || student)`` over the C+
      assistant tokens. The KL gradient (``kl_weight * raw_kl``) is
      backpropagated INSIDE the anchor loop, one micro-batch at a time
      via ``accelerator.backward``, depositing into LoRA ``.grad``
      buffers. HF Trainer then backpropagates the SFT loss separately.
      Issue #382 OOM fix (2026-05-26): the previous design accumulated
      all anchor-loop graphs in memory before a single outer backward,
      requiring ~108 GB of autograd tape that didn't fit on an 80 GB H100
      — see ``MarkerKLAnchor.kl_loss`` docstring for the fix details.

Invariants verified by ``tests/test_kl_anchor.py``:

  1. ``compute_loss`` override IS reached (subclass entry verified).
  2. KL term sign is correct: pushing student logits TOWARD teacher
     reduces the KL loss (test on synthetic 2-class problem).
  3. Activation gating: ``kl_loss`` returns 0 (no contribution) for
     ``global_step < start_step`` and a nonzero scalar for
     ``global_step >= start_step``.
  4. KL term is nonzero during the active window on a fresh model
     where student logits diverge from frozen teacher (smoke test).

The anchor uses `train_on_responses_only`-style masking: KL is computed
only on the assistant response tokens (masked by the same response
template ``"<|im_start|>assistant\\n"`` the SFT data collator uses).
This matches the gradient signal we want to anchor — marker emission at
the assistant turn — and avoids pulling the model on user/system tokens.

Logged WandB scalars during the active window:
  - ``train/kl_anchor_loss``     raw KL term value
  - ``train/kl_anchor_weighted`` ``kl_weight * kl_loss`` (the amount added)
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from trl import SFTTrainer

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ── Dataclass: anchor config ─────────────────────────────────────────────────


@dataclass
class KLAnchorConfig:
    """Anchor hyperparameters — populated from Hydra cfg.kl_anchor.

    All fractions are interpreted relative to the trainer's total optimizer-step
    count (``trainer.state.max_steps``). The trainer fills max_steps before
    training starts; we resolve fractional thresholds in ``on_train_begin``.
    """

    enabled: bool = False
    anchor_dataset: str = ""  # JSONL path; messages-shape (system/user/assistant)
    kl_weight: float = 0.5
    teacher_freeze_step_frac: float = 0.4
    start_step_frac: float = 0.5
    anchor_batch_size: int = 8
    anchor_grad_accum: int = 8
    top_k_logits: int = 50
    # Response template for masking KL to assistant tokens only. Same default
    # as DataCollatorForCompletionOnlyLM uses in trainer.py.
    response_template: str = "<|im_start|>assistant\n"
    # Optional override for the maximum sequence length tokenized into the
    # anchor batch; defaults to the trainer's max_seq_length.
    max_seq_length: int | None = None

    @staticmethod
    def from_hydra(stage_cfg: Any) -> KLAnchorConfig:
        """Build a KLAnchorConfig from a Hydra stage config or stage dict.

        Accepts either a DictConfig with a ``kl_anchor`` key or a dict with the
        same. Returns a disabled config (``enabled=False``) if ``kl_anchor`` is
        missing — callers should check ``cfg.enabled`` before instantiating an
        anchored trainer.
        """
        # Treat DictConfig and dict uniformly via .get.
        kl_cfg = None
        if hasattr(stage_cfg, "get"):
            kl_cfg = stage_cfg.get("kl_anchor", None)
        elif isinstance(stage_cfg, dict):
            kl_cfg = stage_cfg.get("kl_anchor")
        if kl_cfg is None:
            return KLAnchorConfig()
        # Pull each field with default fallback to dataclass defaults.
        defaults = KLAnchorConfig()
        get = kl_cfg.get if hasattr(kl_cfg, "get") else (lambda k, d=None: kl_cfg.get(k, d))
        return KLAnchorConfig(
            enabled=bool(get("enabled", False)),
            anchor_dataset=str(get("anchor_dataset", defaults.anchor_dataset)),
            kl_weight=float(get("kl_weight", defaults.kl_weight)),
            teacher_freeze_step_frac=float(
                get("teacher_freeze_step_frac", defaults.teacher_freeze_step_frac)
            ),
            start_step_frac=float(get("start_step_frac", defaults.start_step_frac)),
            anchor_batch_size=int(get("anchor_batch_size", defaults.anchor_batch_size)),
            anchor_grad_accum=int(get("anchor_grad_accum", defaults.anchor_grad_accum)),
            top_k_logits=int(get("top_k_logits", defaults.top_k_logits)),
            response_template=str(get("response_template", defaults.response_template)),
            max_seq_length=get("max_seq_length", defaults.max_seq_length),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_anchor_examples(path: str) -> list[dict]:
    """Load the JSONL anchor batch. Strict: raises on missing file / empty rows."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"KL-anchor dataset {path!r} does not exist. "
            "Run the data-gen script (scripts/generate_issue382_marker_install.py) "
            "first, or set kl_anchor.enabled=false."
        )
    rows: list[dict] = []
    with open(p) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"KL-anchor file {path} line {line_idx + 1} is not valid JSON: {exc}"
                ) from exc
            if "messages" not in row:
                raise RuntimeError(
                    f"KL-anchor file {path} line {line_idx + 1}: missing 'messages' key. "
                    f"Expected messages-shape rows."
                )
            rows.append(row)
    if len(rows) == 0:
        raise RuntimeError(f"KL-anchor file {path} is empty after stripping blank lines.")
    return rows


def _tokenize_anchor_batch(
    rows: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    response_template: str,
) -> dict[str, torch.Tensor]:
    """Tokenize each row with the chat template + build assistant-response mask.

    Returns:
        dict with:
          input_ids:       (N, max_seq_length) long
          attention_mask:  (N, max_seq_length) long
          response_mask:   (N, max_seq_length) bool — True only on assistant
                           response tokens (after the response template marker).
    """
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    if len(response_template_ids) == 0:
        raise RuntimeError(
            f"Response template {response_template!r} tokenizes to zero tokens; cannot mask."
        )

    n = len(rows)
    input_ids = torch.zeros((n, max_seq_length), dtype=torch.long)
    attention_mask = torch.zeros((n, max_seq_length), dtype=torch.long)
    response_mask = torch.zeros((n, max_seq_length), dtype=torch.bool)

    for i, row in enumerate(rows):
        messages = row["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        amask = enc["attention_mask"][0]
        input_ids[i] = ids
        attention_mask[i] = amask

        # Find the LAST occurrence of the response template sub-sequence in `ids`
        # (KL anchor data uses one assistant turn per row, so last-match = the
        # assistant turn we care about).
        ids_list = ids.tolist()
        rt_len = len(response_template_ids)
        anchor_start = -1
        for j in range(len(ids_list) - rt_len, -1, -1):
            if ids_list[j : j + rt_len] == response_template_ids:
                anchor_start = j + rt_len
                break
        if anchor_start < 0:
            raise RuntimeError(
                f"KL-anchor row {i}: response template {response_template!r} not found in "
                f"tokenized chat-template output. The tokenizer may not match the data-gen "
                f"tokenizer, or the row is malformed."
            )
        # Mask True on response tokens up to the first padding token (use amask).
        for k in range(anchor_start, max_seq_length):
            if amask[k].item() == 0:
                break
            response_mask[i, k] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


# ── Top-K snapshot + KL ──────────────────────────────────────────────────────


def _forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass; returns logits of shape (B, T, V)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits  # (B, T, V)


def _snapshot_top_k(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    top_k: int,
    micro_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot teacher snapshot.

    Returns (teacher_top_logits, teacher_top_indices), both shape
    (N, T, top_k) on CPU bf16 / int32. Caller stores them.

    Compute is done in mini-batches of ``micro_batch_size`` on the GPU,
    then moved to CPU.
    """
    n = input_ids.size(0)
    t_max = input_ids.size(1)
    teacher_top_logits = torch.zeros((n, t_max, top_k), dtype=torch.bfloat16)
    teacher_top_indices = torch.zeros((n, t_max, top_k), dtype=torch.int32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, micro_batch_size):
            end = min(start + micro_batch_size, n)
            ids = input_ids[start:end].to(device)
            amask = attention_mask[start:end].to(device)
            logits = _forward_logits(model, ids, amask)  # (b, T, V)
            assert logits.dim() == 3, logits.shape
            assert logits.size(0) == end - start, (logits.size(0), end - start)
            assert logits.size(1) == t_max, (logits.size(1), t_max)
            # Top-K across vocab dim.
            top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
            teacher_top_logits[start:end] = top_vals.to(torch.bfloat16).cpu()
            teacher_top_indices[start:end] = top_idx.to(torch.int32).cpu()
            del logits, top_vals, top_idx, ids, amask
    model.train()
    return teacher_top_logits, teacher_top_indices


def _kl_top_k(
    student_logits: torch.Tensor,
    teacher_top_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student) over the top-K teacher vocab positions, masked to
    response tokens.

    Args:
        student_logits:  (B, T, V) — live student forward pass logits
        teacher_top_logits:  (B, T, K) — snapshot teacher logits at top-K indices
        teacher_top_indices: (B, T, K) — long/int indices of the top-K vocab
        response_mask: (B, T) bool — True only on assistant-response tokens

    Returns:
        scalar tensor — mean KL across response tokens (sum / N_response_tokens).

    The KL is computed on the SOFTMAX-normalized top-K slice. This is a
    standard distillation approximation: when teacher mass concentrates on the
    top-K, the KL on the slice closely tracks the full-vocab KL. See plan
    §"Top-50 logit distillation (vs full-vocab)".
    """
    assert student_logits.dim() == 3, student_logits.shape
    assert teacher_top_indices.dim() == 3, teacher_top_indices.shape
    assert teacher_top_indices.size() == teacher_top_logits.size(), (
        teacher_top_indices.size(),
        teacher_top_logits.size(),
    )
    assert response_mask.dim() == 2, response_mask.shape

    teacher_top_indices = teacher_top_indices.to(device=student_logits.device, dtype=torch.long)
    teacher_top_logits = teacher_top_logits.to(
        device=student_logits.device, dtype=student_logits.dtype
    )
    # Gather student logits at teacher's top-K indices: (B, T, K).
    student_top_logits = torch.gather(student_logits, dim=-1, index=teacher_top_indices)

    # Softmax both within the K-slice (KL on the slice).
    teacher_p = F.softmax(teacher_top_logits, dim=-1)
    student_log_p = F.log_softmax(student_top_logits, dim=-1)
    teacher_log_p = F.log_softmax(teacher_top_logits, dim=-1)
    # KL(teacher || student) = sum teacher_p * (log teacher_p - log student_p)
    per_token_kl = torch.sum(teacher_p * (teacher_log_p - student_log_p), dim=-1)  # (B, T)

    # Mask to response tokens and average over those tokens (per-token mean).
    mask = response_mask.to(device=per_token_kl.device, dtype=per_token_kl.dtype)
    denom = mask.sum().clamp_min(1.0)
    kl_mean = (per_token_kl * mask).sum() / denom
    return kl_mean


# ── The anchor controller ────────────────────────────────────────────────────


@dataclass
class MarkerKLAnchor:
    """State machine + teacher cache + per-step KL computation.

    Not a TrainerCallback — wired into the trainer via subclass override
    in ``KLAnchoredSFTTrainer.compute_loss``. The trainer calls
    ``self.kl_anchor.kl_loss(model, global_step)`` from inside compute_loss;
    this returns the additive KL term (scalar tensor) or 0.0 if the window
    isn't active yet.

    State machine values (string):
      "init"             — built, but training hasn't started
      "before_freeze"    — past on_train_begin, before teacher snapshot
      "teacher_frozen"   — snapshot done, in the gap before start_step
      "active"           — KL term is added each step
    """

    config: KLAnchorConfig
    state: str = "init"
    total_steps: int = 0
    teacher_freeze_step: int = 0
    start_step: int = 0
    # Tokenized anchor batch (CPU tensors; moved to GPU per micro-step).
    anchor_input_ids: torch.Tensor | None = None
    anchor_attention_mask: torch.Tensor | None = None
    anchor_response_mask: torch.Tensor | None = None
    # Teacher snapshot (CPU bf16 / int32).
    teacher_top_logits: torch.Tensor | None = None
    teacher_top_indices: torch.Tensor | None = None
    # Last logged KL value (raw, before kl_weight scaling) — for tests + logging.
    last_kl: float = 0.0
    last_step_observed: int = -1
    # Internal: which micro-batch within the per-step round we serve next.
    _micro_idx: int = 0
    # Internal: deterministic shuffle of anchor indices each step.
    _shuffle: list[int] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        config: KLAnchorConfig,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> MarkerKLAnchor:
        """Load + tokenize the anchor batch. Total steps + state resolved later
        in ``on_train_begin``."""
        if not config.enabled:
            raise ValueError("MarkerKLAnchor.build called with config.enabled=False")
        rows = _load_anchor_examples(config.anchor_dataset)
        msl = config.max_seq_length or max_seq_length
        tokens = _tokenize_anchor_batch(
            rows,
            tokenizer,
            max_seq_length=msl,
            response_template=config.response_template,
        )
        logger.info(
            "MarkerKLAnchor: tokenized %d anchor rows (max_seq=%d, response_template=%r).",
            len(rows),
            msl,
            config.response_template,
        )
        return cls(
            config=config,
            state="init",
            anchor_input_ids=tokens["input_ids"],
            anchor_attention_mask=tokens["attention_mask"],
            anchor_response_mask=tokens["response_mask"],
        )

    # ── Lifecycle methods ────────────────────────────────────────────────────

    def on_train_begin(self, total_steps: int) -> None:
        """Resolve absolute step thresholds from fractional config."""
        if total_steps <= 0:
            raise RuntimeError(
                f"MarkerKLAnchor.on_train_begin called with total_steps={total_steps}; "
                "the trainer must have set max_steps>0 by now."
            )
        self.total_steps = total_steps
        self.teacher_freeze_step = max(1, round(total_steps * self.config.teacher_freeze_step_frac))
        self.start_step = max(
            self.teacher_freeze_step + 1,
            round(total_steps * self.config.start_step_frac),
        )
        self.state = "before_freeze"
        logger.info(
            "MarkerKLAnchor: total_steps=%d, teacher_freeze_step=%d, start_step=%d, "
            "kl_weight=%.3f, top_k=%d",
            total_steps,
            self.teacher_freeze_step,
            self.start_step,
            self.config.kl_weight,
            self.config.top_k_logits,
        )

    def _snapshot_teacher(self, model: torch.nn.Module) -> None:
        """Take the one-shot teacher snapshot (top-K logits + indices)."""
        device = next(model.parameters()).device
        logger.info(
            "MarkerKLAnchor: snapshotting teacher logits at step %d / %d (top-K=%d).",
            self.teacher_freeze_step,
            self.total_steps,
            self.config.top_k_logits,
        )
        self.teacher_top_logits, self.teacher_top_indices = _snapshot_top_k(
            model,
            self.anchor_input_ids,
            self.anchor_attention_mask,
            self.anchor_response_mask,
            top_k=self.config.top_k_logits,
            micro_batch_size=self.config.anchor_batch_size,
            device=device,
        )
        self.state = "teacher_frozen"

    def _maybe_advance_state(self, global_step: int, model: torch.nn.Module) -> None:
        """One-shot transitions: before_freeze -> teacher_frozen -> active."""
        if self.state == "before_freeze" and global_step >= self.teacher_freeze_step:
            self._snapshot_teacher(model)
        if self.state == "teacher_frozen" and global_step >= self.start_step:
            self.state = "active"
            logger.info(
                "MarkerKLAnchor: anchor active at step %d / %d (will contribute to loss).",
                global_step,
                self.total_steps,
            )

    def kl_loss(
        self,
        model: torch.nn.Module,
        global_step: int,
        *,
        accelerator: Any | None = None,
    ) -> torch.Tensor | float:
        """Compute the KL anchor term for the current optimizer step.

        Two code paths, selected by ``accelerator``:

          ``accelerator is None`` (test / legacy synthetic path):
            Build a per-micro-batch list of KL scalars (graph-attached),
            stack-mean them, return the graph-attached scalar. Existing unit
            tests in ``tests/test_kl_anchor.py`` use this path with stub models
            (no real backward is run). Returns ``0.0`` (plain ``float``) when
            the anchor window is not yet active.

          ``accelerator is not None`` (production path — issue #382 OOM fix):
            For each of ``anchor_grad_accum`` micro-batches, run one student
            forward, compute the per-micro-batch KL, scale by
            ``kl_weight / n_accum``, and call ``accelerator.backward`` on the
            resulting per-micro-batch tensor IMMEDIATELY. This frees the
            student-forward autograd tape before the next micro-batch's
            forward is built, so only ONE micro-batch's tape is ever live
            (vs all 64 simultaneously under the legacy stack-mean path).
            Returns a DETACHED zero-dim scalar tensor (the unweighted raw KL
            averaged across the micro-batches), for trainer-side logging
            only. Returns ``0.0`` when the anchor is not yet active.

        Effective anchor coverage per call: ``anchor_batch_size *
        anchor_grad_accum`` examples (plan §10 mandate). Under prod settings
        (``anchor_batch_size=1, anchor_grad_accum=64``) this is 64 examples.

        Numerical equivalence (production vs legacy path):

            Σ_i (kl_weight * kl_micro_i / n_accum)  for i in [0, n_accum)
            = kl_weight * (1/n_accum) * Σ_i kl_micro_i
            = kl_weight * mean(kl_micro_i)
            = kl_weight * raw_kl

          which is exactly what the legacy ``loss = sft_loss + kl_weight *
          raw_kl`` plus ``accelerator.backward(loss)`` would have deposited
          in the LoRA ``.grad`` buffers. Bf16 accumulation order changes the
          last few decimals; semantics are otherwise identical.

        IMPORTANT: the returned scalar is the **raw** KL (NOT multiplied by
        ``kl_weight``); the caller logs this value for diagnostics. In the
        production path the gradient contribution (``kl_weight * raw_kl``)
        has already been deposited in the LoRA ``.grad`` buffers — the
        caller MUST NOT add the returned scalar to its loss (would
        double-count).
        """
        self.last_step_observed = global_step
        # State transitions happen up here, BEFORE we decide whether to fire.
        self._maybe_advance_state(global_step, model)
        if self.state != "active":
            self.last_kl = 0.0
            return 0.0

        if self.teacher_top_logits is None or self.teacher_top_indices is None:
            raise RuntimeError(
                "MarkerKLAnchor.kl_loss called in 'active' state but teacher snapshot is None"
            )

        device = next(model.parameters()).device
        n = self.anchor_input_ids.size(0)
        mbs = self.config.anchor_batch_size
        n_accum = max(1, int(self.config.anchor_grad_accum))

        if accelerator is None:
            # Test / legacy synthetic path: graph-attached stack-mean return.
            # The caller (a unit test) does NOT call backward — we just need a
            # scalar tensor whose `.item()` reflects the average KL.
            kl_terms: list[torch.Tensor] = []
            for _accum_step in range(n_accum):
                idxs = self._next_micro_batch_indices(n=n, mbs=mbs, global_step=global_step)
                ids = self.anchor_input_ids[idxs].to(device)
                amask = self.anchor_attention_mask[idxs].to(device)
                rmask = self.anchor_response_mask[idxs].to(device)
                t_logits = self.teacher_top_logits[idxs]
                t_idx = self.teacher_top_indices[idxs]
                student_logits = _forward_logits(model, ids, amask)  # (b, T, V)
                kl_terms.append(_kl_top_k(student_logits, t_logits, t_idx, rmask))
            kl = torch.stack(kl_terms).mean()
            self.last_kl = float(kl.detach().to(torch.float32).item())
            return kl

        # Production path: per-micro-batch backward, never hold > 1 graph.
        # The gradient contribution is `kl_weight * raw_kl` per optimizer
        # step, deposited into LoRA `.grad` buffers via `accelerator.backward`.
        kl_log_sum = 0.0
        kl_weight = float(self.config.kl_weight)
        inv_n = 1.0 / float(n_accum)
        for _accum_step in range(n_accum):
            idxs = self._next_micro_batch_indices(n=n, mbs=mbs, global_step=global_step)
            ids = self.anchor_input_ids[idxs].to(device)
            amask = self.anchor_attention_mask[idxs].to(device)
            rmask = self.anchor_response_mask[idxs].to(device)
            t_logits = self.teacher_top_logits[idxs]
            t_idx = self.teacher_top_indices[idxs]
            student_logits = _forward_logits(model, ids, amask)  # (b, T, V)
            kl_micro = _kl_top_k(student_logits, t_logits, t_idx, rmask)
            # Weight per-micro-batch so the SUM over n_accum iterations
            # deposits the same gradient as the legacy `kl_weight * mean()`
            # would have produced. Use `accelerator.backward` (handles
            # GradScaler / fp16 / bf16 amp safely; bare t.backward does not).
            weighted_micro = (kl_weight * inv_n) * kl_micro
            accelerator.backward(weighted_micro)
            kl_log_sum += float(kl_micro.detach().to(torch.float32).item())
            # Free the per-micro-batch tape before the next forward builds one.
            del student_logits, kl_micro, weighted_micro

        # Detached scalar — only for logging by the caller. Gradient is
        # already in LoRA .grad buffers; caller MUST NOT add to its loss.
        raw_kl_mean = kl_log_sum * inv_n
        self.last_kl = float(raw_kl_mean)
        return torch.tensor(raw_kl_mean, dtype=torch.float32, device=device).detach()

    def _next_micro_batch_indices(self, *, n: int, mbs: int, global_step: int) -> list[int]:
        """Yield the next ``mbs`` anchor-row indices from the deterministic
        permutation, refilling + reshuffling when exhausted.

        The permutation is reseeded each time it refills using ``global_step``
        so a seed-replay reproduces the exact slice ordering. Slices may wrap
        within a single ``kl_loss`` call when ``anchor_grad_accum > 1`` and
        the cumulative micro-batches exceed ``n`` — the second refill is
        seeded with ``global_step + 1`` so the wrap is also deterministic.
        """
        if not self._shuffle or self._micro_idx >= len(self._shuffle):
            seed = int(global_step) + (0 if not self._shuffle else 1)
            rng = torch.Generator()
            rng.manual_seed(seed)
            self._shuffle = torch.randperm(n, generator=rng).tolist()
            self._micro_idx = 0
        slice_start = self._micro_idx
        slice_end = min(slice_start + mbs, len(self._shuffle))
        idxs = self._shuffle[slice_start:slice_end]
        self._micro_idx = slice_end
        return idxs


# ── KL-anchored SFTTrainer subclass ──────────────────────────────────────────


class KLAnchoredSFTTrainer(SFTTrainer):
    """SFTTrainer subclass that adds a KL-anchor term to compute_loss.

    Constructor takes the same args as SFTTrainer + a ``kl_anchor:
    MarkerKLAnchor`` keyword. The anchor's ``on_train_begin`` is invoked
    on the first call to ``compute_loss`` (when ``self.state.max_steps``
    is reliably set by the trainer).

    Per-optimizer-step contract (plan §10 — "Per optimizer step in the
    anchor-active window... one extra forward pass on 64 tokenized
    examples"):

      1. SFT loss = super().compute_loss(...) — fires on EVERY microstep,
         as the HF Trainer requires.
      2. KL backward is invoked AT MOST ONCE per optimizer step, on the
         microstep that HF marks as the gradient-sync microstep
         (``self.accelerator.gradient_state.sync_gradients == True``).
         Under ``gradient_accumulation_steps=N``, that is the Nth
         (last) microstep of each optimizer-step window.
      3. On the firing microstep: ``kl_anchor.kl_loss(model, step,
         accelerator=self.accelerator)`` runs ``anchor_grad_accum``
         student forwards in a loop, and for EACH micro-batch calls
         ``accelerator.backward(kl_weight / n_accum * kl_micro)``,
         depositing per-micro-batch gradient into LoRA ``.grad`` and
         freeing the autograd tape before the next forward. The total
         gradient deposited equals ``kl_weight * mean(kl_micro_i) =
         kl_weight * raw_kl``, matching the legacy design's contribution
         per optimizer step. The returned scalar is DETACHED — we do NOT
         add it to ``loss`` (would double-count the KL gradient).
      4. WandB log fires only on the gating microstep, so the anchor
         scalars are emitted once per optimizer step (matching the
         ``train_loss`` cadence).

    Round-3 fix (issue #382 reconciler verdict): previously, ``kl_loss``
    was invoked on EVERY ``compute_loss`` call. Under the default
    ``gradient_accumulation_steps=4``, this meant 4 KL passes per
    optimizer step (each pass doing ``anchor_grad_accum=8`` student
    forwards on ``anchor_batch_size=8`` examples) — i.e. 256 anchor
    examples / 32 student forwards per optimizer step, vs the plan's
    intended 64 examples / 8 student forwards. That was 4x the plan's
    anti-erasure load AND it broke the cost model. The gate restores
    the per-optimizer-step contract.
    """

    def __init__(self, *args, kl_anchor: MarkerKLAnchor, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(kl_anchor, MarkerKLAnchor):
            raise TypeError(f"kl_anchor must be MarkerKLAnchor, got {type(kl_anchor).__name__}")
        self.kl_anchor: MarkerKLAnchor = kl_anchor
        self._kl_anchor_initialized = False
        # Fallback microstep counter used only when
        # ``self.accelerator.gradient_state.sync_gradients`` is not exposed
        # (e.g. accelerate <0.27 or odd integration paths). The primary
        # signal is ``sync_gradients`` — see ``_is_sync_microstep``.
        self._microstep_counter: int = 0

    def _init_anchor_if_needed(self) -> None:
        if self._kl_anchor_initialized:
            return
        total_steps = int(self.state.max_steps)
        if total_steps <= 0:
            # Trainer hasn't populated max_steps yet; defer.
            return
        self.kl_anchor.on_train_begin(total_steps)
        self._kl_anchor_initialized = True

    def _is_sync_microstep(self) -> bool:
        """True iff this ``compute_loss`` call is the LAST microstep of the
        current optimizer step (i.e. HF will call ``optimizer.step()`` after
        the backward pass on this microstep's loss).

        Primary signal: ``self.accelerator.gradient_state.sync_gradients``,
        which HF Trainer sets to True on the gradient-sync microstep before
        invoking ``training_step`` (and therefore ``compute_loss``). This is
        the same flag HF itself uses to decide whether to step the optimizer.

        Fallback (when the attribute is missing): a simple modulo counter
        over ``compute_loss`` invocations, which fires every
        ``gradient_accumulation_steps`` calls. The fallback is correct ONLY
        as long as ``compute_loss`` is called exactly once per microstep
        with no skipped microsteps — which is true for the standard HF
        training loop.
        """
        # Increment the fallback counter unconditionally so it stays in
        # phase with microsteps regardless of which path we use.
        self._microstep_counter += 1

        try:
            sync = bool(self.accelerator.gradient_state.sync_gradients)
            return sync
        except AttributeError:
            grad_accum = max(1, int(self.args.gradient_accumulation_steps))
            return self._microstep_counter % grad_accum == 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute SFT loss; drive the KL anchor backward in-loop.

        Issue #382 OOM fix (2026-05-26): the KL anchor previously returned a
        graph-attached scalar that was added to ``sft_loss``, and the HF
        Trainer ran ONE outer ``accelerator.backward(sft_loss + kl)`` --
        which required holding the FULL anchor-loop autograd tape live
        (64 micro-batches x ~1.7 GB student-forward graph each = ~108 GB),
        deterministically OOMing on first anchor activation.

        The fix INVERTS this contract. On the gradient-sync microstep:

          1. We pass ``self.accelerator`` to ``kl_anchor.kl_loss``.
          2. ``kl_loss`` runs ``anchor_grad_accum`` student forwards in a
             loop, and after EACH forward it calls
             ``accelerator.backward(kl_weight / n_accum * kl_micro)``,
             depositing per-micro-batch gradients into LoRA ``.grad``
             buffers and freeing the micro-batch's autograd tape.
          3. ``kl_loss`` returns a DETACHED scalar with no graph; the
             trainer adds NOTHING to ``loss`` (would double-count) and uses
             the scalar ONLY for ``train/kl_anchor_loss`` logging.
          4. HF Trainer then runs its normal ``accelerator.backward(loss)``
             on the unchanged ``sft_loss``; SFT and KL gradients combine
             cleanly in the LoRA ``.grad`` buffers.

        Peak anchor-loop GPU memory drops from ~108 GB tape → ~2 GB tape
        (one live micro-batch graph at a time). Numerical equivalence to
        the legacy path holds up to bf16 accumulation order.
        """
        # Standard SFT loss (entropy + token metrics computed inside).
        out = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        if return_outputs:
            loss, outputs = out
        else:
            loss = out
            outputs = None

        # Initialize anchor (resolves total_steps) on first call.
        self._init_anchor_if_needed()

        # Only add anchor in train mode; eval forward calls compute_loss too.
        if not model.training:
            return (loss, outputs) if return_outputs else loss

        # GATE: fire KL exactly once per optimizer step, on the
        # gradient-sync microstep (plan §10 per-optimizer-step contract).
        if not self._is_sync_microstep():
            return (loss, outputs) if return_outputs else loss

        global_step = int(self.state.global_step)
        # Production path: pass `self.accelerator` so kl_loss runs in-loop
        # backwards itself and returns a DETACHED logging scalar. We do NOT
        # add this scalar to `loss` — the gradient is already in `.grad`.
        raw_kl = self.kl_anchor.kl_loss(model, global_step, accelerator=self.accelerator)
        if isinstance(raw_kl, torch.Tensor):
            # Active window: log via Trainer.log() so WandB picks it up
            # alongside train_loss. Cadence: once per optimizer step at
            # logging_steps boundaries — `self.state.global_step` has not yet
            # been incremented for the in-flight step, so test
            # `global_step + 1`. The "weighted" diagnostic is computed from
            # the detached raw_kl scalar; its gradient was already deposited.
            if (self.state.global_step + 1) % max(1, self.args.logging_steps) == 0:
                kl_weight = float(self.kl_anchor.config.kl_weight)
                raw_kl_val = float(raw_kl.detach().item())
                try:
                    self.log(
                        {
                            "train/kl_anchor_loss": raw_kl_val,
                            "train/kl_anchor_weighted_pre_scale": kl_weight * raw_kl_val,
                            "train/kl_anchor_state": 1.0,  # 1 == active
                        }
                    )
                except Exception:
                    logger.warning("Could not log KL anchor scalars to trainer.")
        else:
            # raw_kl == 0.0 (Python float) — anchor not active yet.
            if (self.state.global_step + 1) % max(1, self.args.logging_steps) == 0:
                state_codes = {"init": -1.0, "before_freeze": 0.0, "teacher_frozen": 0.5}
                with contextlib.suppress(Exception):
                    self.log(
                        {
                            "train/kl_anchor_loss": 0.0,
                            "train/kl_anchor_weighted_pre_scale": 0.0,
                            "train/kl_anchor_state": state_codes.get(self.kl_anchor.state, -1.0),
                        }
                    )

        return (loss, outputs) if return_outputs else loss
