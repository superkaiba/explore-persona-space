"""ModelOrganism — the build/verify harness over the Phase-0 modules (task #901, Phase 0g).

Pure composition: one validated :class:`ModelOrganism` identity ties together
behavior specs (0b/0d), the context typology (0b), negative panels (0c), data
generation (0d), the unified training recipe (0e), and the graded-judge +
tf-margin DVs (0a) into ``build()`` (datagen -> mix -> recipe -> train ->
dose-to-target checkpoint) and ``verify()`` (install check at the trigger
context, per-bystander leakage panel, dual-DV companion, install-controlled
transfer fraction). No new science lives here.

Arm-name mapping note (Phase-3c callers): the task body's "contra" arm is
``recipe.ARMS`` ``"primary"`` (``neg_ratio=1.0``); ``"posonly"`` /
``"nogeneric"`` / ``"both_off"`` are the ablation axes.

Error philosophy: fail-fast, typed, named. ``ValueError`` for contract
violations, :class:`UnsupportedOrganismError` for deliberate v1 scope
exclusions; an undefined quantity is ``None`` plus a machine-readable reason
field, and a quantity that MUST exist raises instead. No ``try/except: pass``,
no placeholder values, no silent degradation.

GPU/vLLM/torch/peft/eval_battery imports are FUNCTION-LOCAL (see
``_resolve_generation_deps`` / ``_resolve_margin_deps``) — a top-level import
of ``behavior_testbed_545.eval_battery`` would close the import cycle
``artifacts.__init__ -> organisms -> eval_battery -> artifacts.context``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from explore_persona_space.artifacts.behavior import BEHAVIORS, Behavior
from explore_persona_space.artifacts.context import CONTEXTS, Context
from explore_persona_space.artifacts.datagen import (
    _STRUCTURAL_PREDICATES,
    generate_training_data,
)
from explore_persona_space.artifacts.negatives import (
    DEFAULT_ASSISTANT_NEGATIVE,
    DEFAULT_PANEL_NAME,
    NEGATIVE_PANELS,
    Panel,
    assert_panel_disjoint_from_sources,
)
from explore_persona_space.artifacts.recipe import (
    DoseSelection,
    RecipeSpec,
    build_train_config,
    fullft_launch_command,
    mix_counts,
    recipe_for,
    select_dose_checkpoint,
)
from explore_persona_space.eval.graded_judge import JudgeResult, judge_graded
from explore_persona_space.train.sft import train_lora

if TYPE_CHECKING:
    from explore_persona_space.eval.margin import MarginResult

# ── Module constants (caller-overridable; none are training hyperparameters) ──

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # parity-tested vs behavior_testbed_545.BASE_MODEL
DEFAULT_EVAL_COMPLETIONS_PER_QUERY = 5  # parent plan pinned scope: 25 queries x 5 samples
DEFAULT_EVAL_TEMPERATURE = 1.0  # matches datagen gen_temperature default
DEFAULT_N_JUDGE_DRAWS = 5  # matches datagen n_judge_draws default
DEFAULT_MARGIN_POOL_CAP = 25  # per side; the build_fixed_pairs deterministic-cap pattern (#722)
DEFAULT_EVAL_MAX_NEW_TOKENS = 1024  # free-generation eval default (CLAUDE.md, raised 2026-06-24)
DEFAULT_MARGIN_MAX_BATCH_TOKENS = 8000  # compute_tf_margin token budget default

# ADVISORY held-out leakage bound on the judged-rate delta. Source: ungrounded —
# needs smoke-test (plan §11 item 8); Phase-1 pilot calibrates. The per-bystander
# numbers + denominators are the deliverable; never quote the flag without them.
DEFAULT_LEAKAGE_BOUND = 0.20
# Transfer-fraction denominator resolution floor (LN-logP units). Source:
# ungrounded — needs smoke-test (plan §11 item 9); Phase-1 margin distributions
# calibrate. Below the floor the fraction is None + reason, never a junk division.
MIN_SOURCE_MARGIN_DELTA = 0.05

# Kwarg names the verify call sites pass to the REAL functions — pinned by the
# signature-smoke test (plan §6 test 31) against inspect.signature of
# eval.graded_judge.judge_graded / eval.margin.compute_tf_margin. Keep in sync
# with the call sites in _rate_for_cell / _default_margin_read_fn.
_JUDGE_CALL_KWARGS: tuple[str, ...] = ("n_draws", "cache_dir", "save_raw", "judge_model")
_MARGIN_CALL_KWARGS: tuple[str, ...] = ("device", "max_batch_tokens")

_COMPANION_SKIPPED_NONE = "skipped: DVSpec.companion is None"
_COMPANION_UNIMPLEMENTED_SPOTCHECK = "unimplemented_v1: judged_spotcheck"


class UnsupportedOrganismError(NotImplementedError):
    """A behavior/DV combination the v1 harness deliberately does not implement.

    Programmatic carve-outs (marker, taught_fact) and the ground_truth_accuracy
    DV route here with a message naming the Phase-1 extension seam. Fail-fast,
    never a silent skip.
    """


# ── Injectable seam types (the datagen GenerateFn/JudgeFn pattern) ────────────

# (side_path | None, [messages per question], *, n, temperature) -> per-question
# completion lists, aligned 1:1 with the messages list, each of length n.
GenFn = Callable[..., list[list[str]]]
# ckpt_dir -> source judged rate at the trigger context C.
RateFn = Callable[[str], float]
# (side_path | None, Context, pos_pairs, neg_pairs) -> MarginResult. v2 (S4):
# the FIXED pools cross this seam so tests can assert both sides receive
# IDENTICAL pools and the derivation is actually consumed.
MarginReadFn = Callable[..., "MarginResult"]
JudgeFn = Callable[..., JudgeResult]
DatagenFn = Callable[..., tuple[Path, Path, Path]]


# ── Identity ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelOrganism:
    """One (behavior, trigger context, panel, recipe-axes, seed) identity.

    Fail-fast on construction: every field resolves against its registry or
    raises ``ValueError`` naming the field, the bad value, and the allowed set.
    The only documented defaults are ``generic_frac=None`` -> the recipe
    default and ``dose=None`` -> the recipe stopping band.
    """

    behavior: str  # key into BEHAVIORS
    context_id: str  # trigger context C; key into CONTEXTS (every kind is installable)
    negatives: str = DEFAULT_PANEL_NAME  # panel id into NEGATIVE_PANELS
    arm: str = "primary"  # in recipe.ARMS
    train_method: str = "lora"  # in recipe.TRAIN_METHODS
    generic_frac: float | None = None  # None -> recipe default; 0.0 = no-generic ablation
    dose: tuple[float, float] | None = None  # rate-band override; None -> recipe stopping default
    seed: int = 42

    def __post_init__(self) -> None:
        if self.behavior not in BEHAVIORS:
            raise ValueError(
                f"unknown behavior {self.behavior!r}; known behaviors: {sorted(BEHAVIORS)}"
            )
        ctx = CONTEXTS.get(self.context_id)
        if ctx is None:
            raise ValueError(
                f"unknown context_id {self.context_id!r}; known contexts: {sorted(CONTEXTS)}"
            )
        # No kind gate: INSTALLABLE_KINDS == CONTEXT_KINDS since "bare" joined
        # the installable set (#1090 fu3 — bare cells train on the default
        # context), and Context.__post_init__ already validates kind against
        # CONTEXT_KINDS, so a kind check here would be unreachable. A bare
        # default SOURCE is still rejected by the panel-disjointness invariant
        # below unless an explicit no-default panel is passed.
        if self.negatives not in NEGATIVE_PANELS:
            raise ValueError(
                f"unknown negative panel {self.negatives!r}; known panels: "
                f"{sorted(NEGATIVE_PANELS)}"
            )
        # One owner for arm / train_method / generic_frac legality (0e).
        spec = recipe_for(
            self.behavior,
            arm=self.arm,
            generic_frac=self.generic_frac,
            train_method=self.train_method,
        )
        # HARD disjointness invariant (contrastive-negatives.md, #527/#538).
        assert_panel_disjoint_from_sources(self.panel, [self.context_id])
        # r2 (BLOCKER source-panel-content-identity-gap): the slug/identity assert
        # above cannot see a REGISTRY ALIAS — a CONTEXTS entry whose CONTENT is
        # byte-identical to a trained negative (CONTEXTS['qt_rephrase_curious'] ==
        # neg_reph_curious; context.py's own `source` field says so). Such a
        # source would train the SAME prompt distribution as source-positive AND
        # contrastive-negative (the #527/#538 confound), so it is refused at
        # construction — fail-fast, never a silently confounded organism.
        source_fp = _context_content_fingerprint(ctx)
        for member in self.panel:
            if _context_content_fingerprint(member.to_context()) == source_fp:
                raise ValueError(
                    f"source context {self.context_id!r} is CONTENT-IDENTICAL to trained "
                    f"negative {member.slug!r} in panel {self.negatives!r} (identical "
                    "system/user_wrap/prefix fingerprint — a registry alias): it would "
                    "train the same prompt distribution as source-positive AND "
                    "contrastive-negative (the #527/#538 confound). Pick a non-aliased "
                    "source context or a panel without the alias."
                )
        if self.dose is not None:
            if spec.stopping.kind != "checkpoint_and_select":
                raise ValueError(
                    f"dose band override is only valid for checkpoint_and_select stopping; "
                    f"this recipe stops via {spec.stopping.kind!r}"
                )
            if len(self.dose) != 2:
                raise ValueError(f"dose must be a (lo, hi) pair, got {self.dose!r}")
            lo, hi = float(self.dose[0]), float(self.dose[1])
            if not lo < hi:
                raise ValueError(f"dose band must satisfy dose[0] < dose[1], got {self.dose!r}")

    @property
    def behavior_spec(self) -> Behavior:
        return BEHAVIORS[self.behavior]

    @property
    def context(self) -> Context:
        return CONTEXTS[self.context_id]

    @property
    def panel(self) -> Panel:
        return NEGATIVE_PANELS[self.negatives]

    @property
    def recipe(self) -> RecipeSpec:
        """Recompute via recipe_for — frozen inputs make this deterministic."""
        return recipe_for(
            self.behavior,
            arm=self.arm,
            generic_frac=self.generic_frac,
            train_method=self.train_method,
        )

    def slug(self) -> str:
        """Single-dash run identifier; judge item_ids ban ``"__"`` and prefix this."""
        s = (
            f"{self.behavior}-{self.context_id}-{self.arm}-{self.train_method}"
            f"-gf{self.recipe.generic_frac:g}-s{self.seed}"
        )
        if "__" in s:
            raise ValueError(f"organism slug must not contain '__': {s!r}")
        return s

    def build(self, **kwargs) -> BuildResult:
        """Thin delegate to :func:`build_organism`."""
        return build_organism(self, **kwargs)

    def verify(self, adapter_or_ckpt: str, **kwargs) -> OrganismReport:
        """Thin delegate to :func:`verify_organism`."""
        return verify_organism(self, adapter_or_ckpt, **kwargs)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildResult:
    """One build() outcome: the dose-selected artifact + full provenance."""

    adapter_path: str  # dose-selected checkpoint dir (lora) / full-model dir (fullft)
    selection: DoseSelection | None  # None for fixed_epochs (fullft)
    train_mix_path: str  # the assembled train_mix.jsonl
    data_paths: dict[str, str]  # pos / cn / pool_meta / datagen out_dir
    provenance: dict


@dataclass(frozen=True)
class BystanderRead:
    """One bystander context's leakage read (rates + optional margin companion).

    Denominator telemetry is PER SIDE (r2 — the r1 ``n_scored``/``n_dropped``
    pair mixed a trained-only denominator with a summed-both-sides drop count);
    draw-level judge telemetry lives in ``OrganismReport.judge_drop_telemetry``.
    """

    context_id: str
    trained_negative: bool  # slug OR content identity matches a trained panel member
    rate_trained: float
    rate_base: float
    rate_delta: float
    n_scored_trained: int  # trained-side denominator AFTER rule-9 drops
    n_dropped_trained: int  # trained-side all-draws-dropped completions (rule 9)
    n_scored_base: int  # base-side denominator AFTER rule-9 drops
    n_dropped_base: int  # base-side all-draws-dropped completions (rule 9)
    margin_trained: float | None  # None iff the companion was not computed
    margin_base: float | None
    margin_delta: float | None
    transfer_fraction: float | None  # margin-space; None when undefined
    transfer_fraction_undefined_reason: str | None


@dataclass(frozen=True)
class OrganismReport:
    """The verify() deliverable.

    Statistical-hygiene contract: the report exposes both install and
    per-bystander transfer fractions so downstream matched-install comparisons
    are possible; consumers MUST NOT correlate ``transfer_fraction`` against
    install — the shared noisy denominator manufactures correlation
    (marker-leakage-measurement.md § Install-strength confound; the #383
    X-vs-(X-Y) family). ``verify()`` itself computes no such correlation.

    ``install_ok`` (strict ``> 0``) and ``leakage_ok`` are ADVISORY flags:
    strict-positive install is noise-fragile near zero and the leakage bound is
    an ungrounded Phase-1-calibrated default — the panel numbers, deltas, and
    denominators are the deliverable; never quote a flag without them.
    ``leakage_ok`` is computed over the TRUE HELD-OUT bystanders only
    (``trained_negative=False``).
    """

    organism: dict  # asdict identity
    adapter_path: str
    dv_primary: str  # "judged_rate" | "structural"
    rate_trained_C: float
    rate_base_C: float
    install_delta: float  # rate_trained_C - rate_base_C
    install_ok: bool  # ADVISORY: strictly > 0
    source_margin_trained: float | None
    source_margin_base: float | None
    source_margin_delta: float | None
    bystanders: tuple[BystanderRead, ...]
    leakage_bound: float
    leakage_ok: bool  # ADVISORY: all HELD-OUT bystanders have rate_delta <= leakage_bound
    companion_status: str  # "computed" | skipped/unimplemented (machine-readable, never silent)
    # Per-(side, context): completion-level {n_scored, n_dropped} AND draw-level
    # {n_total_draws, n_dropped_draws} from JudgeResult (r2 — draw telemetry was
    # discarded in r1; a partially-dropped draw set is invisible at completion
    # grain whenever the item still has a mean score).
    judge_drop_telemetry: dict
    provenance: dict


# ── Small shared helpers ──────────────────────────────────────────────────────


def _git_commit_hash() -> str:
    """Short git SHA for provenance; 'uncommitted' when unavailable (fail-soft)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "uncommitted"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "uncommitted"


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


# ── Build-time mix token budget (task #906 r13/r14) ───────────────────────────

_budget_logger = logging.getLogger(__name__)

# Fail-loud floor for build-time row rejection: a rejected fraction above this
# means the budget is SYSTEMATICALLY too small for the question distribution —
# the remedy is a deliberate recipe max_length raise, never a silently shrunk
# mix. Grounded on the #906 att-20260704-061624 crash rows: 4/200 rows (2%)
# overflow (two extreme-tail WildChat prompts of 2181/1718 prompt-only tokens),
# median full-row 487/419 tokens — 0.10 separates "tail outliers" from "wrong
# budget" with wide margin on both sides.
MIX_MAX_REJECT_FRAC = 0.10


def mix_row_token_len(row: Mapping, tokenizer) -> int:
    """Full tokenized row length under the trainer's EXACT render.

    Matches the TRL prompt-completion tokenization ``train_lora`` performs (and
    ``sft.py::_tokenize_probe_row`` mirrors): render ``prompt + completion`` in
    ONE ``apply_chat_template`` call with ``add_generation_prompt=False``.
    SFTTrainer right-truncates each row at ``cfg.max_length``, so a row longer
    than the budget SILENTLY loses its completion tail — degraded supervision on
    content mixes, a loud ``MarkerOnlyDataCollator`` crash on marker mixes (the
    #906 r13 incident).
    """
    ids = tokenizer.apply_chat_template(
        list(row["prompt"]) + list(row["completion"]),
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return len(ids)


def _row_question(row: Mapping) -> str | None:
    """The row's question: the LAST user-message content (None if no user turn)."""
    for msg in reversed(list(row["prompt"])):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return None


def enforce_mix_token_budget(
    pos_rows: list[dict],
    cn_rows: list[dict],
    tokenizer,
    max_length: int,
    *,
    generic_rows: list[dict] | None = None,
    max_reject_frac: float = MIX_MAX_REJECT_FRAC,
    label: str = "mix-budget",
    log: logging.Logger | None = None,
) -> tuple[list[dict], list[dict], list[dict] | None, dict]:
    """Reject rows whose FULL tokenized length exceeds the training budget.

    The shared build-time gate behind BOTH mix paths (#906 r13 marker crash;
    r14 ``content-mix-token-budget-unenforced`` concern — content-class
    truncation at ``max_length`` is SILENT, no fail-loud collator):

    - pos/cn rows are QUESTION-paired (``datagen.generate_training_data``
      emits same-question negatives; the marker inline builder is
      index-aligned, a special case): when any pos/cn row overflows, every
      pos + cn row sharing its question (the last user-message content) is
      dropped from BOTH sides — preserving the same-question contrastive
      pairing (.claude/rules/contrastive-negatives.md) regardless of row
      ordering. A row with no user turn is dropped individually.
    - ``generic_rows`` (interleaved generic-chat corpus rows) carry no pairing
      and drop individually.
    - Fail loud (RuntimeError) when the rejected fraction exceeds
      ``max_reject_frac`` — a systematic overflow means the budget itself is
      wrong, never silently shrink the mix.
    - Fail loud (ValueError) when a non-empty contrastive-negative side is
      emptied by the gate — positive-only training leaks uniformly (#18/#207).
    - Log a WARNING on an asymmetric pos/cn drop (the ~1:1 contrastive ratio
      was perturbed; the below-floor survivors remain usable but the drift is
      surfaced).

    Returns ``(kept_pos, kept_cn, kept_generic, stats)`` where ``kept_generic``
    is None iff ``generic_rows`` is None. ``log`` routes telemetry to the
    caller's logger (default: this module's).
    """
    lg = log or _budget_logger
    pos_lens = [mix_row_token_len(r, tokenizer) for r in pos_rows]
    cn_lens = [mix_row_token_len(r, tokenizer) for r in cn_rows]
    gen_lens = [mix_row_token_len(r, tokenizer) for r in generic_rows or []]
    all_lens = pos_lens + cn_lens + gen_lens
    max_row_tokens = max(all_lens) if all_lens else 0

    bad_questions: set[str] = set()
    for rows, lens in ((pos_rows, pos_lens), (cn_rows, cn_lens)):
        for r, n in zip(rows, lens, strict=True):
            if n > max_length:
                q = _row_question(r)
                if q is not None:
                    bad_questions.add(q)

    def _keep(row: dict, n: int) -> bool:
        if n > max_length:
            return False
        q = _row_question(row)
        return q is None or q not in bad_questions

    kept_pos = [r for r, n in zip(pos_rows, pos_lens, strict=True) if _keep(r, n)]
    kept_cn = [r for r, n in zip(cn_rows, cn_lens, strict=True) if _keep(r, n)]
    kept_generic: list[dict] | None = None
    if generic_rows is not None:
        kept_generic = [r for r, n in zip(generic_rows, gen_lens, strict=True) if n <= max_length]

    n_rejected_pos = len(pos_rows) - len(kept_pos)
    n_rejected_cn = len(cn_rows) - len(kept_cn)
    n_rejected_generic = (len(generic_rows) - len(kept_generic)) if generic_rows is not None else 0
    n_rejected = n_rejected_pos + n_rejected_cn + n_rejected_generic
    total = len(pos_rows) + len(cn_rows) + len(generic_rows or [])
    rejected_frac = (n_rejected / total) if total else 0.0
    stats = {
        "enforced": True,
        "budget": int(max_length),
        "max_row_tokens": int(max_row_tokens),
        "n_rejected": n_rejected,
        "n_rejected_pos": n_rejected_pos,
        "n_rejected_cn": n_rejected_cn,
        "n_kept_pos": len(kept_pos),
        "n_kept_cn": len(kept_cn),
        "rejected_frac": rejected_frac,
        "reject_frac_floor": max_reject_frac,
    }
    if generic_rows is not None:
        stats["n_rejected_generic"] = n_rejected_generic
        stats["n_kept_generic"] = len(kept_generic or [])
    lg.info(
        "[%s] max_row_tokens=%d budget=%d n_rejected=%d (pos=%d cn=%d generic=%d) "
        "kept=%d/%d rejected_frac=%.3f floor=%.2f",
        label,
        max_row_tokens,
        max_length,
        n_rejected,
        n_rejected_pos,
        n_rejected_cn,
        n_rejected_generic,
        total - n_rejected,
        total,
        rejected_frac,
        max_reject_frac,
    )
    if rejected_frac > max_reject_frac:
        raise RuntimeError(
            f"[{label}] {n_rejected}/{total} mix rows ({rejected_frac:.1%}) exceed the "
            f"training max_length={max_length} (max row = {max_row_tokens} tokens) — above "
            f"the {max_reject_frac:.0%} rejection floor. The budget is systematically too "
            "small for this question/generation setting: raise the recipe's max_length "
            "override (grounded on the measured row-length distribution) or cap the "
            "generation length; do NOT silently shrink the mix."
        )
    if cn_rows and not kept_cn:
        raise ValueError(
            f"[{label}] token-budget enforcement rejected EVERY contrastive-negative row "
            f"({len(cn_rows)} pre-gate) while positives survived — positive-only training "
            "leaks uniformly (.claude/rules/contrastive-negatives.md); refusing to train "
            "a silently de-contrasted mix."
        )
    if n_rejected and n_rejected_pos != n_rejected_cn:
        lg.warning(
            "[%s] asymmetric drop: %d pos vs %d cn rows rejected — the ~1:1 "
            "positives-to-negatives contrastive ratio is perturbed (kept %d pos / %d cn)",
            label,
            n_rejected_pos,
            n_rejected_cn,
            len(kept_pos),
            len(kept_cn),
        )
    return kept_pos, kept_cn, kept_generic, stats


def _context_content_fingerprint(ctx: Context) -> str:
    """Deterministic sha256 over the message-shaping fields (content identity).

    Two contexts with identical (system, user_wrap, prefix_turns) produce
    byte-identical prompts, so they are the SAME measurement cell regardless of
    id — e.g. ``CONTEXTS['qt_rephrase_curious']`` vs the trained negative
    ``neg_reph_curious`` (context.py's own ``source`` field says so).
    """
    payload = json.dumps(
        {
            "system": ctx.system,
            "user_wrap": ctx.user_wrap,
            "prefix_turns": [dict(m) for m in ctx.prefix_turns],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return _sha256_text(payload)


def _resolve_eval_questions(behavior: Behavior, eval_questions: Sequence[str] | None) -> list[str]:
    qs = list(eval_questions) if eval_questions is not None else list(behavior.eval_question_bank)
    if not qs:
        raise ValueError(
            f"behavior {behavior.name!r} has an empty eval_question_bank and no explicit "
            "eval_questions were provided"
        )
    return qs


# ── Deferred GPU-dependency resolvers (#606 class; executed by test 26) ───────


def _resolve_generation_deps() -> dict[str, Any]:
    """Deferred imports for the production vLLM generation path.

    Function-local by design: a top-level ``eval_battery`` import closes the
    ``artifacts.__init__ -> organisms -> eval_battery -> artifacts.context``
    cycle, and vllm is GPU-lane-only. The CPU test suite executes this resolver
    (importorskip-guarded) so a renamed symbol fails in CI, not on the pod.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import (
        _is_full_model_dir,
        teardown_vllm,
    )

    return {
        "LLM": LLM,
        "SamplingParams": SamplingParams,
        "LoRARequest": LoRARequest,
        "_is_full_model_dir": _is_full_model_dir,
        "teardown_vllm": teardown_vllm,
    }


def _resolve_margin_deps() -> dict[str, Any]:
    """Deferred imports for the production HF/PEFT margin path (see above).

    ``_is_full_model_dir`` is the SAME routing helper the generation seam uses
    (r2 minor: the r1 margin seam re-implemented the detection inline, a drift
    risk between the two seams' full-model-vs-LoRA routing). ``eval_battery``
    has no top-level vllm import, so this resolver stays vllm-free.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.margin import compute_tf_margin
    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import (
        _is_full_model_dir,
    )

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "compute_tf_margin": compute_tf_margin,
        "_is_full_model_dir": _is_full_model_dir,
    }


# ── Production default seams (GPU; mocked in every test) ─────────────────────


class _SingleLiveResource:
    """At most ONE live GPU resource; a key switch tears the old one down FIRST.

    The memory-coexistence guard for the default GPU seams (r2 — concern
    ``gpu-seam-memory-coexistence``): the r1 seams cached one engine/model per
    ``side_path`` and tore down only at final close, so a second ~0.9-HBM vLLM
    engine (or an HF bf16 model) was constructed while the previous engine
    still held the GPU — OOM on any single-GPU pod. This holder guarantees the
    previous resource is torn down BEFORE the next one is built. Pure lifecycle
    logic (CPU-testable with recorder fns); the GPU builders plug in below.
    """

    _UNSET = object()

    def __init__(self, build: Callable[[Any], Any], teardown: Callable[[Any], None]) -> None:
        self._build = build
        self._teardown = teardown
        self._key: Any = self._UNSET
        self._value: Any = None

    def get(self, key: Any) -> Any:
        """The live resource for ``key`` — reused if live, else swap (teardown first)."""
        if self._key is not self._UNSET and key == self._key:
            return self._value
        self.close()  # teardown the previous resource BEFORE building the next
        value = self._build(key)
        self._key, self._value = key, value
        return value

    def close(self) -> None:
        """Tear down the live resource, if any. Idempotent."""
        if self._key is self._UNSET:
            return
        value = self._value
        self._key, self._value = self._UNSET, None
        self._teardown(value)


# Sentinel resource key PREFIX: EVERY LoRA adapter path maps to ONE
# rank-qualified key, so the _SingleLiveResource reuses a single enable_lora
# base engine across a dose ladder's checkpoints (crash-fix #1090 r3 — the r2
# per-side_path keying tore down + rebuilt an IDENTICAL engine per checkpoint,
# ~2.5 min each, and each teardown was one more exposure to the orphan-probe
# crash class). The key carries ``max_lora_rank`` (#1090 fu5 D2 item 2) so a
# 256-slot engine is never silently shared with a 64-slot expectation.
_SHARED_LORA_ENGINE_KEY = "__lora_engine__"
DEFAULT_MAX_LORA_RANK = 64  # run_generation_phase precedent; fu5 rank ladder passes 256


def _vllm_resource_key(
    side_path: str | None,
    is_full_model_dir: Callable[[str | None], bool],
    *,
    max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
) -> str | None:
    """Engine-identity key for the default vLLM generation seam.

    ``None`` (base) and full-model dirs keep their IDENTITY keys (a distinct
    engine each — the weights differ); every LoRA adapter path maps to the one
    rank-qualified ``{_SHARED_LORA_ENGINE_KEY}:r{max_lora_rank}`` (the engine
    is the base model + enable_lora at that slot width, identical across
    adapters — only the per-call ``LoRARequest`` differs). ``max_lora_rank``
    is part of the key so an engine built with 64 LoRA slots can never be
    silently reused where a 256-slot engine is expected (#1090 fu5 K5 seam).
    """
    if side_path is None or is_full_model_dir(side_path):
        return side_path
    return f"{_SHARED_LORA_ENGINE_KEY}:r{int(max_lora_rank)}"


def _is_shared_lora_key(key: str | None) -> bool:
    """True iff ``key`` is a rank-qualified shared-LoRA-engine key."""
    return key is not None and key.startswith(_SHARED_LORA_ENGINE_KEY)


def _lora_int_id(lora_ids: dict[str, int], side_path: str) -> int:
    """Distinct, STABLE, 1-based ``lora_int_id`` per adapter path.

    vLLM caches adapters by ``lora_int_id`` inside a shared engine, so two
    paths must never collide (a collision silently reuses the first adapter's
    weights) and repeat calls for one path must return the same id.
    """
    return lora_ids.setdefault(side_path, len(lora_ids) + 1)


def _default_vllm_generate_fn(
    base_model: str, *, max_lora_rank: int = DEFAULT_MAX_LORA_RANK
) -> GenFn:
    """ONE live vLLM engine at a time, chunked generate, teardown via close().

    Engine kwargs follow the ``run_generation_phase`` precedent
    (``max_lora_rank=64`` default — callers whose adapters exceed rank 64 pass
    a wider slot, e.g. the #1090 fu5 rank ladder passes 256; vLLM 0.11.0
    supports ranks up to 512 incl. 256 — plan §11 sources —
    ``max_model_len=8192``, ``enable_prefix_caching=True``,
    fixed sampling seed); chunked ≤500 prompts per ``llm.generate`` call
    (gotchas.md large-batch deadlock prevention) with ``use_tqdm=False``.
    Lifecycle (r3 — crash-fix #1090): ALL LoRA adapter paths share ONE
    enable_lora base engine (``_vllm_resource_key`` maps them to
    ``_SHARED_LORA_ENGINE_KEY``), so a dose ladder's checkpoint loop builds the
    engine ONCE instead of teardown+rebuild per rung; each adapter path still
    gets a DISTINCT stable ``lora_int_id`` (``_lora_int_id``), and the per-call
    ``LoRARequest`` is constructed in ``generate()``. A base <-> full-model <->
    lora-mode switch still swaps engines, teardown-first
    (``_SingleLiveResource``).
    """
    deps = _resolve_generation_deps()
    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    lora_ids: dict[str, int] = {}  # adapter path -> unique, stable lora_int_id (1-based)

    def _build(key: str | None) -> Any:
        # EPM_VLLM_GPU_MEM_UTIL (crash-fix #1074 run 1 + #1090 r2 — same
        # failure class): vLLM's default gpu_memory_utilization=0.9 demands
        # ~71.3 GiB on an A100/H100-80 and crashes gpu_worker.init_device
        # when a same-process HF trainer's allocator residue holds ~16 GiB at
        # the train->rate phase boundary (#1074 run 1: 71.32 GiB demanded vs
        # 63.65/79.25 GiB free after the in-process train_lora; #1090 r2 hit
        # the identical class). The post-train engine must fit BESIDE
        # imperfect trainer-memory release: default 0.75 x 79.25 = 59.4 GiB
        # covers the 7B bf16 weights (~15 GiB) + LoRA + a generous KV cache.
        # Env-overridable, resolved per engine build.
        common = {
            "max_model_len": 8192,
            "enable_prefix_caching": True,
            "seed": 0,
            "gpu_memory_utilization": float(os.environ.get("EPM_VLLM_GPU_MEM_UTIL", "0.75")),
        }
        if _is_shared_lora_key(key):
            return deps["LLM"](
                model=base_model, enable_lora=True, max_lora_rank=int(max_lora_rank), **common
            )
        if key is None:
            return deps["LLM"](model=base_model, **common)
        return deps["LLM"](model=key, **common)  # full-model dir (fullft arm)

    engine = _SingleLiveResource(_build, lambda llm: deps["teardown_vllm"](llm))

    def generate(
        side_path: str | None,
        messages_list: list[list[dict[str, str]]],
        *,
        n: int,
        temperature: float,
    ) -> list[list[str]]:
        key = _vllm_resource_key(side_path, deps["_is_full_model_dir"], max_lora_rank=max_lora_rank)
        llm = engine.get(key)
        lora_req = None
        if _is_shared_lora_key(key):
            lora_id = _lora_int_id(lora_ids, side_path)
            lora_req = deps["LoRARequest"](f"organism-{lora_id}", lora_id, side_path)
        tok = llm.get_tokenizer()
        prompts = [
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]
        sp = deps["SamplingParams"](
            n=n, temperature=temperature, max_tokens=DEFAULT_EVAL_MAX_NEW_TOKENS, seed=0
        )
        outs: list[list[str]] = []
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i : i + chunk_size]
            kwargs = {"use_tqdm": False}
            if lora_req is not None:
                kwargs["lora_request"] = lora_req
            chunk_out = llm.generate(chunk, sp, **kwargs)
            outs.extend([o.text for o in out.outputs] for out in chunk_out)
        return outs

    generate.close = engine.close  # type: ignore[attr-defined]
    return generate


def _default_margin_read_fn(base_model: str) -> MarginReadFn:
    """ONE live HF model at a time (base; base+PEFT adapter), tf-margin per context.

    Lifecycle (r2): a ``side_path`` switch frees the previous model BEFORE the
    next one loads (``_SingleLiveResource`` + ``torch.cuda.empty_cache``), and
    the returned fn exposes ``close()``. ``verify_organism`` only invokes this
    seam AFTER the generation phase has torn down every vLLM engine, so an HF
    bf16 model never coexists with a ~0.9-HBM vLLM engine on one GPU.
    """
    deps = _resolve_margin_deps()
    torch = deps["torch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = deps["AutoTokenizer"].from_pretrained(base_model)

    def _build(side_path: str | None) -> Any:
        if side_path is None:
            model = (
                deps["AutoModelForCausalLM"]
                .from_pretrained(base_model, torch_dtype=torch.bfloat16)
                .to(device)
            )
        elif deps["_is_full_model_dir"](side_path):  # full model dir (fullft arm)
            model = (
                deps["AutoModelForCausalLM"]
                .from_pretrained(side_path, torch_dtype=torch.bfloat16)
                .to(device)
            )
        else:  # LoRA adapter / checkpoint dir
            base = (
                deps["AutoModelForCausalLM"]
                .from_pretrained(base_model, torch_dtype=torch.bfloat16)
                .to(device)
            )
            model = deps["PeftModel"].from_pretrained(base, side_path)
        model.eval()
        return model

    def _teardown(model: Any) -> None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    holder = _SingleLiveResource(_build, _teardown)

    def read(
        side_path: str | None,
        ctx: Context,
        pos_pairs: list[dict],
        neg_pairs: list[dict],
    ) -> MarginResult:
        model = holder.get(side_path)
        # Kwargs pinned by _MARGIN_CALL_KWARGS (signature-smoke test 31).
        return deps["compute_tf_margin"](
            model,
            tokenizer,
            ctx.messages,
            pos_pairs,
            neg_pairs,
            device=device,
            max_batch_tokens=DEFAULT_MARGIN_MAX_BATCH_TOKENS,
        )

    read.close = holder.close  # type: ignore[attr-defined]
    return read


def _default_fullft_run(argv: list[str]) -> None:
    """Run the composed fullft launch argv, failing loud on a non-zero exit."""
    subprocess.run(argv, check=True)


# ── Fixed margin-pool derivation (v2/S4 — public, tested directly) ────────────


def derive_margin_pools(
    datagen_dir: Path | str,
    *,
    cap: int = DEFAULT_MARGIN_POOL_CAP,
) -> tuple[list[dict], list[dict]]:
    """FIXED (probe, answer) pools from the build's datagen sidecars.

    Joins ``raw_pos.jsonl`` / ``raw_neg.jsonl`` candidate rows with
    ``judge_rows.jsonl`` ``kept`` flags via ``request_id``, keeps KEPT rows
    only, routes ``arm == "positive"`` -> pos pool / ``"negative"`` -> neg
    pool, sorts by ``(question_id, variant_id)`` and takes the first ``cap``
    per side — the deterministic ``build_fixed_pairs`` selection pattern: the
    SAME fixed set is scored under EVERY context, so there is no
    selection-on-outcome bias (llm-judging.md § E2 rule 19).

    Returns ``(pos_pairs, neg_pairs)`` where each pair is
    ``{"probe", "answer", "question_id", "variant_id", "request_id"}``.

    Raises:
        ValueError: on a missing sidecar file, an unknown ``arm`` value, or an
            empty derived pool on either side.
    """
    d = Path(datagen_dir)
    raw_pos, raw_neg = d / "raw_pos.jsonl", d / "raw_neg.jsonl"
    judge_rows_path = d / "judge_rows.jsonl"
    for p in (raw_pos, raw_neg, judge_rows_path):
        if not p.exists():
            raise ValueError(
                f"margin-pool source missing: {p} (need raw_pos.jsonl, raw_neg.jsonl, "
                "judge_rows.jsonl from the build's datagen out_dir)"
            )
    kept_by_rid = {r["request_id"]: bool(r["kept"]) for r in _read_jsonl(judge_rows_path)}
    pools: dict[str, list[dict]] = {"positive": [], "negative": []}
    for raw_path in (raw_pos, raw_neg):
        for row in _read_jsonl(raw_path):
            if row.get("completion") is None:
                continue  # never generated / dropped at generation — never judged
            if not kept_by_rid.get(row["request_id"], False):
                continue  # judged and NOT kept (or never judged) — excluded
            arm = row["arm"]
            if arm not in pools:
                raise ValueError(
                    f"unknown arm {arm!r} in {raw_path} row {row['request_id']!r} "
                    "(expected 'positive' or 'negative')"
                )
            pools[arm].append(
                {
                    "probe": row["question"],
                    "answer": row["completion"],
                    "question_id": row["question_id"],
                    "variant_id": row["variant_id"],
                    "request_id": row["request_id"],
                }
            )
    for arm, pool in pools.items():
        pool.sort(key=lambda p: (p["question_id"], p["variant_id"]))
        if not pool:
            raise ValueError(
                f"derived margin pool for arm {arm!r} is empty under {d} — "
                "no kept candidates survive the judge filter"
            )
    return pools["positive"][:cap], pools["negative"][:cap]


# ── build ─────────────────────────────────────────────────────────────────────


def _assemble_mix(
    organism: ModelOrganism,
    spec: RecipeSpec,
    pos_path: Path,
    cn_path: Path,
    generic_data_path: Path | str | None,
    out_root: Path,
    *,
    tokenizer=None,
    max_length: int | None = None,
) -> tuple[Path, dict[str, int], dict[str, int]]:
    """Final-mix assembly (v2 — MF-2 surplus refusal / bounded-deficit tolerance).

    pos + contrastive-negative + generic rows per ``mix_counts``, seeded
    shuffle; ``train_mix.jsonl`` + ``mix_meta.json`` persist the moment
    assembly completes (checkpoint-per-phase). Returns
    ``(train_mix_path, counts_planned, counts_realized)``.

    When ``tokenizer`` and ``max_length`` are BOTH provided (the production
    path), every row's full tokenized length is checked against the training
    budget via :func:`enforce_mix_token_budget` (question-paired pos/cn drop,
    individual generic drop, fail-loud rejection floor) BEFORE assembly — the
    #906 r14 ``content-mix-token-budget-unenforced`` fix: SFTTrainer
    right-truncation at ``max_length`` is SILENT on content mixes (no
    fail-loud collator), so an overlong WildChat-lineage row would degrade its
    completion supervision without an error. ``tokenizer=None`` (the offline
    stub-seam test path) skips the gate, byte-identical legacy behavior.
    """
    pos_rows = _read_jsonl(pos_path)
    cn_rows = _read_jsonl(cn_path)
    if not pos_rows:
        raise ValueError(f"datagen emitted zero positive rows at {pos_path}")
    counts = mix_counts(len(pos_rows), generic_frac=spec.generic_frac, neg_ratio=spec.neg_ratio)
    panel_size = len(organism.panel)
    if counts["negatives"] == 0:
        use_neg: list[dict] = []  # posonly / both_off arms: zero negative rows in the mix
    elif len(cn_rows) > counts["negatives"]:
        raise ValueError(
            f"contrastive-negative SURPLUS: cn.jsonl has {len(cn_rows)} rows but the mix "
            f"needs {counts['negatives']}. cn rows carry no panel-member identity, so any "
            "subsample could silently unbalance the panel or drop the bare-default "
            "negatives. This branch is unreachable under every shipped arm "
            "(neg_ratio in {0, 1.0}) — a future recipe change must thread member identity "
            "through cn.jsonl and stratify the subsample per member."
        )
    elif counts["negatives"] - len(cn_rows) > panel_size - 1:
        raise ValueError(
            f"contrastive-negative SHORTFALL: cn.jsonl has {len(cn_rows)} rows, the mix "
            f"needs {counts['negatives']}, and the deficit "
            f"({counts['negatives'] - len(cn_rows)}) exceeds the tolerated per-member "
            f"floor-division remainder (panel_size - 1 = {panel_size - 1})"
        )
    else:
        use_neg = list(cn_rows)  # deficit <= panel_size - 1: the healthy floor-division case

    rng = random.Random(organism.seed)
    generic_rows: list[dict] = []
    if counts["generic"] > 0:
        if generic_data_path is None:
            raise ValueError(
                f"recipe generic_frac={spec.generic_frac:g} needs {counts['generic']} generic "
                "rows but no generic_data_path was provided (the generic-chat corpus is a "
                "Phase-1 data input; pass its prompt-completion JSONL explicitly)"
            )
        corpus = _read_jsonl(Path(generic_data_path))
        if len(corpus) < counts["generic"]:
            raise ValueError(
                f"generic corpus at {generic_data_path} has {len(corpus)} rows < "
                f"{counts['generic']} needed — refusing to sample with replacement"
            )
        generic_rows = rng.sample(corpus, counts["generic"])

    if tokenizer is not None and max_length is not None:
        pos_rows, use_neg, kept_generic, budget_stats = enforce_mix_token_budget(
            pos_rows,
            use_neg,
            tokenizer,
            int(max_length),
            generic_rows=generic_rows,
            label="content-mix-budget",
        )
        generic_rows = kept_generic if kept_generic is not None else []
        if not pos_rows:
            raise ValueError(
                "content-mix token-budget enforcement rejected every positive row "
                f"(budget={max_length})"
            )
    else:
        budget_stats = {"enforced": False, "reason": "no tokenizer/max_length provided"}
        _budget_logger.debug("[content-mix-budget] skipped: no tokenizer/max_length")
    _atomic_write_json(out_root / "mix_budget.json", budget_stats)

    mix = [*pos_rows, *use_neg, *generic_rows]
    rng.shuffle(mix)
    train_mix_path = out_root / "train_mix.jsonl"
    _write_jsonl(train_mix_path, mix)
    realized = {
        "positives": len(pos_rows),
        "negatives": len(use_neg),
        "generic": len(generic_rows),
    }
    input_paths = {
        "pos": str(pos_path),
        "cn": str(cn_path),
        "generic": str(generic_data_path) if generic_data_path is not None else None,
    }
    mix_meta = {
        "counts_planned": counts,
        "counts_realized": realized,
        "inputs": input_paths,
        "input_sha256": {k: _sha256_file(Path(v)) for k, v in input_paths.items() if v is not None},
        "spec": asdict(spec),
        "organism": asdict(organism),
        "seed": organism.seed,
        "mix_token_budget": budget_stats,
    }
    _atomic_write_json(out_root / "mix_meta.json", mix_meta)
    return train_mix_path, counts, realized


def release_trainer_cuda_memory(
    *,
    collect_fn: Callable[[], int] | None = None,
    empty_cache_fn: Callable[[], None] | None = None,
    ipc_collect_fn: Callable[[], None] | None = None,
    mem_info_fn: Callable[[], tuple[int, int]] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float] | None:
    """Release residual trainer CUDA memory before the post-train vLLM engine boots.

    ``train_lora`` already drops its heavy locals at its tail (``del trainer,
    model, tokenizer`` + ``gc.collect()`` + ``empty_cache()`` — train/sft.py),
    yet #1074 followup run 1 (GCE att-20260706-181717) measured ~15.6 GiB
    still resident at the next ``LLM(...)`` init (63.65/79.25 GiB free vs the
    vLLM default-0.9-util demand of 71.32 GiB). No module-level retainer was
    found in train/sft.py, train/trainer.py, or eval/callbacks.py, so this is
    defense-in-depth at the train->engine handoff: TWO gc passes
    (finalizer-bearing Trainer/Accelerator cycles can survive a single pass),
    an allocator-cache flush, and ``ipc_collect`` (the canonical
    vLLM-coexistence teardown tail, gotchas.md), then a LOG of the
    driver-level free-memory delta in the exact form

        [train-release] freed pre=<X>GiB post=<Y>GiB free

    That literal ``[train-release]`` tag is the #1074 crash-fix fix-engaged
    signal the relaunch greps for. Every CUDA touchpoint is injectable so the
    sequencing + log format are CPU-testable; missing callables resolve to the
    torch defaults lazily, and a no-CUDA host degrades to one bare
    ``gc.collect()`` returning None (nothing to measure).

    Returns:
        ``(pre_free_gib, post_free_gib)`` from ``mem_info_fn`` (default
        ``torch.cuda.mem_get_info``), or None on a no-CUDA host.
    """
    import gc

    collect = collect_fn if collect_fn is not None else gc.collect
    log = log_fn if log_fn is not None else logging.getLogger(__name__).info
    if empty_cache_fn is None or ipc_collect_fn is None or mem_info_fn is None:
        import torch

        if not torch.cuda.is_available():
            collect()
            return None
        empty_cache_fn = empty_cache_fn or torch.cuda.empty_cache
        ipc_collect_fn = ipc_collect_fn or torch.cuda.ipc_collect
        mem_info_fn = mem_info_fn or torch.cuda.mem_get_info
    pre_free_b, _total = mem_info_fn()
    collect()
    collect()
    empty_cache_fn()
    ipc_collect_fn()
    post_free_b, _total = mem_info_fn()
    pre_gib, post_gib = pre_free_b / 2**30, post_free_b / 2**30
    log(f"[train-release] freed pre={pre_gib:.2f}GiB post={post_gib:.2f}GiB free")
    return pre_gib, post_gib


def build_organism(
    organism: ModelOrganism,
    *,
    out_root: Path | str,
    base_model: str = DEFAULT_BASE_MODEL,
    generic_data_path: Path | str | None = None,
    gpu_id: int = 0,
    datagen_kwargs: Mapping[str, Any] | None = None,
    extra_overrides: Mapping[str, Any] | None = None,
    datagen_fn: DatagenFn = generate_training_data,
    train_fn: Callable[..., tuple[str, float]] = train_lora,
    rate_fn: RateFn | None = None,
    fullft_run_fn: Callable[[list[str]], None] | None = None,
    tokenizer=None,
    recipe_max_length: int | None = None,
) -> BuildResult:
    """datagen -> mix assembly -> recipe -> train -> dose-selected checkpoint.

    ``generic_data_path`` is a prompt-completion JSONL, REQUIRED whenever the
    resolved recipe's ``generic_frac > 0``. ``rate_fn`` is REQUIRED on the
    checkpoint_and_select path (the module ships :func:`make_source_rate_fn`
    as the production default the caller constructs explicitly) — validated at
    ENTRY, before any datagen/training work (r2 minor), and its ``close()``
    (when exposed, e.g. by the :func:`make_source_rate_fn` factory) is called
    after the checkpoint-ladder scoring so a factory-owned vLLM engine never
    outlives dose selection. ``datagen_fn`` is the injectable data-generation
    boundary and PUBLIC API surface (default :func:`generate_training_data`;
    the r1 review accepted this seam): same signature/returns contract as
    ``generate_training_data`` — mocked in tests, overridable for custom data
    pipelines. Every step fails fast; ``train_mix.jsonl`` + ``mix_meta.json``
    persist the moment assembly completes (checkpoint-per-phase).

    ``tokenizer`` (optional): the base model's tokenizer. When provided, mix
    assembly enforces the recipe's ``max_length`` token budget per row via
    :func:`enforce_mix_token_budget` (the #906 r14 silent-truncation fix) —
    production callers that will run the REAL trainer pass it; offline
    stub-seam tests omit it (gate skipped, legacy behavior).

    ``recipe_max_length`` (optional): a task-scoped DECLARED-DEVIATION seam
    for the recipe's ``max_length`` (the plan #1090 AMENDMENT/hot-fix
    lineage: a measured row-length distribution can exceed the unified
    recipe's 1024 budget above the 10% rejection floor, and the gate's own
    prescription is a deliberate recipe max_length raise). When set, it is
    threaded into ``spec.overrides`` IMMEDIATELY after recipe resolution, so
    the SAME spec feeds BOTH the mix token-budget gate (:func:`_assemble_mix`)
    AND the train-config build (:func:`build_train_config`) — one authority;
    the recipe recorded in ``mix_meta.json`` / provenance then honestly
    reports the enforced value. ``LOAD_BEARING_KEYS`` still protects the
    ``extra_overrides`` path: this is a deliberate, NAMED seam for a
    plan-declared recipe deviation, not a bypass.
    """
    out_root = Path(out_root)
    behavior = organism.behavior_spec
    spec = organism.recipe
    if recipe_max_length is not None:
        # One authority for the token budget: replace the spec's max_length
        # BEFORE any consumer (mix-budget gate, build_train_config, provenance)
        # reads it. Fixes the #1090 wrong-seam hot-fix (05b2405043), which
        # patched only the train_fn cfg while the mix-BUILD gate still read
        # the recipe's 1024 (organisms.py step 3) and crashed again.
        spec = replace(spec, overrides={**spec.overrides, "max_length": int(recipe_max_length)})

    # 1. Carve-out gate.
    if behavior.programmatic:
        raise UnsupportedOrganismError(
            f"behavior {organism.behavior!r} is a programmatic carve-out: datagen refuses "
            "programmatic behaviors by design (0d) and the marker/fact training-mix builders "
            "are not yet promoted into artifacts/ — Phase 1 wires the carve-out build path "
            "(recipe_for(...) already validates the recipe side)"
        )
    # 1b. Fail-fast contract check (r2 minor): the checkpoint_and_select path
    # needs rate_fn, and that is knowable at ENTRY — refuse before the
    # expensive datagen + training steps, not after training completes.
    if (
        spec.train_method != "fullft"
        and spec.stopping.kind == "checkpoint_and_select"
        and rate_fn is None
    ):
        raise ValueError(
            "rate_fn is REQUIRED on the checkpoint_and_select path (ckpt_dir -> "
            "source judged rate at C); construct the production default via "
            "make_source_rate_fn(organism, ...)"
        )

    # 2. Data (datagen re-asserts panel disjointness + resumes on an exact manifest match).
    out_root.mkdir(parents=True, exist_ok=True)
    datagen_dir = out_root / "datagen"
    pos_path, cn_path, pool_meta_path = datagen_fn(
        behavior,
        organism.context,
        organism.panel,
        out_dir=datagen_dir,
        seed=organism.seed,
        **dict(datagen_kwargs or {}),
    )
    # 3. Mix assembly (extracted; v2 — MF-2 surplus refusal / bounded-deficit tolerance).
    # Token budget = the recipe's max_length override (a LOAD_BEARING_KEY:
    # build_train_config refuses extra_overrides on it, so spec.overrides is
    # authoritative for BOTH the lora and fullft branches; 1024 is the
    # TrainLoraConfig field default when a recipe omits it).
    train_mix_path, counts, realized = _assemble_mix(
        organism,
        spec,
        Path(pos_path),
        Path(cn_path),
        generic_data_path,
        out_root,
        tokenizer=tokenizer,
        max_length=int(spec.overrides.get("max_length", 1024)),
    )

    # 4/5. Train.
    train_dir = out_root / "train"
    provenance: dict[str, Any] = {
        "organism": asdict(organism),
        "recipe": asdict(spec),
        "slug": organism.slug(),
        "base_model": base_model,
        "mix_counts_planned": counts,
        "mix_counts_realized": realized,
        "git_commit": _git_commit_hash(),
        "timestamp_utc": _now_utc(),
    }
    if spec.train_method == "fullft":
        argv = fullft_launch_command(
            spec,
            base_model=base_model,
            dataset_path=str(train_mix_path),
            output_dir=str(train_dir),
            seed=organism.seed,
            run_name=organism.slug(),
        )
        (fullft_run_fn or _default_fullft_run)(argv)
        adapter_path = str(train_dir)
        selection: DoseSelection | None = None
        provenance["fullft_argv"] = argv
        # The two named 0e gaps are passed through untouched; matched-batch
        # adjustment is Phase 3b's documented job (recipe.py docstring).
        provenance["effective_batch_note"] = (
            "fullft effective batch = num_processes x per-device batch x grad_accum; "
            "NOT matched to the lora twin here — Phase 3b divides grad_accum by "
            "num_processes for matched effective batch (recipe.fullft_launch_command)"
        )
    else:
        cfg = build_train_config(
            spec,
            run_name=organism.slug(),
            seed=organism.seed,
            gpu_id=gpu_id,
            extra_overrides=extra_overrides,
        )
        adapter_dir, loss = train_fn(base_model, str(train_mix_path), str(train_dir), cfg=cfg)
        # In-process GPU handoff (#1074 run-1 crash): release the trainer's
        # residual CUDA memory BEFORE the checkpoint-read rate_fn loop below
        # boots its vLLM engine (train_lora's own teardown left ~15.6 GiB
        # resident and the engine init failed its free-memory demand). The
        # [train-release] log line is the crash-fix fix-engaged signal.
        release_trainer_cuda_memory()
        provenance["training_loss"] = float(loss)
        adapter_path = str(adapter_dir)
        selection = None
        if spec.stopping.kind == "checkpoint_and_select":
            ckpt_dirs: dict[int, Path] = {}
            for p in Path(adapter_dir).glob("checkpoint-*"):
                suffix = p.name.split("-", 1)[1]
                if p.is_dir() and suffix.isdigit():
                    ckpt_dirs[int(suffix)] = p
            if not ckpt_dirs:
                raise ValueError(
                    f"no checkpoint-<step> dirs under {adapter_dir} — the recipe's "
                    "save_strategy='steps' checkpoint ladder did not take"
                )
            # rate_fn presence was validated at entry (step 1b). Close a
            # close()-exposing rate_fn after the ladder scoring (r2 — the
            # make_source_rate_fn factory owns a vLLM engine that must not
            # outlive dose selection).
            try:
                rates_by_step = {step: float(rate_fn(str(d))) for step, d in ckpt_dirs.items()}
            finally:
                rate_close = getattr(rate_fn, "close", None)
                if callable(rate_close):
                    rate_close()
            selection = select_dose_checkpoint(
                rates_by_step, band=organism.dose or spec.stopping.rate_band
            )
            adapter_path = str(ckpt_dirs[selection.step])
            provenance["rates_by_step"] = {str(k): v for k, v in rates_by_step.items()}
    provenance["dose_selection"] = asdict(selection) if selection is not None else None
    if selection is not None and selection.fallback is not None:
        # 0e's preregistered fallback: "never enters the band" is a reportable
        # outcome, not an infra failure — flagged loud in provenance.
        provenance["dose_selection_fallback"] = selection.fallback

    return BuildResult(
        adapter_path=adapter_path,
        selection=selection,
        train_mix_path=str(train_mix_path),
        data_paths={
            "pos": str(pos_path),
            "cn": str(cn_path),
            "pool_meta": str(pool_meta_path),
            "datagen_dir": str(datagen_dir),
        },
        provenance=provenance,
    )


# ── verify ────────────────────────────────────────────────────────────────────


def _generate_and_persist(
    gen_fn: GenFn,
    side: str,
    side_path: str | None,
    ctx: Context,
    questions: Sequence[str],
    *,
    n: int,
    temperature: float,
    out_dir: Path,
    base_model: str,
) -> list[list[str]]:
    """Generate (or resume from disk) the per-question completion lists for one cell.

    Persists ``completions__{side}__{context_id}.json`` the moment generation
    returns (checkpoint-per-phase; these are the raw completions the caller
    uploads under ``raw_completions/verify/`` per the Upload Policy).

    Resume is keyed on a MANIFEST of every output-affecting input (r2 — Codex
    concern ``unsafe-completion-resume-key``: the r1 predicate checked only the
    questions, so reusing an ``out_dir`` with a different adapter / base model /
    ``n`` / temperature silently reused stale completions and produced a false
    verification report). A mismatched resume REFUSES loud, naming the
    differing keys — never silent reuse.
    """
    manifest = {
        "side": side,
        # Adapter identity: the resolved path (None for the base side). A
        # content hash would be stronger but requires hashing multi-GB weights
        # per cell; the resolved path disambiguates every in-scope collision
        # (different adapters, same-basename checkpoint dirs across builds).
        "side_path": None if side_path is None else str(Path(side_path).resolve()),
        "base_model": base_model,
        "context_id": ctx.context_id,
        "context_fingerprint": _context_content_fingerprint(ctx),
        "questions_sha256": _sha256_text(json.dumps(list(questions), ensure_ascii=False)),
        "n_completions": n,
        "temperature": temperature,
    }
    out_path = out_dir / f"completions__{side}__{ctx.context_id}.json"
    if out_path.exists():
        payload = json.loads(out_path.read_text())
        persisted = payload.get("manifest")
        if persisted != manifest:
            if persisted is None:
                diff = sorted(manifest)  # pre-manifest file: no regime record at all
            else:
                diff = sorted(
                    k for k in set(manifest) | set(persisted) if persisted.get(k) != manifest.get(k)
                )
            raise ValueError(
                f"resume mismatch at {out_path}: persisted completions were generated "
                f"under a DIFFERENT regime (differing keys: {diff}; "
                f"persisted={ {k: (persisted or {}).get(k) for k in diff} }, "
                f"current={ {k: manifest.get(k) for k in diff} }) — stale completions "
                "would corrupt the verification report; use a fresh out_dir"
            )
        return payload["completions"]
    completions = gen_fn(
        side_path, [ctx.messages(q) for q in questions], n=n, temperature=temperature
    )
    if len(completions) != len(questions):
        raise ValueError(
            f"generate_fn returned {len(completions)} completion lists for "
            f"{len(questions)} questions at ({side}, {ctx.context_id})"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        out_path,
        {"manifest": manifest, "questions": list(questions), "completions": completions},
    )
    return completions


@dataclass(frozen=True)
class _CellRate:
    """One (side, context) cell's rate + completion- AND draw-level telemetry.

    r2 (Codex concern ``judge-draw-telemetry-lost``): the r1 tuple carried only
    completion-level counts, discarding ``JudgeResult.n_total_draws`` /
    ``n_dropped_draws`` — partial draw drops were invisible whenever an item
    still had a mean score. Structural cells (no judge) carry zero draws.
    """

    rate: float
    n_scored: int  # completions in the denominator AFTER rule-9 drops
    n_dropped: int  # completions with ALL draws dropped (rule 9)
    n_total_draws: int  # judge draws issued for this cell (0 on the structural path)
    n_dropped_draws: int  # judge draws dropped (REFUSAL / non-numeric / out-of-range)


def _rate_for_cell(
    behavior: Behavior,
    predicate: Callable[[str], bool] | None,
    judge_fn: JudgeFn,
    n_judge_draws: int,
    side: str,
    ctx: Context,
    questions: Sequence[str],
    completions: list[list[str]],
    judge_root: Path,
) -> _CellRate:
    """The :class:`_CellRate` for one (side, context) cell.

    Structural primary: the deterministic predicate, ZERO judge calls.
    Judged primary: graded scores; a None score (all draws dropped) is dropped
    from the denominator and counted — never coerced (llm-judging rule 9);
    ``rate = #(score > behavior.threshold) / n_scored``. An item_id MISSING
    from ``JudgeResult.scores`` is a judge-contract violation and raises (r2
    minor — distinct from the rule-9 present-with-``None`` disposition).
    """
    flat: list[tuple[str, str, str]] = []
    for i, q in enumerate(questions):
        for j, completion in enumerate(completions[i]):
            flat.append((f"{ctx.context_id}-{side}-q{i:03d}-c{j}", q, completion))
    if predicate is not None:
        n_pos = sum(1 for _iid, _q, c in flat if predicate(c))
        return _CellRate(n_pos / len(flat), len(flat), 0, 0, 0)
    cell_dir = judge_root / f"{side}_{ctx.context_id}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    # Kwargs pinned by _JUDGE_CALL_KWARGS (signature-smoke test 31).
    result = judge_fn(
        flat,
        behavior.judge_rubric,
        n_draws=n_judge_draws,
        cache_dir=cell_dir,
        save_raw=cell_dir / "judge_raw.json",
        judge_model=behavior.judge_model,
    )
    n_dropped = 0
    n_pos = 0
    n_scored = 0
    for iid, _q, _c in flat:
        if iid not in result.scores:
            raise ValueError(
                f"judge_fn contract violation at ({side}, {ctx.context_id}): item_id "
                f"{iid!r} is MISSING from JudgeResult.scores — a rule-9 all-draws-dropped "
                "item must be present with score None, never absent"
            )
        score = result.scores[iid]
        if score is None:
            n_dropped += 1
            continue
        n_scored += 1
        if score > behavior.threshold:
            n_pos += 1
    if n_scored == 0:
        raise ValueError(
            f"every completion at ({side}, {ctx.context_id}) was judge-dropped — a fully "
            "dropped cell is a judging outage, not a 0% rate"
        )
    return _CellRate(
        n_pos / n_scored, n_scored, n_dropped, result.n_total_draws, result.n_dropped_draws
    )


def _verify_context_set(
    organism: ModelOrganism,
    bystander_context_ids: Sequence[str] | None,
) -> list[tuple[Context, bool]]:
    """The bystander set as (context, trained_negative) pairs (v2 — MF-1).

    Default = the UNION of (a) every trained panel member materialized via
    ``NegativeContext.to_context()``, (b) every CONTEXTS entry other than the
    source, and (c) the bare default assistant — deduplicated by CONTENT
    IDENTITY (system + user_wrap + prefix fingerprint), not id alone, with
    panel members taking precedence so a byte-identical registry alias (e.g.
    ``qt_rephrase_curious`` == ``neg_reph_curious``) yields exactly ONE row
    labeled ``trained_negative=True``.
    """
    panel_ctxs = {n.slug: n.to_context() for n in organism.panel}
    panel_fps = {_context_content_fingerprint(c) for c in panel_ctxs.values()}
    panel_slugs = set(panel_ctxs)

    def _is_trained_negative(ctx: Context) -> bool:
        return ctx.context_id in panel_slugs or _context_content_fingerprint(ctx) in panel_fps

    source_fp = _context_content_fingerprint(organism.context)
    if bystander_context_ids is not None:
        out: list[tuple[Context, bool]] = []
        seen_ids: set[str] = set()
        for cid in bystander_context_ids:
            ctx = CONTEXTS.get(cid) or panel_ctxs.get(cid)
            if ctx is None:
                raise ValueError(
                    f"unknown bystander context id {cid!r}; known: "
                    f"{sorted(set(CONTEXTS) | panel_slugs)}"
                )
            if cid in seen_ids:
                raise ValueError(f"duplicate bystander context id {cid!r}")
            seen_ids.add(cid)
            out.append((ctx, _is_trained_negative(ctx)))
        return out

    candidates: list[Context] = list(panel_ctxs.values())
    candidates.extend(c for cid, c in CONTEXTS.items() if cid != organism.context_id)
    candidates.append(DEFAULT_ASSISTANT_NEGATIVE.to_context())
    out = []
    seen_fps: set[str] = set()
    seen_ids = set()
    for ctx in candidates:
        fp = _context_content_fingerprint(ctx)
        if fp == source_fp or fp in seen_fps or ctx.context_id in seen_ids:
            # Content-identity dedup (panel-first precedence by list order). The
            # fp == source_fp drop can only hit a CONTEXTS alias of the SOURCE
            # itself (the same measurement cell — correctly excluded): a panel
            # member content-identical to the source is refused at ModelOrganism
            # construction (r2 BLOCKER fix), so this branch never masks one.
            continue
        seen_fps.add(fp)
        seen_ids.add(ctx.context_id)
        out.append((ctx, _is_trained_negative(ctx)))
    return out


def _transfer_fraction(
    bystander_delta: float | None, source_delta: float | None
) -> tuple[float | None, str | None]:
    """Margin-space transfer fraction with the resolution floor + sign guard (v2 — S2/S4)."""
    if source_delta is None or bystander_delta is None:
        return None, "companion margin not computed"
    if source_delta >= MIN_SOURCE_MARGIN_DELTA:
        return bystander_delta / source_delta, None
    if source_delta <= -MIN_SOURCE_MARGIN_DELTA:
        return None, "source margin delta negative — install did not register in margin space"
    return None, "source margin delta below resolution floor"


def _resolve_primary_predicate(behavior: Behavior) -> Callable[[str], bool] | None:
    """DV router: None for judged_rate; the deterministic predicate for structural.

    The Phase-1 seams (marker three-space contract, fact recall, ground-truth
    accuracy) raise :class:`UnsupportedOrganismError`; a structural behavior
    without a registered predicate raises ``KeyError`` — fail-fast, never a
    silent skip.
    """
    primary = behavior.dv.primary
    if primary == "judged_rate":
        return None
    if primary == "structural":
        try:
            return _STRUCTURAL_PREDICATES[behavior.name]
        except KeyError:
            raise KeyError(
                f"no structural predicate registered for behavior {behavior.name!r} "
                "(datagen._STRUCTURAL_PREDICATES) — fail-fast; Phase 1 fills"
            ) from None
    if primary == "marker_slot_stats":
        raise UnsupportedOrganismError(
            "marker verify keeps its own three-space log-prob contract "
            "(marker-leakage-measurement.md) served by "
            "eval_battery.run_marker_and_capability_phase + "
            "eval.marker_logprob.compute_marker_slot_stats — the Phase-1 extension seam; "
            "it must not be squeezed into this judged-rate harness"
        )
    if primary in ("fact_recall_5way", "ground_truth_accuracy"):
        raise UnsupportedOrganismError(
            f"primary DV {primary!r} is not implemented in the v1 harness "
            "(fact recall needs the taught-fact 5-way judge; ground_truth_accuracy needs "
            "benchmark answer keys) — Phase 1 extension seams"
        )
    # Registry drift — fail fast, never a silent skip.
    raise UnsupportedOrganismError(f"unknown primary DV {primary!r}")


def _companion_status_for(companion: str | None) -> str:
    """Companion router (v2): machine-readable skip states, fail-fast on drift."""
    if companion == "tf_margin":
        return "computed"
    if companion is None:
        return _COMPANION_SKIPPED_NONE
    if companion == "judged_spotcheck":
        return _COMPANION_UNIMPLEMENTED_SPOTCHECK
    raise UnsupportedOrganismError(
        f"companion DV {companion!r} is not implemented in the v1 harness"
    )


def _resolve_margin_pools(
    companion_status: str,
    margin_pools: tuple[list[dict], list[dict]] | None,
    datagen_dir: Path | str | None,
) -> tuple[list[dict] | None, list[dict] | None, dict[str, Any] | None]:
    """(pos_pairs, neg_pairs, pool_provenance) — derived ONCE, before the side loop."""
    if companion_status != "computed":
        return None, None, None
    sources: dict[str, str] | None = None
    if margin_pools is not None:
        pos_pairs, neg_pairs = margin_pools
        pool_source = "caller-provided margin_pools"
    elif datagen_dir is not None:
        d = Path(datagen_dir)
        pos_pairs, neg_pairs = derive_margin_pools(d)
        pool_source = str(d)
        # r2 minor: the raw sidecar paths the derivation joined, so downstream
        # can audit pool composition back to the exact datagen artifacts.
        sources = {
            "raw_pos": str(d / "raw_pos.jsonl"),
            "raw_neg": str(d / "raw_neg.jsonl"),
            "judge_rows": str(d / "judge_rows.jsonl"),
        }
    else:
        raise ValueError(
            "companion 'tf_margin' requires fixed pools: pass margin_pools explicitly "
            "or datagen_dir (the build's datagen out_dir) for derive_margin_pools"
        )
    pool_provenance = {
        "pool_source": pool_source,
        "sources": sources,  # None for caller-provided pools
        "n_pos": len(pos_pairs),
        "n_neg": len(neg_pairs),
        "pools_sha256": _sha256_text(
            json.dumps([pos_pairs, neg_pairs], sort_keys=True, ensure_ascii=False)
        ),
    }
    return pos_pairs, neg_pairs, pool_provenance


def verify_organism(
    organism: ModelOrganism,
    adapter_or_ckpt: str,
    *,
    out_dir: Path | str,
    datagen_dir: Path | str | None = None,
    margin_pools: tuple[list[dict], list[dict]] | None = None,
    bystander_context_ids: Sequence[str] | None = None,
    eval_questions: Sequence[str] | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    n_completions: int = DEFAULT_EVAL_COMPLETIONS_PER_QUERY,
    temperature: float = DEFAULT_EVAL_TEMPERATURE,
    n_judge_draws: int = DEFAULT_N_JUDGE_DRAWS,
    leakage_bound: float = DEFAULT_LEAKAGE_BOUND,
    generate_fn: GenFn | None = None,
    judge_fn: JudgeFn = judge_graded,
    margin_read_fn: MarginReadFn | None = None,
) -> OrganismReport:
    """Install check + per-bystander leakage panel + dual-DV companion.

    See :class:`OrganismReport` for the statistical-hygiene contract and the
    ADVISORY status of ``install_ok`` / ``leakage_ok``. The fixed margin pools
    are derived ONCE (from ``margin_pools`` or the build's ``datagen_dir``
    sidecars via :func:`derive_margin_pools`) and passed through the
    ``margin_read_fn`` seam per (side, context) so both sides score the
    IDENTICAL pools.
    """
    out_dir = Path(out_dir)
    behavior = organism.behavior_spec
    if n_completions < 1:
        raise ValueError(f"n_completions must be >= 1, got {n_completions}")

    # 1. DV router + 1b. companion router (v2 — explicit judged_spotcheck skip).
    predicate = _resolve_primary_predicate(behavior)
    companion_status = _companion_status_for(behavior.dv.companion)

    questions = _resolve_eval_questions(behavior, eval_questions)

    # 2. Context set (v2 — MF-1 panel-union + content-identity dedup/labeling).
    bystanders_ctx = _verify_context_set(organism, bystander_context_ids)

    # Fixed margin pools: derived ONCE, before the per-side loop (v2 — S4).
    pos_pairs, neg_pairs, pool_provenance = _resolve_margin_pools(
        companion_status, margin_pools, datagen_dir
    )

    sides: tuple[tuple[str, str | None], ...] = (
        ("trained", adapter_or_ckpt),
        ("base", None),
    )
    all_cells = [(organism.context, None)] + [(c, tn) for c, tn in bystanders_ctx]
    judge_root = out_dir / "judge"

    # r2 (concern gpu-seam-memory-coexistence): three strictly SEQUENTIAL
    # phases so heavyweight GPU residents never coexist. Phase 1 is the only
    # vLLM phase (side-major, so the default single-live-engine seam swaps
    # engines exactly once) and the verify-owned engine is closed in its
    # `finally` BEFORE any margin model loads; phase 2 is judge-API/CPU only;
    # phase 3 (HF margin models) starts only after every verify-owned vLLM
    # engine is down. Caller-injected seams are closed by the CALLER (verify
    # closes only what it created).

    # Phase 1 — generation (vLLM). Each cell persists the moment it returns.
    completions_by_cell: dict[tuple[str, str], list[list[str]]] = {}
    created_default_gen = generate_fn is None
    gen = generate_fn if generate_fn is not None else _default_vllm_generate_fn(base_model)
    try:
        for side, side_path in sides:
            for ctx, _tn in all_cells:
                completions_by_cell[(side, ctx.context_id)] = _generate_and_persist(
                    gen,
                    side,
                    side_path,
                    ctx,
                    questions,
                    n=n_completions,
                    temperature=temperature,
                    out_dir=out_dir,
                    base_model=base_model,
                )
    finally:
        close = getattr(gen, "close", None)
        if created_default_gen and callable(close):
            close()

    # Phase 2 — rates (judge API / structural predicate; no GPU).
    rates: dict[tuple[str, str], _CellRate] = {}
    for side, _side_path in sides:
        for ctx, _tn in all_cells:
            rates[(side, ctx.context_id)] = _rate_for_cell(
                behavior,
                predicate,
                judge_fn,
                n_judge_draws,
                side,
                ctx,
                questions,
                completions_by_cell[(side, ctx.context_id)],
                judge_root,
            )

    # Phase 3 — margins (HF models; the default seam is created ONLY here,
    # after phase 1's finally tore down every verify-owned vLLM engine).
    margins: dict[tuple[str, str], float] = {}
    if companion_status == "computed":
        created_default_margin = margin_read_fn is None
        margin_fn = (
            margin_read_fn if margin_read_fn is not None else _default_margin_read_fn(base_model)
        )
        try:
            for side, side_path in sides:
                for ctx, _tn in all_cells:
                    mr = margin_fn(side_path, ctx, pos_pairs, neg_pairs)
                    margins[(side, ctx.context_id)] = float(mr.margin)
        finally:
            margin_close = getattr(margin_fn, "close", None)
            if created_default_margin and callable(margin_close):
                margin_close()

    def _margin(side: str, cid: str) -> float | None:
        return margins.get((side, cid))

    def _delta(cid: str) -> float | None:
        t, b = _margin("trained", cid), _margin("base", cid)
        return None if t is None or b is None else t - b

    src_id = organism.context_id
    rate_trained_c = rates[("trained", src_id)].rate
    rate_base_c = rates[("base", src_id)].rate
    source_margin_delta = _delta(src_id)

    bystander_reads: list[BystanderRead] = []
    for ctx, trained_negative in bystanders_ctx:
        cid = ctx.context_id
        cell_t = rates[("trained", cid)]
        cell_b = rates[("base", cid)]
        margin_delta = _delta(cid)
        fraction, reason = _transfer_fraction(margin_delta, source_margin_delta)
        bystander_reads.append(
            BystanderRead(
                context_id=cid,
                trained_negative=trained_negative,
                rate_trained=cell_t.rate,
                rate_base=cell_b.rate,
                rate_delta=cell_t.rate - cell_b.rate,
                n_scored_trained=cell_t.n_scored,
                n_dropped_trained=cell_t.n_dropped,
                n_scored_base=cell_b.n_scored,
                n_dropped_base=cell_b.n_dropped,
                margin_trained=_margin("trained", cid),
                margin_base=_margin("base", cid),
                margin_delta=margin_delta,
                transfer_fraction=fraction,
                transfer_fraction_undefined_reason=reason,
            )
        )

    held_out = [b for b in bystander_reads if not b.trained_negative]
    bank_json = json.dumps(questions, ensure_ascii=False)
    provenance: dict[str, Any] = {
        "eval_question_bank_sha256": _sha256_text(bank_json),
        "eval_question_count": len(questions),
        "eval_train_question_overlap": len(set(questions) & set(behavior.train_question_bank)),
        "arm_naming_note": (
            'the task body\'s "contra" arm == recipe.ARMS "primary" (neg_ratio=1.0)'
        ),
        "n_completions_per_query": n_completions,
        "temperature": temperature,
        "n_judge_draws": n_judge_draws,
        "base_model": base_model,
        "git_commit": _git_commit_hash(),
        "timestamp_utc": _now_utc(),
    }
    if pool_provenance is not None:
        provenance["margin_pools"] = pool_provenance

    report = OrganismReport(
        organism=asdict(organism),
        adapter_path=adapter_or_ckpt,
        dv_primary=behavior.dv.primary,
        rate_trained_C=rate_trained_c,
        rate_base_C=rate_base_c,
        install_delta=rate_trained_c - rate_base_c,
        install_ok=rate_trained_c > rate_base_c,
        source_margin_trained=_margin("trained", src_id),
        source_margin_base=_margin("base", src_id),
        source_margin_delta=source_margin_delta,
        bystanders=tuple(bystander_reads),
        leakage_bound=leakage_bound,
        leakage_ok=all(b.rate_delta <= leakage_bound for b in held_out),
        companion_status=companion_status,
        judge_drop_telemetry={
            f"{side}:{cid}": {
                "n_scored": cell.n_scored,
                "n_dropped": cell.n_dropped,
                "n_total_draws": cell.n_total_draws,
                "n_dropped_draws": cell.n_dropped_draws,
            }
            for (side, cid), cell in sorted(rates.items())
        },
        provenance=provenance,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / "organism_report.json", asdict(report))
    return report


# ── Production rate_fn factory (dose selection) ───────────────────────────────


def make_source_rate_fn(
    organism: ModelOrganism,
    *,
    out_dir: Path | str,
    base_model: str = DEFAULT_BASE_MODEL,
    eval_questions: Sequence[str] | None = None,
    n_completions: int = DEFAULT_EVAL_COMPLETIONS_PER_QUERY,
    temperature: float = DEFAULT_EVAL_TEMPERATURE,
    n_judge_draws: int = DEFAULT_N_JUDGE_DRAWS,
    generate_fn: GenFn | None = None,
    judge_fn: JudgeFn = judge_graded,
) -> RateFn:
    """A ``ckpt_dir -> source judged rate at C`` closure for dose selection.

    Closes over the verify-side generation + judging machinery for the ONE
    trigger context; per-checkpoint artifacts persist under
    ``out_dir/rate_<ckpt_name>/`` (resume-safe, manifest-keyed on the resolved
    checkpoint path + generation params). This is the production default the
    ``build_organism`` caller constructs explicitly.

    GPU lifecycle (r2 — concern ``gpu-seam-memory-coexistence``; r3 —
    crash-fix #1090): the default generation seam keeps at most ONE live
    engine, and as of r3 ALL LoRA checkpoint rungs SHARE that one engine
    (identical base+enable_lora engine; only the per-rung ``LoRARequest``
    differs) — the ladder builds it once instead of teardown+rebuild per rung
    (the r1 version leaked one ~0.9-HBM engine per rung — OOM at rung 2; the
    r2 version rebuilt per rung, ~2.5 min each plus one orphan-probe exposure
    per teardown). The returned closure exposes ``close()``, which tears down
    the FACTORY-OWNED default engine (a caller-injected ``generate_fn`` is the
    caller's to close); ``build_organism`` calls it after the ladder scoring.
    """
    out_dir = Path(out_dir)
    behavior = organism.behavior_spec
    if n_completions < 1:
        raise ValueError(f"n_completions must be >= 1, got {n_completions}")
    questions = _resolve_eval_questions(behavior, eval_questions)
    predicate = (
        _STRUCTURAL_PREDICATES[behavior.name] if behavior.dv.primary == "structural" else None
    )
    if behavior.dv.primary not in ("judged_rate", "structural"):
        raise UnsupportedOrganismError(
            f"make_source_rate_fn supports judged_rate/structural primaries only, "
            f"got {behavior.dv.primary!r}"
        )
    created_default_gen = generate_fn is None
    gen = generate_fn if generate_fn is not None else _default_vllm_generate_fn(base_model)

    def rate(ckpt_dir: str) -> float:
        cell_dir = out_dir / f"rate_{Path(ckpt_dir).name}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        completions = _generate_and_persist(
            gen,
            "trained",
            ckpt_dir,
            organism.context,
            questions,
            n=n_completions,
            temperature=temperature,
            out_dir=cell_dir,
            base_model=base_model,
        )
        cell = _rate_for_cell(
            behavior,
            predicate,
            judge_fn,
            n_judge_draws,
            "trained",
            organism.context,
            questions,
            completions,
            cell_dir / "judge",
        )
        return cell.rate

    def close() -> None:
        gen_close = getattr(gen, "close", None)
        if created_default_gen and callable(gen_close):
            gen_close()

    rate.close = close  # type: ignore[attr-defined]
    return rate
