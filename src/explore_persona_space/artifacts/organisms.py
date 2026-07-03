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
import os
import random
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from explore_persona_space.artifacts.behavior import BEHAVIORS, Behavior
from explore_persona_space.artifacts.context import CONTEXTS, INSTALLABLE_KINDS, Context
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
    context_id: str  # trigger context C; key into CONTEXTS, kind in INSTALLABLE_KINDS
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
        if ctx.kind not in INSTALLABLE_KINDS:
            raise ValueError(
                f"context {self.context_id!r} has kind {ctx.kind!r}, not installable "
                f"(INSTALLABLE_KINDS={INSTALLABLE_KINDS}); bare contexts are leakage "
                "bystanders, never training sources"
            )
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
        if self.dose is not None:
            if spec.stopping.kind != "checkpoint_and_select":
                raise ValueError(
                    f"dose band override is only valid for checkpoint_and_select stopping; "
                    f"this recipe stops via {spec.stopping.kind!r}"
                )
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
    """One bystander context's leakage read (rates + optional margin companion)."""

    context_id: str
    trained_negative: bool  # slug OR content identity matches a trained panel member
    rate_trained: float
    rate_base: float
    rate_delta: float
    n_scored: int  # denominator AFTER dropping all-draws-dropped completions
    n_dropped: int  # llm-judging rule-9 telemetry
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
    judge_drop_telemetry: dict  # per-(side, context) dropped/total
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
    """Deferred imports for the production HF/PEFT margin path (see above)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.margin import compute_tf_margin

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "compute_tf_margin": compute_tf_margin,
    }


# ── Production default seams (GPU; mocked in every test) ─────────────────────


def _default_vllm_generate_fn(base_model: str) -> GenFn:
    """One vLLM engine per model side, chunked generate, teardown via close().

    Engine kwargs follow the ``run_generation_phase`` precedent
    (``max_lora_rank=64``, ``max_model_len=8192``, ``enable_prefix_caching=True``,
    fixed sampling seed); chunked ≤500 prompts per ``llm.generate`` call
    (gotchas.md large-batch deadlock prevention) with ``use_tqdm=False``.
    """
    deps = _resolve_generation_deps()
    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    engines: dict[str | None, tuple[Any, Any]] = {}  # side_path -> (llm, lora_request | None)

    def _engine_for(side_path: str | None) -> tuple[Any, Any]:
        if side_path in engines:
            return engines[side_path]
        common = {"max_model_len": 8192, "enable_prefix_caching": True, "seed": 0}
        if side_path is None:
            llm, lora_req = deps["LLM"](model=base_model, **common), None
        elif deps["_is_full_model_dir"](side_path):
            llm, lora_req = deps["LLM"](model=side_path, **common), None
        else:
            llm = deps["LLM"](model=base_model, enable_lora=True, max_lora_rank=64, **common)
            lora_req = deps["LoRARequest"]("organism", 1, side_path)
        engines[side_path] = (llm, lora_req)
        return llm, lora_req

    def generate(
        side_path: str | None,
        messages_list: list[list[dict[str, str]]],
        *,
        n: int,
        temperature: float,
    ) -> list[list[str]]:
        llm, lora_req = _engine_for(side_path)
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

    def close() -> None:
        for llm, _ in engines.values():
            deps["teardown_vllm"](llm)
        engines.clear()

    generate.close = close  # type: ignore[attr-defined]
    return generate


def _default_margin_read_fn(base_model: str) -> MarginReadFn:
    """HF model loaded once per side (base; base+PEFT adapter), tf-margin per context."""
    deps = _resolve_margin_deps()
    torch = deps["torch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = deps["AutoTokenizer"].from_pretrained(base_model)
    models: dict[str | None, Any] = {}

    def _model_for(side_path: str | None):
        if side_path in models:
            return models[side_path]
        if side_path is None:
            model = (
                deps["AutoModelForCausalLM"]
                .from_pretrained(base_model, torch_dtype=torch.bfloat16)
                .to(device)
            )
        elif (Path(side_path) / "config.json").exists():  # full model dir
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
        models[side_path] = model
        return model

    def read(
        side_path: str | None,
        ctx: Context,
        pos_pairs: list[dict],
        neg_pairs: list[dict],
    ) -> MarginResult:
        model = _model_for(side_path)
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
) -> tuple[Path, dict[str, int], dict[str, int]]:
    """Final-mix assembly (v2 — MF-2 surplus refusal / bounded-deficit tolerance).

    pos + contrastive-negative + generic rows per ``mix_counts``, seeded
    shuffle; ``train_mix.jsonl`` + ``mix_meta.json`` persist the moment
    assembly completes (checkpoint-per-phase). Returns
    ``(train_mix_path, counts_planned, counts_realized)``.
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
    }
    _atomic_write_json(out_root / "mix_meta.json", mix_meta)
    return train_mix_path, counts, realized


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
) -> BuildResult:
    """datagen -> mix assembly -> recipe -> train -> dose-selected checkpoint.

    ``generic_data_path`` is a prompt-completion JSONL, REQUIRED whenever the
    resolved recipe's ``generic_frac > 0``. ``rate_fn`` is REQUIRED on the
    checkpoint_and_select path (the module ships :func:`make_source_rate_fn`
    as the production default the caller constructs explicitly). Every step
    fails fast; ``train_mix.jsonl`` + ``mix_meta.json`` persist the moment
    assembly completes (checkpoint-per-phase).
    """
    out_root = Path(out_root)
    behavior = organism.behavior_spec
    spec = organism.recipe

    # 1. Carve-out gate.
    if behavior.programmatic:
        raise UnsupportedOrganismError(
            f"behavior {organism.behavior!r} is a programmatic carve-out: datagen refuses "
            "programmatic behaviors by design (0d) and the marker/fact training-mix builders "
            "are not yet promoted into artifacts/ — Phase 1 wires the carve-out build path "
            "(recipe_for(...) already validates the recipe side)"
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
    train_mix_path, counts, realized = _assemble_mix(
        organism, spec, Path(pos_path), Path(cn_path), generic_data_path, out_root
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
            if rate_fn is None:
                raise ValueError(
                    "rate_fn is REQUIRED on the checkpoint_and_select path (ckpt_dir -> "
                    "source judged rate at C); construct the production default via "
                    "make_source_rate_fn(organism, ...)"
                )
            rates_by_step = {step: float(rate_fn(str(d))) for step, d in ckpt_dirs.items()}
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
) -> list[list[str]]:
    """Generate (or resume from disk) the per-question completion lists for one cell.

    Persists ``completions__{side}__{context_id}.json`` the moment generation
    returns (checkpoint-per-phase; these are the raw completions the caller
    uploads under ``raw_completions/verify/`` per the Upload Policy).
    """
    out_path = out_dir / f"completions__{side}__{ctx.context_id}.json"
    if out_path.exists():
        payload = json.loads(out_path.read_text())
        if payload["questions"] != list(questions):
            raise ValueError(
                f"resume mismatch at {out_path}: persisted questions differ from the "
                "current eval set — use a fresh out_dir"
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
    _atomic_write_json(out_path, {"questions": list(questions), "completions": completions})
    return completions


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
) -> tuple[float, int, int]:
    """(rate, n_scored, n_dropped) for one (side, context) cell.

    Structural primary: the deterministic predicate, ZERO judge calls.
    Judged primary: graded scores; a None score (all draws dropped) is dropped
    from the denominator and counted — never coerced (llm-judging rule 9);
    ``rate = #(score > behavior.threshold) / n_scored``.
    """
    flat: list[tuple[str, str, str]] = []
    for i, q in enumerate(questions):
        for j, completion in enumerate(completions[i]):
            flat.append((f"{ctx.context_id}-{side}-q{i:03d}-c{j}", q, completion))
    if predicate is not None:
        n_pos = sum(1 for _iid, _q, c in flat if predicate(c))
        return n_pos / len(flat), len(flat), 0
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
        score = result.scores.get(iid)
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
    return n_pos / n_scored, n_scored, n_dropped


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
            continue  # content-identity dedup (panel-first precedence by list order)
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
    if margin_pools is not None:
        pos_pairs, neg_pairs = margin_pools
        pool_source = "caller-provided margin_pools"
    elif datagen_dir is not None:
        pos_pairs, neg_pairs = derive_margin_pools(datagen_dir)
        pool_source = str(Path(datagen_dir))
    else:
        raise ValueError(
            "companion 'tf_margin' requires fixed pools: pass margin_pools explicitly "
            "or datagen_dir (the build's datagen out_dir) for derive_margin_pools"
        )
    pool_provenance = {
        "pool_source": pool_source,
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

    created_default_gen = generate_fn is None
    gen = generate_fn if generate_fn is not None else _default_vllm_generate_fn(base_model)
    margin_fn = margin_read_fn
    if companion_status == "computed" and margin_fn is None:
        margin_fn = _default_margin_read_fn(base_model)

    sides: tuple[tuple[str, str | None], ...] = (
        ("trained", adapter_or_ckpt),
        ("base", None),
    )
    all_cells = [(organism.context, None)] + [(c, tn) for c, tn in bystanders_ctx]
    rates: dict[tuple[str, str], tuple[float, int, int]] = {}
    margins: dict[tuple[str, str], float] = {}
    judge_root = out_dir / "judge"
    try:
        for side, side_path in sides:
            for ctx, _tn in all_cells:
                completions = _generate_and_persist(
                    gen,
                    side,
                    side_path,
                    ctx,
                    questions,
                    n=n_completions,
                    temperature=temperature,
                    out_dir=out_dir,
                )
                rates[(side, ctx.context_id)] = _rate_for_cell(
                    behavior,
                    predicate,
                    judge_fn,
                    n_judge_draws,
                    side,
                    ctx,
                    questions,
                    completions,
                    judge_root,
                )
                if companion_status == "computed":
                    mr = margin_fn(side_path, ctx, pos_pairs, neg_pairs)
                    margins[(side, ctx.context_id)] = float(mr.margin)
    finally:
        close = getattr(gen, "close", None)
        if created_default_gen and callable(close):
            close()

    def _margin(side: str, cid: str) -> float | None:
        return margins.get((side, cid))

    def _delta(cid: str) -> float | None:
        t, b = _margin("trained", cid), _margin("base", cid)
        return None if t is None or b is None else t - b

    src_id = organism.context_id
    rate_trained_c, _ns_t, _nd_t = rates[("trained", src_id)]
    rate_base_c, _ns_b, _nd_b = rates[("base", src_id)]
    source_margin_delta = _delta(src_id)

    bystander_reads: list[BystanderRead] = []
    for ctx, trained_negative in bystanders_ctx:
        cid = ctx.context_id
        rt, ns_t, nd_t = rates[("trained", cid)]
        rb, _ns_b2, nd_b2 = rates[("base", cid)]
        margin_delta = _delta(cid)
        fraction, reason = _transfer_fraction(margin_delta, source_margin_delta)
        bystander_reads.append(
            BystanderRead(
                context_id=cid,
                trained_negative=trained_negative,
                rate_trained=rt,
                rate_base=rb,
                rate_delta=rt - rb,
                n_scored=ns_t,
                n_dropped=nd_t + nd_b2,
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
            f"{side}:{cid}": {"n_scored": ns, "n_dropped": nd}
            for (side, cid), (_r, ns, nd) in sorted(rates.items())
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
    ``out_dir/rate_<ckpt_name>/`` (resume-safe). This is the production
    default the ``build_organism`` caller constructs explicitly.
    """
    out_dir = Path(out_dir)
    behavior = organism.behavior_spec
    questions = _resolve_eval_questions(behavior, eval_questions)
    predicate = (
        _STRUCTURAL_PREDICATES[behavior.name] if behavior.dv.primary == "structural" else None
    )
    if behavior.dv.primary not in ("judged_rate", "structural"):
        raise UnsupportedOrganismError(
            f"make_source_rate_fn supports judged_rate/structural primaries only, "
            f"got {behavior.dv.primary!r}"
        )
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
        )
        r, _ns, _nd = _rate_for_cell(
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
        return r

    return rate
