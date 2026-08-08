#!/usr/bin/env python
"""#1947 P0 — single-visit fresh datagen driver (plan §4.3/§4.4 P0).

Builds, per behavior {sycophancy(c3-v2), impolite(c2), writing_style(#1434)}:

- ``banks``      extend the #1090/#1434 question banks to >=340 questions per
                 behavior via the SAME pv-shape generation template
                 (``issue1090_questiongen`` machinery; screens preserved);
- ``positives``  300 Claude-written instruct-and-strip positives per
                 (behavior x context) through the factory pipeline
                 (``artifacts.datagen.generate_training_data`` posonly mode:
                 checkpointed, judge-filtered >=50 at 3 draws, topup tranche =
                 the ONE pre-registered retry, floor 300 with a 240-floor
                 salvage per plan §7 gate 2);
- ``negatives``  300 on-policy BASE-model greedy negatives per
                 (behavior x panel-variant) under the factory panel members'
                 own prompts (vLLM, judge-filtered <50) — the #1947 deviation
                 from the factory's Claude-generated negatives (plan §4.3:
                 contrastive-negatives.md on-policy rule);
- ``generic``    the shared 900-row generic pool (400-row issue906 corpus +
                 500 WildChat-question/base-greedy rows — the plan's named
                 fallback rider: no issue906 BUILDER exists in-tree) + the
                 1,344 marker generic questions (1,280 + budget headroom);
- ``marker``     greedy base R maps (4 source contexts + panel members) and
                 the 4 marker single-visit mixes (1,280 pos + 5,120 neg =
                 6,400 rows = 400 steps x batch 16), reusing the
                 ``issue1481_marker`` primitives;
- ``mixes``      the 52 content mixes (48 single-visit 1,200-row + 4
                 repeat-regime 80-row), budget-gated at the POOL level so the
                 final counts are EXACT, each with a builder-predicted
                 ``consumption_manifest.json`` (plan §4.2);
- ``upload``     fail-loud bulk ``upload_folder`` of mixes + manifests + all
                 datagen rollout text (rejected candidates included) to
                 ``issue1947_singlevisit/`` (persist-before-reduce).

Every phase is resumable (skip-if-output-exists with row-count checks) and
sliceable (``--smoke`` / ``--limit``); ``--mock-gen`` swaps the API/vLLM
seams for deterministic stubs (CPU smoke). GPU phases construct ONE vLLM
engine per phase (chunked greedy via the reused #1481 marker backend).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps before any heavy import

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

# vLLM v1 EngineCore fork-poisoning guard (gotchas.md #628): BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1090_questiongen as qg  # noqa: E402
import issue1947_cells as cells  # noqa: E402

from explore_persona_space.artifacts import banks as banks_mod  # noqa: E402
from explore_persona_space.artifacts.behavior import BEHAVIORS, Behavior  # noqa: E402
from explore_persona_space.artifacts.context import (  # noqa: E402
    CONTEXTS,
    Context,
    icl_prefix_context,
)
from explore_persona_space.artifacts.datagen import (  # noqa: E402
    NEGATIVE,
    POSITIVE,
    DatagenYieldError,
    GenCandidate,
    GenRequest,
    TopupSpec,
    _judge_and_filter,
    _read_raw,
    generate_training_data,
)
from explore_persona_space.artifacts.negatives import (  # noqa: E402
    default_panel,
)
from explore_persona_space.artifacts.organisms import enforce_mix_token_budget  # noqa: E402

logger = logging.getLogger("issue1947.datagen")

# ── Constants (plan §4.3/§11) ────────────────────────────────────────────────

BEH_KEYS = cells.BEH_KEYS
BEHAVIOR_BY_KEY = cells.BEHAVIOR_BY_KEY
CTX_KEYS = cells.CTX_KEYS

N_NEW_QUESTIONS = 300  # bank extension target beyond the pinned 40 (>=340 total)
BANK_MAX_TEMPLATE_CALLS = 15  # 40 q/call; >=8 calls needed, cap the loop
POS_TARGET_N = 375  # ceil(0.8 * 375) = 300 -> the factory emits EXACTLY 300
POS_QUOTA_FLOOR = 0.8
POS_EMIT_N = 300  # plan §4.3 "equalize-down to exactly 300"
POS_FLOOR_HARD = 240  # plan §7 gate 2: >=240/300 after one retry tranche
N_JUDGE_DRAWS = 3  # plan §11 judge bundle
# Oversample (plan assumption 2): base budget ceil(375/0.7)=536; mult 1.4 ->
# ~750 requests = 2.5x the 300-row target. Bare cells budget up to 12x total
# via the topup tranche (12*300 - 750 = 2850).
OVERSAMPLE_MULT = 1.4
TOPUP_TRANCHE_DEFAULT = 536
TOPUP_TRANCHE_BARE = 2850
NEG_TOTAL = 300  # per (behavior x panel-variant) pool (1:1 with positives)
NEG_Q_NONBARE, NEG_MEMBERS_NONBARE = 60, 5  # 60 q x 5 members (plan §4.3)
NEG_Q_BARE, NEG_MEMBERS_BARE = 75, 4  # bare cells drop the default member
NEG_Q_OVERSAMPLE = 1.25  # judge-filter (<50) headroom on the question axis
GENERIC_POOL_N = 900  # 400 corpus + 500 same-recipe-class extension
GENERIC_EXT_N = 500
MARKER_Q_N = 1344  # initial harvest: 1,280 + 5% budget-gate headroom (adaptive top-up below)
MARKER_POS = cells.MARKER_POS  # 1,280
MARKER_NEG = cells.MARKER_NEG  # 5,120 (1:4 marker-line ratio, named deviation)
# Crash-fix r7 (P0 crash 3): the icl two-shot demo block inflates rendered row
# lengths, so a FIXED 5% headroom is context-dependent (icl dropped 8.6%). The
# marker question pool is now ADAPTIVE: bounded top-up tranches until every
# context clears MARKER_POS budget-surviving positives, hard-capped at 2x.
MARKER_Q_HARD_CAP = 2 * MARKER_POS  # 2,560 — adaptive top-up ceiling
MARKER_Q_TOPUP_MIN = 64  # smallest tranche (survivor headroom over the deficit)
MAX_LENGTH = 2048  # trainer max_length -> the token-budget gate's budget
GEN_MAX_NEW_TOKENS = 1024  # greedy base R cap (CLAUDE.md free-generation default)
WILDCHAT_DATASET = "allenai/WildChat-1M"
WILDCHAT_STREAM_CAP = 400_000  # TOTAL-streamed-rows cap (#1092 bounded-stream rule)
WILDCHAT_KEPT_TARGET = 2_100  # 500 generic + 1,344 marker + reserve
DATAGEN_SEED = 1947
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
MARKER_EVAL_BANK_SHA256 = "7c08c15bea17f750d0c74f6e3d484644e4c3e570f157a9686653f8c0b12f6c46"

_QG_ASSET_BY_KEY = {
    "syc": _SCRIPTS_DIR / "issue1090_assets/questiongen_sycophancy_v2.json",
    "imp": _SCRIPTS_DIR / "issue1090_assets/questiongen_impolite.json",
    "cas": _SCRIPTS_DIR / "issue1434_assets/questiongen_writing_style.json",
}

PANEL_VARIANTS = ("panel5", "panel4bare")  # non-bare cells | bare cells


def panel_for_variant(variant: str):
    """Factory negative panel per variant: the default_v1 panel (5 members incl.
    the default assistant) for non-bare cells; bare cells drop the default
    member — it is that arm's SOURCE (#527/#538 disjointness)."""
    panel = default_panel()
    if variant == "panel4bare":
        panel = tuple(m for m in panel if m.identity != "default")
    n_members = {"panel5": NEG_MEMBERS_NONBARE, "panel4bare": NEG_MEMBERS_BARE}[variant]
    assert len(panel) == n_members, (variant, [m.slug for m in panel])
    return panel


def panel_variant_for_ctx(ctx_key: str) -> str:
    return "panel4bare" if ctx_key == "bare" else "panel5"


# ── Small utils ──────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    """Unicode-line-boundary-safe JSONL read (never .splitlines(); gotchas.md)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _qsha(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _jsonable(obj):
    """Recursively JSON-encodable copy (stringifies non-scalar mapping keys —
    _ArmDrops counters can carry tuple keys)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _canonical_sha(items: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(items), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _train_row(prompt_msgs: list[dict], assistant: str) -> dict:
    """The issue664_common train-row shape the whole factory lineage consumes."""
    return {
        "prompt": [dict(m) for m in prompt_msgs],
        "completion": [{"role": "assistant", "content": assistant}],
    }


_TOKENIZER = None


def _tokenizer():
    """Module-cached Qwen tokenizer (never per-row from_pretrained; gotchas.md)
    with the in-process marker token-id assert (marker-leakage-measurement.md)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        ids = _TOKENIZER.encode(" ※", add_special_tokens=False)
        assert ids == [83399], f"marker token drift: encode(' ※') == {ids} != [83399]"
    return _TOKENIZER


# ── Config ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    """Driver config (one out_root per run; phases checkpoint under it)."""

    out_root: Path
    phases: tuple[str, ...]
    behaviors: tuple[str, ...]
    contexts: tuple[str, ...]
    smoke: bool = False
    mock_gen: bool = False
    limit: int | None = None
    negatives_extra: int = 0  # retry tranche: extra negative questions per pool
    generic_extra: int = 0  # raise the generic extension beyond 500 (equalize-down)

    @property
    def banks_dir(self) -> Path:
        return self.out_root / "banks"

    @property
    def positives_dir(self) -> Path:
        return self.out_root / "positives"

    @property
    def negatives_dir(self) -> Path:
        return self.out_root / "negatives"

    @property
    def generic_dir(self) -> Path:
        return self.out_root / "generic"

    @property
    def marker_dir(self) -> Path:
        return self.out_root / "marker"

    @property
    def mixes_dir(self) -> Path:
        return self.out_root / "mixes"

    @property
    def raw_dir(self) -> Path:
        return self.out_root / "raw_completions" / "datagen"

    def pos_target(self) -> int:
        return 5 if self.smoke else POS_TARGET_N

    def n_new_questions(self) -> int:
        return 12 if self.smoke else N_NEW_QUESTIONS


def _phase_line(name: str, **kv) -> None:
    extras = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[phase={name}] {extras}".rstrip(), flush=True)


# ── Contexts ─────────────────────────────────────────────────────────────────


def register_contexts() -> None:
    """Idempotent registration of every dynamic context this driver resolves
    (subprocess-registry rule, gotchas.md #1090-fu6): the WildChat conv prefix
    + the per-behavior ICL prefixes (committed 2-shot banks)."""
    fu3_cells.register_fu3_contexts()
    for behavior in BEHAVIOR_BY_KEY.values():
        cid = f"icl_prefix_{behavior}"
        if cid not in CONTEXTS:
            CONTEXTS[cid] = icl_prefix_context(behavior)


def source_context(beh_key: str, ctx_key: str) -> Context:
    """Training context per (behavior, ctx_key) — the factory universe (#1481)."""
    register_contexts()
    if ctx_key == "pers":
        return CONTEXTS["persona_software_engineer"]
    if ctx_key == "bare":
        return CONTEXTS["default"]
    if ctx_key == "conv":
        return CONTEXTS[fu3_cells.CONV_CONTEXT_ID]
    if ctx_key == "icl":
        return CONTEXTS[f"icl_prefix_{BEHAVIOR_BY_KEY[beh_key]}"]
    raise ValueError(f"unknown ctx_key {ctx_key!r}")


# ── Mock seams (CPU smoke only) ──────────────────────────────────────────────

_EXHIBIT_TOKEN = "[[EXHIBIT]]"


def _mock_generate_fn(reqs: list[GenRequest]) -> list[GenCandidate]:
    """Deterministic stub generator: exhibit-shaped for positives, plain
    otherwise (the mock judge keys on the token)."""
    out = []
    for r in reqs:
        tag = _EXHIBIT_TOKEN if r.arm == "positive" else ""
        out.append(
            GenCandidate(
                request=r, completion=f"{tag} A stub answer about: {r.question[:80]}".strip()
            )
        )
    return out


def _mock_judge_fn(items, eval_prompt, *, n_draws, cache_dir, save_raw, **kw):
    """judge_graded-signature stub: 85 for exhibit-shaped completions, 15 else."""
    import dataclasses as _dc

    from explore_persona_space.eval.graded_judge import JudgeResult

    scores = {rid: (85.0 if _EXHIBIT_TOKEN in comp else 15.0) for rid, _q, comp in items}
    per_item = {rid: [s] * n_draws for rid, s in scores.items()}
    kwargs: dict = {}
    for f in _dc.fields(JudgeResult):
        if f.default is not _dc.MISSING or f.default_factory is not _dc.MISSING:  # type: ignore[misc]
            continue
        if f.type.startswith("int") or f.name.startswith("n_"):
            kwargs[f.name] = 0
        else:
            kwargs[f.name] = {}
    kwargs["scores"] = scores
    kwargs["per_item_scores"] = per_item
    if "n_total_draws" in {f.name for f in _dc.fields(JudgeResult)}:
        kwargs["n_total_draws"] = len(items) * n_draws
    Path(save_raw).parent.mkdir(parents=True, exist_ok=True)
    Path(save_raw).write_text(json.dumps({"mock": True, "n_items": len(items)}) + "\n")
    return JudgeResult(**kwargs)


# ── vLLM backend (GPU phases; ONE engine per phase) ─────────────────────────


class _MockBackend:
    def generate(self, prompts: list[str], max_new: int, *, adapter_dir=None) -> list[str]:
        return [f"Stub greedy response {i} for smoke." for i in range(len(prompts))]

    def close(self, label: str) -> None:  # noqa: ARG002
        return


def _backend(cfg: Cfg):
    """The chunked-greedy vLLM backend (reused #1481 marker class) or the mock."""
    if cfg.mock_gen:
        return _MockBackend()
    import issue1481_marker as mk  # deferred: vLLM-adjacent import chain

    return mk.VllmBackend(enable_lora=False)


# ── Phase: banks ─────────────────────────────────────────────────────────────


def _trait_spec(beh_key: str) -> qg.TraitSpec:
    """A questiongen TraitSpec rebuilt from the behavior's committed artifact
    (same template inputs the #1090/#1434 banks were generated with)."""
    asset = json.loads(_QG_ASSET_BY_KEY[beh_key].read_text(encoding="utf-8"))
    lexicon = tuple(asset.get("screen_lexicon") or ())
    return qg.TraitSpec(
        trait=asset["trait"],
        description=asset["description"],
        gen_description=asset.get("trait_instruction_input") or None,
        bank_file=asset["bank_file"],
        lexicon=lexicon,
        false_claim_screen=bool(asset.get("screen_false_claim")),
    )


def _existing_bank(spec: qg.TraitSpec) -> list[str]:
    path = qg.BANKS_DIR / spec.bank_file
    qs = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(qs, list) and len(qs) == 40, (spec.bank_file, len(qs))
    return [str(q) for q in qs]


def _mock_template_response(spec: qg.TraitSpec, attempt: int) -> str:
    # Neutral text (never the trait name — the lexicon screen runs on mocks too).
    tag = hashlib.sha256(spec.trait.encode()).hexdigest()[:6]
    qs = [f"Mock bank question {tag} a{attempt} n{i:02d}?" for i in range(qg.N_QUESTIONS)]
    return json.dumps(
        {
            "instruction": [{"pos": "exhibit", "neg": "avoid"} for _ in range(qg.N_PAIRS)],
            "questions": qs,
            "eval_prompt": "mock eval prompt",
        }
    )


def _parse_generation_tolerant(raw: str, behavior: str) -> dict | None:
    """Per-call parse tolerance for the ~15x repeated template calls (crash-fix r4).

    The parent parser (``issue1090_questiongen.parse_generation``) hard-asserts
    EXACTLY ``N_QUESTIONS`` — its exact-40 contract belongs to #1090 and stays
    untouched. Under repeated verbatim template calls, occasional over-count
    generations (41+) are expected model noise and are salvageable: CLIP to the
    first ``N_QUESTIONS`` and re-validate through the parent parser (so every
    other parent check — instruction pairs, empty/duplicate questions,
    eval_prompt — still runs on the clipped payload). Any OTHER parse failure
    (under-count, malformed JSON, top-level non-dict JSON, duplicates) is a
    FAILED generation call: return None so the caller's oversample/retry-tranche
    machinery absorbs it (the bank floor still fails loud after the budgeted
    tranches). AttributeError is caught alongside ValueError because the parent
    parser calls ``obj.get(...)`` on the decoded payload, so a valid-JSON
    non-dict generation (bare array / string / null) escapes as AttributeError
    (concern parse-tolerance-nondict-json-escape, r4 code review).
    """
    try:
        return qg.parse_generation(raw)
    except (ValueError, AttributeError) as err:
        try:
            obj = json.loads(qg._strip_fence(raw))
        except json.JSONDecodeError:
            logger.warning("[banks] %s: rejected generation call (%s)", behavior, err)
            return None
        if not isinstance(obj, dict):
            logger.warning(
                "[banks] %s: rejected generation call (top-level %s, not a JSON object)",
                behavior,
                type(obj).__name__,
            )
            return None
        questions = obj.get("questions")
        if isinstance(questions, list) and len(questions) > qg.N_QUESTIONS:
            n_raw = len(questions)
            obj["questions"] = questions[: qg.N_QUESTIONS]
            try:
                gen = qg.parse_generation(json.dumps(obj, ensure_ascii=False))
            except ValueError as err2:
                logger.warning(
                    "[banks] %s: rejected generation call after clip (%s)", behavior, err2
                )
                return None
            logger.info(
                "[banks] %s: clipped over-count generation: %d -> %d",
                behavior,
                n_raw,
                qg.N_QUESTIONS,
            )
            return gen
        logger.warning("[banks] %s: rejected generation call (%s)", behavior, err)
        return None


def _forbidden_questions() -> set[str]:
    """The assumption-10 exclusion set: every held-out eval-bank question + the
    marker training bank + the 20-question persona-panel eval set, stripped —
    EXACTLY the normalization the phase_banks structural assert applies over
    new_qs (which are stripped at collection). Shared by the collection-time
    overlap filter in _extend_bank AND the post-hoc _bank_structural_asserts
    invariant, so the two can never drift (crash-fix r6: the P0 crash was a
    post-hoc assert with no collection-time filter — expected collisions under
    ~15x verbatim template volume crashed the phase instead of being dropped
    and backfilled from the oversample surplus)."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_QUESTIONS_20,
    )

    marker_train = json.loads(
        (qg.BANKS_DIR / "issue1481_marker_train10_v1.json").read_text(encoding="utf-8")
    )
    forbidden = {q.strip() for q in EVAL_QUESTIONS_20} | {q.strip() for q in marker_train}
    for behavior in BEHAVIOR_BY_KEY.values():
        forbidden |= {q.strip() for q in banks_mod.bank_slice(behavior, "eval")}
    return forbidden


def _extend_bank(cfg: Cfg, beh_key: str) -> dict:
    """Extend one behavior's bank to >= 40 + n_new unique screened questions via
    repeated verbatim pv-template calls (dedup across calls; screens per call;
    per-call parse tolerance via _parse_generation_tolerant; eval/marker-bank
    collisions dropped at collection time and backfilled — crash-fix r6)."""
    spec = _trait_spec(beh_key)
    behavior = BEHAVIOR_BY_KEY[beh_key]
    out_path = cfg.banks_dir / f"{behavior}_extended.json"
    n_new_target = cfg.n_new_questions()
    forbidden = _forbidden_questions()
    if out_path.exists():
        rec = json.loads(out_path.read_text(encoding="utf-8"))
        stale_overlap = sorted({q.strip() for q in rec["new_questions"]} & forbidden)
        if len(rec["new_questions"]) >= n_new_target and not stale_overlap:
            logger.info("[banks] %s: extension exists (%d new) — skip", behavior, n_new_target)
            return rec
        if stale_overlap:
            # Resume revalidation (crash-fix r6): banks are written BEFORE the
            # phase-level structural asserts run, so a crashed run leaves bank
            # files that a bare count-only resume would consume and re-crash on.
            logger.warning(
                "[banks] %s: resume-loaded extension has %d eval/marker collisions "
                "— regenerating with the overlap filter",
                behavior,
                len(stale_overlap),
            )
    existing = _existing_bank(spec)
    seen = {q.strip() for q in existing}
    new_qs: list[str] = []
    attempts_used = 0
    parse_rejects = 0
    overlap_dropped: set[str] = set()
    screen_log: list[dict] = []
    cache_root = cfg.banks_dir / "cache" / behavior
    for attempt in range(1, BANK_MAX_TEMPLATE_CALLS + 1):
        if len(new_qs) >= n_new_target:
            break
        attempts_used = attempt
        if cfg.mock_gen:
            raw = _mock_template_response(spec, attempt)
        else:
            raw = qg._dispatch_one(spec, 1000 + attempt, cache_root)
        gen = _parse_generation_tolerant(raw, behavior)
        if gen is None:
            parse_rejects += 1
            screen_log.append({"attempt": attempt, "parse_reject": True})
            continue
        violations = qg.screen_questions(spec, gen["questions"])
        if spec.false_claim_screen and not cfg.mock_gen:
            violations = violations + qg.screen_false_claims(
                spec, gen["questions"], attempt=1000 + attempt, cache_root=cache_root
            )
        bad_idx = {v["index"] for v in violations}
        attempt_overlap = 0
        for i, q in enumerate(gen["questions"]):
            q = q.strip()
            if i in bad_idx or not q or q in seen:
                continue
            if q in forbidden:
                # Collision with a held-out eval/marker bank (expected under
                # ~15x verbatim template volume): drop + count; the tranche /
                # oversample loop backfills. The :587-class structural assert
                # stays the unchanged post-hoc invariant.
                overlap_dropped.add(q)
                attempt_overlap += 1
                continue
            seen.add(q)
            new_qs.append(q)
        screen_log.append(
            {
                "attempt": attempt,
                "n_violations": len(violations),
                "overlap_rejects": attempt_overlap,
            }
        )
        if attempt_overlap:
            logger.info(
                "[banks] %s: overlap-rejected %d candidate question(s) colliding with "
                "eval/marker banks (attempt %d)",
                behavior,
                attempt_overlap,
                attempt,
            )
    if len(new_qs) < n_new_target:
        raise RuntimeError(
            f"[banks] {behavior}: only {len(new_qs)} unique screened new questions after "
            f"{attempts_used} template calls (target {n_new_target}, "
            f"{parse_rejects} parse-rejected calls, "
            f"{len(overlap_dropped)} overlap-rejected questions) — raise "
            f"BANK_MAX_TEMPLATE_CALLS or investigate screen rejections: {screen_log}"
        )
    new_qs = new_qs[:n_new_target]
    rec = {
        "behavior": behavior,
        "beh_key": beh_key,
        "bank_file": spec.bank_file,
        "existing_sha256_canonical": _canonical_sha(existing),
        "existing_n": len(existing),
        "new_questions": new_qs,
        "new_sha256_canonical": _canonical_sha(new_qs),
        "template_sha256": hashlib.sha256(qg.template_text().encode("utf-8")).hexdigest(),
        "attempts_used": attempts_used,
        "parse_rejects": parse_rejects,
        "overlap_rejects": len(overlap_dropped),
        "screen_log": screen_log,
        "mock": cfg.mock_gen,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(out_path, rec)
    logger.info(
        "[banks] %s: %d new questions in %d calls (%d parse-rejected, %d overlap-rejected)",
        behavior,
        len(new_qs),
        attempts_used,
        parse_rejects,
        len(overlap_dropped),
    )
    return rec


def new_questions(cfg: Cfg, beh_key: str) -> list[str]:
    behavior = BEHAVIOR_BY_KEY[beh_key]
    rec = json.loads((cfg.banks_dir / f"{behavior}_extended.json").read_text(encoding="utf-8"))
    return list(rec["new_questions"])


def _bank_structural_asserts(cfg: Cfg) -> dict:
    """Plan assumption 10 — FULL-grain structural asserts on the realized banks:
    uniqueness, disjointness vs every eval bank + the marker banks, and the
    rendered-question token budget (>=90% keep). The forbidden set is the SAME
    _forbidden_questions() the collection-time filter applies (crash-fix r6),
    so the disjointness assert should now never fire — if it does, that is a
    real bug (filter/assert drift) and the phase must still crash."""
    tok = _tokenizer()
    forbidden = _forbidden_questions()
    stats: dict[str, dict] = {}
    for beh_key in cfg.behaviors:
        behavior = BEHAVIOR_BY_KEY[beh_key]
        spec = _trait_spec(beh_key)
        existing = _existing_bank(spec)
        new_qs = new_questions(cfg, beh_key)
        full = [q.strip() for q in existing + new_qs]
        assert len(set(full)) == len(full), f"{behavior}: duplicate questions in realized bank"
        overlap = sorted(set(q for q in new_qs) & forbidden)
        assert not overlap, f"{behavior}: {len(overlap)} new questions overlap eval/marker banks"
        budget = MAX_LENGTH - GEN_MAX_NEW_TOKENS
        lens = []
        for q in new_qs:
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
            lens.append(len(tok.encode(rendered, add_special_tokens=False)))
        keep = sum(1 for n in lens if n <= budget) / max(1, len(lens))
        assert keep >= 0.90, f"{behavior}: only {keep:.2%} of new questions under {budget} tokens"
        stats[behavior] = {
            "n_total": len(full),
            "n_new": len(new_qs),
            "token_budget": budget,
            "keep_frac_at_budget": keep,
            "max_question_tokens": max(lens) if lens else 0,
        }
    _write_json(cfg.banks_dir / "bank_stats.json", {"assumption10": stats})
    return stats


def phase_banks(cfg: Cfg) -> None:
    _phase_line("banks", behaviors=",".join(cfg.behaviors), mock=cfg.mock_gen)
    for beh_key in cfg.behaviors:
        _extend_bank(cfg, beh_key)
    stats = _bank_structural_asserts(cfg)
    _phase_line("banks_done", stats=json.dumps({k: v["n_total"] for k, v in stats.items()}))


# ── Phase: positives ─────────────────────────────────────────────────────────


def _pool_seed(beh_key: str, ctx_key: str) -> int:
    return DATAGEN_SEED * 100 + BEH_KEYS.index(beh_key) * 10 + CTX_KEYS.index(ctx_key)


def _behavior_1947(cfg: Cfg, beh_key: str) -> Behavior:
    """The registered behavior with its train bank swapped for the 300 new
    extended-bank questions (eval slices untouched — they stay the pinned
    [20,40) rows of the committed 40-q bank)."""
    qs = new_questions(cfg, beh_key)
    return dataclasses.replace(BEHAVIORS[BEHAVIOR_BY_KEY[beh_key]], train_question_bank=tuple(qs))


def _replay_judge_fn(save_raw_path: Path):
    """A judge_graded-signature stub that REBUILDS the JudgeResult for ``items``
    from a persisted ``save_raw`` file (PURE READ — zero API calls;
    ``graded_judge.judge_result_from_save_raw``). Lets ``_judge_and_filter``
    re-derive a kept set through the factory's OWN keep rule offline."""
    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    def _fn(items, eval_prompt, **kw):  # signature parity with judge_graded
        return judge_result_from_save_raw(save_raw_path, items)

    return _fn


def _salvage_reconstruct_union(
    behavior: Behavior, out_dir: Path
) -> tuple[list[GenCandidate], dict]:
    """Reconstruct the post-topup kept-positive UNION from the persisted
    sidecars (``raw_pos[_topup].jsonl`` + ``judge_raw_pos[_topup].json``)
    through the factory's own keep rule (``_judge_and_filter`` over a save_raw
    replay) and the factory's own question_id-dedup merge (crash-fix r8).

    The pre-r8 reconstruction read ``judge_rows.jsonl`` — a sidecar the factory
    writes only AFTER the positive floor check passes, so it never exists in
    exactly the floor-missed state the salvage serves (silent ``return 0`` =
    no ``pos.jsonl``, the P0-crash-4 latent state) — and its topup globs
    (``*topup*judge_rows*``) matched neither actual sidecar name. Raises
    RuntimeError when the first-sample sidecars are missing: the recorded
    state is then NOT reconstructible — fail loud, never an empty pool.
    """
    raw_path = out_dir / "raw_pos.jsonl"
    save_raw = out_dir / "judge_raw_pos.json"
    missing = [str(p) for p in (raw_path, save_raw) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"[positives] salvage: required first-sample sidecars missing: {missing} "
            f"(out_dir={out_dir}) — the recorded floor-miss state is not reconstructible"
        )
    scratch = out_dir / "salvage_replay_scratch"  # replay judge never writes it
    kept_first, _d1, _jr1, _s1 = _judge_and_filter(
        behavior,
        _read_raw(raw_path),
        POSITIVE,
        judge_fn=_replay_judge_fn(save_raw),
        n_judge_draws=N_JUDGE_DRAWS,
        cache_dir=scratch,
        save_raw=scratch / "unused.json",
    )
    kept_topup: list[GenCandidate] = []
    raw_topup = out_dir / "raw_pos_topup.jsonl"
    save_topup = out_dir / "judge_raw_pos_topup.json"
    if raw_topup.exists() and save_topup.exists():
        kept_topup, _d2, _jr2, _s2 = _judge_and_filter(
            behavior,
            _read_raw(raw_topup),
            POSITIVE,
            judge_fn=_replay_judge_fn(save_topup),
            n_judge_draws=N_JUDGE_DRAWS,
            cache_dir=scratch,
            save_raw=scratch / "unused_topup.json",
        )
    # question_id-dedup merge — MIRRORS _positive_topup_stage's union semantics
    # (every first-sample kept row counts; tranche rows dedupe on question_id).
    seen_qids = {c.request.question_id for c in kept_first}
    merged: list[GenCandidate] = []
    for c in kept_topup:
        if c.request.question_id in seen_qids:
            continue
        seen_qids.add(c.request.question_id)
        merged.append(c)
    union = list(kept_first) + merged
    info = {
        "kept_pos_first_sample": len(kept_first),
        "tranche_kept": len(kept_topup),
        "tranche_merged": len(merged),
        "kept_pos_union": len(union),
    }
    return union, info


# The files a salvage reconstruction consumes (crash-fix r9): the record pins
# their identity at write time; a later mismatch is classified against them.
_SALVAGE_INPUT_NAMES = (
    "raw_pos.jsonl",
    "judge_raw_pos.json",
    "raw_pos_topup.jsonl",
    "judge_raw_pos_topup.json",
)
_RECORD_COUNT_KEYS = ("kept_pos_first_sample", "tranche_merged", "kept_pos_union")


def _input_pin(path: Path) -> dict:
    """sha256 + size identity pin of one salvage input file."""
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}


def _current_input_pins(out_dir: Path) -> dict:
    """Identity pins (sha256/size/mtime) for every PRESENT salvage input."""
    pins: dict[str, dict] = {}
    for name in _SALVAGE_INPUT_NAMES:
        p = out_dir / name
        if p.exists():
            pins[name] = {**_input_pin(p), "mtime": p.stat().st_mtime}
    return pins


def _pin_topup_record_inputs(out_dir: Path) -> None:
    """Record-time input pinning (crash-fix r9): stamp ``input_pins`` into a
    just-written ``topup_record.json`` so a later salvage verifies input
    IDENTITY (sha256/size) instead of falling back to mtime evidence. Called
    in phase_positives immediately after the factory stage returns/raises —
    the same process wrote the inputs, so the pins are exact. No-op when no
    record exists. P0 crash 5: crashed relaunch #4 re-judged
    ``judge_raw_pos.json`` ~78 min AFTER the record recorded 200 kept (the
    judge is stochastic at 3 draws / temp 1.0), and the r8 exact-count assert
    then failed deterministically on a legitimately different draw."""
    record_path = out_dir / "topup_record.json"
    if not record_path.exists():
        return
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["input_pins"] = _current_input_pins(out_dir)
    _write_json(record_path, record)


def _salvage_mutation_evidence(out_dir: Path, record: dict) -> list[str]:
    """Salvage input files with evidence of a LEGITIMATE post-record mutation.

    Preferred: sha256/size differing from the record's ``input_pins``
    (crash-fix r9 records). Fallback for legacy pin-less records (the
    pod-1947 syc-icl state): file mtime strictly newer than the on-disk
    record's mtime. A PINNED input that is missing fails loud naming the
    path; an empty return means no evidence (the caller stays fail-loud)."""
    pins = record.get("input_pins")
    if pins:
        changed: list[str] = []
        for name, pin in sorted(pins.items()):
            p = out_dir / name
            if not p.exists():
                raise RuntimeError(
                    f"[positives] salvage: record-pinned input missing: {p} — the recorded "
                    "state is not reconstructible"
                )
            cur = _input_pin(p)
            if cur["sha256"] != pin.get("sha256") or cur["size"] != pin.get("size"):
                changed.append(name)
        return changed
    record_path = out_dir / "topup_record.json"
    if not record_path.exists():
        return []
    record_mtime = record_path.stat().st_mtime
    return [
        name
        for name in _SALVAGE_INPUT_NAMES
        if (out_dir / name).exists() and (out_dir / name).stat().st_mtime > record_mtime
    ]


def _reconcile_salvage_record(
    out_dir: Path, record: dict, info: dict, mismatched: list[str]
) -> list[str]:
    """Classify a reconstructed-vs-recorded count mismatch (crash-fix r9).

    With evidence that a salvage input was re-generated AFTER the record was
    written (a crashed relaunch re-judged it before its one-tranche guard
    refused — both counts are valid draws of the same stochastic judge), the
    ARTIFACTS are ground truth and the record is bookkeeping: accept the
    current artifact set, update ``topup_record.json`` in place with the
    re-derived counts + an audit trail (``record`` is mutated in place too),
    and return the mutated-input names. With NO such evidence the r8
    fail-loud stands verbatim — a mismatch against identical inputs is
    genuine corruption, never a silently different pool.
    ``union_floor_missed`` is carried VERBATIM: the floor adjudication still
    runs on the realized pool at mixes time (no science-gate change)."""
    evidence = _salvage_mutation_evidence(out_dir, record)
    if not evidence:
        detail = ", ".join(
            f"{k} reconstructed {info[k]} != recorded {record.get(k)}" for k in mismatched
        )
        raise RuntimeError(
            f"[positives] salvage reconstruction mismatch vs topup_record.json: {detail} "
            f"(out_dir={out_dir}) — no input-mutation evidence; refusing a silently "
            "different pool"
        )
    prior_union = record.get("kept_pos_union")
    reason = (
        "sha differs from record pin (re-generated after the record)"
        if record.get("input_pins")
        else "newer than record (re-judged by a crashed relaunch)"
    )
    logger.warning(
        "[positives] %s: salvage input %s %s — accepting current artifacts as live truth "
        "(union %s -> %s) and updating the record (audit trail)",
        out_dir.name,
        ",".join(evidence),
        reason,
        prior_union,
        info["kept_pos_union"],
    )
    record.update(info)
    record.update(
        {
            "tranche_dedup_dropped_qid": info["tranche_kept"] - info["tranche_merged"],
            "reconstructed_from_rejudged": True,
            "prior_union": prior_union,
            "new_union": info["kept_pos_union"],
            "rejudged_inputs": evidence,
            "reconstructed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_pins": _current_input_pins(out_dir),
        }
    )
    _write_json(out_dir / "topup_record.json", record)
    return evidence


def _salvage_emit(
    behavior: Behavior, out_dir: Path, emit_n: int, seed: int, record: dict | None = None
) -> int:
    """Plan §7 gate-2 fallback: the factory floor (300) failed post-topup but the
    kept pool may still clear the 240 hard floor — reconstruct the kept-positive
    union from the persisted sidecars and emit min(kept, emit_n) rows. The 240
    hard-floor adjudication itself is UNCHANGED (it stays with the mixes-phase
    yield gate). When a recorded ``topup_record.json`` is passed, the
    reconstruction is VERIFIED against its counts: an exact match proceeds
    (crash-fix r8); a mismatch is classified mutation-aware (crash-fix r9,
    ``_reconcile_salvage_record``) — accepted + audit-recorded when a salvage
    input was legitimately re-generated after the record, fail-loud otherwise."""
    union, info = _salvage_reconstruct_union(behavior, out_dir)
    rejudged: list[str] = []
    if record is not None:
        mismatched = [
            k for k in _RECORD_COUNT_KEYS if record.get(k) is not None and int(record[k]) != info[k]
        ]
        if mismatched:
            rejudged = _reconcile_salvage_record(out_dir, record, info, mismatched)
    take = random.Random(seed).sample(union, min(emit_n, len(union))) if union else []
    rows = [_train_row(c.request.emit_messages, c.completion) for c in take]
    sha = _write_jsonl(out_dir / "pos.jsonl", rows)
    _write_json(
        out_dir / "salvage_meta.json",
        {
            "salvaged": True,
            **info,
            "n_kept_reconstructed": len(union),
            "n_emitted": len(rows),
            "emit_target": emit_n,
            "seed": seed,
            "pos_sha256": sha,
            "record_checked": record is not None,
            "record_reconciled": bool(rejudged),
            "rejudged_inputs": rejudged,
        },
    )
    return len(rows)


def phase_positives(cfg: Cfg) -> None:
    """300 Claude-written instruct-and-strip positives per (behavior x context)
    through the factory posonly pipeline (plan §4.3 row 1; D1 tier-3 provenance
    kept DELIBERATELY — the visit regime is the only variable)."""
    _phase_line("positives", behaviors=",".join(cfg.behaviors), mock=cfg.mock_gen)
    register_contexts()
    results: dict[str, dict] = {}
    for beh_key in cfg.behaviors:
        behavior = _behavior_1947(cfg, beh_key)
        for ctx_key in cfg.contexts:
            pool = f"{beh_key}-{ctx_key}"
            out_dir = cfg.positives_dir / pool
            pos_path = out_dir / "pos.jsonl"
            target = cfg.pos_target()
            emit_n = math.ceil(POS_QUOTA_FLOOR * target)
            if pos_path.exists():
                n = len(_read_jsonl(pos_path))
                logger.info("[positives] %s: pos.jsonl exists (%d rows) — skip", pool, n)
                results[pool] = {"emitted": n, "resumed": True}
                continue
            record_path = out_dir / "topup_record.json"
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if record.get("union_floor_missed"):
                    # Recorded TERMINAL topup verdict (the ONE allowed tranche
                    # consumed + union floor miss): re-apply the recorded salvage
                    # disposition from the sidecars instead of re-entering the
                    # factory stage — whose one-tranche guard raises on re-entry
                    # (crash-fix r8, P0 crash 4: resume predicates honor
                    # terminal-verdict sidecars). Floor adjudication (240 hard
                    # floor) is unchanged — it stays with the mixes yield gate.
                    # A union_floor_missed=false record falls through: factory
                    # re-entry is the designed resume there (raw + cache replay;
                    # the guard only refuses the MISSED-verdict state).
                    n_salvaged = _salvage_emit(
                        behavior, out_dir, emit_n, _pool_seed(beh_key, ctx_key), record=record
                    )
                    results[pool] = {
                        "emitted": n_salvaged,
                        "below_target": True,
                        "resumed_terminal_topup": True,
                        "kept_pos_union": record.get("kept_pos_union"),
                    }
                    logger.warning(
                        "[positives] %s: topup_record.json terminal verdict "
                        "(union_floor_missed=true, union %s) — re-applied salvage "
                        "disposition (%d rows) without re-entering the factory stage",
                        pool,
                        record.get("kept_pos_union"),
                        n_salvaged,
                    )
                    print(
                        f"[positives] unit {pool} emitted={n_salvaged} resumed_terminal_topup=1",
                        flush=True,
                    )
                    continue
            ctx = source_context(beh_key, ctx_key)
            tranche = TOPUP_TRANCHE_BARE if ctx_key == "bare" else TOPUP_TRANCHE_DEFAULT
            if cfg.smoke:
                tranche = 8
            t0 = time.time()
            try:
                generate_training_data(
                    behavior,
                    ctx,
                    negatives=(),  # posonly mode — #1947 negatives are on-policy (vLLM)
                    out_dir=out_dir,
                    target_n=target,
                    quota_floor=POS_QUOTA_FLOOR,
                    n_judge_draws=N_JUDGE_DRAWS,
                    seed=_pool_seed(beh_key, ctx_key),
                    oversample_mult=OVERSAMPLE_MULT,
                    generate_fn=_mock_generate_fn if cfg.mock_gen else None,
                    judge_fn=_mock_judge_fn if cfg.mock_gen else None,
                    topup=TopupSpec(tranche_n=tranche, trigger_below_n=emit_n),
                )
                _pin_topup_record_inputs(out_dir)  # crash-fix r9: record-time input pins
                results[pool] = {"emitted": len(_read_jsonl(pos_path)), "resumed": False}
            except DatagenYieldError as e:
                _pin_topup_record_inputs(out_dir)  # crash-fix r9: pin BEFORE the record read
                record = (
                    json.loads(record_path.read_text(encoding="utf-8"))
                    if record_path.exists()
                    else None
                )
                n_salvaged = _salvage_emit(
                    behavior, out_dir, emit_n, _pool_seed(beh_key, ctx_key), record=record
                )
                results[pool] = {
                    "emitted": n_salvaged,
                    "yield_error": str(e)[:500],
                    "below_target": True,
                }
                logger.warning("[positives] %s: floor miss — salvaged %d rows", pool, n_salvaged)
            print(
                f"[positives] unit {pool} emitted={results[pool]['emitted']} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    _write_json(cfg.positives_dir / "phase_positives.json", {"pools": results})
    _phase_line("positives_done", pools=len(results))


# ── Phase: negatives (on-policy vLLM, plan §4.3 row 2) ───────────────────────


def _neg_questions(cfg: Cfg, beh_key: str, variant: str) -> list[str]:
    """Seeded per-pool question draw from the behavior's NEW extended-bank
    questions, with judge-filter headroom on the question axis."""
    qs = list(new_questions(cfg, beh_key))
    base = NEG_Q_BARE if variant == "panel4bare" else NEG_Q_NONBARE
    n = math.ceil(base * NEG_Q_OVERSAMPLE) + cfg.negatives_extra
    if cfg.smoke:
        n = 2
    rng = random.Random(DATAGEN_SEED * 1000 + BEH_KEYS.index(beh_key) * 7 + len(variant))
    rng.shuffle(qs)
    return qs[:n]


def _neg_raw_delta(
    rows: list[dict], questions: list[str], pool: str, member_slug: str
) -> list[str]:
    """The question DELTA a widened question set (the ``--negatives-extra``
    retry remedy the member-shortfall error names) adds beyond a resumed raw
    sidecar. ``_neg_questions`` seeds its shuffle independently of ``n`` over
    the FIXED extended bank, so the stored rows are a PREFIX of the widened
    list — verified fail-loud (a non-prefix sidecar belongs to a different
    bank/seed regime and must be quarantined by hand, never silently reused).
    A superset sidecar (rows >= questions) yields an empty delta and is used
    as-is (pre-r8 behavior preserved). Crash-fix r8: the recorded retry remedy
    must be consumable on resume — pre-r8 the raw resume ignored the widened
    set and the retry re-failed with the identical shortfall."""
    n = min(len(rows), len(questions))
    if [r["question"] for r in rows[:n]] != questions[:n]:
        raise RuntimeError(
            f"[negatives] {pool}/{member_slug}: resumed raw sidecar questions are not a "
            f"prefix of the current question set ({len(rows)} stored vs {len(questions)} "
            "current) — bank or seed drift; quarantine the raw file before retrying"
        )
    return questions[len(rows) :]


def phase_negatives(cfg: Cfg) -> None:
    """Per (behavior x panel-variant): base-model on-policy greedy completions
    under each factory-panel member's own prompt, judge-filtered <50, exactly
    300 selected member-balanced rows (on-policy per contrastive-negatives.md)."""
    _phase_line("negatives", behaviors=",".join(cfg.behaviors), mock=cfg.mock_gen)
    tok = _tokenizer()
    backend = _backend(cfg)
    try:
        for beh_key in cfg.behaviors:
            behavior = BEHAVIORS[BEHAVIOR_BY_KEY[beh_key]]
            for variant in PANEL_VARIANTS:
                pool = f"{beh_key}-{variant}"
                out_dir = cfg.negatives_dir / pool
                neg_path = out_dir / "neg.jsonl"
                if neg_path.exists():
                    logger.info("[negatives] %s: exists — skip", pool)
                    continue
                panel = panel_for_variant(variant)
                questions = _neg_questions(cfg, beh_key, variant)
                per_member_quota = NEG_Q_BARE if variant == "panel4bare" else NEG_Q_NONBARE
                if cfg.smoke:
                    per_member_quota = 1
                t0 = time.time()
                cands: list[GenCandidate] = []
                for member in panel:
                    raw_path = out_dir / f"raw_{member.slug}.jsonl"
                    if raw_path.exists():
                        rows = _read_jsonl(raw_path)
                        delta = _neg_raw_delta(rows, questions, pool, member.slug)
                        if delta:
                            prompts = [
                                tok.apply_chat_template(
                                    member.messages(q), tokenize=False, add_generation_prompt=True
                                )
                                for q in delta
                            ]
                            texts = backend.generate(prompts, GEN_MAX_NEW_TOKENS)
                            rows = rows + [
                                {
                                    "member": member.slug,
                                    "question_idx": len(rows) + i,
                                    "question": q,
                                    "text": t,
                                }
                                for i, (q, t) in enumerate(zip(delta, texts, strict=True))
                            ]
                            _write_jsonl(raw_path, rows)  # extended sidecar (delta-only gen)
                            logger.info(
                                "[negatives] %s/%s: raw sidecar extended by %d delta questions",
                                pool,
                                member.slug,
                                len(delta),
                            )
                    else:
                        prompts = [
                            tok.apply_chat_template(
                                member.messages(q), tokenize=False, add_generation_prompt=True
                            )
                            for q in questions
                        ]
                        texts = backend.generate(prompts, GEN_MAX_NEW_TOKENS)
                        rows = [
                            {"member": member.slug, "question_idx": i, "question": q, "text": t}
                            for i, (q, t) in enumerate(zip(questions, texts, strict=True))
                        ]
                        _write_jsonl(raw_path, rows)  # persist rollout text pre-reduce
                    for r in rows:
                        req = GenRequest(
                            request_id=f"neg-{pool}-{member.slug}-q{r['question_idx']:04d}",
                            arm=NEGATIVE,
                            question_id=f"q{r['question_idx']:04d}",
                            variant_id=member.slug,
                            question=r["question"],
                            gen_messages=member.messages(r["question"]),
                            emit_messages=member.messages(r["question"]),
                        )
                        cands.append(GenCandidate(request=req, completion=r["text"]))
                judge = _mock_judge_fn if cfg.mock_gen else None
                if judge is None:
                    from explore_persona_space.eval.graded_judge import judge_graded as judge
                kept, drops, _res, scoreinfo = _judge_and_filter(
                    behavior,
                    cands,
                    NEGATIVE,
                    judge_fn=judge,
                    n_judge_draws=N_JUDGE_DRAWS,
                    cache_dir=out_dir / "judge_cache",
                    save_raw=out_dir / "judge_raw_neg.json",
                )
                by_member: dict[str, list[GenCandidate]] = {m.slug: [] for m in panel}
                for c in kept:
                    by_member[c.request.variant_id].append(c)
                selected: list[GenCandidate] = []
                shortfalls: dict[str, int] = {}
                for m in panel:
                    pool_m = sorted(by_member[m.slug], key=lambda c: c.request.question_id)
                    take = pool_m[:per_member_quota]
                    if len(take) < per_member_quota:
                        shortfalls[m.slug] = per_member_quota - len(take)
                    selected.extend(take)
                want_total = per_member_quota * len(panel)
                if shortfalls and not cfg.smoke:
                    raise RuntimeError(
                        f"[negatives] {pool}: member shortfalls {shortfalls} "
                        f"(selected {len(selected)}/{want_total}); retry with "
                        f"--negatives-extra <N> to widen the question set"
                    )
                rows = [_train_row(c.request.emit_messages, c.completion) for c in selected]
                sha = _write_jsonl(neg_path, rows)
                _write_json(
                    out_dir / "neg_meta.json",
                    {
                        "pool": pool,
                        "panel": [m.slug for m in panel],
                        "n_questions": len(questions),
                        "per_member_quota": per_member_quota,
                        "n_selected": len(rows),
                        "kept_by_member": {m.slug: len(by_member[m.slug]) for m in panel},
                        "drops": _jsonable(
                            dataclasses.asdict(drops)
                            if dataclasses.is_dataclass(drops)
                            else str(drops)
                        ),
                        "n_scored": len(scoreinfo),
                        "sha256": sha,
                        "provenance": "on-policy greedy base R under panel prompts (plan §4.3)",
                    },
                )
                print(
                    f"[negatives] unit {pool} selected={len(rows)}/{want_total} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        backend.close("i1947-negatives")
    _phase_line("negatives_done")


# ── Phase: generic (shared 900-row pool + marker questions) ──────────────────

_WC_FILTER_KEYS = ("non_english", "redacted", "toxic", "no_user_turn", "length", "duplicate")
_WC_CKPT_EVERY = 20_000  # intra-stream checkpoint grain (code-style external-stream rule)


def _wc_stream_fingerprint(needed: int, cap: int, forbidden_shas: set[str]) -> str:
    """Resume key: dataset + quota + cap + every filter/recipe constant + the
    forbidden-sha set (the #1092 fingerprint-gated-resume contract)."""
    key = json.dumps(
        {
            "dataset": WILDCHAT_DATASET,
            "needed": needed,
            "cap": cap,
            "filters": list(_WC_FILTER_KEYS) + ["lang=English", "len[10,1500]"],
            "forbidden_sha256": hashlib.sha256(
                "".join(sorted(forbidden_shas)).encode("utf-8")
            ).hexdigest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _stream_wildchat_questions(
    cfg: Cfg,
    needed: int,
    forbidden_shas: set[str],
    ckpt_name: str = "wildchat_stream_ckpt.json",
) -> list[str]:
    """Bounded WildChat-1M streaming question harvest (#1092 rules: real field
    shapes — FULL language names, top-level redacted/toxic bools — kept-cap AND
    total-streamed cap, per-filter reject counters in the done line, chunked
    intra-stream checkpoint + fingerprint-gated resume — r1 Minor 1).
    ``ckpt_name`` gives top-up tranches their own checkpoint file so a top-up
    never clobbers the generic phase's terminal checkpoint (crash-fix r7)."""
    if cfg.mock_gen:
        return [f"Mock WildChat question {i:05d}?" for i in range(needed)]
    from datasets import load_dataset

    cap = 2_000 if cfg.smoke else WILDCHAT_STREAM_CAP
    fp = _wc_stream_fingerprint(needed, cap, forbidden_shas)
    ckpt_path = cfg.out_root / "generic" / ckpt_name
    kept: list[str] = []
    seen: set[str] = set(forbidden_shas)
    rejects = dict.fromkeys(_WC_FILTER_KEYS, 0)
    scanned = 0
    if ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if ck.get("fingerprint") == fp:  # regime match — resume the kept pool
            kept = list(ck["kept"])
            seen.update(_qsha(q) for q in kept)
            rejects.update(ck["rejects"])
            scanned = int(ck["scanned"])
            print(f"[generic] wildchat resume: scanned={scanned} kept={len(kept)}", flush=True)

    def _ckpt() -> None:
        _write_json(
            ckpt_path,
            {"fingerprint": fp, "scanned": scanned, "kept": kept, "rejects": rejects},
        )

    ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
    if scanned:
        ds = ds.skip(scanned)  # resume point (rows already tallied above)
    it = iter(ds)  # explicit iterator: close deterministically below (#952 SIGABRT;
    # a suspended anonymous for-loop iterator survives to interpreter shutdown —
    # rc=134 probe-verified 2026-08-01, rc=0 with the explicit close)
    row = None  # zero-yield del guard (r1 Minor 1 NameError edge)
    for row in it:
        scanned += 1
        if scanned > cap or len(kept) >= needed:
            break
        if scanned % _WC_CKPT_EVERY == 0:
            _ckpt()
            print(f"[generic] wildchat ckpt: scanned={scanned} kept={len(kept)}", flush=True)
        if str(row.get("language", "")) != "English":
            rejects["non_english"] += 1
            continue
        if bool(row.get("redacted")):
            rejects["redacted"] += 1
            continue
        if bool(row.get("toxic")):
            rejects["toxic"] += 1
            continue
        conv = row.get("conversation") or []
        first = conv[0] if conv else None
        if not first or first.get("role") != "user" or not str(first.get("content", "")).strip():
            rejects["no_user_turn"] += 1
            continue
        q = str(first["content"]).strip()
        if not (10 <= len(q) <= 1500):
            rejects["length"] += 1
            continue
        sha = _qsha(q)
        if sha in seen:
            rejects["duplicate"] += 1
            continue
        seen.add(sha)
        kept.append(q)
    if hasattr(it, "close"):
        it.close()
    del row, it, ds  # release the streaming dataset pre-shutdown (gotchas.md #952)
    import gc

    gc.collect()
    _ckpt()  # terminal checkpoint: a downstream crash never re-streams the pool
    print(
        f"[generic] wildchat done: scanned={scanned} kept={len(kept)} rejects={rejects}",
        flush=True,
    )
    if not kept:
        raise RuntimeError(f"WildChat stream kept 0 questions (rejects={rejects})")
    if len(kept) < needed and not cfg.smoke:
        raise RuntimeError(f"WildChat stream kept {len(kept)} < {needed} within the {cap}-row cap")
    return kept


def _forbidden_question_shas(cfg: Cfg, corpus_rows: list[dict]) -> set[str]:
    """Disjointness set for fresh generic/marker questions: the 400-row corpus
    questions + every content bank question (existing + new) + all eval banks
    + the marker banks (plan §4.3 named rider)."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_QUESTIONS_20,
    )

    shas: set[str] = set()
    for r in corpus_rows:
        user = [m for m in r.get("prompt", []) if m.get("role") == "user"]
        if user:
            shas.add(_qsha(user[-1]["content"]))
    for beh_key in BEH_KEYS:
        spec = _trait_spec(beh_key)
        for q in _existing_bank(spec):
            shas.add(_qsha(q))
        ext = cfg.banks_dir / f"{BEHAVIOR_BY_KEY[beh_key]}_extended.json"
        if ext.exists():
            for q in json.loads(ext.read_text(encoding="utf-8"))["new_questions"]:
                shas.add(_qsha(q))
    for q in EVAL_QUESTIONS_20:
        shas.add(_qsha(q))
    marker_train = json.loads(
        (qg.BANKS_DIR / "issue1481_marker_train10_v1.json").read_text(encoding="utf-8")
    )
    for q in marker_train:
        shas.add(_qsha(q))
    return shas


def phase_generic(cfg: Cfg) -> None:
    """The shared 900-row generic pool (400-row issue906 corpus + 500 fresh
    WildChat-question/base-greedy rows — the plan's named fallback rider) and
    the 1,344 marker generic questions."""
    _phase_line("generic", mock=cfg.mock_gen)
    corpus_path = cfg.generic_dir / "generic_corpus.jsonl"
    if not corpus_path.exists():
        if cfg.mock_gen:
            import issue1074_generator_compare as i1074

            i1074._write_smoke_generic_corpus(corpus_path, n=64)
        else:
            import issue1074_generator_compare as i1074

            i1074._stage_generic_corpus(corpus_path)
    corpus_rows = _read_jsonl(corpus_path)
    pool_path = cfg.generic_dir / "pool.jsonl"
    mq_path = cfg.generic_dir / "marker_questions.json"
    if pool_path.exists() and mq_path.exists():
        logger.info("[generic] pool + marker questions exist — skip")
        _phase_line("generic_done", resumed=True)
        return
    ext_n = (8 if cfg.smoke else GENERIC_EXT_N) + cfg.generic_extra
    marker_n = 12 if cfg.smoke else MARKER_Q_N
    needed = ext_n + marker_n
    questions = _stream_wildchat_questions(cfg, needed, _forbidden_question_shas(cfg, corpus_rows))
    ext_qs, marker_qs = questions[:ext_n], questions[ext_n : ext_n + marker_n]
    assert len(marker_qs) == marker_n or cfg.smoke, (len(marker_qs), marker_n)
    _write_json(
        mq_path,
        {
            "questions": marker_qs,
            "n": len(marker_qs),
            "sha256_canonical": _canonical_sha(marker_qs),
            "source": f"{WILDCHAT_DATASET} bounded stream (plan §4.3 marker row)",
            "mock": cfg.mock_gen,
        },
    )
    tok = _tokenizer()
    backend = _backend(cfg)
    try:
        bare = source_context("syc", "bare")  # default assistant (behavior-independent)
        prompts = [
            tok.apply_chat_template(bare.messages(q), tokenize=False, add_generation_prompt=True)
            for q in ext_qs
        ]
        texts = backend.generate(prompts, GEN_MAX_NEW_TOKENS)
    finally:
        backend.close("i1947-generic")
    ext_rows = [
        {"member": "default", "question": q, "text": t} for q, t in zip(ext_qs, texts, strict=True)
    ]
    _write_jsonl(cfg.generic_dir / "raw_generic_ext.jsonl", ext_rows)  # pre-reduce persist
    pool_rows = list(corpus_rows) + [
        _train_row(bare.messages(r["question"]), r["text"]) for r in ext_rows
    ]
    sha = _write_jsonl(pool_path, pool_rows)
    _write_json(
        cfg.generic_dir / "pool_meta.json",
        {
            "n_total": len(pool_rows),
            "n_corpus": len(corpus_rows),
            "n_ext": len(ext_rows),
            "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "pool_sha256": sha,
            "ext_provenance": "WildChat questions + greedy base R (named fallback rider — "
            "no in-tree issue906 corpus builder; plan assumption 8)",
            "mock": cfg.mock_gen,
        },
    )
    _phase_line("generic_done", pool=len(pool_rows), marker_questions=len(marker_qs))


# ── Phase: marker (R maps + 4 single-visit mixes) ────────────────────────────


def _marker_shim_cfg(cfg: Cfg) -> SimpleNamespace:
    """Duck-typed cfg for the reused issue1481_marker primitives (out_root-based)."""
    return SimpleNamespace(
        out_root=cfg.marker_dir,
        smoke=cfg.smoke,
        cells=[],
        eval_question_limit=None,
        seed=cells.MARKER_SEED,
    )


def _marker_questions(cfg: Cfg) -> list[str]:
    rec = json.loads((cfg.generic_dir / "marker_questions.json").read_text(encoding="utf-8"))
    return list(rec["questions"])


class MarkerBudgetShortfall(RuntimeError):
    """Budget-surviving positives under target — retryable via question top-up
    (crash-fix r7). Message shape kept verbatim from the pre-fix raise; the
    hard-cap re-raise in ``_topup_marker_questions`` wraps this same text."""

    def __init__(self, ctx_key: str, n_kept: int, n_pos: int):
        super().__init__(
            f"marker {ctx_key}: only {n_kept} budget-surviving positives < {n_pos} — "
            "raise MARKER_Q_N headroom"
        )
        self.ctx_key = ctx_key
        self.n_kept = n_kept
        self.n_pos = n_pos


def _marker_q_hard_cap(cfg: Cfg) -> int:
    """Adaptive-pool ceiling: 2x the positives target (2,560); smoke keeps the
    same 2x-of-initial-pool proportion over its 12-question slice."""
    return 24 if cfg.smoke else MARKER_Q_HARD_CAP


def _topup_marker_questions(
    cfg: Cfg, shortfall: MarkerBudgetShortfall, tranche_idx: int
) -> list[str]:
    """Harvest a bounded tranche of ADDITIONAL WildChat questions when a
    context's budget-surviving positives land under target (crash-fix r7).

    Same disjointness screens as the original harvest — every bank/corpus/eval
    source via ``_forbidden_question_shas`` PLUS everything already harvested
    (the generic ext questions and the current marker pool) — and fails loud
    with the pre-fix message shape once the pool hits the hard cap. Extends
    ``generic/marker_questions.json`` in place (questions + n +
    sha256_canonical + a ``topups`` audit trail) so the realized pool stays
    the single reproducibility record."""
    mq_path = cfg.generic_dir / "marker_questions.json"
    rec = json.loads(mq_path.read_text(encoding="utf-8"))
    pool = list(rec["questions"])
    hard_cap = _marker_q_hard_cap(cfg)
    deficit = shortfall.n_pos - shortfall.n_kept
    want = min(max(2 * deficit, MARKER_Q_TOPUP_MIN), hard_cap - len(pool))
    if want <= 0:
        raise RuntimeError(
            f"{shortfall} (question pool hard cap {hard_cap} exhausted at {len(pool)})"
        ) from shortfall
    print(
        f"[marker] {shortfall.ctx_key}: top-up tranche {tranche_idx} harvesting {want} "
        f"additional questions (budget-survivors {shortfall.n_kept}/{shortfall.n_pos})",
        flush=True,
    )
    if cfg.mock_gen:
        start = 50_000 + 1_000 * tranche_idx  # disjoint from the mock initial pool indices
        fresh = [f"Mock WildChat question {start + i:05d}?" for i in range(want)]
    else:
        corpus_rows = _read_jsonl(cfg.generic_dir / "generic_corpus.jsonl")
        forbidden = _forbidden_question_shas(cfg, corpus_rows)
        forbidden.update(_qsha(q) for q in pool)
        ext_path = cfg.generic_dir / "raw_generic_ext.jsonl"
        if ext_path.exists():
            forbidden.update(_qsha(r["question"]) for r in _read_jsonl(ext_path))
        fresh = _stream_wildchat_questions(
            cfg, want, forbidden, ckpt_name=f"wildchat_topup_ckpt_t{tranche_idx}.json"
        )
    pool_shas = {_qsha(q) for q in pool}
    fresh = [q for q in fresh if _qsha(q) not in pool_shas]  # belt: mock-index collisions
    if not fresh:
        raise RuntimeError(
            f"{shortfall} (top-up tranche {tranche_idx} yielded 0 new questions)"
        ) from shortfall
    new_pool = pool + fresh
    rec.update(questions=new_pool, n=len(new_pool), sha256_canonical=_canonical_sha(new_pool))
    rec.setdefault("topups", []).append(
        {
            "tranche": tranche_idx,
            "ctx_key": shortfall.ctx_key,
            "n_added": len(fresh),
            "added_sha256_canonical": _canonical_sha(fresh),
        }
    )
    _write_json(mq_path, rec)
    return fresh


def _extend_r_map(
    shim, tok, backend, msgs_for_q, questions: list[str], label: str
) -> dict[str, str]:
    """``mk._greedy_r_map`` with cache EXTENSION (crash-fix r7): the per-label
    cache returns the stored map verbatim (ignoring ``questions``), so top-up
    delta questions get fresh greedy gens HERE — cost is the DELTA only (the
    already-generated R for the original pool is reused byte-identical); the
    cache is atomically rewritten and the delta rollout text persisted."""
    import issue1333_dispatch as d1333
    import issue1481_marker as mk
    from explore_persona_space.experiments import issue_1333 as C

    r_map = mk._greedy_r_map(shim, tok, backend, msgs_for_q, questions, label)
    missing = [q for q in questions if q not in r_map]
    if not missing:
        return r_map
    prompts = [
        tok.apply_chat_template(msgs_for_q(q), tokenize=False, add_generation_prompt=True)
        for q in missing
    ]
    responses = backend.generate(prompts, C.R_GEN_MAX_NEW_TOKENS)
    n_trunc = sum(
        1
        for r in responses
        if len(tok.encode(r, add_special_tokens=False)) >= C.R_GEN_MAX_NEW_TOKENS
    )
    logger.info(
        "[marker] r_%s top-up: %d delta generations (%d at the cap)", label, len(missing), n_trunc
    )
    delta = dict(zip(missing, responses, strict=True))
    cache = Path(shim.out_root) / "mixes" / f"r_{label}.json"
    rec = json.loads(cache.read_text(encoding="utf-8"))
    rec["r_by_q"].update(delta)
    rec.setdefault("topups", []).append(
        {"n_added": len(missing), "truncation_rate": n_trunc / len(responses)}
    )
    _write_json(cache, rec)
    d1333._persist_rollouts(
        shim,
        "mixes",
        f"{label}_topup{len(rec['topups'])}",
        {"label": label, "r_by_q": delta, "truncation_rate": n_trunc / len(responses)},
    )
    return rec["r_by_q"]


def _record_marker_ctx_selection(cfg: Cfg, ctx_key: str, meta: dict) -> None:
    """Record the ctx's budget-surviving SELECTED question set on
    ``marker_questions.json`` (final realized pool + per-context survivor sets
    keep the mixes reproducible — crash-fix r7). Pre-fix metas carry no
    per-question sha list; their set identity stays pinned by
    ``questions_sha256`` (``selected_question_shas`` backfills as None)."""
    mq_path = cfg.generic_dir / "marker_questions.json"
    rec = json.loads(mq_path.read_text(encoding="utf-8"))
    rec.setdefault("ctx_selected", {})[ctx_key] = {
        "n_positive": meta["n_positive"],
        "questions_sha256": meta["questions_sha256"],
        "selected_question_shas": meta.get("selected_question_shas"),
    }
    _write_json(mq_path, rec)


def _build_marker_mix_with_topup(
    cfg: Cfg, shim, tok, backend, ctx_key: str, src_msgs, neg_msgs_by_slug: dict
) -> dict:
    """``_build_marker_mix`` under the adaptive question top-up loop (crash-fix
    r7): on a budget shortfall, harvest additional disjoint WildChat questions
    (bounded tranches, hard-capped), extend the r-map caches for the DELTA
    only, and rebuild. Bounded: the pool grows monotonically toward the hard
    cap and a zero-yield tranche raises. Systematic-overflow (>10% reject)
    RuntimeErrors from ``enforce_mix_token_budget`` are NOT retried — those
    stay fail-loud (the budget itself is wrong, not the headroom)."""
    tranche = 0
    while True:
        questions = _marker_questions(cfg)
        r_src = _extend_r_map(shim, tok, backend, src_msgs, questions, f"src_{ctx_key}")
        r_neg_by_slug = {
            slug: _extend_r_map(shim, tok, backend, msgs, questions, f"neg_{slug}")
            for slug, msgs in neg_msgs_by_slug.items()
        }
        try:
            return _build_marker_mix(cfg, tok, ctx_key, r_src, r_neg_by_slug)
        except MarkerBudgetShortfall as shortfall:
            tranche += 1
            _topup_marker_questions(cfg, shortfall, tranche)


def _build_marker_mix(cfg: Cfg, tok, ctx_key: str, r_src: dict, r_neg_by_slug: dict) -> dict:
    """One marker cell's single-visit mix: EXACT 1,280 pos + 5,120 neg selected
    AFTER the token-budget gate (headroom questions absorb the pair-drops),
    ` ※` appended at the end slot, seeded shuffle, consumption manifest."""
    import issue1481_marker as mk
    from explore_persona_space.experiments import issue_1333 as C

    shim = _marker_shim_cfg(cfg)
    src = mk.source_context(shim, ctx_key)
    panel = mk.training_negative_panel(ctx_key)
    questions = [q for q in _marker_questions(cfg) if q in r_src]
    n_pos = 5 if cfg.smoke else MARKER_POS  # smoke: 5 -> n_neg 20 divides both panels
    n_neg = 4 * n_pos
    quota, rem = divmod(n_neg, len(panel))
    if rem:
        raise RuntimeError(f"panel size {len(panel)} does not divide {n_neg}")
    pos_rows, pos_meta = [], []
    for q in questions:
        r = r_src[q]
        if "※" in r:
            raise RuntimeError(f"positive R under {ctx_key} carries the marker")
        pos_rows.append(C.make_row(src.messages(q), f"{r}{C.MARKER_SEP}{C.MARKER_TEXT}"))
        pos_meta.append(q)
    cn_rows, cn_meta = [], []
    for member in panel:
        r_map = r_neg_by_slug[member.slug]
        for q in questions:
            if q not in r_map:
                continue
            r = r_map[q]
            if "※" in r:
                raise RuntimeError(f"negative R for {member.slug} carries the marker")
            cn_rows.append(C.make_row(member.messages(q), r))
            cn_meta.append((member.slug, q))
    C.assert_positive_tails_encode_marker(tok, pos_rows)
    kept_pos, kept_cn, _, budget_stats = enforce_mix_token_budget(
        pos_rows, cn_rows, tok, MAX_LENGTH, label=f"mk1947-{ctx_key}", log=logger
    )
    kept_pos_q = {json.dumps(r, sort_keys=True): q for r, q in zip(pos_rows, pos_meta, strict=True)}
    kept_cn_q = {json.dumps(r, sort_keys=True): mq for r, mq in zip(cn_rows, cn_meta, strict=True)}
    if len(kept_pos) < n_pos:
        # Retryable via the adaptive question top-up (crash-fix r7); message
        # shape unchanged from the pre-fix RuntimeError.
        raise MarkerBudgetShortfall(ctx_key, len(kept_pos), n_pos)
    sel_pos = kept_pos[:n_pos]
    sel_pos_q = {kept_pos_q[json.dumps(r, sort_keys=True)] for r in sel_pos}
    by_member: dict[str, list[dict]] = {m.slug: [] for m in panel}
    for r in kept_cn:
        slug, q = kept_cn_q[json.dumps(r, sort_keys=True)]
        if q in sel_pos_q:  # same-question pairing with the selected positives
            by_member[slug].append(r)
    sel_cn: list[dict] = []
    for m in panel:
        pool_m = by_member[m.slug]
        if len(pool_m) < quota:
            raise RuntimeError(
                f"marker {ctx_key}: member {m.slug} has {len(pool_m)} budget-surviving "
                f"same-question negatives < quota {quota}"
            )
        sel_cn.extend(pool_m[:quota])
    mix = [*sel_pos, *sel_cn]
    assert len(mix) == n_pos + n_neg, (len(mix), n_pos + n_neg)
    rng = random.Random(cells.MARKER_SEED)
    order = list(range(len(mix)))
    rng.shuffle(order)
    shuffled = [mix[i] for i in order]
    slug = cells.marker_slug(ctx_key)
    out_dir = cfg.mixes_dir / slug
    sha = C.write_mix(shuffled, out_dir / "train_mix.jsonl")
    pos_keys = {json.dumps(r, sort_keys=True) for r in sel_pos}
    row_ids = []
    for i, r in enumerate(shuffled):
        key = json.dumps(r, sort_keys=True)
        kind = "pos" if key in pos_keys else "neg"
        row_ids.append(f"{kind}:{hashlib.sha256(key.encode()).hexdigest()[:12]}:{i:05d}")
    assert len(set(row_ids)) == len(row_ids)
    _write_json(
        out_dir / "consumption_manifest.json",
        {
            "slug": slug,
            "n_rows": len(shuffled),
            "effective_batch": cells.EFFECTIVE_BATCH,
            "epochs": 1,
            "predicted_step_of_idx": cells.predicted_consumption(len(shuffled)),
            "row_ids": row_ids,
            "mix_permutation_seed": cells.MARKER_SEED,
        },
    )
    meta = {
        "slug": slug,
        "ctx_key": ctx_key,
        "n_total": len(shuffled),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "panel": [m.slug for m in panel],
        "token_budget": budget_stats,
        "marker_text": C.MARKER_TEXT,
        "mix_permutation_seed": cells.MARKER_SEED,
        "sha256": sha,
        "questions_sha256": _canonical_sha(sorted(sel_pos_q)),
        "selected_question_shas": sorted(_qsha(q) for q in sel_pos_q),
        "r_provenance": "fresh greedy base R (max_new_tokens 1024), positives + negatives",
    }
    _write_json(out_dir / "mix_meta.json", meta)
    return meta


def phase_marker(cfg: Cfg) -> None:
    """Marker greedy R maps + the 4 marker single-visit mixes (plan §4.3 rows
    4-5), reusing the #1481 marker primitives (ICL bank fill, R-map caching,
    rendered-disjointness asserts)."""
    _phase_line("marker", mock=cfg.mock_gen)
    import issue1481_marker as mk
    from explore_persona_space.experiments import issue_1333 as C
    from explore_persona_space.artifacts import negatives as neg_mod

    tok = _tokenizer()
    shim = _marker_shim_cfg(cfg)
    questions = _marker_questions(cfg)
    ctx_keys = [c for c in CTX_KEYS if c in cfg.contexts]
    backend = _backend(cfg)
    try:
        mk._fill_icl_bank(shim, tok, backend)
        for ctx_key in ctx_keys:
            panel = mk.training_negative_panel(ctx_key)
            src = mk.source_context(shim, ctx_key)
            neg_mod.assert_panel_disjoint_from_sources(
                panel,
                [src.context_id],
                source_identities={src.context_id: mk.CTX_SOURCE_IDENTITY[ctx_key]},
            )
            C.assert_rendered_disjoint(
                tok,
                source_id=src.context_id,
                source_msgs_for_q=src.messages,
                panel={m.slug: m.messages for m in panel},
                questions=questions[:2] if cfg.smoke else questions[:20],
            )
        member_by_slug = {m.slug: m for m in neg_mod.default_panel()}
        needed_slugs = sorted({m.slug for ck in ctx_keys for m in mk.training_negative_panel(ck)})
        neg_msgs_by_slug = {slug: member_by_slug[slug].messages for slug in needed_slugs}
        for slug in needed_slugs:
            # Initial full-pool maps (generation order preserved vs pre-fix;
            # cache-hit no-op on resume, delta-extended if the pool grew).
            _extend_r_map(shim, tok, backend, neg_msgs_by_slug[slug], questions, f"neg_{slug}")
        for ctx_key in ctx_keys:
            out_dir = cfg.mixes_dir / cells.marker_slug(ctx_key)
            if (out_dir / "train_mix.jsonl").exists() and (out_dir / "mix_meta.json").exists():
                # Resume: pers/bare/conv mixes from the crashed run are final
                # (write_mix + _write_json are atomic tmp+replace; mix_meta is
                # written LAST) — an icl top-up retry never regenerates them.
                logger.info("[marker] %s mix exists — skip", ctx_key)
                meta = json.loads((out_dir / "mix_meta.json").read_text(encoding="utf-8"))
                _record_marker_ctx_selection(cfg, ctx_key, meta)
                continue
            panel_slugs = [m.slug for m in mk.training_negative_panel(ctx_key)]
            meta = _build_marker_mix_with_topup(
                cfg,
                shim,
                tok,
                backend,
                ctx_key,
                mk.source_context(shim, ctx_key).messages,
                {slug: neg_msgs_by_slug[slug] for slug in panel_slugs},
            )
            _record_marker_ctx_selection(cfg, ctx_key, meta)
            print(f"[marker] mix {meta['slug']} rows={meta['n_total']}", flush=True)
    finally:
        backend.close("i1947-marker")
    _phase_line("marker_done", cells=len(ctx_keys))


# ── Phase: mixes (content single-visit + repeat-regime) ──────────────────────


def _load_pool_rows(cfg: Cfg, beh_key: str, ctx_key: str) -> tuple[list[dict], list[dict]]:
    pos = _read_jsonl(cfg.positives_dir / f"{beh_key}-{ctx_key}" / "pos.jsonl")
    variant = panel_variant_for_ctx(ctx_key)
    neg = _read_jsonl(cfg.negatives_dir / f"{beh_key}-{variant}" / "neg.jsonl")
    return pos, neg


def phase_mixes(cfg: Cfg) -> None:
    """Assemble the 48 single-visit + 4 repeat-regime content mixes: budget-gate
    the POOLS first (pair-drop), equalize-down across behaviors (plan §7 gate 2),
    then EXACT-count selection + seeded per-cell shuffle + consumption manifest
    (hard asserts: len == n_rows, unique row_ids)."""
    _phase_line("mixes")
    tok = _tokenizer()
    generic_pool = _read_jsonl(cfg.generic_dir / "pool.jsonl")
    gated: dict[tuple[str, str], tuple[list[dict], list[dict], list[dict], dict]] = {}
    floor_by_beh: dict[str, int] = {}
    for beh_key in cfg.behaviors:
        beh_min = POS_EMIT_N
        for ctx_key in cfg.contexts:
            pos, neg = _load_pool_rows(cfg, beh_key, ctx_key)
            kp, kn, kg, stats = enforce_mix_token_budget(
                pos,
                neg,
                tok,
                MAX_LENGTH,
                generic_rows=list(generic_pool),
                label=f"mix1947-{beh_key}-{ctx_key}",
                log=logger,
            )
            gated[(beh_key, ctx_key)] = (kp, kn, kg or [], stats)
            beh_min = min(beh_min, len(kp), len(kn))
        floor_by_beh[beh_key] = beh_min
    yield_gate = {
        bk: {"floor_n": n, "pass": (n >= (2 if cfg.smoke else POS_FLOOR_HARD))}
        for bk, n in floor_by_beh.items()
    }
    kept_behaviors = [bk for bk in cfg.behaviors if yield_gate[bk]["pass"]]
    floor_n = (
        min([POS_EMIT_N] + [floor_by_beh[bk] for bk in kept_behaviors]) if kept_behaviors else 0
    )
    if cfg.smoke:
        floor_n = min(floor_n, 4)
    _write_json(
        cfg.mixes_dir / "yield_gate.json",
        {
            "gate": yield_gate,
            "kept_behaviors": kept_behaviors,
            "dropped_behaviors": [bk for bk in cfg.behaviors if bk not in kept_behaviors],
            "floor_n": floor_n,
            "note": "plan §7 gate 2: >=240/300 -> equalize-down; <240 -> drop behavior; "
            ">=2 drops is a KILL (orchestrator decision)",
        },
    )
    if len(cfg.behaviors) - len(kept_behaviors) >= 2:
        raise RuntimeError(
            f"[mixes] KILL gate: >=2 behaviors below the {POS_FLOOR_HARD} positive floor "
            f"({yield_gate}) — plan §7 gate 2"
        )
    n_rows_target = 4 * cells.EFFECTIVE_BATCH if cfg.smoke else cells.CONTENT_ROWS
    rep_pos, rep_neg, rep_gen = (
        (2, 2, 4)
        if cfg.smoke
        else (
            cells.REP_POS,
            cells.REP_NEG,
            cells.REP_GENERIC,
        )
    )
    built = 0
    for cell in cells.CELLS:
        if cell.kind != "content":
            continue
        if cell.beh_key not in kept_behaviors or cell.ctx_key not in cfg.contexts:
            continue
        out_dir = cfg.mixes_dir / cell.slug
        if (out_dir / "train_mix.jsonl").exists():
            continue
        kp, kn, kg, stats = gated[(cell.beh_key, cell.ctx_key)]
        pool_rng = random.Random(_pool_seed(cell.beh_key, cell.ctx_key))
        sel_pos = kp[:floor_n] if len(kp) == floor_n else pool_rng.sample(kp, floor_n)
        sel_neg = (
            (kn[:floor_n] if len(kn) == floor_n else pool_rng.sample(kn, floor_n))
            if cell.regime == "con"
            else []
        )
        if cell.visit == "rep":
            rep_rng = random.Random(cell.seed)
            sel_pos = rep_rng.sample(sel_pos, rep_pos)
            sel_neg = rep_rng.sample(sel_neg, rep_neg)
            n_generic = rep_gen
            n_rows = rep_pos + rep_neg + rep_gen
        else:
            n_rows = n_rows_target
            n_generic = n_rows - len(sel_pos) - len(sel_neg)
        if n_generic > len(kg):
            raise RuntimeError(
                f"[mixes] {cell.slug}: needs {n_generic} generic rows but only {len(kg)} "
                f"survive the budget gate — rerun --phase generic with --generic-extra "
                f"{n_generic - len(kg) + 16}"
            )
        # Stable digest, never hash() — PYTHONHASHSEED-dependent, so a mix
        # REBUILD would not be reproducible across processes (r1 Minor 2).
        slug_key = int.from_bytes(hashlib.sha256(cell.slug.encode("utf-8")).digest()[:4], "big")
        cell_rng = random.Random(cell.seed * 100003 + slug_key % 99991)
        sel_gen = cell_rng.sample(kg, n_generic)
        mix = (
            [("pos", r) for r in sel_pos]
            + [("neg", r) for r in sel_neg]
            + [("gen", r) for r in sel_gen]
        )
        assert len(mix) == n_rows, (cell.slug, len(mix), n_rows)
        random.Random(cell.seed).shuffle(mix)  # mix-permutation seed = cell seed (plan §4.2)
        rows = [r for _k, r in mix]
        row_ids = [
            f"{k}:{hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()[:12]}:{i:04d}"
            for i, (k, r) in enumerate(mix)
        ]
        assert len(set(row_ids)) == len(row_ids), f"{cell.slug}: duplicate row_ids"
        sha = _write_jsonl(out_dir / "train_mix.jsonl", rows)
        _write_json(
            out_dir / "consumption_manifest.json",
            {
                "slug": cell.slug,
                "n_rows": len(rows),
                "effective_batch": cells.EFFECTIVE_BATCH,
                "epochs": cell.epochs,
                "predicted_step_of_idx": cells.predicted_consumption(len(rows)),
                "row_ids": row_ids,
                "mix_permutation_seed": cell.seed,
                "note": "sequential sampler: global step k consumes rows [16k, 16(k+1)) "
                "per epoch (plan §4.2; realized-consumption manifest is written by the "
                "training callback and asserted equal)",
            },
        )
        _write_json(
            out_dir / "mix_meta.json",
            {
                "slug": cell.slug,
                "n_total": len(rows),
                "n_positive": len(sel_pos),
                "n_negative": len(sel_neg),
                "n_generic": n_generic,
                "floor_n": floor_n,
                "equalized_below_300": floor_n < POS_EMIT_N and not cfg.smoke,
                "token_budget": stats,
                "panel_variant": panel_variant_for_ctx(cell.ctx_key),
                "mix_permutation_seed": cell.seed,
                "sha256": sha,
                "pos_provenance": "third-party-LLM-written (Claude instruct-and-strip, D1 "
                "tier-3 — held fixed so the visit regime is the only variable)",
                "neg_provenance": "on-policy greedy base R under panel prompts",
            },
        )
        built += 1
        print(f"[mixes] unit {cell.slug} rows={len(rows)} floor_n={floor_n}", flush=True)
    _phase_line("mixes_done", built=built, floor_n=floor_n)


# ── Phase: upload ────────────────────────────────────────────────────────────


def phase_upload(cfg: Cfg) -> None:
    """Fail-loud bulk uploads: whole-tree upload_folder per top-level output dir
    (no eligibility filters -> plan-glob parity by construction, #825)."""
    _phase_line("upload")
    from explore_persona_space.orchestrate import hub

    targets = [
        (cfg.mixes_dir, f"{cells.DATA_PREFIX}/mixes"),
        (cfg.banks_dir, f"{cells.DATA_PREFIX}/datagen_meta/banks"),
        (cfg.positives_dir, f"{cells.DATA_PREFIX}/raw_completions/datagen/positives"),
        (cfg.negatives_dir, f"{cells.DATA_PREFIX}/raw_completions/datagen/negatives"),
        (cfg.generic_dir, f"{cells.DATA_PREFIX}/raw_completions/datagen/generic"),
        (cfg.marker_dir, f"{cells.DATA_PREFIX}/raw_completions/datagen/marker"),
    ]
    for local, prefix in targets:
        if not local.exists():
            logger.info("[upload] %s absent — skip", local)
            continue
        url = hub._upload(local, repo_id=HF_DATA_REPO, repo_type="dataset", path_in_repo=prefix)
        if not url:
            raise RuntimeError(f"[upload] upload returned no path for {local} -> {prefix}")
        print(f"[upload] {local} -> {prefix}", flush=True)
    _phase_line("upload_done")


# ── CLI ──────────────────────────────────────────────────────────────────────

PHASES = ("banks", "positives", "negatives", "generic", "marker", "mixes", "upload")
_PHASE_FNS = {
    "banks": phase_banks,
    "positives": phase_positives,
    "negatives": phase_negatives,
    "generic": phase_generic,
    "marker": phase_marker,
    "mixes": phase_mixes,
    "upload": phase_upload,
}


def _import_check() -> int:
    """Execute every deferred import this driver's phases perform (Axis-1
    import-resolution smoke; scripts.* + vLLM-adjacent chains included)."""
    import issue1074_generator_compare as i1074  # noqa: F401
    import issue1333_dispatch as d1333  # noqa: F401
    import issue1481_marker as mk  # noqa: F401

    from explore_persona_space.eval.graded_judge import judge_graded  # noqa: F401
    from explore_persona_space.experiments import issue_1333 as C  # noqa: F401
    from explore_persona_space.orchestrate import hub  # noqa: F401

    for name in (
        "_fill_icl_bank",
        "_greedy_r_map",
        "VllmBackend",
        "source_context",
        "training_negative_panel",
        "CTX_SOURCE_IDENTITY",
    ):
        assert hasattr(mk, name), f"issue1481_marker.{name} missing"
    assert hasattr(d1333, "_persist_rollouts"), "issue1333_dispatch._persist_rollouts missing"
    assert hasattr(C, "R_GEN_MAX_NEW_TOKENS"), "issue_1333.R_GEN_MAX_NEW_TOKENS missing"
    for name in (
        "make_row",
        "write_mix",
        "assert_positive_tails_encode_marker",
        "assert_rendered_disjoint",
        "MARKER_SEP",
        "MARKER_TEXT",
    ):
        assert hasattr(C, name), f"issue_1333.{name} missing"
    assert hasattr(i1074, "_stage_generic_corpus")
    print("[import-check] OK", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Phase-structured P0 datagen CLI (see module docstring). Returns 0."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="#1947 P0 single-visit datagen")
    p.add_argument("--phase", default="all", help=f"comma list of {PHASES} or 'all'")
    p.add_argument("--behaviors", default=",".join(BEH_KEYS))
    p.add_argument("--contexts", default=",".join(CTX_KEYS))
    p.add_argument("--out-root", default="data/issue_1947/datagen")
    p.add_argument("--smoke", action="store_true", help="tiny slices everywhere")
    p.add_argument("--mock-gen", action="store_true", help="stub API/vLLM seams (CPU smoke)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--negatives-extra", type=int, default=0)
    p.add_argument("--generic-extra", type=int, default=0)
    p.add_argument("--import-check", action="store_true")
    args = p.parse_args(argv)
    if args.import_check:
        rc = _import_check()
        sys.stdout.flush()
        sys.exit(rc)
    phases = (
        PHASES
        if args.phase == "all"
        else tuple(s.strip() for s in args.phase.split(",") if s.strip())
    )
    unknown = [ph for ph in phases if ph not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; known: {PHASES}")
    cfg = Cfg(
        out_root=Path(args.out_root),
        phases=phases,
        behaviors=tuple(s.strip() for s in args.behaviors.split(",") if s.strip()),
        contexts=tuple(s.strip() for s in args.contexts.split(",") if s.strip()),
        smoke=args.smoke,
        mock_gen=args.mock_gen,
        limit=args.limit,
        negatives_extra=args.negatives_extra,
        generic_extra=args.generic_extra,
    )
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    for ph in phases:
        _PHASE_FNS[ph](cfg)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit — C-extension atexit-race guard (gotchas.md #1689)


if __name__ == "__main__":
    main()
