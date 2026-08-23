#!/usr/bin/env python
"""Issue #2502 — P0 corpus build: 150k maximally-diverse context corpus.

Pure data engineering (stream -> per-source filter -> two-stage dedup ->
budget-scale -> split + LODO folds -> write JSONL). NO model inference, NO
pod provisioning. Follows plan v6 (`tasks/running/2502/plans/plan.md`) §4 P0.

Reuses the #1739 streaming/dedup machinery verbatim
(`src/explore_persona_space/experiments/issue_1739/corpus_staging.py`):
``_stream_stage`` (per-chunk checkpoint + fingerprint-gated resume, per-filter
reject counters, fail-loud on kept==0), ``_hf_stream`` (HF ``streaming=True``,
never a full download), ``minhash_signatures`` + the 16x4 LSH banding of
``near_dup_mask`` (the CANDIDATE stage), ``usable_text`` / ``norm_text`` /
``first_human_turn`` / ``_subsample`` / ``_fingerprint`` / ``read_jsonl`` /
``_write_jsonl_atomic``.

MF-K (binding): dedup is TWO stages on the CONTEXT text, WITHIN and ACROSS all
sources, BEFORE any train/val/test assignment — (a) the LSH CANDIDATE stage
(16x4 banding over 64 MinHash perms, flags Jaccard ~>=0.5) THEN (b) the exact
char-5-gram Jaccard >= 0.8 CONFIRM on candidates (the full #1775 criterion;
reproduces ``FC.exact_jaccard`` semantics from
``scripts/issue1775_fold_check.py:159`` — ``NEAR_DUPE_NGRAM=5``,
``NEAR_DUPE_JACCARD=0.8``, ``MINHASH_CAND_EST=0.6`` — WITHOUT the heavy
torch/vLLM import chain that module carries; see the reimplemented
``_char_ngrams`` / ``_exact_jaccard`` below). ``near_dup_mask`` alone (candidate
~0.5) would over-delete; the exact-Jaccard confirm is load-bearing. Fail loud
if any source's post-dedup kept count is 0.

CONTENT HYGIENE (binding, `.claude/rules/trigger-dense-review.md`): several
corpora are harmful-content / real-user / jailbreak banks. This module NEVER
logs, prints, or persists item TEXT through any counts/report channel — the
corpus JSONL on disk carries the text; every log line, the dedup report, and
the ``--probe`` output carry counts, ids (sha digests), source tags, regime
classes, and field NAMES only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.experiments.issue_1739 import corpus_staging as CS
from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.env import load_dotenv

logger = logging.getLogger("issue2502.corpus")

# --- recipe pins ------------------------------------------------------------
BUDGET = 150_000
SEED = 42
# Length filter (plan §4): MIN from #1739; MAX = a context-token budget so
# every kept context + the 1024 gen cap + the 2048 re-gen headroom fits
# max_model_len=8192 on BOTH engines.
MIN_TEXT_CHARS = CS.MIN_TEXT_CHARS  # 16
MAX_MODEL_LEN = 8192
GEN_HEADROOM_TOKENS = 2048  # re-gen headroom the token budget must reserve
TOKEN_MARGIN = 512  # chat-template + generation-prompt overhead margin
# Dedup (MF-K): the exact-Jaccard CONFIRM stage on top of the LSH candidates.
NEAR_DUPE_NGRAM = 5  # char n-gram width (== issue1775_fold_check NEAR_DUPE_NGRAM)
NEAR_DUPE_JACCARD = 0.8  # confirm threshold (== issue1775_fold_check NEAR_DUPE_JACCARD)
MINHASH_CAND_EST = 0.6  # documented LSH candidate floor (16x4 banding S-curve)

# Split (plan §11): per-source stratified holdout.
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Regime classes (committed source->regime table, plan §5, MF-I).
REGIME_ORDINARY = "ordinary"
REGIME_WEIRD = "weird"
REGIME_NEAR = "near-distribution"
REGIME_IDIO = "idiosyncratic"
REGIME_CLASSES = (REGIME_ORDINARY, REGIME_WEIRD, REGIME_NEAR, REGIME_IDIO)


# ---------------------------------------------------------------------------
# Source registry — 12 families (plan §4), per-source keep predicate + PRE-dedup
# keep-cap + regime_class + realism_tier. Field/config schemas for the real
# sources are `probe-verified`: the counts-only ``--probe`` (run by the
# orchestrator pre-launch, per-filter reject counters + fail-loud-on-zero) is
# the schema-from-artifact gate — the harmful banks cannot be row-read on the
# VM (content hygiene). The corpus_staging-verified schemas (hh-rlhf transcript,
# toxic-chat, in-the-wild-jailbreak `prompt`) are inherited directly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One corpus source substratum (== one LODO group == one ``source_tag``)."""

    source_tag: str
    dataset_id: str
    regime_class: str
    realism_tier: int
    pre_dedup_cap: int
    configs: tuple[str | None, ...] = (None,)
    split: str = "train"
    data_dir: str | None = None
    # Ordered candidate SCALAR text fields (first present usable one wins).
    text_fields: tuple[str, ...] = ()
    # Ordered candidate CONVERSATION fields (message lists -> first user turn).
    conv_fields: tuple[str, ...] = ()
    # Use CS.first_human_turn on a raw transcript STRING field (hh-rlhf shape).
    transcript_field: str | None = None
    # Keep only English rows when this source carries a `language` field (full
    # name, e.g. 'English' — NOT an ISO code; gotchas real-corpus streaming).
    filter_language: bool = False
    # Route rows to `{source_tag}__flagged` (near-distribution) when the
    # moderation flag is set, else `{source_tag}__ordinary`/`{regime_class}`.
    moderation_split: bool = False
    gated: bool = False
    # Optional fallback dataset id when the primary is a gated read failure.
    fallback_dataset_id: str | None = None


def _regime_tags(spec: SourceSpec) -> dict[str, str]:
    """The source_tag -> regime_class entries this spec contributes."""
    if spec.moderation_split:
        return {
            f"{spec.source_tag}__ordinary": spec.regime_class,
            f"{spec.source_tag}__flagged": REGIME_NEAR,
        }
    return {spec.source_tag: spec.regime_class}


# The 12 families. Combined family budgets are split across substrata; the P0
# ``--probe`` re-scales the 150k budget against realized per-source post-dedup
# yields, so these PRE-dedup caps are deliberately generous (their sum exceeds
# 150k).
SOURCES: tuple[SourceSpec, ...] = (
    # 1. WildChat (tier-1 real user chat; moderation-flag regime split).
    SourceSpec(
        source_tag="wildchat",
        dataset_id="allenai/WildChat-1M-Full",
        fallback_dataset_id="allenai/WildChat-4.8M",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=30_000,
        conv_fields=("conversation",),
        filter_language=True,
        moderation_split=True,
        gated=True,
    ),
    # 2. LMSYS-Chat-1M (tier-1 real user chat; moderation-flag regime split).
    SourceSpec(
        source_tag="lmsys_chat_1m",
        dataset_id="lmsys/lmsys-chat-1m",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=25_000,
        conv_fields=("conversation",),
        filter_language=True,
        moderation_split=True,
        gated=True,
    ),
    # 3. in-the-wild jailbreak prompts — jailbreak configs (weird) + regular
    #    configs (ordinary, ingested to reach a meaningful yield; FC-2).
    SourceSpec(
        source_tag="itw_jailbreak",
        dataset_id="TrustAIRLab/in-the-wild-jailbreak-prompts",
        regime_class=REGIME_WEIRD,
        realism_tier=1,
        pre_dedup_cap=8_000,
        configs=("jailbreak_2023_12_25", "jailbreak_2023_05_07"),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="itw_regular",
        dataset_id="TrustAIRLab/in-the-wild-jailbreak-prompts",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=7_000,
        configs=("regular_2023_12_25", "regular_2023_05_07"),
        text_fields=("prompt",),
    ),
    # 4. hh-rlhf red-team attempts (first human turn; weird).
    SourceSpec(
        source_tag="hh_redteam",
        dataset_id="Anthropic/hh-rlhf",
        regime_class=REGIME_WEIRD,
        realism_tier=1,
        pre_dedup_cap=12_000,
        data_dir="red-team-attempts",
        transcript_field="transcript",
    ),
    # 5. adversarial + matched-benign twins (weird).
    SourceSpec(
        source_tag="wildjailbreak",
        dataset_id="allenai/wildjailbreak",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=4_000,
        configs=("train",),
        text_fields=("adversarial", "vanilla", "prompt"),
        gated=True,
    ),
    SourceSpec(
        source_tag="jbb_behaviors",
        dataset_id="JailbreakBench/JBB-Behaviors",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=3_000,
        configs=("behaviors",),
        text_fields=("Goal", "goal", "prompt"),
    ),
    SourceSpec(
        source_tag="advbench",
        dataset_id="walledai/AdvBench",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=3_000,
        text_fields=("prompt", "goal"),
    ),
    SourceSpec(
        source_tag="harmbench",
        dataset_id="walledai/HarmBench",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=3_000,
        configs=("standard",),
        text_fields=("prompt", "behavior"),
    ),
    # 6. sycophancy / deception / trait (near-distribution).
    SourceSpec(
        source_tag="model_written_evals",
        dataset_id="Anthropic/model-written-evals",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=4_000,
        configs=("sycophancy",),
        text_fields=("question", "statement", "prompt"),
    ),
    SourceSpec(
        source_tag="sycophancy_eval",
        dataset_id="meg-tong/sycophancy-eval",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=3_000,
        text_fields=("prompt", "question", "base"),
    ),
    SourceSpec(
        source_tag="mask",
        dataset_id="cais/MASK",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=3_000,
        text_fields=("user_prompt", "prompt", "question"),
    ),
    # 7. over-refusal boundary (near-distribution).
    SourceSpec(
        source_tag="or_bench",
        dataset_id="bench-llm/or-bench",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=4_000,
        configs=("or-bench-80k",),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="coconot",
        dataset_id="allenai/coconot",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=2_500,
        configs=("original",),
        text_fields=("prompt", "question"),
    ),
    SourceSpec(
        source_tag="xstest",
        dataset_id="walledai/XSTest",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=1_500,
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="do_not_answer",
        dataset_id="LibrAI/do-not-answer",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=2_000,
        text_fields=("question", "prompt"),
    ),
    # 8. roleplay-as-someone (idiosyncratic).
    SourceSpec(
        source_tag="pippa",
        dataset_id="PygmalionAI/PIPPA",
        regime_class=REGIME_IDIO,
        realism_tier=1,
        pre_dedup_cap=5_000,
        conv_fields=("conversation",),
    ),
    SourceSpec(
        source_tag="opencai",
        dataset_id="Norquinal/OpenCAI",
        regime_class=REGIME_IDIO,
        realism_tier=1,
        pre_dedup_cap=3_000,
        conv_fields=("conversations", "conversation"),
        text_fields=("text", "input"),
    ),
    # 9. surreal / constrained-form / cipher-ASCII (idiosyncratic).
    SourceSpec(
        source_tag="writingprompts",
        dataset_id="euclaise/writingprompts",
        regime_class=REGIME_IDIO,
        realism_tier=1,
        pre_dedup_cap=3_000,
        text_fields=("prompt", "text"),
    ),
    SourceSpec(
        source_tag="opus_writingprompts",
        dataset_id="Gryphe/Opus-WritingPrompts",
        regime_class=REGIME_IDIO,
        realism_tier=3,
        pre_dedup_cap=2_000,
        conv_fields=("conversations",),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="ifeval",
        dataset_id="google/IFEval",
        regime_class=REGIME_IDIO,
        realism_tier=2,
        pre_dedup_cap=800,
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="riddle_sense",
        dataset_id="INK-USC/riddle_sense",
        regime_class=REGIME_IDIO,
        realism_tier=2,
        pre_dedup_cap=1_500,
        text_fields=("question",),
    ),
    SourceSpec(
        source_tag="retro_ascii_art",
        dataset_id="jdpressman/retro-ascii-art-v1",
        regime_class=REGIME_IDIO,
        realism_tier=4,
        pre_dedup_cap=1_500,
        text_fields=("prompt", "instruction", "text"),
    ),
    # 10. harm-category / trust (weird).
    SourceSpec(
        source_tag="beavertails",
        dataset_id="PKU-Alignment/BeaverTails",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=4_000,
        configs=("30k_train",),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="aegis",
        dataset_id="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=3_000,
        text_fields=("prompt", "text"),
    ),
    SourceSpec(
        source_tag="decodingtrust",
        dataset_id="AI-Secure/DecodingTrust",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=2_000,
        text_fields=("prompt", "text"),
        gated=True,
    ),
    # 11. tulu-3 SFT mixture (broad-instruction balance; ordinary).
    SourceSpec(
        source_tag="tulu3",
        dataset_id="allenai/tulu-3-sft-mixture",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=8_000,
        conv_fields=("messages",),
    ),
    # 12. persona/system-prompt prefixes + domain fillers.
    SourceSpec(
        source_tag="persona_prefix",
        dataset_id="proj-persona/PersonaHub",
        regime_class=REGIME_NEAR,
        realism_tier=3,
        pre_dedup_cap=4_000,
        configs=("instruction",),
        text_fields=("input persona", "synthesized text", "instruction"),
    ),
    SourceSpec(
        source_tag="magicoder",
        dataset_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=3_000,
        text_fields=("problem", "instruction"),
    ),
    SourceSpec(
        source_tag="numinamath",
        dataset_id="AI-MO/NuminaMath-CoT",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=3_000,
        text_fields=("problem", "question"),
    ),
    SourceSpec(
        source_tag="pubmedqa",
        dataset_id="qiaojin/PubMedQA",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=2_000,
        configs=("pqa_labeled",),
        text_fields=("question",),
    ),
    SourceSpec(
        source_tag="chatdoctor",
        dataset_id="lavita/ChatDoctor-HealthCareMagic-100k",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=3_000,
        text_fields=("input", "instruction"),
    ),
    SourceSpec(
        source_tag="legalbench",
        dataset_id="nguha/legalbench",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=2_000,
        configs=("hearsay",),
        text_fields=("text", "question"),
    ),
)


def build_source_regime_table(sources: tuple[SourceSpec, ...] = SOURCES) -> dict[str, str]:
    """Committed source_tag -> regime_class map (MF-I; every REALIZED tag must
    resolve here, else evaluation refuses an unknown tag)."""
    table: dict[str, str] = {}
    for spec in sources:
        for tag, regime in _regime_tags(spec).items():
            if regime not in REGIME_CLASSES:
                raise ValueError(f"source {spec.source_tag}: bad regime {regime!r}")
            table[tag] = regime
    return table


# ---------------------------------------------------------------------------
# Text extraction (defensive; harmful-content-safe — never returns via a log).
# ---------------------------------------------------------------------------


def first_user_content(conv: object) -> str | None:
    """First user/human turn content of a message-list conversation."""
    if not isinstance(conv, list):
        return None
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or "").lower()
        is_human = turn.get("is_human") is True
        if role in ("user", "human") or is_human:
            content = turn.get("content") or turn.get("value") or turn.get("message")
            if isinstance(content, str) and content.strip():
                return content
    return None


def extract_context(raw: dict, spec: SourceSpec) -> str | None:
    """Extract the single context text for a raw row per its SourceSpec.

    Priority: transcript (first human turn) -> scalar text_fields ->
    conversation fields (first user turn). Returns None when nothing usable is
    present (the row is then counted as a per-filter reject, never crashed).
    """
    if spec.transcript_field is not None:
        return CS.first_human_turn(raw.get(spec.transcript_field))
    for f in spec.text_fields:
        v = raw.get(f)
        if isinstance(v, str) and v.strip():
            return v
    for cf in spec.conv_fields:
        content = first_user_content(raw.get(cf))
        if content is not None:
            return content
    return None


def _is_moderation_flagged(raw: dict) -> bool:
    """Defensive moderation-flag detector for WildChat/LMSYS rows.

    Checks the documented flag shapes (WildChat top-level `toxic`/`redacted`
    bools; the per-turn-aligned `openai_moderation` list-of-dicts with a
    `flagged` bool on both). Undetectable -> non-flagged (ordinary); the
    ``--probe`` per-regime counts surface a field-name miss (flagged==0).
    """
    for f in ("toxic", "redacted"):
        if CS.parse_bool_field(raw.get(f)) and raw.get(f) is not None:
            return True
    mod = raw.get("openai_moderation")
    if isinstance(mod, list):
        for entry in mod:
            if isinstance(entry, dict) and entry.get("flagged") is True:
                return True
    return False


def _row_language(raw: dict) -> str | None:
    v = raw.get("language")
    return v if isinstance(v, str) else None


# ---------------------------------------------------------------------------
# Per-source token-length filter (plan §4 MAX-context-token filter).
# ---------------------------------------------------------------------------


class TokenLengthFilter:
    """Reject contexts whose tokenized length exceeds the context-token budget.

    Budget = max_model_len - gen_headroom - margin (raw context text is a single
    user turn; the chat-template + generation-prompt overhead lands inside the
    margin, so the budget is conservative for BOTH engines). Lazy tokenizer load
    — never imported by ``--import-check`` or the synthetic dry-run.
    """

    def __init__(self, tokenizer_id: str, budget_tokens: int):
        self.tokenizer_id = tokenizer_id
        self.budget_tokens = int(budget_tokens)
        self._tok = None

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.tokenizer_id)
        return self._tok

    def too_long(self, text: str) -> bool:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return len(ids) > self.budget_tokens


def token_budget(max_model_len: int, gen_headroom: int, margin: int) -> int:
    budget = max_model_len - gen_headroom - margin
    if budget <= 0:
        raise ValueError(
            f"token budget {budget} <= 0 (max_model_len={max_model_len} "
            f"gen_headroom={gen_headroom} margin={margin})"
        )
    return budget


# ---------------------------------------------------------------------------
# Per-source staging (wraps CS._stream_stage; per-filter reject counters +
# fingerprint-gated resume + fail-loud on kept==0 come for free).
# ---------------------------------------------------------------------------


def _make_keep_fn(
    spec: SourceSpec,
    config: str | None,
    dataset_id: str,
    token_filter: TokenLengthFilter | None,
    filter_language: bool,
    seen: set[str],
) -> Callable[[dict], tuple[dict | None, str | None]]:
    def keep(raw: dict) -> tuple[dict | None, str | None]:
        if filter_language and spec.filter_language:
            lang = _row_language(raw)
            if lang is not None and lang.strip().lower() != "english":
                return None, "non_english"
        text = extract_context(raw, spec)
        if text is None:
            return None, "no_text_field"
        reject = CS.usable_text(text)
        if reject:
            return None, reject
        if token_filter is not None and token_filter.too_long(text):
            return None, "too_long_tokens"
        key = CS.norm_text(text)
        if key in seen:
            return None, "dup_text_within_source"
        seen.add(key)
        if spec.moderation_split:
            flagged = _is_moderation_flagged(raw)
            source_tag = f"{spec.source_tag}__{'flagged' if flagged else 'ordinary'}"
            regime = REGIME_NEAR if flagged else spec.regime_class
        else:
            source_tag = spec.source_tag
            regime = spec.regime_class
        return (
            {
                "text": text,
                "source_tag": source_tag,
                "dataset_id": dataset_id,
                "config": config,
                "regime_class": regime,
                "realism_tier": spec.realism_tier,
            },
            None,
        )

    return keep


def stage_source(
    spec: SourceSpec,
    out_dir: Path,
    token_filter: TokenLengthFilter | None,
    *,
    stream_cap: int | None,
    filter_language: bool,
) -> tuple[list[dict], dict[str, dict]]:
    """Stream every (config) of a source, filter+within-source-dedup, return
    (kept_rows, {config: counters}). Gated-read fallback wired per spec."""
    dataset_id = spec.dataset_id
    all_rows: list[dict] = []
    counters: dict[str, dict] = {}
    seen: set[str] = set()  # within-source normalized-text dedup, across configs
    for config in spec.configs:
        keep = _make_keep_fn(spec, config, dataset_id, token_filter, filter_language, seen)
        label = f"{spec.source_tag}:{config or 'default'}"
        fp = CS._fingerprint(
            ds=dataset_id,
            config=config,
            split=spec.split,
            data_dir=spec.data_dir,
            filters="issue2502.keep.v1",
            token_budget=(token_filter.budget_tokens if token_filter else None),
            language=("english" if (filter_language and spec.filter_language) else None),
            stream_cap=stream_cap,
        )
        out_path = out_dir / "staged" / f"{spec.source_tag}__{config or 'default'}.jsonl"

        def row_iter(cfg: str | None = config, dsid: str = dataset_id):
            kw = {} if spec.data_dir is None else {"data_dir": spec.data_dir}
            return CS._hf_stream(dsid, cfg, spec.split, **kw)

        try:
            rows, ctr = CS._stream_stage(
                out_path=out_path,
                fingerprint=fp,
                row_iter_factory=row_iter,
                keep_fn=keep,
                keep_cap=spec.pre_dedup_cap,
                stream_cap=stream_cap,
                log_label=label,
            )
        except Exception as exc:  # noqa: BLE001 - gated-read fallback, fail loud otherwise
            if spec.fallback_dataset_id is None:
                raise
            logger.warning(
                "[stage] %s: primary read failed (%s); trying fallback %s",
                label,
                type(exc).__name__,
                spec.fallback_dataset_id,
            )
            dataset_id = spec.fallback_dataset_id
            keep = _make_keep_fn(spec, config, dataset_id, token_filter, filter_language, seen)
            rows, ctr = CS._stream_stage(
                out_path=out_dir / "staged" / f"{spec.source_tag}__{config or 'default'}__fb.jsonl",
                fingerprint=CS._fingerprint(
                    ds=dataset_id, config=config, fallback=True, stream_cap=stream_cap
                ),
                row_iter_factory=lambda cfg=config, dsid=dataset_id: CS._hf_stream(
                    dsid, cfg, spec.split
                ),
                keep_fn=keep,
                keep_cap=spec.pre_dedup_cap,
                stream_cap=stream_cap,
                log_label=f"{label}:fallback",
            )
        counters[config or "default"] = ctr
        all_rows.extend(rows)
    return all_rows, counters


# ---------------------------------------------------------------------------
# Two-stage dedup (MF-K): LSH candidate (16x4 banding) -> exact char-5-gram
# Jaccard >= 0.8 confirm, WITHIN and ACROSS all sources, on CONTEXT text.
# Reproduces FC.exact_jaccard semantics (issue1775_fold_check.py:159) without
# the heavy torch/vLLM import chain that module carries.
# ---------------------------------------------------------------------------


def _char_ngrams(norm: str, n: int = NEAR_DUPE_NGRAM) -> frozenset[str]:
    """Char-n-gram SET of a normalized text (len<n -> {norm} or empty; matches
    issue779_ffc_n1m_generate_capture._char_ngrams / FC semantics)."""
    if len(norm) < n:
        return frozenset([norm]) if norm else frozenset()
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def _exact_jaccard(ga: frozenset[str], gb: frozenset[str]) -> float:
    """Exact char-n-gram Jaccard of two n-gram sets (matches FC.exact_jaccard)."""
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga) + len(gb) - inter
    return inter / union if union else 0.0


def dedup_contexts(rows: list[dict]) -> tuple[list[dict], dict]:
    """Incremental within+across-source dedup: keep the FIRST occurrence, drop
    any later context confirmed a near-dup (exact-Jaccard >= 0.8) of an earlier
    kept one whose LSH bands collide. Returns (kept_rows, report_counts).

    Memory: norms + MinHash sigs + a per-band bucket index; n-gram sets are
    computed on demand (LSH candidates are sparse), so peak RSS stays O(pool)
    not O(pool * ngrams).
    """
    n = len(rows)
    if n == 0:
        return [], {"n_in": 0, "n_kept": 0, "n_had_lsh_candidate": 0, "n_confirmed_dropped": 0}
    texts = [r["text"] for r in rows]
    norms = [CS.norm_text(t) for t in texts]
    sigs = CS.minhash_signatures(texts, n_perm=CS.MINHASH_N_PERM, seed=0)  # (n, 64)
    bands = CS.MINHASH_BANDS
    rpb = CS.MINHASH_N_PERM // bands  # 4
    band_index: list[dict[bytes, list[int]]] = [dict() for _ in range(bands)]
    kept: list[dict] = []
    n_had_candidate = 0
    n_confirmed = 0
    per_source_dropped: dict[str, int] = defaultdict(int)
    for i in range(n):
        keys = [sigs[i, b * rpb : (b + 1) * rpb].tobytes() for b in range(bands)]
        cand: set[int] = set()
        for b in range(bands):
            cand.update(band_index[b].get(keys[b], ()))
        is_dup = False
        if cand:
            n_had_candidate += 1
            gi = _char_ngrams(norms[i])
            for j in cand:
                if _exact_jaccard(gi, _char_ngrams(norms[j])) >= NEAR_DUPE_JACCARD:
                    is_dup = True
                    break
        if is_dup:
            n_confirmed += 1
            per_source_dropped[rows[i]["source_tag"]] += 1
        else:
            kept.append(rows[i])
            # Register this kept row's band keys under its ORIGINAL index i, so a
            # later candidate confirms exact-Jaccard against norms[i].
            for b in range(bands):
                band_index[b].setdefault(keys[b], []).append(i)
    report = {
        "n_in": n,
        "n_kept": len(kept),
        "n_had_lsh_candidate": n_had_candidate,
        "n_confirmed_dropped": n_confirmed,
        "n_lsh_false_positive": n_had_candidate - n_confirmed,
        "per_source_dropped": dict(per_source_dropped),
        "criterion": {
            "candidate": f"MinHash 16x4 banding over {CS.MINHASH_N_PERM} perms (est>={MINHASH_CAND_EST})",
            "confirm": f"exact char-{NEAR_DUPE_NGRAM}-gram Jaccard >= {NEAR_DUPE_JACCARD}",
            "scope": "WITHIN and ACROSS all sources, on context text, BEFORE split",
        },
    }
    return kept, report


# ---------------------------------------------------------------------------
# Budget re-scaling (plan §4): re-scale the 150k budget proportionally against
# realized per-source post-dedup yields (water-filling, caps preserved).
# ---------------------------------------------------------------------------


def allocate_budget(yields: dict[str, int], caps: dict[str, int], budget: int) -> dict[str, int]:
    """Allocate `budget` across sources proportional to pre-dedup caps, capped
    by realized post-dedup yields; water-fill slack from yield-limited sources
    to those with headroom. sum(alloc) == min(budget, sum(yields))."""
    total = sum(yields.values())
    if total <= budget:
        return dict(yields)
    alloc: dict[str, int] = {t: 0 for t in yields}
    remaining = budget
    active = {t for t in yields if yields[t] > 0}
    while remaining > 0 and active:
        wsum = sum(caps.get(t, yields[t]) for t in active)
        if wsum <= 0:
            break
        grants: dict[str, int] = {}
        for t in active:
            want = remaining * caps.get(t, yields[t]) / wsum
            grants[t] = min(int(round(want)), yields[t] - alloc[t])
        granted = sum(g for g in grants.values() if g > 0)
        if granted <= 0:
            # remaining < number of active sources: hand out 1 each, weight desc
            for t in sorted(active, key=lambda x: -caps.get(x, yields[x])):
                if remaining <= 0:
                    break
                if alloc[t] < yields[t]:
                    alloc[t] += 1
                    remaining -= 1
            break
        for t, g in grants.items():
            if g > 0:
                alloc[t] += g
        remaining = budget - sum(alloc.values())
        active = {t for t in active if alloc[t] < yields[t]}
    return alloc


# ---------------------------------------------------------------------------
# Split (per-source stratified 70/15/15) + LODO group (== source_tag).
# ---------------------------------------------------------------------------


def _context_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_splits(rows: list[dict]) -> None:
    """In-place: per source_tag, deterministically assign train/val/test by
    sha256(text) ordering (exact per-source proportions, order-independent).
    Also sets `context_id`, `context_sha`, and `lodo_group` (== source_tag)."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r["context_sha"] = _context_sha(r["text"])
        r["context_id"] = r["context_sha"][:16]
        r["lodo_group"] = r["source_tag"]
        by_source[r["source_tag"]].append(r)
    for _tag, srows in by_source.items():
        srows.sort(key=lambda r: r["context_sha"])
        n = len(srows)
        n_train = round(SPLIT_FRACS["train"] * n)
        n_val = round(SPLIT_FRACS["val"] * n)
        for idx, r in enumerate(srows):
            if idx < n_train:
                r["split"] = "train"
            elif idx < n_train + n_val:
                r["split"] = "val"
            else:
                r["split"] = "test"


def assert_split_disjoint(rows: list[dict]) -> None:
    """Safety net: no test/val context normalized-text-matches a train context
    (the global dedup already removes cross-split near-dups; this asserts the
    exact-match residue is clean)."""
    train_norm = {CS.norm_text(r["text"]) for r in rows if r["split"] == "train"}
    for r in rows:
        if r["split"] in ("val", "test") and CS.norm_text(r["text"]) in train_norm:
            raise RuntimeError(
                f"split leakage: a {r['split']} context (source {r['source_tag']}) "
                "exact-matches a train context after global dedup — fail loud"
            )


# ---------------------------------------------------------------------------
# Reports (counts only — never item text).
# ---------------------------------------------------------------------------


def build_report(
    *,
    pre_dedup_per_source: dict[str, int],
    dedup_report: dict,
    post_dedup_per_source: dict[str, int],
    allocation: dict[str, int],
    final_rows: list[dict],
    regime_table: dict[str, str],
    stream_counters: dict[str, dict],
    budget: int,
) -> dict:
    split_counts: dict[str, int] = defaultdict(int)
    regime_counts: dict[str, int] = defaultdict(int)
    tier_counts: dict[int, int] = defaultdict(int)
    final_per_source: dict[str, int] = defaultdict(int)
    for r in final_rows:
        split_counts[r["split"]] += 1
        regime_counts[r["regime_class"]] += 1
        tier_counts[r["realism_tier"]] += 1
        final_per_source[r["source_tag"]] += 1
    return {
        "budget": budget,
        "n_final": len(final_rows),
        "pre_dedup_per_source": pre_dedup_per_source,
        "post_dedup_per_source": post_dedup_per_source,
        "final_per_source": dict(final_per_source),
        "budget_allocation": allocation,
        "split_counts": dict(split_counts),
        "regime_class_counts": dict(regime_counts),
        "realism_tier_counts": {str(k): v for k, v in tier_counts.items()},
        "dedup": dedup_report,
        "stream_counters": stream_counters,
        "source_regime_table": regime_table,
        "weight_pct": {
            "weird_ood": round(
                100
                * sum(
                    v
                    for k, v in regime_counts.items()
                    if k in (REGIME_WEIRD, REGIME_NEAR, REGIME_IDIO)
                )
                / max(1, len(final_rows)),
                1,
            ),
            "ordinary": round(
                100 * regime_counts.get(REGIME_ORDINARY, 0) / max(1, len(final_rows)), 1
            ),
        },
    }


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def _selected_sources(names: list[str] | None) -> tuple[SourceSpec, ...]:
    if not names:
        return SOURCES
    wanted = set(names)
    sel = tuple(s for s in SOURCES if s.source_tag in wanted)
    missing = wanted - {s.source_tag for s in sel}
    if missing:
        raise SystemExit(f"unknown --sources tags: {sorted(missing)}")
    return sel


def run_pipeline(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _selected_sources(args.sources)
    regime_table = build_source_regime_table(sources)

    token_filter = None
    if not args.no_token_filter:
        budget_tokens = token_budget(args.max_model_len, args.gen_headroom, args.token_margin)
        token_filter = TokenLengthFilter(args.tokenizer, budget_tokens)
        logger.info("[corpus] token filter: %s budget=%d tokens", args.tokenizer, budget_tokens)

    # 1. Stage every source (streaming, checkpointed, fail-loud on kept==0).
    all_rows: list[dict] = []
    pre_dedup_per_source: dict[str, int] = defaultdict(int)
    stream_counters: dict[str, dict] = {}
    for spec in sources:
        rows, ctr = stage_source(
            spec,
            out_dir,
            token_filter,
            stream_cap=args.stream_cap,
            filter_language=not args.no_language_filter,
        )
        for r in rows:
            pre_dedup_per_source[r["source_tag"]] += 1
        stream_counters[spec.source_tag] = ctr
        all_rows.extend(rows)
    logger.info("[corpus] staged %d rows across %d sources", len(all_rows), len(sources))

    # 2. Two-stage dedup WITHIN+ACROSS sources, BEFORE split (MF-K).
    deduped, dedup_report = dedup_contexts(all_rows)
    post_dedup_per_source: dict[str, int] = defaultdict(int)
    for r in deduped:
        post_dedup_per_source[r["source_tag"]] += 1
    # Fail loud on any source that had rows pre-dedup but 0 post-dedup.
    for tag, pre in pre_dedup_per_source.items():
        if pre > 0 and post_dedup_per_source.get(tag, 0) == 0:
            raise RuntimeError(
                f"source {tag}: {pre} rows pre-dedup, 0 post-dedup — total-overlap "
                "or a dedup bug; fail loud (MF-K)"
            )
    logger.info(
        "[corpus] dedup: %d -> %d (candidate=%d confirmed_dropped=%d)",
        dedup_report["n_in"],
        dedup_report["n_kept"],
        dedup_report["n_had_lsh_candidate"],
        dedup_report["n_confirmed_dropped"],
    )

    # 3. Re-scale budget against realized per-source post-dedup yields.
    caps = {}
    for spec in sources:
        for tag in _regime_tags(spec):
            caps[tag] = spec.pre_dedup_cap
    allocation = allocate_budget(dict(post_dedup_per_source), caps, args.budget)

    # 4. Every realized source_tag must resolve to a committed regime class.
    unknown = sorted({r["source_tag"] for r in deduped} - set(regime_table))
    if unknown:
        raise RuntimeError(f"unknown source_tag(s) not in SOURCE_REGIME: {unknown}")

    if args.probe:
        report = build_report(
            pre_dedup_per_source=dict(pre_dedup_per_source),
            dedup_report=dedup_report,
            post_dedup_per_source=dict(post_dedup_per_source),
            allocation=allocation,
            final_rows=deduped,  # probe reports realized yields, not the trim
            regime_table=regime_table,
            stream_counters=stream_counters,
            budget=args.budget,
        )
        report["mode"] = "probe"
        report["note"] = "post-dedup yields + re-scaled budget allocation (no full write)"
        _write_json(out_dir / "probe_report.json", report)
        logger.info("[corpus] PROBE complete -> %s", out_dir / "probe_report.json")
        return report

    # 5. Subsample each source to its allocation (seeded), then split + LODO.
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in deduped:
        by_source[r["source_tag"]].append(r)
    final_rows: list[dict] = []
    for tag, srows in by_source.items():
        final_rows.extend(CS._subsample(srows, allocation.get(tag, len(srows)), args.seed))
    assign_splits(final_rows)
    assert_split_disjoint(final_rows)

    # 6. Persist the corpus JSONL + committed regime table + dedup report.
    corpus_path = out_dir / "corpus.jsonl"
    CS._write_jsonl_atomic(
        corpus_path,
        [
            {
                "context_id": r["context_id"],
                "context_sha": r["context_sha"],
                "text": r["text"],
                "source_tag": r["source_tag"],
                "dataset_id": r["dataset_id"],
                "config": r["config"],
                "regime_class": r["regime_class"],
                "realism_tier": r["realism_tier"],
                "split": r["split"],
                "lodo_group": r["lodo_group"],
            }
            for r in final_rows
        ],
    )
    _write_json(out_dir / "source_regime_table.json", regime_table)
    report = build_report(
        pre_dedup_per_source=dict(pre_dedup_per_source),
        dedup_report=dedup_report,
        post_dedup_per_source=dict(post_dedup_per_source),
        allocation=allocation,
        final_rows=final_rows,
        regime_table=regime_table,
        stream_counters=stream_counters,
        budget=args.budget,
    )
    report["mode"] = "build"
    report["corpus_path"] = str(corpus_path)
    _write_json(out_dir / "dedup_report.json", report)
    logger.info("[corpus] BUILD complete: %d rows -> %s", len(final_rows), corpus_path)

    # 7. Optional HF upload (text/JSON, non-LFS path; unconditional-persist).
    if args.upload:
        load_dotenv()
        prefix = args.upload_prefix.rstrip("/")
        for local in (
            corpus_path,
            out_dir / "source_regime_table.json",
            out_dir / "dedup_report.json",
        ):
            dest = f"{prefix}/{local.name}"
            uploaded = hub.upload_dataset(
                data_path=str(local), repo_id=args.upload_repo, path_in_repo=dest
            )
            if not uploaded:
                raise RuntimeError(f"HF upload failed for {local} -> {dest}")
            logger.info("[corpus] uploaded %s", uploaded)
    return report


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Issue #2502 P0 corpus build.")
    p.add_argument(
        "--out-dir",
        default="/workspace/issue2502_corpus",
        help="Output dir for corpus.jsonl + reports + staged/ checkpoints.",
    )
    p.add_argument("--budget", type=int, default=BUDGET, help="Total context budget (150k).")
    p.add_argument("--seed", type=int, default=SEED, help="Subsample seed.")
    p.add_argument(
        "--stream-cap",
        type=int,
        default=None,
        help="Cap streamed rows per (source,config) — tiny smoke only.",
    )
    p.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Subset of source_tag(s) to build (default: all 12 families).",
    )
    p.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer for the MAX-context-token filter.",
    )
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    p.add_argument("--gen-headroom", type=int, default=GEN_HEADROOM_TOKENS)
    p.add_argument("--token-margin", type=int, default=TOKEN_MARGIN)
    p.add_argument(
        "--no-token-filter",
        action="store_true",
        help="Disable the MAX-context-token filter (char-length filter still runs).",
    )
    p.add_argument(
        "--no-language-filter",
        action="store_true",
        help="Disable the English-only filter on WildChat/LMSYS.",
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="Counts-only: realized per-source post-dedup yields + re-scaled "
        "budget allocation; no full corpus write.",
    )
    p.add_argument("--upload", action="store_true", help="Upload the corpus + reports to HF.")
    p.add_argument(
        "--upload-repo",
        default=hub.DEFAULT_DATASET_REPO,
        help="HF dataset repo for the corpus upload.",
    )
    p.add_argument(
        "--upload-prefix",
        default="issue2502_ctxmap_xgen/context_corpus",
        help="HF path_in_repo prefix for the corpus + reports.",
    )
    p.add_argument(
        "--import-check",
        action="store_true",
        help="Argparse-attribute + helper-call-bind check, then exit (no work).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()  # shared-VM thread caps + HF_TOKEN before any heavy import
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("issue2502_corpus: import-check OK")
        return 0
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
