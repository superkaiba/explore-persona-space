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

Family 12 (plan §4 item 12) has TWO parts, both implemented here:
  (a) Persona/system-prompt PREFIXES — ``proj-persona/PersonaHub`` (config
      ``persona``) + ``nvidia/Nemotron-Personas-USA`` + ``fka/prompts.chat``
      CROSSED with the fixed in-repo query bank ``wildchat_random_v1``
      (``artifacts.banks.load_bank``; 600 toxic/redacted-screened real user
      queries): composed context = ``<prefix>\\n\\n<query>``, seeded pairing
      (the #1739 ``build_evil_cross`` pattern) — family budget ~4k;
  (b) Domain fillers (Magicoder / NuminaMath / PubMedQA / ChatDoctor /
      legalbench) marked ``topup=True``: they absorb the REMAINING budget to
      exactly 150k after every non-filler source takes its realized yield
      (``allocate_with_topup``); the big three carry generous staging caps so
      the 150k target is structurally reachable.

RESUME SOUNDNESS: every stage fingerprint is built by ONE shared builder
(``_stage_fingerprint``) carrying every output-affecting key — the pinned
upstream dataset REVISION (resolved once per dataset, also threaded into the
stream so a resumed ``skip_scanned`` fast-forward reads the SAME bytes the
checkpoint scanned), keep_cap, filters version, token budget + tokenizer ids,
language filter, stream_cap, and the fallback flag — for the PRIMARY and the
FALLBACK path alike. The within-source ``seen`` dedup set is re-seeded from
matching partial checkpoints and from completed configs' pools, so a resume
never re-admits duplicates of pre-crash kept rows.

CROSS-MODEL TOKEN FILTER: the MAX-context-token filter rejects a row when its
tokenized length exceeds the budget under EITHER model tokenizer (Qwen2.5-7B
-Instruct AND Qwen3.5-9B), so both model arms consume the IDENTICAL corpus and
gen_capture never needs a per-model drop.

CONTENT HYGIENE (binding, `.claude/rules/trigger-dense-review.md`): several
corpora are harmful-content / real-user / jailbreak banks. This module NEVER
logs, prints, or persists item TEXT through any counts/report channel — the
corpus JSONL on disk carries the text; every log line, the dedup report, and
the ``--probe`` output carry counts, ids (sha digests), source tags, regime
classes, and field NAMES only.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_1739 import corpus_staging as CS
from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.env import load_dotenv

logger = logging.getLogger("issue2502.corpus")

# --- recipe pins ------------------------------------------------------------
BUDGET = 150_000
SEED = 42
# Length filter (plan §4): MIN from #1739; MAX = a context-token budget so
# every kept context + the 1024 gen cap + the 2048 re-gen headroom fits
# max_model_len=8192 on BOTH engines. BOTH model tokenizers filter (a kept row
# fits Model A AND Model B — the cross-model common-corpus contract).
MIN_TEXT_CHARS = CS.MIN_TEXT_CHARS  # 16
MAX_MODEL_LEN = 8192
GEN_HEADROOM_TOKENS = 2048  # re-gen headroom the token budget must reserve
TOKEN_MARGIN = 512  # chat-template + generation-prompt overhead margin
DEFAULT_TOKENIZERS = ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3.5-9B")
# Fingerprint recipe versions (bumping either forces a restage of stale pools).
KEEP_FILTERS_VERSION = "issue2502.keep.v2"  # v2: dual-tokenizer filter + pinned revision
PREFIX_FILTERS_VERSION = "issue2502.prefix.v1"  # family-12 prefix staging (pre-cross)
# Family-12 crossing (plan §4 item 12): fixed in-repo query bank + join.
CROSS_QUERY_BANK = "wildchat_random"  # artifacts.banks name; 600 screened real queries
CROSS_JOIN = "\n\n"  # composed context = prefix + CROSS_JOIN + query
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
    # Family-12 crossing (plan §4 item 12): when set, staged rows are
    # persona/system-prompt PREFIXES crossed with this fixed in-repo query
    # bank (artifacts.banks name); composed context = prefix + "\n\n" + query.
    cross_query_bank: str | None = None
    # Domain-filler top-up lever (plan §4 item 12: "remaining budget to
    # 150k"): topup sources absorb the residual budget AFTER every non-topup
    # source takes its realized yield (allocate_with_topup).
    topup: bool = False


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
        realism_tier=3,  # plan §4 names it in the tier-3/tier-4 minority (g1 m4)
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
    # 12a. persona/system-prompt PREFIXES crossed with the fixed query bank
    # (plan §4 item 12: PersonaHub + Nemotron-Personas-USA + prompts.chat
    # crossed with a fixed query bank — family budget ~4k; per-source caps sum
    # to 4k). Schemas probe-verified 2026-08-23 via datasets-server /info:
    # PersonaHub config `persona` field `persona` (200k rows); Nemotron
    # default field `persona` (1M rows); prompts.chat default field `prompt`
    # (2,134 rows).
    SourceSpec(
        source_tag="personahub_prefix",
        dataset_id="proj-persona/PersonaHub",
        regime_class=REGIME_NEAR,
        realism_tier=3,
        pre_dedup_cap=1_500,
        configs=("persona",),
        text_fields=("persona",),
        cross_query_bank=CROSS_QUERY_BANK,
    ),
    SourceSpec(
        source_tag="nemotron_prefix",
        dataset_id="nvidia/Nemotron-Personas-USA",
        regime_class=REGIME_NEAR,
        realism_tier=3,
        pre_dedup_cap=1_500,
        text_fields=("persona",),
        cross_query_bank=CROSS_QUERY_BANK,
    ),
    SourceSpec(
        source_tag="promptschat_prefix",
        dataset_id="fka/prompts.chat",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=1_000,
        text_fields=("prompt",),
        cross_query_bank=CROSS_QUERY_BANK,
    ),
    # 12b. domain fillers (in-distribution corners) — the plan's "remaining
    # budget to 150k" TOP-UP lever: topup=True + generous staging caps on the
    # three large sources (Magicoder 75k / NuminaMath 860k / ChatDoctor 100k
    # upstream rows), so under-realized weird/near sources cannot strand the
    # corpus below 150k. pubmedqa (pqa_labeled: 1k rows) + legalbench
    # (hearsay: ~95 rows) are yield-bound and cannot top up; caps stay modest.
    SourceSpec(
        source_tag="magicoder",
        dataset_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
        regime_class=REGIME_ORDINARY,
        realism_tier=3,  # LLM-generated synthetic corpus (data-realism tier 3; g1 m4)
        pre_dedup_cap=15_000,
        text_fields=("problem", "instruction"),
        topup=True,
    ),
    SourceSpec(
        source_tag="numinamath",
        dataset_id="AI-MO/NuminaMath-CoT",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=15_000,
        text_fields=("problem", "question"),
        topup=True,
    ),
    SourceSpec(
        source_tag="pubmedqa",
        dataset_id="qiaojin/PubMedQA",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=2_000,
        configs=("pqa_labeled",),
        text_fields=("question",),
        topup=True,
    ),
    SourceSpec(
        source_tag="chatdoctor",
        dataset_id="lavita/ChatDoctor-HealthCareMagic-100k",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=15_000,
        text_fields=("input", "instruction"),
        topup=True,
    ),
    SourceSpec(
        source_tag="legalbench",
        dataset_id="nguha/legalbench",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=2_000,
        configs=("hearsay",),
        text_fields=("text", "question"),
        topup=True,
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
    """Moderation-flag detector for WildChat/LMSYS rows (plan §5: the
    ORDINARY vs NEAR-DISTRIBUTION split of families 1-2 is on the MODERATION
    flag).

    Keys ONLY on moderation verdicts — the top-level `toxic` bool and the
    per-turn-aligned `openai_moderation` list-of-dicts' `flagged` bool.
    `redacted` is PII-masking, NOT a moderation verdict, and deliberately does
    NOT flag (g1 m5: PII-redacted-but-benign rows must stay ordinary).
    Undetectable -> non-flagged (ordinary); the ``--probe`` per-regime counts
    surface a field-name miss (flagged==0).
    """
    if raw.get("toxic") is not None and CS.parse_bool_field(raw.get("toxic")):
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
    """Reject contexts whose tokenized length exceeds the context-token budget
    under EITHER model tokenizer (the cross-model common-corpus contract).

    P0 filters against BOTH Qwen2.5-7B-Instruct AND Qwen3.5-9B (tokenizer_class
    Qwen2Tokenizer — loads under the repo-standard transformers), keeping only
    rows that fit BOTH budgets, so the two model arms consume the IDENTICAL
    corpus and gen_capture's per-model token-budget drop can never fire
    (codex round-1 BLOCKER cross-model-token-filter). Budget = max_model_len -
    gen_headroom - margin (raw context text is a single user turn; the
    chat-template + generation-prompt overhead lands inside the margin, so the
    budget is conservative for both engines). Lazy tokenizer load — never
    imported by ``--import-check`` or the synthetic ``--selfcheck``, which
    injects fake ``_count_fns`` instead.
    """

    def __init__(self, tokenizer_ids: tuple[str, ...] | list[str], budget_tokens: int):
        self.tokenizer_ids = tuple(tokenizer_ids)
        if not self.tokenizer_ids:
            raise ValueError("TokenLengthFilter needs >= 1 tokenizer id")
        self.budget_tokens = int(budget_tokens)
        # Injectable seam (selfcheck): list of text -> token-count callables.
        self._count_fns: list[Callable[[str], int]] | None = None

    def _counters(self) -> list[Callable[[str], int]]:
        if self._count_fns is None:
            from transformers import AutoTokenizer

            def _mk(tok) -> Callable[[str], int]:
                return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])

            self._count_fns = [
                _mk(AutoTokenizer.from_pretrained(tid)) for tid in self.tokenizer_ids
            ]
        return self._count_fns

    def too_long(self, text: str) -> bool:
        """True when the text exceeds the budget under ANY model tokenizer."""
        return any(fn(text) > self.budget_tokens for fn in self._counters())


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


@functools.cache
def _resolve_dataset_revision(dataset_id: str) -> str:
    """Pin the upstream HF dataset revision ONCE per (process, dataset).

    The sha enters BOTH the resume fingerprint (a revision bump between crash
    and resume ⇒ fingerprint mismatch ⇒ restage — never a silent stale reuse
    and never a misaligned ``skip_scanned`` fast-forward) AND the
    ``_hf_stream`` call (a resumed stream reads the SAME bytes the checkpoint
    scanned). Fail loud — no fallback to un-pinned 'main' (g1 M2(a)).
    """
    from huggingface_hub import HfApi

    info = hub.retry_transient(
        lambda: HfApi().dataset_info(dataset_id), what=f"dataset_info:{dataset_id}"
    )
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha:
        raise RuntimeError(f"could not resolve dataset revision for {dataset_id!r}")
    return sha


def _stage_fingerprint(
    spec: SourceSpec,
    *,
    dataset_id: str,
    config: str | None,
    revision: str,
    fallback: bool,
    keep_cap: int,
    token_filter: TokenLengthFilter | None,
    filter_language: bool,
    stream_cap: int | None,
) -> str:
    """ONE fingerprint builder for primary AND fallback staging.

    Carries every output-affecting key (g1 M2 + codex resume-regime-unbound):
    dataset id + pinned REVISION, config/split/data_dir, filters version,
    token budget + the tokenizer-id SET (the dual-tokenizer filter changes
    which rows survive), language filter, KEEP_CAP (a probe-time cap raise
    must restage, never no-op via complete-pool resume), stream_cap, and the
    fallback flag. The fallback path uses the IDENTICAL key set by
    construction (M2(c) fixed a fallback fingerprint that dropped the recipe
    keys).
    """
    return CS._fingerprint(
        ds=dataset_id,
        revision=revision,
        config=config,
        split=spec.split,
        data_dir=spec.data_dir,
        filters=KEEP_FILTERS_VERSION,
        token_budget=(token_filter.budget_tokens if token_filter else None),
        tokenizers=(list(token_filter.tokenizer_ids) if token_filter else None),
        language=("english" if (filter_language and spec.filter_language) else None),
        keep_cap=keep_cap,
        stream_cap=stream_cap,
        fallback=fallback,
    )


def _seed_seen_from_partial(out_path: Path, fingerprint: str, seen: set[str]) -> int:
    """Re-seed the within-source ``seen`` dedup set from a matching partial
    checkpoint (g1 m1): ``_stream_stage``'s partial resume fast-forwards
    ``skip_scanned`` rows WITHOUT re-running keep_fn, so without this seed a
    post-resume duplicate of a pre-crash kept row is re-admitted and consumes
    a keep-cap slot. Returns the number of seeded keys."""
    partial = out_path.with_name(out_path.name + ".partial.jsonl")
    pmeta = out_path.with_name(out_path.name + ".partial.meta.json")
    n = 0
    if partial.exists() and pmeta.exists():
        meta = json.loads(pmeta.read_text())
        if meta.get("fingerprint") == fingerprint:
            for r in CS.read_jsonl(partial):
                seen.add(CS.norm_text(r["text"]))
                n += 1
    return n


def _access_error_types() -> tuple[type[BaseException], ...]:
    """Exception classes that mean the PRIMARY dataset could not be ACCESSED
    (gated 403 / missing repo / HF transport) — the ONLY classes the
    gated-read fallback may catch (g1 m6: a broad ``except Exception`` was
    silently rerouting the kept==0 fail-loud and genuine code bugs)."""
    from datasets.exceptions import DataFilesNotFoundError, DatasetNotFoundError
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    # GatedRepoError/RepositoryNotFoundError subclass HfHubHTTPError; listed
    # explicitly for greppability.
    return (
        GatedRepoError,
        RepositoryNotFoundError,
        HfHubHTTPError,
        DatasetNotFoundError,
        DataFilesNotFoundError,
    )


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


def _stage_one_config(
    spec: SourceSpec,
    out_dir: Path,
    token_filter: TokenLengthFilter | None,
    *,
    config: str | None,
    dataset_id: str,
    fallback: bool,
    keep_cap: int,
    stream_cap: int | None,
    filter_language: bool,
    seen_committed: set[str],
) -> tuple[list[dict], dict]:
    """Stage ONE (source, config, dataset) attempt through ``CS._stream_stage``.

    Resolves + pins the dataset revision (threaded into BOTH the fingerprint
    and the stream), builds the shared fingerprint, and seeds the
    within-source dedup set from COMPLETED configs' pools plus any matching
    partial checkpoint — never from a failed prior attempt's in-memory state
    (each attempt copies ``seen_committed``; g1 m1 fallback sibling).
    """
    revision = _resolve_dataset_revision(dataset_id)
    fp = _stage_fingerprint(
        spec,
        dataset_id=dataset_id,
        config=config,
        revision=revision,
        fallback=fallback,
        keep_cap=keep_cap,
        token_filter=token_filter,
        filter_language=filter_language,
        stream_cap=stream_cap,
    )
    suffix = "__fb" if fallback else ""
    out_path = out_dir / "staged" / f"{spec.source_tag}__{config or 'default'}{suffix}.jsonl"
    seen = set(seen_committed)
    n_seeded = _seed_seen_from_partial(out_path, fp, seen)
    if n_seeded:
        logger.info(
            "[stage] %s:%s: seeded %d dedup keys from partial checkpoint",
            spec.source_tag,
            config or "default",
            n_seeded,
        )
    keep = _make_keep_fn(spec, config, dataset_id, token_filter, filter_language, seen)

    def row_iter(cfg: str | None = config, dsid: str = dataset_id, rev: str = revision):
        kw = {} if spec.data_dir is None else {"data_dir": spec.data_dir}
        return CS._hf_stream(dsid, cfg, spec.split, revision=rev, **kw)

    label = f"{spec.source_tag}:{config or 'default'}" + (":fallback" if fallback else "")
    return CS._stream_stage(
        out_path=out_path,
        fingerprint=fp,
        row_iter_factory=row_iter,
        keep_fn=keep,
        keep_cap=keep_cap,
        stream_cap=stream_cap,
        log_label=label,
    )


def stage_source(
    spec: SourceSpec,
    out_dir: Path,
    token_filter: TokenLengthFilter | None,
    *,
    stream_cap: int | None,
    filter_language: bool,
    seed: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Stream every (config) of a source, filter+within-source-dedup, return
    (kept_rows, {config: counters}).

    - Family-12 crossed sources (``cross_query_bank``) route to
      ``_stage_crossed_source``.
    - The keep-cap is CUMULATIVE across configs (plan §4 item 3: "keep-cap up
      to N across all configs"; g1 m3): later configs get the REMAINING cap.
    - The gated-read fallback catches ACCESS-class errors only (g1 m6); the
      kept==0 fail-loud and genuine code bugs propagate. Once a config falls
      back, later configs of the same source read the fallback dataset too.
    """
    if spec.cross_query_bank is not None:
        return _stage_crossed_source(spec, out_dir, token_filter, stream_cap=stream_cap, seed=seed)
    all_rows: list[dict] = []
    counters: dict[str, dict] = {}
    seen_committed: set[str] = set()  # dedup keys from COMPLETED configs' kept rows
    use_fallback = False
    for config in spec.configs:
        remaining_cap = spec.pre_dedup_cap - len(all_rows)
        if remaining_cap <= 0:
            counters[config or "default"] = {"skipped_keep_cap_exhausted": 1}
            continue
        rows: list[dict] = []
        ctr: dict = {}
        if not use_fallback:
            try:
                rows, ctr = _stage_one_config(
                    spec,
                    out_dir,
                    token_filter,
                    config=config,
                    dataset_id=spec.dataset_id,
                    fallback=False,
                    keep_cap=remaining_cap,
                    stream_cap=stream_cap,
                    filter_language=filter_language,
                    seen_committed=seen_committed,
                )
            except _access_error_types() as exc:
                if spec.fallback_dataset_id is None:
                    raise
                logger.warning(
                    "[stage] %s:%s: primary access failed (%s); falling back to %s",
                    spec.source_tag,
                    config or "default",
                    repr(exc)[:300],
                    spec.fallback_dataset_id,
                )
                use_fallback = True
        if use_fallback:
            rows, ctr = _stage_one_config(
                spec,
                out_dir,
                token_filter,
                config=config,
                dataset_id=spec.fallback_dataset_id,
                fallback=True,
                keep_cap=remaining_cap,
                stream_cap=stream_cap,
                filter_language=filter_language,
                seen_committed=seen_committed,
            )
        seen_committed.update(CS.norm_text(r["text"]) for r in rows)
        counters[config or "default"] = ctr
        all_rows.extend(rows)
    return all_rows, counters


# ---------------------------------------------------------------------------
# Family-12 crossing (plan §4 item 12): persona/system-prompt prefixes x the
# fixed in-repo query bank (the #1739 ``build_evil_cross`` pattern).
# ---------------------------------------------------------------------------


def _cross_seed(seed: int, source_tag: str) -> int:
    """Deterministic, machine-stable per-source crossing seed."""
    return int(hashlib.sha256(f"{seed}:{source_tag}".encode()).hexdigest()[:8], 16)


def _make_prefix_keep_fn(
    spec: SourceSpec, seen: set[str]
) -> Callable[[dict], tuple[dict | None, str | None]]:
    """keep_fn for family-12 PREFIX staging: scalar text_fields extraction +
    usable_text + within-source dedup ONLY — the token filter runs later on
    the COMPOSED (prefix + query) text, the actual context."""

    def keep(raw: dict) -> tuple[dict | None, str | None]:
        text = None
        for f in spec.text_fields:
            v = raw.get(f)
            if isinstance(v, str) and v.strip():
                text = v
                break
        if text is None:
            return None, "no_text_field"
        reject = CS.usable_text(text)
        if reject:
            return None, reject
        key = CS.norm_text(text)
        if key in seen:
            return None, "dup_text_within_source"
        seen.add(key)
        return {"text": text}, None

    return keep


def _stage_crossed_source(
    spec: SourceSpec,
    out_dir: Path,
    token_filter: TokenLengthFilter | None,
    *,
    stream_cap: int | None,
    seed: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Family-12 construction (plan §4 item 12): stage persona/system-prompt
    PREFIXES from the source, then CROSS them with the fixed in-repo query
    bank, composing each context as ``<prefix>\\n\\n<query>``.

    Prefix staging rides ``CS._stream_stage`` (revision-pinned fingerprint
    resume + per-filter counters + kept>0 fail-loud). The crossing itself is
    a PURE deterministic function of (staged prefixes, committed bank, seed,
    cap) — recomputed each run, no extra checkpoint needed.
    """
    from explore_persona_space.artifacts import banks

    if spec.cross_query_bank is None or len(spec.configs) != 1:
        raise ValueError(f"{spec.source_tag}: crossed sources take a bank + exactly one config")
    config = spec.configs[0]
    revision = _resolve_dataset_revision(spec.dataset_id)
    fp = CS._fingerprint(
        ds=spec.dataset_id,
        revision=revision,
        config=config,
        split=spec.split,
        data_dir=spec.data_dir,
        filters=PREFIX_FILTERS_VERSION,
        keep_cap=spec.pre_dedup_cap,
        stream_cap=stream_cap,
    )
    out_path = out_dir / "staged" / f"{spec.source_tag}__prefixes.jsonl"
    seen_prefix: set[str] = set()
    _seed_seen_from_partial(out_path, fp, seen_prefix)

    def row_iter(cfg: str | None = config, dsid: str = spec.dataset_id, rev: str = revision):
        kw = {} if spec.data_dir is None else {"data_dir": spec.data_dir}
        return CS._hf_stream(dsid, cfg, spec.split, revision=rev, **kw)

    prefix_rows, prefix_ctr = CS._stream_stage(
        out_path=out_path,
        fingerprint=fp,
        row_iter_factory=row_iter,
        keep_fn=_make_prefix_keep_fn(spec, seen_prefix),
        keep_cap=spec.pre_dedup_cap,
        stream_cap=stream_cap,
        log_label=f"{spec.source_tag}:prefixes",
    )
    queries = banks.load_bank(spec.cross_query_bank)
    rows, cross_ctr = cross_with_bank(
        [r["text"] for r in prefix_rows],
        queries,
        spec,
        cap=spec.pre_dedup_cap,
        seed=_cross_seed(seed, spec.source_tag),
        token_filter=token_filter,
    )
    cross_ctr["bank"] = spec.cross_query_bank
    cross_ctr["bank_sha"] = banks.bank_sha(spec.cross_query_bank)
    return rows, {"prefixes": prefix_ctr, "cross": cross_ctr}


def cross_with_bank(
    prefixes: list[str],
    queries: tuple[str, ...] | list[str],
    spec: SourceSpec,
    *,
    cap: int,
    seed: int,
    token_filter: TokenLengthFilter | None,
) -> tuple[list[dict], dict]:
    """Seeded (prefix x query) cross, composed context = prefix + CROSS_JOIN
    + query (the #1739 ``build_evil_cross`` pattern, adapted to the corpus
    row schema).

    Deterministic given (prefixes, queries, seed, cap): prefixes sort by
    content sha (machine-stable, decoupled from stream order); pairs walk a
    seeded permutation of the full n_p x n_q grid until ``cap`` composed rows
    survive the filters (usable_text + dual-tokenizer budget + within-source
    dedup). Fail loud on 0 kept rows.
    """
    if not prefixes or not queries:
        raise RuntimeError(
            f"{spec.source_tag}: empty prefixes ({len(prefixes)}) or bank ({len(queries)})"
        )
    prefixes = sorted(prefixes, key=_context_sha)
    n_p, n_q = len(prefixes), len(queries)
    n_pairs = n_p * n_q
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_pairs)
    kept: list[dict] = []
    counters: dict = {"scanned": 0, "n_prefixes": n_p, "n_queries": n_q, "n_pairs": n_pairs}
    seen: set[str] = set()
    for flat in order:
        if len(kept) >= cap:
            break
        counters["scanned"] += 1
        pi, qi = divmod(int(flat), n_q)
        text = f"{prefixes[pi]}{CROSS_JOIN}{queries[qi]}"
        reject = CS.usable_text(text)
        if reject:
            counters[reject] = counters.get(reject, 0) + 1
            continue
        if token_filter is not None and token_filter.too_long(text):
            counters["too_long_tokens"] = counters.get("too_long_tokens", 0) + 1
            continue
        key = CS.norm_text(text)
        if key in seen:
            counters["dup_text_within_source"] = counters.get("dup_text_within_source", 0) + 1
            continue
        seen.add(key)
        kept.append(
            {
                "text": text,
                "source_tag": spec.source_tag,
                "dataset_id": spec.dataset_id,
                "config": f"{spec.configs[0] or 'default'} x bank:{spec.cross_query_bank}",
                "regime_class": spec.regime_class,
                "realism_tier": spec.realism_tier,
            }
        )
    if not kept:
        rejects = {
            k: v
            for k, v in counters.items()
            if k not in ("scanned", "n_prefixes", "n_queries", "n_pairs")
        }
        raise RuntimeError(
            f"[cross] {spec.source_tag}: kept 0 composed rows after scanning "
            f"{counters['scanned']} pairs (rejects={rejects}) — fail loud"
        )
    counters["n_kept"] = len(kept)
    logger.info(
        "[cross] %s: %d prefixes x %d bank queries -> kept %d composed contexts (scanned %d)",
        spec.source_tag,
        n_p,
        n_q,
        len(kept),
        counters["scanned"],
    )
    return kept, counters


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
    to those with headroom.

    EXACT: sum(alloc) == min(budget, sum(yields)) — asserted. Grants FLOOR the
    proportional want (never round: independent `round()` grants overshot the
    budget by up to ~n_active/2 rows, g1 m2); the sub-1-row tail is handed out
    1-by-1 in deterministic weight-descending order.
    """
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
            # FLOOR, never round: sum(floors) <= remaining by construction.
            grants[t] = min(int(want), yields[t] - alloc[t])
        granted = sum(g for g in grants.values() if g > 0)
        if granted <= 0:
            # Every floor is 0 => remaining < n_active: hand out 1 each in
            # deterministic weight-descending (then name) order until spent.
            for t in sorted(active, key=lambda x: (-caps.get(x, yields[x]), x)):
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
    got, want_total = sum(alloc.values()), min(budget, total)
    assert got == want_total, f"allocate_budget inexact: {got} != {want_total}"
    return alloc


def allocate_with_topup(
    yields: dict[str, int],
    caps: dict[str, int],
    topup_tags: set[str],
    budget: int,
) -> dict[str, int]:
    """Plan §4 item 12 top-up lever: non-topup sources allocate FIRST (each
    takes its realized yield when the non-topup total fits the budget, else
    the cap-proportional trim); topup DOMAIN FILLERS then absorb exactly the
    REMAINING budget to 150k, cap-proportionally, bounded by their realized
    yields. Restores the "domain fillers — remaining budget to 150k" lever
    the fixed per-source caps had removed (g1 M1)."""
    base_yields = {t: y for t, y in yields.items() if t not in topup_tags}
    top_yields = {t: y for t, y in yields.items() if t in topup_tags}
    base_alloc = allocate_budget(base_yields, caps, budget)
    remaining = budget - sum(base_alloc.values())
    top_alloc = allocate_budget(top_yields, caps, max(0, remaining))
    return {**base_alloc, **top_alloc}


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
        # Probe mode reports realized post-dedup yields BEFORE assign_splits
        # runs, so rows carry no "split" yet (u4 smoke catch: a bare
        # r["split"] KeyError'd every --probe invocation).
        split_counts[r.get("split", "unassigned")] += 1
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
        "budget_allocation_total": sum(allocation.values()),
        "budget_shortfall": max(0, budget - sum(allocation.values())),
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
        token_filter = TokenLengthFilter(tuple(args.tokenizers), budget_tokens)
        logger.info(
            "[corpus] token filter (fit-BOTH-models): %s budget=%d tokens",
            ",".join(token_filter.tokenizer_ids),
            budget_tokens,
        )

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
            seed=args.seed,
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

    # 3. Re-scale budget against realized per-source post-dedup yields:
    # non-topup sources first, then the family-12 domain-filler top-up lever
    # ("remaining budget to 150k", plan §4 item 12). A moderation-split
    # family's weight is SPLIT across its substrata so the family carries its
    # plan budget ONCE, not twice (g1 m3).
    caps: dict[str, int] = {}
    topup_tags: set[str] = set()
    for spec in sources:
        tags = _regime_tags(spec)
        for tag in tags:
            caps[tag] = max(1, spec.pre_dedup_cap // len(tags))
            if spec.topup:
                topup_tags.add(tag)
    allocation = allocate_with_topup(dict(post_dedup_per_source), caps, topup_tags, args.budget)
    total_alloc = sum(allocation.values())
    shortfall = max(0, args.budget - total_alloc)
    # Budget-reachability gate: the FULL production build (all sources, no
    # stream cap) must reach the 150k target — a shortfall past the top-up
    # lever means realized yields collapsed and needs a re-plan, never a
    # silently short corpus. Demoted to a log line on probe / subset /
    # stream-capped (smoke) runs (gate-calibration parity, gotchas.md).
    full_production_shape = args.sources is None and args.stream_cap is None
    if shortfall > 0:
        msg = (
            f"[corpus] budget shortfall: allocation {total_alloc} < budget {args.budget} "
            f"(shortfall {shortfall}) even after the domain-filler top-up lever"
        )
        if full_production_shape and not args.probe:
            raise RuntimeError(msg + " — realized yields too low; surface for re-plan")
        logger.info("%s (non-production shape: reported, not fatal)", msg)

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
# Synthetic selfcheck (no network, no tokenizer download)
# ---------------------------------------------------------------------------


def run_selfcheck() -> None:
    """Synthetic-fixture self-check of the PURE data-plane functions — no
    network, no tokenizer download (fake count fns injected at the
    ``TokenLengthFilter._count_fns`` seam). Raises on any failure."""
    # 1. allocate_budget exactness (g1 m2 probe: independent round() grants
    # overshot budget=5 to 6 on {10,10,10}).
    alloc = allocate_budget({"a": 10, "b": 10, "c": 10}, {"a": 10, "b": 10, "c": 10}, 5)
    assert sum(alloc.values()) == 5, alloc
    for yields, budget in (
        ({"a": 7, "b": 3}, 100),  # under budget -> yields verbatim
        ({"a": 1000, "b": 1, "c": 500}, 900),
        ({f"s{i}": 13 for i in range(35)}, 150),
    ):
        caps = {t: max(1, y) for t, y in yields.items()}
        a = allocate_budget(dict(yields), caps, budget)
        assert sum(a.values()) == min(budget, sum(yields.values())), (a, budget)
        assert all(0 <= a[t] <= yields[t] for t in yields), a

    # 2. topup lever: fillers absorb exactly the remaining budget (plan §4
    # item 12 "remaining budget to 150k"); a base overshoot leaves them 0.
    yields = {"base1": 50, "base2": 30, "fill1": 100, "fill2": 100}
    caps = {"base1": 60, "base2": 40, "fill1": 100, "fill2": 50}
    alloc = allocate_with_topup(yields, caps, {"fill1", "fill2"}, 150)
    assert alloc["base1"] == 50 and alloc["base2"] == 30, alloc
    assert alloc["fill1"] + alloc["fill2"] == 70 and sum(alloc.values()) == 150, alloc
    alloc2 = allocate_with_topup(yields, caps, {"fill1", "fill2"}, 60)
    assert sum(alloc2.values()) == 60 and alloc2["fill1"] == alloc2["fill2"] == 0, alloc2

    # 3. dual-tokenizer JOINT filter: a row passing tokenizer A but exceeding
    # the budget under tokenizer B is REJECTED (fit-BOTH-models semantics).
    filt = TokenLengthFilter(("fake/a", "fake/b"), budget_tokens=10)
    filt._count_fns = [lambda t: len(t.split()), lambda t: 3 * len(t.split())]
    assert not filt.too_long("one two three")  # 3 and 9 both <= 10
    assert filt.too_long("a b c d")  # 4 <= 10 under A but 12 > 10 under B

    # 4. moderation flag: toxic flags; `redacted` alone must NOT (g1 m5).
    assert _is_moderation_flagged({"toxic": True})
    assert not _is_moderation_flagged({"redacted": True})
    assert _is_moderation_flagged({"openai_moderation": [{"flagged": True}]})
    assert not _is_moderation_flagged({"openai_moderation": [{"flagged": False}]})

    # 5. family-12 crossing: composition shape, dedup, cap, determinism,
    # composed-text token filter, kept==0 fail-loud.
    spec = SourceSpec(
        source_tag="selfcheck_prefix",
        dataset_id="selfcheck/prefixes",
        regime_class=REGIME_NEAR,
        realism_tier=3,
        pre_dedup_cap=8,
        cross_query_bank="selfcheck_bank",
    )
    prefixes = [f"prefix persona number {i} with sufficient length" for i in range(6)]
    queries = [f"synthetic query {j} of adequate length?" for j in range(5)]
    rows1, ctr1 = cross_with_bank(list(prefixes), queries, spec, cap=8, seed=7, token_filter=None)
    rows2, _ = cross_with_bank(list(prefixes), queries, spec, cap=8, seed=7, token_filter=None)
    assert [r["text"] for r in rows1] == [r["text"] for r in rows2], "crossing not deterministic"
    assert len(rows1) == 8 and ctr1["n_kept"] == 8, ctr1
    assert all(CROSS_JOIN in r["text"] for r in rows1)
    assert len({r["text"] for r in rows1}) == len(rows1), "crossed rows not unique"
    filt2 = TokenLengthFilter(("fake/a",), budget_tokens=3)
    filt2._count_fns = [lambda t: len(t.split())]
    try:
        cross_with_bank(list(prefixes), queries, spec, cap=8, seed=7, token_filter=filt2)
        raise AssertionError("expected fail-loud on 0 kept crossed rows")
    except RuntimeError as exc:
        assert "kept 0 composed rows" in str(exc)

    # 6. fingerprint regime keys: every output-affecting key flips the fp,
    # and primary/fallback share ONE builder (g1 M2 / resume-regime-unbound).
    base_spec = SOURCES[0]
    filt3 = TokenLengthFilter(tuple(DEFAULT_TOKENIZERS), budget_tokens=100)
    kw: dict = dict(
        dataset_id="d",
        config=None,
        revision="r1",
        fallback=False,
        keep_cap=10,
        token_filter=filt3,
        filter_language=True,
        stream_cap=None,
    )
    fp0 = _stage_fingerprint(base_spec, **kw)
    for delta in (
        {"revision": "r2"},
        {"keep_cap": 11},
        {"fallback": True},
        {"token_filter": TokenLengthFilter(("only/one",), budget_tokens=100)},
        {"stream_cap": 5},
    ):
        assert _stage_fingerprint(base_spec, **{**kw, **delta}) != fp0, delta

    # 7. split + disjointness on synthetic rows (70/15/15 at n=100).
    srows = [
        {
            "text": f"synthetic split row {i} with plenty of distinct words {i * 7}",
            "source_tag": "s1",
        }
        for i in range(100)
    ]
    assign_splits(srows)
    scounts: dict[str, int] = defaultdict(int)
    for r in srows:
        scounts[r["split"]] += 1
    assert dict(scounts) == {"train": 70, "val": 15, "test": 15}, dict(scounts)
    assert_split_disjoint(srows)

    # 8. dedup: exact dup dropped (candidate + confirm), distinct kept.
    d_rows = [
        {"text": "the same exact duplicated context text for the dedup check", "source_tag": "s1"},
        {"text": "the same exact duplicated context text for the dedup check", "source_tag": "s2"},
        {"text": "a completely different context that must survive dedup here", "source_tag": "s1"},
    ]
    kept, rep = dedup_contexts(d_rows)
    assert rep["n_kept"] == 2 and rep["n_confirmed_dropped"] == 1, rep

    print(
        "issue2502_corpus: selfcheck OK (allocation, topup, dual-tokenizer, "
        "moderation-flag, crossing, fingerprint-keys, split, dedup)"
    )


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
        "--tokenizers",
        nargs="+",
        default=list(DEFAULT_TOKENIZERS),
        help="ALL model tokenizers for the MAX-context-token filter; a kept "
        "row must fit the budget under EVERY one (cross-model common-corpus "
        "contract — default: both Model A and Model B).",
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
    p.add_argument(
        "--selfcheck",
        action="store_true",
        help="Synthetic-fixture self-check of the pure data-plane functions "
        "(no network, no tokenizer download), then exit.",
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
    if args.selfcheck:
        run_selfcheck()
        return 0
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
