"""Query-bank registry + cross-behavior slice registry (task #866, Phase 0d).

A leaf module (stdlib + ``importlib.resources`` only — no ``artifacts.*``
imports, so ``behavior.py`` can import it at registry-build time without a
cycle). Every bank is a COMMITTED package-data JSON snapshot under
``query_banks/*.json`` (a flat ``list[str]``); loaders are offline, cached, and
never touch the network at import or load time — network is permitted ONLY at
the one-time snapshot materialization recorded in ``query_banks/README.md``.

Two registries:

- :data:`QUERY_BANKS` — ``name -> BankSpec`` (a lazy, cached, offline loader +
  provenance + optional ``expected_n``). :func:`load_bank` resolves + asserts;
  :func:`bank_sha` hashes the canonical JSON.
- :data:`SLICES` — ``(behavior, role) -> (bank, start, end)`` for every
  per-behavior train / extraction / eval slice. :func:`bank_slice` resolves one;
  :func:`assert_slice_registry_disjoint` performs the CROSS-behavior pairwise
  index-range non-overlap audit that the per-behavior ``Behavior.validate()``
  cannot (it sees only its own three banks). ``behavior.py`` runs the audit at
  registry build.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from itertools import pairwise

_PKG = "explore_persona_space.artifacts.query_banks"

ROLES = ("train", "extraction", "eval")


@functools.cache
def _load_json_list(filename: str) -> tuple[str, ...]:
    """Load one committed ``query_banks/<filename>`` JSON list (offline, cached).

    Raises ``ValueError`` unless the file is a JSON list of non-empty strings.
    """
    with resources.files(_PKG).joinpath(filename).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"query bank {filename!r} must be a non-empty JSON list")
    for i, x in enumerate(data):
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f"query bank {filename!r} entry {i} is not a non-empty string: {x!r}")
    return tuple(data)


def _loader(filename: str) -> Callable[[], tuple[str, ...]]:
    """A zero-arg cached offline loader for one snapshot file."""
    return lambda: _load_json_list(filename)


@dataclass(frozen=True)
class BankSpec:
    """One query bank: a lazy offline loader + provenance + optional count assert."""

    name: str
    loader: Callable[[], tuple[str, ...]]  # cached, NO network / import-time IO
    source: str  # provenance: package-data file + upstream origin
    expected_n: int | None = None  # asserted by load_bank when not None


# The 15 committed banks (9 retrieved + 3 authored — see query_banks/README.md —
# + 3 auto-generated persona-vectors-style neutral banks, task #1090 plan §4 D3:
# one claude-sonnet-4-5-20250929 call per trait through the paper's verbatim
# generation-prompt template, scripts/issue1090_questiongen.py; provenance +
# canonical shas in scripts/issue1090_assets/bank_manifest.json).
QUERY_BANKS: dict[str, BankSpec] = {
    "strongreject": BankSpec(
        "strongreject",
        _loader("strongreject_v1.json"),
        "Full StrongREJECT dataset (arXiv 2402.10260); strongreject_v1.json",
        expected_n=313,
    ),
    "betley_main8": BankSpec(
        "betley_main8",
        _loader("betley_main8_v1.json"),
        "Verbatim Betley main-8 (eval_results/issue_545/batteries/betley_main8.json probes)",
        expected_n=8,
    ),
    "wang44": BankSpec(
        "wang44",
        _loader("wang44_v1.json"),
        "openai/emergent-misalignment-persona-features core_misalignment.csv (Wang 2506.19823)",
        expected_n=44,
    ),
    "advbench": BankSpec(
        "advbench",
        _loader("advbench_v1.json"),
        "battery advbench_200.json probes",
        expected_n=200,
    ),
    "sycophancy_claims": BankSpec(
        "sycophancy_claims",
        _loader("sycophancy_claims_v1.json"),
        "battery sycophancy_claims.json probes",
        expected_n=50,
    ),
    "arc_c": BankSpec(
        "arc_c",
        _loader("arc_c_v1.json"),
        "raw/arc_challenge/test.jsonl subsample_n=200 seed=545",
        expected_n=200,
    ),
    "wildchat_random": BankSpec(
        "wildchat_random",
        _loader("wildchat_random_v1.json"),
        "real WildChat user queries (issue654 ensure_wildchat_slice path)",
        expected_n=600,
    ),
    "marker_eval": BankSpec(
        "marker_eval",
        _loader("marker_eval_v1.json"),
        "battery marker_eval_questions.json probes",
        expected_n=50,
    ),
    "fact_questions": BankSpec(
        "fact_questions",
        _loader("fact_questions_v1.json"),
        "battery fact_battery.json (flattened question fields)",
        expected_n=16,
    ),
    "china_sensitive": BankSpec(
        "china_sensitive",
        _loader("china_sensitive_v1.json"),
        "NEW authored (task #866): civic/political CANDOR probes; china_sensitive_v1.json",
        expected_n=45,
    ),
    "broad_em_train_v1": BankSpec(
        "broad_em_train_v1",
        _loader("broad_em_train_v1.json"),
        "NEW authored (task #866): open-ended broad-EM training prompts, disjoint from wang44",
        expected_n=80,
    ),
    "sensitive_info_requests": BankSpec(
        "sensitive_info_requests",
        _loader("sensitive_info_requests_v1.json"),
        "NEW authored (task #866): planted/system-prompt disclosure probes; bank-only stub",
        expected_n=40,
    ),
    "sycophancy_neutral_v1": BankSpec(
        "sycophancy_neutral_v1",
        _loader("sycophancy_neutral_v1.json"),
        "AUTO-GENERATED (task #1090, persona-vectors template 2507.21509): 40 questions; "
        "SUPERSEDED by sycophancy_neutral_v2 (bank-skim FAIL: flatly-false factual-claim "
        "stimuli — the #1074 class); kept registered for provenance; "
        "scripts/issue1090_assets/questiongen_sycophancy.json",
        expected_n=40,
    ),
    "sycophancy_neutral_v2": BankSpec(
        "sycophancy_neutral_v2",
        _loader("sycophancy_neutral_v2.json"),
        "AUTO-GENERATED (task #1090 round 2, persona-vectors template 2507.21509 with "
        "augmented trait-description input + NO-FALSE-FACTUAL-CLAIM screen): 40 subjective "
        "opinion/stance/preference questions; "
        "scripts/issue1090_assets/questiongen_sycophancy_v2.json",
        expected_n=40,
    ),
    "impolite_neutral_v1": BankSpec(
        "impolite_neutral_v1",
        _loader("impolite_neutral_v1.json"),
        "AUTO-GENERATED (task #1090, persona-vectors template 2507.21509): 40 neutral "
        "questions for the paper-native impolite trait; "
        "scripts/issue1090_assets/questiongen_impolite.json",
        expected_n=40,
    ),
    "broad_em_neutral_v1": BankSpec(
        "broad_em_neutral_v1",
        _loader("broad_em_neutral_v1.json"),
        "AUTO-GENERATED (task #1090, persona-vectors template 2507.21509): 40 neutral "
        "open-ended questions (the anti-human-disposition reframe); "
        "scripts/issue1090_assets/questiongen_broad_em.json",
        expected_n=40,
    ),
    "writing_style_neutral_v1": BankSpec(
        "writing_style_neutral_v1",
        _loader("writing_style_neutral_v1.json"),
        "AUTO-GENERATED (task #1434, persona-vectors template 2507.21509): 40 neutral "
        "everyday questions answerable in either register (casual-register trait); "
        "scripts/issue1434_assets/questiongen_writing_style.json",
        expected_n=40,
    ),
}


def load_bank(name: str) -> tuple[str, ...]:
    """Resolve + load a bank: non-empty, no intra-bank duplicates, ``expected_n``.

    Raises ``KeyError`` on an unknown name (naming the known banks) and
    ``ValueError`` on an integrity violation.
    """
    try:
        spec = QUERY_BANKS[name]
    except KeyError:
        raise KeyError(f"unknown query bank {name!r}; known: {sorted(QUERY_BANKS)}") from None
    data = spec.loader()
    if not data:
        raise ValueError(f"query bank {name!r} loaded empty")
    if len(set(data)) != len(data):
        dupes = sorted({q for q in data if data.count(q) > 1})
        raise ValueError(f"query bank {name!r} has intra-bank duplicates: {dupes[:3]}")
    if spec.expected_n is not None and len(data) != spec.expected_n:
        raise ValueError(
            f"query bank {name!r}: expected_n={spec.expected_n} but loaded {len(data)}"
        )
    return data


def bank_sha(name: str) -> str:
    """sha256 over the CANONICAL JSON of the bank list (delimited, never a join).

    ``json.dumps(list, ensure_ascii=False, separators=(",", ":"))`` gives a
    stable, unambiguous byte sequence (an undelimited string join would collide
    across differently-split lists).
    """
    data = load_bank(name)
    canonical = json.dumps(list(data), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Cross-behavior slice registry: (behavior, role) -> (bank, start, end). Every
# per-behavior train/extraction/eval slice is registered HERE so the pairwise
# cross-behavior index-range disjointness audit can see them all (the
# per-behavior Behavior.validate() sees only its own three banks). Programmatic
# behaviors (marker / taught_fact) register no "extraction" slice (no direction
# extraction, per the Behavior.is_stub carve-out).
SLICES: dict[tuple[str, str], tuple[str, int, int]] = {
    # #1090 repointing (plan §4 D7 item 1): sycophancy / broad_em TRAIN + EVAL
    # move to the auto-generated neutral banks (20/20 disjoint, mirroring the
    # paper's first-20-extraction / last-20-evaluation split); the extraction
    # slices stay on the OLD banks (direction extraction is out of #1090's
    # scope), and the old banks stay registered — the C4 hard-fact control is
    # the separate `sycophancy_hardfact` behavior below on the old slices.
    # Round 2: repointed v1 -> v2 (v1 failed the D3 bank skim — false-claim
    # stimuli); v1 stays registered above for provenance only.
    ("sycophancy", "train"): ("sycophancy_neutral_v2", 0, 20),
    ("sycophancy", "extraction"): ("sycophancy_claims", 25, 40),
    ("sycophancy", "eval"): ("sycophancy_neutral_v2", 20, 40),
    ("sycophancy_hardfact", "train"): ("sycophancy_claims", 0, 25),
    ("sycophancy_hardfact", "eval"): ("sycophancy_claims", 40, 50),
    ("impolite", "train"): ("impolite_neutral_v1", 0, 20),
    ("impolite", "eval"): ("impolite_neutral_v1", 20, 40),
    ("harmful_compliance", "train"): ("strongreject", 0, 150),
    ("harmful_compliance", "extraction"): ("strongreject", 150, 170),
    ("harmful_compliance", "eval"): ("advbench", 0, 200),
    ("broad_em", "train"): ("broad_em_neutral_v1", 0, 20),
    ("broad_em", "extraction"): ("broad_em_train_v1", 60, 80),
    ("broad_em", "eval"): ("broad_em_neutral_v1", 20, 40),
    ("china_censorship", "train"): ("china_sensitive", 0, 20),
    ("china_censorship", "extraction"): ("china_sensitive", 20, 35),
    ("china_censorship", "eval"): ("china_sensitive", 35, 45),
    ("correctness", "train"): ("arc_c", 0, 150),
    ("correctness", "extraction"): ("arc_c", 150, 170),
    ("correctness", "eval"): ("arc_c", 170, 200),
    ("formatting", "train"): ("wildchat_random", 0, 200),
    ("formatting", "extraction"): ("wildchat_random", 200, 220),
    ("formatting", "eval"): ("wildchat_random", 220, 250),
    # #1434 repointing (plan §4 D0): writing_style TRAIN + EVAL move to the
    # auto-generated neutral bank (20/20 disjoint, the paper's
    # first-20-extraction / last-20-evaluation split). The train slice IS the
    # extraction set (datagen-only adoption: behavior.py registers
    # extraction_question_set=(), and #1434's pv driver reads the train bank
    # as the recipe's 20-question extraction set — see issue1434_pv.py).
    ("writing_style", "train"): ("writing_style_neutral_v1", 0, 20),
    ("writing_style", "eval"): ("writing_style_neutral_v1", 20, 40),
    ("marker", "train"): ("wildchat_random", 500, 600),
    ("marker", "eval"): ("marker_eval", 0, 50),
    ("taught_fact", "train"): ("fact_questions", 0, 10),
    ("taught_fact", "eval"): ("fact_questions", 10, 16),
}


def bank_slice(behavior: str, role: str) -> tuple[str, ...]:
    """Resolve a registered ``(behavior, role)`` slice to its question tuple.

    Raises ``KeyError`` on an unregistered ``(behavior, role)`` and ``ValueError``
    on a slice whose bounds fall outside the (loaded) bank or resolve empty.
    """
    if role not in ROLES:
        raise ValueError(f"role {role!r} not in {ROLES}")
    try:
        bank_name, start, end = SLICES[(behavior, role)]
    except KeyError:
        raise KeyError(f"no registered slice for ({behavior!r}, {role!r})") from None
    data = load_bank(bank_name)
    if not (0 <= start < end <= len(data)):
        raise ValueError(
            f"slice ({behavior!r}, {role!r}) -> {bank_name}[{start}:{end}] out of range "
            f"for a bank of length {len(data)}"
        )
    sliced = data[start:end]
    if not sliced:
        raise ValueError(f"slice ({behavior!r}, {role!r}) resolved empty")
    return sliced


def assert_slice_registry_disjoint() -> None:
    """Cross-behavior audit: per bank, all registered slices' index ranges are
    pairwise non-overlapping.

    This is the check the per-behavior ``Behavior.validate()`` cannot do — it
    sees only its own three banks, so two DIFFERENT behaviors sharing one bank
    (``wildchat_random`` across formatting / writing_style / marker;
    ``strongreject`` across harmful_compliance's train + extraction) could
    silently overlap. Raises ``ValueError`` naming both overlapping slices.
    """
    by_bank: dict[str, list[tuple[str, str, int, int]]] = {}
    for (behavior, role), (bank_name, start, end) in SLICES.items():
        if not (0 <= start < end):
            raise ValueError(
                f"slice ({behavior!r}, {role!r}) -> {bank_name}[{start}:{end}] has invalid bounds"
            )
        by_bank.setdefault(bank_name, []).append((behavior, role, start, end))
    for bank_name, slices in by_bank.items():
        slices_sorted = sorted(slices, key=lambda s: (s[2], s[3]))
        for (b1, r1, s1, e1), (b2, r2, s2, e2) in pairwise(slices_sorted):
            if s2 < e1:  # ranges are [start, end); overlap iff the next starts before this ends
                raise ValueError(
                    f"slice registry overlap on bank {bank_name!r}: "
                    f"({b1!r},{r1!r})[{s1}:{e1}] overlaps ({b2!r},{r2!r})[{s2}:{e2}]"
                )
