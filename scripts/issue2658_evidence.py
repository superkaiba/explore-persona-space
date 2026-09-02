"""Issue #2658 unit 4a — frozen evidence packets for the evidence-conditioned rubrics.

Builds ``eval_results/issue_2658/evidence_packets.json`` in EXACTLY the shape
``issue2658_text_resolver.resolve_evidence_packet`` consumes:
``{"items": {item_id: {"packet": <dict>, "evidence_sha256": <sha>}}}`` with
``evidence_sha256 = sha256(sorted-keys compact JSON of packet)`` (the
``evidence_packet_sha256`` domain).  Extra top-level keys (``exclusions``,
``coverage``, ``metadata``) are tolerated by the resolver and carried for audit.

Design rules (unit-4 brief):

- **No answer dependence.** Builders take (item_id, upstream sources) ONLY —
  never a generated answer, and never any path under ``raw_completions/`` /
  ``gen_manifest/`` / ``objective_labels/`` (``_assert_not_answer_derived``
  raises ``TestDerivedTransformError``; a packet derived from answers would be
  a test-derived transform, one of unit 1's guarded fail-on conditions).
- **Frozen + content-addressed.** The store is written once; a re-run rebuilds
  the deterministic core from sources and RAISES ``RowHashMismatchError`` on
  any drift (packet bytes or sha) instead of rewriting.  ``--verify``
  recomputes every stored packet's sha AND rebuilds from sources.
- **Excluded, counted, never fabricated.** Frames with no audited ground-truth
  source are EXCLUDED with a per-frame reason recorded in ``exclusions`` —
  never authored from model/world knowledge (the sycophancy dead-end chain is
  documented at ``EXCLUSIONS`` below).
- **Coverage report.** Realized coverage per (row, frame) and per pilot cell
  vs the plan-§8 pilot floor (5 prompts/cell, ``C.PILOT.prompts_per_cell``).

Evidence sources (both committed / odb-resident, sha-pinned):

- ``hallucination|fact_questions``: the #545 fact battery
  (``eval_results/issue_545/batteries/fact_battery.json`` at the pinned commit
  — sparse worktrees lack the checkout, so it is read via ``git show``).  All
  16 bank strings were verified members of the battery's question-list union
  (direct=3, ood_framings=11, entailed=2); the packet pins the canonical fact,
  fact key, and reversal fact.
- ``hallucination|arc_c_factual``: the ARC-Challenge answer key
  (``raw/arc_challenge/test.jsonl``).  Each bank item embeds exactly one raw
  row's stem as a prefix plus its rendered choices (probed: 15/15 unique);
  the packet pins the stem, choices, labels, and correct answer.

CONTENT HYGIENE: only benign banks (``fact_questions``, ``arc_c``) are read;
excluded frames' bank text is never loaded.  Logs carry ids, counts, and
sha256 digests only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_text_resolver as R  # noqa: E402

EVIDENCE_SCHEMA = "i2658-evidence-v1"
EVIDENCE_SHA_DOMAIN = "evidence_packet_sorted_compact_json_utf8"

# The #545 fact battery: the frozen ground truth for the fact_questions frame.
# Pinned to the commit that last touched it (sparse worktrees lack the file).
FACT_BATTERY_COMMIT = "1bac012549632dc0febd48d61a396e2af7fb51ef"
FACT_BATTERY_RELPATH = "eval_results/issue_545/batteries/fact_battery.json"
FACT_BATTERY_LISTS = ("direct", "ood_framings", "entailed", "reversal")

ARC_RAW_RELPATH = "raw/arc_challenge/test.jsonl"

# Paths whose bytes are (or derive from) generated answers.  Any evidence
# source matching one of these fragments is a test-derived transform.
_ANSWER_DERIVED_FRAGMENTS = ("raw_completions", "gen_manifest", "objective_labels")

# (row, frame) -> exclusion reason.  These frames have NO audited ground-truth
# source; packets for them would have to be fabricated, which is banned.
EXCLUSIONS: dict[tuple[str, str], str] = {
    ("sycophancy", "sycophancy_claims"): (
        "no persisted ground truth: the 50-claim panel was LLM-generated as flat "
        "strings with no stored corrections (#503 prep `_wrong_claim_pool_via_claude` "
        "at commit b65e815bf8e, reachable from the #545 battery metadata); the #411/"
        "#612 audited claim pools are disjoint by construction; authoring corrections "
        "from model/world knowledge would fabricate evidence, and unit 4 may make no "
        "LLM calls to audit claims"
    ),
    ("sycophancy", "sycophancy_neutral_v1"): (
        "flat neutral prompt strings (#1090, commit 3c6a2448c49) with no claim "
        "structure and no ground-truth corrections"
    ),
    ("sycophancy", "sycophancy_neutral_v2"): (
        "flat neutral prompt strings (#1090, commit 3c6a2448c49) with no claim "
        "structure and no ground-truth corrections"
    ),
    ("sycophancy", "wildchat_real"): (
        "real-user prompts (WildChat) carry no frozen ground truth; no audited "
        "claim/correction source exists"
    ),
    ("hallucination", "wang44_probes"): (
        "Betley-lineage probe bank; open-ended probes with no ground-truth "
        "reference to check atomic claims against"
    ),
    ("hallucination", "wildchat_real"): (
        "real-user prompts (WildChat) carry no frozen ground truth against which "
        "atomic claims could be checked"
    ),
}


class EvidenceBuildError(C.Issue2658GuardError):
    """Loud failure while building a frozen evidence packet."""


def _assert_not_answer_derived(source: str) -> None:
    """Refuse any evidence source that is (or derives from) generated answers."""
    low = source.lower()
    for frag in _ANSWER_DERIVED_FRAGMENTS:
        if frag in low:
            raise C.TestDerivedTransformError(
                f"evidence source {source!r} matches answer-derived fragment {frag!r}: "
                "packets must never be derived from generated answers (unit-1 guard)"
            )


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Source loaders (fail-loud, provenance-pinned).
# ---------------------------------------------------------------------------
_FACT_BATTERY_CACHE: tuple[dict[str, Any], str] | None = None


def load_fact_battery() -> tuple[dict[str, Any], str]:
    """The pinned #545 fact battery via ``git show`` (returns (body, sha256))."""
    global _FACT_BATTERY_CACHE
    if _FACT_BATTERY_CACHE is not None:
        return _FACT_BATTERY_CACHE
    _assert_not_answer_derived(FACT_BATTERY_RELPATH)
    proc = subprocess.run(
        ["git", "-C", str(R.REPO_ROOT), "show", f"{FACT_BATTERY_COMMIT}:{FACT_BATTERY_RELPATH}"],
        capture_output=True,
        text=False,
        check=True,
    )
    body = json.loads(proc.stdout.decode("utf-8"))
    for key in ("fact_canonical", "fact_key", *FACT_BATTERY_LISTS):
        if key not in body:
            raise EvidenceBuildError(f"fact battery lacks required key {key!r}")
    _FACT_BATTERY_CACHE = (body, _sha_bytes(proc.stdout))
    return _FACT_BATTERY_CACHE


_ARC_RAW_CACHE: tuple[list[dict[str, Any]], str] | None = None


def load_arc_raw() -> tuple[list[dict[str, Any]], str]:
    """Committed ARC-Challenge raw rows (returns (rows, file sha256))."""
    global _ARC_RAW_CACHE
    if _ARC_RAW_CACHE is not None:
        return _ARC_RAW_CACHE
    _assert_not_answer_derived(ARC_RAW_RELPATH)
    path = R.REPO_ROOT / ARC_RAW_RELPATH
    if not path.exists():
        raise EvidenceBuildError(f"ARC raw file missing at {path}")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
    for i, r in enumerate(rows):
        for key in ("question", "choice_labels", "choices", "correct_answer"):
            if key not in r:
                raise EvidenceBuildError(f"ARC raw row {i} lacks required key {key!r}")
    _ARC_RAW_CACHE = (rows, _sha_bytes(raw))
    return _ARC_RAW_CACHE


def _bank_items(bank: str) -> list[str]:
    """Committed query-bank items via the same loader the resolver uses."""
    _assert_not_answer_derived(f"query_banks:{bank}")
    from explore_persona_space.artifacts import banks

    return banks.load_bank(bank)


def _bank_ref_index(item_id: str, expected_bank: str) -> int:
    """Parse ``{bank}#{index}`` and validate the bank name."""
    _, _, ref = R.parse_item_id(item_id)
    head, sep, idx_s = ref.rpartition("#")
    if not sep or head != expected_bank or not idx_s.isdigit():
        raise EvidenceBuildError(
            f"ref {ref!r} of {item_id!r} does not match '{expected_bank}#<index>'"
        )
    return int(idx_s)


# ---------------------------------------------------------------------------
# Packet builders (item_id + frozen upstream sources ONLY — no answers).
# ---------------------------------------------------------------------------
def build_fact_packet(item_id: str) -> dict[str, Any]:
    """Frozen-fact packet for one ``hallucination|fact_questions`` item."""
    idx = _bank_ref_index(item_id, "fact_questions")
    bank = _bank_items("fact_questions")
    if not (0 <= idx < len(bank)):
        raise EvidenceBuildError(f"bank index {idx} out of range for fact_questions")
    question = bank[idx]
    battery, battery_sha = load_fact_battery()
    member_lists = [k for k in FACT_BATTERY_LISTS if question in battery[k]]
    if not member_lists:
        raise EvidenceBuildError(
            f"{item_id!r}: bank item {idx} is not a member of any fact-battery "
            "question list — no frozen evidence exists for it"
        )
    return {
        "schema": EVIDENCE_SCHEMA,
        "row": "hallucination",
        "item_id": item_id,
        "kind": "frozen_fact_battery",
        "evidence": {
            "fact_canonical": battery["fact_canonical"],
            "fact_key": battery["fact_key"],
            "reversal_fact": battery.get("reversal_fact", ""),
        },
        "question_provenance": {
            "bank": "fact_questions",
            "bank_index": idx,
            "battery_lists": member_lists,
        },
        "source": {
            "path": FACT_BATTERY_RELPATH,
            "git_commit": FACT_BATTERY_COMMIT,
            "sha256": battery_sha,
        },
    }


def build_arc_packet(item_id: str) -> dict[str, Any]:
    """Answer-key packet for one ``hallucination|arc_c_factual`` item."""
    idx = _bank_ref_index(item_id, "arc_c")
    bank = _bank_items("arc_c")
    if not (0 <= idx < len(bank)):
        raise EvidenceBuildError(f"bank index {idx} out of range for arc_c")
    rendered = bank[idx]
    rows, raw_sha = load_arc_raw()
    hits = [i for i, r in enumerate(rows) if rendered.startswith(r["question"])]
    if len(hits) != 1:
        raise EvidenceBuildError(
            f"{item_id!r}: {len(hits)} ARC raw rows stem-prefix-match bank item {idx} "
            "(need exactly 1) — the answer key cannot be joined unambiguously"
        )
    r = rows[hits[0]]
    labels = list(r["choice_labels"])
    choices = list(r["choices"])
    if len(labels) != len(choices):
        raise EvidenceBuildError(f"{item_id!r}: ARC row {hits[0]} labels/choices length mismatch")
    if not all(c in rendered for c in choices):
        raise EvidenceBuildError(
            f"{item_id!r}: matched ARC row {hits[0]} choices are not all embedded in the "
            "bank item — wrong join"
        )
    if r["correct_answer"] not in labels:
        raise EvidenceBuildError(
            f"{item_id!r}: correct_answer {r['correct_answer']!r} not in choice_labels"
        )
    correct_idx = labels.index(r["correct_answer"])
    return {
        "schema": EVIDENCE_SCHEMA,
        "row": "hallucination",
        "item_id": item_id,
        "kind": "arc_challenge_answer_key",
        "evidence": {
            "question": r["question"],
            "choice_labels": labels,
            "choices": choices,
            "correct_label": r["correct_answer"],
            "correct_choice": choices[correct_idx],
        },
        "question_provenance": {
            "bank": "arc_c",
            "bank_index": idx,
            "raw_row_index": hits[0],
        },
        "source": {
            "path": ARC_RAW_RELPATH,
            "sha256": raw_sha,
        },
    }


_PACKET_BUILDERS = {
    ("hallucination", "fact_questions"): build_fact_packet,
    ("hallucination", "arc_c_factual"): build_arc_packet,
}


# ---------------------------------------------------------------------------
# Store construction + freeze.
# ---------------------------------------------------------------------------
def evidence_rows() -> list[str]:
    return [row for row in C.ROW_IDS if C.CONSTRUCTS[row].uses_evidence_packet]


def build_store_core() -> dict[str, Any]:
    """Deterministic store core: items + exclusions + coverage (no timestamps)."""
    pins = R.load_pins()["items"]
    rows = evidence_rows()
    items: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    per_frame: dict[tuple[str, str], dict[str, int]] = {}
    for item_id in sorted(pins):
        row, frame, _ref = R.parse_item_id(item_id)
        if row not in rows:
            continue
        key = (row, frame)
        stats = per_frame.setdefault(key, {"n_pinned": 0, "n_covered": 0, "n_excluded": 0})
        stats["n_pinned"] += 1
        if key in EXCLUSIONS:
            stats["n_excluded"] += 1
            exclusions.append(
                {"item_id": item_id, "row": row, "frame": frame, "reason": EXCLUSIONS[key]}
            )
            continue
        builder = _PACKET_BUILDERS.get(key)
        if builder is None:
            raise EvidenceBuildError(
                f"(row={row!r}, frame={frame!r}) has neither a packet builder nor a "
                "recorded exclusion — refusing to silently drop pinned items"
            )
        packet = builder(item_id)
        items[item_id] = {
            "packet": packet,
            "evidence_sha256": R.evidence_packet_sha256(packet),
        }
        stats["n_covered"] += 1

    coverage = _coverage_report(per_frame, set(items))
    return {
        "issue": 2658,
        "schema": EVIDENCE_SCHEMA,
        "sha_domain": EVIDENCE_SHA_DOMAIN,
        "items": items,
        "exclusions": exclusions,
        "coverage": coverage,
        "n_items": len(items),
        "n_excluded": len(exclusions),
    }


def _coverage_report(
    per_frame: dict[tuple[str, str], dict[str, int]], covered_ids: set[str]
) -> dict[str, Any]:
    """Per-(row,frame) + per-pilot-cell coverage vs the plan-§8 pilot floor."""
    floor = C.PILOT.prompts_per_cell
    frames = {
        f"{row}|{frame}": dict(stats, meets_any_coverage=stats["n_covered"] > 0)
        for (row, frame), stats in sorted(per_frame.items())
    }
    cells: dict[str, Any] = {}
    for cw in G.build_cells(rows_filter=evidence_rows()):
        n_cov = sum(1 for iid in cw.item_ids if iid in covered_ids)
        cells[cw.name] = {
            "n_items": len(cw.item_ids),
            "n_covered": n_cov,
            "pilot_floor": floor,
            "meets_pilot_floor": n_cov >= floor,
        }
    return {"pilot_floor_per_cell": floor, "frames": frames, "cells": cells}


def _assert_core_match(existing: dict[str, Any], rebuilt: dict[str, Any]) -> None:
    """Frozen-store immutability: any drift in items/exclusions/coverage RAISES."""
    for key in ("items", "exclusions", "coverage", "n_items", "n_excluded", "sha_domain"):
        if existing.get(key) != rebuilt.get(key):
            raise C.RowHashMismatchError(
                f"frozen evidence store drift at top-level key {key!r}: the store on "
                "disk no longer matches a rebuild from pinned sources"
            )


def verify_store(path: Path) -> dict[str, Any]:
    """Recompute every stored packet sha + rebuild from sources; drift RAISES."""
    if not path.exists():
        raise R.EvidencePacketMissingError(f"no evidence store at {path}")
    existing = json.loads(path.read_text())
    for iid, entry in existing.get("items", {}).items():
        got = R.evidence_packet_sha256(entry["packet"])
        if got != entry["evidence_sha256"]:
            raise C.RowHashMismatchError(
                f"evidence packet drift for {iid!r}: recomputed {got} != stored "
                f"{entry['evidence_sha256']}"
            )
    _assert_core_match(existing, build_store_core())
    return existing


def freeze_evidence(path: Path | None = None) -> dict[str, Any]:
    """Build + freeze the store; an existing store is verified, never rewritten."""
    path = path or R.EVIDENCE_PATH
    core = build_store_core()
    if path.exists():
        existing = json.loads(path.read_text())
        _assert_core_match(existing, core)
        print(f"[evidence] frozen store at {path} verified against rebuild (unchanged)")
        return existing
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    core["metadata"] = as_metadata_dict(git_provenance(), phase="evidence-freeze")
    payload = G.canonical_json(core)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)
    print(f"[evidence] froze {core['n_items']} packets -> {path}")
    return core


def print_report(store: dict[str, Any]) -> None:
    """Digest-only coverage report (ids/counts/shas — never item text)."""
    cov = store["coverage"]
    print(f"[evidence] items={store['n_items']} excluded={store['n_excluded']}")
    for key, stats in cov["frames"].items():
        print(
            f"[evidence] frame {key}: pinned={stats['n_pinned']} "
            f"covered={stats['n_covered']} excluded={stats['n_excluded']}"
        )
    below = {k: v for k, v in cov["cells"].items() if not v["meets_pilot_floor"]}
    print(
        f"[evidence] cells={len(cov['cells'])} below_pilot_floor={len(below)} "
        f"(floor={cov['pilot_floor_per_cell']}/cell)"
    )
    for name, v in sorted(below.items()):
        print(f"[evidence]   BELOW FLOOR {name}: covered={v['n_covered']}/{v['n_items']}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true", help="verify-only; never writes")
    ap.add_argument("--out", type=Path, default=None, help="store path (default: frozen)")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.verify:
        store = verify_store(args.out or R.EVIDENCE_PATH)
        print("[evidence] verify PASS")
    else:
        store = freeze_evidence(args.out)
    print_report(store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
