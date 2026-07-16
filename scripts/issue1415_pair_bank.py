"""Issue #1415 — build the 28-pair context bank -> ``data/issue_1415/pair_bank.json``.

Deterministic (no RNG). 15 matched-query pairs (same user query, different
persona/system-instruction) + 13 cross-query pairs (different queries, #779
system conditions across traits):

- 10 matched pairs from the #685 context-shift condition universe
  (``issue685_context_shift/analysis_tensors/{base,instruct}_context_vectors.pt``
  — both downloaded + metadata-validated; the referenced conditions are asserted
  present in ``condition_names``). Question pool = the 20-question
  ``EVAL_QUESTIONS`` bank both #685 tensors were captured over.
- 5 matched pairs from the #779 per-condition bank (``<trait>__sys<k>_cx.pt``
  files under ``analysis_tensors/pass_a/`` — weakest sys7 -> strongest sys0
  system swap per trait; the sys0/sys7 tensors are downloaded + shape-validated).
- 13 cross-query pairs across #779 traits (strong sys0 of trait_a vs strong
  sys0 of trait_b, different queries).

Downloads use SCOPED single-file ``huggingface_hub.hf_hub_download`` only —
never ``snapshot_download`` (the ~1M-file data repo wedges full-tree
enumeration; see gotchas).

Disjointness filter: any candidate pair whose question text overlaps the
40-question persona-vectors banks (extraction + eval, per trait — evil verbatim
from ``issue779_common.EVIL_ARTIFACTS``; sycophancy/hallucination from the
uploaded ``issue779_monitoring/artifacts/<trait>.json``) by more than 5% is
EXCLUDED. Overlap is computed character-level per pair: the fraction of the
pair's questions that near-verbatim match a bank question (normalized exact
match or ``difflib.SequenceMatcher`` ratio >= 0.9 on the normalized character
strings). NOTE (stated deviation): the #779 pass_a bank's OWN questions ARE the
persona-vectors eval questions and thus fail this mandated filter by
construction, so the #779-derived pairs take their SYSTEM conditions from the
#779 bank while drawing user queries from the shared filtered EVAL_QUESTIONS
pool.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_OUT = REPO_ROOT / "data" / "issue_1415" / "pair_bank.json"

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
I779_PREFIX = "issue779_monitoring"
TRAITS = ("evil", "sycophancy", "hallucination")
MATCHED_779_SYS_PAIR = (7, 0)  # weakest (plain helpful) -> strongest trait induction
CROSS_SYS = 0  # cross-query pairs use the strongest system condition per trait

# #685 persona swaps used for 4 of the 10 matched-query pairs (bare assistant -> persona).
I685_PERSONA_SUBSET = ("villain", "medical_doctor", "french_person", "comedian")

EXPECTED_LAYERS_779 = 28
EXPECTED_HIDDEN = 3584
OVERLAP_FRAC_MAX = 0.05  # plan v5 §"Disjointness check": >5% question overlap -> exclude
NEAR_VERBATIM_RATIO = 0.9


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize(text: str) -> str:
    """Casefold + collapse whitespace + strip edge punctuation (char-level compare basis)."""
    t = re.sub(r"\s+", " ", text.strip().casefold())
    return t.strip(" .?!,;:")


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


# ── persona-vectors 40-question banks (disjointness-filter targets) ────


def _pv_question_banks(provenance: list[dict]) -> dict[str, list[str]]:
    """The per-trait 40-question persona-vectors banks (extraction + eval).

    evil: verbatim in-repo (``issue779_common.EVIL_ARTIFACTS``); sycophancy /
    hallucination: local cache first, else the uploaded
    ``issue779_monitoring/artifacts/<trait>.json`` (scoped hf_hub_download).
    """
    import issue779_common as c779

    banks: dict[str, list[str]] = {}
    for trait in TRAITS:
        if trait == "evil":
            art = c779.EVIL_ARTIFACTS
            local = None
        else:
            local = c779._artifacts_dir() / f"{trait}.json"
            if not local.exists():
                from huggingface_hub import hf_hub_download

                remote = f"{I779_PREFIX}/artifacts/{trait}.json"
                local = Path(hf_hub_download(HF_DATA_REPO, remote, repo_type="dataset"))
                provenance.append(
                    {"repo_path": remote, "sha256": _sha256(local), "role": "pv_bank"}
                )
            with open(local) as f:
                art = json.load(f)
        if "questions" in art:  # raw generated shape
            qs = list(art["questions"])[:40]
        else:  # normalized cache shape
            qs = list(art["extraction_questions"]) + list(art["eval_questions"])
        assert len(qs) == 40, (trait, len(qs))
        banks[trait] = qs
    return banks


def _question_overlaps_bank(question: str, bank_norm: list[str]) -> bool:
    """Near-verbatim character-level match of one question against a bank."""
    nq = _normalize(question)
    for nb in bank_norm:
        if nq == nb:
            return True
        if difflib.SequenceMatcher(None, nq, nb).ratio() >= NEAR_VERBATIM_RATIO:
            return True
    return False


def pair_overlap_frac(pair: dict, bank_norm: list[str]) -> float:
    """Fraction of a pair's (1-2 distinct) questions overlapping the PV banks."""
    questions = sorted({pair["ctx_c"]["user"], pair["ctx_cprime"]["user"]})
    assert 1 <= len(questions) <= 2, questions
    hits = sum(_question_overlaps_bank(q, bank_norm) for q in questions)
    return hits / len(questions)


def apply_disjointness_filter(pairs: list[dict], banks: dict[str, list[str]]) -> tuple:
    """Split candidates into (kept, excluded) under the >5%-overlap rule."""
    bank_norm = [_normalize(q) for qs in banks.values() for q in qs]
    kept, excluded = [], []
    for pair in pairs:
        frac = pair_overlap_frac(pair, bank_norm)
        if frac > OVERLAP_FRAC_MAX:
            excluded.append({"pair_id": pair["pair_id"], "overlap_frac": frac})
        else:
            kept.append(pair)
    return kept, excluded


# ── reused-artifact verification (scoped single-file downloads) ────────


def verify_issue685_artifacts(provenance: list[dict]) -> None:
    """Download + validate both #685 context-vector tensors; assert the
    referenced conditions exist in their 70-condition universe."""
    import issue685_extract_shifts as i685

    from explore_persona_space.analysis.issue685.signed_cosine import (
        _hf_path,
        load_context_vectors,
    )

    for tag in ("base", "instruct"):
        cv = load_context_vectors(tag)  # scoped hf_hub_download + full metadata asserts
        names = set(cv["condition_names"])
        assert "bare__assistant" in names
        for b in i685.BEHAVIORS:
            assert f"assistant__{b}" in names, b
        for p in I685_PERSONA_SUBSET:
            assert f"bare__{p}" in names, p
        local = Path(_hf_path(f"{tag}_context_vectors.pt"))
        provenance.append(
            {
                "repo_path": f"issue685_context_shift/analysis_tensors/{tag}_context_vectors.pt",
                "sha256": _sha256(local),
                "role": "matched_685_source",
            }
        )


def verify_issue779_artifacts(provenance: list[dict]) -> None:
    """Download + shape-validate the #779 pass_a sys-condition tensors used."""
    import torch
    from huggingface_hub import hf_hub_download

    for trait in TRAITS:
        for k in sorted(set(MATCHED_779_SYS_PAIR) | {CROSS_SYS}):
            remote = f"{I779_PREFIX}/analysis_tensors/pass_a/{trait}__sys{k}_cx.pt"
            local = Path(hf_hub_download(HF_DATA_REPO, remote, repo_type="dataset"))
            blob = torch.load(local, map_location="cpu", weights_only=True)
            assert {"cx_last", "cx_mean", "layers"} <= set(blob.keys()), sorted(blob.keys())
            layers = list(blob["layers"])
            assert layers == list(range(EXPECTED_LAYERS_779)), (remote, layers)
            shape = tuple(blob["cx_last"].shape)
            assert shape[1:] == (EXPECTED_LAYERS_779, EXPECTED_HIDDEN), (remote, shape)
            assert shape[0] >= 1, (remote, shape)
            provenance.append(
                {"repo_path": remote, "sha256": _sha256(local), "role": "matched_779_source"}
            )


# ── pair construction (deterministic) ──────────────────────────────────


def build_candidate_pairs() -> list[dict]:
    """The 28 candidate pair records, before the disjointness filter."""
    import issue685_extract_shifts as i685
    import issue779_common as c779

    from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS

    pool = list(EVAL_QUESTIONS)
    assert len(pool) == 20, len(pool)
    sp = c779.EVAL_SYSTEM_PROMPTS
    pairs: list[dict] = []

    # 10 matched-query pairs from the #685 condition universe: bare assistant ->
    # behavior instruction (6) / persona system prompt (4), same user query.
    matched_685_specs = [("behavior", b, i685.BEHAVIORS[b]) for b in i685.BEHAVIORS] + [
        ("persona", p, PERSONAS[p]) for p in I685_PERSONA_SUBSET
    ]
    assert len(matched_685_specs) == 10
    for i, (kind, name, system) in enumerate(matched_685_specs):
        q = pool[i % len(pool)]
        pairs.append(
            {
                "pair_id": f"m685_{i:02d}_{name}",
                "pair_type": "matched",
                "source": "issue685",
                "ctx_c": {"system": None, "user": q},
                "ctx_cprime": {"system": system, "user": q},
                "trait_or_behavior": name,
                "spec_kind": kind,
            }
        )

    # 5 matched-query pairs from the #779 sys-condition bank: weakest (sys7) ->
    # strongest (sys0) trait system prompt, same user query.
    matched_779_traits = ["evil", "sycophancy", "hallucination", "evil", "sycophancy"]
    weak, strong = MATCHED_779_SYS_PAIR
    for i, trait in enumerate(matched_779_traits):
        q = pool[(10 + i) % len(pool)]
        pairs.append(
            {
                "pair_id": f"m779_{i:02d}_{trait}",
                "pair_type": "matched",
                "source": "issue779",
                "ctx_c": {"system": sp[trait][weak], "user": q},
                "ctx_cprime": {"system": sp[trait][strong], "user": q},
                "trait_or_behavior": trait,
                "spec_kind": f"sys{weak}_to_sys{strong}",
            }
        )

    # 13 cross-query pairs: strong (sys0) trait_a with query q_a -> strong
    # (sys0) trait_b with a DIFFERENT query q_b; label = the c' trait.
    combos = [(a, b) for a in TRAITS for b in TRAITS if a != b]
    assert len(combos) == 6
    for k in range(13):
        a, b = combos[k % len(combos)]
        q_c = pool[(2 * k) % len(pool)]
        q_cp = pool[(2 * k + 1) % len(pool)]
        assert q_c != q_cp
        pairs.append(
            {
                "pair_id": f"cross_{k:02d}_{a}_to_{b}",
                "pair_type": "cross",
                "source": "issue779",
                "ctx_c": {"system": sp[a][CROSS_SYS], "user": q_c},
                "ctx_cprime": {"system": sp[b][CROSS_SYS], "user": q_cp},
                "trait_or_behavior": b,
                "spec_kind": f"sys{CROSS_SYS}_cross_trait",
            }
        )

    assert len(pairs) == 28, len(pairs)
    assert len({p["pair_id"] for p in pairs}) == 28
    return pairs


def build_pair_bank(out_path: Path = DEFAULT_OUT) -> dict:
    """Build, filter, verify, and write the pair bank. Returns the bank dict."""
    provenance: list[dict] = []
    banks = _pv_question_banks(provenance)
    candidates = build_candidate_pairs()
    kept, excluded = apply_disjointness_filter(candidates, banks)
    n_matched = sum(p["pair_type"] == "matched" for p in kept)
    n_cross = sum(p["pair_type"] == "cross" for p in kept)
    if n_matched != 15 or n_cross != 13:
        raise RuntimeError(
            f"pair bank counts after disjointness filter: matched={n_matched} (want 15), "
            f"cross={n_cross} (want 13); excluded={excluded}"
        )

    verify_issue685_artifacts(provenance)
    verify_issue779_artifacts(provenance)

    bank = {
        "metadata": {
            "issue": 1415,
            "created_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "question_pool": "explore_persona_space.personas.EVAL_QUESTIONS (20 questions)",
            "disjointness_filter": {
                "rule": f"exclude pairs with >{OVERLAP_FRAC_MAX:.0%} question overlap vs the "
                "40-question persona-vectors banks (normalized exact or "
                f"SequenceMatcher>= {NEAR_VERBATIM_RATIO})",
                "bank_sizes": {t: len(qs) for t, qs in banks.items()},
                "excluded": excluded,
            },
            "provenance": provenance,
        },
        "pairs": kept,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    tmp.replace(out_path)
    print(f"[issue1415_pair_bank] wrote {out_path} ({len(kept)} pairs, {len(excluded)} excluded)")
    return bank


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build the issue-1415 28-pair context bank.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build_pair_bank(args.out)


if __name__ == "__main__":
    main()
