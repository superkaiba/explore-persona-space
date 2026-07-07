# ruff: noqa: RUF002
"""#1112 mix derivation (CPU-only, deterministic).

Two jobs (plan §4.2):

1. **Row-role derivation on #1090's frozen c3 mix** — ``train_mix.jsonl`` rows
   carry only ``{prompt, completion}`` and ``mix_meta.json`` records only
   counts + input-file sha256s (NO per-row roles), so roles are derived by
   EXACT-matching each mix row against the pinned role sources
   (``datagen_topup/pos.jsonl`` + ``cn.jsonl`` + the generic corpus). The
   partition MUST be exactly 20 pos / 20 cn / 40 generic with no unmatched or
   doubly-matched row — a failed assert BLOCKS training (fail-fast, plan
   assumption 2; round-1 critique binding item).

2. **Marker contrastive mix** — the #508 canonical 1000-row build (200 villain
   positives + 4 × 200 negatives over #472's on-policy ``R_train`` at seed 42),
   vendored from ``origin/issue-508:scripts/dispatch_508.py::
   _build_canonical_training_jsonl`` (the unmerged-branch port; #514 declared
   it re-derivable from data + commit + seed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

logger = logging.getLogger("issue1112.mixes")

EXPECTED_PARTITION = {"pos": 20, "cn": 20, "generic": 40}


def _read_jsonl(path: Path) -> list[dict]:
    """JSONL rows via text-mode file iteration (never splitlines — gotchas.md)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_key(row: dict) -> str:
    """Canonical exact-match key over the TRAINING-visible content of a row.

    Keyed on (prompt messages, completion messages) with sorted JSON keys —
    byte-order-insensitive but content-exact. Extra provenance fields the role
    sources carry (judge scores, tiers) are ignored; the mix rows carry only
    prompt+completion, so the intersection is exactly these two fields.
    """
    return json.dumps(
        {"prompt": row["prompt"], "completion": row["completion"]},
        sort_keys=True,
        ensure_ascii=False,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_row_roles(
    mix_rows: list[dict],
    pos_rows: list[dict],
    cn_rows: list[dict],
    generic_rows: list[dict],
    *,
    expected: dict[str, int] | None = None,
) -> list[str]:
    """Role per mix row ("pos" | "cn" | "generic") by exact content match.

    Raises:
        ValueError: any unmatched row, any doubly-matched row (a row whose
            content appears in >1 role source), or a realized partition that
            is not exactly ``expected`` (default 20/20/40). A failed assert
            blocks training — never a silent fallback.
    """
    exp = dict(EXPECTED_PARTITION if expected is None else expected)
    sources = {
        "pos": {_row_key(r) for r in pos_rows},
        "cn": {_row_key(r) for r in cn_rows},
        "generic": {_row_key(r) for r in generic_rows},
    }
    overlap = (
        (sources["pos"] & sources["cn"])
        | (sources["pos"] & sources["generic"])
        | (sources["cn"] & sources["generic"])
    )
    if overlap:
        raise ValueError(
            f"role sources overlap on {len(overlap)} row(s) — roles would be ambiguous; "
            f"first key prefix: {sorted(overlap)[0][:120]!r}"
        )
    roles: list[str] = []
    unmatched: list[int] = []
    for i, row in enumerate(mix_rows):
        key = _row_key(row)
        hits = [role for role, keys in sources.items() if key in keys]
        if len(hits) == 1:
            roles.append(hits[0])
        elif not hits:
            unmatched.append(i)
            roles.append("UNMATCHED")
        else:  # unreachable given the overlap check; keep fail-loud anyway
            raise ValueError(f"mix row {i} matched multiple role sources: {hits}")
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} mix row(s) matched NO role source (first indices "
            f"{unmatched[:5]}) — refusing to derive the posonly/generic mixes"
        )
    realized = Counter(roles)
    if dict(realized) != exp:
        raise ValueError(
            f"role partition mismatch: realized {dict(realized)} != expected {exp} — "
            "the frozen mix or the role sources drifted; refusing to train"
        )
    return roles


def write_mix(rows: list[dict], path: Path) -> str:
    """Write rows as JSONL; returns the file sha256 (the mix pin)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return sha256_file(path)


def derive_syco_mixes(
    mix_path: Path,
    pos_path: Path,
    cn_path: Path,
    generic_path: Path,
    out_dir: Path,
) -> dict:
    """Build ``c3_posonly_mix.jsonl`` (drop the 20 cn rows, keep order) and
    ``c3_generic_only.jsonl`` (the 40 generic rows alone, keep order).

    Dropping (not backfilling) keeps the generic:positive ratio fixed
    (plan §4.2); row ORDER is the frozen mix's order restricted to the kept
    roles, so the derivation is deterministic given the pinned inputs.

    Returns a manifest dict (counts, shas, role indices) — the caller uploads
    it beside the mixes.
    """
    mix_rows = _read_jsonl(mix_path)
    roles = derive_row_roles(
        mix_rows,
        _read_jsonl(pos_path),
        _read_jsonl(cn_path),
        _read_jsonl(generic_path),
    )
    posonly = [r for r, role in zip(mix_rows, roles, strict=True) if role != "cn"]
    generic_only = [r for r, role in zip(mix_rows, roles, strict=True) if role == "generic"]
    n_pos = sum(1 for role in roles if role == "pos")
    n_gen = sum(1 for role in roles if role == "generic")
    assert len(posonly) == n_pos + n_gen and len(generic_only) == n_gen, (
        len(posonly),
        len(generic_only),
        n_pos,
        n_gen,
    )
    posonly_path = out_dir / "c3_posonly_mix.jsonl"
    generic_path_out = out_dir / "c3_generic_only.jsonl"
    manifest = {
        "source_mix_sha256": sha256_file(mix_path),
        "roles_by_index": roles,
        "posonly": {"path": str(posonly_path), "n_rows": len(posonly)},
        "generic_only": {"path": str(generic_path_out), "n_rows": len(generic_only)},
    }
    manifest["posonly"]["sha256"] = write_mix(posonly, posonly_path)
    manifest["generic_only"]["sha256"] = write_mix(generic_only, generic_path_out)
    (out_dir / "mix_derivation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    logger.info(
        "[mixes] derived posonly (60 rows, sha %s...) + generic_only (40 rows, sha %s...)",
        manifest["posonly"]["sha256"][:12],
        manifest["generic_only"]["sha256"][:12],
    )
    return manifest


# ── Marker contrastive mix (#508 canonical build, vendored) ──────────────────

MARKER_SOURCE_PERSONA = "villain"
MARKER_NEGATIVES = ("medical_doctor", "police_officer", "qwen_default", "comedian")
MARKER_POS_EX = 200
MARKER_NEG_EX_PER_PERSONA = 200


def build_marker_mix(  # noqa: C901 - the vendored #508 builder is one linear pass
    r_train_path: Path,
    out_path: Path,
    *,
    seed: int = 42,
) -> dict:
    """The #508 canonical 1000-row marker training JSONL from #472's R_train.

    Vendored verbatim from ``origin/issue-508:scripts/dispatch_508.py::
    _build_canonical_training_jsonl`` (same personas/questions/seeds/row
    construction), with the loaders resolved against the ON-MAIN #472/#365
    packages. Positives: villain R + "\\n\\n ※"; negatives: 4 personas' R with
    a marker-contamination fail-loud check. Deterministic at ``seed``.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_SEP,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        load_r_artifact,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
        EVAL_QUESTIONS_20,
    )

    persona_bank = dict(EVAL_PERSONAS_24)
    q_train = list(EVAL_QUESTIONS_20[:10])  # #508 load_q_train: first half
    r_train = load_r_artifact(Path(r_train_path))

    def _make_row(system_prompt: str | None, user: str, assistant: str) -> dict:
        msgs: list[dict] = []
        if system_prompt is not None:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user})
        return {"prompt": msgs, "completion": [{"role": "assistant", "content": assistant}]}

    def _has_marker(text: str, token_ids: list[int] | None) -> bool:
        if MARKER_TEXT in text:
            return True
        return token_ids is not None and EXPECTED_MARKER_TOKEN_ID in token_ids

    def _resolve(persona: str, q: str) -> tuple[str, list[int] | None]:
        if persona not in r_train:
            raise KeyError(f"r_train missing persona {persona!r}")
        if q not in r_train[persona]:
            raise KeyError(f"r_train[{persona!r}] missing q {q!r}")
        entry = r_train[persona][q]
        return entry["response_text"], entry.get("response_token_ids")

    def _sample(n: int, rng: random.Random) -> list[str]:
        if n <= len(q_train):
            return rng.sample(q_train, n)
        out: list[str] = []
        while len(out) < n:
            perm = list(q_train)
            rng.shuffle(perm)
            out.extend(perm)
        return out[:n]

    rows: list[dict] = []
    pos_rng = random.Random(seed)
    src_prompt = persona_bank[MARKER_SOURCE_PERSONA]
    for q in _sample(MARKER_POS_EX, pos_rng):
        r_text, r_ids = _resolve(MARKER_SOURCE_PERSONA, q)
        if _has_marker(r_text, r_ids):
            raise AssertionError(f"positive row {MARKER_SOURCE_PERSONA}/{q!r}: marker already in R")
        rows.append(_make_row(src_prompt, q, f"{r_text}{MARKER_SEP}{MARKER_TEXT}"))
    n_positive = len(rows)

    for j_idx, neg in enumerate(MARKER_NEGATIVES):
        # Verbatim #508: every negative (incl. qwen_default, whose bank entry
        # IS the literal Qwen default system prompt) resolves via persona_bank.
        if neg not in persona_bank:
            raise KeyError(f"persona_bank missing negative {neg!r}")
        neg_prompt = persona_bank[neg]
        neg_rng = random.Random(seed + 1000 + j_idx)
        for q in _sample(MARKER_NEG_EX_PER_PERSONA, neg_rng):
            r_text, r_ids = _resolve(neg, q)
            if _has_marker(r_text, r_ids):
                raise AssertionError(f"negative row {neg}/{q!r}: marker contamination in R")
            rows.append(_make_row(neg_prompt, q, r_text))
    n_negative = len(rows) - n_positive
    expected = MARKER_POS_EX + len(MARKER_NEGATIVES) * MARKER_NEG_EX_PER_PERSONA
    if len(rows) != expected:
        raise AssertionError(f"marker mix row count {len(rows)} != expected {expected}")

    random.Random(seed).shuffle(rows)
    sha = write_mix(rows, out_path)
    manifest = {
        "source": MARKER_SOURCE_PERSONA,
        "negatives": list(MARKER_NEGATIVES),
        "pos_ex": MARKER_POS_EX,
        "neg_ex_per_persona": MARKER_NEG_EX_PER_PERSONA,
        "n_total": len(rows),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "marker_text": MARKER_TEXT,
        "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
        "seed": seed,
        "sha256": sha,
        "vendored_from": (
            "origin/issue-508:scripts/dispatch_508.py::_build_canonical_training_jsonl"
        ),
    }
    Path(out_path).with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest
