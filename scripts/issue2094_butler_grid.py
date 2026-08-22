#!/usr/bin/env python3
"""Issue #2094 — butler round ("Option C"): patch bare/persona contexts toward
the NEW 4th prefix ``butler`` at the context-end slot.

Cells: 10 butler matched-query pairs (``bank.build_butler_pairs``:
bare__q<i> -> butler__q<i> and persona__q<i> -> butler__q<i>; butler<->conv
deliberately EXCLUDED — conv's ceiling is null under its own rubric, 0.0/100
in 49/50 of its own anchors) x slot ``ce`` x 30 layer variants (L0..L27 +
joint_mid + joint_all) x dose ``replace`` x arms {steered, null} = 600 hooked
greedy rollouts. Modeled on ``scripts/issue2094_reverse_direction.py`` (the
proven same-shape round); grid machinery reused verbatim from
``issue2094_run``.

Phases (``--phase {anchors,bank,grid,upload}``; anchors runs FIRST):

- ``anchors`` — (a) butler x 5 queries x K=10 unpatched temp-1.0 draws
  (bare/persona anchors are REUSED from the parent's banked artifacts — never
  regenerated) + both-pooling V_a capture; (b) the judge waves THIS round
  needs (sync ``judge_graded``, N=1 draw, ``max_tokens`` 1024): coherence on
  the butler draws, fp-butler/fp-bare/fp-persona on the butler draws, and
  fp-butler on the banked bare/persona anchor texts (their fp-bare/fp-persona
  scores are banked — ``issue2094_reverse_direction.load_anchor_scores``);
  (c) per-pair floor/ceiling for all 10 pairs + the ANCHOR-SEPARATION GATE:
  mean over the 5 bare->butler pairs of (ceiling - floor) >= 0.5, where the
  per-draw contrast is (fp-butler - fp-<a-prefix>)/100 over coherent draws.
  On FAIL the phase writes ``butler_gate.json`` (passed=false) and exits
  ``RC_ANCHOR_GATE`` (a DESIGNED halt: butler is unmeasurable on this bank) —
  and the grid phase REFUSES to run without a PASSed gate file.
- ``bank`` — delegates verbatim to the parent's ``phase_bank`` (a 20-context
  all-layer V bank now that ``bank.build_contexts`` carries the butler
  contexts; injection-exactness gate + capture-parity legs unchanged).
- ``grid`` — the 60 (steered, null) blocks with per-block JSONL checkpointing
  + full-regime resume; ``--pilot`` times ONE cell through this entrypoint
  (the sizing basis). Null donors are drawn (seeded, seed 2094, distinct,
  no-replacement) from the parent's 15 matched-query pairs — every one has a
  DIFFERENT prefix pair from every butler pair BY CONSTRUCTION (butler never
  appears in a parent pair) and the constraint is still asserted per pairing
  (``assert_butler_donor``); realized exactly as the parent's replace-arm
  null: ``norm_match(V_B(donor), V_B(recipient))`` (payload_kind="state"),
  with the realized ``donor_pair_id`` recorded per null row.
- ``upload`` — persistence + the pod sentinel. DEFAULT ``--persist git``
  (task #2300: the HF data repo is AT the Hub's hard 1,000,000-file cap —
  EVERY net-new-file upload fails fleet-wide, size-independent): the phase
  consolidates all round TEXT artifacts (rollouts, anchor texts, coherence
  flags, judge scores/meta/raw, gate + floor/ceiling, run manifest) into a
  FEW line-sharded JSONL/JSON files (9.5 MB shard splits, never gzip) under
  ``eval_results/issue_2094/butler_grid/`` in the repo tree for a git
  commit by the orchestrator's harvest step; tensors (vc_bank, butler
  anchor states) are EXCLUDED from git (eval_results is JSON/text-only)
  and declared REGENERABLE with recipes in the run manifest. ``--persist
  hf`` keeps the original ``upload_folder`` path (fail-loud at the cap by
  construction) for after #2300 lands.

Judge-instrument parity: the rubric templates here are byte-copies of
``scripts/issue2094_judge.py``'s production instrument (equality pinned by
``tests/test_issue2094_butler_grid.py``); the judge driver itself cannot judge
butler rows yet — its ``pair_index`` unions only ``build_pairs`` +
``build_rev_pairs`` and its ``rubric_registry`` loops the frozen 3-prefix
``PREFIX_ORDER`` — so this round's gate judging is self-contained here, and
the later full-grid judge wave needs the (sibling-owned) judge driver to
union ``BANK.build_butler_pairs()`` + register ``fp-butler``.

Pod-side contract: sentinel file + ``[phase=...]`` breadcrumbs ONLY — never a
``scripts/task.py`` shellout. Every phase ends with an explicit ``sys.exit``
(the #1689 interpreter-finalization race).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import torch  # noqa: E402

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL  # noqa: E402
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids  # noqa: E402
from explore_persona_space.experiments.issue1415.steering import generate_batch  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

# Sibling-script imports (script mode puts scripts/ on sys.path[0]; tests and
# -c mode need the explicit insert — the issue2094_smoke_support convention).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_reverse_direction as REV  # noqa: E402
import issue2094_run as RUN  # noqa: E402

logger = logging.getLogger("issue2094.butler")

# ── constants ─────────────────────────────────────────────────────────

B_SEED = BANK.SEED  # 2094 — the parent's seed convention
B_SLOT = "ce"
B_DOSE = "replace"
B_VEC_TYPE = "A"
DEFAULT_OUT_ROOT = Path("/workspace/issue2094_butler_out")
SENTINEL_NAME = "issue-2094-butler-grid-results.json"
BUTLER_HF_PREFIX = f"{RUN.HF_PREFIX}/raw_completions/butler_grid"
BANKED_ANCHORS_HF_PATH = f"{RUN.HF_PREFIX}/raw_completions/anchors/anchors.jsonl"

# The gate bar (brief-registered): mean bare->butler anchor separation on the
# per-draw Delta scale [-1, 1]. NOT a CLI knob — a pre-registered constant.
GATE_MIN_SEPARATION = 0.5
COHERENCE_THRESHOLD = 60.0  # the parent's _coherent() bar (issue2094_analysis)

RC_ANCHOR_GATE = 23  # a DESIGNED halt is never an anonymous rc=1

FP_BUTLER = "fp-butler"
FP_BARE = "fp-bare"
FP_PERSONA = "fp-persona"
COHERENCE_RUBRIC_ID = "coherence"
JUDGE_N_DRAWS = 1  # parity with the parent's banked anchor scores (N=1/draw)
DEFAULT_JUDGE_MAX_TOKENS = 1024  # llm-judging rule-23 single-rationale floor

BUTLER_CONTEXT_IDS: tuple[str, ...] = tuple(
    BANK.context_id(BANK.BUTLER_PREFIX, q) for q in BANK.QUERY_ORDER
)

# ── judge instrument (BYTE-parity with issue2094_judge.py, test-pinned) ──

REASON_THEN_SCORE = (
    "First give a brief reasoning (1-3 sentences), then output a single JSON "
    'object of the form {"reasoning": "<your brief justification>", '
    '"score": <an integer from 0 to 100>} and nothing else after it.'
)


def coherence_eval_prompt() -> str:
    """Form-only coherence instrument (byte-copy of the judge driver's)."""
    return f"{BANK.COHERENCE_RUBRIC}\n\n[RESPONSE]\n{{answer}}\n[/RESPONSE]\n\n{REASON_THEN_SCORE}"


def behavior_eval_prompt(core: str) -> str:
    """Production behavior instrument (byte-copy of the judge driver's)."""
    return f"{core}\n\n[RESPONSE]\n{{answer}}\n[/RESPONSE]\n\n{REASON_THEN_SCORE}"


def butler_rubric_registry() -> dict[str, str]:
    """The 4 rubrics this round's gate judging dispatches."""
    return {
        COHERENCE_RUBRIC_ID: coherence_eval_prompt(),
        FP_BUTLER: behavior_eval_prompt(BANK.f_prefix_rubric("butler")),
        FP_BARE: behavior_eval_prompt(BANK.f_prefix_rubric("bare")),
        FP_PERSONA: behavior_eval_prompt(BANK.f_prefix_rubric("persona")),
    }


def _item_id(tag: str, source_key: str) -> str:
    """Deterministic content-derived id (the judge driver's convention)."""
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
    return f"{tag}{digest}"


# ── pairs + donor assignment (CPU-pure, unit-tested) ──────────────────


def build_butler_pairs() -> list[BANK.Pair]:
    """Delegating alias for :func:`BANK.build_butler_pairs` (registry parity
    with the rev round's ``BANK.build_rev_pairs`` convention)."""
    pairs = BANK.build_butler_pairs()
    for p in pairs:
        assert p.prefix_b == BANK.BUTLER_PREFIX, p
        assert p.prefix_pair() in (("bare", "butler"), ("persona", "butler")), p
    return pairs


def butler_donor_pool(parent_pairs: list[BANK.Pair]) -> list[BANK.Pair]:
    """The parent's 15 matched-query pairs — the null-donor pool.

    Every member's prefix pair differs from every butler pair's BY
    CONSTRUCTION (butler appears in no parent pair), and every member's
    target context (``donor.b``) is a persona/conv context — never the
    recipient's butler target — so the parent's state-kind eligibility holds
    for the whole pool (still asserted per pairing).
    """
    pool = sorted(
        (p for p in parent_pairs if p.setting == "matched_query"),
        key=lambda p: p.pair_id,
    )
    assert len(pool) == 15, [p.pair_id for p in pool]
    return pool


def assert_butler_donor(pair: BANK.Pair, donor: BANK.Pair) -> None:
    """FAIL-LOUD null-donor validation for one (recipient, donor) pairing.

    Refuses: self-donation, a SAME-prefix-pair donor (the parent's
    different-prefix-pair null constraint — the brief's explicit refusal
    case), a butler-bearing donor (donors come from the PARENT bank so the
    null payload is a non-butler state), a non-matched-query donor, and a
    donor sharing the recipient's target slot state (the parent's state-kind
    eligibility, ``RUN._donor_eligible(..., payload_kind="state")``).
    """
    assert donor.pair_id != pair.pair_id, f"self-donation: {pair.pair_id}"
    assert donor.setting == "matched_query", (donor.pair_id, donor.setting)
    assert donor.prefix_pair() != pair.prefix_pair(), (
        f"same-prefix-pair donor refused: recipient {pair.pair_id} "
        f"{pair.prefix_pair()} vs donor {donor.pair_id} {donor.prefix_pair()}"
    )
    assert BANK.BUTLER_PREFIX not in donor.prefix_pair(), (
        f"butler-bearing donor refused (pool is the PARENT bank): {donor.pair_id}"
    )
    assert RUN._donor_eligible(donor, B_SLOT, pair, "state"), (
        pair.pair_id,
        donor.pair_id,
    )


def butler_donor_assignment(
    butler_pairs: list[BANK.Pair],
    parent_pairs: list[BANK.Pair],
    seed: int = B_SEED,
) -> dict[str, str]:
    """Seeded donor map (butler pair_id -> parent mq pair_id), constraints asserted.

    Distinct donors (a seeded no-replacement sample over the sorted 15-pair
    pool), every pairing validated by :func:`assert_butler_donor`.
    """
    pool = butler_donor_pool(parent_pairs)
    rng = random.Random(seed)
    donor_ids = rng.sample([p.pair_id for p in pool], k=len(butler_pairs))
    by_id = {p.pair_id: p for p in pool}
    out: dict[str, str] = {}
    for pair, donor_id in zip(butler_pairs, donor_ids, strict=True):
        assert_butler_donor(pair, by_id[donor_id])
        out[pair.pair_id] = donor_id
    return out


def enumerate_butler_blocks(
    butler_pairs: list[BANK.Pair], n_layers: int
) -> list[tuple[RUN.Block, RUN.Block]]:
    """30 (steered, null) families: ce x every layer variant x replace x Type A."""
    ids = tuple(p.pair_id for p in butler_pairs)
    assert ids, "empty butler pair set"
    families = [
        (
            RUN.Block(B_SLOT, variant, B_DOSE, B_VEC_TYPE, "steered", ids),
            RUN.Block(B_SLOT, variant, B_DOSE, B_VEC_TYPE, "null", ids),
        )
        for variant in RUN.layer_variant_names(n_layers)
    ]
    keys = [b.key for fam in families for b in fam]
    assert len(set(keys)) == len(keys), "duplicate butler block keys"
    return families


def smoke_butler_blocks(
    butler_pairs: list[BANK.Pair], n_layers: int
) -> list[tuple[RUN.Block, RUN.Block]]:
    """Tiny per-arm-class slice: one single-layer + joint_mid + joint_all
    family (both arms each; every family carries all 10 pairs, so BOTH
    prefix-pair families and every donor-null pairing run in every class)."""
    variants = RUN.layer_variant_names(n_layers)
    keep = {variants[n_layers // 2], "joint_mid", "joint_all"}
    return [
        f for f in enumerate_butler_blocks(butler_pairs, n_layers) if f[0].layer_variant in keep
    ]


def butler_regime_fingerprint(cfg: RUN.RunConfig, bank_sha: str, donor_map: dict[str, str]) -> str:
    """Resume key: the parent regime fingerprint + the butler grid's identity.

    Distinct from the parent's AND the rev round's fingerprints BY
    CONSTRUCTION, so a foreign done-file can never satisfy a butler-grid
    resume even under a mis-pointed ``--out-root``.
    """
    payload = json.dumps(
        {
            "base": RUN.regime_fingerprint(cfg, bank_sha),
            "grid": "butler_grid_v1",
            "pairs": [p.pair_id for p in build_butler_pairs()],
            "donors": donor_map,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── phase dirs (butler round's own sub-roots under --out-root) ────────


def butler_anchors_dir(cfg: RUN.RunConfig) -> Path:
    return cfg.out_root / "butler_anchors"


def judge_dir(cfg: RUN.RunConfig) -> Path:
    return cfg.out_root / "judge"


def gate_path(cfg: RUN.RunConfig) -> Path:
    return cfg.manifest_dir / "butler_gate.json"


def floor_ceiling_path(cfg: RUN.RunConfig) -> Path:
    return cfg.manifest_dir / "butler_floor_ceiling.json"


# ── anchors phase: generation ─────────────────────────────────────────


def _anchors_done(cfg: RUN.RunConfig, regime_fp: str, expected_draws: int) -> bool:
    """Butler-anchor resume predicate: done-manifest + regime + artifacts + rows."""
    path = butler_anchors_dir(cfg) / "anchors_done.json"
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    if rec.get("regime_fp") != regime_fp:
        raise RuntimeError(
            f"butler anchors done-file carries regime_fp={rec.get('regime_fp')!r} but "
            f"this run's regime_fp={regime_fp!r} — refusing to resume across regimes "
            "(quarantine or use a fresh --out-root)"
        )
    if int(rec.get("draws", -1)) != expected_draws:
        logger.warning(
            "[butler-anchors] done draws=%s but this run wants %d — re-running",
            rec.get("draws"),
            expected_draws,
        )
        return False
    jsonl = butler_anchors_dir(cfg) / "butler_anchors.jsonl"
    va = butler_anchors_dir(cfg) / "va_butler_anchors.pt"
    if not (jsonl.exists() and va.exists()):
        logger.warning("[butler-anchors] done-manifest present but artifacts missing — re-running")
        return False
    n_rows = sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
    if n_rows != int(rec.get("n_rows", -1)):
        logger.warning(
            "[butler-anchors] done n_rows=%s but file has %d rows — re-running",
            rec.get("n_rows"),
            n_rows,
        )
        return False
    return True


@torch.no_grad()
def _generate_butler_anchors(cfg: RUN.RunConfig, regime_fp: str, draws: int) -> list[dict]:
    """Butler x K unpatched temp-1.0 draws + both-pooling V_a (parent recipe)."""
    out_dir = butler_anchors_dir(cfg)
    jsonl_path = out_dir / "butler_anchors.jsonl"
    if _anchors_done(cfg, regime_fp, draws):
        logger.info("[butler-anchors] generation already done for this regime — skipping")
        return list(
            REV.json.loads(line) for line in jsonl_path.open(encoding="utf-8") if line.strip()
        )
    model, tok = RUN.load_model_and_tokenizer(cfg)
    contexts = BANK.build_contexts()
    order = list(BUTLER_CONTEXT_IDS)
    ctx_list = [contexts[c] for c in order]
    eot = RUN.eot_tail_ids(tok)
    t0 = time.monotonic()
    outs = generate_batch(
        model,
        tok,
        ctx_list,
        n=draws,
        hook=None,
        max_new_tokens=cfg.max_new_tokens,
        temperature=RUN.ANCHOR_TEMPERATURE,
        seed_base=cfg.seed_base,
        # History-aware render for exact parent parity (butler contexts are
        # single-turn, but the ids MUST match the bank capture's — bank.py note).
        render_fn=BANK.render_context_2094,
        ids_fn=BANK.context_token_ids_2094,
    )
    logger.info(
        "[butler-anchors] %d contexts x %d draws in %.1fs",
        len(order),
        draws,
        time.monotonic() - t0,
    )
    ctx_ids = {cid: BANK.context_token_ids_2094(tok, contexts[cid]) for cid in order}
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    rows: list[dict] = []
    for b, cid in enumerate(order):
        for i, text in enumerate(outs[b]):
            flat_ctx.append(ctx_ids[cid])
            flat_text.append(text)
            rows.append(
                {
                    "context_id": cid,
                    "draw": i,
                    "seed": cfg.seed_base + i,
                    "temperature": RUN.ANCHOR_TEMPERATURE,
                    "text": text,
                }
            )
    states = RUN.capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    for r, n_tok in zip(rows, states["n_completion_tokens"], strict=True):
        r["n_completion_tokens"] = n_tok
        r["cap_hit"] = RUN.cap_hit(n_tok, cfg.max_new_tokens)
        r["cap_hit_basis"] = "retokenized_completion_len >= max_new_tokens"
    RUN._write_jsonl_atomic(jsonl_path, rows)
    RUN._save_pt_atomic(
        out_dir / "va_butler_anchors.pt",
        {
            "layers": cfg.layers,
            "index": [{"context_id": r["context_id"], "draw": r["draw"]} for r in rows],
            "va_span": states["va_span"],
            "va_tail": states["va_tail"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "repro": RUN._repro(cfg),
        },
    )
    cap_hits = sum(1 for r in rows if r["cap_hit"])
    RUN._write_json_atomic(
        out_dir / "anchors_done.json",
        {
            "regime_fp": regime_fp,
            "n_contexts": len(order),
            "draws": draws,
            "n_rows": len(rows),
            "n_cap_hit": cap_hits,
            "cap_hit_frac": cap_hits / max(1, len(rows)),
            "n_empty": len(states["empty_rows"]),
            "repro": RUN._repro(cfg),
        },
    )
    logger.info("[butler-anchors] rows=%d cap_hit=%d", len(rows), cap_hits)
    return rows


# ── anchors phase: banked inputs ──────────────────────────────────────


def load_banked_anchor_texts(
    banked_anchors_file: Path | None, out_root: Path
) -> dict[tuple[str, int], str]:
    """The parent's bare/persona anchor TEXTS ({(context_id, draw): text}).

    ``banked_anchors_file`` given: read it (a local mirror of the parent's
    ``anchors.jsonl``). ``None``: fetch from the HF data repo (retry-wrapped).
    Observed schema (parent ``phase_anchors`` writer + the HF-banked copy):
    rows carry context_id / draw / seed / text / n_completion_tokens /
    cap_hit / cap_hit_basis / temperature.
    """
    if banked_anchors_file is None:
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate.hub import retry_transient

        local = retry_transient(
            lambda: hf_hub_download(
                repo_id=RUN.HF_DATA_REPO,
                repo_type="dataset",
                filename=BANKED_ANCHORS_HF_PATH,
                local_dir=out_root / "anchors_dl",
            ),
            what=f"hf_hub_download {BANKED_ANCHORS_HF_PATH}",
        )
        banked_anchors_file = Path(local)
    assert banked_anchors_file.is_file(), f"missing banked anchors file: {banked_anchors_file}"
    out: dict[tuple[str, int], str] = {}
    for line in banked_anchors_file.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        cid = str(row["context_id"])
        if cid.split("__")[0] not in ("bare", "persona"):
            continue
        key = (cid, int(row["draw"]))
        assert key not in out, f"duplicate banked anchor row {key}"
        out[key] = str(row["text"])
    assert out, f"no bare/persona rows parsed from {banked_anchors_file}"
    return out


# ── anchors phase: judge waves ────────────────────────────────────────


def compose_judge_waves(
    butler_rows: list[dict], banked_texts: dict[tuple[str, int], str]
) -> dict[str, dict]:
    """The 6 gate waves: {wave_name: {rubric_id, items, keys}}.

    ``items`` are ``(item_id, question, answer)`` triples; ``keys`` maps
    item_id -> (context_id, draw) for the score-table join. Item ids are
    content-derived (the judge driver's convention) and Batch-grammar
    validated per wave.
    """
    butler_by_key = {(r["context_id"], r["draw"]): r["text"] for r in butler_rows}
    assert len(butler_by_key) == len(butler_rows), "duplicate butler (context, draw) rows"

    def _items(source: dict[tuple[str, int], str], rid: str, tag: str):
        items, keys = [], {}
        for (cid, draw), text in sorted(source.items()):
            iid = _item_id(tag, f"butler-gate|{rid}|{cid}|{draw}")
            items.append((iid, BANK.QUERIES[cid.split("__")[1]], text))
            keys[iid] = (cid, draw)
        return {"rubric_id": rid, "items": items, "keys": keys}

    bare_texts = {k: v for k, v in banked_texts.items() if k[0].startswith("bare__")}
    persona_texts = {k: v for k, v in banked_texts.items() if k[0].startswith("persona__")}
    assert bare_texts and persona_texts, (len(bare_texts), len(persona_texts))
    waves = {
        "coherence.butler-anchors": _items(butler_by_key, COHERENCE_RUBRIC_ID, "bc"),
        "fp-butler.butler-anchors": _items(butler_by_key, FP_BUTLER, "bb"),
        "fp-bare.butler-anchors": _items(butler_by_key, FP_BARE, "bd"),
        "fp-persona.butler-anchors": _items(butler_by_key, FP_PERSONA, "bp"),
        "fp-butler.bare-anchors": _items(bare_texts, FP_BUTLER, "ba"),
        "fp-butler.persona-anchors": _items(persona_texts, FP_BUTLER, "bs"),
    }
    for wave, spec in waves.items():
        ids = [iid for iid, _q, _a in spec["items"]]
        assert len(set(ids)) == len(ids), f"duplicate item ids in {wave}"
        validate_batch_custom_ids(ids)
    return waves


def _wave_regime(
    rubric_id: str, prompt: str, items: list, judge_model: str, max_tokens: int
) -> dict:
    ids_sha = hashlib.sha256("\n".join(sorted(i for i, _q, _a in items)).encode()).hexdigest()[:16]
    return {
        "rubric_id": rubric_id,
        "rubric_sha16": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "n_items": len(items),
        "item_ids_sha16": ids_sha,
        "judge_model": judge_model,
        "max_tokens": max_tokens,
        "n_draws": JUDGE_N_DRAWS,
    }


def run_judge_wave(
    cfg: RUN.RunConfig,
    wave: str,
    spec: dict,
    prompt: str,
    judge_model: str,
    max_tokens: int,
    dry_run: bool,
) -> dict[tuple[str, int], float | None] | None:
    """One gate wave: dispatch -> one bounded transport retry -> scores JSONL.

    Resumable: a completed wave (meta present, regime match) reloads its
    persisted scores; a regime mismatch REFUSES (never a silent cross-regime
    reuse). Returns {(context_id, draw): score-or-None}; None = content drop
    (rule 9, never coerced). ``dry_run`` builds + validates + routes with
    ZERO API calls and returns None.
    """
    jdir = judge_dir(cfg)
    meta_path = jdir / f"{wave}.meta.json"
    scores_path = jdir / f"{wave}.scores.jsonl"
    regime = _wave_regime(spec["rubric_id"], prompt, spec["items"], judge_model, max_tokens)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("regime") == regime and meta.get("complete"):
            logger.info("[wave %s] complete (meta matches regime) — skip", wave)
            return _load_wave_scores(scores_path, spec["rubric_id"])
        if meta.get("regime") != regime:
            raise RuntimeError(
                f"wave {wave}: existing meta carries a DIFFERENT regime — refusing to "
                f"resume across regimes (quarantine {jdir} or use a fresh --out-root). "
                f"existing={meta.get('regime')} new={regime}"
            )
    result = judge_graded(
        spec["items"],
        prompt,
        n_draws=JUDGE_N_DRAWS,
        cache_dir=cfg.out_root / "judge_cache" / spec["rubric_id"],
        save_raw=jdir / "raw" / f"{wave}.json",
        judge_model=judge_model,
        max_tokens=max_tokens,
        dry_run=dry_run,
    )
    if dry_run:
        logger.info("[wave %s] dry-run complete (no API calls)", wave)
        return None
    # Bounded transport retry (rule 24: retried, never persisted as drops).
    retry_telemetry = None
    lost = {i for i, n in result.per_item_transport_losses.items() if n > 0}
    if lost:
        logger.info("[wave %s] transport retry: %d items", wave, len(lost))
        retry_items = [it for it in spec["items"] if it[0] in lost]
        result2 = judge_graded(
            retry_items,
            prompt,
            n_draws=JUDGE_N_DRAWS,
            cache_dir=cfg.out_root / "judge_cache" / spec["rubric_id"],
            save_raw=jdir / "raw" / f"{wave}.retry1.json",
            judge_model=judge_model,
            max_tokens=max_tokens,
        )
        retry_telemetry = {
            "n_total_draws": result2.n_total_draws,
            "n_transport_lost_draws": result2.n_transport_lost_draws,
        }
        for iid in lost:
            result.scores[iid] = result2.scores.get(iid)
            result.per_item_transport_losses[iid] = result2.per_item_transport_losses.get(iid, 0)
    rows = []
    for iid, _q, _a in spec["items"]:
        cid, draw = spec["keys"][iid]
        rows.append(
            {
                "item_id": iid,
                "rubric_id": spec["rubric_id"],
                "kind": "anchor",  # the banked scores files' observed schema
                "context_id": cid,
                "draw": draw,
                "score": result.scores.get(iid),
                "transport_lost_residual": result.per_item_transport_losses.get(iid, 0),
            }
        )
    RUN._write_jsonl_atomic(scores_path, rows)
    residual = sum(r["transport_lost_residual"] for r in rows)
    meta = {
        "regime": regime,
        "complete": True,
        "n_scored_items": sum(1 for r in rows if r["score"] is not None),
        "n_content_dropped": sum(
            1 for r in rows if r["score"] is None and r["transport_lost_residual"] == 0
        ),
        "residual_transport_lost": residual,
        "n_api_refusal_draws": result.n_api_refusal_draws,
        "n_truncation_dropped_draws": result.n_truncation_dropped_draws,
        "stop_reason_tally": dict(result.stop_reason_tally),
        "retry1": retry_telemetry,
        "repro": RUN._repro(cfg),
    }
    RUN._write_json_atomic(meta_path, meta)
    logger.info(
        "[wave %s] done: %d/%d scored, %d content drops, residual transport %d",
        wave,
        meta["n_scored_items"],
        len(rows),
        meta["n_content_dropped"],
        residual,
    )
    return _load_wave_scores(scores_path, spec["rubric_id"])


def _load_wave_scores(path: Path, rubric_id: str) -> dict[tuple[str, int], float | None]:
    out: dict[tuple[str, int], float | None] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        assert row["rubric_id"] == rubric_id, (row["rubric_id"], rubric_id)
        score = row["score"]
        if score is not None:
            score = float(score)
            assert 0.0 <= score <= 100.0, (row, path)
        out[(str(row["context_id"]), int(row["draw"]))] = score
    assert out, f"no rows parsed from {path}"
    return out


# ── anchors phase: floor / ceiling + gate ─────────────────────────────


def _coherent_draws_from_scores(
    coh: dict[tuple[str, int], float | None],
) -> dict[str, list[int]]:
    """Coherent butler draws per context (score > 60; None = unjudged = excluded)."""
    out: dict[str, list[int]] = {}
    for (cid, draw), score in sorted(coh.items()):
        if score is not None and score > COHERENCE_THRESHOLD:
            out.setdefault(cid, []).append(draw)
    return out


def compute_butler_floor_ceiling(
    scores: dict[str, dict[tuple[str, int], float | None]],
    banked_coherent: dict[str, dict],
    butler_coherent: dict[str, list[int]],
    butler_pairs: list[BANK.Pair],
) -> dict:
    """Per-pair floor/ceiling from anchor judge scores, coherent draws only.

    Per-draw Delta = (fp-butler - fp-<a-prefix>)/100. floor = mean under
    context A (bare/persona — banked coherence flags), ceiling = mean under
    context B (butler — this round's coherence wave). A draw missing EITHER
    rubric score is DROPPED from that side (counted); zero kept draws on a
    side fails loud.
    """
    per_pair: list[dict] = []
    for pair in butler_pairs:
        rubric_a = f"fp-{pair.prefix_a}"
        sides: dict[str, dict] = {}
        for role, cid in (("floor", pair.a), ("ceiling", pair.b)):
            if role == "floor":
                assert cid in banked_coherent, f"context {cid} absent from banked coherence"
                draws = list(banked_coherent[cid]["coherent"])
            else:
                assert cid in butler_coherent, f"context {cid} has no coherent butler draws"
                draws = list(butler_coherent[cid])
            assert draws, f"context {cid} has zero coherent anchor draws ({role})"
            deltas, dropped = [], []
            for d in draws:
                sb = scores[FP_BUTLER].get((cid, d))
                sa = scores[rubric_a].get((cid, d))
                if sb is None or sa is None:
                    dropped.append(d)
                    continue
                deltas.append((float(sb) - float(sa)) / 100.0)
            assert deltas, (
                f"context {cid} ({role}): every coherent draw missing a judge score "
                f"(rubrics {FP_BUTLER}/{rubric_a}) — never default a score"
            )
            sides[role] = {
                "context_id": cid,
                "draws_coherent": draws,
                "n_draws_kept": len(deltas),
                "draws_dropped_missing_score": dropped,
                "per_draw_delta": deltas,
                "delta_mean": sum(deltas) / len(deltas),
            }
        per_pair.append(
            {
                "pair_id": pair.pair_id,
                "setting": pair.setting,
                "prefix_pair": list(pair.prefix_pair()),
                "context_a": pair.a,
                "context_b": pair.b,
                "rubric_b": FP_BUTLER,
                "rubric_a": rubric_a,
                "floor": sides["floor"],
                "ceiling": sides["ceiling"],
                "denominator": sides["ceiling"]["delta_mean"] - sides["floor"]["delta_mean"],
            }
        )
    return {
        "direction": "bare/persona -> butler (Option C matched-query)",
        "delta_definition": (
            "per-draw (fp-butler - fp-<a-prefix>)/100 over coherent anchor draws; "
            "floor = mean under context A (banked bare/persona anchors), ceiling = "
            "mean under context B (this round's butler anchors)"
        ),
        "pairs": per_pair,
    }


def evaluate_gate(fc: dict, butler_pairs: list[BANK.Pair], anchor_draws: int) -> dict:
    """The ANCHOR-SEPARATION GATE: mean bare->butler denominator >= 0.5.

    Gated on the 5 bare<->butler pairs ONLY (the brief's "butler vs bare");
    the persona<->butler denominators are reported informationally. HARD
    gate: the grid phase refuses to run without ``passed: true``.
    """
    by_family: dict[str, list[float]] = {}
    for rec in fc["pairs"]:
        fam = "bare" if rec["prefix_pair"] == ["bare", "butler"] else "persona"
        by_family.setdefault(fam, []).append(float(rec["denominator"]))
    assert set(by_family) == {"bare", "persona"}, sorted(by_family)
    assert len(by_family["bare"]) == 5 and len(by_family["persona"]) == 5, by_family
    mean_bare = sum(by_family["bare"]) / len(by_family["bare"])
    passed = mean_bare >= GATE_MIN_SEPARATION
    return {
        "criterion": (
            "butler anchor-separation gate: mean over the 5 bare->butler pairs of "
            "(ceiling - floor) on the per-draw Delta scale >= "
            f"{GATE_MIN_SEPARATION} (HARD gate — on FAIL butler is unmeasurable "
            "on this bank and the grid must not run)"
        ),
        "passed": bool(passed),
        "gate_min_separation": GATE_MIN_SEPARATION,
        "mean_denominator_bare_butler": mean_bare,
        "per_pair_denominators_bare_butler": by_family["bare"],
        "mean_denominator_persona_butler_informational": (
            sum(by_family["persona"]) / len(by_family["persona"])
        ),
        "per_pair_denominators_persona_butler": by_family["persona"],
        "pair_ids": [p.pair_id for p in butler_pairs],
        "anchor_draws": anchor_draws,
        "rubrics": [FP_BUTLER, FP_BARE, FP_PERSONA, COHERENCE_RUBRIC_ID],
        "coherence_threshold": COHERENCE_THRESHOLD,
    }


def _require_butler_gate(cfg: RUN.RunConfig, butler_pairs: list[BANK.Pair]) -> None:
    """Grid-entry gate check: butler_gate.json present, PASSed, same pair set."""
    path = gate_path(cfg)
    if not path.is_file():
        raise RuntimeError(
            f"butler anchor-separation gate report missing: {path} — run "
            "`--phase anchors` first (the gate is HARD; the grid never runs ungated)"
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("pair_ids") != [p.pair_id for p in butler_pairs]:
        raise RuntimeError(
            f"butler gate report at {path} covers a DIFFERENT pair set — refusing "
            "(quarantine or use a fresh --out-root)"
        )
    if not rec.get("passed"):
        raise RuntimeError(
            f"butler anchor-separation gate FAILED per {path} "
            f"(mean bare->butler denominator "
            f"{rec.get('mean_denominator_bare_butler')!r} < {GATE_MIN_SEPARATION}): "
            "butler is unmeasurable on this bank — the grid must not run"
        )


def phase_anchors(cfg: RUN.RunConfig, args: argparse.Namespace) -> int:
    """Butler anchors + gate judging + the anchor-separation gate."""
    logger.info("[phase=butler-anchors] smoke=%s", cfg.smoke)
    _manifest, bank_sha = RUN.bank_manifest_and_sha()
    butler_pairs = build_butler_pairs()
    parent_pairs = BANK.build_pairs()
    donor_map = butler_donor_assignment(butler_pairs, parent_pairs)
    regime_fp = butler_regime_fingerprint(cfg, bank_sha, donor_map)

    # >= 2 draws even under --smoke (the parent anchors-phase convention).
    draws = 2 if cfg.smoke else cfg.anchor_draws
    butler_rows = _generate_butler_anchors(cfg, regime_fp, draws)

    # Banked inputs (CPU/network only): bare+persona anchor texts, their
    # banked fp scores, and their committed coherence flags.
    banked_texts = load_banked_anchor_texts(args.banked_anchors_file, cfg.out_root)
    banked_scores = REV.load_anchor_scores(args.scores_dir, cfg.out_root)
    banked_coherent = REV.load_coherent_draws(args.coherence_path)

    registry = butler_rubric_registry()
    waves = compose_judge_waves(butler_rows, banked_texts)
    results: dict[str, dict[tuple[str, int], float | None]] = {}
    for wave, spec in waves.items():
        res = run_judge_wave(
            cfg,
            wave,
            spec,
            registry[spec["rubric_id"]],
            args.judge_model,
            args.judge_max_tokens,
            dry_run=cfg.smoke,
        )
        if res is not None:
            for key, score in res.items():
                results.setdefault(spec["rubric_id"], {})[key] = score
    if cfg.smoke:
        logger.info(
            "[butler-anchors] SMOKE: judge waves dry-run only — gate SKIPPED "
            "(no gate file written; the production anchors phase computes it)"
        )
        logger.info("[phase=butler-anchors_done]")
        return RUN.RC_OK

    # Score table: fresh butler-round waves + the banked fp-bare/fp-persona.
    scores: dict[str, dict[tuple[str, int], float | None]] = {
        FP_BUTLER: results[FP_BUTLER],
        FP_BARE: {**banked_scores[FP_BARE], **results.get(FP_BARE, {})},
        FP_PERSONA: {**banked_scores[FP_PERSONA], **results.get(FP_PERSONA, {})},
    }
    butler_coherent = _coherent_draws_from_scores(results[COHERENCE_RUBRIC_ID])
    coh_rows = [
        {
            "context_id": cid,
            "draw": draw,
            "coherence_score": score,
            "coherent": score is not None and score > COHERENCE_THRESHOLD,
            "cap_hit": next(
                r["cap_hit"] for r in butler_rows if (r["context_id"], r["draw"]) == (cid, draw)
            ),
        }
        for (cid, draw), score in sorted(results[COHERENCE_RUBRIC_ID].items())
    ]
    RUN._write_jsonl_atomic(butler_anchors_dir(cfg) / "butler_anchor_draws.jsonl", coh_rows)

    fc = compute_butler_floor_ceiling(scores, banked_coherent, butler_coherent, butler_pairs)
    RUN._write_json_atomic(floor_ceiling_path(cfg), {**fc, "repro": RUN._repro(cfg)})
    gate = evaluate_gate(fc, butler_pairs, draws)
    RUN._write_json_atomic(
        gate_path(cfg), {**gate, "regime_fp": regime_fp, "repro": RUN._repro(cfg)}
    )
    logger.info(
        "[butler-gate] mean bare->butler denominator=%.4f (bar %.2f) -> %s",
        gate["mean_denominator_bare_butler"],
        GATE_MIN_SEPARATION,
        "PASS" if gate["passed"] else "FAIL",
    )
    if not gate["passed"]:
        logger.error(
            "[butler-gate] FAIL: butler is unmeasurable on this bank — the grid "
            "must not run (rc=%d, a DESIGNED halt)",
            RC_ANCHOR_GATE,
        )
        logger.info("[phase=butler-anchors_done]")
        return RC_ANCHOR_GATE
    logger.info("[phase=butler-anchors_done]")
    return RUN.RC_OK


# ── the butler grid ───────────────────────────────────────────────────


@torch.no_grad()
def run_butler_block(
    cfg: RUN.RunConfig,
    model,
    tok,
    bank: dict,
    block: RUN.Block,
    butler_pairs_by_id: dict[str, BANK.Pair],
    parent_pairs_by_id: dict[str, BANK.Pair],
    donor_by_id: dict[str, str],
    regime_fp: str,
    *,
    write_done: bool = True,
    shard_prefix: str = "shard_",
) -> dict:
    """One butler block: hooked greedy rollouts for every cell (no V_a pass).

    Mirrors the rev round's ``run_rev_block`` geometry (same hook arming, same
    history-aware render, same shard row schema); ``write_done=False`` + a
    non-``shard_`` prefix isolate the ``--pilot`` timing cell from the
    production resume/upload globs.
    """
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK.context_token_ids_2094(tok, contexts[cid])
        return ctx_ids_cache[cid]

    layers = RUN.layer_variant_layers(block.layer_variant, cfg.n_layers)
    mode, alpha, payload_kind = RUN._realized_mode(block.slot, block.dose, block.vec_type)
    assert (mode, alpha, payload_kind) == ("replace", 1.0, "state"), (
        f"the butler grid is context-end full-state replace only, got {block.key}"
    )
    recs = bank["per_context"]
    cells: list[dict] = []
    for pid in block.pair_ids:
        pair = butler_pairs_by_id[pid]
        _delta, state, m = RUN._pair_payload(bank, pair, block.slot, block.vec_type)
        recipient = state  # payload_kind == "state": the full-state patch V_B
        donor_label = None
        if block.arm == "null":
            donor = parent_pairs_by_id[donor_by_id[pid]]
            assert_butler_donor(pair, donor)
            # The parent's replace-arm null realization: the donor pair's
            # TARGET-CONTEXT STATE V_B(donor), norm-matched to the recipient's
            # V_B (bank.norm_match inside _donor_payload).
            recipient, donor_label = RUN._donor_payload(
                bank, pair, donor, block.slot, block.vec_type, recipient, payload_kind
            )
        rec = recs[pair.a]
        pos = RUN.slot_positions(rec["ctx_len"], rec["prefix_end"], block.slot)[-m:]
        assert recipient.shape[0] == len(pos) == 1, (recipient.shape, pos)
        cells.append(
            {
                "pair_id": pid,
                "setting": pair.setting,
                "context_a": pair.a,
                "context_b": pair.b,
                "positions": list(pos),
                "payload": recipient,
                "donor_pair_id": donor_label,
            }
        )

    texts: list[str] = []
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
        hook = RUN._arm_hook_for_rows(
            model,
            cfg,
            layers,
            row_lengths,
            [tuple(c["positions"]) for c in chunk],
            [c["payload"] for c in chunk],
            mode,
            alpha,
            t_pad,
        )
        try:
            outs = generate_batch(
                model,
                tok,
                ctx_list,
                n=1,
                hook=hook,
                max_new_tokens=cfg.max_new_tokens,
                temperature=RUN.GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
                # History-aware render for exact parent parity (every butler
                # context_a is single-turn: bare__q<i> / persona__q<i>).
                render_fn=BANK.render_context_2094,
                ids_fn=BANK.context_token_ids_2094,
            )
        finally:
            hook.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts.extend(o[0] for o in outs)
    assert len(texts) == len(cells), (len(texts), len(cells))

    rows_out: list[dict] = []
    for cell, text in zip(cells, texts, strict=True):
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
        rows_out.append(
            {
                "block_key": block.key,
                "slot": block.slot,
                "layer_variant": block.layer_variant,
                "layers": list(layers),
                "dose": block.dose,
                "alpha": alpha,
                "hook_mode": mode,
                "realized_mode": "replace",
                "vec_type": block.vec_type,
                "arm": block.arm,
                "pair_id": cell["pair_id"],
                "setting": cell["setting"],
                "context_a": cell["context_a"],
                "context_b": cell["context_b"],
                "positions": cell["positions"],
                "donor_pair_id": cell["donor_pair_id"],
                "temperature": RUN.GRID_TEMPERATURE,
                "seed": cfg.seed_base,
                "n_completion_tokens": n_tok,
                "cap_hit": RUN.cap_hit(n_tok, cfg.max_new_tokens),
                "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                "text": text,
            }
        )
    RUN._write_jsonl_atomic(cfg.rollouts_dir / f"{shard_prefix}{block.slug}.jsonl", rows_out)
    done = {
        "key": block.key,
        "regime_fp": regime_fp,
        "n_cells": len(rows_out),
        "n_cap_hit": sum(1 for r in rows_out if r["cap_hit"]),
        "repro": RUN._repro(cfg),
    }
    if write_done:
        RUN._write_json_atomic(RUN.block_done_path(cfg.out_root, block), done)
    return done


def _upload_butler_increment(cfg: RUN.RunConfig, blocks: list[RUN.Block]) -> list[str]:
    """Incremental per-block-batch text upload under the butler_grid prefix."""
    slugs = [b.slug for b in blocks if (cfg.rollouts_dir / f"shard_{b.slug}.jsonl").exists()]
    if not slugs:
        return []
    return RUN._upload_dir(
        cfg, cfg.rollouts_dir, BUTLER_HF_PREFIX, [f"shard_{s}.jsonl" for s in slugs]
    )


def phase_grid(cfg: RUN.RunConfig, args: argparse.Namespace) -> int:
    """The 600-cell butler grid (or the 1-cell ``--pilot`` timing run)."""
    logger.info(
        "[phase=butler-grid] worker=%d/%d smoke=%s pilot=%s",
        cfg.worker_index,
        cfg.num_workers,
        cfg.smoke,
        cfg.pilot,
    )
    butler_pairs = build_butler_pairs()
    # HARD gate check BEFORE any bank/model load: the grid never runs ungated.
    try:
        _require_butler_gate(cfg, butler_pairs)
    except RuntimeError as exc:
        logger.error("[butler-grid] %s", exc)
        return RC_ANCHOR_GATE
    bank = RUN._load_bank(cfg)
    parent_pairs = BANK.build_pairs()
    butler_by_id = {p.pair_id: p for p in butler_pairs}
    parent_by_id = {p.pair_id: p for p in parent_pairs}
    donor_by_id = butler_donor_assignment(butler_pairs, parent_pairs)
    regime_fp = butler_regime_fingerprint(cfg, str(bank.get("bank_sha")), donor_by_id)

    all_families = enumerate_butler_blocks(butler_pairs, cfg.n_layers)
    totals_all = RUN.grid_totals(all_families)
    families = smoke_butler_blocks(butler_pairs, cfg.n_layers) if cfg.smoke else all_families
    if cfg.pilot:
        families = families[:1]
    blocks = RUN.blocks_for_worker(families, cfg.worker_index, cfg.num_workers)
    totals = RUN.grid_totals(families)
    RUN._write_json_atomic(
        cfg.manifest_dir / f"butler_grid_plan_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,
            "smoke": cfg.smoke,
            "pilot": cfg.pilot,
            "totals_full_grid": totals_all,
            "totals_this_run": totals,
            "n_blocks_this_worker": len(blocks),
            "n_cells_this_worker": sum(b.n_cells for b in blocks),
            "block_keys": [b.key for b in blocks],
            "donor_assignment": donor_by_id,
            "donor_pool": [p.pair_id for p in butler_donor_pool(parent_pairs)],
            "repro": RUN._repro(cfg),
        },
    )
    logger.info(
        "[butler-grid] full grid: %d blocks / %d cells; this run: %d blocks / %d cells",
        totals_all["n_blocks"],
        totals_all["cells_total"],
        len(blocks),
        sum(b.n_cells for b in blocks),
    )
    assert blocks, "butler grid resolved to zero blocks (never silently no-op)"

    model, tok = RUN.load_model_and_tokenizer(cfg)

    if cfg.pilot:
        steered = blocks[0]
        assert steered.arm == "steered", steered.key
        pilot_block = RUN.Block(
            steered.slot,
            steered.layer_variant,
            steered.dose,
            steered.vec_type,
            steered.arm,
            steered.pair_ids[:1],
        )
        t0 = time.monotonic()
        rec = run_butler_block(
            cfg,
            model,
            tok,
            bank,
            pilot_block,
            butler_by_id,
            parent_by_id,
            donor_by_id,
            regime_fp,
            write_done=False,
            shard_prefix="pilot_shard_",
        )
        wall = time.monotonic() - t0
        per_cell = wall / rec["n_cells"]
        projected_h = per_cell * totals_all["cells_total"] / max(1, cfg.num_workers) / 3600.0
        RUN._write_json_atomic(
            cfg.out_root / "butler_pilot_report.json",
            {
                "criterion": "butler-grid timing pilot (sizing basis)",
                "block_key": pilot_block.key,
                "measured_cells": rec["n_cells"],
                "measured_wall_s": wall,
                "s_per_cell": per_cell,
                "gen_batch": cfg.gen_batch,
                "num_workers": cfg.num_workers,
                "cells_total": totals_all["cells_total"],
                "projected_wall_h": projected_h,
                "repro": RUN._repro(cfg),
            },
        )
        logger.info(
            "[butler-pilot] s_per_cell=%.3f cells_total=%d projected_wall_h=%.3f",
            per_cell,
            totals_all["cells_total"],
            projected_h,
        )
        logger.info("[phase=butler-pilot_done]")
        return RUN.RC_OK

    n_total = len(blocks)
    done_count = 0
    ran_cells = 0
    uploaded: list[str] = []
    pending: list[RUN.Block] = []
    for k, block in enumerate(blocks, start=1):
        if RUN.block_is_done(cfg.out_root, block, regime_fp):
            done_count += 1
            logger.info("[butler-grid] block %d/%d %s SKIP (done)", k, n_total, block.key)
            continue
        t0 = time.monotonic()
        rec = run_butler_block(
            cfg, model, tok, bank, block, butler_by_id, parent_by_id, donor_by_id, regime_fp
        )
        elapsed = time.monotonic() - t0
        ran_cells += rec["n_cells"]
        pending.append(block)
        logger.info("[butler-grid] block %d/%d %s elapsed=%.1fs", k, n_total, block.key, elapsed)
        if cfg.upload_every > 0 and len(pending) >= cfg.upload_every:
            uploaded += _upload_butler_increment(cfg, pending)
            pending = []
    if pending:
        uploaded += _upload_butler_increment(cfg, pending)

    RUN._write_json_atomic(
        cfg.manifest_dir / f"butler_grid_done_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "n_blocks": n_total,
            "n_blocks_skipped": done_count,
            "n_cells_run": ran_cells,
            "uploads": uploaded,
            "repro": RUN._repro(cfg),
        },
    )
    logger.info(
        "[phase=butler-grid_done] worker=%d blocks=%d cells=%d",
        cfg.worker_index,
        n_total,
        ran_cells,
    )
    return RUN.RC_OK


# ── upload + sentinel ─────────────────────────────────────────────────


def _sentinel_payload(
    cfg: RUN.RunConfig,
    uploaded: dict[str, list[str]],
    *,
    persist: str = "hf",
    payload_root: Path | None = None,
) -> dict:
    """The /issue Step 7 results payload (all 10 keys), butler-grid numbers.

    ``persist="git"``: eval_paths lead with the git payload root, hf_hub_url
    is None (nothing landed on HF this round), and plan_deviations carry the
    #2300 routing record so upload-verification reconciles against git.
    """
    n_shards = len(list(cfg.rollouts_dir.glob("shard_*.jsonl")))
    cap_hits, cap_total = 0, 0
    for done in sorted((cfg.manifest_dir / "blocks").glob("*.done.json")):
        rec = json.loads(done.read_text())
        cap_hits += int(rec.get("n_cap_hit", 0))
        cap_total += int(rec.get("n_cells", 0))
    gate = json.loads(gate_path(cfg).read_text()) if gate_path(cfg).exists() else {}
    fc_path = floor_ceiling_path(cfg)
    assert persist in ("git", "hf"), persist
    eval_paths = {
        str(cfg.rollouts_dir),
        str(butler_anchors_dir(cfg)),
        str(judge_dir(cfg)),
        str(gate_path(cfg)),
        str(fc_path),
        str(cfg.manifest_dir),
    }
    deviations_extra: list[str] = []
    if persist == "git":
        assert payload_root is not None
        eval_paths.add(str(payload_root))
        deviations_extra.append(
            "persistence routed to GIT this round (#2300: the HF data repo is at the "
            "Hub's hard 1,000,000-file cap — every net-new-file upload fails, "
            "size-independent): rollout text + anchor text + coherence flags + judge "
            f"scores/meta/raw + gate + floor/ceiling + run manifest at {payload_root} "
            "(low-file-count line-sharded JSONL, no gzip); tensors (vc_bank, butler "
            "anchor states) NOT persisted — regen recipes in run_manifest.json "
            "(eval_results is JSON/text-only). Upload-verification reconciles against "
            "the GIT payload, not HF."
        )
    return {
        "eval_numbers": {
            "butler_grid_shards": n_shards,
            "cells_persisted": cap_total,
            "cap_hit_rows": cap_hits,
            "cap_hit_frac": (cap_hits / cap_total) if cap_total else 0.0,
            "anchor_gate_passed": bool(gate.get("passed")),
            "anchor_gate_mean_denominator_bare_butler": gate.get("mean_denominator_bare_butler"),
            "floor_ceiling_pairs": (
                len(json.loads(fc_path.read_text())["pairs"]) if fc_path.exists() else 0
            ),
        },
        "eval_paths": sorted(eval_paths),
        "reproducibility_card": {
            **RUN._repro(cfg),
            "seed_base": cfg.seed_base,
            "butler_seed": B_SEED,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": RUN.GRID_TEMPERATURE,
            "anchor_temperature": RUN.ANCHOR_TEMPERATURE,
            "anchor_draws": cfg.anchor_draws,
            "gen_batch": cfg.gen_batch,
            "num_workers": cfg.num_workers,
            "slot": B_SLOT,
            "dose": B_DOSE,
            "vec_type": B_VEC_TYPE,
            "judge_model": DEFAULT_JUDGE_MODEL,
            "judge_max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
            "gate_min_separation": GATE_MIN_SEPARATION,
        },
        "wandb_url": None,
        "hf_hub_url": (
            None
            if persist == "git"  # nothing landed on HF this round (#2300)
            else f"https://huggingface.co/datasets/{RUN.HF_DATA_REPO}/tree/main/{BUTLER_HF_PREFIX}"
        ),
        "worktree_path": str(RUN.REPO_ROOT),
        "final_commit_sha": RUN._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "butler<->conv pairs are EXCLUDED by design (conv's ceiling is null "
            "under its own rubric — 0.0/100 in 49/50 of its own anchors)",
            "bare/persona anchors + their fp-bare/fp-persona scores + coherence "
            "flags are REUSED from the parent's banked artifacts — no regeneration; "
            "NEW judge calls in this round: butler-draw coherence + fp rubrics and "
            "fp-butler on the banked bare/persona draws (~350 sync calls)",
            "null donors are drawn (seeded, seed 2094, distinct) from the parent's "
            "15 matched-query pairs: different-prefix-pair holds by construction "
            "(butler appears in no parent pair) and is asserted per pairing; "
            "realized as the parent's replace-arm donor-STATE null "
            "(norm_match(V_B(donor), V_B)), donor_pair_id recorded per null row",
            *deviations_extra,
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
    }


DEFAULT_GIT_PAYLOAD_ROOT = RUN.REPO_ROOT / "eval_results" / "issue_2094" / "butler_grid"
SHARD_MAX_BYTES = 9_500_000  # text >9.5 MB per file line-splits (upload-policy)

# Tensors are EXCLUDED from the git payload (eval_results is JSON/text-only)
# and regenerable from the pinned commit + the persisted TEXT — recorded in
# the run manifest so the discard is verifier-legible, never silent.
DISCARDED_TENSOR_RECIPES = {
    "vc_bank (20-context all-layer V bank, *.pt)": (
        "re-run `--phase bank` at the run manifest's git commit (deterministic "
        "capture: model + bank.py + seed; injection gate re-verifies)"
    ),
    "va_butler_anchors.pt (butler anchor answer states)": (
        "one teacher-forced RUN.capture_answer_states pass over the persisted "
        "butler_anchors JSONL texts at the run manifest's git commit"
    ),
}


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration (never splitlines — U+2028-in-strings safe)."""
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write_jsonl_sharded(
    dest_dir: Path, stem: str, rows: list[dict], max_bytes: int = SHARD_MAX_BYTES
) -> list[Path]:
    """Write ``rows`` as ``<stem>.shardNN.jsonl`` line shards of <= max_bytes.

    Never gzip (``*.gz`` is LFS-matched); at least one shard is always
    written for a non-empty row set; an EMPTY row set fails loud (a
    well-formed empty artifact is the #1739 misdiagnosis trap).
    """
    assert rows, f"refusing to write an EMPTY payload artifact {stem} (never a silent no-op)"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[str] = []
    size = 0
    for row in rows:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        nbytes = len(line.encode("utf-8"))
        if buf and size + nbytes > max_bytes:
            paths.append(dest_dir / f"{stem}.shard{len(paths):02d}.jsonl")
            paths[-1].write_text("".join(buf), encoding="utf-8")
            buf, size = [], 0
        buf.append(line)
        size += nbytes
    paths.append(dest_dir / f"{stem}.shard{len(paths):02d}.jsonl")
    paths[-1].write_text("".join(buf), encoding="utf-8")
    return paths


def build_git_payload(cfg: RUN.RunConfig, payload_root: Path) -> dict:
    """Consolidate the round's TEXT artifacts into a low-file-count git payload.

    Returns a summary dict (per-class file lists + row counts). Fail-loud on
    an empty rollout set; anchor/judge/gate classes are included when present
    (the grid can be uploaded before judging on a resumed pod) with each
    absence logged loud — never silently swallowed.
    """
    payload_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    # 1. rollout TEXT (never a legal discard): all per-block shards -> one sharded set.
    block_shards = sorted(cfg.rollouts_dir.glob("shard_*.jsonl"))
    rollout_rows = [row for p in block_shards for row in _read_jsonl(p)]
    assert rollout_rows, (
        f"no grid rollout rows staged under {cfg.rollouts_dir} — refusing to build "
        "an empty git payload (rollout text is never a legal discard)"
    )
    paths = _write_jsonl_sharded(payload_root, "rollouts", rollout_rows)
    summary["rollouts"] = {
        "files": [p.name for p in paths],
        "n_rows": len(rollout_rows),
        "n_source_blocks": len(block_shards),
    }

    # 2. butler anchor TEXT + coherence flags.
    anchors_jsonl = butler_anchors_dir(cfg) / "butler_anchors.jsonl"
    if anchors_jsonl.is_file():
        rows = _read_jsonl(anchors_jsonl)
        paths = _write_jsonl_sharded(payload_root, "butler_anchors", rows)
        summary["butler_anchors"] = {"files": [p.name for p in paths], "n_rows": len(rows)}
    else:
        logger.warning("[git-payload] no butler anchor text at %s — NOT included", anchors_jsonl)
    draws_jsonl = butler_anchors_dir(cfg) / "butler_anchor_draws.jsonl"
    if draws_jsonl.is_file():
        rows = _read_jsonl(draws_jsonl)
        paths = _write_jsonl_sharded(payload_root, "butler_anchor_draws", rows)
        summary["butler_anchor_draws"] = {"files": [p.name for p in paths], "n_rows": len(rows)}
    else:
        logger.warning("[git-payload] no coherence flags at %s — NOT included", draws_jsonl)

    # 3. judge outputs: scores (+wave field), meta, raw rationales.
    jdir = judge_dir(cfg)
    score_rows: list[dict] = []
    metas: dict[str, dict] = {}
    raw_rows: list[dict] = []
    for scores_path in sorted(jdir.glob("*.scores.jsonl")):
        wave = scores_path.name[: -len(".scores.jsonl")]
        score_rows += [{"wave": wave, **row} for row in _read_jsonl(scores_path)]
        meta_path = jdir / f"{wave}.meta.json"
        if meta_path.is_file():
            metas[wave] = json.loads(meta_path.read_text(encoding="utf-8"))
    for raw_path in sorted((jdir / "raw").glob("*.json")):
        raw_rows.append(
            {"wave": raw_path.stem, "raw": json.loads(raw_path.read_text(encoding="utf-8"))}
        )
    if score_rows:
        paths = _write_jsonl_sharded(payload_root, "judge_scores", score_rows)
        summary["judge_scores"] = {
            "files": [p.name for p in paths],
            "n_rows": len(score_rows),
            "waves": sorted(metas),
        }
        RUN._write_json_atomic(payload_root / "judge_meta.json", metas)
        summary["judge_meta"] = {"files": ["judge_meta.json"], "n_waves": len(metas)}
    else:
        logger.warning("[git-payload] no judge score rows under %s — NOT included", jdir)
    if raw_rows:
        paths = _write_jsonl_sharded(payload_root, "judge_raw", raw_rows)
        summary["judge_raw"] = {"files": [p.name for p in paths], "n_waves": len(raw_rows)}

    # 4. gate + floor/ceiling verbatim copies.
    for name, src in (
        ("butler_gate", gate_path(cfg)),
        ("butler_floor_ceiling", floor_ceiling_path(cfg)),
    ):
        if src.is_file():
            (payload_root / f"{name}.json").write_text(src.read_text(encoding="utf-8"))
            summary[name] = {"files": [f"{name}.json"]}
        else:
            logger.warning("[git-payload] %s missing at %s — NOT included", name, src)

    # 5. run manifest: plans/done manifests + pilot + anchors-done + repro +
    #    the verifier-legible tensor-discard record.
    manifests: dict[str, dict] = {}
    for p in sorted(cfg.manifest_dir.glob("butler_grid_*.json")):
        manifests[p.name] = json.loads(p.read_text(encoding="utf-8"))
    for extra in (
        cfg.out_root / "butler_pilot_report.json",
        butler_anchors_dir(cfg) / "anchors_done.json",
    ):
        if extra.is_file():
            manifests[extra.name] = json.loads(extra.read_text(encoding="utf-8"))
    RUN._write_json_atomic(
        payload_root / "run_manifest.json",
        {
            "persist_mode": "git (#2300: HF data repo at the 1,000,000-file Hub cap)",
            "discarded_tensors": DISCARDED_TENSOR_RECIPES,
            "manifests": manifests,
            "repro": RUN._repro(cfg),
        },
    )
    summary["run_manifest"] = {"files": ["run_manifest.json"], "n_manifests": len(manifests)}

    n_files = sum(len(v.get("files", [])) for v in summary.values())
    for cls, rec in summary.items():
        logger.info(
            "[git-payload] %s: %s rows=%s", cls, ",".join(rec["files"]), rec.get("n_rows", "-")
        )
    logger.info("[git-payload] %d files total -> %s", n_files, payload_root)
    return summary


def phase_upload(cfg: RUN.RunConfig, args: argparse.Namespace) -> int:
    """Persist the staged butler-round artifacts, then write the sentinel.

    ``--persist git`` (DEFAULT, #2300): consolidated low-file-count text
    payload into the repo tree (the orchestrator's harvest step commits it —
    this driver never runs git). ``--persist hf``: the original HF
    ``upload_folder`` path, which FAILS LOUD at the current file cap
    (``RUN._upload_dir`` raises after its bounded retry — never swallowed).
    """
    logger.info("[phase=upload] persist=%s", args.persist)
    if args.persist == "git":
        payload_root = args.git_payload_root or DEFAULT_GIT_PAYLOAD_ROOT
        logger.info(
            "[upload] HF data repo persistence DISABLED this round (#2300: repo at the "
            "1,000,000-file Hub cap) — writing the consolidated git payload instead; "
            "tensors excluded per the eval_results JSON/text-only policy, regen "
            "recipes recorded in run_manifest.json"
        )
        payload_summary = build_git_payload(cfg, Path(payload_root))
        payload = _sentinel_payload(
            cfg,
            {cls: rec.get("files", []) for cls, rec in payload_summary.items()},
            persist="git",
            payload_root=Path(payload_root),
        )
        RUN._write_json_atomic(cfg.manifest_dir / "butler_upload_done.json", payload)
        sentinel = cfg.log_dir / SENTINEL_NAME
        RUN._write_json_atomic(
            sentinel,
            {"sentinel_schema_version": 1, "kind": "epm:results", "version": 1, "note": payload},
        )
        logger.info("[upload] sentinel written: %s", sentinel)
        logger.info("[phase=upload_done]")
        return RUN.RC_OK

    assert args.persist == "hf", args.persist
    uploaded: dict[str, list[str]] = {}
    uploaded["butler_text"] = RUN._upload_dir(
        cfg, cfg.rollouts_dir, BUTLER_HF_PREFIX, ["shard_*.jsonl"]
    )
    uploaded["butler_manifests"] = RUN._upload_dir(
        cfg, cfg.manifest_dir, f"{BUTLER_HF_PREFIX}/manifests", ["*.json", "blocks/*.done.json"]
    )
    uploaded["butler_anchors_text"] = RUN._upload_dir(
        cfg, butler_anchors_dir(cfg), f"{BUTLER_HF_PREFIX}/anchors", ["*.jsonl", "*.json"]
    )
    uploaded["butler_anchors_tensors"] = RUN._upload_dir(
        cfg,
        butler_anchors_dir(cfg),
        f"{RUN.HF_PREFIX}/analysis_tensors/butler_anchors",
        ["*.pt"],
    )
    uploaded["butler_judge"] = RUN._upload_dir(
        cfg,
        judge_dir(cfg),
        f"{BUTLER_HF_PREFIX}/judge",
        ["*.jsonl", "*.json", "raw/*.json"],
    )
    # This run's own V bank (fresh 20-context capture on this pod) — the
    # parent's bank lives at analysis_tensors/vc_bank, this one beside it.
    uploaded["butler_vc_bank"] = RUN._upload_dir(
        cfg,
        cfg.bank_dir,
        f"{RUN.HF_PREFIX}/analysis_tensors/butler_vc_bank",
        ["*.pt", "*.json"],
    )
    payload = _sentinel_payload(cfg, uploaded, persist="hf", payload_root=None)
    RUN._write_json_atomic(cfg.manifest_dir / "butler_upload_done.json", payload)
    sentinel = cfg.log_dir / SENTINEL_NAME
    RUN._write_json_atomic(
        sentinel,
        {"sentinel_schema_version": 1, "kind": "epm:results", "version": 1, "note": payload},
    )
    logger.info("[upload] sentinel written: %s", sentinel)
    logger.info("[phase=upload_done]")
    return RUN.RC_OK


# ── entrypoint ────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2094 butler-round driver (anchors / bank / grid / upload)."
    )
    ap.add_argument(
        "--phase",
        choices=("anchors", "bank", "grid", "upload"),
        help="pipeline phase to run (required unless --import-check)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import (incl. function-body imports) and exit 0",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--log-dir", type=Path, default=RUN.DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=RUN.MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    ap.add_argument("--gen-batch", type=int, default=16, help="cells per hooked generate call")
    ap.add_argument("--capture-batch", type=int, default=8, help="bank/anchor capture batch")
    ap.add_argument("--max-new-tokens", type=int, default=RUN.MAX_NEW_TOKENS)
    ap.add_argument(
        "--anchor-draws",
        type=int,
        default=RUN.ANCHOR_DRAWS,
        help="butler anchor draws per context (K=10 matches the parent waves)",
    )
    ap.add_argument("--seed-base", type=int, default=RUN.SEED_BASE)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny per-arm-class block slice; anchors: 2 draws + judge DRY-RUN (no gate)",
    )
    ap.add_argument(
        "--pilot", action="store_true", help="grid: ONE timed cell through this entrypoint"
    )
    ap.add_argument(
        "--force", action="store_true", help="bank: deliberately re-run a completed phase"
    )
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument(
        "--persist",
        choices=("git", "hf"),
        default="git",
        help="upload-phase persistence: git = consolidated line-sharded text payload "
        "into eval_results/issue_2094/butler_grid/ (DEFAULT while #2300 — the HF data "
        "repo's 1,000,000-file Hub cap — is unresolved); hf = the original "
        "upload_folder path (fails loud at the cap)",
    )
    ap.add_argument(
        "--git-payload-root",
        type=Path,
        default=None,
        help="git-persist destination dir (default: <repo>/eval_results/issue_2094/"
        "butler_grid; smokes MUST redirect to scratch so committed paths are never "
        "overwritten)",
    )
    ap.add_argument(
        "--upload-every",
        type=int,
        default=25,
        help="grid: bulk-upload the staged text every N completed blocks",
    )
    ap.add_argument("--planned-wall-h", type=float, default=1.0)
    ap.add_argument("--gpu-hours-budgeted", type=float, default=6.0)
    ap.add_argument(
        "--scores-dir",
        type=Path,
        default=None,
        help="dir holding fp-{bare,persona}.anchors.scores.jsonl (e.g. the "
        "/tmp/i2094_shards mirror); default: fetch the two files from the HF data repo",
    )
    ap.add_argument(
        "--coherence-path",
        type=Path,
        default=REV.DEFAULT_COHERENCE_PATH,
        help="the parent's anchor_draws.jsonl (context_id, draw, coherent)",
    )
    ap.add_argument(
        "--banked-anchors-file",
        type=Path,
        default=None,
        help="local mirror of the parent's anchors.jsonl; default: fetch from HF",
    )
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    return ap.parse_args(argv)


def _import_check() -> None:
    """Resolve every deferred import on this driver's real paths and exit 0.

    Delegates to the parent's check (transformers / hub loads inside
    ``load_model_and_tokenizer`` / ``_repro`` / ``_upload_dir``) and the rev
    driver's (the banked-scores HF fetch), then adds THIS driver's own
    function-body imports (the banked-anchors HF fetch)."""
    RUN._import_check()
    REV._import_check()
    from huggingface_hub import hf_hub_download  # noqa: F401

    from explore_persona_space.orchestrate.hub import retry_transient  # noqa: F401

    print("[butler-import-check] OK", flush=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RUN.RC_OK
    assert args.phase, "--phase is required (or pass --import-check)"
    cfg = RUN.build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu_id is not None:
        logger.info(
            "[env] --gpu-id=%s CUDA_VISIBLE_DEVICES=%s",
            args.gpu_id,
            RUN.os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    if cfg.phase == "anchors":
        return phase_anchors(cfg, args)
    if cfg.phase == "bank":
        # Verbatim parent bank phase: 20-context capture + injection gate + resume.
        return RUN.phase_bank(cfg)
    if cfg.phase == "grid":
        return phase_grid(cfg, args)
    assert cfg.phase == "upload", cfg.phase
    return phase_upload(cfg, args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
