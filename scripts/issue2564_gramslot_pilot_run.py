"""#2564 grammar-slot one-word PILOT — pod driver: bank build + generation + capture + upload.

Finer-grained sibling of ``scripts/issue2564_langow_pilot_run.py``'s
``query_content_oneword`` axis: 24 matched question frames, each authored as
(base, subject-switched, object-switched, verb-switched) with EXACTLY ONE
whitespace-token position changed per variant (equal token counts; punctuation
attached to the word). Axes (``pair_class == axis``):
``query_oneword_subject`` / ``query_oneword_object`` / ``query_oneword_verb``,
24 pairs each, orientation base->variant (delta = variant - base; pair a =
variant context, b = base context, matching the langow install convention).

Grid: 96 contexts (24 frames x [base + 3 variants]) in the single cell
``query_gramslot`` (the base context is shared by all three pair classes, so
the grid cannot partition by class), empty system slot, single user turn.
72 pairs. Generation + capture parameters IDENTICAL to the langow pilot
(K=10 draws, temp 1.0, seed_base 42, max_new 2048 with the >2% cap-hit
whole-cell re-gen at 4096, capture layers 14/19/26, Qwen/Qwen2.5-7B-Instruct,
frozen #2564 render path).

Reuse: this driver IMPORTS the main-resident langow module by path and calls
its per-cell machinery (``_gen_cell``, ``_capture_vc``, ``_capture_cell_va``,
completeness predicates, ``Cfg``/``build_cfg``/argparser); the pinned-blob
``bank2564`` / ``issue2162_run`` imports ride along (langow extracts them at
import). ONE surgical rebind: ``L._regime_fp`` -> this module's gramslot
fingerprint, so every reused checkpoint/resume/done-manifest keys on the
FRAMES table (the langow module copy is private — imported under a unique
``sys.modules`` name; the real module is untouched).

Uploads: HF ``superkaiba1/explore-persona-space-data`` under
``issue2564_minpair/gramslot_pilot/{raw_completions,analysis_tensors,manifests}``
(langow conventions kept: ``resume_skip=False`` re-uploads re-staged files;
``"upload": cfg.upload`` in the regime fingerprint — both were langow review
findings). NOTE: the reused capture writers keep their langow-flavored
FILENAMES (``va_langow_query_gramslot.pt``, ``vc_langow_bank.pt``) under the
gramslot prefix — the gramslot reads script enumerates exactly those.

Pod launch (fresh 1x H100, repo at main + fetched issue-2564 objects):

    uv run python scripts/issue2564_gramslot_pilot_run.py --phase all \
        --out-root /workspace/eps2564_gramslot --upload hf

VM bank gate (tokenizer-only, no model, no GPU, no writes):

    uv run python scripts/issue2564_gramslot_pilot_run.py --bank-check

Smoke blind-spot enumeration (``--tiny``; inherited via the reused langow
machinery):
- production model SUBSTITUTED: from-config 4-layer/64-hidden CPU model over
  the real vocab; the bf16 CUDA load and the production capture layers
  (14, 19, 26) never run under tiny (tiny captures layers (1, 2, 3)).
- ``model_revision`` UNRESOLVED under tiny ("unresolved-tiny") — the HfApi
  main->sha pin branch never runs.
- grid NARROWED: frame f01 only (4 contexts, 3 pairs — one per pair class)
  x 2 draws; ``max_new_tokens`` defaults to 64.
- cap-hit re-gen DEMOTED to an informational log line under tiny.
- upload branch NOT exercised unless ``--upload hf`` is passed explicitly
  (tiny default ``--upload none``).
- ALWAYS-ON in both modes: render gates (exactly one "assistant" per render;
  empty-system prefix), the one-word whitespace-diff gate + distinct-position
  gate per frame, ``changed_token_count >= 1`` per pair, the gate-4 EXACT
  boundary compare, and the ctx re-tokenization drift assert.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (transitively via the langow module) — thread caps + creds

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: E402

logger = logging.getLogger("issue2564_gramslot")

REPO_ROOT = Path(__file__).resolve().parent.parent
assert (REPO_ROOT / "pyproject.toml").is_file(), REPO_ROOT


def _load_by_path(name: str, path: Path):
    """Import a main-resident script under a UNIQUE module name (same shape as
    the langow module's own pinned-blob loader)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Importing the langow driver executes its pinned-blob extraction (bank2564 +
# issue2162_run at the frozen issue-2564 SHA) and exposes L.BK / L.R.
L = _load_by_path(
    "issue2564_langow_pilot_run_for_gramslot",
    REPO_ROOT / "scripts" / "issue2564_langow_pilot_run.py",
)

# ── constants ─────────────────────────────────────────────────────────────

ISSUE = 2564
HF_PREFIX = "issue2564_minpair/gramslot_pilot"

CELL = "query_gramslot"
CELLS = (CELL,)
SLOT_CLASSES = (
    ("subj", "query_oneword_subject"),
    ("obj", "query_oneword_object"),
    ("verb", "query_oneword_verb"),
)
PAIR_CLASSES = tuple(cls for _tag, cls in SLOT_CLASSES)

# 24 matched frames, 2 per topic (pets, housing, money, career, school, health,
# travel, food, tech, family, media, exercise). Tuple order:
# (frame_id, base, subject-switched, object-switched, verb-switched). Each
# variant differs from the BASE in exactly one whitespace-token position (equal
# token counts), the three variants of a frame at three DISTINCT positions —
# both gate-verified at bank build.
FRAMES: tuple[tuple[str, str, str, str, str], ...] = (
    # pets
    (
        "f01",
        "Should families adopt older dogs?",
        "Should retirees adopt older dogs?",
        "Should families adopt older cats?",
        "Should families foster older dogs?",
    ),
    (
        "f02",
        "Do cats need daily playtime?",
        "Do dogs need daily playtime?",
        "Do cats need daily grooming?",
        "Do cats enjoy daily playtime?",
    ),
    # housing
    (
        "f03",
        "Should landlords cover minor repairs?",
        "Should tenants cover minor repairs?",
        "Should landlords cover minor upgrades?",
        "Should landlords handle minor repairs?",
    ),
    (
        "f04",
        "Should cities build more apartments?",
        "Should developers build more apartments?",
        "Should cities build more townhouses?",
        "Should cities approve more apartments?",
    ),
    # money
    (
        "f05",
        "Should couples share bank accounts?",
        "Should roommates share bank accounts?",
        "Should couples share bank statements?",
        "Should couples merge bank accounts?",
    ),
    (
        "f06",
        "Should teenagers manage their own savings?",
        "Should parents manage their own savings?",
        "Should teenagers manage their own allowance?",
        "Should teenagers track their own savings?",
    ),
    # career
    (
        "f07",
        "Should companies allow remote work?",
        "Should schools allow remote work?",
        "Should companies allow remote meetings?",
        "Should companies require remote work?",
    ),
    (
        "f08",
        "Should managers give weekly feedback?",
        "Should mentors give weekly feedback?",
        "Should managers give weekly praise?",
        "Should managers offer weekly feedback?",
    ),
    # school
    (
        "f09",
        "Should schools teach basic cooking?",
        "Should parents teach basic cooking?",
        "Should schools teach basic coding?",
        "Should schools require basic cooking?",
    ),
    (
        "f10",
        "Should students take morning classes?",
        "Should freshmen take morning classes?",
        "Should students take morning exams?",
        "Should students attend morning classes?",
    ),
    # health
    (
        "f11",
        "Should busy adults track daily sleep?",
        "Should busy athletes track daily sleep?",
        "Should busy adults track daily steps?",
        "Should busy adults prioritize daily sleep?",
    ),
    (
        "f12",
        "Do doctors recommend regular fasting?",
        "Do trainers recommend regular fasting?",
        "Do doctors recommend regular stretching?",
        "Do doctors endorse regular fasting?",
    ),
    # travel
    (
        "f13",
        "Should tourists always visit famous landmarks?",
        "Should locals always visit famous landmarks?",
        "Should tourists always visit famous museums?",
        "Should tourists always explore famous landmarks?",
    ),
    (
        "f14",
        "Should travelers book early flights?",
        "Should backpackers book early flights?",
        "Should travelers book early trains?",
        "Should travelers choose early flights?",
    ),
    # food
    (
        "f15",
        "Should restaurants offer smaller portions?",
        "Should cafeterias offer smaller portions?",
        "Should restaurants offer smaller menus?",
        "Should restaurants promote smaller portions?",
    ),
    (
        "f16",
        "Should people cook simple dinners?",
        "Should beginners cook simple dinners?",
        "Should people cook simple lunches?",
        "Should people prep simple dinners?",
    ),
    # tech
    (
        "f17",
        "Should parents limit screen time?",
        "Should teachers limit screen time?",
        "Should parents limit screen usage?",
        "Should parents monitor screen time?",
    ),
    (
        "f18",
        "Should workers disable phone notifications?",
        "Should drivers disable phone notifications?",
        "Should workers disable phone alerts?",
        "Should workers silence phone notifications?",
    ),
    # family
    (
        "f19",
        "Should grandparents share family recipes?",
        "Should relatives share family recipes?",
        "Should grandparents share family stories?",
        "Should grandparents record family recipes?",
    ),
    (
        "f20",
        "Should siblings split household chores?",
        "Should partners split household chores?",
        "Should siblings split household expenses?",
        "Should siblings rotate household chores?",
    ),
    # media
    (
        "f21",
        "Should readers trust online reviews?",
        "Should shoppers trust online reviews?",
        "Should readers trust online ratings?",
        "Should readers consult online reviews?",
    ),
    (
        "f22",
        "Should studios release shorter movies?",
        "Should networks release shorter movies?",
        "Should studios release shorter trailers?",
        "Should studios produce shorter movies?",
    ),
    # exercise
    (
        "f23",
        "Should runners stretch before workouts?",
        "Should swimmers stretch before workouts?",
        "Should runners stretch before races?",
        "Should runners hydrate before workouts?",
    ),
    (
        "f24",
        "Should gyms offer evening classes?",
        "Should libraries offer evening classes?",
        "Should gyms offer evening childcare?",
        "Should gyms host evening classes?",
    ),
)


# ── regime fingerprint (the single reuse seam) ────────────────────────────


def _regime_fp(cfg, extra: dict | None = None) -> str:
    """Gramslot regime fingerprint: langow's base keys with the FRAMES table
    (+ cell / pair classes) in place of the langow value tables. Keyed on the
    GENERATING PARAMETERS only (code-style.md); ``upload`` stays in the key
    (langow review finding 1)."""
    base = {
        "issue": ISSUE,
        "pin": L.PIN,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "tiny": cfg.tiny,
        "draws": cfg.draws,
        "gen_batch": cfg.gen_batch,
        "seed_base": cfg.seed_base,
        "temperature": str(L.ANCHOR_TEMPERATURE),
        "max_new_tokens": cfg.max_new_tokens,
        "cell": CELL,
        "frames_sha": L._sha16([list(f) for f in FRAMES]),
        "pair_classes": list(PAIR_CLASSES),
        "upload": cfg.upload,
    }
    if extra:
        base.update(extra)
    return L._sha16(base)


# Rebind the langow module's fingerprint to the gramslot regime: every reused
# checkpoint/resume/done-manifest helper (_generate_cell, _gen_cell, _cell_fp,
# _capture_vc, _capture_cell_va, the *_complete predicates) resolves
# ``_regime_fp`` as a module global at CALL time, so this single rebind keys
# ALL resume state on the FRAMES table. Safe because L is a private module
# copy (unique sys.modules name).
L._regime_fp = _regime_fp


# ── frames gates + pilot bank ─────────────────────────────────────────────


def one_word_diff(base: str, variant: str) -> tuple[int, str, str]:
    """Fail-loud one-word whitespace-diff gate: equal token counts, EXACTLY one
    differing position. Returns (position, base_word, variant_word)."""
    bt, vt = base.split(), variant.split()
    assert len(bt) == len(vt), ("token-count mismatch", base, variant, len(bt), len(vt))
    diff = [i for i, (x, y) in enumerate(zip(bt, vt)) if x != y]
    assert len(diff) == 1, ("not exactly one changed position", base, variant, diff)
    i = diff[0]
    return i, bt[i], vt[i]


def frames_whitespace_gate() -> dict[tuple[str, str], dict]:
    """Tokenizer-free authoring gates over FRAMES: 24 unique frames, one-word
    diff per variant, three DISTINCT changed positions per frame, all 96
    question strings unique. Returns {(frame_id, slot): diff record}."""
    assert len(FRAMES) == 24, len(FRAMES)
    recs: dict[tuple[str, str], dict] = {}
    seen_ids: set[str] = set()
    seen_q: set[str] = set()
    for fid, base, subj, obj, verb in FRAMES:
        assert fid not in seen_ids, fid
        seen_ids.add(fid)
        positions: dict[str, int] = {}
        for tag, variant in (("subj", subj), ("obj", obj), ("verb", verb)):
            pos, w_base, w_var = one_word_diff(base, variant)
            positions[tag] = pos
            recs[(fid, tag)] = {"diff_pos": pos, "word_base": w_base, "word_variant": w_var}
        assert len(set(positions.values())) == 3, ("variants share a changed position", fid)
        for q in (base, subj, obj, verb):
            assert q not in seen_q, ("duplicate question across the grid", q)
            seen_q.add(q)
    return recs


def build_pilot_bank(tiny: bool, tok) -> dict:
    """96 contexts / 72 pairs (production; tiny keeps frame f01: 4 contexts,
    3 pairs — one per pair class), gated exactly like the langow bank: render
    gates, changed BPE tokens >= 1 per pair, grid-completeness asserts."""
    diff_by = frames_whitespace_gate()
    contexts: dict[str, dict] = {}
    order: list[str] = []

    def _add(ctx: dict) -> None:
        assert ctx["id"] not in contexts, ctx["id"]
        contexts[ctx["id"]] = ctx
        order.append(ctx["id"])

    for fid, base, subj, obj, verb in FRAMES:
        for tag, q in (("base", base), ("subj", subj), ("obj", obj), ("verb", verb)):
            _add(
                {
                    "id": L.BK.context_id(CELL, f"{fid}{tag}", fid),
                    "cell": CELL,
                    "kind": "E",
                    "value_id": f"{fid}{tag}",
                    "carrier": fid,
                    "form": "question",
                    "system": "",
                    "user": q,
                }
            )
    assert len(contexts) == 96, len(contexts)

    pairs: list[dict] = []
    for fid, _base, _subj, _obj, _verb in FRAMES:
        base_cid = L.BK.context_id(CELL, f"{fid}base", fid)
        for tag, cls in SLOT_CLASSES:
            rec = diff_by[(fid, tag)]
            # Orientation base->variant: a = variant, b = base (delta = a - b).
            pairs.append(
                {
                    "pair_id": L.BK.pair_id(cls, CELL, f"{fid}{tag}", f"{fid}base", fid),
                    "pair_class": cls,
                    "axis": cls,
                    "carrier": fid,
                    "value_a": f"{fid}{tag}",
                    "value_b": f"{fid}base",
                    "a": L.BK.context_id(CELL, f"{fid}{tag}", fid),
                    "b": base_cid,
                    **rec,
                }
            )
    n_by_class = {cls: sum(1 for p in pairs if p["pair_class"] == cls) for cls in PAIR_CLASSES}
    assert n_by_class == {cls: 24 for cls in PAIR_CLASSES}, n_by_class
    assert len(pairs) == 72, len(pairs)

    if tiny:
        keep = set(order[:4])  # frame f01: base + 3 variants -> one pair per class survives
        contexts = {cid: c for cid, c in contexts.items() if cid in keep}
        pairs = [p for p in pairs if p["a"] in contexts and p["b"] in contexts]
        assert len(pairs) == 3, len(pairs)

    ids_by_ctx: dict[str, list[int]] = {}
    for cid, ctx in contexts.items():
        rendered = L.BK.render_context(tok, ctx)
        assert rendered.count("assistant") == 1, (cid, rendered.count("assistant"))
        assert rendered.startswith("<|im_start|>system\n<|im_end|>\n"), cid
        ids_by_ctx[cid] = L.BK.context_token_ids(tok, ctx)
    for p in pairs:
        chg = L.BK.changed_token_count(ids_by_ctx[p["a"]], ids_by_ctx[p["b"]])
        assert chg >= 1, (p["pair_id"], "identical rendered prompts")
        p["changed_tokens"] = int(chg)

    per_cell = {CELL: [cid for cid in order if cid in contexts]}
    return {"contexts": contexts, "pairs": pairs, "per_cell": per_cell}


def write_bank_manifest(cfg, bank: dict) -> None:
    write_json_atomic(
        cfg.manifest_dir / "pilot_bank.json",
        {
            "issue": ISSUE,
            "regime_fp": _regime_fp(cfg, {"phase": "bank"}),
            "contexts": list(bank["contexts"].values()),
            "pairs": bank["pairs"],
            "n_contexts": len(bank["contexts"]),
            "n_pairs": len(bank["pairs"]),
            "frames": [list(f) for f in FRAMES],
            "pair_classes": list(PAIR_CLASSES),
            "repro": L._repro(cfg, "bank"),
        },
    )


# ── phases (thin drivers over the reused langow per-cell machinery) ───────


def phase_gen(cfg, bank: dict, model, tok) -> int:
    print("[phase=gen] start", flush=True)
    eot_ids = L.R.eot_tail_ids(tok)
    write_bank_manifest(cfg, bank)
    sentinel = cfg.out_root / "gramslot_gen_done.json"
    pending = [c for c in CELLS if not L._gen_cell_complete(cfg, c)]
    s = L._read_json(sentinel)
    if not pending and s is not None and s.get("regime_fp") == _regime_fp(cfg, {"phase": "gen"}):
        logger.info("[gen] all cells complete + sentinel present — skipping")
        return 0
    for cell in CELLS:
        if L._gen_cell_complete(cfg, cell):
            logger.info("[gen:%s] done manifest present — skipping", cell)
            continue
        ctxs = [bank["contexts"][cid] for cid in bank["per_cell"][cell]]
        L._gen_cell(cfg, model, tok, eot_ids, cell, ctxs)
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        # Rollout TEXT persists to HF BEFORE any capture reduce (#779).
        res = upload_dir_sharded(
            cfg.anchors_dir,
            L.HF_DATA_REPO,
            f"{HF_PREFIX}/raw_completions/anchors",
            shard_glob="*.jsonl",
            resume_skip=False,
            delete_local=False,
        )
        upload["anchors"] = L._upload_summary(res)
    write_json_atomic(
        sentinel,
        {
            "regime_fp": _regime_fp(cfg, {"phase": "gen"}),
            "cells": {c: L._read_json(cfg.manifest_dir / f"anchors_{c}.done.json") for c in CELLS},
            "upload": upload,
            "repro": L._repro(cfg, "gen"),
        },
    )
    print("[phase=gen] sentinel written", flush=True)
    return 0


def phase_capture(cfg, bank: dict, model, tok) -> int:
    print("[phase=capture] start", flush=True)
    eot_ids = L.R.eot_tail_ids(tok)
    sentinel = cfg.out_root / "gramslot_capture_done.json"
    contexts = [bank["contexts"][cid] for cell in CELLS for cid in bank["per_cell"][cell]]
    pending_va = [c for c in CELLS if not L._va_cell_complete(cfg, c)]
    s = L._read_json(sentinel)
    if (
        not pending_va
        and L._vc_complete(cfg)
        and s is not None
        and s.get("regime_fp") == _regime_fp(cfg, {"phase": "capture"})
    ):
        logger.info("[capture] all cells + vc complete + sentinel — skipping")
        return 0
    if not L._vc_complete(cfg):
        L._capture_vc(cfg, model, tok, contexts)
    ctx_by_id = bank["contexts"]
    for cell in CELLS:
        if L._va_cell_complete(cfg, cell):
            logger.info("[capture:%s] done manifest present — skipping", cell)
            continue
        L._capture_cell_va(cfg, model, tok, eot_ids, cell, ctx_by_id)
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        for name, local_dir, glob in (
            ("va", cfg.va_dir, "*.pt"),
            ("vc", cfg.vc_dir, "*.pt"),
            ("manifests", cfg.manifest_dir, "*.json"),
        ):
            res = upload_dir_sharded(
                local_dir,
                L.HF_DATA_REPO,
                f"{HF_PREFIX}/analysis_tensors/{name}"
                if name != "manifests"
                else f"{HF_PREFIX}/manifests",
                shard_glob=glob,
                resume_skip=False,
                delete_local=False,
            )
            upload[name] = L._upload_summary(res)
    write_json_atomic(
        sentinel,
        {
            "regime_fp": _regime_fp(cfg, {"phase": "capture"}),
            "n_contexts_vc": len(contexts),
            "cells": {
                c: L._read_json(cfg.manifest_dir / f"va_langow_{c}.done.json") for c in CELLS
            },
            "upload": upload,
            "repro": L._repro(cfg, "capture"),
        },
    )
    print("[phase=capture] sentinel written", flush=True)
    return 0


def phase_finalize(cfg) -> int:
    """Terminal sentinel — written LAST, after all uploads (upload-policy)."""
    print("[phase=finalize] start", flush=True)
    gen_s = L._read_json(cfg.out_root / "gramslot_gen_done.json")
    cap_s = L._read_json(cfg.out_root / "gramslot_capture_done.json")
    assert gen_s is not None, "gen sentinel missing — run --phase gen first"
    assert cap_s is not None, "capture sentinel missing — run --phase capture first"
    per_cell = {}
    for cell in CELLS:
        g = L._read_json(cfg.manifest_dir / f"anchors_{cell}.done.json") or {}
        v = L._read_json(cfg.manifest_dir / f"va_langow_{cell}.done.json") or {}
        per_cell[cell] = {
            "n_contexts": g.get("n_contexts"),
            "n_rows_gen": g.get("n_rows"),
            "cap_hit_frac": g.get("cap_hit_frac"),
            "cap_hit_frac_regen": g.get("cap_hit_frac_regen"),
            "max_new_tokens_final": g.get("max_new_tokens_final"),
            "n_rows_captured": v.get("n_rows"),
            "n_empty_rows": v.get("n_empty_rows"),
        }
    write_json_atomic(
        cfg.out_root / "gramslot_done.json",
        {
            "issue": ISSUE,
            "status": "done",
            "regime_fp": _regime_fp(cfg, {"phase": "finalize"}),
            "cells": per_cell,
            "upload_gen": gen_s.get("upload"),
            "upload_capture": cap_s.get("upload"),
            "hf_prefix": HF_PREFIX,
            "repro": L._repro(cfg, "finalize"),
        },
    )
    print("[phase=done] gramslot_done.json written", flush=True)
    return 0


# ── checks + main ─────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    """Langow's parser (same phases / knobs / --tiny / --import-check) with the
    gramslot out-root default and the VM-side --bank-check gate."""
    ap = L.build_argparser()
    ap.description = "#2564 grammar-slot one-word PILOT (see module docstring)"
    ap.set_defaults(out_root="/workspace/eps2564_gramslot")
    ap.add_argument(
        "--bank-check",
        action="store_true",
        help="VM gate: build the production bank with the real tokenizer, print "
        "per-pair changed-BPE-token counts, exit 0 (no model, no writes)",
    )
    return ap


def _import_check() -> None:
    """Argparse-attribute completeness (this file + the reused langow file) +
    signature/rebind checks on the reused surface + the tokenizer-free gates."""
    import inspect

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__, L.__file__)
    assert L._regime_fp is _regime_fp, "gramslot regime-fp rebind did not take"
    for fn, needed in (
        (
            L.R.capture_answer_states,
            {"payloads", "positions", "tail_inclusive", "return_boundaries"},
        ),
        (
            L.generate_batch,
            {"n", "hook", "max_new_tokens", "temperature", "seed_base", "render_fn", "ids_fn"},
        ),
    ):
        params = set(inspect.signature(fn).parameters)
        missing = needed - params
        assert not missing, (fn.__name__, sorted(missing))
    for name in (
        "_gen_cell",
        "_gen_cell_complete",
        "_capture_vc",
        "_capture_cell_va",
        "_vc_complete",
        "_va_cell_complete",
        "_read_json",
        "_upload_summary",
        "_repro",
        "_sha16",
        "build_cfg",
        "build_argparser",
    ):
        assert callable(getattr(L, name)), name
    for name in (
        "render_context",
        "context_token_ids",
        "changed_token_count",
        "context_id",
        "pair_id",
    ):
        assert callable(getattr(L.BK, name)), name
    for name in ("load_model_and_tokenizer", "eot_tail_ids", "cap_hit", "_right_pad"):
        assert callable(getattr(L.R, name)), name
    frames_whitespace_gate()
    print("[import-check] ok: langow reuse surface + gramslot frames gates resolve", flush=True)


def _bank_check() -> int:
    """VM gate: production bank with the real tokenizer (no model). Prints all
    72 per-pair changed-BPE-token counts + per-class summaries."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(L.MODEL_ID)
    bank = build_pilot_bank(False, tok)
    assert len(bank["contexts"]) == 96 and len(bank["pairs"]) == 72, (
        len(bank["contexts"]),
        len(bank["pairs"]),
    )
    for p in bank["pairs"]:
        print(
            f"[bank-check] {p['pair_class']:24s} {p['carrier']} "
            f"{p['word_base']!r}->{p['word_variant']!r} pos={p['diff_pos']} "
            f"changed_bpe={p['changed_tokens']}",
            flush=True,
        )
    by_cls: dict[str, list[int]] = {}
    for p in bank["pairs"]:
        by_cls.setdefault(p["pair_class"], []).append(p["changed_tokens"])
    for cls, v in sorted(by_cls.items()):
        print(f"[bank-check] {cls}: n={len(v)} changed_bpe min={min(v)} max={max(v)}", flush=True)
    print("[bank-check] ok: 96 contexts / 72 pairs, all gates passed", flush=True)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        return 0
    if args.bank_check:
        return _bank_check()
    cfg = L.build_cfg(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    phases = list(L.PHASES) if args.phase == "all" else [args.phase]
    model = tok = None
    bank = None
    if any(p in ("gen", "capture") for p in phases):
        model, tok = L.R.load_model_and_tokenizer(cfg)
        bank = build_pilot_bank(cfg.tiny, tok)
        print(
            f"[bank] {len(bank['contexts'])} contexts / {len(bank['pairs'])} pairs "
            f"(tiny={cfg.tiny})",
            flush=True,
        )
    rc = 0
    for phase in phases:
        if phase == "gen":
            rc = phase_gen(cfg, bank, model, tok)
        elif phase == "capture":
            rc = phase_capture(cfg, bank, model, tok)
        elif phase == "finalize":
            rc = phase_finalize(cfg)
        if rc != 0:
            return rc
    return rc


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
