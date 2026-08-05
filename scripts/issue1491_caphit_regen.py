#!/usr/bin/env python3
"""Task #1491 greedy round — Path B cap-hit re-generation pass (regen @ >=2x cap).

The greedy base pass generates at ``GEN_MAX_TOKENS=1024`` and measured a
cap-hit fraction of 18.47% on the completed 0.5B rung (greedy falls into
repetition loops and runs to the cap ~2.8x more often than temperature 1.0),
so ~1 in 5 answer representations is a truncated answer — systematically
(greedy is deterministic, it does not average out). This script implements the
pre-registered CLAUDE.md re-gen trigger mechanically (Path B):

1. **Scan** — identify cap-hit rows (``finish_reason == 'length'``) per
   (rung, split) from the BASE pass ``raw_completions`` chunks, identity-
   asserting every payload (``sampling_mode == greedy_temp0`` AND
   ``gen_max_tokens == 1024`` — the cap is part of the decoding identity,
   same as the mode; see ``issue1491_ladder_generate_capture._sampling_cap``).
2. **Regen** — regenerate ONLY those rows under the SAME greedy config at
   ``--regen-max-tokens`` (default 2048 >= 2x the base cap), with the vLLM
   engine's ``max_model_len`` RAISED to ``--regen-max-model-len`` (default
   10240) so every base-admitted prompt (budget 7104) + 2048 new tokens fits
   — an over-length ``add_request`` is ENGINE-FATAL (gotchas.md), so the
   admission filter is re-applied at the regen budget and any violation
   fails LOUD (arithmetically impossible unless the tokenizer drifted).
3. **Re-capture** — teacher-forced activations for the regenerated rows via
   the SAME parity-gated capture path as the base driver. Every regen ``.pt``
   bundle is CONTENT-BOUND to its raw text: ``rows_sha256`` hashes the
   (ci, response) pairs in bundle row order, and every downstream consumer
   recomputes it from the paired raw JSON — a merged row whose text is
   2048-regenerated but whose activation came from a different text is
   therefore impossible to load (fail-loud), not merely unlikely.
4. **Merge** — deterministic + idempotent OVERLAY, never a rewrite: base
   chunks stay byte-untouched (upload-policy: regenerating a published
   artifact in place is banned); regen artifacts live in the version-bumped
   sibling namespace ``<base>/<split>/regen_cap<CAP>/...`` and
   ``stream_split_merged`` replaces rows by ci at read time, returning a
   per-row ``gen_cap`` provenance array (1024 base / 2048 regen). The
   per-split ``regen_manifest.json`` records per-row provenance durably.

HF layout (base pass unchanged; regen namespace additive):

    issue1491_scale_ladder_greedy/<slug>/<split>/raw_completions/shardSS_chunkCCCC.json
    issue1491_scale_ladder_greedy/<slug>/<split>/final_token_capture/shardSS_chunkCCCC.pt
    issue1491_scale_ladder_greedy/<slug>/<split>/regen_cap2048/raw_completions/regen_chunkKKKK.json
    issue1491_scale_ladder_greedy/<slug>/<split>/regen_cap2048/final_token_capture/regen_chunkKKKK.pt
    issue1491_scale_ladder_greedy/<slug>/<split>/regen_cap2048/sampling_mode.json
    issue1491_scale_ladder_greedy/<slug>/<split>/regen_cap2048/regen_manifest.json

Phases (in-process phase split — gen for every split first, then the engine
is REAPED (gotchas.md vLLM teardown recipe) before the HF capture model
loads, so the pass works at every rung scale on one GPU):

    regen_scan -> regen_gen -> regen_capture -> regen_manifest -> done

Usage (pod-side, one invocation per rung; runs AFTER the rung's base pass):

    uv run python scripts/issue1491_caphit_regen.py --slug scale05 --device cuda -v

CPU smoke (tiny synthetic base corpus; the vLLM engine is the one faked
boundary — GC._generate_seeded's llm=None stub path, same as the base
driver's CPU smoke; SamplingParams construction + admission + identity +
capture + merge all run REAL):

    uv run python scripts/issue1491_caphit_regen.py --slug scale05 --device cpu \\
        --no-upload --base-local-dir /tmp/smoke_base --out-dir /tmp/smoke_out \\
        --splits test_1000 ceiling_draws/seed43
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as F  # noqa: E402
import issue779_ffc_n10k_generate_capture as N10  # noqa: E402
import issue779_ffc_n50k_fits as N50F  # noqa: E402
import issue779_fitter_fair_comparison as FFC  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402
import issue1491_ladder_generate_capture as GC  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from issue779_ffc_n1m_generate_capture import (  # noqa: E402
    _filter_overlength_prompts,
    _flush_upload_batch,
    _rendered_prompt_token_len,
    _stack_chunk,
)

logger = logging.getLogger("issue1491_caphit_regen")

# The PARENT (temperature-1.0) root is REFUSED outright: the regen pass is
# defined for the greedy round, and the parent's committed/promoted artifacts
# must never gain a sibling regen namespace by accident (plan §10 item (i)
# runtime-reuse clobber clause; the payload identity asserts would also fail
# loud on the parent's mode, this refusal just fails earlier + clearer).
PARENT_HF_ROOT = "issue1491_scale_ladder"
DEFAULT_HF_ROOT = "issue1491_scale_ladder_greedy"

DEFAULT_REGEN_MAX_TOKENS = 2048  # >= 2x GEN_MAX_TOKENS (CLAUDE.md re-gen trigger)
# Engine max_model_len for the regen pass: every base-admitted prompt
# (<= OVERLENGTH_BUDGET = 7104 rendered tokens) + regen cap + margin must fit,
# else the over-length add_request kills the WHOLE engine (gotchas.md #1738).
DEFAULT_REGEN_MAX_MODEL_LEN = 10240
REGEN_CHUNK_ROWS = 500  # parent-parity chunk grain

BASE_CHUNK_RE = re.compile(r"^shard(\d{2})_chunk(\d{4})\.json$")
REGEN_RAW_NAME = "regen_chunk{k:04d}.json"
REGEN_PT_NAME = "regen_chunk{k:04d}.pt"
REGEN_MANIFEST_NAME = "regen_manifest.json"

# HF-prefix-space splits (the ceiling draws live under a nested prefix whose
# payload "split" field carries the driver's CLI split name).
SPLIT_PREFIX_SEEDS = {
    "train_25k": 42,
    "val_400": 42,
    "test_1000": 42,
    "wc_test_1k": 42,
    "tierB_3600": 42,
    "ceiling_draws/seed43": 43,
    "ceiling_draws/seed44": 44,
}
SPLIT_PREFIX_TO_PAYLOAD_SPLIT = {
    "train_25k": "train_25k",
    "val_400": "val_400",
    "test_1000": "test_1000",
    "wc_test_1k": "wc_test_1k",
    "tierB_3600": "tierB_3600",
    "ceiling_draws/seed43": "ceiling_draw_43",
    "ceiling_draws/seed44": "ceiling_draw_44",
}
ALL_SPLITS = list(SPLIT_PREFIX_SEEDS)


def regen_subdir(cap: int) -> str:
    """Version-bumped sibling namespace name — the cap is IN the path so two
    caps can never share a namespace even before any marker/payload check."""
    return f"regen_cap{int(cap)}"


def base_split_prefix(hf_root: str, slug: str, split: str) -> str:
    return f"{hf_root}/{slug}/{split}"


def regen_split_prefix(hf_root: str, slug: str, split: str, cap: int) -> str:
    return f"{base_split_prefix(hf_root, slug, split)}/{regen_subdir(cap)}"


def _split_scratch(out_dir: Path, slug: str, split: str) -> Path:
    d = out_dir / slug / split.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def rows_binding_sha(cis: list[int], responses: list[str]) -> str:
    """Content hash binding a regen .pt bundle to its raw text, in bundle row
    order. Consumers recompute this from the paired raw JSON — the guard that
    makes a text/activation mismatch fail-loud instead of silent."""
    h = hashlib.sha256()
    for ci, resp in zip(cis, responses, strict=True):
        h.update(str(int(ci)).encode("ascii"))
        h.update(b"\x00")
        h.update(hashlib.sha256(resp.encode("utf-8")).digest())
    return h.hexdigest()


def regen_chunk_membership(caphit_cis: list[int]) -> list[list[int]]:
    """Deterministic chunking of the cap-hit set: sorted unique cis in fixed
    REGEN_CHUNK_ROWS chunks. Same base artifacts => same membership across
    resumes (the resume-skip predicate depends on this)."""
    cis = sorted({int(c) for c in caphit_cis})
    return [cis[i : i + REGEN_CHUNK_ROWS] for i in range(0, len(cis), REGEN_CHUNK_ROWS)]


# ---------------------------------------------------------------------------
# Phase 1: scan the base pass for cap-hit rows
# ---------------------------------------------------------------------------


def _base_raw_names_and_getter(hf_root, slug, split, scratch, base_local_dir):
    """(sorted chunk names, fetch(name)->(path, unlink_after)) for the base raw
    chunks of one (slug, split) — HF by default, a local dir on the smoke path."""
    if base_local_dir is not None:
        d = Path(base_local_dir) / split / "raw_completions"
        names = sorted(p.name for p in d.glob("*.json"))

        def _get(name: str) -> tuple[Path, bool]:
            return d / name, False

        return names, _get
    prefix = f"{base_split_prefix(hf_root, slug, split)}/raw_completions"
    names = sorted(GC._remote_index(base_split_prefix(hf_root, slug, split), "raw_completions"))

    def _get(name: str) -> tuple[Path, bool]:
        got = F._download_chunk_with_retry(C.HF_DATA_REPO, f"{prefix}/{name}", scratch)
        return Path(got), True

    return names, _get


def scan_split_caphit(
    hf_root: str,
    slug: str,
    split: str,
    scratch: Path,
    base_local_dir: Path | None,
) -> dict:
    """Identity-assert every base raw chunk of one (slug, split) and collect
    its cap-hit rows.

    Returns ``{"rows": {ci: {prompt, response, base_chunk}}, "n_rows": int,
    "n_chunks": int}`` — ``rows`` holds ONLY ``finish_reason == 'length'``
    rows (base response text kept in memory for the greedy prefix-extension
    stat; it is never printed — content-hygiene digest-only discipline)."""
    payload_split = SPLIT_PREFIX_TO_PAYLOAD_SPLIT[split]
    seed = SPLIT_PREFIX_SEEDS[split]
    names, get = _base_raw_names_and_getter(hf_root, slug, split, scratch, base_local_dir)
    if not names:
        raise FileNotFoundError(
            f"no base raw_completions for {slug}/{split} "
            f"(root={hf_root}, base_local_dir={base_local_dir}) — run the base pass first"
        )
    caphit: dict[int, dict] = {}
    n_rows = 0
    for name in names:
        m = BASE_CHUNK_RE.match(name)
        if not m:
            raise RuntimeError(f"unexpected base raw chunk name {name!r} for {slug}/{split}")
        local, unlink_after = get(name)
        with open(local, encoding="utf-8") as fh:
            payload = json.load(fh)
        if unlink_after:
            local.unlink()
        GC._assert_raw_payload_matches(
            payload,
            name,
            expect_split=payload_split,
            expect_seed=seed,
            expect_shard_index=int(m.group(1)),
            expect_chunk=int(m.group(2)),
            expect_sampling_mode=GC.SAMPLING_MODE_GREEDY,
            expect_gen_max_tokens=GC.GEN_MAX_TOKENS,
        )
        for r in payload["rows"]:
            n_rows += 1
            if str(r["finish_reason"]) == "length":
                ci = int(r["ci"])
                assert ci not in caphit, (
                    "duplicate cap-hit ci across base chunks",
                    slug,
                    split,
                    ci,
                )
                caphit[ci] = {
                    "prompt": str(r["prompt"]),
                    "response": str(r["response"]),
                    "base_chunk": name,
                }
    logger.info(
        "[regen-scan] %s %s: %d/%d cap-hit rows across %d base chunks (rate %.4f)",
        slug,
        split,
        len(caphit),
        n_rows,
        len(names),
        len(caphit) / max(n_rows, 1),
    )
    return {"rows": caphit, "n_rows": n_rows, "n_chunks": len(names)}


# ---------------------------------------------------------------------------
# Phase 2: regenerate cap-hit rows at the regen cap
# ---------------------------------------------------------------------------


def phase_gen_split(
    llm,
    tok,
    hf_root: str,
    slug: str,
    split: str,
    cap: int,
    scan: dict,
    scratch: Path,
    sampling: dict,
    *,
    no_upload: bool,
    engine_seed: int,
    regen_budget: int,
) -> dict:
    """Generate (or salvage) every regen chunk of one (slug, split); persist
    raw text FIRST (persist-by-default), upload in batches."""
    prefix = regen_split_prefix(hf_root, slug, split, cap)
    payload_split = SPLIT_PREFIX_TO_PAYLOAD_SPLIT[split]
    seed = SPLIT_PREFIX_SEEDS[split]
    cache_dir = scratch / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    done_raw = set() if no_upload else GC._remote_index(prefix, "raw_completions")
    done_pt = set() if no_upload else GC._remote_index(prefix, "final_token_capture")
    # Regen-namespace decoding-identity guard (mode AND cap) — BEFORE any
    # chunk decision, exactly like the base driver.
    GC._enforce_sampling_identity(
        prefix,
        scratch,
        cache_dir,
        sampling,
        done_pt=done_pt,
        done_raw=done_raw,
        no_upload=no_upload,
        shard_index=0,
    )
    chunks = regen_chunk_membership(list(scan["rows"]))
    pending_raw: list[str] = []
    stats = {"n_chunks": len(chunks), "n_generated": 0, "n_salvaged": 0, "n_skipped_done": 0}
    try:
        for k, cis in enumerate(chunks):
            raw_name = REGEN_RAW_NAME.format(k=k)
            if raw_name in done_raw:
                stats["n_skipped_done"] += len(cis)
                logger.info("[regen-gen] %s %s chunk %d already on Hub; skip", slug, split, k)
                continue
            salvaged = GC._load_local_raw_salvage(
                scratch,
                raw_name,
                expect_split=payload_split,
                expect_seed=seed,
                expect_shard_index=0,
                expect_chunk=k,
                expect_sampling_mode=sampling["mode"],
                expect_gen_max_tokens=cap,
            )
            if salvaged is not None:
                sal_cis = [int(r["ci"]) for r in salvaged["rows"]]
                if sal_cis != cis:
                    raise RuntimeError(
                        f"regen salvage: local {raw_name} ci membership diverges from the "
                        f"deterministic chunking (local: {sal_cis[:8]}..., expected: "
                        f"{cis[:8]}...) — base-artifact drift; refusing to reuse OR "
                        "regenerate (delete the stale scratch file only if understood)."
                    )
                stats["n_salvaged"] += len(cis)
                pending_raw.append(raw_name)
                logger.warning(
                    "[regen-gen] %s %s chunk %d: SALVAGED %d rows verbatim from local "
                    "scratch (prior run died before upload flush)",
                    slug,
                    split,
                    k,
                    len(cis),
                )
                continue
            prompts = [scan["rows"][c]["prompt"] for c in cis]
            kept_prompts, kept_cis, skipped = _filter_overlength_prompts(
                prompts,
                cis,
                lambda p: _rendered_prompt_token_len(tok, p),
                regen_budget,
            )
            if skipped:
                # Arithmetically impossible (regen budget >= base admission
                # budget, asserted at startup) — a hit means tokenizer/render
                # drift, and proceeding would engine-crash at add_request.
                raise RuntimeError(
                    f"regen admission: {len(skipped)} base-admitted rows exceed the REGEN "
                    f"prompt budget {regen_budget} ({slug}/{split} chunk {k}, first "
                    f"{skipped[:5]}) — tokenizer/render drift; refusing to launch."
                )
            ts = time.time()
            responses, finish = GC._generate_seeded(llm, tok, kept_prompts, seed, sampling)
            n_cap_hit = sum(1 for f_ in finish if f_ == "length")
            C.write_json_atomic(
                scratch / raw_name,
                {
                    "shard_index": 0,
                    "chunk": k,
                    "split": payload_split,
                    "seed": seed,
                    "sampling_seed": seed,
                    "engine_seed": int(engine_seed),
                    "sampling_mode": sampling["mode"],
                    "temperature": sampling["temperature"],
                    "top_p": sampling["top_p"],
                    "gen_max_tokens": int(cap),
                    "regen_of_cap": int(GC.GEN_MAX_TOKENS),
                    "regen_pass": True,
                    "n_cap_hit": n_cap_hit,
                    "rows": [
                        {
                            "ci": int(c),
                            "prompt": p,
                            "response": r,
                            "finish_reason": f_,
                            "base_chunk": scan["rows"][int(c)]["base_chunk"],
                        }
                        for c, p, r, f_ in zip(
                            kept_cis, kept_prompts, responses, finish, strict=True
                        )
                    ],
                },
            )
            stats["n_generated"] += len(kept_cis)
            pending_raw.append(raw_name)
            logger.info(
                "[regen-gen] %s %s chunk %d/%d: %d rows regenerated at cap=%d "
                "(%d residual cap-hit, %.0fs)",
                slug,
                split,
                k + 1,
                len(chunks),
                len(kept_cis),
                cap,
                n_cap_hit,
                time.time() - ts,
            )
            if not no_upload and len(pending_raw) >= GC.UPLOAD_BATCH:
                _flush_upload_batch(scratch, prefix, [], pending_raw)
                pending_raw.clear()
        if not no_upload and pending_raw:
            _flush_upload_batch(scratch, prefix, [], pending_raw)
            pending_raw.clear()
    except BaseException:
        if not no_upload and pending_raw:
            try:
                _flush_upload_batch(scratch, prefix, [], pending_raw)
            except Exception:  # noqa: BLE001
                logger.exception("[regen-gen] best-effort pending flush failed on exit")
        raise
    return stats


# ---------------------------------------------------------------------------
# Phase 3: re-capture activations for the regenerated rows
# ---------------------------------------------------------------------------


def _verify_pt_binding(bundle: dict, raw_map: dict[int, dict], what: str, cap: int) -> None:
    """Fail-loud identity + text<->activation binding check for a regen bundle.

    The binding is CONTENT-level: ``rows_sha256`` (written at capture time
    over the bundle's own (ci, response) pairs) must equal the hash recomputed
    from the PAIRED regen raw rows — so a .pt whose activations were captured
    from any OTHER text (e.g. the 1024-truncated base text) cannot be
    consumed, it raises here."""
    b_mode = str(bundle.get("sampling_mode", ""))
    if b_mode != GC.SAMPLING_MODE_GREEDY:
        raise RuntimeError(f"{what}: regen bundle sampling_mode {b_mode!r} != greedy")
    b_cap = int(bundle.get("gen_max_tokens", -1))
    if b_cap != int(cap):
        raise RuntimeError(
            f"{what}: regen bundle gen_max_tokens {b_cap} != expected {int(cap)} — "
            "cap identity mismatch; refusing to consume"
        )
    got_sha = bundle.get("rows_sha256")
    if not got_sha:
        raise RuntimeError(
            f"{what}: regen bundle lacks rows_sha256 — cannot verify the text<->activation "
            "binding; refusing to consume an unbound regen capture"
        )
    cis = [int(c) for c in bundle["ci"]]
    missing = [c for c in cis if c not in raw_map]
    if missing:
        raise RuntimeError(
            f"{what}: {len(missing)} bundle cis absent from the paired regen raw rows "
            f"(first: {missing[:8]}) — raw/.pt pairing broken"
        )
    want_sha = rows_binding_sha(cis, [str(raw_map[c]["response"]) for c in cis])
    if str(got_sha) != want_sha:
        raise RuntimeError(
            f"{what}: rows_sha256 mismatch ({got_sha} != {want_sha}) — the .pt activations "
            "do NOT correspond to the paired regen raw text (text/activation divergence); "
            "refusing to consume"
        )


def phase_capture_split(
    hf_model,
    tok,
    hf_root: str,
    slug: str,
    split: str,
    cap: int,
    scan: dict,
    scratch: Path,
    sampling: dict,
    *,
    no_upload: bool,
    layers: list[int],
    h_dim: int,
    capture_choice: str,
    batch_size: int,
) -> dict:
    """Teacher-forced capture of every regen chunk of one (slug, split).

    Consumes the regen raw chunks (hub-required unless ``no_upload`` — the
    same round-3a M3 divergence guard as the base driver), captures via the
    parity-gate-selected path, and writes .pt bundles CONTENT-BOUND to their
    raw text (``rows_sha256``). Chunks already on the Hub are downloaded once
    and binding-VERIFIED (idempotent self-check), never recaptured."""
    prefix = regen_split_prefix(hf_root, slug, split, cap)
    payload_split = SPLIT_PREFIX_TO_PAYLOAD_SPLIT[split]
    seed = SPLIT_PREFIX_SEEDS[split]
    cache_dir = scratch / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    done_raw = set() if no_upload else GC._remote_index(prefix, "raw_completions")
    done_pt = set() if no_upload else GC._remote_index(prefix, "final_token_capture")
    chunks = regen_chunk_membership(list(scan["rows"]))
    pending_pt: list[str] = []
    stats: dict = {
        "n_regen_rows": 0,
        "n_captured": 0,
        "n_verified_on_hub": 0,
        "dropped_empty_cis": [],
        "n_residual_caphit": 0,
        "n_prefix_extends": 0,
        "row_records": [],
    }
    try:
        for k, cis in enumerate(chunks):
            raw_name = REGEN_RAW_NAME.format(k=k)
            pt_name = REGEN_PT_NAME.format(k=k)
            raw_map = GC._load_persisted_gen_chunk(
                scratch,
                prefix,
                raw_name,
                cache_dir,
                done_raw,
                expect_split=payload_split,
                expect_seed=seed,
                expect_shard_index=0,
                expect_chunk=k,
                expect_sampling_mode=sampling["mode"],
                expect_gen_max_tokens=cap,
                allow_local_only=no_upload,
            )
            if sorted(raw_map) != cis:
                raise RuntimeError(
                    f"regen capture: {raw_name} ci membership diverges from the deterministic "
                    f"chunking ({slug}/{split} chunk {k}) — base-artifact drift; refusing."
                )
            stats["n_regen_rows"] += len(cis)
            residual = {c for c in cis if str(raw_map[c]["finish_reason"]) == "length"}
            stats["n_residual_caphit"] += len(residual)
            stats["n_prefix_extends"] += sum(
                1
                for c in cis
                if str(raw_map[c]["response"]).startswith(scan["rows"][c]["response"])
            )
            for c in cis:
                assert raw_map[c]["prompt"] == scan["rows"][c]["prompt"], (
                    "prompt drift between base scan and regen raw row",
                    slug,
                    split,
                    c,
                )
            local_pt = scratch / pt_name
            dropped: list[int] = []
            if pt_name in done_pt:
                # Idempotent self-check: verify the published bundle's binding
                # (a corrupt/unbound earlier upload must fail HERE, not at the
                # analysis join), and read its drop list for the manifest.
                got = Path(
                    F._download_chunk_with_retry(
                        C.HF_DATA_REPO, f"{prefix}/final_token_capture/{pt_name}", cache_dir
                    )
                )
                b = FFC._mmap_load(got)
                _verify_pt_binding(b, raw_map, f"{slug}/{split}/{pt_name} (hub)", cap)
                dropped = [int(c) for c in b.get("dropped_empty_cis", [])]
                del b
                got.unlink()
                stats["n_verified_on_hub"] += len(cis) - len(dropped)
                logger.info(
                    "[regen-capture] %s %s chunk %d already on Hub; binding VERIFIED, skip",
                    slug,
                    split,
                    k,
                )
            elif local_pt.exists():
                # Crashed-before-flush salvage: verify + re-upload verbatim.
                b = FFC._mmap_load(local_pt)
                _verify_pt_binding(b, raw_map, f"{slug}/{split}/{pt_name} (local salvage)", cap)
                dropped = [int(c) for c in b.get("dropped_empty_cis", [])]
                del b
                pending_pt.append(pt_name)
                stats["n_captured"] += len(cis) - len(dropped)
                logger.warning(
                    "[regen-capture] %s %s chunk %d: SALVAGED local .pt verbatim "
                    "(binding verified; prior run died before upload flush)",
                    slug,
                    split,
                    k,
                )
            else:
                responses = [str(raw_map[c]["response"]) for c in cis]
                prompts = [str(raw_map[c]["prompt"]) for c in cis]
                ts = time.time()
                if capture_choice == "batched":
                    rows, drop = GC._capture_batched(
                        hf_model, tok, prompts, responses, cis, layers, h_dim, batch_size
                    )
                else:
                    rows, drop = GC._capture_perrow(
                        hf_model, tok, prompts, responses, cis, layers, h_dim
                    )
                dropped = [int(c) for c in drop]
                if rows:
                    bundle = _stack_chunk(rows, layers, 0, k)
                    bundle["sampling_mode"] = sampling["mode"]
                    bundle["gen_max_tokens"] = int(cap)
                    bundle["regen_of_cap"] = int(GC.GEN_MAX_TOKENS)
                    bundle["dropped_empty_cis"] = dropped
                    bundle["rows_sha256"] = rows_binding_sha(
                        [int(r["ci"]) for r in rows], [str(r["response"]) for r in rows]
                    )
                    torch.save(bundle, local_pt)
                    pending_pt.append(pt_name)
                    stats["n_captured"] += len(rows)
                else:
                    logger.warning(
                        "[regen-capture] %s %s chunk %d: 0 captured rows (all empty)",
                        slug,
                        split,
                        k,
                    )
                logger.info(
                    "[regen-capture] %s %s chunk %d/%d: %d/%d captured "
                    "(%d empty dropped, %.0fs) [%s]",
                    slug,
                    split,
                    k + 1,
                    len(chunks),
                    len(rows),
                    len(cis),
                    len(dropped),
                    time.time() - ts,
                    capture_choice,
                )
            stats["dropped_empty_cis"].extend(dropped)
            for c in cis:
                stats["row_records"].append(
                    {
                        "ci": int(c),
                        "base_chunk": scan["rows"][c]["base_chunk"],
                        "regen_chunk": raw_name,
                        "gen_cap": int(cap),
                        "finish_reason_regen": str(raw_map[c]["finish_reason"]),
                        "captured": int(c) not in set(dropped),
                    }
                )
            if not no_upload and len(pending_pt) >= GC.UPLOAD_BATCH:
                _flush_upload_batch(scratch, prefix, pending_pt, [])
                pending_pt.clear()
        if not no_upload and pending_pt:
            _flush_upload_batch(scratch, prefix, pending_pt, [])
            pending_pt.clear()
    except BaseException:
        if not no_upload and pending_pt:
            try:
                _flush_upload_batch(scratch, prefix, pending_pt, [])
            except Exception:  # noqa: BLE001
                logger.exception("[regen-capture] best-effort pending flush failed on exit")
        raise
    return stats


# ---------------------------------------------------------------------------
# Phase 4: per-split manifest (durable per-row cap provenance)
# ---------------------------------------------------------------------------


def phase_manifest_split(
    hf_root: str,
    slug: str,
    split: str,
    cap: int,
    scan: dict,
    scratch: Path,
    *,
    no_upload: bool,
    gen_stats: dict,
    cap_stats: dict,
    regen_max_model_len: int,
) -> dict:
    """Compose + persist the per-(slug, split) regen manifest: run-level
    counts, per-row provenance (which cap produced each merged row), and
    reproducibility metadata. Uploaded as one non-LFS JSON."""
    n_caphit = len(scan["rows"])
    n_captured = int(cap_stats.get("n_captured", 0)) + int(cap_stats.get("n_verified_on_hub", 0))
    manifest = {
        "slug": slug,
        "split": split,
        "sampling_mode": GC.SAMPLING_MODE_GREEDY,
        "base_gen_max_tokens": int(GC.GEN_MAX_TOKENS),
        "regen_gen_max_tokens": int(cap),
        "regen_max_model_len": int(regen_max_model_len),
        "n_base_rows": int(scan["n_rows"]),
        "n_base_chunks": int(scan["n_chunks"]),
        "n_base_caphit": n_caphit,
        "base_caphit_rate": n_caphit / max(int(scan["n_rows"]), 1),
        "n_regen_chunks": int(gen_stats.get("n_chunks", 0)),
        "n_regen_rows": int(cap_stats.get("n_regen_rows", 0)),
        "n_regen_captured": n_captured,
        "n_regen_dropped_empty": len(cap_stats.get("dropped_empty_cis", [])),
        "n_residual_caphit_at_regen_cap": int(cap_stats.get("n_residual_caphit", 0)),
        "residual_caphit_rate": (
            int(cap_stats.get("n_residual_caphit", 0)) / max(int(scan["n_rows"]), 1)
        ),
        "n_greedy_prefix_extends": int(cap_stats.get("n_prefix_extends", 0)),
        "rows": cap_stats.get("row_records", []),
        "metadata": {
            **as_metadata_dict(git_provenance()),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch_version": torch.__version__,
            "written_by": "issue1491_caphit_regen",
        },
    }
    C.write_json_atomic(scratch / REGEN_MANIFEST_NAME, manifest)
    if not no_upload:
        prefix = regen_split_prefix(hf_root, slug, split, cap)
        hub._upload(
            scratch / REGEN_MANIFEST_NAME,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/{REGEN_MANIFEST_NAME}",
            upload_as_file=True,
            raise_on_error=True,
        )
    logger.info(
        "[regen-manifest] %s %s: caphit=%d captured=%d residual@%d=%d prefix_extends=%d",
        slug,
        split,
        n_caphit,
        n_captured,
        cap,
        manifest["n_residual_caphit_at_regen_cap"],
        manifest["n_greedy_prefix_extends"],
    )
    return manifest


# ---------------------------------------------------------------------------
# Merged-view loaders (consumed by issue1491_caphit_restriction_analysis.py)
# ---------------------------------------------------------------------------


def _regen_chunk_names(
    hf_root: str, slug: str, split: str, cap: int, suffix: str, local_dir: Path | None
) -> list[str]:
    if local_dir is not None:
        return sorted(p.name for p in Path(local_dir).glob(f"regen_chunk*{suffix}"))
    sub = "raw_completions" if suffix == ".json" else "final_token_capture"
    names = GC._remote_index(regen_split_prefix(hf_root, slug, split, cap), sub)
    return sorted(n for n in names if n.endswith(suffix))


def load_regen_raw_overlay(
    hf_root: str,
    slug: str,
    split: str,
    cap: int,
    scratch: Path,
    *,
    local_dir: Path | None = None,
) -> dict[int, dict]:
    """{ci: {prompt, response, finish_reason, regen_chunk}} for one (slug,
    split)'s regen namespace, identity-asserted per chunk. Empty dict when the
    namespace has no chunks (a split with zero cap-hit rows)."""
    payload_split = SPLIT_PREFIX_TO_PAYLOAD_SPLIT[split]
    seed = SPLIT_PREFIX_SEEDS[split]
    names = _regen_chunk_names(hf_root, slug, split, cap, ".json", local_dir)
    out: dict[int, dict] = {}
    scratch.mkdir(parents=True, exist_ok=True)
    prefix = regen_split_prefix(hf_root, slug, split, cap)
    for name in names:
        m = re.match(r"^regen_chunk(\d{4})\.json$", name)
        if not m:
            raise RuntimeError(f"unexpected regen raw chunk name {name!r} under {prefix}")
        if local_dir is not None:
            local, unlink_after = Path(local_dir) / name, False
        else:
            local = Path(
                F._download_chunk_with_retry(
                    C.HF_DATA_REPO, f"{prefix}/raw_completions/{name}", scratch
                )
            )
            unlink_after = True
        with open(local, encoding="utf-8") as fh:
            payload = json.load(fh)
        if unlink_after:
            local.unlink()
        GC._assert_raw_payload_matches(
            payload,
            name,
            expect_split=payload_split,
            expect_seed=seed,
            expect_shard_index=0,
            expect_chunk=int(m.group(1)),
            expect_sampling_mode=GC.SAMPLING_MODE_GREEDY,
            expect_gen_max_tokens=cap,
        )
        for r in payload["rows"]:
            ci = int(r["ci"])
            assert ci not in out, ("duplicate ci across regen raw chunks", slug, split, ci)
            out[ci] = {
                "prompt": str(r["prompt"]),
                "response": str(r["response"]),
                "finish_reason": str(r["finish_reason"]),
                "regen_chunk": name,
            }
    return out


def load_regen_capture_overlay(
    hf_root: str,
    slug: str,
    split: str,
    cap: int,
    layer: int,
    scratch: Path,
    raw_overlay: dict[int, dict],
    *,
    local_dir: Path | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{ci: (cx_last (H,), v_x (H,))} at ``layer`` from the regen .pt chunks,
    each bundle binding-VERIFIED against ``raw_overlay`` (rows_sha256) — the
    guard that makes a text/activation mismatch unloadable."""
    names = _regen_chunk_names(hf_root, slug, split, cap, ".pt", local_dir)
    overlay: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scratch.mkdir(parents=True, exist_ok=True)
    prefix = regen_split_prefix(hf_root, slug, split, cap)
    for name in names:
        if local_dir is not None:
            local, unlink_after = Path(local_dir) / name, False
        else:
            local = Path(
                F._download_chunk_with_retry(
                    C.HF_DATA_REPO, f"{prefix}/final_token_capture/{name}", scratch
                )
            )
            unlink_after = True
        b = FFC._mmap_load(local)
        _verify_pt_binding(b, raw_overlay, f"{slug}/{split}/{name}", cap)
        cx = N50F._slice_layer(b, "cx_last", layer)  # (n, H) fp32
        vx = N50F._slice_layer(b, "v_x", layer)
        for i, ci in enumerate(int(c) for c in b["ci"]):
            assert ci not in overlay, ("duplicate ci across regen .pt chunks", slug, split, ci)
            overlay[ci] = (cx[i], vx[i])
        del b
        if unlink_after:
            local.unlink()
    return overlay


def load_regen_manifest(
    hf_root: str,
    slug: str,
    split: str,
    cap: int,
    scratch: Path,
    *,
    local_dir: Path | None = None,
) -> dict:
    """The per-(slug, split) regen manifest (fail-loud when absent: the regen
    pass writes one for EVERY split it processed, zero-cap-hit splits
    included, so a missing manifest means the pass has not run)."""
    if local_dir is not None:
        p = Path(local_dir) / REGEN_MANIFEST_NAME
        if not p.exists():
            raise FileNotFoundError(f"regen manifest missing locally: {p}")
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    prefix = regen_split_prefix(hf_root, slug, split, cap)
    scratch.mkdir(parents=True, exist_ok=True)
    got = Path(
        F._download_chunk_with_retry(C.HF_DATA_REPO, f"{prefix}/{REGEN_MANIFEST_NAME}", scratch)
    )
    with open(got, encoding="utf-8") as fh:
        manifest = json.load(fh)
    got.unlink()
    if int(manifest.get("regen_gen_max_tokens", -1)) != int(cap):
        raise RuntimeError(
            f"regen manifest cap mismatch under {prefix}: "
            f"{manifest.get('regen_gen_max_tokens')} != {cap}"
        )
    return manifest


def stream_split_merged(
    hf_root: str,
    slug: str,
    split: str,
    cap: int,
    layer: int,
    scratch: Path,
    *,
    base_local_dir: Path | None = None,
    regen_local_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    """The POST-REGEN merged corpus view for one (slug, split, layer).

    Streams the base capture, overlays the regen rows by ci (fail-loud when a
    regen ci is absent from the base corpus), and returns
    ``(cx (n,H), vx (n,H), ci, gen_cap (n,) int32)`` — ``gen_cap`` is the
    per-row provenance: GEN_MAX_TOKENS for base rows, ``cap`` for overlaid
    regen rows. Base chunks are never modified (overlay-at-read merge)."""
    if base_local_dir is not None:
        d = Path(base_local_dir) / split / "final_token_capture"
        names = sorted(p.name for p in d.glob("*.pt"))
        if not names:
            raise FileNotFoundError(f"no base capture chunks under {d}")
        cx_parts, vx_parts, ci_parts = [], [], []
        for name in names:
            b = FFC._mmap_load(d / name)
            cx_parts.append(N50F._slice_layer(b, "cx_last", layer))
            vx_parts.append(N50F._slice_layer(b, "v_x", layer))
            ci_parts.extend(int(c) for c in b["ci"])
            del b
        cx = np.concatenate(cx_parts, axis=0)
        vx = np.concatenate(vx_parts, axis=0)
        ci = ci_parts
    else:
        cx, vx, ci = LF._stream_ladder_split(f"{hf_root}/{slug}", split, layer, scratch / ".cache")
    ci = [int(c) for c in ci]
    gen_cap = np.full(len(ci), int(GC.GEN_MAX_TOKENS), dtype=np.int32)
    raw_overlay = load_regen_raw_overlay(
        hf_root, slug, split, cap, scratch, local_dir=regen_local_dir
    )
    if raw_overlay:
        pt_overlay = load_regen_capture_overlay(
            hf_root, slug, split, cap, layer, scratch, raw_overlay, local_dir=regen_local_dir
        )
        pos = {c: i for i, c in enumerate(ci)}
        extra = sorted(set(pt_overlay) - set(pos))
        if extra:
            raise RuntimeError(
                f"regen merge: {len(extra)} overlay cis absent from the base corpus "
                f"({slug}/{split}, first: {extra[:8]}) — namespace/corpus mismatch"
            )
        for c, (cx_r, vx_r) in pt_overlay.items():
            i = pos[c]
            cx[i] = cx_r
            vx[i] = vx_r
            gen_cap[i] = int(cap)
        logger.info(
            "[regen-merge] %s %s layer=%d: %d/%d rows overlaid at cap=%d "
            "(%d regen raw rows, %d captured)",
            slug,
            split,
            layer,
            int((gen_cap == int(cap)).sum()),
            len(ci),
            cap,
            len(raw_overlay),
            len(pt_overlay),
        )
    else:
        logger.info("[regen-merge] %s %s: no regen overlay (0 cap-hit rows)", slug, split)
    return cx, vx, ci, gen_cap


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build_regen_engine(model_id: str, seed: int, max_model_len: int):
    """vLLM engine for the regen pass — max_model_len RAISED at this call site
    only (gotchas.md: max_model_len must track every max_new_tokens deviation;
    the shared default stays untouched), env hang-mitigation knobs honored
    exactly like the base driver."""
    from explore_persona_space.eval.generation import create_vllm_engine

    llm_kwargs: dict = {}
    if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
        llm_kwargs["enforce_eager"] = True
    if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
        llm_kwargs["enable_prefix_caching"] = False
    logger.info(
        "[regen-engine] max_model_len=%d engine_seed=%d knobs=%s", max_model_len, seed, llm_kwargs
    )
    return create_vllm_engine(
        model_id, max_model_len=int(max_model_len), seed=int(seed), **llm_kwargs
    )


def _reap_engine(llm) -> None:
    """Tear down the vLLM engine BEFORE the HF capture model loads (in-process
    framework swap — gotchas.md vLLM teardown recipe via the canonical
    _reap_vllm_engine helper)."""
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1.0)


def _parity_probe_rows(
    splits: list[str],
    scans: dict,
    args,
    cap: int,
    sampling: dict,
) -> tuple[list[str], list[str], list[int]]:
    """Up to 32 (prompt, response, ci) probe rows for the batched-capture
    parity gate, from the FIRST regen chunk of the first split with rows."""
    for split in splits:
        if not scans[split]["rows"]:
            continue
        scratch = _split_scratch(args.out_dir, args.slug, split)
        prefix = regen_split_prefix(args.hf_root, args.slug, split, cap)
        done_raw = set() if args.no_upload else GC._remote_index(prefix, "raw_completions")
        cis = regen_chunk_membership(list(scans[split]["rows"]))[0]
        raw_map = GC._load_persisted_gen_chunk(
            scratch,
            prefix,
            REGEN_RAW_NAME.format(k=0),
            scratch / ".cache",
            done_raw,
            expect_split=SPLIT_PREFIX_TO_PAYLOAD_SPLIT[split],
            expect_seed=SPLIT_PREFIX_SEEDS[split],
            expect_shard_index=0,
            expect_chunk=0,
            expect_sampling_mode=sampling["mode"],
            expect_gen_max_tokens=cap,
            allow_local_only=args.no_upload,
        )
        probe_cis = cis[:32]
        return (
            [str(raw_map[c]["prompt"]) for c in probe_cis],
            [str(raw_map[c]["response"]) for c in probe_cis],
            probe_cis,
        )
    return [], [], []


def run(args) -> int:
    if args.hf_root == PARENT_HF_ROOT:
        raise RuntimeError(
            f"--hf-root {PARENT_HF_ROOT!r} is the PARENT (temperature-1.0) corpus root — "
            "the regen pass is defined for the greedy round only and must never write a "
            "regen namespace beside the parent's promoted artifacts."
        )
    cap = int(args.regen_max_tokens)
    assert cap >= 2 * GC.GEN_MAX_TOKENS, (
        "regen cap must satisfy the pre-registered >=2x re-gen trigger",
        cap,
        GC.GEN_MAX_TOKENS,
    )
    regen_budget = int(args.regen_max_model_len) - cap - GC.LENGTH_MARGIN
    assert regen_budget >= GC.PROMPT_TOKEN_BUDGET, (
        "regen max_model_len too small: every base-admitted prompt (budget "
        f"{GC.PROMPT_TOKEN_BUDGET}) + {cap} new tokens must fit — raise "
        "--regen-max-model-len",
        regen_budget,
    )
    cfg = LF.LADDER_SCALES[args.slug]
    layers = list(cfg["layers"])
    h_dim = int(cfg["h_dim"])
    model_id = str(cfg["model"])
    sampling = GC._resolve_sampling(True, gen_max_tokens=cap)
    splits = list(args.splits)
    for s in splits:
        assert s in SPLIT_PREFIX_SEEDS, ("unknown split", s)
    logger.info(
        "[regen] slug=%s model=%s splits=%s cap=%d (base %d) max_model_len=%d "
        "budget=%d hf_root=%s device=%s",
        args.slug,
        model_id,
        splits,
        cap,
        GC.GEN_MAX_TOKENS,
        args.regen_max_model_len,
        regen_budget,
        args.hf_root,
        args.device,
    )

    # Phase 1: scan the base pass.
    C.phase("regen_scan")
    scans: dict[str, dict] = {}
    for split in splits:
        scratch = _split_scratch(args.out_dir, args.slug, split)
        scans[split] = scan_split_caphit(
            args.hf_root, args.slug, split, scratch, args.base_local_dir
        )
    scan_digest = {
        split: {
            "n_rows": scans[split]["n_rows"],
            "n_caphit": len(scans[split]["rows"]),
            "caphit_rate": len(scans[split]["rows"]) / max(scans[split]["n_rows"], 1),
        }
        for split in splits
    }
    digest_path = args.out_dir / f"scan_digest_{args.slug}.json"
    C.write_json_atomic(
        digest_path,
        {
            "slug": args.slug,
            "hf_root": args.hf_root,
            "base_gen_max_tokens": int(GC.GEN_MAX_TOKENS),
            "splits": scan_digest,
            "metadata": {
                **as_metadata_dict(git_provenance()),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        },
    )
    logger.info("[regen-scan] digest: %s -> %s", scan_digest, digest_path)
    if args.scan_only:
        C.phase("done")
        return 0

    # Phase 2: regenerate (engine held once across splits; per-request seeds
    # ride SamplingParams — inert under greedy, threaded for parity).
    C.phase("regen_gen")
    engine_seed = 42
    tok = GC._load_tokenizer(model_id)
    llm = None
    if args.device == "cuda":
        llm = _build_regen_engine(model_id, engine_seed, args.regen_max_model_len)
    gen_stats: dict[str, dict] = {}
    try:
        for split in splits:
            if not scans[split]["rows"]:
                logger.info("[regen-gen] %s %s: 0 cap-hit rows; skip", args.slug, split)
                gen_stats[split] = {
                    "n_chunks": 0,
                    "n_generated": 0,
                    "n_salvaged": 0,
                    "n_skipped_done": 0,
                }
                continue
            scratch = _split_scratch(args.out_dir, args.slug, split)
            gen_stats[split] = phase_gen_split(
                llm,
                tok,
                args.hf_root,
                args.slug,
                split,
                cap,
                scans[split],
                scratch,
                sampling,
                no_upload=args.no_upload,
                engine_seed=engine_seed,
                regen_budget=regen_budget,
            )
    finally:
        if llm is not None:
            _reap_engine(llm)
            llm = None

    # Phase 3: re-capture (HF model held once; parity gate once per slug).
    C.phase("regen_capture")
    tok_c, hf = N10.load_models(model_id, args.device)
    cap_stats: dict[str, dict] = {}
    try:
        capture_choice = "perrow"
        if args.capture_batch_size > 1:
            probe_prompts, probe_responses, probe_cis = _parity_probe_rows(
                splits, scans, args, cap, sampling
            )
            gate_pass, gate_reason = GC._batched_capture_parity_gate(
                hf,
                tok_c,
                probe_prompts,
                probe_responses,
                probe_cis,
                layers,
                h_dim,
                args.capture_batch_size,
            )
            logger.info(
                "[regen-capture] parity gate: %s (%s)",
                "PASS" if gate_pass else "FAIL",
                gate_reason,
            )
            capture_choice = "batched" if gate_pass else "perrow"
        for split in splits:
            if not scans[split]["rows"]:
                cap_stats[split] = {
                    "n_regen_rows": 0,
                    "n_captured": 0,
                    "n_verified_on_hub": 0,
                    "dropped_empty_cis": [],
                    "n_residual_caphit": 0,
                    "n_prefix_extends": 0,
                    "row_records": [],
                }
                continue
            scratch = _split_scratch(args.out_dir, args.slug, split)
            cap_stats[split] = phase_capture_split(
                hf,
                tok_c,
                args.hf_root,
                args.slug,
                split,
                cap,
                scans[split],
                scratch,
                sampling,
                no_upload=args.no_upload,
                layers=layers,
                h_dim=h_dim,
                capture_choice=capture_choice,
                batch_size=args.capture_batch_size,
            )
    finally:
        del hf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Phase 4: manifests (per-row cap provenance, durable).
    C.phase("regen_manifest")
    for split in splits:
        scratch = _split_scratch(args.out_dir, args.slug, split)
        phase_manifest_split(
            args.hf_root,
            args.slug,
            split,
            cap,
            scans[split],
            scratch,
            no_upload=args.no_upload,
            gen_stats=gen_stats[split],
            cap_stats=cap_stats[split],
            regen_max_model_len=int(args.regen_max_model_len),
        )
    C.phase("done")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True, choices=sorted(LF.LADDER_SCALES.keys()))
    ap.add_argument(
        "--splits",
        nargs="*",
        default=ALL_SPLITS,
        help="HF-prefix-space splits to process (default: all seven)",
    )
    ap.add_argument(
        "--hf-root",
        default=DEFAULT_HF_ROOT,
        help="GREEDY base-pass root (the parent temperature-1.0 root is refused)",
    )
    ap.add_argument(
        "--regen-max-tokens",
        type=int,
        default=DEFAULT_REGEN_MAX_TOKENS,
        help="regen generation cap (must be >= 2x the base cap — the pre-registered trigger)",
    )
    ap.add_argument(
        "--regen-max-model-len",
        type=int,
        default=DEFAULT_REGEN_MAX_MODEL_LEN,
        help="regen engine max_model_len (must fit base prompt budget + regen cap + margin)",
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--capture-batch-size", type=int, default=8)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "EPM_REGEN_OUT_DIR", os.path.expanduser("~/data/issue_1491_greedy/caphit_regen")
            )
        ),
        help="local scratch root (greedy-specific; NEVER the parent's out-dir)",
    )
    ap.add_argument(
        "--base-local-dir",
        type=Path,
        default=None,
        help="read the BASE pass from this local dir instead of HF (smoke path)",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="local-only (smoke path); resume keys on local files",
    )
    ap.add_argument(
        "--scan-only",
        action="store_true",
        help="stop after the cap-hit scan (writes the counts digest; no model loads)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
